"""Fusion #3 -- Residual Add + RMSNorm.  Fused vs unfused, tuned independently.

    unfused : add kernel   (read x, read res, write h1)
              norm kernel  (read h1, write x2)                -> 3 reads + 2 writes
    fused   : one kernel   (read x, read res, write h1 + x2)  -> 2 reads + 2 writes

Bandwidth ceiling = 5/4 = 1.25x.  Both sides materialize BOTH live outputs (h1 is the
next block's residual, x2 feeds the router / MoE), so no work is skipped.

Run:
    GLM52_RESULTS_DIR=results/rtx4060 python3 glm52/bench/bench_f03_resadd_rmsnorm.py
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from glm52 import config as C
from glm52 import reference as ref
from glm52.common import (
    autotune,
    bench_chain,
    check,
    main_guard,
    record,
    speedup_row,
)
from glm52.kernels import add_rmsnorm as K

RESULT_ID = "f03_resadd_rmsnorm"
H = C.HIDDEN_SIZE  # 6144
EPS = C.RMS_NORM_EPS
DT = C.DTYPE

REGIMES = [r for r in C.DECODE_REGIMES if r.T in (1, 32, 256)] + [
    r for r in C.PREFILL_REGIMES if r.T in (2048, 8192)
]


# --------------------------------------------------------------------------------------
# mapping search space
# --------------------------------------------------------------------------------------
WARP = C.env().warp_size  # lanes per warp: 32 on sm89, 64 on C500
_SM = C.env().num_sm
# persistent-grid rungs in whole waves of the device, so the knob is actually evaluated at
# a balanced grid: [24,48,96,192,384] here, [104,208,...] on C500's 104 CUs.
CAPS = [_SM * m for m in (1, 2, 4, 8, 16)]


def _valid(cfg: dict) -> bool:
    b, r, w = cfg["BLOCK_N"], cfg["ROWS"], cfg["num_warps"]
    threads = w * WARP
    if b * r > 65536:  # elements per program, not bytes: 1024 threads x 64 elem/thread
        return False
    epr = b * r / threads  # elements per thread per tile
    if epr < 2 or epr > 64:
        return False
    # one-shot (whole row in registers, no tile loop) -> deep pipelining is pointless
    if b >= H and cfg["num_stages"] > 2:
        return False
    return True


def coarse_grid() -> list[dict]:
    """~120 configs spanning: tile width (incl. both the padded power-of-two one-shot
    BLOCK_N=8192 and the multi-pass BLOCK_N in {512..4096}), rows/program, warps, stages.
    Non-persistent grid (one program per row-block)."""
    out = []
    # the warp ladder tops out at 32 so a CTA still spans 32..1024 threads at warp 32;
    # keeping C500's max of 16 would cost the widest tiles 30-40% of their configs
    # (BLOCK_N=8192: 20 -> 12) because epr <= 64 binds first with half the lanes
    for b, r, w, s in itertools.product(
        (512, 1024, 2048, 4096, 8192), (1, 2, 4, 8), (1, 2, 4, 8, 16, 32), (1, 2)
    ):
        cfg = dict(
            ROWS=r, BLOCK_N=b, num_warps=w, num_stages=s, grid_cap=None, eps=EPS
        )
        if _valid(cfg):
            out.append(cfg)
    return out


def refine_grid(best: dict) -> list[dict]:
    """~30-60 neighbours of the coarse winner: +-1 step in rows / tile width / warps,
    all stage counts, the persistent-grid variants, and the cache-eviction-hint variant
    of every one of them."""
    blocks = [512, 1024, 2048, 4096, 8192]
    rows = [1, 2, 4, 8, 16]
    warps = [1, 2, 4, 8, 16, 32]  # must contain every warp count coarse_grid can win with
    bi, ri, wi = (
        blocks.index(best["BLOCK_N"]),
        rows.index(best["ROWS"]),
        warps.index(best["num_warps"]),
    )
    nb = [blocks[i] for i in (bi - 1, bi, bi + 1) if 0 <= i < len(blocks)]
    nr = [rows[i] for i in (ri - 1, ri, ri + 1) if 0 <= i < len(rows)]
    nw = [warps[i] for i in (wi - 1, wi, wi + 1) if 0 <= i < len(warps)]

    out, seen = [], set()
    # (a) neighbourhood in block/rows/warps at the winning stage count
    for b, r, w in itertools.product(nb, nr, nw):
        cfg = dict(
            ROWS=r,
            BLOCK_N=b,
            num_warps=w,
            num_stages=best["num_stages"],
            grid_cap=None,
            eps=EPS,
        )
        key = (b, r, w, cfg["num_stages"], None)
        if _valid(cfg) and key not in seen:
            seen.add(key)
            out.append(cfg)
    # (b) stage sweep at the winning shape
    for s in (1, 2, 3, 4):
        cfg = dict(best)
        cfg["num_stages"] = s
        cfg["grid_cap"] = None
        key = (cfg["BLOCK_N"], cfg["ROWS"], cfg["num_warps"], s, None)
        if _valid(cfg) and key not in seen:
            seen.add(key)
            out.append(cfg)
    # (c) persistent grid at the winning shape and its +-1 row neighbours
    for cap, r in itertools.product(CAPS, nr):
        cfg = dict(best)
        cfg["ROWS"] = r
        cfg["grid_cap"] = cap
        key = (cfg["BLOCK_N"], r, cfg["num_warps"], cfg["num_stages"], cap)
        if _valid(cfg) and key not in seen:
            seen.add(key)
            out.append(cfg)
    # (d) cache-eviction hints (evict_first on the streaming loads, evict_last on W) --
    # a mapping knob, applied to the same source, tuned per side like everything else.
    out += [dict(c, EVICT=1) for c in out]
    return out


def _topk_cfgs(tr, k: int) -> list[dict]:
    rows = [(ms, cfg) for cfg, ms, err in tr.table if ms is not None]
    rows.sort(key=lambda t: t[0])
    return [cfg for _, cfg in rows[:k]]


def _tune2(make_chain, tag: str, warmup: int, rep: int):
    """Two-stage tune (coarse -> refine). Returns (best_cfg, best_ms, coarse, refine)."""
    t0 = time.time()
    cg = coarse_grid()
    tc = autotune(make_chain, cg, warmup=warmup, rep=rep)
    tr = autotune(make_chain, refine_grid(tc.best_cfg), warmup=warmup, rep=rep)
    if tr.best_ms <= tc.best_ms:
        best_cfg, best_ms = tr.best_cfg, tr.best_ms
    else:
        best_cfg, best_ms = tc.best_cfg, tc.best_ms
    print(
        f"    [{tag}] coarse {tc.n_tried} cfgs ({tc.n_failed} fail) -> {tc.best_ms:.4f} ms"
        f" | refine {tr.n_tried} ({tr.n_failed} fail) -> {tr.best_ms:.4f} ms"
        f" | best {best_ms:.4f} ms {best_cfg}  [{time.time()-t0:.0f}s]",
        flush=True,
    )
    return best_cfg, best_ms, tc, tr


# --------------------------------------------------------------------------------------
def run_regime(regime, warmup_t, rep_t, warmup_f, rep_f) -> dict:
    T = regime.T
    torch.manual_seed(1234 + T)
    x = torch.randn(T, H, device="cuda", dtype=DT)
    res = torch.randn(T, H, device="cuda", dtype=DT)
    w = (torch.randn(H, device="cuda", dtype=torch.float32) * 0.1 + 1.0).to(DT)
    h1 = torch.empty_like(x)
    out = torch.empty_like(x)
    h1u = torch.empty_like(x)
    outu = torch.empty_like(x)

    ref_out, ref_h1 = ref.add_rmsnorm(x, res, w, EPS)

    print(f"  == {regime.name} (T={T}) ==", flush=True)

    # ---- FUSED: one kernel, both outputs -------------------------------------------
    f_cfg, f_tune_ms, f_c, f_r = _tune2(
        lambda cfg: [lambda: K.fused_add_rmsnorm(x, res, w, h1, out, cfg)],
        "fused",
        warmup_t,
        rep_t,
    )

    # ---- UNFUSED: each kernel tuned independently -----------------------------------
    a_cfg, a_ms, a_c, a_r = _tune2(
        lambda cfg: [lambda: K.add_only(x, res, h1u, cfg)], "add", warmup_t, rep_t
    )
    n_cfg, n_ms, n_c, n_r = _tune2(
        lambda cfg: [lambda: K.norm_only(h1u, w, outu, cfg)], "norm", warmup_t, rep_t
    )

    # ---- extra (in the unfused side's favour): joint re-tune over the top-4 x top-4
    # of the independent searches, timed as the real chain (one flush, none between).
    joint_cfgs = [
        {"add": ac, "norm": nc}
        for ac, nc in itertools.product(_topk_cfgs(a_c, 2) + _topk_cfgs(a_r, 2),
                                        _topk_cfgs(n_c, 2) + _topk_cfgs(n_r, 2))
    ]
    joint_cfgs.append({"add": a_cfg, "norm": n_cfg})
    seen, uniq = set(), []
    for jc in joint_cfgs:
        key = (tuple(sorted(jc["add"].items())), tuple(sorted(jc["norm"].items())))
        if key not in seen:
            seen.add(key)
            uniq.append(jc)
    tj = autotune(
        lambda jc: [
            lambda: K.add_only(x, res, h1u, jc["add"]),
            lambda: K.norm_only(h1u, w, outu, jc["norm"]),
        ],
        uniq,
        warmup=warmup_t,
        rep=rep_t,
    )
    u_cfg = tj.best_cfg
    print(
        f"    [chain] joint {tj.n_tried} pairs -> {tj.best_ms:.4f} ms"
        f" (independent-best sum would be {a_ms + n_ms:.4f} ms)",
        flush=True,
    )

    # ---- validate the winners --------------------------------------------------------
    h1.zero_(); out.zero_(); h1u.zero_(); outu.zero_()
    K.fused_add_rmsnorm(x, res, w, h1, out, f_cfg)
    K.add_only(x, res, h1u, u_cfg["add"])
    K.norm_only(h1u, w, outu, u_cfg["norm"])
    torch.cuda.synchronize()
    chk = {
        "fused_h1": check(h1, ref_h1, label="fused_h1"),
        "fused_x2": check(out, ref_out, label="fused_x2"),
        "unfused_h1": check(h1u, ref_h1, label="unfused_h1"),
        "unfused_x2": check(outu, ref_out, label="unfused_x2"),
        "fused_eq_unfused_bitwise": bool(
            torch.equal(h1, h1u) and torch.equal(out, outu)
        ),
    }
    assert all(chk[k]["ok"] for k in chk if k != "fused_eq_unfused_bitwise"), chk

    # ---- final timing ----------------------------------------------------------------
    t_fused = bench_chain(
        [lambda: K.fused_add_rmsnorm(x, res, w, h1, out, f_cfg)], warmup_f, rep_f
    )
    t_unfused = bench_chain(
        [
            lambda: K.add_only(x, res, h1u, u_cfg["add"]),
            lambda: K.norm_only(h1u, w, outu, u_cfg["norm"]),
        ],
        warmup_f,
        rep_f,
    )

    # ---- torch production lines -------------------------------------------------------
    def torch_eager():
        hh = x + res
        yy = ref.rmsnorm(hh, w, EPS)
        return hh, yy

    t_eager = bench_chain([torch_eager], warmup_f, rep_f)

    t_compile, compile_err, compile_chk = None, None, None
    try:
        fn = torch.compile(lambda a, b, ww: ref.add_rmsnorm(a, b, ww, EPS))
        o_c, h_c = fn(x, res, w)
        torch.cuda.synchronize()
        compile_chk = {
            "h1": check(h_c, ref_h1, label="compile_h1"),
            "x2": check(o_c, ref_out, label="compile_x2"),
        }
        t_compile = bench_chain([lambda: fn(x, res, w)], warmup_f, rep_f)
    except Exception as exc:  # noqa: BLE001
        compile_err = f"{type(exc).__name__}: {exc}"[:400]
        print(f"    torch.compile FAILED: {compile_err}", flush=True)

    # ---- bytes / bandwidth --------------------------------------------------------------
    row_bytes = H * 2
    b_fused = T * row_bytes * 4  # x, res in; h1, x2 out
    b_unfused = T * row_bytes * 5  # + h1 write/read round trip
    gbps = lambda b, ms: b / (ms * 1e-3) / 1e9
    # the 5/4 model counts DRAM traffic. The round trip the fusion removes is h1, and h1
    # only reaches DRAM if it does not fit L2 -- 32 MB here vs 8 MB on C500, so several
    # regimes that round-tripped there stay resident here and cannot reach the ceiling.
    h1_bytes = T * row_bytes
    l2_bytes = C.env().l2_bytes

    row = speedup_row(
        regime.name,
        t_fused,
        t_unfused,
        {
            "T": T,
            "fused_cfg": f_cfg,
            "unfused_cfg": u_cfg,
            "rel_err_fused_x2": chk["fused_x2"]["rel_err"],
            "rel_err_fused_h1": chk["fused_h1"]["rel_err"],
            "rel_err_unfused_x2": chk["unfused_x2"]["rel_err"],
            "bitwise_identical": chk["fused_eq_unfused_bitwise"],
            "bytes_fused": b_fused,
            "bytes_unfused": b_unfused,
            # _model: derived from the traffic model above, not measured -- where h1 fits
            # L2 the unfused side never moves those bytes and the number is fictional
            "gbps_fused_model": gbps(b_fused, t_fused.p50_ms),
            "gbps_unfused_model": gbps(b_unfused, t_unfused.p50_ms),
            "ideal_speedup": 1.25,
            "h1_bytes": h1_bytes,
            "l2_bytes": l2_bytes,
            "h1_fits_l2": h1_bytes <= l2_bytes,
            "fused_noflush_ms": t_fused.noflush_p50_ms,
            "unfused_noflush_ms": t_unfused.noflush_p50_ms,
            "torch_eager_ms": t_eager.p50_ms,
            "torch_compile_ms": t_compile.p50_ms if t_compile else None,
            "torch_compile_err": compile_err,
            "torch_compile_check": compile_chk,
            "add_only_ms": a_ms,
            "norm_only_ms": n_ms,
        },
    )
    print(
        f"    fused {t_fused.p50_ms:.4f} ms ({gbps(b_fused, t_fused.p50_ms):.0f} GB/s)"
        f" | unfused {t_unfused.p50_ms:.4f} ms ({gbps(b_unfused, t_unfused.p50_ms):.0f} GB/s)"
        f" | speedup {row['speedup']:.3f}x"
        f" | eager {t_eager.p50_ms:.4f}"
        + (f" | compile {t_compile.p50_ms:.4f}" if t_compile else " | compile n/a"),
        flush=True,
    )

    tune_tables = {
        "fused": {"coarse": f_c.as_dict(), "refine": f_r.as_dict()},
        "add_only": {"coarse": a_c.as_dict(), "refine": a_r.as_dict()},
        "norm_only": {"coarse": n_c.as_dict(), "refine": n_r.as_dict()},
        "unfused_joint_chain": tj.as_dict(),
    }
    return row, tune_tables, chk, {
        "fused": t_fused.as_dict(),
        "unfused": t_unfused.as_dict(),
        "torch_eager": t_eager.as_dict(),
        "torch_compile": t_compile.as_dict() if t_compile else None,
    }


def main() -> None:
    env = C.env()  # same cached probe the config grids above were built from
    print(f"device={env.device_name} warp={env.warp_size} CUs={env.num_sm}", flush=True)

    rows, tables, checks, timings = [], {}, {}, {}

    def snapshot(done: bool) -> None:
        payload = {
            "id": RESULT_ID,
            "complete": done,
            "fusion": "#3 Residual Add + RMSNorm (sglang fused_add_rmsnorm semantics)",
            "shape": {"hidden": H, "dtype": "bfloat16", "eps": EPS},
            "traffic_model": {
                "unfused_passes": 5,
                "fused_passes": 4,
                "ceiling_speedup": 1.25,
                "note": "both sides write h1 (next block's residual) AND x2 (norm out)",
                "ceiling_note": "1.25x is a DRAM-traffic ceiling; it is unattainable in "
                "any regime whose row has h1_fits_l2=true (32 MB L2 here vs 8 MB on C500)",
            },
            "fairness": {
                "one_kernel_source": "glm52/kernels/add_rmsnorm.py::add_rmsnorm_kernel",
                "flags": "DO_ADD / DO_NORM constexpr select fused vs the two split kernels",
                "tuning": f"two-stage (coarse {len(coarse_grid())} + a per-winner refine "
                "neighbourhood; exact counts in tune_tables[*].n_tried) per variant per "
                "regime; the unfused side additionally gets a joint chain re-tune over "
                "the top-4 x top-4 configs, which can only help it",
            },
            "env": env.__dict__,
            "rows": rows,
            "checks": checks,
            "timings": timings,
            "tune_tables": tables,
        }
        record(RESULT_ID, payload)

    for regime in REGIMES:
        # small regimes are launch-bound -> more reps; big ones are slow -> fewer
        if regime.T <= 256:
            wt, rt, wf, rf = 20, 50, 100, 400
        elif regime.T <= 2048:
            wt, rt, wf, rf = 10, 30, 50, 200
        else:
            wt, rt, wf, rf = 10, 25, 30, 120
        row, tab, chk, tim = run_regime(regime, wt, rt, wf, rf)
        rows.append(row)
        tables[regime.name] = tab
        checks[regime.name] = chk
        timings[regime.name] = tim
        snapshot(False)  # crash insurance: results file is valid after every regime

    snapshot(True)
    print(f"\nwrote results/{RESULT_ID}.json", flush=True)
    print(f"{'regime':<16}{'fused':>10}{'unfused':>10}{'speedup':>9}{'eager':>10}{'compile':>10}")
    for r in rows:
        tc = r.get("torch_compile_ms")
        print(
            f"{r['regime']:<16}{r['fused_ms']:>10.4f}{r['unfused_ms']:>10.4f}"
            f"{r['speedup']:>9.3f}{r['torch_eager_ms']:>10.4f}"
            f"{(f'{tc:.4f}' if tc else 'n/a'):>10}"
        )


if __name__ == "__main__":
    main_guard(main)
