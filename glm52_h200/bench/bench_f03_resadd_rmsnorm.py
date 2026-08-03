"""Fusion #3 -- Residual Add + RMSNorm.  Fused vs unfused, tuned independently.

    unfused : add kernel   (read x, read res, write h1)
              norm kernel  (read h1, write x2)                -> 3 reads + 2 writes
    fused   : one kernel   (read x, read res, write h1 + x2)  -> 2 reads + 2 writes

Bandwidth ceiling = 5/4 = 1.25x.  Both sides materialize BOTH live outputs (h1 is the
next block's residual, x2 feeds the router / MoE), so no work is skipped.

The ceiling is a DRAM-traffic ceiling.  The round trip the fusion removes is `h1`, and h1
only reaches DRAM if it does not fit L2.  H200's L2 is ~50 MB against the 4060's 32 MB and
C500's 8 MB, so the L2-resident regimes move again: `h1_fits_l2` is recorded per regime and
a 1.0x where it is true is the model working, not the kernel failing.

Run:
    python3 glm52_h200/bench/bench_f03_resadd_rmsnorm.py --gpu auto [--regimes ...] [--quick]

`--gpu auto` picks the idlest of the host's GPUs and masks the process to it before
CUDA initialises; on the 8-GPU measurement host that is the difference between timing an
idle card and timing one another tenant is already using.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from glm52_h200 import bench as B
from glm52_h200 import config as C
from glm52_h200 import reference as ref
from glm52_h200.common import (
    autotune,
    check,
    main_guard,
    record,
    speedup_row,
)
from glm52_h200.kernels import add_rmsnorm as K

RESULT_ID = "f03_resadd_rmsnorm"
H = C.HIDDEN_SIZE  # 6144
EPS = C.RMS_NORM_EPS
DT = C.DTYPE
UNITS = ["f3"]

# --------------------------------------------------------------------------------------
# mapping search space -- every bound below is a function of the probed device
# --------------------------------------------------------------------------------------
_ENV = C.env()
WARP = B.env_int(_ENV, "warp_size")
MAX_THREADS = B.max_threads_per_block(_ENV)
#: elements one program may hold = largest CTA x per-thread budget.  The C500 code wrote
#: this as the literal 65536, which is exactly 1024 threads x 64 elem/thread ON THAT DEVICE.
ELEM_CAP = B.elems_per_program_cap(_ENV)
#: persistent-grid rungs in whole waves, so the knob is evaluated at a balanced grid
CAPS = B.sm_wave_caps(_ENV)
WARPS = B.warp_ladder(_ENV)


def _valid(cfg: dict) -> bool:
    b, r, w = cfg["BLOCK_N"], cfg["ROWS"], cfg["num_warps"]
    threads = w * WARP
    if threads > MAX_THREADS:
        return False
    if b * r > ELEM_CAP:
        return False
    epr = b * r / threads  # elements per thread per tile
    if epr < 2 or epr > B.MAX_ELEMS_PER_THREAD:
        return False
    # one-shot (whole row in registers, no tile loop) -> deep pipelining is pointless
    if b >= H and cfg["num_stages"] > 2:
        return False
    return True


BLOCKS = [512, 1024, 2048, 4096, 8192]
ROWSET = [1, 2, 4, 8, 16]


def coarse_grid() -> list[dict]:
    """Tile width (incl. both the padded power-of-two one-shot BLOCK_N=8192 and the
    multi-pass BLOCK_N in {512..4096}), rows/program, warps, stages.  Non-persistent grid
    (one program per row-block).  The count is device-dependent and is recorded live in
    `fairness.grids` rather than quoted here."""
    out = []
    for b, r, w, s in itertools.product(BLOCKS, (1, 2, 4, 8), WARPS, (1, 2)):
        cfg = dict(ROWS=r, BLOCK_N=b, num_warps=w, num_stages=s, grid_cap=None, eps=EPS)
        if _valid(cfg):
            out.append(cfg)
    return B.widen(out, K)


def refine_grid(best: dict) -> list[dict]:
    """Neighbours of the coarse winner: +-1 step in rows / tile width / warps, all stage
    counts, the persistent-grid variants, and the cache-eviction-hint variant of every one
    of them.  Identical rule for every variant, centred on that variant's own winner."""
    bi = BLOCKS.index(best["BLOCK_N"]) if best["BLOCK_N"] in BLOCKS else 0
    ri = ROWSET.index(best["ROWS"]) if best["ROWS"] in ROWSET else 0
    wi = WARPS.index(best["num_warps"]) if best["num_warps"] in WARPS else 0
    nb = [BLOCKS[i] for i in (bi - 1, bi, bi + 1) if 0 <= i < len(BLOCKS)]
    nr = [ROWSET[i] for i in (ri - 1, ri, ri + 1) if 0 <= i < len(ROWSET)]
    nw = [WARPS[i] for i in (wi - 1, wi, wi + 1) if 0 <= i < len(WARPS)]

    out, seen = [], set()

    def add(cfg):
        if not _valid(cfg):
            return
        key = (cfg["BLOCK_N"], cfg["ROWS"], cfg["num_warps"], cfg["num_stages"],
               cfg.get("grid_cap"), cfg.get("EVICT"))
        if key in seen:
            return
        seen.add(key)
        out.append(cfg)

    # (a) neighbourhood in block/rows/warps at the winning stage count
    for b, r, w in itertools.product(nb, nr, nw):
        add(dict(best, ROWS=r, BLOCK_N=b, num_warps=w, grid_cap=None))
    # (b) stage sweep at the winning shape
    for s in (1, 2, 3, 4):
        add(dict(best, num_stages=s, grid_cap=None))
    # (c) persistent grid at the winning shape and its +-1 row neighbours
    for cap, r in itertools.product(CAPS, nr):
        add(dict(best, ROWS=r, grid_cap=cap))
    # (d) cache-eviction hints (evict_first on the streaming loads, evict_last on W) --
    # a mapping knob, applied to the same source, tuned per side like everything else.
    out += [dict(c, EVICT=1) for c in list(out)]
    return B.dedup(out)


def _tune2(make_chain, tag: str, verify, warmup: int, rep: int, quick: bool, fair, regime):
    """Two-stage tune (coarse -> refine), numerically screened.  Returns
    (best_cfg, best_ms, coarse, refine)."""
    t0 = time.time()
    cg = coarse_grid()
    if quick:
        cg = B.quick_slice(cg, 20)
    tc = B.screened_autotune(f"{tag}/coarse", make_chain, cg, verify, warmup, rep)
    rg = refine_grid(tc.best_cfg)
    if quick:
        rg = B.quick_slice(rg, 12)
    tr = B.screened_autotune(f"{tag}/refine", make_chain, rg, verify, warmup, rep)
    if tr.best_ms <= tc.best_ms:
        best_cfg, best_ms = tr.best_cfg, tr.best_ms
    else:
        best_cfg, best_ms = tc.best_cfg, tc.best_ms
    fair.add(regime, tag, "coarse", tc, grid=cg)
    fair.add(regime, tag, "refine", tr, grid=rg)
    print(
        f"    [{tag}] coarse {tc.n_tried} cfgs ({tc.n_failed} rej) -> {tc.best_ms:.4f} ms"
        f" | refine {tr.n_tried} ({tr.n_failed} rej) -> {tr.best_ms:.4f} ms"
        f" | best {best_ms:.4f} ms {best_cfg}  [{time.time() - t0:.0f}s]",
        flush=True,
    )
    return best_cfg, best_ms, tc, tr


# --------------------------------------------------------------------------------------
def run_regime(regime, quick: bool, fair: B.Fairness) -> tuple:
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
    warmup_t, rep_t, warmup_f, rep_f = B.reps(T, quick)

    print(f"  == {regime.name} (T={T}) ==", flush=True)

    # ---- verifiers: every config is checked against the fp32 reference before timing ----
    def v_fused():
        a = check(out, ref_out, label="x2")
        b = check(h1, ref_h1, label="h1")
        return a["ok"] and b["ok"], f"x2={a['rel_err']:.2e} h1={b['rel_err']:.2e}"

    def v_add():
        a = check(h1u, ref_h1, label="h1")
        return a["ok"], f"h1={a['rel_err']:.2e}"

    def v_norm():
        # norm_only reads h1u, which the add screen has just filled with the reference
        a = check(outu, ref_out, label="x2")
        return a["ok"], f"x2={a['rel_err']:.2e}"

    # ---- FUSED: one kernel, both outputs -------------------------------------------
    f_cfg, f_tune_ms, f_c, f_r = _tune2(
        lambda cfg: [lambda: K.fused_add_rmsnorm(x, res, w, h1, out, cfg)],
        "fused", v_fused, warmup_t, rep_t, quick, fair, regime.name,
    )

    # ---- UNFUSED: each kernel tuned independently -----------------------------------
    a_cfg, a_ms, a_c, a_r = _tune2(
        lambda cfg: [lambda: K.add_only(x, res, h1u, cfg)],
        "add", v_add, warmup_t, rep_t, quick, fair, regime.name,
    )
    h1u.copy_(ref_h1)  # a valid input for the norm screen, independent of the add's winner
    n_cfg, n_ms, n_c, n_r = _tune2(
        lambda cfg: [lambda: K.norm_only(h1u, w, outu, cfg)],
        "norm", v_norm, warmup_t, rep_t, quick, fair, regime.name,
    )

    # ---- extra (in the unfused side's favour): joint re-tune over the top-4 x top-4
    # of the independent searches, timed as the real chain (one flush, none between).
    joint_cfgs = [
        {"add": ac, "norm": nc}
        for ac, nc in itertools.product(
            B.top_cfgs(a_c, k=2) + B.top_cfgs(a_r, k=2),
            B.top_cfgs(n_c, k=2) + B.top_cfgs(n_r, k=2),
        )
    ]
    joint_cfgs.append({"add": a_cfg, "norm": n_cfg})
    seen, uniq = set(), []
    for jc in joint_cfgs:
        key = (tuple(sorted((k, str(v)) for k, v in jc["add"].items())),
               tuple(sorted((k, str(v)) for k, v in jc["norm"].items())))
        if key not in seen:
            seen.add(key)
            uniq.append(jc)
    tj = autotune(
        lambda jc: [
            lambda: K.add_only(x, res, h1u, jc["add"]),
            lambda: K.norm_only(h1u, w, outu, jc["norm"]),
        ],
        uniq, warmup_t, rep_t,
    )
    u_cfg = tj.best_cfg
    fair.add(regime.name, "unfused_chain", "joint", tj)
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

    # ---- final timing: INTERLEAVED, PAIRED --------------------------------------------
    t_fused, t_unfused, pair = B.bench_pair(
        [lambda: K.fused_add_rmsnorm(x, res, w, h1, out, f_cfg)],
        [
            lambda: K.add_only(x, res, h1u, u_cfg["add"]),
            lambda: K.norm_only(h1u, w, outu, u_cfg["norm"]),
        ],
        warmup_f, rep_f, label=regime.name,
    )

    # ---- torch production lines -------------------------------------------------------
    def torch_eager():
        hh = x + res
        yy = ref.rmsnorm(hh, w, EPS)
        return hh, yy

    t_eager, t_compile, compile_err, compile_chk = None, None, None, None
    try:
        from glm52_h200.common import bench_chain

        t_eager = bench_chain([torch_eager], warmup_f, rep_f)
    except Exception as exc:  # noqa: BLE001 -- a reference line must not lose the regime
        print(f"    torch eager FAILED: {type(exc).__name__}: {exc}", flush=True)
    try:
        from glm52_h200.common import bench_chain

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

    # ---- bytes / bandwidth ------------------------------------------------------------
    row_bytes = H * 2
    b_fused = T * row_bytes * 4  # x, res in; h1, x2 out
    b_unfused = T * row_bytes * 5  # + h1 write/read round trip
    gbps = lambda b, ms: b / (ms * 1e-3) / 1e9  # noqa: E731
    h1_bytes = T * row_bytes
    l2_bytes = B.env_int(_ENV, "l2_bytes")

    row = speedup_row(
        regime.name, t_fused, t_unfused,
        {
            "T": T,
            "variant": "f3",
            "fused_cfg": f_cfg,
            "unfused_cfg": u_cfg,
            "paired_speedup": pair.get("paired_speedup_p50"),
            "paired_speedup_trimmed": pair.get("paired_speedup_trimmed_mean"),
            "pair_meta": pair,
            "tick": B.tick_report(t_fused.p50_ms, t_unfused.p50_ms),
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
            "torch_eager_ms": t_eager.p50_ms if t_eager else None,
            "torch_compile_ms": t_compile.p50_ms if t_compile else None,
            "torch_compile_err": compile_err,
            "torch_compile_check": compile_chk,
            "add_only_ms": a_ms,
            "norm_only_ms": n_ms,
        },
    )
    print(
        f"    fused {t_fused.p50_ms:.4f} ms ({gbps(b_fused, t_fused.p50_ms):.0f} GB/s)"
        f" | unfused {t_unfused.p50_ms:.4f} ms "
        f"({gbps(b_unfused, t_unfused.p50_ms):.0f} GB/s)"
        f" | paired {row['paired_speedup']:.3f}x"
        + (f" | eager {t_eager.p50_ms:.4f}" if t_eager else "")
        + (f" | compile {t_compile.p50_ms:.4f}" if t_compile else " | compile n/a")
        + ("  [TICK-LIMITED]" if row["tick"].get("tick_limited") else ""),
        flush=True,
    )

    tune_tables = {
        "fused": {"coarse": f_c.as_dict(), "refine": f_r.as_dict()},
        "add_only": {"coarse": a_c.as_dict(), "refine": a_r.as_dict()},
        "norm_only": {"coarse": n_c.as_dict(), "refine": n_r.as_dict()},
        "unfused_joint_chain": tj.as_dict(),
    }
    timings = {
        "fused": t_fused.as_dict(),
        "unfused": t_unfused.as_dict(),
        "pair": pair,
        "torch_eager": t_eager.as_dict() if t_eager else None,
        "torch_compile": t_compile.as_dict() if t_compile else None,
    }
    del x, res, w, h1, out, h1u, outu, ref_out, ref_h1
    torch.cuda.empty_cache()
    return row, tune_tables, chk, timings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    B.add_std_args(ap, UNITS)
    args = ap.parse_args()
    if args.list:
        print("regimes:", ", ".join(B.REGIME_NAMES))
        print("variants:", ", ".join(UNITS))
        return

    env = C.env()
    B.banner(env)
    B.exact_fp32_matmul()
    B.check_preflight_device(env)
    B.resolve_units(UNITS, args.only)
    regimes = B.resolve_regimes(C, args.regimes)

    fair = B.Fairness(
        one_kernel_source="glm52_h200/kernels/add_rmsnorm.py::add_rmsnorm_kernel",
        flags="DO_ADD / DO_NORM constexpr select fused vs the two split kernels",
        protocol=(
            "two-stage (coarse + a per-winner refine neighbourhood) per variant per "
            "regime, every config numerically screened against the fp32 reference before "
            "it is timed; the unfused side additionally gets a joint chain re-tune over "
            "the top-4 x top-4 configs, which can only help it. Live per-arm counts are in "
            "grids below -- the coarse grid size is a function of the probed warp width and "
            "CTA ceiling, so it is not a constant across devices."
        ),
        h200_axes=(
            "NONE of the sm_90 mapping axes apply to this family, on EITHER arm, and that "
            "is a property of the kernel rather than of the search: add_rmsnorm_kernel is a "
            "row-wise vector pass with no GEMM mainloop -- nothing for warp specialization "
            "to split into producers and consumers, no k-loop for a descriptor to feed, and "
            "no tile whose SMEM budget a cluster would enlarge. The fused and unfused arms "
            "are the same kernel with DO_ADD/DO_NORM flipped, so both are offered exactly "
            "the same (empty) set of axes; their axis_counts are legitimately zero on both "
            "sides. See fairness.h200_axes.per_family for what the module advertises."
        ),
    )
    fair.axis("f03_resadd_rmsnorm", B.h200_axis_report(K))

    rows, tables, checks, timings, pair_meta = [], {}, {}, {}, None

    def snapshot(done: bool) -> None:
        record(RESULT_ID, {
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
                "any regime whose row has h1_fits_l2=true -- H200's L2 is ~50 MB, so more "
                "regimes are resident here than on the 32 MB 4060 or the 8 MB C500",
            },
            "fairness": fair.render(env, pair_meta),
            "env": env.__dict__,
            "rows": rows,
            "checks": checks,
            "timings": timings,
            "tune_tables": tables,
        })

    for regime in regimes:
        ck = B.ckpt_load(RESULT_ID, regime.name, env, force=args.force)
        if ck is not None:
            print(f"  == {regime.name} == (from checkpoint)", flush=True)
            row, tab, chk, tim = ck["row"], ck["tables"], ck["checks"], ck["timings"]
            fair.grids.update(ck.get("fairness_grids", {}))
        else:
            row, tab, chk, tim = run_regime(regime, args.quick, fair)
            pair_meta = tim.get("pair")
            B.ckpt_save(RESULT_ID, regime.name, env, {
                "row": row, "tables": tab, "checks": chk, "timings": tim,
                "fairness_grids": {regime.name: fair.grids.get(regime.name, {})},
            })
        rows.append(row)
        tables[regime.name] = tab
        checks[regime.name] = chk
        timings[regime.name] = tim
        snapshot(False)  # crash insurance: the results file is valid after every regime

    snapshot(True)
    print(f"\nwrote {RESULT_ID}.json", flush=True)
    print(f"{'regime':<16}{'fused':>10}{'unfused':>10}{'paired':>9}{'eager':>10}"
          f"{'compile':>10}{'L2res':>7}")
    for r in rows:
        tc, te = r.get("torch_compile_ms"), r.get("torch_eager_ms")
        print(
            f"{r['regime']:<16}{r['fused_ms']:>10.4f}{r['unfused_ms']:>10.4f}"
            f"{(r.get('paired_speedup') or r['speedup']):>9.3f}"
            f"{(f'{te:.4f}' if te else 'n/a'):>10}"
            f"{(f'{tc:.4f}' if tc else 'n/a'):>10}"
            f"{('yes' if r['h1_fits_l2'] else 'no'):>7}"
        )


if __name__ == "__main__":
    main_guard(main)
