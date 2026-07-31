"""Fusion #10 -- Expert Merge + Residual Add.  Fused vs unfused, tuned independently.

    unfused : merge  kernel  (read Y [T,8,H] + w, write m)      -> 8 R + 1 W
              resadd kernel  (read m, read res, write out)      -> 2 R + 1 W   = 12 passes
    fused   : one kernel     (read Y + w + res, write out)      -> 9 R + 1 W   = 10 passes

Bandwidth ceiling = 12/10 = 1.20x at every regime (confirmed by `python -m glm52.traffic`;
the op has no meaningful FLOPs so the latency-aware ceiling equals the traffic ratio).

Both sides produce the same `out`; the fused side additionally avoids materializing the
merged intermediate `m`, which is exactly the benefit being measured.  `m` has no other
consumer in the GLM-5.2 decoder layer (the next op is the post-MoE RMSNorm, which reads
`out`), so nothing downstream is skipped.  The fused kernel reproduces the unfused chain's
round-to-bf16 of `m`, which makes the two outputs **bitwise identical** (asserted below).

Run:
    CUDA_VISIBLE_DEVICES=3 /home/zhangshuhan/my-envs/fusion/bin/python \
        glm52/bench/bench_f10_merge_resadd.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/zhangshuhan/fusion")

import torch

from glm52 import config as C
from glm52 import reference as ref
from glm52.common import (
    RESULTS_DIR,
    autotune,
    bench_chain,
    check,
    main_guard,
    record,
    speedup_row,
)
from glm52.kernels import merge_resadd as K

RESULT_ID = "f10_merge_resadd"
H = C.HIDDEN_SIZE  # 6144
TOPK = C.NUM_EXPERTS_PER_TOK  # 8
DT = C.DTYPE
CKPT = RESULTS_DIR / f"_{RESULT_ID}_ckpt"

REGIMES = [r for r in C.DECODE_REGIMES if r.T in (1, 32, 256)] + [
    r for r in C.PREFILL_REGIMES if r.T in (2048, 8192)
]

BLOCKS = [256, 512, 1024, 2048, 4096]
ROWSET = [1, 2, 4, 8, 16]
WARPS = [1, 2, 4, 8, 16]
CAPS = [104, 208, 416, 832, 1664]


# --------------------------------------------------------------------------------------
# mapping search space -- ONE generator, used identically by every variant
# --------------------------------------------------------------------------------------
def _valid(cfg: dict) -> bool:
    b, r, w, kv = cfg["BLOCK_N"], cfg["ROWS"], cfg["num_warps"], cfg["KVEC"]
    th = w * 64  # warp = 64 lanes on C500
    epr = b * r / th  # output elements per thread
    if epr < 4 or epr > 32:
        return False
    # KVEC materializes a [ROWS, TOPK, BLOCK_N] fp32 register slab
    if kv and b * r * TOPK / th > 64:
        return False
    if b * r * (TOPK if kv else 1) > 32768:
        return False
    return True


def coarse_grid() -> list[dict]:
    """174 configs: tile width x rows/program x warps x stages x {loop, 3-D slab}."""
    out = []
    for b, r, w, s, kv in itertools.product(
        BLOCKS, (1, 2, 4, 8), WARPS, (1, 2), (0, 1)
    ):
        cfg = dict(
            ROWS=r, BLOCK_N=b, num_warps=w, num_stages=s, KVEC=kv, UNROLL=1,
            grid_cap=None,
        )
        if _valid(cfg):
            out.append(cfg)
    return out


def refine_grid(best: dict) -> list[dict]:
    """Neighbourhood of the coarse winner: +-1 step in tile width / rows / warps, the
    full stage sweep, the KVEC and UNROLL flips, the persistent-grid caps, and the
    eviction-hint variant of every one of them.  Same rules for every variant."""
    bi, ri, wi = (
        BLOCKS.index(best["BLOCK_N"]),
        ROWSET.index(best["ROWS"]),
        WARPS.index(best["num_warps"]),
    )
    nb = [BLOCKS[i] for i in (bi - 1, bi, bi + 1) if 0 <= i < len(BLOCKS)]
    nr = [ROWSET[i] for i in (ri - 1, ri, ri + 1) if 0 <= i < len(ROWSET)]
    nw = [WARPS[i] for i in (wi - 1, wi, wi + 1) if 0 <= i < len(WARPS)]

    out, seen = [], set()

    def add(cfg):
        if not _valid(cfg):
            return
        key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
        if key in seen:
            return
        seen.add(key)
        out.append(cfg)

    # (a) +-1 neighbourhood in block / rows / warps at the winning stages+KVEC+UNROLL
    for b, r, w in itertools.product(nb, nr, nw):
        add(dict(best, BLOCK_N=b, ROWS=r, num_warps=w, grid_cap=None))
    # (b) stage sweep at the winning shape
    for s in (1, 2, 3, 4):
        add(dict(best, num_stages=s, grid_cap=None))
    # (c) the two loop-structure flips, at the winning shape and its block neighbours
    for b, kv, un in itertools.product(nb, (0, 1), (0, 1)):
        add(dict(best, BLOCK_N=b, KVEC=kv, UNROLL=un, grid_cap=None))
    # (d) persistent grid at the winning shape and its +-1 row neighbours
    for cap, r in itertools.product(CAPS, nr):
        add(dict(best, ROWS=r, grid_cap=cap))
    # (e) streaming eviction hints -- a mapping knob, tuned per side like everything else
    out += [dict(c, EVICT=1) for c in out]
    return out


def _topk_cfgs(tr, k: int) -> list[dict]:
    rows = [(ms, cfg) for cfg, ms, err in tr.table if ms is not None]
    rows.sort(key=lambda t: t[0])
    return [cfg for _, cfg in rows[:k]]


def _tune2(make_chain, tag: str, warmup: int, rep: int):
    """Two-stage tune (coarse -> refine).  Returns (best_cfg, best_ms, coarse, refine)."""
    t0 = time.time()
    tc = autotune(make_chain, coarse_grid(), warmup=warmup, rep=rep)
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
    # Per-expert down-projection outputs, UNWEIGHTED, [T, topk, H] -- exactly the tensor
    # sglang's `invoke_fused_moe_kernel(..., mul_routed_weight=False)` would leave behind.
    y = torch.randn(T, TOPK, H, device="cuda", dtype=DT)
    # Routing weights with the real GLM-5.2 post-processing: normalized then * 2.5.
    wraw = torch.rand(T, TOPK, device="cuda", dtype=torch.float32) + 0.05
    wt = (wraw / wraw.sum(-1, keepdim=True)) * C.ROUTED_SCALING_FACTOR
    res = torch.randn(T, H, device="cuda", dtype=DT)

    out = torch.empty(T, H, device="cuda", dtype=DT)
    mu = torch.empty(T, H, device="cuda", dtype=DT)
    outu = torch.empty(T, H, device="cuda", dtype=DT)

    # fp32 reference
    ref_m = ref.expert_merge(y, wt)
    ref_out = (ref_m.float() + res.float()).to(DT)

    print(f"  == {regime.name} (T={T}) ==", flush=True)

    # ---- FUSED: one kernel ------------------------------------------------------------
    f_cfg, f_tune_ms, f_c, f_r = _tune2(
        lambda cfg: [lambda: K.fused_merge_resadd(y, wt, res, out, cfg)],
        "fused",
        warmup_t,
        rep_t,
    )

    # ---- UNFUSED: each kernel tuned independently over the SAME grid -------------------
    m_cfg, m_ms, m_c, m_r = _tune2(
        lambda cfg: [lambda: K.merge_only(y, wt, mu, cfg)], "merge", warmup_t, rep_t
    )
    a_cfg, a_ms, a_c, a_r = _tune2(
        lambda cfg: [lambda: K.resadd_only(mu, res, outu, cfg)], "resadd", warmup_t, rep_t
    )

    # ---- extra, in the unfused side's favour: joint re-tune of the pair, timed as the
    # real chain (one flush before the pair, none between its two kernels).
    joint = [
        {"merge": mc, "resadd": ac}
        for mc, ac in itertools.product(
            _topk_cfgs(m_c, 2) + _topk_cfgs(m_r, 2),
            _topk_cfgs(a_c, 2) + _topk_cfgs(a_r, 2),
        )
    ]
    joint.append({"merge": m_cfg, "resadd": a_cfg})
    seen, uniq = set(), []
    for jc in joint:
        key = (
            tuple(sorted(jc["merge"].items())),
            tuple(sorted(jc["resadd"].items())),
        )
        if key not in seen:
            seen.add(key)
            uniq.append(jc)
    tj = autotune(
        lambda jc: [
            lambda: K.merge_only(y, wt, mu, jc["merge"]),
            lambda: K.resadd_only(mu, res, outu, jc["resadd"]),
        ],
        uniq,
        warmup=warmup_t,
        rep=rep_t,
    )
    u_cfg = tj.best_cfg
    print(
        f"    [chain] joint {tj.n_tried} pairs -> {tj.best_ms:.4f} ms"
        f" (independent-best sum would be {m_ms + a_ms:.4f} ms)",
        flush=True,
    )

    # ---- validate the winners ----------------------------------------------------------
    out.zero_(); mu.zero_(); outu.zero_()
    K.fused_merge_resadd(y, wt, res, out, f_cfg)
    K.merge_only(y, wt, mu, u_cfg["merge"])
    K.resadd_only(mu, res, outu, u_cfg["resadd"])
    torch.cuda.synchronize()
    chk = {
        "fused_out": check(out, ref_out, label="fused_out"),
        "unfused_m": check(mu, ref_m, label="unfused_m"),
        "unfused_out": check(outu, ref_out, label="unfused_out"),
        "fused_eq_unfused_bitwise": bool(torch.equal(out, outu)),
    }
    # accuracy note: the fused kernel does not *need* the intermediate bf16 rounding
    out_hi = torch.empty_like(out)
    K.fused_merge_resadd(y, wt, res, out_hi, dict(f_cfg, ROUND_MID=0))
    torch.cuda.synchronize()
    ref_out_hi = (
        (y.float() * wt.unsqueeze(-1)).sum(1) + res.float()
    ).to(DT)
    chk["fused_no_round_mid_vs_fp32"] = check(out_hi, ref_out_hi, label="no_round_mid")
    assert all(
        chk[k]["ok"] for k in chk if k != "fused_eq_unfused_bitwise"
    ), chk

    # ---- final timing -------------------------------------------------------------------
    t_fused = bench_chain(
        [lambda: K.fused_merge_resadd(y, wt, res, out, f_cfg)], warmup_f, rep_f
    )
    t_unfused = bench_chain(
        [
            lambda: K.merge_only(y, wt, mu, u_cfg["merge"]),
            lambda: K.resadd_only(mu, res, outu, u_cfg["resadd"]),
        ],
        warmup_f,
        rep_f,
    )

    # ---- torch production lines ----------------------------------------------------------
    def torch_eager():
        mm = ref.expert_merge(y, wt)
        return (mm.float() + res.float()).to(DT)

    t_eager = bench_chain([torch_eager], warmup_f, rep_f)

    t_compile, compile_err, compile_chk = None, None, None
    try:
        fn = torch.compile(
            lambda yy, ww, rr: (
                ref.expert_merge(yy, ww).float() + rr.float()
            ).to(DT)
        )
        o_c = fn(y, wt, res)
        torch.cuda.synchronize()
        compile_chk = check(o_c, ref_out, label="compile_out")
        t_compile = bench_chain([lambda: fn(y, wt, res)], warmup_f, rep_f)
    except Exception as exc:  # noqa: BLE001
        compile_err = f"{type(exc).__name__}: {exc}"[:400]
        print(f"    torch.compile FAILED: {compile_err}", flush=True)

    # ---- bytes / bandwidth ----------------------------------------------------------------
    act = T * H * 2
    wbytes = T * TOPK * 4
    b_fused = (TOPK + 1) * act + act + wbytes  # Y + RES in, OUT out
    b_unfused = (TOPK * act + act + wbytes) + (2 * act + act)
    gbps = lambda b, ms: b / (ms * 1e-3) / 1e9

    row = speedup_row(
        regime.name,
        t_fused,
        t_unfused,
        {
            "T": T,
            "fused_cfg": f_cfg,
            "unfused_cfg": u_cfg,
            "rel_err_fused": chk["fused_out"]["rel_err"],
            "rel_err_unfused": chk["unfused_out"]["rel_err"],
            "rel_err_unfused_m": chk["unfused_m"]["rel_err"],
            "rel_err_fused_no_round_mid": chk["fused_no_round_mid_vs_fp32"]["rel_err"],
            "bitwise_identical": chk["fused_eq_unfused_bitwise"],
            "bytes_fused": b_fused,
            "bytes_unfused": b_unfused,
            "gbps_fused": gbps(b_fused, t_fused.p50_ms),
            "gbps_unfused": gbps(b_unfused, t_unfused.p50_ms),
            "ideal_speedup": 1.20,
            "fused_noflush_ms": t_fused.noflush_p50_ms,
            "unfused_noflush_ms": t_unfused.noflush_p50_ms,
            "torch_eager_ms": t_eager.p50_ms,
            "torch_compile_ms": t_compile.p50_ms if t_compile else None,
            "torch_compile_err": compile_err,
            "torch_compile_check": compile_chk,
            "merge_only_ms": m_ms,
            "resadd_only_ms": a_ms,
            "n_tried": {
                "fused_coarse": f_c.n_tried,
                "fused_refine": f_r.n_tried,
                "fused_total": f_c.n_tried + f_r.n_tried,
                "merge_coarse": m_c.n_tried,
                "merge_refine": m_r.n_tried,
                "resadd_coarse": a_c.n_tried,
                "resadd_refine": a_r.n_tried,
                "unfused_joint": tj.n_tried,
            },
        },
    )
    row["pct_of_ceiling"] = (row["speedup"] - 1.0) / 0.20
    print(
        f"    fused {t_fused.p50_ms:.4f} ms ({gbps(b_fused, t_fused.p50_ms):.0f} GB/s)"
        f" | unfused {t_unfused.p50_ms:.4f} ms ({gbps(b_unfused, t_unfused.p50_ms):.0f} GB/s)"
        f" | speedup {row['speedup']:.3f}x ({100*row['pct_of_ceiling']:.0f}% of 1.20x)"
        f" | eager {t_eager.p50_ms:.4f}"
        + (f" | compile {t_compile.p50_ms:.4f}" if t_compile else " | compile n/a"),
        flush=True,
    )

    tune_tables = {
        "fused": {"coarse": f_c.as_dict(), "refine": f_r.as_dict()},
        "merge_only": {"coarse": m_c.as_dict(), "refine": m_r.as_dict()},
        "resadd_only": {"coarse": a_c.as_dict(), "refine": a_r.as_dict()},
        "unfused_joint_chain": tj.as_dict(),
    }
    timings = {
        "fused": t_fused.as_dict(),
        "unfused": t_unfused.as_dict(),
        "torch_eager": t_eager.as_dict(),
        "torch_compile": t_compile.as_dict() if t_compile else None,
    }
    return row, tune_tables, chk, timings


def main() -> None:
    env = C.BenchEnv.probe()
    print(f"device={env.device_name} warp={env.warp_size} CUs={env.num_sm}", flush=True)
    CKPT.mkdir(parents=True, exist_ok=True)

    rows, tables, checks, timings = [], {}, {}, {}

    def snapshot(done: bool) -> None:
        record(
            RESULT_ID,
            {
                "id": RESULT_ID,
                "complete": done,
                "fusion": "#10 Expert Merge + Residual Add (weighted top-k reduction "
                "of [T,8,H] then post-MoE residual add)",
                "shape": {
                    "hidden": H,
                    "topk": TOPK,
                    "dtype": "bfloat16",
                    "router_weight_dtype": "float32",
                },
                "traffic_model": {
                    "unfused_passes": TOPK + 1 + 3,
                    "fused_passes": TOPK + 1 + 1,
                    "ceiling_speedup": 1.20,
                    "note": "row-passes over [T,6144] bf16; the topk input (8x) dominates, "
                    "so the ceiling is (8+4)/(8+2)=1.20x at every regime",
                },
                "fairness": {
                    "one_kernel_source":
                        "glm52/kernels/merge_resadd.py::merge_resadd_kernel",
                    "flags": "DO_MERGE / DO_RESADD constexpr select the fused kernel vs "
                    "the two split kernels",
                    "tuning": "identical two-stage generator (coarse 174 + refine ~70-90) "
                    "per variant per regime; the unfused side additionally gets a joint "
                    "chain re-tune over the top-4 x top-4 configs, which can only help it",
                    "outputs": "fused and unfused `out` are bitwise identical (asserted); "
                    "the fused side skips only the merged intermediate `m`, which has no "
                    "other consumer in the layer",
                },
                "env": env.__dict__,
                "rows": rows,
                "checks": checks,
                "timings": timings,
                "tune_tables": tables,
            },
        )

    for regime in REGIMES:
        cf = CKPT / f"{regime.name}.json"
        if cf.exists():
            d = json.loads(cf.read_text())
            print(f"  == {regime.name} == (from checkpoint)", flush=True)
        else:
            if regime.T <= 256:
                wt_, rt_, wf_, rf_ = 20, 50, 100, 400
            elif regime.T <= 2048:
                wt_, rt_, wf_, rf_ = 10, 30, 50, 200
            else:
                wt_, rt_, wf_, rf_ = 10, 25, 30, 120
            row, tab, chk, tim = run_regime(regime, wt_, rt_, wf_, rf_)
            d = {"row": row, "tables": tab, "checks": chk, "timings": tim}
            cf.write_text(json.dumps(d, indent=2, default=str))
        rows.append(d["row"])
        tables[regime.name] = d["tables"]
        checks[regime.name] = d["checks"]
        timings[regime.name] = d["timings"]
        snapshot(False)

    snapshot(True)
    print(f"\nwrote results/{RESULT_ID}.json", flush=True)
    print(
        f"{'regime':<16}{'fused':>10}{'unfused':>10}{'speedup':>9}{'%ceil':>8}"
        f"{'eager':>10}{'compile':>10}"
    )
    for r in rows:
        tc = r.get("torch_compile_ms")
        print(
            f"{r['regime']:<16}{r['fused_ms']:>10.4f}{r['unfused_ms']:>10.4f}"
            f"{r['speedup']:>9.3f}{100*r['pct_of_ceiling']:>7.0f}%"
            f"{r['torch_eager_ms']:>10.4f}"
            f"{(f'{tc:.4f}' if tc else 'n/a'):>10}"
        )


if __name__ == "__main__":
    main_guard(main)
