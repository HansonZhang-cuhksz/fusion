"""Fusion #10 -- Expert Merge + Residual Add.  Fused vs unfused, tuned independently.

    unfused : merge  kernel  (read Y [T,8,H] + w, write m)      -> 8 R + 1 W
              resadd kernel  (read m, read res, write out)      -> 2 R + 1 W   = 12 passes
    fused   : one kernel     (read Y + w + res, write out)      -> 9 R + 1 W   = 10 passes

Bandwidth ceiling = 12/10 = 1.20x at every regime (the op has no meaningful FLOPs, so the
latency-aware ceiling equals the traffic ratio).

Both sides produce the same `out`; the fused side additionally avoids materializing the
merged intermediate `m`, which is exactly the benefit being measured.  `m` has no other
consumer in the GLM-5.2 decoder layer (the next op is the post-MoE RMSNorm, which reads
`out`), so nothing downstream is skipped.  The fused kernel reproduces the unfused chain's
round-to-bf16 of `m`, which makes the two outputs **bitwise identical** whenever both sides
tune to the same reduction order (recorded as `bitwise_identical`, not asserted -- the KVEC
slab and the KVEC=0 loop sum the 8 experts in a different order).

Run:
    python3 glm52_h200/bench/bench_f10_merge_resadd.py --gpu auto [--regimes ...] [--quick]

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
    bench_chain,
    check,
    main_guard,
    record,
    speedup_row,
)
from glm52_h200.kernels import merge_resadd as K

RESULT_ID = "f10_merge_resadd"
H = C.HIDDEN_SIZE  # 6144
TOPK = C.NUM_EXPERTS_PER_TOK  # 8
DT = C.DTYPE
UNITS = ["f10"]

_ENV = C.env()
WARP = B.env_int(_ENV, "warp_size")
MAX_THREADS = B.max_threads_per_block(_ENV)
ELEM_CAP = B.elems_per_program_cap(_ENV) // 2  # the KVEC slab holds TOPK planes at once

BLOCKS = [256, 512, 1024, 2048, 4096]
ROWSET = [1, 2, 4, 8, 16]
#: the warp ladder is a CTA-size ladder and it is the DEVICE's, not a remembered one:
#: 32..1024 threads is 1..16 warps at 64 lanes and 1..32 warps at 32 lanes.
WARPS = B.warp_ladder(_ENV)
CAPS = B.sm_wave_caps(_ENV)  # persistent-grid rungs in whole waves of this device


# --------------------------------------------------------------------------------------
# mapping search space -- ONE generator, used identically by every variant
# --------------------------------------------------------------------------------------
def _valid(cfg: dict) -> bool:
    b, r, w, kv = cfg["BLOCK_N"], cfg["ROWS"], cfg["num_warps"], cfg["KVEC"]
    th = w * WARP  # real lane count; halving it doubles every per-thread footprint below
    if th > MAX_THREADS:
        return False
    epr = b * r / th  # output elements per thread
    if epr < 4 or epr > 32:
        return False
    # KVEC materializes a [ROWS, TOPK, BLOCK_N] fp32 register slab
    if kv and b * r * TOPK / th > 64:
        return False
    if b * r * (TOPK if kv else 1) > ELEM_CAP:
        return False
    return True


def coarse_grid() -> list[dict]:
    """Tile width x rows/program x warps x stages x {loop, 3-D slab}.

    The count moves with the device because `_valid` counts REAL lanes -- 188 configs on
    sm89, 174 on C500.  It is recorded live in `fairness.grids`, never quoted.
    """
    out = []
    for b, r, w, s, kv in itertools.product(BLOCKS, (1, 2, 4, 8), WARPS, (1, 2), (0, 1)):
        cfg = dict(
            ROWS=r, BLOCK_N=b, num_warps=w, num_stages=s, KVEC=kv, UNROLL=1, grid_cap=None
        )
        if _valid(cfg):
            out.append(cfg)
    return B.widen(out, K)


def refine_grid(best: dict) -> list[dict]:
    """Neighbourhood of the coarse winner: +-1 step in tile width / rows / warps, the full
    stage sweep, the KVEC and UNROLL flips, the persistent-grid caps, and the
    eviction-hint variant of every one of them.  Same rules for every variant."""
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
        key = tuple(sorted((k, str(v)) for k, v in cfg.items()))
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
    out += [dict(c, EVICT=1) for c in list(out)]
    return B.dedup(out)


def _tune2(make_chain, tag, verify, warmup, rep, quick, fair, regime):
    """Two-stage tune (coarse -> refine), numerically screened."""
    t0 = time.time()
    cg = coarse_grid()
    if quick:
        cg = B.quick_slice(cg, 20)
    tc = B.screened_autotune(f"{tag}/coarse", make_chain, cg, verify, warmup, rep)
    rg = refine_grid(tc.best_cfg)
    if quick:
        rg = B.quick_slice(rg, 12)
    tr = B.screened_autotune(f"{tag}/refine", make_chain, rg, verify, warmup, rep)
    best_cfg, best_ms = (
        (tr.best_cfg, tr.best_ms) if tr.best_ms <= tc.best_ms else (tc.best_cfg, tc.best_ms)
    )
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
    warmup_t, rep_t, warmup_f, rep_f = B.reps(T, quick)
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

    def v_fused():
        c = check(out, ref_out, label="fused_out")
        return c["ok"], f"rel_err={c['rel_err']:.2e}"

    def v_merge():
        c = check(mu, ref_m, label="merge")
        return c["ok"], f"rel_err={c['rel_err']:.2e}"

    def v_resadd():
        c = check(outu, ref_out, label="resadd")
        return c["ok"], f"rel_err={c['rel_err']:.2e}"

    # ---- FUSED: one kernel ------------------------------------------------------------
    f_cfg, f_tune_ms, f_c, f_r = _tune2(
        lambda cfg: [lambda: K.fused_merge_resadd(y, wt, res, out, cfg)],
        "fused", v_fused, warmup_t, rep_t, quick, fair, regime.name,
    )

    # ---- UNFUSED: each kernel tuned independently over the SAME grid -------------------
    m_cfg, m_ms, m_c, m_r = _tune2(
        lambda cfg: [lambda: K.merge_only(y, wt, mu, cfg)],
        "merge", v_merge, warmup_t, rep_t, quick, fair, regime.name,
    )
    mu.copy_(ref_m)  # a valid input for the resadd screen, independent of merge's winner
    a_cfg, a_ms, a_c, a_r = _tune2(
        lambda cfg: [lambda: K.resadd_only(mu, res, outu, cfg)],
        "resadd", v_resadd, warmup_t, rep_t, quick, fair, regime.name,
    )

    # ---- extra, in the unfused side's favour: joint re-tune of the pair, timed as the
    # real chain (one flush before the pair, none between its two kernels).
    joint = [
        {"merge": mc, "resadd": ac}
        for mc, ac in itertools.product(
            B.top_cfgs(m_c, k=2) + B.top_cfgs(m_r, k=2),
            B.top_cfgs(a_c, k=2) + B.top_cfgs(a_r, k=2),
        )
    ]
    joint.append({"merge": m_cfg, "resadd": a_cfg})
    seen, uniq = set(), []
    for jc in joint:
        key = (tuple(sorted((k, str(v)) for k, v in jc["merge"].items())),
               tuple(sorted((k, str(v)) for k, v in jc["resadd"].items())))
        if key not in seen:
            seen.add(key)
            uniq.append(jc)
    tj = autotune(
        lambda jc: [
            lambda: K.merge_only(y, wt, mu, jc["merge"]),
            lambda: K.resadd_only(mu, res, outu, jc["resadd"]),
        ],
        uniq, warmup_t, rep_t,
    )
    u_cfg = tj.best_cfg
    fair.add(regime.name, "unfused_chain", "joint", tj)
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
    # single-rounding fp32 reference, chunked over T.  The unchunked form holds two [T,8,H]
    # fp32 temps live at once (3.0 GiB at T=8192); chunking cannot move a bit, because the
    # reduction is per row over topk and never split across rows.
    ref_out_hi = torch.empty_like(out)
    for i in range(0, T, 512):
        j = min(i + 512, T)
        ref_out_hi[i:j] = (
            (y[i:j].float() * wt[i:j].unsqueeze(-1)).sum(1) + res[i:j].float()
        ).to(DT)
    chk["fused_no_round_mid_vs_fp32"] = check(out_hi, ref_out_hi, label="no_round_mid")
    assert all(chk[k]["ok"] for k in chk if k != "fused_eq_unfused_bitwise"), chk
    del out_hi, ref_out_hi
    torch.cuda.empty_cache()

    # ---- final timing: INTERLEAVED, PAIRED ----------------------------------------------
    t_fused, t_unfused, pair = B.bench_pair(
        [lambda: K.fused_merge_resadd(y, wt, res, out, f_cfg)],
        [
            lambda: K.merge_only(y, wt, mu, u_cfg["merge"]),
            lambda: K.resadd_only(mu, res, outu, u_cfg["resadd"]),
        ],
        warmup_f, rep_f, label=regime.name,
    )

    # ---- torch production lines ----------------------------------------------------------
    def torch_eager():
        mm = ref.expert_merge(y, wt)
        return (mm.float() + res.float()).to(DT)

    t_eager, eager_err = None, None
    try:
        t_eager = bench_chain([torch_eager], warmup_f, rep_f)
    except torch.cuda.OutOfMemoryError as exc:
        eager_err = f"{type(exc).__name__}: {exc}"[:400]
        torch.cuda.empty_cache()
        print(f"    torch eager FAILED: {eager_err}", flush=True)

    t_compile, compile_err, compile_chk = None, None, None
    try:
        fn = torch.compile(
            lambda yy, ww, rr: (ref.expert_merge(yy, ww).float() + rr.float()).to(DT)
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
    gbps = lambda b, ms: b / (ms * 1e-3) / 1e9  # noqa: E731
    l2 = B.env_int(_ENV, "l2_bytes")

    tmodel = B.traffic_ceilings(regime)
    row = speedup_row(regime.name, t_fused, t_unfused, {
        "T": T,
        "variant": "f10",
        "ceiling": (tmodel.get("F10_merge_resadd") or {}).get("roofline_ceiling"),
        "ceiling_with_launch": (tmodel.get("F10_merge_resadd") or {})
        .get("roofline_ceiling_with_launch"),
        "traffic_ratio_model": (tmodel.get("F10_merge_resadd") or {})
        .get("traffic_ratio"),
        "fused_cfg": f_cfg,
        "unfused_cfg": u_cfg,
        "paired_speedup": pair.get("paired_speedup_p50"),
        "paired_speedup_trimmed": pair.get("paired_speedup_trimmed_mean"),
        "pair_meta": pair,
        "tick": B.tick_report(t_fused.p50_ms, t_unfused.p50_ms),
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
        # `m` is the intermediate the fusion deletes; where it is already L2-resident the
        # deletion saves no DRAM traffic, so the 1.20x is not on the table.
        "m_bytes": act,
        "l2_bytes": l2,
        "m_fits_in_l2": act <= l2,
        "ceiling_note": "1.20x is a DRAM-traffic ceiling; unattainable where m fits in L2",
        "torch_eager_ms": t_eager.p50_ms if t_eager else None,
        "torch_eager_err": eager_err,
        "torch_compile_ms": t_compile.p50_ms if t_compile else None,
        "torch_compile_err": compile_err,
        "torch_compile_check": compile_chk,
        "merge_only_ms": m_ms,
        "resadd_only_ms": a_ms,
    }, pair=pair)  # headline `speedup` = PAIRED median; `speedup_sequential` kept
    sp = row.get("paired_speedup") or row["speedup"]
    row["pct_of_ceiling"] = (sp - 1.0) / 0.20
    print(
        f"    fused {t_fused.p50_ms:.4f} ms ({gbps(b_fused, t_fused.p50_ms):.0f} GB/s)"
        f" | unfused {t_unfused.p50_ms:.4f} ms "
        f"({gbps(b_unfused, t_unfused.p50_ms):.0f} GB/s)"
        f" | paired {sp:.3f}x ({100 * row['pct_of_ceiling']:.0f}% of 1.20x)"
        + (f" | eager {t_eager.p50_ms:.4f}" if t_eager else " | eager n/a")
        + (f" | compile {t_compile.p50_ms:.4f}" if t_compile else " | compile n/a")
        + ("  [TICK-LIMITED]" if row["tick"].get("tick_limited") else ""),
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
        "pair": pair,
        "torch_eager": t_eager.as_dict() if t_eager else None,
        "torch_compile": t_compile.as_dict() if t_compile else None,
    }
    del y, wt, res, out, mu, outu, ref_m, ref_out
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
    B.calibration_gate(args)
    B.exact_fp32_matmul()
    B.check_preflight_device(env)
    B.resolve_units(UNITS, args.only)
    regimes = B.resolve_regimes(C, args.regimes)

    fair = B.Fairness(
        one_kernel_source="glm52_h200/kernels/merge_resadd.py::merge_resadd_kernel",
        flags="DO_MERGE / DO_RESADD constexpr select the fused kernel vs the two split "
              "kernels",
        protocol="identical two-stage generator (coarse + per-winner refine) per variant "
                 "per regime, every config screened against the fp32 reference before it "
                 "is timed; the unfused side additionally gets a joint chain re-tune over "
                 "the top-4 x top-4 configs, which can only help it",
        outputs="fused and unfused `out` are bitwise identical whenever both sides tune to "
                "the same reduction order -- recorded per regime as `bitwise_identical`, "
                "not asserted; the fused side skips only the merged intermediate `m`, "
                "which has no other consumer in the layer",
        h200_axes=(
            "NONE of the sm_90 mapping axes apply to this family, on EITHER arm. "
            "merge_resadd_kernel is a weighted top-k reduction over [T,8,H] -- a strided "
            "vector pass, not a GEMM: there is no MMA pipeline for warp specialization to "
            "overlap against, no k-loop tile for a TMA descriptor, and no SMEM tile whose "
            "budget a cluster would enlarge. Both arms are therefore offered the same empty "
            "set, and their axis_counts are legitimately zero. What this family does gain "
            "from the H200 is the 60 MB L2, which is a residency effect the traffic model "
            "and the per-regime `*_fits_l2` fields already carry."
        ),
    )
    fair.axis("f10_merge_resadd", B.h200_axis_report(K))

    rows, tables, checks, timings, pair_meta = [], {}, {}, {}, None

    def snapshot(done: bool) -> None:
        record(RESULT_ID, {
            "id": RESULT_ID,
            "complete": done,
            "fusion": "#10 Expert Merge + Residual Add (weighted top-k reduction of "
                      "[T,8,H] then post-MoE residual add)",
            "shape": {"hidden": H, "topk": TOPK, "dtype": "bfloat16",
                      "router_weight_dtype": "float32"},
            "traffic_model": {
                "unfused_passes": TOPK + 1 + 3,
                "fused_passes": TOPK + 1 + 1,
                "ceiling_speedup": 1.20,
                "note": "row-passes over [T,6144] bf16; the topk input (8x) dominates, so "
                        "the ceiling is (8+4)/(8+2)=1.20x at every regime",
                "production_note": "1.20x is measured against the 3-kernel baseline. "
                "sglang ships `moe_sum` then defers the add into the next layer's "
                "`fused_add_rmsnorm` (13 passes), so the end-to-end saving an engineer can "
                "bank is 13/12 = 1.083x -- LOG-06 7.1",
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
        snapshot(False)

    snapshot(True)
    print(f"\nwrote {RESULT_ID}.json", flush=True)
    print(f"{'regime':<16}{'fused':>10}{'unfused':>10}{'paired':>9}{'%ceil':>8}"
          f"{'eager':>10}{'compile':>10}")
    for r in rows:
        tc, te = r.get("torch_compile_ms"), r.get("torch_eager_ms")
        print(
            f"{r['regime']:<16}{r['fused_ms']:>10.4f}{r['unfused_ms']:>10.4f}"
            f"{(r.get('paired_speedup') or r['speedup']):>9.3f}"
            f"{100 * r['pct_of_ceiling']:>7.0f}%"
            f"{(f'{te:.4f}' if te else 'n/a'):>10}"
            f"{(f'{tc:.4f}' if tc else 'n/a'):>10}"
        )


if __name__ == "__main__":
    main_guard(main)
