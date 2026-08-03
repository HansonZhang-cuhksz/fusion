"""Fusion #6 -- MoE Up/Gate grouped GEMM + SwiGLU epilogue.

FUSED    : one kernel, grid over N = I = 2048, two fp32 accumulators (gate cols and up
           cols) sharing one K-loop over the gathered A tile, `silu(g)*u` in the epilogue,
           writes only [rows, 2048].
UNFUSED  : the SAME kernel with FUSE_ACT=False -- grid over N = 2I = 4096, one accumulator,
           writes the full [rows, 4096] intermediate -- followed by a separate element-wise
           `silu_and_mul` kernel reading [rows, 4096] and writing [rows, 2048].  This is
           exactly what sglang 0.5.10 does today.

Both sides produce the identical downstream tensor `down_input` [rows, 2048] bf16.  The
unfused side additionally materialises the [rows, 4096] intermediate -- that extra traffic
IS the fusion opportunity.

WHY THIS FAMILY EXISTS AGAIN.  On C500 the unfused winner's tile (BM128/BN128/BK32/s4)
needed 96 KB when fused against a hard 64 KB ceiling, so the fused arm was *uncompilable*
at the shape that mattered and the family scored 0.553x at prefill.  That was a hard bar,
not a performance effect, and it is a property of the ceiling -- which is why the ceiling
here comes from the probe and the footprint model comes from `C.smem_stage_bytes` rather
than from the kernel module's own (Triton-3.0-era) estimate.  The 4060 could not test it at
all: w13 alone is 12.0 GB against 7.4 GB usable.  H200 can.

Run:
    python3 glm52_h200/bench/bench_f06_upgate_swiglu.py [--regimes ...] [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from glm52_h200 import bench as B
from glm52_h200 import config as C
from glm52_h200 import reference as R
from glm52_h200.common import (
    autotune,
    bench_chain,
    check,
    main_guard,
    record,
    speedup_row,
)
from glm52_h200.kernels import moe_gateup as KG
from glm52_h200.kernels.moe_gateup import launch_gateup, launch_silu_and_mul

RESULT_ID = "f06_upgate_swiglu"
H = C.HIDDEN_SIZE  # 6144
I = C.MOE_INTERMEDIATE_SIZE  # 2048
E = C.N_ROUTED_EXPERTS  # 256
TOPK = C.NUM_EXPERTS_PER_TOK  # 8
UNITS = ["f6"]

_ENV = C.env()
SMEM_LIMIT = B.env_int(_ENV, "smem_bytes")
WARP = B.env_int(_ENV, "warp_size")
MAX_THREADS = B.max_threads_per_block(_ENV)
WARPS = B.warp_ladder(_ENV, lo=2)
ACC_PER_LANE_MAX = 128
ACC_PER_LANE_MIN = 2


# --------------------------------------------------------------------------------------
# Config-space generation.  Identical rules for both variants; the SMEM / accumulator
# filters differ only because the fused kernel genuinely stages a second B tile and holds a
# second accumulator.  That asymmetry is physical, and it is the whole subject of the
# comparison -- so it is modelled explicitly rather than absorbed into a shared constant.
# --------------------------------------------------------------------------------------
def _ok(cfg: dict, fused: bool, acc_lo: float = 4.0) -> bool:
    bm, bn, bk = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"]
    w, s = cfg["num_warps"], cfg["num_stages"]
    # `bn_mult=2` for the fused variant: it stages the gate tile AND the up tile.
    if C.smem_stage_bytes(bm, bn, bk, s, bn_mult=2 if fused else 1) > SMEM_LIMIT:
        return False
    threads = w * WARP
    if threads > MAX_THREADS:
        return False
    acc_per_lane = (2 if fused else 1) * bm * bn / threads
    return acc_lo <= acc_per_lane <= ACC_PER_LANE_MAX


def gemm_grid(fused: bool, big: bool) -> list[dict]:
    if big:  # T >= 2048: drop mappings that are structurally hopeless for a big GEMM
        bms, bns, bks = [32, 64, 128, 256], [64, 128, 256], [32, 64, 128]
    else:
        bms, bns, bks = [16, 32, 64, 128], [32, 64, 128, 256], [32, 64, 128]
    out = []
    for bm, bn, bk, w, s in itertools.product(bms, bns, bks, WARPS, [2, 3, 4]):
        cfg = dict(
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, GROUP_M=8
        )
        if _ok(cfg, fused):
            out.append(cfg)
    return B.widen(out, KG)


def gemm_refine(best: dict, fused: bool) -> list[dict]:
    def nb(v, lo, hi):
        return sorted({max(lo, v // 2), v, min(hi, v * 2)})

    cands = []
    for bm in nb(best["BLOCK_M"], 16, 256):
        for bn in nb(best["BLOCK_N"], 32, 256):
            for w in nb(best["num_warps"], 1, WARPS[-1]):
                cands.append((bm, bn, best["BLOCK_K"], w, best["num_stages"], 8))
    for bk in nb(best["BLOCK_K"], 32, 128):
        for s in (2, 3, 4, 5):
            cands.append((best["BLOCK_M"], best["BLOCK_N"], bk, best["num_warps"], s, 8))
    for g in (1, 4, 8, 16):
        cands.append((best["BLOCK_M"], best["BLOCK_N"], best["BLOCK_K"],
                      best["num_warps"], best["num_stages"], g))
    out, seen = [], set()
    for bm, bn, bk, w, s, g in cands:
        cfg = dict(
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, GROUP_M=g
        )
        key = tuple(sorted(cfg.items()))
        if key in seen or not _ok(cfg, fused, acc_lo=ACC_PER_LANE_MIN):
            continue
        seen.add(key)
        out.append(cfg)
    return out


def act_grid() -> list[dict]:
    out = []
    for bm, bn, w, s in itertools.product(
        [1, 2, 4, 8, 16, 32, 64], [64, 128, 256, 512, 1024, 2048], WARPS, [1, 2]
    ):
        tile = bm * bn
        if tile > 8192 or tile < 256:
            continue
        threads = w * WARP
        if threads > MAX_THREADS or tile < threads:
            continue
        out.append(dict(BLOCK_M=bm, BLOCK_N=bn, num_warps=w, num_stages=s))
    return out


# --------------------------------------------------------------------------------------
# Per-regime problem setup
# --------------------------------------------------------------------------------------
class Problem:
    def __init__(self, regime, w13, gate_w, seed=0):
        torch.manual_seed(seed + regime.T)
        dev = "cuda"
        self.regime = regime
        self.T = regime.T
        self.rows = regime.T * TOPK
        self.w13 = w13
        # post_attention_layernorm output feeding the expert GEMM
        self.x = (torch.randn(self.T, H, device=dev, dtype=torch.float32) * 0.1).to(
            torch.bfloat16
        )
        _, self.topk_weights, self.topk_ids = R.router(self.x, gate_w)
        self.layouts: dict[int, tuple] = {}
        # output buffers (allocated once, as sglang allocates a workspace)
        self.c_fused = torch.zeros(self.rows, I, device=dev, dtype=torch.bfloat16)
        self.c_inter = torch.zeros(self.rows, 2 * I, device=dev, dtype=torch.bfloat16)
        self.c_unfused = torch.zeros(self.rows, I, device=dev, dtype=torch.bfloat16)

    def layout(self, block_m: int):
        if block_m not in self.layouts:
            self.layouts[block_m] = R.moe_align_block_size(self.topk_ids, block_m, E)
        return self.layouts[block_m]

    # --- callable factories ---------------------------------------------------------
    def fused_fn(self, cfg):
        sti, eids, ntp = self.layout(cfg["BLOCK_M"])
        return lambda: launch_gateup(
            self.x, self.w13, self.c_fused, sti, eids, ntp,
            self.rows, TOPK, I, cfg, fused=True,
        )

    def gemm_fn(self, cfg):
        sti, eids, ntp = self.layout(cfg["BLOCK_M"])
        return lambda: launch_gateup(
            self.x, self.w13, self.c_inter, sti, eids, ntp,
            self.rows, TOPK, I, cfg, fused=False,
        )

    def act_fn(self, cfg):
        return lambda: launch_silu_and_mul(self.c_inter, self.c_unfused, cfg)


def reference_rows(prob: Problem, n_sample: int = 2048):
    """fp32 reference on a sampled row subset.

    A full-size fp32 reference at T=8192 is 3.3 TFLOP of fp32 matmul; sampling keeps
    validation honest and affordable, and the SAME sampled rows judge both arms.
    """
    rows = prob.rows
    if rows <= n_sample:
        idx = torch.arange(rows, device="cuda")
    else:
        g = torch.Generator(device="cuda").manual_seed(1234)
        idx = torch.randperm(rows, device="cuda", generator=g)[:n_sample].sort().values
    tok = (idx // TOPK).long()
    kk = (idx % TOPK).long()
    experts = prob.topk_ids.long()[tok, kk]
    ref = torch.empty(idx.numel(), I, device="cuda", dtype=torch.float32)
    xs = prob.x.float()[tok]
    for e in torch.unique(experts).tolist():
        sel = (experts == e).nonzero(as_tuple=True)[0]
        h = xs[sel] @ prob.w13[e].float().t()
        ref[sel] = (torch.nn.functional.silu(h[:, :I]) * h[:, I:]).float()
    return idx, ref


def vendor_chain(prob: Problem):
    """Vendor-BLAS production reference: per-expert torch.matmul + torch silu_and_mul.

    A rows are pre-gathered per expert OUTSIDE the timed region, so this is the best case
    for the vendor path -- pure GEMM + activation, no gather cost.  Stated in the log.
    """
    flat = prob.topk_ids.flatten().long()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E)
    starts = torch.cat(
        [torch.zeros(1, dtype=torch.long, device="cuda"), counts.cumsum(0)[:-1]]
    )
    a_sorted = prob.x[(order // TOPK)].contiguous()
    out = torch.empty(prob.rows, I, device="cuda", dtype=torch.bfloat16)
    cs, ss = counts.tolist(), starts.tolist()
    segs = [(e, ss[e], ss[e] + cs[e]) for e in range(E) if cs[e]]
    wt = [prob.w13[e].t() for e, _, _ in segs]  # bf16 views, no copy

    def run():
        for (e, s, t), w in zip(segs, wt):
            h = torch.matmul(a_sorted[s:t], w)
            out[s:t] = torch.nn.functional.silu(h[:, :I]) * h[:, I:]

    return [run]


def make_weights():
    """w13 with an sglang-style trailing pad.

    sglang guards its `fused_moe` weights with `SGLANG_MOE_PADDING`.  Here the pad exists
    because Triton's software pipeline issues speculative (unpredicated) B-tile loads for
    the peeled prologue/epilogue stages, so the last expert's `up` tile can be fetched one
    BLOCK_K past the end of the tensor.  On MACA that was a page fault that killed the
    context; on CUDA it is an out-of-bounds read of whatever follows the allocation.  The
    pad is allocated for BOTH variants and never read into an accumulator, so it changes no
    arithmetic and gives neither side an advantage.
    """
    numel = E * 2 * I * H
    pad = 1 << 20  # 2 MiB of slack, >> any BLOCK_K * BLOCK_N tile
    buf = torch.empty(numel + pad, device="cuda", dtype=torch.bfloat16)
    w13 = buf[:numel].view(E, 2 * I, H)
    for e in range(E):  # chunked init: avoids a 12.9 GB fp32 temporary
        w13[e].normal_(0, 0.02)
    buf[numel:].zero_()
    return buf, w13


def run_regime(regime, w13, gate_w, quick: bool, fair: B.Fairness) -> tuple[dict, dict]:
    big = regime.T >= 2048
    w_t, r_t, w_f, r_f = B.reps(regime.T, quick)

    with torch.no_grad():
        print(f"\n===== {regime.name} (T={regime.T}, rows={regime.T * TOPK}) =====",
              flush=True)
        prob = Problem(regime, w13, gate_w)
        idx, ref = reference_rows(prob)

        def v_fused():
            c = check(prob.c_fused[idx], ref, label="fused")
            return c["ok"], f"rel_err={c['rel_err']:.2e}"

        def v_gemm():
            # the GEMM alone writes [rows, 2I]; validate silu(g)*u derived from it, which
            # is the tensor the chain actually hands downstream
            got = (
                torch.nn.functional.silu(prob.c_inter[idx][:, :I].float())
                * prob.c_inter[idx][:, I:].float()
            )
            c = check(got, ref, label="unfused_gemm")
            return c["ok"], f"rel_err={c['rel_err']:.2e}"

        def v_act():
            c = check(prob.c_unfused[idx], ref, label="unfused_act")
            return c["ok"], f"rel_err={c['rel_err']:.2e}"

        # ---------------- FUSED ----------------
        cg = gemm_grid(True, big)
        if quick:
            cg = B.quick_slice(cg, 12)
        print(f"  fused coarse: {len(cg)} cfgs", flush=True)
        tf_c = B.screened_autotune(
            "fused/coarse", lambda c: [prob.fused_fn(c)], cg, v_fused, w_t, r_t
        )
        rg = gemm_refine(tf_c.best_cfg, True)
        if quick:
            rg = B.quick_slice(rg, 8)
        print(f"  fused coarse best {tf_c.best_cfg} {tf_c.best_ms:.4f} ms; refine {len(rg)}",
              flush=True)
        tf_r = B.screened_autotune(
            "fused/refine", lambda c: [prob.fused_fn(c)], rg, v_fused, w_t, r_t
        )
        fused_cfg = tf_c.best_cfg if tf_c.best_ms <= tf_r.best_ms else tf_r.best_cfg
        fair.add(regime.name, "fused", "coarse", tf_c)
        fair.add(regime.name, "fused", "refine", tf_r)
        print(f"  FUSED best {fused_cfg} {min(tf_c.best_ms, tf_r.best_ms):.4f} ms",
              flush=True)

        # ---------------- UNFUSED: GEMM ----------------
        cg2 = gemm_grid(False, big)
        if quick:
            cg2 = B.quick_slice(cg2, 12)
        print(f"  unfused-GEMM coarse: {len(cg2)} cfgs", flush=True)
        tu_c = B.screened_autotune(
            "unfusedGEMM/coarse", lambda c: [prob.gemm_fn(c)], cg2, v_gemm, w_t, r_t
        )
        rg2 = gemm_refine(tu_c.best_cfg, False)
        if quick:
            rg2 = B.quick_slice(rg2, 8)
        print(f"  unfused-GEMM coarse best {tu_c.best_cfg} {tu_c.best_ms:.4f} ms; "
              f"refine {len(rg2)}", flush=True)
        tu_r = B.screened_autotune(
            "unfusedGEMM/refine", lambda c: [prob.gemm_fn(c)], rg2, v_gemm, w_t, r_t
        )
        fair.add(regime.name, "unfused_gemm", "coarse", tu_c)
        fair.add(regime.name, "unfused_gemm", "refine", tu_r)

        # ---------------- UNFUSED: silu_and_mul ----------------
        ag = act_grid()
        if quick:
            ag = B.quick_slice(ag, 8)
        print(f"  act grid: {len(ag)} cfgs", flush=True)
        prob.gemm_fn(tu_c.best_cfg)()  # a valid [rows, 2I] input for the act screen
        torch.cuda.synchronize()
        ta = B.screened_autotune(
            "unfusedACT", lambda c: [prob.act_fn(c)], ag, v_act, w_t, r_t
        )
        fair.add(regime.name, "unfused_act", "tune", ta)
        print(f"  ACT best {ta.best_cfg} {ta.best_ms:.4f} ms", flush=True)

        # top-3 x top-3 joint chain re-time (guards against a separately-tuned optimum that
        # is not the joint optimum).  Unfused-side only, so it can only help the baseline.
        best_chain_ms, best_pair = float("inf"), None
        joint = []
        for gc in B.top_cfgs(tu_c, tu_r, k=3):
            for ac in B.top_cfgs(ta, k=3):
                try:
                    t = bench_chain([prob.gemm_fn(gc), prob.act_fn(ac)], w_t, r_t)
                    joint.append(({"gemm": gc, "act": ac}, t.p50_ms, None))
                    if t.p50_ms < best_chain_ms:
                        best_chain_ms, best_pair = t.p50_ms, (gc, ac)
                except Exception as exc:  # noqa: BLE001
                    joint.append(({"gemm": gc, "act": ac}, None, str(exc)[:160]))
        if best_pair is None:
            raise RuntimeError(f"{regime.name}: no unfused chain combination ran")
        gemm_cfg, act_cfg = best_pair
        fair.add(regime.name, "unfused_chain", "joint", size=len(joint))
        print(f"  UNFUSED best chain {gemm_cfg} + {act_cfg} {best_chain_ms:.4f} ms",
              flush=True)

        # ---------------- validate the winners ----------------
        prob.c_fused.zero_(); prob.c_inter.zero_(); prob.c_unfused.zero_()
        prob.fused_fn(fused_cfg)()
        prob.gemm_fn(gemm_cfg)()
        prob.act_fn(act_cfg)()
        torch.cuda.synchronize()
        chk_f = check(prob.c_fused[idx], ref, label="fused")
        chk_u = check(prob.c_unfused[idx], ref, label="unfused")
        agree = check(prob.c_fused[idx], prob.c_unfused[idx].float(),
                      label="fused_vs_unfused")
        print(f"  rel_err fused={chk_f['rel_err']:.3e} unfused={chk_u['rel_err']:.3e} "
              f"agree={agree['rel_err']:.3e}", flush=True)
        if not (chk_f["ok"] and chk_u["ok"]):
            raise RuntimeError(f"validation failed at {regime.name}: {chk_f} {chk_u}")

        # ---------------- final timing: INTERLEAVED, PAIRED ----------------
        t_fused, t_unfused, pair = B.bench_pair(
            [prob.fused_fn(fused_cfg)],
            [prob.gemm_fn(gemm_cfg), prob.act_fn(act_cfg)],
            w_f, r_f, label=regime.name,
        )
        t_gemm_only = bench_chain([prob.gemm_fn(gemm_cfg)], w_f, r_f)
        t_act_only = bench_chain([prob.act_fn(act_cfg)], w_f, r_f)
        t_vendor = bench_chain(
            vendor_chain(prob), max(2, w_f // 3), max(5, r_f // 3)
        )

        rows_n = prob.rows
        flops = 2.0 * rows_n * (2 * I) * H
        row = speedup_row(regime.name, t_fused, t_unfused, extra={
            "T": regime.T,
            "variant": "f6",
            "moe_rows": rows_n,
            "fused_cfg": fused_cfg,
            "unfused_gemm_cfg": gemm_cfg,
            "unfused_act_cfg": act_cfg,
            "paired_speedup": pair.get("paired_speedup_p50"),
            "paired_speedup_trimmed": pair.get("paired_speedup_trimmed_mean"),
            "pair_meta": pair,
            "tick": B.tick_report(t_fused.p50_ms, t_unfused.p50_ms),
            "unfused_gemm_ms": t_gemm_only.p50_ms,
            "unfused_act_ms": t_act_only.p50_ms,
            "vendor_blas_ms": t_vendor.p50_ms,
            "vendor_tflops": flops / (t_vendor.p50_ms * 1e-3) / 1e12,
            "fused_tflops": flops / (t_fused.p50_ms * 1e-3) / 1e12,
            "unfused_tflops": flops / (t_unfused.p50_ms * 1e-3) / 1e12,
            "rel_err": chk_f["rel_err"],
            "rel_err_unfused": chk_u["rel_err"],
            "fused_vs_unfused_rel_err": agree["rel_err"],
            "gflop": flops / 1e9,
            "traffic": {
                "A_bytes": rows_n * H * 2,
                "unfused_intermediate_bytes": rows_n * 2 * I * 2 * 2 + rows_n * I * 2,
                "fused_intermediate_bytes": rows_n * I * 2,
                "saved_bytes": rows_n * 2 * I * 2 * 2,
                "weight_bytes_min": int(
                    torch.unique(prob.topk_ids).numel() * 2 * I * H * 2
                ),
            },
            # The C500 blocker was compilability, not speed: record the modelled footprint
            # of BOTH winners against this device's actual ceiling so the question "would
            # the unfused winner even compile fused here?" is answerable from the JSON.
            "smem_model": {
                "ceiling_bytes": SMEM_LIMIT,
                "fused_winner_bytes": C.smem_stage_bytes(
                    fused_cfg["BLOCK_M"], fused_cfg["BLOCK_N"], fused_cfg["BLOCK_K"],
                    fused_cfg["num_stages"], bn_mult=2),
                "unfused_winner_bytes": C.smem_stage_bytes(
                    gemm_cfg["BLOCK_M"], gemm_cfg["BLOCK_N"], gemm_cfg["BLOCK_K"],
                    gemm_cfg["num_stages"]),
                "unfused_winner_if_fused_bytes": C.smem_stage_bytes(
                    gemm_cfg["BLOCK_M"], gemm_cfg["BLOCK_N"], gemm_cfg["BLOCK_K"],
                    gemm_cfg["num_stages"], bn_mult=2),
            },
            "fused_kernel_stats": B.kernel_stats(
                prob.fused_fn(fused_cfg), getattr(KG, "moe_gateup_kernel", None)),
            "unfused_kernel_stats": B.kernel_stats(
                prob.gemm_fn(gemm_cfg), getattr(KG, "moe_gateup_kernel", None)),
        })
        tuning = {
            "fused_coarse": tf_c.as_dict(),
            "fused_refine": tf_r.as_dict(),
            "unfused_gemm_coarse": tu_c.as_dict(),
            "unfused_gemm_refine": tu_r.as_dict(),
            "unfused_act": ta.as_dict(),
            "unfused_joint_chain": joint,
        }
        print(
            f"  RESULT {regime.name}: fused {t_fused.p50_ms:.4f} ms | unfused "
            f"{t_unfused.p50_ms:.4f} ms (gemm {t_gemm_only.p50_ms:.4f} + act "
            f"{t_act_only.p50_ms:.4f}) | paired {row['paired_speedup']:.3f}x | "
            f"vendor {t_vendor.p50_ms:.4f} ms",
            flush=True,
        )
        del prob
        torch.cuda.empty_cache()
        return row, tuning


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

    # The C500 study ran one process per regime because the MACA runtime disabled the whole
    # context after an ATU fault.  That does not apply here, and re-allocating 12.9 GB of
    # expert weights seven times would dominate the run; the equivalent protection is the
    # per-regime checkpoint, which lets a re-invocation resume after any hard failure.
    need = E * 2 * I * H * 2
    cap = B.mem_guard(need, "w13 [256, 4096, 6144] bf16")
    if not cap["fits"]:
        raise RuntimeError(
            f"w13 needs {need / 2**30:.1f} GB and only {cap['free_bytes'] / 2**30:.1f} GB "
            f"is free. #6 cannot run at exact GLM-5.2 spec on this device as configured "
            f"(check MIG mode and other tenants); reducing the expert count would change "
            f"the grouped-GEMM tiling and is not a valid substitute."
        )
    torch.manual_seed(0)
    _buf, w13 = make_weights()
    gate_w = torch.randn(E, H, device="cuda", dtype=torch.float32) * 0.02

    fair = B.Fairness(
        one_kernel_source="glm52_h200/kernels/moe_gateup.py::moe_gateup_kernel",
        flags="FUSE_ACT constexpr selects the SwiGLU epilogue; unfused = the same kernel "
              "with the flag off over N=2I, plus silu_and_mul_kernel",
        protocol=(
            "Per regime, per side: coarse grid (SMEM- and accumulator-prefiltered by "
            "IDENTICAL rules, differing only in the physically real second staged B tile "
            "and second accumulator the fused kernel holds) then a neighbourhood refine "
            "around that side's own coarse winner. Every config is validated against a "
            "sampled fp32 reference before it is timed. The unfused chain's two kernels "
            "are tuned SEPARATELY and then the top-3 x top-3 combinations are re-timed AS "
            "A CHAIN, so the separately-tuned optimum cannot under-sell the unfused side."
        ),
        smem_model="C.smem_stage_bytes (num_stages-1 buffers, floor 2) against the probed "
                   "opt-in per-block ceiling -- NOT the kernel module's Triton-3.0-era "
                   "estimate, which over-predicts by 1.33-1.5x and would reject tiles this "
                   "stack runs",
    )

    rows, tuning, pair_meta = [], {}, None
    for regime in regimes:
        ck = B.ckpt_load(RESULT_ID, regime.name, env, force=args.force)
        if ck is not None:
            print(f"  == {regime.name} == (from checkpoint)", flush=True)
            rows.append(ck["row"])
            tuning[regime.name] = ck["tuning"]
            fair.grids.update(ck.get("fairness_grids", {}))
            continue
        try:
            row, tun = run_regime(regime, w13, gate_w, args.quick, fair)
        except Exception as exc:  # noqa: BLE001 -- one regime must not lose the rest
            import traceback

            traceback.print_exc()
            tuning[regime.name] = {"regime_failed": f"{type(exc).__name__}: {exc}"[:300]}
            torch.cuda.empty_cache()
            continue
        pair_meta = row.get("pair_meta")
        B.ckpt_save(RESULT_ID, regime.name, env, {
            "row": row, "tuning": tun,
            "fairness_grids": {regime.name: fair.grids.get(regime.name, {})},
        })
        rows.append(row)
        tuning[regime.name] = tun

    payload = {
        "id": RESULT_ID,
        "fusion": "#6 MoE Up/Gate grouped GEMM + SwiGLU epilogue",
        "shapes": {"H": H, "I": I, "E": E, "topk": TOPK, "K": H,
                   "N_unfused": 2 * I, "N_fused": I},
        "env": env.__dict__,
        "capacity": cap,
        "fairness": fair.render(env, pair_meta),
        "rows": rows,
        "tuning": tuning,
    }
    p = record(RESULT_ID, payload)
    print(f"\nwrote {p}", flush=True)
    print(f"{'regime':16s} {'fused':>10s} {'unfused':>10s} {'paired':>8s} "
          f"{'vendor':>10s} {'rel_err':>10s}")
    for r in rows:
        print(f"{r['regime']:16s} {r['fused_ms']:10.4f} {r['unfused_ms']:10.4f} "
              f"{(r.get('paired_speedup') or r['speedup']):8.3f} "
              f"{r['vendor_blas_ms']:10.4f} {r['rel_err']:10.2e}")


if __name__ == "__main__":
    main_guard(main)
