"""Fusions #8 (down GEMM + expert merge) and #9 (+ residual add 2).

GEMM shape (GLM-5.2 MoE, second expert GEMM):
    A = act        [rows = T*8, I = 2048]   bf16   (SwiGLU output, one row per (tok,k))
    B = w2         [E = 256, H = 6144, I = 2048] bf16
    N = H = 6144,  K = I = 2048

Sides compared (all produce the identical downstream tensor `out [T, 6144]` bf16):

  UNFUSED #8  : down GEMM (MUL_ROUTED_WEIGHT, FUSE_MERGE=False) -> intermediate_cache3
                [T, 8, 6144]  ;  then `moe_sum_kernel` reduces over topk -> [T, 6144].
                == what sglang 0.5.10 runs in production.
  UNFUSED #9  : the above + a separate `resadd_kernel`  (strict 3-kernel baseline).
  UNFUSED #9b : down GEMM + `moe_sum_kernel(ADD_RESIDUAL=True)` (2-kernel baseline; a cheap
                production improvement that costs nothing, reported as a check that the #9
                win is not just "we compared against a needlessly bad baseline").

  FUSED #8 (a) atomic       : seed out[T,6144] with zeros, then the SAME down GEMM with
                              FUSE_MERGE=True atomically accumulating into out.  The
                              seeding kernel IS inside the timed chain.
  FUSED #8 (b) token-major  : one CTA per (token, n-block), loops over that token's 8
                              experts, sums in registers, one store.  No atomics, no
                              [T, 8, 6144] tensor at all.
  FUSED #9 (a) atomic       : identical, except the seed kernel writes the residual.
  FUSED #9 (b) token-major  : identical, plus one residual load in the epilogue.

The #8-vs-#10 crossover is the one large, robust effect in the whole study (LOG-11 4): #8
wins at small decode batches and costs 5.7 % at prefill_t2048.  The C500 explanation was L2
residency of the [T, 6144] atomic accumulator, and the measurement said the direction was
right and the threshold wrong.  H200's L2 is ~50 MB against C500's 8 MB, which moves the
predicted threshold by 6x -- so `accum_bytes` / `accum_fits_l2` are recorded per regime and
decode_bs512 / decode_bs1024 are in scope precisely to locate it again.

Run:
    python3 glm52_h200/bench/bench_f08f09_down_merge_resadd.py --gpu auto [--regimes ...] [--only ...]

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
from glm52_h200 import reference as R
from glm52_h200.common import (
    bench_chain,
    check,
    main_guard,
    record,
    speedup_row,
)
from glm52_h200.kernels import moe_down_merge as KD
from glm52_h200.kernels.moe_down_merge import (
    launch_down,
    launch_down_token_major,
    launch_moe_sum,
    launch_resadd,
    launch_seed,
)

RESULT_ID = "f08f09_down_merge_resadd"
H = C.HIDDEN_SIZE  # 6144  -> GEMM N
I = C.MOE_INTERMEDIATE_SIZE  # 2048  -> GEMM K
E = C.N_ROUTED_EXPERTS  # 256
TOPK = C.NUM_EXPERTS_PER_TOK  # 8
UNITS = ["f8_atomic", "f8_token_major", "f9_atomic", "f9_token_major"]

_ENV = C.env()
SMEM_LIMIT = B.env_int(_ENV, "smem_bytes")
WARP = B.env_int(_ENV, "warp_size")
MAX_THREADS = B.max_threads_per_block(_ENV)
WARPS = B.warp_ladder(_ENV, lo=2)
#: fp32 accumulator elements per lane.  ACC_HI comes from the 256-entry per-thread register
#: window (128); the old literal 64 excluded `BM128 BN256 num_warps=8`, which is exactly the
#: mapping the H200 preflight measured at 96 % of cuBLAS.
ACC_HI, ACC_LO = B.MAX_ACC_ELEMS_PER_THREAD, 4
#: Tile ladders derived from THIS device's opt-in SMEM ceiling: [16..256] on the H200.
TILES = B.tile_ladder(_ENV)
BKS = B.bk_ladder(_ENV, hi=128)
#: Coarse-grid trial budget AFTER the sm_90 overlays multiply the space.
COARSE_CAP = 200


# --------------------------------------------------------------------------------------
# Config-space generation.  The two down-GEMM variants (FUSE_MERGE off/on) get the
# IDENTICAL generator -- the atomic epilogue changes neither SMEM nor accumulator count, so
# there is no legitimate reason for their grids to differ by even one config.
# --------------------------------------------------------------------------------------
def _gemm_ok(cfg: dict, acc_lo: float = ACC_LO, acc_hi: float = ACC_HI) -> bool:
    bm, bn, bk = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"]
    w, s = cfg["num_warps"], cfg["num_stages"]
    # `B.smem_predict` fits the multi-buffer count to the preflight's own smem_probe
    # observations rather than assuming Triton 3.0's or sm_89's.
    if B.smem_predict(bm, bn, bk, s) > SMEM_LIMIT:
        return False
    threads = w * WARP
    if threads > MAX_THREADS:
        return False
    acc_per_lane = bm * bn / threads
    return acc_lo <= acc_per_lane <= acc_hi


def gemm_grid(big: bool) -> list[dict]:
    """Coarse grid for the down GEMM (N=6144, K=2048).

    Prefiltered by the PROBED shared-memory ceiling and by fp32-accumulator registers per
    lane (`BLOCK_M*BLOCK_N/(num_warps*warp_size)` -- the warp width is the probe's, which
    is the single most-repeated hardcode in the C500 suite).  The `big` variant is a strict
    SUBSET of the small one, so prefill regimes reuse the decode compile cache.
    """
    if big:
        bms = [t for t in TILES if t >= 32]
        bns = [t for t in TILES if t >= 64]
    else:
        bms = [t for t in TILES if t <= 128]
        bns = [t for t in TILES if t >= 32]
    out = []
    for bm, bn, bk, w, s in itertools.product(bms, bns, BKS, WARPS, [2, 3, 4]):
        cfg = dict(
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, GROUP_M=8
        )
        if _gemm_ok(cfg):
            out.append(cfg)
    # One generator, one widened list, handed to BOTH the FUSE_MERGE=off and FUSE_MERGE=on
    # searches -- so the sm_90 axes reach both arms by construction, not by convention.
    return B.widen(out, KD, cap=COARSE_CAP,
                   tag=f"f08f09/{'big' if big else 'small'}")


def gemm_refine(best: dict) -> list[dict]:
    def nb(v, lo, hi):
        return sorted({max(lo, v // 2), v, min(hi, v * 2)})

    overlay = {kk: vv for kk, vv in best.items()
               if kk in ("USE_TMA", "TMA_A", "TMA_B", "TMA_MODE", "WARP_SPECIALIZE",
                         "warp_specialize", "num_consumer_groups",
                         "num_buffers_warp_spec", "num_ctas")}
    tile_hi = TILES[-1]
    cands = []
    for bm in nb(best["BLOCK_M"], TILES[0], tile_hi):
        for bn in nb(best["BLOCK_N"], 32, tile_hi):
            for w in nb(best["num_warps"], 1, WARPS[-1]):
                cands.append((bm, bn, best["BLOCK_K"], w, best["num_stages"], 8))
    for bk in nb(best["BLOCK_K"], BKS[0], BKS[-1]):
        for s in (2, 3, 4, 5):
            cands.append((best["BLOCK_M"], best["BLOCK_N"], bk, best["num_warps"], s, 8))
    for g in (1, 4, 8, 16, 32):
        cands.append((best["BLOCK_M"], best["BLOCK_N"], best["BLOCK_K"],
                      best["num_warps"], best["num_stages"], g))
    out, seen = [], set()
    for bm, bn, bk, w, s, g in cands:
        cfg = dict(
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, GROUP_M=g,
            **overlay,
        )
        key = tuple(sorted((kk, str(vv)) for kk, vv in cfg.items()))
        if key in seen or not _gemm_ok(cfg, acc_lo=2, acc_hi=ACC_HI):
            continue
        seen.add(key)
        out.append(cfg)
    return out


def _tokmaj_ok(cfg: dict, acc_lo: float, acc_hi: float) -> bool:
    bn, bk, w, s, ud = (cfg["BLOCK_N"], cfg["BLOCK_K"], cfg["num_warps"],
                        cfg["num_stages"], cfg["USE_DOT"])
    # token-major stages only the B tile (A is a single row / a masked M=16 tile), which is
    # the staging model with BLOCK_M = 0.
    if B.smem_predict(0, bn, bk, s) > SMEM_LIMIT:
        return False
    threads = w * WARP
    if threads > MAX_THREADS:
        return False
    # fp32 register accumulator per lane: [BK, BN] for the GEMV path, [16, BN] for dot
    acc = (bk * bn if not ud else 16 * bn) / threads
    return acc_lo <= acc <= acc_hi


def tokmaj_grid() -> list[dict]:
    out = []
    for bn, bk, w, s, ud in itertools.product(
        [64, 128, 256, 512], [32, 64, 128, 256], WARPS, [1, 2, 3], [False, True]
    ):
        cfg = dict(
            BLOCK_M=16, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, USE_DOT=ud
        )
        if _tokmaj_ok(cfg, 8, 64):
            out.append(cfg)
    return out


def tokmaj_refine(best: dict) -> list[dict]:
    def nb(v, lo, hi):
        return sorted({max(lo, v // 2), v, min(hi, v * 2)})

    cands = []
    for bn in nb(best["BLOCK_N"], 32, 1024):
        for bk in nb(best["BLOCK_K"], 32, 512):
            for w in nb(best["num_warps"], 1, WARPS[-1]):
                cands.append((bn, bk, w, best["num_stages"], best["USE_DOT"]))
    for s in (1, 2, 3, 4):
        cands.append((best["BLOCK_N"], best["BLOCK_K"], best["num_warps"], s,
                      best["USE_DOT"]))
    cands.append((best["BLOCK_N"], best["BLOCK_K"], best["num_warps"],
                  best["num_stages"], not best["USE_DOT"]))
    out, seen = [], set()
    for bn, bk, w, s, ud in cands:
        cfg = dict(
            BLOCK_M=16, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, USE_DOT=ud
        )
        key = tuple(sorted(cfg.items()))
        if key in seen or not _tokmaj_ok(cfg, 1, 128):
            continue
        seen.add(key)
        out.append(cfg)
    return out


def sum_grid() -> list[dict]:
    out = []
    for bm, bd, w, s in itertools.product(
        [1, 2, 4, 8, 16, 32], [256, 512, 1024, 2048], WARPS, [1, 2]
    ):
        tile = bm * bd
        threads = w * WARP
        if tile > 16384 or tile < 512 or threads > MAX_THREADS:
            continue
        r = tile / threads
        if r < 2 or r > 32:
            continue
        out.append(dict(BLOCK_M=bm, BLOCK_DIM=bd, num_warps=w, num_stages=s))
    return out


def elemwise_grid(small: bool = False) -> list[dict]:
    """Grid shared by `resadd_kernel` and `seed_kernel` (same shape of work).

    `small=True` is used only for the atomic variants' output-seeding kernel, whose cost is
    <2 % of the fused chain and which also competes against a torch memset/copy
    pseudo-config; the FULL grid goes to `resadd_kernel`, which is part of the UNFUSED
    baseline and must not be under-tuned.
    """
    bms = [1, 4, 16, 64] if small else [1, 2, 4, 8, 16, 32, 64]
    bns = [512, 2048] if small else [256, 512, 1024, 2048]
    ws = [w for w in WARPS if w in (2, 8)] if small else WARPS
    out = []
    for bm, bn, w, s in itertools.product(bms, bns, ws, [1, 2]):
        tile = bm * bn
        threads = w * WARP
        if tile > 8192 or tile < 512 or threads > MAX_THREADS:
            continue
        r = tile / threads
        if r < 2 or r > 32:
            continue
        out.append(dict(BLOCK_M=bm, BLOCK_N=bn, num_warps=w, num_stages=s))
    return out


# --------------------------------------------------------------------------------------
class Problem:
    def __init__(self, regime, w2, gate_w, seed=0):
        torch.manual_seed(seed + regime.T)
        dev = "cuda"
        self.regime = regime
        self.T = regime.T
        self.rows = regime.T * TOPK
        self.w2 = w2

        # router -> topk_ids / topk_weights (routed_scaling_factor already folded in by
        # reference.router, so the merge is a plain sum on both sides)
        xr = (torch.randn(self.T, H, device=dev, dtype=torch.float32) * 0.1).to(
            torch.bfloat16
        )
        _, self.topk_weights, self.topk_ids = R.router(xr, gate_w)
        self.topk_weights = self.topk_weights.contiguous()
        self.topk_ids = self.topk_ids.contiguous()
        self.tw_flat = self.topk_weights.view(-1).contiguous()
        del xr

        # A = SwiGLU output [rows, I]; padded like the weights (speculative prologue loads)
        pad = 1 << 18
        abuf = torch.empty(self.rows * I + pad, device=dev, dtype=torch.bfloat16)
        abuf[: self.rows * I].normal_(0, 0.1)
        abuf[self.rows * I:].zero_()
        self._abuf = abuf
        self.a = abuf[: self.rows * I].view(self.rows, I)

        self.residual = (torch.randn(self.T, H, device=dev) * 0.1).to(torch.bfloat16)
        # intermediate_cache3 [T, topk, H] -- only the unfused side materialises this
        self.c3 = torch.zeros(self.rows, H, device=dev, dtype=torch.bfloat16)
        self.c3v = self.c3.view(self.T, TOPK, H)
        self.out_u = torch.zeros(self.T, H, device=dev, dtype=torch.bfloat16)
        self.out_u2 = torch.zeros(self.T, H, device=dev, dtype=torch.bfloat16)
        self.out_f = torch.zeros(self.T, H, device=dev, dtype=torch.bfloat16)
        self.out_t = torch.zeros(self.T, H, device=dev, dtype=torch.bfloat16)
        self.layouts: dict[int, tuple] = {}

    def layout(self, block_m: int):
        if block_m not in self.layouts:
            self.layouts[block_m] = R.moe_align_block_size(self.topk_ids, block_m, E)
        return self.layouts[block_m]

    # --- callable factories ----------------------------------------------------------
    def gemm_fn(self, cfg):
        sti, eids, ntp = self.layout(cfg["BLOCK_M"])
        return lambda: launch_down(
            self.a, self.w2, self.c3, self.tw_flat, sti, eids, ntp,
            self.rows, TOPK, cfg, fuse_merge=False,
        )

    def atomic_fn(self, cfg, out=None):
        sti, eids, ntp = self.layout(cfg["BLOCK_M"])
        o = self.out_f if out is None else out
        return lambda: launch_down(
            self.a, self.w2, o, self.tw_flat, sti, eids, ntp,
            self.rows, TOPK, cfg, fuse_merge=True,
        )

    def sum_fn(self, cfg, add_residual: bool, out=None):
        o = self.out_u if out is None else out
        return lambda: launch_moe_sum(
            self.c3v, o, self.residual, TOPK, cfg, add_residual=add_residual
        )

    def resadd_fn(self, cfg):
        return lambda: launch_resadd(self.out_u, self.residual, self.out_u2, cfg)

    def seed_fn(self, cfg, from_residual: bool, out=None):
        o = self.out_f if out is None else out
        if cfg.get("impl") == "torch":
            if from_residual:
                return lambda: o.copy_(self.residual)
            return lambda: o.zero_()
        return lambda: launch_seed(o, self.residual, cfg, from_residual=from_residual)

    def tokmaj_fn(self, cfg, add_residual: bool):
        return lambda: launch_down_token_major(
            self.a, self.w2, self.out_t, self.topk_weights, self.topk_ids,
            self.residual, TOPK, cfg, add_residual=add_residual,
        )


def reference_tokens(prob: Problem, n_sample: int = 512):
    """fp32 reference on a sampled TOKEN subset (a token's full topk sum is needed, so we
    sample tokens, not rows).  The same tokens judge every variant."""
    T = prob.T
    if T <= n_sample:
        tsel = torch.arange(T, device="cuda")
    else:
        g = torch.Generator(device="cuda").manual_seed(1234)
        tsel = torch.randperm(T, device="cuda", generator=g)[:n_sample].sort().values
    nt = tsel.numel()
    ref = torch.zeros(nt, H, device="cuda", dtype=torch.float32)
    ids = prob.topk_ids.long()[tsel]  # [nt, TOPK]
    wts = prob.topk_weights[tsel]  # [nt, TOPK]
    rowidx = tsel[:, None] * TOPK + torch.arange(TOPK, device="cuda")[None, :]
    flat_ids = ids.reshape(-1)
    flat_row = rowidx.reshape(-1)
    flat_w = wts.reshape(-1)
    flat_dst = torch.arange(nt, device="cuda")[:, None].expand(nt, TOPK).reshape(-1)
    for e in torch.unique(flat_ids).tolist():
        sel = (flat_ids == e).nonzero(as_tuple=True)[0]
        y = prob.a[flat_row[sel]].float() @ prob.w2[e].float().t()
        ref.index_add_(0, flat_dst[sel], y * flat_w[sel][:, None])
    return tsel, ref


def vendor_chains(prob: Problem):
    """Vendor-BLAS reference line: per-expert torch.matmul for the down GEMM, then torch
    `index_add_` for the merge.  A is pre-gathered into expert-sorted order OUTSIDE the
    timed region -- best case for the vendor path."""
    flat = prob.topk_ids.flatten().long()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E)
    starts = torch.cat(
        [torch.zeros(1, dtype=torch.long, device="cuda"), counts.cumsum(0)[:-1]]
    )
    a_sorted = prob.a[order].contiguous()
    w_sorted = prob.tw_flat[order].contiguous()
    tok_sorted = (order // TOPK).int()
    c3s = torch.empty(prob.rows, H, device="cuda", dtype=torch.bfloat16)
    out = torch.zeros(prob.T, H, device="cuda", dtype=torch.bfloat16)
    cs, ss = counts.tolist(), starts.tolist()
    segs = [(e, ss[e], ss[e] + cs[e]) for e in range(E) if cs[e]]
    wt = [prob.w2[e].t() for e, _, _ in segs]  # bf16 views, no copy

    def gemm():
        for (e, s, t), w in zip(segs, wt):
            c3s[s:t] = torch.matmul(a_sorted[s:t], w) * w_sorted[s:t, None].to(
                torch.bfloat16
            )

    def merge():
        out.zero_()
        out.index_add_(0, tok_sorted, c3s)

    return [gemm], [gemm, merge]


def make_w2():
    """w2 [E, H, I] bf16 with an sglang-style trailing pad.

    Triton's pipeliner issues speculative (unpredicated) B-tile loads for the peeled
    prologue/epilogue stages, so the last expert's tile can be fetched one BLOCK_K past the
    end of the tensor.  Allocated identically for every variant; never read into an
    accumulator, so it changes no arithmetic.
    """
    numel = E * H * I
    pad = 1 << 20
    buf = torch.empty(numel + pad, device="cuda", dtype=torch.bfloat16)
    w2 = buf[:numel].view(E, H, I)
    for e in range(E):
        w2[e].normal_(0, 0.02)
    buf[numel:].zero_()
    return buf, w2


# --------------------------------------------------------------------------------------
def run_regime(regime, w2, gate_w, quick: bool, units: list[str],
               fair: B.Fairness) -> tuple[list, dict]:
    big = regime.T >= 2048
    w_t, r_t, w_f, r_f = B.reps(regime.T, quick)

    with torch.no_grad():
        print(f"\n===== {regime.name} (T={regime.T}, rows={regime.T * TOPK}) =====",
              flush=True)
        prob = Problem(regime, w2, gate_w)
        tuning: dict = {}
        tsel, ref = reference_tokens(prob)
        ref_res = ref + prob.residual[tsel].float()

        # ---- verifiers.  Every config is checked against the fp32 reference before it is
        # allowed to compete on time; a fast wrong config would otherwise become the
        # reported winner and nobody here can re-run it.
        def v_gemm():
            got = prob.c3v[tsel].float().sum(1)
            c = check(got, ref, label="unfused_gemm")
            return c["ok"], f"rel_err={c['rel_err']:.2e}"

        def v_atomic_zero():
            c = check(prob.out_f[tsel], ref, label="f8_atomic")
            return c["ok"], f"rel_err={c['rel_err']:.2e}"

        def v_tokmaj8():
            c = check(prob.out_t[tsel], ref, label="f8_token_major")
            return c["ok"], f"rel_err={c['rel_err']:.2e}"

        def v_tokmaj9():
            c = check(prob.out_t[tsel], ref_res, label="f9_token_major")
            return c["ok"], f"rel_err={c['rel_err']:.2e}"

        def v_sum():
            c = check(prob.out_u[tsel], ref, label="moe_sum")
            return c["ok"], f"rel_err={c['rel_err']:.2e}"

        def v_sum_res():
            c = check(prob.out_u2[tsel], ref_res, label="moe_sum+res")
            return c["ok"], f"rel_err={c['rel_err']:.2e}"

        def v_resadd():
            c = check(prob.out_u2[tsel], ref_res, label="resadd")
            return c["ok"], f"rel_err={c['rel_err']:.2e}"

        def v_seed_zero():
            return bool(prob.out_f.abs().max().item() == 0.0), "seed!=0"

        def v_seed_res():
            c = check(prob.out_f, prob.residual.float(), label="seed_res")
            return c["ok"], f"rel_err={c['rel_err']:.2e}"

        # ============================= UNFUSED down GEMM ==============================
        cg = gemm_grid(big)
        if quick:
            cg = B.quick_slice(cg, 12)
        print(f"  [unfused GEMM] coarse {len(cg)} cfgs", flush=True)
        tu_c = B.screened_autotune(
            "unfusedGEMM/coarse", lambda c: [prob.gemm_fn(c)], cg, v_gemm, w_t, r_t
        )
        rg = gemm_refine(tu_c.best_cfg)
        if quick:
            rg = B.quick_slice(rg, 8)
        tu_r = B.screened_autotune(
            "unfusedGEMM/refine", lambda c: [prob.gemm_fn(c)], rg, v_gemm, w_t, r_t
        )
        fair.add(regime.name, "unfused_gemm", "coarse", tu_c, grid=cg)
        fair.add(regime.name, "unfused_gemm", "refine", tu_r, grid=rg)

        # ============================= FUSED-atomic down GEMM =========================
        # IDENTICAL grid generator, tuned from scratch, with the seed inside the screen so
        # a config is judged on the value the chain actually produces.
        cga = gemm_grid(big)
        if quick:
            cga = B.quick_slice(cga, 12)
        print(f"  [atomic GEMM] coarse {len(cga)} cfgs", flush=True)
        ta_c = B.screened_autotune(
            "atomicGEMM/coarse",
            lambda c: [prob.seed_fn({"impl": "torch"}, False), prob.atomic_fn(c)],
            cga, v_atomic_zero, w_t, r_t,
        )
        rga = gemm_refine(ta_c.best_cfg)
        if quick:
            rga = B.quick_slice(rga, 8)
        ta_r = B.screened_autotune(
            "atomicGEMM/refine",
            lambda c: [prob.seed_fn({"impl": "torch"}, False), prob.atomic_fn(c)],
            rga, v_atomic_zero, w_t, r_t,
        )
        fair.add(regime.name, "atomic_gemm", "coarse", ta_c, grid=cga)
        fair.add(regime.name, "atomic_gemm", "refine", ta_r, grid=rga)

        # ============================= merge / resadd / seed ==========================
        prob.gemm_fn(tu_c.best_cfg)()  # a valid c3 for the merge screens
        torch.cuda.synchronize()
        sg = sum_grid()
        if quick:
            sg = B.quick_slice(sg, 10)
        print(f"  [moe_sum] {len(sg)} cfgs", flush=True)
        ts0 = B.screened_autotune(
            "moe_sum", lambda c: [prob.sum_fn(c, False)], sg, v_sum, w_t, r_t
        )
        # ADD_RESIDUAL=True is one extra load on an otherwise identical mapping; it is
        # searched over the top-10 shortlist of the plain merge instead of the full grid.
        # Both are UNFUSED-side kernels, so this shortlist can only hurt the baseline.
        ts1 = B.screened_autotune(
            "moe_sum+res",
            lambda c: [prob.sum_fn(c, True, out=prob.out_u2)],
            B.top_cfgs(ts0, k=10), v_sum_res, w_t, r_t,
        )
        fair.add(regime.name, "unfused_sum", "tune", ts0)
        fair.add(regime.name, "unfused_sum_res", "tune", ts1)

        eg = elemwise_grid()
        if quick:
            eg = B.quick_slice(eg, 8)
        print(f"  [resadd] {len(eg)} cfgs", flush=True)
        prob.sum_fn(ts0.best_cfg, False)()
        torch.cuda.synchronize()
        tra = B.screened_autotune(
            "resadd", lambda c: [prob.resadd_fn(c)], eg, v_resadd, w_t, r_t
        )
        egt = elemwise_grid(small=True) + [{"impl": "torch"}]
        tz = B.screened_autotune(
            "seed_zero", lambda c: [prob.seed_fn(c, False)], egt, v_seed_zero, w_t, r_t
        )
        tsr = B.screened_autotune(
            "seed_res", lambda c: [prob.seed_fn(c, True)],
            B.top_cfgs(tz, k=6) + [{"impl": "torch"}], v_seed_res, w_t, r_t,
        )
        fair.add(regime.name, "unfused_resadd", "tune", tra)
        fair.add(regime.name, "fused_seed_zero", "tune", tz)
        fair.add(regime.name, "fused_seed_res", "tune", tsr)

        # ============================= token-major ====================================
        tg = tokmaj_grid()
        # Token-major re-reads a full w2[e] per (token, expert) pair -> T * 8 * 25.2 MB of
        # weight traffic.  At prefill that is hundreds of GB per launch, so probe first and
        # shrink the grid rather than burn an hour on a mapping already known to lose.  The
        # shrink is recorded; it under-sells a FUSED arm, never the baseline.
        probe_cfg = dict(BLOCK_M=16, BLOCK_N=128, BLOCK_K=64, num_warps=4,
                         num_stages=2, USE_DOT=False)
        _pf = prob.tokmaj_fn(probe_cfg, False)
        _pf()
        torch.cuda.synchronize()
        _t0 = time.time()
        _pf()
        torch.cuda.synchronize()
        probe_ms = (time.time() - _t0) * 1e3
        print(f"  [token-major] probe {probe_ms:.2f} ms, full grid {len(tg)}", flush=True)
        if probe_ms > 1500.0:
            tg = B.quick_slice(tg, max(4, len(tg) // 4))
            w_tm, r_tm, do_refine = 1, 2, False
        elif probe_ms > 200.0:
            tg = B.quick_slice(tg, max(6, len(tg) // 8))
            w_tm, r_tm, do_refine = 1, 3, False
        elif probe_ms > 20.0:
            tg = B.quick_slice(tg, 30)
            w_tm, r_tm, do_refine = 2, 5, True
        else:
            if quick:
                tg = B.quick_slice(tg, 10)
            w_tm, r_tm, do_refine = w_t, r_t, True
        print(f"    using {len(tg)} cfgs (warmup={w_tm} rep={r_tm})", flush=True)

        tt8_c = B.screened_autotune(
            "tokmaj8/coarse", lambda c: [prob.tokmaj_fn(c, False)], tg, v_tokmaj8,
            w_tm, r_tm, prep=lambda: prob.out_t.zero_(),
        )
        tt8_tables = [tt8_c]
        best8, best8_ms = tt8_c.best_cfg, tt8_c.best_ms
        fair.add(regime.name, "tokmaj8", "coarse", tt8_c, grid=tg)
        if do_refine:
            rtg = tokmaj_refine(tt8_c.best_cfg)
            if quick:
                rtg = B.quick_slice(rtg, 8)
            tt8_r = B.screened_autotune(
                "tokmaj8/refine", lambda c: [prob.tokmaj_fn(c, False)], rtg, v_tokmaj8,
                w_tm, r_tm,
            )
            tt8_tables.append(tt8_r)
            fair.add(regime.name, "tokmaj8", "refine", tt8_r)
            if tt8_r.best_ms < best8_ms:
                best8, best8_ms = tt8_r.best_cfg, tt8_r.best_ms
            tuning["tokmaj8_refine"] = tt8_r.as_dict()
        print(f"    tokmaj#8 best {best8} {best8_ms:.4f} ms", flush=True)

        # #9 token-major = #8 plus one residual load, so its optimum mapping is the same
        # family: searched over the top-8 shortlist of #8 rather than the full grid.  This
        # shortlist can only under-sell the FUSED side, never the baseline.
        tg9 = B.top_cfgs(*tt8_tables, k=8)
        tt9_c = B.screened_autotune(
            "tokmaj9/coarse", lambda c: [prob.tokmaj_fn(c, True)], tg9, v_tokmaj9,
            w_tm, r_tm, prep=lambda: prob.out_t.zero_(),
        )
        best9, best9_ms = tt9_c.best_cfg, tt9_c.best_ms
        fair.add(regime.name, "tokmaj9", "coarse", tt9_c)
        if do_refine and probe_ms <= 20.0:
            rtg9 = tokmaj_refine(tt9_c.best_cfg)
            if quick:
                rtg9 = B.quick_slice(rtg9, 8)
            tt9_r = B.screened_autotune(
                "tokmaj9/refine", lambda c: [prob.tokmaj_fn(c, True)], rtg9, v_tokmaj9,
                w_tm, r_tm,
            )
            fair.add(regime.name, "tokmaj9", "refine", tt9_r)
            if tt9_r.best_ms < best9_ms:
                best9, best9_ms = tt9_r.best_cfg, tt9_r.best_ms
            tuning["tokmaj9_refine"] = tt9_r.as_dict()
        print(f"    tokmaj#9 best {best9} {best9_ms:.4f} ms", flush=True)

        # ============================= joint chain re-times ===========================
        def joint(pairs, label):
            best_ms, best, tab = float("inf"), None, []
            for combo, fns in pairs:
                try:
                    t = bench_chain(fns, w_t, r_t)
                    tab.append((combo, t.p50_ms, None))
                    if t.p50_ms < best_ms:
                        best_ms, best = t.p50_ms, combo
                except Exception as exc:  # noqa: BLE001
                    tab.append((combo, None, str(exc)[:160]))
            fair.add(regime.name, label, "joint", size=len(pairs))
            print(f"    JOINT {label}: {best} {best_ms:.4f} ms", flush=True)
            if best is None:
                raise RuntimeError(f"{label}: every chain combination failed")
            return best, best_ms, tab

        gemm_top = B.top_cfgs(tu_c, tu_r, k=3)
        sum0_top = B.top_cfgs(ts0, k=3)
        sum1_top = B.top_cfgs(ts1, k=3)
        ra_top = B.top_cfgs(tra, k=2)
        atom_top = B.top_cfgs(ta_c, ta_r, k=3)
        z_top = B.top_cfgs(tz, k=2)
        sr_top = B.top_cfgs(tsr, k=2)

        u8_best, u8_ms, u8_tab = joint(
            [({"gemm": g, "sum": s}, [prob.gemm_fn(g), prob.sum_fn(s, False)])
             for g in gemm_top for s in sum0_top], "unfused8")
        u9_best, u9_ms, u9_tab = joint(
            [({"gemm": g, "sum": s, "resadd": r},
              [prob.gemm_fn(g), prob.sum_fn(s, False), prob.resadd_fn(r)])
             for g in gemm_top[:2] for s in sum0_top[:2] for r in ra_top], "unfused9_3k")
        u9b_best, u9b_ms, u9b_tab = joint(
            [({"gemm": g, "sum_res": s},
              [prob.gemm_fn(g), prob.sum_fn(s, True, out=prob.out_u2)])
             for g in gemm_top for s in sum1_top], "unfused9_2k")
        f8a_best, f8a_ms, f8a_tab = joint(
            [({"seed": z, "gemm": g}, [prob.seed_fn(z, False), prob.atomic_fn(g)])
             for g in atom_top for z in z_top], "fused8_atomic")
        f9a_best, f9a_ms, f9a_tab = joint(
            [({"seed": z, "gemm": g}, [prob.seed_fn(z, True), prob.atomic_fn(g)])
             for g in atom_top for z in sr_top], "fused9_atomic")

        # ============================= validate the winners ===========================
        prob.c3.zero_(); prob.out_u.zero_(); prob.out_u2.zero_()
        prob.gemm_fn(u8_best["gemm"])()
        prob.sum_fn(u8_best["sum"], False)()
        torch.cuda.synchronize()
        chk_u8 = check(prob.out_u[tsel], ref, label="unfused8")

        prob.gemm_fn(u9_best["gemm"])()
        prob.sum_fn(u9_best["sum"], False)()
        prob.resadd_fn(u9_best["resadd"])()
        torch.cuda.synchronize()
        chk_u9 = check(prob.out_u2[tsel], ref_res, label="unfused9")

        prob.out_u2.zero_()
        prob.gemm_fn(u9b_best["gemm"])()
        prob.sum_fn(u9b_best["sum_res"], True, out=prob.out_u2)()
        torch.cuda.synchronize()
        chk_u9b = check(prob.out_u2[tsel], ref_res, label="unfused9_2k")

        prob.seed_fn(f8a_best["seed"], False)()
        prob.atomic_fn(f8a_best["gemm"])()
        torch.cuda.synchronize()
        chk_f8a = check(prob.out_f[tsel], ref, label="f8_atomic")

        prob.seed_fn(f9a_best["seed"], True)()
        prob.atomic_fn(f9a_best["gemm"])()
        torch.cuda.synchronize()
        chk_f9a = check(prob.out_f[tsel], ref_res, label="f9_atomic")

        prob.out_t.zero_()
        prob.tokmaj_fn(best8, False)()
        torch.cuda.synchronize()
        chk_f8t = check(prob.out_t[tsel], ref, label="f8_token_major")

        prob.out_t.zero_()
        prob.tokmaj_fn(best9, True)()
        torch.cuda.synchronize()
        chk_f9t = check(prob.out_t[tsel], ref_res, label="f9_token_major")

        for c in (chk_u8, chk_u9, chk_u9b, chk_f8a, chk_f9a, chk_f8t, chk_f9t):
            print(f"    rel_err {c['label']:16s} {c['rel_err']:.3e} ok={c['ok']}",
                  flush=True)
            if not c["ok"]:
                raise RuntimeError(f"validation failed at {regime.name}: {c}")

        # ============================= final timings ==================================
        # Each fused variant is timed INTERLEAVED against its own baseline chain; that is
        # what replaces the C500 suite's separate `--retime` pass (which existed because a
        # single sequential measurement at the end of a long autotune caught a 55 % drift
        # excursion and would have published it).
        w_tm_f, r_tm_f = (1, 3) if probe_ms > 200 else (w_f, r_f)
        u8_chain = [prob.gemm_fn(u8_best["gemm"]), prob.sum_fn(u8_best["sum"], False)]
        u9_chain = [prob.gemm_fn(u9_best["gemm"]), prob.sum_fn(u9_best["sum"], False),
                    prob.resadd_fn(u9_best["resadd"])]
        u9b_chain = [prob.gemm_fn(u9b_best["gemm"]),
                     prob.sum_fn(u9b_best["sum_res"], True, out=prob.out_u2)]

        pairs, tf, tu = {}, {}, {}
        if "f8_atomic" in units:
            tf["f8_atomic"], tu["f8_atomic"], pairs["f8_atomic"] = B.bench_pair(
                [prob.seed_fn(f8a_best["seed"], False), prob.atomic_fn(f8a_best["gemm"])],
                u8_chain, w_f, r_f, label=f"{regime.name}/f8_atomic")
        if "f9_atomic" in units:
            tf["f9_atomic"], tu["f9_atomic"], pairs["f9_atomic"] = B.bench_pair(
                [prob.seed_fn(f9a_best["seed"], True), prob.atomic_fn(f9a_best["gemm"])],
                u9_chain, w_f, r_f, label=f"{regime.name}/f9_atomic")
        if "f8_token_major" in units:
            tf["f8_token_major"], tu["f8_token_major"], pairs["f8_token_major"] = \
                B.bench_pair([prob.tokmaj_fn(best8, False)], u8_chain, w_tm_f, r_tm_f,
                             label=f"{regime.name}/f8_tokmaj")
        if "f9_token_major" in units:
            tf["f9_token_major"], tu["f9_token_major"], pairs["f9_token_major"] = \
                B.bench_pair([prob.tokmaj_fn(best9, True)], u9_chain, w_tm_f, r_tm_f,
                             label=f"{regime.name}/f9_tokmaj")

        t_u9b = bench_chain(u9b_chain, w_f, r_f)
        t_gemm = bench_chain([prob.gemm_fn(u8_best["gemm"])], w_f, r_f)
        t_sum = bench_chain([prob.sum_fn(u8_best["sum"], False)], w_f, r_f)
        t_ra = bench_chain([prob.resadd_fn(u9_best["resadd"])], w_f, r_f)
        t_atom_only = bench_chain([prob.atomic_fn(f8a_best["gemm"])], w_f, r_f)
        t_seed_only = bench_chain([prob.seed_fn(f8a_best["seed"], False)], w_f, r_f)

        vg, vc = vendor_chains(prob)
        t_vg = bench_chain(vg, max(2, w_f // 3), max(5, r_f // 3))
        t_vc = bench_chain(vc, max(2, w_f // 3), max(5, r_f // 3))

        rows_n = prob.rows
        flops = 2.0 * rows_n * H * I
        n_exp = int(torch.unique(prob.topk_ids).numel())
        l2 = B.env_int(_ENV, "l2_bytes")
        accum_bytes = regime.T * H * 2
        common = {
            "T": regime.T,
            "moe_rows": rows_n,
            "gflop": flops / 1e9,
            "distinct_experts": n_exp,
            "vendor_blas_gemm_ms": t_vg.p50_ms,
            "vendor_blas_gemm_merge_ms": t_vc.p50_ms,
            "vendor_gemm_tflops": flops / (t_vg.p50_ms * 1e-3) / 1e12,
            "unfused9_2kernel_ms": t_u9b.p50_ms,
            "unfused_gemm_only_ms": t_gemm.p50_ms,
            "unfused_sum_only_ms": t_sum.p50_ms,
            "unfused_resadd_only_ms": t_ra.p50_ms,
            "atomic_gemm_only_ms": t_atom_only.p50_ms,
            "seed_only_ms": t_seed_only.p50_ms,
            "tokmaj_probe_ms": probe_ms,
            # the #8/#10 crossover hypothesis, in the numbers that test it
            "atomic_accum_bytes": accum_bytes,
            "l2_bytes": l2,
            "atomic_accum_fits_l2": accum_bytes <= l2,
            "traffic_bytes": {
                "A": rows_n * I * 2,
                "w2_min_touched": n_exp * H * I * 2,
                "unfused_c3_write": rows_n * H * 2,
                "unfused_c3_read": rows_n * H * 2,
                "unfused_out_write": regime.T * H * 2,
                "unfused9_resadd_rw": 3 * regime.T * H * 2,
                "fused_atomic_seed": 2 * regime.T * H * 2,
                "fused_atomic_rmw": 2 * rows_n * H * 2,
                "fused_tokmaj_out_write": regime.T * H * 2,
                "tokmaj_w2_reread": regime.T * TOPK * H * I * 2,
            },
        }

        meta = {
            "f8_atomic": (dict(fused_cfg={"seed": f8a_best["seed"],
                                          "gemm": f8a_best["gemm"]},
                               unfused_cfg=u8_best, rel_err=chk_f8a["rel_err"],
                               rel_err_unfused=chk_u8["rel_err"])),
            "f8_token_major": (dict(fused_cfg=best8, unfused_cfg=u8_best,
                                    rel_err=chk_f8t["rel_err"],
                                    rel_err_unfused=chk_u8["rel_err"])),
            "f9_atomic": (dict(fused_cfg={"seed": f9a_best["seed"],
                                          "gemm": f9a_best["gemm"]},
                               unfused_cfg=u9_best, rel_err=chk_f9a["rel_err"],
                               rel_err_unfused=chk_u9["rel_err"])),
            "f9_token_major": (dict(fused_cfg=best9, unfused_cfg=u9_best,
                                    rel_err=chk_f9t["rel_err"],
                                    rel_err_unfused=chk_u9["rel_err"])),
        }
        rows = []
        for v in units:
            extra = dict(common, variant=v, **meta[v])
            extra["paired_speedup"] = pairs[v].get("paired_speedup_p50")
            extra["paired_speedup_trimmed"] = pairs[v].get("paired_speedup_trimmed_mean")
            extra["pair_meta"] = pairs[v]
            extra["tick"] = B.tick_report(tf[v].p50_ms, tu[v].p50_ms)
            if v.startswith("f9"):
                extra["speedup_vs_2kernel"] = t_u9b.p50_ms / tf[v].p50_ms
            if v == "f8_atomic":
                extra["kernel_stats"] = B.kernel_stats(
                    prob.atomic_fn(f8a_best["gemm"]), getattr(KD, "moe_down_kernel", None))
            rows.append(speedup_row(regime.name, tf[v], tu[v], extra=extra))

        tuning.update({
            "unfused_gemm_coarse": tu_c.as_dict(),
            "unfused_gemm_refine": tu_r.as_dict(),
            "atomic_gemm_coarse": ta_c.as_dict(),
            "atomic_gemm_refine": ta_r.as_dict(),
            "moe_sum": ts0.as_dict(),
            "moe_sum_with_residual": ts1.as_dict(),
            "resadd": tra.as_dict(),
            "seed_zero": tz.as_dict(),
            "seed_residual": tsr.as_dict(),
            "tokmaj8_coarse": tt8_c.as_dict(),
            "tokmaj9_coarse": tt9_c.as_dict(),
            "joint_unfused8": u8_tab,
            "joint_unfused9_3k": u9_tab,
            "joint_unfused9_2k": u9b_tab,
            "joint_fused8_atomic": f8a_tab,
            "joint_fused9_atomic": f9a_tab,
        })

        for r in rows:
            print(f"  RESULT {regime.name:14s} {r['variant']:16s} "
                  f"fused {r['fused_ms']:9.4f} unfused {r['unfused_ms']:9.4f} "
                  f"paired {(r.get('paired_speedup') or r['speedup']):6.3f}x "
                  f"rel_err {r['rel_err']:.2e}"
                  + ("  [TICK-LIMITED]" if r["tick"].get("tick_limited") else ""),
                  flush=True)
        print(f"  vendor BLAS: gemm {t_vg.p50_ms:.4f} ms "
              f"({common['vendor_gemm_tflops']:.1f} TF/s) | gemm+merge {t_vc.p50_ms:.4f} ms",
              flush=True)

        del prob
        torch.cuda.empty_cache()
        return rows, tuning


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
    units = B.resolve_units(UNITS, args.only)
    regimes = B.resolve_regimes(C, args.regimes)

    need = E * H * I * 2
    cap = B.mem_guard(need, "w2 [256, 6144, 2048] bf16")
    if not cap["fits"]:
        raise RuntimeError(
            f"w2 needs {need / 2**30:.1f} GB and only {cap['free_bytes'] / 2**30:.1f} GB "
            f"is free; #8/#9 cannot run at exact GLM-5.2 spec on this device as configured."
        )
    torch.manual_seed(0)
    _buf, w2 = make_w2()
    gate_w = torch.randn(E, H, device="cuda", dtype=torch.float32) * 0.02

    fair = B.Fairness(
        one_kernel_source="glm52_h200/kernels/moe_down_merge.py::moe_down_kernel "
                          "(+ moe_down_token_major_kernel for the token-major variant)",
        flags="FUSE_MERGE constexpr turns the down GEMM's epilogue into a tl.atomic_add "
              "into [T,H]; the unfused side is the same kernel with it off plus "
              "moe_sum_kernel (and resadd_kernel for #9)",
        protocol=(
            "Per regime, per side: coarse grid then a neighbourhood refine around that "
            "side's own coarse winner. The unfused and atomic down GEMMs use the IDENTICAL "
            "generator -- the atomic epilogue changes neither SMEM nor accumulator count, "
            "so their grids must not differ by a single config. Every config is validated "
            "against a sampled fp32 reference before it is timed. Every chain is then "
            "re-timed jointly over top-k x top-k combinations, so a separately-tuned "
            "optimum cannot under-sell any side. The seed kernel's grid also contains a "
            "'torch' pseudo-config (tensor.zero_() / .copy_()) so the fused side gets the "
            "vendor memset if that is faster."
        ),
        token_major_note=(
            "token-major re-reads a full w2[e] per (token,expert) pair, so at prefill its "
            "grid is probe-shrunk; that under-sells a FUSED arm, never the baseline, and "
            "the probe time is recorded per regime"
        ),
        final_timing=(
            "each fused variant is timed A/B interleaved against its own baseline chain in "
            "one loop; this replaces the C500 suite's separate --retime pass, which existed "
            "because one sequential end-of-run measurement drifted 55 %"
        ),
        h200_axes=(
            "USE_TMA / warp_specialize / num_ctas come from ONE call to gemm_grid(), whose "
            "widened output is handed to the FUSE_MERGE=off and FUSE_MERGE=on searches "
            "alike, so both arms search them by construction. The TOKEN-MAJOR variant is "
            "the one place an axis is structurally one-sided: it has no grouped-GEMM "
            "mainloop over a gathered A (one CTA walks a token's 8 experts and sums in "
            "registers), so its own grid is generated separately and carries no sm_90 "
            "overlay at all -- its axis_counts are legitimately zero, and that is a "
            "property of the kernel, not of the search. It is compared against the same "
            "baseline as the atomic variant, which DOES carry them."
        ),
    )
    fair.axis("f08f09_down_merge", B.h200_axis_report(KD))

    rows, tuning, pair_meta = [], {}, None
    for regime in regimes:
        ck = B.ckpt_load(RESULT_ID, regime.name, env, force=args.force)
        if ck is not None:
            print(f"  == {regime.name} == (from checkpoint)", flush=True)
            rows.extend(ck["rows"])
            tuning[regime.name] = ck["tuning"]
            fair.grids.update(ck.get("fairness_grids", {}))
            continue
        try:
            rr, tun = run_regime(regime, w2, gate_w, args.quick, units, fair)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            tuning[regime.name] = {"regime_failed": f"{type(exc).__name__}: {exc}"[:300]}
            torch.cuda.empty_cache()
            continue
        pair_meta = rr[0].get("pair_meta") if rr else None
        B.ckpt_save(RESULT_ID, regime.name, env, {
            "rows": rr, "tuning": tun,
            "fairness_grids": {regime.name: fair.grids.get(regime.name, {})},
        })
        rows.extend(rr)
        tuning[regime.name] = tun

    payload = {
        "id": RESULT_ID,
        "fusion": "#8 down GEMM + expert merge; #9 + residual add 2",
        "shapes": {"H": H, "I": I, "E": E, "topk": TOPK, "gemm_N": H, "gemm_K": I},
        "env": env.__dict__,
        "capacity": cap,
        "variants": {
            "f8_atomic": "seed out[T,H] with zeros + down GEMM with tl.atomic_add merge",
            "f8_token_major": "one CTA per (token, n-block), topk summed in registers",
            "f9_atomic": "seed out[T,H] with the residual + same atomic GEMM",
            "f9_token_major": "token-major + residual load in the epilogue",
        },
        "baselines": {
            "unfused8": "down GEMM -> [T,8,H] then moe_sum_kernel (== sglang 0.5.10)",
            "unfused9_3kernel": "unfused8 + a separate resadd kernel (primary #9 baseline)",
            "unfused9_2kernel": "down GEMM + moe_sum_kernel(ADD_RESIDUAL=True) (bonus)",
        },
        "fairness": fair.render(env, pair_meta),
        "rows": rows,
        "tuning": tuning,
    }
    p = record(RESULT_ID, payload)
    print(f"\nwrote {p}", flush=True)
    print(f"{'regime':15s} {'variant':16s} {'fused':>10s} {'unfused':>10s} "
          f"{'paired':>8s} {'vendor':>10s} {'rel_err':>10s}")
    for r in rows:
        print(f"{r['regime']:15s} {r['variant']:16s} {r['fused_ms']:10.4f} "
              f"{r['unfused_ms']:10.4f} "
              f"{(r.get('paired_speedup') or r['speedup']):8.3f} "
              f"{r['vendor_blas_gemm_merge_ms']:10.4f} {r['rel_err']:10.2e}")


if __name__ == "__main__":
    main_guard(main)
