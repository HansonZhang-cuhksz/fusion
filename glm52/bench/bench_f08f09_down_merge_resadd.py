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
  UNFUSED #9b : down GEMM + `moe_sum_kernel(ADD_RESIDUAL=True)` (2-kernel baseline; a
                cheap production improvement that costs nothing, reported as a check that
                the #9 win is not just "we compared against a needlessly bad baseline").

  FUSED #8 (a) atomic       : seed out[T,6144] with zeros, then the SAME down GEMM with
                              FUSE_MERGE=True atomically accumulating into out.  The
                              seeding kernel IS inside the timed chain.
  FUSED #8 (b) token-major  : one CTA per (token, n-block), loops over that token's 8
                              experts, sums in registers, one store.  No atomics, no
                              [T, 8, 6144] tensor at all.
  FUSED #9 (a) atomic       : identical, except the seed kernel writes the residual
                              instead of zeros.
  FUSED #9 (b) token-major  : identical, plus one residual load in the epilogue.

Tuning: every side is autotuned independently with the SAME grid-generation rules
(coarse -> neighbourhood refine), and the multi-kernel chains get a joint top-k x top-k
re-time so a separately-tuned optimum cannot under-sell them.

Run:
  CUDA_VISIBLE_DEVICES=0 /home/zhangshuhan/my-envs/fusion/bin/python \
      glm52/bench/bench_f08f09_down_merge_resadd.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from glm52 import config as C  # noqa: E402
from glm52 import reference as R  # noqa: E402
from glm52.common import (  # noqa: E402
    autotune,
    bench_chain,
    check,
    main_guard,
    record,
    speedup_row,
)
from glm52.kernels.moe_down_merge import (  # noqa: E402
    launch_down,
    launch_down_token_major,
    launch_moe_sum,
    launch_resadd,
    launch_seed,
    smem_bytes,
    smem_bytes_tokmaj,
)

RESULT_ID = "f08f09_down_merge_resadd"
SMEM_LIMIT = 65536
H = C.HIDDEN_SIZE             # 6144  -> GEMM N
I = C.MOE_INTERMEDIATE_SIZE   # 2048  -> GEMM K
E = C.N_ROUTED_EXPERTS        # 256
TOPK = C.NUM_EXPERTS_PER_TOK  # 8

REGIMES = [r for r in C.DECODE_REGIMES if r.T in (1, 32, 256)] + [
    r for r in C.PREFILL_REGIMES if r.T in (2048, 8192)
]


# --------------------------------------------------------------------------------------
# Config-space generation.  The two down-GEMM variants (FUSE_MERGE off/on) get the
# IDENTICAL generator -- the atomic epilogue changes neither SMEM nor accumulator count.
# --------------------------------------------------------------------------------------
def gemm_grid(big: bool) -> list[dict]:
    """Coarse grid for the down GEMM (N=6144, K=2048).

    Prefiltered by the C500 SMEM ceiling and by fp32-accumulator registers per lane
    (warp = 64 lanes here, so `BLOCK_M*BLOCK_N/(num_warps*64)`).  The `big` variant is a
    strict SUBSET of the small one, so prefill regimes reuse the decode compile cache.
    """
    if big:
        bms, bns, bks = [32, 64, 128], [64, 128, 256], [32, 64, 128]
    else:
        bms, bns, bks = [16, 32, 64, 128], [32, 64, 128, 256], [32, 64, 128]
    out = []
    for bm, bn, bk, w, s in itertools.product(bms, bns, bks, [4, 8, 16], [2, 3]):
        cfg = dict(
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, GROUP_M=8
        )
        if smem_bytes(cfg) > SMEM_LIMIT:
            continue
        acc_per_lane = bm * bn / (w * 64)
        if acc_per_lane > 64 or acc_per_lane < 4:
            continue
        out.append(cfg)
    return out


def gemm_refine(best: dict) -> list[dict]:
    def nb(v, lo, hi):
        return sorted({max(lo, v // 2), v, min(hi, v * 2)})

    cands = []
    for bm in nb(best["BLOCK_M"], 16, 256):
        for bn in nb(best["BLOCK_N"], 32, 256):
            for w in nb(best["num_warps"], 1, 16):
                cands.append((bm, bn, best["BLOCK_K"], w, best["num_stages"], 8))
    for bk in nb(best["BLOCK_K"], 32, 128):
        for s in (2, 3, 4):
            cands.append((best["BLOCK_M"], best["BLOCK_N"], bk, best["num_warps"], s, 8))
    for g in (1, 4, 8, 16, 32):
        cands.append(
            (
                best["BLOCK_M"],
                best["BLOCK_N"],
                best["BLOCK_K"],
                best["num_warps"],
                best["num_stages"],
                g,
            )
        )
    out, seen = [], set()
    for bm, bn, bk, w, s, g in cands:
        cfg = dict(
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, GROUP_M=g
        )
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        if smem_bytes(cfg) > SMEM_LIMIT:
            continue
        acc_per_lane = bm * bn / (w * 64)
        if acc_per_lane > 128 or acc_per_lane < 2:
            continue
        out.append(cfg)
    return out


def tokmaj_grid() -> list[dict]:
    out = []
    for bn, bk, w, s, ud in itertools.product(
        [64, 128, 256, 512], [32, 64, 128, 256], [2, 4, 8, 16], [1, 2], [False, True]
    ):
        cfg = dict(
            BLOCK_M=16, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, USE_DOT=ud
        )
        if smem_bytes_tokmaj(cfg) > SMEM_LIMIT:
            continue
        # fp32 register accumulator per lane: [BK, BN] for the GEMV path, [16, BN] for dot
        acc = (bk * bn if not ud else 16 * bn) / (w * 64)
        if acc > 64 or acc < 8:
            continue
        out.append(cfg)
    return out


def tokmaj_refine(best: dict) -> list[dict]:
    def nb(v, lo, hi):
        return sorted({max(lo, v // 2), v, min(hi, v * 2)})

    cands = []
    for bn in nb(best["BLOCK_N"], 32, 1024):
        for bk in nb(best["BLOCK_K"], 32, 512):
            for w in nb(best["num_warps"], 1, 16):
                cands.append((bn, bk, w, best["num_stages"], best["USE_DOT"]))
    for s in (1, 2, 3, 4):
        cands.append(
            (best["BLOCK_N"], best["BLOCK_K"], best["num_warps"], s, best["USE_DOT"])
        )
    cands.append(
        (
            best["BLOCK_N"],
            best["BLOCK_K"],
            best["num_warps"],
            best["num_stages"],
            not best["USE_DOT"],
        )
    )
    out, seen = [], set()
    for bn, bk, w, s, ud in cands:
        cfg = dict(
            BLOCK_M=16, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, USE_DOT=ud
        )
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        if smem_bytes_tokmaj(cfg) > SMEM_LIMIT:
            continue
        acc = (bk * bn if not ud else 16 * bn) / (w * 64)
        if acc > 128 or acc < 1:
            continue
        out.append(cfg)
    return out


def sum_grid() -> list[dict]:
    out = []
    for bm, bd, w, s in itertools.product(
        [1, 2, 4, 8, 16, 32], [256, 512, 1024, 2048], [2, 4, 8, 16], [1, 2]
    ):
        tile = bm * bd
        if tile > 16384 or tile < 512:
            continue
        r = tile / (w * 64)
        if r < 2 or r > 32:
            continue
        out.append(dict(BLOCK_M=bm, BLOCK_DIM=bd, num_warps=w, num_stages=s))
    return out


def elemwise_grid(small: bool = False) -> list[dict]:
    """Grid shared by `resadd_kernel` and `seed_kernel` (same shape of work).

    `small=True` is used only for the atomic variants' output-seeding kernel, whose cost
    is <2% of the fused chain and which also competes against a torch memset/copy
    pseudo-config; the full grid goes to `resadd_kernel`, which is part of the UNFUSED
    baseline and must not be under-tuned.
    """
    bms = [1, 4, 16, 64] if small else [1, 2, 4, 8, 16, 32, 64]
    bns = [512, 2048] if small else [256, 512, 1024, 2048]
    ws = [2, 8] if small else [2, 4, 8]
    out = []
    for bm, bn, w, s in itertools.product(bms, bns, ws, [1, 2]):
        tile = bm * bn
        if tile > 8192 or tile < 512:
            continue
        r = tile / (w * 64)
        if r < 2 or r > 32:
            continue
        out.append(dict(BLOCK_M=bm, BLOCK_N=bn, num_warps=w, num_stages=s))
    return out


# --------------------------------------------------------------------------------------
# Per-regime problem setup
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
        abuf[self.rows * I :].zero_()
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


# --------------------------------------------------------------------------------------
# fp32 reference on a sampled TOKEN subset (a token's full topk sum is needed, so we
# sample tokens, not rows).
# --------------------------------------------------------------------------------------
def reference_tokens(prob: Problem, n_sample: int = 512):
    T = prob.T
    if T <= n_sample:
        tsel = torch.arange(T, device="cuda")
    else:
        g = torch.Generator(device="cuda").manual_seed(1234)
        tsel = torch.randperm(T, device="cuda", generator=g)[:n_sample].sort().values
    nt = tsel.numel()
    ref = torch.zeros(nt, H, device="cuda", dtype=torch.float32)
    ids = prob.topk_ids.long()[tsel]           # [nt, TOPK]
    wts = prob.topk_weights[tsel]              # [nt, TOPK]
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


# --------------------------------------------------------------------------------------
# Vendor-BLAS reference line: per-expert torch.matmul (dispatches to the MetaX BLAS) for
# the down GEMM, then torch `index_add_` for the merge.  A is pre-gathered into
# expert-sorted order OUTSIDE the timed region -- best case for the vendor path.
# --------------------------------------------------------------------------------------
def vendor_chains(prob: Problem):
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
    segs = []
    cs, ss = counts.tolist(), starts.tolist()
    for e in range(E):
        if cs[e]:
            segs.append((e, ss[e], ss[e] + cs[e]))
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


# --------------------------------------------------------------------------------------
def top_k_cfgs(*tables, k=3):
    rowsx = [(m, c) for tb in tables for (c, m, err) in tb if m is not None]
    rowsx.sort(key=lambda t: t[0])
    seen, out = set(), []
    for m, c in rowsx:
        key = tuple(sorted(c.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) == k:
            break
    return out


def kernel_stats(fn):
    try:
        k = fn()
        return {
            "n_regs": getattr(k, "n_regs", None),
            "n_spills": getattr(k, "n_spills", None),
            "shared_bytes": getattr(getattr(k, "metadata", None), "shared", None),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:160]}


def make_w2():
    """w2 [E, H, I] bf16 with an sglang-style trailing pad.

    The MACA pipeliner issues speculative (unpredicated) B-tile loads for the peeled
    prologue/epilogue stages, so the last expert's tile can be fetched one BLOCK_K past
    the end of the tensor -- an ATU fault without the pad.  Allocated identically for
    every variant; never read into an accumulator, so it changes no arithmetic.
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
def run_regime(regime, w2, gate_w, quick: bool) -> tuple[list, dict]:
    big = regime.T >= 2048
    if quick:
        w_t, r_t = 2, 5
    elif regime.T >= 8192:
        w_t, r_t = 3, 8
    elif regime.T >= 2048:
        w_t, r_t = 4, 12
    elif regime.T >= 256:
        w_t, r_t = 8, 20
    else:
        w_t, r_t = 10, 30
    w_f, r_f = (3, 10) if big else (25, 100)

    with torch.no_grad():
        print(f"\n===== {regime.name} (T={regime.T}, rows={regime.T*TOPK}) =====", flush=True)
        prob = Problem(regime, w2, gate_w)
        tuning: dict = {}

        # ============================= UNFUSED down GEMM ==============================
        cg = gemm_grid(big)
        if quick:
            cg = cg[::7]
        print(f"  [unfused GEMM] coarse {len(cg)} cfgs", flush=True)
        tu_c = autotune(lambda c: [prob.gemm_fn(c)], cg, warmup=w_t, rep=r_t)
        rg = gemm_refine(tu_c.best_cfg)
        print(f"    coarse best {tu_c.best_cfg} {tu_c.best_ms:.4f} ms; refine {len(rg)}", flush=True)
        tu_r = autotune(lambda c: [prob.gemm_fn(c)], rg, warmup=w_t, rep=r_t)
        print(f"    refine best {tu_r.best_cfg} {tu_r.best_ms:.4f} ms", flush=True)

        # ============================= FUSED-atomic down GEMM =========================
        # IDENTICAL grid generator, tuned from scratch.
        cga = gemm_grid(big)
        if quick:
            cga = cga[::7]
        print(f"  [atomic GEMM] coarse {len(cga)} cfgs", flush=True)
        ta_c = autotune(lambda c: [prob.atomic_fn(c)], cga, warmup=w_t, rep=r_t)
        rga = gemm_refine(ta_c.best_cfg)
        print(f"    coarse best {ta_c.best_cfg} {ta_c.best_ms:.4f} ms; refine {len(rga)}", flush=True)
        ta_r = autotune(lambda c: [prob.atomic_fn(c)], rga, warmup=w_t, rep=r_t)
        print(f"    refine best {ta_r.best_cfg} {ta_r.best_ms:.4f} ms", flush=True)

        # ============================= merge / resadd / seed ==========================
        sg = sum_grid()
        if quick:
            sg = sg[::7]
        print(f"  [moe_sum] {len(sg)} cfgs", flush=True)
        ts0 = autotune(lambda c: [prob.sum_fn(c, False)], sg, warmup=w_t, rep=r_t)
        # ADD_RESIDUAL=True is one extra load on an otherwise identical mapping; it is
        # searched over the top-10 shortlist of the plain merge instead of the full grid
        # (each fresh config costs a ~4 s MACA compile).  Both are UNFUSED-side kernels,
        # so this shortlist can only ever hurt the unfused baseline, never the fused one.
        ts1 = autotune(
            lambda c: [prob.sum_fn(c, True)], top_k_cfgs(ts0.table, k=10),
            warmup=w_t, rep=r_t,
        )
        print(f"    sum best {ts0.best_cfg} {ts0.best_ms:.4f} ms | "
              f"sum+res best {ts1.best_cfg} {ts1.best_ms:.4f} ms", flush=True)

        eg = elemwise_grid()
        if quick:
            eg = eg[::5]
        print(f"  [resadd] {len(eg)} cfgs", flush=True)
        tra = autotune(lambda c: [prob.resadd_fn(c)], eg, warmup=w_t, rep=r_t)
        egt = elemwise_grid(small=True) + [{"impl": "torch"}]
        tz = autotune(lambda c: [prob.seed_fn(c, False)], egt, warmup=w_t, rep=r_t)
        tsr = autotune(
            lambda c: [prob.seed_fn(c, True)],
            top_k_cfgs(tz.table, k=6) + [{"impl": "torch"}],
            warmup=w_t, rep=r_t,
        )
        print(f"    resadd {tra.best_cfg} {tra.best_ms:.4f} | zero {tz.best_cfg} "
              f"{tz.best_ms:.4f} | seedres {tsr.best_cfg} {tsr.best_ms:.4f}", flush=True)

        # ============================= token-major ====================================
        tg = tokmaj_grid()
        # Token-major re-reads a full w2[e] per (token, expert) pair -> T * 8 * 25.2 MB of
        # weight traffic.  At prefill that is hundreds of GB per launch, so probe first
        # and shrink the grid rather than burn an hour on a mapping we already know loses.
        probe_cfg = dict(BLOCK_M=16, BLOCK_N=128, BLOCK_K=64, num_warps=4,
                         num_stages=2, USE_DOT=False)
        _pf = prob.tokmaj_fn(probe_cfg, False)
        _pf()                       # compile + warm
        torch.cuda.synchronize()
        _t0 = time.time()
        _pf()
        torch.cuda.synchronize()
        probe_ms = (time.time() - _t0) * 1e3

        class _P:  # tiny stand-in so the payload always carries the probe number
            p50_ms = probe_ms
        t_probe = _P()
        print(f"  [token-major] probe {probe_ms:.2f} ms, full grid {len(tg)}", flush=True)
        if probe_ms > 1500.0:
            tg = tg[:: max(1, len(tg) // 4)]
            w_tm, r_tm = 1, 2
            do_refine = False
        elif t_probe.p50_ms > 200.0:
            step = max(1, len(tg) // 8)
            tg = tg[::step]
            w_tm, r_tm = 1, 3
            do_refine = False
        elif t_probe.p50_ms > 20.0:
            step = max(1, len(tg) // 30)
            tg = tg[::step]
            w_tm, r_tm = 2, 5
            do_refine = True
        else:
            if quick:
                tg = tg[::7]
            w_tm, r_tm = w_t, r_t
            do_refine = True
        print(f"    using {len(tg)} cfgs (warmup={w_tm} rep={r_tm})", flush=True)

        tt8_c = autotune(lambda c: [prob.tokmaj_fn(c, False)], tg, warmup=w_tm, rep=r_tm)
        tt8_tables = [tt8_c.table]
        best8, best8_ms = tt8_c.best_cfg, tt8_c.best_ms
        if do_refine:
            rtg = tokmaj_refine(tt8_c.best_cfg)
            if quick:
                rtg = rtg[::3]
            tt8_r = autotune(lambda c: [prob.tokmaj_fn(c, False)], rtg, warmup=w_tm, rep=r_tm)
            tt8_tables.append(tt8_r.table)
            if tt8_r.best_ms < best8_ms:
                best8, best8_ms = tt8_r.best_cfg, tt8_r.best_ms
            tuning["tokmaj8_refine"] = tt8_r.as_dict()
        print(f"    tokmaj#8 best {best8} {best8_ms:.4f} ms", flush=True)

        # #9 token-major = #8 plus one residual load, so its optimum mapping is the same
        # family: searched over the top-8 shortlist of #8 (+ a neighbourhood refine where
        # affordable) rather than the full grid, because ADD_RESIDUAL=True is a distinct
        # constexpr and every fresh config costs a ~4 s MACA compile.  This shortlist can
        # only under-sell the FUSED side, never the baseline.
        tg9 = top_k_cfgs(*tt8_tables, k=8)
        tt9_c = autotune(lambda c: [prob.tokmaj_fn(c, True)], tg9, warmup=w_tm, rep=r_tm)
        best9, best9_ms = tt9_c.best_cfg, tt9_c.best_ms
        if do_refine and t_probe.p50_ms <= 20.0:
            rtg9 = tokmaj_refine(tt9_c.best_cfg)
            if quick:
                rtg9 = rtg9[::3]
            tt9_r = autotune(lambda c: [prob.tokmaj_fn(c, True)], rtg9, warmup=w_tm, rep=r_tm)
            if tt9_r.best_ms < best9_ms:
                best9, best9_ms = tt9_r.best_cfg, tt9_r.best_ms
            tuning["tokmaj9_refine"] = tt9_r.as_dict()
        print(f"    tokmaj#9 best {best9} {best9_ms:.4f} ms", flush=True)

        # ============================= joint chain re-times ===========================
        def joint(pairs, label):
            best_ms, best = float("inf"), None
            tab = []
            for combo, fns in pairs:
                try:
                    t = bench_chain(fns, warmup=w_t, rep=r_t)
                    tab.append((combo, t.p50_ms, None))
                    if t.p50_ms < best_ms:
                        best_ms, best = t.p50_ms, combo
                except Exception as exc:  # noqa: BLE001
                    tab.append((combo, None, str(exc)[:160]))
            print(f"    JOINT {label}: {best} {best_ms:.4f} ms", flush=True)
            return best, best_ms, tab

        gemm_top = top_k_cfgs(tu_c.table, tu_r.table, k=3)
        sum0_top = top_k_cfgs(ts0.table, k=3)
        sum1_top = top_k_cfgs(ts1.table, k=3)
        ra_top = top_k_cfgs(tra.table, k=2)
        atom_top = top_k_cfgs(ta_c.table, ta_r.table, k=3)
        z_top = top_k_cfgs(tz.table, k=2)
        sr_top = top_k_cfgs(tsr.table, k=2)

        u8_best, u8_ms, u8_tab = joint(
            [({"gemm": g, "sum": s}, [prob.gemm_fn(g), prob.sum_fn(s, False)])
             for g in gemm_top for s in sum0_top],
            "unfused#8",
        )
        u9_best, u9_ms, u9_tab = joint(
            [({"gemm": g, "sum": s, "resadd": r},
              [prob.gemm_fn(g), prob.sum_fn(s, False), prob.resadd_fn(r)])
             for g in gemm_top[:2] for s in sum0_top[:2] for r in ra_top],
            "unfused#9(3k)",
        )
        u9b_best, u9b_ms, u9b_tab = joint(
            [({"gemm": g, "sum_res": s}, [prob.gemm_fn(g), prob.sum_fn(s, True, out=prob.out_u2)])
             for g in gemm_top for s in sum1_top],
            "unfused#9(2k)",
        )
        f8a_best, f8a_ms, f8a_tab = joint(
            [({"seed": z, "gemm": g}, [prob.seed_fn(z, False), prob.atomic_fn(g)])
             for g in atom_top for z in z_top],
            "fused#8-atomic",
        )
        f9a_best, f9a_ms, f9a_tab = joint(
            [({"seed": z, "gemm": g}, [prob.seed_fn(z, True), prob.atomic_fn(g)])
             for g in atom_top for z in sr_top],
            "fused#9-atomic",
        )

        # ============================= validate =======================================
        tsel, ref = reference_tokens(prob)
        ref_res = ref + prob.residual[tsel].float()

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
            print(f"    rel_err {c['label']:16s} {c['rel_err']:.3e} ok={c['ok']}", flush=True)
            if not c["ok"]:
                raise RuntimeError(f"validation failed at {regime.name}: {c}")

        # ============================= final timings ==================================
        w_tm_f, r_tm_f = (1, 3) if probe_ms > 200 else (w_f, r_f)
        t_u8 = bench_chain(
            [prob.gemm_fn(u8_best["gemm"]), prob.sum_fn(u8_best["sum"], False)],
            warmup=w_f, rep=r_f)
        t_u9 = bench_chain(
            [prob.gemm_fn(u9_best["gemm"]), prob.sum_fn(u9_best["sum"], False),
             prob.resadd_fn(u9_best["resadd"])], warmup=w_f, rep=r_f)
        t_u9b = bench_chain(
            [prob.gemm_fn(u9b_best["gemm"]),
             prob.sum_fn(u9b_best["sum_res"], True, out=prob.out_u2)], warmup=w_f, rep=r_f)
        t_gemm = bench_chain([prob.gemm_fn(u8_best["gemm"])], warmup=w_f, rep=r_f)
        t_sum = bench_chain([prob.sum_fn(u8_best["sum"], False)], warmup=w_f, rep=r_f)
        t_ra = bench_chain([prob.resadd_fn(u9_best["resadd"])], warmup=w_f, rep=r_f)

        t_f8a = bench_chain(
            [prob.seed_fn(f8a_best["seed"], False), prob.atomic_fn(f8a_best["gemm"])],
            warmup=w_f, rep=r_f)
        t_f9a = bench_chain(
            [prob.seed_fn(f9a_best["seed"], True), prob.atomic_fn(f9a_best["gemm"])],
            warmup=w_f, rep=r_f)
        t_atom_only = bench_chain([prob.atomic_fn(f8a_best["gemm"])], warmup=w_f, rep=r_f)
        t_seed_only = bench_chain([prob.seed_fn(f8a_best["seed"], False)], warmup=w_f, rep=r_f)

        t_f8t = bench_chain([prob.tokmaj_fn(best8, False)], warmup=w_tm_f, rep=r_tm_f)
        t_f9t = bench_chain([prob.tokmaj_fn(best9, True)], warmup=w_tm_f, rep=r_tm_f)

        vg, vc = vendor_chains(prob)
        t_vg = bench_chain(vg, warmup=max(2, w_f // 3), rep=max(5, r_f // 3))
        t_vc = bench_chain(vc, warmup=max(2, w_f // 3), rep=max(5, r_f // 3))

        rows_n = prob.rows
        flops = 2.0 * rows_n * H * I
        n_exp = int(torch.unique(prob.topk_ids).numel())
        common = {
            "T": regime.T,
            "moe_rows": rows_n,
            "gflop": flops / 1e9,
            "distinct_experts": n_exp,
            "vendor_blas_gemm_ms": t_vg.p50_ms,
            "vendor_blas_gemm_merge_ms": t_vc.p50_ms,
            "vendor_gemm_tflops": flops / (t_vg.p50_ms * 1e-3) / 1e12,
            "unfused8_ms": t_u8.p50_ms,
            "unfused9_3kernel_ms": t_u9.p50_ms,
            "unfused9_2kernel_ms": t_u9b.p50_ms,
            "unfused_gemm_only_ms": t_gemm.p50_ms,
            "unfused_sum_only_ms": t_sum.p50_ms,
            "unfused_resadd_only_ms": t_ra.p50_ms,
            "atomic_gemm_only_ms": t_atom_only.p50_ms,
            "seed_only_ms": t_seed_only.p50_ms,
            "tokmaj_probe_ms": t_probe.p50_ms,
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

        rows = []
        rows.append(speedup_row(regime.name, t_f8a, t_u8, extra=dict(
            common, variant="f8_atomic",
            fused_cfg={"seed": f8a_best["seed"], "gemm": f8a_best["gemm"]},
            unfused_cfg=u8_best, rel_err=chk_f8a["rel_err"],
            rel_err_unfused=chk_u8["rel_err"],
            fused_noflush_ms=t_f8a.noflush_p50_ms, unfused_noflush_ms=t_u8.noflush_p50_ms,
            kernel_stats=kernel_stats(prob.atomic_fn(f8a_best["gemm"])))))
        rows.append(speedup_row(regime.name, t_f8t, t_u8, extra=dict(
            common, variant="f8_token_major", fused_cfg=best8, unfused_cfg=u8_best,
            rel_err=chk_f8t["rel_err"], rel_err_unfused=chk_u8["rel_err"],
            fused_noflush_ms=t_f8t.noflush_p50_ms, unfused_noflush_ms=t_u8.noflush_p50_ms,
            kernel_stats=kernel_stats(prob.tokmaj_fn(best8, False)))))
        rows.append(speedup_row(regime.name, t_f9a, t_u9, extra=dict(
            common, variant="f9_atomic",
            fused_cfg={"seed": f9a_best["seed"], "gemm": f9a_best["gemm"]},
            unfused_cfg=u9_best, rel_err=chk_f9a["rel_err"],
            rel_err_unfused=chk_u9["rel_err"],
            speedup_vs_2kernel=t_u9b.p50_ms / t_f9a.p50_ms,
            fused_noflush_ms=t_f9a.noflush_p50_ms, unfused_noflush_ms=t_u9.noflush_p50_ms)))
        rows.append(speedup_row(regime.name, t_f9t, t_u9, extra=dict(
            common, variant="f9_token_major", fused_cfg=best9, unfused_cfg=u9_best,
            rel_err=chk_f9t["rel_err"], rel_err_unfused=chk_u9["rel_err"],
            speedup_vs_2kernel=t_u9b.p50_ms / t_f9t.p50_ms,
            fused_noflush_ms=t_f9t.noflush_p50_ms, unfused_noflush_ms=t_u9.noflush_p50_ms)))

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
            print(f"  RESULT {regime.name:14s} {r['variant']:16s} fused {r['fused_ms']:9.4f} "
                  f"unfused {r['unfused_ms']:9.4f} speedup {r['speedup']:6.3f}x "
                  f"rel_err {r['rel_err']:.2e}", flush=True)
        print(f"  vendor BLAS: gemm {t_vg.p50_ms:.4f} ms ({common['vendor_gemm_tflops']:.1f} TF/s)"
              f" | gemm+merge {t_vc.p50_ms:.4f} ms", flush=True)

        del prob
        torch.cuda.empty_cache()
        return rows, tuning


# --------------------------------------------------------------------------------------
# Re-timing pass.  The tuning pass measures every chain exactly once at the end of a long
# autotune; a single perturbed measurement (we caught one: decode_bs32's 3-kernel #9 chain
# came out 55% slower in the final block than the identical config measured minutes earlier
# in the joint search) would silently become a headline speedup.  `--retime` replays ONLY
# the winning configs from results/<id>.json, timing every chain `REPEATS` times in an
# interleaved round-robin and keeping the min of the medians.  Interleaving means any drift
# (clocks, power) hits every chain equally; the min-of-medians rejects one-off excursions.
# Applied identically to fused and unfused chains.
# --------------------------------------------------------------------------------------
RETIME_REPEATS = 3


def retime_regime(regime, w2, gate_w, rows_in: list, tuning_in: dict) -> list:
    prob = Problem(regime, w2, gate_w)
    by_var = {r["variant"]: r for r in rows_in}
    u8 = by_var["f8_atomic"]["unfused_cfg"]          # {"gemm":..., "sum":...}
    u9 = by_var["f9_atomic"]["unfused_cfg"]          # {"gemm":..., "sum":..., "resadd":...}
    a8 = by_var["f8_atomic"]["fused_cfg"]            # {"seed":..., "gemm":...}
    a9 = by_var["f9_atomic"]["fused_cfg"]
    t8 = by_var["f8_token_major"]["fused_cfg"]
    t9 = by_var["f9_token_major"]["fused_cfg"]

    u9b = None
    for combo, ms, err in tuning_in.get("joint_unfused9_2k", []):
        if ms is not None and (u9b is None or ms < u9b[1]):
            u9b = (combo, ms)
    probe_ms = by_var["f8_token_major"].get("tokmaj_probe_ms", 0.0)

    chains = {
        "unfused8": [prob.gemm_fn(u8["gemm"]), prob.sum_fn(u8["sum"], False)],
        "unfused9_3kernel": [prob.gemm_fn(u9["gemm"]), prob.sum_fn(u9["sum"], False),
                             prob.resadd_fn(u9["resadd"])],
        "f8_atomic": [prob.seed_fn(a8["seed"], False), prob.atomic_fn(a8["gemm"])],
        "f9_atomic": [prob.seed_fn(a9["seed"], True), prob.atomic_fn(a9["gemm"])],
        "f8_token_major": [prob.tokmaj_fn(t8, False)],
        "f9_token_major": [prob.tokmaj_fn(t9, True)],
    }
    if u9b is not None:
        chains["unfused9_2kernel"] = [
            prob.gemm_fn(u9b[0]["gemm"]),
            prob.sum_fn(u9b[0]["sum_res"], True, out=prob.out_u2),
        ]

    slow = probe_ms > 200.0
    w_n, r_n = (3, 10) if regime.T >= 2048 else (25, 100)
    w_s, r_s = (1, 3)
    best: dict[str, float] = {}
    for it in range(RETIME_REPEATS):
        for name, fns in chains.items():
            is_tok = "token_major" in name
            w, r = (w_s, r_s) if (is_tok and slow) else (w_n, r_n)
            if is_tok and slow and it > 0:
                continue  # a single ~250-1500 ms launch x 7; one pass is enough
            t = bench_chain(fns, warmup=w, rep=r)
            best[name] = min(best.get(name, float("inf")), t.p50_ms)
        print(f"    retime pass {it}: "
              + " ".join(f"{k}={v:.4f}" for k, v in best.items()), flush=True)

    rows_out = []
    for r in rows_in:
        v = r["variant"]
        base = "unfused8" if v.startswith("f8") else "unfused9_3kernel"
        o = dict(r)
        o["first_pass_fused_ms"] = r["fused_ms"]
        o["first_pass_unfused_ms"] = r["unfused_ms"]
        o["fused_ms"] = best[v]
        o["unfused_ms"] = best[base]
        o["speedup"] = best[base] / best[v]
        o["retimed"] = True
        o["retime_repeats"] = RETIME_REPEATS
        o["unfused8_ms"] = best["unfused8"]
        o["unfused9_3kernel_ms"] = best["unfused9_3kernel"]
        if "unfused9_2kernel" in best:
            o["unfused9_2kernel_ms"] = best["unfused9_2kernel"]
            if v.startswith("f9"):
                o["speedup_vs_2kernel"] = best["unfused9_2kernel"] / best[v]
        rows_out.append(o)
        print(f"  RETIMED {regime.name:14s} {v:16s} fused {best[v]:9.4f} "
              f"unfused {best[base]:9.4f} speedup {o['speedup']:6.3f}x", flush=True)
    del prob
    torch.cuda.empty_cache()
    return rows_out


def retime_worker(regime_name: str, out_path: str):
    regime = next(r for r in REGIMES if r.name == regime_name)
    payload = json.loads((Path(__file__).resolve().parents[2] / "results"
                          / f"{RESULT_ID}.json").read_text())
    rows_in = [r for r in payload["rows"] if r["regime"] == regime_name]
    tuning_in = payload["tuning"].get(regime_name, {})
    torch.manual_seed(0)
    _buf, w2 = make_w2()
    gate_w = torch.randn(E, H, device="cuda", dtype=torch.float32) * 0.02
    with torch.no_grad():
        print(f"\n===== RETIME {regime_name} =====", flush=True)
        rows = retime_regime(regime, w2, gate_w, rows_in, tuning_in)
    Path(out_path).write_text(json.dumps({"rows": rows}, default=str))


def retime_driver(only: list[str] | None):
    import subprocess

    res = Path(__file__).resolve().parents[2] / "results" / f"{RESULT_ID}.json"
    payload = json.loads(res.read_text())
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"{RESULT_ID}_retime"
    tmp.mkdir(parents=True, exist_ok=True)
    new_rows = {r["regime"]: [] for r in payload["rows"]}
    for regime in REGIMES:
        if only and regime.name not in only:
            continue
        if not any(r["regime"] == regime.name for r in payload["rows"]):
            continue
        part = tmp / f"{regime.name}.json"
        if part.exists():
            part.unlink()
        rc = subprocess.call([sys.executable, "-u", str(Path(__file__).resolve()),
                              "--retime-worker", regime.name, "--out", str(part)])
        if rc != 0 or not part.exists():
            print(f"!! retime {regime.name} failed (rc={rc}) -- keeping first-pass rows",
                  flush=True)
            continue
        new_rows[regime.name] = json.loads(part.read_text())["rows"]
    out = []
    for r in payload["rows"]:
        repl = [x for x in new_rows.get(r["regime"], [])
                if x["variant"] == r["variant"]]
        out.append(repl[0] if repl else r)
    payload["rows"] = out
    payload["retime"] = (
        f"Final timings replayed with --retime: winning configs only, every chain timed "
        f"{RETIME_REPEATS}x in an interleaved round-robin, min of medians kept. Applied "
        f"identically to fused and unfused chains. First-pass numbers are preserved as "
        f"first_pass_fused_ms / first_pass_unfused_ms. Token-major at regimes whose probe "
        f"exceeded 200 ms is timed once (a single launch there is 0.25-1.5 s)."
    )
    p = record(RESULT_ID, payload)
    print(f"\nwrote {p}", flush=True)


# --------------------------------------------------------------------------------------
# Worker / driver.  One process per regime: the MACA runtime kills the whole context
# after an ATU fault, so a bad launch would otherwise lose the entire run.
# --------------------------------------------------------------------------------------
def worker(regime_name: str, out_path: str, quick: bool):
    regime = next(r for r in REGIMES if r.name == regime_name)
    torch.manual_seed(0)
    print("allocating w2 [%d, %d, %d] bf16 = %.2f GB"
          % (E, H, I, E * H * I * 2 / 2**30), flush=True)
    _buf, w2 = make_w2()
    gate_w = torch.randn(E, H, device="cuda", dtype=torch.float32) * 0.02
    rows, tuning = run_regime(regime, w2, gate_w, quick)
    Path(out_path).write_text(json.dumps({"rows": rows, "tuning": tuning}, default=str))
    print(f"worker wrote {out_path}", flush=True)


def driver(quick: bool, only: list[str] | None):
    import subprocess

    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"{RESULT_ID}_parts"
    tmp.mkdir(parents=True, exist_ok=True)
    env = C.BenchEnv.probe()
    payload = {
        "id": RESULT_ID,
        "fusion": "#8 down GEMM + expert merge; #9 + residual add 2",
        "shapes": {"H": H, "I": I, "E": E, "topk": TOPK, "gemm_N": H, "gemm_K": I},
        "env": env.__dict__,
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
        "tuning_protocol": (
            "Per regime, per side: coarse grid (SMEM- and accumulator-prefiltered; the "
            "unfused and atomic down GEMMs use the IDENTICAL generator) then a "
            "neighbourhood refine around the coarse winner. Every kernel of every chain "
            "is tuned on its own, then the chains are re-timed jointly over the top-k x "
            "top-k combinations and the best chain is reported, so a separately-tuned "
            "optimum cannot under-sell any side. The seed kernel's grid also contains a "
            "'torch' pseudo-config (tensor.zero_() / .copy_()) so the fused side gets the "
            "vendor memset if that is faster. Token-major re-reads a full w2[e] per "
            "(token,expert) pair, so at prefill its grid is probe-shrunk (documented in "
            "the log) -- it loses there by two orders of magnitude, not by mistuning. "
            "Each regime runs in its own process for crash isolation."
        ),
        "rows": [],
        "tuning": {},
        "failed_regimes": {},
    }
    for regime in REGIMES:
        if only and regime.name not in only:
            continue
        part = tmp / f"{regime.name}.json"
        if part.exists():
            part.unlink()
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--worker", regime.name, "--out", str(part)]
        if quick:
            cmd.append("--quick")
        ok = False
        for attempt in range(2):
            rc = subprocess.call(cmd)
            if rc == 0 and part.exists():
                ok = True
                break
            print(f"!! {regime.name} worker failed (rc={rc}), attempt {attempt+1}", flush=True)
        if not ok:
            payload["failed_regimes"][regime.name] = "worker process aborted twice"
            continue
        d = json.loads(part.read_text())
        payload["rows"].extend(d["rows"])
        payload["tuning"][regime.name] = d["tuning"]

    p = record(RESULT_ID, payload)
    print(f"\nwrote {p}", flush=True)
    print(f"{'regime':15s} {'variant':16s} {'fused':>10s} {'unfused':>10s} "
          f"{'speedup':>8s} {'vendor':>10s} {'rel_err':>10s}")
    for r in payload["rows"]:
        print(f"{r['regime']:15s} {r['variant']:16s} {r['fused_ms']:10.4f} "
              f"{r['unfused_ms']:10.4f} {r['speedup']:8.3f} "
              f"{r['vendor_blas_gemm_merge_ms']:10.4f} {r['rel_err']:10.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--worker", default=None, help="internal: run one regime")
    ap.add_argument("--retime-worker", default=None, help="internal: retime one regime")
    ap.add_argument("--out", default=None, help="internal: worker output json")
    ap.add_argument("--only", default=None, help="comma-separated regime names")
    ap.add_argument("--retime", action="store_true",
                    help="replay the winning configs from results/<id>.json and replace "
                         "the reported timings with an interleaved min-of-medians")
    a = ap.parse_args()
    only = a.only.split(",") if a.only else None
    if a.worker:
        worker(a.worker, a.out, a.quick)
    elif a.retime_worker:
        retime_worker(a.retime_worker, a.out)
    elif a.retime:
        retime_driver(only)
    else:
        driver(a.quick, only)


if __name__ == "__main__":
    main_guard(main)
