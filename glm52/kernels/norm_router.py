"""Fusions #4 and #5 -- (ResAdd +) RMSNorm + Router GEMM (+ sigmoid top-8).

ONE kernel source.  Five `tl.constexpr` flags select which stages run, so every
variant below is *the same* code with a different flag combination -- only the
mapping (BLOCK_M / BLOCK_K / BLOCK_E / num_warps / num_stages / eviction hints) is
allowed to differ between the fused and the unfused side.

    DO_ADD  DO_NORM  DO_GEMM  DO_TOPK   what it is
    ------  -------  -------  -------   -------------------------------------------
      -        T        -        -      rmsnorm kernel            (unfused #5, part 1)
      T        T        -        -      add+rmsnorm kernel        (unfused #4, part 1)
      -        -        T        -      router GEMM kernel        (unfused, part 2)
      -        -        -        T      sigmoid + top-8 kernel    (unfused, part 3)
      -        T        T        -      FUSED #5
      T        T        T        -      FUSED #4
      -        T        T        T      FUSED #5 + FUSE_TOPK
      T        T        T        T      FUSED #4 + FUSE_TOPK

Semantics reproduced exactly (see `glm52.reference`):

    h1     = (x.float() + res.float()).to(bf16)                        # DO_ADD
    x2     = ((h1.float()*rsqrt(mean(h1^2)+eps)).to(bf16).float()*w).to(bf16)   # DO_NORM
    logits = x2.float() @ Wg.float().T                                 # DO_GEMM, fp32
    s      = sigmoid(logits)                                           # DO_TOPK
    v,i    = topk(s, 8);  w = v / v.sum() * 2.5                        # DO_TOPK

`moe_router_dtype = float32`: the accumulation is fp32 and both operands are bf16
values, whose products are *exact* in fp32 (8+8 mantissa bits fit in 24), so
`tl.dot(bf16, bf16, acc=fp32)` is the fp32 reference matmul up to summation order.

GLM-5.2 has `n_group = topk_group = 1`, so `noaux_tc` degenerates to a plain top-k over
the sigmoid scores (`reference.router` takes the same branch) -- there is no group mask
to reproduce.

Structure of the fused kernel (this is the "free normalization" shape):

    pass 1  read x (+ res, write h1), accumulate sum-of-squares  -> rstd
    pass 2  re-read the row (L2-resident), normalize, WRITE x2, and feed the same
            registers straight into `tl.dot` against the router weight tile.

So the router's *read* of x2 disappears; x2 itself is still written because the expert
GEMMs consume it.  The router weight is 6144*256*2 B = 3.0 MB against an 8 MB L2, so
its re-read by every CTA is an L2 hit, not HBM traffic.

Mapping knobs
-------------
BLOCK_M   rows per program
BLOCK_K   k-tile of the GEMM (and of pass 2's normalize)
NORM_BK   k-tile of pass 1 (the sum of squares).  Decoupled from BLOCK_K on purpose: the
          GEMM wants 32-64, but a sum-of-squares tiled that narrowly runs 96-192
          sequential cross-lane reductions per row, while the stand-alone norm kernel
          picks 2048 and runs 3.  Defaults to BLOCK_K.
BLOCK_E   expert-tile.  BLOCK_E < 256 splits the 256 experts over `NSPLIT = 256/BLOCK_E`
          programs *per row block* -- the only source of extra parallelism at T=1, where
          the router is a GEMV and there is exactly one row block.  The norm work is then
          done redundantly by each split (its reads are L2 hits) and only split 0 writes
          h1 / x2.  DO_TOPK requires BLOCK_E = 256 (all logits of a row in one program).
REREAD_H1 pass 2 re-reads h1 (safe only when NSPLIT == 1; the store and the load have
          identical layouts so each thread reads back its own element) instead of
          re-adding x + res.
EVICT     eviction hints: streaming activations `evict_first`, router weight + rmsnorm
          weight `evict_last` (we *want* Wg to stay in L2).
STORE_X2  attribution only (`fused_norm_router_no_x2`): suppresses the x2 store so the
          fused kernel's cost can be split into prologue-instructions vs the extra store.
          Never used by a delivered variant -- it would skip a live output.

KNOWN MACA CODEGEN DEFECTS worked around here (both are silent wrong answers, see
LOG-03 §4): a row-wise reduction over a `tl.dot` accumulator is only correct when the
mma tile fits one warp-row (so FUSE_TOPK is capped at BLOCK_M=16, or 32 with 4 warps),
and broadcasting a reduced [BLOCK_M] vector back into a [BLOCK_M, 8] tile after a dot
returns garbage -- hence the store/reload of the top-k winners below.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HIDDEN = 6144
NUM_EXPERTS = 256
TOP_K = 8


# `T` is argument 14 -- do-not-specialize keeps ONE binary per config across all five
# regimes, otherwise every regime recompiles the whole grid.
@triton.jit(do_not_specialize=[14])
def norm_router_kernel(
    X,  # [T, N] bf16   attn output (#4) / h1 (#5) / x2 (router-only)
    RES,  # [T, N] bf16 residual in
    W,  # [N]    bf16   rmsnorm weight
    H1,  # [T, N] bf16  new residual out
    X2,  # [T, N] bf16  normed out  (input when DO_NORM=0 and DO_GEMM=1)
    WGT,  # [N, E] bf16 router weight, TRANSPOSED (see note in the launcher)
    LOGITS,  # [T, E] fp32
    TOPW,  # [T, TOPK] fp32
    TOPI,  # [T, TOPK] int32
    stride_x,
    stride_r,
    stride_h,
    stride_o,
    stride_l,
    T,
    eps,
    scale,
    N: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_KN: tl.constexpr,
    BLOCK_E: tl.constexpr,
    NSPLIT: tl.constexpr,
    N_TILES: tl.constexpr,
    N_TILES_N: tl.constexpr,
    DO_ADD: tl.constexpr,
    DO_NORM: tl.constexpr,
    DO_GEMM: tl.constexpr,
    DO_TOPK: tl.constexpr,
    STORE_X2: tl.constexpr,
    WRITE_LOGITS: tl.constexpr,
    NORM_PROB: tl.constexpr,
    REREAD_H1: tl.constexpr,
    EP: tl.constexpr,
    EPW: tl.constexpr,
):
    pid = tl.program_id(0)
    if NSPLIT > 1:
        pid_m = pid // NSPLIT
        pid_e = pid % NSPLIT
    else:
        pid_m = pid
        pid_e = 0

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < T
    ko = tl.arange(0, BLOCK_K)
    eo = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)

    # ---------------- pass 1: (residual add and) sum of squares ----------------------
    # Pass 1 has its OWN k-tile (BLOCK_KN).  It must: the GEMM wants BLOCK_K = 32-64, but
    # a sum-of-squares tiled that narrowly runs 96-192 sequential cross-lane reductions
    # over the row, which is exactly what the stand-alone norm kernel avoids by choosing
    # BLOCK_K = 2048.  Tying the two together would under-tune the fused side.
    if DO_NORM:
        kn = tl.arange(0, BLOCK_KN)
        ss = tl.zeros([BLOCK_M], dtype=tl.float32)
        xp = X + rows[:, None] * stride_x + kn[None, :]
        rp = RES + rows[:, None] * stride_r + kn[None, :]
        hp = H1 + rows[:, None] * stride_h + kn[None, :]
        for _ in range(N_TILES_N):
            x = tl.load(xp, mask=rmask[:, None], other=0.0, eviction_policy=EP)
            if DO_ADD:
                r = tl.load(rp, mask=rmask[:, None], other=0.0, eviction_policy=EP)
                h = (x.to(tl.float32) + r.to(tl.float32)).to(X.dtype.element_ty)
                if pid_e == 0:
                    tl.store(hp, h, mask=rmask[:, None])
                rp += BLOCK_KN
                hp += BLOCK_KN
            else:
                h = x
            hf = h.to(tl.float32)
            ss += tl.sum(hf * hf, axis=1)
            xp += BLOCK_KN
        rstd = 1.0 / tl.sqrt(ss / N + eps)

    # ---------------- pass 2: normalize -> x2, and the router GEMM -------------------
    if DO_GEMM:
        acc = tl.zeros([BLOCK_M, BLOCK_E], dtype=tl.float32)
        bp = WGT + ko[:, None] * E + eo[None, :]

    if DO_NORM or DO_GEMM:
        wp = W + ko
        op = X2 + rows[:, None] * stride_o + ko[None, :]
        if DO_ADD and REREAD_H1:
            sp = H1 + rows[:, None] * stride_h + ko[None, :]
        else:
            sp = X + rows[:, None] * stride_x + ko[None, :]
        r2p = RES + rows[:, None] * stride_r + ko[None, :]

        for _ in range(N_TILES):
            if DO_NORM:
                s_ = tl.load(sp, mask=rmask[:, None], other=0.0, eviction_policy=EP)
                if DO_ADD and not REREAD_H1:
                    r2 = tl.load(r2p, mask=rmask[:, None], other=0.0, eviction_policy=EP)
                    s_ = (s_.to(tl.float32) + r2.to(tl.float32)).to(X.dtype.element_ty)
                    r2p += BLOCK_K
                ww = tl.load(wp, eviction_policy=EPW)
                y = (s_.to(tl.float32) * rstd[:, None]).to(X.dtype.element_ty)
                y = y.to(tl.float32) * ww.to(tl.float32)[None, :]
                a = y.to(X.dtype.element_ty)
                if STORE_X2:
                    if pid_e == 0:
                        tl.store(op, a, mask=rmask[:, None])
                sp += BLOCK_K
                wp += BLOCK_K
                op += BLOCK_K
            else:
                a = tl.load(op, mask=rmask[:, None], other=0.0, eviction_policy=EP)
                op += BLOCK_K
            if DO_GEMM:
                b = tl.load(bp, eviction_policy=EPW)
                acc += tl.dot(a, b)
                bp += BLOCK_K * E

    # ---------------- epilogue: logits / sigmoid+top-k -------------------------------
    if DO_GEMM and WRITE_LOGITS:
        tl.store(
            LOGITS + rows[:, None] * stride_l + eo[None, :], acc, mask=rmask[:, None]
        )

    if DO_TOPK:
        if DO_GEMM:
            lg = acc
        else:
            lg = tl.load(
                LOGITS + rows[:, None] * stride_l + eo[None, :],
                mask=rmask[:, None],
                other=-1e30,
            )
        sc = tl.sigmoid(lg)
        cur = tl.where(rmask[:, None], sc, -1.0)
        wsum = tl.zeros([BLOCK_M], dtype=tl.float32)
        # NOTE (MACA codegen bug, see log): assembling the 8 winners into a [BLOCK_M, 8]
        # tile with `tl.where(arange(8)==i, v[:, None], 0)` produces GARBAGE whenever
        # `cur` descends from a `tl.dot` accumulator (ids come back as -1, weights nan) --
        # the mma->blocked layout conversion of the broadcast is miscompiled.  Writing the
        # reduced [BLOCK_M] vectors out one at a time is correct, so the winners are
        # parked in TOPW and re-read (L1/L2 resident, same thread reads back its own
        # element) to apply the norm_topk_prob scaling.  The stand-alone top-k kernel
        # runs this identical epilogue, so both sides pay it.
        for i in tl.static_range(TOPK):
            v = tl.max(cur, axis=1)
            idx = tl.argmax(cur, axis=1)
            wsum += v
            tl.store(TOPW + rows * TOPK + i, v, mask=rmask)
            tl.store(TOPI + rows * TOPK + i, idx.to(tl.int32), mask=rmask)
            cur = tl.where(eo[None, :] == idx[:, None], -1.0, cur)
        if NORM_PROB:
            inv = scale / tl.maximum(wsum, 1e-20)
        else:
            inv = scale + tl.zeros([BLOCK_M], dtype=tl.float32)
        for i in tl.static_range(TOPK):
            v = tl.load(TOPW + rows * TOPK + i, mask=rmask, other=0.0)
            tl.store(TOPW + rows * TOPK + i, v * inv, mask=rmask)


# --------------------------------------------------------------------------------------
# launchers -- every one of them goes through the single kernel above
# --------------------------------------------------------------------------------------
def wgt_from_gate(gate_w: torch.Tensor) -> torch.Tensor:
    """`gate_w` is [E, H] (nn.Linear).  The kernel wants [H, E] so that the B tile is
    contiguous in the expert dimension.  This is a *weight* layout choice made once at
    load time (free at inference), and BOTH sides of the comparison get the same one."""
    return gate_w.t().contiguous()


def _launch(
    cfg,
    T,
    x,
    res,
    w,
    h1,
    x2,
    wgt,
    logits,
    topw,
    topi,
    do_add,
    do_norm,
    do_gemm,
    do_topk,
    write_logits,
    N=HIDDEN,
    E=NUM_EXPERTS,
    topk=TOP_K,
):
    bm = cfg["BLOCK_M"]
    bk = cfg["BLOCK_K"]
    # pass-1 k-tile; only meaningful when DO_NORM.  Defaults to the GEMM's k-tile.
    bkn = int(cfg.get("NORM_BK") or bk) if do_norm else bk
    be = cfg["BLOCK_E"] if do_gemm else E
    nsplit = E // be
    if do_topk:
        assert be == E, "top-k fusion needs all E logits of a row in one program"
    dummy = x
    nblk = triton.cdiv(T, bm)
    ev = bool(cfg.get("EVICT"))
    norm_router_kernel[(nblk * nsplit,)](
        x,
        res if res is not None else dummy,
        w if w is not None else dummy,
        h1 if h1 is not None else dummy,
        x2 if x2 is not None else dummy,
        wgt if wgt is not None else dummy,
        logits if logits is not None else dummy,
        topw if topw is not None else dummy,
        topi if topi is not None else dummy,
        x.stride(0),
        res.stride(0) if res is not None else 0,
        h1.stride(0) if h1 is not None else 0,
        x2.stride(0) if x2 is not None else 0,
        logits.stride(0) if logits is not None else 0,
        T,
        cfg.get("eps", 1e-5),
        cfg.get("scale", 2.5),
        N=N,
        E=E,
        TOPK=topk,
        BLOCK_M=bm,
        BLOCK_K=bk,
        BLOCK_KN=bkn,
        BLOCK_E=be,
        NSPLIT=nsplit,
        N_TILES=N // bk,
        N_TILES_N=N // bkn,
        DO_ADD=do_add,
        DO_NORM=do_norm,
        DO_GEMM=do_gemm,
        DO_TOPK=do_topk,
        STORE_X2=bool(cfg.get("STORE_X2", True)),
        WRITE_LOGITS=write_logits,
        NORM_PROB=bool(cfg.get("NORM_PROB", True)),
        # re-reading h1 in pass 2 is only safe when the pass-1 store and the pass-2 load
        # have the SAME tile shape (so each thread reads back its own element) and no
        # other program wrote it
        REREAD_H1=bool(cfg.get("REREAD_H1", True)) and nsplit == 1 and bkn == bk,
        EP="evict_first" if ev else "",
        EPW="evict_last" if ev else "",
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


# ---- unfused pieces -------------------------------------------------------------------
def rmsnorm_only(x, w, x2, cfg):
    _launch(cfg, x.shape[0], x, None, w, None, x2, None, None, None, None,
            False, True, False, False, False)


def add_rmsnorm_only(x, res, w, h1, x2, cfg):
    _launch(cfg, x.shape[0], x, res, w, h1, x2, None, None, None, None,
            True, True, False, False, False)


def router_gemm(x2, wgt, logits, cfg):
    _launch(cfg, x2.shape[0], x2, None, None, None, x2, wgt, logits, None, None,
            False, False, True, False, True)


def topk_only(logits, topw, topi, cfg):
    _launch(cfg, logits.shape[0], logits, None, None, None, None, None, logits,
            topw, topi, False, False, False, True, False)


# ---- fused ----------------------------------------------------------------------------
def fused_norm_router(x, w, x2, wgt, logits, cfg):
    """#5 fused: rmsnorm + router GEMM."""
    _launch(cfg, x.shape[0], x, None, w, None, x2, wgt, logits, None, None,
            False, True, True, False, True)


def fused_add_norm_router(x, res, w, h1, x2, wgt, logits, cfg):
    """#4 fused: residual add + rmsnorm + router GEMM."""
    _launch(cfg, x.shape[0], x, res, w, h1, x2, wgt, logits, None, None,
            True, True, True, False, True)


def fused_norm_router_topk(x, w, x2, wgt, topw, topi, cfg):
    """#5 fused + FUSE_TOPK: logits never leave registers."""
    _launch(cfg, x.shape[0], x, None, w, None, x2, wgt, None, topw, topi,
            False, True, True, True, False)


def fused_norm_router_no_x2(x, w, x2, wgt, logits, cfg):
    """ATTRIBUTION ONLY -- fused #5 with the x2 store suppressed (`STORE_X2=False`).

    It does not produce x2, so it is NOT a candidate implementation and never appears in
    a speedup row; it exists to split the fused kernel's cost into "the prologue's extra
    load + arithmetic" and "the extra store", the same way LOG-01 split F1's epilogue."""
    _launch(dict(cfg, STORE_X2=False), x.shape[0], x, None, w, None, x2, wgt, logits,
            None, None, False, True, True, False, True)


def fused_add_norm_router_topk(x, res, w, h1, x2, wgt, topw, topi, cfg):
    """#4 fused + FUSE_TOPK."""
    _launch(cfg, x.shape[0], x, res, w, h1, x2, wgt, None, topw, topi,
            True, True, True, True, False)
