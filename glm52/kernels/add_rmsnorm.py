"""Fusion #3 -- Residual Add + RMSNorm (`fused_add_rmsnorm`).

ONE kernel source, `tl.constexpr` flags select the fused / split behaviour:

    DO_ADD=True , DO_NORM=True   -> FUSED    : read X, RES; write H1 and OUT   (2R + 2W)
    DO_ADD=True , DO_NORM=False  -> unfused#1: read X, RES; write H1           (2R + 1W)
    DO_ADD=False, DO_NORM=True   -> read H1 (as X); write OUT   (1R + 1W)

So the unfused chain moves 5 row-passes and the fused kernel moves 4 -> the pure
bandwidth ceiling of this fusion is 5/4 = 1.25x.

Semantics are exactly `glm52.reference.add_rmsnorm` (= sglang `fused_add_rmsnorm`):

    h1 = (x.float() + residual.float()).to(bf16)          # the NEW residual, written out
    x2 = ((h1.float() * rsqrt(mean(h1^2)+eps)).to(bf16).float() * w).to(bf16)

Note the intermediate round-to-bf16 of `h1` *before* the sum-of-squares, and the second
round-to-bf16 before multiplying by the weight -- both are what torch does in the
reference, and reproducing them keeps fused and unfused directly comparable. (The second
one costs ~3% of runtime and is what torch *inductor* does NOT do; see the log.)

Mapping knobs -- the only thing allowed to differ between the two sides:
  ROWS        rows handled per program
  BLOCK_N     tile width over the hidden dim (6144 = 3*2048 is NOT a power of two)
  ONE_SHOT    derived from BLOCK_N:
              BLOCK_N >= 6144 -> padded power-of-two tile + column mask, the whole row
                                 lives in registers, one load feeds both the reduction
                                 and the normalize
              BLOCK_N <  6144 -> multi-pass: a static-unrolled loop of N_TILES tiles for
                                 the reduction, then a second loop that re-reads the row
                                 (L2-resident) to normalize
  PERSISTENT  False -> grid = one program per row-block, no outer loop at all
              True  -> capped grid, each program strides over row-blocks
  EVICT       cache hints on the streaming loads (evict_first / evict_last on W)
  num_warps / num_stages
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HIDDEN = 6144


# `T` is arg 9. Marking it do-not-specialize keeps ONE binary per config across all five
# regimes (otherwise T=1 and T=32 compile separately, doubling the tuning wall time and
# making the regimes run subtly different code). T only feeds a `rows < T` mask.
@triton.jit(do_not_specialize=[9])
def add_rmsnorm_kernel(
    X,  # [T, N] bf16 -- attn/moe output (fused) or h1 (norm-only)
    RES,  # [T, N] bf16 -- residual in
    W,  # [N]    bf16 -- rmsnorm weight
    H1,  # [T, N] bf16 -- new residual out
    OUT,  # [T, N] bf16 -- normed out
    stride_x,
    stride_r,
    stride_h,
    stride_o,
    T,
    N,
    eps,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_TILES: tl.constexpr,
    ONE_SHOT: tl.constexpr,
    PERSISTENT: tl.constexpr,
    EP: tl.constexpr,
    EPW: tl.constexpr,
    DO_ADD: tl.constexpr,
    DO_NORM: tl.constexpr,
):
    pid = tl.program_id(0)
    rofs = tl.arange(0, ROWS)

    if PERSISTENT:
        nprog = tl.num_programs(0)
        nblk = tl.cdiv(T, ROWS)
        for blk in range(pid, nblk, nprog):
            _body(
                blk * ROWS + rofs, X, RES, W, H1, OUT, stride_x, stride_r, stride_h,
                stride_o, T, N, eps, ROWS, BLOCK_N, N_TILES, ONE_SHOT, EP, EPW,
                DO_ADD, DO_NORM,
            )
    else:
        _body(
            pid * ROWS + rofs, X, RES, W, H1, OUT, stride_x, stride_r, stride_h,
            stride_o, T, N, eps, ROWS, BLOCK_N, N_TILES, ONE_SHOT, EP, EPW,
            DO_ADD, DO_NORM,
        )


@triton.jit
def _body(
    rows,
    X, RES, W, H1, OUT,
    stride_x, stride_r, stride_h, stride_o,
    T, N, eps,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_TILES: tl.constexpr,
    ONE_SHOT: tl.constexpr,
    EP: tl.constexpr,
    EPW: tl.constexpr,
    DO_ADD: tl.constexpr,
    DO_NORM: tl.constexpr,
):
    rmask = rows < T

    if ONE_SHOT:
        cols = tl.arange(0, BLOCK_N)
        cmask = cols < N
        m = rmask[:, None] & cmask[None, :]

        x = tl.load(
            X + rows[:, None] * stride_x + cols[None, :], mask=m, other=0.0,
            eviction_policy=EP,
        )
        if DO_ADD:
            r = tl.load(
                RES + rows[:, None] * stride_r + cols[None, :], mask=m, other=0.0,
                eviction_policy=EP,
            )
            h = (x.to(tl.float32) + r.to(tl.float32)).to(X.dtype.element_ty)
            tl.store(H1 + rows[:, None] * stride_h + cols[None, :], h, mask=m)
        else:
            h = x
        if DO_NORM:
            hf = h.to(tl.float32)
            ssq = tl.sum(hf * hf, axis=1)
            rstd = 1.0 / tl.sqrt(ssq / N + eps)
            w = tl.load(W + cols, mask=cmask, other=0.0, eviction_policy=EPW)
            y = (hf * rstd[:, None]).to(X.dtype.element_ty).to(tl.float32)
            y = y * w.to(tl.float32)[None, :]
            tl.store(
                OUT + rows[:, None] * stride_o + cols[None, :],
                y.to(X.dtype.element_ty),
                mask=m,
            )
    else:
        # ---- pass 1: add (+ store h1) and accumulate the sum of squares ----
        acc = tl.zeros([ROWS], dtype=tl.float32)
        for t in tl.static_range(N_TILES):
            cols = t * BLOCK_N + tl.arange(0, BLOCK_N)
            cmask = cols < N
            m = rmask[:, None] & cmask[None, :]
            x = tl.load(
                X + rows[:, None] * stride_x + cols[None, :], mask=m, other=0.0,
                eviction_policy=EP,
            )
            if DO_ADD:
                r = tl.load(
                    RES + rows[:, None] * stride_r + cols[None, :], mask=m, other=0.0,
                    eviction_policy=EP,
                )
                h = (x.to(tl.float32) + r.to(tl.float32)).to(X.dtype.element_ty)
                tl.store(H1 + rows[:, None] * stride_h + cols[None, :], h, mask=m)
            else:
                h = x
            if DO_NORM:
                hf = h.to(tl.float32)
                acc += tl.sum(hf * hf, axis=1)

        # ---- pass 2: normalize, re-reading the row (L2-resident) ----
        if DO_NORM:
            rstd = 1.0 / tl.sqrt(acc / N + eps)
            for t in tl.static_range(N_TILES):
                cols = t * BLOCK_N + tl.arange(0, BLOCK_N)
                cmask = cols < N
                m = rmask[:, None] & cmask[None, :]
                if DO_ADD:
                    h = tl.load(
                        H1 + rows[:, None] * stride_h + cols[None, :], mask=m,
                        other=0.0, eviction_policy=EP,
                    )
                else:
                    h = tl.load(
                        X + rows[:, None] * stride_x + cols[None, :], mask=m,
                        other=0.0, eviction_policy=EP,
                    )
                hf = h.to(tl.float32)
                w = tl.load(W + cols, mask=cmask, other=0.0, eviction_policy=EPW)
                y = (hf * rstd[:, None]).to(X.dtype.element_ty).to(tl.float32)
                y = y * w.to(tl.float32)[None, :]
                tl.store(
                    OUT + rows[:, None] * stride_o + cols[None, :],
                    y.to(X.dtype.element_ty),
                    mask=m,
                )


# --------------------------------------------------------------------------------------
# launchers -- all three go through the single kernel above
# --------------------------------------------------------------------------------------
def _grid(T: int, cfg: dict):
    nblk = triton.cdiv(T, cfg["ROWS"])
    cap = cfg.get("grid_cap")
    return (nblk if not cap else min(nblk, int(cap)),)


def _launch(x, res, w, h1, out, cfg, do_add: bool, do_norm: bool, N: int = HIDDEN):
    T = x.shape[0]
    block_n = cfg["BLOCK_N"]
    cap = cfg.get("grid_cap")
    nblk = triton.cdiv(T, cfg["ROWS"])
    add_rmsnorm_kernel[_grid(T, cfg)](
        x,
        res if res is not None else x,
        w if w is not None else x,
        h1 if h1 is not None else x,
        out if out is not None else x,
        x.stride(0),
        res.stride(0) if res is not None else 0,
        h1.stride(0) if h1 is not None else 0,
        out.stride(0) if out is not None else 0,
        T,
        N,
        cfg.get("eps", 1e-5),
        ROWS=cfg["ROWS"],
        BLOCK_N=block_n,
        N_TILES=triton.cdiv(N, block_n),
        ONE_SHOT=block_n >= N,
        PERSISTENT=bool(cap) and int(cap) < nblk,
        EP="evict_first" if cfg.get("EVICT") else "",
        EPW="evict_last" if cfg.get("EVICT") else "",
        DO_ADD=do_add,
        DO_NORM=do_norm,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def fused_add_rmsnorm(x, res, w, h1, out, cfg):
    """FUSED: 2 reads + 2 writes."""
    _launch(x, res, w, h1, out, cfg, do_add=True, do_norm=True)


def add_only(x, res, h1, cfg):
    """unfused kernel #1: h1 = x + res  (2 reads + 1 write)."""
    _launch(x, res, None, h1, None, cfg, do_add=True, do_norm=False)


def norm_only(h1, w, out, cfg):
    """unfused kernel #2: out = rmsnorm(h1) * w  (1 read + 1 write)."""
    _launch(h1, None, w, None, out, cfg, do_add=False, do_norm=True)
