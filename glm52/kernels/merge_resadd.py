"""Fusion #10 -- Expert Merge + Residual Add (the tail of the MoE block).

The op, at the framework level (`glm52.reference.expert_merge` followed by `+ residual`):

    m   = (Y.float() * w[..., None]).sum(dim=1).to(bf16)     # Y : [T, topk, H] UNWEIGHTED
    out = (m.float() + res.float()).to(bf16)

`Y` is the per-expert output of the down projection *before* the routing weight is
applied, `w` is `[T, topk]` fp32 (sigmoid / noaux_tc router output, already normalized and
scaled by `routed_scaling_factor`), `res` is the pre-MoE residual `[T, H]`.

ONE kernel source; two `tl.constexpr` flags select the behaviour:

    DO_MERGE=True , DO_RESADD=True   -> FUSED     : read Y (topk*act) + RES; write OUT
                                                    -> (topk+1) reads + 1 write
    DO_MERGE=True , DO_RESADD=False  -> unfused#1 : read Y; write M   -> topk R + 1 W
    DO_MERGE=False, DO_RESADD=True   -> unfused#2 : read M, RES; write OUT -> 2 R + 1 W

With topk=8 the unfused chain moves (8+1) + (2+1) = 12 row-passes and the fused kernel
moves (8+1) + 1 = 10, so the pure-bandwidth ceiling is 12/10 = **1.20x**, at every regime
(the op has no FLOPs worth counting, so the latency-aware ceiling in `glm52.traffic`
equals the traffic ratio).

NOTE on the intermediate rounding.  The fused kernel rounds the fp32 weighted sum to bf16
*before* adding the residual, exactly as the unfused chain is forced to (it has to store
`m` as bf16).  That is not required -- keeping the sum in fp32 would be strictly more
accurate and costs nothing -- but it makes the two sides **bitwise identical**, which is
the strongest possible statement that they do the same work.  `ROUND_MID=False` turns the
rounding off and is reported separately in the bench as an accuracy note.

Mapping knobs -- the only thing allowed to differ between the two sides:
  ROWS        tokens handled per program
  BLOCK_N     tile width over the hidden dim (6144 = 3*2048, not a power of two)
  KVEC        False -> loop over the topk axis, one [ROWS, BLOCK_N] tile per expert,
                       accumulating into an fp32 register tile
              True  -> load the whole [ROWS, TOPK, BLOCK_N] slab as one 3-D block and
                       `tl.sum(axis=1)` it (topk is a compile-time constant, so the
                       address arithmetic is fully static)
  UNROLL      when KVEC=False: `tl.static_range` (fully unrolled, TOPK loads in flight)
              vs `tl.range` (rolled, pipelined by num_stages)
  PERSISTENT  grid_cap -> capped grid, each program strides over (row-block, n-tile) pairs
  EVICT       `evict_first` streaming hints on the loads
  num_warps / num_stages
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HIDDEN = 6144
TOPK = 8


# `T` is arg 11.  Marking it do-not-specialize keeps ONE binary per config across all five
# regimes (otherwise T=1, 32, 256, 2048, 8192 each compile separately -- a ~5x tuning wall
# time -- and the regimes would silently run different code).  T only feeds a row mask.
@triton.jit(do_not_specialize=[11])
def merge_resadd_kernel(
    Y,  # [T, TOPK, N] bf16 -- per-expert outputs, UNWEIGHTED
    WT,  # [T, TOPK]    fp32 -- routing weights
    RES,  # [T, N]      bf16 -- residual in
    M,  # [T, N]        bf16 -- merged out (DO_RESADD=False) / merged in (DO_MERGE=False)
    OUT,  # [T, N]      bf16 -- final out
    stride_yt,
    stride_yk,
    stride_wt,
    stride_r,
    stride_m,
    stride_o,
    T,
    N,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_TILES: tl.constexpr,
    TOPK_C: tl.constexpr,
    KVEC: tl.constexpr,
    UNROLL: tl.constexpr,
    PERSISTENT: tl.constexpr,
    EP: tl.constexpr,
    ROUND_MID: tl.constexpr,
    DO_MERGE: tl.constexpr,
    DO_RESADD: tl.constexpr,
):
    pid = tl.program_id(0)
    if PERSISTENT:
        nprog = tl.num_programs(0)
        nblk = tl.cdiv(T, ROWS) * N_TILES
        for b in range(pid, nblk, nprog):
            _body(
                b, Y, WT, RES, M, OUT, stride_yt, stride_yk, stride_wt, stride_r,
                stride_m, stride_o, T, N, ROWS, BLOCK_N, N_TILES, TOPK_C, KVEC,
                UNROLL, EP, ROUND_MID, DO_MERGE, DO_RESADD,
            )
    else:
        _body(
            pid, Y, WT, RES, M, OUT, stride_yt, stride_yk, stride_wt, stride_r,
            stride_m, stride_o, T, N, ROWS, BLOCK_N, N_TILES, TOPK_C, KVEC,
            UNROLL, EP, ROUND_MID, DO_MERGE, DO_RESADD,
        )


@triton.jit
def _body(
    b,
    Y, WT, RES, M, OUT,
    stride_yt, stride_yk, stride_wt, stride_r, stride_m, stride_o,
    T, N,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_TILES: tl.constexpr,
    TOPK_C: tl.constexpr,
    KVEC: tl.constexpr,
    UNROLL: tl.constexpr,
    EP: tl.constexpr,
    ROUND_MID: tl.constexpr,
    DO_MERGE: tl.constexpr,
    DO_RESADD: tl.constexpr,
):
    # consecutive programs walk the n-tiles of one row-block first (contiguous HBM)
    rb = b // N_TILES
    tb = b % N_TILES

    rows = rb * ROWS + tl.arange(0, ROWS)
    cols = tb * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < T
    cmask = cols < N
    m2 = rmask[:, None] & cmask[None, :]

    if DO_MERGE:
        if KVEC:
            ks = tl.arange(0, TOPK_C)
            p = (
                Y
                + rows[:, None, None] * stride_yt
                + ks[None, :, None] * stride_yk
                + cols[None, None, :]
            )
            m3 = rmask[:, None, None] & cmask[None, None, :]
            yv = tl.load(p, mask=m3, other=0.0, eviction_policy=EP)
            wv = tl.load(
                WT + rows[:, None] * stride_wt + ks[None, :],
                mask=rmask[:, None],
                other=0.0,
            )
            acc = tl.sum(yv.to(tl.float32) * wv[:, :, None], axis=1)
        else:
            acc = tl.zeros([ROWS, BLOCK_N], dtype=tl.float32)
            if UNROLL:
                for k in tl.static_range(TOPK_C):
                    acc += _wy(Y, WT, rows, cols, k, stride_yt, stride_yk, stride_wt,
                               rmask, m2, EP)
            else:
                for k in range(TOPK_C):
                    acc += _wy(Y, WT, rows, cols, k, stride_yt, stride_yk, stride_wt,
                               rmask, m2, EP)
        if ROUND_MID:
            mg = acc.to(Y.dtype.element_ty).to(tl.float32)
        else:
            mg = acc
        if not DO_RESADD:
            tl.store(
                M + rows[:, None] * stride_m + cols[None, :],
                acc.to(Y.dtype.element_ty),
                mask=m2,
            )
    else:
        mg = tl.load(
            M + rows[:, None] * stride_m + cols[None, :],
            mask=m2, other=0.0, eviction_policy=EP,
        ).to(tl.float32)

    if DO_RESADD:
        r = tl.load(
            RES + rows[:, None] * stride_r + cols[None, :],
            mask=m2, other=0.0, eviction_policy=EP,
        )
        o = mg + r.to(tl.float32)
        tl.store(
            OUT + rows[:, None] * stride_o + cols[None, :],
            o.to(Y.dtype.element_ty),
            mask=m2,
        )


@triton.jit
def _wy(Y, WT, rows, cols, k, stride_yt, stride_yk, stride_wt, rmask, m2,
        EP: tl.constexpr):
    """One expert's contribution: w[:, k] * Y[:, k, :]  as fp32 [ROWS, BLOCK_N]."""
    y = tl.load(
        Y + rows[:, None] * stride_yt + k * stride_yk + cols[None, :],
        mask=m2, other=0.0, eviction_policy=EP,
    )
    w = tl.load(WT + rows * stride_wt + k, mask=rmask, other=0.0)
    return y.to(tl.float32) * w[:, None]


# --------------------------------------------------------------------------------------
# launchers -- all three variants go through the single kernel above
# --------------------------------------------------------------------------------------
def _nblk(T: int, cfg: dict, N: int = HIDDEN) -> int:
    return triton.cdiv(T, cfg["ROWS"]) * triton.cdiv(N, cfg["BLOCK_N"])


def _launch(y, wt, res, m, out, cfg, do_merge: bool, do_resadd: bool, T: int,
            N: int = HIDDEN):
    block_n = cfg["BLOCK_N"]
    cap = cfg.get("grid_cap")
    nblk = _nblk(T, cfg, N)
    persistent = bool(cap) and int(cap) < nblk
    grid = (min(nblk, int(cap)) if persistent else nblk,)
    merge_resadd_kernel[grid](
        y,
        wt if wt is not None else y,
        res if res is not None else y,
        m if m is not None else y,
        out if out is not None else y,
        y.stride(0) if y.dim() == 3 else 0,
        y.stride(1) if y.dim() == 3 else 0,
        wt.stride(0) if wt is not None else 0,
        res.stride(0) if res is not None else 0,
        m.stride(0) if m is not None else 0,
        out.stride(0) if out is not None else 0,
        T,
        N,
        ROWS=cfg["ROWS"],
        BLOCK_N=block_n,
        N_TILES=triton.cdiv(N, block_n),
        TOPK_C=cfg.get("TOPK", TOPK),
        KVEC=bool(cfg.get("KVEC", 0)),
        UNROLL=bool(cfg.get("UNROLL", 1)),
        PERSISTENT=persistent,
        EP="evict_first" if cfg.get("EVICT") else "",
        ROUND_MID=bool(cfg.get("ROUND_MID", 1)),
        DO_MERGE=do_merge,
        DO_RESADD=do_resadd,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def fused_merge_resadd(y, wt, res, out, cfg):
    """FUSED: read Y [T,topk,H] + RES, write OUT.  `m` is never materialized."""
    _launch(y, wt, res, None, out, cfg, True, True, T=y.shape[0])


def merge_only(y, wt, m, cfg):
    """unfused kernel #1: m = sum_k w_k * Y[:,k,:]   (topk reads + 1 write)."""
    _launch(y, wt, None, m, None, cfg, True, False, T=y.shape[0])


def resadd_only(m, res, out, cfg):
    """unfused kernel #2: out = m + res   (2 reads + 1 write)."""
    _launch(m, None, res, m, out, cfg, False, True, T=m.shape[0])
