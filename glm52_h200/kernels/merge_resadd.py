"""Fusion #10 -- Expert Merge + Residual Add (the tail of the MoE block).  **H200 port.**

The op, at the framework level (`glm52_h200.reference.expert_merge` then `+ residual`):

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
(the op has no FLOPs worth counting, so the latency-aware ceiling in
`glm52_h200.traffic` equals the traffic ratio).

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
  CACHE_MOD   NEW on H200 -- `cache_modifier` on the streaming loads (see below)
  num_warps / num_stages

--------------------------------------------------------------------------------------
H200 (sm_90, Hopper) notes -- what changed from glm52/kernels/merge_resadd.py and why
--------------------------------------------------------------------------------------
1. **No Hopper-specific code path here, deliberately** -- same argument as #3's kernel.
   This is a pure gather-reduce over `topk` contiguous row slabs with no `tl.dot` in it.
   Warp specialization exists to overlap loads with an MMA pipeline that this kernel does
   not have; TMA would replace an access pattern that is already a fully coalesced
   `[ROWS, TOPK, BLOCK_N]` slab of a contiguous tensor; clusters buy DSMEM sharing that a
   per-row weighted sum has no use for.  The Hopper features are exercised in
   `lazy_prenorm.py`, where the algorithm's precondition genuinely is warp specialization.

2. **The new regimes make this the largest-footprint vector kernel in the study.**  `Y` is
   `[T, 8, 6144]` bf16 = 96 KB per token, so `decode_bs1024` is a **100 MB** read and
   `prefill_t8192` is **805 MB**.  Both dwarf the ~50 MB L2, which is exactly why the
   harness's flush buffer has to be derived from the probed L2 (>= 4x) rather than
   assumed: an under-sized flush leaves `m` warm between the unfused chain's two kernels
   and manufactures a fusion loss.  On 143 GB of HBM the allocation itself is
   comfortable -- this fusion is one of the ones that could not be run at all on the 4060.

3. **`CACHE_MOD` is a new mapping knob** (`cache_modifier` on the loads; `.cg` bypasses
   L1, `.cs` marks the line evict-first).  It is orthogonal to `eviction_policy` (an L2
   hint) and is the one Hopper-relevant lever a bandwidth-bound kernel has now that L2 is
   big enough for the choice to matter.  Default `""` reproduces the C500/4060 kernel byte
   for byte.  It is NOT applied to the store of `m`: making that store streaming would
   evict the line the unfused chain's *second* kernel is about to read, i.e. penalise one
   arm of the pair through a knob the other arm cannot use.

4. **Nothing about the device appears in this file.**  `HIDDEN`/`TOPK` are model constants
   from the HF config; every tile bound, warp count and grid cap comes from the bench's
   cached device probe.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HIDDEN = 6144
TOPK = 8

# `cache_modifier` values Triton accepts on a LOAD.  Validated in the launcher so a typo
# in a config dict fails as a plain Python error at grid-build time rather than as a
# Triton compile error partway through an autotune.
_LOAD_CACHE_MODS = ("", ".ca", ".cg", ".cs", ".lu", ".cv")


# `T` is arg 11.  Marking it do-not-specialize keeps ONE binary per config across all
# seven regimes (otherwise T=1, 32, 256, 512, 1024, 2048, 8192 each compile separately --
# a ~7x tuning wall time -- and the regimes would silently run different code).  T only
# feeds a row mask.
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
    CM: tl.constexpr,
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
                UNROLL, EP, CM, ROUND_MID, DO_MERGE, DO_RESADD,
            )
    else:
        _body(
            pid, Y, WT, RES, M, OUT, stride_yt, stride_yk, stride_wt, stride_r,
            stride_m, stride_o, T, N, ROWS, BLOCK_N, N_TILES, TOPK_C, KVEC,
            UNROLL, EP, CM, ROUND_MID, DO_MERGE, DO_RESADD,
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
    CM: tl.constexpr,
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
            yv = tl.load(p, mask=m3, other=0.0, eviction_policy=EP, cache_modifier=CM)
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
                               rmask, m2, EP, CM)
            else:
                for k in range(TOPK_C):
                    acc += _wy(Y, WT, rows, cols, k, stride_yt, stride_yk, stride_wt,
                               rmask, m2, EP, CM)
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
            mask=m2, other=0.0, eviction_policy=EP, cache_modifier=CM,
        ).to(tl.float32)

    if DO_RESADD:
        r = tl.load(
            RES + rows[:, None] * stride_r + cols[None, :],
            mask=m2, other=0.0, eviction_policy=EP, cache_modifier=CM,
        )
        o = mg + r.to(tl.float32)
        tl.store(
            OUT + rows[:, None] * stride_o + cols[None, :],
            o.to(Y.dtype.element_ty),
            mask=m2,
        )


@triton.jit
def _wy(Y, WT, rows, cols, k, stride_yt, stride_yk, stride_wt, rmask, m2,
        EP: tl.constexpr, CM: tl.constexpr):
    """One expert's contribution: w[:, k] * Y[:, k, :]  as fp32 [ROWS, BLOCK_N]."""
    y = tl.load(
        Y + rows[:, None] * stride_yt + k * stride_yk + cols[None, :],
        mask=m2, other=0.0, eviction_policy=EP, cache_modifier=CM,
    )
    # WT is [T, 8] fp32 = 32 B/token: it is L1-resident by construction and gets no
    # streaming hint of any kind.
    w = tl.load(WT + rows * stride_wt + k, mask=rmask, other=0.0)
    return y.to(tl.float32) * w[:, None]


# --------------------------------------------------------------------------------------
# launchers -- all three variants go through the single kernel above
# --------------------------------------------------------------------------------------
def _cache_mod(cfg: dict) -> str:
    cm = str(cfg.get("CACHE_MOD", "") or "")
    if cm not in _LOAD_CACHE_MODS:
        raise ValueError(
            f"CACHE_MOD={cm!r} is not a Triton load cache_modifier; "
            f"expected one of {_LOAD_CACHE_MODS}"
        )
    return cm


def _nblk(T: int, cfg: dict, N: int = HIDDEN) -> int:
    return triton.cdiv(T, cfg["ROWS"]) * triton.cdiv(N, cfg["BLOCK_N"])


def _launch(y, wt, res, m, out, cfg, do_merge: bool, do_resadd: bool, T: int,
            N: int = HIDDEN):
    block_n = cfg["BLOCK_N"]
    cap = cfg.get("grid_cap")
    nblk = _nblk(T, cfg, N)
    persistent = bool(cap) and int(cap) < nblk
    grid = (min(nblk, int(cap)) if persistent else nblk,)
    return merge_resadd_kernel[grid](
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
        CM=_cache_mod(cfg),
        ROUND_MID=bool(cfg.get("ROUND_MID", 1)),
        DO_MERGE=do_merge,
        DO_RESADD=do_resadd,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def fused_merge_resadd(y, wt, res, out, cfg):
    """FUSED: read Y [T,topk,H] + RES, write OUT.  `m` is never materialized."""
    return _launch(y, wt, res, None, out, cfg, True, True, T=y.shape[0])


def merge_only(y, wt, m, cfg):
    """unfused kernel #1: m = sum_k w_k * Y[:,k,:]   (topk reads + 1 write)."""
    return _launch(y, wt, None, m, None, cfg, True, False, T=y.shape[0])


def resadd_only(m, res, out, cfg):
    """unfused kernel #2: out = m + res   (2 reads + 1 write)."""
    return _launch(m, None, res, m, out, cfg, False, True, T=m.shape[0])


def launch_flags(cfg: dict, T: int, N: int = HIDDEN) -> dict:
    """What `_launch` will actually do with this config -- for the result JSON.

    `PERSISTENT` is *derived* from grid_cap vs the natural block count, so it depends on T
    as well as on the config: two regimes running "the same config" can run different code.
    Recording it per (arm, regime) is what makes that visible afterwards (audit lesson 7).
    """
    cap = cfg.get("grid_cap")
    nblk = _nblk(T, cfg, N)
    persistent = bool(cap) and int(cap) < nblk
    return {
        "ROWS": int(cfg["ROWS"]),
        "BLOCK_N": int(cfg["BLOCK_N"]),
        "N_TILES": triton.cdiv(N, int(cfg["BLOCK_N"])),
        "n_blocks": nblk,
        "grid": min(nblk, int(cap)) if persistent else nblk,
        "PERSISTENT": persistent,
        "KVEC": bool(cfg.get("KVEC", 0)),
        "UNROLL": bool(cfg.get("UNROLL", 1)),
        "ROUND_MID": bool(cfg.get("ROUND_MID", 1)),
        "EVICT": bool(cfg.get("EVICT")),
        "CACHE_MOD": _cache_mod(cfg),
        "num_warps": int(cfg["num_warps"]),
        "num_stages": int(cfg["num_stages"]),
    }
