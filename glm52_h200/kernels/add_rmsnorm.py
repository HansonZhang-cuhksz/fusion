"""Fusion #3 -- Residual Add + RMSNorm (`fused_add_rmsnorm`).  **H200 / sm_90 port.**

ONE kernel source, `tl.constexpr` flags select the fused / split behaviour:

    DO_ADD=True , DO_NORM=True   -> FUSED    : read X, RES; write H1 and OUT   (2R + 2W)
    DO_ADD=True , DO_NORM=False  -> unfused#1: read X, RES; write H1           (2R + 1W)
    DO_ADD=False, DO_NORM=True   -> read H1 (as X); write OUT   (1R + 1W)

So the unfused chain moves 5 row-passes and the fused kernel moves 4 -> the pure
bandwidth ceiling of this fusion is 5/4 = 1.25x.

Semantics are exactly `glm52_h200.reference.add_rmsnorm` (= sglang `fused_add_rmsnorm`):

    h1 = (x.float() + residual.float()).to(bf16)          # the NEW residual, written out
    x2 = ((h1.float() * rsqrt(mean(h1^2)+eps)).to(bf16).float() * w).to(bf16)

Note the intermediate round-to-bf16 of `h1` *before* the sum-of-squares, and the second
round-to-bf16 before multiplying by the weight -- both are what torch does in the
reference, and reproducing them keeps fused and unfused directly comparable. (The second
one costs ~3% of runtime and is what torch *inductor* does NOT do; see log/LOG-02.)

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
  CACHE_MOD   NEW on H200 -- `cache_modifier` on the streaming loads (see below)
  num_warps / num_stages

--------------------------------------------------------------------------------------
H200 (sm_90, Hopper) notes -- what changed from glm52/kernels/add_rmsnorm.py and why
--------------------------------------------------------------------------------------
1. **There is deliberately NO Hopper-specific code path in this file.**  The three
   features this port could reach for -- TMA, warp specialization, thread-block clusters
   -- all exist to feed and overlap an *MMA* pipeline.  This kernel has no `tl.dot`:
   warp specialization would split a loop whose consumer side is two FMAs and a store,
   TMA would replace an access pattern that is already a perfectly coalesced full row of
   a row-major bf16 matrix, and clusters buy DSMEM sharing that a per-row reduction has
   no use for.  Writing an untestable Hopper path here would be complexity with no
   modelled gain, and it would put an unverifiable difference between the two arms of the
   pair.  Fusion #11 (`lazy_prenorm.py`) is where the Hopper features are exercised; that
   is the file where the algorithm's precondition actually is warp specialization.

2. **Nothing about the device is written down here.**  `HIDDEN`/`TOPK` are *model*
   constants (HF config), not hardware constants.  Every tile bound, grid cap and warp
   count comes from the bench's cached device probe.  On C500 the study hardcoded warp
   64 / 104 SMs / 65536 B SMEM into guards, which pruned the two arms' grids by different
   amounts and therefore moved the ratio that is this study's only output.

3. **L2 is ~50 MB here** -- 6x C500's 8 MB, 1.6x the 4060's 32 MB.  Two consequences the
   caller must honour:
     (a) the harness's L2 flush buffer must be *derived* (>= 4x the probed L2), never a
         literal.  A flush smaller than L2 turns every decode measurement into a
         warm-cache one, which flatters the UNFUSED arm specifically, because it is the
         one that re-reads `h1` from a second kernel;
     (b) `EVICT` changes meaning by regime.  At decode (T <= 1024, one activation pass is
         <= 12 MB) the whole working set is L2-resident and `evict_first` on the
         activations is actively wrong; at prefill_t8192 one pass is 96 MB and streaming
         hints are right.  It is a tuned knob on both sides, so this is a tuning-space
         observation, not a correctness one.

4. **`CACHE_MOD` is a new mapping knob**: `cache_modifier` on the *loads* only (`.cg`
   bypasses L1, `.cs` marks the line evict-first in L1/L2).  It is orthogonal to
   `eviction_policy` (which is an L2 hint) and is the one Hopper-relevant lever a
   bandwidth-bound kernel actually has, now that L2 is big enough for the choice to
   matter.  Default `""` reproduces the C500/4060 kernel byte for byte.
   It is deliberately NOT applied to the stores: marking the `h1` store streaming would
   evict the line that the unfused chain's second kernel is about to read, i.e. it would
   penalise one arm of the pair through a knob the other arm cannot use.  Store policy is
   therefore fixed at the default for both sides.

5. **132 SMs.**  At `decode_bs1` the grid is one row-block, i.e. one CTA on a 132-SM
   machine, on BOTH sides -- the fusion ratio is still fair, but the absolute numbers are
   launch-bound.  The bench must record the timer tick from the preflight and flag any
   speedup whose two operands are within a few ticks of each other.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HIDDEN = 6144

# `cache_modifier` values Triton accepts on a LOAD.  Validated in the launcher so a typo
# in a config dict fails as a plain Python error at grid-build time instead of as a
# Triton compile error 300 configs into an autotune.
_LOAD_CACHE_MODS = ("", ".ca", ".cg", ".cs", ".lu", ".cv")


# `T` is arg 9. Marking it do-not-specialize keeps ONE binary per config across all seven
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
    CM: tl.constexpr,
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
                stride_o, T, N, eps, ROWS, BLOCK_N, N_TILES, ONE_SHOT, EP, EPW, CM,
                DO_ADD, DO_NORM,
            )
    else:
        _body(
            pid * ROWS + rofs, X, RES, W, H1, OUT, stride_x, stride_r, stride_h,
            stride_o, T, N, eps, ROWS, BLOCK_N, N_TILES, ONE_SHOT, EP, EPW, CM,
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
    CM: tl.constexpr,
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
            eviction_policy=EP, cache_modifier=CM,
        )
        if DO_ADD:
            r = tl.load(
                RES + rows[:, None] * stride_r + cols[None, :], mask=m, other=0.0,
                eviction_policy=EP, cache_modifier=CM,
            )
            h = (x.to(tl.float32) + r.to(tl.float32)).to(X.dtype.element_ty)
            tl.store(H1 + rows[:, None] * stride_h + cols[None, :], h, mask=m)
        else:
            h = x
        if DO_NORM:
            hf = h.to(tl.float32)
            ssq = tl.sum(hf * hf, axis=1)
            rstd = 1.0 / tl.sqrt(ssq / N + eps)
            # W is 12 KB and read by every program: it wants to STAY cached, so it gets
            # neither the streaming eviction hint nor the streaming cache modifier.
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
                eviction_policy=EP, cache_modifier=CM,
            )
            if DO_ADD:
                r = tl.load(
                    RES + rows[:, None] * stride_r + cols[None, :], mask=m, other=0.0,
                    eviction_policy=EP, cache_modifier=CM,
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
                    # Pass 1 stored this line and pass 2 reads it back.  Same program,
                    # same thread, identical tile shape -- it is a register spill through
                    # L2, not a cross-thread handoff, so no barrier is required.  (The
                    # analogous handoff in norm_router.py is the one that had to be
                    # disabled under warp specialization; there is no warp specialization
                    # here, so this one is untouched.)
                    h = tl.load(
                        H1 + rows[:, None] * stride_h + cols[None, :], mask=m,
                        other=0.0, eviction_policy=EP, cache_modifier=CM,
                    )
                else:
                    h = tl.load(
                        X + rows[:, None] * stride_x + cols[None, :], mask=m,
                        other=0.0, eviction_policy=EP, cache_modifier=CM,
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
def _cache_mod(cfg: dict) -> str:
    cm = str(cfg.get("CACHE_MOD", "") or "")
    if cm not in _LOAD_CACHE_MODS:
        raise ValueError(
            f"CACHE_MOD={cm!r} is not a Triton load cache_modifier; "
            f"expected one of {_LOAD_CACHE_MODS}"
        )
    return cm


def _grid(T: int, cfg: dict):
    nblk = triton.cdiv(T, cfg["ROWS"])
    cap = cfg.get("grid_cap")
    return (nblk if not cap else min(nblk, int(cap)),)


def _launch(x, res, w, h1, out, cfg, do_add: bool, do_norm: bool, N: int = HIDDEN):
    T = x.shape[0]
    block_n = cfg["BLOCK_N"]
    cap = cfg.get("grid_cap")
    nblk = triton.cdiv(T, cfg["ROWS"])
    return add_rmsnorm_kernel[_grid(T, cfg)](
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
        CM=_cache_mod(cfg),
        DO_ADD=do_add,
        DO_NORM=do_norm,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def fused_add_rmsnorm(x, res, w, h1, out, cfg):
    """FUSED: 2 reads + 2 writes."""
    return _launch(x, res, w, h1, out, cfg, do_add=True, do_norm=True)


def add_only(x, res, h1, cfg):
    """unfused kernel #1: h1 = x + res  (2 reads + 1 write)."""
    return _launch(x, res, None, h1, None, cfg, do_add=True, do_norm=False)


def norm_only(h1, w, out, cfg):
    """unfused kernel #2: out = rmsnorm(h1) * w  (1 read + 1 write)."""
    return _launch(h1, None, w, None, out, cfg, do_add=False, do_norm=True)


def launch_flags(cfg: dict, T: int, N: int = HIDDEN) -> dict:
    """What `_launch` will actually do with this config -- for the result JSON.

    Lesson 7 of the audit: an unfair comparison is only detectable after the fact if each
    arm records the shape of the search it ran.  `ONE_SHOT` and `PERSISTENT` are *derived*
    from BLOCK_N / grid_cap rather than given, and `PERSISTENT` additionally depends on T,
    so two regimes running "the same config" can run different code and a config table
    alone does not say which path was timed.
    """
    block_n = int(cfg["BLOCK_N"])
    cap = cfg.get("grid_cap")
    nblk = triton.cdiv(T, int(cfg["ROWS"]))
    persistent = bool(cap) and int(cap) < nblk
    return {
        "ROWS": int(cfg["ROWS"]),
        "BLOCK_N": block_n,
        "N_TILES": triton.cdiv(N, block_n),
        "ONE_SHOT": bool(block_n >= N),
        "n_blocks": nblk,
        "grid": min(nblk, int(cap)) if cap else nblk,
        "PERSISTENT": persistent,
        "grid_cap": cap,
        "EVICT": bool(cfg.get("EVICT")),
        "CACHE_MOD": _cache_mod(cfg),
        "num_warps": int(cfg["num_warps"]),
        "num_stages": int(cfg["num_stages"]),
    }
