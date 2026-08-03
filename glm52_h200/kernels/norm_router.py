"""Fusions #4 and #5 -- (ResAdd +) RMSNorm + Router GEMM (+ sigmoid top-8).
**H200 / sm_90 port.**

ONE kernel source.  Five `tl.constexpr` flags select which stages run, so every
variant below is *the same* code with a different flag combination -- only the
mapping (BLOCK_M / BLOCK_K / BLOCK_E / num_warps / num_stages / eviction hints, and on
this device WARP_SPECIALIZE / num_ctas) is allowed to differ between the two sides.

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

Semantics reproduced exactly (see `glm52_h200.reference`):

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
GEMMs consume it.  The router weight is 6144*256*2 B = 3.0 MB against a **measured
62 914 560 B (60 MB) of L2 on this H200** (it was 8 MB on C500 and 32 MB on the 4060), so
its re-read by every CTA is an L2 hit with a lot of room to spare -- the assumption this
fusion rests on is *stronger* here than on either previous device, and it holds for both
arms equally.

The device's balance point moved even more than its L2 did: 4.23-4.25 TB/s of copy
bandwidth against 821.6 TF/s of bf16 cuBLAS is **~185 FLOP/byte**, where C500 was 82 and
the 4060 was 84.  This fusion's entire product is *bytes removed* (one full activation read
of the router's input), and bytes are 2.2x more expensive here per unit of compute than on
either earlier device.  That is the strongest a-priori case any fusion in this study has on
this machine -- and it is a case about the memory system, not about warp specialization, so
the two effects must be reported separately rather than added up.

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
REREAD_H1 pass 2 re-reads h1 instead of re-adding x + res.  Safe only when the pass-1
          store and the pass-2 load have identical layouts *and* execute on the same
          thread -- see the WARP SPECIALIZATION section below, which is why this knob is
          force-disabled under WARP_SPECIALIZE.
EVICT     eviction hints: streaming activations `evict_first`, router weight + rmsnorm
          weight `evict_last` (we *want* Wg to stay in L2).
WARP_SPECIALIZE
          NEW on H200 -- warp-specialize the pass-2 loop (see below).
num_ctas  NEW on H200 -- thread-block cluster width (see below).
STORE_X2  attribution only (`fused_norm_router_no_x2`): suppresses the x2 store so the
          fused kernel's cost can be split into prologue-instructions vs the extra store.
          Never used by a delivered variant -- it would skip a live output.

======================================================================================
WARP SPECIALIZATION on H200, and the two unsynchronised handoffs it endangers
======================================================================================
`WARP_SPECIALIZE=True` turns the **pass-2 loop** -- the one containing `tl.dot` -- into
`tl.range(..., warp_specialize=True)`, so Triton may split it into a producer partition
(the A/B loads) and a consumer partition (the wgmma).  Pass 1 stays a plain loop: it has
no MMA for a producer/consumer split to overlap with, so specializing it would be noise.
For the same reason the launcher only enables it when `DO_GEMM` is set; a norm-only or a
top-k-only kernel gets nothing from it and would just double its own tuning grid with
configs that compile to identical code.

Which *spelling* this Triton has -- the source-level `tl.range(warp_specialize=)` above,
or the older forked `num_consumer_groups` launch kwargs -- is decided once by
`kernels/hopper.py`, and the launcher passes whichever it reports.  Everything below
applies to both: they reshape the warps the same way.

Measured on this box (`preflight_h200.json`, triton 3.6.0, and these are facts, not
expectations -- nothing here re-derives them):

    warp_specialize_tl_range              OK     <- the spelling this kernel is written for
    warp_specialize_num_consumer_groups   FAIL   "Keyword argument ... unrecognised"
    thread_block_cluster_num_ctas         OK
    tl.range(..., warp_specialize=False, disable_licm=False)   -- the kwarg exists

so `hopper.ws_mode()` resolves to `"range"` here and `_ws()` returns `(True, {}, ...)`.
The launch-kwarg branch is dead code on this stack and stays only because it is what a
forked Triton would offer; it is never passed speculatively, since an unrecognised launch
kwarg is a hard `KeyError` that would kill an entire autotune.

**Three arms at one config.**  `arm_chains()` below hands back {`unfused`, `fused`,
`fused_ws`} (plus `unfused_ws` when asked) as ready-to-time chains from ONE cfg, so the fused /
unfused ratio and the specialized / classic ratio come from the same tiles, warps, stages
and eviction hints and differ only in the flags named by the arm.  `WARP_SPECIALIZE` stays
**default-off** for every other entry point: for #4/#5 warp specialization is a tuning
knob, not the experiment, so the default must still reproduce the audited C500/4060 kernel
exactly.  (In `lazy_prenorm.py`, where choosing the variant IS the experiment, the same
default holds for the same reason.)

This kernel contains **two unsynchronised same-thread handoffs** through memory.  Both
are correct on a classic launch by the same argument -- one thread stores a value and the
same thread loads it back, so program order alone orders them -- and warp specialization
is exactly the transformation that can invalidate that argument, because it changes which
warp executes which part of the loop.

1. **REREAD_H1** (pass 1 stores `h1`, pass 2 loads it back).  Guarded on the classic path
   by `nsplit == 1 and bkn == bk`, which makes the two tiles the same shape so each thread
   reads back its own element.  Under warp specialization the pass-2 load can be hoisted
   onto a *producer* warp that never executed pass 1, and it can run ahead of the
   consumer's stores.
   **The safe option is taken, unconditionally: warp specialization force-disables
   REREAD_H1.**  It is enforced in TWO places on purpose -- the launcher clears it (and
   `launch_flags()` records that it did), *and* the kernel re-ANDs `not WARP_SPECIALIZE`
   into every use of the flag, so that a direct `norm_router_kernel[grid](...)` that
   bypasses the launcher cannot resurrect the handoff.  Both are constexpr, so the belt and
   the braces cost nothing: Triton emits only the taken side.  Pass 2 then re-adds
   `x + res` from its own loads: a self-contained per-thread computation with no handoff at
   all -- correct by construction, at the cost of one extra L2-resident read of `res`.
   The alternative, a `tl.debug_barrier()` between the passes, was rejected and the reason
   is specific to this transform, not general caution: `tl.debug_barrier()` lowers to a
   CTA-wide `barrier` that every thread in the block must arrive at, and Triton's automatic
   warp specialization parks the extra warp group outside the `ttg.warp_specialize` region
   in a wait loop that will never arrive.  That is a *hang*, not an error, and a hang on a
   device nobody can iterate on costs the whole round trip.  Since REREAD_H1 is itself a
   tuned knob, dropping the (WS=True, REREAD_H1=True) combination removes nothing the tuner
   can reach by another route -- (WS=True, REREAD_H1=False) is in the grid, and both arms
   get the same rule.
2. **The TOPW store/reload** in the top-k epilogue.  This one is *outside* the
   specialized loop -- `tl.range(warp_specialize=True)` partitions the loop body only;
   everything before and after it runs on the default warps -- so both the store and the
   reload execute on the same thread of the same warp, and the same-thread argument still
   holds verbatim.  No barrier is added and none is needed; this note exists so a future
   reader does not have to re-derive it.  The one way to break it would be to move the
   epilogue *into* the loop, so: do not.  (Belt and braces here would have to be the
   barrier, which as argued above can hang under this exact transform -- which is why the
   two handoffs get opposite treatments rather than one uniform rule.)

`num_ctas` (thread-block clusters) is offered because the 3 MB router weight is read by
every CTA and cluster-level reuse is the one Hopper feature with a plausible story here.
It collapses to 1 -- and the kwarg is then omitted from the launch entirely -- unless
`hopper.caps().clusters` says a cluster launch really worked on this box.  The grid is
NOT divided by it: Triton multiplies gridDimX internally, so one Triton program becomes
one cluster.  No ceiling is invented; an over-large cluster is left to fail at launch and
be recorded as a failed config, like any other tile parameter.  This kernel has no early
`return`, which is why clusters are allowed here and not in `lazy_prenorm`'s grouped
GEMM, where a partial-cluster exit would risk a hang rather than an error.

Tiles, and the one thing that is NOT free about bigger ones
-----------------------------------------------------------
Nothing in this file caps a tile: BLOCK_M / BLOCK_K / BLOCK_E come from the config and the
only ceiling is what the compiler will allocate (232 448 B per block on this device, opt-in;
the preflight compiled a 196 608 B tile).  But this kernel walks the hidden dimension with
`N_TILES = N // BLOCK_K` and splits the experts with `NSPLIT = E // BLOCK_E`, both integer
divisions with no remainder handling anywhere -- a BLOCK_K that does not divide 6144, or a
BLOCK_E that does not divide 256, silently normalizes over a PREFIX of the row and returns a
plausible wrong answer rather than failing.  That was unreachable while the grids stopped at
tiles a 4060 could hold; with 192 KB tiles now compiling it is one grid edit away.  `_resolve`
therefore asserts both divisibilities, so the failure mode is an exception at launch instead
of a quiet 2 % error in the router logits that no rel_err threshold would necessarily catch.

A note on reading the small-T rows
----------------------------------
The preflight's `harness_floor_us` (40.55) and `timer_tick_us` (0.256, matching only 3 % of
samples where the detector wants >= 98 %) were measured while another tenant was using the
GPU -- `mem_free` was 98.8 GB of 150 GB.  Neither number is physical, and both are exactly
what qualifies a decode-sized measurement as "tick-limited" or "below the launch floor".
These kernels at T=1 are among the shortest in the suite, so those verdicts matter most
precisely where they are least trustworthy.  Re-probe on an idle GPU (`--gpu`), and until
then report the caveat rather than silently flagging or unflagging a cell.

======================================================================================
MACA CODEGEN WORKAROUNDS -- retained deliberately, and why
======================================================================================
Two silent wrong-answer defects on C500's Triton 3.0/MACA shaped this kernel (LOG-03 s4):
a row-wise reduction over a `tl.dot` accumulator was only correct when the mma tile fit
one warp-row, and broadcasting a reduced [BLOCK_M] vector back into a [BLOCK_M, 8] tile
after a dot returned garbage -- hence the store/reload of the top-k winners below.

Neither defect is expected on NVIDIA.  The workaround is **kept anyway**: it is paid by
the fused kernel and by the stand-alone top-k kernel alike, so it cannot bias the ratio,
and keeping it makes the H200 numbers directly comparable with the C500 and 4060 tables,
which is the entire point of a port.  What it costs on Hopper is an open question worth
one measurement, not a silent code change.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# `kernels/hopper.py` is the suite's single runtime feature verdict -- TMA, warp
# specialization, thread-block clusters -- and the ONLY place any of them is decided.  A
# second opinion (re-reading the preflight JSON here, say) can disagree with the first,
# and two disagreeing capability tables is exactly how one arm of a fused/unfused pair
# ends up compiled with a feature the other arm did not get.  Import failure must not take
# this module down: "cannot tell" resolves to the classic path, which is always correct.
try:  # package import (glm52_h200.kernels.norm_router)
    from . import hopper
except ImportError:  # pragma: no cover -- imported without package context
    try:
        import hopper  # type: ignore
    except ImportError:  # pragma: no cover -- broken checkout, not a device condition
        hopper = None  # type: ignore


def _ws(enable: bool) -> "tuple[bool, dict, str]":
    """(constexpr flag, launch kwargs, why) for warp specialization.

    Two return values because the two mechanisms live in different places: upstream Triton
    spells it at *source* level (`tl.range(warp_specialize=True)`, so it must reach the
    compiler as a constexpr), a forked Triton spells it as *launch* kwargs.  In "launch"
    mode the flag is False and the kwargs are non-empty and the kernel runs its classic
    mainloop -- correct, not a downgrade.  Never guessed: an unrecognised launch kwarg is
    a hard `KeyError` in Triton, so `hopper` decides it from a probe.
    """
    if not enable:
        return False, {}, "off (requested)"
    if hopper is None:
        return False, {}, "refused: kernels/hopper.py not importable"
    try:
        flag = bool(hopper.ws_source_flag(True))
        kw = dict(hopper.ws_kwargs(True))
    except Exception as exc:  # noqa: BLE001 -- a detection bug costs the feature, not
        # the run: "cannot tell" resolves to the classic path, which is always correct.
        return False, {}, f"refused: hopper raised {type(exc).__name__}: {exc}"[:160]
    if flag or kw:
        return flag, kw, f"on (mode={hopper.ws_mode()})"
    return False, {}, f"refused: hopper says warp spec unusable (mode={hopper.ws_mode()})"


# Every spelling of the knob a config dict may carry.  The lowercase one is not decoration:
# `bench.h200_cfg_overlays()` widens a grid with `{"warp_specialize": True}`, so a resolver
# that only recognised the SHOUTING name would tune a grid whose two halves are identical
# code -- and then report the noise between them as a warp-specialization effect.
_CFG_WS_KEYS = ("WARP_SPECIALIZE", "WS", "warp_specialize")

# The H200 mapping axes this module's launcher forwards, for `bench.widen(grid, mod)`.
# Both are honoured by `_resolve` for every variant that goes through `_launch`, which is
# all of them -- there is one kernel here, so there is no per-kernel exception to make.
H200_CFG_KEYS = ("warp_specialize", "num_ctas")


def _caps_ws() -> bool:
    """`caps().warp_specialize`, False whenever the feature layer cannot answer."""
    if hopper is None:
        return False
    try:
        return bool(hopper.caps().warp_specialize)
    except Exception:  # noqa: BLE001 -- a detection bug costs the feature, not the run
        return False


def cfg_warp_specialize(cfg: "dict | None") -> bool:
    """Warp specialization requested by a config dict.  **Default OFF**, `"auto"` asks caps.

    Off by default on purpose: for #4/#5 warp specialization is a tuning knob, not the
    experiment, so a config that never mentions it must produce the audited C500/4060
    kernel.  A caps-driven default would have made every untouched call site measure a
    different kernel than the cross-device tables it is compared against.
    """
    for k in _CFG_WS_KEYS:
        if cfg and k in cfg:
            v = cfg[k]
            if isinstance(v, str):
                return _caps_ws() if v.strip().lower() == "auto" else bool(v)
            return bool(v)
    return False


def _clusters(cfg: dict) -> "tuple[int, dict, str]":
    """(num_ctas, launch kwargs, why).  `num_ctas` does NOT change the grid we pass --
    Triton multiplies gridDimX internally, so one Triton program becomes one cluster."""
    try:
        n = int(cfg.get("num_ctas", 1) or 1)
    except (TypeError, ValueError):
        n = 1
    if n <= 1:
        return 1, {}, "off"
    if hopper is None:
        return 1, {}, "refused: kernels/hopper.py not importable"
    kw = dict(hopper.cluster_kwargs(n))
    if kw:
        return int(kw["num_ctas"]), kw, f"on ({kw['num_ctas']} CTAs/cluster)"
    return 1, {}, "refused: caps say no clusters"


HIDDEN = 6144
NUM_EXPERTS = 256
TOP_K = 8


# `T` is argument 14 -- do-not-specialize keeps ONE binary per config across all seven
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
    WARP_SPECIALIZE: tl.constexpr,
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
    # Not warp-specialized: no MMA here for a producer/consumer split to hide behind.
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
        # `and not WARP_SPECIALIZE` is the belt to the launcher's braces.  Re-reading H1
        # here is a store-in-pass-1 / load-in-pass-2 handoff whose ONLY guarantee is that
        # the same thread does both; warp specialization is precisely the transform that
        # breaks that (the pass-2 load can land on a producer warp that never ran pass 1,
        # and runs ahead of the consumer's stores).  `_resolve()` already clears the flag,
        # but a direct kernel launch bypasses `_resolve()`, and a data race that only
        # appears under a feature nobody can test on is not a bug anyone would find.  Both
        # operands are constexpr, so this costs nothing: Triton emits only the taken side.
        if DO_ADD and REREAD_H1 and not WARP_SPECIALIZE:
            sp = H1 + rows[:, None] * stride_h + ko[None, :]
        else:
            sp = X + rows[:, None] * stride_x + ko[None, :]
        r2p = RES + rows[:, None] * stride_r + ko[None, :]

        # The loop body appears TWICE on purpose.  `warp_specialize=` has to be written at
        # the `tl.range` call site, and it must not appear in the source at all on a
        # Triton that predates the kwarg -- Triton visits only the taken side of a
        # constexpr `if`, so the control path never parses it.  Factoring the body into a
        # shared @triton.jit helper would have perturbed the control arm's codegen, and
        # the control arm has to stay identical to the audited C500/4060 kernel for the
        # cross-device comparison to mean anything.  EDIT BOTH COPIES TOGETHER.
        # The one place the two copies deliberately DIFFER is the residual re-add: inside
        # this `if` the handoff through H1 is off by construction (see the pointer setup
        # above), so pass 2 re-adds `x + res` unconditionally.  Reading `REREAD_H1` here
        # would be dead but misleading -- it would suggest the specialized path can still
        # take the handoff, which is the exact race this file must not have.
        if WARP_SPECIALIZE:
            for _ in tl.range(0, N_TILES, 1, warp_specialize=True):
                if DO_NORM:
                    s_ = tl.load(sp, mask=rmask[:, None], other=0.0, eviction_policy=EP)
                    if DO_ADD:
                        r2 = tl.load(r2p, mask=rmask[:, None], other=0.0,
                                     eviction_policy=EP)
                        s_ = (s_.to(tl.float32) + r2.to(tl.float32)).to(
                            X.dtype.element_ty
                        )
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
        else:
            for _ in range(N_TILES):
                if DO_NORM:
                    s_ = tl.load(sp, mask=rmask[:, None], other=0.0, eviction_policy=EP)
                    if DO_ADD and not REREAD_H1:
                        r2 = tl.load(r2p, mask=rmask[:, None], other=0.0,
                                     eviction_policy=EP)
                        s_ = (s_.to(tl.float32) + r2.to(tl.float32)).to(
                            X.dtype.element_ty
                        )
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
        # NOTE (MACA codegen bug, see LOG-03): assembling the 8 winners into a [BLOCK_M,8]
        # tile with `tl.where(arange(8)==i, v[:, None], 0)` produces GARBAGE whenever
        # `cur` descends from a `tl.dot` accumulator (ids come back as -1, weights nan) --
        # the mma->blocked layout conversion of the broadcast is miscompiled.  Writing the
        # reduced [BLOCK_M] vectors out one at a time is correct, so the winners are
        # parked in TOPW and re-read (L1/L2 resident, same thread reads back its own
        # element) to apply the norm_topk_prob scaling.  The stand-alone top-k kernel
        # runs this identical epilogue, so both sides pay it.
        #
        # H200: this epilogue sits OUTSIDE the (optionally) warp-specialized pass-2 loop,
        # so it runs on the default warps and the store/reload is still the same thread
        # reading back its own element.  No barrier is required.  Whether the workaround
        # is needed at all on NVIDIA is an open question; it is retained so the three
        # devices' numbers stay comparable, and it is paid by both arms.
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


def _resolve(cfg: dict, do_norm: bool, do_gemm: bool, T: int,
             E: int = NUM_EXPERTS, N: int = HIDDEN) -> dict:
    """Everything derived from `cfg` that the kernel cannot re-derive, in one place.

    Used by `_launch` and exposed through `launch_flags()`, so the result JSON records the
    *effective* flags rather than the requested ones.  Three of them can be silently
    downgraded -- WARP_SPECIALIZE (feature absent, or no GEMM to overlap), num_ctas (no
    cluster support) and REREAD_H1 (unsafe combination) -- and a config table on its own
    would not say which code actually ran.

    It is also where the tile shape is checked for *divisibility*, which is the one way a
    bigger tile can hurt here.  Nothing caps a tile below the device's SMEM ceiling (232 448
    B opt-in on this H200, against 49 152 B default-visible), and nothing should; but the
    kernel walks the hidden dimension with `N // BLOCK_K` and the experts with
    `E // BLOCK_E`, neither of which handles a remainder.  An indivisible tile therefore
    normalizes over a PREFIX of the row, or covers a subset of the experts, and returns a
    plausible wrong answer.  Assert instead: a launch that raises costs one config, a
    wrong-but-plausible router logit costs the credibility of every row it appears in.
    """
    bm = cfg["BLOCK_M"]
    bk = cfg["BLOCK_K"]
    # pass-1 k-tile; only meaningful when DO_NORM.  Defaults to the GEMM's k-tile.
    bkn = int(cfg.get("NORM_BK") or bk) if do_norm else bk
    be = cfg["BLOCK_E"] if do_gemm else E
    if do_norm or do_gemm:
        assert N % bk == 0, (
            f"BLOCK_K={bk} does not divide N={N}: the k-loop would cover "
            f"{N // bk * bk} of {N} columns and silently return a wrong answer"
        )
    if do_norm:
        assert N % bkn == 0, (
            f"NORM_BK={bkn} does not divide N={N}: the sum of squares would cover "
            f"{N // bkn * bkn} of {N} columns, i.e. normalize by the wrong rstd"
        )
    if do_gemm:
        assert E % be == 0, (
            f"BLOCK_E={be} does not divide E={E}: {E - E // be * be} experts would get no "
            f"logits at all"
        )
    nsplit = E // be
    nblk = triton.cdiv(T, bm)
    grid = nblk * nsplit

    # Default OFF, deliberately: for #4/#5 warp specialization is a *tuning knob* the
    # bench may sweep, not the experiment, so the default must reproduce the audited
    # C500/4060 kernel exactly.  (`lazy_prenorm.py`, where picking the variant IS the
    # experiment, defaults off for the same reason and names its arms explicitly.)  Both
    # arms of a pair must build this axis from `hopper.ws_choices()`, so it is pruned
    # identically on the two sides.
    ws_flag, ws_kw, ws_why = _ws(cfg_warp_specialize(cfg))
    # Warp specialization is a mainloop transform; with no `tl.dot` in pass 2 there is
    # nothing for a producer/consumer split to overlap with, and enabling it would only
    # duplicate the norm-only kernel's tuning grid with identical code.
    if not do_gemm and (ws_flag or ws_kw):
        ws_flag, ws_kw = False, {}
        ws_why = "off: no GEMM in this variant, nothing to overlap"

    # re-reading h1 in pass 2 is only safe when the pass-1 store and the pass-2 load have
    # the SAME tile shape (so each thread reads back its own element), no other program
    # wrote it, AND the two passes are not split across warp partitions.  See the module
    # docstring: the safe option under warp specialization is to drop the handoff, not to
    # fence it.  The launch-kwarg spelling reshapes the warps too, so it disables the
    # handoff just as the source-level one does.
    reread = bool(cfg.get("REREAD_H1", True)) and nsplit == 1 and bkn == bk
    reread_why = "on" if reread else "off (nsplit>1 or NORM_BK!=BLOCK_K or requested off)"
    if reread and (ws_flag or ws_kw):
        reread, reread_why = False, "forced off: unsafe cross-partition handoff under WS"

    nctas, ct_kw, ct_why = _clusters(cfg)
    return {
        "BLOCK_M": bm,
        "BLOCK_K": bk,
        "BLOCK_KN": bkn,
        "BLOCK_E": be,
        "NSPLIT": nsplit,
        "N_TILES": N // bk,
        "N_TILES_N": N // bkn,
        "n_blocks": nblk,
        "grid": grid,
        "WARP_SPECIALIZE": ws_flag,
        "WARP_SPECIALIZE_kwargs": ws_kw,
        "WARP_SPECIALIZE_why": ws_why,
        "REREAD_H1": reread,
        "REREAD_H1_why": reread_why,
        "num_ctas": nctas,
        "num_ctas_kwargs": ct_kw,
        "num_ctas_why": ct_why,
        "EVICT": bool(cfg.get("EVICT")),
        "num_warps": cfg["num_warps"],
        "num_stages": cfg["num_stages"],
    }


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
    r = _resolve(cfg, do_norm, do_gemm, T, E, N)
    if do_topk:
        assert r["BLOCK_E"] == E, "top-k fusion needs all E logits of a row in one program"
    dummy = x
    ev = r["EVICT"]
    # The Hopper launch kwargs are only *mentioned* when the feature layer actually
    # resolved them, so the default launch is byte-for-byte the audited one on any Triton,
    # however old.  An unrecognised kwarg raises in Triton; it is never tried speculatively.
    extra = {**r["WARP_SPECIALIZE_kwargs"], **r["num_ctas_kwargs"]}
    return norm_router_kernel[(r["grid"],)](
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
        BLOCK_M=r["BLOCK_M"],
        BLOCK_K=r["BLOCK_K"],
        BLOCK_KN=r["BLOCK_KN"],
        BLOCK_E=r["BLOCK_E"],
        NSPLIT=r["NSPLIT"],
        N_TILES=r["N_TILES"],
        N_TILES_N=r["N_TILES_N"],
        DO_ADD=do_add,
        DO_NORM=do_norm,
        DO_GEMM=do_gemm,
        DO_TOPK=do_topk,
        STORE_X2=bool(cfg.get("STORE_X2", True)),
        WRITE_LOGITS=write_logits,
        NORM_PROB=bool(cfg.get("NORM_PROB", True)),
        REREAD_H1=r["REREAD_H1"],
        EP="evict_first" if ev else "",
        EPW="evict_last" if ev else "",
        WARP_SPECIALIZE=r["WARP_SPECIALIZE"],
        num_warps=r["num_warps"],
        num_stages=r["num_stages"],
        **extra,
    )


def launch_flags(cfg: dict, T: int, do_add: bool, do_norm: bool, do_gemm: bool,
                 do_topk: bool, N: int = HIDDEN, E: int = NUM_EXPERTS) -> dict:
    """Effective flags for one variant, without launching -- for the result JSON."""
    r = _resolve(cfg, do_norm, do_gemm, T, E, N)
    r.update({"DO_ADD": do_add, "DO_NORM": do_norm, "DO_GEMM": do_gemm,
              "DO_TOPK": do_topk})
    return r


# ---- unfused pieces -------------------------------------------------------------------
def rmsnorm_only(x, w, x2, cfg):
    return _launch(cfg, x.shape[0], x, None, w, None, x2, None, None, None, None,
                   False, True, False, False, False)


def add_rmsnorm_only(x, res, w, h1, x2, cfg):
    return _launch(cfg, x.shape[0], x, res, w, h1, x2, None, None, None, None,
                   True, True, False, False, False)


def router_gemm(x2, wgt, logits, cfg):
    return _launch(cfg, x2.shape[0], x2, None, None, None, x2, wgt, logits, None, None,
                   False, False, True, False, True)


def topk_only(logits, topw, topi, cfg):
    return _launch(cfg, logits.shape[0], logits, None, None, None, None, None, logits,
                   topw, topi, False, False, False, True, False)


# ---- fused ----------------------------------------------------------------------------
def fused_norm_router(x, w, x2, wgt, logits, cfg):
    """#5 fused: rmsnorm + router GEMM."""
    return _launch(cfg, x.shape[0], x, None, w, None, x2, wgt, logits, None, None,
                   False, True, True, False, True)


def fused_add_norm_router(x, res, w, h1, x2, wgt, logits, cfg):
    """#4 fused: residual add + rmsnorm + router GEMM."""
    return _launch(cfg, x.shape[0], x, res, w, h1, x2, wgt, logits, None, None,
                   True, True, True, False, True)


def fused_norm_router_topk(x, w, x2, wgt, topw, topi, cfg):
    """#5 fused + FUSE_TOPK: logits never leave registers."""
    return _launch(cfg, x.shape[0], x, None, w, None, x2, wgt, None, topw, topi,
                   False, True, True, True, False)


def fused_norm_router_no_x2(x, w, x2, wgt, logits, cfg):
    """ATTRIBUTION ONLY -- fused #5 with the x2 store suppressed (`STORE_X2=False`).

    It does not produce x2, so it is NOT a candidate implementation and never appears in
    a speedup row; it exists to split the fused kernel's cost into "the prologue's extra
    load + arithmetic" and "the extra store", the same way LOG-01 split F1's epilogue."""
    return _launch(dict(cfg, STORE_X2=False), x.shape[0], x, None, w, None, x2, wgt,
                   logits, None, None, False, True, True, False, True)


def fused_add_norm_router_topk(x, res, w, h1, x2, wgt, topw, topi, cfg):
    """#4 fused + FUSE_TOPK."""
    return _launch(cfg, x.shape[0], x, res, w, h1, x2, wgt, None, topw, topi,
                   True, True, True, True, False)


# ======================================================================================
# THE ARMS: one config, one kernel source, one flag apart
#
# The bench must be able to time the unfused chain, the fused kernel, and the fused kernel
# with warp specialization *at a single config*, because the whole question is whether the
# specialized variant hides the normalization -- and a comparison across two independently
# tuned configs cannot answer it (the tuner would be free to trade the effect away).  This
# is the same isolation that produced `isolation_fuse_on_vs_off_same_cfg`, with one axis
# added because H200 adds one.
#
#   unfused      norm kernel + router GEMM, two launches   (WS off)
#   fused        one kernel                                (WS off)  <- C500 / sm_89 comparable
#   fused_ws     one kernel                                (WS on)
#   unfused_ws   the unfused chain with WS on              (opt-in)
#
# `unfused_ws` is off by default here, unlike in `lazy_prenorm.py`.  The reason is
# structural, not stylistic: the unfused chain's GEMM is `DO_GEMM` with `DO_NORM=0`, and
# `_resolve` keeps warp specialization for it (there IS a `tl.dot` to overlap), so the arm
# is meaningful -- but the chain also contains a norm kernel that `_resolve` correctly
# refuses to specialize, so the arm is only *partly* specialized and its ratio is not the
# clean control that `lazy_prenorm`'s single-kernel `unfused_ws` is.  Ask for it when you
# want it, and read it knowing which half of the chain moved.
# ======================================================================================
ARM_UNFUSED = "unfused"
ARM_FUSED = "fused"
ARM_FUSED_WS = "fused_ws"
ARM_UNFUSED_WS = "unfused_ws"

# name -> (fused?, warp specialization?).  Reporting order, control first.
ARM_SPEC = {
    ARM_UNFUSED: (False, False),
    ARM_FUSED: (True, False),
    ARM_FUSED_WS: (True, True),
    ARM_UNFUSED_WS: (False, True),
}


def with_ws(cfg: dict, on: bool) -> dict:
    """`cfg` with warp specialization forced on/off, every spelling of the key agreeing.

    All three keys are set, not just one, because `cfg_warp_specialize()` takes the first
    it finds: a cfg that arrived from a widened grid carrying `warp_specialize=True` would
    otherwise keep it and quietly turn the control arm into a second specialized run.
    """
    return dict(cfg, **{k: bool(on) for k in _CFG_WS_KEYS})


def warp_specialize_available() -> bool:
    """True iff a warp-specialized launch is expected to work here.

    Ask before building an arm set or a grid axis, so a run on a stack without the feature
    *records* that the axis was dropped instead of tuning two identical halves of a grid.
    """
    if hopper is None:
        return False
    try:
        return bool(hopper.caps().warp_specialize and hopper.ws_mode() in ("range", "launch"))
    except Exception:  # noqa: BLE001
        return False


def arms_available(include_ws: "bool | None" = None, include_unfused_ws: bool = False) -> tuple:
    """Arm names this device can actually run, in reporting order.

    Without warp specialization the WS arms are DROPPED, never aliased onto their controls:
    an arm that is secretly its own control reports a ratio of 1.000, which in a table is
    indistinguishable from a real measurement that found no effect.
    """
    if include_ws is None:
        include_ws = warp_specialize_available()
    out = [ARM_UNFUSED, ARM_FUSED]
    if include_ws:
        out.append(ARM_FUSED_WS)
        if include_unfused_ws:
            out.append(ARM_UNFUSED_WS)
    return tuple(out)


def arm_chains(x, w, x2, wgt, logits, cfg, res=None, h1=None, do_add: bool = False,
               norm_cfg=None, arms=None, include_ws: "bool | None" = None,
               include_unfused_ws: bool = False) -> dict:
    """{arm name -> list of zero-arg callables} for #4/#5 at ONE config.

    Values are chains, not single callables: the unfused arms are two launches, ready for
    `common.bench_chain` / `bench.bench_pair`, which flush L2 once per chain rather than
    between its links (rule 1: in the real layer the producer's output is still in L2 when
    the consumer starts).  `do_add=False` is fusion #5 (rmsnorm + router GEMM);
    `do_add=True` is #4 (residual add first), which needs `res` and `h1`.

    The three fused-side arms differ ONLY in the flags their name declares: same BLOCK_M /
    BLOCK_K / BLOCK_E / NORM_BK / warps / stages / eviction hints.  `_resolve` will still
    force `REREAD_H1=False` on the WS arms -- a *correctness* requirement, not a mapping
    choice (see the module docstring) -- and `launch_flags()` records that it did, so the
    one remaining difference is visible in the result file rather than inferred.

    **`norm_cfg` decides which question the unfused arms answer, so pass it deliberately.**

      * `norm_cfg=<the norm kernel's own tuned config>` -- the unfused arm is the FAIR
        baseline: each kernel in the chain runs at the mapping it was tuned for.  This is
        the number a `fused vs unfused` speedup row may quote.
      * `norm_cfg=None` -- the norm kernel inherits the fused kernel's config, which for the
        GEMM-shaped configs here means a BLOCK_K of 32-64 where the stand-alone norm kernel
        would have picked 2048, i.e. ~100 sequential cross-lane reductions per row instead
        of 3.  That is a strict same-config ISOLATION, and it is a badly under-tuned
        baseline: quoting it as a speedup would be the textbook "under-tuned unfused arm".

    Either way the `fused_ws / fused` comparison is untouched by the choice -- the norm
    kernel is byte-identical in both -- which is why the default is allowed to be the
    isolation rather than the fair chain.
    """
    names = arms_available(include_ws, include_unfused_ws) if arms is None else tuple(arms)
    bad = [a for a in names if a not in ARM_SPEC]
    if bad:
        raise KeyError(f"unknown arm(s) {bad}; known: {tuple(ARM_SPEC)}")
    if do_add and (res is None or h1 is None):
        raise ValueError("do_add=True (fusion #4) needs both `res` and `h1`")

    out: dict = {}
    for name in names:
        fused, ws = ARM_SPEC[name]
        c = with_ws(cfg, ws)
        # The norm kernel gets no warp specialization either way (`_resolve` drops it with
        # no GEMM to overlap), so `nc` differs between the two unfused arms in nothing.
        nc = with_ws(cfg if norm_cfg is None else norm_cfg, ws)
        if fused and do_add:
            out[name] = [lambda c=c: fused_add_norm_router(x, res, w, h1, x2, wgt,
                                                           logits, c)]
        elif fused:
            out[name] = [lambda c=c: fused_norm_router(x, w, x2, wgt, logits, c)]
        elif do_add:
            out[name] = [
                lambda nc=nc: add_rmsnorm_only(x, res, w, h1, x2, nc),
                lambda c=c: router_gemm(x2, wgt, logits, c),
            ]
        else:
            out[name] = [
                lambda nc=nc: rmsnorm_only(x, w, x2, nc),
                lambda c=c: router_gemm(x2, wgt, logits, c),
            ]
    return out


def arm_flags(cfg: dict, T: int, do_add: bool = False, norm_cfg=None, arms=None,
              include_ws: "bool | None" = None, include_unfused_ws: bool = False,
              N: int = HIDDEN, E: int = NUM_EXPERTS) -> dict:
    """{arm name -> the effective flags of each kernel in that arm's chain}.

    Belongs in the result file next to the timings.  "fused_ws was 3 % faster" is not a
    claim about warp specialization unless the file also shows the fused_ws arm resolved
    `WARP_SPECIALIZE` to True -- a refused feature degrades to the control path by design,
    and then the two arms are the same kernel and the 3 % is drift.  It is also where a
    reader sees which `norm_cfg` the unfused arms actually ran (see `arm_chains`).
    """
    names = arms_available(include_ws, include_unfused_ws) if arms is None else tuple(arms)
    out: dict = {}
    for name in names:
        fused, ws = ARM_SPEC[name]
        c = with_ws(cfg, ws)
        nc = with_ws(cfg if norm_cfg is None else norm_cfg, ws)
        if fused:
            out[name] = [launch_flags(c, T, do_add, True, True, False, N, E)]
        else:
            out[name] = [
                launch_flags(nc, T, do_add, True, False, False, N, E),
                launch_flags(c, T, False, False, True, False, N, E),
            ]
    return out


def arm_ratios(ms_by_arm: dict) -> dict:
    """The derived numbers this experiment is about, from {arm -> milliseconds}.

    Keys are omitted, never faked, when the arm they need was not run:

      fusion_speedup_classic  unfused    / fused       the C500 / sm_89 comparable
      fusion_speedup_ws       unfused_ws / fused_ws    the same, both sides specialized
      ws_effect_fused         fused      / fused_ws    > 1 => specialization helped
      ws_effect_unfused       unfused    / unfused_ws  how much of that was the GEMM alone
    """
    def r(num, den):
        a, b = ms_by_arm.get(num), ms_by_arm.get(den)
        if a is None or b is None or not b:
            return None
        return float(a) / float(b)

    out = {
        "fusion_speedup_classic": r(ARM_UNFUSED, ARM_FUSED),
        "fusion_speedup_ws": r(ARM_UNFUSED_WS, ARM_FUSED_WS),
        "ws_effect_fused": r(ARM_FUSED, ARM_FUSED_WS),
        "ws_effect_unfused": r(ARM_UNFUSED, ARM_UNFUSED_WS),
    }
    return {k: v for k, v in out.items() if v is not None}
