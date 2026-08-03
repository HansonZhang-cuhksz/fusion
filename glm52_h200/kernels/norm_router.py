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
GEMMs consume it.  The router weight is 6144*256*2 B = 3.0 MB against **~50 MB of L2 on
H200** (it was 8 MB on C500 and 32 MB on the 4060), so its re-read by every CTA is an L2
hit with a lot of room to spare -- the assumption this fusion rests on is *stronger* here
than on either previous device, and it holds for both arms equally.

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
   **The safe option is taken: warp specialization force-disables REREAD_H1** (the
   launcher clears it, and `launch_flags()` records that it did).  Pass 2 then re-adds
   `x + res` from its own loads: a self-contained per-thread computation with no handoff at
   all -- correct by construction, at the cost of one extra L2-resident read of `res`.
   The alternative, a `tl.debug_barrier()` between the passes, was rejected: a full-CTA
   `bar.sync` inside a kernel whose warps have been split into asymmetric partitions is a
   *hang* risk, not an error risk, and a hang on a device nobody can test on costs the
   whole round trip.  Since REREAD_H1 is itself a tuned knob, dropping the
   (WS=True, REREAD_H1=True) combination removes nothing the tuner can reach by another
   route -- (WS=True, REREAD_H1=False) is in the grid, and both arms get the same rule.
2. **The TOPW store/reload** in the top-k epilogue.  This one is *outside* the
   specialized loop -- `tl.range(warp_specialize=True)` partitions the loop body only;
   everything before and after it runs on the default warps -- so both the store and the
   reload execute on the same thread of the same warp, and the same-thread argument still
   holds verbatim.  No barrier is added and none is needed; this note exists so a future
   reader does not have to re-derive it.

`num_ctas` (thread-block clusters) is offered because the 3 MB router weight is read by
every CTA and cluster-level reuse is the one Hopper feature with a plausible story here.
It collapses to 1 -- and the kwarg is then omitted from the launch entirely -- unless
`hopper.caps().clusters` says a cluster launch really worked on this box.  The grid is
NOT divided by it: Triton multiplies gridDimX internally, so one Triton program becomes
one cluster.  No ceiling is invented; an over-large cluster is left to fail at launch and
be recorded as a failed config, like any other tile parameter.  This kernel has no early
`return`, which is why clusters are allowed here and not in `lazy_prenorm`'s grouped
GEMM, where a partial-cluster exit would risk a hang rather than an error.

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
        if DO_ADD and REREAD_H1:
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
        # (The launcher guarantees REREAD_H1 is False whenever warp specialization is
        # on; see the module docstring, section "WARP SPECIALIZATION".)
        if WARP_SPECIALIZE:
            for _ in tl.range(0, N_TILES, 1, warp_specialize=True):
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
             E: int = NUM_EXPERTS) -> dict:
    """Everything derived from `cfg` that the kernel cannot re-derive, in one place.

    Used by `_launch` and exposed through `launch_flags()`, so the result JSON records the
    *effective* flags rather than the requested ones.  Three of them can be silently
    downgraded -- WARP_SPECIALIZE (feature absent, or no GEMM to overlap), num_ctas (no
    cluster support) and REREAD_H1 (unsafe combination) -- and a config table on its own
    would not say which code actually ran.
    """
    bm = cfg["BLOCK_M"]
    bk = cfg["BLOCK_K"]
    # pass-1 k-tile; only meaningful when DO_NORM.  Defaults to the GEMM's k-tile.
    bkn = int(cfg.get("NORM_BK") or bk) if do_norm else bk
    be = cfg["BLOCK_E"] if do_gemm else E
    nsplit = E // be
    nblk = triton.cdiv(T, bm)
    grid = nblk * nsplit

    # Default OFF, deliberately: for #4/#5 warp specialization is a *tuning knob* the
    # bench may sweep, not the experiment, so the default must reproduce the audited
    # C500/4060 kernel exactly.  (Auto-from-`caps()` belongs in lazy_prenorm.py, where
    # picking the variant IS the experiment.)  Both arms of a pair must build this axis
    # from `hopper.ws_choices()`, so it is pruned identically on the two sides.
    ws_flag, ws_kw, ws_why = _ws(bool(cfg.get("WARP_SPECIALIZE", False)))
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
        "N_TILES": None,  # filled by the caller, which knows N
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
    r = _resolve(cfg, do_norm, do_gemm, T, E)
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
        N_TILES=N // r["BLOCK_K"],
        N_TILES_N=N // r["BLOCK_KN"],
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
    r = _resolve(cfg, do_norm, do_gemm, T, E)
    r["N_TILES"] = N // r["BLOCK_K"]
    r["N_TILES_N"] = N // r["BLOCK_KN"]
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
