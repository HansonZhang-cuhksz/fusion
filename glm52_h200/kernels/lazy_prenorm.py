"""Fusion #11 -- **Lazy Pre-Norm**: RMSNorm fused into a GEMM as a *prologue*.

**H200 / sm_90 port.  This is the one file in the suite where the target device changes
the science rather than the tuning, so read section "Warp specialization" below before
changing anything in it.**

Algorithm (Zhou et al., "Towards Free Normalization: Fusing Normalization into GEMM and
Attention Kernels", PyTorch blog 2026-07-10, section 2).  For an affine-free RMSNorm the
row scale commutes with the matmul::

    (A * rstd[:, None]) @ B  ==  (A @ B) * rstd[:, None]

so a GEMM CTA can accumulate ``acc += tile_A @ tile_B`` and ``sq += (tile_A*tile_A).sum(-1)``
in the SAME k-loop and apply ``rstd = rsqrt(sq/K + eps)`` as an **epilogue** scale.  The
cyclic dependency (rstd is needed before the k-loop, but only known after it) disappears.

GLM-5.2's RMSNorm *does* have an affine weight ``w`` [K].  The blog calls this a blocker;
at inference it is not one, because ``w`` is a *column* scale of A and B is a constant::

    ((A * rstd) * w) @ B  ==  (A @ (w[:, None] * B)) * rstd

``w`` is therefore folded into B's **rows** once, offline, outside every timed region --
a load-time weight transform, exactly like a quantization scale merge.  ``fold_weight_*``
below does it; the bench validates the folded result against the unfolded
``reference.rmsnorm`` + matmul before it times anything.

ONE kernel source per consumer, a single ``tl.constexpr FUSE_NORM`` flag selecting the
behaviour:

* ``FUSE_NORM=False`` -- plain GEMM.  A is the **pre-normalized** ``x2`` and B is the raw
  weight.  The unfused chain therefore needs a separate RMSNorm kernel first
  (``glm52_h200.kernels.add_rmsnorm.norm_only``, i.e. F3's tuned norm kernel).
* ``FUSE_NORM=True``  -- the SAME GEMM reads the **un-normalized** ``h1`` and B is the
  ``w``-folded weight; a per-row fp32 sum-of-squares rides along in the k-loop and the
  epilogue applies ``rstd``.  ``x2`` is never materialized.

Only the *mapping* (BLOCK_M/N/K, num_warps, num_stages, GROUP_M, WARP_SPECIALIZE, num_ctas)
differs between the two sides, and each side is tuned independently over grids generated
by the same rules.

Two consumers of ``post_attention_layernorm`` have K == hidden == 6144 and are covered:

  (a) ``moe_gateup_prenorm_kernel`` -- the routed-expert w13 grouped GEMM, structurally a
      copy of sglang 0.5.10's ``fused_moe_kernel`` (same sorted_token_ids / expert_ids /
      num_tokens_post_padded dispatch, same ``offs_token // top_k`` gather).  N = 2*I =
      4096, so a row's sum-of-squares is recomputed by ``cdiv(4096, BLOCK_N)`` n-tiles
      *and* by each of the ``top_k`` gathered copies of the token -> big redundancy.
      **This one runs on H200 and could not run on the 4060**: w13 is 12.9 GB and the
      fused/unfused pair needs two of them, against 143 GB here and 8 GB there.
  (b) ``router_gemm_kernel`` -- the dense router GEMM ``[T,6144] @ [6144,256]``.  N = 256
      means 1-4 n-tiles, so redundancy is ~1x: the near-ideal case for this fusion.

``SQ_MODE`` selects among four numerically equivalent ways to accumulate the sum of
squares.  They are NOT equally fast and which one wins depends on ``BLOCK_N`` and on the
backend -- on C500/MACA mode 1 won the router and mode 2 won w13 (log/LOG-07 section 5):

* ``0`` -- ``sq += tl.sum(af*af, axis=1)`` per k-step; the blog's pseudocode.
* ``1`` -- ``sqt += af*af`` into a [BLOCK_M, BLOCK_K] fp32 tile, one reduce after the
  loop: no per-step cross-lane reduction, but BLOCK_M*BLOCK_K extra fp32 registers.
* ``2`` -- ``sqd += tl.dot(a*a, ones[BLOCK_K, 16])``: the sum of squares runs on the
  **tensor core**, so ``a`` never leaves its dot-operand layout and no cross-lane
  reduction or layout conversion is emitted.  Costs ``16/BLOCK_N`` extra MMA flops, so it
  only pays off for wide tiles.  ``a*a`` rounds to bf16 before the MMA; measured cost is
  2e-6 of extra relative error.
* ``3`` -- ``a`` re-loaded with a second ``tl.load`` and then mode 0.  Intended to test
  whether the dot-operand -> blocked layout conversion is the cost; Triton CSEs the two
  loads, so it compiled to exactly mode 0 on MACA and timed identically.  Kept as
  evidence, and worth re-checking on this stack: whether Triton still CSEs across a
  wgmma-operand use is a backend question, not a language guarantee.

The mode is picked by a documented pre-study (recorded as ``sq_mode_study`` in the result
JSON) and then held FIXED so that the fused and unfused tuning grids have identical size.
**The C500 pick must not be inherited** -- ``a`` lives in a wgmma operand layout here, not
an MFMA one, so the pre-study is re-run on this device.

======================================================================================
Warp specialization -- why H200 is the first device that can test the paper's claim
======================================================================================
The blog's kernels do not merely put the reduction in the k-loop; their own diagram puts
it on **dedicated warps**, so the sum-of-squares occupies issue slots that the MMA warps
were not going to use.  That is the algorithm's precondition, and until this port no
device in the study had it:

* **C500 / Triton 3.0 / MACA** -- no warp specialization, no TMA, no clusters.  The
  reduction *displaced* MMA issue slots instead of hiding behind them.  Measured on the
  router GEMM at prefill_t2048: **+0.78 % arithmetic cost bought -12.3 % throughput**
  (REPORT-lazy-prenorm.md section 5.3), and up to +65-68 % where the GEMM was most
  compute-bound.  The paper's central claim simply did not hold there.
* **RTX 4060 / sm_89** -- ``tl.range(warp_specialize=...)`` exists in Triton 3.6 but
  Ada has no warp-group hardware, so the kwarg compiles and does nothing.  Same
  displacement result (log/LOG-13).
* **H200 / sm_90** -- Hopper has warp groups, ``wgmma`` is asynchronous, and Triton's
  automatic warp specialization targets sm_90+.  **This is the first device in the study
  where the precondition holds**, so it is the first time the technique is being measured
  under the conditions its authors designed it for.

What ``tl.range(warp_specialize=True)`` actually does is worth stating precisely, because
it is *not* literally the blog's diagram: Triton's automatic pass splits the loop into a
**producer** partition (the A/B loads, driven by mbarriers) and a **consumer** partition
(the ``wgmma`` and everything that depends on the loaded tiles -- including our
sum-of-squares).  So the reduction still shares a warp group with the MMA; what it stops
sharing is the load pipeline.  Since ``wgmma`` is async-issue, the consumer group's issue
bandwidth is mostly free, which is the mechanism by which the reduction could become
free.  Whether that is enough is exactly the open question, and it is why this file ships
BOTH variants rather than picking one:

    WARP_SPECIALIZE=False   the control -- byte-for-byte the C500/4060 kernel
    WARP_SPECIALIZE=True    the specialized variant

Both are reachable for both ``FUSE_NORM`` values, so the bench can time the full 2x2 and
the fused/unfused ratio is never contaminated by only one arm getting the new feature.
``ws_evidence()`` below pulls the compiled kernel's metadata and greps its TTGIR for
``warp_specialize`` so the result file can state whether specialization *actually
engaged*, rather than assuming it did because a kwarg was accepted.

The variant is selected by ``caps().warp_specialize`` when the caller passes
``warp_specialize=None`` (auto), and forced when the caller passes True/False.  The bench
must pass explicit values for the A/B comparison -- auto is for the whole-layer runs.
The capability verdict itself is NOT re-derived here: ``kernels/hopper.py`` owns it for
the whole suite, including *which spelling* of warp specialization this Triton has
(``tl.range(warp_specialize=)`` at source level, or ``num_consumer_groups`` launch kwargs
on a forked build).  Both launchers below pass whichever of the two the feature layer
reports, so the same call works on either stack.

Defensive notes for a device nobody can test on
-----------------------------------------------
* The ``warp_specialize=`` kwarg appears **only inside** ``if WARP_SPECIALIZE:``.  Triton's
  frontend visits only the taken side of a constexpr ``if``, so on a Triton whose
  ``tl.range`` predates the kwarg the control path never even parses it.  That is why the
  k-loop body is written out twice; a shared ``@triton.jit`` helper would have been
  prettier, but it would have changed the control arm's codegen, and the control arm has
  to stay identical to the audited kernel for the cross-device comparison to mean
  anything.
* ``num_ctas`` (thread-block clusters) is offered on the **router** GEMM only.
  ``moe_gateup_prenorm_kernel`` has an early ``return`` for out-of-range dispatch blocks;
  a partial-cluster exit against any cluster-scoped barrier the backend might emit is a
  *hang*, not an error, and a hang on an untestable device costs the whole round trip.
  A launch that fails outright is fine (the autotuner records it); a launch that never
  returns is not.
* TMA is not used.  For (a) the A operand is a *gather* (``offs_token // top_k``), which
  a tensor descriptor cannot express; for (b) B is 3 MB and L2-resident, so there is
  nothing for a bulk copy to win.  ``caps().tma`` is still reported so the result file
  records that the feature was present and deliberately unused.
* **Expect the tile shape to gate the effect, and record it.**  ``wgmma`` is a
  *warpgroup* instruction: one warpgroup is 4 warps and it wants ``BLOCK_M >= 64``.  The
  router grid inherited from C500 sweeps ``BLOCK_M`` down to 16, and at decode the tuner
  chose small tiles precisely to get any CTAs at all.  Those configs very likely lower to
  ``mma.sync`` instead, and the "the consumer group's issue bandwidth is mostly free"
  argument above is an argument about ``wgmma`` -- it does not apply to a synchronous
  MMA.  So a null result at decode would be evidence about the *tile*, not about the
  technique, and ``ws_evidence()`` (which reports ``shared``/``num_warps`` and whether
  the TTGIR mentions specialization) is what separates the two.  This is the single most
  likely way to misread this experiment.
"""

from __future__ import annotations

import warnings

import torch
import triton
import triton.language as tl

# ======================================================================================
# Hopper feature access -- exactly ONE runtime-detected verdict for the whole suite
#
# Two probes with deliberately different failure policies, and the difference matters:
#
#   * HARDWARE CONSTANTS (SM count, SMEM ceiling, warp width, L2) build the autotuning
#     grids.  A degraded probe there must **crash** -- the C500 study's fallback to
#     hardcoded defaults would have built C500-shaped grids inside a file labelled H200,
#     and grids that prune the two arms unequally move the ratio that is this study's only
#     output.  That probe lives in the harness (`glm52_h200.config.env()`), not here.
#   * FEATURE CAPABILITIES (warp specialization, TMA, clusters) select between two code
#     paths that are both correct.  There the safe degradation is **"feature off"**: the
#     classic arm runs, the measurement is still valid, and the reason is recorded.
#     Crashing on an unreadable preflight would strand a run that could have produced the
#     control numbers.
#
# `kernels/hopper.py` owns that second verdict for every module in the suite.  This file
# deliberately does NOT re-derive it from the preflight JSON: a second opinion can
# disagree with the first, and two disagreeing capability tables is precisely how one arm
# of a pair ends up compiled with a feature the other arm did not get.
# ======================================================================================
try:  # package import (glm52_h200.kernels.lazy_prenorm)
    from . import hopper
except ImportError:  # pragma: no cover -- imported without package context
    try:
        import hopper  # type: ignore
    except ImportError:  # pragma: no cover -- broken checkout, not a device condition
        hopper = None  # type: ignore


def caps():
    """The suite's single capability verdict (`kernels/hopper.py::caps()`)."""
    if hopper is None:  # pragma: no cover
        raise RuntimeError(
            "glm52_h200/kernels/hopper.py is not importable; it owns the Hopper feature "
            "verdict for every kernel module and there is no second copy on purpose"
        )
    return hopper.caps()


def _cap(name: str) -> bool:
    """One capability bit, False whenever the feature layer cannot answer.

    Unknown resolves to "off" and therefore to the classic path, which is always
    executable -- every fusion in this study has a working non-Hopper arm.
    """
    if hopper is None:
        return False
    try:
        return bool(getattr(hopper.caps(), name, False))
    except Exception:  # noqa: BLE001 -- a detection bug costs the feature, not the run
        return False


def _ws_mode() -> str:
    """"range" (source-level constexpr), "launch" (num_consumer_groups kwargs) or "none"."""
    if hopper is None:
        return "none"
    try:
        return str(hopper.ws_mode())
    except Exception:  # noqa: BLE001
        return "none"


def _ws_source() -> str:
    """Where the warp-specialization verdict came from: env / probe / preflight / arch."""
    if hopper is None:
        return "none"
    try:
        return str(hopper.caps().sources.get("warp_specialize", "?"))
    except Exception:  # noqa: BLE001
        return "?"


# The most recent resolution, so a bench can record what the kernels actually did without
# threading return values through every launcher (they return the CompiledKernel, which
# the F11 bench already consumes for register stats).
LAST_WS_DECISION: dict = {"requested": None, "flag": False, "kwargs": {},
                          "why": "not yet resolved"}
LAST_CTAS_DECISION: dict = {"requested": 1, "used": 1, "why": "not yet resolved"}


def warp_specialize_available() -> bool:
    """True iff a warp-specialized launch is expected to work here, by either mechanism.

    Call this before building a grid that sweeps warp specialization, so the bench can
    *record* that the axis was dropped rather than silently tuning two identical halves of
    a grid.  `hopper.ws_choices()` is the canonical way to build that axis -- both arms of
    a pair must call it, so the grid is pruned identically on both sides.
    """
    return bool(_cap("warp_specialize") and _ws_mode() in ("range", "launch"))


def resolve_warp_specialize(
    cfg: "dict | None" = None, explicit: "bool | None" = None
) -> "tuple[bool, dict, str]":
    """Decide warp specialization for one launch.  Returns (constexpr flag, kwargs, why).

    Precedence: explicit argument > `cfg["WARP_SPECIALIZE"]` > `caps().warp_specialize`
    (auto).  The two return values are NOT redundant -- the two mechanisms live in
    different places.  Upstream Triton spells warp specialization at *source* level
    (`tl.range(..., warp_specialize=True)`), so it has to reach the compiler as a
    constexpr; the older forked spelling is a pair of *launch* kwargs.  In "launch" mode
    the flag is False and the kwargs are non-empty, and the kernel runs its classic
    mainloop -- which is correct, not a downgrade.

    A request that cannot be honoured degrades to the control path with a warning.  The
    alternative is a `KeyError: 'Keyword argument num_consumer_groups was specified but
    unrecognised'` that kills an entire autotune on a device we cannot iterate on.
    """
    cfg = cfg or {}
    requested = explicit
    if requested is None:
        requested = cfg.get("WARP_SPECIALIZE", cfg.get("WS"))
    auto = requested is None
    if auto:
        requested = _cap("warp_specialize")

    flag, kwargs = False, {}
    if not requested:
        why = "auto: caps say unavailable" if auto else "off (requested)"
    elif hopper is None:
        why = "refused: kernels/hopper.py not importable"
    else:
        why = ""
        try:
            flag = bool(hopper.ws_source_flag(True))
            kwargs = dict(hopper.ws_kwargs(True))
        except Exception as exc:  # noqa: BLE001 -- a detection bug costs the feature,
            # not the run: "cannot tell" resolves to the classic path, which is correct.
            flag, kwargs = False, {}
            why = f"refused: hopper raised {type(exc).__name__}: {exc}"[:160]
        if flag or kwargs:
            why = (f"on ({'auto' if auto else 'requested'}; mode={_ws_mode()}, "
                   f"source={_ws_source()})")
        elif not why:
            why = f"refused: hopper says warp specialization unusable (mode={_ws_mode()})"

    if requested and not (flag or kwargs) and not auto:
        warnings.warn(f"WARP_SPECIALIZE=True refused -- {why}", stacklevel=2)
    LAST_WS_DECISION.update(
        {"requested": bool(requested), "flag": flag, "kwargs": dict(kwargs), "why": why}
    )
    return flag, kwargs, why


def resolve_num_ctas(cfg: "dict | None" = None) -> "tuple[int, dict, str]":
    """Decide the thread-block cluster width for one launch.  Returns (n, kwargs, why).

    Delegated to `hopper.cluster_kwargs`, which is also the authority on the launch
    semantics: `num_ctas` does NOT change the grid you pass -- Triton multiplies gridDimX
    internally, so one Triton program becomes one cluster of n CTAs.  No ceiling is
    invented here; an over-large cluster is left to fail at launch and be recorded as a
    failed config, exactly like any other tile parameter.  Inventing a limit would prune
    the grid from a literal, which is the one thing this suite has agreed never to do.
    """
    cfg = cfg or {}
    try:
        n = int(cfg.get("num_ctas", 1) or 1)
    except (TypeError, ValueError):
        n = 1
    if n <= 1:
        used, kwargs, why = 1, {}, "off"
    elif hopper is None:
        used, kwargs, why = 1, {}, "refused: kernels/hopper.py not importable"
    else:
        kwargs = dict(hopper.cluster_kwargs(n))
        used = int(kwargs.get("num_ctas", 1))
        why = f"on ({used} CTAs/cluster)" if kwargs else "refused: caps say no clusters"
    LAST_CTAS_DECISION.update({"requested": n, "used": used, "why": why})
    return used, kwargs, why


def caps_report() -> dict:
    """Everything about feature selection that belongs in the result JSON."""
    d = {"triton_version": getattr(triton, "__version__", None)}
    try:
        d["hopper_caps"] = hopper.caps_dict() if hopper is not None else None
    except Exception as exc:  # noqa: BLE001
        d["hopper_caps"] = {"_error": f"{type(exc).__name__}: {exc}"}
    d["hopper_importable"] = hopper is not None
    d["ws_mode"] = _ws_mode()
    d["last_warp_specialize_decision"] = dict(LAST_WS_DECISION)
    d["last_num_ctas_decision"] = dict(LAST_CTAS_DECISION)
    return d


def ws_evidence(compiled) -> dict:
    """Did warp specialization actually *engage*, or was the kwarg merely accepted?

    On sm_89 the preflight's `tl.range(warp_specialize=True)` probe PASSES and specializes
    nothing -- Ada has no warp groups.  So "the launch worked" is not evidence.  What is
    evidence: the partitioned form shows up in the TTGIR, and a specialized kernel asks
    the driver for extra warps and usually more shared memory than the same tile without
    it.  Pure string/attribute inspection of an already-compiled kernel; launches nothing.
    """
    out: dict = {}
    md = getattr(compiled, "metadata", None)
    for f in ("name", "num_warps", "num_stages", "num_ctas", "shared", "global_scratch_size",
              "n_regs", "n_spills", "cluster_dims", "maxnreg"):
        v = getattr(md, f, None)
        if v is not None:
            out[f] = v
    asm = getattr(compiled, "asm", None)
    if isinstance(asm, dict):
        for key in ("ttgir", "ttir", "ptx"):
            text = asm.get(key)
            if isinstance(text, str):
                out[f"{key}_mentions_warp_specialize"] = (
                    "warp_specialize" in text or "ws.op" in text
                )
                if key == "ptx":
                    # A specialized Hopper kernel diverges on warp-group id; this is a
                    # cheap corroborating signal, not a proof.
                    out["ptx_mentions_warpgroup"] = "wgmma" in text or "%warpid" in text
    return out


# ======================================================================================
# offline weight folding  (load-time transform, never inside a timed region)
# ======================================================================================
def fold_weight_rowmajor(b: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """B is [K, N] (contraction dim first): scale ROW k by w[k]."""
    return (b.float() * w.float()[:, None]).to(b.dtype)


def fold_weight_nk(bt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """B^T is [..., N, K] (nn.Linear / sglang w13 layout): scale COLUMN k by w[k]."""
    return (bt.float() * w.float()).to(bt.dtype)


# ======================================================================================
# (a) routed-expert w13 grouped GEMM, sglang `fused_moe_kernel` shape + lazy pre-norm
# ======================================================================================
@triton.jit
def moe_gateup_prenorm_kernel(
    a_ptr,  # [T, K] bf16 -- x2 (FUSE_NORM=False) or h1 (FUSE_NORM=True)
    b_ptr,  # [E, N, K] bf16 -- w13 (raw) or w13 folded with the rmsnorm weight
    c_ptr,  # [T*top_k, N] bf16
    rstd_ptr,  # [T] fp32 -- only read when USE_RSTD
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
    FUSE_NORM: tl.constexpr,
    SQ_MODE: tl.constexpr,
    USE_RSTD: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    # ---- grouped pid swizzle (sglang) ------------------------------------------------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- dispatch --------------------------------------------------------------------
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Padded dispatch slots carry the sentinel `num_valid_tokens`.  The MACA pipeliner
    # emitted speculative (unpredicated) prologue loads, so the sentinel row must not even
    # be *addressed*; clamping to row 0 keeps every address in range and `token_mask`
    # still discards the value.  Kept on Hopper: warp specialization moves the loads onto
    # a producer partition that runs AHEAD of the consumer's predication, which is the
    # same speculative-address hazard by a different mechanism, and the clamp costs one
    # `select`.
    safe_token = tl.where(token_mask, offs_token, 0)
    a_ptrs = a_ptr + (
        safe_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if FUSE_NORM:
        if SQ_MODE == 1:
            sqt = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        elif SQ_MODE == 2:
            sqd = tl.zeros((BLOCK_SIZE_M, 16), dtype=tl.float32)
            sq_ones = tl.full((BLOCK_SIZE_K, 16), 1.0, dtype=a_ptr.dtype.element_ty)
        else:
            sq = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # ---- mainloop ---------------------------------------------------------------------
    # The body appears TWICE on purpose.  `warp_specialize=` must be written at the
    # `tl.range` call site, and it must not appear in the source at all on a Triton that
    # predates it -- Triton visits only the taken side of a constexpr `if`, so the control
    # path never parses the kwarg.  Factoring the body into a shared @triton.jit helper
    # would have perturbed the control arm's codegen, and the control arm has to stay
    # identical to the audited C500/4060 kernel for the cross-device comparison to hold.
    # Any edit below must be made to BOTH copies.
    if WARP_SPECIALIZE:
        for k_start in tl.range(0, K, BLOCK_SIZE_K, warp_specialize=True):
            if even_Ks:
                a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                b = tl.load(b_ptrs)
            else:
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                    other=0.0,
                )
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

            acc += tl.dot(a, b)
            if FUSE_NORM:
                if SQ_MODE == 2:
                    sqd += tl.dot(a * a, sq_ones)
                elif SQ_MODE == 3:
                    a2 = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)
                    sq += tl.sum(a2 * a2, axis=1)
                else:
                    af = a.to(tl.float32)
                    if SQ_MODE == 1:
                        sqt += af * af
                    else:
                        sq += tl.sum(af * af, axis=1)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
    else:
        for k_start in range(0, K, BLOCK_SIZE_K):
            if even_Ks:
                a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                b = tl.load(b_ptrs)
            else:
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                    other=0.0,
                )
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

            acc += tl.dot(a, b)
            if FUSE_NORM:
                if SQ_MODE == 2:
                    # sum of squares on the TENSOR CORE: (a*a) @ ones.  Keeps `a` in its
                    # dot-operand layout, so no cross-lane reduction and no layout
                    # conversion; costs 16/BLOCK_N extra MMA flops instead.
                    sqd += tl.dot(a * a, sq_ones)
                elif SQ_MODE == 3:
                    # A loaded a SECOND time, so the copy feeding the reduction gets a
                    # plain blocked layout (the extra read is L2-resident).
                    a2 = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)
                    sq += tl.sum(a2 * a2, axis=1)
                else:
                    af = a.to(tl.float32)
                    if SQ_MODE == 1:
                        sqt += af * af
                    else:
                        sq += tl.sum(af * af, axis=1)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    # ---- lazy epilogue: the whole normalization, deferred past the k-loop -------------
    if FUSE_NORM:
        if SQ_MODE == 1:
            sq = tl.sum(sqt, axis=1)
        elif SQ_MODE == 2:
            sq = tl.sum(sqd, axis=1) * 0.0625  # all 16 columns hold the same value
        rstd = 1.0 / tl.sqrt(sq / K + eps)
        acc = acc * rstd[:, None]
    elif USE_RSTD:
        # "half-fused" variant: rstd came from a tiny separate reduction kernel, so the
        # k-loop is byte-for-byte the unfused one and only the epilogue scale is added.
        rstd = tl.load(rstd_ptr + safe_token // top_k, mask=token_mask, other=1.0)
        acc = acc * rstd[:, None]

    out = acc.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * safe_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, out, mask=c_mask)


# ======================================================================================
# (b) dense router GEMM  [M, K] @ [K, N] -> fp32 logits, with lazy pre-norm
# ======================================================================================
@triton.jit
def router_gemm_kernel(
    a_ptr,  # [M, K] bf16 -- x2 (FUSE_NORM=False) or h1 (FUSE_NORM=True)
    b_ptr,  # [K, N] bf16 -- gate weight^T (raw) or folded with the rmsnorm weight
    c_ptr,  # [M, N] fp32 -- router logits (moe_router_dtype == float32)
    rstd_ptr,  # [M] fp32 -- only read when USE_RSTD
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    even_Ks: tl.constexpr,
    FUSE_NORM: tl.constexpr,
    SQ_MODE: tl.constexpr,
    USE_RSTD: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = offs_m < M
    safe_m = tl.where(m_mask, offs_m, 0)  # never address past the end of A
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (safe_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if FUSE_NORM:
        if SQ_MODE == 1:
            sqt = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        elif SQ_MODE == 2:
            sqd = tl.zeros((BLOCK_SIZE_M, 16), dtype=tl.float32)
            sq_ones = tl.full((BLOCK_SIZE_K, 16), 1.0, dtype=a_ptr.dtype.element_ty)
        else:
            sq = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # See the note on the twin loop in `moe_gateup_prenorm_kernel`: the body is duplicated
    # so that `warp_specialize=` is never parsed by a Triton that does not have it, and so
    # the control arm's codegen is untouched.  Edit both copies together.
    if WARP_SPECIALIZE:
        for k_start in tl.range(0, K, BLOCK_SIZE_K, warp_specialize=True):
            if even_Ks:
                a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
                b = tl.load(b_ptrs)
            else:
                a = tl.load(
                    a_ptrs,
                    mask=m_mask[:, None] & (offs_k[None, :] < K - k_start),
                    other=0.0,
                )
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

            acc += tl.dot(a, b)
            if FUSE_NORM:
                if SQ_MODE == 2:
                    sqd += tl.dot(a * a, sq_ones)
                elif SQ_MODE == 3:
                    a2 = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
                    sq += tl.sum(a2 * a2, axis=1)
                else:
                    af = a.to(tl.float32)
                    if SQ_MODE == 1:
                        sqt += af * af
                    else:
                        sq += tl.sum(af * af, axis=1)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
    else:
        for k_start in range(0, K, BLOCK_SIZE_K):
            if even_Ks:
                a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
                b = tl.load(b_ptrs)
            else:
                a = tl.load(
                    a_ptrs,
                    mask=m_mask[:, None] & (offs_k[None, :] < K - k_start),
                    other=0.0,
                )
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

            acc += tl.dot(a, b)
            if FUSE_NORM:
                if SQ_MODE == 2:
                    # sum of squares on the TENSOR CORE: (a*a) @ ones.  Keeps `a` in its
                    # dot-operand layout, so no cross-lane reduction and no layout
                    # conversion; costs 16/BLOCK_N extra MMA flops instead.
                    sqd += tl.dot(a * a, sq_ones)
                elif SQ_MODE == 3:
                    # A loaded a SECOND time, so the copy feeding the reduction gets a
                    # plain blocked layout (the extra read is L2-resident).
                    a2 = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
                    sq += tl.sum(a2 * a2, axis=1)
                else:
                    af = a.to(tl.float32)
                    if SQ_MODE == 1:
                        sqt += af * af
                    else:
                        sq += tl.sum(af * af, axis=1)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if FUSE_NORM:
        if SQ_MODE == 1:
            sq = tl.sum(sqt, axis=1)
        elif SQ_MODE == 2:
            sq = tl.sum(sqd, axis=1) * 0.0625  # all 16 columns hold the same value
        rstd = 1.0 / tl.sqrt(sq / K + eps)
        acc = acc * rstd[:, None]
    elif USE_RSTD:
        rstd = tl.load(rstd_ptr + safe_m, mask=m_mask, other=1.0)
        acc = acc * rstd[:, None]

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * safe_m[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & (offs_cn[None, :] < N))


# ======================================================================================
# "half-fused" helper: rstd-only reduction  ([T, K] bf16 -> [T] fp32)
#
# This is the third point on the design axis and it exists because of what the
# measurements below show.  Traffic per activation pass (`act = T*H*2`):
#
#   unfused      norm kernel reads act, writes act ; GEMM reads act        -> 3 act
#   rstd-only    rstd kernel reads act             ; GEMM reads act        -> 2 act
#   lazy prenorm                                     GEMM reads act        -> 1 act
#
# so it captures 2/3 of the fusion's byte saving while leaving the GEMM's k-loop
# byte-for-byte identical to the unfused one -- no fp32 conversion of the A tile, no
# per-step reduction, no extra live registers in the mainloop.  On C500 it was the variant
# actually worth shipping (LOG-07 section 9); it is the control that tells us how much of
# any H200 win comes from warp specialization rather than from removing bytes.
# ======================================================================================
@triton.jit
def rstd_kernel(
    X,  # [T, N] bf16
    RSTD,  # [T]    fp32
    stride_x,
    T,
    N,
    eps,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    rmask = rows < T
    acc = tl.zeros([ROWS], dtype=tl.float32)
    for t in tl.static_range(N_TILES):
        cols = t * BLOCK_N + tl.arange(0, BLOCK_N)
        m = rmask[:, None] & (cols[None, :] < N)
        x = tl.load(X + rows[:, None] * stride_x + cols[None, :], mask=m, other=0.0)
        xf = x.to(tl.float32)
        acc += tl.sum(xf * xf, axis=1)
    tl.store(RSTD + rows, 1.0 / tl.sqrt(acc / N + eps), mask=rmask)


def launch_rstd(x, rstd, cfg, eps: float = 1e-5):
    T, N = x.shape
    bn = cfg["BLOCK_N"]
    grid = (triton.cdiv(T, cfg["ROWS"]),)
    return rstd_kernel[grid](
        x,
        rstd,
        x.stride(0),
        T,
        N,
        eps,
        ROWS=cfg["ROWS"],
        BLOCK_N=bn,
        N_TILES=triton.cdiv(N, bn),
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


# ======================================================================================
# SMEM pre-filter and thin launchers
# ======================================================================================
def smem_bytes(cfg: dict) -> int:
    """Triton pipeline SMEM footprint for one mainloop tile pair, in bytes.

    **How many buffers Triton stages is version-dependent and must never be hardcoded.**
    Triton 3.0 (the C500 stack) allocated `num_stages`; Triton 3.6 allocates
    `num_stages - 1` with a floor of 2 -- verified on sm89 by launching 68 configs and
    reading `CompiledKernel.metadata.shared`.  The 3.0 formula over-predicts by 1.33-1.5x,
    which rejects tiles the hardware can actually run, and it does not reject them equally
    on the two arms.  So: defer to the harness's model when it exists, and treat even that
    as a *pre-filter* -- `measured_smem()` below is the authority, because it reads what
    the compiler actually asked for.

    Warp specialization is not modelled here at all.  A specialized kernel needs extra
    mbarriers and may stage differently; the honest procedure is to let the config compile
    and record `metadata.shared`, exactly as the F6 and F11 SMEM findings were obtained.
    """
    try:
        from glm52_h200 import config as _C  # type: ignore

        return int(
            _C.smem_stage_bytes(
                cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"], cfg["num_stages"]
            )
        )
    except Exception:  # noqa: BLE001 -- harness may not exist yet; model it locally
        return max(2, int(cfg["num_stages"]) - 1) * 2 * cfg["BLOCK_K"] * (
            cfg["BLOCK_M"] + cfg["BLOCK_N"]
        )


def measured_smem(compiled) -> "int | None":
    """What the compiler actually allocated -- the only number that is not a model."""
    return getattr(getattr(compiled, "metadata", None), "shared", None)


def launch_moe_gateup(
    a,
    b,
    c,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    num_valid_tokens,
    top_k,
    cfg,
    fuse_norm: bool,
    eps: float = 1e-5,
    sq_mode: int = 0,
    rstd=None,
    warp_specialize: "bool | None" = None,
):
    """`a`: [T,K] bf16.  `b`: [E,N,K] bf16.  `c`: [T*top_k, N] bf16.

    `warp_specialize`: None = auto from `caps()`, True/False = forced.  The bench forces
    both values to get the WS-vs-control comparison; auto is for whole-layer runs.

    No `num_ctas` here on purpose -- this kernel has an early `return` for out-of-range
    dispatch blocks, and a partial-cluster exit against a cluster-scoped barrier hangs
    rather than errors.  See the module docstring.
    """
    N = c.shape[1]
    K = a.shape[1]
    EM = sorted_token_ids.shape[0]
    ws, ws_kw, _ = resolve_warp_specialize(cfg, warp_specialize)
    grid = (triton.cdiv(EM, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    return moe_gateup_prenorm_kernel[grid](
        a,
        b,
        c,
        rstd if rstd is not None else c,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_valid_tokens,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(2),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        eps,
        BLOCK_SIZE_M=cfg["BLOCK_M"],
        BLOCK_SIZE_N=cfg["BLOCK_N"],
        BLOCK_SIZE_K=cfg["BLOCK_K"],
        GROUP_SIZE_M=cfg["GROUP_M"],
        top_k=top_k,
        compute_type=tl.bfloat16,
        even_Ks=(K % cfg["BLOCK_K"] == 0),
        FUSE_NORM=fuse_norm,
        SQ_MODE=sq_mode,
        USE_RSTD=rstd is not None,
        WARP_SPECIALIZE=ws,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        # `**ws_kw` is empty on upstream Triton (where the flag above is the mechanism)
        # and carries num_consumer_groups / num_buffers_warp_spec on a forked one.  It is
        # never passed speculatively: an unrecognised kwarg is a hard KeyError, not a
        # soft failure, which is why hopper.py decides it from a probe.
        **ws_kw,
    )


def launch_router(
    a,
    b,
    c,
    cfg,
    fuse_norm: bool,
    eps: float = 1e-5,
    sq_mode: int = 0,
    rstd=None,
    warp_specialize: "bool | None" = None,
):
    """`a`: [M,K] bf16.  `b`: [K,N] bf16.  `c`: [M,N] fp32.

    `warp_specialize`: None = auto from `caps()`, True/False = forced.
    `cfg["num_ctas"]`: Hopper thread-block cluster width, collapsed to 1 (and the kwarg
    omitted entirely) unless `hopper.caps().clusters`.  B is the 3 MB router gate read by
    every CTA, which is the one place in this study where cluster-level reuse of a weight
    tile could plausibly pay.  The grid is NOT divided by it: Triton multiplies gridDimX
    internally, so one Triton program becomes one cluster.
    """
    M, K = a.shape
    N = c.shape[1]
    ws, ws_kw, _ = resolve_warp_specialize(cfg, warp_specialize)
    # `num_ctas` is only *mentioned* when a cluster was actually resolved, so the default
    # launch is byte-for-byte the audited one on any Triton, however old.
    _, ct_kw, _ = resolve_num_ctas(cfg)
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    return router_gemm_kernel[grid](
        a,
        b,
        c,
        rstd if rstd is not None else c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        eps,
        BLOCK_SIZE_M=cfg["BLOCK_M"],
        BLOCK_SIZE_N=cfg["BLOCK_N"],
        BLOCK_SIZE_K=cfg["BLOCK_K"],
        GROUP_SIZE_M=cfg["GROUP_M"],
        even_Ks=(K % cfg["BLOCK_K"] == 0),
        FUSE_NORM=fuse_norm,
        SQ_MODE=sq_mode,
        USE_RSTD=rstd is not None,
        WARP_SPECIALIZE=ws,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        **ws_kw,
        **ct_kw,
    )


def launch_flags(cfg: dict, fuse_norm: bool, sq_mode: int = 0, rstd=None,
                 warp_specialize: "bool | None" = None) -> dict:
    """Everything a launch would resolve, without launching -- for the result JSON.

    Audit lesson 7: `n_tried` / `n_failed` per arm is what makes an unfair comparison
    detectable after the fact, and that only works if the *effective* flags are recorded
    too.  Warp specialization and `num_ctas` can both be silently downgraded (feature
    absent, wrong Triton spelling), so a config table on its own does not say what ran.
    """
    ws, ws_kw, ws_why = resolve_warp_specialize(cfg, warp_specialize)
    nctas, _, ct_why = resolve_num_ctas(cfg)
    return {
        "BLOCK_M": cfg.get("BLOCK_M"),
        "BLOCK_N": cfg.get("BLOCK_N"),
        "BLOCK_K": cfg.get("BLOCK_K"),
        "GROUP_M": cfg.get("GROUP_M"),
        "num_warps": cfg.get("num_warps"),
        "num_stages": cfg.get("num_stages"),
        "FUSE_NORM": bool(fuse_norm),
        "SQ_MODE": int(sq_mode),
        "USE_RSTD": rstd is not None,
        "WARP_SPECIALIZE": ws,
        "WARP_SPECIALIZE_kwargs": ws_kw,
        "WARP_SPECIALIZE_why": ws_why,
        "num_ctas": nctas,
        "num_ctas_why": ct_why,
        "smem_model_bytes": smem_bytes(cfg) if "BLOCK_M" in cfg else None,
    }
