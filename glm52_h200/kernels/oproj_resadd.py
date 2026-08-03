"""Fusion #1 -- o_proj GEMM + residual add (dense GEMM epilogue fusion).  H200 / sm_90.

Ported from `glm52/kernels/oproj_resadd.py`.  The fairness contract is unchanged and is the
reason the file is shaped the way it is:

    oproj_gemm_kernel(..., FUSE_RESADD=True )   -> h1 = A @ B + residual   (fused)
    oproj_gemm_kernel(..., FUSE_RESADD=False)   -> c  = A @ B              (unfused GEMM)
    epilogue_kernel(..., HAS_RES=True)          -> h1 = c + residual       (split-out add)
    epilogue_kernel(..., HAS_RES=False)         -> h1 = c                  (split-K cast only)

ONE GEMM source; `tl.constexpr` flags select the epilogue, so the two arms cannot diverge
except in the fused work itself.  Only the *mapping* -- BLOCK_M/N/K, GROUP_M, SPLIT_K,
num_warps, num_stages, and now the H200 axes -- may differ between the sides, and each side
is tuned independently over the same offered grid.

Split-K note (unchanged): with SPLIT_K > 1 the GEMM accumulates into an fp32 buffer with
`tl.atomic_add`, so the chain becomes [zero fp32 buf, gemm, epilogue-cast].  Both sides pay
that structure identically; the fused side's epilogue is a pure cast (2 passes) while the
unfused side's also reads the residual (3 passes).

--------------------------------------------------------------------------------------
The H200 axes, and why every one of them is a RUNTIME choice
--------------------------------------------------------------------------------------
Nothing sm_90-specific is decided at authoring time.  `kernels/hopper.py` is the single
place that decides what this process may emit; this module only adapts.  The cfg keys it
forwards are advertised in `H200_CFG_KEYS`, which is what `bench.widen()` reads before it
adds any overlay -- so on a stack without these features the grids are byte-identical to
the classic ones.

The 2026-08-03 H200 preflight (`preflight_h200.json`) turned four of those runtime
questions into measured answers.  They are recorded here because they decide which paths
are worth carrying -- NOT because anything below now assumes them; every gate is still a
`caps()` read at launch time:

  * `tl.range(..., warp_specialize=True)` compiles on this stack.  The launch-kwarg
    spelling does NOT: triton 3.6.0 answers "Keyword argument num_consumer_groups was
    specified but unrecognised", and an unrecognised kwarg is a hard `KeyError`, not a
    soft failure.  `_ws_launch` therefore emits that spelling only when
    `hopper.ws_mode()` says the stack takes it, and RAISES when a cfg pins it anyway --
    forwarding it blind would turn every warp-spec config on this device into an opaque
    launch error rather than a measurement.
  * `num_ctas > 1` compiles, so thread-block clusters are a real axis here.
  * TMA's preflight probe was a FALSE NEGATIVE.  It fed a host-side `TensorDescriptor`
    into `tl.make_tensor_descriptor()` -- the DEVICE-side constructor, which takes a raw
    pointer -- and mixing the two APIs is a CompilationError on any hardware, so that
    probe measured the probe.  Both spellings this file emits (a descriptor argument
    consumed as `desc.load(...)`, and `tl.make_tensor_descriptor(ptr, shape=..., ...)`)
    were verified to compile, launch and produce correct values on triton 3.6.0.  The
    gate is still `hopper.caps().tma`, so a stale probe costs the TMA arm and nothing
    else -- but it costs it silently, which is why `h200_report()` exists.
  * The device's own best bf16 Triton GEMM is BM128 BN256 BK64 GM8 w8 s3 at 788 TF/s
    (96 % of cuBLAS).  Nothing here caps BLOCK_*; the only ceiling is `smem_limit()`, and
    under the corrected staging rule (see `smem_stage_bytes`) that shape models at
    147456 B against a 232448 B opt-in ceiling, as does BM256 BN256 BK64 s3 at 196608 B.
    Both are expressible and both fit.

  `USE_TMA` / `TMA_A` / `TMA_B`
      Fetch that operand through a tensor-descriptor box instead of a masked pointer tile.
      A and B are separate axes on purpose: at decode M is 1..1024 while K is 32768, so the
      weight is the entire bandwidth story and a descriptor on B is worth having even where
      a box over A is not.  `USE_TMA=True` means "wherever the layout allows"; an explicit
      `TMA_A`/`TMA_B` that cannot be honoured RAISES.  If no operand can be described the
      config is a duplicate of the classic one and it raises rather than being timed and
      recorded as a TMA result.
  `TMA_MODE`  ("host" | "device", default from `hopper.caps().tma_form()`)
      Host-side descriptors make Triton's launcher call `cuTensorMapEncodeTiled` on EVERY
      launch; at decode a kernel resolves to 9-17 CUDA-event ticks, so that per-launch host
      tax lands inside the measured window and is charged to the TMA arm alone.  The
      device-side spelling (`tl.make_tensor_descriptor`) pays global scratch instead.  Both
      are offered; the tuner decides, which is the study's method.
  `warp_specialize` (+ `num_consumer_groups` / `num_buffers_warp_spec` on forks that use
      the launch-kwarg spelling -- measured DEAD on this H200's triton 3.6.0)
      `tl.range(..., warp_specialize=True)` on the k-loop.  The loop header is written out
      twice under a `tl.constexpr` `if` rather than forwarding the flag into the kwarg:
      `warp_specialize=` lowers to an MLIR attribute, some Triton builds reject a
      `tl.constexpr` there, and a rejection would take the WS=False (classic) path down
      with it.  The loop *body* lives in `_gemm_step` alone, so the two copies cannot
      drift -- the only property of them that could move the ratio.
  `num_ctas`
      Thread-block cluster width, forwarded straight to the Triton launch.

Larger tiles: nothing in this file caps BLOCK_*.  Hopper opts in to ~228 KB of shared
memory per SM and `smem_limit()` reads that number off the device probe; the C500 study's
hardcoded 65536 is exactly the bug that pruned one arm of a pair harder than the other.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ======================================================================================
# H200 bridge.
#
# `kernels/hopper.py` is the ONE place that decides which sm_90 mechanism this process may
# emit, and it probes in a subprocess because a malformed TMA descriptor raises an
# ASYNCHRONOUS illegal access that poisons the CUDA context -- and nobody can retry on this
# machine.  Everything below is a thin, defensive adapter: if that module is missing or
# raises, every H200 path reports unavailable and the classic mainloop (the one C500 and
# the RTX 4060 measured) runs unchanged.
# ======================================================================================
try:
    from . import hopper as _H
except Exception:  # noqa: BLE001 -- a missing helper must cost the H200 path, not the run
    try:
        from glm52_h200.kernels import hopper as _H  # type: ignore
    except Exception:  # noqa: BLE001
        _H = None

# cfg keys this module's launchers actually forward.  `bench.widen()` reads this tuple and
# refuses to overlay an axis the kernel cannot carry, so advertising is a promise.
H200_CFG_KEYS = (
    "USE_TMA",
    "TMA_A",
    "TMA_B",
    "TMA_MODE",
    "warp_specialize",
    "num_consumer_groups",
    "num_buffers_warp_spec",
    "num_ctas",
)


class HopperPathUnavailable(RuntimeError):
    """A config asked for an sm_90 mapping this stack cannot actually deliver.

    Raised, never swallowed.  Accepting the key and compiling the classic mainloop would
    put a row in the result table labelled `warp_specialize=True` / `USE_TMA=True` that
    measured the classic loop -- a fabricated measurement, and precisely what
    `tl.range(warp_specialize=True)` does on sm_89, where it compiles, runs, and is
    silently not the Hopper producer/consumer scheme.  The autotuner catches this and
    counts the config in that arm's `n_failed`, which is the number that makes an unfair
    comparison visible after the fact.
    """


class TmaUnsupported(HopperPathUnavailable):
    """No operand of this launch could be described by a TMA box."""


class _NoHopper:
    """Stand-in carrying `HopperCaps`'s attribute surface, all negative."""

    tma = warp_specialize = clusters = wgmma = False
    tma_host = tma_device = False
    ws_mode = "none"
    cc = (0, 0)
    device_name = ""
    sources: dict = {}

    def tma_form(self) -> str:
        return "none"


_NO_HOPPER = _NoHopper()


def caps():
    """`kernels.hopper.caps()`, or an all-False stand-in when that module is unavailable."""
    if _H is None:
        return _NO_HOPPER
    try:
        return _H.caps()
    except Exception:  # noqa: BLE001 -- a capability probe must never be fatal
        return _NO_HOPPER


_CFG_MOD = None
_CFG_TRIED = False


def _config():
    """`glm52_h200.config`, or None.  Cached -- this sits on the grid-construction path."""
    global _CFG_MOD, _CFG_TRIED
    if not _CFG_TRIED:
        _CFG_TRIED = True
        try:
            from .. import config as _C  # type: ignore
        except Exception:  # noqa: BLE001 -- also importable as a top-level package
            try:
                from glm52_h200 import config as _C  # type: ignore
            except Exception:  # noqa: BLE001
                _C = None
        _CFG_MOD = _C
    return _CFG_MOD


def smem_limit() -> int:
    """Per-block shared-memory ceiling in bytes, from the device probe -- never a literal.

    The H200 opts in to 232448 B per block (233472 per SM), which is why nothing in this
    file caps BLOCK_*; this number is the only ceiling.  If the probe cannot supply it we
    RAISE rather than substitute a plausible constant: a grid built from the wrong ceiling
    prunes the two arms unequally, and that is the one failure mode this study cannot
    detect afterwards.
    """
    n, err = 0, ""
    try:
        _C = _config()
        if _C is None:
            raise ImportError("glm52_h200.config not importable")
        n = int(getattr(_C.env(), "smem_bytes", 0) or 0)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    if n <= 0:
        raise RuntimeError(
            "shared-memory ceiling unknown -- refusing to build an autotuning grid.\n"
            f"  glm52_h200.config.env(): {err or 'returned 0'}\n"
            "  A guessed ceiling silently prunes one arm of a fused/unfused pair harder "
            "than the other, which is invisible in the result file."
        )
    return n


def ws_ok(cfg: dict) -> bool:
    """Is `warp_specialize=True` worth OFFERING for this config?

    Triton splits the CTA into producer and consumer warp groups, so a CTA with fewer than
    four warps has nothing to partition and a single-buffered mainloop nothing to overlap.
    Both are Triton-side constraints, not device constants, and are written as such.  The
    bench offers one grid to both arms, so this prunes both identically.
    """
    return (
        bool(getattr(caps(), "warp_specialize", False))
        and int(cfg.get("num_warps", 4)) >= 4
        and int(cfg.get("num_stages", 2)) >= 2
    )


_WS_LAUNCH_KEYS = ("num_consumer_groups", "num_buffers_warp_spec")


def _ws_launch(cfg: dict) -> tuple:
    """(constexpr flag, extra launch kwargs) for this config's warp-specialization request.

    The two spellings live in different places -- the source-level one must reach the
    compiler as a constexpr, the forked-Triton one must reach the launcher as kwargs -- so
    `hopper` answers both and this merges them.  A request neither can serve raises; see
    `HopperPathUnavailable`.

    The launch-kwarg spelling is forwarded ONLY when `hopper.ws_mode()` reports that this
    Triton takes it.  On the measured H200 (triton 3.6.0) it does not: the preflight probe
    `warp_specialize_num_consumer_groups` fails with "Keyword argument num_consumer_groups
    was specified but unrecognised", and Triton raises that as a `KeyError` at launch
    rather than ignoring it.  Copying the key out of the cfg unconditionally -- which is
    what this did -- would therefore turn any config carrying it into an opaque launch
    failure on the device the suite is actually for, WS requested or not.  Note also that
    on a launch-mode stack the kwargs ARE the enabling mechanism (there is no source flag
    to pair them with), so they are forwarded there whether or not `warp_specialize` is
    also set; that is why the gate is the stack's mode and not `want`.
    """
    want = bool(cfg.get("warp_specialize", cfg.get("WARP_SPECIALIZE", False)))
    pinned = {k: int(cfg[k]) for k in _WS_LAUNCH_KEYS if k in cfg}
    mode = str(getattr(caps(), "ws_mode", "none") or "none")
    if pinned and mode != "launch":
        raise HopperPathUnavailable(
            f"cfg pins {'/'.join(sorted(pinned))} but this stack does not accept the "
            f"consumer-group launch-kwarg spelling (ws_mode={mode!r}); Triton would raise "
            "KeyError: 'Keyword argument num_consumer_groups was specified but "
            "unrecognised'.  On this H200 the live spelling is tl.range(warp_specialize=True)."
        )
    kw = dict(pinned) if mode == "launch" else {}
    if not want:
        return False, kw
    flag = bool(_H.ws_source_flag(True)) if _H is not None else False
    if _H is not None:
        for k, v in _H.ws_kwargs(True).items():
            kw.setdefault(k, v)
    if not flag and not kw:
        c = caps()
        raise HopperPathUnavailable(
            "cfg asks for warp_specialize but this stack offers neither "
            "tl.range(warp_specialize=) nor consumer-group launch kwargs "
            f"(ws_mode={getattr(c, 'ws_mode', 'none')!r}, cc={getattr(c, 'cc', (0, 0))}, "
            f"sources={getattr(c, 'sources', {})})"
        )
    return flag, kw


# --- TMA -------------------------------------------------------------------------------
_TMA_OFF, _TMA_HOST, _TMA_DEVICE = 0, 1, 2
_TMA_MEMO: dict = {}
_MEMO_CAP = 4096


def clear_caches() -> None:
    """Drop memoized descriptors.  Call when a big weight tensor is freed.

    A `TensorDescriptor` holds a reference to the tensor it describes, so the memo keeps
    that buffer alive.  Harmless inside one bench (the tensors are allocated once for all
    regimes); the hook exists so a whole-layer run can release between fusions.
    """
    _TMA_MEMO.clear()


def _tma_reject(t, block) -> "str | None":
    if _H is None:
        return "kernels/hopper.py unavailable"
    try:
        return _H.tma_reject_reason(t, block)
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def _tma_check(t, block, build: bool) -> tuple:
    """(descriptor, reason).  `reason is None` means usable; `descriptor` is None unless
    `build` -- the device-side spelling constructs its own descriptor inside the kernel.

    MEMOIZED, and that is not an optimisation.  Autotuning launches the same (tensor, box)
    pair thousands of times, and both the legality check and
    `TensorDescriptor.from_tensor` are python: tens of microseconds that would land INSIDE
    the timed region and be charged to the TMA arm alone.  At decode a kernel resolves to
    9-17 CUDA-event ticks, so that is not a rounding error -- it is exactly the kind of
    asymmetric harness cost the fairness audit exists to catch.  The key carries the data
    pointer AND the shape/stride/dtype/box, so a reallocated buffer can never reuse a stale
    descriptor.
    """
    key = (t.data_ptr(), tuple(t.shape), tuple(t.stride()), tuple(block), str(t.dtype), build)
    hit = _TMA_MEMO.get(key)
    if hit is None:
        why = _tma_reject(t, block)
        desc = None
        if why is None and build:
            desc = _H.descriptor(t, block)
            if desc is None:
                why = "hopper.descriptor() declined the box"
        hit = (desc, why)
        if len(_TMA_MEMO) >= _MEMO_CAP:
            _TMA_MEMO.clear()
        _TMA_MEMO[key] = hit
    return hit


def _ensure_allocator() -> None:
    """Register Triton's global-scratch allocator before ANY descriptor kernel launches.

    `triton.set_allocator(...)` is required by BOTH spellings on triton 3.6 -- the
    device-side constructor builds its descriptor out of global scratch, and the host-side
    form needs device memory for the tensormap the launcher fills.  Skipping it is the
    classic silent TMA failure: the symptom is a launch error deep inside the driver that
    never says "TMA", so it reads as a flaky config rather than a missing one-line setup.
    `hopper.ensure_allocator()` is idempotent (a dict lookup after the first call), so this
    is called on the TMA planning path unconditionally rather than only where it is
    provably needed.
    """
    if _H is None or not _H.ensure_allocator():
        raise TmaUnsupported(
            "triton.set_allocator is unavailable, so no TMA descriptor kernel can launch "
            f"(hopper.ensure_allocator() -> False, hopper_module={_H is not None})"
        )


def _tma_spelling(cfg: dict) -> int:
    """Host-side descriptor argument vs device-side `tl.make_tensor_descriptor`.

    `hopper.caps().tma_form()` already encodes the preference (device when both work, to
    keep `cuTensorMapEncodeTiled` off the per-launch path at decode).  `TMA_MODE` in the
    cfg pins it, which is how the bench can measure the two spellings against each other
    instead of assuming which one wins.
    """
    c = caps()
    pin = str(cfg.get("TMA_MODE", "") or "").lower()
    form = pin if pin in ("host", "device") else (
        c.tma_form() if hasattr(c, "tma_form") else "none"
    )
    return {"device": _TMA_DEVICE, "host": _TMA_HOST}.get(form, _TMA_OFF)


def _plan_tma(cfg: dict, a, b, bm: int, bn: int, bk: int) -> tuple:
    """(mode, tma_a, tma_b, a_desc, b_desc) for this launch.

    `USE_TMA=True` means "use a descriptor wherever this stack and this tensor layout allow
    it".  That is a deterministic function of the tensors and the tile -- identical for both
    arms of the pair, since #1's two arms consume the same A and B -- so it cannot bias the
    ratio, but it does let the WEIGHT use a descriptor at decode where a box over A buys
    nothing.  An explicit `TMA_A`/`TMA_B` is a pin: if it cannot be honoured it raises
    rather than degrading.  And if NOTHING can be described, the config is a byte-identical
    duplicate of the classic one, so it raises rather than being timed and reported as TMA.

    Every legality check and every descriptor build goes through the MEMOIZED `_tma_check`,
    including the ones that only produce an error message.  This is the launch path, not
    the setup path: autotuning replays the same (tensor, box) pair thousands of times, and
    `TensorDescriptor.from_tensor` is tens of microseconds of python that would land inside
    the timed window and be charged to the TMA arm alone.
    """
    both = bool(cfg.get("USE_TMA", False))
    want_a, want_b = bool(cfg.get("TMA_A", both)), bool(cfg.get("TMA_B", both))
    if not (want_a or want_b):
        return _TMA_OFF, False, False, None, None

    mode = _tma_spelling(cfg)
    if mode == _TMA_OFF:
        c = caps()
        raise TmaUnsupported(
            f"TMA requested but no descriptor form is available (tma={getattr(c,'tma',False)}, "
            f"host={getattr(c,'tma_host',False)}, device={getattr(c,'tma_device',False)}, "
            f"cc={getattr(c,'cc',(0,0))}, sources={getattr(c,'sources',{})})"
        )
    # Before the first descriptor of either spelling is built OR launched.
    _ensure_allocator()
    build = mode == _TMA_HOST

    def _plan(tag, t, block, want, pinned):
        if not want:
            return False, None
        desc, why = _tma_check(t, block, build)
        if why is None:
            return True, desc
        if pinned:
            raise TmaUnsupported(f"{tag} box {block}: {why}")
        return False, None

    tma_a, a_desc = _plan("A", a, [bm, bk], want_a, "TMA_A" in cfg)
    tma_b, b_desc = _plan("B", b, [bk, bn], want_b, "TMA_B" in cfg)
    if not (tma_a or tma_b):
        raise TmaUnsupported(
            f"neither operand can be described (A box [{bm},{bk}]: "
            f"{_tma_check(a, [bm, bk], False)[1]}; "
            f"B box [{bk},{bn}]: {_tma_check(b, [bk, bn], False)[1]})"
        )
    return mode, tma_a, tma_b, a_desc, b_desc


def h200_report() -> dict:
    """What this module's H200 paths actually did.  Record it next to n_tried/n_failed.

    A TMA arm that declined every descriptor and ran the classic path all along is
    otherwise indistinguishable from a TMA arm that did nothing useful (LOG-08).
    """
    rep = {"hopper_module": _H is not None}
    c = caps()
    for k in ("tma", "tma_host", "tma_device", "warp_specialize", "clusters", "ws_mode"):
        rep[k] = getattr(c, k, None)
    rep["cc"] = list(getattr(c, "cc", (0, 0)))
    if _H is not None:
        try:
            rep["tma_stats"] = _H.tma_stats()
        except Exception as exc:  # noqa: BLE001
            rep["tma_stats"] = f"{type(exc).__name__}: {exc}"
    # The shared-memory model that pruned this run's grid, evaluated at the one shape
    # preflight actually measured on this device (BM128 BN256 BK64 s3 -> 147456 B), next to
    # the ceiling it was compared against.  A grid pruned by the wrong rule leaves NO trace
    # -- the rejected configs are simply absent from the table -- so the rule has to travel
    # in the result file alongside the numbers it shaped.
    try:
        rep["smem_model_bm128_bn256_bk64_s3"] = smem_stage_bytes(128, 256, 64, 3)
        rep["smem_limit_bytes"] = smem_limit()
    except Exception as exc:  # noqa: BLE001
        rep["smem_model_error"] = f"{type(exc).__name__}: {exc}"
    return rep


# ======================================================================================
# The single GEMM source.  FUSE_RESADD selects the epilogue.
# ======================================================================================
@triton.jit
def _gemm_step(
    a_desc,
    b_desc,
    a_ptrs,
    b_ptrs,
    acc,
    m0,
    n0,
    kcur,
    offs_k,
    mask_m,
    mask_n,
    K,
    stride_ak,
    stride_bk,
    STEP: tl.constexpr,
    EVEN_K: tl.constexpr,
    TMA_A: tl.constexpr,
    TMA_B: tl.constexpr,
):
    """One k-step of the mainloop, in ONE place.

    The k-loop header is written out twice (warp-specialized and not); the body is here so
    the two copies are the same text by construction.  Returns the advanced pointer tiles,
    so the caller carries no duplicated address arithmetic either.

    A TMA box needs no k-mask: an out-of-bounds box is zero-filled by the hardware, which
    is the same value the masked pointer load supplies.
    """
    if TMA_A:
        a = a_desc.load([m0, kcur])
    else:
        if EVEN_K:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        else:
            kmask = (kcur + offs_k) < K
            a = tl.load(a_ptrs, mask=mask_m[:, None] & kmask[None, :], other=0.0)
        a_ptrs += STEP * stride_ak

    if TMA_B:
        b = b_desc.load([kcur, n0])
    else:
        if EVEN_K:
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        else:
            kmask = (kcur + offs_k) < K
            b = tl.load(b_ptrs, mask=kmask[:, None] & mask_n[None, :], other=0.0)
        b_ptrs += STEP * stride_bk

    return tl.dot(a, b, acc), a_ptrs, b_ptrs


@triton.jit
def oproj_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    r_ptr,
    a_desc,  # host-built descriptor, or None (unused / device-side spelling)
    b_desc,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_rm,
    stride_rn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    TMA_A: tl.constexpr,
    TMA_B: tl.constexpr,
    TMA_DEVICE: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    FUSE_RESADD: tl.constexpr,
    ATOMIC_OUT: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    if SPLIT_K == 1:
        k0 = 0
        num_iters = tl.cdiv(K, BLOCK_K)
    else:
        k0 = pid_k * BLOCK_K
        num_iters = tl.cdiv(K - k0, BLOCK_K * SPLIT_K)

    # Device-side descriptors are built here, from the same pointers and strides the
    # classic path uses, and REBIND the (None) host-descriptor parameters -- so the
    # mainloop below is one piece of code regardless of which spelling is live.  The
    # launcher has already checked that the innermost stride is 1 for whichever operand
    # this covers, which is what makes `strides=[..., 1]` legal.
    if TMA_A:
        if TMA_DEVICE:
            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[M, K],
                strides=[stride_am, 1],
                block_shape=[BLOCK_M, BLOCK_K],
            )
    if TMA_B:
        if TMA_DEVICE:
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[K, N],
                strides=[stride_bk, 1],
                block_shape=[BLOCK_K, BLOCK_N],
            )

    # Address tiles for the pointer path.  Built unconditionally (they are hoisted and
    # dead-code-eliminated under TMA) so `_gemm_step`'s arity never changes.
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak
    b_ptrs = b_ptr + (k0 + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    step: tl.constexpr = BLOCK_K * SPLIT_K
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    # The header is duplicated, not parameterised: `warp_specialize=` lowers to an MLIR
    # attribute and some Triton builds refuse a tl.constexpr there, which would break the
    # WS=False path too.  A constexpr `if` has worked in every Triton ever shipped, and the
    # LITERAL spelling below is the exact one preflight compiled on this H200
    # (`warp_specialize_tl_range` OK) -- so the proven form is the one emitted.
    if WARP_SPECIALIZE:
        for k in tl.range(0, num_iters, warp_specialize=True):
            acc, a_ptrs, b_ptrs = _gemm_step(
                a_desc, b_desc, a_ptrs, b_ptrs, acc, m0, n0, k0 + k * step,
                offs_k, mask_m, mask_n, K, stride_ak, stride_bk,
                step, EVEN_K, TMA_A, TMA_B,
            )
    else:
        for k in range(0, num_iters):
            acc, a_ptrs, b_ptrs = _gemm_step(
                a_desc, b_desc, a_ptrs, b_ptrs, acc, m0, n0, k0 + k * step,
                offs_k, mask_m, mask_n, K, stride_ak, stride_bk,
                step, EVEN_K, TMA_A, TMA_B,
            )

    cmask = mask_m[:, None] & mask_n[None, :]

    if FUSE_RESADD:
        r_ptrs = r_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn
        if SPLIT_K == 1:
            # plain beta=1 epilogue
            acc += tl.load(r_ptrs, mask=cmask, other=0.0).to(tl.float32)
        else:
            # only the first K-slice folds in the residual
            if pid_k == 0:
                acc += tl.load(r_ptrs, mask=cmask, other=0.0).to(tl.float32)

    # The store stays a masked pointer store even under TMA: the fused epilogue is
    # predicated (SPLIT_K != 1 folds the residual only at pid_k == 0) and the split-K path
    # is an atomic, neither of which a descriptor store expresses.  Both arms store the
    # same way, so this costs the comparison nothing.
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if ATOMIC_OUT:
        tl.atomic_add(c_ptrs, acc, mask=cmask)
    else:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=cmask)


# ======================================================================================
# The split-out elementwise work.  HAS_RES selects add-vs-plain-copy/cast.
# ======================================================================================
@triton.jit
def epilogue_kernel(
    c_ptr,
    r_ptr,
    o_ptr,
    n_elements,
    HAS_RES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elements
    v = tl.load(c_ptr + offs, mask=m, other=0.0).to(tl.float32)
    if HAS_RES:
        v += tl.load(r_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + offs, v.to(o_ptr.dtype.element_ty), mask=m)


# ======================================================================================
# Launchers
# ======================================================================================
def _pipeline_buffers(num_stages: int) -> int:
    """How many mainloop buffers Triton really allocates -- an ARCHITECTURE question.

    sm_89 / triton 3.6 stages `max(2, num_stages - 1)`, measured by launching 68 configs
    and reading `CompiledKernel.metadata.shared` (exact on 64, conservative on the 4 with
    num_stages=2).  **The H200 stages `num_stages`.**  All five of
    `preflight_h200.json`'s `smem_probe` trial compiles are matched EXACTLY by
    `num_stages * 2 * BK * (BM + BN)`:

        128/128/64 s3  ->  98304      128/256/64 s3 -> 147456
        128/256/64 s4  -> 196608      256/256/64 s3 -> 196608
        128/256/128 s3 -> 294912  (> the 232448 B opt-in ceiling; it is the one that fails)

    so the sm_89 rule under-predicts Hopper by 1.5x at s3.  Under-prediction is the benign
    direction -- the config is offered, fails to compile and lands in `n_failed` where the
    fairness check can see it -- but the error is proportional to the tile, so it is larger
    for whichever arm stages more (here the split-K/large-BN mappings).  It does not prune
    the two arms equally, which is the failure this study cannot detect afterwards.  Hence
    a runtime read of the device, never a literal.
    """
    return int(num_stages) if tuple(getattr(caps(), "cc", (0, 0))) >= (9, 0) \
        else max(2, int(num_stages) - 1)


def smem_stage_bytes(bm: int, bn: int, bk: int, num_stages: int, bn_mult: int = 1) -> int:
    """Shared memory a Triton GEMM mainloop stages, in bytes.

    `glm52_h200.config.smem_stage_bytes` owns the canonical rule and `config.smem_model_check()`
    audits it against preflight's measurements, so this DELEGATES there when the package is
    importable; the local fallback (see `_pipeline_buffers`) exists only so a kernel module
    stays usable standalone.  Two copies of a pruning rule that disagree would prune the
    fused and unfused grids differently, which is worse than either rule being wrong.

    TMA stages the same tile bytes (plus a handful of mbarriers), so the model is unchanged
    for the descriptor path.  This is an ESTIMATE either way: the authoritative number is
    `CompiledKernel.metadata.shared` after a trial compile, which the bench records.
    """
    _C = _config()
    if _C is not None and hasattr(_C, "smem_stage_bytes"):
        try:
            return int(_C.smem_stage_bytes(bm, bn, bk, num_stages, bn_mult))
        except Exception:  # noqa: BLE001 -- fall through to the local copy
            pass
    return _pipeline_buffers(num_stages) * 2 * bk * (bm + bn_mult * bn)


def smem_bytes(cfg: dict) -> int:
    """Estimated mainloop SMEM footprint for a GEMM config (Triton >= 3.3 model)."""
    return smem_stage_bytes(
        cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"], cfg["num_stages"]
    )


def smem_fits(cfg: dict, limit: "int | None" = None) -> bool:
    """Does this config fit the DEVICE's per-block ceiling?  No literal anywhere."""
    return smem_bytes(cfg) <= (smem_limit() if limit is None else limit)


def gemm_launch(a, b, c, r, cfg, fuse_resadd: bool, atomic_out: bool):
    """a:[M,K] b:[K,N] (strided, any layout) c:[M,N] r:[M,N] or None.

    Classic cfg keys: BLOCK_M/BLOCK_N/BLOCK_K/GROUP_M/SPLIT_K/num_warps/num_stages.
    H200 cfg keys: see `H200_CFG_KEYS` and the module docstring.
    """
    M, K = a.shape
    N = c.shape[1]
    sk = cfg.get("SPLIT_K", 1)
    bm, bn, bk = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"]

    mode, tma_a, tma_b, a_desc, b_desc = _plan_tma(cfg, a, b, bm, bn, bk)
    ws_flag, ws_kw = _ws_launch(cfg)

    # A k-mask can be skipped only when every slice's last box lands inside K.  With
    # SPLIT_K the slices step by BLOCK_K*SPLIT_K from staggered starts, and K divisible by
    # that product is exactly the condition that keeps all of them in range.
    even_k = (K % (bk * sk)) == 0

    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn), sk)
    return oproj_gemm_kernel[grid](
        a,
        b,
        c,
        r if r is not None else a,  # unused pointer when FUSE_RESADD is off
        a_desc,
        b_desc,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        r.stride(0) if r is not None else 0,
        r.stride(1) if r is not None else 0,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=cfg["GROUP_M"],
        SPLIT_K=sk,
        EVEN_K=even_k,
        TMA_A=tma_a,
        TMA_B=tma_b,
        TMA_DEVICE=(mode == _TMA_DEVICE),
        WARP_SPECIALIZE=ws_flag,
        FUSE_RESADD=fuse_resadd,
        ATOMIC_OUT=atomic_out,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        num_ctas=int(cfg.get("num_ctas", 1)),
        **ws_kw,
    )


def epilogue_launch(c, r, o, cfg, has_res: bool):
    n = o.numel()
    grid = (triton.cdiv(n, cfg["BLOCK"]),)
    return epilogue_kernel[grid](
        c,
        r if r is not None else c,
        o,
        n,
        HAS_RES=has_res,
        BLOCK=cfg["BLOCK"],
        num_warps=cfg["num_warps"],
        num_stages=cfg.get("num_stages", 1),
    )


# ======================================================================================
# Chain builders.  These are what `autotune`/`bench_chain` time.
# ======================================================================================
def make_fused_chain(a, b, r, out, acc32, cfg):
    """FUSED: h1 = A@B + residual, written straight to `out`.

    SPLIT_K == 1 -> one kernel, no intermediate materialized at all.
    SPLIT_K  > 1 -> [zero acc32, atomic gemm (residual folded at pid_k==0), cast].
    """
    sk = cfg.get("SPLIT_K", 1)
    if sk == 1:
        return [lambda: gemm_launch(a, b, out, r, cfg, True, False)]
    ecfg = cfg["EPI"]
    return [
        lambda: acc32.zero_(),
        lambda: gemm_launch(a, b, acc32, r, cfg, True, True),
        lambda: epilogue_launch(acc32, None, out, ecfg, False),
    ]


def make_unfused_chain(a, b, r, out, cmat, acc32, gcfg, ecfg):
    """UNFUSED: GEMM materializes C, then the elementwise kernel writes h1 = C + r."""
    sk = gcfg.get("SPLIT_K", 1)
    if sk == 1:
        return [
            lambda: gemm_launch(a, b, cmat, None, gcfg, False, False),
            lambda: epilogue_launch(cmat, r, out, ecfg, True),
        ]
    return [
        lambda: acc32.zero_(),
        lambda: gemm_launch(a, b, acc32, None, gcfg, False, True),
        lambda: epilogue_launch(acc32, r, out, ecfg, True),
    ]
