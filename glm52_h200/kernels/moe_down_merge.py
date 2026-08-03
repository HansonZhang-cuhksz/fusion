"""Fusions #8 / #9 -- MoE **down** grouped GEMM + expert merge (+ residual add).  H200.

Ported from `glm52/kernels/moe_down_merge.py`.  What sglang 0.5.10 does *today* (the
unfused production path):

    invoke_fused_moe_kernel(intermediate_cache2, w2, intermediate_cache3,
                            ..., mul_routed_weight=True, top_k=1)
    -> intermediate_cache3 : [T, topk, H]        (the routing weight is applied here)
    moe_sum_reduce(intermediate_cache3, out)     -> a SEPARATE reduction kernel

so the ``[T, topk, 6144]`` tensor is fully materialised and immediately re-read.  sglang
*does* carry an atomic variant (``FUSE_SUM_ALL_REDUCE``, kernel line 607) but it is gated
behind a server flag and only used together with a fused all-reduce.  Fusion #8 is exactly
the question "should that be the default?", and #9 adds the post-MoE residual add on top.

ONE grouped-GEMM source, ``tl.constexpr`` flags select the epilogue:

* ``moe_down_kernel``  -- the sglang ``fused_moe_kernel`` shape, expert-major grid over
  ``sorted_token_ids``.
    - ``FUSE_MERGE=False``  : ``tl.store`` into ``c[offs_token, n]`` i.e. ``[rows, H]``.
      This is the unfused GEMM; a separate ``moe_sum_kernel`` then reduces over topk.
    - ``FUSE_MERGE=True``   : ``tl.atomic_add`` into ``c[offs_token // top_k, n]`` i.e.
      ``[T, H]``.  Accumulation strategy **(a)**.  The output buffer must be pre-seeded
      (zeroed for #8, filled with the residual for #9) and that seeding is part of the
      fused chain's cost.
  Everything before the epilogue -- dispatch, gather, K-loop, ``even_Ks``, ``GROUP_SIZE_M``
  swizzle, and every H200 axis below -- is byte-for-byte the same code for both flag
  values, so both arms of the pair are offered the identical mapping space.

* ``moe_down_token_major_kernel`` -- accumulation strategy **(b)**.  Same arithmetic, same
  epilogue algebra, but a different **grid order and loop order** (which fairness rule 1
  explicitly permits): one CTA owns ONE token's ``BLOCK_N`` output columns and loops over
  that token's ``topk`` experts internally, summing in registers.  No atomics, no
  ``[T, topk, H]`` tensor, and the residual add (#9) is a single extra load.  Because the
  token tile is necessarily 1 row, the inner product is a GEMV; ``USE_DOT`` lets the tuner
  choose between a padded ``tl.dot`` (M=16, 15 rows masked off) and a broadcast/reduce
  GEMV, both of which are pure mapping choices.

* ``moe_sum_kernel``   -- the split-out merge, a port of sglang's ``_moe_sum_reduce_kernel``
  (lightllm lineage).  ``ADD_RESIDUAL`` folds the #9 residual add into it, which lets the
  bench also report the "2-kernel" #9 baseline alongside the strict 3-kernel one.
* ``resadd_kernel``    -- the split-out residual add for the strict 3-kernel #9 baseline.

Shapes for GLM-5.2: A = ``[rows=T*8, I=2048]`` (SwiGLU output), B = ``w2 [E, H=6144,
I=2048]``, so the GEMM is ``N = H = 6144``, ``K = I = 2048``.  The routing weight index is
``offs_token`` (flat ``token*topk + k``), exactly as sglang does with ``top_k=1`` for the
A-gather of the second GEMM.

--------------------------------------------------------------------------------------
The H200 axes, and why every one of them is a RUNTIME choice
--------------------------------------------------------------------------------------
Nothing sm_90-specific is decided at authoring time.  `kernels/hopper.py` decides what
this process may emit; this module only adapts.  The cfg keys the launchers forward are
advertised in `H200_CFG_KEYS`, which `bench.widen()` reads before adding any overlay -- so
on a stack without these features the grids are byte-identical to the classic ones.

The 2026-08-03 H200 preflight (`preflight_h200.json`) turned four of those runtime
questions into measured answers.  They are recorded because they decide which paths are
worth carrying -- NOT because anything below assumes them; every gate is still a `caps()`
read at launch time:

  * `tl.range(..., warp_specialize=True)` compiles.  The launch-kwarg spelling does NOT:
    triton 3.6.0 answers "Keyword argument num_consumer_groups was specified but
    unrecognised", and an unrecognised kwarg is a hard `KeyError`, not a soft failure.
    `_ws_launch` emits it only where `hopper.ws_mode()` says the stack takes it.
  * `num_ctas > 1` compiles, so thread-block clusters are a real axis here.
  * TMA's preflight probe was a FALSE NEGATIVE -- it fed a host-side `TensorDescriptor`
    into `tl.make_tensor_descriptor()`, the DEVICE-side constructor, and mixing the two
    APIs is a CompilationError on any hardware.  Both spellings this file emits were
    verified to compile, launch and produce correct values on triton 3.6.0.
  * The device's best bf16 Triton GEMM is BM128 BN256 BK64 GM8 w8 s3 (788 TF/s) and the
    per-block SMEM ceiling is 232448 B.  Under the corrected staging rule (see
    `smem_stage_bytes`) that shape models at 147456 B and BM256 BN256 BK64 s3 at 196608 B,
    so both are expressible here and both fit -- and since `moe_down_kernel`'s two arms
    differ only in their epilogue, they see the identical ceiling on the identical model.

One number from that preflight bears on #8/#9's verdict rather than on any knob: the
device's FLOP/byte balance is ~185 (C500 was 82, the RTX 4060 84).  On a machine that
compute-dense the merge traffic these fusions DELETE is worth relatively more, while work
added to the GEMM mainloop -- which is where strategy (b) puts its topk loop -- costs
relatively more.  Those pull in opposite directions, which is precisely why both strategies
are built and timed here rather than one being chosen from the C500-era ranking.

  `USE_TMA` / `TMA_B`  -- the w2 tile through a tensor-descriptor box.  **Only B**: A is
      gathered through ``sorted_token_ids`` (or, in the token-major kernel, is one row
      broadcast across a padded dot tile), so no descriptor can express it.  B is where the
      bandwidth is: a decode step reads ~200 MB of expert weights to produce 12 KB.
  `TMA_MODE` ("host" | "device") -- host-side descriptors make Triton's launcher call
      ``cuTensorMapEncodeTiled`` on every launch, which at decode lands inside the measured
      window and is charged to the TMA arm alone; the device-side spelling pays global
      scratch instead.  Both are offered; the tuner decides.
  `warp_specialize` (+ the forked-Triton launch-kwarg spelling, measured DEAD here) -- on
      ``moe_down_kernel``'s k-loop only.  The token-major kernel's k-loop is NESTED inside
      its topk loop, so it is not the CTA's top-level mainloop and Triton's
      producer/consumer partition has nothing to hang off; its launcher REJECTS the flag
      rather than accepting and ignoring it, because a silently-ignored knob would appear
      in the result file as a measured no-op.
  `num_ctas` -- thread-block cluster width, forwarded straight to the Triton launch.

Larger tiles: nothing here caps BLOCK_*; `smem_limit()` reads the ceiling off the device
probe.  Both arms of every pair see the same ceiling, which is the whole point.

One deliberate departure from the C500 source, applied to every variant so it cannot move
any ratio: accumulation is written ``tl.dot(a, b, acc)`` rather than ``acc += tl.dot(a, b)``
-- same arithmetic, but it names the wgmma accumulator explicitly.
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
    counts the config in that arm's `n_failed`.
    """


class TmaUnsupported(HopperPathUnavailable):
    """The weight tile could not be described by a TMA box."""


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

    The H200 opts in to 232448 B per block (233472 per SM).  If the probe cannot supply it
    we RAISE rather than substitute a plausible constant: a grid built from the wrong
    ceiling prunes the two arms unequally, and that is the one failure mode this study
    cannot detect after the fact.
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
    """Is `warp_specialize=True` worth OFFERING for this config (grouped kernel only)?

    Triton splits the CTA into producer and consumer warp groups, so a CTA with fewer than
    four warps has nothing to partition and a single-buffered mainloop nothing to overlap.
    Both are Triton-side constraints, not device constants.  The bench offers one grid to
    both arms, so this prunes both identically.
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
    `hopper` answers both and this merges them.  A request neither can serve raises.

    The launch-kwarg spelling is forwarded ONLY when `hopper.ws_mode()` reports that this
    Triton takes it.  The measured H200 does not: `warp_specialize_num_consumer_groups`
    fails with "Keyword argument num_consumer_groups was specified but unrecognised", which
    Triton raises as a `KeyError` at launch rather than ignoring.  Copying the key out of
    the cfg unconditionally would turn any config carrying it into an opaque launch failure
    on the device this suite is for.  On a launch-mode stack the kwargs ARE the enabling
    mechanism (there is no source flag to pair them with), so they are forwarded there
    whether or not `warp_specialize` is also set -- hence the gate is the stack's mode, not
    `want`.
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
_FLAT_MEMO: dict = {}
_MEMO_CAP = 4096


def clear_caches() -> None:
    """Drop memoized descriptors and flattened weight views.  Call when w2 is freed.

    Both memos hold a reference to the tensor they describe, so they keep 6 GB of expert
    weights alive.  Harmless inside one bench (the weights are allocated once for all
    regimes); the hook exists so a whole-layer run can release between fusions.
    """
    _TMA_MEMO.clear()
    _FLAT_MEMO.clear()


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
    pair thousands of times, and both the legality check and `TensorDescriptor.from_tensor`
    are python: tens of microseconds that would land INSIDE the timed region and be charged
    to the TMA arm alone.  At decode a kernel resolves to single-digit CUDA-event ticks, so
    that is not a rounding error -- it is exactly the kind of asymmetric harness cost the
    fairness audit exists to catch.  The key carries the data pointer AND the
    shape/stride/dtype/box, so a reallocated buffer can never reuse a stale descriptor.
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
    `hopper.ensure_allocator()` is idempotent, so this is called on the TMA planning path
    unconditionally rather than only where it is provably needed.
    """
    if _H is None or not _H.ensure_allocator():
        raise TmaUnsupported(
            "triton.set_allocator is unavailable, so no TMA descriptor kernel can launch "
            f"(hopper.ensure_allocator() -> False, hopper_module={_H is not None})"
        )


def _tma_spelling(cfg: dict) -> int:
    """Host-side descriptor argument vs device-side `tl.make_tensor_descriptor`."""
    c = caps()
    pin = str(cfg.get("TMA_MODE", "") or "").lower()
    form = pin if pin in ("host", "device") else (
        c.tma_form() if hasattr(c, "tma_form") else "none"
    )
    return {"device": _TMA_DEVICE, "host": _TMA_HOST}.get(form, _TMA_OFF)


def _flat_weight(w):
    """[E, N, K] contiguous expert weights viewed as [E*N, K] -- no copy.

    The expert index becomes a ROW OFFSET, which keeps the TMA box 2-D: Hopper caps every
    box dimension at 256, so a 3-D box with a leading 1 buys nothing and costs a reshape in
    registers.  The box then comes back N-major -- exactly how sglang stores w2
    ([E, H, I], I contiguous) -- and `tl.trans` hands it to wgmma, whose B operand consumes
    an N-major fragment directly.  So the descriptor path needs no repacking of the
    production weight layout, which is the whole reason it is worth having here.

    Memoized for the same reason the descriptor is: this runs per LAUNCH, and a torch
    `reshape` inside the timed window is charged to the TMA arm alone.
    """
    key = (w.data_ptr(), tuple(w.shape), tuple(w.stride()))
    flat = _FLAT_MEMO.get(key)
    if flat is None:
        if not w.is_contiguous():
            raise TmaUnsupported("expert weights not contiguous; cannot flatten to [E*N, K]")
        flat = w.reshape(w.shape[0] * w.shape[1], w.shape[2])
        if len(_FLAT_MEMO) >= _MEMO_CAP:
            _FLAT_MEMO.clear()
        _FLAT_MEMO[key] = flat
    return flat


def _plan_tma_b(cfg: dict, w, bn: int, bk: int) -> tuple:
    """(mode, tma_b, b_desc) for this launch.  Raises rather than degrading silently."""
    if not (bool(cfg.get("USE_TMA", False)) or bool(cfg.get("TMA_B", False))):
        return _TMA_OFF, False, None
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
    flat = _flat_weight(w)
    desc, why = _tma_check(flat, [bn, bk], mode == _TMA_HOST)
    if why is not None:
        raise TmaUnsupported(f"w2 box [{bn},{bk}] over [E*H, I]: {why}")
    return mode, True, desc


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


# `do_not_specialize` on the token-count arguments: they are used only in comparisons and
# `cdiv`, never in address arithmetic, so Triton's divisible-by-16 / equal-to-1 hints buy
# nothing -- but WITHOUT this, every regime (rows = 8, 256, 2048, 8192, 65536) is a fresh
# specialization and therefore a fresh compile of every config in the grid.  Suppressing it
# lets all regimes share one compiled binary per config.  Applied identically to every
# kernel here, so it cannot favour either side of the comparison.


# ======================================================================================
# (1) The one grouped down-GEMM.  FUSE_MERGE picks the epilogue.
# ======================================================================================
@triton.jit
def _down_step(
    a_ptrs,
    b_ptrs,
    b_desc,
    acc,
    brow,
    kcur,
    offs_k,
    token_mask,
    K,
    EVEN_K: tl.constexpr,
    TMA_B: tl.constexpr,
):
    """One k-step of the grouped mainloop, in ONE place.

    The k-loop header is written out twice (warp-specialized and not); the body is here so
    the two copies are the same text by construction -- the only property of them that
    could move the ratio.  `brow` is the descriptor's row origin in the flattened [E*H, I]
    weight view: the expert index is folded into the row so the box stays 2-D.  A TMA box
    needs no k-mask -- an out-of-bounds box is zero-filled by the hardware, the same value
    the masked pointer load supplies.
    """
    if EVEN_K:
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
    else:
        a = tl.load(
            a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - kcur), other=0.0
        )
    if TMA_B:
        b = tl.trans(b_desc.load([brow, kcur]))
    else:
        if EVEN_K:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - kcur, other=0.0)
    return tl.dot(a, b, acc)


@triton.jit(do_not_specialize=["EM", "num_valid_tokens"])
def moe_down_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_desc,  # host-built descriptor over w2 flattened to [E*H, I], or None
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # dims
    N,  # = H = 6144
    K,  # = I = 2048
    EM,
    num_valid_tokens,  # = rows = T * top_k
    b_rows,  # E * H -- row extent of the flattened weight view (device-side descriptor)
    # strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # meta
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
    TMA_B: tl.constexpr,
    TMA_DEVICE: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    FUSE_MERGE: tl.constexpr,
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
    # Padded dispatch slots carry the sentinel `num_valid_tokens`.  Triton's pipeliner
    # emits speculative (unpredicated) prologue loads on more than one backend, so the
    # sentinel row must not even be *addressed*; clamping to row 0 keeps every address in
    # range and `token_mask` discards the value exactly as sglang's mask does.
    safe_token = tl.where(token_mask, offs_token, 0)

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A of the down GEMM is the [rows, I] SwiGLU output, one row per (token, k) pair --
    # sglang gathers it with top_k=1, i.e. `offs_token` directly (no // top_k).
    a_ptrs = a_ptr + (safe_token[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # Device-side descriptor is built here, from the same base pointer and row stride the
    # pointer path uses, and REBINDS the (None) host-descriptor parameter -- so the mainloop
    # below is one piece of code whichever spelling is live.  `stride_bn` IS the row stride
    # of the flattened [E*H, I] view (w2.stride(1)), and the launcher has already checked
    # that the innermost stride is 1, which is what makes `strides=[stride_bn, 1]` legal.
    if TMA_B:
        if TMA_DEVICE:
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[b_rows, K],
                strides=[stride_bn, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )

    # Row origin in the flattened [E*N, K] weight view.  int32 because a descriptor offset
    # is int32; 256 experts x 6144 rows = 1.57e6, three decimal orders inside the range.
    brow = (off_experts * N + pid_n * BLOCK_SIZE_N).to(tl.int32)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # The header is duplicated, not parameterised: `warp_specialize=` lowers to an MLIR
    # attribute and some Triton builds refuse a tl.constexpr there, which would break the
    # WS=False path too.  A constexpr `if` has worked in every Triton ever shipped, and the
    # LITERAL spelling below is the exact one preflight compiled on this H200
    # (`warp_specialize_tl_range` OK) -- so the proven form is the one emitted.
    if WARP_SPECIALIZE:
        for k_start in tl.range(0, K, BLOCK_SIZE_K, warp_specialize=True):
            acc = _down_step(
                a_ptrs, b_ptrs, b_desc, acc, brow, k_start, offs_k, token_mask,
                K, even_Ks, TMA_B,
            )
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
    else:
        for k_start in range(0, K, BLOCK_SIZE_K):
            acc = _down_step(
                a_ptrs, b_ptrs, b_desc, acc, brow, k_start, offs_k, token_mask,
                K, even_Ks, TMA_B,
            )
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    # ---- epilogue: routing weight, then either scatter-store or atomic merge ----------
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + safe_token, mask=token_mask, other=0.0)
        acc = acc * moe_weight[:, None]
    out = acc.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    if FUSE_MERGE:
        # strategy (a): accumulate straight into [T, H]; [T, topk, H] never exists.
        rows_out = safe_token // top_k
        c_ptrs = c_ptr + stride_cm * rows_out[:, None] + stride_cn * offs_cn[None, :]
        tl.atomic_add(c_ptrs, out, mask=c_mask)
    else:
        c_ptrs = c_ptr + stride_cm * safe_token[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_ptrs, out, mask=c_mask)


# ======================================================================================
# (2) Strategy (b): token-major -- one token per CTA, topk summed in registers.
# ======================================================================================
@triton.jit
def _tokmaj_dot_step(
    a_ptrs,
    b_ptrs,
    b_desc,
    part,
    brow,
    kcur,
    offs_k,
    m_mask,
    K,
    EVEN_K: tl.constexpr,
    TMA_B: tl.constexpr,
):
    """One k-step of the token-major padded-dot mainloop, in ONE place.

    Only the B side can use a descriptor: A is a single token row broadcast across a
    16-row dot tile, which no TMA box describes.
    """
    if EVEN_K:
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
    else:
        a = tl.load(
            a_ptrs, mask=m_mask[:, None] & (offs_k[None, :] < K - kcur), other=0.0
        )
    if TMA_B:
        b = tl.trans(b_desc.load([brow, kcur]))
    else:
        if EVEN_K:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - kcur, other=0.0)
    return tl.dot(a, b, part)


@triton.jit(do_not_specialize=["T"])
def moe_down_token_major_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_desc,  # host-built descriptor over w2 flattened to [E*H, I], or None
    topk_weights_ptr,
    topk_ids_ptr,
    residual_ptr,
    T,
    N,
    K,
    b_rows,  # E * H -- row extent of the flattened weight view (device-side descriptor)
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_rm,
    TOPK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # padded dot tile (>=16); unused when USE_DOT=False
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
    USE_DOT: tl.constexpr,
    TMA_B: tl.constexpr,
    TMA_DEVICE: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    tok = (pid // num_pid_n).to(tl.int64)
    pid_n = pid % num_pid_n
    if tok >= T:
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    if TMA_B:
        if TMA_DEVICE:
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[b_rows, K],
                strides=[stride_bn, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )

    if USE_DOT:
        # tl.dot needs M >= 16; only row 0 is a real token, the rest are masked to zero.
        offs_m = tl.arange(0, BLOCK_SIZE_M)
        m_mask = offs_m < 1
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for kk in range(TOPK):
            e = tl.load(topk_ids_ptr + tok * TOPK + kk).to(tl.int64)
            row = tok * TOPK + kk
            # every row of the M-tile addresses the SAME A row; rows 1.. are masked off.
            a_ptrs = (
                a_ptr
                + row * stride_am
                + offs_k[None, :] * stride_ak
                + tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.int64)
            )
            b_ptrs = (
                b_ptr
                + e * stride_be
                + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            )
            brow = (e * N + pid_n * BLOCK_SIZE_N).to(tl.int32)
            part = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            # No warp specialization here: this k-loop is NESTED inside the topk loop, so
            # it is not the CTA's top-level mainloop and Triton's producer/consumer split
            # has nothing to hang off.  `launch_down_token_major` rejects the flag rather
            # than accepting and ignoring it.
            for k_start in range(0, K, BLOCK_SIZE_K):
                part = _tokmaj_dot_step(
                    a_ptrs, b_ptrs, b_desc, part, brow, k_start, offs_k, m_mask,
                    K, even_Ks, TMA_B,
                )
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk
            if MUL_ROUTED_WEIGHT:
                w = tl.load(topk_weights_ptr + tok * TOPK + kk)
                part = part * w
            acc += part
        out = tl.sum(acc, 0)
    else:
        # GEMV: fold the routing weight into A (fp32, exact) so a SINGLE [BK, BN] fp32
        # register accumulator can absorb every k-step of every expert; one cross-lane
        # reduction at the very end.  No TMA here: the B tile is consumed by an elementwise
        # broadcast-multiply, not by wgmma, so the N-major box a descriptor returns would
        # have to be transposed for nothing.  `launch_down_token_major` rejects TMA_B
        # without USE_DOT rather than silently ignoring it.
        acc2 = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        for kk in range(TOPK):
            e = tl.load(topk_ids_ptr + tok * TOPK + kk).to(tl.int64)
            row = tok * TOPK + kk
            if MUL_ROUTED_WEIGHT:
                w = tl.load(topk_weights_ptr + tok * TOPK + kk)
            else:
                w = 1.0
            a_ptrs = a_ptr + row * stride_am + offs_k * stride_ak
            b_ptrs = (
                b_ptr
                + e * stride_be
                + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            )
            for k_start in range(0, K, BLOCK_SIZE_K):
                if even_Ks:
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs)
                else:
                    a = tl.load(a_ptrs, mask=offs_k < K - k_start, other=0.0)
                    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)
                acc2 += (a.to(tl.float32) * w)[:, None] * b.to(tl.float32)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk
        out = tl.sum(acc2, 0)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_cn < N
    if ADD_RESIDUAL:
        r = tl.load(residual_ptr + tok * stride_rm + offs_cn, mask=n_mask, other=0.0)
        out = out + r.to(tl.float32)
    tl.store(
        c_ptr + tok * stride_cm + offs_cn * stride_cn,
        out.to(compute_type),
        mask=n_mask,
    )


# ======================================================================================
# (3) The split-out merge kernel used by the unfused chain.
#     Port of sglang `_moe_sum_reduce_kernel`; ADD_RESIDUAL is the #9 variant.
# ======================================================================================
@triton.jit(do_not_specialize=["token_num", "hidden_dim"])
def moe_sum_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    output_ptr,
    output_stride_0,
    residual_ptr,
    residual_stride_0,
    token_num,
    hidden_dim,
    topk_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)
    residual_stride_0 = tl.cast(residual_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim
    mask = mask_token[:, None] & mask_dim[None, :]

    base_ptrs = input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)
    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
        tile = tl.load(base_ptrs + i * input_stride_1, mask=mask, other=0.0)
        acc += tile.to(tl.float32)

    if ADD_RESIDUAL:
        r = tl.load(
            residual_ptr + offs_token[:, None] * residual_stride_0 + offs_dim[None, :],
            mask=mask,
            other=0.0,
        )
        acc += r.to(tl.float32)

    tl.store(
        output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :],
        acc.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


# ======================================================================================
# (4) The split-out residual add, for the strict 3-kernel #9 baseline.
# ======================================================================================
@triton.jit(do_not_specialize=["M", "Nd"])
def resadd_kernel(
    x_ptr,
    r_ptr,
    o_ptr,
    M,
    Nd,
    stride_xm,
    stride_rm,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(Nd, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < Nd)
    x = tl.load(x_ptr + rm[:, None] * stride_xm + rn[None, :], mask=mask, other=0.0)
    r = tl.load(r_ptr + rm[:, None] * stride_rm + rn[None, :], mask=mask, other=0.0)
    tl.store(
        o_ptr + rm[:, None] * stride_om + rn[None, :],
        (x.to(tl.float32) + r.to(tl.float32)).to(o_ptr.dtype.element_ty),
        mask=mask,
    )


# ======================================================================================
# (5) Output seeding for the atomic strategy.  ZEROING/SEEDING IS PART OF THE FUSED COST.
# ======================================================================================
@triton.jit(do_not_specialize=["M", "Nd"])
def seed_kernel(
    o_ptr,
    r_ptr,
    M,
    Nd,
    stride_om,
    stride_rm,
    FROM_RESIDUAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(Nd, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < Nd)
    if FROM_RESIDUAL:
        v = tl.load(r_ptr + rm[:, None] * stride_rm + rn[None, :], mask=mask, other=0.0)
    else:
        v = tl.zeros((BLOCK_M, BLOCK_N), dtype=o_ptr.dtype.element_ty)
    tl.store(o_ptr + rm[:, None] * stride_om + rn[None, :], v, mask=mask)


# ======================================================================================
# Thin python launchers
# ======================================================================================
def _pipeline_buffers(num_stages: int) -> int:
    """How many mainloop buffers Triton really allocates -- an ARCHITECTURE question.

    sm_89 / triton 3.6 stages `max(2, num_stages - 1)`, measured by launching 68 configs
    and reading `CompiledKernel.metadata.shared`.  **The H200 stages `num_stages`.**  All
    five of `preflight_h200.json`'s `smem_probe` trial compiles are matched EXACTLY by
    `num_stages * 2 * BK * (BM + BN)`:

        128/128/64 s3  ->  98304      128/256/64 s3 -> 147456
        128/256/64 s4  -> 196608      256/256/64 s3 -> 196608
        128/256/128 s3 -> 294912  (> the 232448 B opt-in ceiling; it is the one that fails)

    so the sm_89 rule under-predicts Hopper by 1.5x at s3.  Under-prediction is the benign
    direction -- the config is offered, fails to compile and lands in `n_failed` where the
    fairness check can see it -- but the error scales with the tile, so it differs between
    the grouped mainloop (which stages an A tile and a B tile) and the token-major one
    (B only, `smem_bytes_tokmaj`).  A pruning rule whose error differs between the two
    strategies whose relative timing DECIDES the #8/#9 verdict is not acceptable, so the
    rule is read from the device at runtime.
    """
    return int(num_stages) if tuple(getattr(caps(), "cc", (0, 0))) >= (9, 0) \
        else max(2, int(num_stages) - 1)


def smem_stage_bytes(bm: int, bn: int, bk: int, num_stages: int, bn_mult: int = 1) -> int:
    """Shared memory a Triton GEMM mainloop stages, in bytes.

    `glm52_h200.config.smem_stage_bytes` owns the canonical rule and
    `config.smem_model_check()` audits it against preflight's measurements, so this
    DELEGATES there when the package is importable; the local fallback (see
    `_pipeline_buffers`) exists only so a kernel module stays usable standalone.  Two
    copies of a pruning rule that disagree would prune the fused and unfused grids
    differently, which is worse than either rule being wrong.

    TMA stages the same tile bytes (plus a few mbarriers).  This is an ESTIMATE either way
    -- the authoritative number is `CompiledKernel.metadata.shared` after a trial compile.
    """
    _C = _config()
    if _C is not None and hasattr(_C, "smem_stage_bytes"):
        try:
            return int(_C.smem_stage_bytes(bm, bn, bk, num_stages, bn_mult))
        except Exception:  # noqa: BLE001 -- fall through to the local copy
            pass
    return _pipeline_buffers(num_stages) * 2 * bk * (bm + bn_mult * bn)


def smem_bytes(cfg: dict) -> int:
    """Mainloop SMEM footprint for the grouped down GEMM (one A + one B tile)."""
    return smem_stage_bytes(
        cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"], cfg["num_stages"]
    )


def smem_bytes_tokmaj(cfg: dict) -> int:
    """Token-major stages only the B tile (A is a single row / a masked M=16 tile)."""
    m = cfg["BLOCK_M"] if cfg.get("USE_DOT") else 1
    return smem_stage_bytes(m, cfg["BLOCK_N"], cfg["BLOCK_K"], cfg["num_stages"])


def smem_fits(cfg: dict, limit: "int | None" = None) -> bool:
    """Does this config fit the DEVICE's per-block ceiling?  No literal anywhere."""
    return smem_bytes(cfg) <= (smem_limit() if limit is None else limit)


def smem_fits_tokmaj(cfg: dict, limit: "int | None" = None) -> bool:
    return smem_bytes_tokmaj(cfg) <= (smem_limit() if limit is None else limit)


def launch_down(
    a,
    w2,
    c,
    topk_weights_flat,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    num_valid_tokens,
    top_k,
    cfg,
    fuse_merge: bool,
):
    """a: [rows, I] bf16.  w2: [E, H, I] bf16.
    c: [rows, H] when fuse_merge=False, else the pre-seeded [T, H] output.

    Classic cfg keys: BLOCK_M/BLOCK_N/BLOCK_K/GROUP_M/num_warps/num_stages.
    H200 cfg keys: see `H200_CFG_KEYS` and the module docstring.
    """
    N = w2.shape[1]
    K = a.shape[1]
    EM = sorted_token_ids.shape[0]
    bm, bn, bk = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"]

    mode, tma_b, b_desc = _plan_tma_b(cfg, w2, bn, bk)
    ws_flag, ws_kw = _ws_launch(cfg)

    grid = (triton.cdiv(EM, bm) * triton.cdiv(N, bn),)
    return moe_down_kernel[grid](
        a,
        w2,
        c,
        b_desc,
        topk_weights_flat,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_valid_tokens,
        w2.shape[0] * w2.shape[1],
        a.stride(0),
        a.stride(1),
        w2.stride(0),
        w2.stride(2),
        w2.stride(1),
        c.stride(0),
        c.stride(1),
        top_k=top_k,
        BLOCK_SIZE_M=bm,
        BLOCK_SIZE_N=bn,
        BLOCK_SIZE_K=bk,
        GROUP_SIZE_M=cfg["GROUP_M"],
        MUL_ROUTED_WEIGHT=True,
        compute_type=tl.bfloat16,
        even_Ks=(K % bk == 0),
        TMA_B=tma_b,
        TMA_DEVICE=(mode == _TMA_DEVICE),
        WARP_SPECIALIZE=ws_flag,
        FUSE_MERGE=fuse_merge,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        num_ctas=int(cfg.get("num_ctas", 1)),
        **ws_kw,
    )


def launch_down_token_major(
    a, w2, out, topk_weights, topk_ids, residual, top_k, cfg, add_residual: bool
):
    """a: [T*top_k, I].  out: [T, H].  topk_ids/topk_weights: [T, top_k].

    TMA is available only in the USE_DOT mapping; warp specialization is REJECTED rather
    than ignored.  Both refusals are deliberate: a config row labelled with a knob it did
    not actually get is a fabricated measurement, and this kernel is one of the two
    strategies whose relative timing decides which #8/#9 fused arm gets reported.
    """
    if cfg.get("warp_specialize") or cfg.get("WARP_SPECIALIZE"):
        raise HopperPathUnavailable(
            "moe_down_token_major_kernel has no warp-specialized variant: its k-loop is "
            "nested inside the topk loop, so it is not the CTA's top-level mainloop."
        )
    T = topk_ids.shape[0]
    N = w2.shape[1]
    K = a.shape[1]
    bn, bk = cfg["BLOCK_N"], cfg["BLOCK_K"]
    use_dot = bool(cfg.get("USE_DOT", False))

    want_tma = bool(cfg.get("USE_TMA", False)) or bool(cfg.get("TMA_B", False))
    if want_tma and not use_dot:
        raise HopperPathUnavailable(
            "TMA_B requires USE_DOT in moe_down_token_major_kernel: the GEMV mapping "
            "multiplies B elementwise, so an N-major descriptor box would have to be "
            "transposed for nothing."
        )
    mode, tma_b, b_desc = _plan_tma_b(cfg, w2, bn, bk) if want_tma else (_TMA_OFF, False, None)

    grid = (T * triton.cdiv(N, bn),)
    return moe_down_token_major_kernel[grid](
        a,
        w2,
        out,
        b_desc,
        topk_weights,
        topk_ids,
        residual,
        T,
        N,
        K,
        w2.shape[0] * w2.shape[1],
        a.stride(0),
        a.stride(1),
        w2.stride(0),
        w2.stride(2),
        w2.stride(1),
        out.stride(0),
        out.stride(1),
        residual.stride(0),
        TOPK=top_k,
        BLOCK_SIZE_M=cfg.get("BLOCK_M", 16),
        BLOCK_SIZE_N=bn,
        BLOCK_SIZE_K=bk,
        MUL_ROUTED_WEIGHT=True,
        compute_type=tl.bfloat16,
        even_Ks=(K % bk == 0),
        USE_DOT=use_dot,
        TMA_B=tma_b,
        TMA_DEVICE=(mode == _TMA_DEVICE),
        ADD_RESIDUAL=add_residual,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        num_ctas=int(cfg.get("num_ctas", 1)),
    )


def launch_moe_sum(c3, out, residual, topk, cfg, add_residual: bool):
    """c3: [T, topk, H] (or a [rows, H] view).  out: [T, H]."""
    T = out.shape[0]
    Hd = out.shape[1]
    grid = (triton.cdiv(T, cfg["BLOCK_M"]), triton.cdiv(Hd, cfg["BLOCK_DIM"]))
    return moe_sum_kernel[grid](
        c3,
        c3.stride(0),
        c3.stride(1),
        out,
        out.stride(0),
        residual,
        residual.stride(0),
        T,
        Hd,
        topk_num=topk,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_DIM=cfg["BLOCK_DIM"],
        NUM_STAGE=cfg["num_stages"],
        ADD_RESIDUAL=add_residual,
        num_warps=cfg["num_warps"],
    )


def launch_resadd(x, r, o, cfg):
    M, Nd = x.shape
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(Nd, cfg["BLOCK_N"]),)
    return resadd_kernel[grid](
        x,
        r,
        o,
        M,
        Nd,
        x.stride(0),
        r.stride(0),
        o.stride(0),
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def launch_seed(o, r, cfg, from_residual: bool):
    M, Nd = o.shape
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(Nd, cfg["BLOCK_N"]),)
    return seed_kernel[grid](
        o,
        r,
        M,
        Nd,
        o.stride(0),
        r.stride(0),
        FROM_RESIDUAL=from_residual,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
