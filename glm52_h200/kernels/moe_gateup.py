"""Fusion #6 -- MoE Up/Gate grouped GEMM with a fused SwiGLU epilogue.  H200 / sm_90.

Ported from `glm52/kernels/moe_gateup.py`.  ONE kernel source, one `tl.constexpr` flag
(`FUSE_ACT`) selecting the epilogue:

* ``FUSE_ACT=False`` -- exactly sglang 0.5.10's ``fused_moe_kernel`` shape: the grid covers
  the full ``N = 2*I = 4096`` output width, one fp32 accumulator, the block is written
  straight out as bf16.  The chain then needs a second, element-wise ``silu_and_mul``
  kernel that reads ``[rows, 4096]`` and writes ``[rows, 2048]``.
* ``FUSE_ACT=True``  -- the grid covers ``N = I = 2048``.  Each program keeps **two**
  accumulators, one for the gate columns ``[n, n+BN)`` and one for the up columns
  ``[I+n, I+n+BN)``, sharing a single K-loop over the gathered A tile.  The epilogue applies
  ``silu(gate) * up`` in fp32 and writes only ``[rows, 2048]``.  The 4096-wide intermediate
  is never materialised.

Everything else -- the ``sorted_token_ids`` / ``expert_ids`` / ``num_tokens_post_padded``
dispatch, the ``offs_token // top_k`` gather on A, the per-expert ``stride_be`` weight
offset, the ``even_Ks`` fast path, the grouped ``GROUP_SIZE_M`` pid swizzle -- is a direct
mirror of sglang's kernel, because "mimic existing high performance libraries first" is the
study's stated priority.  Only the *mapping* (BLOCK_M/N/K, num_warps, num_stages, GROUP_M,
and the H200 axes below) may differ between the two variants, and each is tuned
independently over the same offered grid.

This is also the GEMM that fusion #11a folds the pre-norm into; `kernels/lazy_prenorm.py`
carries its own copy of this shape for that reason, and the two must stay structurally
aligned so that #6's and #11a's numbers remain comparable.

--------------------------------------------------------------------------------------
The H200 axes, and why every one of them is a RUNTIME choice
--------------------------------------------------------------------------------------
Written on a box with no sm_90 in it and never tested against one, so nothing sm_90-
specific is decided at authoring time.  `kernels/hopper.py` decides what this process may
emit; this module only adapts.  The cfg keys the launcher forwards are advertised in
`H200_CFG_KEYS`, which `bench.widen()` reads before adding any overlay -- so on a stack
without these features the grids are byte-identical to the classic ones.

  `USE_TMA` / `TMA_B`
      Fetch the weight tile through a tensor-descriptor box.  **Only B**: A is *gathered*
      through ``sorted_token_ids``, so its rows are not a contiguous box and no descriptor
      can express them.  B is the right half to accelerate anyway -- the expert weights are
      the entire bandwidth story at every decode regime.  If the box cannot be built the
      config RAISES rather than quietly running the pointer mainloop under a TMA label.
  `TMA_MODE`  ("host" | "device", default from `hopper.caps().tma_form()`)
      Host-side descriptors make Triton's launcher call ``cuTensorMapEncodeTiled`` on every
      launch, which at decode lands inside the measured window and is charged to the TMA
      arm alone; the device-side spelling pays global scratch instead.  Both are offered
      and the tuner decides, which is the study's method rather than an assumption.
  `warp_specialize` (+ `num_consumer_groups` / `num_buffers_warp_spec` on forks using the
      launch-kwarg spelling)
      ``tl.range(..., warp_specialize=True)`` on the k-loop.  The loop header is written
      out twice under a constexpr `if` rather than forwarding the flag into the kwarg (some
      Triton builds refuse a `tl.constexpr` there, and that would take the classic path
      down with it).  The body lives once, in `_gateup_step`.
  `num_ctas`
      Thread-block cluster width, forwarded straight to the Triton launch.

Larger tiles: nothing here caps BLOCK_*; `smem_limit()` reads the ceiling off the device
probe.  On C500 the fused variant's best mapping needed 96 KB against a 64 KB limit and was
simply uncompilable; Hopper's ~228 KB per SM removes that bar -- but the bar must come from
the probe, not from this file.

Two deliberate departures from the C500 source, both applied to BOTH variants so neither
can move the ratio:

* accumulation is written ``tl.dot(a, b, acc)`` rather than ``acc += tl.dot(a, b)``.  Same
  arithmetic; it names the wgmma accumulator explicitly.
* when ``FUSE_ACT=False`` the second accumulator is created at shape (1, 1) instead of
  (BLOCK_M, BLOCK_N).  Relying on dead-code elimination to remove an unused loop-carried
  [128, 256] fp32 tile would put the *unfused* arm one missed canonicalisation away from a
  register-pressure cliff -- which would manufacture a fusion win out of nothing.
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


def smem_limit() -> int:
    """Per-block shared-memory ceiling in bytes, from the device probe -- never a literal.

    If the probe cannot supply it we RAISE rather than substitute a plausible constant: a
    grid built from the wrong ceiling prunes the two arms unequally, and that is the one
    failure mode this study cannot detect after the fact.
    """
    n, err = 0, ""
    try:
        try:
            from .. import config as _C  # type: ignore
        except Exception:  # noqa: BLE001 -- also importable as a top-level package
            from glm52_h200 import config as _C  # type: ignore
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
    Both are Triton-side constraints, not device constants.  The bench offers one grid to
    both variants, so this prunes both identically.
    """
    return (
        bool(getattr(caps(), "warp_specialize", False))
        and int(cfg.get("num_warps", 4)) >= 4
        and int(cfg.get("num_stages", 2)) >= 2
    )


def _ws_launch(cfg: dict) -> tuple:
    """(constexpr flag, extra launch kwargs) for this config's warp-specialization request.

    The two spellings live in different places -- the source-level one must reach the
    compiler as a constexpr, the forked-Triton one must reach the launcher as kwargs -- so
    `hopper` answers both and this merges them.  A request neither can serve raises.
    """
    want = bool(cfg.get("warp_specialize", cfg.get("WARP_SPECIALIZE", False)))
    kw = {k: int(cfg[k]) for k in ("num_consumer_groups", "num_buffers_warp_spec") if k in cfg}
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


def _tma_reject(t, block) -> "str | None":
    if _H is None:
        return "kernels/hopper.py unavailable"
    try:
        return _H.tma_reject_reason(t, block)
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


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
    registers.  The box then comes back N-major -- exactly how sglang stores w13
    ([E, 2I, H], H contiguous) -- and `tl.trans` hands it to wgmma, whose B operand
    consumes an N-major fragment directly.  So the descriptor path needs no repacking of
    the production weight layout, which is the whole reason it is worth having here.
    """
    if not w.is_contiguous():
        raise TmaUnsupported("expert weights not contiguous; cannot flatten to [E*N, K]")
    return w.reshape(w.shape[0] * w.shape[1], w.shape[2])


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
    flat = _flat_weight(w)
    why = _tma_reject(flat, [bn, bk])
    if why is not None:
        raise TmaUnsupported(f"w13 box [{bn},{bk}] over [E*2I, H]: {why}")
    if mode == _TMA_HOST:
        d = _H.descriptor(flat, [bn, bk])
        if d is None:
            raise TmaUnsupported(f"hopper.descriptor declined w13 box [{bn},{bk}]")
        return mode, True, d
    if _H is not None:
        # device-side descriptors are built from global scratch, which needs the allocator
        _H.ensure_allocator()
    return mode, True, None


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
    return rep


# ======================================================================================
# The one grouped-GEMM kernel.  FUSE_ACT picks the epilogue.
# ======================================================================================
@triton.jit
def _gateup_step(
    a_ptrs,
    b_ptrs,
    b_desc,
    acc,
    acc2,
    brow,
    kcur,
    offs_k,
    token_mask,
    K,
    stride_bn,
    I: tl.constexpr,
    EVEN_K: tl.constexpr,
    TMA_B: tl.constexpr,
    FUSE_ACT: tl.constexpr,
):
    """One k-step of the mainloop, in ONE place.

    The k-loop header is written out twice (warp-specialized and not); the body is here so
    the two copies are the same text by construction -- the only property of them that
    could move the ratio.

    `brow` is the descriptor's row origin for the GATE tile in the flattened [E*2I, H]
    weight view; the UP tile is exactly `I` rows further on, which is the same `+ I` the
    pointer path applies along w13's N axis.  A TMA box needs no k-mask: an out-of-bounds
    box is zero-filled by the hardware, the same value the masked load supplies.
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
    acc = tl.dot(a, b, acc)

    if FUSE_ACT:
        if TMA_B:
            b2 = tl.trans(b_desc.load([brow + I, kcur]))
        else:
            if EVEN_K:
                b2 = tl.load(b_ptrs + I * stride_bn)
            else:
                b2 = tl.load(
                    b_ptrs + I * stride_bn, mask=offs_k[:, None] < K - kcur, other=0.0
                )
        acc2 = tl.dot(a, b2, acc2)

    return acc, acc2


@triton.jit
def moe_gateup_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_desc,  # host-built descriptor over w13 flattened to [E*2I, H], or None
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # dims
    N,  # width of THIS kernel's output: I when FUSE_ACT else 2*I
    K,
    EM,
    num_valid_tokens,
    b_rows,  # E * 2I -- row extent of the flattened weight view (device-side descriptor)
    # strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # meta
    I: tl.constexpr,  # moe_intermediate_size -- gate/up column split in w13
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
    TMA_B: tl.constexpr,
    TMA_DEVICE: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    FUSE_ACT: tl.constexpr,
):
    # ---- grouped pid swizzle (sglang) --------------------------------------------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- dispatch ------------------------------------------------------------------
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Padded dispatch slots carry the out-of-range sentinel `num_valid_tokens`.  sglang
    # relies on `token_mask` alone; Triton's pipeliner emits speculative (unpredicated)
    # prologue loads on more than one backend, so the sentinel row must not even be
    # *addressed*.  Clamping to row 0 keeps every generated address inside `a`; the value
    # is discarded by `token_mask` exactly as before.  Shared by both variants.
    safe_token = tl.where(token_mask, offs_token, 0)
    a_ptrs = a_ptr + (
        safe_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # Device-side descriptor is built here, from the same base pointer and row stride the
    # pointer path uses, and REBINDS the (None) host-descriptor parameter -- so the mainloop
    # below is one piece of code whichever spelling is live.  `stride_bn` IS the row stride
    # of the flattened [E*2I, H] view (w13.stride(1)), and the launcher has already checked
    # that the innermost stride is 1, which is what makes `strides=[stride_bn, 1]` legal.
    if TMA_B:
        if TMA_DEVICE:
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[b_rows, K],
                strides=[stride_bn, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )

    # Row origin of this program's GATE tile inside the flattened [E*2I, H] view.  w13
    # always has 2*I rows per expert, whichever variant is running: when FUSE_ACT is off
    # the grid simply spans all 2*I of them.  int32 because a descriptor offset is int32
    # and 256 experts x 4096 rows = 2^20, three decimal orders inside the range.
    brow = (off_experts * (2 * I) + pid_n * BLOCK_SIZE_N).to(tl.int32)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if FUSE_ACT:
        acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    else:
        # never read; keeps `_gateup_step`'s arity fixed without asking DCE to remove an
        # unused loop-carried [BM, BN] fp32 tile from the UNFUSED arm.
        acc2 = tl.zeros((1, 1), dtype=tl.float32)

    # The header is duplicated, not parameterised: `warp_specialize=` lowers to an MLIR
    # attribute and some Triton builds refuse a tl.constexpr there, which would break the
    # WS=False path too.  A constexpr `if` has worked in every Triton ever shipped.
    if WARP_SPECIALIZE:
        for k_start in tl.range(0, K, BLOCK_SIZE_K, warp_specialize=True):
            acc, acc2 = _gateup_step(
                a_ptrs, b_ptrs, b_desc, acc, acc2, brow, k_start, offs_k, token_mask,
                K, stride_bn, I, even_Ks, TMA_B, FUSE_ACT,
            )
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
    else:
        for k_start in range(0, K, BLOCK_SIZE_K):
            acc, acc2 = _gateup_step(
                a_ptrs, b_ptrs, b_desc, acc, acc2, brow, k_start, offs_k, token_mask,
                K, stride_bn, I, even_Ks, TMA_B, FUSE_ACT,
            )
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    # ---- epilogue -------------------------------------------------------------------
    if FUSE_ACT:
        out = (acc * tl.sigmoid(acc)) * acc2  # silu(gate) * up, fp32
    else:
        out = acc
    out = out.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * safe_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, out, mask=c_mask)


# ======================================================================================
# The split-out element-wise kernel used only by the unfused chain.
# ======================================================================================
@triton.jit
def silu_and_mul_kernel(
    x_ptr,
    y_ptr,
    M,
    I,
    stride_xm,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(I, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < I)

    base = x_ptr + rm[:, None] * stride_xm + rn[None, :]
    g = tl.load(base, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(base + I, mask=mask, other=0.0).to(tl.float32)
    o = (g * tl.sigmoid(g)) * u
    tl.store(
        y_ptr + rm[:, None] * stride_ym + rn[None, :],
        o.to(y_ptr.dtype.element_ty),
        mask=mask,
    )


# ======================================================================================
# Thin python launchers
# ======================================================================================
def smem_stage_bytes(bm: int, bn: int, bk: int, num_stages: int, bn_mult: int = 1) -> int:
    """Shared memory a Triton GEMM mainloop stages, in bytes.

    Triton 3.0 (the C500 stack) allocated `num_stages` buffers; Triton 3.6 allocates
    `num_stages - 1` with a floor of 2 -- verified on sm89 by launching 68 configs and
    reading `CompiledKernel.metadata.shared`.  The old model over-predicts by 1.33-1.5x and
    rejects configs the hardware can run; every rejected-but-legal config narrows the
    search grid, and if it narrows one arm more than the other it biases the ratio.

    TMA stages the same tile bytes (plus a few mbarriers).  This is an ESTIMATE -- the
    authoritative number is `CompiledKernel.metadata.shared` after a trial compile.
    `glm52_h200.config.smem_stage_bytes` is the same formula; this copy exists so a kernel
    module stays usable standalone.
    """
    return max(2, num_stages - 1) * 2 * bk * (bm + bn_mult * bn)


def smem_bytes(cfg: dict, fused: bool) -> int:
    """Mainloop SMEM footprint.  The fused variant stages a SECOND B tile (the up half)."""
    return smem_stage_bytes(
        cfg["BLOCK_M"],
        cfg["BLOCK_N"],
        cfg["BLOCK_K"],
        cfg["num_stages"],
        bn_mult=2 if fused else 1,
    )


def smem_fits(cfg: dict, fused: bool, limit: "int | None" = None) -> bool:
    """Does this config fit the DEVICE's per-block ceiling?  No literal anywhere."""
    return smem_bytes(cfg, fused) <= (smem_limit() if limit is None else limit)


def launch_gateup(
    a,
    w13,
    c,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    num_valid_tokens,
    top_k,
    I,
    cfg,
    fused: bool,
):
    """`a`: [T, H] bf16.  `w13`: [E, 2I, H] bf16.  `c`: [T*top_k, I or 2I] bf16.

    Classic cfg keys: BLOCK_M/BLOCK_N/BLOCK_K/GROUP_M/num_warps/num_stages.
    H200 cfg keys: see `H200_CFG_KEYS` and the module docstring.
    """
    N = c.shape[1]
    K = a.shape[1]
    EM = sorted_token_ids.shape[0]
    bm, bn, bk = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"]

    mode, tma_b, b_desc = _plan_tma_b(cfg, w13, bn, bk)
    ws_flag, ws_kw = _ws_launch(cfg)

    grid = (triton.cdiv(EM, bm) * triton.cdiv(N, bn),)
    return moe_gateup_kernel[grid](
        a,
        w13,
        c,
        b_desc,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_valid_tokens,
        w13.shape[0] * w13.shape[1],
        a.stride(0),
        a.stride(1),
        w13.stride(0),
        w13.stride(2),
        w13.stride(1),
        c.stride(0),
        c.stride(1),
        I=I,
        BLOCK_SIZE_M=bm,
        BLOCK_SIZE_N=bn,
        BLOCK_SIZE_K=bk,
        GROUP_SIZE_M=cfg["GROUP_M"],
        top_k=top_k,
        compute_type=tl.bfloat16,
        even_Ks=(K % bk == 0),
        TMA_B=tma_b,
        TMA_DEVICE=(mode == _TMA_DEVICE),
        WARP_SPECIALIZE=ws_flag,
        FUSE_ACT=fused,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        num_ctas=int(cfg.get("num_ctas", 1)),
        **ws_kw,
    )


def launch_silu_and_mul(x, y, cfg):
    M, twoI = x.shape
    I = twoI // 2
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(I, cfg["BLOCK_N"]),)
    return silu_and_mul_kernel[grid](
        x,
        y,
        M,
        I,
        x.stride(0),
        y.stride(0),
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
