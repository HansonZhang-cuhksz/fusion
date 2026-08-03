"""Fused/unfused kernel pairs for the GLM-5.2 MoE-layer fusion study, H200 edition.

One module per fusion, each holding **one** kernel source in which `tl.constexpr` flags
select the fused prologue/epilogue. The unfused arm is that same kernel with the flag off
plus the split-out kernel; only the mapping (tile shape, loop order, warps, stages, and on
this device the Hopper levers) may differ between the two arms, and each is tuned
independently. Anything else and the fused/unfused ratio -- the study's only output -- stops
meaning what it says.

`hopper` is the shared feature-abstraction layer, and it is re-exported here so that every
kernel module reaches TMA, warp specialization and thread-block clusters through exactly one
runtime-detected code path:

    from . import hopper
    desc = hopper.descriptor(w, [BLOCK_K, BLOCK_N])        # None -> classic pointer path
    kern[grid](..., WS=hopper.ws_source_flag(use_ws),
               **hopper.ws_kwargs(use_ws), **hopper.cluster_kwargs(n_ctas))

Importing this package pulls in **no** torch, Triton or CUDA state: `hopper` keeps those
imports inside its functions, so capability detection happens on first `caps()` and not as a
side effect of an import. The kernel modules themselves are not imported here -- they are
imported by name by the bench that needs them, so a compile error in one fusion cannot take
out the other ten.
"""

from . import hopper
from .hopper import (
    HopperCaps,
    banner,
    caps,
    caps_dict,
    cluster_choices,
    cluster_kwargs,
    cross_check,
    descriptor,
    ensure_allocator,
    report,
    tma_reject_reason,
    tma_stats,
    ws_choices,
    ws_kwargs,
    ws_mode,
    ws_source_flag,
)

__all__ = [
    "hopper",
    "HopperCaps",
    "banner",
    "caps",
    "caps_dict",
    "cluster_choices",
    "cluster_kwargs",
    "cross_check",
    "descriptor",
    "ensure_allocator",
    "report",
    "tma_reject_reason",
    "tma_stats",
    "ws_choices",
    "ws_kwargs",
    "ws_mode",
    "ws_source_flag",
]
