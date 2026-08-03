"""Hopper (sm_90) feature-abstraction layer: the ONE place that decides, at runtime, which
H200-only mechanism a kernel is allowed to use.

Three levers matter for this study, and **no previous device in it had any of them** -- the
MetaX C500 and the RTX 4060 (sm_89) both lack all three:

* **TMA** (`cp.async.bulk.tensor`, reached through tensor descriptors). One instruction moves
  a whole tile between global and shared memory against a descriptor. It deletes per-element
  address arithmetic from the mainloop and frees the registers that arithmetic occupied --
  which is precisely the resource the *fused* arm of most of these pairs is short of, because
  a fused epilogue costs registers and on C500/4060 that cost showed up as lost occupancy.

* **Warp specialization** (`tl.range(..., warp_specialize=True)`). Producer warps issue the
  copies while consumer warps do the math, instead of every warp alternating. This is the
  mechanism the "Towards Free Normalization" paper relies on to hide a normalization epilogue
  behind a GEMM mainloop; without it, #3/#4/#5/#11 can only overlap by software pipelining,
  which is one reason their measured wins were small on the earlier two devices.

* **Thread-block clusters + DSMEM** (`num_ctas`). Up to 8 CTAs (16 non-portably, which does
  work on H100/H200) are co-scheduled on one GPC and can address each other's shared memory.
  A tile gets a larger effective SMEM budget, and an expert-merge reduction can happen in
  DSMEM instead of through L2.

**Whether any of the three helps here is the experiment, not an assumption.** Every one of the
eleven fusions must still run, and be measured, with all three off; the Hopper path is an
extra arm, never a precondition. So every helper below degrades to a no-op -- `{}` launch
kwargs, a `False` constexpr, a `None` descriptor -- instead of raising, and records what it
decided so the result file can state which arm actually ran.

Detection, in priority order, per capability:

  1. an explicit env override (below), else
  2. an architecture veto: below sm_90 the answer is False regardless of what compiles, else
  3. a one-time trial compile+launch of a tiny kernel, else
  4. `preflight_h200.json`, when it demonstrably describes *this* device and Triton, else
  5. False.

Two deliberate inversions of the obvious ordering, each of which cost something real:

*Why the arch veto outranks a passing probe.* `tl.range(warp_specialize=True)` **compiles and
runs on sm_89** (the preflight in this repo records exactly that) and is then silently not the
Hopper producer/consumer scheme. Reporting a "warp specialization" measurement from a device
that ignored the flag would be a fabricated result, so the cap is ANDed with sm_90+. Set
`GLM52_H200_WS=1` to measure the pre-Hopper behaviour deliberately.

*Why the trial compile outranks the preflight for TMA.* `preflight.py` is fixed and cannot be
edited, and its `tma_tensor_descriptor` probe passes a host-side `TensorDescriptor` object
into a kernel that calls `tl.make_tensor_descriptor(base, ...)`, whose `base` must be a
pointer. That mixes the two TMA APIs, so **a FAIL from that probe is inconclusive on an
sm_90 device** -- it may be reporting the API misuse, not the hardware. Trusting it would
silently disable TMA everywhere on the one machine the suite exists to measure, and the result
file would still say "H200". This module therefore runs its own two-form probe (host-side
descriptor argument, and device-side `tl.make_tensor_descriptor`) and records the
disagreement when there is one.

By default that probe runs in a **subprocess** (`GLM52_H200_PROBE=subproc`). A TMA launch with
a malformed descriptor raises an *asynchronous* illegal-memory-access, which is sticky: it
poisons the CUDA context and every later measurement in the process dies. Nobody can test on
the H200, so a crash costs a whole round trip; a one-off ~10 s subprocess is cheap insurance.
`inproc` and `off` are available when spawning is not.

Env overrides (all optional, all recorded in `caps().sources`):

    GLM52_H200_TMA=0|1          force the TMA capability
    GLM52_H200_WS=0|1           force warp specialization
    GLM52_H200_CLUSTERS=0|1     force thread-block clusters
    GLM52_H200_WGMMA=0|1        force the wgmma verdict
    GLM52_H200_CLASSIC=1        force ALL FOUR off -- the control arm, and the fastest way
                                to prove a Hopper-path result is not an artefact
    GLM52_H200_PROBE=subproc|inproc|off        how the trial compile runs (default subproc)
    GLM52_H200_PROBE_TIMEOUT=<seconds>         subprocess budget (default 300)
    GLM52_H200_PREFLIGHT=<path>                alternate preflight JSON (shared with config.py)
    GLM52_H200_TMA_SCRATCH_REUSE=0             allocate fresh TMA scratch on every launch

Public surface:

    caps()                      -> HopperCaps, cached, never raises
    caps_dict() / banner() / report() / cross_check(env)
    ensure_allocator()          -> bool, registers the global-scratch allocator TMA needs
    descriptor(t, block_shape)  -> TensorDescriptor | None  (None => use the classic path)
    tma_reject_reason(t, blk)   -> str | None, why `descriptor` would decline
    tma_stats()                 -> how many descriptors were built vs declined
    ws_mode() / ws_source_flag(enable) / ws_kwargs(enable) / ws_choices()
    cluster_kwargs(n) / cluster_choices(cands)

`ws_choices()` and `cluster_choices()` exist so an autotuning grid is built from device facts
rather than literals, and -- the part that matters -- so the fused and unfused arms are pruned
by the *same* call. LOG-08 traces several corrupted ratios to guards that pruned one arm only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent          # glm52_h200/kernels
PKG = HERE.parent                               # glm52_h200

# Shared with config.py by name on purpose: one env var moves both readers of the JSON.
PREFLIGHT_PATH = Path(
    os.environ.get("GLM52_H200_PREFLIGHT", PKG / "preflight_h200.json")
)

# sm_90. `>=` everywhere, never `==`: a Blackwell box running this suite should keep the
# Hopper path rather than silently drop to the classic one.
_HOPPER = (9, 0)

_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n"}

# The four capability names, in the order they are reported.
CAP_NAMES = ("tma", "warp_specialize", "clusters", "wgmma")

_ENV_KEYS = {
    "tma": "GLM52_H200_TMA",
    "warp_specialize": "GLM52_H200_WS",
    "clusters": "GLM52_H200_CLUSTERS",
    "wgmma": "GLM52_H200_WGMMA",
}


def _env_tristate(name: str, notes: list) -> "bool | None":
    """`True`/`False` from an env var, or None when unset. An unparsable value is a note,
    never an exception -- detection is not allowed to raise."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    notes.append(f"{name}={raw!r} is neither true nor false; ignored")
    return None


# ======================================================================================
# preflight_h200.json
# ======================================================================================
_PREFLIGHT_CACHE: "dict | None" = None
_PREFLIGHT_STATUS = ""


def preflight_data(path: "Path | str | None" = None) -> dict:
    """Parse the preflight JSON, or `{}` when it is absent or unreadable.

    Absence is normal (a fresh box before `preflight.py` has been run) and must not stop the
    suite; it only means the trial compile has no prior to cross-check against.
    """
    global _PREFLIGHT_CACHE, _PREFLIGHT_STATUS
    if path is None and _PREFLIGHT_CACHE is not None:
        return _PREFLIGHT_CACHE
    p = Path(path) if path is not None else PREFLIGHT_PATH
    data: dict = {}
    status = ""
    try:
        if p.exists():
            data = json.loads(p.read_text())
            if not isinstance(data, dict):
                data, status = {}, f"{p} is not a JSON object"
        else:
            status = f"{p} not found"
    except Exception as exc:  # noqa: BLE001 -- recorded, never fatal
        data, status = {}, f"{p} unreadable: {type(exc).__name__}: {exc}"
    if path is None:
        _PREFLIGHT_CACHE, _PREFLIGHT_STATUS = data, status
    return data


def preflight_status() -> str:
    """Why the default preflight file was not used, or "" when it was."""
    preflight_data()
    return _PREFLIGHT_STATUS


def _preflight_probes(pre: dict) -> dict:
    return ((pre.get("triton_features") or {}).get("compile_probes") or {})


def _probe_verdict(probes: dict, key: str) -> "bool | None":
    entry = probes.get(key)
    if isinstance(entry, dict) and "ok" in entry:
        return bool(entry["ok"])
    return None


# ======================================================================================
# capabilities
# ======================================================================================
@dataclass
class HopperCaps:
    """What this process is allowed to emit. Treat as immutable; `caps()` caches one.

    The four booleans are the contract every kernel module reads. Everything else is
    provenance -- which is not decoration: `n_tried`/`n_failed` per arm plus "which Hopper
    path was live" is what makes an unfair comparison detectable after the fact (LOG-08).
    """

    tma: bool = False
    warp_specialize: bool = False
    clusters: bool = False
    wgmma: bool = False

    # ---- how TMA is reachable. The two forms are not interchangeable, see `tma_form`. ----
    tma_host: bool = False      # host-built TensorDescriptor passed as a kernel argument
    tma_device: bool = False    # device-side tl.make_tensor_descriptor(base, ...)

    # ---- how warp specialization is spelled on this Triton -------------------------------
    ws_mode: str = "none"       # "range" | "launch" | "none"

    # ---- device / stack ------------------------------------------------------------------
    device_name: str = ""
    cc_major: int = 0
    cc_minor: int = 0
    torch_version: str = ""
    triton_version: str = ""

    # ---- provenance ----------------------------------------------------------------------
    sources: dict = field(default_factory=dict)   # cap -> "env"|"probe"|"preflight"|"arch"|"none"
    probe_mode: str = "none"                      # "subproc"|"inproc"|"off"|"skipped"|"none"
    probe_raw: dict = field(default_factory=dict)  # per-probe ok/error, verbatim
    preflight_used: bool = False
    preflight_note: str = ""
    notes: list = field(default_factory=list)
    errors: dict = field(default_factory=dict)

    # ---------------------------------------------------------------------------------
    @property
    def cc(self) -> tuple:
        return (self.cc_major, self.cc_minor)

    @property
    def sm_arch(self) -> str:
        return f"sm_{self.cc_major}{self.cc_minor}"

    @property
    def arch_ok(self) -> bool:
        """sm_90 or newer: the three levers are architecturally present."""
        return self.cc >= _HOPPER

    @property
    def any_hopper(self) -> bool:
        """Is any Hopper path live at all? False means every bench is running the classic
        arm, which is a valid -- and clearly labelled -- outcome, not a failure."""
        return bool(self.tma or self.warp_specialize or self.clusters)

    def tma_form(self) -> str:
        """Which TMA spelling a kernel should emit: "device", "host" or "none".

        "device" is preferred when both work. A host-side descriptor argument makes Triton's
        launcher call `fill_tma_descriptor` (a `cuTensorMapEncodeTiled`) on the host **on
        every launch**; at decode sizes a kernel resolves to only 9-17 CUDA-event ticks, so
        that per-launch host tax lands inside the measured window and is charged to the TMA
        arm alone. The device-side form pays global scratch instead, which `ensure_allocator`
        already keeps out of the allocator path.
        """
        if not self.tma:
            return "none"
        if self.tma_device:
            return "device"
        return "host" if self.tma_host else "none"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["sm_arch"] = self.sm_arch
        d["arch_ok"] = self.arch_ok
        d["tma_form"] = self.tma_form()
        d["any_hopper"] = self.any_hopper
        return d


_CAPS: "HopperCaps | None" = None
# Reentrant on purpose: `_detect()` runs while holding it, and a future helper that reached
# for `caps()` from inside detection would otherwise deadlock the whole benchmark -- a
# failure mode that looks exactly like a hung GPU from the outside.
_CAPS_LOCK = threading.RLock()


def caps() -> HopperCaps:
    """The cached capability verdict for this process. Never raises.

    Cost: zero on anything below sm_90 (the arch veto short-circuits before any probe), one
    subprocess on the first call on an sm_90+ box, nothing afterwards.
    """
    global _CAPS
    if _CAPS is None:
        with _CAPS_LOCK:
            if _CAPS is None:
                try:
                    _CAPS = _detect()
                except Exception as exc:  # noqa: BLE001
                    # A detection bug must cost the Hopper path, not the run. All-False is
                    # always executable: every fusion has a working classic arm.
                    c = HopperCaps()
                    c.errors["detect"] = f"{type(exc).__name__}: {exc}"
                    c.notes.append(
                        "capability detection raised; all Hopper features disabled and the "
                        "classic path used everywhere. The measurement is still valid, it is "
                        "just not an H200-feature measurement."
                    )
                    c.sources = {k: "none" for k in CAP_NAMES}
                    _CAPS = c
    return _CAPS


def reset_caps() -> None:
    """Drop the cache. For tests and for re-reading after changing an env var; a benchmark
    must never call this mid-run -- the recorded provenance would stop describing the run."""
    global _CAPS, _PREFLIGHT_CACHE
    with _CAPS_LOCK:
        _CAPS = None
    _PREFLIGHT_CACHE = None


def caps_dict() -> dict:
    """JSON-safe capability block for a result file's `_meta`."""
    return caps().as_dict()


def banner() -> str:
    c = caps()
    return (
        f"[hopper] {c.device_name or '?'} ({c.sm_arch}) | tma={c.tma}"
        f"{'(' + c.tma_form() + ')' if c.tma else ''} "
        f"ws={c.warp_specialize}{'(' + c.ws_mode + ')' if c.warp_specialize else ''} "
        f"clusters={c.clusters} wgmma={c.wgmma} | "
        f"src {'/'.join(sorted(set(c.sources.values())))} | probe {c.probe_mode}"
    )


def report() -> str:
    """Multi-line human summary: the banner, then every note and error. Print this once at
    bench startup -- a silently-degraded Hopper path is the failure mode this whole module
    exists to make visible."""
    c = caps()
    lines = [banner()]
    for k in CAP_NAMES:
        lines.append(f"  {k:<16} {getattr(c, k)!s:<6} source={c.sources.get(k, '?')}")
    if c.preflight_note:
        lines.append(f"  preflight: {c.preflight_note}")
    for n in c.notes:
        lines.append(f"  [note] {n}")
    for k, v in c.errors.items():
        lines.append(f"  [err ] {k}: {v}")
    return "\n".join(lines)


def cross_check(env_obj) -> list:
    """Compare against `config.BenchEnv`'s preflight-derived flags; return disagreements.

    Deliberately takes the env object as an argument instead of importing `config`: kernels
    must not depend on the harness (and a cycle here would be a very silly way to lose a
    round trip). A disagreement is expected and *informative* for TMA -- see the module
    docstring on why the preflight's TMA probe is inconclusive -- so it is reported, not
    resolved.
    """
    c = caps()
    out = []
    for cap, attr in (
        ("tma", "tma_supported"),
        ("clusters", "clusters_supported"),
        ("warp_specialize", "warp_spec_supported"),
    ):
        theirs = getattr(env_obj, attr, None)
        if theirs is None:
            continue
        mine = getattr(c, cap)
        if bool(theirs) != bool(mine):
            out.append(
                f"{cap}: hopper.caps()={mine} (source {c.sources.get(cap)}) vs "
                f"BenchEnv.{attr}={bool(theirs)} (preflight). "
                f"The kernels follow hopper.caps()."
            )
    return out


# --------------------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------------------
def _live_device(c: HopperCaps) -> bool:
    """Fill in device/stack identity from torch. torch, not Triton: Triton's property query
    JIT-builds and dlopens a C extension on first use, so it is exactly the call that fails
    on a fresh box -- and this module must never answer from a stale table (see config.py's
    BenchEnv.probe, and the C500-defaults bug it documents)."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        c.errors["torch_import"] = f"{type(exc).__name__}: {exc}"
        c.notes.append("torch unimportable; no device, no Hopper features")
        return False
    c.torch_version = getattr(torch, "__version__", "")
    try:
        if not torch.cuda.is_available():
            c.notes.append("torch.cuda.is_available() is False; no Hopper features")
            return False
        idx = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(idx)
        c.device_name = p.name
        c.cc_major, c.cc_minor = int(p.major), int(p.minor)
    except Exception as exc:  # noqa: BLE001
        c.errors["device"] = f"{type(exc).__name__}: {exc}"
        return False
    try:
        import triton

        c.triton_version = getattr(triton, "__version__", "")
    except Exception as exc:  # noqa: BLE001
        c.errors["triton_import"] = f"{type(exc).__name__}: {exc}"
        c.notes.append("triton unimportable; no Hopper features (and no kernels either)")
        return False
    return True


def _preflight_view(c: HopperCaps) -> dict:
    """Preflight's verdicts, but only if the file describes THIS device and THIS Triton.

    Lesson 4 generalised: a payload from another GPU is worse than no payload, because it is
    plausible. A stale-device preflight would have TMA=False from the 4060 and would sit,
    unremarked, inside a file labelled H200.
    """
    pre = preflight_data()
    if not pre:
        c.preflight_note = preflight_status() or "no preflight"
        return {}
    d = pre.get("device") or {}
    s = pre.get("stack") or {}
    same_dev = (
        str(d.get("name", "")) == c.device_name
        and str(d.get("compute_capability", "")) == f"{c.cc_major}.{c.cc_minor}"
    )
    same_stack = str(s.get("triton", "")) == c.triton_version
    if not same_dev:
        c.preflight_note = (
            f"{PREFLIGHT_PATH.name} describes {d.get('name')!r} "
            f"(sm_{d.get('major')}{d.get('minor')}) but this is {c.device_name!r} "
            f"({c.sm_arch}); its feature verdicts are IGNORED. Re-run preflight.py here."
        )
        return {}
    if not same_stack:
        c.preflight_note = (
            f"{PREFLIGHT_PATH.name} was written under triton {s.get('triton')} but this "
            f"process has triton {c.triton_version}; feature verdicts are IGNORED "
            f"(what compiles is a property of the compiler, not only the device)."
        )
        return {}
    c.preflight_used = True
    c.preflight_note = f"{PREFLIGHT_PATH.name} matches this device and stack ({pre.get('timestamp')})"
    probes = _preflight_probes(pre)
    return {
        "tma": _probe_verdict(probes, "tma_tensor_descriptor"),
        "ws_range": _probe_verdict(probes, "warp_specialize_tl_range"),
        "ws_launch": _probe_verdict(probes, "warp_specialize_num_consumer_groups"),
        "clusters": _probe_verdict(probes, "thread_block_cluster_num_ctas"),
        "dot_bf16": _probe_verdict(probes, "tl_dot_bf16"),
    }


def _detect() -> HopperCaps:
    c = HopperCaps()
    notes = c.notes

    overrides = {k: _env_tristate(v, notes) for k, v in _ENV_KEYS.items()}
    if _env_tristate("GLM52_H200_CLASSIC", notes) is True:
        overrides = {k: False for k in CAP_NAMES}
        notes.append("GLM52_H200_CLASSIC=1: all Hopper features forced off (control arm)")

    if not _live_device(c):
        for k in CAP_NAMES:
            c.sources[k] = "none"
        _apply_overrides(c, overrides)
        return c

    pv = _preflight_view(c)

    # --- the architecture veto ---------------------------------------------------------
    # Below sm_90 none of the four is real, whatever compiles. Short-circuiting here also
    # means `caps()` costs nothing on the sm_89 box this file was written on.
    if not c.arch_ok and not any(v is True for v in overrides.values()):
        for k in CAP_NAMES:
            c.sources[k] = "arch"
        c.probe_mode = "skipped"
        notes.append(
            f"{c.sm_arch} < sm_90: TMA, warp specialization, clusters and wgmma are all "
            f"unavailable; every bench runs its classic arm. This is a correct, fully "
            f"measurable configuration -- it is simply not an H200 measurement."
        )
        _apply_overrides(c, overrides)
        return c

    # --- trial compile+launch ----------------------------------------------------------
    mode = (os.environ.get("GLM52_H200_PROBE") or "subproc").strip().lower()
    if mode not in ("subproc", "inproc", "off"):
        notes.append(f"GLM52_H200_PROBE={mode!r} unrecognised; using 'subproc'")
        mode = "subproc"
    # Nothing left to learn if every capability is pinned by env.
    if all(v is not None for v in overrides.values()):
        mode = "off"
        notes.append("all four capabilities pinned by env; trial compile skipped")

    probe: dict = {}
    if mode == "subproc":
        probe = _probe_subprocess(c)
    elif mode == "inproc":
        probe = _probe_inprocess()
    c.probe_mode = mode
    c.probe_raw = probe.get("probes", {}) if probe else {}
    for k, v in (probe.get("errors") or {}).items():
        c.errors[f"probe.{k}"] = v

    def _p(key: str) -> "bool | None":
        e = c.probe_raw.get(key)
        return bool(e["ok"]) if isinstance(e, dict) and "ok" in e else None

    def _resolve(cap: str, probe_keys: tuple, pre_key: "str | None", record: bool = True) -> bool:
        """probe > preflight > False. `record=False` for the sub-flags (which TMA spelling,
        which warp-spec spelling) that feed a capability rather than being one."""
        vals = [_p(k) for k in probe_keys]
        if any(v is not None for v in vals):
            if record:
                c.sources[cap] = "probe"
            return any(v is True for v in vals)
        pv_val = pv.get(pre_key) if pre_key else None
        if record:
            c.sources[cap] = "preflight" if pv_val is not None else "none"
        return bool(pv_val)

    c.tma_host = _resolve("tma", ("tma_host_descriptor",), None, record=False)
    c.tma_device = _resolve("tma", ("tma_device_descriptor",), None, record=False)
    if _p("tma_host_descriptor") is None and _p("tma_device_descriptor") is None:
        # No probe ran (off, or it failed to start). Fall back to preflight, knowing that a
        # False there is inconclusive -- say so rather than pretending otherwise.
        c.tma = bool(pv.get("tma")) if pv.get("tma") is not None else False
        c.tma_host = c.tma
        c.sources["tma"] = "preflight" if pv.get("tma") is not None else "none"
        if pv.get("tma") is False:
            notes.append(
                "TMA disabled on the preflight's say-so, and no trial compile ran. That "
                "probe passes a host TensorDescriptor into tl.make_tensor_descriptor(), "
                "which wants a pointer, so its FAIL may be API misuse rather than missing "
                "hardware. Re-run with GLM52_H200_PROBE=inproc, or GLM52_H200_TMA=1, if TMA "
                "is expected on this device."
            )
    else:
        c.tma = bool(c.tma_host or c.tma_device)
        c.sources["tma"] = "probe"
        if pv.get("tma") is not None and bool(pv["tma"]) != c.tma:
            notes.append(
                f"TMA disagreement: preflight probe says {bool(pv['tma'])}, this module's "
                f"probe says {c.tma} (host={c.tma_host} device={c.tma_device}). The local "
                f"probe wins; see the module docstring for why the preflight's is ambiguous."
            )

    # Warp specialization: two spellings, and they are NOT interchangeable. `tl.range(...,
    # warp_specialize=True)` is source-level, so the kernel has to carry a constexpr;
    # `num_consumer_groups`/`num_buffers_warp_spec` are launch kwargs on the older forked
    # Triton and only mean anything with `tl.async_task` regions in the source. Prefer the
    # source-level form when both work: that is the one this suite's kernels are written for.
    ws_range = _resolve("warp_specialize", ("ws_tl_range",), "ws_range", record=False)
    ws_launch = _resolve(
        "warp_specialize", ("ws_num_consumer_groups",), "ws_launch", record=False
    )
    if ws_range:
        c.ws_mode, c.warp_specialize = "range", True
        c.sources["warp_specialize"] = "probe" if _p("ws_tl_range") is not None else "preflight"
    elif ws_launch and _p("async_task_api") is not False:
        c.ws_mode, c.warp_specialize = "launch", True
        c.sources["warp_specialize"] = (
            "probe" if _p("ws_num_consumer_groups") is not None else "preflight"
        )
        notes.append(
            "warp specialization is the launch-kwarg flavour (num_consumer_groups); the "
            "kernels in this suite are written for tl.range(warp_specialize=...) and will "
            "not be specialized by it unless they also carry tl.async_task regions"
        )
    else:
        c.ws_mode, c.warp_specialize = "none", False
        if _p("ws_tl_range") is not None or _p("ws_num_consumer_groups") is not None:
            c.sources["warp_specialize"] = "probe"
        elif pv.get("ws_range") is not None or pv.get("ws_launch") is not None:
            c.sources["warp_specialize"] = "preflight"
        else:
            c.sources["warp_specialize"] = "none"

    c.clusters = _resolve("clusters", ("cluster_num_ctas",), "clusters")
    # wgmma is not a switch. Triton selects it for sm_90 when the tile shape and dtype allow,
    # so the cap says "wgmma-class tensor cores are reachable" -- it exists to let a grid
    # builder prefer wgmma-eligible tiles (BM >= 64, BN a multiple of 8, bf16 operands), not
    # to toggle anything at launch.
    c.wgmma = _resolve("wgmma", ("dot_bf16",), "dot_bf16")

    # The arch veto again, now against probe results: a pass on the wrong arch is not
    # evidence (sm_89 compiles warp_specialize=True and then ignores it).
    if not c.arch_ok:
        vetoed, kept = [], []
        for k in CAP_NAMES:
            if not getattr(c, k):
                continue
            if overrides.get(k) is True:
                kept.append(k)
                continue
            setattr(c, k, False)
            c.sources[k] = "arch"
            vetoed.append(k)
        if vetoed:
            notes.append(
                f"{c.sm_arch} < sm_90: probe passed for {', '.join(vetoed)} but the arch veto "
                f"turns them off -- they compile here and are then silently ignored"
            )
        if kept:
            notes.append(
                f"{c.sm_arch} < sm_90 and {', '.join(kept)} kept ON by explicit env "
                f"override: whatever this run measures, it is not the Hopper mechanism"
            )

    _apply_overrides(c, overrides)
    return c


def _apply_overrides(c: HopperCaps, overrides: dict) -> None:
    """Env wins over everything, and says so in `sources`. Forcing a capability ON when the
    probe says otherwise is a legitimate debugging move and a fast way to get one more data
    point out of a machine nobody can log into -- so it is allowed, and loudly recorded."""
    for k, v in overrides.items():
        if v is None:
            continue
        was = getattr(c, k)
        setattr(c, k, v)
        c.sources[k] = "env"
        if v and k == "tma" and not (c.tma_host or c.tma_device):
            c.tma_host = True  # nothing else to try; descriptor() will still validate
        if v and k == "warp_specialize" and c.ws_mode == "none":
            c.ws_mode = "range"
        if was != v:
            c.notes.append(
                f"{_ENV_KEYS[k]} forced {k}={v} (detection said {was}); "
                f"this result is NOT a clean feature measurement"
            )
    # Keep the reported state self-consistent: `ws_mode` describes how warp specialization
    # would be spelled, and must not survive the capability being off -- `ws_source_flag`
    # already checks both, but a result file that says ws=False/mode=range invites the
    # reader to conclude the wrong thing about which arm ran.
    if not c.warp_specialize:
        c.ws_mode = "none"


# ======================================================================================
# trial compile + launch
# ======================================================================================
_PROBE_MARKER = "@@GLM52_HOPPER_PROBE@@"


def _probe_inprocess() -> dict:
    """Compile AND launch one tiny kernel per capability. Attribute existence is not
    evidence -- several Triton releases export symbols that fail at compile time, which is
    why preflight.py probes this way too.

    Ordered safest-first, and `synchronize()`d after each: a TMA fault is asynchronous, so
    without the sync a later, unrelated measurement would be the thing that dies.
    """
    out: dict = {"probes": {}, "errors": {}}
    try:
        import torch
        import triton
        import triton.language as tl
    except Exception as exc:  # noqa: BLE001
        out["errors"]["import"] = f"{type(exc).__name__}: {exc}"
        return out

    # ---- probe kernels. Nested so that importing this module never imports triton. ------
    @triton.jit
    def _p_copy(X, Y, N, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        tl.store(Y + i, tl.load(X + i, mask=m, other=0.0) * 2.0, mask=m)

    @triton.jit
    def _p_dot(A, B, C, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a = tl.load(A + rm[:, None] * BK + rk[None, :])
        b = tl.load(B + rk[:, None] * BN + rn[None, :])
        tl.store(C + rm[:, None] * BN + rn[None, :], tl.dot(a, b, out_dtype=tl.float32))

    @triton.jit
    def _p_ws(X, Y, N, BLOCK: tl.constexpr):
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.range(0, N, BLOCK, warp_specialize=True):
            i = k + tl.arange(0, BLOCK)
            acc += tl.load(X + i, mask=i < N, other=0.0)
        tl.store(Y + tl.arange(0, BLOCK), acc)

    @triton.jit
    def _p_tma_host(Desc, Out, BM: tl.constexpr, BN: tl.constexpr):
        t = Desc.load([0, 0])
        tl.store(Out + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], t)

    @triton.jit
    def _p_tma_dev(X, Out, M, N, BM: tl.constexpr, BN: tl.constexpr):
        d = tl.make_tensor_descriptor(X, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
        t = d.load([0, 0])
        tl.store(Out + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], t)

    stop = {"hit": False}

    def run(name: str, fn) -> None:
        if stop["hit"]:
            out["probes"][name] = {"ok": False, "error": "skipped: context suspect"}
            return
        try:
            fn()
            torch.cuda.synchronize()
            out["probes"][name] = {"ok": True}
        except Exception as exc:  # noqa: BLE001
            out["probes"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
            # If the *sync* is what failed, the context may be poisoned; stop touching it.
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                stop["hit"] = True
                out["errors"]["context"] = "CUDA context faulted during probing"

    # Everything below touches the device. It is guarded as a whole as well as per probe:
    # an allocation failure between probes must still return the results already gathered,
    # because "TMA worked, then we ran out of memory" and "TMA never worked" are different
    # facts and only one of them means the Hopper path should be off.
    try:
        N = 4096
        x = torch.randn(N, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        run("baseline", lambda: _p_copy[(N // 256,)](x, y, N, BLOCK=256))

        a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        cc = torch.empty(64, 64, device="cuda", dtype=torch.float32)
        run("dot_bf16", lambda: _p_dot[(1,)](a, b, cc, BM=64, BN=64, BK=64))

        # num_warps=4 is Triton's own precondition for the warp-specialize transform on
        # several releases; a config that cannot be specialized simply fails to compile, and
        # the autotuner records it as a failed config rather than guessing a rule here.
        run("ws_tl_range", lambda: _p_ws[(1,)](x, y, N, BLOCK=256, num_warps=4))
        run(
            "ws_num_consumer_groups",
            lambda: _p_copy[(N // 256,)](
                x, y, N, BLOCK=256, num_consumer_groups=1, num_buffers_warp_spec=2
            ),
        )
        out["probes"]["async_task_api"] = {"ok": hasattr(tl, "async_task")}

        # TMA, both spellings, with the allocator registered first -- forgetting it is the
        # classic silent failure (the launcher asks for global scratch and gets None).
        src = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        dst = torch.empty(64, 64, device="cuda", dtype=torch.bfloat16)
        out["probes"]["allocator"] = {"ok": bool(ensure_allocator())}

        def _tma_host():
            from triton.tools.tensor_descriptor import TensorDescriptor

            desc = TensorDescriptor.from_tensor(src, [64, 64])
            _p_tma_host[(1,)](desc, dst, BM=64, BN=64)

        run("tma_host_descriptor", _tma_host)
        run(
            "tma_device_descriptor",
            lambda: _p_tma_dev[(1,)](src, dst, 128, 128, BM=64, BN=64),
        )

        # Clusters. Triton multiplies gridDimX by num_ctas, so the grid needs no
        # divisibility fix-up; a too-large cluster fails at launch and is recorded like any
        # other config.
        run("cluster_num_ctas", lambda: _p_copy[(2,)](x, y, N, BLOCK=256, num_ctas=2))
    except Exception as exc:  # noqa: BLE001
        out["errors"]["probe_body"] = f"{type(exc).__name__}: {exc}"[:400]
    return out


def _probe_subprocess(c: HopperCaps) -> dict:
    """Run `_probe_inprocess` in a fresh interpreter and read back its JSON.

    Why: an ill-formed TMA descriptor raises an asynchronous illegal-memory-access, which is
    sticky -- the CUDA context is dead and every later measurement in the process fails. On a
    machine nobody can log into, that is a whole wasted round trip. A second context costs a
    few hundred MB of the H200's 143 GB and ~10 s, once.

    Failure handling is deliberately asymmetric. A non-zero exit or a timeout is *evidence
    the probe itself killed the interpreter*, so we do not then repeat it in this one. A
    spawn failure (no interpreter, sandbox) tells us nothing about the GPU, so we retry
    in-process rather than throw away the whole Hopper path.
    """
    try:
        timeout = float(os.environ.get("GLM52_H200_PROBE_TIMEOUT", "300"))
    except ValueError:
        timeout = 300.0
    cmd = [sys.executable, str(Path(__file__).resolve()), "--probe-json"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as exc:
        c.errors["probe_subprocess"] = f"{type(exc).__name__}: {exc}"[:300]
        c.notes.append(
            "the capability probe timed out or died in its own subprocess -- which is what "
            "the subprocess is for. Treating all Hopper features as unavailable; the classic "
            "arms still measure correctly. GLM52_H200_PROBE=inproc to see the crash."
        )
        return {}
    except OSError as exc:
        c.errors["probe_spawn"] = f"{type(exc).__name__}: {exc}"[:300]
        c.notes.append("could not spawn the probe subprocess; falling back to in-process")
        return _probe_inprocess()
    for line in reversed((p.stdout or "").splitlines()):
        if line.startswith(_PROBE_MARKER):
            try:
                return json.loads(line[len(_PROBE_MARKER):])
            except Exception as exc:  # noqa: BLE001
                c.errors["probe_parse"] = f"{type(exc).__name__}: {exc}"[:200]
                break
    c.errors["probe_subprocess"] = (
        f"rc={p.returncode} no probe JSON on stdout; stderr: {(p.stderr or '')[-400:]}"
    )
    if p.returncode == 0:
        c.notes.append("probe subprocess exited cleanly but printed nothing; retrying inproc")
        return _probe_inprocess()
    c.notes.append(
        f"probe subprocess exited rc={p.returncode}; NOT retried in-process (a crash there "
        f"would take the benchmark with it). All Hopper features off."
    )
    return {}


# ======================================================================================
# TMA: allocator + descriptors
# ======================================================================================
_ALLOC = {
    "installed": False,
    "buffers": {},      # stream id -> int8 scratch tensor
    "peak_bytes": 0,
    "calls": 0,
    "reuse": os.environ.get("GLM52_H200_TMA_SCRATCH_REUSE", "1").strip().lower() not in _FALSE,
}


def _scratch(size: int, alignment: int, stream):
    """Triton's global-scratch allocator.

    Called on **every launch** of a kernel whose `global_scratch_size > 0` -- which is every
    device-side-descriptor kernel -- with `grid * num_ctas * per_program_bytes`. Triton's own
    example allocates a fresh tensor each time; here that allocation would sit inside the
    timed region and be charged to the TMA arm only, and at decode sizes the whole kernel is
    9-17 event ticks. So the buffer is cached per stream and grown geometrically, and reuse
    is safe because launches on one stream are serialized. `GLM52_H200_TMA_SCRATCH_REUSE=0`
    restores the allocate-every-time behaviour for anyone who wants to check that this
    caching is not itself the effect being measured.
    """
    import torch

    _ALLOC["calls"] += 1
    _ALLOC["peak_bytes"] = max(_ALLOC["peak_bytes"], int(size))
    pad = alignment if alignment and alignment > 512 else 0  # torch already gives 512 B
    if not _ALLOC["reuse"]:
        buf = torch.empty(size + pad, dtype=torch.int8, device="cuda")
    else:
        key = int(stream) if stream is not None else -1
        buf = _ALLOC["buffers"].get(key)
        if buf is None or buf.numel() < size + pad:
            grow = max(size + pad, 2 * (buf.numel() if buf is not None else 0), 4096)
            buf = torch.empty(grow, dtype=torch.int8, device="cuda")
            _ALLOC["buffers"][key] = buf
    if pad:
        off = (-buf.data_ptr()) % alignment
        return buf[off:off + size]
    return buf[:size]


def ensure_allocator() -> bool:
    """Register the global scratch allocator that TMA descriptors require. Idempotent.

    H200 Triton needs a scratch allocator registered before ANY descriptor kernel launches;
    forgetting it is the classic silent failure, because the symptom is a launch error deep
    inside the driver rather than anything that names TMA. Call this from every launcher that
    might take the TMA path -- it costs a dict lookup after the first time.
    """
    if _ALLOC["installed"]:
        return True
    try:
        import triton

        if not hasattr(triton, "set_allocator"):
            return False
        triton.set_allocator(_scratch)
        _ALLOC["installed"] = True
        return True
    except Exception:  # noqa: BLE001 -- caller falls back to the classic path
        return False


_TMA_STATS = {"built": 0, "declined": 0, "reasons": {}}


def tma_reject_reason(tensor, block_shape) -> "str | None":
    """Why `descriptor()` would decline, or None if it would succeed.

    Every rule here is a hard TMA/Triton requirement (16-byte aligned base, 16-byte aligned
    leading strides, contiguous innermost dimension, power-of-two block dims, innermost block
    row a multiple of 16 bytes). They are checked explicitly rather than left to the
    `assert`s inside `TensorDescriptor.__post_init__` because those vanish under `python -O`,
    and a silently-wrong descriptor is an async fault, not an exception.

    The rules are identical for the device-side spelling, so a kernel emitting
    `tl.make_tensor_descriptor` should gate on `tma_reject_reason(...) is None` and still
    call `ensure_allocator()` -- it needs global scratch just the same.
    """
    c = caps()
    if not c.tma:
        return f"tma capability off (source {c.sources.get('tma')})"
    try:
        rank = tensor.dim()
        blk = list(block_shape)
        if len(blk) != rank:
            return f"block rank {len(blk)} != tensor rank {rank}"
        if not (2 <= rank <= 5):
            return f"rank {rank} outside TMA's 2..5"
        itemsize = tensor.element_size()
        if tensor.data_ptr() % 16:
            return "base pointer not 16-byte aligned"
        strides = list(tensor.stride())
        if strides[-1] != 1:
            return "innermost dimension is not contiguous"
        for i, s in enumerate(strides[:-1]):
            if (s * itemsize) % 16:
                return f"stride[{i}]={s} is not a multiple of 16 bytes"
        numel = 1
        for i, d in enumerate(blk):
            if not isinstance(d, int) or d <= 0 or (d & (d - 1)):
                return f"block_shape[{i}]={d} is not a positive power of two"
            numel *= d
        if numel > 1048576:  # triton._utils.TRITON_MAX_TENSOR_NUMEL
            return f"block numel {numel} exceeds Triton's tensor limit"
        if (blk[-1] * itemsize) % 16:
            return f"innermost block row {blk[-1]}*{itemsize}B is not a multiple of 16 B"
        for i, d in enumerate(tensor.shape):
            if d <= 0:
                return f"shape[{i}]={d} is not positive"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


def descriptor(tensor, block_shape, padding: str = "zero"):
    """Host-side TMA descriptor for `tensor`, or **None** when TMA cannot describe it.

    None is a normal answer, not an error: the caller takes its classic pointer-arithmetic
    path. Returning None rather than raising is what lets a single kernel source serve both
    arms on both architectures, which is the property that keeps the fused/unfused comparison
    honest -- the two arms must differ in mapping, not in which machine they can run on.

    Note the cost model: a host-side descriptor argument makes Triton's launcher call
    `fill_tma_descriptor` (a host `cuTensorMapEncodeTiled`) on every launch. That is fine at
    prefill and material at decode; see `HopperCaps.tma_form`, which prefers the device-side
    spelling when both are available.
    """
    why = tma_reject_reason(tensor, block_shape)
    if why is not None:
        _TMA_STATS["declined"] += 1
        _TMA_STATS["reasons"][why] = _TMA_STATS["reasons"].get(why, 0) + 1
        return None
    if not ensure_allocator():
        _TMA_STATS["declined"] += 1
        _TMA_STATS["reasons"]["no global scratch allocator"] = (
            _TMA_STATS["reasons"].get("no global scratch allocator", 0) + 1
        )
        return None
    try:
        from triton.tools.tensor_descriptor import TensorDescriptor

        desc = TensorDescriptor.from_tensor(tensor, list(block_shape), padding)
    except TypeError:
        # Older signature without `padding`.
        try:
            from triton.tools.tensor_descriptor import TensorDescriptor

            desc = TensorDescriptor.from_tensor(tensor, list(block_shape))
        except Exception as exc:  # noqa: BLE001
            _TMA_STATS["declined"] += 1
            _TMA_STATS["reasons"][f"{type(exc).__name__}"] = (
                _TMA_STATS["reasons"].get(f"{type(exc).__name__}", 0) + 1
            )
            return None
    except Exception as exc:  # noqa: BLE001
        _TMA_STATS["declined"] += 1
        _TMA_STATS["reasons"][f"{type(exc).__name__}: {exc}"[:120]] = (
            _TMA_STATS["reasons"].get(f"{type(exc).__name__}: {exc}"[:120], 0) + 1
        )
        return None
    _TMA_STATS["built"] += 1
    return desc


def tma_stats() -> dict:
    """Descriptors built vs declined, and why. Record this next to `n_tried`/`n_failed`: a
    TMA arm that quietly declined every descriptor and ran the classic path all along is
    otherwise indistinguishable from a TMA arm that did nothing useful."""
    return {
        "built": _TMA_STATS["built"],
        "declined": _TMA_STATS["declined"],
        "reasons": dict(_TMA_STATS["reasons"]),
        "scratch_calls": _ALLOC["calls"],
        "scratch_peak_bytes": _ALLOC["peak_bytes"],
        "scratch_reuse": bool(_ALLOC["reuse"]),
        "allocator_installed": bool(_ALLOC["installed"]),
    }


# ======================================================================================
# warp specialization
# ======================================================================================
def ws_mode() -> str:
    """"range" (source-level `tl.range(warp_specialize=...)`), "launch" (`num_consumer_groups`
    launch kwargs) or "none"."""
    return caps().ws_mode


def ws_source_flag(enable: bool = True) -> bool:
    """Value for the kernel's `WS: tl.constexpr`, i.e. what to pass to
    `tl.range(..., warp_specialize=WS)`.

    Split from `ws_kwargs` because the two mechanisms live in different places: the
    source-level form must reach the *compiler* as a constexpr, the older forked-Triton form
    must reach the *launcher* as kwargs. A kernel wanting warp specialization writes

        kern[grid](..., WS=hopper.ws_source_flag(True), **hopper.ws_kwargs(True))

    and gets the classic mainloop, unchanged, on any device or Triton lacking either.
    """
    return bool(enable and caps().warp_specialize and caps().ws_mode == "range")


def ws_kwargs(
    enable: bool = True,
    num_consumer_groups: int = 1,
    num_buffers_warp_spec: int = 2,
) -> dict:
    """Launch kwargs for warp specialization on THIS Triton, or `{}`.

    `{}` for the "range" mode is correct, not a failure: there the enabling happens at source
    level (see `ws_source_flag`). Passing an unrecognised kwarg is not a soft failure in
    Triton -- it raises `KeyError: 'Keyword argument num_consumer_groups was specified but
    unrecognised'`, which is why this is decided from a probe rather than tried optimistically
    inside a launcher.
    """
    if not enable or not caps().warp_specialize or caps().ws_mode != "launch":
        return {}
    return {
        "num_consumer_groups": int(num_consumer_groups),
        "num_buffers_warp_spec": int(num_buffers_warp_spec),
    }


def ws_choices(include_off: bool = True) -> tuple:
    """The warp-specialization settings an autotuning grid should sweep: `(False,)` when the
    feature is unavailable, `(False, True)` when it is.

    Both arms of a fused/unfused pair must call this, so the grid is pruned identically on
    both sides. LOG-08 traces several corrupted ratios to a guard that pruned one arm only.
    """
    if not caps().warp_specialize:
        return (False,)
    return (False, True) if include_off else (True,)


# ======================================================================================
# thread-block clusters
# ======================================================================================
def cluster_kwargs(n: int = 2) -> dict:
    """`{"num_ctas": n}` when clusters are supported and n > 1, else `{}`.

    No clamping on `n`. Hopper takes up to 8 portably and 16 non-portably (Triton's CUDA
    launcher special-cases 16 for H100/H200), but the exact ceiling interacts with SMEM per
    CTA and register pressure, so an over-large cluster is left to fail at compile/launch
    and be recorded as a failed config -- the same treatment every other tile parameter gets.
    Inventing a limit here would prune the grid from a literal, which is the one thing this
    suite has agreed never to do.

    Note that `num_ctas` does *not* change the grid you pass: Triton multiplies gridDimX by
    `num_ctas` internally, so one Triton program becomes one cluster of `n` CTAs.
    """
    n = int(n)
    if n <= 1 or not caps().clusters:
        return {}
    return {"num_ctas": n}


def cluster_choices(candidates=(1, 2, 4)) -> tuple:
    """Cluster sizes an autotuning grid should sweep, collapsed to `(1,)` without cluster
    support. As with `ws_choices`, both arms must call this so they are pruned equally."""
    if not caps().clusters:
        return (1,)
    return tuple(int(n) for n in candidates if int(n) >= 1)


# ======================================================================================
# CLI: `python3 glm52_h200/kernels/hopper.py` prints the verdict; --probe-json is the
# subprocess entry point used by `_probe_subprocess` (invoked by path, so this file needs
# no package import to work).
# ======================================================================================
def _probe_main() -> int:
    try:
        res = _probe_inprocess()
    except Exception as exc:  # noqa: BLE001
        res = {"probes": {}, "errors": {"fatal": f"{type(exc).__name__}: {exc}"}}
    sys.stdout.write(_PROBE_MARKER + json.dumps(res, default=str) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    if "--probe-json" in sys.argv:
        raise SystemExit(_probe_main())
    print(report())
    print()
    print("tma_stats:", json.dumps(tma_stats(), indent=2))
