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
  copies while consumer warps do the math, instead of every warp alternating. **The H200 is
  the first device in this study on which warp specialization exists at all**, which matters
  beyond one more autotuning axis: it is the precondition the "Towards Free Normalization"
  technique needs. That technique hides a normalization epilogue behind a GEMM mainloop by
  giving it to warps that are not doing the math; without producer/consumer warps there is
  nowhere to hide it, and #3/#4/#5/#11 can only overlap by software pipelining -- which is one
  reason their measured wins were small on the earlier two devices. A null result for those
  four on H200 therefore means something the earlier boxes could not tell us.

* **Thread-block clusters + DSMEM** (`num_ctas`). Up to 8 CTAs (16 non-portably, which does
  work on H100/H200) are co-scheduled on one GPC and can address each other's shared memory.
  A tile gets a larger effective SMEM budget, and an expert-merge reduction can happen in
  DSMEM instead of through L2.

**Whether any of the three helps here is the experiment, not an assumption.** Every one of the
eleven fusions must still run, and be measured, with all three off; the Hopper path is an
extra arm, never a precondition. So every helper below degrades to a no-op -- `{}` launch
kwargs, a `False` constexpr, a `None` descriptor -- instead of raising, and records what it
decided so the result file can state which arm actually ran.

The measured H200 raises the stakes on getting that recording right. Its FLOP/byte balance is
~185 (C500 82, RTX 4060 84): 4.25 TB/s against 788 TF/s of achievable Triton bf16. Relative to
those devices it is far more compute-dense, so a memory-bound vector fusion is worth *more*
here and a fusion that displaces GEMM mainloop work is worth *less*. Both effects run through
the same three levers, and a lever that silently did not engage would look exactly like the
effect being absent.

Detection, in priority order, per capability:

  1. an explicit env override (below), else
  2. an architecture veto: below sm_90 the answer is False regardless of what compiles, else
  3. `preflight_h200.json`, when it demonstrably describes *this* device and Triton, else
  4. a one-time trial compile+launch, run ONLY for the capabilities still undecided, else
  5. False.

Three deliberate choices, each of which cost something real:

*Why the arch veto outranks everything.* `tl.range(warp_specialize=True)` **compiles and runs
on sm_89** (this repo's own sm_89 preflight records exactly that) and is then silently not the
Hopper producer/consumer scheme. Reporting a "warp specialization" measurement from a device
that ignored the flag would be a fabricated result, so the cap is ANDed with sm_90+. Set
`GLM52_H200_WS=1` to measure the pre-Hopper behaviour deliberately.

*Why the preflight outranks the trial compile.* The preflight was run on the real H200 by the
only person who can reach it, and it is the artefact under review; a local probe that
contradicted it would be a second opinion nobody can adjudicate. It also costs nothing --
which matters on an 8-GPU box shared with other tenants, where every avoidable CUDA context is
avoidable interference. So the trial compile is a *gap filler*: it runs only for capabilities
the preflight did not answer, and on the measured H200 that is TMA alone.

*Why the preflight's TMA verdict is the one thing not taken at face value.* Its
`tma_tensor_descriptor` probe passes a **host-side** `TensorDescriptor` object into
`tl.make_tensor_descriptor()`, which is the **device-side** constructor and wants a raw
pointer. Mixing the two APIs is a `CompilationError` on any hardware, so that probe's FAIL on
the H200 is a **false negative** -- it reports API misuse, not missing hardware. Both correct
spellings have since been verified to compile, launch and return correct values on triton
3.6.0. Hence:

  - the old key `tma_tensor_descriptor` is **ignored when False** (its known bug can only
    produce false negatives), but honoured when True (that bug cannot produce a false
    positive), and
  - a preflight carrying the new per-form keys (`tma_host_descriptor` /
    `tma_device_descriptor`, or the `*_tensor_descriptor_host/_device` spellings) is
    authoritative in both directions, and
  - failing all of that, this module runs its own two-form probe.

Trusting the buggy key would have silently disabled TMA everywhere on the one machine the
suite exists to measure, and the result file would still have said "H200".

By default the gap-filling probe runs in a **subprocess** (`GLM52_H200_PROBE=subproc`). A TMA
launch with a malformed descriptor raises an *asynchronous* illegal-memory-access, which is
sticky: it poisons the CUDA context and every later measurement in the process dies. Nobody
can test on the H200, so a crash costs a whole round trip; a one-off subprocess is cheap
insurance. `inproc` and `off` are available when spawning is not.

What the H200 measured, and what this module therefore does:

    tl_dot_bf16                          OK    -> wgmma-class tiles are reachable
    warp_specialize_tl_range             OK    -> the ONLY warp-spec spelling here
    thread_block_cluster_num_ctas        OK    -> num_ctas > 1 launches
    warp_specialize_num_consumer_groups  FAIL  -> KeyError: unrecognised keyword
    tma_tensor_descriptor                FAIL  -> false negative, see above

`num_consumer_groups`/`num_buffers_warp_spec` are launch kwargs from an older *forked* Triton
and do not exist on 3.6.0; passing one is not a soft failure, it raises `KeyError: 'Keyword
argument num_consumer_groups was specified but unrecognised'` and kills the launch. So
`ws_kwargs()` returns `{}` unconditionally and this module never emits either kwarg. Warp
specialization is a **source-level** argument to `tl.range`, so it reaches kernels as a
`tl.constexpr` flag they branch on (`ws_source_flag()`), never as a launch kwarg.

A note on the preflight's timing calibration, which this module reads only to *warn* about:
`harness_floor_us=40.55` with `timer_tick_match_frac=0.03` (the tick detector needs >=0.98)
are not physical numbers, and `mem_free` was 98.8 GB of 150 GB -- another tenant held ~51 GB
of the probed GPU. Anything derived from `launch_us`/`timer_tick_us` is unreliable until
re-probed on an idle GPU. `caps().preflight_timing_suspect` says so out loud rather than
letting a tick-limited cell be flagged, or not flagged, on contaminated evidence. The
*feature* verdicts in the same file are unaffected: what compiles does not depend on who else
is using the device.

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
    device_identity()           -> index/uuid/CUDA_VISIBLE_DEVICES of the measured GPU
    ensure_allocator()          -> bool, registers the global-scratch allocator TMA needs
    descriptor(t, block_shape)  -> TensorDescriptor | None   (host-side form; None => classic)
    device_tma_ready(t, blk)    -> (bool, reason)            (device-side form)
    tma_reject_reason(t, blk)   -> str | None, why either form would decline
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

# ---- preflight compile-probe keys -----------------------------------------------------
# Several spellings are accepted for the two corrected TMA probes because preflight.py is
# maintained separately: a key this module does not recognise is silently no evidence, which
# would put TMA back on the local probe -- correct, but slower and one more CUDA context on a
# shared box. Matching liberally costs nothing.
_PRE_TMA_HOST_KEYS = (
    "tma_host_descriptor",
    "tma_tensor_descriptor_host",
    "tma_descriptor_host",
)
_PRE_TMA_DEVICE_KEYS = (
    "tma_device_descriptor",
    "tma_tensor_descriptor_device",
    "tma_descriptor_device",
)
# The pre-fix key. Its probe mixes the host and device TMA APIs, so a FAIL is uninformative
# (see the module docstring) -- but a PASS still could not have happened by accident.
_PRE_TMA_LEGACY_KEY = "tma_tensor_descriptor"

_PRE_WS_RANGE_KEY = "warp_specialize_tl_range"
_PRE_WS_LAUNCH_KEY = "warp_specialize_num_consumer_groups"
_PRE_CLUSTER_KEY = "thread_block_cluster_num_ctas"
_PRE_DOT_KEY = "tl_dot_bf16"

# Which local probes answer which capability. Used to run ONLY the probes whose capability is
# still undecided after env + preflight; on the measured H200 that is the TMA group alone.
_PROBE_GROUPS = {
    "tma": ("allocator", "tma_host_descriptor", "tma_device_descriptor"),
    "warp_specialize": ("ws_tl_range",),
    "clusters": ("cluster_num_ctas",),
    "wgmma": ("dot_bf16",),
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
    suite; it only means every capability falls through to the local trial compile.
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
    """A preflight probe's verdict, or None when it did not run that probe.

    `ok` alone is not the verdict. preflight.py's TMA probes also record `values_correct`,
    because a descriptor that is subtly wrong still *launches* -- it just moves the wrong
    bytes, and an exception-based probe would call that a pass. A launch that returned the
    wrong tile is a FAIL here, matching preflight's own `_ok()` helper: absent means "not
    checked", False means "checked and wrong".
    """
    entry = probes.get(key)
    if isinstance(entry, dict) and "ok" in entry:
        if entry.get("values_correct") is False:
            return False
        return bool(entry["ok"])
    return None


def _probe_verdict_any(probes: dict, keys) -> "bool | None":
    """First recognised spelling wins; None when the file names none of them."""
    for k in keys:
        v = _probe_verdict(probes, k)
        if v is not None:
            return v
    return None


# ======================================================================================
# capabilities
# ======================================================================================
@dataclass
class HopperCaps:
    """What this process is allowed to emit. Treat as immutable; `caps()` caches one.

    The four booleans are the contract every kernel module reads. Everything else is
    provenance -- which is not decoration: `n_tried`/`n_failed` per arm plus "which Hopper
    path was live, on which physical GPU" is what makes an unfair comparison detectable after
    the fact (LOG-08).
    """

    tma: bool = False
    warp_specialize: bool = False
    clusters: bool = False
    wgmma: bool = False

    # ---- how TMA is reachable. The two forms are not interchangeable, see `tma_form`. ----
    tma_host: bool = False      # host-built TensorDescriptor passed as a kernel argument
    tma_device: bool = False    # device-side tl.make_tensor_descriptor(base, ...)

    # ---- how warp specialization is spelled on this Triton -------------------------------
    # "range" or "none". The launch-kwarg flavour is deliberately NOT a value here: it does
    # not exist on triton 3.6 and this module never emits it. See `ws_kwargs`.
    ws_mode: str = "none"

    # ---- device / stack ------------------------------------------------------------------
    device_name: str = ""
    cc_major: int = 0
    cc_minor: int = 0
    torch_version: str = ""
    triton_version: str = ""

    # ---- which physical GPU this verdict describes ---------------------------------------
    # On an 8-GPU host with other tenants, "H200" does not identify a device. The UUID does,
    # and it is the only field that survives CUDA_VISIBLE_DEVICES remapping: with `--gpu N`
    # the chosen GPU is always index 0 to torch, so the visible index alone is a lie. Both are
    # recorded, plus the mask that produced them, so a number in a result file can be traced
    # to a device (and to whether it was sharing that device with somebody).
    device_index: int = -1              # index within the *visible* set (0 under --gpu N)
    device_uuid: str = ""
    device_count: int = 0               # visible devices, not devices on the host
    cuda_visible_devices: str = ""      # the mask, verbatim
    # Unset and set-to-empty are different states -- all GPUs vs none -- and only the flag
    # distinguishes them, so "which GPU did this run use" has an unambiguous answer.
    cuda_visible_devices_set: bool = False

    # ---- provenance ----------------------------------------------------------------------
    sources: dict = field(default_factory=dict)   # cap -> "env"|"preflight"|"probe"|"arch"|"none"
    probe_mode: str = "none"                      # "subproc"|"inproc"|"off"|"skipped"|"none"
    probe_scope: tuple = ()                       # which capabilities the local probe tested
    probe_raw: dict = field(default_factory=dict)  # per-probe ok/error, verbatim
    preflight_used: bool = False
    preflight_note: str = ""
    preflight_device_uuid: str = ""
    preflight_timing_suspect: bool = False        # launch_us / timer_tick_us are contaminated
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
        already keeps out of the timed allocator path.
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
        d["probe_scope"] = list(self.probe_scope)
        return d


_CAPS: "HopperCaps | None" = None
# Reentrant on purpose: `_detect()` runs while holding it, and a future helper that reached
# for `caps()` from inside detection would otherwise deadlock the whole benchmark -- a
# failure mode that looks exactly like a hung GPU from the outside.
_CAPS_LOCK = threading.RLock()


def caps() -> HopperCaps:
    """The cached capability verdict for this process. Never raises.

    Cost: zero on anything below sm_90 (the arch veto short-circuits before any probe); with
    a matching preflight, one file read plus a probe for whatever the preflight left
    undecided; nothing afterwards.
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


def device_identity() -> dict:
    """Which physical GPU produced this verdict. Put it in every result file.

    With `--gpu N` the harness sets `CUDA_VISIBLE_DEVICES=N`, so the chosen device is index 0
    to torch and the index alone cannot distinguish GPU 0 from GPU 5. The UUID can, and it is
    what `nvidia-smi` reports for the processes holding memory -- which is how a suspicious
    number gets traced back to "that run shared GPU 3 with somebody".
    """
    c = caps()
    return {
        "name": c.device_name,
        "uuid": c.device_uuid,
        "visible_index": c.device_index,
        "visible_count": c.device_count,
        "cuda_visible_devices": c.cuda_visible_devices if c.cuda_visible_devices_set else None,
        "sm_arch": c.sm_arch,
        "preflight_device_uuid": c.preflight_device_uuid,
        "same_device_as_preflight": bool(
            c.device_uuid and c.preflight_device_uuid
            and c.device_uuid == c.preflight_device_uuid
        ),
    }


def banner() -> str:
    c = caps()
    gpu = c.device_uuid[:8] if c.device_uuid else "?"
    return (
        f"[hopper] {c.device_name or '?'} ({c.sm_arch}) gpu={gpu} | tma={c.tma}"
        f"{'(' + c.tma_form() + ')' if c.tma else ''} "
        f"ws={c.warp_specialize}{'(' + c.ws_mode + ')' if c.warp_specialize else ''} "
        f"clusters={c.clusters} wgmma={c.wgmma} | "
        f"src {'/'.join(sorted(set(c.sources.values())))} | probe {c.probe_mode}"
        f"{'[' + ','.join(c.probe_scope) + ']' if c.probe_scope else ''}"
    )


def report() -> str:
    """Multi-line human summary: the banner, then every note and error. Print this once at
    bench startup -- a silently-degraded Hopper path is the failure mode this whole module
    exists to make visible."""
    c = caps()
    lines = [banner()]
    ident = device_identity()
    mask = ident["cuda_visible_devices"]
    lines.append(
        f"  device           uuid={ident['uuid'] or '?'} visible_index={ident['visible_index']} "
        f"of {ident['visible_count']} "
        f"CUDA_VISIBLE_DEVICES={'<unset>' if mask is None else repr(mask)}"
    )
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
    round trip). A disagreement is expected and *informative* for TMA -- `BenchEnv` reads the
    preflight's buggy `tma_tensor_descriptor` key, which this module ignores when False -- so
    it is reported, not resolved.
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
    _mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    c.cuda_visible_devices_set = _mask is not None
    c.cuda_visible_devices = _mask if _mask is not None else ""
    try:
        if not torch.cuda.is_available():
            c.notes.append("torch.cuda.is_available() is False; no Hopper features")
            return False
        idx = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(idx)
        c.device_name = p.name
        c.cc_major, c.cc_minor = int(p.major), int(p.minor)
        c.device_index = int(idx)
        c.device_count = int(torch.cuda.device_count())
        # `uuid` is a `uuid.UUID` on torch >= 2.1 and absent before that; str() either way.
        u = getattr(p, "uuid", None)
        c.device_uuid = str(u) if u is not None else ""
    except Exception as exc:  # noqa: BLE001
        c.errors["device"] = f"{type(exc).__name__}: {exc}"
        return False
    if c.device_count > 1 and not c.cuda_visible_devices_set:
        c.notes.append(
            f"{c.device_count} GPUs visible and CUDA_VISIBLE_DEVICES is unset: this process "
            f"took cuda:{c.device_index} by default and may be sharing it. Use --gpu N (or "
            f"--gpu auto) so the run pins one idle device -- a co-tenant is what produced the "
            f"preflight's impossible 40 us harness floor."
        )
    return True


def _preflight_view(c: HopperCaps) -> dict:
    """Preflight's verdicts, but only if the file describes THIS device and THIS Triton.

    Lesson 4 generalised: a payload from another GPU is worse than no payload, because it is
    plausible. A stale-device preflight would have TMA=False from the 4060 and would sit,
    unremarked, inside a file labelled H200.

    Matching is on model + compute capability + Triton version, NOT on UUID. On this host all
    eight GPUs are H200s, and `--gpu 5` must not invalidate a preflight taken on GPU 0: what
    compiles is a property of the silicon model and the compiler, not of the individual board.
    A UUID difference is recorded (it changes which *calibration* numbers apply) but never
    vetoes a feature verdict.
    """
    pre = preflight_data()
    if not pre:
        c.preflight_note = preflight_status() or "no preflight"
        return {}
    d = pre.get("device") or {}
    s = pre.get("stack") or {}
    c.preflight_device_uuid = str(d.get("uuid", "") or "")
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
    c.preflight_note = (
        f"{PREFLIGHT_PATH.name} matches this device and stack ({pre.get('timestamp')})"
    )
    if c.device_uuid and c.preflight_device_uuid and c.device_uuid != c.preflight_device_uuid:
        c.notes.append(
            f"preflight was taken on GPU {c.preflight_device_uuid[:8]} and this run is on "
            f"{c.device_uuid[:8]}; same model, so the FEATURE verdicts still hold, but the "
            f"file's bandwidth/GEMM/tick calibration describes a different board"
        )

    # The timing calibration in this file was taken while another tenant held ~1/3 of the
    # device. Say so once, here, where the file is already parsed -- a tick-limited flag
    # computed from a tick that matched 3% of samples is worse than no flag at all.
    cal = pre.get("calibration") or {}
    frac = cal.get("timer_tick_match_frac")
    floor = cal.get("harness_floor_us")
    free, total = d.get("mem_free_bytes"), d.get("mem_total_bytes")
    reasons = []
    try:
        if frac is not None and float(frac) < 0.9:
            reasons.append(f"timer_tick_match_frac={float(frac):.2f} (needs >=0.98)")
        if floor is not None and float(floor) > 20.0:
            reasons.append(f"harness_floor_us={float(floor):.1f}")
        if free and total and float(free) < 0.9 * float(total):
            busy = (float(total) - float(free)) / 2**30
            reasons.append(f"{busy:.0f} GB already allocated by another process at probe time")
    except (TypeError, ValueError):
        pass
    if reasons:
        c.preflight_timing_suspect = True
        c.notes.append(
            "preflight TIMING calibration is CONTAMINATED (" + "; ".join(reasons) + "). "
            "launch_us and timer_tick_us from that file are unreliable and any "
            "tick-limited/floor-limited flag derived from them should be read as unknown, "
            "not as false. Re-run preflight.py pinned to an idle GPU. The feature probes in "
            "the same file are unaffected -- what compiles does not depend on co-tenants."
        )

    probes = _preflight_probes(pre)
    # preflight also publishes a rolled-up `tma_forms` so a consumer need not know that the
    # host-side and device-side descriptors are two separate APIs. Prefer the raw probes (they
    # carry the error text) and fall back to the summary, so either shape of the file works.
    forms = ((pre.get("triton_features") or {}).get("tma_forms") or {})
    tma_host = _probe_verdict_any(probes, _PRE_TMA_HOST_KEYS)
    tma_device = _probe_verdict_any(probes, _PRE_TMA_DEVICE_KEYS)
    if tma_host is None and "host_descriptor" in forms:
        tma_host = bool(forms["host_descriptor"])
    if tma_device is None and "device_descriptor" in forms:
        tma_device = bool(forms["device_descriptor"])
    return {
        "tma_host": tma_host,
        "tma_device": tma_device,
        "tma_legacy": _probe_verdict(probes, _PRE_TMA_LEGACY_KEY),
        "ws_range": _probe_verdict(probes, _PRE_WS_RANGE_KEY),
        "ws_launch": _probe_verdict(probes, _PRE_WS_LAUNCH_KEY),
        "clusters": _probe_verdict(probes, _PRE_CLUSTER_KEY),
        "dot_bf16": _probe_verdict(probes, _PRE_DOT_KEY),
    }


def _preflight_tma(pv: dict, notes: list) -> "tuple[bool | None, bool, bool]":
    """(verdict, host, device) from the preflight's TMA probes, or (None, ...) for no evidence.

    The corrected per-form keys are authoritative in both directions. The pre-fix
    `tma_tensor_descriptor` key is asymmetric evidence and is treated that way: its probe
    hands a host `TensorDescriptor` to the device-side `tl.make_tensor_descriptor`, which
    cannot compile on ANY hardware, so a FAIL says nothing about the GPU -- while a PASS
    could not have happened without working TMA and is therefore believed.
    """
    host, dev = pv.get("tma_host"), pv.get("tma_device")
    if host is not None or dev is not None:
        return bool(host or dev), bool(host), bool(dev)
    legacy = pv.get("tma_legacy")
    if legacy is True:
        notes.append(
            "preflight's legacy tma_tensor_descriptor probe PASSED; TMA taken as available "
            "(that probe can produce false negatives, never false positives). Which spelling "
            "works is left to the local probe."
        )
        return True, False, False
    if legacy is False:
        notes.append(
            "preflight's tma_tensor_descriptor=FAIL is IGNORED: that probe passes a host-side "
            "TensorDescriptor into tl.make_tensor_descriptor(), the device-side constructor, "
            "which is a CompilationError on any hardware. It is a false negative, not a "
            "hardware verdict -- TMA is decided by this module's own two-form probe."
        )
    return None, False, False


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

    try:
        import triton

        c.triton_version = getattr(triton, "__version__", "")
    except Exception as exc:  # noqa: BLE001
        c.errors["triton_import"] = f"{type(exc).__name__}: {exc}"
        c.notes.append("triton unimportable; no Hopper features (and no kernels either)")
        for k in CAP_NAMES:
            c.sources[k] = "none"
        _apply_overrides(c, overrides)
        return c

    pv = _preflight_view(c)

    # --- the architecture veto ---------------------------------------------------------
    # Below sm_90 none of the four is real, whatever compiles. Short-circuiting here also
    # means `caps()` costs nothing on the sm_89 box this file was developed on.
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

    # --- stage 1: env pins ------------------------------------------------------------
    pending = {k for k in CAP_NAMES if overrides.get(k) is None}

    # --- stage 2: the preflight -------------------------------------------------------
    # It outranks the local probe: it was measured on the real device by the only person who
    # can reach it, and skipping a probe means not creating a second CUDA context on a host
    # that has other tenants.
    pre_tma, pre_tma_host, pre_tma_device = _preflight_tma(pv, notes)
    pre_vals = {
        "tma": pre_tma,
        "warp_specialize": pv.get("ws_range"),
        "clusters": pv.get("clusters"),
        "wgmma": pv.get("dot_bf16"),
    }
    for k in sorted(pending):
        v = pre_vals.get(k)
        if v is None:
            continue
        setattr(c, k, bool(v))
        c.sources[k] = "preflight"
        pending.discard(k)
    if c.tma and c.sources.get("tma") == "preflight":
        c.tma_host, c.tma_device = pre_tma_host, pre_tma_device
        if not (c.tma_host or c.tma_device):
            # Legacy-PASS case: TMA works, spelling unknown. Ask the local probe which.
            pending.add("tma")
    if pv.get("ws_launch") is False:
        notes.append(
            "preflight: num_consumer_groups is unrecognised on this Triton, as expected on "
            "3.6.0. tl.range(warp_specialize=True) is the only warp-spec spelling here and "
            "ws_kwargs() emits nothing."
        )

    # --- stage 3: trial compile+launch, for whatever is still undecided ----------------
    mode = (os.environ.get("GLM52_H200_PROBE") or "subproc").strip().lower()
    if mode not in ("subproc", "inproc", "off"):
        notes.append(f"GLM52_H200_PROBE={mode!r} unrecognised; using 'subproc'")
        mode = "subproc"
    if not pending:
        mode = "skipped"
        notes.append(
            "every capability was answered by env or preflight; no trial compile ran and no "
            "second CUDA context was created on this device"
        )

    probe: dict = {}
    scope = tuple(sorted(pending))
    if mode == "subproc":
        probe = _probe_subprocess(c, scope)
    elif mode == "inproc":
        probe = _probe_inprocess(scope)
    c.probe_mode = mode
    c.probe_scope = scope if mode in ("subproc", "inproc") else ()
    c.probe_raw = probe.get("probes", {}) if probe else {}
    for k, v in (probe.get("errors") or {}).items():
        c.errors[f"probe.{k}"] = v

    def _p(key: str) -> "bool | None":
        e = c.probe_raw.get(key)
        return bool(e["ok"]) if isinstance(e, dict) and "ok" in e else None

    if "tma" in pending:
        host, dev = _p("tma_host_descriptor"), _p("tma_device_descriptor")
        if host is None and dev is None:
            # No probe result (off, refused to spawn, or it died). Keep whatever the legacy
            # preflight key allowed, else False -- never guess a capability into existence.
            if c.sources.get("tma") != "preflight":
                c.tma, c.tma_host, c.tma_device = False, False, False
                c.sources["tma"] = "none"
                notes.append(
                    "TMA undecided: the preflight gives no usable verdict and no trial "
                    "compile ran. Falling back to the classic path everywhere. Re-run with "
                    "GLM52_H200_PROBE=inproc, or GLM52_H200_TMA=1, if TMA is expected here."
                )
            else:
                c.tma_host = True   # legacy PASS, spelling unknown: the safer of the two
                notes.append(
                    "TMA is on from the legacy preflight probe but no local probe resolved "
                    "the spelling; assuming the host-side descriptor form"
                )
        else:
            c.tma_host, c.tma_device = bool(host), bool(dev)
            c.tma = bool(host or dev)
            c.sources["tma"] = "probe"
            if c.tma and not c.tma_device:
                notes.append(
                    "only the host-side TMA descriptor works here; every launch pays a host "
                    "cuTensorMapEncodeTiled inside the timed window, which at decode sizes "
                    "is a real cost charged to the TMA arm alone"
                )
    if "warp_specialize" in pending:
        v = _p("ws_tl_range")
        c.warp_specialize = bool(v)
        c.sources["warp_specialize"] = "probe" if v is not None else "none"
    if "clusters" in pending:
        v = _p("cluster_num_ctas")
        c.clusters = bool(v)
        c.sources["clusters"] = "probe" if v is not None else "none"
    if "wgmma" in pending:
        # wgmma is not a switch. Triton selects it for sm_90 when the tile shape and dtype
        # allow, so the cap means "wgmma-class tensor cores are reachable" -- it exists to let
        # a grid builder prefer wgmma-eligible tiles (BM >= 64, BN a multiple of 8, bf16
        # operands), not to toggle anything at launch.
        v = _p("dot_bf16")
        c.wgmma = bool(v)
        c.sources["wgmma"] = "probe" if v is not None else "none"

    for k in CAP_NAMES:
        c.sources.setdefault(k, "none")

    # `tl.range(warp_specialize=True)` is the only spelling this stack has; the launch-kwarg
    # flavour raises KeyError on 3.6.0 and is never emitted (see `ws_kwargs`).
    c.ws_mode = "range" if c.warp_specialize else "none"

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

    # Install the global-scratch allocator the moment TMA is decided to be live, in the
    # process that will launch the kernels. Every TMA helper below re-ensures it, but doing it
    # here means no ordering mistake downstream can reach a descriptor launch without it --
    # which is the classic silent TMA failure this module refuses to leave reachable.
    if c.tma and not ensure_allocator():
        c.tma = False
        c.tma_host = c.tma_device = False
        c.sources["tma"] = "none"
        notes.append(
            f"TMA disabled: the global scratch allocator could not be registered "
            f"({_ALLOC['error'] or 'unknown'}). A descriptor kernel launched without it fails "
            f"inside the driver with an error that never names TMA."
        )
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
        if was != v:
            c.notes.append(
                f"{_ENV_KEYS[k]} forced {k}={v} (detection said {was}); "
                f"this result is NOT a clean feature measurement"
            )
    # Keep the reported state self-consistent: `ws_mode` describes how warp specialization
    # would be spelled, and must not survive the capability being off -- `ws_source_flag`
    # already checks both, but a result file that says ws=False/mode=range invites the
    # reader to conclude the wrong thing about which arm ran.
    c.ws_mode = "range" if c.warp_specialize else "none"
    if not c.tma:
        c.tma_host = c.tma_device = False


# ======================================================================================
# trial compile + launch
# ======================================================================================
_PROBE_MARKER = "@@GLM52_HOPPER_PROBE@@"


def _probe_inprocess(scope=None) -> dict:
    """Compile AND launch one tiny kernel per undecided capability. Attribute existence is not
    evidence -- several Triton releases export symbols that fail at compile time, which is why
    preflight.py probes this way too.

    `scope` is the set of capability names still undecided; anything outside it is skipped, so
    a run with a matching preflight touches the device only for TMA. Ordered safest-first, and
    `synchronize()`d after each: a TMA fault is asynchronous, so without the sync a later,
    unrelated measurement would be the thing that dies.

    The TMA probes check the **values**, not just that the launch returned. A descriptor that
    is subtly wrong still launches; it just moves the wrong bytes, and "TMA is available" would
    then be recorded from a kernel that silently corrupted its tile.
    """
    want = set(scope) if scope is not None else set(CAP_NAMES)
    out: dict = {"probes": {}, "errors": {}, "scope": sorted(want)}
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
        # The one supported warp-spec spelling: a SOURCE-level argument to tl.range.
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.range(0, N, BLOCK, warp_specialize=True):
            i = k + tl.arange(0, BLOCK)
            acc += tl.load(X + i, mask=i < N, other=0.0)
        tl.store(Y + tl.arange(0, BLOCK), acc)

    # TMA form 1: a host-built TensorDescriptor arrives as a kernel argument and is loaded
    # from directly. NOT tl.make_tensor_descriptor -- that is the other constructor, and
    # feeding it a descriptor object is the mistake the preflight's probe makes.
    @triton.jit
    def _p_tma_host(Desc, Out, BM: tl.constexpr, BN: tl.constexpr):
        t = Desc.load([0, 0])
        tl.store(Out + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], t)

    # TMA form 2: the descriptor is built on the device from a raw pointer plus runtime
    # shape/strides. This is what tl.make_tensor_descriptor is for, and it needs global
    # scratch -- hence ensure_allocator() below.
    @triton.jit
    def _p_tma_dev(X, Out, M, N, BM: tl.constexpr, BN: tl.constexpr):
        d = tl.make_tensor_descriptor(X, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
        t = d.load([0, 0])
        tl.store(Out + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], t)

    stop = {"hit": False}

    def run(name: str, fn, verify=None) -> None:
        if stop["hit"]:
            out["probes"][name] = {"ok": False, "error": "skipped: context suspect"}
            return
        try:
            fn()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            out["probes"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
            # If the *sync* is what failed, the context may be poisoned; stop touching it.
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                stop["hit"] = True
                out["errors"]["context"] = "CUDA context faulted during probing"
            return
        if verify is not None:
            try:
                good = bool(verify())
            except Exception as exc:  # noqa: BLE001
                out["probes"][name] = {
                    "ok": False, "error": f"verification raised {type(exc).__name__}: {exc}"[:200]
                }
                return
            if not good:
                out["probes"][name] = {
                    "ok": False,
                    "error": "launched but produced wrong values (descriptor moved the wrong "
                             "bytes); treated as unavailable",
                }
                return
        out["probes"][name] = {"ok": True}

    # Everything below touches the device. It is guarded as a whole as well as per probe:
    # an allocation failure between probes must still return the results already gathered,
    # because "TMA worked, then we ran out of memory" and "TMA never worked" are different
    # facts and only one of them means the Hopper path should be off.
    try:
        N = 4096
        x = torch.randn(N, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        # Always run: if this fails, nothing below means anything, and the reason it failed
        # (OOM on a device someone else filled) is the single most useful line in the file.
        run("baseline", lambda: _p_copy[(N // 256,)](x, y, N, BLOCK=256))

        if "wgmma" in want:
            a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
            cc = torch.empty(64, 64, device="cuda", dtype=torch.float32)
            run("dot_bf16", lambda: _p_dot[(1,)](a, b, cc, BM=64, BN=64, BK=64))

        if "warp_specialize" in want:
            # num_warps=4 is Triton's own precondition for the warp-specialize transform on
            # several releases; a config that cannot be specialized simply fails to compile,
            # and the autotuner records it as a failed config rather than guessing a rule here.
            run("ws_tl_range", lambda: _p_ws[(1,)](x, y, N, BLOCK=256, num_warps=4))
            # The num_consumer_groups spelling is NOT probed: it raises KeyError on this
            # Triton, the kernels are not written for it, and a probe that can only ever
            # return False is a compile we do not need to pay for.

        if "tma" in want:
            # Both spellings, with the allocator registered first -- forgetting it is the
            # classic silent failure (the launcher asks for global scratch and gets None).
            src = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
            dst = torch.empty(64, 64, device="cuda", dtype=torch.bfloat16)
            out["probes"]["allocator"] = {"ok": bool(ensure_allocator())}

            def _tma_host():
                from triton.tools.tensor_descriptor import TensorDescriptor

                desc = TensorDescriptor.from_tensor(src, [64, 64])
                _p_tma_host[(1,)](desc, dst, BM=64, BN=64)

            dst.zero_()
            run("tma_host_descriptor", _tma_host,
                verify=lambda: torch.equal(dst, src[:64, :64]))
            dst.zero_()
            run("tma_device_descriptor",
                lambda: _p_tma_dev[(1,)](src, dst, 128, 128, BM=64, BN=64),
                verify=lambda: torch.equal(dst, src[:64, :64]))

        if "clusters" in want:
            # Triton multiplies gridDimX by num_ctas, so the grid needs no divisibility
            # fix-up; a too-large cluster fails at launch and is recorded like any other
            # config.
            run("cluster_num_ctas", lambda: _p_copy[(2,)](x, y, N, BLOCK=256, num_ctas=2))
    except Exception as exc:  # noqa: BLE001
        out["errors"]["probe_body"] = f"{type(exc).__name__}: {exc}"[:400]
    return out


def _probe_subprocess(c: HopperCaps, scope=()) -> dict:
    """Run `_probe_inprocess` in a fresh interpreter and read back its JSON.

    Why: an ill-formed TMA descriptor raises an asynchronous illegal-memory-access, which is
    sticky -- the CUDA context is dead and every later measurement in the process fails. On a
    machine nobody can log into, that is a whole wasted round trip. A second context costs a
    few hundred MB of the H200's 143 GB and a few seconds, once.

    The child inherits this process's environment, and therefore `CUDA_VISIBLE_DEVICES`: it
    probes the same GPU the harness selected with `--gpu`, not GPU 0. That is not incidental
    -- probing a device other than the one being measured is exactly the class of mistake this
    module exists to prevent.

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
    if scope:
        cmd.append(",".join(sorted(scope)))
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as exc:
        c.errors["probe_subprocess"] = f"{type(exc).__name__}: {exc}"[:300]
        c.notes.append(
            "the capability probe timed out or died in its own subprocess -- which is what "
            "the subprocess is for. Treating the undecided features as unavailable; the "
            "classic arms still measure correctly. GLM52_H200_PROBE=inproc to see the crash."
        )
        return {}
    except OSError as exc:
        c.errors["probe_spawn"] = f"{type(exc).__name__}: {exc}"[:300]
        c.notes.append("could not spawn the probe subprocess; falling back to in-process")
        return _probe_inprocess(scope)
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
        return _probe_inprocess(scope)
    c.notes.append(
        f"probe subprocess exited rc={p.returncode}; NOT retried in-process (a crash there "
        f"would take the benchmark with it). Undecided Hopper features stay off."
    )
    return {}


# ======================================================================================
# TMA: allocator + descriptors
#
# Both spellings need one thing before anything else: a registered global-scratch allocator.
# Forgetting it is the classic silent TMA failure -- the launcher asks Triton for scratch,
# gets None, and the launch dies inside the driver with a message that never says "TMA". It
# is unreachable through this API: `_detect()` installs the allocator as soon as TMA is
# decided to be live, and `tma_reject_reason()` (which both `descriptor()` and the device-side
# gate go through) re-checks it and declines if it is missing.
# ======================================================================================
_ALLOC = {
    "installed": False,
    "error": "",
    "attempts": 0,
    "buffers": {},      # stream id -> int8 scratch tensor
    "peak_bytes": 0,
    "calls": 0,
    "reuse": os.environ.get("GLM52_H200_TMA_SCRATCH_REUSE", "1").strip().lower() not in _FALSE,
}
# Guards installation only. NOT taken by `_scratch`, which runs on the launch path. Lock
# order is _CAPS_LOCK -> _ALLOC_LOCK and never the reverse, so nothing under this lock may
# call `caps()`.
_ALLOC_LOCK = threading.Lock()


def _scratch(size: int, alignment: int, stream):
    """Triton's global-scratch allocator.

    Called on **every launch** of a kernel whose `global_scratch_size > 0` -- which is every
    device-side-descriptor kernel -- with `grid * num_ctas * per_program_bytes`. Triton's own
    example allocates a fresh tensor each time; here that allocation would sit inside the
    timed region and be charged to the TMA arm only, and at decode sizes the whole kernel is
    9-17 event ticks. So the buffer is cached per stream and grown geometrically, and reuse
    is safe because launches on one stream are serialized (two threads launching on the *same*
    stream would already be a correctness bug for reasons that have nothing to do with TMA).
    `GLM52_H200_TMA_SCRATCH_REUSE=0` restores the allocate-every-time behaviour for anyone who
    wants to check that this caching is not itself the effect being measured.
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
    """Register the global scratch allocator that TMA descriptors require. Idempotent,
    thread-safe, and `triton.set_allocator` is called **exactly once per process**.

    Both TMA spellings need it: the device-side form allocates its descriptor from global
    scratch, and the host-side form's launcher path wants scratch too. Cheap to call -- one
    bool read after the first success -- so every TMA helper calls it and no caller has to
    remember. Double-checked locking rather than a bare flag because the autotuner builds
    configs from more than one thread on some stacks, and two concurrent `set_allocator`
    calls racing to install different closures is a bug that would only ever appear on the
    machine nobody can debug on.

    Never calls `caps()`: `_detect()` calls *this* while holding the caps lock, and the
    reverse edge would close the cycle.
    """
    if _ALLOC["installed"]:
        return True
    with _ALLOC_LOCK:
        if _ALLOC["installed"]:
            return True
        _ALLOC["attempts"] += 1
        try:
            import triton

            if not hasattr(triton, "set_allocator"):
                _ALLOC["error"] = "triton.set_allocator missing (Triton predates TMA support)"
                return False
            triton.set_allocator(_scratch)
            _ALLOC["installed"] = True
            _ALLOC["error"] = ""
            return True
        except Exception as exc:  # noqa: BLE001 -- caller falls back to the classic path
            _ALLOC["error"] = f"{type(exc).__name__}: {exc}"[:200]
            return False


def allocator_status() -> dict:
    """Whether the TMA scratch allocator is installed, and why not if it is not."""
    return {
        "installed": bool(_ALLOC["installed"]),
        "attempts": int(_ALLOC["attempts"]),
        "error": _ALLOC["error"],
        "reuse": bool(_ALLOC["reuse"]),
    }


_TMA_STATS = {"built": 0, "device_ok": 0, "declined": 0, "reasons": {}}


def _decline(why: str):
    _TMA_STATS["declined"] += 1
    _TMA_STATS["reasons"][why] = _TMA_STATS["reasons"].get(why, 0) + 1
    return None


def tma_reject_reason(tensor, block_shape) -> "str | None":
    """Why a TMA descriptor over `tensor` would be declined, or None if it would work.

    Shared by both spellings on purpose -- the legality rules are identical, and a gate that
    differed between the host-side and device-side paths would make the two arms measure
    different sets of tensors.

    Every rule here is a hard TMA/Triton requirement (16-byte aligned base, 16-byte aligned
    leading strides, contiguous innermost dimension, power-of-two block dims, innermost block
    row a multiple of 16 bytes). They are checked explicitly rather than left to the
    `assert`s inside `TensorDescriptor.__post_init__` because those vanish under `python -O`,
    and a silently-wrong descriptor is an async fault, not an exception.

    **Side effect, deliberately:** this installs the scratch allocator. It is the one call
    every TMA path makes -- `descriptor()` for the host form, `device_tma_ready()` for the
    device form -- so putting the install here is what makes "TMA without an allocator"
    unrepresentable rather than merely documented.
    """
    c = caps()
    if not c.tma:
        return f"tma capability off (source {c.sources.get('tma')})"
    if not ensure_allocator():
        return f"no global scratch allocator ({_ALLOC['error'] or 'unknown'})"
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
    """**TMA form 1** -- a host-side `TensorDescriptor` for `tensor`, to be passed into the
    kernel as an argument and loaded with `desc.load([i, j])`. Returns **None** when TMA
    cannot describe this tensor.

    None is a normal answer, not an error: the caller takes its classic pointer-arithmetic
    path. Returning None rather than raising is what lets a single kernel source serve both
    arms on both architectures, which is the property that keeps the fused/unfused comparison
    honest -- the two arms must differ in mapping, not in which machine they can run on.

        desc = hopper.descriptor(w, [BLOCK_K, BLOCK_N])     # host side
        kern[grid](desc, ...)                               # kernel does desc.load([k, n])

    Note the cost model: a host-side descriptor argument makes Triton's launcher call
    `fill_tma_descriptor` (a host `cuTensorMapEncodeTiled`) on every launch. That is fine at
    prefill and material at decode; see `HopperCaps.tma_form`, which prefers form 2 when both
    are available.
    """
    why = tma_reject_reason(tensor, block_shape)
    if why is not None:
        return _decline(why)
    try:
        from triton.tools.tensor_descriptor import TensorDescriptor

        try:
            desc = TensorDescriptor.from_tensor(tensor, list(block_shape), padding)
        except TypeError:
            # Older signature, without `padding`.
            desc = TensorDescriptor.from_tensor(tensor, list(block_shape))
    except Exception as exc:  # noqa: BLE001
        return _decline(f"{type(exc).__name__}: {exc}"[:120])
    _TMA_STATS["built"] += 1
    return desc


def device_tma_ready(tensor, block_shape) -> tuple:
    """**TMA form 2** -- clearance for the device-side spelling, `(ok, reason)`.

    There is no host object to build here: the kernel constructs the descriptor itself from a
    raw pointer, so all this call has to do is (a) confirm the same legality rules the host
    form obeys and (b) guarantee the scratch allocator is installed, because the descriptor is
    built *in global scratch* and a missing allocator kills the launch with an error that
    never mentions TMA. Gate the config on this and pass the tensor as a plain pointer:

        ok, why = hopper.device_tma_ready(w, [BLOCK_K, BLOCK_N])
        kern[grid](w, ..., K, N, BLOCK_K, BLOCK_N)          # kernel does
        #   d = tl.make_tensor_descriptor(w, shape=[K, N], strides=[N, 1],
        #                                 block_shape=[BLOCK_K, BLOCK_N])
        #   t = d.load([k, n])

    Counted separately from `descriptor()` in `tma_stats()`: "the device form was cleared N
    times" and "N host descriptors were built" are different facts, and a result file that
    conflated them could not say which spelling produced the number.
    """
    why = tma_reject_reason(tensor, block_shape)
    if why is not None:
        _decline(why)
        return False, why
    _TMA_STATS["device_ok"] += 1
    return True, None


def tma_stats() -> dict:
    """Descriptors built vs declined, and why. Record this next to `n_tried`/`n_failed`: a
    TMA arm that quietly declined every descriptor and ran the classic path all along is
    otherwise indistinguishable from a TMA arm that did nothing useful."""
    return {
        "built": _TMA_STATS["built"],
        "device_ok": _TMA_STATS["device_ok"],
        "declined": _TMA_STATS["declined"],
        "reasons": dict(_TMA_STATS["reasons"]),
        "scratch_calls": _ALLOC["calls"],
        "scratch_peak_bytes": _ALLOC["peak_bytes"],
        "scratch_reuse": bool(_ALLOC["reuse"]),
        "allocator_installed": bool(_ALLOC["installed"]),
        "allocator_error": _ALLOC["error"],
    }


# ======================================================================================
# warp specialization
#
# The H200 supports exactly one spelling: `tl.range(..., warp_specialize=True)`, verified by
# the preflight (`warp_specialize_tl_range: ok`). The launch-kwarg flavour that older forked
# Tritons carried is not merely absent -- passing it raises
#   KeyError: 'Keyword argument num_consumer_groups was specified but unrecognised'
# and takes the launch with it. So the enable lives in the SOURCE as a constexpr, and nothing
# in this module ever puts it in a launch dict.
# ======================================================================================
def ws_mode() -> str:
    """"range" (source-level `tl.range(warp_specialize=...)`) or "none". There is no third
    value on this stack."""
    return caps().ws_mode


def ws_source_flag(enable: bool = True) -> bool:
    """Value for the kernel's `WS: tl.constexpr`, i.e. what to pass to
    `tl.range(..., warp_specialize=WS)`.

    This -- not a launch kwarg -- is how warp specialization is turned on here, because
    `warp_specialize` is an argument to `tl.range` and therefore has to reach the *compiler*.
    A kernel wanting it writes

        kern[grid](..., WS=hopper.ws_source_flag(True), **hopper.ws_kwargs(True))

    and gets the classic mainloop, unchanged, on any device or Triton lacking the feature.
    `ws_kwargs` stays in that call so the two mechanisms have one call site between them and
    a future stack can reintroduce a launch-level form without touching eleven kernels.
    """
    return bool(enable and caps().warp_specialize and caps().ws_mode == "range")


def ws_kwargs(
    enable: bool = True,
    num_consumer_groups: int = 1,
    num_buffers_warp_spec: int = 2,
) -> dict:
    """Always `{}` on this stack, and that is the correct answer, not a failure.

    Warp specialization here is enabled at source level (`ws_source_flag`), so there is
    nothing to add to a launch. The two kwargs this function nominally carries --
    `num_consumer_groups` and `num_buffers_warp_spec` -- belong to an older *forked* Triton
    and are unrecognised by 3.6.0, where an unrecognised kwarg is not ignored but raises
    `KeyError` and kills the launch. The preflight confirms exactly that on the H200
    (`warp_specialize_num_consumer_groups: KeyError ... unrecognised`), so this function
    never emits them and the arguments are retained only to keep every existing call site
    valid.
    """
    return {}


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

    `num_ctas > 1` is confirmed working on this H200 (preflight
    `thread_block_cluster_num_ctas: ok`), so this is the one Hopper lever that is a plain
    launch kwarg.

    No clamping on `n`. Hopper takes up to 8 portably and 16 non-portably (Triton's CUDA
    launcher special-cases 16 for H100/H200), but the exact ceiling interacts with SMEM per
    CTA and register pressure -- and this device's opt-in SMEM ceiling is 232448 B, which the
    preflight's own tile probe already runs into at BM128/BN256/BK128 -- so an over-large
    cluster is left to fail at compile/launch and be recorded as a failed config, the same
    treatment every other tile parameter gets. Inventing a limit here would prune the grid
    from a literal, which is the one thing this suite has agreed never to do.

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
# no package import to work). An optional comma-separated capability list restricts the
# probe to what the caller could not resolve from the preflight.
# ======================================================================================
def _probe_main(argv) -> int:
    scope = None
    for i, a in enumerate(argv):
        if a == "--probe-json" and i + 1 < len(argv):
            nxt = argv[i + 1]
            if not nxt.startswith("-"):
                scope = {s.strip() for s in nxt.split(",") if s.strip()}
    try:
        res = _probe_inprocess(scope)
    except Exception as exc:  # noqa: BLE001
        res = {"probes": {}, "errors": {"fatal": f"{type(exc).__name__}: {exc}"}}
    sys.stdout.write(_PROBE_MARKER + json.dumps(res, default=str) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    if "--probe-json" in sys.argv:
        raise SystemExit(_probe_main(sys.argv))
    print(report())
    print()
    print("device:", json.dumps(device_identity(), indent=2))
    print("tma_stats:", json.dumps(tma_stats(), indent=2))
