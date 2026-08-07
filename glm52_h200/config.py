"""GLM-5.2 (zai-org/GLM-5.2, `glm_moe_dsa`) architecture constants, benchmark shapes, and the
H200 device probe.

Source of truth for the model: https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json
Fetched 2026-07-27. Every number in the verbatim block below is copied from that config;
nothing is inferred except the DERIVED block, which is annotated with its arithmetic.

Ported from `glm52/config.py` (the audited C500 / RTX-4060 suite). Three things change here:

  * **The regime ladder.** H200 has 143 GB of HBM, so the 18 GB of routed-expert weights fit
    with room to spare: every one of the eleven fusions and the whole-layer combination
    benchmark are in scope, and decode extends to bs512 / bs1024.
  * **Hopper feature flags.** `tma_supported`, `clusters_supported`, `warp_spec_supported` are
    resolved at RUNTIME -- from `preflight_h200.json` when that file demonstrably describes
    THIS machine, and from a conservative sm_90+ capability check otherwise. Never at
    authoring time: this file is written on an sm_89 box that has neither TMA nor clusters,
    so any code path that assumed either would be untested and wrong.
  * **A device fingerprint**, so a checkpoint written by another GPU can be refused rather
    than republished as a fresh measurement.

The first real H200 preflight (2026-08-03, `preflight_h200.json`) forced three corrections
here, all of them cases where the sm_89 development box had taught us something that is not
true on Hopper:

  * **The TMA verdict was a false negative.** The old `tma_tensor_descriptor` probe fed a
    HOST-side `triton.tools.tensor_descriptor.TensorDescriptor` into `tl.make_tensor_
    descriptor()`, which is the DEVICE-side constructor taking a raw pointer. Mixing the two
    APIs is a CompilationError on any hardware, so that probe measured the probe, not the
    GPU. Both correct spellings compile and run on triton 3.6.0. `tma_supported` therefore
    reads `tma_host_descriptor` / `tma_device_descriptor` and falls back to the sm_90+
    capability check -- it must NEVER be gated on the old key again (see `TMA_LEGACY_PROBE_KEY`).
  * **`warp_specialize_num_consumer_groups` is dead on this stack.** triton 3.6.0 answers
    "Keyword argument num_consumer_groups was specified but unrecognised". Only
    `tl.range(..., warp_specialize=True)` may set `warp_spec_supported`, and no kernel may
    emit the launch-kwarg spelling.
  * **The shared-memory staging rule differs by arch.** sm_89 stages `num_stages - 1`
    buffers; the H200 stages `num_stages`. See the SMEM section at the bottom -- the rule is
    now fitted to preflight's measurements rather than assumed.

One further caveat lives in `calibration_health()`: that preflight ran on a GPU another
tenant already held ~51 GB on, and its `launch_us` / `timer_tick_us` show it. Those two
numbers are carried but marked untrusted, because a study that silently flags (or silently
fails to flag) tick-limited cells against a contaminated tick is worse than one that says
"unknown".

What does NOT change is the rule the whole study rests on: **no hardware constant is ever a
literal**. Everything an autotuning guard touches comes from `env()`, which probes torch (the
source of truth -- it exposes every field we need and needs no JIT) and keeps Triton as a
cross-check only. C500 literals (warp 64, 104 SMs, 65536 B of SMEM, 131072 regs) baked into
guards pruned grids on other devices, and did not prune the two arms of a fused/unfused pair
equally -- which manufactures or destroys the ratio that is this study's only output.

bf16 only this round; no FP8 constants appear here deliberately.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

# --------------------------------------------------------------------------------------
# Verbatim from zai-org/GLM-5.2 config.json
# --------------------------------------------------------------------------------------
HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 12288  # dense MLP (first 3 layers only)
MOE_INTERMEDIATE_SIZE = 2048  # routed + shared expert MLP
NUM_HIDDEN_LAYERS = 78
FIRST_K_DENSE_REPLACE = 3  # layers [0,3) dense, [3,78) MoE
NUM_ATTENTION_HEADS = 64
NUM_KEY_VALUE_HEADS = 64
N_ROUTED_EXPERTS = 256
N_SHARED_EXPERTS = 1
NUM_EXPERTS_PER_TOK = 8  # top-k
ROUTED_SCALING_FACTOR = 2.5
NORM_TOPK_PROB = True
SCORING_FUNC = "sigmoid"
TOPK_METHOD = "noaux_tc"
N_GROUP = 1
TOPK_GROUP = 1
MOE_ROUTER_DTYPE = torch.float32

# MLA
KV_LORA_RANK = 512
Q_LORA_RANK = 2048
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = 256  # 192 + 64
V_HEAD_DIM = 256

# DSA (DeepSeek Sparse Attention) indexer
INDEX_HEAD_DIM = 128
INDEX_N_HEADS = 32
INDEX_TOPK = 2048

RMS_NORM_EPS = 1e-5
HIDDEN_ACT = "silu"  # SwiGLU = silu(gate) * up
VOCAB_SIZE = 154880
DTYPE = torch.bfloat16
ACC_DTYPE = torch.float32

# --------------------------------------------------------------------------------------
# DERIVED
# --------------------------------------------------------------------------------------
# Non-absorbed MLA (prefill path): attention emits [T, num_heads, v_head_dim].
OPROJ_K_PREFILL = NUM_ATTENTION_HEADS * V_HEAD_DIM  # 64 * 256 = 16384
# Absorbed MLA (FlashMLA decode path): attention emits [T, num_heads, kv_lora_rank]
# and W_UV is folded into o_proj.
OPROJ_K_DECODE = NUM_ATTENTION_HEADS * KV_LORA_RANK  # 64 * 512 = 32768

# Gate+Up are stored fused as w13 = [2 * moe_intermediate, hidden], matching
# sglang's FusedMoE layout (w13_weight / w2_weight).
W13_N = 2 * MOE_INTERMEDIATE_SIZE  # 4096
W2_K = MOE_INTERMEDIATE_SIZE  # 2048

# Expert-weight footprint, in bytes at bf16 (2 B/elem). This is the number that decided the
# scope of every previous port: 18.0 GiB did not fit the RTX 4060's 8 GB, which is why #6/#8/
# #9/#11a and the whole-layer bench were dropped there. On H200 (143 GB) it fits with ~7x
# headroom, so nothing is out of scope -- but the bench that allocates it must still CHECK,
# because a MIG slice or a shared node changes the answer.
EXPERT_W13_BYTES = N_ROUTED_EXPERTS * W13_N * HIDDEN_SIZE * 2  # 256*4096*6144*2 = 12.0 GiB
EXPERT_W2_BYTES = N_ROUTED_EXPERTS * HIDDEN_SIZE * W2_K * 2  # 256*6144*2048*2 =  6.0 GiB
EXPERT_WEIGHT_BYTES = EXPERT_W13_BYTES + EXPERT_W2_BYTES  # 18.0 GiB
# shared expert: same shapes as one routed expert -> 72 MiB
SHARED_EXPERT_BYTES = N_SHARED_EXPERTS * (W13_N * HIDDEN_SIZE + HIDDEN_SIZE * W2_K) * 2
ROUTER_GATE_BYTES = N_ROUTED_EXPERTS * HIDDEN_SIZE * 2  # 3.0 MiB


@dataclass(frozen=True)
class Regime:
    """One benchmark point: `T` tokens through one MoE decoder layer."""

    name: str
    T: int  # number of tokens in the batch (batch*seq for prefill, batch for decode)
    oproj_k: int
    kv_len: int = 0  # context length, for attention-core reference timing only

    @property
    def moe_rows(self) -> int:
        """Rows entering the routed-expert GEMMs = T * topk."""
        return self.T * NUM_EXPERTS_PER_TOK

    @property
    def is_prefill(self) -> bool:
        return self.oproj_k == OPROJ_K_PREFILL


# The regimes of the H200 study. Unlike `glm52/config.py` -- which carried a wider
# ladder that each bench then filtered down to five -- these ARE the study set, so a
# bench should iterate `ALL_REGIMES` directly rather than re-filtering by `T`.
# bs512/bs1024 are new: they are the batch sizes at which the routed-expert GEMMs stop being
# skinny (moe_rows = 4096 / 8192 against MOE_INTERMEDIATE 2048), i.e. where a decode kernel
# starts to behave like a prefill one. That transition is only measurable on a device whose
# memory holds all 256 experts, which is why it appears for the first time here.
# bs2/bs4/bs8/bs16 were added after the campaign to resolve the bs1 -> bs32 cliff in the
# whole-layer fusion gains (launch-latency-bound at bs1, compute-bound by bs32): they were
# measured by the dedicated `run_bs_extra_h200.py` overlay, never by a full re-campaign.
DECODE_REGIMES = [
    Regime(f"decode_bs{t}", t, OPROJ_K_DECODE, kv_len=4096)
    for t in (1, 2, 4, 8, 16, 32, 256, 512, 1024)
]
PREFILL_REGIMES = [
    Regime(f"prefill_t{t}", t, OPROJ_K_PREFILL, kv_len=t) for t in (2048, 8192)
]
ALL_REGIMES = DECODE_REGIMES + PREFILL_REGIMES
REGIMES_BY_NAME = {r.name: r for r in ALL_REGIMES}


def regime(name: str) -> Regime:
    """Look a regime up by name, failing loudly on a typo rather than silently skipping it.

    A bench that spells a regime wrong in a filter expression drops that row from its result
    table with no error, and the missing row is easy to read as "did not fit" rather than
    "never ran".
    """
    try:
        return REGIMES_BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"unknown regime {name!r}; known: {sorted(REGIMES_BY_NAME)}"
        ) from None


# --------------------------------------------------------------------------------------
# preflight_h200.json
# --------------------------------------------------------------------------------------
# `preflight.py` is the only thing in this suite that gets to say "this Triton really can
# compile and launch a TMA kernel on this GPU". It writes its findings to this path; we
# consume them, but only after proving the file describes the machine we are actually running
# on (see `BenchEnv._attach_preflight`). A preflight from a different box is worse than no
# preflight: it would switch H200 code paths on or off for reasons that have nothing to do
# with this device.
#
# `_example_preflight_sm89.json` in this directory is the sample output from the sm_89
# development box. It is deliberately NOT loaded by default -- it describes a GPU with no TMA
# and no clusters, and adopting its verdicts on an H200 would disable every Hopper path in the
# suite. Point `GLM52_H200_PREFLIGHT` at it only to exercise the *plumbing*; the device fence
# will refuse its contents on any machine that is not that laptop, which is the point.
PREFLIGHT_PATH = Path(
    os.environ.get(
        "GLM52_H200_PREFLIGHT", Path(__file__).resolve().parent / "preflight_h200.json"
    )
)

_PREFLIGHT_CACHE: "tuple[dict | None, str] | None" = None


def load_preflight(path: "Path | str | None" = None) -> "dict | None":
    """Parse `preflight_h200.json`, or return None if it is absent or unreadable.

    Absent is a normal state, not an error: the suite must run (with conservative feature
    flags and no calibration constants) on a machine where preflight has not been run yet.
    A *corrupt* file is also non-fatal for the same reason, but the reason is recorded so it
    shows up in the result file's env block instead of vanishing.
    """
    global _PREFLIGHT_CACHE
    p = Path(path) if path is not None else PREFLIGHT_PATH
    if path is None and _PREFLIGHT_CACHE is not None:
        return _PREFLIGHT_CACHE[0]
    data: dict | None = None
    why = ""
    try:
        if p.is_file():
            data = json.loads(p.read_text())
            if not isinstance(data, dict):
                data, why = None, f"{p}: top level is {type(data).__name__}, expected object"
        else:
            why = f"{p}: not found"
    except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
        data, why = None, f"{p}: {type(exc).__name__}: {exc}"
    if path is None:
        _PREFLIGHT_CACHE = (data, why)
    return data


def preflight_status() -> str:
    """Why the default preflight file was not used, or "" if it was."""
    load_preflight()
    return _PREFLIGHT_CACHE[1] if _PREFLIGHT_CACHE else ""


# --------------------------------------------------------------------------------------
# Is the timing calibration believable?
# --------------------------------------------------------------------------------------
# The peaks (bandwidth, GEMM) are throughput measurements over ~GB of traffic and survive a
# co-tenant reasonably well. The *microstructure* numbers -- per-launch cost, harness floor,
# CUDA-event tick -- do not: they are sub-10-microsecond quantities measured by sampling, and
# another process's kernels land in the same samples.
#
# The 2026-08-03 H200 preflight is exactly that case. It reports launch_us 8.89 but
# harness_floor_us 40.55, and a timer tick of 0.256 us that matched only 3 % of samples when
# the detector needs 98 % to call a tick. Both are what you get next to a co-tenant, and
# `device.mem_free_bytes` confirms one (98.8 of 150 GB free, so ~51 GB was already somebody
# else's). So we keep the numbers -- deleting a measurement is its own kind of lie -- and
# mark them, so a bench can say "tick-limited: unknown" instead of flagging (or not flagging)
# cells against a tick nobody actually measured.
#
# IMPORTANT -- what a high harness floor does NOT mean, learned the hard way (2026-08-07):
# the 2026-08-03 diagnosis blamed the 40.55 us floor itself, but every CLEAN H200 preflight
# since (36.9-42.2 us, tick match 1.0, no tenants) measures the same ~40 us floor. On H200
# the harness floor is genuinely ~4x the launch cost (fixed event-pair + first-kernel
# pipeline), so the absolute/ratio floor bars must be machine-generous. The signal that
# actually separates clean from contended is the TICK MATCH fraction (1.0 clean, 0.03-0.18
# contended), not the floor. The bars below are sanity bounds, not co-tenant detectors.
TICK_MATCH_MIN = 0.98  # preflight's own bar for declaring a tick found
FLOOR_LAUNCH_RATIO_MAX = 8.0  # floor/launch: 4060 ~0.8x, idle H200 3.6-4.7x, co-tenant >>8x
FLOOR_US_MAX = 50.0  # absolute sanity bound: idle H200 floors are 37-42 us, 4060 is 2.8 us
# Both must be exceeded before memory occupancy counts as contention: a display server or a
# CUDA context holds a gigabyte on any desktop GPU, which is 13 % of an 8 GB laptop card and
# means nothing. 4 GiB *and* 5 % is a tenant. (A small-but-busy co-tenant is caught by the
# floor/launch rule above instead, which is the signal that actually measures interference.)
PROBE_MEM_USED_FRAC_MAX = 0.05
PROBE_MEM_USED_BYTES_MIN = 4 * 2**30


def _cal_float(d: dict, key: str) -> float:
    """`d[key]` as a float, NaN when absent/null/unparsable. NaN means "not measured"."""
    try:
        v = (d or {}).get(key)
        return float(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def calibration_health(pre: "dict | None" = None) -> dict:
    """Judge whether a preflight's *timing* calibration was taken on a quiet GPU.

    Pure arithmetic on the parsed JSON -- no CUDA, no device -- so `traffic.py` (which must
    stay importable on a CPU-only checkout) can reuse this rather than growing a second,
    divergent copy of the thresholds.

    Returns a dict with `timer_tick_trusted` / `launch_trusted` / `contended`, the raw values
    behind them, and `reasons` (empty == nothing looked wrong). A field the preflight never
    recorded produces no reason: absence of evidence is not evidence of contention, and an
    older preflight that predates `timer_tick_match_frac` must not be retro-condemned.
    """
    pre = pre if pre is not None else load_preflight()
    cal = ((pre or {}).get("calibration", {}) or {})
    dev = ((pre or {}).get("device", {}) or {})

    tick = _cal_float(cal, "timer_tick_us")
    frac = _cal_float(cal, "timer_tick_match_frac")
    launch = _cal_float(cal, "launch_us")
    floor = _cal_float(cal, "harness_floor_us")
    total = _cal_float(dev, "mem_total_bytes")
    if math.isnan(total):
        total = _cal_float(dev, "total_memory")
    free = _cal_float(dev, "mem_free_bytes")
    used_frac = (
        (total - free) / total if total > 0 and not math.isnan(free) else float("nan")
    )

    tick_reasons: list = []
    timing_reasons: list = []
    if not math.isnan(frac) and frac < TICK_MATCH_MIN:
        tick_reasons.append(
            f"timer tick {tick} us matched only {frac:.0%} of samples (needs "
            f">={TICK_MATCH_MIN:.0%}); the quantum reported is a guess, not a detection"
        )
    if not math.isnan(floor) and (
        floor > FLOOR_US_MAX
        or (not math.isnan(launch) and launch > 0 and floor > FLOOR_LAUNCH_RATIO_MAX * launch)
    ):
        timing_reasons.append(
            f"harness floor {floor:.2f} us against a {launch:.2f} us launch -- a floor is a "
            f"launch plus a sync, so this one is measuring somebody else's kernels too"
        )
    if (
        not math.isnan(used_frac)
        and used_frac > PROBE_MEM_USED_FRAC_MAX
        and (total - free) > PROBE_MEM_USED_BYTES_MIN
    ):
        timing_reasons.append(
            f"{used_frac:.0%} of HBM ({(total - free) / 2**30:.0f} of "
            f"{total / 2**30:.0f} GiB) was already allocated when preflight ran -- the "
            f"device was shared"
        )

    reasons = timing_reasons + tick_reasons
    return {
        "timer_tick_us": tick,
        "timer_tick_match_frac": frac,
        "launch_us": launch,
        "harness_floor_us": floor,
        "mem_used_frac_at_probe": used_frac,
        # contention taints every sub-microsecond quantity; a bad match fraction on its own
        # only taints the tick (the launch cost is a mean over many samples, not a quantum).
        "launch_trusted": not timing_reasons,
        "timer_tick_trusted": not reasons,
        "contended": bool(timing_reasons),
        "reasons": reasons,
        "msg": (
            ""
            if not reasons
            else "timing calibration is UNRELIABLE (peaks are unaffected): "
            + "; ".join(reasons)
            + ". Re-run preflight.py on an idle GPU (`--gpu auto`) before quoting a launch "
            "cost or flagging a cell as tick-limited."
        ),
    }


# --------------------------------------------------------------------------------------
# Device probe
# --------------------------------------------------------------------------------------
_HOPPER = (9, 0)  # sm_90: TMA (tensor descriptors), thread-block clusters, warp specialization

# ---- preflight compile-probe key names, in one place so a rename cannot silently -------
# ---- disable a Hopper path by leaving a gate reading a key nobody emits any more. -------
#
# TMA has two legitimate spellings on triton 3.6 and they are NOT interchangeable in kernel
# source, so preflight probes both and either one passing means the hardware/compiler pair
# can do TMA:
#   host   -- `TensorDescriptor.from_tensor(x, [BM, BN])` built on the host, passed in as an
#             argument, used as `desc.load([...])` inside the kernel;
#   device -- `tl.make_tensor_descriptor(ptr, shape=..., strides=..., block_shape=...)` built
#             inside the kernel from a raw pointer.
# Both need `triton.set_allocator(...)` called once before the first descriptor launch or the
# launch fails at runtime -- the classic silent TMA failure.
TMA_PROBE_KEYS = ("tma_host_descriptor", "tma_device_descriptor")
# The superseded key. Its FAIL on the H200 is API misuse (a host descriptor object handed to
# the device-side constructor), not a hardware verdict, so it is recorded as evidence and
# never allowed to decide anything. Do not add it to TMA_PROBE_KEYS.
TMA_LEGACY_PROBE_KEY = "tma_tensor_descriptor"
WARP_SPEC_PROBE_KEY = "warp_specialize_tl_range"
# Recorded, never acted on: triton 3.6.0 rejects this kwarg outright ("Keyword argument
# num_consumer_groups was specified but unrecognised"). It belongs to a forked Triton that
# this suite does not target; emitting it turns a working kernel into a launch error.
WARP_SPEC_REJECTED_PROBE_KEY = "warp_specialize_num_consumer_groups"
CLUSTER_PROBE_KEY = "thread_block_cluster_num_ctas"


def _triton_api_surface() -> dict:
    """Cheap attribute inventory of the installed Triton.

    Existence is NOT evidence that a feature compiles -- the sm_89 preflight in this repo is
    exactly that case: `tl.make_tensor_descriptor` is present and the TMA kernel still fails
    at compile time. So this is used only to *veto* an optimistic capability check, never to
    grant one. Positive evidence comes from preflight's compile+launch probes.
    """
    surface = {}
    try:
        import triton
        import triton.language as tl

        surface["make_tensor_descriptor"] = hasattr(tl, "make_tensor_descriptor") or hasattr(
            tl, "_experimental_make_tensor_descriptor"
        )
        surface["set_allocator"] = hasattr(triton, "set_allocator")
        try:
            from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: F401

            surface["tensor_descriptor_cls"] = True
        except Exception:  # noqa: BLE001
            surface["tensor_descriptor_cls"] = False
        try:
            import inspect

            sig = str(inspect.signature(tl.range))
            surface["tl_range_signature"] = sig
            surface["tl_range_warp_specialize"] = "warp_specialize" in sig
        except Exception as exc:  # noqa: BLE001
            surface["tl_range_signature"] = f"unavailable: {exc}"
            surface["tl_range_warp_specialize"] = False
    except Exception as exc:  # noqa: BLE001
        surface["_error"] = f"{type(exc).__name__}: {exc}"
    return surface


def _probe_verdict(probes: dict, key: str) -> "bool | None":
    """True/False from a preflight compile probe, or None when it was never run."""
    entry = probes.get(key)
    if isinstance(entry, dict) and "ok" in entry:
        return bool(entry["ok"])
    return None


@dataclass
class BenchEnv:
    """Device/environment facts probed once, recorded into every result file."""

    device_name: str = ""
    device_index: int = 0
    device_count: int = 0
    uuid: str = ""
    cc_major: int = 0
    cc_minor: int = 0
    warp_size: int = 0
    num_sm: int = 0
    smem_bytes: int = 0
    smem_per_sm: int = 0
    regs_per_sm: int = 0
    threads_per_sm: int = 0
    l2_bytes: int = 0
    total_mem_bytes: int = 0
    free_mem_bytes: int = 0
    torch_version: str = ""
    triton_version: str = ""
    cuda_version: str = ""
    probe_ok: bool = False
    cross_checked: bool = False  # did Triton answer, so the cross-check actually happened?

    # ---- Hopper features. Resolved at runtime; see `feature_evidence` for the basis. ----
    tma_supported: bool = False
    # Which TMA spelling was demonstrated. They are different kernel source, so a kernel that
    # can only write one of them must ask for that one rather than for `tma_supported`.
    tma_host_supported: bool = False
    tma_device_supported: bool = False
    clusters_supported: bool = False
    warp_spec_supported: bool = False
    feature_source: str = "none"  # "preflight" | "capability-check" | "none"
    feature_evidence: dict = field(default_factory=dict)

    # ---- preflight provenance and calibration passthrough ----
    preflight_path: str = ""
    preflight_device_match: bool = False
    preflight_stack_match: bool = False
    preflight_timestamp: str = ""
    timer_tick_us: float = float("nan")
    launch_us: float = float("nan")
    harness_floor_us: float = float("nan")
    # False when the preflight's sub-microsecond numbers were taken next to a co-tenant; see
    # `calibration_health()`. The values above are still populated -- they are just not
    # something a bench may quote or gate a "tick-limited" flag on.
    timer_tick_trusted: bool = False
    launch_trusted: bool = False
    calib_health: dict = field(default_factory=dict)
    calib: dict = field(default_factory=dict)

    warnings: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    # ---------------------------------------------------------------------------------
    @property
    def sm_arch(self) -> str:
        return f"sm_{self.cc_major}{self.cc_minor}"

    @property
    def cc(self) -> tuple:
        return (self.cc_major, self.cc_minor)

    @property
    def is_hopper(self) -> bool:
        """sm_90 exactly -- the arch this suite is written for."""
        return self.cc == _HOPPER

    @property
    def arch_ok(self) -> bool:
        """sm_90 or newer: TMA, clusters and warp specialization are architecturally present."""
        return self.cc >= _HOPPER

    # ---------------------------------------------------------------------------------
    @staticmethod
    def probe() -> "BenchEnv":
        """Probe the device. Never guess a hardware constant.

        An earlier version of this probe fell back to C500 literals (`warpSize` 64,
        `max_shared_mem` 65536, `max_num_regs` 131072) when Triton's property query raised.
        That query JIT-builds and dlopens a C extension on first use, so it is exactly the
        call that fails on a fresh box or a stale build cache -- and when it did, every
        autotuning grid was silently built at C500 shape while the result file recorded the
        *real* device name. A wrong-but-plausible table is worse than a crash.

        So: torch is the source of truth (it exposes every field we need and needs no JIT),
        Triton is only a cross-check, and `probe_ok` records whether the two agreed. A field
        torch does not expose is taken from Triton if Triton has it and left at 0 otherwise --
        0 fails `require_ok()`, which is the intended outcome. There is no default table.
        """
        import triton

        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch.cuda.is_available() is False -- there is no device to probe. "
                "Nothing in this suite can run, and inventing device constants so that it "
                "appears to is precisely the failure this probe exists to prevent."
            )

        idx = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(idx)
        warnings: list = []

        try:
            props = triton.runtime.driver.active.utils.get_device_properties(idx)
            cross_checked = True
        except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
            props = {"_probe_error": repr(exc)}
            cross_checked = False
            warnings.append(
                f"triton device-property query failed ({type(exc).__name__}); torch values "
                f"are used unchecked. This is survivable -- torch is the source of truth -- "
                f"but the cross-check that would catch a mislabelled device did not run."
            )

        # torch names differ across versions; smem_per_block_optin is the ceiling a kernel can
        # actually reach (49152 default vs 101376 opt-in on sm89, 232448 on sm90; C500 had no
        # opt-in path and reported 65536 for both).
        smem = getattr(p, "shared_memory_per_block_optin", 0) or getattr(
            p, "shared_memory_per_block", 0
        )
        # warp_size only appeared on `_CudaDeviceProperties` in recent torch; fall back to
        # Triton rather than to a literal, and to 0 (== refuse to run) if neither knows.
        warp = int(getattr(p, "warp_size", 0) or props.get("warpSize", 0) or 0)
        if not getattr(p, "warp_size", 0) and warp:
            warnings.append("warp_size came from Triton; this torch does not expose it")

        try:
            free_b, total_b = torch.cuda.mem_get_info(idx)
        except Exception:  # noqa: BLE001
            free_b, total_b = 0, int(getattr(p, "total_memory", 0))

        env = BenchEnv(
            device_name=p.name,
            device_index=int(idx),
            device_count=torch.cuda.device_count(),
            uuid=str(getattr(p, "uuid", "")),
            cc_major=int(getattr(p, "major", 0)),
            cc_minor=int(getattr(p, "minor", 0)),
            warp_size=warp,
            num_sm=p.multi_processor_count,
            smem_bytes=smem,
            smem_per_sm=getattr(p, "shared_memory_per_multiprocessor", smem),
            regs_per_sm=p.regs_per_multiprocessor,
            threads_per_sm=p.max_threads_per_multi_processor,
            l2_bytes=getattr(p, "L2_cache_size", 0),
            total_mem_bytes=int(getattr(p, "total_memory", total_b)),
            free_mem_bytes=int(free_b),
            torch_version=torch.__version__,
            triton_version=triton.__version__,
            cuda_version=str(torch.version.cuda),
            cross_checked=cross_checked,
            warnings=warnings,
            extras={k: v for k, v in props.items()},
        )
        env.extras["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "all")

        # Cross-check against Triton where it answered; disagreement means one of them is
        # describing a different device than the kernels will actually run on. A key Triton
        # does not report is not a disagreement.
        agree = all(
            props.get(k, v) == v
            for k, v in (
                ("warpSize", env.warp_size),
                ("multiprocessor_count", env.num_sm),
                ("max_shared_mem", env.smem_bytes),
                ("max_num_regs", env.regs_per_sm),
            )
        )
        env.probe_ok = bool(
            env.warp_size and env.num_sm and env.smem_bytes and env.regs_per_sm and agree
        )

        # Publish the arch BEFORE anything can read the shared-memory model: the staging rule
        # is arch-dependent (sm_89 stages num_stages-1, sm_90 stages num_stages) and
        # `_attach_preflight` immediately evaluates it. `smem_stage_bytes` must not call
        # `env()` itself -- that would recurse straight through this line.
        _set_arch_hint((env.cc_major, env.cc_minor))
        env._attach_preflight()

        if not env.arch_ok:
            env.warnings.append(
                f"device is {env.sm_arch}, not sm_90+ -- this suite is the H200 port. It will "
                f"run (every Hopper path has a classic fallback and the flags above are off), "
                f"but the numbers are NOT H200 numbers. Check the device field before quoting."
            )
        return env

    # ---------------------------------------------------------------------------------
    def _attach_preflight(self) -> None:
        """Resolve Hopper feature flags and calibration constants.

        Two independent gates, because they protect against different mistakes:

        * `preflight_device_match` -- same GPU model, same compute capability, same SM count.
          Guards the *hardware* facts (bandwidth, timer tick, launch cost).
        * `preflight_stack_match`  -- same torch and Triton. Guards the *feature* verdicts,
          which are properties of the compiler as much as of the chip: the identical H200 with
          a different Triton can lose `tl.make_tensor_descriptor` entirely.

        Failing either gate is not fatal. We fall back to the conservative capability check
        and say so in `feature_evidence`, which lands in every result file.
        """
        pre = load_preflight()
        self.preflight_path = str(PREFLIGHT_PATH)
        if pre is None:
            self.warnings.append(
                f"no usable preflight ({preflight_status()}); Hopper features fall back to a "
                f"capability check and no measured calibration constants are available"
            )
        else:
            self.preflight_timestamp = str(pre.get("timestamp", ""))
            d = pre.get("device", {}) or {}
            s = pre.get("stack", {}) or {}
            self.preflight_device_match = bool(
                d.get("name") == self.device_name
                and int(d.get("major", -1)) == self.cc_major
                and int(d.get("minor", -1)) == self.cc_minor
                and int(d.get("multi_processor_count", -1)) == self.num_sm
            )
            self.preflight_stack_match = bool(
                str(s.get("torch")) == self.torch_version
                and str(s.get("triton")) == self.triton_version
            )
            if not self.preflight_device_match:
                self.warnings.append(
                    f"preflight_h200.json describes {d.get('name')!r} (sm_{d.get('major')}"
                    f"{d.get('minor')}, {d.get('multi_processor_count')} SM) "
                    f"but this is {self.device_name!r} ({self.sm_arch}, {self.num_sm} SM) -- "
                    f"ignored entirely. Re-run preflight.py on this machine."
                )
            elif not self.preflight_stack_match:
                self.warnings.append(
                    f"preflight_h200.json was written under torch {s.get('torch')} / triton "
                    f"{s.get('triton')}, this run is torch {self.torch_version} / triton "
                    f"{self.triton_version} -- its calibration is kept, its feature verdicts "
                    f"are not (they are compiler properties)."
                )

        usable_features = (
            bool(pre) and self.preflight_device_match and self.preflight_stack_match
        )
        usable_calib = bool(pre) and self.preflight_device_match
        self._resolve_features(pre if usable_features else None)

        if usable_calib:
            cal = (pre or {}).get("calibration", {}) or {}
            self.calib = cal

            # A probe that raised leaves its key absent or null; NaN is the honest value and
            # every consumer already has to handle "not calibrated".
            self.timer_tick_us = _cal_float(cal, "timer_tick_us")
            self.launch_us = _cal_float(cal, "launch_us")
            self.harness_floor_us = _cal_float(cal, "harness_floor_us")
            self.calib_health = calibration_health(pre)
            self.timer_tick_trusted = bool(self.calib_health["timer_tick_trusted"])
            self.launch_trusted = bool(self.calib_health["launch_trusted"])
            if self.calib_health["msg"]:
                self.warnings.append(self.calib_health["msg"])
            for note in smem_model_check(pre):
                if not note["ok"]:
                    self.warnings.append(note["msg"])

        # Recorded unconditionally: how the grid was pruned is part of what a result table
        # means, and it is exactly the field that is missing when an old table turns out to
        # be unreadable. `stage_rule()` reads the arch hint set in `probe()` above.
        self.extras["smem_stage_rule"] = smem_model_description()

    def _resolve_features(self, pre: "dict | None") -> None:
        """Positive evidence first (a probe that compiled AND launched), capability second."""
        api = _triton_api_surface()
        probes = ((pre or {}).get("triton_features", {}) or {}).get("compile_probes", {}) or {}
        self.feature_source = "preflight" if probes else "capability-check"
        ev: dict = {"triton_api_surface": api, "arch": self.sm_arch, "arch_ok": self.arch_ok}

        def decide(name: str, probe_keys: tuple, api_ok: bool, api_note: str) -> bool:
            """Any probe that PASSED grants the capability; all-FAIL denies it; no probe at
            all falls through to the capability check. Several keys because a capability can
            have more than one legal spelling (TMA has two) and demonstrating either one
            proves the hardware/compiler pair can do it."""
            ran = {
                k: v
                for k, v in ((k, _probe_verdict(probes, k)) for k in probe_keys)
                if v is not None
            }
            if not ran:
                # No probe: sm_90+ is necessary, and the Triton API surface must at least
                # exist. This is a *belief*, not a measurement -- kernels that act on it must
                # still try the Hopper path inside a try/except and fall back on failure.
                got = bool(self.arch_ok and api_ok)
                ev[name] = (
                    f"capability check (NOT compile-verified): arch {self.sm_arch} "
                    f"{'ok' if self.arch_ok else '< sm_90'}, api {api_note}"
                    f"{'' if api_ok else ' MISSING'} -> {got}"
                )
                return got
            got = any(ran.values())
            if got and not self.arch_ok:
                # e.g. `tl.range(warp_specialize=True)` compiles happily on sm_89 and then
                # silently does nothing. A probe passing on the wrong arch is not evidence.
                ev[name] = (
                    f"preflight probe(s) {'/'.join(ran)} PASSED but arch is {self.sm_arch} "
                    f"< sm_90; treated as unsupported (the feature compiles, then is ignored)"
                )
                return False
            ev[name] = "preflight compile+launch: " + "; ".join(
                f"{k}={'PASS' if v else 'FAIL'}"
                + (
                    ""
                    if v
                    else " (" + str((probes.get(k) or {}).get("error", ""))[:120] + ")"
                )
                for k, v in ran.items()
            ) + f" -> {got}"
            return got

        # TMA. Deliberately NOT gated on TMA_LEGACY_PROBE_KEY: that probe handed a host-side
        # TensorDescriptor to the device-side constructor, so its FAIL is a bug in the probe
        # and holds on every GPU ever built. When neither new key is present (a preflight that
        # predates them, e.g. the file the H200 first returned) we fall through to the sm_90+
        # capability check -- which is the correct answer for an H200 and the conservative one
        # everywhere else.
        self.tma_supported = decide(
            "tma",
            TMA_PROBE_KEYS,
            bool(
                api.get("make_tensor_descriptor")
                and api.get("tensor_descriptor_cls")
                and api.get("set_allocator")
            ),
            "make_tensor_descriptor + TensorDescriptor + set_allocator",
        )
        # Per-spelling detail. A key that never ran leaves its flag at the capability answer,
        # since neither spelling has been ruled out in that case.
        for attr, key in (
            ("tma_host_supported", TMA_PROBE_KEYS[0]),
            ("tma_device_supported", TMA_PROBE_KEYS[1]),
        ):
            v = _probe_verdict(probes, key)
            setattr(self, attr, self.tma_supported if v is None else bool(v and self.arch_ok))
        legacy = _probe_verdict(probes, TMA_LEGACY_PROBE_KEY)
        if legacy is not None:
            ev["tma_legacy_probe"] = (
                f"{TMA_LEGACY_PROBE_KEY}={'PASS' if legacy else 'FAIL'} -- NOT USED as "
                f"evidence: that probe passes a host TensorDescriptor object into "
                f"tl.make_tensor_descriptor(), which takes a raw pointer, so it fails to "
                f"compile regardless of hardware. Re-run a preflight that emits "
                f"{'/'.join(TMA_PROBE_KEYS)}."
            )
        self.clusters_supported = decide(
            "clusters",
            (CLUSTER_PROBE_KEY,),
            True,  # num_ctas is a launch kwarg in every Triton this suite supports
            "num_ctas launch kwarg",
        )
        self.warp_spec_supported = decide(
            "warp_spec",
            (WARP_SPEC_PROBE_KEY,),
            bool(api.get("tl_range_warp_specialize")),
            "tl.range(warp_specialize=...)",
        )
        # The older forked-Triton spelling. Recorded, never acted on and never emitted:
        # triton 3.6.0 rejects the kwarg outright, so a kernel that carried it would not
        # launch at all. `warp_spec_supported` above is about `tl.range` and nothing else.
        alt = _probe_verdict(probes, WARP_SPEC_REJECTED_PROBE_KEY)
        if alt is not None:
            ev["warp_spec_num_consumer_groups"] = (
                f"{'PASS' if alt else 'FAIL'} -- recorded only; this spelling is never "
                f"emitted by this suite (unrecognised keyword on triton 3.6)"
            )
        self.feature_evidence = ev

    # ---------------------------------------------------------------------------------
    def require_ok(self) -> "BenchEnv":
        """Fail loudly before any grid is built. Call once at bench startup."""
        if not self.probe_ok:
            raise RuntimeError(
                f"device probe degraded or inconsistent -- refusing to autotune.\n"
                f"  {self.device_name}: warp={self.warp_size} sm={self.num_sm} "
                f"smem={self.smem_bytes} regs={self.regs_per_sm}\n"
                f"  triton props: {self.extras}\n"
                f"Building a search grid from wrong hardware constants produces a "
                f"plausible table that is silently wrong; fix the probe instead."
            )
        return self

    def l2_flush_bytes(self, min_mb: int = 256, mult: int = 4) -> int:
        """Bytes a between-measurement L2 flush buffer must cover.

        Derived, never assumed. The C500 suite used a bare 128 MB, justified by "C500's L2 is
        8 MB"; that survived the 4060 (32 MB L2, still 4x) by luck. **H200's L2 measures
        62914560 B (60 MiB)**, where 128 MB is barely 2x -- close enough that a flush would
        leave part of the working set resident and quietly turn every measurement into a
        warm-cache one, which flatters whichever arm re-reads an intermediate. That is a
        fusion-ratio bias, not just noise.

        `min_mb` matches the harness floor in `common.py` (`MIN_FLUSH_BYTES`) so the two
        cannot disagree about how much memory a flush touches. On the measured H200 the two
        terms nearly coincide -- `4 * L2` is 240 MiB against the 256 MiB floor -- so the floor
        binds by a hair and the flush covers 4.27x L2. If that floor is ever lowered, the
        `mult * L2` term takes over automatically, which is the point of computing both.
        """
        return max(min_mb * 2**20, mult * int(self.l2_bytes))

    def fingerprint(self) -> str:
        """Stable identity of the GPU these numbers came from.

        Write it into every checkpoint and refuse to reuse a checkpoint whose fingerprint
        differs: a stale checkpoint from another GPU was one call away from being republished
        as a fresh measurement in an earlier port. UUID first because it is the only field
        that distinguishes two identical H200s in one node -- but it is not always exposed,
        so the shape fields are part of the key too.
        """
        return "|".join(
            [
                self.device_name or "?",
                self.sm_arch,
                f"{self.num_sm}SM",
                f"{self.total_mem_bytes >> 20}MiB",
                self.uuid or "no-uuid",
            ]
        )

    def banner(self) -> str:
        return (
            f"[env] {self.device_name} ({self.sm_arch}) | {self.num_sm} SM | "
            f"warp {self.warp_size} | smem {self.smem_bytes} B | regs/SM {self.regs_per_sm} | "
            f"threads/SM {self.threads_per_sm} | L2 {self.l2_bytes >> 20} MB | "
            f"HBM {self.total_mem_bytes >> 30} GB | "
            f"torch {self.torch_version} triton {self.triton_version} cuda {self.cuda_version}"
        )

    def feature_banner(self) -> str:
        src = self.feature_source
        if src == "preflight":
            src += (
                f" ({self.preflight_timestamp}"
                f"{'' if self.preflight_stack_match else ', stack MISMATCH'})"
            )
        # NaN means "no calibration for this device"; don't print a tick nobody measured.
        # A tick that WAS measured but on a contended GPU is worse than no tick if it is
        # printed bare, so it never is: the marker travels with the number.
        tick = (
            ""
            if math.isnan(self.timer_tick_us)
            else f" | timer tick {self.timer_tick_us} us"
            + ("" if self.timer_tick_trusted else " (UNTRUSTED)")
        )
        return (
            f"[hopper] tma={self.tma_supported} clusters={self.clusters_supported} "
            f"warp_spec={self.warp_spec_supported} | source: {src}{tick}"
        )

    def warning_banner(self) -> str:
        """Every recorded warning, one per line, or "" when there are none."""
        return "\n".join(f"[env!] {w}" for w in self.warnings)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["sm_arch"] = self.sm_arch
        d["fingerprint"] = self.fingerprint()
        d["l2_flush_bytes"] = self.l2_flush_bytes()
        return d


_ENV_CACHE: "BenchEnv | None" = None


def env() -> "BenchEnv":
    """Probe the device once and reuse it.

    Every hardware constant a bench needs -- shared-memory ceiling, warp width, SM count,
    register file, L2 size, and now the three Hopper feature flags -- must come from here
    rather than a literal. The C500 study hardcoded `65536` and `* 64` throughout; on any
    other device those silently prune the autotuning grid, and they do not prune both arms of
    a fused/unfused pair equally, which corrupts the ratio that is this study's only output.
    """
    global _ENV_CACHE
    if _ENV_CACHE is None:
        # require_ok() here rather than at each call site: every bench reaches its hardware
        # constants through env(), so one choke point protects all of them, including any
        # that print no environment banner an operator could check.
        _ENV_CACHE = BenchEnv.probe().require_ok()
    return _ENV_CACHE


def same_device(meta: "dict | None") -> bool:
    """True if `meta` (a checkpoint's `_meta`) was written by the GPU we are running on."""
    if not meta:
        return False
    fp = meta.get("device_fingerprint")
    if fp:
        return str(fp) == env().fingerprint()
    # Older payloads carry only a device name. Weaker, but still catches the cross-machine
    # case that matters; absence of both fields is treated as "not this device".
    name = meta.get("device") or meta.get("device_name")
    return bool(name) and str(name) == env().device_name


def require_same_device(meta: "dict | None", path: str = "") -> None:
    """Refuse to reuse a checkpoint that another device wrote.

    A stale checkpoint from another GPU was one call away from being republished as a fresh
    measurement during the 4060 port. Deleting it is the operator's decision; silently
    re-timing against it is never ours.
    """
    if not same_device(meta):
        m = meta or {}
        who = m.get("device_fingerprint") or m.get("device") or "an unknown device"
        raise RuntimeError(
            f"checkpoint {path or '(unnamed)'} was written by {who}, this run is "
            f"{env().fingerprint()} -- refusing to reuse it. Delete the checkpoint or point "
            f"GLM52_H200_RESULTS_DIR somewhere else."
        )


def capacity_check(free_bytes: "int | None" = None) -> dict:
    """Does this device hold the routed experts, and the whole layer?

    H200's 150 GB (143771 MiB) is the reason every fusion and the whole-layer combination
    benchmark are in scope this round -- but "H200" is not the same as "150 GB free": a MIG
    slice, a second tenant, or a leaked allocation from an earlier bench all change the
    answer, and an allocation of 18 GiB that fails takes the run down after the tuning has
    already been paid for. Thresholds match `preflight.py`'s capacity section (85 % of free
    for the weights alone; 60 % for the whole layer, which must also hold activations, the
    [T*topk, 6144] intermediate and the autotuner's transient scratch) so the two cannot
    disagree.

    The measured case is the instructive one: the 2026-08-03 preflight ran with 51 GB already
    held by another tenant, saw 98.6 GB free, and still returned `whole_layer_feasible: True`
    (19.3 GB is 20 % of what was left). Even a crowded H200 has the headroom -- which is why
    picking an *idle* GPU is about measurement noise, not about fitting.
    """
    free = int(free_bytes if free_bytes is not None else env().free_mem_bytes)
    return {
        "expert_weight_bytes": EXPERT_WEIGHT_BYTES,
        "free_bytes": free,
        "expert_weights_fit": EXPERT_WEIGHT_BYTES < free * 0.85,
        "whole_layer_feasible": EXPERT_WEIGHT_BYTES < free * 0.60,
        "note": f"{EXPERT_WEIGHT_BYTES / 2**30:.1f} GiB of expert weights against "
                f"{free / 2**30:.1f} GiB free",
    }


# --------------------------------------------------------------------------------------
# Shared-memory model
# --------------------------------------------------------------------------------------
# GROUND TRUTH. The number of mainloop buffers Triton allocates is a property of the
# *backend*, not just of the Triton version, and the two stacks we have measured disagree
# even though both are triton 3.6.0:
#
#   sm_89 / triton 3.6  ->  num_stages - 1 buffers, floor 2.
#       68 configs launched, `CompiledKernel.metadata.shared` read back: exact on 64,
#       conservative on 4 (all num_stages=2, where the floor binds anyway).
#
#   sm_90 / triton 3.6  ->  num_stages buffers.  All FIVE points in preflight_h200.json
#       reproduce EXACTLY under `num_stages * 2 * BK * (BM + BN)`; none reproduces under
#       `num_stages - 1`:
#         BM128 BN128 BK64 s3 ->  98304 == 3 * 2*64*(128+128)   [ns-1 would say  65536]
#         BM128 BN256 BK64 s3 -> 147456 == 3 * 2*64*(128+256)   [ns-1 would say  98304]
#         BM128 BN256 BK64 s4 -> 196608 == 4 * 2*64*(128+256)   [ns-1 would say 147456]
#         BM256 BN256 BK64 s3 -> 196608 == 3 * 2*64*(256+256)   [ns-1 would say 131072]
#         BM128 BN256 BK128 s3 -> the compiler's own OutOfResources message says
#                                 "Required: 294912" == 3 * 2*128*(128+256), against the
#                                 232448 B opt-in ceiling.  (`ns-1` would have predicted
#                                 196608 and waved this config through.)
#
# So the `num_stages - 1` model does NOT hold on Hopper: it under-predicts by (s-1)/s, i.e.
# 33 % at s=3. That is the *benign* direction -- the config is tried and, if it really does
# not fit, fails to compile into `n_failed` where the fairness check sees it -- but it wastes
# a compile on every over-large config and it would have let BM128/BN256/BK128/s3 into the
# grid. Nothing observed here needs an extra term for TMA descriptors or mbarriers: the five
# points are exact with none.
#
# Rather than swap one hardcoded rule for another, the rule is now *fitted* to whatever
# preflight measured on this stack, with the two verified rules as the only candidates and an
# arch-keyed default when there is no preflight to fit to.
STAGE_RULES: dict = {
    "num_stages": lambda s: max(2, int(s)),
    "num_stages-1": lambda s: max(2, int(s) - 1),
}
# Arch defaults, used only when no preflight observation is available to fit. sm_90+ gets the
# H200-measured rule; everything else keeps the sm_89-measured one, which is also the
# under-predicting (safe-direction) choice when the arch is unknown.
STAGE_RULE_HOPPER = "num_stages"
STAGE_RULE_CLASSIC = "num_stages-1"
SMEM_MODEL_VERIFIED_ON = (
    "triton 3.6 / sm_89 (68 configs) -> num_stages-1; "
    "triton 3.6 / sm_90 H200 (5 preflight configs) -> num_stages"
)

# Set by `BenchEnv.probe()`. `stage_rule()` must never call `env()` -- `probe()` evaluates the
# smem model while it is still constructing the env, so that would recurse.
_ARCH_HINT: "tuple[int, int] | None" = None
_STAGE_RULE_CACHE: "tuple[tuple, tuple[str, str]] | None" = None


def _set_arch_hint(cc: "tuple[int, int]") -> None:
    global _ARCH_HINT
    _ARCH_HINT = (int(cc[0]), int(cc[1]))


def _parse_smem_probe(pre: "dict | None") -> list:
    """preflight's `calibration.smem_probe` as [(bm, bn, bk, num_stages, shared_bytes)].

    Both outcomes are evidence. A config that compiled reports `shared_bytes`; a config that
    did NOT compile still carries the compiler's own "Required: N" in its OutOfResources text,
    and that N is the same quantity `metadata.shared` would have held. Reading it is how the
    BK128 point -- the only one in the file that exceeds the hardware ceiling, and therefore
    the only one that tests the model where it actually matters -- gets used at all.
    """
    obs = ((pre or {}).get("calibration", {}) or {}).get("smem_probe", {}) or {}
    out = []
    for key, rec in obs.items():
        if not isinstance(rec, dict):
            continue
        actual = rec.get("shared_bytes")
        if not isinstance(actual, int) or actual <= 0:
            m = re.search(r"Required:\s*(\d+)", str(rec.get("error", "")))
            actual = int(m.group(1)) if m else 0
        if actual <= 0:
            continue
        try:
            parts = dict(
                (p[:2].lower(), int(p[2:])) if p[0] == "B" else ("s", int(p[1:]))
                for p in key.split("_")
            )
            out.append(
                (key, parts["bm"], parts["bn"], parts["bk"], parts["s"], actual)
            )
        except Exception:  # noqa: BLE001 -- an unparsable key is preflight's business
            continue
    return out


def stage_rule() -> tuple:
    """(rule name, provenance). Which multi-buffer rule this stack actually uses.

    Order: explicit `GLM52_H200_SMEM_RULE` override -> fit to preflight's measurements ->
    arch default. The fit only runs when preflight describes the same arch we are on (or the
    arch is not yet known), because the staging rule is a backend property: adopting an
    sm_90 fit on sm_89 would OVER-predict by s/(s-1) and silently prune legal configs, which
    is the one failure mode this whole model is written to avoid.
    """
    global _STAGE_RULE_CACHE
    override = os.environ.get("GLM52_H200_SMEM_RULE", "").strip()
    key = (override, _ARCH_HINT, str(PREFLIGHT_PATH))
    if _STAGE_RULE_CACHE is not None and _STAGE_RULE_CACHE[0] == key:
        return _STAGE_RULE_CACHE[1]

    res: tuple
    if override in STAGE_RULES:
        res = (override, "GLM52_H200_SMEM_RULE")
    else:
        pre = load_preflight()
        obs = _parse_smem_probe(pre)
        pre_cc = None
        d = (pre or {}).get("device", {}) or {}
        try:
            pre_cc = (int(d["major"]), int(d["minor"]))
        except Exception:  # noqa: BLE001 -- an older preflight may not carry them
            pre_cc = None
        arch_compatible = _ARCH_HINT is None or pre_cc is None or pre_cc == _ARCH_HINT
        fitted = ""
        if obs and arch_compatible:
            for name, fn in STAGE_RULES.items():
                if all(
                    fn(s) * 2 * bk * (bm + bn) == actual
                    for _, bm, bn, bk, s, actual in obs
                ):
                    fitted = name
                    break
        if fitted:
            res = (fitted, f"fitted to {len(obs)} preflight smem_probe observation(s)")
        else:
            default = (
                STAGE_RULE_HOPPER
                if (_ARCH_HINT or (0, 0)) >= _HOPPER
                else STAGE_RULE_CLASSIC
            )
            why = (
                "no preflight smem_probe observations"
                if not obs
                else f"preflight describes sm_{pre_cc[0]}{pre_cc[1]}, this is "
                     f"sm_{(_ARCH_HINT or (0, 0))[0]}{(_ARCH_HINT or (0, 0))[1]}"
                if not arch_compatible
                else "NO candidate rule reproduces preflight's measurements -- the staging "
                     "rule changed again and must be re-derived"
            )
            res = (default, f"arch default ({why})")
        if override:
            res = (res[0], res[1] + f"; ignored GLM52_H200_SMEM_RULE={override!r} (unknown)")
    _STAGE_RULE_CACHE = (key, res)
    return res


def stage_buffers(num_stages: int) -> int:
    """How many mainloop buffers the compiler will allocate for `num_stages`."""
    return STAGE_RULES[stage_rule()[0]](num_stages)


def smem_model_description() -> str:
    """One line naming the active rule and where it came from, for a result file's metadata.

    Benches record how their grid was pruned; a literal like "num_stages-1 buffers" in that
    metadata is a claim that goes stale the moment the rule is refitted, and a stale claim
    about how the grid was pruned is exactly the kind of thing that makes an old table
    unreadable. Ask here instead.
    """
    rule, why = stage_rule()
    return f"{rule} buffers (floor 2) x 2 x BK x (BM + bn_mult*BN); {why}"


def smem_stage_bytes(bm: int, bn: int, bk: int, num_stages: int, bn_mult: int = 1) -> int:
    """Shared memory a Triton GEMM mainloop stages, in bytes.

    `buffers(num_stages) * 2 B/elem * BK * (BM + bn_mult*BN)`, where `buffers` is the rule
    resolved by `stage_rule()` -- `num_stages` on sm_90, `num_stages - 1` (floor 2) on sm_89,
    both measured, neither assumed. See the block comment above for the five H200 points.

    Two standing caveats survive the correction:

      * prefer a **trial compilation** -- launch the config and read `metadata.shared` -- over
        this model wherever a bench can afford to. Hopper has terms this formula does not
        model at all: TMA descriptors and their mbarriers occupy shared memory that no
        `BM/BN/BK` term accounts for, and warp specialization can multi-buffer differently
        from the classic mainloop. The five measured points happen to need no such term, but
        all five are classic mainloops.
      * where the model is used as a pre-filter, remember which way it fails. Under-predicting
        is benign: the config is tried, fails to compile, and lands in `n_failed` where the
        fairness check can see it. **Over-predicting is the dangerous direction** -- the config
        is never tried, disappears from the grid without a trace, and (because the fused and
        unfused arms have different tile shapes) does not prune both arms equally.

    `smem_model_check()` compares this model against whatever `preflight_h200.json` actually
    measured, and `env()` promotes any over-prediction to a recorded warning.
    """
    return stage_buffers(num_stages) * 2 * bk * (bm + bn_mult * bn)


def smem_model_check(pre: "dict | None" = None) -> list:
    """Compare `smem_stage_bytes()` against preflight's measured `metadata.shared`.

    preflight launches a handful of `BM*_BN*_BK*_s*` configs and records the shared-memory
    figure the compiler actually reserved (or, for a config that did not fit, the figure the
    compiler said it would have needed). This replays the model over those observations and
    reports each one. It is not circular even though `stage_rule()` may have been fitted to
    the same data: the fit adopts a rule only if it is exact on EVERY point, so this either
    confirms that or reports the point where a hand-pinned / arch-defaulted rule breaks.
    Returns [] when there is nothing to check.
    """
    pre = pre if pre is not None else load_preflight()
    rule, why = stage_rule()
    out = []
    for key, bm, bn, bk, st, actual in _parse_smem_probe(pre):
        model = smem_stage_bytes(bm, bn, bk, st)
        # model <= actual is the safe direction: the config passes the pre-filter, is tried,
        # and if it really does not fit it fails to compile and lands in `n_failed`, where the
        # fairness check can see it. model > actual is the silent one.
        ok = model <= actual
        out.append(
            {
                "cfg": key,
                "model_bytes": model,
                "actual_bytes": actual,
                "rule": rule,
                "ok": ok,
                "msg": (
                    f"smem model [{rule}, {why}] "
                    f"{'matches' if model == actual else 'differs'} at {key}: "
                    f"model {model} B vs measured {actual} B"
                    + (
                        ""
                        if ok
                        else " -- the model OVER-predicts, so legal configs are being pruned "
                        "from the grid without ever being tried. Re-derive the staging rule "
                        f"for this stack (verified rules: {SMEM_MODEL_VERIFIED_ON})."
                    )
                ),
            }
        )
    return out
