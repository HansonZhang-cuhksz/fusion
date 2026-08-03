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


# The seven regimes of the H200 study. Unlike `glm52/config.py` -- which carried a wider
# ladder that each bench then filtered down to five -- these seven ARE the study set, so a
# bench should iterate `ALL_REGIMES` directly rather than re-filtering by `T`.
# bs512/bs1024 are new: they are the batch sizes at which the routed-expert GEMMs stop being
# skinny (moe_rows = 4096 / 8192 against MOE_INTERMEDIATE 2048), i.e. where a decode kernel
# starts to behave like a prefill one. That transition is only measurable on a device whose
# memory holds all 256 experts, which is why it appears for the first time here.
DECODE_REGIMES = [
    Regime(f"decode_bs{t}", t, OPROJ_K_DECODE, kv_len=4096)
    for t in (1, 32, 256, 512, 1024)
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
# Device probe
# --------------------------------------------------------------------------------------
_HOPPER = (9, 0)  # sm_90: TMA (tensor descriptors), thread-block clusters, warp specialization


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

            def _f(key: str) -> float:
                # A probe that raised leaves its key absent or null; NaN is the honest value
                # and every consumer already has to handle "not calibrated".
                try:
                    v = cal.get(key)
                    return float(v) if v is not None else float("nan")
                except (TypeError, ValueError):
                    return float("nan")

            self.timer_tick_us = _f("timer_tick_us")
            self.launch_us = _f("launch_us")
            self.harness_floor_us = _f("harness_floor_us")
            for note in smem_model_check(pre):
                if not note["ok"]:
                    self.warnings.append(note["msg"])

    def _resolve_features(self, pre: "dict | None") -> None:
        """Positive evidence first (a probe that compiled AND launched), capability second."""
        api = _triton_api_surface()
        probes = ((pre or {}).get("triton_features", {}) or {}).get("compile_probes", {}) or {}
        self.feature_source = "preflight" if probes else "capability-check"
        ev: dict = {"triton_api_surface": api, "arch": self.sm_arch, "arch_ok": self.arch_ok}

        def decide(name: str, probe_key: str, api_ok: bool, api_note: str) -> bool:
            verdict = _probe_verdict(probes, probe_key)
            if verdict is None:
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
            if verdict and not self.arch_ok:
                # e.g. `tl.range(warp_specialize=True)` compiles happily on sm_89 and then
                # silently does nothing. A probe passing on the wrong arch is not evidence.
                ev[name] = (
                    f"preflight probe {probe_key} PASSED but arch is {self.sm_arch} < sm_90; "
                    f"treated as unsupported (the feature compiles and is then ignored)"
                )
                return False
            detail = ""
            if not verdict:
                detail = " -- " + str((probes.get(probe_key) or {}).get("error", ""))[:160]
            ev[name] = (
                f"preflight compile+launch probe {probe_key}: "
                f"{'PASS' if verdict else 'FAIL'}{detail}"
            )
            return bool(verdict)

        self.tma_supported = decide(
            "tma",
            "tma_tensor_descriptor",
            bool(
                api.get("make_tensor_descriptor")
                and api.get("tensor_descriptor_cls")
                and api.get("set_allocator")
            ),
            "make_tensor_descriptor + TensorDescriptor + set_allocator",
        )
        self.clusters_supported = decide(
            "clusters",
            "thread_block_cluster_num_ctas",
            True,  # num_ctas is a launch kwarg in every Triton this suite supports
            "num_ctas launch kwarg",
        )
        self.warp_spec_supported = decide(
            "warp_spec",
            "warp_specialize_tl_range",
            bool(api.get("tl_range_warp_specialize")),
            "tl.range(warp_specialize=...)",
        )
        # The older forked-Triton spelling. Recorded rather than acted on: it changes the
        # kernel's source, so a bench that wants it must ask for it explicitly.
        alt = _probe_verdict(probes, "warp_specialize_num_consumer_groups")
        if alt is not None:
            ev["warp_spec_num_consumer_groups"] = "PASS" if alt else "FAIL"
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
        8 MB"; that survived the 4060 (32 MB L2, still 4x) by luck. **H200's L2 is ~50 MB**,
        where 128 MB is only 2.5x -- close enough that a flush would leave part of the working
        set resident and quietly turn every measurement into a warm-cache one, which flatters
        whichever arm re-reads an intermediate. That is a fusion-ratio bias, not just noise.

        `min_mb` matches the harness floor in `common.py` (`MIN_FLUSH_BYTES`) so the two
        cannot disagree about how much memory a flush touches; on H200 the `4 * L2` term is
        the smaller of the two and the floor is what binds.
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
        tick = (
            "" if math.isnan(self.timer_tick_us) else f" | timer tick {self.timer_tick_us} us"
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

    H200's 143 GB is the reason every fusion and the whole-layer combination benchmark are in
    scope this round -- but "H200" is not the same as "143 GB free": a MIG slice, a second
    tenant, or a leaked allocation from an earlier bench all change the answer, and an
    allocation of 18 GB that fails takes the run down after the tuning has already been paid
    for. Thresholds match `preflight.py`'s capacity section (85 % of free for the weights
    alone; 60 % for the whole layer, which must also hold activations, the [T*topk, 6144]
    intermediate and the autotuner's transient scratch) so the two cannot disagree.
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
def smem_stage_bytes(bm: int, bn: int, bk: int, num_stages: int, bn_mult: int = 1) -> int:
    """Shared memory a Triton GEMM mainloop stages, in bytes.

    Triton 3.0 (the C500 stack) allocated `num_stages` buffers; **Triton 3.6 allocates
    `num_stages - 1`**, with a floor of 2. That figure was measured on **Triton 3.6 / sm_89**
    by launching 68 configs and reading `CompiledKernel.metadata.shared`: exact on 64 of them
    and conservative on the other 4 (all `num_stages=2`).

    **It has not been re-verified on the H200 stack and must not be assumed there.** Hopper
    changes the picture in at least two ways this formula does not model: TMA descriptors and
    their mbarriers occupy shared memory that no `BM/BN/BK` term accounts for, and warp
    specialization can multi-buffer differently from the classic mainloop. So:

      * prefer a **trial compilation** -- launch the config and read `metadata.shared` -- over
        this model wherever a bench can afford to;
      * where the model is used as a pre-filter, remember which way it fails. Under-predicting
        is benign: the config is tried, fails to compile, and lands in `n_failed` where the
        fairness check can see it. **Over-predicting is the dangerous direction** -- the config
        is never tried, disappears from the grid without a trace, and (because the fused and
        unfused arms have different tile shapes) does not prune both arms equally.

    `smem_model_check()` compares this model against whatever `preflight_h200.json` actually
    measured, and `env()` promotes any over-prediction to a recorded warning.
    """
    return max(2, num_stages - 1) * 2 * bk * (bm + bn_mult * bn)


SMEM_MODEL_VERIFIED_ON = "triton 3.6 / sm_89 (68 configs, CompiledKernel.metadata.shared)"


def smem_model_check(pre: "dict | None" = None) -> list:
    """Compare `smem_stage_bytes()` against preflight's measured `metadata.shared`.

    preflight launches a handful of `BM*_BN*_BK*_s*` configs and records the shared-memory
    figure the compiler actually reserved. This replays the model over those observations and
    reports each one, so the H200 run can confirm -- or refute -- the `num_stages - 1` rule
    before a single grid is pruned by it. Returns [] when there is nothing to check.
    """
    pre = pre if pre is not None else load_preflight()
    obs = ((pre or {}).get("calibration", {}) or {}).get("smem_probe", {}) or {}
    out = []
    for key, rec in obs.items():
        if not isinstance(rec, dict) or not rec.get("ok"):
            continue  # a config that failed to compile tells us nothing about the formula
        actual = rec.get("shared_bytes")
        if not isinstance(actual, int) or actual <= 0:
            continue
        try:
            parts = dict(
                (p[:2].lower(), int(p[2:])) if p[0] == "B" else ("s", int(p[1:]))
                for p in key.split("_")
            )
            bm, bn, bk, st = parts["bm"], parts["bn"], parts["bk"], parts["s"]
        except Exception:  # noqa: BLE001 -- an unparsable key is preflight's business
            continue
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
                "ok": ok,
                "msg": (
                    f"smem model {'matches' if model == actual else 'differs'} at {key}: "
                    f"model {model} B vs measured {actual} B"
                    + (
                        ""
                        if ok
                        else " -- the model OVER-predicts, so legal configs are being pruned "
                        "from the grid without ever being tried. Re-derive the staging rule "
                        f"for this stack (model verified on {SMEM_MODEL_VERIFIED_ON})."
                    )
                ),
            }
        )
    return out
