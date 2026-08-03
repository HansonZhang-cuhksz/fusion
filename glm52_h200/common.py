"""Shared benchmark / autotune / validation harness for the H200 port.

Ported from `glm52/common.py` (the audited C500 + RTX 4060 suite) with the fixes this
study paid for in failed runs.  The rules below are what make a fused-vs-unfused ratio
mean anything; each one exists because breaking it produced a plausible, publishable,
WRONG table.

 1. **Chain timing.**  An unfused variant is a *sequence* of kernels.  We time the whole
    sequence as one unit, with a single L2 flush before the sequence -- never between its
    kernels.  Flushing between them fabricates a fusion win, because in real execution the
    producer's output is still resident in L2 when the consumer starts.

 2. **Independent tuning.**  `autotune()` searches each variant's own config space and
    returns that variant's own optimum.  A fused kernel and its unfused counterpart never
    share a config.  Comparing a tuned kernel against an untuned one is the single easiest
    way to manufacture a fake result.

 3. **Same source, flags differ.**  Kernels are written once with `tl.constexpr` flags
    selecting the fused epilogue/prologue.  Mapping (tile sizes, warps, stages, loop order)
    is the only thing allowed to differ between the two arms.

 4. **Interleave the arms in the final measurement.**  `bench_pair()` alternates
    A/B/A/B inside ONE loop and reports the PAIRED ratio.  The 4060 campaign timed the
    whole fused arm and then the whole unfused arm; the GPU drifted 22% thermally *within
    one run* and the resulting speedup landed above the cell's own physical ceiling
    (LOG-13 Sec 9.1).  Per-regime headline numbers must come from `bench_pair`.

 5. **Never assume a hardware constant.**  The L2 flush buffer is sized from the device
    (>= 4x L2, >= 256 MB).  H200's L2 is ~50 MB; the 8 MB-shaped buffer inherited from
    C500 would have turned every "cold" measurement into a warm-cache one, flattering
    whichever arm re-reads an intermediate.  If L2 cannot be determined we CRASH -- a
    quietly-undersized flush is undetectable after the fact.

 6. **Device-fence every checkpoint.**  `ckpt_load()` refuses a checkpoint written by a
    different device.  A stale C500 checkpoint was one call away from being republished
    inside a freshly-probed 4060 `env` block, i.e. a result file indistinguishable from a
    real run.

 7. **Record grid sizes and compile failures per arm.**  `n_tried` / `n_failed` /
    `n_guard_rejected` per side are what make an unfair comparison detectable after the
    fact.  A compile failure is recorded and skipped, never fatal.

 8. **Timer granularity is a first-class number.**  Decode kernels resolve to 9-17 CUDA
    event ticks; `ticks()` / `tick_note()` expose that so a bench can flag a speedup whose
    two operands are only a few ticks apart instead of quoting 4 significant figures.

This module builds no autotuning grids and holds no architecture constants: grid guards
belong to `glm52_h200/config.py`, which owns the single cached device probe.  What lives
here is only what the *measurement* needs.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

HERE = Path(__file__).resolve().parent
HARNESS_SCHEMA = 1

# The surface every bench is expected to reach for.  Anything not here is an internal.
__all__ = [
    # timing
    "Timing", "PairTiming", "bench_chain", "bench_pair",
    # tuning
    "TuneResult", "autotune", "neighbours",
    # correctness
    "rel_err", "check", "screen",
    # recording
    "RESULTS_DIR", "CKPT_ROOT", "record", "speedup_row", "paired_row",
    "ckpt_load", "ckpt_save", "ckpt_dir",
    # environment / provenance
    "preflight", "preflight_ok", "features", "harness_info", "banner", "main_guard",
    "timer_tick_ms", "ticks", "tick_note", "flush_bytes",
    "calib_bandwidth_gbs", "calib_gemm_tfs", "calib_launch_ms",
]

# --------------------------------------------------------------------------------------
# Where things go.
#
# `GLM52_H200_RESULTS_DIR` keeps this port's results from overwriting another platform's
# (C500 -> results/c500, RTX 4060 -> results/rtx4060, this port -> results/h200).
# Checkpoints hang off the SAME root, so isolating a run's outputs also isolates its
# inputs -- otherwise `--results-dir` gives you a fresh output file that silently
# republishes another run's cached timings.
# --------------------------------------------------------------------------------------
RESULTS_DIR = Path(
    os.environ.get(
        "GLM52_H200_RESULTS_DIR", HERE.parent / "results" / "h200"
    )
)
CKPT_ROOT = RESULTS_DIR / "_ckpt"

# preflight.py writes this; every calibrated ceiling and every H200 feature decision reads
# it.  Absent is a supported state -- we then degrade to the classic path everywhere.
PREFLIGHT_PATH = Path(
    os.environ.get("GLM52_H200_PREFLIGHT", HERE / "preflight_h200.json")
)

# The floor for the L2 flush buffer.  Not a hardware constant: the real size is
# max(this, 4 * measured L2), and this only keeps small-L2 devices from flushing with a
# buffer so small the kernel launch dominates.
MIN_FLUSH_BYTES = 256 * 2**20


# --------------------------------------------------------------------------------------
# Preflight JSON: consumed when present, degraded sanely when absent
# --------------------------------------------------------------------------------------
_PREFLIGHT_CACHE: dict | None = None
_PREFLIGHT_WARNED: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _PREFLIGHT_WARNED:
        _PREFLIGHT_WARNED.add(key)
        print(f"[harness] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# Machine-state capture lives in glm52_h200/hwinfo.py (clocks, power, throttle reasons,
# ECC/MIG).  It is imported lazily and defensively: this harness is what produces the
# numbers, and it must keep working if that module is mid-edit, absent, or on a box with no
# nvidia-smi.  Nothing here fails because machine state could not be read -- the failure is
# recorded in the payload instead, which is the whole point of recording it.
# --------------------------------------------------------------------------------------
_HWMOD: object = "unset"


def _hwmod():
    global _HWMOD
    if _HWMOD == "unset":
        m = None
        try:
            from . import hwinfo as m  # noqa: PLC0415 -- lazy on purpose
        except Exception:  # noqa: BLE001 -- absent, mid-edit, or run outside the package
            try:
                import importlib

                m = importlib.import_module("glm52_h200.hwinfo")
            except Exception as exc:  # noqa: BLE001
                m = None
                _warn_once(
                    "hwmod",
                    f"glm52_h200.hwinfo not importable ({exc!r}); machine-state capture "
                    f"(clocks / throttle / thermal drift evidence) is disabled",
                )
        _HWMOD = m
    return _HWMOD


def _hw_call(name: str, *args, **kw) -> dict:
    """Call `hwinfo.<name>(...)`, returning an `{"available": False, ...}` marker instead of
    raising.  Absent evidence must look absent in the payload, never like a clean machine."""
    m = _hwmod()
    if m is None:
        return {"available": False, "reason": "glm52_h200.hwinfo not importable"}
    fn = getattr(m, name, None)
    if fn is None:
        return {"available": False, "reason": f"hwinfo.{name} missing"}
    try:
        return fn(*args, **kw)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"[:200]}


def _live_device_name() -> str:
    """Device name, or "" if CUDA is not up.  Never raises: this is called from `record`
    and from the checkpoint fence, and neither may take down a finished run."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 -- reported through `_meta`, not raised
        pass
    return ""


def preflight() -> dict:
    """Load (and cache) the preflight JSON.

    Returns `{}` when the file is missing or unreadable -- that is a supported state, not
    an error: the suite then takes the classic path everywhere and records that it did.

    The blob is tagged with `_status`:
      "ok"      device name in the JSON matches the live device
      "stale"   it does not -- the JSON describes some other GPU
      "nodev"   CUDA is not up, so we cannot tell
      "absent"  no file
    Only "ok" unlocks calibration numbers and H200 feature paths.  This is the same fence
    as the checkpoint one, for the same reason: a preflight from another box supplies
    plausible ceilings and feature flags for hardware that is not underneath us.
    """
    global _PREFLIGHT_CACHE
    if _PREFLIGHT_CACHE is not None:
        return _PREFLIGHT_CACHE
    blob: dict = {}
    if PREFLIGHT_PATH.exists():
        try:
            blob = json.loads(PREFLIGHT_PATH.read_text())
        except Exception as exc:  # noqa: BLE001
            _warn_once("pf_read", f"preflight {PREFLIGHT_PATH} unreadable: {exc!r}")
            blob = {}
    if not blob:
        blob = {"_status": "absent", "_path": str(PREFLIGHT_PATH)}
        _warn_once(
            "pf_absent",
            f"no preflight at {PREFLIGHT_PATH} -- calibrated ceilings and H200 feature "
            f"paths are DISABLED (classic fallbacks only). Run preflight.py first.",
        )
    else:
        live = _live_device_name()
        pf_dev = (blob.get("device") or {}).get("name", "")
        if not live:
            blob["_status"] = "nodev"
        elif pf_dev and pf_dev != live:
            blob["_status"] = "stale"
            _warn_once(
                "pf_stale",
                f"preflight describes {pf_dev!r} but this box is {live!r} -- ignoring its "
                f"calibration and feature flags (classic fallbacks only).",
            )
        else:
            blob["_status"] = "ok"
        blob["_path"] = str(PREFLIGHT_PATH)
    _PREFLIGHT_CACHE = blob
    return blob


def preflight_ok() -> bool:
    return preflight().get("_status") == "ok"


def _pf_get(*path, default=None):
    """Fetch `blob[path[0]][path[1]]...` only if the preflight matched this device."""
    if not preflight_ok():
        return default
    cur = preflight()
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def calib_bandwidth_gbs() -> float:
    """Measured HBM bandwidth (GB/s) -- the denominator of every memory roofline in this
    study.  NaN when unknown, so a ceiling computed from it is visibly NaN rather than
    silently vendor-spec."""
    bw = _pf_get("calibration", "bandwidth", default={}) or {}
    vals = [v for k, v in bw.items() if k.startswith("copy_") and isinstance(v, (int, float))]
    if not vals:
        vals = [v for v in bw.values() if isinstance(v, (int, float))]
    return max(vals) if vals else float("nan")


def calib_gemm_tfs() -> float:
    """Best measured bf16 GEMM throughput (TF/s), Triton or cuBLAS, whichever is higher."""
    g = _pf_get("calibration", "gemm", default={}) or {}
    vals = [v for k, v in g.items() if k.endswith("_TFs") and isinstance(v, (int, float))]
    return max(vals) if vals else float("nan")


def calib_launch_ms() -> float:
    """Measured cost of one kernel launch, in ms.  At decode_bs1 this is most of the
    number; a bench that cannot quote it cannot explain its own result."""
    us = _pf_get("calibration", "launch_us")
    return us / 1000.0 if isinstance(us, (int, float)) else float("nan")


# --------------------------------------------------------------------------------------
# CUDA-event timer granularity
# --------------------------------------------------------------------------------------
_TICK_MS: float | None = None


def _measure_timer_tick() -> float:
    """Fallback tick probe, used only when preflight is missing/stale.

    Same method as preflight.py, including its fix: the true granularity is the LARGEST
    quantum that divides essentially every sample (0.256 us trivially divides anything
    1.024 us divides, so scanning upward and keeping the first hit always reports the
    finest candidate and understates the quantisation).
    """
    if not torch.cuda.is_available():
        return float("nan")
    buf = torch.empty(1024, device="cuda", dtype=torch.float32)
    vals = []
    for _ in range(220):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        buf.add_(1.0)
        e.record()
        torch.cuda.synchronize()
        vals.append(s.elapsed_time(e))
    vals = vals[20:]  # drop the first launches: they include one-off setup
    cands = {}
    for q_us in (0.256, 0.512, 1.024, 2.048, 4.096):
        hit = sum(
            1 for v in vals if abs((v * 1000.0) / q_us - round((v * 1000.0) / q_us)) < 1e-3
        )
        cands[q_us] = hit / len(vals)
    ok = [q for q, f in cands.items() if f >= 0.98]
    q = max(ok) if ok else float("nan")
    del buf
    return q / 1000.0 if not math.isnan(q) else float("nan")


def timer_tick_ms() -> float:
    """CUDA-event timer quantum in ms.  Preflight first, self-measured second, NaN last.

    NaN is deliberate: this number only ever *annotates* a result (see `tick_note`), so a
    wrong-but-plausible value would be worse than an honest gap.  It is never used to
    build a grid or scale a measurement.
    """
    global _TICK_MS
    if _TICK_MS is None:
        us = _pf_get("calibration", "timer_tick_us")
        if isinstance(us, (int, float)) and us > 0:
            _TICK_MS = float(us) / 1000.0
        else:
            try:
                _TICK_MS = _measure_timer_tick()
            except Exception as exc:  # noqa: BLE001
                _warn_once("tick", f"timer-tick probe failed: {exc!r}")
                _TICK_MS = float("nan")
    return _TICK_MS


def ticks(ms: float) -> float:
    """How many CUDA-event ticks a duration is worth.

    On the 4060 the tick was 1.024 us and f03's decode_bs1 arms measured 12.0 and 16.0
    ticks -- integers, so that "1.455x" carried +-8% of pure quantisation.  Benches quote
    `ticks()` next to any decode speedup so a reader can see when the ratio is a ratio of
    small integers.
    """
    t = timer_tick_ms()
    if math.isnan(t) or t <= 0:
        return float("nan")
    return ms / t


def tick_note(fused_ms: float, unfused_ms: float, min_gap_ticks: float = 3.0) -> dict:
    """Quantisation annotation for one fused/unfused pair.

    `flagged` is True when the two operands are within `min_gap_ticks` of each other or
    when either arm is under ~20 ticks, i.e. when the ratio's leading digits are an
    artifact of the timer rather than of the kernels.
    """
    ft, ut = ticks(fused_ms), ticks(unfused_ms)
    if math.isnan(ft) or math.isnan(ut) or ft <= 0 or ut <= 0:
        return {"tick_ms": timer_tick_ms(), "flagged": None, "note": "tick unknown"}
    gap = abs(ut - ft)
    # +-1 tick on each operand propagates to roughly this much relative error in the ratio.
    quant = 1.0 / ft + 1.0 / ut
    flagged = bool(gap < min_gap_ticks or min(ft, ut) < 20.0)
    return {
        "tick_ms": timer_tick_ms(),
        "fused_ticks": ft,
        "unfused_ticks": ut,
        "gap_ticks": gap,
        "ratio_quantisation_frac": quant,
        "flagged": flagged,
        "note": (
            "operands within a few timer ticks -- quote as launch-dominated, not as a "
            "precise speedup" if flagged else "well above timer granularity"
        ),
    }


# --------------------------------------------------------------------------------------
# H200 feature availability -- decided at RUNTIME, never at authoring time
# --------------------------------------------------------------------------------------
_FEATURES: dict | None = None


def features() -> dict:
    """Which Hopper-only Triton features are proven to COMPILE AND LAUNCH here.

    Sourced from preflight's compile+launch probes, not from attribute sniffing: several
    Triton releases export `make_tensor_descriptor` and then raise `CompilationError` on
    first use (observed on sm_89 with Triton 3.6).

    Every flag is False unless a matching preflight proved otherwise, so the default on an
    unprobed box is the classic path.  `GLM52_H200_DISABLE_FEATURES=tma,clusters` can turn
    flags OFF (to A/B a path, or to work around a driver bug); there is deliberately no
    way to turn one ON, because the only evidence that would justify it is a probe.
    """
    global _FEATURES
    if _FEATURES is not None:
        return _FEATURES
    probes = _pf_get("triton_features", "compile_probes", default={}) or {}

    def ok(key: str) -> bool:
        v = probes.get(key)
        return bool(isinstance(v, dict) and v.get("ok"))

    cc = (0, 0)
    try:
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
    except Exception:  # noqa: BLE001
        pass
    sm90 = cc >= (9, 0)
    f = {
        "source": preflight().get("_status", "absent"),
        "cc": f"{cc[0]}.{cc[1]}",
        "sm90": sm90,
        # TMA and clusters are architecturally impossible below sm_90; AND-ing with the
        # live capability means a name-matching but otherwise wrong JSON still cannot
        # enable a path the silicon does not have.
        "tma": ok("tma_tensor_descriptor") and sm90,
        "clusters": ok("thread_block_cluster_num_ctas") and sm90,
        "warp_specialize": ok("warp_specialize_tl_range"),
        "num_consumer_groups": ok("warp_specialize_num_consumer_groups"),
        "dot_bf16": ok("tl_dot_bf16"),
        "disabled": [],
    }
    off = [s.strip() for s in os.environ.get("GLM52_H200_DISABLE_FEATURES", "").split(",")]
    for name in off:
        if name and name in f and isinstance(f[name], bool):
            f[name] = False
            f["disabled"].append(name)
    _FEATURES = f
    return f


# --------------------------------------------------------------------------------------
# L2 flush -- derived from the device, never assumed
# --------------------------------------------------------------------------------------
_FLUSH_BYTES: int | None = None
_FLUSH_SOURCE = ""
_flush_buf: torch.Tensor | None = None


def _l2_bytes() -> int:
    """Measured L2 size.  torch first (no JIT, always available once CUDA is up), then a
    preflight that PROVABLY describes this device, then give up.

    The preflight fallback goes through `_pf_get`, so a JSON written on another box is not
    consulted: a 32 MB Ada L2 standing in for a 50 MB Hopper one would size the flush at
    128 MB and leave a third of L2 warm, and nothing in the output would show it.
    """
    try:
        n = int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0) or 0)
        if n > 0:
            return n
    except Exception:  # noqa: BLE001 -- fall through to the JSON
        pass
    return int(_pf_get("device", "L2_cache_size", default=0) or 0)


def flush_bytes() -> int:
    """Size of the L2-eviction buffer: `max(256 MB, 4 * L2)`, derived from this device.

    C500's L2 is 8 MB, Ada's 32 MB, H200's ~50 MB.  A buffer sized for the smallest of
    those leaves the largest one warm, and a warm L2 flatters whichever arm re-reads an
    intermediate -- which is exactly the quantity a fusion study is trying to measure.  If
    L2 cannot be determined we raise: there is no safe default, and the failure mode of
    guessing is a silently warm measurement that looks perfectly normal.
    """
    global _FLUSH_BYTES, _FLUSH_SOURCE
    if _FLUSH_BYTES is None:
        override = os.environ.get("GLM52_H200_FLUSH_MB")
        if override:
            _FLUSH_BYTES = int(override) * 2**20
            _FLUSH_SOURCE = f"GLM52_H200_FLUSH_MB={override}"
            _warn_once("flush_env", f"L2 flush buffer forced to {override} MB by env")
        else:
            l2 = _l2_bytes()
            if l2 <= 0:
                raise RuntimeError(
                    "cannot determine L2 size -- refusing to flush with a guessed buffer.\n"
                    "  An undersized flush turns every 'cold' measurement into a warm-cache "
                    "one and leaves no trace in the result file.\n"
                    "  Fix the device probe, or set GLM52_H200_FLUSH_MB explicitly (it is "
                    "recorded in _meta so the choice stays auditable)."
                )
            _FLUSH_BYTES = max(MIN_FLUSH_BYTES, 4 * l2)
            _FLUSH_SOURCE = f"max(256MB, 4 x L2={l2 >> 20}MB)"
    return _FLUSH_BYTES


def _flush_l2() -> None:
    """Evict L2 so the next timed region starts cold.  Called BETWEEN timed regions,
    never inside one."""
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.empty(flush_bytes() // 4, dtype=torch.int32, device="cuda")
    _flush_buf.zero_()


# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------
def _pct(sorted_vals: Sequence[float], q: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    return sorted_vals[min(n - 1, max(0, int(q * n)))]


def _trimmed_mean(vals: Sequence[float], trim: float = 0.1) -> float:
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return float("nan")
    k = int(n * trim)
    core = v[k : n - k] or v
    return statistics.fmean(core)


@dataclass
class Timing:
    p50_ms: float
    p10_ms: float
    p90_ms: float
    mean_ms: float
    n: int
    noflush_p50_ms: float = float("nan")
    min_ms: float = float("nan")
    # Monotone drift over the measurement window, as a fraction of the opening median:
    # (median of last third - median of first third) / median of first third.  A large
    # value is the signature that killed the 4060's f01 number; the PAIRED statistic in
    # `bench_pair` is immune to it, a standalone `bench_chain` is not.
    drift_frac: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_samples(vals: Sequence[float], noflush_p50: float = float("nan")) -> "Timing":
        s = sorted(vals)
        n = len(s)
        if n == 0:
            return Timing(float("nan"), float("nan"), float("nan"), float("nan"), 0)
        third = max(1, n // 3)
        head = statistics.median(vals[:third])
        tail = statistics.median(vals[-third:])
        return Timing(
            p50_ms=_pct(s, 0.5),
            p10_ms=_pct(s, 0.1),
            p90_ms=_pct(s, 0.9),
            mean_ms=statistics.fmean(s),
            n=n,
            noflush_p50_ms=noflush_p50,
            min_ms=s[0],
            drift_frac=(tail - head) / head if head > 0 else float("nan"),
        )


def _as_chain(fns) -> list:
    """Accept a single callable or a sequence of them; a bench that passes one kernel
    should not have to wrap it."""
    if callable(fns):
        return [fns]
    return list(fns)


def _run(fns: Sequence[Callable[[], object]]) -> None:
    for fn in fns:
        fn()


def bench_chain(
    fns: Sequence[Callable[[], object]],
    warmup: int = 25,
    rep: int = 100,
    flush: bool = True,
    noflush: bool = True,
) -> Timing:
    """Time `fns` executed back-to-back as ONE logical operation.

    One L2 flush before each timed repetition of the whole chain -- never between the
    kernels inside it (rule 1).  That is precisely what makes an unfused chain a fair
    baseline: in the real layer the producer's output is still in L2 when the consumer
    starts, so flushing between them would hand the fused arm a win it does not have.

    Returns median / p10 / p90 / mean over `rep` repetitions plus a no-flush median (set
    `noflush=False` to skip that second pass and halve the cost).  Use this for TUNING and
    for per-kernel breakdowns; use `bench_pair` for the headline fused-vs-unfused ratio,
    because a chain timed here and a chain timed ten minutes later are not comparable to
    four significant figures on a thermally live GPU.
    """
    fns = _as_chain(fns)
    if rep < 1:
        raise ValueError("rep must be >= 1")

    for _ in range(warmup):
        _run(fns)
    torch.cuda.synchronize()

    def _measure(do_flush: bool) -> list[float]:
        start = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        for i in range(rep):
            if do_flush:
                _flush_l2()
            start[i].record()
            _run(fns)
            end[i].record()
        torch.cuda.synchronize()
        return [s.elapsed_time(e) for s, e in zip(start, end)]

    times = _measure(flush)
    nf_p50 = float("nan")
    if noflush:
        try:
            nf = sorted(_measure(False))
            nf_p50 = _pct(nf, 0.5)
        except Exception:  # noqa: BLE001 -- a reference number, not the measurement
            pass
    return Timing.from_samples(times, nf_p50)


# --------------------------------------------------------------------------------------
# Paired (interleaved) timing -- the fix for the 4060's thermal-drift artifact
# --------------------------------------------------------------------------------------
@dataclass
class PairTiming:
    """Result of one interleaved fused/unfused measurement.

    `ratio_p50` is the study's headline number for the regime.  It is the median of the
    PER-REPETITION ratios, not the ratio of the two medians -- those differ exactly when
    the machine is drifting, which is the case this class exists to survive.
    """

    fused: Timing
    unfused: Timing
    ratio_p50: float
    ratio_p10: float
    ratio_p90: float
    ratio_trimmed: float
    ratio_of_medians: float  # what the old sequential protocol would have reported
    n: int
    frac_fused_faster: float
    ratio_p50_first_half: float
    ratio_p50_second_half: float
    ratio_p50_fused_first: float
    ratio_p50_unfused_first: float
    order_gap_frac: float
    drift_frac_fused: float
    drift_frac_unfused: float
    noflush_ratio_p50: float = float("nan")
    n_discarded: int = 0
    tick: dict = field(default_factory=dict)
    # Machine state bracketing the loop (hwinfo.compare_snapshots): interleaving makes
    # drift cancel, and this is the evidence saying whether there was any to cancel.
    machine: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["fused"] = self.fused.as_dict()
        d["unfused"] = self.unfused.as_dict()
        return d


def bench_pair(
    fused_fns: Sequence[Callable[[], object]],
    unfused_fns: Sequence[Callable[[], object]],
    warmup: int = 25,
    rep: int = 100,
    flush: bool = True,
    noflush: bool = False,
    machine_check: bool = True,
    machine_detail: bool = False,
) -> PairTiming:
    """Time both arms INTERLEAVED inside one loop and return the PAIRED ratio.

    WHY THIS EXISTS.  The 4060 campaign timed the entire fused arm and then the entire
    unfused arm (`bench_f01:408-409`).  Within that single run the GPU drifted 22%
    thermally -- the coarse sweep measured the fused chain at 137.25 ms and the final
    measurement of the same chain at 167.20 ms -- and because the two arms occupied
    different parts of the thermal ramp, fusion #1's prefill_t8192 came out at 1.027,
    which is ABOVE that cell's own physical ceiling.  Locked clocks are a cap, not a
    floor.  Re-measured with the arms interleaved A/B/A/B at n=120, the same pair gave
    1.0143 (trimmed 1.0145), and a per-primitive decomposition independently implied
    1.0151.  The interleaved number was right and the sequential one was not.

    HOW.  Each repetition runs BOTH arms, each preceded by its own L2 flush (so both start
    cold, rule 1 still holding *within* an arm), and the leading arm alternates every
    repetition.  Alternating gives an A-B-B-A pattern across repetition pairs, so a linear
    drift cancels to first order in the paired ratio; taking the median over per-repetition
    ratios then also survives non-linear drift, since any monotone rescaling of the whole
    window multiplies both operands of a given repetition by nearly the same factor.

    WHAT COMES BACK.  `ratio_p50` (headline), `ratio_p10` / `ratio_p90` (spread of the
    per-rep ratio, i.e. the real error bar), `ratio_trimmed` (10% trimmed mean),
    `ratio_of_medians` (what the old sequential protocol would have said -- publish the
    two side by side when they disagree), each arm's own `Timing`, plus the diagnostics
    that say whether the protocol worked: drift per arm, first-half vs second-half ratio,
    fused-first vs unfused-first ratio, the fraction of repetitions in which the fused arm
    actually won, and a timer-tick annotation.

    `machine_check` brackets the loop with two `hwinfo.snapshot()` reads (two short
    nvidia-smi queries, milliseconds against a measurement of seconds) so the result file
    can SHOW that the clocks and temperature held rather than assume it.  Interleaving is
    what makes drift cancel; this is the evidence about how much there was to cancel.  Set
    `machine_detail=True` to keep both raw snapshots and not just the comparison.
    """
    f_fns, u_fns = _as_chain(fused_fns), _as_chain(unfused_fns)
    if rep < 2:
        raise ValueError("rep must be >= 2 for a paired measurement")
    snap_before = _hw_call("snapshot") if machine_check else {"available": False,
                                                             "reason": "not requested"}

    # Warm both arms in the same alternating order the measurement uses, so neither pays a
    # first-touch/JIT cost inside the timed window.
    for i in range(warmup):
        if i % 2 == 0:
            _run(f_fns)
            _run(u_fns)
        else:
            _run(u_fns)
            _run(f_fns)
    torch.cuda.synchronize()

    def _measure(do_flush: bool) -> tuple[list[float], list[float]]:
        fs = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        fe = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        us = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        ue = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        for i in range(rep):
            seq = [(fs[i], fe[i], f_fns), (us[i], ue[i], u_fns)]
            if i % 2:  # alternate which arm leads: A B | B A | A B | ...
                seq.reverse()
            for s, e, fns in seq:
                if do_flush:
                    _flush_l2()
                s.record()
                _run(fns)
                e.record()
        torch.cuda.synchronize()
        return (
            [s.elapsed_time(e) for s, e in zip(fs, fe)],
            [s.elapsed_time(e) for s, e in zip(us, ue)],
        )

    ft, ut = _measure(flush)

    nf_ratio = float("nan")
    nf_f = nf_u = float("nan")
    if noflush:
        try:
            nft, nut = _measure(False)
            r = [u / f for f, u in zip(nft, nut) if f > 0]
            nf_ratio = _pct(sorted(r), 0.5) if r else float("nan")
            nf_f = _pct(sorted(nft), 0.5)
            nf_u = _pct(sorted(nut), 0.5)
        except Exception:  # noqa: BLE001 -- reference number only
            pass

    # Per-repetition ratios. A non-positive sample means the event pair collapsed (the
    # work was shorter than the timer could resolve); drop it and say how many.
    pairs = [(i, f, u) for i, (f, u) in enumerate(zip(ft, ut)) if f > 0 and u > 0]
    ratios = [u / f for _, f, u in pairs]
    n_disc = rep - len(pairs)
    if not ratios:
        raise RuntimeError(
            "every paired sample was non-positive -- the timer cannot resolve this work; "
            "raise `rep`, or batch the chain, and re-run"
        )

    sr = sorted(ratios)
    half = len(ratios) // 2
    even = [r for (i, _, _), r in zip(pairs, ratios) if i % 2 == 0]  # fused ran first
    odd = [r for (i, _, _), r in zip(pairs, ratios) if i % 2 == 1]  # unfused ran first
    r_even = _pct(sorted(even), 0.5) if even else float("nan")
    r_odd = _pct(sorted(odd), 0.5) if odd else float("nan")
    denom = (
        (r_even + r_odd) / 2
        if not (math.isnan(r_even) or math.isnan(r_odd))
        else float("nan")
    )

    tf = Timing.from_samples(ft, nf_f)
    tu = Timing.from_samples(ut, nf_u)

    machine: dict = {"available": False, "reason": "not requested"}
    if machine_check:
        snap_after = _hw_call("snapshot")
        cmp_ = _hw_call("compare_snapshots", snap_before, snap_after)
        machine = {"compare": cmp_}
        if machine_detail:
            machine["before"], machine["after"] = snap_before, snap_after
        # Say it on stdout too: the operator reads the log, and "the machine moved" is
        # something to know while the run is still going, not after the report is written.
        if isinstance(cmp_, dict) and cmp_.get("suspect"):
            print(
                f"    !! machine moved during this measurement: "
                f"{'; '.join(cmp_.get('reasons') or [])} -- the PAIRED ratio absorbs it, "
                f"but the absolute ms are not comparable across regimes",
                flush=True,
            )
    return PairTiming(
        fused=tf,
        unfused=tu,
        ratio_p50=_pct(sr, 0.5),
        ratio_p10=_pct(sr, 0.1),
        ratio_p90=_pct(sr, 0.9),
        ratio_trimmed=_trimmed_mean(ratios, 0.1),
        ratio_of_medians=tu.p50_ms / tf.p50_ms if tf.p50_ms > 0 else float("nan"),
        n=len(ratios),
        frac_fused_faster=sum(1 for _, f, u in pairs if f < u) / len(pairs),
        ratio_p50_first_half=_pct(sorted(ratios[:half]), 0.5) if half else float("nan"),
        ratio_p50_second_half=_pct(sorted(ratios[half:]), 0.5) if half else float("nan"),
        ratio_p50_fused_first=r_even,
        ratio_p50_unfused_first=r_odd,
        order_gap_frac=(
            abs(r_even - r_odd) / denom
            if not math.isnan(denom) and denom > 0
            else float("nan")
        ),
        drift_frac_fused=tf.drift_frac,
        drift_frac_unfused=tu.drift_frac,
        noflush_ratio_p50=nf_ratio,
        n_discarded=n_disc,
        tick=tick_note(tf.p50_ms, tu.p50_ms),
        machine=machine,
    )


# --------------------------------------------------------------------------------------
# Autotuning
# --------------------------------------------------------------------------------------
@dataclass
class TuneResult:
    best_cfg: dict
    best_ms: float
    n_tried: int  # configs actually timed or attempted (coarse + refine, post-guard)
    n_failed: int  # of those, how many failed to compile/launch
    table: list = field(default_factory=list)  # [(cfg, ms|None, err|None), ...]
    n_offered: int = 0  # configs presented, before the guard pruned any
    n_guard_rejected: int = 0
    n_coarse: int = 0
    n_refine: int = 0
    best_stage: str = "coarse"
    failure_summary: dict = field(default_factory=dict)  # error class -> count

    def as_dict(self) -> dict:
        return {
            "best_cfg": self.best_cfg,
            "best_ms": self.best_ms,
            "n_tried": self.n_tried,
            "n_failed": self.n_failed,
            "n_offered": self.n_offered,
            "n_guard_rejected": self.n_guard_rejected,
            "n_coarse": self.n_coarse,
            "n_refine": self.n_refine,
            "best_stage": self.best_stage,
            "failure_summary": self.failure_summary,
            "table": self.table,
        }

    def topk(self, k: int) -> list[dict]:
        """The k fastest configs that actually ran. Benches use this to seed a joint
        (chain-level) refine stage."""
        rows = [(ms, cfg) for cfg, ms, err in self.table if ms is not None]
        rows.sort(key=lambda t: t[0])
        return [cfg for _, cfg in rows[:k]]


def _cfg_key(cfg):
    """A stable dedup key for one grid element.

    Not every grid element is a dict. `bench_f01`'s joint stage tunes the GEMM and its
    epilogue together and therefore searches over `(gemm_cfg, epi_cfg)` TUPLES; the layer
    bench searches over tuples of per-site configs. Assuming `.items()` turned that into an
    AttributeError that aborted the whole regime -- so recurse over containers instead, and
    fall back to `repr` for anything else.
    """
    if isinstance(cfg, dict):
        return tuple(sorted((k, _cfg_key(v)) for k, v in cfg.items()))
    if isinstance(cfg, (tuple, list)):
        return tuple(_cfg_key(v) for v in cfg)
    return repr(cfg)


def neighbours(best: dict, grid: Sequence[dict], max_out: int = 96) -> list[dict]:
    """Default refine stage: the winner's neighbourhood ON THE GRID'S OWN VALUE LATTICE.

    For every key that varies across `grid` we collect the offered values, and move the
    winner one index up and one index down in that sorted list (categorical keys -- None,
    bools, strings -- take all offered values instead).  We emit every single-key move,
    then two-key moves over the three widest keys, capped at `max_out`.

    Values are never invented.  If the coarse grid never offered `BLOCK_N=16384`, refine
    will not try it: the bench's own validity model (SMEM, register pressure, tile-shape
    legality) was written against the values it offered, and a config outside that set is
    one this harness has no basis to believe is legal.  What refine *can* do is reach
    COMBINATIONS a sparse coarse grid skipped.  Against an already-exhaustive coarse grid
    it is a no-op, which is the correct behaviour -- and `n_refine` records that it was.
    """
    if not best or not grid:
        return []
    if not isinstance(best, dict) or not all(isinstance(c, dict) for c in grid):
        # Composite grids (f01's (gemm_cfg, epi_cfg) pairs, the layer bench's per-site
        # tuples) have no single value lattice to step along. Refining them is not
        # meaningful here, and returning [] is honest: `n_refine == 0` in the result JSON
        # then records that only the coarse stage ran, rather than a crash losing the regime.
        return []
    vals: dict[str, list] = {}
    for cfg in grid:
        for k, v in cfg.items():
            vals.setdefault(k, [])
            if not any(v == e and type(v) is type(e) for e in vals[k]):
                vals[k].append(v)

    moves: dict[str, list] = {}
    for k, vs in vals.items():
        if len(vs) < 2 or k not in best:
            continue  # constant across the grid: not a tuning knob
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vs)
        if numeric:
            s = sorted(vs)
            try:
                i = s.index(best[k])
            except ValueError:
                continue
            moves[k] = [s[j] for j in (i - 1, i + 1) if 0 <= j < len(s)]
        else:
            moves[k] = [v for v in vs if v != best[k]]
    if not moves:
        return []

    out, seen = [], {_cfg_key(best)}

    def _emit(cfg: dict) -> None:
        key = _cfg_key(cfg)
        if key not in seen and len(out) < max_out:
            seen.add(key)
            out.append(cfg)

    for k in sorted(moves):  # deterministic order: the same grid always refines the same
        for v in moves[k]:
            _emit(dict(best, **{k: v}))
    wide = sorted(moves, key=lambda k: (-len(vals[k]), k))[:3]
    for a in wide:
        for b in wide:
            if a >= b:
                continue
            for va in moves[a]:
                for vb in moves[b]:
                    _emit(dict(best, **{a: va, b: vb}))
    return out


def autotune(
    build_fn: Callable[[dict], Sequence[Callable[[], object]]] | None = None,
    grid: Iterable[dict] | None = None,
    warmup: int = 10,
    rep: int = 30,
    guard: Callable[[dict], object] | None = None,
    refine: Callable[[dict], Sequence[dict]] | bool | None = None,
    refine_max: int = 96,
    verbose: bool = False,
    tag: str = "",
    make_chain: Callable[[dict], Sequence[Callable[[], object]]] | None = None,
    configs: Iterable[dict] | None = None,
) -> TuneResult:
    """Two-stage search: coarse over `grid`, then refine around that stage's winner.

    `build_fn(cfg)` returns the callables to time for that config -- one entry for a fused
    kernel, several for an unfused chain.  Each arm of a fusion pair calls this separately
    with its own grid and keeps its own winner; nothing is ever shared (rule 2).

    `guard(cfg)` -> bool, or (bool, reason).  Guard rejections are counted SEPARATELY from
    compile failures, because they mean different things: a guard rejection is a deliberate
    prune (and every hardware constant it uses must come from the cached device probe in
    config.py -- a guard built from another device's numbers prunes the two arms unequally
    and manufactures or destroys the win), while a compile failure is the hardware
    answering back.

    `refine`: None -> the generic `neighbours()` lattice walk; a callable -> the bench's
    own `refine_grid(best)`; False -> single stage.  Both arms must use the same setting.

    A config that fails to compile (SMEM overflow, illegal tile shape, register limit,
    unsupported feature on this arch) is RECORDED and skipped, never fatal -- on a Hopper
    grid a good fraction of tiles legitimately fail somewhere, and losing the run to the
    first one costs a whole round trip on hardware nobody here can test on.

    Every offered config lands in `table` as `(cfg, ms | None, err | None)`, and
    `n_offered` / `n_tried` / `n_failed` / `n_guard_rejected` / `n_coarse` / `n_refine` are
    recorded per call.  Those counters are the only way an unfair comparison -- one arm
    searched 120 configs, the other 12 -- is detectable after the fact (rule 7).
    """
    # Legacy aliases: the audited benches call `autotune(make_chain, configs, ...)`.
    build_fn = build_fn if build_fn is not None else make_chain
    grid = grid if grid is not None else configs
    if build_fn is None or grid is None:
        raise TypeError("autotune requires build_fn and grid")

    table: list = []
    counters = {"tried": 0, "failed": 0, "guard": 0, "offered": 0}
    timed: dict = {}  # cfg key -> ms, so refine never re-times a coarse config
    fails: dict = {}

    def _stage(cfgs: Sequence[dict], stage: str) -> tuple[dict | None, float, int]:
        best_ms, best_cfg, n = float("inf"), None, 0
        for cfg in cfgs:
            counters["offered"] += 1
            key = _cfg_key(cfg)
            if key in timed:  # already measured in an earlier stage
                if timed[key] is not None and timed[key] < best_ms:
                    best_ms, best_cfg = timed[key], cfg
                continue
            if guard is not None:
                try:
                    g = guard(cfg)
                except Exception as exc:  # noqa: BLE001 -- a broken guard is a rejection
                    g = (False, f"guard raised {type(exc).__name__}: {exc}"[:120])
                ok, why = g if isinstance(g, tuple) else (bool(g), "")
                if not ok:
                    counters["guard"] += 1
                    timed[key] = None
                    table.append((cfg, None, f"guard: {why or 'rejected'}"))
                    continue
            n += 1
            counters["tried"] += 1
            try:
                t = bench_chain(build_fn(cfg), warmup=warmup, rep=rep, flush=True,
                                noflush=False)
                timed[key] = t.p50_ms
                table.append((cfg, t.p50_ms, None))
                if t.p50_ms < best_ms:
                    best_ms, best_cfg = t.p50_ms, cfg
                if verbose:
                    print(f"  [{stage}] {cfg} -> {t.p50_ms:.4f} ms", flush=True)
            except Exception as exc:  # noqa: BLE001 - deliberate: keep searching
                counters["failed"] += 1
                timed[key] = None
                msg = f"{type(exc).__name__}: {exc}"
                fails[type(exc).__name__] = fails.get(type(exc).__name__, 0) + 1
                table.append((cfg, None, msg[:300]))
                if verbose:
                    print(f"  [{stage}] {cfg} -> FAIL {type(exc).__name__}", flush=True)
            finally:
                # Failed compiles can strand large allocations; a 1000-config sweep that
                # never releases them dies of fragmentation halfway through.
                torch.cuda.empty_cache()
        return best_cfg, best_ms, n

    coarse = list(grid)
    best_cfg, best_ms, n_coarse = _stage(coarse, "coarse")
    if best_cfg is None:
        raise RuntimeError(
            f"every one of {len(coarse)} configs failed{' for ' + tag if tag else ''} "
            f"({counters['failed']} compile/launch, {counters['guard']} guard). "
            f"First few: " + "; ".join(str(r[2])[:90] for r in table[:3])
        )

    n_refine, stage = 0, "coarse"
    if refine is not False:
        try:
            rg = refine(best_cfg) if callable(refine) else neighbours(
                best_cfg, coarse, max_out=refine_max
            )
        except Exception as exc:  # noqa: BLE001 -- a bad refine must not lose the coarse win
            print(f"  [refine] generator failed ({exc!r}); keeping the coarse winner",
                  flush=True)
            rg = []
        if rg:
            r_cfg, r_ms, n_refine = _stage(list(rg), "refine")
            if r_cfg is not None and r_ms < best_ms:
                best_cfg, best_ms, stage = r_cfg, r_ms, "refine"

    if verbose or tag:
        print(
            f"    [{tag or 'tune':<12}] coarse {n_coarse} + refine {n_refine} cfgs "
            f"({counters['failed']} fail, {counters['guard']} guarded) -> "
            f"{best_ms:.4f} ms {best_cfg} [{stage}]",
            flush=True,
        )
    return TuneResult(
        best_cfg=best_cfg,
        best_ms=best_ms,
        n_tried=counters["tried"],
        n_failed=counters["failed"],
        table=table,
        n_offered=counters["offered"],
        n_guard_rejected=counters["guard"],
        n_coarse=n_coarse,
        n_refine=n_refine,
        best_stage=stage,
        failure_summary=fails,
    )


# --------------------------------------------------------------------------------------
# Correctness
# --------------------------------------------------------------------------------------
def rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Max-abs relative error against the reference's dynamic range.

    Shape mismatch raises rather than broadcasting: a silent broadcast compares a kernel
    against a *different* quantity and reports a small error for it.
    """
    if tuple(got.shape) != tuple(ref.shape):
        raise ValueError(f"shape mismatch: got {tuple(got.shape)} vs ref {tuple(ref.shape)}")
    got32, ref32 = got.float(), ref.float()
    denom = ref32.abs().max().clamp_min(1e-6)
    return ((got32 - ref32).abs().max() / denom).item()


def check(got: torch.Tensor, ref: torch.Tensor, tol: float = 2e-2, label: str = "") -> dict:
    """bf16 chains through K=6144..32768 accumulate real error; `tol` is on the *relative*
    max-abs error and the reference is computed in fp32.  Returns a dict rather than
    asserting, so one failed check does not lose an hour of timings -- but it is recorded
    in the result file and the report treats a failed check as a disqualified row."""
    try:
        err = rel_err(got, ref)
        ok = err <= tol and bool(torch.isfinite(got).all().item())
        return {"label": label, "rel_err": err, "tol": tol, "ok": bool(ok)}
    except Exception as exc:  # noqa: BLE001
        return {"label": label, "rel_err": float("inf"), "tol": tol, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:200]}


def screen(
    tag: str,
    run: Callable[[dict], object],
    verify: Callable[[], object],
    grid: Sequence[dict],
    tol: float = 2e-2,
    on_empty: str = "all",
    verbose: bool = True,
) -> tuple[list[dict], list]:
    """Validate EVERY config numerically before it is allowed into a timing grid.

    A wrong answer is not a crash.  MACA's Triton 3.0 miscompiled row-wise (`axis=1`)
    reductions over a `tl.dot` accumulator whenever the mma tile spanned more than one
    warp-row: `tl.max`/`tl.argmax` returned a per-warp partial result, silently, and only
    for some tile shapes.  A miscompiling config is usually also a FAST config -- it skips
    work -- so an unscreened autotuner preferentially selects it, and the winner of the
    search is then the most-wrong config in the grid.  Screening is cheap insurance
    against that on any new stack, which by definition includes this one.

    `run(cfg)` executes the config once; `verify()` returns `bool`, `(bool, detail)`, or a
    float relative error compared against `tol`.  Returns `(ok, rejected)` where
    `rejected` entries are `(cfg, None, reason)` -- the same shape as `TuneResult.table`,
    so a bench can splice them straight in.

    If NOTHING survives (`on_empty="all"`, the default), the full grid is returned with a
    loud warning: a too-tight screening tolerance must not silently kill an hour-long run,
    and the rejections stay in `rejected` so the result file still shows what happened.
    """
    ok: list[dict] = []
    rej: list = []
    for cfg in grid:
        try:
            run(cfg)
            torch.cuda.synchronize()
            v = verify()
            if isinstance(v, tuple):
                good, detail = bool(v[0]), str(v[1])
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                good, detail = float(v) <= tol, f"rel_err={float(v):.3e} tol={tol:.1e}"
            else:
                good, detail = bool(v), ""
            if good:
                ok.append(cfg)
            else:
                rej.append((cfg, None, f"NUMERIC {detail}"))
        except Exception as exc:  # noqa: BLE001 -- compile failures are rejections too
            rej.append((cfg, None, f"{type(exc).__name__}: {exc}"[:200]))
        finally:
            torch.cuda.empty_cache()

    num = [r for r in rej if str(r[2]).startswith("NUMERIC")]
    if verbose:
        print(
            f"    [screen {tag:<9}] {len(grid):>3} offered -> {len(ok):>3} valid "
            f"({len(rej) - len(num)} compile-fail, {len(num)} wrong-answer)",
            flush=True,
        )
        for cfg, _, why in num[:4]:
            print(f"        wrong-answer: {cfg} {str(why)[:110]}", flush=True)
    if not ok and on_empty == "all":
        print(
            f"    !! [screen {tag}] rejected EVERY config; timing the grid UNSCREENED. "
            f"Treat this regime's numbers as unvalidated.",
            flush=True,
        )
        return list(grid), rej
    return ok, rej


# --------------------------------------------------------------------------------------
# Hardware info block, stamped into every result file
# --------------------------------------------------------------------------------------
_HWINFO: dict | None = None


def harness_info() -> dict:
    """What the HARNESS itself was configured with when a number was produced.

    Deliberately distinct from `hwinfo.collect()`, which captures the MACHINE (clocks,
    power, throttle reasons, ECC/MIG).  This is the measurement apparatus: flush size and
    where it came from, timer tick and where it came from, which Hopper paths were unlocked,
    which preflight was trusted -- plus enough device identity to tie the two together.

    Never raises.  It is called from `record()` at the end of a long run, and losing an
    hour of measurements to a missing torch attribute would be an absurd way to fail.
    Whatever could not be probed is reported as an `_errors` entry, so "unknown" is
    visible rather than absent.

    `config.env()` is the authority on the constants that build grids; this is the
    *reporting* view and deliberately soft, which is why it does not call `require_ok()`.
    """
    global _HWINFO
    if _HWINFO is not None:
        return _HWINFO
    info: dict = {"_errors": {}}
    try:
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            p = torch.cuda.get_device_properties(0)
            info.update(
                device_name=p.name,
                compute_capability=f"{p.major}.{p.minor}",
                num_sm=p.multi_processor_count,
                warp_size=getattr(p, "warp_size", None),
                smem_per_block_optin=getattr(p, "shared_memory_per_block_optin", None)
                or getattr(p, "shared_memory_per_block", None),
                smem_per_sm=getattr(p, "shared_memory_per_multiprocessor", None),
                regs_per_sm=getattr(p, "regs_per_multiprocessor", None),
                threads_per_sm=getattr(p, "max_threads_per_multi_processor", None),
                l2_bytes=getattr(p, "L2_cache_size", None),
                total_memory=getattr(p, "total_memory", None),
                uuid=str(getattr(p, "uuid", "")),
            )
            try:
                free, total = torch.cuda.mem_get_info()
                info["mem_free_bytes"], info["mem_total_bytes"] = free, total
            except Exception as exc:  # noqa: BLE001
                info["_errors"]["mem_get_info"] = repr(exc)
    except Exception as exc:  # noqa: BLE001
        info["_errors"]["torch_props"] = repr(exc)
    # If torch could not answer, fill the identity fields from a preflight that provably
    # describes this device, so a result file never says "?" about hardware we do know.
    info.setdefault("device_name", _live_device_name())
    for key, pf_key in (
        ("compute_capability", "compute_capability"),
        ("num_sm", "multi_processor_count"),
        ("l2_bytes", "L2_cache_size"),
    ):
        if not info.get(key):
            v = _pf_get("device", pf_key)
            if v:
                info[key] = v
                info.setdefault("_from_preflight", []).append(key)
    info["torch"] = torch.__version__
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        info["_errors"]["triton"] = repr(exc)
    try:
        info["flush_bytes"] = flush_bytes()
        info["flush_source"] = _FLUSH_SOURCE
    except Exception as exc:  # noqa: BLE001 -- reported, and the bench will have died first
        info["_errors"]["flush_bytes"] = repr(exc)
    info["timer_tick_ms"] = timer_tick_ms()
    info["features"] = features()
    info["preflight"] = {
        "status": preflight().get("_status"),
        "path": str(PREFLIGHT_PATH),
        "timestamp": preflight().get("timestamp"),
        "bandwidth_GBs": calib_bandwidth_gbs(),
        "gemm_TFs": calib_gemm_tfs(),
        "launch_ms": calib_launch_ms(),
    }
    # The nvidia-smi snapshot answers "was it throttled / was ECC on / was it MIG'd" after
    # the fact, which is the first question asked of any surprising ratio.
    smi = preflight().get("nvidia_smi")
    if preflight_ok() and smi:
        keep = ("name", "driver_version", "pstate", "clocks.max.sm", "enforced.power.limit",
                "ecc.mode.current", "mig.mode.current", "persistence_mode")
        info["nvidia_smi"] = [{k: g.get(k) for k in keep if k in g} for g in smi]
    _HWINFO = info
    return info


def banner() -> str:
    h = harness_info()
    f = h.get("features", {})
    feat = ",".join(k for k in ("tma", "clusters", "warp_specialize") if f.get(k)) or "none"
    return (
        f"[harness] {h.get('device_name', '?')} sm{h.get('compute_capability', '?')} | "
        f"{h.get('num_sm', '?')} SM | L2 {(h.get('l2_bytes') or 0) >> 20} MB | "
        f"flush {(h.get('flush_bytes') or 0) >> 20} MB | tick {h.get('timer_tick_ms')} ms | "
        f"hopper feats [{feat}] | preflight {h['preflight']['status']} | "
        f"results -> {RESULTS_DIR}"
    )


# --------------------------------------------------------------------------------------
# Result recording
# --------------------------------------------------------------------------------------
def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + replace, so a crash mid-write cannot leave a truncated
    JSON that the report script then parses as a real (short) result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def record(name: str, payload: dict) -> Path:
    """Write `results/<name>.json` under `$GLM52_H200_RESULTS_DIR` (default results/h200).

    `_meta` is stamped by the harness and overrides anything the caller put there: the
    device name, the hardware block and the timestamp are the provenance of the numbers
    and must not be settable by the code that produced them.
    """
    path = RESULTS_DIR / f"{name}.json"
    payload = dict(payload)
    meta = dict(payload.get("_meta") or {})
    meta.update(
        {
            "schema": HARNESS_SCHEMA,
            "harness": "glm52_h200/common.py",
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": _live_device_name(),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            "torch": torch.__version__,
            "results_dir": str(RESULTS_DIR),
            "harness_info": harness_info(),
            # Machine state at the moment of writing (clocks, power, throttle, ECC/MIG).
            "hwinfo": _hw_call("collect"),
        }
    )
    payload["_meta"] = meta
    _atomic_write(path, json.dumps(payload, indent=2, default=str))
    return path


def speedup_row(
    regime: str,
    fused: Timing,
    unfused: Timing,
    extra: dict | None = None,
    pair: PairTiming | None = None,
) -> dict:
    """One result row from two independently timed arms.

    Prefer `paired_row()`.  This form is kept for per-kernel breakdowns and for tuning
    tables, where the two numbers were never meant to be divided under drift.  Passing
    `pair=` upgrades the row: `speedup` then becomes the PAIRED median and the sequential
    ratio is retained as `speedup_sequential` for comparison.
    """
    row = {
        "regime": regime,
        "fused_ms": fused.p50_ms,
        "unfused_ms": unfused.p50_ms,
        "speedup": unfused.p50_ms / fused.p50_ms if fused.p50_ms > 0 else float("nan"),
        "fused_p10_p90": [fused.p10_ms, fused.p90_ms],
        "unfused_p10_p90": [unfused.p10_ms, unfused.p90_ms],
        "paired": False,
    }
    if pair is not None:
        row["speedup_sequential"] = row["speedup"]
        row.update(_paired_fields(pair))
    if extra:
        row.update(extra)
    return row


def _paired_fields(p: PairTiming) -> dict:
    return {
        "speedup": p.ratio_p50,
        "speedup_p10_p90": [p.ratio_p10, p.ratio_p90],
        "speedup_trimmed": p.ratio_trimmed,
        "speedup_of_medians": p.ratio_of_medians,
        "n_pairs": p.n,
        "frac_fused_faster": p.frac_fused_faster,
        "order_gap_frac": p.order_gap_frac,
        "drift_frac": [p.drift_frac_fused, p.drift_frac_unfused],
        "ratio_halves": [p.ratio_p50_first_half, p.ratio_p50_second_half],
        "tick": p.tick,
        # One boolean the report can filter on, with the full comparison kept alongside.
        "machine_suspect": bool((p.machine or {}).get("compare", {}).get("suspect")),
        "machine": p.machine,
        "paired": True,
    }


def paired_row(regime: str, pair: PairTiming, extra: dict | None = None) -> dict:
    """The headline row for a regime, from an interleaved measurement.

    `speedup` is the paired median (rule 4).  `speedup_of_medians` is what the old
    sequential protocol would have reported; when the two disagree by more than the p10/p90
    band, the machine drifted and only the paired number is meaningful.
    """
    row = {
        "regime": regime,
        "fused_ms": pair.fused.p50_ms,
        "unfused_ms": pair.unfused.p50_ms,
        "fused_p10_p90": [pair.fused.p10_ms, pair.fused.p90_ms],
        "unfused_p10_p90": [pair.unfused.p10_ms, pair.unfused.p90_ms],
    }
    row.update(_paired_fields(pair))
    if extra:
        row.update(extra)
    return row


# --------------------------------------------------------------------------------------
# Checkpoints
#
# Long benches (f04/f05, f08/f09, f11) run for hours; a checkpoint per regime is what makes
# them resumable.  It is also how a result file can end up containing another machine's
# timings inside a freshly-probed local `env` block, which is indistinguishable from a real
# run.  So: the writing device is stamped into the payload, and a checkpoint whose device
# differs from the current one is REFUSED (not warned about, not merged -- refused).
# --------------------------------------------------------------------------------------
def ckpt_dir(name: str) -> Path:
    """Checkpoints follow the results tree, so isolating a run's outputs
    (`GLM52_H200_RESULTS_DIR=...`) also isolates its inputs."""
    return CKPT_ROOT / name


def _force_requested(name: str) -> bool:
    """`GLM52_H200_FORCE=1` re-runs everything; `GLM52_H200_FORCE=f01,f11` re-runs those."""
    v = os.environ.get("GLM52_H200_FORCE", "").strip()
    if not v:
        return False
    if v in ("1", "all", "true", "yes"):
        return True
    return name in {s.strip() for s in v.split(",")}


def ckpt_save(name: str, regime: str, payload: dict) -> Path:
    """Persist one regime's work, stamped with the device that produced it."""
    path = ckpt_dir(name) / f"{regime}.json"
    blob = dict(payload)
    h = harness_info()
    blob.update(
        {
            "device": _live_device_name(),
            "device_uuid": h.get("uuid", ""),
            "torch": torch.__version__,
            "triton": h.get("triton", ""),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "_ckpt_schema": HARNESS_SCHEMA,
        }
    )
    _atomic_write(path, json.dumps(blob, indent=2, default=str))
    return path


def ckpt_load(name: str, regime: str, force: bool | None = None) -> dict | None:
    """Return a previously saved payload, or None if it must not be reused.

    Refuses when:
      * the file is missing or unparseable;
      * `force` (or `$GLM52_H200_FORCE`) asks for a fresh measurement;
      * the checkpoint carries no device stamp -- pre-fence files are indistinguishable
        from foreign ones, so they are treated as foreign;
      * the device name differs from this box's.

    A differing torch/Triton version is only WARNED about: it changes codegen, so the
    timings are suspect, but they were still produced by this device and the operator may
    legitimately be resuming across an upgrade.  The mismatch is left in the payload.
    """
    if force is None:
        force = _force_requested(name)
    path = ckpt_dir(name) / f"{regime}.json"
    if force or not path.exists():
        return None
    try:
        blob = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"[ckpt] {path} unreadable ({exc!r}); re-measuring", flush=True)
        return None

    live = _live_device_name()
    dev = blob.get("device")
    if not dev:
        print(f"[ckpt] {path} carries no device stamp; re-measuring", flush=True)
        return None
    if live and dev != live:
        print(
            f"[ckpt] IGNORING {path}: written by {dev!r}, this box is {live!r}. "
            f"Republishing another GPU's timings under this device's env block is the one "
            f"failure this fence exists to stop.",
            flush=True,
        )
        return None
    if not live:
        print(f"[ckpt] cannot identify this device; refusing to reuse {path}", flush=True)
        return None

    h = harness_info()
    if blob.get("torch") != torch.__version__ or blob.get("triton") != h.get("triton", ""):
        print(
            f"[ckpt] {path} was written by torch {blob.get('torch')} / triton "
            f"{blob.get('triton')}, now torch {torch.__version__} / triton "
            f"{h.get('triton')} -- codegen may differ; reusing anyway (stamp kept)",
            flush=True,
        )
    print(f"[ckpt] reusing {path}", flush=True)
    return blob


# --------------------------------------------------------------------------------------
def main_guard(fn: Callable[[], None]) -> None:
    """Run `fn`, printing a full traceback to stdout on failure.

    The operator on the H200 sends back console output; a bare exception message is not
    enough to debug a Triton compile from here, and a failed round trip costs a day.  The
    two banners go first for the same reason: if the run dies in minute three, the log
    still says which machine it was, at what clocks, with which Hopper paths enabled.
    """
    print(banner(), flush=True)
    m = _hwmod()
    if m is not None and hasattr(m, "banner"):
        try:
            print(m.banner(), flush=True)
        except Exception as exc:  # noqa: BLE001 -- a banner may never kill a run
            print(f"[hw] banner unavailable: {exc!r}", flush=True)
    try:
        fn()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        # stdout is the deliverable when the run dies; make sure it reached the log.
        try:
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass
