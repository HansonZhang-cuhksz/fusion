"""Machine-state capture: what the hardware was doing when a number was produced.

Every result file in this study embeds `collect()`. The reason is not bookkeeping, it is that
a speedup ratio is only interpretable next to the machine state that produced it, and three
separate incidents in this study's history turned on state nobody had recorded:

  * the RTX 4060's SM clock was pinned at 1020 MHz of a 3105 MHz maximum -- 33 %. Every
    absolute number from that machine is meaningless without that fact, and its *ratios* are
    only comparable to C500's because the clamp happened to reproduce C500's compute/bandwidth
    balance. Nothing in the result files said so; it had to be reconstructed afterwards.
  * a fused arm measured hot and an unfused arm measured cold produced a speedup above the
    device's own physical ceiling. Thermal drift within one run was 22 %. `snapshot()` and
    `compare_snapshots()` exist so a bench can bracket its timing loop and *prove* the machine
    did not move underneath it.
  * a result file carried a freshly-probed device header over timings from another GPU.

So: clocks current AND maximum, whether they are locked, power draw against limit, both
temperatures, ECC/MIG/persistence/compute mode, PCIe link state, and the driver's own throttle
reasons -- decoded, because `0x0000000000000001` is not something an operator reads.

Everything is guarded. A machine with no `nvidia-smi`, a driver that renamed a query field, a
torch too old to expose NVML helpers -- each degrades to a recorded error and the rest of the
collection continues. This module must never be the reason a benchmark run dies, because the
H200 cannot be re-run cheaply: one crash costs a whole round trip.

    from glm52_h200 import hwinfo
    print(hwinfo.banner())
    payload["_hw"] = hwinfo.collect()

Deliberately independent of `config.py`: nothing here initialises CUDA on its own (the NVML
helpers are skipped when `torch.cuda.is_available()` is False), so it is safe to call before a
device probe, from a report script, or on a box with no GPU at all.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import socket
import subprocess
import time

# --------------------------------------------------------------------------------------
# subprocess plumbing -- a missing or unhappy nvidia-smi must never raise
# --------------------------------------------------------------------------------------
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _run(cmd: list, timeout: int = 60) -> tuple:
    """(returncode, stdout, stderr); returncode is None if the command could not run."""
    if not cmd or not shutil.which(cmd[0]):
        return (None, "", f"{cmd[0] if cmd else '?'} not on PATH")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.returncode, p.stdout or "", p.stderr or "")
    except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
        return (None, "", f"{type(exc).__name__}: {exc}")


def _text(cmd: list, limit: int = 4000, timeout: int = 60) -> str:
    rc, out, err = _run(cmd, timeout)
    if rc is None:
        return f"<unavailable: {err}>"
    if rc != 0 and not out.strip():
        return f"<rc={rc}: {err.strip()[:200]}>"
    return _ANSI.sub("", out)[:limit]


def _smi_batch(fields: list, timeout: int = 30) -> tuple:
    """One `--query-gpu` call. Returns (rows_of_values, error)."""
    rc, out, err = _run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"], timeout
    )
    if rc is None:
        return None, err
    blob = f"{out}\n{err}".lower()
    if rc != 0 or "not a valid field" in blob or "unrecognized" in blob:
        return None, (err or out).strip()[:200] or f"rc={rc}"
    rows = []
    for line in out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        vals = [v.strip() for v in line.split(",")]
        if len(vals) != len(fields):
            return None, f"expected {len(fields)} values per row, got {len(vals)}"
        rows.append(vals)
    return (rows or None), ("" if rows else "no rows")


def _smi_query(fields: list, timeout: int = 30) -> tuple:
    """Query `fields` for every GPU, tolerating field names this driver does not know.

    nvidia-smi rejects the WHOLE query when one field name is unknown -- and the names have
    moved across driver branches (`clocks_throttle_reasons.*` was renamed
    `clocks_event_reasons.*` in the R555 branch, `temperature.memory` appeared later still).
    The H200 host's driver version is not knowable from here, so: try the batch, and on any
    failure re-query field by field, which costs one field instead of all of them.

    Returns (list of per-GPU dicts, dict of per-field errors).
    """
    errors: dict = {}
    rows, err = _smi_batch(fields, timeout)
    if rows is not None:
        return [dict(zip(fields, r)) for r in rows], errors
    errors["_batch"] = err
    per: dict = {}
    ngpu = 0
    for f in fields:
        r, e = _smi_batch([f], timeout)
        if r is None:
            errors[f] = e
            continue
        per[f] = [v[0] for v in r]
        ngpu = max(ngpu, len(per[f]))
    gpus = [{f: v[i] for f, v in per.items() if i < len(v)} for i in range(ngpu)]
    return gpus, errors


def _num(v) -> "float | None":
    """First number in an nvidia-smi value ("1020 MHz" -> 1020.0); None for [N/A]."""
    if v is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return float(m.group()) if m else None


# --------------------------------------------------------------------------------------
# nvidia-smi field sets
# --------------------------------------------------------------------------------------
# Short aliases (`clocks.sm`, `clocks.max.mem`) are used deliberately: they have been accepted
# by every driver branch this suite has met, where some long forms have not.
_STATIC_FIELDS = [
    "index", "name", "uuid", "serial", "driver_version", "vbios_version",
    "pci.bus_id", "pcie.link.gen.max", "pcie.link.width.max",
    "clocks.max.sm", "clocks.max.mem", "clocks.max.gr",
    "clocks.default_applications.graphics", "clocks.default_applications.memory",
    "ecc.mode.current", "ecc.mode.pending",
    "mig.mode.current", "mig.mode.pending",
    "persistence_mode", "compute_mode", "accounting.mode",
    "power.max_limit", "power.min_limit", "power.default_limit", "enforced.power.limit",
    "memory.total",
]
_DYNAMIC_FIELDS = [
    "index", "pstate",
    "clocks.sm", "clocks.mem", "clocks.gr",
    "clocks.applications.graphics", "clocks.applications.memory",
    "clocks_throttle_reasons.active", "clocks_event_reasons.active",
    "power.draw", "power.limit",
    "temperature.gpu", "temperature.memory", "fan.speed",
    "utilization.gpu", "utilization.memory",
    "memory.used", "memory.free",
    "pcie.link.gen.current", "pcie.link.width.current",
]

# NVML clocksThrottleReasons bitmask. Recorded as names because a hex mask in a result file is
# not something anyone decodes six weeks later -- and "SwPowerCap was active for this run" is
# the difference between a real regression and a hot laptop.
_THROTTLE_BITS = [
    (0x0000000000000001, "GpuIdle"),
    (0x0000000000000002, "ApplicationsClocksSetting"),
    (0x0000000000000004, "SwPowerCap"),
    (0x0000000000000008, "HwSlowdown"),
    (0x0000000000000010, "SyncBoost"),
    (0x0000000000000020, "SwThermalSlowdown"),
    (0x0000000000000040, "HwThermalSlowdown"),
    (0x0000000000000080, "HwPowerBrakeSlowdown"),
    (0x0000000000000100, "DisplayClockSetting"),
]


def decode_throttle(mask) -> list:
    """Bitmask (hex string or int) -> reason names. [] when idle/unknown."""
    if mask is None:
        return []
    try:
        m = int(str(mask).strip(), 16) if "x" in str(mask).lower() else int(_num(mask) or 0)
    except (TypeError, ValueError):
        return []
    out = [name for bit, name in _THROTTLE_BITS if m & bit]
    unknown = m & ~sum(bit for bit, _ in _THROTTLE_BITS)
    if unknown:
        out.append(f"unknown:0x{unknown:x}")
    return out


def clock_state(g: dict) -> dict:
    """Are the clocks locked, and how far below maximum are they sitting?

    Tri-state on purpose. "unknown" is an honest and common answer -- application clocks read
    `[N/A]` on consumer parts and on many datacentre configurations -- and it is much more
    useful than a confident guess, because the follow-up action (ask the operator to run
    `nvidia-smi -q -d CLOCK`) is the same either way.

    The ratios are reported unconditionally: a GPU parked at a third of its maximum SM clock
    is the single most important fact about any absolute number it produced, whether the cause
    is a lock, a power cap, or thermals.
    """
    sm, sm_max = _num(g.get("clocks.sm")), _num(g.get("clocks.max.sm"))
    mem, mem_max = _num(g.get("clocks.mem")), _num(g.get("clocks.max.mem"))
    app_gr = _num(g.get("clocks.applications.graphics"))
    app_mem = _num(g.get("clocks.applications.memory"))
    def_gr = _num(g.get("clocks.default_applications.graphics"))
    def_mem = _num(g.get("clocks.default_applications.memory"))
    reasons = decode_throttle(
        g.get("clocks_throttle_reasons.active") or g.get("clocks_event_reasons.active")
    )

    basis, state = [], "unknown"
    if "ApplicationsClocksSetting" in reasons:
        state = "locked"
        basis.append("throttle reason ApplicationsClocksSetting is active")
    if app_gr is not None and def_gr is not None and app_gr != def_gr:
        state = "locked"
        basis.append(f"application graphics clock {app_gr:.0f} != default {def_gr:.0f} MHz")
    if app_mem is not None and def_mem is not None and app_mem != def_mem:
        state = "locked"
        basis.append(f"application memory clock {app_mem:.0f} != default {def_mem:.0f} MHz")
    if state != "locked":
        if app_gr is None and def_gr is None:
            basis.append("application clocks report [N/A]; a `-lgc` lock is not visible here")
        else:
            state = "unlocked"
            basis.append("application clocks equal their defaults")
    for r in ("SwPowerCap", "HwPowerBrakeSlowdown", "SwThermalSlowdown", "HwThermalSlowdown"):
        if r in reasons:
            basis.append(f"{r} active -- clocks are being held down, which is not the same "
                         f"as locked and is not reproducible run to run")

    return {
        "state": state,
        "basis": basis,
        "throttle_reasons": reasons,
        "sm_mhz": sm, "sm_max_mhz": sm_max,
        "sm_frac_of_max": (sm / sm_max) if sm and sm_max else None,
        "mem_mhz": mem, "mem_max_mhz": mem_max,
        "mem_frac_of_max": (mem / mem_max) if mem and mem_max else None,
        "applications_sm_mhz": app_gr, "applications_mem_mhz": app_mem,
        "default_applications_sm_mhz": def_gr, "default_applications_mem_mhz": def_mem,
    }


# --------------------------------------------------------------------------------------
# torch / triton side
# --------------------------------------------------------------------------------------
def stack_info() -> dict:
    """Versions of everything whose change would change a measured number."""
    s = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        s["torch"] = torch.__version__
        s["torch_cuda"] = torch.version.cuda
        s["torch_hip"] = getattr(torch.version, "hip", None)
        s["torch_git"] = getattr(torch.version, "git_version", None)
        try:
            s["cudnn"] = torch.backends.cudnn.version()
        except Exception as exc:  # noqa: BLE001
            s["cudnn"] = f"<{type(exc).__name__}>"
        s["cuda_available"] = torch.cuda.is_available()
        s["device_count"] = torch.cuda.device_count() if s["cuda_available"] else 0
        try:
            s["cuda_arch_list"] = torch.cuda.get_arch_list()
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        s["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import triton

        s["triton"] = triton.__version__
        s["triton_file"] = getattr(triton, "__file__", None)
    except Exception as exc:  # noqa: BLE001
        s["triton_error"] = f"{type(exc).__name__}: {exc}"
    return s


def torch_flags() -> dict:
    """Numeric-behaviour switches. These change GEMM throughput without changing any source.

    `allow_tf32` and `allow_bf16_reduced_precision_reduction` in particular move a bf16 GEMM's
    measured time; they are process-global, they are inherited from whatever ran before, and
    they are invisible in a bench's own code. Record them next to the timings.
    """
    f: dict = {}
    try:
        import torch

        f["float32_matmul_precision"] = torch.get_float32_matmul_precision()
        f["cuda.matmul.allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
        f["cudnn.allow_tf32"] = torch.backends.cudnn.allow_tf32
        f["cudnn.benchmark"] = torch.backends.cudnn.benchmark
        for name in (
            "allow_bf16_reduced_precision_reduction",
            "allow_fp16_reduced_precision_reduction",
        ):
            if hasattr(torch.backends.cuda.matmul, name):
                f[f"cuda.matmul.{name}"] = getattr(torch.backends.cuda.matmul, name)
    except Exception as exc:  # noqa: BLE001
        f["_error"] = f"{type(exc).__name__}: {exc}"
    f["env"] = {
        k: os.environ[k]
        for k in (
            "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_LAUNCH_BLOCKING",
            "PYTORCH_CUDA_ALLOC_CONF", "TORCH_CUDA_ARCH_LIST", "NVIDIA_TF32_OVERRIDE",
            "TRITON_CACHE_DIR", "TRITON_PRINT_AUTOTUNING", "TRITON_ALWAYS_COMPILE",
            "GLM52_RESULTS_DIR", "GLM52_H200_RESULTS_DIR", "GLM52_H200_PREFLIGHT",
            "GLM52_H200_DISABLE_FEATURES",
        )
        if k in os.environ
    }
    return f


def device_properties(index: "int | None" = None) -> dict:
    """*Every* field torch exposes for the device, not a curated subset.

    Curated subsets are how a port discovers, too late, that the one property it needed was
    never recorded. `dir()` costs nothing and survives torch renaming fields between versions.
    """
    d: dict = {}
    try:
        import torch

        if not torch.cuda.is_available():
            return {"_error": "cuda not available"}
        idx = torch.cuda.current_device() if index is None else index
        p = torch.cuda.get_device_properties(idx)
        d["_index"] = idx
        for a in sorted(dir(p)):
            if a.startswith("_"):
                continue
            try:
                v = getattr(p, a)
            except Exception:  # noqa: BLE001
                continue
            if callable(v):
                continue
            d[a] = v if isinstance(v, (int, float, bool, str)) else str(v)
        try:
            free, total = torch.cuda.mem_get_info(idx)
            d["mem_free_bytes"], d["mem_total_bytes"] = free, total
        except Exception:  # noqa: BLE001
            pass
        try:
            d["compute_capability"] = "%d.%d" % torch.cuda.get_device_capability(idx)
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        d["_error"] = f"{type(exc).__name__}: {exc}"
    return d


def all_devices() -> list:
    """One compact row per visible GPU -- an H200 node usually has eight of them, and which
    one a run landed on is not otherwise obvious from a result file."""
    out = []
    try:
        import torch

        if not torch.cuda.is_available():
            return out
        for i in range(torch.cuda.device_count()):
            try:
                p = torch.cuda.get_device_properties(i)
                out.append(
                    {
                        "index": i,
                        "name": p.name,
                        "uuid": str(getattr(p, "uuid", "")),
                        "total_memory": getattr(p, "total_memory", None),
                        "compute_capability": f"{getattr(p, 'major', '?')}."
                                              f"{getattr(p, 'minor', '?')}",
                        "multi_processor_count": getattr(p, "multi_processor_count", None),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                out.append({"index": i, "_error": f"{type(exc).__name__}: {exc}"})
    except Exception:  # noqa: BLE001
        pass
    return out


def nvml_via_torch(index: "int | None" = None) -> dict:
    """torch's NVML helpers -- a second opinion on the volatile state.

    Worth having independently of nvidia-smi: it works when the binary is not on PATH (common
    in containers), and a disagreement between the two is itself a signal that the process is
    not looking at the GPU it thinks it is. Skipped entirely when CUDA is unavailable, so this
    module never initialises a context on its own.
    """
    n: dict = {}
    try:
        import torch

        if not torch.cuda.is_available():
            return {"_skipped": "cuda not available"}
        idx = torch.cuda.current_device() if index is None else index
        for name in ("clock_rate", "power_draw", "temperature", "utilization", "memory_usage"):
            fn = getattr(torch.cuda, name, None)
            if fn is None:
                continue
            try:
                n[name] = fn(idx)
            except Exception as exc:  # noqa: BLE001
                n[name] = f"<{type(exc).__name__}: {exc}>"[:120]
    except Exception as exc:  # noqa: BLE001
        n["_error"] = f"{type(exc).__name__}: {exc}"
    return n


def host_info() -> dict:
    """The box, not the GPU. CPU count and host RAM bound what the harness itself can do, and
    the hostname is what ties a result file to the machine the operator ran it on."""
    h = {
        "hostname": "",
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "kernel": platform.release(),
    }
    try:
        h["hostname"] = socket.gethostname()
    except Exception:  # noqa: BLE001
        pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("model name"):
                    h["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:  # noqa: BLE001
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    h["mem_total_kb"] = int(_num(line) or 0)
                    break
    except Exception:  # noqa: BLE001
        pass
    return h


# --------------------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------------------
_STATIC_CACHE: dict = {}  # keyed by requested device index (None == "current")
_DYNAMIC_CACHE: dict = {}


def collect(index: "int | None" = None, topology: bool = True, refresh: bool = True) -> dict:
    """Everything, as a JSON-serialisable dict. Embed this in every result file.

    The static half (versions, device properties, ECC/MIG/persistence, topology) is cached
    after the first call -- it costs half a dozen subprocesses and cannot change under a
    running process. The volatile half (clocks, temperature, power, throttle reasons, free
    memory) is re-read every time, because "the machine state that produced this number" means
    the state at the moment the number was produced, not at import time.

    `refresh=False` reuses the last volatile read too, for a caller writing several files from
    one measurement that should all carry the same machine state.
    """
    errors: dict = {}

    if index not in _STATIC_CACHE:
        static_smi, e_static = _smi_query(_STATIC_FIELDS)
        if e_static:
            errors["nvidia_smi_static"] = e_static
        _STATIC_CACHE[index] = {
            "host": host_info(),
            "stack": stack_info(),
            "devices": all_devices(),
            "torch_device_properties": device_properties(index),
            "nvidia_smi_static": static_smi,
            "nvidia_smi_version": _text(["nvidia-smi", "--version"], 600).strip(),
            "nvidia_smi_list": _text(["nvidia-smi", "-L"], 2000).strip(),
            # `-q -d CLOCK` is the only place a `-lgc`-style lock reliably shows up in text
            # form; kept raw and truncated rather than parsed, because its layout differs
            # across driver branches and a parse that silently misses is worse than a blob.
            "nvidia_smi_clock_report": _text(["nvidia-smi", "-q", "-d", "CLOCK"], 4000),
            "topology": _text(["nvidia-smi", "topo", "-m"], 4000) if topology else "<skipped>",
            "nvlink": _text(["nvidia-smi", "nvlink", "-s"], 2000) if topology else "<skipped>",
            "_static_errors": errors.copy(),
        }

    info = dict(_STATIC_CACHE[index])
    info["schema"] = 1
    info["collected_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info["torch_flags"] = torch_flags()

    if refresh or index not in _DYNAMIC_CACHE:
        dyn_smi, e_dyn = _smi_query(_DYNAMIC_FIELDS)
        _DYNAMIC_CACHE[index] = (dyn_smi, e_dyn, nvml_via_torch(index))
    dyn_smi, e_dyn, nvml = _DYNAMIC_CACHE[index]
    if e_dyn:
        errors["nvidia_smi_dynamic"] = e_dyn
    info["nvidia_smi_dynamic"] = dyn_smi
    info["nvml_via_torch"] = nvml

    # Merge the static and dynamic rows for the GPU this process is actually using, then
    # decide the clock question once, here, so no caller has to re-derive it.
    idx = info.get("torch_device_properties", {}).get("_index", 0)
    merged = _merge_rows(info.get("nvidia_smi_static"), dyn_smi, idx)
    info["gpu"] = merged
    info["clocks"] = (
        clock_state(merged)
        if merged
        else {"state": "unknown", "basis": ["no nvidia-smi data for this GPU"]}
    )
    info["errors"] = {**info.get("_static_errors", {}), **errors}
    return info


def _merge_rows(static_rows, dyn_rows, idx: int) -> dict:
    """Static+dynamic nvidia-smi rows for GPU `idx`, matched on the reported `index` field.

    Matched on `index` rather than position because CUDA_DEVICE_ORDER, MIG and
    CUDA_VISIBLE_DEVICES all make torch's ordinal and nvidia-smi's ordinal disagree; when they
    cannot be matched we fall back to the single row if there is only one, and to nothing
    otherwise. Attributing another GPU's clocks to this one would be worse than reporting none.
    """
    def pick(rows):
        if not rows:
            return {}
        for r in rows:
            if str(r.get("index", "")).strip() == str(idx):
                return r
        return rows[0] if len(rows) == 1 else {}

    out = dict(pick(static_rows))
    out.update(pick(dyn_rows))
    return out


def snapshot(index: "int | None" = None) -> dict:
    """The volatile state alone, cheap enough to take twice around a timing loop.

    Bracketing a measurement with two snapshots is how monotone drift is *shown* rather than
    assumed absent: interleaving the two arms makes drift cancel out of the ratio, and these
    two rows are the evidence that the interleaving was needed (or that it was not).
    """
    # Short timeout on purpose: this may be called between two arms of a measurement, and a
    # wedged driver query must not become the thing that stalls (or dominates) the run.
    rows, err = _smi_query(
        ["index", "clocks.sm", "clocks.mem", "temperature.gpu", "temperature.memory",
         "power.draw", "utilization.gpu", "pstate", "clocks_throttle_reasons.active"],
        timeout=15,
    )
    idx = index
    if idx is None:
        try:
            import torch

            idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
        except Exception:  # noqa: BLE001
            idx = 0
    g = _merge_rows(None, rows, idx)
    return {
        "t": time.time(),
        "sm_mhz": _num(g.get("clocks.sm")),
        "mem_mhz": _num(g.get("clocks.mem")),
        "temp_c": _num(g.get("temperature.gpu")),
        "mem_temp_c": _num(g.get("temperature.memory")),
        "power_w": _num(g.get("power.draw")),
        "util_pct": _num(g.get("utilization.gpu")),
        "pstate": g.get("pstate"),
        "throttle": decode_throttle(g.get("clocks_throttle_reasons.active")),
        "error": err or None,
    }


def compare_snapshots(before: dict, after: dict, clock_tol: float = 0.02,
                      temp_tol: float = 10.0) -> dict:
    """Did the machine move underneath the measurement?

    `suspect` is not a failure -- it is a flag for the report. A 22 % SM-clock drop or a 10 C
    rise between the start and the end of a tuning loop means the two arms were not timed
    under the same conditions, and any speedup from that loop needs an interleaved re-run
    before it is quoted.
    """
    def d(k):
        a, b = before.get(k), after.get(k)
        return (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None

    sm0, sm1 = before.get("sm_mhz"), after.get("sm_mhz")
    sm_rel = ((sm1 - sm0) / sm0) if isinstance(sm0, (int, float)) and sm0 and \
        isinstance(sm1, (int, float)) else None
    dt = d("temp_c")
    reasons = []
    if sm_rel is not None and abs(sm_rel) > clock_tol:
        reasons.append(f"SM clock moved {sm_rel * 100:+.1f}% ({sm0:.0f} -> {sm1:.0f} MHz)")
    if dt is not None and abs(dt) > temp_tol:
        reasons.append(f"temperature moved {dt:+.0f} C")
    new_throttle = sorted(set(after.get("throttle") or []) - set(before.get("throttle") or []))
    if new_throttle:
        reasons.append(f"new throttle reasons: {', '.join(new_throttle)}")
    return {
        "elapsed_s": d("t"),
        "d_sm_mhz": d("sm_mhz"), "sm_rel": sm_rel,
        "d_mem_mhz": d("mem_mhz"), "d_temp_c": dt, "d_power_w": d("power_w"),
        "new_throttle": new_throttle,
        "suspect": bool(reasons),
        "reasons": reasons,
    }


# --------------------------------------------------------------------------------------
# printable
# --------------------------------------------------------------------------------------
def banner(info: "dict | None" = None) -> str:
    """A printable block for the top of a bench's stdout. Never raises."""
    try:
        info = info or collect()
    except Exception as exc:  # noqa: BLE001 -- a banner must not be able to kill a run
        return f"[hw] unavailable: {type(exc).__name__}: {exc}"

    g = info.get("gpu", {}) or {}
    st = info.get("stack", {}) or {}
    tp = info.get("torch_device_properties", {}) or {}
    ck = info.get("clocks", {}) or {}
    host = info.get("host", {}) or {}

    def fmt(v, unit=""):
        return "?" if v is None else f"{v:g}{unit}"

    def pct(v):
        return "?" if v is None else f"{v * 100:.0f}%"

    total_gb = (tp.get("total_memory") or _num(g.get("memory.total")) or 0)
    total_gb = total_gb / 2**30 if total_gb > 2**20 else total_gb / 1024  # bytes vs MiB
    lines = [
        f"[hw] {g.get('name') or tp.get('name', '?')}  "
        f"sm_{tp.get('major', '?')}{tp.get('minor', '?')}  "
        f"{tp.get('multi_processor_count', '?')} SM  "
        f"{total_gb:.0f} GB  uuid {str(tp.get('uuid', g.get('uuid', '?')))[:20]}",
        f"[hw] driver {g.get('driver_version', '?')}  vbios {g.get('vbios_version', '?')}  "
        f"cuda {st.get('torch_cuda', '?')}  torch {st.get('torch', '?')}  "
        f"triton {st.get('triton', '?')}  python {st.get('python', '?')}",
        f"[hw] clocks SM {fmt(ck.get('sm_mhz'), ' MHz')} / "
        f"{fmt(ck.get('sm_max_mhz'), ' MHz')} max ({pct(ck.get('sm_frac_of_max'))})  "
        f"MEM {fmt(ck.get('mem_mhz'), ' MHz')} / {fmt(ck.get('mem_max_mhz'), ' MHz')} max "
        f"({pct(ck.get('mem_frac_of_max'))})  -> {ck.get('state', '?')}",
        f"[hw] pstate {g.get('pstate', '?')}  power {g.get('power.draw', '?')} of "
        f"{g.get('enforced.power.limit', g.get('power.limit', '?'))}  "
        f"temp {g.get('temperature.gpu', '?')} C (mem {g.get('temperature.memory', '?')})  "
        f"util {g.get('utilization.gpu', '?')}",
        f"[hw] ecc {g.get('ecc.mode.current', '?')}  mig {g.get('mig.mode.current', '?')}  "
        f"persistence {g.get('persistence_mode', '?')}  compute-mode "
        f"{g.get('compute_mode', '?')}  pcie gen{g.get('pcie.link.gen.current', '?')}"
        f"x{g.get('pcie.link.width.current', '?')}",
        f"[hw] host {host.get('hostname', '?')}  {host.get('cpu_count', '?')} cpu  "
        f"visible GPUs {st.get('device_count', '?')}  "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}",
    ]
    if ck.get("throttle_reasons"):
        lines.append(f"[hw] throttle: {', '.join(ck['throttle_reasons'])}")
    for b in ck.get("basis", [])[:3]:
        lines.append(f"[hw] clock basis: {b}")
    if info.get("errors"):
        lines.append(f"[hw!] probe errors: {list(info['errors'])}")
    return "\n".join(lines)


if __name__ == "__main__":  # `python3 -m glm52_h200.hwinfo` prints the block and exits
    print(banner())
