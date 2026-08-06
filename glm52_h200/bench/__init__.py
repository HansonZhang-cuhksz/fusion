"""Shared driver harness for the H200 bench suite.

`glm52/bench/__init__.py` is empty; this one is not, and the reason is the H200 itself.

Nobody can test on that machine. Eight drivers each re-implementing argument parsing,
checkpoint fencing, interleaved pair timing and grid bookkeeping means eight independent
chances to crash a run that costs a whole round trip to discover. So every piece of
machinery that is *identical* across the drivers lives here once, is written defensively
once, and -- where it depends on an API another module owns (`common`, `config`,
`kernels`) -- adapts to what is actually there at runtime instead of assuming.

What this module guarantees to the drivers:

*  **Every hardware constant comes from `config.env()`**, through `env_int()`, which
   raises rather than substituting a plausible default.  A wrong-but-plausible constant
   prunes an autotuning grid, and it does not prune the two arms of a fused/unfused pair
   equally -- which manufactures or destroys the ratio that is this study's only output.
*  **The preflight JSON is cross-checked against the live device.**  A `preflight_h200.json`
   copied from another box would otherwise silently supply that box's timer tick, L2 size
   and feature list to a run labelled H200.  Mismatch is fatal, not a warning.
*  **Final timings are interleaved and paired.**  `bench_pair()` prefers
   `common.bench_pair`; if that entry point is missing or has an incompatible signature it
   falls back to a local interleaved implementation and *records which one ran*.  On the
   4060 a sequential fused-then-unfused measurement drifted 22 % thermally inside one run
   and produced a speedup above the cell's own physical ceiling; monotone drift cancels in
   a paired ratio and does not cancel in two sequential medians.
*  **Checkpoints are fenced on the device.**  A stale checkpoint from another GPU was one
   call away from being republished as a fresh measurement on the 4060 port.
*  **H200-only mapping axes are opted into at runtime**, never at authoring time: a config
   overlay is offered only when a LIVE capability probe says the feature compiles and runs
   *and* the kernel module advertises the corresponding cfg key.  Absent either, the grids
   are exactly the classic ones and the sm_90 path simply never appears.  Every offered axis
   goes to BOTH arms of every pair -- an axis offered to one arm only is precisely the
   one-sided grid bias the fairness accounting exists to catch.
*  **One GPU, chosen deliberately.**  The measured host has EIGHT H200s and other tenants:
   at preflight time 51 GB of GPU 0 was already someone else's, and the launch/timer
   calibration it produced is visibly contaminated (a 40 us harness floor, a tick that
   matched 3 % of samples).  `--gpu N` / `--gpu auto` masks the process down to one device
   before CUDA initialises, refuses a device that is already busy unless `--allow-busy`, and
   records the chosen index AND UUID in every result file so a number can be traced to a
   physical card.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

HERE = Path(__file__).resolve().parent
PKG = HERE.parent  # glm52_h200/
ROOT = PKG.parent  # repository root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ======================================================================================
# GPU selection.
#
# THIS BLOCK RUNS BEFORE `import torch` ON PURPOSE.  `CUDA_VISIBLE_DEVICES` is read by the
# CUDA driver exactly once, at `cuInit`, and torch triggers that on its first
# `torch.cuda.*` call.  Set the variable after that point and it is silently ignored --
# the process keeps all eight devices, `cuda:0` is whatever it always was, and the operator
# gets a result file that names a GPU nobody selected.  So the resolution happens at import
# time of this package, which every driver imports before it touches the device.
#
# `--gpu` is parsed straight out of `sys.argv` here rather than through argparse, because
# argparse cannot run until the driver's `main()`, which is long after CUDA is up.  The
# driver still declares the flag (see `add_gpu_args`) so `--help` documents it and a typo is
# rejected; by then this block has already acted on it.  Nothing happens at all unless
# `--gpu` is literally present or `$GLM52_H200_GPU` is set, so importing this package from
# an unrelated tool with an unrelated argv is inert.
# ======================================================================================
GPU_ENV_VAR = "GLM52_H200_GPU"  # alternative to --gpu, for wrappers that cannot pass argv
_GPU_DECISION_VAR = "GLM52_H200_GPU_SELECTION"  # carries the parent's decision to a re-exec
_GPU_REEXEC_VAR = "GLM52_H200_GPU_REEXEC"  # loop guard for the last-resort re-exec

#: Fraction of a device's memory that may already be in use before it counts as busy, and
#: the absolute floor below which "in use" is just the driver's own footprint.  Persistence
#: mode alone shows ~4 MiB on the measured host; a second tenant showed 22-48 GB, so the
#: separation is four orders of magnitude and the exact threshold is not delicate.
GPU_BUSY_MEM_FRACTION = 0.01
GPU_BUSY_MEM_FLOOR_MB = 1024

_GPU_SELECTION: dict = {"requested": None, "applied": False}


def _smi(query: str, entity: str = "gpu") -> list[list[str]]:
    """One `nvidia-smi --query-<entity>` call, as rows of stripped fields.

    Returns `[]` on any failure.  This runs before CUDA is initialised and must never be the
    reason a run does not start: without it `--gpu auto` degrades to "cannot rank", which is
    reported, and `--gpu N` still works because masking a device needs no inventory.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, f"--query-{entity}={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except Exception:  # noqa: BLE001 -- no smi, no ranking; not a reason to abort
        return []
    if out.returncode != 0:
        return []
    return [
        [f.strip() for f in line.split(",")]
        for line in out.stdout.splitlines() if line.strip()
    ]


def _as_num(s: str, default: float = -1.0) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default  # "[N/A]" from a MIG or a driver that will not answer


def gpu_inventory() -> list[dict]:
    """Every GPU on the host with its utilisation, memory use and foreign compute processes.

    The process list is the part that matters.  `memory.used` alone cannot distinguish a
    tenant who has allocated 20 GB and is hammering the SMs from a leaked allocation nobody
    is using, and `utilization.gpu` is a 1-second duty cycle that reads 0 % for a job between
    kernels.  A live compute process is unambiguous.
    """
    rows = _smi("index,uuid,name,utilization.gpu,memory.used,memory.total")
    apps = _smi("gpu_uuid,pid,used_memory", entity="compute-apps")
    by_uuid: dict[str, list[dict]] = {}
    for a in apps:
        if len(a) >= 3:
            by_uuid.setdefault(a[0], []).append(
                {"pid": a[1], "used_mb": _as_num(a[2], 0.0)}
            )
    inv = []
    for r in rows:
        if len(r) < 6:
            continue
        uuid = r[1]
        total = _as_num(r[5], 0.0)
        used = _as_num(r[4], 0.0)
        procs = by_uuid.get(uuid, [])
        inv.append(
            {
                "index": int(_as_num(r[0], -1)),
                "uuid": uuid,
                "name": r[2],
                "util_pct": _as_num(r[3]),
                "mem_used_mb": used,
                "mem_total_mb": total,
                "n_procs": len(procs),
                "procs": procs,
                "busy_threshold_mb": max(
                    GPU_BUSY_MEM_FLOOR_MB, GPU_BUSY_MEM_FRACTION * total
                ),
            }
        )
    for g in inv:
        g["busy"] = bool(
            g["n_procs"] > 0
            or g["mem_used_mb"] > g["busy_threshold_mb"]
            or g["util_pct"] > 5.0
        )
    return inv


def _rank_gpus(inv: list[dict]) -> list[dict]:
    """Idlest first: utilisation, then resident memory, then index for determinism."""
    return sorted(
        inv, key=lambda g: (g["n_procs"], g["util_pct"], g["mem_used_mb"], g["index"])
    )


def _render_ranking(ranked: list[dict], chosen: int | None) -> list[str]:
    lines = [
        f"[gpu] {'':2} {'idx':>3} {'util':>6} {'mem used':>12} {'procs':>6}  uuid",
    ]
    for g in ranked:
        lines.append(
            f"[gpu] {'->' if g['index'] == chosen else '  ':2} {g['index']:>3} "
            f"{g['util_pct']:>5.0f}% {g['mem_used_mb']:>8.0f} MiB {g['n_procs']:>6}  "
            f"{g['uuid']}"
        )
    return lines


def select_gpu(spec: str, allow_busy: bool = False,
               busy_mb: float | None = None) -> dict:
    """Resolve `--gpu <spec>` to one physical device, with the full ranking for the record.

    `spec` is an integer index or `auto`.  `auto` takes the idlest device by
    (foreign processes, utilisation, resident memory) and refuses if even that one is busy;
    an explicit index refuses if THAT one is busy.  `allow_busy` downgrades the refusal to a
    warning, which is occasionally the right call and must always be a deliberate one: a
    contended device is exactly what produced the preflight's 40 us harness floor and its
    3 %-match timer tick, and neither is recoverable after the fact.
    """
    inv = gpu_inventory()
    if busy_mb is not None:
        for g in inv:
            g["busy_threshold_mb"] = float(busy_mb)
            g["busy"] = bool(
                g["n_procs"] > 0 or g["mem_used_mb"] > busy_mb or g["util_pct"] > 5.0
            )
    ranked = _rank_gpus(inv)
    sel: dict = {
        "requested": spec,
        "applied": False,
        "allow_busy": bool(allow_busy),
        "inventory": [{k: v for k, v in g.items() if k != "procs"} for g in ranked],
        "ranking": [g["index"] for g in ranked],
    }
    if str(spec).strip().lower() == "auto":
        if not ranked:
            sel["error"] = (
                "--gpu auto needs nvidia-smi to rank the devices and it did not answer; "
                "pass an explicit --gpu N instead"
            )
            return sel
        pick = ranked[0]
    else:
        try:
            want = int(str(spec).strip())
        except ValueError:
            sel["error"] = f"--gpu {spec!r} is neither an integer index nor 'auto'"
            return sel
        by_idx = {g["index"]: g for g in inv}
        if inv and want not in by_idx:
            sel["error"] = (
                f"--gpu {want} does not exist; this host reports "
                f"{sorted(by_idx)} "
            )
            return sel
        # No inventory (no nvidia-smi) is survivable for an explicit index: masking needs no
        # inventory. The device then carries no busy verdict, and that is recorded as such.
        pick = by_idx.get(want, {"index": want, "uuid": None, "name": None,
                                 "busy": None, "util_pct": None, "mem_used_mb": None,
                                 "n_procs": None,
                                 "note": "nvidia-smi unavailable; not screened for tenants"})
    sel.update(
        {
            "index": pick["index"],
            "uuid": pick.get("uuid"),
            "name": pick.get("name"),
            "busy": pick.get("busy"),
            "util_pct": pick.get("util_pct"),
            "mem_used_mb": pick.get("mem_used_mb"),
            "n_procs": pick.get("n_procs"),
        }
    )
    if pick.get("busy"):
        sel["busy_reason"] = (
            f"GPU {pick['index']} has {pick.get('n_procs')} foreign compute process(es), "
            f"{pick.get('mem_used_mb')} MiB resident and {pick.get('util_pct')}% "
            f"utilisation"
        )
    return sel


def _gpu_spec_from_argv(argv: Sequence[str]) -> tuple[str | None, bool, float | None]:
    """`(spec, allow_busy, busy_mb)` scraped out of argv before argparse can run.

    Deliberately literal-minded: only the exact spellings `--gpu N`, `--gpu=N`,
    `--allow-busy` and `--gpu-busy-mb` are recognised.  Anything else -- including a bare
    `--gpu` with no value -- leaves the spec None and lets the driver's own argparse produce
    the error message, which is the one the operator can act on.
    """
    spec, allow_busy, busy_mb = None, False, None
    args = list(argv)
    for i, a in enumerate(args):
        if a == "--gpu" and i + 1 < len(args):
            spec = args[i + 1]
        elif a.startswith("--gpu="):
            spec = a.split("=", 1)[1]
        elif a == "--allow-busy":
            allow_busy = True
        elif a == "--gpu-busy-mb" and i + 1 < len(args):
            busy_mb = _as_num(args[i + 1], -1) or None
        elif a.startswith("--gpu-busy-mb="):
            busy_mb = _as_num(a.split("=", 1)[1], -1) or None
    if spec is None:
        spec = os.environ.get(GPU_ENV_VAR) or None
    if busy_mb is not None and busy_mb < 0:
        busy_mb = None
    return spec, allow_busy, busy_mb


def _bootstrap_gpu_selection() -> None:
    """Mask this process to one GPU, at import time, before torch exists."""
    global _GPU_SELECTION

    carried = os.environ.get(_GPU_DECISION_VAR)
    if carried:
        # We are the re-exec'd child (or a subprocess of a selected run). Adopt the parent's
        # decision verbatim rather than re-ranking: two rankings seconds apart can disagree,
        # and a child that quietly moved to a different card is the worst possible outcome.
        try:
            _GPU_SELECTION = json.loads(carried)
            return
        except Exception:  # noqa: BLE001
            pass

    spec, allow_busy, busy_mb = _gpu_spec_from_argv(sys.argv[1:])
    if spec is None:
        _GPU_SELECTION = {
            "requested": None,
            "applied": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset: all)"),
            "note": "no --gpu given; the process sees whatever devices it inherited. On a "
                    "multi-tenant host that is how a contended GPU gets measured -- see the "
                    "preflight's 40 us harness floor.",
        }
        return

    sel = select_gpu(spec, allow_busy=allow_busy, busy_mb=busy_mb)
    for line in _render_ranking(
        [dict(g) for g in sel.get("inventory", [])], sel.get("index")
    ):
        print(line, flush=True)
    if sel.get("error"):
        raise SystemExit(f"[gpu] {sel['error']}")
    if sel.get("busy"):
        msg = (
            f"[gpu] !! {sel['busy_reason']}.\n"
            f"[gpu] !! A contended device is what produced this suite's contaminated "
            f"launch/tick calibration (harness floor 40 us, tick match 3 %). Pick another "
            f"GPU, or pass --allow-busy to measure it anyway and label the result."
        )
        if not allow_busy:
            raise SystemExit(msg)
        print(msg, flush=True)
        sel["busy_override"] = True

    os.environ["CUDA_VISIBLE_DEVICES"] = str(sel["index"])
    sel["applied"] = True
    sel["cuda_visible_devices"] = str(sel["index"])
    print(
        f"[gpu] CUDA_VISIBLE_DEVICES={sel['index']} -> this process sees ONE device as "
        f"cuda:0 ({sel.get('name') or '?'}, uuid {sel.get('uuid') or '?'})",
        flush=True,
    )
    _GPU_SELECTION = sel
    try:
        os.environ[_GPU_DECISION_VAR] = json.dumps(sel)
    except Exception:  # noqa: BLE001 -- the decision is already applied; carrying it is a bonus
        pass


_bootstrap_gpu_selection()

import torch  # noqa: E402 -- MUST follow the CUDA_VISIBLE_DEVICES bootstrap above


def _verify_gpu_mask() -> None:
    """Confirm the mask actually bit, and re-exec once if some import beat us to `cuInit`.

    The bootstrap above runs before this module imports torch, which is early enough in
    every driver here.  It is not early enough in a process that had already touched CUDA
    before importing this package, and in that case the mask is a no-op that no exception
    reports.  `device_count() != 1` is the direct evidence, so it is checked rather than
    assumed, and the repair is a single re-exec with the variable already in the
    environment.  The guard variable makes that at most once.
    """
    sel = _GPU_SELECTION
    if not sel.get("applied"):
        return
    try:
        n = int(torch.cuda.device_count())
    except Exception as exc:  # noqa: BLE001
        sel["mask_verified"] = f"unknown: {type(exc).__name__}"
        return
    if n == 1:
        sel["mask_verified"] = True
        return
    sel["mask_verified"] = False
    if os.environ.get(_GPU_REEXEC_VAR):
        raise SystemExit(
            f"[gpu] CUDA_VISIBLE_DEVICES={sel['index']} did not take effect even after a "
            f"re-exec (torch still sees {n} devices). Something initialised CUDA before "
            f"this package was imported. Set CUDA_VISIBLE_DEVICES in the shell instead."
        )
    print(
        f"[gpu] mask did not bite (torch sees {n} devices) -- CUDA was already initialised "
        f"when this package was imported; re-exec'ing once with CUDA_VISIBLE_DEVICES="
        f"{sel['index']}",
        flush=True,
    )
    os.environ[_GPU_REEXEC_VAR] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)


_verify_gpu_mask()


def gpu_selection() -> dict:
    """The GPU decision this process made, for the `fairness` block of every result file."""
    return dict(_GPU_SELECTION)

from glm52_h200 import common as _common  # noqa: E402  -- hard dependency, by design

# `traffic` supplies roofline ceilings.  It is a reporting nicety, not a measurement, so a
# missing or broken module must not take a bench down with it.
try:  # noqa: SIM105
    from glm52_h200 import traffic as _traffic
except Exception as _exc:  # noqa: BLE001
    _traffic = None
    _TRAFFIC_ERR = f"{type(_exc).__name__}: {_exc}"[:200]
else:
    _TRAFFIC_ERR = None


# ======================================================================================
# Search-space POLICY constants.
#
# These are not hardware constants and must not be confused with them.  They bound how
# much per-thread state a candidate mapping may hold before it is not worth compiling, and
# they are applied to BOTH arms of every pair, so they cannot bias a ratio.  The hardware
# half of each guard (lane count, register file, shared memory) comes from `config.env()`.
# ======================================================================================
MAX_ELEMS_PER_THREAD = 64  # fp32 values a lane may hold in a VECTOR kernel's tile
MIN_ELEMS_PER_THREAD = 1  # below this the CTA is mostly idle lanes
CUDA_MAX_THREADS_PER_BLOCK = 1024  # CUDA *programming model* cap, not a device property

#: Per-thread register window.  A thread addresses R0..R255, of which R255 is the reserved
#: zero register, so 255 are usable -- an ISA constant on every NVIDIA generation from Kepler
#: to Blackwell, not a device fact, so it does not belong in `config.env()` and needs no
#: per-card probe.
REG_WINDOW_PER_THREAD = 256
#: Share of that window a GEMM's fp32 accumulator tile may claim.  The rest holds the operand
#: fragments, the addressing and the pipeline's in-flight state.  A SEARCH POLICY number,
#: applied to both arms of every pair.
ACC_REG_FRACTION = 0.5


def acc_elems_per_thread_cap() -> int:
    """fp32 accumulator elements one lane may hold in a GEMM tile.

    This used to be `MAX_ELEMS_PER_THREAD` (64), and on the H200 that single number threw
    away the device's own best mapping: the preflight measured Triton at 788 TF/s -- 96 % of
    cuBLAS -- with `BM128 BN256 BK64 num_warps=8`, which is 128 accumulator elements per
    lane.  A grid that cannot express the calibrated peak cannot report a fused kernel's
    distance from it, and a fused arm that is register-hungrier than its unfused twin is
    exactly the arm a too-tight cap keeps out of the winner's circle.

    Derived from the ISA window and the policy fraction above rather than written down, so
    the reason survives next to the number.  On any current NVIDIA part that is 128, which
    admits the calibrated winner exactly and still rejects `BM256 BN256` at 8 warps (256
    elements per lane, a guaranteed spill) unless the config also widens the CTA.
    """
    return int(REG_WINDOW_PER_THREAD * ACC_REG_FRACTION)


#: Convenience alias.  Vector kernels keep the tighter `MAX_ELEMS_PER_THREAD`; GEMM
#: accumulator guards use this.
MAX_ACC_ELEMS_PER_THREAD = acc_elems_per_thread_cap()


# ======================================================================================
# Preflight
# ======================================================================================
_PF_CACHE: dict | None = None
# Honour the same override config.py / common.py / kernels.hopper use. Without it a
# re-probe written to a side-file would leave THIS module reading the stale JSON while the
# rest of the suite read the new one -- two different feature tables in one run.
_PF_PATH = Path(os.environ.get("GLM52_H200_PREFLIGHT", str(PKG / "preflight_h200.json")))
_PF_NOTICE_DONE = False


def preflight() -> dict:
    """The preflight probe's JSON, or `{}` if it was never run.

    Loaded once.  Everything derived from it (timer tick, feature availability, calibrated
    peaks) is optional by construction: the benches degrade to "unknown" rather than to a
    guess, because a guessed timer tick silently un-flags a tick-quantised speedup.
    """
    global _PF_CACHE, _PF_NOTICE_DONE
    if _PF_CACHE is None:
        try:
            _PF_CACHE = json.loads(_PF_PATH.read_text())
        except Exception as exc:  # noqa: BLE001 -- absence is a supported state
            _PF_CACHE = {}
            if not _PF_NOTICE_DONE:
                print(
                    f"[preflight] {_PF_PATH.name} unavailable ({type(exc).__name__}); "
                    f"feature gates default to OFF and no timer tick will be recorded",
                    flush=True,
                )
                _PF_NOTICE_DONE = True
    return _PF_CACHE


def pf_get(*path, default=None):
    """`pf_get('calibration', 'timer_tick_us')` -- never raises on a missing branch."""
    node = preflight()
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node if node is not None else default


def feature(name: str) -> bool:
    """True only if the preflight COMPILED AND LAUNCHED that feature on this stack.

    Attribute existence is not evidence -- several Triton releases export
    `make_tensor_descriptor` and then fail at compile time.  Known probe names:
    `tma_tensor_descriptor`, `warp_specialize_tl_range`,
    `warp_specialize_num_consumer_groups`, `thread_block_cluster_num_ctas`, `tl_dot_bf16`.
    """
    return bool(pf_get("triton_features", "compile_probes", name, "ok", default=False))


def _pf_float(*path) -> float | None:
    v = pf_get(*path)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def timer_tick_us() -> float | None:
    """CUDA-event granularity in microseconds, as measured, or None.

    Read `calibration_status()` before believing it -- see there.
    """
    v = _pf_float("calibration", "timer_tick_us")
    return v if v else None


def launch_cost_us() -> float | None:
    v = _pf_float("calibration", "launch_us")
    return v if v else None


def harness_floor_us() -> float | None:
    v = _pf_float("calibration", "harness_floor_us")
    return v if v else None


def timer_tick_match_frac() -> float | None:
    """Fraction of preflight samples that landed on an integer multiple of the tick."""
    return _pf_float("calibration", "timer_tick_match_frac")


#: Below this, the "tick" the preflight reported is not a tick.  The detector calls a
#: quantum only at >= 0.98; anything under ~0.9 means the samples are dominated by something
#: that is not the timer, and the only such something on this host is another tenant.
TICK_MATCH_MIN = 0.9
_CALIB_WARNED = False


def calibration_status() -> dict:
    """Is the preflight's launch/tick calibration usable, and if not, why not.

    The H200 preflight reported `launch_us 8.89`, `harness_floor_us 40.55` and
    `timer_tick_us 0.256` **with a match fraction of 0.03**.  A 40 us floor on a card whose
    kernels resolve in single-digit microseconds is not physical, and a quantum that fits
    3 % of samples is not a quantum; both are what measuring a shared GPU looks like.  The
    device block of that same file says 98.8 GB free of 150 GB, i.e. ~51 GB belonged to
    someone else while the probe ran.

    So these two numbers are treated as UNKNOWN rather than as data.  The consequence that
    matters is in `tick_report`: with an invented tick, a decode cell either gets flagged
    `tick_limited` when it is not, or -- worse -- is silently NOT flagged when it is, and a
    quantised ratio goes into the report at three decimal places.  `tick_limited: null` with
    a reason is the honest answer, and it is what this returns.
    """
    frac = timer_tick_match_frac()
    free = pf_get("device", "mem_free_bytes")
    total = pf_get("device", "mem_total_bytes") or pf_get("device", "total_memory")
    out = {
        "timer_tick_us": _pf_float("calibration", "timer_tick_us"),
        "timer_tick_match_frac": frac,
        "launch_us": _pf_float("calibration", "launch_us"),
        "harness_floor_us": _pf_float("calibration", "harness_floor_us"),
        "match_frac_min": TICK_MATCH_MIN,
        "trusted": None,
        "reason": None,
    }
    if isinstance(free, (int, float)) and isinstance(total, (int, float)) and total:
        out["mem_in_use_by_others_bytes"] = int(total) - int(free)
        out["mem_in_use_by_others_frac"] = 1.0 - float(free) / float(total)
    if not preflight():
        out["reason"] = "no preflight_h200.json; nothing was calibrated"
        return out
    if frac is None:
        out["reason"] = (
            "preflight recorded no timer_tick_match_frac, so the tick cannot be validated"
        )
        return out
    if frac < TICK_MATCH_MIN:
        used = out.get("mem_in_use_by_others_bytes")
        # A low match fraction has TWO very different causes, and conflating them was wrong.
        #
        #  (a) contended GPU: the timings are noise, so no lattice fits. Signature -- a
        #      harness floor far above a launch, and/or another process holding memory.
        #  (b) a timer FINER than every candidate granularity: no lattice fits because the
        #      quantum is smaller than 0.256 us. Signature -- a sane harness floor, an idle
        #      device, and a LOW match at every candidate rather than a good one somewhere.
        #
        # Case (b) is the H200's actual behaviour on an idle card (floor 5.7 us, 149 GB
        # free, every candidate <= 0.18) and it is GOOD news: finer than the finest quantum
        # tested means quantisation is not a limiting factor for any measurement here. It
        # must not suppress verdicts or print contention warnings.
        cands = pf_get("calibration", "timer_tick_candidates", default={}) or {}
        try:
            best_any = max(float(v) for v in cands.values()) if cands else frac
        except (TypeError, ValueError):
            best_any = frac
        floor = out.get("harness_floor_us")
        floor_sane = isinstance(floor, (int, float)) and floor < 15.0
        idle = not (isinstance(used, int) and used > 2**30)
        if floor_sane and idle and best_any < TICK_MATCH_MIN:
            out["trusted"] = True
            out["timer_tick_us"] = None
            out["tick_finer_than_tested"] = True
            out["reason"] = (
                f"no quantisation lattice at any tested granularity (best match "
                f"{best_any * 100:.0f}% at {min(cands, key=lambda k: float(k)) if cands else '?'} us), "
                f"on an idle device with a sane harness floor of {floor} us. The event timer "
                f"is finer than the finest quantum probed, so timer quantisation is NOT a "
                f"limiting factor and no cell is tick-limited. This is the good outcome; "
                f"contrast the RTX 4060, where 200/200 samples landed exactly on 1.024 us."
            )
            return out
        out["trusted"] = False
        out["reason"] = (
            f"timer_tick_match_frac={frac:.2f} < {TICK_MATCH_MIN}: the reported "
            f"{out['timer_tick_us']} us 'tick' matches only {frac * 100:.0f}% of samples, "
            f"and harness_floor_us={out['harness_floor_us']} is far above a launch. That is "
            f"a contended GPU, not a timer"
            + (
                f" -- {used / 2**30:.0f} GB of this device was already allocated by another "
                f"process when the preflight ran"
                if isinstance(used, int) and used > 2**30 else ""
            )
            + ". Re-run the preflight on an IDLE device -- `CUDA_VISIBLE_DEVICES=<n> "
              "python3 glm52_h200/preflight.py` works whatever flags that script grows -- "
              "and every tick_limited verdict below becomes answerable."
        )
        return out
    # The harness floor is an INDEPENDENT contention signal and must be checked even when
    # the tick is clean. Structuring it inside the low-tick branch (as this did) made it
    # unreachable exactly when the tick matched perfectly -- which is how a run with
    # harness_floor_us=39.87 against config.FLOOR_US_MAX=20.0 recorded `trusted: true`.
    # config.py already owns these thresholds; do not restate them here.
    floor = out.get("harness_floor_us")
    launch = out.get("launch_us")
    try:
        from glm52_h200 import config as _C  # local: config is another module's to own
    except Exception:  # noqa: BLE001 -- a missing config must not silence the guard
        _C = None
    fmax = getattr(_C, "FLOOR_US_MAX", 20.0)
    rmax = getattr(_C, "FLOOR_LAUNCH_RATIO_MAX", 3.0)
    if isinstance(floor, (int, float)):
        why = None
        if floor > fmax:
            why = (f"harness_floor_us={floor:.3f} exceeds config.FLOOR_US_MAX={fmax}: a clean "
                   f"floor on an idle device is single-digit microseconds")
        elif isinstance(launch, (int, float)) and launch > 0 and floor > rmax * launch:
            why = (f"harness_floor_us={floor:.3f} is {floor / launch:.1f}x launch_us="
                   f"{launch:.3f}, above config.FLOOR_LAUNCH_RATIO_MAX={rmax}")
        if why:
            out["trusted"] = False
            out["reason"] = (
                why + ". That is the signature of a co-tenant on the GPU, not a timer "
                "property, and it inflates every absolute millisecond in this file. Ratios "
                "measured by an interleaved paired loop survive it; sequential ratios and "
                "any ceiling comparison do not."
            )
            return out
    out["trusted"] = True
    return out


def warn_calibration_once() -> None:
    """One line, once per process, when the tick/launch calibration cannot be believed."""
    global _CALIB_WARNED
    if _CALIB_WARNED:
        return
    _CALIB_WARNED = True
    st = calibration_status()
    if st.get("trusted") is False:
        print(f"[calib] !! {st['reason']}", flush=True)
    elif st.get("trusted") is None and st.get("reason"):
        print(f"[calib] {st['reason']}", flush=True)


def check_preflight_device(env) -> dict:
    """Refuse to run if the preflight on disk describes a different GPU.

    This is the same failure the 4060 port nearly shipped in its checkpoints, one level up:
    a `preflight_h200.json` carried over from another machine would supply that machine's
    timer tick, L2 size and feature list to a result file whose `env` block correctly
    identifies *this* device.  The two must agree or nothing below is trustworthy.
    """
    pf_name = pf_get("device", "name")
    info = {
        "preflight_present": bool(preflight()),
        "preflight_device": pf_name,
        "live_device": getattr(env, "device_name", None),
        "preflight_timestamp": pf_get("timestamp"),
    }
    if pf_name and getattr(env, "device_name", None) and pf_name != env.device_name:
        raise RuntimeError(
            f"preflight_h200.json describes {pf_name!r} but this device is "
            f"{env.device_name!r}.\nThat file supplies the timer tick, the feature gates "
            f"and the calibrated peaks used to interpret every number this bench writes. "
            f"Re-run `python3 glm52_h200/preflight.py` on THIS machine, or delete the "
            f"stale JSON -- do not mix them."
        )
    cc = pf_get("device", "compute_capability")
    if cc and cc != "9.0":
        info["compute_capability_warning"] = (
            f"preflight recorded sm_{cc.replace('.', '')}, not sm_90; H200-specific "
            f"feature gates were evaluated on that device"
        )
        print(f"[preflight] !! {info['compute_capability_warning']}", flush=True)
    info["compute_capability"] = cc
    info["features"] = {
        k: bool(v.get("ok"))
        for k, v in (pf_get("triton_features", "compile_probes", default={}) or {}).items()
        if isinstance(v, dict)
    }
    return info


# ======================================================================================
# Device constants -- one choke point, and it raises instead of guessing
# ======================================================================================
def env_int(env, name: str) -> int:
    """A hardware constant from the probe.  Missing or zero is fatal.

    The C500 study baked `warp 64`, `104 SMs`, `SMEM 65536`, `regs 131072` into guards.
    On another device those silently prune the autotuning grid -- and not symmetrically
    across a fused/unfused pair, which is the one failure mode that fabricates a result.
    So there is no default here at all: if the probe did not answer, the run stops.
    """
    v = getattr(env, name, None)
    if v is None:
        extras = getattr(env, "extras", {}) or {}
        v = extras.get(name)
    try:
        v = int(v)
    except (TypeError, ValueError):
        v = 0
    if v <= 0:
        raise RuntimeError(
            f"device probe did not supply {name!r} (got {getattr(env, name, None)!r}).\n"
            f"Refusing to build a search grid from a hardware constant this process had "
            f"to invent; fix config.BenchEnv.probe() instead."
        )
    return v


def max_threads_per_block(env) -> int:
    """Largest CTA a launch may request.

    Preferred order: an explicit probe field, then Triton's property table, then the CUDA
    programming-model cap clamped by threads/SM.  The last branch is the only derived one
    and it is derived from a *specification* limit (1024 threads/block is an API constant
    on every CUDA device), not from a remembered device.
    """
    for src in (env, getattr(env, "extras", {}) or {}):
        for key in ("max_threads_per_block", "max_threads_per_multi_processor_block",
                    "maxThreadsPerBlock"):
            v = src.get(key) if isinstance(src, dict) else getattr(src, key, None)
            if isinstance(v, int) and 0 < v <= 4096:
                return v
    return min(CUDA_MAX_THREADS_PER_BLOCK, env_int(env, "threads_per_sm"))


def warp_ladder(env, lo: int = 1) -> list[int]:
    """Every power-of-two warp count a CTA can legally hold on THIS device.

    On C500 (64 lanes) this tops out at 16; on sm_89/sm_90 (32 lanes) it reaches 32.  The
    C500 study hardcoded 16, which cost the widest tiles 30-40 % of their configs on a
    32-lane device -- the sort of one-sided truncation that moves a ratio.
    """
    warp = env_int(env, "warp_size")
    cap = max_threads_per_block(env)
    out, w = [], lo
    while w * warp <= cap:
        out.append(w)
        w *= 2
    return out or [1]


def sm_wave_caps(env, mults: Sequence[int] = (1, 2, 4, 8, 16)) -> list[int]:
    """Persistent-grid rungs in whole waves of the device.

    A persistent grid is only meaningfully evaluated at a balanced size, so the rungs are
    multiples of the SM count rather than round numbers: 132*m on an H200, 24*m on a 4060,
    104*m on C500.
    """
    sm = env_int(env, "num_sm")
    return [sm * m for m in mults]


def elems_per_program_cap(env) -> int:
    """Ceiling on elements a single program may hold in registers, derived.

    = (largest CTA) x (per-thread element budget).  The C500 code wrote this as the literal
    65536, which is exactly `1024 threads * 64 elem/thread` on that device -- correct there,
    an accident anywhere else.
    """
    return max_threads_per_block(env) * MAX_ELEMS_PER_THREAD


# ======================================================================================
# Shared memory: the model, CALIBRATED against what this stack actually reserved
# ======================================================================================
_SMEM_FIT: dict | None = None


def smem_stage_fit() -> dict:
    """How many `2*BK*(BM+BN)` buffers this stack really stages, fitted to the preflight.

    `config.smem_stage_bytes` uses `max(2, num_stages - 1)`, measured on **triton 3.6 /
    sm_89**.  The H200 preflight's own `smem_probe` block says that is one buffer short
    here -- every observation is `num_stages` exactly:

        BM128 BN128 BK64 s3 -> 98304  = 3 * 2*64*(128+128)
        BM128 BN256 BK64 s3 -> 147456 = 3 * 2*64*(128+256)
        BM128 BN256 BK64 s4 -> 196608 = 4 * 2*64*(128+256)
        BM256 BN256 BK64 s3 -> 196608 = 3 * 2*64*(256+256)
        BM128 BN256 BK128 s3 -> "Required: 294912" = 3 * 2*128*(128+256)   [did not fit]

    Under-predicting is the benign direction (the config is offered, fails to compile, lands
    in `n_failed` where the fairness check sees it) but it is not free: on a grid that now
    reaches BK=128 at BN=256 it offers a whole tile family that cannot exist, and pays two
    arms' compile time to find out.  Fitting the offset makes the prediction exact, so the
    grids reach every shape the device can run and no shape it cannot.

    The fit is only adopted when it reproduces EVERY observation exactly.  Otherwise this
    returns the conservative `config` model, because a model that over-predicts prunes legal
    configs silently, and silent pruning is the one failure this suite cannot detect later.
    """
    global _SMEM_FIT
    if _SMEM_FIT is not None:
        return _SMEM_FIT
    obs = pf_get("calibration", "smem_probe", default={}) or {}
    parsed = []
    for key, rec in obs.items():
        if not isinstance(rec, dict):
            continue
        try:
            parts = dict(
                (p[:2].lower(), int(p[2:])) if p[0] == "B" else ("s", int(p[1:]))
                for p in key.split("_")
            )
            bm, bn, bk, st = parts["bm"], parts["bn"], parts["bk"], parts["s"]
        except Exception:  # noqa: BLE001 -- an unparsable key is preflight's business
            continue
        actual = rec.get("shared_bytes")
        if not (isinstance(actual, int) and actual > 0):
            # A failed config still reports what it needed; that is an observation too.
            import re

            m = re.search(r"Required:\s*(\d+)", str(rec.get("error", "")))
            actual = int(m.group(1)) if m else None
        if not actual:
            continue
        unit = 2 * bk * (bm + bn)
        if unit <= 0 or actual % unit:
            parsed = []  # a non-integer buffer count means the shape of the model is wrong
            break
        parsed.append((st, actual // unit))
    fit = {
        "observations": len(parsed),
        "source": "config.smem_stage_bytes (triton 3.6 / sm_89)",
        "offset": None,
        "formula": "max(2, num_stages - 1) * 2*BK*(BM+BN)",
    }
    offsets = {st - buffers for st, buffers in parsed}
    if parsed and len(offsets) == 1:
        off = offsets.pop()
        if 0 <= off <= 1:
            fit.update(
                {
                    "source": f"fitted to {len(parsed)} preflight smem_probe observations",
                    "offset": off,
                    "formula": f"max(2, num_stages - {off}) * 2*BK*(BM+BN)",
                }
            )
    _SMEM_FIT = fit
    return fit


def smem_predict(bm: int, bn: int, bk: int, num_stages: int, bn_mult: int = 1) -> int:
    """Bytes of shared memory a Triton GEMM mainloop stages, per `smem_stage_fit()`."""
    fit = smem_stage_fit()
    off = fit.get("offset")
    if off is None:
        from glm52_h200 import config as _C  # local: config is another agent's module

        return _C.smem_stage_bytes(bm, bn, bk, num_stages, bn_mult=bn_mult)
    return max(2, num_stages - int(off)) * 2 * bk * (bm + bn_mult * bn)


def smem_fits(env, bm: int, bn: int, bk: int, num_stages: int, bn_mult: int = 1) -> bool:
    """Does this tile fit THIS device's opt-in shared-memory ceiling?"""
    return smem_predict(bm, bn, bk, num_stages, bn_mult) <= env_int(env, "smem_bytes")


def tile_ladder(env, lo: int = 16, bk: int = 64, num_stages: int = 3,
                bn_mult: int = 1) -> list[int]:
    """Power-of-two BM/BN values a square tile can reach inside the SMEM ceiling.

    Derived, never written down.  The two previous devices in this study had 64 KB (C500)
    and 101 KB (sm_89) of opt-in shared memory and their grids topped out accordingly; the
    H200 has 232448 B, which is why `BM256 BN256 BK64 s3` (196608 B) compiles here and
    nowhere before.  A grid that still stops at the old ceiling would under-search the new
    device -- symmetrically, so not a fairness bug, but it would leave the headline speedups
    measured against a handicapped pair of arms.
    """
    out, t = [], lo
    while t <= 1024 and smem_fits(env, t, t, bk, num_stages, bn_mult):
        out.append(t)
        t *= 2
    return out or [lo]


def bk_ladder(env, lo: int = 32, hi: int = 256, ref: int = 64,
              num_stages: int = 2) -> list[int]:
    """Power-of-two BLOCK_K values worth ENUMERATING on this device.

    Referenced against a small square tile (`ref` x `ref`) at the shallowest useful pipeline
    on purpose: this decides which values a grid generator *offers*, and the per-config SMEM
    filter then rejects the (BM, BN, BK, stages) combinations that do not fit.  Referencing a
    large tile instead would delete BK=128 from the enumeration entirely because it does not
    fit at `BM128 BN256` -- and with it every legal BK=128 config at a smaller tile, in both
    arms, invisibly.

    `hi` is a search POLICY bound, not a hardware one: past a few hundred the k-loop trip
    count collapses and there is nothing left for the mainloop to pipeline against.
    """
    out, k = [], lo
    while k <= hi and smem_fits(env, ref, ref, k, num_stages):
        out.append(k)
        k *= 2
    return out or [lo]


def exact_fp32_matmul() -> dict:
    """Turn TF32 off so the fp32 reference is actually fp32.

    This mattered nowhere in the C500 study (MACA has no TF32 path) and matters here: an
    H200's fp32 `torch.matmul` defaults to TF32 tensor cores, i.e. ~10 bits of mantissa.
    Every reference in this suite is computed in fp32 and compared at 2e-2; a TF32
    reference would move the goalposts for both arms and mask a genuinely wrong kernel.
    """
    state = {}
    for path, val in (("torch.backends.cuda.matmul.allow_tf32", False),
                      ("torch.backends.cudnn.allow_tf32", False)):
        try:
            obj = torch
            *mods, attr = path.split(".")[1:]
            for m in mods:
                obj = getattr(obj, m)
            state[path] = getattr(obj, attr, None)
            setattr(obj, attr, val)
        except Exception as exc:  # noqa: BLE001 -- newer torch renames these
            state[path] = f"unavailable: {type(exc).__name__}"
    # torch >= 2.9 spells it as a precision string; set both, keep whichever exists
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        state["fp32_precision"] = "ieee"
    except Exception:  # noqa: BLE001
        pass
    return state


def l2_flush_audit(env) -> dict:
    """Record whether `common`'s L2-flush buffer really exceeds THIS device's L2.

    H200's L2 is ~50 MB.  A flush buffer sized for an 8 MB cache turns every measurement
    into a warm-cache one, which flatters whichever arm re-reads an intermediate -- i.e. it
    flatters the unfused side and understates fusion, silently.  The buffer is `common`'s
    to size; this only checks it and puts the answer in the result file.
    """
    l2 = env_int(env, "l2_bytes")
    got = getattr(_common, "_FLUSH_BYTES", None)
    audit = {"l2_bytes": l2, "flush_bytes": got, "required_min_bytes": 4 * l2}
    if isinstance(got, int):
        audit["ok"] = got >= 4 * l2
        if not audit["ok"]:
            print(
                f"[l2] !! flush buffer {got / 2**20:.0f} MB is under 4x this device's L2 "
                f"({l2 / 2**20:.0f} MB) -- measurements may be warm-cache",
                flush=True,
            )
    else:
        audit["ok"] = None
        audit["note"] = "common._FLUSH_BYTES not exposed; could not audit"
    return audit


# ======================================================================================
# H200 mapping axes -- runtime-gated, never authored in
# ======================================================================================
#: Cluster sizes worth offering.  `num_ctas` does NOT change the grid a launcher passes:
#: Triton multiplies gridDimX internally, so one program becomes one cluster of this many
#: CTAs.  Kept to a single rung by default because every overlay multiplies BOTH arms' coarse
#: grid, and a 2-CTA cluster is where DSMEM first exists at all.
CLUSTER_SIZES = (2,)

_CAPS_CACHE: object | None = None
#: One line per kernel module (or per-kernel axis view), so a driver that widens two
#: different grids reports both instead of only the first.
_AXIS_NOTICE_DONE: set = set()


def hopper_caps():
    """`kernels.hopper.caps()`, or None if that module cannot answer.

    This is the LIVE verdict -- a trial compile-and-launch of each mechanism, run in a
    subprocess -- and it outranks `preflight_h200.json` for exactly one reason that matters
    here.  The preflight's `tma_tensor_descriptor` probe passes a HOST-side
    `TensorDescriptor` object into `tl.make_tensor_descriptor()`, which is the DEVICE-side
    constructor and wants a raw pointer.  Mixing the two APIs is a `CompilationError` on any
    hardware, so that probe's `ok: false` on the H200 is a **false negative about the
    device**.  Gating TMA on it would disable TMA on the one machine this suite exists to
    measure, and the result file would still say "H200".
    """
    global _CAPS_CACHE
    if _CAPS_CACHE is None:
        try:
            from glm52_h200.kernels import hopper as _hop

            _CAPS_CACHE = _hop.caps()
        except Exception as exc:  # noqa: BLE001 -- no caps module, no Hopper axes
            _CAPS_CACHE = False
            print(f"[h200] kernels/hopper.py unavailable ({type(exc).__name__}: {exc}); "
                  f"no sm_90 mapping axis will be offered", flush=True)
    return _CAPS_CACHE or None


def axis_available(name: str) -> tuple[bool, str]:
    """`(available, why)` for one sm_90 mapping axis, from the live probe then the preflight.

    `name` is `tma`, `warp_specialize` or `clusters`.  The two sources disagree on TMA by
    construction (see `hopper_caps`), so the disagreement is reported in the string rather
    than resolved silently.
    """
    probe = {
        "tma": "tma_tensor_descriptor",
        "warp_specialize": "warp_specialize_tl_range",
        "clusters": "thread_block_cluster_num_ctas",
    }.get(name, "")
    pf = feature(probe) if probe else False
    c = hopper_caps()
    if c is None:
        return bool(pf), f"preflight probe {probe}={pf}; no live capability module"
    live = bool(getattr(c, name, False))
    src = (getattr(c, "sources", {}) or {}).get(name, "?")
    why = f"hopper.caps().{name}={live} (source {src}); preflight probe {probe}={pf}"
    if live and not pf:
        why += " -- LIVE PROBE WINS: the preflight probe for this axis is inconclusive"
    return live, why


def h200_cfg_overlays(kernel_mod=None) -> list[dict]:
    """Extra config-dict overlays this device+stack+kernel actually supports.

    Two independent gates, both checked at RUNTIME:

      1. a live trial compile+launch says the mechanism works on this stack, and
      2. the kernel module advertises the cfg key by exporting `H200_CFG_KEYS`
         (a tuple of strings its launcher forwards to the Triton launch).

    If either is missing the overlay list is empty and the grids below are byte-identical
    to the classic ones.  That is the whole point: this file cannot be tested on sm_90, so
    it must be incapable of *requiring* sm_90.

    Overlays are applied to the coarse grid of BOTH arms of a pair, so they cannot bias a
    ratio; they only widen the search for both.  `USE_TMA + warp specialization` is offered
    as a combination as well as separately, because that pairing is the mechanism the
    "free normalization" claim rests on -- descriptor loads issued by producer warps while
    consumer warps run the MMA -- and neither half alone tests it.
    """
    keys = tuple(getattr(kernel_mod, "H200_CFG_KEYS", ()) or ())
    if not keys:
        return []

    # The warp-specialization cfg key is spelled differently by different kernel modules
    # (`warp_specialize` where the launcher forwards it, `WARP_SPECIALIZE` where it is a
    # kernel constexpr). Emit the one the module actually advertises; emitting the other
    # would be silently ignored, and a row labelled warp-specialized that ran the classic
    # mainloop is a fabricated measurement.
    ws_key = next((k for k in ("WARP_SPECIALIZE", "warp_specialize") if k in keys), None)
    tma_key = "USE_TMA" if "USE_TMA" in keys else None

    out: list[dict] = []
    ws_ovl: dict | None = None
    tma_ovl: dict | None = None

    if "num_ctas" in keys and axis_available("clusters")[0]:
        out += [{"num_ctas": int(n)} for n in CLUSTER_SIZES if int(n) > 1]
    if ws_key and axis_available("warp_specialize")[0]:
        ws_ovl = {ws_key: True}
        out.append(ws_ovl)
    elif ("num_consumer_groups" in keys
          and feature("warp_specialize_num_consumer_groups")):
        # The forked-Triton spelling. The measured H200 stack REJECTS it outright
        # ("Keyword argument num_consumer_groups was specified but unrecognised"), so this
        # branch exists only for a stack that has it and lacks tl.range(warp_specialize=).
        ws_ovl = {"num_consumer_groups": 1, "num_buffers_warp_spec": 2}
        out.append(ws_ovl)
    if tma_key and axis_available("tma")[0]:
        tma_ovl = {tma_key: True}
        out.append(tma_ovl)
    if ws_ovl and tma_ovl:
        out.append({**tma_ovl, **ws_ovl})
    return out


def _mod_name(kernel_mod) -> str:
    """A printable name for a kernel module OR for a per-kernel axis view object.

    Some modules advertise DIFFERENT axes per kernel (`K.ROUTER_AXES` vs `K.MOE_AXES`) via a
    small duck-type carrying only `H200_CFG_KEYS`.  Those have no `__name__`, and a report
    that labels them all "?" cannot be traced back to the kernel it describes.
    """
    n = getattr(kernel_mod, "__name__", None)
    if n:
        return str(n)
    if kernel_mod is None:
        return "(no kernel module)"
    return f"{type(kernel_mod).__module__}.{type(kernel_mod).__name__}"


def h200_axis_report(kernel_mod=None) -> dict:
    """What each sm_90 axis is, why, and which cfg key carries it -- for the result file."""
    keys = tuple(getattr(kernel_mod, "H200_CFG_KEYS", ()) or ())
    rep: dict = {
        "advertised_by": _mod_name(kernel_mod),
        "kernel_cfg_keys": list(keys) or "module advertises none",
        "axes": {},
        "overlays_offered": h200_cfg_overlays(kernel_mod),
    }
    for name, cfgkeys in (
        ("tma", ("USE_TMA",)),
        ("warp_specialize", ("WARP_SPECIALIZE", "warp_specialize",
                             "num_consumer_groups")),
        ("clusters", ("num_ctas",)),
    ):
        ok, why = axis_available(name)
        carried = [k for k in cfgkeys if k in keys]
        rep["axes"][name] = {
            "available": ok,
            "evidence": why,
            "kernel_key": carried[0] if carried else None,
            "offered": bool(ok and carried),
            "not_offered_because": (
                None if (ok and carried)
                else ("this kernel module advertises no cfg key for it"
                      if ok else "the live capability probe says it is unavailable")
            ),
        }
    c = hopper_caps()
    if c is not None:
        rep["tma_form"] = c.tma_form() if hasattr(c, "tma_form") else None
        rep["ws_mode"] = getattr(c, "ws_mode", None)
        rep["probe_mode"] = getattr(c, "probe_mode", None)
    return rep


def widen(grid: list[dict], kernel_mod=None, cap: int = 0, tag: str = "") -> list[dict]:
    """`grid` plus every H200 overlay of every config in it, deduplicated.

    `cap`, when non-zero, samples the widened list back down to a trial budget.  Widening
    multiplies a grid by `1 + len(overlays)` -- on the measured H200 that is 5x -- and the
    budget has to be spent on the widened space rather than on the classic one, or the new
    axes are only ever tried at whatever tiles happened to survive an earlier cap.  Both
    arms are handed the same widened, same-sampled list, so this cannot bias a ratio.
    """
    ovl = h200_cfg_overlays(kernel_mod)
    seen_key = _mod_name(kernel_mod)
    if not ovl:
        if seen_key not in _AXIS_NOTICE_DONE:
            _AXIS_NOTICE_DONE.add(seen_key)
            print(f"    [h200 axes] none offered for {seen_key}: grids are the classic ones",
                  flush=True)
        return cap_grid(list(grid), cap, tag) if cap else list(grid)
    if seen_key not in _AXIS_NOTICE_DONE:
        _AXIS_NOTICE_DONE.add(seen_key)
        print(
            f"    [h200 axes] {seen_key}: offering {len(ovl)} overlay(s) to BOTH arms: "
            + ", ".join(
                "+".join(f"{k}={v}" for k, v in o.items()) for o in ovl
            ),
            flush=True,
        )
    out = list(grid)
    for c in grid:
        for o in ovl:
            out.append(dict(c, **o))
    out = dedup(out)
    return cap_grid(out, cap, tag) if cap else out


def axis_counts(*grids) -> dict:
    """Live per-axis counts over one or more config lists.

    This is what turns "the fused arm was offered warp specialization" from a claim into a
    number.  Where an axis is structurally meaningful for only one arm (a kernel with no
    mainloop cannot be warp-specialized), the counts show it as a zero on the other side
    instead of leaving the asymmetry to be inferred from prose.
    """
    axes = ("USE_TMA", "TMA_A", "TMA_B", "TMA_MODE", "WARP_SPECIALIZE",
            "warp_specialize", "num_consumer_groups", "num_ctas")
    out: dict = {}
    for g in grids:
        for cfg in g or ():
            for a in axes:
                if a not in cfg:
                    continue
                v = cfg[a]
                if a == "num_ctas" and int(v or 1) <= 1:
                    continue
                if isinstance(v, bool) and not v:
                    continue
                out[a] = out.get(a, 0) + 1
    out["_total_cfgs"] = sum(len(g or ()) for g in grids)
    return out


# ======================================================================================
# Small grid utilities (identical semantics to the C500 suite's copies)
# ======================================================================================
def dedup(cfgs: Iterable[dict]) -> list[dict]:
    seen, out = set(), []
    for c in cfgs:
        key = tuple(sorted((k, str(v)) for k, v in c.items()))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def top_cfgs(*tables, k: int = 3) -> list[dict]:
    """The k fastest distinct configs across one or more autotune tables."""
    rows = []
    for tb in tables:
        if tb is None:
            continue
        it = tb.table if hasattr(tb, "table") else tb
        rows += [(ms, cfg) for cfg, ms, _err in it if ms is not None]
    rows.sort(key=lambda t: t[0])
    seen, out = set(), []
    for _ms, cfg in rows:
        key = tuple(sorted((kk, str(vv)) for kk, vv in cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
        if len(out) == k:
            break
    return out


def cap_grid(grid: list[dict], cap: int, tag: str = "", seed: int = 20260803) -> list[dict]:
    """Deterministic uniform sample of a legal grid down to a trial budget.

    H200's shared-memory ceiling is ~3.5x C500's and ~2.3x sm_89's, so the SMEM prefilter
    that used to do the pruning now rejects almost nothing: the grouped-GEMM grids come out
    at 500+ legal configs, and 500 compiles x 2 arms x 7 regimes on 12 GB of expert weights
    is a run nobody will wait for.  A LARGER legal space is not licence to search less
    carefully -- it is a budget problem -- so the sample is:

      * uniform over the whole legal set (not a stride of a product order, which would
        correlate with whichever axis varies slowest),
      * seeded, so two arms asked for the same cap of the same list get the same list, and
      * order-preserving, so `quick_slice` and the refine seeds behave predictably.

    Both the legal count and the sampled count go into `fairness.grids`, because "the fused
    arm searched 140 of 320 legal configs" is exactly the kind of fact that has to survive
    into the result file.
    """
    if cap <= 0 or len(grid) <= cap:
        return list(grid)
    import random

    rng = random.Random(seed)
    keep = sorted(rng.sample(range(len(grid)), cap))
    out = [grid[i] for i in keep]
    if tag:
        print(f"    [grid {tag}] {len(grid)} legal -> {len(out)} sampled (budget {cap})",
              flush=True)
    return out


def quick_slice(grid: list, keep: int) -> list:
    """Stride a grid down to ~`keep` entries for --quick, preserving its first element.

    Striding (rather than truncating) keeps the sample spread over every axis, so a quick
    run is a coarse version of the real search rather than a corner of it.
    """
    if keep <= 0 or len(grid) <= keep:
        return list(grid)
    step = max(1, len(grid) // keep)
    out = grid[::step]
    if grid and grid[0] not in out:
        out = [grid[0]] + out
    return out[:keep] if len(out) > keep else out


# ======================================================================================
# Numerical screening
# ======================================================================================
def screen(tag: str, run: Callable[[dict], object], verify: Callable[[], tuple],
           grid: list[dict], max_report: int = 4) -> tuple[list[dict], list[tuple]]:
    """Run every config once and check its OUTPUT before it is allowed into a timing grid.

    A miscompiled reduction is a wrong answer, not a crash: MACA Triton 3.0 returned
    per-warp partial `tl.max`/`tl.argmax` over a `tl.dot` accumulator whenever the mma tile
    spanned more than one warp-row.  Nothing says Hopper's wgmma path is immune to its own
    version of that, and a wrong config that happens to be fast becomes the reported
    winner.  Rejections are recorded per arm, so an asymmetric screen-out is visible after
    the fact rather than invisible.

    `verify()` returns `(ok: bool, detail)`.
    """
    ok, rej = [], []
    for cfg in grid:
        try:
            run(cfg)
            torch.cuda.synchronize()
            good, detail = verify()
            if good:
                ok.append(cfg)
            else:
                rej.append((cfg, None, f"NUMERIC {detail}"))
        except Exception as exc:  # noqa: BLE001 -- a compile failure is data, not an abort
            rej.append((cfg, None, f"{type(exc).__name__}: {exc}"[:160]))
    num = [r for r in rej if str(r[2]).startswith("NUMERIC")]
    print(
        f"    [screen {tag:<12}] {len(grid):>3} offered -> {len(ok):>3} valid "
        f"({len(rej) - len(num)} compile-fail, {len(num)} wrong-answer)",
        flush=True,
    )
    for cfg, _, why in num[:max_report]:
        print(f"        wrong-answer: {cfg} {str(why)[:110]}", flush=True)
    return ok, rej


class ScreenRejectedAll(RuntimeError):
    """Every config of one kernel failed the correctness/compile screen.

    Carried as a typed exception so a caller can record the kernel as unmeasurable and move
    on -- the user's decision was "fatal for that kernel, record and continue", not "fatal
    for the regime" (which is what destroyed every #11b result) and not "publish a number
    anyway" (which is what produced a timing for a kernel that never compiled).
    """

    def __init__(self, tag, n_offered, rejects, top_reasons):
        self.tag, self.n_offered = tag, n_offered
        self.rejects, self.top_reasons = rejects, top_reasons
        super().__init__(
            f"[{tag}] screening rejected all {n_offered} configs; "
            + "; ".join(f"{n}x {m}" for m, n in top_reasons))


def screened_autotune(tag, make_chain, grid, verify, warmup, rep, prep=None):
    """`screen` then `common.autotune`, with the screen folded into n_tried/n_failed.

    The returned TuneResult reports the OFFERED grid size and the total rejects, so the
    per-arm fairness accounting counts what the arm was actually given, not what survived.
    """
    t0 = time.time()
    if prep is not None:
        prep()
    # `make_chain(c)` may return a bare callable OR a sequence of them. Iterating it
    # directly raised `TypeError: 'function' object is not iterable` for f11's two
    # bare-callable sites (prob.norm_fn, prob.rstd_fn), which rejected EVERY config for a
    # reason that had nothing to do with the kernel -- and then the fallback below timed
    # them anyway. Normalise through the same helper the timing path uses.
    ok, rej = screen(tag, lambda c: [f() for f in _common._as_chain(make_chain(c))],
                     verify, grid)
    if not ok:
        # An all-rejected screen is FATAL FOR THIS KERNEL. The previous behaviour timed the
        # unscreened grid and reported a number -- `[rstd] 124/124 cfgs timed -> 0.0345 ms`
        # for a kernel where nothing compiled -- and additionally reset `rej = []`, deleting
        # the evidence of why. A figure with nothing behind it is worse than a gap.
        reasons = {}
        for entry in rej[:200]:
            msg = str(entry[-1] if isinstance(entry, (tuple, list)) else entry)
            reasons[msg[:120]] = reasons.get(msg[:120], 0) + 1
        top = sorted(reasons.items(), key=lambda kv: -kv[1])[:3]
        print(f"    !! [{tag}] screening rejected ALL {len(grid)} configs -- recording as "
              f"UNMEASURABLE, publishing no timing", flush=True)
        for msg, n in top:
            print(f"       {n:>4}x {msg}", flush=True)
        raise ScreenRejectedAll(tag, len(grid), rej, top)
    # `refine=False`: `common.autotune`'s default refine calls `neighbours()`, which invents
    # configs one lattice step from the winner. Those go through NEITHER `screen()` nor the
    # bench's `_ok()` filter, and `if r_ms < best_ms` can promote one to the returned winner.
    # That is how a wrong-answer config became campaign 1's tuned winner while the screen
    # correctly rejected its siblings. Refinement must be screened to be trusted; until it
    # is, the coarse grid is the whole search.
    tr = _common.autotune(make_chain, ok, warmup, rep, refine=False)
    tr.table = list(tr.table) + rej
    tr.n_tried = len(grid)
    tr.n_failed = len(rej) + tr.n_failed
    print(
        f"    [{tag:<12}] {len(ok):>3}/{len(grid)} cfgs timed -> {tr.best_ms:.4f} ms  "
        f"{tr.best_cfg}  [{time.time() - t0:.0f}s]",
        flush=True,
    )
    return tr


# ======================================================================================
# Interleaved paired timing
# ======================================================================================
@dataclass
class _LocalTiming:
    p50_ms: float
    p10_ms: float
    p90_ms: float
    mean_ms: float
    n: int
    noflush_p50_ms: float = float("nan")

    def as_dict(self) -> dict:
        return {
            "p50_ms": self.p50_ms, "p10_ms": self.p10_ms, "p90_ms": self.p90_ms,
            "mean_ms": self.mean_ms, "n": self.n,
            "noflush_p50_ms": self.noflush_p50_ms,
        }


def _mk_timing(samples: list[float]) -> object:
    xs = sorted(samples)
    n = len(xs)
    kw = dict(
        p50_ms=xs[n // 2],
        p10_ms=xs[max(0, int(0.1 * n))],
        p90_ms=xs[min(n - 1, int(0.9 * n))],
        mean_ms=statistics.fmean(xs),
        n=n,
    )
    cls = getattr(_common, "Timing", None)
    if cls is not None:
        try:
            return cls(**kw)
        except Exception:  # noqa: BLE001 -- field set differs; local shape is equivalent
            pass
    return _LocalTiming(**kw)


def _flush_l2() -> None:
    fn = getattr(_common, "_flush_l2", None)
    if fn is not None:
        fn()
        return
    global _LOCAL_FLUSH
    if _LOCAL_FLUSH is None:
        l2 = torch.cuda.get_device_properties(0).L2_cache_size
        _LOCAL_FLUSH = torch.empty(
            max(4 * l2, 256 * 2**20) // 4, dtype=torch.int32, device="cuda"
        )
    _LOCAL_FLUSH.zero_()


_LOCAL_FLUSH: torch.Tensor | None = None


def _run(fns) -> None:
    if callable(fns):
        fns()
        return
    for f in fns:
        f()


def _local_bench_pair(a_fns, b_fns, warmup: int, rep: int) -> tuple:
    """A/B/B/A interleaved timing of two chains, with a paired ratio per round.

    Each round times A once and B once, each preceded by its own L2 flush (so both arms see
    a cold cache, exactly as `bench_chain` gives a single chain).  The order flips on odd
    rounds, so neither arm systematically inherits the other's cache or clock state.

    The reported speedup is the MEDIAN OF PER-ROUND RATIOS, not the ratio of two medians.
    Any drift that is monotone within a round -- thermal, clock, power -- multiplies both
    arms by nearly the same factor and cancels in the ratio.  This is the fix for the 4060
    run where the fused arm measured 137 ms during tuning and 167 ms in the final block of
    the SAME run, producing a headline speedup above the cell's own physical ceiling.
    """
    for _ in range(max(1, warmup)):
        _run(a_fns)
        _run(b_fns)
    torch.cuda.synchronize()

    ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
    sa = [ev() for _ in range(rep)]
    ea = [ev() for _ in range(rep)]
    sb = [ev() for _ in range(rep)]
    eb = [ev() for _ in range(rep)]

    def time_a(i):
        _flush_l2()
        sa[i].record()
        _run(a_fns)
        ea[i].record()

    def time_b(i):
        _flush_l2()
        sb[i].record()
        _run(b_fns)
        eb[i].record()

    for i in range(rep):
        if i % 2 == 0:
            time_a(i)
            time_b(i)
        else:
            time_b(i)
            time_a(i)
    torch.cuda.synchronize()

    ta = [s.elapsed_time(e) for s, e in zip(sa, ea)]
    tb = [s.elapsed_time(e) for s, e in zip(sb, eb)]
    ratios = [b / a for a, b in zip(ta, tb) if a > 0]
    ratios.sort()
    n = len(ratios)
    trim = ratios[n // 10: n - n // 10] or ratios
    meta = {
        "impl": "glm52_h200.bench._local_bench_pair",
        "interleaved": True,
        "order_alternated": True,
        "rounds": rep,
        "paired_speedup_p50": ratios[n // 2] if n else float("nan"),
        "paired_speedup_trimmed_mean": statistics.fmean(trim) if trim else float("nan"),
        "paired_speedup_p10_p90": (
            [ratios[max(0, int(0.1 * n))], ratios[min(n - 1, int(0.9 * n))]] if n else None
        ),
        "unpaired_speedup_of_medians": (
            sorted(tb)[len(tb) // 2] / sorted(ta)[len(ta) // 2] if ta and tb else None
        ),
    }
    return _mk_timing(ta), _mk_timing(tb), meta


def _as_timing(obj):
    """Accept a Timing, a dataclass-ish object, or a dict; return something with p50_ms."""
    if obj is None:
        return None
    if hasattr(obj, "p50_ms"):
        return obj
    if isinstance(obj, dict) and "p50_ms" in obj:
        return _LocalTiming(
            p50_ms=obj["p50_ms"],
            p10_ms=obj.get("p10_ms", obj["p50_ms"]),
            p90_ms=obj.get("p90_ms", obj["p50_ms"]),
            mean_ms=obj.get("mean_ms", obj["p50_ms"]),
            n=obj.get("n", 0),
            noflush_p50_ms=obj.get("noflush_p50_ms", float("nan")),
        )
    return None


def _normalise_pair(res):
    """Map whatever `common.bench_pair` returned onto (Timing_a, Timing_b, meta)."""
    if isinstance(res, (tuple, list)) and len(res) >= 2:
        a, b = _as_timing(res[0]), _as_timing(res[1])
        meta = res[2] if len(res) > 2 and isinstance(res[2], dict) else {}
        if a and b:
            return a, b, dict(meta)
        return None
    if isinstance(res, dict):
        for ka, kb in (("a", "b"), ("fused", "unfused"), ("first", "second")):
            a, b = _as_timing(res.get(ka)), _as_timing(res.get(kb))
            if a and b:
                meta = {k: v for k, v in res.items() if k not in (ka, kb)}
                return a, b, meta
        return None
    for ka, kb in (("a", "b"), ("fused", "unfused"), ("first", "second")):
        a, b = _as_timing(getattr(res, ka, None)), _as_timing(getattr(res, kb, None))
        if a and b:
            # Harvest EVERY scalar field rather than a fixed name list.  The earlier version
            # looked for six specific names, none of which `common.PairTiming` happens to
            # use (it names its headline `ratio_p50`), so the metadata came back empty and
            # the caller formatted a None -- caught by a local smoke run, which is exactly
            # the crash that would otherwise have surfaced only on the H200.
            meta = {}
            for k in dir(res):
                if k.startswith("_") or k in (ka, kb):
                    continue
                try:
                    v = getattr(res, k)
                except Exception:  # noqa: BLE001
                    continue
                if callable(v):
                    continue
                if isinstance(v, (int, float, str, bool, dict, list, type(None))):
                    meta[k] = v
            return a, b, meta
    return None


# Names different implementations have used for the same quantity. The wrapper canonicalises
# onto `paired_speedup_p50` / `paired_speedup_trimmed_mean`, which is what the drivers read.
_PAIR_ALIASES = {
    "paired_speedup_p50": ("paired_speedup_p50", "ratio_p50", "paired_speedup",
                           "paired_p50", "speedup"),
    "paired_speedup_trimmed_mean": ("paired_speedup_trimmed_mean", "ratio_trimmed",
                                    "trimmed_mean", "paired_speedup_trimmed"),
    "paired_speedup_p10": ("paired_speedup_p10", "ratio_p10"),
    "paired_speedup_p90": ("paired_speedup_p90", "ratio_p90"),
    "ratio_of_medians": ("ratio_of_medians", "sequential_speedup"),
}


def _canonicalise_pair_meta(meta: dict) -> dict:
    """Fill the canonical paired-speedup keys from whatever the implementation called them."""
    for canon, names in _PAIR_ALIASES.items():
        if meta.get(canon) is not None:
            continue
        for n in names:
            v = meta.get(n)
            if v is not None:
                meta[canon] = v
                break
    return meta


def bench_pair(fused_fns, unfused_fns, warmup: int, rep: int, label: str = "") -> tuple:
    """Final per-regime timing of a fused/unfused pair.  Interleaved, paired, never two
    sequential `bench_chain` calls.

    Prefers `common.bench_pair`; falls back to the local implementation above if that entry
    point is absent or its signature does not fit, and records which one ran in the returned
    metadata (and therefore in the result JSON).  An operator reading a speedup must be able
    to tell which timer produced it.
    """
    fn = getattr(_common, "bench_pair", None)
    if fn is not None:
        why = None
        try:
            params = inspect.signature(fn).parameters
            kw = {}
            if "warmup" in params:
                kw["warmup"] = warmup
            if "rep" in params:
                kw["rep"] = rep
            if "flush" in params:
                kw["flush"] = True
            if "label" in params and label:
                kw["label"] = label
            res = fn(fused_fns, unfused_fns, **kw)
            norm = _normalise_pair(res)
            if norm is not None:
                a, b, meta = norm
                meta = dict(meta)
                meta.setdefault("impl", "common.bench_pair")
                meta.setdefault("interleaved", True)
                _canonicalise_pair_meta(meta)
                if meta.get("paired_speedup_p50") is None:
                    # Last resort: the ratio of the two medians. Weaker than a paired
                    # statistic (it does not cancel drift) so it is labelled as such rather
                    # than silently standing in for one.
                    if a.p50_ms > 0:
                        meta["paired_speedup_p50"] = b.p50_ms / a.p50_ms
                        meta["paired_speedup_is_ratio_of_medians"] = True
                return a, b, meta
            why = f"return value {type(res).__name__} not (Timing, Timing[, meta])"
        except Exception as exc:  # noqa: BLE001
            why = f"{type(exc).__name__}: {exc}"[:200]
        print(f"    [pair] common.bench_pair unusable ({why}); using local interleaver",
              flush=True)
        a, b, meta = _local_bench_pair(fused_fns, unfused_fns, warmup, rep)
        meta["common_bench_pair_error"] = why
        return a, b, _canonicalise_pair_meta(meta)
    a, b, meta = _local_bench_pair(fused_fns, unfused_fns, warmup, rep)
    meta["common_bench_pair_error"] = "absent from common"
    return a, b, _canonicalise_pair_meta(meta)


def bench_multi(chains: dict, warmup: int, rep: int, baseline: str | None = None) -> tuple:
    """N-way interleaved timing of several chains that share one config.

    `bench_pair` cancels drift between TWO arms by timing them inside one round and taking
    the median of the per-round ratios.  The F11 headline needs THREE (unfused,
    fused-nonspecialized, fused-warp-specialized) and honestly four (the unfused arm with
    specialization on, which is the control that stops warp specialization being credited to
    fusion).  Timing those as three separate pairs would re-introduce exactly the drift
    `bench_pair` exists to remove -- the arms would no longer share a round.

    So: every round times every chain once, each behind its own L2 flush, and the ORDER
    ROTATES by one position per round, so no arm systematically inherits another's cache or
    clock state.  Ratios are formed within a round against `baseline` (the first key by
    default) and reported as the median over rounds.

    Returns `({name: Timing}, meta)`.
    """
    names = list(chains)
    if not names:
        return {}, {"error": "no chains"}
    base = baseline if baseline in names else names[0]

    for _ in range(max(1, warmup)):
        for n in names:
            _run(chains[n])
    torch.cuda.synchronize()

    ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
    starts = {n: [ev() for _ in range(rep)] for n in names}
    ends = {n: [ev() for _ in range(rep)] for n in names}
    for i in range(rep):
        rot = names[i % len(names):] + names[: i % len(names)]
        for n in rot:
            _flush_l2()
            starts[n][i].record()
            _run(chains[n])
            ends[n][i].record()
    torch.cuda.synchronize()

    samples = {
        n: [s.elapsed_time(e) for s, e in zip(starts[n], ends[n])] for n in names
    }
    timings = {n: _mk_timing(samples[n]) for n in names}
    ratios: dict[str, dict] = {}
    for n in names:
        if n == base:
            continue
        rs = sorted(
            b / a for a, b in zip(samples[base], samples[n]) if a > 0
        )
        if not rs:
            continue
        k = len(rs)
        ratios[n] = {
            "vs": base,
            "p50": rs[k // 2],
            "p10": rs[max(0, int(0.1 * k))],
            "p90": rs[min(k - 1, int(0.9 * k))],
            # >1 means this chain is SLOWER than the baseline
            "pct_slower_than_baseline": 100.0 * (rs[k // 2] - 1.0),
        }
    meta = {
        "impl": "glm52_h200.bench.bench_multi",
        "interleaved": True,
        "order_rotated": True,
        "rounds": rep,
        "arms": names,
        "baseline": base,
        "per_round_ratios": ratios,
        "protocol": "every arm is timed once per round behind its own L2 flush, with the "
                    "order rotating one position each round; ratios are formed WITHIN a "
                    "round and reported as the median over rounds, so monotone drift "
                    "cancels the same way bench_pair makes it cancel",
    }
    return timings, meta


def tick_report(fused_ms: float, unfused_ms: float) -> dict:
    """Flag a speedup whose two operands are only a few timer ticks apart.

    At decode the kernels resolve to 9-17 CUDA-event ticks on the 4060 (1.024 us there);
    an H200 tick is measured by the preflight.  A ratio built from two integers that differ
    by 2 is quantised to tens of percent, and reporting it to three decimals is a lie of
    precision.  `tick_limited` is the flag the report must carry.

    **`tick_limited` is `None`, not a verdict, when the tick itself is not trustworthy.**
    The H200 preflight on file reports a 0.256 us tick that matches 3 % of its samples and a
    40 us harness floor, both measured while another tenant held ~51 GB of the card.  Deriving
    `tick_limited: false` from that would tell an operator a quantised decode ratio is
    clean -- a stronger and more damaging claim than admitting the tick is unknown.  The
    reason travels with the null so it lands in the result file, and the warning prints once.
    """
    st = calibration_status()
    tick = timer_tick_us()
    out = {
        "timer_tick_us": tick,
        "timer_tick_match_frac": st.get("timer_tick_match_frac"),
        "calibration_trusted": st.get("trusted"),
    }
    if st.get("trusted") is not True:
        warn_calibration_once()
        out["tick_limited"] = None
        out["tick_limited_reason"] = st.get("reason") or (
            "the preflight's timer tick could not be validated"
        )
        return out
    if not tick or fused_ms <= 0 or unfused_ms <= 0:
        out["tick_limited"] = None
        out["tick_limited_reason"] = "no timer tick recorded, or a non-positive timing"
        return out
    f_t, u_t = fused_ms * 1e3 / tick, unfused_ms * 1e3 / tick
    out.update(
        {
            "fused_ticks": f_t,
            "unfused_ticks": u_t,
            "gap_ticks": abs(u_t - f_t),
            # <=3 ticks of separation, or either arm under 20 ticks, means the quantisation
            # is a visible fraction of the ratio.
            "tick_limited": bool(abs(u_t - f_t) <= 3.0 or min(f_t, u_t) < 20.0),
            "quantisation_pct": 100.0 / min(f_t, u_t) if min(f_t, u_t) > 0 else None,
        }
    )
    return out


# ======================================================================================
# Checkpoints -- device-fenced
# ======================================================================================
def _ckpt_dir(result_id: str) -> Path:
    """Checkpoints live UNDER the results tree.

    On the 4060 port `CKPT_DIR` was derived from the repo root while `record()` honoured
    `$GLM52_RESULTS_DIR`; setting that variable then produced the worst case -- another
    device's checkpoint read from one tree and republished into a correctly labelled one.
    Isolating outputs without isolating inputs is worse than isolating neither.
    """
    base = getattr(_common, "RESULTS_DIR", None) or (ROOT / "results")
    return Path(base) / f"_{result_id}_ckpt"


def _bind_ckpt_args(fn, result_id: str, key: str, env, payload):
    """Positional args for `fn`, chosen from its real signature.

    Implementations of ckpt_save/ckpt_load in this project have taken either
    (name, regime, payload) or (name, regime, env, payload). Rather than guess, look: an
    arity mismatch raises TypeError once per regime, silently falls back to the local
    writer, and drops the device fence that makes a checkpoint safe to reuse.
    """
    try:
        params = [
            p for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        params = []
    args = []
    for p in params:
        n = p.name.lower()
        if n in ("name", "result_id", "id"):
            args.append(result_id)
        elif n in ("regime", "key"):
            args.append(key)
        elif n == "env":
            args.append(env)
        elif n in ("payload", "data", "row"):
            if payload is None:
                break
            args.append(payload)
        elif p.default is not p.empty:
            break
        else:  # an unrecognised required parameter: let the caller's except handle it
            raise TypeError(f"cannot bind checkpoint arg {p.name!r} of {fn.__name__}")
    return args


def ckpt_save(result_id: str, key: str, env, payload: dict) -> Path | None:
    """Write a per-regime checkpoint, stamped with the device that produced it."""
    fn = getattr(_common, "ckpt_save", None) or getattr(_common, "ckpt_write", None)
    if fn is not None:
        try:
            # `common`'s signature is (name, regime, payload); an older shape also took the
            # env. Bind by inspection rather than by assuming an arity -- guessing it wrong
            # is a TypeError per regime, which degrades silently to the local writer and
            # loses the device fence that `common` applies.
            return fn(*_bind_ckpt_args(fn, result_id, key, env, payload))
        except Exception as exc:  # noqa: BLE001 -- a checkpoint is a convenience, not data
            print(f"    [ckpt] common.ckpt_save failed ({type(exc).__name__}: {exc}); "
                  f"writing locally", flush=True)
    d = _ckpt_dir(result_id)
    try:
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{key}.json"
        p.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "device": getattr(env, "device_name", None),
                    "torch": getattr(env, "torch_version", None),
                    "triton": getattr(env, "triton_version", None),
                    "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "payload": payload,
                },
                indent=1,
                default=str,
            )
        )
        return p
    except Exception as exc:  # noqa: BLE001
        print(f"    [ckpt] could not write {key}: {type(exc).__name__}: {exc}", flush=True)
        return None


def ckpt_load(result_id: str, key: str, env, force: bool = False) -> dict | None:
    """Read a per-regime checkpoint, but ONLY if this device wrote it.

    `results/c500/_f01_oproj_resadd_ckpt/prefill_t8192.json` held `speedup 0.8458` -- the
    C500 study's headline datapoint -- and the 4060 port was one call away from
    republishing it inside a freshly probed RTX 4060 `env` block.  Every other field in
    that file would have identified the right machine.  The device stamp is the fence.
    """
    if force:
        return None
    fn = getattr(_common, "ckpt_load", None) or getattr(_common, "ckpt_read", None)
    if fn is not None:
        try:
            return fn(*_bind_ckpt_args(fn, result_id, key, env, None))
        except Exception as exc:  # noqa: BLE001
            print(f"    [ckpt] common.ckpt_load failed ({type(exc).__name__}: {exc}); "
                  f"reading locally", flush=True)
    p = _ckpt_dir(result_id) / f"{key}.json"
    if not p.exists():
        return None
    try:
        blob = json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"    [ckpt] {p.name} unreadable ({type(exc).__name__}); ignoring", flush=True)
        return None
    dev = blob.get("device")
    if dev != getattr(env, "device_name", None):
        print(
            f"    [ckpt] ignoring {p.name}: written by {dev or 'an unrecorded device'}, "
            f"this is {getattr(env, 'device_name', None)!r}",
            flush=True,
        )
        return None
    return blob.get("payload", blob)


# ======================================================================================
# Fairness bookkeeping
# ======================================================================================
class Fairness:
    """Per-arm grid accounting, written into the result JSON under `fairness`.

    `n_tried` / `n_failed` / live grid sizes PER SIDE are what makes an unfair comparison
    detectable after the fact.  Without them, "the fused arm searched 34 % fewer configs
    because the SMEM guard prunes its only legal tile family" is invisible in a table that
    looks completely reasonable.
    """

    def __init__(self, **static):
        self.static = dict(static)
        self.grids: dict[str, dict[str, dict]] = {}
        self.axes: dict[str, dict] = {}

    def add(self, regime: str, arm: str, stage: str, tune=None, size: int | None = None,
            grid: Sequence[dict] | None = None):
        node = self.grids.setdefault(regime, {}).setdefault(arm, {})
        if tune is not None:
            node[stage] = {
                "n_tried": getattr(tune, "n_tried", None),
                "n_failed": getattr(tune, "n_failed", None),
                "best_ms": getattr(tune, "best_ms", None),
            }
        else:
            node[stage] = {"n_tried": size}
        if grid is not None:
            # LIVE counts, per arm: how many of the configs this arm was actually handed
            # carry each sm_90 mapping axis. An axis offered to one arm only is the exact
            # bias this whole class exists to make visible, and prose cannot show it.
            node[stage]["axis_counts"] = axis_counts(grid)
        return self

    def axis(self, family: str, report: dict) -> "Fairness":
        """Record which sm_90 axes were offered to a fusion family, and why not otherwise."""
        self.axes[family] = report
        return self

    def totals(self, regime: str) -> dict:
        out = {}
        for arm, stages in self.grids.get(regime, {}).items():
            tried = sum(s.get("n_tried") or 0 for s in stages.values())
            failed = sum(s.get("n_failed") or 0 for s in stages.values())
            out[arm] = {"n_tried": tried, "n_failed": failed}
        return out

    def render(self, env, pair_meta: dict | None = None) -> dict:
        out = dict(self.static)
        out["grids"] = {
            r: {**arms, "_totals": self.totals(r)} for r, arms in self.grids.items()
        }
        out["device_probe"] = {
            "device": getattr(env, "device_name", None),
            "warp_size": getattr(env, "warp_size", None),
            "num_sm": getattr(env, "num_sm", None),
            "smem_bytes": getattr(env, "smem_bytes", None),
            "regs_per_sm": getattr(env, "regs_per_sm", None),
            "threads_per_sm": getattr(env, "threads_per_sm", None),
            "max_threads_per_block": max_threads_per_block(env),
            "l2_bytes": getattr(env, "l2_bytes", None),
            "probe_ok": getattr(env, "probe_ok", None),
        }
        out["l2_flush"] = l2_flush_audit(env)
        out["preflight"] = check_preflight_device(env)
        # WHICH PHYSICAL CARD.  Eight H200s on this host and other tenants on several of
        # them, so "NVIDIA H200" does not identify a device. The index is only meaningful
        # together with the mask (under CUDA_VISIBLE_DEVICES the torch index is always 0),
        # and the UUID is meaningful on its own -- so both are recorded.
        out["gpu"] = {
            "selection": gpu_selection(),
            "torch_device_index": getattr(env, "device_index", None),
            "torch_device_count": getattr(env, "device_count", None),
            "uuid": getattr(env, "uuid", None) or gpu_selection().get("uuid"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset: all)"),
        }
        out["smem_model"] = smem_stage_fit()
        out["h200_axes"] = {
            "per_family": self.axes or "no family recorded an axis report",
            "generic_overlays": h200_cfg_overlays(None)
            or "none at module scope (overlays are per kernel module; see per_family)",
            "policy": "every offered axis is applied to the coarse grid of BOTH arms; "
                      "per-arm live counts are under grids.<regime>.<arm>.<stage>."
                      "axis_counts",
        }
        out["timing"] = {
            "protocol": "final per-regime timings are A/B interleaved within one loop with "
            "the order alternating each round; the reported speedup is the median of the "
            "per-round ratios, so monotone drift cancels",
            "timer_tick_us": timer_tick_us(),
            "launch_cost_us": launch_cost_us(),
            "harness_floor_us": harness_floor_us(),
            "calibration": calibration_status(),
        }
        if pair_meta:
            out["timing"]["pair_impl"] = pair_meta.get("impl")
        if _TRAFFIC_ERR:
            out["traffic_model"] = f"unavailable: {_TRAFFIC_ERR}"
        return out


def traffic_ceilings(regime) -> dict:
    """`{fusion_name: row}` from the shared roofline model, or `{}` if it is unavailable.

    A missing ceiling must degrade a *column of the report*, never the measurement.
    """
    if _traffic is None:
        return {}
    try:
        return {t.fusion: t.row() for t in _traffic.model(regime)}
    except Exception as exc:  # noqa: BLE001
        print(f"[traffic] model unavailable ({type(exc).__name__}: {exc}); "
              f"ceilings omitted", flush=True)
        return {}


# ======================================================================================
# Kernel resource reporting
# ======================================================================================
def kernel_stats(run: Callable[[], object], jit_fn=None) -> dict:
    """n_regs / n_spills / shared for one compiled kernel, cache cleared first.

    Triton 3.x replaced `JITFunction.cache` with `device_caches[dev]`, a 5-tuple whose [0]
    is the compiled-kernel dict.  The old name raised AttributeError inside a bare except
    on the 4060 port, so every register report came back as `{"error": ...}` -- precisely
    the diagnostic needed to explain a fused-arm regression.  Both spellings are tried.
    """
    kc = None
    if jit_fn is not None:
        try:
            dev = torch.cuda.current_device()
            caches = getattr(jit_fn, "device_caches", None)
            if caches is not None:
                kc = caches[dev][0]
            else:
                kc = getattr(jit_fn, "cache", {}).get(dev)
            if kc is not None:
                kc.clear()
        except Exception:  # noqa: BLE001 -- diagnostics only
            kc = None
    try:
        k = run()
        torch.cuda.synchronize()
        out = {
            "n_regs": getattr(k, "n_regs", None),
            "n_spills": getattr(k, "n_spills", None),
            "shared_bytes": getattr(getattr(k, "metadata", None), "shared", None),
        }
        if all(v is None for v in out.values()) and kc:
            vals = list(kc.values())
            if len(vals) == 1:
                kk = vals[0]
                out = {
                    "n_regs": getattr(kk, "n_regs", None),
                    "n_spills": getattr(kk, "n_spills", None),
                    "shared_bytes": getattr(getattr(kk, "metadata", None), "shared", None),
                }
        return out
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:160]}


# ======================================================================================
# CLI + regimes
# ======================================================================================
#: The seven regimes this study reports.  H200 has 143 GB, so unlike the 4060 port none of
#: them is dropped for capacity and the whole layer fits.
REGIME_NAMES = [
    "decode_bs1", "decode_bs32", "decode_bs256", "decode_bs512", "decode_bs1024",
    "prefill_t2048", "prefill_t8192",
]

_REGIME_SHAPE = {
    "decode_bs1": (1, "decode", 4096),
    "decode_bs32": (32, "decode", 4096),
    "decode_bs256": (256, "decode", 4096),
    "decode_bs512": (512, "decode", 4096),
    "decode_bs1024": (1024, "decode", 4096),
    "prefill_t2048": (2048, "prefill", 2048),
    "prefill_t8192": (8192, "prefill", 8192),
}


def all_regimes(C) -> list:
    """The seven study regimes as `config.Regime` objects.

    Prefers the objects `config` already defines (so any future field it grows comes along);
    synthesises the rest from `config`'s own constants.  It never invents a shape: `T` and
    the o_proj K come from the table above and `C.OPROJ_K_DECODE` / `C.OPROJ_K_PREFILL`.
    """
    have = {r.name: r for r in getattr(C, "ALL_REGIMES", [])}
    out = []
    for name in REGIME_NAMES:
        if name in have:
            out.append(have[name])
            continue
        T, kind, kv = _REGIME_SHAPE[name]
        k = C.OPROJ_K_DECODE if kind == "decode" else C.OPROJ_K_PREFILL
        out.append(C.Regime(name, T, k, kv_len=kv))
    return out


def add_gpu_args(ap: argparse.ArgumentParser) -> None:
    """Declare `--gpu` / `--allow-busy` / `--gpu-busy-mb`.

    They are declared here so `--help` documents them and a typo is rejected, but they were
    ACTED ON at import time -- see the bootstrap at the top of this module.  argparse cannot
    run early enough: by the time a driver's `main()` is entered, the driver has already
    built its `config.env()` at module scope and CUDA is initialised, after which
    `CUDA_VISIBLE_DEVICES` is inert.
    """
    ap.add_argument(
        "--gpu", default=None, metavar="N|auto",
        help="run on exactly one GPU: an index, or 'auto' to take the idlest. Sets "
             "CUDA_VISIBLE_DEVICES before CUDA initialises, so every bench sees it as "
             "cuda:0. Refuses a device with foreign processes or resident memory.",
    )
    ap.add_argument(
        "--allow-busy", action="store_true",
        help="measure the chosen GPU even if another tenant is on it. This is what "
             "produced the preflight's 40 us harness floor; the result file records the "
             "override.",
    )
    ap.add_argument(
        "--gpu-busy-mb", type=float, default=None, metavar="MB",
        help="resident-memory threshold above which a GPU counts as busy "
             "(default: 1%% of the device, floor 1024 MiB)",
    )


def add_std_args(ap: argparse.ArgumentParser, units: Sequence[str] = ()) -> None:
    """`--regimes`, `--quick`, `--only`, `--gpu`, plus the switches every driver shares."""
    add_gpu_args(ap)
    ap.add_argument(
        "--regimes", default="",
        help="comma-separated subset of: " + ",".join(REGIME_NAMES) + " (default: all)",
    )
    ap.add_argument(
        "--quick", action="store_true",
        help="stride the search grids and cut the rep counts; for smoke-testing the "
             "driver end to end, NOT for a reportable number",
    )
    ap.add_argument(
        "--only", default="",
        help="comma-separated subset of this bench's variants"
             + (" (" + ",".join(units) + ")" if units else ""),
    )
    ap.add_argument(
        "--force", action="store_true",
        help="ignore existing checkpoints and re-measure every regime",
    )
    ap.add_argument(
        "--list", action="store_true", help="print the regimes and variants and exit",
    )


def resolve_regimes(C, arg: str) -> list:
    regs = all_regimes(C)
    if not arg:
        return regs
    want = [s.strip() for s in arg.split(",") if s.strip()]
    by_name = {r.name: r for r in regs}
    bad = [w for w in want if w not in by_name]
    if bad:
        raise SystemExit(
            f"unknown regime(s) {bad}; available: {', '.join(by_name)}"
        )
    return [by_name[w] for w in want]


def resolve_units(units: Sequence[str], arg: str) -> list[str]:
    if not arg:
        return list(units)
    want = [s.strip() for s in arg.split(",") if s.strip()]
    bad = [w for w in want if w not in units]
    if bad:
        raise SystemExit(f"unknown variant(s) {bad}; available: {', '.join(units)}")
    return want


def banner(env, extra: Sequence[str] = ()) -> None:
    """Print the environment banner at startup.

    Two of the 4060 benches printed none.  A degraded probe is then invisible to the
    operator until the result file is read on another continent -- and `require_ok()` only
    catches a probe that KNOWS it is degraded.
    """
    try:
        print(env.banner(), flush=True)
    except Exception:  # noqa: BLE001 -- a banner must never be the thing that fails
        print(
            f"[env] {getattr(env, 'device_name', '?')} | "
            f"{getattr(env, 'num_sm', '?')} SM | warp {getattr(env, 'warp_size', '?')} | "
            f"smem {getattr(env, 'smem_bytes', '?')} B",
            flush=True,
        )
    sel = gpu_selection()
    if sel.get("applied"):
        print(
            f"[gpu] measuring physical GPU {sel.get('index')} "
            f"(uuid {sel.get('uuid') or '?'}), mask verified={sel.get('mask_verified')}"
            + ("  [--allow-busy OVERRIDE]" if sel.get("busy_override") else ""),
            flush=True,
        )
    elif getattr(env, "device_count", 0) and int(getattr(env, "device_count", 0)) > 1:
        print(
            f"[gpu] !! this process sees {env.device_count} devices and no --gpu was given. "
            f"On a shared host that means another tenant's kernels can land on the card "
            f"being timed -- pass --gpu auto for a clean measurement.",
            flush=True,
        )
    pf = preflight()
    if pf:
        feats = ", ".join(
            f"{k}={'ok' if v.get('ok') else 'no'}"
            for k, v in (pf_get("triton_features", "compile_probes", default={}) or {}).items()
            if isinstance(v, dict)
        )
        print(f"[preflight] {pf_get('timestamp', default='?')} | {feats}", flush=True)
        print(
            f"[preflight] timer tick {timer_tick_us()} us | launch {launch_cost_us()} us | "
            f"L2 {env_int(env, 'l2_bytes') >> 20} MB",
            flush=True,
        )
    warn_calibration_once()
    fit = smem_stage_fit()
    print(
        f"[smem] ceiling {env_int(env, 'smem_bytes')} B | staging model "
        f"{fit['formula']} ({fit['source']})",
        flush=True,
    )
    c = hopper_caps()
    if c is not None:
        try:
            from glm52_h200.kernels import hopper as _hop

            print(_hop.banner(), flush=True)
        except Exception:  # noqa: BLE001 -- a banner must never be the thing that fails
            pass
    for line in extra:
        print(line, flush=True)


def mem_guard(need_bytes: int, label: str, headroom: float = 1.15) -> dict:
    """Refuse a huge allocation up front instead of dying inside `torch.empty`.

    On the 4060 `bench_f11` OOMed in `make_w13` before a single number existed.  H200 has
    143 GB and the whole layer fits, but "fits" is a property of the machine as configured
    (MIG, other tenants, fragmentation), not of the datasheet -- so it is checked, and the
    refusal names the number.
    """
    free, total = torch.cuda.mem_get_info()
    info = {
        "label": label, "need_bytes": int(need_bytes),
        "free_bytes": int(free), "total_bytes": int(total),
        "fits": bool(need_bytes * headroom < free),
    }
    print(
        f"[mem] {label}: need {need_bytes / 2**30:.2f} GB, free {free / 2**30:.2f} GB of "
        f"{total / 2**30:.2f} GB -> {'ok' if info['fits'] else 'DOES NOT FIT'}",
        flush=True,
    )
    return info


def reps(T: int, quick: bool) -> tuple[int, int, int, int]:
    """(tune warmup, tune rep, final warmup, final rep) as a function of problem size.

    Small decode kernels are launch-bound and need many reps to resolve; prefill kernels
    cost tens of milliseconds each and need few.  Identical for both arms of every pair.
    """
    if quick:
        return 2, 5, 3, 12
    if T <= 32:
        return 20, 50, 60, 300
    if T <= 512:
        return 15, 40, 40, 200
    if T <= 1024:
        return 10, 30, 30, 150
    if T <= 2048:
        return 10, 30, 25, 120
    return 5, 15, 12, 60
