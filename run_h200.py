#!/usr/bin/env python3
"""GLM-5.2 MoE fusion study -- the one command the operator runs on the H200.

    python3 run_h200.py

Everything else in `glm52_h200/` is a library or a single-family benchmark; this file is the
campaign driver. It exists because the H200 is a machine none of the authors can touch: the
operator runs this, sends back `results/h200/`, and that transcript is the entire record. So
the driver's job is not "launch some benchmarks" -- it is to make the record self-describing
and to make every way the run can go wrong *visible* rather than silent.

What that means concretely, and why each rule is here (each cost a real failure earlier in
this study -- see log/LOG-08 for the fairness audit and log/LOG-13 for the RTX 4060 port):

0.  **One GPU, chosen on purpose, and proven idle.** The node has eight H200s and other
    users. `--gpu N` exports `CUDA_VISIBLE_DEVICES=N` (plus `CUDA_DEVICE_ORDER=PCI_BUS_ID`)
    to every child, so each bench sees exactly that card as `cuda:0` and no bench needed a
    line changed; `--gpu auto` ranks the cards by (utilization, memory used) and prints the
    ranking that produced the choice. A card with another tenant is refused unless
    `--allow-busy`, and it is re-checked between families. This is not fastidiousness: the
    first preflight on this node ran on a card with ~51 GB already allocated by someone else
    and returned a 40.55 us harness floor against an 8.89 us launch, with the CUDA-event tick
    detector matching 3 samples in 100 where a real tick matches ~100. Those two numbers are
    what decides which cells are printed as UNRESOLVED, so a shared card does not add noise,
    it changes verdicts -- invisibly.

1.  **Preflight first, and the device is gated.** `glm52_h200/preflight.py` writes the single
    cached device probe every bench reads, and it is run on the SAME card the benches will
    use. If it is missing we run it; if it describes a different GPU than the pinned one --
    by model OR by UUID, because eight identical H200s all pass a name comparison -- we
    re-run it. A run on a non-sm_90 device is refused unless `--force`, because a C500- or
    Ada-shaped autotuning grid inside a file labelled "H200" is worse than no file at all.

2.  **Serial, always.** Two benchmarks on one GPU corrupt every timing in both. The families
    run one at a time, in an order that surfaces cheap failures first (f03 is minutes; the
    whole-layer bench is hours), so a broken stack is discovered in the first ten minutes
    rather than the eighth hour.

3.  **Device-fenced resume.** A family whose result JSON already exists is skipped -- but only
    after checking `_meta.device` against the GPU actually present. A stale checkpoint from
    another machine was one call away from being republished as a fresh measurement in the
    4060 port; here a mismatched result is quarantined (moved, never deleted) and re-run.

4.  **hwinfo at both ends, and around every family.** The 4060 run drifted 22 % thermally
    *within one run* and produced a speedup above its own physical ceiling. We cannot stop an
    H200 from clocking down, but we can make it undeniable after the fact, so clocks / temp /
    power / throttle reasons are snapshotted before and after each family and diffed at the
    end.

5.  **Timer-tick honesty.** decode_bs1 kernels resolve to a handful of CUDA-event ticks. Any
    cell whose two arms differ by less than `--unresolved-ticks` (default 3) ticks is printed
    as UNRESOLVED instead of a ratio that looks precise to four digits and is not.

6.  **One family failing never kills the run.** Every family is caught, logged, and the run
    continues; the ones that did not produce a result are listed at the end.

This file deliberately imports **nothing from the suite and nothing from torch**. It must be
able to report that the stack is broken, which it cannot do if importing the stack is what
crashes it -- and it must not hold a CUDA context for the hours the children are running.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent
SUITE = REPO / "glm52_h200"
PREFLIGHT_PY = SUITE / "preflight.py"
PREFLIGHT_JSON = SUITE / "preflight_h200.json"
DEFAULT_RESULTS = REPO / "results" / "h200"
DEFAULT_LOGDIR = REPO / "log" / "run_h200"

# The H200 has 143 GB, so unlike the 4060 port every regime and every fusion is in scope.
KNOWN_REGIMES = [
    "decode_bs1",
    "decode_bs32",
    "decode_bs256",
    "decode_bs512",
    "decode_bs1024",
    "prefill_t2048",
    "prefill_t8192",
]
# Short column headers -- seven full regime names do not fit in a terminal row.
REGIME_ABBR = {
    "decode_bs1": "d1",
    "decode_bs32": "d32",
    "decode_bs256": "d256",
    "decode_bs512": "d512",
    "decode_bs1024": "d1024",
    "prefill_t2048": "p2048",
    "prefill_t8192": "p8192",
}

# CUDA-event granularity if the preflight did not measure it. Deliberately the COARSEST tick
# any device in this study has shown (C500 0.256 us, RTX 4060 1.024 us): over-flagging cells
# as UNRESOLVED is recoverable, publishing a 4-digit ratio built out of 3 timer ticks is not.
DEFAULT_TICK_US = 1.024


# ======================================================================================
# family table
# ======================================================================================
@dataclass(frozen=True)
class Family:
    """One benchmark family: which script runs it, which JSON proves it ran.

    Scripts and result names are matched by *glob*, most-specific first, and resolved at
    runtime. The benches are written by other hands and their exact filenames are not this
    driver's to assume -- a driver that hardcodes a path it cannot see is a driver that
    reports "missing" for a family that is sitting right there.
    """

    key: str
    title: str
    script_globs: tuple[str, ...]
    result_globs: tuple[str, ...]
    timeout_s: int
    note: str = ""


# Cheap-and-likely-to-break first, most expensive last. f03 is a pure vector fusion that
# compiles in seconds: if the Triton stack is wrong, it fails in minute one, not hour eight.
FAMILIES: tuple[Family, ...] = (
    Family("f03", "#3  ResAdd+RMSNorm",
           ("bench_f03_resadd_rmsnorm.py", "bench_f03*.py"),
           ("f03_resadd_rmsnorm.json", "f03*.json"), 3 * 3600,
           "pure vector; smoke-tests the whole stack"),
    Family("f10", "#10 ExpertMerge+ResAdd",
           ("bench_f10_merge_resadd.py", "bench_f10*.py"),
           ("f10_merge_resadd.json", "f10*.json"), 4 * 3600,
           "needs the [T*topk, 6144] intermediate, not the expert weights"),
    Family("f01", "#1  o_proj+ResAdd",
           ("bench_f01_oproj_resadd.py", "bench_f01*.py"),
           ("f01_oproj_resadd.json", "f01*.json"), 8 * 3600,
           "first GEMM family; K=16384 prefill / 32768 decode"),
    Family("f04f05", "#4  ResAdd+RMSNorm+Router / #5 RMSNorm+Router",
           ("bench_f04f05_norm_router.py", "bench_f04f05*.py", "bench_f04*.py"),
           ("f04f05_norm_router.json", "f04f05*.json", "f04*.json"), 8 * 3600),
    Family("f11", "#11a LazyPreNorm->w13 / #11b LazyPreNorm->router",
           ("bench_f11_lazy_prenorm.py", "bench_f11_*.py"),
           ("f11_lazy_prenorm.json", "f11_*.json"), 10 * 3600,
           "#11a needs the 12 GB w13; fits here, unlike the 4060"),
    Family("f06", "#6  UpGate+SwiGLU",
           ("bench_f06_upgate_swiglu.py", "bench_f06*.py"),
           ("f06_upgate_swiglu.json", "f06*.json"), 10 * 3600,
           "needs 256-expert w13 (12 GB)"),
    Family("f08f09", "#8  Down+ExpertMerge / #9 Down+Merge+ResAdd2",
           ("bench_f08f09_down_merge_resadd.py", "bench_f08f09*.py", "bench_f08*.py"),
           ("f08f09_down_merge_resadd.json", "f08f09*.json", "f08*.json"), 14 * 3600,
           "needs 256-expert w2 (6 GB); four variants x seven regimes"),
    Family("layer", "whole-layer combination benchmark",
           ("bench_layer.py", "bench_layer_h200.py"),
           ("layer_configurations.json", "layer_*.json"), 16 * 3600,
           "LAST: most expensive; needs the whole 19.3 GB expert set resident"),
)
FAMILY_BY_KEY = {f.key: f for f in FAMILIES}

# (result-file id, lowercased variant substring) -> the fusion label printed in the table.
# Rules are tried in order and `""` is the catch-all, mirroring glm52/consolidate.py's
# CEILING_RULES: `f11a`/`f11b` and `f8`/`f9` are not separable by a common prefix, so the
# mapping is explicit rather than derived.
FUSION_LABELS: dict[str, tuple[tuple[str, str], ...]] = {
    "f01": (("", "#1  o_proj+ResAdd"),),
    "f03": (("", "#3  ResAdd+RMSNorm"),),
    "f04f05": (("f4", "#4  ResAdd+RMSNorm+Router"),
               ("f5", "#5  RMSNorm+Router"),
               ("", "#4/#5 norm+router")),
    "f06": (("", "#6  UpGate+SwiGLU"),),
    "f08f09": (("f9", "#9  Down+Merge+ResAdd2"),
               ("f8", "#8  Down+ExpertMerge"),
               ("", "#8  Down+ExpertMerge")),
    "f10": (("", "#10 ExpertMerge+ResAdd"),),
    "f11": (("router", "#11b LazyPreNorm->router"),
            ("w13", "#11a LazyPreNorm->w13"),
            ("combined", "#11a+#11b combined"),
            ("half", "#11b' half-fused"),
            ("", "#11 lazy pre-norm")),
    "layer": (("", "layer"),),
}


def fusion_label(key: str, variant: str) -> str:
    v = str(variant or "").lower()
    for sub, label in FUSION_LABELS.get(key, ()):
        if sub == "" or sub in v:
            return label
    return key


# ======================================================================================
# logging -- everything the operator sees also lands in log/run_h200/driver.log
# ======================================================================================
class Log:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fh = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = path.open("a", encoding="utf-8")
        except OSError as exc:  # noqa: BLE001 -- a read-only log dir must not kill the run
            print(f"!! cannot open driver log {path}: {exc}", flush=True)

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        if self.fh:
            try:
                self.fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
                self.fh.flush()
            except OSError:
                pass

    def close(self) -> None:
        if self.fh:
            try:
                self.fh.close()
            except OSError:
                pass


def rule(log: Log, title: str = "") -> None:
    log("=" * 92)
    if title:
        log(title)
        log("=" * 92)


# ======================================================================================
# hwinfo -- nvidia-smi only. No torch: the driver must not hold a CUDA context for hours,
# and must stay able to report that torch itself is broken.
# ======================================================================================
SMI_FIELDS = [
    "index", "name", "uuid", "driver_version", "pstate",
    "clocks.sm", "clocks.mem", "clocks.max.sm", "clocks.max.mem",
    "clocks_throttle_reasons.active",
    "power.draw", "enforced.power.limit",
    "temperature.gpu", "temperature.memory",
    "memory.total", "memory.used", "memory.free",
    "ecc.mode.current", "mig.mode.current", "persistence_mode", "compute_mode",
    "utilization.gpu", "utilization.memory",
]


def hwinfo() -> list[dict]:
    """Snapshot every GPU. Returns [] rather than raising -- a missing nvidia-smi is a
    documented gap in the record, not a reason to abandon a multi-hour run."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(SMI_FIELDS)}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:  # noqa: BLE001
        return []
    gpus = []
    for row in out.stdout.strip().splitlines():
        if not row.strip():
            continue
        vals = [v.strip() for v in row.split(",")]
        g = dict(zip(SMI_FIELDS, vals))
        g["_t"] = time.strftime("%Y-%m-%d %H:%M:%S")
        gpus.append(g)
    return gpus


def _num(s: object) -> float | None:
    """First number in an nvidia-smi cell ('1020 MHz' -> 1020.0, '[N/A]' -> None)."""
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group()) if m else None


def hw_line(g: dict) -> str:
    def f(k: str, unit: str = "") -> str:
        v = _num(g.get(k))
        return f"{v:g}{unit}" if v is not None else "n/a"

    thr = str(g.get("clocks_throttle_reasons.active", "")).strip()
    return (
        f"GPU{g.get('index', '?')} {g.get('name', '?')} | "
        f"sm {f('clocks.sm')}/{f('clocks.max.sm')} MHz | "
        f"mem {f('clocks.mem')}/{f('clocks.max.mem')} MHz | "
        f"{f('temperature.gpu')} C | {f('power.draw')}/{f('enforced.power.limit')} W | "
        f"used {f('memory.used')}/{f('memory.total')} MiB | util {f('utilization.gpu')}% | "
        f"pstate {g.get('pstate', '?')} | throttle {thr or 'n/a'}"
    )


def hw_drift(start: list[dict], end: list[dict]) -> list[dict]:
    """Start-vs-end delta per GPU. This is the artifact that would have caught the 4060's
    22 %-within-one-run thermal drift before it reached a report."""
    drift = []
    by_idx = {g.get("index"): g for g in end}
    for a in start:
        b = by_idx.get(a.get("index"))
        if not b:
            continue
        d = {"index": a.get("index"), "name": a.get("name")}
        for k in ("clocks.sm", "clocks.mem", "temperature.gpu", "power.draw",
                  "memory.used", "utilization.gpu"):
            va, vb = _num(a.get(k)), _num(b.get(k))
            d[k] = {"start": va, "end": vb,
                    "delta": (vb - va) if (va is not None and vb is not None) else None,
                    "pct": ((vb - va) / va * 100.0)
                           if (va not in (None, 0) and vb is not None) else None}
        d["throttle_start"] = a.get("clocks_throttle_reasons.active")
        d["throttle_end"] = b.get("clocks_throttle_reasons.active")
        drift.append(d)
    return drift


def _norm_dev(s: object) -> str:
    """Normalise a device name for comparison. Whitespace only -- 'H200' and 'H200 NVL' are
    genuinely different products and must not be collapsed into each other."""
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _norm_uuid(v: object) -> str:
    """nvidia-smi writes `GPU-b2318e71-...`, torch writes `b2318e71-...`. Same card."""
    s = str(v or "").strip().lower()
    return s[4:] if s.startswith("gpu-") else s


# ======================================================================================
# GPU selection -- eight cards, other tenants, one campaign that must own its device
# ======================================================================================
# The node has 8 H200s and other users. The first preflight ran on whichever card CUDA
# handed out; that card already had ~51 GB allocated by somebody else, and the two SHORT
# calibrations came back impossible -- launch 8.89 us against a 40.55 us "harness floor",
# and a CUDA-event tick that matched 3 of every 100 samples where a real tick matches
# essentially all of them. Those two numbers decide which cells this driver prints as
# UNRESOLVED, so a contaminated card does not just add noise, it changes the verdicts.
#
# Hence: choose a card explicitly, prove it was idle when we chose it, pin it for every
# child, and keep checking it stays ours for the whole multi-hour campaign.
#
# Every index below is an nvidia-smi (physical) index. CUDA renumbers under
# CUDA_VISIBLE_DEVICES and sorts FASTEST_FIRST unless told otherwise, so pinning always sets
# CUDA_DEVICE_ORDER=PCI_BUS_ID too -- without it `--gpu 3` can hand a child a different card
# than the one this driver inspected and vouched for.
_MIB = 1024 * 1024
# An untouched H200 on this node reports 4 MiB used; a card with a tenant reported 22-48 GB.
# 1 GiB separates those without any tuning. The free-memory floor comes from the study: the
# 256-expert weight set is 19.3 GB before a single activation is allocated.
GPU_MAX_USED_BYTES = 1 * 2 ** 30
GPU_MAX_UTIL_PCT = 5.0
GPU_MIN_FREE_BYTES = 32 * 2 ** 30

_APP_FIELDS = ["gpu_uuid", "pid", "process_name", "used_gpu_memory"]


def compute_apps() -> tuple[list[dict], str]:
    """(rows, error) from `nvidia-smi --query-compute-apps` -- who else is on each card.

    The error is returned rather than swallowed because "no rows" and "cannot see the rows"
    are different claims: under some container and MIG configurations this query is silently
    empty, and certifying a busy GPU idle is the one mistake this whole section exists to
    prevent.
    """
    if not shutil.which("nvidia-smi"):
        return [], "nvidia-smi not on PATH"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-compute-apps={','.join(_APP_FIELDS)}",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"
    blob = f"{out.stdout}\n{out.stderr}".lower()
    if out.returncode != 0 or "not supported" in blob:
        return [], (out.stderr or out.stdout).strip()[:200] or f"rc={out.returncode}"
    rows = []
    for line in out.stdout.strip().splitlines():
        line = line.strip()
        if not line or "no running processes" in line.lower():
            continue
        vals = [v.strip() for v in line.split(",")]
        if len(vals) < len(_APP_FIELDS):
            continue
        d = dict(zip(_APP_FIELDS, vals))
        rows.append({"gpu_uuid": d["gpu_uuid"], "pid": int(_num(d["pid"]) or 0),
                     "name": d["process_name"],
                     "used_bytes": int((_num(d["used_gpu_memory"]) or 0) * _MIB)})
    return rows, ""


def _is_descendant(pid: int) -> bool:
    """Is `pid` inside this driver's own process tree?

    Needed to tell "a neighbour appeared on our GPU" from "our own bench's per-regime worker
    has not finished exiting". Conservative in the useful direction: a pid whose ancestry
    cannot be read is treated as a stranger, because a false alarm costs one log line and a
    missed tenant costs the campaign's short-kernel numbers.
    """
    me, cur, hops = os.getpid(), int(pid or 0), 0
    while cur > 1 and hops < 64:
        if cur == me:
            return True
        try:
            with open(f"/proc/{cur}/stat", encoding="utf-8", errors="replace") as fh:
                data = fh.read()
            cur = int(data[data.rindex(")") + 1:].split()[1])  # comm may contain spaces
        except (OSError, ValueError, IndexError):
            return False
        hops += 1
    return False


def gpu_rows(hw: list[dict] | None = None) -> list[dict]:
    """Per-GPU state in bytes, with each card's compute processes attached."""
    hw = hwinfo() if hw is None else hw
    apps, app_err = compute_apps()
    by_uuid: dict[str, list[dict]] = {}
    for a in apps:
        by_uuid.setdefault(_norm_uuid(a["gpu_uuid"]), []).append(a)
    rows = []
    for g in hw:
        idx = _num(g.get("index"))
        rows.append({
            "index": int(idx) if idx is not None else None,
            "uuid": g.get("uuid"), "name": g.get("name"),
            "utilization_pct": _num(g.get("utilization.gpu")),
            "memory_total_bytes": int((_num(g.get("memory.total")) or 0) * _MIB),
            "memory_used_bytes": int((_num(g.get("memory.used")) or 0) * _MIB),
            "memory_free_bytes": int((_num(g.get("memory.free")) or 0) * _MIB),
            "compute_mode": g.get("compute_mode"), "mig_mode": g.get("mig.mode.current"),
            "persistence_mode": g.get("persistence_mode"),
            "temp_c": _num(g.get("temperature.gpu")),
            "processes": by_uuid.get(_norm_uuid(g.get("uuid")), []),
            # Non-null means this card's tenancy is UNKNOWN, never "proven empty".
            "process_query_error": app_err or None,
        })
    return rows


def gpu_busy_reasons(g: dict) -> list[str]:
    """Facts about *other people's* use of this card. [] means nobody else is on it.

    Tenancy only. "Not enough free VRAM" is deliberately NOT here: it is a capacity fact
    about the study, not evidence of a neighbour, and conflating the two would let a small
    development GPU be reported as occupied by a stranger who does not exist.
    """
    out: list[str] = []
    procs = [p for p in (g.get("processes") or []) if not _is_descendant(p["pid"])]
    if procs:
        shown = ", ".join(f"pid {p['pid']} {p['name']} ({p['used_bytes'] / 2**30:.1f} GB)"
                          for p in procs[:4])
        extra = f" (+{len(procs) - 4} more)" if len(procs) > 4 else ""
        out.append(f"{len(procs)} other compute process(es): {shown}{extra}")
    if (g.get("memory_used_bytes") or 0) > GPU_MAX_USED_BYTES:
        out.append(f"{g['memory_used_bytes'] / 2**30:.1f} GB already allocated by someone")
    if (g.get("utilization_pct") or 0) > GPU_MAX_UTIL_PCT:
        out.append(f"utilization {g['utilization_pct']:.0f}%")
    if str(g.get("compute_mode") or "").lower() not in ("default", "", "[n/a]", "n/a"):
        out.append(f"compute mode is {g['compute_mode']}, not Default")
    if str(g.get("mig_mode") or "").lower() == "enabled":
        out.append("MIG is enabled; this suite assumes a whole, undivided GPU")
    return out


def gpu_capacity_notes(g: dict) -> list[str]:
    """Can the study physically fit here? Reported and ranked on, but never a refusal --
    a bench that runs out of memory says so loudly and immediately, and blocking a
    deliberately-forced dry run on a small card would help nobody."""
    if (g.get("memory_free_bytes") or 0) < GPU_MIN_FREE_BYTES:
        return [f"only {(g.get('memory_free_bytes') or 0) / 2**30:.1f} GB free; the "
                f"whole-layer bench wants {GPU_MIN_FREE_BYTES / 2**30:.0f} GB (19.3 GB of "
                f"expert weights before a single activation)"]
    return []


def rank_gpus(rows: list[dict]) -> list[dict]:
    """Idlest first: unoccupied before occupied, roomy before cramped, then (utilization,
    memory used) ascending. Index breaks ties so repeated calls on an idle node agree."""
    ranked = []
    for g in rows:
        r = dict(g)
        r["reasons"] = gpu_busy_reasons(g)
        r["busy"] = bool(r["reasons"])
        r["capacity_notes"] = gpu_capacity_notes(g)
        r["capacity_short"] = bool(r["capacity_notes"])
        r["n_processes"] = len(g.get("processes") or [])
        ranked.append(r)
    ranked.sort(key=lambda r: (r["busy"], r["capacity_short"],
                               r.get("utilization_pct") or 0.0,
                               r.get("memory_used_bytes") or 0,
                               r["index"] if r["index"] is not None else 1 << 30))
    for i, r in enumerate(ranked):
        r["rank"] = i
    return ranked


def gpu_table(ranked: list[dict]) -> list[str]:
    """The ranking as printable lines -- the choice has to be second-guessable later."""
    if not ranked:
        return ["  (no GPU data: nvidia-smi unavailable)"]
    head = (f"  {'rank':>4} {'idx':>3}  {'util':>5} {'used GB':>9} {'free GB':>9} "
            f"{'proc':>4} {'temp':>5}  state")
    lines = [head, "  " + "-" * (len(head) - 2)]
    for r in ranked:
        util = "?" if r.get("utilization_pct") is None else f"{r['utilization_pct']:.0f}%"
        state = ("BUSY: " + "; ".join(r["reasons"])) if r["busy"] else "idle"
        if r.get("capacity_short"):
            state += ("  " if r["busy"] else " -- ") + "; ".join(r["capacity_notes"])
        lines.append(f"  {r['rank']:>4} {str(r['index']):>3}  {util:>5} "
                     f"{(r.get('memory_used_bytes') or 0) / 2**30:9.1f} "
                     f"{(r.get('memory_free_bytes') or 0) / 2**30:9.1f} "
                     f"{r.get('n_processes', 0):>4} "
                     f"{(r.get('temp_c') or 0):5.0f}  {state[:110]}")
    for r in ranked:
        lines.append(f"       gpu {r['index']}  {r.get('name')}  uuid {r.get('uuid')}")
    return lines


def pick_hw_row(hw: list[dict], index: int | None) -> dict:
    """The nvidia-smi row for the pinned card. Falls back to the first row when unpinned --
    but never silently to GPU 0 when a different one was pinned, because attributing GPU 0's
    clocks to a run on GPU 5 is the exact provenance bug this driver is built to catch."""
    if index is not None:
        for g in hw:
            if _num(g.get("index")) == index:
                return g
        return {}
    return hw[0] if hw else {}


def foreign_tenants(index: int, rows: list[dict] | None = None) -> tuple[list[dict], str]:
    """Compute processes on GPU `index` that are not ours. (processes, query error)."""
    rows = gpu_rows() if rows is None else rows
    for g in rows:
        if g.get("index") == index:
            return ([p for p in (g.get("processes") or []) if not _is_descendant(p["pid"])],
                    g.get("process_query_error") or "")
    return [], f"nvidia-smi reports no GPU with index {index}"


def resolve_gpu(log: Log, args: argparse.Namespace, hw: list[dict],
                warnings: list[str], measuring: bool = True) -> dict:
    """Turn --gpu into one physical card, and refuse to measure on somebody else's.

    Returns the whole decision, not just an index: the ranking that produced it, the card's
    tenancy at the moment it was chosen, and the baseline this driver will re-check against
    between families. `refuse=True` means main() should stop -- deliberately a return value
    rather than a raise, so the caller can still close its log.

    `measuring=False` (--summary-only, --dry-run) keeps the busy check as a warning: nothing
    is being timed, so a neighbour is a fact to record, not a reason to refuse. A bad index
    or an unparseable --gpu is still fatal either way, because that is an operator error and
    silently doing something else is how the wrong card gets used.
    """
    want = str(args.gpu or "").strip().lower()
    rows = gpu_rows(hw)
    ranked = rank_gpus(rows)
    sel: dict = {
        "requested": args.gpu, "index": None, "uuid": None, "name": None,
        "pinned": False, "allow_busy": bool(args.allow_busy), "refuse": False,
        "busy": None, "busy_reasons": [], "reason": "", "tenant_events": [],
        "thresholds": {"max_used_bytes": GPU_MAX_USED_BYTES,
                       "max_util_pct": GPU_MAX_UTIL_PCT,
                       "min_free_bytes": GPU_MIN_FREE_BYTES},
        "ranking": ranked,
        "env_CUDA_VISIBLE_DEVICES_before": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if len(ranked) > 1 or want == "auto":
        rule(log, f"GPU SELECTION -- {len(ranked)} card(s) on this host")
        for ln in gpu_table(ranked):
            log(ln)
        log("")

    if not ranked:
        sel["reason"] = "nvidia-smi returned no GPUs; the device is whatever CUDA picks"
        if want and want != "none":
            warnings.append("--gpu was requested but no GPU list is available; nothing pinned")
            log(f"!! --gpu {args.gpu} requested but nvidia-smi lists no GPUs -- not pinning.")
        return sel

    if not want or want == "none":
        sel["reason"] = "no --gpu given; CUDA_VISIBLE_DEVICES left as the environment set it"
        if len(ranked) > 1 and "CUDA_VISIBLE_DEVICES" not in os.environ:
            # Not fatal, but on a shared 8-card node it is the single likeliest way for this
            # campaign to end up incomparable with itself: families are only comparable if
            # every one of them ran on the SAME card, and unpinned they need not.
            idle = [r for r in ranked if not r["busy"]]
            msg = (f"{len(ranked)} GPUs are visible and none is pinned. Every number in this "
                   f"campaign is only comparable if every family ran on the SAME card, and "
                   f"nothing here guarantees that. Pass --gpu auto"
                   + (f" (GPU {idle[0]['index']} is idle right now)" if idle else ""))
            warnings.append(msg)
            log(f"!! {msg}")
        return sel

    if want == "auto":
        chosen = ranked[0]
        sel["reason"] = (f"idlest of {len(ranked)}: "
                         f"{(chosen.get('utilization_pct') or 0):.0f}% utilization, "
                         f"{(chosen.get('memory_used_bytes') or 0) / 2**30:.1f} GB used, "
                         f"{chosen['n_processes']} other compute process(es)")
    else:
        try:
            idx = int(want)
        except ValueError:
            log(f"!! --gpu {args.gpu!r} is neither an index, 'auto' nor 'none'.")
            sel["refuse"] = True
            sel["reason"] = f"unparseable --gpu {args.gpu!r}"
            return sel
        chosen = next((r for r in ranked if r["index"] == idx), None)
        if chosen is None:
            log(f"!! no GPU with nvidia-smi index {idx}; this host has "
                f"{sorted(r['index'] for r in ranked)}.")
            sel["refuse"] = True
            sel["reason"] = f"--gpu {idx} does not exist on this host"
            return sel
        sel["reason"] = f"--gpu {idx} as requested"

    sel.update(index=chosen["index"], uuid=chosen.get("uuid"), name=chosen.get("name"),
               busy=chosen["busy"], busy_reasons=chosen["reasons"],
               capacity_notes=chosen.get("capacity_notes") or [],
               process_query_error=chosen.get("process_query_error"),
               baseline_used_bytes=chosen.get("memory_used_bytes"),
               baseline_pids=sorted(p["pid"] for p in (chosen.get("processes") or [])))
    if chosen["busy"] and not args.allow_busy and measuring:
        log("")
        log("!" * 92)
        log(f"!! REFUSING TO RUN: GPU {sel['index']} already has another tenant.")
        for r in chosen["reasons"]:
            log(f"!!   - {r}")
        log("!!")
        log("!! This is not fastidiousness. The preflight on this node was taken on a shared")
        log("!! card and returned a 40.55 us harness floor against an 8.89 us launch, with")
        log("!! the CUDA-event tick detector matching 3 samples in 100 where a real tick")
        log("!! matches ~100. Neither number is physical. Those two numbers are what decides")
        log("!! which cells this driver prints as UNRESOLVED, so a shared card does not")
        log("!! merely add noise -- it changes the verdicts, and it does so invisibly.")
        log("!!")
        idle = [r for r in ranked if not r["busy"]]
        if idle:
            log(f"!! Idle right now: {', '.join(str(r['index']) for r in idle)}")
            log(f"!!     python3 run_h200.py --gpu {idle[0]['index']}")
        else:
            log("!! No card on this host is idle right now; waiting is the cheap option.")
        log("!! Or, accepting short-kernel numbers you will not be able to defend:")
        log(f"!!     python3 run_h200.py --gpu {sel['index']} --allow-busy")
        log("!" * 92)
        sel["refuse"] = True
        return sel

    sel["pinned"] = True
    log(f"  [gpu] pinning GPU {sel['index']} ({sel['name']}) uuid {sel['uuid']}")
    log(f"  [gpu] {sel['reason']}")
    log(f"  [gpu] every child gets CUDA_VISIBLE_DEVICES={sel['index']} and "
        f"CUDA_DEVICE_ORDER=PCI_BUS_ID, so each one sees exactly this card as cuda:0 -- "
        f"which is why no bench needed changing.")
    pre = os.environ.get("CUDA_VISIBLE_DEVICES")
    if pre is not None and pre.strip() != str(sel["index"]):
        msg = (f"the environment already had CUDA_VISIBLE_DEVICES={pre!r}; --gpu "
               f"{sel['index']} overrides it for every child")
        warnings.append(msg)
        log(f"  !! {msg}")
    if chosen.get("process_query_error"):
        msg = (f"the compute-process list is unavailable ({chosen['process_query_error']}), "
               f"so GPU {sel['index']} is UNPROVEN idle rather than proven idle")
        warnings.append(msg)
        log(f"  !! {msg}")
    for note in chosen.get("capacity_notes") or []:
        warnings.append(f"GPU {sel['index']}: {note}")
        log(f"  !! {note} -- the whole-layer family will fail here, the vector ones will not")
    if chosen["busy"]:
        why = "--allow-busy" if args.allow_busy else "nothing is being timed"
        msg = (f"{why}: GPU {sel['index']} is NOT idle "
               f"({'; '.join(chosen['reasons'])}). Short-kernel timings from this run "
               f"(decode_bs1 above all) are not trustworthy, and neither is any UNRESOLVED "
               f"verdict derived from a tick measured in the same conditions.")
        warnings.append(msg)
        log("")
        log("!" * 92)
        log(f"!! {msg}")
        log("!" * 92)
        log("")
    return sel


def check_tenants(log: Log, sel: dict, when: str, warnings: list[str]) -> None:
    """Did a neighbour move onto our card mid-campaign?

    Called between families. A multi-hour run that silently acquires a co-tenant in hour six
    produces exactly the drift this study has already been burned by twice, and the only
    thing worse than losing those numbers is not knowing which ones to lose. Descendants of
    this driver are excluded, so a bench worker still exiting is not mistaken for a stranger.
    """
    idx = sel.get("index")
    if idx is None:
        return
    rows = gpu_rows()  # one nvidia-smi pair per family boundary, reused for both checks
    procs, err = foreign_tenants(idx, rows)
    base = set(sel.get("baseline_pids") or [])
    new = [p for p in procs if p["pid"] not in base]
    row = next((r for r in rows if r["index"] == idx), {})
    used = row.get("memory_used_bytes") or 0
    grew = used - (sel.get("baseline_used_bytes") or 0)
    if not new and grew <= 2 * 2 ** 30:
        return
    detail = []
    if new:
        detail.append("new compute process(es): " + ", ".join(
            f"pid {p['pid']} {p['name']} ({p['used_bytes'] / 2**30:.1f} GB)" for p in new[:4]))
    if grew > 2 * 2 ** 30:
        detail.append(f"memory in use grew {grew / 2**30:+.1f} GB since the campaign started")
    msg = (f"a co-tenant appeared on GPU {idx} {when}: {'; '.join(detail)}. Families measured "
           f"from here on are not comparable with the earlier ones.")
    sel["tenant_events"].append({"when": when, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                 "new_processes": new, "memory_growth_bytes": grew,
                                 "query_error": err or None})
    warnings.append(msg)
    log("")
    log("!" * 92)
    log(f"!! {msg}")
    log("!! Nothing is stopped -- an aborted campaign loses more than a flagged one -- but")
    log("!! summary.json records when this happened so the affected families can be re-run.")
    log("!" * 92)
    log("")


# ======================================================================================
# preflight
# ======================================================================================
def read_preflight() -> dict | None:
    if not PREFLIGHT_JSON.exists():
        return None
    try:
        return json.loads(PREFLIGHT_JSON.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"!! {PREFLIGHT_JSON} is unreadable ({exc}); it will be regenerated.")
        return None


def run_preflight(log: Log, python: str, logdir: Path, quick: bool,
                  gpu: int | None = None) -> dict | None:
    """Probe the device -- on the SAME card the campaign will use, or the probe is fiction.

    The GPU is handed over as a flag when preflight advertises one and as CUDA_VISIBLE_DEVICES
    otherwise, so an older copy of preflight.py still lands on the right card. Both must not
    be left to chance: preflight's own default is to pick the idlest card, and if it picked a
    different one than the benches run on, every constant the benches read would describe a
    device that produced none of their timings.
    """
    if not PREFLIGHT_PY.exists():
        log(f"!! {PREFLIGHT_PY} not found -- cannot probe the device.")
        return None
    cmd = [python, str(PREFLIGHT_PY)] + (["--quick"] if quick else [])
    env = dict(os.environ)
    flags = script_flags(PREFLIGHT_PY)
    if "--gpu" in flags:
        cmd += ["--gpu", str(gpu) if gpu is not None else "none"]
    elif gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        log("    (this preflight.py has no --gpu flag; pinning via CUDA_VISIBLE_DEVICES)")
    log(f"    running preflight: {' '.join(cmd)}")
    logpath = logdir / "preflight.log"
    try:
        with logpath.open("w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=fh,
                                stderr=subprocess.STDOUT, timeout=3600).returncode
    except Exception as exc:  # noqa: BLE001
        log(f"!! preflight failed to launch: {type(exc).__name__}: {exc}")
        return None
    log(f"    preflight exit={rc}, log -> {logpath}")
    return read_preflight()


def preflight_digest(pf: dict) -> dict:
    """The handful of preflight facts that belong in the summary and the banner."""
    dev = pf.get("device", {}) or {}
    stack = pf.get("stack", {}) or {}
    feats = pf.get("triton_features", {}) or {}
    cal = pf.get("calibration", {}) or {}
    probes = {k: bool(v.get("ok")) for k, v in (feats.get("compile_probes") or {}).items()
              if isinstance(v, dict)}
    bw = cal.get("bandwidth", {}) or {}
    gemm = cal.get("gemm", {}) or {}
    return {
        "timestamp": pf.get("timestamp"),
        "device_name": dev.get("name"),
        "compute_capability": dev.get("compute_capability"),
        "sm_count": dev.get("multi_processor_count"),
        "warp_size": dev.get("warp_size"),
        "smem_per_block_optin": dev.get("shared_memory_per_block_optin"),
        "regs_per_sm": dev.get("regs_per_multiprocessor"),
        "l2_bytes": dev.get("L2_cache_size"),
        "total_memory": dev.get("total_memory"),
        "torch": stack.get("torch"),
        "triton": stack.get("triton"),
        "cuda": stack.get("torch_cuda"),
        "compile_probes": probes,
        "bandwidth_GBs": {k: v for k, v in bw.items() if k.endswith("_GBs")},
        "gemm_TFs": {k: v for k, v in gemm.items() if k.endswith("_TFs")},
        "launch_us": cal.get("launch_us"),
        "harness_floor_us": cal.get("harness_floor_us"),
        "timer_tick_us": cal.get("timer_tick_us"),
        "timer_tick_match_frac": cal.get("timer_tick_match_frac"),
        # The preflight now judges its own launch/tick numbers, because they are the only
        # ones a co-tenant can silently ruin. Carried through so the driver can refuse to
        # present a tick-based UNRESOLVED verdict as if it were solid.
        "launch_timer_trustworthy": cal.get("launch_timer_trustworthy"),
        "launch_timer_doubts": cal.get("launch_timer_doubts") or [],
        "gpu_selection": {k: v for k, v in (pf.get("gpu_selection") or {}).items()
                          if k in ("index", "uuid", "name", "reason", "busy", "busy_reasons")},
        "capacity": pf.get("capacity", {}),
        "probe_errors": list((pf.get("probe_errors") or {}).keys()),
    }


def banner(log: Log, pf: dict | None, hw: list[dict], results: Path, tick: dict,
           gpu: dict) -> None:
    rule(log, "GLM-5.2 MoE fusion study -- H200 campaign")
    log(f"  repo            {REPO}")
    log(f"  results dir     {results}")
    log(f"  driver started  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if gpu.get("index") is not None:
        log(f"  gpu             {gpu['index']}  uuid {gpu.get('uuid')}  ({gpu.get('reason')})")
    else:
        log(f"  gpu             not pinned ({gpu.get('reason') or 'no --gpu given'})")
    log("")
    if pf:
        d = preflight_digest(pf)
        gb = 2 ** 30
        mem = f"{d['total_memory'] / gb:.1f} GB" if d.get("total_memory") else "n/a"
        l2 = f"{d['l2_bytes'] / 2**20:.0f} MB" if d.get("l2_bytes") else "n/a"
        cc = str(d.get("compute_capability") or "?").replace(".", "")
        log(f"  [device] {d['device_name']}  sm_{cc}"
            f" | {d['sm_count']} SM | warp {d['warp_size']} | "
            f"smem/block {d['smem_per_block_optin']} B | regs/SM {d['regs_per_sm']} | "
            f"L2 {l2} | VRAM {mem}")
        log(f"  [stack]  torch {d['torch']} (cuda {d['cuda']}) | triton {d['triton']}")
        if d["compile_probes"]:
            ok = " ".join(f"{'+' if v else '-'}{k}" for k, v in d["compile_probes"].items())
            log(f"  [triton features, compiled AND launched] {ok}")
        if d["bandwidth_GBs"] or d["gemm_TFs"]:
            bw = "  ".join(f"{k}={v:.0f}" for k, v in d["bandwidth_GBs"].items())
            gm = "  ".join(f"{k}={v:.1f}" for k, v in d["gemm_TFs"].items())
            log(f"  [calib]  {bw}   {gm}")
        lu = f"{d['launch_us']:.2f}" if isinstance(d.get("launch_us"), (int, float)) else "n/a"
        log(f"  [calib]  launch {lu} us | timer tick "
            f"{tick['tick_us']} us ({tick['source']})"
            + ("" if tick.get("trusted", True) else "  <== NOT TRUSTWORTHY"))
        if not tick.get("trusted", True):
            log("  [calib!] the preflight flagged its own launch/tick calibration as "
                "contaminated:")
            for reason in tick.get("distrust_reasons", []):
                log(f"  [calib!]   - {reason}")
            log("  [calib!] every UNRESOLVED verdict below is computed from that tick, so "
                "treat them as advisory until the preflight is re-run on an idle GPU:")
            log("  [calib!]     python3 glm52_h200/preflight.py --gpu <idle index>")
        cap = d.get("capacity") or {}
        if cap:
            log(f"  [capacity] expert weights "
                f"{(cap.get('expert_weights_bytes') or 0) / 2**30:.1f} GB | fit="
                f"{cap.get('expert_weights_fit')} | whole layer feasible="
                f"{cap.get('whole_layer_feasible')}")
        if d["probe_errors"]:
            log(f"  [preflight] {len(d['probe_errors'])} probe(s) recorded errors: "
                f"{', '.join(d['probe_errors'])}")
    else:
        log("  [device] NO PREFLIGHT JSON -- running blind; nothing below is device-fenced.")
    log("")
    if hw:
        for g in hw:
            log(f"  [hwinfo@start] {hw_line(g)}")
    else:
        log("  [hwinfo@start] nvidia-smi unavailable -- clock/thermal drift will NOT be "
            "recorded for this run.")
    log("")


# ======================================================================================
# device gate
# ======================================================================================
def device_gate(log: Log, pf: dict | None, hw_row: dict, force: bool,
                warnings: list[str]) -> dict:
    """Refuse to run a study labelled H200 on something that is not an H200.

    The C500 study baked `warpSize 64` / `smem 65536` into autotuning guards; on any other
    device those silently prune the search grid, and they do not prune both arms of a
    fused/unfused pair equally -- which manufactures or destroys a fusion win. This suite
    reads every constant from the probe instead, but the probe is only trustworthy if the
    device it describes is the device present, so that is checked here rather than assumed.
    """
    dev = (pf or {}).get("device", {}) or {}
    cc = str(dev.get("compute_capability") or "")
    name = _norm_dev(dev.get("name"))
    # `hw_row` is the PINNED card's nvidia-smi row, not GPU 0's. On an eight-card node those
    # differ, and comparing the probe against the wrong card is how a stale probe passes.
    smi_name = _norm_dev((hw_row or {}).get("name"))
    info = {"name": name or smi_name, "compute_capability": cc or None,
            "sm90": cc == "9.0", "forced": bool(force), "nvidia_smi_name": smi_name or None,
            "nvidia_smi_index": (hw_row or {}).get("index"),
            "uuid": (hw_row or {}).get("uuid")}
    # The probe must describe the card we pinned, not merely a card of the same model: eight
    # identical H200s all pass a name comparison.
    pf_uuid = _norm_uuid((pf or {}).get("gpu_selection", {}).get("uuid") or dev.get("uuid"))
    row_uuid = _norm_uuid((hw_row or {}).get("uuid"))
    if pf_uuid and row_uuid and pf_uuid != row_uuid:
        msg = (f"the preflight probed GPU uuid {pf_uuid} but this campaign is pinned to "
               f"{row_uuid} -- the device constants every bench reads were measured on a "
               f"DIFFERENT card. Re-run the preflight with --gpu "
               f"{(hw_row or {}).get('index')}.")
        warnings.append(msg)
        log(f"!! {msg}")
        info["preflight_uuid_mismatch"] = True

    if smi_name and name and smi_name != name:
        msg = (f"preflight describes '{name}' but nvidia-smi reports '{smi_name}' -- "
               f"the cached probe is stale")
        warnings.append(msg)
        log(f"!! {msg}")
        info["stale_preflight"] = True

    if cc == "9.0":
        log(f"  [gate] sm_90 confirmed ({name}). Hopper paths (TMA, clusters, warp "
            f"specialisation) are eligible; each is still chosen at RUNTIME from the probe.")
        return info

    detail = f"compute capability {cc or 'UNKNOWN'} ({name or 'unknown device'})"
    if force:
        msg = (f"--force: proceeding on a NON-sm_90 device ({detail}). Results are NOT an "
               f"H200 measurement; do not file them as one.")
        warnings.append(msg)
        log("")
        log("!" * 92)
        log(f"!! {msg}")
        log(f"!! Use --results-dir to keep this out of results/h200/.")
        log("!" * 92)
        log("")
        return info

    log("")
    log("!" * 92)
    log(f"!! REFUSING TO RUN: this suite targets an NVIDIA H200 (sm_90) and found {detail}.")
    log("!!")
    log("!! Why this is fatal rather than a warning: the benches size their autotuning grids")
    log("!! from the device probe, so on a different device they would produce a correct")
    log("!! search over the WRONG hardware and write it into a file named 'h200'. A wrong-")
    log("!! but-plausible table is worse than no table.")
    log("!!")
    log("!! If you know what you are doing (a dry run on a lab GPU, a stack smoke test):")
    log("!!     python3 run_h200.py --force --results-dir results/<thisdevice>")
    log("!! and do not merge the output into the H200 record.")
    log("!" * 92)
    raise SystemExit(2)


# ======================================================================================
# result-file device fence
# ======================================================================================
def quarantine_foreign_results(log: Log, results: Path, device: str,
                               enabled: bool, warnings: list[str]) -> list[dict]:
    """Move any pre-existing result whose `_meta.device` is not this GPU out of the way.

    In the 4060 port a C500 checkpoint was one call away from being republished as a fresh
    measurement. Resumability is only safe if "this file already exists" also means "this
    file was produced by the GPU in this box". Files are MOVED, never deleted -- the mistake
    to prevent is silent reuse, not the existence of old data.
    """
    moved: list[dict] = []
    if not results.exists():
        return moved
    if not device:
        warnings.append("device name unknown -- result files were not device-fenced")
        log("!! device name unknown; skipping the result device fence (nothing to compare).")
        return moved
    if not enabled:
        warnings.append("--no-device-fence: pre-existing results were reused unchecked")
        log("!! --no-device-fence: results are being reused WITHOUT provenance checking.")
        return moved

    qdir = results / f"_quarantine_foreign_{time.strftime('%Y%m%d_%H%M%S')}"
    for p in sorted(results.rglob("*.json")):
        rel_parts = p.relative_to(results).parts
        if any(part.startswith("_quarantine_foreign_") for part in rel_parts):
            continue
        if p.name == "summary.json" and p.parent == results:
            continue  # written by this driver, fenced by its own _meta below
        try:
            payload = json.loads(p.read_text())
        except Exception:  # noqa: BLE001 -- unparseable is its own problem, handled at resume
            continue
        got = _norm_dev((payload.get("_meta") or {}).get("device")) \
            if isinstance(payload, dict) else ""
        if got == device:
            continue
        rel = p.relative_to(results)
        dst = qdir / rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dst))
            moved.append({"file": str(rel), "recorded_device": got or None,
                          "present_device": device, "moved_to": str(dst)})
            log(f"!! quarantined {rel}: recorded device "
                f"'{got or 'MISSING'}' != present '{device}' -> {dst}")
        except OSError as exc:
            log(f"!! could not quarantine {rel}: {exc}")
    if moved:
        warnings.append(f"{len(moved)} pre-existing result file(s) were from another device "
                        f"and were quarantined under {qdir.name}; they will be re-measured")
        log(f"   ({len(moved)} file(s) moved to {qdir.name} -- they will be re-measured)")
    return moved


# ======================================================================================
# family resolution + execution
# ======================================================================================
_ADDARG = re.compile(r"""add_argument\(\s*["'](--[A-Za-z0-9][A-Za-z0-9_-]*)["']""")


def script_flags(path: Path) -> set[str]:
    """Which `--flags` a bench accepts, read from its source.

    Statically, by regex, rather than by running `script --help`: importing a bench pulls in
    torch and initialises CUDA, and this driver spends hours *not* holding a context. The
    cost of being wrong is small (an unsupported flag is simply not passed).
    """
    try:
        return set(_ADDARG.findall(path.read_text(encoding="utf-8", errors="replace")))
    except OSError:
        return set()


def find_script(fam: Family) -> Path | None:
    for d in (SUITE / "bench", SUITE):
        if not d.is_dir():
            continue
        for pat in fam.script_globs:
            hits = sorted(p for p in d.glob(pat) if p.is_file())
            exact = [p for p in hits if p.name == pat]
            if exact:
                return exact[0]
            if hits:
                # shortest name wins: `bench_layer.py` over `bench_layer_ab.py`
                return sorted(hits, key=lambda p: (len(p.name), p.name))[0]
    return None


def find_result(fam: Family, results: Path) -> Path | None:
    if not results.is_dir():
        return None
    for pat in fam.result_globs:
        hits = sorted(p for p in results.glob(pat)
                      if p.is_file() and not p.name.startswith("_"))
        exact = [p for p in hits if p.name == pat]
        if exact:
            return exact[0]
        if hits:
            return sorted(hits, key=lambda p: (len(p.name), p.name))[0]
    return None


def result_is_usable(path: Path, device: str, fence: bool) -> tuple[bool, str]:
    """A result counts as 'done' only if it parses, is complete, and is OURS."""
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable ({type(exc).__name__})"
    if not isinstance(payload, dict):
        return False, "not a JSON object"
    if payload.get("complete") is False:
        return False, "payload says complete=false"
    got = _norm_dev((payload.get("_meta") or {}).get("device"))
    if fence and device:
        if not got:
            return False, "no _meta.device -- provenance unprovable"
        if got != device:
            return False, f"recorded device '{got}' != present '{device}'"
    return True, "complete"


def _kill_group(proc: subprocess.Popen, log: Log) -> None:
    """Kill the child AND its workers. f06/f08f09 fan out into per-regime subprocesses;
    killing only the parent leaves those holding the GPU for the next family."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            proc.wait(timeout=20)
            return
        except subprocess.TimeoutExpired:
            continue


def fmt_dur(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds < 3600 else f"{seconds / 3600:.1f} h"


def tail(path: Path, n: int = 3, maxbytes: int = 262144) -> list[str]:
    try:
        with path.open("rb") as fh:
            try:
                fh.seek(-maxbytes, os.SEEK_END)
            except OSError:
                fh.seek(0)
            data = fh.read()
    except OSError:
        return []
    lines = [ln for ln in data.decode("utf-8", "replace").splitlines() if ln.strip()]
    return lines[-n:]


def run_family(log: Log, fam: Family, script: Path, args: argparse.Namespace,
               results: Path, logdir: Path, extra: list[str], gpu: dict) -> dict:
    """Launch one family, stream a heartbeat, and return a status record.

    Never raises for a benchmark failure. The whole point of an eight-hour campaign on a
    machine we cannot reach is that hour seven still runs after hour three broke.
    """
    flags = script_flags(script)
    cmd = [args.python, str(script)]
    unhonoured = []
    if args.quick:
        if "--quick" in flags:
            cmd.append("--quick")
        else:
            unhonoured.append("--quick")
    if args.regimes:
        if "--regimes" in flags:
            cmd += ["--regimes", args.regimes]
        elif "--only" in flags:
            cmd += ["--only", args.regimes]
        else:
            unhonoured.append("--regimes")
    if args.results_dir and "--results-dir" in flags:
        cmd += ["--results-dir", str(results)]
    cmd += extra

    env = dict(os.environ)
    # `glm52_h200/common.py` reads GLM52_H200_*; the original `glm52/common.py` reads
    # GLM52_RESULTS_DIR. Both spellings are set: it costs nothing and removes a guess about
    # which harness a given bench ended up importing.
    env["GLM52_RESULTS_DIR"] = str(results)
    env["GLM52_H200_RESULTS_DIR"] = str(results)
    env["GLM52_PREFLIGHT"] = str(PREFLIGHT_JSON)
    env["GLM52_H200_PREFLIGHT"] = str(PREFLIGHT_JSON)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    if args.quick:
        env["GLM52_QUICK"] = "1"
    if args.regimes:
        env["GLM52_REGIMES"] = args.regimes
    if args.force_rerun:
        # The driver's own resume check only sees the family's final JSON; the benches keep
        # their own per-regime checkpoints under <results>/_ckpt. --force-rerun has to reach
        # those too, or "re-run" quietly means "re-print the cached numbers".
        env["GLM52_H200_FORCE"] = "1"
    if args.disable_features:
        # `GLM52_H200_DISABLE_FEATURES` alone only reaches `common.features()`, which is
        # METADATA -- it is written into the result file and gates nothing. The real
        # capability switches live in `kernels/hopper.py` under per-feature keys. Setting
        # only the former produced the worst possible outcome: the feature stayed live while
        # every result file recorded it as disabled. And this flag is the operator's one
        # remote escape hatch on a machine nobody can log into, so it has to actually work.
        env["GLM52_H200_DISABLE_FEATURES"] = args.disable_features
        _FEATURE_ENV = {
            "tma": "GLM52_H200_TMA",
            "clusters": "GLM52_H200_CLUSTERS",
            "cluster": "GLM52_H200_CLUSTERS",
            "ws": "GLM52_H200_WS",
            "warp_specialize": "GLM52_H200_WS",
            "warp-specialize": "GLM52_H200_WS",
            "wgmma": "GLM52_H200_WGMMA",
        }
        unknown = []
        for name in (s.strip().lower() for s in args.disable_features.split(",")):
            if not name:
                continue
            if name in ("all", "classic"):
                env["GLM52_H200_CLASSIC"] = "1"
            elif name in _FEATURE_ENV:
                env[_FEATURE_ENV[name]] = "0"
            else:
                unknown.append(name)
        if unknown:
            print(f"!! --disable-features: unrecognised {unknown}; "
                  f"known: tma, ws, clusters, wgmma, all", flush=True)
    if args.flush_mb:
        env["GLM52_H200_FLUSH_MB"] = str(args.flush_mb)
    if gpu.get("index") is not None:
        # This is the whole reason no bench needed changing: with one device visible, every
        # child's `cuda:0` is this card and nothing downstream has to know about device
        # selection. PCI_BUS_ID because the default FASTEST_FIRST ordering would let the
        # index mean a different card than the one nvidia-smi (and this driver) inspected.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    logpath = logdir / f"{fam.key}.log"
    timeout_s = args.timeout if args.timeout else fam.timeout_s
    rec: dict = {
        "family": fam.key, "title": fam.title, "script": str(script),
        "cmd": cmd, "log": str(logpath), "timeout_s": timeout_s,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        # exactly what this child saw, so a surprising number can be traced to a flag
        "env_overrides": {k: env[k] for k in sorted(env)
                          if k.startswith(("GLM52", "CUDA_"))},
        # Which physical card produced this family, by index AND uuid: an index alone is
        # meaningless once someone re-reads the file on another host.
        "gpu_index": gpu.get("index"), "gpu_uuid": gpu.get("uuid"),
        "script_flags_detected": sorted(flags),
    }
    log(f"  cmd     {' '.join(cmd)}")
    log(f"  log     {logpath}")
    log(f"  timeout {fmt_dur(timeout_s)}")
    if unhonoured:
        # Say so rather than assume the env var was read: "I asked for two regimes and got
        # seven" is a four-hour surprise on a machine nobody can watch.
        rec["unhonoured_flags"] = unhonoured
        log(f"  !! {script.name} advertises no {', '.join(unhonoured)} flag; the request was "
            f"passed only via {'/'.join('GLM52_' + f.strip('-').upper() for f in unhonoured)}"
            f" and may be ignored by this bench.")

    hw_before = hwinfo()
    row_before = pick_hw_row(hw_before, gpu.get("index"))
    if row_before:
        log(f"  [hw before] {hw_line(row_before)}")
    rec["hw_before"] = hw_before

    t0 = time.time()
    timed_out = False
    interrupted = False
    try:
        with logpath.open("w", encoding="utf-8") as fh:
            fh.write(f"# {' '.join(cmd)}\n# started {rec['started']}\n")
            fh.flush()
            # start_new_session: the child gets its own process group, so a Ctrl-C at the
            # terminal does NOT reach it -- which is why every exit path below has to kill
            # the group explicitly. An orphaned bench holding the GPU would silently corrupt
            # whatever the operator runs next.
            proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=fh,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            last_beat = t0
            try:
                while True:
                    try:
                        rc = proc.wait(timeout=5)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    now = time.time()
                    if now - t0 > timeout_s:
                        timed_out = True
                        log(f"  !! [{fam.key}] exceeded {fmt_dur(timeout_s)} -- killing "
                            f"process group")
                        _kill_group(proc, log)
                        rc = -signal.SIGKILL
                        break
                    if now - last_beat >= args.heartbeat:
                        last_beat = now
                        t = tail(logpath, 1)
                        log(f"    [{fam.key}] {(now - t0) / 60:6.1f} min | "
                            f"{(t[0] if t else '(no output yet)')[:110]}")
            except KeyboardInterrupt:
                # Anywhere in the poll loop, including inside tail()/log().
                interrupted = True
                log(f"  !! [{fam.key}] interrupted -- killing process group")
                _kill_group(proc, log)
                rc = -signal.SIGINT
    except FileNotFoundError as exc:
        rec.update(status="launch_failed", error=f"{type(exc).__name__}: {exc}")
        log(f"  !! could not launch: {exc}")
        return rec
    except Exception as exc:  # noqa: BLE001 -- a driver bug must not lose the campaign
        rec.update(status="driver_error", error=f"{type(exc).__name__}: {exc}")
        log(f"  !! driver error: {type(exc).__name__}: {exc}")
        return rec

    dt = time.time() - t0
    rec.update(returncode=rc, wall_s=dt, timed_out=timed_out,
               finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    hw_after = hwinfo()
    rec["hw_after"] = hw_after
    row_after = pick_hw_row(hw_after, gpu.get("index"))
    if row_after:
        log(f"  [hw after ] {hw_line(row_after)}")
    if hw_before and hw_after:
        rec["hw_drift"] = hw_drift(hw_before, hw_after)
        # The drift that matters is the PINNED card's, not GPU 0's.
        d = next((x for x in rec["hw_drift"]
                  if gpu.get("index") is None or _num(x.get("index")) == gpu["index"]),
                 rec["hw_drift"][0] if rec["hw_drift"] else {})
        sm, tmp = d.get("clocks.sm", {}), d.get("temperature.gpu", {})
        if sm.get("pct") is not None and abs(sm["pct"]) >= 5:
            log(f"  !! SM clock moved {sm['pct']:+.1f}% during this family "
                f"({sm['start']:g} -> {sm['end']:g} MHz) -- interleaved A/B timing is what "
                f"protects the ratio from this; check the bench recorded a paired statistic.")
        elif tmp.get("delta") is not None and abs(tmp["delta"]) >= 10:
            log(f"  .. temperature moved {tmp['delta']:+.0f} C ({tmp['start']:g} -> "
                f"{tmp['end']:g})")

    if interrupted:
        rec["status"] = "interrupted"
        rec["interrupted"] = True
    elif timed_out:
        rec["status"] = "timeout"
    elif rc == 0:
        rec["status"] = "ok"
    else:
        rec["status"] = "failed"

    log(f"  exit={rc} status={rec['status']} wall={dt / 60:.1f} min")
    for ln in tail(logpath, 4):
        log(f"    | {ln[:140]}")
    return rec


# ======================================================================================
# result parsing -> speedup cells
# ======================================================================================
# Keys that are containers, not variant names: a row found under `payload["rows"]` has no
# variant, it IS the family's only variant. Without this every single-variant family prints
# as "#3 ResAdd+RMSNorm [rows]".
_CONTAINER_KEYS = {"rows", "regimes", "timings", "results", "data", "by_regime", "table"}


def walk_rows(obj, out: list[dict], parent_key: str | None = None) -> None:
    """Find every dict that looks like a benchmark row, anywhere in the payload.

    Lifted from `glm52/consolidate.py` so the two agree. `parent_key` supplies a variant
    label for families that nest sub-variants under named keys instead of carrying a
    `variant` field -- f11 does this (`f11b_router`, `f11a_w13`, `combined`, `half_fused`).
    """
    if isinstance(obj, dict):
        if "regime" in obj and ("speedup" in obj or "fused_ms" in obj):
            row = dict(obj)
            if not row.get("variant"):
                row["variant"] = "-" if (parent_key in _CONTAINER_KEYS or not parent_key) \
                    else parent_key
            out.append(row)
            return  # a row's own sub-dicts are diagnostics, not further rows
        for k, v in obj.items():
            walk_rows(v, out, k)
    elif isinstance(obj, list):
        for v in obj:
            walk_rows(v, out, parent_key)


def layer_rows(payload: dict) -> list[dict]:
    """The whole-layer bench is shaped differently: {regime: {rows: {config: {ms: ...}}}},
    with one config being the all-unfused baseline rather than an explicit `unfused_ms`.
    Reconstruct fused/unfused pairs from it so those cells land in the same table."""
    out: list[dict] = []
    blocks = payload.get("regimes")
    if not isinstance(blocks, dict):
        blocks = {k: v for k, v in payload.items()
                  if isinstance(v, dict) and (k.startswith("decode_") or
                                              k.startswith("prefill_"))}
    for regime, block in (blocks or {}).items():
        rows = block.get("rows") if isinstance(block, dict) else None
        if not isinstance(rows, dict):
            continue
        base_key = next((k for k in rows if "all_unfused" in k), None) or \
            next((k for k in rows if "unfused" in k.lower()), None)
        base = rows.get(base_key) if base_key else None
        base_ms = base.get("ms") if isinstance(base, dict) else None
        if not base_ms:
            continue
        for name, cfg in rows.items():
            if not isinstance(cfg, dict) or cfg.get("ms") is None:
                continue
            if name == base_key:
                continue  # the baseline is not a speedup against itself
            out.append({
                "regime": regime, "variant": name,
                "fused_ms": cfg["ms"], "unfused_ms": base_ms,
                "speedup": base_ms / cfg["ms"] if cfg["ms"] else None,
                "baseline_config": base_key,
                "n_kernels": cfg.get("n_kernels"),
                "correct": cfg.get("correct"),
            })
    return out


def collect_cells(log: Log, key: str, path: Path, tick_us: float,
                  unresolved_ticks: int) -> list[dict]:
    """Turn one result file into per-(fusion, regime) cells, with the resolution verdict.

    Two things are deliberate here. First, a *paired* statistic wins over a ratio of two
    independently-taken medians when the bench recorded one -- interleaved A/B/A/B timing is
    the only thing that makes a sub-2 % speedup survive thermal drift. Second, a cell whose
    two arms are within a few CUDA-event ticks of each other is reported as UNRESOLVED, not
    as a four-digit ratio: at decode_bs1 the arms are 9-17 ticks long and the ratio is
    quantised to roughly +-8 %.
    """
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        log(f"!! {path.name}: unreadable ({exc})")
        return []
    rows: list[dict] = []
    walk_rows(payload, rows)
    if (key == "layer" or not rows) and isinstance(payload, dict):
        seen = {(str(r.get("regime")), str(r.get("variant"))) for r in rows}
        rows += [r for r in layer_rows(payload)
                 if (str(r["regime"]), str(r["variant"])) not in seen]

    cells = []
    for r in rows:
        regime = str(r.get("regime", ""))
        f_ms, u_ms = r.get("fused_ms"), r.get("unfused_ms")
        try:
            f_ms = float(f_ms) if f_ms is not None else None
            u_ms = float(u_ms) if u_ms is not None else None
        except (TypeError, ValueError):
            f_ms = u_ms = None

        sp, sp_src = None, None
        # Explicit paired markers win over the bare `speedup` key: a row that records
        # both is saying "here is my interleaved number, use it".  After the paired
        # upgrade `speedup` itself already holds the paired median, so the order only
        # matters for rows that carry both spellings.
        for k in ("paired_speedup", "speedup_paired", "speedup"):
            v = r.get(k)
            try:
                if v is not None and float(v) > 0:
                    sp, sp_src = float(v), k
                    break
            except (TypeError, ValueError):
                continue
        if sp is None and f_ms and u_ms:
            sp, sp_src = u_ms / f_ms, "derived from fused_ms/unfused_ms"

        cell = {
            "family": key,
            "variant": str(r.get("variant", "-")),
            "fusion": fusion_label(key, r.get("variant", "")),
            "regime": regime,
            "fused_ms": f_ms,
            "unfused_ms": u_ms,
            "speedup_raw": sp,
            "speedup_source": sp_src,
            "flags": [],
            "resolved": True,
        }

        if f_ms is not None and u_ms is not None and tick_us > 0:
            cell["fused_ticks"] = f_ms * 1000.0 / tick_us
            cell["unfused_ticks"] = u_ms * 1000.0 / tick_us
            cell["gap_ticks"] = abs(f_ms - u_ms) * 1000.0 / tick_us
            if cell["gap_ticks"] < unresolved_ticks:
                cell["resolved"] = False
                cell["flags"].append(
                    f"UNRESOLVED: arms differ by {cell['gap_ticks']:.1f} timer ticks "
                    f"(< {unresolved_ticks}); tick = {tick_us} us")
            short = min(cell["fused_ticks"], cell["unfused_ticks"])
            if short < 10:
                cell["flags"].append(
                    f"COARSE: shorter arm is {short:.1f} ticks; the ratio is quantised to "
                    f"roughly +-{100.0 / max(short, 1e-9):.0f}%")
        # `glm52_h200/common.py:paired_row()` puts the A/B/A/B PAIRED median in `speedup`
        # and keeps the old ratio-of-medians alongside. A cell that is not paired is not
        # protected against the drift that broke #1 on the 4060; say so.
        if "paired" in r:
            cell["paired"] = bool(r.get("paired"))
            if not r.get("paired"):
                cell["flags"].append(
                    "SEQUENTIAL: arms were not interleaved A/B/A/B, so monotone clock or "
                    "thermal drift does not cancel in this ratio")
        for k in ("speedup_of_medians", "speedup_trimmed", "speedup_p10_p90"):
            if k in r:
                cell[k] = r[k]
        # The sequential ratio the paired number replaced: kept so a report can show
        # both, and so the paired upgrade is auditable in the raw cell.
        if "speedup_sequential" in r:
            cell["speedup_sequential"] = r["speedup_sequential"]
        som = r.get("speedup_of_medians")
        try:
            if sp and som and abs(float(som) - sp) / sp > 0.02:
                cell["flags"].append(
                    f"DRIFT: paired {sp:.4f}x vs ratio-of-medians {float(som):.4f}x differ "
                    f"by >2%; the machine moved during this cell")
        except (TypeError, ValueError):
            pass
        # A speedup above its own traffic ceiling is a red flag, not a triumph -- that check
        # caught a real bad measurement on C500 and again on the 4060.
        ceil = r.get("ceiling")
        try:
            if ceil is not None and sp is not None and sp > float(ceil) * 1.02:
                cell["flags"].append(
                    f"ABOVE CEILING: {sp:.3f}x vs modelled ceiling {float(ceil):.3f}x")
        except (TypeError, ValueError):
            pass
        for k in ("rel_err", "rel_err_fused", "correct", "n_tried", "n_failed"):
            if k in r:
                cell[k] = r[k]
        for k in ("ceiling_with_launch", "traffic_ratio_model"):
            if k in r:
                cell[k] = r[k]
        cell["speedup"] = cell["speedup_raw"] if cell["resolved"] else None
        cells.append(cell)
    return cells


# ======================================================================================
# summary table
# ======================================================================================
def order_regimes(regimes) -> list[str]:
    known = [r for r in KNOWN_REGIMES if r in regimes]
    return known + sorted(r for r in regimes if r not in KNOWN_REGIMES)


def fmt_cell(c: dict | None) -> str:
    if c is None:
        return "-"
    if not c.get("resolved"):
        return "UNRES"
    sp = c.get("speedup")
    if sp is None:
        return "?"
    marks = "".join(m for pre, m in (("COARSE", "~"), ("DRIFT", "*"),
                                     ("SEQUENTIAL", "s"), ("ABOVE", "!"))
                    if any(f.startswith(pre) for f in c.get("flags", [])))
    return f"{sp:.3f}{marks}"


def render_table(cells: list[dict], tick: dict, unresolved_ticks: int) -> list[str]:
    if not cells:
        return ["(no speedup cells were produced -- every family is missing or unparsed)"]
    regimes = order_regimes({c["regime"] for c in cells if c["regime"]})
    # one line per (fusion label, variant), so f8_atomic and f8_token_major stay distinct
    keys, seen = [], set()
    for c in sorted(cells, key=lambda c: (list(FAMILY_BY_KEY).index(c["family"])
                                          if c["family"] in FAMILY_BY_KEY else 99,
                                          c["fusion"], c["variant"])):
        k = (c["family"], c["fusion"], c["variant"])
        if k not in seen:
            seen.add(k)
            keys.append(k)
    idx = {(c["family"], c["fusion"], c["variant"], c["regime"]): c for c in cells}

    label_w = max(28, max(len(f"{f}  [{v}]") for _, f, v in keys) + 1)
    cw = 9
    head = "fusion / variant".ljust(label_w) + "".join(
        REGIME_ABBR.get(r, r).rjust(cw) for r in regimes)
    lines = [head, "-" * len(head)]
    for fam, fus, var in keys:
        lab = f"{fus}  [{var}]" if var not in ("-", "None", "") else fus
        row = lab[:label_w - 1].ljust(label_w)
        for r in regimes:
            row += fmt_cell(idx.get((fam, fus, var, r))).rjust(cw)
        lines.append(row)
    lines.append("-" * len(head))
    lines.append("columns: " + "  ".join(f"{REGIME_ABBR.get(r, r)}={r}" for r in regimes))
    lines.append(
        f"UNRES = the two arms differ by < {unresolved_ticks} CUDA-event ticks "
        f"(tick = {tick['tick_us']} us, {tick['source']}); the ratio is not resolvable, "
        f"so no ratio is printed.")
    lines.append(
        "~ = shorter arm is under 10 ticks, so the printed ratio is coarsely quantised.   "
        "* = paired and sequential ratios disagree by >2 % (the machine moved).")
    lines.append(
        "s = arms were NOT interleaved A/B/A/B, so drift does not cancel.   "
        "! = speedup exceeds its own modelled traffic ceiling: treat as suspect.")
    counts: dict[str, int] = {}
    for c in cells:
        for f in c.get("flags", []):
            counts[f.split(":", 1)[0]] = counts.get(f.split(":", 1)[0], 0) + 1
    if counts:
        lines.append("flagged cells: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                     + f"  (of {len(cells)} total)")
    return lines


# ======================================================================================
def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="run_h200.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="GLM-5.2 MoE-layer fusion study -- serial H200 campaign driver.",
        epilog="Typical use:\n"
               "  python3 run_h200.py --gpu auto           # pick the idlest card, then run\n"
               "  python3 run_h200.py --gpu 3              # pin nvidia-smi GPU 3\n"
               "  python3 run_h200.py --quick              # short sweeps, stack smoke test\n"
               "  python3 run_h200.py --families f03,f10   # just these\n"
               "  python3 run_h200.py --list               # show the plan, run nothing\n"
               "\nOn a shared multi-GPU node ALWAYS pass --gpu: every number in a campaign\n"
               "is only comparable if every family ran on the same, idle card.\n",
    )
    ap.add_argument("--families", default="",
                    help=f"comma-separated subset of {','.join(f.key for f in FAMILIES)}")
    ap.add_argument("--regimes", default="",
                    help=f"comma-separated subset of {','.join(KNOWN_REGIMES)}")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--log-dir", default=str(DEFAULT_LOGDIR))
    ap.add_argument("--quick", action="store_true",
                    help="pass --quick to every bench that offers it (shorter sweeps)")
    ap.add_argument("--force", action="store_true",
                    help="run even if the device is not sm_90 (results are NOT an H200 "
                         "measurement)")
    ap.add_argument("--force-rerun", action="store_true",
                    help="re-run families whose result JSON already exists")
    ap.add_argument("--timeout", type=int, default=0,
                    help="per-family timeout in seconds (default: per-family, 3-16 h)")
    ap.add_argument("--heartbeat", type=int, default=60,
                    help="seconds between progress lines (default 60)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter used for the benches (default: this one)")
    ap.add_argument("--gpu", default=None, metavar="N|auto",
                    help="which GPU the whole campaign runs on. N pins nvidia-smi index N "
                         "(exported as CUDA_VISIBLE_DEVICES=N plus "
                         "CUDA_DEVICE_ORDER=PCI_BUS_ID, so every child sees exactly that "
                         "card as cuda:0 and no bench needs to know); 'auto' ranks the "
                         "host's GPUs by (utilization, memory used) and takes the idlest, "
                         "printing the full ranking. Omit to leave the choice to CUDA -- not "
                         "recommended on a shared multi-GPU node.")
    ap.add_argument("--allow-busy", action="store_true",
                    help="run even though the chosen GPU already has another tenant. A "
                         "shared card is what produced the preflight's impossible 40.55 us "
                         "harness floor and a CUDA-event tick matching 3 samples in 100; "
                         "short-kernel results from such a run are not defensible.")
    ap.add_argument("--unresolved-ticks", type=int, default=3,
                    help="cells whose arms differ by fewer than this many CUDA-event ticks "
                         "are reported UNRESOLVED (default 3)")
    ap.add_argument("--family-args", action="append", default=[], metavar="KEY=ARGS",
                    help="extra CLI args for one family, e.g. --family-args f11=--router-only")
    ap.add_argument("--disable-features", default="",
                    help="comma list of Hopper paths to switch off for every bench "
                         "(e.g. tma,clusters,ws) -- exported as "
                         "GLM52_H200_DISABLE_FEATURES. Use this if a Hopper path misbehaves "
                         "on the real device; the classic fallback then runs in BOTH arms.")
    ap.add_argument("--flush-mb", type=int, default=0,
                    help="override the L2-flush buffer size in MiB (GLM52_H200_FLUSH_MB). "
                         "Only for diagnosis: the default is derived from the device's L2 "
                         "and a too-small flush measures a warm cache.")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="do not run preflight even if its JSON is missing (not recommended)")
    ap.add_argument("--no-device-fence", action="store_true",
                    help="DANGEROUS: reuse existing results without checking which GPU "
                         "produced them")
    ap.add_argument("--allow-concurrent", action="store_true",
                    help="do not refuse when another bench appears to be running")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve everything and print what would run, but launch nothing")
    ap.add_argument("--summary-only", action="store_true",
                    help="re-emit summary.json and the table from existing results")
    return ap.parse_args(argv)


def another_bench_running() -> list[str]:
    """Crude but sufficient: two benches on one GPU corrupt every timing in both."""
    if not shutil.which("pgrep"):
        return []
    try:
        out = subprocess.run(["pgrep", "-af", "glm52_h200/bench"],
                             capture_output=True, text=True, timeout=20)
    except Exception:  # noqa: BLE001
        return []
    me = str(os.getpid())
    return [ln for ln in out.stdout.strip().splitlines()
            if ln.strip() and not ln.split(None, 1)[0] == me]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    results = Path(args.results_dir).expanduser().resolve()
    logdir = Path(args.log_dir).expanduser().resolve()
    results.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    log = Log(logdir / "driver.log")
    warnings: list[str] = []
    t_start = time.time()

    # --- family selection ------------------------------------------------------------
    if args.families:
        want = [k.strip() for k in args.families.split(",") if k.strip()]
        unknown = [k for k in want if k not in FAMILY_BY_KEY]
        if unknown:
            log(f"!! unknown families {unknown}; known: {list(FAMILY_BY_KEY)}")
            return 2
        # Always canonical order, whatever order the operator typed: the ordering IS the
        # policy (cheapest failure first, whole-layer last), not a preference.
        plan = [f for f in FAMILIES if f.key in set(want)]
        if [f.key for f in plan] != want:
            log(f"  note: families reordered to the campaign order "
                f"{[f.key for f in plan]} (cheapest failure first).")
    else:
        plan = list(FAMILIES)
    if args.regimes:
        bad = [r.strip() for r in args.regimes.split(",")
               if r.strip() and r.strip() not in KNOWN_REGIMES]
        if bad:
            warnings.append(f"regimes not in the canonical set were requested: {bad}")
            log(f"!! regimes {bad} are not in {KNOWN_REGIMES}; passing them through anyway")

    extra_args: dict[str, list[str]] = {}
    for spec in args.family_args:
        k, _, rest = spec.partition("=")
        if k not in FAMILY_BY_KEY:
            log(f"!! --family-args for unknown family '{k}' ignored")
            continue
        extra_args.setdefault(k, []).extend(rest.split())

    if args.list:
        rule(log, "PLAN")
        for f in plan:
            s = find_script(f)
            r = find_result(f, results)
            log(f"  {f.key:<7} {f.title}")
            log(f"          script  {s or 'NOT FOUND'}")
            log(f"          result  {r or '(none yet)'}")
            log(f"          timeout {fmt_dur(args.timeout or f.timeout_s)}"
                + (f"   note: {f.note}" if f.note else ""))
        log.close()
        return 0

    # --- GPU selection ----------------------------------------------------------------
    # Before the preflight, because the preflight must probe the SAME card the benches will
    # run on: it is where every hardware constant in the suite comes from, and a probe of
    # GPU 1 governing kernels that run on GPU 5 is worse than no probe at all.
    hw_start = hwinfo()
    gpu = resolve_gpu(log, args, hw_start, warnings,
                      measuring=not (args.summary_only or args.dry_run))
    if gpu.get("refuse"):
        log.close()
        return 4
    hw_row = pick_hw_row(hw_start, gpu.get("index"))

    # --- preflight -------------------------------------------------------------------
    pf = read_preflight()
    if pf is None and not args.skip_preflight:
        log(f"  no {PREFLIGHT_JSON.name}; running the preflight probe first "
            f"(it is what every bench reads its hardware constants from).")
        pf = run_preflight(log, args.python, logdir, quick=args.quick, gpu=gpu.get("index"))
    elif pf is not None and hw_row:
        # A cached probe from another box -- or from another CARD in this box -- would build
        # wrong-shaped grids under a correct-looking device label. Cheapest possible checks,
        # run every time: model name, then UUID, because eight identical H200s all pass a
        # name comparison and only one of them produced the cached numbers.
        pf_name = _norm_dev((pf.get("device") or {}).get("name"))
        pf_uuid = _norm_uuid((pf.get("gpu_selection") or {}).get("uuid")
                             or (pf.get("device") or {}).get("uuid"))
        row_uuid = _norm_uuid(hw_row.get("uuid"))
        why = ""
        if pf_name != _norm_dev(hw_row.get("name")):
            why = f"model differs ({pf_name!r} vs {_norm_dev(hw_row.get('name'))!r})"
        elif gpu.get("index") is not None and pf_uuid and row_uuid and pf_uuid != row_uuid:
            why = f"same model but a different card (probe {pf_uuid}, pinned {row_uuid})"
        if why:
            log(f"!! cached preflight does not describe the pinned GPU: {why} -- re-probing "
                f"before anything is tuned.")
            pf = run_preflight(log, args.python, logdir, quick=args.quick,
                               gpu=gpu.get("index")) or pf

    tick_us = None
    tick_src = "default (no preflight calibration)"
    if pf:
        v = ((pf.get("calibration") or {}).get("timer_tick_us"))
        try:
            if v and float(v) > 0:
                tick_us, tick_src = float(v), "measured by preflight"
        except (TypeError, ValueError):
            pass
    if tick_us is None:
        tick_us = DEFAULT_TICK_US
        warnings.append(f"CUDA-event tick not measured; assumed {DEFAULT_TICK_US} us "
                        f"(the coarsest seen in this study, so UNRESOLVED over-flags rather "
                        f"than under-flags)")
    tick = {"tick_us": tick_us, "source": tick_src, "unresolved_ticks": args.unresolved_ticks}

    # A tick measured on a shared GPU is not a tick. The preflight now says so about its own
    # numbers; carry that verdict into every place a tick-based judgement is printed, because
    # a silently-wrong UNRESOLVED threshold either hides real differences or invents them --
    # and unlike a missing tick, a contaminated one looks perfectly well-formed.
    cal = (pf or {}).get("calibration") or {}
    trusted, distrust = True, []
    if cal.get("launch_timer_trustworthy") is False:
        trusted = False
        distrust = list(cal.get("launch_timer_doubts") or ["flagged by the preflight"])
    else:
        frac = cal.get("timer_tick_match_frac")
        if isinstance(frac, (int, float)) and frac < 0.98:
            trusted = False
            distrust = [f"the preflight's winning tick quantum matched only "
                        f"{frac * 100:.0f}% of its samples; a real tick matches ~100%"]
    tick["trusted"] = trusted
    tick["distrust_reasons"] = distrust
    tick["match_frac"] = cal.get("timer_tick_match_frac")
    if not trusted:
        warnings.append(
            f"the CUDA-event tick ({tick_us} us) used for every UNRESOLVED verdict is itself "
            f"untrustworthy: {'; '.join(distrust)}. Re-run the preflight on an idle GPU "
            f"before quoting any decode_bs1 cell.")

    banner(log, pf, hw_start, results, tick, gpu)
    dev = device_gate(log, pf, hw_row, args.force, warnings)
    device_name = dev.get("name") or ""

    if args.disable_features:
        msg = (f"Hopper paths disabled for this run: {args.disable_features}. Both arms of "
               f"every pair fall back to the classic path, so the ratios stay fair -- but "
               f"they are no longer a measurement of the Hopper path.")
        warnings.append(msg)
        log(f"  [features] {msg}")
    if args.flush_mb:
        warnings.append(f"L2-flush buffer forced to {args.flush_mb} MiB instead of being "
                        f"derived from the device's L2")
        log(f"  [flush] buffer forced to {args.flush_mb} MiB (normally >= 4x L2, derived)")
    if args.quick:
        log("  [quick] short sweeps: narrower autotuning grids and fewer reps. Fine for a "
            "stack smoke test, NOT a publishable measurement.")

    if not args.summary_only:
        busy = another_bench_running()
        if busy and not args.allow_concurrent:
            log("!! another glm52_h200 bench appears to be running:")
            for b in busy:
                log(f"     {b}")
            log("!! Two benchmarks on one GPU corrupt every timing in both. Refusing.")
            log("!! Wait for it, or pass --allow-concurrent if you know it is a stale match.")
            log.close()
            return 3
        if busy:
            warnings.append("--allow-concurrent: another bench process was detected")

    moved = quarantine_foreign_results(
        log, results, device_name, enabled=not args.no_device_fence, warnings=warnings)

    # --- run -------------------------------------------------------------------------
    fam_records: list[dict] = []
    interrupted = False
    if not args.summary_only:
        rule(log, f"RUNNING {len(plan)} FAMILIES SERIALLY "
                  f"(order = cheapest failure first; whole-layer last)")
        for i, fam in enumerate(plan, 1):
            log("")
            log(f"--- [{i}/{len(plan)}] {fam.key}: {fam.title} "
                f"{'-' * max(0, 40 - len(fam.title))}")
            if fam.note:
                log(f"  note    {fam.note}")
            script = find_script(fam)
            if script is None:
                log(f"  !! no script found for {fam.key} "
                    f"(looked for {', '.join(fam.script_globs)} in "
                    f"{SUITE / 'bench'} and {SUITE}) -- SKIPPING")
                fam_records.append({"family": fam.key, "title": fam.title,
                                    "status": "script_missing",
                                    "searched": list(fam.script_globs)})
                continue
            existing = find_result(fam, results)
            if existing is not None and not args.force_rerun:
                usable, why = result_is_usable(existing, device_name,
                                               fence=not args.no_device_fence)
                if usable:
                    log(f"  skip: {existing.name} already exists on this device ({why}). "
                        f"--force-rerun to redo it.")
                    fam_records.append({"family": fam.key, "title": fam.title,
                                        "status": "skipped_existing",
                                        "result": str(existing), "reason": why})
                    continue
                log(f"  {existing.name} exists but is not reusable: {why} -- re-running.")
            if args.dry_run:
                log(f"  dry-run: would run {script}")
                fam_records.append({"family": fam.key, "title": fam.title,
                                    "status": "dry_run", "script": str(script)})
                continue
            try:
                rec = run_family(log, fam, script, args, results, logdir,
                                 extra_args.get(fam.key, []), gpu)
            except KeyboardInterrupt:
                rec = {"family": fam.key, "title": fam.title, "status": "interrupted"}
            except Exception as exc:  # noqa: BLE001 -- one family must never kill the run
                log(f"  !! unexpected driver failure on {fam.key}: "
                    f"{type(exc).__name__}: {exc}")
                rec = {"family": fam.key, "title": fam.title, "status": "driver_error",
                       "error": f"{type(exc).__name__}: {exc}"}
            # Between families, not during: a campaign is 20+ hours and a neighbour that
            # arrives in hour six leaves the later families incomparable with the earlier
            # ones. Silence here is what produced the drift this study has been burned by
            # twice; the check costs one nvidia-smi call per family.
            before_events = len(gpu.get("tenant_events") or [])
            check_tenants(log, gpu, f"after {fam.key}", warnings)
            if len(gpu.get("tenant_events") or []) > before_events:
                rec["tenant_appeared_after"] = gpu["tenant_events"][-1]
            got = find_result(fam, results)
            rec["result"] = str(got) if got else None
            if rec.get("status") == "ok" and got is None:
                rec["status"] = "ok_but_no_result_file"
                log(f"  !! exited 0 but wrote no result matching "
                    f"{', '.join(fam.result_globs)}")
            fam_records.append(rec)
            if rec.get("status") == "interrupted":
                log("")
                log("!! interrupted by the operator -- stopping here. The summary below "
                    "covers whatever completed; re-running the driver resumes at the first "
                    "family with no device-fenced result.")
                interrupted = True
                break

    # --- collect ---------------------------------------------------------------------
    rule(log, "COLLECTING RESULTS")
    cells: list[dict] = []
    missing: list[str] = []
    for fam in plan:
        got = find_result(fam, results)
        if got is None:
            missing.append(fam.key)
            log(f"  {fam.key:<7} MISSING")
            continue
        usable, why = result_is_usable(got, device_name, fence=not args.no_device_fence)
        got_cells = collect_cells(log, fam.key, got, tick_us, args.unresolved_ticks)
        cells += got_cells
        if not usable:
            missing.append(fam.key)
            warnings.append(f"{fam.key}: result present but not trusted ({why})")
        log(f"  {fam.key:<7} {got.name:<38} {len(got_cells):3d} cells   "
            f"{'ok' if usable else 'UNTRUSTED: ' + why}")

    hw_end = hwinfo()
    drift = hw_drift(hw_start, hw_end)
    check_tenants(log, gpu, "at the end of the campaign", warnings)

    rule(log, "SPEEDUP TABLE  (fused / unfused, per fusion per regime)")
    table = render_table(cells, tick, args.unresolved_ticks)
    for ln in table:
        log(ln)

    log("")
    rule(log, "FAMILY STATUS")
    for r in fam_records:
        log(f"  {r['family']:<7} {r.get('status', '?'):<22} "
            f"{(r.get('wall_s') or 0) / 60:7.1f} min   {r.get('result') or ''}")
    if missing:
        log("")
        log(f"  !! MISSING / UNTRUSTED: {', '.join(missing)}")
        log(f"     logs for these are in {logdir}; re-running the driver resumes at the "
            f"first missing family.")
    else:
        log("")
        log("  every planned family produced a device-fenced result.")

    log("")
    rule(log, "HWINFO -- START vs END (clock and thermal drift across the whole run)")
    if gpu.get("index") is not None:
        log(f"  (this campaign ran on GPU {gpu['index']}, uuid {gpu.get('uuid')}; the other "
            f"rows are context, not this run)")
    for g in hw_start:
        log(f"  start {hw_line(g)}")
    for g in hw_end:
        log(f"  end   {hw_line(g)}")
    if gpu.get("tenant_events"):
        log("")
        log(f"  !! {len(gpu['tenant_events'])} time(s) during this campaign another tenant "
            f"appeared on GPU {gpu['index']}:")
        for ev in gpu["tenant_events"]:
            log(f"     {ev['at']}  {ev['when']}  "
                f"{len(ev.get('new_processes') or [])} new process(es), "
                f"memory {ev.get('memory_growth_bytes', 0) / 2**30:+.1f} GB")
        log("     Families measured after the first of these are not comparable with the "
            "ones before it; re-run them on a clean card before quoting the table above.")
    for d in drift:
        sm, tmp = d.get("clocks.sm", {}), d.get("temperature.gpu", {})
        pct = f"{sm['pct']:+.1f}%" if sm.get("pct") is not None else "n/a"
        dt = f"{tmp['delta']:+.0f} C" if tmp.get("delta") is not None else "n/a"
        log(f"  GPU{d['index']} SM clock {pct}, temperature {dt}, "
            f"throttle {d.get('throttle_start')} -> {d.get('throttle_end')}")
        if sm.get("pct") is not None and abs(sm["pct"]) >= 5:
            log("  !! The GPU did not hold its clock across this run. Every final timing in "
                "this suite is interleaved A/B/A/B and reported as a paired statistic "
                "precisely so monotone drift cancels -- but any number NOT taken that way "
                "(coarse tuning passes, single-shot diagnostics) is suspect.")
    if not hw_start and not hw_end:
        log("  nvidia-smi was unavailable -- no drift record exists for this run.")

    # --- summary ---------------------------------------------------------------------
    summary = {
        "schema": 1,
        "id": "glm52_h200_summary",
        # Self-fenced exactly like every result file: a summary is only meaningful as a claim
        # about a specific GPU, and the next run must be able to tell whose it is.
        "_meta": {
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device_name,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            # Index AND uuid: on an eight-card node an index alone does not identify a
            # device, and a number in this file has to be traceable to the card that
            # produced it, not merely to the host.
            "gpu_index": gpu.get("index"),
            "gpu_uuid": gpu.get("uuid"),
            "gpu_pinned": bool(gpu.get("pinned")),
            "gpu_was_idle": (None if gpu.get("busy") is None else not gpu["busy"]),
        },
        "driver": {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "repo": str(REPO),
            "results_dir": str(results),
            "log_dir": str(logdir),
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_s": time.time() - t_start,
            "quick": bool(args.quick),
            "regimes_requested": args.regimes or "(all)",
            "disable_features": args.disable_features or None,
            "flush_mb_override": args.flush_mb or None,
            "force_rerun": bool(args.force_rerun),
            "device_fence": not args.no_device_fence,
            "families_planned": [f.key for f in plan],
            "interrupted": interrupted,
            "summary_only": bool(args.summary_only),
            "gpu_requested": args.gpu,
            "allow_busy": bool(args.allow_busy),
        },
        # The whole selection decision, ranking included, so "why GPU 3?" is answerable from
        # the record alone rather than from whoever happened to run the command.
        "gpu": gpu,
        "device": dev,
        "preflight": (preflight_digest(pf) | {"path": str(PREFLIGHT_JSON)}) if pf
                     else {"path": str(PREFLIGHT_JSON), "present": False},
        "timer": tick,
        "hwinfo_start": hw_start,
        "hwinfo_end": hw_end,
        "hwinfo_drift": drift,
        "quarantined": moved,
        "families": fam_records,
        "missing_families": missing,
        "cells": cells,
        "table": table,
        "warnings": warnings,
        "what_is_measured": {
            "cells": "fused/unfused speedup per fusion per regime, each arm independently "
                     "autotuned from the same kernel source; see the per-family JSON for "
                     "tuning tables, grid sizes and correctness checks",
            "unresolved": f"cells whose two arms differ by < {args.unresolved_ticks} "
                          f"CUDA-event ticks ({tick_us} us, {tick_src}) carry speedup=null; "
                          f"their raw operands are still present as speedup_raw"
                          + ("" if tick.get("trusted")
                             else ". THE TICK ITSELF IS NOT TRUSTWORTHY here (see "
                                  "timer.distrust_reasons): it was measured on a GPU shared "
                                  "with another tenant, so these verdicts are advisory until "
                                  "the preflight is re-run on an idle card"),
            "not_measured": "roofline ceilings are MODELLED (glm52_h200/traffic.py); the "
                            "layer-level saving of a set of fusions is additive-estimated "
                            "unless bench_layer measured that combination end to end",
        },
    }
    out = results / "summary.json"
    try:
        out.write_text(json.dumps(summary, indent=2, default=str))
        log("")
        log(f"  wrote {out}")
    except OSError as exc:
        log(f"!! could not write {out}: {exc}")

    if warnings:
        log("")
        rule(log, "WARNINGS (also in summary.json)")
        for w in warnings:
            log(f"  - {w}")

    log("")
    rule(log, "SEND BACK")
    log(f"  1. {results}            (every result JSON + summary.json)")
    log(f"  2. {PREFLIGHT_JSON}")
    log(f"  3. {logdir}             (driver.log + one log per family)")
    log(f"  total wall {(time.time() - t_start) / 3600:.2f} h")
    log.close()
    return 0 if not missing else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n interrupted", flush=True)
        raise SystemExit(130)
