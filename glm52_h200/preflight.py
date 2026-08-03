#!/usr/bin/env python3
"""H200 preflight probe -- RUN THIS FIRST, then send back the JSON it writes.

Self-contained: imports nothing from this project, installs nothing, mutates nothing, and is
designed never to crash. Every probe is individually guarded; a failure is *recorded* and the
run continues, because "this feature raised TypeError: unexpected keyword 'warp_specialize'"
is exactly the information the kernels need to be written correctly.

It answers five questions the benchmark suite cannot be written without:

  0. WHICH GPU are we even on?   The node has eight cards and other tenants. This is question
     zero because the first run of this script got it wrong: it probed whichever device CUDA
     handed out, that device already had ~51 GB allocated by somebody else, and the two SHORT
     calibrations came back impossible -- an 8.9 us launch against a 40.55 us "harness floor",
     and a CUDA-event tick matching 3 % of samples where a real tick matches ~100 %. The long
     measurements (bandwidth, GEMM) survived it; the ones every UNRESOLVED verdict downstream
     is built from did not. So the GPUs are enumerated, the idlest is chosen by default, and
     the tenancy of the chosen one is recorded next to the numbers it produced.
  1. What is the stack?          torch / triton / CUDA / driver versions.
  2. What is the device?         SMs, shared memory ceiling, registers, L2, clocks, ECC, MIG.
  3. Which H200 features are REALLY usable from this Triton?  TMA, warp specialization,
     thread-block clusters, fp8 dot. Probed by COMPILING AND LAUNCHING a tiny kernel for
     each -- attribute existence is not evidence, as several Triton releases expose symbols
     that fail at compile time. (And a probe that fails must be a real failure: the first TMA
     probe here mixed the host-side and device-side descriptor APIs and reported a false
     negative on hardware that supports TMA perfectly well. Both forms are now probed
     separately, under names that say which is which.)
  4. What are this machine's achievable peaks?  Bandwidth, bf16 GEMM (Triton and cuBLAS),
     kernel launch cost, CUDA-event timer granularity. Every roofline ceiling in the study is
     computed from these, never from vendor spec.

Usage:
    python3 glm52_h200/preflight.py                 # enumerate all, probe the idlest GPU
    python3 glm52_h200/preflight.py --gpu 3         # probe nvidia-smi GPU 3 specifically
    python3 glm52_h200/preflight.py --gpu none      # whatever CUDA hands out (old behaviour)
    python3 glm52_h200/preflight.py --quick         # skips the slower calibration sweeps

Send back:  glm52_h200/preflight_h200.json   (and the console output if convenient)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "preflight_h200.json"

R: dict = {"schema": 1, "probe_errors": {}}


def _err(key: str, exc: BaseException) -> None:
    R["probe_errors"][key] = f"{type(exc).__name__}: {exc}"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ======================================================================================
# 0. GPU selection -- which of the eight cards is clean enough to measure on
# ======================================================================================
# Deliberately duplicated from glm52_h200/hwinfo.py rather than imported. This script's one
# invariant is that it depends on nothing in the project: the operator can scp preflight.py
# alone onto a fresh node and run it, and it has to keep working when the rest of the suite
# is broken -- which is exactly the state it is usually run in. ~80 lines is a cheap price.
#
# Indices below are nvidia-smi (physical/NVML) indices, NOT CUDA ordinals. CUDA renumbers
# under CUDA_VISIBLE_DEVICES and orders FASTEST_FIRST by default, so pinning also sets
# CUDA_DEVICE_ORDER=PCI_BUS_ID; without it "--gpu 3" can select a different card than the
# one this script inspected and reported on.
_MIB = 1024 * 1024
IDLE_MAX_USED_BYTES = 1 * 2 ** 30     # an untouched H200 here reports 4 MiB; a tenant, 20+ GB
IDLE_MAX_UTIL_PCT = 5.0
IDLE_MIN_FREE_BYTES = 32 * 2 ** 30    # 19.3 GB of expert weights before any activation


def _first_num(v):
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return float(m.group()) if m else None


def _smi_csv(query: str, fields: list, timeout: int = 30) -> tuple:
    """(rows as dicts, error). Never raises: a missing nvidia-smi is an answer, not a crash."""
    if not shutil.which("nvidia-smi"):
        return [], "nvidia-smi not on PATH"
    try:
        p = subprocess.run(["nvidia-smi", f"--{query}={','.join(fields)}",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return [], f"{type(e).__name__}: {e}"
    blob = f"{p.stdout}\n{p.stderr}".lower()
    if p.returncode != 0 or "not a valid field" in blob or "not supported" in blob:
        return [], (p.stderr or p.stdout).strip()[:200] or f"rc={p.returncode}"
    rows = []
    for line in p.stdout.strip().splitlines():
        line = line.strip()
        if not line or "no running processes" in line.lower():
            continue
        vals = [v.strip() for v in line.split(",")]
        if len(vals) < len(fields):
            continue
        rows.append(dict(zip(fields, vals)))
    return rows, ""


def _norm_uuid(v) -> str:
    s = str(v or "").strip().lower()
    return s[4:] if s.startswith("gpu-") else s


def enumerate_gpus() -> tuple:
    """Every physical GPU with the facts needed to judge it idle. ([], error) if unavailable."""
    rows, err = _smi_csv("query-gpu", [
        "index", "uuid", "name", "utilization.gpu", "memory.total", "memory.used",
        "memory.free", "compute_mode", "mig.mode.current", "persistence_mode",
        "pstate", "clocks.sm", "temperature.gpu",
    ])
    if not rows:
        return [], err or "no GPUs reported"
    apps, app_err = _smi_csv("query-compute-apps",
                             ["gpu_uuid", "pid", "process_name", "used_gpu_memory"])
    by_uuid: dict = {}
    for a in apps:
        by_uuid.setdefault(_norm_uuid(a["gpu_uuid"]), []).append({
            "pid": int(_first_num(a["pid"]) or 0),
            "name": a["process_name"],
            "used_bytes": int((_first_num(a["used_gpu_memory"]) or 0) * _MIB),
        })
    out = []
    for g in rows:
        out.append({
            "index": int(_first_num(g.get("index")) or 0),
            "uuid": g.get("uuid"),
            "name": g.get("name"),
            "utilization_pct": _first_num(g.get("utilization.gpu")),
            "memory_total_bytes": int((_first_num(g.get("memory.total")) or 0) * _MIB),
            "memory_used_bytes": int((_first_num(g.get("memory.used")) or 0) * _MIB),
            "memory_free_bytes": int((_first_num(g.get("memory.free")) or 0) * _MIB),
            "compute_mode": g.get("compute_mode"),
            "mig_mode": g.get("mig.mode.current"),
            "persistence_mode": g.get("persistence_mode"),
            "pstate": g.get("pstate"),
            "sm_mhz": _first_num(g.get("clocks.sm")),
            "temp_c": _first_num(g.get("temperature.gpu")),
            "processes": by_uuid.get(_norm_uuid(g.get("uuid")), []),
            # Non-empty means the tenancy of this card is UNKNOWN, never "proven empty".
            "process_query_error": app_err or None,
        })
    return out, ""


def busy_reasons(g: dict) -> list:
    """Facts about *other people's* use of this card. [] means nobody else is on it.

    Tenancy only. "Not enough free VRAM" lives in `capacity_notes` instead: it is a fact
    about the study's appetite, not evidence of a neighbour, and a probe run on a small
    development GPU must not report a stranger who does not exist.
    """
    out = []
    procs = g.get("processes") or []
    if procs:
        out.append(f"{len(procs)} other compute process(es): " + ", ".join(
            f"pid {p['pid']} {p['name']} ({p['used_bytes'] / 2**30:.1f} GB)" for p in procs[:4]))
    if (g.get("memory_used_bytes") or 0) > IDLE_MAX_USED_BYTES:
        out.append(f"{g['memory_used_bytes'] / 2**30:.1f} GB already allocated")
    if (g.get("utilization_pct") or 0) > IDLE_MAX_UTIL_PCT:
        out.append(f"utilization {g['utilization_pct']:.0f}%")
    if str(g.get("compute_mode") or "").lower() not in ("default", "", "[n/a]", "n/a"):
        out.append(f"compute mode {g['compute_mode']}")
    if str(g.get("mig_mode") or "").lower() == "enabled":
        out.append("MIG enabled")
    return out


def capacity_notes(g: dict) -> list:
    """Does the study fit here? Ranked on, reported, never fatal."""
    if (g.get("memory_free_bytes") or 0) < IDLE_MIN_FREE_BYTES:
        return [f"only {(g.get('memory_free_bytes') or 0) / 2**30:.1f} GB free "
                f"(the whole-layer bench wants {IDLE_MIN_FREE_BYTES / 2**30:.0f} GB)"]
    return []


def rank_gpus(gpus: list) -> list:
    """Idlest first: unoccupied before occupied, roomy before cramped, then (utilization,
    memory used) ascending, with the index breaking ties deterministically."""
    ranked = []
    for g in gpus:
        r = dict(g)
        r["reasons"] = busy_reasons(g)
        r["busy"] = bool(r["reasons"])
        r["capacity_notes"] = capacity_notes(g)
        r["capacity_short"] = bool(r["capacity_notes"])
        ranked.append(r)
    ranked.sort(key=lambda r: (r["busy"], r["capacity_short"],
                               r.get("utilization_pct") or 0.0,
                               r.get("memory_used_bytes") or 0, r.get("index", 0)))
    for i, r in enumerate(ranked):
        r["rank"] = i
    return ranked


def print_gpu_table(ranked: list) -> None:
    head = (f"  {'rank':>4} {'idx':>3}  {'util':>5} {'used GB':>9} {'free GB':>9} "
            f"{'proc':>4}  {'temp':>5}  state")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in ranked:
        util = "?" if r.get("utilization_pct") is None else f"{r['utilization_pct']:.0f}%"
        state = ("BUSY: " + "; ".join(r["reasons"])) if r["busy"] else "idle"
        if r.get("capacity_short"):
            state += ("  " if r["busy"] else " -- ") + "; ".join(r["capacity_notes"])
        print(f"  {r['rank']:>4} {r['index']:>3}  {util:>5} "
              f"{(r.get('memory_used_bytes') or 0) / 2**30:9.1f} "
              f"{(r.get('memory_free_bytes') or 0) / 2**30:9.1f} "
              f"{len(r.get('processes') or []):>4}  "
              f"{(r.get('temp_c') or 0):5.0f}  {state[:110]}")
    for r in ranked:
        print(f"       gpu {r['index']}  {r.get('name')}  uuid {r.get('uuid')}")


def foreign_processes(index: int) -> tuple:
    """Compute processes on GPU `index` that are not this one. (list, error)."""
    gpus, err = enumerate_gpus()
    if not gpus:
        return [], err
    me = os.getpid()
    for g in gpus:
        if g.get("index") == index:
            return ([p for p in (g.get("processes") or []) if p.get("pid") != me],
                    g.get("process_query_error") or "")
    return [], f"no GPU with index {index}"


def select_gpu(want: str) -> dict:
    """Resolve --gpu into a physical index, and pin it before torch is ever imported.

    `want` is "auto" (default), "none" (leave CUDA alone), or a decimal nvidia-smi index.
    Returns the whole decision -- ranking, chosen card, tenancy, and what was pinned -- so
    the JSON says which device produced the numbers and how confidently it was picked.
    """
    section("0. GPU SELECTION (8-card node; the wrong card is how the last probe was ruined)")
    gpus, err = enumerate_gpus()
    sel: dict = {"requested": want, "error": err or None,
                 "thresholds": {"max_used_bytes": IDLE_MAX_USED_BYTES,
                                "max_util_pct": IDLE_MAX_UTIL_PCT,
                                "min_free_bytes": IDLE_MIN_FREE_BYTES},
                 "env_before": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER")}}
    ranked = rank_gpus(gpus) if gpus else []
    sel["ranking"] = [{k: v for k, v in r.items() if k != "process_query_error"}
                      for r in ranked]
    if not gpus:
        print(f"  nvidia-smi gave no GPU list ({err}); leaving device selection to CUDA.")
        sel.update(index=None, uuid=None, pinned=False,
                   reason=f"could not enumerate GPUs: {err}")
        return sel
    print(f"  {len(gpus)} GPU(s) on this host:")
    print_gpu_table(ranked)

    pre = os.environ.get("CUDA_VISIBLE_DEVICES")
    if want == "none":
        sel.update(index=None, uuid=None, pinned=False,
                   reason="--gpu none: CUDA_VISIBLE_DEVICES left as the caller set it")
        print(f"\n  --gpu none: not pinning. CUDA_VISIBLE_DEVICES={pre or 'unset (all)'}")
        return sel
    if want == "auto" and pre and pre.strip().isdigit():
        # A parent (run_h200.py) already chose. Honour it rather than second-guessing, or the
        # probe would describe one card while the campaign measures another.
        chosen = int(pre.strip())
        reason = f"inherited CUDA_VISIBLE_DEVICES={chosen} from the caller"
    elif want == "auto":
        chosen = ranked[0]["index"]
        reason = (f"idlest of {len(ranked)}: {(ranked[0].get('utilization_pct') or 0):.0f}% "
                  f"util, {(ranked[0].get('memory_used_bytes') or 0) / 2**30:.1f} GB used, "
                  f"{len(ranked[0].get('processes') or [])} other process(es)")
    else:
        try:
            chosen = int(want)
        except ValueError:
            print(f"  !! --gpu {want!r} is not an index, 'auto' or 'none'; falling back to auto")
            chosen = ranked[0]["index"]
            reason = "unparseable --gpu; fell back to the idlest"
        else:
            reason = f"--gpu {chosen} as requested"
    row = next((r for r in ranked if r["index"] == chosen), None)
    if row is None:
        print(f"  !! no GPU with nvidia-smi index {chosen}; falling back to the idlest")
        row = ranked[0]
        chosen = row["index"]
        reason = f"requested index not present; fell back to GPU {chosen}"

    sel.update(index=chosen, uuid=row.get("uuid"), name=row.get("name"),
               busy=row["busy"], busy_reasons=row["reasons"], reason=reason,
               capacity_notes=row.get("capacity_notes") or [],
               process_query_error=next((g.get("process_query_error") for g in gpus
                                         if g.get("index") == chosen), None))
    # Pin BEFORE torch is imported anywhere -- CUDA_VISIBLE_DEVICES is read once, at context
    # creation, and setting it afterwards silently does nothing.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    sel["pinned"] = True
    print(f"\n  -> GPU {chosen} ({row.get('name')}) uuid {row.get('uuid')}")
    print(f"     {reason}")
    print(f"     CUDA_VISIBLE_DEVICES={chosen}  CUDA_DEVICE_ORDER=PCI_BUS_ID  "
          f"(this process now sees exactly one device as cuda:0)")
    if row["busy"]:
        print("\n  " + "!" * 74)
        print("  !! THE SELECTED GPU IS NOT IDLE: " + "; ".join(row["reasons"]))
        print("  !! This is exactly what corrupted the previous probe: launch cost came back")
        print("  !! at 8.9 us against a 40.55 us harness floor, and the CUDA-event tick")
        print("  !! matched 3 % of samples instead of ~100 %. The bandwidth and GEMM numbers")
        print("  !! below will probably still be usable; launch_us and timer_tick_us will not.")
        print("  !! Re-run on an idle card:  python3 glm52_h200/preflight.py --gpu <idx>")
        print("  " + "!" * 74)
    if row.get("process_query_error"):
        print(f"  .. process list unavailable ({row['process_query_error']}); this card is "
              f"UNPROVEN idle, not proven idle.")
    for note in row.get("capacity_notes") or []:
        print(f"  .. {note} -- the capacity section below will say which benches fit.")
    return sel


# ======================================================================================
# 1. stack
# ======================================================================================
def probe_stack() -> None:
    section("1. STACK")
    s = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        s["torch"] = torch.__version__
        s["torch_cuda"] = torch.version.cuda
        s["torch_hip"] = getattr(torch.version, "hip", None)
        s["cudnn"] = getattr(torch.backends.cudnn, "version", lambda: None)()
        s["cuda_available"] = torch.cuda.is_available()
        s["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception as e:  # noqa: BLE001
        _err("torch_import", e)
    try:
        import triton

        s["triton"] = triton.__version__
        s["triton_file"] = getattr(triton, "__file__", None)
    except Exception as e:  # noqa: BLE001
        _err("triton_import", e)
    R["stack"] = s
    for k, v in s.items():
        print(f"  {k:<20} {v}")


# ======================================================================================
# 2. device
# ======================================================================================
def probe_device() -> None:
    section("2. DEVICE (torch properties)")
    try:
        import torch

        if not torch.cuda.is_available():
            print("  CUDA not available -- nothing further can be probed.")
            return
        p = torch.cuda.get_device_properties(0)
        want = [
            "name", "major", "minor", "multi_processor_count", "warp_size",
            "shared_memory_per_block", "shared_memory_per_block_optin",
            "shared_memory_per_multiprocessor", "regs_per_multiprocessor",
            "max_threads_per_multi_processor", "L2_cache_size", "total_memory",
            "is_multi_gpu_board", "gcnArchName", "uuid",
        ]
        d = {}
        for a in want:
            try:
                v = getattr(p, a)
                d[a] = str(v) if a == "uuid" else v
            except Exception:
                pass
        free, total = torch.cuda.mem_get_info()
        d["mem_free_bytes"], d["mem_total_bytes"] = free, total
        d["compute_capability"] = f"{d.get('major')}.{d.get('minor')}"
        # Which PHYSICAL card this is. torch says "cuda:0" whatever we pinned, so the CUDA
        # ordinal alone cannot identify a device on an eight-GPU node -- carry the nvidia-smi
        # index and the UUID so a number in this file traces to one specific card.
        sel = R.get("gpu_selection") or {}
        d["nvidia_smi_index"] = sel.get("index")
        d["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        d["cuda_device_order"] = os.environ.get("CUDA_DEVICE_ORDER")
        if sel.get("uuid") and _norm_uuid(sel["uuid"]) != _norm_uuid(d.get("uuid")):
            # The pin did not land where the enumeration said it would -- almost always
            # CUDA_DEVICE_ORDER. Loud, because everything else in this file would be
            # attributed to the wrong card.
            msg = (f"pinned nvidia-smi GPU {sel.get('index')} (uuid {sel.get('uuid')}) but "
                   f"torch opened uuid {d.get('uuid')}")
            R["probe_errors"]["gpu_pin_mismatch"] = msg
            print(f"\n  !! {msg}")
        R["device"] = d
        for k, v in d.items():
            print(f"  {k:<36} {v}")
        if (d.get("major"), d.get("minor")) != (9, 0):
            print(f"\n  !! expected sm_90 (Hopper/H200); got sm_{d.get('major')}{d.get('minor')}")
    except Exception as e:  # noqa: BLE001
        _err("device", e)
        traceback.print_exc()


# ======================================================================================
# 3. nvidia-smi -- clocks, power, ECC, MIG, persistence
# ======================================================================================
def probe_smi() -> None:
    section("3. nvidia-smi (clocks / power / ECC / MIG)")
    if not shutil.which("nvidia-smi"):
        print("  nvidia-smi not on PATH")
        return
    # index and uuid first: without them a row in this list cannot be tied to the GPU the
    # rest of the file describes, which on an eight-card node makes the whole block ambiguous.
    fields = [
        "index", "uuid", "name", "driver_version", "vbios_version", "pstate",
        "clocks.sm", "clocks.mem", "clocks.gr", "clocks.max.sm", "clocks.max.mem",
        "clocks_throttle_reasons.active",
        "power.draw", "power.limit", "enforced.power.limit", "power.max_limit",
        "temperature.gpu", "memory.total", "memory.used", "memory.free",
        "ecc.mode.current", "mig.mode.current", "persistence_mode",
        "compute_mode", "utilization.gpu", "pcie.link.gen.current", "pcie.link.width.current",
    ]
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        )
        rows = [r.strip() for r in out.stdout.strip().split("\n") if r.strip()]
        chosen = (R.get("gpu_selection") or {}).get("index")
        gpus = []
        for i, row in enumerate(rows):
            vals = [v.strip() for v in row.split(",")]
            g = dict(zip(fields, vals))
            g["_selected"] = (_first_num(g.get("index")) == chosen) if chosen is not None \
                else None
            gpus.append(g)
            mark = "  <== THIS RUN" if g["_selected"] else ""
            print(f"  --- GPU {g.get('index', i)} ---{mark}")
            for k, v in g.items():
                if not k.startswith("_"):
                    print(f"    {k:<34} {v}")
        R["nvidia_smi"] = gpus
        if out.stderr.strip():
            R["probe_errors"]["nvidia_smi_stderr"] = out.stderr.strip()[:500]
    except Exception as e:  # noqa: BLE001
        _err("nvidia_smi", e)
    # topology is useful if this turns out to be multi-GPU
    try:
        t = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=60)
        R["nvidia_smi_topo"] = t.stdout[:4000]
    except Exception as e:  # noqa: BLE001
        _err("nvidia_smi_topo", e)


# ======================================================================================
# 4. Triton feature probes -- COMPILE AND LAUNCH, do not trust attributes
# ======================================================================================
PROBE_SRC = r'''
import torch, triton, triton.language as tl

# ---- baseline: does anything compile at all? ----
@triton.jit
def k_baseline(X, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    tl.store(Y + i, tl.load(X + i, mask=m, other=0.0) * 2.0, mask=m)

# ---- wgmma / tl.dot on Hopper ----
@triton.jit
def k_dot(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM); rn = tl.arange(0, BN); rk = tl.arange(0, BK)
    a = tl.load(A + rm[:, None] * K + rk[None, :])
    b = tl.load(B + rk[:, None] * N + rn[None, :])
    acc = tl.dot(a, b, out_dtype=tl.float32)
    tl.store(C + rm[:, None] * N + rn[None, :], acc)

# ---- warp specialization via tl.range(warp_specialize=True)  [Triton >= 3.3-ish] ----
@triton.jit
def k_ws_range(X, Y, N, BLOCK: tl.constexpr):
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.range(0, N, BLOCK, warp_specialize=True):
        i = k + tl.arange(0, BLOCK)
        acc += tl.load(X + i, mask=i < N, other=0.0)
    tl.store(Y + tl.arange(0, BLOCK), acc)

# ---- TMA, form 1: HOST-side descriptor object, built by TensorDescriptor.from_tensor() and
# passed in as an argument. Inside the kernel it is used directly -- desc.load(...) -- and is
# NOT fed to tl.make_tensor_descriptor. Mixing the two is a CompilationError on any hardware,
# which is how the previous run of this script reported a false negative for TMA on an H200.
@triton.jit
def k_tma_host(desc, Out, BM: tl.constexpr, BN: tl.constexpr):
    t = desc.load([0, 0])
    tl.store(Out + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], t)

# ---- TMA, form 2: DEVICE-side, descriptor constructed in the kernel from a raw pointer.
# This is what tl.make_tensor_descriptor is for; shape and strides are runtime values.
@triton.jit
def k_tma_device(ptr, Out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    d = tl.make_tensor_descriptor(ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    t = d.load([0, 0])
    tl.store(Out + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], t)
'''


def probe_triton_features(quick: bool) -> None:
    section("4. TRITON FEATURE PROBES (compile + launch, not attribute sniffing)")
    feats: dict = {}
    try:
        import torch
        import triton
        import triton.language as tl
    except Exception as e:  # noqa: BLE001
        _err("triton_features", e)
        return

    # --- 4a. attribute inventory (cheap, informative, but NOT proof) -------------------
    inv = {}
    for path in [
        "triton.language.make_tensor_descriptor",
        "triton.language._experimental_make_tensor_descriptor",
        "triton.language._experimental_descriptor_load",
        "triton.language._experimental_descriptor_store",
        "triton.tools.tensor_descriptor.TensorDescriptor",
        "triton.tools.experimental_descriptor.create_2d_tma_descriptor",
        "triton.language.async_task",
        "triton.set_allocator",
        "triton.language.range",
    ]:
        mod, _, attr = path.rpartition(".")
        try:
            m = __import__(mod, fromlist=["_"])
            inv[path] = hasattr(m, attr)
        except Exception:
            inv[path] = False
    feats["attribute_inventory"] = inv
    print("  -- attribute inventory (existence only) --")
    for k, v in inv.items():
        print(f"    {'yes' if v else ' no'}  {k}")

    # tl.range signature tells us whether warp_specialize is accepted
    try:
        import inspect
        feats["tl_range_signature"] = str(inspect.signature(tl.range))
    except Exception as e:  # noqa: BLE001
        feats["tl_range_signature"] = f"unavailable: {e}"
    print(f"    tl.range{feats['tl_range_signature']}")

    # --- 4b. real compile+launch probes ------------------------------------------------
    src = HERE / "_probe_kernels.py"
    try:
        src.write_text(PROBE_SRC)
        sys.path.insert(0, str(HERE))
        import importlib
        pk = importlib.import_module("_probe_kernels")
    except Exception as e:  # noqa: BLE001
        _err("probe_kernels_import", e)
        feats["compile_probes"] = {"_module": f"failed: {e}"}
        R["triton_features"] = feats
        return

    probes: dict = {}

    def run(name: str, fn) -> None:
        """Compile, launch, synchronise. `fn` may return a dict of extra evidence."""
        try:
            extra = fn()
            torch.cuda.synchronize()
            probes[name] = {"ok": True}
            if isinstance(extra, dict):
                probes[name].update(extra)
            print(f"    OK    {name}"
                  + (f"  {extra}" if isinstance(extra, dict) and extra else ""))
        except Exception as e:  # noqa: BLE001
            probes[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
            print(f"    FAIL  {name}: {type(e).__name__}: {str(e)[:150]}")

    print("  -- compile + launch --")
    N = 4096
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    run("baseline_elementwise", lambda: pk.k_baseline[(N // 256,)](x, y, N, BLOCK=256))

    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(64, 64, device="cuda", dtype=torch.float32)
    run("tl_dot_bf16", lambda: pk.k_dot[(1,)](a, b, c, 64, 64, 64, BM=64, BN=64, BK=64))

    run("warp_specialize_tl_range",
        lambda: pk.k_ws_range[(1,)](x, y, N, BLOCK=256, num_warps=4))

    # num_consumer_groups style warp spec (older/forked Triton)
    def _ws_kwargs():
        pk.k_baseline[(N // 256,)](x, y, N, BLOCK=256, num_consumer_groups=1,
                                   num_buffers_warp_spec=2)
    run("warp_specialize_num_consumer_groups", _ws_kwargs)

    # --- TMA. Two DIFFERENT APIs, probed separately and named so the JSON is unambiguous.
    #
    # The previous run of this script reported `tma_tensor_descriptor: FAIL` on this H200.
    # That was a bug in the probe, not a hardware or Triton limitation: it passed a HOST-side
    # TensorDescriptor object into tl.make_tensor_descriptor(), which is the DEVICE-side
    # constructor and expects a raw pointer. That combination is a CompilationError on every
    # device. The old probe NAME is gone rather than fixed in place, so no reader can compare
    # a new JSON against an old one and think the hardware changed.
    #
    # triton.set_allocator() is called once, first, for both forms. TMA descriptors need a
    # scratch buffer that Triton asks the host for at launch; without an allocator the kernel
    # compiles cleanly and then fails at launch. That is the classic silent TMA failure and it
    # is why the allocator is installed here rather than inside either probe.
    tma_ready, tma_err = False, None
    try:
        if hasattr(triton, "set_allocator"):
            triton.set_allocator(
                lambda size, align, stream: torch.empty(size, device="cuda", dtype=torch.int8)
            )
            tma_ready = True
        else:
            tma_err = "triton.set_allocator missing; TMA descriptors cannot be given scratch"
    except Exception as e:  # noqa: BLE001
        tma_err = f"{type(e).__name__}: {e}"
    feats["tma_allocator_installed"] = {"ok": tma_ready, "error": tma_err}
    if tma_err:
        print(f"    ..    tma allocator: {tma_err}")

    # 64x64 fp32: the innermost block dimension is 256 B, comfortably over TMA's 16 B minimum.
    # Values are checked, not just the launch: a descriptor that loads the wrong tile still
    # "works" as far as an exception-based probe can tell.
    def _tma_host():
        from triton.tools.tensor_descriptor import TensorDescriptor
        src = torch.randn(64, 64, device="cuda", dtype=torch.float32)
        o = torch.zeros(64, 64, device="cuda", dtype=torch.float32)
        desc = TensorDescriptor.from_tensor(src, [64, 64])
        pk.k_tma_host[(1,)](desc, o, BM=64, BN=64)
        torch.cuda.synchronize()
        return {"values_correct": bool(torch.equal(o, src))}
    run("tma_host_descriptor", _tma_host)

    def _tma_device():
        src = torch.randn(64, 64, device="cuda", dtype=torch.float32)
        o = torch.zeros(64, 64, device="cuda", dtype=torch.float32)
        pk.k_tma_device[(1,)](src, o, 64, 64, BM=64, BN=64)
        torch.cuda.synchronize()
        return {"values_correct": bool(torch.equal(o, src))}
    run("tma_device_descriptor", _tma_device)

    # cluster / DSMEM launch attribute
    def _cluster():
        pk.k_baseline[(N // 256,)](x, y, N, BLOCK=256, num_ctas=2)
    run("thread_block_cluster_num_ctas", _cluster)

    feats["compile_probes"] = probes
    # Single place to ask "can this stack do TMA at all", so no consumer has to know that the
    # host-side and device-side descriptors are two separate APIs with two separate probes.
    def _ok(name: str) -> bool:
        p = probes.get(name) or {}
        return bool(p.get("ok")) and p.get("values_correct", True) is not False
    feats["tma_available"] = _ok("tma_host_descriptor") or _ok("tma_device_descriptor")
    feats["tma_forms"] = {"host_descriptor": _ok("tma_host_descriptor"),
                          "device_descriptor": _ok("tma_device_descriptor")}
    feats["warp_specialize_available"] = _ok("warp_specialize_tl_range")
    feats["clusters_available"] = _ok("thread_block_cluster_num_ctas")
    print(f"  tma usable: {feats['tma_available']}  {feats['tma_forms']}")

    # --- 4c. what the compiler reports it can do ---------------------------------------
    try:
        tgt = triton.runtime.driver.active.get_current_target()
        feats["triton_target"] = {"backend": tgt.backend, "arch": tgt.arch,
                                  "warp_size": getattr(tgt, "warp_size", None)}
        print(f"  triton target: {feats['triton_target']}")
    except Exception as e:  # noqa: BLE001
        _err("triton_target", e)
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(0)
        feats["triton_device_properties"] = props
        print(f"  triton device properties: {props}")
    except Exception as e:  # noqa: BLE001
        _err("triton_device_properties", e)

    R["triton_features"] = feats
    try:
        src.unlink()
        for p in (HERE / "__pycache__").glob("_probe_kernels*"):
            p.unlink()
    except Exception:
        pass


# ======================================================================================
# 5. calibration -- every roofline ceiling comes from these, not from spec sheets
# ======================================================================================
CALIB_SRC = r'''
import triton, triton.language as tl

@triton.jit
def read_only(P, O, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0); i = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(P + i, mask=i < N, other=0.0)
    tl.store(O + pid, tl.sum(v.to(tl.float32)))

@triton.jit
def mm(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0); nm = tl.cdiv(M, BM); nn = tl.cdiv(N, BN)
    ng = GM * nn; g = pid // ng; fm = g * GM; gs = min(nm - fm, GM)
    pm = fm + ((pid % ng) % gs); pn = (pid % ng) // gs
    rm = (pm * BM + tl.arange(0, BM)) % M
    rn = (pn * BN + tl.arange(0, BN)) % N
    rk = tl.arange(0, BK)
    a = A + (rm[:, None] * sam + rk[None, :] * sak)
    b = B + (rk[:, None] * sbk + rn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        am = tl.load(a, mask=rk[None, :] < K - k * BK, other=0.0)
        bm = tl.load(b, mask=rk[:, None] < K - k * BK, other=0.0)
        acc = tl.dot(am, bm, acc)
        a += BK * sak; b += BK * sbk
    c = C + (rm[:, None] * scm + rn[None, :] * scn)
    tl.store(c, acc.to(C.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))

@triton.jit
def nop(P):
    pass
'''


def probe_calibration(quick: bool) -> None:
    section("5. CALIBRATION (measured on this machine)")
    try:
        import torch
        import triton
    except Exception as e:  # noqa: BLE001
        _err("calibration_import", e)
        return
    src = HERE / "_calib_kernels.py"
    try:
        src.write_text(CALIB_SRC)
        sys.path.insert(0, str(HERE))
        import importlib
        ck = importlib.import_module("_calib_kernels")
    except Exception as e:  # noqa: BLE001
        _err("calib_import", e)
        return

    cal: dict = {}

    def t_s(fn, w=10, n=40):
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    # --- bandwidth. buffer must far exceed H200's ~50 MB L2 --------------------------
    print("  -- bandwidth (buffers >> L2) --")
    bw = {}
    for mb in ([1024] if quick else [1024, 2048]):
        try:
            n = mb * 2**20 // 2
            a = torch.randn(n, device="cuda", dtype=torch.bfloat16)
            b = torch.empty_like(a)
            bw[f"copy_{mb}MB_GBs"] = 2 * n * 2 / t_s(lambda: b.copy_(a)) / 1e9
            bw[f"rmw_{mb}MB_GBs"] = 2 * n * 2 / t_s(lambda: a.mul_(1.0001)) / 1e9
            g = triton.cdiv(n, 8192)
            o = torch.empty(g, device="cuda", dtype=torch.float32)
            bw[f"read_{mb}MB_GBs"] = n * 2 / t_s(
                lambda: ck.read_only[(g,)](a, o, n, BLOCK=8192)) / 1e9
            for k in (f"copy_{mb}MB_GBs", f"rmw_{mb}MB_GBs", f"read_{mb}MB_GBs"):
                print(f"    {k:<24} {bw[k]:8.1f} GB/s")
            del a, b, o
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            _err(f"bandwidth_{mb}MB", e)
    cal["bandwidth"] = bw

    # --- bf16 GEMM: cuBLAS and Triton, at the study's o_proj shape -------------------
    print("  -- bf16 GEMM --")
    gemm = {}
    shapes = [(4096, 16384, 6144)] if quick else [(4096, 16384, 6144), (8192, 16384, 6144)]
    for (M, K, N) in shapes:
        try:
            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
            c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            fl = 2 * M * N * K
            gemm[f"cublas_{M}x{K}x{N}_TFs"] = fl / t_s(lambda: torch.mm(a, b), 5, 20) / 1e12
            best, bcfg = 0.0, None
            grid_cfgs = [(128, 128, 64, 8, 8, 3), (128, 256, 64, 8, 8, 3),
                         (64, 256, 64, 8, 4, 4), (128, 128, 64, 8, 4, 4),
                         (256, 128, 64, 8, 8, 3), (64, 128, 64, 8, 4, 4)]
            for (BM, BN, BK, GM, w, s) in grid_cfgs:
                try:
                    g = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
                    f = lambda BM=BM, BN=BN, BK=BK, GM=GM, w=w, s=s, g=g: ck.mm[g](
                        a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK, GM=GM,
                        num_warps=w, num_stages=s)
                    tf = fl / t_s(f, 3, 10) / 1e12
                    if tf > best:
                        best, bcfg = tf, dict(BM=BM, BN=BN, BK=BK, GM=GM, num_warps=w,
                                              num_stages=s)
                except Exception:
                    pass
            gemm[f"triton_{M}x{K}x{N}_TFs"] = best
            gemm[f"triton_{M}x{K}x{N}_cfg"] = bcfg
            print(f"    {M}x{K}x{N}: cuBLAS {gemm[f'cublas_{M}x{K}x{N}_TFs']:7.2f} TF/s | "
                  f"Triton {best:7.2f} TF/s ({best / max(gemm[f'cublas_{M}x{K}x{N}_TFs'], 1e-9) * 100:.0f}% of vendor)")
            del a, b, c
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            _err(f"gemm_{M}x{K}x{N}", e)
    cal["gemm"] = gemm

    # --- SMEM ceiling actually reachable from Triton ----------------------------------
    print("  -- reachable shared-memory ceiling --")
    smem = {}
    try:
        M, K, N = 512, 512, 512
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        for (BM, BN, BK, s) in [(128, 128, 64, 3), (128, 256, 64, 3), (128, 256, 64, 4),
                                (256, 256, 64, 3), (128, 256, 128, 3)]:
            try:
                g = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
                k = ck.mm[g](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0),
                             b.stride(1), c.stride(0), c.stride(1),
                             BM=BM, BN=BN, BK=BK, GM=8, num_warps=8, num_stages=s)
                torch.cuda.synchronize()
                got = getattr(getattr(k, "metadata", None), "shared", None)
                smem[f"BM{BM}_BN{BN}_BK{BK}_s{s}"] = {"ok": True, "shared_bytes": got}
                print(f"    OK    BM{BM} BN{BN} BK{BK} s{s} -> shared={got}")
            except Exception as e:  # noqa: BLE001
                smem[f"BM{BM}_BN{BN}_BK{BK}_s{s}"] = {"ok": False,
                                                      "error": f"{type(e).__name__}: {e}"[:200]}
                print(f"    FAIL  BM{BM} BN{BN} BK{BK} s{s}: {str(e)[:110]}")
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        _err("smem_probe", e)
    cal["smem_probe"] = smem

    # --- launch cost + timer granularity ----------------------------------------------
    # These two are the ONLY calibrations a neighbouring tenant can silently destroy, because
    # they are the only ones short enough to be dominated by someone else's time slice. The
    # previous H200 probe returned launch 8.89 us with a 40.55 us harness floor and a tick
    # matching 3 % of samples; both are impossible on an idle card. So the tenancy of the GPU
    # is captured on both sides of this measurement and a verdict is recorded next to it --
    # the failure mode to avoid is not "wrong numbers", it is wrong numbers that no downstream
    # consumer can tell are wrong.
    print("  -- launch cost and timer granularity --")
    sel = R.get("gpu_selection") or {}
    tenants_before, tenant_err = (foreign_processes(sel["index"])
                                  if sel.get("index") is not None else ([], "no GPU pinned"))
    try:
        flush = torch.empty(256 * 2**20 // 4, device="cuda", dtype=torch.int32)

        def once(nlaunch):
            flush.zero_()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(nlaunch):
                ck.nop[(1,)](flush)
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e)

        pts = {}
        for n in (1, 2, 4, 8):
            for _ in range(20):
                once(n)
            v = sorted(once(n) for _ in range(150))
            pts[n] = v[len(v) // 2]
        L = (pts[8] - pts[1]) / 7
        cal["launch_us"] = L * 1000
        cal["harness_floor_us"] = (pts[1] - L) * 1000
        vals = [once(1) for _ in range(200)]
        ticks = sorted({round(v * 1e6, 3) for v in vals})
        # The true granularity is the LARGEST quantum that divides essentially every
        # sample: 0.256 trivially divides anything 1.024 divides, so scanning upward and
        # keeping the first match would always report the finest candidate.
        cands = {}
        for q in (0.256, 0.512, 1.024, 2.048, 4.096):
            cands[q] = sum(1 for v in vals
                           if abs((v * 1000) / q - round((v * 1000) / q)) < 1e-3) / len(vals)
        ok = [q for q, f in cands.items() if f >= 0.98]
        cal["timer_tick_us"] = max(ok) if ok else min(cands, key=lambda q: -cands[q])
        cal["timer_tick_match_frac"] = cands[cal["timer_tick_us"]]
        cal["timer_tick_candidates"] = cands
        best_q, best_hits = cal["timer_tick_us"], int(cal["timer_tick_match_frac"] * len(vals))
        print(f"    launch cost      {L * 1000:7.2f} us")
        print(f"    harness floor    {cal['harness_floor_us']:7.2f} us")
        print(f"    timer tick       {best_q} us  ({best_hits}/{len(vals)} exact multiples)")
        del flush
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        _err("launch_timer", e)

    # --- verdict on the two contaminable numbers --------------------------------------
    tenants_after, _ = (foreign_processes(sel["index"])
                        if sel.get("index") is not None else ([], ""))
    doubts = []
    if tenants_before or tenants_after:
        doubts.append(f"{len(tenants_before)} foreign compute process(es) before and "
                      f"{len(tenants_after)} after this measurement; the GPU was shared")
    if tenant_err:
        doubts.append(f"tenancy could not be verified ({tenant_err})")
    frac = cal.get("timer_tick_match_frac")
    if isinstance(frac, (int, float)) and frac < 0.98:
        doubts.append(f"the winning tick quantum matches only {frac * 100:.0f}% of samples; "
                      f"a real CUDA-event tick matches ~100%, so this is not a tick, it is "
                      f"noise wide enough to hide one")
    floor = cal.get("harness_floor_us")
    launch = cal.get("launch_us")
    if isinstance(floor, (int, float)) and isinstance(launch, (int, float)) \
            and floor > 8 * max(launch, 1e-9):
        doubts.append(f"harness floor {floor:.2f} us is {floor / launch:.0f}x the per-launch "
                      f"cost {launch:.2f} us; on an idle device the two are the same order")
    cal["timing_environment"] = {
        "gpu_index": sel.get("index"), "gpu_uuid": sel.get("uuid"),
        "foreign_processes_before": tenants_before,
        "foreign_processes_after": tenants_after,
        "tenancy_query_error": tenant_err or None,
    }
    # None, not True, when the measurement never happened: "no numbers" and "numbers we
    # stand behind" must not look the same to a consumer that only reads this flag.
    measured = isinstance(cal.get("launch_us"), (int, float))
    cal["launch_timer_trustworthy"] = (not doubts) if measured else None
    cal["launch_timer_doubts"] = doubts
    if doubts and measured:
        print("\n    " + "!" * 70)
        print("    !! launch_us / harness_floor_us / timer_tick_us are NOT TRUSTWORTHY:")
        for d in doubts:
            print(f"    !!   - {d}")
        print("    !! Downstream this decides which cells are printed as UNRESOLVED, so a")
        print("    !! contaminated tick either hides real differences or invents them.")
        print("    !! Re-run on an idle card: python3 glm52_h200/preflight.py --gpu <idx>")
        print("    " + "!" * 70)
    elif measured:
        print("    launch/tick calibration taken on a GPU with no other compute processes.")
    else:
        print("    launch/tick calibration did not complete; nothing to trust or distrust.")

    R["calibration"] = cal
    try:
        src.unlink()
        for p in (HERE / "__pycache__").glob("_calib_kernels*"):
            p.unlink()
    except Exception:
        pass


# ======================================================================================
# 6. can the full GLM-5.2 layer be instantiated here?
# ======================================================================================
def probe_capacity() -> None:
    section("6. GLM-5.2 CAPACITY CHECK")
    H, MI, E, ER = 6144, 2048, 256, 256
    w13 = E * 2 * MI * H * 2
    w2 = E * H * MI * 2
    need = w13 + w2
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
    except Exception as e:  # noqa: BLE001
        _err("capacity", e)
        return
    GB = 2**30
    cap = {
        "w13_bytes": w13, "w2_bytes": w2, "expert_weights_bytes": need,
        "free_bytes": free, "total_bytes": total,
        "expert_weights_fit": need < free * 0.85,
        "whole_layer_feasible": need < free * 0.6,
    }
    R["capacity"] = cap
    print(f"  w13 (256 experts)      {w13 / GB:8.2f} GB")
    print(f"  w2  (256 experts)      {w2 / GB:8.2f} GB")
    print(f"  expert weights total   {need / GB:8.2f} GB")
    print(f"  free / total VRAM      {free / GB:8.2f} / {total / GB:.2f} GB")
    print(f"  expert weights fit     {cap['expert_weights_fit']}")
    print(f"  whole-layer feasible   {cap['whole_layer_feasible']}")
    if not cap["expert_weights_fit"]:
        print("  !! #6/#8/#9/#11a and the whole-layer benchmark cannot run on this device.")


# ======================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="H200 preflight probe -- run this first, send back the JSON.")
    ap.add_argument("--quick", action="store_true", help="skip the slower calibration sweeps")
    ap.add_argument("--gpu", default="auto", metavar="N|auto|none",
                    help="which nvidia-smi GPU to probe. 'auto' (default) enumerates all and "
                         "picks the idlest; N pins that physical index; 'none' leaves the "
                         "device to CUDA. Pinning also sets CUDA_DEVICE_ORDER=PCI_BUS_ID so "
                         "the index means what nvidia-smi says it means.")
    a = ap.parse_args()

    print(textwrap.dedent(f"""
        GLM-5.2 fusion study -- H200 preflight probe
        writing: {OUT_JSON}
        This script only reads; it installs nothing and changes no device state.
    """).strip())

    R["argv"] = sys.argv
    R["cwd"] = os.getcwd()
    R["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # FIRST, before anything can import torch: CUDA_VISIBLE_DEVICES is consumed once, at
    # context creation, and setting it after the first `import torch` that touches CUDA is a
    # silent no-op that would leave this probe describing a card it never measured.
    R["gpu_selection"] = select_gpu(str(a.gpu).strip().lower())

    probe_stack()
    probe_device()
    probe_smi()
    probe_triton_features(a.quick)
    probe_calibration(a.quick)
    probe_capacity()

    # Device fence on the OUTPUT, not just on reads.
    #
    # This file is the suite's only record of what the H200 actually supports, and its
    # default path is fixed -- so a probe run on any other machine silently overwrites it
    # with that machine's device, feature table and calibration. That happened once here:
    # a local sm_89 run replaced the real H200 probe, and every downstream consumer would
    # then have sized H200 grids from a laptop GPU while the filename still said "h200".
    #
    # Same lesson as the checkpoint fence, one level up: isolating reads is not enough if
    # writes are unguarded.
    existing = None
    try:
        existing = json.loads(OUT_JSON.read_text())
    except Exception:  # noqa: BLE001 -- absent or unreadable is the normal first run
        pass
    prev = ((existing or {}).get("device") or {}).get("name")
    now = (R.get("device") or {}).get("name")
    if prev and now and prev != now and "--force" not in sys.argv:
        alt = OUT_JSON.with_name(
            "preflight_" + "".join(
                ch if ch.isalnum() else "_" for ch in str(now).lower()
            ).strip("_") + ".json"
        )
        alt.write_text(json.dumps(R, indent=2, default=str))
        print(f"\n  !! {OUT_JSON.name} already describes {prev!r}, but this probe ran on "
              f"{now!r}.\n     REFUSING to overwrite it -- that file is what sizes the "
              f"benchmark grids.\n     This probe was written to {alt.name} instead.\n"
              f"     Pass --force if you really mean to replace it.")
        return
    OUT_JSON.write_text(json.dumps(R, indent=2, default=str))
    section("DONE")
    sel = R.get("gpu_selection") or {}
    if sel.get("index") is not None:
        print(f"  probed nvidia-smi GPU {sel['index']} ({sel.get('name')}) "
              f"uuid {sel.get('uuid')}")
        print(f"    selection: {sel.get('reason')}")
        if sel.get("busy"):
            print(f"  !! that GPU was NOT idle: {'; '.join(sel.get('busy_reasons') or [])}")
    cal = R.get("calibration") or {}
    if cal.get("launch_timer_trustworthy") is False:
        print("  !! launch_us / harness_floor_us / timer_tick_us are flagged UNTRUSTWORTHY "
              "in this file:")
        for d in cal.get("launch_timer_doubts") or []:
            print(f"       - {d}")
        print("     Re-run on an idle card to replace them; everything else stands.")
    if R["probe_errors"]:
        print(f"  {len(R['probe_errors'])} probe(s) reported errors (recorded, not fatal):")
        for k, v in R["probe_errors"].items():
            print(f"    {k}: {v[:160]}")
    else:
        print("  all probes completed without error")
    print(f"\n  -> send back: {OUT_JSON}")


if __name__ == "__main__":
    main()
