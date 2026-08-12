"""Whole-layer fusion measurement on H200 with the GPU's own uncertainty removed.

WHAT THIS FIXES.  `bench_layer.py` produced a defensible campaign, but four sources of
uncertainty survived it, and every open question in the study traces back to one of them:

  1. **The card/session seam.**  `decode_bs1` was measured on `GPU-b2318e71` on 2026-08-07
     and `decode_bs2/4/8/16` on `GPU-3aa19cef` on 2026-08-10, so the T=1->2 and T=16->32
     segments of every curve cross a physical device.  Here `--gpu` is MANDATORY, every
     regime runs in ONE process against ONE card, and the run aborts if another tenant is
     resident on it.
  2. **Unlocked clocks.**  Round-to-round spread ran 1.15 us at `decode_bs4` and 424 us at
     `prefill_t8192`; the decision threshold that follows is 2.8x-1440x the pass-to-pass
     band.  Here SM and memory clocks are pinned (when the host permits), the card is
     driven to thermal steady state BEFORE the first measurement, and every block is
     bracketed by a clock/temperature read so a drifted block is discarded, not averaged in.
  3. **The tuning confound.**  Each regime independently re-tuned ~23 kernel configs
     INCLUDING the all-unfused baseline, so "the fusion helped" and "the baseline happened
     to tune better" are not separable -- which is exactly the ambiguity behind the T=2
     dip.  Here every config is FROZEN from `results/h200/layer_configurations.json`.
     Nothing is tuned, so nothing can be tuned unequally.  It is also what makes the run
     fit in minutes instead of hours: tuning was ~99 % of the old cost.
  4. **One number for two questions.**  A wall-clock time at `decode_bs1` is mostly launch
     overhead; the same measurement at `prefill_t8192` is mostly work.  Here every
     configuration is timed THREE ways -- wall clock, CUDA-graph replay, and the CUPTI sum
     of its kernels' device time -- which decomposes the layer exactly:

         wall  =  work  +  in-graph gaps  +  launch cost
                  \____________________/
                        graph replay

     That answers, per regime and per configuration, how much of a fusion's isolated
     kernel-level saving survives into the assembled layer -- the 33 % at T=2 against 52 %
     at T=4 that the old data could show but not explain.

STATISTIC.  Blocks, not passes.  One block times every configuration once, in an order that
reverses on odd blocks, with the all-unfused baseline run at BOTH ends.  A configuration's
speedup for that block is the baseline sandwich mean divided by its own time, so linear
drift inside a block divides out and every ratio is built from measurements milliseconds
apart.  The headline is the median over blocks with a percentile-bootstrap 95 % CI, and
blocks keep being added until the CI is tight enough (`--target-ci`) or `--max-blocks` is
reached.  Two configurations are TIED when their CIs overlap; a configuration beats the
unfused layer only when its CI excludes 1.0.  That is a stated confidence statement rather
than the round-spread heuristic of LOG-11 §3.

COST.  Frozen configs make measurement the entire cost.  One block over all 18
configurations and all 11 regimes is ~0.9 s of GPU time, so even 200 blocks is minutes.
`--estimate` prints the plan and predicted wall time without touching the GPU.

FIRST RUN ON A NEW HOST -- do this before spending the budget:

    python3 glm52_h200/bench/bench_layer_certain.py --gpu 7 --smoke

2 regimes x 4 configurations x 6 blocks, about a minute, and it exercises every code path
(clock lock, graph capture, CUPTI profile, bootstrap, output write). Then the real run:

    python3 glm52_h200/bench/bench_layer_certain.py --gpu 7

Output: `results/h200/layer_certain.json`, plus a printed table per regime.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Importing `glm52_h200.bench` masks this process to `--gpu` BEFORE torch exists.
# Nothing above this line may import torch, or the mask arrives too late to matter.
from glm52_h200 import bench as B  # noqa: E402
from glm52_h200 import config as C  # noqa: E402
from glm52_h200 import hwinfo  # noqa: E402

import torch  # noqa: E402

from glm52_h200.bench.bench_layer import (  # noqa: E402
    CONFIGS,
    PRENORM_CONFIGS,
    LayerProblem,
    build_chain,
    tune_shared,
)
from glm52_h200.common import RESULTS_DIR, check, main_guard, record  # noqa: E402

RESULT_ID = "layer_certain"
FROZEN_SRC = "layer_configurations"
UNFUSED = "A_all_unfused"

E = C.N_ROUTED_EXPERTS
I = C.MOE_INTERMEDIATE_SIZE
H = C.HIDDEN_SIZE

#: Throttle reasons that invalidate a measurement.  `ApplicationsClocksSetting` is
#: DELIBERATELY absent: it is the bit a clock lock itself sets, so treating it as a fault
#: would discard 100 % of blocks on a correctly locked card while looking like a working
#: gate.  `GpuIdle` and `DisplayClockSetting` are likewise not slowdowns.
HARMFUL_THROTTLE = {"SwPowerCap", "HwSlowdown", "SwThermalSlowdown",
                    "HwThermalSlowdown", "HwPowerBrakeSlowdown"}


# ======================================================================================
# 1. The host: lock what can be locked, prove what cannot
# ======================================================================================
def smi(args: list[str], timeout: int = 30) -> tuple[int, str, str]:
    try:
        r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except Exception as exc:  # noqa: BLE001 -- a missing or wedged smi must not end the run
        return -1, "", f"{type(exc).__name__}: {exc}"


def resolve_target(sel: dict) -> dict:
    """Identify the ONE card this process is masked to, in both nvidia-smi's terms and ours.

    Two identifiers, because they are not interchangeable.  nvidia-smi indexes the host's
    cards (`--gpu 7` is card 7) while this process, once masked, sees a single device at
    ordinal 0; and nvidia-smi spells a UUID `GPU-b2318e71-...` where torch spells it
    `b2318e71-...`.  Comparing the two raw is how a filter silently matches nothing -- or,
    worse, is skipped entirely.
    """
    idx = sel.get("index")
    uuid = sel.get("uuid")
    if not uuid and idx is not None:
        # Second source: ask nvidia-smi directly rather than initialising CUDA to find out,
        # which would put THIS process on the card before the tenant check has run.
        rc, out, _ = smi(["-i", str(idx), "--query-gpu=uuid", "--format=csv,noheader"])
        if rc == 0 and out.strip():
            uuid = out.strip().splitlines()[0].strip()
    return {"uuid": uuid or None, "index": idx,
            "smi_selector": (uuid or (str(idx) if idx is not None else None))}


def tenant_check(target: dict, my_pid: int) -> dict:
    """Who else is on OUR card -- reported as evidence, and only gating when it is certain.

    The campaign's preflight recorded a 36.9 us harness floor against a 10.3 us launch on a
    card it did not request; that ratio is the signature of a contended device.  But
    `--query-compute-apps` lists every GPU on the host, so this must filter to our card and
    must FAIL OPEN when it cannot: a check that treats "I could not identify the card" as
    "everything matches" turns other people's jobs on other cards into a refusal to run.
    `select_gpu` has already screened this card at import time; this is the record of what
    was on it, not a second opinion that can veto the first.
    """
    rows, err = hwinfo.compute_apps()
    mine = hwinfo._norm_uuid(target.get("uuid"))
    everyone = [{"pid": r["pid"], "name": r.get("name"),
                 "used_mib": round(r.get("used_bytes", 0) / (1024 * 1024)),
                 "uuid": r.get("gpu_uuid"),
                 "on_our_card": bool(mine) and hwinfo._norm_uuid(r.get("gpu_uuid")) == mine}
                for r in rows]
    if err:
        return {"checked": False, "reason": err, "gating": False,
                "host_wide_procs": len(everyone)}
    if not mine:
        return {"checked": False, "gating": False, "host_wide_procs": len(everyone),
                "reason": "could not resolve this card's UUID, so the host-wide process "
                          "list cannot be filtered to it. NOT treating that as contention.",
                "all_procs": everyone}
    ours = [a for a in everyone if a["on_our_card"] and a["pid"] != my_pid]
    return {"checked": True, "gating": True, "uuid": target.get("uuid"),
            "foreign_procs": ours, "clean": not ours,
            "host_wide_procs": len(everyone),
            "self_pid_excluded": my_pid}


def supported_clocks(selector: str) -> dict:
    rc, out, _ = smi(["-i", selector,
                      "--query-gpu=clocks.max.sm,clocks.max.memory,clocks.sm,clocks.mem",
                      "--format=csv,noheader,nounits"])
    if rc != 0 or not out:
        return {}
    v = [x.strip() for x in out.splitlines()[0].split(",")]
    try:
        return {"max_sm": int(v[0]), "max_mem": int(v[1]),
                "cur_sm": int(v[2]), "cur_mem": int(v[3])}
    except (ValueError, IndexError):
        return {}


def lock_clocks(selector: str, sm_mhz: int | None, mem_mhz: int | None,
                headroom: float) -> dict:
    """Pin SM and memory clocks, and register the restore.

    Pinned at `headroom` x max rather than AT max, on purpose.  A card held at its boost
    ceiling cannot stay there: it heats, hits a power or thermal cap partway through, and
    reintroduces exactly the drift the lock was meant to remove -- and a throttled lock is
    invisible unless you read `clocks_throttle_reasons`.  A slightly lower pin is one the
    card can hold for the whole run, which is what reproducibility actually needs.
    """
    if not selector:
        return {"locked": False, "reason": "no nvidia-smi selector for this card"}
    caps = supported_clocks(selector)
    if not caps:
        return {"locked": False, "reason": "could not read supported clocks"}
    sm = sm_mhz or max(200, int(caps["max_sm"] * headroom))
    mem = mem_mhz or caps["max_mem"]

    rc_sm, _, err_sm = smi(["-i", selector, "-lgc", f"{sm},{sm}"])
    rc_mem, _, err_mem = smi(["-i", selector, "-lmc", f"{mem},{mem}"])

    if rc_sm == 0:
        def _restore() -> None:
            smi(["-i", selector, "-rgc"])
            smi(["-i", selector, "-rmc"])
            print(f"[clocks] restored default clocks on {selector}", flush=True)
        atexit.register(_restore)

    return {
        "locked": rc_sm == 0,
        "sm_requested_mhz": sm, "mem_requested_mhz": mem,
        "sm_max_mhz": caps["max_sm"], "mem_max_mhz": caps["max_mem"],
        "headroom": headroom,
        "mem_locked": rc_mem == 0,
        "sm_error": err_sm or None, "mem_error": err_mem or None,
        "note": None if rc_sm == 0 else
                "clock lock needs root (or persistence mode plus the right permissions). "
                "The run continues UNLOCKED: the per-block drift gate is then the only "
                "defence and it is a weaker one -- treat sub-0.2 % differences as "
                "unresolved rather than as rankings.",
    }


def clocks_now() -> dict:
    s = hwinfo.snapshot()
    return {"sm_mhz": s.get("sm_mhz"), "mem_mhz": s.get("mem_mhz"),
            "temp_c": s.get("temp_c"), "power_w": s.get("power_w"),
            "throttle": s.get("throttle") or []}


def thermal_precondition(max_seconds: float, plateau_mhz: float,
                         sample_every: float = 2.0) -> dict:
    """Drive the card to steady state and MEASURE the plateau instead of sleeping blindly.

    The campaign's cold start showed up as round 0 running 2.5x the median of the other
    seven at the four smallest regimes.  A fixed `sleep` cannot know whether that has
    settled on this card, today.  This runs a dense GEMM and stops when three consecutive
    clock samples sit inside `plateau_mhz` of each other.
    """
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    o = torch.empty(4096, 4096, device="cuda", dtype=torch.bfloat16)
    trace, t0, last, plateaued = [], time.time(), 0.0, False
    while time.time() - t0 < max_seconds:
        for _ in range(200):
            torch.mm(a, b, out=o)
        torch.cuda.synchronize()
        now = time.time()
        if now - last >= sample_every:
            last = now
            row = clocks_now()
            row["t"] = round(now - t0, 1)
            trace.append(row)
            sm = [x["sm_mhz"] for x in trace[-3:] if x["sm_mhz"] is not None]
            if len(sm) == 3 and (max(sm) - min(sm)) <= plateau_mhz:
                plateaued = True
                break
    del a, b, o
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return {"seconds": round(time.time() - t0, 1), "plateaued": plateaued,
            "plateau_mhz_bar": plateau_mhz, "trace": trace}


# ======================================================================================
# 2. Frozen configuration: measure what was chosen, never re-choose it
# ======================================================================================
def load_frozen(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"missing {path}. This script deliberately does NOT tune -- it measures the "
            f"configuration the campaign already chose. Run bench_layer.py first, or point "
            f"--cfg-json at another campaign file.")
    doc = json.loads(path.read_text())
    regs = doc.get("regimes") or doc.get("payload", {}).get("regimes")
    if not regs:
        raise SystemExit(f"{path} has no `regimes` block -- wrong schema?")
    return regs


def shared_cfg_for(p: LayerProblem, env, cache: bool) -> dict:
    """Shared-expert configs are NOT in the campaign JSON (only their timing is), so they
    are the one thing this script must still tune -- once, then checkpointed.

    They cannot change the RANKING (the shared expert is identical in every configuration)
    but they do set a common term in every ratio, which is why the T=2 block costing
    195.7 us against 83.7 us at its neighbours mattered at all.  Recording the chosen
    config makes that term auditable instead of invisible.
    """
    if cache:
        got = B.ckpt_load("layer_shared_cfgs", p.regime, env)
        if got and all(k in got for k in ("w13", "act", "w2")):
            print(f"  [shared] reusing checkpointed config for {p.regime}", flush=True)
            return got
    got = tune_shared(p)
    if cache:
        B.ckpt_save("layer_shared_cfgs", p.regime, env, got)
    return got


# ======================================================================================
# 3. Timing primitives
# ======================================================================================
def run_chain(fns) -> None:
    for f in fns:
        f()


def capture_graph(fns, warmup: int = 5):
    """Capture a chain into a CUDA graph, or return (None, reason).

    Graph replay is what separates real work from launch cost: identical kernels, no
    per-launch driver work.  Capture can legitimately fail (an op that syncs, a dynamic
    allocation), so a failure is recorded per configuration and that configuration keeps
    its wall-clock numbers rather than dropping out of the run.
    """
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                run_chain(fns)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run_chain(fns)
        for _ in range(2):
            g.replay()
        torch.cuda.synchronize()
        return g, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"[:200]


def time_sequence(seq, runner, flush: bool) -> list[tuple[str, float]]:
    """Time every entry of `seq` once, in order, with ONE sync for the whole sequence.

    Returns a LIST, not a dict: the baseline appears twice per block and the two readings
    are the sandwich.  Collapsing them into a dict key silently discards the first one and
    with it the drift correction, which is the whole point of the block.
    """
    evs = []
    for name in seq:
        if flush:
            B._flush_l2()
        a = torch.cuda.Event(enable_timing=True)
        z = torch.cuda.Event(enable_timing=True)
        a.record()
        runner(name)
        z.record()
        evs.append((name, a, z))
    torch.cuda.synchronize()
    return [(n, a.elapsed_time(z)) for n, a, z in evs]


def boot_ci(xs: list[float], n: int = 2000, alpha: float = 0.05,
            seed: int = 0) -> tuple[float, float, float]:
    """Median with a percentile-bootstrap CI. Returns (median, lo, hi)."""
    if not xs:
        return float("nan"), float("nan"), float("nan")
    med = statistics.median(xs)
    if len(xs) < 4:
        return med, min(xs), max(xs)
    rng = random.Random(seed)
    k = len(xs)
    meds = sorted(statistics.median(rng.choices(xs, k=k)) for _ in range(n))
    return med, meds[int(alpha / 2 * n)], meds[min(n - 1, int((1 - alpha / 2) * n))]


def median_se(xs: list[float]) -> float:
    """Cheap 95 % half-width for the STOPPING rule only (the published CI is bootstrapped).

    A full bootstrap after every block would cost more CPU than the measurement costs GPU.
    1.253 * sigma / sqrt(n) is the asymptotic standard error of the median; 1.96 of those
    is the half-width being compared against `--target-ci`.
    """
    if len(xs) < 4:
        return float("inf")
    try:
        return 1.96 * 1.253 * statistics.stdev(xs) / (len(xs) ** 0.5)
    except statistics.StatisticsError:
        return float("inf")


# ======================================================================================
# 4. The blocked, sandwiched, adaptive measurement
# ======================================================================================
def measure(names, runner, *, flush: bool, min_blocks: int, max_blocks: int,
            target_ci: float, drift_mhz: float, tag: str) -> dict:
    """Blocks of one-run-per-configuration, baseline sandwiched at both ends.

    Per block b, configuration i gets ratio_i(b) = mean(t_A at block start, t_A at block
    end) / t_i(b).  Both operands come from the same block, so a linear drift across the
    block divides out; the sandwich is what makes that true for configurations timed late
    in the block as well as early.  Order reverses on odd blocks so no configuration
    permanently inherits another's cache state.
    """
    others = [n for n in names if n != UNFUSED]
    ratios: dict[str, list[float]] = {n: [] for n in others}
    raw: dict[str, list[float]] = {n: [] for n in names}
    kept = dropped = 0
    drift_log: list[dict] = []
    warned = False

    for b in range(1, max_blocks + 1):
        order = others if b % 2 else list(reversed(others))
        before = clocks_now()
        timed = time_sequence([UNFUSED, *order, UNFUSED], runner, flush)
        after = clocks_now()

        # Discard rather than average: on an unlocked card, quietly folding drifted blocks
        # into the median is how every CI in the file silently widens.
        c0, c1 = before["sm_mhz"], after["sm_mhz"]
        thr = sorted(set(before["throttle"]) | set(after["throttle"]))
        harmful = [t for t in thr if t in HARMFUL_THROTTLE]
        moved = c0 is not None and c1 is not None and abs(c1 - c0) > drift_mhz
        if moved or harmful:
            dropped += 1
            drift_log.append({"block": b, "sm_before": c0, "sm_after": c1,
                              "throttle": thr, "harmful": harmful})
            if not warned and dropped > max(10, max_blocks // 4):
                warned = True
                print(f"    !! {tag}: {dropped} blocks discarded for drift/throttle -- the "
                      f"card is not holding still; expect wide CIs", flush=True)
            continue

        kept += 1
        base = (timed[0][1] + timed[-1][1]) / 2.0        # the sandwich
        raw[UNFUSED].append(base)
        for name, ms in timed[1:-1]:
            raw[name].append(ms)
            ratios[name].append(base / ms)

        if b >= min_blocks and kept >= 4:
            if max((median_se(ratios[n]) for n in others), default=float("inf")) <= target_ci:
                break

    out = {"blocks_kept": kept, "blocks_dropped": dropped,
           "drift_log": drift_log[:40], "per_config": {}}
    for n in names:
        entry: dict = {}
        if raw[n]:
            entry.update({"ms_p50": statistics.median(raw[n]),
                          "ms_min": min(raw[n]), "ms_max": max(raw[n]),
                          "n": len(raw[n])})
        if n != UNFUSED and ratios[n]:
            m, lo, hi = boot_ci(ratios[n])
            entry.update({"speedup_p50": m, "ci_lo": lo, "ci_hi": hi,
                          "ci_halfwidth": (hi - lo) / 2,
                          "beats_unfused": lo > 1.0, "loses_to_unfused": hi < 1.0})
        out["per_config"][n] = entry

    # Tie sets, stated as CI overlap rather than as a spread heuristic.
    scored = {n: v for n, v in out["per_config"].items() if "ci_lo" in v}
    if scored:
        best = max(scored, key=lambda n: scored[n]["speedup_p50"])
        blo, bhi = scored[best]["ci_lo"], scored[best]["ci_hi"]
        tied = sorted(n for n, v in scored.items() if v["ci_hi"] >= blo and v["ci_lo"] <= bhi)
        out["verdict"] = {
            "best": best, "tied_with_best": tied,
            "separated": len(tied) == 1,
            "unfused_in_tie_set": bhi >= 1.0 >= blo,
            "rule": "CIs overlap => tied; a configuration beats the unfused layer only "
                    "when its 95 % CI excludes 1.0",
        }
    return out


# ======================================================================================
# 5. Per-kernel device time: the work / gap / launch decomposition
# ======================================================================================
def profile_kernels(runner, name: str, reps: int) -> dict:
    """CUPTI per-kernel device time for one configuration, INSIDE the assembled layer.

    This is the measurement the campaign never had.  Summed over the chain it is the
    layer's real WORK; graph replay minus it is in-graph scheduling gap; wall minus graph
    is launch cost.  A fusion whose isolated kernels save 70 us but whose layer moves 20 us
    shows up here as which of those three terms failed to shrink.
    """
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception as exc:  # noqa: BLE001
        return {"error": f"profiler unavailable: {exc}"}
    try:
        for _ in range(3):
            runner(name)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            for _ in range(reps):
                runner(name)
            torch.cuda.synchronize()
        per: dict[str, float] = {}
        total = 0.0
        for ev in prof.key_averages():
            us = getattr(ev, "self_device_time_total", None)
            if us is None:
                us = getattr(ev, "self_cuda_time_total", 0.0)
            if not us or us <= 0:
                continue
            key = str(ev.key)
            # The L2 flush buffer zero is harness scaffolding, not layer work.
            if "zero_" in key or "fill_" in key or "Memset" in key or "memset" in key:
                continue
            per[key] = per.get(key, 0.0) + us / reps
            total += us / reps
        return {"kernels": dict(sorted(per.items(), key=lambda kv: -kv[1])),
                "work_us": total, "n_kernels_seen": len(per), "reps": reps}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


# ======================================================================================
# 6. Driver
# ======================================================================================
def estimate_seconds(regimes, n_cfg: int, blocks: int, graph: bool, frozen: dict,
                     profile_reps: int = 0, warm_s: float = 0.0) -> float:
    """Predict wall time from the campaign's own layer timings, before touching the GPU.

    Every term the run actually pays is in here.  An estimate that omits a phase is worse
    than no estimate at all, because `--budget-min` refuses to start on the strength of it.
    """
    total = float(warm_s)
    for r in regimes:
        mop = ((frozen.get(r.name, {}).get("verdict") or {}).get("median_of_passes") or {})
        ms = mop.get(UNFUSED) or 1.0
        total += ms * (n_cfg + 1) * (2.0 if graph else 1.0) * blocks / 1000.0
        # CUPTI adds roughly its own weight again in host-side event handling.
        total += ms * n_cfg * profile_reps * 2.0 / 1000.0
        total += 8.0                      # build + reference check + capture, per regime
    return total


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Whole-layer fusion measurement with the GPU's uncertainty removed.")
    B.add_gpu_args(ap)
    ap.add_argument("--regimes", default="all")
    ap.add_argument("--only", default="", help="comma-separated configuration ids")
    ap.add_argument("--cfg-json", default="", help="override the frozen-config source")
    ap.add_argument("--min-blocks", type=int, default=20)
    ap.add_argument("--max-blocks", type=int, default=200)
    ap.add_argument("--target-ci", type=float, default=5e-4,
                    help="stop once every speedup CI half-width is below this "
                         "(0.0005 = 0.05%%)")
    ap.add_argument("--profile-reps", type=int, default=20)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the fp32 reference COMPARISON; the reference is still built "
                         "by LayerProblem.__init__, so this saves seconds, not minutes")
    ap.add_argument("--no-prenorm", action="store_true")
    ap.add_argument("--no-lock-clocks", action="store_true")
    ap.add_argument("--lock-sm-mhz", type=int, default=0)
    ap.add_argument("--lock-mem-mhz", type=int, default=0)
    ap.add_argument("--clock-headroom", type=float, default=0.85)
    ap.add_argument("--drift-mhz", type=float, default=15.0)
    ap.add_argument("--warm-seconds", type=float, default=90.0)
    ap.add_argument("--plateau-mhz", type=float, default=15.0)
    ap.add_argument("--no-cfg-cache", action="store_true")
    ap.add_argument("--allow-tenants", action="store_true")
    ap.add_argument("--budget-min", type=float, default=30.0)
    ap.add_argument("--force", action="store_true", help="run even if over --budget-min")
    ap.add_argument("--estimate", action="store_true", help="print the plan and exit")
    ap.add_argument("--smoke", action="store_true",
                    help="2 regimes x 4 configs x 6 blocks -- validates every code path in "
                         "about a minute before you spend the budget")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.regimes = "decode_bs2,decode_bs4"
        args.only = args.only or "A_all_unfused,B_f3,H_f3_f10,J_greedy_all"
        args.min_blocks, args.max_blocks = 4, 6
        args.warm_seconds = min(args.warm_seconds, 15.0)
        args.profile_reps = 5

    src = Path(args.cfg_json) if args.cfg_json else RESULTS_DIR / f"{FROZEN_SRC}.json"
    frozen = load_frozen(src)

    regimes = B.resolve_regimes(C, args.regimes)
    want = [k.strip() for k in args.only.split(",") if k.strip()] or list(CONFIGS)
    unknown = [k for k in want if k not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configuration id(s): {unknown}")
    if UNFUSED not in want:
        want = [UNFUSED, *want]
    absent = [r.name for r in regimes if r.name not in frozen]
    if absent:
        raise SystemExit(f"no frozen configs for {absent} -- this script never tunes. Run "
                         f"bench_layer.py for those regimes, or drop them from --regimes.")

    est = estimate_seconds(regimes, len(want), args.max_blocks, not args.no_graph, frozen,
                           profile_reps=0 if args.no_profile else args.profile_reps,
                           warm_s=args.warm_seconds)
    floor = estimate_seconds(regimes, len(want), args.min_blocks, not args.no_graph, frozen,
                             profile_reps=0 if args.no_profile else args.profile_reps,
                             warm_s=args.warm_seconds)
    print(f"[plan] {len(regimes)} regimes x {len(want)} configurations, "
          f"{args.min_blocks}-{args.max_blocks} blocks, target CI "
          f"+-{args.target_ci * 100:.3f}%  (frozen configs from {src.name})", flush=True)
    print(f"[plan] {floor / 60:.1f}-{est / 60:.1f} min of measurement, plus a one-off "
          f"~2-3 min of Triton compiles for the shared expert on the FIRST run "
          f"(checkpointed thereafter)", flush=True)
    if args.estimate:
        return
    if est / 60 > args.budget_min and not args.force:
        raise SystemExit(f"[plan] estimate {est / 60:.1f} min exceeds --budget-min "
                         f"{args.budget_min}. Lower --max-blocks, drop regimes, or --force.")

    # ---- host lockdown ---------------------------------------------------------------
    sel = B.gpu_selection()
    if not sel.get("applied"):
        raise SystemExit(
            "[gpu] --gpu is MANDATORY here. The card/session seam is the clearest defect in "
            "the existing data and it came from letting the process take whatever device it "
            "inherited. Pass --gpu <index>.")
    target = resolve_target(sel)
    uuid = target["uuid"]
    tenants = tenant_check(target, os.getpid())
    print(f"[gpu] index={target['index']} uuid={uuid}", flush=True)
    if tenants.get("checked"):
        print(f"[gpu] {len(tenants['foreign_procs'])} foreign process(es) on this card "
              f"({tenants['host_wide_procs']} elsewhere on the host, not counted)",
              flush=True)
        for a in tenants["foreign_procs"]:
            print(f"[gpu]   pid {a['pid']} {a['name']} {a['used_mib']} MiB", flush=True)
    else:
        print(f"[gpu] tenant check did not run ({tenants.get('reason')}) -- continuing; "
              f"`select_gpu` already screened this card at import", flush=True)

    # `_bootstrap_gpu_selection` already refused a busy card at import time unless
    # --allow-busy was given, so this gate fires only on a process it can POSITIVELY
    # attribute to our UUID. It never fires on "could not tell".
    if tenants.get("gating") and not tenants.get("clean") and not args.allow_tenants:
        pids = [a["pid"] for a in tenants["foreign_procs"]]
        raise SystemExit(
            f"[gpu] {len(pids)} process(es) hold card {uuid}: {pids}. "
            f"Pick an idle card, or pass --allow-tenants and label the result contended.")

    clocks = ({"locked": False, "reason": "--no-lock-clocks"} if args.no_lock_clocks
              else lock_clocks(target["smi_selector"], args.lock_sm_mhz or None,
                               args.lock_mem_mhz or None, args.clock_headroom))
    print(f"[clocks] locked={clocks.get('locked')} sm={clocks.get('sm_requested_mhz')} "
          f"mem={clocks.get('mem_requested_mhz')}", flush=True)
    if clocks.get("note"):
        print(f"[clocks] !! {clocks['note']}", flush=True)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    warm = thermal_precondition(args.warm_seconds, args.plateau_mhz)
    print(f"[warm] {warm['seconds']}s, plateaued={warm['plateaued']}", flush=True)

    env = C.env()
    base_need = (E * 2 * I * H * 2) + (E * H * I * 2)
    fold_need = E * 2 * I * H * 2
    cap = B.mem_guard(base_need + fold_need, "w13 + w2 + folded w13")
    with_prenorm = (not args.no_prenorm) and cap["fits"]
    if not with_prenorm:
        want = [k for k in want if k not in PRENORM_CONFIGS]
        print(f"[scope] pre-norm configurations dropped: fits={cap.get('fits')}", flush=True)

    out_all: dict = {}
    t_start = time.time()

    for regime in regimes:
        print(f"\n===== {regime.name} =====", flush=True)
        p = LayerProblem(regime, with_prenorm)
        cfg = frozen[regime.name]["cfgs"]
        shared = shared_cfg_for(p, env, cache=not args.no_cfg_cache)

        chains: dict = {}
        correctness: dict = {}
        for name in want:
            sel_cfg = CONFIGS[name]
            try:
                chain = build_chain(p, cfg, sel_cfg, shared)
                p.out.zero_(); p.routed.zero_(); p.y3.zero_()
                run_chain(chain)
                torch.cuda.synchronize()
                if args.no_verify:
                    correctness[name] = {"verified": False, "n_kernels": len(chain)}
                else:
                    ck = check(p.out.clone(), p.ref_out, tol=5e-2, label=name)
                    correctness[name] = {"rel_err": ck["rel_err"], "ok": ck["ok"],
                                         "n_kernels": len(chain), "verified": True}
                    if not ck["ok"]:
                        print(f"  {name:<18} !! FAILS relerr={ck['rel_err']:.3e} "
                              f"-- excluded", flush=True)
                        continue
                chains[name] = chain
                print(f"  {name:<18} ok ({len(chain)} kernels)", flush=True)
            except Exception as exc:  # noqa: BLE001 -- one config must not lose the regime
                correctness[name] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                print(f"  {name:<18} FAILED {type(exc).__name__}: {exc}", flush=True)
            finally:
                torch.cuda.empty_cache()

        if UNFUSED not in chains:
            raise RuntimeError(f"{regime.name}: the baseline itself failed -- there is "
                               f"nothing to take a ratio against")
        names = list(chains)

        wall = measure(names, lambda n: run_chain(chains[n]), flush=True,
                       min_blocks=args.min_blocks, max_blocks=args.max_blocks,
                       target_ci=args.target_ci, drift_mhz=args.drift_mhz,
                       tag=f"{regime.name}/wall")
        print(f"  wall : {wall['blocks_kept']} kept / {wall['blocks_dropped']} dropped",
              flush=True)

        graphs: dict = {}
        graph_err: dict = {}
        graph = None
        if not args.no_graph:
            for n in names:
                g, err = capture_graph(chains[n])
                if g is None:
                    graph_err[n] = err
                    print(f"  [graph] {n:<18} capture failed: {err}", flush=True)
                else:
                    graphs[n] = g
            if UNFUSED in graphs and len(graphs) > 1:
                graph = measure(list(graphs), lambda n: graphs[n].replay(), flush=True,
                                min_blocks=args.min_blocks, max_blocks=args.max_blocks,
                                target_ci=args.target_ci, drift_mhz=args.drift_mhz,
                                tag=f"{regime.name}/graph")
                print(f"  graph: {graph['blocks_kept']} kept / "
                      f"{graph['blocks_dropped']} dropped", flush=True)

        prof: dict = {}
        if not args.no_profile:
            for n in names:
                if n in graphs:
                    prof[n] = profile_kernels(lambda nn: graphs[nn].replay(), n,
                                              args.profile_reps)
                else:
                    prof[n] = profile_kernels(lambda nn: run_chain(chains[nn]), n,
                                              args.profile_reps)

        decomp = {}
        for n in names:
            w = (wall["per_config"].get(n) or {}).get("ms_p50")
            g = ((graph or {}).get("per_config", {}).get(n) or {}).get("ms_p50")
            k = (prof.get(n) or {}).get("work_us")
            row: dict = {"wall_us": w * 1000 if w else None,
                         "graph_us": g * 1000 if g else None,
                         "work_us": k,
                         "n_kernels": (correctness.get(n) or {}).get("n_kernels")}
            if row["graph_us"] and row["work_us"]:
                row["gap_us"] = row["graph_us"] - row["work_us"]
            if row["wall_us"] and row["graph_us"]:
                row["launch_us"] = row["wall_us"] - row["graph_us"]
                if row["n_kernels"]:
                    row["launch_us_per_kernel"] = row["launch_us"] / row["n_kernels"]
            decomp[n] = row

        out_all[regime.name] = {
            "T": p.T, "oproj_K": p.Kq, "moe_rows": p.rows,
            "correctness": correctness,
            "cfgs_frozen_from": str(src),
            "shared_cfg": {k: v for k, v in shared.items() if k != "ms"},
            "shared_expert_tuning_ms": shared.get("ms"),
            "wall": wall, "graph": graph, "graph_capture_errors": graph_err,
            "per_kernel": prof, "decomposition": decomp,
            "clocks_after": clocks_now(),
        }

        rows = sorted(((v.get("speedup_p50") or 0.0, n)
                       for n, v in wall["per_config"].items() if n != UNFUSED),
                      reverse=True)
        print(f"  {'config':<18} {'wall x':>9} {'95% CI':>19} {'graph x':>9}  "
              f"{'launch us':>10}  verdict")
        for sp, n in rows:
            v = wall["per_config"][n]
            gv = (graph or {}).get("per_config", {}).get(n) or {}
            lu = decomp[n].get("launch_us")
            verdict = ("beats unfused" if v.get("beats_unfused")
                       else "SLOWER" if v.get("loses_to_unfused")
                       else "tied with unfused")
            print(f"  {n:<18} {sp:>9.4f} "
                  f"[{v.get('ci_lo', float('nan')):.4f},{v.get('ci_hi', float('nan')):.4f}]"
                  f" {gv.get('speedup_p50', float('nan')):>9.4f} "
                  f" {(lu if lu is not None else float('nan')):>10.1f}  {verdict}",
                  flush=True)
        if wall.get("verdict"):
            print(f"  --> {wall['verdict']['best']}"
                  f"{' (SEPARATED)' if wall['verdict']['separated'] else ''}"
                  f"  tied: {', '.join(wall['verdict']['tied_with_best'])}"
                  f"{'  [unfused is in the tie set]' if wall['verdict']['unfused_in_tie_set'] else ''}",
                  flush=True)

        del p, chains, graphs
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    record(RESULT_ID, {
        "id": RESULT_ID,
        "scope": "S3-S11 + shared expert. Configurations FROZEN from "
                 f"{src.name} -- nothing is tuned here, so no arm can be tuned unequally. "
                 "One process, one card, every regime.",
        "protocol": {
            "blocks": "one run per configuration per block, order reversed on odd blocks, "
                      "all-unfused baseline run at BOTH ends and averaged",
            "statistic": "median over blocks of (baseline sandwich / configuration), "
                         "percentile bootstrap 95 % CI, 2000 resamples",
            "tie_rule": "CIs overlap => tied; a configuration beats the unfused layer only "
                        "when its CI excludes 1.0",
            "drift_gate": f"a block is DISCARDED if SM clock moved > {args.drift_mhz} MHz "
                          f"across it or any of {sorted(HARMFUL_THROTTLE)} was active. "
                          "ApplicationsClocksSetting is excluded on purpose -- it is the "
                          "bit the clock lock itself sets.",
            "decomposition": "wall = work + in-graph gaps + launch cost, where work is the "
                             "CUPTI sum of per-kernel device time and graph replay is wall "
                             "minus launch cost",
        },
        "host": {"gpu_selection": sel, "gpu_uuid": uuid, "tenants": tenants,
                 "clocks": clocks, "thermal_precondition": warm,
                 "calibration": B.calibration_status()},
        "args": vars(args),
        "elapsed_s": round(time.time() - t_start, 1),
        "regimes": out_all,
    })
    print(f"\n[done] {time.time() - t_start:.0f}s -> "
          f"{RESULTS_DIR / (RESULT_ID + '.json')}", flush=True)


if __name__ == "__main__":
    main_guard(main)
