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

STATISTIC.  Blocks, not passes.  One block times every configuration once.  The running
order is a ROTATION of the configuration list by the block index, so over any n consecutive
blocks every configuration visits every slot exactly once -- INCLUDING the all-unfused
baseline, which is not privileged in any way.  The slot-0 configuration is run a second time
at the tail of the block; that repeat is the block's own drift probe.  A configuration's
speedup for that block is the (drift-detrended) baseline time divided by its own
(drift-detrended) time, so drift inside a block divides out and every ratio is built from
measurements milliseconds apart.  The headline is the median over blocks with a
percentile-bootstrap 95 % CI.  Two configurations are TIED when their CIs overlap; a
configuration beats the unfused layer only when its CI excludes 1.0.  That is a stated
confidence statement rather than the round-spread heuristic of LOG-11 §3.

WHAT THE 2026-08-13 RUN GOT WRONG, AND WHAT CHANGED HERE.  The first version of this file
sandwiched the baseline at BOTH ends of every block and rotated nothing: the sequence was
literally `[A_all_unfused, *others, A_all_unfused]`.  That pinned the baseline -- and only
the baseline -- to slot 0 of every single block, and slot 0 is not like the other slots.
`time_sequence` issues the whole block asynchronously and synchronises once at the end, so
from slot 1 onward the host runs far ahead of the device and the event pair measures device
time; at slot 0 the launch queue is EMPTY and the device is fed one kernel at a time at host
launch speed.  The run also spent two `nvidia-smi` subprocesses per block bracketing the
block, ~0.6-0.8 s of host dead time that left the launch path cold when slot 0 began.  The
measured cost, isolated against `F_f8` and `N_f11b` (which have the SAME 14 kernels as the
baseline and device work within 0.1 % of it), was a fixed 28-79 us charged to the baseline
in every block -- 2.6-5.6 us of launch cost per kernel against 0.7-1.4 us for everyone else.
Averaging the two sandwich readings halved it and did not remove it, so every
speedup-vs-unfused number in `results/h200/layer_certain.json` is inflated: +8.5 % at
`decode_bs4`, +3.6 % at `decode_bs8`, +0.9 % at `decode_bs256`.

It is worth being precise about the mechanism, because the obvious story is wrong and the
obvious story would have led to the wrong fix.  This is NOT a clock ramp after the idle gap.
The card read 1980 MHz -- its own maximum -- with an empty throttle list after every regime
up to `decode_bs256`, at 38-42 C and 137-353 W of a 700 W budget, and ZERO of 200 blocks were
discarded for clock movement in seven regimes x two modes.  Where drift was logged at all
(T >= 512) the clock FELL during the block rather than rising into it, and 100 % of those
discards were `SwPowerCap` on an idle read, not clock movement.  The penalty is also four to
five times larger in wall mode (14 exposed launches) than in graph mode (1 exposed launch),
which a device-side clock ramp cannot distinguish.  Locking the clocks would not have fixed
this.  What fixes it is: (1) a discarded warm-up run at the head of every block, issued
WITHOUT a synchronise so the queue is already full when slot 0's first event is recorded;
(2) rotating the whole running order so no configuration owns a slot; (3) getting
`nvidia-smi` out of the per-block path.

COST.  Frozen configs make measurement the entire cost.  The first run predicted 4-9 min and
took 60, because the 5e-4 CI target was unreachable in 20 of 22 regime x mode cells and the
run burned all 200 blocks in each of them anyway.  `estimate_seconds` now charges for the
warm-up runs, the `nvidia-smi` reads and the CUPTI pass, and quotes the CEILING as the number
to budget against; and `measure` stops when the CI is projected to be UNREACHABLE within
`--max-blocks`, or when it has stopped improving, not only when it hits target.
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
    """The full volatile snapshot.  Used once per regime and during preconditioning ONLY.

    `hwinfo.snapshot()` issues one `--query-gpu` of ten fields with no `-i` selector, so it
    enumerates every card on the host and then filters by UUID.  On the eight-card H200 node
    that measured 0.3-0.4 s per call, and the first version of this file made two of them
    around EVERY block: 8478 calls, ~92 % of a 3592 s run spent outside the measurement, and
    a cold host launch path handed to slot 0 of every block.  Never call this per block --
    call `clock_probe` instead.
    """
    s = hwinfo.snapshot()
    return {"sm_mhz": s.get("sm_mhz"), "mem_mhz": s.get("mem_mhz"),
            "temp_c": s.get("temp_c"), "power_w": s.get("power_w"),
            "throttle": s.get("throttle") or []}


def clock_probe(selector: str | None) -> dict:
    """One targeted read of ONE card and THREE fields -- the per-window drift sample.

    `-i <selector>` makes the driver report a single GPU instead of all eight, and three
    fields instead of ten.  That is the difference between ~0.35 s and a few tens of ms.  It
    is still a subprocess, which is exactly why it is now taken once per WINDOW of blocks
    rather than twice per block, and why the block that follows a probe still gets a
    discarded warm-up run to absorb whatever the probe left cold.
    """
    if not selector:
        return {**clocks_now(), "probe": "full-snapshot (no smi selector for this card)"}
    rc, out, err = smi(["-i", str(selector),
                        "--query-gpu=clocks.sm,temperature.gpu,"
                        "clocks_throttle_reasons.active",
                        "--format=csv,noheader,nounits"], timeout=15)
    if rc != 0 or not out:
        return {"sm_mhz": None, "temp_c": None, "throttle": [],
                "probe_error": (err or f"rc={rc}")[:120]}
    v = [x.strip() for x in out.splitlines()[0].split(",")]
    def _f(i):
        try:
            return float(v[i])
        except (ValueError, IndexError):
            return None
    return {"sm_mhz": _f(0), "temp_c": _f(1),
            "throttle": hwinfo.decode_throttle(v[2] if len(v) > 2 else None),
            "t": round(time.time(), 3)}


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


def time_sequence(seq, runner, flush: bool, warmup: int = 0) -> list[tuple[str, float]]:
    """Time every entry of `seq` once, in order, with ONE sync for the whole sequence.

    Returns a LIST, not a dict: `seq[0]` is run again at the tail as the block's drift probe
    and the two readings are not interchangeable.  Collapsing them into a dict key discards
    one of them and with it the drift correction, which is the whole point of the block.

    THE WARM-UP IS NOT SYNCHRONISED, AND THAT IS THE ENTIRE POINT.  `warmup` executions of
    `seq[0]` are issued and deliberately left in flight.  When the first `a.record()` is
    enqueued the device is still draining them, so (a) the device timestamps `a` only after
    the warm-up has drained -- CUDA events are device-side, so none of the warm-up's time is
    attributed to slot 0 -- and (b) the HOST is by then running ahead of the device, which is
    the state slots 1..N enjoy for free and slot 0 previously did not.  A
    `torch.cuda.synchronize()` here would empty the queue again and restore exactly the bias
    this is here to remove.  Two executions is enough: the host needs ~140 us to issue a
    14-kernel chain and the shortest layer here runs ~400 us.
    """
    for _ in range(max(0, warmup)):
        runner(seq[0])
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


# ---- the block design: pure functions, so they can be tested without a GPU --------------
def block_order(names: list, b: int) -> list:
    """Slot assignment for block `b`: rotate by `b`, reverse on alternate rotation CYCLES.

    ROTATION is what removes the position bias.  Over any `n` consecutive blocks every
    configuration occupies every slot exactly once, so the block-entry penalty at slot 0 --
    or any other slot-dependent effect -- is charged equally to every configuration instead
    of being pinned to the baseline, and it therefore cancels out of the ratio distribution
    rather than inflating its denominator.

    THE REVERSAL IS KEPT, but keyed on the rotation cycle (`b // n`) rather than on the block
    (`b % 2`), for two reasons.  (1) Rotation alone leaves the neighbour structure completely
    unbalanced: `names[j]` is preceded by `names[j-1]` in every single block regardless of
    the rotation, so any carry-over from one configuration into the next -- cache residency,
    a lingering tail effect -- is a permanent confound between that fixed pair.  Reversing
    gives each configuration both of its neighbours equally often.  (2) Keying the reversal
    on `b % 2` would SILENTLY DESTROY the slot balance whenever `n` is even, which it is here
    (14 configurations in ten of the eleven regimes).  With `n` even, `b % n` and `b % 2` are
    aliased -- even blocks always get an even rotation -- and the two effects compose so that
    each configuration is confined to slots of a single parity forever.  Keying on `b // n`
    decouples them: cycle 0 sweeps all `n` slots forward, cycle 1 sweeps all `n` slots
    reversed, and the balance is exact after every cycle rather than never.
    """
    n = len(names)
    if n == 0:
        return []
    rot = b % n
    seq = list(names[rot:]) + list(names[:rot])
    if (b // n) % 2:
        seq = list(reversed(seq))
    return seq


def block_sequence(order: list) -> list:
    """The runs actually issued for a block: the rotation, plus a repeat of slot 0.

    The repeat is the block's own drift probe.  The same configuration is measured at slot 0
    and at slot n, milliseconds apart under identical conditions, so their ratio is a DIRECT
    in-band measurement of how much the card slowed (or the harness stalled) across the
    block.  That costs one run -- exactly what the old baseline sandwich cost -- and unlike
    the two `nvidia-smi` snapshots it replaces, it measures conditions INSIDE the block
    instead of bracketing it with two reads taken while the device was idle.
    """
    return [*order, order[0]] if order else []


def block_readings(order: list, timed: list, detrend: bool = True) -> tuple:
    """Collapse one block's raw readings into one time per configuration.

    `timed` is `len(order) + 1` pairs in slot order; the tail repeats slot 0.  Returns
    `(times_by_config, drift_frac)` where `drift_frac = t_tail / t_slot0 - 1`.

    DETRENDING GENERALISES THE OLD SANDWICH AND STRICTLY IMPROVES ON IT.  Averaging the
    baseline's two readings estimated its value at the block MIDPOINT, which cancels a linear
    drift only for a configuration that also sits at the midpoint; a configuration at slot 1
    or slot n-1 kept most of the error.  Here the same two readings define a linear
    multiplicative trend `f(s) = 1 + g*s` with `g = drift_frac / n`, and EVERY slot is divided
    by its own `f(s)`.  By construction the two probe readings detrend to the same value, so
    the slot-0 configuration is not silently given a variance-reduced (averaged) reading that
    its peers do not get -- an asymmetry the old sandwich did have.

    When `detrend` is off the old behaviour is used for the probe configuration (mean of its
    two readings), which is the honest fallback rather than throwing one reading away.

    CALLERS MUST GATE ON `drift_frac` BEFORE TRUSTING A DETRENDED BLOCK.  The trend is fitted
    to two readings, so a single corrupt one -- a residual block-entry penalty at slot 0, a
    one-off host stall -- produces a large spurious `drift_frac` and the correction then
    spreads that error smoothly over all n slots, converting one outlier into a systematic
    bias that no longer looks like an outlier.  `measure` therefore calls this twice: once
    with `detrend=False` to obtain the raw probe ratio it gates on, and again only for the
    blocks that survived.
    """
    n = len(order)
    if n == 0 or len(timed) < n + 1:
        return {}, 0.0
    x = [ms for _, ms in timed]
    x0, xt = x[0], x[n]
    drift = (xt / x0 - 1.0) if x0 > 0 else 0.0
    if detrend and x0 > 0:
        g = drift / n
        vals = [x[s] / (1.0 + g * s) if (1.0 + g * s) > 0 else x[s] for s in range(n)]
    else:
        vals = list(x[:n])
        vals[0] = (x0 + xt) / 2.0
    return {order[s]: vals[s] for s in range(n)}, drift


def block_ratios(times_by_cfg: dict, baseline: str) -> dict:
    """speedup_i = t_baseline / t_i, both from the same block, both detrended."""
    base = times_by_cfg.get(baseline)
    if not base or base <= 0:
        return {}
    return {n: base / t for n, t in times_by_cfg.items() if n != baseline and t > 0}


def project_blocks_needed(halfwidth: float, n: int, target: float) -> float:
    """How many blocks would reach `target`, if the CI keeps shrinking as 1/sqrt(n)?

    This is the escape hatch the first run needed and did not have.  At `decode_bs4` it had a
    half-width of 4.6e-3 against a 5e-4 target after 20 blocks; the projection says that needs
    ~1700 blocks, so continuing to 200 was 180 blocks of measurement that could not possibly
    reach the goal.  Twenty of twenty-two regime x mode cells were in that position.
    """
    if not (halfwidth > 0) or target <= 0 or n <= 0:
        return float("inf")
    if halfwidth <= target:
        return float(n)
    return n * (halfwidth / target) ** 2


def stop_decision(halfwidth: float, n: int, target: float, max_blocks: int,
                  history: list, *, min_blocks: int, patience: int, plateau: float,
                  enabled: bool) -> tuple:
    """(stop, reason).  Three ways to stop, only one of which is "we succeeded"."""
    if n < max(4, min_blocks):
        return False, None
    if halfwidth <= target:
        return True, "target_ci_met"
    if not enabled:
        return False, None
    # `target_unreachable` may only fire in the SECOND half of the budget.  Under the 1/sqrt(n)
    # model the projection need = n*(hw_n/target)^2 = (C/target)^2 is independent of n, so an
    # ungated test is identically "will the target be missed at the ceiling?" -- it can only
    # ever fire at the FIRST check and would end every such cell at ~min_blocks. On the
    # 2026-08-13 run that was 18 of 22 regime x mode cells, i.e. a re-run publishing CIs ~2.7x
    # wider than the run it replaces while leaving most of a non-binding budget unspent.
    # Missing the target is not a reason to stop while the CI is still shrinking as 1/sqrt(n);
    # it is a reason not to spend the LAST half of the budget chasing it. Half the blocks costs
    # sqrt(2) in width, not 2.7x. A genuinely flat CI is the `ci_plateau` rule's job, below.
    need = project_blocks_needed(halfwidth, n, target)
    if need > max_blocks and n >= max_blocks // 2:
        return True, (f"target_unreachable: CI half-width {halfwidth:.2e} after {n} blocks "
                      f"projects to ~{min(need, 9.99e9):.0f} blocks for {target:.1e}, "
                      f"over --max-blocks {max_blocks}; stopped at the half-budget mark")
    if len(history) > patience:
        old = history[-1 - patience][1]
        if old > 0:
            gained = (old - halfwidth) / old
            if gained < plateau:
                return True, (f"ci_plateau: half-width improved {gained * 100:.2f}% over the "
                              f"last {patience} checks ({old:.2e} -> {halfwidth:.2e}), "
                              f"below --ci-plateau {plateau * 100:.1f}%")
    return False, None


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
            target_ci: float, drift_mhz: float, drift_frac: float, tag: str,
            selector: str | None = None, smi_every: int = 25, warmup: int = 2,
            detrend: bool = True, gate_throttle: bool = False, early_stop: bool = True,
            patience: int = 2, plateau: float = 0.05, store_blocks: bool = True) -> dict:
    """Rotated blocks of one-run-per-configuration, with an in-band drift probe.

    Per block b the running order is `block_order(names, b)` -- a rotation, reversed on
    alternate cycles -- followed by a repeat of whatever landed in slot 0.  Every
    configuration including the baseline visits every slot equally often, so no configuration
    inherits the block-entry penalty as a permanent tax.  Every ratio is still built from two
    measurements taken milliseconds apart inside one block; that property is what the block
    structure exists for and it is untouched.

    TWO GATES, AND THE CHEAP ONE IS THE GOOD ONE.

    In-band, every block, zero subprocesses: the slot-0 repeat gives `drift_frac`, the
    fractional change in one configuration's own time across the block.  Exceed
    `--drift-frac` and the block is discarded.  This is strictly better evidence than what it
    replaces.  The old gate read `nvidia-smi` immediately before and immediately after the
    block -- both times with the device idle and the measurement over -- and asked whether the
    clock had moved between two idle reads.  It could not see a stall inside the block at all,
    which is why it discarded 0 of 200 blocks in seven regimes while the baseline was being
    over-charged by up to 8.5 % in every one of them, and why at T >= 512 every one of its
    discards was `SwPowerCap` observed on an idle read rather than any measured slowdown.

    Out-of-band, once per WINDOW of blocks, one targeted subprocess: `clock_probe` samples the
    SM clock and throttle reasons.  If the clock has stepped by more than `--drift-mhz` since
    the previous sample, the entire window of blocks since that sample is discarded -- the
    step happened somewhere inside it and there is no way to say where, so all of it is
    suspect.  Windows are rounded to a whole number of rotation cycles so a discard never
    unbalances the slot design, and the sampling rate drops the subprocess count by ~50x.

    Harmful throttle reasons are RECORDED but do not discard by default (`--gate-throttle`
    turns the old behaviour back on).  `SwPowerCap` was asserted continuously on this card
    under load; gating on it threw away 70-85 % of blocks at the four largest regimes while
    selecting, not clean blocks, but the blocks that happened to have relaxed by the time of
    the idle read after them.  `drift_frac` measures whether the block was actually slow.
    """
    n = len(names)
    others = [x for x in names if x != UNFUSED]
    ratios: dict[str, list[float]] = {x: [] for x in others}
    raw: dict[str, list[float]] = {x: [] for x in names}
    slot_hist: dict[str, list[int]] = {x: [0] * n for x in names}
    kept = dropped = 0
    drift_log: list[dict] = []
    blocks_rec: list[dict] = []
    all_drift: list[float] = []
    warned = False

    # Align the sampling window to whole rotation cycles: a discarded window then removes
    # complete sweeps of the slot design and cannot bias the slot histogram.
    period = max(n, int(round(max(1, smi_every) / n)) * n)
    period = max(n, min(period, (max_blocks // n) * n or n))

    prev = clock_probe(selector)
    samples: list[dict] = [{**prev, "after_block": 0}]
    window: list[tuple] = []
    hist: list[tuple] = []
    stop_reason = f"max_blocks ({max_blocks}) reached"

    def _commit(win) -> None:
        nonlocal kept
        for rec, times, rat, order in win:
            kept += 1
            for nm, t in times.items():
                raw[nm].append(t)
            for nm, v in rat.items():
                ratios[nm].append(v)
            for s_i, nm in enumerate(order):
                slot_hist[nm][s_i] += 1
            if store_blocks:
                blocks_rec.append(rec)

    for b in range(max_blocks):
        order = block_order(names, b)
        timed = time_sequence(block_sequence(order), runner, flush, warmup=warmup)
        # GATE FIRST, ON UNDETRENDED READINGS, THEN DETREND.  The detrend is driven by the
        # very probe the gate is judging, so the two must not be entangled: a block whose
        # slot-0 reading is corrupt (a residual block-entry penalty, a one-off stall) yields
        # a large spurious `drift_frac`, and applying a trend fitted to that would spray the
        # error across all n slots -- turning one bad reading into n bad readings and, worse,
        # into a SMOOTH bias that no longer looks like an outlier. Offline this inflated the
        # recovered speedups by up to 7.6 % when the gate was disabled. The gate sees the raw
        # probe ratio, which is exactly the corruption signal, and only survivors are trended.
        _, dfrac = block_readings(order, timed, detrend=False)
        all_drift.append(dfrac)
        rec = {"b": b, "slots": [names.index(x) for x in order],
               "ms": [round(ms, 6) for _, ms in timed],
               "drift_frac": round(dfrac, 6), "kept": True}

        # In-band gate.  `ms` is stored for dropped blocks too: the threshold can then be
        # re-chosen offline against the observed distribution instead of by another 60 min.
        # A block failing this gate must NOT skip the window-commit logic below.  It used to
        # `continue` straight to the next block, which deferred the commit past its boundary
        # and -- when the FINAL block was the one that failed -- ended the loop with an
        # uncommitted window whose blocks were counted in neither `kept` nor `dropped` and
        # never written to `blocks_rec`. Unlike an honest discard those were unrecoverable
        # offline. Drop the block, then fall through to the boundary check regardless.
        if abs(dfrac) > drift_frac:
            dropped += 1
            rec["kept"] = False
            rec["drop"] = "in_block_drift"
            drift_log.append({"block": b, "reason": "in_block_drift",
                              "drift_frac": rec["drift_frac"],
                              "probe_cfg": order[0]})
            if store_blocks:
                blocks_rec.append(rec)
            if not warned and dropped > max(10, max_blocks // 4):
                warned = True
                print(f"    !! {tag}: {dropped} blocks discarded on the in-block drift probe "
                      f"(median drift {statistics.median(all_drift) * 100:+.2f}%) -- the card "
                      f"is not holding still inside a block", flush=True)
        else:
            times, _ = block_readings(order, timed, detrend=detrend)
            rat = block_ratios(times, UNFUSED)
            rec["ratio"] = [round(rat.get(x, 1.0), 6) for x in order]
            window.append((rec, times, rat, order))

        at_boundary = ((b + 1) % period == 0) or (b + 1 == max_blocks)
        if not at_boundary:
            continue

        cur = clock_probe(selector)
        thr = sorted(set(cur.get("throttle") or []))
        harmful = [t for t in thr if t in HARMFUL_THROTTLE]
        c0, c1 = prev.get("sm_mhz"), cur.get("sm_mhz")
        moved = c0 is not None and c1 is not None and abs(c1 - c0) > drift_mhz
        samples.append({**cur, "after_block": b, "window": len(window),
                        "sm_step_mhz": (None if (c0 is None or c1 is None) else c1 - c0),
                        "harmful": harmful,
                        "action": "drop_window" if (moved or (gate_throttle and harmful))
                                  else "commit"})
        if moved or (gate_throttle and harmful):
            dropped += len(window)
            for rec_i, _, _, _ in window:
                rec_i["kept"] = False
                rec_i["drop"] = "clock_window"
                if store_blocks:
                    blocks_rec.append(rec_i)
            drift_log.append({"block": b, "reason": "clock_window",
                              "sm_before": c0, "sm_after": c1,
                              "blocks_discarded": len(window),
                              "throttle": thr, "harmful": harmful})
        else:
            _commit(window)
        window = []
        prev = cur

        hw = max((median_se(ratios[x]) for x in others), default=float("inf"))
        stop, why = stop_decision(hw, kept, target_ci, max_blocks, hist,
                                  min_blocks=min_blocks, patience=patience,
                                  plateau=plateau, enabled=early_stop)
        hist.append((kept, hw))
        if stop and why == "target_ci_met":
            # Confirm with the PUBLISHED estimator before stopping.  `median_se` is an
            # asymptotic proxy and the bootstrap is what gets printed; in the first run they
            # disagreed across the threshold at decode_bs256/graph, which stopped at 39
            # blocks on the proxy while its bootstrap half-width was still above target.
            hw_boot = max(((lambda t: (t[2] - t[1]) / 2)(boot_ci(ratios[x])) for x in others),
                          default=float("inf"))
            if hw_boot > target_ci:
                stop, why = False, None
                hist[-1] = (kept, max(hw, hw_boot))
        if stop:
            stop_reason = why
            break

    # Invariant: every measured block is either committed or recorded as dropped. With the
    # fall-through above the final block always reaches the boundary check, so `window`
    # should be empty here -- flush it anyway and SAY SO, because the alternative failure
    # (blocks that were measured and then silently vanished) is the one defect in this
    # harness that cannot be detected from its own output file.
    lost = len(window)
    if window:
        _commit(window)
        window = []

    out = {"blocks_kept": kept, "blocks_dropped": dropped, "blocks_flushed_at_exit": lost,
           "drift_log": drift_log[:40], "per_config": {}}
    for x in names:
        entry: dict = {}
        if raw[x]:
            entry.update({"ms_p50": statistics.median(raw[x]),
                          "ms_min": min(raw[x]), "ms_max": max(raw[x]),
                          "n": len(raw[x])})
        if x != UNFUSED and ratios[x]:
            m, lo, hi = boot_ci(ratios[x])
            entry.update({"speedup_p50": m, "ci_lo": lo, "ci_hi": hi,
                          "ci_halfwidth": (hi - lo) / 2,
                          "beats_unfused": lo > 1.0, "loses_to_unfused": hi < 1.0})
        out["per_config"][x] = entry

    ach = max((v.get("ci_halfwidth", float("inf"))
               for k, v in out["per_config"].items() if k != UNFUSED), default=float("nan"))
    out["stop_reason"] = stop_reason
    out["ci_target"] = target_ci
    out["ci_achieved"] = ach
    out["ci_target_met"] = bool(ach == ach and ach <= target_ci)
    out["ci_history"] = [{"blocks": k, "halfwidth_proxy": (v if v == v and v != float("inf")
                                                           else None)} for k, v in hist]
    out["names"] = list(names)
    out["slot_histogram"] = slot_hist
    # max-minus-min visits per slot, over kept blocks. 0 == a perfectly balanced design.
    out["slot_imbalance"] = {x: (max(h) - min(h)) for x, h in slot_hist.items()}
    out["clock_samples"] = samples
    out["drift_probe"] = {
        "cfg": "whichever configuration the rotation put in slot 0 of that block",
        "n": len(all_drift),
        "median_frac": statistics.median(all_drift) if all_drift else None,
        "min_frac": min(all_drift) if all_drift else None,
        "max_frac": max(all_drift) if all_drift else None,
        "limit": drift_frac,
        "note": "t(slot n, repeat of slot 0) / t(slot 0) - 1. A systematically NEGATIVE "
                "median means the head of the block is still slower than its tail, i.e. the "
                "warm-up is not fully absorbing the block-entry penalty: raise "
                "--warmup-runs. Near zero means it is.",
    }
    out["design"] = {
        "order": "rotation by block index; reversed on alternate rotation cycles "
                 "(b // n), NOT on alternate blocks -- with n even, b % 2 aliases with the "
                 "rotation and would confine every configuration to slots of one parity",
        "baseline_position": "rotating, identical to every other configuration",
        "block_runs": n + 1,
        "warmup_runs_discarded": warmup,
        "warmup_synchronised": False,
        "detrended": detrend,
        "smi_period_blocks": period,
        "gate_throttle": gate_throttle,
    }
    if store_blocks:
        out["blocks"] = blocks_rec

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
_FLUSH_KEYS: set | None = None


def flush_kernel_keys(reps: int = 20) -> set:
    """Profile the L2 flush ALONE, once, to learn exactly which kernel names it produces.

    The flush now runs inside the profiled region (see `profile_kernels`), so its kernel has
    to come back out of `work_us`.  The old substring filter -- drop any key containing
    `zero_`, `fill_`, `memset` -- is not safe for that job in both directions: it misses the
    flush when the driver spells it `...FillFunctor<int>...` (no underscore), and it silently
    deletes REAL layer work when a configuration legitimately zero-fills an accumulator, which
    `F_f8` and `K_f3_f8` do -- their `FillFunctor` row is the atomics buffer initialisation and
    is part of the layer.  Measuring the flush's own key set removes both errors: exclusion
    becomes exact-match on names that were observed to come from the flush and nothing else.
    """
    global _FLUSH_KEYS
    if _FLUSH_KEYS is not None:
        return _FLUSH_KEYS
    keys: set = set()
    try:
        from torch.profiler import ProfilerActivity, profile
        for _ in range(3):
            B._flush_l2()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            for _ in range(reps):
                B._flush_l2()
            torch.cuda.synchronize()
        for ev in prof.key_averages():
            us = getattr(ev, "self_device_time_total", None)
            if us is None:
                us = getattr(ev, "self_cuda_time_total", 0.0)
            if us and us > 0:
                keys.add(str(ev.key))
    except Exception:  # noqa: BLE001 -- fall back to the substring filter below
        keys = set()
    _FLUSH_KEYS = keys
    return keys


def profile_kernels(runner, name: str, reps: int, *, flush: bool = True,
                    selector: str | None = None) -> dict:
    """CUPTI per-kernel device time for one configuration, INSIDE the assembled layer.

    This is the measurement the campaign never had.  Summed over the chain it is the layer's
    real WORK; graph replay minus it is in-graph scheduling gap; wall minus graph is launch
    cost.  A fusion whose isolated kernels save 70 us but whose layer moves 20 us shows up
    here as which of those three terms failed to shrink.

    MAKING IT COMPARABLE TO THE TIMING BLOCKS.  In the first run this pass was not comparable
    to them, and the decomposition inverted: `work_us > wall_us` in 40 of 158 rows, with
    in-graph "gaps" as negative as -905 us.  Two causes, both fixed here, and one of them
    fixed by DETECTION rather than by removal because it cannot be removed:

      1. Different cache state.  `time_sequence` flushes L2 before every timed run; this pass
         flushed nothing and ran `reps` back to back, so `work_us` was measured warm and the
         thing it was subtracted from was measured cold.  Now it flushes between reps exactly
         as the blocks do, with the flush's own kernels excluded by measured name.
      2. Different clock state, which INTERLEAVING CANNOT FIX.  Twenty back-to-back profiled
         reps are the densest sustained load in the harness, and CUPTI's own host overhead
         changes the duty cycle on top of that; the card simply does not run at the same clock
         here as it does inside a gated block, and putting this pass inside a block would
         contaminate the block instead of cleaning up the profile.  So it is not interleaved.
         Instead the SAME executions are ALSO timed with CUDA events, giving `pass_wall_us`
         from the identical reps that produced `work_us`.  Two facts follow that the first run
         could not establish: `work_us <= pass_wall_us` must hold by construction, so a
         violation indicts CUPTI rather than the run; and `pass_wall_us / wall_us` from the
         blocks is a direct per-configuration measurement of how far the two passes' operating
         conditions differ, which is exactly the quantity whose absence made the negative gaps
         uninterpretable.  `decomposition` carries it as `profile_vs_block`.
    """
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception as exc:  # noqa: BLE001
        return {"error": f"profiler unavailable: {exc}"}
    try:
        excl = flush_kernel_keys() if flush else set()
        for _ in range(3):
            runner(name)
        torch.cuda.synchronize()
        before = clock_probe(selector)
        evs = []
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            for _ in range(reps):
                if flush:
                    B._flush_l2()
                a = torch.cuda.Event(enable_timing=True)
                z = torch.cuda.Event(enable_timing=True)
                a.record()
                runner(name)
                z.record()
                evs.append((a, z))
            torch.cuda.synchronize()
        after = clock_probe(selector)
        per: dict[str, float] = {}
        total = 0.0
        excluded_us = 0.0
        for ev in prof.key_averages():
            us = getattr(ev, "self_device_time_total", None)
            if us is None:
                us = getattr(ev, "self_cuda_time_total", 0.0)
            if not us or us <= 0:
                continue
            key = str(ev.key)
            if key in excl or "Memset" in key or "memset" in key:
                excluded_us += us / reps
                continue
            per[key] = per.get(key, 0.0) + us / reps
            total += us / reps
        wall_us = [a.elapsed_time(z) * 1000.0 for a, z in evs]
        pass_wall = statistics.median(wall_us) if wall_us else None
        return {"kernels": dict(sorted(per.items(), key=lambda kv: -kv[1])),
                "work_us": total, "n_kernels_seen": len(per), "reps": reps,
                "flushed": bool(flush),
                "excluded_flush_keys": sorted(excl),
                "excluded_us": excluded_us,
                "pass_wall_us": pass_wall,
                "pass_wall_min_us": min(wall_us) if wall_us else None,
                "work_le_pass_wall": (None if pass_wall is None else total <= pass_wall),
                "clocks_before": before, "clocks_after": after}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


# ======================================================================================
# 6. Driver
# ======================================================================================
def estimate_seconds(regimes, n_cfg: int, blocks: int, graph: bool, frozen: dict,
                     profile_reps: int = 0, warm_s: float = 0.0, warmup_runs: int = 2,
                     smi_every: int = 25, smi_cost_s: float = 0.35) -> float:
    """Predict wall time from the campaign's own layer timings, before touching the GPU.

    Every term the run actually pays is in here.  An estimate that omits a phase is worse than
    no estimate at all, because `--budget-min` refuses to start on the strength of it -- and
    the first version of this function omitted the single largest term.  It predicted 4-9 min;
    the run took 60.  The gap was almost entirely `nvidia-smi`: two un-filtered `--query-gpu`
    calls over all eight cards around every block, ~0.35 s each, 8478 of them, ~92 % of a
    3592 s run spent outside the measurement it was estimating.  Charged explicitly now, at a
    rate `--smi-cost-s` the operator can correct from their own host.

    The other half of the miss was the STOPPING rule rather than the per-block cost: the
    estimate quoted a min_blocks..max_blocks range as though the floor were the likely
    outcome, when the 5e-4 target was unreachable in 20 of 22 regime x mode cells and every
    one of them ran to the ceiling.  `main` now quotes the ceiling as the number to budget
    against and says so.
    """
    total = float(warm_s)
    n_modes = 2.0 if graph else 1.0
    period = max(1, int(round(max(1, smi_every) / max(1, n_cfg))) * max(1, n_cfg))
    for r in regimes:
        mop = ((frozen.get(r.name, {}).get("verdict") or {}).get("median_of_passes") or {})
        ms = mop.get(UNFUSED) or 1.0
        # n_cfg timed runs + 1 drift-probe repeat + the discarded warm-up runs, per block.
        total += ms * (n_cfg + 1 + max(0, warmup_runs)) * n_modes * blocks / 1000.0
        # One targeted nvidia-smi per window of `period` blocks, per mode, plus the opening
        # sample.  This is the term whose absence produced the 4-9 min prediction.
        total += (blocks / period + 1.0) * n_modes * smi_cost_s
        # CUPTI adds roughly its own weight again in host-side event handling.
        total += ms * n_cfg * profile_reps * 2.0 / 1000.0
        total += 8.0                      # build + reference check + capture, per regime
    return total


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Whole-layer fusion measurement with the GPU's uncertainty removed.")
    B.add_gpu_args(ap)
    ap.add_argument("--regimes", default="",
                    help="comma-separated subset; empty or 'all' means every regime")
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
    ap.add_argument("--drift-mhz", type=float, default=15.0,
                    help="window gate: SM clock STEP between consecutive samples that "
                         "discards the whole window of blocks between them")
    ap.add_argument("--drift-frac", type=float, default=0.02,
                    help="in-block gate: |t(slot n)/t(slot 0) - 1| above this discards the "
                         "block. The slot-0 configuration is run again at the tail of every "
                         "block, so this measures conditions INSIDE the block")
    ap.add_argument("--smi-every", type=int, default=25,
                    help="take one targeted nvidia-smi sample per this many blocks (rounded "
                         "to a whole number of rotation cycles). The first version sampled "
                         "TWICE PER BLOCK and that cost ~92%% of the run")
    ap.add_argument("--smi-cost-s", type=float, default=0.35,
                    help="measured seconds per nvidia-smi call, for --estimate only")
    ap.add_argument("--warmup-runs", type=int, default=2,
                    help="executions discarded at the head of every block to absorb the "
                         "block-entry penalty. NOT synchronised, on purpose")
    ap.add_argument("--no-detrend", action="store_true",
                    help="use the old sandwich (mean of the two probe readings) instead of "
                         "detrending every slot by the measured within-block trend")
    ap.add_argument("--gate-throttle", action="store_true",
                    help="also discard a window when a harmful throttle reason is sampled. "
                         "OFF by default: SwPowerCap was continuously asserted on this card "
                         "under load and gating on it discarded 70-85%% of blocks at the "
                         "large regimes without selecting for anything measurable")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="disable the unreachable-target and CI-plateau escapes and run to "
                         "--max-blocks unless the target is actually met")
    ap.add_argument("--ci-patience", type=int, default=2,
                    help="how many consecutive CI checks may pass without improvement")
    ap.add_argument("--ci-plateau", type=float, default=0.05,
                    help="fractional CI improvement over --ci-patience checks below which "
                         "the measurement is judged to have stopped improving")
    ap.add_argument("--no-store-blocks", action="store_true",
                    help="omit the raw per-position timings and per-block ratios. Storing "
                         "them is the default: without them the position bias in the "
                         "2026-08-13 run could only be reconstructed, never measured")
    ap.add_argument("--no-profile-flush", action="store_true",
                    help="do not flush L2 between profiled reps (the old behaviour, which "
                         "made work_us incomparable to the flushed timing blocks)")
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


def _report_measure(label: str, m: dict) -> None:
    """One line per mode saying what was measured AND whether the design held.

    `stop_reason` and `ci_achieved` are printed because the first run's headline claim was a
    +-0.05 % CI it missed by 10-30x in every regime but two, and nothing on screen said so.
    `slot_imbalance` is printed because a rotation that has been unbalanced by dropped blocks
    is a rotation that is no longer cancelling position, and that must be visible while the
    run is happening rather than inferred from the file afterwards.
    """
    dp = m.get("drift_probe") or {}
    med = dp.get("median_frac")
    imb = max((m.get("slot_imbalance") or {}).values(), default=0)
    print(f"  {label}: {m['blocks_kept']} kept / {m['blocks_dropped']} dropped, "
          f"CI +-{m.get('ci_achieved', float('nan')):.2e} vs target "
          f"{m.get('ci_target', float('nan')):.1e} "
          f"({'MET' if m.get('ci_target_met') else 'NOT met'}), "
          f"stop: {m.get('stop_reason')}", flush=True)
    print(f"  {label}: slot imbalance {imb} (0 = every configuration visited every slot "
          f"equally), in-block drift probe median "
          f"{(med * 100 if med is not None else float('nan')):+.3f}%", flush=True)
    if med is not None and abs(med) > 0.003:
        print(f"  {label}: !! the drift probe is systematically "
              f"{'NEGATIVE' if med < 0 else 'POSITIVE'} -- slot 0 is still not like the "
              f"other slots. Raise --warmup-runs before trusting these ratios.", flush=True)


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

    # `resolve_regimes` spells "everything" as the empty string. Accept the spellings an
    # operator will actually type rather than failing on them after the GPU is already masked.
    if args.regimes.strip().lower() in ("all", "*"):
        args.regimes = ""
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

    _est = dict(profile_reps=0 if args.no_profile else args.profile_reps,
                warm_s=args.warm_seconds, warmup_runs=args.warmup_runs,
                smi_every=args.smi_every, smi_cost_s=args.smi_cost_s)
    est = estimate_seconds(regimes, len(want), args.max_blocks, not args.no_graph, frozen,
                           **_est)
    floor = estimate_seconds(regimes, len(want), args.min_blocks, not args.no_graph, frozen,
                             **_est)
    print(f"[plan] {len(regimes)} regimes x {len(want)} configurations, "
          f"{args.min_blocks}-{args.max_blocks} blocks, target CI "
          f"+-{args.target_ci * 100:.3f}%  (frozen configs from {src.name})", flush=True)
    print(f"[plan] BUDGET AGAINST {est / 60:.1f} min (every regime runs to --max-blocks). "
          f"{floor / 60:.1f} min is the floor, and it is only reached where the CI target is "
          f"actually MET -- on 2026-08-13 that happened in 2 of 22 regime x mode cells, so "
          f"the ceiling is the honest number.", flush=True)
    print(f"[plan] early stop: {'DISABLED' if args.no_early_stop else 'on'} -- a cell also "
          f"stops when {args.target_ci:.1e} is projected to need more than "
          f"--max-blocks {args.max_blocks} blocks, or when the CI improves by less than "
          f"{args.ci_plateau * 100:.0f}% over {args.ci_patience} checks. Expect most cells to "
          f"stop well short of the ceiling for that reason.", flush=True)
    print(f"[plan] plus a one-off ~2-3 min of Triton compiles for the shared expert on the "
          f"FIRST run (checkpointed thereafter)", flush=True)
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

        mkw = dict(min_blocks=args.min_blocks, max_blocks=args.max_blocks,
                   target_ci=args.target_ci, drift_mhz=args.drift_mhz,
                   drift_frac=args.drift_frac, selector=target["smi_selector"],
                   smi_every=args.smi_every, warmup=args.warmup_runs,
                   detrend=not args.no_detrend, gate_throttle=args.gate_throttle,
                   early_stop=not args.no_early_stop, patience=args.ci_patience,
                   plateau=args.ci_plateau, store_blocks=not args.no_store_blocks)

        wall = measure(names, lambda n: run_chain(chains[n]), flush=True,
                       tag=f"{regime.name}/wall", **mkw)
        _report_measure("wall ", wall)

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
                                tag=f"{regime.name}/graph", **mkw)
                _report_measure("graph", graph)

        prof: dict = {}
        if not args.no_profile:
            for n in names:
                pk = dict(flush=not args.no_profile_flush, selector=target["smi_selector"])
                if n in graphs:
                    prof[n] = profile_kernels(lambda nn: graphs[nn].replay(), n,
                                              args.profile_reps, **pk)
                else:
                    prof[n] = profile_kernels(lambda nn: run_chain(chains[nn]), n,
                                              args.profile_reps, **pk)

        decomp = {}
        for n in names:
            w = (wall["per_config"].get(n) or {}).get("ms_p50")
            g = ((graph or {}).get("per_config", {}).get(n) or {}).get("ms_p50")
            pk = prof.get(n) or {}
            k = pk.get("work_us")
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

            # Is `work_us` comparable to the blocked timings it is being subtracted from?
            # `pass_wall_us` was measured on the SAME executions that produced `work_us`, so
            # the first test indicts CUPTI and the second indicts the operating conditions.
            # Without these two numbers the first run's 40 negative gaps (to -905 us) and 6
            # negative launch costs (to -118 us/kernel) could not be attributed to anything.
            pw = pk.get("pass_wall_us")
            row["profile_wall_us"] = pw
            row["profile_flushed"] = pk.get("flushed")
            row["work_le_profile_wall"] = pk.get("work_le_pass_wall")
            if pw and row["wall_us"]:
                row["profile_vs_block"] = pw / row["wall_us"]
            ok_signs = all(row.get(t) is None or row[t] >= 0 for t in ("gap_us", "launch_us"))
            pvb = row.get("profile_vs_block")
            row["comparable"] = bool(pvb is not None and abs(pvb - 1.0) <= 0.05)
            row["decomposition_valid"] = bool(ok_signs and row["comparable"])
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
            "blocks": "one run per configuration per block. The running order is a ROTATION "
                      "by the block index, reversed on alternate rotation cycles, so every "
                      "configuration INCLUDING the all-unfused baseline visits every slot "
                      "equally often. The slot-0 configuration is repeated at the tail as "
                      f"the block's drift probe. {args.warmup_runs} execution(s) are run and "
                      "DISCARDED at the head of each block, un-synchronised, so the host is "
                      "already running ahead of the device when slot 0 is timed.",
            "statistic": "median over blocks of (baseline / configuration), both detrended "
                         "by the within-block trend measured by the slot-0 repeat; "
                         "percentile bootstrap 95 % CI, 2000 resamples",
            "tie_rule": "CIs overlap => tied; a configuration beats the unfused layer only "
                        "when its CI excludes 1.0",
            "drift_gate": (
                f"TWO gates. In-band, every block: the slot-0 repeat gives "
                f"t(tail)/t(head)-1 and the block is DISCARDED if |that| > "
                f"{args.drift_frac}. Out-of-band, one targeted nvidia-smi per ~"
                f"{args.smi_every} blocks: if the SM clock stepped > {args.drift_mhz} MHz "
                f"since the previous sample the whole window of blocks between them is "
                f"discarded, since the step cannot be localised within it. Harmful throttle "
                f"reasons {sorted(HARMFUL_THROTTLE)} are RECORDED and gate only under "
                f"--gate-throttle (here: {args.gate_throttle}); on this card SwPowerCap is "
                f"asserted continuously under load and gating on it discarded 70-85 % of "
                f"blocks at the large regimes while selecting for nothing measurable. "
                f"ApplicationsClocksSetting is excluded on purpose -- it is the bit the "
                f"clock lock itself sets."),
            "position_bias": (
                "The 2026-08-13 run pinned A_all_unfused to slot 0 of every block and "
                "sandwiched it. Slot 0 is timed into an EMPTY launch queue (the block is "
                "issued asynchronously with one sync at the end) and, in that run, "
                "immediately after two un-filtered nvidia-smi subprocesses; the baseline "
                "therefore paid a fixed 28-79 us block-entry penalty in every block, "
                "2.6-5.6 us of launch cost per kernel against 0.7-1.4 us for configurations "
                "with the identical kernel count, and every published speedup-vs-unfused was "
                "inflated by 0.6-8.5 %. It was NOT a clock ramp: the card read its maximum "
                "1980 MHz before and after every block with 0/200 blocks dropped for clock "
                "movement at T<=256, and the penalty was 4-5x larger in wall mode than in "
                "graph mode, which a device-side clock effect cannot distinguish. Fixed by "
                "the rotation and the discarded warm-up, not by the clock lock."),
            "raw_storage": (
                "regimes.*.wall.blocks and .graph.blocks carry, per block, the slot->config "
                "assignment, the raw per-position timings (including the tail probe, and "
                "including blocks that were DISCARDED so a gate threshold can be re-chosen "
                "offline) and the derived per-position ratio. The first run stored only the "
                "sandwich MEAN, which is why its position bias had to be reconstructed from "
                "a separate CUPTI pass instead of measured."),
            "stopping": (
                f"target {args.target_ci:.1e}; also stops when the target is projected "
                f"unreachable within --max-blocks {args.max_blocks} under 1/sqrt(n), or when "
                f"the CI improves < {args.ci_plateau:.0%} over {args.ci_patience} checks. A "
                f"target-met stop is confirmed with the published bootstrap, not the cheap "
                f"median_se proxy that decides it. See wall.stop_reason / graph.stop_reason."),
            "decomposition": "wall = work + in-graph gaps + launch cost, where work is the "
                             "CUPTI sum of per-kernel device time and graph replay is wall "
                             "minus launch cost. The CUPTI pass now flushes L2 between reps "
                             "as the timing blocks do, excludes the flush by measured kernel "
                             "name, and CUDA-event-times the same reps it profiles, so "
                             "decomposition.*.profile_vs_block states how comparable the two "
                             "passes' conditions were and decomposition.*.decomposition_valid "
                             "states whether the identity work <= graph <= wall actually held.",
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
