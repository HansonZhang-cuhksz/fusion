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
   overlay is offered only when the preflight probe COMPILED AND LAUNCHED that feature *and*
   the kernel module advertises the corresponding cfg key.  Absent either, the grids are
   exactly the classic ones and the sm_90 path simply never appears.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

HERE = Path(__file__).resolve().parent
PKG = HERE.parent  # glm52_h200/
ROOT = PKG.parent  # repository root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
MAX_ELEMS_PER_THREAD = 64  # fp32 values a lane may hold in a vector kernel's tile
MIN_ELEMS_PER_THREAD = 1  # below this the CTA is mostly idle lanes
CUDA_MAX_THREADS_PER_BLOCK = 1024  # CUDA *programming model* cap, not a device property


# ======================================================================================
# Preflight
# ======================================================================================
_PF_CACHE: dict | None = None
_PF_PATH = PKG / "preflight_h200.json"
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


def timer_tick_us() -> float | None:
    """CUDA-event granularity in microseconds, as measured, or None."""
    v = pf_get("calibration", "timer_tick_us")
    try:
        return float(v) if v else None
    except (TypeError, ValueError):
        return None


def launch_cost_us() -> float | None:
    v = pf_get("calibration", "launch_us")
    try:
        return float(v) if v else None
    except (TypeError, ValueError):
        return None


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
def h200_cfg_overlays(kernel_mod=None) -> list[dict]:
    """Extra config-dict overlays this device+stack+kernel actually supports.

    Two independent gates, both checked at RUNTIME:

      1. the preflight COMPILED AND LAUNCHED the feature on this stack, and
      2. the kernel module advertises the cfg key by exporting `H200_CFG_KEYS`
         (a tuple of strings its launcher forwards to the Triton launch).

    If either is missing the overlay list is empty and the grids below are byte-identical
    to the classic ones.  That is the whole point: this file cannot be tested on sm_90, so
    it must be incapable of *requiring* sm_90.

    Overlays are applied to the coarse grid of BOTH arms of a pair, so they cannot bias a
    ratio; they only widen the search for both.
    """
    keys = tuple(getattr(kernel_mod, "H200_CFG_KEYS", ()) or ())
    out: list[dict] = []
    if "num_ctas" in keys and feature("thread_block_cluster_num_ctas"):
        out += [{"num_ctas": 2}]
    if "warp_specialize" in keys and feature("warp_specialize_tl_range"):
        out += [{"warp_specialize": True}]
    if "num_consumer_groups" in keys and feature("warp_specialize_num_consumer_groups"):
        out += [{"num_consumer_groups": 1, "num_buffers_warp_spec": 2}]
    if "USE_TMA" in keys and feature("tma_tensor_descriptor"):
        out += [{"USE_TMA": True}]
    return out


def widen(grid: list[dict], kernel_mod=None) -> list[dict]:
    """`grid` plus every H200 overlay of every config in it, deduplicated."""
    ovl = h200_cfg_overlays(kernel_mod)
    if not ovl:
        return grid
    out = list(grid)
    for c in grid:
        for o in ovl:
            out.append(dict(c, **o))
    return dedup(out)


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


def screened_autotune(tag, make_chain, grid, verify, warmup, rep, prep=None):
    """`screen` then `common.autotune`, with the screen folded into n_tried/n_failed.

    The returned TuneResult reports the OFFERED grid size and the total rejects, so the
    per-arm fairness accounting counts what the arm was actually given, not what survived.
    """
    t0 = time.time()
    if prep is not None:
        prep()
    ok, rej = screen(tag, lambda c: [f() for f in make_chain(c)], verify, grid)
    if not ok:
        # Never let a screening tolerance destroy an hour-long run: fall back to timing the
        # unscreened grid and say so loudly, so the operator can judge the result.
        print(f"    !! [{tag}] screening rejected EVERY config; timing unscreened",
              flush=True)
        ok, rej = list(grid), []
    tr = _common.autotune(make_chain, ok, warmup, rep)
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


def tick_report(fused_ms: float, unfused_ms: float) -> dict:
    """Flag a speedup whose two operands are only a few timer ticks apart.

    At decode the kernels resolve to 9-17 CUDA-event ticks on the 4060 (1.024 us there);
    an H200 tick is measured by the preflight.  A ratio built from two integers that differ
    by 2 is quantised to tens of percent, and reporting it to three decimals is a lie of
    precision.  `tick_limited` is the flag the report must carry.
    """
    tick = timer_tick_us()
    out = {"timer_tick_us": tick}
    if not tick or fused_ms <= 0 or unfused_ms <= 0:
        out["tick_limited"] = None
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

    def add(self, regime: str, arm: str, stage: str, tune=None, size: int | None = None):
        node = self.grids.setdefault(regime, {}).setdefault(arm, {})
        if tune is not None:
            node[stage] = {
                "n_tried": getattr(tune, "n_tried", None),
                "n_failed": getattr(tune, "n_failed", None),
                "best_ms": getattr(tune, "best_ms", None),
            }
        else:
            node[stage] = {"n_tried": size}
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
        out["h200_axes_offered"] = h200_cfg_overlays(None) or "none (kernel opt-in absent)"
        out["timing"] = {
            "protocol": "final per-regime timings are A/B interleaved within one loop with "
            "the order alternating each round; the reported speedup is the median of the "
            "per-round ratios, so monotone drift cancels",
            "timer_tick_us": timer_tick_us(),
            "launch_cost_us": launch_cost_us(),
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


def add_std_args(ap: argparse.ArgumentParser, units: Sequence[str] = ()) -> None:
    """`--regimes`, `--quick`, `--only`, plus the switches every driver shares."""
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
