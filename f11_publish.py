#!/usr/bin/env python3
"""Re-measure fusion #11 (#11a / #11b / #11b') to a publishable standard.  ONE FILE.

Run:  python3 f11_publish.py --gpu auto
Send back:  results/h200/f11_publish.json  and  log/f11_publish.log

WHY THIS EXISTS RATHER THAN ANOTHER bench_f11 RUN
=================================================
The repaired bench_f11 run produced numbers that its own verification rejected.  Four
things blocked publication, and none of them is in the kernels -- they are all in how the
measurement was taken and judged.  This script closes each one, and deliberately does NOT
reuse `glm52_h200/bench/__init__.py`'s timing or screening path, because that is where the
defects live.  It DOES reuse the kernels and the fp32 reference, because those are correct
and re-implementing them would be the actual mistake.

  B1  **#11b exceeded its own physical ceiling** at decode_bs1 (measured 2.155 against a
      1.0078 byte-roofline) and decode_bs256.  The byte roofline is the wrong bar for this
      fusion: at T=1 the eliminated activation pass is worth 0.8%, and essentially the whole
      win is one fewer KERNEL LAUNCH, which a bytes-only ceiling does not model at all.
      -> This script computes a LAUNCH-AWARE ceiling and refuses to publish any cell that
         still exceeds it, instead of quietly printing a number above its own bound.

  B2  **The decomposition data was unusable.**  `components.norm_only_ms` fell OUTSIDE the
      full min-max range of that same kernel's own 164-config sweep in 5 of 7 regimes (at
      decode_bs1: sweep [0.0096, 0.0238] ms, published 0.0879) and was nearly flat in T.
      -> This script never reads `components`.  It measures the standalone norm kernel
         itself, in the same harness, at its own tuned config, and reports the sweep range
         next to it so the reader can see the number is inside its own distribution.

  B3  **The invariance screen that was supposed to catch the wgmma defect never ran.**
      `invariance_verdict` / `invariance_partner` in kernels/lazy_prenorm.py have zero call
      sites; what executed instead compared at `check()`'s default tol=2e-2 -- 2000x looser
      than the 1e-5 the API documents -- and probed 1 of the 8 keys it declares invariant.
      Four MoE probes passed only on that tolerance, all with bitwise_identical=false, and a
      BLOCK_M=64 wrong-answer config still became a tuned winner.
      -> This script screens the winner against EVERY invariant key at tol 1e-5, and a
         failure REJECTS the config rather than annotating it.

  B4  **The calibration said `trusted: true` on a contended card.**  harness_floor_us=39.87
      against config.FLOOR_US_MAX=20.0.  A floor that size is comparable to the entire decode
      measurement, and it is added to BOTH arms, so a ratio built on it is not a ratio of the
      work.
      -> This script measures the floor itself before anything else and REFUSES TO RUN if it
         is above the bar, rather than recording a verdict nobody reads.

WHAT IT MEASURES, AND WHY TWICE
===============================
Each arm is timed two ways at the same tuned config:

  wall    ordinary chain timing, L2-flushed, arms INTERLEAVED A/B/A/B so monotone drift
          cancels in the paired ratio.  Includes per-launch cost.  This is the number that
          corresponds to deployment.
  graph   the same chain captured in a CUDA graph and replayed N times inside one timed
          window.  Launch cost is amortised to near zero.

`wall / graph` is the decomposition B2 could not supply: if a fusion's win survives in the
graph number it is real work; if it collapses, the win was the eliminated launch.  Both are
reported.  Neither is "the" answer -- for a decode-bound server the wall number is what you
feel, and for understanding the kernel the graph number is what matters.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from glm52_h200 import config as C  # noqa: E402
from glm52_h200 import reference as R  # noqa: E402
from glm52_h200.kernels import lazy_prenorm as K  # noqa: E402

EPS = 1e-5
TOPK = 8
INVARIANT_TOL = 1e-5  # the tolerance the API documents; NOT check()'s 2e-2 default
FLOOR_US_MAX = getattr(C, "FLOOR_US_MAX", 20.0)


# ======================================================================================
# 0. GPU selection and the calibration gate (B4)
# ======================================================================================
def pick_gpu(want: str) -> tuple:
    """(index, why). `auto` takes the idlest card; a digit pins it; `none` leaves it alone."""
    if want == "none":
        return None, "not pinned"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
        cards = []
        for line in out:
            i, used, util = (v.strip() for v in line.split(","))
            cards.append((int(i), int(float(used)), float(util)))
    except Exception as exc:  # noqa: BLE001
        return (int(want) if want.isdigit() else None), f"nvidia-smi unavailable ({exc})"
    if want.isdigit():
        c = next((x for x in cards if x[0] == int(want)), None)
        if c and (c[1] > 1024 or c[2] > 5):
            print(f"!! GPU {c[0]} has {c[1]} MiB used / {c[2]:.0f}% util -- a co-tenant is "
                  f"exactly what produced the 39.87 us harness floor that blocked the last "
                  f"run. Use --gpu auto, or --allow-busy to override.", flush=True)
        return int(want), "pinned by --gpu"
    idle = sorted(cards, key=lambda x: (x[2], x[1]))
    for i, used, util in idle:
        print(f"   GPU {i}: {used:>7} MiB used, {util:>3.0f}% util", flush=True)
    return idle[0][0], f"idlest of {len(cards)} (used {idle[0][1]} MiB, util {idle[0][2]:.0f}%)"


_FLUSH = None


def flush_l2() -> None:
    global _FLUSH
    if _FLUSH is None:
        l2 = torch.cuda.get_device_properties(0).L2_cache_size
        _FLUSH = torch.empty(max(256 * 2**20, 4 * l2) // 4, device="cuda", dtype=torch.int32)
    _FLUSH.zero_()


def calibrate() -> dict:
    """Measure the harness floor and per-launch cost BEFORE trusting any ratio.

    The floor is what an empty timed region costs. It is added to both arms, so when it is
    comparable to the kernels it compresses every ratio toward 1 and, worse, makes the
    ceiling comparison meaningless. The last run recorded 39.87 us here and published
    anyway; this one stops.
    """
    import triton
    import triton.language as tl

    src = ROOT / "_f11_nop.py"
    src.write_text("import triton, triton.language as tl\n"
                   "@triton.jit\ndef nop(P):\n    pass\n")
    sys.path.insert(0, str(ROOT))
    import importlib
    nk = importlib.import_module("_f11_nop")

    buf = torch.zeros(1, device="cuda")

    def timed(n: int) -> float:
        flush_l2()
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        for _ in range(n):
            nk.nop[(1,)](buf)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)

    for _ in range(30):
        timed(1)
    pts = {n: statistics.median([timed(n) for _ in range(120)]) for n in (1, 2, 4, 8)}
    launch = (pts[8] - pts[1]) / 7
    floor = pts[1] - launch
    try:
        src.unlink()
    except OSError:
        pass
    return {"harness_floor_us": floor * 1e3, "launch_us": launch * 1e3,
            "floor_bar_us": FLOOR_US_MAX,
            "ok": (floor * 1e3) <= FLOOR_US_MAX}


# ======================================================================================
# 1. timing: interleaved wall-clock, and CUDA-graph replay
# ======================================================================================
def time_wall(fns, warmup: int, rep: int) -> list:
    """Per-repetition times, L2 flushed before each. Returns the raw list, not a summary."""
    for _ in range(warmup):
        for f in fns:
            f()
    torch.cuda.synchronize()
    out = []
    for _ in range(rep):
        flush_l2()
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        for f in fns:
            f()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return out


def time_graph(fns, inner: int = 64, warmup: int = 5, rep: int = 40) -> "float | None":
    """ms per chain iteration with launch cost amortised by CUDA-graph replay.

    `inner` iterations are captured into ONE graph, so replaying it costs one launch for
    `inner` chains. The difference from `time_wall` is the per-launch overhead -- which is
    the whole of #11b's decode win, and the thing the byte roofline cannot see.
    Returns None if capture fails (some Triton launches are not capturable).
    """
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                for f in fns:
                    f()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(inner):
                for f in fns:
                    f()
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()

        vals = []
        for _ in range(rep):
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize()
            vals.append(a.elapsed_time(b) / inner)
        return statistics.median(vals)
    except Exception:  # noqa: BLE001 -- graph capture is a bonus, never a requirement
        return None


def paired(fused_fns, unfused_fns, warmup: int, rep: int) -> dict:
    """Interleaved A/B/A/B. Monotone drift cancels in the per-repetition ratio; it does not
    cancel in a ratio of two separately-measured medians, which is how the campaign got a
    speedup above its own ceiling."""
    for _ in range(warmup):
        for f in fused_fns:
            f()
        for f in unfused_fns:
            f()
    torch.cuda.synchronize()
    F, U, ratios = [], [], []
    for _ in range(rep):
        f = time_wall(fused_fns, 0, 1)[0]
        u = time_wall(unfused_fns, 0, 1)[0]
        F.append(f)
        U.append(u)
        ratios.append(u / f if f > 0 else float("nan"))
    rs = sorted(r for r in ratios if not math.isnan(r))
    half = len(ratios) // 2
    return {
        "fused_ms": statistics.median(F), "unfused_ms": statistics.median(U),
        "paired_p50": rs[len(rs) // 2] if rs else float("nan"),
        "paired_p10": rs[max(0, len(rs) // 10)] if rs else float("nan"),
        "paired_p90": rs[min(len(rs) - 1, 9 * len(rs) // 10)] if rs else float("nan"),
        "ratio_of_medians": statistics.median(U) / statistics.median(F),
        "drift_first_half": statistics.median(ratios[:half]) if half else float("nan"),
        "drift_second_half": statistics.median(ratios[half:]) if half else float("nan"),
        "n": len(ratios),
    }


# ======================================================================================
# 2. the strict invariance screen (B3)
# ======================================================================================
#: Legal alternative values per mapping key, for building an invariance partner. One
#: generic list cannot serve all of them: `num_stages=16` and `warp_specialize=16` are not
#: configs, they are compile failures that a screen would then score as "did not run".
#: BLOCK_M deliberately crosses 64, the wgmma lowering boundary the defect lives on.
_PARTNER_CHOICES = {
    "BLOCK_N": (32, 64, 128, 256),
    "GROUP_M": (1, 4, 8, 16),
    "num_stages": (2, 3, 4, 5),
    "num_warps": (4, 8, 16),
    "num_ctas": (1, 2),
    "WARP_SPECIALIZE": (False, True),
    "warp_specialize": (False, True),
    "WS": (False, True),
}


def invariance_screen(run_at, cfg: dict, keys=None, tol: float = INVARIANT_TOL) -> dict:
    """Reject a config whose output depends on a mapping key it is mathematically invariant to.

    `sq` is a full-K reduction over ONE row, held in CTA-local registers and recomputed
    identically by every n-tile. Nothing about which rows a CTA owns, how many warps it has,
    how deeply the loop is staged, or whether it is warp specialized can change `rstd` for a
    given row. So a dependence on any of these keys is a codegen defect BY CONSTRUCTION --
    which is what the 320 wrong-answer configs (all BLOCK_M in {64,128}, none in {16,32})
    were.

    The previous screen probed BLOCK_M only, at tol 2e-2. This probes every declared key at
    1e-5 and returns `ok=False`, so the caller drops the config instead of annotating it.
    """
    # `K.INVARIANT_CFG_KEYS` omits `num_warps`, and the campaign proved that axis is live:
    # at prefill_t8192 the published winner differed from a config wrong by 4.84e-1 in
    # `num_warps` + `GROUP_M` only, and at prefill_t2048 in `num_stages` alone. num_warps is
    # invariant by the same argument as the rest -- how many warps a CTA has cannot change a
    # per-row full-K reduction -- so it belongs in the screen. BLOCK_K and SQ_MODE are
    # deliberately NOT here: they legitimately change fp32 summation order.
    keys = keys or [k for k in (tuple(K.INVARIANT_CFG_KEYS) + ("num_warps",)) if k in cfg]
    base = run_at(cfg)
    if base is None:
        return {"ok": False, "reason": "base config did not run"}
    probes = []
    worst = 0.0
    for key in keys:
        # Build the partner from a PER-KEY table. `K.invariance_partner` applies one
        # generic choices=(16,32,64,128,256) list to every key, so it proposes
        # `num_stages=16`, `warp_specialize=16`, `num_ctas=16` -- nonsense configs that fail
        # to compile, get recorded as `ran: False` rather than as failures, and let the
        # screen pass silently. A screen that cannot fail is worse than no screen.
        alts = _PARTNER_CHOICES.get(key)
        if alts is None and hasattr(K, "invariance_partner") and key == "BLOCK_M":
            partner = K.invariance_partner(cfg, key)   # its tile choices ARE right for BM
        else:
            cur = cfg.get(key)
            pick = next((v for v in (alts or ()) if v != cur), None)
            partner = dict(cfg, **{key: pick}) if pick is not None else None
        if partner is None or partner.get(key) == cfg.get(key):
            probes.append({"key": key, "skipped": "no legal partner value"})
            continue
        other = run_at(partner)
        if other is None:
            # FAIL CLOSED. A probe that could not run has not shown invariance, and treating
            # it as a pass is how `11a_w13` at decode_bs1 was marked publishable with the
            # BLOCK_M axis -- the one the wgmma defect lives on -- never tested.
            probes.append({"key": key, "partner": partner[key], "ran": False,
                           "pass": False,
                           "why": "partner config did not run; invariance UNTESTED on this "
                                  "axis, which is not the same as invariant"})
            continue
        d = (base.float() - other.float()).abs()
        scale = base.float().abs().max().clamp_min(1e-30)
        rel = (d.max() / scale).item()
        worst = max(worst, rel)
        probes.append({"key": key, "from": cfg.get(key), "to": partner[key],
                       "rel_err": rel, "bitwise": bool(torch.equal(base, other)),
                       "pass": rel <= tol})
    failed = [p for p in probes if p.get("pass") is False]
    untested = [p for p in probes if p.get("ran") is False]
    return {"ok": not failed, "tol": tol, "worst_rel_err": worst,
            "n_untested": len(untested),
            "n_probed": len(probes), "keys": keys, "probes": probes,
            "reason": ("" if not failed else
                       ("output depends on "
                        + ", ".join(sorted({p["key"] for p in failed if p.get("ran") is not False}))
                        + " -- mathematically invariant, so this is a codegen defect"
                        if any(p.get("ran") is not False for p in failed) else "")
                       + ("" if not untested else
                          ("; " if any(p.get("ran") is not False for p in failed) else "")
                          + "UNTESTED on " + ", ".join(sorted({p["key"] for p in untested}))
                          + " (partner did not run) -- untested is not invariant"))}


# ======================================================================================
# 3. ceilings (B1)
# ======================================================================================
def self_consistency(wall_ratio: float, graph_f: "float | None", graph_u: "float | None",
                     n_kern_f: int, n_kern_u: int, launch_s: float, floor_s: float,
                     tol: float = 0.05) -> dict:
    """Can this wall ratio be produced by its OWN graph work plus its OWN overheads?

    wall_arm ~= floor + n_launches*launch + work, and `graph` measures `work` directly. So

        bound = (graph_u + n_u*L + floor) / (graph_f + n_f*L + floor)

    is the largest wall ratio these components can generate. A wall figure ABOVE it is not
    explainable by anything the run itself measured, and means the two arms are not doing
    what the model says they are.

    This is the gate that would have caught the defect that produced the last unpublishable
    table: the unfused chain's GEMM read a different buffer than its norm wrote, so it paid a
    cold DRAM read its real counterpart would not, inflating the unfused arm. Every wall
    figure then sat 13-34 % above this bound while the graph figures were self-consistent.
    """
    if not graph_f or not graph_u or graph_f <= 0:
        return {"checked": False, "why": "no graph timing (capture failed)"}
    f = graph_f * 1e-3 + n_kern_f * launch_s + floor_s
    u = graph_u * 1e-3 + n_kern_u * launch_s + floor_s
    bound = u / f if f > 0 else float("nan")
    return {"checked": True, "bound": bound, "wall": wall_ratio,
            "excess_pct": (wall_ratio / bound - 1.0) * 100.0 if bound > 0 else float("nan"),
            "ok": bool(wall_ratio <= bound * (1.0 + tol))}


def ceilings(bytes_f: int, bytes_u: int, n_kern_f: int, n_kern_u: int,
             bw_bytes_per_s: float, launch_s: float) -> dict:
    """Byte ceiling, and the launch-aware ceiling that actually bounds this fusion.

    The byte ceiling is `bytes_u / bytes_f`. At decode that is ~1.008 while the measured
    speedup was 2.155, because the win is not traffic -- it is one fewer kernel launch. The
    launch-aware bound charges each arm its ideal traffic time PLUS its launches, which is
    the smallest honest upper bound on what fusing can buy.
    """
    t_f = bytes_f / bw_bytes_per_s + n_kern_f * launch_s
    t_u = bytes_u / bw_bytes_per_s + n_kern_u * launch_s
    return {"ceiling_bytes": (bytes_u / bytes_f) if bytes_f else float("nan"),
            "ceiling_launch_aware": (t_u / t_f) if t_f > 0 else float("nan"),
            "ideal_fused_ms": t_f * 1e3, "ideal_unfused_ms": t_u * 1e3}


# ======================================================================================
# 4. the problem
# ======================================================================================
class Problem:
    """Tensors and dispatch for one regime. Mirrors bench_f11's Problem, kept local so the
    measurement does not depend on the module whose defects this script exists to bypass."""

    def __init__(self, T: int, with_w13: bool, seed: int = 0):
        H, ER = C.HIDDEN_SIZE, C.N_ROUTED_EXPERTS
        self.T, self.H, self.ER, self.with_w13 = T, H, ER, with_w13
        g = torch.Generator(device="cuda").manual_seed(seed)
        dt = torch.bfloat16
        self.h1 = torch.empty(T, H, device="cuda", dtype=dt).normal_(0, .1, generator=g)
        self.w = (torch.randn(H, generator=g, device="cuda") * .1 + 1).to(dt)
        self.b_raw = torch.empty(H, ER, device="cuda", dtype=dt).normal_(0, .02, generator=g)
        self.b_fold = (self.w[:, None].float() * self.b_raw.float()).to(dt).contiguous()
        self.x2 = R.rmsnorm(self.h1, self.w)
        self.logits_f = torch.empty(T, ER, device="cuda", dtype=torch.float32)
        self.logits_u = torch.empty_like(self.logits_f)
        self.logits_h = torch.empty_like(self.logits_f)
        self.rstd = torch.empty(T, device="cuda", dtype=torch.float32)
        # THE UNFUSED CHAIN MUST BE A REAL CHAIN.
        #
        # `norm_only` writes x2_out and the unfused GEMM must READ THAT SAME BUFFER. Pointing
        # the GEMM at a separately-allocated `self.x2` left the two kernels unconnected: after
        # flush_l2() the GEMM paid a full cold DRAM read for a tensor that, in the real
        # unfused path, its predecessor had just written and left L2-warm. That inflates the
        # unfused arm, and therefore the speedup -- which is why every `wall` figure came out
        # above what its own graph work plus launch could produce, while the `graph` figures
        # (replayed back-to-back, so the buffer stays warm) were self-consistent.
        #
        # Seeded with the correct values so the GEMM can also be tuned STANDALONE, before any
        # norm has run in that chain; `norm_only` then overwrites it with the same numbers.
        self.x2_out = self.x2.clone()
        self.ref_logits = (self.x2.float() @ self.b_raw.float())

        self.w13_raw = self.w13_fold = None
        if with_w13:
            I2 = C.MOE_INTERMEDIATE_SIZE
            NW13 = 2 * I2
            self.NW13 = NW13
            self.w13_raw = torch.empty(ER, NW13, H, device="cuda", dtype=dt)
            self.w13_raw.normal_(0, .02, generator=g)
            self.w13_fold = (self.w13_raw.float()
                             * self.w[None, None, :].float()).to(dt).contiguous()
            logits = self.ref_logits
            self.topk_ids = logits.topk(TOPK, dim=-1).indices.to(torch.int32)
            self.rows = T * TOPK
            self.c_f = torch.empty(self.rows, NW13, device="cuda", dtype=dt)
            self.c_u = torch.empty_like(self.c_f)
            self.c_h = torch.empty_like(self.c_f)

    def layout(self, bm: int):
        return R.moe_align_block_size(self.topk_ids, bm, self.ER)

    # ---- arms ----
    def router_fused(self, cfg, ws=None):
        return [lambda: K.launch_router(self.h1, self.b_fold, self.logits_f, cfg, True,
                                        EPS, 0, None, ws)]

    def router_unfused(self, cfg, ws=None):
        return [lambda: K.launch_router(self.x2_out, self.b_raw, self.logits_u, cfg, False,
                                        EPS, 0, None, ws)]

    def norm_only(self, cfg):
        from glm52_h200.kernels import add_rmsnorm as NK
        return [lambda: NK.norm_only(self.h1, self.w, self.x2_out, cfg)]

    def rstd_only(self, cfg):
        return [lambda: K.launch_rstd(self.h1, self.rstd, cfg, EPS)]

    def router_half(self, cfg):
        return [lambda: K.launch_router(self.h1, self.b_fold, self.logits_h, cfg, False,
                                        EPS, 0, self.rstd)]

    def moe_fused(self, cfg, ws=None):
        sti, eid, ntp = self.layout(cfg["BLOCK_M"])
        return [lambda: K.launch_moe_gateup(self.h1, self.w13_fold, self.c_f, sti, eid, ntp,
                                            self.rows, TOPK, cfg, True, EPS, 0, None, ws)]

    def moe_unfused(self, cfg, ws=None):
        sti, eid, ntp = self.layout(cfg["BLOCK_M"])
        return [lambda: K.launch_moe_gateup(self.x2_out, self.w13_raw, self.c_u, sti, eid,
                                            ntp, self.rows, TOPK, cfg, False, EPS, 0, None,
                                            ws)]


# ======================================================================================
# 5. grids and tuning
# ======================================================================================
def gemm_grid(env) -> list:
    smem = env.smem_bytes
    out = []
    for bm in (16, 32, 64, 128):
        for bn in (32, 64, 128, 256):
            for bk in (32, 64, 128):
                for w in (4, 8):
                    for s in (2, 3, 4):
                        if max(2, s - 1) * 2 * bk * (bm + bn) > smem:
                            continue
                        out.append(dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                                        num_warps=w, num_stages=s, GROUP_M=8))
    return out


def vec_grid() -> list:
    return [dict(ROWS=r, BLOCK_N=b, num_warps=w, num_stages=s, grid_cap=None, eps=EPS)
            for b in (1024, 2048, 4096, 8192) for r in (1, 2, 4)
            for w in (4, 8, 16) for s in (1, 2)]


def tune(tag, make, grid, verify, warmup, rep, log) -> dict:
    """Screen every config against the fp32 reference, then time only the survivors.

    No refine stage: `common.autotune`'s default refine invents configs via `neighbours()`
    that pass through neither the screen nor the legality filter, and one of those became a
    wrong-answer winner in the last run. The coarse grid is the whole search here.
    """
    ok, rej = [], []
    for cfg in grid:
        try:
            for f in make(cfg):
                f()
            torch.cuda.synchronize()
            good, detail = verify()
            (ok if good else rej).append((cfg, detail))
        except Exception as exc:  # noqa: BLE001
            rej.append((cfg, f"{type(exc).__name__}: {str(exc)[:90]}"))
    log(f"    [{tag}] {len(grid)} offered -> {len(ok)} valid "
        f"({len(grid) - len(ok)} rejected)")
    if not ok:
        return {"ok": False, "reason": "every config rejected", "n_offered": len(grid),
                "rejects": [str(r[1])[:120] for r in rej[:5]]}
    best, best_ms = None, float("inf")
    for cfg, _ in ok:
        ts = time_wall(make(cfg), 2, 7)
        m = statistics.median(ts)
        if m < best_ms:
            best, best_ms = cfg, m
    log(f"    [{tag}] best {best_ms:.4f} ms  {best}")
    return {"ok": True, "cfg": best, "ms": best_ms, "n_offered": len(grid),
            "n_valid": len(ok),
            "sweep_min_ms": min(statistics.median(time_wall(make(c), 1, 3))
                                for c, _ in ok[:1]) if ok else None}


# ======================================================================================
# 6. per-regime measurement
# ======================================================================================
def run_regime(name: str, T: int, args, env, calib, log) -> dict:
    log(f"\n===== {name} (T={T}) =====")
    with_w13 = args.with_w13 and not args.router_only
    prob = Problem(T, with_w13)
    row: dict = {"regime": name, "T": T, "arms": {}}
    warm, rep = (3, 15) if args.quick else (10, 40)
    bw = float(args.bandwidth_gbs) * 1e9
    launch_s = calib["launch_us"] * 1e-6

    def rel(a, b):
        return ((a.float() - b.float()).abs().max()
                / b.float().abs().max().clamp_min(1e-30)).item()

    # ---- router arms ----
    grid = gemm_grid(env)
    tf = tune("routerF", lambda c: prob.router_fused(c),
              grid, lambda: (rel(prob.logits_f, prob.ref_logits) < 2e-2,
                             f"{rel(prob.logits_f, prob.ref_logits):.2e}"), warm, rep, log)
    tu = tune("routerU", lambda c: prob.router_unfused(c),
              grid, lambda: (rel(prob.logits_u, prob.ref_logits) < 2e-2,
                             f"{rel(prob.logits_u, prob.ref_logits):.2e}"), warm, rep, log)
    tn = tune("norm", lambda c: prob.norm_only(c), vec_grid(),
              lambda: (rel(prob.x2_out, prob.x2) < 2e-2,
                       f"{rel(prob.x2_out, prob.x2):.2e}"), warm, rep, log)
    # The rstd producer MUST be verified like every other kernel. It was previously tuned
    # with `lambda: (True, "")` -- an unconditional pass -- so no config was ever compared
    # against anything, and #11b' inherited a number with no numerical evidence behind it.
    ref_rstd = torch.rsqrt(prob.h1.float().pow(2).mean(-1) + EPS)

    def _rstd_ok():
        e = ((prob.rstd.float() - ref_rstd).abs().max()
             / ref_rstd.abs().max().clamp_min(1e-30)).item()
        return e < 2e-2, f"{e:.2e}"

    tr = tune("rstd", lambda c: prob.rstd_only(c), vec_grid(), _rstd_ok, warm, rep, log)

    if tf["ok"] and tu["ok"] and tn["ok"]:
        # B3: strict invariance screen on the FUSED winner before it is allowed to publish
        def run_at(cfg):
            try:
                for f in prob.router_fused(cfg):
                    f()
                torch.cuda.synchronize()
                return prob.logits_f.clone()
            except Exception:  # noqa: BLE001
                return None

        inv = invariance_screen(run_at, tf["cfg"])
        log(f"    [invariance routerF] {'PASS' if inv['ok'] else 'REJECT'} "
            f"worst={inv.get('worst_rel_err', float('nan')):.2e} over {inv['n_probed']} keys")

        fused = prob.router_fused(tf["cfg"])
        unf = prob.norm_only(tn["cfg"]) + prob.router_unfused(tu["cfg"])
        p = paired(fused, unf, warm, rep)
        gf, gu = time_graph(fused), time_graph(unf)
        H, ER = prob.H, prob.ER
        b_f = T * H * 2 + H * ER * 2 + T * ER * 4
        b_u = (T * H * 2 * 2) + (T * H * 2 + H * ER * 2 + T * ER * 4)
        cl = ceilings(b_f, b_u, 1, 2, bw, launch_s)
        sc = self_consistency(p["paired_p50"], gf, gu, 1, 2, launch_s,
                              calib["harness_floor_us"] * 1e-6)
        row["arms"]["11b_router"] = {
            **p, "graph_fused_ms": gf, "graph_unfused_ms": gu,
            "graph_speedup": (gu / gf) if (gf and gu) else None,
            **cl, "invariance": inv, "self_consistency": sc,
            "fused_cfg": tf["cfg"], "unfused_gemm_cfg": tu["cfg"], "norm_cfg": tn["cfg"],
            "norm_only_ms": tn["ms"], "gemm_only_ms": tu["ms"],
            "publishable": bool(inv["ok"]
                                and p["paired_p50"] <= cl["ceiling_launch_aware"]
                                and sc.get("ok", False)),
        }
        a = row["arms"]["11b_router"]
        log(f"  #11b  wall {p['paired_p50']:.3f}x | graph "
            f"{a['graph_speedup'] if a['graph_speedup'] else float('nan'):.3f}x | "
            f"ceiling {cl['ceiling_launch_aware']:.3f} | "
            f"self-consistent<={sc.get('bound', float('nan')):.3f} | "
            f"{'PUBLISHABLE' if a['publishable'] else 'BLOCKED'}")

        if tr["ok"]:
            half = prob.rstd_only(tr["cfg"]) + prob.router_half(tf["cfg"])

            def run_half(cfg):
                try:
                    for f in prob.rstd_only(tr["cfg"]) + prob.router_half(cfg):
                        f()
                    torch.cuda.synchronize()
                    return prob.logits_h.clone()
                except Exception:  # noqa: BLE001
                    return None

            inv_h = invariance_screen(run_half, tf["cfg"])
            e_h = rel(prob.logits_h, prob.ref_logits)
            log(f"    [invariance routerHalf] {'PASS' if inv_h['ok'] else 'REJECT'} "
                f"worst={inv_h.get('worst_rel_err', float('nan')):.2e} | "
                f"vs fp32 ref {e_h:.2e}")
            ph = paired(half, unf, warm, rep)
            # Byte count: the half-fused path reads h1 TWICE -- once in the rstd kernel and
            # again in the router GEMM -- plus the tiny rstd vector. Omitting the second read
            # made the ceiling 55% too generous at t8192 and is what let this arm look
            # bounded when it was not.
            b_half = (T * H * 2) + (T * 4) + (T * H * 2 + H * ER * 2 + T * ER * 4)
            cl_h = ceilings(b_half, b_u, 2, 2, bw, launch_s)
            row["arms"]["11b_half"] = {
                **ph, "graph_fused_ms": time_graph(half), "graph_unfused_ms": gu,
                **cl_h, "invariance": inv_h, "rel_err_vs_fp32": e_h,
                "rstd_cfg": tr["cfg"], "router_cfg": tf["cfg"],
                "publishable": bool(inv_h["ok"] and e_h < 2e-2
                                    and ph["paired_p50"] <= cl_h["ceiling_launch_aware"]),
            }
            log(f"  #11b' wall {ph['paired_p50']:.3f}x | "
                f"{'PUBLISHABLE' if row['arms']['11b_half']['publishable'] else 'BLOCKED'}")
    else:
        row["arms"]["11b_router"] = {"unmeasurable": True,
                                     "why": {k: v for k, v in
                                             (("fused", tf), ("unfused", tu), ("norm", tn))
                                             if not v["ok"]}}

    # ---- #11a: the w13 consumer ----
    if with_w13:
        idx = torch.arange(min(prob.rows, 1024), device="cuda")
        tokm = (idx // TOPK).long()
        ex = prob.topk_ids.long()[tokm, (idx % TOPK).long()]
        ref = torch.empty(idx.numel(), prob.NW13, device="cuda", dtype=torch.float32)
        xs = prob.x2.float()[tokm]
        for e in torch.unique(ex).tolist():
            sel = (ex == e).nonzero(as_tuple=True)[0]
            ref[sel] = xs[sel] @ prob.w13_raw[e].float().t()

        def vf():
            prob.c_f.fill_(float("nan"))
            return None

        mg = [c for c in gemm_grid(env) if c["BLOCK_N"] <= 256]
        mf = tune("w13F", lambda c: prob.moe_fused(c), mg,
                  lambda: (rel(prob.c_f[idx], ref) < 2e-2, f"{rel(prob.c_f[idx], ref):.2e}"),
                  warm, rep, log)
        mu = tune("w13U", lambda c: prob.moe_unfused(c), mg,
                  lambda: (rel(prob.c_u[idx], ref) < 2e-2, f"{rel(prob.c_u[idx], ref):.2e}"),
                  warm, rep, log)
        if mf["ok"] and mu["ok"] and tn["ok"]:
            def run_at_moe(cfg):
                try:
                    for f in prob.moe_fused(cfg):
                        f()
                    torch.cuda.synchronize()
                    return prob.c_f[idx].clone()
                except Exception:  # noqa: BLE001
                    return None

            inv = invariance_screen(run_at_moe, mf["cfg"])
            log(f"    [invariance w13F] {'PASS' if inv['ok'] else 'REJECT'} "
                f"worst={inv.get('worst_rel_err', float('nan')):.2e}")
            fused = prob.moe_fused(mf["cfg"])
            unf = prob.norm_only(tn["cfg"]) + prob.moe_unfused(mu["cfg"])
            p = paired(fused, unf, warm, rep)
            row["arms"]["11a_w13"] = {
                **p, "graph_fused_ms": time_graph(fused), "graph_unfused_ms": time_graph(unf),
                "invariance": inv, "fused_cfg": mf["cfg"], "unfused_cfg": mu["cfg"],
                "publishable": bool(inv["ok"]),
            }
            log(f"  #11a  wall {p['paired_p50']:.3f}x | "
                f"{'PUBLISHABLE' if inv['ok'] else 'BLOCKED (invariance)'}")
        else:
            row["arms"]["11a_w13"] = {"unmeasurable": True}
    return row


# ======================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpu", default="auto", help="auto | <index> | none")
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--regimes", default="decode_bs1,decode_bs32,decode_bs256,decode_bs512,"
                                          "decode_bs1024,prefill_t2048,prefill_t8192")
    ap.add_argument("--router-only", action="store_true", help="skip #11a (no 12 GB w13)")
    ap.add_argument("--with-w13", action="store_true", default=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bandwidth-gbs", default="4250",
                    help="measured streaming r+w bandwidth, for the ceilings")
    ap.add_argument("--out", default="results/h200/f11_publish.json")
    ap.add_argument("--log", default="log/f11_publish.log")
    ap.add_argument("--force-calib", action="store_true",
                    help="run even if the harness floor fails its bar (NOT publishable)")
    a = ap.parse_args()

    gpu, why = pick_gpu(a.gpu)
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    logp = ROOT / a.log
    logp.parent.mkdir(parents=True, exist_ok=True)
    fh = logp.open("w")

    def log(m=""):
        print(m, flush=True)
        fh.write(m + "\n")
        fh.flush()

    torch.cuda.init()
    env = C.env()
    log(f"# f11_publish.py  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"# gpu: {gpu} ({why})")
    log(env.banner())

    log("\n=== calibration gate ===")
    calib = calibrate()
    log(f"  harness floor {calib['harness_floor_us']:.3f} us "
        f"(bar {calib['floor_bar_us']}) | launch {calib['launch_us']:.3f} us")
    if not calib["ok"]:
        log(f"  !! FLOOR ABOVE BAR. A floor this size is comparable to the whole decode\n"
            f"     measurement and is added to BOTH arms, so no ratio taken here bounds the\n"
            f"     work. This is exactly what blocked the last run (39.87 us).\n"
            f"     Pick an idle GPU (--gpu auto) or pass --force-calib to record anyway.")
        if not a.force_calib:
            json.dump({"aborted": "calibration", "calibration": calib},
                      (ROOT / a.out).open("w"), indent=2)
            return 2
    else:
        log("  floor within bar -- ratios measured here are trustworthy")

    regimes = []
    for nm in a.regimes.split(","):
        nm = nm.strip()
        if not nm:
            continue
        try:
            regimes.append((nm, C.regime(nm).T))
        except Exception:  # noqa: BLE001
            log(f"  !! unknown regime {nm!r}, skipping")

    out = {"id": "f11_publish", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "gpu": {"index": gpu, "why": why, "name": env.device_name},
           "env": env.__dict__ if hasattr(env, "__dict__") else {},
           "calibration": calib, "bandwidth_gbs": float(a.bandwidth_gbs), "rows": []}

    for nm, T in regimes:
        try:
            out["rows"].append(run_regime(nm, T, a, env, calib, log))
        except torch.cuda.OutOfMemoryError as exc:
            log(f"  !! {nm}: OOM ({exc})"[:200])
            out["rows"].append({"regime": nm, "T": T, "failed": "OOM"})
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 -- one regime must not cost the rest
            log(f"  !! {nm}: {type(exc).__name__}: {exc}"[:300])
            out["rows"].append({"regime": nm, "T": T,
                                "failed": f"{type(exc).__name__}: {exc}"[:300]})
            torch.cuda.empty_cache()

    (ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, (ROOT / a.out).open("w"), indent=2, default=str)

    log("\n=== SUMMARY (wall = with launch, graph = launch amortised) ===")
    log(f"{'regime':<15}{'arm':<12}{'wall':>9}{'graph':>9}{'ceiling':>10}  verdict")
    for r in out["rows"]:
        for arm, v in (r.get("arms") or {}).items():
            if v.get("unmeasurable"):
                log(f"{r['regime']:<15}{arm:<12}{'-':>9}{'-':>9}{'-':>10}  UNMEASURABLE")
                continue
            g = v.get("graph_speedup")
            log(f"{r['regime']:<15}{arm:<12}{v.get('paired_p50', float('nan')):>9.3f}"
                f"{(g if g else float('nan')):>9.3f}"
                f"{v.get('ceiling_launch_aware', float('nan')):>10.3f}  "
                f"{'PUBLISHABLE' if v.get('publishable') else 'BLOCKED'}")
    log(f"\nwrote {a.out}")
    fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
