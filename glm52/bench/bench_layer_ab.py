"""Interleaved A/B re-measurement of the top layer configurations.

The first pass (`bench_layer.py`) separates the clear losers, but its winners land within
~0.4 % of each other — at or below this machine's noise floor (the study logged one-off
excursions of 25-320 %). Declaring a winner from a single sequential pass would be
over-reading the data.

This runs R rounds, and within each round times every candidate once, in the same order.
Drift that affects a whole round cancels in the per-config median. A winner is reported only
when its median is separated from the runner-up's by more than the round-to-round spread of
both; otherwise the configurations are reported as tied.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from glm52.bench.bench_layer import (  # noqa: E402
    CONFIGS, FAMILY_TUNED, LayerProblem, REGIMES, build_chain, load_cfgs,
    tune_layer_cfgs, tune_shared,
)
from glm52.common import bench_chain, check, record  # noqa: E402

RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--rep", type=int, default=15)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--force", default="", help="comma-separated configs to always include")
    ap.add_argument("--out", default="layer_configurations_ab")
    args = ap.parse_args()
    forced = [s.strip() for s in args.force.split(",") if s.strip()]

    first = json.loads((RESULTS / "layer_configurations.json").read_text())["regimes"]
    out = {}

    for regime in args.regimes.split(","):
        regime = regime.strip()
        print(f"\n===== {regime} (interleaved, {args.rounds} rounds) =====", flush=True)
        prev = {k: v["ms"] for k, v in first[regime]["rows"].items()
                if "ms" in v and v.get("correct")}
        cands = sorted(prev, key=prev.get)[: args.top]
        for extra in forced + ["A_all_unfused"]:
            if extra not in cands and extra in CONFIGS:
                cands.append(extra)
        print(f"  candidates: {', '.join(cands)}", flush=True)

        p = LayerProblem(regime)
        cfg = load_cfgs(regime) if regime in FAMILY_TUNED else tune_layer_cfgs(p)
        shared = tune_shared(p)
        chains = {}
        for name in cands:
            chains[name] = build_chain(p, cfg, CONFIGS[name], shared)
            p.out.zero_()
            for fn in chains[name]:
                fn()
            torch.cuda.synchronize()
            ck = check(p.out.clone(), p.ref_out, tol=5e-2, label=name)
            if not ck["ok"]:
                print(f"  {name} FAILS CORRECTNESS ({ck['rel_err']:.2e}) — dropped", flush=True)
                chains.pop(name)

        rounds: dict[str, list[float]] = {k: [] for k in chains}
        for r in range(args.rounds):
            for name, chain in chains.items():
                t = bench_chain(chain, warmup=5, rep=args.rep)
                rounds[name].append(t.p50_ms)
            print(f"  round {r}: " + "  ".join(
                f"{n}={rounds[n][-1]:.4f}" for n in chains), flush=True)

        stats = {}
        for name, xs in rounds.items():
            xs_s = sorted(xs)
            stats[name] = {
                "median": statistics.median(xs),
                "min": xs_s[0], "max": xs_s[-1],
                "spread_pct": (xs_s[-1] - xs_s[0]) / statistics.median(xs) * 100,
                "rounds": xs,
            }

        order = sorted(stats, key=lambda k: stats[k]["median"])
        best, runner = order[0], order[1] if len(order) > 1 else None
        verdict = {}
        if runner:
            gap = stats[runner]["median"] - stats[best]["median"]
            # separated only if the gap exceeds the larger of the two round-to-round spreads
            noise = max(stats[best]["max"] - stats[best]["min"],
                        stats[runner]["max"] - stats[runner]["min"])
            separated = gap > noise
            tied = [n for n in order
                    if stats[n]["median"] - stats[best]["median"] <= noise]
            verdict = {"best": best, "runner_up": runner, "gap_ms": gap,
                       "noise_ms": noise, "separated": separated, "tied_set": tied}
            print(f"  --> {best} median {stats[best]['median']:.4f} ms; "
                  f"gap to {runner} = {gap:.4f} ms, round noise = {noise:.4f} ms -> "
                  f"{'SEPARATED' if separated else 'TIED with ' + ', '.join(tied)}", flush=True)
        base = stats.get("A_all_unfused", {}).get("median")
        if base:
            print(f"      vs all-unfused: {base / stats[best]['median']:.4f}x", flush=True)
        out[regime] = {"stats": stats, "verdict": verdict,
                       "speedup_vs_unfused": (base / stats[best]["median"]) if base else None}
        del p
        torch.cuda.empty_cache()

    record(args.out, {
        "id": args.out,
        "protocol": f"{args.rounds} interleaved rounds, rep={args.rep} per round; "
                    "per-config median across rounds; a winner is declared only if its gap "
                    "to the runner-up exceeds the round-to-round spread of both",
        "regimes": out,
    })
    print(f"\nwrote results/{args.out}.json")


if __name__ == "__main__":
    main()
