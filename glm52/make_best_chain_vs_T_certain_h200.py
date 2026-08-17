"""`report_glm52_h200/best_chain_vs_T.png` -- one curve: the best fusion CHAIN per regime.

THIS IS THE CANONICAL FIGURE, rebuilt from the 2026-08-13 uncertainty-controlled re-run
(`results/h200/layer_certain.json`). The campaign version is preserved beside it as
`best_chain_vs_T_campaign.png`; the two agree to <= 0.3 % on absolute layer time at
T = 4..1024, so the older one is a replication rather than a mistake.

WHAT THE CAMPAIGN VERSION NEEDED AND THIS ONE DOES NOT.

  * **No broken y-axis.** The old figure split its axis between 1.07 and 1.26 for a
    `decode_bs1` point at 1.289x. That point was an artifact: the campaign's T=1 all-unfused
    baseline -- the denominator of the whole column -- carried a cold-start excursion, round 0
    running 2.5x the median of the other seven. Re-measured, T=1 is 1.069x and the eleven
    points span 1.004-1.069, which is one linear axis.
  * **No "no gain resolved" state.** The campaign had `A_all_unfused` inside the tie set at
    T = 1024, 2048 and 8192 -- three regimes where it could not distinguish its own winner
    from not fusing at all. Here the unfused layer is in NO regime's tie set, so every point
    on this curve is a gain the run can actually defend.
  * **Real confidence intervals.** The campaign drew a bar spanning the tie set's spread, a
    proxy. Each point here carries the winner's own percentile-bootstrap 95 % CI, and the
    tie set is the set of chains whose CIs overlap the winner's.

TWO MARKER STATES REMAIN, not three:
    filled  -- unique winner: no other chain's CI overlaps it (6 of 11 regimes)
    hollow  -- winner not resolved: 2-4 chains tied, but all of them beat unfused

READ THE CAVEATS. Clocks were not locked (`nvidia-smi -lgc` needs root), so at T >= 512 the
drift gate discarded 39-88 % of blocks and the CIs widen 3-10x. `prefill_t2048` kept only 21
of 168 blocks and is the weakest point on the curve.

Run:  /home/zhangshuhan/my-envs/fusion/bin/python glm52/make_best_chain_vs_T_certain_h200.py
      (from the repo root; the default python3 on this box has no matplotlib)
"""
from __future__ import annotations

import csv
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glm52.plot_f8_sweep import THEME  # noqa: E402

OUT = ROOT / "report_glm52_h200"
SRC = OUT / "certain" / "layer_certain_per_regime.csv"
VERD = OUT / "certain" / "layer_certain_verdicts.csv"

S1 = "#2a78d6"
PREFILL_FROM = 1448.0
XLIM = (0.80, 1.15e4)
TVALS = [1, 2, 4, 8, 16, 32, 256, 512, 1024, 2048, 8192]


def load() -> list[dict]:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC} -- run glm52/make_certain_report_h200.py first")
    rows = list(csv.DictReader(SRC.open()))
    verd = {int(r["T"]): r for r in csv.DictReader(VERD.open())}
    out = []
    for t in TVALS:
        v = verd[t]
        win = v["winner_wall"]
        w = next(r for r in rows if int(r["T"]) == t and r["config_id"] == win)
        tied = [r for r in rows if int(r["T"]) == t and r["tied_with_winner"] == "1"
                and r["speedup_wall"]]
        sp = [float(r["speedup_wall"]) for r in tied] or [float(w["speedup_wall"])]
        out.append(dict(
            T=t, regime=v["regime"], config=win, chain=w["fusion_set"],
            gain=float(w["speedup_wall"]),
            lo=float(w["ci_lo"]), hi=float(w["ci_hi"]),
            n_tie=int(v["n_tied_wall"]), tie_lo=min(sp), tie_hi=max(sp),
            sep=v["separated_wall"] == "1",
            null=v["unfused_in_tie_set"] == "1",
            kept=int(v["blocks_kept"]),
            run=int(v["blocks_kept"]) + int(v["blocks_dropped"]),
            campaign=w["campaign_speedup"],
        ))
    return out


def main() -> None:
    rows = load()
    c = THEME["light"]
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    fig = plt.figure(figsize=(13.6, 10.4))
    fig.patch.set_facecolor(c["surface"])
    gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 2.9], left=0.075, right=0.982,
                          top=0.822, bottom=0.170, hspace=0.30)
    ax = fig.add_subplot(gs[0])
    axt = fig.add_subplot(gs[1])

    ax.set_facecolor(c["surface"])
    ax.set_axisbelow(True)
    ax.axvspan(PREFILL_FROM, XLIM[1], color="#eef2f8", zorder=0)
    ax.grid(True, which="major", color=c["grid"], linewidth=0.7, zorder=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(c["grid"])
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(0.9975, 1.0830)
    ax.xaxis.set_major_locator(FixedLocator(TVALS))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.yaxis.set_major_locator(FixedLocator([1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06,
                                             1.07]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}x"))
    ax.tick_params(labelsize=8.5, colors=c["secondary"], length=3, width=0.8)
    ax.axhline(1.0, color="#bcbab3", linewidth=1.2, zorder=3)
    ax.text(70.0, 1.0012, "1.000x  =  the all-unfused layer", fontsize=8.5,
            color=c["secondary"], ha="center", va="bottom", zorder=8,
            bbox=dict(boxstyle="square,pad=0.16", fc=c["surface"], ec="none", alpha=0.9))

    xs = [r["T"] for r in rows]
    ax.plot(xs, [r["gain"] for r in rows], color=S1, linewidth=2.0, zorder=5,
            solid_capstyle="round")
    ax.fill_between(xs, [r["lo"] for r in rows], [r["hi"] for r in rows], color=S1,
                    alpha=0.20, linewidth=0, zorder=4)
    for r in rows:
        # Tie-set span: every chain whose CI overlaps the winner's. Drawn behind the marker.
        if r["n_tie"] > 1:
            ax.plot([r["T"]] * 2, [r["tie_lo"], r["tie_hi"]], color=S1, linewidth=7.0,
                    alpha=0.20, solid_capstyle="butt", zorder=3)
        if r["sep"]:
            ax.plot([r["T"]], [r["gain"]], marker="o", markersize=8.5, color=S1,
                    markeredgecolor=c["surface"], markeredgewidth=1.4, zorder=7,
                    linestyle="none")
        else:
            ax.plot([r["T"]], [r["gain"]], marker="o", markersize=8.5,
                    markerfacecolor=c["surface"], markeredgecolor=S1, markeredgewidth=2.0,
                    zorder=7, linestyle="none")
        # T=512's label collides with T=256's when both sit above the curve; the region
        # below the ascending 256->512 segment is empty, so that one goes underneath.
        below = r["T"] == 512
        ax.annotate(f"{r['chain'].replace(' ', '')}\n{r['gain']:.4f}x",
                    xy=(r["T"], r["gain"]),
                    xytext=(0, -30 if below else 11), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.3, color=c["secondary"],
                    linespacing=1.35, zorder=8,
                    bbox=dict(boxstyle="square,pad=0.16", fc=c["surface"], ec="none",
                              alpha=0.88))

    ax.set_xlabel("total tokens T per call   (decode: batch size at kv 4096   ·   "
                  "prefill: sequence length, shaded)", fontsize=9.5, color=c["secondary"])
    ax.set_ylabel("whole-layer speedup vs the all-unfused layer\n"
                  "(ribbon = the winner's own 95 % bootstrap CI)",
                  fontsize=9.5, color=c["secondary"], linespacing=1.5)

    key = [("unique winner — no other chain's CI overlaps it", True),
           ("winner not resolved — 2-4 chains tied, all of them still beat unfused", False)]
    for i, (text, filled) in enumerate(key):
        y = 0.115 - 0.062 * i
        ax.plot([0.030], [y], marker="o", markersize=8.0,
                color=S1 if filled else c["surface"],
                markerfacecolor=S1 if filled else c["surface"], markeredgecolor=S1,
                markeredgewidth=1.4 if filled else 2.0, transform=ax.transAxes,
                clip_on=False, linestyle="none", zorder=9)
        ax.text(0.049, y, text, fontsize=8.4, color=c["secondary"], va="center", ha="left",
                transform=ax.transAxes, zorder=9,
                bbox=dict(boxstyle="square,pad=0.18", fc=c["surface"], ec="none",
                          alpha=0.88))

    # ---- table ---------------------------------------------------------------------------
    axt.set_facecolor(c["surface"])
    axt.set_xlim(0, 1)
    axt.set_ylim(0, 1)
    axt.axis("off")
    cols = [(0.045, "right", "T"), (0.062, "left", "regime"), (0.190, "left", "winning chain"),
            (0.360, "left", "config"), (0.545, "right", "layer gain"),
            (0.560, "left", "95 % CI"), (0.735, "left", "tie set"),
            (0.842, "left", "blocks kept"), (0.985, "right", "campaign")]
    head = 0.945
    for x, ha, name in cols:
        axt.text(x, head, name, fontsize=8.5, color=c["primary"], ha=ha, va="bottom",
                 transform=axt.transAxes)
    axt.plot([0.0, 1.0], [head - 0.030] * 2, color=c["rule"], linewidth=0.9,
             transform=axt.transAxes, clip_on=False)
    for i, r in enumerate(rows):
        y = head - 0.082 - 0.0775 * i
        tie = "unique" if r["n_tie"] == 1 else f"{r['n_tie']} chains tied"
        cells = [(0.045, "right", str(r["T"])), (0.062, "left", r["regime"]),
                 (0.190, "left", r["chain"]), (0.360, "left", r["config"]),
                 (0.545, "right", f"{r['gain']:.4f}x"),
                 (0.560, "left", f"[{r['lo']:.4f}, {r['hi']:.4f}]"),
                 (0.735, "left", tie),
                 (0.842, "left", f"{r['kept']}/{r['run']}"),
                 (0.985, "right", r["campaign"] or "—")]
        for x, ha, text in cells:
            axt.text(x, y, text, fontsize=8.4, color=c["secondary"], ha=ha, va="bottom",
                     transform=axt.transAxes)
        axt.plot([0.014], [y + 0.018], marker="o", markersize=5.0,
                 color=S1 if r["sep"] else c["surface"], markerfacecolor=S1 if r["sep"]
                 else c["surface"], markeredgecolor=S1, markeredgewidth=1.0,
                 transform=axt.transAxes, clip_on=False, linestyle="none")

    n_sep = sum(r["sep"] for r in rows)
    n_null = sum(r["null"] for r in rows)
    fig.text(0.075, 0.972, "GLM-5.2 MoE decoder layer on NVIDIA H200 — best fusion chain "
             "per regime", fontsize=16, color=c["primary"], ha="left", va="top")
    sub = (
        "One curve. Each point is the fastest validated COMBINATION of fusions for that "
        f"regime, timed as a whole layer against the all-unfused layer. {n_sep} of 11 regimes "
        f"resolve a unique winner (the campaign resolved 3), and the unfused layer sits in "
        f"{n_null} tie sets (the campaign: 3) — so unlike the campaign, every point here is a "
        "gain the run can defend. Four distinct chains win across the eleven regimes."
    )
    fig.text(0.075, 0.936, "\n".join(textwrap.wrap(sub, 158)), fontsize=9.5,
             color=c["secondary"], ha="left", va="top", linespacing=1.5)

    notes = [
        "Layer gain = the all-unfused layer's time divided by this chain's, both measured "
        "inside the same block; blocks rotate every configuration through every slot so no "
        "configuration has a position advantage. Median over blocks, percentile-bootstrap "
        "95 % CI, 2000 resamples. The tie set is every chain whose CI overlaps the winner's. "
        "NOT the per-fusion kernel gain in gain_vs_T.png, whose denominator is a two-kernel "
        "chain rather than the whole layer.",
        "CLOCKS WERE NOT LOCKED — nvidia-smi -lgc needs root and it was unavailable. The "
        "per-block drift gate is then the only defence: it discarded 39-88 % of blocks at "
        "T >= 512. prefill_t2048 kept only 21 of 168 blocks and is the least certain point "
        "on this curve; decode_bs8 through decode_bs512 are the most certain.",
        "The campaign column is report_glm52_h200/layer_optimal_per_regime.csv for the SAME "
        "chain, not that regime's campaign winner. decode_bs1 is the one regime where the two "
        "disagree materially (1.0693x here against 1.2886x): the campaign's T=1 all-unfused "
        "baseline carried a cold-start excursion, and the campaign itself declared that "
        "regime TIED with nothing resolving.",
        "source: report_glm52_h200/certain/layer_certain_per_regime.csv from "
        "results/h200/layer_certain.json  ·  glm52_h200/bench/bench_layer_certain.py on "
        "GPU-338e7fe0, 2026-08-13, 428 s, configurations frozen from the campaign  ·  "
        "campaign figure preserved as best_chain_vs_T_campaign.png.",
    ]
    fig.text(0.075, 0.018, "\n".join(l for n in notes for l in textwrap.wrap(n, 186) or [""]),
             fontsize=7.7, color=c["secondary"], ha="left", va="bottom", linespacing=1.6)

    p = OUT / "best_chain_vs_T.png"
    fig.savefig(p, dpi=140, facecolor=c["surface"])
    print(f"wrote {p}  ({len(rows)} regimes, "
          f"{len({r['config'] for r in rows})} distinct winning chains, {n_sep} separated, "
          f"{n_null} tied-with-unfused)")


if __name__ == "__main__":
    main()
