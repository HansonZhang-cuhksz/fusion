"""`report_glm52_h200/chain_gain_vs_T.png` -- the nine chains, re-measured.

THIS IS THE CANONICAL FIGURE. It supersedes the campaign version, which is preserved
beside it as `chain_gain_vs_T_campaign.png` (built by `make_chain_gain_vs_T_h200.py`)
because the two are independent measurements that agree to <= 0.3 % and the older one is
therefore a replication, not a mistake. Same nine fusion
chains, same question -- whole-layer speedup against the all-unfused layer, across all
eleven regimes -- but measured by the uncertainty-controlled harness
(`glm52_h200/bench/bench_layer_certain.py`, 2026-08-13) rather than by the campaign.

THREE THINGS THE OLD FIGURE NEEDED AND THIS ONE DOES NOT.

1. **No broken y-axis.** The old figure split its axis between 1.07 and 1.26 because
   `decode_bs1` sat at 1.289x while every other regime landed inside 1.000-1.051. That
   outlier was an artifact: the campaign's `decode_bs1` all-unfused baseline -- the
   denominator of every point in that column -- carried a cold-start excursion, round 0
   running 2.5x the median of the other seven. Re-measured on one card in one session with
   the baseline rotated through every slot, T=1 comes in at 1.069x and the whole field fits
   on one linear axis. The break is gone because the thing that forced it was not real.

2. **No modelled noise band.** The old figure shaded a band whose half-width was the
   per-regime p90 of |run1 - run2| -- a pass-to-pass repeatability proxy, explicitly NOT a
   significance band. Here every chain carries its own percentile-bootstrap 95 % CI from
   its own per-block ratios, drawn as a ribbon around its own curve. Where a ribbon crosses
   1.000x that chain is not distinguishable from doing nothing, per chain, per regime.

3. **No "tied with doing nothing" state.** The campaign had the unfused layer inside the
   tie set at three regimes, so at T = 1024, 2048 and 8192 it could not distinguish its own
   winner from not fusing at all. Here the unfused layer is in **no** regime's tie set.

WHAT DID NOT CHANGE, WHICH IS THE POINT. At T = 4..1024 every configuration including the
baseline reproduces the campaign's absolute layer time to <= 0.3 %. The two measurements
agree on the physics; they disagree on how much of it was resolvable.

READ THE CAVEATS ON THE FIGURE. Clock locking needs root and was unavailable, so the card
ran unlocked; at T >= 512 the drift gate discarded 39-88 % of blocks and the CIs widen by
3-10x. `R_f1_f10_f11ab` is a single point at T=1: it and the other three `#11a`-bearing
chains fail the fp32 reference at every other regime (rel_err 0.06-0.61 against a 5e-2 bar).

Run:  /home/zhangshuhan/my-envs/fusion/bin/python glm52/make_chain_gain_vs_T_certain_h200.py
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

OUT = ROOT / "report_glm52_h200" / "certain"
SRC = OUT / "layer_certain_per_regime.csv"
VERD = OUT / "layer_certain_verdicts.csv"

PREFILL_FROM = 1448.0
XLIM = (0.80, 1.15e4)
TVALS = [1, 2, 4, 8, 16, 32, 256, 512, 1024, 2048, 8192]

#: The five chains the campaign named as per-regime winners, plus the four its tie sets
#: contained. Kept identical to `chain_gain_vs_T.png` so the two figures are comparable --
#: and it still covers every winner of THIS run (J, I, H, C are all here).
WINNERS = [
    ("J_greedy_all",   "#1+#6+#9", "#2a78d6", "o"),
    ("H_f3_f10",       "#3+#10",   "#4b3fbb", "s"),
    ("D_f6",           "#6",       "#1a7f52", "D"),
    ("B_f3",           "#3",       "#d6478f", "^"),
    ("C_f1",           "#1",       "#b3760a", "v"),
]
CO_LEADERS = [
    ("I_f3_f9",        "#3+#9",    "#3d3c39", (0, (5, 2)), "o"),
    ("K_f3_f8",        "#3+#8",    "#63625d", (0, (2, 2)), "s"),
    ("E_f10",          "#10",      "#8a8882", (0, (7, 2, 1, 2)), "^"),
]
SINGLE = ("R_f1_f10_f11ab", "#1+#10+#11a+#11b", "#2a78d6", "d")


def load() -> tuple[dict, dict]:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC} -- run glm52/make_certain_report_h200.py first")
    series: dict = {}
    for r in csv.DictReader(SRC.open()):
        if not r["speedup_wall"]:
            continue
        series.setdefault(r["config_id"], {})[int(r["T"])] = (
            float(r["speedup_wall"]), float(r["ci_lo"]), float(r["ci_hi"]),
            r["regime_winner"] == "1",
        )
    verd = {int(r["T"]): r for r in csv.DictReader(VERD.open())}
    return series, verd


def dress(ax, c, *, xlabels: bool) -> None:
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
    ax.xaxis.set_major_locator(FixedLocator(TVALS))
    if xlabels:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    else:
        ax.set_xticklabels([])
    ax.tick_params(labelsize=8.5, colors=c["secondary"], length=3, width=0.8)


def spread(targets: list[float], gap: float, lo: float, hi: float) -> list[float]:
    """Push labels apart just enough to stop them overlapping, keeping their order."""
    ys = sorted(range(len(targets)), key=lambda i: targets[i])
    out = list(targets)
    for k, i in enumerate(ys):
        if k and out[i] - out[ys[k - 1]] < gap:
            out[i] = out[ys[k - 1]] + gap
    over = out[ys[-1]] - hi
    if over > 0:
        for i in ys:
            out[i] -= over
    for k, i in enumerate(ys):
        if k and out[i] - out[ys[k - 1]] < gap:
            out[i] = out[ys[k - 1]] + gap
    return [min(max(v, lo), hi) for v in out]


def main() -> None:
    series, verd = load()
    c = THEME["light"]
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    fig = plt.figure(figsize=(14.2, 12.0))
    fig.patch.set_facecolor(c["surface"])
    # hspace is generous between the delta panel and the table: the shared x-label sits in
    # that gap and collided with the table header at the previous spacing.
    gs = fig.add_gridspec(3, 1, height_ratios=[4.4, 1.5, 3.4], left=0.065, right=0.845,
                          top=0.828, bottom=0.150, hspace=0.30)
    ax = fig.add_subplot(gs[0])
    axd = fig.add_subplot(gs[1])
    axt = fig.add_subplot(gs[2])

    dress(ax, c, xlabels=False)
    dress(axd, c, xlabels=True)
    ax.set_ylim(0.905, 1.080)
    ax.yaxis.set_major_locator(FixedLocator([0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04,
                                             1.06, 1.08]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}x"))
    ax.axhline(1.0, color="#bcbab3", linewidth=1.2, zorder=3)
    ax.text(60.0, 1.0022, "1.000x  =  the all-unfused layer", fontsize=8.5,
            color=c["secondary"], ha="center", zorder=8,
            bbox=dict(boxstyle="square,pad=0.16", fc=c["surface"], ec="none", alpha=0.9))

    ends: list[tuple[float, str, str]] = []
    for cfg, label, col, mk in WINNERS:
        d = series.get(cfg, {})
        ts = sorted(d)
        ax.fill_between(ts, [d[t][1] for t in ts], [d[t][2] for t in ts],
                        color=col, alpha=0.16, linewidth=0, zorder=4)
        ax.plot(ts, [d[t][0] for t in ts], color=col, linewidth=2.0, zorder=6,
                solid_capstyle="round")
        for t in ts:
            ax.plot([t], [d[t][0]], marker=mk, markersize=5.4, color=col,
                    markeredgecolor=c["surface"], markeredgewidth=0.9, zorder=7,
                    linestyle="none")
        ends.append((d[max(ts)][0], label, col))

    for cfg, label, col, dash, mk in CO_LEADERS:
        d = series.get(cfg, {})
        ts = sorted(d)
        ax.fill_between(ts, [d[t][1] for t in ts], [d[t][2] for t in ts],
                        color=col, alpha=0.10, linewidth=0, zorder=2)
        ax.plot(ts, [d[t][0] for t in ts], color=col, linewidth=1.3, linestyle=dash,
                zorder=5)
        for t in ts:
            ax.plot([t], [d[t][0]], marker=mk, markersize=4.2, markerfacecolor=c["surface"],
                    markeredgecolor=col, markeredgewidth=1.1, zorder=5, linestyle="none")
        ends.append((d[max(ts)][0], label, col))

    scfg, slab, scol, smk = SINGLE
    if scfg in series:
        t1, (v, lo, hi, _) = 1, series[scfg][1]
        ax.plot([t1], [v], marker=smk, markersize=9.0, markerfacecolor=c["surface"],
                markeredgecolor=scol, markeredgewidth=2.0, zorder=8, linestyle="none")
        # Parked in the empty lower-left of the panel: every curve lives above 1.00 out to
        # T=1024, so this quadrant is the only region a three-line note does not cover data.
        ax.annotate(f"{slab}  {v:.4f}x at T=1 only\n"
                    f"single point — this and the three other #11a-bearing chains\n"
                    f"fail the fp32 reference at every other regime",
                    xy=(t1, v), xytext=(1.35, 0.9365), fontsize=8.0, color=c["secondary"],
                    ha="left", va="center", linespacing=1.4, zorder=9,
                    arrowprops=dict(arrowstyle="-", color=scol, linewidth=0.9,
                                    shrinkA=4, shrinkB=6, alpha=0.75,
                                    connectionstyle="angle,angleA=0,angleB=90,rad=6"))

    # End-of-line labels, de-collided.
    ends.sort()
    ys = spread([e[0] for e in ends], 0.0068, 0.906, 1.079)
    for (val, label, col), y in zip(ends, ys):
        ax.annotate(f"{label}  {val:.3f}x", xy=(8192, val), xytext=(1.028, y),
                    textcoords=("axes fraction", "data"), fontsize=8.5, color=col,
                    va="center", ha="left", annotation_clip=False, zorder=9,
                    arrowprops=dict(arrowstyle="-", color=col, linewidth=0.8, alpha=0.5,
                                    shrinkA=2, shrinkB=0))

    # ---- delta panel -------------------------------------------------------------------
    axd.set_ylim(-0.075, 0.075)
    axd.yaxis.set_major_locator(FixedLocator([-0.05, 0.0, 0.05]))
    axd.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.2f}"))
    axd.axhline(0.0, color="#bcbab3", linewidth=1.2, zorder=3)
    delta = {}
    for r in csv.DictReader(SRC.open()):
        if r["delta_vs_campaign"] and r["config_id"] in series:
            delta.setdefault(r["config_id"], {})[int(r["T"])] = float(r["delta_vs_campaign"])
    # T=1 is DROPPED from this panel, not clipped. The campaign's decode_bs1 all-unfused
    # baseline carried a cold-start excursion, so its whole T=1 column is not a valid
    # comparator; the deltas there run to -0.22 and would either squash the panel by 3x or
    # run off the bottom pretending to be data. The exclusion is stated on the panel.
    for cfg, label, col, mk in WINNERS:
        ts = [t for t in sorted(delta.get(cfg, {})) if t > 1]
        axd.plot(ts, [delta[cfg][t] for t in ts], color=col, linewidth=1.6, marker=mk,
                 markersize=4.0, zorder=6)
    for cfg, label, col, dash, mk in CO_LEADERS:
        ts = [t for t in sorted(delta.get(cfg, {})) if t > 1]
        axd.plot(ts, [delta[cfg][t] for t in ts], color=col, linewidth=1.0, linestyle=dash,
                 zorder=5)
    axd.set_ylabel("this run\nminus campaign", fontsize=8.5, color=c["secondary"],
                   linespacing=1.4)
    axd.axvspan(XLIM[0], 1.42, color=c["harm"], zorder=1)
    axd.text(1.0, -0.062, "T=1 excluded:\ncampaign column\nnot a valid\ncomparator",
             fontsize=6.8, color=c["secondary"], ha="center", va="bottom", linespacing=1.3,
             zorder=9)
    axd.text(2.6, 0.055, "agreement is <= 0.3 % of layer time from T=4 to T=1024; the two "
             "prefill regimes diverge because 76-88 % of their blocks were discarded here",
             fontsize=8.0, color=c["secondary"], ha="left", va="center", zorder=9)
    axd.set_xlabel("total tokens T per call   (decode: batch size at kv 4096   ·   "
                   "prefill: sequence length, shaded)", fontsize=9.5, color=c["secondary"])
    ax.set_ylabel("whole-layer speedup vs the all-unfused layer\n"
                  "(ribbon = that chain's own 95 % bootstrap CI)",
                  fontsize=9.5, color=c["secondary"], linespacing=1.5)

    # ---- table --------------------------------------------------------------------------
    axt.set_facecolor(c["surface"])
    axt.set_xlim(0, 1)
    axt.set_ylim(0, 1)
    axt.axis("off")
    # Value columns start past 0.21: `#1+#10+#11a+#11b` is 16 characters and collided with
    # the T=1 column when they started at 0.152.
    xs = [0.215 + 0.0725 * i for i in range(11)]
    axt.text(0.005, 0.955, "chain", fontsize=8.5, color=c["primary"], ha="left", va="bottom")
    for x, t in zip(xs, TVALS):
        axt.text(x, 0.955, f"T={t}", fontsize=8.2, color=c["primary"], ha="right",
                 va="bottom")
    axt.plot([0.0, 1.0], [0.930] * 2, color=c["rule"], linewidth=0.9, clip_on=False)

    allc = ([(a, b, col, True) for a, b, col, _ in WINNERS]
            + [(a, b, col, False) for a, b, col, _, _ in CO_LEADERS]
            + [(scfg, slab, scol, False)])
    for i, (cfg, label, col, bold) in enumerate(allc):
        y = 0.865 - 0.0755 * i
        axt.plot([0.012], [y + 0.016], marker="s", markersize=5.0, color=col,
                 linestyle="none", clip_on=False)
        axt.text(0.030, y, label, fontsize=8.5,
                 color=c["primary"] if bold else c["secondary"], ha="left", va="bottom")
        d = series.get(cfg, {})
        for x, t in zip(xs, TVALS):
            if t not in d:
                axt.text(x, y, "—", fontsize=8.2, color=c["grid"], ha="right", va="bottom")
                continue
            v, lo, hi, win = d[t]
            tied = lo <= 1.0 <= hi
            ink = c["grid"] if tied else (c["primary"] if bold else c["secondary"])
            axt.text(x, y, f"{v:.4f}" + ("◀" if win else ""), fontsize=8.2, color=ink,
                     ha="right", va="bottom",
                     fontweight="bold" if win else "normal")
    ylast = 0.865 - 0.0755 * len(allc)
    axt.plot([0.0, 1.0], [ylast + 0.052] * 2, color=c["rule"], linewidth=0.7, clip_on=False)
    axt.text(0.030, ylast - 0.004, "blocks kept", fontsize=8.2, color=c["secondary"],
             ha="left", va="bottom")
    for x, t in zip(xs, TVALS):
        v = verd[t]
        axt.text(x, ylast - 0.004,
                 f"{v['blocks_kept']}/{int(v['blocks_kept']) + int(v['blocks_dropped'])}",
                 fontsize=8.0, color=c["secondary"], ha="right", va="bottom")
    axt.text(0.030, ylast - 0.078, "◀ = regime winner    grey = CI includes 1.000x, i.e. "
             "not distinguishable from the unfused layer at that regime", fontsize=8.2,
             color=c["secondary"], ha="left", va="bottom")

    n_sep = sum(int(v["separated_wall"]) for v in verd.values())
    n_null = sum(int(v["unfused_in_tie_set"]) for v in verd.values())
    fig.text(0.065, 0.968, "GLM-5.2 MoE decoder layer on NVIDIA H200 — nine fusion chains, "
             "uncertainty-controlled re-measurement", fontsize=15.5, color=c["primary"],
             ha="left", va="top")
    sub = (
        "One card, one 7.1-minute session, all eleven regimes, configurations frozen from "
        "the campaign so nothing could be tuned unequally. Every chain carries its own "
        f"bootstrap 95 % CI from its own per-block ratios. {n_sep} of 11 regimes now resolve "
        f"a unique winner against the campaign's 3, and the unfused layer sits in {n_null} "
        "tie sets against the campaign's 3 — so every regime here has SOME chain that beats "
        "not fusing. The lower panel is the difference from the campaign: at T = 4..1024 the "
        "two agree to <= 0.3 % on absolute layer time, so what changed is resolution, not "
        "physics."
    )
    fig.text(0.065, 0.938, "\n".join(textwrap.wrap(sub, 168)),
             fontsize=9.4, color=c["secondary"], ha="left", va="top", linespacing=1.5)

    notes = [
        "Speedup = the all-unfused layer's time divided by this chain's, both measured "
        "inside the same block; blocks rotate every configuration through every slot, so no "
        "configuration has a position advantage. Median over blocks, percentile-bootstrap "
        "95 % CI, 2000 resamples. A chain beats the unfused layer only where its CI excludes "
        "1.000x; two chains are tied where their CIs overlap.",
        "CLOCKS WERE NOT LOCKED — nvidia-smi -lgc needs root and it was unavailable, so the "
        "card ran unlocked with SwPowerCap active under load. The per-block drift gate is "
        "then the only defence: it discarded 39-88 % of blocks at T >= 512, and the CIs "
        "there are 3-10x wider than at mid-decode (+-0.43 % at prefill_t8192 against "
        "+-0.02 % at decode_bs16). Differences below a chain's own ribbon are not results.",
        "NOT comparable with gain_vs_T.png, whose denominator is a two- or three-kernel "
        "unfused chain rather than the whole layer. Directly comparable with "
        "chain_gain_vs_T.png, which is the same nine chains measured by the campaign — that "
        "figure needed a broken y-axis for a decode_bs1 point at 1.289x that this run puts "
        "at 1.069x once the baseline's cold-start excursion is removed.",
        "source: report_glm52_h200/certain/layer_certain_per_regime.csv from "
        "results/h200/layer_certain.json  ·  glm52_h200/bench/bench_layer_certain.py on "
        "GPU-338e7fe0, 2026-08-13, 428 s  ·  18 configurations attempted per regime, 14 "
        "measured (the four #11a-bearing chains fail the fp32 reference outside decode_bs1) "
        "·  0 blocks measured-then-lost.",
    ]
    fig.text(0.065, 0.020, "\n".join(l for n in notes for l in textwrap.wrap(n, 196) or [""]),
             fontsize=7.7, color=c["secondary"], ha="left", va="bottom", linespacing=1.6)

    p = OUT.parent / "chain_gain_vs_T.png"
    fig.savefig(p, dpi=140, facecolor=c["surface"])
    print(f"wrote {p}  ({len(allc)} chains, 11 regimes, {n_sep} separated, "
          f"{n_null} tied-with-unfused)")


if __name__ == "__main__":
    main()
