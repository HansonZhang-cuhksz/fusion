"""`report_glm52_h200/best_chain_vs_T.png` -- one curve: the best fusion CHAIN per regime.

Companion to `gain_vs_T.png`, and NOT comparable with it. That figure plots one line per
individual fusion, each against its own unfused two-or-three-kernel chain. This one plots a
single line: for every regime, the whole-layer time of the fastest validated COMBINATION of
fusions divided by the whole-layer time of the all-unfused layer. The winning combination is
free to change from regime to regime, and it does -- five different chains win across the
eleven regimes. The denominator is the entire decoder layer, so the numbers are much smaller
than the per-fusion ones: a 2.2x win on a kernel that is 3 % of the layer is a 1.02x layer.

WHAT THE MARKERS MEAN, and why a plain line would misreport this data. The source records a
tie set per pass under the LOG-11 S3 rule -- a winner is declared only where its gap to the
runner-up exceeds the round-to-round spread (max - min over 8 rounds) of BOTH. Taking the
union over the two passes:

  * unique winner (3 regimes)          filled marker; one chain, resolved
  * chain not resolved (5 regimes)     hollow blue marker; SOME chain beats unfused, but
                                       2-4 of them are tied for which
  * no gain resolved (3 regimes)       hollow grey marker; `A_all_unfused` is ITSELF in the
                                       tie set, so the layer gain is not distinguishable
                                       from doing nothing at all

The vertical bar under each point is the spread of the tie set. Where it crosses 1.0 the
"win" includes candidates that are slower than unfused. At T = 1024, 2048 and 8192 it does.

y is broken between 1.07 and 1.26: decode_bs1 wins 1.289x and every other regime lands
inside 1.000-1.051, so one linear axis would flatten ten of the eleven points into a line.

Run:  /home/zhangshuhan/my-envs/fusion/bin/python glm52/make_best_chain_vs_T_h200.py
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
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glm52.plot_f8_sweep import THEME  # noqa: E402

OUT = ROOT / "report_glm52_h200"
SRC = OUT / "layer_optimal_per_regime.csv"

T = {
    "decode_bs1": 1, "decode_bs2": 2, "decode_bs4": 4, "decode_bs8": 8,
    "decode_bs16": 16, "decode_bs32": 32, "decode_bs256": 256,
    "decode_bs512": 512, "decode_bs1024": 1024,
    "prefill_t2048": 2048, "prefill_t8192": 8192,
}
UNFUSED = "A_all_unfused"
PREFILL_FROM = 1448.0

S1 = "#2a78d6"                      # the curve, and every resolved-gain marker
MUTED = "#6f6e69"                   # the "tied with doing nothing" state

XLIM = (0.80, 1.15e4)
XTICKS = [1, 2, 4, 8, 16, 32, 256, 1024, 8192]
YLO = (0.9775, 1.0625)              # lower segment of the broken axis
YHI = (1.2640, 1.3080)              # upper segment: decode_bs1 only


def short_chain(fusion_set: str) -> str:
    """`#1+#6+#9 (greedy)` -> `#1+#6+#9`; `#3 + #10` -> `#3+#10`."""
    return fusion_set.split("(")[0].replace(" ", "").strip() or fusion_set


def load() -> list[dict]:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC} -- run glm52/make_layer_report_h200.py first")
    rows = list(csv.DictReader(SRC.open()))
    out = []
    for reg, t in sorted(T.items(), key=lambda kv: kv[1]):
        rr = [r for r in rows if r["regime"] == reg]
        if not rr:
            print(f"!! no rows for {reg}")
            continue
        scored = [r for r in rr if r["speedup_vs_unfused"].strip()]
        best = max(scored, key=lambda r: float(r["speedup_vs_unfused"]))
        tie = {r["config_id"] for r in rr
               if r["tied_with_best_run1"] == "1" or r["tied_with_best_run2"] == "1"}
        tie_sp = [float(r["speedup_vs_unfused"]) for r in scored if r["config_id"] in tie]
        null = UNFUSED in tie
        out.append(dict(
            regime=reg, T=t, config=best["config_id"], chain=best["fusion_set"],
            short=short_chain(best["fusion_set"]),
            gain=float(best["speedup_vs_unfused"]), ms=float(best["best_ms"]),
            n_tie=len(tie), tie_lo=min(tie_sp), tie_hi=max(tie_sp), null=null,
            n_cand=len(rr),
            state=("no gain resolved" if null else
                   "unique winner" if len(tie) == 1 else "chain not resolved"),
        ))
    return out


def dress(ax, c, ylim, yticks, *, xlabels: bool):
    ax.set_facecolor(c["surface"])
    ax.set_axisbelow(True)
    ax.axvspan(PREFILL_FROM, XLIM[1], color=c["harm"], zorder=0)
    ax.grid(True, which="major", color=c["grid"], linewidth=0.7, zorder=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(FixedLocator(XTICKS))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{int(v)}") if xlabels else NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}x"))
    ax.tick_params(labelsize=8.5, colors=c["secondary"], length=3, width=0.8)


def draw_points(ax, c, rows):
    """Tie-set bar, then the marker, styled by how well the winner is resolved."""
    for r in rows:
        col = MUTED if r["null"] else S1
        ax.plot([r["T"], r["T"]], [r["tie_lo"], r["tie_hi"]], color=col, linewidth=6.5,
                alpha=0.22, solid_capstyle="butt", zorder=4)
        if r["state"] == "unique winner":
            ax.plot([r["T"]], [r["gain"]], marker="o", markersize=8.5, color=S1,
                    markeredgecolor=c["surface"], markeredgewidth=1.4, zorder=7,
                    linestyle="none")
        else:
            ax.plot([r["T"]], [r["gain"]], marker="o", markersize=8.5,
                    markerfacecolor=c["surface"], markeredgecolor=col,
                    markeredgewidth=2.0, zorder=7, linestyle="none")


def main() -> None:
    rows = load()
    c = THEME["light"]
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    fig = plt.figure(figsize=(13.8, 10.6))
    fig.patch.set_facecolor(c["surface"])
    outer = fig.add_gridspec(2, 1, height_ratios=[3.95, 2.85], left=0.062, right=0.984,
                             top=0.855, bottom=0.170, hspace=0.215)
    inner = outer[0].subgridspec(2, 1, height_ratios=[1.0, 3.5], hspace=0.07)
    ax_hi = fig.add_subplot(inner[0])
    ax_lo = fig.add_subplot(inner[1])
    ax_tab = fig.add_subplot(outer[1])

    dress(ax_hi, c, YHI, [1.28], xlabels=False)
    dress(ax_lo, c, YLO, [0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06],
          xlabels=True)
    ax_hi.spines["bottom"].set_visible(False)
    ax_hi.tick_params(bottom=False)
    ax_lo.axhline(1.0, color=c["rule"], linewidth=1.1, linestyle=(0, (4, 3)), zorder=3)
    ax_lo.text(90.0, 0.9988, "1.000x  =  the all-unfused layer", fontsize=8.5,
               color=c["secondary"], va="top", ha="center", zorder=8,
               bbox=dict(boxstyle="square,pad=0.16", fc=c["surface"], ec="none",
                         alpha=0.85))

    # The single curve, drawn across the break in both segments.
    xs = [r["T"] for r in rows]
    ys = [r["gain"] for r in rows]
    for ax in (ax_hi, ax_lo):
        ax.plot(xs, ys, color=S1, linewidth=2.0, zorder=5, solid_capstyle="round")
        draw_points(ax, c, rows)

    # Break marks on the shared edge.
    kw = dict(transform=None, color=c["rule"], linewidth=1.1, clip_on=False, zorder=9)
    for ax, y_ax in ((ax_hi, 0.0), (ax_lo, 1.0)):
        for x_ax in (0.0, 1.0):
            ax.plot([x_ax - 0.008, x_ax + 0.008], [y_ax - 0.022, y_ax + 0.022],
                    **dict(kw, transform=ax.transAxes))

    for r in rows:                      # chain + gain above each point
        ax = ax_hi if r["gain"] > YLO[1] else ax_lo
        span = (YHI[1] - YHI[0]) if ax is ax_hi else (YLO[1] - YLO[0])
        ax.annotate(f"{r['short']}\n{r['gain']:.3f}x", xy=(r["T"], r["gain"]),
                    xytext=(0, 9), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color=c["secondary"], linespacing=1.35, zorder=8,
                    bbox=dict(boxstyle="square,pad=0.16", fc=c["surface"], ec="none",
                              alpha=0.85))
        del span

    ax_lo.set_xlabel("total tokens T per call   (decode: batch size at kv 4096   ·   "
                     "prefill: sequence length, shaded)", fontsize=9.5,
                     color=c["secondary"])
    ax_lo.set_ylabel("whole-layer speedup vs the all-unfused layer", fontsize=9.5,
                     color=c["secondary"])
    ax_hi.text(1.55, 1.3035, "decode", fontsize=8.5, color=c["secondary"], ha="left")
    ax_hi.text(1750, 1.3035, "prefill", fontsize=8.5, color=c["secondary"], ha="left")

    # Marker key -- three states, spelled out rather than left to the caption.
    key = [("unique winner", S1, True), ("chain not resolved (2-4 tied)", S1, False),
           ("no gain resolved (tied with doing nothing)", MUTED, False)]
    for i, (text, col, filled) in enumerate(key):
        y = 0.145 - 0.066 * i
        ax_lo.plot([0.038], [y], marker="o", markersize=8.0,
                   color=col if filled else c["surface"],
                   markerfacecolor=col if filled else c["surface"],
                   markeredgecolor=col, markeredgewidth=1.4 if filled else 2.0,
                   transform=ax_lo.transAxes, clip_on=False, linestyle="none", zorder=9)
        ax_lo.text(0.058, y, text, fontsize=8.5, color=c["secondary"], va="center",
                   ha="left", transform=ax_lo.transAxes, zorder=9,
                   bbox=dict(boxstyle="square,pad=0.18", fc=c["surface"], ec="none",
                             alpha=0.85))
    ax_lo.text(0.038, 0.212, "bar under each point = spread of that regime's tie set",
               fontsize=8.5, color=c["secondary"], va="center", ha="left",
               transform=ax_lo.transAxes, zorder=9,
               bbox=dict(boxstyle="square,pad=0.18", fc=c["surface"], ec="none",
                         alpha=0.85))

    # ---- the table -------------------------------------------------------------
    ax_tab.set_facecolor(c["surface"])
    ax_tab.set_xlim(0, 1)
    ax_tab.set_ylim(0, 1)
    ax_tab.axis("off")
    cols = [(0.052, "right", "T"), (0.068, "left", "regime"),
            (0.198, "left", "winning chain"), (0.408, "left", "config"),
            (0.570, "right", "layer gain"), (0.600, "left", "tie set"),
            (0.822, "left", "gain range over tie set")]
    head_y = 0.945
    for x, ha, name in cols:
        ax_tab.text(x, head_y, name, fontsize=8.5, color=c["primary"], ha=ha,
                    va="bottom", transform=ax_tab.transAxes)
    ax_tab.plot([0.0, 1.0], [head_y - 0.030] * 2, color=c["rule"], linewidth=0.9,
                transform=ax_tab.transAxes, clip_on=False, zorder=3)

    for i, r in enumerate(rows):
        y = head_y - 0.082 - 0.0782 * i
        ink = MUTED if r["null"] else c["secondary"]
        tie = ("unique" if r["n_tie"] == 1 else
               f"{r['n_tie']} chains tied" + (" — incl. do nothing" if r["null"] else ""))
        cells = [(0.052, "right", str(r["T"])), (0.068, "left", r["regime"]),
                 (0.198, "left", r["chain"]), (0.408, "left", r["config"]),
                 (0.570, "right", f"{r['gain']:.4f}x"), (0.600, "left", tie),
                 (0.822, "left", f"{r['tie_lo']:.4f}x – {r['tie_hi']:.4f}x")]
        for x, ha, text in cells:
            ax_tab.text(x, y, text, fontsize=8.5, color=ink, ha=ha, va="bottom",
                        transform=ax_tab.transAxes)
        ax_tab.plot([0.014], [y + 0.018], marker="s", markersize=4.5,
                    color=MUTED if r["null"] else S1, transform=ax_tab.transAxes,
                    clip_on=False, linestyle="none")

    n_uni = sum(1 for r in rows if r["state"] == "unique winner")
    n_null = sum(1 for r in rows if r["null"])
    fig.text(0.062, 0.972, "GLM-5.2 MoE decoder layer on NVIDIA H200 "
             "— best fusion chain per regime",
             fontsize=16, color=c["primary"], ha="left", va="top")
    fig.text(0.062, 0.936,
             "One curve. Each point is the fastest validated COMBINATION of fusions for "
             "that regime, timed as a whole layer against the all-unfused layer; the "
             "winning combination changes with T\n"
             f"(five distinct chains across eleven regimes). "
             f"Only {n_uni} of {len(rows)} regimes resolve a unique winner, and at "
             f"{n_null} the tie set contains the unfused layer itself — there, the gain is "
             "not distinguishable from zero.",
             fontsize=9.5, color=c["secondary"], ha="left", va="top", linespacing=1.5)
    notes = [
        "Whole-layer gain = min(pass1, pass2) of A_all_unfused ÷ min(pass1, pass2) of this "
        "chain, a ratio of medians across separately-timed candidates — NOT the per-fusion "
        "kernel gain in gain_vs_T.png, whose denominator is one two-kernel chain rather "
        "than the whole layer. The two figures are not comparable.",
        "Tie sets follow the LOG-11 S3 rule, union over both passes: a winner is declared "
        "only where its gap to the runner-up exceeds the round-to-round spread (max − min "
        "over 8 rounds) of both. The measurement host was multi-tenant — contended "
        "calibration, 42.19 us harness floor against a 9.08 us launch — so sub-percent "
        "gaps on it are noise.",
        "decode_bs1 drew from 18 candidate chains and every other regime from 14: the four "
        "chains containing #11a were only built there. Its co-leader R_f1_f10_f11ab "
        "(1.2856x) is one of those, and f11_publish.json finds #11a unmeasurable on this "
        "device, so the resolved winner is J_greedy_all. That regime's pass 1 also opens "
        "with a cold-start excursion (round 0 up to 2.4x the other seven), inflating its "
        "noise floor.",
        "source: report_glm52_h200/layer_optimal_per_regime.csv  ·  18 configurations × "
        "2 independent passes × 8 interleaved rounds × 15 reps",
    ]
    wrapped = "\n".join(line for n in notes
                        for line in textwrap.wrap(n, 188) or [""])
    fig.text(0.062, 0.018, wrapped, fontsize=7.8, color=c["secondary"], ha="left",
             va="bottom", linespacing=1.62)

    p = OUT / "best_chain_vs_T.png"
    fig.savefig(p, dpi=140, facecolor=c["surface"])
    print(f"wrote {p}  ({len(rows)} regimes, {len({r['config'] for r in rows})} distinct "
          f"winning chains, {n_uni} unique / {n_null} tied-with-unfused)")


if __name__ == "__main__":
    main()
