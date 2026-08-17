"""Two figures for `report_glm52_h200/report.md` -- C500 against H200, layer gain vs T.

FORM. Two jobs, two forms:

  fig 1 `report_c500_vs_h200_best.png` -- change-over-T for three series on ONE axis.
        y is "percent faster than the all-unfused layer" on a LOG scale, because the
        quantity spans 0.15 % to 6.93 % (46x) and a linear axis would flatten six of the
        seven C500 points onto the baseline. Percent-faster is the reader's unit; log is
        what makes the whole range legible at once. Never a second y-axis.
  fig 2 `report_c500_vs_h200_perchain.png` -- nine chains x two devices. Small multiples,
        shared linear y, because this panel's job is POLARITY (which fusions change sign
        between devices) and sign needs a linear axis through 1.000.

COLOUR. Two categorical slots, assigned to device identity and fixed across both figures:
H200 = slot 1 blue `#2a78d6`, C500 = slot 2 orange `#eb6834`. Validated, not eyeballed:

    node scripts/validate_palette.js "#2a78d6,#eb6834" --mode light --pairs all
    -> ALL CHECKS PASS. worst all-pairs CVD dE 24.7 (protan), normal-vision dE 33.6,
       both clear of the >=8 / >=15 floors; both slots >= 3:1 on the #fcfcfb surface.

The third line in fig 1 (H200 running C500's chosen chain) is NOT a third hue -- it is the
H200 hue dashed, because it is still H200. Identity stays two-valued.

GAPS ARE REAL. C500 never measured T = 2/4/8/16, and several (chain, regime) cells were
never run. Those break the line rather than interpolating across them, and the count is
stated on the figure.

Run:  /home/zhangshuhan/my-envs/fusion/bin/python glm52/make_c500_vs_h200_figs.py
      (from the repo root; the default python3 on this box has no matplotlib)
"""
from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report_glm52_h200"
C500 = ROOT / "report_glm52_c500" / "layer_optimal_per_regime.csv"
H200 = OUT / "certain" / "layer_certain_per_regime.csv"

# --- tokens ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
PRIMARY = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#8d8b85"
GRID = "#e4e3df"
H_COL = "#2a78d6"      # categorical slot 1 -- H200
C_COL = "#eb6834"      # categorical slot 2 -- C500

T = {"decode_bs1": 1, "decode_bs32": 32, "decode_bs256": 256, "decode_bs512": 512,
     "decode_bs1024": 1024, "prefill_t2048": 2048, "prefill_t8192": 8192}
REGS = sorted(T, key=lambda r: T[r])
COMMON = ["B_f3", "C_f1", "D_f6", "E_f10", "F_f8", "G_f9", "H_f3_f10", "I_f3_f9", "K_f3_f8"]
LAB = {"B_f3": "#3", "C_f1": "#1", "D_f6": "#6", "E_f10": "#10", "F_f8": "#8",
       "G_f9": "#9", "H_f3_f10": "#3+#10", "I_f3_f9": "#3+#9", "K_f3_f8": "#3+#8",
       "J_greedy_all": "#1+#6+#9"}
PREFILL_FROM = 1448.0


def load() -> tuple[dict, dict]:
    c = {(r["regime"], r["config_id"]): float(r["speedup_vs_unfused"])
         for r in csv.DictReader(C500.open()) if r["speedup_vs_unfused"].strip()}
    h = {(r["regime"], r["config_id"]): float(r["speedup_wall"])
         for r in csv.DictReader(H200.open()) if r["speedup_wall"].strip()}
    return c, h


def dress(ax, *, ylab: str = "", xlab: bool = True, small: bool = False) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.axvspan(PREFILL_FROM, 1.15e4, color="#f2f1ed", zorder=0)
    ax.grid(True, which="major", color=GRID, linewidth=0.7, zorder=1)  # solid, hairline
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_xscale("log")
    ax.set_xlim(0.80, 1.15e4)
    # Small-multiple panels are ~1/3 the width, so the full 7-value tick list collides at
    # 256/512/1024/2048. Label a readable subset there; the full set on the wide figure.
    ax.xaxis.set_major_locator(
        FixedLocator([1, 32, 256, 1024, 8192] if small else list(T.values())))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{int(v)}") if xlab else NullFormatter())
    ax.tick_params(labelsize=7.5 if small else 8.5, colors=SECONDARY, length=3, width=0.8)
    if ylab:
        ax.set_ylabel(ylab, fontsize=9.5, color=SECONDARY, linespacing=1.5)


# ======================================================================================
def fig_best(c: dict, h: dict) -> None:
    """Best chain per regime, plus H200 running C500's pick. Three series, one axis."""
    cb, hb, hs, picks, hbest = [], [], [], [], []
    for r in REGS:
        cv = [(c[(r, k)], k) for k in COMMON if (r, k) in c]
        best_c, pick = max(cv)
        ha = max((v, k) for (rr, k), v in h.items() if rr == r)
        cb.append(100 * (best_c - 1))
        hb.append(100 * (ha[0] - 1))
        hs.append(100 * (h[(r, pick)] - 1))
        picks.append(pick)
        hbest.append(ha[1])

    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.085, right=0.775, top=0.760, bottom=0.170)
    dress(ax, ylab="how much faster than the all-unfused layer\n(percent, log scale)")
    ax.set_yscale("log")
    ax.set_ylim(0.022, 12.0)
    ax.yaxis.set_major_locator(
        FixedLocator([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} %"))
    ax.yaxis.set_minor_formatter(NullFormatter())

    xs = list(T.values())
    ax.plot(xs, hb, color=H_COL, linewidth=2.0, marker="o", markersize=6.0,
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=6,
            solid_capstyle="round")
    ax.plot(xs, hs, color=H_COL, linewidth=1.6, linestyle=(0, (5, 2.5)), marker="o",
            markersize=4.6, markerfacecolor=SURFACE, markeredgecolor=H_COL,
            markeredgewidth=1.4, zorder=5)
    ax.plot(xs, cb, color=C_COL, linewidth=2.0, marker="s", markersize=5.6,
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=6,
            solid_capstyle="round")

    # Direct labels on the two device series at the right edge; the legend carries the rest.
    # All three series converge inside a factor of 2 at T=8192, so the end labels are
    # spread onto fixed, non-overlapping anchors and leadered back to their own line.
    for y, anchor, col, txt in ((hb[-1], 1.55, H_COL, "H200\nits own best chain"),
                                (hs[-1], 0.62, H_COL, "H200\nrunning C500's chain"),
                                (cb[-1], 0.235, C_COL, "C500\nits own best chain")):
        ax.annotate(txt, xy=(8192, y), xytext=(1.035, anchor),
                    textcoords=("axes fraction", "data"), fontsize=8.7, color=col,
                    va="center", ha="left", annotation_clip=False, zorder=8,
                    linespacing=1.45,
                    arrowprops=dict(arrowstyle="-", color=col, linewidth=0.8, alpha=0.45,
                                    shrinkA=3, shrinkB=3))

    ax.annotate(f"{hb[0]:.2f} %", xy=(1, hb[0]), xytext=(11, 5),
                textcoords="offset points", ha="left", fontsize=8.6, color=H_COL,
                zorder=8, fontweight="bold")
    ax.annotate(f"{cb[0]:.2f} %", xy=(1, cb[0]), xytext=(0, -17),
                textcoords="offset points", ha="center", fontsize=8.6, color=C_COL,
                zorder=8, fontweight="bold")
    ax.text(2400, 0.027, "prefill", fontsize=8.2, color=MUTED, ha="left", zorder=3)

    ax.legend(handles=[
        Line2D([], [], color=H_COL, lw=2.0, marker="o", markersize=6.0,
               markeredgecolor=SURFACE, label="H200 — best chain for that regime"),
        Line2D([], [], color=H_COL, lw=1.6, linestyle=(0, (5, 2.5)), marker="o",
               markersize=4.6, markerfacecolor=SURFACE, markeredgecolor=H_COL,
               label="H200 — running the chain C500 picked"),
        Line2D([], [], color=C_COL, lw=2.0, marker="s", markersize=5.6,
               markeredgecolor=SURFACE, label="C500 — best chain for that regime"),
    ], loc="lower left", fontsize=8.5, frameon=False, labelcolor=SECONDARY,
        handlelength=2.6, borderpad=0.2)

    fig.text(0.085, 0.955, "Layer gain vs T — H200 against C500", fontsize=15.5,
             color=PRIMARY, ha="left", va="top")
    sub = ("C500 is flat: every regime lands between 0.15 % and 0.46 % faster. H200 peaks "
           "at 6.93 % in single-token decode and decays to 0.38 % by T = 1024. The dashed "
           "line is the part that is NOT the device: H200 held to C500's chain gives back "
           "most of the decode advantage, so the gap is mostly about which chains each GPU "
           "makes available, not raw speed.")
    fig.text(0.085, 0.900, "\n".join(textwrap.wrap(sub, 118)),
             fontsize=9.3, color=SECONDARY, ha="left", va="top", linespacing=1.55)
    note = ("Whole-layer speedup over the all-unfused layer, GLM-5.2 MoE decoder subgraph. "
            "C500 has no T = 2/4/8/16. The dashed line dips to 0.03 % at T = 256: C500's "
            "pick there is #8, which is worth 0.37 % on C500 and almost nothing on H200.  "
            "·  sources: report_glm52_c500/layer_optimal_per_regime.csv, "
            "report_glm52_h200/certain/layer_certain_per_regime.csv")
    fig.text(0.085, 0.028, "\n".join(textwrap.wrap(note, 150)),
             fontsize=7.6, color=MUTED, ha="left", va="bottom", linespacing=1.5)

    p = OUT / "report_c500_vs_h200_best.png"
    fig.savefig(p, dpi=140, facecolor=SURFACE)
    print(f"wrote {p}")


# ======================================================================================
def fig_perchain(c: dict, h: dict) -> None:
    """Nine chains, two devices. Polarity is the job, so a linear axis through 1.000."""
    fig, axes = plt.subplots(3, 3, figsize=(11.6, 8.9), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.075, right=0.972, top=0.760, bottom=0.115,
                        hspace=0.32, wspace=0.11)

    n_missing = 0
    for ax, cfg in zip(axes.flat, COMMON):
        dress(ax, small=True)
        ax.set_ylim(0.845, 1.068)
        ax.yaxis.set_major_locator(FixedLocator([0.86, 0.90, 0.94, 0.98, 1.02, 1.06]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.axhline(1.0, color=MUTED, linewidth=1.0, zorder=3)
        for src, col, mk, z in ((h, H_COL, "o", 6), (c, C_COL, "s", 5)):
            xs = [T[r] for r in REGS if (r, cfg) in src]
            ys = [src[(r, cfg)] for r in REGS if (r, cfg) in src]
            if src is c:
                n_missing += len(REGS) - len(xs)
            ax.plot(xs, ys, color=col, linewidth=1.8, marker=mk, markersize=4.6,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=z)
        ax.set_title(LAB[cfg], fontsize=10.5, color=PRIMARY, pad=5.0, loc="left")

    for ax in axes[-1]:
        ax.set_xlabel("T", fontsize=8.5, color=SECONDARY)
    for ax in axes[:, 0]:
        ax.set_ylabel("speedup", fontsize=8.5, color=SECONDARY)

    axes.flat[0].legend(handles=[
        Line2D([], [], color=H_COL, lw=1.8, marker="o", markersize=4.6,
               markeredgecolor=SURFACE, label="H200"),
        Line2D([], [], color=C_COL, lw=1.8, marker="s", markersize=4.6,
               markeredgecolor=SURFACE, label="C500"),
    ], loc="lower left", fontsize=8.2, frameon=False, labelcolor=SECONDARY,
        handlelength=2.2, borderpad=0.2)

    fig.text(0.075, 0.962, "Every chain, both GPUs — where the sign flips", fontsize=15.5,
             color=PRIMARY, ha="left", va="top")
    sub2 = ("Same nine chains measured on both devices. The line through 1.00 is the "
            "all-unfused layer: above it the fusion helps, below it the fusion hurts. Six "
            "chains behave the same way on both. Two do not — #6 falls to 0.864 on C500 at "
            "T = 1024 while holding 1.003 on H200, and #1 turns from 0.978 to 1.004 at "
            "T = 2048. Those two are why C500's rule does not port.")
    fig.text(0.075, 0.912, "\n".join(textwrap.wrap(sub2, 125)),
             fontsize=9.3, color=SECONDARY, ha="left", va="top", linespacing=1.55)
    fig.text(0.075, 0.030,
             f"Shared axes. Gaps are cells C500 never measured ({n_missing} of "
             f"{len(COMMON) * len(REGS)}), not zeroes — the line breaks rather than "
             "interpolating. Shaded band on the right is prefill.",
             fontsize=7.6, color=MUTED, ha="left", va="bottom")

    p = OUT / "report_c500_vs_h200_perchain.png"
    fig.savefig(p, dpi=140, facecolor=SURFACE)
    print(f"wrote {p}  ({n_missing} unmeasured C500 cells)")


def main() -> None:
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    c, h = load()
    fig_best(c, h)
    fig_perchain(c, h)


if __name__ == "__main__":
    main()
