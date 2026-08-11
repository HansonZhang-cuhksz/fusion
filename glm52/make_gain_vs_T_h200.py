"""`report_glm52_h200/gain_vs_T.png` -- per-fusion speedup against total tokens T.

SMALL MULTIPLES, one panel per fusion family. The previous version drew all sixteen
(fusion, variant) series on a single axis and was unreadable for five separate reasons:
the legend box covered the whole data region, the end-of-line labels landed on top of one
another, sixteen series were coloured from a cycled fifteen-colour list (three of them near
duplicates), the two `token-major` variants collapse to 0.01x and forced a three-decade log
y-axis that squashed the other fourteen into the top ~15 % of the canvas, and `tight_layout`
let the axes overflow the figure. Faceting fixes the first four at once; the fifth was a
layout call.

Layout: a 3x4 grid. Ten family panels (#1, #3, #4, #5, #6, #8, #9, #10, #11a, #11b) share
one x (total tokens T, log) and one y (speedup, log, 0.55-2.6x, so a 2x win and a 2x loss
are the same distance from breakeven). Each panel draws every other series in grey behind
its own, so a panel is readable on its own AND against the field. The bottom-right cell
spans two columns and re-plots the two `token-major` series on the full 0.008-3x range --
they are the only series that leave the shared band, and an annotation is not enough for a
100x effect.

Colour carries at most two series per panel (blue = the headline fusion, orange = its
variant), validated all-pairs for CVD. Each panel carries a value block -- colour chip,
name, and the first -> last speedup -- parked in whichever corner the data leaves empty, so
identity is never colour-alone and the numbers are on the figure rather than only in the CSV.

x: decode regimes have T = batch size at kv 4096; prefill has T = sequence length (2048,
8192) and is shaded. y: the `speedup` column of the per-regime CSVs, unfused/fused.

Run:  python3 glm52/make_gain_vs_T_h200.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:          # allow `python3 glm52/make_gain_vs_T_h200.py`
    sys.path.insert(0, str(ROOT))

from glm52.plot_f8_sweep import THEME  # noqa: E402

OUT = ROOT / "report_glm52_h200"

T = {
    "decode_bs1": 1, "decode_bs2": 2, "decode_bs4": 4, "decode_bs8": 8,
    "decode_bs16": 16, "decode_bs32": 32, "decode_bs256": 256,
    "decode_bs512": 512, "decode_bs1024": 1024,
    "prefill_t2048": 2048, "prefill_t8192": 8192,
}
PREFILL_FROM = 1448.0                  # geometric midpoint of 1024 and 2048

# Panel order and titles. A series joins a panel by its fusion id (the first token of the
# `fusion` column); `PANEL_OF` overrides where the id is not its own panel. Anything the
# CSVs grow that is not covered here gets its own panel appended, with a warning -- the
# figure must not silently drop a fusion.
PANELS = [
    ("#1",   "#1  o_proj + ResAdd"),
    ("#3",   "#3  ResAdd + RMSNorm"),
    ("#4",   "#4  ResAdd + RMSNorm + Router"),
    ("#5",   "#5  RMSNorm + Router"),
    ("#6",   "#6  Up_Gate + SwiGLU"),
    ("#8",   "#8  Down + Expert Merge"),
    ("#9",   "#9  Down + Expert Merge + ResAdd2"),
    ("#10",  "#10  Expert Merge + ResAdd"),
    ("#11a", "#11a  Lazy Pre-Norm -> w13 GEMM"),
    ("#11b", "#11b  Lazy Pre-Norm -> router GEMM"),
]
PANEL_OF = {"#11a+#11b": "#11a", "#11b'": "#11b"}

# Short in-panel names. Keyed by (panel, fusion id, variant); the value is the direct label.
SHORT = {
    ("#4", "#4", "F4"): "router",
    ("#4", "#4", "F4_topk"): "+ TopK",
    ("#5", "#5", "F5"): "router",
    ("#5", "#5", "F5_topk"): "+ TopK",
    ("#8", "#8", "atomic (sglang FUSE_SUM_ALL_REDUCE)"): "atomic",
    ("#8", "#8", "token-major"): "token-major",
    ("#9", "#9", "atomic (sglang FUSE_SUM_ALL_REDUCE)"): "atomic",
    ("#9", "#9", "token-major"): "token-major",
    ("#11a", "#11a", "lazy pre-norm (prologue)"): "#11a alone",
    ("#11a", "#11a+#11b", "combined"): "+ #11b combined",
    ("#11b", "#11b", "lazy pre-norm (prologue)"): "#11b full",
    ("#11b", "#11b'", "rstd + epilogue scale"): "#11b' half",
}

# Two slots only, so every panel validates all-pairs (validate_palette.js, light surface
# #fcfcfb: CVD dE 24.7, normal-vision dE 33.6, both >= their floors).
S1, S2 = "#2a78d6", "#eb6834"

YLIM = (0.55, 2.6)
YTICKS = [0.6, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5]
XLIM = (0.80, 1.15e4)
XTICKS = [1, 32, 1024, 8192]
BACKDROP = "#d9d8d1"                   # the other fusions, behind the panel's own


def fmt_x(v, _):
    return f"{int(v)}"


def fmt_y(v, _):
    return ("1x" if v == 1.0 else f"{v:g}x")


def load() -> dict[tuple[str, str], dict[int, float]]:
    """{(fusion, variant): {T: speedup}} over every per-regime CSV that exists."""
    series: dict[tuple[str, str], dict[int, float]] = {}
    for reg, t in T.items():
        p = OUT / f"fusion_{reg}.csv"
        if not p.exists():
            print(f"!! missing {p}")
            continue
        with p.open() as fh:
            for r in csv.DictReader(fh):
                sp = r["speedup"].strip()
                if sp:
                    series.setdefault((r["fusion"], r["variant"]), {})[t] = float(sp)
    return series


def fusion_id(fusion: str) -> str:
    """`#11a Lazy Pre-Norm -> w13 grouped GEMM` -> `#11a`."""
    return re.split(r"\s", fusion.strip(), maxsplit=1)[0]


def assign(series) -> tuple[list[tuple[str, str, list]], list[str]]:
    """Bucket every series into a panel; return (panels, warnings). Panels keep PANELS
    order, then any unknown id in CSV order so nothing is dropped."""
    order = [k for k, _ in PANELS]
    titles = dict(PANELS)
    buckets: dict[str, list] = {k: [] for k in order}
    warn = []
    for (fusion, variant), pts in series.items():
        fid = fusion_id(fusion)
        panel = PANEL_OF.get(fid, fid)
        if panel not in buckets:
            warn.append(f"unlisted fusion {fusion!r} ({variant!r}) -> own panel {panel}")
            buckets[panel] = []
            order.append(panel)
            titles[panel] = f"{panel}  {fusion}"
        buckets[panel].append((fusion, variant, pts))
    # Headline fusion first (its id IS the panel id, else the shortest name) -> blue.
    for k, items in buckets.items():
        items.sort(key=lambda it: (fusion_id(it[0]) != k, len(it[0])))
    return [(k, titles[k], buckets[k]) for k in order if buckets[k]], warn


def value_block(ax, c, rows, ylim, ys_in_panel, *, fs=8.5, corner=None):
    """Chip + name + `first -> last` per series, parked in the emptier of the two corners.

    Replaces per-line end labels: at four panels across, a label to the right of the last
    point either overflows into the next panel or forces so much x-margin that the measured
    range shrinks. The corner is chosen from the data, not fixed, because #11a lives low
    (0.62x) while #3 and #10 live high (2.2x) and no single corner is free in both.
    """
    import math
    lo, hi = math.log10(ylim[0]), math.log10(ylim[1])
    need = 0.085 * len(rows) + 0.045                 # block height, axes fraction

    def frac(y):
        return (math.log10(min(max(y, ylim[0]), ylim[1])) - lo) / (hi - lo)

    fr = [frac(y) for y in ys_in_panel]
    top_busy = sum(1 for f in fr if f > 1.0 - need - 0.05)
    bot_busy = sum(1 for f in fr if f < need + 0.05)
    top = (top_busy < bot_busy) if corner is None else (corner == "top")
    y0 = (1.0 - 0.045 - 0.085) if top else (need - 0.085 + 0.010)

    for k, (name, first, last, col, off) in enumerate(rows):
        y = y0 - 0.085 * k if top else y0 + 0.085 * (len(rows) - 1 - k)
        ax.plot([0.033], [y + 0.012], marker="s", markersize=4.5, color=col,
                transform=ax.transAxes, clip_on=False, linestyle="none", zorder=8)
        tail = (f"{last:.3f}x  off axis" if off else
                f"{last:.3f}x" if last < 0.1 else f"{last:.2f}x")
        # The halo keeps the block legible where it crosses a grey backdrop line; the
        # corner picker guarantees it never crosses the panel's OWN series.
        ax.text(0.072, y, f"{name}   {first:.2f} → {tail}", fontsize=fs,
                color=c["secondary"], transform=ax.transAxes, va="bottom", ha="left",
                zorder=8, bbox=dict(boxstyle="square,pad=0.18", fc=c["surface"],
                                    ec="none", alpha=0.82))


def dress(ax, c, *, ylim, yticks, xlabels: bool):
    ax.set_facecolor(c["surface"])
    ax.set_axisbelow(True)
    ax.axvspan(PREFILL_FROM, XLIM[1], color=c["harm"], zorder=0)
    ax.grid(True, which="major", color=c["grid"], linewidth=0.7, zorder=1)
    ax.axhline(1.0, color=c["rule"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(FixedLocator(XTICKS))
    ax.xaxis.set_minor_locator(FixedLocator(sorted(set(T.values())) ))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_x) if xlabels else NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_y))
    ax.tick_params(labelsize=8, colors=c["secondary"], length=3, width=0.8)


def draw_series(ax, pts, color, *, lw=2.0, z=5, surface="#fcfcfb", dashed=False):
    xs = sorted(pts)
    ys = [pts[x] for x in xs]
    ax.plot(xs, ys, color=color, linewidth=lw, zorder=z, solid_capstyle="round",
            dashes=[4.5, 2.5] if dashed else (),
            marker="o", markersize=5, markeredgecolor=surface, markeredgewidth=1.1,
            clip_on=True)
    return xs[-1], ys[-1]


def main() -> None:
    series = load()
    if not series:
        raise SystemExit("no speedup rows found under " + str(OUT))
    panels, warn = assign(series)
    for w in warn:
        print(f"!! {w}")

    c = THEME["light"]
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig = plt.figure(figsize=(14.2, 10.6))
    fig.patch.set_facecolor(c["surface"])
    gs = fig.add_gridspec(3, 4, left=0.045, right=0.988, top=0.845, bottom=0.082,
                          hspace=0.30, wspace=0.20)

    slots = [gs[r, col] for r in range(3) for col in range(4)]
    for i, (key, title, items) in enumerate(panels[:11]):
        ax = fig.add_subplot(slots[i])
        dress(ax, c, ylim=YLIM, yticks=YTICKS, xlabels=True)
        for other_key, _, other_items in panels:      # the field, in grey, for context
            if other_key == key:
                continue
            for _, _, pts in other_items:
                xs = sorted(pts)
                ax.plot(xs, [pts[x] for x in xs], color=BACKDROP, linewidth=1.1,
                        zorder=3, solid_capstyle="round")
        rows, own_y = [], []
        for j, (fusion, variant, pts) in enumerate(items):
            col = (S1, S2)[j % 2]
            draw_series(ax, pts, col, surface=c["surface"])
            xs = sorted(pts)
            own_y += [pts[x] for x in xs]
            name = SHORT.get((key, fusion_id(fusion), variant), fusion_id(fusion))
            rows.append((name, pts[xs[0]], pts[xs[-1]], col, pts[xs[-1]] < YLIM[0]))
        value_block(ax, c, rows, YLIM, own_y)
        ax.set_title(title, fontsize=10.5, color=c["primary"], loc="left", pad=6)
        if i % 4 == 0:
            ax.set_ylabel("speedup", fontsize=9, color=c["secondary"])
        if i == 0:
            ax.text(1.05, 2.42, "decode", fontsize=8, color=c["secondary"], ha="left")
            ax.text(2400, 2.42, "prefill", fontsize=8, color=c["secondary"], ha="left")

    # The two series that leave the shared band, on their own full-range axis.
    ax = fig.add_subplot(gs[2, 2:])
    dress(ax, c, ylim=(0.0075, 3.4),
          yticks=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0], xlabels=True)
    for _, _, items in panels:
        for _, _, pts in items:
            xs = sorted(pts)
            ax.plot(xs, [pts[x] for x in xs], color=BACKDROP, linewidth=1.1, zorder=3)
    rows, own_y = [], []
    for j, (fusion, variant) in enumerate([("#8 Down + Expert Merge", "token-major"),
                                           ("#9 Down + Expert Merge + ResAdd2",
                                            "token-major")]):
        pts = series.get((fusion, variant))
        if not pts:
            continue
        # The two curves coincide from T=4 on -- same pathology, same slope -- so the
        # second is dashed over the first instead of hiding under it.
        col = (S1, S2)[j]
        draw_series(ax, pts, col, surface=c["surface"], dashed=bool(j), lw=2.4 - 0.6 * j,
                    z=5 + j)
        xs = sorted(pts)
        own_y += [pts[x] for x in xs]
        rows.append((f"{fusion_id(fusion)} token-major", pts[xs[0]], pts[xs[-1]], col,
                     False))
    value_block(ax, c, rows, (0.0075, 3.4), own_y, corner="bottom")
    ax.set_title("the two series that leave the shared band above  (full y-range, log)",
                 fontsize=10.5, color=c["primary"], loc="left", pad=6)
    ax.set_xlabel("total tokens T per fused call", fontsize=9, color=c["secondary"])
    ax.annotate("token-major re-reads the expert accumulator once per\n"
                "token, so its cost grows with T instead of amortising:\n"
                "by prefill it is ~100x slower than the unfused chain.\n"
                "The other fourteen series (grey) stay in the 0.6-2.5x\n"
                "band of the panels above.",
                xy=(0.035, 0.36), xycoords="axes fraction", fontsize=8.5,
                color=c["secondary"], va="bottom", ha="left", linespacing=1.55,
                bbox=dict(boxstyle="square,pad=0.4", fc=c["surface"], ec="none",
                          alpha=0.82))

    for i in (8, 9):                     # bottom-row family panels carry the x label
        fig.axes[i].set_xlabel("total tokens T per fused call", fontsize=9,
                               color=c["secondary"])

    n_series = len(series)
    fig.text(0.045, 0.965, "GLM-5.2 MoE decoder layer on NVIDIA H200 "
             "— kernel-fusion speedup vs tokens per call",
             fontsize=16, color=c["primary"], ha="left", va="top")
    fig.text(0.045, 0.928,
             "One panel per fusion. y = unfused-chain time / fused time on a log scale, so "
             "a 2x win and a 2x loss sit the same distance from the dashed 1x breakeven. "
             "Grey lines are all\nthe other fusions, for context. "
             "x = total tokens T per call: decode regimes are batch size at kv 4096 "
             "(T ≤ 1024); prefill is sequence length (shaded). Markers are the eleven "
             "measured regimes.",
             fontsize=9.5, color=c["secondary"], ha="left", va="top", linespacing=1.5)
    fig.text(0.045, 0.017,
             f"source: report_glm52_h200/fusion_<regime>.csv  ·  "
             f"{n_series} (fusion, variant) series × {len(T)} regimes  ·  "
             f"#11 cells are the repaired f11_lazy_prenorm campaign; see README §1.3 "
             f"for the gated adjudication of #11",
             fontsize=8, color=c["secondary"], ha="left", va="bottom")

    p = OUT / "gain_vs_T.png"
    fig.savefig(p, dpi=140, facecolor=c["surface"])
    print(f"wrote {p}  ({len(panels)} panels, {n_series} series, "
          f"{len(T)} regimes)")


if __name__ == "__main__":
    main()
