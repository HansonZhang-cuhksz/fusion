"""`report_glm52_h200/chain_gain_vs_T.png` -- nine chains, everywhere, including where they lose.

Companion to `best_chain_vs_T.png`, and directly comparable with it. That figure draws ONE
curve, the per-regime winner, so a chain appears only where it happens to lead. This one takes
NINE chains -- the five that figure names as per-regime winners, plus four co-leaders recovered
from the CSV's `tied_with_best_*` flags (that figure reports its tie sets only as COUNTS, so
`#3+#9`, `#3+#8`, `#10` and `#1+#10+#11a+#11b` are never named there) -- and draws each one's
whole-layer gain across all eleven regimes. The pointwise maximum of these nine reproduces that
figure's single curve exactly at all eleven regimes, which is the check that the two are the
same measurement. NOT comparable with `gain_vs_T.png`, whose denominator is one two- or
three-kernel unfused chain rather than the whole S3-S11 + shared-expert subgraph.

THIS IS NOT THE WHOLE FIELD. Fourteen configurations were timed at each regime (eighteen at
`decode_bs1`); nine chains are drawn. Five measured chains are never drawn -- `#8`, `#9`, `#5`,
`#4`, `#11b`, plus the three other `#11a` chains that exist only at `decode_bs1` -- and some of
them outrank drawn ones: at T = 1 `#4` alone is 1.1895x, ahead of six of the nine here. The
selection rule is "what `best_chain_vs_T.png` resolves", not "the top nine". Both the figure
and the notes say so on canvas.

What drawing the losses buys you: the greedy chain that wins decode_bs1 by 1.289x is the WORST
of the nine at prefill_t8192 (0.939x), and `#6`, the WINNER at T = 512 (1.0107x, 4 tied), is
the worst of all fourteen at T = 2048 (0.9367x). A winner-only curve cannot show that.
Nineteen of the eighty T>=2 cells are below 1.000x.

Three stacked panels, one shared log-T axis:
  * upper   -- broken y. Every T = 1 point (1.0575-1.2886), with the ranked fan and the key.
  * middle  -- broken y. Every T >= 2 point (0.9367-1.0503). The break sits in a genuinely
               empty data band -- nothing lies between 1.0503 and 1.0575 -- and every curve is
               drawn into BOTH segments, so the T=1 -> T=2 collapse is a continuous line.
  * lower    -- MAGNIFIER, same x, y stretched ~3x over 0.9895-1.0295, where eight of the nine
               chains spend every regime from T = 8 on. Curves that leave it are off-scale
               there, not missing; read them in the middle panel.

THE SHADED BAND IS NOT A SIGNIFICANCE BAND. Its per-regime half-width is the p90 of
|run1_ms - run2_ms| / min(run1_ms, run2_ms) over every candidate timed in that regime (18 at
decode_bs1, 14 elsewhere), i.e. how well the two passes' medians repeat -- a U shape, 1.41 % at
T=1, 0.03 % at T=8-16, 0.80 % at T=8192. The threshold the report actually uses to decide
anything is the LOG-11 S3 round-to-round spread, which is wider than this band at every regime
by a factor computed at run time from results/h200/layer_configurations.json. Leaving this band
is not evidence of a win; see the notes.

Identity is carried four ways, never by hue alone: the five winners take validated categorical
hues as solid lines, each with its own marker shape (greyscale reads the shape, not the hue);
the three co-leaders take a neutral lightness ramp with distinct dashes and open markers; every
curve is direct-labelled at both ends -- a fan at T=1 and a de-collided gutter at T=8192 -- and
the full 9 x 11 matrix is printed underneath.

Run:  /home/zhangshuhan/my-envs/fusion/bin/python glm52/make_chain_gain_vs_T_h200.py
      (from the repo root; the default python3 on this box has no matplotlib)
"""
from __future__ import annotations

import csv
import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402
from matplotlib.transforms import blended_transform_factory, offset_copy  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glm52.plot_f8_sweep import THEME  # noqa: E402

OUT = ROOT / "report_glm52_h200"
SRC = OUT / "layer_optimal_per_regime.csv"
RAW = ROOT / "results" / "h200" / "layer_configurations.json"

T = {
    "decode_bs1": 1, "decode_bs2": 2, "decode_bs4": 4, "decode_bs8": 8,
    "decode_bs16": 16, "decode_bs32": 32, "decode_bs256": 256,
    "decode_bs512": 512, "decode_bs1024": 1024,
    "prefill_t2048": 2048, "prefill_t8192": 8192,
}
REGIMES = [r for r, _ in sorted(T.items(), key=lambda kv: kv[1])]
TVALS = [T[r] for r in REGIMES]
UNFUSED = "A_all_unfused"
PREFILL_FROM = 1448.0

# Tier 1 -- the five per-regime winners. Hue is the primary channel and was validated all-pairs
# on #fcfcfb (CVD PASS worst dE 13.0 protan, normal-vision PASS worst dE 16.3), but hue alone
# dies in greyscale, so every winner also carries its own MARKER SHAPE at every one of its
# eleven points -- that is the greyscale key. `#1` was darkened from #eda100 to #b3760a
# (L* 72 -> 54) because at the original lightness it read as a gridline artefact once
# desaturated; the darkening only widens the lightness spread, so CVD separability improves.
# `zorder` is set per chain so the narrower-gap series draws on top: `#3+#10` (violet, the most
# frequent winner) sits above `#3` (pink), which used to bury it from T = 32 to T = 1024, and
# `#6` (green) sits above everything because it owns both of the figure's extremes -- the
# 1.0107x peak at T = 512 and the 0.9367x global minimum at T = 2048 -- and the greedy chain
# passes within 4-7 px of it at both.
WINNERS = {
    "J_greedy_all": dict(color="#2a78d6", lw=1.9, marker="o", z=6.6),
    "H_f3_f10":     dict(color="#4a3aa7", lw=1.9, marker="s", z=6.5),
    "D_f6":         dict(color="#008300", lw=1.9, marker="D", z=6.7),
    "B_f3":         dict(color="#e87ba4", lw=2.2, marker="^", z=6.2),
    "C_f1":         dict(color="#b3760a", lw=2.2, marker="v", z=6.1),
}
# Tier 2 -- tie-set co-leaders. No hue at all: a neutral lightness ramp crossed with three dash
# patterns and three open marker shapes. Lightness and dash both survive protanopia,
# deuteranopia, tritanopia AND greyscale unchanged. The ramp was darkened so that its lightest
# member (`#10`) no longer collides with the 1.000x reference rule, which it used to share a
# hex value with.
COLEADERS = {
    "I_f3_f9": dict(color="#3d3c39", lw=1.25, dashes=[5.0, 2.5], marker="o", z=4.3),
    "K_f3_f8": dict(color="#63625d", lw=1.25, dashes=[7.0, 2.4, 1.6, 2.4], marker="s", z=4.2),
    "E_f10":   dict(color="#8a8882", lw=1.25, dashes=[1.5, 2.0], marker="^", z=4.1),
}
LONE = "R_f1_f10_f11ab"          # decode_bs1 only -- a single marker, never a line
LONE_COLOR = "#2a78d6"           # J's blue: R is the other member of J's T = 1 tie set
LONE_DX = 27.0                   # dots: R is nudged right of J so the two do not merge
CHAIN_ORDER = list(WINNERS) + list(COLEADERS) + [LONE]

REF = "#bcbab3"                  # the 1.000x rule -- lighter than every co-leader grey, solid
PREFILL_FILL = "#edf0f3"         # tinted, so it is not competing with the band on lightness

XLIM = (0.80, 1.15e4)
XTICKS = TVALS                   # every measured regime is labelled
YLO = (0.9330, 1.0525)           # middle segment: every T >= 2 point (max 1.0503)
YHI = (1.0505, 1.3300)           # upper segment: every T = 1 point  (min 1.0575)
YMAG = (0.9895, 1.0295)          # magnifier
YT_LO = [0.94, 0.96, 0.98, 1.00, 1.02, 1.04]
YT_HI = [1.10, 1.15, 1.20, 1.25, 1.30]
YT_MAG = [0.99, 1.00, 1.01, 1.02]

EXPECT_P90 = [0.014092, 0.001557, 0.000614, 0.000330, 0.000337, 0.000409,
              0.001121, 0.002667, 0.006929, 0.003862, 0.007976]
EXPECT_SPREAD_RATIO = (2.8, 1440.1)   # max round-to-round spread / band, floor and peak
EXPECT_RELERR = (0.064, 0.698)        # O/P/Q/R fp32-reference failure range, T >= 2
EXPECT_RESOLVED = 22                  # of the 89 drawn points, vs A_all_unfused


def short_chain(fusion_set: str) -> str:
    """`#1+#6+#9 (greedy)` -> `#1+#6+#9`; `#3 + #10` -> `#3+#10`."""
    return fusion_set.split("(")[0].replace(" ", "").strip() or fusion_set


def p90(xs: list[float]) -> float:
    """numpy.percentile(xs, 90) with linear interpolation, without importing numpy."""
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = 0.90 * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (pos - lo) * (s[hi] - s[lo])


def xfrac(t: float) -> float:
    """T -> axes fraction on the shared log x axis."""
    a, b = math.log10(XLIM[0]), math.log10(XLIM[1])
    return (math.log10(t) - a) / (b - a)


def from_raw(band: dict) -> dict:
    """Everything the figure asserts that lives in the campaign JSON, not the CSV.

    Recomputed every run so a stale literal cannot survive: the round-to-round spread the
    LOG-11 S3 rule actually uses, the fp32-reference failure range for the four prenorm="all"
    chains, and the pairwise (chain vs A_all_unfused) resolution count.
    """
    out = dict(ratio=None, med_ratio=None, min_ratio=None, relerr=None,
               resolved=None, n_points=None)
    if not RAW.exists():
        print(f"!! {RAW} missing -- the derived note figures are omitted")
        return out
    regs = json.loads(RAW.read_text())["regimes"]

    # (1) round-to-round spread = max - min over the 8 rounds, as a fraction of that
    #     configuration's median. Aggregation: the WIDEST such spread in each regime, over
    #     every configuration and both passes -- the same aggregation that yields the 1440x
    #     peak (decode_bs4). Ratio to this figure's plotted band half-width.
    ratios, med, mn = [], [], []
    for reg in REGIMES:
        rd = regs[reg]
        sp = sorted(v["spread_ms"] / v["median"] for p in ("pass1", "pass2")
                    for v in rd[p].values())
        ratios.append(max(sp) / band[reg])
        n = len(sp)
        med.append((sp[n // 2] if n % 2 else 0.5 * (sp[n // 2 - 1] + sp[n // 2]))
                   / band[reg])
        mn.append(min(sp) / band[reg])
    out["ratio"] = (min(ratios), max(ratios))
    out["med_ratio"] = (min(med), max(med))
    out["min_ratio"] = (min(mn), max(mn))
    if (round(min(ratios), 1), round(max(ratios), 1)) != EXPECT_SPREAD_RATIO:
        print(f"!! spread/band ratio drifted: got {min(ratios):.1f}-{max(ratios):.1f}, "
              f"expected {EXPECT_SPREAD_RATIO[0]}-{EXPECT_SPREAD_RATIO[1]}")

    # (2) rel_err of the four prenorm="all" chains against the independent fp32 reference,
    #     over the ten regimes where they failed.
    bad = ("O_f11ab", "P_f10_f11ab", "Q_f8_f11ab", "R_f1_f10_f11ab")
    errs = [v["rel_err"] for reg, rd in regs.items() if reg != "decode_bs1"
            for k, v in (rd.get("correctness") or {}).items()
            if k in bad and isinstance(v, dict) and "rel_err" in v]
    if errs:
        out["relerr"] = (min(errs), max(errs), len(errs))
        if (round(min(errs), 3), round(max(errs), 3)) != EXPECT_RELERR:
            print(f"!! rel_err range drifted: got {min(errs):.3f}-{max(errs):.3f}, "
                  f"expected {EXPECT_RELERR}")

    # (3) the LOG-11 rule applied pairwise to (chain, A_all_unfused) -- a comparison the
    #     report does not itself publish. Both passes must separate AND agree on the sign.
    res = tot = 0
    for reg in REGIMES:
        rd = regs[reg]
        for cid in CHAIN_ORDER:
            if cid not in rd["pass1"]:
                continue
            tot += 1
            ok = True
            for p in ("pass1", "pass2"):
                a, b = rd[p][UNFUSED], rd[p][cid]
                if abs(a["median"] - b["median"]) <= max(a["spread_ms"], b["spread_ms"]):
                    ok = False
            d1 = rd["pass1"][cid]["median"] - rd["pass1"][UNFUSED]["median"]
            d2 = rd["pass2"][cid]["median"] - rd["pass2"][UNFUSED]["median"]
            res += bool(ok and d1 * d2 > 0)
    out["resolved"], out["n_points"] = res, tot
    if res != EXPECT_RESOLVED:
        print(f"!! pairwise-resolved count drifted: got {res}, expected {EXPECT_RESOLVED}")
    return out


def load() -> dict:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC} -- run glm52/make_layer_report_h200.py first")
    rows = list(csv.DictReader(SRC.open()))

    gain: dict[str, dict[str, float]] = {}
    label: dict[str, str] = {}
    all_gain: dict[str, dict[str, float]] = {}
    all_label: dict[str, str] = {}
    band: dict[str, float] = {}
    winner: dict[str, str] = {}
    verdict: dict[str, str] = {}
    tiesets: dict[str, set[str]] = {}
    n_cand: dict[str, int] = {}

    for reg in REGIMES:
        rr = [r for r in rows if r["regime"] == reg]
        if not rr:
            raise SystemExit(f"no rows for {reg}")
        n_cand[reg] = len(rr)
        # noise band: pass-to-pass repeatability over EVERY candidate timed in the regime,
        # A_all_unfused included -- 158 rows in total across the eleven regimes.
        band[reg] = p90([abs(float(r["run1_ms"]) - float(r["run2_ms"]))
                         / min(float(r["run1_ms"]), float(r["run2_ms"])) for r in rr])

        scored = [r for r in rr if r["speedup_vs_unfused"].strip()]
        best = max(scored, key=lambda r: float(r["speedup_vs_unfused"]))
        winner[reg] = best["config_id"]
        tie = {r["config_id"] for r in rr
               if r["tied_with_best_run1"] == "1" or r["tied_with_best_run2"] == "1"}
        tiesets[reg] = tie
        verdict[reg] = ("SEPARATED" if len(tie) == 1 else
                        f"null ({len(tie)})" if UNFUSED in tie else f"{len(tie)} tied")

        for r in rr:
            cid, v = r["config_id"], float(r["speedup_vs_unfused"])
            all_gain.setdefault(cid, {})[reg] = v
            all_label[cid] = short_chain(r["fusion_set"])
            if cid in CHAIN_ORDER:
                gain.setdefault(cid, {})[reg] = v
                label[cid] = short_chain(r["fusion_set"])

    missing = [c for c in CHAIN_ORDER if c not in gain]
    if missing:
        raise SystemExit(f"chains absent from the CSV: {missing}")
    for cid in CHAIN_ORDER:
        want = 1 if cid == LONE else len(REGIMES)
        if len(gain[cid]) != want:
            print(f"!! {cid}: {len(gain[cid])} regimes, expected {want}")

    got = [round(band[r], 6) for r in REGIMES]
    if got != EXPECT_P90:
        print(f"!! noise band drifted from the audited values\n   got    {got}\n"
              f"   expect {EXPECT_P90}")

    # The check that makes this figure comparable with best_chain_vs_T.png.
    envelope = [max(gain[c][r] for c in CHAIN_ORDER if r in gain[c]) for r in REGIMES]
    peak = [max(float(x["speedup_vs_unfused"]) for x in rows if x["regime"] == r)
            for r in REGIMES]
    if any(abs(a - b) > 1e-9 for a, b in zip(envelope, peak)):
        print(f"!! envelope != best_chain_vs_T curve\n   {envelope}\n   {peak}")
    else:
        print("ok  pointwise max of the 9 curves == best_chain_vs_T.png at all 11 regimes")

    # what is NOT drawn, and how it ranks -- stated on canvas rather than left to the reader
    undrawn = [c for c in all_gain if c not in CHAIN_ORDER and c != UNFUSED]
    undrawn.sort(key=lambda c: -all_gain[c].get("decode_bs1", 0.0))
    everywhere = [c for c in undrawn if len(all_gain[c]) == len(REGIMES)]
    t1_only = [c for c in undrawn if len(all_gain[c]) == 1]
    top_undrawn = undrawn[0] if undrawn else None
    beats = 0
    if top_undrawn:
        v = all_gain[top_undrawn]["decode_bs1"]
        beats = sum(1 for c in CHAIN_ORDER if gain[c]["decode_bs1"] < v)

    return dict(gain=gain, label=label, band=band, winner=winner, verdict=verdict,
                tiesets=tiesets, n_cand=n_cand, rows=rows, all_gain=all_gain,
                all_label=all_label, undrawn=undrawn, undrawn_everywhere=everywhere,
                undrawn_t1_only=t1_only, top_undrawn=top_undrawn, top_undrawn_beats=beats,
                raw=from_raw(band))


def roles(d: dict) -> dict[str, str]:
    """Regime-specific role text. A tie set containing A_all_unfused is a null result,
    not a leadership credential, and is flagged as such."""
    out = {}
    for cid in CHAIN_ORDER:
        mine = [r for r in REGIMES if r in d["gain"][cid]]
        won = [r for r in mine if d["winner"][r] == cid]
        co = [r for r in mine if cid in d["tiesets"][r] and r not in won
              and UNFUSED not in d["tiesets"][r]]
        null = [r for r in mine if cid in d["tiesets"][r] and UNFUSED in d["tiesets"][r]]
        if won:
            star = "*" if any(UNFUSED in d["tiesets"][r] for r in won) else ""
            out[cid] = "wins T = " + ", ".join(str(T[r]) for r in won) + star
        elif co:
            out[cid] = "co-leads T = " + ", ".join(str(T[r]) for r in co)
        elif null:
            out[cid] = ("T = " + "/".join(str(T[r]) for r in null)
                        + " tie sets only *")
        else:
            out[cid] = "never leads, never tied"
    return out


def spread_labels(items, gap, lo, hi):
    """Place labels at their true y, then push apart to `gap` while staying centred.

    Merge overlapping runs into blocks, centre each block on the mean of its members'
    true values, and only then slide the whole set into [lo, hi]. Members keep their
    descending order inside a block, so a label never crosses its neighbour's line.
    If the spread out-grows the segment, the gap is tightened until it fits -- better a
    cramped stack than a label rendered outside the axes.
    """
    order = sorted(items, key=lambda kv: -kv[1])

    def place(g_gap):
        groups = [[kv] for kv in order]

        def mean(g):
            return sum(y for _, y in g) / len(g)

        def top(g):
            return mean(g) + (len(g) - 1) * g_gap / 2.0

        def bot(g):
            return mean(g) - (len(g) - 1) * g_gap / 2.0

        merged = True
        while merged:
            merged = False
            i = 0
            while i < len(groups) - 1:
                if bot(groups[i]) - top(groups[i + 1]) < g_gap:
                    groups[i:i + 2] = [groups[i] + groups[i + 1]]
                    merged = True
                else:
                    i += 1
        out = {}
        for g in groups:
            t = top(g)
            for j, (k, _) in enumerate(g):
                out[k] = t - j * g_gap
        return out

    out = place(gap)
    for _ in range(8):
        span = max(out.values()) - min(out.values())
        if span <= hi - lo:
            break
        gap *= 0.97 * (hi - lo) / span
        out = place(gap)
    if max(out.values()) > hi:
        out = {k: v - (max(out.values()) - hi) for k, v in out.items()}
    if min(out.values()) < lo:
        out = {k: v + (lo - min(out.values())) for k, v in out.items()}
    return out


def style(cid):
    if cid in WINNERS:
        return dict(WINNERS[cid], tier="winner")
    if cid in COLEADERS:
        return dict(COLEADERS[cid], tier="coleader")
    return dict(color=LONE_COLOR, lw=0.0, marker="D", z=7.0, tier="lone")


def dress(ax, c, ylim, yticks, *, xlabels: bool):
    ax.set_facecolor(c["surface"])
    ax.set_axisbelow(True)
    ax.axvspan(PREFILL_FROM, XLIM[1], color=PREFILL_FILL, zorder=0)
    ax.axvline(PREFILL_FROM, color="#c9d3dc", linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
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
    ax.tick_params(which="minor", length=0)


def draw_curves(ax, c, d, *, big: bool):
    """Every chain into every segment; clipping decides what each segment shows."""
    ms = 5.0 if big else 4.2
    for cid in list(WINNERS) + list(COLEADERS):
        st = style(cid)
        ys = [d["gain"][cid][r] for r in REGIMES]
        kw = dict(color=st["color"], linewidth=st["lw"], zorder=st["z"], marker=st["marker"],
                  markersize=ms if st["tier"] == "winner" else ms - 0.9,
                  markeredgewidth=1.0 if st["tier"] == "winner" else 1.3,
                  markerfacecolor=st["color"] if st["tier"] == "winner" else c["surface"],
                  markeredgecolor=c["surface"] if st["tier"] == "winner" else st["color"])
        if "dashes" in st:
            kw["dashes"] = st["dashes"]
        else:
            kw["solid_capstyle"] = "round"
        ax.plot(TVALS, ys, **kw)


def draw_T1_cluster(ax, fig, c, d):
    """J and R are 0.23 % apart at T = 1 and merge into one glyph at any usable scale.

    R is nudged LONE_DX dots to the right, tied back to its true T = 1 position by a dotted
    connector, and drawn as an open diamond -- a shape no series uses -- so its "single point,
    no line" status reads at the marker. J is then drawn last, on top of everything, with a
    white halo, so the figure's headline number owns the line that leaves it.
    """
    yr = d["gain"][LONE]["decode_bs1"]
    tr = offset_copy(ax.transData, fig=fig, x=LONE_DX, y=0.0, units="dots")
    ax.annotate("", xy=(1.0, yr), xytext=(LONE_DX - 9.0, 0.0), textcoords="offset points",
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=LONE_COLOR, lw=0.9,
                                linestyle=(0, (1.4, 1.4)), shrinkA=0, shrinkB=0))
    ax.plot([1.0], [yr], transform=tr, marker="D", markersize=8.5, linestyle="none",
            markerfacecolor=c["surface"], markeredgecolor=LONE_COLOR, markeredgewidth=1.9,
            zorder=7.5, clip_on=False)
    # J on top, with a white halo
    yj = d["gain"]["J_greedy_all"]["decode_bs1"]
    ax.plot([1.0], [yj], marker="o", markersize=8.0, linestyle="none",
            markerfacecolor=WINNERS["J_greedy_all"]["color"], markeredgecolor=c["surface"],
            markeredgewidth=2.2, zorder=12, clip_on=False)


def main() -> None:
    d = load()
    gain, lab, raw = d["gain"], d["label"], d["raw"]
    role = roles(d)
    c = THEME["light"]
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    fig = plt.figure(figsize=(15.6, 19.6))
    fig.patch.set_facecolor(c["surface"])

    gs_plot = fig.add_gridspec(1, 1, left=0.062, right=0.858, top=0.900, bottom=0.455)
    inner = gs_plot[0].subgridspec(3, 1, height_ratios=[3.35, 3.95, 2.90], hspace=0.05)
    ax_hi = fig.add_subplot(inner[0])
    ax_lo = fig.add_subplot(inner[1])
    ax_mg = fig.add_subplot(inner[2])
    ax_tab = fig.add_subplot(fig.add_gridspec(1, 1, left=0.062, right=0.984,
                                              top=0.428, bottom=0.290)[0])

    dress(ax_hi, c, YHI, YT_HI, xlabels=False)
    dress(ax_lo, c, YLO, YT_LO, xlabels=False)
    dress(ax_mg, c, YMAG, YT_MAG, xlabels=True)
    for ax in (ax_hi, ax_lo):
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(which="both", bottom=False)

    # ---- the noise band, then the 1.000x rule, then the curves on top ------------
    # The band runs to both axis edges: at five regimes its half-width is 1-8 px, so without
    # the edge hairlines and the flat extensions the wedge at the left and the sleeve at the
    # right do not read as the same object, and it used to stop dead at T = 8192.
    bx = [XLIM[0]] + TVALS + [XLIM[1]]
    blo = [1.0 - d["band"][REGIMES[0]]] + [1.0 - d["band"][r] for r in REGIMES] \
        + [1.0 - d["band"][REGIMES[-1]]]
    bhi = [1.0 + d["band"][REGIMES[0]]] + [1.0 + d["band"][r] for r in REGIMES] \
        + [1.0 + d["band"][REGIMES[-1]]]
    for ax in (ax_lo, ax_mg):
        ax.fill_between(bx, blo, bhi, color=c["rule"], alpha=0.16, zorder=1.5, linewidth=0)
        ax.plot(bx, blo, color=c["rule"], lw=0.6, alpha=0.55, zorder=1.6)
        ax.plot(bx, bhi, color=c["rule"], lw=0.6, alpha=0.55, zorder=1.6)
        ax.axhline(1.0, color=REF, linewidth=1.1, zorder=2)
    for ax, big in ((ax_hi, True), (ax_lo, True), (ax_mg, False)):
        draw_curves(ax, c, d, big=big)
    draw_T1_cluster(ax_hi, fig, c, d)

    # break marks, repeated across the seam so a reader scanning the middle sees them
    kw = dict(color=c["rule"], linewidth=1.1, clip_on=False, zorder=9)
    for ax, y_ax in ((ax_hi, 0.0), (ax_lo, 1.0)):
        for x_ax in (0.0, 0.36, 0.60, 0.84, 1.0):
            ax.plot([x_ax - 0.006, x_ax + 0.006], [y_ax - 0.026, y_ax + 0.026],
                    transform=ax.transAxes, **kw)

    fig.canvas.draw()

    def gap_for(ax, ylim, pt):
        """One comfortable text line, in data units of that segment."""
        h = ax.get_window_extent().height
        return (pt * 1.42 * fig.dpi / 72.0) / h * (ylim[1] - ylim[0])

    tlo = blended_transform_factory(ax_lo.transAxes, ax_lo.transData)
    thi = blended_transform_factory(ax_hi.transAxes, ax_hi.transData)
    X_T1, X_T8192 = xfrac(1.0), xfrac(8192.0)
    DX_LONE = LONE_DX / 72.0 * fig.dpi / ax_hi.get_window_extent().width

    # ---- left fan: all nine T = 1 points, ranked --------------------------------
    hi_items = sorted(((cid, gain[cid]["decode_bs1"]) for cid in CHAIN_ORDER),
                      key=lambda kv: -kv[1])
    band1 = d["band"]["decode_bs1"]
    tight = {hi_items[i][0] for i in range(1, len(hi_items))
             if (hi_items[i - 1][1] - hi_items[i][1]) / hi_items[i][1] < band1}
    placed = spread_labels(hi_items, gap_for(ax_hi, YHI, 8.0),
                           YHI[0] + 0.004, YHI[1] - 0.024)
    for cid, y in hi_items:
        st, yl = style(cid), placed[cid]
        x0 = X_T1 + (DX_LONE if cid == LONE else 0.0)
        # leaders go BEHIND the curves (zorder 3): the T = 1 -> T = 2 collapse must stay a
        # continuous line, and nothing opaque may sit on it.
        ax_hi.plot([x0, 0.198], [y, yl], transform=thi, color=st["color"],
                   linewidth=0.8, alpha=0.55, zorder=3, clip_on=False,
                   linestyle=(0, (1.0, 2.2)))
        ax_hi.plot([0.206], [yl], transform=thi, marker=st["marker"], markersize=4.6,
                   color=st["color"], markerfacecolor=(c["surface"] if st["tier"] != "winner"
                                                       else st["color"]),
                   markeredgecolor=st["color"], markeredgewidth=1.5,
                   linestyle="none", zorder=8, clip_on=False)
        extra = "  ‡" if cid in tight else ""
        tail = "   ·  single point" if cid == LONE else ""
        ax_hi.text(0.220, yl, f"{lab[cid]}   {y:.3f}x{extra}{tail}", transform=thi,
                   fontsize=8.0, color=c["secondary"], va="center", ha="left", zorder=8)
    ax_hi.text(0.020, 0.988, "THE NINE CHAINS DRAWN HERE, at T = 1, ranked   ·   "
               "18 configurations were timed at this regime",
               transform=ax_hi.transAxes, fontsize=8.0, color=c["primary"], va="top",
               ha="left", zorder=8)

    # ---- key, in the upper panel's right half (it carries one data column) -------
    key_x, key_y, key_dy = 0.500, 0.900, 0.056

    def key_row(i, draw, text, indent=0.0, bold=False):
        y = key_y - key_dy * i
        if draw is not None:
            draw(y)
        ax_hi.text(key_x + 0.034 + indent, y, text, fontsize=8.0,
                   color=c["primary"] if bold else c["secondary"],
                   va="center", ha="left", transform=ax_hi.transAxes, zorder=9)

    key_row(0, lambda y: ax_hi.plot([key_x, key_x + 0.024], [y, y], color=REF,
                                    linewidth=1.1, transform=ax_hi.transAxes, zorder=9),
            "1.000x = the all-unfused layer (A_all_unfused). Thin SOLID rule — no data "
            "series is solid grey.")
    key_row(1, lambda y: ax_hi.fill_between([key_x, key_x + 0.024], [y - 0.013] * 2,
                                            [y + 0.013] * 2, color=c["rule"], alpha=0.28,
                                            transform=ax_hi.transAxes, zorder=9, linewidth=0),
            "shaded band = ± p90 of |run1 − run2| ÷ min(run1, run2) over every candidate "
            "timed in that regime.")
    key_row(2, None, "Pass-to-pass repeatability, NOT a significance band. Around 1.000x, "
                     "so it is off-scale in this segment.")
    key_row(3, lambda y: (ax_hi.plot([key_x, key_x + 0.024], [y, y], color="#2a78d6",
                                     linewidth=1.9, solid_capstyle="round",
                                     transform=ax_hi.transAxes, zorder=9),
                          ax_hi.plot([key_x + 0.012], [y], marker="o", markersize=5.0,
                                     color="#2a78d6", markeredgecolor=c["surface"],
                                     markeredgewidth=1.0, linestyle="none",
                                     transform=ax_hi.transAxes, zorder=9)),
            "5 per-regime winners — solid, coloured, ONE MARKER SHAPE EACH (o  s  ◆  ▲  ▼).")
    key_row(4, lambda y: (ax_hi.plot([key_x, key_x + 0.024], [y, y], color="#63625d",
                                     linewidth=1.25, dashes=[5.0, 2.5],
                                     transform=ax_hi.transAxes, zorder=9),
                          ax_hi.plot([key_x + 0.012], [y], marker="s", markersize=4.1,
                                     markerfacecolor=c["surface"], markeredgecolor="#63625d",
                                     markeredgewidth=1.3, linestyle="none",
                                     transform=ax_hi.transAxes, zorder=9)),
            "3 tie-set co-leaders — thin, dark grey, dashed, open markers. Shape and dash, "
            "not hue, are")
    key_row(5, None, "the greyscale key: this figure does not require colour.")
    key_row(6, lambda y: ax_hi.plot([key_x + 0.012], [y], marker="D", markersize=7.0,
                                    markerfacecolor=c["surface"], markeredgecolor=LONE_COLOR,
                                    markeredgewidth=1.9, linestyle="none",
                                    transform=ax_hi.transAxes, zorder=9),
            "R_f1_f10_f11ab (= #1+#10+#11a+#11b in the fan) — ONE POINT at T = 1, no line.")
    key_row(7, None, f"Nudged {LONE_DX:.0f} px right of #1+#6+#9, dotted tie-back to T = 1, "
                     "so the two do not merge.")
    key_row(8, lambda y: ax_hi.plot([key_x + 0.012], [y], marker="v", markersize=5.0,
                                    color=c["rule"], linestyle="none",
                                    transform=ax_hi.transAxes, zorder=9),
            "the 3 regimes with a resolved unique winner (all #1+#6+#9) — marked at the top "
            "of the")
    key_row(9, None, "middle panel. Elsewhere 2–14 chains are tied.")
    key_row(10, None, f"‡  gap to the chain above it is INSIDE the ± {100 * band1:.3f} % "
                      "band at T = 1 — the widest in the figure.", bold=True)
    key_row(11, None, f"Only {raw['resolved']} of the {raw['n_points']} points drawn here "
                      "resolve against A_all_unfused at all; none of the nine at T = 1 do.",
             bold=True)

    # ---- middle panel furniture --------------------------------------------------
    for reg in REGIMES:
        if len(d["tiesets"][reg]) == 1:
            ax_lo.plot([T[reg]], [YLO[1] - 0.0038], marker="v", markersize=5.0,
                       color=c["rule"], linestyle="none", zorder=8)

    null_T = [T[r] for r in REGIMES if UNFUSED in d["tiesets"][r]]
    n_from, n_to = min(null_T) / 1.14, max(null_T) * 1.34
    y_br = YLO[1] - 0.0175                      # ABOVE the data, not on the global minimum
    ax_lo.plot([n_from, n_to], [y_br, y_br], color=c["rule"], linewidth=1.0, zorder=8)
    for t in (n_from, n_to):
        ax_lo.plot([t, t], [y_br, y_br - 0.0032], color=c["rule"], linewidth=1.0, zorder=8)
    ax_lo.text(n_to, y_br + 0.0036,
               "tie set here contains A_all_unfused itself — "
               "\"no measurable difference from doing nothing\"",
               fontsize=8.0, color=c["secondary"], va="bottom", ha="right", zorder=8)

    ax_lo.text(0.040, 0.215, "NOT THE WHOLE FIELD — "
               f"{len(d['undrawn_everywhere'])} of the "
               f"{len(d['all_gain']) - 1 - len(d['undrawn_t1_only'])} chains measured at "
               "every regime are never drawn:",
               transform=ax_lo.transAxes, fontsize=8.0, color=c["primary"],
               va="center", ha="left", zorder=9)
    ax_lo.text(0.040, 0.150,
               ", ".join(d["all_label"][x] for x in d["undrawn_everywhere"]) + "   (plus "
               + ", ".join(d["all_label"][x] for x in d["undrawn_t1_only"]) + ", T = 1 only).",
               transform=ax_lo.transAxes, fontsize=8.0, color=c["secondary"],
               va="center", ha="left", zorder=9)
    if d["top_undrawn"]:
        tu = d["top_undrawn"]
        ax_lo.text(0.040, 0.085,
                   f"At T = 1, {d['all_label'][tu]} alone is "
                   f"{d['all_gain'][tu]['decode_bs1']:.4f}x — ahead of "
                   f"{d['top_undrawn_beats']} of the nine drawn here. See the notes.",
                   transform=ax_lo.transAxes, fontsize=8.0, color=c["secondary"],
                   va="center", ha="left", zorder=9)

    # ---- right gutter: every chain at T = 8192, de-collided with leaders ---------
    lo_items = [(cid, gain[cid]["prefill_t8192"]) for cid in CHAIN_ORDER
                if "prefill_t8192" in gain[cid]]
    placed = spread_labels(lo_items, gap_for(ax_lo, YLO, 8.0),
                           YLO[0] + 0.006, YLO[1] - 0.028)
    ax_lo.text(1.030, max(placed.values()) + gap_for(ax_lo, YLO, 8.0) * 0.92, "at T = 8192",
               transform=tlo, fontsize=8.0, color=c["primary"], va="center", ha="left",
               clip_on=False)
    for cid, y in lo_items:
        st, yl = style(cid), placed[cid]
        ax_lo.plot([X_T8192, 1.012, 1.030], [y, y, yl], transform=tlo, color=st["color"],
                   linewidth=0.8, alpha=0.70, zorder=5, clip_on=False)
        ax_lo.plot([1.040], [yl], transform=tlo, marker=st["marker"], markersize=4.6,
                   color=st["color"],
                   markerfacecolor=(st["color"] if st["tier"] == "winner" else c["surface"]),
                   markeredgecolor=st["color"], markeredgewidth=1.3,
                   linestyle="none", zorder=8, clip_on=False)
        ax_lo.text(1.056, yl, f"{lab[cid]}  {y:.3f}x", transform=tlo, fontsize=8.0,
                   color=c["secondary"], va="center", ha="left", zorder=8, clip_on=False)

    # ---- magnifier: bracket the magnified band in the middle panel, then connect --
    for yv in YMAG:
        ax_lo.plot([0.0, 0.030], [yv, yv], transform=tlo, color=c["rule"], linewidth=1.1,
                   zorder=8)
    ax_lo.plot([0.030, 0.030], list(YMAG), transform=tlo, color=c["rule"], linewidth=1.1,
               zorder=8)
    ax_lo.text(0.004, YMAG[0] - 0.0044, "this band is magnified below", transform=tlo,
               fontsize=8.0, color=c["secondary"], va="top", ha="left", zorder=8)
    for xa, xb in ((0.0, 0.0), (1.0, 1.0)):
        fig.add_artist(ConnectionPatch(
            xyA=(xa, (YMAG[0] - YLO[0]) / (YLO[1] - YLO[0])), coordsA=ax_lo.transAxes,
            xyB=(xb, 1.0), coordsB=ax_mg.transAxes,
            color=c["rule"], linewidth=0.9, linestyle=(0, (3, 2.5)), zorder=1))
    ax_mg.text(0.330, 0.975, "MAGNIFIER — the same eleven regimes, y stretched "
               f"{(YLO[1] - YLO[0]) / (YMAG[1] - YMAG[0]):.1f}× over "
               f"{YMAG[0]:.3f}–{YMAG[1]:.3f}, where eight of the",
               transform=ax_mg.transAxes, fontsize=8.0, color=c["primary"], va="top",
               ha="left", zorder=9)
    ax_mg.text(0.330, 0.905, "nine chains sit from T = 8 on. A curve that leaves this panel "
               "is off-scale here, not missing —",
               transform=ax_mg.transAxes, fontsize=8.0, color=c["secondary"], va="top",
               ha="left", zorder=9)
    ax_mg.text(0.330, 0.835, "read it in the panel above.",
               transform=ax_mg.transAxes, fontsize=8.0, color=c["secondary"], va="top",
               ha="left", zorder=9)

    ax_mg.set_xlabel("total tokens T per call   (decode: batch size at kv 4096   ·   "
                     "prefill: sequence length, shaded)", fontsize=9.5,
                     color=c["secondary"])
    ax_lo.set_ylabel("whole-layer speedup vs the all-unfused layer", fontsize=9.5,
                     color=c["secondary"])
    # decode / prefill named once, in the empty foot of the upper panel: the dotted boundary
    # at T = 1448 runs through all three panels, so one label serves all three.
    ax_hi.text(PREFILL_FROM / 1.10, YHI[0] + 0.040 * (YHI[1] - YHI[0]), "decode",
               fontsize=8.5, color=c["secondary"], ha="right", va="bottom")
    ax_hi.text(PREFILL_FROM * 1.10, YHI[0] + 0.040 * (YHI[1] - YHI[0]), "prefill",
               fontsize=8.5, color="#5b7387", ha="left", va="bottom")

    # ---- the 9 x 11 matrix -------------------------------------------------------
    ax_tab.set_facecolor(c["surface"])
    ax_tab.set_xlim(0, 1)
    ax_tab.set_ylim(0, 1)
    ax_tab.axis("off")
    colx = [0.340 + 0.0600 * (i + 1) for i in range(11)]
    head_y = 0.945

    ax_tab.text(0.024, head_y, "chain", fontsize=8.0, color=c["primary"], ha="left",
                va="bottom", transform=ax_tab.transAxes)
    ax_tab.text(0.130, head_y, "config", fontsize=8.0, color=c["primary"], ha="left",
                va="bottom", transform=ax_tab.transAxes)
    ax_tab.text(0.208, head_y, "role", fontsize=8.0, color=c["primary"], ha="left",
                va="bottom", transform=ax_tab.transAxes)
    for x, t in zip(colx, TVALS):
        ax_tab.text(x, head_y, f"T={t}", fontsize=8.0, color=c["primary"], ha="right",
                    va="bottom", transform=ax_tab.transAxes)
    ax_tab.plot([0.0, 1.0], [head_y - 0.030] * 2, color=c["rule"], linewidth=0.9,
                transform=ax_tab.transAxes, clip_on=False, zorder=3)

    for i, cid in enumerate(CHAIN_ORDER):
        y = head_y - 0.082 - 0.0755 * i
        st = style(cid)
        ax_tab.plot([0.012], [y + 0.016], marker=st["marker"], markersize=4.6,
                    color=st["color"],
                    markerfacecolor=(st["color"] if st["tier"] == "winner" else c["surface"]),
                    markeredgecolor=st["color"], markeredgewidth=1.4,
                    transform=ax_tab.transAxes, clip_on=False, linestyle="none")
        ax_tab.text(0.024, y, lab[cid], fontsize=8.0, color=c["secondary"],
                    ha="left", va="bottom", transform=ax_tab.transAxes)
        ax_tab.text(0.130, y, cid, fontsize=8.0, color=c["secondary"], ha="left",
                    va="bottom", transform=ax_tab.transAxes)
        ax_tab.text(0.208, y, role[cid], fontsize=8.0,
                    color=c["secondary"] if st["tier"] == "winner" else "#8a8882",
                    ha="left", va="bottom", transform=ax_tab.transAxes)
        for x, reg in zip(colx, REGIMES):
            v = gain[cid].get(reg)
            if v is None:
                ax_tab.text(x - 0.026, y, "—", fontsize=8.0, color=c["grid"], ha="center",
                            va="bottom", transform=ax_tab.transAxes)
                continue
            win = d["winner"][reg] == cid
            ax_tab.text(x - 0.010, y, f"{v:.4f}x", fontsize=8.0,
                        color=c["primary"] if win else c["secondary"],
                        ha="right", va="bottom", transform=ax_tab.transAxes)
            if win:
                ax_tab.text(x, y, "◀", fontsize=8.0, color=c["primary"], ha="right",
                            va="bottom", transform=ax_tab.transAxes)

    foot_y = head_y - 0.082 - 0.0755 * len(CHAIN_ORDER) - 0.010
    ax_tab.plot([0.0, 1.0], [foot_y + 0.050] * 2, color=c["rule"], linewidth=0.9,
                transform=ax_tab.transAxes, clip_on=False, zorder=3)
    ax_tab.text(0.024, foot_y, "◀ = regime winner over all 14–18 candidates    ·    "
                "the report's own verdict", fontsize=8.0, color=c["primary"], ha="left",
                va="bottom", transform=ax_tab.transAxes)
    for x, reg in zip(colx, REGIMES):
        v = d["verdict"][reg]
        ax_tab.text(x - 0.010, foot_y, v, fontsize=8.0,
                    color=c["primary"] if v == "SEPARATED" else c["secondary"],
                    ha="right", va="bottom", transform=ax_tab.transAxes)
    ax_tab.text(0.024, foot_y - 0.066, "shaded-band half-width (± p90, % of the layer)",
                fontsize=8.0, color=c["secondary"], ha="left", va="bottom",
                transform=ax_tab.transAxes)
    for x, reg in zip(colx, REGIMES):
        ax_tab.text(x - 0.010, foot_y - 0.066, f"{100 * d['band'][reg]:.3f} %",
                    fontsize=8.0, color=c["secondary"], ha="right", va="bottom",
                    transform=ax_tab.transAxes)
    also = ", ".join(d["all_label"][x] for x in d["undrawn_everywhere"])
    ax_tab.text(0.024, foot_y - 0.140,
                "*  that regime's tie set contains A_all_unfused itself, so no gain is "
                f"resolved there.  By that same criterion {also} are co-leaders too — and "
                "none of them is drawn above.",
                fontsize=8.0, color=c["secondary"], ha="left", va="bottom",
                transform=ax_tab.transAxes)

    # ---- titles and notes --------------------------------------------------------
    n_below = sum(1 for cid in CHAIN_ORDER for r in REGIMES
                  if r != "decode_bs1" and gain[cid].get(r, 2.0) < 1.0)
    n_cells = sum(1 for cid in CHAIN_ORDER for r in REGIMES
                  if r != "decode_bs1" and r in gain[cid])
    d512 = gain["D_f6"]["decode_bs512"]
    n512 = len(d["tiesets"]["decode_bs512"])
    fig.text(0.062, 0.984, "GLM-5.2 MoE decoder layer on NVIDIA H200 "
             "— nine fusion chains, in every regime",
             fontsize=16, color=c["primary"], ha="left", va="top")
    tu = d["top_undrawn"]
    sub = (
        "The five chains best_chain_vs_T.png names as per-regime winners — #1+#6+#9, #3+#10, "
        "#6, #3, #1 — plus four chains from its TIE SETS, which that figure reports only as "
        "counts, never by name. Each is drawn across all eleven regimes, including the ones "
        "where it loses; R (#1+#10+#11a+#11b) exists only at T = 1, so it is a single point, "
        "not a curve. The pointwise maximum of these nine is that figure's single curve, "
        f"exactly, at all eleven regimes.   {n_below} of the {n_cells} T ≥ 2 cells are below "
        "1.000x: the greedy chain that wins decode_bs1 by 1.289x is the worst of the nine at "
        f"prefill_t8192 (0.939x), and #6 — the WINNER at T = 512 ({d512:.4f}x, {n512} tied) — "
        f"is the worst of all fourteen at T = 2048 (0.9367x).   NOT THE WHOLE FIELD: 14 "
        "configurations were timed at every regime and 18 at T = 1; nine chains are drawn. "
        f"{len(d['undrawn_everywhere'])} measured chains are never drawn, and at T = 1 the "
        f"best of them ({d['all_label'][tu]}, {d['all_gain'][tu]['decode_bs1']:.4f}x) "
        f"outranks {d['top_undrawn_beats']} of the nine here.   Only {raw['resolved']} of the "
        f"{raw['n_points']} points resolve against the unfused layer at all. Colour is not "
        "required: every series carries its own marker shape as well as its own hue.")
    fig.text(0.062, 0.9645, textwrap.fill(sub, 188),
             fontsize=9.5, color=c["secondary"], ha="left", va="top", linespacing=1.5)

    rr = raw["ratio"]
    re_ = raw["relerr"]
    notes = [
        "Whole-layer speedup = best_ms(A_all_unfused) ÷ best_ms(this chain), where best_ms "
        "= min(run1_ms, run2_ms) and run1/run2 are that configuration's median over the 8 "
        "interleaved rounds of pass 1 and of pass 2 (protocol: 2 passes × 8 rounds × 15 "
        "reps; within a round every candidate is timed once in an order that reverses on "
        "odd rounds). The denominator is the whole S3–S11 + shared-expert subgraph — "
        "attention core, MLA projections and the DSA indexer are excluded, and MoE "
        "dispatch-layout construction sits outside the timed region.",

        "WHICH NINE, AND WHY. best_chain_vs_T.png draws one curve — the per-regime winner — "
        "and names five chains: #1+#6+#9, #3+#10, #6, #3, #1. Its tie-set column reports "
        "only counts (\"3 chains tied\", \"14 chains tied — incl. do nothing\"), so the four "
        "co-leaders drawn here — #3+#9, #3+#8, #10 and #1+#10+#11a+#11b — had to be "
        "recovered from the CSV's tied_with_best_run1/run2 flags. The selection rule is "
        "therefore \"the chains best_chain_vs_T.png resolves\", NOT \"the top nine\". Five "
        "measured chains are consequently absent: #8, #9, #5, #4 and #11b, plus #11a+#11b, "
        "#10+#11a+#11b and #8+#11a+#11b which exist only at decode_bs1. Several outrank "
        "drawn chains: at T=1 #4 is 1.1895x (4th of the eighteen, ahead of six of the nine "
        "here) and #10+#11a+#11b is 1.1656x; at T=8 #9 (1.0061x) beats #6 (1.0036x); at "
        "T=8192 #4 (1.0031x) beats #10 (1.0011x). The T = 1 fan ranks the nine drawn chains "
        "against each other, not against the field.",

        "This is a RATIO OF MEDIANS across separately-timed candidates, not a paired "
        "interleaved A/B. The harness also recorded a paired head-vs-unfused ratio at 8 of "
        "the 11 regimes and the two disagree by up to 5.6 points (T=4: 1.0503 here vs "
        "1.1060 paired p50; T=2: 1.0399 vs 1.0834). That paired statistic is deliberately "
        "kept out of this column and is not plotted here.",

        "The shaded band is NOT a significance band. Tie sets follow the LOG-11 §3 rule as "
        "implemented, whose exact wording lives in results/h200/layer_configurations.json → "
        "protocol.rule: \"within a round every candidate is timed once, in a fixed order "
        "that reverses on odd rounds; a winner is declared only when its gap to the "
        "runner-up exceeds the round-to-round spread of BOTH, in BOTH passes, and both "
        "passes name the same configuration -- otherwise the set is reported as TIED\". "
        "(LOG-11 §3's own prose is shorter — \"exceeds the round-to-round spread of both; "
        "otherwise the set is reported as tied\" — and does not carry the both-passes and "
        "same-configuration clauses; the JSON is the implemented rule.) Round-to-round "
        "spread = max − min over the 8 rounds of a pass. That rule adjudicates which chain "
        "LEADS; it never tests one chain against the unfused layer. Applying it pairwise to "
        "(chain, A_all_unfused) — derived here, a comparison the report does not itself "
        f"publish — resolves only {raw['resolved']} of the {raw['n_points']} points drawn "
        f"above; the other {raw['n_points'] - raw['resolved']} are not distinguishable from "
        "1.000x, including all nine at T = 1 and all eight at T = 2, 4 and 8. Taking the "
        "WIDEST round-to-round "
        "spread in each regime, over every configuration and both passes, that threshold is "
        + (f"{rr[0]:.1f}× to {rr[1]:.0f}× " if rr else "") +
        "the plotted band. Which aggregation is used matters, so it is stated: the MEDIAN "
        "spread over the same population is "
        + (f"{raw['med_ratio'][0]:.1f}×–{raw['med_ratio'][1]:.1f}× " if raw["med_ratio"]
           else "") +
        "the band, still wider at every regime; only the single narrowest configuration in a "
        "regime drops below it (as low as "
        + (f"{raw['min_ratio'][0]:.2f}×" if raw["min_ratio"] else "1×") +
        "). Leaving the band is not evidence of a win.",

        "Instrument, from results/h200/layer_configurations.json → fairness.timing."
        "calibration (floor / launch / tick) and env.calib_health (contended, "
        "launch_trusted): harness floor (empty timed region) 36.914 µs against a 10.318 µs "
        "launch, CUDA-event tick 0.032 µs matching 100 % of samples, calib_health.contended "
        "= false, launch_trusted = true. No --gpu was requested "
        "(fairness.gpu.selection.requested = null, applied = false) on an 8-GPU "
        "multi-tenant host, but the card it got is recorded and was clean: GPU-b2318e71, "
        "cuda_visible_devices = \"0\", torch_device_count = 1, 0.37 % of memory in use by "
        "others at probe, and README §0a logs it as 0 MiB used at start and end with no "
        "co-tenant on this card for the whole campaign (the tenant was on GPU2). The floor "
        "is added to BOTH arms of every ratio, so in the report's own words \"a ratio "
        "understates the true work ratio\"; it is 6.8 % of the layer at T=1 and 0.2 % at "
        "T=8192, so every curve is compressed toward 1.000x, most at small T. (The report "
        "header separately records a +42.185 µs floor against a 9.079 µs launch on GPU "
        "6c4cc3d3 — a different card from the one this sweep ran on.)",

        "Cold start: pass 1 opens with an excursion on the all-unfused baseline — the "
        "denominator of every point here — at the four smallest regimes. Round 0 = 1363 / "
        "1356 / 1514 / 1325 µs at T = 1 / 2 / 4 / 8, i.e. 2.5× / 2.4× / 1.9× / 1.1× the "
        "median of the remaining seven rounds (pass-1 spread 153 / 143 / 88 / 8 %). The "
        "8-round median absorbs it — pass-1 and pass-2 medians agree to under 1 % — so the "
        "plotted values are unaffected; the round-to-round spread does not, so pass 1 "
        "resolves nothing against the baseline at T = 1, 2, 4 or 8. At T=1 the excursion "
        "also lands on B_f3 (1.85×), C_f1 (1.60×), D_f6 (1.36×) and E_f10 (1.22×), and "
        "spares J, H, I, K and R.",

        "decode_bs1 drew from 18 candidate chains and every other regime from 14, which is "
        "why R_f1_f10_f11ab is a lone marker: the four prenorm=\"all\" configurations "
        "O/P/Q/R failed the independent fp32 reference of the whole subgraph everywhere "
        "except decode_bs1, and a failing configuration is excluded outright rather than "
        "timed. README §1.2(b) quotes that failure as \"rel_err 0.16–0.67 against tol "
        "0.02\"; "
        + (f"the raw correctness blocks in results/h200/layer_configurations.json actually "
           f"span {re_[0]:.3f}–{re_[1]:.3f} over the {re_[2]} (configuration × regime) "
           "checks, so both ends of the quoted range are wrong. " if re_ else "") +
        "The conclusion is unaffected — every value is ≫ tol 0.02. R also contains #11a, "
        "whose report status is \"unmeasurable on this device, not measured-and-lost\": at "
        "decode_bs1 the strict 1e-5 invariance screen recorded #11a as UNTESTED on BLOCK_M "
        "because no legal cross-boundary partner tile fits this device's shared memory, and "
        "the rule is \"Untested is not invariant. This report fails closed.\" The layer "
        "harness's own check runs at 2e-2, \"which is not the tolerance at which #11a "
        "fails\". So 1.2856x is a real measurement of a configuration that cannot be "
        "validated — and the CSV's notes field for O/P/Q/R is empty, so this is the only "
        "warning the reader gets.",

        "Session seam: T = 2, 4, 8 and 16 were measured on 2026-08-10 by "
        "run_bs_extra_h200.py (session 9bf19b39a381bc89, GPU-3aa19cef) and merged into the "
        "campaign file; T = 1, 32, 256, 512, 1024, 2048 and 8192 are the 2026-08-07 "
        "campaign on GPU-b2318e71. Every point is a within-session ratio, so no point is "
        "corrupted, but the segments joining T=1→2 and T=16→32 cross a session and a "
        "physical card.",

        "Comparable with best_chain_vs_T.png — same CSV, same numerator and denominator. NOT "
        "comparable with gain_vs_T.png, whose denominator is one two- or three-kernel "
        "unfused chain rather than the whole layer: a 2.2x win on a kernel that is 3 % of "
        "the layer is a 1.02x layer.   ·   source: "
        "report_glm52_h200/layer_optimal_per_regime.csv (158 rows) from "
        "results/h200/layer_configurations.json — 18 configurations at decode_bs1 and 14 at "
        "each other regime × 2 independent passes × 8 interleaved rounds × 15 reps.",
    ]
    wrapped = "\n".join(line for n in notes for line in textwrap.wrap(n, 250) or [""])
    fig.text(0.062, 0.009, wrapped, fontsize=7.3, color=c["secondary"], ha="left",
             va="bottom", linespacing=1.56)

    p = OUT / "chain_gain_vs_T.png"
    fig.savefig(p, dpi=140, facecolor=c["surface"])
    print(f"wrote {p}  ({len(CHAIN_ORDER)} chains, {len(REGIMES)} regimes, "
          f"{sum(len(v) for v in gain.values())} points, "
          f"{len(wrapped.splitlines())} note lines)")


if __name__ == "__main__":
    main()
