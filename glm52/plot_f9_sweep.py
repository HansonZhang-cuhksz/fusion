"""Plot fusion #9 (down-GEMM + expert merge + ResAdd2) gain vs decode batch size, and a
combined figure against fusion #8.

Two deliverables:
  f9_gain_vs_batch.png   panel A: #9's gain against BOTH unfused baselines (2-kernel and
                         3-kernel) — the choice is worth up to 8x and is shown, not buried;
                         panel B: the fused time and the 2-kernel baseline it should be
                         judged against.
  f8_f9_gain_vs_batch.png  #8 and #9 (each against its honest baseline) on one panel.

Every panel carries the measured ±2.2 % between-run uncertainty envelope. The narrow band on
each line is the min–max over 5 interleaved rounds, which is WITHIN-run only and ~20x
smaller — it is drawn because it is real, not because it bounds the uncertainty.

Only the two validated categorical slots are used (blue/orange). A third slot would have
tripped a light-mode contrast WARN, so the 3-kernel absolute time is omitted from panel B
rather than shipped needing relief.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

from glm52.plot_f8_sweep import THEME  # noqa: E402
from glm52.plot_f8_sweep import load as load_f8  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPORT = ROOT / "report"
BETWEEN_RUN = 0.022


def load_f9() -> list[dict]:
    by_T: dict[int, dict] = {}
    for p in sorted(RESULTS.glob("f9_sweep_lane*.json")):
        for r in json.loads(p.read_text())["rows"]:
            by_T[r["T"]] = dict(r, source="lane sweep")
    for p in sorted(RESULTS.glob("f9_sweep_verify*.json")):
        for r in json.loads(p.read_text())["rows"]:
            if r["T"] in by_T:
                print(f"  T={r['T']}: wide re-tune supersedes "
                      f"({by_T[r['T']]['gain_vs_2k']:.4f} -> {r['gain_vs_2k']:.4f})")
            by_T[r["T"]] = dict(r, source="wide re-tune (486 cfgs/side)")
    return [by_T[t] for t in sorted(by_T)]


def write_csv(rows: list[dict]) -> Path:
    out = REPORT / "f9_gain_vs_batch.csv"
    fields = ["T", "rows", "rows_per_expert", "accum_MB", "fused_ms", "unfused2_ms",
              "unfused3_ms", "gain_vs_2k", "gain_vs_2k_min", "gain_vs_2k_max",
              "gain_vs_3k", "gain_vs_3k_min", "gain_vs_3k_max", "gain_spread_pct",
              "unfused_cfg", "moe_sum_res0_cfg", "moe_sum_res1_cfg", "resadd_cfg",
              "fused_cfg", "seed_cfg", "source"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            d = dict(r)
            for k in ("unfused_cfg", "moe_sum_res0_cfg", "moe_sum_res1_cfg", "resadd_cfg",
                      "fused_cfg", "seed_cfg"):
                if isinstance(d.get(k), dict):
                    d[k] = " ".join(f"{a}{b}" for a, b in d[k].items())
            for k in fields:
                if isinstance(d.get(k), float):
                    d[k] = round(d[k], 4)
            w.writerow(d)
    return out


def _frame(c, nrows=2):
    if nrows == 2:
        fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.9, 1], hspace=0.10))
        fig.subplots_adjust(top=0.825, left=0.095, right=0.975, bottom=0.115)
    else:
        fig, ax = plt.subplots(figsize=(10.0, 5.8))
        axes = [ax]
        fig.subplots_adjust(top=0.775, left=0.095, right=0.975, bottom=0.135)
    fig.patch.set_facecolor(c["surface"])
    for a in (axes if nrows == 2 else axes):
        a.set_facecolor(c["surface"])
        a.grid(True, which="major", color=c["grid"], linewidth=0.8, zorder=0)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(c["grid"])
        a.tick_params(colors=c["secondary"], labelsize=9, length=0)
    return fig, axes


def _xaxis(ax, T, c):
    ax.set_xscale("log", base=2)
    ticks = [t for t in T if t in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlim(min(T) * 0.8, max(T) * 1.25)
    ax.set_xlabel("decode batch size T        (rows per expert = T × 8 / 256)",
                  color=c["secondary"], fontsize=10)


def _band_and_rule(ax, c, ylo):
    ax.axhspan(ylo, 1.0, color=c["harm"], zorder=0)
    ax.axhline(1.0, color=c["rule"], linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
    ax.axhspan(1 - BETWEEN_RUN, 1 + BETWEEN_RUN, color=c["rule"], alpha=0.10, zorder=1)


def plot_f9(rows, mode) -> Path:
    c = THEME[mode]
    T = [r["T"] for r in rows]
    g2 = [r["gain_vs_2k"] for r in rows]
    g3 = [r["gain_vs_3k"] for r in rows]
    fig, (ax, ax2) = _frame(c, 2)

    allv = g2 + g3
    ylo, yhi = min(allv) - 0.02, max(allv) + 0.03
    ax.set_ylim(ylo, yhi)
    _band_and_rule(ax, c, ylo)
    for series, col, ls, mk, lab in (
            (g3, c["s2"], (0, (4, 2.5)), "s", "vs 3-kernel baseline  (GEMM → moe_sum → resadd)"),
            (g2, c["s1"], "-", "o", "vs 2-kernel baseline  (GEMM → moe_sum+residual)")):
        ax.plot(T, series, color=col, linewidth=2.0, linestyle=ls, zorder=4, label=lab)
        ax.plot(T, series, mk, color=col, markersize=5.0, markeredgecolor=c["surface"],
                markeredgewidth=1.5, zorder=5)
    for r, lo_k, hi_k, col in ((rows, "gain_vs_2k_min", "gain_vs_2k_max", c["s1"]),
                               (rows, "gain_vs_3k_min", "gain_vs_3k_max", c["s2"])):
        ax.fill_between(T, [x[lo_k] for x in r], [x[hi_k] for x in r], color=col,
                        alpha=0.16, linewidth=0, zorder=3)
    leg = ax.legend(frameon=False, fontsize=9, loc="lower left", handletextpad=0.6)
    for t in leg.get_texts():
        t.set_color(c["secondary"])
    ax.set_ylabel("gain   (unfused / fused)", color=c["secondary"], fontsize=10)
    ax.text(0.006, 0.955, "shaded ±2.2 % = between-run uncertainty; differences inside it "
            "are NOT resolved", transform=ax.transAxes, fontsize=8.5,
            color=c["secondary"], va="top")

    ax2.plot(T, [r["unfused2_ms"] for r in rows], color=c["s1"], linewidth=2.0,
             label="unfused 2-kernel")
    ax2.plot(T, [r["unfused2_ms"] for r in rows], "o", color=c["s1"], markersize=4.4,
             markeredgecolor=c["surface"], markeredgewidth=1.3)
    ax2.plot(T, [r["fused_ms"] for r in rows], color=c["s2"], linewidth=2.0,
             linestyle=(0, (4, 2.5)), label="fused  (seed + atomic merge)")
    ax2.plot(T, [r["fused_ms"] for r in rows], "s", color=c["s2"], markersize=4.0,
             markeredgecolor=c["surface"], markeredgewidth=1.3)
    ax2.set_yscale("log")
    ax2.set_ylabel("kernel time (ms)", color=c["secondary"], fontsize=10)
    leg2 = ax2.legend(frameon=False, fontsize=9, loc="upper left", ncols=2,
                      handletextpad=0.6, columnspacing=2.0)
    for t in leg2.get_texts():
        t.set_color(c["secondary"])
    _xaxis(ax2, T, c)

    fig.text(0.095, 0.968, "Fusion #9 (down-GEMM + expert merge + ResAdd2): no resolvable gain\n"
                           "at any batch size, and clearly harmful from T ≈ 896",
             color=c["primary"], fontsize=14, fontweight="semibold", va="top", linespacing=1.35)
    fig.text(0.095, 0.884, "GLM-5.2 · MetaX C500 · bf16 · both sides independently retuned at every batch size\n"
                           "the 3-kernel baseline flatters the fusion by 0.2–0.5 % (2 % at T=1, where it "
                           "flips the sign)",
             fontsize=9, color=c["secondary"], va="top")
    out = REPORT / f"f9_gain_vs_batch{'' if mode == 'light' else '_dark'}.png"
    fig.savefig(out, dpi=170, facecolor=c["surface"])
    plt.close(fig)
    return out


def plot_combined(f8, f9, mode) -> Path:
    c = THEME[mode]
    T8 = [r["T"] for r in f8]
    T9 = [r["T"] for r in f9]
    g8 = [r["gain"] for r in f8]
    g9 = [r["gain_vs_2k"] for r in f9]
    fig, (ax,) = _frame(c, 1)
    allv = g8 + g9
    ylo, yhi = min(allv) - 0.02, max(allv) + 0.03
    ax.set_ylim(ylo, yhi)
    _band_and_rule(ax, c, ylo)
    ax.plot(T8, g8, color=c["s1"], linewidth=2.0, label="#8  down-GEMM + expert merge")
    ax.plot(T8, g8, "o", color=c["s1"], markersize=5.0, markeredgecolor=c["surface"],
            markeredgewidth=1.5)
    ax.plot(T9, g9, color=c["s2"], linewidth=2.0, linestyle=(0, (4, 2.5)),
            label="#9  + ResAdd2  (vs its 2-kernel baseline)")
    ax.plot(T9, g9, "s", color=c["s2"], markersize=4.6, markeredgecolor=c["surface"],
            markeredgewidth=1.5)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="lower left", handletextpad=0.6)
    for t in leg.get_texts():
        t.set_color(c["secondary"])
    ax.set_ylabel("gain   (unfused / fused)", color=c["secondary"], fontsize=10)
    ax.text(0.006, 0.955, "shaded ±2.2 % = between-run uncertainty; differences inside it "
            "are NOT resolved", transform=ax.transAxes, fontsize=8.5,
            color=c["secondary"], va="top")
    _xaxis(ax, T8, c)
    fig.text(0.095, 0.955, "Folding ResAdd2 into the fused down-GEMM (#8 → #9) is close to free",
             color=c["primary"], fontsize=14, fontweight="semibold", va="top")
    fig.text(0.095, 0.888, "GLM-5.2 · MetaX C500 · bf16 · both fusions measured against their "
                           "own independently tuned unfused chains",
             fontsize=9, color=c["secondary"], va="top")
    out = REPORT / f"f8_f9_gain_vs_batch{'' if mode == 'light' else '_dark'}.png"
    fig.savefig(out, dpi=170, facecolor=c["surface"])
    plt.close(fig)
    return out


def main() -> None:
    f9 = load_f9()
    if not f9:
        raise SystemExit("no results/f9_sweep_lane*.json yet")
    f8 = load_f8()
    REPORT.mkdir(exist_ok=True)
    print(f"#9: {len(f9)} points; #8: {len(f8)} points")
    print(f"wrote {write_csv(f9)}")
    for mode in ("light", "dark"):
        print(f"wrote {plot_f9(f9, mode)}")
        if f8:
            print(f"wrote {plot_combined(f8, f9, mode)}")


if __name__ == "__main__":
    main()
