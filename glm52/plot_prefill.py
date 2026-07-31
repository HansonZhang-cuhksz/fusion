"""Plot fusions #8 and #9 gain across the PREFILL range, T = 256 .. 262144.

These two kernels have no distinct "prefill mode" — they touch only the down-projection,
whose shapes depend on rows = T*8 and nothing else. So this is the same measurement as the
decode sweep, continued to larger T, and is presented separately only because the study's
vocabulary splits decode from prefill.

T = 262144 is the hard ceiling on this GPU: the problem allocates ~167 KB per token (act,
the [8T, 6144] unfused intermediate, and three [T, 6144] buffers) on top of the fixed 6.4 GB
w2, reaching 47 GiB of 64. T = 1M would need ~175 GB.

Points from `*_sweep_prefill_verify*.json` supersede the lane sweeps at the same T: the lane
sweep's tile search holds (warps, stages) at the seed and demonstrably under-tunes some
points, which the cross-campaign overlap check at T=256/512/1024 exposed.
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

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPORT = ROOT / "report"
BETWEEN_RUN = 0.022


def _load(prefix: str, gain_key: str) -> dict:
    """Lane sweeps first, then verify runs, so verify wins at any shared T (sorted order
    already puts 'lane' before 'verify')."""
    by_T: dict[int, dict] = {}
    for p in sorted(RESULTS.glob(f"{prefix}_sweep_prefill_*.json")):
        src = "wide re-tune" if "verify" in p.name else "lane sweep"
        for r in json.loads(p.read_text())["rows"]:
            if r["T"] in by_T and src == "wide re-tune":
                print(f"  {prefix} T={r['T']}: wide re-tune supersedes "
                      f"({by_T[r['T']][gain_key]:.4f} -> {r[gain_key]:.4f})")
            by_T[r["T"]] = dict(r, source=src)
    return by_T


def main() -> None:
    d8 = _load("f8", "gain")
    d9 = _load("f9", "gain_vs_2k")
    T = sorted(set(d8) & set(d9))
    if not T:
        raise SystemExit("no prefill results found")
    print(f"{len(T)} prefill points: {T}")

    REPORT.mkdir(exist_ok=True)
    out_csv = REPORT / "f8_f9_gain_vs_prefill.csv"
    fields = ["T", "rows", "rows_per_expert", "accum_MB",
              "f8_unfused_ms", "f8_fused_ms", "f8_gain", "f8_spread_pct", "f8_source",
              "f9_unfused2_ms", "f9_unfused3_ms", "f9_fused_ms", "f9_gain_vs_2k",
              "f9_gain_vs_3k", "f9_spread_pct", "f9_source"]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for t in T:
            a, b = d8[t], d9[t]
            w.writerow({
                "T": t, "rows": a["rows"],
                "rows_per_expert": round(a["rows_per_expert"], 1),
                "accum_MB": round(a["accum_MB"], 1),
                "f8_unfused_ms": round(a["unfused_ms"], 4),
                "f8_fused_ms": round(a["fused_ms"], 4),
                "f8_gain": round(a["gain"], 4),
                "f8_spread_pct": round(a.get("gain_spread_pct", 0), 3),
                "f8_source": a["source"],
                "f9_unfused2_ms": round(b["unfused2_ms"], 4),
                "f9_unfused3_ms": round(b.get("unfused3_ms", float("nan")), 4),
                "f9_fused_ms": round(b["fused_ms"], 4),
                "f9_gain_vs_2k": round(b["gain_vs_2k"], 4),
                "f9_gain_vs_3k": round(b.get("gain_vs_3k", float("nan")), 4),
                "f9_spread_pct": round(b.get("gain_spread_pct", 0), 3),
                "f9_source": b["source"]})
    print(f"wrote {out_csv}")

    for mode in ("light", "dark"):
        c = THEME[mode]
        g8 = [d8[t]["gain"] for t in T]
        g9 = [d9[t]["gain_vs_2k"] for t in T]
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.0, 7.6), sharex=True,
                                      gridspec_kw=dict(height_ratios=[1.9, 1], hspace=0.10))
        fig.patch.set_facecolor(c["surface"])
        fig.subplots_adjust(top=0.825, left=0.095, right=0.975, bottom=0.115)
        for a in (ax, ax2):
            a.set_facecolor(c["surface"])
            a.grid(True, which="major", color=c["grid"], linewidth=0.8, zorder=0)
            a.set_axisbelow(True)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                a.spines[s].set_color(c["grid"])
            a.tick_params(colors=c["secondary"], labelsize=9, length=0)

        ylo, yhi = min(g8 + g9) - 0.025, max(g8 + g9) + 0.03
        ax.set_ylim(ylo, yhi)
        ax.axhspan(ylo, 1.0, color=c["harm"], zorder=0)
        ax.axhline(1.0, color=c["rule"], linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
        ax.axhspan(1 - BETWEEN_RUN, 1 + BETWEEN_RUN, color=c["rule"], alpha=0.10, zorder=1)
        ax.plot(T, g8, color=c["s1"], linewidth=2.0, zorder=4,
                label="#8  down-GEMM + expert merge")
        ax.plot(T, g8, "o", color=c["s1"], markersize=5.2, markeredgecolor=c["surface"],
                markeredgewidth=1.5, zorder=5)
        ax.plot(T, g9, color=c["s2"], linewidth=2.0, linestyle=(0, (4, 2.5)), zorder=4,
                label="#9  + ResAdd2  (vs its 2-kernel baseline)")
        ax.plot(T, g9, "s", color=c["s2"], markersize=4.8, markeredgecolor=c["surface"],
                markeredgewidth=1.5, zorder=5)
        leg = ax.legend(frameon=False, fontsize=9.5, loc="lower left", handletextpad=0.6)
        for t_ in leg.get_texts():
            t_.set_color(c["secondary"])
        ax.set_ylabel("gain   (unfused / fused)", color=c["secondary"], fontsize=10)
        ax.text(0.006, 0.955, "shaded ±2.2 % = between-run uncertainty; differences inside it "
                "are NOT resolved", transform=ax.transAxes, fontsize=8.5,
                color=c["secondary"], va="top")
        ax.annotate(f"{g8[-1]:.3f}×", xy=(T[-1], g8[-1]), xytext=(-6, -16),
                    textcoords="offset points", ha="right", fontsize=9.5,
                    color=c["primary"], fontweight="medium")

        ax2.plot(T, [d8[t]["unfused_ms"] for t in T], color=c["s1"], linewidth=2.0,
                 label="#8 unfused chain")
        ax2.plot(T, [d9[t]["unfused2_ms"] for t in T], color=c["s2"], linewidth=2.0,
                 linestyle=(0, (4, 2.5)), label="#9 unfused chain (2-kernel)")
        ax2.set_yscale("log")
        ax2.set_ylabel("kernel time (ms)", color=c["secondary"], fontsize=10)
        leg2 = ax2.legend(frameon=False, fontsize=9, loc="upper left", ncols=2,
                          handletextpad=0.6, columnspacing=2.0)
        for t_ in leg2.get_texts():
            t_.set_color(c["secondary"])
        ax2.set_xscale("log", base=2)
        ax2.xaxis.set_major_locator(FixedLocator(T))
        ax2.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{int(v)//1024}K" if v >= 1024 else f"{int(v)}"))
        ax2.set_xlim(min(T) * 0.8, max(T) * 1.3)
        ax2.set_xlabel("prefill tokens T        (rows per expert = T × 8 / 256)",
                       color=c["secondary"], fontsize=10)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

        fig.text(0.095, 0.968, "At prefill scale both #8 and #9 lose 9–15 %:\n"
                               "there is no prefill size where either fusion pays",
                 color=c["primary"], fontsize=14, fontweight="semibold", va="top",
                 linespacing=1.35)
        fig.text(0.095, 0.884, "GLM-5.2 · MetaX C500 · bf16 · both sides independently retuned "
                               "at every point\nT=262144 is the memory ceiling (47 GiB of 64); "
                               "T=1M would need ~175 GB",
                 fontsize=9, color=c["secondary"], va="top")
        out = REPORT / f"f8_f9_gain_vs_prefill{'' if mode == 'light' else '_dark'}.png"
        fig.savefig(out, dpi=170, facecolor=c["surface"])
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
