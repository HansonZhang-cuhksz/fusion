"""Plot fusion #8's kernel-level gain against decode batch size.

Two panels, one axis each (never a dual-axis chart):
  A  gain = unfused/fused vs batch size, with the min-max band across the 5 interleaved
     rounds and a break-even reference line at 1.0. One series, so no legend box — the
     title names it; a handful of selective direct labels, never one per point.
  B  the two absolute times that produce that ratio, as a small multiple below.

Palette: the validated default categorical slots 1 and 2 (blue/orange), which pass every
gate in both modes — light ΔE 24.7 CVD / 33.6 normal, dark 26.8 / 31.8. Light and dark
figures are both emitted; the dark one uses that mode's own steps, not an inverted flip.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPORT = ROOT / "report"

THEME = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  grid="#e4e3df", s1="#2a78d6", s2="#eb6834", band="#2a78d6",
                  rule="#8d8b85", harm="#f2f1ed"),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 grid="#333331", s1="#3987e5", s2="#d95926", band="#3987e5",
                 rule="#6f6e69", harm="#232322"),
}


def load() -> list[dict]:
    """Lane sweeps, with any batch size that was re-tuned on the wide grid superseding it.

    The lane sweep varies (BM,BN,BK) with warps/stages held at the seed. At T=896/1024 the
    fused optimum lives at a different (warps, stages) — the wide re-tune found
    BM64/BN128/BK64/w16/s2 at 6.07 ms where the sweep reported 6.97 ms — so those points are
    replaced by the wide-grid values (486 configs per side, both sides equally). T=768 was
    re-tuned as a control and agreed to 0.2 %, which is why the mid-range is left as swept.
    """
    by_T: dict[int, dict] = {}
    for p in sorted(RESULTS.glob("f8_sweep_lane*.json")):
        for r in json.loads(p.read_text())["rows"]:
            by_T[r["T"]] = dict(r, source="lane sweep")
    for p in sorted(RESULTS.glob("f8_sweep_verify*.json")):
        for r in json.loads(p.read_text())["rows"]:
            prev = by_T.get(r["T"])
            if prev:
                print(f"  T={r['T']}: wide re-tune supersedes lane sweep "
                      f"({prev['gain']:.4f}x -> {r['gain']:.4f}x)")
            by_T[r["T"]] = dict(r, source="wide re-tune (486 cfgs/side)")
    return [by_T[t] for t in sorted(by_T)]


def write_csv(rows: list[dict]) -> Path:
    out = REPORT / "f8_gain_vs_batch.csv"
    fields = ["T", "rows", "rows_per_expert", "accum_MB", "unfused_ms", "fused_ms", "gain",
              "gain_min", "gain_max", "gain_spread_pct", "rel_err_unfused", "rel_err_fused",
              "unfused_cfg", "moe_sum_cfg", "fused_cfg", "source"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            d = dict(r)
            for k in ("unfused_cfg", "moe_sum_cfg", "fused_cfg"):
                d[k] = " ".join(f"{a}{b}" for a, b in d[k].items())
            for k in ("rows_per_expert", "accum_MB", "unfused_ms", "fused_ms", "gain",
                      "gain_min", "gain_max", "gain_spread_pct"):
                if k in d:
                    d[k] = round(float(d[k]), 4)
            w.writerow(d)
    return out


def plot(rows: list[dict], mode: str) -> Path:
    c = THEME[mode]
    T = [r["T"] for r in rows]
    gain = [r["gain"] for r in rows]
    lo = [r.get("gain_min", r["gain"]) for r in rows]
    hi = [r.get("gain_max", r["gain"]) for r in rows]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10.0, 7.6), sharex=True,
        gridspec_kw=dict(height_ratios=[1.9, 1], hspace=0.10))
    fig.patch.set_facecolor(c["surface"])
    fig.subplots_adjust(top=0.825, left=0.095, right=0.975, bottom=0.115)

    for a in (ax, ax2):
        a.set_facecolor(c["surface"])
        a.grid(True, which="major", color=c["grid"], linewidth=0.8, zorder=0)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(c["grid"])
        a.tick_params(colors=c["secondary"], labelsize=9, length=0)

    # --- Panel A: gain -----------------------------------------------------------------
    BETWEEN_RUN = 0.022   # measured between-run (fresh process + fresh tuning) variation
    ylo = min(min(lo), min(gain)) - 0.02
    yhi = max(max(hi), max(gain)) + 0.025
    ax.set_ylim(ylo, yhi)
    ax.axhspan(ylo, 1.0, color=c["harm"], zorder=0)        # region where fusion hurts
    ax.axhline(1.0, color=c["rule"], linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
    ax.fill_between(T, lo, hi, color=c["band"], alpha=0.18, linewidth=0, zorder=3)
    ax.plot(T, gain, color=c["s1"], linewidth=2.0, zorder=4)
    ax.plot(T, gain, "o", color=c["s1"], markersize=5.4, markeredgecolor=c["surface"],
            markeredgewidth=1.6, zorder=5)

    # No break-even LINE and no break-even range: across most of the sweep the curve sits
    # inside the between-run uncertainty, so a crossing point is not identifiable at all.
    # Mark only the first T that leaves the band — the one claim the data supports.
    exit_T = next((t for t, g in zip(T, gain) if 1.0 - g > BETWEEN_RUN), None)
    if exit_T:
        ax.annotate(f"leaves the uncertainty band at T={exit_T}:\n"
                    f"clearly harmful from here on",
                    xy=(exit_T, gain[T.index(exit_T)]), xytext=(-12, 26),
                    textcoords="offset points", ha="right", va="bottom", fontsize=9,
                    color=c["primary"], linespacing=1.3,
                    arrowprops=dict(arrowstyle="-", color=c["rule"], linewidth=1.0,
                                    shrinkA=0, shrinkB=4))

    # selective direct labels only: best, worst, and the last point
    marked = {T[gain.index(max(gain))], T[gain.index(min(gain))], T[-1]}
    for t, g in zip(T, gain):
        if t in marked:
            ax.annotate(f"{g:.3f}×", xy=(t, g),
                        xytext=(0, 12) if g >= 1 else (-4, -18),
                        textcoords="offset points",
                        ha="center" if g >= 1 else "right",
                        fontsize=9.5, color=c["primary"], fontweight="medium")

    # The plotted band is WITHIN-run only. Measured between-run variation (independent
    # process + independent tuning) is ~2.2 % — about 20x larger — so the plateau's shape is
    # not resolved and must not be read as structure. Draw that envelope explicitly.
    ax.axhspan(1 - BETWEEN_RUN, 1 + BETWEEN_RUN, color=c["rule"], alpha=0.10, zorder=1)
    ax.text(0.006, 0.885, "shaded ±2.2 % = between-run uncertainty (fresh process + fresh\n"
            "tuning); differences inside it are NOT resolved",
            transform=ax.transAxes, fontsize=8.5, color=c["secondary"], va="top",
            linespacing=1.35)

    ax.set_ylabel("gain   (unfused / fused)", color=c["secondary"], fontsize=10)
    ax.text(0.006, 0.055, "fusion is harmful below the dashed line", transform=ax.transAxes,
            fontsize=9, color=c["secondary"], style="italic")
    fig.text(0.095, 0.968, "Fusion #8 (MoE down-GEMM + expert merge): never clearly worth\n"
                           "anything, and clearly harmful from T ≈ 896",
             color=c["primary"], fontsize=14, fontweight="semibold", va="top", linespacing=1.35)
    fig.text(0.095, 0.884, "GLM-5.2 · MetaX C500 · bf16 · both sides independently retuned at every batch size\n"
                           "narrow band = min–max over 5 interleaved rounds (within-run only)",
             fontsize=9, color=c["secondary"], va="top")

    # --- Panel B: the two times behind the ratio ---------------------------------------
    # They coincide almost exactly below T~768, which IS the message; the dashed fused line
    # stays legible where it sits on top of the solid unfused one.
    u = [r["unfused_ms"] for r in rows]
    f = [r["fused_ms"] for r in rows]
    ax2.plot(T, u, color=c["s1"], linewidth=2.0, solid_capstyle="round",
             label="unfused  (GEMM → moe_sum)")
    ax2.plot(T, u, "o", color=c["s1"], markersize=4.4, markeredgecolor=c["surface"],
             markeredgewidth=1.3)
    ax2.plot(T, f, color=c["s2"], linewidth=2.0, linestyle=(0, (4, 2.5)),
             label="fused  (atomic merge epilogue)")
    ax2.plot(T, f, "s", color=c["s2"], markersize=4.0, markeredgecolor=c["surface"],
             markeredgewidth=1.3)
    ax2.set_yscale("log")
    ax2.set_ylabel("kernel time (ms)", color=c["secondary"], fontsize=10)
    ax2.set_xlabel("decode batch size T        (rows per expert = T × 8 / 256)",
                   color=c["secondary"], fontsize=10)
    leg = ax2.legend(frameon=False, fontsize=9, loc="upper left", ncols=2,
                     handletextpad=0.6, columnspacing=2.0, borderaxespad=0.2)
    for txt in leg.get_texts():
        txt.set_color(c["secondary"])

    ax2.set_xscale("log", base=2)
    # a readable subset of ticks -- 20 labels on a log axis collide at the top end
    ticks = [t for t in T if t in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)]
    ax2.xaxis.set_major_locator(FixedLocator(ticks))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax2.set_xlim(min(T) * 0.8, max(T) * 1.25)
    out = REPORT / (f"f8_gain_vs_batch{'' if mode == 'light' else '_dark'}.png")
    fig.savefig(out, dpi=170, facecolor=c["surface"])
    plt.close(fig)
    return out


def main() -> None:
    rows = load()
    if not rows:
        raise SystemExit("no results/f8_sweep_lane*.json found yet")
    REPORT.mkdir(exist_ok=True)
    print(f"{len(rows)} batch points: {[r['T'] for r in rows]}")
    print(f"wrote {write_csv(rows)}")
    for mode in ("light", "dark"):
        print(f"wrote {plot(rows, mode)}")


if __name__ == "__main__":
    main()
