"""Plot fusion #11b (Lazy Pre-Norm folded into the router GEMM) gain vs T.

Two figures matching the #8/#9 layout — one for the decode range (T=1..1024), one for the
prefill range (T=256..262144).

Panel B plots the **sum-of-squares redundancy factor**, which is the whole mechanism: the
fused kernel recomputes the entire row reduction once per n-tile, so redundancy =
ceil(256 / BLOCK_N). The tuner picks BLOCK_N per point, and the gain tracks that choice
almost perfectly — every point that landed on BLOCK_N=32 (8x redundancy) is a loss, and
every point that reached BLOCK_N=128 (2x) is a win.

Why the tuner cannot just always pick BLOCK_N=128: the router GEMM has N=256, so BLOCK_N=128
yields only 2 n-tiles. At small T there are few m-tiles too, so the grid collapses to a
handful of CTAs on 104 SMs. At decode sizes the kernel must buy parallelism with redundancy,
and that trade is what makes #11b decode-hostile and prefill-friendly.
"""

from __future__ import annotations

import csv
import glob
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


def load() -> dict[str, dict]:
    out: dict[str, dict] = {"decode": {}, "prefill": {}}
    for f in glob.glob(str(RESULTS / "f11b_*.json")):
        r = json.loads(Path(f).read_text())
        out[r["regime_kind"]][r["T"]] = r
    return out


def write_csv(rows: dict, kind: str) -> Path:
    out = REPORT / f"f11b_gain_vs_{'batch' if kind == 'decode' else 'prefill'}.csv"
    fields = ["T", "act_MB", "unfused_ms", "norm_only_ms", "gemm_only_ms", "fused_ms",
              "gain", "gain_min", "gain_max", "gain_spread_pct", "BLOCK_N", "n_tiles",
              "sq_redundancy", "rel_err_unfused", "rel_err_fused",
              "norm_cfg", "unfused_cfg", "fused_cfg", "worker"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for T in sorted(rows):
            r = dict(rows[T])
            r["BLOCK_N"] = r["fused_cfg"]["BLOCK_N"]
            for k in ("norm_cfg", "unfused_cfg", "fused_cfg"):
                r[k] = " ".join(f"{a}{b}" for a, b in r[k].items() if a != "eps")
            for k in ("act_MB", "unfused_ms", "norm_only_ms", "gemm_only_ms", "fused_ms",
                      "gain", "gain_min", "gain_max", "gain_spread_pct"):
                r[k] = round(float(r[k]), 4)
            w.writerow(r)
    return out


def plot(rows: dict, kind: str, mode: str) -> Path:
    c = THEME[mode]
    T = sorted(rows)
    gain = [rows[t]["gain"] for t in T]
    lo = [rows[t]["gain_min"] for t in T]
    hi = [rows[t]["gain_max"] for t in T]
    red = [rows[t]["sq_redundancy"] for t in T]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.0, 7.6), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2.0, 1], hspace=0.10))
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

    ylo, yhi = min(lo) - 0.03, max(hi) + 0.04
    ax.set_ylim(ylo, yhi)
    ax.axhspan(ylo, 1.0, color=c["harm"], zorder=0)
    ax.axhline(1.0, color=c["rule"], linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
    ax.axhspan(1 - BETWEEN_RUN, 1 + BETWEEN_RUN, color=c["rule"], alpha=0.10, zorder=1)
    ax.fill_between(T, lo, hi, color=c["s1"], alpha=0.18, linewidth=0, zorder=3)
    ax.plot(T, gain, color=c["s1"], linewidth=2.0, zorder=4)
    ax.plot(T, gain, "o", color=c["s1"], markersize=5.2, markeredgecolor=c["surface"],
            markeredgewidth=1.5, zorder=5)
    ax.set_ylabel("gain   (unfused / fused)", color=c["secondary"], fontsize=10)
    ax.text(0.006, 0.955, "shaded ±2.2 % = between-run uncertainty; differences inside it "
            "are NOT resolved", transform=ax.transAxes, fontsize=8.5,
            color=c["secondary"], va="top")
    best = max(range(len(T)), key=lambda i: gain[i])
    ax.annotate(f"{gain[best]:.3f}×", xy=(T[best], gain[best]), xytext=(0, 12),
                textcoords="offset points", ha="center", fontsize=9.5,
                color=c["primary"], fontweight="medium")

    ax2.step(T, red, where="mid", color=c["s2"], linewidth=2.0)
    ax2.plot(T, red, "s", color=c["s2"], markersize=4.6, markeredgecolor=c["surface"],
             markeredgewidth=1.3)
    ax2.set_yscale("log", base=2)
    ax2.set_yticks([2, 4, 8])
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}×"))
    ax2.set_ylim(1.5, 11)
    ax2.set_ylabel("sum-of-squares\nredundancy", color=c["secondary"], fontsize=10)
    ax2.text(0.006, 0.90, "= ceil(256 / BLOCK_N): how many times the fused k-loop redoes "
             "the row reduction", transform=ax2.transAxes, fontsize=8.5,
             color=c["secondary"], va="top")

    ax2.set_xscale("log", base=2)
    if kind == "decode":
        ticks = [t for t in T if t in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)]
        fmt = lambda v, _: f"{int(v)}"  # noqa: E731
        xlabel = "decode batch size T"
    else:
        ticks = T
        fmt = lambda v, _: f"{int(v)//1024}K" if v >= 1024 else f"{int(v)}"  # noqa: E731
        xlabel = "prefill tokens T"
    ax2.xaxis.set_major_locator(FixedLocator(ticks))
    ax2.xaxis.set_major_formatter(FuncFormatter(fmt))
    ax2.set_xlim(min(T) * 0.8, max(T) * 1.3)
    ax2.set_xlabel(xlabel, color=c["secondary"], fontsize=10)
    if kind == "prefill":
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    if kind == "decode":
        title = ("Fusion #11b (Lazy Pre-Norm → router GEMM) loses across the whole\n"
                 "decode range — the fused k-loop redoes the reduction 8×")
    else:
        title = ("Fusion #11b turns profitable once the GEMM is wide enough to afford\n"
                 "a big BLOCK_N — up to 1.41× at prefill scale")
    fig.text(0.095, 0.968, title, color=c["primary"], fontsize=14, fontweight="semibold",
             va="top", linespacing=1.35)
    fig.text(0.095, 0.884, "GLM-5.2 · MetaX C500 · bf16 · both sides independently retuned at "
                           "every T\nunfused = rmsnorm kernel → router GEMM; fused = one GEMM "
                           "reading un-normalized h1, rstd applied as an epilogue scale",
             fontsize=9, color=c["secondary"], va="top")
    out = REPORT / (f"f11b_gain_vs_{'batch' if kind == 'decode' else 'prefill'}"
                    f"{'' if mode == 'light' else '_dark'}.png")
    fig.savefig(out, dpi=170, facecolor=c["surface"])
    plt.close(fig)
    return out


def main() -> None:
    data = load()
    REPORT.mkdir(exist_ok=True)
    for kind in ("decode", "prefill"):
        rows = data[kind]
        if not rows:
            continue
        print(f"{kind}: {len(rows)} points")
        print(f"wrote {write_csv(rows, kind)}")
        for mode in ("light", "dark"):
            print(f"wrote {plot(rows, kind, mode)}")


if __name__ == "__main__":
    main()
