"""layer_optimal_per_regime.csv for the RTX 4060 port.

IMPORTANT DIFFERENCE FROM THE C500 REPORT. The C500 file's run1_ms / run2_ms / best_ms are
WHOLE-LAYER times from bench_layer.py, measured over fusion-set combinations that include
#6/#8/#9. None of that is reproducible here:

  * a GLM-5.2 MoE layer needs w13 (12.0 GB) + w2 (6.0 GB) = 18.0 GB of expert weights against
    ~7.4 GB usable, so the layer cannot be instantiated at exact spec on this device at all;
  * #6/#8/#9 were therefore never measured;
  * bench_layer.py was never run here -- it still carries C500 constants (SMEM 65536, warp 64).

So `layer_total_ms` and `speedup_vs_unfused` are LEFT EMPTY rather than modelled. What IS
measurable without a layer total is the absolute time each fusion set removes from a layer,
since that is just the sum of measured per-call deltas -- and that is what this file reports.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "rtx4060"
OUT = ROOT / "report_glm52_rtx4060"
REGIMES = ["decode_bs1", "decode_bs32", "decode_bs256", "prefill_t2048", "prefill_t8192"]

# Sites per GLM-5.2 decoder layer. #3 runs twice (input_layernorm and
# post_attention_layernorm, both fused_add_rmsnorm); #10 and #1 run once.
SITES = {"#3": 2, "#10": 1, "#1": 1}

def delta(fam: str, regime: str) -> float:
    """unfused - fused, in ms, for one call site."""
    if fam == "#3":
        d = json.loads((RES / "f03_resadd_rmsnorm.json").read_text())
        r = next(x for x in d["rows"] if x["regime"] == regime)
        return r["unfused_ms"] - r["fused_ms"]
    if fam == "#10":
        d = json.loads((RES / "f10_merge_resadd.json").read_text())
        r = next(x for x in d["rows"] if x["regime"] == regime)
        return r["unfused_ms"] - r["fused_ms"]
    if fam == "#1":
        if regime == "prefill_t8192":      # corrected: interleaved re-measure, see LOG-13 S9.1
            e = json.loads((RES / "exp1_f01_interleaved.json").read_text())
            return e["unfused_p50"] - e["fused_p50"]
        d = json.loads((RES / "f01_oproj_resadd.json").read_text())
        r = next(x for x in d["rows"] if x["regime"] == regime)
        return r["unfused_ms"] - r["fused_ms"]
    raise KeyError(fam)

SETS = [
    ("none (all unfused)", []),
    ("#3", ["#3"]),
    ("#10", ["#10"]),
    ("#1", ["#1"]),
    ("#3 + #10", ["#3", "#10"]),
    ("#1 + #10", ["#1", "#10"]),
]

EXCL = ("#1 and #3 compete for ResAdd1 (it folds into either o_proj's epilogue or the "
        "post-attention RMSNorm, never both); #10 and #3 likewise compete for ResAdd2 (the "
        "merge tail or the next layer's input_layernorm). Sets are therefore ADDITIVE "
        "ESTIMATES from independently measured deltas, not measured combinations.")
COMPOSE = ("additive estimate -- the C500 study measured combinations end-to-end precisely "
           "because they do NOT compose additively (LOG-11 S6: shared producers and competition "
           "for the same operation). Ranks here are indicative, not measured.")
ROUTER = ("#4/#5/#11b fuse the post-attention RMSNorm into the router, but that norm also feeds "
          "the w13 grouped GEMM and the shared expert, so fusing it into the router does not "
          "remove it -- worth ~0 per layer unless every K=6144 consumer is fused (#11a), which "
          "needs 12 GB of w13 and is unmeasurable here. Scored 0.0 rather than by their "
          "standalone chain speedups.")
FIELDS = ["regime", "fusion_set", "components", "sites_per_layer", "ms_saved_per_layer",
          "rank", "basis", "layer_total_ms", "speedup_vs_unfused", "layer_measurable", "notes"]

def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = []
    for reg in REGIMES:
        scored = []
        for name, fams in SETS:
            ms = sum(delta(f, reg) * SITES[f] for f in fams)
            scored.append((name, fams, ms))
        # router-norm family, scored honestly at 0 per layer
        scored.append(("#4 / #5 / #11b (router-norm family)", ["router"], 0.0))
        scored.sort(key=lambda x: -x[2])
        for i, (name, fams, ms) in enumerate(scored, 1):
            note, basis = "", "measured"
            if fams == ["router"]:
                note, basis = ROUTER, "structural (0 by shared-producer argument)"
            elif len(fams) > 1:
                note, basis = f"{EXCL} {COMPOSE}", "additive estimate"
            elif name == "none (all unfused)":
                note, basis = "baseline", "-"
            rows.append({
                "regime": reg, "fusion_set": name,
                "components": " + ".join(fams) if fams and fams != ["router"] else ("#4/#5/#11b" if fams else "-"),
                "sites_per_layer": "+".join(str(SITES[f]) for f in fams) if fams and fams != ["router"] else "",
                "ms_saved_per_layer": f"{ms:.4f}", "rank": i, "basis": basis,
                "layer_total_ms": "", "speedup_vs_unfused": "",
                "layer_measurable": "FALSE", "notes": note})
    p = OUT / "layer_optimal_per_regime.csv"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"  wrote {p.relative_to(ROOT)}  ({len(rows)} rows)")
    for reg in REGIMES:
        best = [r for r in rows if r["regime"] == reg and r["rank"] == 1][0]
        print(f"    {reg:<15} best: {best['fusion_set']:<34} saves {best['ms_saved_per_layer']:>9} ms/layer")

if __name__ == "__main__":
    main()
