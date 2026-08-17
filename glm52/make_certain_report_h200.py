"""`report_glm52_h200/certain/` -- tables from the uncertainty-controlled H200 re-run.

WHAT THIS IS, AND WHY IT SITS BESIDE THE CAMPAIGN RATHER THAN REPLACING IT.

`layer_optimal_per_regime.csv` is the 2026-08-07/08-10 campaign: two independent passes of
eight interleaved rounds, per-regime autotuning, split across two physical cards. This
directory is a single 7.1-minute run of `glm52_h200/bench/bench_layer_certain.py` on
2026-08-13 that holds four things fixed the campaign could not:

  * ONE process, ONE card (`GPU-338e7fe0`), all eleven regimes -- the campaign's `decode_bs1`
    was measured on a different card from `decode_bs2/4/8/16`, so its T=1->2 and T=16->32
    segments crossed a physical device.
  * FROZEN configs, read from the campaign's own `layer_configurations.json`. Nothing is
    tuned here, so no arm can be tuned unequally -- the confound behind the campaign's
    unexplained T=2 dip.
  * A blocked, rotated, drift-gated protocol with a real bootstrap CI per configuration,
    replacing the round-spread tie heuristic of LOG-11 §3.
  * Every configuration timed three ways -- wall clock, CUDA-graph replay, and the CUPTI sum
    of per-kernel device time -- which decomposes the layer as
    `wall = work + in-graph gaps + launch cost`.

WHAT CHANGED, AND WHAT DID NOT. The two agree closely where the campaign was trustworthy:
at T = 4..1024 every configuration INCLUDING the all-unfused baseline reproduces the
campaign's absolute time to <= 0.3 %. The differences are concentrated where the campaign
had known problems -- `decode_bs1`, whose campaign baseline carried a cold-start excursion
(round 0 at 2.5x the median of the other seven) and which the campaign itself declared TIED
with nothing resolving.

The substantive gain is RESOLUTION, not different physics. The campaign resolved a unique
winner at 3 of 11 regimes and had the unfused layer inside the tie set at 3 more, i.e. three
regimes where its own rule could not distinguish fusion from doing nothing. Here 6 of 11
resolve a unique winner and **the unfused layer is in no regime's tie set at all**.

READ THE CAVEATS. Clock locking needs root and was NOT available, so the card ran unlocked;
at T >= 512 the drift gate discarded 39-88 % of blocks and the CIs there are 3-10x wider
than at mid-decode. `decode_bs1`'s CI is +-0.17 %, `prefill_t8192`'s +-0.43 %. Treat
differences below those as unresolved. The four `#11a`-bearing configurations fail the fp32
reference at every regime except `decode_bs1` (rel_err 0.06-0.61 against a 5e-2 bar) and are
excluded by the harness, reproducing the campaign's own finding that `#11a` is unmeasurable
on this device.

Run:  /home/zhangshuhan/my-envs/fusion/bin/python glm52/make_certain_report_h200.py
      (from the repo root; the default python3 on this box has no matplotlib, though this
       script itself needs none)
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "h200" / "layer_certain.json"
CAMPAIGN = ROOT / "report_glm52_h200" / "layer_optimal_per_regime.csv"
OUT = ROOT / "report_glm52_h200" / "certain"

UNFUSED = "A_all_unfused"

#: Display name per config id, taken from the campaign CSV's `fusion_set` column so the two
#: tables can be joined on sight rather than on a lookup.
CHAIN = {
    "A_all_unfused": "none (all unfused)", "B_f3": "#3", "C_f1": "#1", "D_f6": "#6",
    "E_f10": "#10", "F_f8": "#8", "G_f9": "#9", "H_f3_f10": "#3 + #10",
    "I_f3_f9": "#3 + #9", "J_greedy_all": "#1+#6+#9 (greedy)", "K_f3_f8": "#3 + #8",
    "L_f5": "#5", "M_f4": "#4", "N_f11b": "#11b", "O_f11ab": "#11a + #11b",
    "P_f10_f11ab": "#10 + #11a + #11b", "Q_f8_f11ab": "#8 + #11a + #11b",
    "R_f1_f10_f11ab": "#1 + #10 + #11a + #11b",
}


def load() -> tuple[dict, dict]:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC} -- run glm52_h200/bench/bench_layer_certain.py")
    doc = json.loads(SRC.read_text())
    doc = doc.get("payload", doc)
    camp = {}
    if CAMPAIGN.exists():
        for r in csv.DictReader(CAMPAIGN.open()):
            camp[(r["regime"], r["config_id"])] = r
    return doc, camp


def verdict_of(block: dict, cfg: str) -> str:
    """How this configuration stands against the unfused layer, by its own CI."""
    v = block["per_config"].get(cfg, {})
    if "ci_lo" not in v:
        return ""
    if v.get("beats_unfused"):
        return "faster"
    if v.get("loses_to_unfused"):
        return "slower"
    return "tied with unfused"


def main() -> None:
    doc, camp = load()
    regs = doc["regimes"]
    order = sorted(regs, key=lambda r: regs[r]["T"])
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. the per-regime table -------------------------------------------------------
    rows = []
    for r in order:
        b = regs[r]
        wall, graph = b["wall"], (b.get("graph") or {})
        wv = wall.get("verdict") or {}
        for cfg, v in wall["per_config"].items():
            if "speedup_p50" not in v and cfg != UNFUSED:
                continue
            gv = (graph.get("per_config") or {}).get(cfg, {})
            dec = (b.get("decomposition") or {}).get(cfg, {})
            ck = (b.get("correctness") or {}).get(cfg, {})
            c = camp.get((r, cfg))
            camp_sp = (float(c["speedup_vs_unfused"])
                       if c and c["speedup_vs_unfused"].strip() else None)
            rows.append({
                "regime": r, "T": b["T"], "config_id": cfg,
                "fusion_set": CHAIN.get(cfg, cfg),
                "n_kernels": ck.get("n_kernels", ""),
                "rel_err": f"{ck['rel_err']:.3e}" if "rel_err" in ck else "",
                "wall_ms": f"{v['ms_p50']:.5f}" if "ms_p50" in v else "",
                "graph_ms": f"{gv['ms_p50']:.5f}" if "ms_p50" in gv else "",
                "work_us": f"{dec['work_us']:.1f}" if dec.get("work_us") else "",
                "gap_us": f"{dec['gap_us']:.1f}" if dec.get("gap_us") is not None else "",
                "launch_us": (f"{dec['launch_us']:.1f}"
                              if dec.get("launch_us") is not None else ""),
                "speedup_wall": f"{v['speedup_p50']:.4f}" if "speedup_p50" in v else "",
                "ci_lo": f"{v['ci_lo']:.4f}" if "ci_lo" in v else "",
                "ci_hi": f"{v['ci_hi']:.4f}" if "ci_hi" in v else "",
                "speedup_graph": (f"{gv['speedup_p50']:.4f}"
                                  if "speedup_p50" in gv else ""),
                "vs_unfused": verdict_of(wall, cfg) if cfg != UNFUSED else "baseline",
                "regime_winner": "1" if cfg == wv.get("best") else "",
                "tied_with_winner": "1" if cfg in (wv.get("tied_with_best") or []) else "",
                "campaign_speedup": f"{camp_sp:.4f}" if camp_sp is not None else "",
                "delta_vs_campaign": (f"{v['speedup_p50'] - camp_sp:+.4f}"
                                      if camp_sp is not None and "speedup_p50" in v else ""),
            })
    dst = OUT / "layer_certain_per_regime.csv"
    with dst.open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"wrote {dst}  ({len(rows)} rows)")

    # ---- 2. the per-regime verdict summary ---------------------------------------------
    vrows = []
    for r in order:
        b = regs[r]
        w, g = b["wall"], (b.get("graph") or {})
        wv, gv = (w.get("verdict") or {}), (g.get("verdict") or {})
        cam_tie = sum(1 for (rr, _), c in camp.items()
                      if rr == r and (c["tied_with_best_run1"] == "1"
                                      or c["tied_with_best_run2"] == "1"))
        vrows.append({
            "regime": r, "T": b["T"],
            "winner_wall": wv.get("best", ""), "separated_wall": int(bool(wv.get("separated"))),
            "n_tied_wall": len(wv.get("tied_with_best") or []),
            "unfused_in_tie_set": int(bool(wv.get("unfused_in_tie_set"))),
            "winner_graph": gv.get("best", ""),
            "graph_agrees": int(wv.get("best") == gv.get("best")),
            "blocks_kept": w["blocks_kept"], "blocks_dropped": w["blocks_dropped"],
            "blocks_lost": w.get("blocks_flushed_at_exit", 0),
            "ci_achieved": f"{w.get('ci_achieved', float('nan')):.2e}",
            "ci_target_met": int(bool(w.get("ci_target_met"))),
            "stop_reason": str(w.get("stop_reason", "")).split(":")[0],
            "campaign_n_tied": cam_tie,
        })
    dst2 = OUT / "layer_certain_verdicts.csv"
    with dst2.open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(vrows[0]))
        wtr.writeheader()
        wtr.writerows(vrows)
    print(f"wrote {dst2}  ({len(vrows)} rows)")

    # ---- 3. printed summary -------------------------------------------------------------
    print(f"\n{'T':>6}  {'winner':<17}{'sep':>4}{'tied':>5}{'gain':>9}"
          f"{'  95% CI':<20}{'campaign said':<20}{'blocks':>10}")
    for v, r in zip(vrows, order):
        b = regs[r]
        w = b["wall"]
        best = v["winner_wall"]
        pv = w["per_config"].get(best, {})
        c = camp.get((r, best))
        cs = (f"{float(c['speedup_vs_unfused']):.4f}"
              if c and c["speedup_vs_unfused"].strip() else "-")
        print(f"{v['T']:>6}  {CHAIN.get(best, best):<17}"
              f"{'Y' if v['separated_wall'] else 'n':>4}{v['n_tied_wall']:>5}"
              f"{pv.get('speedup_p50', float('nan')):>9.4f}"
              f"  [{pv.get('ci_lo', float('nan')):.4f},{pv.get('ci_hi', float('nan')):.4f}]"
              f"    {cs:<16}{v['blocks_kept']:>4}/{v['blocks_kept'] + v['blocks_dropped']:<5}")

    n_sep = sum(v["separated_wall"] for v in vrows)
    n_null = sum(v["unfused_in_tie_set"] for v in vrows)
    n_agree = sum(v["graph_agrees"] for v in vrows)
    print(f"\n  {n_sep}/11 regimes resolve a unique winner (campaign: 3/11)")
    print(f"  {n_null}/11 have the unfused layer inside the tie set (campaign: 3/11)")
    print(f"  {n_agree}/11 agree between wall clock and CUDA-graph replay")
    lost = sum(v["blocks_lost"] for v in vrows)
    print(f"  {lost} blocks measured-then-lost (must be 0; anything else is a harness bug)")


if __name__ == "__main__":
    sys.exit(main())
