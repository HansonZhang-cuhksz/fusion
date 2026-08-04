"""`report_glm52_h200/layer_optimal_per_regime.csv` -- the whole-layer combination sweep.

This is a GENUINELY MEASURED layer file, the first since C500. It uses the C500 schema
unchanged (regime, fusion_set, config_id, run1_ms, run2_ms, best_ms, speedup_vs_unfused,
tied_with_best_run1, tied_with_best_run2) so the three reports diff against each other.

Contrast with the RTX 4060 port, where `layer_total_ms` / `speedup_vs_unfused` were left
EMPTY because a GLM-5.2 MoE layer does not fit in 7.4 GB and `bench_layer.py` was never run
there at all. The H200 has 138 GB free, so w13 + w2 + the folded w13 (30 GB total) all fit and
the real sweep ran: 18 configurations x 7 regimes x 2 independent passes x 8 interleaved
rounds x 15 reps.

SOURCE  results/h200/layer_configurations.json (read-only; never regenerated here)
  regimes.<regime>.pass1 / .pass2   per-config {median, min, max, spread_ms, rounds[8]}
  regimes.<regime>.correctness      per-config {rel_err, ok, n_kernels, sel}
  regimes.<regime>.verdict          the harness's own two-pass verdict

WHAT EACH COLUMN MEANS HERE
  run1_ms / run2_ms  the per-configuration MEDIAN over the 8 interleaved rounds of pass 1 and
                     pass 2. Both passes exist for every regime, so neither column is ever
                     empty in this file -- unlike C500, where several cells are blank because
                     a configuration only appeared in one of the two source runs.
  best_ms            min(run1_ms, run2_ms), exactly as C500 computes it.
  speedup_vs_unfused best_ms of A_all_unfused divided by this row's best_ms. This is a
                     RATIO OF MEDIANS across separately-timed candidates, not a paired
                     interleaved A/B ratio; the two are different measurements. The paired
                     head-vs-baseline ratio the harness also recorded is printed by this
                     script but deliberately kept OUT of the CSV, because putting it in the
                     same column as a ratio of medians would present them as interchangeable.
  tied_with_best_*   1 if this configuration is in that pass's tied set under the LOG-11 S3
                     rule, recomputed here from the raw round data rather than trusted:
                     a winner is declared only when its gap to the runner-up exceeds the
                     round-to-round spread (max - min over the 8 rounds) of BOTH; every
                     configuration within that spread of the pass leader is tied with it.
                     The recomputation is cross-checked against the recorded verdict and any
                     disagreement is reported loudly.

TWO CAVEATS THE SCHEMA HAS NOWHERE TO PUT, so they are recorded here and belong in the
report README rather than being silently dropped:

  1. The measurement host is multi-tenant and the run took whatever device it inherited
     (`fairness.gpu.selection.requested = null`, CUDA_VISIBLE_DEVICES=3, no --gpu). The
     harness's own calibration marks itself UNRELIABLE and `contended: true` -- a 42.19 us
     harness floor against a 9.08 us launch, i.e. "a floor is a launch plus a sync, so this
     one is measuring somebody else's kernels too". Every layer time here is 0.46 ms or
     larger, so the floor is not what is being reported, but at decode_bs1 it is ~9 % of the
     total and it is common to all candidates. Sub-percent gaps on this host are noise.
  2. decode_bs1 pass 1 opens with a cold-start excursion -- round 0 runs up to 2.4x the
     median of the remaining seven rounds, on many configurations at once. The per-config
     median absorbs it, but it inflates that pass's round-to-round spread and hence its
     noise floor, which is part of why decode_bs1 comes out TIED.

CONFIGURATIONS THAT NEVER APPEAR IN A ROW. A configuration that fails the independent fp32
reference is excluded from timing outright (LOG-11 S7), so it has no times to report and gets
no row -- it is not written with empty cells and it is certainly not modelled. On H200 that is
the whole `prenorm="all"` group (O/P/Q/R, i.e. #11a+#11b) at every regime except decode_bs1,
where it passed and is timed. Those exclusions are printed per regime.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "h200" / "layer_configurations.json"
OUT = ROOT / "report_glm52_h200"

# C500's names, plus the six configurations C500 never ran (the norm->router family and the
# lazy pre-norm group), named on the same "#n" convention.
NAMES = {
    "A_all_unfused": "none (all unfused)",
    "B_f3": "#3",
    "C_f1": "#1",
    "D_f6": "#6",
    "E_f10": "#10",
    "F_f8": "#8",
    "G_f9": "#9",
    "H_f3_f10": "#3 + #10",
    "I_f3_f9": "#3 + #9",
    "J_greedy_all": "#1+#6+#9 (greedy)",
    "K_f3_f8": "#3 + #8",
    "L_f5": "#5",
    "M_f4": "#4",
    "N_f11b": "#11b",
    "O_f11ab": "#11a + #11b",
    "P_f10_f11ab": "#10 + #11a + #11b",
    "Q_f8_f11ab": "#8 + #11a + #11b",
    "R_f1_f10_f11ab": "#1 + #10 + #11a + #11b",
}

ORDER = ["decode_bs1", "decode_bs32", "decode_bs256", "decode_bs512", "decode_bs1024",
         "prefill_t2048", "prefill_t8192"]

BASE = "A_all_unfused"

FIELDS = ["regime", "fusion_set", "config_id", "run1_ms", "run2_ms", "best_ms",
          "speedup_vs_unfused", "tied_with_best_run1", "tied_with_best_run2"]


def tied_set(stats: dict) -> list[str]:
    """LOG-11 S3 / bench_layer.verdict_of, recomputed from the recorded round medians.

    Order the pass by median. The noise floor is the larger of the round-to-round spreads
    (max - min over the 8 interleaved rounds) of the leader and the runner-up. Every
    configuration whose median sits within that floor of the leader is tied with it; the
    leader is separated only if the gap to the runner-up exceeds the floor.
    """
    order = sorted(stats, key=lambda k: stats[k]["median"])
    if not order:
        return []
    if len(order) < 2:
        return order
    best, runner = order[0], order[1]
    noise = max(spread(stats[best]), spread(stats[runner]))
    return [n for n in order if stats[n]["median"] - stats[best]["median"] <= noise]


def spread(s: dict) -> float:
    """Round-to-round spread. Recomputed from `rounds` when present so the tie rule rests on
    the raw per-round numbers, not on a summary field."""
    r = s.get("rounds")
    if r:
        return max(r) - min(r)
    return s["spread_ms"]


def main() -> None:
    doc = json.loads(SRC.read_text())
    regimes = doc.get("regimes", {})
    rows, missing, mismatches = [], [], []

    print(f"source   {SRC.relative_to(ROOT)}")
    print(f"device   {doc.get('env', {}).get('device_name')}   "
          f"passes={doc.get('protocol', {}).get('passes')} "
          f"rounds/pass={doc.get('protocol', {}).get('rounds_per_pass')} "
          f"rep/round={doc.get('protocol', {}).get('rep_per_round')}\n")

    for regime in ORDER:
        r = regimes.get(regime)
        if not r:
            missing.append(regime)
            continue
        s1, s2 = r.get("pass1", {}), r.get("pass2", {})
        if not s1 and not s2:
            missing.append(regime)
            continue

        # Recompute the tie verdicts, then check them against what the harness recorded.
        t1, t2 = tied_set(s1), tied_set(s2)
        rec = r.get("verdict", {})
        for tag, mine, theirs in (("pass1", t1, rec.get("pass1", {}).get("tied_set")),
                                  ("pass2", t2, rec.get("pass2", {}).get("tied_set"))):
            if theirs is not None and set(mine) != set(theirs):
                mismatches.append(f"{regime}/{tag}: recomputed {sorted(mine)} != "
                                  f"recorded {sorted(theirs)}")

        names = set(s1) | set(s2)

        def med(cid: str) -> float | None:
            vals = [s[cid]["median"] for s in (s1, s2) if cid in s]
            return min(vals) if vals else None

        base = med(BASE)

        for cid in sorted(names, key=lambda k: (med(k), k)):
            m1 = s1.get(cid, {}).get("median")
            m2 = s2.get(cid, {}).get("median")
            best = med(cid)
            rows.append({
                "regime": regime,
                "fusion_set": NAMES.get(cid, cid),
                "config_id": cid,
                "run1_ms": f"{m1:.4f}" if m1 is not None else "",
                "run2_ms": f"{m2:.4f}" if m2 is not None else "",
                "best_ms": f"{best:.4f}",
                "speedup_vs_unfused": f"{base / best:.4f}" if base else "",
                "tied_with_best_run1": int(cid in t1),
                "tied_with_best_run2": int(cid in t2),
            })

        # ---- what the file says about this regime, in the terms LOG-11 uses --------------
        lead = sorted(names, key=lambda k: (med(k), k))[0]
        status = rec.get("status")
        winner = rec.get("winner")
        # Configurations that were built but never timed, because they failed the
        # independent fp32 reference. No row, no empty row, no estimate.
        failed = sorted(k for k, c in r.get("correctness", {}).items()
                        if not c.get("ok") and k not in names)
        print(f"{regime:<15} {status:<9} "
              f"{'winner ' + winner if winner else 'tied: ' + ', '.join(rec.get('tied_set', []))}")
        print(f"{'':<15} fastest row {NAMES.get(lead, lead)} ({lead}) "
              f"{med(lead):.4f} ms  vs all-unfused {base:.4f} ms "
              f"= {base / med(lead):.4f}x")
        for tag, v in (("pass1", rec.get("pass1", {})), ("pass2", rec.get("pass2", {}))):
            gap, noise = v.get("gap_ms"), v.get("noise_ms")
            print(f"{'':<15}   {tag}: best {v.get('best')} gap "
                  f"{gap:.5f} ms vs round noise {noise:.5f} ms -> "
                  f"{'SEPARATED' if v.get('separated') else 'tied'}")
        p = r.get("paired_head_vs_unfused")
        if p:
            pm = p.get("pair_meta", {})
            print(f"{'':<15}   paired interleaved A/B of head {p['head']}: "
                  f"{p['paired_speedup_p50']:.4f}x p50 "
                  f"(ratio-of-medians {pm.get('ratio_of_medians'):.4f}x) "
                  f"-- NOT the same statistic as the CSV column, kept out of it")
        else:
            print(f"{'':<15}   no paired A/B recorded (the harness pairs only the head "
                  f"configuration, and here the head is the baseline itself)")
        if failed:
            print(f"{'':<15}   excluded, failed the fp32 reference (no row): "
                  f"{', '.join(failed)}")
        print()

    OUT.mkdir(exist_ok=True)
    out = OUT / "layer_optimal_per_regime.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.relative_to(ROOT)}  ({len(rows)} rows, "
          f"{len(ORDER) - len(missing)}/{len(ORDER)} regimes)")
    if missing:
        print(f"MISSING (no pass data): {', '.join(missing)}")
    if mismatches:
        print("TIE-RULE MISMATCH vs recorded verdict:")
        for m in mismatches:
            print(f"  {m}")
    else:
        print("tie sets recomputed from the raw rounds agree with the recorded verdict "
              "in all 14 passes")


if __name__ == "__main__":
    main()
