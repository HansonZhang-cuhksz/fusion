"""Assemble `report/layer_optimal_per_regime.csv` from the interleaved A/B runs.

Every regime is measured TWICE, independently. A configuration is only credited with beating
another when the gap survives both runs; the `tied_with_best_runN` columns carry each run's
own verdict so a reader can see where the two disagree (they do, at the decode sizes, which
is exactly why two runs are reported rather than one).

Sources, per regime:
  decode_bs1/32/256, prefill_t2048/t8192 -> layer_configurations_ab.json  + _ab2.json
  decode_bs512                           -> layer_configurations_ab_new.json
                                            + layer_configurations_ab_new2_bs512.json
  decode_bs1024                          -> layer_configurations_ab_new.json
                                            + layer_configurations_ab_new2_bs1024.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPORT = ROOT / "report"

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
}

ORDER = ["decode_bs1", "decode_bs32", "decode_bs256", "decode_bs512", "decode_bs1024",
         "prefill_t2048", "prefill_t8192"]

# regime -> (run1 file, run2 file)
SOURCES = {
    **{g: ("layer_configurations_ab", "layer_configurations_ab2")
       for g in ("decode_bs1", "decode_bs32", "decode_bs256",
                 "prefill_t2048", "prefill_t8192")},
    "decode_bs512": ("layer_configurations_ab_new", "layer_configurations_ab_new2_bs512"),
    "decode_bs1024": ("layer_configurations_ab_new", "layer_configurations_ab_new2_bs1024"),
}

FIELDS = ["regime", "fusion_set", "config_id", "run1_ms", "run2_ms", "best_ms",
          "speedup_vs_unfused", "tied_with_best_run1", "tied_with_best_run2"]


def _load(name: str) -> dict:
    p = RESULTS / f"{name}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("regimes", {})


def main() -> None:
    rows, missing = [], []
    for regime in ORDER:
        f1, f2 = SOURCES[regime]
        r1 = _load(f1).get(regime, {})
        r2 = _load(f2).get(regime, {})
        if not r1 and not r2:
            missing.append(regime)
            continue
        s1, s2 = r1.get("stats", {}), r2.get("stats", {})
        v1, v2 = r1.get("verdict", {}), r2.get("verdict", {})
        names = set(s1) | set(s2)
        base = min([s.get("A_all_unfused", {}).get("median", float("inf"))
                    for s in (s1, s2)] or [float("inf")])

        def med(name):
            vals = [s[name]["median"] for s in (s1, s2) if name in s]
            return min(vals) if vals else None

        for cid in sorted(names, key=lambda k: med(k) or float("inf")):
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
                "speedup_vs_unfused": f"{base / best:.4f}" if base < float("inf") else "",
                "tied_with_best_run1": int(cid in v1.get("tied_set", [])),
                "tied_with_best_run2": int(cid in v2.get("tied_set", [])),
            })
        print(f"{regime:<16} run1={'ok' if s1 else '--'} run2={'ok' if s2 else '--'}  "
              f"best={v1.get('best') or v2.get('best')}  "
              f"{base / med(v1.get('best') or v2.get('best')):.4f}x vs unfused")

    REPORT.mkdir(exist_ok=True)
    out = REPORT / "layer_optimal_per_regime.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows, {len(ORDER) - len(missing)} regimes)")
    if missing:
        print(f"MISSING (no A/B data yet): {', '.join(missing)}")


if __name__ == "__main__":
    main()
