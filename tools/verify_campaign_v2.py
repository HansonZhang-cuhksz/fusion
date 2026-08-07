#!/usr/bin/env python3
"""Campaign-v2 validity gate: run on the copied-back `results/h200/` BEFORE regenerating
the report, and only a PASS justifies publishing.

The first H200 campaign shipped 21 cells ABOVE CEILING, 84/84 SEQUENTIAL, #11a blocked by
a kernel defect, and cells measured under co-tenancy.  This verifier exists so the re-run
cannot silently repeat any of it: every check below corresponds to a failure mode of the
first campaign, and a FAIL means the report must NOT be regenerated from this data.

    python3 tools/verify_campaign_v2.py [--results-dir results/h200]

Exit code 0 = publishable; 1 = FAIL (report must not be regenerated); 2 = FATAL (input
missing or unreadable -- also not publishable).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Absolute harness-floor sanity bar -- same value as glm52_h200/config.py:FLOOR_US_MAX,
#: which owns the number (idle H200 floors are 37-42 us; 50 is deliberately generous and
#: the preflight's tick match is the real co-tenant detector). Keep them in sync.
FLOOR_US_MAX = 50.0
TICK_MATCH_MIN = 0.9
#: The 12 (family, variant) groups that make up the 84-cell layer-level report.
EXPECTED_GROUPS = [
    ("f01", "triton"), ("f03", "f3"),
    ("f04f05", "F4"), ("f04f05", "F4_topk"), ("f04f05", "F5"), ("f04f05", "F5_topk"),
    ("f06", "f6"),
    ("f08f09", "f8_atomic"), ("f08f09", "f8_token_major"),
    ("f08f09", "f9_atomic"), ("f08f09", "f9_token_major"),
    ("f10", "f10"),
]
REGIMES = ["decode_bs1", "decode_bs32", "decode_bs256", "decode_bs512",
           "decode_bs1024", "prefill_t2048", "prefill_t8192"]


class Verdict:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []  # (check, status, detail)

    def ok(self, name: str, detail: str = "") -> None:
        self.rows.append((name, "PASS", detail))

    def fail(self, name: str, detail: str) -> None:
        self.rows.append((name, "FAIL", detail))

    def info(self, name: str, detail: str) -> None:
        self.rows.append((name, "INFO", detail))

    @property
    def n_fail(self) -> int:
        return sum(1 for _, s, _ in self.rows if s == "FAIL")

    def print(self) -> None:
        w = max(len(r[0]) for r in self.rows) + 1
        for name, status, detail in self.rows:
            print(f"  {name:<{w}} {status:<5} {detail}")
        print(f"\n  {self.n_fail} FAIL, "
              f"{sum(1 for _, s, _ in self.rows if s == 'PASS')} PASS, "
              f"{sum(1 for _, s, _ in self.rows if s == 'INFO')} INFO")


def load_json(path: Path) -> dict:
    try:
        blob = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"FATAL: {path} missing -- nothing to verify") from exc
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"FATAL: {path} unreadable ({type(exc).__name__}: {exc})") from exc
    return blob


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default="results/h200")
    args = ap.parse_args(argv)
    res = Path(args.results_dir)
    if not res.is_dir():
        print(f"FATAL: {res} is not a directory", file=sys.stderr)
        return 2

    v = Verdict()
    summary = load_json(res / "summary.json")

    # ------------------------------------------------------------------ completeness
    cells = summary.get("cells", [])
    got = {(c["family"], c["variant"]) for c in cells}
    missing_groups = [g for g in EXPECTED_GROUPS if g not in got]
    if missing_groups:
        v.fail("completeness", f"missing variant groups: {missing_groups}")
    else:
        v.ok("completeness", f"all 12 variant groups present ({len(cells)} cells)")
    per_regime = {r: 0 for r in REGIMES}
    for c in cells:
        per_regime[c["regime"]] = per_regime.get(c["regime"], 0) + 1
    bad = {r: n for r, n in per_regime.items() if n != len(EXPECTED_GROUPS)}
    if bad:
        v.fail("per-regime", f"expected 12 cells per regime, got: {bad}")
    else:
        v.ok("per-regime", "12 cells in every regime")
    if summary.get("missing_families"):
        v.fail("missing-families", f"{summary['missing_families']}")
    else:
        v.ok("missing-families", "none")
    if summary.get("quarantined"):
        v.fail("quarantined", f"{summary['quarantined']}")
    else:
        v.ok("quarantined", "no foreign/stale files were mixed in")

    # ------------------------------------------------------------------ calibration
    pf = summary.get("preflight") or {}
    floor = pf.get("harness_floor_us")
    launch = pf.get("launch_us")
    match = pf.get("timer_tick_match_frac")
    if not isinstance(floor, (int, float)) or not isinstance(launch, (int, float)):
        v.fail("calibration", f"preflight missing floor/launch: {pf.get('path')}")
    else:
        if floor <= 0 or floor > FLOOR_US_MAX:
            v.fail("calibration",
                   f"harness_floor_us={floor:.3f} (bar: 0 < f <= {FLOOR_US_MAX})")
        elif launch <= 0:
            v.fail("calibration", f"launch_us={launch:.3f} is not positive")
        else:
            v.ok("calibration",
                 f"floor {floor:.2f} us / launch {launch:.2f} us / "
                 f"tick {pf.get('timer_tick_us')} us")
    if isinstance(match, (int, float)):
        if match < TICK_MATCH_MIN:
            v.fail("tick-match", f"{match:.3f} < {TICK_MATCH_MIN}: the 'tick' is not a "
                                 f"tick; the GPU was contended during preflight")
        else:
            v.ok("tick-match", f"{match:.3f}")
    elif summary.get("timer", {}).get("finer_than_tested"):
        v.ok("tick-match", "timer finer than every tested granularity (idle device)")
    else:
        v.info("tick-match", "not recorded")

    # ------------------------------------------------------------------ pairedness
    seq = [c for c in cells if c.get("paired") is False]
    if seq:
        v.fail("paired", f"{len(seq)} of {len(cells)} cells SEQUENTIAL "
                         f"(not interleaved A/B/A/B): {seq[0]['family']}/{seq[0]['regime']} "
                         "first")
    else:
        v.ok("paired", f"all {len(cells)} cells paired")
    bad_src = [c for c in cells if c.get("speedup_source") not in
               ("paired_speedup", "speedup")]
    if bad_src:
        v.fail("speedup-source", f"{len(bad_src)} cells with unexpected source "
                                 f"(first: {bad_src[0].get('speedup_source')})")
    else:
        v.ok("speedup-source", "all cells carry a speedup from a known source")

    # ------------------------------------------------------------------ ceilings
    above = [c for c in cells if any(f.startswith("ABOVE CEILING") for f in c["flags"])]
    if above:
        v.fail("ceiling", f"{len(above)} cells ABOVE CEILING "
                          f"(first: {above[0]['family']}/{above[0]['regime']} "
                          f"{above[0]['speedup_raw']:.3f}x)")
    else:
        v.ok("ceiling", f"no cell exceeds its modelled traffic ceiling "
                        f"({sum(1 for c in cells if 'ceiling' in c)} cells carry one)")
    drift = [c for c in cells if any(f.startswith("DRIFT") for f in c["flags"])]
    if drift:
        v.fail("drift", f"{len(drift)} cells: paired vs ratio-of-medians disagree >2% "
                        f"(first: {drift[0]['family']}/{drift[0]['regime']})")
    else:
        v.ok("drift", "paired statistic agrees with ratio-of-medians in every cell")

    # ------------------------------------------------------------------ resolution
    unres = [c for c in cells if not c["resolved"]]
    if unres:
        v.info("unresolved", f"{len(unres)} cells within 3 timer ticks (honest blank, "
                             f"not a fake ratio): "
                             + ", ".join(f"{c['family']}/{c['regime']}"
                                         for c in unres[:5]))
    else:
        v.ok("unresolved", "every cell resolved")

    # ------------------------------------------------------------------ device
    uuids = {}
    for fam in summary.get("families", []):
        uuids.setdefault(fam.get("family"), set()).add(
            (fam.get("env_overrides") or {}).get("CUDA_VISIBLE_DEVICES"))
    dev = summary.get("device") or {}
    if dev.get("name") == "NVIDIA H200":
        v.ok("device", f"{dev.get('name')} {dev.get('uuid', '')[:8]}...")
    else:
        v.fail("device", f"campaign device is {dev.get('name')!r}, not an H200")

    # ------------------------------------------------------------------ f11 (separate machinery)
    f11_files = [f for f in sorted(res.iterdir()) if f.name.startswith("f11_") and f.suffix == ".json"]
    if f11_files:
        v.ok("f11-files", f"{len(f11_files)} f11 result files present")
    else:
        v.fail("f11-files", "no f11 result files in the results dir")

    v.print()
    return 1 if v.n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
