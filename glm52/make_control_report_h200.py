#!/usr/bin/env python3
"""Compare the staged Hopper CONTROL ARM against the committed H200 campaign, cell for cell.

    python3 glm52/make_control_report_h200.py

WHY THIS FILE EXISTS SEPARATELY FROM `glm52/make_report_h200.py`.

The H200-vs-C500 comparison found that the four TMA-using families improved the LEAST on
Hopper (median 1.00-1.06x) while `f03` and `f10` -- which advertise no Hopper cfg key at all
and therefore cannot use TMA, warp specialization or thread-block clusters -- improved 1.86x
and 1.48x. That comparison is confounded: the H200 is simply a better GPU than the C500. The
decisive experiment is the control arm, `GLM52_H200_CLASSIC=1`, which forces every Hopper
capability off and re-measures. `run_control_h200.py` runs it on the operator's machine and
stages the result under `results/h200/_control_arm/`; this file turns that staging tree plus
the committed campaign into a diff.

WHAT THIS FILE IS NOT. It is not a merge, and there is no merge step anywhere in this repo.
Nothing under `results/h200/_control_arm/` is ever written into `results/h200/*.json`. The
campaign files are the BASELINE of the comparison; writing to them would destroy the thing
being compared against. This script opens the campaign read-only and writes only under
`--out` (default `report_glm52_h200/control_arm/`).

THE ONE IDEA THE WHOLE REPORT RESTS ON -- the f03/f10 noise floor.

The operator chose an unpaired design: the control arm is measured now and diffed against a
campaign measured on 2026-08-07. Cross-session drift (thermals, co-tenancy, clock state,
driver state) is therefore confounded with the effect of turning the Hopper levers off. The
defence is not statistical, it is structural: `f03` (ResAdd+RMSNorm) and `f10` (ExpertMerge+
ResAdd) advertise NO Hopper cfg key, so their classic arm is byte-identically configured to
their Hopper arm. Their classic-vs-campaign delta contains ONLY run-to-run and cross-session
variation. It IS the drift band. Every other family's delta is judged against that band, and
a family whose delta sits inside it has shown NOTHING -- which is not the same claim as "the
Hopper features did nothing".

Do not delete the f03/f10 handling as redundant "families with no axes to disable". They are
the control, not filler. Without them this report has no defence against the confound the
design already concedes, and `--force-unverified` deliberately does not override the guard
that refuses to publish when too few of their cells are usable.

REUSE DISCIPLINE. Cells come from `run_h200.collect_cells` -- the SAME function that produced
the campaign's own `summary.json` cells -- so parity between the two sides is by construction
rather than by a re-derivation that could drift. Labels, mapping strings and the Hopper-axis
token scan come from `glm52/make_report_h200.py`'s pure helpers. Neither module's
directory-frozen globals are used; see the import block for why they cannot be.

Pure stdlib, no torch, no GPU, no network. Runs on the local box against a tarball the
operator sends back.

EXIT CODES.
    0   publishable: the CSVs and README under `--out` were written
    1   REFUSED: a validity gate failed (unverified arm -- sentinel file OR the summary's
        own verified/sentinel/engagement_summary records -- device-anchor loss, UUID
        mismatch, harness-floor mismatch, an arm that still selected an axis it was supposed
        to force off, or too few usable noise-floor cells). Nothing is written.

WHAT THE NUMBERS ARE. Every delta here is a ratio of FUSION GAINS (`unfused_ms / fused_ms`)
measured in two different sessions. It moves when the UNFUSED chain moves just as much as
when the fused kernel does, so no sentence in this report may say "arm X ran faster" -- the
statistic does not say that. And because the design is unpaired, no sentence may assert
causation either. The only two defensible readings are "the delta exceeds the drift band"
and "the delta is inside the drift band, so nothing is shown".
    2   FATAL: an input is missing or unreadable -- including the ordinary "the operator has
        not run the control arm yet" case, which prints where the data is expected and what
        command produces it. Nothing is written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib          # only for check 6, the campaign fingerprint re-hash
import json
import math
import re               # only for the FLOOR_US_MAX source-text sync check
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# --------------------------------------------------------------------------------------
# imports that can fail, each with the reason a reader needs to fix it
# --------------------------------------------------------------------------------------
try:
    import run_h200 as R
except Exception as exc:  # noqa: BLE001
    print(f"FATAL: cannot import run_h200.py from {REPO}: {type(exc).__name__}: {exc}\n"
          f"       This report reuses run_h200.collect_cells so its cells are the SAME\n"
          f"       objects the campaign's summary.json was built from. Run from the repo "
          f"root.", flush=True)
    raise SystemExit(2) from exc

try:
    # ONLY the pure helpers. `load`, `RES`, `OUT`, `SUMMARY`, `CELLS`, `FAMILY_CARD` and
    # `rows_for` are all frozen at import time against the hardcoded campaign directory and
    # cannot be retargeted at a staging tree, so they are deliberately not imported.
    # Exactly the names this module uses -- nothing is imported "for parity". Six further
    # names (best_speedup, best_of, pct, FIELDS, FUSION_FAMILY, annot) used to be pulled in
    # and kept alive by a keep-alive tuple; they were dead weight that diluted the comment
    # above and defeated the linter that would otherwise flag a genuinely broken import.
    from glm52.make_report_h200 import (m, hop_axes, hop_note, r4, _nan, NAME, HALF,
                                        REGIMES)
except Exception as exc:  # noqa: BLE001
    print(f"FATAL: cannot import glm52/make_report_h200.py -- it reads "
          f"results/h200/summary.json at IMPORT time (line 238, unguarded). That file must "
          f"exist and parse.\n       {type(exc).__name__}: {exc}", flush=True)
    raise SystemExit(2) from exc


# ======================================================================================
# constants
# ======================================================================================
DEFAULT_CONTROL = REPO / "results" / "h200" / "_control_arm"
DEFAULT_CAMPAIGN = REPO / "results" / "h200"
DEFAULT_OUT = REPO / "report_glm52_h200" / "control_arm"
DEFAULT_ARM = "classic"

SUMMARY_NAME = "control_arm_summary.json"
SENTINEL_NAME = "ARM_NOT_VERIFIED"
ENGAGE_DIRNAME = "_engagement"

#: The two families that define the drift band. They advertise no Hopper cfg key
#: (`kernel_cfg_keys == "module advertises none"`), all three axes are `offered: false` in the
#: campaign, and a full key-path scan of both committed files finds zero Hopper tokens in any
#: cfg, tune table or axis_counts. Their control arm is therefore configured identically to
#: their Hopper arm and their delta is pure session-to-session variation.
NOISE_FAMILIES = ("f03", "f10")

#: Below this many usable f03/f10 cells the band is not defensible and the report refuses.
#: 22 is the full sample (2 families x 11 regimes); 16 leaves room for a couple of UNRESOLVED
#: cells without letting the band be defined by a handful of points.
MIN_NOISE_CELLS = 16

#: A class band (decode / prefill) is only used for verdicts when it has at least this many
#: samples. decode has 18 (2 families x 9 regimes), prefill has 4 -- so prefill cells fall
#: back to the global band, and the README must say that this is a known weakness.
CLASS_MIN_N = 8

#: Absolute harness-floor sanity bar -- same value as glm52_h200/config.py:FLOOR_US_MAX,
#: which owns the number (idle H200 floors are 37-42 us; 50 is deliberately generous and the
#: preflight's tick match is the real co-tenant detector). Copied rather than imported
#: because glm52_h200/config.py does `import torch` at module level and this report must run
#: on a box with no GPU stack. `check_bar_sync()` re-reads the literal out of that file's
#: SOURCE TEXT so the copy cannot drift silently. Keep them in sync.
FLOOR_US_MAX = 50.0
TICK_MATCH_MIN = 0.9
#: A between-session difference in the harness floor larger than this is a loud flag (it is
#: itself a drift measurement), never a refusal.
FLOOR_DELTA_WARN_US = 5.0

#: 15 (family, variant) groups x 11 regimes = 165 cells, both sides.
EXPECTED_CELLS = 165

#: `#11b'` publishes `half_fused.router_speedup_vs_unfused` directly instead of going through
#: `best_speedup`, so it is not comparable through this path and is excluded by name.
HALF_VARIANTS = ("half_fused", "half")

_AXIS_ORDER = ("tma", "warp_specialize", "clusters")

#: `hop_axes` labels ("TMA", "warp-spec", "clusters(num_ctas=2)") -> canonical axis token.
#: Kept as a prefix match because the cluster label carries the selected `num_ctas`.
_AXIS_LABEL_PREFIX = (("tma", "tma"), ("warp-spec", "warp_specialize"),
                      ("warp_specialize", "warp_specialize"), ("clusters", "clusters"))

#: Which axes each advertised arm forces OFF, i.e. which tokens must be ABSENT from the
#: winning configs of that arm. Mirrors `run_control_h200.ARMS[*].axes_off`; copied rather
#: than imported because that module is the driver and importing it here would drag the
#: driver's own import-time surface into a report that must run on a GPU-less box. The two
#: tables are small, fixed and named in both files' comments -- if an arm is added there,
#: add it here. An arm missing from this table gets NO axis check and a loud warning.
ARM_AXES_OFF: dict[str, frozenset[str]] = {
    "classic": frozenset({"tma", "warp_specialize", "clusters"}),
    "no-tma": frozenset({"tma"}),
    "no-ws": frozenset({"warp_specialize"}),
    "no-clusters": frozenset({"clusters"}),
    "hopper": frozenset(),      # forces nothing off; checked in the INVERSE direction below
}

#: The `hopper` arm is the converse sanity check: it disables nothing, so it must SELECT at
#: least one Hopper axis somewhere on the GEMM-carrying families. If it selected none, the
#: build is not producing Hopper code at all and the "control" it is being compared to is
#: meaningless. These are the families the campaign offers all three axes to.
HOPPER_ARM_FAMILIES = ("f01", "f06", "f08f09")

#: The smallest effect this study was built to resolve. The H200-vs-C500 comparison that
#: motivated the control arm found the four TMA-using families improving 1.00-1.06x while
#: f03/f10 improved 1.86x/1.48x; 0.06 is therefore the smallest motivating effect size. A
#: drift band wider than this cannot distinguish "no effect" from "the effect we came for",
#: and the report says so rather than publishing an underpowered null as a clean null.
RESOLUTION_TARGET = 0.06

VERDICT_SENTENCE = {
    "INSIDE": ("inside the drift band -- nothing is shown"),
    "OUTSIDE-HIGH": ("above the drift band: the fusion gain (unfused/fused) was LARGER in "
                     "the {arm} arm than in the campaign by more than the drift band "
                     "explains -- which of the two arms' kernels moved, and why, this "
                     "unpaired design cannot say"),
    "OUTSIDE-LOW": ("below the drift band: the fusion gain (unfused/fused) was SMALLER in "
                    "the {arm} arm than in the campaign by more than the drift band "
                    "explains -- which of the two arms' kernels moved, and why, this "
                    "unpaired design cannot say"),
    "NOISE-FLOOR": ("this family defines the drift band and therefore has no verdict of its "
                    "own"),
    "UNRESOLVED-ONE-SIDE": ("UNRESOLVED on at least one side; the delta is undefined and is "
                            "excluded from the band and from every verdict"),
    "MISSING-CONTROL": "not measured in the control arm",
    "MISSING-CAMPAIGN": "not present in the campaign",
    "UNMAPPED": "no (fusion, variant) label for this cell; reported, never aggregated",
    "INCOMPARABLE-BASIS": ("the two sides came from different result files, so the timing "
                           "basis differs (CUDA-graph replay vs L2-flushed wall)"),
    "EXCLUDED-HALF": ("#11b' half-fused publishes its own router speedup rather than going "
                      "through best_speedup, so it is not comparable through this path"),
    "ENGAGEMENT-BROKEN": ("the arm selected a Hopper axis it was supposed to have forced "
                          "off -- it did not engage, and nothing here can be read as a "
                          "control"),
}

FAMILY_VERDICT_SENTENCE = {
    "NOISE-FLOOR": ("defines the drift band; its delta IS the cross-session noise this "
                    "design concedes"),
    "INSIDE-BAND": ("every usable cell sits inside the drift band -- nothing is shown for "
                    "this family, which is not the same claim as 'no effect'"),
    "MIXED": ("the usable cells do not agree: some fall outside the drift band, some "
              "inside, and/or they fall out on both sides; read the per-regime CSVs before "
              "claiming anything"),
    "OUTSIDE-BAND-HIGH": ("a strict majority of usable cells sit ABOVE the drift band and "
                          "none below it: the fusion gain (unfused/fused) was LARGER in the "
                          "{arm} arm by more than the drift band explains"),
    "OUTSIDE-BAND-LOW": ("a strict majority of usable cells sit BELOW the drift band and "
                         "none above it: the fusion gain (unfused/fused) was SMALLER in the "
                         "{arm} arm by more than the drift band explains"),
    "NO-DATA": "no usable cell; nothing can be said",
}

#: EVERY user-facing sentence about a delta goes through these two helpers. The measured
#: quantity is a RATIO OF FUSION SPEEDUPS (unfused/fused) taken in two different sessions:
#: it moves when the UNFUSED chain moves just as much as when the fused kernel moves, so a
#: sentence of the form "arm X was faster/slower" is simply a different claim from the one
#: the number supports. And with an unpaired design no sentence may assert causation ("the
#: levers made it faster"): the only two defensible readings are "outside the drift band"
#: and "inside the drift band, so nothing is shown".
def verdict_sentence(token: str, arm: str) -> str:
    return VERDICT_SENTENCE.get(token, token).format(arm=arm)


def family_verdict_sentence(token: str, arm: str) -> str:
    return FAMILY_VERDICT_SENTENCE[token].format(arm=arm)


# ======================================================================================
# tiny helpers
# ======================================================================================
def fatal(msg: str) -> int:
    print(f"FATAL: {msg}", flush=True)
    return 2


def refuse(msg: str) -> int:
    print("!" * 92, flush=True)
    print(f"REFUSED: {msg}", flush=True)
    print("!" * 92, flush=True)
    return 1


def load_json(path: Path) -> tuple[dict | None, str]:
    """(payload, error). Never raises: a corrupt input is a fact to report, not a traceback."""
    if not path.exists():
        return None, "missing"
    try:
        blob = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        return None, f"unreadable ({type(exc).__name__}: {exc})"
    if not isinstance(blob, dict):
        return None, "not a JSON object"
    return blob, ""


def sha256_of(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        return f"unreadable: {exc}"


def head_commit() -> str:
    """The repo's HEAD, read straight out of `.git` -- no subprocess, no git dependency.

    This is the working tree at REPORT-GENERATION time, which is not necessarily the commit
    that produced the campaign files; the README says so rather than implying provenance it
    does not have.
    """
    git = REPO / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            p = git / ref
            if p.exists():
                return p.read_text().strip()[:12]
            packed = git / "packed-refs"
            if packed.exists():
                for line in packed.read_text().splitlines():
                    if line.endswith(" " + ref):
                        return line.split(" ", 1)[0][:12]
            return "unknown (ref not found)"
        return head[:12]
    except OSError:
        return "unknown (no .git)"


def check_bar_sync(warnings: list[str]) -> None:
    """Re-read FLOOR_US_MAX out of glm52_h200/config.py's SOURCE so the copy cannot drift.

    config.py cannot be imported here (it does `import torch` at module level and this
    report is torch-free by design), so the constant is copied. A copied constant that is
    never checked is a copied constant that goes stale, which is how two verifiers end up
    disagreeing about what a valid harness floor is.
    """
    src = REPO / "glm52_h200" / "config.py"
    try:
        text = src.read_text()
    except OSError:
        warnings.append(f"could not read {src} to cross-check FLOOR_US_MAX; using the "
                        f"local copy {FLOOR_US_MAX}")
        return
    hit = re.search(r"^FLOOR_US_MAX\s*=\s*([0-9.]+)", text, re.M)
    if not hit:
        warnings.append(f"{src} no longer defines FLOOR_US_MAX at top level; the local copy "
                        f"{FLOOR_US_MAX} is now unverified")
        return
    theirs = float(hit.group(1))
    if abs(theirs - FLOOR_US_MAX) > 1e-9:
        warnings.append(f"FLOOR_US_MAX drift: glm52_h200/config.py says {theirs}, this file "
                        f"copies {FLOOR_US_MAX}. config.py owns the number -- fix the copy.")


def regime_class(regime: str) -> str:
    return "decode" if str(regime).startswith("decode") else "prefill"


def uuid8(v: object) -> str:
    s = R._norm_uuid(v)
    return s[:8] if s else ""


def fnum(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ======================================================================================
# cell extraction -- one function, both sides
# ======================================================================================
def cells_for(results_dir: Path, tick_us: float, unresolved_ticks: int
              ) -> tuple[dict[tuple[str, str, str], dict], dict[str, str], list[str]]:
    """{(family, variant, regime) -> cell} for every family file in `results_dir`.

    Uses run_h200.collect_cells, which is the SAME function that produced the campaign's
    summary.json cells. Cell-for-cell parity is therefore by construction rather than by a
    re-derivation that could drift: same speedup preference (paired_speedup > speedup_paired
    > speedup > unfused_ms/fused_ms), same UNRESOLVED/COARSE/DRIFT/SEQUENTIAL flagging, same
    variant naming (f11's sub-arms become variants f11a_w13 / f11b_router / combined via
    walk_rows' parent_key).

    Returns (cells, {family: result-file basename}, notes).

    DUPLICATE TIE-BREAK: FIRST occurrence wins, and every later one is counted into `notes`.
    `rows_by_cell` uses the SAME rule deliberately -- it used to `setdefault` (first) while
    this function overwrote (last), so a resumed arm that appended a second row for one
    regime would have published the later row's speedup next to the earlier row's
    `classic_axes_selected`, and `classic_axes_selected` is what the ENGAGEMENT-BROKEN
    refusal reads. Two views of one cell must describe the same measurement, and a duplicate
    must be reported rather than silently resolved.
    """
    notes: list[str] = []
    log = lambda msg="": notes.append(str(msg))  # noqa: E731 -- collect_cells wants a callable
    cells: dict[tuple[str, str, str], dict] = {}
    files: dict[str, str] = {}
    dups: list[str] = []
    for fam in R.FAMILIES:
        if fam.key == "layer":
            continue  # excluded from the control arm by operator decision (16 h timeout)
        path = R.find_result(fam, results_dir)
        if path is None:
            continue
        files[fam.key] = path.name
        for c in R.collect_cells(log, fam.key, path, tick_us, unresolved_ticks):
            key = (c["family"], c["variant"], c["regime"])
            if key in cells:
                dups.append("/".join(key))
                continue
            cells[key] = c
    if dups:
        notes.append(f"{len(dups)} duplicate (family, variant, regime) cell(s) in "
                     f"{results_dir}: {', '.join(sorted(set(dups))[:8])}"
                     f"{' ...' if len(set(dups)) > 8 else ''}. The FIRST occurrence is "
                     f"published on both the speedup and the cfg/axes side; the later "
                     f"one(s) are dropped. A duplicated cell usually means a resumed run "
                     f"appended a second measurement -- decide which one is the run.")
    return cells, files, notes


def rows_by_cell(results_dir: Path) -> tuple[dict[tuple[str, str, str], dict], list[str]]:
    """The RAW bench row behind each cell, so the cfg dicts can be scanned for Hopper axes.

    Same FIRST-wins duplicate tie-break as `cells_for`; see the note there for why the two
    must agree. Returns (rows, notes)."""
    out: dict[tuple[str, str, str], dict] = {}
    notes: list[str] = []
    dups: list[str] = []
    for fam in R.FAMILIES:
        if fam.key == "layer":
            continue
        path = R.find_result(fam, results_dir)
        if path is None:
            continue
        payload, err = load_json(path)
        if payload is None:
            continue
        rows: list[dict] = []
        R.walk_rows(payload, rows)
        for r in rows:
            key = (fam.key, str(r.get("variant", "-")), str(r.get("regime", "")))
            if key in out:
                dups.append("/".join(key))
                continue
            out[key] = r
    if dups:
        notes.append(f"{len(dups)} duplicate raw bench row(s) in {results_dir}: "
                     f"{', '.join(sorted(set(dups))[:8])}"
                     f"{' ...' if len(set(dups)) > 8 else ''}. The FIRST is used for the "
                     f"cfg/axes columns, matching the cell tie-break.")
    return out, notes


def axes_in_row(row: dict | None) -> list[str]:
    """Every Hopper axis the tuner actually selected anywhere in this row's configs.

    Scans only keys whose name contains `cfg` (`fused_cfg`, `unfused_cfg`,
    `unfused_gemm_cfg`, `unfused_act_cfg`, `unfused_norm_cfg`, `best_cfg`) and lets
    `hop_axes` recurse into their sub-configs -- f08f09's polymorphic `{seed, gemm}`, f01's
    nested `EPI`, f03's `{add, norm}`. Deliberately NOT a blind whole-row walk: a tuner
    table lists candidates that were OFFERED, and a candidate carrying `USE_TMA` is not
    evidence that the winner used it. `None` sub-configs (f04f05's `unfused_cfg.topk` is
    null in 22 of 44 rows) are tolerated.
    """
    got: list[str] = []

    def visit(node) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and "cfg" in k.lower():
                    for a in hop_axes(v):
                        if a not in got:
                            got.append(a)
                visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)

    visit(row or {})
    return got


def canonical_axes(selected: str) -> list[str]:
    """"TMA, clusters(num_ctas=2)" -> ["tma", "clusters"].

    `hop_axes` speaks display labels; the arm table speaks the canonical axis tokens
    (`tma` / `warp_specialize` / `clusters`) that `run_control_h200.ARMS[*].axes_off` uses.
    The engagement gate compares the two, so the translation has to be explicit rather than
    a substring guess. An unrecognised label is kept verbatim so it can never silently
    become "no axis selected".
    """
    out: list[str] = []
    for tok in str(selected or "").split(","):
        tok = tok.strip()
        if not tok or tok == "none":
            continue
        low = tok.lower()
        for prefix, axis in _AXIS_LABEL_PREFIX:
            if low.startswith(prefix):
                if axis not in out:
                    out.append(axis)
                break
        else:
            if tok not in out:
                out.append(tok)
    return out


def offered_axes_for(campaign_dir: Path) -> dict[str, list[str]]:
    """{family: sorted axes the CAMPAIGN offered it}, read from the campaign files.

    Iterate `fairness.h200_axes.per_family` -- never string-build the key. f08f09's key is
    `f08f09_down_merge` (not `..._resadd`) and f11 has three (`f11a_w13_gemm`,
    `f11b_router_gemm`, `f11_norm_kernel`). An axis counts as offered when it is offered
    under ANY per_family key of that file.
    """
    out: dict[str, list[str]] = {}
    for fam in R.FAMILIES:
        if fam.key == "layer":
            continue
        path = R.find_result(fam, campaign_dir)
        payload, _ = load_json(path) if path else (None, "missing")
        offered: set[str] = set()
        pf = (((payload or {}).get("fairness") or {}).get("h200_axes") or {}) \
            .get("per_family") or {}
        for block in pf.values():
            for axis, rec in ((block or {}).get("axes") or {}).items():
                if isinstance(rec, dict) and rec.get("offered") is True:
                    offered.add(axis)
        out[fam.key] = [a for a in _AXIS_ORDER if a in offered]
    return out


def floors_for(results_dir: Path) -> dict[str, float | None]:
    """{family: fairness.timing.harness_floor_us} for every family file present."""
    out: dict[str, float | None] = {}
    for fam in R.FAMILIES:
        if fam.key == "layer":
            continue
        path = R.find_result(fam, results_dir)
        if path is None:
            continue
        payload, _ = load_json(path)
        out[fam.key] = fnum((((payload or {}).get("fairness") or {})
                             .get("timing") or {}).get("harness_floor_us"))
    return out


def uuids_for(results_dir: Path) -> dict[str, str]:
    """{family: normalised env.uuid} -- the per-file card record, not the campaign-wide one.

    Reading only the campaign-wide uuid is exactly the mistake that would have hidden the
    first campaign's silent two-card split (make_report_h200.py:379-390).
    """
    out: dict[str, str] = {}
    for fam in R.FAMILIES:
        if fam.key == "layer":
            continue
        path = R.find_result(fam, results_dir)
        if path is None:
            continue
        payload, _ = load_json(path)
        uu = ((payload or {}).get("env") or {}).get("uuid") \
            or ((((payload or {}).get("fairness") or {}).get("gpu")) or {}).get("uuid")
        out[fam.key] = R._norm_uuid(uu)
    return out


# ======================================================================================
# THE NOISE FLOOR -- the intellectual core of this report
# ======================================================================================
def band_stats(samples: list[float]) -> dict:
    """min / p10 / median / p90 / max / stdev of a list of signed relative deltas.

    p10 and p90 use `statistics.quantiles(..., n=10, method="inclusive")`, i.e. linear
    interpolation between order statistics. At n=22 the 10th percentile is interpolated
    between the 2nd and 3rd smallest values and the 90th between the 20th and 21st -- which
    is why min/max, not the percentiles, is the DEFAULT band (see `--band`).
    """
    s = sorted(float(x) for x in samples)
    n = len(s)
    if n == 0:
        return {"n": 0, "min": None, "p10": None, "median": None, "p90": None,
                "max": None, "stdev": None}
    p10 = p90 = None
    if n >= 2:
        q = statistics.quantiles(s, n=10, method="inclusive")
        p10, p90 = q[0], q[8]
    return {
        "n": n,
        "min": s[0],
        "p10": p10,
        "median": statistics.median(s),
        "p90": p90,
        "max": s[-1],
        "stdev": statistics.stdev(s) if n >= 2 else 0.0,
    }


def band_edges(stats: dict, mode: str) -> tuple[float | None, float | None]:
    """(lo, hi) for a band, under the chosen statistic. Falls back to min/max when a
    percentile is undefined (n < 2), because a band with a None edge cannot judge anything."""
    if not stats or not stats.get("n"):
        return None, None
    if mode == "p10p90" and stats.get("p10") is not None:
        return stats["p10"], stats["p90"]
    return stats["min"], stats["max"]


def build_noise_floor(joined: list[dict], mode: str) -> dict:
    """Everything the drift band is made of, computed from the f03/f10 cells only.

    WHY THIS EXISTS, in one paragraph, because a future reader will otherwise see two
    families being special-cased and delete it:

    The operator chose an unpaired design -- the control arm is measured in a fresh session
    and diffed against a campaign from 2026-08-07. Anything that changed between the two
    sessions (thermals, another tenant, clock state, driver state) shows up in every cell's
    delta on top of whatever the Hopper levers did. `f03` and `f10` advertise no Hopper cfg
    key at all, so `GLM52_H200_CLASSIC=1` cannot change a single config they run; their
    delta is therefore a direct measurement of the session-to-session variation, and it is
    the ONLY thing standing between this report and an uninterpretable comparison. Remove
    it and every other family's number becomes unreadable.

    The statistic is the signed RELATIVE delta `d = classic/hopper - 1`, not the absolute
    difference: f03 runs at ~2.16x at decode_bs1 and f10 lives on a different scale, and an
    absolute band would simply be whichever family has the larger speedup.
    """
    samples: list[dict] = []
    for row in joined:
        if row["family"] not in NOISE_FAMILIES:
            continue
        usable = row["verdict"] not in ("UNRESOLVED-ONE-SIDE", "MISSING-CONTROL",
                                        "MISSING-CAMPAIGN", "UNMAPPED",
                                        "INCOMPARABLE-BASIS", "EXCLUDED-HALF") \
            and row["delta_rel"] is not None
        why = "" if usable else row["verdict"]
        samples.append({"family": row["family"], "variant": row["variant_raw"],
                        "regime": row["regime"],
                        "hopper_speedup": row["hopper_speedup"],
                        "classic_speedup": row["classic_speedup"],
                        "delta_rel": row["delta_rel"], "delta_abs": row["delta_abs"],
                        "usable": usable, "why_unusable": why})

    def vals(pred) -> list[float]:
        return [s["delta_rel"] for s in samples if s["usable"] and pred(s)]

    glob = band_stats(vals(lambda s: True))
    dec = band_stats(vals(lambda s: regime_class(s["regime"]) == "decode"))
    pre = band_stats(vals(lambda s: regime_class(s["regime"]) == "prefill"))
    by_regime = {}
    for reg in REGIMES:
        by_regime[reg] = band_stats(vals(lambda s, r=reg: s["regime"] == r))
    return {"samples": samples, "mode": mode, "global": glob, "decode": dec,
            "prefill": pre, "by_regime": by_regime,
            "n_usable": glob["n"], "n_total": len(samples)}


def band_for_cell(nf: dict, regime: str, mode: str
                  ) -> tuple[float | None, float | None, str]:
    """(lo, hi, basis) -- a cell is judged against its own class band when that class has
    enough samples, otherwise against the global band.

    decode has 18 samples and qualifies; prefill has 4 and does not, so prefill cells are
    judged against a global band dominated by decode's launch-latency-bound variance. That
    is a real weakness of this design and the README says so instead of hiding it.
    """
    cls = regime_class(regime)
    stats = nf.get(cls) or {}
    if stats.get("n", 0) >= CLASS_MIN_N:
        lo, hi = band_edges(stats, mode)
        return lo, hi, f"{cls}-{mode}"
    lo, hi = band_edges(nf["global"], mode)
    return lo, hi, f"global-{mode}"


# ======================================================================================
# the join
# ======================================================================================
def join_cells(camp: dict, ctrl: dict, camp_rows: dict, ctrl_rows: dict,
               offered: dict[str, list[str]], engagement: dict[str, str],
               camp_files: dict[str, str], ctrl_files: dict[str, str],
               arm: str) -> tuple[list[dict], list[str]]:
    """One row per (fusion, variant, regime), campaign side joined to control side."""
    notes: list[str] = []
    keys = sorted(set(camp) | set(ctrl))
    incomparable_families = {
        fam for fam in set(camp_files) & set(ctrl_files)
        if camp_files[fam] != ctrl_files[fam]
    }
    for fam in sorted(incomparable_families):
        notes.append(f"{fam}: campaign published {camp_files[fam]} but the control arm "
                     f"published {ctrl_files[fam]}; those two files do not share a timing "
                     f"basis (CUDA-graph replay vs L2-flushed wall), so every {fam} cell is "
                     f"marked INCOMPARABLE-BASIS and excluded from every verdict")

    out: list[dict] = []
    for family, variant, regime in keys:
        c_hop = camp.get((family, variant, regime))
        c_cls = ctrl.get((family, variant, regime))
        ref = c_hop or c_cls
        # The label lookup make_report_h200.main() does UNGUARDED at line 1896: a variant
        # string it has never seen raises KeyError *after* the CSVs are on disk. Guarded.
        mapped = NAME.get((family, variant))
        if mapped:
            fusion, variant_label = mapped
        elif variant in HALF_VARIANTS:
            fusion, variant_label = HALF
        else:
            fusion, variant_label = R.fusion_label(family, variant), variant

        row = {
            "family": family, "variant_raw": variant, "regime": regime,
            "fusion": fusion, "variant": variant_label, "arm": arm,
            "offered_axes": "|".join(offered.get(family, [])) or "none",
            "engagement": engagement.get(family, ""),
            "hopper_speedup": (c_hop or {}).get("speedup"),
            "classic_speedup": (c_cls or {}).get("speedup"),
            "hopper_speedup_raw": (c_hop or {}).get("speedup_raw"),
            "classic_speedup_raw": (c_cls or {}).get("speedup_raw"),
            "hopper_fused_ms": (c_hop or {}).get("fused_ms"),
            "classic_fused_ms": (c_cls or {}).get("fused_ms"),
            "hopper_unfused_ms": (c_hop or {}).get("unfused_ms"),
            "classic_unfused_ms": (c_cls or {}).get("unfused_ms"),
            "hopper_flags": "; ".join((c_hop or {}).get("flags") or []),
            "classic_flags": "; ".join((c_cls or {}).get("flags") or []),
            "ratio": None, "delta_abs": None, "delta_rel": None, "delta_rel_raw": None,
            "band_lo": None, "band_hi": None, "band_basis": "",
        }

        hr = camp_rows.get((family, variant, regime))
        cr = ctrl_rows.get((family, variant, regime))
        row["hopper_axes_selected"] = ", ".join(axes_in_row(hr)) or "none"
        row["classic_axes_selected"] = ", ".join(axes_in_row(cr)) or "none"
        row["hopper_mapping_fused"] = m((hr or {}).get("fused_cfg")) if hr else ""
        row["classic_mapping_fused"] = m((cr or {}).get("fused_cfg")) if cr else ""
        row["hop_note_hopper"] = hop_note((hr or {}).get("fused_cfg"),
                                          (hr or {}).get("unfused_cfg")) if hr else ""
        row["hop_note_classic"] = hop_note((cr or {}).get("fused_cfg"),
                                           (cr or {}).get("unfused_cfg")) if cr else ""

        # --- verdict precedence: structural problems first, statistics only if comparable
        if variant in HALF_VARIANTS:
            row["verdict"] = "EXCLUDED-HALF"
        elif not mapped:
            row["verdict"] = "UNMAPPED"
        elif family in incomparable_families:
            row["verdict"] = "INCOMPARABLE-BASIS"
        elif c_cls is None:
            row["verdict"] = "MISSING-CONTROL"
        elif c_hop is None:
            row["verdict"] = "MISSING-CAMPAIGN"
        else:
            sh, sc = fnum(c_hop.get("speedup")), fnum(c_cls.get("speedup"))
            rh, rc = fnum(c_hop.get("speedup_raw")), fnum(c_cls.get("speedup_raw"))
            if rh and rc:
                row["delta_rel_raw"] = rc / rh - 1.0
            if sh and sc:
                row["ratio"] = sc / sh
                row["delta_rel"] = sc / sh - 1.0
                row["delta_abs"] = sc - sh
                row["verdict"] = "PENDING"   # filled in once the band exists
            else:
                # `speedup` is None exactly when collect_cells called the cell UNRESOLVED.
                # The published delta uses `speedup`; the raw one survives in its own column
                # and is never averaged into the band or a family verdict.
                row["verdict"] = "UNRESOLVED-ONE-SIDE"
        if ref is None:  # unreachable; keeps the shape honest if collect_cells ever changes
            row["verdict"] = "MISSING-CAMPAIGN"
        out.append(row)

    order = {r: i for i, r in enumerate(REGIMES)}
    out.sort(key=lambda r: (order.get(r["regime"], 99), r["fusion"], r["variant"]))
    return out, notes


def apply_verdicts(joined: list[dict], nf: dict, mode: str) -> None:
    """Second pass: the band only exists after the f03/f10 rows have been joined."""
    for row in joined:
        lo, hi, basis = band_for_cell(nf, row["regime"], mode)
        row["band_lo"], row["band_hi"], row["band_basis"] = lo, hi, basis
        if row["verdict"] != "PENDING":
            continue
        if row["family"] in NOISE_FAMILIES:
            row["verdict"] = "NOISE-FLOOR"
            continue
        d = row["delta_rel"]
        if lo is None or hi is None:
            row["verdict"] = "UNRESOLVED-ONE-SIDE"
        elif d > hi:
            row["verdict"] = "OUTSIDE-HIGH"
        elif d < lo:
            row["verdict"] = "OUTSIDE-LOW"
        else:
            row["verdict"] = "INSIDE"


def family_verdicts(joined: list[dict]) -> list[dict]:
    """One aggregate row per (family, variant), with a fixed sentence per verdict token."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in joined:
        if row["verdict"] in ("UNMAPPED", "EXCLUDED-HALF"):
            continue
        groups.setdefault((row["family"], row["variant_raw"]), []).append(row)
    out = []
    for (family, variant), rows in groups.items():
        usable = [r for r in rows if r["verdict"] in ("INSIDE", "OUTSIDE-HIGH",
                                                      "OUTSIDE-LOW", "NOISE-FLOOR")
                  and r["delta_rel"] is not None]
        ds = [r["delta_rel"] for r in usable]
        hi = sum(1 for r in rows if r["verdict"] == "OUTSIDE-HIGH")
        lo = sum(1 for r in rows if r["verdict"] == "OUTSIDE-LOW")
        inside = sum(1 for r in rows if r["verdict"] == "INSIDE")
        # A family-level OUTSIDE-BAND verdict requires a STRICT majority of the usable cells
        # on one side AND no cell at all on the other. The old rule
        # (`hi > lo and hi >= max(1, len(usable) // 2)`) let 5 of 11 cells -- 45%, with 4
        # cells moving the opposite way -- publish "the majority sit ABOVE the band" and
        # erase the 4. Same-family cells falling out on both sides is exactly what
        # drift-driven scatter looks like, and MIXED is the token that exists for it.
        n_u = len(usable)
        if family in NOISE_FAMILIES:
            verdict = "NOISE-FLOOR"
        elif not usable:
            verdict = "NO-DATA"
        elif hi + lo == 0:
            verdict = "INSIDE-BAND"
        elif hi > n_u / 2 and lo == 0:
            verdict = "OUTSIDE-BAND-HIGH"
        elif lo > n_u / 2 and hi == 0:
            verdict = "OUTSIDE-BAND-LOW"
        else:
            verdict = "MIXED"
        # The counts ride along in the sentence for EVERY verdict, so no reader has to open
        # family_verdicts.csv to discover that the cells disagreed.
        counts = (f" [{n_u} usable cell(s), every one of them a SAMPLE of the band rather "
                  f"than a measurement against it]" if verdict == "NOISE-FLOOR" else
                  f" [{hi} of {n_u} usable cells above the band, {lo} below, "
                  f"{inside} inside]")
        out.append({
            "fusion": rows[0]["fusion"], "variant": rows[0]["variant"], "family": family,
            "offered_axes": rows[0]["offered_axes"], "engagement": rows[0]["engagement"],
            "n_cells": len(rows), "n_usable": len(usable),
            "n_outside_high": hi, "n_outside_low": lo, "n_inside": inside,
            "median_delta_rel": statistics.median(ds) if ds else None,
            "min_delta_rel": min(ds) if ds else None,
            "max_delta_rel": max(ds) if ds else None,
            # A family spans decode and prefill, which can be judged against different
            # bands; name every basis actually used rather than the first row's.
            "band_basis": "|".join(sorted({r["band_basis"] for r in rows})),
            "verdict": verdict,
            "verdict_sentence": family_verdict_sentence(verdict, rows[0]["arm"]) + counts,
        })
    out.sort(key=lambda r: (r["family"], r["variant"]))
    return out


# ======================================================================================
# engagement audit trail
# ======================================================================================
def engagement_rows(control_dir: Path, arm: str) -> tuple[list[dict], list[str]]:
    """Flatten `_engagement/<arm>/<family>.verify.json` into CSV rows."""
    notes: list[str] = []
    base = control_dir / ENGAGE_DIRNAME / arm
    rows: list[dict] = []
    if not base.is_dir():
        notes.append(f"no engagement directory at {base}; the driver writes one "
                     f".verify.json per family and its absence means the arm was never "
                     f"verified on the machine that produced it")
        return rows, notes
    for path in sorted(base.glob("*.verify.json")):
        payload, err = load_json(path)
        if payload is None:
            notes.append(f"{path.name}: {err}")
            continue
        fam = str(payload.get("family") or path.name.split(".")[0])
        for chk in payload.get("checks") or []:
            if not isinstance(chk, dict):
                continue
            rows.append({
                "arm": str(payload.get("arm") or arm),
                "family": fam,
                "check_id": str(chk.get("id", "")),
                "axis": str(chk.get("axis") or ""),
                "json_path": str(chk.get("path", "")),
                "want": str(chk.get("want", "")),
                "got": str(chk.get("got", "")),
                "status": str(chk.get("status", "")),
                "detail": str(chk.get("detail", "")),
            })
    return rows, notes


# ======================================================================================
# provenance -- the two sessions, side by side
# ======================================================================================
def _hw_row(summary: dict, which: str, uuid: str) -> dict:
    for r in summary.get(which) or []:
        if isinstance(r, dict) and R._norm_uuid(r.get("uuid")) == uuid:
            return r
    rows = summary.get(which) or []
    return rows[0] if rows and isinstance(rows[0], dict) else {}


def _match_frac(timer: dict, preflight: dict) -> tuple[float | None, str]:
    """(timer tick-match fraction, where it came from) -- (None, "") when nobody recorded it.

    `run_h200.py` copies the preflight's `timer_tick_match_frac` into `timer.match_frac`;
    `run_control_h200.py` builds its own timer block and older tarballs may not carry it at
    all. The preflight digest (`preflight.timer_tick_match_frac`) is written by both. Same
    number, two homes -- read either, and say which one was read.
    """
    v = fnum((timer or {}).get("match_frac"))
    if v is not None:
        return v, "timer.match_frac"
    v = fnum((preflight or {}).get("timer_tick_match_frac"))
    if v is not None:
        return v, "preflight.timer_tick_match_frac"
    return None, ""


def _drift_row(summary: dict, uuid: str, index: object) -> dict:
    for r in summary.get("hwinfo_drift") or []:
        if isinstance(r, dict) and str(r.get("index")) == str(index):
            return r
    rows = summary.get("hwinfo_drift") or []
    return rows[0] if rows and isinstance(rows[0], dict) else {}


def provenance_rows(camp_sum: dict, ctrl_sum: dict, camp_floors: dict, ctrl_floors: dict,
                    camp_uuids: dict, ctrl_uuids: dict) -> list[dict]:
    rows: list[dict] = []

    def add(item, a, b, severity_when_differs="info", match=None, fmt=str):
        same = (a == b) if match is None else match
        rows.append({"item": item, "campaign": "" if a is None else fmt(a),
                     "control": "" if b is None else fmt(b),
                     "match": "yes" if same else "no",
                     "severity": "info" if same else severity_when_differs})

    cu = R._norm_uuid((camp_sum.get("_meta") or {}).get("gpu_uuid"))
    xu = R._norm_uuid((ctrl_sum.get("_meta") or {}).get("gpu_uuid"))
    add("gpu_uuid", cu or "?", xu or "?", "REFUSE")
    add("device_name", R._norm_dev((camp_sum.get("_meta") or {}).get("device")),
        R._norm_dev((ctrl_sum.get("_meta") or {}).get("device")), "REFUSE")

    for fam in [f.key for f in R.FAMILIES if f.key != "layer"]:
        a, b = camp_floors.get(fam), ctrl_floors.get(fam)
        if a is None or b is None:
            # One side simply did not run this family. That is already reported per cell as
            # MISSING-CONTROL / MISSING-CAMPAIGN; calling it a floor mismatch here would be
            # a second, misleading alarm for the same fact.
            rows.append({"item": f"harness_floor_us_{fam}",
                         "campaign": "" if a is None else f"{a:.3f}",
                         "control": "" if b is None else f"{b:.3f}",
                         "match": "n/a", "severity": "info"})
            continue
        ok = abs(a - b) <= FLOOR_DELTA_WARN_US
        sev = "info" if ok else "FLAG"
        if b > FLOOR_US_MAX or a > FLOOR_US_MAX:
            ok, sev = False, "REFUSE"
        rows.append({"item": f"harness_floor_us_{fam}",
                     "campaign": f"{a:.3f}", "control": f"{b:.3f}",
                     "match": "yes" if ok else "no", "severity": sev})

    ct, xt = camp_sum.get("timer") or {}, ctrl_sum.get("timer") or {}
    cp, xp = camp_sum.get("preflight") or {}, ctrl_sum.get("preflight") or {}
    add("timer_tick_us", ct.get("tick_us"), xt.get("tick_us"), "FLAG")

    # The timer tick-match fraction lives in TWO places and neither is guaranteed:
    # `run_h200.py` copies it into `timer.match_frac`, while both drivers always carry the
    # preflight digest's `preflight.timer_tick_match_frac`. Read the timer block first and
    # fall back to the preflight digest. A MISSING value is NOT evidence of a contended
    # card -- it is an older or differently-built summary -- so it is a FLAG, never a
    # REFUSE. Only a value that is present AND below the bar refuses. (Scoring the absence
    # as REFUSE is what made every honest control run publishable only under --force-floor,
    # i.e. branded "PUBLISHED OVER A REFUSAL".)
    mf_c, mf_c_src = _match_frac(ct, cp)
    mf_x, mf_x_src = _match_frac(xt, xp)
    if mf_x is None:
        mf_match, mf_sev = "unknown", "FLAG"
    elif mf_x >= TICK_MATCH_MIN:
        mf_match, mf_sev = "yes", "info"
    else:
        mf_match, mf_sev = "no", "REFUSE"
    rows.append({"item": "timer_match_frac",
                 "campaign": "unknown" if mf_c is None else f"{mf_c:.3f} ({mf_c_src})",
                 "control": "unknown (neither timer.match_frac nor "
                            "preflight.timer_tick_match_frac was recorded)"
                            if mf_x is None else f"{mf_x:.3f} ({mf_x_src})",
                 "match": mf_match, "severity": mf_sev})
    add("unresolved_ticks", ct.get("unresolved_ticks"), xt.get("unresolved_ticks"), "FLAG")


    add("preflight_recorded_at", cp.get("timestamp"), xp.get("timestamp"), "info")
    add("torch_version", cp.get("torch"), xp.get("torch"), "FLAG")
    add("triton_version", cp.get("triton"), xp.get("triton"), "FLAG")

    ch = _hw_row(camp_sum, "hwinfo_start", cu)
    xh = _hw_row(ctrl_sum, "hwinfo_start", xu)
    add("driver_version", ch.get("driver_version"), xh.get("driver_version"), "FLAG")
    cd = _drift_row(camp_sum, cu, ch.get("index"))
    xd = _drift_row(ctrl_sum, xu, xh.get("index"))
    for field, label in (("start", "sm_clock_start"), ("end", "sm_clock_end"),
                         ("pct", "sm_clock_drift_pct")):
        a = ((cd.get("clocks.sm") or {}) if isinstance(cd.get("clocks.sm"), dict) else {}) \
            .get(field)
        b = ((xd.get("clocks.sm") or {}) if isinstance(xd.get("clocks.sm"), dict) else {}) \
            .get(field)
        rows.append({"item": label, "campaign": "" if a is None else f"{float(a):.1f}",
                     "control": "" if b is None else f"{float(b):.1f}",
                     "match": "n/a", "severity": "info"})

    add("tenant_events", len((camp_sum.get("gpu") or {}).get("tenant_events") or []),
        len((ctrl_sum.get("gpu") or {}).get("tenant_events") or []), "FLAG")
    add("session_recorded_at", (camp_sum.get("_meta") or {}).get("recorded_at"),
        (ctrl_sum.get("_meta") or {}).get("recorded_at"), "info")
    add("wall_s", (camp_sum.get("driver") or {}).get("wall_s"),
        (ctrl_sum.get("driver") or {}).get("wall_s"), "info",
        fmt=lambda v: f"{float(v):.0f}")
    # Card-record per family: the campaign already split across two cards once.
    for fam in [f.key for f in R.FAMILIES if f.key != "layer"]:
        a, b = camp_uuids.get(fam, ""), ctrl_uuids.get(fam, "")
        if not a or not b:
            rows.append({"item": f"env_uuid_{fam}", "campaign": a[:8], "control": b[:8],
                         "match": "n/a", "severity": "info"})
            continue
        rows.append({"item": f"env_uuid_{fam}", "campaign": a[:8], "control": b[:8],
                     "match": "yes" if a == b else "no",
                     "severity": "info" if a == b else "REFUSE"})
    return rows


# ======================================================================================
# writers
# ======================================================================================
CELL_COLUMNS = [
    "fusion", "variant", "family", "regime", "arm",
    "offered_axes", "engagement",
    "hopper_speedup", "classic_speedup", "ratio", "delta_abs", "delta_rel",
    "band_lo", "band_hi", "band_basis", "verdict",
    "hopper_fused_ms", "classic_fused_ms", "hopper_unfused_ms", "classic_unfused_ms",
    "hopper_axes_selected", "classic_axes_selected",
    "hopper_mapping_fused", "classic_mapping_fused",
    "hopper_flags", "classic_flags",
    "notes",
    # Appended AFTER the fixed order so the spec's column list stays an exact prefix: the
    # raw-speedup delta for cells that are UNRESOLVED on one side and therefore excluded
    # from the band and every verdict.
    "delta_rel_raw",
]


def cell_note(row: dict, provenance_sentence: str, banner: str, arm: str) -> str:
    bits = []
    if banner:
        bits.append(banner)
    if row["hopper_flags"]:
        bits.append("campaign: " + row["hopper_flags"])
    if row["classic_flags"]:
        bits.append("control: " + row["classic_flags"])
    if row["hop_note_hopper"]:
        bits.append("campaign " + row["hop_note_hopper"])
    if row["hop_note_classic"]:
        bits.append("control " + row["hop_note_classic"])
    bits.append(verdict_sentence(row["verdict"], arm))
    bits.append(provenance_sentence)
    return "; ".join(b for b in bits if b)


def write_cell_csvs(out: Path, joined: list[dict], arm: str, provenance_sentence: str,
                    banner: str) -> list[Path]:
    written = []
    regimes = R.order_regimes({r["regime"] for r in joined})
    for regime in regimes:
        path = out / f"control_{regime}.csv"
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CELL_COLUMNS)
            w.writeheader()
            for row in [r for r in joined if r["regime"] == regime]:
                w.writerow({
                    "fusion": row["fusion"], "variant": row["variant"],
                    "family": row["family"], "regime": row["regime"], "arm": arm,
                    "offered_axes": row["offered_axes"], "engagement": row["engagement"],
                    "hopper_speedup": r4(row["hopper_speedup"]),
                    "classic_speedup": r4(row["classic_speedup"]),
                    "ratio": r4(row["ratio"]), "delta_abs": r4(row["delta_abs"]),
                    "delta_rel": r4(row["delta_rel"]),
                    "band_lo": r4(row["band_lo"]), "band_hi": r4(row["band_hi"]),
                    "band_basis": row["band_basis"], "verdict": row["verdict"],
                    "hopper_fused_ms": r4(row["hopper_fused_ms"], 6),
                    "classic_fused_ms": r4(row["classic_fused_ms"], 6),
                    "hopper_unfused_ms": r4(row["hopper_unfused_ms"], 6),
                    "classic_unfused_ms": r4(row["classic_unfused_ms"], 6),
                    "hopper_axes_selected": row["hopper_axes_selected"],
                    "classic_axes_selected": row["classic_axes_selected"],
                    "hopper_mapping_fused": row["hopper_mapping_fused"],
                    "classic_mapping_fused": row["classic_mapping_fused"],
                    "hopper_flags": row["hopper_flags"],
                    "classic_flags": row["classic_flags"],
                    "notes": cell_note(row, provenance_sentence, banner, arm),
                    "delta_rel_raw": r4(row["delta_rel_raw"]),
                })
        written.append(path)
    return written


def write_noise_floor(out: Path, nf: dict) -> None:
    with (out / "noise_floor.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "variant", "regime", "hopper_speedup", "classic_speedup",
                    "delta_rel", "delta_abs", "usable", "why_unusable"])
        order = {r: i for i, r in enumerate(REGIMES)}
        for s in sorted(nf["samples"], key=lambda s: (s["family"],
                                                      order.get(s["regime"], 99))):
            w.writerow([s["family"], s["variant"], s["regime"], r4(s["hopper_speedup"]),
                        r4(s["classic_speedup"]), r4(s["delta_rel"]), r4(s["delta_abs"]),
                        "yes" if s["usable"] else "no", s["why_unusable"]])

    with (out / "noise_floor_by_class.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["basis", "n", "min", "p10", "median", "p90", "max", "stdev",
                    "used_for_verdicts"])

        def emit(name: str, st: dict, used: str) -> None:
            w.writerow([name, st["n"], r4(st["min"]), r4(st["p10"]), r4(st["median"]),
                        r4(st["p90"]), r4(st["max"]), r4(st["stdev"]), used])

        emit("global", nf["global"], "yes (fallback for any class with n < %d)" % CLASS_MIN_N)
        emit("decode", nf["decode"],
             "yes" if nf["decode"]["n"] >= CLASS_MIN_N else f"no (n < {CLASS_MIN_N})")
        emit("prefill", nf["prefill"],
             "yes" if nf["prefill"]["n"] >= CLASS_MIN_N else f"no (n < {CLASS_MIN_N})")
        for reg in REGIMES:
            emit(f"regime:{reg}", nf["by_regime"][reg],
                 "no (INFO only; 2 samples cannot define a band)")


def write_family_verdicts(out: Path, fams: list[dict]) -> None:
    cols = ["fusion", "variant", "family", "offered_axes", "engagement", "n_cells",
            "n_usable", "n_outside_high", "n_outside_low", "n_inside", "median_delta_rel",
            "min_delta_rel", "max_delta_rel", "band_basis", "verdict", "verdict_sentence"]
    with (out / "family_verdicts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for f in fams:
            row = dict(f)
            for k in ("median_delta_rel", "min_delta_rel", "max_delta_rel"):
                row[k] = r4(row[k])
            w.writerow(row)


def write_engagement(out: Path, rows: list[dict]) -> None:
    cols = ["arm", "family", "check_id", "axis", "json_path", "want", "got", "status",
            "detail"]
    with (out / "engagement.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_provenance(out: Path, rows: list[dict]) -> None:
    cols = ["item", "campaign", "control", "match", "severity"]
    with (out / "provenance.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ======================================================================================
# README
# ======================================================================================
def _mdcell(c) -> str:
    """A pipe inside a cell silently breaks a markdown table, and `offered_axes` is
    pipe-joined (`tma|warp_specialize|clusters`). Escape rather than re-spell the token, so
    the README and the CSV say literally the same thing."""
    return "" if c is None else str(c).replace("|", "\\|").replace("\n", " ")


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(_mdcell(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_mdcell(c) for c in r) + " |")
    return "\n".join(out)


def sig(x, nd: int = 3) -> str:
    """Signed, because every number it formats is a delta and the sign is the point."""
    return "n/a" if _nan(x) else f"{float(x):+.{nd}f}"


def mag(x, nd: int = 3) -> str:
    """Unsigned, for quantities like a standard deviation that have no direction."""
    return "n/a" if _nan(x) else f"{float(x):.{nd}f}"


def resolving_power(lo, hi) -> tuple[float | None, float | None]:
    """(minimum detectable effect, band half-width) for a band [lo, hi].

    An effect only clears the band when it exceeds `hi` upward or falls below `lo` downward,
    so the smallest |delta| this report can call OUTSIDE **in either direction** is
    `max(|lo|, |hi|)`. That number, not the word "nothing", is what an INSIDE verdict
    actually means: a wide band and a tight band produce the same sentence, and without the
    MDE printed next to it an underpowered run is indistinguishable from a well-powered
    null -- which is the very error the f03/f10 apparatus exists to prevent.
    """
    a, b = fnum(lo), fnum(hi)
    if a is None or b is None:
        return None, None
    return max(abs(a), abs(b)), (b - a) / 2.0


def baseline_digest(campaign_dir: Path, fp: dict) -> str:
    """A 12-hex digest of the campaign files AS THEY ARE ON DISK NOW.

    The README used to identify the baseline by the working tree's HEAD commit, which does
    not describe the baseline files at all when the tree is dirty (it is, routinely: the
    campaign JSONs get regenerated in place). This hashes what was actually compared.
    """
    names = sorted(fp) if fp else []
    if not names:
        seen = []
        for fam in R.FAMILIES:
            if fam.key == "layer":
                continue
            p = R.find_result(fam, campaign_dir)
            if p is not None:
                seen.append(p.name)
        if (campaign_dir / "summary.json").exists():
            seen.append("summary.json")
        names = sorted(set(seen))
    if not names:
        return "unknown (no campaign files)"
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode())
        h.update(sha256_of(campaign_dir / name).encode())
    return f"{h.hexdigest()[:12]} over {len(names)} file(s)"


def write_readme(out: Path, arm: str, nf: dict, fams: list[dict], joined: list[dict],
                 prov: list[dict], eng: list[dict], args, camp_sum: dict, ctrl_sum: dict,
                 warnings: list[str], forced: list[str], offered: dict,
                 headline: str) -> None:
    mode = nf["mode"]
    glo = nf["global"]
    lo, hi = band_edges(glo, mode)
    gen = (ctrl_sum.get("_meta") or {}).get("recorded_at") or "unknown"
    lines: list[str] = []
    A = lines.append

    mde, half = resolving_power(lo, hi)
    fp = ((ctrl_sum.get("campaign") or {}).get("fingerprint") or {})
    recorded_digest = ((ctrl_sum.get("campaign") or {}).get("fingerprint_digest")
                       or "not recorded by the driver")

    A(f"# Hopper control arm -- `{arm}` vs the committed H200 campaign")
    A("")
    A(f"**Generated** from the control run recorded {gen} · **Arm** {arm} · "
      f"**Verdict** {headline}")
    A("")
    # The baseline is identified by CONTENT, not by a commit. head_commit() reads the
    # working tree's HEAD, which says nothing about files that have been regenerated in
    # place since that commit -- and in this repo they routinely have been.
    A(f"**Baseline** `{args.campaign_dir}` · content digest "
      f"`{baseline_digest(args.campaign_dir, fp)}` (recomputed here, at report time) · "
      f"driver-recorded fingerprint digest `{recorded_digest}` · report-time repo HEAD "
      f"`{head_commit()}` *(the HEAD of the working tree that generated this report; it may "
      f"be dirty and is NOT a claim about which commit produced the baseline files -- the "
      f"content digest is)*")
    A("")
    A(f"**Resolving power**: the drift band is `[{sig(lo)}, {sig(hi)}]`, so the smallest "
      f"effect this report can call OUTSIDE in either direction is "
      f"**{mag(mde)}** (band half-width {mag(half)}). Any INSIDE verdict below means "
      f"*\"smaller than {mag(mde)}, or absent\"* -- it does not mean zero.")
    if mde is not None and mde > RESOLUTION_TARGET:
        A("")
        A(f"> **UNDERPOWERED.** {mag(mde)} is wider than the {mag(RESOLUTION_TARGET)} "
          f"smallest effect that motivated this experiment (the four TMA-using families "
          f"gained 1.00-1.06x on H200-vs-C500 while `f03`/`f10` gained 1.86x/1.48x). A "
          f"null result at this band width is **not** evidence of no effect; it is evidence "
          f"that this session pair could not resolve one. Re-run as a same-session paired "
          f"A/B before reading anything into an INSIDE verdict.")
    A("")
    if forced:
        A("## UNVERIFIED")
        A("")
        A("This report was published over a refusal. Every number below is suspect:")
        A("")
        for f in forced:
            A(f"* {f}")
        A("")

    A("## 0. What this is, and the one thing it cannot tell you")
    A("")
    A(f"The operator's design decision, honoured exactly: **one arm vs the existing "
      f"campaign** -- run only the `{arm}` arm now and diff it against the "
      f"already-committed `results/h200/*.json`. There is no same-session paired arm.")
    A("")
    A("The cost of that choice is stated plainly and not buried: because the two arms were "
      "measured in different sessions, **cross-session drift -- thermals, co-tenancy, clock "
      "state, driver state -- is confounded with the arm's feature switch.** A delta "
      "between the two numbers is not, on its own, evidence about TMA, warp specialization "
      "or thread-block clusters.")
    A("")
    A("**What the delta actually is.** Every number in this report compares a FUSION GAIN "
      "to a FUSION GAIN: `speedup = unfused_ms / fused_ms`, measured separately in each "
      "session, and `delta_rel = classic_speedup / hopper_speedup - 1`. It is a ratio of "
      "ratios. It moves when the UNFUSED chain moves exactly as much as when the fused "
      "kernel moves, and it can go up while both arms' absolute times go down. No sentence "
      "in this report may therefore be read as *\"arm X ran faster\"* -- the quantity does "
      "not say that, and the raw `*_fused_ms` / `*_unfused_ms` columns are published in "
      "every per-regime CSV precisely so a reader can check which side moved.")
    A("")
    A("The defence is the f03/f10 noise floor described in §2. With it, the only two "
      "conclusions this design supports are *\"the delta exceeds the drift band\"* and "
      "*\"the delta is inside the drift band, so nothing is shown\"*. Neither of them is "
      "*\"the Hopper features did nothing\"*, and neither is a causal claim of any kind: an "
      "unpaired two-session comparison cannot support one.")
    A("")
    A("Two further limits on attribution:")
    A("")
    # This bullet is arm-specific and used to be hardcoded to the classic arm's four
    # capabilities. Now that the other arms can publish at all (their engagement gate was
    # missing), it must describe the arm actually being reported on.
    if arm == "classic":
        A("* `GLM52_H200_CLASSIC=1` forces **four** capabilities off, including `wgmma`, "
          "not just the three levers in the question. A delta on a GEMM-carrying family "
          "(`f01`, `f06`, `f08f09`) cannot be attributed to TMA / warp-spec / clusters "
          "alone from this arm; the per-feature arms exist for that and are behind a flag.")
    elif ARM_AXES_OFF.get(arm):
        A(f"* The `{arm}` arm forces exactly "
          f"`{', '.join(sorted(ARM_AXES_OFF[arm]))}` off and leaves every other capability "
          f"(including `wgmma`) live, so a delta here is at least *about* that axis -- but "
          f"it is still measured across two sessions and still confounded with drift.")
    elif arm in ARM_AXES_OFF:
        A(f"* The `{arm}` arm forces **nothing** off: it is the converse sanity check, a "
          f"re-measurement with every lever live. A delta here is drift plus whatever else "
          f"changed between the two sessions, and nothing else.")
    else:
        A(f"* `{arm}` is not one of the arms this report knows "
          f"({', '.join(sorted(ARM_AXES_OFF))}), so **no axis check could be run** and "
          f"nothing here is verified to have engaged.")
    A("* The band comes from two short memory-bound vector kernels. Whether their "
      "run-to-run variability bounds the variability of a 14-hour GEMM family is an "
      "**assumption, not a measurement**. It is the best available control under the "
      "chosen design.")
    A("")

    A("## 1. Headline")
    A("")
    # One row per (fusion, VARIANT) group -- the same grouping family_verdicts.csv uses.
    # Without the variant column two pairs of rows ("#8 Down + Expert Merge" atomic vs
    # token-major, "#9 ..." likewise) are indistinguishable and carry different numbers.
    rows = []
    for f in fams:
        outside = f["n_outside_high"] + f["n_outside_low"]
        rows.append([f["fusion"], f["variant"], f["family"], f["offered_axes"],
                     f["engagement"] or "-", f"{outside} / {f['n_usable']}",
                     sig(f["median_delta_rel"]), f["verdict"], f["verdict_sentence"]])
    A(md_table(["fusion", "variant", "family", "offered axes", "engagement",
                "cells outside the band", "median delta", "verdict", "reading"], rows))
    A("")
    A(f"Rows are **(fusion, variant) groups**, not families: {len(fams)} groups over "
      f"{len({f['family'] for f in fams})} families. Two groups can share a fusion label "
      f"and differ only in variant, which is why the variant column is not optional.")
    A("")
    A(f"**Noise floor (f03+f10, {glo['n']} usable of {nf['n_total']} cells): "
      f"delta_rel in [{sig(lo)}, {sig(hi)}], median {sig(glo['median'])} "
      f"(basis: global-{mode}); minimum detectable effect {mag(mde)}.**")
    A("")
    A(headline)
    A("")

    A("## 2. The noise floor, and why it is the whole argument")
    A("")
    A("`f03` (ResAdd+RMSNorm) and `f10` (ExpertMerge+ResAdd) advertise no Hopper cfg key at "
      "all -- `kernel_cfg_keys` reads `\"module advertises none\"`, all three axes are "
      "`offered: false` in the campaign, and a full key-path scan of both committed files "
      "finds zero Hopper tokens in any config, tune table or `axis_counts`. Their classic "
      "arm is therefore configured **byte-identically** to their Hopper arm, and their "
      "delta contains only run-to-run and cross-session variation. That is the drift band.")
    A("")
    A("The statistic is the signed relative delta `d = classic_speedup / hopper_speedup - "
      "1`, relative rather than absolute because the two families' speedups live on "
      "different scales and an absolute band would simply be dominated by the larger one.")
    A("")
    A(f"The published band is **min/max**, not a percentile. At n = {glo['n']} a percentile "
      "is barely resolvable -- the 10th and 90th are interpolated between the 2nd/3rd and "
      "20th/21st order statistics -- and the question being asked is *\"could cross-session "
      "drift alone have produced this?\"*, whose conservative answer is the observed "
      "extremes. The p10/p90 band is published alongside in `noise_floor_by_class.csv` and "
      "`--band p10p90` switches which one drives the verdict column.")
    A("")
    A(f"`median(d) = {sig(glo['median'])}` is the **drift bias**: a systematic session "
      "offset that every family's delta should be read relative to, not a result.")
    A("")
    band_rows = []
    for name, key in (("global", "global"), ("decode", "decode"), ("prefill", "prefill")):
        st = nf[key]
        if key == "global":
            used = f"yes -- fallback for any class with n < {CLASS_MIN_N}"
        else:
            used = "yes" if st["n"] >= CLASS_MIN_N else f"no (n = {st['n']} < {CLASS_MIN_N})"
        b_lo, b_hi = band_edges(st, mode)
        b_mde, _ = resolving_power(b_lo, b_hi)
        band_rows.append([name, st["n"], sig(st["min"]), sig(st["p10"]), sig(st["median"]),
                          sig(st["p90"]), sig(st["max"]), mag(st["stdev"]), mag(b_mde),
                          used])
    A(md_table(["basis", "n", "min", "p10", "median", "p90", "max", "stdev",
                f"min detectable effect ({mode})", "used for verdicts"], band_rows))
    A("")
    A(f"**Resolving power.** `min detectable effect` is `max(|lo|, |hi|)` of that basis's "
      f"band under the active `--band {mode}` statistic: the smallest |delta| that could be "
      f"called OUTSIDE in either direction. For the band actually driving the verdicts it "
      f"is **{mag(mde)}**. The smallest effect this study was built to see is "
      f"{mag(RESOLUTION_TARGET)} -- the four TMA-using families gained 1.00-1.06x on "
      f"H200-vs-C500 while `f03`/`f10` gained 1.86x/1.48x, and it is that contrast the "
      f"control arm exists to interrogate. "
      + (f"**{mag(mde)} > {mag(RESOLUTION_TARGET)}: this run is UNDERPOWERED and an INSIDE "
         f"verdict below carries no information about effects smaller than {mag(mde)}.**"
         if (mde is not None and mde > RESOLUTION_TARGET) else
         f"{mag(mde)} <= {mag(RESOLUTION_TARGET)}, so the band is tight enough to resolve "
         f"the motivating effect size."))
    A("")
    A(f"A cell is judged against its own class band when that class has at least "
      f"{CLASS_MIN_N} samples. decode qualifies; **prefill has only 4 samples (2 families x "
      "2 regimes) and therefore falls back to the global band, which is dominated by "
      "decode's launch-latency-bound variance.** A genuine prefill effect could be judged "
      "against an inappropriately wide band. Per-regime bands (n = 2) are published in "
      "`noise_floor_by_class.csv` as INFO only and are never used for a verdict.")
    A("")
    A(f"Full sample: `noise_floor.csv` ({nf['n_total']} rows, {glo['n']} usable; the guard "
      f"refuses to publish below {MIN_NOISE_CELLS}).")
    A("")

    A("## 3. Did the arm actually engage?")
    A("")
    if eng:
        by_fam: dict[str, dict[str, int]] = {}
        for r in eng:
            d = by_fam.setdefault(r["family"], {})
            d[r["status"]] = d.get(r["status"], 0) + 1
        rows = []
        for fam in sorted(by_fam):
            counts = "; ".join(f"{k}={v}" for k, v in sorted(by_fam[fam].items()))
            rows.append([fam, "|".join(offered.get(fam, [])) or "none", counts])
        A(md_table(["family", "axes offered by the campaign", "check outcomes"], rows))
    else:
        A("No `_engagement/<arm>/*.verify.json` files were found in the returned tree. "
          "The driver writes one per family; their absence means the arm's engagement was "
          "never verified on the machine that produced it, and `engagement.csv` is empty.")
    A("")
    A("`f03` and `f10` are **VACUOUS** for the config-level checks: they have nothing to "
      "disable, so a clean token scan for them is not evidence of engagement. The proof for "
      "those two is capability-level -- `available` flips `true -> false` with `evidence` "
      "containing `(source env)`, and `not_offered_because` changes from *\"this kernel "
      "module advertises no cfg key for it\"* to *\"the live capability probe says it is "
      "unavailable\"*. Full audit trail: `engagement.csv`.")
    A("")

    A("## 4. Per-cell results")
    A("")
    A("One CSV per regime:")
    A("")
    for reg in R.order_regimes({r["regime"] for r in joined}):
        n = sum(1 for r in joined if r["regime"] == reg)
        A(f"* `control_{reg}.csv` ({n} rows)")
    A("")
    A("Column glossary:")
    A("")
    A("* `hopper_*` is the committed campaign side; `classic_*` is the `--arm` side -- the "
      "column names stay `classic_*` whatever the arm is, and the `arm` column names it.")
    A("* Each side's `speedup` is that side's own FUSION GAIN, `unfused_ms / fused_ms`. "
      "`ratio = classic_speedup / hopper_speedup` is therefore a ratio of ratios: it rises "
      "when the arm's unfused chain got slower just as readily as when its fused kernel got "
      "faster. Read `*_fused_ms` and `*_unfused_ms` before saying which side moved.")
    A("* `delta_rel = ratio - 1`; `delta_abs = classic_speedup - hopper_speedup` (a "
      "difference of gains, not of milliseconds).")
    A("* `band_lo` / `band_hi` / `band_basis` -- the drift band this particular cell was "
      "judged against.")
    A("* `delta_rel_raw` is populated only for `UNRESOLVED-ONE-SIDE` cells, from "
      "`speedup_raw`; it is excluded from the band and from every verdict.")
    A("* **An empty numeric cell means NOT MEASURED. It is never coerced to zero.**")
    A("")
    A("Verdict tokens, one sentence each:")
    A("")
    for tok in ("OUTSIDE-HIGH", "OUTSIDE-LOW", "INSIDE", "NOISE-FLOOR",
                "UNRESOLVED-ONE-SIDE", "MISSING-CONTROL", "MISSING-CAMPAIGN", "UNMAPPED",
                "INCOMPARABLE-BASIS", "EXCLUDED-HALF"):
        A(f"* `{tok}` -- {verdict_sentence(tok, arm)}.")
    A("")
    A(f"`{HALF[0]}` is excluded by name: it publishes "
      "`half_fused.router_speedup_vs_unfused` directly rather than through `best_speedup`, "
      "so it is not comparable through this path.")
    A("")

    A("## 5. Provenance and the two sessions")
    A("")
    A(md_table(["item", "campaign", "control", "match", "severity"],
               [[r["item"], r["campaign"], r["control"], r["match"], r["severity"]]
                for r in prov]))
    A("")
    if warnings:
        A("Flags raised while generating this report:")
        A("")
        for w in warnings:
            A(f"* {w}")
        A("")

    A("## 6. What would change the verdict")
    A("")
    A("* **The same-session paired A/B the operator declined.** Measuring both arms back to "
      "back on one card removes the cross-session confound entirely and makes the f03/f10 "
      "band a cross-check rather than the only defence.")
    A("* **The per-feature decomposition arms.** They separate the three levers from each "
      "other and from `wgmma`:")
    A("")
    A("      python3 run_control_h200.py --arms no-tma,no-ws,no-clusters")
    A("")
    if arm == "classic":
        A("* `GLM52_H200_CLASSIC=1` also forces `wgmma` off, so **a GEMM-family delta "
          "cannot be attributed to TMA / warp-spec / clusters alone from this arm.** This "
          "report does not pick the convenient attribution.")
    A("")

    A("## 7. Reproduce")
    A("")
    A("On the H200 (the operator's machine; nobody else can reach it):")
    A("")
    A(f"      python3 {REPO}/run_control_h200.py --arms {arm}")
    A("")
    A("Locally, after the tarball comes back -- this is the EXACT command that produced "
      "this report, override flags included, so re-running it reproduces this file rather "
      "than exiting 1 on a gate this run was published over:")
    A("")
    # Every flag that changed the outcome has to appear here. The report used to print the
    # command WITHOUT --force-floor / --force-unverified, so the printed command exited 1
    # against the very inputs the report was built from.
    extra = ""
    if args.band != "minmax":
        extra += f" --band {args.band}"
    if getattr(args, "force_floor", False):
        extra += " --force-floor"
    if getattr(args, "force_unverified", False):
        extra += " --force-unverified"
    A(f"      python3 {REPO}/glm52/make_control_report_h200.py \\")
    A(f"              --control-dir {args.control_dir} --arm {arm} \\")
    A(f"              --campaign-dir {args.campaign_dir} --out {args.out}{extra}")
    if extra.strip():
        A("")
        A(f"(`{extra.strip()}` is part of the command because this report was generated "
          f"with it. A force flag in that line means a validity gate FAILED and was "
          f"overridden -- see the `## UNVERIFIED` section.)")
    A("")
    A("Nothing under `results/h200/_control_arm/` is ever merged into `results/h200/*.json`. "
      "The control arm is a diff, not an append, and there is no merge step in this repo.")
    A("")
    (out / "README.md").write_text("\n".join(lines))


# ======================================================================================
# CLI
# ======================================================================================
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="make_control_report_h200.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Diff the staged Hopper control arm against the committed H200 "
                    "campaign, cell for cell, and publish the comparison.",
        epilog="""Typical use:

  python3 glm52/make_control_report_h200.py
  python3 glm52/make_control_report_h200.py --arm no-tma
  python3 glm52/make_control_report_h200.py --band p10p90

Exit codes: 0 published; 1 REFUSED by a validity gate (nothing written); 2 FATAL, an input
is missing or unreadable -- including the ordinary case where the operator has not run
run_control_h200.py on the H200 yet.

This script never writes outside --out, and never writes into results/h200/. The control arm
is a DIFF against the campaign, not an append to it; there is no merge step.""")
    p.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL,
                   help="the staging root the operator sends back (default: %(default)s). "
                        "Read-only.")
    p.add_argument("--arm", default=DEFAULT_ARM,
                   help="which staged arm is the treatment side (default: %(default)s; "
                        "known: " + ", ".join(sorted(ARM_AXES_OFF)) + "). The CSV columns "
                        "stay named classic_* whatever this is. Each known arm is checked "
                        "against the axes it is supposed to force off -- and 'hopper', "
                        "which forces nothing off, is checked in the INVERSE direction. An "
                        "unknown name gets no axis check and says so loudly.")
    p.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN,
                   help="the committed campaign, used as the baseline (default: "
                        "%(default)s). Opened read-only and never modified.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="where the CSVs and README are written (default: %(default)s). "
                        "The only directory this script writes to.")
    p.add_argument("--band", choices=("minmax", "p10p90"), default="minmax",
                   help="which noise-floor statistic drives the verdict column. minmax is "
                        "the default because at n=22 a percentile is interpolated between "
                        "order statistics and the conservative answer to 'could drift alone "
                        "have done this' is the observed extremes. Both are always "
                        "published.")
    p.add_argument("--force-unverified", action="store_true",
                   help="publish anyway despite an ARM_NOT_VERIFIED sentinel file, an "
                        "arms.<arm>.verified=false / non-null arms.<arm>.sentinel / FAILED "
                        "engagement_summary entry in the control summary, a device-uuid "
                        "mismatch or a lost device anchor. Adds a banner to every CSV note "
                        "and an ## UNVERIFIED section to the README. Does NOT override the "
                        "noise-floor guard or a broken-engagement refusal.")
    p.add_argument("--force-floor", action="store_true",
                   help="publish anyway despite a harness floor above "
                        f"{FLOOR_US_MAX} us or a timer match_frac KNOWN to be below "
                        f"{TICK_MATCH_MIN} on the control side. That combination is the "
                        "contended-card signature that once produced an impossible 40.55 us "
                        "preflight and changes VERDICTS, not just noise. A match_frac that "
                        "was never recorded is a flag, not a refusal, and does not need "
                        "this flag.")
    return p.parse_args(argv)


def nothing_to_compare(args, why: str, present: list[str]) -> int:
    print("=" * 92, flush=True)
    print("NOTHING TO COMPARE YET.", flush=True)
    print("", flush=True)
    print(f"  {why}", flush=True)
    print(f"  expected at: {args.control_dir / args.arm}", flush=True)
    if present:
        print(f"  arms present in {args.control_dir}: {', '.join(present)}", flush=True)
    print("", flush=True)
    for line in (
        "  The control arm has to be measured on the H200 first -- that machine is the",
        "  operator's and nobody else can reach it. On that box, from the repo root:",
        "",
        "      python3 run_control_h200.py",
        "",
        "  then send back results/h200/_control_arm/ (the whole tree, including",
        "  _engagement/ and any ARM_NOT_VERIFIED sentinel), log/run_control_h200/ and",
        "  glm52_h200/preflight_h200.json, unpack them in place, and re-run this script.",
        "",
        "  Exiting 2 (FATAL: input missing). Nothing was written.",
    ):
        print(line, flush=True)
    print("=" * 92, flush=True)
    return 2


# ======================================================================================
# main
# ======================================================================================
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    warnings: list[str] = []
    check_bar_sync(warnings)

    # ---- inputs, and the ordinary "not measured yet" exit -----------------------------
    if not args.campaign_dir.is_dir():
        return fatal(f"campaign directory {args.campaign_dir} does not exist. It is the "
                     f"BASELINE of this comparison and cannot be reconstructed.")
    camp_sum, err = load_json(args.campaign_dir / "summary.json")
    if camp_sum is None:
        return fatal(f"{args.campaign_dir / 'summary.json'}: {err}. Regenerate it with "
                     f"`python3 run_h200.py --summary-only`.")

    if not args.control_dir.is_dir():
        return nothing_to_compare(args, f"{args.control_dir} does not exist.", [])
    present = sorted(p.name for p in args.control_dir.iterdir()
                     if p.is_dir() and not p.name.startswith("_"))
    arm_dir = args.control_dir / args.arm
    if not arm_dir.is_dir():
        return nothing_to_compare(args, f"no staged arm directory named '{args.arm}'.",
                                  present)
    staged = [p for p in sorted(arm_dir.glob("*.json")) if not p.name.startswith("_")]
    if not staged:
        return nothing_to_compare(args, f"{arm_dir} exists but holds no result JSON.",
                                  present)

    ctrl_sum, err = load_json(args.control_dir / SUMMARY_NAME)
    if ctrl_sum is None:
        return fatal(f"{args.control_dir / SUMMARY_NAME}: {err}. The driver writes it after "
                     f"every family; without it there is no timer tick, no device anchor "
                     f"and no engagement summary for the control side, and its absence "
                     f"means the returned tarball is incomplete.")

    # ---- timers: each side uses its OWN, never a shared one ---------------------------
    ct = camp_sum.get("timer") or {}
    xt = ctrl_sum.get("timer") or {}
    camp_tick = fnum(ct.get("tick_us")) or R.DEFAULT_TICK_US
    ctrl_tick = fnum(xt.get("tick_us")) or R.DEFAULT_TICK_US
    camp_ut = int(ct.get("unresolved_ticks") or 3)
    ctrl_ut = int(xt.get("unresolved_ticks") or 3)
    if camp_tick != ctrl_tick:
        warnings.append(f"the two sessions measured different timer ticks "
                        f"({camp_tick} us vs {ctrl_tick} us); each side's cells use its own")

    # ---- cells, rows, offered axes ----------------------------------------------------
    camp_cells, camp_files, n1 = cells_for(args.campaign_dir, camp_tick, camp_ut)
    ctrl_cells, ctrl_files, n2 = cells_for(arm_dir, ctrl_tick, ctrl_ut)
    warnings += n1 + n2
    camp_rows, n3 = rows_by_cell(args.campaign_dir)
    ctrl_rows, n4 = rows_by_cell(arm_dir)
    warnings += n3 + n4
    offered = offered_axes_for(args.campaign_dir)

    # INFO-level parity check: the cells derived here must equal the ones the campaign's
    # own summary.json carries. Any disagreement is reported, never silently absorbed.
    published = {(c["family"], c["variant"], c["regime"]): c
                 for c in camp_sum.get("cells") or []}
    mism = 0
    for key, c in camp_cells.items():
        p = published.get(key)
        if p is None:
            mism += 1
            continue
        a, b = fnum(c.get("speedup")), fnum(p.get("speedup"))
        if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9):
            mism += 1
    if mism:
        warnings.append(f"{mism} of {len(camp_cells)} campaign cells re-derived here differ "
                        f"from results/h200/summary.json's published cells; the campaign "
                        f"files and its summary are out of sync (regenerate with "
                        f"`run_h200.py --summary-only`)")
    if len(camp_cells) != EXPECTED_CELLS:
        warnings.append(f"campaign yielded {len(camp_cells)} cells, expected "
                        f"{EXPECTED_CELLS} (15 groups x 11 regimes)")
    if len(ctrl_cells) != EXPECTED_CELLS:
        warnings.append(f"control arm '{args.arm}' yielded {len(ctrl_cells)} cells, "
                        f"expected {EXPECTED_CELLS}; the arm is incomplete")

    engagement = dict(((ctrl_sum.get("arms") or {}).get(args.arm) or {})
                      .get("engagement_summary") or {})

    joined, jnotes = join_cells(camp_cells, ctrl_cells, camp_rows, ctrl_rows, offered,
                                engagement, camp_files, ctrl_files, args.arm)
    warnings += jnotes

    nf = build_noise_floor(joined, args.band)
    apply_verdicts(joined, nf, args.band)
    # The band only exists after the first pass, so the f03/f10 samples must be recomputed
    # with their final verdicts attached (NOISE-FLOOR rather than PENDING).
    nf = build_noise_floor(joined, args.band)
    fams = family_verdicts(joined)
    eng_rows, eng_notes = engagement_rows(args.control_dir, args.arm)
    warnings += eng_notes

    camp_floors, ctrl_floors = floors_for(args.campaign_dir), floors_for(arm_dir)
    camp_uuids, ctrl_uuids = uuids_for(args.campaign_dir), uuids_for(arm_dir)
    prov = provenance_rows(camp_sum, ctrl_sum, camp_floors, ctrl_floors,
                           camp_uuids, ctrl_uuids)

    # ==================================================================================
    # cross-checks that can REFUSE. Run before anything is written; on refusal, write
    # NOTHING.
    # ==================================================================================
    forced: list[str] = []

    # (1) the driver's own verdict that this arm is unverified. FOUR independent records say
    # so and any ONE of them is enough: the sentinel FILE, `arms.<arm>.verified == False`,
    # a non-null `arms.<arm>.sentinel`, and any FAILED value in `engagement_summary`. The
    # gate used to key on the file alone -- but the driver sends the summary and the staging
    # tree back as separate items of a four-path list, so a tarball can easily carry the
    # summary that says "not verified" while the sentinel file itself never made it across.
    # Believing the missing file over the summary is exactly the wrong way round.
    arm_rec = ((ctrl_sum.get("arms") or {}).get(args.arm) or {})
    unverified: list[str] = []
    sentinel = arm_dir / SENTINEL_NAME
    if sentinel.exists():
        body = sentinel.read_text().strip()
        first = body.splitlines()[0] if body else "(empty)"
        unverified.append(f"the {SENTINEL_NAME} sentinel file is present in {arm_dir}: "
                          f"{first}")
    else:
        body = ""
    if arm_rec.get("verified") is False:
        unverified.append(f"{SUMMARY_NAME} records arms.{args.arm}.verified = false")
    if arm_rec.get("sentinel") not in (None, ""):
        unverified.append(f"{SUMMARY_NAME} records arms.{args.arm}.sentinel = "
                          f"{arm_rec.get('sentinel')!r}"
                          + ("" if sentinel.exists() else
                             f" while the {SENTINEL_NAME} file is ABSENT from the returned "
                             f"tree -- the tarball is missing the sentinel, not the arm's "
                             f"failure"))
    failed = sorted(k for k, v in (arm_rec.get("engagement_summary") or {}).items()
                    if str(v).upper() == "FAILED")
    if failed:
        unverified.append(f"{SUMMARY_NAME} records engagement_summary FAILED for "
                          f"{', '.join(failed)}")
    if unverified:
        msg = ("the driver could not verify that this arm engaged:\n  - "
               + "\n  - ".join(unverified)
               + (f"\n--- {SENTINEL_NAME} ---\n{body}\n--- end ---" if body else "")
               + "\nAn arm that did not engage is indistinguishable from a positive "
                 "result, which is this study's hypothesis.")
        if not args.force_unverified:
            return refuse(msg)
        forced += unverified

    # (2) same physical card, both sides, cross-checked against every family file
    cu = R._norm_uuid((camp_sum.get("_meta") or {}).get("gpu_uuid"))
    xu = R._norm_uuid((ctrl_sum.get("_meta") or {}).get("gpu_uuid"))
    split = sorted({u for u in list(camp_uuids.values()) + list(ctrl_uuids.values()) if u})
    # Cross-check the summary's claim against every family file's own env.uuid: reading only
    # the campaign-wide uuid is what would have hidden the first campaign's two-card split.
    fam_mismatch = sorted(fam for fam in set(camp_uuids) & set(ctrl_uuids)
                          if camp_uuids[fam] and ctrl_uuids[fam]
                          and camp_uuids[fam] != ctrl_uuids[fam])
    if (cu and xu and cu != xu) or fam_mismatch:
        msg = (f"the two sessions ran on different cards: campaign {cu[:8]}, control "
               f"{xu[:8]}"
               + (f"; per-family env.uuid disagrees for {', '.join(fam_mismatch)}"
                  if fam_mismatch else "")
               + ". Card-to-card variation would be confounded with the effect on top of "
                 "the cross-session drift this design already concedes.")
        if not args.force_unverified:
            return refuse(msg)
        forced.append("measured on a different physical card; card-to-card variation is now "
                      "confounded with the effect on top of the cross-session drift this "
                      "design already concedes.")
    elif len(split) > 1:
        warnings.append(f"the per-family env.uuid records name more than one card "
                        f"({', '.join(u[:8] for u in split)}); see provenance.csv")

    # (3) the driver's own device anchor
    if ((ctrl_sum.get("device_anchor") or {}).get("matched") is False):
        note = (ctrl_sum.get("device_anchor") or {}).get("note") or "(no note)"
        msg = f"the control run lost its device anchor: {note}"
        if not args.force_unverified:
            return refuse(msg)
        forced.append(f"device anchor lost: {note}")

    # (4) harness floor / timer trust -- the contended-card signature
    floor_problems = [r for r in prov if r["severity"] == "REFUSE"
                      and (r["item"].startswith("harness_floor_us_")
                           or r["item"] == "timer_match_frac")]
    if floor_problems and not args.force_floor:
        detail = "; ".join(f"{r['item']}: campaign={r['campaign']} control={r['control']}"
                           for r in floor_problems)
        return refuse(f"harness floor / timer trust bar failed on the control session "
                      f"({detail}). Bars: 0 < floor <= {FLOOR_US_MAX} us, match_frac >= "
                      f"{TICK_MATCH_MIN}. That is the contended-card signature that once "
                      f"produced an impossible 40.55 us preflight, and it changes VERDICTS, "
                      f"not just noise. --force-floor overrides.")
    if floor_problems:
        forced.append("harness floor / timer trust bar failed: "
                      + "; ".join(r["item"] for r in floor_problems))
    flagged = [r for r in prov if r["severity"] == "FLAG"]
    for r in flagged:
        if r["match"] == "unknown":
            # e.g. a summary that records no timer tick-match fraction in EITHER of its two
            # homes. Not knowing is not the same as failing, and it is not a refusal.
            warnings.append(f"{r['item']} could not be checked: campaign={r['campaign']} "
                            f"control={r['control']} -- the value is UNKNOWN, which is not "
                            f"evidence either way, so this is a flag and not a refusal")
        else:
            warnings.append(f"{r['item']} differs between the two sessions: campaign="
                            f"{r['campaign']} control={r['control']} -- this is itself a "
                            f"drift measurement, not a refusal")

    # (5) engagement: an arm that still selected an axis it was supposed to force off did
    # not engage, and its numbers are not a control at all. No override -- this is the
    # study-killing failure mode. The check used to be hardcoded to `args.arm == "classic"`,
    # so the four other arms the CLI advertises published with NO axis check whatsoever: a
    # byte-copy of the Hopper campaign staged as `no-tma` sailed through with exit 0.
    axes_off = ARM_AXES_OFF.get(args.arm)
    if axes_off is None:
        warnings.append(f"arm '{args.arm}' is not in this report's ARM_AXES_OFF table "
                        f"({', '.join(sorted(ARM_AXES_OFF))}), so NO engagement axis check "
                        f"could be run. Nothing below is verified to have engaged. Add the "
                        f"arm to ARM_AXES_OFF (mirror run_control_h200.ARMS[*].axes_off).")
    elif axes_off:
        broken = []
        for r in joined:
            still = sorted(axes_off & set(canonical_axes(r["classic_axes_selected"])))
            if still:
                broken.append((r, still))
        if broken:
            for r, still in broken[:20]:
                print(f"  ENGAGEMENT-BROKEN {r['family']}/{r['variant_raw']}/{r['regime']}: "
                      f"arm '{args.arm}' must force {', '.join(sorted(axes_off))} off but "
                      f"the winning config still selected {r['classic_axes_selected']}",
                      flush=True)
            return refuse(f"{len(broken)} cell(s) in the '{args.arm}' arm selected an axis "
                          f"this arm forces off ({', '.join(sorted(axes_off))}). The arm "
                          f"did not engage, so its numbers are not a control at all. There "
                          f"is no override for this: it is the study-killing failure mode. "
                          f"Re-check the driver's _engagement/ verdicts.")
    else:
        # The INVERSE check. `hopper` disables nothing, so it must SELECT at least one axis
        # somewhere on the GEMM-carrying families; a `hopper` arm with a clean token scan is
        # either a mis-staged classic tree or a build that never emitted Hopper code, and
        # either way it is not the same-session baseline it claims to be.
        seen: set[str] = set()
        n_fam_cells = 0
        for r in joined:
            if r["family"] in HOPPER_ARM_FAMILIES:
                n_fam_cells += 1
                seen |= set(canonical_axes(r["classic_axes_selected"]))
        if n_fam_cells and not seen:
            return refuse(f"the '{args.arm}' arm forces nothing off, yet not one of the "
                          f"{n_fam_cells} cell(s) on {', '.join(HOPPER_ARM_FAMILIES)} "
                          f"selected any Hopper axis. This arm is supposed to be the "
                          f"CONVERSE check -- a same-session arm with the levers LIVE. A "
                          f"clean token scan here means it is not measuring what it claims "
                          f"(a mis-staged classic tree, or a build emitting no Hopper "
                          f"code), and comparing anything against it is meaningless. There "
                          f"is no override for this.")
        if not n_fam_cells:
            warnings.append(f"arm '{args.arm}' forces nothing off, but none of "
                            f"{', '.join(HOPPER_ARM_FAMILIES)} is present in it, so the "
                            f"inverse engagement check could not run")
        else:
            warnings.append(f"arm '{args.arm}' engagement (inverse check): axes still live "
                            f"on {', '.join(HOPPER_ARM_FAMILIES)}: {', '.join(sorted(seen))}")

    # (6) has the baseline moved since the run? warn loudly, never refuse.
    fp = ((ctrl_sum.get("campaign") or {}).get("fingerprint") or {})
    moved = []
    for name, rec in fp.items():
        p = args.campaign_dir / name
        if not p.exists():
            moved.append(f"{name} (gone)")
        elif isinstance(rec, dict) and rec.get("sha256") and sha256_of(p) != rec["sha256"]:
            moved.append(f"{name} (content changed)")
    if moved:
        warnings.append("THE BASELINE MOVED between the control run and this report: "
                        + ", ".join(moved) + ". The comparison is against files that are "
                        "no longer the ones the operator measured against.")

    # (7) the noise-floor guard. Deliberately NOT overridable by --force-unverified: the
    # band is the only defence this design has against the cross-session confound, and a
    # report published without it is not a weaker claim, it is no claim at all.
    if nf["n_usable"] < MIN_NOISE_CELLS:
        unusable = [f"{s['family']}/{s['regime']}: {s['why_unusable']}"
                    for s in nf["samples"] if not s["usable"]]
        detail = "; ".join(unusable[:12]) or "(no f03/f10 cells at all)"
        return refuse(f"the noise floor rests on {nf['n_usable']} usable f03/f10 cells "
                      f"(need {MIN_NOISE_CELLS} of {2 * len(REGIMES)}; "
                      f"{len(unusable)} unusable). The control arm's only defence against the "
                      f"cross-session confound is that band; without it no cell verdict is "
                      f"defensible. Unusable: {detail}")

    # ==================================================================================
    # publish
    # ==================================================================================
    args.out.mkdir(parents=True, exist_ok=True)
    banner = ""
    if forced:
        banner = "PUBLISHED OVER A REFUSAL (" + "; ".join(forced) + ")"

    fam_floor = ctrl_floors.get("f03") or ctrl_floors.get("f10")
    prov_sentence = (
        f"campaign card {cu[:8] or '?'} floor "
        f"{(camp_floors.get('f03') or float('nan')):.1f} us / control card "
        f"{xu[:8] or '?'} floor {(fam_floor or float('nan')):.1f} us")

    write_cell_csvs(args.out, joined, args.arm, prov_sentence, banner)
    write_noise_floor(args.out, nf)
    write_family_verdicts(args.out, fams)
    write_engagement(args.out, eng_rows)
    write_provenance(args.out, prov)

    lo, hi = band_edges(nf["global"], args.band)
    mde, _half = resolving_power(lo, hi)

    # THE HEADLINE. Two things it must never do, both of which it used to do:
    #   1. call a (fusion, variant) group a "family" -- 15 groups span 7 families, and two
    #      groups can carry the SAME fusion label and differ only in variant, so the old
    #      sentence printed the same name twice and miscounted the noun;
    #   2. fold NO-DATA into "everything else is inside the band and shows nothing". A group
    #      that was never measured in this arm has no delta at all. Silently counting it as
    #      a null is the single most misleading thing this report could say, and with a
    #      partial arm it is the common case.
    def label(f: dict) -> str:
        return f"{f['fusion']} ({f['variant']})"

    # f03/f10 are the band itself and are never "judged"; NO-DATA groups have no delta.
    judged = [f for f in fams if f["verdict"] not in ("NOISE-FLOOR", "NO-DATA")]
    out_groups = [f for f in judged if f["verdict"] in ("OUTSIDE-BAND-HIGH",
                                                        "OUTSIDE-BAND-LOW", "MIXED")]
    nodata = [f for f in fams if f["verdict"] == "NO-DATA"]
    n_out = len(out_groups)
    nodata_clause = ""
    if nodata:
        nodata_clause = (
            f"{len(nodata)} of {len(fams)} (fusion, variant) group(s) have NO usable cell "
            f"in the '{args.arm}' arm and therefore NO delta at all -- they are NOT inside "
            f"the band and NOT a null result, they are UNMEASURED: "
            + "; ".join(label(f) for f in nodata)
            + " (the per-cell `verdict` column says why: MISSING-CONTROL, "
              "UNRESOLVED-ONE-SIDE, INCOMPARABLE-BASIS). ")
    if not judged:
        headline = (nodata_clause + "No (fusion, variant) group could be judged against the "
                                    "drift band at all. This report shows nothing.")
    elif n_out == 0:
        headline = (nodata_clause
                    + f"Of the {len(judged)} group(s) that could be judged (f03/f10 "
                      f"excluded -- they define the band), every delta sits inside the "
                      f"[{sig(lo)}, {sig(hi)}] drift band -- nothing is shown at a "
                      f"resolution of {mag(mde)}, which is not the same claim as 'the "
                      f"Hopper features did nothing'.")
    else:
        names = ", ".join(label(f) for f in out_groups)
        n_out_fams = len({f["family"] for f in out_groups})
        n_jud_fams = len({f["family"] for f in judged})
        headline = (nodata_clause
                    + f"{n_out} of {len(judged)} judged (fusion, variant) group(s), "
                      f"spanning {n_out_fams} of {n_jud_fams} families, have cells outside "
                      f"the [{sig(lo)}, {sig(hi)}] drift band ({names}); the other "
                      f"{len(judged) - n_out} judged group(s) sit inside it and show "
                      f"nothing at a resolution of {mag(mde)}.")
    write_readme(args.out, args.arm, nf, fams, joined, prov, eng_rows, args, camp_sum,
                 ctrl_sum, warnings, forced, offered, headline)

    print(f"wrote {args.out}/ : {len(R.order_regimes({r['regime'] for r in joined}))} "
          f"control_<regime>.csv, noise_floor.csv, noise_floor_by_class.csv, "
          f"family_verdicts.csv, engagement.csv, provenance.csv, README.md", flush=True)
    print(f"noise floor (f03+f10): n={nf['n_usable']}/{nf['n_total']} usable, "
          f"delta_rel in [{sig(lo)}, {sig(hi)}], median "
          f"{sig(nf['global']['median'])}", flush=True)
    print(headline, flush=True)
    for w in warnings:
        print(f"  ! {w}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
