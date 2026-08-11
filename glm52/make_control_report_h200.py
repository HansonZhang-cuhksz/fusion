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

CO-TENANCY, AND WHY IT IS PARSED OUT OF A LOG FILE.

A family measured while another process held the card is not a noisy measurement, it is not
a measurement: the campaign it is diffed against was taken on an idle card, and a neighbour
holding 122 GB of a 143.8 GB card removes the comparison rather than widening it. Such a
family is EXCLUDED by name, with the co-tenant's pid, process and size printed beside it, and
flagged for re-measurement -- never averaged in and never quietly dropped.

The driver that produced the 2026-08-11 arm only WARNED and continued (a policy inherited
from run_h200.py, right for a campaign and wrong for a diff), and it rewrites
`control_arm_summary.json` on every invocation, so the surviving summary carries
`gpu.tenant_events == []` and `fam_stages[f01|f04f05] == {wall_s: 0.0, attempts: []}` -- it
reads as though the two contaminated families were never launched. `log/run_control_h200/
driver.log` is append-only and still holds every `[hw before]`/`[hw after ]` snapshot and
every `!! a co-tenant appeared` line, so THAT is the source (`--driver-log`), and the
summary's own fields only top it up.

EXIT CODES.
    0   publishable: the CSVs and README under `--out` were written
    1   REFUSED: a validity gate failed (unverified arm -- sentinel file OR the summary's
        own verified/sentinel/engagement_summary records -- device-anchor loss, UUID
        mismatch, harness-floor mismatch, an arm that still selected an axis it was supposed
        to force off, an unreadable `--driver-log` (which would publish contaminated
        families as clean; `--driver-log none` is the explicit opt-out), or too few usable
        noise-floor cells). Nothing is written.

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
import re               # the FLOOR_US_MAX source-text sync check, and the driver-log parse
import statistics
import sys
from datetime import datetime, timedelta
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

#: The driver's own transcript, and for the 2026-08-11 run the ONLY surviving record of the
#: co-tenants. `run_control_h200.py` writes one summary per invocation and the LAST one wins;
#: the 19:32 run took the "re-verify only" path for the four families measured earlier, so it
#: rewrote `control_arm_summary.json` with `gpu.tenant_events == []`, `fam_stages[f01].wall_s
#: == 0.0` and `attempts == []`. The summary now reads as though f01/f04f05 were never
#: launched. driver.log is APPENDED to, never rewritten, so every invocation's `[hw before]`
#: / `[hw after ]` pair and every `!! a co-tenant appeared` line is still there. Contamination
#: is therefore derived from the log and only TOPPED UP from the summary -- never the reverse.
DEFAULT_DRIVER_LOG = REPO / "log" / "run_control_h200" / "driver.log"

SUMMARY_NAME = "control_arm_summary.json"
SENTINEL_NAME = "ARM_NOT_VERIFIED"
ENGAGE_DIRNAME = "_engagement"

#: The two families that define the drift band. They advertise no Hopper cfg key
#: (`kernel_cfg_keys == "module advertises none"`), all three axes are `offered: false` in the
#: campaign, and a full key-path scan of both committed files finds zero Hopper tokens in any
#: cfg, tune table or axis_counts. Their control arm is therefore configured identically to
#: their Hopper arm and their delta is pure session-to-session variation.
NOISE_FAMILIES = ("f03", "f10")

#: Families that are NOT part of the published band but whose delta is, on this run, arguably
#: a second drift measurement rather than a treatment contrast: their offered tuner grid did
#: not change between the arms (engagement check V9 reports "0 of 21" and "0 of 63" offered-
#: grid stages differing, and total n_tried is 16070 vs 16036 and 22408 vs 22406 -- a ratio of
#: 1.00), because both arms sample the grid down to the same fixed budget even though the
#: LEGAL grid collapsed 5x. An EXTENDED band including them is computed and published as a
#: DIAGNOSTIC in README section 2c -- never as the band that drives a verdict, because judging
#: a family against a band it is itself a constituent of is circular. It is there to answer
#: one question honestly: is a band built from two short elementwise kernels wide enough to
#: bound the session-to-session tuner variance of the GEMM-heavy families?
BAND_CANDIDATE_FAMILIES = ("f06", "f08f09")

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

#: nvidia-smi `used` MiB at a family's `[hw before]` / `[hw after ]` snapshot above which the
#: card is judged to have been holding SOMEBODY ELSE'S allocation. Both snapshots are taken by
#: the driver while no child of its own is running, so on an idle card they read 0-4 MiB; the
#: two contaminated families on 2026-08-11 read 124498 and 63259 MiB. 1 GiB is far above the
#: observed idle noise and far below any real neighbour, and it is a CLI flag because the bar
#: is a judgement about the host, not a property of this report.
TENANT_MIB = 1024.0

#: Slack allowed when matching a staged file's `_meta.recorded_at` into a stage window. The
#: child writes its JSON a second or two before the driver takes `[hw after ]`, and on this
#: run every recorded_at in fact falls strictly inside its window, so the slack only has to
#: cover a clock skew, not a real gap.
STAGE_SLACK_S = 180.0

#: How long after a family's last driver timestamp a co-tenant line still counts as "in flight
#: during that family". The driver runs its tenant check immediately after the child exits, so
#: a genuine attribution lands 1-2 s after `[hw after ]` (f01: hw after 14:47:29, event
#: 14:47:30; f04f05: 15:51:16 / 15:51:18). Deliberately TIGHT: the 15:51:59 event of the same
#: run names a second neighbour but arrives 43 s after f04f05's payload was written and during
#: an f11 attempt that was killed and left nothing, so it must attach to no published family
#: rather than being smeared onto the nearest one.
TENANT_EVENT_SLACK_S = 30.0

#: A cell whose FUSED time falls below its own session's measured `harness_floor_us` is
#: rejected as an instrument-validity failure, on BOTH sides, before any statistic is taken.
#: This is a pre-registerable rule computed per cell from the published JSON without reference
#: to the speedup: the harness floor is what the session itself measured an empty timed region
#: to cost, so a "measurement" below it is not a measurement of the kernel. Over the 2026-08-11
#: pair it rejects exactly one cell of 330 -- control f10/decode_bs16 at 16.42 us against that
#: session's 39.46 us floor -- and no campaign cell at all. The rejected cell is PUBLISHED,
#: with its diagnostics, in `excluded.csv`, and the drift band is reported BOTH WAYS so a
#: reader can price the exclusion instead of taking it on trust.
CELL_FLOOR_RULE = ("fused time below the session's own measured harness_floor_us")

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

def binom_sf(k: int, n: int, p: float) -> float:
    """P(X > k) for X ~ Binomial(n, p). Exact, stdlib only.

    Here to give the headline exceedance count a null expectation. Without one, "12 of 86
    cells fall outside the band" reads as a finding when a min/max band over 21 reference
    cells is exceeded 2/22 = 9.1 % of the time by construction -- 7.8 of those 12 are the
    band's own false-positive rate.
    """
    if n <= 0 or k >= n:
        return 0.0
    k = max(k, -1)
    total = 0.0
    for i in range(k + 1, n + 1):
        total += math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
    return min(1.0, max(0.0, total))


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
    "TENANT-CONTAMINATED": ("EXCLUDED and marked for RE-MEASUREMENT: this family was measured "
                            "while another process held the card, and the campaign baseline "
                            "it is diffed against was measured on an idle card. A neighbour "
                            "holding most of a 143.8 GB card does not add noise to a "
                            "memory-bound ratio, it removes the comparison. The delta is "
                            "printed for inspection and enters no band, no aggregate and no "
                            "verdict"),
    "INVALID-HARNESS-FLOOR": ("EXCLUDED and marked for RE-MEASUREMENT: the fused time falls "
                              "below the harness floor that this very session measured, so "
                              "the cell is not commensurable with the cell it is diffed "
                              "against. The delta is printed for inspection and enters no "
                              "band, no aggregate and no verdict; `excluded.csv` carries the "
                              "diagnostics and the band is reported both with and without it"),
    "PROVENANCE-SPLICED": ("EXCLUDED and marked for RE-MEASUREMENT: this cell's checkpoint "
                           "was saved outside its family's measuring window, i.e. it was "
                           "inherited from an earlier abandoned attempt and reused rather "
                           "than re-measured, so it was not taken in the session the rest of "
                           "the family was taken in"),
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
    "TENANT-CONTAMINATED": ("EXCLUDED from the published comparison and marked for "
                            "RE-MEASUREMENT: every cell was measured while another process "
                            "held the card, against a baseline measured on an idle card. "
                            "This is NOT an inside-the-band result and NOT a null -- it is "
                            "the absence of a usable measurement"),
    "EXCLUDED-CELLS": ("EXCLUDED from the published comparison and marked for "
                       "RE-MEASUREMENT: every cell failed a stated per-cell validity rule, "
                       "so this group has no usable measurement -- which is not a null"),
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


def parse_ts(s: object) -> datetime | None:
    """`"2026-08-11 14:47:27"` -> datetime. Anything else -> None, never an exception."""
    try:
        return datetime.strptime(str(s).strip(), "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def fmt_ts(dt: datetime | None) -> str:
    return "" if dt is None else dt.strftime("%Y-%m-%d %H:%M:%S")


# ======================================================================================
# CO-TENANCY, derived from the driver's OWN LOG -- because the summary no longer knows
# ======================================================================================
# The driver that produced the 2026-08-11 arm only WARNED about a co-tenant and carried on
# (a policy inherited from run_h200.py: correct for a campaign that is measuring absolute
# numbers on whatever card it gets, wrong for a DIFF against an idle-card baseline). It has
# since been changed to stop, but the operator ran the committed version, so the arm exists
# and has to be read as it is.
#
# Worse, the co-tenant warnings were recorded in the summary of the invocation that SAW them,
# and `control_arm_summary.json` is rewritten from scratch by every later invocation. The
# 19:32 run re-verified f01/f04f05 without re-measuring them, so the surviving summary carries
# `gpu.tenant_events == []`, two unrelated warnings, and `fam_stages[f01|f04f05].wall_s == 0.0`
# with `attempts == []`. Keying the exclusion on the summary -- which is what this report used
# to do -- silently publishes both contaminated families as clean.
#
# driver.log is append-only across invocations. Every family stage there is bracketed by an
# `[hw before]` / `[hw after ]` nvidia-smi snapshot taken while no child of the driver's own
# is running, and every co-tenant detection is a `!! a co-tenant appeared on GPU N` line
# naming pid, process and size. That is the evidence, so that is what is parsed.
_RE_LOG_START = re.compile(r"driver started\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})")
_RE_LOG_STAGE = re.compile(r"^\s*(\d{2}:\d{2}:\d{2})\s+([A-Za-z0-9][\w.-]*)/([A-Za-z0-9_]+)"
                           r"\s+--\s")
_RE_LOG_HW = re.compile(r"^\s*(\d{2}:\d{2}:\d{2})\s+\[hw\s+(before|after)\s*\]"
                        r".*?\bused\s+(\d+)\s*/\s*(\d+)\s*MiB")
_RE_LOG_EXIT = re.compile(r"^\s*(\d{2}:\d{2}:\d{2})\s+exit=(-?\d+)\s+status=(\S+)"
                          r"(?:\s+wall=([\d.]+)\s*min)?")
_RE_LOG_TENANT = re.compile(r"^\s*(\d{2}:\d{2}:\d{2})\s+!!\s+a co-tenant appeared on GPU\s+"
                            r"(\d+)\s+(.*)$")
_RE_LOG_TENANT_FAM = re.compile(r"\bafter\s+([\w.-]+)/([A-Za-z0-9_]+)\b")
_RE_LOG_PROC = re.compile(r"pid\s+(\d+)\s+(.+?)\s+\(([\d.]+)\s*GB\)")


def parse_driver_log(path: Path) -> tuple[list[dict], list[dict], list[str]]:
    """(stages, tenant_events, notes) from the control driver's append-only transcript.

    A "stage" is ONE launch of ONE family by ONE invocation -- not a family. The 2026-08-11
    log holds six invocations and several families appear more than once, because a run that
    was killed mid-family is re-attempted by the next one. Attempts that produced nothing are
    the reason a family cannot simply be looked up by name: the abandoned f04f05 attempt of
    14:47:30 started on an already-contaminated card, and the abandoned f11 attempt of
    15:51:19 did too, yet the f11 that was actually published came from the clean 19:32 run.
    Which attempt produced the staged payload is decided later, by timestamp, in
    `attribute_contamination`.

    Timestamps in the body of the log are HH:MM:SS only; the date comes from the
    `driver started YYYY-MM-DD HH:MM:SS` header of each invocation and is rolled forward if
    the clock ever goes backwards by more than 12 h (a run crossing midnight).
    """
    notes: list[str] = []
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        return [], [], [f"could not read the control driver log {path}: {exc}. Co-tenancy "
                        f"CANNOT be derived from control_arm_summary.json for this run (its "
                        f"gpu.tenant_events was overwritten by a later invocation), so with "
                        f"the log unreadable NO family can be cleared of contamination."]

    stages: list[dict] = []
    events: list[dict] = []
    cur: dict | None = None
    run_idx = 0
    day: datetime | None = None
    prev: datetime | None = None

    first = _RE_LOG_START.search(text)
    if first:
        day = datetime.strptime(first.group(1), "%Y-%m-%d")
    else:
        notes.append(f"{path.name} carries no 'driver started YYYY-MM-DD' header, so its "
                     f"HH:MM:SS timestamps cannot be dated and no stage can be matched to a "
                     f"staged file's recorded_at")

    def stamp(hms: str) -> datetime | None:
        nonlocal day, prev
        if day is None:
            return None
        t = datetime.strptime(hms, "%H:%M:%S")
        dt = day.replace(hour=t.hour, minute=t.minute, second=t.second)
        if prev is not None and dt < prev - timedelta(hours=12):
            dt += timedelta(days=1)
            day = dt.replace(hour=0, minute=0, second=0)
        prev = dt
        return dt

    def close(reason: str) -> None:
        nonlocal cur
        if cur is not None:
            cur["closed_by"] = cur.get("closed_by") or reason
            stages.append(cur)
            cur = None

    for lineno, line in enumerate(text.splitlines(), 1):
        mstart = _RE_LOG_START.search(line)
        if mstart:
            close("a new driver invocation started before this stage reported an exit")
            run_idx += 1
            day = datetime.strptime(mstart.group(1), "%Y-%m-%d")
            prev = None
            stamp(mstart.group(2))
            continue

        mstage = _RE_LOG_STAGE.match(line)
        if mstage:
            close("the next family stage started before this one reported an exit")
            ts = stamp(mstage.group(1))
            cur = {"run": max(run_idx, 1), "arm": mstage.group(2), "family": mstage.group(3),
                   "start": ts, "end": None, "line": lineno,
                   "mib_before": None, "mib_after": None, "mib_total": None,
                   "exit": None, "status": None, "wall_min": None, "closed_by": None}
            continue

        mhw = _RE_LOG_HW.match(line)
        if mhw and cur is not None:
            ts = stamp(mhw.group(1))
            cur["mib_" + mhw.group(2)] = float(mhw.group(3))
            cur["mib_total"] = float(mhw.group(4))
            cur["end"] = ts or cur["end"]
            continue

        mexit = _RE_LOG_EXIT.match(line)
        if mexit and cur is not None:
            ts = stamp(mexit.group(1))
            cur["end"] = ts or cur["end"]
            cur["exit"] = int(mexit.group(2))
            cur["status"] = mexit.group(3)
            cur["wall_min"] = fnum(mexit.group(4))
            close(f"exit={cur['exit']} status={cur['status']}")
            continue

        mten = _RE_LOG_TENANT.match(line)
        if mten:
            ts = stamp(mten.group(1))
            body = mten.group(3)
            fam = _RE_LOG_TENANT_FAM.search(body)
            procs = [{"pid": p[0], "name": p[1], "gb": fnum(p[2])}
                     for p in _RE_LOG_PROC.findall(body)]
            events.append({"ts": ts, "gpu": mten.group(2), "line": lineno,
                           "named_arm": fam.group(1) if fam else "",
                           "named_family": fam.group(2) if fam else "",
                           "procs": procs, "text": body.strip()})
            continue

        # An unmatched line is not interesting, but a timestamp going backwards inside one
        # invocation is: it means the rollover heuristic above has been given something it
        # cannot date, and every window comparison downstream would be wrong.
        mts = re.match(r"^\s*(\d{2}:\d{2}:\d{2})\s", line)
        if mts:
            stamp(mts.group(1))

    close("end of log")
    if not stages:
        notes.append(f"{path.name} parsed but yielded no '<arm>/<family> -- ' stage header; "
                     f"the log format has changed and co-tenancy cannot be derived from it")
    return stages, events, notes


def describe_procs(procs: list[dict]) -> str:
    if not procs:
        return "process not named in the log line"
    return ", ".join(f"pid {p['pid']} {p['name']}"
                     + (f" ({p['gb']:.1f} GB)" if p.get("gb") is not None else "")
                     for p in procs)


def attribute_contamination(stages: list[dict], events: list[dict], arm: str,
                            recorded_at: dict[str, datetime | None], tenant_mib: float,
                            ) -> tuple[dict[str, dict], list[str]]:
    """Which family each co-tenant event was in flight during, and which stage was published.

    Attribution is to a STAGE, never to a family name, and the published stage is the one
    whose window contains the staged file's own `_meta.recorded_at`. That distinction is the
    whole point: f11 has an attempt that began at 15:51:19 on a card holding 63259 MiB and was
    killed 34 s later leaving no payload, and the f11 that IS published came from the 19:32
    run on a card at 4 -> 0 MiB. Attributing by family name would condemn a clean family;
    attributing by stage does not.

    A published stage is contaminated when ANY of:
      * its `[hw before]` snapshot is above `tenant_mib` -- somebody was already there;
      * its `[hw after ]` snapshot is above `tenant_mib` -- somebody arrived during it, and
        the driver cannot say when in the window, so no cell of it can be exonerated;
      * a `!! a co-tenant appeared` line names it;
      * such a line lands inside its window (plus `STAGE_SLACK_S`, since the driver runs the
        tenant check just after the child exits).
    """
    notes: list[str] = []
    out: dict[str, dict] = {}
    slack = timedelta(seconds=STAGE_SLACK_S)
    ev_slack = timedelta(seconds=TENANT_EVENT_SLACK_S)

    for fam, rec_at in sorted(recorded_at.items()):
        cands = [s for s in stages if s["arm"] == arm and s["family"] == fam]
        if not cands:
            notes.append(f"{fam}: the control driver log records no '{arm}/{fam}' stage at "
                         f"all, so this family's card occupancy is UNKNOWN -- it is neither "
                         f"cleared nor condemned by the log")
            out[fam] = {"family": fam, "stage": None, "attempts": 0, "contaminated": False,
                        "unknown": True, "reasons": [], "events": [], "other_attempts": []}
            continue
        chosen = None
        if rec_at is not None:
            inside = [s for s in cands if s["start"] and s["end"]
                      and s["start"] - slack <= rec_at <= s["end"] + slack]
            if inside:
                chosen = max(inside, key=lambda s: s["start"])
        if chosen is None:
            ok = [s for s in cands if s.get("status") == "ok"]
            chosen = (max(ok, key=lambda s: s["start"] or datetime.min) if ok
                      else max(cands, key=lambda s: s["start"] or datetime.min))
            notes.append(
                f"{fam}: could not match the staged file's recorded_at "
                f"({fmt_ts(rec_at) or 'unrecorded'}) to any '{arm}/{fam}' stage window in the "
                f"driver log; falling back to the last stage that exited ok "
                f"({fmt_ts(chosen.get('start'))}). Check the log by hand before trusting this "
                f"family's provenance.")

        reasons: list[str] = []
        mb, ma = chosen.get("mib_before"), chosen.get("mib_after")
        tot = chosen.get("mib_total") or 0.0
        if mb is not None and mb > tenant_mib:
            reasons.append(f"the card already held {mb:.0f}/{tot:.0f} MiB at [hw before] "
                           f"{fmt_ts(chosen['start'])} -- the family started on an occupied "
                           f"card")
        if ma is not None and ma > tenant_mib:
            reasons.append(f"the card held {ma:.0f}/{tot:.0f} MiB at [hw after ] "
                           f"{fmt_ts(chosen['end'])} against "
                           f"{'' if mb is None else f'{mb:.0f} MiB'} at [hw before] -- a "
                           f"neighbour arrived at an unrecorded point inside the measuring "
                           f"window, so no cell of this family can be exonerated")
        mine: list[dict] = []
        for ev in events:
            named = (ev["named_family"] == fam and ev["named_arm"] in ("", arm))
            in_window = (ev["ts"] is not None and chosen["start"] is not None
                         and chosen["end"] is not None
                         and chosen["start"] <= ev["ts"] <= chosen["end"] + ev_slack)
            if named and in_window:
                mine.append(ev)
                reasons.append(f"the driver named this family: at {fmt_ts(ev['ts'])} it "
                               f"reported {describe_procs(ev['procs'])} on GPU {ev['gpu']}")
            elif named and ev["ts"] is not None and chosen["end"] is not None \
                    and ev["ts"] > chosen["end"] + ev_slack:
                # A later invocation re-attempted this family and picked up a tenant then.
                # It says nothing about the attempt that was actually published.
                notes.append(f"{fam}: a co-tenant line at {fmt_ts(ev['ts'])} names this "
                             f"family but falls outside the window of the attempt that "
                             f"produced the staged payload "
                             f"({fmt_ts(chosen['start'])}-{fmt_ts(chosen['end'])}); it is "
                             f"recorded and NOT used to exclude this family")
            elif in_window:
                mine.append(ev)
                reasons.append(f"a co-tenant line landed inside this family's window at "
                               f"{fmt_ts(ev['ts'])}: {describe_procs(ev['procs'])}")

        others = []
        for s in cands:
            if s is chosen:
                continue
            dirty = ((s.get("mib_before") or 0) > tenant_mib
                     or (s.get("mib_after") or 0) > tenant_mib)
            others.append(f"{fmt_ts(s['start'])} (run {s['run']}, "
                          f"{'exit=' + str(s['exit']) if s.get('exit') is not None else 'no exit line: ' + str(s.get('closed_by'))}"
                          f", card {('%.0f' % s['mib_before']) if s.get('mib_before') is not None else '?'}"
                          f" -> {('%.0f' % s['mib_after']) if s.get('mib_after') is not None else '?'} MiB"
                          f"{', ON AN OCCUPIED CARD' if dirty else ''}) -- produced no "
                          f"published payload")
        out[fam] = {"family": fam, "stage": chosen, "attempts": len(cands),
                    "contaminated": bool(reasons), "unknown": False,
                    "reasons": reasons, "events": mine, "other_attempts": others}

    claimed = {id(ev) for rec in out.values() for ev in rec["events"]}
    for ev in events:
        if id(ev) in claimed:
            continue
        notes.append(f"co-tenant event at {fmt_ts(ev['ts'])} ({describe_procs(ev['procs'])}) "
                     f"is NOT attributable to any published family stage: "
                     f"\"{ev['text']}\". It is recorded here and excludes nothing -- "
                     f"but it is evidence about the host, and a re-measurement should not be "
                     f"scheduled on the assumption that the card is quiet.")
    return out, notes


def ckpt_saved_at(results_dir: Path, fam_key: str) -> dict[str, datetime | None]:
    """{regime: when the bench wrote that regime's checkpoint}, from `_ckpt/<stem>/*.json`.

    This is the ONLY place a spliced cell is visible. f01's published `decode_bs1` row was
    saved at 13:40:31, by the 13:37 invocation that was killed mid-family; the 14:39:56
    invocation that produced the other ten rows found that checkpoint on disk and reused it
    instead of re-measuring. Neither the result file's `_meta.recorded_at` nor the summary
    shows this -- only the checkpoint's own `saved_at` does.
    """
    out: dict[str, datetime | None] = {}
    path = None
    for fam in R.FAMILIES:
        if fam.key == fam_key:
            path = R.find_result(fam, results_dir)
            break
    if path is None:
        return out
    ck = results_dir / "_ckpt" / path.stem
    if not ck.is_dir():
        return out
    for p in sorted(ck.glob("*.json")):
        payload, _ = load_json(p)
        out[p.stem] = parse_ts((payload or {}).get("saved_at"))
    return out


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


def invalid_cells(cells: dict[tuple[str, str, str], dict], floors: dict[str, float | None],
                  side: str) -> dict[tuple[str, str, str], str]:
    """{cell key -> why it is not a valid measurement}, under one stated, uniform rule.

    THE RULE, stated before the numbers so it cannot be a post-hoc rescue of an inconvenient
    point: a cell is invalid if its FUSED time falls below the `harness_floor_us` that its own
    session measured. The harness floor is what that session clocked an EMPTY timed region at,
    so a kernel timing underneath it is not a measurement of the kernel -- it is the harness
    failing to enclose the work. The rule is computed per cell from the published JSON, makes
    no reference to the speedup, to the drift band, or to which arm the cell is in, and is
    applied identically to the campaign and to the control.

    On the 2026-08-11 pair it rejects one cell of 330: control `f10/decode_bs16`, fused
    16.42 us against that session's own 39.46 us floor (513 timer ticks). Zero campaign cells
    are rejected. That single cell is worth a rule of its own because f10 is one of the two
    families that DEFINE the drift band, and at +73.1 % it would widen the band from about
    +/-6 % to +/-73 % -- against which no cell in any family is resolvable and the study
    reports nothing at all. The band is therefore published BOTH WAYS (see `excluded.csv`
    and README section 2) rather than the exclusion being taken on trust.
    """
    out: dict[tuple[str, str, str], str] = {}
    for key, c in cells.items():
        floor = fnum(floors.get(key[0]))
        fused = fnum(c.get("fused_ms"))
        if floor is None or floor <= 0 or fused is None:
            continue
        fused_us = fused * 1000.0
        if fused_us < floor:
            out[key] = (f"{side} fused time {fused_us:.2f} us is BELOW the harness floor that "
                        f"session measured for itself ({floor:.2f} us): the timed region did "
                        f"not enclose the work, so the number is not commensurable with the "
                        f"cell it is diffed against")
    return out


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


def staged_recorded_at(results_dir: Path) -> dict[str, str]:
    """{family: `_meta.recorded_at`} -- when the CHILD wrote each staged result file.

    This is what pins a staged payload to one of the driver log's several attempts at the
    same family, and it is why an abandoned attempt on a dirty card cannot condemn a family
    that was later re-measured on a clean one.
    """
    out: dict[str, str] = {}
    for fam in R.FAMILIES:
        if fam.key == "layer":
            continue
        path = R.find_result(fam, results_dir)
        if path is None:
            continue
        payload, _ = load_json(path)
        out[fam.key] = str(((payload or {}).get("_meta") or {}).get("recorded_at") or "")
    return out


def staged_child_hw(results_dir: Path) -> dict[str, dict]:
    """{family: `_meta.hwinfo.gpu`} -- the CHILD's own nvidia-smi snapshot as it finished.

    A second, independent witness to the driver's `[hw before]`/`[hw after ]` pair: it is
    recorded by a different process, into a different file, at a different moment. On this
    run it corroborates the log exactly -- f01 records 126989 MiB used / 16168 free with
    util 4 % and a throttle reason set, f04f05 66288 / 76869, while the five clean families
    record only their own 983-25761 MiB footprints.
    """
    out: dict[str, dict] = {}
    for fam in R.FAMILIES:
        if fam.key == "layer":
            continue
        path = R.find_result(fam, results_dir)
        if path is None:
            continue
        payload, _ = load_json(path)
        gpu = (((payload or {}).get("_meta") or {}).get("hwinfo") or {}).get("gpu")
        if isinstance(gpu, dict):
            out[fam.key] = gpu
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


#: Verdict tokens that mean "this cell contributes no measurement". They never enter the
#: drift band, a family aggregate, or a headline count. Kept as one list so the band, the
#: family verdicts and the headline cannot drift apart about what "usable" means.
UNUSABLE_VERDICTS = ("UNRESOLVED-ONE-SIDE", "MISSING-CONTROL", "MISSING-CAMPAIGN",
                     "UNMAPPED", "INCOMPARABLE-BASIS", "EXCLUDED-HALF",
                     "TENANT-CONTAMINATED", "INVALID-HARNESS-FLOOR", "PROVENANCE-SPLICED")

#: The subset of those that are EXCLUSIONS -- a measurement was taken and was then rejected
#: for a stated reason -- as opposed to absences. The distinction matters in the headline: an
#: exclusion has to be named and marked for re-measurement, never folded into a null.
EXCLUSION_VERDICTS = ("TENANT-CONTAMINATED", "INVALID-HARNESS-FLOOR", "PROVENANCE-SPLICED")


def build_noise_floor(joined: list[dict], mode: str, readmit: tuple[str, ...] = ()) -> dict:
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

    `readmit` names verdict tokens to count as usable ANYWAY. It exists for exactly one
    purpose: building the SHADOW band that retains the cells the validity rule threw out, so
    the report can print the band both ways and let a reader price the exclusion instead of
    trusting it. It is never used for the band that drives verdicts.
    """
    samples: list[dict] = []
    for row in joined:
        if row["family"] not in NOISE_FAMILIES:
            continue
        blocked = [v for v in UNUSABLE_VERDICTS if v not in readmit]
        usable = row["verdict"] not in blocked and row["delta_rel"] is not None
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
               arm: str, tainted: dict[str, str] | None = None,
               bad_cells: dict[tuple[str, str, str], str] | None = None,
               spliced: dict[tuple[str, str], str] | None = None,
               ) -> tuple[list[dict], list[str]]:
    """One row per (fusion, variant, regime), campaign side joined to control side.

    `tainted` (family -> why), `bad_cells` (cell key -> why) and `spliced` ((family, regime)
    -> why) are the three exclusions. All three keep the measured numbers in the row and only
    change the VERDICT: nothing is deleted, because a reader has to be able to see what the
    exclusion cost. `UNUSABLE_VERDICTS` is what stops them reaching a band or an aggregate.
    """
    tainted = tainted or {}
    bad_cells = bad_cells or {}
    spliced = spliced or {}
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
            "exclusion": "", "exclusion_reason": "",
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
                base = "PENDING"   # filled in once the band exists
            else:
                # `speedup` is None exactly when collect_cells called the cell UNRESOLVED.
                # The published delta uses `speedup`; the raw one survives in its own column
                # and is never averaged into the band or a family verdict.
                base = "UNRESOLVED-ONE-SIDE"
            # The three exclusions outrank the statistic and each other in this order. Each
            # leaves the measured numbers in place -- only the verdict changes -- so the CSV
            # still shows exactly what was thrown away and what it would have contributed.
            if family in tainted:
                row["verdict"] = "TENANT-CONTAMINATED"
                row["exclusion"] = "TENANT-CONTAMINATED"
                # A cell can fail more than one way. The strongest token wins the verdict,
                # but every reason is kept: f01/decode_bs1 is both contaminated and spliced,
                # and a re-measurement plan needs to know about both.
                row["exclusion_reason"] = "; ALSO ".join(
                    [tainted[family]]
                    + ([spliced[(family, regime)]] if (family, regime) in spliced else [])
                    + ([bad_cells[(family, variant, regime)]]
                       if (family, variant, regime) in bad_cells else []))
            elif (family, regime) in spliced:
                row["verdict"] = "PROVENANCE-SPLICED"
                row["exclusion"], row["exclusion_reason"] = "PROVENANCE-SPLICED", \
                    spliced[(family, regime)]
            elif (family, variant, regime) in bad_cells:
                row["verdict"] = "INVALID-HARNESS-FLOOR"
                row["exclusion"], row["exclusion_reason"] = "INVALID-HARNESS-FLOOR", \
                    bad_cells[(family, variant, regime)]
            else:
                row["verdict"] = base
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
        n_excluded = sum(1 for r in rows if r["verdict"] in EXCLUSION_VERDICTS)
        # An EXCLUSION is not an absence. A group every one of whose cells was measured and
        # then rejected for a stated reason must say so and carry a RE-MEASURE flag; folding
        # it into NO-DATA ("no usable cell") loses the reason, and folding it into the
        # inside-the-band population would be the single most misleading thing this report
        # could do.
        if not usable and n_excluded:
            tok = {r["verdict"] for r in rows if r["verdict"] in EXCLUSION_VERDICTS}
            verdict = ("TENANT-CONTAMINATED" if tok == {"TENANT-CONTAMINATED"}
                       else "EXCLUDED-CELLS")
        elif family in NOISE_FAMILIES:
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
        if n_excluded:
            counts += (f" [{n_excluded} of {len(rows)} cell(s) EXCLUDED by a stated rule and "
                       f"marked for RE-MEASUREMENT: "
                       + ", ".join(sorted({r["verdict"] for r in rows
                                           if r["verdict"] in EXCLUSION_VERDICTS})) + "]")
        out.append({
            "fusion": rows[0]["fusion"], "variant": rows[0]["variant"], "family": family,
            "offered_axes": rows[0]["offered_axes"], "engagement": rows[0]["engagement"],
            "exclusion": "|".join(sorted({r["exclusion"] for r in rows if r["exclusion"]})),
            "n_cells": len(rows), "n_usable": len(usable), "n_excluded": n_excluded,
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


def driverlog_provenance_rows(log_path: Path, arm: str, contam: dict[str, dict],
                              events: list[dict], child_hw: dict[str, dict],
                              tenant_mib: float) -> list[dict]:
    """The per-family card-occupancy table, straight out of the driver's transcript.

    This is the section of `provenance.csv` that `control_arm_summary.json` cannot produce:
    its `gpu.tenant_events` is `[]` and its `fam_stages[*].wall_s` is `0.0` for every family
    the last invocation did not re-measure. Two independent witnesses are printed side by
    side -- the DRIVER's nvidia-smi snapshots either side of the stage, and the CHILD's own
    end-of-run snapshot recorded in the staged file's `_meta.hwinfo.gpu` -- so the exclusion
    does not rest on a single source.
    """
    rows: list[dict] = []
    rows.append({"item": "co-tenancy source", "campaign": "summary: gpu.tenant_events",
                 "control": f"{log_path} (the summary's tenant_events was overwritten by a "
                            f"later invocation of the driver and is empty)",
                 "match": "n/a", "severity": "info"})
    for fam in sorted(contam):
        rec = contam[fam]
        st = rec.get("stage") or {}
        win = f"{fmt_ts(st.get('start'))} - {fmt_ts(st.get('end'))}" if st else "no stage"
        mb = "?" if st.get("mib_before") is None else f"{st['mib_before']:.0f}"
        ma = "?" if st.get("mib_after") is None else f"{st['mib_after']:.0f}"
        wall = "" if st.get("wall_min") is None else f", wall {st['wall_min']:.1f} min"
        rows.append({
            "item": f"card_mib_{fam} (driver, before -> after)",
            "campaign": "0 -> 0 MiB (idle card; results/h200/summary.json _meta.gpu_was_idle)",
            "control": f"{mb} -> {ma} MiB over {win}{wall}"
                       + (f" [{rec['attempts']} attempt(s) in the log; this is the one whose "
                          f"window contains the staged file's recorded_at]"
                          if rec.get("attempts", 0) > 1 else ""),
            "match": "no" if rec.get("contaminated") else
                     ("unknown" if rec.get("unknown") else "yes"),
            # Severity tokens of their own, deliberately NOT "REFUSE"/"FLAG": those two are
            # read by the publish gates in main() and mean "stop" / "warn about a session
            # difference". Contamination is neither -- it excludes cells and lets the rest of
            # the report publish -- so it gets its own vocabulary and its own warnings.
            "severity": "EXCLUDES-CELLS" if rec.get("contaminated") else
                        ("UNKNOWN" if rec.get("unknown") else "info"),
        })
        hw = child_hw.get(fam) or {}
        if hw:
            rows.append({
                "item": f"card_mib_{fam} (child's own snapshot at record)",
                "campaign": "",
                "control": f"used {hw.get('memory.used', '?')} / free "
                           f"{hw.get('memory.free', '?')}, util "
                           f"{hw.get('utilization.gpu', '?')}, throttle "
                           f"{hw.get('clocks_throttle_reasons.active', '?')}",
                "match": "n/a",
                "severity": "info",
            })
        for reason in rec.get("reasons", []):
            rows.append({"item": f"tenant_evidence_{fam}", "campaign": "", "control": reason,
                         "match": "no", "severity": "EXCLUDES-CELLS"})
        for other in rec.get("other_attempts", []):
            rows.append({"item": f"other_attempt_{fam}", "campaign": "", "control": other,
                         "match": "n/a", "severity": "info"})
    for i, ev in enumerate(events, 1):
        rows.append({"item": f"tenant_event_{i}", "campaign": "none recorded",
                     "control": f"{fmt_ts(ev['ts'])} GPU {ev['gpu']}: "
                                f"{describe_procs(ev['procs'])}",
                     "match": "no", "severity": "TENANT"})
    rows.append({"item": "tenant_mib_bar", "campaign": "", "control":
                 f"a driver snapshot above {tenant_mib:.0f} MiB is read as somebody else's "
                 f"allocation (--tenant-mib)", "match": "n/a", "severity": "info"})
    return rows


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

    # NOT the co-tenancy verdict -- just what each summary happens to record. The control
    # driver rewrites its summary on every invocation, so a 0 here is entirely compatible
    # with a contaminated arm; the driver-log rows below are the evidence.
    rows.append({"item": "tenant_events (as recorded in each summary)",
                 "campaign": str(len((camp_sum.get("gpu") or {}).get("tenant_events") or [])),
                 "control": f"{len((ctrl_sum.get('gpu') or {}).get('tenant_events') or [])}"
                            f" -- NOT evidence of an idle card: the control driver rewrites "
                            f"this file on every invocation and the last one only re-verified "
                            f"the already-staged families. See the driver-log rows below.",
                 "match": "n/a", "severity": "info"})
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
    # The exclusion machinery, appended for the same reason: keep the documented column
    # order an exact prefix. `exclusion` repeats the verdict token for the three EXCLUDED
    # cases so a reader can filter on one column, and `exclusion_reason` carries the
    # evidence -- the pid, size and timestamp of the co-tenant, or the floor the cell fell
    # under. A cell with an exclusion still shows all of its measured numbers.
    "exclusion", "exclusion_reason",
]


def cell_note(row: dict, provenance_sentence: str, banner: str, arm: str) -> str:
    bits = []
    if banner:
        bits.append(banner)
    if row.get("exclusion"):
        bits.append(f"EXCLUDED ({row['exclusion']}) -- RE-MEASURE: {row['exclusion_reason']}")
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
                    "exclusion": row.get("exclusion", ""),
                    "exclusion_reason": row.get("exclusion_reason", ""),
                })
        written.append(path)
    return written


def write_noise_floor(out: Path, nf: dict, shadow: dict | None = None) -> None:
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
        # THE BAND, THE OTHER WAY. Published on the same sheet as the band that drives the
        # verdicts, not in a footnote, because the difference between the two IS the price of
        # the exclusion and a reader is entitled to see it without recomputing anything.
        if shadow is not None:
            emit("global-if-excluded-cells-retained", shadow["global"],
                 "no -- SHADOW: what the band would be if the cells excluded by the stated "
                 "validity rule were kept. Published so the exclusion can be priced.")
            emit("decode-if-excluded-cells-retained", shadow["decode"], "no -- SHADOW")
            emit("prefill-if-excluded-cells-retained", shadow["prefill"], "no -- SHADOW")


def write_family_verdicts(out: Path, fams: list[dict]) -> None:
    cols = ["fusion", "variant", "family", "offered_axes", "engagement", "exclusion",
            "n_cells", "n_usable", "n_excluded", "n_outside_high", "n_outside_low",
            "n_inside", "median_delta_rel",
            "min_delta_rel", "max_delta_rel", "band_basis", "verdict", "verdict_sentence"]
    with (out / "family_verdicts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for f in fams:
            row = dict(f)
            for k in ("median_delta_rel", "min_delta_rel", "max_delta_rel"):
                row[k] = r4(row[k])
            w.writerow(row)


def _diag(row: dict | None, floor_us: float | None) -> dict:
    """The per-cell diagnostics that let a reader overrule an exclusion on the evidence.

    Every field is read straight out of the bench row; none of them is derived from the
    speedup, which is the point -- the validity rule and its corroboration have to be
    checkable without looking at the answer.
    """
    row = row or {}
    fused = fnum(row.get("fused_ms"))
    p = row.get("fused_p10_p90") or []
    drift = row.get("drift_frac") or []
    su, ceil = fnum(row.get("speedup")), fnum(row.get("ceiling_with_launch"))
    spread = None
    if len(p) == 2 and fnum(p[0]):
        spread = fnum(p[1]) / fnum(p[0])
    return {
        "fused_us": None if fused is None else fused * 1000.0,
        "floor_us": floor_us,
        "fused_ticks": fnum(((row.get("tick") or {}).get("fused_ticks"))),
        "p90_over_p10": spread,
        "drift_frac_fused": fnum(drift[0]) if len(drift) == 2 else None,
        "order_gap_frac": fnum(row.get("order_gap_frac")),
        "speedup_over_ceiling": (su / ceil) if (su and ceil) else None,
    }


def write_excluded(out: Path, joined: list[dict], camp_rows: dict, ctrl_rows: dict,
                   camp_floors: dict, ctrl_floors: dict, contam: dict) -> list[dict]:
    """`excluded.csv` -- every cell this report refused to use, with the evidence.

    NOTHING IS DROPPED SILENTLY. A reader who disagrees with an exclusion can find the cell
    here with its measured numbers, the rule it failed, and the diagnostics that rule was
    checked against, and can price the disagreement against the both-ways band in the README.
    """
    rows: list[dict] = []
    for r in joined:
        if r["verdict"] not in EXCLUSION_VERDICTS:
            continue
        key = (r["family"], r["variant_raw"], r["regime"])
        dc = _diag(camp_rows.get(key), camp_floors.get(r["family"]))
        dx = _diag(ctrl_rows.get(key), ctrl_floors.get(r["family"]))
        ev = contam.get(r["family"]) or {}
        stage = ev.get("stage") or {}
        rows.append({
            "exclusion": r["verdict"],
            "re_measure": "YES",
            "family": r["family"], "fusion": r["fusion"], "variant": r["variant"],
            "regime": r["regime"],
            "campaign_speedup": r4(r["hopper_speedup"]),
            "control_speedup": r4(r["classic_speedup"]),
            "delta_rel_withheld": r4(r["delta_rel"]),
            "campaign_fused_us": r4(dc["fused_us"], 2),
            "control_fused_us": r4(dx["fused_us"], 2),
            "campaign_floor_us": r4(dc["floor_us"], 2),
            "control_floor_us": r4(dx["floor_us"], 2),
            "control_fused_ticks": r4(dx["fused_ticks"], 0),
            "campaign_p90_over_p10": r4(dc["p90_over_p10"]),
            "control_p90_over_p10": r4(dx["p90_over_p10"]),
            "campaign_drift_frac_fused": r4(dc["drift_frac_fused"]),
            "control_drift_frac_fused": r4(dx["drift_frac_fused"]),
            "campaign_order_gap_frac": r4(dc["order_gap_frac"]),
            "control_order_gap_frac": r4(dx["order_gap_frac"]),
            "campaign_speedup_over_ceiling": r4(dc["speedup_over_ceiling"]),
            "control_speedup_over_ceiling": r4(dx["speedup_over_ceiling"]),
            "measuring_window": (f"{fmt_ts(stage.get('start'))} - {fmt_ts(stage.get('end'))}"
                                 if stage else ""),
            "card_mib_before": "" if stage.get("mib_before") is None
                               else f"{stage['mib_before']:.0f}",
            "card_mib_after": "" if stage.get("mib_after") is None
                              else f"{stage['mib_after']:.0f}",
            "evidence": r["exclusion_reason"],
        })
    cols = list(rows[0]) if rows else [
        "exclusion", "re_measure", "family", "fusion", "variant", "regime",
        "campaign_speedup", "control_speedup", "delta_rel_withheld", "evidence"]
    with (out / "excluded.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


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
                 headline: str, shadow: dict, excluded: list[dict],
                 contam: dict[str, dict]) -> None:
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
        A(f"> **UNDERPOWERED -- THIS IS THE RESULT.** {mag(mde)} is wider than "
          f"{mag(RESOLUTION_TARGET)}, and {mag(RESOLUTION_TARGET)} is the LARGEST of the "
          f"six lever-offering variants' H200-vs-C500 effects, not the smallest -- so the "
          f"deficit below understates the problem for most of them. A null at this band "
          f"width is **not** evidence of no effect; it is evidence that this session pair "
          f"could not resolve one. Nor do the exceedances rescue it: they arrive at very "
          f"close to the rate a min/max band produces by construction (see the Verdict "
          f"above), and under a leave-one-family-out band, or against the two arms' own "
          f"within-cell p10-p90 dispersion, they largely disappear. **The defensible "
          f"headline for this run is that the design could not answer the question**, not "
          f"that it answered it either way. Re-run as a same-session paired A/B.")
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
      "to a FUSION GAIN: each side's `paired_speedup`, the median of the per-round ratios "
      "from that session's own interleaved A/B loop, and "
      "`delta_rel = classic_speedup / hopper_speedup - 1`. It is a ratio of "
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
        # "0 / 0" for a group whose every cell was excluded reads like a clean null. It is
        # the opposite of one, so it says EXCLUDED instead.
        cellcol = (f"EXCLUDED ({f['n_excluded']} of {f['n_cells']} cells)"
                   if not f["n_usable"] and f["n_excluded"]
                   else f"{outside} / {f['n_usable']}")
        rows.append([f["fusion"], f["variant"], f["family"], f["offered_axes"],
                     f["engagement"] or "-", cellcol,
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
      f"is **{mag(mde)}**. The effect this study was built to see is AT MOST "
      f"{mag(RESOLUTION_TARGET)}, and that is the LARGEST of the six lever-offering "
      f"variants, not the smallest -- most are far below it, so the deficit understates. "
      f"The four TMA-using families gained 1.00-1.06x on "
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

    # ---- 2b. the band both ways -------------------------------------------------------
    slo, shi = band_edges(shadow["global"], mode)
    smde, _ = resolving_power(slo, shi)
    readmitted = [r for r in excluded if r["exclusion"] in ("INVALID-HARNESS-FLOOR",
                                                            "PROVENANCE-SPLICED")
                  and r["family"] in NOISE_FAMILIES]
    A("### 2b. The band BOTH WAYS -- what the exclusions cost")
    A("")
    if not readmitted:
        A("No f03/f10 cell was excluded by the per-cell validity rule, so the band that "
          "drives the verdicts and the band with every measured cell retained are the same "
          "band. Nothing here turns on an exclusion.")
    else:
        A("The drift band is built from two families and a handful of cells, so a single "
          "cell can change every verdict in this report. It is therefore published **both "
          "ways**, in the body and not in a footnote, and the reader is told what the "
          "difference buys:")
        A("")
        n_out_primary = sum(1 for r in joined if r["verdict"] in ("OUTSIDE-HIGH",
                                                                  "OUTSIDE-LOW"))
        n_out_shadow = 0
        if slo is not None:
            for r in joined:
                if r["family"] in NOISE_FAMILIES or r["delta_rel"] is None:
                    continue
                if r["verdict"] in ("INSIDE", "OUTSIDE-HIGH", "OUTSIDE-LOW") \
                        and not (slo <= r["delta_rel"] <= shi):
                    n_out_shadow += 1
        A(md_table(
            ["band", "n f03/f10 cells", "min", "max", "min detectable effect",
             "judged cells outside it"],
            [["PUBLISHED -- excluded cells removed", glo["n"], sig(lo), sig(hi), mag(mde),
              n_out_primary],
             ["SHADOW -- excluded cells retained", shadow["global"]["n"], sig(slo), sig(shi),
              mag(smde), n_out_shadow]]))
        A("")
        for r in readmitted:
            A(f"* The cell in question is **`{r['family']}/{r['regime']}`** "
              f"({r['campaign_speedup']} -> {r['control_speedup']}, delta "
              f"{r['delta_rel_withheld']}), excluded as `{r['exclusion']}`. "
              f"{r['evidence']}")
        A("")
        A("**The decision is fully consequential, which is exactly why it is shown rather "
          "than buried.** Against the shadow band the report resolves "
          + ("nothing at all" if n_out_shadow == 0 else f"only {n_out_shadow} cell(s)")
          + "; against the published band it resolves "
          + (f"{n_out_primary} cell(s)" if n_out_primary else "nothing")
          + ". A reader who rejects the exclusion should read the shadow row and treat every "
            "verdict in this report as unresolved. A reader who accepts it should note that "
            "the rule was stated as a property of the instrument -- the measured harness "
            "floor -- and applied to every cell of both arms, not chosen after seeing which "
            "cell it would remove: over the "
          + f"{len(joined)} joined cells of both arms it rejects "
          + f"{sum(1 for r in excluded if r['exclusion'] == 'INVALID-HARNESS-FLOOR')}, "
            f"and no campaign cell at all. (Co-tenancy is a separate exclusion with a "
            f"separate cause; the two are counted separately everywhere in this report and "
            f"in `excluded.csv`.)")
        A("")
        A("Full detail, including the diagnostics the rule was cross-checked against "
          "(fused ticks, p90/p10 spread, drift_frac, order_gap_frac, speedup over the "
          "launch-adjusted traffic ceiling): `excluded.csv`.")
    A("")

    # ---- 2c. is a two-family band wide enough? ----------------------------------------
    A("### 2c. Is a two-family band wide enough? (diagnostic, drives no verdict)")
    A("")
    A("The band above rests on 21 cells from two short memory-bound elementwise kernels, and "
      "the report's own §0 already concedes that whether their variability bounds a "
      "GEMM-heavy family's is an assumption. Two measurements bear on it directly, and both "
      "are printed here rather than left for a reader to discover.")
    A("")
    v9 = [r for r in eng if r["family"] in NOISE_FAMILIES
          and r["check_id"].upper().startswith("V9")
          and "stage" in r["got"].lower() and "differ" in r["got"].lower()]
    if v9:
        A("**(a) The autotuner picks a different winner between the two sessions even where "
          "the two arms are identical by construction.** The engagement records report, for "
          "the very families that define the band:")
        A("")
        A(md_table(["family", "check", "what it compared", "result"],
                   [[r["family"], r["check_id"], r["want"] or r["json_path"], r["got"]]
                    for r in v9]))
        A("")
        A("Read the rows together: the OFFERED grid is identical between the two arms for "
          "these families (`V9`, `V9b`), and yet the WINNER the autotuner selected out of "
          "that identical grid differs in a large fraction of stages (`V9d`). A winner "
          "change is the mechanism that moves a delta, so the families defining the noise "
          "floor are subject to the same mechanism as the families being judged -- but at "
          "whatever amplitude two elementwise kernels happen to show, which is not "
          "necessarily the amplitude a GEMM family shows.")
        A("")
    ext_rows = []
    ext_vals = []
    for row in joined:
        if row["family"] not in tuple(NOISE_FAMILIES) + BAND_CANDIDATE_FAMILIES:
            continue
        if row["verdict"] in UNUSABLE_VERDICTS or row["delta_rel"] is None:
            continue
        ext_vals.append(row["delta_rel"])
    if ext_vals:
        ext = band_stats(ext_vals)
        elo, ehi = band_edges(ext, mode)
        emde, _ = resolving_power(elo, ehi)
        n_out_ext = sum(1 for r in joined
                        if r["family"] not in tuple(NOISE_FAMILIES) + BAND_CANDIDATE_FAMILIES
                        and r["delta_rel"] is not None
                        and r["verdict"] in ("INSIDE", "OUTSIDE-HIGH", "OUTSIDE-LOW")
                        and not (elo <= r["delta_rel"] <= ehi))
        n_judge_ext = sum(1 for r in joined
                          if r["family"] not in tuple(NOISE_FAMILIES) + BAND_CANDIDATE_FAMILIES
                          and r["verdict"] in ("INSIDE", "OUTSIDE-HIGH", "OUTSIDE-LOW"))
        A(f"**(b) What the band becomes if `{'`, `'.join(BAND_CANDIDATE_FAMILIES)}` are "
          f"treated as further drift constituents.** On this run their offered tuner grid did "
          f"not change between the arms (engagement check V9), so their delta is arguably a "
          f"second drift measurement rather than a treatment contrast:")
        A("")
        ext_rows.append(["published band (f03+f10)", nf["global"]["n"], sig(lo), sig(hi),
                         mag(mde), "yes -- drives every verdict"])
        ext_rows.append([f"extended ({'+'.join(tuple(NOISE_FAMILIES) + BAND_CANDIDATE_FAMILIES)})",
                         ext["n"], sig(elo), sig(ehi), mag(emde),
                         "NO -- diagnostic only; judging a family against a band it belongs "
                         "to is circular"])
        A(md_table(["band", "n", "min", "max", "min detectable effect", "used for verdicts"],
                   ext_rows))
        A("")
        A(f"Against the extended band, {n_out_ext} of the {n_judge_ext} remaining judged "
          f"cell(s) would fall outside. Read that as a statement about the INSTRUMENT, not "
          f"about the arms: it says how much of what §1 reports as resolved survives a wider "
          f"and arguably more honest estimate of cross-session variation. Settling the "
          f"question needs replicate runs of the no-axis families inside a single session, "
          f"so the band measures within-design variance rather than one draw of it.")
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

    # ==================================================================================
    # 3b. THE EXCLUSIONS. Named, evidenced, and flagged for re-measurement.
    # ==================================================================================
    A("## 3b. What was excluded from the comparison, and why")
    A("")
    if not excluded:
        A("Nothing was excluded: no family was measured on an occupied card, no cell failed "
          "the instrument-validity rule, and no cell was spliced in from another session.")
        A("")
    else:
        A("Excluded cells keep every measured number in the per-regime CSVs and in "
          "`excluded.csv`. What they lose is standing: they enter no drift band, no family "
          "aggregate and no headline count. **An exclusion is not a null result** -- it is "
          "the absence of a usable measurement, and each one below carries a RE-MEASURE "
          "flag.")
        A("")
        # Family-wide exclusions are summarised one row per family: printing the same
        # co-tenant sentence 44 times would bury the single-cell exclusions among them.
        # Every individual cell is still in excluded.csv, which is where a reader who wants
        # the per-cell numbers is sent.
        rows = []
        fam_wide: dict[str, list[dict]] = {}
        for r in excluded:
            if r["exclusion"] == "TENANT-CONTAMINATED":
                fam_wide.setdefault(r["family"], []).append(r)
            else:
                rows.append([r["exclusion"], f"{r['family']}/{r['regime']}", r["variant"],
                             r["campaign_speedup"], r["control_speedup"],
                             r["delta_rel_withheld"], "YES", r["evidence"]])
        for fam, rs in sorted(fam_wide.items()):
            rows.insert(0, ["TENANT-CONTAMINATED", f"{fam}: ALL {len(rs)} cell(s)",
                            ", ".join(sorted({x["variant"] for x in rs})), "-", "-",
                            f"{len(rs)} deltas withheld", "YES", rs[0]["evidence"]])
        A(md_table(["exclusion", "cell(s)", "variant(s)", "campaign speedup",
                    "control speedup", "delta withheld", "re-measure", "evidence"], rows))
        A("")
        A("Per-cell numbers for every row above -- including all "
          f"{sum(len(v) for v in fam_wide.values())} cells of the family-wide exclusions -- "
          "are in `excluded.csv`.")
        A("")

    A("### 3b.1 Co-tenancy, derived from the driver's log and not from the summary")
    A("")
    if not contam:
        A("> **PROVENANCE UNVERIFIED.** This report was generated with `--driver-log none`, "
          "so card occupancy was NOT checked for any family. `control_arm_summary.json` "
          "cannot substitute: the driver rewrites it on every invocation and this run's copy "
          "records `gpu.tenant_events == []` regardless of what happened. **No family below "
          "has been cleared of contamination** -- the absence of an exclusion here is the "
          "absence of evidence, not evidence of an idle card.")
        A("")
    A(f"Co-tenancy is parsed out of **`{args.driver_log}`**, the driver's append-only "
      f"transcript, and NOT out of `{SUMMARY_NAME}`. The driver rewrites its summary on "
      f"every invocation; the last invocation of this run only re-verified the families "
      f"that were already staged, so the surviving summary records "
      f"`gpu.tenant_events == []` and `wall_s == 0.0, attempts == []` for exactly the "
      f"families whose contamination matters. Reading it would publish them as clean. The "
      f"log still carries every `[hw before]` / `[hw after ]` nvidia-smi snapshot and every "
      f"`!! a co-tenant appeared` line.")
    A("")
    A("Attribution is to a **stage**, never to a family name: a family can appear several "
      "times in the log because a killed invocation is re-attempted, and the stage that "
      "counts is the one whose window contains the staged file's own `_meta.recorded_at`. "
      "That is what lets an attempt begun on a dirty card and abandoned sit in the log "
      "without condemning a family that was later re-measured on a clean one.")
    A("")
    if contam:
        rows = []
        for fam in sorted(contam):
            rec = contam[fam]
            st = rec.get("stage") or {}
            rows.append([
                fam,
                f"{fmt_ts(st.get('start'))} - {fmt_ts(st.get('end'))}" if st else "-",
                "" if st.get("wall_min") is None else f"{st['wall_min']:.1f} min",
                "?" if st.get("mib_before") is None else f"{st['mib_before']:.0f}",
                "?" if st.get("mib_after") is None else f"{st['mib_after']:.0f}",
                rec.get("attempts", 0),
                "CONTAMINATED -- EXCLUDED" if rec.get("contaminated")
                else ("UNKNOWN" if rec.get("unknown") else "clean"),
                "; ".join(rec.get("reasons", [])) or "no co-tenant line, both snapshots "
                                                     "below the bar",
            ])
        A(md_table(["family", "measuring window (the stage that produced the staged file)",
                    "wall", "card MiB before", "card MiB after", "attempts in log",
                    "occupancy", "evidence"], rows))
        A("")
        A(f"The bar is `--tenant-mib {args.tenant_mib:.0f}`. Both snapshots are taken by the "
          f"driver while none of its own children is running, so on an idle card they read "
          f"0-4 MiB. A second, independent witness -- the child process's own nvidia-smi "
          f"snapshot in the staged file's `_meta.hwinfo.gpu` -- is printed per family in "
          f"`provenance.csv`.")
        A("")
    A("A contaminated family is excluded rather than down-weighted because the baseline it "
      "is diffed against was measured on an idle card. A neighbour holding most of a "
      "143.8 GB card does not add variance to a memory-bound ratio; it removes the "
      "comparison. And because the driver's snapshots cannot say WHEN inside the window the "
      "neighbour arrived, no individual cell of such a family can be exonerated.")
    A("")

    A("### 3b.2 The per-cell instrument-validity rule")
    A("")
    A("Stated before the numbers, computed per cell from the published JSON, applied "
      "identically to both arms, and making no reference to the speedup or to the drift "
      "band:")
    A("")
    A(f"> A cell is INVALID if its fused time falls below the `harness_floor_us` that its "
      f"own session measured.")
    A("")
    A("The harness floor is what that session clocked an *empty* timed region at. A kernel "
      "timing underneath it is not a fast kernel; it is the harness failing to enclose the "
      "work. Because it is a property of the instrument rather than of the answer, the rule "
      "can be checked on every cell without knowing what it will remove -- and it is "
      "reported in §2b exactly how much the report's conclusions depend on what it did "
      "remove.")
    A("")
    A("### 3b.3 Spliced provenance")
    A("")
    A("A cell whose `_ckpt/<family>/<regime>.json` `saved_at` falls outside its family's "
      "measuring window was inherited from an earlier, abandoned attempt and reused rather "
      "than re-measured -- so it was not taken in the session this comparison is diffing. "
      "This is visible nowhere else: not in the result file's `_meta.recorded_at`, which "
      "records only when the merged file was written, and not in the summary.")
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
    A("* Each side's `speedup` is that side's own `paired_speedup` -- the median of the "
      "per-round ratios from its interleaved A/B loop, which cancels monotone drift within "
      "a session. It is NOT `unfused_ms / fused_ms`: those two columns are ratio-of-medians "
      "over separately-timed arms and differ from the paired statistic in 329 of 330 cells "
      "(worst gap 21 %). Use them to see WHICH SIDE MOVED, never to re-derive `speedup`. "
      "`ratio = classic_speedup / hopper_speedup` is therefore a ratio of ratios: it rises "
      "when the arm's unfused chain got slower just as readily as when its fused kernel got "
      "faster. Read `*_fused_ms` and `*_unfused_ms` before saying which side moved.")
    A("* `delta_rel = ratio - 1`; `delta_abs = classic_speedup - hopper_speedup` (a "
      "difference of gains, not of milliseconds).")
    A("* `band_lo` / `band_hi` / `band_basis` -- the drift band this particular cell was "
      "judged against.")
    A("* `delta_rel_raw` is populated only for `UNRESOLVED-ONE-SIDE` cells, from "
      "`speedup_raw`; it is excluded from the band and from every verdict.")
    A("* `exclusion` / `exclusion_reason` -- populated when this report refused to use the "
      "cell. The measured numbers are still printed; the exclusion only removes the cell's "
      "standing. Every excluded cell is also listed, with its diagnostics, in "
      "`excluded.csv`.")
    A("* **An empty numeric cell means NOT MEASURED. It is never coerced to zero.** An "
      "excluded cell, by contrast, has numbers AND a verdict saying they are not being "
      "used -- the two states are never conflated.")
    A("")
    A("Verdict tokens, one sentence each:")
    A("")
    for tok in ("OUTSIDE-HIGH", "OUTSIDE-LOW", "INSIDE", "NOISE-FLOOR",
                "TENANT-CONTAMINATED", "INVALID-HARNESS-FLOOR", "PROVENANCE-SPLICED",
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

    A("### 5.1 Two comparisons this report deliberately does NOT make")
    A("")
    A("* **Control wall time against `results/h200/summary.json` `families[*].wall_s`.** That "
      "field was frozen on 2026-08-07 and covers SEVEN regimes; `decode_bs2/4/8/16` were "
      "appended later by a separate bs-extra run and `wall_s` was never regenerated. The "
      "control arm ran ELEVEN. Diffing the two manufactures a slowdown out of a scope "
      "difference -- which is exactly what produced the driver's own \"took 24 min vs the "
      "campaign's 15 min\" alarm on `f06`. Normalise per regime, or add the bs-extra stage's "
      "wall, before comparing anything. No wall-time verdict is published here.")
    A("* **Anything about a per-feature lever.** See §0: this arm forces four capabilities "
      "off at once, in a different session from the baseline.")
    A("")

    A("## 6. What would change the verdict")
    A("")
    A("* **Re-measuring the excluded families in one idle session.** With the contaminated "
      "families out, the only clean family whose offered tuner grid measurably changed is "
      "`f11`; the two families with a real grid collapse are precisely the two that were "
      "contaminated. A one-family contrast cannot carry the study's question, so the "
      "re-measurement is not cleanup -- it is the experiment. Use a driver that STOPS on "
      "co-tenancy rather than warning, and re-measure a clean family alongside them as an "
      "in-session anchor.")
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
    p.add_argument("--driver-log", type=Path, default=DEFAULT_DRIVER_LOG,
                   help="the control driver's append-only transcript (default: %(default)s). "
                        "CO-TENANCY IS DERIVED FROM THIS FILE, not from "
                        "control_arm_summary.json: the driver rewrites its summary on every "
                        "invocation, so a run whose later invocations only re-verified an "
                        "already-staged family carries gpu.tenant_events == [] and "
                        "fam_stages[*].wall_s == 0.0 for exactly the families whose "
                        "contamination matters. Pass 'none' to skip the log -- which means "
                        "no family can be cleared OR condemned, and the report says so.")
    p.add_argument("--tenant-mib", type=float, default=TENANT_MIB,
                   help="nvidia-smi used-MiB at a family's [hw before]/[hw after ] snapshot "
                        "above which the card is judged to have been holding another "
                        "process's allocation (default: %(default)s). Both snapshots are "
                        "taken while no child of the driver's own is running.")
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
    # Needed before the join, not after it: the per-cell validity rule compares each cell's
    # fused time against its own session's measured harness floor.
    camp_floors, ctrl_floors = floors_for(args.campaign_dir), floors_for(arm_dir)

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

    # ==================================================================================
    # CO-TENANCY IS DISQUALIFYING, per family -- and it is derived from the DRIVER LOG.
    #
    # A family measured while a neighbour held the card is not noisy, it is not comparable:
    # the baseline it is diffed against was measured on an idle card, and a neighbour holding
    # 122 GB of a 143.8 GB card does not add variance to a memory-bound ratio, it removes the
    # comparison. On 2026-08-11 a VLLM worker took 121.6 GB during f01 and a 61.8 GB python
    # process arrived during f04f05.
    #
    # The summary-field path below (`tenant_contaminated`, `fam_stages[*].tenant_contaminated`)
    # is what the FIXED driver writes and is still honoured, but it cannot be the source for
    # this run: the driver rewrites control_arm_summary.json on every invocation, the 19:32
    # invocation only re-verified f01/f04f05, and the surviving summary therefore records
    # gpu.tenant_events == [] and fam_stages[f01|f04f05] == {wall_s: 0.0, attempts: []}. It
    # reads as though the two contaminated families were never launched. driver.log is
    # appended to and still holds every snapshot and every warning, so it leads and the
    # summary only tops up.
    # ==================================================================================
    arm_rec = (ctrl_sum.get("arms") or {}).get(args.arm) or {}
    ctrl_recorded_at = {fam: parse_ts(v) for fam, v in staged_recorded_at(arm_dir).items()}
    contam: dict[str, dict] = {}
    log_events: list[dict] = []
    use_log = str(args.driver_log).lower() not in ("none", "")
    if use_log:
        stages, log_events, lnotes = parse_driver_log(args.driver_log)
        warnings += lnotes
        # A log that cannot be read is not "no co-tenancy". For THIS run the summary carries
        # no tenant record at all, so an unreadable log means every family would be published
        # as clean on no evidence whatsoever -- the exact silent failure this whole path
        # exists to prevent. Refuse, and name the explicit opt-out.
        if not stages:
            return refuse(
                f"the control driver log {args.driver_log} could not be read or yielded no "
                f"family stage, so card occupancy is UNKNOWN for every family. "
                f"{SUMMARY_NAME} cannot stand in for it: the driver rewrites that file on "
                f"every invocation and this run's copy records gpu.tenant_events == [] even "
                f"though the log shows co-tenants. Publishing now would report contaminated "
                f"families as clean. Point --driver-log at the returned log, or pass "
                f"--driver-log none to publish with provenance explicitly unverified.")
        contam, cnotes = attribute_contamination(stages, log_events, args.arm,
                                                 ctrl_recorded_at, args.tenant_mib)
        warnings += cnotes
    else:
        warnings.append(
            "--driver-log none: co-tenancy was NOT derived from the driver's transcript. "
            "control_arm_summary.json's gpu.tenant_events is rewritten by every invocation "
            "and is empty for this run, so NO family below has been cleared of contamination "
            "-- absence of an exclusion here is absence of evidence, not evidence of an idle "
            "card.")

    tainted: dict[str, str] = {}
    for fam, rec in sorted(contam.items()):
        if rec.get("contaminated"):
            tainted[fam] = ("measured on a card that was NOT idle, against a campaign "
                            "baseline that was: " + "; ".join(rec["reasons"])
                            + f". Source: {args.driver_log}.")
    # The fixed driver's own fields, kept so this report does not go stale the moment the
    # driver starts recording contamination itself.
    from_summary = {t.split("/")[-1] for t in (ctrl_sum.get("tenant_contaminated") or [])}
    from_summary |= {t.split("/")[-1] for t in (arm_rec.get("tenant_contaminated") or [])}
    from_summary |= {k for k, st in (arm_rec.get("fam_stages") or {}).items()
                     if isinstance(st, dict) and st.get("tenant_contaminated")}
    for fam in sorted(from_summary):
        tainted.setdefault(fam, f"{SUMMARY_NAME} records this family as tenant-contaminated")
    if tainted:
        warnings.append(
            "CO-TENANT CONTAMINATION -> EXCLUDED and marked for RE-MEASUREMENT: "
            + "; ".join(f"{k} ({v})" for k, v in sorted(tainted.items()))
            + " The campaign baseline was measured on an idle card, so these families are "
              "not comparable at any confidence. They are named in the headline, in "
              "family_verdicts.csv, in excluded.csv and in every per-regime CSV, and they "
              "enter no band and no aggregate.")

    # Per-cell provenance: a checkpoint saved outside its family's measuring window was
    # inherited from an earlier, abandoned attempt and reused instead of re-measured. Only
    # the checkpoints show this -- the result file's own recorded_at does not.
    spliced: dict[tuple[str, str], str] = {}
    for fam, rec in sorted(contam.items()):
        st = rec.get("stage") or {}
        if not st.get("start") or not st.get("end"):
            continue
        for regime, saved in sorted(ckpt_saved_at(arm_dir, fam).items()):
            if saved is None:
                continue
            if not (st["start"] - timedelta(seconds=STAGE_SLACK_S) <= saved
                    <= st["end"] + timedelta(seconds=STAGE_SLACK_S)):
                spliced[(fam, regime)] = (
                    f"this cell's checkpoint was saved at {fmt_ts(saved)}, OUTSIDE the "
                    f"{fmt_ts(st['start'])}-{fmt_ts(st['end'])} window in which the rest of "
                    f"{fam} was measured: it was inherited from an earlier abandoned attempt "
                    f"and reused rather than re-measured, so it was not taken in the session "
                    f"this comparison is diffing")
    if spliced:
        warnings.append(
            "SPLICED PROVENANCE -> EXCLUDED and marked for RE-MEASUREMENT: "
            + "; ".join(f"{f}/{r}" for f, r in sorted(spliced))
            + ". These cells were carried over from an abandoned attempt in a different "
              "session and are visible only in results/.../_ckpt/<family>/<regime>.json's "
              "saved_at field.")

    # Per-cell instrument validity, applied identically to BOTH sides before any statistic.
    bad_cells = dict(invalid_cells(camp_cells, camp_floors, "campaign"))
    bad_cells.update(invalid_cells(ctrl_cells, ctrl_floors, f"control ({args.arm})"))
    if bad_cells:
        warnings.append(
            f"INSTRUMENT VALIDITY -> EXCLUDED and marked for RE-MEASUREMENT: "
            + "; ".join(f"{'/'.join(k)}: {v}" for k, v in sorted(bad_cells.items()))
            + f". The rule ({CELL_FLOOR_RULE}) is applied to all "
              f"{len(camp_cells) + len(ctrl_cells)} cells of both arms and rejects "
              f"{len(bad_cells)}. The drift band is published BOTH WAYS in "
              f"noise_floor_by_class.csv and README section 2 so the exclusion can be priced.")

    joined, jnotes = join_cells(camp_cells, ctrl_cells, camp_rows, ctrl_rows, offered,
                                engagement, camp_files, ctrl_files, args.arm,
                                tainted, bad_cells, spliced)
    warnings += jnotes

    nf = build_noise_floor(joined, args.band)
    apply_verdicts(joined, nf, args.band)
    # The band only exists after the first pass, so the f03/f10 samples must be recomputed
    # with their final verdicts attached (NOISE-FLOOR rather than PENDING).
    nf = build_noise_floor(joined, args.band)
    # THE SAME BAND WITH THE EXCLUDED CELLS PUT BACK. Never used for a verdict; published
    # beside the real one so the reader sees how much the whole report turns on one point.
    shadow = build_noise_floor(joined, args.band,
                               readmit=("INVALID-HARNESS-FLOOR", "PROVENANCE-SPLICED"))
    fams = family_verdicts(joined)
    eng_rows, eng_notes = engagement_rows(args.control_dir, args.arm)
    warnings += eng_notes

    camp_uuids, ctrl_uuids = uuids_for(args.campaign_dir), uuids_for(arm_dir)
    prov = provenance_rows(camp_sum, ctrl_sum, camp_floors, ctrl_floors,
                           camp_uuids, ctrl_uuids)
    if use_log:
        prov += driverlog_provenance_rows(args.driver_log, args.arm, contam, log_events,
                                          staged_child_hw(arm_dir), args.tenant_mib)

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
    write_noise_floor(args.out, nf, shadow)
    write_family_verdicts(args.out, fams)
    write_engagement(args.out, eng_rows)
    write_provenance(args.out, prov)
    excluded = write_excluded(args.out, joined, camp_rows, ctrl_rows, camp_floors,
                              ctrl_floors, contam)

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

    # f03/f10 are the band itself and are never "judged". NO-DATA groups have no delta, and
    # EXCLUDED groups have one that this report refuses to use -- three different states,
    # named separately, none of them folded into "inside the band".
    excluded_verdicts = ("TENANT-CONTAMINATED", "EXCLUDED-CELLS")
    judged = [f for f in fams
              if f["verdict"] not in ("NOISE-FLOOR", "NO-DATA") + excluded_verdicts]
    out_groups = [f for f in judged if f["verdict"] in ("OUTSIDE-BAND-HIGH",
                                                        "OUTSIDE-BAND-LOW", "MIXED")]
    nodata = [f for f in fams if f["verdict"] == "NO-DATA"]
    excl_groups = [f for f in fams if f["verdict"] in excluded_verdicts]
    n_out = len(out_groups)
    excl_clause = ""
    if excl_groups:
        by_fam = sorted({f["family"] for f in excl_groups})
        excl_clause = (
            f"EXCLUDED, NOT JUDGED, AND MARKED FOR RE-MEASUREMENT: "
            f"{len(excl_groups)} of {len(fams)} (fusion, variant) group(s) across "
            f"{len(by_fam)} famil(y/ies) -- {', '.join(by_fam)} -- "
            + "; ".join(f"{label(f)}: {f['verdict']}" for f in excl_groups)
            + ". These are NOT inside the drift band and NOT a null result: their cells were "
              "measured and then rejected for a stated reason (see excluded.csv). "
              "Re-measuring them is not cleanup, it is the experiment. ")
    partial = [f for f in judged if f["n_excluded"]]
    if partial:
        excl_clause += (
            f"{len(partial)} further group(s) are judged on their surviving cells with some "
            f"cells excluded: "
            + "; ".join(f"{label(f)} ({f['n_excluded']} of {f['n_cells']})"
                        for f in partial) + ". ")
    nodata_clause = ""
    if nodata:
        nodata_clause = (
            f"{len(nodata)} of {len(fams)} (fusion, variant) group(s) have NO usable cell "
            f"in the '{args.arm}' arm and therefore NO delta at all -- they are NOT inside "
            f"the band and NOT a null result, they are UNMEASURED: "
            + "; ".join(label(f) for f in nodata)
            + " (the per-cell `verdict` column says why: MISSING-CONTROL, "
              "UNRESOLVED-ONE-SIDE, INCOMPARABLE-BASIS). ")
    nodata_clause = excl_clause + nodata_clause
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
        # THE NULL EXPECTATION, IN THE HEADLINE, NOT A FOOTNOTE. A min/max band over n
        # reference cells has a per-cell false-exceedance rate of 2/(n+1) under exchange-
        # ability: a fresh draw is outside exactly when it is the new min or the new max.
        # Reporting a raw exceedance count without that baseline invites the reader to treat
        # drift as signal, which is the one error this whole report exists to prevent.
        n_band = int(nf.get("n_usable") or 0)
        n_judged_cells = sum(1 for r in joined
                             if r["family"] not in NOISE_FAMILIES
                             and r["verdict"] in ("INSIDE", "OUTSIDE-HIGH", "OUTSIDE-LOW"))
        n_out_cells = sum(1 for r in joined
                          if r["family"] not in NOISE_FAMILIES
                          and r["verdict"] in ("OUTSIDE-HIGH", "OUTSIDE-LOW"))
        rate = (2.0 / (n_band + 1)) if n_band else None
        stat = ""
        if rate and n_judged_cells:
            exp = rate * n_judged_cells
            p = binom_sf(n_out_cells - 1, n_judged_cells, rate)
            verdict_word = ("CONSISTENT WITH DRIFT" if p > 0.05 else
                            "in excess of drift")
            stat = (f" WHAT THAT IS WORTH: {n_out_cells} of {n_judged_cells} judged CELLS "
                    f"fall outside, against {exp:.1f} expected by chance alone -- a min/max "
                    f"band over {n_band} reference cells is exceeded by a fresh "
                    f"exchangeable draw {100 * rate:.1f} % of the time, by construction. "
                    f"One-sided binomial P(X >= {n_out_cells}) = {p:.3f}, so the count is "
                    f"**{verdict_word}** and is NOT a finding. Per group of 11 cells the "
                    f"chance of at least one exceedance is "
                    f"{100 * (1 - (1 - rate) ** 11):.0f} %, which is why most groups have "
                    f"one; the group-level count below is an artefact of that, not a result.")
        headline = (nodata_clause
                    + f"{n_out} of {len(judged)} judged (fusion, variant) group(s), "
                      f"spanning {n_out_fams} of {n_jud_fams} families, have at least one "
                      f"cell outside the [{sig(lo)}, {sig(hi)}] drift band ({names}); the "
                      f"other {len(judged) - n_out} judged group(s) sit entirely inside it."
                    + stat)
    # THE EXCLUSION, PRICED, in the headline itself. The band that drives every verdict above
    # depends on throwing cells out; a reader must be told in the same breath what the
    # verdicts would have been had they stayed, rather than having to find a CSV.
    slo, shi = band_edges(shadow["global"], args.band)
    smde, _ = resolving_power(slo, shi)
    n_readmitted = shadow["n_usable"] - nf["n_usable"]
    if n_readmitted > 0 and slo is not None:
        n_out_shadow = 0
        for r in joined:
            if r["family"] in NOISE_FAMILIES or r["delta_rel"] is None:
                continue
            if r["verdict"] in ("INSIDE", "OUTSIDE-HIGH", "OUTSIDE-LOW") \
                    and not (slo <= r["delta_rel"] <= shi):
                n_out_shadow += 1
        headline += (
            f" BAND SENSITIVITY: the band above is [{sig(lo)}, {sig(hi)}] over "
            f"{nf['n_usable']} f03/f10 cells after {n_readmitted} cell(s) were excluded by "
            f"the stated validity rule; retaining them instead gives [{sig(slo)}, "
            f"{sig(shi)}] over {shadow['n_usable']} cells (min detectable effect "
            f"{mag(smde)} instead of {mag(mde)}), against which {n_out_shadow} judged cell(s) "
            f"in any family would fall outside. Both bands are published; neither licenses a "
            f"causal claim.")

    write_readme(args.out, args.arm, nf, fams, joined, prov, eng_rows, args, camp_sum,
                 ctrl_sum, warnings, forced, offered, headline, shadow, excluded, contam)

    print(f"wrote {args.out}/ : {len(R.order_regimes({r['regime'] for r in joined}))} "
          f"control_<regime>.csv, noise_floor.csv, noise_floor_by_class.csv, "
          f"family_verdicts.csv, engagement.csv, provenance.csv, excluded.csv "
          f"({len(excluded)} row(s)), README.md", flush=True)
    print(f"noise floor (f03+f10): n={nf['n_usable']}/{nf['n_total']} usable, "
          f"delta_rel in [{sig(lo)}, {sig(hi)}], median "
          f"{sig(nf['global']['median'])}", flush=True)
    print(headline, flush=True)
    for w in warnings:
        print(f"  ! {w}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
