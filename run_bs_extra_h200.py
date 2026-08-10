#!/usr/bin/env python3
"""Stand-alone overlay runner that adds the bs2/bs4/bs8/bs16 decode regimes to the H200
suite WITHOUT re-measuring the campaign's existing regimes.

    python3 run_bs_extra_h200.py --gpu auto

WHY THIS FILE EXISTS SEPARATELY FROM `run_h200.py`.

The campaign table samples decode at `bs1` and `bs32`, and the whole-layer fusion gains
collapse between them (1.26x at bs1 -- launch-latency-bound -- to ~1.01x at bs32 --
compute-bound). To see where the crossover actually happens we need bs2/4/8/16. Re-running
the campaign would re-measure what is already clean, device-fenced data in `results/h200/`.
This driver measures ONLY the new regimes, into a STAGING tree, then appends the new-regime
slices into the campaign's result files.

It reuses `run_h200.py` wholesale for GPU selection, tenancy refusal, the preflight, the
sm_90 device gate, hwinfo and child-process supervision -- the same single-implementation
rule `run_f11_h200.py` exists to protect.

WHAT IS DIFFERENT HERE, and why each difference is load-bearing:

1.  **Every bench runs into a STAGING results directory, never the campaign's.** Several
    benches (`f01`, `f06`, `f08f09`) call `record()` ONCE at the end and rewrite the whole
    result file; every bench rewrites it when asked for any regime. Pointing them at
    `results/h200/` with only the new regimes would silently DELETE the campaign's rows --
    the exact accident this task exists to avoid. So the new measurements land in
    `<results>/_bs_extra_rerun/`, and a separate, guarded merge appends only the new
    regimes' rows across.

2.  **The benches are launched with only the new regimes and resume from their own
    per-regime checkpoints in the staging tree.** An attempt that dies hard is relaunched;
    the quarantined regime is only the one it died on, so the loop terminates and a poisoned
    regime can never cost the other three.

3.  **The merge is append-only, idempotent and device-fenced.** A fresh row is added only
    for a regime the canonical file has no row for (so re-running the merge is a no-op and a
    canonical row is never silently replaced); the fresh payload and the canonical file must
    name the same device or the merge is refused; a backup is taken before any write; nothing
    outside the new regimes' rows is touched. `summary.json` can be regenerated afterwards
    with `run_h200.py --summary-only`.

4.  **The whole-layer bench is staged separately** (into
    `<results>/_bs_extra_layer_rerun/`) with all 18 configurations, and only whole NEW
    regime blocks of `layer_configurations.json` are merged in --never the campaign's, and
    never a partial block.

Like `run_h200.py` and `run_f11_h200.py`, this file imports nothing from the suite and
nothing from torch: it must stay able to report that the stack is broken, and it must not
hold a CUDA context for the hours its children are running.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    import run_h200 as R
except Exception as exc:  # noqa: BLE001
    print(f"!! cannot import run_h200.py from {REPO}: {type(exc).__name__}: {exc}\n"
          f"!! This driver deliberately reuses run_h200's GPU selection, tenancy refusal\n"
          f"!! and device gate rather than restating them. Run it from the repo root.",
          flush=True)
    raise SystemExit(2) from exc

DEFAULT_RESULTS = R.DEFAULT_RESULTS
DEFAULT_LOGDIR = REPO / "log" / "run_bs_extra_h200"
REGIME_ABBR = R.REGIME_ABBR

LAYER_RESULT = "layer_configurations"
STAGING_DIRNAME = "_bs_extra_rerun"
LAYER_STAGING_DIRNAME = "_bs_extra_layer_rerun"

# Post-campaign regimes this overlay exists to add. Overridable with --regimes.
NEW_REGIMES = ["decode_bs2", "decode_bs4", "decode_bs8", "decode_bs16"]

#: Family keys for the post-campaign sweep, cheapest-failure-first like `run_h200.FAMILIES`.
FAMILY_KEYS = ("f03", "f10", "f01", "f04f05", "f11", "f06", "f08f09")

#: Lists in a family result file that run PARALLEL to `rows` (one entry per row, same
#: order), so appending a slice of `rows` requires appending the same slice of these.
PARALLEL_ROWS_LISTS: dict[str, tuple[str, ...]] = {
    "f01": ("tuning",),
}

_REGIME_ALT = "|".join(re.escape(r) for r in R.KNOWN_REGIMES)


# ======================================================================================
# small JSON helpers (same contract as run_f11_h200.py)
# ======================================================================================
def load_json(path: Path) -> tuple[dict | None, str]:
    """(payload, error). Never raises: a corrupt result file is a fact to report, not a
    reason to abandon a run that can still measure everything else."""
    if not path.exists():
        return None, "missing"
    try:
        blob = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        return None, f"unreadable ({type(exc).__name__}: {exc})"
    if not isinstance(blob, dict):
        return None, "not a JSON object"
    return blob, ""


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write via a sibling temp file and rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def digest(obj) -> str:
    """Stable short digest, used to make the merge idempotent across re-invocations."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8", "replace")
    return hashlib.sha256(blob).hexdigest()[:16]


def meta_device(payload: dict | None) -> str:
    return R._norm_dev(((payload or {}).get("_meta") or {}).get("device"))


def last_regime_in_log(path: Path) -> str | None:
    """Last regime-name token in a bench log, used to pick the write after a hard abort.
    Regime names are echoed by every bench regardless of banner style."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    last = None
    for m in re.finditer(rf"\b({_REGIME_ALT})\b", text):
        last = m.group(1)
    if last in R.KNOWN_REGIMES:
        return last
    return None


# ======================================================================================
# reading the staged / canonical result files
# ======================================================================================
def regime_rows(payload: dict | None) -> dict[str, list[dict]]:
    """{regime: [rows]} from a family result file, tolerating a missing-shape."""
    out: dict[str, list[dict]] = {}
    for row in ((payload or {}).get("rows") or []):
        if isinstance(row, dict) and row.get("regime"):
            out.setdefault(str(row["regime"]), []).append(row)
    return out


def family_canonical(fam: R.Family, results: Path) -> Path:
    """The canonical result file for a family, resolved via the driver's glob rules, so the
    merge targets exactly the file the campaign wrote."""
    got = R.find_result(fam, results)
    return got if got is not None else results / f"{fam.key}.json"


def staged_family_path(fam: R.Family, staging: Path, results: Path) -> Path:
    """The result file a bench writes into `staging` for `fam`.

    Benches name their file after their own RESULT_ID (f03_resadd_rmsnorm.json), NOT after
    the family key (f03.json). The driver used `f"{fam.key}.json"` and every merge refused
    with "missing" while the staged file sat next to it under its RESULT_ID name -- the
    reason the whole first run merged nothing.
    """
    return staging / family_canonical(fam, results).name


def canonical_mult(fam: R.Family, results: Path) -> int:
    """Rows-per-regime the canonical file carries (the modal count over its completed
    regimes); the completeness bar used for the new regimes too."""
    payload, _ = load_json(family_canonical(fam, results))
    counts = Counter(len(v) for v in regime_rows(payload).values())
    return counts.most_common(1)[0][0] if counts else 1


def same_device(canonical: dict | None, fresh: dict | None, device_name: str,
                enabled: bool) -> str | None:
    """Error string if `fresh` may not be merged into `canonical`, else None."""
    if not enabled:
        return None
    can_dev = meta_device(canonical)
    fresh_dev = meta_device(fresh)
    if can_dev and fresh_dev and can_dev != fresh_dev:
        return (f"canonical records device '{can_dev}' but the re-measurement records "
                f"'{fresh_dev}'")
    if device_name and fresh_dev and fresh_dev != device_name:
        return (f"the re-measurement records device '{fresh_dev}' but this box is "
                f"'{device_name}'")
    if not fresh_dev:
        return "the fresh payload records no device"
    return None


# ======================================================================================
# child supervision -- one attempt at a time
# ======================================================================================
def launch(log: R.Log, args: argparse.Namespace, key: str, title: str,
           script: Path, timeout_s: int, regimes: list[str], results: Path,
           logdir: Path, extra: list[str], gpu: dict, note: str = "") -> dict:
    """One supervised child, through `run_h200.run_family`."""
    fam = R.Family(key=key, title=title, script_globs=(script.name,), result_globs=(),
                   timeout_s=timeout_s, note=note)
    sub = argparse.Namespace(**vars(args))
    sub.regimes = ",".join(regimes)
    extra = list(extra)
    if "--regimes" not in R.script_flags(script):
        # The benches register --regimes inside add_std_args() (glm52_h200/bench/__init__.py);
        # run_h200.script_flags() only scans the bench's own source and cannot see it, so
        # run_family falls back to the GLM52_REGIMES env, which NO bench reads: with the
        # campaign's 7 + our 4 regimes in REGIME_NAMES, every bench silently measured all 11
        # into the staging tree last run. The explicit flag is the only thing that works.
        extra += ["--regimes", ",".join(regimes)]
    return R.run_family(log, fam, script, sub, results, logdir, extra, gpu)


def run_family_stage(log, args, fam: R.Family, script: Path, results: Path,
                     staging: Path, logdir: Path, gpu: dict, scope: list[str],
                     extra: list[str], warnings: list[str]) -> dict:
    """Drive one family's new-regime measurement into the staging tree, resumable per regime.

    THE INVARIANT: an attempt is handed only the regimes still in scope.  The benches
    checkpoint per regime, so a completed regime is replayed from its checkpoint on the next
    attempt and only the unfinished ones are re-measured.  A hard abort that made no progress
    has the regime it died on quarantined out of the next attempt.  Both moves are monotone,
    so the loop is bounded by `len(scope)` whichever way the bench fails.
    """
    R.rule(log, f"{fam.key} -- {fam.title} over {len(scope)} regime(s)")
    result_path = staged_family_path(fam, staging, results)
    mult = canonical_mult(fam, results)
    poisoned: dict[str, str] = {}
    attempts: list[dict] = []
    max_attempts = args.max_attempts or (len(scope) + 2)

    for n in range(1, max_attempts + 1):
        live = [r for r in scope if r not in poisoned]
        have = regime_rows(load_json(result_path)[0])
        todo = [r for r in live if len(have.get(r, ())) < mult]
        if not live:
            log("  !! every regime in scope has been quarantined; nothing left to attempt.")
            break
        if not todo:
            log(f"  all {len(live)} regime(s) in scope have rows; {fam.key} is done.")
            break

        log("")
        log(f"  attempt {n}/{max_attempts}: {len(todo)} regime(s) to measure "
            f"({', '.join(todo)})")
        done = [r for r in live if len(have.get(r, ())) >= mult]
        if done:
            log(f"    {len(done)} already done: {', '.join(sorted(done))}")
        if poisoned:
            log("    quarantined out: "
                + "; ".join(f"{k} ({v})" for k, v in poisoned.items()))

        rec = launch(log, args, f"{fam.key}.a{n}", fam.title, script,
                     args.timeout or fam.timeout_s, todo, staging, logdir, extra, gpu,
                     note=fam.note)
        rec["attempt"] = n
        rec["regimes_requested"] = todo
        attempts.append(rec)

        fresh = regime_rows(load_json(result_path)[0])
        gained = [r for r in todo if len(fresh.get(r, ())) >= mult]
        done_live = [r for r in live if len(fresh.get(r, ())) >= mult]
        rec["regimes_gained"] = gained
        if rec.get("status") == "interrupted":
            log("  !! interrupted -- stopping this family's stage here.")
            break
        if gained:
            log(f"    +{len(gained)} regime(s): {', '.join(gained)}")
            continue
        if rec.get("status") == "ok" and set(done_live) == set(live):
            log(f"  {fam.key} finished cleanly.")
            break

        died_on = last_regime_in_log(Path(rec.get("log") or ""))
        if died_on and died_on in live and len(fresh.get(died_on, ())) < mult:
            poisoned[died_on] = (f"attempt {n} exited {rec.get('returncode')} "
                                 f"({rec.get('status')}) while measuring it, with no clean row")
            warnings.append(f"{fam.key} regime {died_on} was quarantined: {poisoned[died_on]}")
            log(f"  !! QUARANTINING regime {died_on}: {poisoned[died_on]}")
            continue
        log(f"  !! attempt {n} made no progress and the failing regime could not be "
            f"identified from its log; stopping the {fam.key} stage.")
        break

    payload, err = load_json(result_path)
    rows = regime_rows(payload)
    done = [r for r in scope if len(rows.get(r, ())) >= mult]
    missing = [r for r in scope if len(rows.get(r, ())) < mult]
    return {
        "family": fam.key, "result": str(result_path), "result_error": err or None,
        "attempts": attempts, "poisoned": poisoned,
        "regimes_done": done, "regimes_missing": missing, "rows_mult": mult,
    }


# ======================================================================================
# the merges -- append-only, idempotent, device-fenced
# ======================================================================================
def merge_family(log: object, canonical_path: Path, staged_path: Path,
                 args: argparse.Namespace, fam_key: str, scope: list[str],
                 warnings: list[str], device_name: str) -> dict:
    """Append the new-regime rows of a staged family file into the campaign file, guarded.

    Rules, applied to every key that carries per-regime data:
      1. DEVICE.  Fresh and canonical must name the same device, or the merge is refused.
      2. APPEND-ONLY.  A row (regime, variant) already present in the canonical file is
         never touched.  The merge only ever fills gaps -- which is what makes it idempotent
         and makes a repeated merge a no-op.
      3. SCOPE.  Only rows whose regime is in `scope` are copied; nothing else in the
         canonical file is modified.
      4. PARALLEL LISTS.  For families whose result carries a list parallel to `rows`
         (f01's `tuning`), the same slice is appended so the two stay aligned.
    """
    report: dict = {"canonical": str(canonical_path), "decisions": [], "written": False,
                    "backup": None, "refused": None}
    canonical, cerr = load_json(canonical_path)
    if canonical is None:
        report["refused"] = f"{canonical_path.name}: {cerr}"
        log(f"  !! REFUSING TO MERGE: {report['refused']}.")
        return report
    fresh, ferr = load_json(staged_path)
    if fresh is None:
        report["refused"] = f"{staged_path.name}: {ferr}"
        log(f"  !! REFUSING TO MERGE: {report['refused']}. The staging tree has not "
            f"produced a result for this family.")
        return report

    err_dev = same_device(canonical, fresh, device_name, args.device_fence)
    if err_dev:
        report["refused"] = f"device mismatch: {err_dev}"
        warnings.append(report["refused"])
        log(f"  !! REFUSING TO MERGE: {report['refused']}")
        return report
    if args.quick and not args.merge_quick:
        report["refused"] = "--quick without --merge-quick"
        log("  !! [merge] REFUSED: --quick is a smoke test and its numbers must not enter a "
            "campaign file. --merge-quick forces it.")
        return report

    merged = copy.deepcopy(canonical)
    want = set(scope)
    old = {(str(r.get("regime")), str(r.get("variant") or ""))
           for r in ((canonical.get("rows")) or []) if isinstance(r, dict)}

    staged_rows = fresh.get("rows") or []
    new_rows: list[dict] = []
    new_idx: list[int] = []
    for i, row in enumerate(staged_rows):
        if not isinstance(row, dict) or str(row.get("regime")) not in want:
            continue
        key = (str(row.get("regime")), str(row.get("variant") or ""))
        if key in old:
            continue
        new_rows.append(row)
        new_idx.append(i)

    for row in new_rows:
        merged.setdefault("rows", []).append(row)
        report["decisions"].append({"action": "append_row",
                                    "regime": str(row.get("regime")),
                                    "variant": str(row.get("variant") or "")})

    for key in sorted(fresh.keys()):
        fv = fresh[key]
        if not isinstance(fv, dict):
            continue
        overlapped = want & set(fv)
        if not overlapped:
            continue
        mv = merged.get(key)
        if not isinstance(mv, dict):
            mv = merged[key] = {}
        for reg in sorted(overlapped):
            if reg not in mv:
                mv[reg] = fv[reg]

    for key in PARALLEL_ROWS_LISTS.get(fam_key, ()):
        src = fresh.get(key)
        if not isinstance(src, list):
            continue
        dst = merged.setdefault(key, [])
        if not isinstance(dst, list):
            dst = merged[key] = []
        for i in new_idx:
            if i < len(src):
                dst.append(src[i])

    session = digest({"regimes": {
        r: len([x for x in new_rows if str(x.get("regime")) == r]) for r in sorted(want)}})
    amendments = merged.setdefault("_meta", {}).setdefault("amendments", [])
    amendments.append({
        "at": time.strftime("%Y-%m-%d %H:%M:%S"), "by": "run_bs_extra_h200.py",
        "session": session,
        "note": f"appended rows for {', '.join(sorted(want))} only; campaign rows untouched"})

    if json.dumps(merged, sort_keys=True, default=str) == \
            json.dumps(canonical, sort_keys=True, default=str):
        log(f"  [merge] nothing changed -- {canonical_path.name} is already up to date.")
        report.update(written=False, n_rows=0, session=session)
        return report
    if args.dry_run:
        log(f"  [merge] --dry-run: {len(new_rows)} row(s) WOULD be appended to "
            f"{canonical_path.name}; nothing was modified.")
        report.update(written=False, n_rows=len(new_rows), session=session, dry_run=True)
        return report

    backup = canonical_path.with_name(
        f"{canonical_path.stem}.pre_bs_extra_{time.strftime('%Y%m%d_%H%M%S')}.json")
    try:
        shutil.copy2(canonical_path, backup)
        report["backup"] = str(backup)
        log(f"  [merge] backup -> {backup.name}")
    except OSError as exc:
        report["refused"] = f"backup failed: {exc}"
        log(f"  !! could not write a backup ({exc}); refusing to modify the campaign file.")
        return report
    atomic_write_json(canonical_path, merged)
    report.update(written=True, n_rows=len(new_rows), session=session)
    log(f"  [merge] appended {len(new_rows)} row(s) to {canonical_path.name}.")
    return report


def _regime_block_measured(block: dict) -> bool:
    """Did this regime block actually produce usable whole-layer rows?"""
    ck = (block.get("correctness") or {})
    ok = [n for n, v in ck.items() if isinstance(v, dict) and v.get("ok") is True]
    if not ok:
        return False
    return bool(block.get("pass1")) and bool(block.get("pass2"))


def accumulate_layer(staging: Path, accum_path: Path, scope: list[str],
                     logf=None) -> dict:
    """Fold the `layer_configurations.json` into a per-attempt accumulator.

    `bench_layer.py` rewrites its result file after every regime, keeping only the regimes it
    measured in THIS process; across attempts the newest file is not the most complete one.
    The accumulator keeps every regime block it has ever seen, which is what makes the layer
    stage resumable even though the bench has no per-regime measurement checkpoint.
    """
    fresh, err = load_json(staging / f"{LAYER_RESULT}.json")
    accum, _ = load_json(accum_path)
    accum = accum or {"id": f"{LAYER_RESULT}__bs_extra", "regimes": {}}
    accum.setdefault("regimes", {})
    if fresh is None:
        if err != "missing" and logf is not None:
            logf(f"  !! staging {LAYER_RESULT}.json: {err}")
        return accum
    for key in ("id", "protocol", "env", "fairness", "configs", "_meta"):
        if key in fresh:
            accum[key] = fresh[key]
    got = []
    for name, block in (fresh.get("regimes") or {}).items():
        if name in scope:
            accum["regimes"][name] = block
            got.append(name)
    if got and logf is not None:
        logf(f"  [accum] absorbed {len(got)} regime(s): {', '.join(sorted(got))}")
    atomic_write_json(accum_path, accum)
    return accum


def run_layer_stage(log, args, script: Path, staging: Path, logdir: Path,
                    gpu: dict, scope: list[str], warnings: list[str]) -> dict:
    """Measure `bench_layer.py` for the new regimes into a staging tree."""
    R.rule(log, f"WHOLE-LAYER measurement over {', '.join(scope)}")
    log(f"  staging dir      {staging}")
    log(f"  protocol         two interleaved passes x {args.rounds} rounds, LOG-11 tie rule")
    log("")
    log("  WHY SEPARATE STAGING: common.record() rewrites layer_configurations.json")
    log("  wholesale, and bench_layer writes only the regimes it was asked for. Running it")
    log("  straight into results/h200/ would REPLACE the campaign's regimes with a file")
    log("  holding just these four. The merge appends whole NEW regime blocks.")

    accum_path = staging / "layer_rerun_accumulated.json"
    attempts: list[dict] = []
    max_attempts = args.max_attempts or (len(scope) + 2)
    poisoned: dict[str, str] = {}

    for n in range(1, max_attempts + 1):
        accum = accumulate_layer(staging, accum_path, scope, log)
        have = set(accum.get("regimes") or {})
        live = [r for r in scope if r not in poisoned]
        todo = [r for r in live if r not in have]
        if not todo:
            log(f"  every regime in scope has a layer measurement "
                f"({', '.join(sorted(have & set(scope)))}).")
            break
        log("")
        log(f"  attempt {n}/{max_attempts}: {', '.join(todo)}")
        extra = ["--rounds", str(args.rounds), "--rep", str(args.rep)]
        extra += list(args.layer_args)
        rec = launch(log, args, "layer.bn", "whole-layer bench", script,
                     args.timeout or args.layer_timeout, todo, staging, logdir, extra, gpu,
                     note="staged; merged separately and guarded")
        rec["attempt"] = n
        rec["regimes_requested"] = todo
        attempts.append(rec)

        accum = accumulate_layer(staging, accum_path, scope, log)
        gained = sorted(set(accum.get("regimes") or {}) - have)
        rec["regimes_gained"] = gained
        if rec.get("status") == "interrupted":
            log("  !! interrupted -- stopping the layer stage here.")
            break
        if gained:
            log(f"    +{len(gained)} regime(s): {', '.join(gained)}")
            continue
        died_on = last_regime_in_log(Path(rec.get("log") or ""))
        if died_on and died_on in todo:
            poisoned[died_on] = (f"attempt {n} exited {rec.get('returncode')} "
                                 f"({rec.get('status')}) while measuring it")
            log(f"  !! QUARANTINING layer regime {died_on}: {poisoned[died_on]}")
            warnings.append(f"layer regime {died_on} quarantined: {poisoned[died_on]}")
            continue
        log(f"  !! attempt {n} made no progress; stopping the layer stage.")
        break

    accum = accumulate_layer(staging, accum_path, scope, log)
    return {
        "staging": str(staging), "accumulated": str(accum_path), "payload": accum,
        "attempts": attempts, "poisoned": poisoned,
        "regimes_done": sorted(set(accum.get("regimes") or {}) & set(scope)),
    }


def merge_layer(log: object, canonical_path: Path, fresh: dict, args: argparse.Namespace,
                scope: list[str], warnings: list[str], device_name: str) -> dict:
    """Append whole new-regime blocks into `layer_configurations.json`, guarded."""
    report: dict = {"canonical": str(canonical_path), "written": False, "backup": None,
                    "refused": None, "decisions": []}
    canonical, cerr = load_json(canonical_path)
    if canonical is None:
        report["refused"] = f"{canonical_path.name}: {cerr}"
        log(f"  !! REFUSING TO MERGE: {report['refused']}.")
        return report
    err_dev = same_device(canonical, fresh, device_name, args.device_fence)
    if err_dev:
        report["refused"] = f"device mismatch: {err_dev}"
        warnings.append(report["refused"])
        log(f"  !! REFUSING TO MERGE: {report['refused']}")
        return report
    if args.quick and not args.merge_quick:
        report["refused"] = "--quick without --merge-quick"
        log("  !! [merge] REFUSED: --quick is a smoke test and its numbers must not enter a "
            "campaign file. --merge-quick forces it.")
        return report

    merged = copy.deepcopy(canonical)
    fresh_blocks = (fresh.get("regimes") or {})
    scope_set = set(scope)
    session = digest({"regimes": {name: block.get("verdict")
                                  for name, block in sorted(fresh_blocks.items())
                                  if name in scope_set}})

    written = []
    for name in sorted(scope_set):
        block = fresh_blocks.get(name)
        if not isinstance(block, dict):
            report["decisions"].append({"regime": name, "action": "skip_no_block"})
            continue
        if name in (merged.get("regimes") or {}) and not args.remerge_good:
            report["decisions"].append({"regime": name, "action": "skip_already_present"})
            log(f"  {name:<16} SKIP -- already present in the canonical file")
            continue
        if not _regime_block_measured(block):
            report["decisions"].append({"regime": name, "action": "skip_not_measured"})
            log(f"  {name:<16} SKIP   block did not produce clean whole-layer rows")
            continue
        entry = dict(block)
        entry["_source"] = {"bs_extra_session": session, "by": "run_bs_extra_h200.py"}
        merged.setdefault("regimes", {})[name] = entry
        written.append(name)
        log(f"  {name:<16} WRITE  new regime block")

    prov = merged.setdefault("bs_extra", {})
    prov.setdefault("schema", 1)
    prov.setdefault("why", (
        "decode_bs2/4/8/16 were added after the campaign to resolve the bs1 -> bs32 cliff in "
        "the whole-layer fusion gains. Their `regimes[*]` blocks here come from "
        "run_bs_extra_h200.py and are complete regime measurements with their own verdict; "
        "every other regime's block is the campaign's, untouched."))
    prov.setdefault("sessions", []).append({
        "session": session, "driver": "run_bs_extra_h200.py",
        "device": device_name or None, "regimes": written,
        "merged_at": time.strftime("%Y-%m-%d %H:%M:%S")})

    amendments = merged.setdefault("_meta", {}).setdefault("amendments", [])
    amendments.append({"at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "by": "run_bs_extra_h200.py", "session": session, "regimes": written})

    if json.dumps(merged, sort_keys=True, default=str) == \
            json.dumps(canonical, sort_keys=True, default=str):
        log(f"  [merge] nothing changed -- {canonical_path.name} is already up to date.")
        report.update(written=False, n_rows=len(written), session=session)
        return report
    if args.dry_run:
        log(f"  [merge] --dry-run: {len(written)} regime block(s) WOULD be appended to "
            f"{canonical_path.name}; nothing was modified.")
        report.update(written=False, n_rows=len(written), session=session, dry_run=True)
        return report

    backup = canonical_path.with_name(
        f"{canonical_path.stem}.pre_bs_extra_{time.strftime('%Y%m%d_%H%M%S')}.json")
    try:
        shutil.copy2(canonical_path, backup)
        report["backup"] = str(backup)
        log(f"  [merge] backup -> {backup.name}")
    except OSError as exc:
        report["refused"] = f"backup failed: {exc}"
        log(f"  !! could not write a backup ({exc}); refusing to modify the campaign file.")
        return report
    atomic_write_json(canonical_path, merged)
    report.update(written=True, n_rows=len(written), session=session)
    log(f"  [merge] appended {len(written)} regime block(s) to {canonical_path.name}.")
    return report


# ======================================================================================
# summary regeneration
# ======================================================================================
def regen_summary(log, args: argparse.Namespace, results: Path, gpu: dict) -> dict:
    """Regenerate `summary.json` via `run_h200.py --summary-only` (no measurements; just the
    per-family table over the merged files)."""
    cmd = [args.python, str(REPO / "run_h200.py"), "--summary-only",
           "--results-dir", str(results)]
    if gpu.get("index") is not None:
        cmd += ["--gpu", str(gpu["index"])]
    log(f"  regen summary: {' '.join(cmd)}")
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        out = (cp.stdout or "")[-1500:] + "\n" + (cp.stderr or "")[-1500:]
        return {"command": cmd, "returncode": cp.returncode, "tail": out}
    except Exception as exc:  # noqa: BLE001
        return {"command": cmd, "error": f"{type(exc).__name__}: {exc}"}


# ======================================================================================
# final status table
# ======================================================================================
def _cell(text: str, width: int = 8) -> str:
    return f"{text:>{width}}"


def status_table(stages: dict, scope: list[str]) -> list[str]:
    table = ["  " + f"{'family':<10}"
             + "".join(_cell(REGIME_ABBR.get(r, r)) for r in scope),
             "  " + "-" * (68 + len(scope) * 8)]
    for key in FAMILY_KEYS:
        st = stages.get(key) or {}
        cells = []
        for r in scope:
            if r in (st.get("regimes_done") or []):
                cells.append(_cell("merged"))
            elif r in (st.get("regimes_missing") or []):
                cells.append(_cell("MISS"))
            elif r in (st.get("poisoned") or {}):
                cells.append(_cell("quart"))
            else:
                cells.append(_cell("-"))
        table.append(f"  {key:<10}" + "".join(cells))
    table.append("")
    table.append("  merged = rows appended into the campaign file; MISS = no row produced; "
                 "quart = quarantined")
    return table


# ======================================================================================
# CLI + main
# ======================================================================================
def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="run_bs_extra_h200.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Measure the small decode batch sizes (bs2/4/8/16) on the H200 and "
                    "append the rows into results/h200, without re-running the campaign.",
        epilog="Typical use:\n"
               f"  python3 run_bs_extra_h200.py --gpu auto              # measure + merge\n"
               f"  python3 run_bs_extra_h200.py --gpu 3 --quick     # stack smoke test\n"
               f"  python3 run_bs_extra_h200.py --merge-only        # merge a finished run\n"
               f"  python3 run_bs_extra_h200.py --list              # show the plan, run nothing\n"
               "\n"
               "Everything is measured into <results>/_bs_extra_rerun/ and "
               "_bs_extra_layer_rerun/ first; the merge is the only step that touches the\n"
               "campaign files, and it refuses under --quick and on a device mismatch.\n",
    )
    ap.add_argument("--gpu", default=None, metavar="N|auto",
                    help="which GPU this run uses (see run_h200.py --help). Same selection "
                         "and same refusal-on-tenanted-card.")
    ap.add_argument("--allow-busy", action="store_true",
                    help="measure on a card that already has another tenant (this is what "
                         "produced the preflight's impossible harness floor; short-kernel "
                         "numbers from such a run are not defensible)")
    ap.add_argument("--regimes", default="",
                    help=f"comma-separated ({','.join(R.KNOWN_REGIMES)}) "
                         "default: the four new decode sizes")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--log-dir", default=str(DEFAULT_LOGDIR))
    ap.add_argument("--quick", action="store_true",
                    help="short sweeps. Smoke test only; the merge refuses to run unless "
                         "--merge-quick is also given.")
    ap.add_argument("--merge-quick", action="store_true",
                    help="allow a --quick re-measurement to be merged into a "
                         "campaign file. Almost never what you want.")
    ap.add_argument("--force", action="store_true",
                    help="run even if the device is not sm_90 (results are NOT an H200 "
                         "measurement; use --results-dir to keep them out of results/h200)")
    ap.add_argument("--force-rerun", action="store_true",
                    help="ignore existing checkpoints and re-measure every regime")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="do not run the preflight probe even if its JSON is missing")
    ap.add_argument("--families-only", action="store_true",
                    help="stop after the family benches; skip the whole-layer stage")
    ap.add_argument("--layer-only", action="store_true",
                    help="skip the per-family benches; only the whole-layer stage")
    ap.add_argument("--merge-only", action="store_true",
                    help="run nothing; merge whatever the staging trees already hold")
    ap.add_argument("--remerge-good", action="store_true",
                    help="allow the merge to overwrite rows/blocks that ALREADY exist cleanly"
                         " in the campaign file. Off by default: the merge fills gaps.")
    ap.add_argument("--rounds", type=int, default=8,
                    help="interleaved rounds per pass in the layer bench (LOG-11 used 8)")
    ap.add_argument("--rep", type=int, default=15,
                    help="reps per candidate per round in the layer bench")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="cap on relaunches per family/layer stage (default: regimes + 2)")
    ap.add_argument("--timeout", type=int, default=0,
                    help="per-attempt timeout in seconds, overriding family defaults")
    ap.add_argument("--layer-timeout", type=int, default=18 * 3600)
    ap.add_argument("--heartbeat", type=int, default=60,
                    help="seconds between progress lines (default 60)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter used for the benches (default: this one)")
    ap.add_argument("--layer-args", action="append", default=[], metavar="ARG",
                    help="extra CLI argument passed through to bench_layer.py")
    ap.add_argument("--regen-summary", action="store_true",
                    help="regenerate summary.json afterwards (default off; saved for the "
                         "end so a crash mid-merge cannot lose the register edits)")
    ap.add_argument("--no-device-fence", action="store_true",
                    help="DANGEROUS: merge results without checking which GPU produced them")
    ap.add_argument("--disable-features", default="",
                    help="comma list of Hopper paths to switch off for every child "
                         "(tma,clusters,ws,wgmma,all). Passed through to the benches.")
    ap.add_argument("--flush-mb", type=int, default=0,
                    help="override the L2-flush buffer size in MiB (diagnosis only)")
    ap.add_argument("--allow-concurrent", action="store_true",
                    help="do not refuse when another bench appears to be running")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve everything, print the plan and the merge decisions, but "
                         "launch nothing and write nothing")
    args = ap.parse_args(argv)
    args.device_fence = not args.no_device_fence
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    results = Path(args.results_dir).expanduser().resolve()
    logdir = Path(args.log_dir).expanduser().resolve()
    results.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    log = R.Log(logdir / "driver.log")
    warnings: list[str] = []
    t_start = time.time()

    scope = [r.strip() for r in args.regimes.split(",") if r.strip()] or list(NEW_REGIMES)
    bad = [r for r in scope if r not in R.KNOWN_REGIMES]
    if bad:
        warnings.append(f"regimes not in the canonical set were requested: {bad}")
        log(f"!! regimes {bad} are not in {R.KNOWN_REGIMES}; passing them through anyway")

    staging = results / STAGING_DIRNAME
    layer_staging = results / LAYER_STAGING_DIRNAME
    staging.mkdir(parents=True, exist_ok=True)
    layer_staging.mkdir(parents=True, exist_ok=True)

    if args.list:
        R.rule(log, "PLAN")
        log(f"  regimes          {', '.join(scope)}")
        log(f"  families         {', '.join(FAMILY_KEYS)}  + layer")
        log(f"  family staging   {staging}")
        log(f"  layer staging    {layer_staging}")
        log(f"  merge target     {results / '*.json'}")
        if args.families_only:
            log("  layer stage      SKIPPED (--families-only)")
        if args.layer_only:
            log("  family stages    SKIPPED (--layer-only)")
        if args.merge_only:
            log("  measurement      SKIPPED (--merge-only)")
        log(f"  logs             {logdir}")
        log.close()
        return 0

    # --- GPU selection, preflight, device gate: run_h200's, unchanged --------------------
    hw_start = R.hwinfo()
    gpu = R.resolve_gpu(log, args, hw_start, warnings,
                        measuring=not (args.dry_run or args.merge_only))
    if gpu.get("refuse"):
        log("")
        log("  (the same command for this driver: python3 run_bs_extra_h200.py --gpu <idle index>)")
        log.close()
        return 4
    hw_row = R.pick_hw_row(hw_start, gpu.get("index"))

    pf = R.read_preflight()
    if pf is None and not args.skip_preflight and not args.merge_only \
            and not args.dry_run:
        log(f"  no {R.PREFLIGHT_JSON.name}; running the preflight probe first.")
        pf = R.run_preflight(log, args.python, logdir, quick=args.quick,
                             gpu=gpu.get("index"))
    elif pf is not None and hw_row and not args.merge_only:
        pf_name = R._norm_dev((pf.get("device") or {}).get("name"))
        pf_uuid = R._norm_uuid((pf.get("gpu_selection") or {}).get("uuid")
                               or (pf.get("device") or {}).get("uuid"))
        row_uuid = R._norm_uuid(hw_row.get("uuid"))
        why = ""
        if pf_name != R._norm_dev(hw_row.get("name")):
            why = f"model differs ({pf_name!r} vs {R._norm_dev(hw_row.get('name'))!r})"
        elif gpu.get("index") is not None and pf_uuid and row_uuid and pf_uuid != row_uuid:
            why = f"same model but a different card (probe {pf_uuid}, pinned {row_uuid})"
        if why:
            log(f"!! cached preflight does not describe the pinned GPU: {why} -- re-probing.")
            pf = R.run_preflight(log, args.python, logdir, quick=args.quick,
                                 gpu=gpu.get("index")) or pf

    tick = {"tick_us": ((pf or {}).get("calibration") or {}).get("timer_tick_us")
                       or R.DEFAULT_TICK_US,
            "source": "measured by preflight" if pf else "default (no preflight)",
            "trusted": True, "distrust_reasons": []}
    R.banner(log, pf, hw_start, results, tick, gpu)
    try:
        dev = R.device_gate(log, pf, hw_row, args.force, warnings)
    except SystemExit:
        log("!! (escape hatch: python3 run_bs_extra_h200.py --force --results-dir results/<thisdevice>)")
        log.close()
        raise
    device_name = dev.get("name") or ""

    if not (args.dry_run or args.merge_only):
        busy = R.another_bench_running()
        if busy and not args.allow_concurrent:
            log("!! another glm52_h200 bench appears to be running; refusing (two on one "
                "GPU corrupt every timing).")
            log.close()
            return 3
        if busy:
            warnings.append("--allow-concurrent: another bench process was detected")

    # --- measure the families into staging, merge the new rows --------------------------
    summary_path = results / "bs_extra_summary.json"
    fam_stages: dict[str, dict] = {}
    layer_canonical = family_canonical(R.FAMILY_BY_KEY["layer"], results)
    layer_stage: dict = {}
    merge_report: dict = {}
    save = lambda: (None if args.dry_run
                    else atomic_write_json(summary_path, {
                        "driver": "run_bs_extra_h200.py", "schema": 1,
                        "regimes": scope, "families": list(FAMILY_KEYS),
                        "quick": bool(args.quick), "device": device_name or "",
                        "fam_stages": fam_stages, "layer_merge": merge_report,
                        "warnings": warnings, "wall_s": time.time() - t_start,
                        "live": True}))
    try:
        for key in FAMILY_KEYS:
            fam = R.FAMILY_BY_KEY[key]
            script = R.find_script(fam)
            if script is None:
                warnings.append(f"no script found for {key}")
                log(f"!! no bench found for {key} -- skipped.")
                continue
            canonical_path = family_canonical(fam, results)
            staged_path = staged_family_path(fam, staging, results)
            if not (args.merge_only or args.layer_only or args.dry_run):
                R.rule(log, f"STAGE {key} -- measuring new regimes into the staging tree")
                rec = run_family_stage(log, args, fam, script, results, staging, logdir,
                                       gpu, scope, [], warnings)
                R.check_tenants(log, gpu, f"after {key}", warnings)
                fam_stages[key] = rec
                save()
            if not args.layer_only:
                if args.merge_only:
                    fam_stages.setdefault(key, {})
                report = merge_family(log, canonical_path, staged_path, args, key, scope,
                                      warnings, device_name)
                fam_stages.setdefault(key, {})["merge"] = report
                save()

        # --- the whole-layer stage ------------------------------------------------------
        layer_script = R.find_script(R.FAMILY_BY_KEY["layer"])
        if args.families_only or layer_script is None:
            if layer_script is None:
                warnings.append("bench_layer.py not found; layer stage skipped")
                log("!! no layer bench found -- layer stage skipped.")
            else:
                log("  layer stage skipped (--families-only).")
        else:
            if not (args.merge_only or args.dry_run):
                layer_stage = run_layer_stage(log, args, layer_script, layer_staging,
                                              logdir, gpu, scope)
                R.check_tenants(log, gpu, "after the layer stage", warnings)
                save()
            fresh, err = load_json(layer_staging / "layer_rerun_accumulated.json")
            if fresh is None:
                log(f"  !! no accumulated layer payload to merge: {err}")
            else:
                R.rule(log, f"MERGE into {layer_staging.name} -> {layer_canonical.name}")
                merge_report = merge_layer(log, layer_canonical, fresh, args, scope,
                                           warnings, device_name)
                save()

        # --- regenerate summary if asked ------------------------------------------------
        if args.regen_summary and not args.dry_run:
            R.rule(log, "REGENERATING summary.json")
            sum_rec = regen_summary(log, args, results, gpu)
            if sum_rec.get("returncode") != 0:
                warnings.append(f"summary regen exited {sum_rec.get('returncode')}")
            log(f"  {sum_rec.get('tail', '')[-400:]}")
    except Exception as exc:  # noqa: BLE001 -- a driver must not die silently mid-run
        warnings.append(f"run aborted: {type(exc).__name__}: {exc}")
        log(f"!! RUN ABORTED: {type(exc).__name__}: {exc}")
        import traceback
        log(traceback.format_exc()[-3000:])
        save()

    # --- final report ------------------------------------------------------------------
    hw_end = R.hwinfo()
    drift = R.hw_drift(hw_start, hw_end)
    R.check_tenants(log, gpu, "at the end of the run", warnings)
    R.rule(log, "STATUS -- family x regime")
    for ln in status_table(fam_stages, scope):
        log(ln)
    if merge_report:
        log("")
        if merge_report.get("refused"):
            log(f"  layer merge REFUSED: {merge_report['refused']}")
        else:
            log(f"  layer merge: {merge_report.get('n_rows', 0)} regime block(s) written"
                + (f", backup {Path(merge_report['backup']).name}"
                   if merge_report.get("backup") else ""))
    log("")
    R.rule(log, "SEND BACK")
    log("  1. the whole results dir (or at least the merged *.json + the two staging trees)")
    log("  2. this log dir")

    summary = {
        "driver": "run_bs_extra_h200.py", "schema": 1,
        "regimes": scope, "families": list(FAMILY_KEYS),
        "quick": bool(args.quick), "device": device_name or "",
        "fam_stages": fam_stages, "layer_merge": merge_report,
        "warnings": warnings, "wall_s": time.time() - t_start,
        "live": False,
    }
    if not args.dry_run:
        atomic_write_json(summary_path, summary)
    log.close()
    incomplete = sum(len(v.get("regimes_missing") or []) for v in fam_stages.values())
    layer_missing = 0
    if not args.families_only and not args.dry_run:
        lc, _ = load_json(layer_canonical)
        have_now = set((lc or {}).get("regimes") or {})
        layer_missing = len([r for r in scope if r not in have_now])
    return 1 if (incomplete > 0 or layer_missing > 0) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n interrupted", flush=True)
        raise SystemExit(130)