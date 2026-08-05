#!/usr/bin/env python3
"""Stand-alone re-runner for fusion #11 (lazy pre-norm) on the H200 -- and only #11.

    python3 run_f11_h200.py --gpu auto

WHY THIS FILE EXISTS SEPARATELY FROM `run_h200.py`.

The last H200 campaign produced `results/h200/f11_lazy_prenorm.json` with `complete: true`
and an EMPTY `rows` array. #11 is the campaign's headline experiment -- H200 is the first
device in this study with warp specialization, which is the mechanism "Towards Free
Normalization" (Zhou et al. 2026) depends on -- so the one family that had to be answered is
the one that produced nothing. Re-running the whole campaign to recover it would cost ~20
hours of a machine none of the authors can reach and would re-measure seven families that
already have clean, device-fenced results. This driver re-runs #11 alone, then repairs the
four whole-layer configurations that contain #11 and were excluded for failing the layer
bench's fp32 reference (`O_f11ab`, `P_f10_f11ab`, `Q_f8_f11ab`, `R_f1_f10_f11ab`).

It reuses `run_h200.py` wholesale for GPU selection, tenancy refusal, the preflight, the
sm_90 device gate, hwinfo and child-process supervision. Those rules each cost a real
failure earlier in this study and there must be exactly one implementation of them: this
file IMPORTS them rather than restating them, so a fix to either driver fixes both.

WHAT IS DIFFERENT HERE, and why each difference is load-bearing:

1.  **The f11 bench is launched repeatedly, not once.** `bench_f11_lazy_prenorm.py` catches a
    per-regime exception and checkpoints what it has, but a Triton MLIR assertion or a CUDA
    illegal access ABORTS the process, and everything after that regime is lost. So this
    driver relaunches, always passing the full regime list so the bench reloads its
    device-fenced checkpoints and the result file stays complete; and if an attempt died hard
    without making progress, the regime it died on is QUARANTINED out of the next attempt's
    regime list. Each attempt therefore either finishes or removes one regime, so the loop
    terminates and one poisoned regime can never cost the other six.

2.  **The layer bench is run into a STAGING results directory, never the real one.**
    `common.record()` writes the whole result file every time, and `bench_layer.py` writes
    only the configurations it was asked for. Running it with `--only O_f11ab,...` straight
    into `results/h200/` would silently DELETE the fourteen configurations that already
    measured cleanly -- the exact accident this task exists to avoid. The re-measurement
    lands in `<results>/_f11_layer_rerun/`, and a separate, explicitly guarded merge step
    copies the four target rows across.

3.  **The layer stage is gated on #11's own correctness, and the gate fails CLOSED.** O/P/Q/R
    all use `prenorm="all"`, i.e. both the router GEMM and the w13 GEMM in their fused form.
    Re-measuring them on top of a #11 that is still wrong would republish the same defect
    under a fresh timestamp, which is worse than leaving the gap: the gap is visible. If this
    driver cannot PROVE from the result file that both #11 arms validated in a regime, it
    does not re-measure that regime's layer configurations, and it says so.

4.  **The merge is conservative, idempotent, and never overwrites a good row.** A target row
    is written only when the canonical file has no clean measurement of it; a fresh row that
    failed correctness is never written at all; nothing outside the four target names is
    touched; `verdict` is left exactly as the campaign computed it, because a winner declared
    over two different measurement sessions is not a winner. The cross-session comparison
    lives under its own `f11_rerun` key, anchored on `A_all_unfused` re-measured in the same
    session as the targets -- a ratio to an anchor measured alongside is defensible where a
    ratio across sessions is not.

Like `run_h200.py`, this file imports nothing from the suite and nothing from torch: it must
stay able to report that the stack is broken, and it must not hold a CUDA context for the
hours its children are running.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import re
import shutil
import sys
import time
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
DEFAULT_LOGDIR = REPO / "log" / "run_f11_h200"
KNOWN_REGIMES = R.KNOWN_REGIMES
REGIME_ABBR = R.REGIME_ABBR

F11_RESULT = "f11_lazy_prenorm"
LAYER_RESULT = "layer_configurations"
STAGING_DIRNAME = "_f11_layer_rerun"

# The four whole-layer configurations that contain #11 and were excluded in the campaign for
# failing the layer bench's independent fp32 reference (rel_err 0.127-0.668 at every regime
# above decode_bs1). They are the ONLY names this driver will ever write into
# `layer_configurations.json`.
TARGET_LAYER_CONFIGS = ("O_f11ab", "P_f10_f11ab", "Q_f8_f11ab", "R_f1_f10_f11ab")

# Measured ALONGSIDE the targets and never merged. `A_all_unfused` is the denominator of
# every layer speedup and must be timed in the same interleaved rounds as the targets, or the
# ratio compares a fresh measurement against a session that ended hours earlier on a card at
# a different temperature. `N_f11b` is #11b-only and already measured cleanly in the
# campaign; re-timing it here gives a within-session control for "does adding #11a to #11b
# help at all", which is the actual question these four configurations were built to answer.
DEFAULT_LAYER_ANCHORS = ("A_all_unfused", "N_f11b")

# (key in this driver, key in the f11 row, key in the bench's `arms_unmeasurable`, label)
F11_ARMS: tuple[tuple[str, str, str, str], ...] = (
    ("f11a", "f11a_w13", "f11a_w13", "#11a  lazy pre-norm -> w13 grouped MoE GEMM"),
    ("f11b", "f11b_router", "f11b_router", "#11b  lazy pre-norm -> router GEMM"),
    ("comb", "combined", "combined", "#11a+#11b combined (x2 never materialized)"),
    ("halfR", "half_fused.router", "half_fused.router", "#11b' half-fused router"),
    ("halfM", "half_fused.moe", "half_fused.moe", "#11b' half-fused w13"),
)

# Post-tuning checks in the f11 row's `checks` block that each arm's honesty rests on.
ARM_CHECKS: dict[str, tuple[str, ...]] = {
    "f11a_w13": ("moe_fused", "moe_unfused"),
    "f11b_router": ("router_fused", "router_unfused"),
    "combined": ("router_fused", "router_unfused", "moe_fused", "moe_unfused"),
    "half_fused.router": ("router_half",),
    "half_fused.moe": ("moe_half",),
}

# The layer's `prenorm="all"` configurations run BOTH fused GEMMs, so both #11 arms have to
# be right before re-measuring them is worth a single second of H200 time.
LAYER_GATE_ARMS = ("f11a_w13", "f11b_router")


# ======================================================================================
# small JSON helpers
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
    """Write via a sibling temp file and rename. `common.record` does the same, for the same
    reason: a driver killed mid-write must not leave a half-parsed result file behind, and on
    this campaign the file it would truncate is the one holding fourteen clean rows."""
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


# ======================================================================================
# reading the f11 result: which arms, in which regimes, are real
# ======================================================================================
def f11_rows(payload: dict | None) -> dict[str, dict]:
    """{regime: row} from an f11 result file, tolerating both `rows` shapes."""
    out: dict[str, dict] = {}
    for row in ((payload or {}).get("rows") or []):
        if isinstance(row, dict) and row.get("regime"):
            out[str(row["regime"])] = row
    return out


def _arm_value(row: dict, arm: str) -> float | None:
    """The published speedup for one arm, or None if this row does not carry it.

    Omission is the bench's own language for "not measured" (a null in a speedup column
    silently corrupts every table downstream), so a missing key is read as absent, never as
    zero.
    """
    if arm.startswith("half_fused."):
        hf = row.get("half_fused")
        if not isinstance(hf, dict):
            return None
        key = "router_speedup_vs_unfused" if arm.endswith("router") \
            else "moe_speedup_vs_unfused"
        v = hf.get(key)
        return float(v) if isinstance(v, (int, float)) else None
    blk = row.get(arm)
    if not isinstance(blk, dict):
        return None
    for key in ("paired_speedup", "speedup"):
        v = blk.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def arm_status(row: dict | None, arm: str) -> dict:
    """Per-arm verdict for one regime: measured / unmeasurable / failed / absent.

    Deliberately defensive about the result schema. The repaired bench is expected to emit
    `arms_measured` / `arms_unmeasurable`, but this driver must also be able to read a file
    written before that landed, and must never upgrade "I could not tell" into "it passed" --
    the layer stage is gated on this answer and an optimistic default would re-publish the
    campaign's defect under a fresh timestamp.
    """
    if row is None:
        return {"state": "no_row", "why": "the regime produced no row", "value": None}

    unmeas = row.get("arms_unmeasurable")
    if isinstance(unmeas, dict) and arm in unmeas:
        rec = unmeas[arm] if isinstance(unmeas[arm], dict) else {}
        why = rec.get("reason") or rec.get("why") or "recorded unmeasurable by the bench"
        return {"state": "unmeasurable", "why": str(why)[:200], "value": None}
    if isinstance(unmeas, (list, tuple)) and arm in unmeas:
        return {"state": "unmeasurable", "why": "recorded unmeasurable by the bench",
                "value": None}

    checks = row.get("checks") if isinstance(row.get("checks"), dict) else {}
    bad = [c for c in ARM_CHECKS.get(arm, ()) if isinstance(checks.get(c), dict)
           and checks[c].get("ok") is not True]
    if bad:
        detail = ", ".join(
            f"{c} rel_err={checks[c].get('rel_err')}" for c in bad)
        return {"state": "failed_check", "why": detail[:200], "value": None}

    value = _arm_value(row, arm)
    measured = row.get("arms_measured")
    if isinstance(measured, (list, tuple)) and arm not in measured and value is None:
        return {"state": "unmeasurable",
                "why": "not listed in arms_measured and no timing present", "value": None}
    if value is None:
        return {"state": "absent", "why": "no timing in the row for this arm", "value": None}

    # A value with no check behind it is reported as unverified rather than as a result: the
    # campaign published 0.0345 ms for a kernel where nothing compiled, and the only defence
    # against that is refusing to treat a number as evidence that it was validated.
    if ARM_CHECKS.get(arm) and not any(isinstance(checks.get(c), dict)
                                       for c in ARM_CHECKS[arm]):
        return {"state": "unverified", "why": "timing present but no correctness check "
                                              "recorded for it", "value": value}
    return {"state": "ok", "why": "", "value": value}


def f11_layer_gate(payload: dict | None, regimes: list[str]) -> tuple[dict, list[str]]:
    """Per-regime: may the layer re-measurement run here? Fails closed.

    Returns ({regime: {ok, why}}, [regimes that passed]).
    """
    verdict: dict[str, dict] = {}
    if payload is None:
        for name in regimes:
            verdict[name] = {"ok": False, "why": f"no {F11_RESULT}.json to read"}
        return verdict, []
    rows = f11_rows(payload)
    passed = []
    for name in regimes:
        row = rows.get(name)
        if row is None:
            verdict[name] = {"ok": False, "why": "#11 produced no row for this regime"}
            continue
        blockers = []
        for arm in LAYER_GATE_ARMS:
            st = arm_status(row, arm)
            if st["state"] != "ok":
                blockers.append(f"{arm}: {st['state']}"
                                + (f" ({st['why']})" if st["why"] else ""))
        if blockers:
            verdict[name] = {"ok": False, "why": "; ".join(blockers)[:300]}
            continue
        verdict[name] = {"ok": True, "why": "both #11 arms measured and validated"}
        passed.append(name)
    return verdict, passed


# ======================================================================================
# targeted device fence -- only the two files this driver writes
# ======================================================================================
def fence_one(log: R.Log, path: Path, device: str, enabled: bool,
              warnings: list[str], dry_run: bool = False) -> tuple[dict | None, bool]:
    """(payload, ours). Quarantine a result file recorded by a DIFFERENT device.

    `run_h200.quarantine_foreign_results` sweeps the whole results tree; that is right for a
    full campaign and wrong here, because this driver must not move f01..f10 -- they are the
    raw record and nothing in this run re-measures them. So the same rule is applied to
    exactly the two files this driver may write.
    """
    payload, err = load_json(path)
    if payload is None:
        if err != "missing":
            log(f"  !! {path.name}: {err}")
        return None, False
    got = meta_device(payload)
    if not enabled:
        return payload, True
    if not device:
        warnings.append(f"{path.name} was reused without a device check "
                        f"(the present device name is unknown)")
        return payload, True
    if got == device:
        return payload, True
    if dry_run:
        # A dry run resolves and reports; moving a file is a write like any other, and an
        # operator who asked to see the plan has not agreed to have the results tree
        # rearranged.
        log(f"  !! dry-run: {path.name} records device '{got or 'MISSING'}' but this box is "
            f"'{device}'; a real run would quarantine it and re-measure.")
        return None, False

    qdir = path.parent / f"_quarantine_foreign_{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        qdir.mkdir(parents=True, exist_ok=True)
        dst = qdir / path.name
        shutil.move(str(path), str(dst))
        msg = (f"{path.name} records device '{got or 'MISSING'}' but this box is "
               f"'{device}'; moved to {qdir.name}/ and will be re-measured")
        warnings.append(msg)
        log(f"  !! {msg}")
    except OSError as exc:
        log(f"  !! could not quarantine {path.name}: {exc}")
    return None, False


# ======================================================================================
# child supervision -- one attempt at a time, quarantining a regime that aborts the process
# ======================================================================================
_REGIME_LINE = re.compile(r"={5}\s+([A-Za-z0-9_]+)\s+\(T=")
_CKPT_LINE = re.compile(r"==\s+([A-Za-z0-9_]+)\s+==\s+\(from checkpoint\)")


def last_regime_in_log(path: Path, known: list[str]) -> str | None:
    """Which regime the bench had reached when its log stopped.

    Used only to decide which regime to quarantine after a HARD abort (a signal or an MLIR
    assertion, where no Python-level handler ran). Purely advisory: an unrecognised name is
    ignored, and a regime that already produced a row is never quarantined.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    hits = [m.group(1) for m in _REGIME_LINE.finditer(text)]
    hits += [m.group(1) for m in _CKPT_LINE.finditer(text)]
    for name in reversed(hits):
        if name in known:
            return name
    return None


def launch(log: R.Log, args: argparse.Namespace, key: str, title: str,
           script: Path, timeout_s: int, regimes: list[str], results: Path,
           logdir: Path, extra: list[str], gpu: dict, note: str = "") -> dict:
    """One supervised child, through `run_h200.run_family`.

    The heartbeat, process-group kill, per-family hwinfo snapshot and env plumbing are
    non-trivial and already correct there; duplicating them here would create a second place
    for the "orphaned bench still holding the GPU" bug to live.
    """
    fam = R.Family(key=key, title=title, script_globs=(script.name,), result_globs=(),
                   timeout_s=timeout_s, note=note)
    sub = argparse.Namespace(**vars(args))
    sub.regimes = ",".join(regimes)
    return R.run_family(log, fam, script, sub, results, logdir, extra, gpu)


def run_f11(log: R.Log, args: argparse.Namespace, script: Path, results: Path,
            logdir: Path, gpu: dict, scope: list[str], device: str,
            warnings: list[str]) -> dict:
    """Drive `bench_f11_lazy_prenorm.py` to completion over `scope`, resumable per regime.

    THE INVARIANT: every attempt is handed the full list of regimes still in scope, so the
    bench reloads the device-fenced checkpoints of the ones already done and rewrites a
    COMPLETE result file. Passing only the missing regimes would make each attempt's
    `record()` overwrite the file with just those regimes -- the same wholesale-overwrite
    hazard that makes the layer stage use a staging directory.

    THE TERMINATION ARGUMENT: an attempt either adds at least one regime's row, or (if it
    died hard) removes exactly one regime from the next attempt's scope. Both are monotone,
    so the loop is bounded by `len(scope)` regardless of how the bench fails.
    """
    R.rule(log, f"#11 -- {script.name} over {len(scope)} regime(s)")
    result_path = results / f"{F11_RESULT}.json"
    poisoned: dict[str, str] = {}
    attempts: list[dict] = []
    max_attempts = args.max_attempts or (len(scope) + 2)

    for n in range(1, max_attempts + 1):
        live = [r for r in scope if r not in poisoned]
        payload, _ = load_json(result_path)
        have = set(f11_rows(payload)) if not args.force_rerun or n > 1 else set()
        todo = [r for r in live if r not in have]
        if not live:
            log("  !! every regime in scope has been quarantined; nothing left to attempt.")
            break
        if not todo:
            log(f"  all {len(live)} regime(s) in scope have rows; #11 is done.")
            break

        log("")
        log(f"  attempt {n}/{max_attempts}: {len(todo)} regime(s) to measure "
            f"({', '.join(todo)})")
        if have:
            log(f"    {len(have)} already done and reloaded from checkpoint: "
                f"{', '.join(sorted(have))}")
        if poisoned:
            log(f"    quarantined out of this attempt: "
                + "; ".join(f"{k} ({v})" for k, v in poisoned.items()))

        extra: list[str] = list(args.f11_args)
        if args.router_only:
            extra.append("--router-only")
        rec = launch(log, args, f"f11.a{n}", "#11 lazy pre-norm", script,
                     args.timeout or args.f11_timeout, live, results, logdir, extra, gpu,
                     note="#11a needs the 12.9 GB w13 pair; --router-only skips it")
        rec["attempt"] = n
        rec["regimes_requested"] = live
        rec["regimes_to_measure"] = todo
        attempts.append(rec)

        payload, err = load_json(result_path)
        if payload is not None and args.device_fence and device:
            got = meta_device(payload)
            if got and got != device:
                warnings.append(f"{F11_RESULT}.json was just written by device '{got}', "
                                f"not '{device}' -- refusing to trust it")
                log(f"  !! the bench wrote a result recording device '{got}'; expected "
                    f"'{device}'. Stopping the #11 stage.")
                break
        now = set(f11_rows(payload))
        gained = sorted(now - have)
        rec["regimes_gained"] = gained
        if gained:
            log(f"    +{len(gained)} regime(s): {', '.join(gained)}")

        if rec.get("status") == "interrupted":
            log("  !! interrupted by the operator -- stopping the #11 stage here.")
            break
        if rec.get("status") == "ok" and not (set(live) - now):
            log("  #11 finished cleanly.")
            break
        if gained:
            continue  # progress was made; a plain relaunch will resume from checkpoints

        # No progress and a non-zero exit: the process died before it could checkpoint. The
        # regime it died on is the only thing that can make the next attempt die identically.
        died_on = last_regime_in_log(Path(rec.get("log") or ""), KNOWN_REGIMES)
        if died_on and died_on in live and died_on not in now:
            why = (f"attempt {n} exited {rec.get('returncode')} "
                   f"({rec.get('status')}) while measuring it, with no checkpoint written")
            poisoned[died_on] = why
            warnings.append(f"#11 regime {died_on} was quarantined: {why}")
            log(f"  !! QUARANTINING regime {died_on}: {why}")
            log(f"     The remaining regimes are re-attempted without it. A regime that "
                f"aborts the interpreter cannot be caught by the bench's per-regime "
                f"handler, so removing it is the only way the others can be measured.")
            continue
        log(f"  !! attempt {n} made no progress and the failing regime could not be "
            f"identified from {rec.get('log')}; stopping the #11 stage.")
        break

    payload, err = load_json(result_path)
    rows = f11_rows(payload)
    done = [r for r in scope if r in rows]
    missing = [r for r in scope if r not in rows]
    return {
        "result": str(result_path), "result_error": err or None,
        "attempts": attempts, "poisoned": poisoned,
        "regimes_done": done, "regimes_missing": missing,
        "payload_status": (payload or {}).get("status"),
        "payload_complete": (payload or {}).get("complete"),
    }


# ======================================================================================
# layer stage: re-measure O/P/Q/R into a staging tree, then merge
# ======================================================================================
def prepare_staging(log: R.Log, results: Path, staging: Path, reuse_cfgs: bool) -> None:
    """Create the staging results tree the layer re-run writes into.

    `bench_layer.py` caches its per-regime kernel mappings under `<results>/_ckpt/layer_cfgs`.
    Copying the campaign's cache in makes the re-run cheap, but it also pins the two
    lazy-pre-norm mappings (`router_prenorm`, `w13_prenorm`) to configurations chosen by the
    tuner that produced the wrong answers -- which is the one thing this whole exercise is
    trying to get away from. So reuse is OPT-IN, and the default is to re-tune. The cost is
    hours; the cost of the alternative is re-measuring the same defect and believing it.
    """
    staging.mkdir(parents=True, exist_ok=True)
    src = results / "_ckpt" / "layer_cfgs"
    dst = staging / "_ckpt" / "layer_cfgs"
    if not reuse_cfgs:
        log(f"  [cfgs] the layer re-run will TUNE ITS OWN kernel mappings in {staging.name}/."
            f" --reuse-layer-cfgs would copy the campaign's cache in, but that cache pins "
            f"router_prenorm/w13_prenorm to configurations chosen before #11 was repaired.")
        return
    if not src.is_dir():
        log(f"  [cfgs] --reuse-layer-cfgs: no cache at {src} -- tuning from scratch.")
        return
    if dst.exists():
        log(f"  [cfgs] reusing the cache already staged at {dst}")
        return
    try:
        shutil.copytree(src, dst)
        log(f"  [cfgs] --reuse-layer-cfgs: copied {len(list(dst.glob('*.json')))} cached "
            f"mapping file(s) into the staging tree. The #11 mappings in them predate the "
            f"repair; if O/P/Q/R still fail correctness, re-run WITHOUT this flag first.")
    except OSError as exc:
        log(f"  !! could not copy {src} -> {dst}: {exc}; tuning from scratch.")


def accumulate_layer(staging: Path, accum_path: Path, log: R.Log) -> dict:
    """Fold the staging `layer_configurations.json` into a per-regime accumulator.

    `bench_layer.py` rewrites its result file after every regime with only the regimes THIS
    process measured. Across several attempts that means the newest file is not the most
    complete one, so the driver keeps its own accumulator: regimes are copied in as they
    appear and never dropped, which is what makes the layer stage resumable per regime even
    though the bench has no per-regime measurement checkpoint.
    """
    fresh, err = load_json(staging / f"{LAYER_RESULT}.json")
    accum, _ = load_json(accum_path)
    accum = accum or {"id": f"{LAYER_RESULT}__f11_rerun", "regimes": {}}
    accum.setdefault("regimes", {})
    if fresh is None:
        if err != "missing":
            log(f"  !! staging {LAYER_RESULT}.json: {err}")
        return accum
    for key in ("id", "scope", "protocol", "configs", "prenorm", "env", "fairness", "_meta"):
        if key in fresh:
            accum[key] = fresh[key]
    got = []
    for name, block in (fresh.get("regimes") or {}).items():
        accum["regimes"][name] = block
        got.append(name)
    if got:
        log(f"  [accum] absorbed {len(got)} regime(s) from the staging file: "
            f"{', '.join(sorted(got))}")
    atomic_write_json(accum_path, accum)
    return accum


def run_layer(log: R.Log, args: argparse.Namespace, script: Path, results: Path,
              logdir: Path, gpu: dict, regimes: list[str], configs: list[str]) -> dict:
    """Re-measure the target configurations, isolated from the canonical result file."""
    staging = results / STAGING_DIRNAME
    accum_path = staging / "layer_rerun_accumulated.json"
    R.rule(log, f"WHOLE-LAYER re-measurement of {', '.join(TARGET_LAYER_CONFIGS)}")
    log(f"  staging results dir  {staging}")
    log(f"  configurations       {', '.join(configs)}")
    log(f"  regimes              {', '.join(regimes)}")
    log(f"  protocol             two interleaved passes x {args.rounds} rounds, LOG-11 tie "
        f"rule (a winner only when its gap to the runner-up exceeds the round-to-round "
        f"spread of both, in both passes)")
    log("")
    log("  WHY A STAGING DIRECTORY: common.record() rewrites the whole result file, and")
    log("  bench_layer writes only the configurations it was asked for. Pointing this run at")
    log(f"  {results.name}/ directly would replace {LAYER_RESULT}.json -- fourteen clean")
    log("  configurations across seven regimes -- with a file holding these few. The merge")
    log("  below copies across only the four target rows, and only when they are better.")

    prepare_staging(log, results, staging, args.reuse_layer_cfgs)
    attempts: list[dict] = []
    max_attempts = args.max_attempts or (len(regimes) + 2)
    poisoned: dict[str, str] = {}

    for n in range(1, max_attempts + 1):
        accum = accumulate_layer(staging, accum_path, log)
        have = set(accum.get("regimes") or {})
        live = [r for r in regimes if r not in poisoned]
        todo = [r for r in live if r not in have]
        if not todo:
            log(f"  every regime in scope has a layer measurement "
                f"({', '.join(sorted(have & set(regimes)))}).")
            break
        log("")
        log(f"  attempt {n}/{max_attempts}: {', '.join(todo)}")
        extra = ["--only", ",".join(configs),
                 "--rounds", str(args.rounds), "--rep", str(args.rep)]
        extra += list(args.layer_args)
        rec = launch(log, args, f"layer.a{n}", "whole-layer #11 configurations", script,
                     args.timeout or args.layer_timeout, todo, staging, logdir, extra, gpu,
                     note="writes to the staging tree; merged separately and guarded")
        rec["attempt"] = n
        rec["regimes_requested"] = todo
        attempts.append(rec)

        accum = accumulate_layer(staging, accum_path, log)
        gained = sorted(set(accum.get("regimes") or {}) - have)
        rec["regimes_gained"] = gained
        if rec.get("status") == "interrupted":
            log("  !! interrupted -- stopping the layer stage here.")
            break
        if gained:
            continue
        died_on = last_regime_in_log(Path(rec.get("log") or ""), KNOWN_REGIMES)
        if died_on and died_on in todo:
            poisoned[died_on] = (f"attempt {n} exited {rec.get('returncode')} "
                                 f"({rec.get('status')}) while measuring it")
            log(f"  !! QUARANTINING layer regime {died_on}: {poisoned[died_on]}")
            continue
        log(f"  !! attempt {n} made no progress; stopping the layer stage.")
        break

    accum = accumulate_layer(staging, accum_path, log)
    return {"staging": str(staging), "accumulated": str(accum_path),
            "payload": accum, "attempts": attempts, "poisoned": poisoned,
            "regimes_done": sorted(set(accum.get("regimes") or {}) & set(regimes))}


# ======================================================================================
# the merge -- the only place this driver writes into the campaign's record
# ======================================================================================
def _config_measured(block: dict, name: str) -> bool:
    """Did this regime block actually TIME `name` and accept it?

    All three must hold: the correctness entry says ok, and both interleaved passes carry
    round timings for it. `bench_layer` records a `correctness` entry for every configuration
    it *tried*, including the ones it then excluded, so `ok` alone is not enough -- and a
    configuration present in one pass but not the other was never subject to the LOG-11 tie
    rule at all.
    """
    ck = (block.get("correctness") or {}).get(name)
    if not isinstance(ck, dict) or ck.get("ok") is not True:
        return False
    return all(isinstance((block.get(p) or {}).get(name), dict) for p in ("pass1", "pass2"))


def merge_layer(log: R.Log, canonical_path: Path, fresh: dict, args: argparse.Namespace,
                warnings: list[str], device: str) -> dict:
    """Copy the four target rows into `layer_configurations.json`, guarded.

    THE FOUR RULES, in the order they are applied:

      1. DEVICE. The fresh payload and the canonical file must name the same device, or the
         merge is refused outright. Merging one machine's timings into another's file is the
         single mistake this whole study's fencing exists to prevent.
      2. SCOPE. Only the four target names are ever written. The anchors (`A_all_unfused`,
         `N_f11b`) are re-measured to give the fresh session a denominator and a control, and
         are deliberately NOT merged: they already measured cleanly, and replacing a clean
         campaign row with a row from a different session would silently make the file a
         mixture nobody asked for.
      3. NEVER DOWNGRADE. A fresh row that failed correctness, or that only one pass timed,
         is never written. A canonical row that already measured cleanly is left alone unless
         `--remerge-good` says otherwise. Both rules together make the merge idempotent: run
         it twice and the second run finds every target already clean and writes nothing.
      4. NEVER RESTATE THE VERDICT. `verdict` stays exactly as the campaign computed it. Its
         tie rule compares candidates timed in the SAME interleaved rounds; recomputing it
         over a mixture of two sessions would produce a winner whose margin is smaller than
         the difference between the sessions. The cross-session comparison goes under
         `f11_rerun`, anchored on `A_all_unfused` measured in both.
    """
    canonical, err = load_json(canonical_path)
    report: dict = {"canonical": str(canonical_path), "decisions": [], "written": False,
                    "backup": None, "refused": None}
    if canonical is None:
        report["refused"] = f"{canonical_path.name}: {err}"
        log(f"  !! REFUSING TO MERGE: {report['refused']}")
        log("     There is nothing to merge INTO. The re-measurement is intact under the")
        log("     staging tree and can be merged once the campaign file is restored.")
        return report

    can_dev, fresh_dev = meta_device(canonical), meta_device(fresh)
    if args.device_fence and can_dev and fresh_dev and can_dev != fresh_dev:
        report["refused"] = (f"device mismatch: {canonical_path.name} records "
                             f"'{can_dev}', the re-measurement records '{fresh_dev}'")
        warnings.append(report["refused"])
        log(f"  !! REFUSING TO MERGE: {report['refused']}")
        return report
    if args.device_fence and device and fresh_dev and fresh_dev != device:
        report["refused"] = (f"the re-measurement records device '{fresh_dev}' but this box "
                             f"is '{device}'")
        warnings.append(report["refused"])
        log(f"  !! REFUSING TO MERGE: {report['refused']}")
        return report

    merged = copy.deepcopy(canonical)
    session = digest({
        "regimes": {n: {"pass1": b.get("pass1"), "pass2": b.get("pass2"),
                        "correctness": b.get("correctness")}
                    for n, b in sorted((fresh.get("regimes") or {}).items())},
    })
    report["session"] = session

    rerun = merged.setdefault("f11_rerun", {})
    rerun.setdefault("schema", 1)
    rerun.setdefault("why", (
        "O_f11ab / P_f10_f11ab / Q_f8_f11ab / R_f1_f10_f11ab were excluded from the campaign "
        "for failing bench_layer's independent fp32 reference, which was a symptom of the "
        "same #11 defect that left f11_lazy_prenorm.json with zero rows. They were "
        "re-measured alone by run_f11_h200.py once #11 passed its correctness screen. Their "
        "rows in `regimes[*].pass1/pass2/correctness` come from THAT session; every other "
        "configuration's rows are the campaign's, untouched. `verdict` is the campaign's and "
        "was NOT recomputed -- see `sessions[*].regimes[*].session_verdict` for the tie "
        "analysis over the re-measured set, and `anchor_drift` for how far the two sessions' "
        "A_all_unfused differ, which bounds how comparable they are."))
    sessions = rerun.setdefault("sessions", [])
    already = next((s for s in sessions if s.get("session") == session), None)
    if already is not None:
        log(f"  [merge] session {session} is already recorded in "
            f"{canonical_path.name} -- this merge is a no-op unless a target row changed.")

    per_regime: dict[str, dict] = {}
    n_written = 0
    for reg_name, fresh_block in sorted((fresh.get("regimes") or {}).items()):
        can_block = (merged.get("regimes") or {}).get(reg_name)
        fresh_med = ((fresh_block.get("verdict") or {}).get("median_of_passes") or {})
        anchor_fresh = fresh_med.get("A_all_unfused")
        info = {
            "session_verdict": fresh_block.get("verdict"),
            "session_median_of_passes": fresh_med,
            "session_correctness": fresh_block.get("correctness"),
            "session_paired_head_vs_unfused": fresh_block.get("paired_head_vs_unfused"),
            "session_shared_expert_ms": fresh_block.get("shared_expert_ms"),
            "anchor": "A_all_unfused",
            "anchor_ms_this_session": anchor_fresh,
            "speedup_vs_unfused_this_session": {},
            "decisions": {},
        }
        if can_block is None:
            # Nothing to merge into, and synthesising a regime block that holds five
            # configurations where every other regime holds eighteen would read as a full
            # measurement to anything that walks this file. Recorded, not fabricated.
            info["decisions"]["*"] = {"state": "regime_absent_from_canonical", "why": (
                f"regime {reg_name} is not in {canonical_path.name}; a partial regime block "
                f"would be indistinguishable from a complete one downstream")}
            per_regime[reg_name] = info
            report["decisions"].append({"regime": reg_name, "config": "*",
                                        "action": "skip", "why": info["decisions"]["*"]["why"]})
            log(f"  {reg_name:<15} SKIP -- regime absent from the canonical file")
            continue

        can_med = ((can_block.get("verdict") or {}).get("median_of_passes") or {})
        anchor_can = can_med.get("A_all_unfused")
        if isinstance(anchor_fresh, (int, float)) and isinstance(anchor_can, (int, float)) \
                and anchor_can:
            info["anchor_ms_campaign"] = anchor_can
            info["anchor_drift_pct"] = (anchor_fresh - anchor_can) / anchor_can * 100.0
        for name in TARGET_LAYER_CONFIGS:
            if isinstance(anchor_fresh, (int, float)) and fresh_med.get(name):
                info["speedup_vs_unfused_this_session"][name] = \
                    anchor_fresh / fresh_med[name]

        for name in TARGET_LAYER_CONFIGS:
            fresh_ok = _config_measured(fresh_block, name)
            can_ok = _config_measured(can_block, name)
            fck = (fresh_block.get("correctness") or {}).get(name) or {}
            if not fresh_ok:
                why = (f"the re-measurement did not produce a clean timed row "
                       f"(correctness ok={fck.get('ok')}, rel_err={fck.get('rel_err')}"
                       + (f", error={fck.get('error')}" if fck.get("error") else "") + ")")
                action = "skip_fresh_not_clean"
            elif can_ok and not args.remerge_good:
                why = ("the canonical file already has a clean measurement of this "
                       "configuration; --remerge-good would replace it")
                action = "skip_already_clean"
            else:
                action = "write"
                why = ("replaced a failed/absent row" if not can_ok
                       else "--remerge-good: replaced a clean row")
            report["decisions"].append({"regime": reg_name, "config": name,
                                        "action": action, "why": why})
            mark = {"write": "WRITE", "skip_already_clean": "keep",
                    "skip_fresh_not_clean": "SKIP"}[action]
            log(f"  {reg_name:<15} {name:<16} {mark:<6} {why}")
            if action != "write":
                continue

            can_block.setdefault("correctness", {})[name] = dict(
                fresh_block["correctness"][name],
                _source={"f11_rerun_session": session})
            for p in ("pass1", "pass2"):
                can_block.setdefault(p, {})[name] = fresh_block[p][name]
            n_written += 1

        # What lands IN THE FILE is the resulting STATE of each row, not what this
        # invocation happened to do to it. The action ("write" the first time, "keep" the
        # second) is a property of when the merge ran; the state ("this row came from the
        # re-measurement") is a property of the record, and only the second one can be
        # written down without making a re-run of the merge change the file. The actions are
        # still reported -- to the console and to f11_rerun_summary.json -- where they belong.
        for name in TARGET_LAYER_CONFIGS:
            fck = (fresh_block.get("correctness") or {}).get(name) or {}
            src = ((can_block.get("correctness") or {}).get(name) or {}).get("_source") or {}
            if src.get("f11_rerun_session") == session:
                state = "merged_from_this_session"
            elif _config_measured(can_block, name):
                state = "canonical_row_kept"
            elif not _config_measured(fresh_block, name):
                state = "not_merged_re_measurement_failed"
            else:
                state = "not_merged"
            info["decisions"][name] = {
                "state": state,
                "fresh_ok": bool(fck.get("ok")),
                "fresh_rel_err": fck.get("rel_err"),
                "fresh_error": fck.get("error"),
            }

        can_block["f11_rerun_note"] = (
            f"{', '.join(TARGET_LAYER_CONFIGS)} were re-measured by run_f11_h200.py "
            f"(session {session}); rows carrying `_source.f11_rerun_session` come from that "
            f"session and are NOT part of `verdict`, which the campaign computed over the "
            f"configurations timed in ITS interleaved rounds. The tie analysis over the "
            f"re-measured set is in the top-level `f11_rerun` block.")
        per_regime[reg_name] = info

    # What this session CONTRIBUTED is read back out of the merged file rather than counted
    # as this invocation went along. The two differ the moment the merge is re-run: the
    # second invocation writes nothing because the rows are already there, and an
    # invocation counter would record 0 and rewrite the provenance -- so the file would
    # change on every run while claiming nothing had. A count derived from the file's own
    # state is the same number no matter how many times the merge is invoked, which is what
    # makes the whole operation idempotent.
    owned = sorted(
        (reg, name)
        for reg, blk in (merged.get("regimes") or {}).items()
        for name in TARGET_LAYER_CONFIGS
        if ((blk.get("correctness") or {}).get(name) or {}).get("_source", {})
        .get("f11_rerun_session") == session
    )

    # Built WITHOUT its timestamp first and compared field by field against any entry this
    # session already wrote: re-stamping `merged_at` on an otherwise identical record would
    # make every re-invocation modify the campaign's file and take another backup.
    entry = {
        "session": session,
        "driver": "run_f11_h200.py",
        "device": fresh_dev or None,
        "targets": list(TARGET_LAYER_CONFIGS),
        "anchors_measured_not_merged": [c for c in args.layer_anchors],
        "protocol": (fresh.get("protocol") or {}),
        "staging_payload_meta": (fresh.get("_meta") or {}),
        "rows_from_this_session": [f"{r}/{n}" for r, n in owned],
        "n_rows_from_this_session": len(owned),
        "regimes": per_regime,
    }
    if already is not None and {k: v for k, v in already.items()
                                if k not in ("merged_at", "argv")} == entry:
        pass  # identical provenance already on record; leave it exactly as it is
    else:
        entry["merged_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        entry["argv"] = list(sys.argv)
        if already is not None:
            sessions[sessions.index(already)] = entry
        else:
            sessions.append(entry)

    amendments = merged.setdefault("_meta", {}).setdefault("amendments", [])
    prior = next((a for a in amendments if a.get("session") == session), None)
    note = ("re-measured #11 whole-layer configurations merged in; every other "
            "configuration is the campaign's, untouched")
    if prior is None or prior.get("rows") != len(owned) or prior.get("note") != note:
        stamp = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "by": "run_f11_h200.py",
                 "session": session, "rows": len(owned), "note": note}
        if prior is None:
            amendments.append(stamp)
        else:
            amendments[amendments.index(prior)] = stamp

    # Compared as SERIALISED JSON, not as Python objects. `layer_configurations.json`
    # contains NaN (`pair_meta.noflush_ratio_p50` in four regimes), and `nan != nan`, so an
    # object comparison reports a difference in a file nothing touched -- which would take a
    # fresh backup and rewrite the campaign's record on every single invocation. The
    # serialised form is what is actually being written, so it is the right thing to compare.
    if json.dumps(merged, sort_keys=True, default=str) == \
            json.dumps(canonical, sort_keys=True, default=str):
        log("  [merge] nothing changed -- the canonical file is already up to date.")
        report.update(written=False, n_rows=0, per_regime=per_regime)
        return report
    if args.dry_run:
        log(f"  [merge] --dry-run: {n_written} row(s) WOULD be written to "
            f"{canonical_path.name}; nothing was modified.")
        report.update(written=False, n_rows=n_written, per_regime=per_regime,
                      dry_run=True)
        return report

    backup = canonical_path.with_name(
        f"{canonical_path.stem}.pre_f11_merge_{time.strftime('%Y%m%d_%H%M%S')}.json")
    try:
        shutil.copy2(canonical_path, backup)
        report["backup"] = str(backup)
        log(f"  [merge] backup of the pre-merge file -> {backup.name}")
    except OSError as exc:
        log(f"  !! could not write a backup ({exc}); REFUSING to modify the campaign file.")
        report["refused"] = f"backup failed: {exc}"
        return report

    atomic_write_json(canonical_path, merged)
    log(f"  [merge] wrote {n_written} row(s) into {canonical_path.name} "
        f"(session {session}).")
    report.update(written=True, n_rows=n_written, per_regime=per_regime)
    return report


# ======================================================================================
# final status table
# ======================================================================================
def _cell(text: str, width: int = 8) -> str:
    return f"{text:>{width}}"


def status_table(f11_payload: dict | None, layer_report: dict | None,
                 gate: dict, scope: list[str]) -> list[str]:
    """Per-arm x per-regime, with the reason in the cell when there is no number.

    A blank where a number should be is what let the campaign's loss go unnoticed for a
    whole run, so every cell says something: a speedup, or WHY there is no speedup.
    """
    abbr = [REGIME_ABBR.get(r, r) for r in scope]
    head = f"  {'arm':<40}" + "".join(_cell(a) for a in abbr)
    lines = [head, "  " + "-" * (len(head) - 2)]
    rows = f11_rows(f11_payload)

    legend_needed = set()
    for _key, arm, _u, label in F11_ARMS:
        cells = []
        for reg in scope:
            st = arm_status(rows.get(reg), arm)
            if st["state"] == "ok":
                cells.append(_cell(f"{st['value']:.3f}"))
            else:
                short = {"no_row": "no-row", "unmeasurable": "UNMEAS",
                         "failed_check": "WRONG", "absent": "-",
                         "unverified": "unver"}[st["state"]]
                cells.append(_cell(short))
                legend_needed.add(st["state"])
        lines.append(f"  {label:<40}" + "".join(cells))

    lines.append("")
    lines.append(f"  {'whole-layer configuration (merge state)':<40}"
                 + "".join(_cell(a) for a in abbr))
    lines.append("  " + "-" * (len(head) - 2))
    per_regime = (layer_report or {}).get("per_regime") or {}
    for name in TARGET_LAYER_CONFIGS:
        cells = []
        for reg in scope:
            info = per_regime.get(reg)
            if info is None:
                cells.append(_cell("not-run" if gate.get(reg, {}).get("ok") else "gated"))
                continue
            dec = (info.get("decisions") or {}).get(name) \
                or (info.get("decisions") or {}).get("*") or {}
            state = dec.get("state")
            spd = (info.get("speedup_vs_unfused_this_session") or {}).get(name)
            if state == "merged_from_this_session":
                cells.append(_cell(f"{spd:.3f}" if isinstance(spd, (int, float)) else "ok"))
            elif state == "canonical_row_kept":
                cells.append(_cell("kept"))
            elif state == "not_merged_re_measurement_failed":
                cells.append(_cell("WRONG"))
            else:
                cells.append(_cell("skip"))
        lines.append(f"  {name:<40}" + "".join(cells))

    lines.append("")
    lines.append("  #11 arms: the value is the paired fused/unfused speedup where the bench "
                 "recorded one.")
    if "unmeasurable" in legend_needed:
        lines.append("  UNMEAS  = the bench recorded this arm as unmeasurable (a screen "
                     "rejected every config, or a prerequisite kernel did); NO timing is "
                     "published for it, by design.")
    if "failed_check" in legend_needed:
        lines.append("  WRONG   = the arm was timed but failed its post-tuning correctness "
                     "check; the number is deliberately not shown.")
    if "unverified" in legend_needed:
        lines.append("  unver   = a timing exists with no correctness check recorded behind "
                     "it. Treat as unmeasured until re-run.")
    if "no_row" in legend_needed:
        lines.append("  no-row  = the regime produced no row at all (see the attempt log).")
    if "absent" in legend_needed:
        lines.append("  -       = this arm was out of scope for the run (e.g. --router-only "
                     "omits every #11a key rather than emitting null).")
    lines.append("  layer:  a number is that configuration's median vs A_all_unfused MEASURED "
                 "IN THE SAME SESSION; `kept` = the canonical file already had a clean row "
                 "and it was left alone; `gated` = #11 did not pass its correctness screen "
                 "in that regime, so nothing was re-measured.")
    return lines


# ======================================================================================
def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="run_f11_h200.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Re-run fusion #11 (lazy pre-norm) alone on the H200, then repair the "
                    "four whole-layer configurations that contain it.",
        epilog="Typical use:\n"
               "  python3 run_f11_h200.py --gpu auto          # the whole repair\n"
               "  python3 run_f11_h200.py --gpu 3 --quick     # stack smoke test\n"
               "  python3 run_f11_h200.py --gpu auto --skip-layer\n"
               "  python3 run_f11_h200.py --merge-only        # merge a finished staging run\n"
               "  python3 run_f11_h200.py --list              # show the plan, run nothing\n"
               "\nOn a shared multi-GPU node ALWAYS pass --gpu: these numbers are only\n"
               "comparable with the campaign's if they came from the same idle card.\n",
    )
    ap.add_argument("--gpu", default=None, metavar="N|auto",
                    help="which GPU this run uses. N pins nvidia-smi index N (exported as "
                         "CUDA_VISIBLE_DEVICES=N plus CUDA_DEVICE_ORDER=PCI_BUS_ID, so "
                         "every child sees exactly that card as cuda:0); 'auto' ranks the "
                         "host's GPUs and takes the idlest, printing the ranking. Same "
                         "selection and same refusal-on-tenanted-card as run_h200.py.")
    ap.add_argument("--allow-busy", action="store_true",
                    help="measure on a card that already has another tenant. This is what "
                         "produced the preflight's impossible 40.55 us harness floor; "
                         "short-kernel results from such a run are not defensible.")
    ap.add_argument("--regimes", default="",
                    help=f"comma-separated subset of {','.join(KNOWN_REGIMES)}")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--log-dir", default=str(DEFAULT_LOGDIR))
    ap.add_argument("--quick", action="store_true",
                    help="short sweeps everywhere. A stack smoke test, NOT a publishable "
                         "measurement -- and the merge refuses to run under it unless "
                         "--merge-quick is also given.")
    ap.add_argument("--merge-quick", action="store_true",
                    help="allow a --quick re-measurement to be merged into the campaign "
                         "file. Almost never what you want.")
    ap.add_argument("--force", action="store_true",
                    help="run even if the device is not sm_90 (results are NOT an H200 "
                         "measurement; use --results-dir to keep them out of results/h200)")
    ap.add_argument("--force-rerun", action="store_true",
                    help="ignore existing #11 rows and per-regime checkpoints and "
                         "re-measure from scratch")
    ap.add_argument("--router-only", action="store_true",
                    help="#11b only: skip #11a's w13 GEMM and its 2 x 12.9 GB of weights. "
                         "Implies --skip-layer, because O/P/Q/R all fuse BOTH GEMMs.")
    ap.add_argument("--skip-layer", action="store_true",
                    help="stop after #11; do not re-measure the whole-layer configurations")
    ap.add_argument("--layer-only", action="store_true",
                    help="skip the #11 bench and go straight to the layer stage (the gate "
                         "still reads the existing f11 result)")
    ap.add_argument("--merge-only", action="store_true",
                    help="run nothing; merge an already-completed staging re-measurement "
                         "into layer_configurations.json")
    ap.add_argument("--force-layer", action="store_true",
                    help="re-measure the layer configurations even where #11 did not pass "
                         "its correctness screen. bench_layer still validates every "
                         "configuration against its own fp32 reference, so this cannot "
                         "publish a wrong row -- it can only waste hours.")
    ap.add_argument("--remerge-good", action="store_true",
                    help="allow the merge to replace a target row that ALREADY measured "
                         "cleanly. Off by default: the merge exists to fill gaps, not to "
                         "restate rows the campaign got right.")
    ap.add_argument("--reuse-layer-cfgs", action="store_true",
                    help="copy the campaign's cached per-regime layer kernel mappings into "
                         "the staging tree instead of re-tuning. Saves hours, but those "
                         "mappings pin router_prenorm/w13_prenorm to configurations chosen "
                         "before #11 was repaired.")
    ap.add_argument("--layer-anchors", default=",".join(DEFAULT_LAYER_ANCHORS),
                    help="configurations timed ALONGSIDE the targets to give the re-run its "
                         "own denominator and control. Measured, never merged. "
                         f"(default: {','.join(DEFAULT_LAYER_ANCHORS)})")
    ap.add_argument("--rounds", type=int, default=8,
                    help="interleaved rounds per pass in the layer bench (LOG-11 used 8)")
    ap.add_argument("--rep", type=int, default=15,
                    help="reps per candidate per round in the layer bench")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="cap on relaunches per stage (default: regimes + 2)")
    ap.add_argument("--timeout", type=int, default=0,
                    help="per-attempt timeout in seconds, overriding both stage defaults")
    ap.add_argument("--f11-timeout", type=int, default=10 * 3600)
    ap.add_argument("--layer-timeout", type=int, default=16 * 3600)
    ap.add_argument("--heartbeat", type=int, default=60,
                    help="seconds between progress lines (default 60)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter used for the benches (default: this one)")
    ap.add_argument("--f11-args", action="append", default=[], metavar="ARG",
                    help="extra CLI argument passed through to bench_f11_lazy_prenorm.py")
    ap.add_argument("--layer-args", action="append", default=[], metavar="ARG",
                    help="extra CLI argument passed through to bench_layer.py")
    ap.add_argument("--disable-features", default="",
                    help="comma list of Hopper paths to switch off for every child "
                         "(tma,clusters,ws,wgmma,all). Every H200-only path in this suite "
                         "keeps a runtime-detected classic fallback; this forces it in BOTH "
                         "arms, so the ratios stay fair but stop being a Hopper measurement.")
    ap.add_argument("--flush-mb", type=int, default=0,
                    help="override the L2-flush buffer size in MiB (diagnosis only)")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="do not run the preflight probe even if its JSON is missing")
    ap.add_argument("--no-device-fence", action="store_true",
                    help="DANGEROUS: reuse and merge results without checking which GPU "
                         "produced them")
    ap.add_argument("--allow-concurrent", action="store_true",
                    help="do not refuse when another bench appears to be running")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve everything, print the plan and the merge decisions, but "
                         "launch nothing and write nothing")
    args = ap.parse_args(argv)
    args.device_fence = not args.no_device_fence
    args.layer_anchors = [s.strip() for s in args.layer_anchors.split(",") if s.strip()]
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

    scope = [r.strip() for r in args.regimes.split(",") if r.strip()] or list(KNOWN_REGIMES)
    bad = [r for r in scope if r not in KNOWN_REGIMES]
    if bad:
        warnings.append(f"regimes not in the canonical set were requested: {bad}")
        log(f"!! regimes {bad} are not in {KNOWN_REGIMES}; passing them through anyway")

    f11_script = R.find_script(R.FAMILY_BY_KEY["f11"])
    layer_script = R.find_script(R.FAMILY_BY_KEY["layer"])
    layer_configs = list(dict.fromkeys(list(args.layer_anchors) +
                                       list(TARGET_LAYER_CONFIGS)))

    if args.router_only and not args.skip_layer:
        args.skip_layer = True
        log("  note: --router-only implies --skip-layer -- O/P/Q/R all use prenorm='all', "
            "which fuses BOTH the router and the w13 GEMM, so there is nothing to "
            "re-measure without #11a.")

    if args.list:
        R.rule(log, "PLAN")
        log(f"  stage 1  #11    {f11_script or 'SCRIPT NOT FOUND'}")
        log(f"           regimes {', '.join(scope)}")
        log(f"           writes  {results / (F11_RESULT + '.json')}")
        log(f"  stage 2  layer  {layer_script or 'SCRIPT NOT FOUND'}"
            + ("   (SKIPPED)" if args.skip_layer else ""))
        log(f"           configs {', '.join(layer_configs)}")
        log(f"                   (merged: {', '.join(TARGET_LAYER_CONFIGS)};"
            f" measured but never merged: {', '.join(args.layer_anchors)})")
        log(f"           staging {results / STAGING_DIRNAME}")
        log(f"           merges  {results / (LAYER_RESULT + '.json')}")
        log(f"  logs            {logdir}")
        log.close()
        return 0

    # --- GPU selection, preflight, device gate: run_h200's, unchanged ------------------
    hw_start = R.hwinfo()
    gpu = R.resolve_gpu(log, args, hw_start, warnings,
                        measuring=not (args.dry_run or args.merge_only))
    if gpu.get("refuse"):
        log("")
        log("  (the same command for this driver: "
            f"python3 run_f11_h200.py --gpu <idle index>)")
        log.close()
        return 4
    hw_row = R.pick_hw_row(hw_start, gpu.get("index"))

    pf = R.read_preflight()
    if pf is None and not args.skip_preflight and not args.merge_only:
        log(f"  no {R.PREFLIGHT_JSON.name}; running the preflight probe first (it is what "
            f"every bench reads its hardware constants from).")
        pf = R.run_preflight(log, args.python, logdir, quick=args.quick,
                             gpu=gpu.get("index"))
    elif pf is not None and hw_row:
        pf_name = R._norm_dev((pf.get("device") or {}).get("name"))
        pf_uuid = R._norm_uuid((pf.get("gpu_selection") or {}).get("uuid")
                               or (pf.get("device") or {}).get("uuid"))
        row_uuid = R._norm_uuid(hw_row.get("uuid"))
        why = ""
        if pf_name != R._norm_dev(hw_row.get("name")):
            why = f"model differs ({pf_name!r} vs {R._norm_dev(hw_row.get('name'))!r})"
        elif gpu.get("index") is not None and pf_uuid and row_uuid and pf_uuid != row_uuid:
            why = f"same model but a different card (probe {pf_uuid}, pinned {row_uuid})"
        if why and not args.merge_only:
            log(f"!! cached preflight does not describe the pinned GPU: {why} -- re-probing "
                f"before anything is tuned.")
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
        log("!! (for this driver the escape hatch is: "
            "python3 run_f11_h200.py --force --results-dir results/<thisdevice>)")
        log.close()
        raise
    device_name = dev.get("name") or ""

    if args.quick and not args.merge_quick:
        log("  [quick] short sweeps. The layer merge is DISABLED under --quick: a quick "
            "sweep is a stack smoke test and must not overwrite a campaign row. Pass "
            "--merge-quick if you really mean to.")
    if args.disable_features:
        msg = (f"Hopper paths disabled for this run: {args.disable_features}. Both arms of "
               f"every pair fall back to the classic path, so the ratios stay fair -- but "
               f"they stop being a measurement of the Hopper path, which is the entire "
               f"point of testing #11 on this device.")
        warnings.append(msg)
        log(f"  [features] {msg}")

    if not (args.dry_run or args.merge_only):
        busy = R.another_bench_running()
        if busy and not args.allow_concurrent:
            log("!! another glm52_h200 bench appears to be running:")
            for b in busy:
                log(f"     {b}")
            log("!! Two benchmarks on one GPU corrupt every timing in both. Refusing.")
            log.close()
            return 3
        if busy:
            warnings.append("--allow-concurrent: another bench process was detected")

    # Only the two files this driver may write are fenced -- f01..f10 are the raw record and
    # nothing here re-measures them, so nothing here may move them.
    f11_path = results / f"{F11_RESULT}.json"
    layer_path = results / f"{LAYER_RESULT}.json"
    fence_one(log, f11_path, device_name, args.device_fence, warnings,
              dry_run=args.dry_run)

    # ---------------------------------------------------------------- stage 1: #11 ----
    f11_stage: dict = {}
    if args.merge_only or args.layer_only:
        log("")
        log(f"  stage 1 skipped ({'--merge-only' if args.merge_only else '--layer-only'}); "
            f"reading the existing {F11_RESULT}.json for the layer gate.")
    elif f11_script is None:
        warnings.append("bench_f11_lazy_prenorm.py not found; #11 could not be re-run")
        log(f"!! no #11 bench found under {R.SUITE / 'bench'} -- stage 1 skipped.")
    elif args.dry_run:
        log(f"  dry-run: would run {f11_script} over {', '.join(scope)}")
    else:
        f11_stage = run_f11(log, args, f11_script, results, logdir, gpu, scope,
                            device_name, warnings)
        R.check_tenants(log, gpu, "after the #11 stage", warnings)
        if f11_stage.get("regimes_missing"):
            warnings.append("#11 produced no row for: "
                            + ", ".join(f11_stage["regimes_missing"]))

    f11_payload, f11_err = load_json(f11_path)
    gate, gate_pass = f11_layer_gate(f11_payload, scope)

    R.rule(log, "#11 CORRECTNESS GATE for the whole-layer re-measurement")
    log("  O/P/Q/R fuse the normalization into BOTH the router GEMM and the w13 GEMM, so")
    log("  both #11 arms must be measured AND validated before re-timing them is worth an")
    log("  hour of this machine. The gate fails closed: an answer this driver cannot prove")
    log("  from the result file counts as 'did not pass'.")
    for name in scope:
        v = gate.get(name, {})
        log(f"    {name:<16} {'PASS' if v.get('ok') else 'no  '}  {v.get('why', '')[:110]}")

    # -------------------------------------------------------------- stage 2: layer ----
    layer_stage: dict = {}
    merge_report: dict = {}
    layer_regimes = list(gate_pass)
    if args.force_layer:
        layer_regimes = [r for r in scope if r in scope]
        if set(layer_regimes) - set(gate_pass):
            warnings.append("--force-layer: the layer configurations were re-measured in "
                            "regimes where #11 did not pass its correctness screen")

    if args.skip_layer:
        log("")
        log("  stage 2 skipped (--skip-layer).")
    elif layer_script is None:
        warnings.append("bench_layer.py not found; the whole-layer configurations were "
                        "not re-measured")
        log(f"!! no layer bench found under {R.SUITE / 'bench'} -- stage 2 skipped.")
    elif not layer_regimes:
        log("")
        log("  stage 2 NOT RUN: #11 passed its correctness screen in no regime in scope.")
        log("  Nothing is re-measured and nothing is merged. That gap is the honest state")
        log("  of the record; --force-layer overrides it if you want the layer bench's own")
        log("  fp32 reference to be the only judge.")
    elif args.merge_only:
        log("")
        log("  stage 2 measurement skipped (--merge-only); merging the staged run.")
        accum, err = load_json(results / STAGING_DIRNAME / "layer_rerun_accumulated.json")
        if accum is None:
            log(f"  !! no staged re-measurement to merge: {err}")
        layer_stage = {"payload": accum or {}, "attempts": [], "poisoned": {},
                       "regimes_done": sorted((accum or {}).get("regimes") or {})}
    elif args.dry_run:
        log(f"  dry-run: would run {layer_script} for "
            f"{', '.join(layer_configs)} over {', '.join(layer_regimes)}")
    else:
        layer_stage = run_layer(log, args, layer_script, results, logdir, gpu,
                                layer_regimes, layer_configs)
        R.check_tenants(log, gpu, "after the layer stage", warnings)

    fresh = (layer_stage or {}).get("payload") or {}
    if fresh.get("regimes"):
        if args.quick and not args.merge_quick:
            log("")
            log("  [merge] REFUSED: this was a --quick run. Its rounds and reps are cut for "
                "a smoke test and its numbers must not replace a campaign row. The "
                "re-measurement is intact under the staging tree; --merge-quick forces it.")
            merge_report = {"refused": "--quick without --merge-quick"}
        else:
            R.rule(log, f"MERGE into {layer_path.name}")
            merge_report = merge_layer(log, layer_path, fresh, args, warnings, device_name)

    # ------------------------------------------------------------------- report ------
    hw_end = R.hwinfo()
    drift = R.hw_drift(hw_start, hw_end)
    R.check_tenants(log, gpu, "at the end of the run", warnings)

    f11_payload, _ = load_json(f11_path)
    R.rule(log, "STATUS -- per arm x per regime")
    table = status_table(f11_payload, merge_report, gate, scope)
    for ln in table:
        log(ln)

    log("")
    R.rule(log, "STAGE STATUS")
    for tag, stage in (("f11", f11_stage), ("layer", layer_stage)):
        if not stage:
            log(f"  {tag:<7} not run")
            continue
        for rec in stage.get("attempts", []):
            log(f"  {tag:<7} attempt {rec.get('attempt')}  {rec.get('status', '?'):<16} "
                f"{(rec.get('wall_s') or 0) / 60:7.1f} min  "
                f"+{len(rec.get('regimes_gained') or [])} regime(s)  {rec.get('log', '')}")
        for reg, why in (stage.get("poisoned") or {}).items():
            log(f"  {tag:<7} QUARANTINED {reg}: {why}")
        log(f"  {tag:<7} done: {', '.join(stage.get('regimes_done') or []) or '(none)'}")

    if merge_report:
        log("")
        if merge_report.get("refused"):
            log(f"  merge REFUSED: {merge_report['refused']}")
        else:
            log(f"  merge: {merge_report.get('n_rows', 0)} row(s) written"
                + (f", backup {Path(merge_report['backup']).name}"
                   if merge_report.get("backup") else "")
                + (" (dry run)" if merge_report.get("dry_run") else ""))

    log("")
    R.rule(log, "HWINFO -- START vs END")
    for d in drift:
        sm, tmp = d.get("clocks.sm", {}), d.get("temperature.gpu", {})
        pct = f"{sm['pct']:+.1f}%" if sm.get("pct") is not None else "n/a"
        dt = f"{tmp['delta']:+.0f} C" if tmp.get("delta") is not None else "n/a"
        log(f"  GPU{d['index']} SM clock {pct}, temperature {dt}, "
            f"throttle {d.get('throttle_start')} -> {d.get('throttle_end')}")
    if not hw_start and not hw_end:
        log("  nvidia-smi was unavailable -- no drift record exists for this run.")

    summary = {
        "schema": 1,
        "id": "glm52_h200_f11_rerun_summary",
        "_meta": {
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device_name,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            "gpu_index": gpu.get("index"), "gpu_uuid": gpu.get("uuid"),
            "gpu_pinned": bool(gpu.get("pinned")),
            "gpu_was_idle": (None if gpu.get("busy") is None else not gpu["busy"]),
        },
        "driver": {
            "file": "run_f11_h200.py", "argv": sys.argv, "repo": str(REPO),
            "results_dir": str(results), "log_dir": str(logdir),
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_s": time.time() - t_start,
            "regimes": scope, "quick": bool(args.quick),
            "router_only": bool(args.router_only),
            "device_fence": bool(args.device_fence),
            "reuse_layer_cfgs": bool(args.reuse_layer_cfgs),
            "layer_targets": list(TARGET_LAYER_CONFIGS),
            "layer_anchors_measured_not_merged": list(args.layer_anchors),
        },
        "gpu": gpu, "device": dev,
        "preflight": (R.preflight_digest(pf) | {"path": str(R.PREFLIGHT_JSON)}) if pf
                     else {"path": str(R.PREFLIGHT_JSON), "present": False},
        "f11_stage": f11_stage,
        "f11_layer_gate": gate,
        "layer_stage": {k: v for k, v in (layer_stage or {}).items() if k != "payload"},
        "merge": {k: v for k, v in (merge_report or {}).items() if k != "per_regime"},
        "merge_per_regime": (merge_report or {}).get("per_regime"),
        "arms": {key: label for key, _a, _u, label in F11_ARMS},
        "table": table,
        "hwinfo_start": hw_start, "hwinfo_end": hw_end, "hwinfo_drift": drift,
        "warnings": warnings,
        "what_is_measured": {
            "f11": "three arms of lazy pre-norm -- #11a into the w13 grouped MoE GEMM, "
                   "#11b into the router GEMM, #11b' half-fused (rstd from a separate "
                   "reduction, applied as a pure epilogue scale) -- each arm independently "
                   "autotuned from the same kernel source and screened against an fp32 "
                   "reference before any timing.",
            "unmeasurable": "an arm whose numerical screen rejected every configuration "
                            "publishes NO timing and is recorded as unmeasurable with the "
                            "reason. The previous campaign printed 0.0345 ms for a kernel "
                            "where nothing compiled; that is the defect this rule closes.",
            "layer": "O/P/Q/R re-measured with the existing two-pass interleaved protocol "
                     "and the LOG-11 tie rule, alongside A_all_unfused as an in-session "
                     "denominator. Only the four target rows are merged; `verdict` in the "
                     "campaign file is NOT recomputed across sessions.",
        },
    }
    out = results / "f11_rerun_summary.json"
    if not args.dry_run:
        try:
            atomic_write_json(out, summary)
            log("")
            log(f"  wrote {out}")
        except OSError as exc:
            log(f"!! could not write {out}: {exc}")

    if warnings:
        log("")
        R.rule(log, "WARNINGS (also in f11_rerun_summary.json)")
        for w in warnings:
            log(f"  - {w}")

    log("")
    R.rule(log, "SEND BACK")
    log(f"  1. {f11_path}")
    log(f"  2. {layer_path}   (+ any .pre_f11_merge_*.json backup beside it)")
    log(f"  3. {results / STAGING_DIRNAME}   (the un-merged re-measurement)")
    log(f"  4. {out}")
    log(f"  5. {logdir}")
    log(f"  total wall {(time.time() - t_start) / 3600:.2f} h")

    rows = f11_rows(f11_payload)
    incomplete = [r for r in scope if r not in rows]
    log.close()
    return 0 if not incomplete else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n interrupted", flush=True)
        raise SystemExit(130)
