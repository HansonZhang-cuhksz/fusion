#!/usr/bin/env python3
"""The Hopper CONTROL ARM: re-measure the H200 suite with every sm_90 lever forced off.

    setsid nohup python3 run_control_h200.py > control.out 2>&1 &

RUN IT DETACHED. The command above is the recommended form, not decoration: the 2026-08-11
run died 10 minutes into `f04f05` having written no traceback, no exit line and no summary
update -- the last line in driver.log is a routine heartbeat. That is the signature of a
signal, and a dropped SSH session (SIGHUP) is the cheapest explanation and the only one the
operator can eliminate for free. `setsid` detaches from the terminal so a lost connection
cannot reach the process; `nohup` is belt and braces. SIGTERM/SIGHUP/SIGINT are now trapped
and recorded (see `_install_signal_handlers`), so a kill leaves a diagnosis instead of a
silence -- but SIGKILL and the host OOM killer cannot be trapped by anyone, which is why the
summary is saved after every family and why `verified` defaults to False.

    python3 run_control_h200.py           # equivalent, attached; fine for --list/--dry-run

WHY THIS FILE EXISTS SEPARATELY FROM `run_h200.py`.

`glm52_h200/kernels/hopper.py` gates three sm_90-only levers behind runtime capability
detection -- TMA tensor descriptors, warp specialization via `tl.range(warp_specialize=True)`
and thread-block clusters via `num_ctas`. Comparing the committed H200 campaign against the
C500 campaign produced a result that reads backwards: the THREE families that are OFFERED
TMA (`f01`, `f06`, `f08f09`) improved the LEAST on the H200 -- medians 0.98-1.06x across
their six variants -- while `f03` (ResAdd+RMSNorm) and `f10` (ExpertMerge+ResAdd), which
advertise no Hopper cfg key at all and therefore cannot use any of the three, improved 1.86x
and 1.48x. Hopper-feature usage does not predict Hopper benefit.

The per-family axis surface, read out of the committed `results/h200/*.json`
(`fairness.h200_axes.per_family.*.axes.<axis>.offered`) rather than assumed:

    f01, f06, f08f09   tma, warp_specialize, clusters
    f04f05             warp_specialize, clusters ONLY -- `tma.offered` is false, with
                       `not_offered_because = "this kernel module advertises no cfg key
                       for it"`, and `kernel_cfg_keys = ["warp_specialize", "num_ctas"]`
    f11                warp_specialize only, on its two GEMM arms (`f11a_w13_gemm`,
                       `f11b_router_gemm`); `f11_norm_kernel` is offered nothing
    f03, f10           nothing at all

The H200/C500 medians over the seven shared regimes, recomputed from
`report_glm52_c500/fusion_*.csv` and `report_glm52_h200/fusion_*.csv` rather than quoted from
memory: f01 0.997, f06 1.064, f08f09 #8/#9 atomic 1.017/1.027 and token-major 0.977/0.976
(the six TMA-offered variants); f03 1.861, f10 1.479 (offered nothing); and f04f05 3.639 /
7.530 / 3.053 / 7.385 (offered warp-spec and clusters, and the four LARGEST gains in the
study). f04f05 is why "offered TMA" and not "offered any Hopper axis" is the grouping that
reads backwards.

That comparison is confounded, and obviously so: the H200 is simply a much better GPU than
the C500, and nothing in a cross-device ratio separates "the levers did nothing" from "the
memory system did everything". The decisive experiment is the control arm -- the same
benchmarks on the same card with `GLM52_H200_CLASSIC=1`, which forces the capabilities off
at the source. A grep over `results/` and `log/` for `GLM52_H200_CLASSIC` returns nothing:
no results file and no log in this repo has ever recorded that env var. It has never been
run.

Everything execution-related is `run_h200`'s and is imported, not restated: GPU selection,
tenancy refusal, the preflight probe, the sm_90 device gate, `hwinfo`, and child-process
supervision -- the same single-implementation rule `run_f11_h200.py` exists to protect.

WHAT IS DIFFERENT HERE, and why each difference is load-bearing:

1.  **Every arm measures into a staging tree and NOTHING is ever merged.** A bench calls
    `record()` and rewrites its result file in full; the campaign files under
    `results/h200/` are this experiment's BASELINE, so a bench pointed at them would not
    merely lose rows, it would destroy the thing the control arm is being compared against.
    The control arm is a DIFF, not an append. There is no merge step in this file, no
    `shutil.copy`, no `shutil.move`, no function whose name contains "merge", and no
    `.pre_*` backup path. There is nothing to invoke by accident.

2.  **The staging tree is correctness, not tidiness.** `glm52_h200/common.py` defines
    `CKPT_ROOT = RESULTS_DIR/"_ckpt"`, and `ckpt_load()` fences on DEVICE only -- never on
    feature state. A classic run pointed at `results/h200/` would find the Hopper campaign's
    per-regime checkpoints, pass the device fence because it is the same card, replay them,
    and publish Hopper timings as the control arm. That is a null result that looks exactly
    like a finding. Redirecting `GLM52_H200_RESULTS_DIR` moves `_ckpt` with it, which is the
    only thing that makes the two arms independent.

3.  **Every family is verified to have actually engaged the arm before the run continues.**
    LOG-14 B2 is the precedent: `--disable-features` once disabled nothing at all while
    every result file dutifully recorded the feature as off, on the operator's only remote
    escape hatch. A control arm that silently did not engage is indistinguishable from "the
    Hopper features made no difference" -- which is this study's hypothesis. That is worse
    than having no control arm, so it is a hard abort, not a warning.

4.  **`f03` and `f10` are the noise floor, not filler.** They advertise no Hopper cfg key, so
    their classic arm is byte-identically configured to their Hopper arm and their
    classic-vs-campaign delta IS the cross-session run-to-run band that every other family's
    delta must be judged against. They run FIRST -- the campaign measured them in 57 s and
    108 s, so a non-engaging arm is caught within a couple of minutes rather than at the end
    of the arm. (Their 3 h and 4 h numbers are TIMEOUT CEILINGS, not estimates; see the
    runtime note in `print_plan`.) A run that loses them has lost its
    control -- and the verifier must never count their trivially-clean config-token scan as
    evidence that anything was disabled (see the vacuity guard in `verify_family`).

5.  **The whole-layer bench is excluded, loudly.** `bench_layer.py` carries a 16 h timeout
    (`run_h200.py:172`) and the per-fusion question does not need it. `--families` rejects
    the token `layer` with a non-zero exit rather than silently enabling a 16-hour arm on a
    machine nobody can log into.

A HAZARD THIS FILE CAN ONLY HALF-FIX, stated rather than left to be discovered.
`run_h200.quarantine_foreign_results` (run_h200.py:994-1016) walks `results.rglob("*.json")`
-- which reaches INTO this driver's staging tree, because it skips only
`_quarantine_foreign_*` directories, not every underscore-prefixed one -- and reads
`payload["_meta"]["device"]` (run_h200.py:1004-1005). Any staged JSON without that key
resolves to `''`, compares unequal to the live device name, and is MOVED. Two kinds of file
are exposed:

  * every `_engagement/<arm>/<fam>.verify.json` this driver writes. FIXED here: each one is
    stamped with `_meta: {"device": <device name>, ...}` before it is written.
  * every `<arm>/_ckpt/**/*.json`. NOT FIXABLE HERE. Those are written by
    `glm52_h200/common.py:ckpt_save`, which stamps a TOP-LEVEL `"device"` key and no `_meta`
    block at all (common.py:1482-1498), and this driver must not edit the harness the
    campaign was measured with -- changing `ckpt_save` would mean the two arms were produced
    by two different harnesses, which is the one thing the comparison cannot survive. So a
    later `run_h200.py` run on the same results dir can quarantine the staged checkpoints.
    They are a cache, not evidence: losing them costs re-measurement, not correctness, and
    the arm's own result JSON (which common.record() DOES stamp with `_meta.device`) is
    untouched. Ship the staging tree back before running anything else against
    `results/h200/`.

Like `run_h200.py`, `run_bs_extra_h200.py` and `run_f11_h200.py`, this file imports nothing
from torch, triton, numpy or matplotlib at driver level. It must stay able to report that the
stack itself is broken, it must not hold a CUDA context for the hours its children run, and
the measured box has python 3.10.12 with no numpy. The repo root is derived from
`Path(__file__).resolve().parent` -- on the measured box that is `/home/guancheng/zsh/fusion`,
which is not this checkout's path and must never be hardcoded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sys
import time
from dataclasses import asdict, dataclass
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


# ======================================================================================
# module constants
# ======================================================================================
DEFAULT_RESULTS = R.DEFAULT_RESULTS                       # results/h200
DEFAULT_LOGDIR = REPO / "log" / "run_control_h200"        # name == filename, house rule
STAGING_ROOT = "_control_arm"                             # leading _ : find_result skips it
SUMMARY_NAME = "control_arm_summary.json"
SENTINEL_NAME = "ARM_NOT_VERIFIED"
ENGAGE_DIRNAME = "_engagement"

#: The 7 in-scope families, cheapest-failure-first exactly as `run_h200.FAMILIES` orders
#: them. `layer` is absent by design; see EXCLUDED_FAMILIES.
FAMILY_KEYS: tuple[str, ...] = ("f03", "f10", "f01", "f04f05", "f11", "f06", "f08f09")
EXCLUDED_FAMILIES: tuple[str, ...] = ("layer",)

#: The campaign's physical card, bare (no `GPU-` prefix). Only a fallback: the real value is
#: read out of `results/h200/summary.json` at runtime by `campaign_gpu_uuid()`.
CAMPAIGN_UUID_FALLBACK = "b2318e71-edbf-aa1b-dcf3-3cc3e6ea7db0"
_REGIME_ALT = "|".join(re.escape(r) for r in R.KNOWN_REGIMES)

#: The three Hopper axis names as they appear in three different vocabularies.
#: axis -> (cfg key in a winning config, axis key in fairness.h200_axes, axis_counts key)
AXES: tuple[tuple[str, str, str], ...] = (
    ("tma", "USE_TMA", "USE_TMA"),
    ("warp_specialize", "warp_specialize", "warp_specialize"),
    ("clusters", "num_ctas", "num_ctas"),
)
AXIS_NAMES: tuple[str, ...] = tuple(a for a, _, _ in AXES)

#: Which axes the CAMPAIGN actually OFFERED each family. Anything not here cannot be
#: disabled, so a clean token scan for it proves nothing -- this is the vacuity guard's
#: fallback table, used only when the campaign file cannot be read. Verified against the
#: committed `results/h200/*.json` on 2026-08-10.
CAMPAIGN_OFFERED: dict[str, frozenset[str]] = {
    "f03": frozenset(),
    "f10": frozenset(),
    "f01": frozenset({"tma", "warp_specialize", "clusters"}),
    "f06": frozenset({"tma", "warp_specialize", "clusters"}),
    "f08f09": frozenset({"tma", "warp_specialize", "clusters"}),
    "f04f05": frozenset({"warp_specialize", "clusters"}),
    "f11": frozenset({"warp_specialize"}),
}

#: The evidence string `glm52_h200/bench/__init__.py:1173` builds from `hopper.caps()`, and
#: the reason string it writes at `:1274` when the live capability probe says no. Asserting
#: on producer text is brittle by construction; the producing lines are named here so that a
#: reword is traceable to the check it breaks instead of silently failing a good control arm.
EVIDENCE_ENV_MARKER = "(source env)"                     # bench/__init__.py:1173
REASON_UNAVAILABLE = "the live capability probe says it is unavailable"   # :1274
REASON_NO_CFG_KEY = "this kernel module advertises no cfg key for it"     # :1273
CLASSIC_CAPS_NOTE = "GLM52_H200_CLASSIC=1: all Hopper features forced off (control arm)"
#: kernels/hopper.py:723 emits CLASSIC_CAPS_NOTE into caps().notes; :170 defines CAP_NAMES.

#: Subtrees the config-token scan must NOT walk, matched by ANCESTOR KEY NAME so the rule
#: holds across all seven families' different shapes. Every one of these is a
#: PREFLIGHT-DERIVED or CAPABILITY-DERIVED record, not a config the tuner ever handed to
#: Triton, and every one of them stays truthy under `GLM52_H200_CLASSIC=1`.
#:   _meta.harness_info.features.warp_specialize is `true` in f03 and f10 TODAY -- the only
#:   two Hopper key occurrences in either file. A scan that walked it would report the two
#:   noise-floor families as "still using warp specialization" on both arms.
SCAN_EXCLUDED_KEYS: frozenset[str] = frozenset({
    "_meta", "env", "harness_info", "features", "feature_evidence",
    "h200_axes", "grids", "axis_counts", "preflight", "compile_probes",
    "kernel_caps_report", "hopper_caps", "device_probe",
})

#: Keys whose subtree is a tuner TABLE (a list of configs that were OFFERED) rather than a
#: winner. Both must be empty under a disabled axis, but the distinction is what lets the
#: verifier say "the tuner never picked it" separately from "the grid never contained it".
SCAN_TABLE_KEYS: frozenset[str] = frozenset({"table", "tables"})

#: `fairness.grids.<regime>.<arm>.<stage>` stage names whose grid is a PURE FUNCTION OF THE
#: CODE -- the grid the bench OFFERS, built before anything is timed. Only these may have
#: their `n_tried` asserted equal across two runs.
#:
#: Stated as an ALLOW-list, not a deny-list, so a stage name added to a bench later defaults
#: to "not comparable" (INFO) instead of silently becoming a fatal equality assertion.
#:
#:   coarse -- the bench's own `coarse_grid()`; f03 164 configs, f10 188, both uncapped.
#:             The capped families sit at their cap (f01 220, f06/f08f09 200, f11 180).
#:   tune   -- a single-stage search over a grid the bench built, not over a winner's
#:             neighbourhood. Observed sizes: f04f05 42/83/240/301/557 by sub-arm, f06 270,
#:             f08f09 7/10/17/132/144, f11 164. One caveat, harmless and recorded rather
#:             than left to be found: f08f09's `unfused_sum_res` / `fused_seed_res` tune
#:             stages search `B.top_cfgs(ts0, k=10)`, whose CONTENT is winner-derived but
#:             whose SIZE is pinned at k. f08f09 is not in the strict set below in any case,
#:             so nothing fatal rests on it.
#:
#: EVERYTHING ELSE is downstream of a measurement and moves run-to-run:
#:   refine -- `common.py:1069` builds it as `neighbours(best_cfg, coarse)` / the bench's own
#:             `refine_grid(tc.best_cfg)`, where `best_cfg` is the COARSE STAGE'S TIMING
#:             WINNER. A different winner is a different neighbourhood is a different size.
#:             `_stage()` also skips any config already timed (common.py:1017-1021), so the
#:             count depends on how far the refine set overlaps the coarse set.
#:   joint  -- the unfused-chain re-tune, built from `top_cfgs(...)` over the two MEASURED
#:             per-side searches and then DEDUPLICATED (bench_f03:216-231). Top-k by measured
#:             ms; the dedup then collapses a data-dependent number of pairs.
#:   extra  -- f01:400-406, `refine_grid(cfg)` over the fused side's own top configs.
#:
#: MEASURED, not asserted. `results/h200/_bs_extra_rerun/*.json` is an independent rerun of
#: the same benches, same code, same card, with the Hopper levers ON -- i.e. the campaign's
#: own arm, run twice. Against `results/h200/*.json` its `coarse`/`tune` sizes are identical
#: in 231 of 231 shared stages across all seven families (126 coarse + 105 tune; per family
#: f01 14, f03 21, f04f05 56, f06 21, f08f09 63, f10 21, f11 35), while `refine`/`joint`/
#: `extra` differ in 46 of 240 and in BOTH DIRECTIONS: f03 16 of 28, f10 18 of 28, f01 4 of
#: 35, f11 5 of 42, f06 2 of 21, f08f09 1 of 58. Two-directional movement is noise, not a
#: shrinking grid. (`extra` happened to match 7 of 7 in that one pair. It stays OUT of the
#: allow-list anyway: it is `refine_grid()` over a measured top-k by construction, and one
#: clean pair is not a licence to assert bit-determinism of a winner-derived size.)
#:
#: SECOND PROPERTY, the one `V9b` rests on: within a family, an offered stage's size is also
#: constant ACROSS REGIMES, because these grids are built from the module's advertised cfg
#: keys and the probed device, not from the shape. Every (sub-arm, stage) pair above holds a
#: single distinct value over the campaign's 7 regimes AND over the rerun's 11 -- with one
#: real exception that proves the rule and that V9b must therefore detect rather than assume
#: away: f08f09's `tokmaj8/coarse` takes 30 or 243 depending on the regime. So V9b may only
#: extrapolate a value the campaign recorded IDENTICALLY on every regime it did record, and
#: must fall back to INFO otherwise.
OFFERED_GRID_STAGES: frozenset[str] = frozenset({"coarse", "tune"})

#: There is deliberately NO floor bar constant here. `glm52_h200/config.py:272` defines
#: `FLOOR_US_MAX = 50.0` and `glm52/make_control_report_h200.py:139` keeps its own copy in
#: sync with it through `check_bar_sync()`. A third copy in this driver was defined and never
#: read: it enforced nothing, it was covered by no sync check, and it read as though the
#: driver refused a contended card, which it does not. The bar belongs to the report
#: generator, which is where the refusal actually happens.

#: The preflight's own bar for declaring a timer tick FOUND, restated from
#: `glm52_h200/config.py:270` (`TICK_MATCH_MIN`) and from `run_h200.py:1796`, which is the
#: implementation this driver mirrors in `build_tick()`.
TICK_MATCH_MIN = 0.98


# ======================================================================================
# small JSON helpers (duplicated verbatim from run_bs_extra_h200.py:105-130 -- run_h200 has
# no equivalents and duplicating these three is the established precedent in this repo)
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


def digest(obj) -> str:
    """Stable short digest, used so a repeated summary write is recognisably the same run."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8", "replace")
    return hashlib.sha256(blob).hexdigest()[:16]


# ======================================================================================
# THE WRITE GATE -- mechanism 1 of 3 protecting the campaign files
# ======================================================================================
#: Computed exactly once in main() as `(args.results_dir / STAGING_ROOT).resolve()`. Every
#: write this driver performs goes through `guard_write`. Left None until then so that an
#: unguarded write attempted during import or argument parsing is a crash, not a silent one.
#: Set by the signal handler so the summary can say HOW the run ended.
TERMINATED_BY_SIGNAL: str | None = None

WRITE_ROOT: Path | None = None


class StagingBreach(SystemExit):
    """A write was attempted outside the staging root.

    `SystemExit("message")` exits 1 -- python prints the string and uses status 1 -- and
    exit 1 is defined by this driver's own table as "finished, but some family is missing
    rows; partial data worth shipping back". A staging breach is the opposite of that: it
    means the write gate caught an attempt on the campaign baseline. So the message is
    printed and the INTEGER 6 is what the exception carries.
    """

    def __init__(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)
        super().__init__(6)


def guard_write(path: Path) -> Path:
    """Every path this driver writes -- and its one delete -- passes through here.

    A write outside the staging root is a bug that would destroy the baseline the whole
    experiment is a diff against, so it is fatal rather than a warning. The one exemption is
    the log directory, which is written only through `R.Log` and `R.run_family` and never
    touches the results tree.
    """
    p = Path(path).resolve()
    if WRITE_ROOT is None:
        raise StagingBreach(f"!! REFUSING to write {p}: the staging root has not been "
                            f"established yet. This is a programming error in "
                            f"run_control_h200.py. Exit 6.")
    if WRITE_ROOT not in p.parents and p != WRITE_ROOT:
        raise StagingBreach(f"!! REFUSING to write outside {WRITE_ROOT}: {p}\n"
                            f"!! The control arm is a DIFF, never an append. Exit 6.")
    return p


#: The env vars that ARE the experiment's switch. `R.run_family` copies `os.environ` into
#: every child (run_h200.py:1152) and only ADDS to it, so an inherited one is never cleared:
#: `--arms no-tma` in a shell that already exports `GLM52_H200_CLASSIC=1` measures the FULL
#: classic arm while the verifier -- which only inspects `arm.axes_off` -- passes every check
#: V1..V11 and files the result as a TMA-only decomposition.
OWNED_ENV_VARS: tuple[str, ...] = (
    "GLM52_H200_CLASSIC", "GLM52_H200_TMA", "GLM52_H200_WS", "GLM52_H200_CLUSTERS",
    "GLM52_H200_WGMMA", "GLM52_H200_DISABLE_FEATURES",
)


def refuse_inherited_env() -> int:
    """0 if the environment is clean; 2 (after printing why) if it is not.

    Checked before ANYTHING else, including --list, because the whole point is that the
    operator sees this on the rehearsal rather than partway into a contaminated arm.
    """
    found = [(k, os.environ[k]) for k in OWNED_ENV_VARS if k in os.environ]
    if not found:
        return 0
    print("!" * 92, flush=True)
    print("!! REFUSING to start: this driver OWNS the Hopper feature switch, and the "
          "environment", flush=True)
    print("!! it inherited already sets it:", flush=True)
    for k, v in found:
        print(f"!!     {k}={v!r}", flush=True)
    print("!!", flush=True)
    print("!! run_h200.run_family copies os.environ into every child and only ADDS to it, so "
          "an", flush=True)
    print("!! inherited value is never cleared. `--arms no-tma` under GLM52_H200_CLASSIC=1 "
          "measures", flush=True)
    print("!! the full classic arm, and the verifier -- which only inspects the axes the arm "
          "claims", flush=True)
    print("!! to disable -- passes every check and files it as a TMA-only decomposition. "
          "That is a", flush=True)
    print("!! wrong answer that looks right, which is the one failure this study cannot "
          "absorb.", flush=True)
    print("!!", flush=True)
    print("!! Clear them and re-run, e.g.:", flush=True)
    print("!!     unset " + " ".join(k for k, _ in found), flush=True)
    print("!! or run this driver in a clean shell:", flush=True)
    print("!!     env -u " + " -u ".join(k for k, _ in found)
          + " python3 run_control_h200.py ...", flush=True)
    print("!! An arm is a named row of the ARMS table and nothing else. Exit 2.", flush=True)
    print("!" * 92, flush=True)
    return 2


def ensure_dir(path: Path) -> Path:
    """mkdir -p, through the write gate."""
    p = guard_write(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write via a sibling temp file and rename -- through the write gate."""
    p = guard_write(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(p)


def atomic_write_text(path: Path, text: str) -> None:
    """The sentinel and the staging README -- through the write gate."""
    p = guard_write(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(p)


# ======================================================================================
# the arm table
# ======================================================================================
@dataclass(frozen=True)
class Arm:
    """One measurement arm, expressed ONLY as a --disable-features string.

    Arms are never expressed as raw env vars. `run_h200.run_family` (1172-1201) already owns
    the token -> env mapping and applies it per child; restating it here would be a second
    implementation of the one switch the whole experiment rests on.
    """

    name: str
    disable_features: str
    caps_off: frozenset[str]      # hopper.CAP_NAMES this arm forces off
    axes_off: frozenset[str]      # subset of {tma, warp_specialize, clusters} to verify
    note: str


ARMS: tuple[Arm, ...] = (
    Arm("classic", "all,tma,ws,warp_specialize,clusters,wgmma",
        frozenset({"tma", "warp_specialize", "clusters", "wgmma"}),
        frozenset({"tma", "warp_specialize", "clusters"}),
        "the control arm: GLM52_H200_CLASSIC=1, all four capabilities off"),
    Arm("no-tma", "tma",
        frozenset({"tma"}), frozenset({"tma"}),
        "per-feature decomposition: TMA only"),
    Arm("no-ws", "ws,warp_specialize",
        frozenset({"warp_specialize"}), frozenset({"warp_specialize"}),
        "per-feature decomposition: warp specialization only"),
    Arm("no-clusters", "clusters",
        frozenset({"clusters"}), frozenset({"clusters"}),
        "per-feature decomposition: thread-block clusters only"),
    Arm("hopper", "",
        frozenset(), frozenset(),
        "same-session re-measured baseline; the CONVERSE sanity check"),
)
ARM_BY_NAME = {a.name: a for a in ARMS}
DEFAULT_ARMS = "classic"

# Exactly what each string does inside `R.run_family` (run_h200.py:1179-1198, read directly;
# every token below is in run_h200's known set, so no `!!` unrecognised line is ever
# printed):
#
#   arm         | child env run_h200 sets                    | common.features() flipped
#   ------------|--------------------------------------------|---------------------------
#   classic     | DISABLE_FEATURES=<string>, CLASSIC=1,       | tma, ws, warp_specialize,
#               | TMA=0, WS=0, CLUSTERS=0, WGMMA=0           | clusters -> false
#   no-tma      | DISABLE_FEATURES=tma, TMA=0                 | tma -> false
#   no-ws       | DISABLE_FEATURES=ws,warp_specialize, WS=0   | ws, warp_specialize -> false
#   no-clusters | DISABLE_FEATURES=clusters, CLUSTERS=0       | clusters -> false
#   hopper      | nothing (the `if args.disable_features:`    | nothing
#               |  block is skipped entirely)                 |
#
# WHY `all` IS NOT ENOUGH ON ITS OWN. `GLM52_H200_CLASSIC=1` is the *real* switch --
# kernels/hopper.py:721-723 sets `overrides = {k: False for k in CAP_NAMES}` -- but `"all"`
# is not a key of `common.features()`'s own dict (common.py:433-437, exact-match and
# case-sensitive), so with `all` alone the result file would still record `tma: true`,
# `clusters: true`, `warp_specialize: true`, `disabled: []`. The extra tokens make the
# switch and the metadata agree. This is the exact inverse of the LOG-14 B2 defect -- there
# the metadata said off while the feature stayed live -- and it is why both halves are named
# in one string.
#
# KNOWN RESIDUAL GAP, stated rather than papered over. `features.dot_bf16` stays `true`
# under classic even though `wgmma` is forced off, because `dot_bf16` is a features-dict key
# but not a `_FEATURE_ENV` token; adding it to the string would make run_h200 print an
# `!! unrecognised` line to stdout that never reaches driver.log. It is left alone and
# documented. The verifier does not read `features.*` at all -- see SCAN_EXCLUDED_KEYS and
# the "blocks never read" note in `verify_family`.
#
# BREADTH. `GLM52_H200_CLASSIC` forces FOUR capabilities off, including `wgmma`
# (`CAP_NAMES`, hopper.py:170), one more than the three levers the question names. The
# report must not attribute an f01/f06/f08f09 delta purely to TMA/warp-spec/clusters; the
# per-feature arms behind `--arms` exist for exactly that decomposition.


def arm_expected_env(arm: Arm) -> dict[str, str]:
    """The env `R.run_family` will set for this arm, recomputed here for the summary only.

    This is a PREDICTION written into the record so the returned artefact can be checked
    against what the children actually saw (`fam_stages[...]["env_overrides"]`, which
    `R.run_family` captures from the real environment). It is never used to build the env.
    """
    env: dict[str, str] = {}
    if not arm.disable_features:
        return env
    env["GLM52_H200_DISABLE_FEATURES"] = arm.disable_features
    mapping = {"tma": "GLM52_H200_TMA", "clusters": "GLM52_H200_CLUSTERS",
               "cluster": "GLM52_H200_CLUSTERS", "ws": "GLM52_H200_WS",
               "warp_specialize": "GLM52_H200_WS", "warp-specialize": "GLM52_H200_WS",
               "wgmma": "GLM52_H200_WGMMA"}
    for tok in (s.strip().lower() for s in arm.disable_features.split(",")):
        if not tok:
            continue
        if tok in ("all", "classic"):
            env["GLM52_H200_CLASSIC"] = "1"
        elif tok in mapping:
            env[mapping[tok]] = "0"
    return env


# ======================================================================================
# the campaign: its card, its files, its fingerprint (mechanism 2 of 3)
# ======================================================================================
def campaign_files(results: Path) -> list[Path]:
    """Every non-underscore `*.json` directly in the campaign dir -- the 7 family files,
    `summary.json`, and the `.pre_*` backups. Underscore-prefixed staging and quarantine
    directories are skipped, which is the same rule that keeps `run_h200.find_result` and
    `quarantine_foreign_results` out of this driver's own tree."""
    if not results.is_dir():
        return []
    return sorted(p for p in results.glob("*.json")
                  if p.is_file() and not p.name.startswith("_"))


def fingerprint_campaign(results: Path) -> dict[str, dict]:
    """{name: {size, mtime_ns, sha256}} over the campaign files, taken before the first
    launch and re-checked after every family."""
    out: dict[str, dict] = {}
    for p in campaign_files(results):
        try:
            st = p.stat()
            h = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError as exc:
            out[p.name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        out[p.name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": h}
    return out


def check_campaign(fp: dict[str, dict], results: Path,
                   warnings: list[str]) -> list[str]:
    """Re-hash the campaign and report every difference.

    This catches the one failure mode intent cannot: a bench that ignores
    `GLM52_H200_RESULTS_DIR` and writes where its own default points. An empty return value
    is the proof, written into the summary, that the baseline was untouched.
    """
    now = fingerprint_campaign(results)
    diffs: list[str] = []
    for name in sorted(set(fp) | set(now)):
        was, is_ = fp.get(name), now.get(name)
        if was is None:
            diffs.append(f"{name}: APPEARED during the run")
        elif is_ is None:
            diffs.append(f"{name}: DISAPPEARED during the run")
        elif was.get("sha256") != is_.get("sha256"):
            diffs.append(f"{name}: CONTENT CHANGED "
                         f"({was.get('size')} -> {is_.get('size')} bytes)")
    for d in diffs:
        warnings.append(f"campaign canary: {d}")
    return diffs


def results_dir_matches(got: object, staging: Path) -> tuple[bool, str]:
    """Did `_meta.results_dir` name the staging tree? (ok, how it was decided).

    Two acceptances, not one. The exact resolved path is the strong form and is what a live
    run on the measured box produces. The weak form compares only the trailing
    `_control_arm/<arm>` components, because the returned tarball is unpacked at a different
    absolute path than the one it was measured at -- `/home/guancheng/zsh/fusion/...` on the
    H200 versus wherever it lands here -- and a check that failed every family after an
    honest relocation would be a check nobody could act on.
    """
    if got is None:
        return False, "no _meta.results_dir in the file"
    want = Path(staging).resolve()
    try:
        have = Path(str(got)).expanduser()
    except (OSError, ValueError):
        return False, "unparseable _meta.results_dir"
    try:
        if have.resolve() == want:
            return True, "exact path match"
    except (OSError, ValueError):
        pass
    if have.parts[-2:] and tuple(have.parts[-2:]) == tuple(want.parts[-2:]):
        return True, (f"relocated tree: the absolute path differs but the trailing "
                      f"{'/'.join(want.parts[-2:])} components match")
    return False, f"recorded {str(got)!r}, which is not under {want}"


def canary_or_raise(log: R.Log, fp: dict[str, dict], results: Path,
                    warnings: list[str], when: str) -> None:
    """Re-check the campaign fingerprint and STOP the run if anything moved.

    One implementation, three call sites (per attempt, per family, and once at the end), so
    the message an operator six hours away reads is the same wherever it fired.
    """
    diffs = check_campaign(fp, results, warnings)
    if not diffs:
        return
    log("")
    log("!" * 92)
    log(f"!! CAMPAIGN CANARY TRIPPED ({when}): a file under the campaign directory changed "
        f"during this run.")
    for d in diffs:
        log(f"!!   {d}")
    log("!! A bench ignored GLM52_H200_RESULTS_DIR, or something else wrote into the "
        "baseline.")
    log("!! The control arm is a DIFF and its baseline has moved; stopping now, before more "
        "of it")
    log("!! is overwritten. Send back the whole results dir AND its git status.")
    log("!" * 92)
    raise _CanaryTripped(diffs)


def check_results_dir_honoured(log: R.Log, result_path: Path, staging: Path,
                               warnings: list[str]) -> None:
    """Did the bench write where it was told to? Read from the file it just produced.

    `common.record()` stamps `_meta.results_dir` with the directory it actually wrote to
    (glm52_h200/common.py:1342; the committed f03 file carries
    '/home/guancheng/zsh/fusion/results/h200'). Everything else about the redirect is an
    assumption: `R.script_flags()` returns an EMPTY set for six of the seven benches, so
    run_h200.py's `--results-dir` branch never fires for them and `run_family`'s `unhonoured`
    list -- which does cover --quick and --regimes -- says nothing about it either. If
    somebody later moves a `--results-dir` argument with a `results/h200` default into
    `bench/__init__.py:add_std_args()`, invisible to script_flags exactly as --regimes is
    today, argparse's default wins over common.RESULTS_DIR and the bench writes classic-arm
    timings straight over the baseline.
    """
    payload, err = load_json(result_path)
    if payload is None:
        # Nothing was produced, so there is nothing to check and nothing to conclude. The
        # attempt's own status already reports the failure.
        log(f"  [redirect] no result file to check yet ({result_path.name}: {err}).")
        return
    got = _jget(payload, "_meta", "results_dir")
    want = Path(staging).resolve()
    ok, how = results_dir_matches(got, staging)
    if ok:
        log(f"  [redirect] confirmed ({how}): the bench recorded _meta.results_dir = {got}")
        return
    log("")
    log("!" * 92)
    log("!! THE STAGING REDIRECT WAS NOT HONOURED.")
    log(f"!!   expected  {want}")
    log(f"!!   recorded  {got if got is not None else '(no _meta.results_dir in the file)'}")
    log(f"!!   file      {result_path}")
    log("!! GLM52_H200_RESULTS_DIR is the ONLY thing that keeps this arm's measurements and")
    log("!! its per-regime _ckpt cache out of the campaign directory this run is a DIFF")
    log("!! against. Stopping after the first attempt of the first family rather than")
    log("!! discovering it at the end of the arm. Nothing else has been launched.")
    log("!" * 92)
    msg = (f"the staging redirect was not honoured: {result_path.name} recorded "
           f"_meta.results_dir={got!r}, expected {want}")
    warnings.append(msg)
    raise _RedirectIgnored(msg)


def campaign_gpu_uuid(results: Path) -> tuple[str, str]:
    """(bare-uuid, where-it-came-from). Resolution order, first hit wins:

       summary.json _meta.gpu_uuid  ->  summary.json gpu.uuid  ->  summary.json device.uuid
       ->  the modal env.uuid across results/<family>.json  ->  CAMPAIGN_UUID_FALLBACK.

    Always normalised with `R._norm_uuid`: nvidia-smi and summary.json write
    'GPU-b2318e71-...', the family files write 'b2318e71-...'. Same card. A cross-file
    comparison with `==` fails on every honest file, which is why this returns the bare form.
    """
    summary, err = load_json(results / "summary.json")
    if summary:
        for path, label in ((("_meta", "gpu_uuid"), "summary.json _meta.gpu_uuid"),
                            (("gpu", "uuid"), "summary.json gpu.uuid"),
                            (("device", "uuid"), "summary.json device.uuid")):
            node: object = summary
            for k in path:
                node = (node or {}).get(k) if isinstance(node, dict) else None
            got = R._norm_uuid(node)
            if got:
                return got, label
    counts: dict[str, int] = {}
    for key in FAMILY_KEYS:
        fam = R.FAMILY_BY_KEY.get(key)
        if fam is None:
            continue
        path = R.find_result(fam, results)
        if path is None:
            continue
        payload, _ = load_json(path)
        u = R._norm_uuid(((payload or {}).get("env") or {}).get("uuid"))
        if u:
            counts[u] = counts.get(u, 0) + 1
    if counts:
        best = max(counts.items(), key=lambda kv: kv[1])
        return best[0], f"modal env.uuid across {sum(counts.values())} family file(s)"
    return R._norm_uuid(CAMPAIGN_UUID_FALLBACK), (
        f"hardcoded fallback (summary.json {err or 'unusable'})")


def campaign_floors(results: Path) -> dict[str, float | None]:
    """`fairness.timing.harness_floor_us` per family, carried into the summary so the two
    sessions' floors can be compared without re-opening 25 MB of JSON."""
    out: dict[str, float | None] = {}
    for key in FAMILY_KEYS:
        fam = R.FAMILY_BY_KEY.get(key)
        path = R.find_result(fam, results) if fam else None
        if path is None:
            out[key] = None
            continue
        payload, _ = load_json(path)
        v = (((payload or {}).get("fairness") or {}).get("timing") or {}) \
            .get("harness_floor_us")
        try:
            out[key] = float(v) if v is not None else None
        except (TypeError, ValueError):
            out[key] = None
    return out


# ======================================================================================
# the CUDA-event tick, and whether it may be trusted
# ======================================================================================
def build_tick(pf: dict | None, warnings: list[str],
               no_preflight_source: str = "default (no preflight)") -> dict:
    """The tick dict, built the way `run_h200.py:1770-1807` builds it and not another way.

    This driver used to hardcode `trusted: True, distrust_reasons: []` and never read
    `timer_tick_match_frac` at all, which had two consequences. (1) The tick feeds
    `R.collect_cells`, so every UNRESOLVED verdict in the arm would have been computed from a
    tick the preflight had already declared untrustworthy, with no warning anywhere -- on a
    machine nobody can log into. (2) `glm52/make_control_report_h200.py:803` reads
    `timer.match_frac` out of the summary and treats a MISSING value as a REFUSE-severity
    provenance row, so a fully successful, fully verified control arm was unpublishable
    without `--force-floor` -- which simultaneously disables the genuine contended-card
    refusal that exists because of the 40.55 us preflight.

    The three rules are run_h200's, restated only in the sense that this is a second call
    site: `launch_timer_trustworthy is False` distrusts outright with the preflight's own
    doubts; otherwise a `timer_tick_match_frac` below TICK_MATCH_MIN distrusts; and the same
    warning text run_h200 appends is appended here, so the two drivers' transcripts say the
    same thing about the same condition.
    """
    cal = (pf or {}).get("calibration") or {}
    tick_us = None
    tick_src = no_preflight_source
    if pf:
        v = cal.get("timer_tick_us")
        try:
            if v and float(v) > 0:
                tick_us, tick_src = float(v), "measured by preflight"
        except (TypeError, ValueError):
            pass
    if tick_us is None:
        tick_us = R.DEFAULT_TICK_US
        warnings.append(f"CUDA-event tick not measured; assumed {R.DEFAULT_TICK_US} us "
                        f"(the coarsest seen in this study, so UNRESOLVED over-flags rather "
                        f"than under-flags)")

    trusted, distrust = True, []
    if cal.get("launch_timer_trustworthy") is False:
        trusted = False
        distrust = list(cal.get("launch_timer_doubts") or ["flagged by the preflight"])
    else:
        frac = cal.get("timer_tick_match_frac")
        if isinstance(frac, (int, float)) and not isinstance(frac, bool) \
                and frac < TICK_MATCH_MIN:
            trusted = False
            distrust = [f"the preflight's winning tick quantum matched only "
                        f"{frac * 100:.0f}% of its samples; a real tick matches ~100%"]
    if not trusted:
        warnings.append(
            f"the CUDA-event tick ({tick_us} us) used for every UNRESOLVED verdict is itself "
            f"untrustworthy: {'; '.join(distrust)}. Re-run the preflight on an idle GPU "
            f"before quoting any decode_bs1 cell.")
    return {"tick_us": tick_us, "source": tick_src, "trusted": trusted,
            "distrust_reasons": distrust,
            "match_frac": cal.get("timer_tick_match_frac")}


# ======================================================================================
# GPU selection -- prefer the campaign's physical card
# ======================================================================================
def pin_campaign_card(log: R.Log, args: argparse.Namespace, hw: list[dict],
                      warnings: list[str], anchor: dict) -> dict:
    """Turn the campaign UUID into an nvidia-smi index, then hand off to `R.resolve_gpu`.

    Returns the `R.resolve_gpu` decision dict, or a dict carrying `refuse=True` when this
    driver itself refuses (campaign card missing, or an explicit `--gpu` contradicting it).
    `anchor` is filled in place and lands verbatim in the summary.
    """
    want_uuid, src = campaign_gpu_uuid(args.results_dir)
    anchor.update({"matched": None, "want_uuid": want_uuid, "want_uuid_source": src,
                   "got_uuid": None, "note": ""})
    # Same rule `R.resolve_gpu` applies to a busy card: when nothing is being TIMED, a wrong
    # or absent card is a fact to record, not a reason to refuse. --dry-run and --list exist
    # so an operator can rehearse the command on a laptop before booking the card; refusing
    # them on the grounds that the H200 is not present would defeat the rehearsal.
    measuring = not (args.dry_run or args.list or args.verify_only)
    rows = R.gpu_rows(hw)
    match = next((r for r in rows if R._norm_uuid(r.get("uuid")) == want_uuid), None)
    here = ", ".join(f"{R._norm_uuid(r.get('uuid'))[:8]}@{r.get('index')}"
                     for r in rows) or "(nvidia-smi listed no cards)"

    if match is not None:
        idx = str(match.get("index"))
        if args.gpu and str(args.gpu).strip().lower() not in ("auto", "", idx):
            log("!" * 92)
            log(f"!! --gpu {args.gpu} was given, but the campaign's card {want_uuid[:8]} is "
                f"nvidia-smi index {idx} on this host.")
            log("!! An explicit --gpu that contradicts the campaign card is operator error, "
                "and silently")
            log("!! doing something else is how the wrong card gets used. Drop --gpu (the "
                "campaign card is")
            log(f"!! pinned automatically), or pass --gpu {idx}, or accept the weaker claim "
                f"with --any-gpu.")
            log("!" * 92)
            anchor.update({"matched": False, "note": f"--gpu {args.gpu} vs campaign index "
                                                     f"{idx}"})
            warnings.append(f"--gpu {args.gpu} contradicts the campaign card at index {idx}")
            if measuring:
                return {"refuse": True, "index": None, "uuid": None,
                        "reason": "explicit --gpu contradicts the campaign card"}
            log("  (not measuring, so this is recorded rather than refused)")
        if not args.gpu or str(args.gpu).strip().lower() == "auto":
            args.gpu = idx
        log(f"[gpu] pinning the campaign's own card {want_uuid[:8]} at nvidia-smi index "
            f"{idx} (from {src})")
        anchor.update({"matched": True, "got_uuid": want_uuid,
                       "note": f"pinned the campaign card at index {idx}"})
    else:
        if not args.any_gpu:
            log("!" * 92)
            log(f"!! the campaign's card {want_uuid[:8]} is not on this host. This host has: "
                f"{here}.")
            log("!! Cross-session comparison against results/h200/ assumes the SAME physical "
                "card.")
            log("!! Re-run on the campaign's node, or accept the weaker claim with --any-gpu.")
            log("!" * 92)
            anchor.update({"matched": False,
                           "note": f"campaign card {want_uuid[:8]} absent; host has {here}"})
            warnings.append(f"campaign card {want_uuid[:8]} is not on this host")
            if measuring:
                return {"refuse": True, "index": None, "uuid": None,
                        "reason": "the campaign's card is not on this host"}
            log("  (not measuring, so this is recorded rather than refused; a real run "
                "would exit 4 here)")
        args.gpu = args.gpu or "auto"
        if args.any_gpu:
            msg = (f"DEVICE ANCHOR LOST: the campaign ran on {want_uuid[:8]} which is not "
                   f"on this host ({here}). Every delta in this run is confounded with "
                   f"card-to-card variation on top of the cross-session drift the design "
                   f"already concedes.")
            log(f"!! {msg}")
            warnings.append(msg)
            anchor.update({"matched": False, "note": msg})

    gpu = R.resolve_gpu(log, args, hw, warnings, measuring=measuring)
    if gpu.get("refuse"):
        log("")
        log("  the campaign's card is busy; waiting is the cheap option, because a different")
        log("  card makes the classic-vs-campaign delta uninterpretable -- or --any-gpu and")
        log("  accept that. (this driver: python3 run_control_h200.py [--any-gpu])")
        return gpu
    got = R._norm_uuid(gpu.get("uuid"))
    if got:
        anchor["got_uuid"] = got
        if anchor.get("matched") is None:
            anchor["matched"] = (got == want_uuid)
    return gpu


# ======================================================================================
# staging layout
# ======================================================================================
def staging_for(args: argparse.Namespace, arm_name: str) -> Path:
    return args.results_dir / STAGING_ROOT / arm_name


def engagement_dir(args: argparse.Namespace, arm_name: str) -> Path:
    return args.results_dir / STAGING_ROOT / ENGAGE_DIRNAME / arm_name


def family_canonical(fam: R.Family, results: Path) -> Path:
    """The campaign's own file for a family, resolved via the driver's glob rules. READ
    ONLY for the whole lifetime of this file."""
    got = R.find_result(fam, results)
    return got if got is not None else results / f"{fam.key}.json"


def staged_family_path(fam: R.Family, staging: Path, results: Path) -> Path:
    """The result file a bench writes into `staging` for `fam`.

    Benches name their file after their own RESULT_ID (`f03_resadd_rmsnorm.json`), NOT after
    the family key (`f03.json`). `run_bs_extra_h200.py` used `f"{fam.key}.json"` and every
    merge refused with "missing" while the staged file sat next to it under its RESULT_ID
    name -- the reason its whole first run merged nothing.
    """
    return staging / family_canonical(fam, results).name


def regime_rows(payload: dict | None) -> dict[str, list[dict]]:
    """{regime: [rows]} from a family result file, tolerating a missing shape."""
    out: dict[str, list[dict]] = {}
    for row in ((payload or {}).get("rows") or []):
        if isinstance(row, dict) and row.get("regime"):
            out.setdefault(str(row["regime"]), []).append(row)
    return out


def canonical_mult(fam: R.Family, results: Path) -> int:
    """Rows-per-regime the CAMPAIGN file carries (the modal count over its regimes); the
    completeness bar this arm is held to. 11 for f01/f03/f06/f10/f11 (mult 1) and 44 for
    f04f05/f08f09 (mult 4)."""
    payload, _ = load_json(family_canonical(fam, results))
    counts: dict[int, int] = {}
    for v in regime_rows(payload).values():
        counts[len(v)] = counts.get(len(v), 0) + 1
    if not counts:
        return 1
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def last_regime_in_log(path: Path) -> str | None:
    """Last regime-name token in a bench log, used to pick the write after a hard abort."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    last = None
    for m in re.finditer(rf"\b({_REGIME_ALT})\b", text):
        last = m.group(1)
    return last if last in R.KNOWN_REGIMES else None


# ======================================================================================
# ENGAGEMENT VERIFICATION
# ======================================================================================
@dataclass(frozen=True)
class Check:
    id: str            # "V1".."V11"
    axis: str | None   # tma | warp_specialize | clusters | None
    path: str          # the exact JSON key path read
    want: str
    got: str
    status: str        # PASS | FAIL | VACUOUS | INFO
    detail: str = ""


def _jget(node: object, *keys: str) -> object:
    """Safe nested lookup. `_meta.device` is a STRING, not a dict -- indexing it as one
    throws TypeError, which is exactly the bug this helper exists to make impossible."""
    for k in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(k)
    return node


def _axis_truthy(axis: str, key: str, value: object) -> bool:
    """Is this cfg key/value pair an ACTIVE use of `axis`?

    Hopper keys are OMITTED when off, never written as `false`/`1`: a full scan of all seven
    committed files found zero occurrences of `USE_TMA: false`, `warp_specialize: false` or
    `num_ctas: 1`. So this tests key PRESENCE with a truthy value. A check written as
    `cfg.get("USE_TMA") is False` passes vacuously on BOTH arms and proves nothing.
    """
    if isinstance(value, (dict, list)):
        return False
    if axis == "clusters":
        # num_ctas is an int; 1 is "no cluster", which the launcher writes explicitly.
        try:
            return value is not None and value is not False and int(value) > 1
        except (TypeError, ValueError):
            return False
    if key == "TMA_MODE":
        # a string spelling ("device"/"host"/"none"), not a boolean
        return str(value).strip().lower() not in ("", "none", "off", "false")
    return bool(value)


#: cfg key -> axis. `TMA_MODE` is included because a kernel that still emits a descriptor
#: form is still using TMA even if `USE_TMA` were somehow absent.
CFG_KEY_AXIS: dict[str, str] = {
    "USE_TMA": "tma", "TMA_A": "tma", "TMA_B": "tma", "TMA_MODE": "tma",
    "warp_specialize": "warp_specialize", "WARP_SPECIALIZE": "warp_specialize",
    "num_consumer_groups": "warp_specialize", "num_buffers_warp_spec": "warp_specialize",
    "num_ctas": "clusters",
}


def scan_hopper_tokens(node: object, path: str = "$",
                       in_table: bool = False) -> list[tuple[str, str, object, bool]]:
    """(axis, json-path, value, in_table) for every truthy Hopper cfg key in a payload.

    Implemented as a generic recursive walk because the cfg paths differ per family and are
    polymorphic: f08f09's `rows[*].fused_cfg` is flat for the 22 token-major rows and nested
    `{seed, gemm}` for the 22 atomic ones; f06 uses sibling `unfused_gemm_cfg` /
    `unfused_act_cfg`; f01 nests `EPI`; f04f05's `unfused_cfg.topk` is `null` in 22 of 44
    rows; f11 has eight distinct cfg path templates and no top-level `fused_cfg` at all. A
    per-family path list would be seven chances to miss one.

    Two things this walk deliberately does NOT visit, each because visiting it produces a
    guaranteed false reading:
      * SCAN_EXCLUDED_KEYS -- preflight/capability records. `_meta.harness_info.features
        .warp_specialize` is `true` in f03 and f10 today and stays `true` under the control
        arm; it is the ONLY Hopper key in either of those two files.
      * `fairness.h200_axes.overlays_offered` and `fairness.grids.*.axis_counts`, which are
        checked by V3 and V8 with their own semantics rather than as config tokens.

    `in_table` marks a hit that lives inside a tuner TABLE (a list of configs that were
    OFFERED) rather than a winner. Anything not under a key named `table`/`tables` counts as
    a winner -- deliberately over-inclusive, because for a disabled axis both counts must be
    zero and an over-inclusive winner count fails loud rather than quiet.
    """
    out: list[tuple[str, str, object, bool]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k in SCAN_EXCLUDED_KEYS:
                continue
            sub = f"{path}.{k}"
            axis = CFG_KEY_AXIS.get(k)
            if axis is not None and not isinstance(v, (dict, list)):
                if _axis_truthy(axis, k, v):
                    out.append((axis, sub, v, in_table))
                continue
            out += scan_hopper_tokens(v, sub, in_table or k in SCAN_TABLE_KEYS)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += scan_hopper_tokens(v, f"{path}[{i}]", in_table)
    return out


def offered_axes_from_campaign(campaign: dict | None,
                               fam_key: str) -> tuple[frozenset[str], str]:
    """(axes the CAMPAIGN offered this family, where that came from).

    Reads `fairness.h200_axes.per_family` and ITERATES the dict -- never string-builds the
    key. f08f09's per_family key is `f08f09_down_merge` (not `..._resadd`, which is the
    RESULT_ID) and f11 carries three (`f11a_w13_gemm`, `f11b_router_gemm`,
    `f11_norm_kernel`). An axis is in the surface if `offered is True` under ANY per_family
    key of that file.
    """
    per = _jget(campaign, "fairness", "h200_axes", "per_family")
    if isinstance(per, dict) and per:
        found: set[str] = set()
        for _kname, block in per.items():
            axes = _jget(block, "axes")
            if not isinstance(axes, dict):
                continue
            for axis in AXIS_NAMES:
                if _jget(axes, axis, "offered") is True:
                    found.add(axis)
        return frozenset(found), "campaign fairness.h200_axes.per_family"
    return (CAMPAIGN_OFFERED.get(fam_key, frozenset()),
            "CAMPAIGN_OFFERED fallback (the campaign file was unreadable)")


def per_family_keys(payload: dict | None) -> list[str]:
    per = _jget(payload, "fairness", "h200_axes", "per_family")
    return sorted(per) if isinstance(per, dict) else []


def _grid_stages(payload: dict | None):
    """Yield (regime, arm, stage, stage-dict) over `fairness.grids`, skipping `_totals`."""
    grids = _jget(payload, "fairness", "grids")
    if not isinstance(grids, dict):
        return
    for regime, arms in grids.items():
        if not isinstance(arms, dict):
            continue
        for armk, stages in arms.items():
            if armk == "_totals" or not isinstance(stages, dict):
                continue
            for stage, sd in stages.items():
                if isinstance(sd, dict):
                    yield str(regime), str(armk), str(stage), sd


def _f11_caps_reports(payload: dict | None):
    """Yield (json-path, hopper_caps dict) for f11's per-row capability dumps."""
    for i, row in enumerate((payload or {}).get("rows") or []):
        iso = _jget(row, "isolation_fuse_on_vs_off_same_cfg")
        if not isinstance(iso, dict):
            continue
        for side in ("router", "moe"):
            caps = _jget(iso, side, "kernel_caps_report", "hopper_caps")
            if isinstance(caps, dict):
                yield (f"$.rows[{i}].isolation_fuse_on_vs_off_same_cfg.{side}"
                       f".kernel_caps_report.hopper_caps"), caps


#: There is no `grep_child_logs(paths, needle)` helper any more. It existed only to look for
#: two strings that provably never reach a bench transcript (see below), and once the check
#: was rewritten to read the banner the helper had no caller. An unused helper in this file
#: is the same defect as an unused constant: a third place a rule can be restated, guarded by
#: nothing.

#: The needle the child-log corroboration actually greps for, and the axis names it can carry.
#: `glm52_h200/common.py:banner()` (common.py:1299-1309) is printed by EVERY bench as its
#: first line -- confirmed in the committed campaign transcript, `log/run_h200/f03.log:3`:
#:
#:   [harness] NVIDIA H200 sm9.0 | ... | hopper feats [tma,clusters,warp_specialize] | ...
#:
#: and it renders exactly the three axes this driver disables, in that order, or the literal
#: token `none` when all three are off. Under `GLM52_H200_CLASSIC=1` the line reads
#: `hopper feats [none]`; under `--arms no-tma` it reads `hopper feats
#: [clusters,warp_specialize]`. That single string discriminates every arm in the table.
#:
#: WHAT THIS REPLACES, and why. The check used to grep for CLASSIC_CAPS_NOTE on the classic
#: arm and for EVIDENCE_ENV_MARKER ('(source env)') on the others. Neither can ever appear:
#: CLASSIC_CAPS_NOTE is appended to `hopper.caps().notes`, which is rendered ONLY by
#: `hopper.report()` (hopper.py:488-511) and nothing under glm52_h200/ calls it; and
#: EVIDENCE_ENV_MARKER is the parenthesised form built by bench/__init__.py:1173 into the
#: RESULT FILE, while `hopper.banner()` spells the same fact `src env`. So the check reported
#: `0 of N log(s)` on a perfectly good arm and was dead in exactly the case its docstring
#: claimed -- a bench that died at hour six before writing any JSON.
FEATS_NEEDLE = "hopper feats ["
FEATS_AXES: tuple[str, ...] = ("tma", "clusters", "warp_specialize")
_FEATS_RE = re.compile(re.escape(FEATS_NEEDLE) + r"([^\]]*)\]")


def feats_in_child_logs(paths: list[Path]) -> tuple[int, list[tuple[str, frozenset[str]]]]:
    """(logs carrying the banner, [(log name, axes the harness reported live)]).

    `none` maps to the empty set. An axis name the banner grows later that this driver does
    not know is carried through verbatim rather than dropped, so a reword shows up as an
    unexpected token instead of as a silent pass.
    """
    seen: list[tuple[str, frozenset[str]]] = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        m = _FEATS_RE.search(text)
        if m is None:
            continue
        toks = {t.strip() for t in m.group(1).split(",") if t.strip()}
        seen.append((p.name, frozenset(t for t in toks if t != "none")))
    return len(seen), seen


def verify_family(arm: Arm, fam_key: str, payload: dict | None, child_logs: list[Path],
                  campaign: dict | None, gpu: dict, device_name: str,
                  device_fence: bool = True, campaign_error: str = "",
                  staging: Path | None = None) -> dict:
    """Did this family's child actually engage `arm`? Eleven checks, V0b and V1..V11.

    BLOCKS THIS VERIFIER MUST NEVER READ, and why each one is a guaranteed false negative:
      * `_meta.harness_info.features.*`      -- preflight-derived; still true under classic
      * `env.tma_supported` / `tma_host_supported` / `tma_device_supported` /
        `clusters_supported` / `warp_spec_supported`  -- ditto, they describe the DEVICE
      * `env.feature_evidence.*`             -- ditto
      * `fairness.preflight.features.*`      -- literally a copy of the preflight JSON
      * `probe_mode`                         -- `"skipped"` in BOTH arms; not a discriminator
    Asserting on any of them would pass on both arms and prove nothing, and asserting on
    them INVERTED would fail a perfectly good control arm.
    """
    checks: list[Check] = []
    surface, surface_src = offered_axes_from_campaign(campaign, fam_key)
    is_hopper_arm = (arm.name == "hopper")
    target = frozenset(arm.axes_off) & surface
    vacuous = frozenset(arm.axes_off) - surface
    pkeys = per_family_keys(payload) or per_family_keys(campaign)

    def add(cid, axis, path, want, got, status, detail=""):
        checks.append(Check(cid, axis, path, str(want), str(got), status, detail))

    if payload is None:
        add("V0", None, "$", "a parseable result file", "missing or unreadable", "FAIL",
            "the bench produced no result file for this family; nothing can be verified")
        return {"family": fam_key, "arm": arm.name, "verdict": "FAILED",
                "checks": [asdict(c) for c in checks], "n_fail": 1,
                "surface": sorted(surface), "surface_source": surface_src,
                "vacuous_axes": sorted(vacuous), "engaged_axes": [],
                "target_axes": sorted(target)}

    axes_root = "$.fairness.h200_axes.per_family"

    # ---- V0b: the staging redirect was actually HONOURED ------------------------------
    # The whole isolation of the two arms rests on GLM52_H200_RESULTS_DIR, and until this
    # check existed nothing ever verified it was obeyed. `R.script_flags()` returns an EMPTY
    # set for six of the seven benches, so run_h200.py:1148's `--results-dir` branch is dead
    # code for them and `run_family`'s `unhonoured` list does not cover --results-dir either.
    # `common.record()` stamps the directory it actually wrote to (common.py:1342), so this
    # is an exact, cheap, after-the-fact confirmation rather than an assumption.
    if staging is not None:
        got_dir = _jget(payload, "_meta", "results_dir")
        same_dir, how = results_dir_matches(got_dir, staging)
        add("V0b", None, "$._meta.results_dir", str(Path(staging)),
            got_dir if got_dir is not None else "(absent)",
            "PASS" if same_dir else "FAIL",
            "the bench must have written into the staging tree, not into the campaign "
            "directory its own default points at; common.py:1342 records where record() "
            f"actually wrote. {how}")

    # ---- V0c: is this family verifiable AT ALL? ----------------------------------------
    # `pkeys` empty on BOTH sides silently collapses V1/V2/V3/V4/V5 to zero iterations: the
    # loops below run zero times, V7 finds nothing to object to, V8 and V11 report INFO, and
    # the family passes with n_fail == 0 under the verdict NO-DIFFERENCE-AVAILABLE. That is
    # "read a missing key and treat absence as a pass", which is the failure this whole
    # verifier exists to prevent, so it is fatal instead.
    if not pkeys:
        add("V0c", None, f"{axes_root} (this file AND the campaign baseline)",
            "at least one per_family key on either side", "none on either side", "FAIL",
            f"nothing to verify: this result file carries no fairness.h200_axes.per_family "
            f"block and the campaign baseline is unusable "
            f"({campaign_error or 'no per_family block either'}). A family that cannot be "
            f"checked is not a family that passed.")

    # ---- the CONVERSE arm: assert the levers are PRESENT ------------------------------
    if is_hopper_arm:
        if not surface:
            add("V1", None, axes_root, "n/a", "no axis is offered to this family",
                "INFO", "f03/f10/f11_norm advertise no cfg key; the converse check is "
                        "NOT-APPLICABLE for them")
            hits = scan_hopper_tokens(payload)
            # n_fail is COUNTED, never assumed zero: V0b/V0c are added above this branch and
            # a hardcoded 0 here would let a staging breach ride out as NOT-APPLICABLE.
            n_fail = sum(1 for c in checks if c.status == "FAIL")
            return {"family": fam_key, "arm": arm.name,
                    "verdict": "FAILED" if n_fail else "NOT-APPLICABLE",
                    "checks": [asdict(c) for c in checks], "n_fail": n_fail,
                    "surface": [], "surface_source": surface_src, "vacuous_axes": [],
                    "engaged_axes": [], "target_axes": [],
                    "winner_token_hits": len([h for h in hits if not h[3]])}
        for axis in sorted(surface):
            for k in pkeys:
                avail = _jget(payload, "fairness", "h200_axes", "per_family", k, "axes",
                              axis, "available")
                add("V1", axis, f"{axes_root}.{k}.axes.{axis}.available", "True",
                    avail, "PASS" if avail is True else "FAIL",
                    "the converse: a re-measured Hopper baseline must still HAVE the "
                    "capability")
                off = _jget(payload, "fairness", "h200_axes", "per_family", k, "axes",
                            axis, "offered")
                nob = _jget(payload, "fairness", "h200_axes", "per_family", k, "axes",
                            axis, "not_offered_because")
                # A per_family key may legitimately not offer an axis the FAMILY offers
                # elsewhere (f11_norm_kernel offers nothing while f11a/f11b offer ws).
                status = "PASS" if off is True else (
                    "INFO" if nob == REASON_NO_CFG_KEY else "FAIL")
                add("V2", axis, f"{axes_root}.{k}.axes.{axis}.offered", "True", off,
                    status, f"not_offered_because={nob!r}")
        winners = [h for h in scan_hopper_tokens(payload) if not h[3]]
        found = {h[0] for h in winners}
        need = surface
        add("V7", None, "$ (recursive cfg scan, winners only)",
            f"at least one of {sorted(need)} present", sorted(found),
            "PASS" if (found & need) else "FAIL",
            "if the re-measured Hopper arm selects none of the axes it is offered, it is "
            "not measuring what it claims and the whole comparison is meaningless")
        ovl_ok = any(bool(_jget(payload, "fairness", "h200_axes", "per_family", k,
                                "overlays_offered")) for k in pkeys)
        add("V3", None, f"{axes_root}.*.overlays_offered", "non-empty", ovl_ok,
            "PASS" if ovl_ok else "FAIL", "the widened grid must exist on this arm")
        n_fail = sum(1 for c in checks if c.status == "FAIL")
        return {"family": fam_key, "arm": arm.name,
                "verdict": "FAILED" if n_fail else "ENGAGED",
                "checks": [asdict(c) for c in checks], "n_fail": n_fail,
                "surface": sorted(surface), "surface_source": surface_src,
                "vacuous_axes": [], "engaged_axes": sorted(found & need),
                "target_axes": sorted(surface)}

    # ---- V1/V2/V4: CAPABILITY-level, non-vacuous even for f03 and f10 ------------------
    # These are what prove engagement for a Hopper-BLIND family. f03/f10 have no cfg key, so
    # every config-level check is trivially clean on both arms -- but their `available` flag
    # and their `not_offered_because` string BOTH change under the env override, because
    # `bench/__init__.py:axis_available()` reads `hopper.caps()`, which the env reaches.
    for axis in sorted(arm.axes_off):
        for k in pkeys:
            base = ("fairness", "h200_axes", "per_family", k, "axes", axis)
            avail = _jget(payload, *base, "available")
            evid = str(_jget(payload, *base, "evidence") or "")
            ok = (avail is False) and (EVIDENCE_ENV_MARKER in evid)
            add("V1", axis, f"{axes_root}.{k}.axes.{axis}.available + .evidence",
                f"available=False and evidence containing '{EVIDENCE_ENV_MARKER}'",
                f"available={avail}, evidence={evid[:120]!r}",
                "PASS" if ok else "FAIL",
                "the ONLY per-family text proving the env override reached hopper.caps(); "
                "the campaign reads available=true with '(source preflight)'. "
                "Producer: glm52_h200/bench/__init__.py:1173")
            nob = _jget(payload, *base, "not_offered_because")
            off = _jget(payload, *base, "offered")
            ok2 = (off is False) and (nob == REASON_UNAVAILABLE)
            add("V2", axis, f"{axes_root}.{k}.axes.{axis}.offered + .not_offered_because",
                f"offered=False and not_offered_because == {REASON_UNAVAILABLE!r}",
                f"offered={off}, not_offered_because={nob!r}",
                "PASS" if ok2 else "FAIL",
                "non-vacuous for f03/f10 too: their campaign string is "
                f"{REASON_NO_CFG_KEY!r} and it MUST change. "
                "Producer: glm52_h200/bench/__init__.py:1271-1274")

    for k in pkeys:
        if "tma" in arm.axes_off:
            v = _jget(payload, "fairness", "h200_axes", "per_family", k, "tma_form")
            add("V4", "tma", f"{axes_root}.{k}.tma_form", "none", v,
                "PASS" if v == "none" else "FAIL",
                "scalar mirror of the capability; the campaign records 'device' in all "
                "seven files. hopper.py:377 tma_form()")
        if "warp_specialize" in arm.axes_off:
            v = _jget(payload, "fairness", "h200_axes", "per_family", k, "ws_mode")
            add("V4", "warp_specialize", f"{axes_root}.{k}.ws_mode", "none", v,
                "PASS" if v == "none" else "FAIL",
                "scalar mirror; the campaign records 'range'. hopper.py:937 sets "
                "ws_mode = 'range' if warp_specialize else 'none'")
        if "clusters" in arm.axes_off:
            add("V4", "clusters", f"{axes_root}.{k}", "n/a", "no scalar mirror exists",
                "INFO", "clusters has no tma_form/ws_mode equivalent; recorded so the "
                        "check list is complete rather than silently short")

    # ---- V3: the widened overlay list ---------------------------------------------------
    for k in pkeys:
        got = _jget(payload, "fairness", "h200_axes", "per_family", k, "overlays_offered")
        was = _jget(campaign, "fairness", "h200_axes", "per_family", k, "overlays_offered")
        got_l = got if isinstance(got, list) else []
        bad = [o for o in got_l if isinstance(o, dict)
               and any(CFG_KEY_AXIS.get(ck) in arm.axes_off for ck in o)]
        if isinstance(was, list) and not was:
            add("V3", None, f"{axes_root}.{k}.overlays_offered", "[] (already empty)",
                got_l, "VACUOUS",
                "the campaign already offered no overlay for this kernel; an empty list "
                "here shows nothing")
        else:
            want = "[]" if arm.name == "classic" else "no overlay carrying a disabled axis"
            add("V3", None, f"{axes_root}.{k}.overlays_offered", want, got_l,
                "PASS" if not bad and not (arm.name == "classic" and got_l) else "FAIL",
                f"campaign had {len(was) if isinstance(was, list) else '?'} overlay(s); "
                f"h200_cfg_overlays() returns [] once every axis is unavailable")

    # ---- V5: static ground truth must NOT have changed ----------------------------------
    for k in pkeys:
        got = _jget(payload, "fairness", "h200_axes", "per_family", k, "kernel_cfg_keys")
        was = _jget(campaign, "fairness", "h200_axes", "per_family", k, "kernel_cfg_keys")
        if was is None:
            add("V5", None, f"{axes_root}.{k}.kernel_cfg_keys", "(campaign value)", got,
                "INFO", "the campaign file carries no value for this key to compare against")
        else:
            same = json.dumps(got, sort_keys=True) == json.dumps(was, sort_keys=True)
            add("V5", None, f"{axes_root}.{k}.kernel_cfg_keys", was, got,
                "PASS" if same else "FAIL",
                "a STATIC module attribute (H200_CFG_KEYS). If it changed, somebody edited "
                "the kernel modules between the two sessions and the arm comparison is not "
                "a comparison")

    # ---- V6: the device fence -----------------------------------------------------------
    dev = R._norm_dev(_jget(payload, "_meta", "device"))   # a STRING, not a dict
    uuid_got = R._norm_uuid(_jget(payload, "env", "uuid"))
    uuid_want = R._norm_uuid(gpu.get("uuid"))
    dev_ok = (not device_fence) or (not device_name) or (dev == R._norm_dev(device_name))
    add("V6", None, "$._meta.device", device_name or "(unknown)", dev,
        "PASS" if dev_ok else "FAIL",
        "note _meta.device is a string; indexing it as a dict throws TypeError")
    uuid_ok = (not device_fence) or (not uuid_want) or (not uuid_got) \
        or (uuid_got == uuid_want)
    add("V6", None, "$.env.uuid", uuid_want or "(no pinned uuid)", uuid_got or "(absent)",
        "PASS" if uuid_ok else "FAIL",
        "the staged file must name the card this driver pinned")

    # ---- V7: zero tokens in any winning mapping, fused AND unfused, all regimes ---------
    hits = scan_hopper_tokens(payload)
    win: dict[str, list[tuple[str, object]]] = {a: [] for a in AXIS_NAMES}
    tab: dict[str, list[tuple[str, object]]] = {a: [] for a in AXIS_NAMES}
    for axis, jpath, val, in_tab in hits:
        (tab if in_tab else win)[axis].append((jpath, val))
    for axis in sorted(arm.axes_off):
        nw, nt = len(win.get(axis, [])), len(tab.get(axis, []))
        ex = (win.get(axis) or tab.get(axis) or [("", "")])[0][0]
        clean = (nw == 0 and nt == 0)
        if axis in vacuous:
            add("V7", axis, "$ (recursive cfg scan)", "0 winner and 0 table hits",
                f"{nw} winner, {nt} table", "VACUOUS" if clean else "FAIL",
                "the campaign never offered this axis to this family; a clean scan shows "
                "nothing" if clean else f"unexpected token at {ex}")
        else:
            add("V7", axis, "$ (recursive cfg scan)", "0 winner and 0 table hits",
                f"{nw} winner, {nt} table", "PASS" if clean else "FAIL",
                "winners prove the tuner never selected it; table hits prove the grid never "
                "contained it. Under classic widen() returns [] so neither can appear"
                + ("" if clean else f"; first offender {ex}"))

    # ---- V8: axis_counts key absence ----------------------------------------------------
    # `axis_counts` emits a key ONLY when a truthy value is seen and always appends
    # `_total_cfgs`, so under classic every dict collapses to exactly {"_total_cfgs": N}.
    # This is the most reliable engagement assertion in the corpus: a live count of what the
    # tuner was actually handed, not a claim about it. Joint/size-only stages carry only
    # {n_tried} -- `axis_counts` ABSENT, not null -- and a missing dict is "not recorded",
    # never a violation.
    counted = 0
    offenders: dict[str, list[str]] = {a: [] for a in AXIS_NAMES}
    for regime, armk, stage, sd in _grid_stages(payload):
        ac = sd.get("axis_counts")
        if not isinstance(ac, dict):
            continue
        counted += 1
        for axis, cfg_key, count_key in AXES:
            if axis in arm.axes_off and count_key in ac:
                offenders[axis].append(
                    f"$.fairness.grids.{regime}.{armk}.{stage}.axis_counts.{count_key}"
                    f"={ac[count_key]}")
    for axis in sorted(arm.axes_off):
        bad = offenders[axis]
        if counted == 0:
            add("V8", axis, "$.fairness.grids.*.*.*.axis_counts", "no key for this axis",
                "no axis_counts dict was recorded anywhere", "INFO",
                "this family records no per-stage axis_counts; nothing to read")
        elif axis in vacuous:
            add("V8", axis, "$.fairness.grids.*.*.*.axis_counts", "no key for this axis",
                f"{len(bad)} offending stage(s) of {counted}",
                "VACUOUS" if not bad else "FAIL",
                "the campaign never offered this axis to this family; an absent count "
                "shows nothing" if not bad else "; ".join(bad[:3]))
        else:
            add("V8", axis, "$.fairness.grids.*.*.*.axis_counts", "no key for this axis",
                f"{len(bad)} offending stage(s) of {counted}",
                "PASS" if not bad else "FAIL",
                "every dict should collapse to {_total_cfgs: N}"
                if not bad else "; ".join(bad[:3]))

    # ---- V9: OFFERED-grid corroboration, INFO only unless f03/f10 ----------------------
    # Coarse `_total_cfgs` is pinned at each bench's COARSE_CAP (f01 220, f06/f08f09 200,
    # f11 180, f04f05 router 240), so shrinkage is NOT predictable there. f03 (164) and f10
    # (188) are UNCAPPED base grids and must be identical across arms -- and that is the one
    # grid check that is meaningful for the two noise-floor families, so for them it FAILS.
    #
    # WHAT THIS CHECK MAY AND MAY NOT READ, and why the distinction cost a round trip.
    # Only the stages named in OFFERED_GRID_STAGES -- `coarse` and `tune`, the grid the bench
    # OFFERS, a pure function of the code -- are compared. Everything else (`refine`, `joint`,
    # `extra`) is built from the COARSE STAGE'S TIMING WINNER: `glm52_h200/common.py:1069`
    # takes the winner of `_stage(coarse, "coarse")` and expands it through
    # `neighbours(best_cfg, coarse)` -- or the bench's own `refine_grid(tc.best_cfg)` -- so an
    # edge winner yields a strictly smaller neighbourhood than an interior one. Their size is
    # downstream of a measurement and is NOT reproducible even Hopper-vs-Hopper.
    #
    # DO NOT "RESTORE" THE STRICT COMPARISON OVER ALL STAGES. The 2026-08-11 control arm
    # was aborted by this check reporting "19 of 49 stage(s) differ" on f03 -- 14 refine and
    # 5 joint, zero coarse, and moving in both directions (88->50 but also 28->46). The same
    # comparison run between `results/h200/_bs_extra_rerun/f03_resadd_rmsnorm.json` and the
    # campaign -- two Hopper arms, same code, same card -- reports 16 of 49, so the old check
    # would have failed the campaign against itself. It asserted bit-determinism of a
    # timing-derived quantity.
    #
    # The discriminating power is not weakened by the narrowing: the OFFERED grid is exactly
    # what changes if the harness changed or if an overlay was applied, and the coarse/tune
    # sizes were identical in 231 of 231 shared stages over that pair AND in 21 of 21 for the
    # real classic arm. V5 covers the static module attribute; V8 covers the per-axis live
    # counts inside the same stages; V9b below covers the offered stages the intersection
    # drops, so the narrowing is not a net weakening.
    grid_now = {f"{r}/{a}/{s}": (s, sd.get("n_tried"))
                for r, a, s, sd in _grid_stages(payload)}
    grid_was = {f"{r}/{a}/{s}": (s, sd.get("n_tried"))
                for r, a, s, sd in _grid_stages(campaign)}
    both = sorted(set(grid_now) & set(grid_was))
    # A stage present on only ONE side is NO COMPARISON, never a disagreement: the campaign
    # `fairness.grids` block covers the 7 original regimes only -- decode_bs2/4/8/16 were
    # merged in from `run_bs_extra_h200.py`, which carried rows but no grids block -- and
    # every arm here measures 11. Same for an `n_tried` that was never recorded.
    offered = [k for k in both
               if grid_now[k][0] in OFFERED_GRID_STAGES
               and grid_now[k][1] is not None and grid_was[k][1] is not None]
    unrecorded = [k for k in both
                  if grid_now[k][0] in OFFERED_GRID_STAGES
                  and (grid_now[k][1] is None or grid_was[k][1] is None)]
    derived = [k for k in both if grid_now[k][0] not in OFFERED_GRID_STAGES]
    changed = [k for k in offered if grid_now[k][1] != grid_was[k][1]]
    drifted = [k for k in derived if grid_now[k][1] != grid_was[k][1]]
    strict = fam_key in ("f03", "f10")
    if not offered:
        add("V9", None, "$.fairness.grids.*.*.<coarse|tune>.n_tried",
            "identical to the campaign", "no offered-grid stage on both sides", "INFO",
            "NO COMPARISON AVAILABLE: this family records no coarse/tune stage that the "
            "campaign baseline also records, so there is no reproducible grid size to "
            "compare. Absence is not agreement and is not disagreement.")
    else:
        add("V9", None, "$.fairness.grids.*.*.<coarse|tune>.n_tried",
            "identical to the campaign" if strict else "(corroboration only)",
            f"{len(changed)} of {len(offered)} offered-grid stage(s) differ",
            ("FAIL" if (strict and changed) else ("PASS" if strict else "INFO")),
            # The wording MUST branch on `strict`. "no arm switch can do this" is true only
            # for f03/f10, which are offered no axis and whose widen() is a no-op. For the
            # five overlay families a disabling arm is SUPPOSED to shrink the offered grid
            # -- bench/__init__.py widen() stops adding the Hopper overlays -- so printing
            # "which no arm switch can do" there is exactly the authoritative-but-false
            # phrasing that made the V9 false abort read as decisive (LOG-18, 2026-08-11).
            (("the OFFERED grid changed, which no arm switch can do for a family that is "
              "offered no Hopper axis -- " if strict else
              ("the offered grid SHRANK, which is what this arm predicts (widen() stops "
               "adding the Hopper overlays) -- " if all(
                   grid_now[k][1] < grid_was[k][1] for k in changed) else
               "the offered grid changed, not uniformly a shrink -- inspect before "
               "trusting this arm -- "))
             + "; ".join(f"{k} {grid_was[k][1]}->{grid_now[k][1]}" for k in changed[:4]))
            if changed else
            ("the offered grid is byte-identical to the campaign's; a capped coarse grid "
             "cannot shrink, so this is corroboration, not proof"
             + (f" ({len(unrecorded)} further offered stage(s) carry no n_tried on one "
                f"side and were not compared)" if unrecorded else "")))

    # ---- V9b: the offered stages the INTERSECTION DROPS ---------------------------------
    # V9 above compares `set(now) & set(was)`, and the campaign's `fairness.grids` block
    # covers only the 7 original regimes: decode_bs2/4/8/16 were merged in from
    # `run_bs_extra_h200.py`, which carried rows but no grids block, while every arm here
    # measures 11. For f03 that silently drops 28 staged stages, 12 of them COARSE -- the one
    # genuinely deterministic quantity in the whole block. Narrowing V9 without closing this
    # hole would leave an arm whose bs-extra coarse grid had been HALVED passing every grid
    # check, i.e. the repair would be a net weakening.
    #
    # A per-regime value the campaign never recorded cannot be compared against a per-regime
    # value; asserting on an absent record is precisely the failure mode this whole check is
    # being repaired for. So V9b compares against something the campaign DID record: the
    # family's (sub-arm, stage) CONSTANT. An offered grid is built from the module's
    # advertised cfg keys and the probed device, not from the shape, so it does not vary by
    # regime -- but that is asserted only where the campaign DEMONSTRATES it, by recording one
    # single distinct value over at least two of its own regimes. Where it does not, the stage
    # is reported as having no campaign record to compare against and is NOT judged. That
    # escape hatch is load-bearing, not defensive boilerplate: f08f09's `tokmaj8/coarse` is
    # genuinely 30 on some regimes and 243 on others.
    #
    # Validated on real data: run with the campaign as baseline and the 11-regime Hopper
    # `_bs_extra_rerun` as the arm -- which is exactly this case, four regimes the campaign
    # has no grids for -- V9b judges every extrapolable bs-extra stage CORRECT for all seven
    # families and raises zero false positives.
    camp_const: dict[tuple[str, str], set] = {}
    camp_regimes: dict[tuple[str, str], set] = {}
    for _r, _a, _s, _sd in _grid_stages(campaign):
        if _s in OFFERED_GRID_STAGES and _sd.get("n_tried") is not None:
            camp_const.setdefault((_a, _s), set()).add(_sd.get("n_tried"))
            camp_regimes.setdefault((_a, _s), set()).add(_r)
    v9b_ok: list[str] = []
    v9b_bad: list[str] = []
    v9b_none: list[str] = []
    for _r, _a, _s, _sd in _grid_stages(payload):
        if _s not in OFFERED_GRID_STAGES or f"{_r}/{_a}/{_s}" in grid_was:
            continue                     # winner-derived, or V9 already compared it directly
        got = _sd.get("n_tried")
        if got is None:
            continue
        seen = camp_const.get((_a, _s)) or set()
        if len(seen) == 1 and len(camp_regimes.get((_a, _s), ())) >= 2:
            want = next(iter(seen))
            (v9b_ok if got == want else v9b_bad).append(f"{_r}/{_a}/{_s} {want}->{got}")
        else:
            v9b_none.append(f"{_r}/{_a}/{_s}")
    if v9b_ok or v9b_bad:
        add("V9b", None, "$.fairness.grids.<arm-only regime>.*.<coarse|tune>.n_tried",
            "equal to the campaign's family-constant offered size",
            f"{len(v9b_bad)} of {len(v9b_ok) + len(v9b_bad)} arm-only offered stage(s) differ",
            ("FAIL" if (strict and v9b_bad) else ("PASS" if not v9b_bad else "INFO")),
            ("the offered grid changed on a regime the campaign has no grids block for -- "
             + "; ".join(v9b_bad[:4])) if v9b_bad else
            ("closes the hole the V9 intersection leaves: these stages have no per-regime "
             "campaign counterpart, but their size is a code property the campaign pins on "
             "its own regimes, and the arm reproduces it"
             + (f"; {len(v9b_none)} further stage(s) had no campaign constant to extrapolate "
                f"and were not judged" if v9b_none else "")))
    else:
        add("V9b", None, "$.fairness.grids.<arm-only regime>.*.<coarse|tune>.n_tried",
            "equal to the campaign's family-constant offered size",
            f"0 of 0 judged, {len(v9b_none)} not extrapolable", "INFO",
            "NO CAMPAIGN RECORD TO COMPARE AGAINST: the arm records no offered stage outside "
            "the campaign's own regimes whose size the campaign pins to a single value over "
            "at least two regimes. Absence is not agreement and is not disagreement.")

    if derived:
        add("V9d", None, "$.fairness.grids.*.*.<refine|joint|extra>.n_tried",
            "(not comparable -- winner-derived)",
            f"{len(drifted)} of {len(derived)} stage(s) differ", "INFO",
            "refine is neighbours(<the coarse stage's timing winner>) (common.py:1069), "
            "joint is a deduplicated top_cfgs() product over measured stages, extra is "
            "refine_grid() over a measured top-k -- so these sizes move run-to-run on "
            "IDENTICAL code and hardware (Hopper-vs-Hopper: f03 16 of 28, f10 18 of 28, in "
            "both directions). Recorded so the drift stays visible, never asserted on. "
            + "; ".join(f"{k} {grid_was[k][1]}->{grid_now[k][1]}" for k in drifted[:4]))

    # ---- V10: f11's caps dump, the strongest smoking gun in the corpus ------------------
    if fam_key == "f11":
        seen = 0
        v10_fail: list[str] = []
        note_seen = 0
        for jpath, caps in _f11_caps_reports(payload):
            seen += 1
            srcs = caps.get("sources") if isinstance(caps.get("sources"), dict) else {}
            for cap in sorted(arm.caps_off):
                if cap not in srcs:
                    continue           # a missing key is INFO, not a failure
                if srcs.get(cap) != "env":
                    v10_fail.append(f"{jpath}.sources.{cap}={srcs.get(cap)!r}")
            if arm.name == "classic":
                if caps.get("any_hopper") is not False:
                    v10_fail.append(f"{jpath}.any_hopper={caps.get('any_hopper')!r}")
                notes = caps.get("notes")
                if isinstance(notes, list) and any(CLASSIC_CAPS_NOTE in str(n)
                                                   for n in notes):
                    note_seen += 1
        if seen == 0:
            add("V10", None, "$.rows[*]...kernel_caps_report.hopper_caps",
                "sources[cap]=='env'", "no caps report present", "INFO",
                # Do NOT read this as a crash. The dumps are attached by
                # bench_f11_lazy_prenorm.py's _ws_isolation, and under a warp-spec-disabling
                # arm that function returns at its `ws_offered()` guard BEFORE the
                # kernel_caps_report block runs -- so a HEALTHY classic f11 has zero dumps
                # by construction. Saying "the bench may have died" here would send the
                # operator hunting a failure that did not happen, which is the same
                # authoritative-but-false phrasing that made the V9 abort read as decisive.
                "f11 records 11 router and 8 moe caps dumps in the campaign; none here. "
                "Under an arm that disables warp specialization this is EXPECTED, not a "
                "crash: bench_f11_lazy_prenorm.py's _ws_isolation returns at its "
                "ws_offered() guard before the kernel_caps_report block, so the dumps are "
                "never attached. V10 is therefore structurally vacuous on this arm and "
                "engagement rests on V1/V2/V4/V7/V8/V11, which are not")
        else:
            add("V10", None, "$.rows[*]...kernel_caps_report.hopper_caps.sources",
                "'env' for " + ", ".join(sorted(arm.caps_off)),
                f"{len(v10_fail)} disagreement(s) over {seen} dump(s)",
                "PASS" if not v10_fail else "FAIL", "; ".join(v10_fail[:4]))
            if arm.name == "classic":
                add("V10", None, "$.rows[*]...hopper_caps.notes", CLASSIC_CAPS_NOTE,
                    f"present in {note_seen}/{seen} dump(s)",
                    "PASS" if note_seen else "FAIL",
                    "kernels/hopper.py:723 appends this note only under "
                    "GLM52_H200_CLASSIC=1")
        add("V10", "clusters", "lazy_prenorm.AXES_DECLINED", "unchanged", "declined",
            "INFO",
            "f11 already withholds the cluster axis from BOTH arms for a verified CGALayout "
            "reason, so only warp_specialize can change for f11; expecting all three to "
            "flip would produce a spurious failure")

    # ---- V11: differential collapse -- the positive engagement proof --------------------
    camp_hits = scan_hopper_tokens(campaign) if campaign is not None else []
    camp_win: dict[str, int] = {a: 0 for a in AXIS_NAMES}
    for axis, _p, _v, in_tab in camp_hits:
        if not in_tab:
            camp_win[axis] += 1
    camp_counts: dict[str, int] = {a: 0 for a in AXIS_NAMES}
    for _r, _a, _s, sd in _grid_stages(campaign):
        ac = sd.get("axis_counts")
        if isinstance(ac, dict):
            for axis, _ck, count_key in AXES:
                if count_key in ac:
                    camp_counts[axis] += 1
    engaged: list[str] = []
    for axis in sorted(target):
        before = camp_win.get(axis, 0) + camp_counts.get(axis, 0)
        after = len(win.get(axis, [])) + len(tab.get(axis, [])) \
            + len(offenders.get(axis, []))
        if before == 0:
            add("V11", axis, "campaign vs this arm", "campaign presence > 0 and arm == 0",
                f"campaign 0, arm {after}", "INFO",
                "NO-DIFFERENCE-AVAILABLE: the campaign's tuner never selected this axis "
                "anyway, so turning it off cannot change anything for this family")
        elif after == 0:
            engaged.append(axis)
            add("V11", axis, "campaign vs this arm", "campaign presence > 0 and arm == 0",
                f"campaign {before}, arm 0", "PASS",
                "a genuine campaign->arm collapse: this is the positive proof of engagement")
        else:
            add("V11", axis, "campaign vs this arm", "campaign presence > 0 and arm == 0",
                f"campaign {before}, arm {after}", "FAIL",
                "the axis survived into this arm; it did not engage")

    # ---- child-log corroboration, INFO only ---------------------------------------------
    # Reads common.banner()'s `hopper feats [...]`, which every bench prints as its first
    # line and which the committed campaign transcript demonstrably contains
    # (log/run_h200/f03.log:3). See FEATS_NEEDLE for what this replaced and why the old
    # needles could never match.
    n_banner, feats = feats_in_child_logs(child_logs)
    want_gone = frozenset(arm.axes_off) & frozenset(FEATS_AXES)
    still_on = sorted({a for _n, s in feats for a in (s & want_gone)})
    if n_banner == 0:
        add("LOG", None, "; ".join(p.name for p in child_logs) or "(no child log)",
            f"a line containing {FEATS_NEEDLE!r}", f"0 of {len(child_logs)} log(s)", "INFO",
            "corroboration only; it survives a bench that died before writing JSON. No "
            "child log carried the harness banner -- either none ran yet, or the bench died "
            "before common.banner() printed.")
    else:
        add("LOG", None, "; ".join(n for n, _s in feats),
            f"{FEATS_NEEDLE}...] without {sorted(want_gone) or 'any change'}",
            "; ".join(f"{n}: [{','.join(sorted(s)) or 'none'}]" for n, s in feats)[:400],
            "INFO",
            (f"corroboration only. {len(still_on)} banner(s) still report "
             f"{', '.join(still_on)} live, which contradicts this arm"
             if still_on else
             f"every one of {n_banner} banner(s) reports the axes this arm disables as off; "
             f"the campaign's own transcript reads [tma,clusters,warp_specialize]"))

    n_fail = sum(1 for c in checks if c.status == "FAIL")
    if n_fail:
        verdict = "FAILED"
    elif not target:
        verdict = "NOTHING-TO-DISABLE"
    elif engaged:
        verdict = "ENGAGED"
    else:
        verdict = "NO-DIFFERENCE-AVAILABLE"
    return {
        "family": fam_key, "arm": arm.name, "verdict": verdict,
        "checks": [asdict(c) for c in checks], "n_fail": n_fail,
        "surface": sorted(surface), "surface_source": surface_src,
        "vacuous_axes": sorted(vacuous), "target_axes": sorted(target),
        "engaged_axes": engaged,
    }


def verdict_line(rec: dict) -> str:
    """The one line the operator reads for a family."""
    fam, arm, v = rec["family"], rec["arm"], rec["verdict"]
    if v == "ENGAGED":
        return (f"[verify] {fam}/{arm} ENGAGED: "
                f"{', '.join(rec['engaged_axes'])} collapsed")
    if v == "NOTHING-TO-DISABLE":
        return (f"[verify] {fam}/{arm} NOTHING-TO-DISABLE (no cfg key; classic == hopper by "
                f"construction). Capability-level checks V1/V2/V4 still passed, which is "
                f"what proves the env var reached this child.")
    if v == "NO-DIFFERENCE-AVAILABLE":
        return (f"[verify] {fam}/{arm} NO-DIFFERENCE-AVAILABLE: the campaign's tuner never "
                f"selected {', '.join(rec['target_axes'])} for this family anyway")
    if v == "NOT-APPLICABLE":
        return f"[verify] {fam}/{arm} NOT-APPLICABLE (no Hopper axis is offered here)"
    return f"[verify] {fam}/{arm} FAILED with {rec['n_fail']} fatal check(s)"


def sentinel_text(arm: Arm, fam_key: str, rec: dict) -> str:
    fails = [c for c in rec["checks"] if c["status"] == "FAIL"]
    lines = [
        f"ARM NOT VERIFIED -- {arm.name}",
        f"written {time.strftime('%Y-%m-%d %H:%M:%S')} by run_control_h200.py",
        f"family that failed: {fam_key}",
        "",
        "failed checks:",
    ]
    for c in fails:
        lines.append(f"  {c['id']:4s} axis={c['axis']}  {c['path']}")
        lines.append(f"        want {c['want']}")
        lines.append(f"        got  {c['got']}")
        if c.get("detail"):
            lines.append(f"        {c['detail']}")
    lines += [
        "",
        "the staged JSON in this directory is NOT a verified control arm and must not be",
        "published. It has deliberately NOT been moved, renamed or deleted: a control arm",
        "that did not engage is evidence, and this sentinel is the fence that replaces",
        "quarantining. glm52/make_control_report_h200.py refuses to publish this arm while",
        "this file exists.",
        "",
    ]
    return "\n".join(lines)


# ======================================================================================
# per-family execution
# ======================================================================================
def launch(log: R.Log, args: argparse.Namespace, arm: Arm, key: str, title: str,
           script: Path, timeout_s: int, regimes: list[str], staging: Path,
           logdir: Path, extra: list[str], gpu: dict, note: str = "") -> dict:
    """One supervised child, through `run_h200.run_family`, with the arm applied."""
    fam = R.Family(key=key, title=title, script_globs=(script.name,), result_globs=(),
                   timeout_s=timeout_s, note=note)
    sub = argparse.Namespace(**vars(args))
    sub.regimes = ",".join(regimes)
    sub.disable_features = arm.disable_features        # <- the ONLY place an arm is applied
    extra = list(extra)
    if "--regimes" not in R.script_flags(script):
        # The benches register --regimes inside add_std_args() (glm52_h200/bench/__init__.py);
        # run_h200.script_flags() only scans the bench's own source and cannot see it, so
        # run_family falls back to the GLM52_REGIMES env, which NO bench reads. Last time
        # that silently measured all 11 regimes into the staging tree. The explicit flag is
        # the only thing that works.
        extra += ["--regimes", ",".join(regimes)]
    rec = R.run_family(log, fam, script, sub, staging, logdir, extra, gpu)
    # `run_h200.run_family` (run_h200.py:1141-1147, :1229-1235) decides a flag was UNHONOURED
    # purely from `script_flags()`, which greps the bench's own source and so misses every
    # flag registered in `bench/__init__.py:add_std_args()`. It therefore logs
    # "<bench> advertises no --regimes flag; the request was passed only via GLM52_REGIMES
    # and may be ignored by this bench" and stamps `unhonoured_flags: ["--regimes"]` into the
    # summary -- for a bench that DOES accept --regimes, that we DID put on argv four lines
    # up, and that ran every regime we asked for. On 2026-08-11 that false warning sat in
    # `control_arm_summary.json` next to a real abort and made regime coverage look doubtful
    # when all 11 regimes had rows. Correct the record from the argv the record itself
    # carries -- never from our own bookkeeping, and never for any other flag.
    if isinstance(rec, dict) and isinstance(rec.get("unhonoured_flags"), list):
        argv = [str(x) for x in (rec.get("cmd") or [])]
        if "--regimes" in rec["unhonoured_flags"] and "--regimes" in argv:
            rest = [f for f in rec["unhonoured_flags"] if f != "--regimes"]
            if rest:
                rec["unhonoured_flags"] = rest
            else:
                rec.pop("unhonoured_flags", None)
            rec["regimes_flag"] = ("passed explicitly on argv; add_std_args registers it, "
                                   "so script_flags() cannot see it by static scan")
            log("  (the --regimes warning above is a static-scan artefact: the flag IS on "
                "this child's argv and IS accepted via add_std_args)")
    return rec


def run_family_stage(log: R.Log, args: argparse.Namespace, arm: Arm, fam: R.Family,
                     script: Path, results: Path, staging: Path, logdir: Path,
                     gpu: dict, scope: list[str], extra: list[str],
                     warnings: list[str], fp: dict[str, dict] | None = None,
                     first_stage: bool = False) -> dict:
    """Drive one family's measurement for one arm into the staging tree, resumable.

    THE INVARIANT: an attempt is handed only the regimes still in scope. The benches
    checkpoint per regime into `<staging>/_ckpt`, so a completed regime is replayed from its
    checkpoint on the next attempt and only the unfinished ones are re-measured. A hard abort
    that made no progress has the regime it died on quarantined out of the next attempt. Both
    moves are monotone, so the loop is bounded by `len(scope)` whichever way the bench fails.
    """
    R.rule(log, f"{arm.name}/{fam.key} -- {fam.title} over {len(scope)} regime(s)")
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

        rec = launch(log, args, arm, f"{fam.key}.{arm.name}.a{n}", fam.title, script,
                     args.timeout or fam.timeout_s, todo, staging, logdir, extra, gpu,
                     note=fam.note)
        rec["attempt"] = n
        rec["arm"] = arm.name
        rec["regimes_requested"] = todo
        attempts.append(rec)

        # -- the campaign canary, after EVERY ATTEMPT ---------------------------------
        # Not after every FAMILY: `max_attempts` is len(scope) + 2 = 13, and f08f09 carries a
        # 14 h per-attempt timeout, so a bench that ignores GLM52_H200_RESULTS_DIR could
        # overwrite the baseline on attempt 1 and be relaunched twelve more times before a
        # family-level canary ever ran. Hashing the 28 campaign files is ~65 MB of sha256,
        # well under a second, and it caps the blast radius at one attempt.
        if fp is not None:
            canary_or_raise(log, fp, results, warnings, f"{arm.name}/{fam.key} attempt {n}")

        # -- was the staging redirect honoured? PRE-COMMITMENT, on the first attempt ----
        # `--results-dir` is invisible to R.script_flags() for six of the seven benches, so
        # the redirect rests entirely on the env var and nothing verified it before this run
        # spent the whole arm on it. V0b re-checks the same key per family at verify time; it fires
        # after the FIRST attempt of the FIRST family, before anything else is launched.
        if first_stage and n == 1:
            check_results_dir_honoured(log, result_path, staging, warnings)

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
                                 f"({rec.get('status')}) while measuring it, with no clean "
                                 f"row")
            warnings.append(f"{arm.name}/{fam.key} regime {died_on} was quarantined: "
                            f"{poisoned[died_on]}")
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
        "family": fam.key, "arm": arm.name, "result": str(result_path),
        "result_error": err or None, "attempts": attempts, "poisoned": poisoned,
        "regimes_done": done, "regimes_missing": missing, "rows_mult": mult,
        "wall_s": sum(float(a.get("wall_s") or 0.0) for a in attempts),
    }


def campaign_wall_s_from(summary: dict | None, fam_key: str) -> float | None:
    """The campaign's own wall time for a family, from an already-parsed summary.json.

    `families` is a dict in the campaign's summary and a list in older ones; both shapes are
    handled rather than assumed, because a driver that raises on a shape it did not expect
    loses a whole arm over a sanity check it did not need to perform.
    """
    fams = (summary or {}).get("families")
    if isinstance(fams, dict):
        v = (fams.get(fam_key) or {}).get("wall_s")
    elif isinstance(fams, list):
        v = next((f.get("wall_s") for f in fams
                  if isinstance(f, dict) and f.get("family") == fam_key), None)
    else:
        v = None
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ======================================================================================
# the closing status table -- no cell is ever blank
# ======================================================================================
def status_table(arm_recs: dict, scope: list[str], arms: list[Arm]) -> list[str]:
    """arm x family x regime, with an explicit token in every cell.

    A blank cell in a table an operator reads once, six hours away, is indistinguishable
    from a 1.000x. Every cell carries a token and the legend lists only the tokens that
    actually appeared.
    """
    width = 9
    used: set[str] = set()
    lines: list[str] = []
    for arm in arms:
        rec = arm_recs.get(arm.name) or {}
        lines.append("")
        lines.append(f"  arm: {arm.name}"
                     + ("" if rec.get("verified", True)
                        else "   [ARM_NOT_VERIFIED -- not publishable]"))
        lines.append("  " + f"{'family':<10}"
                     + "".join(R.REGIME_ABBR.get(r, r).rjust(width) for r in scope))
        lines.append("  " + "-" * (10 + len(scope) * width))
        cells = rec.get("cells") or []
        best: dict[tuple[str, str], dict] = {}
        for c in cells:
            k = (c.get("family"), c.get("regime"))
            cur = best.get(k)
            if cur is None or ((c.get("speedup") or -1) > (cur.get("speedup") or -1)):
                best[k] = c
        for key in FAMILY_KEYS:
            st = (rec.get("fam_stages") or {}).get(key) or {}
            eng = (rec.get("engagement_summary") or {}).get(key)
            row = f"  {key:<10}"
            for r in scope:
                cell = best.get((key, r))
                if eng == "FAILED":
                    tok = "UNVER"
                elif not st:
                    tok = "NOTRUN"
                elif r in (st.get("poisoned") or {}):
                    tok = "QUAR"
                elif cell is None:
                    tok = "MISS" if r in (st.get("regimes_missing") or []) else "-"
                elif cell.get("speedup") is None:
                    tok = "UNRES"
                else:
                    tok = f"{cell['speedup']:.3f}"
                used.add(tok if not tok[0].isdigit() else "speedup")
                row += tok.rjust(width)
            lines.append(row)
    legend = {
        "speedup": "a number = the staged arm's measured speedup for that cell",
        "MISS": "MISS   = the regime was in scope but produced no complete row",
        "NOTRUN": "NOTRUN = the family was not launched in this arm",
        "UNVER": "UNVER  = engagement verification FAILED; these numbers must not be "
                 "published",
        "QUAR": "QUAR   = the regime was quarantined after a hard abort",
        "UNRES": "UNRES  = the two arms differ by fewer timer ticks than --unresolved-ticks",
        "-": "-      = not measured and not expected (outside this run's scope)",
    }
    lines.append("")
    for tok in ("speedup", "MISS", "NOTRUN", "UNVER", "QUAR", "UNRES", "-"):
        if tok in used:
            lines.append("  " + legend[tok])
    return lines


# ======================================================================================
# CLI
# ======================================================================================
def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="run_control_h200.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="The Hopper CONTROL ARM: re-measure the H200 fusion suite with the "
                    "sm_90 levers forced off, into a staging tree that is DIFFED against "
                    "results/h200 and never merged into it.",
        epilog="Typical use:\n"
               "  python3 run_control_h200.py                       "
               "# the control arm, all 7 families, 11 regimes\n"
               "  python3 run_control_h200.py --list                "
               "# print the plan, launch nothing\n"
               "  python3 run_control_h200.py --arms classic,hopper "
               "# control + same-session baseline (2x57h)\n"
               "  python3 run_control_h200.py --arms no-tma,no-ws,no-clusters   "
               "# the follow-up decomposition\n"
               "  python3 run_control_h200.py --verify-only         "
               "# re-check staged data, no GPU work\n"
               "\n"
               "THE HAZARD, stated once: nothing this driver writes may ever be merged into\n"
               "results/h200/*.json. The control arm is a DIFF against those files, they are\n"
               "its baseline, and there is no merge step in this repo -- writing into them\n"
               "would destroy the comparison rather than extend it.\n",
    )
    ap.add_argument("--arms", default=DEFAULT_ARMS,
                    help="comma-separated subset of "
                         + ",".join(a.name for a in ARMS)
                         + f" (default: {DEFAULT_ARMS}). Validated before anything "
                           "launches. 'hopper' is always moved LAST: it is the least "
                           "informative arm and must not consume the GPU before the control "
                           "arm exists.")
    ap.add_argument("--families", default="",
                    help="comma-separated subset of " + ",".join(FAMILY_KEYS)
                         + " (default: all 7). The token 'layer' is REJECTED: bench_layer.py "
                           "carries a 16 h timeout and the per-fusion question does not need "
                           "it.")
    ap.add_argument("--regimes", default="",
                    help="comma-separated subset of " + ",".join(R.KNOWN_REGIMES)
                         + " (default: all 11). Abbreviations (d1, p8192, ...) accepted.")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS),
                    help="the CAMPAIGN directory. READ-ONLY baseline; the staging root "
                         f"<results>/{STAGING_ROOT}/ is built under it and is the only "
                         "thing this driver writes.")
    ap.add_argument("--log-dir", default=str(DEFAULT_LOGDIR),
                    help="driver.log, preflight.log and one log per family per arm")
    ap.add_argument("--gpu", default=None, metavar="N|auto",
                    help="normally OMIT: the driver pins the campaign's own physical card "
                         "by UUID so the cross-session delta is not also a cross-card "
                         "delta. An explicit --gpu that contradicts it is refused.")
    ap.add_argument("--any-gpu", action="store_true",
                    help="do not require the campaign's physical card. Adds a first-class "
                         "DEVICE ANCHOR LOST warning to the summary, which the report "
                         "generator turns into a refusal.")
    ap.add_argument("--on-tenant", choices=("stop", "flag"), default="stop",
                    help="what to do when a neighbour appears on the measurement card "
                         "mid-run. 'stop' (default) ends the arm: this is a DIFF against an "
                         "idle-card baseline, so a co-tenant removes the comparison rather "
                         "than adding noise, and continuing spends GPU hours on numbers that "
                         "must be discarded. 'flag' keeps run_h200.py's campaign policy and "
                         "continues, marking every later family contaminated.")
    ap.add_argument("--allow-busy", action="store_true",
                    help="measure on a card that already has another tenant (this is what "
                         "produced the impossible 40.55 us harness floor; short-kernel "
                         "numbers from such a run are not defensible)")
    ap.add_argument("--allow-concurrent", action="store_true",
                    help="do not refuse when another glm52_h200 bench appears to be running")
    ap.add_argument("--force", action="store_true",
                    help="run even if the device is not sm_90. The result is not an H200 "
                         "measurement and must not go under results/h200.")
    ap.add_argument("--force-rerun", action="store_true",
                    help="NOT NEEDED: the staging tree already isolates _ckpt, so this arm "
                         "cannot see the campaign's checkpoints. Passing it sets "
                         "GLM52_H200_FORCE=1, which reaches the staged checkpoints too and "
                         "restarts the whole arm from zero.")
    ap.add_argument("--timeout", type=int, default=0,
                    help="per-attempt timeout in seconds, overriding the per-family budgets. "
                         "0 keeps them (3/4/8/8/10/10/14 h = 57 h per arm).")
    ap.add_argument("--heartbeat", type=int, default=60,
                    help="seconds between progress lines (default 60)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter used for the benches (default: this one)")
    ap.add_argument("--quick", action="store_true",
                    help="short sweeps, stack smoke test. BARRED FROM A REPORTABLE ARM: the "
                         "driver writes ARM_NOT_VERIFIED into every arm's staging dir so "
                         "the report generator cannot publish it.")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="cap on relaunches per family per arm (default: regimes + 2)")
    ap.add_argument("--flush-mb", type=int, default=0,
                    help="override the L2-flush buffer size in MiB (diagnosis only)")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="do not run the preflight probe even if its JSON is missing")
    ap.add_argument("--no-device-fence", action="store_true",
                    help="DANGEROUS: accept staged results without checking which GPU "
                         "produced them")
    ap.add_argument("--unresolved-ticks", type=int, default=3,
                    help="cells whose two arms differ by fewer CUDA-event ticks than this "
                         "are reported UNRESOLVED in the closing table")
    ap.add_argument("--continue-on-verify-fail", action="store_true",
                    help="do not abort the whole run when engagement verification fails. "
                         "The sentinel is still written and the data still cannot be "
                         "published -- it exists because an operator who already spent the "
                         "measured hours may want the remaining families' evidence anyway.")
    ap.add_argument("--verify-only", action="store_true",
                    help="run no benches: verify whatever is already staged, rewrite the "
                         ".verify.json files and the summary, and exit 0 or 5. Touches the "
                         "GPU only for one nvidia-smi snapshot.")
    ap.add_argument("--list", action="store_true",
                    help="print the plan (arms, families, regimes, staging paths, log keys, "
                         "per-arm timeout budget) and exit 0")
    ap.add_argument("--dry-run", action="store_true",
                    help="do everything except launch children and write files")
    args = ap.parse_args(argv)
    args.device_fence = not args.no_device_fence
    # `R.run_family` reads args.disable_features; the real value is written onto the
    # per-arm namespace copy in launch(). --disable-features is deliberately NOT exposed:
    # an arm is a named, validated row of the ARMS table, never a free-text token that
    # run_h200 would merely print a warning about.
    args.disable_features = ""
    return args


def resolve_selection(args: argparse.Namespace
                      ) -> tuple[list[Arm], list[str], list[str]]:
    """Validate --arms/--families/--regimes BEFORE anything launches. Exit 2 on any typo.

    run_h200 only `print()`s an unrecognised `--disable-features` token (run_h200.py:
    1199-1201, a bare print that never reaches driver.log). On a machine nobody can log into,
    a typo'd arm name would otherwise produce a full-length run of the wrong arm.
    """
    names = [s.strip() for s in str(args.arms).split(",") if s.strip()]
    bad = [n for n in names if n not in ARM_BY_NAME]
    if bad:
        print(f"!! unknown arm(s) {bad}. Known arms:", flush=True)
        for a in ARMS:
            print(f"     {a.name:<12} --disable-features {a.disable_features!r:<45} "
                  f"{a.note}", flush=True)
        raise SystemExit(2)
    if not names:
        names = [DEFAULT_ARMS]
    # canonical ARMS order, with `hopper` always last
    chosen = [a for a in ARMS if a.name in names and a.name != "hopper"]
    if "hopper" in names:
        chosen.append(ARM_BY_NAME["hopper"])

    fam_in = [s.strip() for s in str(args.families).split(",") if s.strip()]
    for tok in fam_in:
        if tok in EXCLUDED_FAMILIES or tok == "all":
            print(f"!! --families {tok!r} is refused. bench_layer.py carries a 16 h timeout "
                  f"(run_h200.py:172)\n"
                  f"!! and the per-fusion question this control arm answers does not need "
                  f"it. Enabling a\n"
                  f"!! 16-hour arm by typing one word on a machine nobody can log into is "
                  f"not a thing this\n"
                  f"!! driver will do silently. In-scope families: "
                  f"{', '.join(FAMILY_KEYS)}", flush=True)
            raise SystemExit(2)
    bad = [t for t in fam_in if t not in FAMILY_KEYS]
    if bad:
        print(f"!! unknown famil(ies) {bad}; known: {', '.join(FAMILY_KEYS)}", flush=True)
        raise SystemExit(2)
    fams = [k for k in FAMILY_KEYS if k in fam_in] if fam_in else list(FAMILY_KEYS)

    reg_in = [s.strip() for s in str(args.regimes).split(",") if s.strip()]
    abbr_to_full = {v: k for k, v in R.REGIME_ABBR.items()}
    regs: list[str] = []
    for tok in reg_in:
        full = tok if tok in R.KNOWN_REGIMES else abbr_to_full.get(tok)
        if full is None:
            print(f"!! unknown regime {tok!r}; known: {', '.join(R.KNOWN_REGIMES)}\n"
                  f"!! abbreviations also accepted: "
                  f"{', '.join(sorted(abbr_to_full))}", flush=True)
            raise SystemExit(2)
        regs.append(full)
    scope = [r for r in R.KNOWN_REGIMES if r in regs] if regs else list(R.KNOWN_REGIMES)
    return chosen, fams, scope


# ======================================================================================
# main
# ======================================================================================
class _CanaryTripped(Exception):
    """The campaign fingerprint changed -- exit 6, immediately, even under
    --continue-on-verify-fail."""


class _RedirectIgnored(Exception):
    """A bench wrote outside the staging tree it was pointed at -- exit 6, immediately."""


class _TenantAppeared(Exception):
    """A neighbour moved onto the measurement card mid-arm. Fatal for a DIFF (see the
    check_tenants call site); recoverable by waiting and re-running."""


class _VerifyFailed(Exception):
    """Engagement verification failed and --continue-on-verify-fail was not given."""


WHAT_IS_MEASURED = (
    "This run measured the following arms and nothing else: {arms}. "
    "{hopper}, so the comparison against results/h200/*.json is CROSS-SESSION: thermals, "
    "co-tenancy, clock state and driver state all differ between the two readings and are "
    "confounded with the effect being measured. That confound is conceded by the design "
    "(the operator chose classic-only vs the existing campaign over a same-session paired "
    "A/B) and it has exactly one defence: f03 (ResAdd+RMSNorm) and f10 (ExpertMerge+ResAdd) "
    "advertise no Hopper cfg key at all -- kernel_cfg_keys is literally 'module advertises "
    "none' -- so their classic arm is byte-identically configured to their Hopper arm and "
    "their classic-vs-campaign delta IS the run-to-run/cross-session noise floor. Every "
    "other family's delta must be judged against that band, and a family whose delta sits "
    "inside it has shown NOTHING -- which is not the same claim as 'the Hopper features did "
    "nothing'. Note also that GLM52_H200_CLASSIC forces four capabilities off, including "
    "wgmma, one more than the three levers the question names, so no GEMM-family delta from "
    "this arm alone can be attributed to TMA, warp specialization or clusters "
    "specifically. Nothing in this run has been merged into results/h200/ and nothing ever "
    "will be: the control arm is a diff, not an append, and this repo contains no merge "
    "step for it."
)


def main(argv: list[str] | None = None) -> int:
    global WRITE_ROOT
    # BEFORE argument parsing does anything expensive, and before --list, because the whole
    # value of this fence is that the operator meets it during the rehearsal.
    contaminated = refuse_inherited_env()
    if contaminated:
        return contaminated
    args = parse_args(sys.argv[1:] if argv is None else argv)
    arms, fams, scope = resolve_selection(args)

    args.results_dir = Path(args.results_dir).expanduser().resolve()
    logdir = Path(args.log_dir).expanduser().resolve()
    logdir.mkdir(parents=True, exist_ok=True)
    WRITE_ROOT = (args.results_dir / STAGING_ROOT).resolve()

    log = R.Log(logdir / "driver.log")
    warnings: list[str] = []
    tenant_contaminated: list[str] = []
    t_start = time.time()
    exit_code = 0

    R.rule(log, "GLM-5.2 fusion study -- H200 HOPPER CONTROL ARM")
    log(f"  repo             {REPO}")
    log(f"  campaign (READ)  {args.results_dir}")
    log(f"  staging (WRITE)  {WRITE_ROOT}")
    log(f"  logs             {logdir}")
    log(f"  started          {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")
    log("  NOTHING under the staging root is ever merged into the campaign files. The")
    log("  control arm is a DIFF, not an append: results/h200/*.json is its BASELINE, and")
    log("  a write into them would destroy the comparison rather than extend it. There is")
    log("  no merge function in this file and no script in this repo may add one.")

    # --- the plan ----------------------------------------------------------------------
    # Read the campaign's own summary BEFORE the plan is printed, not after: the plan is the
    # only place the operator is told how long this costs, and the honest number for that is
    # the campaign's MEASURED wall time, not a timeout. `summary.json` is 380 KB; the 25 MB
    # of family files (campaign_floors) are still read once, later.
    camp_summary, camp_summary_err = load_json(args.results_dir / "summary.json")
    camp_recorded_at = _jget(camp_summary, "_meta", "recorded_at")
    camp_wall = {k: campaign_wall_s_from(camp_summary, k) for k in FAMILY_KEYS}

    budget = sum((args.timeout or R.FAMILY_BY_KEY[k].timeout_s) for k in fams)
    R.rule(log, "PLAN")
    log(f"  arms             {', '.join(a.name for a in arms)}")
    for a in arms:
        log(f"    {a.name:<12} --disable-features {a.disable_features!r}")
        log(f"    {'':<12} {a.note}")
        log(f"    {'':<12} staging {staging_for(args, a.name)}")
        log(f"    {'':<12} log keys "
            f"{', '.join(f'{k}.{a.name}.a<n>.log' for k in fams[:3])}"
            + (", ..." if len(fams) > 3 else ""))
    log(f"  families         {', '.join(fams)}   (excluded by design: "
        f"{', '.join(EXCLUDED_FAMILIES)})")
    log(f"  regimes          {len(scope)}: {', '.join(scope)}")
    log("")
    # HOW LONG THIS ACTUALLY TAKES. The per-family numbers in run_h200.FAMILIES (3/4/8/8/
    # 10/10/14 h = 57 h) are TIMEOUT CEILINGS -- the point at which a child is killed -- and
    # printing them as though they were expectations overstates reality by about 100x: the
    # campaign measured all seven families in 1.3 h. An operator who blocks out 2.5 days of a
    # confidential machine for a ~2 h job, and who believes a non-engaging arm costs 3 h to
    # detect when it costs about a minute, was misled by this driver's own plan. So both
    # numbers are printed, each labelled as what it is, and a family with no recorded
    # campaign wall time prints nothing rather than a guess.
    known = [k for k in fams if camp_wall.get(k)]
    log("  per-family cost   campaign wall time (MEASURED) vs timeout ceiling (a kill "
        "switch, not an estimate)")
    for k in fams:
        cw = camp_wall.get(k)
        ceil_s = args.timeout or R.FAMILY_BY_KEY[k].timeout_s
        log(f"    {k:<10} "
            + (f"{R.fmt_dur(cw):>10} measured" if cw else f"{'--':>10} not recorded")
            + f"   ceiling {R.fmt_dur(ceil_s)}")
    if known:
        total_w = sum(camp_wall[k] for k in known)
        log(f"    {'TOTAL':<10} {R.fmt_dur(total_w):>10} measured over "
            f"{len(known)}/{len(fams)} famil(ies)"
            + f"   ceiling {R.fmt_dur(budget)}")
        log(f"  expected          about {R.fmt_dur(total_w)} per arm, "
            f"{R.fmt_dur(total_w * len(arms))} for {len(arms)} arm(s), if this arm behaves "
            f"like the campaign.")
        log("                    The classic arm's tuner grid COLLAPSES (h200_cfg_overlays() "
            "returns [] once")
        log("                    every axis is unavailable), so it should be no slower. "
            "Source: the campaign's")
        log(f"                    own results/h200/summary.json families[*].wall_s, "
            f"recorded {camp_recorded_at or 'at an unrecorded time'}.")
    else:
        log(f"  expected          unknown: no per-family wall_s could be read from "
            f"{args.results_dir / 'summary.json'}"
            + (f" ({camp_summary_err})" if camp_summary_err else "")
            + ". No estimate is printed rather than a guessed one.")
    log(f"  timeout budget    {R.fmt_dur(budget)} per arm, "
        f"{R.fmt_dur(budget * len(arms))} for {len(arms)} arm(s) -- the CEILING, i.e. when a "
        f"child is killed.")
    log("")
    if "f03" in fams:
        f03_w = camp_wall.get("f03")
        f10_w = camp_wall.get("f10")
        log("  f03 runs first and is the earliest engagement check. "
            + (f"The campaign measured it in {R.fmt_dur(f03_w)}, so a non-engaging arm is "
               f"caught" if f03_w else "A non-engaging arm is caught"))
        log("  within minutes rather than at the end of the run; its 3 h budget is a timeout "
            "ceiling,")
        log("  not an estimate."
            + (f" f10 follows ({R.fmt_dur(f10_w)} in the campaign)." if f10_w
               else " f10 follows."))
        log("  Those two families advertise no Hopper cfg key, so their classic-vs-campaign")
        log("  delta IS the cross-session noise floor every other family's delta is judged "
            "against.")
    else:
        msg = ("f03 is NOT in the family list. It is the cheapest family (the campaign "
               "measured it in 57 s) AND half of the noise floor: without it a non-engaging "
               "arm is not caught until the first expensive family finishes, and the report "
               "has no drift band to judge anything against.")
        warnings.append(msg)
        log(f"  !! {msg}")
    if "f10" not in fams:
        msg = ("f10 is NOT in the family list; the noise floor loses half its 22 samples "
               "and make_control_report_h200.py will refuse to publish.")
        warnings.append(msg)
        log(f"  !! {msg}")
    if args.quick:
        log("  !! --quick: every arm will be sentinelled ARM_NOT_VERIFIED. Smoke test only.")

    if args.list:
        log.close()
        return 0

    # --- GPU selection, preflight, device gate ------------------------------------------
    anchor: dict = {}
    hw_start = R.hwinfo()
    dev: dict = {}
    device_name = ""
    pf: dict | None = None
    tick = {"tick_us": R.DEFAULT_TICK_US, "source": "default (no preflight read)",
            "trusted": True, "distrust_reasons": [], "match_frac": None}
    want_uuid, uuid_src = campaign_gpu_uuid(args.results_dir)

    if args.verify_only:
        log("")
        log("  --verify-only: no GPU work beyond one nvidia-smi snapshot, no child is "
            "launched.")
        gpu = {"index": None, "uuid": want_uuid, "name": None, "refuse": False,
               "reason": "--verify-only: the card is not selected because nothing is "
                         "measured"}
        anchor = {"matched": None, "want_uuid": want_uuid, "want_uuid_source": uuid_src,
                  "got_uuid": None, "note": "--verify-only; the device fence checks the "
                                            "staged files against the campaign's UUID"}
        pf = R.read_preflight()
        device_name = R._norm_dev(_jget(camp_summary, "_meta", "device"))
        tick = build_tick(pf, warnings, "default (no preflight on this host)")
    else:
        gpu = pin_campaign_card(log, args, hw_start, warnings, anchor)
        if gpu.get("refuse"):
            log.close()
            return 4
        hw_row = R.pick_hw_row(hw_start, gpu.get("index"))

        pf = R.read_preflight()
        need_probe = False
        if pf is None:
            need_probe = not args.skip_preflight
            if need_probe:
                log(f"  no {R.PREFLIGHT_JSON.name}; running the preflight probe first.")
        elif hw_row:
            # Do NOT re-probe when the cached preflight already matches the pinned card:
            # the two arms sharing ONE probe is part of what makes them comparable.
            pf_name = R._norm_dev((pf.get("device") or {}).get("name"))
            pf_uuid = R._norm_uuid((pf.get("gpu_selection") or {}).get("uuid")
                                   or (pf.get("device") or {}).get("uuid"))
            row_uuid = R._norm_uuid(hw_row.get("uuid"))
            why = ""
            if pf_name != R._norm_dev(hw_row.get("name")):
                why = f"model differs ({pf_name!r} vs {R._norm_dev(hw_row.get('name'))!r})"
            elif gpu.get("index") is not None and pf_uuid and row_uuid \
                    and pf_uuid != row_uuid:
                why = f"same model but a different card (probe {pf_uuid}, pinned {row_uuid})"
            if why:
                if args.skip_preflight:
                    log(f"!! cached preflight does not describe the pinned GPU: {why} -- "
                        f"--skip-preflight, so the CACHED probe is used as-is.")
                elif args.dry_run:
                    # Say what would happen, not what did. This message used to be printed
                    # unconditionally and the re-probe was then correctly suppressed by
                    # `if need_probe and not args.dry_run`, so a --dry-run rehearsal on a
                    # laptop announced a re-probe, did not perform one, and went on to print
                    # `[gate] sm_90 confirmed (NVIDIA H200)` and a full H200 calibration
                    # banner while pinned to whatever card is actually present.
                    log(f"!! cached preflight does not describe the pinned GPU: {why} -- "
                        f"would re-probe (SKIPPED: --dry-run).")
                    log("!! The banner, the calibration and the device gate below therefore "
                        "describe the CACHED")
                    log("!! probe, not this host. On a rehearsal box that is the difference "
                        "between reading")
                    log("!! 'sm_90 confirmed' and being on an sm_90 card.")
                else:
                    log(f"!! cached preflight does not describe the pinned GPU: {why} -- "
                        f"re-probing.")
                need_probe = not args.skip_preflight
        if need_probe and not args.dry_run:
            # `glm52_h200/preflight.py:1056` refuses to overwrite OUT_JSON only when the
            # device NAME differs, and 'NVIDIA H200' == 'NVIDIA H200' for two different H200s
            # in the same node -- so a re-probe here REPLACES glm52_h200/preflight_h200.json
            # in place and the campaign's own probe file is gone, leaving only its digest
            # inside results/h200/summary.json and each family's fairness.preflight. That is
            # a write to campaign provenance by a run whose whole premise is that it touches
            # nothing of the campaign's. Copy it aside first (into the LOG dir, which is
            # outside the results tree and outside the write gate by the same rule R.Log is)
            # and say so as a first-class warning.
            if R.PREFLIGHT_JSON.exists():
                keep = logdir / "preflight_h200.campaign.json"
                try:
                    # read_bytes/write_bytes, not shutil.copy: this file promises in its
                    # docstring that it contains no shutil.copy and no shutil.move, and that
                    # promise is worth more than the two characters it saves.
                    keep.write_bytes(R.PREFLIGHT_JSON.read_bytes())
                    # The preserved file is the DISPLACED CACHE, not necessarily the
                    # campaign's own probe: we are here precisely because the cache did not
                    # describe the pinned card, so it may describe some other GPU entirely.
                    # On 2026-08-11 it described GPU 7 (uuid 3aa19cef) while the campaign ran
                    # on b2318e71. The filename says "campaign" for backward compatibility
                    # with the runs that already wrote it; say what it actually holds.
                    kept_uuid = "unknown"
                    try:
                        _k = json.loads(keep.read_text())
                        kept_uuid = str(_jget(_k, "gpu_selection", "uuid")
                                        or _jget(_k, "gpu", "uuid") or "unknown")
                    except (OSError, ValueError, TypeError):
                        pass
                    msg = (f"the cached preflight did not describe the pinned card, so it is "
                           f"being re-probed and {R.PREFLIGHT_JSON} will be REPLACED "
                           f"in place. The DISPLACED CACHE was preserved at {keep} before "
                           f"the probe ran -- it describes gpu uuid {kept_uuid}, which is "
                           f"NOT necessarily the campaign's card, so do not read it as the "
                           f"campaign's probe despite the filename; the campaign's own probe "
                           f"survives in results/h200/summary.json.preflight and in each "
                           f"family's fairness.preflight. Pass --skip-preflight to reuse the "
                           f"cached probe instead.")
                except OSError as exc:
                    msg = (f"the cached preflight is about to be REPLACED in place by the "
                           f"re-probe and it could NOT be preserved first "
                           f"({type(exc).__name__}: {exc}). The campaign's probe file will "
                           f"be lost; only its digest survives, in summary.json.preflight "
                           f"and each family's fairness.preflight.")
                warnings.append(msg)
                log(f"  !! {msg}")
            pf = R.run_preflight(log, args.python, logdir, quick=args.quick,
                                 gpu=gpu.get("index")) or pf

        tick = build_tick(pf, warnings)
        R.banner(log, pf, hw_start, args.results_dir, tick, gpu)
        try:
            dev = R.device_gate(log, pf, hw_row, args.force, warnings)
        except SystemExit:
            log("!! (escape hatch: python3 run_control_h200.py --force --results-dir "
                "results/<thisdevice>)")
            log.close()
            raise
        device_name = dev.get("name") or ""

        if not args.dry_run:
            busy = R.another_bench_running()
            if busy and not args.allow_concurrent:
                log("!! another glm52_h200 bench appears to be running; refusing (two on "
                    "one GPU corrupt every timing).")
                log.close()
                return 3
            if busy:
                warnings.append("--allow-concurrent: another bench process was detected")

    # --- the staging tree and the campaign canary ---------------------------------------
    if not args.dry_run:
        ensure_dir(WRITE_ROOT)
    fp = fingerprint_campaign(args.results_dir)
    log("")
    log(f"  campaign canary: fingerprinted {len(fp)} file(s) under {args.results_dir}; "
        f"re-checked after every family.")

    summary_path = args.results_dir / STAGING_ROOT / SUMMARY_NAME
    arm_recs: dict[str, dict] = {}
    all_cells: list[dict] = []

    # Read once, not once per save(): `campaign_floors` opens all seven family files (~25 MB
    # of JSON). `save()` runs after every family of every arm, and re-parsing 25 MB 35 times
    # is minutes of wall time spent on nothing. summary.json (380 KB) was already read once,
    # above the PLAN block, because the plan needs its per-family wall times.
    camp_floors = campaign_floors(args.results_dir)

    def build_summary(live: bool) -> dict:
        return {
            "schema": 1,
            "id": "glm52_h200_control_arm_summary",
            "_meta": {
                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": device_name or "",
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
                "gpu_index": gpu.get("index"), "gpu_uuid": gpu.get("uuid"),
                "gpu_pinned": bool(gpu.get("pinned")),
                "gpu_was_idle": (not gpu.get("busy")) if gpu.get("busy") is not None
                                else None,
            },
            "driver": {
                "file": "run_control_h200.py", "argv": list(sys.argv),
                "cwd": str(Path.cwd()), "repo": str(REPO),
                "results_dir": str(args.results_dir),
                "staging_root": str(WRITE_ROOT), "log_dir": str(logdir),
                "started": time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(t_start)),
                "finished": None if live else time.strftime("%Y-%m-%d %H:%M:%S"),
                "wall_s": time.time() - t_start,
                "arms_requested": [a.name for a in arms],
                "families_planned": list(fams), "regimes": list(scope),
                "quick": bool(args.quick), "device_fence": bool(args.device_fence),
                "any_gpu": bool(args.any_gpu), "force_rerun": bool(args.force_rerun),
                "excluded_families": list(EXCLUDED_FAMILIES),
                "verify_only": bool(args.verify_only), "dry_run": bool(args.dry_run),
            },
            "campaign": {
                "results_dir": str(args.results_dir),
                "gpu_uuid": want_uuid, "gpu_uuid_source": uuid_src,
                "harness_floor_us_by_family": camp_floors,
                "summary_recorded_at": camp_recorded_at,
                "fingerprint": fp, "fingerprint_ok": True,
                "fingerprint_digest": digest(fp),
            },
            "device_anchor": anchor,
            "gpu": gpu,
            "device": dev,
            "preflight": (R.preflight_digest(pf) if pf else
                          {"path": str(R.PREFLIGHT_JSON), "present": False}),
            # The SAME shape run_h200 writes into results/h200/summary.json --
            # {'tick_us', 'source', 'unresolved_ticks', 'trusted', 'distrust_reasons',
            # 'match_frac'}. `match_frac` in particular is not optional: it is read by
            # glm52/make_control_report_h200.py:803 as a provenance row and a MISSING value
            # is REFUSE-severity there, so dropping it made every honest control arm
            # unpublishable except via --force-floor -- which also switches off the genuine
            # contended-card refusal.
            "timer": {"tick_us": tick.get("tick_us"), "source": tick.get("source"),
                      "unresolved_ticks": args.unresolved_ticks,
                      "trusted": tick.get("trusted"),
                      "distrust_reasons": tick.get("distrust_reasons") or [],
                      "match_frac": tick.get("match_frac")},
            "arms": arm_recs,
            "cells": all_cells,
            "hwinfo_start": hw_start, "hwinfo_end": [], "hwinfo_drift": [],
            "warnings": warnings,
            # "live" is rewritten False by the final save; a summary still saying True is a
            # run that never reached its own end -- killed, crashed or still going. Read it
            # with `terminated_by` and each arm's `verified_reason`.
            "live": live,
            "terminated_by": TERMINATED_BY_SIGNAL,
            "tenant_contaminated": list(tenant_contaminated),
            "what_is_measured": WHAT_IS_MEASURED.format(
                arms=", ".join(a.name for a in arms),
                hopper=("the hopper arm WAS re-measured in this session"
                        if any(a.name == "hopper" for a in arms)
                        else "the Hopper arm was NOT re-measured in this session"),
            ),
        }

    def save(live: bool = True) -> None:
        if args.dry_run:
            return
        try:
            atomic_write_json(summary_path, build_summary(live))
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            log(f"!! could not write {summary_path.name}: {type(exc).__name__}: {exc}")

    # --- the arms ------------------------------------------------------------------------
    canary_tripped = False
    launched_any = False        # has any child been launched yet, across all arms?
    try:
        for arm in arms:
            staging = staging_for(args, arm.name)
            engdir = engagement_dir(args, arm.name)
            sentinel = staging / SENTINEL_NAME
            if not args.dry_run:
                ensure_dir(staging)
                ensure_dir(engdir)
            rec = arm_recs.setdefault(arm.name, {
                "arm": arm.name, "disable_features": arm.disable_features,
                "expected_env": arm_expected_env(arm),
                # PESSIMISTIC BY DEFAULT. `verified` used to start True and be flipped to
                # False on failure, so a run KILLED mid-arm left the last incremental save
                # claiming a verified arm: the 2026-08-11 run died during f04f05 with 3 of 7
                # families measured and recorded `verified: true, sentinel: null`. Only the
                # ARM_NOT_VERIFIED file left over from an EARLIER abort kept that from
                # reading as publishable, which is luck, not a safety net. An arm is verified
                # only where something has affirmatively verified it, at the end of a
                # COMPLETE arm; every other state -- killed, crashed, partial -- must inherit
                # the unverified default.
                "staging": str(staging), "verified": False,
                "sentinel": SENTINEL_NAME, "verified_reason": "arm has not completed",
                "wall_s": 0.0, "fam_stages": {}, "engagement_summary": {},
                "cells": [], "table": [], "note": arm.note,
            })
            t_arm = time.time()
            R.rule(log, f"ARM {arm.name} -- {arm.note}")
            log(f"  --disable-features {arm.disable_features!r}")
            log(f"  expected child env  {arm_expected_env(arm)}")
            log(f"  staging             {staging}")
            # Read the sentinel state BEFORE --quick writes its own, or --quick would look
            # like a resumed unverified arm and measure nothing at all.
            resume_verify_only = sentinel.exists() and not args.force_rerun
            if resume_verify_only:
                log(f"  !! {SENTINEL_NAME} is present from an earlier run: this arm is "
                    f"RE-VERIFIED but NOT re-measured.")
                log("     (that is how an operator recovers from a verifier bug without "
                    "re-measuring the whole arm;")
                log("      --force-rerun would re-measure it from zero.)")
            if args.force_rerun:
                log("  !! --force-rerun: GLM52_H200_FORCE=1 reaches the STAGED checkpoints "
                    "under")
                log(f"     {staging / '_ckpt'} and restarts this arm from zero. It is never "
                    "needed --")
                log("     the staging tree already isolates _ckpt from the campaign's.")
            if args.quick and not args.dry_run:
                atomic_write_text(sentinel,
                                  "ARM NOT VERIFIED -- --quick\n\n"
                                  "This arm was measured with --quick, which shortens every "
                                  "sweep. Its numbers are a\nstack smoke test and must not "
                                  "be published as a control arm.\n")
                rec["verified"] = False
                rec["sentinel"] = "--quick"

            arm_failed = False
            for key in fams:
                fam = R.FAMILY_BY_KEY.get(key)
                if fam is None:
                    warnings.append(f"{key} is not a known run_h200 family; skipped")
                    log(f"!! {key} is not in run_h200.FAMILIES -- skipped.")
                    continue
                script = R.find_script(fam)
                if script is None:
                    warnings.append(f"no bench script found for {key}")
                    log(f"!! no bench found for {key} -- skipped.")
                    continue

                # The sentinel resume is PER FAMILY, not arm-wide. It used to be arm-wide,
                # and that turned the recovery path into a trap: after the V9 false abort
                # (LOG-18, 2026-08-11) f03 was staged and the other six had never launched,
                # so a plain relaunch skipped ALL SEVEN, every unlaunched family failed V0
                # "missing or unreadable", and the run aborted having measured nothing --
                # costing a third round trip on a machine nobody can reach. A family is
                # re-verified instead of re-measured only when its own staged payload
                # already covers the whole scope.
                staged_covers_scope = False
                if resume_verify_only:
                    _pay, _ = load_json(staged_family_path(fam, staging, args.results_dir))
                    _rows = regime_rows(_pay)
                    _mult = canonical_mult(fam, args.results_dir)
                    staged_covers_scope = bool(_rows) and all(
                        len(_rows.get(r, ())) >= _mult for r in scope)
                    log(f"  [{arm.name}/{key}] staged payload "
                        f"{'covers' if staged_covers_scope else 'does NOT cover'} the "
                        f"{len(scope)}-regime scope -> "
                        f"{'re-verify only' if staged_covers_scope else 'MEASURE'}")

                if args.verify_only or args.dry_run or staged_covers_scope:
                    why = ("--verify-only" if args.verify_only else
                           "--dry-run" if args.dry_run else
                           f"{SENTINEL_NAME} present and this family is already staged; "
                           f"re-verifying only")
                    log("")
                    log(f"  [{arm.name}/{key}] not launching ({why}).")
                    stage = {"family": key, "arm": arm.name,
                             "result": str(staged_family_path(fam, staging,
                                                              args.results_dir)),
                             "attempts": [], "poisoned": {}, "regimes_done": [],
                             "regimes_missing": [], "rows_mult":
                                 canonical_mult(fam, args.results_dir),
                             "not_launched": why, "wall_s": 0.0}
                    payload_path = staged_family_path(fam, staging, args.results_dir)
                    payload, err = load_json(payload_path)
                    rows = regime_rows(payload)
                    mult = stage["rows_mult"]
                    stage["regimes_done"] = [r for r in scope
                                             if len(rows.get(r, ())) >= mult]
                    stage["regimes_missing"] = [r for r in scope
                                                if len(rows.get(r, ())) < mult]
                    stage["result_error"] = err or None
                else:
                    stage = run_family_stage(log, args, arm, fam, script,
                                             args.results_dir, staging, logdir, gpu,
                                             scope, [], warnings, fp=fp,
                                             first_stage=not launched_any)
                    # Only a stage that actually launched a child counts: otherwise a
                    # resumed run marks the redirect PRE-COMMITMENT check as already spent
                    # and never performs it on the first family that really does launch.
                    launched_any = launched_any or bool(stage.get("attempts"))
                    # CO-TENANCY IS FATAL HERE, unlike in run_h200.py. That driver's policy
                    # is "flag and keep going -- an aborted campaign loses more than a
                    # flagged one", which is right for a campaign that IS the baseline. This
                    # arm is a DIFF against a baseline measured on an idle card, so a
                    # neighbour does not add noise, it removes the comparison: on
                    # 2026-08-11 a VLLM worker took 121.6 GB of the 143.8 GB card during
                    # f01, f01 came out at 1.93x the campaign's tune time, and every family
                    # after it would have been diffed against an idle-card baseline while
                    # sharing the card. Continuing spends GPU hours producing numbers that
                    # must then be thrown away.
                    n_warn = len(warnings)
                    R.check_tenants(log, gpu, f"after {arm.name}/{key}", warnings)
                    if any("co-tenant" in w for w in warnings[n_warn:]):
                        stage["tenant_contaminated"] = True
                        tenant_contaminated.append(f"{arm.name}/{key}")
                        log("!" * 92)
                        log(f"!! CO-TENANT on the measurement card during {arm.name}/{key}.")
                        log(f"!! {key} was measured while sharing the card and is NOT "
                            f"comparable with the campaign baseline, which was measured on "
                            f"an idle card. Its staged JSON is kept -- it is evidence -- but "
                            f"it is marked contaminated and must not be published.")
                        if args.on_tenant == "stop":
                            log("!! STOPPING before the remaining families: continuing would "
                                "spend GPU hours on numbers that must then be discarded.")
                            log("!! Wait for the card to clear, then re-run; --force-rerun "
                                f"re-measures {key}. --on-tenant flag continues anyway.")
                            log("!" * 92)
                            rec["fam_stages"][key] = stage
                            arm_failed = True
                            raise _TenantAppeared(f"co-tenant during {arm.name}/{key}")
                        log("!! --on-tenant flag: continuing, every later family is tainted.")
                        log("!" * 92)

                rec["fam_stages"][key] = stage
                rec["wall_s"] = time.time() - t_arm

                # -- the ENV RESIDUAL check: did the child see only what this arm sets? --
                # `arm_expected_env(arm)` has always been computed and written into the
                # summary as a PREDICTION, and nothing ever diffed it against the
                # `env_overrides` dict `R.run_family` captures from the real child
                # environment. `refuse_inherited_env()` at startup is the primary fence;
                # this is the after-the-fact confirmation that it held, and it is the only
                # thing that would notice a GLM52_H200_* variable arriving by some other
                # route (a wrapper script, a systemd unit, a mutated os.environ).
                expected_env = arm_expected_env(arm)
                surprises: dict[str, str] = {}
                for att in (stage.get("attempts") or []):
                    for k_env, v_env in (att.get("env_overrides") or {}).items():
                        if k_env in OWNED_ENV_VARS and expected_env.get(k_env) != v_env:
                            surprises[k_env] = v_env
                if surprises:
                    msg = (f"{arm.name}/{key}: the child saw Hopper switch variable(s) this "
                           f"arm did not set: {surprises}. Expected exactly {expected_env}. "
                           f"This arm disabled MORE than it claims, so its per-feature "
                           f"attribution is not what the file says it is.")
                    warnings.append(msg)
                    stage["env_residual"] = surprises
                    log(f"  !! {msg}")

                # -- the campaign canary, after EVERY family ---------------------------
                # (also after every ATTEMPT, inside run_family_stage -- see canary_or_raise)
                canary_or_raise(log, fp, args.results_dir, warnings,
                                f"after {arm.name}/{key}")

                # -- wall-time sanity (report, do not stop) -----------------------------
                camp_w = camp_wall.get(key)
                if camp_w and stage.get("wall_s") and stage["wall_s"] > camp_w * 1.5 \
                        and arm.name != "hopper":
                    msg = (f"{arm.name}/{key} took {R.fmt_dur(stage['wall_s'])} vs the "
                           f"campaign's {R.fmt_dur(camp_w)} (>50 % longer). The classic "
                           f"arm's widened grid collapses (h200_cfg_overlays() returns []), "
                           f"so it should be FASTER to tune; a slower classic arm is a "
                           f"signal something is wrong.")
                    warnings.append(msg)
                    log(f"  !! {msg}")

                # -- engagement verification -------------------------------------------
                if args.dry_run:
                    # A dry run has nothing staged to verify, and a V0 "no result file"
                    # FAIL would abort the rehearsal it exists to provide. Say what WOULD
                    # be checked instead of inventing a verdict.
                    log("")
                    log(f"  [{arm.name}/{key}] --dry-run: engagement verification would run "
                        f"V1..V11 here against")
                    log(f"      {staged_family_path(fam, staging, args.results_dir)}")
                    surface, ssrc = offered_axes_from_campaign(
                        load_json(family_canonical(fam, args.results_dir))[0], key)
                    tgt = sorted(frozenset(arm.axes_off) & surface)
                    vac = sorted(frozenset(arm.axes_off) - surface)
                    log(f"      campaign offered: {', '.join(sorted(surface)) or 'nothing'} "
                        f"({ssrc})")
                    log(f"      axes this arm must collapse: {', '.join(tgt) or 'none'}"
                        + (f"; vacuous (nothing to disable): {', '.join(vac)}" if vac
                           else ""))
                    rec["engagement_summary"][key] = "DRY-RUN"
                    continue
                payload, perr = load_json(
                    staged_family_path(fam, staging, args.results_dir))
                # The campaign load ERROR is kept, not discarded: verify_family needs it to
                # tell "this family offers no axes" apart from "the baseline could not be
                # read, so nothing here was actually compared against anything".
                campaign_payload, cerr = load_json(
                    family_canonical(fam, args.results_dir))
                child_logs = sorted(logdir.glob(f"{key}.{arm.name}.a*.log"))
                vrec = verify_family(arm, key, payload, child_logs, campaign_payload,
                                     gpu, device_name, device_fence=args.device_fence,
                                     campaign_error=cerr, staging=staging)
                stage["engagement"] = vrec
                rec["engagement_summary"][key] = vrec["verdict"]
                if not args.dry_run:
                    # written FIRST, before the sentinel and before anything else can die.
                    # The `_meta.device` stamp is not decoration: run_h200's
                    # quarantine_foreign_results (run_h200.py:994-1016) rglobs INTO this
                    # tree, reads payload['_meta']['device'] (:1004-1005), resolves a missing
                    # one to '' and MOVES the file. An unstamped .verify.json is the evidence
                    # for whether the arm engaged, so losing it to a later run_h200.py
                    # invocation is the worst possible thing to lose quietly.
                    #
                    # PRESERVE A DISSENTING PRIOR VERDICT BEFORE OVERWRITING IT. --verify-only
                    # is the documented recovery path after a verifier bug: re-judge data
                    # already on disk, no GPU. But it rewrites this very file, so the FAILED
                    # record that motivated the repair is destroyed by the act of repairing
                    # it -- and that record is evidence, the same argument the ARM_NOT_VERIFIED
                    # sentinel is built on. On 2026-08-11 a V9 defect aborted the run and the
                    # first --verify-only pass silently replaced the failing f03.verify.json
                    # with a passing one. Only ever copies a record whose verdict DISAGREES
                    # with the new one, so a re-run that changes nothing leaves no litter.
                    engpath = engdir / f"{key}.verify.json"
                    prior, _perr = load_json(engpath)
                    if isinstance(prior, dict) and prior.get("verdict") != vrec["verdict"]:
                        stamp = time.strftime("%Y%m%d_%H%M%S")
                        kept = engdir / f"{key}.verify.superseded_{stamp}.json"
                        try:
                            atomic_write_json(kept, prior)
                            log(f"    the previous verdict for {key} was "
                                f"{prior.get('verdict')}, now {vrec['verdict']}; the "
                                f"superseded record is preserved at {kept.name}")
                        except OSError as exc:
                            log(f"  !! could NOT preserve the superseded {key} verdict "
                                f"({prior.get('verdict')}) before overwriting it: "
                                f"{type(exc).__name__}: {exc}")
                    atomic_write_json(engpath, {
                        "_meta": {
                            "device": device_name or "",
                            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "writer": "run_control_h200.py",
                            "why_device": "run_h200.quarantine_foreign_results reads "
                                          "_meta.device and MOVES any file under the results "
                                          "tree that lacks it",
                        },
                        **vrec,
                    })
                log("")
                log("  " + verdict_line(vrec))
                # Logged UNCONDITIONALLY, not only when vacuous_axes is non-empty. For
                # f01/f06/f08f09 the campaign offers all three axes, so vacuous_axes is empty
                # and the old branch never fired -- exactly the families where the axis
                # surface is most load-bearing, and exactly where a silent fall back to the
                # hardcoded CAMPAIGN_OFFERED table needs to be visible in the transcript.
                marker = "!! " if str(vrec["surface_source"]).startswith("CAMPAIGN_OFFERED") \
                    else ""
                log(f"    {marker}axis surface for {key}: "
                    f"{', '.join(vrec['surface']) or 'nothing'} "
                    f"(source: {vrec['surface_source']})")
                if str(vrec["surface_source"]).startswith("CAMPAIGN_OFFERED"):
                    warnings.append(
                        f"{arm.name}/{key}: the axis surface came from the hardcoded "
                        f"CAMPAIGN_OFFERED table, not from the baseline being diffed "
                        f"against -- the campaign file was unreadable")
                if vrec["vacuous_axes"]:
                    log(f"    vacuity guard: {', '.join(vrec['vacuous_axes'])} were never "
                        f"offered to {key} by the campaign, so a clean scan for them is not "
                        f"evidence (source: {vrec['surface_source']}).")

                # -- cells for the closing table ----------------------------------------
                try:
                    cells = R.collect_cells(
                        log, key, staged_family_path(fam, staging, args.results_dir),
                        float(tick.get("tick_us") or R.DEFAULT_TICK_US),
                        args.unresolved_ticks)
                except Exception as exc:  # noqa: BLE001
                    log(f"  !! could not read cells for {key}: {type(exc).__name__}: {exc}")
                    cells = []
                for c in cells:
                    c["arm"] = arm.name
                rec["cells"] = (rec.get("cells") or []) + cells
                all_cells.extend(cells)

                if vrec["verdict"] == "FAILED":
                    arm_failed = True
                    exit_code = 5
                    rec["verified"] = False
                    rec["sentinel"] = f"{key}: " + ", ".join(
                        sorted({c["id"] for c in vrec["checks"]
                                if c["status"] == "FAIL"}))
                    if not args.dry_run:
                        atomic_write_text(sentinel, sentinel_text(arm, key, vrec))
                    log("")
                    log("!" * 92)
                    log(f"!! ENGAGEMENT VERIFICATION FAILED for {arm.name}/{key}.")
                    log("!! A control arm that did not engage is worse than no control arm: "
                        "it is")
                    log("!! indistinguishable from a positive result -- which is this "
                        "study's hypothesis.")
                    for c in vrec["checks"]:
                        if c["status"] == "FAIL":
                            log(f"!!   {c['id']:4s} {c['path']}")
                            log(f"!!        want {c['want']}")
                            log(f"!!        got  {c['got']}")
                    log(f"!! sentinel written: {sentinel}")
                    log(f"!! engagement record: {engdir / f'{key}.verify.json'}")
                    log("!! The staged JSON has NOT been moved, renamed or deleted. A "
                        "control arm that did")
                    log("!! not engage is evidence; the sentinel is the fence that replaces "
                        "quarantining.")
                    log("!" * 92)
                    warnings.append(f"engagement verification FAILED for {arm.name}/{key}")
                    save()
                    # The hard abort exists to save GPU HOURS on a measuring run. Under
                    # --verify-only there are no GPU hours to save: aborting there meant one
                    # family's diagnosis per invocation, and the operator had to rediscover
                    # --continue-on-verify-fail to see the other six -- on a flag whose whole
                    # stated purpose is "verify whatever is already staged". So --verify-only
                    # always verifies everything and returns 5 at the end.
                    if args.verify_only:
                        log("!! --verify-only: continuing so that EVERY staged family is "
                            "diagnosed in this one")
                        log("!! pass -- no GPU time is at stake here. The exit code is still "
                            "5 and the sentinel")
                        log("!! still makes this arm unpublishable.")
                        continue
                    if not args.continue_on_verify_fail:
                        log("!! aborting the whole run (remaining families and remaining "
                            "arms). --continue-on-verify-fail")
                        log("!! keeps going if the remaining families' evidence is worth "
                            "the GPU time.")
                        raise _VerifyFailed(f"{arm.name}/{key}")
                    log("!! --continue-on-verify-fail: skipping the rest of THIS arm and "
                        "moving to the next.")
                    break

                save()

            # -- sentinel removal on a recovered arm ---------------------------------
            # ONLY on a COMPLETE arm. `arm_failed` says "nothing in scope failed", which is
            # not the same as "the arm is verified": with `--families f03` the other six
            # never enter the loop, so a 1-of-7 arm would clear the arm-wide sentinel, set
            # verified=True and sentinel=None, and read as fully publishable. The sentinel
            # is one of four independent "this arm is unverified" records and the only one
            # the report generator can see when a tarball loses the rest.
            full_scope = [f.key for f in R.FAMILIES if f.key not in EXCLUDED_FAMILIES]
            partial = sorted(k for k in full_scope
                             if k not in (rec.get("fam_stages") or {}))
            rec["partial_scope"] = partial or None
            if partial and sentinel.exists() and not args.dry_run:
                log(f"  {arm.name}: {SENTINEL_NAME} KEPT -- this run covered "
                    f"{len(full_scope) - len(partial)} of {len(full_scope)} families; "
                    f"{', '.join(partial)} were never adjudicated in this arm.")
            # The arm is verified iff it COMPLETED: every family in the full scope was
            # adjudicated and none failed. Decided here, not inside the sentinel branch
            # below -- a clean first run has no sentinel to remove and must still be able to
            # come out verified, and a killed run must never reach this line at all.
            arm_complete = (not arm_failed and not partial and not args.quick
                            and not args.dry_run)
            if arm_complete and not tenant_contaminated:
                rec["verified"] = True
                rec["verified_reason"] = (
                    f"all {len(full_scope)} families adjudicated, none failed")
            elif tenant_contaminated:
                rec["verified_reason"] = (
                    "a co-tenant appeared on the measurement card during this arm; "
                    f"contaminated families: {', '.join(tenant_contaminated)}")
            elif partial:
                rec["verified_reason"] = (
                    f"incomplete arm: {', '.join(partial)} never adjudicated")
            elif arm_failed:
                rec["verified_reason"] = "at least one family failed engagement verification"

            if arm_complete and not tenant_contaminated and sentinel.exists():
                try:
                    # Through the gate, like every other filesystem mutation in this file.
                    # The path is safe today because it is built from staging_for(), but the
                    # module docstring claims EVERY write is gated, and an ungated delete
                    # makes that claim false the moment anyone parameterises the sentinel
                    # name or derives it from a result file's parent.
                    guard_write(sentinel).unlink()
                    msg = (f"{arm.name}: {SENTINEL_NAME} removed -- re-verification now "
                           f"passes for every family in scope.")
                    log(f"  {msg}")
                    warnings.append(msg)
                    rec["sentinel"] = None
                    rec["sentinel_removed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                except OSError as exc:
                    log(f"  !! could not remove {sentinel}: {exc}")
            rec["wall_s"] = time.time() - t_arm
            try:
                rec["table"] = R.render_table([c for c in rec.get("cells") or []],
                                              tick, args.unresolved_ticks)
            except Exception as exc:  # noqa: BLE001
                rec["table"] = [f"(render_table failed: {type(exc).__name__}: {exc})"]
            save()

    except _CanaryTripped as exc:
        canary_tripped = True
        exit_code = 6
        for r in arm_recs.values():
            r.setdefault("campaign_canary", list(exc.args[0] if exc.args else []))
        save()
    except _RedirectIgnored as exc:
        # Exit 6, the same code as a tripped canary: both mean "a write went somewhere it
        # must not", and both mean the operator should ship the whole results dir back rather
        # than trust anything in it.
        exit_code = 6
        for r in arm_recs.values():
            r.setdefault("redirect_ignored", str(exc))
        save()
    except _VerifyFailed:
        exit_code = 5
    except _TenantAppeared as exc:
        # Exit 7: distinct from 5 (a family failed engagement) and 6 (a write escaped),
        # because the remedy is different and the data is not suspect -- everything measured
        # BEFORE the tenant is still good. The operator waits for the card and re-runs.
        exit_code = 7
        warnings.append(f"run stopped: {exc}")
        for r in arm_recs.values():
            r["tenant_contaminated"] = list(tenant_contaminated)
        save()
    except KeyboardInterrupt:
        how = TERMINATED_BY_SIGNAL or "KeyboardInterrupt"
        warnings.append(f"run terminated early: {how}")
        log(f"!! TERMINATED ({how}) -- writing what exists and stopping.")
        log("!! Everything measured before this point is intact and staged; re-running "
            "resumes from it. If this was SIGHUP, run detached next time (see --help).")
        for r in arm_recs.values():
            r["terminated_by"] = how
        exit_code = max(exit_code, 1)
    except Exception as exc:  # noqa: BLE001 -- a driver must not die silently mid-run
        warnings.append(f"run aborted: {type(exc).__name__}: {exc}")
        log(f"!! RUN ABORTED: {type(exc).__name__}: {exc}")
        import traceback
        log(traceback.format_exc()[-3000:])
        exit_code = max(exit_code, 1)

    # --- closing sequence -----------------------------------------------------------
    # Interrupt-safe. The closing sequence used to sit OUTSIDE every handler, so a signal
    # arriving here -- during hwinfo, the end-of-run tenant check or the campaign canary,
    # all of which shell out to nvidia-smi and to sha256 over 28 files -- escaped straight
    # past main() and left the summary saying `live: true, terminated_by: null`, i.e. it
    # looked like a run still in progress rather than one that was killed. Reproduced by
    # SIGTERMing this driver mid-close. Everything below is best-effort: a second signal
    # must not stop the record from being written.
    hw_end, drift = [], []
    try:
        hw_end = R.hwinfo() if not args.list else []
        drift = R.hw_drift(hw_start, hw_end) if hw_end else []
        if not args.verify_only and not args.dry_run:
            R.check_tenants(log, gpu, "at the end of the run", warnings)
    except KeyboardInterrupt:
        how = TERMINATED_BY_SIGNAL or "KeyboardInterrupt"
        warnings.append(f"terminated during the closing sequence: {how}")
        log(f"!! TERMINATED ({how}) during the closing sequence -- recording it anyway.")
        for r in arm_recs.values():
            r["terminated_by"] = how
        exit_code = max(exit_code, 1)
    diffs = check_campaign(fp, args.results_dir, warnings)
    if diffs and not canary_tripped:
        canary_tripped = True
        exit_code = 6
        log("!" * 92)
        log("!! CAMPAIGN CANARY TRIPPED at the end of the run:")
        for d in diffs:
            log(f"!!   {d}")
        log("!" * 92)

    R.rule(log, "STATUS")
    for ln in status_table(arm_recs, scope, arms):
        log(ln)

    incomplete = sum(len(st.get("regimes_missing") or [])
                     for rec in arm_recs.values()
                     for st in (rec.get("fam_stages") or {}).values())
    not_run = sum(1 for rec in arm_recs.values() for k in fams
                  if k not in (rec.get("fam_stages") or {}))
    if exit_code == 0 and (incomplete or not_run) and not args.dry_run:
        # A dry run measures nothing on purpose; reporting that as "partial data worth
        # shipping back" would train the operator to ignore exit 1.
        exit_code = 1

    log("")
    R.rule(log, "SEND BACK")
    log(f"  1. {args.results_dir / STAGING_ROOT}/           "
        f"(the WHOLE tree, including {ENGAGE_DIRNAME}/ and any {SENTINEL_NAME} sentinel --")
    log("     ESPECIALLY if the run failed)")
    log(f"  2. {summary_path}")
    log(f"  3. {logdir}/           (driver.log, preflight.log, one log per family per arm)")
    log(f"  4. {R.PREFLIGHT_JSON}")
    log("")
    log("  send the logs even when -- especially when -- the run was incomplete.")
    log("")
    log("  then, locally:  python3 glm52/make_control_report_h200.py")
    log("")
    log(f"  exit code {exit_code}")

    summary = build_summary(live=False)
    summary["hwinfo_end"] = hw_end
    summary["hwinfo_drift"] = drift
    summary["campaign"]["fingerprint_ok"] = not canary_tripped
    if canary_tripped:
        summary["campaign_canary"] = diffs
    if not args.dry_run:
        try:
            atomic_write_json(summary_path, summary)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            log(f"!! could not write the final summary: {type(exc).__name__}: {exc}")
    log.close()
    return exit_code


def _install_signal_handlers() -> None:
    """Turn a kill into a RECORD instead of a silence.

    The 2026-08-11 run died 10 minutes into f04f05 and wrote nothing at all: no traceback,
    no exit line, no summary update -- the last line in driver.log is a routine heartbeat.
    That is the signature of an uncatchable-by-default termination (SIGTERM from an operator
    or a scheduler, SIGHUP from a dropped SSH session, or the host OOM killer), and it left
    the incident undiagnosable from the artefacts: we cannot tell which of those it was.

    SIGTERM and SIGHUP are catchable, so catch them, raise KeyboardInterrupt to unwind
    through main's existing handler (which saves the summary and writes the closing status),
    and record which signal it was. SIGKILL cannot be caught by anyone; the defence against
    that one is the incremental `save()` after every family plus the pessimistic `verified`
    default, so a killed arm reads as unverified rather than as finished.
    """
    def _bail(signum, _frame):
        name = signal.Signals(signum).name
        print(f"\n!! {name} received -- writing what exists and stopping.", flush=True)
        globals()["TERMINATED_BY_SIGNAL"] = name
        raise KeyboardInterrupt(name)

    for _sig in ("SIGTERM", "SIGHUP", "SIGINT"):
        h = getattr(signal, _sig, None)
        if h is not None:
            try:
                signal.signal(h, _bail)
            except (OSError, ValueError):
                pass          # not the main thread, or the platform disallows it


if __name__ == "__main__":
    _install_signal_handlers()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(f"\n interrupted ({TERMINATED_BY_SIGNAL or 'KeyboardInterrupt'})", flush=True)
        raise SystemExit(130)
