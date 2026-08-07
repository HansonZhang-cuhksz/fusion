# LOG-17 — H200 campaign v2: validity-gated re-run

**Date** 2026-08-06 · **Status** IN PROGRESS · **Mode** build (session resumed per user instructions)

## Mandate (verbatim from the user, 2026-08-06)

"Please proceed. You are at build mode now. Log your steps properly. You may create subagents
to help you. Do not assume, ask questions at any time."

Scope decided by Q&A with the user:
1. **Full clean re-run** of all 7 fusion families + whole-layer sweep on one pinned idle H200.
2. **Harness upgrades allowed** — close the tooling defects documented in LOG-14/15/16 and in
   `report_glm52_h200/README.md` (paired-estimator bug, missing ceiling keys, no calibration
   gate, `verify=lambda: (True, "")` hole).
3. **No ssh access to the H200**: deliver ONE self-contained script the user runs on the box;
   user copies back the results (single tarball).
4. **#11a: investigate and fix** the invariance violation; only report unmeasurable if the
   defect is unfixable (Triton codegen bug).

## Why the tables need a re-run (state on entry)

`report_glm52_h200/` is filled from runs of 2026-08-03..06 but carries documented validity
failures: 84/84 cells SEQUENTIAL (published stat is ratio-of-medians, not the paired p50 —
`common.py:1374` hardcodes `paired: False`), 21 cells ABOVE CEILING + 58 never ceiling-checked
(only f04f05/f11 benches record a `ceiling` key), f01/f03/f10 from a different card/day than
the rest, co-tenancy during f06/f08f09/layer, and #11 publishing 3 of 28 cells (25 blocked).

## Work log

### 2026-08-06 — session start
- [ ] Phase 0a: read `glm52_h200/kernels/lazy_prenorm.py` + `f11_publish.py` in full.
- [ ] Phase 0b: local repro of the #11a invariance violation on the RTX 4060.
- [ ] Phase 0c: root cause + kernel fix.
- [ ] Phase 0d: verification (invariance tol 1e-5, layer fp32 checks).
- [ ] Phase 1: harness upgrades (common.py, traffic.py, benches, run_h200.py).
- [ ] Phase 2: `run_h200_v2.py` (the one script the user runs on the H200).
- [ ] Phase 3: `tools/verify_campaign_v2.py` (local validation).
- [ ] Phase 4: generators + report README + this log when data returns.

### 2026-08-06 — #11a repair: SQ_MODE=4 (transposed independent load)
- [x] Phase 0a: kernel + tooling reading done (this entry starts from that state).
- [x] Phase 0b: local repro harness `tools/repro_f11a_invariance.py` (sm_89/4060).
- [x] Phase 0c: SQ_MODE=4 added to both kernels (all 4 loop copies) + structural evidence.
- [x] Phase 0d (local part): mode 4 passes correctness + invariance + repeat screens.
- [ ] Phase 0e: H200 probe (part of `run_h200_v2.py` sq-mode pre-study + invariance gate).
- [ ] Phase 1-4: pending.

**Mechanism confirmed by cross-compiled sm_90a TTGIR (evidence_sq_mode4_sm90.py):**
mode 0/3's `tt.reduce` input chain terminates at `ttg.local_load` of the SAME staged SMEM
memdesc (`%a_122`) that `ttng.warp_group_dot` consumes — the reduction shares the
pipeliner's A buffer with the MMA.  Mode 4's reduce input terminates at a fresh `tt.load`
(transposed `[BK, BM]` shape, so Triton cannot CSE it the way it merged mode 3's second
same-shape load).  Both kernels, WS and non-WS variants, sm_90a.

**Harness-correctness findings along the way (all my test-harness bugs, not kernel bugs):**
1. Router `b` must be the gate TRANSPOSED `[K, N]` (`gate.t().contiguous()`); passing the
   [N, K] gate reads garbage (huge values, 12 dropped columns).  Fixed.
2. moe BLOCK_M config and `moe_align_block_size` BLOCK_M must always match; mixing them
   misaligns expert segments and (with GROUP_M) indexes `expert_ids` out of bounds.
3. Invariance-partner choice: from BLOCK_M=128 the cross-boundary partner is 32 (64 is
   ON the threshold, so it is not "cross"), a 4x row change — the largest repartition.

**New measurement — the invariance screen's tolerance is dtype-dependent (affects B3):**
the invariant keys repartition the per-row fp32 reduce across lanes/rows; on the w13 bf16
output that moves ONE element by one bf16 ulp when it sits at a rounding boundary:
measured max_rel 8.6e-4..1.7e-3 (BLOCK_M 128->32, num_warps 4->16), while BLOCK_M
64<->128 and all BLOCK_N pairs stayed bit-exact; SQ_MODE=2 (tensor-core "reduce", no
tree) was bit-exact across every key — mechanism confirmed as the tree repartition.
Class bound ~2^-7 relative; defect class 0.37+; so screens need 1e-5 (fp32 outputs) /
2e-2 (bf16 outputs): 5x above worst legit, 18x below the defect class.  Applied to
`f11_publish.py` (`INVARIANT_TOL_BF16`) + kernel docstring + repro.

**Local verification result (69 valid cells, 16 skipped = 4060 SMEM legality):** all
SQ_MODE 0..4 pass correctness vs exact fp32 ref (rel<=2e-2), invariance (dtype-aware
tol), repeat (bit-exact, tol=0).  Mode 4 is numerically indistinguishable from modes
0/1/3 and bit-stable; structurally it is the only mode whose reduce does not read the
MMA's staged A copy.  The H200 probe decides whether that repairs the invariance.

---

## 2026-08-06 -- harness v2 upgrades (paired, ceiling, B4 gate) + verifier

All edits verified with `ast.parse`; paired-preference and gate behavior exercised
locally on the 4060 (real CUDA runs, not just imports).

### Benches now emit `ceiling` / `ceiling_with_launch` / `traffic_ratio_model`
`bench_f01_oproj_resadd.py`, `bench_f03_resadd_rmsnorm.py`,
`bench_f06_upgate_swiglu.py`, `bench_f08f09_down_merge_resadd.py` (per-variant F8/F9
key), `bench_f10_merge_resadd.py` -- via `B.traffic_ceilings(regime)` row lookups.
f04f05/f11 already had them.  `run_h200.collect_cells` passes the two extra ceiling
keys through to every cell.

### B4 calibration gate (NEW: `B.calibration_gate(args)` in bench/__init__.py)
- `calibrate_live()`: re-measures harness floor + per-launch cost RIGHT NOW (empty
  timed region = floor; marginal nop-launch = launch; floor must be positive and
  <= config.FLOOR_US_MAX=20 us, launch positive) -- ported from f11_publish.calibrate.
- `calibration_gate(args)`: refuses to start a bench unless (a) the preflight on disk
  is `trusted` (tick lattice, floor bars, device match) AND (b) the LIVE re-measurement
  passes.  A co-tenant arriving after preflight is caught by (b).
- `--skip-calib` added to `add_std_args` for smoke runs only; the gate prints a loud
  "must NOT use for reportable numbers" line when skipped.
- Wired into all 7 benches right after `B.banner(env)`.
- Locally: gate correctly refused the stale H200 preflight (floor 39.872 us, the exact
  bad value from the last campaign); `--skip-calib` and live calibration pass
  (4060 floor 2.05 us, launch 2.05 us).

### collect_cells paired preference (run_h200.py)
- Key loop reordered: `("paired_speedup", "speedup_paired", "speedup")` -- an explicit
  paired marker wins over the bare `speedup` key.
- Cells now carry `speedup_sequential` (the sequential ratio the paired number
  replaced, for auditability) and `ceiling_with_launch` / `traffic_ratio_model`.
- Synthetic-payload test: a row with speedup=2.0 + paired_speedup=2.05 selects 2.05
  (source=paired_speedup, paired=True); sequential rows still fall back correctly.

### BUG FIX (would have crashed the H200 run): PairTiming is not a dict
7 benches called `pair.get("paired_speedup_p50")` / `pairs[v].get(...)` / `pm.get(...)`
on `PairTiming` objects (no `.get`, no such field) -- AttributeError on the first
bench_pair'd row.  Fixed to `ratio_p50` / `ratio_trimmed` attribute access
(f08f09/f11 via `getattr(..., None)` where the pair may be the `{}` no-pair default);
`pair_meta` now stores `PairTiming.as_dict()` instead of the raw object (which
`json.dumps(default=str)` was silently mangling into a repr).

### tools/verify_campaign_v2.py (NEW)
Local gate over copied-back `results/h200/` before regenerating the report.  Checks:
completeness (12 variant groups x 7 regimes = 84 cells, no missing families, no
quarantine events), calibration (floor 0<f<=20 us, launch>0, tick match >=0.9 or
finer-than-tested), pairedness (ZERO sequential cells allowed), ceiling (zero
ABOVE-CEILING flags), drift (zero DRIFT flags), resolution (unresolved = INFO:
honest blank), device (H200), f11 files present.  Exit 0 = publishable.
Dry run against the stale results: 3 FAIL -- floor 37.669 us (co-tenancy), 84/84
sequential, 20 cells above ceiling -- i.e. it fails the old campaign on exactly the
defects it exists to catch.

### Report generators
`glm52/make_report_h200.py` dry-run on stale results: no diff (deterministic output),
already prefers paired ratios (`best_speedup`) and blocks f11a invariance rejects.
No generator changes needed for the new schema.

### Repo state
Restored the results/h200 files my 4060 `--summary-only` dry run quarantined
(device-gate did its job; moved back, git clean apart from intended source edits).

### Next
Run `python3 run_h200.py --gpu N` on an idle H200 (preflight runs first, benches gate
on it, cells come out paired + ceiling-gated), copy back `results/h200/` + preflight
+ `log/run_h200/`, run `tools/verify_campaign_v2.py` (expect 0 FAIL), then regenerate
`report_glm52_h200/`.

---

## 2026-08-07 H200 run returned: f11 crashed (0/7 regimes) -> full re-run required

### What came back (commit 80d1fe4 "h200 done", 2026-08-07 10:07:14 +0800)
- `log/run_h200/f11.log`: NEW harness ran on H200 (GPU-3aa1, CUDA_VISIBLE_DEVICES=7);
  B4 calibration gate PASSED at start (live floor 12.75 us, launch 14.57 us); SQ_MODE
  study ran; then EVERY regime's final timing crashed:
  `AttributeError: 'dict' object has no attribute 'ratio_p50'` at common.py:1386
  (`_paired_fields`).  `specialization_study` PassManager failures were recorded but
  not fatal.  Result: f11_lazy_prenorm.json complete=False, 0/7 regimes.
- `summary.json`: preflight ts 2026-08-06 19:01:29, floor **-5.58 us (negative, invalid)**
  -- the preflight itself ran on a contended machine (f11.log also shows 4x
  SwPowerCap "machine moved" events mid-run).  Device H200 GPU-3aa1, tick 0.032 us.
  84 cells but ALL 84 paired=False SEQUENTIAL + 19 ABOVE CEILING, missing_families
  ['f11'], quarantined=True (driver quarantined the foreign Aug-4 `_ckpt` dirs:
  recorded_device=None vs present NVIDIA H200).
- Only f11 has `started: 2026-08-06 19:01:40` in the new summary; f01..f10 carry
  `started: None` -> their cells come from the OLD Aug-4 top-level jsons (sequential,
  not the paired protocol).  So the returned campaign is NOT publishable and contains
  ZERO paired cells; verify_campaign_v2.py would fail it.

### Root cause (local repro on 4060 reproduced exactly)
There are TWO bench_pair implementations:
- `glm52_h200/common.py:672` `bench_pair` -> returns `PairTiming` dataclass.
- `glm52_h200/bench/__init__.py:1743` `B.bench_pair(fused_fns, unfused_fns, warmup,
  rep, label)` is a WRAPPER: calls common.bench_pair (or `_local_bench_pair`
  fallback), normalises to `(Timing, Timing, meta_DICT)` via `_normalise_pair` /
  `_canonicalise_pair_meta`, canonical keys `paired_speedup_p50`,
  `paired_speedup_trimmed_mean`, etc.  Every bench gets the meta DICT back.

The paired-upgrade edits (commit e155f94) had changed bench `.get("paired_speedup_p50")`
calls into `getattr(pair, "ratio_p50")` and passed `pair=` (the meta dict) down to
`speedup_row` -> `_paired_fields`, which did `p.ratio_p50` on a dict -> AttributeError.
(My earlier "f08f09 pairs[v].get fix" session had misread the wrapper as returning
PairTiming; the original `.get(...)` on the dict was CORRECT.)

### Fix (uncommitted in working tree, all verified locally on 4060)
- `common.py`: rewrote `_paired_fields` to accept EITHER a PairTiming or the canonical
  meta dict (new `_pfield(p, key, aliases, default)` helper reads
  paired_speedup_p50 / ratio_p50 / unpaired_speedup_of_medians /
  paired_speedup_p10_p90 / paired_speedup_trimmed_mean).  `machine` default {}.
- `common.py speedup_row`: `if pair is not None and isinstance(pair, dict) and not
  pair: pair = None` -> documented "no pair" default {} yields a sequential row.
- Reverted ALL benches (f01, f03, f04f05, f06, f08f09, f10, f11, layer) back to
  dict `.get("paired_speedup_p50")` / `.get("paired_speedup_trimmed_mean")` and
  `pair_meta: pair` (dict serialises as-is).  Removed the f11 debug print.
- Verified: f11 quick repro -> OK 1/1 regimes, row paired=True, speedup == paired
  median, pair_meta impl=common.bench_pair, n_pairs=12.  f01 quick repro -> paired
  1.0064x + vendor 1.0219x rows.  All 8 bench files + common.py compile.
- NOTE: my first local repro used the default results dir and overwrote
  results/h200/f11_lazy_prenorm.json -> restored from HEAD; subsequent repros used
  GLM52_H200_RESULTS_DIR=/tmp.  Working tree now: only the 9 source files modified.

### Next (H200 re-run)
1. Commit the fix, sync to H200 box (git pull), `python3 run_h200.py --gpu N` on an
   IDLE H200 (Aug-6 preflight floor was negative/contended; SwPowerCap throttling
   during f11).  The driver will quarantine the old foreign ckpts again; f01..f10
   MUST be re-measured (their top-level jsons are pre-paired-protocol Aug-4 data and
   the driver's device-fenced resume will skip them as "already exists on this
   device") -- use run_h200.py's --force-rerun flag so every cell is fresh + paired:
   `python3 run_h200.py --gpu N --force-rerun`.
2. Copy back results/h200/ + preflight + log/run_h200/.
3. `tools/verify_campaign_v2.py` (expect 0 FAIL), regenerate report_glm52_h200/.
