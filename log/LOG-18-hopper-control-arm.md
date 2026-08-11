# LOG-18 — the Hopper control arm: does turning the sm_90 levers off change anything?

**Date** 2026-08-10 · **Status** IN PROGRESS (scripts written; awaiting the operator's run)
· **Trigger** the H200-vs-C500 comparison found that the three TMA-offered families
(`f01`, `f06`, `f08f09`; six variants) improved LEAST — medians 0.98–1.06x — while `f03` and
`f10`, which are offered no Hopper axis at all, improved 1.86x and 1.48x · **Verdict** *(unfilled — nothing has been measured)*

---

## 1. The question

`glm52_h200/kernels/hopper.py` gates three sm_90-only levers behind runtime capability
detection: TMA tensor descriptors, warp specialization via `tl.range(warp_specialize=True)`,
and thread-block clusters via `num_ctas`. The campaign records, per family, which of the three
the tuner was *offered*. Reading those flags out of the committed result files and then
dividing each family's H200 speedup by its C500 speedup, over the seven regimes the two
devices share, gives this:

| family | axes OFFERED by the campaign | median H200 gain / C500 gain | min–max |
|---|---|---|---|
| `f01` #1 o_proj+ResAdd | TMA, warp-spec, clusters | **0.997** | 0.993–1.197 |
| `f06` #6 UpGate+SwiGLU | TMA, warp-spec, clusters | **1.064** | 1.025–1.714 |
| `f08f09` #8 atomic | TMA, warp-spec, clusters | **1.017** | 0.982–1.168 |
| `f08f09` #9 atomic | TMA, warp-spec, clusters | **1.027** | 0.983–1.352 |
| `f08f09` #8 token-major | TMA, warp-spec, clusters | **0.977** | 0.502–1.563 |
| `f08f09` #9 token-major | TMA, warp-spec, clusters | **0.976** | 0.512–2.058 |
| `f04f05` #4 F4 | warp-spec, clusters | 3.639 | 1.235–3.760 |
| `f04f05` #4 F4_topk | warp-spec, clusters | 7.530 | 1.704–8.210 |
| `f04f05` #5 F5 | warp-spec, clusters | 3.053 | 1.681–3.321 |
| `f04f05` #5 F5_topk | warp-spec, clusters | 7.385 | 2.593–8.018 |
| `f11` #11b router | warp-spec | 2.425 | 1.269–2.835 |
| `f11` #11a w13 | warp-spec | 1.033 | 1.027–1.624 |
| `f03` #3 ResAdd+RMSNorm | **none** | **1.861** | 1.010–1.989 |
| `f10` #10 ExpertMerge+ResAdd | **none** | **1.479** | 0.997–1.856 |

(Computed from `report_glm52_c500/fusion_*.csv` and `report_glm52_h200/fusion_*.csv` over
`decode_bs1/32/256/512/1024` and `prefill_t2048/t8192`; `f04f05`, `f11` **and the two
`f08f09` token-major variants** have 5 of those 7 on the C500 side — `decode_bs512` and
`decode_bs1024` carry no rows for any of them there. `#11b′` half-fused reads 1.385 and is
omitted here because it publishes
`half_fused.router_speedup_vs_unfused` directly rather than through `best_speedup`, so it does
not join the same way. The "1.00–1.06x" band this work was briefed with is the four TMA-family
medians that round into it (`f01` measures 0.997 and only reads as 1.00 at 2 dp); the two
token-major variants sit slightly below it, on 5 regimes not 7.)

The ordering is the wrong way round. All six variants offered all three Hopper levers land in
**0.976–1.064** — near "the H200 helps exactly as much as it helps everything else" — while the
two families `hopper.py` cannot reach at all (their JSON records `kernel_cfg_keys: "module
advertises none"` and `not_offered_because: "this kernel module advertises no cfg key for it"`
on all three axes) gained the most of the memory-bound set.

**This is a correlation over a confounded comparison and it cannot support any claim.** The
H200 beats the C500 on every axis this study touches: 4245 GB/s of streaming bandwidth against
1300, Triton at 94 % of vendor GEMM against 50 %, a 0.032 µs event tick against 0.256 µs. A
cross-device ratio measures all of that at once, so every H200/C500 number is a sum of "the
device is faster" and "the Hopper path did something" with nothing separating the terms. It is
entirely consistent with the table that the Hopper features are worth 30 % on `f01` and that
`f01`'s fusion loses most of its C500 advantage for an unrelated reason. The decisive
experiment is a control arm: rerun the same benchmarks on the same device with the Hopper
capabilities forced off, and see whether the numbers move.

## 2. Mandate (verbatim from the user, 2026-08-10)

The operator's three decisions, recorded before any code was written, with the decisive words
bolded:

1. **Design: classic-only against the existing campaign.** Run only the classic arm now and
   diff it against the already-committed `results/h200/*.json`. I recommended a same-session
   paired A/B — both arms alternating on one pinned card inside one process lifetime, which
   is the design that removes the confound rather than bounding it — and the operator chose
   otherwise. That is their call. §4 records what the choice costs and how it is mitigated;
   this is a decision record, not an argument.
2. **Scope: all 7 bench families, all 11 regimes, NO whole-layer bench.** `bench_layer.py`
   carries a **16 h** timeout (`run_h200.py:172` — the brief said 18 h; the file says 16) and
   the question is per-fusion, so the whole-layer combination sweep answers nothing here and
   would nearly double the arm.
3. **Arms: the script must SUPPORT `classic, no-tma, no-ws, no-clusters, hopper` but default
   to `classic` only.** The per-feature decomposition arms exist behind `--arms` for a
   follow-up and are not run now.

## 3. What the levers actually are, and what the control arm actually turns off

The switch already exists. `hopper.py:721-723`:

```python
if _env_tristate("GLM52_H200_CLASSIC", notes) is True:
    overrides = {k: False for k in CAP_NAMES}
    notes.append("GLM52_H200_CLASSIC=1: all Hopper features forced off (control arm)")
```

and it is already wired to the driver: `run_h200.py:1193-1194` maps the `--disable-features`
tokens `all` and `classic` onto `GLM52_H200_CLASSIC=1`, and `_FEATURE_ENV`
(`run_h200.py:1180-1188`) maps `tma`/`ws`/`warp_specialize`/`clusters`/`wgmma` onto the
individual `GLM52_H200_*=0` variables. The module docstring at `hopper.py:119` calls
`GLM52_H200_CLASSIC=1` "the control arm, and the fastest way to prove a Hopper-path result is
not an artefact".

**It has never been run.** Before this build, `grep -rl GLM52_H200_CLASSIC` returned exactly
two source files — `run_h200.py` and `glm52_h200/kernels/hopper.py` — and their `.pyc`; zero
hits under `results/`, `report_glm52_h200/` or `log/`. It now also matches this build's own
files, but `results/` and `report_glm52_h200/` stay at zero until the operator runs. Not one
measurement in this study has ever been taken with the Hopper path disabled: the affordance was
built, documented, and left unused.

**Who is offered what**, read out of `fairness.h200_axes.per_family` in the committed files
rather than from the kernel sources:

| per_family key | file | tma | warp_specialize | clusters | overlays offered |
|---|---|---|---|---|---|
| `f01_oproj_resadd` | `f01_oproj_resadd.json` | ✓ | ✓ | ✓ | 4 |
| `f06_upgate_swiglu` | `f06_upgate_swiglu.json` | ✓ | ✓ | ✓ | 4 |
| `f08f09_down_merge` | `f08f09_down_merge_resadd.json` | ✓ | ✓ | ✓ | 4 |
| `f04f05_norm_router` | `f04f05_norm_router.json` | — | ✓ | ✓ | 2 |
| `f11a_w13_gemm` | `f11_lazy_prenorm.json` | — | ✓ | — | 1 |
| `f11b_router_gemm` | `f11_lazy_prenorm.json` | — | ✓ | — | 1 |
| `f11_norm_kernel` | `f11_lazy_prenorm.json` | — | — | — | 0 |
| `f03_resadd_rmsnorm` | `f03_resadd_rmsnorm.json` | — | — | — | 0 |
| `f10_merge_resadd` | `f10_merge_resadd.json` | — | — | — | 0 |

`f11`'s missing cluster axis is not "the kernel declined" but a *verified refusal*:
`lazy_prenorm.py:608-620` (`AXES_DECLINED`) records that `num_ctas > 1` is legal for the
unfused and half-fused arms and illegal for the fused one — under a CGALayout the cross-lane
`tl.reduce` computing the sum of squares lowers to `nvvm.mapa` on addrspace-3 scratch, which
`ConvertTritonGPUToLLVM` rejects, confirmed by cross-compiling 12/12 configs for sm_90 — so
sweeping it would hand one arm of the pair a larger grid than the other. It is withheld from
**both** arms deliberately. A verifier expecting all three axes to flip for `f11` would fail a
perfectly good control arm; only `warp_specialize` changes there.

**The correction that matters for attribution.** `CAP_NAMES` at `hopper.py:170` is
`("tma", "warp_specialize", "clusters", "wgmma")` — four names, and `GLM52_H200_CLASSIC=1`
forces all four off, one more than the three levers the question names. The control arm is
therefore *broader* than the question, and **no `f01`/`f06`/`f08f09` delta from this arm can be
attributed to TMA, warp specialization or clusters alone**; the `tl.dot` bf16 path is off too.
The per-feature arms are the answer to that and they are behind a flag by operator decision, so
the headline may end up genuinely un-attributable. The report must say so rather than pick the
convenient attribution.

One residual, stated rather than papered over: `common.features()["dot_bf16"]` stays `true`
under classic, because `dot_bf16` is a key of the features dict (`common.py:414-431`) but not a
token of `_FEATURE_ENV`. Adding it to the string would make `run_h200` print an
`!! unrecognised` line (`run_h200.py:1199-1201`, a bare `print` that never reaches
`driver.log`), so it is left alone and documented. The verifier does not read `features.*` at
all — see §5.

## 4. The confound, and the only defence against it

Decision 1 gives no same-session Hopper arm. The baseline is a campaign that ran on
2026-08-07 11:46–13:33; the control arm will run on some later day. Between the two sessions
sit thermals, co-tenancy on the other seven cards of the host, clock state, driver state and
whatever else moved. **Cross-session drift is confounded with the effect, structurally, and no
amount of care in the driver removes it.** This study has already been bitten by exactly this
class of thing twice: the RTX 4060 drifted 22 % thermally *within* a single run and produced a
speedup above its own physical ceiling, and the first H200 preflight ran on a card another
tenant held ~51 GB of and returned a 40.55 µs harness floor with a 0.03 tick-match fraction —
which changes verdicts, not just noise.

The mitigation is not a footnote; it is the entire reason this design is interpretable at all.
**`f03` and `f10` advertise no Hopper cfg key, so their classic arm is byte-identically
configured to their Hopper arm.** Their files record `kernel_cfg_keys: "module advertises
none"`, all three axes `offered: false`, `overlays_offered: []`, and a full key-path scan finds
zero Hopper tokens in any winning config, tune table or `axis_counts` dict. Nothing
`GLM52_H200_CLASSIC=1` turns off was ever reachable by these two kernels, so whatever their
classic-vs-campaign delta turns out to be it is **not** an effect of the Hopper features — it
is run-to-run and cross-session variation, on the same card, through the same harness, on the
same two kernels. **That delta IS the noise floor**, and every other family is judged
against it.

The statistic, decided now so it cannot be chosen after the numbers are in:

- Per cell, the signed **relative** delta `d = classic_speedup / hopper_speedup − 1`. Relative
  and not absolute, because `f03` reads ~2.16x at `decode_bs1` and `f10` lives on a different
  scale; an absolute band would be whichever family is larger.
- **Band = [min(d), max(d)]** over the 22 samples (2 families × 1 variant × 11 regimes). With
  n = 22 a percentile is barely resolvable — p10 and p90 interpolate between the 2nd/3rd and
  20th/21st order statistics — and the question is "could cross-session drift alone have
  produced this?", whose conservative answer is the observed extremes. p10/p90 is computed and
  published alongside, and `--band p10p90` switches which one drives the verdict column, but
  the default and the headline are min/max.
- **`median(d)` is the drift bias**: a systematic session offset that every family's delta
  should be read relative to. If the whole card ran 3 % slow that day, `f03` and `f10` will
  say so.
- **Class bands.** Decode and prefill have different launch-latency regimes, so `band_decode`
  (9 regimes × 2 = 18 samples) and `band_prefill` (2 × 2 = 4) are computed too. A cell is
  judged against its own class band when that class has ≥ 8 samples. Decode qualifies;
  **prefill does not**, so prefill cells fall back to the global band — which is dominated by
  decode's launch-latency-bound variance and may therefore be inappropriately wide for them.
  Recorded as a known weakness, not hidden.

And the assumption underneath all of it, stated plainly because it is an assumption and not a
measurement: `f03` and `f10` are both short memory-bound vector kernels. Whether their
run-to-run variability bounds the variability of a 14-hour GEMM family like `f08f09` is not
something this design establishes. It is the best available control under the chosen design.

## 5. Verification bar

**An unverified control arm is worse than no control arm.** An arm that silently did not engage
is indistinguishable from "the Hopper features made no difference" — this study's hypothesis.
It would not read as a broken run; it would read as a finding. So engagement is a hard gate,
checked per family the moment that family's JSON lands, and a failure aborts rather than
collecting 50 more hours of unusable data.

The precedent is LOG-14 B2, same suite, same escape hatch: `--disable-features` once set only
`GLM52_H200_DISABLE_FEATURES`, which mutates `common.features()` — *metadata* — while the real
gates read `GLM52_H200_TMA` / `_WS` / `_CLUSTERS` / `_CLASSIC`. So `--disable-features tma`
left TMA fully live **and wrote `"tma": false` into every result file**. On a machine nobody
can log into that is the worst possible failure shape. It is fixed; it is also the exact defect
this verifier exists to catch a recurrence of.

The bar, per family, per arm:

1. **Capability forced off, sourced from env.** For every per_family key,
   `fairness.h200_axes.per_family.<K>.axes.<a>.available` must be `false` **and** its
   `evidence` string must contain `(source env)`. Today every one of the seven files reads
   `available: true` with `hopper.caps().tma=True (source preflight); preflight probe …`. This
   is the only per-family text that proves the env override reached `hopper.caps()`, and it is
   **non-vacuous for `f03` and `f10`** — their `available` flips too.
2. **`offered: false` with the right reason.** `not_offered_because` must change from
   `"this kernel module advertises no cfg key for it"` (the `f03`/`f10` string) or from absent
   (the offered families) to `"the live capability probe says it is unavailable"`. Also
   non-vacuous for the noise-floor families: the string must change.
3. **Zero Hopper tokens in any winning config**, fused and unfused, across all 11 regimes,
   tested by key **absence with a truthy value** — never by `cfg.get("USE_TMA") is False`. A
   scan of all seven committed files finds `"USE_TMA": false` zero times and
   `"warp_specialize": false` zero times: the keys are simply omitted when off, so an
   `is False` test passes vacuously on **both** arms and proves nothing. The `num_ctas` half
   of the rule is defensive, not measured: the same scan finds no `num_ctas` value of 1
   anywhere in the seven files — only 2, plus `axis_counts` tallies (46, 82, 98, 194, 27, 28,
   33, 37, 40, 44), which are counts of configs carrying the axis and not config values.
   `num_ctas` is nonetheless counted as engaged only when it is not in `(None, False, 1)`,
   because 1 is the semantic no-op.
4. **`axis_counts` collapsed.** Under `fairness.grids.<regime>.<arm>.<stage>`, every
   `axis_counts` dict must reduce to exactly `{"_total_cfgs": N}` — the most reliable assertion
   available, because it is a live count of what the tuner was handed rather than a claim about
   it. Campaign reference, `f01` at `decode_bs1`, identical for both arms:
   `{"USE_TMA": 91, "num_ctas": 46, "warp_specialize": 86, "_total_cfgs": 220}`. Joint and
   size-only stages carry only `{n_tried}` with `axis_counts` **absent** — treat missing as
   "not recorded", never as a violation. (Note the campaign's `grids` block covers its own
   7 regimes; the bs-extra merge added cells, not grids.)
5. **Scalar mirrors and overlays.** `overlays_offered == []` (campaign: 4 / 4 / 4 / 2 / 1 / 1 /
   0 / 0 / 0 per the §3 table), `tma_form == "none"` and `ws_mode == "none"` (campaign:
   `"device"` and `"range"` in all seven files). Clusters have no scalar mirror; that is an
   INFO, not a failure.
6. **`f11`'s caps dump**, the strongest smoking gun in the corpus and present only in that
   file: `rows[*].isolation_fuse_on_vs_off_same_cfg.{router,moe}.kernel_caps_report
   .hopper_caps` must show `sources.<cap> == "env"` for every forced cap, `any_hopper: false`,
   and the literal note `"GLM52_H200_CLASSIC=1: all Hopper features forced off (control arm)"`.
   The campaign value is `{"tma": true, "warp_specialize": true, "clusters": true,
   "wgmma": true, …, "sources": {"clusters": "preflight", "tma": "preflight", …}}`.
7. **Static ground truth and device unchanged.** `kernel_cfg_keys` must be byte-identical to
   the campaign's for the same per_family key — it is a static module attribute, and if it
   moved then somebody edited the kernels and this is not a comparison. `_meta.device` (a
   **string**, not a dict) must normalise to `NVIDIA H200` and `env.uuid` to
   `b2318e71-edbf-aa1b-dcf3-3cc3e6ea7db0` — bare in the family files, `GPU-` prefixed in
   `summary.json`, same card, which is why every comparison normalises first.
8. **Differential collapse**, the positive proof: the campaign must show a non-zero presence
   for the axis and this arm zero. If the campaign showed zero too the axis is recorded
   `NO-DIFFERENCE-AVAILABLE` — reported, not fatal. Converse, if the `hopper` arm is ever run:
   `f01`/`f06`/`f08f09` must show the tokens **present** with non-empty overlays, or that arm
   fails exactly as a non-engaging classic arm does.

**And the blocks that must NOT be trusted**, each a guaranteed false negative:
`_meta.harness_info.features.*` (reads `{"tma": true, "clusters": true, "warp_specialize": true,
… "disabled": []}` and is preflight-derived, so it stays `true` under the control arm),
`env.tma_supported` / `tma_host_supported` / `tma_device_supported` / `clusters_supported` /
`warp_spec_supported`, `env.feature_evidence.*`, and `fairness.preflight.features.*`. Also
`probe_mode`, `"skipped"` in both arms and a discriminator of nothing.

The verifier asserts on evidence strings produced by `glm52_h200/bench/__init__.py` and
`hopper.py`. If those are ever reworded it will fail a good control arm and, under the abort
rule, throw the run away — so every check quotes its producing location in a comment.

## 6. The vacuity guard

`f03` and `f10` pass check 3 trivially: with no Hopper cfg key, a token scan of their classic
arm finds zero tokens exactly as a scan of their *Hopper* arm does. **A clean scan there is not
evidence that anything was disabled**, and counting it as evidence would let the two families
that define the noise floor also certify the arm engaged — circular.

So the verifier computes a **per-family axis surface first**, from the CAMPAIGN's own `offered`
flags, iterating the `per_family` dict rather than string-building the key (`f08f09`'s key is
`f08f09_down_merge`, not `…_resadd`; `f11` has three). An arm's axes split into
`target = axes_off ∩ surface` and `vacuous = axes_off − surface`. A zero on a targeted axis is
a PASS; the same zero on a vacuous axis is recorded `VACUOUS` — `"the campaign never offered
this axis to this family; a clean scan shows nothing"` — and excluded from `engaged_axes`. An
empty `target` gives the verdict **`NOTHING-TO-DISABLE`**, not `ENGAGED`.

The noise-floor pair's engagement proof is capability-level instead, and genuinely
non-vacuous: checks 1, 2 and 5 read `available`, `not_offered_because`, `tma_form` and
`ws_mode`, all four of which change for them under `GLM52_H200_CLASSIC=1` even though no config
of theirs does. That is what proves the env var reached the two children that are the control.

## 7. What was built, and exactly what the operator runs

Two scripts and one output namespace, each separate for one reason (the placeholder READMEs,
`log/run_control_h200/.gitignore` and the `## 0c.` preamble land with them — see the Work log):

- **`run_control_h200.py`** — the driver the operator runs on the H200. It **imports**
  `run_h200` as `R` and reuses `resolve_gpu`, `check_tenants`, `read_preflight`,
  `run_preflight`, `banner`, `device_gate`, `another_bench_running`, `hwinfo`, `hw_drift`,
  `run_family`, `Log`, `collect_cells` and the rest wholesale — the single-implementation rule
  that `run_f11_h200.py` exists to protect. It does not shell out to
  `run_h200.py --disable-features all`, which really would run the arm correctly, because a
  subprocess offers no per-family verify hook (a non-engaging arm would cost the whole arm
  before anyone knew), no per-arm loop over one shared GPU decision, no access to `run_family`'s
  provenance record, and no arm-token validation.
- **`glm52/make_control_report_h200.py`** — the comparison generator, pure stdlib, local, no
  GPU. Joins the two sides cell-for-cell on `(fusion, variant, regime)` through
  `R.collect_cells`, the same function that produced the campaign's own `summary.json` cells,
  so parity is by construction rather than by a re-derivation that could drift.
- **`report_glm52_h200/control_arm/`** — a **subdirectory** of the device namespace, not a
  sibling: `report_glm52_<device>/` members diff row-for-row against each other
  (`report_glm52_h200/README.md:3-5`), and the control arm is a second reading of the *same*
  device. Precedent: LOG-17:317-318 proposed `_campaign1_20260805/` inside the same directory
  for exactly this reason.

**The invariant, in bold, because it is the one that destroys the experiment if broken:
nothing under `results/h200/_control_arm/` is ever merged into the campaign files. The control
arm is a DIFF, not an append, and there is no merge step and no script in this repo may add
one.** A bench writes its ENTIRE result file on `record()`, so pointing a partial run at
`results/h200/` silently deletes the campaign's other rows. Here that is worse than in the
`run_bs_extra_h200.py` case it echoes: the campaign files *are* the baseline, so overwriting
them destroys the comparison itself. Three mechanisms enforce it — a single `guard_write()`
gate that fatally refuses any path outside `_control_arm/`; a campaign canary (size/mtime/
sha256 over every non-underscore `*.json` directly in `results/h200/`, taken pre-launch,
re-checked after every family and at the end, exit 6 on change, which catches the one failure
intent cannot: a bench that ignores `GLM52_H200_RESULTS_DIR`); and no merge surface at all
(no `shutil.copy*`/`move`, no `merge`-named function, no `.pre_*` backup path). The driver's
docstring carries the same three; this is the decision record for *why*.

The staging tree is **correctness, not tidiness**. `common.py:101` defines
`CKPT_ROOT = RESULTS_DIR / "_ckpt"`, and `ckpt_load()` fences on device only — never on feature
state. A classic run pointed at `results/h200/` would replay the *Hopper* campaign's
checkpoints, pass the device fence, and republish Hopper timings as the control arm: a null
result that looks like a finding. Redirecting `RESULTS_DIR` moves the checkpoints with it,
which is the whole reason the arm gets its own directory.

**A caveat that is worse than it first looks, recorded so nobody relies on the wrong
mechanism.** The leading underscore protects against `R.find_result` and **nothing else**:
`find_result` globs non-recursively and filters filenames starting with `_`
(`run_h200.py:1060-1071`), but `R.quarantine_foreign_results` uses `results.rglob("*.json")`
and skips only `_quarantine_foreign_*` path parts (`run_h200.py:993-997`), so it descends
into `_control_arm/` and judges every file there on one key: `payload["_meta"]["device"]`
(`run_h200.py:1004-1005`).

Only the 7 per-family result JSONs carry that key, and they record `NVIDIA H200`, so they
survive. **The other two classes of staged file do not.** Checkpoints written by
`glm52_h200/common.py` `ckpt_save` (line 1482) put `device` at **top level with no `_meta`
key at all**, and the driver's `_engagement/<arm>/<family>.verify.json` payloads have no
`_meta` either. Both resolve to `got == ""`, which never equals the device string, so every
`_control_arm/<arm>/_ckpt/**/*.json` and every `_control_arm/_engagement/<arm>/*.verify.json`
**WILL BE MOVED** into `results/h200/_quarantine_foreign_<ts>/` the moment anyone runs
`run_h200.py` on that box while the arm is staged. That costs the arm its resume ability and
its entire engagement audit trail — the only evidence that `GLM52_H200_CLASSIC` engaged. This
is not hypothetical: `results/h200/` already holds six `_quarantine_foreign_*` directories and
their contents are exactly `_ckpt/**/*.json` files. The campaign canary does not catch it
either; it fingerprints only non-underscore `*.json` directly in `results/h200/`.

The driver is being fixed to stamp `_meta: {"device": …}` into every `.verify.json` it writes,
which saves the audit trail. **The checkpoints cannot be fixed without changing
`common.py:ckpt_save`, and they are not being changed, so the residual hazard stands.** The
only real fence is operational: `run_control_h200.py` never calls
`quarantine_foreign_results`, and the operator must not run `run_h200.py` on that box while
`_control_arm/` is staged. That instruction belongs in the SEND BACK block and in
`results/h200/_control_arm/README.md`, not only here.

### The commands, in order

On the H200, from the repo root — this is the whole thing: the control arm, all 7 families,
all 11 regimes, on the campaign's own card.

    python3 run_control_h200.py

Then, back here, with the returned tree in place and no GPU needed:

    python3 glm52/make_control_report_h200.py

Useful variants of the first:

    python3 run_control_h200.py --list                # print the plan, launch nothing
    python3 run_control_h200.py --verify-only         # re-check staged data, no GPU work
    python3 run_control_h200.py --arms classic,hopper # control + same-session baseline (2 arms)
    python3 run_control_h200.py --arms no-tma,no-ws,no-clusters   # the follow-up decomposition

Families run cheapest-failure-first. The campaign measured them at `f03` 57 s, `f10` 108 s,
`f01` 4 min, `f06` 15 min, `f08f09` 19 min, `f11` 19 min, `f04f05` 21 min — **~1.3 h per arm
measured**, against 57 h of *timeout ceiling* (3/4/8/8/10/10/14 h). The ceilings are the point
a child is killed, not an expectation; quoting them as elapsed time overstates the job by
~40x. The whole-layer bench's 16 h is excluded by design (`--families layer` → exit 2). The
order is load-bearing: `f03` and `f10` are the noise floor *and* the earliest engagement
check, so a non-engaging arm is caught within a couple of minutes rather than at the end. The driver logs it before the first
launch and warns if `f03` is absent.

The driver pins the campaign's physical card, `GPU-b2318e71-edbf-aa1b-dcf3-3cc3e6ea7db0`,
resolving its nvidia-smi index from `summary.json`. If that card is busy or absent it refuses
(exit 4) rather than silently using another — waiting is cheap; a different card makes the
delta uninterpretable. `--any-gpu` overrides, records `DEVICE ANCHOR LOST` in the summary, and
the report generator turns that into a refusal the reader has to override deliberately.

### Send back, and send it even when the run failed

1. `results/h200/_control_arm/` — the whole tree, including `_engagement/` and any
   `ARM_NOT_VERIFIED` sentinel. **Especially** if the run failed: a control arm that did not
   engage is evidence, so the staged JSON is never moved or deleted; the sentinel is the fence,
   and it is what makes the report generator refuse to publish.
2. `results/h200/_control_arm/control_arm_summary.json`
3. `log/run_control_h200/` — `driver.log`, `preflight.log`, one log per family per arm.
4. `glm52_h200/preflight_h200.json`

**And one prohibition, until the tarball is off the box: do not run `run_h200.py` there while
`_control_arm/` is staged.** Its `quarantine_foreign_results` sweep will move the staged
`_ckpt/` and `_engagement/` files, which are not device-fenced — see the caveat above.

Child logs run 6–10 MB each (`log/run_h200/`: f01 2.5 MB, f06 6.3, f08f09 6.3, f11 6.1) and
`f04f05.log` once hit **107.7 MiB**, over GitHub's 100 MiB limit — five arms multiplies that.
`log/run_control_h200/.gitignore` is committed up front so an oversize transcript cannot block
the commit, and the compile-failure spam is itself a result: `tools/split_h200_log.py` splits
and gzips it (only 189 of that file's 1,240,800 lines are log; the rest is Triton's
warp-specialization pass failing on sm_90), and the gz is kept. The raw file is ignored, never
deleted.

### The baseline, named so it cannot move

The comparison is against the H200 campaign v2 (`6d699b4` "h200 done", 2026-08-07 13:33:48
+0800) as extended by the bs-extra merge of 2026-08-10: 7 family files, 165 cells over 15
(fusion, variant) groups × 11 regimes, all paired, harness floor **36.914 µs** identical across
all seven families, timer tick 0.032 µs matching 100 % of samples, card
`GPU-b2318e71-…-3cc3e6ea7db0`, `gpu_was_idle: true`.

**Action item before the operator runs anything:** at the time of writing, `git status` shows
`results/h200/summary.json` (+1776 lines) and eight `report_glm52_h200/*.csv` — the seven
`fusion_*.csv` plus `layer_optimal_per_regime.csv`, with `fusion_decode_bs{2,4,8,16}.csv` still
untracked — as **modified and uncommitted**: the bs-extra merge is in the working tree, not in
`5cd815c`. "The already-committed results" is therefore not yet a fixed target. Commit them
first and record the resulting hash here; the campaign fingerprint the driver takes is
meaningless against a baseline that is still moving.

## 8. How to read the output, and what it cannot say

The report publishes exactly three per-cell verdicts against the band and one wording rule per
verdict, enforced by making the CSV carry the token and the README carry one sentence template
for it:

- **`OUTSIDE-HIGH` / `OUTSIDE-LOW`** — the delta exceeds the drift band. This is the only
  positive claim the design supports: something changed by more than cross-session variation
  on the two feature-blind families accounts for.
- **`INSIDE`** — rendered as **"inside the drift band — nothing is shown"**. Never "no effect",
  never "confirms". A family whose delta sits inside the band has produced no information, and
  that is a different statement from the features having done nothing.
- **`NOISE-FLOOR`** (`f03`, `f10`) — these cells *define* the band and have no verdict.
- `UNRESOLVED-ONE-SIDE`, `MISSING-CONTROL`, `MISSING-CAMPAIGN`, `UNMAPPED`,
  `INCOMPARABLE-BASIS` — excluded from every aggregate. A cell UNRESOLVED on either side has
  its delta computed from `speedup_raw` into a separate column and is never averaged into
  anything.

What this arm **cannot** tell anyone, no matter how the numbers land:

1. It cannot say the Hopper features "did nothing". Only "the delta exceeds the drift band" or
   "the delta is inside the drift band, so nothing is shown".
2. It cannot attribute an `f01`/`f06`/`f08f09` delta to TMA, warp specialization or clusters
   individually — `wgmma` is off too, and the per-feature arms were not run (§3).
3. It cannot separate the effect from cross-session drift beyond what the 22-sample band
   bounds; prefill has only 4 of those samples (§4).
4. It cannot claim thermal equivalence between the arms. Under classic, `h200_cfg_overlays()`
   returns `[]`, so the tuning grid collapses and the arm should run **faster** — a different
   wall time means a different thermal trajectory across the run. The driver flags the reverse
   case (an arm exceeding the campaign's wall time by >50 % is a signal something is wrong),
   but a systematically cooler card is a confound the `f03`/`f10` band only partly absorbs.

## 9. What would falsify this

Each of these makes the exercise say nothing, and each maps to a named refusal with an exit
code rather than to a caveat in prose:

| condition | who refuses | code |
|---|---|---|
| engagement FAILED for any (arm, family) | driver: `ARM_NOT_VERIFIED` sentinel + abort; report refuses to publish that arm | 5 |
| fewer than 16 of the 22 `f03`/`f10` cells usable | report — and `--force-unverified` does **not** override this one | 1 |
| device-anchor loss (campaign card absent, or a UUID mismatch between sessions) | driver refuses pre-launch; report refuses at publish | 4 / 1 |
| control-session harness floor > 50 µs, or `timer.match_frac < 0.9` | report — this is the contended-card signature that produced the impossible 40.55 µs preflight | 1 |
| any classic-arm cell with a non-empty `classic_axes_selected` | report, `ENGAGEMENT-BROKEN`, **no override** | 1 |
| a file under `results/h200/` changed during the run | driver campaign canary, immediately, even under `--continue-on-verify-fail` | 6 |
| another `glm52_h200/bench` process running | driver | 3 |

The floor *difference* between the two sessions is deliberately not a refusal. It is itself a
drift measurement and belongs in the README next to the noise floor.

---

## Work log

### 2026-08-10 — scripts written, nothing measured

The build's deliverables are `run_control_h200.py`, `glm52/make_control_report_h200.py`, the
two placeholder READMEs (`results/h200/_control_arm/README.md`,
`report_glm52_h200/control_arm/README.md`), `log/run_control_h200/.gitignore`, the `## 0c.`
preamble stub in `report_glm52_h200/README.md`, and this log — check the tree, not this
sentence.

What does **not** exist is **any measurement**. Every number in §1, §3, §5 and §7 is read out
of the already-committed campaign, the C500 report or the sources cited beside it; not one
comes from a run with the Hopper features off, because no such run has ever happened in this
repo (§3). Where a value is an expectation rather than a reading it says so in place — §5's
`num_ctas` guard is the one case, and it is labelled defensive. The `## 0c.` preamble carries
an unfilled verdict line, and the **Verdict** field in this file's metadata line is empty on
purpose.

#### Next

1. Commit the bs-extra working-tree changes so the baseline stops moving; record the hash in §7.
2. Operator runs `python3 run_control_h200.py` on `GPU-b2318e71-…-3cc3e6ea7db0`, then tarballs
   back the four §7 paths — failure artefacts included.
3. Locally: `python3 glm52/make_control_report_h200.py`.
4. Fill §1's **Verdict** line, append a dated Work log entry with the delta table and the band,
   fill the `## 0c.` preamble in `report_glm52_h200/README.md`, and flip **Status** from
   IN PROGRESS to the finding.

### *(appended when the data returns)* — results

To carry: the delta table in LOG-16 §8.4's form — regime rows, one column per family, values
bolded where they fall outside the band, with the `f03`/`f10` columns visually separated as the
band itself — then a one-paragraph prose reading with the verdict in bold. Then the engagement
table (which axes were offered, which collapsed, which were vacuous). Then, explicitly and by
name, the negative results: every family whose delta landed inside the band, each with the
sentence *"inside the drift band; nothing is shown"*.
