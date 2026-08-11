# LOG-18 — the Hopper control arm: does turning the sm_90 levers off change anything?

**Date** 2026-08-10, arm completed 2026-08-11 · **Status** ARM COMPLETE, COMPARISON WITHHELD —
**all 7 families are measured, staged and engagement-verified** (`f03`/`f10`
`NOTHING-TO-DISABLE`, the other five `ENGAGED`, `n_fail 0` everywhere, sentinel removed
2026-08-11 20:46:19). But **`f01` and `f04f05` are EXCLUDED as tenant-contaminated** — a
neighbour took 121.6 GB and 61.8 GB of the 143.8 GB card inside their measurement windows —
and the single cell `f10/decode_bs16` is excluded as a harness artefact. The two excluded
families are **exactly the two whose offered tuner grid actually collapsed under the switch**,
so the intended classic-vs-campaign comparison is **not published** — see the 2026-08-12 Work
log entry
· **Trigger** the H200-vs-C500 comparison found that the three TMA-offered families
(`f01`, `f06`, `f08f09`; six variants) improved LEAST — medians 0.98–1.06x — while `f03` and
`f10`, which are offered no Hopper axis at all, improved 1.86x and 1.48x · **Verdict**
*(withheld — drift band −6.07 %/+4.70 % over 21 cells; five clean families, of which only
`f11` had its offered grid change at all, and no exceedance that can be separated from
cross-session tuner variance. `f01` and `f04f05` must be re-measured on a clear card.)*

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

### 2026-08-11 — the arm ran, `f03` measured clean, and my own verifier threw the run away

The operator ran `python3 run_control_h200.py` on the H200 and committed the tree back as
`d63a3de` ("error control"). **The measurement worked. The verification did not.** `f03`
completed all 11 regimes in one minute and the driver then aborted the whole arm on a single
fatal check — `V9` — that asserts something that is not true of the quantity it reads.

#### What came back

From `log/run_control_h200/driver.log`, in order:

- `10:57:29` the driver **refused** the operator's first invocation (`--gpu 7`) because index 7
  is not the campaign card; they re-ran without it and it pinned `GPU-b2318e71-…` at index 0
  itself. `control_arm_summary.json` `device_anchor` reads `matched: true`, want == got.
- `10:57:37` preflight `exit=0`. `[calib] copy_2048MB_GBs=4247 rmw_2048MB_GBs=4262
  read_2048MB_GBs=4649 cublas_4096…_TFs=821.6`, `launch 10.27 us | timer tick 0.032 us`.
  Against the campaign's own preflight (`results/h200/summary.json`, 2026-08-07): copy_2048MB
  4246.84 → 4247.27 GB/s (+0.01 %), read 4647.74 → 4649 (+0.03 %), cublas 822.90 → 821.58 TF
  (−0.16 %), launch 10.3177 → 10.2720 µs. **The machine did not move between the two
  sessions**, measured by probes that touch none of the benches.
- `10:57:38` `f03` launched. `[hw before] GPU0 NVIDIA H200 | sm 1980/1980 MHz | mem 3201/3201
  MHz | 32 C | used 0/143771 MiB | util 0% | throttle 0x0000000000000000` — idle card, clocks
  pinned, no co-tenant.
- `10:58:40` `exit=0 status=ok wall=1.0 min`; `[redirect] confirmed (exact path match)`;
  `all 11 regime(s) in scope have rows; f03 is done.`
- `10:58:41` `[verify] f03/classic FAILED with 1 fatal check(s)` — and then, verbatim:

```
10:58:41 !! ENGAGEMENT VERIFICATION FAILED for classic/f03.
10:58:41 !! A control arm that did not engage is worse than no control arm: it is
10:58:41 !! indistinguishable from a positive result -- which is this study's hypothesis.
10:58:41 !!   V9   $.fairness.grids.*.*.*.n_tried
10:58:41 !!        want identical to the campaign
10:58:41 !!        got  19 of 49 stage(s) differ
10:58:41 !! sentinel written: .../results/h200/_control_arm/classic/ARM_NOT_VERIFIED
10:58:41 !! engagement record: .../_control_arm/_engagement/classic/f03.verify.json
10:58:42 !! aborting the whole run (remaining families and remaining arms).
```

and the `detail` string recorded in `f03.verify.json`, which is the sentence that made the
abort look authoritative:

> `f03/f10 grids are uncapped and carry no Hopper overlay, so a size change means the harness
> itself changed: decode_bs1/add/refine 88->50; decode_bs1/fused/refine 56->44;
> decode_bs1/norm/refine 50->38; decode_bs1024/fused/refine 62->54`

`V9` was the **only** failure: `verdict: FAILED, n_fail: 1`. Every other check passed or was
correctly recorded vacuous.

#### The early-abort design worked, and it should be kept

Say this plainly, because the temptation after a false positive is to loosen the gate itself
rather than the one wrong check inside it. `f03` runs **first** precisely so that a problem
surfaces in a minute instead of at the end of a 1.3 h arm, and that is exactly what happened.
The cost of this false positive was **~1 minute of GPU time plus one round trip** — not 1.3 h,
not seven families of unusable data. Had `V9` been non-fatal and the arm run to completion,
the same defect would have surfaced on `f10` too (§ below) and the operator would have found
out after the whole arm instead of after one family. The ordering rule in §7 is load-bearing
and stays. What changes is the content of one check, not the abort policy.

#### Diagnosis: V9 read a measured quantity and demanded bit-equality of it

Three independent diagnosis passes over the returned tree agree, and the finding is
demonstrated rather than argued.

**1. The offered grid did not change at all.** Split the 49 compared stages by stage name
(`results/h200/_control_arm/classic/f03_resadd_rmsnorm.json` vs
`results/h200/f03_resadd_rmsnorm.json`, through the driver's own `_grid_stages`):

| stage | compared | differ |
|---|---|---|
| `coarse` | 21 | **0** — every one at `n_tried` 164 on both sides |
| `refine` | 21 | 14 |
| `unfused_chain/joint` | 7 | 5 |

21 + 14 + 5 gives V9's 19 of 49 exactly. **Zero coarse differences.** The grid the bench
*offers* is byte-identical between the arms, which is the only part of the block a Hopper
overlay could ever have touched.

**2. The differences move in BOTH directions.** Systematic removal of Hopper configs shrinks
monotonically. These do not: `decode_bs1/add/refine 88→50`, `decode_bs32/add/refine 82→46`,
`decode_bs512/fused/refine 66→28`, `prefill_t8192/add/refine 66→28` — but also
`decode_bs512/norm/refine 28→46`, `decode_bs256/add/refine 54→66`,
`prefill_t2048/norm/refine 46→62`, `decode_bs32/norm/refine 50→56`,
`decode_bs256/unfused_chain/joint 12→16`. Range −38 to +18.

**3. The mechanism: `refine` is built from the coarse stage's TIMING WINNER.** The
preliminary diagnosis named `common.py:893 neighbours(best, grid, max_out=96)` and the
property is real there — `i = s.index(best[k]); moves[k] = [s[j] for j in (i-1, i+1) …]` — but
`f03` does not use it. `common.py:1069` reads `rg = refine(best_cfg) if callable(refine) else
neighbours(best_cfg, coarse, max_out=refine_max)`, and `f03` passes its **own** generator:
`bench_f03_resadd_rmsnorm.py:98-109` `refine_grid(best)` indexes `BLOCKS`/`ROWSET`/`WARPS` at
the winner and takes `(i-1, i, i+1)` clipped to the array. **A winner on an edge of the ladder
yields a strictly smaller neighbourhood than an interior one**, and `_stage()` additionally
skips configs already timed (`common.py:1017-1021`), so overlap with the coarse set moves the
count again. Worked example, from the `tune_tables` block of the two files:
`decode_bs1/add`, campaign winner `BLOCK_N=2048` (interior of
`BLOCKS=[512,1024,2048,4096,8192]`, three neighbours) → refine 88; classic winner
`BLOCK_N=8192` (boundary, two neighbours) → refine 50. The coarse `best_ms` that chose between
them: **0.006816 ms vs 0.006272 ms.** A half-microsecond picks a different lattice corner and
moves the stage size by 38 configs. A diagnosis pass re-implemented `refine_grid` offline from
the probed H200 constants and it reproduces the coarse size (164) and then **predicts all 42
observed refine sizes exactly, in both arms, from the winner alone — zero residual.** The
generator provably did not change; only the winner did.

The `joint` stage is worse: `bench_f03_resadd_rmsnorm.py:214-223` builds it as the deduplicated
Cartesian product of `B.top_cfgs(a_c, k=2) + B.top_cfgs(a_r, k=2)` against the `norm`
equivalent — i.e. of the **measured ranking** — so overlap between the coarse and refine top-2
collapses the product by a data-dependent amount.

**4. The campaign has no `fairness.grids` record for `decode_bs2/4/8/16`.** Confirmed: the
campaign block covers only the 7 original regimes (the bs-extra merge carried rows but no
grids block), the arm carries all 11 — 77 arm stages against 49 campaign stages. **But this
contributed nothing to the abort**, and the preliminary diagnosis had the mechanism backwards:
V9 compared `set(grid_now) & set(grid_was)`, so those 28 stages were **silently dropped**, not
counted as differences. The denominator 49 *is* the intersection. It is nonetheless a real hole
pointing the other way — 28 stages including **12 coarse** stages were checked by nothing, so a
classic arm whose bs-extra coarse grid had been halved would have passed V9. Recorded here
because a narrowing patch that did not also address it would have left the check net weaker.

#### Why the check was wrong in principle

`V9` asserted **bit-determinism of a quantity selected by p50 timings that differ run-to-run by
~1e-3 ms**. That is not a property the harness has or should have, so the check would fail a
Hopper-vs-Hopper rerun of identical code on identical hardware. This is not an inference — the
repo already contains the counterfactual.

`results/h200/_bs_extra_rerun/` is a full independent rerun of all seven families on the same
card **with the Hopper levers ON** (`log/run_bs_extra_h200/f03.a1.log`: `hopper feats
[tma,clusters,warp_specialize]`; `_bs_extra_rerun`'s f03 axes read `available=true` for all
three, against the classic arm's `false`). It is the campaign's own arm, run twice. Replaying
**V9's exact comparator** on it:

| pair | stages differing | of which coarse |
|---|---|---|
| `f03` classic arm vs campaign (the abort) | 19 of 49 | **0** |
| `f03` **Hopper rerun** vs campaign | 16 of 49 | **0** |
| `f10` **Hopper rerun** vs campaign | 18 of 49 | **0** |

**The old V9 would have failed the campaign against itself**, and — since `strict` was set for
exactly `f03` and `f10`, the two families the driver runs *first* — this abort was structurally
guaranteed on every invocation, not bad luck. Fixing it for `f03` alone would have bought the
operator exactly one more family before the next false abort.

Two corrections to the brief, so nobody re-derives them wrongly:

- The demonstration comes from `_bs_extra_rerun/`, **not** from the `pre_bs_extra_*` snapshots.
  Those are not independent runs: `f03_resadd_rmsnorm.pre_bs_extra_20260810_113328.json`
  replays V9 at **0 of 49** against the campaign and carries identical timings. It is a
  snapshot, not a second measurement.
- Widened to all seven families, campaign vs `_bs_extra_rerun` (both Hopper-ON) gives
  **231 offered-grid stages** (`coarse` 126 + `tune` 105; 238 counting the 7 `extra`) with
  **0 differing**, against **233 winner-derived stages** with **46 differing** (`refine` 42 of
  121, `joint` 4 of 112, `extra` 0 of 7). The split is total and it is not close.

And `GLM52_H200_CLASSIC=1` could not have caused it even in principle: `f03` is offered no
Hopper axis on **either** arm (`kernel_cfg_keys: "module advertises none"`, all three axes
`offered: false`, `overlays_offered: []`), `B.widen()` returns the grid untouched when the
overlay list is empty (`bench/__init__.py:1296-1312`), the child log records
`[h200 axes] none offered for glm52_h200.kernels.add_rmsnorm: grids are the classic ones`, and
`wgmma` has no consumer in the grid path at all — `add_rmsnorm.py:43` states the kernel has no
`tl.dot`. A three-way comparison closes the last door: over the 49 stages present in campaign,
Hopper-rerun and classic arm, the classic arm sides with the rerun against the campaign on 4
stages and with the campaign against the rerun on 3. **There is no arm-specific structure.**

#### The fix

Landed in `run_control_h200.py` by the concurrent repair pass (I did not edit that file; this
entry records the decision, the code is the authority).

1. **New module constant `OFFERED_GRID_STAGES = frozenset({"coarse", "tune"})`**
   (`run_control_h200.py:245`), written as an **allow-list, not a deny-list**, so a stage name
   a bench grows later defaults to "not comparable" rather than silently becoming a fatal
   equality assertion. This polarity is deliberate and must not be reverted: a deny-list of
   `{refine, joint}` would still have asserted on `f01`'s `extra` stage, which is
   `refine_grid(cfg)` over the fused side's own top configs
   (`bench_f01_oproj_resadd.py:400-406`) and therefore winner-derived too. The docstring names
   the producer line for every excluded stage.
2. **V9 now compares `n_tried` only over `coarse`/`tune`, and only where both sides recorded a
   value.** Still **fatal** for `f03`/`f10`. The assertion it now makes — *the OFFERED grid
   changed, which no arm switch can do* — is the one its own docstring always claimed to make.
3. **An empty comparison set is `INFO`, not `PASS`.** The old code computed `changed = []` and
   reported `PASS` against `want: "identical to the campaign"` — a silent vacuous pass on the
   only grid check that is fatal for the two noise-floor families. It now reads
   `NO COMPARISON AVAILABLE … absence is not agreement and is not disagreement`.
4. **New `V9d`, INFO only**: the refine/joint drift is still counted and printed, so the signal
   is not discarded, only demoted. On the returned `f03` it reads `19 of 28 stage(s) differ`.
5. The detail prose was rewritten. `"so a size change means the harness itself changed"` is
   gone; the premise (uncapped, no overlay) is true and is exactly why the **coarse** size is
   assertable, but the conclusion was being drawn over stages that are not offered grids.

**The general rule adopted, and it governs every future check in this verifier:**

> A fatal comparison may only read a quantity that is a **pure function of the code and the
> probed device** — capability flags, static module attributes, offered grid sizes, live
> `axis_counts`. Anything downstream of a *timing* is recorded as INFO and never asserted on.
> A record the campaign does not carry is **VACUOUS / NO-COMPARISON**, never FAIL — and never
> a silent PASS either.

**Discriminating power was not lost, tested both directions.** On a deepcopy fixture of the
staged file (built under the scratchpad; nothing under `_control_arm/` was written),
perturbing `fairness.grids.decode_bs1.add.coarse.n_tried` 164→100 still gives
`V9 FAIL "1 of 21 offered-grid stage(s) differ"`, and 164→820 (simulating overlays still being
applied) likewise. Judged as a classic arm, Hopper data still produces 8 fatal failures each
for `f03` and `f10` from `V1`/`V2`/`V4` alone — the capability-level checks, which are what
actually proves engagement for these two families and which now carry the whole burden. V9 was
never load-bearing for `f03`; it was surplus risk.

**The rest of the verifier was audited and is clear.** All 16 checks were run over all 7
families against two Hopper-vs-Hopper pairs and a bs-extra-only fixture; post-fix, **zero
arm-independent checks fire**. `V5` (static `H200_CFG_KEYS`), `V8` (live `axis_counts`), `V11`
(differential presence), `V6` (device/uuid), `V1`/`V2`/`V4` (capability-level), `V3`
(overlays), `V10` (`f11`'s caps dump) each read a capability-, config- or static-module-derived
quantity and each already treats campaign-side absence as INFO. `V9` was the only check in
`verify_family` asserting equality of a data-dependent quantity. One item flagged and
deliberately **left alone**: `V7` ("at least one offered axis appears as a winner") is
timing-downstream by construction, but it fires only on the `hopper` arm, which is not in
`DEFAULT_ARMS`, its margins are wide (winner counts 12–244 per family), and it produced no
false positive in the counterfactual. Weakening a check with no observed failure would be the
wrong trade; it is recorded so it is a known quantity if `--arms hopper` is ever run.

#### Re-verification of the returned data — no GPU needed

`verify_family` is a pure function of committed JSON. Re-run against the staged file with the
fixed code:

```
VERDICT NOTHING-TO-DISABLE   n_fail 0
V9  PASS  0 of 21 offered-grid stage(s) differ
V9d INFO  19 of 28 stage(s) differ
```

with `V0b` PASS (exact redirect path match), `V1`/`V2` PASS on all three axes, `V4` PASS
(`tma_form=none`, `ws_mode=none`), `V5` PASS, `V6` PASS (device + uuid), `V3`/`V7`/`V8`
VACUOUS as before. **`NOTHING-TO-DISABLE`, not `ENGAGED`** — the §6 vacuity guard is doing its
job: the arm is *proven* to have reached `hopper.caps()`, but for this family there was never
anything at config level to turn off.

That proof is independent of V9 and rests on four surfaces:

- `axes.*.evidence` reads `hopper.caps().tma=False (source env); preflight probe
  tma_tensor_descriptor=False` against the campaign's `…=True (source preflight)`. The
  **env-vs-probe disagreement** recorded in the `ws`/`clusters` strings — env says `False`,
  the live probe says `True` — is producible only by an override.
- `not_offered_because` flipped from `"this kernel module advertises no cfg key for it"` to
  `"the live capability probe says it is unavailable"`, exactly as §5 check 2 required.
- `tma_form` `"device"`→`"none"`, `ws_mode` `"range"`→`"none"`.
- `f03.classic.a1.log:5` `… | hopper feats [none] | preflight ok | results ->
  …/_control_arm/classic` and `:18` `tma=False ws=False clusters=False wgmma=False | src env |
  probe skipped`, against the campaign's `hopper feats [tma,clusters,warp_specialize]` and
  `src preflight`. **`src env` vs `src preflight` is the discriminator.**

`kernel_cfg_keys` stayed `"module advertises none"` on both sides, so nobody edited the kernels
between sessions, and `_meta.harness_info.features.disabled` went `[]` →
`["tma","ws","warp_specialize","clusters"]`.

#### What the operator does next — and `f03` is NOT re-measured

**The returned `f03` data does not need re-measuring.** It is 11/11 regimes, `complete: true`,
paired, on the campaign's own card, and the only thing wrong with it was the adjudication.

On the H200 box, after pulling the V9 fix:

    python3 run_control_h200.py --verify-only --families f03

No GPU work beyond one `nvidia-smi` snapshot, no child launched (`run_control_h200.py:2202`,
`:2471`). Expect `NOTHING-TO-DISABLE`, `n_fail 0`. The driver removes
`results/h200/_control_arm/classic/ARM_NOT_VERIFIED` itself when every family *in scope*
re-verifies (`:2687-2703`) — so scope it to `f03`, since the six unmeasured families would
otherwise fail `V0` "missing or unreadable" and keep the sentinel. Then:

    python3 run_control_h200.py

and **note the three traps**:

- **Drop `--gpu 7`.** The driver refused it on 2026-08-11 because it contradicts the campaign
  card at index 0, which it pins automatically.
- **Do not pass `--force-rerun`.** `f03`'s staged file already has all 11 regimes at mult 1, so
  `run_family_stage` reports "f03 is done" and skips it at zero cost. `--force-rerun` would
  discard the very data this entry validates and re-measure from zero.
- **Do not delete the sentinel by hand while it is still present** and then relaunch expecting
  a measurement — with the sentinel in place the driver takes the `resume_verify_only` path
  (`:2436`) and launches **nothing**, so a plain relaunch would re-verify `f03`, hit `f10` with
  no staged JSON, fail `V0`, and measure nothing at all. Order matters: verify, then let the
  sentinel be removed, then relaunch.

Expected cost for the six remaining families: **~1.3 h** (campaign wall times `f10` 108 s,
`f01` 4 min, `f06` 15 min, `f08f09` 19 min, `f11` 19 min, `f04f05` 21 min). Then, locally,
`python3 glm52/make_control_report_h200.py`.

The §7 prohibition still stands: **do not run `run_h200.py` on that box while `_control_arm/`
is staged** — its `quarantine_foreign_results` sweep will move the staged `_ckpt/` files.

#### Status of the measurement itself

`f03`, 11 of 11 regimes, and it is **the noise floor** the whole design rests on. Paired
speedup, campaign → classic arm (`rows[*].speedup`, both files):

| regime | campaign | classic | d |
|---|---|---|---|
| `decode_bs1` | 2.1644 | 2.2407 | **+3.52 %** |
| `decode_bs2` | 2.1522 | 2.0434 | **−5.06 %** |
| `decode_bs4` | 2.1727 | 2.1736 | +0.04 % |
| `decode_bs8` | 2.1543 | 2.1943 | +1.86 % |
| `decode_bs16` | 2.0986 | 2.1911 | **+4.41 %** |
| `decode_bs32` | 2.2024 | 2.1020 | **−4.56 %** |
| `decode_bs256` | 2.0325 | 1.9676 | −3.19 % |
| `decode_bs512` | 2.0513 | 1.9628 | **−4.31 %** |
| `decode_bs1024` | 1.9486 | 1.9706 | +1.13 % |
| `prefill_t2048` | 1.9597 | 1.9327 | −1.38 % |
| `prefill_t8192` | 1.3287 | 1.3351 | +0.48 % |

mean **−0.64 %**, median **+0.04 %**, sample stdev **3.31 %**, max |d| **5.06 %**, **no sign
flip anywhere**. On the 7 regimes that have a same-session campaign baseline the classic arm's
spread (max |d| 4.56 %, stdev 3.04 %) is **smaller** than the Hopper-vs-Hopper repeat's
(max |d| 5.83 %, stdev 3.39 %) — the arm deviates from the campaign by *less* than identical
code does. This is half of the §4 band (`f10` supplies the other 11 samples), so the band is
not yet computed and §4's `median(d)` drift-bias figure stays unfilled.

**Publish the ratio, not the milliseconds.** Raw times moved much further — `fused_ms` −15.1 %
to +29.6 %, `unfused_ms` −11.6 % to +22.2 % — but they move the *same sign on both sides of
every pair* (`decode_bs256` fused +29.62 % / unfused +21.56 %), which is correlated session
drift, not an arm effect; the Hopper-vs-Hopper repeat shows the same behaviour. For scale, the
p10–p90 spread *inside* a single cell is ~3× (`decode_bs1` classic fused p10 0.0364 / p90
0.1077 ms) and `ratio_halves` drift within a single run is 4–22 % in both arms.

Numerics and session health are clean: all 11 regimes `complete: true`, `paired: true`,
`n_discarded: 0`, `machine_suspect: false`, `tick_limited: false`, `timer_tick_match_frac: 1.0`,
`frac_fused_faster` ≥ 0.9833, `rel_err` 0.0 on every decode regime and bit-identical to the
campaign at the large ones (`bs1024` 5.03e-03, `t2048` 2.59e-03, `t8192` 2.38e-03, tol 0.02);
preflight harness floor **39.46 µs** (bar `FLOOR_US_MAX=50.0`, `glm52_h200/config.py:272`),
floor/launch 3.84× (bar 8.0), tick 0.032 µs on 200/200 exact multiples. The campaign tree was
untouched: canary `fingerprint_ok: true` over 28 files, and every `results/h200/*.json` predates
the run.

**Six families remain unmeasured (`f10`, `f01`, `f04f05`, `f11`, `f06`, `f08f09`) and no
verdict on the study question is possible.** §1's **Verdict** line stays empty. Nothing here
says anything about whether the Hopper levers matter; it says the harness and the control arm
work, and it fixes the reason we could not find out.

#### Three smaller findings from the run record, none of them verifier defects

1. **A false `--regimes` warning.** `driver.log 10:57:38` prints
   `!! bench_f03_resadd_rmsnorm.py advertises no --regimes flag; the request was passed only
   via GLM52_REGIMES and may be ignored by this bench`, and `control_arm_summary.json` records
   `unhonoured_flags: ["--regimes"]`. **This is false**: the bench does accept `--regimes`
   (registered by `B.add_std_args`, consumed at `bench_f03_resadd_rmsnorm.py:393`
   `B.resolve_regimes(C, args.regimes)`), the driver passed it on the command line, and all 11
   regimes ran. The static flag scan simply cannot see a flag added by a helper. Cosmetic — but
   it plants a false doubt about regime coverage in the exact artefact an auditor reads, and it
   will reappear for the other six families. The message should say *not detectable by static
   flag scan*.
2. **`log/run_control_h200/preflight_h200.campaign.json` is misnamed.** The driver re-probed
   and replaced `glm52_h200/preflight_h200.json` in place (`10:57:29`) because the cached probe
   described GPU 7. The copy it preserved is **that stale GPU-7 probe** (uuid `3aa19cef`,
   `argv --gpu 7`, ts 2026-08-10 11:33:19), **not** the campaign f03's own 2026-08-07 10:30:37
   probe. Nothing is lost — the campaign's probe survives in substance in
   `results/h200/summary.json.preflight` — but anyone reading that filename for the campaign's
   card gets the wrong UUID.
3. **A doc-count error inside the landed V9 comment.** It reads "identical in 126 of 126 stages
   across all seven families". 126 is the `coarse`-only count; adding `tune` (105) gives
   **231 of 231**, and 238 of 238 including `extra`. The true claim is *stronger* than the one
   written. The `(see OFFERED_GRID_STAGES)` reference at ~`:1420` also reads backwards after
   the rename, since it sits in a sentence describing which stages are winner-*derived*.
   Both are for the owner of `run_control_h200.py`, not for this log.

#### Two things worth building before the next round trip

- **A regression fixture, so V9 cannot re-acquire this defect.** The adversarial input already
  exists in the repo: `results/h200/f03_resadd_rmsnorm.json` vs
  `results/h200/_bs_extra_rerun/f03_resadd_rmsnorm.json` (and the `f10` pair). A test asserting
  *V9 PASSES on a Hopper-vs-Hopper pair, and FAILS when a coarse `n_tried` is perturbed* pins
  both halves. Read the result files; never write to them.
- **Close the 28-stage coverage hole.** The campaign carries no grids block for
  `decode_bs2/4/8/16`, so 28 arm stages — 12 of them `coarse` — are compared against nothing.
  Since the campaign's `f03` coarse values are a single distinct value (164), the arm-side
  coarse stages in those regimes can be asserted against it directly; where a family's campaign
  coarse values are not unique, fall back to INFO rather than guessing. And a **collapse
  guard** is the assertion `refine`/`joint` should carry instead of equality: FAIL only when a
  stage that was non-zero in the campaign is 0 or missing in the arm. `common.py:1072` swallows
  a raising refine generator and continues with `rg = []`, which is a genuine engagement
  failure the old V9 could not distinguish from noise. On the returned data there is no
  collapse: min refine 28, max 82, min joint 6, and all 33 arm-side coarse stages equal 164.

#### Caveats that survive and belong in the eventual writeup

1. `f03`'s correct verdict is **`NOTHING-TO-DISABLE`, not `ENGAGED`** — it constrains the
   harness, not the hypothesis.
2. The comparison is **cross-session by design** (§4): control 2026-08-11 against a campaign of
   2026-08-07. Mitigated — same card by UUID, DRAM bandwidth reproducing to 0.03 % — never
   eliminated.
3. `GLM52_H200_CLASSIC=1` forces `wgmma` off too (§3), one lever beyond the three the study
   names.
4. `f03` sits **above** its modelled 1.25× DRAM ceiling in every regime in **both** arms. That
   is launch-overhead dominance (`ceiling_with_launch` is 1.999 at `decode_bs1`) and a
   pre-existing campaign property, not a control-arm artefact — but it must not be quietly
   carried into a bandwidth-framed claim.

### 2026-08-11 (later) — four more defects found by re-verifying the repair, three of them in the recovery path itself

Repairing V9 was not the end of it. Verifying the repair surfaced a **critical** defect in the
path the operator would take next, plus three majors. All are fixed; each is recorded because
each would have cost another round trip or produced a false record.

**1. (critical) A plain relaunch would have measured NOTHING and aborted at `f10`.**
`resume_verify_only = sentinel.exists() and not args.force_rerun` was **arm-wide**, and it
gated the launch decision for *every* family. With `ARM_NOT_VERIFIED` on disk from the V9
abort, all seven families would have taken the "not launching" branch — including the six that
have no staged JSON at all. Those six would then fail V0 *"missing or unreadable"*, the arm
would fail, and the run would exit having burned a round trip to measure nothing. The recovery
path was a trap. It is now **per family**: a family is re-verified instead of re-measured only
when its own staged payload already covers the whole regime scope. Demonstrated against a
mirror of the returned tree:

```
  [classic/f03]    staged payload covers the 11-regime scope     -> re-verify only
  [classic/f10]    staged payload does NOT cover the 11-regime scope -> MEASURE
  [classic/f01]    ... -> MEASURE      [classic/f04f05] ... -> MEASURE
  [classic/f11]    ... -> MEASURE      [classic/f06]    ... -> MEASURE
  [classic/f08f09] ... -> MEASURE
```

**2. (major) A 1-of-7 arm could clear the arm-wide "unverified" record.** The sentinel removal
was gated on `not arm_failed`, which means "nothing *in scope* failed" — not "the arm is
verified". Under `--families f03` the other six never enter the loop, so `arm_failed` stays
False and the run would unlink the sentinel, set `verified: True` and `sentinel: None`. That is
one of only four independent records saying an arm is unpublishable, and the only one the
report generator can still see if a tarball loses the rest. Removal now additionally requires a
**complete** family scope; a partial run logs `ARM_NOT_VERIFIED KEPT` and records
`partial_scope`. Confirmed: `--verify-only` against the returned tree re-verifies `f03` to
`NOTHING-TO-DISABLE, n_fail=0` and **keeps** the sentinel, correctly, because six of seven
families are unmeasured.

**3. (major) `launched_any` was set even when a stage launched no child**, which silently
retired the staging-redirect pre-commitment check for the whole resumed run — the check that
fires after the first attempt of the first family and stops the run before the campaign's
`_ckpt` cache can be read. Now `launched_any = launched_any or bool(stage.get("attempts"))`.

**4. (major) V9's own prose repeated the mistake that caused this incident.** It printed *"the
OFFERED grid changed, which no arm switch can do"* for **any** family. That is true only of
`f03`/`f10`, which are offered no axis and whose `widen()` is a no-op. For the five overlay
families a disabling arm is *supposed* to shrink the offered grid, so the sentence was
authoritative and false in exactly the way the original V9 detail string was. The wording now
branches three ways, verified by mutation against real files:

| input | verdict | wording |
|---|---|---|
| `f03` strict, offered grid halved | **FAIL** | "…which no arm switch can do **for a family that is offered no Hopper axis**" |
| `f01`, uniform shrink | INFO | "the offered grid **SHRANK, which is what this arm predicts**" |
| `f01`, mixed shrink+grow | INFO | "changed, **not uniformly a shrink — inspect before trusting this arm**" |

**The verifier is still a verifier.** After all of the above, the real campaign files (Hopper
levers ON) judged as a `classic` arm are refused for every family — `f03`/`f10` `n_fail=8`
(V1/V2/V4); `f01`/`f06`/`f08f09` `n_fail=18`; `f04f05` `n_fail=15`; `f11` `n_fail=31`
(adding V10). The `f03`/`f10` vacuity guard still reports `NOTHING-TO-DISABLE` rather than
`ENGAGED`, on forged de-Hopperised fixtures and on the real returned data alike.

**One process note, and it is embarrassing enough to record.** Local `--list` / `--dry-run`
runs during this repair appended 234 lines to the operator's returned
`log/run_control_h200/driver.log` — the incident record itself — because they default to that
log directory. It was restored from git (`git checkout --`) and subsequent local runs were
redirected with `--log-dir`. `log/run_control_h200/.gitignore` already warns about exactly this
("a local `--dry-run` / `--list` on a dev box writes the same filename here… Delete a local one
before committing"); the warning was right and I still walked into it. Everything under
`results/h200/_control_arm/` is byte-identical to what the operator sent.

### 2026-08-11 (evening) — the arm ran three families, then the process was killed and a neighbour took the card

**What the operator got.** The V9 repair worked: `f03` re-verified in place without
re-measuring, `f10` and `f01` measured, and the per-family resume did exactly what it was
rewritten to do. Then the driver died 10 minutes into `f04f05` having written **nothing** — no
traceback, no exit line, no summary update. The last line of `driver.log` is a routine
heartbeat at 14:57:30. `f04f05` left no result and no checkpoints.

**Three separate faults, only one of which is the crash.**

**1. A co-tenant took the card mid-arm, and the driver kept going.** At 14:47:30, between
`f01` and `f04f05`:

```
!! a co-tenant appeared on GPU 0 after classic/f01: new compute process(es):
!!   pid 2916934 VLLM::Worker_TP0_EP0 (121.6 GB); memory in use grew +121.6 GB
!! Nothing is stopped -- an aborted campaign loses more than a flagged one
```

That policy is inherited from `run_h200.py` and it is **right for a campaign and wrong here**.
The campaign *is* the baseline; this arm is a **diff against a baseline measured on an idle
card**, so a neighbour does not add noise, it removes the comparison. The hw snapshots bracket
it precisely — `used=0 MiB` at every boundary through 14:39:59, `used=124498 MiB` at 14:47:29
— so the tenant arrived **during `f01`**, and `f01` took **450 s against the campaign's 233 s,
1.93×**, which the driver itself flagged as a signal that something was wrong. `f01` is
therefore contaminated and is not publishable. `f03` (10:58, idle) and `f10` (13:37–13:39,
idle) are clean.

Fixed: co-tenancy is now **fatal by default** in this driver (`--on-tenant stop`, new exit
code **7**), the contaminated family is marked in its stage record and the summary, and
`make_control_report_h200.py` drops any family flagged `TENANT-CONTAMINATED` from the
comparison rather than averaging it into a drift band. `--on-tenant flag` restores the old
campaign behaviour.

**2. The killed run recorded itself as verified.** `control_arm_summary.json` came back with
`verified: true, sentinel: null` on an arm with **3 of 7** families measured, because the arm
record was initialised optimistically (`"verified": True`) and only flipped to False on
failure — so any path that never reaches the end of the arm leaves the optimistic value in the
last incremental save. The only thing standing between that and a 3-of-7 arm reading as
publishable was the `ARM_NOT_VERIFIED` file left over from the *earlier* V9 abort, which is
luck, not a safety net.

Fixed: `verified` now defaults to **False** with a `verified_reason`, and is set True only on a
**complete** arm (every family adjudicated, none failed, no tenant). Verified against the real
returned tree: `verified: false`, reason *"at least one family failed engagement
verification"*. Also fixed: `build_summary(live)` took a `live` parameter and never emitted it
— which is why the returned summary read `live: null` instead of telling us it had never
closed.

**3. The termination left no diagnosis.** We still cannot say *which* signal it was, and that
is the part worth fixing rather than guessing. SIGTERM/SIGHUP/SIGINT are catchable, so they are
now trapped, named in `driver.log` and recorded as `terminated_by` in the summary; SIGKILL and
the host OOM killer cannot be trapped by anyone, which is what the incremental save and the
pessimistic `verified` default are for. Reproduced end to end by SIGTERMing the driver: it now
exits recording `live=false, terminated_by=SIGTERM, verified=false`. A first attempt still lost
the record when the signal landed in the **closing sequence**, which sat outside every handler
— that path is now interrupt-safe too.

The module docstring now leads with `setsid nohup python3 run_control_h200.py &`. A dropped SSH
session (SIGHUP) is the cheapest explanation for a silent death and the only one the operator
can eliminate for free.

**An anomaly in the noise floor that is NOT explained by any of the above, and must not be
waved through.** `f10` at `decode_bs16` moved **2.1607 → 3.7394, +73.1 %**, while its ten
sibling regimes sit within ±6 %:

| regime | campaign | control | delta |
|---|---|---|---|
| decode_bs8 | 2.1650 | 2.1286 | −1.7 % |
| **decode_bs16** | **2.1607** | **3.7394** | **+73.1 %** |
| decode_bs32 | 2.1338 | 2.1701 | +1.7 % |

`f10` is offered no Hopper axis, so the switch cannot have done this, and `f10` was measured on
an idle card, so co-tenancy cannot either. It matters because `f03`+`f10` **are** the drift
band: taken at face value this single cell widens the band to ±73 %, which is larger than any
effect the study is looking for and would make every other family unresolvable. It needs a
cause before the band is computed. Untouched pending that — recording it, not fixing it.

**Status.** `f03` and `f10` clean and verified (`NOTHING-TO-DISABLE`). `f01` measured but
contaminated. `f04f05`, `f11`, `f06`, `f08f09` never ran. No verdict on the study's question is
possible, and none is implied by anything above.

### 2026-08-12 — the arm completed, and the comparison it was built to publish is withheld

All seven families came back (`8cdef5d` "h200 control done"). **The arm engaged, and the
measurement of five of the seven families is sound.** The two families whose tuner grid the
switch actually changed — `f01` and `f04f05` — were each measured while a neighbour was taking
the card, were never re-measured, and are **excluded by name**. What remains cannot carry the
study's question. That is the entry.

Everything below is read out of `results/h200/_control_arm/`, `log/run_control_h200/driver.log`
and the committed campaign. Nothing under either was written, moved or edited
(`git status --porcelain results/h200/ log/run_control_h200/` is empty); all fixtures were built
under a scratchpad.

#### 1. What came back, and how many runs it took

`driver.log` carries **seven appended banners**, not the five that produced data:

| # | started | what it did |
|---|---|---|
| 1 | `10:57:21` | **aborted** on the `--gpu 7` / campaign-card contradiction; no work |
| 2 | `10:57:29` | measured `f03`; V9 failed (the defect repaired in the earlier entry); sentinel written; exit 5 |
| 3 | `13:37:09` | measured `f10`, launched `f01`, killed mid-`f01` ~13:41, no footer |
| 4 | `14:39:42` | **interrupted 1 s after launching `f01`** (`14:39:45 !! interrupted`); no `[hw before]` taken |
| 5 | `14:39:56` | measured `f01`, launched `f04f05`, killed mid-`f04f05` ~14:57, no footer |
| 6 | `15:43:32` | measured `f04f05`, launched `f11`, interrupted 15:51:52; exit 1 |
| 7 | `19:32:10` | measured `f11`, `f06`, `f08f09`; sentinel removed; exit 0 |

Engagement passed everywhere, `n_fail 0`:

| family | axes the campaign offered | verdict | V9 offered grid | V9d winner-derived | Hopper-ON winner cfgs, campaign → control |
|---|---|---|---|---|---|
| `f03` | **none** | `NOTHING-TO-DISABLE` | 0 of 21 differ | 19 of 28 | 0 / 7601 → 0 / 7397 |
| `f10` | **none** | `NOTHING-TO-DISABLE` | 0 of 21 | 19 of 28 | 0 / 8659 → 0 / 8667 |
| `f01` | tma, ws, clusters | `ENGAGED` | 14 of 14 | 10 of 35 | **4267** / 7022 → **0** / 4794 |
| `f04f05` | ws, clusters | `ENGAGED` | 35 of 56 | 0 of 28 | **14839** / 25742 → **0** / 18845 |
| `f11` | ws | `ENGAGED` | 2 of 35 | 14 of 42 | **4835** / 12945 → **0** / 12256 |
| `f06` | tma, ws, clusters | `ENGAGED` | 0 of 21 | 12 of 21 | **3989** / 8198 → **0** / 8224 |
| `f08f09` | tma, ws, clusters | `ENGAGED` | 0 of 63 | 12 of 58 | **4541** / 11894 → **0** / 11887 |

(Winner-config counts are an independent recount over every dict in each file carrying a
`BLOCK_*` or `num_warps` key, scoring a config hot on `USE_TMA`/`TMA_A`/`TMA_B`/`warp_specialize`
being `True`, `num_ctas > 1`, or a non-`none` `TMA_MODE` — exact keys, never substrings, because
`"ROWS"` contains `WS` and occurs 7395 times in `f03`'s control file alone. All seven control
files also record `_meta.harness_info.features.disabled ==
["tma","ws","warp_specialize","clusters"]` against `[]` in all seven campaign files, and every
returned child log opens `tma=False ws=False clusters=False wgmma=False | src env` against the
campaign's `src preflight`. `src env` vs `src preflight` is the discriminator, and it is the
only engagement evidence that reaches `f03` and `f10` at all.)

**`verified: true` in the summary certifies engagement, not provenance.** It says the switch
reached every child. It says nothing about who else was on the card, and §3 is why that
distinction is now load-bearing.

#### 2. Provenance, family by family

Every family ran on the campaign's own card (`GPU-b2318e71-…-3cc3e6ea7db0`, re-asserted in each
staged `_meta` and in check V6), all `exit=0 status=ok`, all 11 regimes, all on 2026-08-11:

| family | measuring run | window | wall | card used, before → after | staged `recorded_at` | child's own `nvidia-smi` at record | state |
|---|---|---|---|---|---|---|---|
| `f03` | `10:57:29` | 10:57:38–10:58:40 | 62 s | 0 → 0 MiB | 10:58:38 | 1269 MiB, util 0 % | **CLEAN** |
| `f10` | `13:37:09` | 13:37:10–13:39:13 | 123 s | 0 → 0 MiB | 13:39:10 | 983 MiB, util 0 % | **CLEAN** |
| `f01` | `14:39:56` | 14:39:58–14:47:29 | 450 s | **0 → 124498 MiB** | 14:47:27 | **126989 MiB used / 16168 free, util 4 %, throttle 0x4** | **CONTAMINATED** |
| `f04f05` | `15:43:32` | 15:43:34–15:51:16 | 462 s | **0 → 63259 MiB** | 15:51:14 | **66288 MiB used / 76869 free** | **CONTAMINATED** |
| `f11` | `19:32:10` | 19:32:13–19:58:24 | 1570 s | 4 → 0 MiB | 19:58:21 | 25761 MiB, util 0 % | **CLEAN** |
| `f06` | `19:32:10` | 19:58:26–20:22:24 | 1436 s | 0 → 0 MiB | 20:22:22 | 13425 MiB, util 76 % (its own work) | **CLEAN** |
| `f08f09` | `19:32:10` | 20:22:25–20:46:18 | 1432 s | 0 → 0 MiB | 20:46:16 | 9993 MiB, util 0 % | **CLEAN** |

File-vs-log agreement is exact: every `_meta.recorded_at` falls inside its family's driver
window, every `_meta.results_dir` is the staging path, every `_meta.hwinfo.gpu.uuid` is the
campaign card. **The children's own end-of-run snapshots corroborate the memory story without
using `driver.log` at all** — the two contaminated families record a card two-thirds and
half full; the five clean ones record only their own footprint.

One provenance defect inside a family that is excluded anyway, recorded so nobody resurrects
the cell: **`f01`'s published row set is a two-session splice.** Its `decode_bs1` checkpoint is
`saved_at 2026-08-11 13:40:31` — from the *abandoned* 13:37 run — and was reused rather than
re-measured; the other ten span 14:40:26–14:47:27. It is visible only in
`_ckpt/f01_oproj_resadd/*/saved_at`, in nothing else.

**The wall times the driver alarmed on were measured against the wrong denominator.**
`results/h200/summary.json families[*].wall_s` was frozen on 2026-08-07 13:33 and covers
**seven** regimes; `decode_bs2/4/8/16` were appended later by the bs-extra run, whose per-family
benches each measured all **eleven** on the same idle card. Comparing an 11-regime control
against a 7-regime campaign manufactures a slowdown:

| family | campaign, 7 reg | bs-extra, 11 reg | campaign true 11-reg cost | control, 11 reg | vs 7-reg (what the driver did) | vs true 11-reg |
|---|---|---|---|---|---|---|
| `f03` | 57 s | 84 s | 141 s | 62 s | 1.09x "slower" | **0.44x** |
| `f10` | 108 s | 162 s | 270 s | 123 s | 1.14x "slower" | **0.46x** |
| `f01` | 233 s | 516 s | 749 s | 450 s | 1.93x "slower" (**warned**) | **0.60x** |
| `f04f05` | 1230 s | 1680 s | 2910 s | 462 s | 0.38x | **0.16x** |
| `f11` | 1169 s | 1362 s | 2531 s | 1570 s | 1.34x "slower" (unwarned) | **0.62x** |
| `f06` | 891 s | 1068 s | 1959 s | 1436 s | 1.61x "slower" (**warned**) | **0.73x** |
| `f08f09` | 1121 s | 1302 s | 2423 s | 1432 s | 1.28x "slower" (unwarned) | **0.59x** |

On a like-for-like basis **every family in the control arm was faster than the campaign's true
cost for the same eleven cells**, and both of the driver's wall-time alarms are artefacts of
the comparator. `f01`'s 1.93x is therefore *not* corroborating evidence of its contamination and
must not be quoted as such — the contamination evidence is the memory, not the clock.

#### 3. The contamination, and why it disqualifies rather than degrades

Two co-tenants arrived inside measurement windows, named in `driver.log` and nowhere else:

```
14:47:30 !! a co-tenant appeared on GPU 0 after classic/f01: new compute process(es):
14:47:30 !!   pid 2916934 VLLM::Worker_TP0_EP0 (121.6 GB); memory in use grew +121.6 GB
15:51:18 !! a co-tenant appeared on GPU 0 after classic/f04f05: new compute process(es):
15:51:18 !!   pid 3254840 /usr/bin/python (61.8 GB); memory in use grew +61.8 GB
```

A third event at `15:51:59` names `pid 3254840` grown to 68.5 GB plus
`pid 3264022 sglang::scheduler_DP0_TP0_EP0` (32.1 GB), +87.1 GB. It lands **after** `f04f05` was
written (15:51:14) and during the 34-second `f11` attempt that was killed and left no payload,
so it contaminates nothing that survives — the next run still printed `[classic/f11] staged
payload does NOT cover the 11-regime scope -> MEASURE`. The abandoned `f04f05` attempt of the
14:39:56 run likewise started at 14:47:30 on an already-124498 MiB card and left no payload.

**Neither contaminated family was ever re-measured.** Every later run found their staged payload
already covering the 11-regime scope and took the re-verify-only path: `f01` decided MEASURE at
13:39:14, 14:39:44 and 14:39:58 (the completing one), then "covers → re-verify only" at 15:43:33
and 19:32:12; `f04f05` MEASURE at 14:47:30 and 15:43:34, then "covers → re-verify only" at
19:32:12. That path is correct behaviour — it is what stops a resumed run re-burning good data —
but it means a family measured under a neighbour is carried forward untouched and unflagged.

**Why this is exclusion and not a caveat.** The baseline these two are diffed against was
measured on an idle card (`summary.json` `_meta.gpu_was_idle: true`, `tenant_events: []`,
`hwinfo_drift` memory start 0.0 / end 0.0, and the campaign `driver.log` shows `used 0/143771
MiB` at every family boundary). A neighbour holding 122 GB or 62 GB of a 143.8 GB card does not
add noise to a memory-bound ratio — **it removes the comparison**. And the artefacts cannot say
*when* in a 7.5-minute window the neighbour arrived, so no individual cell can be exonerated
either. `f01` and `f04f05` are named as excluded, flagged for re-measurement, and kept out of
every aggregate. They are not quietly averaged in and they are not silently dropped.

Three things about this that are our fault, not the operator's, and all three are worth naming:

1. **The driver of the day only warned.** `!! Nothing is stopped -- an aborted campaign loses
   more than a flagged one` is inherited from `run_h200.py` and is right for a campaign and
   wrong for a diff. It has since been changed: `--on-tenant` now defaults to **`stop`**
   (`run_control_h200.py:2066`, exit code 7), with `--on-tenant flag` restoring the old
   behaviour. **The operator ran the committed version, so the fix did not apply to this run.**
   The fix protects the *next* run and nothing about this one.
2. **The co-tenant record was LOST from the summary.** The warnings were written into the
   summary of the run that saw them, and each later run overwrote
   `control_arm_summary.json`. The final summary carries `gpu.tenant_events: []` and exactly two
   warnings — the `f06` wall time and the sentinel removal — with **no co-tenant string
   anywhere**. It also records `fam_stages[f03|f10|f01|f04f05].wall_s == 0.0` with
   `attempts == []`, i.e. the two contaminated families read as though they were never launched.
   **The entire evidence for the central exclusion in this study survives in one unstructured
   text file.** That is a real provenance weakness, it is being recorded as one, and the summary
   is not being edited to paper over it. The report generator must derive contamination by
   parsing `driver.log`'s `!! a co-tenant appeared` lines together with the `[hw before]` /
   `[hw after ]` used-MiB pair bracketing each family, and must read the per-family
   `_ckpt/*/saved_at` fields, which are the only place `f01`'s splice is visible.
3. **`f04f05`'s child transcript never came back.** `log/run_control_h200/.gitignore:24`
   excludes `f04f05*.log` on size grounds and `tools/split_h200_log.py` was not run, so the one
   family with the arm's worst cells has no transcript, no `src env` banner and no way for
   anyone to diagnose it. The campaign directory has `f04.log`, `f05.log` and
   `f04f05_compiler.log.gz`; the control directory has none of the three.

#### 4. Anomaly A — `f10` at `decode_bs16`, +73.1 % — RESOLVED as a harness artefact, cell excluded

The cell moved 2.1607 → 3.7394 while its ten siblings sit inside ±6 %. It is resolved, and the
answer is that **nothing got faster**: both sides of the ratio collapsed, the fused side further.

| | campaign | control |
|---|---|---|
| `fused_ms` | 0.06323 | **0.01642** (−74 %) |
| `unfused_ms` | 0.13690 | **0.06109** (−55 %) |
| `speedup` | 2.1607 | **3.7394** |
| `fused_cfg` | `ROWS 4, BLOCK_N 256, nw 2, KVEC 0` | `ROWS 1, BLOCK_N 256, nw 2, KVEC 1` |
| `bitwise_identical` | true (`rel_err` 0.0) | false (`rel_err` 2.77e-04) |
| `order_gap_frac` | 0.00248 | 0.05449 (22x) |
| `ratio_halves` | [2.006, 2.310] | [3.705, 3.777] |

The workload is provably identical — `T=16`, `bytes_fused` 1966592 in both, `torch_eager` 0.1433
vs 0.1479 ms and `torch_compile` 0.0685 vs 0.0670 ms unchanged — so this is not a shape change.
Neither the switch nor the neighbours can reach it: `f10` is offered no Hopper cfg key, and the
card read `0/143771 MiB` at both ends of the family window.

**The decisive diagnostic is the flush tail, and it is mechanical.** `common.py:498` zeroes a
256 MB buffer on-stream immediately before each timed region, and the dirty-L2 writeback drains
into the measurement. Writing `C_f = final_fused − tuned_fused_best` from the child logs, every
comparable cell in both arms pays it and this one does not:

| cell | `C_f` | `C_u` |
|---|---|---|
| 20 of 22 decode / `prefill_t2048` cells, both arms | 0.047–0.079 ms | 0.071–0.147 ms |
| `prefill_t8192`, both arms | 0.0004 ms | 0.0001 ms |
| control `decode_bs8` (the neighbour cell) | 0.0558 ms | 0.0822 ms |
| **control `decode_bs16`** | **0.0074 ms** | **0.0056 ms** |

Re-derived here from `log/run_control_h200/f10.classic.a1.log`: tuned fused best 0.0090 ms →
final 0.0164 ms; chain joint 0.0555 ms → final 0.0611 ms. Its tuning phase was entirely normal
(siblings tune to 0.0057–0.0106 ms fused, 0.0522–0.0572 ms chain), so no kernel or tile effect
is involved — only the final paired window. It is also **the only cell in either arm whose fused
time falls below its own session's measured harness floor**: 16.42 µs against 39.46 µs (0 of 165
campaign cells, 1 of 165 control cells), and it sits at 1.90x its own `ceiling_with_launch` where
no other cell in either arm exceeds 1.38x.

**The cell is excluded on a stated, uniform, pre-registerable criterion** — *a cell is invalid
if its measured additive constant `C_f` falls below 15 % of the family's cohort floor for that
arm* — which is computable per cell without reference to its speedup and which removes this cell
and no other across all seven families. It is excluded loudly, with its numbers above, so a
reader can overrule the exclusion and see exactly what that costs (§6). It goes on the
re-measurement list, from a **different cause** than `f01`/`f04f05`: harness artefact, not
co-tenancy.

One standing caveat this surfaced, which is a property of the **committed campaign** and not of
the control arm: at small `T` the flush tail *is* the measurement, so `f10`'s 2.0–2.2x figures
are ratios of flush tails rather than of DRAM traffic — every such cell exceeds its own modelled
ceiling on both arms. The one cell where the tail is negligible, `prefill_t8192`, lands at
1.196x against a modelled ceiling of 1.20x. That should be disclosed wherever those numbers are
published.

#### 5. Anomaly B — `f06` at 24 min vs 15 min — the alarm dissolves; a 1.34x residual is UNDETERMINED

The driver's warning was: *"The classic arm's widened grid collapses (`h200_cfg_overlays()`
returns `[]`), so it should be FASTER to tune; a slower classic arm is a signal something is
wrong."* Three independent things are wrong with it and one thing survives.

1. **Wrong denominator** (§2): 1436 s over 11 regimes against 891 s over 7. Against the
   campaign's true 11-regime cost of 1959 s the control ran at **0.73x**.
2. **Wrong reference point.** 14.9 min is the *fastest* of three successful same-scope `f06`
   attempts in the campaign (`log/run_h200/driver.log`: 28.1 min, 17.3 min, 14.9 min). 23.9 min
   is inside the campaign's own spread.
3. **The premise is false for this family.** The *legal* grid did collapse exactly 5.0x — the
   control log reads `289 / 396 / 508 / 566 legal` where the campaign reads
   `1445 / 1980 / 2540 / 2830`, base plus four overlays — but **both arms then sample to
   `budget 200`**, so the *offered* grid is unchanged and the tuner does the same work. The
   driver's own V9 check already said so (`0 of 21 offered-grid stage(s) differ`, `n_tried` 16070
   control vs 16036 bs-extra vs 16018 campaign, ratio 1.00) and nobody wired it into the
   heuristic. The same holds for `f08f09` (`0 of 63`, 22408 vs 22426). The grid measurably shrank
   only for `f01` (0.59x), `f04f05` (0.72x) and marginally `f11` (0.94x).

**What survives.** Against the closest like-for-like comparator — the 2026-08-07 bs-extra run,
same card, idle, Hopper ON, all 11 regimes, 1068 s — the control took 1436 s, **1.34x on
identical tuner work**. `f11` (1.15x) and `f08f09` (1.10x) show the same direction. A candidate
mechanism exists and is not confirmed: compile failures stopped being free (`f06`'s `n_failed`
collapses 573 → 38 over the shared regimes at flat `n_tried`, so ~12 % more configs per regime
survive to a full L2-flushed timing run). **The residual is recorded as UNDETERMINED.** It
carries no contamination marker — the card was 0 → 0 MiB, the child's own footprint was
13425 MiB — and it cannot reach the published ratios in any case: the A/B measurement is 7.5 s
of a 1436 s wall (0.5 %), row health matches the campaign (`paired: true` on all 11,
`n_pairs` identical per regime, `ratio_halves` tight, no retries or OOM). The extra time is
tuning, and tuning is not what is published.

#### 6. The result, and what an unpaired design cannot say

The band is the `f03` + `f10` relative delta of paired speedups, `d = classic / campaign − 1`,
computed cell-for-cell over the 165-cell join of the two summaries (163 pairs resolve on both
sides; the two `f03`/`f10` families contribute 22 of them):

| band | n | min | median | max |
|---|---|---|---|---|
| `f03` + `f10`, **as measured** | 22 | −6.07 % | +0.61 % | **+73.06 %** |
| `f03` + `f10`, **less `f10/decode_bs16`** | 21 | **−6.07 %** | +0.48 % | **+4.70 %** |

**Both are stated, and the difference is fully consequential.** Against the 73.1 % band *zero*
cells in *any* family fall outside and the study resolves nothing whatsoever. Against the
6.07 %/4.70 % band it resolves exceedances in three of the five publishable families (and in
both of the excluded ones). The defensible band is the second
one, on the §4 criterion, and the first is printed beside it so the choice is visible rather
than buried.

**And then the exceedances have to be priced against the band's own false-positive rate, which
is the step that decides this run.** A min/max band over *n* reference cells is exceeded by a
fresh exchangeable draw exactly when that draw is the new minimum or the new maximum, i.e. with
probability `2/(n+1)`. At n = 21 that is **9.1 % per cell, by construction and before any
effect exists.** Over the 86 judged cells it predicts **7.8** exceedances; the report observes
**12**. One-sided binomial `P(X ≥ 12 | n = 86, p = 2/22) = 0.089` — not significant at any
conventional threshold. Per group of eleven cells the chance of at least one exceedance is
`1 − (1 − 0.0909)¹¹ = 65 %`, which is precisely why seven of the eight judged groups have one:
the group-level count is an artefact of the band's construction, not a result.

Two further tests, both computable from what is already published, point the same way:

* **Leave-one-family-out.** Rebuilding the band from the other families in turn — over the four
  whose offered grid provably did not change (V9 0-of-21, 0-of-21, 0-of-21, 0-of-63) — gives
  **6 exceedances over 107 cells against 2.9 expected**. The signal does not survive being
  asked to hold under a band it did not help construct.
* **Within-cell dispersion.** Comparing each cell's own `speedup_p10_p90` between the two arms,
  **8 of the 12 exceedances overlap** — the two arms' spreads are not separated. Of the four
  that do not, all four are `prefill` cells, and prefill is exactly where §2 says the band is
  weakest: it has 4 samples and falls back to a global band dominated by decode's
  launch-latency variance.

**So the honest headline for this run is that the design could not answer the question.** Not
"the levers did nothing" — that is a claim this design cannot make in either direction — and
not "seven of eight groups show an effect", which is what the raw exceedance count looks like
until it is priced. The published README now states the null expectation in its Verdict field
rather than in a footnote.

**The bitterest part is structural, and it is worth stating plainly for whoever runs this
next.** The two families excluded for co-tenancy, `f01` and `f04f05`, are **exactly the two
whose offered tuner grid demonstrably collapsed under the switch** — the two where the control
arm provably changed what the tuner was allowed to try, and therefore the two most likely to
carry a real effect. The run lost its best evidence to a neighbour and kept its weakest. That
is not a reason to read anything into what remains; it is the reason a re-measurement is worth
the GPU time.

Per family, against **[−6.07 %, +4.70 %]**:

| family | n | min | median | max | outside the band | reading |
|---|---|---|---|---|---|---|
| `f03` | 11 | −5.06 % | +0.04 % | +4.41 % | 0 of 11 | **band constituent** — no verdict |
| `f10` | 10 (+1 excl.) | −6.07 % | +1.07 % | +4.70 % | 0 of 10 | **band constituent** — no verdict; `decode_bs16` excluded (§4) |
| `f06` | 11 | −12.33 % | +0.09 % | +9.64 % | **2** (`prefill_t8192` −12.33 %, `prefill_t2048` +9.64 %) | 2 cells exceed the band; 9 inside, so nothing is shown for those |
| `f08f09` | 42 | −16.69 % | −0.30 % | +11.64 % | **4** (`f9_token_major/decode_bs1` −16.69 %, `f8_token_major/decode_bs1` −12.46 %, `f9_atomic/decode_bs1` +11.64 %, `f8_token_major/prefill_t8192` +4.96 %) | 4 cells exceed the band; 38 inside, so nothing is shown for those |
| `f11` | 33 | −28.17 % | −0.25 % | +15.82 % | **6** (`combined/decode_bs1` −28.17 %†, `f11b_router/decode_bs1` +15.82 %, `f11b_router/decode_bs2` +14.71 %, `combined/prefill_t2048` +8.99 %, `f11a_w13/prefill_t2048` +8.60 %, `f11b_router/prefill_t8192` +6.91 %) | 6 cells exceed the band; 27 inside, so nothing is shown for those |
| ~~`f01`~~ | 11 | −6.20 % | +0.24 % | +1.56 % | 1 | **EXCLUDED — tenant-contaminated** |
| ~~`f04f05`~~ | 44 | −64.60 % | −0.33 % | +10.62 % | 15 | **EXCLUDED — tenant-contaminated** |

† `f11 combined/decode_bs1` is a coverage artefact, not drift: the campaign's `moe` sub-arm there
records `unmeasurable: "RuntimeError: PassManager::run failed"`. Annotate or drop it; do not read
it as a delta. (Two `f08f09` cells are unresolved on one side — `f9_atomic/decode_bs4` in the
campaign, `f9_atomic/decode_bs8` in the control — hence n=42 not 44. Counting exceedances with a
symmetric `|d| > 6.07 %` rule instead of the asymmetric band gives 14 for `f04f05` and 3 for
`f08f09`; the band is asymmetric because the data are, and the rule is stated before the counts.)

**Now the honest part, and it is most of the result.**

- **Every verdict above is "the delta exceeds the drift band" or "the delta is inside the drift
  band, so nothing is shown".** Neither sentence attributes anything to TMA, warp specialization,
  clusters or wgmma, in either direction. `GLM52_H200_CLASSIC=1` also forces `wgmma` off (§3),
  so even an exceedance would not be attributable to the three levers the question names.
- **The band is not sound as a bound, and saying so is not modesty.** V9d reports that for `f03`
  and `f10` — the two families where the two arms are *identical by construction* — the
  autotuner picked a **different winner in 19 of 28 stages**. A 68 % cross-session winner-change
  rate in the families that are supposed to define the noise floor is the same mechanism, at
  smaller amplitude, that produces `f04f05/F5_topk/decode_bs8` at −64.6 %, `f11/combined/
  decode_bs1` at −28.2 % and `f08f09/f9_token_major/decode_bs1` at −16.7 % — in every case
  driven by the **unfused baseline** being re-tuned to a different winner (its own cross-session
  change reaches −71.9 %, −33.0 % and −29.9 % respectively). Two short elementwise families
  cannot bound the session-to-session tuner variance of the GEMM-heavy families. The exceedances
  in `f06`, `f08f09` and `f11` are therefore reported as exceedances of *this* band and not as
  evidence of anything.
- **The design is unpaired and that is not repairable after the fact.** Control 2026-08-11
  against a campaign of 2026-08-07 (itself a two-session composite: `decode_bs2/4/8/16` come
  byte-identically from the 2026-08-07 16:22–18:05 bs-extra run, so four of eleven regimes carry
  a *different* cross-session gap than the other seven — and `f10`'s outlier sits in that half).
  A real, measured session offset exists and applies to every family: the harness floor is
  **36.914 µs** in the campaign and **39.456 µs** in the control, **+6.9 %**, recorded identically
  in all seven staged files. The timer tick is identical (0.032 µs, `match_frac` 1.0, trusted in
  both). `f03`/`f10` absorb the floor difference into the band, which is exactly what they are
  for, but it is also why a sub-40 µs control measurement deserved the scrutiny §4 gave it.
- **What is left cannot carry the question.** Of the five publishable families, `f03`, `f10`,
  `f06` and `f08f09` had their offered grid *unchanged* by the switch (V9 `0 of 21`, `0 of 21`,
  `0 of 21`, `0 of 63` stages differing; total `n_tried` ratios 0.97 / 1.00 / 1.00 / 1.00), so
  their deltas are further **drift
  measurements**, not treatment contrasts. `f11` is the only clean family whose offered grid
  moved at all, and it moved marginally (`2 of 35`, `n_tried` 0.94x). The two families with a
  real grid collapse — `f01` (0.59x) and `f04f05` (0.72x) — are exactly the two that are
  contaminated. **A one-family contrast against a band built from feature-blind families does
  not answer "do the sm_90 levers explain anything".** Re-measuring `f01` and `f04f05` is not
  cleanup; it is the experiment.

#### 7. What remains

1. **Re-measure `f01` and `f04f05` on a clear card** — the one blocking item. On the H200 box,
   preserving the contaminated payload first, because it is the only record of what a
   contaminated cell looks like and it must not be overwritten:

   ```
   # 0. preserve the returned tree OUTSIDE results/ before touching anything
   tar czf ~/control_arm_20260811_contaminated.tgz results/h200/_control_arm log/run_control_h200

   # 1. move ONLY the two contaminated families out of the staging path, so the driver
   #    decides MEASURE instead of "covers -> re-verify only". Do NOT pass --force-rerun:
   #    it sets GLM52_H200_FORCE=1 and restarts the whole arm from zero, discarding the
   #    five clean families as well.
   mkdir -p ~/control_arm_20260811_contaminated/staged
   mv results/h200/_control_arm/classic/f01_oproj_resadd.json \
      results/h200/_control_arm/classic/f04f05_norm_router.json \
      ~/control_arm_20260811_contaminated/staged/
   mv results/h200/_control_arm/classic/_ckpt/f01_oproj_resadd \
      results/h200/_control_arm/classic/_ckpt/f04f05_norm_router \
      ~/control_arm_20260811_contaminated/staged/

   # 2. re-measure. --on-tenant stop is now the DEFAULT and is named here on purpose:
   #    a neighbour must end the arm, not annotate it. Omit --gpu; the campaign card is
   #    pinned by UUID. setsid nohup so a dropped SSH session cannot SIGHUP it.
   setsid nohup python3 run_control_h200.py --families f01,f04f05 --on-tenant stop &

   # 3. split f04f05's transcript so it survives .gitignore, then send back
   #    results/h200/_control_arm/, log/run_control_h200/ and glm52_h200/preflight_h200.json
   python3 tools/split_h200_log.py log/run_control_h200/f04f05.classic.a1.log
   ```

   Expected cost is under 20 min for the pair (control walls were 450 s and 462 s). **Measure
   one clean family in the same session as an in-session anchor** if the card is free — the
   design is unpaired, and a fresh session otherwise adds a third unquantified gap.
2. **Re-measure `f10/decode_bs16`**, from the separate cause in §4. Three items are pending from
   two distinct causes; do not collapse them into one "excluded" bucket.
3. **Fix the report generator** (owned by another agent, not touched here): derive contamination
   from `driver.log` rather than the summary; read `_ckpt/*/saved_at`; never diff a control wall
   against `summary.json families[*].wall_s`; read `f11`'s speedups from
   `rows[i].{f11a_w13,f11b_router,combined}.speedup`, and build no check on `hopper_caps` for
   `f11` — the control file has no caps report at all and the driver's own V10 says so.
4. **Do not use `f03` + `f10` alone as the band on the re-run.** Add `f06` and `f08f09` as band
   constituents — their offered grids and `n_tried` are unchanged by the switch — and take
   replicate runs of all four in a single session, so the band measures within-design variance
   rather than one draw of cross-session tuner variance.
5. **Publish the session's own `harness_floor_us` next to the campaign's** (39.456 vs
   36.914 µs, +6.9 %) on the face of the report, not buried in `fairness.timing`.

§1's **Status** and **Verdict** lines are updated to read *withheld* rather than *unfilled* —
a different and stronger statement: the arm ran, it engaged, and what it returned does not
answer the question it was built to answer. The `## 0c.` preamble in
`report_glm52_h200/README.md` still carries the unfilled verdict line and needs the same
wording; that file and `glm52/make_control_report_h200.py` are owned by the concurrent report
pass and were deliberately not touched from here.
