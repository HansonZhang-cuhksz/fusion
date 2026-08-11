# Hopper control arm -- `classic` vs the committed H200 campaign

**Generated** from the control run recorded 2026-08-11 20:46:20 · **Arm** classic · **Verdict** EXCLUDED, NOT JUDGED, AND MARKED FOR RE-MEASUREMENT: 5 of 15 (fusion, variant) group(s) across 2 famil(y/ies) -- f01, f04f05 -- #1 o_proj + ResAdd (triton): TENANT-CONTAMINATED; #4 ResAdd + RMSNorm + Router (F4): TENANT-CONTAMINATED; #4 ResAdd + RMSNorm + Router + TopK (F4_topk): TENANT-CONTAMINATED; #5 RMSNorm + Router (F5): TENANT-CONTAMINATED; #5 RMSNorm + Router + TopK (F5_topk): TENANT-CONTAMINATED. These are NOT inside the drift band and NOT a null result: their cells were measured and then rejected for a stated reason (see excluded.csv). Re-measuring them is not cleanup, it is the experiment. 7 of 8 judged (fusion, variant) group(s), spanning 3 of 3 families, have at least one cell outside the [-0.061, +0.047] drift band (#6 Up_Gate + SwiGLU (-), #9 Down + Expert Merge + ResAdd2 (atomic (sglang FUSE_SUM_ALL_REDUCE)), #8 Down + Expert Merge (token-major), #9 Down + Expert Merge + ResAdd2 (token-major), #11a+#11b combined (one norm charged once) (combined), #11a Lazy Pre-Norm -> w13 grouped GEMM (lazy pre-norm (prologue)), #11b Lazy Pre-Norm -> router GEMM (lazy pre-norm (prologue))); the other 1 judged group(s) sit entirely inside it. WHAT THAT IS WORTH: 12 of 86 judged CELLS fall outside, against 7.8 expected by chance alone -- a min/max band over 21 reference cells is exceeded by a fresh exchangeable draw 9.1 % of the time, by construction. One-sided binomial P(X >= 12) = 0.089, so the count is **CONSISTENT WITH DRIFT** and is NOT a finding. Per group of 11 cells the chance of at least one exceedance is 65 %, which is why most groups have one; the group-level count below is an artefact of that, not a result. BAND SENSITIVITY: the band above is [-0.061, +0.047] over 21 f03/f10 cells after 1 cell(s) were excluded by the stated validity rule; retaining them instead gives [-0.061, +0.731] over 22 cells (min detectable effect 0.731 instead of 0.061), against which 4 judged cell(s) in any family would fall outside. Both bands are published; neither licenses a causal claim.

**Baseline** `/home/shuhan/fusion/results/h200` · content digest `b23c4f6530c6 over 28 file(s)` (recomputed here, at report time) · driver-recorded fingerprint digest `60509c7745bf5748` · report-time repo HEAD `8cdef5d4fadf` *(the HEAD of the working tree that generated this report; it may be dirty and is NOT a claim about which commit produced the baseline files -- the content digest is)*

**Resolving power**: the drift band is `[-0.061, +0.047]`, so the smallest effect this report can call OUTSIDE in either direction is **0.061** (band half-width 0.054). Any INSIDE verdict below means *"smaller than 0.061, or absent"* -- it does not mean zero.

> **UNDERPOWERED -- THIS IS THE RESULT.** 0.061 is wider than 0.060, and 0.060 is the LARGEST of the six lever-offering variants' H200-vs-C500 effects, not the smallest -- so the deficit below understates the problem for most of them. A null at this band width is **not** evidence of no effect; it is evidence that this session pair could not resolve one. Nor do the exceedances rescue it: they arrive at very close to the rate a min/max band produces by construction (see the Verdict above), and under a leave-one-family-out band, or against the two arms' own within-cell p10-p90 dispersion, they largely disappear. **The defensible headline for this run is that the design could not answer the question**, not that it answered it either way. Re-run as a same-session paired A/B.

## 0. What this is, and the one thing it cannot tell you

The operator's design decision, honoured exactly: **one arm vs the existing campaign** -- run only the `classic` arm now and diff it against the already-committed `results/h200/*.json`. There is no same-session paired arm.

The cost of that choice is stated plainly and not buried: because the two arms were measured in different sessions, **cross-session drift -- thermals, co-tenancy, clock state, driver state -- is confounded with the arm's feature switch.** A delta between the two numbers is not, on its own, evidence about TMA, warp specialization or thread-block clusters.

**What the delta actually is.** Every number in this report compares a FUSION GAIN to a FUSION GAIN: each side's `paired_speedup`, the median of the per-round ratios from that session's own interleaved A/B loop, and `delta_rel = classic_speedup / hopper_speedup - 1`. It is a ratio of ratios. It moves when the UNFUSED chain moves exactly as much as when the fused kernel moves, and it can go up while both arms' absolute times go down. No sentence in this report may therefore be read as *"arm X ran faster"* -- the quantity does not say that, and the raw `*_fused_ms` / `*_unfused_ms` columns are published in every per-regime CSV precisely so a reader can check which side moved.

The defence is the f03/f10 noise floor described in §2. With it, the only two conclusions this design supports are *"the delta exceeds the drift band"* and *"the delta is inside the drift band, so nothing is shown"*. Neither of them is *"the Hopper features did nothing"*, and neither is a causal claim of any kind: an unpaired two-session comparison cannot support one.

Two further limits on attribution:

* `GLM52_H200_CLASSIC=1` forces **four** capabilities off, including `wgmma`, not just the three levers in the question. A delta on a GEMM-carrying family (`f01`, `f06`, `f08f09`) cannot be attributed to TMA / warp-spec / clusters alone from this arm; the per-feature arms exist for that and are behind a flag.
* The band comes from two short memory-bound vector kernels. Whether their run-to-run variability bounds the variability of a 14-hour GEMM family is an **assumption, not a measurement**. It is the best available control under the chosen design.

## 1. Headline

| fusion | variant | family | offered axes | engagement | cells outside the band | median delta | verdict | reading |
|---|---|---|---|---|---|---|---|---|
| #1 o_proj + ResAdd | triton | f01 | tma\|warp_specialize\|clusters | ENGAGED | EXCLUDED (11 of 11 cells) | n/a | TENANT-CONTAMINATED | EXCLUDED from the published comparison and marked for RE-MEASUREMENT: every cell was measured while another process held the card, against a baseline measured on an idle card. This is NOT an inside-the-band result and NOT a null -- it is the absence of a usable measurement [0 of 0 usable cells above the band, 0 below, 0 inside] [11 of 11 cell(s) EXCLUDED by a stated rule and marked for RE-MEASUREMENT: TENANT-CONTAMINATED] |
| #3 ResAdd + RMSNorm | - | f03 | none | NOTHING-TO-DISABLE | 0 / 11 | +0.000 | NOISE-FLOOR | defines the drift band; its delta IS the cross-session noise this design concedes [11 usable cell(s), every one of them a SAMPLE of the band rather than a measurement against it] |
| #4 ResAdd + RMSNorm + Router | F4 | f04f05 | warp_specialize\|clusters | ENGAGED | EXCLUDED (11 of 11 cells) | n/a | TENANT-CONTAMINATED | EXCLUDED from the published comparison and marked for RE-MEASUREMENT: every cell was measured while another process held the card, against a baseline measured on an idle card. This is NOT an inside-the-band result and NOT a null -- it is the absence of a usable measurement [0 of 0 usable cells above the band, 0 below, 0 inside] [11 of 11 cell(s) EXCLUDED by a stated rule and marked for RE-MEASUREMENT: TENANT-CONTAMINATED] |
| #4 ResAdd + RMSNorm + Router + TopK | F4_topk | f04f05 | warp_specialize\|clusters | ENGAGED | EXCLUDED (11 of 11 cells) | n/a | TENANT-CONTAMINATED | EXCLUDED from the published comparison and marked for RE-MEASUREMENT: every cell was measured while another process held the card, against a baseline measured on an idle card. This is NOT an inside-the-band result and NOT a null -- it is the absence of a usable measurement [0 of 0 usable cells above the band, 0 below, 0 inside] [11 of 11 cell(s) EXCLUDED by a stated rule and marked for RE-MEASUREMENT: TENANT-CONTAMINATED] |
| #5 RMSNorm + Router | F5 | f04f05 | warp_specialize\|clusters | ENGAGED | EXCLUDED (11 of 11 cells) | n/a | TENANT-CONTAMINATED | EXCLUDED from the published comparison and marked for RE-MEASUREMENT: every cell was measured while another process held the card, against a baseline measured on an idle card. This is NOT an inside-the-band result and NOT a null -- it is the absence of a usable measurement [0 of 0 usable cells above the band, 0 below, 0 inside] [11 of 11 cell(s) EXCLUDED by a stated rule and marked for RE-MEASUREMENT: TENANT-CONTAMINATED] |
| #5 RMSNorm + Router + TopK | F5_topk | f04f05 | warp_specialize\|clusters | ENGAGED | EXCLUDED (11 of 11 cells) | n/a | TENANT-CONTAMINATED | EXCLUDED from the published comparison and marked for RE-MEASUREMENT: every cell was measured while another process held the card, against a baseline measured on an idle card. This is NOT an inside-the-band result and NOT a null -- it is the absence of a usable measurement [0 of 0 usable cells above the band, 0 below, 0 inside] [11 of 11 cell(s) EXCLUDED by a stated rule and marked for RE-MEASUREMENT: TENANT-CONTAMINATED] |
| #6 Up_Gate + SwiGLU | - | f06 | tma\|warp_specialize\|clusters | ENGAGED | 2 / 11 | +0.001 | MIXED | the usable cells do not agree: some fall outside the drift band, some inside, and/or they fall out on both sides; read the per-regime CSVs before claiming anything [1 of 11 usable cells above the band, 1 below, 9 inside] |
| #8 Down + Expert Merge | atomic (sglang FUSE_SUM_ALL_REDUCE) | f08f09 | tma\|warp_specialize\|clusters | ENGAGED | 0 / 11 | -0.004 | INSIDE-BAND | every usable cell sits inside the drift band -- nothing is shown for this family, which is not the same claim as 'no effect' [0 of 11 usable cells above the band, 0 below, 11 inside] |
| #9 Down + Expert Merge + ResAdd2 | atomic (sglang FUSE_SUM_ALL_REDUCE) | f08f09 | tma\|warp_specialize\|clusters | ENGAGED | 1 / 9 | -0.005 | MIXED | the usable cells do not agree: some fall outside the drift band, some inside, and/or they fall out on both sides; read the per-regime CSVs before claiming anything [1 of 9 usable cells above the band, 0 below, 8 inside] |
| #8 Down + Expert Merge | token-major | f08f09 | tma\|warp_specialize\|clusters | ENGAGED | 2 / 11 | -0.003 | MIXED | the usable cells do not agree: some fall outside the drift band, some inside, and/or they fall out on both sides; read the per-regime CSVs before claiming anything [1 of 11 usable cells above the band, 1 below, 9 inside] |
| #9 Down + Expert Merge + ResAdd2 | token-major | f08f09 | tma\|warp_specialize\|clusters | ENGAGED | 1 / 11 | -0.002 | MIXED | the usable cells do not agree: some fall outside the drift band, some inside, and/or they fall out on both sides; read the per-regime CSVs before claiming anything [0 of 11 usable cells above the band, 1 below, 10 inside] |
| #10 Expert Merge + ResAdd | - | f10 | none | NOTHING-TO-DISABLE | 0 / 10 | +0.011 | NOISE-FLOOR | defines the drift band; its delta IS the cross-session noise this design concedes [10 usable cell(s), every one of them a SAMPLE of the band rather than a measurement against it] [1 of 11 cell(s) EXCLUDED by a stated rule and marked for RE-MEASUREMENT: INVALID-HARNESS-FLOOR] |
| #11a+#11b combined (one norm charged once) | combined | f11 | warp_specialize | ENGAGED | 2 / 11 | -0.003 | MIXED | the usable cells do not agree: some fall outside the drift band, some inside, and/or they fall out on both sides; read the per-regime CSVs before claiming anything [1 of 11 usable cells above the band, 1 below, 9 inside] |
| #11a Lazy Pre-Norm -> w13 grouped GEMM | lazy pre-norm (prologue) | f11 | warp_specialize | ENGAGED | 1 / 11 | -0.003 | MIXED | the usable cells do not agree: some fall outside the drift band, some inside, and/or they fall out on both sides; read the per-regime CSVs before claiming anything [1 of 11 usable cells above the band, 0 below, 10 inside] |
| #11b Lazy Pre-Norm -> router GEMM | lazy pre-norm (prologue) | f11 | warp_specialize | ENGAGED | 3 / 11 | +0.003 | MIXED | the usable cells do not agree: some fall outside the drift band, some inside, and/or they fall out on both sides; read the per-regime CSVs before claiming anything [3 of 11 usable cells above the band, 0 below, 8 inside] |

Rows are **(fusion, variant) groups**, not families: 15 groups over 7 families. Two groups can share a fusion label and differ only in variant, which is why the variant column is not optional.

**Noise floor (f03+f10, 21 usable of 22 cells): delta_rel in [-0.061, +0.047], median +0.005 (basis: global-minmax); minimum detectable effect 0.061.**

EXCLUDED, NOT JUDGED, AND MARKED FOR RE-MEASUREMENT: 5 of 15 (fusion, variant) group(s) across 2 famil(y/ies) -- f01, f04f05 -- #1 o_proj + ResAdd (triton): TENANT-CONTAMINATED; #4 ResAdd + RMSNorm + Router (F4): TENANT-CONTAMINATED; #4 ResAdd + RMSNorm + Router + TopK (F4_topk): TENANT-CONTAMINATED; #5 RMSNorm + Router (F5): TENANT-CONTAMINATED; #5 RMSNorm + Router + TopK (F5_topk): TENANT-CONTAMINATED. These are NOT inside the drift band and NOT a null result: their cells were measured and then rejected for a stated reason (see excluded.csv). Re-measuring them is not cleanup, it is the experiment. 7 of 8 judged (fusion, variant) group(s), spanning 3 of 3 families, have at least one cell outside the [-0.061, +0.047] drift band (#6 Up_Gate + SwiGLU (-), #9 Down + Expert Merge + ResAdd2 (atomic (sglang FUSE_SUM_ALL_REDUCE)), #8 Down + Expert Merge (token-major), #9 Down + Expert Merge + ResAdd2 (token-major), #11a+#11b combined (one norm charged once) (combined), #11a Lazy Pre-Norm -> w13 grouped GEMM (lazy pre-norm (prologue)), #11b Lazy Pre-Norm -> router GEMM (lazy pre-norm (prologue))); the other 1 judged group(s) sit entirely inside it. WHAT THAT IS WORTH: 12 of 86 judged CELLS fall outside, against 7.8 expected by chance alone -- a min/max band over 21 reference cells is exceeded by a fresh exchangeable draw 9.1 % of the time, by construction. One-sided binomial P(X >= 12) = 0.089, so the count is **CONSISTENT WITH DRIFT** and is NOT a finding. Per group of 11 cells the chance of at least one exceedance is 65 %, which is why most groups have one; the group-level count below is an artefact of that, not a result. BAND SENSITIVITY: the band above is [-0.061, +0.047] over 21 f03/f10 cells after 1 cell(s) were excluded by the stated validity rule; retaining them instead gives [-0.061, +0.731] over 22 cells (min detectable effect 0.731 instead of 0.061), against which 4 judged cell(s) in any family would fall outside. Both bands are published; neither licenses a causal claim.

## 2. The noise floor, and why it is the whole argument

`f03` (ResAdd+RMSNorm) and `f10` (ExpertMerge+ResAdd) advertise no Hopper cfg key at all -- `kernel_cfg_keys` reads `"module advertises none"`, all three axes are `offered: false` in the campaign, and a full key-path scan of both committed files finds zero Hopper tokens in any config, tune table or `axis_counts`. Their classic arm is therefore configured **byte-identically** to their Hopper arm, and their delta contains only run-to-run and cross-session variation. That is the drift band.

The statistic is the signed relative delta `d = classic_speedup / hopper_speedup - 1`, relative rather than absolute because the two families' speedups live on different scales and an absolute band would simply be dominated by the larger one.

The published band is **min/max**, not a percentile. At n = 21 a percentile is barely resolvable -- the 10th and 90th are interpolated between the 2nd/3rd and 20th/21st order statistics -- and the question being asked is *"could cross-session drift alone have produced this?"*, whose conservative answer is the observed extremes. The p10/p90 band is published alongside in `noise_floor_by_class.csv` and `--band p10p90` switches which one drives the verdict column.

`median(d) = +0.005` is the **drift bias**: a systematic session offset that every family's delta should be read relative to, not a result.

| basis | n | min | p10 | median | p90 | max | stdev | min detectable effect (minmax) | used for verdicts |
|---|---|---|---|---|---|---|---|---|---|
| global | 21 | -0.061 | -0.046 | +0.005 | +0.035 | +0.047 | 0.032 | 0.061 | yes -- fallback for any class with n < 8 |
| decode | 17 | -0.061 | -0.048 | +0.011 | +0.039 | +0.047 | 0.035 | 0.061 | yes |
| prefill | 4 | -0.014 | -0.009 | +0.003 | +0.007 | +0.007 | 0.009 | 0.014 | no (n = 4 < 8) |

**Resolving power.** `min detectable effect` is `max(|lo|, |hi|)` of that basis's band under the active `--band minmax` statistic: the smallest |delta| that could be called OUTSIDE in either direction. For the band actually driving the verdicts it is **0.061**. The effect this study was built to see is AT MOST 0.060, and that is the LARGEST of the six lever-offering variants, not the smallest -- most are far below it, so the deficit understates. The four TMA-using families gained 1.00-1.06x on H200-vs-C500 while `f03`/`f10` gained 1.86x/1.48x, and it is that contrast the control arm exists to interrogate. **0.061 > 0.060: this run is UNDERPOWERED and an INSIDE verdict below carries no information about effects smaller than 0.061.**

A cell is judged against its own class band when that class has at least 8 samples. decode qualifies; **prefill has only 4 samples (2 families x 2 regimes) and therefore falls back to the global band, which is dominated by decode's launch-latency-bound variance.** A genuine prefill effect could be judged against an inappropriately wide band. Per-regime bands (n = 2) are published in `noise_floor_by_class.csv` as INFO only and are never used for a verdict.

Full sample: `noise_floor.csv` (22 rows, 21 usable; the guard refuses to publish below 16).

### 2b. The band BOTH WAYS -- what the exclusions cost

The drift band is built from two families and a handful of cells, so a single cell can change every verdict in this report. It is therefore published **both ways**, in the body and not in a footnote, and the reader is told what the difference buys:

| band | n f03/f10 cells | min | max | min detectable effect | judged cells outside it |
|---|---|---|---|---|---|
| PUBLISHED -- excluded cells removed | 21 | -0.061 | +0.047 | 0.061 | 12 |
| SHADOW -- excluded cells retained | 22 | -0.061 | +0.731 | 0.731 | 4 |

* The cell in question is **`f10/decode_bs16`** (2.1607 -> 3.7394, delta 0.7306), excluded as `INVALID-HARNESS-FLOOR`. control (classic) fused time 16.42 us is BELOW the harness floor that session measured for itself (39.46 us): the timed region did not enclose the work, so the number is not commensurable with the cell it is diffed against

**The decision is fully consequential, which is exactly why it is shown rather than buried.** Against the shadow band the report resolves only 4 cell(s); against the published band it resolves 12 cell(s). A reader who rejects the exclusion should read the shadow row and treat every verdict in this report as unresolved. A reader who accepts it should note that the rule was stated as a property of the instrument -- the measured harness floor -- and applied to every cell of both arms, not chosen after seeing which cell it would remove: over the 165 joined cells of both arms it rejects 1, and no campaign cell at all. (Co-tenancy is a separate exclusion with a separate cause; the two are counted separately everywhere in this report and in `excluded.csv`.)

Full detail, including the diagnostics the rule was cross-checked against (fused ticks, p90/p10 spread, drift_frac, order_gap_frac, speedup over the launch-adjusted traffic ceiling): `excluded.csv`.

### 2c. Is a two-family band wide enough? (diagnostic, drives no verdict)

The band above rests on 21 cells from two short memory-bound elementwise kernels, and the report's own §0 already concedes that whether their variability bounds a GEMM-heavy family's is an assumption. Two measurements bear on it directly, and both are printed here rather than left for a reader to discover.

**(a) The autotuner picks a different winner between the two sessions even where the two arms are identical by construction.** The engagement records report, for the very families that define the band:

| family | check | what it compared | result |
|---|---|---|---|
| f03 | V9 | identical to the campaign | 0 of 21 offered-grid stage(s) differ |
| f03 | V9b | equal to the campaign's family-constant offered size | 0 of 12 arm-only offered stage(s) differ |
| f03 | V9d | (not comparable -- winner-derived) | 19 of 28 stage(s) differ |
| f10 | V9 | identical to the campaign | 0 of 21 offered-grid stage(s) differ |
| f10 | V9b | equal to the campaign's family-constant offered size | 0 of 12 arm-only offered stage(s) differ |
| f10 | V9d | (not comparable -- winner-derived) | 19 of 28 stage(s) differ |

Read the rows together: the OFFERED grid is identical between the two arms for these families (`V9`, `V9b`), and yet the WINNER the autotuner selected out of that identical grid differs in a large fraction of stages (`V9d`). A winner change is the mechanism that moves a delta, so the families defining the noise floor are subject to the same mechanism as the families being judged -- but at whatever amplitude two elementwise kernels happen to show, which is not necessarily the amplitude a GEMM family shows.

**(b) What the band becomes if `f06`, `f08f09` are treated as further drift constituents.** On this run their offered tuner grid did not change between the arms (engagement check V9), so their delta is arguably a second drift measurement rather than a treatment contrast:

| band | n | min | max | min detectable effect | used for verdicts |
|---|---|---|---|---|---|
| published band (f03+f10) | 21 | -0.061 | +0.047 | 0.061 | yes -- drives every verdict |
| extended (f03+f10+f06+f08f09) | 74 | -0.167 | +0.116 | 0.167 | NO -- diagnostic only; judging a family against a band it belongs to is circular |

Against the extended band, 3 of the 33 remaining judged cell(s) would fall outside. Read that as a statement about the INSTRUMENT, not about the arms: it says how much of what §1 reports as resolved survives a wider and arguably more honest estimate of cross-session variation. Settling the question needs replicate runs of the no-axis families inside a single session, so the band measures within-design variance rather than one draw of it.

## 3. Did the arm actually engage?

| family | axes offered by the campaign | check outcomes |
|---|---|---|
| f01 | tma\|warp_specialize\|clusters | INFO=5; PASS=22 |
| f03 | none | INFO=3; PASS=14; VACUOUS=7 |
| f04f05 | warp_specialize\|clusters | INFO=5; PASS=19; VACUOUS=2 |
| f06 | tma\|warp_specialize\|clusters | INFO=4; PASS=23 |
| f08f09 | tma\|warp_specialize\|clusters | INFO=4; PASS=23 |
| f10 | none | INFO=3; PASS=14; VACUOUS=7 |
| f11 | warp_specialize | INFO=9; PASS=35; VACUOUS=5 |

`f03` and `f10` are **VACUOUS** for the config-level checks: they have nothing to disable, so a clean token scan for them is not evidence of engagement. The proof for those two is capability-level -- `available` flips `true -> false` with `evidence` containing `(source env)`, and `not_offered_because` changes from *"this kernel module advertises no cfg key for it"* to *"the live capability probe says it is unavailable"*. Full audit trail: `engagement.csv`.

## 3b. What was excluded from the comparison, and why

Excluded cells keep every measured number in the per-regime CSVs and in `excluded.csv`. What they lose is standing: they enter no drift band, no family aggregate and no headline count. **An exclusion is not a null result** -- it is the absence of a usable measurement, and each one below carries a RE-MEASURE flag.

| exclusion | cell(s) | variant(s) | campaign speedup | control speedup | delta withheld | re-measure | evidence |
|---|---|---|---|---|---|---|---|
| TENANT-CONTAMINATED | f04f05: ALL 44 cell(s) | F4, F4_topk, F5, F5_topk | - | - | 44 deltas withheld | YES | measured on a card that was NOT idle, against a campaign baseline that was: the card held 63259/143771 MiB at [hw after ] 2026-08-11 15:51:16 against 0 MiB at [hw before] -- a neighbour arrived at an unrecorded point inside the measuring window, so no cell of this family can be exonerated; the driver named this family: at 2026-08-11 15:51:18 it reported pid 3254840 /usr/bin/python (61.8 GB) on GPU 0. Source: /home/shuhan/fusion/log/run_control_h200/driver.log. |
| TENANT-CONTAMINATED | f01: ALL 11 cell(s) | triton | - | - | 11 deltas withheld | YES | measured on a card that was NOT idle, against a campaign baseline that was: the card held 124498/143771 MiB at [hw after ] 2026-08-11 14:47:29 against 0 MiB at [hw before] -- a neighbour arrived at an unrecorded point inside the measuring window, so no cell of this family can be exonerated; the driver named this family: at 2026-08-11 14:47:30 it reported pid 2916934 VLLM::Worker_TP0_EP0 (121.6 GB) on GPU 0. Source: /home/shuhan/fusion/log/run_control_h200/driver.log.; ALSO this cell's checkpoint was saved at 2026-08-11 13:40:31, OUTSIDE the 2026-08-11 14:39:58-2026-08-11 14:47:29 window in which the rest of f01 was measured: it was inherited from an earlier abandoned attempt and reused rather than re-measured, so it was not taken in the session this comparison is diffing |
| INVALID-HARNESS-FLOOR | f10/decode_bs16 | - | 2.1607 | 3.7394 | 0.7306 | YES | control (classic) fused time 16.42 us is BELOW the harness floor that session measured for itself (39.46 us): the timed region did not enclose the work, so the number is not commensurable with the cell it is diffed against |

Per-cell numbers for every row above -- including all 55 cells of the family-wide exclusions -- are in `excluded.csv`.

### 3b.1 Co-tenancy, derived from the driver's log and not from the summary

Co-tenancy is parsed out of **`/home/shuhan/fusion/log/run_control_h200/driver.log`**, the driver's append-only transcript, and NOT out of `control_arm_summary.json`. The driver rewrites its summary on every invocation; the last invocation of this run only re-verified the families that were already staged, so the surviving summary records `gpu.tenant_events == []` and `wall_s == 0.0, attempts == []` for exactly the families whose contamination matters. Reading it would publish them as clean. The log still carries every `[hw before]` / `[hw after ]` nvidia-smi snapshot and every `!! a co-tenant appeared` line.

Attribution is to a **stage**, never to a family name: a family can appear several times in the log because a killed invocation is re-attempted, and the stage that counts is the one whose window contains the staged file's own `_meta.recorded_at`. That is what lets an attempt begun on a dirty card and abandoned sit in the log without condemning a family that was later re-measured on a clean one.

| family | measuring window (the stage that produced the staged file) | wall | card MiB before | card MiB after | attempts in log | occupancy | evidence |
|---|---|---|---|---|---|---|---|
| f01 | 2026-08-11 14:39:58 - 2026-08-11 14:47:29 | 7.5 min | 0 | 124498 | 3 | CONTAMINATED -- EXCLUDED | the card held 124498/143771 MiB at [hw after ] 2026-08-11 14:47:29 against 0 MiB at [hw before] -- a neighbour arrived at an unrecorded point inside the measuring window, so no cell of this family can be exonerated; the driver named this family: at 2026-08-11 14:47:30 it reported pid 2916934 VLLM::Worker_TP0_EP0 (121.6 GB) on GPU 0 |
| f03 | 2026-08-11 10:57:38 - 2026-08-11 10:58:40 | 1.0 min | 0 | 0 | 1 | clean | no co-tenant line, both snapshots below the bar |
| f04f05 | 2026-08-11 15:43:34 - 2026-08-11 15:51:16 | 7.7 min | 0 | 63259 | 2 | CONTAMINATED -- EXCLUDED | the card held 63259/143771 MiB at [hw after ] 2026-08-11 15:51:16 against 0 MiB at [hw before] -- a neighbour arrived at an unrecorded point inside the measuring window, so no cell of this family can be exonerated; the driver named this family: at 2026-08-11 15:51:18 it reported pid 3254840 /usr/bin/python (61.8 GB) on GPU 0 |
| f06 | 2026-08-11 19:58:26 - 2026-08-11 20:22:24 | 23.9 min | 0 | 0 | 1 | clean | no co-tenant line, both snapshots below the bar |
| f08f09 | 2026-08-11 20:22:25 - 2026-08-11 20:46:18 | 23.9 min | 0 | 0 | 1 | clean | no co-tenant line, both snapshots below the bar |
| f10 | 2026-08-11 13:37:10 - 2026-08-11 13:39:13 | 2.0 min | 0 | 0 | 1 | clean | no co-tenant line, both snapshots below the bar |
| f11 | 2026-08-11 19:32:13 - 2026-08-11 19:58:24 | 26.2 min | 4 | 0 | 2 | clean | no co-tenant line, both snapshots below the bar |

The bar is `--tenant-mib 1024`. Both snapshots are taken by the driver while none of its own children is running, so on an idle card they read 0-4 MiB. A second, independent witness -- the child process's own nvidia-smi snapshot in the staged file's `_meta.hwinfo.gpu` -- is printed per family in `provenance.csv`.

A contaminated family is excluded rather than down-weighted because the baseline it is diffed against was measured on an idle card. A neighbour holding most of a 143.8 GB card does not add variance to a memory-bound ratio; it removes the comparison. And because the driver's snapshots cannot say WHEN inside the window the neighbour arrived, no individual cell of such a family can be exonerated.

### 3b.2 The per-cell instrument-validity rule

Stated before the numbers, computed per cell from the published JSON, applied identically to both arms, and making no reference to the speedup or to the drift band:

> A cell is INVALID if its fused time falls below the `harness_floor_us` that its own session measured.

The harness floor is what that session clocked an *empty* timed region at. A kernel timing underneath it is not a fast kernel; it is the harness failing to enclose the work. Because it is a property of the instrument rather than of the answer, the rule can be checked on every cell without knowing what it will remove -- and it is reported in §2b exactly how much the report's conclusions depend on what it did remove.

### 3b.3 Spliced provenance

A cell whose `_ckpt/<family>/<regime>.json` `saved_at` falls outside its family's measuring window was inherited from an earlier, abandoned attempt and reused rather than re-measured -- so it was not taken in the session this comparison is diffing. This is visible nowhere else: not in the result file's `_meta.recorded_at`, which records only when the merged file was written, and not in the summary.

## 4. Per-cell results

One CSV per regime:

* `control_decode_bs1.csv` (15 rows)
* `control_decode_bs2.csv` (15 rows)
* `control_decode_bs4.csv` (15 rows)
* `control_decode_bs8.csv` (15 rows)
* `control_decode_bs16.csv` (15 rows)
* `control_decode_bs32.csv` (15 rows)
* `control_decode_bs256.csv` (15 rows)
* `control_decode_bs512.csv` (15 rows)
* `control_decode_bs1024.csv` (15 rows)
* `control_prefill_t2048.csv` (15 rows)
* `control_prefill_t8192.csv` (15 rows)

Column glossary:

* `hopper_*` is the committed campaign side; `classic_*` is the `--arm` side -- the column names stay `classic_*` whatever the arm is, and the `arm` column names it.
* Each side's `speedup` is that side's own `paired_speedup` -- the median of the per-round ratios from its interleaved A/B loop, which cancels monotone drift within a session. It is NOT `unfused_ms / fused_ms`: those two columns are ratio-of-medians over separately-timed arms and differ from the paired statistic in 329 of 330 cells (worst gap 21 %). Use them to see WHICH SIDE MOVED, never to re-derive `speedup`. `ratio = classic_speedup / hopper_speedup` is therefore a ratio of ratios: it rises when the arm's unfused chain got slower just as readily as when its fused kernel got faster. Read `*_fused_ms` and `*_unfused_ms` before saying which side moved.
* `delta_rel = ratio - 1`; `delta_abs = classic_speedup - hopper_speedup` (a difference of gains, not of milliseconds).
* `band_lo` / `band_hi` / `band_basis` -- the drift band this particular cell was judged against.
* `delta_rel_raw` is populated only for `UNRESOLVED-ONE-SIDE` cells, from `speedup_raw`; it is excluded from the band and from every verdict.
* `exclusion` / `exclusion_reason` -- populated when this report refused to use the cell. The measured numbers are still printed; the exclusion only removes the cell's standing. Every excluded cell is also listed, with its diagnostics, in `excluded.csv`.
* **An empty numeric cell means NOT MEASURED. It is never coerced to zero.** An excluded cell, by contrast, has numbers AND a verdict saying they are not being used -- the two states are never conflated.

Verdict tokens, one sentence each:

* `OUTSIDE-HIGH` -- above the drift band: the fusion gain (unfused/fused) was LARGER in the classic arm than in the campaign by more than the drift band explains -- which of the two arms' kernels moved, and why, this unpaired design cannot say.
* `OUTSIDE-LOW` -- below the drift band: the fusion gain (unfused/fused) was SMALLER in the classic arm than in the campaign by more than the drift band explains -- which of the two arms' kernels moved, and why, this unpaired design cannot say.
* `INSIDE` -- inside the drift band -- nothing is shown.
* `NOISE-FLOOR` -- this family defines the drift band and therefore has no verdict of its own.
* `TENANT-CONTAMINATED` -- EXCLUDED and marked for RE-MEASUREMENT: this family was measured while another process held the card, and the campaign baseline it is diffed against was measured on an idle card. A neighbour holding most of a 143.8 GB card does not add noise to a memory-bound ratio, it removes the comparison. The delta is printed for inspection and enters no band, no aggregate and no verdict.
* `INVALID-HARNESS-FLOOR` -- EXCLUDED and marked for RE-MEASUREMENT: the fused time falls below the harness floor that this very session measured, so the cell is not commensurable with the cell it is diffed against. The delta is printed for inspection and enters no band, no aggregate and no verdict; `excluded.csv` carries the diagnostics and the band is reported both with and without it.
* `PROVENANCE-SPLICED` -- EXCLUDED and marked for RE-MEASUREMENT: this cell's checkpoint was saved outside its family's measuring window, i.e. it was inherited from an earlier abandoned attempt and reused rather than re-measured, so it was not taken in the session the rest of the family was taken in.
* `UNRESOLVED-ONE-SIDE` -- UNRESOLVED on at least one side; the delta is undefined and is excluded from the band and from every verdict.
* `MISSING-CONTROL` -- not measured in the control arm.
* `MISSING-CAMPAIGN` -- not present in the campaign.
* `UNMAPPED` -- no (fusion, variant) label for this cell; reported, never aggregated.
* `INCOMPARABLE-BASIS` -- the two sides came from different result files, so the timing basis differs (CUDA-graph replay vs L2-flushed wall).
* `EXCLUDED-HALF` -- #11b' half-fused publishes its own router speedup rather than going through best_speedup, so it is not comparable through this path.

`#11b' half-fused pre-norm -> router GEMM` is excluded by name: it publishes `half_fused.router_speedup_vs_unfused` directly rather than through `best_speedup`, so it is not comparable through this path.

## 5. Provenance and the two sessions

| item | campaign | control | match | severity |
|---|---|---|---|---|
| gpu_uuid | b2318e71-edbf-aa1b-dcf3-3cc3e6ea7db0 | b2318e71-edbf-aa1b-dcf3-3cc3e6ea7db0 | yes | info |
| device_name | NVIDIA H200 | NVIDIA H200 | yes | info |
| harness_floor_us_f03 | 36.914 | 39.456 | yes | info |
| harness_floor_us_f10 | 36.914 | 39.456 | yes | info |
| harness_floor_us_f01 | 36.914 | 39.456 | yes | info |
| harness_floor_us_f04f05 | 36.914 | 39.456 | yes | info |
| harness_floor_us_f11 | 36.914 | 39.456 | yes | info |
| harness_floor_us_f06 | 36.914 | 39.456 | yes | info |
| harness_floor_us_f08f09 | 36.914 | 39.456 | yes | info |
| timer_tick_us | 0.032 | 0.032 | yes | info |
| timer_match_frac | 1.000 (timer.match_frac) | 1.000 (timer.match_frac) | yes | info |
| unresolved_ticks | 3 | 3 | yes | info |
| preflight_recorded_at | 2026-08-07 10:30:37 | 2026-08-11 10:57:30 | no | info |
| torch_version | 2.11.0+cu130 | 2.11.0+cu130 | yes | info |
| triton_version | 3.6.0 | 3.6.0 | yes | info |
| driver_version | 580.173.02 | 580.173.02 | yes | info |
| sm_clock_start | 345.0 | 345.0 | n/a | info |
| sm_clock_end | 375.0 | 390.0 | n/a | info |
| sm_clock_drift_pct | 8.7 | 13.0 | n/a | info |
| tenant_events (as recorded in each summary) | 0 | 0 -- NOT evidence of an idle card: the control driver rewrites this file on every invocation and the last one only re-verified the already-staged families. See the driver-log rows below. | n/a | info |
| session_recorded_at | 2026-08-07 13:33:16 | 2026-08-11 20:46:20 | no | info |
| wall_s | 6421 | 4455 | no | info |
| env_uuid_f03 | b2318e71 | b2318e71 | yes | info |
| env_uuid_f10 | b2318e71 | b2318e71 | yes | info |
| env_uuid_f01 | b2318e71 | b2318e71 | yes | info |
| env_uuid_f04f05 | b2318e71 | b2318e71 | yes | info |
| env_uuid_f11 | b2318e71 | b2318e71 | yes | info |
| env_uuid_f06 | b2318e71 | b2318e71 | yes | info |
| env_uuid_f08f09 | b2318e71 | b2318e71 | yes | info |
| co-tenancy source | summary: gpu.tenant_events | /home/shuhan/fusion/log/run_control_h200/driver.log (the summary's tenant_events was overwritten by a later invocation of the driver and is empty) | n/a | info |
| card_mib_f01 (driver, before -> after) | 0 -> 0 MiB (idle card; results/h200/summary.json _meta.gpu_was_idle) | 0 -> 124498 MiB over 2026-08-11 14:39:58 - 2026-08-11 14:47:29, wall 7.5 min [3 attempt(s) in the log; this is the one whose window contains the staged file's recorded_at] | no | EXCLUDES-CELLS |
| card_mib_f01 (child's own snapshot at record) |  | used 126989 MiB / free 16168 MiB, util 4 %, throttle 0x0000000000000004 | n/a | info |
| tenant_evidence_f01 |  | the card held 124498/143771 MiB at [hw after ] 2026-08-11 14:47:29 against 0 MiB at [hw before] -- a neighbour arrived at an unrecorded point inside the measuring window, so no cell of this family can be exonerated | no | EXCLUDES-CELLS |
| tenant_evidence_f01 |  | the driver named this family: at 2026-08-11 14:47:30 it reported pid 2916934 VLLM::Worker_TP0_EP0 (121.6 GB) on GPU 0 | no | EXCLUDES-CELLS |
| other_attempt_f01 |  | 2026-08-11 13:39:14 (run 2, no exit line: a new driver invocation started before this stage reported an exit, card 0 -> ? MiB) -- produced no published payload | n/a | info |
| other_attempt_f01 |  | 2026-08-11 14:39:44 (run 3, no exit line: a new driver invocation started before this stage reported an exit, card ? -> ? MiB) -- produced no published payload | n/a | info |
| card_mib_f03 (driver, before -> after) | 0 -> 0 MiB (idle card; results/h200/summary.json _meta.gpu_was_idle) | 0 -> 0 MiB over 2026-08-11 10:57:38 - 2026-08-11 10:58:40, wall 1.0 min | yes | info |
| card_mib_f03 (child's own snapshot at record) |  | used 1269 MiB / free 141888 MiB, util 0 %, throttle 0x0000000000000000 | n/a | info |
| card_mib_f04f05 (driver, before -> after) | 0 -> 0 MiB (idle card; results/h200/summary.json _meta.gpu_was_idle) | 0 -> 63259 MiB over 2026-08-11 15:43:34 - 2026-08-11 15:51:16, wall 7.7 min [2 attempt(s) in the log; this is the one whose window contains the staged file's recorded_at] | no | EXCLUDES-CELLS |
| card_mib_f04f05 (child's own snapshot at record) |  | used 66288 MiB / free 76869 MiB, util 0 %, throttle 0x0000000000000000 | n/a | info |
| tenant_evidence_f04f05 |  | the card held 63259/143771 MiB at [hw after ] 2026-08-11 15:51:16 against 0 MiB at [hw before] -- a neighbour arrived at an unrecorded point inside the measuring window, so no cell of this family can be exonerated | no | EXCLUDES-CELLS |
| tenant_evidence_f04f05 |  | the driver named this family: at 2026-08-11 15:51:18 it reported pid 3254840 /usr/bin/python (61.8 GB) on GPU 0 | no | EXCLUDES-CELLS |
| other_attempt_f04f05 |  | 2026-08-11 14:47:30 (run 4, no exit line: a new driver invocation started before this stage reported an exit, card 124498 -> ? MiB, ON AN OCCUPIED CARD) -- produced no published payload | n/a | info |
| card_mib_f06 (driver, before -> after) | 0 -> 0 MiB (idle card; results/h200/summary.json _meta.gpu_was_idle) | 0 -> 0 MiB over 2026-08-11 19:58:26 - 2026-08-11 20:22:24, wall 23.9 min | yes | info |
| card_mib_f06 (child's own snapshot at record) |  | used 13425 MiB / free 129732 MiB, util 76 %, throttle 0x0000000000000000 | n/a | info |
| card_mib_f08f09 (driver, before -> after) | 0 -> 0 MiB (idle card; results/h200/summary.json _meta.gpu_was_idle) | 0 -> 0 MiB over 2026-08-11 20:22:25 - 2026-08-11 20:46:18, wall 23.9 min | yes | info |
| card_mib_f08f09 (child's own snapshot at record) |  | used 9993 MiB / free 133164 MiB, util 0 %, throttle 0x0000000000000000 | n/a | info |
| card_mib_f10 (driver, before -> after) | 0 -> 0 MiB (idle card; results/h200/summary.json _meta.gpu_was_idle) | 0 -> 0 MiB over 2026-08-11 13:37:10 - 2026-08-11 13:39:13, wall 2.0 min | yes | info |
| card_mib_f10 (child's own snapshot at record) |  | used 983 MiB / free 142174 MiB, util 0 %, throttle 0x0000000000000000 | n/a | info |
| card_mib_f11 (driver, before -> after) | 0 -> 0 MiB (idle card; results/h200/summary.json _meta.gpu_was_idle) | 4 -> 0 MiB over 2026-08-11 19:32:13 - 2026-08-11 19:58:24, wall 26.2 min [2 attempt(s) in the log; this is the one whose window contains the staged file's recorded_at] | yes | info |
| card_mib_f11 (child's own snapshot at record) |  | used 25761 MiB / free 117395 MiB, util 0 %, throttle 0x0000000000000000 | n/a | info |
| other_attempt_f11 |  | 2026-08-11 15:51:18 (run 5, no exit line: a new driver invocation started before this stage reported an exit, card 63259 -> ? MiB, ON AN OCCUPIED CARD) -- produced no published payload | n/a | info |
| tenant_event_1 | none recorded | 2026-08-11 14:47:30 GPU 0: pid 2916934 VLLM::Worker_TP0_EP0 (121.6 GB) | no | TENANT |
| tenant_event_2 | none recorded | 2026-08-11 15:51:18 GPU 0: pid 3254840 /usr/bin/python (61.8 GB) | no | TENANT |
| tenant_event_3 | none recorded | 2026-08-11 15:51:59 GPU 0: pid 3254840 /usr/bin/python (68.5 GB), pid 3264022 sglang::scheduler_DP0_TP0_EP0 (32.1 GB) | no | TENANT |
| tenant_mib_bar |  | a driver snapshot above 1024 MiB is read as somebody else's allocation (--tenant-mib) | n/a | info |

Flags raised while generating this report:

* co-tenant event at 2026-08-11 15:51:59 (pid 3254840 /usr/bin/python (68.5 GB), pid 3264022 sglang::scheduler_DP0_TP0_EP0 (32.1 GB)) is NOT attributable to any published family stage: "at the end of the run: new compute process(es): pid 3254840 /usr/bin/python (68.5 GB), pid 3264022 sglang::scheduler_DP0_TP0_EP0 (32.1 GB); memory in use grew +87.1 GB since the campaign started. Families measured from here on are not comparable with the earlier ones.". It is recorded here and excludes nothing -- but it is evidence about the host, and a re-measurement should not be scheduled on the assumption that the card is quiet.
* CO-TENANT CONTAMINATION -> EXCLUDED and marked for RE-MEASUREMENT: f01 (measured on a card that was NOT idle, against a campaign baseline that was: the card held 124498/143771 MiB at [hw after ] 2026-08-11 14:47:29 against 0 MiB at [hw before] -- a neighbour arrived at an unrecorded point inside the measuring window, so no cell of this family can be exonerated; the driver named this family: at 2026-08-11 14:47:30 it reported pid 2916934 VLLM::Worker_TP0_EP0 (121.6 GB) on GPU 0. Source: /home/shuhan/fusion/log/run_control_h200/driver.log.); f04f05 (measured on a card that was NOT idle, against a campaign baseline that was: the card held 63259/143771 MiB at [hw after ] 2026-08-11 15:51:16 against 0 MiB at [hw before] -- a neighbour arrived at an unrecorded point inside the measuring window, so no cell of this family can be exonerated; the driver named this family: at 2026-08-11 15:51:18 it reported pid 3254840 /usr/bin/python (61.8 GB) on GPU 0. Source: /home/shuhan/fusion/log/run_control_h200/driver.log.) The campaign baseline was measured on an idle card, so these families are not comparable at any confidence. They are named in the headline, in family_verdicts.csv, in excluded.csv and in every per-regime CSV, and they enter no band and no aggregate.
* SPLICED PROVENANCE -> EXCLUDED and marked for RE-MEASUREMENT: f01/decode_bs1. These cells were carried over from an abandoned attempt in a different session and are visible only in results/.../_ckpt/<family>/<regime>.json's saved_at field.
* INSTRUMENT VALIDITY -> EXCLUDED and marked for RE-MEASUREMENT: f10/f10/decode_bs16: control (classic) fused time 16.42 us is BELOW the harness floor that session measured for itself (39.46 us): the timed region did not enclose the work, so the number is not commensurable with the cell it is diffed against. The rule (fused time below the session's own measured harness_floor_us) is applied to all 330 cells of both arms and rejects 1. The drift band is published BOTH WAYS in noise_floor_by_class.csv and README section 2 so the exclusion can be priced.

### 5.1 Two comparisons this report deliberately does NOT make

* **Control wall time against `results/h200/summary.json` `families[*].wall_s`.** That field was frozen on 2026-08-07 and covers SEVEN regimes; `decode_bs2/4/8/16` were appended later by a separate bs-extra run and `wall_s` was never regenerated. The control arm ran ELEVEN. Diffing the two manufactures a slowdown out of a scope difference -- which is exactly what produced the driver's own "took 24 min vs the campaign's 15 min" alarm on `f06`. Normalise per regime, or add the bs-extra stage's wall, before comparing anything. No wall-time verdict is published here.
* **Anything about a per-feature lever.** See §0: this arm forces four capabilities off at once, in a different session from the baseline.

## 6. What would change the verdict

* **Re-measuring the excluded families in one idle session.** With the contaminated families out, the only clean family whose offered tuner grid measurably changed is `f11`; the two families with a real grid collapse are precisely the two that were contaminated. A one-family contrast cannot carry the study's question, so the re-measurement is not cleanup -- it is the experiment. Use a driver that STOPS on co-tenancy rather than warning, and re-measure a clean family alongside them as an in-session anchor.
* **The same-session paired A/B the operator declined.** Measuring both arms back to back on one card removes the cross-session confound entirely and makes the f03/f10 band a cross-check rather than the only defence.
* **The per-feature decomposition arms.** They separate the three levers from each other and from `wgmma`:

      python3 run_control_h200.py --arms no-tma,no-ws,no-clusters

* `GLM52_H200_CLASSIC=1` also forces `wgmma` off, so **a GEMM-family delta cannot be attributed to TMA / warp-spec / clusters alone from this arm.** This report does not pick the convenient attribution.

## 7. Reproduce

On the H200 (the operator's machine; nobody else can reach it):

      python3 /home/shuhan/fusion/run_control_h200.py --arms classic

Locally, after the tarball comes back -- this is the EXACT command that produced this report, override flags included, so re-running it reproduces this file rather than exiting 1 on a gate this run was published over:

      python3 /home/shuhan/fusion/glm52/make_control_report_h200.py \
              --control-dir /home/shuhan/fusion/results/h200/_control_arm --arm classic \
              --campaign-dir /home/shuhan/fusion/results/h200 --out /home/shuhan/fusion/report_glm52_h200/control_arm

Nothing under `results/h200/_control_arm/` is ever merged into `results/h200/*.json`. The control arm is a diff, not an append, and there is no merge step in this repo.
