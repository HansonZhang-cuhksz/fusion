# report_glm52_h200 — NVIDIA H200 (sm_90, Hopper) fusion report

Third device in the study, after `report_glm52_c500/` (the schema) and
`report_glm52_rtx4060/` (the first port). Same CSV field names, same order, so the three
directories diff against each other. **The differences below are measured facts about this
run, not formatting choices.**

**Device** NVIDIA H200, sm_90 (cc 9.0), 132 SM, warp 32, **232 448 B** per-block opt-in SMEM
(227 KiB), 65 536 regs/SM, 2048 threads/SM, **60 MiB L2**, **150.1 GB** VRAM. Clocks at their
ceiling and not throttled during selection: SM **1980 / 1980 MHz**, MEM **3201 / 3201 MHz**,
`clocks_throttle_reasons.active = 0x0`, power limit 700 W, ECC on, MIG off, persistence on,
driver 580.173.02, PCIe gen 5 x16. Stack: python 3.10.12, torch 2.11.0+cu130, CUDA 13.0,
cuDNN 91900, **triton 3.6.0**.

**Calibrated** (`glm52_h200/preflight_h200.json`, probed 2026-08-04 10:29:18 on GPU 3 of 8):

| quantity | measured |
|---|---|
| streaming copy BW (2048 MB) | **4245.3 GB/s** (1024 MB: 4237.7) |
| read-modify-write BW | 4253.9 GB/s |
| read-only BW | **4646.8 GB/s** (1024 MB: 4583.6) |
| cuBLAS bf16 GEMM 4096x16384x6144 | **829.3 TF/s** |
| Triton bf16, same shape | **782.9 TF/s** = **94.4 % of vendor** |
| cuBLAS / Triton at 8192x16384x6144 | 719.7 / 670.0 TF/s = 93.1 % |
| largest tile that compiles | `BM256 BN256 BK64 s3` and `BM128 BN256 BK64 s4` at 196 608 B; `BM128 BN256 BK128 s3` fails, needs 294 912 B > 232 448 |
| kernel launch cost | **9.079 us** |
| harness floor (empty timed region) | **+42.185 us on GPU `6c4cc3d3`, −5.975 us on GPU `59aa5198`** — see §0 |
| CUDA-event timer tick | **0.032 us**, matching **200/200** samples; `launch_timer_trustworthy: true`, no doubts recorded |

For reference, the same three rows on the sibling devices, from *their* records
(`log/LOG-14` §7 and `report_glm52_rtx4060/README.md`), not from anything measured here:
C500 1300 GB/s / Triton 107 TF/s at **50 % of vendor** / tick 0.256 us; RTX 4060 140 GB/s /
11.81 TF/s at 102 % / tick 1.024 us. The C500 study's headline — *"the real lever is not fusion,
it is the gap to the vendor BLAS"* — has no purchase on this device: Triton is at 94 % here and
102 % on the 4060.

Source data `results/h200/`. Per-family run logs `log/run_h200/`. Build and audit log
`log/LOG-14-h200-suite.md`. **Nothing under `results/h200/` or `log/run_h200/` was modified to
produce this report**; every number here was read out of the JSON.


## 0. Read this first: what the re-run fixed, and what it did not

This directory is generated from a **re-run** of the whole campaign. The first attempt split
silently across two H200s with harness floors of **−5.975 us** and **+42.185 us**, which made
its decode numbers incomparable both with each other and with its own layer sweep.

**Fixed.** All eight families now ran on **one pinned, idle card** — GPU index 1,
`GPU-59aa5198-70aa-0e6f-16b9-a6d483af9c4e`, `gpu_was_idle: True` — with a single **positive**
harness floor of **+37.669 us** and a 0.032 us timer tick. The unphysical negative floor is
gone. Every row's `notes` column still names the card and floor it was measured on, so the
next split is visible rather than silent.

**Changed since the first run: this report publishes the PAIRED estimator.** Every row carries
two ratios — a *sequential* one (two medians, one arm after the other, so a monotone clock or
thermal ramp does not cancel) and a *paired* one (median of per-repetition ratios from an
interleaved A/B/A/B loop, which does cancel it). `summary.json` published the sequential one;
the CSVs here publish the paired one. They disagree by up to **13.8 %** (f04f05 prefill_t8192
F4_topk: 0.841 sequential vs 0.957 paired), which is far too large to treat as
interchangeable. Publishing the sequential ratio is what let a physically impossible speedup
through on the RTX 4060.

**Not fixed: the layer-level effects are mostly not resolvable.** Under LOG-11's tie protocol,
**six of seven regimes are TIED** — at `decode_bs1024`, 14 configurations are statistically
indistinguishable. Only `decode_bs32` separates. Comparing the two runs shows why that matters:

| regime | run 1 winner | run 1 | run 2 winner | run 2 |
|---|---|---|---|---|
| decode_bs1 | #3 + #9 | 1.2007 | #1+#10+#11a+#11b | 1.2718 |
| decode_bs32 | greedy | 1.0089 | **greedy** (separated) | 1.0165 |
| decode_bs256 | #11b | 1.0219 | greedy | 1.0092 |
| decode_bs512 | #4 | 1.0185 | greedy | 1.0167 |
| decode_bs1024 | greedy | 1.0111 | #9 | 1.0009 |
| prefill_t2048 | #3 + #10 | 1.0148 | **#3 + #10** | 1.0101 |
| prefill_t8192 | #3 + #10 | 1.0071 | #3 | 1.0087 |

The *named winner* changes in four of seven regimes between runs. That is not a contradiction —
it is what "TIED" means, and the protocol now says so. **Do not quote a layer winner from a
tied regime as a result.** What survives both runs is: `decode_bs1` gains ~1.20–1.27x, greedy
fusion is competitive at small decode batches, and prefill sits near 1.01x.

**Lost: fusion #11 produced no rows at all.** `bench_f11` wrote `complete: true` with an empty
`rows` array. Three separate causes, all visible in `log/run_h200/f11.log`:
its F11a (`moe_fused`) arm failed the fp32 reference at rel_err **0.37–0.77** against a 0.02
tolerance, and `run_regime` raises on that — destroying the F11b router results measured in
the same regime, which were fine; the `norm` and `rstd` screens rejected **every** config
(`124 offered -> 0 valid`) and the harness then reported a time anyway
(`!! screening rejected EVERY config; timing unscreened`); and `specialization_study` hit
`PassManager::run failed` at three regimes (caught and recorded, not fatal). So `#11a`,
`#11b` and `#11b'` are **absent** from every per-regime CSV, and the four layer configurations
containing them (`O`, `P`, `Q`, `R`) were excluded for failing the fp32 reference.


## Files

| file | rows | status |
|---|---|---|
| `fusion_decode_bs1.csv`, `..._bs32`, `..._bs256`, `..._bs512`, `..._bs1024`, `..._prefill_t2048`, `..._t8192` | 16 each | measured; 4 rows empty in 5 of the 7 (§1.2) |
| `layer_optimal_per_regime.csv` | 102 | **measured end-to-end, two independent passes** (§4) |

All times are **milliseconds**, bf16, one exclusive-by-request H200 (`--gpu 3`) — see §3.5 and
§3.6 for the two ways that "exclusive" was not fully true.

---

## 1. This is the first device where the whole layer ran — and the four places it still did not

### 1.1 What capacity bought

Measured, from `preflight_h200.json` -> `capacity`:

| | bytes | |
|---|---|---|
| `w13` (256 experts) | 12 884 901 888 | 12.0 GiB |
| `w2` (256 experts) | 6 442 450 944 | 6.0 GiB |
| expert weights total | 19 327 352 832 | **18.0 GiB / 19.3 GB** |
| free at probe | 149 356 085 248 | 139.1 GiB |
| `expert_weights_fit` | **true** | |
| `whole_layer_feasible` | **true** | |

The RTX 4060 had ~7.4 GB usable against that same 18 GiB, which is why its report has no
`#6`/`#8`/`#9`/`#11a` rows and no measured layer total. Here all of it fits with 7.7x headroom,
and the whole-layer combination sweep additionally allocated the folded `w13` copy
(`layer_configurations.json` -> `prenorm.capacity`: need 30.0 GiB, fits). So:

- **all 7 regimes** ran (the 4060 has 5; `decode_bs512`/`decode_bs1024` were whole-layer regimes there);
- **all 11 implemented fusions** are rows in every per-regime CSV — #1, #3, #4, #5, #6, #8, #9, #10, #11a, #11b, #11b', 16 rows once the F5/F4/topk and atomic/token-major variants are counted separately, against **7 fusions / 9 rows** in the 4060 CSVs;
- the **whole-layer sweep is a real measurement**, not derived — §4;
- **nothing here is estimated, modelled into a measured column, or dropped for capacity.**

### 1.2 What still did not run, and why

Four honest gaps. None is a capacity gap.

**(a) #11 (lazy pre-norm) produced valid numbers at only 2 of 7 regimes.** `bench_f11` attempted
all seven; five raised and were recorded rather than hidden
(`f11_lazy_prenorm.json` -> `tune_tables.<regime>.regime_failed`):

| regime | recorded failure |
|---|---|
| `decode_bs1` | `KeyError: 'ms'` — a harness fault, not a correctness result |
| `decode_bs32` | ran |
| `decode_bs256` | ran |
| `decode_bs512` | `validation failed: moe_fused rel_err 0.4119` (tol 0.02) |
| `decode_bs1024` | `validation failed: moe_fused rel_err 0.4110` |
| `prefill_t2048` | `validation failed: moe_fused rel_err 0.3953` |
| `prefill_t8192` | `validation failed: moe_fused rel_err 0.8333` |

So the four `#11a` / `#11b` / `#11a+#11b combined` / `#11b'` rows are **EMPTY** in
`fusion_decode_bs1.csv`, `_bs512`, `_bs1024`, `_prefill_t2048`, `_prefill_t8192`, and their
`notes` say the cell was not measured. Nothing is interpolated into them. The fused w13
prologue kernel is **wrong**, not slow, at T >= 512.

**(b) The same failure removed 4 of 18 whole-layer configurations at 6 of 7 regimes.** Every
configuration that sets `prenorm: "all"` — `O_f11ab`, `P_f10_f11ab`, `Q_f8_f11ab`,
`R_f1_f10_f11ab` — failed the independent fp32 reference of the whole subgraph everywhere
except `decode_bs1` (rel_err 0.16–0.67 against tol 0.02), and a failing configuration is
excluded outright rather than timed. `N_f11b` (router prologue only) passed everywhere and is
present in every regime. Hence `layer_optimal_per_regime.csv` has 18 rows at `decode_bs1` and
14 at the other six.

**(c) #2 and #7 are still absent, and #2's blocking argument no longer holds here.** Both were
filtered on analysis without implementation (`LOG-00` §3), same as C500 and the 4060. But #2's
rejected variant (b) was *"Multi-CTA Norm via CTA clusters + distributed shared memory …
requires Hopper/Blackwell thread-block clusters and DSMEM. C500 has neither"*. **H200 has
clusters** (§2). #2 was not revisited on this device. That is a gap in the study, not a
property of the hardware, and it is recorded here rather than left implicit.

**(d) #11b' is exploratory by the bench's own label.** `f11_lazy_prenorm.json` -> `half_fused`
carries `"note": "EXPLORATORY, not the headline"`. It has no cell of its own in `summary.json`;
its CSV row is filled from that block at the two regimes where #11 ran, and its `notes` say so.

---

## 2. Hopper features: available, offered, and actually used

"Available" is a **compile-and-launch** probe, not `hasattr` — each was compiled and run, and
`tma_host_descriptor` / `tma_device_descriptor` additionally checked their output values.

| feature | probe | available |
|---|---|---|
| TMA, host descriptor (`TensorDescriptor.from_tensor`) | `tma_host_descriptor` | **PASS**, values correct |
| TMA, device descriptor (`tl.make_tensor_descriptor`) | `tma_device_descriptor` | **PASS**, values correct |
| warp specialization (`tl.range(warp_specialize=True)`) | `warp_specialize_tl_range` | **PASS** |
| warp specialization (`num_consumer_groups=` kwarg) | `warp_specialize_num_consumer_groups` | **FAIL** — `unrecognised keyword` on triton 3.6.0; this spelling is never emitted by the suite |
| thread-block clusters (`num_ctas=2`) | `thread_block_cluster_num_ctas` | **PASS** |
| `tl.dot` bf16 | `tl_dot_bf16` | **PASS** |

**DSMEM was never probed.** The cluster probe launches `num_ctas=2` and checks it runs; no
kernel in this suite reads or writes distributed shared memory. Read "clusters: available" as
"the launch attribute works", nothing more.

Availability is not use. Each kernel module advertises which cfg keys it exposes, and the
tuner can only offer an axis the module advertises
(`fairness.h200_axes.per_family.<family>.axes`):

| family | TMA offered | warp-spec offered | clusters offered | not offered because |
|---|---|---|---|---|
| f01 o_proj+ResAdd | yes | yes | yes | — |
| f03 ResAdd+RMSNorm | **no** | **no** | **no** | "this kernel module advertises no cfg key for it" |
| f04f05 norm+router | **no** | yes | yes | TMA: no cfg key |
| f06 UpGate+SwiGLU | yes | yes | yes | — |
| f08f09 down+merge | yes | yes | yes | — |
| f10 ExpertMerge+ResAdd | **no** | **no** | **no** | "this kernel module advertises no cfg key for it" |
| f11 lazy pre-norm | **no** | yes | yes | TMA: no cfg key |

And what the tuner actually **selected** into a winning mapping, counted over the 90 published
cells by walking every `*_cfg` block including its nested `gemm` / `norm` sub-configs:

| axis | selected on a fused arm | selected on an unfused arm | by family (fused / unfused) |
|---|---|---|---|
| `USE_TMA: true` | **19** | **26** | f01 4/4, f06 7/2, f08f09 8/20 |
| `warp_specialize: true` | **22** | **13** | f01 2/3, f04f05 12/4, f06 6/2, f08f09 2/4 |
| `num_ctas: 2` (clusters) | **2** | **30** | f04f05 0/20, f06 0/1, f08f09 2/8, f11 0/1 |
| `num_consumer_groups` | 0 | 0 | the spelling that failed the probe is never emitted |

Three consequences worth stating plainly. **TMA is a first-class participant** — it wins the
mapping search on both arms in all three GEMM-shaped families that offer it. **Warp
specialization tilts fused** (22 vs 13), most sharply in f04f05, where the fused kernel takes it
12 times and the unfused chain 4. **Thread-block clusters tilt hard the other way**: 30 of the
32 selections are on an unfused arm, and the only two fused-arm cluster winners in the whole
campaign are `f8_atomic` and `f9_atomic` at `decode_bs512`. The f04f05 unfused router GEMM in
particular chose a 2-CTA cluster in **all 20 of its decode cells and none of its 8 prefill
cells**, while the fused kernel never chose one anywhere.

That last asymmetry is a **tuner outcome, not an unfair grid**: the suite's stated policy is
*"every offered axis is applied to the coarse grid of BOTH arms"*, and `fairness.grids` confirms
it — at `decode_bs1`, f04f05 offered `num_ctas` on 82 of the 240 unfused-router configs and on
194 of the 557 fused configs. Both arms were shown clusters; only the unfused one kept them. It
does not explain §3.3's ABOVE CEILING flags — a *faster* baseline pushes a speedup down, not up
— but it is the kind of per-arm difference that has to be checked before a fusion result is
believed, so it is recorded.

And the two families that produce the study's largest speedups, **#3 and #10, offer no Hopper
axis at all** — plain vector kernels, whose result owes nothing to Hopper features.

On the paper's precondition (LOG-14 §1b: H200 is the first device that can test whether warp
specialization lets the sum-of-squares hide behind the MMA pipeline), the answer the suite
recorded is **mixed and regime-dependent**, from the isolation study that holds the config fixed
and toggles `FUSE_NORM` (`f11_lazy_prenorm.json` -> `headline`):

| | fusion cost, classic mainloop | fusion cost, warp-specialized |
|---|---|---|
| `decode_bs32` router | +44.19 % | **-1.75 %** ("specialization absorbs the reduction") |
| `decode_bs32` moe (w13) | +0.90 % | +0.87 % (does **not** absorb) |
| `decode_bs256` router | +39.70 % | **+63.37 %** (does **not** absorb) |
| `decode_bs256` moe (w13) | +0.37 % | +0.38 % (does **not** absorb) |

Three of four cases do not reproduce the paper's mechanism. Specialization itself also made
**both** arms far slower in absolute terms at the router (e.g. `decode_bs32` unfused
0.0226 -> 0.0879 ms). Two regimes is not a result; it is two data points, and #11 could not be
measured at the other five (§1.2a).

---

## 3. Fairness caveats — read these before quoting any number

### 3.1 Every one of the 90 cells carries the SEQUENTIAL flag

`summary.json` -> `cells[*].flags` contains, on **90 of 90** cells, verbatim:

> `SEQUENTIAL: arms were not interleaved A/B/A/B, so monotone clock or thermal drift does not cancel in this ratio`

That flag is carried into the `notes` column of every affected CSV row. It is the exact failure
that produced a physically impossible number on the RTX 4060 — #1 at `prefill_t8192` measured
1.0267x, above its own ceiling, because the fused arm ran entirely before the unfused one while
the GPU drifted 22 % (`common.py:684-692`, `LOG-13`).

**What the flag establishes, and what it does not.** The flag fires on the row's `paired` field
(`run_h200.py:1463-1468`), and `common.speedup_row()` hardcodes `"paired": False`
(`common.py:1374`) unless the caller passes `pair=`. **No H200 bench passed `pair=`.** But every
H200 bench did time through `B.bench_pair(...)`, and every one of the 90 rows records
`pair_meta.impl = "common.bench_pair"` with `pair_meta.interleaved = true`. `bench_pair`
(`common.py:672`) runs both arms inside one loop, each behind its own L2 flush, alternating
which arm leads every repetition.

So: **the arms were interleaved; the published statistic is not the paired one.** The flag's
wording overstates the problem and its substance understates nothing — the number in the
`speedup` column is still not the drift-cancelling estimator. Both readings are in the data and
both are in the notes; neither was deleted.

### 3.2 `speedup_source` — one source, and it is the weaker of the two available

`collect_cells` takes the first of `("speedup", "paired_speedup", "speedup_paired")` that is
present (`run_h200.py:1422`). On this device **all 90 cells resolve to `speedup_source =
"speedup"`** — none fell through to `paired_speedup`, none to the
`"derived from fused_ms/unfused_ms"` fallback. That single value is
`unfused.p50_ms / fused.p50_ms`: a **ratio of two medians**.

A ratio of medians and a paired per-round median are **not the same measurement**, and this
report does not present them as interchangeable. `bench_pair`'s own docstring says to publish
both when they disagree, so here is the disagreement. Every row carries the paired statistic
alongside as `paired_speedup` / `pair_meta.paired_speedup_p50`, from the *same* interleaved run:

- median divergence across the 90 cells: **0.34 %**; mean **1.52 %**;
- **24 of 90 cells differ by more than 2 %**; 35 by more than 1 %;
- worst cases:

| cell | ratio of medians (published) | paired p50 | diff | fused-first vs unfused-first gap |
|---|---|---|---|---|
| #5 `F5_topk` `prefill_t8192` | 1.3024 | **1.4724** | 13.05 % | 9.0 % |
| #4 `F4_topk` `prefill_t2048` | 1.7199 | **1.8790** | 9.25 % | 15.8 % |
| #5 `F5_topk` `decode_bs1024` | 2.1122 | **2.2893** | 8.38 % | 12.3 % |
| #4 `F4_topk` `decode_bs1024` | 1.8377 | **1.9701** | 7.21 % | 16.1 % |
| #9 `token-major` `decode_bs1` | 2.1490 | **2.0074** | 6.59 % | 9.1 % |
| #5 `F5` `prefill_t8192` | 1.0933 | **1.1565** | 5.78 % | 5.7 % |
| #9 `atomic` `decode_bs1` | 1.3194 | **1.3915** | 5.46 % | 8.0 % |
| #4 `F4` `decode_bs32` | 1.5931 | **1.6776** | 5.31 % | 7.6 % |
| #10 `decode_bs1024` | 2.0108 | **1.9278** | 4.13 % | 2.8 % |
| #3 `prefill_t8192` | 1.3124 | **1.3464** | 2.59 % | 5.9 % |

The last column is `pair_meta.order_gap_frac` — the gap between the per-round ratio when the
fused arm led and when the unfused arm led, **within the interleaved run**. It reaches **26.6 %**
(#4 `F4_topk` `decode_bs32`) and **25.8 %** (#4 `F4_topk` `prefill_t8192`). Order sensitivity of
that size means alternating the lead did not fully cancel it, and the affected cells' third
decimal place is not real. The f04f05 family carries **18 of the 24** >2 % cells; the other six
are #9 token-major and #9 atomic at `decode_bs1`, #8 token-major at `decode_bs1`, #10 at
`decode_bs1024` and `prefill_t2048`, and #3 at `prefill_t8192`.

Rule for reading the CSVs: **treat the `speedup` column as the ratio of medians it is**, and for
any f04f05 or token-major cell read the paired value out of the row's `notes` before quoting a
figure.

### 3.3 ABOVE CEILING — 23 flagged cells, and 58 that were never checked at all

23 cells exceed their own modelled traffic ceiling by more than 2 % and carry, verbatim,
`ABOVE CEILING: <x>x vs modelled ceiling <y>x`:

- **21 f04f05 cells**: every `F4`/`F5`/`F4_topk`/`F5_topk` cell at `decode_bs1` through
  `decode_bs512`, plus `F5`/`F5_topk`/`F4_topk` at `decode_bs1024` and `F5_topk`/`F4_topk` at
  `prefill_t2048`. Worst: `F5_topk` `decode_bs1024` at **2.112x against a 1.453x ceiling**.
- **2 f11 cells**: `f11b_router` at `decode_bs32` (2.211x vs 1.220x) and `decode_bs256`
  (2.085x vs 1.960x).

**The ceiling is MODELLED** (`glm52_h200/traffic.py`), and a cell is only checked if its family
JSON recorded a key literally named `ceiling` — which only f04f05 (28 rows) and f11 (4 of its 6)
did. So the check ran on **32 of 90 cells** and **never ran on f01, f03, f06, f08f09 or f10**.
**13 of the 58 unchecked cells would have flagged**: all 7 f03 cells and 6 of the 7 f10 cells
(f10 at `prefill_t8192`, 1.195x, is the one that clears its 1.20x ceiling).
`f03` records `ideal_speedup: 1.25` and measures **2.305x** at `decode_bs1`; `f10` records
`ideal_speedup: 1.20` and measures **2.132x** there, which its own `pct_of_ceiling` field —
defined as `(speedup - 1) / 0.20`, i.e. the excess over 1.0 against the ceiling's headroom —
reports as **5.77**, printed by the bench as *"577 % of 1.20x"*. Both fusions sit above their
own recorded traffic ceilings at every decode regime, and neither carries a flag saying so,
purely because of a key name. Treat the absence of an ABOVE CEILING flag on
f01/f03/f06/f08f09/f10 as **"not checked"**, never as "passed".

Two things bear on the size of these numbers and are recorded rather than argued:

- the **harness floor is +42.185 us on GPU `6c4cc3d3`** (and −5.975 us on `59aa5198`, which is
  unphysical — see §0; the argument below applies to the `6c4cc3d3` families only) against a
  9.079 us launch cost, and several decode
  arms are only 1.2–3x that floor (#3's fused arm at `decode_bs1` is 52.6 us). Ratios of numbers
  that large in fixed cost are not pure kernel-time ratios. The study's own vocabulary already
  names this: LOG-14 §6.2 recorded f10 at "208 % of its bandwidth ceiling — the
  launch-elimination signature";
- the **f04f05 unfused baselines are weak on their own recorded evidence**. At `decode_bs1` the
  tuned Triton unfused chain is 0.1316 ms while the same file records `torch_ref_ms = 0.0712 ms`
  for the whole unfused chain and `blas_router_bf16_ms = 0.0101 ms` for the router GEMM alone.
  The 4060 report flagged the same family for the same reason. A 1.79x over a baseline that
  torch beats by 1.85x is not a 1.79x.

No value anywhere in this directory was corrected, clipped or re-measured to fit a ceiling. The
flags are reported; the numbers are as recorded.

### 3.4 Timer resolution — the one thing that is clean here

Nothing on this device is tick-limited, and no cell carries a `COARSE` or `UNRESOLVED` flag.

| | C500 | RTX 4060 | **H200** |
|---|---|---|---|
| CUDA-event tick | 0.256 us | 1.024 us | **0.032 us** (200/200 samples) |
| shortest published arm | — | **9–17 ticks** at decode | **1626 ticks** (`decode_bs32`) |
| ratio quantisation | — | about **+-8 %** | **+-0.06 %** or better in every regime |

`resolved` is `true` on 90/90 cells and every `*_ticks` field is carried into the CSV notes.
Note precisely what `resolved: true` means: the two arms differ by at least 3 timer ticks
(`run_h200.py:1450`). It is a **quantisation** verdict, not a significance verdict. The smallest
gap is 15 ticks — #1 at `decode_bs1`, 1.0037x — and that cell is nowhere near resolvable against
the 0.34–13 % estimator spread of §3.2.

### 3.5 A co-tenant appeared mid-campaign

`summary.json` -> `warnings` and `gpu.tenant_events`. The driver re-checks tenancy between
families and recorded three events on the pinned card:

| detected | new processes | memory growth |
|---|---|---|
| 11:20:10, after `f11` | pid 1777470 (0.66 GB), pid 1778659 (4.46 GB) | +5.1 GB |
| 11:48:20, after `f06` | pid 1777470 (0.66 GB) | +1.1 GB |
| 12:20:29, after `f08f09` | pid 1777470 (0.66 GB) | +1.1 GB |

The driver's own words: *"Families measured from here on are not comparable with the earlier
ones."* Nothing was stopped — an aborted campaign loses more than a flagged one. Against the
family timeline (`f04f05` 10:29–10:43, `f11` 10:43–11:20, `f06` 11:20–11:48, `f08f09`
11:48–12:20, `layer` 12:20–13:51), the affected work is **f11, f06, f08f09 and the entire
whole-layer sweep**. Their CSV rows carry a `CO-TENANCY` note. `f04f05` and the three families
of §3.6 predate the first event.

### 3.6 #1, #3 and #10 were measured on a different physical GPU, on a different day

This is the largest caveat in the report and it is not visible from `summary.json`'s table.

The driver reports `f03`, `f10` and `f01` as `status: skipped_existing`, `reason: complete` — it
accepted result files already on disk. Those files' `_meta` says:

| file | GPU uuid | nvidia-smi index | recorded | preflight it was tuned against |
|---|---|---|---|---|
| `f01_oproj_resadd.json` | `59aa5198-…` | **1** | 2026-08-03 17:26 | 2026-08-03 16:47 |
| `f03_resadd_rmsnorm.json` | `59aa5198-…` | **1** | 2026-08-03 16:55 | 2026-08-03 16:47 |
| `f10_merge_resadd.json` | `59aa5198-…` | **1** | 2026-08-03 17:02 | 2026-08-03 16:47 |
| `f04f05`, `f06`, `f08f09`, `f11`, `layer` | `6c4cc3d3-…` | **3** | 2026-08-04 | 2026-08-04 10:29 |

So the study's two headline winners, **#3 and #10, plus #1**, come from a different card and a
different day than everything they are tabulated beside — including the whole-layer sweep that
scores them.

Why the fence did not catch it: `run_h200.result_is_usable()` compares `_meta.device`, the
device **name** — and both cards are called "NVIDIA H200", so the check passed. The GPU **uuid
is recorded in every file and is never compared**. LOG-14 §8 anticipated exactly this
(*"which physical card this ran on is not implied by the device name"*). Twenty-four stale
per-regime checkpoints under `_ckpt/` **were** quarantined at campaign start
(`results/h200/_quarantine_foreign_20260804_102926/`) — but only because they carried
`recorded_device: null`, no stamp at all, not because their card differed.

How much this matters, from the two probes themselves — they agree closely but not exactly:

| | GPU 1, 2026-08-03 | GPU 3, 2026-08-04 | diff |
|---|---|---|---|
| streaming BW | 4246.48 GB/s | 4245.31 GB/s | 0.03 % |
| cuBLAS GEMM | 826.74 TF/s | 829.34 TF/s | 0.31 % |
| launch cost | 10.775 us | 9.079 us | **15.7 %** |
| timer tick | 0.032 us | 0.032 us | — |
| Hopper features | tma/clusters/ws all true | identical | — |

Bandwidth and GEMM are interchangeable at this precision. **Launch cost is not**, and #3 and
#10 are precisely the launch-dominated fusions (§3.3). Their absolute times should not be
differenced against f04f05/f06/f08f09/f11 times at the microsecond level. The `notes` column of
every row of every per-regime CSV records the card it was measured on and that card's
harness floor, in the `notes` column (`MEASURED ON GPU <uuid> (family <f>), harness floor
<x> us`). Grep for `MEASURED ON GPU` to confirm.

Everything in this directory quotes the **2026-08-04 GPU-3** preflight, because that is the file
in the repo; the 2026-08-03 probe survives only as the summary stub embedded in those three
result files.

---

## 4. `layer_optimal_per_regime.csv` is a GENUINE measurement here

Unlike the 4060 — where a full MoE layer could not be instantiated at all, `bench_layer.py` was
never run, and `layer_total_ms` / `speedup_vs_unfused` were deliberately left **empty** — the
H200 ran the real whole-layer combination sweep. The C500 schema is filled with measured
numbers throughout: `regime, fusion_set, config_id, run1_ms, run2_ms, best_ms,
speedup_vs_unfused, tied_with_best_run1, tied_with_best_run2`.

**Both passes exist.** `run2_ms` is not fabricated and not copied from `run1_ms`. The protocol,
verbatim from `layer_configurations.json` -> `protocol`: `passes: 2`, `rounds_per_pass: 8`,
`rep_per_round: 15`; within a round every candidate is timed once in a fixed order that reverses
on odd rounds. `run1_ms` / `run2_ms` are the per-configuration **medians over the 8 rounds** of
pass 1 and pass 2; `best_ms = min(run1_ms, run2_ms)`, as C500 computes it;
`speedup_vs_unfused = best_ms(A_all_unfused) / best_ms(row)`.

**Tie rule (LOG-11 §3), recomputed by the generator and cross-checked against the harness's own
verdict:** order the pass by median; the noise floor is the larger of the leader's and
runner-up's round-to-round spread; every configuration within that floor of the leader is tied
with it. A winner is declared only when the gap exceeds the noise floor in **both** passes and
both passes name the same configuration; otherwise **TIED**. `tied_with_best_run1/2` carry each
pass's own verdict, so disagreement between passes is visible rather than averaged away.

Scope, as on C500: S3–S11 plus the shared expert. Attention core, MLA projections and the DSA
indexer are excluded (no fusion candidate touches them). MoE dispatch-layout construction is
outside the timed region. Routing is frozen so every configuration routes identical tokens to
identical experts. Every configuration is validated against an **independent** fp32 reference of
the whole subgraph, never against another configuration; failures are excluded (§1.2b).

### What it says

| regime | verdict | fastest row | best_ms | vs all-unfused | tied with |
|---|---|---|---|---|---|
| `decode_bs1` | **TIED** | `#3 + #9` (`I_f3_f9`) | 0.4626 | **1.2007x** | `J_greedy_all` |
| `decode_bs32` | **SEPARATED** | `#1+#6+#9 (greedy)` (`J`) | 2.9155 | 1.0089x | — |
| `decode_bs256` | **SEPARATED** | `#11b` (`N_f11b`) | 4.6668 | 1.0219x | — |
| `decode_bs512` | **TIED** | `#4` (`M_f4`) | 4.9483 | 1.0185x | `N_f11b`, `J_greedy_all`, `D_f6` |
| `decode_bs1024` | **TIED** | `#1+#6+#9 (greedy)` | 5.4915 | 1.0111x | 14 configs **including all-unfused**; the two passes disagree on which is best |
| `prefill_t2048` | **TIED** | `#3 + #10` (`H_f3_f10`) | 6.3640 | 1.0148x | 11 configs **including all-unfused**; passes disagree |
| `prefill_t8192` | **TIED** | `#3 + #10` (`H_f3_f10`) | 16.0569 | 1.0071x | 8 configs **including all-unfused** |

Read the ties literally. At `decode_bs1024`, `prefill_t2048` and `prefill_t8192` the tied set
**contains `A_all_unfused`**, so the honest statement at those three regimes is *no measurable
difference from doing nothing*. Only `decode_bs32` and `decode_bs256` produce a separated winner
in both passes. C500 was in a comparable position for a different reason — it separated its
winners at four regimes but the whole effect was 0.24–0.46 % of layer time on a machine that
throws 25–320 % excursions (LOG-11 §2–3).

Two results diverge from C500 and are worth flagging rather than smoothing:

- **"Never fuse greedily" does not hold at decode here.** On C500 `J_greedy_all` was worse than
  doing nothing at every regime, up to +48 % at `prefill_t2048`. On the H200 it is the
  **separated winner at `decode_bs32`** (1.0089x) and nominally fastest at `decode_bs1024`. At
  prefill the C500 rule survives intact: 0.9442x at `t2048` and 0.9397x at `t8192`, the worst
  rows in the file.
- **`decode_bs1` shows a far larger gain than C500 ever did**: the excess over 1.0 is **0.2007**
  here against **0.0046** on C500 at the same regime, a factor of 44. At T=1 the whole subgraph
  is 0.556 ms and the configurations differ in kernel count (recorded per config:
  `A_all_unfused` 14, `I_f3_f9` 12, `J_greedy_all` 11). But kernel count is not the whole story
  — `R_f1_f10_f11ab` also launches 11 and is the **slowest** row at 0.5597 ms — so no mechanism
  is asserted here beyond what `n_kernels` records. Note also that this is the regime where the
  pass-1 spread is worst (`A_all_unfused` round 1 is 1.357 ms against a 0.555 ms median, a 145 %
  spread), which is part of why the regime comes out TIED.

**One statistic that exists but is deliberately kept OUT of the CSV.** For four regimes the
harness also ran an interleaved paired A/B of the head configuration against all-unfused
(`paired_head_vs_unfused`): `decode_bs1` 1.1868x p50 (ratio-of-medians 1.1746x), `decode_bs32`
1.0091x, `decode_bs256` 1.0229x, `decode_bs512` 1.0093x. That is **not the same statistic** as
the `speedup_vs_unfused` column, which is a ratio of pass medians, so it is not mixed into that
column. `decode_bs1024`, `prefill_t2048` and `prefill_t8192` have no paired head at all — there
the head *is* the baseline. The generator prints all of this on stdout when it runs.

---

## 5. Measured / modelled / absent

**Measured** (every one of these is read out of `results/h200/*.json`):

- `fused_ms`, `unfused_total_ms`, `speedup` in **92 of the 112** CSV cells (7 regimes x 16 rows).
  20 are empty — the four `#11` rows at five regimes (§1.2a). Of the 92 filled, **90 have a
  `summary.json` cell**; the remaining two are `#11b'` at `decode_bs32` and `decode_bs256`, read
  out of the f11 row's `half_fused` block, which produces no cell of its own (§1.2d);
- the per-kernel `unfused_kN_ms` breakdowns wherever the bench timed a kernel alone, and blank
  where it did not (the CSV `notes` say which);
- `run1_ms`, `run2_ms`, `best_ms`, `speedup_vs_unfused` and both tie verdicts for all 102
  whole-layer rows — 18 at `decode_bs1`, 14 at each of the other six (§4);
- every winning mapping, including which Hopper axis it selected (§2);
- correctness. A recursive scan of every `{rel_err, tol, ok}` record in `results/h200/` returns
  **zero failing checks in all seven family files** — every published cell's fused and unfused
  arms were validated against an fp32 reference and passed. The only failures anywhere are the
  **24** in `layer_configurations.json` (4 configurations x 6 regimes, §1.2b), and those
  configurations were excluded rather than timed;
- the device, the clocks, the calibration and the timer tick (`preflight_h200.json`);
- the paired per-round statistic and its p10/p90 for all 90 cells — present in `notes`, not in
  the `speedup` column (§3.2);
- third baselines the raw JSON records and the CSVs surface: `torch_eager_ms`,
  `torch_compile_ms` (f03, f10), `blas_router_bf16_ms` / `torch_ref_ms` (f04f05),
  `vendor_blas_*` (f01, f06, f08f09, f11). Worth knowing before quoting a headline: at
  `decode_bs1`, torch.compile does the #3 chain in **0.0474 ms** against the fused Triton
  kernel's **0.0526 ms**, and the #10 chain in **0.0414 ms** against **0.0644 ms**. The winning
  fusions beat *their own unfused Triton arm*, which is the question this study asks; they do
  not beat torch.compile at small decode.

**Modelled — never in a column whose name implies measurement:**

- the roofline **ceilings** (`glm52_h200/traffic.py`) behind every `ABOVE CEILING` flag, and the
  `ideal_speedup` / `pct_of_ceiling` fields on f03 and f10. These are model outputs, they are
  reported as such in `notes`, and §3.3 explains that the check only ran on 32 of 90 cells;
- `gbps_fused_model` / `gbps_unfused_model` on f03, which the bench itself annotates: *"derived
  from the traffic model above, not measured — where h1 fits L2 the unfused side never moves
  those bytes and the number is fictional"*;
- `summary.json` -> `what_is_measured.not_measured` states the same boundary for the campaign as
  a whole.

**Absent — left EMPTY, never interpolated:**

- `#11a`, `#11b`, `#11a+#11b combined`, `#11b'` at `decode_bs1`, `decode_bs512`, `decode_bs1024`,
  `prefill_t2048`, `prefill_t8192` (§1.2a);
- the four `prenorm: "all"` whole-layer configurations at every regime but `decode_bs1` — no row
  at all, rather than an empty one, because they were never timed (§1.2b);
- `#2` and `#7`, filtered on analysis in LOG-00 and never implemented on any device (§1.2c);
- any per-kernel `unfused_kN_ms` the bench did not time alone;
- a re-measurement of the 24 cells whose two estimators disagree by >2 % (§3.2), and of the three
  families measured on the other card (§3.6). Neither was run; neither is estimated.

---

## 6. Honest comparison with the sibling reports

| | C500 | RTX 4060 | **H200** |
|---|---|---|---|
| regimes in the report | 7 | **5** — `decode_bs512`/`bs1024` absent, they were whole-layer regimes | **7** |
| per-regime CSV | 15 rows / **11 fusions** | **9 rows / 7 fusions** — #6, #8, #9, #11a impossible at spec | **16 rows / 11 fusions**; 8 have data at all 7 regimes, `#11a`/`#11b`/`#11b'` at **2 of 7** (§1.2a) |
| whole layer | measured, 2 passes | **impossible** — 18 GiB of experts vs ~7.4 GB usable; layer columns left empty | **measured, 2 passes** |
| layer CSV | genuine | **partly derived** — `ms_saved_per_layer` with a `basis` column, `layer_total_ms` empty | **genuine, every column filled** |
| timing protocol | sequential; one cell re-measured interleaved after an impossible result | sequential; one cell corrected the same way | **interleaved throughout**, but published as a ratio of medians (§3.1–3.2) |
| timer resolution | tick 0.256 us | tick 1.024 us; **decode rows are 9–17 ticks, +-8 %** | tick 0.032 us; **nothing tick-limited**, shortest arm 1626 ticks |
| one exclusive GPU | yes | yes (one GPU on the host) | **no** — co-tenant present for the last four families (§3.5), and 3 families from a different card (§3.6) |
| Triton vs vendor BLAS | **50 %** | 102 % | **94.4 %** |

None of the three is clean. C500 could not resolve sub-percent layer differences on a machine
that throws 25–320 % excursions, and says so. The 4060 could not fit the model and left columns
empty rather than model them, and says so. The H200 fits everything and measured everything it
could compile correctly — and its remaining problems are **statistical and provenance
problems**, not capacity problems: an estimator choice that moves 24 cells by more than 2 %, a
ceiling check that ran on 32 of 90 cells, a co-tenant across the last four families, and three
families — including both headline winners — from a different card.

The one claim this device does support cleanly and the others do not: **the whole-layer
combination sweep and all seven regimes ran at exact GLM-5.2 spec, with no substitution, no
shrunken expert count, and nothing derived for capacity reasons.**

---

## 7. Reproducing

```bash
# measurement (do NOT re-run to regenerate this report; results/h200/ is the raw record)
python3 glm52_h200/preflight.py --gpu 3          # writes glm52_h200/preflight_h200.json
python3 run_h200.py --gpu 3                      # serial, resumable, 3 h 22 min wall

# report (no CUDA, reads JSON only)
python3 glm52/make_report_h200.py                # the seven fusion_<regime>.csv
python3 glm52/make_layer_report_h200.py          # layer_optimal_per_regime.csv
```

`run_h200.py` refuses to start on a non-sm_90 device and on a tenanted card, pins
`CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER=PCI_BUS_ID` for every child, and re-checks tenancy
between families — which is how §3.5 came to be recorded rather than missed.
