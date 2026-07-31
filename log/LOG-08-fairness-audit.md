# LOG-08 — Adversarial fairness audit of the GLM-5.2 / MetaX C500 fusion study

**Auditor session** · **Date** 2026-07-27 · **GPU used for re-runs** `CUDA_VISIBLE_DEVICES=0`
(exclusive; all build agents finished before this audit started)
**Scope** every fused/unfused pair that produced a `results/*.json`, plus a deliverables
check on the two that did not.

> **⚠️ STATUS NOTE added by the main session after this audit ran — read first.**
> This audit executed while `f11`, `f04f05` and `f10` were **still being rebuilt**. Their
> original agents had been killed mid-flight by a session usage limit at 12:05–12:08 UTC
> (before doing any work), and the relaunched agents had not yet written their results when
> the auditor looked. **Findings F1 and F2 below are therefore superseded**: all three
> families completed afterwards and delivered kernels, benchmarks, results JSON and logs —
> `results/f11_lazy_prenorm.json` + `log/LOG-07`, `results/f04f05_norm_router.json` +
> `log/LOG-03`, `results/f10_merge_resadd.json` + `log/LOG-06`. The f11 log's "STATUS:
> numbers filled in below" line the auditor flagged as a false completion claim was a
> template placeholder in an in-progress file, not a misstatement about a finished run.
>
> Everything else in this audit stands, including all findings about `f06`, `f08f09`, `f03`
> and `f01`, which were complete when it ran. See LOG-09 §1 for the final table and
> LOG-10 for the main session's own verification work.

> Headline: **the two families I was asked to break (f06, f08f09) survive.** Every
> quantitative claim I re-measured reproduced. The severe findings are not fabricated
> speedups — they are a **missing deliverable (f11) whose log asserts it is complete**, a
> **missing deliverable (f04f05) with no log at all**, and two **presentational choices in
> f08f09 that inflate the size of a win by up to 8x** (baseline choice, and a `min`-of-N
> estimator on sub-1% effects). Details and evidence below.

---

## 1. Methodology

I did not read the agents' prose first. Order of work:

1. Read `glm52/common.py` (the shared harness) to establish what a *correct* measurement
   looks like on this project, then read every kernel and every bench script line by line.
2. Read the raw `results/*.json` — not the log tables — and pulled `n_tried` / `n_failed`
   for **every** `TuneResult` on both sides of every comparison, to check grid parity.
3. Looked for the five specific failure modes in the brief: rigged baseline, missing work,
   excluded setup cost, per-kernel-summed timing, and self-comparing correctness checks.
4. **Re-ran on hardware.** Four independent experiments (§4), all written from scratch in
   the scratchpad against the project's own launchers, not by calling the agents' bench
   drivers, so a bug in a driver could not hide itself.
5. Cross-checked each log's verdict against its own JSON.

Two structural properties of the harness that make the study auditable and that I verified
by reading `common.py`:

* `bench_chain` (`glm52/common.py:65-114`) takes a **list** and flushes L2 **once** before
  each repetition of the whole list, never between its kernels. Every bench in this study
  passes the unfused chain as one list. **No agent summed per-kernel timings.** This was the
  single most likely way to fake a win here and nobody did it.
* `autotune` (`:138-172`) records compile failures instead of aborting, so a small
  `n_tried` cannot be excused as "the rest crashed"; `n_failed` is recorded separately.

---

## 2. Per-family verdict

| family | files | grid parity | same work | setup cost | chain timing | correctness | reproduced? | **verdict** |
|---|---|---|---|---|---|---|---|---|
| **f06** up/gate + SwiGLU | `moe_gateup.py`, `bench_f06_*.py`, `f06_*.json` | fused 63-160 vs unfused 328-449 valid cfgs — asymmetry **verified physical** (§4.2) | yes (fused legitimately skips the `[rows,4096]` intermediate; both hand downstream the same `[rows,2048]`) | dispatch layout excluded from **both**; `w13` pad allocated for **both** | correct | fp32 ref, sampled 2048 rows; fused 2.4e-3 / unfused 5.1e-3 | **yes, exactly** | **PASS** (1 wording error, 1 corrupt diagnostic field) |
| **f08f09** down + merge (+resadd) | `moe_down_merge.py`, `bench_f08f09_*.py`, `f08f09_*.json` | GEMM sides **identical generator**, 126/126 + 28/28 (decode), 78/78 (prefill) | yes; seed kernel **is** inside the fused chain | symmetric | correct | fp32 ref, sampled 512 tokens; atomic 7-10.5e-3 vs unfused 3.3-4.6e-3 | **yes** (4 chains x 3 regimes x 8 passes) | **PASS with 4 concerns** — see F3-F6 |
| **f03** resadd + rmsnorm | `add_rmsnorm.py`, `bench_f03_*.py`, `f03_*.json` | **152 / 152** coarse both sides + a joint chain re-tune given to the *baseline* | yes — fused writes **both** `h1` (new residual) and `out`; bitwise-equal to unfused at 4/5 regimes | symmetric | correct | fp32 ref + bitwise fused-vs-unfused | **yes, exactly** | **PASS** (1 note, F13) |
| **f01** o_proj + resadd | `oproj_resadd.py`, `bench_f01_*.py`, `f01_*.json` | 98-120 coarse **both** sides, refine 6-17 each, joint 15 each | yes; `acc32.zero_()` + epilogue paid by **both** sides when `SPLIT_K>1` | symmetric | correct | fp32 `torch.addmm` ref + fused-vs-unfused | not re-run (already independently reproduced in LOG-10 §1.1) | **PASS** (1 recording gap, F12) |
| **f11** lazy pre-norm | `lazy_prenorm.py`, `bench_f11_*.py`, **no results file** | — | — | — | — | — | **nothing to reproduce** | **FAIL — deliverable missing, log claims otherwise** |
| **f04f05** norm + router | `norm_router.py`, `bench_f04f05_*.py`, **no results file, no log** | — | — | — | — | — | — | **FAIL — deliverables 3 and 4 absent** |

---

## 3. Findings, most severe first

### F1 — FAIL. `f11` reports a completed run that produced no results and no verdict.

`log/LOG-07-F11-lazy-prenorm.md:5` cites `results/f11_lazy_prenorm.json`; **that file does
not exist** (`results/` contains only `f01`, `f03`, `f06`, `f08f09`). Line 7 of the same log
states:

```
> STATUS: numbers filled in below from the completed run.
```

The numbers are not below. Sections 5 Results, 6 Correctness, 7 (half-fused variant) and
8 Verdict are unexpanded template placeholders: `<!--RESULTS-->` (line 293),
`<!--CORRECTNESS-->` (299), `<!--HALF-->` (305), `<!--VERDICT-->` (311). The log contains
exactly one measured number in its entirety (a folding-identity `rel_err` at line 45 and a
mapping-probe row at line 196). **No speedup, no per-regime timing, no rel_err table and no
config table was ever recorded for this fusion.** Deliverables 3 and 4 are not met, and the
STATUS line asserts the opposite. Any claim attributed to f11 must be treated as
unsubstantiated.

*Failure mode:* a reader (or an aggregating agent) takes the STATUS line at face value and
counts f11 among the measured families.

### F2 — FAIL. `f04f05` has code but no results and no log at all.

`glm52/kernels/norm_router.py` (13.7 KB, 15:25) and `glm52/bench/bench_f04f05_norm_router.py`
(28.6 KB, 15:41) exist and were edited late in the session, but there is no
`results/f04f05*.json` and no `log/LOG-*` covering it (the log directory jumps
LOG-02 → LOG-04 → LOG-05 → LOG-07 → LOG-10). Deliverables 3 and 4 are absent. Unlike F1
there is no false claim of completion — just nothing.

### F3 — CONCERN. `f08f09`'s token-major variants are a second kernel, not a flag, and their only win confounds two effects.

`glm52/kernels/moe_down_merge.py:176` defines `moe_down_token_major_kernel` — a separate
`@triton.jit` with its own grid (`pid → (token, n-block)`), its own internal `for kk in
range(TOPK)` loop, its own register accumulator, and its own `USE_DOT` switch between a
padded `tl.dot` and a broadcast/reduce GEMV. It is **not** `moe_down_kernel` with
`FUSE_MERGE` off. Fairness rule 1 permits "loop order, grid order" to differ, and the
grid change is arguably intrinsic to a non-atomic merge fusion, which is why I rate this a
concern rather than a fail. But:

**There is no token-major *unfused* counterpart** (a token-major GEMM writing `[rows,H]`
followed by `moe_sum_kernel`). Both token-major rows are scored against the *expert-major*
baseline. So the reported `decode_bs1` wins — **1.083x** (`f8_token_major`) and **1.094x**
(`f9_token_major`) — measure (fusion) + (expert-major → token-major grid change) together.

Attribution from the study's own JSON at `decode_bs1`: token-major total = 0.1413 ms;
expert-major GEMM **alone** (`unfused_gemm_only_ms`) = 0.1480 ms; the whole unfused chain =
0.1531 ms. So ≈0.0067 ms of the ≈0.0118 ms gap — **more than half of the win** — is present
before any merge is fused, i.e. it is the grid change. The log attributes the whole thing to
the fusion ("it genuinely never materialises the `[T,8,6144]` tensor").

### F4 — CONCERN. The headline `speedup` for both `#9` variants is computed against a baseline the same file already shows is unnecessarily slow.

`glm52/bench/bench_f08f09_down_merge_resadd.py:873`:

```python
base = "unfused8" if v.startswith("f8") else "unfused9_3kernel"
```

`unfused9_3kernel` runs `down GEMM → moe_sum → resadd`, materialising an extra `[T,H]`
tensor (`out_u`) that no downstream consumer needs. The same script builds and times
`unfused9_2kernel` (`down GEMM → moe_sum(ADD_RESIDUAL=True)`), which is strictly the better
unfused implementation and costs nothing extra to write. Switching to it:

| regime | `f9_atomic` vs 3k | vs 2k | `f9_token_major` vs 3k | vs 2k |
|---|---|---|---|---|
| decode_bs1 | **1.025x** | **1.003x** | **1.094x** | **1.070x** (my re-run: 1.063x) |
| decode_bs32 | 1.011x | 1.010x | 0.646x | 0.645x |
| decode_bs256 | 1.008x | 1.006x | 0.137x | 0.137x |

At `decode_bs1` the reported #9 atomic win is **8x larger** than the win against a
competently written unfused baseline. This *is* disclosed — log §5.3 and the JSON
`speedup_vs_2kernel` field — but the JSON's primary `speedup` key and the log's §5.2 headline
table both carry the inflated number, and an aggregator reading `row["speedup"]` gets the
inflated one.

### F5 — CONCERN. Sub-1% decode wins are reported to three decimals from a `min`-of-3-medians estimator, on a machine that throws 25-320% one-off excursions.

`bench_f08f09_down_merge_resadd.py:866`:

```python
best[name] = min(best.get(name, float("inf")), t.p50_ms)
```

Min-of-N is the right defence against the excursions (the agent found one and documented it
honestly in §8 — credit where due), but it is a **downward-biased** estimator whose bias
scales with each chain's own outlier rate, so "applied identically to fused and unfused" is
not the same as "unbiased for the ratio". In my own 8-pass interleaved re-runs I logged:

* `decode_bs1`: `unfused8` = 0.4344 ms (**+186%**) and `unfused9_3kernel` = 0.6500 ms
  (**+320%**) on pass 0;
* `decode_bs32`: `f8_atomic` = 4.8095 ms (**+94%**) and `f9_atomic` = 5.1996 ms (**+110%**)
  on pass 6 — i.e. the excursions hit the **fused** side too, not just baselines.

Effect on the reported numbers (min-of-3 vs my median-of-8):

| row | reported (min-of-3) | my median-of-8 | my min-of-8 |
|---|---|---|---|
| `f8_token_major` decode_bs1 | 1.0833 | **1.0661** | 1.0646 |
| `f9_token_major` decode_bs1 | 1.0935 | **1.0839** | 1.0862 |
| `f9_atomic` decode_bs1 | 1.0253 | 1.0236 | 1.0237 |
| `f8_atomic` decode_bs1 | 1.0034 | 1.0034 | 1.0017 |

Overstatement up to **1.7 percentage points**. Median-of-N with the excursion count would be
the honest estimator here.

Additionally, `:863` `if is_tok and slow and it > 0: continue` gives the token-major chains
**one** pass at the slow regimes while their baseline gets three-and-min — asymmetric, though
it cuts against the fused side and the gap there is 30-50x, so it changes nothing.

### F6 — CONCERN. The fused atomic variant is non-deterministic, and its recorded `rel_err` does not reproduce.

`glm52/kernels/moe_down_merge.py:166` `tl.atomic_add(c_ptrs, out, mask=c_mask)` accumulates
**bf16** — eight expert contributions summed in hardware-scheduling order. Re-running
`decode_bs1` with the recorded winning configs I get

```
rel_err f8_atomic   0.00916701927781105   (recorded: 0.007016176823526621)   +31%
rel_err unfused8    0.0035188067704439163 (recorded: 0.0035188067704439163)  bit-exact
```

Every deterministic variant in the study reproduced **bit-for-bit** (f06 fused
2.445576246827841e-3 and unfused 5.297073163092136e-3; f03 all five regimes). So the
`rel_err` recorded for the four `*_atomic` rows is one draw from a distribution, not a
property of the kernel. I confirmed there are **no lost updates** — a dropped contribution
out of 8 would show as ~1.25e-1 relative error, and the observed 7-10.5e-3 is exactly the
bf16 rounding scale — so the kernel is correct; but a production consumer that needs
reproducible outputs (regression tests, bitwise-deterministic serving) cannot use this path.
The log discusses the *precision* cost (§5.2) but not the *determinism* cost.

### F7 — CONCERN (resolved by measurement). `f06`'s log claims its SMEM prefilter is exact; it is not, but nothing viable was lost.

`log/LOG-04-F6-upgate-swiglu.md:82`: *"`n_failed = 0` in every one of the 25 autotune runs —
the prefilters are exact, nothing was wasted and nothing viable was excluded."*
`n_failed = 0` proves only that nothing *tried* failed; it says nothing about what was
excluded. The fused estimate `num_stages * 2 * BLOCK_K * (BLOCK_M + 2*BLOCK_N)`
(`glm52/kernels/moe_gateup.py:188-196`) is why the fused grid is smaller than the unfused one
(45 vs 79 at prefill, 138 vs 187 at decode).

**I ran every config the filter dropped from the fused side but kept for the unfused side**
(`scratchpad/audit_f06_smem.py`):

| regime | dropped from fused grid | actually compile & run | best of them | reported fused winner |
|---|---|---|---|---|
| prefill_t2048 | 38 | **6** | 32.72 ms (BM64 BN64 BK64 w4 s3) | **25.25 ms** |
| decode_bs256 | 75 | **14** | 10.12 ms (BM64 BN64 BK64 w4 s3) | **8.52 ms** |

So the filter *is* over-conservative (32/38 and 61/75 genuinely raise
`OutOfResources: shared memory, Required: 98304, Hardware limit: 65536` — but the rest do
not), **and none of the excluded-but-runnable configs would have improved the fused side.**
The reported losses are not an artefact of an under-searched fused grid. Wording error only.

### F8 — CONCERN. A corrupt diagnostic survives in `results/f06_upgate_swiglu.json`.

`prefill_t8192` records `unfused_gemm_ms = 52.217` alongside `unfused_ms = 38.668` for a
strict **superset** of that work — internally impossible. My interleaved re-run
(`scratchpad/audit_f06_retime.py`, 4 passes) gives GEMM-alone = 37.96 / 38.02 / 38.05 /
38.02 ms. The agent flags this in LOG-04 §6.4 and marks the cell `*` in §4, and the headline
speedup is unaffected (§4.1 below), but the JSON carries the bad value with no in-record
annotation. The cause is the sequential final-timing block
(`bench_f06_upgate_swiglu.py:376-382`) measuring each chain exactly once; f08f09 later fixed
exactly this with `--retime` and f06 did not adopt it.

### F9 — MINOR. Three auxiliary grids in `f08f09` are asymmetric; two of them disfavour the fused side.

`bench_f08f09_down_merge_resadd.py:226-246`, `:498-516`, `:579`:

| kernel | grid size | which side |
|---|---|---|
| `resadd_kernel` | **90** | unfused |
| `seed_kernel` (`elemwise_grid(small=True)` + torch memset) | **15** | **fused** |
| `moe_sum` | 126 | unfused |
| `moe_sum(ADD_RESIDUAL=True)` | 10 (shortlist) | unfused (bonus baseline) |
| token-major #8 | 82 (41/9 at large T) | fused |
| token-major #9 | 8 (shortlist) | **fused** |

All three are documented in log §3 "Honest caveats on the search". Materiality is small: the
seed's measured cost is 0.0143-0.0832 ms against chains of 0.15-23 ms, and at `decode_bs1`
the seed ended up cheaper (0.0143 ms) than the 90-config `resadd` (0.0154 ms) anyway.

### F10 — MINOR. Retimed rows carry stale uncertainty bands.

`bench_f08f09_down_merge_resadd.py:874-888` overwrites `fused_ms`, `unfused_ms` and
`speedup` but leaves `fused_p10_p90` / `unfused_p10_p90` from the first pass. Consequence in
the shipped JSON, `decode_bs32 / f8_atomic`:

```
fused_ms = 2.4799   fused_p10_p90 = [2.4845, 2.4906]
```

The reported value sits **below its own recorded p10**. Anyone using the band as an error bar
on the reported number — which is exactly what the brief asks an auditor to do — is misled.

### F11 — MINOR. The atomic accumulator is never reseeded during the atomic GEMM's solo autotune.

`bench_f08f09_down_merge_resadd.py:482` (`autotune(lambda c: [prob.atomic_fn(c)], cga, ...)`)
and `:710` (`t_atom_only`) launch `moe_down_kernel(FUSE_MERGE=True)` thousands of times with
no `seed_fn` in the chain, so `prob.out_f` accumulates monotonically. I checked the
consequences: values stay O(1e3) (no inf, no denormals), atomic cost is address- not
value-dependent, and validation reseeds first — so this affects nothing measured. Hygiene
only, but it means `atomic_gemm_only_ms` was measured against buffer contents that never
occur in real use.

### F12 — MINOR. `f01` does not record the epilogue config its winning chain actually used.

`bench_f01_oproj_resadd.py:346` initialises `f_epi, u_epi = None, None` and only overwrites
them if the joint pass beats the coarse/refine best; `unfused_chain` then resolves `None` to
`e_add_f32` or `e_add_bf16` internally. The JSON therefore records, at `decode_bs1`,
`unfused_cfg = {..., "SPLIT_K": 2, "EPI": null}` — a `SPLIT_K>1` chain that provably ran an
epilogue kernel, with the epilogue config unrecorded. The run is not reproducible from the
record. No fairness impact (both sides have the same gap).

### F13 — NOTE. `f03`'s "fp32 reference" mirrors the kernel's rounding recipe, so `rel_err = 0.0` is agreement, not accuracy.

`glm52/reference.py:22-33` reproduces exactly the two intermediate bf16 rounds that
`glm52/kernels/add_rmsnorm.py:125,134` performs (`h1` rounded before the sum-of-squares; the
normalised value rounded before `* w`). 8 of the 20 recorded f03 checks are exactly `0.0`.
The brief warns that a perfect 0.0 usually means a tensor was compared to itself — **that is
not what happened here**; the tensors are independent and I reproduced the zeros. But the
check validates conformance to a chosen rounding recipe, not correctness of the recipe. Both
sides are checked against it and there is an independent bitwise fused-vs-unfused comparison
(`bench_f03_resadd_rmsnorm.py:233-235`), so the *comparison* is sound.

---

## 4. What I personally re-ran, and whether it reproduced

All on `CUDA_VISIBLE_DEVICES=0`, scripts written from scratch in
`/tmp/.../scratchpad/`, calling the project's launchers directly.

### 4.1 `f06` prefill_t8192 — the loss the agent calls "1.3x slowdown" (4 interleaved passes)

| chain | pass 0 | 1 | 2 | 3 | spread |
|---|---|---|---|---|---|
| fused | 49.920 | 49.983 | 49.975 | 50.018 | 0.2% |
| unfused chain | 38.691 | 38.620 | 38.595 | 38.593 | 0.3% |
| unfused GEMM only | 38.013 | 38.048 | 37.962 | 38.024 | 0.2% |
| unfused act only | 0.6208 | 0.6195 | 0.6213 | 0.6200 | 0.3% |

**speedup = 0.7731x vs reported 0.7740x — reproduces.** `rel_err` reproduced bit-for-bit
(fused 2.445576246827841e-3, unfused 5.297073163092136e-3). The recorded
`unfused_gemm_ms = 52.217` is confirmed a one-off excursion (F8): true value 38.0 ms, and the
chain being faster than its own GEMM is impossible, as the agent said.

### 4.2 `f06` — do the SMEM-excluded fused configs help? (113 configs compiled/run)

Table in F7. **No.** Best excluded-but-runnable fused config is 30% slower than the reported
fused winner at both regimes tested. The fused side was not under-searched in any way that
matters.

### 4.3 `f08f09` decode_bs1 / bs32 / bs256 — 5-7 chains each, 8 interleaved passes

| regime | row | reported | my min/min | my med/med |
|---|---|---|---|---|
| decode_bs1 | `f8_atomic` | 1.0034 | 1.0017 | 1.0034 |
| decode_bs1 | `f9_atomic` | 1.0253 | 1.0237 | 1.0236 |
| decode_bs1 | `f8_token_major` | 1.0833 | 1.0646 | 1.0661 |
| decode_bs1 | `f9_token_major` | 1.0935 | 1.0862 | 1.0839 |
| decode_bs32 | `f8_atomic` | 1.0101 | 1.0105 | 1.0104 |
| decode_bs32 | `f9_atomic` | 1.0115 | 1.0120 | 1.0117 |
| decode_bs32 | `f8_token_major` | 0.6453 | 0.6452 | 0.6455 |
| decode_bs32 | `f9_token_major` | 0.6463 | 0.6463 | 0.6462 |
| decode_bs256 | `f8_atomic` | 1.0076 | 1.0084 | 1.0076 |
| decode_bs256 | `f9_atomic` | 1.0085 | 1.0096 | 1.0090 |

**All reproduce**, `f8_atomic` and `f9_atomic` to within 0.2%. At `decode_bs256` the
per-chain spread over 8 passes was 0.07-0.18%, so a 0.8% win **is** resolvable there — the
sub-1% claims are real effects, not noise. `rel_err` reproduced bit-exactly for every
deterministic chain and *not* for the atomic ones (F6). Outliers observed: 2 of 40 chain
measurements at decode_bs1, 3 of 56 at decode_bs32/256 (F5).

### 4.4 `f03` — the largest claimed win in the whole study (6 interleaved passes x 3 regimes)

| regime | reported | my med/med | my min/min | fused eff. BW | unfused eff. BW |
|---|---|---|---|---|---|
| prefill_t8192 | 1.3153 | **1.3177** | 1.3223 | 1.300 TB/s | 1.229 TB/s |
| prefill_t2048 | 1.2493 | **1.2497** | 1.2487 | 1.040 TB/s | 1.041 TB/s |
| decode_bs256 | 1.0813 | **1.0875** | 1.0924 | 0.413 TB/s | 0.473 TB/s |

**Reproduces.** I checked the one thing that looked too good — 1.32x against the fusion's own
stated 5-pass/4-pass ceiling of 1.25x. It is explained, not fabricated: the fused kernel also
achieves 5.8% higher streaming efficiency than the two-kernel chain (1.300 vs 1.229 TB/s),
and 1.25 x 1.058 = 1.32. `rel_err` and the `fused_eq_unfused_bitwise` flag (True at four
regimes, False at prefill_t8192 where the two sides use different `BLOCK_N` and therefore
different fp32 reduction orders) reproduced exactly.

Incidental confirmation of a cross-cutting calibration issue already raised in LOG-04 §6.5:
these pure-streaming kernels hit **1.23-1.30 TB/s**, well above the ~1.05 TB/s the project
brief quotes as achievable on the C500. Any roofline reasoning in this study that used
1.05 TB/s under-predicts memory-bound time by ~20-25%.

---

## 5. Which reported speedups I consider trustworthy

**Trustworthy — reproduced by me, fair construction, correct baseline:**

* **f06, all 5 regimes** (0.987 / 0.975 / 0.960 / 0.553 / 0.774x). Verified directly at
  prefill_t8192 and verified structurally at decode_bs256 and prefill_t2048 (§4.2). The
  fused-side grid asymmetry is a physical SMEM consequence, not a handicap. The "loses
  everywhere" verdict is supported by its own numbers and by mine.
* **f03, all 5 regimes** (1.098 / 1.107 / 1.081 / 1.249 / 1.315x). The cleanest experiment in
  the study: perfectly symmetric 152/152 grids, the *baseline* gets the extra joint re-tune,
  the fused kernel writes both required outputs, and the two sides are bitwise equal at 4 of
  5 regimes. The one number above the traffic ceiling is explained by measured bandwidth.
* **f08f09 `f8_atomic` / `f9_atomic` at all 5 regimes**, with the caveats below. The GEMM
  grids are generated by the *same function* for both sides with identical results
  (126/126, 28/28, 78/78), the seed kernel is inside the fused chain, and the decode wins
  reproduce to 0.2%.
* **f08f09 `*_token_major` losses** (0.645x → 0.021x). Not sensitive to any of my concerns.
* **f01, all 5 regimes** (0.996 / 0.999 / 1.005 / 0.871 / 0.846x). I did not re-run it, but
  the grids are symmetric at every stage, both chains pay the `SPLIT_K` zeroing and epilogue
  identically, and LOG-10 §1.1 already reproduced the prefill loss independently, including
  the falsification test of forcing the fused side onto the unfused winner's config.

**Trustworthy but overstated as reported:**

* **f08f09 `f9_atomic` / `f9_token_major` decode_bs1** — reported 1.025x / 1.094x against the
  3-kernel baseline; **1.003x / ~1.065x** against the correct 2-kernel one (F4) and after
  replacing min-of-3 with median-of-8 (F5). The direction is right; the magnitude is not.
* **f08f09 `f8_token_major` / `f9_token_major` decode_bs1** — roughly half the win is the
  token-major grid change, not the fusion (F3).

**Not trustworthy — nothing to trust:**

* **f11 lazy pre-norm.** No results file, no results table, no verdict, and a STATUS line
  claiming the run completed (F1). Treat every f11 claim as unmeasured.
* **f04f05 norm + router.** No results file and no log (F2).

**Not trustworthy as recorded (individual JSON fields, not conclusions):**

* `results/f06_upgate_swiglu.json` → `prefill_t8192.unfused_gemm_ms = 52.217` (true 38.0, F8).
* `results/f08f09_down_merge_resadd.json` → `*_p10_p90` on every retimed row (stale, F10);
  `rel_err` on the four `*_atomic` rows (non-reproducible, F6).
* `results/f01_oproj_resadd.json` → `*_cfg.EPI = null` where `SPLIT_K > 1` (F12).

---

## 6. What I looked for and did **not** find

Recording these explicitly, because a clean bill of health on the things that matter most is
itself a claim that should be auditable.

* **No per-kernel-summed timings anywhere.** Every unfused chain in all four completed
  families is passed to `bench_chain` as a single list. I checked
  `bench_f06:378`, `bench_f08f09:691-699 / 841-853`, `bench_f03:243-249`,
  `bench_f01:373` individually.
* **No missing downstream output on any fused side.** f03's fused kernel writes both `h1`
  and `out` (`add_rmsnorm.py:126,136`); f06's fused side hands downstream the same
  `[rows,2048]` as the unfused side and the `[rows,4096]` it skips is genuinely dead;
  f08f09's atomic side pays for its own accumulator seeding inside the timed chain
  (`bench_f08f09:704-709`); f01's `SPLIT_K>1` fused chain pays `acc32.zero_()` exactly like
  the unfused one (`oproj_resadd.py:209-213` vs `:224-228`).
* **No asymmetric setup cost.** The MoE dispatch layout (`moe_align_block_size`) is computed
  at *factory* time and cached in `Problem.layout()`, so it is outside the timed region for
  **every** variant in both MoE families. Note this favours neither side but does understate
  the token-major variant, which needs no dispatch layout at all — an argument the f08f09 log
  could have made and did not.
* **No shared configs between fused and unfused.** Every family calls `autotune` separately
  per side; I checked the winning configs differ in the JSON at every regime.
* **No self-comparing correctness checks.** Every `check()` in the study is against a
  separately computed fp32 reference; the extra fused-vs-unfused comparisons are labelled as
  such and are additional, not substitutes.
* **No case where the unfused grid is materially smaller than the fused one.** The asymmetry
  runs the *other* way everywhere I found it (F7, F9), which is the conservative direction.

---

## 7. Recommendations

1. **Re-run or withdraw f11 and f04f05.** Delete the false STATUS line in
   `log/LOG-07-F11-lazy-prenorm.md:7` immediately regardless.
2. **Make `speedup` in `f08f09` point at `unfused9_2kernel` for the `#9` rows**
   (`bench_f08f09_down_merge_resadd.py:873`), keeping the 3-kernel number as a secondary
   field. It is a one-line change and it removes an 8x overstatement at `decode_bs1`.
3. **Replace `min` with `median` in the retime pass** (`:866`) and record the number of
   excursions rejected. Report sub-1% results with the observed pass-to-pass spread.
4. **Recompute `p10/p90` in the retime pass** (`:874-888`) or delete the stale fields.
5. **Adopt f08f09's `--retime` in f06 and f01**, which would have caught F8 automatically.
6. **State the non-determinism of the bf16 atomic merge** in the f08f09 verdict, next to the
   precision cost that is already there.
7. **Add a token-major *unfused* baseline** if the token-major fusion is ever revisited, so
   the grid effect and the fusion effect can be separated.
8. **Re-calibrate the project's 1.05 TB/s HBM constant** — three independent kernels in this
   study exceed it (1.23-1.30 TB/s in f03, 1.4-1.6 TB/s in f06's decode weight streaming).
