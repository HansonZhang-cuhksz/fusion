# Kernel fusion in a GLM-5.2 MoE decoder layer on MetaX C500 — concluding report

**Model** GLM-5.2 (`zai-org/GLM-5.2`, `glm_moe_dsa`) · **Hardware** MetaX C500 (104 CUs, 64 KB
SMEM/CTA, 131072 regs/SM, 8 MB L2, warp = 64 lanes)
**Stack** MACA 3.7.0.38, `torch 2.8.0+metax`, `triton 3.0.0+metax` · **dtype** bf16 / fp32 accumulate
**Sources** `log/LOG-00` … `LOG-12`, `results/*.json`, `REPORT-lazy-prenorm.md`
**Regenerate the master table** `python -m glm52.consolidate`

---

## 0. Headline

Eleven fusions were proposed, **2 filtered on analysis**, **9 built, tuned and measured**
against independently-autotuned unfused counterparts cut from the same source.

**Only the memory-bound vector fusions pay. Every attempt to fuse work *into* a GEMM either
did nothing or actively hurt.** Three fusions are worth shipping; they are collectively worth
**0.24 %–0.46 % of layer time**. The single most valuable operational finding is negative:
**fusing everything is worse than fusing nothing at every regime**, by up to **48 %**.

The study's real conclusion is that fusion is the wrong lever on this machine. The layer's
cost is three large GEMMs, and the Triton GEMM runs at **107 TF/s against the vendor BLAS's
215** — closing that gap is worth ~30 ms on a ~76 ms layer, two orders of magnitude more than
every fusion here combined.

---

## 1. Assumptions

Everything below is load-bearing. The first four affect whether the numbers mean what they
appear to mean.

**A1 — Triton, not production kernels.** The dense Triton GEMM reaches ~107 TF/s vs the vendor
BLAS's ~215. **Fused-vs-unfused *ratios* are the deliverable; absolute times are not
comparable to a production engine.** The gap is not uniform: for the *grouped* MoE GEMM the
unfused Triton kernel reaches 0.93× the vendor path at prefill and **2.2×** at decode_bs256
(where the vendor route pays 256 kernel launches). Conclusions about fusion should port;
absolute times should not.

**A2 — Measured scope is a subtotal.** S3–S11 plus the shared expert. **Attention core, the
MLA projections and the DSA indexer are excluded.** No fusion candidate touches them, so the
*ranking* is unaffected, but every percentage is relative to a subtotal — as a fraction of a
full layer every gain is smaller than stated.

**A3 — Fair-comparison protocol.** Both sides of every pair are independently autotuned over
audited grids (`n_tried`/`n_failed` recorded per side). `bench_chain` (`glm52/common.py:65`)
flushes L2 **once before each repetition of the whole list**, never between kernels inside it,
and the unfused chain is always passed as one list. **No per-kernel timings were summed** —
the most likely way to fake a win here, and the audit (LOG-08) confirms nobody did it.

**A4 — No hardware performance counters were obtained.** Attempts via `mcProfiler` (Flask
client/server with encrypted `.pcd` configs; three `perf_exec` runs produced correct schemas
with **zero rows**) and inspection of MCPTI (present, a faithful CUPTI 1.x clone with the
right metric catalogue, but needs a C harness that was not built) both failed.
`torch.profiler` works but exposes only timings. **All traffic figures are analytic; all
mechanism attributions are inferred from timing and compiler output, not counter-measured.**
Detail in `REPORT-lazy-prenorm.md` §6.

**A5 — Noise floor.** This is a shared machine with documented **25–320 % one-off timing
excursions**. Sub-1 % effects are at or below what it can resolve; the whole-layer results in
§4 were therefore re-measured **twice independently, 8 interleaved rounds per run**, and a
winner is declared only when its gap to the runner-up exceeds the round-to-round spread of
both. Where that test fails, configurations are reported as **tied**, not ranked.

**A6 — Routing is frozen** after being computed once from the pipeline's own `h1`, so every
configuration routes identical tokens to identical experts. The router GEMM and top-k still
run and are still timed; they write to scratch.

**A7 — Dispatch-layout construction is outside the timed region**, identically in every
configuration. Our `moe_align_block_size` is a torch reference with a Python loop over 256
experts; production uses a fused kernel at ~10–30 µs.

**A8 — Decode `T` = batch size** (one token per sequence).

**A9 — Weight folds are offline.** The `#11` family folds the RMSNorm affine weight into the
consumer's weight matrix before any timed region. Legitimate at inference; **assumes `B` is
constant** — a LoRA adapter or dynamically-quantized gate would push the fold into the hot
path.

**A10 — Decode MoE has no headroom by construction.** At T ≥ 32 the layer streams 4–13 GB of
expert weights, so every MoE-GEMM fusion has a **1.00–1.01× ceiling** there. A large decode
"win" in this study would be launch overhead or an artifact, not fusion.

**A11 — Two atomic variants are non-deterministic** (bf16 `tl.atomic_add`, 8 contributions in
hardware-scheduling order). Correct — no lost updates, errors at the bf16 rounding scale — but
unusable where bitwise reproducibility is required.

**A12 — Roofline ceilings** come from `glm52/traffic.py`, a latency-aware model calibrated to
this device. "% of ceiling" figures inherit its assumptions.

---

## 2. The candidates

| | count | which |
|---|---|---|
| proposed | 11 | #1 … #11 |
| **filtered on analysis** | **2** | **#2**, **#7** |
| built, tuned, measured | 9 | #1, #3, #4, #5, #6, #8, #9, #10, #11 (a & b) |

**#2** o_proj+ResAdd+RMSNorm — three independent blockers: `tile_n = N` needs N ≤ 512 even on a
228 KB-SMEM B200 and our N is 6144 against C500's 64 KB; the Multi-CTA alternative needs
Hopper/Blackwell clusters + DSMEM; the atomic two-pass fallback is provably traffic-neutral.

**#7** Up_Gate+Act+Down — a 393 KB fp32 accumulator against 64 KB SMEM, *or* 24–96× activation
recompute, *or* 12.9 GB of fp32 atomics against the 268 MB it saves. **Every margin > 10×.**

---

## 3. Gain–regime table for every fusion

Speedup = unfused / fused, so **> 1 means the fusion helps**. Bold = the regime that decides
the verdict.

| # | fusion | kind | decode bs1 | bs32 | bs256 | prefill 2048 | prefill 8192 | % of ceiling | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **3** | ResAdd + RMSNorm | vector | 1.098× | 1.107× | 1.081× | **1.249×** | **1.315×** | 87–100 % | ✅ **SHIP** |
| **10** | Expert Merge + ResAdd | vector | **1.175×** | **1.181×** | **1.147×** | **1.204×** | **1.198×** | 87–100 % | ✅ **SHIP** |
| **11b′** | *half-fused* pre-norm → router | epilogue scale | 1.034× | 1.030× | 1.100× | **1.184×** | **1.239×** | 95–100 % | ⚠️ **standalone win, worth ~0 in-layer** (§4.3) |
| **11b** | Lazy Pre-Norm → router GEMM | in-mainloop | 0.684× | 0.680× | 0.771× | 1.096× | 1.127× | — | ⚠️ prefill-only; beaten by 11b′ |
| **8** | Down + Merge (atomic) | GEMM epilogue | 1.003× | 1.010× | 1.008× | 0.904× | 0.870× | — | ➖ neutral → harmful |
| **9** | Down + Merge + ResAdd2 (atomic) | GEMM epilogue | 1.003×¹ | 1.010× | 1.006× | 0.908× | 0.874× | — | ➖ neutral → harmful |
| **1** | o_proj + ResAdd | GEMM epilogue | 0.996× | 0.999× | 1.005× | 0.871× | **0.846×** | — | ❌ **harmful** |
| **6** | Up_Gate + SwiGLU | GEMM epilogue | 0.987× | 0.975× | 0.960× | **0.553×** | 0.774× | — | ❌ **harmful** |
| **11a** | Lazy Pre-Norm → w13 GEMM | in-mainloop | 0.902× | 0.961× | 0.965× | **0.476×** | 0.603× | — | ❌ **harmful** |
| **5** | RMSNorm + Router | norm-shaped | 0.473× | 0.464× | 0.467× | 0.475× | 0.680× | — | ❌ **harmful** |
| **4** | ResAdd + RMSNorm + Router | norm-shaped | 0.391× | 0.379× | 0.388× | 0.437× | 0.669× | — | ❌ **harmful** |
| 8/9 | token-major merge variant | — | 1.083× | 0.645× | 0.137× | 0.032× | 0.021× | — | ❌ catastrophic |
| 4/5 | + `FUSE_TOPK` | — | 0.210× | 0.207× | 0.227× | 0.418× | 0.534× | — | ❌ worse than plain |

¹ against the *2-kernel* unfused baseline; the family's own headline used a 3-kernel baseline,
inflating this cell to 1.025× (LOG-08 §F4).

### 3.1 #11b at fine resolution — 31 points, T = 1 … 262144

The only fusion swept densely enough to locate its crossover (`results/f11b_*_T*.json`):

| T | 1 | 32 | 512 | 1024 | 2048 | **4096** | 8192 | 16384 | 65536 | 262144 |
|---|---|---|---|---|---|---|---|---|---|---|
| gain | 0.652× | 0.632× | 0.856× | 0.761× | 0.951× | **1.026×** | 1.188× | 1.354× | **1.411×** | 1.395× |

Crossover at **T ≈ 4096**, plateau at ~1.39–1.41×. Full architectural analysis of this kernel
— traffic, instruction mix, occupancy, and why 0.78 % more arithmetic costs 12.3 % of MMA
throughput — is in **`REPORT-lazy-prenorm.md`** and `log/LOG-12`.

**A known inconsistency, stated rather than reconciled:** this sweep and the LOG-09 table
disagree about #11b, and at `t2048` they **disagree on the sign** (LOG-09 1.096× vs sweep
0.951×). They agree that decode loses and t8192 wins. The disagreement sits exactly in the
crossover region where the effect is smallest, and the two came from different sessions with
different tuning budgets. Neither has been shown wrong; the sweep is the more finely resolved.

---

## 4. The optimal fusion strategy, per regime

Measured end-to-end, not composed from the per-fusion table (§4.3 explains why that
distinction is essential). Layer time = the S3–S11 subtotal of A2.

| regime | **optimal plan** | layer time | vs all-unfused | confidence |
|---|---|---|---|---|
| **decode_bs1** | **#3 + #10** | 0.8564 ms | 1.0046× | borderline — one of two runs ties it with doing nothing |
| **decode_bs32** | **#3 + #8** | 8.4361 ms | 1.0037× | tied with `#3+#9`, `#8` |
| **decode_bs256** | **#8** or **#3 + #8** | 13.9625 ms | 1.0037× | tied with `#3+#8`, `#3+#9`, `#9` |
| **decode_bs512** | **#3 + #10** | 16.4796 ms | 1.0015× | tied with `#10`, `#1` |
| **decode_bs1024** | **#3 + #10** | 21.5194 ms | 1.0024× | tied with `#10`, `#1` |
| **prefill_t2048** | **#3 + #10** | 27.8321 ms | 1.0024× | solid, both runs |
| **prefill_t8192** | **#3 + #10** | 80.6149 ms | 1.0039× | solid, both runs |

### 4.1 As one decision rule

```
ALWAYS   enable #3  (ResAdd + RMSNorm)          — wins at every regime, 87-100% of ceiling
DOWN-PROJECTION AXIS:
    T <= 256   enable #8   (atomic down + merge)
    T >= 512   enable #10  (merge + ResAdd2)
NEVER    enable #6, #4, #5, #11a                — harmful at every regime
NEVER    enable #1 at prefill                   — 0.846-0.871x
NEVER    fuse greedily                          — see 4.2
DO NOT   include #11b/#11b' in a layer plan     — see 4.3
```

**The down-projection axis is the one place branching genuinely matters**, and it is a real
crossover, not noise:

| down-axis choice | bs32 | bs256 | **bs512** | **bs1024** | t2048 | t8192 |
|---|---|---|---|---|---|---|
| **#8** atomic down+merge | **1.0035×** | **1.0037×** | 0.9955× | 0.9749× | 0.9430× | 0.9661× |
| **#10** merge + ResAdd2 | 1.0005× | 1.0010× | **1.0011×** | **1.0018×** | **1.0018×** | **1.0027×** |

`#8` costs **5.7 %** at prefill_t2048; `#10` is the reverse at small decode. A single static
plan is wrong for at least one regime. *(The predicted crossover — where `#8`'s `[T,6144]`
bf16 accumulator stops fitting in 8 MB L2, T ≈ 683 — is **wrong**: `#8` already loses at
T=512 where the accumulator is 6.3 MB and fits. The L2 argument gets the direction right and
the threshold wrong; atomic contention is the likelier driver. Recorded as a failed
prediction rather than quietly dropped.)*

### 4.2 Greedy fusion is always wrong — the largest effect in the study

`#1 + #6 + #9` together:

| regime | all-unfused | greedy-all | penalty |
|---|---|---|---|
| decode_bs1 | 0.859 ms | 0.910 | **+6 %** |
| decode_bs32 | 8.460 | 8.593 | +2 % |
| decode_bs256 | 14.005 | 14.365 | +3 % |
| prefill_t2048 | 27.912 | **41.179** | **+48 %** |
| prefill_t8192 | 80.825 | **98.653** | **+22 %** |

Against the *best* plan the greedy plan is up to **1.48× slower**. This penalty is **two
orders of magnitude larger than the benefit of choosing the best plan over the baseline** — so
if this study has one operational finding, it is *don't fuse greedily*, not *fuse optimally*.

### 4.3 Why the standalone table does not compose into the plan

Two effects, both invisible without the end-to-end build:

**Shared producers.** #4, #5, #11b and #11b′ all fuse the post-attention RMSNorm into the
router — but that same normalization also feeds the w13 grouped GEMM *and* the shared expert.
Fusing it into the router **does not remove it**; you still pay it for the other consumers.
Their standalone wins (up to 1.239× for #11b′) are worth **zero** in the layer. Deleting the
norm outright requires fusing it into *every* K=6144 consumer, which is F11's `combined`
configuration at **0.478× / 0.605×**. **The entire prologue-fusion family is dominated in
context** — which is why the best standalone non-vector fusion appears in no optimal plan.

**Competing for the same operation.** ResAdd1 can be folded into o_proj's epilogue (#1) *or*
into the post-attention RMSNorm (#3), never both. #1 regresses at prefill and #3 does not, so
that axis resolves to **#3 everywhere**.

---

## 5. Why the GEMM fusions lose — three distinct mechanisms

A single-cause cost model would misattribute at least two of these.

| fusion | registers | spills | mechanism |
|---|---|---|---|
| **#1** o_proj + ResAdd | **identical** (126 vs 126) | 0 | **Codegen cliff.** The residual's DRAM traffic costs **+0.1 %** — proven with a stride-0 broadcast: same instructions, no traffic, same time. The epilogue's *instructions* cost **+22.8 %**. It only harms the *fast* config (107.5 → 87.4 TF/s, i.e. down onto the slow config's number): adding any epilogue disables the mainloop schedule that makes `BK=32, GM≥4` fast. |
| **#6** Up_Gate + SwiGLU | 104 → **214** | 0 | **Occupancy collapse + a hard SMEM bar.** Two accumulators halve CTAs/SM (4 → 2), and the unfused winner's `BM128 BN128` tile is **uncompilable** fused — 96 KB required against a 64 KB limit. |
| **#11** Lazy Pre-Norm | +26…+28 | 0 | **Displacement, not overlap.** Same CTAs/SM at t2048, so not occupancy. The in-mainloop sum-of-squares costs +1.0–1.2 % where the GEMM is memory-bound but **+65–68 % where it is compute-bound**. Plus a narrower SMEM bar: 7 of 79 configs fail fused by *exactly +4096 B* of cross-warp reduction scratch — including the unfused winner's tile family. |

**Why the published technique does not port.** *"Towards Free Normalization"* (Zhou et al.,
2026) relies on **warp specialization** to put the reduction on dedicated warps so it overlaps
the MMA pipeline. Triton 3.0 on MACA has **no warp specialization, no TMA, no clusters**. The
free lane the algorithm needs does not exist, so the reduction *displaces* MMA work instead of
hiding behind it. Four sum-of-squares implementations were compared (per-step `tl.sum`,
tile-accumulate, tensor-core dot-with-ones, second-load); they move the cost by 10–30 % and
**none removes it**.

**Direction matters more than membership.** #11b and #5 fuse *the same two operators* under
the same algebraic identity and reach opposite verdicts:

| | shape of the fused kernel | prefill |
|---|---|---|
| **#11b** | GEMM-shaped, norm as epilogue | **1.10–1.13×** ✅ |
| **#5** | norm-shaped (row-per-program), GEMM as epilogue | **0.47–0.68×** ❌ |

Forcing the router GEMM into the normalization's row-per-program tiling drops it from **76
TF/s to 32 TF/s**. *Which operator's tiling survives* decides the outcome, not which operators
are fused.

**And the corollary that generalizes best:** the winning norm fusion is the one that stays
*out* of the k-loop. #11b′ computes `rstd` in a tiny separate reduction and applies it as a
pure epilogue scale on an untouched GEMM — beating full Lazy Pre-Norm at **every** regime and
reaching 95–100 % of its ceiling. On C500/Triton 3.0, **normalization is not free, and the
closest to free is to stop putting it inside the mainloop.**

---

## 6. What the study is worth, and where the real lever is

At `prefill_t8192` the layer is three GEMMs:

| component | time |
|---|---|
| w13 grouped GEMM | 38.2 ms |
| w2 grouped GEMM | 20.2 ms |
| o_proj GEMM | 15.5 ms |
| **subtotal** | **~74 ms** |
| all fusible glue (#3, #10, router) | ~1.8 ms |

The three shippable fusions save ≈ **0.34 ms on a ~76 ms layer — about 0.4 %**. They are free,
correct, and worth taking. But **fusion cannot recover what it does not touch**, and it does
not touch the GEMMs.

**The lever is the 107 vs 215 TF/s gap to the vendor BLAS — worth ~30 ms on this layer, two
orders of magnitude more than every fusion in this study combined.**

### 6.1 Where production practice is confirmed

Three independent confirmations that the reference engines already have this right:

- **sglang keeps GEMM1 and `silu_and_mul` as separate launches.** #6 measures **0.553×** fused.
- **sglang gates `FUSE_SUM_ALL_REDUCE` behind a flag.** #8 measures 1.00–1.01× decode and
  0.87–0.90× prefill — exactly the shape that justifies a flag rather than a default.
- **`fused_add_rmsnorm` ships enabled everywhere.** #3 measures 1.08–1.32× and saturates its
  ceiling.

---

## 7. Threats to validity

- **The bytes-based model was wrong about the ranking.** The roofline predicted #4/#5 would be
  the best remaining fusion (ceiling up to 1.97×); they measured **worst** (0.21–0.68×).
  Traffic was the wrong currency for GEMM fusions on this backend — recorded as a failed
  prediction.
- **The `#8` L2-residency crossover prediction was wrong** (§4.1).
- **#11b's two measurement campaigns disagree on the sign at t2048** (§3.1).
- **min-of-N estimators are downward-biased** on a machine throwing 25–320 % excursions;
  LOG-08 §F5 re-estimates the affected decode_bs1 cells with median-of-8 (overstatement up to
  1.7 pp).
- **#11b is not a bit-exact drop-in.** It is *more* accurate than the framework path
  (1.5e-3 vs 2.5e-3 against exact fp32, because it skips the bf16 rounding of `x2`) but flips
  **~1.2 % of top-8 routing decisions** on near-ties. Note that the `rel_err` columns in
  `results/f11b_*.json` appear to say the opposite; they are measured against a reference that
  itself rounds `x2` to bf16 and therefore flatters the unfused path — see
  `REPORT-lazy-prenorm.md` A6.
- **A bug that nearly shipped:** the first end-to-end run had every non-atomic configuration
  double-applying the routing weights. Caught before the results were used; noted because it
  is the class of error that silently changes a ranking.
- **No hardware counters** (A4) — mechanism attributions in §5 are inferred from timing,
  compiler output and targeted ablations, not measured.

---

*Per-fusion detail: LOG-01 (#1), LOG-02 (#3), LOG-03 (#4/#5), LOG-04 (#6), LOG-05 (#8/#9),
LOG-06 (#10), LOG-07 (#11). Audit: LOG-08. Consolidated table: LOG-09. Main-session
verification: LOG-10. Whole-layer optimum: LOG-11. #11b architecture: LOG-12 +
`REPORT-lazy-prenorm.md`.*
