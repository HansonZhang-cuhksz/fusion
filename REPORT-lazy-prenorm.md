# Lazy Pre-Norm on the GLM-5.2 router GEMM — findings, assumptions, and system impact

**Fusion id** `#11b` · **GPU** MetaX C500 (104 CUs, 64 KB SMEM/CTA, 131072 regs/SM, 8 MB L2,
warp = 64 lanes) · **Stack** MACA 3.7, `torch 2.8.0+metax`, `triton 3.0.0+metax`
**Kernel** `glm52/kernels/lazy_prenorm.py` · **Sweep** `glm52/bench/bench_f11b_sweep.py`
**Data** `results/f11b_{decode,prefill}_T*.json` (31 points, T = 1 … 262144),
`results/f11b_arch_analysis.json` · **Analysis script** `glm52/analyze_f11b_arch.py`
**Companion logs** `log/LOG-07` (original build), `log/LOG-12` (architectural comparison)

---

## 0. Summary

Lazy Pre-Norm folds an RMSNorm into the router GEMM's k-loop, eliminating the normalized
activation `x2` entirely. It is **correct, numerically sound, and a net loss almost
everywhere it would actually run**.

| | result |
|---|---|
| **Decode (T = 1 … 1024)** | **0.62× – 0.91×** — loses at every single point |
| **Crossover** | **T ≈ 4096** (1.026×) |
| **Prefill (T ≥ 8192)** | **1.19× – 1.41×**, asymptoting to ~1.39× |
| **Best case** | 1.411× at T = 65536 |
| **Worth to the system** | **+0.114 % of layer time** at prefill t8192; **−0.04 % to −4.5 %** everywhere else |

The kernel-level speedup is real and as large as 1.41×. It is worth roughly **one part in
900** of a decoder layer, because the norm+router chain is only 0.6–0.8 % of that layer. The
interesting finding is not the win — it is *why* a fusion that removes 65 % of the memory
traffic returns only 41 % more speed, and why it inverts below T ≈ 4096.

---

## 1. What the kernel does

For an affine-free RMSNorm the row scale commutes with the matmul:

```
(A * rstd[:, None]) @ B  ==  (A @ B) * rstd[:, None]
```

so one CTA can accumulate `acc += tile_A @ tile_B` **and** `sq += (tile_A*tile_A).sum(-1)` in
the same k-loop, then apply `rstd = rsqrt(sq/K + eps)` as an epilogue scale. The cyclic
dependency that normally forbids prologue-fusing a normalization — you need `rstd` *before*
the loop but only know it *after* — disappears. Hence "lazy".

GLM-5.2's elementwise affine `w` is not a blocker at inference: because `B` is constant,
`((A*rstd)*w) @ B == (A @ (w[:,None]*B)) * rstd`, so `w` folds into `B`'s rows **once,
offline**, exactly like merging a quantization scale.

The two workloads compared throughout, computing `rmsnorm(h1) @ W_gᵀ`,
`[T, 6144] × [6144, 256] → [T, 256]` fp32:

| | kernels | what happens |
|---|---|---|
| **UNFUSED** | 2 | ① read `h1`, row Σx², `rstd`, write `x2` ② read `x2`, MMA, write logits |
| **FUSED** | 1 | read `h1`, MMA **and** row Σx² in one k-loop, `rstd` in the epilogue, write logits. `x2` never exists. |

---

## 2. Assumptions — what this analysis takes for granted

These are load-bearing. Several materially affect the numbers.

**A1 — The affine weight is folded offline, and `B` is constant.** Both `fold_weight_nk`
calls happen before any timed region. This is legitimate at inference but **assumes no
per-step weight modification** — a LoRA adapter, a dynamically-quantized gate, or any runtime
`w` update would force the fold into the hot path and likely erase the win.

**A2 — No hardware performance counters were read.** *This is the largest caveat and it is
the one the report title's "PC reg" question runs into.* Every number in §3 is one of:
(a) **compiler-derived** — registers, spills, shared memory, TTGIR op counts, read directly
out of the Triton `CompiledKernel`; (b) **analytic** — byte counts derived from shapes and
dtypes from first principles; or (c) **measured-and-divided** — achieved GB/s and TF/s from
wall-clock against known work. What was tried, and why it failed, is in §6. The practical
consequence: the traffic figures in §3.2 are **modelled, not observed**, and the attribution
in §5.3 is **inferred, not measured**.

**A3 — Occupancy is computed, not measured.** CTAs/SM = `min(131072/(regs·threads),
65536/smem)`. This is the standard static bound; it ignores launch-tail effects and assumes
the scheduler achieves the bound.

**A4 — Timing protocol.** `bench_chain` (`glm52/common.py:65`) flushes L2 **once before each
repetition of the whole list**, never between the kernels inside it. The unfused chain is
passed as one list, so the fused side is not credited with an artificial cache advantage and
no per-kernel timings were summed. p50 of 5 rounds; round spread is recorded
(`gain_spread_pct`, ~1.25 % at T=65536).

**A5 — Both sides are independently autotuned**, and the reported configs are the tuner's
winners for each side separately. This is the fair protocol, but it means **the comparison
is between two differently-configured kernels** — which turns out to be the single most
important fact in §3.4. `analyze_f11b_arch.py` therefore also reports a matched-config column
to separate "what fusion does" from "what the tuner chose".

**A6 — The `rel_err` columns in the sweep JSON must not be read as an accuracy verdict.**
The sweep's reference is `R.rmsnorm(h1, w)` then an fp32 linear, and `reference.rmsnorm`
(`glm52/reference.py:25`) computes `(xf * rstd).to(bf16) * w` — it **rounds `x2` to bf16**,
which is precisely the rounding the unfused path performs and the fused path skips. That
reference therefore shares the unfused algorithm's error and flatters it: the sweep records
`rel_err_unfused ≈ 3e-4` vs `rel_err_fused ≈ 2.6e-3` across all 31 points. Against an
**exact fp32** reference (LOG-07 §6.1) the ordering reverses and the fused path is closer to
truth at every regime (1.5–2.2e-3 vs 2.5–2.7e-3). §3.6 uses the exact-fp32 numbers.

**A7 — Peak figures are this machine's measured achievable rates**, not vendor spec:
**107 TF/s** bf16 (Triton-achievable; the vendor BLAS reaches ~215 TF/s) and **1.30 TB/s**
DRAM. Utilization percentages in §3.7 are against these, so they read high; against vendor
peak they would roughly halve.

**A8 — Isolated-kernel measurement.** The chain is benchmarked alone, with L2 flushed. In a
real layer the router runs immediately after an attention output that may leave `h1` partly
resident in the 8 MB L2. At T ≤ 1024 the activation is ≤ 12 MB, so a warm L2 would shrink the
unfused path's `x2` round-trip and **reduce the fused path's advantage further**. The decode
losses in §4 are therefore, if anything, optimistic about fusion.

**A9 — Decode `T` = batch size** (one token per sequence). `T=32` is `decode_bs32`.

**A10 — Single exclusive GPU on a shared machine** that this study has documented throwing
25–320 % one-off timing excursions. Sub-1 % differences are not resolvable; the effects
reported here (0.62×, 1.41×) are far outside that noise, but the system-level percentages in
§4 are near it.

**A11 — The tuner found near-optimal configs for both sides.** Grid parity was audited
(LOG-08); no claim is made that a hand-written config could not beat either side.

---

## 3. Performance differences, aspect by aspect

### 3.1 Execution time — the whole sweep

| T | regime | unfused (ms) | fused (ms) | gain | norm | gemm | BLOCK_N | Σx² redundancy |
|---|---|---|---|---|---|---|---|---|
| 1 | decode | 0.0724 | 0.1111 | **0.652×** | 0.0225 | 0.0627 | 32 | 8× |
| 32 | decode | 0.0699 | 0.1106 | **0.632×** | 0.0197 | 0.0630 | 32 | 8× |
| 128 | decode | 0.0735 | 0.1111 | 0.661× | 0.0233 | 0.0635 | 32 | 8× |
| 512 | decode | 0.1157 | 0.1352 | 0.856× | 0.0348 | 0.0968 | 32 | 8× |
| 1024 | decode | 0.1400 | 0.1841 | 0.761× | 0.0468 | 0.1078 | 32 | 8× |
| 2048 | prefill | 0.2163 | 0.2273 | 0.951× | 0.0717 | 0.1577 | 64 | 4× |
| **4096** | prefill | 0.3471 | 0.3382 | **1.026×** | 0.1211 | 0.2406 | 128 | 2× |
| 8192 | prefill | 0.5796 | 0.4879 | **1.188×** | 0.2196 | 0.3753 | 64 | 4× |
| 16384 | prefill | 1.0826 | 0.7995 | **1.354×** | 0.4147 | 0.6799 | 128 | 2× |
| 65536 | prefill | 4.1316 | 2.9284 | **1.411×** | 1.5959 | 2.5495 | 128 | 2× |
| 262144 | prefill | 16.2150 | 11.6216 | **1.395×** | 6.3155 | 9.9118 | 128 | 2× |

Monotone in T apart from tuner noise at T=896–1024, with a clean sign change at T≈4096 and a
plateau near 1.39–1.41× beyond T=65536.

### 3.2 Memory system

Analytic DRAM traffic (bf16 activations, fp32 logits, 3 MB gate weight):

| T | unfused chain | fused | reduction |
|---|---|---|---|
| 32 | 4.16 MB | 3.41 MB | **1.22×** |
| 65536 | 2371 MB | 835 MB | **2.84×** |

The fusion removes exactly two of three activation passes — the `x2` write and the `x2`
re-read. What remains is one read of `h1`, the gate weight, and the logits.

### 3.3 Compute instructions (TTGIR op counts; identical at both T)

| op | norm | unfused GEMM | fused GEMM | fused @ *unfused* cfg |
|---|---|---|---|---|
| `tt.dot` (MMA) | 0 | 2 | **1** | 1 |
| `tt.load` (global) | 2 | 6 | **2** | 4 |
| `local_store` (SMEM) | 0 | 6 | **2** | 4 |
| `local_load` (SMEM) | 0 | 4 | 2 | 2 |
| `tt.reduce` | 2 | 0 | **2** | 2 |
| `math.rsqrt` | 1 | 0 | **1** | 1 |
| `arith.mulf` / `addf` | 4 / 2 | 0 / 0 | 3 / 3 | 3 / 3 |
| `convert_layout` | 1 | 1 | **2** | 2 |

The fused kernel inherits the norm's reduction ops verbatim — that *is* the fusion. It also
carries **one extra `convert_layout`**: the row-wise reduction result must be re-laid-out to
scale an MMA accumulator whose fragments are distributed differently across lanes.

**The redundant arithmetic is nearly free in FLOPs and expensive in time:**

| T | redundancy | extra arithmetic | achieved MMA throughput lost |
|---|---|---|---|
| 65536 | 2× | **+0.78 %** | **−12.3 %** |
| 32 | 8× | **+3.12 %** | **−41.3 %** |

A **16× amplification** between added work and lost throughput.

### 3.4 Registers, shared memory, occupancy — and the counter-intuitive result

| T | | regs | spills | SMEM | CTAs/SM | **warps/SM** | limited by | `num_stages` |
|---|---|---|---|---|---|---|---|---|
| 65536 | unfused GEMM | 204 | 0 | 24.6 KB | 2 | **8** | regs | **3** |
| 65536 | **fused GEMM** | **140** | 0 | 16.9 KB | 3 | **12** | regs | **1** |
| 32 | unfused GEMM | 136 | 0 | 24.6 KB | 2 | **8** | SMEM | 3 |
| 32 | **fused GEMM** | **112** | 0 | 12.3 KB | 4 | **16** | regs | 1 |

**The fused kernel is the lighter one** — fewer registers, less shared memory, 1.5–2× the
occupancy, zero spills on either side. This inverts the pattern from fusions #1 and #6, where
fusing raised register pressure and cost occupancy. The reason is the last column, and it is
explained in §5.4.

### 3.5 Utilization

| T=65536 | GB/s | % of 1.30 TB/s | TF/s | % of 107 TF/s | bound by |
|---|---|---|---|---|---|
| norm kernel | **1009** | **78 %** | — | — | **memory** |
| unfused GEMM | 343 | 26 % | **80.9** | **76 %** | **compute** |
| fused GEMM | 299 | 23 % | 71.0 | 66 % | compute |

| T=32 | GB/s | % | TF/s | % | bound by |
|---|---|---|---|---|---|
| norm | 40 | 3 % | — | — | latency |
| unfused GEMM | 57 | 4 % | 1.6 | 1 % | latency |
| fused GEMM | 32 | 2 % | 0.9 | 1 % | latency |

### 3.6 Numerics

Against an **exact fp32** reference (LOG-07 §6.1 — see assumption A6 for why the sweep's own
`rel_err` columns say the opposite):

| regime | fused | unfused |
|---|---|---|
| decode_bs1 | **2.18e-03** | 2.70e-03 |
| decode_bs256 | **1.47e-03** | 2.53e-03 |
| prefill_t8192 | **1.70e-03** | 2.57e-03 |

The fused path is **closer to truth at every regime**, because it never rounds `x2` to bf16.
But *different* is what matters downstream: the router feeds a top-8, and the fused path
flips **~1.2 % of top-8 expert selections** on near-ties (100 % agreement at bs1/bs32,
98.73 % at bs256). Both kernels are deterministic.

---

## 4. Contribution to system execution time

Whole-layer times from `log/LOG-11` (measured scope S3–S11 + shared expert; attention, MLA
projections and the DSA indexer are **excluded**, so shares of a *full* layer are smaller
still):

| regime | layer (ms) | norm+router chain (ms) | chain as % of layer | fused (ms) | Δ (ms) | **Δ as % of layer** |
|---|---|---|---|---|---|---|
| decode_bs1 | 0.8564 | 0.0724 | **8.46 %** | 0.1111 | −0.0387 | **−4.51 %** |
| decode_bs32 | 8.4361 | 0.0699 | 0.83 % | 0.1106 | −0.0407 | −0.48 % |
| decode_bs256 | 13.9625 | 0.0819 | 0.59 % | 0.1139 | −0.0320 | −0.23 % |
| decode_bs512 | 16.4796 | 0.1157 | 0.70 % | 0.1357 | −0.0200 | −0.12 % |
| decode_bs1024 | 21.5194 | 0.1388 | 0.64 % | 0.1823 | −0.0435 | −0.20 % |
| prefill_t2048 | 27.8321 | 0.2163 | 0.78 % | 0.2273 | −0.0110 | −0.04 % |
| **prefill_t8192** | 80.6149 | 0.5796 | 0.72 % | 0.4879 | **+0.0916** | **+0.114 %** |

Three things follow.

**The chain is a 0.6–0.8 % slice of the layer at every batch size that matters.** No
achievable speedup on it can be a system-level effect. Even the 1.411× peak at T=65536,
scaled by that ratio, is worth **≈0.21 % of layer time**. Across 75 MoE layers at t8192 the
saving is **6.9 ms per forward pass** — real, but against a ~6 s pass.

**Decode is where it is enabled that it hurts most.** At `decode_bs1` the chain is 8.5 % of
the layer (everything else is launch-bound too), so a 0.652× regression costs **4.5 % of the
whole layer** — by far the largest single effect in this report, and it is a loss.

**The regime that wins is the one that runs least often.** Prefill at T ≥ 4096 is a
prompt-processing burst; decode dominates wall-clock in serving. A single always-on
configuration would take the 0.12–4.5 % decode penalty to buy a 0.11 % prefill gain.

---

## 5. Why these differences exist

### 5.1 Why the traffic reduction collapses at small T

The gate weight is a **fixed 3 MB** regardless of T. At T=32 the activation is 0.4 MB, so the
constant dominates and the two removed passes are worth almost nothing (1.22×); at T=65536
the activation is 805 MB and dominates completely (2.84×). This single ratio is the
memory-side explanation of the entire decode/prefill split.

### 5.2 Why the reduction is redundant at all

Every CTA that owns an m-tile must know the full row Σx² over all K=6144, but each CTA only
owns a `BLOCK_N` slice of the output. So every CTA sharing an m-tile recomputes the *whole*
row reduction: **`⌈256/BLOCK_N⌉` times**. N=256 is small, so this factor is large — 8× at
decode's `BLOCK_N=32`, 2× at prefill's `BLOCK_N=128`.

This creates a trap with no exit at small T. The only way to raise the CTA count — and at
T=32 the grid is **16 CTAs across 104 CUs**, so ~85 % of the machine is idle — is to shrink
`BLOCK_N`, which *directly multiplies the redundancy*. **Parallelism must be bought with
redundant work.** The tuner picks `BLOCK_N=32` at decode because idle CUs cost more than 8×
redundant reduction, and neither choice is good.

### 5.3 Why 0.78 % more arithmetic costs 12.3 % of throughput

Because the cost is not arithmetic. The reduction sits on the **critical path inside the
k-loop**: it consumes the A-fragment that the MMA also needs, so it *displaces* tensor-core
issue slots rather than filling idle ones. Triton on this backend offers no warp
specialization, so the reduction cannot be moved onto warps that would otherwise be waiting.
The GEMM was already at 76 % of achievable compute — there was no headroom for the reduction
to hide in. *(Attribution inferred from the FLOP/time ratio, not counter-measured — A2.)*

### 5.4 Why the fused kernel uses **fewer** registers

The tuner independently chose `num_stages=3` for the unfused GEMM and **`num_stages=1`** for
the fused one. **The fused kernel won its configuration search by giving up software
pipelining.** With the reduction interleaved into the k-loop there is already enough
independent work in flight to cover load latency, so prefetch stages stop earning their cost
in registers and SMEM. Higher occupancy is a *consequence* of abandoning pipelining, not a
free gift — and it explains the otherwise-puzzling instruction counts in §3.3, where the
fused kernel shows 1 `tt.dot` and 2 global loads against the unfused kernel's 2 and 6: those
are per-stage groups, not per-tile work.

### 5.5 Why the crossover sits at T ≈ 4096

Two curves cross. The **benefit** — eliminated traffic — grows with T, from 1.22× at T=32
toward an asymptote of 3× as the fixed 3 MB weight is amortized. The **cost** — displaced MMA
issue slots — is roughly constant in *proportional* terms once the machine is full, and is
worst at small T where redundancy is 8× *and* the GPU is mostly idle. Below T≈2048 the grid
cannot fill 104 CUs, so the unfused chain's extra kernel launch and extra traffic are hidden
by latency the fused kernel cannot escape; above T≈4096 both kernels saturate and the traffic
saving becomes the dominant term.

### 5.6 Why decode can never win

At decode all three conditions fail simultaneously: the weight dominates traffic (so there is
little to remove), redundancy is at its worst 8× (because `BLOCK_N` must be small to get any
parallelism), and the machine is ~85 % idle anyway (so neither kernel is resource-bound and
the fused one just has a longer critical path). The 0.62–0.91× band is not a tuning failure —
it is structural.

---

## 6. On hardware performance counters

The counter-level questions — measured DRAM bytes, L2/VL1 hit rates, tensor-core issue duty —
**were attempted and not obtained.** For the record:

- **`torch.profiler` with `ProfilerActivity.CUDA` works** on this stack and gives per-kernel
  device timings through the MACA kineto path (verified: it resolves vendor kernel names such
  as `mcblas__Mck_bf16gemm_nn_128x128x128_8m1n8k_256t_4stage`). It exposes **no** counters.
- **MCPTI is installed** (`/opt/maca/lib/libmcpti.so` + headers) and is a faithful CUPTI 1.x
  clone — `mcptiMetricGetIdFromName`, `mcptiMetricCreateEventGroupSets`, `mcptiMetricGetValue`,
  `mcptiEventGroup*`. Its embedded metric catalogue contains exactly what this report wants:
  `dram_read_bytes`, `dram_write_bytes`, `dram_utilization`, `gld_transactions`,
  `gst_transactions`, `global_hit_rate`, `achieved_occupancy`, `inst_executed_fma_pipe_*`,
  `flop_hp_efficiency`. Reading them needs a C harness (subscribe a launch callback, multi-pass
  event replay, evaluate) — **not built**.
- **`mcProfiler`** exposes an ideal MetaX-native metric set (`Total`/`Compute`/`Memory
  Instructions`, `AP busy Duty`, `WAVES`, `Global Read/Write Instructions`, `L2C Hit Rate`,
  `VL1 Hit Rate`, `average conflict cycles per instruction`). Blockers, in order: it writes its
  log into its root-owned install directory (worked around by copying the tree to a writable
  path); it is a Flask client/server with **encrypted `.pcd` metric configs**; and three
  separate `perf_exec` runs produced SQLite databases with the correct schema and **zero rows
  in every table**. This matches the note carried from the earlier project: *"mcProfiler
  `perf_exec` value-dump is metric-group-sensitive; prefer MCPTI directly."*

**What counters would add:** the 2.84× traffic figure would become a measurement rather than a
model; L2/VL1 hit rates would settle how much of the `x2` round-trip the unfused chain
actually serves from the 8 MB L2 (directly relevant to assumption A8); and `AP busy Duty`
would confirm or refute the issue-displacement attribution in §5.3. None of the report's
conclusions depend on them — the sign, magnitude and crossover of the effect are all measured
wall-clock — but §3.2 and §5.3 remain **modelled and inferred respectively**.

---

## 7. Verdict

**Do not enable this fusion unconditionally.** It is a prefill-only optimization worth
~0.11 % of layer time at t8192, and it costs 0.12–4.5 % of layer time at every decode batch
size. If it ships at all it must be gated on `T ≥ 4096`, and the ~1.2 % top-8 routing flip
rate must be acceptable.

The deeper finding is the one this fusion shares with every other GEMM fusion in this study:
**on this backend, work added to a tuned Triton GEMM costs far more than the work itself.**
0.78 % more arithmetic bought a 12.3 % throughput loss. The fusion still won at large T — but
only because it was removing 65 % of the memory traffic, which is an unusually large prize.
Fusions with a smaller prize (#1, #6, #11a, #4/#5) all lost. The lever that would change this
is not fusion at all: it is the **107 vs 215 TF/s gap to the vendor BLAS**, worth ~30 ms on a
~76 ms layer — two orders of magnitude more than every fusion in this study combined.
