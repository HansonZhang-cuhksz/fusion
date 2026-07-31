# LOG-12 — #11b: architectural comparison of the fused kernel vs the unfused chain

**Date** 2026-07-30 · **GPU** MetaX C500 (104 CUs, 64 KB SMEM/CTA, 131072 regs/SM, 8 MB L2,
warp = 64 lanes) · **Stack** `torch 2.8.0+metax`, `triton 3.0.0+metax` (backend `maca`, arch 80)

Two operationally distinct workloads computing the *same* result — the GLM-5.2 router logits
`softmax-free logits = rmsnorm(h1) · W_gᵀ`, `[T, 6144] × [6144, 256] → [T, 256]` fp32:

| | kernels | what each does |
|---|---|---|
| **UNFUSED** | 2 | ① `norm`: read `h1`, per-row Σx², `rstd`, write `x2` ② `gemm`: read `x2`, MMA, write logits |
| **FUSED** | 1 | `gemm`: read `h1`, MMA **and** per-row Σx² in the same k-loop, `rstd` as epilogue scale, write logits. `x2` never exists. |

Analysed at two batch sizes that bracket the sign change: **T=65536** (fusion wins 1.411×) and
**T=32** (fusion loses 0.632×). Reproduce with `glm52/analyze_f11b_arch.py`.

---

## 1. Memory system

Analytic DRAM traffic, from first principles (bf16 activations, fp32 logits, 3 MB gate weight):

| | unfused chain | fused | reduction |
|---|---|---|---|
| **T=32** | 4.16 MB | 3.41 MB | **1.22×** |
| **T=65536** | 2371 MB | 835 MB | **2.84×** |

The fusion removes exactly two of three activation passes — the `x2` write and the `x2`
re-read. What survives is one read of `h1`, the gate weight, and the logits.

**Why the reduction factor collapses at small T:** the gate weight is a *fixed* 3 MB. At T=32
the activation is only 0.4 MB, so the constant term dominates and there is essentially nothing
to remove; at T=65536 the activation is 805 MB and dominates completely. This single ratio —
1.22× vs 2.84× — is the memory-side explanation of the whole decode/prefill split.

Measured bandwidth utilisation (bytes ÷ measured time, against the 1.30 TB/s achievable):

| T | kernel | GB/s | % of achievable BW | TF/s | % of 107 TF/s | bound by |
|---|---|---|---|---|---|---|
| 65536 | norm | **1009** | **78 %** | — | — | **memory** |
| 65536 | unfused GEMM | 343 | 26 % | **80.9** | **76 %** | **compute** |
| 65536 | fused GEMM | 299 | 23 % | **71.0** | **66 %** | compute |
| 32 | norm | 40 | 3 % | — | — | latency |
| 32 | unfused GEMM | 57 | 4 % | 1.6 | 1 % | latency |
| 32 | fused GEMM | 32 | 2 % | 0.9 | 1 % | latency |

At T=65536 the two workloads are in **different regimes**: the norm kernel is a pure streaming
reduction at 78 % of bandwidth, the GEMM is at 76 % of compute. Fusion is therefore trading a
bandwidth-saturated kernel for extra work inside a compute-saturated one — which is exactly
the trade that can pay.

At T=32 *nothing* is saturated: 2–4 % of bandwidth, 1 % of compute. The workload is entirely
launch/latency-bound, so there is no resource for fusion to reclaim.

---

## 2. Compute instructions

TTGIR op counts (identical at both T; the `fused @ unfused cfg` column holds the mapping fixed
so it isolates what fusion itself adds rather than what the tuner chose):

| op | norm | unfused GEMM | fused GEMM | fused @ unfused cfg |
|---|---|---|---|---|
| `tt.dot` (MMA) | 0 | 2 | **1** | 1 |
| `tt.load` (global) | 2 | 6 | **2** | 4 |
| `local_store` (SMEM) | 0 | 6 | **2** | 4 |
| `local_load` (SMEM) | 0 | 4 | 2 | 2 |
| `tt.reduce` | 2 | 0 | **2** | 2 |
| `math.rsqrt` | 1 | 0 | **1** | 1 |
| `arith.mulf` / `addf` | 4 / 2 | 0 / 0 | 3 / 3 | 3 / 3 |
| `convert_layout` | 1 | 1 | **2** | 2 |

The fused kernel inherits the norm's reduction ops verbatim (`tt.reduce` ×2, `rsqrt`, the
mul/add chain) — that is the fusion. It also carries **one extra `convert_layout`**, because the
row-wise reduction result must be re-laid-out to scale an MMA accumulator whose fragments are
distributed differently.

**The redundant arithmetic is almost free; the redundancy is not.** Each CTA sharing an m-tile
recomputes the entire row Σx², so the reduction runs `⌈256/BLOCK_N⌉` times. In FLOPs that is
tiny:

| T | redundancy | extra arithmetic | MMA throughput lost |
|---|---|---|---|
| 65536 | 2× | **+0.78 %** | **−12.3 %** |
| 32 | 8× | **+3.12 %** | **−41.3 %** |

A 0.78 % increase in arithmetic costs 12.3 % of achieved MMA throughput — a **16× amplification**.
So the cost is not the mathematics. It is that the reduction sits on the critical path inside
the k-loop, consuming the A-fragment before the next MMA can be issued, with no warp
specialisation available to move it onto independent warps. It *displaces* tensor-core issue
slots rather than filling idle ones.

---

## 3. Occupancy and the register/pipeline trade — the counter-intuitive part

| T | | regs | spills | SMEM | CTAs/SM | **warps/SM** | limited by | `num_stages` |
|---|---|---|---|---|---|---|---|---|
| 65536 | unfused GEMM | 204 | 0 | 24.6 KB | 2 | **8** | regs | **3** |
| 65536 | **fused GEMM** | **140** | 0 | 16.9 KB | 3 | **12** | regs | **1** |
| 32 | unfused GEMM | 136 | 0 | 24.6 KB | 2 | **8** | SMEM | 3 |
| 32 | **fused GEMM** | **112** | 0 | 12.3 KB | 4 | **16** | regs | 1 |

**The fused kernel is the *lighter* one.** It uses fewer registers and less shared memory, and
achieves 1.5–2× the occupancy. Zero spills on either side.

That inverts the expectation set by fusions #1 and #6 (where fusing raised register pressure
and cost occupancy), and the reason is in the last column: the tuner independently chose
`num_stages=3` for the unfused GEMM and `num_stages=1` for the fused one. **The fused kernel
won its configuration search by giving up software pipelining.** With the reduction interleaved
into the k-loop there is enough independent work in flight to cover load latency without
prefetch stages, so the stages — and the registers and SMEM they cost — are not worth their
price. Higher occupancy is a *consequence* of abandoning pipelining, not a free win.

**At T=32 occupancy is irrelevant regardless.** The grid is 16 CTAs (2 m-tiles × 8 n-tiles)
across **104 CUs**: ~85 % of the machine is idle no matter how many warps fit per SM. This is
also why the tuner is trapped — the only way to raise the CTA count is to shrink `BLOCK_N`,
which directly multiplies the redundancy. Parallelism must be bought with redundant work.

---

## 4. Consolidated architectural differences

| dimension | unfused chain | fused kernel | net |
|---|---|---|---|
| kernel launches | 2 | 1 | −1 launch (~12 µs on this stack) |
| DRAM activation passes | 3 | 1 | **−2 passes** |
| DRAM traffic @ T=65536 | 2371 MB | 835 MB | **2.84× less** |
| DRAM traffic @ T=32 | 4.16 MB | 3.41 MB | 1.22× less (weight-dominated) |
| MMA instructions | 2 (3 stages) | 1 (1 stage) | fewer, but unpipelined |
| reduction ops | in a separate kernel, ×1 | inside the k-loop, **×⌈256/BLOCK_N⌉** | 2–8× redundant |
| extra arithmetic | — | +0.78 % (T=65536) / +3.12 % (T=32) | negligible in FLOPs |
| achieved MMA throughput | 80.9 TF/s (76 %) | 71.0 TF/s (66 %) | **−12.3 %** |
| registers / thread | 204 | 140 | fused is lighter |
| SMEM / CTA | 24.6 KB | 16.9 KB | fused is lighter |
| occupancy | 8 warps/SM | 12 warps/SM | fused is higher |
| software pipelining | 3 stages | **none** | given up by the fused kernel |
| layout conversions | 1 | 2 | reduction result must be re-laid-out |
| intermediate materialised | `x2` `[T, 6144]` bf16 (805 MB @ T=65536) | none | never leaves the chip |
| numerics | `x2` rounded to bf16 | never rounded | fused is **more** accurate; flips ~1.2 % of top-8 ties |
| determinism | deterministic | deterministic | unchanged |

**The trade in one sentence:** fusion converts a bandwidth-saturated streaming kernel (78 % of
DRAM bandwidth) into redundant, latency-critical arithmetic inside a compute-saturated GEMM
(76 % of peak), and it wins precisely when the eliminated traffic is worth more than the 12 %
of tensor-core throughput the un-overlapped reduction costs — which requires both a large
activation relative to the fixed weight *and* enough m-parallelism to keep the redundancy
factor at 2×.

---

## 5. Hardware counters: attempted, not obtained

Everything above is compiler-derived (registers, SMEM, TTGIR op counts), analytic (byte counts
from first principles), or measured-and-divided (achieved bandwidth and TF/s from wall-clock
against known work). **No hardware performance counters were read.** What was tried:

- **`torch.profiler` with `ProfilerActivity.CUDA` — works**, and gives per-kernel device
  timings via the MACA kineto path (verified: it resolves vendor kernel names such as
  `mcblas__Mck_bf16gemm_nn_128x128x128_8m1n8k_256t_4stage`). It exposes **no** counters.
- **MCPTI is present** (`/opt/maca/lib/libmcpti.so` + headers) and is a faithful CUPTI 1.x
  clone: `mcptiMetricGetIdFromName`, `mcptiMetricCreateEventGroupSets`, `mcptiMetricGetValue`,
  `mcptiEventGroup*`. The metric catalogue embedded in the library includes exactly what this
  analysis wants — `dram_read_bytes`, `dram_write_bytes`, `dram_utilization`,
  `gld_transactions`, `gst_transactions`, `global_hit_rate`, `achieved_occupancy`,
  `inst_executed_fma_pipe_*`, `flop_hp_efficiency`, `half_precision_fu_utilization`.
  Using it requires a C harness (subscribe a launch callback, multi-pass event replay,
  `mcptiMetricGetValue`) — not attempted here.
- **`mcProfiler`** exposes a MetaX-native metric set that maps well onto the questions
  (`Total`/`Compute`/`Memory Instructions`, `AP busy Duty`, `WAVES`, `Global Read/Write
  Instructions`, `L2C Hit Rate`, `VL1 Hit Rate`, `Dnoc Read Average Latency`,
  `average conflict cycles per instruction`). Blockers hit, in order: it writes its log into
  its root-owned install directory (worked around by copying the 527 MB tree to
  `~/.cache/mcprof`); it is a Flask client/server with **encrypted `.pcd` metric configs**;
  and `perf_exec` produced no output within 15 minutes on a 10-iteration workload. This
  matches the note carried over from the earlier project — *"mcProfiler `perf_exec` value-dump
  is metric-group-sensitive; prefer MCPTI directly."*

**What counters would add that this analysis cannot:** measured (rather than modelled) DRAM
bytes, so the 2.84× traffic prediction becomes a measurement; L2/VL1 hit rates, which would
settle how much of the `x2` round-trip the unfused chain actually serves from cache; and
`AP busy Duty`, which would confirm directly that the fused k-loop's problem is tensor-core
issue displacement rather than anything else. The conclusions above do not depend on them, but
the 2.84× figure and the "displacement" attribution remain **inferred, not observed**.
