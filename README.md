# Kernel fusion in a GLM-5.2 MoE transformer layer, on MetaX C500

A Triton implementation and measurement of every plausible kernel fusion in one **GLM-5.2**
MoE decoder layer, run on a **MetaX C500** (domestic accelerator, MACA 3.7 / `torch
2.8.0+metax` / `triton 3.0.0+metax`). Each fusion is built as a fused kernel *and* an unfused
counterpart cut from the same source, each independently autotuned, and compared against a
latency-aware roofline ceiling and the vendor BLAS.

## The question

Eleven candidate fusions were proposed. Two were rejected on analysis, nine were built and
measured. The result is mostly negative, and the negatives are the interesting part.

## Headline

**Only three of the nine fusions are worth doing, and the profitable ones are exactly the
memory-bound vector fusions.** Every attempt to fuse work *into a GEMM* either did nothing or
actively hurt — on this backend, adding an epilogue or prologue to a tuned Triton GEMM costs
15–47 % of its throughput for reasons that are usually **not** traffic, registers, or occupancy.

| verdict | fusions |
|---|---|
| ✅ ship | **#3** ResAdd+RMSNorm (1.08–1.32×), **#10** Merge+ResAdd (1.15–1.20×), **#11b′** half-fused pre-norm on the router (1.03–1.24×) |
| ➖ neutral, then harmful | **#8/#9** Down+Merge (1.00–1.01× decode → 0.87–0.90× prefill) |
| ❌ harmful | **#1** o_proj+ResAdd (0.85×), **#6** UpGate+SwiGLU (0.55×), **#11a** pre-norm→w13 (0.48×), **#4/#5** Norm+Router (0.21–0.68×) |
| 🚫 filtered on analysis | **#2** o_proj+ResAdd+RMSNorm, **#7** UpGate+Act+Down |

And the honest caveat: the three wins are worth **~0.34 ms on a ~76 ms layer (≈0.4 %)**. The
layer's cost is the three big GEMMs, and those are exactly what cannot be fused into here.
The real lever is the 107 vs 215 TF/s gap to the vendor BLAS — worth ~30 ms, two orders of
magnitude more than every fusion in this study combined.

See [`log/LOG-09-consolidated-results.md`](log/LOG-09-consolidated-results.md) for the full
table and [`log/LOG-10-main-session-findings.md`](log/LOG-10-main-session-findings.md) for the
independent verification and attribution work.

## Layout

```
glm52/
  config.py       GLM-5.2 architecture constants (verbatim from the HF config.json)
  common.py       benchmark harness: chain timing with L2 flush, autotune, correctness
  reference.py    fp32 ground truth + sglang-compatible MoE dispatch layout
  traffic.py      latency-aware roofline ceiling per fusion per regime
  consolidate.py  joins every results/*.json against the ceilings
  kernels/        the Triton kernels, one module per fusion family
  bench/          one runnable benchmark driver per family
results/          raw JSON: every timing, every tuning table, every correctness check
log/              LOG-00 plan+filter, LOG-01..07 per fusion, LOG-08 audit,
                  LOG-09 consolidated, LOG-10 main-session verification
```

## Reproducing

```bash
cd /home/zhangshuhan/fusion
PY=~/my-envs/fusion/bin/python

$PY -m glm52.traffic                      # roofline ceilings for every fusion x regime
CUDA_VISIBLE_DEVICES=0 $PY glm52/bench/bench_f03_resadd_rmsnorm.py    # one family
$PY -m glm52.consolidate                  # cross-fusion table vs ceilings
```

Benchmarks need an exclusive GPU — two concurrent runs on one device corrupt every timing.
GPU 2 on this machine is hardware-dead (`DMAQueue create failed`); use 0, 1 or 3.

## Method, and why it is trustworthy

- **One kernel source per fusion.** The unfused variant is the *same* Triton kernel with a
  `tl.constexpr` flag off, plus a separate kernel for the split-out work. Only the mapping
  (tile sizes, `num_warps`, `num_stages`, loop/grid order) differs, so the measurement
  isolates fusion rather than two people's coding.
- **Both sides tuned independently**, typically 100–450 configs per side per regime, in two
  stages (coarse then refine around the winner). Full tuning tables are in the JSON.
- **Chains timed as chains.** An unfused variant is a *sequence*; it is timed as one unit with
  a single L2 flush before the sequence, never between its kernels. Flushing between them
  would fabricate a fusion win. The independent audit confirmed no family did this.
- **Measured against a ceiling.** `traffic.py` models each kernel as
  `max(flops/C_PEAK, bytes/B_PEAK)` and a chain as the sum over its kernels. A speedup above
  its own ceiling is treated as a red flag, not a triumph — that check caught a real bad
  measurement.
- **Adversarially audited.** A separate agent re-derived results from the raw JSON and
  re-ran experiments from scratch against the project's launchers, hunting for rigged
  baselines, skipped work, and asymmetric setup costs — [`log/LOG-08`](log/LOG-08-fairness-audit.md).

Calibration measured on this machine, not taken from spec sheets: Triton dense-GEMM bf16
ceiling **107 TF/s** (vendor BLAS ~215), HBM **1.29 TB/s** mixed read+write and
**1.43–1.62 TB/s** read-only. Note the "Triton is 50 % of vendor" ratio is a *dense*-GEMM
property — for the grouped MoE GEMM, Triton reaches 0.93× the vendor path at prefill and
2.2× at decode_bs256, where the vendor route pays 256 kernel launches.

## Target configuration

GLM-5.2 (`zai-org/GLM-5.2`, `glm_moe_dsa`, 743B total / 39B active): hidden **6144**,
**256** routed experts + 1 shared, top-**8**, moe_intermediate **2048**, sigmoid scoring with
`noaux_tc` grouped top-k, `routed_scaling_factor` 2.5, SwiGLU, MLA with `kv_lora_rank` 512 and
`v_head_dim` 256, bf16 with fp32 accumulation and fp32 router math. One layer's expert weights
are 19.3 GB, which fits C500's 64 GB, so the benchmarks use the real 256-expert weight set.

Regimes: decode T ∈ {1, 32, 256} with o_proj K=32768 (absorbed MLA), prefill T ∈ {2048, 8192}
with K=16384 (non-absorbed).
