# LOG-00 — Plan, target spec, and fusion-candidate filter

**Date:** 2026-07-27 · **Hardware:** MetaX C500 ×4 (using GPUs 0, 1, 3; GPU 2 reports
`Not Available`) · **Stack:** MACA 3.7.0.38, `torch 2.8.0+metax3.7.1.3`, `triton 3.0.0+metax`

---

## 1. Target model — GLM-5.2 (`zai-org/GLM-5.2`, arch `glm_moe_dsa`)

Config fetched from `huggingface.co/zai-org/GLM-5.2/blob/main/config.json` on 2026-07-27.
743B total / 39B active, 78 layers (3 dense + 75 MoE), 1M context.

| Field | Value |
|---|---|
| `hidden_size` | **6144** |
| `moe_intermediate_size` | **2048** |
| `n_routed_experts` / `n_shared_experts` | **256** / 1 |
| `num_experts_per_tok` (top-k) | **8** |
| `scoring_func` / `topk_method` | `sigmoid` / `noaux_tc` |
| `norm_topk_prob` / `routed_scaling_factor` | `true` / **2.5** |
| `moe_router_dtype` | `float32` |
| `num_attention_heads` | 64 (MLA, `num_key_value_heads`=64) |
| `kv_lora_rank` / `q_lora_rank` | 512 / 2048 |
| `qk_nope` / `qk_rope` / `v_head_dim` | 192 / 64 / **256** |
| DSA indexer | `index_n_heads`=32, `index_head_dim`=128, `index_topk`=2048 |
| `rms_norm_eps` / `hidden_act` | 1e-5 / `silu` (SwiGLU) |

**Decisions taken with the user (2026-07-27):**

1. `attn` in candidates 1–2 means *attention core **then** o_proj* — the fusion is a GEMM
   epilogue fusion on o_proj. (Fusing ResAdd into the FA2/FlashMLA core itself is
   dimensionally invalid: the core emits `[T, 64, v_head_dim]`, the residual is `[T, 6144]`.)
2. o_proj K follows the MLA path per regime: **decode = absorbed**, `K = 64×512 = 32768`
   (FlashMLA emits `[T, 64, kv_lora_rank]`, `W_UV` folded into o_proj); **prefill =
   non-absorbed**, `K = 64×256 = 16384`.
3. dtype = **bf16** activations and weights, fp32 accumulate; router math in fp32.
4. Both regimes measured. Tuning + reporting regimes: decode T∈{1,32,256}, prefill T∈{2048,8192}.
5. Exhaustive per-kernel autotune, fused and unfused tuned **independently**.
6. Lazy Pre-Norm added as candidate **#11** (replaces the filtered #2).

Single GPU, no TP/EP. One MoE layer's expert weights in bf16 =
256 × (2·2048·6144 + 6144·2048) × 2 B = **19.3 GB**, which fits C500's 64 GB, so the
benchmarks use the real 256-expert weight set rather than a scaled-down proxy.

---

## 2. Layer dataflow (MoE layer, index ≥ 3)

```
h_in ─┬──────────────────────────────────────────────── residual ──┐
      └─ RMSNorm ─ MLA(q_a/q_b, kv_a/kv_b, DSA indexer, core) ─ o_proj ─┤
                                                          ResAdd1 ──────┴─ h1 ─┬──── residual ──┐
                                                                               │                │
              ┌── RMSNorm(post_attn) ── x2 ─┬─ Router(6144→256) ─ sigmoid ─ noaux_tc top-8       │
              │                             ├─ w13[e] (6144→2·2048) ─ SwiGLU ─ w2[e] (2048→6144) │
              │                             └─ shared expert (same shape)                        │
              └──────────────────── Expert Merge (Σ_k w_k · y_k) ── ResAdd2 ────────────────────┘
```

---

## 3. Candidate filter

Verdicts. **BUILD** = implemented, tuned, benchmarked. **FILTERED** = rejected with the
argument below, not implemented.

| # | Candidate | Verdict | Basis |
|---|---|---|---|
| 1 | o_proj + ResAdd | **BUILD** | Standard GEMM epilogue (cuBLASLt `beta=1` / CUTLASS `LinearCombinationResidualBlock`). |
| 2 | o_proj + ResAdd + RMSNorm | **FILTERED** | Three independent blockers — §3.1. |
| 3 | ResAdd + RMSNorm | **BUILD** | sglang `fused_add_rmsnorm`. 5 memory passes → 4. |
| 4 | ResAdd + RMSNorm + Router | **BUILD** | Router weight is 3 MB, fits C500's 8 MB L2 — §3.2. |
| 5 | RMSNorm + Router | **BUILD** | Same kernel, `HAS_RESIDUAL=False`. |
| 6 | Up_Gate + SwiGLU | **BUILD** | Removes the `[M, 4096]` intermediate entirely. |
| 7 | Up_Gate + Act + Down | **FILTERED** | Accumulator/recompute/atomic trilemma — §3.3. |
| 8 | Down + Expert Merge | **BUILD** | sglang's `MUL_ROUTED_WEIGHT` + scatter-accumulate. |
| 9 | Down + Expert Merge + ResAdd2 | **BUILD** | ResAdd2 seeds the accumulator; ~free on top of #8. |
| 10 | Expert Merge + ResAdd | **BUILD** | The non-GEMM baseline that #9 must beat. |
| 11 | **Lazy Pre-Norm** (RMSNorm ⇒ GEMM prologue) | **BUILD** (added) | §3.4. The norm fusion that actually works at N=6144. |

### 3.1 Why #2 (o_proj + ResAdd + RMSNorm) is filtered

RMSNorm reduces over the full hidden dim, which is the GEMM's **N** = 6144. A GEMM tiles
N, so no CTA owns a whole output row. Three ways out, all closed here:

- **(a) Force `tile_n = N`.** The PyTorch/Meta study "Towards Free Normalization" (Zhou et
  al., 2026-07-10) measures exactly this: **+25.6 % to +32.2 % at N ≤ 128**, then
  **−2.5 % at N=256 → −64.4 % at (K,N)=(256,256)**, and it derives a hard ceiling of
  `tile_n ≤ 512` from SMEM even on B200's 228 KB. C500 has **64 KB** of SMEM, i.e. an
  8× *smaller* budget than the machine where the technique already collapses at 256.
  N = 6144 is not reachable by more than an order of magnitude.
- **(b) Multi-CTA Norm via CTA clusters + distributed shared memory** (§3 of the same
  paper). Requires Hopper/Blackwell thread-block clusters and DSMEM. **C500 has neither**,
  and Triton 3.0 on MACA exposes no cluster API.
- **(c) Split-N partial sums + a second normalize pass.** Traffic-neutral by construction,
  so it cannot win:

  | | pass 1 | pass 2 |
  |---|---|---|
  | #1 (baseline) | GEMM+ResAdd → writes `h1` | RMSNorm: reads `h1`, writes `x2` |
  | #2 variant (c) | GEMM+ResAdd → writes `h1` **+ atomic row sum-of-squares** | normalize: reads `h1`, writes `x2` |

  Identical bytes moved, plus atomic contention on `T` row accumulators. Strictly worse
  than #1 — no measurement required, and none of the traffic terms depend on a tuning
  choice that could rescue it.

The real opportunity that #2 was reaching for is **prologue** fusion instead — see #11.

### 3.2 Why #4/#5 (RMSNorm + Router) survives, despite looking like a GEMM

The concern is weight re-reads: each CTA owns rows and must see all of `W_gate`
`[6144, 256]`. But `W_gate` is 6144·256·2 B = **3.0 MB**, and C500's L2 is **8 MB** — the
router weight is resident after the first CTA touches it, so the re-reads hit L2, not HBM.
HBM traffic for the weight stays ~3 MB regardless of T. The fusion then saves one full
read of `x2` (T·12 KB), which is 1 of the 5 memory passes in the
`add → norm → router` chain. Router FLOPs are negligible (T=8192 → 25.8 GFLOP, ~0.12 ms).
Top-k/sigmoid can also be folded in, since all 256 logits for a row live in one CTA — built
as an additional `FUSE_TOPK` variant.

### 3.3 Why #7 (Up_Gate + Act + Down) is filtered

The down-projection's output width is `hidden = 6144`. A CTA that produces activations
`act[BLOCK_M, BLOCK_I]` must accumulate into `out[BLOCK_M, 6144]`. Three formulations,
with GLM-5.2's numbers:

- **Hold the full output row.** `BLOCK_M=16` → 16 × 6144 × 4 B = **393 KB** of fp32
  accumulator against **64 KB** SMEM, and 98 304 registers of the SM's 131 072 (≈384
  regs/thread at 256 threads, past the per-thread ceiling). Infeasible by 6×.
- **Stream N-tiles, recompute activations per tile.** Recompute factor `6144 / BLOCK_N` =
  **24× (BN=256) to 96× (BN=64)** on the gate+up GEMMs, which are the expensive half.
- **Stream N-tiles, atomically accumulate.** Activation stays resident, but partial results
  go to HBM as fp32 atomics: `(2048 / BLOCK_I)` partials × `T·topk` × 6144 × 4 B. At
  T=4096, BLOCK_I=128 that is **12.9 GB** of read-modify-write traffic, against the
  **268 MB** the unfused version spends writing and re-reading the activation.

All three lose by more than an order of magnitude, and none of the margins is within reach
of a tuning choice. Notably sglang does not fuse these either: `fused_moe` runs GEMM1,
then `silu_and_mul`, then GEMM2 as separate launches. Filtered on analysis per user
decision.

### 3.4 Why Lazy Pre-Norm (#11) is added

From §2 of the same paper: for `C = rmsnorm(A) @ B`, the cyclic dependency (needing `rstd`
before the K-loop can start) dissolves because row-scaling commutes with matmul:

```
(A * rstd[:, None]) @ B  ==  (A @ B) * rstd[:, None]
```

so the K-loop accumulates `acc += tile_A @ tile_B` and `sq_sum += (tile_A*tile_A).sum(-1)`
side by side, and the normalization becomes an **epilogue scale**. Reported gains: **41–98 %
of the norm kernel's latency hidden**, at K,N up to 2048 — a regime that includes ours.

This fits GLM-5.2 structurally: `post_attention_layernorm`'s consumers (router GEMM, expert
`w13` GEMM, shared-expert GEMM) all have **K = hidden = 6144 = the norm dimension**, and a
GEMM CTA scans entire rows of A by construction.

The paper lists "cannot support elementwise affine" as limitation #1, and GLM-5.2's RMSNorm
does have a weight. **That limitation does not bind at inference**: the affine is a
column-wise scale of A, and B is a constant weight matrix, so

```
((A * rstd) * w) @ B  ==  (A @ (w[:, None] * B)) * rstd
```

lets `w` be pre-folded into B's rows **offline**, at zero inference cost. Implemented that
way, and validated against the unfolded reference.

---

## 4. Build plan

Per fusion: one kernel source with `tl.constexpr` flags selecting the fused
epilogue/prologue; the unfused variant is *the same kernel* with flags off plus a separate
kernel for the split-out work. Only the mapping (tile sizes, `num_warps`, `num_stages`,
loop order, grid order) differs between the two, and each is tuned independently
(`common.autotune`). Unfused chains are timed end-to-end with a single L2 flush before the
chain, never between its kernels.

**Production reference priority.** cuBLAS-equivalent (vendor BLAS via `torch.matmul`) is
the absolute reference line for every GEMM-shaped kernel; MoE dispatch layout
(`sorted_token_ids` / `expert_ids` / `num_tokens_post_padded`, BLOCK_M-padded) is a port of
sglang 0.5.10's `moe_align_block_size`, and the grouped-GEMM kernel structure follows
sglang's `fused_moe_kernel`.

**Calibration measured before starting** (bf16, 324-config sweep, `scratchpad/sweep.py`):

| shape | vendor BLAS | best Triton | ratio |
|---|---|---|---|
| M=4096, N=6144, K=16384 (o_proj prefill) | 3.83 ms / 215 TF/s | 7.80 ms / 106 TF/s | **0.49×** |
| M=4096, N=4096, K=6144 | 1.00 ms / 206 TF/s | 1.97 ms / 105 TF/s | **0.51×** |

Best config both times: `BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=2,
GROUP_M=8`. **Triton on this MACA backend reaches ~50 % of the vendor BLAS**, so absolute
TF/s is not comparable to production; the fused-vs-unfused *ratio* is the metric, and every
GEMM result additionally carries the vendor-BLAS line for context.

> **Correction (added after the runs, from LOG-04 §issues).** The "~50 % of vendor BLAS"
> figure is a property of the **dense** GEMM, not of the backend. For the **grouped MoE**
> GEMM the unfused Triton kernel reaches **0.93×** the vendor path at `prefill_t8192`
> (38.67 vs 36.05 ms) and **2.2×** the vendor path at `decode_bs256`, where the vendor route
> pays 256 separate kernel launches. Do not generalise the dense-GEMM ratio.
>
> Achievable HBM bandwidth was also revised upward twice: 1.05 TB/s (inherited, too low) →
> **1.29 TB/s** measured on mixed read+write (F3) → **1.43–1.62 TB/s** measured on near-pure
> weight reads (F6). `glm52/traffic.py` uses the conservative mixed figure; see LOG-10 §4.

C500 mapping constraints for the search space: warp = **64 lanes** (so `num_warps=8` is 512
threads and 16 is the 1024-thread ceiling), SMEM = 64 KB, 131 072 regs/SM, 104 CUs, 8 MB L2.
