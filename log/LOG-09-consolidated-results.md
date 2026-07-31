# LOG-09 — Consolidated results across all fusions

**Date** 2026-07-27 · **Hardware** MetaX C500 (MACA 3.7.0.38) · **Stack** `torch 2.8.0+metax`,
`triton 3.0.0+metax` · **Model** GLM-5.2 (`glm_moe_dsa`) · **dtype** bf16 / fp32 accumulate

Eleven candidates proposed, **2 filtered on analysis** (LOG-00 §3), **9 built, tuned and
measured**. Generated from `results/*.json` by `python -m glm52.consolidate`, joined against
the latency-aware ceilings from `glm52/traffic.py`.

---

## 1. Verdict table

Speedup = unfused / fused, so **>1 means the fusion helps**. "of ceiling" = measured speedup
divided by the latency-aware roofline ceiling for that fusion at that regime.

| # | fusion | decode bs1 | bs32 | bs256 | prefill 2048 | prefill 8192 | verdict |
|---|---|---|---|---|---|---|---|
| **3** | ResAdd + RMSNorm | 1.098× | 1.107× | 1.081× | **1.249×** | **1.315×** | ✅ **SHIP** |
| **10** | Expert Merge + ResAdd | **1.175×** | **1.181×** | **1.147×** | **1.204×** | **1.198×** | ✅ **SHIP** |
| **11b′** | *half-fused* pre-norm → **router only** | 1.034× | 1.030× | 1.100× | **1.184×** | **1.239×** | ✅ **SHIP** — but router-only; end-to-end ≈1.00× (§4) |
| **11b** | Lazy Pre-Norm → router GEMM | 0.684× | 0.680× | 0.771× | 1.096× | 1.127× | ⚠️ prefill-only, beaten by 11b′ |
| **8** | Down + Expert Merge (atomic) | 1.003× | 1.010× | 1.008× | 0.904× | 0.870× | ❌ neutral then harmful |
| **9** | Down + Merge + ResAdd2 (atomic) | 1.003×¹ | 1.010× | 1.006× | 0.908× | 0.874× | ❌ neutral then harmful |
| **1** | o_proj + ResAdd | 0.996× | 0.999× | 1.005× | 0.871× | **0.846×** | ❌ **harmful** |
| **6** | Up_Gate + SwiGLU | 0.987× | 0.975× | 0.960× | **0.553×** | 0.774× | ❌ **harmful** |
| **11a** | Lazy Pre-Norm → w13 GEMM | 0.902× | 0.961× | 0.965× | **0.476×** | 0.603× | ❌ **harmful** |
| **5** | RMSNorm + Router | 0.473× | 0.464× | 0.467× | 0.475× | 0.680× | ❌ **harmful** |
| **4** | ResAdd + RMSNorm + Router | 0.391× | 0.379× | 0.388× | 0.437× | 0.669× | ❌ **harmful** |
| 8/9 | token-major merge variant | 1.083× | 0.645× | 0.137× | 0.032× | 0.021× | ❌ bs1 only, then catastrophic |
| 4/5 | + `FUSE_TOPK` | 0.210× | 0.207× | 0.227× | 0.418× | 0.534× | ❌ worse than plain fusion |

¹ against the *2-kernel* unfused baseline. The family's own headline used a 3-kernel baseline,
which inflated this cell to 1.025× — see LOG-08 §F4.

**Filtered without implementation** (LOG-00 §3.1, §3.3): **#2** o_proj+ResAdd+RMSNorm
(`tile_n = N` needs N ≤ 512 on a 228 KB-SMEM B200 and collapses at N=256; C500 has 64 KB and
our N is 6144; the Multi-CTA alternative needs Hopper/Blackwell clusters + DSMEM; the
atomic two-pass fallback is provably traffic-neutral). **#7** Up_Gate+Act+Down (a 393 KB fp32
accumulator against 64 KB SMEM, *or* 24–96× activation recompute, *or* 12.9 GB of fp32 atomics
against the 268 MB it saves — every margin >10×).

## 2. The one-line result

**Only the memory-bound vector fusions pay.** #3 and #10 win at every regime and sit at
87–100 % of their roofline ceilings. **Every fusion that puts extra work inside a GEMM's
mainloop or epilogue loses**, and loses hardest exactly where the GEMM is compute-bound.

This is not a traffic story. The roofline model predicted #4/#5 would be the best remaining
fusion in the study (ceiling up to 1.97×); it measured **worst** (0.21–0.68×). Bytes were the
wrong currency.

## 3. Why the GEMM fusions lose — three distinct mechanisms

The study produced three *different* toxic-fusion mechanisms. A single-cause cost model
would misattribute at least two of them.

| fusion | registers | spills | mechanism |
|---|---|---|---|
| **#1** o_proj + ResAdd | **identical** (126 vs 126) | 0 | **Codegen cliff.** The residual's DRAM traffic costs **+0.1 %** (proven with a stride-0 broadcast: same instructions, no traffic, same time). The epilogue's *instructions* cost **+22.8 %**. It only harms the *fast* config — 107.5 → 87.4 TF/s, i.e. onto the slow config's number. Adding any epilogue disables the mainloop schedule that makes `BK=32, GM≥4` fast. |
| **#6** Up_Gate + SwiGLU | 104 → **214**, 160 → **242** | 0 | **Occupancy collapse + hard SMEM bar.** Two accumulators halve CTAs/SM (4 → 2), and the unfused winner's tile `BM128 BN128` is **uncompilable** fused (96 KB required vs 64 KB limit). |
| **#11** Lazy Pre-Norm | +26…+28 | 0 | **Displacement, not overlap.** Same CTAs/SM at t2048, so not occupancy. The in-mainloop sum-of-squares costs +1.0–1.2 % where the GEMM is memory-bound but **+65–68 % where it is compute-bound**. Plus a narrower SMEM bar: 7 of 79 configs fail fused by *exactly +4096 B* (cross-warp reduction scratch), including the unfused winner's tile family. |

**Why the paper's technique doesn't port.** "Towards Free Normalization" relies on **warp
specialization** to put the reduction on dedicated warps so it overlaps the MMA pipeline.
Triton 3.0 on MACA has no warp specialization, no TMA, no clusters. The free lane the
algorithm needs does not exist, so the reduction *displaces* MMA work instead of hiding behind
it. Four sum-of-squares implementations were compared (per-step `tl.sum`, tile-accumulate,
tensor-core dot-with-ones, second-load); they move the cost by 10–30 % and none removes it.

**Direction matters more than membership.** #11b and #5 fuse the *same two operators* with the
same identity and reach opposite verdicts:

| | shape of the fused kernel | prefill |
|---|---|---|
| **#11b** | GEMM-shaped, norm as epilogue | **1.10–1.13×** ✅ |
| **#5** | norm-shaped (row-per-program), GEMM as epilogue | **0.47–0.68×** ❌ |

Forcing the router GEMM into the normalization's row-per-program tiling drops it from
**76 TF/s to 32 TF/s** — the tiling mismatch of the paper's §1, in the wrong direction.

## 4. The best norm fusion is the one that stays out of the k-loop

The F11 agent's exploratory **half-fused** variant computes `rstd` in a tiny separate
reduction kernel and applies it as a **pure epilogue scale** on an otherwise-untouched GEMM.
It beats full Lazy Pre-Norm on the router at **every** regime, never regresses, and reaches
**95–100 % of its own ceiling** (1.241× at prefill):

| regime | full Lazy Pre-Norm | half-fused | ceiling |
|---|---|---|---|
| decode_bs1 | 0.684× | **1.034×** | 1.01× |
| decode_bs256 | 0.771× | **1.100×** | 1.24× |
| prefill_t2048 | 1.096× | **1.184×** | 1.241× |
| prefill_t8192 | 1.127× | **1.239×** | 1.241× |

It captures 2 of the 3 activation passes the full fusion saves, with none of the mainloop
cost, and composes better — one `rstd` kernel serves all consumers. **On C500/Triton 3.0,
normalization is not free, and the closest to free is to stop putting it inside the k-loop.**

**Scope this correctly.** Those numbers are for the **router GEMM only**. Applied to the w13
grouped GEMM the same trick is neutral-to-slightly-negative (0.953–1.007×), and the agent's
own end-to-end `combined` row is **0.955× (t2048) to 1.011× (bs1)** — i.e. roughly a wash
across the whole norm→consumers subgraph. The router win is real but small in absolute terms:
0.566 → 0.457 ms at prefill_t8192, a saving of **0.109 ms**.

## 4b. Absolute impact — the honest framing

Ratios flatter these results. At `prefill_t8192` the layer's cost is dominated by three GEMMs:

| component | time |
|---|---|
| w13 grouped GEMM | 38.2 ms |
| w2 grouped GEMM | 20.2 ms |
| o_proj GEMM | 15.5 ms |
| **subtotal** | **~74 ms** |
| ResAdd+RMSNorm (#3), Merge+ResAdd (#10), router | ~1.8 ms |

The three shippable fusions save roughly **0.09 + 0.14 + 0.11 ≈ 0.34 ms**, against a layer of
~76 ms — about **0.4 % end-to-end**. They are free, correct, and worth taking, but no
combination of fusions available here moves this layer meaningfully. **The layer's cost is the
GEMMs, and on this backend the GEMMs are exactly what cannot be fused into.**

The larger lever is not fusion at all: the dense Triton GEMM runs at 107 TF/s against the
vendor BLAS's 215. Closing *that* gap is worth ~30 ms on this layer — two orders of magnitude
more than every fusion in this study combined.

## 5. Where production practice is confirmed

Three independent confirmations that the reference engines have this right:

- **sglang keeps GEMM1 and `silu_and_mul` as separate launches.** #6 measures 0.553× if you
  fuse them. Correct call, by a wide margin.
- **sglang gates `FUSE_SUM_ALL_REDUCE` behind a flag** rather than enabling it always. #8
  measures 1.00–1.01× in decode and 0.87–0.90× at prefill — exactly the shape that justifies
  a flag.
- **`fused_add_rmsnorm` ships enabled everywhere.** #3 measures 1.08–1.32× and saturates its
  ceiling. Correct call.

## 6. Caveats a reader must carry

- **Triton, not production kernels.** The dense Triton GEMM reaches ~107 TF/s vs the vendor
  BLAS's ~215. Fused-vs-unfused *ratios* are the deliverable; absolute TF/s is not comparable
  to a production engine. The gap is **not** uniform — for the grouped MoE GEMM the unfused
  Triton kernel reaches 0.93× the vendor path at prefill and **2.2×** at decode_bs256, where
  the vendor route pays 256 kernel launches. And the vendor's per-expert matmul loop (20.10 ms
  at t2048) *loses* to our unfused Triton grouped GEMM (13.83 ms).
- **The atomic merge variants are non-deterministic** (bf16 `tl.atomic_add`, 8 contributions in
  hardware-scheduling order). Correct — no lost updates, errors sit at the bf16 rounding scale
  — but unusable where bitwise reproducibility is required (LOG-08 §F6).
- **#11b is not a bit-exact drop-in.** The fused path is *more* accurate than the framework
  path (1.5e-3 vs 2.5e-3 against fp32) because it skips the bf16 rounding of `x2`, but that
  flips ~1.2 % of top-8 routing decisions on near-ties.
- **#11a/#11b assume all `K=6144` consumers are fused**, so `x2` is genuinely dead. The
  `combined` row charges ONE norm kernel to the unfused side, which is the honest end-to-end
  number; the per-family rows double-count it.
- **Decode MoE has no headroom by construction.** At T ≥ 32 the layer streams 4–13 GB of
  expert weights, so every MoE-GEMM fusion has a 1.00–1.01× ceiling there. A large decode
  "win" in this study would be launch overhead or an artifact, not fusion.
- **One measurement caveat:** min-of-N estimators on sub-1 % effects are downward-biased on a
  machine that throws 25–320 % one-off excursions; LOG-08 §F5 re-estimates the affected
  decode_bs1 cells with median-of-8 (overstatement up to 1.7 pp).

## 7. Practical recommendation for a GLM-5.2 MoE layer on C500

**Fuse:** ResAdd+RMSNorm (#3), Expert Merge+ResAdd (#10), and the half-fused rstd-epilogue
pre-norm (#11b′). Together these are the layer's cheap, bandwidth-bound glue, and all three
run at 87–100 % of their ceilings.

**Do not fuse:** anything into the o_proj, w13 or w2 GEMMs. Keep `silu_and_mul` separate
(as sglang does), keep the expert merge separate at prefill, and use the **vendor BLAS's**
fused epilogue rather than a Triton one where a residual add into a GEMM is wanted —
`torch.addmm` regresses 0.5–1.7 % where the Triton epilogue regresses 15 %.

---

*Per-fusion detail: LOG-01 (#1), LOG-02 (#3), LOG-03 (#4/#5), LOG-04 (#6), LOG-05 (#8/#9),
LOG-06 (#10), LOG-07 (#11). Independent audit: LOG-08. Main-session verification and
attribution: LOG-10. Full numeric table: `python -m glm52.consolidate`.*
