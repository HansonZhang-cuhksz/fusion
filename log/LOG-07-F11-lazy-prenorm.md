# LOG-07 — Fusion #11: Lazy Pre-Norm (RMSNorm fused into a GEMM as a prologue)

**Result id** `f11_lazy_prenorm` · **GPU** MetaX C500, `CUDA_VISIBLE_DEVICES=1` (exclusive)
**Kernels** `glm52/kernels/lazy_prenorm.py` · **Bench** `glm52/bench/bench_f11_lazy_prenorm.py`
**Results** `results/f11_lazy_prenorm.json`

**Status: complete.** All 5 regimes measured, both consumers, plus an exploratory third
variant. Headline: F11a regresses everywhere (0.48-0.97x, ceiling 1.00x); F11b wins only
at prefill (1.10-1.13x, ceiling 1.64x); the "half-fused" variant of section 7 is the one
worth shipping (1.03-1.24x on the router, at 95-100% of its own ceiling).

---

## 1. The fusion

From *"Towards Free Normalization: Fusing Normalization into GEMM and Attention Kernels"*
(Zhou et al., PyTorch blog, 2026-07-10), section 2, pp. 7–11.

For an **affine-free** RMSNorm the row scale commutes with the matmul:

```
(A * rstd[:, None]) @ B  ==  (A @ B) * rstd[:, None]
```

so the GEMM CTA can accumulate `acc += tile_A @ tile_B` **and**
`sq += (tile_A * tile_A).sum(-1)` in the same k-loop, then apply
`rstd = rsqrt(sq/K + eps)` as an epilogue scale. The cyclic dependency that normally
makes prologue-fusing a normalization impossible (you need `rstd` *before* the k-loop,
but only know it *after*) disappears. The paper calls this "lazy" because the
elementwise half of the norm is deferred past the loop it logically precedes.

### 1.1 GLM-5.2's affine weight is not a blocker at inference

The blog lists elementwise affine as limitation #1: `w` is a **column**-wise scale of `A`,
which breaks the row-wise-multiplication precondition. At inference it folds away, because
`B` is a constant:

```
((A * rstd) * w) @ B  ==  (A @ (w[:, None] * B)) * rstd
```

`w` is folded into `B`'s **rows** once, offline — a load-time weight transform exactly like
merging a quantization scale. Both `fold_weight_nk` calls happen before any timed region,
and the identity is validated against the unfolded `reference.rmsnorm` + matmul before the
benchmark times anything:

```
folding identity: router rel_err 3.260e-03,  w13 rel_err 2.878e-03
```

Both residuals are exactly the bf16 rounding of `x2` that the *unfused* path performs and
the fused path skips — see §6.1; the fused path is the more accurate of the two.

### 1.2 The two consumers with K == hidden == 6144

| | GEMM | shape | N | n-tiles at BLOCK_N=128 |
|---|---|---|---|---|
| **F11a** | routed-expert w13 grouped GEMM | `[T·8, 6144] × [6144, 4096]` per expert | 4096 | 32 |
| **F11b** | router GEMM | `[T, 6144] × [6144, 256]` | 256 | 2 |

F11a is structurally sglang 0.5.10's `fused_moe_kernel` (same `sorted_token_ids` /
`expert_ids` / `num_tokens_post_padded` dispatch, same `offs_token // top_k` gather,
same `even_Ks` fast path, same grouped pid swizzle). F11b is a plain dense GEMM.

### 1.3 The `x2` materialization question — answered explicitly

`x2 = rmsnorm(h1)` feeds **three** consumers: the router, the routed-expert w13 GEMM, and
the shared expert's w13 GEMM. A fused variant that never writes `x2` is only valid if
**all** of them are fused.

**We take option (ii): fuse all K==6144 consumers, so `x2` is genuinely dead.** Both
benchmarked consumers are fused; the shared expert is the identical transform applied to a
one-expert weight (its `w13_shared` gets the same offline fold), so the precondition is
satisfiable in the real layer with no extra machinery.

The consequence for accounting is that `glm52/traffic.py` charges the *same* norm kernel
to F11a **and** to F11b, double-counting it. The per-family rows below reproduce that
model so the measured/ceiling comparison is apples-to-apples, and a third **`combined`**
row gives the honest end-to-end number:

```
combined UNFUSED = [1 norm kernel] + [router GEMM] + [w13 grouped GEMM]
combined FUSED   =                   [router GEMM] + [w13 GEMM]      (x2 never written)
```

---

## 2. Traffic analysis

Per activation pass `act = T·H·2` bytes. `ne` = expected distinct experts touched.

| regime | T | act (MB) | W_gate (MB) | ne | w13 traffic (GB) | w13 GEMM (TFLOP) | router GEMM (GFLOP) |
|---|---|---|---|---|---|---|---|
| decode_bs1 | 1 | 0.01 | 3.00 | 8 | 0.37 | 0.000 | 0.00 |
| decode_bs32 | 32 | 0.38 | 3.00 | 162 | 7.59 | 0.013 | 0.10 |
| decode_bs256 | 256 | 3.00 | 3.00 | 256 | 12.00 | 0.103 | 0.81 |
| prefill_t2048 | 2048 | 24.00 | 3.00 | 256 | 12.00 | 0.825 | 6.44 |
| prefill_t8192 | 8192 | 96.00 | 3.00 | 256 | 12.00 | 3.299 | 25.77 |

Bytes moved, fused vs unfused (`SZ`=2 for bf16, logits fp32):

```
F11a  unfused : norm kernel  2·act        +  GEMM  (act + w13 + rows·I·2)
      fused   :                              GEMM  (act + w13 + rows·I·2)
      saving  : 2·act — against 0.37–12 GB of expert weights.

F11b  unfused : norm kernel  2·act        +  GEMM  (act + 3 MB + T·256·4)
      fused   :                              GEMM  (act + 3 MB + T·256·4)
      saving  : 2·act — against a 3 MB weight that fits in the 8 MB L2.
```

`python -m glm52.traffic` (latency-aware ceilings, `C_PEAK`=107 TF/s, `B_PEAK`=1.30 TB/s):

| regime | F11a traffic | F11a **ceiling** | F11b traffic | F11b **ceiling** |
|---|---|---|---|---|
| decode_bs1 | 1.00× | **1.00×** | 1.01× | **1.01×** |
| decode_bs32 | 1.00× | **1.00×** | 1.22× | **1.22×** |
| decode_bs256 | 1.00× | **1.00×** | 1.96× | **1.64×** |
| prefill_t2048 | 1.00× | **1.00×** | 2.66× | **1.64×** |
| prefill_t8192 | 1.01× | **1.01×** | 2.79× | **1.64×** |

So **F11a has no headroom by construction** — expert-weight traffic (0.37–12 GB) swamps
the 2·act (0.02–192 MB) the fusion removes, and at prefill the GEMM is compute-bound on
top of that. Anything F11a measures below 1.00× is pure fusion overhead. **F11b is where
the value is**, and it is the paper's ideal case: N=256 means 1–4 n-tiles, so the
sum-of-squares redundancy is ~1–4×, not the 16–64× the brief warns about for w13.

### 2.1 Redundancy factors (the catch, quantified)

The sum of squares of one row is recomputed by every CTA that shares that row's m-tile.
Relative to the norm kernel, which computes it exactly once per token:

```
F11b redundancy = cdiv(256,  BLOCK_N)                      -> 1 … 8
F11a redundancy = cdiv(4096, BLOCK_N) × top_k(=8)          -> 128 … 512
```

The `× top_k` factor is specific to the MoE gather and is *not* in the paper: each token
appears `top_k` times in the grouped GEMM's row space, so its sum of squares is
recomputed 8× on top of the n-tile redundancy. Per-config values are recorded in
`rows[*].f11a_w13.sq_redundancy` / `rows[*].f11b_router.sq_redundancy`.

Extra **FLOPs** are small either way — 2 flops per A element against `2·BLOCK_N` for the
matmul, i.e. `1/BLOCK_N` = 0.4–1.6 % — which is exactly the paper's argument for why the
redundancy should hide behind the MMA pipeline. Whether it does is §5.

---

## 3. What was built

`glm52/kernels/lazy_prenorm.py`:

* `moe_gateup_prenorm_kernel` — the w13 grouped GEMM, `FUSE_NORM: tl.constexpr`.
* `router_gemm_kernel` — the dense router GEMM, same flag.
* `rstd_kernel` + `launch_rstd` — the exploratory "half-fused" reduction (§7).
* `USE_RSTD: tl.constexpr` on both GEMMs — the half-fused epilogue (§7).
* `fold_weight_nk` / `fold_weight_rowmajor` — offline weight folding.
* `smem_bytes` — the pre-filter, `num_stages · 2 · BLOCK_K · (BLOCK_M + BLOCK_N)`.

`FUSE_NORM=False` is the unfused GEMM: A is the materialized `x2`, B is the raw weight.
`FUSE_NORM=True` is the same kernel reading the un-normalized `h1` with the folded weight.
The split-out RMSNorm on the unfused side is F3's already-tuned kernel,
`glm52.kernels.add_rmsnorm.norm_only` (`DO_ADD=False, DO_NORM=True`) — reused rather than
rewritten so the baseline is the best norm kernel this study has produced.

### 3.1 `SQ_MODE` — four ways to accumulate the sum of squares

The first version of this kernel implemented the blog's pseudocode literally and lost
badly, so before spending hours of tuning I ran a focused probe
(`scratchpad/f11/sqprobe.py`) that times **the same config, on the same buffers, with
`FUSE_NORM` off vs on**. Because this fusion adds *no extra input tensor*, that comparison
is a **pure instruction-cost measurement** — the exact analogue of F1's stride-0-broadcast
trick, only exact rather than approximate. Four implementations were compared:

| mode | implementation | state |
|---|---|---|
| 0 | `sq += tl.sum(af*af, axis=1)` per k-step — the blog's pseudocode | `[BM]` fp32 |
| 1 | `sqt += af*af`, one reduce after the loop | `[BM,BK]` fp32 |
| 2 | `sqd += tl.dot(a*a, ones[BK,16])` — **sum of squares on the tensor core** | `[BM,16]` fp32 |
| 3 | `a` re-loaded with a second `tl.load`, then mode 0 | `[BM]` fp32 |

**Router GEMM (N=256), instruction cost over the unfused GEMM at the same config:**

| config | unfused | m0 | m1 | m2 | m3 |
|---|---|---|---|---|---|
| T=2048 BM32 BN64 BK64 w4s2 | 214.3 µs | +21.4 % | **+14.1 %** | +101.1 % | +21.3 % |
| T=2048 BM64 BN64 BK64 w4s2 | **165.4 µs** | +42.6 % | **+22.8 %** | +87.0 % | +42.3 % |
| T=2048 BM64 BN128 BK32 w4s2 | 201.7 µs | +67.6 % | +62.2 % | **+58.2 %** | +67.8 % |
| T=2048 BM128 BN128 BK32 w8s2 | 272.1 µs | +68.8 % | **+63.0 %** | SMEM fail | +69.3 % |
| T=8192 BM32 BN64 BK64 w4s2 | 633.3 µs | +26.0 % | +45.7 % | +107.8 % | **+25.9 %** |
| T=8192 BM64 BN64 BK64 w4s2 | **390.1 µs** | +70.9 % | **+48.6 %** | +96.3 % | +70.9 % |
| T=8192 BM64 BN128 BK32 w4s2 | 446.5 µs | +14.3 % | **+12.6 %** | +102.5 % | +14.4 % |
| T=8192 BM128 BN128 BK32 w8s2 | 528.4 µs | +73.0 % | **+67.4 %** | SMEM fail | +73.0 % |

**w13 grouped GEMM (N=4096), T=2048, 16 384 rows, 32 experts:**

| config | unfused | m0 | m1 | m2 | m3 |
|---|---|---|---|---|---|
| BM64 BN128 BK32 w8s2 | 16.266 ms | +66.0 % | +51.3 % | **+18.9 %** | +65.8 % |
| BM128 BN128 BK32 w8s2 | **12.875 ms** | +124.4 % | +116.1 % | **+76.1 %** | +123.6 % |

Three things fall out of this, and they are the core technical findings of this family:

1. **The sum of squares is not free on this backend.** The blog's central claim is that the
   redundant normalization work "is fully overlapped with TensorCore". Here it costs
   **+12 % to +124 %** of the GEMM, for an arithmetic addition of `1/BLOCK_N` = 0.4–1.6 %
   of the flops. The overlap does not happen.
2. **Mode 3 times *identically* to mode 0 to within 0.1 %** (25.9 vs 26.0, 70.9 vs 70.9,
   14.4 vs 14.3 …). Triton CSEs the two identical `tl.load`s, so mode 3 compiles to mode 0
   — it does not isolate the layout question, but it does prove the cost is **not** a
   second memory access. Kept in the kernel as recorded evidence.
3. **Moving the reduction onto the tensor core (mode 2) can be a big lever, but its sign
   flips with `BLOCK_N` *and* with how compute-bound the GEMM is.** It replaces a
   cross-lane reduction over a dot-operand-layout tile with `16/BLOCK_N` extra MMA flops.
   On the router's narrow `BLOCK_N=64` tiles the extra 25 % of MMA work costs far more
   than it saves (+21 % → +101 %). On the w13 probe above it cut the overhead 3.5×
   (+66 % → +19 %) — **but that probe used only 32 experts**, which makes the grouped GEMM
   compute-bound; see the caveat below.

**Caveat on the w13 probe rows.** They use a 32-expert weight (1.5 GB) so the tensor
fits comfortably and the GEMM is compute-bound. The real regime has **256 experts /
12.9 GB**, where the kernel is weight-bandwidth-bound and heavily padded (≈64 real rows
per expert at T=2048 against `BLOCK_M`=64–128), so extra MMA work has *less* to hide behind
and mode 2's extra flops stop paying. The pre-study inside the benchmark runs on the real
256-expert weight and picks accordingly:

```
SQ study [router] over 3 common cfgs: m0 0.837ms, m1 0.784ms, m2 1.064ms, m3 1.033ms -> SQ_MODE=1
SQ study [moe]    over 2 common cfgs: m0 111.06ms, m1 107.87ms, m2 117.97ms, m3 110.16ms -> SQ_MODE=1
```

**`SQ_MODE=1` (tile-accumulate) for both families.** The lesson stands regardless of which
mode wins: the cost is a *code-generation* cost in the mainloop, and the four
implementations only move it around by 10–30 %; none of them makes it disappear.

Because `SQ_MODE` exists only on the fused side, putting it in the tuning grid would make
the fused grid 4× the unfused grid — precisely the auditability defect flagged in the
brief. It is therefore resolved by this **separate, recorded pre-study** at `prefill_t2048`
(`sq_mode_study` in the JSON), compared only over configs where *every* mode compiled
(so a mode is never rewarded for having failed on the slow shapes), and then held **fixed**
for every regime, so the two search grids stay the same size.

Numerics are unaffected by the choice: mode 2 rounds `a*a` to bf16 before the MMA, which
measures as `2.708e-03` vs `2.706e-03` relative error — 2e-6 of extra error, because the
6144 squared terms are all positive and their relative errors average out.

---

## 4. Mapping search space

The coarse-grid generator **takes no `fused` argument**. Unlike F6 — where the fused
kernel staged a second B tile and the SMEM filter therefore legitimately rejected
different configs on each side — the lazy-prenorm kernel stages **no extra tile**, so its
SMEM footprint is identical with the flag on or off. The two coarse grids are literally
the same config list.

```
router GEMM : BLOCK_M ∈ {16,32,64,128}  (capped at next_pow2(T), same cap both sides)
              BLOCK_N ∈ {32,64,128,256} (=N)
              BLOCK_K ∈ {32,64,128}
              num_warps ∈ {4,8}, num_stages ∈ {2,3}, GROUP_M=8
              filters: smem ≤ 64 KB; 2 ≤ BLOCK_M·BLOCK_N/(warps·64) ≤ 128
              -> 30 cfgs (T=1), 64 (T=32), 116 (T≥256)

w13 GEMM    : T<2048  : BM ∈ {16,32,64,128}, BN ∈ {64,128,256}, BK ∈ {32,64,128},
                        warps ∈ {4,8,16}, stages ∈ {2,3}      -> 99 cfgs
              T≥2048  : BM ∈ {32,64,128},    BN ∈ {64,128,256}, BK ∈ {32,64,128},
                        warps ∈ {4,8,16}, stages ∈ {2,3}      -> 79 cfgs
              filters: smem ≤ 64 KB; 4 ≤ acc/lane ≤ 128
```

The SMEM pre-filter `num_stages · 2 · BLOCK_K · (BLOCK_M + BLOCK_N)` **under-estimates**
what this MACA backend actually allocates: `BM128 BN256 BK32 s2` is predicted at 49 152 B
but Triton reports `Required: 81920`. The filter therefore lets a few configs through that
fail at launch; `autotune` records them as failures rather than aborting, and the filter is
applied identically to both sides, so this costs the comparison nothing.

Refinement uses the same rule on both sides (half/same/double in BM, BN and warps at the
winning BK/stages; a BK × stages sweep; a GROUP_M ∈ {1,4,8,16} sweep) centred on **each
side's own coarse winner**, so refine counts can differ by a few configs. Every count is
recorded per regime in `rows[*].grid_sizes` and every table in `tune_tables`.

**Extras handed to the unfused (baseline) side, which can only help it:**

1. an independent search over the RMSNorm kernel's own 152-config space (F3's grid);
2. a **joint chain re-tune** over top-3 GEMM × top-3 norm configs, timed as the real
   two-kernel chain with a single L2 flush — guarding against an independently-tuned
   optimum that is not the joint optimum.

Neither extra is available to the fused side, because the fused side is one kernel.

---

## 5. Results

All times are p50 ms over 100–400 reps (20 at prefill), one L2 flush before each repetition
of the whole chain, none between a chain's kernels. Every row validated first (§6).

### 5.1 Headline

| regime | **F11b router** | ceiling | % of headroom | **F11a w13** | ceiling | **combined** |
|---|---|---|---|---|---|---|
| decode_bs1 | 0.684× | 1.01× | — | 0.902× | 1.00× | 0.828× |
| decode_bs32 | 0.680× | 1.22× | — | 0.961× | 1.00× | 0.955× |
| decode_bs256 | 0.771× | 1.64× | — | 0.965× | 1.00× | 0.961× |
| prefill_t2048 | **1.096×** | 1.64× | 15 % | **0.476×** | 1.00× | 0.478× |
| prefill_t8192 | **1.127×** | 1.64× | 20 % | **0.603×** | 1.01× | 0.605× |

"% of headroom" = `(measured − 1)/(ceiling − 1)`; it is undefined for the regressions.
No measured speedup exceeds its ceiling anywhere, which is the sanity check the brief asks
for.

### 5.2 F11b — router GEMM, the case the fusion was supposed to win

| regime | norm kernel | unfused GEMM | unfused chain | fused | speedup | fused BLOCK_N | n-tiles = redundancy |
|---|---|---|---|---|---|---|---|
| decode_bs1 | 0.0202 | 0.0635 | 0.0699 | 0.1021 | 0.684× | 32 | 8× |
| decode_bs32 | 0.0197 | 0.0627 | 0.0696 | 0.1024 | 0.680× | 32 | 8× |
| decode_bs256 | 0.0276 | 0.0681 | 0.0819 | 0.1062 | 0.771× | 32 | 8× |
| prefill_t2048 | 0.0696 | 0.1618 | 0.2161 | 0.1971 | **1.096×** | 64 | 4× |
| prefill_t8192 | 0.2084 | 0.3748 | 0.5660 | 0.5020 | **1.127×** | 128 | **2×** |

Winning configs (fused / unfused GEMM):

```
decode_bs1    BM16 BN32  BK128 w4 s3 GM8    /  BM16 BN32  BK128 w4 s3 GM8
decode_bs32   BM16 BN32  BK128 w4 s3 GM8    /  BM16 BN32  BK128 w4 s3 GM8
decode_bs256  BM16 BN32  BK128 w4 s3 GM16   /  BM32 BN32  BK128 w4 s3 GM4
prefill_2048  BM64 BN64  BK64  w4 s3 GM8    /  BM64 BN64  BK64  w4 s4 GM8
prefill_8192  BM64 BN128 BK32  w4 s2 GM8    /  BM64 BN64  BK64  w4 s3 GM4
```

Note the **mapping optimum moves** at t8192: the fused kernel abandons `BLOCK_N=64` (which
the unfused kernel prefers) for `BLOCK_N=128`, halving the sum-of-squares redundancy from
4× to 2× at the price of a less-favourable GEMM tile. That is the fusion paying for its own
redundancy in the mapping, and it is visible in the matched-config table below: at
`BM64 BN64 BK64` the fused kernel is 1.50× the unfused time, but at `BM64 BN128 BK32` only
1.12×.

### 5.3 F11a — w13 grouped GEMM, the case with no headroom

| regime | unfused GEMM | unfused chain | fused | speedup | fused TF/s | unfused TF/s | redundancy (n-tile × top_k) |
|---|---|---|---|---|---|---|---|
| decode_bs1 | 0.278 | 0.284 | 0.315 | 0.902× | 1.3 | 1.4 | 64 × 8 = **512×** |
| decode_bs32 | 5.044 | 5.053 | 5.258 | 0.961× | 2.5 | 2.6 | 64 × 8 = **512×** |
| decode_bs256 | 8.154 | 8.180 | 8.472 | 0.965× | 12.2 | 12.6 | 64 × 8 = **512×** |
| prefill_t2048 | 13.829 | 13.885 | 29.159 | **0.476×** | 28.3 | 59.6 | 32 × 8 = **256×** |
| prefill_t8192 | 38.031 | 38.217 | 63.340 | **0.603×** | 52.1 | 86.7 | 32 × 8 = **256×** |

### 5.4 Attribution — why the fused kernel loses

Three independent measurements, all in the JSON.

**(a) Isolation: same config, same buffers, `FUSE_NORM` on vs off.** This fusion adds no
extra input tensor, so this is an exact instruction-cost measurement — no DRAM-traffic
confound to subtract (F1 needed a stride-0 broadcast to approximate what is exact here).

| regime | router | w13 |
|---|---|---|
| decode_bs1 | +61.7 % | +12.5 % |
| decode_bs32 | +63.0 % | +1.2 % |
| decode_bs256 | +17.8 % | +1.0 % |
| prefill_t2048 | +21.5 % | **+67.7 %** |
| prefill_t8192 | +12.4 % | **+65.0 %** |

The w13 pattern is decisive: **the fusion is nearly free (+1 %) exactly where the GEMM is
memory-bound, and costs +65–68 % exactly where the GEMM is compute-bound and its mainloop
is running at speed.** That is the same shape as F1's codegen cliff — the extra work does
not overlap with the MMA pipeline, it *displaces* it.

**(b) Matched-config sweep** (both variants at the identical mapping, from the recorded
tune tables) — the fused side is not merely differently tuned, it is uniformly slower:

`prefill_t8192`, w13:

| config | unfused | fused | ratio |
|---|---|---|---|
| BM128 BN128 BK64 w8 s2 GM8 | 39.05 | 65.13 | 1.67× |
| BM64 BN128 BK64 w16 s2 GM8 | 47.63 | 81.29 | 1.71× |
| BM64 BN64 BK64 w16 s2 GM8 | 54.31 | 112.39 | 2.07× |

`prefill_t8192`, router:

| config | unfused | fused | ratio |
|---|---|---|---|
| BM64 BN64 BK64 w4 s3 GM8 | 375.8 µs | 562.4 µs | 1.50× |
| BM64 BN128 BK32 w4 s2 GM8 | 448.3 µs | 503.8 µs | **1.12×** |

**(c) Registers and SMEM, cache cleared between compiles.** No spills anywhere.

| regime | kernel | n_regs | n_spills | SMEM |
|---|---|---|---|---|
| prefill_t2048 | w13 fused (BM64 BN128 BK64 w8 s2) | 122 | 0 | 40 960 |
| prefill_t2048 | w13 unfused, **same config** | 96 | 0 | 49 152 |
| prefill_t8192 | w13 fused (BM128 BN128 BK64 w8 s2) | 172 | 0 | 49 152 |
| prefill_t8192 | w13 unfused, **same config** | 144 | 0 | 65 536 |
| prefill_t8192 | router fused (BM64 BN128 BK32 w4 s2) | 156 | 0 | 20 480 |
| prefill_t8192 | router unfused, same config | 150 | 0 | 24 576 |

Registers rise by **+26 to +28** per thread with the fusion on. At `prefill_t2048` that is
122 × 512 = 62 464 vs 96 × 512 = 49 152 registers per CTA against C500's 131 072/SM —
**both still fit 2 CTAs/SM**, so unlike F6 this is *not* an occupancy collapse. There are
no spills. The cost is in the instruction stream, not in the resource budget.

**(d) A hard SMEM bar, exactly as in F6 — but a smaller one.** 7 of the 79 coarse configs
compile for the unfused kernel and **fail for the fused kernel**, all of them at
`num_warps=16` or `BLOCK_N=256`:

```
BM128 BN128 BK64 w16 s2  ->  OutOfResources: Required 69632, limit 65536   (unfused: OK)
BM128 BN128 BK32 w16 s2/3->  Required 69632
BM128 BN256 BK32 w8/w16 s2-> Required 67584
BM64  BN256 BK32 w16 s2/3->  Required 67584
```

The overflow is exactly **+4096 B over the unfused kernel** — scratch that Triton allocates
for the cross-warp reduction of the sum of squares. And `BM128 BN128 BK64 **w16** s2` is
precisely the tile family of the **unfused winner at `prefill_t8192`**
(`BM128 BN128 BK64 w16 s2 GM1`, 38.15 ms); the fused kernel's best is the same tile with
`w8` (63.18 ms). So, as in F6, **the fused kernel is structurally barred from the mapping
that makes the unfused kernel fast** — though here the bar accounts for only part of the
gap (the fused kernel is already 1.67× slower at the matched `w8` config).

**Grid-size accounting (fairness rule 2).** Coarse grids are identical by construction —
`router_coarse_fused == router_coarse_unfused` and `moe_coarse_fused == moe_coarse_unfused`
at every regime (30/30, 64/64, 116/116; 99/99, 79/79). Refine grids differ by a few configs
because they are the same neighbourhood rule centred on each side's own winner
(e.g. t8192: router 32 vs 27, w13 18 vs 14). The **only** asymmetry in *valid* configs is
the SMEM bar in (d), which is a real property of the fused kernel and is reported as such,
not hidden. Full per-regime counts are in `rows[*].grid_sizes`; full tables (148/143 router
entries, 97/93 w13 entries at t8192) are in `tune_tables`.

### 5.5 Vendor-BLAS reference lines

| regime | router: vendor bf16 | vendor fp32 | our unfused Triton | our fused Triton |
|---|---|---|---|---|
| decode_bs256 | 0.0445 | 0.1546 | 0.0681 | 0.1062 |
| prefill_t2048 | 0.0776 | 0.4145 | 0.1618 | 0.1971 |
| prefill_t8192 | 0.2004 | 1.1433 | 0.3748 | 0.5020 |

| regime | w13: vendor grouped loop | vendor dense 1-expert | (dense TF/s) | our unfused | our fused |
|---|---|---|---|---|---|
| decode_bs256 | 16.46 | 0.129 | 99.7 | 8.15 | 8.47 |
| prefill_t2048 | 20.10 | 0.513 | 200.8 | 13.83 | 29.16 |
| prefill_t8192 | 38.38 | 1.845 | 223.5 | 38.03 | 63.34 |

Notes:

* the vendor BLAS reaches **200–224 TF/s** on the dense-equivalent shape, confirming the
  ~215 TF/s figure in `traffic.py`; the best Triton grouped GEMM here reaches 86.7 TF/s,
  limited by 256-expert weight traffic and by `BLOCK_M` padding (~64 real rows per expert
  at T=2048), not by the fusion;
* the vendor's **per-expert `torch.matmul` loop** (rows pre-gathered outside the timed
  region) is *slower* than our unfused Triton grouped GEMM at t2048 (20.10 vs 13.83 ms) and
  level at t8192 — 256 small GEMMs do not amortise;
* running the router in **true fp32 matmul** (the literal `moe_router_dtype: float32`
  semantics) costs 2.9–5.7× the bf16-input/fp32-accumulate path, which is why both sides
  here take bf16 inputs with an fp32 accumulator and an fp32 output.

### 5.6 Launch-overhead check

At decode the chains are launch-bound, so the `noflush` medians are reported too. They do
**not** rescue the fused side: e.g. `decode_bs32` router, flush 0.1024 vs 0.0696, noflush
0.0988 vs 0.0945. Removing a kernel launch is worth ~5 µs here and the fusion costs ~40 µs.

---

## 6. Correctness

Everything is validated against an fp32 reference before anything is timed; the w13 GEMM is
checked on a 1024-row random sample (a full fp32 reference at T=8192 would be 3.3 TFLOP of
fp32 matmul). Tolerance 2e-2 on the max-abs relative error; all checks pass.

| regime | router fused | router unfused | w13 fused | w13 unfused | half-fused router | half-fused w13 |
|---|---|---|---|---|---|---|
| decode_bs1 | 3.34e-03 | 2.58e-06 | 3.66e-03 | 2.37e-03 | 3.34e-03 | 3.66e-03 |
| decode_bs32 | 2.46e-03 | 3.02e-06 | 4.58e-03 | 2.00e-03 | 2.46e-03 | 4.58e-03 |
| decode_bs256 | 3.30e-03 | 3.08e-06 | 4.24e-03 | 3.35e-03 | 3.30e-03 | 4.24e-03 |
| prefill_t2048 | 3.10e-03 | 3.82e-04 | 4.29e-03 | 1.94e-03 | 3.10e-03 | 4.29e-03 |
| prefill_t8192 | 2.96e-03 | 5.32e-04 | 4.07e-03 | 1.86e-03 | 2.96e-03 | 4.07e-03 |

### 6.1 The fused path is *more* accurate, not less

The 3e-03 in the "fused" column is **not** error introduced by the fusion — it is error the
*unfused* path introduces and the fused path avoids. The reference `rmsnorm` rounds
`x * rstd` to bf16 before multiplying by `w`, so the unfused GEMM consumes a bf16-rounded
`x2`. The fused GEMM never materializes `x2` and applies `rstd` in fp32 in the epilogue, so
it *skips* that rounding. Measured against an **exact fp32** reference
(`(h1 * rstd * w) @ B` computed entirely in fp32, no intermediate rounding):

| regime | fused vs exact fp32 | unfused vs exact fp32 |
|---|---|---|
| decode_bs1 | **2.18e-03** | 2.70e-03 |
| decode_bs32 | **1.63e-03** | 2.03e-03 |
| decode_bs256 | **1.47e-03** | 2.53e-03 |
| prefill_t2048 | **1.50e-03** | 2.52e-03 |
| prefill_t8192 | **1.70e-03** | 2.57e-03 |

The fused kernel is closer to the truth at every regime. The offline weight fold itself is
exact to the same order (`router 3.26e-03`, `w13 2.88e-03` against the unfolded
`reference.rmsnorm` + matmul, both dominated by the same bf16 rounding of `x2`).

### 6.2 …but it changes routing decisions on near-ties

Being more accurate is still *different*, and the router's output feeds a top-k:

| regime | top-8 expert-set agreement, fused vs unfused |
|---|---|
| decode_bs1 / bs32 | 100.00 % |
| decode_bs256 | 98.73 % |
| prefill_t2048 | 98.82 % |
| prefill_t8192 | 98.85 % |

~1.2 % of tokens select a different 8th expert. With sigmoid scoring and 256 experts,
near-ties at the top-k boundary are common and *any* numerics change flips some of them —
the unfused kernel would do the same against a differently-ordered reduction. It is not a
correctness failure, but it does mean **F11b is not a bit-exact drop-in for the router**,
and that is worth stating before anyone ships it.

### 6.3 A note on the split-out norm kernel

`check(x2_kernel, x2_reference)` is exactly **0.00e+00** at all three decode regimes (the
tuner picks the one-shot `BLOCK_N=8192` mapping, whose reduction order matches torch) and
2.4e-03/2.6e-03 at prefill (the tuner picks a multi-pass `BLOCK_N=1024/2048` mapping, whose
different fp32 summation order moves `rstd` by ~1 bf16 ulp). Both are within tolerance and
this is the *baseline's* variation, not the fusion's.

---

## 7. Exploratory third variant: "half-fused" (rstd kernel + epilogue scale)

Once the isolation measurement showed the cost is the *in-mainloop* sum of squares and not
the epilogue, an obvious third point on the design axis appeared: compute `rstd` in a tiny
separate reduction kernel (`[T,6144] bf16 -> [T] fp32`), leave the GEMM's k-loop
**byte-for-byte identical to the unfused one**, and apply `rstd` as a pure epilogue scale
(`USE_RSTD=True`). Activation passes:

```
unfused      norm kernel reads act, writes act ; GEMM reads act   -> 3 act
half-fused   rstd kernel reads act             ; GEMM reads act   -> 2 act
lazy prenorm                                     GEMM reads act   -> 1 act
```

It captures 2/3 of the byte saving with none of the mainloop cost. **This is labelled
exploratory and is NOT the headline** — it is tuned only over the top-3 configs from each
side's completed search, plus its own 106-config rstd-kernel grid, so it has had less
search than either headline variant.

| regime | rstd kernel | router | vs unfused | w13 | vs unfused | combined | vs unfused |
|---|---|---|---|---|---|---|---|
| decode_bs1 | 0.0174 | 0.0676 | **1.034×** | 0.2806 | 1.014× | 0.3308 | 1.011× |
| decode_bs32 | 0.0174 | 0.0676 | **1.030×** | 5.0637 | 0.998× | 5.1154 | 0.998× |
| decode_bs256 | 0.0197 | 0.0745 | **1.100×** | 8.1948 | 0.998× | 8.2481 | 0.998× |
| prefill_t2048 | 0.0356 | 0.1825 | **1.184×** | 14.5636 | 0.953× | 14.6916 | 0.955× |
| prefill_t8192 | 0.0791 | 0.4570 | **1.239×** | 37.9635 | 1.007× | 38.4069 | 1.004× |

**The half-fused router beats the fully-fused router at every single regime** — 1.239× vs
1.127× at t8192, and 1.03–1.10× vs 0.68–0.77× at decode where full fusion regresses.

And it is *at its own roofline*. Its ceiling, computed the same latency-aware way
(`rstd` kernel = one activation read instead of the norm kernel's read+write):

| regime | half-fused ceiling | measured | % of ceiling |
|---|---|---|---|
| prefill_t2048 | 1.241× | 1.184× | **95 %** |
| prefill_t8192 | 1.241× | 1.239× | **100 %** |

So the honest summary of this family is: **the byte saving that Lazy Pre-Norm chases is
real and fully collectable on this hardware — but only the two-thirds of it that can be had
without touching the GEMM's mainloop.** The last third costs more in mainloop instructions
than it saves in DRAM traffic.

The half-fused w13 loses 4.7 % at t2048 (0.953×) — that is the F1 effect again: adding *any*
epilogue (here a single `[BLOCK_M]` fp32 load and a multiply) to a fast compute-bound
grouped GEMM costs a few percent of the mainloop schedule, and at t2048 the w13 GEMM has
essentially no byte saving to pay for it.

---

## 8. Surprises

1. **The paper's central claim does not hold on this backend.** The blog argues the
   redundant normalization "is fully overlapped with TensorCore" and that the redundancy is
   therefore "acceptable". Here an arithmetic addition of 0.4–1.6 % of the flops costs
   **+12 % to +68 %** of runtime, and it costs *most* exactly where the MMA pipeline is
   busiest. The blog's kernels use **warp specialization** (its own diagram splits the
   reduction onto dedicated warps) on Blackwell; Triton 3.0 on MACA has no warp
   specialization, no TMA, no clusters, so the "free" lane the algorithm needs does not
   exist. This is a portability finding about the *technique*, not a bug in the port.
2. **The reduction costs shared memory, not just registers.** The fused kernel needs
   exactly 4 096 B more SMEM than the unfused one at the same tile, which is what pushes 7
   configs — including the unfused winner's `num_warps=16` family — over C500's 64 KB
   ceiling. Same failure mode as F6, different mechanism (F6 doubled the *staged B tile*;
   here it is cross-warp reduction scratch).
3. **Four different sum-of-squares implementations move the cost by 10–30 % and none of
   them removes it**, including putting the reduction on the tensor core. Which one wins
   flips with `BLOCK_N` and with whether the GEMM is compute- or bandwidth-bound.
4. **Triton CSEs two identical `tl.load`s**, so the "load A twice" variant compiles to the
   single-load variant and times identically to 0.1 %. Worth knowing before designing any
   experiment that tries to separate a load from its layout.
5. **The fusion improves accuracy.** It skips the bf16 rounding of `x2` that the framework
   path performs, and lands 1.5e-03 from an exact fp32 reference where the unfused path
   lands 2.5e-03 — while flipping ~1.2 % of top-8 routing decisions on near-ties.
6. **The affine weight really is a non-issue at inference.** The blog lists it as
   limitation #1; folding it into `B`'s rows offline costs nothing at runtime and validates
   to 2.9e-03 (i.e. to the same bf16 rounding everything else is at). Anyone reading the
   blog should not treat elementwise affine as a blocker for an inference kernel.

---

## 9. Verdict

**F11a (Lazy Pre-Norm → w13 grouped GEMM): do not ship. Clear, large regression.**
0.90–0.97× at decode and **0.48–0.60× at prefill**, against a roofline ceiling of
**1.00–1.01×** — i.e. the fusion had *no headroom to begin with* (expert-weight traffic is
0.37–12.9 GB against the 0.02–192 MB the fusion removes) and then paid +65 % of mainloop
instructions for it, plus a 4 KB SMEM overflow that bars it from the unfused winner's
mapping. The redundancy factor is `cdiv(4096, BLOCK_N) × top_k` = **256–512×**, which is
the regime the brief warned about, and it does not hide behind the MMA pipeline.
This is a third distinct toxic-fusion mechanism for the study: F1 = codegen cliff with
identical registers, F6 = register pressure + SMEM bar, F11a = **mainloop instruction cost
that scales with how compute-bound the GEMM is**, plus a smaller SMEM bar.

**F11b (Lazy Pre-Norm → router GEMM): a genuine but small prefill win, and a decode loss.**
**1.096×** at t2048 and **1.127×** at t8192 (15 % / 20 % of the 1.64× headroom), and
**0.68–0.77×** at decode. Redundancy here is only 2–8×, which is why it can win at all.
Ship it *only* for prefill, and only if the ~1.2 % top-k flip rate is acceptable.

**The thing actually worth shipping is the half-fused variant** (§7): `rstd` from a tiny
reduction kernel, applied as an epilogue scale on an otherwise-untouched GEMM. It beats
full lazy pre-norm on the router at **every** regime (1.03–1.24×), never regresses on the
router, and reaches **95–100 % of its own roofline ceiling**. It requires the same offline
weight fold and has the same numerics as full fusion. Given that `x2` is dead only if
*every* consumer is fused, the half-fused variant also composes far better: one `rstd`
kernel serves the router, the routed experts and the shared expert, each of which then just
scales its accumulator.

**Combined, end-to-end** (one norm kernel serving both consumers, `x2` never materialized),
full lazy pre-norm gives 0.83× / 0.96× / 0.96× / 0.48× / 0.61× — a regression at every
regime, dominated by F11a. The half-fused combined chain gives 1.011× / 0.998× / 0.998× /
0.955× / 1.004×: break-even. **On C500 with Triton 3.0, normalization is not free, and the
closest you can get to free is to stop trying to put it inside the k-loop.**

### 9.1 What would change the answer

* **Warp specialization.** The algorithm is designed for a backend that can put the
  reduction on its own warps. Triton 3.0/MACA cannot. If a future MACA Triton exposes it,
  F11b's remaining 80 % of headroom is the thing to re-measure.
* **Split-K for the decode router.** Both sides of F11b run at 43–60 GB/s at decode
  because the router GEMM at N=256 offers only 8–128 CTAs. Neither side was given split-K
  (it needs either a zeroing launch or a reduction launch, which would dominate at these
  sizes), so the decode rows measure a badly-parallelised GEMM fairly rather than a
  well-parallelised one. The ceiling there is 1.01–1.64× and the fusion loses by 30 %, so
  split-K would not flip the sign, but the absolute numbers are not production-grade.
* **A wider `BLOCK_N` for w13.** The fused kernel's redundancy is `4096/BLOCK_N × 8`;
  `BLOCK_N=256` would halve it, but that is exactly one of the tiles the +4 KB SMEM
  overflow rejects.
