# LOG-05 — Fusions #8 (Down GEMM + Expert Merge) and #9 (+ ResAdd2)

GPU: MetaX C500, `CUDA_VISIBLE_DEVICES=0` (exclusive).
Result id: `f08f09_down_merge_resadd` → `results/f08f09_down_merge_resadd.json`.
Kernels: `glm52/kernels/moe_down_merge.py`. Bench: `glm52/bench/bench_f08f09_down_merge_resadd.py`.

**Verdict up front: fusion #8 is not worth doing on this hardware, and #9 is free but rides
on #8. Neither accumulation strategy beats the sglang-style unfused chain except at
`decode_bs1`. Details and numbers in §5–§7.**

---

## 1. What is being fused, and what sglang actually does today

The MoE layer's second expert GEMM ("down projection") on GLM-5.2:

```
A = act  [rows = T*8, I = 2048]   bf16     (SwiGLU output, one row per (token, k) pair)
B = w2   [E = 256, H = 6144, I = 2048] bf16
=> GEMM  N = H = 6144,  K = I = 2048,  M = rows
```

sglang 0.5.10's production path (`fused_moe.py` ~line 587, `fused_moe_triton_kernels.py` line 310):

```python
invoke_fused_moe_kernel(intermediate_cache2, w2, intermediate_cache3,
                        ..., mul_routed_weight=True, top_k=1)
#  -> intermediate_cache3 : [T, 8, 6144]     routing weight already applied
moe_sum_reduce(intermediate_cache3, out)     #  a SEPARATE reduction kernel
```

so the `[T, 8, 6144]` tensor is **fully materialised and immediately re-read**. sglang does
carry an atomic merge (`FUSE_SUM_ALL_REDUCE`, kernel line 607 — `tl.atomic_add` into
`c[offs_token // ROUTER_TOPK]` with `out_slice.zero_()` before it), but it is gated behind
`--enable-fused-moe-sum-all-reduce` and is only used jointly with a fused all-reduce. So
the merge is **not** fused by default, and #8 is exactly the question "should it be?".

Fusion #9 adds the post-MoE residual add on top.

### The three baselines and four fused variants

| | chain | output |
|---|---|---|
| `unfused8` | down GEMM (`FUSE_MERGE=False`) → `[T,8,H]`; `moe_sum_kernel` | `out [T,H]` |
| `unfused9_3kernel` | + a separate `resadd_kernel` (**primary #9 baseline**) | `out [T,H]` |
| `unfused9_2kernel` | down GEMM; `moe_sum_kernel(ADD_RESIDUAL=True)` (bonus, "cheap fix") | `out [T,H]` |
| `f8_atomic` | seed `out` with **zeros**; down GEMM `FUSE_MERGE=True` → `tl.atomic_add` | `out [T,H]` |
| `f8_token_major` | one CTA per (token, n-block), loops the token's 8 experts in registers | `out [T,H]` |
| `f9_atomic` | seed `out` with the **residual**; same atomic GEMM | `out [T,H]` |
| `f9_token_major` | token-major + one residual load in the epilogue | `out [T,H]` |

All seven produce the identical bf16 `[T, 6144]` tensor the next layer consumes. The
seed/zeroing kernel is **inside** the fused chain's timing — it is not free and is not
excluded.

### Fairness (rule 1: one kernel source, flags differ)

`moe_down_kernel` is written once. `FUSE_MERGE` selects only the epilogue:

```python
if FUSE_MERGE:
    rows_out = safe_token // top_k
    tl.atomic_add(c_ptr + stride_cm*rows_out[:,None] + stride_cn*offs_cn[None,:], out, mask=c_mask)
else:
    tl.store(c_ptr + stride_cm*safe_token[:,None] + stride_cn*offs_cn[None,:], out, mask=c_mask)
```

Everything before that — the `sorted_token_ids` / `expert_ids` / `num_tokens_post_padded`
dispatch, the gather, the K-loop, `even_Ks`, the `GROUP_SIZE_M` swizzle, the
`MUL_ROUTED_WEIGHT` multiply — is byte-for-byte the same code for both flag values, and is
a direct mirror of sglang's `fused_moe_kernel`.

The **token-major** variant is a different *grid order and loop order*, which rule 1
explicitly permits, but it is not literally the same function: with one token per CTA the
M-tile is 1 row, so the inner product cannot be a `tl.dot` over real data. The kernel
therefore offers a `USE_DOT` mapping knob — a padded `tl.dot` with `BLOCK_M=16` and 15
rows masked to zero, versus a broadcast/reduce GEMV with a single `[BLOCK_K, BLOCK_N]`
fp32 register accumulator — and the tuner picks. **This is stated as a caveat, not hidden**:
strategy (b) is a genuinely different mapping, and that is the experiment.

---

## 2. Memory-traffic analysis (the prediction, written before measuring)

`H=6144`, `I=2048`, `topk=8`, bf16 = 2 B. `rows = 8T`. Per-token byte counts, excluding
the weight stream:

| chain | A read | intermediate | out / residual | **total B/token** |
|---|---|---|---|---|
| `unfused8` | 32,768 | 98,304 w + 98,304 r | 12,288 w | **241,664** |
| `unfused9_3kernel` | 32,768 | 196,608 | 12,288 w + 12,288 r + 12,288 r + 12,288 w | **278,528** |
| `unfused9_2kernel` | 32,768 | 196,608 | 12,288 r + 12,288 w | **253,952** |
| `f8_atomic` | 32,768 | — | 12,288 seed w + 98,304 atomic w + 98,304 atomic r | **241,664** |
| `f9_atomic` | 32,768 | — | 12,288 r + 12,288 w + 196,608 RMW | **253,952** |
| `f8_token_major` | 32,768 | — | 12,288 w | **45,056** |
| `f9_token_major` | 32,768 | — | 12,288 r + 12,288 w | **57,344** |

Two things fall out of this table immediately:

**(a) The atomic variant moves exactly as many bytes as the unfused chain.** It does not
delete traffic; it *relocates* it — a streaming write+read of a 98 KB/token buffer becomes
a read-modify-write of a 12 KB/token buffer. The only way it can win is if `out [T, 6144]`
(= 12,288·T bytes) is **L2-resident**. C500 L2 is 8 MB, so `out` fits for `T ≤ 682`. The
prediction is therefore: atomic helps at decode, and is a wash or a loss at prefill.

**(b) Token-major deletes 5.4× of the non-weight traffic** (241,664 → 45,056 B/token) — but
it re-reads a full `w2[e]` per (token, expert) pair. That is the whole story, and the
weight stream dominates everything:

| regime | rows | expert-major row-blocks | expert-major weight bytes | token-major weight bytes | ratio |
|---|---|---|---|---|---|
| decode_bs1 | 8 | 8 (BM=16) | 0.20 GB | 0.20 GB | **1.0×** |
| decode_bs32 | 256 | ~256 (BM=16) | 6.4 GB | 6.4 GB | **~1.0×** |
| decode_bs256 | 2048 | ~256 (BM=16) | 6.4 GB | 51.5 GB | **8×** |
| prefill_t2048 | 16384 | ~256 (BM=64) | 6.4 GB | 412 GB | **64×** |
| prefill_t8192 | 65536 | ~512 (BM=128) | 12.9 GB | 1650 GB | **128×** |

(`w2[e]` is 6144·2048·2 = 25.17 MB; expert-major reads it once per row-block, summed over
the `pid_n` grid; token-major reads it once per (token, expert) pair.)

At decode the weight stream utterly dominates: at `decode_bs32` the GEMM is 6.4 GB of
weights (≈ 6.1 ms at 1.05 TB/s) against 6.4 GFLOP of math (≈ 0.06 ms at 106 TF/s) —
**100× memory bound**, and the `[T,8,H]` intermediate is 6.3 MB, i.e. **0.1 % of the
traffic**. So the honest prediction for #8 at decode is *no measurable win from deleting
the intermediate*, with any observed difference coming from launch overhead and L2 effects.

At prefill the intermediate matters more: at `prefill_t8192` it is 1.61 GB against a
12.9 GB weight stream and 1.65 TFLOP of math — worth roughly **8–10 %** of the down GEMM.

And token-major should be at parity at T=1/T=32 (where each expert is touched once anyway)
and lose by 1–2 orders of magnitude from T=256 up. The bench probes token-major once
before tuning it and shrinks its grid when a launch exceeds 200 ms, so prefill token-major
is measured, not extrapolated — it just gets a coarse (8-config) search, because it loses
there by 64–128× in bytes, not by mistuning.

---

## 3. Tuning protocol

Per regime, per side, two-stage: a coarse grid, then a neighbourhood refine around the
coarse winner. Grid sizes (after SMEM and accumulator-register prefiltering):

| kernel | coarse | refine | notes |
|---|---|---|---|
| `moe_down_kernel` FUSE_MERGE=False | 126 (decode) / 78 (prefill) | ~25 | |
| `moe_down_kernel` FUSE_MERGE=True | 126 / 78 | ~25 | **identical generator** |
| `moe_down_token_major_kernel` #8 | 82 | ~21 | probe-shrunk at prefill |
| `moe_down_token_major_kernel` #9 | top-8 of #8 | ~21 | see caveat below |
| `moe_sum_kernel` ADD_RESIDUAL=False | 126 | — | |
| `moe_sum_kernel` ADD_RESIDUAL=True | top-10 of the above | — | see caveat below |
| `resadd_kernel` | 90 | — | |
| `seed_kernel` | 14 + a `torch` pseudo-config | — | `.zero_()` / `.copy_()` |

Prefilters: `num_stages · 2 B · BLOCK_K · (BLOCK_M + BLOCK_N) ≤ 65536`, and fp32
accumulator registers per lane `BLOCK_M·BLOCK_N/(num_warps·64) ∈ [4, 64]` (warp = **64**
lanes on C500). The prefill (`big`) grid is a strict subset of the decode grid, so prefill
regimes reuse the decode compile cache.

After the per-kernel searches, every multi-kernel chain is **re-timed jointly** over the
top-k × top-k combinations of its members, and the best chain is what gets reported. That
guards against a separately-tuned optimum that is not the joint optimum, in both
directions.

### Honest caveats on the search

* `moe_sum_kernel(ADD_RESIDUAL=True)` and `seed_kernel(FROM_RESIDUAL=True)` are searched
  over the top-10 / top-6 shortlist of their `False` twin rather than the full grid. Both
  are one extra load on an identical mapping. The `moe_sum` one is on the **unfused** side,
  so the shortlist can only hurt the baseline — i.e. it is conservative in the wrong
  direction for me, and I note it rather than hide it.
* `moe_down_token_major_kernel(ADD_RESIDUAL=True)` is likewise searched over the top-8 of
  the `False` variant plus a refine. That is on the **fused** side, so it could under-sell
  the fusion; the measured #9-vs-#8 token-major gap (≈ the cost of one extra `[T,H]` read)
  is the check that it did not.
* Why shortlists at all: each fresh Triton config costs a **~4 s compile** on this MACA
  backend (measured: 3.6–4.4 s for *every* kernel here, from the 4-line `seed_kernel` to
  the grouped GEMM — it is a fixed backend cost, not a function of kernel complexity). A
  naive full-cross-product search would have been >4 h of pure compilation.
* `do_not_specialize` is applied to the token-count arguments of all five kernels
  (`EM`, `num_valid_tokens`, `T`, `token_num`, `M`, …). They are used only in comparisons
  and `cdiv`, never in address arithmetic, so Triton's divisible-by-16 / equal-to-1 hints
  buy nothing — but without it, every regime (`rows` = 8, 256, 2048, 16384, 65536) is a
  fresh specialization and therefore a fresh 4 s compile of every config in the grid.
  Applied identically to every kernel, so it cannot favour either side.

---

## 4. Hardware probes done before building the grids

| probe | result |
|---|---|
| `tl.atomic_add` fp32, 2-D tile + mask | works |
| `tl.atomic_add` **bf16**, 2-D tile + mask | **works** (so the atomic variant can write bf16 directly, matching sglang's `FUSE_SUM_ALL_REDUCE`, with no fp32 staging buffer and no extra cast kernel) |
| `tl.atomic_add` fp16 | works |
| `tl.dot` with `BLOCK_M=16`, 15 rows masked | works, 64 regs / 10 KB SMEM |
| broadcast/reduce GEMV (`tl.sum(a[:,None]*b, 0)`) | works, 48 regs / 128 B SMEM |
| Triton compile time, any kernel | **3.6–4.4 s** cold, ~0 warm (disk cache) |

The bf16 atomic result is what makes strategy (a) viable at all: an fp32 accumulation
buffer would have needed a `[T, 6144]` fp32 seed (2× the write) plus a separate cast
kernel (read 4 B + write 2 B per element), which would have swamped the fusion benefit.
The cost is precision — 8 contributions each rounded to bf16 — quantified in the results
table below.

---

## 5. Results

Device: MetaX C500, warp = 64, 104 CUs, 64 KB SMEM/CTA, 8 MB L2. torch 2.8.0+metax3.7.1.3,
triton 3.0.0. All timings p50 over 100 reps (10 at prefill), L2 flushed once before each
repetition of the whole chain. **Zero compile failures** across all 2 906 config trials —
the SMEM/accumulator prefilter was doing its job.

### 5.1 Component timings (ms)

| regime | T | rows | GEMM (store) | GEMM (atomic) | `moe_sum` | `resadd` | seed | **unfused8** | **unfused9 (3k)** | unfused9 (2k) |
|---|---|---|---|---|---|---|---|---|---|---|
| decode_bs1 | 1 | 8 | 0.1480 | 0.1472 | 0.0148 | 0.0154 | 0.0143 | 0.1531 | 0.1556 | 0.1523 |
| decode_bs32 | 32 | 256 | 2.5016 | 2.4819 | 0.0210 | 0.0195 | 0.0161 | 2.5050 | 2.5091 | 2.5057 |
| decode_bs256 | 256 | 2048 | 4.1953 | 4.1677 | 0.0335 | 0.0276 | 0.0182 | 4.2161 | 4.2291 | 4.2194 |
| prefill_t2048 | 2048 | 16384 | 7.7978 | 8.8013 | 0.1748 | 0.0883 | 0.0346 | 7.9639 | 8.0381 | 7.9803 |
| prefill_t8192 | 8192 | 65536 | 19.5853 | 23.0991 | 0.6444 | 0.2913 | 0.0829 | 20.2161 | 20.3904 | 20.2806 |

The single most informative column pair is **GEMM (store)** vs **GEMM (atomic)** — the same
kernel, same tuning grid, only the epilogue flag differs:

* decode: the atomic epilogue is **0.5–0.7 % faster** than the plain store. `out [T,6144]`
  is 0.01–3.0 MB, i.e. L2-resident, so the 8× read-modify-write never reaches HBM while the
  plain store *does* stream 8× more bytes (`[T,8,6144]`) out to memory.
* prefill: the atomic epilogue is **13 % / 18 % slower**. `out` is 24 MB / 96 MB, well past
  the 8 MB L2, so the RMW goes to HBM — and an atomic RMW is strictly worse there than a
  streaming store. This is exactly prediction (a) from §2, and it is the whole result.

### 5.2 Fused vs unfused

| variant | regime | fused ms | unfused ms | **speedup** | rel_err (fused) | rel_err (unfused) |
|---|---|---|---|---|---|---|
| `f8_atomic` | decode_bs1 | 0.1526 | 0.1531 | **1.003×** | 7.0e-03 | 3.5e-03 |
| `f8_atomic` | decode_bs32 | 2.4799 | 2.5050 | **1.010×** | 8.3e-03 | 3.8e-03 |
| `f8_atomic` | decode_bs256 | 4.1843 | 4.2161 | **1.008×** | 8.8e-03 | 3.5e-03 |
| `f8_atomic` | prefill_t2048 | 8.8097 | 7.9639 | **0.904×** | 7.4e-03 | 3.3e-03 |
| `f8_atomic` | prefill_t8192 | 23.2364 | 20.2161 | **0.870×** | 7.6e-03 | 3.3e-03 |
| `f8_token_major` | decode_bs1 | 0.1413 | 0.1531 | **1.083×** | 3.5e-03 | 3.5e-03 |
| `f8_token_major` | decode_bs32 | 3.8817 | 2.5050 | **0.645×** | 2.7e-03 | 3.8e-03 |
| `f8_token_major` | decode_bs256 | 30.7863 | 4.2161 | **0.137×** | 2.4e-03 | 3.5e-03 |
| `f8_token_major` | prefill_t2048 | 246.62 | 7.9639 | **0.032×** | 2.1e-03 | 3.3e-03 |
| `f8_token_major` | prefill_t8192 | 985.75 | 20.2161 | **0.021×** | 2.2e-03 | 3.3e-03 |
| `f9_atomic` | decode_bs1 | 0.1518 | 0.1556 | **1.025×** | 8.9e-03 | 3.6e-03 |
| `f9_atomic` | decode_bs32 | 2.4806 | 2.5091 | **1.011×** | 8.8e-03 | 4.0e-03 |
| `f9_atomic` | decode_bs256 | 4.1935 | 4.2291 | **1.008×** | 9.3e-03 | 4.6e-03 |
| `f9_atomic` | prefill_t2048 | 8.8507 | 8.0381 | **0.908×** | 1.05e-02 | 4.1e-03 |
| `f9_atomic` | prefill_t8192 | 23.3403 | 20.3904 | **0.874×** | 9.2e-03 | 4.6e-03 |
| `f9_token_major` | decode_bs1 | 0.1423 | 0.1556 | **1.094×** | 2.2e-03 | 3.6e-03 |
| `f9_token_major` | decode_bs32 | 3.8822 | 2.5091 | **0.646×** | 2.4e-03 | 4.0e-03 |
| `f9_token_major` | decode_bs256 | 30.7899 | 4.2291 | **0.137×** | 3.0e-03 | 4.6e-03 |
| `f9_token_major` | prefill_t2048 | 246.61 | 8.0381 | **0.033×** | 3.1e-03 | 4.1e-03 |
| `f9_token_major` | prefill_t8192 | 986.17 | 20.3904 | **0.021×** | 2.8e-03 | 4.6e-03 |

Tolerance is 2e-2 on the max-abs relative error against an fp32 reference; every variant
passes with ≥ 2× margin. Note the systematic ordering: **token-major (2–3e-3) < unfused
(3.3–4.6e-3) < atomic (7–10.5e-3)**. Token-major accumulates all 8 experts in one fp32
register accumulator and rounds once; the unfused chain rounds each expert's output to bf16
then re-accumulates in fp32; the atomic variant rounds each expert to bf16 *and* accumulates
in bf16 (this is sglang's own `FUSE_SUM_ALL_REDUCE` numerics). The atomic path costs about
one extra bit of precision. Still comfortably inside tolerance, but it is a real cost and
it is not free.

### 5.3 #9 against the cheap 2-kernel baseline

`f9_atomic` vs `down GEMM + moe_sum(ADD_RESIDUAL=True)`:
1.003× / 1.010× / 1.006× / 0.902× / 0.869×. i.e. **identical conclusion** — the #9 result is
not an artefact of comparing against a needlessly bad 3-kernel baseline. The separate
`resadd` kernel costs only 0.015–0.29 ms, and folding it into `moe_sum` recovers almost all
of that for free, without any fusion into the GEMM.

`f9_*` minus `f8_*` (the marginal cost of the residual add on the fused side) is
+0.0000 to +0.0410 ms — i.e. one extra `[T, 6144]` read, exactly as predicted. **#9 is
essentially free on top of #8, confirmed.** But since #8 itself does not pay off outside
`decode_bs1`, "free" is free of nothing.

### 5.4 Vendor-BLAS reference line

Per-expert `torch.matmul` (dispatches to the MetaX BLAS) with A pre-gathered into
expert-sorted order **outside** the timed region — the best case for the vendor path — plus
`index_add_` for the merge:

| regime | GFLOP | vendor GEMM ms | vendor TF/s | vendor GEMM+merge ms | Triton GEMM ms | Triton TF/s |
|---|---|---|---|---|---|---|
| decode_bs1 | 0.2 | 0.3315 | 0.6 | 0.3720 | 0.1480 | 1.4 |
| decode_bs32 | 6.4 | 7.1148 | 0.9 | 7.1529 | 2.5016 | 2.6 |
| decode_bs256 | 51.5 | 12.0097 | 4.3 | 12.1167 | 4.1953 | 12.3 |
| prefill_t2048 | 412.3 | 19.7491 | 20.9 | 20.9531 | 7.7978 | 52.9 |
| prefill_t8192 | 1649.3 | 30.9614 | 53.3 | 35.6526 | 19.5853 | 84.2 |

**Triton beats the vendor path by 1.6–2.9× at every regime here**, which looks like it
contradicts the calibration note ("Triton reaches only ~50 % of the vendor BLAS"). It does
not: that calibration is for *one large dense GEMM*. This is a **grouped** MoE GEMM over 256
experts, and the only way to express it with the vendor BLAS is 256 separate launches with
M = 8–256 rows each. Launch overhead and tiny-M inefficiency dominate; the vendor kernel
never gets to show its peak. The honest reading is: the vendor BLAS is 2× better *per
GEMM*, and a fused grouped Triton kernel is 2–3× better *per MoE layer*, because it turns
256 launches into one. (For scale: 84.2 TF/s at `prefill_t8192` against the 106 TF/s the
calibration got from the best plain Triton GEMM — the grouped dispatch, gather and
`sorted_token_ids` indirection cost about 20 %.)

---

## 6. Where the prediction held and where it did not

| prediction (§2, written before measuring) | outcome |
|---|---|
| atomic moves the *same* bytes, so it only wins when `out` is L2-resident (`T ≤ 682`) | **held exactly.** Wins by 0.3–1.1 % at T = 1/32/256, loses by 10–13 % at T = 2048/8192. |
| #8 is worth ≈ 0.1 % at decode because the weight stream dominates 100:1 | **held.** Measured 0.3–1.0 %. |
| #8 is worth ≈ 8–10 % at prefill | **too optimistic.** The ideal saving is ~6 % at `prefill_t8192` (0.63 ms of merge kernel + ~0.56 ms of avoided `c3` write out of 20.2 ms). The atomic epilogue costs +3.5 ms, so the net is −13 %. |
| #9 is nearly free on top of #8 | **held.** +0.00 to +0.04 ms. |
| token-major is at parity at T = 1/32 and loses badly from T = 256 | **half held.** It *wins* at T = 1 (1.08×) and loses 1.55× at T = 32, not parity. At T = 256 / 2048 / 8192 it loses 7.3× / 31× / 49×. |

The T = 32 miss is instructive and worth spelling out. I predicted parity because 256 draws
over 256 experts touch ~161 *distinct* experts, so expert-major and token-major would read
similar weight volume. Wrong: expert-major reads each **distinct** expert's `w2[e]` once
(161 × 25.17 MB = 4.05 GB), whereas token-major reads one per **(token, expert) pair**
(256 × 25.17 MB = 6.44 GB). The ratio 6.44/4.05 = 1.59 predicts the measured 3.88/2.50 =
1.55 almost exactly. Deduplicating experts is the entire value of the `sorted_token_ids`
dispatch, and token-major throws it away. `decode_bs1` is the one regime where there is
nothing to deduplicate (8 rows, 8 distinct experts), and that is precisely the one regime
where token-major wins.

A second surprise: the measured weight-stream bandwidth is higher than the 1.05 TB/s I was
given. `decode_bs32` moves ≥ 4.05 GB of weights in 2.50 ms → **≥ 1.62 TB/s effective**.
Either the C500's streaming-read bandwidth is above the calibration figure, or L2 is
catching some of the duplicated expert slabs across concurrently-resident CTAs. I did not
chase this down; it does not change any fused-vs-unfused ratio, which is the metric here.

---

## 7. Verdict

**Fusion #8 (down GEMM + expert merge): not worth doing on the C500.**

* At **decode** (T = 1…256) the fused atomic version wins **0.3–1.0 %**. That is real and
  reproducible (three interleaved re-timing passes agree to < 0.1 %), but it is far inside
  the noise of anything else in the layer, and it costs ~1 bit of output precision
  (rel_err 3.5e-3 → 8.8e-3). The reason it is so small is structural, not fixable: the down
  GEMM at decode is 100:1 memory bound on **streaming `w2`** (3.8–6.0 GB per layer), and the
  `[T, 8, 6144]` intermediate the fusion deletes is 0.2–48 MB, i.e. **0.1–0.8 % of the
  traffic**. There is simply nothing to win.
* At **prefill** the fused atomic version **loses 10–13 %**. The `[T, 6144]` accumulator no
  longer fits in L2, so `tl.atomic_add` becomes an HBM read-modify-write and the GEMM itself
  slows from 7.80 → 8.80 ms (t2048) and 19.59 → 23.10 ms (t8192). The merge it eliminates is
  only worth 0.17 / 0.64 ms. This is a clean loss and I am reporting it as one.
* **sglang's current design is right.** Keeping `FUSE_SUM_ALL_REDUCE` behind a flag rather
  than making it the default matches what this measurement says. The only place I would
  turn it on is small-batch decode, and even then for the L2-residency reason rather than
  for the "we avoided materialising a tensor" reason — the win comes from *where* the bytes
  land, not from *how many* there are.

**Strategy (b), token-major, is a decode_bs1-only curiosity.** It is the fastest variant at
T = 1 (1.083× / 1.094×) and the most accurate everywhere, and it genuinely never
materialises the `[T, 8, 6144]` tensor — 45 KB/token of non-weight traffic versus 242 KB.
But from T = 32 up it destroys the expert deduplication that makes grouped MoE tractable,
and it degrades to **0.021×** at `prefill_t8192` (986 ms vs 20 ms). The task asked me to
verify the intuition "strong in decode, poor in prefill"; the correct statement is narrower:
*strong only when tokens outnumber nothing — i.e. when `T·topk ≈ distinct experts`, which on
GLM-5.2's 256-expert top-8 router means `T = 1` and nothing else.*

**Fusion #9 (+ ResAdd2): free, and confirmed free.** Seeding the atomic accumulator with the
residual instead of zeros costs 0.00–0.04 ms (one `[T, 6144]` read); adding a residual load
to the token-major epilogue costs ~0.001 ms. If you are already doing #8 you should do #9.
But the cheaper move is to fold the residual into `moe_sum` (`ADD_RESIDUAL=True`), which
needs no GEMM fusion at all and recovers 0.005–0.11 ms of the separate `resadd` kernel.

### What would change the answer

The atomic loss at prefill is entirely the HBM read-modify-write. A split-K/L2-tiled merge —
grid ordered so that all contributions to one token-block are resident together, letting the
atomics stay in L2 — would remove it, but that requires the dispatch to be sorted by token
*within* an expert-major grid, which the `moe_align_block_size` layout does not provide. That
is a different fusion (#8 + a dispatch-layout change), not the one I was asked to measure,
and I am not claiming a result for it.

---

## 8. Methodology note: one measurement was rejected, and why

The first pass reported `f9_atomic` at `decode_bs32` as **1.568×**, which would have been the
headline number of this whole log. It was wrong. The joint tuning search had measured the
identical 3-kernel unfused chain at 2.511 ms minutes earlier; the final timing block put it
at 3.900 ms — a 55 % excursion on one chain while every other chain in the same block was
stable, and internally impossible given that `unfused8` = 2.506 ms and the `resadd` kernel
alone is 0.0195 ms.

Rather than patch that one number, I added a `--retime` pass to the bench script:
it reloads `results/<id>.json`, replays **only the winning configs**, and times every chain
`3×` in an **interleaved round-robin**, keeping the min of the medians. Interleaving means
any drift hits all chains equally; min-of-medians rejects one-off excursions. It is applied
identically to fused and unfused chains, so it cannot bias the ratio. The re-timed numbers
agree with the first pass to < 0.5 % everywhere except that one cell, where
`unfused9_3kernel` came back as 2.5091 ms (repeatable to 0.00 ms over three passes) and
`f9_atomic` dropped from 1.568× to **1.011×**.

Both sets are in the JSON: the reported `fused_ms` / `unfused_ms` are the re-timed values,
and the originals are preserved as `first_pass_fused_ms` / `first_pass_unfused_ms` so the
discrepancy is auditable. Reproduce with:

```
CUDA_VISIBLE_DEVICES=0 /home/zhangshuhan/my-envs/fusion/bin/python \
    glm52/bench/bench_f08f09_down_merge_resadd.py           # full tune + measure (~1h40m)
CUDA_VISIBLE_DEVICES=0 /home/zhangshuhan/my-envs/fusion/bin/python \
    glm52/bench/bench_f08f09_down_merge_resadd.py --retime  # replay winners (~6 min)
```

---

## 9. Winning configs

Full `TuneResult.table`s for every kernel at every regime (config → ms, plus the failure
reason for any config that did not compile — there were none) are in
`results/f08f09_down_merge_resadd.json` under `tuning.<regime>.*`. Summary:

| regime | unfused GEMM | atomic GEMM | `moe_sum` | token-major |
|---|---|---|---|---|
| decode_bs1 | BM16 BN64 BK64 w4 s4 G8 | BM16 BN32 BK64 w4 s3 G8 | BM2 BD256 w4 s1 | BN64 BK256 w8 s1 GEMV |
| decode_bs32 | BM16 BN32 BK64 w2 s3 G8 | BM16 BN32 BK64 w4 s3 G8 | BM2 BD512 w8 s2 | BN128 BK128 w4 s1 GEMV |
| decode_bs256 | BM16 BN64 BK64 w4 s2 G1 | BM16 BN64 BK64 w4 s2 G1 | BM1 BD1024 w8 s2 | BN32 BK256 w2 s1 GEMV |
| prefill_t2048 | BM128 BN128 BK64 w8 s2 G1 | BM128 BN128 BK64 w16 s2 G4 | BM2 BD1024 w16 s2 | BN64 BK128 w8 s1 GEMV |
| prefill_t8192 | BM128 BN128 BK64 w16 s2 G16 | BM128 BN128 BK64 w16 s2 G1 | BM2 BD1024 w16 s2 | BN64 BK128 w8 s1 GEMV |

Observations worth recording:

* The prefill tile `BLOCK_M=128, BLOCK_N=128, BLOCK_K=64` is the seed config from the
  calibration note with `BLOCK_K` doubled, and it came out on top on both sides at both
  prefill regimes. It uses all 65536 B of SMEM with 94 registers and **zero spills**.
  `num_warps`/`GROUP_M` are where the two sides diverge (next bullet).
* The two sides pick *different* mappings at three of five regimes, which is the whole point
  of tuning them separately: the atomic epilogue prefers more warps at prefill
  (16 vs 8 at t2048) because the RMW is latency-bound and needs more occupancy to hide.
* `USE_DOT=False` (the broadcast/reduce GEMV) won token-major at **every single regime**.
  The padded `tl.dot` at M = 16 with 15 masked rows is never worth it — at M = 1 the tensor
  cores have nothing to do and the GEMV path uses 48 registers / 128 B SMEM against
  64 registers / 10 KB.
* The `torch` pseudo-config (`tensor.zero_()` / `.copy_()`) won the atomic seed at 7 of 10
  (variant, regime) points; the hand-written Triton `seed_kernel` won the rest. Giving the
  fused side the vendor memset where it is faster is deliberate — it is the fused side's
  best case, and it still is not enough.
