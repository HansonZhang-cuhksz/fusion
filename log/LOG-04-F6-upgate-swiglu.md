# LOG-04 — Fusion #6: MoE Up/Gate grouped GEMM + SwiGLU epilogue

**Date:** 2026-07-27 · **GPU:** MetaX C500 (`CUDA_VISIBLE_DEVICES=0`, exclusive) ·
**Stack:** `torch 2.8.0+metax3.7.1.3`, `triton 3.0.0+metax`, MACA
**Result id:** `f06_upgate_swiglu` → `results/f06_upgate_swiglu.json`
**Code:** `glm52/kernels/moe_gateup.py` · `glm52/bench/bench_f06_upgate_swiglu.py`

```
CUDA_VISIBLE_DEVICES=0 /home/zhangshuhan/my-envs/fusion/bin/python \
    glm52/bench/bench_f06_upgate_swiglu.py
```

**Verdict up front: this fusion is a loss at every regime measured — 0.99× at
decode_bs1 down to 0.55× at prefill_t2048. Do not ship it. §7 explains why, and the
reason is a hardware property of the C500 (64 KB SMEM), not a tuning failure.**

---

## 1. What is being fused

GLM-5.2's first expert GEMM: `A[rows, 6144] @ w13[e][6144, 4096]` with
`rows = T · top_k` (top_k = 8), `w13 = [256, 4096, 6144]` in sglang's `[E, 2I, H]`
gate-then-up layout, followed by `silu(gate) · up` → `[rows, 2048]`.

| | grid N | fp32 accumulators / program | B tiles staged | materialises |
|---|---|---|---|---|
| **UNFUSED** (= what sglang 0.5.10 does today) | `2I = 4096` | 1 | 1 | `[rows, 4096]` **and** `[rows, 2048]` |
| **FUSED** | `I = 2048` | 2 (gate tile + up tile, one shared K-loop) | 2 | `[rows, 2048]` only |

Both sides hand the next stage (the `w2` down-projection) the **same**
`down_input [rows, 2048] bf16`. The fused variant legitimately never materialises the
4096-wide intermediate; that is the fusion's entire benefit and it is quantified in §3.

---

## 2. Implementation and the fairness construction

### 2.1 One kernel source, one `tl.constexpr` flag

`glm52/kernels/moe_gateup.py::moe_gateup_kernel` is written once; `FUSE_ACT` selects the
epilogue.

* `FUSE_ACT=False` → one `acc`, `out = acc`, grid covers N = 4096. Structurally a mirror of
  sglang's `fused_moe_kernel`: `num_tokens_post_padded` early-out, `sorted_token_ids` →
  `offs_token` with the `offs_token // top_k` gather on A, `off_experts * stride_be` weight
  offset, `even_Ks` unmasked-K fast path, `GROUP_SIZE_M` pid swizzle, masked store under
  `token_mask`.
* `FUSE_ACT=True` → adds `b2_ptrs = b_ptrs + I * stride_bn` (the `up` half), a second
  accumulator `acc2`, and `out = (acc * sigmoid(acc)) * acc2` in fp32.

The unfused chain's second kernel `silu_and_mul_kernel` is the split-out element-wise work
(the analogue of sglang's `act_and_mul_kernel`). Dispatch layouts come from
`reference.moe_align_block_size` (the port of sglang's) and are rebuilt per candidate
`BLOCK_M` for **both** sides.

### 2.2 Independent, equal tuning

Per regime, per side:

1. **Coarse grid** produced by the *same* generator (`gemm_grid`) for both sides with the
   same prefilters:
   * SMEM `num_stages · 2 B · BK · (BM + nacc·BN) ≤ 65536`, `nacc = 2` fused / `1` unfused
     (the fused kernel genuinely stages two B tiles). This formula was validated against
     the compiler: predicted 49 152 B / 32 768 B vs `CompiledKernel.metadata.shared`
     49 152 B / 32 768 B for `BM128/BN128/BK32/s2` fused/unfused — exact.
   * accumulator elements per lane `nacc·BM·BN / (num_warps · 64)` ∈ [4, 128]
     (warp = **64** lanes on C500).
   * decode: `BM ∈ {16,32,64,128} × BN ∈ {32,64,128,256} × BK ∈ {32,64,128} ×
     warps ∈ {2,4,8,16} × stages ∈ {2,3}`, `GROUP_M = 8`.
     prefill: `BM ∈ {32,64,128} × BN ∈ {64,128,256} × BK × warps ∈ {4,8,16}`.
   * survivors: **138 fused / 187 unfused** (decode), **45 / 79** (prefill).
2. **Refine**: neighbourhood of the coarse winner — `{BM/2,BM,2BM} × {BN/2,BN,2BN} ×
   {w/2,w,2w}`, plus `BK` and `stages ∈ {2,3,4}` sweeps, plus `GROUP_M ∈ {1,4,8,16}`
   (14–27 configs).
3. The **`silu_and_mul`** kernel gets its own 226-config grid
   (`BM ∈ {1..64} × BN ∈ {64..2048} × warps ∈ {1,2,4,8} × stages ∈ {1,2}`).
4. **Joint re-check.** Separately-optimal kernels need not be jointly optimal, so the top-3
   GEMM configs × top-3 act configs are re-timed **as a chain** (one L2 flush before the
   pair, never between them) and the best chain is what is reported as `unfused_ms`. This
   can only help the baseline.

`n_failed = 0` in every one of the 25 autotune runs — the prefilters are exact, nothing was
wasted and nothing viable was excluded. Full `TuneResult.table`s are in the JSON under
`tuning.<regime>.*` (**2 593 timed configs** in total).

**Where the search is asymmetric, and why that is honest.** The fused side gets a *smaller*
grid (45 vs 79 at prefill) purely because two staged B tiles need 1.5× the SMEM. That is a
real cost of the fusion, not a handicap I imposed: the probe in
`verification.smem_probe` shows the unfused prefill winner
`BM128/BN128/BK64/w8/s2` needs **98 304 B** as a fused kernel against C500's
**65 536 B** limit and fails with `OutOfResources: out of resource: shared memory,
Required: 98304, Hardware limit: 65536`.

### 2.3 Validation

Every variant is checked against an fp32 reference (per-expert
`x_e.float() @ w13[e].float().T` then `silu·mul` in fp32) on a sampled 2048-row subset — a
full-size fp32 reference at T = 8192 would be 3.3 TFLOP of fp32 matmul. `rel_err` is in §4;
all ≤ 2.5e-3 against a 2e-2 tolerance.

The fused variant is *more* accurate (2.4e-3 vs 5.1e-3 unfused): the unfused chain
round-trips the intermediate through bf16 before the activation, the fused one applies
`silu(g)·u` straight on the fp32 accumulators. A small side benefit, not a fairness issue.

---

## 3. Memory-traffic analysis

`rows = T · 8`, `H = 6144`, `I = 2048`, bf16 = 2 B. Each expert's `w13[e]` is
`4096 · 6144 · 2 B = 50.33 MB`.

**Intermediate traffic — the term the fusion removes**

| | bytes |
|---|---|
| unfused: write `[rows,4096]` + read `[rows,4096]` + write `[rows,2048]` | `rows · 20480` |
| fused: write `[rows,2048]` | `rows · 4096` |
| **saved** | **`rows · 16384`** |

**The whole picture** (weight bytes are measured, `traffic.weight_bytes_min` in the JSON;
with the `GROUP_M` swizzle each touched expert's weight is read essentially once):

| regime | rows | experts touched | weight | A | unfused intermediate | fused intermediate | **saved** | unfused total | fused total | traffic ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| decode_bs1    | 8      | 8   | 402.7 MB  | 0.10 MB  | 0.16 MB   | 0.03 MB  | 0.13 MB   | 0.403 GB | 0.403 GB | 1.0003 |
| decode_bs32   | 256    | 156 | 7 851.7 MB| 3.15 MB  | 5.24 MB   | 1.05 MB  | 4.19 MB   | 7.860 GB | 7.856 GB | 1.0005 |
| decode_bs256  | 2048   | 256 | 12 884.9 MB| 25.2 MB | 41.9 MB   | 8.4 MB   | 33.6 MB   | 12.952 GB| 12.918 GB| 1.0026 |
| prefill_t2048 | 16384  | 256 | 12 884.9 MB| 201.3 MB| 335.5 MB  | 67.1 MB  | 268.4 MB  | 13.422 GB| 13.153 GB| 1.0204 |
| prefill_t8192 | 65536  | 256 | 12 884.9 MB| 805.3 MB| 1 342.2 MB| 268.4 MB | 1 073.7 MB| 15.032 GB| 13.959 GB| **1.0769** |

**The structural point.** GLM-5.2 routes to **256 experts** with only a **2048-wide**
intermediate. Any batch of `T ≳ 256` tokens touches essentially the whole **12.9 GB** expert
weight set, so this grouped GEMM is *weight*-bandwidth bound and the 4096-wide intermediate
is rounding error. The theoretical ceiling on the fusion is therefore **+0.03 % at
decode_bs1, +0.26 % at decode_bs256, +2.0 % at prefill_t2048, +7.7 % at prefill_t8192** —
and only if nothing else changes.

**Second, smaller effect that favours the fusion.** For a given `BLOCK_N` the fused grid has
half as many N-tiles (2048/BN vs 4096/BN), so each A row-tile is re-fetched half as often
across the pid_n sweep. With the `GROUP_M` swizzle most of those re-fetches hit L2, so it
shows up as L2 pressure rather than HBM traffic.

**The cost that turns out to dominate.** Two `[BM, BN]` fp32 accumulators and two staged B
tiles → **1.5× the SMEM** and, empirically, **1.5–2.7× the registers** for the same tile
shape (§4, `*_kernel_stats`). On a 64 KB-SMEM part this does not cost a few percent of
occupancy — it removes the good tile shapes from the search space entirely.

---

## 4. Results

Wall-clock p50 over `bench_chain`, one L2 flush before each repetition of the whole chain,
never between its kernels. bf16 in/out, fp32 accumulate.

| regime | rows | **fused ms** | **unfused ms** | **speedup** | (gemm ms) | (act ms) | vendor BLAS ms | fused TF/s | unfused TF/s | vendor TF/s | rel_err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| decode_bs1    | 8     | 0.285  | 0.281  | **0.987×** | 0.278  | 0.018 | 0.427  | 1.4  | 1.4  | 0.9  | 2.46e-03 |
| decode_bs32   | 256   | 4.970  | 4.843  | **0.975×** | 4.838  | 0.020 | 9.385  | 2.6  | 2.7  | 1.4  | 2.38e-03 |
| decode_bs256  | 2048  | 8.519  | 8.177  | **0.960×** | 8.150  | 0.038 | 17.879 | 12.1 | 12.6 | 5.8  | 2.12e-03 |
| prefill_t2048 | 16384 | 25.245 | 13.966 | **0.553×** | 13.798 | 0.174 | 20.899 | 32.7 | 59.0 | 39.5 | 2.36e-03 |
| prefill_t8192 | 65536 | 49.958 | 38.668 | **0.774×** | 38.5*  | 0.622 | 36.053 | 66.0 | 85.3 | 91.5 | 2.45e-03 |

`rel_err` is the fused variant's; the unfused chain's is 5.0e-3–6.1e-3 (bf16 round-trip of
the intermediate). Tolerance 2e-2. `*` see §6 for the one bad sample in the main run.

**Vendor-BLAS reference line** (per-expert `torch.matmul` + torch `silu_and_mul`, A rows
pre-gathered contiguously *outside* the timed region, i.e. best case for the vendor path).
Triton beats it by ~2× at decode (256 small matmuls are launch-bound for the BLAS) and
loses to it at prefill_t8192 (36.05 ms / 91.5 TF/s vs unfused Triton 38.67 ms / 85.3 TF/s =
0.93×). Note the *fused* Triton kernel at prefill_t8192 (49.96 ms / 66.0 TF/s) is 0.72× the
vendor line, i.e. the fusion pushes this kernel from "roughly at production parity" to
"clearly below it".

### Winning mappings (independently tuned)

| regime | fused | unfused GEMM | unfused act |
|---|---|---|---|
| decode_bs1    | BM16 BN32 BK64 w2 s4 G8   | BM16 BN64 BK64 w4 s4 G8   | BM4 BN128 w4 s2 |
| decode_bs32   | BM16 BN32 BK64 w2 s3 G4   | BM16 BN64 BK128 w4 s3 G1  | BM2 BN512 w2 s2 |
| decode_bs256  | BM16 BN32 BK64 w2 s3 G1   | BM16 BN64 BK128 w4 s3 G1  | BM4 BN512 w4 s1 |
| prefill_t2048 | BM32 BN64 BK64 w4 s3 G16  | **BM128 BN128 BK64 w8 s2 G4**  | BM8 BN512 w4 s2 |
| prefill_t8192 | BM128 BN64 BK64 w4 s2 G1  | **BM128 BN128 BK64 w16 s2 G1** | BM4 BN2048 w4 s2 |

Note the fused winner is never the same tile as the unfused winner, and at prefill it is
strictly smaller in at least one dimension — exactly the tension the experiment was set up
to measure.

### Occupancy / register evidence (from the compiled kernels)

| regime | fused `n_regs` / spills / SMEM | unfused `n_regs` / spills / SMEM |
|---|---|---|
| decode_bs1    | 200 / 0 / 20 480 | 94 / 0 / 20 480 |
| decode_bs32   | 164 / 0 / 20 480 | 118 / 0 / 40 960 |
| decode_bs256  | 164 / 0 / 20 480 | 118 / 0 / 40 960 |
| prefill_t2048 | 214 / 0 / **40 960** | 144 / 0 / 65 536 |
| prefill_t8192 | 242 / 0 / **65 536** | 90 / 0 / 65 536 |

No spills anywhere — the fused kernel is not register-starved in the crude sense. What it
*is*, at prefill, is SMEM-starved: with 40 960–65 536 B per CTA it runs **1 CTA/SM**, and
because its tile is 4–8× smaller in output area than the unfused winner's 128×128, it has
far less work in flight per SM to hide HBM latency with. That is the whole 0.55× at
prefill_t2048: achieved bandwidth **0.52 TB/s fused vs 0.96 TB/s unfused** on nearly
identical byte counts.

### Achieved HBM bandwidth (total modelled bytes ÷ p50)

| regime | fused | unfused |
|---|---|---|
| decode_bs1    | 1.41 TB/s | 1.43 TB/s |
| decode_bs32   | 1.58 TB/s | 1.62 TB/s |
| decode_bs256  | 1.52 TB/s | 1.58 TB/s |
| prefill_t2048 | 0.52 TB/s | 0.96 TB/s |
| prefill_t8192 | 0.28 TB/s | 0.39 TB/s |

At decode both variants sit at ~1.5 TB/s and are purely weight-streaming; the prefill rows
are compute-bound so the "bandwidth" there is just an inverse-throughput proxy.

---

## 5. Cross-check: interleaved re-measurement

`results/f06_upgate_swiglu.json → verification`. Four A/B/A/B rounds of the tuned winners,
fused and unfused alternating so any drift hits both equally:

| regime | speedup, 4 interleaved rounds | main run |
|---|---|---|
| decode_bs256  | 0.975, 0.954, 0.960, 0.957 | 0.960 |
| prefill_t2048 | 0.560, 0.560, 0.561, 0.560 | 0.553 |
| prefill_t8192 | 0.774, 0.773, 0.773, 0.774 | 0.774 |

Reproducible to 1–2 %. The conclusion is not a measurement artefact.

---

## 6. Surprises

1. **The fusion loses everywhere, and loses *badly* at prefill.** I expected ~neutral at
   decode and a few percent win at prefill_t8192 (the traffic model says +7.7 % at best).
   Instead prefill_t2048 is 0.553×. The traffic saving is real but is swamped by the
   mapping-space cost: doubling the staged B tile takes the 128×128 tile off the table
   (98 304 B > 65 536 B), and on C500 the 128×128 tile is worth far more than 2 % of
   traffic. On a 228 KB-SMEM Blackwell this trade would likely go the other way — the
   result is device-specific, and specifically a **64 KB SMEM** result.

2. **A real out-of-bounds hazard in the sglang kernel structure on this backend.** The
   original port faulted at decode_bs256 with
   `trapType: Xnack Error/ATU Fault(0x8), kernelName: moe_gateup_kernel`, which disables the
   whole MACA context (`the mcruntime api will be disabled`) and takes the process with it.
   Running each config once in isolation never reproduced it — it needs the sustained launch
   rate of an autotune sweep. sglang relies on `token_mask` alone to neutralise the padded
   dispatch slots (whose `offs_token` is the out-of-range sentinel `num_valid_tokens`), but
   this Triton's pipeliner emits speculative, unpredicated prologue loads, so the sentinel
   row must not even be *addressed*. Fix, in the shared source so both variants get it:

   ```python
   safe_token = tl.where(token_mask, offs_token, 0)   # clamp, value still masked off
   a_ptrs = a_ptr + (safe_token[:, None] // top_k * stride_am + ...)
   c_ptrs = c_ptr + stride_cm * safe_token[:, None] + ...
   ```

   plus a 2 MiB trailing pad on the `w13` allocation (the same guard sglang exposes as
   `SGLANG_MOE_PADDING`) so a speculative B-tile fetch past the last expert lands in mapped
   memory. Numerics are bit-identical before and after. The benchmark now also runs **one
   regime per subprocess** so a fault cannot lose the whole sweep.

3. **The `silu_and_mul` kernel is nearly free in-chain.** At decode_bs1 it measures
   0.018 ms standalone but the unfused chain (0.281 ms) is *below* gemm-alone + act
   (0.278 + 0.018) — its input is 64 KB and still hot in the 8 MB L2 when it runs. This is
   precisely what the harness's "flush once per chain, never between kernels" rule exists to
   capture, and it removes most of the fusion's nominal upside at decode before register
   pressure is even considered.

4. **One bad timing sample.** In the main run's sequential timing block at prefill_t8192,
   GEMM-alone came out at 52.2 ms — larger than the same GEMM inside the unfused chain
   (38.7 ms), which is impossible. The interleaved re-run (§5) gives 37.9 ms for GEMM-alone,
   consistent with the chain. It was a one-off outlier in a `rep=10` block; every other
   number reproduces. Reported rather than quietly deleted; the affected cell is marked `*`
   in §4.

5. **Achieved HBM bandwidth at decode is ~1.5 TB/s, above the ~1.05 TB/s the project brief
   quotes as achievable.** decode_bs32 streams a measured 7.86 GB of expert weights in
   4.84 ms = 1.62 TB/s; decode_bs1 streams 0.403 GB in 0.281 ms = 1.43 TB/s. These are
   pure weight-streaming kernels with no reuse (each expert's `w13` is touched by one
   row-block), so the number is close to a real STREAM figure. Worth re-calibrating the
   1.05 TB/s constant used elsewhere in this project — using it would under-predict decode
   MoE GEMM time by ~40 %.

6. **Triton is not uniformly at ~50 % of the vendor BLAS here.** The LOG-00 calibration
   (dense o_proj) found 0.49–0.51×. For this grouped MoE GEMM the unfused Triton kernel
   reaches **0.93×** of the vendor path at prefill_t8192 and **2.2×** at decode_bs256 (where
   the vendor path pays 256 kernel launches). The 50 % figure is a dense-GEMM property, not
   a backend-wide constant.

---

## 7. Verdict — is this fusion worth it?

**No, not on a MetaX C500, at any GLM-5.2 regime. It should not be shipped.**

* **decode (T = 1 / 32 / 256): 0.987× / 0.975× / 0.960×.** A 1–4 % regression. The upside
  available was 0.03–0.26 % of traffic (§3) and the `silu_and_mul` kernel it deletes costs
  0.018–0.038 ms out of 0.28–8.5 ms (0.5 %), most of which is already hidden in L2. The
  fused kernel's higher register count (200 vs 94 at bs1, 164 vs 118 at bs32/bs256) costs
  more than that. The loss grows monotonically with batch size across the decode range.
* **prefill (T = 2048 / 8192): 0.553× / 0.774×.** A 1.8× and a 1.3× *slowdown*. Here the
  traffic argument is at its strongest (7.7 % of bytes at t8192) and it still loses by a
  wide margin, because the fused kernel cannot use the 128×128 tile that the unfused one
  wins with — 98 304 B of SMEM against a 65 536 B hardware limit. It is 1 CTA/SM with a
  quarter of the output area.

**Where it would flip.** The fusion is not wrong in principle; it is wrong for *these*
numbers. It needs (a) more SMEM, so that two staged B tiles still permit a 128×128 tile —
this is a 96 KB+ device requirement, which C500 does not meet; or (b) a shape where the
intermediate is a large share of traffic, i.e. few experts and/or a wide intermediate. GLM-5.2
is the opposite: 256 experts × 50.33 MB of weights per expert against a 2048-wide
intermediate, so ≥ 92 % of the bytes are weights that the fusion cannot touch. sglang's
choice to keep GEMM1 and `act_and_mul` as separate launches is the right call for this model
on this hardware, and this measurement supports it rather than contradicting it.

**What I would keep from this work.** The `safe_token` clamp (§6.2) is a genuine robustness
fix for the sglang kernel structure on MACA and applies to the unfused production path too —
it is in the shared kernel source, costs nothing measurable, and removes a class of fault
that silently kills the runtime.
