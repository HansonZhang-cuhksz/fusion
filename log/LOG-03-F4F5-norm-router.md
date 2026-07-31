# LOG-03 — Fusions #4 and #5: (ResAdd +) RMSNorm + Router (+ sigmoid top-8)

GPU 3 (MetaX C500). Deliverables:
`glm52/kernels/norm_router.py`, `glm52/bench/bench_f04f05_norm_router.py`,
`results/f04f05_norm_router.json` (complete, all 5 regimes, full `tune_tables`).

> **Verdict: the fusion loses everywhere — 0.21×–0.68×.** It is the worst-performing
> fusion in the study so far, worse than F6. The roofline said it should have been the
> best remaining one (up to 1.97×). The gap is a *codegen* effect, not a traffic effect:
> putting the normalization prologue inside the router GEMM's mainloop drops that GEMM
> from **76 TF/s to 32 TF/s**, and the "free" normalization ends up costing **2.1× more
> than running it as its own kernel** (3.8× more at T=2048).

---

## 1. What was built

One Triton kernel, `norm_router_kernel`, with `tl.constexpr` flags selecting which stages
run. Every variant below — fused and unfused — is that same source, differing only in
flags and mapping:

| DO_ADD | DO_NORM | DO_GEMM | DO_TOPK | kernel |
|---|---|---|---|---|
| – | T | – | – | rmsnorm (unfused #5, part 1) |
| T | T | – | – | add+rmsnorm (unfused #4, part 1) |
| – | – | T | – | router GEMM (unfused, part 2) |
| – | – | – | T | sigmoid + top-8 (unfused, part 3) |
| – | T | T | – | **FUSED #5** |
| T | T | T | – | **FUSED #4** |
| – | T | T | T | **FUSED #5 + FUSE_TOPK** |
| T | T | T | T | **FUSED #4 + FUSE_TOPK** |

Fused shape (the "free normalization" shape):

```
pass 1   read x (+ res, write h1), accumulate sum-of-squares       -> rstd
pass 2   re-read the row (L2-resident), normalize, WRITE x2, and feed the
         same registers straight into tl.dot against the router weight tile
epilogue store logits  — or, with FUSE_TOPK, sigmoid + iterative argmax top-8
```

Both sides write x2 (the expert GEMMs consume it); the fusion removes the router's
*read* of x2. `FUSE_TOPK` additionally never materialises the `[T,256]` fp32 logits —
nothing downstream needs them, only `topk_weights` / `topk_ids`.

Semantics are `glm52.reference` exactly, including the two intermediate bf16 roundings in
rmsnorm, `norm_topk_prob`, and `routed_scaling_factor = 2.5`. GLM-5.2 has
`n_group = topk_group = 1`, so `noaux_tc` degenerates to a plain top-k over sigmoid
scores (`reference.router` takes the same branch) — no group mask to reproduce.

**fp32 router math.** `moe_router_dtype: float32` is honoured by accumulating in fp32
from bf16 operands: a bf16×bf16 product is *exact* in fp32 (8+8 mantissa bits fit in 24),
so `tl.dot(bf16, bf16, acc=fp32)` is the fp32 reference matmul up to summation order.
Measured logits error vs the fp32 reference: **3.1e-6** at decode.

**Weight layout.** W_gate is `[256, 6144]` (nn.Linear); both sides consume the same
transposed `[6144, 256]` copy so the B tile is contiguous in the expert dimension. That
is a load-time weight-prep choice applied identically to fused and unfused.

**Mapping knobs** — the only thing allowed to differ between the sides: `BLOCK_M`,
`BLOCK_K`, `BLOCK_E`, `NORM_BK`, `num_warps`, `num_stages`, `EVICT`, `REREAD_H1`.
`BLOCK_E < 256` splits the 256 experts over `NSPLIT = 256/BLOCK_E` programs per row block
(the only extra parallelism at T=1, where the router is a GEMV with a single row block);
each split then redoes the norm and only split 0 writes h1/x2. `NORM_BK` gives pass 1 its
own k-tile (§5.2).

---

## 2. Traffic and the roofline ceiling

`act = T·6144·2 B`, W_gate = 3.0 MB, logits = `T·1 KB`.

| | unfused | fused | traffic ratio |
|---|---|---|---|
| #5 | 2·act (norm) + act + 3 MB + logits (router) | 2·act + 3 MB + logits | 1.45–1.47× |
| #4 | 3·act (add) + 2·act (norm) + act + 3 MB + logits | 4·act + 3 MB + logits | 1.48–1.49× |

`python -m glm52.traffic` latency-aware ceilings — these *exceed* the traffic ratio,
because fusing is supposed to let the memory-bound norm hide behind the router GEMM's
compute:

| regime | F5 ceiling | F4 ceiling |
|---|---|---|
| decode_bs1 | 1.00× | 1.01× |
| decode_bs32 | 1.10× | 1.17× |
| decode_bs256 | 1.64× | 1.60× |
| prefill_t2048 | 1.64× | 1.93× |
| prefill_t8192 | 1.64× | 1.97× |

**The L2 premise checks out.** W_gate is 3.0 MB against 8 MB of L2, and the measured
stand-alone router GEMM reaches **76.1 TF/s at T=8192 (71 % of the 107 TF/s Triton
ceiling)** — i.e. it is compute-bound, so its weight re-reads are indeed not costing HBM
traffic. The premise was right; the fusion still fails, for an unrelated reason (§6).

---

## 3. Results

Fused vs its own independently tuned unfused chain, p50 ms, one L2 flush per chain
repetition (never between kernels of a chain).

| regime | variant | fused ms | unfused ms | **speedup** | ceiling | % of ceiling | rel_err (x2 / logits or topk_w) |
|---|---|---|---|---|---|---|---|
| decode_bs1 | #5 | 0.1569 | 0.0742 | **0.473×** | 1.00× | 47 % | 0.0 / 3.3e-6 |
| decode_bs1 | #5+topk | 0.3761 | 0.0911 | **0.242×** | 1.00× | 24 % | 0.0 / 3.7e-7 |
| decode_bs1 | #4 | 0.1930 | 0.0755 | **0.391×** | 1.01× | 39 % | 0.0 / 3.1e-6 |
| decode_bs1 | #4+topk | 0.4421 | 0.0927 | **0.210×** | 1.01× | 21 % | 0.0 / 3.4e-7 |
| decode_bs32 | #5 | 0.1623 | 0.0753 | **0.464×** | 1.10× | 42 % | 0.0 / 3.2e-6 |
| decode_bs32 | #5+topk | 0.4211 | 0.0950 | **0.226×** | 1.10× | 21 % | 0.0 / 5.3e-7 |
| decode_bs32 | #4 | 0.2028 | 0.0768 | **0.379×** | 1.17× | 32 % | 0.0 / 2.9e-6 |
| decode_bs32 | #4+topk | 0.4590 | 0.0952 | **0.207×** | 1.17× | 18 % | 0.0 / 6.0e-7 |
| decode_bs256 | #5 | 0.1810 | 0.0845 | **0.467×** | 1.64× | 28 % | 0.0 / 3.3e-6 |
| decode_bs256 | #5+topk | 0.4147 | 0.1016 | **0.245×** | 1.64× | 15 % | 0.0 / 9.4e-7 |
| decode_bs256 | #4 | 0.2289 | 0.0888 | **0.388×** | 1.60× | 24 % | 3.0e-3 / 4.3e-4 |
| decode_bs256 | #4+topk | 0.4669 | 0.1060 | **0.227×** | 1.60× | 14 % | 3.0e-3 / 1.1e-4 |
| prefill_t2048 | #5 | 0.4239 | 0.2012 | **0.475×** | 1.64× | 29 % | 1.2e-3 / 1.1e-4 |
| prefill_t2048 | #5+topk | 0.5071 | 0.2304 | **0.454×** | 1.64× | 28 % | 1.2e-3 / 1.5e-5 |
| prefill_t2048 | #4 | 0.5363 | 0.2342 | **0.437×** | 1.93× | 23 % | 2.6e-3 / 5.8e-4 |
| prefill_t2048 | #4+topk | 0.6316 | 0.2637 | **0.418×** | 1.93× | 22 % | 2.6e-3 / 7.2e-5 |
| prefill_t8192 | #5 | 0.8074 | 0.5491 | **0.680×** | 1.64× | 41 % | 2.4e-3 / 5.2e-4 |
| prefill_t8192 | #5+topk | 1.1858 | 0.6331 | **0.534×** | 1.64× | 33 % | 2.4e-3 / 1.0e-4 |
| prefill_t8192 | #4 | 1.0207 | 0.6833 | **0.669×** | 1.97× | 34 % | 2.6e-3 / 4.4e-4 |
| prefill_t8192 | #4+topk | 1.3555 | 0.7677 | **0.566×** | 1.97× | 29 % | 2.6e-3 / 1.4e-4 |

x2 and h1 are **bitwise identical** to the fp32 reference at decode; the 1–3e-3 at larger
T is a single element rounded one bf16 ULP differently because the fp32 sum-of-squares is
accumulated in a different order (one ULP = 3.9e-3 relative). Top-8 **ids are exactly the
reference's** in every regime except prefill_t2048 #4+topk, where 1 row in 2048
(4.88e-4) flips its 8th expert — and the *unfused* chain flips the same row, so it is
fp32 summation order, not a kernel defect.

### Per-kernel breakdown (each independently tuned)

| regime | rmsnorm | add+rmsnorm | router GEMM | top-8 | GEMM TF/s | norm GB/s |
|---|---|---|---|---|---|---|
| decode_bs1 | 0.0230 | 0.0246 | 0.0650 | 0.0307 | 0.05 | 1 |
| decode_bs32 | 0.0236 | 0.0253 | 0.0694 | 0.0335 | 1.5 | 33 |
| decode_bs256 | 0.0289 | 0.0333 | 0.0722 | 0.0310 | 11.2 | 217 |
| prefill_t2048 | 0.0727 | 0.1060 | 0.1446 | 0.0433 | 44.5 | 692 |
| prefill_t8192 | 0.2281 | 0.3630 | 0.3384 | 0.0975 | **76.1** | **883** |

Below T≈256 everything is launch-bound: a kernel that touches 24 KB still costs ~23 µs.
That is why the unfused chain's absolute numbers barely move from T=1 to T=256 — and it
is also why the fused side's single launch should have had an advantage it never cashed.

### Vendor / torch reference lines

| regime | `reference.router(rmsnorm(x))` (torch, fp32 matmul + topk) | vendor BLAS bf16 `x2 @ Wgᵀ` | vendor BLAS fp32 | our unfused #5+topk |
|---|---|---|---|---|
| decode_bs1 | 0.1403 | 0.0276 | 0.0302 | 0.0911 |
| decode_bs32 | 0.1733 | 0.0274 | 0.0453 | 0.0950 |
| decode_bs256 | 0.2409 | 0.0384 | 0.0717 | 0.1016 |
| prefill_t2048 | 0.8333 | 0.0694 | 0.3180 | 0.2304 |
| prefill_t8192 | 2.7807 | 0.1930 | 1.1100 | 0.6331 |

Our **unfused** chain beats the torch production path by 1.5–4.4×. But the vendor bf16
GEMM alone runs the router at **134 TF/s** vs our Triton kernel's 76 — the same ~1.8×
Triton-vs-BLAS gap LOG-00/LOG-10 measured. Note the fused kernel (0.807 ms at t8192) is
**4.2× slower than the vendor GEMM alone** (0.193 ms).

---

## 4. Two MACA codegen defects, both silent wrong answers

Neither crashes. Both were caught only because **every config of every variant is
numerically validated against the fp32 reference before it is allowed into a timing
grid** (`screen()` in the bench). Without that step the tuner would have selected a
wrong-answer config and reported it as a win.

### 4.1 Row-wise reduction over a `tl.dot` accumulator

`tl.max` / `tl.argmax` along `axis=1` over a value descending from `tl.dot` returns a
**per-warp partial** result whenever the mma tile spans more than one warp-row. The
accumulator itself is always correct (store it, reload it, reduce — right answer).

Measured (`scratchpad/f04/t6.py`, `[BM,32]×[32,256]`, K=6144, both max and argmax):

| BLOCK_M | warps=4 | warps=8 | warps=16 |
|---|---|---|---|
| 16 | ok | ok | ok |
| 32 | ok | **wrong** | **wrong** |
| 64 | **wrong** | **wrong** | n/a (SMEM) |

Consequence: **FUSE_TOPK may not use `BLOCK_M ≥ 32` with 8+ warps at all** — 23–24 of the
48 offered configs are rejected for wrong answers in every regime. That is a hard cap on
the top-k fusion, not a tuning outcome, and it is why the FUSE_TOPK rows are the worst in
the table.

A second form of the same bug: assembling the 8 winners into a `[BLOCK_M, 8]` tile with
`tl.where(arange(8)==i, v[:,None], 0)` — an mma→blocked broadcast — returns garbage
(`ids = -1`, `weights = nan`) even at `BLOCK_M=16`. The kernel therefore writes the
reduced `[BLOCK_M]` vectors out one at a time and re-reads them (L1-resident, same thread
reads back its own element) to apply the `norm_topk_prob` scaling. The stand-alone top-8
kernel runs the identical epilogue, so both sides pay it.

### 4.2 `num_warps=16` with a *computed* dot operand

`BLOCK_M=16, BLOCK_K=32, num_warps=16` produces **completely wrong logits**
(rel_err ≈ 1.0) in the fused kernel, while the same config in the stand-alone router GEMM
— where the A tile is loaded rather than computed — is correct. 2 of 160 fused configs
per regime. `BLOCK_M=128, NORM_BK=256` similarly corrupts x2 (rel_err 0.92).

### 4.3 Methodological warning for the rest of the study

A 2e-2 tolerance on the top-k **weights** would have accepted every §4.1 miscompile. With
`norm_topk_prob` on, a *completely wrong* expert selection still yields weights within
1–10 % of 2.5/8 = 0.3125. Worse, if W_gate is initialised too large the sigmoid scores
saturate at 1.0, every weight becomes exactly 0.3125, and a wrong router is numerically
invisible. This bench therefore (a) initialises W_gate at 1/√H so logits are ~N(0,1),
(b) checks the **ids**, not just the weights, and (c) uses 2e-3 on the weights.

---

## 5. Fairness and the mapping search

### 5.1 Grids

| variant | grid | offered | screened out (compile / wrong-answer) |
|---|---|---|---|
| rmsnorm | `norm_grid()` | 58 | 0 / 0 |
| add+rmsnorm | `norm_grid()` | 58 | 0 / 0 |
| router GEMM | `gemm_grid()` | 80 | 2 / 0 |
| top-8 | `topk_grid()` | 21 | 0 / 0 |
| fused #5 | `fused_grid()` | 160 | 2 / 4 |
| fused #4 | `fused_grid()` | 160 | 2 / 5 |
| fused #5+topk | `fused_grid()`, BLOCK_E=256 | 48 | 0 / 23 |
| fused #4+topk | `fused_grid()`, BLOCK_E=256 | 48 | 0 / 24 |

`fused_grid()` = `gemm_grid()` × {NORM_BK tied, NORM_BK widened}. So the unfused #5 side
is tuned over **80 + 58 = 138** configs and the fused side over **160** — the fused side
gets *more*, deliberately, since it is the side under suspicion. The FUSE_TOPK grid is
smaller for a structural reason (`BLOCK_E` must be 256 so all of a row's logits live in
one program) and is then halved again by §4.1.

On top of that, **each unfused chain gets a joint re-tune** over the top-3 × top-3 (× top-2)
configs of its pieces, timed as the real chain — 9 or 18 extra combinations that can only
help the baseline. It helped by 0–2 %.

Every `TuneResult.table` — including the rejected configs and their reasons — is in
`results/f04f05_norm_router.json` under `tune_tables`.

### 5.2 The one knob that had to be added: `NORM_BK`

The first version tied pass 1's k-tile to the GEMM's `BLOCK_K`. That is a rigged fused
side: the GEMM wants `BLOCK_K = 32–64`, so the sum of squares ran **96–192 sequential
cross-lane reductions per row**, while the stand-alone norm kernel (which tunes its own
k-tile) always picks `BLOCK_K = 2048` — 3 reductions. `NORM_BK` decouples them. Measured
effect (`scratchpad/f04/nbk.py`, fused #5):

| config | NORM_BK tied | NORM_BK widened |
|---|---|---|
| BM64 BK64 BE128, T=2048 | 0.7286 | **0.5174** |
| BM64 BK64 BE128, T=8192 | 2.3332 | **1.7098** |
| BM32 BK32 BE256, T=2048 | 0.5268 | **0.4785** |
| BM16 BK64 BE64, T=2048 | **0.4657** | 0.5714 |

Up to −29 % for some shapes, and it is chosen by the tuner in 11 of the 20 winning fused
configs. It did **not** change the verdict (best fused at t2048 went 0.4239 → 0.4239).

### 5.3 Winning configs (fused #5 / unfused #5)

| regime | fused #5 | unfused norm | unfused GEMM |
|---|---|---|---|
| decode_bs1 | BM16 BK64 BE32 w4 s2 NORM_BK2048 | BM1 BK2048 w8 s2 evict | BM16 BK64 BE32 w4 s2 |
| decode_bs32 | BM16 BK64 BE32 w8 s2 NORM_BK2048 | BM1 BK2048 w8 s2 | BM16 BK64 BE32 w4 s2 |
| decode_bs256 | BM16 BK64 BE64 w8 s2 NORM_BK2048 | BM1 BK2048 w4 s2 | BM16 BK64 BE64 w4 s2 |
| prefill_t2048 | BM64 BK64 BE64 w4 s2 | BM2 BK2048 w4 s2 | BM64 BK64 BE64 w4 s2 |
| prefill_t8192 | BM32 BK32 BE256 w4 s1 | BM2 BK2048 w4 s2 | BM64 BK64 BE64 w4 s2 |

---

## 6. Attribution — why it loses

All three diagnostics from LOG-10 were run, per regime, and are in the JSON under
`timings.<regime>.attribution`.

### 6.1 It is not registers and not spills

Cache cleared between compiles (`kernel_stats()`), fused vs stand-alone GEMM at each
side's own winner:

| regime | fused #5 n_regs / SMEM | router n_regs / SMEM | spills |
|---|---|---|---|
| decode_bs256 | 150 / 18432 | 88 / 16384 | 0 / 0 |
| prefill_t2048 | 204 / 24576 | 130 / 32768 | 0 / 0 |
| prefill_t8192 | 140 / 18432 | 130 / 32768 | 0 / 0 |

At t8192 the fused kernel uses **140 registers against the GEMM's 130** and *less* shared
memory, with zero spills — this is not F6's register-pressure/occupancy collapse, and
unlike F6 nothing is barred by the SMEM ceiling.

### 6.2 It is not the extra DRAM traffic

The F1 stride-0 trick: point the fused kernel's activation input at `x[:1].expand(T,H)`,
so the instruction stream and masks are identical but the read collapses from 100 MB to
12 KB. Also, `STORE_X2=False` removes the x2 store (attribution only — that variant
produces no x2 and never appears in a speedup row).

| prefill_t8192 | ms | Δ |
|---|---|---|
| fused #5 (as measured) | 0.8074 | — |
| fused #5, x2 store removed | 0.7429 | −0.065 (8 %) |
| fused #5, activation read from a broadcast row | 0.6830 | −0.124 (15 %) |
| **stand-alone router GEMM** | **0.3384** | |
| stand-alone rmsnorm (does the reads AND the store) | 0.2281 | |

At prefill_t2048 the same two isolations are worth only 4 % and 3 % (0.4239 → 0.4058 and
→ 0.4111).

So the fused kernel pays **0.469 ms over the bare GEMM at t8192**, of which at most
0.19 ms is traffic + store. The stand-alone norm kernel does *all* of that work — the
same reads, the same store, the same arithmetic — in **0.228 ms**. Fusing the
normalization into the GEMM makes it **2.1× more expensive than running it separately**
(3.8× at t2048), which is the exact opposite of "free normalization".

### 6.3 It is the GEMM's schedule collapsing — the F1 cliff again

| | TF/s at t8192 |
|---|---|
| stand-alone Triton router GEMM | **76.1** |
| fused #5 (same FLOPs) | **31.9** |
| vendor BLAS bf16 | 134 |

The prologue does not merely add its own cost; it **halves the GEMM**. This is exactly
LOG-10 §1's finding for F1 (an epilogue global load dropped a 107.5 TF/s GEMM to
87.4 TF/s with identical registers and no spills), and it is stronger here because the
prologue sits *inside* the mainloop: every k-tile now does load → fp32 mul → bf16 round →
masked store → layout conversion into the mma operand, and the software pipeline that
made the bare GEMM fast does not survive it.

### 6.4 The two kernels want incompatible mappings

Cross-config timings (each kernel forced onto the other's winner):

| prefill_t8192 | own best | on the other's best |
|---|---|---|
| fused #5 | 0.8074 (BM32 BE256 s1) | 1.1313 (BM64 BK64 BE64) |
| router GEMM | 0.3384 (BM64 BK64 BE64) | 0.5094 (BM32 BE256 s1) |

Neither is under-tuned; they genuinely optimise in different directions. The fused kernel
is pushed toward small `BLOCK_M` and `BLOCK_E=256` (the norm wants few, wide rows and no
expert-split redundancy) while the GEMM wants `BM64 BE64`. Same shape of finding as F6
§2.2, by a different mechanism (there it was a hard SMEM wall, here a soft optimum
conflict).

### 6.5 Why decode is even worse (0.21–0.47×)

At T ≤ 256 the router is a GEMV and the fused kernel's only source of CTA parallelism is
splitting the expert dimension — which makes every split redo the whole normalization.
`BLOCK_E=32` gives 8× the CTAs and 8× the norm work. The unfused side does not face the
choice: its norm kernel uses `BLOCK_M=1, BLOCK_K=2048` (one row per program, 6144 CUs'
worth of parallelism available) and its GEMM independently uses `BM16 BE32`. FUSE_TOPK is
worst of all because it *forbids* the expert split (`BLOCK_E` must be 256), leaving
`T/BLOCK_M` = 1–16 CTAs on a 104-CU machine — hence the flat ~0.42 ms across T=1…256.

---

## 7. Verdict

**Do not fuse the router GEMM into the normalization on C500 with Triton 3.0.**

* Both #4 and #5 regress at every regime: **0.21×–0.68×**, i.e. 14–47 % of the roofline
  ceiling. The best case (prefill_t8192, #5, 0.68×) still loses 32 %.
* `FUSE_TOPK` is worse than plain fusion everywhere, and additionally cannot be tuned:
  the MACA reduction bug (§4.1) rejects half its grid and forbids `BLOCK_M ≥ 32`.
* The mechanism is the same codegen cliff LOG-10 §1 identified for F1, in a stronger
  form: registers are nearly equal, spills are zero, the added DRAM traffic accounts for
  ≤ 15 % of the loss, and the GEMM's throughput halves purely from the presence of the
  in-mainloop prologue.
* The *premise* was sound — the 3 MB router weight does live in L2 and the stand-alone
  router GEMM is compute-bound at 76 TF/s — which is why the roofline model predicted up
  to 1.97×. **The model is right about the hardware and wrong about the compiler.** Any
  cost model for this backend needs a term for "work fused into a GEMM mainloop costs
  ~2× what the same work costs in its own kernel".
* Production recommendation for this layer on this hardware: keep the three launches, use
  the F3 fused `add_rmsnorm` kernel (1.25× and a real win), and call the **vendor BLAS**
  for the router GEMM — 0.193 ms vs our best Triton 0.338 ms and vs the fused kernel's
  0.807 ms at t8192. Fusing the top-k into a separate small kernel is fine (0.03–0.10 ms);
  fusing it into the GEMM is not.

### What would have to change for this fusion to pay

1. A Triton/MACA backend that does not lose the mainloop schedule when a prologue is
   added (the same fix F1 needs).
2. A correct cross-warp `axis=1` reduction over mma values, without which FUSE_TOPK is
   restricted to 16-row tiles.
3. Split-K with `tl.atomic_add` on the fp32 logits for the decode regimes, so the router
   GEMV is not limited to `256/BLOCK_E` CTAs. This was scoped and deliberately not built:
   it cannot be combined with FUSE_TOPK (top-k needs all of a row's logits in one
   program), it needs the logits buffer zeroed by an extra launch, and at decode the
   ceiling is 1.00–1.17× anyway — it would improve both sides' absolute numbers without
   changing the verdict.

---

## 8. Files

* `glm52/kernels/norm_router.py` — the single kernel + 9 thin launchers (8 variants plus
  the `STORE_X2=False` attribution-only launcher).
* `glm52/bench/bench_f04f05_norm_router.py` — validates, screens, tunes 8 kernels × 5
  regimes, joint-retunes the 4 unfused chains, runs the attribution, writes the JSON with
  per-regime checkpoints in `results/_f04f05_norm_router_ckpt/`.
* `results/f04f05_norm_router.json` — 20 rows, all checks, all timings, all attribution,
  and the complete `tune_tables` (including every rejected config and why).
* Scratch experiments referenced above: `scratchpad/f04/{t3,t4,t6}.py` (the codegen-bug
  probes), `scratchpad/f04/nbk.py` (the NORM_BK study), `scratchpad/f04/ct.py` (SMEM
  formula: `num_stages · 2 · BLOCK_K · (BLOCK_M + BLOCK_E)`, confirmed exactly).
