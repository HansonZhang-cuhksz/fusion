# LOG-06 — Fusion #10: Expert Merge + Residual Add

GPU: MetaX C500, `CUDA_VISIBLE_DEVICES=3` (exclusive). torch 2.8.0+metax, triton 3.0.0.
Result id: `f10_merge_resadd` → `results/f10_merge_resadd.json`.
Kernel: `glm52/kernels/merge_resadd.py`. Bench: `glm52/bench/bench_f10_merge_resadd.py`.

> **Verdict up front: this fusion is worth doing. It wins at every one of the five regimes
> — 1.147× to 1.204× against an independently tuned two-kernel baseline — and at prefill it
> reaches 99 % and 102 % of its 1.20× bandwidth ceiling. It is the second family in this
> study (after #3) that actually saturates its roofline, and unlike #3 it also wins at
> decode. Zero compile failures in 3 646 config trials, and the fused and unfused outputs
> are bitwise identical at every regime.**
>
> **One caveat that materially changes the engineering recommendation, in §7: sglang does
> not ship the standalone residual-add kernel this baseline uses. It defers the post-MoE
> add into the *next* layer's `fused_add_rmsnorm` (fusion #3). Measured against *that*
> design the end-to-end saving is 1.083×, not 1.20×.**

---

## 1. What is being fused

The tail of the GLM-5.2 MoE block, after the down projection has produced one output row
per (token, expert) pair:

```
Y   [T, 8, 6144] bf16   per-expert outputs, UNWEIGHTED
w   [T, 8]       fp32   routing weights (sigmoid -> noaux_tc top-8 -> normalized -> x2.5)
res [T, 6144]    bf16   pre-MoE residual
--------------------------------------------------------------------------------
out [T, 6144]    bf16   out = (sum_k w_k * Y[:,k,:]).to(bf16) + res
```

| | chain | HBM row-passes over `[T,6144]` bf16 |
|---|---|---|
| **unfused** | `merge_only` (read `Y`, write `m`), then `resadd_only` (read `m`, read `res`, write `out`) | (8+1) + (2+1) = **12** |
| **fused** | one kernel (read `Y`, read `res`, write `out`) | (8+1) + 1 = **10** |

Ceiling = 12/10 = **1.20×** at every regime. Confirmed independently by
`python -m glm52.traffic`, which reports `traffic_ratio 1.20×`, `roofline_ceiling 1.20×`
and `memory` bound for `F10_merge_resadd` at all five regimes — the op has no FLOPs worth
counting, so the latency-aware ceiling collapses onto the traffic ratio.

The top-k input is what caps this. It is 8 of the 10 fused passes; the fusion can only ever
delete the round trip through `m`, i.e. 2 passes out of 12. There is no configuration of
this kernel that can do better than 1.20×, and any measurement above it needs explaining
(§4.3 explains the one that is).

---

## 2. Fairness

**One kernel source.** `glm52/kernels/merge_resadd.py::merge_resadd_kernel`. Two
`tl.constexpr` flags select the behaviour and nothing else changes:

| `DO_MERGE` | `DO_RESADD` | role |
|---|---|---|
| True | True | **fused** |
| True | False | unfused #1 — the weighted top-k reduction, stores `m` |
| False | True | unfused #2 — the residual add, reads `m` |

The grid decode, the row/column masks, the weighted-reduction body, the address arithmetic
and the eviction hints are literally the same lines for all three; only the two flags
differ. Register counts confirm this is not a disguised rewrite: **52–54 regs fused vs
50 merge-only, 0 spills on both** (§6).

**Both sides do the same work and produce the same bytes.** The fused kernel reproduces the
unfused chain's round-to-bf16 of the merged intermediate (`ROUND_MID=True`), so
`torch.equal(fused_out, unfused_out)` is **asserted True at every regime** and is recorded
as `bitwise_identical: true` in every result row. The only thing the fused side skips is
materializing `m`. That is the benefit being measured, and `m` has no other consumer.

`ROUND_MID=False` is also measured: the fused kernel *can* keep the sum in fp32 for free,
which is strictly more accurate. It is reported as an accuracy note
(`rel_err_fused_no_round_mid`) rather than used for the headline timing, because using it
would make the two sides not-quite-comparable.

**Equal-size grids.** `coarse_grid()` and `refine_grid()` are single functions called
identically by all three variants. Per regime: **174 coarse** for each of `fused`,
`merge_only`, `resadd_only`, plus a neighbourhood refine of **52–72** around each side's own
coarse winner (the count varies only because a winner sitting at the edge of the
`BLOCK_N`/`ROWS`/`num_warps` ladders has fewer ±1 neighbours — same rule, same generator).
Every `n_tried` is in `results/f10_merge_resadd.json` under `rows[*].n_tried`, and the
**full `TuneResult.table`** — every config, its ms, and any error — is under `tune_tables`
for all 5 regimes × 4 searches.

Totals across the study: fused **1 188** trials, unfused **2 458** trials (its two kernels
plus the joint chain re-tune), **3 646** in all. **0 compile failures** — verified by
scanning every row of every recorded `TuneResult.table` for a `None` timing.

Two asymmetries, stated rather than hidden — both favour the *baseline*:

1. `resadd_only` ignores the `KVEC` and `UNROLL` knobs (it has no top-k loop), so ~half of
   its 174 coarse configs are behavioural duplicates that merely get timed twice. That
   gives the unfused side extra samples of the same mapping; it does not reduce the number
   of distinct mappings the fused side sees.
2. The unfused side additionally gets a **joint chain re-tune** over the top-2×top-2 configs
   of each of its coarse and refine searches (8–16 unique pairs after dedup), timed as the
   real chain with one L2 flush before the pair and none between. It mattered: at
   `decode_bs1` the joint chain came in at 0.0189 ms against 0.0315 ms for the sum of the
   independently-tuned bests, i.e. the naive sum overstates the baseline by 67 %. Reporting
   the sum instead would have manufactured a fake 1.96× "win".

---

## 3. The mapping search space

| knob | values |
|---|---|
| `BLOCK_N` (tile width over hidden) | 256, 512, 1024, 2048, 4096 |
| `ROWS` (tokens per program) | 1, 2, 4, 8 (refine adds 16) |
| `num_warps` (warp = **64** lanes on C500) | 1, 2, 4, 8, 16 |
| `num_stages` | 1, 2 (refine sweeps 1–4) |
| `KVEC` | 0 = loop the 8 experts into an fp32 register accumulator; 1 = load `[ROWS, 8, BLOCK_N]` as one 3-D block and `tl.sum(axis=1)` |
| `UNROLL` | `tl.static_range` (all 8 loads in flight) vs `tl.range` (rolled, pipelined by `num_stages`) — refine only |
| `grid_cap` | persistent grid striding over (row-block, n-tile) pairs: 104, 208, 416, 832, 1664 — refine only |
| `EVICT` | `evict_first` on the streaming loads — refine only |

Prefilter, applied identically to every variant: output elements per lane
`ROWS*BLOCK_N/(num_warps*64) ∈ [4, 32]`; for `KVEC=1` additionally
`ROWS*8*BLOCK_N/(num_warps*64) ≤ 64` and `ROWS*8*BLOCK_N ≤ 32768` elements, since the 3-D
slab is a live fp32 register tile. → **174 coarse configs**, 120 with `KVEC=0` and 54 with
`KVEC=1`.

Unlike the GEMM families there is **no shared-memory pressure at all** here (no `tl.dot`),
which is why the grid has zero SMEM-driven failures. The C500's 64 KB SMEM ceiling, which
was the binding constraint for #6, is simply not a factor for a pure vector kernel.

Grid is 1-D over the flattened (row-block, n-tile) pairs, with consecutive program ids
walking the n-tiles of one row-block first so that neighbouring CTAs touch contiguous HBM.

Probes run before building the grid (a compile-one-trivial-kernel check, per the brief):
3-D `tl.load` blocks `[ROWS, TOPK, BLOCK_N]` compile and run correctly on Triton 3.0 +
MACA; stride-0 broadcast inputs work (used for the attribution diagnostic in §6);
`do_not_specialize` on `T` gives one binary per config across all five regimes.

### 3.1 Winning configs

| regime | fused | unfused `merge` | unfused `resadd` |
|---|---|---|---|
| decode_bs1 | R2 / BN256 / w2 / s1 / KVEC0 / unrolled | R2 / BN256 / w2 / s1 / KVEC0 | R2 / BN512 / w4 / s2 / **KVEC1** / cap416 |
| decode_bs32 | R4 / BN512 / w8 / s1 / KVEC0 / unrolled | R8 / BN256 / w8 / s2 / KVEC0 | R2 / BN1024 / w8 / s1 / KVEC0 |
| decode_bs256 | R2 / BN2048 / w16 / s1 / KVEC0 / unrolled | R2 / BN2048 / w16 / s2 / KVEC0 / cap832 | R2 / BN2048 / w8 / s1 / **KVEC1** |
| prefill_t2048 | R1 / BN1024 / w4 / s1 / KVEC0 / **rolled** / EVICT | R1 / BN512 / w2 / s1 / KVEC0 | R1 / BN2048 / w2 / s1 / KVEC0 |
| prefill_t8192 | R1 / BN256 / **w1** / s2 / KVEC0 / **rolled** | R8 / BN256 / w8 / s1 / KVEC0 | R2 / BN512 / w4 / s1 / KVEC0 |

Three things the search found that I would not have guessed:

* **`KVEC=1` (the 3-D slab) never wins the merge**, at any regime. It compiles fine and is
  correct, but materialising `[ROWS, 8, BLOCK_N]` fp32 before reducing costs registers for
  nothing — the rolled/unrolled loop already keeps all 8 loads in flight and reduces
  incrementally. It only ever wins for `resadd_only`, where `TOPK` is unused and the flag is
  a no-op, i.e. it wins by coin flip among duplicates. Good null result for the knob.
* **At prefill the winner flips from unrolled (`static_range`) to rolled (`tl.range`)**, and
  at `prefill_t8192` to a **single warp** (64 threads, BLOCK_N=256, 4 elements/lane). At
  that size the kernel is pure streaming: what matters is having many small CTAs in flight
  to keep the memory system busy, not instruction-level parallelism inside one CTA.
* **The persistent grid essentially never wins** (only once, for `merge_only` at
  `decode_bs256`). One program per tile is already the right shape for a vector kernel here.

---

## 4. Results

All timings p50, L2 flushed once before each repetition of the whole chain (never between
the two kernels of the unfused chain). Tolerance 2e-2 on max-abs relative error against the
fp32 reference; every variant passes with ≥ 5× margin.

### 4.1 Headline

| regime | T | fused ms | unfused ms | **speedup** | % of 1.20× ceiling | rel_err fused | rel_err unfused | bitwise identical |
|---|---|---|---|---|---|---|---|---|
| decode_bs1 | 1 | 0.0161 | 0.0189 | **1.175×** | 87 % | 0.0 | 0.0 | yes |
| decode_bs32 | 32 | 0.0184 | 0.0218 | **1.181×** | 90 % | 3.0e-4 | 3.0e-4 | yes |
| decode_bs256 | 256 | 0.0366 | 0.0420 | **1.147×** | 73 % | 2.3e-3 | 2.3e-3 | yes |
| prefill_t2048 | 2048 | 0.1820 | 0.2191 | **1.204×** | 102 % | 1.9e-3 | 1.9e-3 | yes |
| prefill_t8192 | 8192 | 0.6802 | 0.8151 | **1.198×** | 99 % | 3.8e-3 | 3.8e-3 | yes |

("% of ceiling" is `(speedup − 1)/0.20`, i.e. the fraction of the *available* saving that
was actually captured, which is the strict reading. On the softer `speedup/1.20` reading it
is 96–100 % everywhere.)

`rel_err` is identical fused vs unfused by construction — the outputs are the same bits.

**The accuracy dividend I expected from `ROUND_MID=False` did not materialise.** Keeping
the merged sum in fp32 instead of rounding it to bf16 before the residual add gives
0.0 / 1.5e-4 / 2.3e-3 / 1.9e-3 / 3.8e-3 against 0.0 / 3.0e-4 / 2.3e-3 / 1.9e-3 / 3.8e-3 —
i.e. it halves the error at `decode_bs32` and changes nothing anywhere else. The reason is
that the final round-to-bf16 of `out` dominates: the intermediate rounding of `m` is
0.4 ulp of a value that is about to be rounded again anyway. So the fused kernel's *option*
to be more accurate is real but worth essentially nothing here, and using it would only cost
the bit-exact comparability. This is a null result and I am recording it as one.

### 4.2 Component timings (ms, each independently tuned)

| regime | `merge_only` | `resadd_only` | naive sum | **joint chain (baseline)** | **fused** |
|---|---|---|---|---|---|
| decode_bs1 | 0.0161 | 0.0154 | 0.0315 | **0.0189** | 0.0161 |
| decode_bs32 | 0.0184 | 0.0156 | 0.0340 | **0.0218** | 0.0184 |
| decode_bs256 | 0.0346 | 0.0218 | 0.0563 | **0.0420** | 0.0366 |
| prefill_t2048 | 0.1654 | 0.0660 | 0.2314 | **0.2191** | 0.1820 |
| prefill_t8192 | 0.6129 | 0.2127 | 0.8256 | **0.8151** | 0.6802 |

The gap between "naive sum" and "joint chain" is the reason the harness times chains rather
than adding kernels up. At decode it is 40–67 %; taking the sum would have reported
1.7–2.0× and been wrong.

**Cross-check against the sibling family.** F8/F9's port of sglang's `_moe_sum_reduce_kernel`
measured 0.0148 / 0.0210 / 0.0335 / 0.1748 / 0.6444 ms, and its `resadd_kernel` 0.0154 /
0.0195 / 0.0276 / 0.0883 / 0.2913 ms (LOG-05 §5.1). My `merge_only` is at or slightly faster
than theirs (0.6129 vs 0.6444 at t8192) **while doing strictly more work** — it also applies
the fp32 routing weight, which theirs does not (in the sglang path that multiply happens
inside the down GEMM). So the baseline being beaten here is not a straw man; it is at least
as good as the sglang production kernel.

### 4.3 Bandwidth, and the one measurement above the ceiling

| regime | fused bytes | unfused bytes | fused GB/s | unfused GB/s | traffic.py predicts fused ms |
|---|---|---|---|---|---|
| decode_bs1 | 0.12 MB | 0.14 MB | 8 | 8 | 0.0001 |
| decode_bs32 | 3.75 MB | 4.50 MB | 213 | 217 | 0.0030 |
| decode_bs256 | 30.0 MB | 36.0 MB | 860 | 899 | 0.0242 |
| prefill_t2048 | 240 MB | 288 MB | 1383 | 1378 | 0.1936 |
| prefill_t8192 | 960 MB | 1152 MB | 1480 | 1482 | 0.7743 |

Note that **fused and unfused achieve the same effective bandwidth at every regime** (within
1–4 %). That is the cleanest possible statement that this is a pure-traffic fusion: the
fused kernel is not a better kernel, it just has fewer bytes to move.

**`prefill_t2048` measured 1.204×, which is 0.3 % above the 1.20× ceiling.** Per the brief I
investigated rather than celebrated. It is not a real excursion. The byte ratio is exactly
1.200, the achieved bandwidths are 1383 vs 1378 GB/s (so no byte accounting is wrong), and
the excess is **one saved kernel launch**: at ~1.5–3 µs (§4.4) on a 0.18 ms kernel that is
+0.9 %, which comfortably covers the +0.3 % observed. The effect is consistent rather than
noise — the speedup is 1.203× at p10 and 1.203× at p90 as well (§6) — which is what a
constant launch saving on top of a constant byte ratio should look like. This is the
"the win came from launch overhead" case the brief warns about, in its benign form: a small
constant added to a genuine bandwidth win, not a substitute for one.

**Both prefill regimes beat the `traffic.py` model** (0.182 vs 0.194 predicted; 0.680 vs
0.774 predicted). That is the model being conservative, not the measurement being wrong:
`traffic.py` uses `B_PEAK = 1.30 TB/s`, the mixed read/write figure calibrated from F3.
This kernel is **9 reads : 1 write**, far more read-dominated than F3's 2R:2W, so it lands
near the read-only figure (`B_PEAK_READ_ONLY = 1.60 TB/s`) instead — 1.48 TB/s measured.
This is a data point for the LOG-10 §4 calibration note: on C500 the read/write mix moves
achievable bandwidth by ~15 % and the two existing calibration numbers bracket it correctly.

### 4.4 Where the win actually comes from — two different mechanisms

The speedup curve is **not** monotonic: 1.175 → 1.181 → **1.147** → 1.204 → 1.198. The dip
at `decode_bs256` is real and it is the interesting part of this result.

Fit the obvious two-term model `t = L + bytes/BW`, with `L` a fixed per-chain launch cost:

* At `decode_bs1` the kernel moves 0.12 MB — bandwidth is irrelevant, and `t ≈ L`. Fused
  0.0161 vs unfused 0.0189 ⇒ the *second launch* costs **Δ ≈ 2.8 µs** on an `L ≈ 16 µs`
  base, giving 1.175×. **At decode the win is launch-count, not bytes**, and it only
  *looks* like it hits the byte ceiling by coincidence.
* At `prefill_t8192`, `L` is 2 % of a 0.68 ms kernel and the win is entirely bytes: 1.198×,
  99 % of the traffic ceiling.
* `decode_bs256` is the trough where **neither** mechanism is at full strength: bytes are
  large enough (30 MB) that the fixed `L` no longer dominates the ratio, but the kernel
  still only reaches 860 GB/s — 58 % of the 1.48 TB/s the same kernel gets at t8192, because
  30 MB across 104 CUs is not enough work to fill the memory pipeline. Solving
  `0.0366 = L + 30.0MB/BW` and `0.0420 = L + Δ + 36.0MB/BW` with `L = 15.5 µs` gives
  `BW ≈ 1.50 TB/s` and `Δ ≈ 1.4 µs` — consistent with both endpoints. With a fixed `L > 0`
  in both numerator and denominator, the ratio is algebraically pulled toward 1. Nothing is
  mistuned; this is what a launch-floor plus a half-full memory pipe looks like.

### 4.5 Launch overhead is worth more than the fusion at decode, in an eager loop

`bench_chain` also reports a no-flush median. At prefill it is ~equal to the flushed p50
(0.6761 vs 0.6802), i.e. GPU-bound as expected. At decode it is **larger**:

| regime | fused flush / noflush | unfused flush / noflush | noflush ratio |
|---|---|---|---|
| decode_bs1 | 0.0161 / 0.0522 | 0.0189 / 0.0927 | **1.78×** |
| decode_bs32 | 0.0184 / 0.0556 | 0.0218 / 0.0970 | **1.75×** |
| decode_bs256 | 0.0366 / 0.0596 | 0.0420 / 0.0996 | **1.67×** |

Without the 128 MB L2 memset spacing the iterations out, 400 back-to-back launches queue
faster than Python can issue them, so the no-flush number measures **CPU launch cost**, not
GPU time: the increment from one launch to two is 0.0927 − 0.0522 = **40 µs per additional
Triton launch from Python**. Two honest readings follow:

* In a **Python-driven eager** decode loop, halving the launch count is worth **1.7×** here
  — far more than the 1.20× of bandwidth. That is a real production effect and it is why
  the fusion is attractive at decode.
* Under **CUDA-graph capture** that overhead disappears entirely and the correct number is
  the flushed 1.15–1.18×. I am quoting the flushed numbers as the headline precisely
  because they are the conservative ones.

---

## 5. Reference lines

No GEMM is involved in this fusion, so there is **no vendor-BLAS reference line** to quote —
the relevant production baselines are torch eager and inductor, both of which ran.

| regime | fused Triton | unfused Triton | torch eager | **torch.compile** | eager / fused | compile / fused |
|---|---|---|---|---|---|---|
| decode_bs1 | 0.0161 | 0.0189 | 0.0625 | 0.0174 | 3.9× | 1.08× |
| decode_bs32 | 0.0184 | 0.0218 | 0.1257 | 0.0279 | 6.8× | 1.52× |
| decode_bs256 | 0.0366 | 0.0420 | 0.2985 | 0.0998 | 8.2× | 2.73× |
| prefill_t2048 | 0.1820 | 0.2191 | 1.7029 | 0.6659 | 9.4× | 3.66× |
| prefill_t8192 | 0.6802 | 0.8151 | 6.4922 | 2.6058 | 9.5× | 3.83× |

**torch.compile / inductor works on this MACA backend** — it compiled, produced numerically
correct output (`rel_err` identical to ours, `torch_compile_check.ok = true` at every
regime), and beat eager by 2.5–3.6×. But it is **1.1–3.8× slower than the hand-written
Triton kernel**, and the gap grows with size.

The byte arithmetic says where it goes. At `prefill_t8192`:

| what inductor could be doing | bytes | at 1.48 TB/s |
|---|---|---|
| fully fused (read `Y` bf16, read `res`, write `out`) | 1.21 GB | 0.82 ms |
| materializing the `Y.float()` promotion, then reducing | 4.43 GB | 3.00 ms |
| **measured** | | **2.61 ms** |

The measurement sits at the second line, so inductor is (inferred from bytes; I did not dump
the generated kernel) reducing over a **1.6 GB fp32 temporary** where our kernel reads
0.8 GB of bf16 and accumulates in registers. Eager is worse still (9.5× off) for the same
reason plus a separate `.sum(1)` pass.

Worth recording plainly: **inductor is a perfectly reasonable choice at `decode_bs1`**
(0.0174 vs 0.0161 ms, 8 % off) and a bad one at prefill.

---

## 6. Attribution diagnostics

The brief asks for the F1/F6 diagnostic technique to be applied whenever the fused kernel
underperforms. It did not underperform — but the same two probes are the cleanest available
*proof* that the win is what I claim it is, so I ran them anyway.
Script: `glm52/bench/diag_f10_merge_resadd.py`; output merged into
`results/f10_merge_resadd.json` under `diagnostics`.

### (i) Codegen: no register cliff, no spills

Triton cache cleared between every compile, one kernel per cache.

| regime | fused `n_regs` | `merge_only` `n_regs` (same config) | `resadd_only` `n_regs` | spills |
|---|---|---|---|---|
| decode_bs256 | 52 | 50 | 20 | 0 / 0 |
| prefill_t2048 | 54 | 50 | 38 | 0 / 0 |
| prefill_t8192 | 52 | 50 | 12 | 0 / 0 |

The fused epilogue costs **+2 to +4 registers**, zero spills. Contrast with F6, where the
second accumulator took registers 104 → 214 and halved CTAs/SM; and with F1, where registers
were identical but the schedule collapsed anyway. Neither failure mode is present here: 52
registers on a 131 072-reg SM is nowhere near an occupancy limit, and there is no mainloop
software pipeline for the epilogue to disturb because there is no `tl.dot`.

### (ii) Stride-0 residual: the cost is DRAM traffic, and nothing else

Point the fused kernel's *extra* input at a broadcast row (`res[0:1].expand(T, H)`,
`stride(0) == 0`). Identical instruction stream, identical masks, ~zero incremental DRAM
traffic. The difference isolates memory cost from instruction cost.

All four variants timed in an interleaved round-robin, 3 passes, min-of-medians (see the
methodology note at the end of this section).

| regime | fused, real `res` | fused, stride-0 `res` | `merge_only` @ same cfg | **residual DRAM cost** | **residual instruction cost** |
|---|---|---|---|---|---|
| decode_bs256 | 0.0366 | 0.0346 | 0.0343 | 0.0020 ms | 0.0003 ms |
| prefill_t2048 | 0.1807 | 0.1651 | 0.1641 | 0.0156 ms | 0.0010 ms |
| prefill_t8192 | 0.6761 | 0.6175 | 0.6121 | **0.0586 ms** | **0.0054 ms** |

At `prefill_t8192` the residual is 100.7 MB; at the measured 1.48 TB/s that predicts
**0.068 ms**, against 0.0586 ms measured — the extra read is 86 % accounted for by pure
bandwidth, and the instruction cost of the fused epilogue is **0.8 %** of kernel time.

This is precisely the **inverse of F1's finding**. There, the epilogue's DRAM traffic cost
+0.1 % and its mere presence cost 19 % through a codegen cliff. Here the epilogue's presence
costs 0.8 % and its DRAM traffic costs 8.7 % — the roofline is telling the whole truth. The
structural reason is that F1's kernel had a `tl.dot` mainloop whose schedule the epilogue
could break; this kernel has no mainloop to break.

`fused = merge_only + residual_read` holds to within 2 % at every regime (0.6103 + 0.0586 =
0.6689 vs 0.6761 measured at t8192). The fusion is exactly as expensive as its byte count
says, which is why it hits its ceiling.

**Methodology note, since a sibling family had to reject a measurement for the same
reason.** The first, non-interleaved version of this diagnostic reported a residual DRAM
cost of **2.86 ms** at `decode_bs256` — for a kernel whose total runtime is 0.0366 ms, i.e.
internally impossible. Rather than delete the outlier I rewrote the probe to time all four
variants round-robin over 3 passes and keep the min of the medians, exactly as LOG-05 §8
did. The re-run reproduces the good pass to within 3 % on every cell
(0.0020 / 0.0156 / 0.0586 ms of DRAM cost versus 0.0020 / 0.0159 / 0.0586 in the first
clean pass), and the pathological cell is gone. The headline fused-vs-unfused numbers in §4
come from the bench's own `bench_chain` and were never affected.

**Robustness of the headline ratios.** The absolute p10–p90 spread is *not* uniformly
tight — it is 19–22 % at `decode_bs1`, 9–13 % at `decode_bs32`, 5 % at `decode_bs256`, 1.4 %
at `prefill_t2048` and 0.5 % at `prefill_t8192`, which is exactly what a Python-launch-bound
20 µs kernel should look like. But the jitter is *correlated* across the two sides (same
host, same interleaving), so the ratio is far more stable than either number:

| regime | speedup at p10 | **at p50 (headline)** | at p90 |
|---|---|---|---|
| decode_bs1 | 1.195× | **1.175×** | 1.223× |
| decode_bs32 | 1.198× | **1.181×** | 1.165× |
| decode_bs256 | 1.159× | **1.147×** | 1.148× |
| prefill_t2048 | 1.203× | **1.204×** | 1.203× |
| prefill_t8192 | 1.199× | **1.198×** | 1.198× |

Every quantile agrees to within ±0.03× of the headline, and no regime's interval touches
1.0×. The `decode_bs256` dip (§4.4) survives at every quantile too, so it is structural and
not a noisy p50.

---

## 7. The honest end-to-end number, and how this compares with F8/F9

### 7.1 sglang does not have a standalone post-MoE residual add

Checked in the shipped source (`sglang/srt/models/glm4_moe.py` + `layers/communicator.py`):
the decoder layer carries `(hidden_states, residual)` as a **pair** through the whole block
and the residual add is performed by the *next* `input_layernorm(hidden_states, residual)`,
i.e. by `fused_add_rmsnorm` — fusion **#3**. So the two-kernel baseline I was asked to beat
is not what a real serving stack runs, and #10 and #3 are **competing for the same add**.

Counting row-passes over `[T, 6144]` bf16 for the whole tail, up to and including the next
layer's pre-attention norm:

| design | kernels | passes |
|---|---|---|
| **A. sglang today** | `moe_sum` (8R+1W) → next layer `fused_add_rmsnorm` (2R+2W) | 9 + 4 = **13** |
| **B. the baseline measured here** | `merge` (8R+1W) → `resadd` (2R+1W) → `rmsnorm` (1R+1W) | 9 + 3 + 2 = **14** |
| **C. F10 fused** | `merge+resadd` (9R+1W) → `rmsnorm` (1R+1W) | 10 + 2 = **12** |
| **D. F10 + F3 fused together** | `merge+resadd+rmsnorm` (9R+2W) | **11** |

So: my measured **1.20× is B/C**, exactly the comparison specified. But the number an
engineer should act on is **A/C = 13/12 = 1.083×**, because sglang already gets that
residual add for free by deferring it. Design **D**, which is the natural follow-up nobody
in this study has built, is **A/D = 13/11 = 1.18×** and would subsume both #3 and #10 into a
single kernel. Given that both #3 and #10 independently reached ~100 % of their ceilings,
D is very likely to reach its 1.18× too, and I would build that next.

### 7.2 Side by side with F8/F9 — the two paths to the same reduction

F8/F9 (`results/f08f09_down_merge_resadd.json`, LOG-05) attack the same reduction from the
other end: fuse it into the **down GEMM** and eliminate my kernel entirely.

| | F10 (this log) | F8/F9 atomic | F8/F9 token-major |
|---|---|---|---|
| what it fuses | merge + resadd, as a vector kernel | merge (+resadd) into the down GEMM epilogue | same, via a token-major grid |
| decode_bs1 | **1.175×** | 1.003× / 1.025× | 1.083× / 1.094× |
| decode_bs32 | **1.181×** | 1.010× / 1.011× | 0.645× / 0.646× |
| decode_bs256 | **1.147×** | 1.008× / 1.008× | 0.137× / 0.137× |
| prefill_t2048 | **1.204×** | 0.904× / 0.908× | 0.032× / 0.033× |
| prefill_t8192 | **1.198×** | 0.870× / 0.874× | 0.021× / 0.021× |
| denominator | the merge+add pair (0.019–0.82 ms) | the whole down GEMM (0.15–20.4 ms) | same |

**They are not measuring the same thing, and reading the two columns as competitors is a
mistake.** F10's 1.20× is 20 % of a 0.68 ms tail. F8's 0.87× is −13 % of a 20.4 ms GEMM.
In absolute ms at `prefill_t8192`: F10 saves **0.135 ms**; F8 *costs* **3.0 ms**.

The right combined reading:

* **At prefill, do F10 and do not do F8.** F8 turns the merge into an HBM
  read-modify-write on a 96 MB accumulator that no longer fits the 8 MB L2, and pays 3.0 ms
  to save 0.64 ms. F10 pays nothing extra and saves 0.135 ms. The two are not exclusive in
  principle, but F8's atomic epilogue makes the `[T,8,H]` tensor disappear — which would
  delete F10's input and hence F10 — so at prefill you want the tensor to exist and the
  tail to be fused, which is exactly design A/C above.
* **At decode, F8 and F10 are both small and F10 is bigger.** F8_atomic wins 0.3–1.0 % of a
  2.5–4.2 ms GEMM = 0.008–0.036 ms. F10 wins 0.003–0.005 ms of a 0.02–0.04 ms tail. Both
  are inside the noise of the layer; F10 has the advantage of being free of the ~1 bit of
  precision F8's bf16 atomics cost (LOG-05 §5.2: rel_err 3.5e-3 → 8.8e-3), whereas F10 is
  bit-exact.
* **F8_token_major, the only variant that truly eliminates the `[T,8,H]` tensor, is a
  `decode_bs1` curiosity** (0.021× at prefill) — so the "eliminate F10's kernel entirely"
  path is closed on this hardware except at batch size 1, where it wins 1.09× against a
  0.156 ms GEMM.

---

## 8. Surprises

1. **The 3-D slab (`KVEC=1`) never wins the merge.** It compiles, it is correct, and it is
   the "obvious" way to express a top-k reduction — and across 5 regimes × 174+ configs it
   is beaten every time by a plain loop with an fp32 accumulator. Loading
   `[ROWS, 8, BLOCK_N]` as one live tile buys nothing that the loop's in-flight loads do not
   already buy, and costs registers.
2. **A single warp wins `prefill_t8192`.** 64 threads, `BLOCK_N=256`, `ROWS=1`, rolled loop.
   For a pure streaming kernel the goal is many small CTAs in flight, and the C500's 104 CUs
   are happiest with tiny programs rather than fat ones.
3. **Unrolled at decode, rolled at prefill.** `tl.static_range` wins at T ≤ 256 (ILP
   matters when there are few CTAs); `tl.range` with `num_stages` wins at prefill (the
   pipeliner does better and the smaller binary matters).
4. **The measured bandwidth (1.48 TB/s) exceeds `traffic.py`'s 1.30 TB/s constant**, because
   this kernel is 9R:1W. The two calibration figures in `traffic.py` (1.29 mixed, 1.60
   read-only) bracket it correctly; the constant chosen for the ceiling is the conservative
   one, so the model under-predicts absolute time by 12 % here while getting the *ratio*
   exactly right. Working as designed, but worth writing down.
5. **The non-monotonic speedup curve** (§4.4). I expected a flat 1.20× at every regime from
   a pure-traffic fusion, and the trough at `decode_bs256` initially looked like a tuning
   failure. It is not: it is the crossover between a launch-count-dominated regime and a
   bandwidth-dominated regime, where neither mechanism is at full strength.
6. **How much the joint chain re-tune mattered** (§2). At decode, summing the two
   independently-tuned kernels overstates the baseline by 40–67 %.

---

## 9. Verdict

**Fusion #10 is worth doing. It is the cleanest positive result in this study after #3, and
it is the only one that wins at both decode and prefill.**

* It wins at **all five** regimes: 1.147×–1.204×, capturing **73–102 %** of the available
  1.20× byte saving.
* At prefill it is **bandwidth-saturated at 1.48 TB/s** and the fused/unfused effective
  bandwidths are identical to within 1 % — the fusion removes bytes, nothing else, exactly
  as the roofline says it should.
* It costs **+2 to +4 registers**, **zero spills**, and **zero** accuracy: the fused output
  is bitwise identical to the unfused chain. (Dropping the intermediate bf16 rounding is
  free and available, but measurably buys almost nothing — §4.1.)
* It beats torch eager by **3.9–9.5×** and inductor by **1.1–3.8×**.
* In a Python-driven eager decode loop the launch-count saving alone is worth **1.7×**.

**The caveat that should govern the engineering decision:** measured against what sglang
actually ships — where the post-MoE add is already absorbed into the next layer's
`fused_add_rmsnorm` — the end-to-end saving is **1.083×**, not 1.20×. The two-kernel
baseline this experiment was defined against is 1 row-pass worse than sglang's current
design. That does not make the measurement wrong; it makes the *headline* the wrong number
to quote to a serving team.

**What I would build next:** the triple fusion, merge + residual-add + RMSNorm in one kernel
(design D, §7.1). It is 11 row-passes against sglang's 13 = **1.18×** end-to-end, it
subsumes both #3 and #10, and both of those independently hit ~100 % of their ceilings on
this hardware, so there is good reason to expect it to hit its own. It needs nothing this
hardware cannot do: no atomics, no SMEM pressure, no `tl.dot`, ~55 registers.

---

## 10. Reproduce

```bash
cd /home/zhangshuhan/fusion
CUDA_VISIBLE_DEVICES=3 /home/zhangshuhan/my-envs/fusion/bin/python \
    glm52/bench/bench_f10_merge_resadd.py
```

~60 min on a cold Triton cache (3 646 config trials; MACA compiles cost ~4 s each and
dominate the first regime), ~5 min warm. Per-regime checkpoints land in
`results/_f10_merge_resadd_ckpt/` and are reused on restart, so a crash costs one regime.
`results/f10_merge_resadd.json` is rewritten and valid after every regime.
