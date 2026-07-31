# LOG-10 — Main-session verification and cross-cutting findings

Work done by the orchestrating session, separate from the per-family agent logs. Everything
here was measured on an otherwise-idle GPU 1 while the other lanes were between agents.

---

## 1. Verification of F1 (o_proj + ResAdd): the prefill regression is real

The F1 agent reported the fusion **losing** at prefill: 0.871× (t2048) and 0.846× (t8192).
Two things made me suspect a rigged/under-tuned fused side rather than a real effect:

- the fused winner was `BLOCK_K=64, GROUP_M=1` while the unfused winner was
  `BLOCK_K=32, GROUP_M=8` — and `GROUP_M=1` wrecks L2 reuse, so it should not win anything;
- the agent recorded `tune_tables` as **empty**, so the grid could not be audited from the JSON.

### 1.1 The result reproduces exactly

Re-tuned the fused kernel independently over a `GROUP_M × BLOCK_K` sweep at `prefill_t8192`
(`scratchpad/f1check.py`):

| config (BM128 BN128 w8 s2) | FUSED ms | UNFUSED-gemm ms |
|---|---|---|
| BK64 GM1 *(agent's fused winner)* | **18.379** | 18.333 |
| BK32 GM8 *(agent's unfused winner)* | 18.851 | **15.343** |
| BK32 GM1 | 18.963 | 16.641 |
| BK32 GM4 | 18.850 | 15.358 |
| BK64 GM8 | 18.578 | 18.536 |
| BK64 GM16 | 18.579 | 18.521 |

Best fused = 18.379 ms, best unfused chain = 15.548 ms → **0.846×**, identical to the agent's
number. **The fused side was not under-tuned.** Forcing it onto the unfused winner's config
makes it *worse* (18.851). Finding withdrawn; the agent's result stands.

### 1.2 Attribution — it is neither occupancy nor bandwidth

Compiled both variants with the cache cleared between compiles so each report is
unambiguous (`scratchpad/f1regs2.py`):

| config | `FUSE_RESADD` | n_regs | n_spills |
|---|---|---|---|
| BM128 BN128 BK32 GM8 | False | 126 | 0 |
| BM128 BN128 BK32 GM8 | **True** | **126** | **0** |
| BM128 BN128 BK64 GM8 | False | 202 | 0 |
| BM128 BN128 BK64 GM8 | **True** | **202** | **0** |

Register count is **identical** and there are **no spills**, so the classic
register-pressure / occupancy-collapse explanation does not apply here.

Then I separated the epilogue's *instructions* from the residual's *DRAM traffic* by pointing
the residual at a stride-0 broadcast (`r[0:1].expand(M, N)`) — same instruction stream,
same masks, but 12 KB of L2-resident data instead of 100 MB from DRAM
(`scratchpad/f1iso.py`, M=8192, K=16384, N=6144):

| config | unfused | fused, residual in DRAM | fused, residual in L2 |
|---|---|---|---|
| BK32 GM8 | 15.348 ms (**107.5 TF/s**) | 18.869 ms (87.4) | 18.847 ms (**87.5**) |
| BK64 GM1 | 18.341 ms (89.9 TF/s) | 18.382 ms (89.7) | 18.363 ms (89.8) |

- cost of the residual's **DRAM traffic**: **+0.1 %** — negligible.
- cost of the epilogue's **instructions**: **+22.8 %**.

And the decisive detail is the second row: at `BK64 GM1` the unfused GEMM is *already* slow
(89.9 TF/s) and the epilogue costs **+0.1 %**. The epilogue only harms the **fast**
configuration, dragging 107.5 TF/s down to 87.4 — essentially onto the slow config's number.

**Conclusion.** On this MACA Triton backend, `BLOCK_K=32, GROUP_M≥4` reaches 107 TF/s through
a mainloop schedule that the mere *presence* of an epilogue global load disables; once any
epilogue exists, every config converges to ~87–90 TF/s. This is a **codegen / pipelining
cliff**, not a hardware-resource effect — registers are identical, spills are zero, and the
added traffic is irrelevant. It is invisible to a cost model built on occupancy and layout
penalties, which is worth noting given that is exactly what `~/fusion-anaylsis` models.

**Practical consequence:** on C500, `torch.addmm` (vendor BLAS, epilogue fused in the library)
regresses only 0.5–1.7 %, while the Triton epilogue fusion regresses 15 %. For fusion #1 the
right production answer on this hardware is *use the vendor's fused epilogue, do not write a
Triton one*.

---

## 2. Verification of F6 (Up_Gate + SwiGLU): regression real, but for a different reason

The F6 agent reported the fusion losing everywhere, worst at prefill: **0.553×** (t2048),
**0.774×** (t8192), 0.96–0.99× decode. Two red flags in its JSON:

- the fused side was tuned over **45** configs at prefill while the unfused GEMM got **79**
  (138 vs 187 at decode) — the classic rigged-baseline signature;
- the fused winner at t2048 was `BLOCK_M=32, BLOCK_N=64`, an implausibly small tile for a
  16 384-row GEMM, against the unfused winner's `BLOCK_M=128, BLOCK_N=128`.

### 2.1 The fused optimum is nevertheless correct

Independent 72-config sweep of the fused kernel at `prefill_t2048`, deliberately including
the large tiles the agent's grid appears to have missed (`scratchpad/f6check.py`, 60 compiled
/ 12 failed):

```
BEST FUSED found here : 24.978 ms   BM32 BN64 BK64 GM8 w4 s3
agent reported fused  : 24.851 ms   BM32 BN64 BK64 GM8 w4 s3   <- same config
agent reported unfused: 13.810 ms   BM128 BN128 BK64 w8 s2
```

Same winning configuration, same time to 0.5 %. **The grid asymmetry did not change the
answer** — the large tiles compiled and were simply slower. The 0.55× regression is real.
(The grid-size asymmetry is still a process defect worth flagging, and is recorded as such;
it just did not bite here.)

### 2.2 Attribution — register pressure and a hard SMEM ceiling

Compiled both variants at matched tiles, cache cleared between compiles
(`scratchpad/f6regs.py`):

| tile | `FUSE_ACT` | n_regs | spills | accumulator |
|---|---|---|---|---|
| BM32 BN64 BK64 w4 s3 | False | 104 | 0 | 8 KB/CTA |
| BM32 BN64 BK64 w4 s3 | **True** | **214** | 0 | 16 KB/CTA |
| BM128 BN64 BK64 w4 s2 | False | 160 | 0 | 32 KB/CTA |
| BM128 BN64 BK64 w4 s2 | **True** | **242** | 0 | 64 KB/CTA |
| BM128 BN128 BK64 w8 s2 | False | 144 | 0 | 64 KB/CTA |
| BM128 BN128 BK64 w8 s2 | **True** | **won't compile** | — | `OutOfResources: shared memory, Required: 98304, limit 65536` |

Two distinct costs, both hardware-grounded:

1. **Register pressure → occupancy loss.** Holding a gate *and* an up accumulator roughly
   **doubles** registers/thread (104 → 214, 160 → 242). At 242 regs × 256 threads = 61 952
   registers per CTA against C500's 131 072/SM, only **2 CTAs/SM** fit versus **4** for the
   unfused kernel. Spills are zero throughout — this is occupancy, not spilling.
2. **A hard SMEM ceiling.** The unfused side's *winning* tile `BM128 BN128` requires 96 KB of
   shared memory in the fused kernel against C500's **64 KB** limit, so it is not merely
   slower — it is **uncompilable**. The fused kernel is structurally barred from the
   configuration that makes the unfused kernel fast.

**This is the textbook toxic fusion**, and it is a different mechanism from F1 (§1), where
registers were identical and the cause was a codegen cliff. Two of the study's fusions
regress for two unrelated reasons; a single-cause model would mis-attribute one of them.

It also independently explains a production choice: **sglang keeps GEMM1 and `silu_and_mul`
as separate launches** rather than fusing the activation into the grouped GEMM. On this
hardware that is the correct call by a wide margin.

---

## 3. A transient F8/F9 defect, diagnosed then self-corrected (finding WITHDRAWN)

**Status: withdrawn — the delivered artifact is correct.** This section is kept because the
diagnosis is a useful worked example of catching a bad measurement from internal evidence
alone, and because withdrawing it honestly matters more than looking right.

I read `results/f08f09_down_merge_resadd.json` while the agent was still running and saw
`f9_atomic @ decode_bs32` = **1.568×** against a roofline ceiling of **1.002×**. A measured
speedup 56 % above a traffic-derived ceiling is impossible for a fusion whose saving is pure
traffic, so one of the two timings had to be wrong.

I concluded it was the **baseline**, from the agent's own tuning table:

| tuning entry @ decode_bs32 | best |
|---|---|
| `joint_unfused8` (gemm + moe_sum) | 2.5075 ms |
| `joint_unfused9_3k` (gemm + moe_sum + resadd) | **2.5106 ms** |
| `joint_unfused9_2k` (gemm + moe_sum_with_residual) | **2.5073 ms** |
| `resadd` kernel, tuned alone (90 cfgs) | **0.0174 ms** |

The F9 unfused chain tunes to **2.507–2.511 ms**, and the extra residual-add kernel costs
**0.017 ms** — exactly what a 1.2 MB elementwise op should cost. But the reported row uses
**3.8996 ms** for that same chain, a **+1.39 ms** inflation with no counterpart anywhere in
the tuning data. The final re-measurement of the F9 unfused chain is the faulty number; the
fusion did not get faster, the baseline got slower.

**Predicted correction at `decode_bs32`** (using the agent's own tuned unfused chain, 2.507 ms):

| row | intermediate value | predicted correct | **final artifact (15:51)** |
|---|---|---|---|
| `f9_atomic` | 1.568× | ~1.01× | **1.0115×** ✓ |
| `f9_token_major` | 1.003× | ~0.645× | **0.6463×** ✓ |

**The agent rewrote the file at 15:51 with values matching the prediction**, so the shipped
result was never wrong — I had read a mid-run intermediate write. The transient value came
from a single noisy timing pass, not a logic error: the chain construction
(`[gemm_fn, sum_fn(add_residual=False), resadd_fn]`) is correct on inspection.

Two lessons worth keeping:

1. **Do not audit a results file while its producer is still running.** An intermediate write
   is not a deliverable. Check the process is finished (or the mtime is stable) first.
2. The **ceiling cross-check earned its keep.** "Measured speedup exceeds the roofline
   ceiling" flagged a bad number from the table alone, with no hardware access and no
   re-run — and the correction derived from internal evidence landed within 0.2 % of what
   the agent independently converged to. That check is worth applying to every row in the
   final table.

The independent auditor (LOG-08 §F4) separately found a *real* issue in the same family that
survives into the artifact: `#9`'s headline `speedup` uses the 3-kernel unfused baseline when
the same script also builds a strictly better 2-kernel one, which inflates the `decode_bs1`
win by **8×** (1.025× vs 1.003×). That is disclosed in the JSON's `speedup_vs_2kernel` field,
and the consolidated table uses the 2-kernel number.

---

## 3b. Verification of F11b (Lazy Pre-Norm → router GEMM): the one GEMM-side win is real

Wins deserve more scrutiny than losses, and F11b is the study's only fusion that improves a
GEMM. Independently re-swept the router GEMM at `prefill_t8192` on GPU 1
(`scratchpad/f11check.py`), fused and unfused, from a freshly generated grid:

| | my re-run | agent |
|---|---|---|
| fused (norm folded into the GEMM epilogue) | **0.5133 ms** | 0.502 ms |
| unfused router GEMM **alone** | **0.3771 ms** | — |
| agent's full unfused chain (norm kernel + GEMM) | — | 0.5660 ms |

The residual implies a standalone norm kernel of `0.5660 − 0.3771 = 0.189 ms`, which is
exactly what that kernel must cost: it moves `h1` in and `x2` out = 201 MB, which at the
measured 1.3 TB/s is 0.155 ms plus launch overhead. Every number is mutually consistent.

Correctness independently confirmed: `rel_err = 2.8e-3` for the fused result against an fp32
`rmsnorm → linear` reference, with the RMSNorm affine weight **pre-folded into the GEMM's B
rows** — confirming the folding identity `((A*rstd)*w) @ B == (A @ (w[:,None]*B)) * rstd`
that makes Lazy Pre-Norm applicable to GLM-5.2 despite the paper listing elementwise affine
as a blocker.

**The mechanism, stated as a budget:** fusing costs the GEMM +0.136 ms (0.377 → 0.513, the
in-mainloop sum-of-squares) and saves the whole 0.189 ms norm kernel. Net **−0.053 ms**, i.e.
**1.10×** by my sweep and 1.13× by the agent's better-refined one. The win is genuine but
narrow, and it exists only because the router's `N = 256` means 1–2 n-tiles, so the
sum-of-squares is computed ~once per row rather than once per n-tile.

---

## 4. Calibration update to `glm52/traffic.py`

The initial roofline constants were `C_PEAK = 106 TF/s` and `B_PEAK = 1.05 TB/s`. Measurements
since then refine them:

- **Compute:** the unfused o_proj GEMM at `BK32 GM8` sustains **107.5 TF/s** — confirms 106–107
  as the Triton bf16 ceiling on C500 (the vendor BLAS reaches ~215).
- **Bandwidth:** F3's fused kernel moves 4 × 100 MB in 0.312 ms = **1.29 TB/s**, and its unfused
  chain 5 × 100 MB in 0.410 ms = 1.23 TB/s. The 1.05 TB/s figure inherited from the earlier
  project was too low; `B_PEAK` updated to **1.3 TB/s**.

This does not change any fused-vs-unfused *ratio* for a purely memory-bound fusion (both sides
scale together), but it does change which kernels the model classifies as compute- vs
memory-bound, and hence several latency-aware ceilings.

---

## 5. Methodology correction: traffic ratio is not the ceiling

The first version of `traffic.py` reported `unfused_bytes / fused_bytes` as the achievable
ceiling. That overstates every compute-bound case. The F1 agent flagged the same thing
independently in LOG-01. `traffic.py` now models each kernel as
`max(flops / C_PEAK, bytes / B_PEAK)` and a chain as the sum over its kernels.

Effect on the headline numbers:

| fusion @ prefill_t8192 | traffic ratio | latency-aware ceiling |
|---|---|---|
| F1 o_proj + ResAdd | 1.30× | **1.02×** |
| F11b prenorm → router | 2.66× | **1.79×** |
| F5 rmsnorm + router | 1.47× | **1.79×** |
| F3 resadd + rmsnorm | 1.25× | 1.25× |

Note F5/F11b move the *other* way — their latency ceiling **exceeds** their traffic ratio,
because fusing lets the memory-bound normalization hide behind the router GEMM's compute
rather than costing a separate pass. That is the "free normalization" mechanism, showing up in
the model before it shows up in a measurement.
