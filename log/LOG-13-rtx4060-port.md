# LOG-13 — Porting the GLM-5.2 fusion study to an RTX 4060 Laptop (sm89)

**Date** 2026-07-31 · **Status** IN PROGRESS — calibration and port fixes done, benchmarks pending
**Host** new machine (`shuhan@`), repo transferred from the C500 host
**GPU** NVIDIA GeForce RTX 4060 **Laptop** GPU, sm_89 (Ada Lovelace), 8 GB
**Clocks** LOCKED: SM **1020 MHz** (of 3105 max, 33 %), MEM **5501 MHz** (of 8001 max, 69 %)
**Stack** torch 2.11.0+cu130, triton 3.6.0, CUDA 13.0
**Baseline being compared against** `results/c500/` (67 files, archived from the C500 study)

---

## 1. The two machines

| resource | MetaX C500 | RTX 4060 Laptop | ratio |
|---|---|---|---|
| SMs / CUs | 104 | **24** | 0.23× |
| warp / wavefront | 64 | **32** | 0.5× |
| SMEM per block, default | 65536 | **49152** | 0.75× |
| SMEM per block, **opt-in** | 65536 (no opt-in path) | **101376** | **1.55×** |
| SMEM per SM | 65536 | 102400 | 1.56× |
| registers per SM | 131072 | **65536** | **0.5×** |
| threads per SM | 2048 | **1536** | 0.75× |
| L2 | 8 MB | **32 MB** | **4×** |
| VRAM | 64 GB | **8 GB** | 0.125× |
| triton | 3.0.0 (`maca`) | 3.6.0 (`cuda`) | — |

Four of these invert conclusions from the C500 study rather than merely scaling them, and are
the reason this port is interesting rather than repetitive:

- **SMEM per block is 1.55× larger.** C500's hard 64 KB ceiling is what made fusion #6
  *uncompilable* at the unfused winner's tile (96 KB required). Here 96 KB **fits**.
- **Registers per SM are halved.** Every occupancy conclusion has to be recomputed; register
  pressure bites twice as hard for the same kernel.
- **L2 is 4× larger.** #8's atomic-accumulator residency argument, and the general question of
  how much of an intermediate the cache absorbs, both move substantially.
- **VRAM is 8× smaller** — the binding constraint on what can be measured at all (§2).

## 2. Scope decisions

**The MoE-expert fusions cannot run here and are excluded.** GLM-5.2's `w13` at 256 experts is
**12.0 GB** and `w13 + w2` is 18.0 GB, against ~7.4 GB usable. No arrangement fits. Rather than
deviate from the model spec (the alternative considered was reducing the expert count to 64,
which preserves every tile shape but multiplies per-expert row count by 4× and distorts the
grouped-GEMM tiling), **#6, #8, #9 and #11a are dropped** and everything measured stays at
*exact* GLM-5.2 spec.

**Measured here — all at exact spec, no deviation:**

| # | fusion | why it fits |
|---|---|---|
| **1** | o_proj + ResAdd | o_proj weight 0.19 GB (prefill) / 0.38 GB (decode) |
| **3** | ResAdd + RMSNorm | pure vector |
| **4/5** | (ResAdd+) RMSNorm + Router | router gate is 3 MB |
| **10** | Expert Merge + ResAdd | needs the `[T·topk, 6144]` intermediate, **not** expert weights |
| **11b** | Lazy Pre-Norm → router GEMM | router gate is 3 MB |

**Regimes:** `decode_bs1`, `decode_bs32`, `decode_bs256`, `prefill_t2048`, `prefill_t8192` —
the benches' native five, identical to the set the C500 verdict table used, so the comparison
is direct. Two-stage autotune (coarse + refine) retained; configs are re-tuned from scratch,
never inherited from C500.

## 3. Calibration (measured on this machine, at the locked clocks)

| quantity | value | method |
|---|---|---|
| streaming BW, read+write | **140 GB/s** | `copy_` / `mul_` at 256–512 MB |
| streaming BW, read-only | **159 GB/s** | Triton reduce kernel, 90 % of theoretical |
| theoretical BW | 176 GB/s | 128-bit × 5501 MHz × 2 |
| **Triton dense bf16 GEMM** | **11.81 TF/s** | M4096 K16384 N6144, BM64 BN256 BK32 w4 s3 |
| cuBLAS bf16, same shape | 11.62 TF/s | `torch.mm` |
| **FLOP/byte balance** | **83.6** | 11.81e12 / 140e9 |

Recorded in `results/device_4060_calibration.json` and `results/rtx4060_gemm_ceiling.json`.

### 3.1 Two calibration findings that matter more than the numbers

**(a) The arithmetic-intensity balance is nearly identical to C500's.** 83.6 FLOP/byte here vs
**82.3** for C500-with-Triton. Locking the SM clock to 33 % while memory sits at 69 % has
coincidentally reproduced C500's compute/bandwidth ratio on a chip 10–19× smaller. Fused-vs-
unfused *ratios* should therefore be meaningfully comparable, which is the whole deliverable.

**(b) Triton reaches 102 % of cuBLAS here — on C500 it reached 50 % of the vendor BLAS.**
This deletes the C500 study's headline escape hatch. LOG-09 §4b and both reports concluded that
*"the real lever is not fusion at all: it is the 107 vs 215 TF/s gap to the vendor BLAS, worth
~30 ms on a ~76 ms layer — two orders of magnitude more than every fusion in this study
combined."* **On this machine there is no vendor gap to close.** Whatever fusion is worth here,
it is not being dwarfed by a codegen deficit. This is the single most important reason the port
is worth doing.

## 4. Port fixes applied so far

| # | file | change | why |
|---|---|---|---|
| 1 | — | archived 67 C500 result files to `results/c500/` | benches write fixed filenames and would have **overwritten the baseline being compared against** |
| 2 | `glm52/common.py:34` | `RESULTS_DIR` now honours `$GLM52_RESULTS_DIR` | keeps the 4060 run (`results/rtx4060/`) from colliding with C500 data |
| 3 | `glm52/common.py:38` | `_FLUSH_BYTES = max(128 MB, 4 × device L2)` | was a bare 128 MB constant justified by *"C500 L2 is 8 MB"*. Still ≥4× here (32 MB L2), so **no bias existed** — but it is now derived rather than assumed, so the next port cannot silently measure warm-L2 |

### 4.1 Verified non-issues

- **The L2 flush was already safe.** 128 MB vs this machine's 32 MB L2 is a 4× margin, and
  `_flush_l2()` is enqueued *before* `start.record()`, so it stays outside the timed region.
  This was the highest-risk item going in; it is clean.
- **At prefill sizes the flush is irrelevant anyway** — measured `flush ≈ noflush` at
  T=8192 (1.370 vs 1.376 ms), because the 192 MiB working set dwarfs 32 MB of L2. It matters
  only at decode sizes, where it is correctly applied.
- **Kernels compile and run unmodified on Triton 3.6 / sm89** — `norm_only` smoke-tested to
  rel_err 2.8e-3 against the fp32 reference.

### 4.2 The shared-memory model is wrong on Triton 3.6 — measured, not assumed

The study models a kernel's SMEM footprint as `num_stages · 2 · BK · (BM + BN)`
(`glm52/kernels/moe_gateup.py:188` and the `_smem_ok` guards elsewhere). That was right for
Triton 3.0. **Triton 3.6 buffers `num_stages − 1` tiles**, verified by launching 68 configs and
reading `CompiledKernel.metadata.shared`:

| config | study model | Triton 3.6 actual | `(st−1)·2·BK·(BM+BN)` |
|---|---|---|---|
| BM128 BN128 BK32 s3 | 48 KB | **32 KB** | 32 KB ✓ |
| BM128 BN128 BK32 s4 | 64 KB | **48 KB** | 48 KB ✓ |
| BM128 BN128 BK64 s3 | 96 KB | **64 KB** | 64 KB ✓ |
| BM128 BN256 BK64 s3 | 144 KB | **96 KB** | 96 KB ✓ |

Exact on **64 of 68** launchable configs; the four misses are all `num_stages=2`, where Triton
keeps a 2-buffer minimum, so `max(2, st−1) · 2 · BK · (BM + BN)` is exact or conservative
everywhere.

**Why this is a fairness defect, not a cosmetic one.** The old model over-predicts by
1.33–1.5×, so the guard rejects configs that are legal here. `BM128 BN256 BK64 s3` is modelled
at 144 KB (rejected against any ceiling) but actually uses 96 KB and launches fine. Every
rejected-but-legal config silently narrows the search grid — and if it narrows one arm of a
fused/unfused pair more than the other, it biases the ratio that is this study's entire output.

### 4.3 Fusion #6's C500 blocker does not exist here

Not benchmarked (needs 12 GB of expert weights), but the *compilability* question needs no
benchmark and is settled. LOG-09 §3 recorded that #6's unfused winner `BM128 BN128 BK32 s4`
required **96 KB fused against C500's 64 KB limit** and was therefore uncompilable — a hard
bar, not a performance effect.

On this device the opt-in ceiling is 101376 B and **Triton 3.6 genuinely reaches it** (a 96 KB
config launched successfully). Under the corrected model the same fused config needs only
48 KB. So the barrier that produced C500's 0.553× is absent on Ada, and the earlier claim that
it was "C500-specific" is confirmed by direct measurement rather than inference.

### 4.4 Known porting constraint

Triton 3.6 requires `@jit` functions to live in a real file (`ValueError: @jit functions should
be defined in a Python file`); Triton 3.0 accepted heredoc/stdin definitions. Affects ad-hoc
test scripts only — every kernel in `glm52/kernels/` is already file-resident.

## 5. The port audit — 76 findings, and one that would have faked the whole result

Six agents audited the shared harness and all five in-scope fusion families statically (no
CUDA — the single GPU was reserved), then a synthesiser verified every claim line-by-line
against the files and discarded the ones that did not hold. Full plan: `log/_port_audit_plan.md`.

**76 findings: 10 blockers, 24 fairness, 12 correctness, 11 perf, 19 cosmetic.**

### 5.1 The one that mattered most

**`bench_f10_merge_resadd.py:410` reuses stale C500 checkpoints with no escape hatch.** It
would have run **zero kernels**, printed "(from checkpoint)" five times, and written a result
file containing **C500 timings wrapped in a freshly-probed RTX 4060 `env` block and
`_meta.device`**. `bench_f01_oproj_resadd.py:465` does the same behind an `F01_FORCE` flag.

I confirmed the payload before removing it:

```
results/c500/_f01_oproj_resadd_ckpt/prefill_t8192.json
    fused_ms 18.383   unfused_ms 15.549   speedup 0.8458
```

That is the C500 study's headline datapoint. It would have been republished as an RTX 4060
measurement, in a file whose every other field correctly identified this machine.

**And my own earlier fix made it worse, not better.** `bench_f01`'s `CKPT_DIR` is
`parents[2]/"results"` and does *not* honour `GLM52_RESULTS_DIR`, while its `record()` does.
Setting that env var alone produces the worst case: C500 checkpoint read from `results/`,
written into `results/rtx4060/` under a genuine 4060 environment block. Isolating outputs
without isolating inputs is worse than doing neither.

Neutralised by moving all three checkpoint directories to `results/c500/`, plus a structural
fix (derive `CKPT_DIR` from `common.RESULTS_DIR`; fence every checkpoint read on
`torch.cuda.get_device_name(0)`).

### 5.2 Grid pruning that biases one arm

The C500 constants do not merely produce wrong absolute numbers — several prune **one arm of a
fused/unfused pair harder than the other**, which manufactures or destroys a speedup:

| finding | effect | direction |
|---|---|---|
| **A1** SMEM ceiling `65536` at 5 sites | deletes 20–38 % of the legal tile grid | in `f04f05`, the unfused router GEMM loses 12/91 configs (13 %) but the `FUSE_TOPK` fused arm — pinned to `BLOCK_E=256` by an assert — loses 24/70 (**34 %**). Biases **against fusion**. |
| **A1** in `f11b` | cap admits **zero** configs with `BLOCK_N=256 AND BLOCK_K≥64` | `BLOCK_N=256` is the fused arm's ideal shape (`sq_redundancy=1`, the only width where sum-of-squares isn't recomputed per n-tile). Deletes the fused kernel's best family; the unfused GEMM is indifferent. Biases **against fusion**. |
| **A2** `threads = num_warps * 64` at 12 sites | overestimates threads 2×, so **underestimates** per-thread registers 2× — on a register file that is **half** C500's | |
| **A2** at `bench_f01:202` (`epi_grid`) | the epilogue kernel is the entire unfused-side overhead; the fused arm with `SPLIT_K==1` has none | under-tunes only the unfused arm → **inflates the fusion win** |
| **A2** at `bench_f04f05:152` (`_norm_ok`) | feeds only the unfused `t_norm`/`t_addn` | **inflates the fusion win** |
| **A2** at `bench_f11:173/186` | admits 28 configs at a true 128 fp32/thread — guaranteed spill | a **spilling unfused norm kernel manufactures a fused win at every regime** |
| **A2** at `bench_f10:68` | rejects `(256,1,w2)` — the exact 32-lane analogue of C500's prefill winner — while admitting shapes where the *fused* kernel hits the register cliff first | biases **both ways** |

The biases run in *both* directions depending on the site, so they would not have cancelled;
they would have produced a plausible-looking table that was wrong fusion-by-fusion.

### 5.3 Blockers

- **B2** — `sys.path.insert(0, "/home/zhangshuhan/fusion")` hardcoded in three benches.
- **B3** — `bench_f11_lazy_prenorm.py` **OOMs at startup**: `main()` calls `make_w13()`
  unconditionally, allocating two 12.884 GB buffers, with no flag to skip. F11a is out of
  scope here, but the process dies before a single F11b number exists. Needs `--router-only`
  and ~15 gated sites; target peak ~0.44 GB at T=8192.
- **B4** — `JITFunction.cache` was **removed in Triton 3.x** (it is `device_caches` now, a
  5-tuple per device). Four sites: two crash outright, two sit inside bare `except Exception`
  and fail **silently**, returning `kernel_stats = {"error": ...}`. On a port whose single
  largest hardware delta is the register file per SM, silently losing `n_regs`/`n_spills`
  removes precisely the diagnostic needed to explain any fused-arm regression.
- **B5** — `reference.expert_merge` peaks at ~3.0 GiB of fp32 transients at T=8192 (two temps
  before the reduction), the tightest point in the port at ~4.97 GB of a ~7.4 GB budget.

### 5.4 Claims the synthesiser rejected

Recorded because a verification pass that never rejects anything isn't verifying:

- The `_FLUSH_BYTES` finding — **already fixed** (§4); two auditors read the pre-edit file.
- `results/_f04f05_norm_router_ckpt/` as a reuse hazard — **false**; that bench only *writes*
  checkpoints, never reads them. Only f01 and f10 reuse.
- "F1 prefill may flip compute→memory once `C_PEAK` is honest" — **not supported**; the
  balance point moves 82.3 → 84.3 FLOP/byte, a 2.5 % shift, so `compute_bound` labels survive.
- The auditors' proposed constants `B_PEAK=128.8e9`, `C_PEAK=11.11e12` — **superseded** by the
  on-disk calibration this session measured (140e9 / 11.81e12); the auditors used my earlier,
  pessimistic `copy_` figure.

## 6. Fixes applied — 157 edits across 11 files

Nine agents, one file each (exclusive ownership, so no two could touch the same file), then an
independent adversarial verifier that **re-derived every grid count offline** by AST-extracting
the real guard source from each bench and replaying it against a stub `env()` — no CUDA, so the
GPU stayed reserved. All files compile; `git diff --stat` confirms no agent edited a file it
did not own.

Two plan errors were correctly overridden in favour of the source, which is the behaviour worth
having:

- the plan listed `(256,w16)` as a wrongly-pruned f01 epilogue config; it is genuinely
  0.5 elem/lane at 32 lanes and correctly rejected on **both** devices;
- the plan wanted `ref_out_hi` routed through `reference.expert_merge` in f10 — but that
  function rounds the merge to bf16, which is exactly what `ref_out_hi` exists to avoid, so
  routing it would have turned `fused_no_round_mid_vs_fp32` into a tautology.

One plan omission was caught: `Path` was *not* already imported in `bench_f03`.

### 6.1 The four pre-flight conditions

The verifier returned **CONDITIONAL GO** with four items, all now closed:

**C3 — `BenchEnv.probe()` degraded silently to C500.** The most dangerous survivor, and it was
in the file I had edited myself. `props.get("warpSize", 64)`, `props.get("max_shared_mem",
65536)`, `props.get("max_num_regs", 131072)` — Triton's property query JIT-builds and dlopens a
C extension on first use, so it is precisely the call that fails on a fresh box. When it did,
every grid was built at **C500 shape** while the result file recorded the real device name
(f01 epi 29 not 28, f04f05 gemm 80 not 97, f11 router 116 not 168). Rewritten so **torch is the
source of truth** — it exposes every field and needs no JIT — with Triton kept only as a
cross-check, plus `probe_ok`, `require_ok()` and a `banner()`. `require_ok()` is called inside
`env()` itself: one choke point protects all six benches, including the two that print no
banner an operator could check.

**Results isolation.** `export GLM52_RESULTS_DIR=.../results/rtx4060` is load-bearing and
nothing enforced it. The verifier found a second instance of the B1 forgery one file over:
`analyze_f11b_arch.py:144` globs `RESULTS_DIR/f11b_*_T*.json` and would have read **C500's
tuned configs**, then reported occupancy for them under a freshly-probed 4060 header.

**A one-sided grid truncation.** `bench_f11`'s `norm_grid` — which feeds only the *unfused*
arm — lacked `num_warps=32`, so it searched **130 of 164** legal configs while f03's identical
space searched all 164. A 21 % truncation of a one-sided grid, biased toward **inflating the
f11b fused win**. Both f11 ladders extended; `norm_grid` now returns 164, exactly matching f03.

**C1 — unsynchronised handoffs in `norm_router.py`.** Assessed and *deliberately not changed*.
Both are same-thread, same-address (`rows` maps identically in the store and the load) and
`REREAD_H1` is already guarded by `nsplit == 1 and bkn == bk`. Changing kernel semantics would
make these numbers less comparable to C500's, and the failure mode is **observable**: f04f05
prints `[F5]/[F4] N/M cfgs timed` and rejections land in `n_failed`. Watched, not pre-empted —
if the fused arm screens out disproportionately, that is C1 firing and the run is repeated with
a `tl.debug_barrier()`.

### 6.2 Known limitations, to be stated in the report rather than papered over

- `bench_f11b_sweep` records no `n_failed` for either router arm, so an asymmetric fused-side
  compile failure is **undetectable there**. State it; do not claim symmetric budgets.
- f04f05's fused arm now searches **226** configs against the unfused chain's **150** (C500:
  160 vs 138) because of the `NORM_BK` ladder fix. The JSON's `fairness.grids` records live
  counts; the report must too.
- `traffic.py` documents exactly one `compute_bound` label flip (F5 @ decode_bs256, 83.0
  FLOP/byte, inside the 82.3 → 84.3 window). Footnote it; do not tabulate C500-vs-4060 labels
  as if identical.
- **Out of bounds for this campaign:** F6, F8, F9, `bench_layer*.py`, `diag_f10*.py`. All still
  carry `SMEM_LIMIT = 65536` and `* 64`; running any of them on this device produces
  C500-capped grids.

## 7. The campaign

Serialized — one GPU, where the C500 study had four with one lane each. `./run_4060.sh`, order
chosen by the verifier (self-contained and low-risk first, so a harness bug surfaces cheaply):

```
f03 -> f10 -> f01 -> f04f05 -> f11 --router-only    [then f11b_sweep, analyze_f11b_arch]
```

Watched observables:
- **f04f05**: `[F5]/[F4] N/M cfgs timed` — a fused variant screening out far more than `router`
  means C1 is firing.
- **f11**: `router fused: coarse N(Xf)` vs `router unfused: coarse N(Yf)` — `X ≠ Y` means the
  unmodelled cross-warp reduction scratch (A6) is firing, which is **new risk on this
  hardware**: halving the warp width doubles the warps spanning `BLOCK_K`, which is exactly
  when that scratch appears.

## 8. Results — #3 and #10 (complete)

Both families completed and their result files carry genuine 4060 environment blocks
(`warp_size 32, num_sm 24, smem_bytes 101376, regs_per_sm 65536`). The C500 originals in
`results/` are byte-identical to the `results/c500/` backup — **nothing was overwritten**.
Every tuning round in both families reported `(0 fail)`: **60/60 with zero compile failures**,
so no asymmetric-grid effect is in play for these two.

| regime | T | working set | C500 L2 (8 MB) | 4060 L2 (32 MB) | **#3** C500 → 4060 | **#10** C500 → 4060 |
|---|---|---|---|---|---|---|
| decode_bs1 | 1 | 0.05 MB | fits | fits | 1.098 → **1.455** | 1.175 → **1.545** |
| decode_bs32 | 32 | 1.5 MB | fits | fits | 1.107 → **1.412** | 1.181 → **1.216** |
| **decode_bs256** | 256 | **12 MB** | **spills** | **fits** | 1.081 → **0.978** | 1.147 → **1.070** |
| prefill_t2048 | 2048 | 96 MB | spills | spills | 1.249 → 1.075 | 1.204 → 1.088 |
| prefill_t8192 | 8192 | 384 MB | spills | spills | 1.315 → 1.245 | 1.198 → **1.224** |

*(working set = the four `[T, 6144]` bf16 activation tensors the fused #3 kernel touches)*

### 8.1 The shape of the difference is mechanistic, not noise

**`decode_bs256` is the single regime where L2 residency differs between the two machines** —
12 MB spills C500's 8 MB cache but sits comfortably inside the 4060's 32 MB — and it is exactly
the regime where both fusions lose the most ground, with #3 flipping to an outright **loss**
(0.978×). This is the audit's A7 prediction confirmed by measurement: *when the intermediate is
L2-resident, the traffic the fusion eliminates was never DRAM traffic in the first place*, so
there is nothing to save — while the fusion still pays its costs.

At the small-T end both fusions do **markedly better here than on C500** (1.455 / 1.545 at
bs1 against 1.098 / 1.175). That win is not bandwidth at all — at T=1 the working set is 48 KB
— it is **kernel-launch elimination**, and on 24 SMs at a locked 1020 MHz a launch is a far
larger fraction of a tiny kernel's runtime than on 104 CUs at full clock. `f10`'s own ceiling
column says so directly: **273 % of its bandwidth roofline at decode_bs1**, i.e. the roofline
cannot explain the win because the win is not made of bytes.

At prefill both machines spill L2 and the numbers reconverge (#3 1.245 vs 1.315; #10 1.224 vs
1.198 — the 4060 slightly *ahead* on #10).

So the value of these vector fusions is bracketed by **two different mechanisms at the two
ends** — launch overhead at small T, DRAM traffic at large T — and it collapses in the middle,
at whatever batch size makes the intermediate exactly L2-resident. That crossover is a property
of the cache, so it **moves with the hardware**: it sits below bs256 on C500 and above it here.

### 8.2 What this does to the C500 conclusion

LOG-09 ranked #3 and #10 as the study's two reliable winners, wins at *every* regime. On this
device that is no longer true: both have a regime where they do nothing or lose. The ranking
survives — they are still the best two of the five measured here — but "wins everywhere"
was a C500 statement, not a general one.

## 9. All five families — and an adversarial review that killed three of my claims

Full campaign complete. Then five skeptics attacked the results before publication, and an
adjudicator verified their findings against the files. **Three of my four claims did not
survive as stated.** What follows is what survived, plus the confirming experiments I ran
afterwards on the freed GPU.

| fusion | machine | d_bs1 | d_bs32 | d_bs256 | p_t2048 | p_t8192 |
|---|---|---|---|---|---|---|
| **#3** | C500 | 1.098 | 1.107 | 1.081 | 1.249 | 1.315 |
| | 4060 | 1.455ᵃ | 1.412ᵃ | **0.978** | 1.075 | 1.245 |
| **#10** | C500 | 1.175 | 1.181 | 1.147 | 1.204 | 1.198 |
| | 4060 | 1.545ᵃ | 1.216 | 1.070 | 1.088 | 1.224 |
| **#1** | C500 | 0.996 | 0.999 | 1.005 | 0.871 | **0.846** |
| | 4060 | 1.002 | 1.002 | 0.999 | 1.014 | **1.014**ᵇ |
| **#5** | C500 | 0.473 | 0.464 | 0.467 | 0.475 | 0.680 |
| | 4060 | 0.738ᶜ | 0.771ᶜ | 0.758ᶜ | **1.063**ᶜ | **1.300** |
| **#4** | C500 | 0.391 | 0.379 | 0.388 | 0.437 | 0.669 |
| | 4060 | 0.668ᶜ | 0.673ᶜ | 0.811ᶜ | 1.002ᶜ | **1.232** |
| **#11b** | C500 | 0.684 | 0.680 | 0.771 | 1.096 | 1.127 |
| | 4060 | 0.781ᵃ | 0.738ᵃ | 0.898 | **1.411** | **1.545** |

ᵃ launch-dominated and tick-quantised — see §9.2. ᵇ **corrected from the campaign's 1.027**,
see §9.1. ᶜ unfused router GEMM under-tuned at decode; true values are worse for fusion — §9.3.

### 9.1 #1's headline number was wrong, and the corrected one is better evidence

The campaign reported **1.027** at prefill_t8192. That is **above the cell's own physical
ceiling** and it is an artifact of thermal drift: `bench_f01:408-409` times the entire fused
arm and then the entire unfused arm, and the run log shows the coarse sweep measuring the
fused chain at **137.25 ms** against the final measurement's **167.20 ms** — a 22 % slowdown
*within one run*. Locked clocks are a cap, not a floor; a laptop 4060 still throttles.

Re-measured with the arms **interleaved A/B/A/B** (n=120, so monotone drift cancels in the
paired ratio), clocks logged flat at 1020 MHz, 68–72 °C:

```
fused   p50 136.784   unfused p50 139.106   PAIRED p50 1.0143  (trimmed 1.0145)
```

And decomposed into primitives at the shared config `BM64 BN64 BK64 GM8 SK1 w8 s3`:

| primitive | ms |
|---|---|
| `gemm(FUSE_RESADD=True)` | **134.969** |
| `gemm(FUSE_RESADD=False)` | **135.050** |
| epilogue kernel | 1.956 |

implied speedup `(135.050 + 1.956) / 134.969 = ` **1.0151**, matching the interleaved 1.0143.

My "ceiling violation" was a **wrong ceiling**: I had charged the fused GEMM a 0.72 ms residual
read, but at 12.2 TF/s this GEMM is compute-bound and that read hides entirely under the math.

**The corrected result is stronger than the wrong one.** Folding a residual add into this
GEMM's epilogue costs **+0.06 %** of GEMM time on Ada (0.081 ms of 135). The same fusion on
C500 cost **+22.8 %** (107.5 → 87.4 TF/s, LOG-10 §1). Same kernel source, same fusion, same
algorithm — **380× difference in what it costs.**

### 9.2 The decode_bs1 numbers are launch overhead, measured

Measured on this machine: **kernel launch L = 3.36 µs**, harness floor O = 2.93 µs, and the
CUDA event timer quantises to **1.024 µs — 200/200 sampled timings are exact multiples of it**
(4× coarser than C500's 0.256 µs).

f03 decode_bs1 re-measured: fused **12.0 ticks**, unfused **16.0 ticks** — integers, so the
ratio is quantised to ±8 %. Gap 4.10 µs against a launch cost of 3.36 µs: **82 % of the win is
one eliminated kernel launch**. At T=1 the working set is 12 KB, so essentially none of it is
bandwidth.

So CLAIM 4's *arithmetic* survives — one launch does account for the gap — but the numbers
1.455 / 1.545 should be reported as "launch-dominated, ±8 % quantisation", not as precise
speedups. (An amortised 100-iterations-per-window variant returns 1.911, but that measures
Triton's **Python-side** launch throughput with the GPU idling, not kernel time; recorded so
nobody re-derives it and thinks it is a GPU result.)

### 9.3 What else the review killed

- **CLAIM 1 as stated is dead.** "Every GEMM fusion improves on Ada" — 9 of 20 GEMM-fusion
  cells here are still **losses of 10–33 %**, all at decode. The defensible claim is a
  **prefill** flip, not a universal one.
- **f04f05's unfused router GEMM is under-tuned at decode.** `bench_f04f05` offers
  `BLOCK_K ∈ {32,64}`, no GROUP_M axis and no refine; `bench_f11` tunes the *byte-identical*
  op with `BLOCK_K ∈ {32,64,128}` + GROUP_M + refine and gets **0.0522 ms vs 0.1116 ms** at
  bs1 — a 2.14× gap that does not exist on C500 (1.02×). This makes the F4/F5 **decode losses
  worse**, and puts a real uncertainty band on t2048. It does not touch t8192 (1.011×).
- **CLAIM 3 as written is false.** bs256 is *not* "the single regime where L2 residency
  differs" — residency differs at prefill_t2048 too, and my "12 MB working set" figure was
  wrong for #10. The mechanism is real and directly measured; the localisation was not.

### 9.4 The dangerous alternative, settled

The worst possibility was that this is **badly-tuned-vs-well-tuned** rather than a hardware
comparison, since the port fixed real grid bugs. It is not, on four independent lines:

1. **C500's 65536 SMEM ceiling was correct for C500.** The MACA runtime itself reports
   `OutOfResources: ... Required: 69632, Hardware limit: 65536`, and required amounts
   systematically *exceeded* the guard's own estimate — so the C500 filter never pruned a
   config C500 could have run. `warpSize=64` came from the device probe, not a literal.
2. The one genuinely wrong C500 guard is worth **8 configs of 88**, and all 8 are `BLOCK_E=256`
   shapes — i.e. it would have helped C500's **fused** arm.
3. **Every 4060 prefill winner was legal on C500** (checked against C500's as-run guards).
4. **The 4060's baselines are harder to beat.** Its unfused router GEMM runs at **102 % of its
   Triton ceiling**; C500's ran at **64 % of its Triton ceiling and 32 % of its vendor BLAS**.
   The classic way to fabricate a fusion win — a slack baseline — is closed here and *open*
   there. If anything C500's negatives were conservative.

### 9.5 The measurement that actually carries the result

Not the vendor-BLAS ratio. `isolation_fuse_on_vs_off_same_cfg` — one kernel source, one
config, one launch, same tensors, only the `FUSE_NORM` constexpr flips (code **unchanged by
the port**):

| prologue instruction cost, same config | bs1 | bs32 | bs256 | t2048 | t8192 |
|---|---|---|---|---|---|
| **C500** router | 61.7 % | 63.0 % | 17.9 % | 21.5 % | 12.4 % |
| **RTX 4060** router | **2.9 %** | **8.3 %** | **7.6 %** | **4.0 %** | **3.0 %** |

And against the extra activation pass the two-pass fused kernel physically requires:

| | measured tax | traffic model | deficit |
|---|---|---|---|
| C500 prefill_t2048 | +193 % | +13 % | **14×** |
| C500 prefill_t8192 | +139 % | +23 % | **6×** |
| 4060 prefill_t2048 | +49 % | +30 % | 1.6× |
| 4060 prefill_t8192 | +26 % | +33 % | **0.8×** |

**The correct statement of this study's result:** *on Ada the cost of folding a normalisation
into a Triton GEMM is approximately the traffic it adds; on C500/MACA it was 6–14× the traffic
it adds.* Formulation matters as much as device — #11b's deferred-scale prologue costs 3 % on
Ada while #4/#5's naive two-pass costs 21–26 %, because the latter re-reads the activation.

### 9.6 What may and may not be claimed

**May:** the GEMM-fusion penalty that dominated the C500 study is largely **toolchain**, not
architecture — quantified by the same-config isolation measurement and by #1's +0.06 % vs
+22.8 %. Note sm89 has **no warp specialization either**, so the C500 study's own explanation
cannot account for the same fusions now winning on a device that also lacks it.

**Must not:** claim the effect is *identified* as "the MACA backend". MACA-vs-CUDA is perfectly
confounded with **Triton 3.0 vs 3.6** (six minor versions). Separating them needs the version-
control experiment (oldest Triton that compiles these kernels on sm89, re-read
`isolation_fuse_on_vs_off_same_cfg`) — **not run**.

**Must not:** claim GEMM fusion is now good in general. It still loses 10–33 % at every decode
regime here, and #4/#5's decode losses are understated by an under-tuned baseline.

## 10. Outstanding

- `glm52/traffic.py` hardcodes `C_PEAK = 107e12`, `B_PEAK = 1.30e12` (C500). Every roofline
  ceiling is wrong here until these are made device-derived. **Deliberately not yet edited** —
  a parallel audit covers this file and its findings are pending.
- Full static port audit of the shared harness and all five fusion families is running
  (6 agents, static analysis only — the single GPU is reserved for serialized benchmarking).
- Benchmarks not yet started; they must be **serialized**, since this host has one GPU where
  the C500 study had four with one lane each.

*Next: apply the audit's fix list, recalibrate `traffic.py`, then run the five families
sequentially.*
