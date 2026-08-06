# LOG-16 — Repairing fusion #11 (#11a / #11b / #11b′) after the H200 re-run lost it

**Date** 2026-08-04, re-run 2026-08-05 · **Status** §7 is final: 3 of 28 cells published, on a graph-replay basis; #11a unmeasurable
**Trigger** the H200 re-run wrote `f11_lazy_prenorm.json` with `complete: true` and an **empty
`rows` array** — the campaign's headline experiment produced nothing.

---

## 1. Why this matters more than the other families

H200 is the **first device in the study with warp specialization**, which is the mechanism
"Towards Free Normalization" (Zhou et al. 2026) depends on. On C500 and the RTX 4060 the
in-mainloop sum-of-squares *displaced* MMA issue slots instead of overlapping them — measured
on C500 as **+0.78 % arithmetic for −12.3 % throughput**, a 16× amplification — and LOG-09
attributed that to the missing hardware lane. #11 on H200 is the direct test of that
attribution. It is currently **unanswered**.

## 2. Three independent defects, confirmed from the run record

### D-A — #11a is numerically wrong, config- and scale-dependently

```
[screen w13F/coarse] 180 offered -> 133 valid (27 compile-fail, 20 wrong-answer)
[screen w13U/coarse] 180 offered -> 153 valid (27 compile-fail,  0 wrong-answer)
```

The **fused** w13 arm has 20–34 wrong-answer configs; the **unfused** arm has zero. After
tuning, the post-tuning check fails at `moe_fused rel_err` **0.37 / 0.41 / 0.53 / 0.77**
against `tol 0.02` (bs512 / bs1024 / t2048 / t8192) — while an *earlier* check in the same run
passes at **3.78e-03**. So some configs are correct, some are not, and the tuner's winner is
among the wrong ones.

**What I ruled out by local reproduction** (correctness is device-neutral, so the RTX 4060
serves):

| shape | configs | result |
|---|---|---|
| E=8, T=64 | 48 | **all correct**, rel_err 4.07e-03 |
| E=32, T=512 | 16 | **all correct**, rel_err 3.52e-03 |

Both used a *synthetic regular* dispatch (`topk_ids = (t·TOPK + k) % E`), which covers every
expert evenly. The failure therefore needs something those runs lacked. Ranked hypotheses:

1. **The real 256-expert dispatch leaves experts with zero rows.** The production `topk_ids`
   comes from the router (sigmoid + noaux_tc top-8) and is data-dependent, so
   `moe_align_block_size` can emit blocks that are entirely padding. A fused block of all
   padding reduces `sq = 0`, giving `rstd = rsqrt(0/K + eps) ≈ 316` rather than a real scale.
2. **Row sampling in the checker.** `reference_rows(prob, n_sample=1024)` samples when
   `rows > 1024` — a threshold crossed at *exactly* the failing regimes (bs512 → 4096 rows).
   If a sampled row was never written by the kernel, `c_f` holds whatever it was allocated
   with, and the rel_err is meaningless. NaN-filling the output separates "unwritten" from
   "wrong", which are different bugs.
3. `num_valid_tokens` / `token_mask` semantics at scale — does the kernel's C-row index
   (`safe_token`) agree with the checker's (`idx // TOPK`, `idx % TOPK`) for every row?
4. `sq` redundancy across n-tiles — is the full-K reduction done exactly once per output row
   for every `BLOCK_N` tiling?

### D-B — one arm's correctness failure destroys the others' valid data

`bench_f11_lazy_prenorm.py:895` raises on a failed validation. That propagates out of
`run_regime`, **past the `B.ckpt_save` that would have persisted the regime**. #11b's router
results were fine in every regime and were discarded along with #11a's failure. The file then
recorded `complete: true` with zero rows, which is what made the loss invisible.

### D-C — an all-configs-rejected screen still publishes a timing

```
[screen rstd] 124 offered -> 0 valid (124 compile-fail, 0 wrong-answer)
!! [rstd] screening rejected EVERY config; timing unscreened
[rstd] 124/124 cfgs timed -> 0.0345 ms
```

A number for a kernel where nothing compiled. Same for `norm`, in every regime.

## 3. Decisions taken with the user

Asked rather than assumed, since each changes the work materially:

| question | decision |
|---|---|
| scope | **fix all three arms**, chasing the #11a numerical bug properly |
| the four excluded whole-layer configs (`O_f11ab`, `P_f10_f11ab`, `Q_f8_f11ab`, `R_f1_f10_f11ab`) | **re-measure them**, once #11 passes its correctness screen |
| all-configs-rejected screen | **fatal for that kernel** — record the reason, publish no number, continue |

## 4. Deliverables

- kernel fix in `glm52_h200/kernels/lazy_prenorm.py` (only if the fault is really there —
  a fix applied to the wrong file is worse than none)
- harness fixes in `glm52_h200/bench/bench_f11_lazy_prenorm.py`: per-arm validation status,
  `--with-w13` / `--router-only` so #11a can never again take #11b/#11b′ down with it, and
  `complete: false` with a per-arm block whenever any arm fails
- **`run_f11_h200.py`** in the repo root: a standalone #11-only re-runner with the same
  `--gpu` idle-selection and tenanted-card refusal as `run_h200.py`, resumable per regime,
  which then merges the four layer configurations back into `layer_configurations.json`
  idempotently and without overwriting rows that already measured cleanly

## 5. Verification bar

The last two H200 hand-offs each shipped a defect that only a real run exposed — a key-name
mismatch (`ratio_p50` vs `paired_speedup_p50`) that crashed every regime, and a checkpoint
arity mismatch that silently dropped the device fence. So the verification phase must
**demonstrate the #11a fix by running the correctness repro at E=256 with a real
router-derived dispatch on the local sm_89 box**, not merely assert it. An unverified kernel
fix is worse than an acknowledged bug, because it will be trusted.


---

## 6. The repaired re-run (2026-08-05)

`run_f11_h200.py` ran on the H200. **All seven regimes produced rows** (previously zero), with
`unmeasurable: none` and `complete: false` — the last being honest rather than a defect: one
arm still fails, and the file now says so instead of claiming completeness with an empty body.

### 6.1 The harness fixes worked

The two screens that rejected 100 % of configs in the last campaign now pass everything:

| screen | before | after |
|---|---|---|
| `norm` | 164 offered → **0 valid** | 14 → **14 valid** |
| `rstd` | 124 offered → **0 valid** | 10 → **10 valid** |

Their trigger was never a tolerance — it was `TypeError: 'function' object is not iterable`
from `screened_autotune` iterating a bare callable, booked as a compile failure and then
"rescued" by a fallback that published a timing anyway. Both the cause and the rescue are gone.

### 6.2 Results, all three arms

| regime | **#11b** router | **#11a** w13 | **#11b′** half-fused |
|---|---|---|---|
| decode_bs1 | 2.155 | 1.010 | 1.564 |
| decode_bs32 | 0.847 | *(failed)* | 1.056 |
| decode_bs256 | 2.211 | 0.992 | 1.488 |
| decode_bs512 | 2.135 | 0.950 | 1.640 |
| decode_bs1024 | 2.034 | 0.961 | 1.521 |
| prefill_t2048 | 2.004 | 0.859 | 1.463 |
| prefill_t8192 | 1.593 | 0.616 | 1.329 |

**#11a still loses**, as it has on every device in this study (C500 0.48–0.60, and here
0.62–1.01). **#11b′ wins everywhere** (1.06–1.64). **#11b reads 2.0–2.2×**, which is far above
C500 (0.68–1.13) and the RTX 4060 (0.74–1.55) and is *not yet cleared* — see §6.4.

### 6.3 The headline experiment finally ran, and its answer is negative

This is what the H200 was for. The 2×2 at one shared config, decode_bs256 / router:

| arm | ms |
|---|---|
| unfused | 0.05021 |
| fused, classic mainloop | 0.05181 → **+3.19 %** |
| unfused + warp specialization | 0.05546 |
| fused + warp specialization | 0.05702 → **+2.83 %** |

Warp specialization made **both** arms ~10 % slower (−10.07 % fused, −10.45 % unfused) and did
not absorb the reduction: the fused arm pays ~3 % with it and ~3 % without.

Recorded verdict: *"Specialization does NOT absorb the reduction here."*

That is a direct test of LOG-09's attribution, which blamed C500's +12.4–61.7 % in-mainloop
cost on the **absence** of a warp-specialization lane. On the first device that has one, the
lane exists, is applied, and does not help. The reduction's cost on H200 is small (~3 %) — but
the 2×2 shows that is a property of the device and toolchain, not of warp specialization.

### 6.4 Why #11b's 2.0–2.2× is not yet reportable

#11b compares a **single fused kernel** against an **unfused chain** (a separate rmsnorm
kernel plus the GEMM). Most of a 2× win at decode should therefore be the eliminated kernel
launch and activation pass, not the mainloop — the mainloop tax is the +3.19 % measured above.
Until that decomposition is confirmed from the recorded per-kernel timings, the number is an
observation, not a result. An adversarial verification is running against the raw JSON, also
checking:

- whether the **invariance test** (the D-A mitigation) actually ran and passed in every regime,
  and what `BLOCK_M` each fused winner chose — if every winner landed in {16, 32} it matters
  whether small tiles won on merit or the grid was restricted, the latter being a one-sided bias;
- whether **warp specialization was genuinely applied** in the headline 2×2 rather than
  silently skipped — a "WS does not help" verdict from a kernel where WS never ran is worthless,
  so `ttgir_mentions_wgmma` / `ptx_mentions_wgmma` must confirm it;
- whether the **layer merge** added exactly the four previously-excluded configurations
  (`O_f11ab`, `P_f10_f11ab`, `Q_f8_f11ab`, `R_f1_f10_f11ab`) without disturbing any row that
  already measured cleanly (a timestamped pre-merge backup exists);
- **which estimator** each number uses — the campaign publishes the sequential ratio while a
  paired one exists, and elsewhere they disagreed by up to 13.8 %.


---

## 7. The gated re-measurement, and what may be published (2026-08-06)

`f11_publish.py` re-measured #11 on an idle H200 (GPU 0 of 8, 0 MiB used, 0 % util) with four
gates the old harness lacked. Three independent adjudicators then re-derived every cell from
the raw JSON rather than trusting the script's own flags — which was the right call, because
they overturned three of them.

### 7.1 The calibration gate passed for the first time

| | blocked run | this run |
|---|---|---|
| harness floor | 39.87 µs | **15.321 µs** (bar 20.0) |
| launch cost | 9.02 µs | 8.327 µs |
| floor / launch | 4.42 | **1.84** (bar 3.0) |

That is what makes this the first #11 measurement whose *basis* is sound. Everything below
follows from having a trustworthy floor.

### 7.2 Published: 3 cells, and only on a graph-replay basis

| regime | arm | published | basis |
|---|---|---|---|
| decode_bs1024 | #11b | **0.955** | CUDA-graph replay |
| prefill_t2048 | #11b | **1.377** | CUDA-graph replay |
| prefill_t8192 | #11b | **1.396** | CUDA-graph replay |

Everything else — 25 cells — is **empty with a stated reason**, never omitted and never filled.

**Why the graph basis and not the wall clock.** Each `wall` figure exceeds the largest ratio
its own graph work plus its own calibrated launch and floor can produce: prefill_t2048 by
**34 %** (1.833 against a self-consistent bound of 1.365), prefill_t8192 by 13 %, and the four
decode cells by 2.08–2.15×, above the hard `n_unfused/n_fused = 2.0` asymptote that no launch
cost can breach. The wall numbers are also **non-monotone in T**, which the mechanism forbids.
The graph numbers, by contrast, sit exactly where the study's other devices bracket them
(C500 1.096 → **H200 1.377** → RTX 4060 1.411 at t2048) and equal the theoretical maximum
`(T_gemm + T_norm) / T_gemm_fused` for this fusion.

The published rows carry an explicit warning that their timing basis differs from every other
row in the file, so `fused_ms` must not be compared across rows.

**decode_bs1024 publishes a regression.** 0.955 means fusion *loses* 4.5 % once launch
overhead is amortised. Its 1.147 wall figure was the harness floor plus one saved launch — a
launch-count win, not a work win. Publishing the regression is the honest reading.

### 7.3 #11a: unmeasurable, not measured-and-lost

**0 of 7 cells.** Six fail the strict invariance screen outright — `BLOCK_M` rel_err up to
**1.18e-01** against tol 1e-5 — and the seventh was overturned: at decode_bs1 its `BLOCK_M`
probe recorded `ran: false`, so the one axis the wgmma defect lives on was **never tested**,
and the arm has no ceiling of any kind. Untested is not invariant.

This changes the claim the study makes. #11a on H200 is not "measured and lost at 0.62–1.01";
it is **not measurable on this toolchain**, because its winning configs cannot be shown to
compute the right answer. The two are different statements and only the second is supportable.

One finding deserves recording. At prefill_t8192, four independent perturbations —
`BLOCK_M`, `BLOCK_N`, `GROUP_M`, `num_stages` — all report `rel_err = 0.02628391422331333` to
**17 significant figures**. Four probes agreeing with each other to the last bit while
disagreeing with the tuned winner by an identical amount does not describe four miscompiles;
it says **the tuned winner is the outlier**. The screen rejected the right config.

### 7.4 Two defects in my own script, found by the adjudication and now fixed

Recorded because both would have produced plausible numbers with nothing behind them:

1. **The `rstd` producer was tuned with `lambda: (True, "")`** — an unconditional pass. No
   config was ever compared against anything, and `#11b′` inherited a figure with **no
   numerical evidence at all**. It now verifies against `rsqrt(mean(h1²)+eps)`.
2. **`#11b′` had no invariance screen, and its ceiling omitted `router_half`'s re-read of
   `h1`** — making the bound 55 % too generous at t8192, which is how the arm looked bounded
   when it was not. Both corrected.

`#11b′` is therefore **unpublished for now** and rescuable on a re-run, not refuted.

### 7.5 What the study can now claim about #11 on H200

That fusing an RMSNorm into the router GEMM is **worth ~1.38–1.40× at prefill** on real work,
consistent with the other two devices; that it is **neutral-to-negative at decode** once launch
accounting is removed; that fusing it into the **w13 grouped GEMM cannot be validated at all**
on this toolchain; and — from the 2×2 in §6.3 — that **warp specialization does not absorb the
reduction**, which was the hypothesis H200 was brought in to test.
