# LOG-16 — Repairing fusion #11 (#11a / #11b / #11b′) after the H200 re-run lost it

**Date** 2026-08-04 · **Status** diagnosis + fix in progress
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
