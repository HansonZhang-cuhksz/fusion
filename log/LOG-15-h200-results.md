# LOG-15 — H200 campaign: log handling and first findings

**Date** 2026-08-04 · **Device** NVIDIA H200 sm_90, GPU index 3 of 8, `hgx-h200-233`
**Run** all eight families completed, including the whole-layer combination sweep
**Status** log split done; report generation and its adversarial verification in progress

---

## 1. The 107 MB log is a result, not a formatting accident

`log/run_h200/f04f05.log` came back at **107.7 MB / 1,240,800 lines**, over GitHub's 100 MiB
limit, and had been `.gitignore`d as a stop-gap — which meant the record of the study's
most-measured family was not in the repository at all.

Splitting it by fusion alone would have produced two ~54 MB files, because the size is not
F4-vs-F5 volume. **Only 189 of 1,240,800 lines are log — 0.015 %.** The rest is Triton
compiler output, dominated by one internal assertion repeated once per attempted config:

```
python: .../hopper/lib/Transforms/WarpSpecialization/WSLowerToken.cpp:73:
        processProducerCommitOp(...): Assertion `false' failed.
glm52_h200/kernels/norm_router.py:317:0: error: Failures have been detected while
        processing an MLIR pass pipeline
glm52_h200/kernels/norm_router.py:317:0: note: Pipeline failed while executing
        [`NVGPUWarpSpecialization` on 'builtin.module' operation]
```

each followed by an MLIR reproducer whose `pipeline:` string alone is ~2.5 KB, plus a full
TTGIR dump.

### 1.1 What that assertion means for the study

**Triton's warp-specialization pass crashes on these kernels.** Not "warp specialization is
unavailable" — the preflight probed `tl.range(warp_specialize=True)` and it compiled and
launched fine on a toy kernel. It fails on the *real* fused kernels, in the compiler, with an
internal assertion.

Assertion counts per family log, and the pattern is not random:

| family | assertions | what it fuses |
|---|---|---|
| f04f05 | 563 | norm **into a GEMM** |
| f11 | 418 | sum-of-squares **into a GEMM mainloop** |
| f06 | 309 | SwiGLU **into a GEMM epilogue** |
| f08f09 | 267 | merge **into a GEMM epilogue** |
| f01 | 90 | residual add **into a GEMM epilogue** |
| **f03** | **0** | vector + vector |
| **f10** | **0** | vector + vector |

**Every family that touches a GEMM mainloop spams the assertion; the two pure vector fusions
produce none** (f03 and f10 are 89 % readable log, the GEMM families under 0.3 %). Concentrated
on `norm_router.py:317`, with 8901 hits.

The visible cost inside the tuner is large: `[screen F5] 557 offered -> 349 valid (208
compile-fail, 0 wrong-answer)` — **208 of 557 configs in the fused arm never compiled.** Note
`0 wrong-answer`: nothing produced incorrect results, so the numeric screen is clean; this is
purely a compilation loss.

This matters because H200 was supposed to be the device that finally tested the paper's
mechanism. Warp specialization exists here — and Triton cannot apply it to the kernel shape
the technique needs. Whether that changes the verdict is for the report; that it happened is
recorded here.

*(It is not a total loss: some winning configs DO carry `'warp_specialize': True`, and the
router winner at decode_bs1 selected `'num_ctas': 2`, so clusters and warp specialization are
both reachable when the pass does not assert.)*

## 2. The split

`tools/split_h200_log.py` reproduces the whole file as three committed artifacts:

| file | size | content |
|---|---|---|
| `f04.log` | 12 KB / 147 lines | readable log, fusion #4 view |
| `f05.log` | 12 KB / 147 lines | readable log, fusion #5 view |
| `f04f05_compiler.log.gz` | **9.0 MB** | all 1,240,611 compiler lines, gzipped |

Design notes worth keeping:

- **Allowlist, not blocklist.** Blocklisting the compiler output was tried first and leaked —
  MLIR attribute definitions (`#blocked = #ttg.blocked<...>`), module terminators (`#-}`) and
  the Python source echoes inside tracebacks all slipped through, each needing another pattern.
  The readable log has a tiny fixed grammar, so recognising *it* and quarantining everything
  else is shorter and fails safe: an unrecognised line goes to the compiler archive, where it
  is preserved, not into a "readable" log it would corrupt.
- **Both arms travel together.** `[F4]`/`[F4+topk]` are the fused arms and `[U4]`/`[U4+topk]`
  the unfused ones; a per-fusion log takes both, or it holds a speedup with no denominator.
- **Shared context is duplicated, not split.** The header, regime banners and the shared
  baselines (`[norm]`, `[add+norm]`, `[router]`, `[topk]`, `[grid …]`) go to *both* files,
  because each fusion is scored against them and a log without them cannot be audited alone.
- The compiler stream is **archived, not discarded** — it is the evidence for §1.1.

`log/run_h200/.gitignore` now keeps only the raw `f04f05.log` out, and explains why the three
replacements carry the same information. The other family logs (f01 4.1 MB, f06 6.7, f08f09
6.1, f11 9.1) have the same 99 %-compiler-output character but are under the limit, so they
are committed unsplit.

## 3. First whole-layer numbers — measured, two passes

The H200 is the first device since C500 where the whole-layer combination sweep could run
at all (the RTX 4060 could not fit the 19.3 GB expert set). Two independent passes, tie
protocol per `LOG-11`:

| regime | best set | layer ms | vs all-unfused |
|---|---|---|---|
| **decode_bs1** | **#3 + #9** | 0.4626 | **1.2007×** |
| decode_bs32 | #1+#6+#9 (greedy) | 2.9155 | 1.0089× |
| decode_bs256 | **#11b** | 4.6668 | 1.0219× |
| decode_bs512 | **#4** | 4.9483 | 1.0185× |
| decode_bs1024 | #1+#6+#9 (greedy) | 5.4915 | 1.0111× |
| prefill_t2048 | #3 + #10 | 6.3640 | 1.0148× |
| prefill_t8192 | #3 + #10 | 16.0569 | 1.0071× |

Each winner is separated from its runner-up in both runs (only the top row carries
`tied_with_best`, except at decode_bs1 where `#3+#9` and greedy tie with each other).

**These are not yet cleared for reporting.** Three of them contradict findings the earlier
campaigns stated strongly, and an adversarial verification against the raw JSON is still
running:

1. **decode_bs1 at 1.20×** is two orders of magnitude above C500's best layer-level gain
   (1.0046×). Plausible mechanically — the H200 layer is only 0.46 ms at bs1, so fixed glue and
   launch costs are a far larger share than on a 0.86 ms C500 layer — but it needs checking.
2. **Greedy fusion wins two regimes.** C500's single strongest operational finding was
   *"fusing everything is worse than fusing nothing at every regime,"* by up to 48 %.
3. **#4 and #11b win regimes.** On C500 these were the study's *worst* fusions (0.21–0.68×).

Also outstanding: some `summary.json` cells carry a `SEQUENTIAL` flag — the arms were not
interleaved A/B/A/B — which is exactly the condition that produced a physically impossible
speedup on the RTX 4060. Those cells must be marked in the report, not averaged into it.
