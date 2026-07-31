# LOG-11 — The optimal fusion combination per regime (whole-layer measurement)

**Date** 2026-07-28 · **GPU** MetaX C500, exclusive · **Method** end-to-end pipeline timing,
`glm52/bench/bench_layer.py` + `glm52/bench/bench_layer_ab.py`
**Data** `results/layer_configurations{,_ab,_ab2}.json`, `report/layer_optimal_per_regime.csv`

---

## 1. Answer

| regime | optimal combination | layer time | vs all-unfused | confidence |
|---|---|---|---|---|
| **decode_bs1** | **#3 + #10** | 0.8564 ms | **1.0046×** | borderline — one of two runs ties it with doing nothing |
| **decode_bs32** | **#3 + #8** | 8.4361 ms | **1.0037×** | tied with `#3+#9` and `#8`; the *family* beats baseline in both runs |
| **decode_bs256** | **#8** (or **#3 + #8**) | 13.9625 ms | **1.0037×** | tied with `#3+#8`, `#3+#9`, `#9` |
| **decode_bs512** | **#3 + #10** | 16.4796 ms | **1.0015×** | tied with `#10` and `#1` in both runs; separated from baseline |
| **decode_bs1024** | **#3 + #10** | 21.5194 ms | **1.0024×** | tied with `#10` and `#1` in both runs; separated from baseline |
| **prefill_t2048** | **#3 + #10** | 27.8321 ms | **1.0024×** | solid — both runs, separated from baseline |
| **prefill_t8192** | **#3 + #10** | 80.6149 ms | **1.0039×** | solid — both runs, separated from baseline |

Where "tied" appears, the configurations listed are **statistically indistinguishable** under
the protocol in §3; pick any of them.

**As one rule:** *always* enable **#3** (ResAdd+RMSNorm); on the down-projection axis use
**#8** (atomic down+merge) for **decode** and **#10** (merge+ResAdd2) for **prefill**; never
enable **#6**, never enable **#1** at prefill, and never fuse greedily.

## 2. The magnitude, stated plainly

**The best available combination is worth 0.24 %–0.46 % of layer time.** That is the honest
headline, and it is a subtotal — the measured scope is S3–S11 plus the shared expert;
attention, the MLA projections and the DSA indexer are excluded, so as a fraction of a *full*
layer the gain is smaller still.

The reason is structural and was predicted by the roofline model before any of this was
measured: at T ≥ 32 the layer streams 4–13 GB of expert weights, so the fusible glue kernels
are a percent-level slice of the total. Fusion cannot recover what it does not touch.

## 3. Protocol — why the winners needed a second pass

The first sequential pass produced a "winner" per regime whose margin over the runner-up was
0.2–0.4 %, on a machine this study has already documented throwing 25–320 % one-off
excursions. That margin is not resolvable in one pass.

So each regime was re-measured **twice independently**, 8 interleaved rounds per run: within a
round every candidate is timed once, in a fixed order, so drift affecting a whole round
cancels in the per-config median. A winner is declared only when its gap to the runner-up
exceeds the round-to-round spread of both; otherwise the set is reported as tied.

This mattered. Under a single pass, `decode_bs256` looked like a clean win for `#8` over the
baseline. Under two interleaved runs, one run separates them and the other ties them — so the
honest statement is "≈0.35 %, at the edge of what this machine can resolve", not "1.0037×".

## 4. The crossover — the one large, robust effect

The down-projection axis genuinely flips between regimes, and by a wide margin:

| down-axis choice | bs32 | bs256 | **bs512** | **bs1024** | t2048 | t8192 |
|---|---|---|---|---|---|---|
| **#8** atomic down+merge | **1.0035×** | **1.0037×** | 0.9955× | 0.9749× | 0.9430× | 0.9661× |
| **#10** merge + ResAdd2 | 1.0005× | 1.0010× | **1.0011×** | **1.0018×** | **1.0018×** | **1.0027×** |

`#8` is the best choice at small decode batches and costs **5.7 %** at prefill_t2048; `#10` is
the reverse. A single static fusion plan is therefore wrong for at least one regime — this is
the one place in the study where the choice actually matters, and it is worth branching on.

**Where the crossover actually is (prediction corrected).** T=512 and T=1024 were added
specifically to locate it. The prediction was that `#8` fails once its `[T, 6144]` bf16 atomic
accumulator stops fitting C500's 8 MB L2, i.e. at T ≈ 8·2²⁰/(6144·2) = **683 tokens**. The
measurement says otherwise: `#8` is **already losing at T=512** (0.9955×), where the
accumulator is 6.3 MB and still fits comfortably. So the L2-residency argument gets the
*direction* right and the *threshold* wrong — the real crossover lies between **T=256 and
T=512**, and something other than pure L2 capacity (atomic contention rising with rows per
output line, most likely) drives it. Recorded as a failed prediction rather than quietly
dropped.

Practical rule: use `#8` only for **T ≤ 256**; use `#10` from T=512 upward.

This reproduces the per-fusion result exactly (`#8` measured 1.00–1.01× decode, 0.87–0.90 %
prefill in LOG-05) and explains sglang gating `FUSE_SUM_ALL_REDUCE` behind a flag rather than
enabling it unconditionally.

## 5. Greedy fusion is always wrong

From the first pass, `J_greedy_all` (#1 + #6 + #9 together):

| regime | all-unfused | greedy-all | penalty |
|---|---|---|---|
| decode_bs1 | 0.859 ms | 0.910 | **+6 %** |
| decode_bs32 | 8.460 | 8.593 | +2 % |
| decode_bs256 | 14.005 | 14.365 | +3 % |
| prefill_t2048 | 27.912 | **41.179** | **+48 %** |
| prefill_t8192 | 80.825 | **98.653** | **+22 %** |

**Fusing everything is worse than fusing nothing at every regime**, and catastrophically so at
prefill. Against the best combination the greedy plan is up to **1.48× slower**. If this study
has one operational finding, it is this one — it is two orders of magnitude larger than the
benefit of choosing the *best* plan over the baseline.

## 6. Why the standalone per-fusion numbers do not compose

Two effects, both of which required the end-to-end build to see:

1. **Shared producers.** #4, #5, #11b and #11b′ all fuse the post-attention RMSNorm into the
   router. But that same normalization also feeds the w13 grouped GEMM *and* the shared
   expert, so fusing it into the router does not remove it — you still pay it for the other
   consumers. Their standalone wins (up to 1.24× for #11b′) are worth **zero** in the layer.
   Deleting the norm entirely requires fusing it into *every* K=6144 consumer, which is F11's
   `combined` configuration at **0.478×/0.605×**. The whole prologue-fusion family is
   dominated in context.
2. **Competing for the same operation.** ResAdd1 can be folded into o_proj's epilogue (#1) or
   into the post-attention RMSNorm (#3), never both. #1 regresses at prefill (0.978× measured
   in-layer) and #3 does not, so that axis resolves to #3 everywhere.

## 7. Threats to validity

- **Scope is a subtotal.** Attention core, MLA projections and the DSA indexer are excluded.
  No fusion candidate touches them, so the *ranking* is unaffected, but every percentage here
  is relative to a subtotal, not a full layer.
- **Dispatch-layout construction is outside the timed region**, identical in every
  configuration (so it cannot change the ranking). Our `moe_align_block_size` is a torch
  reference with a Python loop over 256 experts; production uses a fused kernel at ~10–30 µs.
- **Routing is frozen** after being computed from the pipeline's own `h1`, so every
  configuration routes identical tokens to identical experts. The router GEMM and top-k still
  run and are still timed; they write to scratch.
- **Triton, not production kernels.** The dense GEMM reaches ~107 TF/s against the vendor
  BLAS's ~215. Conclusions about *fusion* should port; absolute times should not be read as
  what a production engine would achieve.
- **A bug this nearly shipped.** The first end-to-end run had every non-atomic configuration
  double-applying the routing weights (`launch_down` already sets `MUL_ROUTED_WEIGHT=True`,
  and f10's `merge_only` applies them again). Because *all* of them shared the bug they agreed
  with each other at `rel_err = 0.0`, and the *correct* atomic configurations were the ones
  that looked wrong. Config-vs-config agreement is not a correctness check; every
  configuration is now validated against an independent fp32 reference of the whole subgraph,
  and a configuration that fails is excluded from "best" outright.
