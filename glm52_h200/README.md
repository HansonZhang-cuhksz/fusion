# Kernel fusion in a GLM-5.2 MoE decoder layer — NVIDIA H200 (sm_90) suite

Eleven candidate kernel fusions in **one GLM-5.2 MoE decoder layer**, each built twice — a
fused kernel and an unfused counterpart **cut from the same source** — each arm independently
autotuned, and compared. **The deliverable is the fused/unfused speedup ratio**, per fusion,
per regime, plus a whole-layer combination benchmark.

This is the H200 port of `glm52/`, the audited suite that has already run on a MetaX C500 and
an RTX 4060 Laptop. Structure, fairness protocol and kernel sources are preserved; what
changes is that **every hardware constant now comes from a device probe** and that the Hopper
paths (TMA, thread-block clusters, warp specialisation) are selected **at run time from what
actually compiles and launches on this machine**, never at authoring time.

The H200 has 143 GB, so unlike the 4060 port **nothing is out of scope**: all eleven fusions
and the whole-layer benchmark run at exact GLM-5.2 spec (256 routed experts, `w13` 12.0 GB +
`w2` 6.0 GB = 19.3 GB resident).

---

## 1. Run it

```bash
cd /home/shuhan/fusion

# 1. probe the machine (fast; writes glm52_h200/preflight_h200.json)
python3 glm52_h200/preflight.py

# 2. the whole campaign, serial and resumable. Hours. Use tmux/screen.
python3 run_h200.py
```

`run_h200.py` lives in the **repo root**, not in this directory — it is the single command
the operator runs. It runs the preflight itself if the JSON is missing.

Useful variants:

```bash
python3 run_h200.py --list                       # show the plan, run nothing
python3 run_h200.py --quick                      # short sweeps: stack smoke test only
python3 run_h200.py --families f03,f10           # a subset
python3 run_h200.py --regimes decode_bs1,prefill_t8192
python3 run_h200.py --force-rerun --families f01 # redo one family, ignoring checkpoints
python3 run_h200.py --summary-only               # rebuild summary.json from what exists
python3 run_h200.py --disable-features tma,clusters   # classic path in BOTH arms
```

**Run it in `tmux`/`screen`.** The campaign is multi-hour; if it is interrupted, re-running
the same command resumes at the first family with no result.

**If the box has more than one GPU, pin one:** `python3 run_h200.py --gpu 0`. Every number in
the campaign is only comparable to the others if every family ran on the same device; the
driver warns loudly when several GPUs are visible and none is pinned. The GPU must also be
**exclusive** for the duration — another job on the same device corrupts every timing in both,
which is also why the driver refuses to start when it sees a second bench running.

### What the driver guarantees

| behaviour | why it is there |
|---|---|
| refuses to run on a device that is not sm_90 unless `--force` | the benches size their autotuning grids from the device probe; on another device they produce a correct search over the **wrong** hardware, written into a file labelled `h200` |
| runs the families **serially**, `f03 → f10 → f01 → f04f05 → f11 → f06 → f08f09 → layer` | two benches on one GPU corrupt every timing in both; the order puts the cheapest family first so a broken stack fails in minute one, not hour eight |
| **resumable**, and the resume is device-fenced | a family whose result JSON exists is skipped only if that JSON's `_meta.device` is the GPU in this box; a foreign result is **moved** to `_quarantine_foreign_*/` and re-measured, never silently reused |
| hwinfo before/after every family and at both ends of the run | the 4060 drifted 22 % thermally *within one run* and produced a speedup above its own physical ceiling; the drift record is what makes that detectable |
| one family failing never kills the run | catch, log, continue; the missing ones are listed at the end |
| cells within 3 CUDA-event ticks are printed `UNRESOLVED` | at `decode_bs1` the arms are 9–17 ticks long; a four-digit ratio built from a 3-tick gap is not a measurement |

---

## 2. Send back

Three things, exactly:

1. **`results/h200/`** — every result JSON plus `summary.json`.
2. **`glm52_h200/preflight_h200.json`** — the device probe every number in the results is
   relative to.
3. **`log/run_h200/`** — `driver.log` plus one log per family (`f03.log`, `f01.log`, …).

If a family failed, its log is the only evidence of *why*, so send the logs even when — 
especially when — the run was incomplete. `summary.json` names every family that produced no
trusted result.

---

## 3. What each result file contains

The benches write to `$GLM52_H200_RESULTS_DIR` (default `results/h200/`). The driver sets it.

| file | fusion(s) | contents |
|---|---|---|
| `f01_oproj_resadd.json` | **#1** o_proj + ResAdd | GEMM rows at K=16384 (prefill, non-absorbed MLA) and K=32768 (decode, absorbed) |
| `f03_resadd_rmsnorm.json` | **#3** ResAdd + RMSNorm | pure vector; the cheapest family and the stack's smoke test |
| `f04f05_norm_router.json` | **#4** ResAdd+RMSNorm+Router, **#5** RMSNorm+Router | four variants (`F4`, `F4_topk`, `F5`, `F5_topk`) — with and without the top-k epilogue |
| `f06_upgate_swiglu.json` | **#6** UpGate + SwiGLU | grouped GEMM over 256 experts; needs the 12 GB `w13` |
| `f08f09_down_merge_resadd.json` | **#8** Down+ExpertMerge, **#9** Down+Merge+ResAdd2 | four variants (`f8_atomic`, `f8_token_major`, `f9_atomic`, `f9_token_major`) |
| `f10_merge_resadd.json` | **#10** ExpertMerge + ResAdd | needs the `[T·topk, 6144]` intermediate, not the expert weights |
| `f11_lazy_prenorm.json` | **#11a** LazyPreNorm→w13, **#11b** LazyPreNorm→router | plus `combined`, and an exploratory `half_fused` sub-result |
| `layer_configurations.json` | whole layer | each fusion set A…K timed end to end against the all-unfused baseline |
| `summary.json` | — | written by `run_h200.py`: the cross-fusion table, every cell's resolution verdict, family statuses, and the hwinfo drift record |
| `_ckpt/` | — | per-regime checkpoints, device-fenced; `--force-rerun` bypasses them |
| `_quarantine_foreign_*/` | — | results found in this directory that were produced by a *different* GPU |

Every per-family JSON carries the same skeleton:

- **`rows`** — one row per regime (and per variant), with `fused_ms`, `unfused_ms`, `speedup`,
  the winning config for *each arm*, and the correctness check;
- **`tune_tables`** / `tuning` — the full search for **both** arms: every config tried, its
  time or its compile error, and `n_tried` / `n_failed` per side. This is what makes an
  unfair comparison detectable after the fact;
- **`env`** — the device probe as the bench saw it;
- **`fairness`** — what was held identical between the arms and what was allowed to differ;
- **`_meta`** — timestamp, **device name**, visible devices, torch version.

### Reading a cell in `summary.json`

```jsonc
{
  "fusion": "#3  ResAdd+RMSNorm", "regime": "decode_bs1",
  "fused_ms": 0.0210, "unfused_ms": 0.0230,
  "speedup": 1.098,          // null when the cell is UNRESOLVED
  "speedup_raw": 1.098,      // always present: the operands are never discarded
  "fused_ticks": 82.0, "unfused_ticks": 90.0, "gap_ticks": 8.0,
  "resolved": true, "paired": true, "flags": []
}
```

Markers in the printed table:

| mark | meaning |
|---|---|
| `UNRES` | the two arms differ by fewer than 3 CUDA-event ticks. **No ratio is printed** — the timer cannot resolve it. The raw operands stay in `speedup_raw`. |
| `~` | the shorter arm is under 10 ticks; the ratio is quantised (roughly ±8 % at 12 ticks) |
| `*` | the paired ratio and the ratio-of-medians disagree by more than 2 % — the machine moved during that cell |
| `s` | the arms were **not** interleaved A/B/A/B, so drift does not cancel in this ratio |
| `!` | the speedup exceeds its own **modelled** traffic ceiling — suspect the measurement or the model, in that order |

---

## 4. Measured vs modelled — the explicit list

Nothing in this suite is allowed to look measured when it is not. This is the boundary.

### Measured on the H200

- **Every `fused_ms` / `unfused_ms`.** CUDA events, one L2 flush before each timed repetition
  of the whole chain, final numbers taken **interleaved A/B/A/B in one loop** and reported as
  a **paired** median (`paired: true`).
- **Both arms' full autotuning searches**, including every config that failed to compile and
  why, and the grid size per side.
- **Correctness**, as relative max-abs error against an fp32 reference (`reference.py`), for
  both arms.
- **Device facts** — SMs, warp size, opt-in shared-memory ceiling, registers/SM, L2, VRAM —
  from `torch.cuda.get_device_properties`, cross-checked against Triton. If the two disagree,
  the suite **refuses to autotune** rather than proceeding on a plausible-looking table.
- **Which Hopper features actually work**, by *compiling and launching* a kernel for each:
  TMA (`tl.make_tensor_descriptor`), thread-block clusters (`num_ctas>1`), warp specialisation
  (`tl.range(warp_specialize=True)`), `tl.dot` bf16. Attribute existence is **not** evidence —
  several Triton releases export symbols that fail at compile time.
- **The reachable shared-memory ceiling**, by launching configs and reading
  `CompiledKernel.metadata.shared`. Triton 3.0 staged `num_stages` mainloop buffers; 3.6
  stages `num_stages − 1`. Neither number is hardcoded anywhere.
- **Calibration**: streaming/read-only bandwidth, bf16 GEMM throughput for both Triton and
  cuBLAS at the study's o_proj shape, kernel launch cost, harness floor, and the CUDA-event
  timer tick.
- **Clocks, temperature, power, throttle reasons** at run start, run end, and around every
  family.

### Modelled, not measured

- **Roofline ceilings** (`traffic.py`): `bytes_unfused / bytes_fused` per fusion per regime,
  with weight traffic counted once per distinct expert touched. A measured speedup is only
  interesting relative to this; above it is a red flag, not a triumph.
- **The L2-residency caveat on those ceilings.** One `[T, 6144]` bf16 activation is 12.6 MB at
  T=1024 against the H200's ~50 MB L2, so for the vector fusions at decode the "saved" HBM
  traffic is really L2 traffic and the printed HBM ceiling is **unattainable rather than
  merely unmet**. `traffic.py` computes this per cell (`l2_resident`) instead of asserting it
  per regime. Prefill is where the roofline is unambiguously an HBM story on this device.
- **Expected distinct experts** under uniform routing, `E·(1 − (1 − 1/E)^(T·topk))`. Real
  routing is less uniform, so this is an upper bound on expert-weight traffic.
- **Per-layer savings for a *set* of fusions**, wherever `bench_layer` did not measure that
  exact combination end to end. Such rows are labelled `additive estimate` and their ranks are
  indicative only — the fusions do **not** compose additively: **#1 and #3 compete for
  ResAdd1**, and **#10 and #3 compete for ResAdd2**.
- **SMEM footprint predictions** used to pre-filter autotuning grids. The compiler is the
  authority; a guard only avoids launching configs that will certainly fail, and a guard that
  prunes one arm harder than the other is a fairness bug (this is exactly what the C500
  constants did on other devices).

### Not measured at all — deliberately out of scope

- **Attention core, MLA projections, the DSA indexer.** No fusion candidate touches them; they
  are excluded from the timed region in every family, on **both** arms.
- **MoE dispatch-layout construction** — excluded from the timed region, identically for both
  arms, in every family.
- **FP8.** bf16 activations with fp32 accumulation only, this round.
- **End-to-end model latency.** The scope is one decoder layer (S3–S11 plus the shared
  expert); a 78-layer figure would be an extrapolation, not a measurement.

---

## 5. Target configuration

Verbatim from `zai-org/GLM-5.2` `config.json` (`glm_moe_dsa`):

| | |
|---|---|
| hidden | 6144 |
| moe_intermediate | 2048 (`w13` = `[4096, 6144]` per expert, `w2` = `[6144, 2048]`) |
| experts | **256 routed + 1 shared**, top-**8** |
| routing | sigmoid scoring, `noaux_tc` grouped top-k, `norm_topk_prob`, `routed_scaling_factor` **2.5** |
| dtype | bf16 activations, **fp32** accumulation, fp32 router math |
| layers | 78, first 3 dense |
| o_proj K | **16384** prefill (non-absorbed MLA, 64 heads × 256) / **32768** decode (absorbed, 64 × 512) |

**Regimes:** `decode_bs1`, `decode_bs32`, `decode_bs256`, `decode_bs512`, `decode_bs1024`,
`prefill_t2048`, `prefill_t8192`.

---

## 6. Why the numbers mean something

Four rules, enforced by `common.py`, each of which exists because breaking it produced a
plausible, publishable, **wrong** table earlier in this study:

1. **Chains are timed as chains.** An unfused variant is a *sequence* of kernels, timed as one
   unit with a single L2 flush **before** the sequence, never between its kernels. Flushing
   between them fabricates a fusion win, because in real execution the producer's output is
   still in L2 when the consumer starts.
2. **Both arms tuned independently**, from the same source with a `tl.constexpr` flag toggled.
   Only the mapping — tile sizes, `num_warps`, `num_stages`, loop and grid order — may differ.
   Comparing a tuned kernel against an untuned one is the easiest way to manufacture a result.
3. **Final timings interleave A/B/A/B** inside one loop and report the **paired** ratio, so
   monotone clock/thermal drift cancels. On the 4060, timing the whole fused arm then the whole
   unfused arm produced a speedup above that cell's physical ceiling.
4. **No hardware constant is ever hardcoded.** One cached device probe feeds every guard. The
   probe **crashes** rather than falling back to defaults — a wrong-but-plausible constant
   table silently prunes autotuning grids, and it does not prune both arms equally, which
   manufactures or destroys a fusion win.

The L2-flush buffer is sized from the device (≥ 4× L2, ≥ 256 MB). The H200's ~50 MB L2 is
6× the C500's, so the buffer that was safe there would have turned every "cold" measurement
here into a warm-cache one.

---

## 7. Hopper paths and their fallbacks

Every H200-specific path has a working classic fallback, and **the choice is made at run
time** from the preflight's compile-and-launch probes — never from a version check and never
at authoring time. If TMA or clusters do not work on this stack, the suite runs the classic
path and **records that it did**; it does not fail and it does not silently produce a
different measurement under the same label.

The important fairness property: a fallback applies to **both arms of a pair**. A Hopper path
that is available to the fused kernel but not to its unfused counterpart would be a fusion win
that is really a codegen win.

Escape hatches, for when something misbehaves on a device we cannot debug:

| env var (or driver flag) | effect |
|---|---|
| `GLM52_H200_DISABLE_FEATURES=tma,clusters,ws`  (`--disable-features`) | force the classic path in both arms |
| `GLM52_H200_FLUSH_MB=N`  (`--flush-mb`) | override the L2-flush size (diagnosis only — the default is derived from L2, and a too-small flush measures a warm cache) |
| `GLM52_H200_RESULTS_DIR` (`--results-dir`) | where results are written; keeps one platform's run from overwriting another's |
| `GLM52_H200_PREFLIGHT` | path to the probe JSON |
| `GLM52_H200_FORCE=1` (`--force-rerun`) | ignore existing checkpoints and re-measure |

Anything set this way is recorded in `summary.json` under `families[].env_overrides` and
surfaced in the run's `warnings`, so a degraded run can never be mistaken for a clean one.

---

## 8. Layout

```
run_h200.py                 <- repo root: the one command the operator runs
glm52_h200/
  preflight.py              stack / device / feature / calibration probe  -> preflight_h200.json
  config.py                 GLM-5.2 constants (verbatim from the HF config) + the cached device probe
  common.py                 harness: chain timing, paired A/B timing, autotune, checkpoints, record
  hwinfo.py                 clocks / power / throttle snapshots and start-vs-end comparison
  reference.py              fp32 ground truth + sglang-compatible MoE dispatch layout
  traffic.py                analytical roofline ceiling per fusion per regime  (MODELLED)
  kernels/                  the Triton kernels, one module per fusion family, plus hopper.py
  bench/                    one runnable driver per family
results/h200/               raw JSON: every timing, every tuning table, every check
log/run_h200/               driver.log + one log per family
```

## 9. Provenance

- `glm52/` — the audited C500 + RTX 4060 suite this is ported from.
- `log/LOG-00` plan and the analysis that filtered **#2** and **#7**; `LOG-01..07` per fusion;
  `LOG-08` the 76-finding fairness audit; `LOG-09` consolidated results; `LOG-11` optimal set
  per regime; `LOG-13` the RTX 4060 port, whose §9 is where the interleaving, timer-tick and
  device-fence rules above were paid for.
- `report_glm52_c500/`, `report_glm52_rtx4060/` — the two published predecessors. Their
  headline: only the memory-bound vector fusions were worth doing, and fusing work *into* a
  GEMM cost 15–47 % of its throughput on C500 while costing ~0 on Ada. Which of those two the
  H200 resembles is the question this run answers.
