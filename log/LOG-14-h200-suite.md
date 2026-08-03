# LOG-14 — Building the H200 (sm_90) benchmark suite

**Date** 2026-08-03 · **Status** IN PROGRESS — preflight shipped, suite under construction
**Target** NVIDIA H200, sm_90 (Hopper), ~143 GB · **Constraint** the H200 environment is
confidential; **nobody on this side can test on it.** The user runs the scripts and returns
results. Every round trip is expensive, so the code must be defensive rather than clever.

---

## 1. Why this port is different from the previous two

| | C500 | RTX 4060 | **H200** |
|---|---|---|---|
| whole GLM-5.2 layer fits? | yes (64 GB) | **no** (8 GB vs 18 GB of expert weights) | **yes** (143 GB) |
| warp specialization | no | no | **YES** |
| TMA | no | no | **YES** |
| thread-block clusters / DSMEM | no | no | **YES** |
| fusions measurable | 9 | 6 | **all 11** |
| whole-layer benchmark | yes | impossible | **yes** |

Two consequences make this the most scientifically interesting device in the study:

**(a) The full protocol is reproducible again.** All 11 fusions *and* `bench_layer`'s
combination sweep, which the 4060 could not run at all.

**(b) The paper's precondition finally holds.** #11's technique ("Towards Free Normalization",
Zhou et al. 2026) relies on **warp specialization** to park the sum-of-squares reduction on
dedicated warps so it overlaps the MMA pipeline. No device in this study has had it. On C500
and the 4060 the reduction *displaced* MMA issue slots instead of hiding behind them — measured
on C500 as **+0.78 % arithmetic for −12.3 % throughput**. LOG-09 §3 attributed the failure to
exactly this missing hardware lane. **H200 is the first device that can test that attribution
directly**, so the suite implements a warp-specialized variant *and keeps the non-specialized
one as a control*, timing both.

## 2. Decisions taken with the user

| question | decision |
|---|---|
| stack discovery | **probe-first** — ship `preflight.py`, user returns its JSON, kernels then targeted precisely |
| scope | **all 11 fusions + whole-layer** |
| precision | **bf16 only** — keeps every number comparable to C500/4060; FP8 would be a separate study |
| budget | several hours, **no package installs** → resumable, and any unavailable feature is skipped rather than required |

## 3. `glm52_h200/preflight.py` — shipped, and validated against a device that *lacks* the features

The single artifact that unblocks everything. Self-contained (imports nothing from the
project), installs nothing, mutates no device state, and is written so that **no probe can
crash the run** — a failure is recorded, because *"this raised `TypeError: unexpected keyword
'warp_specialize'`"* is precisely the information needed to write the kernels correctly.

It reports:

1. **Stack** — torch / triton / CUDA / cuDNN / driver, device count.
2. **Device** — every torch property incl. `shared_memory_per_block_optin`, `regs_per_multiprocessor`, L2, free/total VRAM, compute capability.
3. **`nvidia-smi`** — SM and memory clocks *current and max*, throttle reasons, power draw/limit, temperature, **ECC / MIG / persistence / compute mode**, PCIe gen+width, and `topo -m`.
4. **Triton feature probes** — and this is the part that matters: each candidate feature is
   **compiled and launched**, not attribute-sniffed. Several Triton releases export symbols
   that fail at compile time, so `hasattr` is not evidence. Probes cover TMA
   (`tl.make_tensor_descriptor` + host `TensorDescriptor` + `triton.set_allocator`), warp
   specialization (both the `tl.range(warp_specialize=True)` source form and the
   `num_consumer_groups` launch-kwarg form), thread-block clusters (`num_ctas>1`), and
   `tl.dot` bf16. It also dumps `tl.range`'s signature and Triton's own device properties.
5. **Calibration** — achievable bandwidth (copy / read-modify-write / read-only, buffers sized
   well past L2), bf16 GEMM ceiling for **both cuBLAS and Triton** at the study's o_proj shape,
   the reachable shared-memory ceiling (by trial-compiling progressively larger tiles and
   reading `metadata.shared`), kernel launch cost, and the CUDA-event timer tick.
6. **Capacity check** — whether the 256-expert `w13` (12.0 GB) + `w2` (6.0 GB) fit, i.e.
   whether #6/#8/#9/#11a and the whole-layer benchmark can run at all.

### 3.1 Validated on hardware that fails the features on purpose

Run on the local RTX 4060 (sm_89), which has **no** TMA and **no** clusters. It completed with
zero unhandled errors and correctly reported:

```
OK    baseline_elementwise
OK    tl_dot_bf16
OK    warp_specialize_tl_range
FAIL  warp_specialize_num_consumer_groups: KeyError: 'Keyword argument num_consumer_groups ... unrecognised'
FAIL  tma_tensor_descriptor: CompilationError
FAIL  thread_block_cluster_num_ctas: ValueError: num_ctas > 1 requires NVIDIA SM90+ (Hopper). Current target is sm_89.
```

plus a correct capacity refusal (`expert weights fit: False`). That is the exact failure mode
that cannot be rehearsed on the H200, so rehearsing it on a device that *lacks* the features
is the closest available test. Sample output kept as
`glm52_h200/_example_preflight_sm89.json` — **clearly named so it cannot be mistaken for H200
data**, which is the same hazard that nearly published C500 checkpoints as 4060 results.

### 3.2 One bug found and fixed during that validation

The timer-tick detector scanned candidate quanta upward and kept the first that matched, so it
always reported the *finest* candidate — 0.256 µs trivially divides everything 1.024 µs
divides. It now takes the **largest** quantum matching ≥98 % of samples and records the full
candidate table. (On the 4060 the true tick is 1.024 µs; the buggy version reported 0.256 µs.)

## 4. Lessons from C500 and the 4060 that are being encoded into the suite

Each of these cost a real failure, and each is a build requirement rather than a suggestion:

1. **No hardcoded hardware constants.** C500 literals (`warp 64`, `104`, `65536`, `131072`)
   baked into autotuning guards silently pruned grids on other devices — and *not equally for
   both arms of a pair*, which manufactures or destroys a fusion win.
2. **A degraded probe must crash, not fall back.** The 4060 port nearly shipped a probe that
   returned C500 defaults when Triton's property query raised, which would have built
   C500-shaped grids inside a correctly-labelled result file.
3. **Interleave the two arms when timing.** `bench_f01` on the 4060 timed the whole fused arm
   then the whole unfused arm; the GPU drifted **22 % thermally within one run** and produced a
   speedup *above the cell's physical ceiling*. The suite gets a new `bench_pair()` that
   alternates A/B/A/B and reports a paired statistic.
4. **Device-fence every checkpoint.** A stale C500 checkpoint was one call away from being
   republished as a 4060 measurement.
5. **Size the L2 flush from the device.** H200's L2 is ~50 MB; a buffer sized for 8 MB turns
   every measurement warm and flatters whichever arm re-reads an intermediate.
6. **Triton's SMEM model is version-dependent** — 3.0 stages `num_stages` buffers, 3.6 stages
   `num_stages − 1` (measured across 68 configs). Prefer trial compilation over any model.
7. **Record `n_tried`/`n_failed` per arm** — this is what makes an unfair comparison detectable
   after the fact.
8. **Record the timer tick** and flag any ratio whose operands are within a few ticks.

## 5. Suite layout being built

```
glm52_h200/
  preflight.py        SHIPPED - run first, returns preflight_h200.json
  config.py           GLM-5.2 constants, regimes, hardened device probe, Hopper caps
  hwinfo.py           clocks/power/ECC/MIG/persistence/PCIe/topology, embedded in every result
  common.py           bench_chain, NEW bench_pair (interleaved), autotune, device-fenced ckpt
  reference.py        fp32 ground truth (+ an exact_fp32 variant, see REPORT-lazy-prenorm A6)
  traffic.py          latency-aware roofline, peaks from preflight calibration
  kernels/hopper.py   TMA / warp-spec / cluster abstraction, runtime-detected with fallbacks
  kernels/*.py        the 7 fusion kernel families, one source per family + constexpr flags
  bench/*.py          one driver per family + bench_layer.py (combination sweep, 2 passes)
run_h200.py           the single script the user runs; serial, resumable, never fatal
```

## 6. Build status — complete, and validated as far as sm_89 allows

Seven modules were built in parallel (~13 000 lines). Three agents hit a session limit before
reporting, but had already written their files; **the adversarial review agent was one of
them, so I did that review myself** — which turned out to matter.

### 6.1 Three real bugs, each of which would have crashed the H200 run

Static review (imports, py_compile, constant grep) passed everything. The bugs only appeared
when I **actually ran the benches on the local sm_89 box**, which is the strongest test
available without the H200 — every Hopper path falls back there, so it exercises the fallback
code the H200 will not.

| # | defect | consequence |
|---|---|---|
| 1 | `bench/__init__.py` aliased `("paired_speedup", "speedup", "paired_p50")` onto `paired_speedup_p50`, but `common.bench_pair` returns a `PairTiming` whose headline field is **`ratio_p50`**. `.get()` swallowed it to `None`. | `TypeError: unsupported format string passed to NoneType.__format__` — **crash on every regime of every bench** |
| 2 | harness called `common.ckpt_save(result_id, key, env, payload)`; the real signature is `(name, regime, payload)`. | `TypeError` per regime → silently fell back to a local writer, **losing the device fence** that stops another GPU's checkpoint being republished |
| 3 | `common._cfg_key()` and `neighbours()` assumed every grid element is a dict; `bench_f01`'s joint stage tunes **`(gemm_cfg, epi_cfg)` tuples**. | `AttributeError: 'tuple' object has no attribute 'items'` — **f01 lost entirely** |

All three are the same class: **agents building against each other's interfaces without being
able to run them.** Fixes: harvest every scalar field from the returned dataclass rather than a
fixed name list, plus a comprehensive alias map; bind checkpoint arguments by
`inspect.signature` instead of assuming an arity; and make `_cfg_key` recurse over containers
while `neighbours` returns `[]` for composite grids (so `n_refine == 0` records honestly that
only the coarse stage ran, instead of aborting the regime).

### 6.2 What now runs end-to-end on the local box

| bench | result on sm_89 |
|---|---|
| `f03` | ✅ paired **1.333×** at decode_bs1 — matches my independent 4060 measurement of the same cell exactly |
| `f10` | ✅ paired **1.417×** (208 % of its bandwidth ceiling — the launch-elimination signature) |
| `f01` | ✅ paired **1.0117×**, and it reports the median-of-medians (0.9988×) alongside — the interleaved protocol exposing exactly the discrepancy the 4060 taught us to look for |
| `f04f05` | ✅ all four variants |
| `f11 --router-only` | ✅ including the **isolation measurement** (`router +9.26 %`, same config, `FUSE_NORM` on/off) — the number that carries the scientific result |
| `f08f09` | ✅ runs at bs1 (only 8 experts touched, so the w2 subset fits 8 GB) |
| `f06` | ✅ **refuses correctly**: *"w13 needs 12.0 GB and only 6.9 GB is free … check MIG mode and other tenants; reducing the expert count would change the grouped-GEMM tiling and is not a valid substitute."* |
| `run_h200.py` | ✅ `--list`, `--dry-run`, and a **fatal refusal on non-sm_90** with an explicit `--force` escape hatch |

That last refusal is deliberate: the benches size their grids from the device probe, so on the
wrong device they would produce a correct search over the wrong hardware and write it into a
file named `h200`. Same failure family as the C500 checkpoints.

## 7. Outstanding

- Build workflow running (7 modules in parallel, then an adversarial review whose brief is
  "you will be blamed if the user burns a multi-hour H200 run on broken code").
- Kernels are being written **feature-detected with classic fallbacks**, so they run whatever
  the probe reports; once the real `preflight_h200.json` comes back the Hopper paths get
  specialised to the exact APIs present.
- Nothing here has executed on sm_90. Every H200-specific claim in this log is a design intent,
  not a measurement.
