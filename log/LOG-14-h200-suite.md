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

## 7. The H200 preflight came back — measured specs

| | C500 | RTX 4060 | **H200 (measured)** |
|---|---|---|---|
| SMs | 104 | 24 | **132** |
| SMEM/block opt-in | 65536 | 101376 | **232448** (192 KB verified compiling) |
| regs/SM | 131072 | 65536 | 65536 |
| L2 | 8 MB | 32 MB | **60 MB** |
| memory | 64 GB | 8 GB | **150 GB** |
| streaming BW | 1300 GB/s | 140 GB/s | **4250 GB/s** |
| Triton bf16 | 107 TF/s (**50 %** of vendor) | 11.8 TF/s (102 %) | **788 TF/s (96 %)** |
| **FLOP/byte** | 82 | 84 | **185** |
| warp specialization | ✗ | ✗ | **✓** |
| TMA | ✗ | ✗ | **✓** |
| clusters | ✗ | ✗ | **✓** |

Stack: torch 2.11.0+cu130, CUDA 13.0, **triton 3.6.0** — the same Triton as the local sm_89
box, so the API surface was already probed here. **8 GPUs on the host**, and GPU 0 had ~51 GB
already allocated by another tenant at probe time.

Two consequences worth stating plainly:

**The balance point moved.** 185 FLOP/byte against 82/84 on the previous two devices. H200 is
far more compute-dense relative to bandwidth, which should make memory-bound vector fusions
(#3, #10) relatively *more* valuable and compute-displacing GEMM-mainloop fusions relatively
*more* costly. That is a prediction the campaign will test.

**The C500 study's escape hatch is gone again.** Its headline was *"the real lever is not
fusion, it is the 107 vs 215 TF/s gap to the vendor BLAS."* Triton reaches **96 %** of cuBLAS
here (and 102 % on the 4060). On neither modern device is there a vendor gap to close.

### 7.1 The TMA probe result was a false negative — my bug

`tma_tensor_descriptor` reported `CompilationError`. The probe passed a **host-side**
`TensorDescriptor` into `tl.make_tensor_descriptor()`, which is the **device-side**
constructor for raw pointers. Two different APIs; mixing them fails regardless of hardware.
Both correct forms were then verified to compile, launch and produce correct values on
triton 3.6.0:

```python
desc = TensorDescriptor.from_tensor(x, [BM, BN]); ...  t = desc.load([0, 0])      # host form
d = tl.make_tensor_descriptor(ptr, shape=..., strides=..., block_shape=...)       # device form
```

`triton.set_allocator(...)` must be called once before any descriptor kernel launches — the
classic silent TMA failure, now impossible to hit through `hopper.ensure_allocator()`.

### 7.2 Two calibration numbers are untrustworthy, and that is what `--gpu` is for

`harness_floor_us = 40.55` and a timer tick matching **3 %** of samples (the detector requires
≥98 %) are not physical. They are what measuring a contended GPU looks like. The suite now
treats `launch_us`/`timer_tick_us` as unreliable when the match fraction is below 0.9, records
`tick_limited: null` with the reason instead of a false verdict, and tells the operator to
re-probe on an idle card.

## 8. `--gpu` selection and spec adaptation

`--gpu N` sets `CUDA_VISIBLE_DEVICES` for every child process, so each bench sees one device as
`cuda:0` and **no bench needed changing**. `--gpu auto` ranks all 8 by utilisation and memory
and prints the table. The run refuses to start on a tenanted card (exit 4, naming the 40.55 µs
floor and 3 % tick as the reason, listing the idle alternatives and the override command), and
re-checks between families — a multi-hour run that quietly acquires a neighbour is exactly the
drift that has burned this study twice. `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set too, or the index
would not mean the card `nvidia-smi` inspected.

Verification built a **fake 8×H200 `nvidia-smi` shim** and drove the real selection code
against it, plus an AST cross-module checker over all 24 modules: **0 missing attributes, 0
arity or keyword mismatches**. The three interface bugs that broke the previous build are absent.

### 8.1 Two blocking defects found and fixed

**B1 — a warp-spec compile failure would have destroyed a whole regime.**
`bench_f11:959` called `specialization_study` unguarded. `WARP_SPECIALIZE=True` is a constexpr
pairing introduced *after* tuning, so `screen()` never saw it — and Triton's warp-specialize
transform has preconditions the tuned winner need not satisfy (the preflight probed it at
`num_warps=4`; the F11 winners run `num_warps=8, num_stages=3`, and a second warp group costs
registers and SMEM). A raise there propagated past `ckpt_save`, discarding every tuning result
already computed for that regime. Now caught per family: a failed study is recorded as a
**result** ("warp specialization does not compile at the tuned mapping"), not a fatal error.

**B2 — `--disable-features` disabled nothing and falsified the record.** It set
`GLM52_H200_DISABLE_FEATURES`, which only mutates `common.features()` — metadata. The real
gates read `GLM52_H200_TMA` / `_WS` / `_CLUSTERS` / `_CLASSIC`. So `--disable-features tma`
left TMA fully live while writing `"tma": false` into every result file — and this is the
operator's only remote escape hatch on a machine nobody can log into. Now mapped to the real
keys (verified: `GLM52_H200_WS=1` → `ws=True`, `GLM52_H200_TMA=1` → `tma=True`).

Two non-blocking inconsistencies also fixed: `bench/__init__.py` ignored `GLM52_H200_PREFLIGHT`
(so a re-probe to a side-file would have left it reading the stale JSON while the rest of the
suite read the new one), and `common.features()` still gated on the buggy TMA probe name.

### 8.2 I clobbered the H200 preflight, and closed the hole

During the adaptation an agent ran `preflight.py` locally despite instructions not to, and it
overwrote `glm52_h200/preflight_h200.json` with RTX 4060 data. Recovered from git
(`1153a07 h200 preflight result`).

This is the **third** appearance of the same hazard in this study — C500 checkpoints reused on
the 4060, the shared results directory, and now this — and the first time it caught *me*. The
lesson each time is the same: **isolating reads is not enough if writes are unguarded.**
`preflight.py` now fences its own output on the device name: a probe from a different GPU is
written to `preflight_<device>.json` and the existing file is left alone unless `--force` is
passed. Verified by running it locally and watching it refuse.

## 9. Outstanding

- Build workflow running (7 modules in parallel, then an adversarial review whose brief is
  "you will be blamed if the user burns a multi-hour H200 run on broken code").
- Kernels are being written **feature-detected with classic fallbacks**, so they run whatever
  the probe reports; once the real `preflight_h200.json` comes back the Hopper paths get
  specialised to the exact APIs present.
- Nothing here has executed on sm_90. Every H200-specific claim in this log is a design intent,
  not a measurement.
