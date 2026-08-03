{
  "summary": "Audit the GLM-5.2 fusion benchmark suite for correct+fair porting from MACA C500 to RTX 4060 (sm89)",
  "agentCount": 7,
  "logs": [
    "audit complete: 76 findings across 6 modules -- {"blocker":10,"fairness":24,"correctness":12,"perf":11,"cosmetic":19}"
  ],
  "result": {
    "finding_count": 76,
    "by_severity": {
      "blocker": 10,
      "fairness": 24,
      "correctness": 12,
      "perf": 11,
      "cosmetic": 19
    },
    "plan": "I've verified every blocker and fairness claim against the actual files. Here is the consolidated plan.

---

# RTX 4060 Port Plan — verified, deduplicated, ordered

## 0. What I dropped, and why

| Dropped claim | Why |
|---|---|
| `common.py:37` `_FLUSH_BYTES = 128*2**20` hardcoded with stale "C500 L2 is 8 MB" comment | **Already fixed.** `common.py` was edited at 14:34 (rest of repo 14:12; `git status` shows `M glm52/common.py`). Lines 41–46 now read `_FLUSH_BYTES = max(128*2**20, 4*torch.cuda.get_device_properties(0).L2_cache_size)` with a comment naming the 4060's 32 MB L2. Two auditors read the pre-edit file. **The L2 flush is correct and unbiased: 128 MiB = exactly 4× the 4060's 32 MiB L2.** |
| Follow-on advice to shrink `_FLUSH_BYTES` to 64 MiB | Retained only as an optional wall-clock lever (below), not a correctness fix. |
| `results/_f04f05_norm_router_ckpt/` is a stale-reuse hazard | **False.** `bench_f04f05_norm_router.py:873` only *writes* checkpoints. There is no read/reuse path anywhere in that file. Harmless. Only **f01** and **f10** reuse. |
| "F1 prefill can flip compute→memory once `C_PEAK` is honest" | **Not supported.** `results/rtx4060_gemm_ceiling.json` (measured, on disk) gives Triton bf16 = **11.81 TF/s**; `device_4060_calibration.json` gives stream r/w = **140 GB/s**. Balance point = 84.3 flop/byte vs C500's 82.3 — a 2.5% move. `compute_bound` labels survive. The constants must still be fixed (absolute ms are 10× wrong), but the narrative-flip risk is not real. |
| Auditors' proposed constants `B_PEAK=128.8e9`, `C_PEAK=11.11e12` | **Superseded by on-disk measurements.** Use 140e9 / 11.81e12 (see A3). The brief's 128.8/11.11 do not match the calibration files this repo already contains. |

Everything else below I confirmed by reading the file at the cited line.

---

## PART A — Measurement bias (fix before any kernel runs)

These change *which configs get benchmarked*. They are the reason to do this work at all.

### A1. SMEM ceiling hardcoded to 65536 — deletes 20–38% of the legal tile grid
**Confirmed at 5 in-scope sites.** Triton 3.6's sm89 ceiling is `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` = **101376 B**, and the launcher opts in automatically (`triton/backends/nvidia/driver.c:144–156`: `if (shared > 49152 && shared_optin > 49152) cuFuncSetAttribute(..., MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_optin - shared_static)`). `compiler/compiler.py:454–456` only raises `OutOfResources` above `max_shared_mem`. So the 65536..101376 band is silently deleted with **no hardware justification**.

I re-derived the loss over `bench_layer._ALTS`' 48-tile cross product:

| num_stages | admitted @65536 | legal @101376 | lost |
|---|---|---|---|
| 2 | 34/48 | 43/48 | 9 |
| **3 (the `SEED_GEMM` default)** | **27/48** | **34/48** | **7** |
| 4 | 20/48 | 32/48 | 12 |

Tiles lost at s=3: `(32,64,128) (64,32,128) (64,64,128) (64,128,64) (128,64,64) (128,128,64) (128,256,32)` — including `128×128×64`, the canonical Ada bf16 GEMM tile.

Add a shared probe to `glm52/config.py` (after the `BenchEnv` class):

```python
_ENV_CACHE: "BenchEnv | None" = None

def env() -> "BenchEnv":
    """Probe once, reuse everywhere. Every hardware constant in the benches comes from here."""
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = BenchEnv.probe()
    return _ENV_CACHE
```

Then, per site:

| File | Line | Replace |
|---|---|---|
| `glm52/bench/bench_layer.py` | 80–81 | `def _smem_ok(bm, bn, bk, s, mult=1, limit=None):`<br>`    return s * 2 * bk * (bm + mult * bn) <= (limit or C.env().smem_bytes)` |
| `glm52/bench/bench_f01_oproj_resadd.py` | 54 | `SMEM_LIMIT = C.env().smem_bytes  # per-block opt-in; 101376 sm89 / 65536 C500` |
| `glm52/bench/bench_f04f05_norm_router.py` | 72 | `SMEM = C.env().smem_bytes` |
| `glm52/bench/bench_f11_lazy_prenorm.py` | 60 | `SMEM_LIMIT = C.env().smem_bytes` |
| `glm52/config.py` | 99, 119 | `smem_bytes: int = 0` / `smem_bytes=props.get("max_shared_mem", 101376)` |

**Asymmetry warning (f04f05):** all 12 configs the 65536 cap drops from `gemm_grid()` have `BLOCK_E=256`. The unfused router GEMM loses 12/91 (13%); the `FUSE_TOPK` fused arm — structurally pinned to `BLOCK_E=256` (`kernels/norm_router.py:286` asserts it) — loses 24/70 (34%). Every C500 unfused winner used `BLOCK_K=64`, and `BK=64 × BE=256` is exactly the deleted family.

**Asymmetry warning (f11b):** with the 65536 cap the router grid contains **zero** configs with `BLOCK_N=256 AND BLOCK_K>=64`. `BLOCK_N=256` is the fused arm's structurally ideal shape (`sq_redundancy = cdiv(256,256) = 1`, the only width where sum-of-squares isn't recomputed per n-tile). The cap deletes the fused kernel's best shape family while the unfused GEMM, indifferent to `BLOCK_N`, keeps alternatives.

### A2. Warp size 64 hardcoded — every per-thread guard off by exactly 2×
**Confirmed at 12 in-scope sites.** sm89 warps are 32 lanes (`driver.c:93` exports `warpSize`; `device_4060_calibration.json` records 32). Every `threads = num_warps * 64` overestimates thread count 2×, therefore *underestimates* per-thread register footprint 2× — on a register file that is **half** C500's (65536/SM vs 131072/SM). Simultaneously too permissive at the high end and too restrictive at the low end.

Add once per file: `WARP = C.env().warp_size`, then:

| File | Line | Current | Replace |
|---|---|---|---|
| `bench_layer.py` | 105 | `threads = c.get("num_warps",4) * 64` | `threads = c.get("num_warps", 4) * C.env().warp_size` |
| `bench_f01_oproj_resadd.py` | 74 | `threads = w * 64` | `threads = w * WARP` |
| `bench_f01_oproj_resadd.py` | 202 | `per_thread = blk / (w * 64)` | `per_thread = blk / (w * WARP)` |
| `bench_f03_resadd_rmsnorm.py` | 52 | `threads = w * 64` | `threads = w * WARP` |
| `bench_f04f05_norm_router.py` | 89 | `threads = w * 64` | `threads = w * WARP` |
| `bench_f04f05_norm_router.py` | 152 | `epr = bm * bk / (w * 64)` | `epr = bm * bk / (w * WARP)` |
| `bench_f10_merge_resadd.py` | 68 | `th = w * 64  # warp = 64 lanes on C500` | `th = w * WARP` |
| `bench_f11_lazy_prenorm.py` | 86 | `... / (cfg["num_warps"] * 64)` | `... / (cfg["num_warps"] * WARP)` |
| `bench_f11_lazy_prenorm.py` | 173 | `2 <= b*r/(w*64) <= 64` | `2 <= b * r / (w * WARP) <= 64` |
| `bench_f11_lazy_prenorm.py` | 186 | `threads = w * 64` | `threads = w * WARP` |
| `bench_f11b_sweep.py` | 85–86 | `... * 4 / (c.get("num_warps",4) * 64) <= 2048` | see A4 |
| `analyze_f11b_arch.py` | 80–83, 86 | `* 64` / `// 64` | `* 32` / `// 32` — **must change together with line 20 (A8) or the compensating `by_reg` error breaks** |

**One-sided sites (these bias the ratio directly):**
- `bench_f01:202` `epi_grid()` — the epilogue kernel is the *entire* unfused-side overhead; the fused arm with `SPLIT_K==1` has no epilogue at all. Pruning `(256,w8) (256,w16) (512,w16)` under-tunes only the unfused arm → **inflates the fusion win at decode**.
- `bench_f04f05:152` `_norm_ok` — `norm_grid()` feeds only `t_norm` (line 384) and `t_addn` (386), both unfused. Wrongly rejects `{BM1,BK512,w8} {BM2,BK512,w16} {BM1,BK1024,w16}` → **inflates the fusion win**.
- `bench_f11:186` `norm_grid` / `:173` `rstd_grid` — the unfused arm's bonus RMSNorm search. 152→130 correct configs; 28 wrongly admitted at a true 128 fp32/thread (guaranteed spill). **A spilling unfused norm kernel manufactures a fused win at every regime.**
- `bench_f10:68` — rejects 10 shapes whose true per-lane work is 4 elements, including `(256,1,w2)`, the exact 32-lane analogue of C500's prefill_t8192 winner `BLOCK_N=256/ROWS=1/w1`. Meanwhile admits 14 shapes at a true 64 elem/lane, where the **fused** kernel (one extra live residual tile) hits the register cliff first.

Also extend the warp ladders so CTA sizes still span 32..1024 threads as they did on C500:
- `bench_f03_resadd_rmsnorm.py:70` → `(1, 2, 4, 8, 16, 32)` **and line 86** `warps = [1, 2, 4, 8, 16, 32]`. Both together — if 32 is added to coarse only, `warps.index(best["num_warps"])` at line 90 raises `ValueError` and kills the refine stage.
- `bench_f10_merge_resadd.py:58` → `WARPS = [1, 2, 4, 8, 16, 32]`.

### A3. Roofline constants are C500 — every printed ceiling is wrong
`glm52/traffic.py:55–57`. Consumed by `bench_f04f05:706`, `bench_f11:651/666`, and every `roofline_ceiling` / `unfused_ms` / `fused_ms` / `compute_bound` field. **Use the measurements already on disk**, not the brief's numbers:

```python
# Measured on RTX 4060 Laptop @ locked 1020 MHz SM / 5501 MHz MEM, 2026-07-31.
# See results/device_4060_calibration.json and results/rtx4060_gemm_ceiling.json.
#   compute: 11.81 TF/s -- best *Triton* bf16 GEMM, cfg 64/256/32/w8/sk4/st3, M4096 K16384 N6144.
#            (cuBLAS measures 11.62 TF/s; Triton is the honest denominator for these kernels.)
#   bandwidth: 140 GB/s mixed read+write stream; 159 GB/s read-only.
# Balance point 84.3 flop/byte, vs C500's 82.3 -- essentially unchanged, so compute_bound
# labels carry over; only the absolute predictions move (~10x).
C_PEAK = 11.81e12
B_PEAK = 140.0e9
B_PEAK_READ_ONLY = 159.0e9
```
Delete the C500 calibration block at lines 36–54 (do not keep it as "reference" — `_time()` reads these unconditionally).

### A4. `bench_f11b_sweep.py` — the unfused arm cannot reach a usable norm config
**Confirmed.** Line 88 tunes the unfused RMSNorm with `_coord_tune(..., dict(SEED_NORM), ...)`, which sweeps only keys in the seed using `bench_layer._ALTS`. `_ALTS["BLOCK_N"] = (32, 64, 128, 256)` — GEMM tile widths. `SEED_NORM` (line 48) is `BLOCK_N=8192`. So the reachable space is `{8192, 32, 64, 128, 256}`; **1024/2048/4096 are unreachable.**

`BLOCK_N=8192` takes the ONE_SHOT path (`add_rmsnorm.py:223`, `BLOCK_N >= 6144`): at `ROWS=1/w=4` that is `8192/(4*32) = 64` fp32/thread on the 4060 vs 32 on C500 — a 2× register-pressure jump on half the register file. The multi-pass configs that fix it are exactly the unreachable ones. **This inflates the fused arm's win at every T in the crossover sweep — the module's headline deliverable.**

Replace `bench_f11b_sweep.py:84–89`:
```python
from glm52.common import autotune
from glm52.bench.bench_f11_lazy_prenorm import norm_grid

WARP = C.env().warp_size
guard = lambda c: (_smem_ok(c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"], c["num_stages"])
                   and 2 <= c["BLOCK_M"] * c["BLOCK_N"] / (c["num_warps"] * WARP) <= 128)
# NVIDIA has no MACA 4 KB/thread private-memory cap; the row kernel needs a register bound.
row_guard = lambda c: c.get("ROWS", 1) * c.get("BLOCK_N", 1) / (
    c.get("num_warps", 4) * WARP) <= 64

cn = autotune(lambda c: [p.unfused(c, SEED_GEMM)[0]], norm_grid(),
              warmup=tw, rep=tr_).best_cfg
```
The composed `guard` also closes a second hole: lines 91/93 pass **only** `_smem_ok`, no register bound at all. `BM128/BN256/BK32/st2` passes SMEM but needs 256 fp32 accumulators/thread (above sm89's 255-reg cap). ptxas spills rather than erroring, so `autotune` records a slow time and keeps it — and the **fused** arm, carrying the extra sq state, spills at strictly lower thresholds than the unfused arm at the same config. Coordinate descent then walks the two arms into structurally different regions, the exact failure `_tile_then_coord`'s own docstring (lines 110–118) was written to prevent.

### A5. `_row_guard` enforces a MACA-only launch cap on unfused-only kernels
`bench_layer.py:100–106`. The docstring states outright that this exists to dodge C500's `mcErrorMemoryValueTooLarge`. **That error does not exist on CUDA.** It is applied to `add` / `norm` / `addnorm` (lines 200–207) — kernels that appear **only in the unfused arm** of F3/F4/F5/F11b.

```python
def _row_guard(c):
    """Register-pressure bound for row kernels. (The original 4 KB/thread rule was a MACA
    launch-failure workaround; sm89 has no such cap, but the register file is half C500's.)"""
    threads = c.get("num_warps", 4) * C.env().warp_size
    return threads <= 1024 and c.get("ROWS", 1) * c.get("BLOCK_N", 1) / threads <= 64
```

### A6. `smem_bytes()` models only the unfused arm — asymmetric compile failures
`glm52/kernels/lazy_prenorm.py:393–399`. The docstring asserts "Identical for FUSE_NORM on and off". **The C500 run already disproved this**: 7 of 79 configs compiled unfused and failed fused by exactly +4096 B. `FUSE_NORM=True` emits a `tt.reduce` (SQ_MODE 0/1/3) or a second `tt.dot` (mode 2) needing cross-warp scratch.

Halving the warp size **doubles** the number of warps spanning the reduced axis for every tile (`BK=64` goes from 1 warp — warp-synchronous, zero scratch — to 2). Result: `n_tried` is equal for both arms while `n_failed` is not — **the fused arm silently searches a smaller effective grid.** Today this is masked only because `SMEM_LIMIT` is still 65536; **fixing A1 brings the asymmetric failures straight back at the new boundary.** Fix A1 and A6 together.

```python
def smem_bytes(cfg: dict, fused: bool = True, warp_size: int = 32) -> int:
    pipe = cfg["num_stages"] * 2 * cfg["BLOCK_K"] * (cfg["BLOCK_M"] + cfg["BLOCK_N"])
    if not fused:
        return pipe
    w_k = max(1, min(cfg["num_warps"], cfg["BLOCK_K"] // warp_size))
    return pipe + 4 * cfg["BLOCK_M"] * w_k  # cross-warp sum-of-squares scratch
```
Model the **worst** arm and apply it to **both**, so the two grids stay byte-identical.

### A7. L2 is 4× larger — four of five regimes no longer round-trip to DRAM
`glm52/traffic.py:21` `L2_BYTES = 8 * 2**20`. I confirmed it is a **dead constant** (only occurrence in the file) but it encodes the model's premise, restated at `traffic.py:10`, `:168`, and `bench_layer.py:64–66`.

`act = T*6144*2`: T=1 → 12 KiB, T=32 → 384 KiB, T=256 → 3.0 MiB, **T=2048 → 24.0 MiB (fits 32 MiB; did NOT fit C500's 8 MiB)**, T=8192 → 96.0 MiB (does not fit).

`bench_chain` correctly flushes once before the chain, so the **measured** speedup is fine. What breaks is the **ceiling** every bench prints next to it: F3's 1.25× is really ~1.0× at T≤2048; F10's 1.20× is really ~1.10×; F5/F4/F11b degrade the same way. A genuine 1.03× at prefill_t2048 will read as "far below the 1.25× ceiling" when the ceiling is unreachable *because the L2 already absorbed the traffic the fusion removes*.

Minimum fix — set `L2_BYTES = 32 * 2**20` and record residency so the report can label unreachable ceilings rather than presenting them as attainable:
- `bench_f03_resadd_rmsnorm.py:296` `extra` dict — add `"h1_bytes": T*row_bytes, "l2_bytes": env.l2_bytes, "h1_fits_l2": T*row_bytes <= env.l2_bytes,` and rename `gbps_*` → `gbps_*_model`.
- `bench_f10_merge_resadd.py:316` — **keep `ideal_speedup: 1.20` unchanged** (it must stay identical to C500 for comparability); add `"m_bytes": act, "l2_bytes": env.l2_bytes, "m_fits_in_l2": act <= env.l2_bytes,` and a `"ceiling_note": "1.20x is a DRAM-traffic ceiling; unattainable where m fits in L2"`.
- `bench_f04f05_norm_router.py:708` `bytes_model` — annotate the F5/F4 note strings.
- Update the stale crossover comment at `bench_layer.py:64–66`: `8*2**20/(6144*2) = 683` becomes **2730** tokens on a 32 MiB L2, which moves both `decode_bs512` and `decode_bs1024` onto the "fits L2" side.

### A8. Config-grid pruning is invisible in the output
`bench_layer.py:122–126` and `:149–156` drop guard-rejected configs **before** `autotune` sees them. `autotune` then reports `n_tried = len(cfgs)` (`common.py:180`) — the count of *survivors*. Nothing records how many were rejected or by which predicate. `n_failed` is recorded correctly; only the pre-filter is invisible.

This is what makes a biased grid undetectable: a reviewer comparing C500's `n_tried` to this run's cannot tell whether the difference is a different device or a stale constant.

```python
# common.py:128-143 -- add to TuneResult
n_rejected: int = 0
# and in as_dict(): "n_rejected": self.n_rejected,

# bench_layer.py:122-128
    cands = []
    for bm, bn, bk in tiles:
        c = dict(seed, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
        if guard is None or guard(c):
            cands.append(c)
    n_rejected = len(tiles) - len(cands)
    tr = autotune(make, cands, warmup=warmup, rep=rep)
    tr.n_rejected = n_rejected
    print(f"  [cfgs] {tag:<16} tile sweep {tr.best_ms:9.4f} ms "
          f"({tr.n_tried} cfgs, {n_rejected} guard-rejected)", flush=True)
```

### A9. `bench_f04f05` — `NORM_BK` offers exactly one value, and it spills at warp 32
`bench_f04f05_norm_router.py:141–144`: `wide = min(2048, 32768 // c["BLOCK_M"])` fixes the pass-1 tile at exactly `BLOCK_M*NORM_BK = 32768` elements for **every** config, and `fused_grid()` offers only that single value.

At warp 64 that was 128/64/32 elem/thread for 4/8/16 warps — affordable. **At warp 32 it is 256/128/64.** At `num_warps=4` the pass-1 tile alone is 256 bf16 elements/thread plus its fp32 copy: >255 registers/thread on half the register file. ptxas spills. On C500, 10 of 20 winning fused configs carried `NORM_BK=2048`, 8 of them at 4 or 8 warps — **the tuner will now fall back to `NORM_BK=BLOCK_K` and the fused arm loses the knob entirely**, for a reason that is a C500 register budget, not a real result.

```python
    for nbk in (256, 512, 1024, 2048):
        if (nbk > c["BLOCK_K"] and C.HIDDEN_SIZE % nbk == 0
                and c["BLOCK_M"] * nbk <= 64 * WARP * c["num_warps"]):
            out.append(dict(c, NORM_BK=nbk))
```

Also at `bench_f04f05:87`: `if bm * be * 4 > SMEM` is a **register** budget wearing a shared-memory constant. Replace with `if bm * be > 128 * WARP * w: return False`.

### A10. `bench_f01` stage-3 gives the unfused arm ~11% more trials
`bench_f01_oproj_resadd.py:327–336`. Confirmed: `if cfg.get("SPLIT_K",1) == 1: continue` skips the fused arm entirely, so `joint_f` is empty; the unfused arm unconditionally gets 3 GEMM × 5 epilogue = **15 extra timed chains**. The C500 checkpoints show this fires at prefill_t8192 and decode_bs256 — i.e. **at the headline 0.846× regime the unfused arm was searched over 15 more points.** The module docstring and the recorded `notes.fairness` both claim "identical, independent budgets".

Final numbers are re-measured equally (lines 372–373), so this biases config *selection*, not measurement — but it pushes the ratio down, the same direction as the regression under study. After line 336:
```python
    if not joint_f and joint_u:
        extra = refine_grid(f_cfg, M, seen_f)[:len(joint_u)]
        t_f_extra = autotune(lambda c: fused_chain(c, None), extra, tw, tr)
        # fold into merge alongside t_f_coarse / t_f_ref
    assert log["joint_grid_size_fused"] == log["joint_grid_size_unfused"], log
```

### A11. Persistent-grid caps are multiples of 104 CUs
`bench_f03_resadd_rmsnorm.py:121` and `bench_f10_merge_resadd.py:59`: `(104, 208, 416, 832, 1664)`. The 4060 has **24** SMs, so 104 CTAs is 4.33 waves — every rung is a fractional wave and the knob can never be evaluated at a balanced grid. In f10 this burns 25 of ~45 refine configs; in f03, 30 of ~92. Symmetric across arms (not a ratio bias), but it wastes a large share of the refine budget on configs that cannot win for a structural reason.

```python
_SM = C.env().num_sm
CAPS = [_SM * m for m in (1, 2, 4, 8, 16)]   # [24,48,96,192,384] here; [104,...] on C500
```

---

## PART B — Blockers (crash, or silently emit C500 numbers)

### B1. ⚠️ Stale C500 checkpoints are reused — produces a perfect forgery
**Confirmed, and worse than reported.**

- `bench_f10_merge_resadd.py:410` — `if cf.exists(): d = json.loads(...)` with **no force-escape at all**. All five checkpoints exist. The bench runs **zero kernels**, prints "(from checkpoint)" five times, and writes `f10_merge_resadd.json` containing C500 timings wrapped in a freshly-probed RTX 4060 `env` and `_meta.device`.
- `bench_f01_oproj_resadd.py:465` — same, gated by `F01_FORCE`. I read `results/_f01_oproj_resadd_ckpt/prefill_t8192.json`: `fused_ms 18.383, unfused_ms 15.549, speedup 0.8458`. That is the study's headline datapoint and it would be republished as a 4060 measurement.

**And a trap:** `bench_f01`'s `CKPT_DIR` is `Path(__file__).resolve().parents[2] / "results"` (line 454) — it does **not** honour `GLM52_RESULTS_DIR`, while its `record()` does. Setting the env var alone gives you the **worst** case: C500 checkpoint read from `results/`, written into `results/rtx4060/` under a genuine 4060 env block.

Do both:
```bash
mkdir -p results/c500
git mv results/_f01_oproj_resadd_ckpt results/c500/_f01_oproj_resadd_ckpt
git mv results/_f10_merge_resadd_ckpt results/c500/_f10_merge_resadd_ckpt
git mv results/_f04f05_norm_router_ckpt results/c500/_f04f05_norm_router_ckpt
export GLM52_RESULTS_DIR=/home/shuhan/fusion/results/rtx4060
```
And make it structural. `bench_f01_oproj_resadd.py:454`:
```python
from glm52.common import RESULTS_DIR
CKPT_DIR = RESULTS_DIR / f"_{RESULT_ID}_ckpt"
```
Plus a device fence at `bench_f01:465` and `bench_f10:410`:
```python
if cf.exists() and json.loads(cf.read_text()).get("device") == torch.cuda.get_device_name(0):
```
writing `"device": torch.cuda.get_device_name(0)` into the payload at `bench_f01:473` / `bench_f10:423`.

**`GLM52_RESULTS_DIR` gives only partial isolation** — three conventions coexist: `common.RESULTS_DIR` (env-aware: f03/f04f05/f10/f11 output, f04f05+f10 checkpoints), `parents[2]/"results"` (f01 checkpoints), and `ROOT/"results"` (`bench_f11b_sweep.py:44–45` `RESULTS`/`QDIR`, `bench_layer.py:58`). Point all three at `common.RESULTS_DIR`.

### B2. `sys.path.insert(0, "/home/zhangshuhan/fusion")` — ModuleNotFoundError at import
Confirmed at **exactly three in-scope benches** plus one diag (f01, f11, f11b_sweep already use `parents[2]`/`ROOT` and are fine):

`bench_f03_resadd_rmsnorm.py:21`, `bench_f04f05_norm_router.py:35`, `bench_f10_merge_resadd.py:29`, `diag_f10_merge_resadd.py:13`
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
```
(`Path` is already imported in all three benches.) Also `diag_f10_merge_resadd.py:21` `RES_JSON` → derive from that root.

### B3. `bench_f11_lazy_prenorm.py` OOMs at startup — 25.8 GB before any regime
**Confirmed.** `main()` line ~908 calls `make_w13(w_norm)` unconditionally; lines 229–231 allocate two `E*NW13*H + 2**20` bf16 buffers = **12.884 GB each**. There is **no CLI flag to skip it** — `argparse` has only `--quick` and `--regimes` (lines 895–896). F11a is explicitly out of scope. The process dies in `torch.empty` before a single F11b number exists.

Add at line 896:
```python
ap.add_argument("--router-only", action="store_true", default=True,
                help="skip F11a w13 (12.9 GB x2); required on 8 GB cards")
```
Then gate: lines 907–909 (`w13_raw = w13_fold = None`), the `lhs13/rhs13` fold check (921–923), the `F11a : w13` block (419–456), `("moe", prob.moe_half, ...)` at 465, lines 491–492/495, chk entries 523–525 and `moe_*` at 526–529, timings 548–552 and 559–561, `("moe", mo_f_cfg, prob.moe_fused)` at 596, moe kstats 621–625, vendor 637–643, and the `f11a_w13`/`combined` rows (682–718). In `Problem.__init__`, drop `c_f`/`c_u`/`c_h` (lines 269–271, `[T*8, 4096]` bf16 = 512 MB each at T=8192) and `self.layouts` when `w13 is None`. Peak becomes ~0.44 GB at T=8192.

### B4. `JITFunction.cache` removed in Triton 3.x — all attribution diagnostics dead
Verified statically: `triton/runtime/jit.py:771` `self.device_caches = defaultdict(self.create_binder)`; line 707 unpacks the 5-tuple `(kernel_cache, kernel_key_cache, target, backend, binder)`. There is no `.cache` attribute. Four sites:

| File | Line | Fix |
|---|---|---|
| `analyze_f11b_arch.py` | 40, 49 | `L.router_gemm_kernel.device_caches.clear()` then `ent = [v for tup in L.router_gemm_kernel.device_caches.values() for v in tup[0].values()]`; same for `NK.add_rmsnorm_kernel` |
| `diag_f10_merge_resadd.py` | 27, 41 | `cache = ks.device_caches[dev][0]` / `K.merge_resadd_kernel.device_caches[dev][0].clear()`; also `getattr(v, "shared", None)` → `getattr(v.metadata, "shared", None)` |
| `bench_f04f05_norm_router.py` | 245 | `kc = K.norm_router_kernel.device_caches[dev][0]` / `kc.clear()`; line 249 `vals = list(kc.values())` |
| `bench_f11_lazy_prenorm.py` | 344 | `kern.device_caches.clear()`; drop the unused `dev` on 341 |

`analyze_f11b_arch` and `diag_f10` **crash**. `bench_f04f05:245` and `bench_f11:344` are inside bare `except Exception` so they degrade silently — `kernel_stats` comes back as `{"error": "AttributeError: ..."}`. On a port whose single biggest hardware delta is register file per SM (65536 vs 131072), losing `n_regs`/`n_spills` removes exactly the diagnostic needed to explain a fused-arm regression.

### B5. `reference.expert_merge` — 3.0 GiB transient inside a timed arm
`glm52/reference.py:159–163`. Confirmed: `per_expert_out.float()` allocates one `[T,topk,H]` fp32 temp (1536 MiB at T=8192) and `* topk_weights.unsqueeze(-1)` allocates a second before `.sum(1)` reduces — **3072 MiB peak transient**, on top of ~1664 MiB resident. Total ~4.97 GB of a ~7.4 GB budget: it *should* fit, but it is the tightest point in the port.

`bench_f10_merge_resadd.py:274` `t_eager = bench_chain([torch_eager], warmup_f, rep_f)` has **no try/except** (the `torch.compile` arm below it at 276–289 does). An OOM there aborts the regime *after* ~10 min of tuning and **before** the checkpoint write at 423 — losing the entire prefill_t8192 tuning result, the one regime where the 1.20× ceiling is actually attainable.

```python
def expert_merge(per_expert_out, topk_weights, chunk: int = 512):
    """per_expert_out: [T, topk, H] (unweighted) -> [T, H].
    Chunked over T: bounds the fp32 transient at 2*chunk*topk*H*4 = 192 MiB at chunk=512.
    Bit-identical to the unchunked form (same fp32 accumulate, same bf16 round)."""
    T, k, H = per_expert_out.shape
    out = torch.empty(T, H, device=per_expert_out.device, dtype=per_expert_out.dtype)
    for i in range(0, T, chunk):
        j = min(i + chunk, T)
        out[i:j] = (per_expert_out[i:j].float()
                    * topk_weights[i:j].unsqueeze(-1)).sum(1).to(per_expert_out.dtype)
    return out
```
Plus at `bench_f10:270–274`, wrap in the same `try/except torch.cuda.OutOfMemoryError` used for `torch.compile` and guard the consumers at 318 and 341; and after line 251 add `del ref_out_hi, out_hi; torch.cuda.empty_cache()`. (Line 248–250 builds `ref_out_hi` with the same unchunked 2× pattern inline — route it through `ref.expert_merge` too.)

### B6. `bench_f11b_sweep.py` has no T bound — C500 drove it to T=262144
`make_queue` (line 137) accepts arbitrary T. The binding constraint is `reference()` (lines 74–76), pure eager fp32: `R.rmsnorm` holds `h1.float()` (24576 B/tok) and `xf*rstd` (24576) live with the bf16 result (12288), then `x2.float()` adds 24576 → **87040 B/token**. T=262144 needs 21.3 GB; T=131072 needs 10.6 GB.

**Max safe T as written = 65536** (~5.5 GB with the flush buffer and weights). Chunking `reference()` drops the peak to the 25600 B/tok persistent set and makes T=131072 safe.

The sweep does not need those points: the 4060's 84.3 flop/byte vs C500's 82.3 puts the crossover near the same T≈4096, and a sweep to 32768 fully brackets it. In `make_queue`, after the `for t in ...`:
```python
            if t > 65536:
                raise SystemExit(f"T={t} needs {t*87040/2**30:.1f} GB; 8 GB cap is T<=65536")
```

---

## PART C — Correctness of the kernels themselves

### C1. Two unsynchronised cross-thread handoffs in `norm_router.py`
These are **latent wrong answers**, and the C500 code got away with them because MACA picked compatible layouts.

- **`kernels/norm_router.py:236–245`** (`FUSE_TOPK` epilogue): stores 8 winners to global `TOPW` and re-loads them with **no barrier**. The comment at 224–231 asserts "same thread reads back its own element", but the store's value `v = tl.max(cur, axis=1)` carries a layout sliced from `cur`'s MMA layout (descended from the `tl.dot` accumulator), while the load at 244 produces a fresh `[BLOCK_M]` tensor with a default blocked layout mapping element *m* to a different lane. Insert `tl.debug_barrier()` immediately before line 239 (`if NORM_PROB:`) — lowers to `bar.sync`, one instruction.

- **`kernels/norm_router.py:328`** `REREAD_H1`: pass 1 stores `h` at line 157 with a plain blocked layout; pass 2 loads the same addresses at line 183 into what becomes the `tl.dot` A-operand at 203, where Triton 3.6's pipeliner may choose cp.async / a dot-operand encoding. No barrier between the loops. Live for every `BLOCK_E=256` config without `NORM_BK` (~24 of the fused grid). **Zero-risk port option:** change the default at line 328 to `cfg.get("REREAD_H1", False)` — the kernel already has the re-add path at 184–187 — until the re-read is validated on sm89.

Either way the failure mode is a `screen()`-time NUMERIC rejection that **quietly deletes fused-#4 configs**, or a wrong answer at the tolerance edge.

### C2. `analyze_f11b_arch.py:20` — occupancy model describes a machine that does not exist
`H, ER, SMEM, REGS_SM, NSM = 6144, 256, 65536, 131072, 104`. This script produces the `ctas_per_sm` / `limited_by` / `warps_per_sm` figures the **entire #11b architectural narrative** in `REPORT-lazy-prenorm.md` rests on.

Two errors cancel (`REGS_SM` 2× high × `threads` 2× high → `by_reg` accidentally right), but `by_sh = SMEM // shared` does not: `SMEM` should be **102400** (per-SM), so smem-limited occupancy is understated 1.56× and `limited_by` reports "smem" where the real limiter is registers — the exact opposite of the true 4060 constraint. `NSM` is 4.3× high.

```python
H, ER, SMEM, REGS_SM, NSM = 6144, 256, 102400, 65536, 24   # SMEM/REGS are per-SM
```
plus the `* 64`→`* 32` / `// 64`→`// 32` edits from A2 (**together**, or `by_reg` breaks), and add the caps that actually bind on Ada:
```python
    return min(by_reg, by_sh, 1536 // threads, 24), by_reg, by_sh
```
Also note `BenchEnv.regs_per_sm` is **misnamed**: `driver.c:79` sources `max_num_regs` from `CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK`, not per-SM. On sm89 both are 65536 so the value is right by coincidence; C500's recorded 131072 is a per-SM figure. **The field means different things in the two result sets.** Rename to `max_regs_per_block` and add a real `regs_per_sm`.

### C3. `BenchEnv.probe()` degrades silently to half-C500
`config.py:111–120`. If the Triton probe throws (`CudaUtils()` JIT-builds and dlopens a C extension on first use — exactly what fails on a fresh box), lines 117/119/120 fall back to `warp_size=64`, `smem_bytes=65536`, `regs_per_sm=131072`, while line 118 still picks up the real 24 SMs and 121 the real 32 MB L2. `env.__dict__` is written verbatim into every results JSON (`bench_f03:355`, `bench_f10:401`, `bench_f11:998`) — **indistinguishable from a genuine C500 run.**

I verified torch 2.11 exposes `warp_size`, `major`, `minor`, `max_threads_per_multi_processor`, `shared_memory_per_block`, `shared_memory_per_multiprocessor`, `L2_cache_size` (`torch/_C/__init__.pyi:12448–12465`) — but **not** `shared_memory_per_block_optin` or `regs_per_multiprocessor`, so Triton remains the only source for those two.

```python
@dataclass
class BenchEnv:
    device_name: str = ""
    warp_size: int = 0
    num_sm: int = 0
    smem_bytes: int = 0             # per-block OPT-IN: the Triton grid-admission ceiling
    smem_default_per_block: int = 0
    smem_per_sm: int = 0
    max_regs_per_block: int = 0     # CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
    threads_per_sm: int = 0
    cc_major: int = 0
    cc_minor: int = 0
    l2_bytes: int = 0
    torch_version: str = ""
    triton_version: str = ""
    probe_ok: bool = True
    extras: dict = field(default_factory=dict)

    @staticmethod
    def probe() -> "BenchEnv":
        import triton
        p = torch.cuda.get_device_properties(0)
        props, ok = {}, True
        try:
            props = triton.runtime.driver.active.utils.get_device_properties(0)
        except Exception:
            props, ok = {}, False
        return BenchEnv(
            device_name=p.name,
            warp_size=props.get("warpSize", p.warp_size),
            num_sm=props.get("multiprocessor_count", p.multi_processor_count),
            smem_bytes=props.get("max_shared_mem", p.shared_memory_per_block),
            smem_default_per_block=p.shared_memory_per_block,
            smem_per_sm=p.shared_memory_per_multiprocessor,
            max_regs_per_block=props.get("max_num_regs", 0),
            threads_per_sm=p.max_threads_per_multi_processor,
            cc_major=p.major, cc_minor=p.minor,
            l2_bytes=p.L2_cache_size,
            torch_version=torch.__version__,
            triton_version=triton.__version__,
            probe_ok=ok,
            extras={**props, "triton_props_ok": ok},
        )
```
Delete the `64` / `65536` / `131072` literals from lines 97–101 and 117–120. **Assert `probe_ok` at each bench's startup** — a degraded probe must abort, not warn.

---

## PART D — Wall clock (this GPU is ~10× slower per byte)

### D1. `bench_chain` runs a wasted second measurement pass for every tuning config
`common.py:106–112`: `nf = sorted(_measure(False))` runs unconditionally. `autotune` (line 165) calls `bench_chain` once per config and uses only `t.p50_ms` — the no-flush number is computed and discarded for **every config in every grid**.

At 140 GB/s the 128 MiB flush costs ~0.96 ms/rep (it cost ~0.10 ms on C500 at 1300 GB/s). At `rep=30` that is ~29 ms of pure memset per config, and the wasted pass adds another 30 reps of kernel time on top. Across a 48-tile sweep plus two coordinate rounds, per variant, per regime, this is the difference between a tractable serialized campaign and one that does not finish.

```python
def bench_chain(fns, warmup=25, rep=100, flush=True, noflush=True) -> Timing:
    ...
    times = sorted(_measure(flush))
    noflush_ms = float("nan")
    if noflush:
        try:
            nf = sorted(_measure(False))
            noflush_ms = statistics.median(nf)
        except Exception:
            pass
```
and at `common.py:165`: `t = bench_chain(chain, warmup=warmup, rep=rep, flush=True, noflush=False)`.

Optionally halve the flush: `_FLUSH_BYTES = max(64*2**20, 2*L2)` = 64 MiB is still 2× L2 and a valid full eviction, saving ~0.5 ms/rep. Only do this if the campaign is otherwise too long — 4× is the safer margin.

### D2. `moe_align_block_size` — 769 forced syncs per call
`reference.py:104–118`: loops over 256 experts doing `counts[e].item()`, `starts[e].item()`, `padded[e].item()`, plus `int(padded.sum().item())`. Called from `bench_f11_lazy_prenorm.py:277` for each candidate `BLOCK_M`. Setup-only (biases nothing), but tens of ms per call under WSL2. Hoist once before line 112:
```python
    counts_l, starts_l, padded_l = counts.tolist(), starts.tolist(), padded.tolist()
```
and index the Python lists inside the loop. One sync instead of 769, identical output.

### D3. `bench_f03:70` `num_stages` sweep is 100% redundant in the coarse grid
For every non-persistent config the tile loop is `tl.static_range` (`add_rmsnorm.py:144, 168`), unrolled in the AST — the pipeliner has nothing to schedule and both stage counts compile to identical machine code. Half of the 152 coarse configs are exact duplicates. The C500 archive corroborates: the winning `num_stages` flips randomly between 1 and 2 across regimes at the same shape — pure timing noise over identical binaries. Drop `(1, 2)` → `(1,)` at line 70; refine section (b) at line 112 already sweeps stages 1–4 at the winning shape. *(Skip if you want the grid shape byte-identical to C500's.)*

---

## PART E — Lower priority (do after the run starts, or before writing the report)

- **`bench_f01`/`f04f05`/`f11` under-parallelised at decode.** `bench_f04f05:100` `gemm_grid()` starts `BLOCK_M` at 16 and `BLOCK_E` at 32, so decode_bs1 emits at most 8 programs and decode_bs32 at most 16 — on 24 SMs that leaves 67%/33% idle for **both** arms. Triton 3.6's sm89 backend reports `min_dot_size = (1,1,16)` (verified at `backends/nvidia/compiler.py:19`), so `BLOCK_M ∈ {1,2,4,8}` and `BLOCK_E=16` are now legal. Widening line 100 to `(8,16,32,64,128)` and adding 16 to the `BLOCK_E` tuple raises decode_bs32 to 64 programs. Symmetric (not a ratio bias), but the decode absolutes are pessimistic. `BLOCK_E` must stay 256 for `FUSE_TOPK` (`norm_router.py:286`).
- **`bench_f11b_sweep.py:71` never passes `sq_mode`** — silently takes the default 0, inheriting the C500 answer by omission and never recording it. Mode 0 is exactly the mode whose cost is warp-size dependent (per-k-step cross-lane reduction; at warp 32 it needs an extra shuffle stage plus, for `BLOCK_K>=64`, a cross-warp SMEM round trip that did not exist at warp 64). Thread it through and re-run the `sq_study` (`bench_f11:850–889`) on this hardware; record `sq_mode` in the row at line 117.
- **`norm_router.py:232–245` global-memory top-k spill** is a documented MACA codegen workaround (lines 67–71) for a defect that does not exist on sm89 — the identical broadcast is already used correctly at line 238. Make it a `TOPK_SPILL: tl.constexpr` and offer the **same flag value to both arms** (fused variant and standalone `topk_only`), or it becomes a fairness defect.
- **`oproj_resadd.py:89` has no `EVEN_K`** — K is always 16384/32768, a multiple of `BLOCK_K*SPLIT_K` for essentially every grid config, but Triton emits a live residue comparison plus predicated cp.async for all 128–512 mainloop iterations. Pure win, identical on both arms; matters more here because mainloop register pressure binds on half the register file.
- **`add_rmsnorm.py:126, 158`** — the `EVICT` knob reaches loads but not the H1 stores, so the producer→consumer handoff can only be hinted in one direction. The missing half can only help the arm that re-reads, i.e. omitting it slightly favours the fused arm.
- **`lazy_prenorm.py:191, 304`** — `SQ_MODE=3`'s reload omits the `offs_k < K - k_start` term. Latent only (K=6144 is divisible by all `BLOCK_K` here), but it would silently corrupt `rstd` on reuse.
- **`bench_f11:135`** — `256 if max_bn <= 256 else 256` is dead; the `max_bn` parameter has no effect on the `BLOCK_N` neighbourhood. Inert today, silently misrepresents the refine neighbourhood in the fairness note at 989–993.
- **`common.py:116`** `p50_ms = times[n//2]` returns the 51st of 100, not the median. Sub-1% and cancels in the ratio; use `statistics.median(times)`.
- **Provenance:** `common.py:206–219` `record()` omits Triton version, compute capability, SMEM limits and **the clock lock** — the single fact most needed to interpret an absolute latency from this box. Have `record()` merge a cached `asdict(BenchEnv.probe())` into `_meta` so every result file is self-contained, and drop the per-bench `"env": env.__dict__` lines.
- **Stale prose that will be published as fact:** `common.py:156–157` (autotune docstring, now says the opposite of the truth), `traffic.py:10, 36–54, 168`, `bench_layer.py:64–66`, `bench_f04f05:829` (hardcodes C500 grid sizes — "80 cfgs", "160", "48", "58", "80+58=138 vs 160" — into the deliverable JSON as the study's own fairness accounting; compute these from live `len()` at line 298), `bench_f10:398` (asserts "bitwise identical (asserted)" when line 252–254 deliberately excludes that check, and identity only holds when both arms pick `KVEC=0`), `bench_f10:383` (`traffic_model` omits the 13/12 = 1.083× production caveat from LOG-06 §7.1), and the run recipes at `bench_f01:4`, `f03:11`, `f04f05:23`, `f10:17`, `f11:27` (all name a nonexistent interpreter and `CUDA_VISIBLE_DEVICES=1/3` on a single-GPU box — copy-pasting costs a scheduling slot for no data).
- **`bench_f01:367`** `check(out_f, out.float(), tol=0.0)` — zero-tolerance fused-vs-unfused comparison between two independently-tuned arms that may differ in `SPLIT_K` and atomic ordering. Already `ok:false` at both prefill regimes on C500. Give it `tol=2e-2`.

---

## 5. The dangerous ones — silently wrong-but-plausible

Every item here produces a result file that **looks completely normal**. Ranked by how badly it would mislead:

1. **B1 — stale checkpoint reuse (f01, f10).** A perfect forgery: C500 timings inside a genuine RTX 4060 `env` block and `_meta.device` header. f10 has no escape hatch at all. Nothing in the output distinguishes it from a real run. **Fix first.**
2. **A6 — `smem_bytes()` models only the unfused arm.** `n_tried` is equal for both arms while `n_failed` is not, so the fused arm searches a smaller effective grid *while the JSON reports symmetric budgets*. C500 already exhibited it (7 of 79 configs). Currently masked by the stale 65536 cap — **fixing A1 alone re-exposes it at the new boundary.**
3. **A1 + A8 together — corrected probe, uncorrected grid.** `BenchEnv` probes 101376 and then throws it away; the grids are filtered by independent literals. A port that fixes only `BenchEnv` writes a result file whose `env` says 101376 while the searched grid was capped at 65536 — **a result documenting a search it did not perform**, with no `n_rejected` field to reveal it.
4. **A2 one-sided sites (`bench_f01:202`, `bench_f04f05:152`, `bench_f11:173/186`).** Prune only the unfused arm's grid, or leave it stuck on a spilling config. Both directions **inflate the fusion win**, in the same direction as the study's thesis.
5. **A4 — f11b unfused norm unreachable.** Inflates the fused arm at every T in the crossover sweep, which *is* the module's deliverable.
6. **A7 — 32 MB L2.** The measured ratio stays honest; the **ceiling** printed beside it does not. "1.03× against a 1.25× roofline" reads as an under-performing kernel when the ceiling is unreachable because the L2 absorbed the traffic. Affects 4 of 5 regimes.
7. **A10 — f01 stage-3.** 15 extra trials for the unfused arm at the 0.846× headline regime, while `notes.fairness` claims identical budgets.
8. **C3 — degraded probe.** Stamps `warp_size: 64, smem_bytes: 65536, regs_per_sm: 131072` into an RTX 4060 result file with no error and no flag.
9. **C1 — unsynchronised handoffs.** Wrong numbers, or (more likely) `screen()`-time NUMERIC rejections that quietly shrink the fused grid.
10. **C2 — `analyze_f11b_arch`.** Wrong `limited_by` labels feeding the #11b architectural argument. Crashes first (B4), so it fails loudly — but if B4 is fixed without C2, it starts emitting plausible nonsense.

---

## 6. Verdict

**Not safe to run today. Safe after Parts A + B + C1 + C3.**

As it stands: **f03 / f04f05 / f10 die at import** (B2), **f11 dies at startup** (B3), and **f01 / f10 would silently republish C500 numbers** (B1). Of the three that could produce data, all would search grids truncated by C500 constants.

Minimum bar before the first kernel runs: **A1, A2, A5, A6, A8** (grid integrity — every arm searched over the same, correct, auditable space), **B1, B2, B3, B5, B6** (it runs at all, on real data), **C3** (`probe_ok` asserted, so a degraded probe aborts). A3, A7, A9, A10, A11 and C1 should land in the same pass — they are cheap and each one otherwise bends a published number.

**Unresolved and needing a decision:**

1. **A3's `C_PEAK`.** `traffic.py`'s own rule (line 43) is to use the *Triton* ceiling, and `results/rtx4060_gemm_ceiling.json` has it: 11.81 TF/s from a dense sweep at `M4096 K16384 N6144`. But that sweep ran on the **pre-fix** grid, capped at SMEM 65536 — so it may itself be understated. **Re-run the GEMM ceiling calibration after A1 lands, before fixing `C_PEAK`.** The balance point barely moves either way (84.3 vs C500's 82.3), so no `compute_bound` label is at risk; only absolute predictions are.

2. **The brief's numbers disagree with the repo's.** Brief: 128.8 GB/s, 11.11 TF/s. On disk: 140 GB/s stream r/w, 159 GB/s read-only, 11.81 TF/s Triton, 11.62 TF/s cuBLAS. **Reconcile before anything is published** — a report quoting one while the code uses the other is exactly the kind of inconsistency a reviewer checks first.

3. **C1's two barriers are unverified on hardware.** I confirmed statically that the layout assumption is unjustified in both places, but I could not run anything. If the fused #4/#5 arms show unexpected `screen()` NUMERIC rejections or an anomalous `n_failed` gap versus their unfused counterparts, that is this defect. The zero-risk path is `REREAD_H1=False` by default until validated.

4. **A6's scratch formula is a model, not a measurement.** `pipe + 4*BLOCK_M*w_k` is my best static estimate of Triton's cross-warp reduction scratch. **Validate empirically before trusting the grid:** compile a handful of configs at both `FUSE_NORM` settings and compare `kernel.metadata.shared`. If it under-predicts, the asymmetric-failure hole reopens silently. Until then, **check `n_failed` is comparable between each fused/unfused pair in every `TuneResult`** — that is the observable that catches it, and it is the check LOG-06's "0 compile failures" claim must be re-derived from rather than inherited.

5. **`GLM52_RESULTS_DIR` is not sufficient on its own.** Until `bench_f01`'s `CKPT_DIR` and `bench_f11b_sweep`'s `RESULTS`/`QDIR` are routed through `common.RESULTS_DIR`, setting the env var produces a *mixed* state — which for f01 is worse than not setting it. Do the `git mv` **and** the code change."
  },
  "workflowProgress": [
    {
      "type": "workflow_phase",
      "index": 1,
      "title": "Audit"
    },
    {
      "type": "workflow_phase",
      "index": 2,
      "title": "Synthesize"
    },
    {
      "type": "workflow_agent",
      "index": 1,
      "label": "common+config",
      "phaseIndex": 1,
      "phaseTitle": "Audit",
      "agentId": "a06043ad0f7408de9",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785479398939,
      "queuedAt": 1785479398890,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "harness — glm52/common.py, glm52/config.py, glm52/reference…",
      "promptPreview": "You are auditing a Triton kernel-fusion benchmark suite that was written for and tuned on a
MetaX C500 datacenter GPU, and is now being ported to run on a consumer RTX 4060 Laptop GPU.
The repo root is /home/shuhan/fusion. Study code lives in glm52/.

OLD platform (what the code was written for):
  MetaX C500, MACA 3.7, torch 2.8.0+metax, triton 3.0.0, backend 'maca', arch 80
  104 CUs | warp_size…",
      "lastProgressAt": 1785480215238,
      "tokens": 90961,
      "toolCalls": 31,
      "durationMs": 816297,
      "resultPreview": "{"module":"harness — glm52/common.py, glm52/config.py, glm52/reference.py, glm52/traffic.py (+ the genuinely shared glm52/bench/bench_layer.py, imported by in-scope bench_f11b_sweep.py)","findings":[{"severity":"blocker","category":"memory-footprint","file":"glm52/reference.py","line":161,"problem":"`expert_merge` materialises TWO full fp32 [T, topk, H] temporaries: `per_expert_out.float()` alloca…"
    },
    {
      "type": "workflow_agent",
      "index": 2,
      "label": "f01-oproj",
      "phaseIndex": 1,
      "phaseTitle": "Audit",
      "agentId": "ad5540f276c987920",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785479398948,
      "queuedAt": 1785479398890,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "f01 — o_proj GEMM + ResAdd (glm52/kernels/oproj_resadd.py, …",
      "promptPreview": "You are auditing a Triton kernel-fusion benchmark suite that was written for and tuned on a
MetaX C500 datacenter GPU, and is now being ported to run on a consumer RTX 4060 Laptop GPU.
The repo root is /home/shuhan/fusion. Study code lives in glm52/.

OLD platform (what the code was written for):
  MetaX C500, MACA 3.7, torch 2.8.0+metax, triton 3.0.0, backend 'maca', arch 80
  104 CUs | warp_size…",
      "lastProgressAt": 1785480102680,
      "tokens": 90045,
      "toolCalls": 25,
      "durationMs": 703732,
      "resultPreview": "{"module":"f01 — o_proj GEMM + ResAdd (glm52/kernels/oproj_resadd.py, glm52/bench/bench_f01_oproj_resadd.py)","findings":[{"severity":"blocker","category":"stale-c500-checkpoints","file":"glm52/bench/bench_f01_oproj_resadd.py","line":465,"problem":"`results/_f01_oproj_resadd_ckpt/` already contains committed C500 results for exactly the five regimes that will be run (decode_bs1.json, decode_bs32.j…"
    },
    {
      "type": "workflow_agent",
      "index": 3,
      "label": "f03-resadd-norm",
      "phaseIndex": 1,
      "phaseTitle": "Audit",
      "agentId": "a6774ef9492a637ca",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785479398949,
      "queuedAt": 1785479398890,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "f03 — ResAdd + RMSNorm (glm52/kernels/add_rmsnorm.py + glm5…",
      "promptPreview": "You are auditing a Triton kernel-fusion benchmark suite that was written for and tuned on a
MetaX C500 datacenter GPU, and is now being ported to run on a consumer RTX 4060 Laptop GPU.
The repo root is /home/shuhan/fusion. Study code lives in glm52/.

OLD platform (what the code was written for):
  MetaX C500, MACA 3.7, torch 2.8.0+metax, triton 3.0.0, backend 'maca', arch 80
  104 CUs | warp_size…",
      "lastProgressAt": 1785480176215,
      "tokens": 99866,
      "toolCalls": 24,
      "durationMs": 777266,
      "resultPreview": "{"module":"f03 — ResAdd + RMSNorm (glm52/kernels/add_rmsnorm.py + glm52/bench/bench_f03_resadd_rmsnorm.py)","findings":[{"severity":"blocker","category":"hardcoded-c500","file":"glm52/bench/bench_f03_resadd_rmsnorm.py","line":21,"problem":"`sys.path.insert(0, \\"/home/zhangshuhan/fusion\\")` points at the C500 box's home dir, which does not exist here. The repo is not installed (no pyproject/setup.p…"
    },
    {
      "type": "workflow_agent",
      "index": 4,
      "label": "f04f05-norm-router",
      "phaseIndex": 1,
      "phaseTitle": "Audit",
      "agentId": "a502770d5427f762a",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785479398949,
      "queuedAt": 1785479398890,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "f04f05",
      "promptPreview": "You are auditing a Triton kernel-fusion benchmark suite that was written for and tuned on a
MetaX C500 datacenter GPU, and is now being ported to run on a consumer RTX 4060 Laptop GPU.
The repo root is /home/shuhan/fusion. Study code lives in glm52/.

OLD platform (what the code was written for):
  MetaX C500, MACA 3.7, torch 2.8.0+metax, triton 3.0.0, backend 'maca', arch 80
  104 CUs | warp_size…",
      "lastProgressAt": 1785480212915,
      "tokens": 130137,
      "toolCalls": 25,
      "durationMs": 813966,
      "resultPreview": "{"module":"f04f05","findings":[{"severity":"blocker","category":"hardcoded-c500","file":"glm52/bench/bench_f04f05_norm_router.py","line":35,"problem":"`sys.path.insert(0, \\"/home/zhangshuhan/fusion\\")` points at the old machine's repo root, which does not exist here (repo is /home/shuhan/fusion). When the file is run the way its own docstring says (`python glm52/bench/bench_f04f05_norm_router.py`)…"
    },
    {
      "type": "workflow_agent",
      "index": 5,
      "label": "f10-merge-resadd",
      "phaseIndex": 1,
      "phaseTitle": "Audit",
      "agentId": "a3e61f65738a45ba5",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785479398950,
      "queuedAt": 1785479398890,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "f10 Expert Merge + ResAdd — glm52/kernels/merge_resadd.py +…",
      "promptPreview": "You are auditing a Triton kernel-fusion benchmark suite that was written for and tuned on a
MetaX C500 datacenter GPU, and is now being ported to run on a consumer RTX 4060 Laptop GPU.
The repo root is /home/shuhan/fusion. Study code lives in glm52/.

OLD platform (what the code was written for):
  MetaX C500, MACA 3.7, torch 2.8.0+metax, triton 3.0.0, backend 'maca', arch 80
  104 CUs | warp_size…",
      "lastProgressAt": 1785480074419,
      "tokens": 113772,
      "toolCalls": 31,
      "durationMs": 675469,
      "resultPreview": "{"module":"f10 Expert Merge + ResAdd — glm52/kernels/merge_resadd.py + glm52/bench/bench_f10_merge_resadd.py (+ glm52/bench/diag_f10_merge_resadd.py)\
\
KERNEL VERDICT: glm52/kernels/merge_resadd.py is CLEAN. No defect found on any axis. It is Triton-3.6 compatible as written: `do_not_specialize=[11]` still accepts integer indices (triton/runtime/jit.py:766 `i in do_not_specialize or param.name in…"
    },
    {
      "type": "workflow_agent",
      "index": 6,
      "label": "f11-lazy-prenorm",
      "phaseIndex": 1,
      "phaseTitle": "Audit",
      "agentId": "ad17015294bfd555d",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785479398950,
      "queuedAt": 1785479398890,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "f11 — #11b Lazy Pre-Norm → router GEMM (glm52/kernels/lazy_…",
      "promptPreview": "You are auditing a Triton kernel-fusion benchmark suite that was written for and tuned on a
MetaX C500 datacenter GPU, and is now being ported to run on a consumer RTX 4060 Laptop GPU.
The repo root is /home/shuhan/fusion. Study code lives in glm52/.

OLD platform (what the code was written for):
  MetaX C500, MACA 3.7, torch 2.8.0+metax, triton 3.0.0, backend 'maca', arch 80
  104 CUs | warp_size…",
      "lastProgressAt": 1785480257817,
      "tokens": 136695,
      "toolCalls": 32,
      "durationMs": 858867,
      "resultPreview": "{"module":"f11 — #11b Lazy Pre-Norm → router GEMM (glm52/kernels/lazy_prenorm.py, glm52/bench/bench_f11_lazy_prenorm.py, glm52/bench/bench_f11b_sweep.py, + glm52/analyze_f11b_arch.py)","findings":[{"severity":"blocker","category":"memory-footprint","file":"glm52/bench/bench_f11_lazy_prenorm.py","line":907,"problem":"main() unconditionally builds the F11a w13 weights before any regime runs: make_w1…"
    },
    {
      "type": "workflow_agent",
      "index": 7,
      "label": "synthesize-plan",
      "phaseIndex": 2,
      "phaseTitle": "Synthesize",
      "agentId": "ad021fbe457743cc2",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785480263408,
      "queuedAt": 1785480257834,
      "attempt": 1,
      "lastToolName": "Bash",
      "lastToolSummary": "P=/home/shuhan/miniconda3/lib/python3.13/site-packages/torc…",
      "promptPreview": "You are auditing a Triton kernel-fusion benchmark suite that was written for and tuned on a
MetaX C500 datacenter GPU, and is now being ported to run on a consumer RTX 4060 Laptop GPU.
The repo root is /home/shuhan/fusion. Study code lives in glm52/.

OLD platform (what the code was written for):
  MetaX C500, MACA 3.7, torch 2.8.0+metax, triton 3.0.0, backend 'maca', arch 80
  104 CUs | warp_size…",
      "lastProgressAt": 1785480754023,
      "tokens": 154558,
      "toolCalls": 27,
      "durationMs": 490615,
      "resultPreview": "I've verified every blocker and fairness claim against the actual files. Here is the consolidated plan.

---

# RTX 4060 Port Plan — verified, deduplicated, ordered

## 0. What I dropped, and why

| Dropped claim | Why |
|---|---|
| `common.py:37` `_FLUSH_BYTES = 128*2**20` hardcoded with stale "C500 L2 is 8 MB" comment | **Already fixed.** `common.py` was edited at 14:34 (rest of repo 14:12; `git…"
    }
  ],
  "totalTokens": 816034,
  "totalToolCalls": 195
}