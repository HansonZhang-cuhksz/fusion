# LOG-02 — Fusion #3: Residual Add + RMSNorm (`fused_add_rmsnorm`)

**Date:** 2026-07-27 · **GPU:** MetaX C500, device 3 (exclusive) · **Stack:** MACA,
`torch 2.8.0+metax3.7.1.3`, `triton 3.0.0+metax` · **Result id:** `f03_resadd_rmsnorm`

Deliverables
* kernel  — `glm52/kernels/add_rmsnorm.py`
* bench   — `glm52/bench/bench_f03_resadd_rmsnorm.py`
* results — `results/f03_resadd_rmsnorm.json` (full config tables for every variant ×
  every regime, coarse + refine + the joint chain search)

---

## 1. What is being fused

sglang's `fused_add_rmsnorm` (`sglang/jit_kernel/norm.py` →
`elementwise/fused_add_rmsnorm.cuh`; the torch-level wrapper is
`sglang/srt/layers/layernorm.py`). Semantics, matched exactly against
`glm52.reference.add_rmsnorm`:

```
h1 = (x.float() + residual.float()).to(bf16)                  # NEW residual — written out
x2 = ((h1.float() * rsqrt(mean(h1²)+eps)).to(bf16).float() * w).to(bf16)   # written out
```

Both outputs are live: `h1` is the residual the next block adds into, `x2` feeds the
router / MoE. **Neither variant is allowed to skip either output**, and neither does.

Shapes: `[T, 6144]` bf16, `T ∈ {1, 32, 256, 2048, 8192}`, `eps = 1e-5`, sum-of-squares
accumulated in fp32.

## 2. Memory-traffic analysis — the entire content of this fusion

The op is pure streaming; there is no arithmetic intensity to speak of
(≈3 flop/byte). One row = 6144 × 2 B = **12 288 B**.

| | passes over the row | bytes / row |
|---|---|---|
| **unfused** `add` kernel: read `x`, read `residual`, write `h1` | 3 | 36 864 |
| **unfused** `rmsnorm` kernel: read `h1`, write `x2` | 2 | 24 576 |
| **unfused total** | **5** | **61 440** |
| **fused**: read `x`, read `residual`, write `h1`, write `x2` | **4** | **49 152** |

**Ceiling = 5/4 = 1.25×**, achieved only if both sides are perfectly bandwidth-bound.
The saving is exactly the `h1` write→read round trip that the fused kernel keeps in
registers (it still *writes* `h1`, it just never reads it back).

| regime | T | bytes fused | bytes unfused | ideal fused @1.05 TB/s | ideal unfused |
|---|---|---|---|---|---|
| decode_bs1 | 1 | 49 KB | 61 KB | 0.05 µs | 0.06 µs |
| decode_bs32 | 32 | 1.57 MB | 1.97 MB | 1.5 µs | 1.9 µs |
| decode_bs256 | 256 | 12.6 MB | 15.7 MB | 12.0 µs | 15.0 µs |
| prefill_t2048 | 2048 | 100.7 MB | 125.8 MB | 95.9 µs | 119.8 µs |
| prefill_t8192 | 8192 | 402.7 MB | 503.3 MB | 383.5 µs | 479.3 µs |

Two effects are predicted to eat into the 1.25× at small T:

1. **L2.** C500 has 8 MB of L2. At T ≤ 256 one tensor is ≤ 3.1 MB, so in the *unfused*
   chain `h1` is still resident in L2 when the norm kernel reads it (the harness flushes
   L2 once before the chain, never between its kernels — `common.bench_chain`). The 5th
   pass is then an L2 hit, not HBM traffic, and the traffic argument for the fusion
   evaporates.
2. **Fixed overheads.** Measured on this box (see §6): a null kernel timed through this
   harness costs **13.8 µs**, and each *additional* kernel in a chain adds **≈ 2.8 µs**.
   At T=1 the whole tensor is 12 KB, so 100 % of the measurement is overhead and the
   fusion's benefit is "one launch instead of two", not bandwidth.

## 3. Implementation — one source, flags differ

`glm52/kernels/add_rmsnorm.py` has exactly one Triton kernel, `add_rmsnorm_kernel`, with
two behaviour flags:

| variant | `DO_ADD` | `DO_NORM` | traffic |
|---|---|---|---|
| fused | True | True | read x, res → write h1, x2 |
| unfused #1 (`add_only`) | True | False | read x, res → write h1 |
| unfused #2 (`norm_only`) | False | True | read h1 → write x2 |

`DO_NORM=False` compiles the reduction and the second pass away; `DO_ADD=False` compiles
the residual load and the `h1` store away. Nothing else differs — the unfused side runs
*the same instructions* on the same data layout, just split.

Mapping knobs (the only legal difference between the sides), all `tl.constexpr`:

* `BLOCK_N` — tile width over the hidden dim. 6144 = 3·2048 is **not** a power of two, and
  `tl.arange` requires one, so the two strategies asked for are both implemented and both
  tuned:
  * `BLOCK_N ≥ 6144` (i.e. 8192) → **one-shot**: padded power-of-two tile + column mask,
    the whole row stays in registers, one load feeds both the reduction and the
    normalize. 25 % of the lanes are masked off (no wasted memory traffic, wasted ALU).
  * `BLOCK_N ∈ {512,1024,2048,4096}` → **multi-pass**: `tl.static_range` loop of
    `N_TILES = 6144/BLOCK_N` tiles (exact, no masking at 512/1024/2048) accumulating the
    sum of squares, then a second loop that re-reads the row — from L2, not HBM — and
    normalizes. This is exactly the shape inductor generates (§7).
* `ROWS` — rows per program (1/2/4/8), i.e. a 2-D `[ROWS, BLOCK_N]` tile.
* `PERSISTENT` — off: grid = one program per row-block, **no outer loop at all**; on:
  capped grid (104/208/416/832/1664 programs) striding over row-blocks.
* `EVICT` — `evict_first` on the streaming loads, `evict_last` on the norm weight.
* `num_warps ∈ {1,2,4,8,16}` (warp = 64 lanes here, so 16 warps = 1024 threads = ceiling),
  `num_stages ∈ {1,2,3,4}`.

No shared memory is used explicitly, so the 64 KB SMEM ceiling never binds; the pre-filter
is instead on **register footprint**: `2 ≤ ROWS·BLOCK_N / (64·num_warps) ≤ 64` elements per
thread and `ROWS·BLOCK_N ≤ 65536`. Zero configs failed to compile in the whole search
(0 failures out of 152 coarse × 3 variants × 5 regimes), which is what that filter is for.

## 4. Tuning protocol

Two-stage, **run separately for the fused kernel and for each of the two unfused
kernels**, at **every one of the 5 regimes** (no config is ever carried across variants or
across regimes):

* **coarse** — 152 configs: `BLOCK_N ∈ {512,1024,2048,4096,8192} × ROWS ∈ {1,2,4,8} ×
  num_warps ∈ {1,2,4,8,16} × num_stages ∈ {1,2}`, filtered as above, non-persistent.
* **refine** — 20–88 neighbours of the coarse winner: ±1 step in `BLOCK_N`, `ROWS`,
  `num_warps`; the full `num_stages ∈ {1,2,3,4}` sweep at the winning shape; the five
  persistent-grid caps; and an `EVICT=1` twin of every one of those.

The unfused side then gets **more** tuning than the fused side, deliberately: a **joint
chain search** over the cross product of the top-2 coarse × top-2 refine configs of the
`add` kernel and of the `norm` kernel, timed as the real 2-kernel chain (one L2 flush
before the pair, none between). This catches the case where the best isolated `norm`
config is not the best config given that `h1` is L2-warm from the `add` kernel — and it
does happen: the joint chain is consistently ~10–25 % faster than the sum of the two
independently-timed kernels.

Timing: `common.bench_chain`, p50 of 120–400 reps after 30–100 warmups, one L2 flush
(128 MB `zero_`) before each rep of the whole chain.
