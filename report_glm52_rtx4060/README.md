# report_glm52_rtx4060 — RTX 4060 Laptop (sm89) fusion report

Counterpart to `report_glm52_c500/`. Same schema where the data supports it; **the differences
below are real and load-bearing, not formatting choices.**

**Device** NVIDIA GeForce RTX 4060 Laptop GPU, sm_89, 24 SM, warp 32, 101376 B per-block opt-in
SMEM, 65536 regs/SM, 32 MB L2, 8 GB VRAM. Clocks **locked**: SM 1020 MHz (of 3105), MEM
5501 MHz (of 8001). torch 2.11.0+cu130, triton 3.6.0.
**Calibrated** 140 GB/s streaming r+w, 159 GB/s read-only, Triton bf16 GEMM 11.81 TF/s
(= 102 % of cuBLAS; on C500 Triton reached only 50 % of the vendor BLAS).

Source data `results/rtx4060/`. Port and audit log `log/LOG-13-rtx4060-port.md`.

## Files

| file | status |
|---|---|
| `fusion_decode_bs1.csv`, `..._bs32`, `..._bs256`, `..._prefill_t2048`, `..._t8192` | measured |
| `layer_optimal_per_regime.csv` | **partly derived — read §2** |

## 1. Five regimes, not seven; six fusions, not eleven

**Regimes.** C500 has `decode_bs512` and `decode_bs1024`; those were whole-layer regimes and
are absent here for the reason in §2.

**Fusions.** `#6`, `#8`, `#9` and `#11a` are **not present**. They need the 256-expert weights —
`w13` is **12.0 GB** and `w2` **6.0 GB** — against ~7.4 GB usable. There is no arrangement that
fits. Rather than shrink the expert count (which preserves tile shapes but multiplies per-expert
row counts and distorts the grouped-GEMM tiling), they were dropped, so **everything reported
here is at exact GLM-5.2 spec with no deviation**.

## 2. `layer_optimal_per_regime.csv` is NOT the same measurement as C500's

C500's `run1_ms` / `run2_ms` / `best_ms` are **whole-layer times** from `bench_layer.py`, over
fusion sets including #6/#8/#9. None of that is reproducible here:

- a full MoE layer needs **18.0 GB** of expert weights against ~7.4 GB usable, so **the layer
  cannot be instantiated at exact spec on this device at all**;
- #6/#8/#9 were therefore never measured;
- `bench_layer.py` was never run — it still carries C500 constants (SMEM 65536, `* 64` warp
  arithmetic) and would produce C500-capped grids here.

So `layer_total_ms` and `speedup_vs_unfused` are **left empty**, and `layer_measurable` is
`FALSE`. They are not modelled, because modelling them would put fabricated numbers under
column names that mean "measured" in the C500 file.

What *is* reported is `ms_saved_per_layer` — the absolute time each set removes from a layer,
which follows from measured per-call deltas and needs no layer total. Read the `basis` column:

- `measured` — a single fusion, straight from its measured delta × its sites per layer
  (#3 runs twice per layer, #10 and #1 once).
- `additive estimate` — a set. **#1 and #3 compete for ResAdd1**, and **#10 and #3 compete for
  ResAdd2**, so these are sums of independently measured deltas, not measured combinations.
  C500 measured combinations end-to-end precisely because they do *not* compose additively
  (LOG-11 §6). **Ranks are indicative, not measured.**
- `structural (0 by shared-producer argument)` — #4/#5/#11b fuse the post-attention RMSNorm into
  the router, but that norm also feeds the w13 grouped GEMM and the shared expert, so fusing it
  into the router does not remove it. Worth ~0 per layer unless every K=6144 consumer is fused
  (#11a), which needs 12 GB of w13. Scored 0.0 rather than by their standalone chain speedups.

## 3. Numbers that are corrected relative to the raw campaign

- **#1 at `prefill_t8192` is 1.0143, not the campaign's 1.0267.** The campaign value exceeds the
  cell's physical ceiling; that run drifted 22 % thermally (coarse 137.25 ms vs final 167.20 ms)
  and times the fused arm entirely before the unfused. The value here is an interleaved A/B
  re-measurement (n=120, paired p50). A primitive decomposition at the shared config agrees:
  `gemm(FUSE=True)` 134.969 ms vs `gemm(FUSE=False)` 135.050 ms + epilogue 1.956 ms → 1.0151.
- **`decode_bs1` / `decode_bs32` are launch-dominated and coarsely quantised.** Measured launch
  cost is 3.36 µs against a 4.10 µs gap, and this device's CUDA-event timer ticks at **1.024 µs**
  (200/200 sampled timings are exact multiples; C500's tick is 0.256 µs). Those rows are 9–17
  ticks, so the ratios resolve to roughly ±8 %.
- **#4/#5 decode speedups are optimistic.** `bench_f04f05` under-tunes the unfused router GEMM
  (`BLOCK_K ∈ {32,64}`, no GROUP_M axis, no refine) where `bench_f11` tunes the byte-identical
  op to 0.0522 ms vs 0.1116 ms at bs1. The true decode figures are **worse** than shown. This
  does not affect `prefill_t8192` (1.011× agreement between the two benches).

## 4. Reproducing

```bash
export GLM52_RESULTS_DIR=/home/shuhan/fusion/results/rtx4060
./run_4060.sh                                   # serialized; ONE GPU on this host
python3 glm52/make_report_rtx4060.py            # per-regime CSVs
python3 glm52/make_layer_report_rtx4060.py      # layer_optimal_per_regime.csv
```
