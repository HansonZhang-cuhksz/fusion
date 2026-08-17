# Cross-device frozen-config comparison — C500 vs H200 at identical Triton mappings

Answering one question: **how much of the reported C500↔H200 fusion-gain difference is the
device, and how much is the unfair backend?** The C500 campaign ran `triton 3.0.0+metax`,
a young vendor fork that reaches ~50 % of its vendor BLAS on dense GEMMs and collapses a
good mainloop schedule to −23 % the moment any epilogue exists (LOG-10's codegen cliff);
the H200 ran upstream `triton 3.6.0` at 94.4 % of cuBLAS. Both campaigns also autotuned
independently, so each device's fused-vs-unfused *ratio* was computed at whatever mapping
its own tuner and its own compiler favored — the two confounds are inseparable in
`report_glm52_h200/report.md`'s tables.

## The approach

1. **Freeze the mapping universe.** Every cell here is a *pair of configs* — one for the
   fused arm, one for the unfused GEMM — with both configs read out of **both** devices'
   own coarse tuning tables (`results/*.json`, `results/h200/*.json`). A cell exists only
   if the identical (tile, warps, stages, GROUP_M[, SPLIT_K]) dict appears on both
   devices. The fused and unfused sides of the study's own one-source-kernel design stay
   intact; only the mapping space is equalized.
2. **Hopper features are excluded by construction.** A C500 config can never carry
   `USE_TMA`/`warp_specialize`/`num_ctas`, so any H200 config that does fails the match.
   The H200 side of every cell here is its plain-load `mma` path — nothing TMA/WS/clusters
   can do moves these numbers.
3. **Junk cells are pruned, not curated.** A pair is dropped when either of its arms sits
   >1.5× above that device's own best config *inside the shared grid* (mirroring the
   campaign's coarse→refine protocol), or when either fused time is pathological relative
   to its own chain and published fused time. Pruned rows stay in the CSV, flagged.
4. **Gain = chain_ms / fused_ms**, the study's own convention, where chain_ms = the
   unfused GEMM's table time at the *same* mapping + the device's separately tuned
   best companion kernel (f01: residual-add epilogue; f06: SwiGLU activation; f08f09:
   `moe_sum_with_residual`, i.e. the #9 chain).

Two views, one per file set:

| file | what a row is |
|---|---|
| `<fam>_frozen_cfg.csv` | **strict**: fused cfg == unfused cfg == same dict on both devices. The purest cell; small n (a config must clear four tables). |
| `<fam>_frozen_pairs.csv` | **pairs**: fused cfg and unfused cfg each drawn from the shared intersection, all clean combinations. Larger n; the "each device optimizes inside an equal universe" view. |

Generator: `python3 glm52/make_frozen_cfg_comparison.py [f01,f06,f08f09] [--pairs]`.
Reads only JSON; nothing here was re-measured.

## What fell out (p50 over clean cells; see the CSVs for the rest)

| family | regime | c500 gain p50 | h200 gain p50 | Δ (pp) |
|---|---|---|---|---|
| f01 [#1] | decode_bs1 | 1.044 | 1.051 | +0.8 |
| | decode_bs32 | 1.042 | 1.134 | +9.3 |
| | decode_bs256 | 1.019 | 1.030 | +1.1 |
| | prefill_t2048 | 1.021 | 1.185 | +18.3 |
| | prefill_t8192 | 1.000 | 1.030 | +15.0 |
| f06 [#6] | decode_bs1 | 1.188 | 1.321 | +19.8 |
| | decode_bs32 | 1.147 | 1.308 | +18.1 |
| | decode_bs256 | 0.885 | 1.268 | +17.9 |
| | prefill_t2048 | 0.538 | 1.021 | +36.0 |
| | prefill_t8192 | 0.763 | 1.270 | +49.6 |
| f08f09 [#9] | decode_bs1 | 1.045 | 1.284 | +29.8 |
| | decode_bs32 | 0.998 | 1.002 | +1.0 |
| | decode_bs256 | 0.997 | 1.013 | +2.4 |
| | prefill_t2048 | 0.869 | 0.951 | +9.8 |
| | prefill_t8192 | 0.884 | 0.954 | +10.2 |

Read #6@t2048 (±36 pp) as the bound case: at mappings **its own tuner could have picked**,
C500's fused UpGate+SwiGLU is 0.54–0.76×, H200's is 1.02–1.51×. And the strict view
f08f09 prefill reproduces the published levels almost exactly (C500 0.90 vs published
0.87–0.90; H200 0.98 vs 0.983) — a sanity check that the reconstruction is faithful where
the two protocols overlap.

## What compares to the published tables — the two claims that survive

1. **The prefill sign flips are not a tuning artifact.** Holding the mapping universe
   fixed removes the "each device picked its own winner" confound, and the H200's
   fused-GEMM edge over the C500 remains +8 to +50 pp at prefill in all three GEMM
   families (and the C500's #6 loss in particular is present at essentially every clean
   cell, not just at its published winner).
2. **At decode the devices mostly agree, except where they don't.** f01 and f08f09 agree
   to ~1–3 pp at bs32/bs256 once mappings are frozen; f01's decode_bs1 "loss" from the
   older tables dissolves (+0.8 to +1.1 pp here). What stays odd at decode is f06
   (+18–20 pp) and f08f09@bs1 (+24–30 pp): the narrow-tile MoE kernels' fusion cost
   genuinely differs between the backends even at fixed mappings.

One nuance the older reports got wrong at that already matters at f01/prefill: **the C500
#1 prefill loss (published 0.978) flips to +2.1 % at the frozen grid.** The published
number is the refine-tier winner's paired statistic; the coarse-tier frozen universe says
the fusion itself is neutral-to-positive there. Do not quote 0.978 as "the fusion costs".

## Limits, stated once

- **Same mapping ≠ same machine code.** 3.0.0+metax and 3.6.0 lower the same Triton
  source to different SASS-equivalents — this protocol equalizes the *mapping and tuning*
  inputs to that lowering; it cannot and does not equalize the compiler. A remaining
  delta is exactly the quantity that cannot be attributed further from this data.
- **Coarse-tier inflation.** The chain side is built from coarse-table times plus a
  separately-tuned companion; the published rows use refine/joint tiers. Absolute gains
  here run higher than published **on both devices** (the deltas are the robust column).
  f01 `!`-marked cells are SPLIT_K>1 rows whose zero-init+cast kernel is not priced.
- **Single-pass timestamps.** Coarse-table times are one-shot medians, not the paired
  interleaved statistic; a cell here is worth an order of magnitude less than the
  `certain/` re-measurement. This is a *structure* analysis over grids already paid for,
  not new evidence to precedence over LOG'd measurements.
- Scope: f01, f06, f08f09 (the GEMM-carrying families with matching tier structures on
  both devices). #3/#10 contain no GEMM mapping to freeze; #11's C500 tables use a
  different shape (`rows`-keyed) and are not included.

The natural follow-up is a live run: both devices' benches already accept explicit
configs, so a *measured* frozen-config pass (same kProtocol as `certain/`, configs pinned
from these intersections, on the C500 here and the H200 remotely) would upgrade every
cell here from "grid table" to "paired measurement". The config lists to pin exist in the
CSVs above; nothing else needs to be invented.