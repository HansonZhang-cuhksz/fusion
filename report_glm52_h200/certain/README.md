# H200, uncertainty-controlled re-measurement (2026-08-13)

A single 428-second run of `glm52_h200/bench/bench_layer_certain.py` on **one card**
(`GPU-338e7fe0`), all eleven regimes, **configurations frozen** from the campaign's own
`results/h200/layer_configurations.json`. Nothing is tuned here, so no arm can be tuned
unequally.

The two figures in the parent directory (`best_chain_vs_T.png`, `chain_gain_vs_T.png`) are
now built from these tables; the campaign versions they replaced are preserved as
`*_campaign.png`. The campaign's own CSVs are untouched. The two measurements agree on the
physics and disagree on how much of it was resolvable.

| | campaign (2026-08-07/08-10) | this run |
|---|---|---|
| cards | **two** (`decode_bs1` on one, `bs2/4/8/16` on another) | **one**, all regimes |
| tuning | re-tuned per regime, baseline included | frozen, identical across arms |
| statistic | ratio of pass medians, round-spread tie rule | median of per-block paired ratios, **bootstrap 95 % CI** |
| unique winner resolved | 3 / 11 | **6 / 11** |
| unfused inside the tie set | 3 / 11 | **0 / 11** |
| timings per configuration | 1 (wall) | 3 (wall, CUDA-graph, CUPTI per-kernel) |
| wall time | hours | 428 s |

## Results

| T | winner | sep | tied | gain | 95 % CI | campaign | blocks kept |
|---:|---|:--:|--:|--:|---|--:|---|
| 1 | `#1+#6+#9` | ✔ | 1 | 1.0693 | 1.0679 – 1.0709 | 1.2886 | 106/108 |
| 2 | `#3+#9` | | 2 | 1.0521 | 1.0505 – 1.0529 | 1.0420 | 112/112 |
| 4 | `#3+#10` | ✔ | 1 | 1.0524 | 1.0515 – 1.0532 | 1.0503 | 112/112 |
| 8 | `#1+#6+#9` | ✔ | 1 | 1.0288 | 1.0284 – 1.0293 | 1.0270 | 200/200 |
| 16 | `#1+#6+#9` | ✔ | 1 | 1.0179 | 1.0176 – 1.0182 | 1.0185 | 84/84 |
| 32 | `#1+#6+#9` | | 3 | 1.0098 | 1.0093 – 1.0100 | 1.0079 | 112/112 |
| 256 | `#1+#6+#9` | ✔ | 1 | 1.0156 | 1.0154 – 1.0160 | 1.0122 | 84/84 |
| 512 | `#1+#6+#9` | ✔ | 1 | 1.0136 | 1.0132 – 1.0139 | 1.0096 | 120/196 |
| 1024 | `#3+#10` | | 4 | 1.0038 | 1.0032 – 1.0048 | 0.9986 | 70/200 |
| 2048 | `#1` | | 3 | 1.0044 | 1.0032 – 1.0047 | 1.0061 | 21/168 |
| 8192 | `#3+#10` | | 2 | 1.0070 | 1.0055 – 1.0118 | 1.0081 | 46/200 |

Wall clock and CUDA-graph replay name the same winner at **8 of 11** regimes; the three
disagreements (T = 2, 32, 2048) are all regimes where wall reports a tie rather than a
unique winner, so neither contradicts the other.

## Two findings the campaign could not reach

**1. The decode gains are real work removal, not launch elimination.** The campaign's
calibration put a kernel launch at 10.3 µs and read the small-T wins as a
"launch-elimination signature". Measured directly, launch cost is **0.2–3.1 % of the layer**
and about **1.1 µs per launch**, an order of magnitude below the calibrated figure. At
`decode_bs4` the all-unfused layer decomposes as 783.8 µs of device work + 8.0 µs of
in-graph gap + 15.4 µs of launch, and fusing `#3` removes **33.5 µs of work against 2.0 µs
of launch**. The win is the norm kernel's work disappearing, not its launch.

**2. Fewer kernels is not less work.** At `decode_bs4`, `J_greedy_all` runs 11 kernels to
`H_f3_f10`'s 12, yet does *more* device work (759.1 µs vs 747.3 µs). Greedy stacking buys
launches it does not need at the cost of work it does.

Also confirmed: `#3` realises about **49 %** of its isolated kernel-level saving once
assembled into the layer (33.5 µs of 68.9 µs at T=4), independently reproducing the
realisation gap the campaign could show but not explain.

## Caveats — read before quoting anything

- **Clocks were not locked.** `nvidia-smi -lgc` needs root and it was unavailable, so the
  card ran unlocked with `SwPowerCap` active under load. The per-block drift gate is then
  the only defence: it discarded **39–88 % of blocks at T ≥ 512**, and the CIs there are
  3–10× wider than at mid-decode (±0.43 % at `prefill_t8192` against ±0.02 % at
  `decode_bs16`). **`prefill_t2048` kept only 21 of 168 blocks** — treat its 1.0044 as the
  weakest number in the table.
- **`decode_bs1` disagrees with the campaign by design, not by error.** The campaign's T=1
  all-unfused baseline — the denominator of that whole column — carried a cold-start
  excursion (round 0 at 2.5× the median of the other seven), and the campaign itself
  declared that regime TIED with nothing resolving. 1.0693 supersedes 1.2886.
- **The four `#11a`-bearing configurations fail the fp32 reference** at every regime except
  `decode_bs1` (rel_err 0.06–0.61 against a 5e-2 bar) and are excluded by the harness — 18
  configurations attempted per regime, 14 measured. This reproduces the campaign's own
  finding that `#11a` is unmeasurable on this device.
- Differences smaller than a chain's own CI are not results.

## Provenance: a harness bug was found and fixed between two runs

The **first** run of this harness (2026-08-13 05:53, 60 min) is **superseded and must not be
used**. It pinned the all-unfused baseline to slot 0 of every block, where it alone paid a
per-block entry cost, inflating it by up to +8.5 % and therefore inflating every
speedup — the file reported 13/13 configurations beating unfused at `decode_bs4`. Every
*other* configuration in that run reproduced the campaign to ≤ 0.36 %, which is how the bug
was caught.

The harness now rotates every configuration through every slot, discards a warm-up run at
each block head, and uses an in-band zero-subprocess drift probe (nvidia-smi calls dropped
from ~8800 to ~179, which is why 60 min became 7). Verified in this run: baseline excess
**0.01–0.27 %** at T = 4..1024, decomposition identity `work ≤ graph ≤ wall` holds with
**0 violations in all 11 regimes**, and **0 blocks measured-then-lost**.

## Files

| file | what |
|---|---|
| `layer_certain_per_regime.csv` | 158 rows — per regime × configuration: wall/graph time, work/gap/launch, speedup + CI, verdict, and the delta against the campaign |
| `layer_certain_verdicts.csv` | 11 rows — winner, tie set, block retention, CI achieved, stop reason |
| *(figures)* | the two canonical figures built from these tables live one level up: `../chain_gain_vs_T.png` (nine chains) and `../best_chain_vs_T.png` (best chain per regime). The campaign versions they replaced are kept as `../*_campaign.png`. |

Regenerate:

```bash
python3 glm52/make_certain_report_h200.py                       # tables (no matplotlib needed)
~/my-envs/fusion/bin/python glm52/make_chain_gain_vs_T_certain_h200.py        # ../chain_gain_vs_T.png
~/my-envs/fusion/bin/python glm52/make_best_chain_vs_T_certain_h200.py   # ../best_chain_vs_T.png
```

Source: `results/h200/layer_certain.json`.
