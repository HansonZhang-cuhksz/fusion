# report_glm52_h200/control_arm — AWAITING DATA

**Nothing has been measured.** This directory is a committed placeholder. When the operator's
control-arm tarball comes back, `glm52/make_control_report_h200.py` **overwrites this file
wholesale** along with the CSVs beside it. If you are reading this text, the run has not
happened (or its output has not been generated locally yet).

## What the control arm is

A rerun of the same H200 benchmarks on the same card with `GLM52_H200_CLASSIC=1`, which forces
the sm_90 levers off — TMA, warp specialization via `tl.range(warp_specialize=True)`,
thread-block clusters via `num_ctas`, and `wgmma`. It exists to test whether those features
explain any of the H200-vs-C500 pattern in `../README.md`. The design, the noise floor it is
judged against, and what it can and cannot say are in `log/LOG-18-hopper-control-arm.md`.

## Why this is a subdirectory of `report_glm52_h200/` and not a sibling

`report_glm52_<device>/` is a **device namespace**: its members carry the same CSV field names
in the same order so the directories diff row-for-row against each other
(`report_glm52_h200/README.md:3-5`). A sibling `report_glm52_h200_classic/` would claim to be a
fourth *device*, which it is not — the control arm is a **second reading of the same device**,
and a row-for-row diff against `report_glm52_c500/` or `report_glm52_rtx4060/` would be
meaningless for it.

Precedent inside this same namespace: `log/LOG-17-campaign-v2.md:317-318` proposed archiving
the campaign-1 tables under a `_campaign1_20260805/` **directory in `report_glm52_h200/`** so
that both the old and the new reading of this device stayed searchable together. Same shape,
same reason. The dated `## 0a.` / `## 0b.` / `## 0c.` preambles in `../README.md` are the other
half of the convention: every re-measurement of this device is absorbed by a preamble that says
where its numbers live.

## Where the numbers are, and are not

The control arm's numbers are **not** in `../fusion_*.csv` or `../layer_optimal_per_regime.csv`
and never will be. Those CSVs are the Hopper-path campaign. Everything produced by the control
arm lands here, and the comparison is published as a **delta** — never merged back. See
`../README.md` §0c.

## The command that fills this directory

With `results/h200/_control_arm/` in place (the operator's tarball unpacked), from the repo
root, no GPU needed:

    python3 glm52/make_control_report_h200.py

It refuses to publish an arm whose `ARM_NOT_VERIFIED` sentinel is still present, and refuses
outright if fewer than 16 of the 22 `f03`/`f10` noise-floor cells are usable.
