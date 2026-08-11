# `results/h200/_control_arm/` — staging root for the Hopper control arm

**Status: AWAITING DATA.** This README was committed *before* any measurement. If this tree
still holds nothing but this file, the operator has not run yet.

## The invariant

**Nothing in this tree is ever merged into `results/h200/*.json`. The control arm is a DIFF,
not an append. There is no merge step and no script in this repo may add one.**

The campaign files one level up (`f01_oproj_resadd.json` … `f11_lazy_prenorm.json`,
`summary.json`) **are the baseline this arm is compared against**. A bench writes its ENTIRE
result file on `record()`, so any script that pointed a control-arm run at `results/h200/`
would not append — it would overwrite the campaign and destroy the comparison itself.
Read `log/LOG-18-hopper-control-arm.md` §7 before touching anything here.

## Expected layout

    _control_arm/
      README.md                                 <- this file
      control_arm_summary.json                  driver summary: gpu, campaign fingerprint,
                                                 per-arm status, cells, warnings, exit code
      <arm>/                                    arm = classic (default) | hopper | no-tma |
                                                       no-ws | no-clusters
        ARM_NOT_VERIFIED                        sentinel; present until every family in the arm
                                                 passes engagement. Its presence makes
                                                 make_control_report_h200.py refuse to publish.
        <RESULT_ID>.json                        one per family, same filenames as the campaign
                                                 (f03_resadd_rmsnorm.json, f10_merge_resadd.json,
                                                  f01_oproj_resadd.json, f04f05_norm_router.json,
                                                  f11_lazy_prenorm.json, f06_upgate_swiglu.json,
                                                  f08f09_down_merge_resadd.json)
        _ckpt/<family>/<regime>.json            per-regime checkpoints; `CKPT_ROOT` follows
                                                 `GLM52_H200_RESULTS_DIR` here on purpose, so a
                                                 classic run cannot replay Hopper checkpoints
      _engagement/<arm>/<family>.verify.json    the engagement audit trail: per-axis verdicts
                                                 (PASS / VACUOUS / FAIL / NOTHING-TO-DISABLE)

## Who writes here

`run_control_h200.py`, and nothing else. Every write it makes passes `guard_write()`, which
resolves the path and fatally refuses anything outside this directory; a campaign canary
(size/mtime/sha256 over every non-underscore `*.json` directly in `results/h200/`) re-checks
the baseline after every family and exits 6 on any change.

## What the leading underscore does and does NOT protect

It protects against `run_h200.find_result` only — that globs non-recursively and skips names
starting with `_` (`run_h200.py:1060-1071`).

It does **not** protect against `run_h200.quarantine_foreign_results`, which uses
`results.rglob("*.json")` and skips only `_quarantine_foreign_*` path parts
(`run_h200.py:993-997`). That sweep judges each file on `payload["_meta"]["device"]`
(`run_h200.py:1004-1005`). The 7 per-family result JSONs carry that key and record
`NVIDIA H200`, so they survive. **The checkpoints and the `.verify.json` files do not:**
`common.py:ckpt_save` puts `device` at top level with no `_meta`, so it resolves to `""` and
**every `_ckpt/**/*.json` here WILL BE MOVED** into `results/h200/_quarantine_foreign_<ts>/`.
Six such directories already exist one level up and their contents are exactly that.

**So: do not run `run_h200.py` on the benchmark host while this tree is staged.** Tarball this
directory back first. `run_control_h200.py` never calls the sweep; that, plus this instruction,
is the whole fence.

Nothing here is ever deleted, including failed and partial arms: a control arm that did not
engage is evidence, not garbage.
