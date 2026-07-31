"""Wide re-tune of fusion #9 at specific batch sizes, for points the sweep may have
under-tuned.

The sweep's `_tile_then_coord` sweeps (BM,BN,BK) with num_warps/num_stages held at the seed,
then refines those greedily. That can miss a tile whose optimum lives at a different
(warps, stages) — which is what appears to have happened at T=1024: the sweep found
6.97 ms for the fused side while an independently tuned layer run found 6.24 ms for the same
kernel at the same size.

This does the full cross: every (BM,BN,BK) x (num_warps, num_stages) x GROUP_M, SMEM-filtered,
for BOTH sides equally. Slower, but it is the honest way to settle whether the tail of the
curve is real or a search artifact.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from glm52.bench.bench_f8_sweep import SEED_SUM, E, H, I  # noqa: E402
from glm52.bench.bench_f9_sweep import SEED_RA, Down9  # noqa: E402
from glm52.bench.bench_layer import _coord_tune, _smem_ok  # noqa: E402
from glm52.common import autotune, bench_chain, rel_err  # noqa: E402

RESULTS = ROOT / "results"


def wide_grid():
    out = []
    for bm, bn, bk, w, s, gm in itertools.product(
            (16, 32, 64, 128), (32, 64, 128, 256), (32, 64, 128),
            (4, 8, 16), (2, 3, 4), (1, 8)):
        if not _smem_ok(bm, bn, bk, s):
            continue
        out.append(dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm,
                        num_warps=w, num_stages=s))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    g = torch.Generator(device="cuda").manual_seed(0)
    w2 = torch.empty(E, H, I, device="cuda", dtype=torch.bfloat16)
    for i in range(E):
        w2[i].normal_(0, 0.02, generator=g)

    grid = wide_grid()
    print(f"wide grid: {len(grid)} configs per side", flush=True)
    out_rows = []
    for T in [int(x) for x in args.batches.split(",")]:
        p = Down9(T, w2, g)
        print(f"\n--- T={T} rows={p.rows} ({p.rows/E:.1f}/expert) ---", flush=True)

        tu = autotune(lambda c: p.unfused2(c, SEED_SUM)[:1], grid, warmup=4, rep=12)
        print(f"  unfused GEMM best {tu.best_ms:.4f} ({tu.n_tried - tu.n_failed} ok) "
              f"{tu.best_cfg}", flush=True)
        cs = _coord_tune(f"moe_sum(res=1) T{T}", lambda c: [p.unfused2(tu.best_cfg, c)[1]],
                         dict(SEED_SUM), rounds=1)
        tf = autotune(lambda c: p.fused9(c, {"impl": "torch"}), grid, warmup=4, rep=12)
        print(f"  fused   best {tf.best_ms:.4f} ({tf.n_tried - tf.n_failed} ok) "
              f"{tf.best_cfg}", flush=True)

        idx, ref = p.reference()
        ref_full = ref + p.h1[idx].float()
        p.out.zero_()
        for fn in p.unfused2(tu.best_cfg, cs):
            fn()
        torch.cuda.synchronize()
        err_u = rel_err(p.out[idx], ref_full)
        p.out.zero_()
        for fn in p.fused9(tf.best_cfg, {"impl": "torch"}):
            fn()
        torch.cuda.synchronize()
        err_f = rel_err(p.out[idx], ref_full)

        us, fs, gains = [], [], []
        for _ in range(5):
            a = bench_chain(p.unfused2(tu.best_cfg, cs), warmup=8, rep=25).p50_ms
            b = bench_chain(p.fused9(tf.best_cfg, {"impl": "torch"}), warmup=8, rep=25).p50_ms
            us.append(a); fs.append(b); gains.append(a / b)
        us_s, fs_s, g_s = sorted(us), sorted(fs), sorted(gains)
        row = dict(T=T, rows=p.rows, rows_per_expert=p.rows / E,
                   unfused2_ms=us_s[2], fused_ms=fs_s[2], gain_vs_2k=g_s[2],
                   gain_vs_2k_min=g_s[0], gain_vs_2k_max=g_s[-1],
                   gain_spread_pct=(g_s[-1] - g_s[0]) / g_s[2] * 100,
                   rounds_unfused=us, rounds_fused=fs, rounds_gain=gains,
                   unfused_cfg=tu.best_cfg, moe_sum_cfg=cs, fused_cfg=tf.best_cfg,
                   rel_err_unfused=err_u, rel_err_fused=err_f,
                   accum_MB=T * H * 2 / 2**20, n_cfgs_per_side=len(grid))
        out_rows.append(row)
        print(f"  T={T}: unfused {us_s[2]:.4f}  fused {fs_s[2]:.4f}  gain {g_s[2]:.4f}x",
              flush=True)
        del p
        torch.cuda.empty_cache()

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"f9_sweep_{args.tag}.json").write_text(
        json.dumps({"id": f"f9_sweep_{args.tag}",
                    "note": "WIDE re-tune: full (tile x warps x stages x group) cross, "
                            "both sides equally; supersedes the same T in the lane sweeps",
                    "rows": out_rows}, indent=2))
    print(f"\nwrote results/f8_sweep_{args.tag}.json")


if __name__ == "__main__":
    main()
