"""Fusion #9 (Down GEMM + Expert Merge + ResAdd2) — kernel-level gain vs decode batch size.

Same protocol as the #8 sweep: both sides independently retuned at every batch size, each
point the median of 5 interleaved rounds, both sides validated against an fp32 reference.

#9's unfused side has TWO legitimate forms, and the choice is worth up to 8x in the reported
gain (LOG-08 F4 caught it inflating a result), so both are measured and both are reported:

    UNFUSED-3K : down GEMM -> moe_sum(ADD_RESIDUAL=False) -> separate resadd
                 materialises an extra [T,H] tensor nothing downstream needs
    UNFUSED-2K : down GEMM -> moe_sum(ADD_RESIDUAL=True)
                 strictly better; the comparison a competent implementation actually faces

    FUSED      : seed the output with the residual, then the down GEMM with sglang's
                 FUSE_SUM_ALL_REDUCE atomic-accumulate epilogue. The seed IS inside the
                 fused timing, and is taken as the better of a torch `.copy_()` and a tuned
                 Triton seed kernel — giving the fused side its own best option.

Run:  CUDA_VISIBLE_DEVICES=<n> python glm52/bench/bench_f9_sweep.py --batches 1,32 --tag lane0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from glm52.bench.bench_f8_sweep import SEED_SUM, SEED_W2, DownProblem, E, H, TOPK  # noqa: E402
from glm52.bench.bench_layer import _coord_tune, _smem_ok, _tile_then_coord  # noqa: E402
from glm52.common import bench_chain, rel_err  # noqa: E402
from glm52.kernels import moe_down_merge as KD  # noqa: E402

RESULTS = ROOT / "results"
SEED_RA = dict(BLOCK_M=2, BLOCK_N=256, num_warps=4, num_stages=2)


def rep_budget(T: int) -> tuple[int, int, int, int]:
    """Scale measurement effort with problem size.

    A single chain launch costs ~0.15 ms at T=32 but ~1.7 s at T=262144, so a fixed rep
    count would make the largest points cost ~80 min each while measuring them far more
    precisely than needed: timing noise on a 1.7 s kernel is proportionally tiny. Returns
    (tune_warmup, tune_rep, measure_warmup, measure_rep).
    """
    if T <= 8192:
        return 4, 12, 8, 25
    if T <= 65536:
        return 2, 6, 4, 10
    return 1, 3, 2, 5


class Down9(DownProblem):
    def __init__(self, T, w2, g):
        super().__init__(T, w2, g)
        self.out = torch.zeros(T, H, device="cuda", dtype=torch.bfloat16)

    # --- unfused, 3 kernels -------------------------------------------------------------
    def unfused3(self, cg, cs, cr):
        s, e, n = self.layout(cg["BLOCK_M"])
        return [lambda: KD.launch_down(self.act, self.w2, self.y3, self.tw_flat, s, e, n,
                                       self.rows, TOPK, cg, False),
                lambda: KD.launch_moe_sum(self.y3v, self.routed, self.h1, TOPK, cs, False),
                lambda: KD.launch_resadd(self.routed, self.h1, self.out, cr)]

    # --- unfused, 2 kernels (moe_sum folds the residual in) -----------------------------
    def unfused2(self, cg, cs):
        s, e, n = self.layout(cg["BLOCK_M"])
        return [lambda: KD.launch_down(self.act, self.w2, self.y3, self.tw_flat, s, e, n,
                                       self.rows, TOPK, cg, False),
                lambda: KD.launch_moe_sum(self.y3v, self.out, self.h1, TOPK, cs, True)]

    # --- fused: seed with the residual, then atomically accumulate ----------------------
    def fused9(self, cg, seed_cfg):
        s, e, n = self.layout(cg["BLOCK_M"])
        if seed_cfg.get("impl") == "torch":
            seed = lambda: self.out.copy_(self.h1)
        else:
            seed = lambda: KD.launch_seed(self.out, self.h1, seed_cfg, from_residual=True)
        return [seed,
                lambda: KD.launch_down(self.act, self.w2, self.out, self.tw_flat, s, e, n,
                                       self.rows, TOPK, cg, True)]


def sweep(batches: list[int], tag: str) -> None:
    g = torch.Generator(device="cuda").manual_seed(0)
    print(f"allocating w2 = {E*H*2048*2/2**30:.1f} GiB", flush=True)
    w2 = torch.empty(E, H, 2048, device="cuda", dtype=torch.bfloat16)
    for i in range(E):
        w2[i].normal_(0, 0.02, generator=g)

    guard = lambda c: _smem_ok(c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"], c["num_stages"])
    rows_out = []
    for T in batches:
        p = Down9(T, w2, g)
        tw, tr_, mw, mr = rep_budget(T)
        print(f"\n--- T={T} rows={p.rows} ({p.rows/E:.1f}/expert) "
              f"[tune {tw}/{tr_}, measure {mw}/{mr}] ---", flush=True)

        cg_u = _tile_then_coord(f"w2 unfused T{T}", lambda c: p.unfused2(c, SEED_SUM)[:1],
                                dict(SEED_W2), guard=guard, warmup=tw, rep=tr_)
        cs_f = _coord_tune(f"moe_sum(res=0) T{T}",
                           lambda c: [p.unfused3(cg_u, c, SEED_RA)[1]], dict(SEED_SUM), rounds=1, warmup=tw, rep=tr_)
        cs_t = _coord_tune(f"moe_sum(res=1) T{T}",
                           lambda c: [p.unfused2(cg_u, c)[1]], dict(SEED_SUM), rounds=1, warmup=tw, rep=tr_)
        cr = _coord_tune(f"resadd T{T}", lambda c: [p.unfused3(cg_u, cs_f, c)[2]],
                         dict(SEED_RA), rounds=1, warmup=tw, rep=tr_)
        cg_f = _tile_then_coord(f"w2 fused T{T}",
                                lambda c: p.fused9(c, {"impl": "torch"}), dict(SEED_W2),
                                guard=guard, warmup=tw, rep=tr_)
        # give the fused side its own best seed: torch copy vs a tuned Triton seed kernel
        seed_tri = _coord_tune(f"seed T{T}",
                               lambda c: [lambda: KD.launch_seed(p.out, p.h1, c,
                                                                 from_residual=True)],
                               dict(SEED_RA), rounds=1, warmup=tw, rep=tr_)
        t_torch = bench_chain([lambda: p.out.copy_(p.h1)], warmup=mw, rep=mr).p50_ms
        t_tri = bench_chain([lambda: KD.launch_seed(p.out, p.h1, seed_tri, from_residual=True)],
                            warmup=mw, rep=mr).p50_ms
        seed_cfg = {"impl": "torch"} if t_torch <= t_tri else seed_tri
        print(f"  [cfgs] seed            torch {t_torch:.4f} vs triton {t_tri:.4f} -> "
              f"{'torch' if t_torch <= t_tri else 'triton'}", flush=True)

        # correctness: every variant must equal (top-k weighted sum) + residual
        idx, ref = p.reference()
        ref_full = ref + p.h1[idx].float()
        errs = {}
        for name, chain in (("fused", p.fused9(cg_f, seed_cfg)),
                            ("unfused2", p.unfused2(cg_u, cs_t)),
                            ("unfused3", p.unfused3(cg_u, cs_f, cr))):
            p.out.zero_()
            for fn in chain:
                fn()
            torch.cuda.synchronize()
            errs[name] = rel_err(p.out[idx], ref_full)
        assert max(errs.values()) < 5e-2, errs

        rounds = {k: [] for k in ("fused", "unfused2", "unfused3")}
        for _ in range(5):
            rounds["fused"].append(bench_chain(p.fused9(cg_f, seed_cfg), warmup=mw, rep=mr).p50_ms)
            rounds["unfused2"].append(bench_chain(p.unfused2(cg_u, cs_t), warmup=mw, rep=mr).p50_ms)
            rounds["unfused3"].append(bench_chain(p.unfused3(cg_u, cs_f, cr), warmup=mw,
                                                  rep=mr).p50_ms)
        med = {k: sorted(v)[2] for k, v in rounds.items()}
        g2 = sorted(u / f for u, f in zip(rounds["unfused2"], rounds["fused"]))
        g3 = sorted(u / f for u, f in zip(rounds["unfused3"], rounds["fused"]))

        row = dict(T=T, rows=p.rows, rows_per_expert=p.rows / E,
                   fused_ms=med["fused"], unfused2_ms=med["unfused2"],
                   unfused3_ms=med["unfused3"],
                   gain_vs_2k=g2[2], gain_vs_2k_min=g2[0], gain_vs_2k_max=g2[-1],
                   gain_vs_3k=g3[2], gain_vs_3k_min=g3[0], gain_vs_3k_max=g3[-1],
                   gain_spread_pct=(g2[-1] - g2[0]) / g2[2] * 100,
                   rounds=rounds, rel_err=errs,
                   unfused_cfg=cg_u, moe_sum_res0_cfg=cs_f, moe_sum_res1_cfg=cs_t,
                   resadd_cfg=cr, fused_cfg=cg_f, seed_cfg=seed_cfg,
                   accum_MB=T * H * 2 / 2**20)
        rows_out.append(row)
        print(f"  T={T:<5} fused {med['fused']:.4f} | 2k {med['unfused2']:.4f} "
              f"({g2[2]:.4f}x) | 3k {med['unfused3']:.4f} ({g3[2]:.4f}x)   "
              f"relerr {max(errs.values()):.1e}", flush=True)
        del p
        torch.cuda.empty_cache()

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"f9_sweep_{tag}.json").write_text(json.dumps(
        {"id": f"f9_sweep_{tag}",
         "fusion": "#9 Down GEMM + Expert Merge + ResAdd2 (atomic)",
         "note": "both sides retuned per point; two unfused baselines reported",
         "rows": rows_out}, indent=2))
    print(f"\nwrote results/f9_sweep_{tag}.json", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    sweep([int(x) for x in args.batches.split(",")], args.tag)


if __name__ == "__main__":
    main()
