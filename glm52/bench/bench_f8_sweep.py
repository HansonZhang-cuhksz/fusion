"""Fusion #8 (Down GEMM + Expert Merge) — kernel-level gain vs decode batch size.

For each batch size T, BOTH sides are tuned independently at that T (joint (BM,BN,BK) sweep
plus coordinate refinement) and then timed with `bench_chain`:

    UNFUSED : w2 grouped down GEMM -> [rows, H]  ,  then moe_sum over top-8 -> [T, H]
    FUSED   : w2 grouped down GEMM with the sglang FUSE_SUM_ALL_REDUCE epilogue, atomically
              accumulating into [T, H]  (plus the output zero-init it requires, which IS
              inside the fused timing)

Re-tuning at every point is not optional here: the optimal BLOCK_M shifts with rows-per-expert
(T*8/256), so a fixed mapping would turn mapping staleness into apparent batch-size structure —
exactly the artifact that previously manufactured a fake 1.16x result for fusion #6.

Only the down-projection is allocated (w2, 6.4 GB); w13 is not needed, which keeps each
process small enough to run three regimes concurrently on separate GPUs.

Run:  CUDA_VISIBLE_DEVICES=<n> python glm52/bench/bench_f8_sweep.py --batches 1,32,256 --tag lane0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from glm52 import config as C  # noqa: E402
from glm52 import reference as R  # noqa: E402
from glm52.bench.bench_layer import _coord_tune, _smem_ok, _tile_then_coord  # noqa: E402
from glm52.common import bench_chain, rel_err  # noqa: E402
from glm52.kernels import moe_down_merge as KD  # noqa: E402

H, I, E, TOPK = C.HIDDEN_SIZE, C.MOE_INTERMEDIATE_SIZE, C.N_ROUTED_EXPERTS, C.NUM_EXPERTS_PER_TOK
RESULTS = ROOT / "results"

SEED_W2 = dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=1, num_warps=8, num_stages=3)
SEED_SUM = dict(BLOCK_M=2, BLOCK_DIM=512, num_warps=8, num_stages=1)


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


class DownProblem:
    def __init__(self, T: int, w2: torch.Tensor, g: torch.Generator):
        self.T, self.rows = T, T * TOPK
        self.w2 = w2
        dev, dt = "cuda", torch.bfloat16
        self.act = torch.empty(self.rows, I, device=dev, dtype=dt).normal_(0, 0.1, generator=g)
        self.h1 = torch.empty(T, H, device=dev, dtype=dt).normal_(0, 0.1, generator=g)
        self.y3 = torch.zeros(self.rows, H, device=dev, dtype=dt)
        self.y3v = self.y3.view(T, TOPK, H)
        self.routed = torch.zeros(T, H, device=dev, dtype=dt)
        self.topi = torch.randint(0, E, (T, TOPK), device=dev, dtype=torch.int32, generator=g)
        w = torch.rand(T, TOPK, device=dev, generator=g) + 0.1
        self.topw = (w / w.sum(-1, keepdim=True) * C.ROUTED_SCALING_FACTOR).float()
        self.tw_flat = self.topw.flatten().contiguous()
        self._lay: dict[int, tuple] = {}

    def layout(self, bm: int):
        if bm not in self._lay:
            self._lay[bm] = R.moe_align_block_size(self.topi, bm, E)
        return self._lay[bm]

    def unfused(self, cg, cs):
        s, e, n = self.layout(cg["BLOCK_M"])
        return [lambda: KD.launch_down(self.act, self.w2, self.y3, self.tw_flat, s, e, n,
                                       self.rows, TOPK, cg, False),
                lambda: KD.launch_moe_sum(self.y3v, self.routed, self.h1, TOPK, cs, False)]

    def fused(self, cg):
        s, e, n = self.layout(cg["BLOCK_M"])
        return [lambda: self.routed.zero_(),
                lambda: KD.launch_down(self.act, self.w2, self.routed, self.tw_flat, s, e, n,
                                       self.rows, TOPK, cg, True)]

    def reference(self, n_tokens: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 ground truth on a sample of tokens: out[t] = sum_k w_k * act[t,k] @ w2[e]^T"""
        idx = torch.arange(min(n_tokens, self.T), device="cuda")
        out = torch.zeros(len(idx), H, device="cuda", dtype=torch.float32)
        for j, t in enumerate(idx.tolist()):
            for k in range(TOPK):
                e_id = int(self.topi[t, k])
                a = self.act[t * TOPK + k].float()
                out[j] += (a @ self.w2[e_id].float().T) * float(self.topw[t, k])
        return idx, out


def sweep(batches: list[int], tag: str) -> dict:
    g = torch.Generator(device="cuda").manual_seed(0)
    print(f"allocating w2 [{E}, {H}, {I}] bf16 = {E*H*I*2/2**30:.1f} GiB", flush=True)
    w2 = torch.empty(E, H, I, device="cuda", dtype=torch.bfloat16)
    for i in range(E):
        w2[i].normal_(0, 0.02, generator=g)

    rows_out = []
    for T in batches:
        p = DownProblem(T, w2, g)
        print(f"\n--- T={T}  rows={p.rows}  ({p.rows/E:.1f} rows/expert) ---", flush=True)
        tw, tr_, mw, mr = rep_budget(T)
        guard = lambda c: _smem_ok(c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"], c["num_stages"])

        cg_u = _tile_then_coord(f"w2 unfused T{T}",
                                lambda c: p.unfused(c, SEED_SUM)[:1], dict(SEED_W2), guard=guard, warmup=tw, rep=tr_)
        cs = _coord_tune(f"moe_sum T{T}",
                         lambda c: [p.unfused(cg_u, c)[1]], dict(SEED_SUM), rounds=1, warmup=tw, rep=tr_)
        cg_f = _tile_then_coord(f"w2 fused T{T}", lambda c: p.fused(c), dict(SEED_W2),
                                guard=guard, warmup=tw, rep=tr_)

        # correctness of both sides against an fp32 reference
        for fn in p.unfused(cg_u, cs):
            fn()
        torch.cuda.synchronize()
        idx, ref = p.reference()
        # moe_sum here runs with add_residual=False, so `routed` is the bare top-k sum and
        # must be compared to the reference directly (subtracting h1 would be wrong).
        err_u = rel_err(p.routed[idx], ref)
        p.routed.zero_()
        for fn in p.fused(cg_f):
            fn()
        torch.cuda.synchronize()
        err_f = rel_err(p.routed[idx], ref)

        # At small T the whole effect is ~1 % and single-pass run-to-run variation is also
        # ~1 %, so each point is an INTERLEAVED median: within a round both sides are timed
        # once, in the same order, and the per-side median is taken across rounds. Drift
        # affecting a round cancels in the ratio.
        ROUNDS = 5
        us, fs, gains = [], [], []
        for _ in range(ROUNDS):
            a = bench_chain(p.unfused(cg_u, cs), warmup=mw, rep=mr).p50_ms
            b = bench_chain(p.fused(cg_f), warmup=mw, rep=mr).p50_ms
            us.append(a)
            fs.append(b)
            gains.append(a / b)
        us_s, fs_s, g_s = sorted(us), sorted(fs), sorted(gains)
        mid = ROUNDS // 2
        gain = g_s[mid]
        row = dict(T=T, rows=p.rows, rows_per_expert=p.rows / E,
                   unfused_ms=us_s[mid], fused_ms=fs_s[mid], gain=gain,
                   gain_min=g_s[0], gain_max=g_s[-1],
                   gain_spread_pct=(g_s[-1] - g_s[0]) / gain * 100,
                   rounds_unfused=us, rounds_fused=fs, rounds_gain=gains,
                   unfused_cfg=cg_u, moe_sum_cfg=cs, fused_cfg=cg_f,
                   rel_err_unfused=err_u, rel_err_fused=err_f,
                   accum_MB=T * H * 2 / 2**20)
        rows_out.append(row)
        print(f"  T={T:<5} unfused {us_s[mid]:8.4f}  fused {fs_s[mid]:8.4f}  "
              f"gain {gain:.4f}x [{g_s[0]:.4f}-{g_s[-1]:.4f}]   "
              f"relerr u={err_u:.1e} f={err_f:.1e}", flush=True)
        del p
        torch.cuda.empty_cache()

    payload = {"id": f"f8_sweep_{tag}", "fusion": "#8 Down GEMM + Expert Merge (atomic)",
               "note": "both sides independently tuned at every batch size",
               "rows": rows_out}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"f8_sweep_{tag}.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote results/f8_sweep_{tag}.json", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    sweep([int(x) for x in args.batches.split(",")], args.tag)


if __name__ == "__main__":
    main()
