"""Focused diagnostic: is the BM=128 vs BM=64 moe disagreement layout coupling or
kernel arithmetic?  Runs the same config pair under both layout regimes and under the
router family (no layout coupling at all)."""
from __future__ import annotations

import torch

from glm52_h200 import config as C
from glm52_h200 import reference as R
from glm52_h200.kernels import lazy_prenorm as K

from tools.repro_f11a_invariance import Problem

DT = C.DTYPE
EPS = C.RMS_NORM_EPS
TOPK = C.NUM_EXPERTS_PER_TOK

A = dict(BLOCK_M=64,  BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=8, num_stages=2)
B = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=8, num_stages=2)

prob = Problem()


def run_moe(cfg, layout_bm):
    sti, eids, ntp = prob.layout(layout_bm)
    out = torch.full((prob.rows, prob.w13_raw.shape[1]), float("nan"), device="cuda", dtype=DT)
    K.launch_moe_gateup(prob.h1_m, prob.w13_fold, out, sti, eids, ntp,
                        prob.rows, TOPK, cfg, True, EPS, 0)
    return out


def run_router(cfg):
    out = torch.full((512, C.N_ROUTED_EXPERTS), float("nan"), device="cuda",
                     dtype=torch.float32)
    K.launch_router(prob.h1, (prob.gate.float() * prob.w.float()).to(C.DTYPE).t().contiguous(), out, cfg, True, EPS, 0)
    return out


def cmp(a, b, name):
    a, b = a.float(), b.float()
    scale = max(float(a.abs().max()), 1e-30)
    diff = (a - b).abs()
    rel = float(diff.max()) / scale
    nz = int((diff > scale * 1e-4).sum())
    print(f"  {name:48s} max_rel={rel:.3e}  elems>1e-4*scale: {nz}")
    if nz:
        r = diff.max(1).values.argmax().item()
        print(f"    worst row {r} (tok {r // 8}, k {r % 8})  ref-vs-row scale "
              f"{float(a[r].abs().max()):.3e}")


print("moe family, SQ_MODE=0 (each config with its OWN layout):")
cmp(run_moe(A, 64), run_moe(B, 128), "cfgA(BM64,LAYOUT64) vs cfgB(BM128,LAYOUT128)")
cmp(run_moe(A, 64), run_moe(A, 64), "cfgA twice, repeat sanity")
print("router family, SQ_MODE=0:")
cmp(run_router(A), run_router(B), "router cfgA vs cfgB")
