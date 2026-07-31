"""Minimal, single-purpose profiling target for #11b: launch ONE workload N times.

Kept deliberately bare so a hardware-counter profile contains nothing but the kernels under
study (no autotuning, no validation, no extra allocations inside the timed region).

  --mode fused    : the Lazy Pre-Norm router GEMM (1 kernel)
  --mode unfused  : rmsnorm kernel -> router GEMM (2 kernels)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from glm52.kernels import add_rmsnorm as NK
from glm52.kernels import lazy_prenorm as L

H, ER = 6144, 256

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fused", "unfused"], required=True)
    ap.add_argument("--T", type=int, default=65536)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()

    row = json.loads(next(ROOT.glob(f"results/f11b_*_T{a.T}.json")).read_text())
    g = torch.Generator(device="cuda").manual_seed(0)
    h1 = torch.empty(a.T, H, device="cuda", dtype=torch.bfloat16).normal_(0, .1, generator=g)
    x2 = torch.empty_like(h1)
    w = (torch.randn(H, generator=g, device="cuda") * .1 + 1).to(torch.bfloat16)
    b_raw = torch.empty(H, ER, device="cuda", dtype=torch.bfloat16).normal_(0, .02, generator=g)
    b_fold = (w[:, None].float() * b_raw.float()).to(torch.bfloat16).contiguous()
    lg = torch.empty(a.T, ER, device="cuda", dtype=torch.float32)

    if a.mode == "fused":
        cfg = row["fused_cfg"]
        fns = [lambda: L.launch_router(h1, b_fold, lg, cfg, fuse_norm=True)]
    else:
        cn, cg = row["norm_cfg"], row["unfused_cfg"]
        fns = [lambda: NK.norm_only(h1, w, x2, cn),
               lambda: L.launch_router(x2, b_raw, lg, cg, fuse_norm=False)]

    for fn in fns:      # warm up / JIT compile outside the measured region as far as possible
        fn()
    torch.cuda.synchronize()
    for _ in range(a.iters):
        for fn in fns:
            fn()
    torch.cuda.synchronize()
    print(f"{a.mode} T={a.T} done, {a.iters} iters x {len(fns)} kernel(s)")

if __name__ == "__main__":
    main()
