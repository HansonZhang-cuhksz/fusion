"""Collect hardware performance counters for the #3 / #11b / #11b' kernels.

REQUIRES ROOT. The MetaX driver gates the performance-counter channel behind admin
privilege: as a normal user every collection returns

    mx_perf_counter.cpp:517 : Access perfcount channel failed!
    mc_runtime_api.cpp:1898 : mcProfilerConfig: Returned mcErrorNotPermitted

(the analogue of NVIDIA's ERR_NVGPUCTRPERM). The same gate is why the vendor's `mcProfiler
perf_exec` writes result databases with correct schemas and zero rows. Run as:

    sudo -E env CUDA_VISIBLE_DEVICES=0 ~/my-envs/fusion/bin/python glm52/bench/prof_counters.py

Output: results/f11b_counters.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from glm52.kernels import add_rmsnorm as NK      # noqa: E402
from glm52.kernels import lazy_prenorm as L      # noqa: E402
from glm52.mcpti import MCPTI, MCPTIError        # noqa: E402

H, ER = 6144, 256

# The four questions this study could not answer without counters, and the metric that
# answers each. Kept small: every extra metric is another set of kernel replays.
METRICS = [
    "dram_read_bytes",       # does the fused kernel really move 2.84x less?
    "dram_write_bytes",      # "
    "gld_transactions",      # coalescing of the h1 read vs the x2 read
    "gst_transactions",
    "global_hit_rate",       # how much of the x2 round-trip does L2 actually absorb?
    "achieved_occupancy",    # is the computed 12-vs-8 warps/SM real?
    "inst_executed",         # the instruction-count difference, measured
]


def build(T: int):
    g = torch.Generator(device="cuda").manual_seed(0)
    h1 = torch.empty(T, H, device="cuda", dtype=torch.bfloat16).normal_(0, .1, generator=g)
    x2 = torch.empty_like(h1)
    w = (torch.randn(H, generator=g, device="cuda") * .1 + 1).to(torch.bfloat16)
    b = torch.empty(H, ER, device="cuda", dtype=torch.bfloat16).normal_(0, .02, generator=g)
    bf = (w[:, None].float() * b.float()).to(torch.bfloat16).contiguous()
    lg = torch.empty(T, ER, device="cuda", dtype=torch.float32)
    return h1, x2, w, b, bf, lg


def main() -> None:
    Ts = [int(x) for x in sys.argv[1:]] or [8192, 65536]
    out = {}
    for T in Ts:
        row = json.loads(next(ROOT.glob(f"results/f11b_*_T{T}.json")).read_text())
        h1, x2, w, b, bf, lg = build(T)
        cf, cu, cn = row["fused_cfg"], row["unfused_cfg"], row["norm_cfg"]

        work = {
            "norm_only":     lambda: NK.norm_only(h1, w, x2, cn),
            "gemm_unfused":  lambda: L.launch_router(x2, b, lg, cu, fuse_norm=False),
            "gemm_fused":    lambda: L.launch_router(h1, bf, lg, cf, fuse_norm=True),
        }
        out[str(T)] = {}
        for name, fn in work.items():
            fn(); torch.cuda.synchronize()
            try:
                with MCPTI() as m:
                    out[str(T)][name] = m.collect(METRICS, fn)
                print(f"T={T} {name}: ok")
            except MCPTIError as e:
                out[str(T)][name] = {"error": str(e)}
                print(f"T={T} {name}: {e}")
        del h1, x2, w, b, bf, lg
        torch.cuda.empty_cache()

    dst = ROOT / "results" / "f11b_counters.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")
    for T, ks in out.items():
        for k, v in ks.items():
            if "error" in v:
                continue
            print(f"\n  T={T} {k}")
            for mname, mv in v.items():
                if "error" in mv:
                    print(f"    {mname:<22} ERROR {mv['error']}")
                else:
                    print(f"    {mname:<22} {mv['value']:>18,.2f}  ({mv['kind']})")


if __name__ == "__main__":
    main()
