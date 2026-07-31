"""Architectural comparison of #11b's fused kernel vs the unfused chain.

Three layers of evidence, cheapest first:
  1. compiler resources  -- n_regs / n_spills / shared, hence CTAs per SM (occupancy)
  2. TTGIR op counts     -- MMA, global load/store, reduction and arithmetic ops, both at the
                            TUNED configs (explains measured time) and at a MATCHED config
                            (isolates what fusion itself adds)
  3. analytic traffic    -- bytes each workload must move through DRAM, from first principles
"""
from __future__ import annotations
import re, sys, json
from collections import Counter
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from glm52.kernels import lazy_prenorm as L
from glm52.kernels import add_rmsnorm as NK

H, ER, SMEM, REGS_SM, NSM = 6144, 256, 65536, 131072, 104

OPS = {
    "tt.dot (MMA)":        r"\btt\.dot\b",
    "tt.load (global)":    r"\btt\.load\b",
    "tt.store (global)":   r"\btt\.store\b",
    "local_load (smem)":   r"local_load\b",
    "local_store (smem)":  r"local_(?:alloc|store)\b",
    "arith.mulf":          r"\barith\.mulf\b",
    "arith.addf":          r"\barith\.addf\b",
    "reduce":              r"\btt\.reduce\b",
    "math.sqrt/rsqrt":     r"math\.(?:sqrt|rsqrt)",
    "arith.divf":          r"\barith\.divf\b",
    "convert_layout":      r"convert_layout\b",
}

def counts(ir: str) -> Counter:
    return Counter({k: len(re.findall(p, ir)) for k, p in OPS.items()})

def compile_router(a, b, c, cfg, fuse):
    ck = L.router_gemm_kernel.cache
    for d in list(ck.keys()): d.clear() if isinstance(d, dict) else ck[d].clear()
    L.launch_router(a, b, c, cfg, fuse_norm=fuse)
    torch.cuda.synchronize()
    ent = [v for d in ck.values() for v in d.values()]
    assert len(ent) == 1, len(ent)
    return ent[0]

def compile_norm(h1, w, x2, cfg):
    ck = NK.add_rmsnorm_kernel.cache
    for d in list(ck.keys()): ck[d].clear()
    NK.norm_only(h1, w, x2, cfg)
    torch.cuda.synchronize()
    ent = [v for d in ck.values() for v in d.values()]
    assert len(ent) == 1, len(ent)
    return ent[0]

def ctas_per_sm(k, threads):
    by_reg = REGS_SM // max(1, k.n_regs * threads)
    sh = getattr(k.metadata, "shared", 0) or 1
    by_sh = SMEM // sh
    return min(by_reg, by_sh), by_reg, by_sh

def report(T: int, row: dict) -> dict:
    g = torch.Generator(device="cuda").manual_seed(0)
    h1 = torch.empty(T, H, device="cuda", dtype=torch.bfloat16).normal_(0, .1, generator=g)
    x2 = torch.empty_like(h1)
    w = (torch.randn(H, generator=g, device="cuda") * .1 + 1).to(torch.bfloat16)
    b = torch.empty(H, ER, device="cuda", dtype=torch.bfloat16).normal_(0, .02, generator=g)
    lg = torch.empty(T, ER, device="cuda", dtype=torch.float32)

    cf, cu, cn = row["fused_cfg"], row["unfused_cfg"], row["norm_cfg"]
    out = {"T": T, "cfg": {"fused": cf, "unfused_gemm": cu, "norm": cn}}

    kf = compile_router(h1, b, lg, cf, True)
    ku = compile_router(x2, b, lg, cu, False)
    kn = compile_norm(h1, w, x2, cn)
    km = compile_router(x2, b, lg, cu, True)     # fused at the UNFUSED config (matched)

    res = {}
    for nm, k, thr in (("fused", kf, cf["num_warps"] * 64),
                       ("unfused_gemm", ku, cu["num_warps"] * 64),
                       ("norm", kn, cn["num_warps"] * 64),
                       ("fused_at_unfused_cfg", km, cu["num_warps"] * 64)):
        occ, byr, bys = ctas_per_sm(k, thr)
        res[nm] = dict(n_regs=k.n_regs, n_spills=k.n_spills,
                       shared=getattr(k.metadata, "shared", 0), threads=thr,
                       ctas_per_sm=occ, limited_by="regs" if byr <= bys else "smem",
                       warps_per_sm=occ * thr // 64,
                       ops={k2: v for k2, v in counts(k.asm["ttgir"]).items() if v})
    out["kernels"] = res

    # analytic DRAM traffic (bf16 activations, fp32 logits)
    act = T * H * 2
    out["traffic_MB"] = {
        "unfused_chain": dict(norm_read_h1=act, norm_write_x2=act, gemm_read_x2=act,
                              gemm_read_B=H*ER*2, gemm_write_logits=T*ER*4),
        "fused":         dict(gemm_read_h1=act, gemm_read_B=H*ER*2,
                              gemm_write_logits=T*ER*4),
    }
    for k2 in out["traffic_MB"]:
        d = out["traffic_MB"][k2]
        d["TOTAL_MB"] = round(sum(d.values()) / 2**20, 2)
        for kk in list(d):
            if kk != "TOTAL_MB": d[kk] = round(d[kk] / 2**20, 2)
    # grid sizes
    out["grid"] = {
        "fused": (-(-T // cf["BLOCK_M"])) * (-(-ER // cf["BLOCK_N"])),
        "unfused_gemm": (-(-T // cu["BLOCK_M"])) * (-(-ER // cu["BLOCK_N"])),
        "sq_redundancy_fused": -(-ER // cf["BLOCK_N"]),
    }
    out["measured_ms"] = {k2: row[k2] for k2 in
                          ("unfused_ms", "fused_ms", "norm_only_ms", "gemm_only_ms", "gain")}
    return out

if __name__ == "__main__":
    Ts = [int(x) for x in sys.argv[1:]] or [65536]
    allout = []
    for T in Ts:
        cands = list(ROOT.glob(f"results/f11b_*_T{T}.json"))
        row = json.loads(cands[0].read_text())
        allout.append(report(T, row))
        torch.cuda.empty_cache()
    (ROOT / "results" / "f11b_arch_analysis.json").write_text(json.dumps(allout, indent=2))
    print(json.dumps(allout, indent=2)[:200])
    print("\nwrote results/f11b_arch_analysis.json")
