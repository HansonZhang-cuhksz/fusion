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
from glm52 import config as C
from glm52.common import RESULTS_DIR
from glm52.kernels import lazy_prenorm as L
from glm52.kernels import add_rmsnorm as NK

H, ER = 6144, 256
# Every occupancy limit below is per-SM and is probed, never hardcoded: the C500 numbers
# baked in here (65536 smem, 131072 regs, 2048 threads, 104 SMs) are each wrong in a
# different direction on sm89 (102400 smem, 65536 regs, 1536 threads, 24 SMs), so the
# `limited_by` labels this script feeds into the #11b narrative flip.

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
    # Triton 3.x dropped JITFunction.cache; it is device_caches[dev] = (kernel_cache, ...).
    kern = L.router_gemm_kernel
    kern.device_caches.clear()
    L.launch_router(a, b, c, cfg, fuse_norm=fuse)
    torch.cuda.synchronize()
    ent = [v for tup in kern.device_caches.values() for v in tup[0].values()]
    assert len(ent) == 1, len(ent)
    return ent[0]

def compile_norm(h1, w, x2, cfg):
    kern = NK.add_rmsnorm_kernel
    kern.device_caches.clear()
    NK.norm_only(h1, w, x2, cfg)
    torch.cuda.synchronize()
    ent = [v for tup in kern.device_caches.values() for v in tup[0].values()]
    assert len(ent) == 1, len(ent)
    return ent[0]

def sm_limits():
    """(regs, shared bytes, threads) available per SM. Triton's probe exports the register
    file but not per-SM smem or threads/SM, so those two come from torch's properties --
    102400 B / 1536 on sm89, vs the 65536 / 2048 this script used to assume."""
    e = C.env()
    p = torch.cuda.get_device_properties(0)
    regs = (getattr(e, "regs_per_sm", 0) or getattr(e, "max_regs_per_block", 0)
            or e.extras.get("max_num_regs", 0))
    smem = getattr(e, "smem_per_sm", 0) or p.shared_memory_per_multiprocessor
    thr = getattr(e, "threads_per_sm", 0) or p.max_threads_per_multi_processor
    assert regs and smem and thr, f"device probe gave no per-SM limits: {regs},{smem},{thr}"
    return regs, smem, thr

def ctas_per_sm(k, threads):
    regs_sm, smem_sm, thr_sm = sm_limits()
    by_reg = regs_sm // max(1, k.n_regs * threads)
    sh = getattr(k.metadata, "shared", 0) or 1
    by_sh = smem_sm // sh
    # threads/SM binds on Ada as well (1536). The 24-CTAs/SM hardware cap is not exported
    # by either probe, so it is left out; it only bites at num_warps == 1.
    return min(by_reg, by_sh, thr_sm // max(1, threads)), by_reg, by_sh

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
    warp = C.env().warp_size          # 32 lanes on sm89, 64 on C500 -- threads/CTA is 2x off
    for nm, k, thr in (("fused", kf, cf["num_warps"] * warp),
                       ("unfused_gemm", ku, cu["num_warps"] * warp),
                       ("norm", kn, cn["num_warps"] * warp),
                       ("fused_at_unfused_cfg", km, cu["num_warps"] * warp)):
        occ, byr, bys = ctas_per_sm(k, thr)
        res[nm] = dict(n_regs=k.n_regs, n_spills=k.n_spills,
                       shared=getattr(k.metadata, "shared", 0), threads=thr,
                       ctas_per_sm=occ, limited_by="regs" if byr <= bys else "smem",
                       warps_per_sm=occ * thr // warp,
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
        # RESULTS_DIR, not ROOT/results: the tuned configs must be this device's own.
        cands = sorted(RESULTS_DIR.glob(f"f11b_*_T{T}.json"))
        if not cands:
            raise SystemExit(f"no f11b_*_T{T}.json in {RESULTS_DIR}; run bench_f11b_sweep first")
        row = json.loads(cands[0].read_text())
        allout.append(report(T, row))
        torch.cuda.empty_cache()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "f11b_arch_analysis.json"
    out_path.write_text(json.dumps(allout, indent=2))
    print(json.dumps(allout, indent=2)[:200])
    print(f"\nwrote {out_path}")
