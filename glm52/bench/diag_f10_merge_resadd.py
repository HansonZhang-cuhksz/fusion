"""F10 attribution diagnostics.

(i)  n_regs / n_spills / shared for the fused kernel vs the two split kernels, compiled
     with the Triton cache cleared between compiles (the technique LOG-10 prescribes).
(ii) point the fused kernel's EXTRA input (the residual) at a stride-0 broadcast row --
     identical instruction stream, ~zero DRAM traffic -- to separate the instruction cost
     of the fused epilogue from its memory cost.
(iii) same trick on the unfused resadd kernel, for symmetry.
"""
import json
import sys

sys.path.insert(0, "/home/zhangshuhan/fusion")
import torch

from glm52 import config as C
from glm52.common import bench_chain
from glm52.kernels import merge_resadd as K

H, TOPK, DT = C.HIDDEN_SIZE, C.NUM_EXPERTS_PER_TOK, C.DTYPE
RES_JSON = "/home/zhangshuhan/fusion/results/f10_merge_resadd.json"
dev = torch.cuda.current_device()


def kstats():
    ks = K.merge_resadd_kernel
    cache = ks.cache[dev]
    out = []
    for k, v in cache.items():
        out.append(
            {
                "n_regs": getattr(v, "n_regs", None),
                "n_spills": getattr(v, "n_spills", None),
                "shared": getattr(v, "shared", None),
            }
        )
    return out


def clear():
    K.merge_resadd_kernel.cache[dev].clear()


res_doc = json.loads(open(RES_JSON).read())
rows = {r["regime"]: r for r in res_doc["rows"]}

report = {}
for name in ("decode_bs256", "prefill_t2048", "prefill_t8192"):
    r = rows[name]
    T = r["T"]
    fcfg = r["fused_cfg"]
    mcfg = r["unfused_cfg"]["merge"]
    acfg = r["unfused_cfg"]["resadd"]

    torch.manual_seed(7)
    y = torch.randn(T, TOPK, H, device="cuda", dtype=DT)
    wt = torch.rand(T, TOPK, device="cuda", dtype=torch.float32)
    res = torch.randn(T, H, device="cuda", dtype=DT)
    out = torch.empty(T, H, device="cuda", dtype=DT)
    mu = torch.empty(T, H, device="cuda", dtype=DT)

    # --- (i) codegen stats, one kernel per cleared cache -------------------------------
    stats = {}
    clear(); K.fused_merge_resadd(y, wt, res, out, fcfg); torch.cuda.synchronize()
    stats["fused"] = kstats()
    clear(); K.merge_only(y, wt, mu, mcfg); torch.cuda.synchronize()
    stats["merge_only"] = kstats()
    clear(); K.resadd_only(mu, res, out, acfg); torch.cuda.synchronize()
    stats["resadd_only"] = kstats()
    # fused at the merge_only winning config -> apples-to-apples register delta
    clear(); K.merge_only(y, wt, mu, mcfg); torch.cuda.synchronize()
    s_m = kstats()
    clear(); K.fused_merge_resadd(y, wt, res, out, mcfg); torch.cuda.synchronize()
    s_f = kstats()
    stats["same_cfg_merge"] = s_m
    stats["same_cfg_fused"] = s_f
    clear()

    # --- (ii) stride-0 residual: same instructions, ~no DRAM traffic -------------------
    # The four variants are timed in an INTERLEAVED round-robin, 3 passes, min-of-medians.
    # Interleaving means any drift hits all four equally; min-of-medians rejects one-off
    # excursions.  (A first, non-interleaved version of this script produced a 2.86 ms
    # reading for a 0.036 ms kernel at decode_bs256 -- see LOG-06 section 6.)
    res_bc = res[0:1].expand(T, H)  # stride(0) == 0
    variants = {
        "fused_real_res_ms": lambda: K.fused_merge_resadd(y, wt, res, out, fcfg),
        "fused_stride0_res_ms": lambda: K.fused_merge_resadd(y, wt, res_bc, out, fcfg),
        "merge_only_ms": lambda: K.merge_only(y, wt, mu, mcfg),
        "merge_only_at_fused_cfg_ms": lambda: K.merge_only(y, wt, mu, fcfg),
    }
    best = {k: float("inf") for k in variants}
    for _ in range(3):
        for k, fn in variants.items():
            best[k] = min(best[k], bench_chain([fn], 20, 100).p50_ms)

    report[name] = {
        "T": T,
        "cfgs": {"fused": fcfg, "merge": mcfg, "resadd": acfg},
        "codegen": stats,
        **best,
        "residual_dram_cost_ms": best["fused_real_res_ms"]
        - best["fused_stride0_res_ms"],
        "residual_instr_cost_ms": best["fused_stride0_res_ms"]
        - best["merge_only_at_fused_cfg_ms"],
    }
    print(name, json.dumps(report[name], indent=1, default=str), flush=True)

# Merge into the result file so the diagnostics are auditable next to the numbers they
# explain.  Re-runnable: it overwrites only the `diagnostics` key.
doc = json.loads(open(RES_JSON).read())
doc["diagnostics"] = {
    "what": "(i) n_regs/n_spills with the Triton cache cleared between compiles; "
    "(ii) the fused kernel's extra input (residual) pointed at a stride-0 broadcast "
    "row, which keeps the instruction stream identical but removes the DRAM traffic, "
    "separating the fused epilogue's instruction cost from its memory cost.",
    "script": "glm52/bench/diag_f10_merge_resadd.py",
    "by_regime": report,
}
open(RES_JSON, "w").write(json.dumps(doc, indent=2, default=str))
print("DIAG OK -> wrote diagnostics into", RES_JSON)
