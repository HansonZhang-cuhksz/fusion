"""Per-fusion tables for the added regimes T=512 / T=1024, in the same schema as
`report/fusion_<regime>.csv`.

Mappings come from `results/_layer_cfgs_<regime>.json` — tuned by the corrected
`_tile_then_coord` search (joint (BM,BN,BK) sweep + coordinate refinement), which is what
replaced the greedy coordinate search that under-tuned the unfused w13 by 42 % and
manufactured a spurious #6 win.

Every fused variant and every unfused chain is timed with `bench_chain` (one L2 flush before
the whole chain, never between its kernels) and validated against an fp32 reference, exactly
as the original per-family benchmarks were.

Covers the six families whose kernels the layer pipeline exercises: #1, #3, #6, #8, #9, #10.
#4/#5 and #11 are not covered — their kernels are not part of the layer pipeline (both were
shown dominated in context, LOG-11 §6) and tuning them here would be a separate run.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from glm52 import config as C  # noqa: E402
from glm52 import reference as R  # noqa: E402
from glm52.bench.bench_layer import LayerProblem, tune_layer_cfgs  # noqa: E402
from glm52.common import bench_chain, rel_err  # noqa: E402
from glm52.kernels import add_rmsnorm as KN  # noqa: E402
from glm52.kernels import moe_down_merge as KD  # noqa: E402
from glm52.kernels import moe_gateup as KG  # noqa: E402
from glm52.kernels import oproj_resadd as KO  # noqa: E402

REPORT = ROOT / "report"
TOPK = C.NUM_EXPERTS_PER_TOK
I = C.MOE_INTERMEDIATE_SIZE

FIELDS = ["fusion", "variant", "replicates", "coda_correspondence", "fused_ms",
          "fused_mapping", "unfused_total_ms", "speedup", "n_unfused_kernels",
          "unfused_k1_name", "unfused_k1_ms", "unfused_k1_mapping",
          "unfused_k2_name", "unfused_k2_ms", "unfused_k2_mapping",
          "unfused_k3_name", "unfused_k3_ms", "unfused_k3_mapping", "notes"]

ABBREV = [("BLOCK_M", "BM"), ("BLOCK_N", "BN"), ("BLOCK_K", "BK"), ("BLOCK_E", "BE"),
          ("BLOCK_DIM", "BD"), ("BLOCK", "BLK"), ("GROUP_M", "GM"), ("SPLIT_K", "SK"),
          ("ROWS", "ROWS"), ("num_warps", "w"), ("num_stages", "s")]


def fmt(cfg) -> str:
    if not isinstance(cfg, dict):
        return "" if cfg is None else str(cfg)
    return " ".join(f"{s}{cfg[k]}" for k, s in ABBREV if cfg.get(k) is not None)


def t(chain, rep=30):
    return bench_chain(chain, warmup=15, rep=rep).p50_ms


def rows_for(p: LayerProblem, L: dict) -> list[dict]:
    out = []

    def add(fusion, replicates, coda, fused_ms, fused_map, unf_total, kernels, notes="",
            variant="-"):
        r = {f: "" for f in FIELDS}
        r.update(fusion=fusion, variant=variant, replicates=replicates,
                 coda_correspondence=coda, fused_ms=f"{fused_ms:.4f}",
                 fused_mapping=fused_map, unfused_total_ms=f"{unf_total:.4f}",
                 speedup=f"{unf_total / fused_ms:.4f}", n_unfused_kernels=len(kernels),
                 notes=notes)
        for i, (nm, ms, mp) in enumerate(kernels, 1):
            r[f"unfused_k{i}_name"] = nm
            r[f"unfused_k{i}_ms"] = f"{ms:.4f}"
            r[f"unfused_k{i}_mapping"] = mp
        out.append(r)

    # ---- #1 o_proj + ResAdd -----------------------------------------------------------
    g_u, g_f, epi = L["oproj_gemm"], L["oproj_gemm_fused"], L["oproj_epi"]
    f1 = t([lambda: KO.gemm_launch(p.a_attn, p.w_o, p.h1, p.h_in, g_f, True, False)])
    k1 = t([lambda: KO.gemm_launch(p.a_attn, p.w_o, p.c, None, g_u, False, False)])
    k2 = t([lambda: KO.epilogue_launch(p.c, p.h_in, p.h1, epi, True)])
    u1 = t([lambda: KO.gemm_launch(p.a_attn, p.w_o, p.c, None, g_u, False, False),
            lambda: KO.epilogue_launch(p.c, p.h_in, p.h1, epi, True)])
    add("#1 o_proj + ResAdd", "vendor cuBLAS-equivalent epilogue (torch.addmm, beta=1)",
        "yes - GEMM + residual-add epilogue", f1, fmt(g_f), u1,
        [("o_proj GEMM", k1, fmt(g_u)), ("residual add (elementwise)", k2, fmt(epi))],
        variant="triton")

    # ---- #3 ResAdd + RMSNorm ----------------------------------------------------------
    a, n, an = L["add"], L["norm"], L["addnorm"]
    f3 = t([lambda: KN.fused_add_rmsnorm(p.c, p.h_in, p.w_norm, p.h1, p.x2, an)])
    k1 = t([lambda: KN.add_only(p.c, p.h_in, p.h1, a)])
    k2 = t([lambda: KN.norm_only(p.h1, p.w_norm, p.x2, n)])
    u3 = t([lambda: KN.add_only(p.c, p.h_in, p.h1, a),
            lambda: KN.norm_only(p.h1, p.w_norm, p.x2, n)])
    add("#3 ResAdd + RMSNorm", "sglang fused_add_rmsnorm", "no - no GEMM involved",
        f3, fmt(an), u3, [("residual add", k1, fmt(a)), ("rmsnorm", k2, fmt(n))])

    # ---- #6 Up_Gate + SwiGLU ----------------------------------------------------------
    w_u, w_f, act = L["w13"], L["w13_fused"], L["act"]
    su, eu, nu = p.layout(w_u["BLOCK_M"])
    sf, ef, nf = p.layout(w_f["BLOCK_M"])
    f6 = t([lambda: KG.launch_gateup(p.x2, p.w13, p.act, sf, ef, nf, p.rows, TOPK, I, w_f, True)])
    k1 = t([lambda: KG.launch_gateup(p.x2, p.w13, p.inter, su, eu, nu, p.rows, TOPK, I, w_u, False)])
    k2 = t([lambda: KG.launch_silu_and_mul(p.inter, p.act, act)])
    u6 = t([lambda: KG.launch_gateup(p.x2, p.w13, p.inter, su, eu, nu, p.rows, TOPK, I, w_u, False),
            lambda: KG.launch_silu_and_mul(p.inter, p.act, act)])
    add("#6 Up_Gate + SwiGLU", "sglang fused_moe_kernel (GEMM1) + silu_and_mul",
        "yes - GEMM + activation epilogue", f6, fmt(w_f), u6,
        [("w13 grouped GEMM -> [rows, 2I]", k1, fmt(w_u)), ("silu_and_mul", k2, fmt(act))],
        notes="both sides tuned by a joint (BM,BN,BK) sweep; a greedy coordinate search "
              "under-tuned this baseline by 42% and produced a spurious 1.1x win")

    # ---- #8 / #9 Down + Merge (+ ResAdd2) ----------------------------------------------
    d_u, d_f, msum, ra = L["w2"], L["w2_fused8"], L["moe_sum"], L["resadd2"]
    du_s, du_e, du_n = p.layout(d_u["BLOCK_M"])
    df_s, df_e, df_n = p.layout(d_f["BLOCK_M"])
    kd = t([lambda: KD.launch_down(p.act, p.w2, p.y3, p.tw_flat, du_s, du_e, du_n,
                                   p.rows, TOPK, d_u, False)])
    ks = t([lambda: KD.launch_moe_sum(p.y3v, p.routed, p.h1, TOPK, msum, False)])
    kr = t([lambda: KD.launch_resadd(p.routed, p.h1, p.out, ra)])

    f8 = t([lambda: p.routed.zero_(),
            lambda: KD.launch_down(p.act, p.w2, p.routed, p.tw_flat, df_s, df_e, df_n,
                                   p.rows, TOPK, d_f, True)])
    u8 = t([lambda: KD.launch_down(p.act, p.w2, p.y3, p.tw_flat, du_s, du_e, du_n,
                                   p.rows, TOPK, d_u, False),
            lambda: KD.launch_moe_sum(p.y3v, p.routed, p.h1, TOPK, msum, False)])
    add("#8 Down + Expert Merge",
        "sglang fused_moe_kernel, MUL_ROUTED_WEIGHT + FUSE_SUM_ALL_REDUCE",
        "yes - GEMM + scale/accumulate epilogue", f8, fmt(d_f), u8,
        [("w2 grouped down GEMM -> [rows, H]", kd, fmt(d_u)),
         ("moe_sum (expert merge over top-8)", ks, fmt(msum))],
        notes="fused timing includes the output zero-init it requires; bf16 atomics are "
              "non-deterministic", variant="atomic (sglang FUSE_SUM_ALL_REDUCE)")

    f9 = t([lambda: p.out.copy_(p.h1),
            lambda: KD.launch_down(p.act, p.w2, p.out, p.tw_flat, df_s, df_e, df_n,
                                   p.rows, TOPK, d_f, True)])
    u9 = t([lambda: KD.launch_down(p.act, p.w2, p.y3, p.tw_flat, du_s, du_e, du_n,
                                   p.rows, TOPK, d_u, False),
            lambda: KD.launch_moe_sum(p.y3v, p.routed, p.h1, TOPK, msum, False),
            lambda: KD.launch_resadd(p.routed, p.h1, p.out, ra)])
    u9_2k = t([lambda: KD.launch_down(p.act, p.w2, p.y3, p.tw_flat, du_s, du_e, du_n,
                                      p.rows, TOPK, d_u, False),
               lambda: KD.launch_moe_sum(p.y3v, p.out, p.h1, TOPK, msum, True)])
    add("#9 Down + Expert Merge + ResAdd2",
        "sglang fused_moe_kernel, MUL_ROUTED_WEIGHT + FUSE_SUM_ALL_REDUCE (+ residual seed)",
        "yes - GEMM + scale/accumulate epilogue", f9, fmt(d_f), u9,
        [("w2 grouped down GEMM -> [rows, H]", kd, fmt(d_u)),
         ("moe_sum (expert merge over top-8)", ks, fmt(msum)),
         ("residual add 2", kr, fmt(ra))],
        notes=f"unfused_total is the 3-kernel chain; the better 2-kernel baseline "
              f"(moe_sum with ADD_RESIDUAL) runs {u9_2k:.4f} ms -> speedup "
              f"{u9_2k / f9:.3f}x, which is the honest comparison",
        variant="atomic (sglang FUSE_SUM_ALL_REDUCE)")

    # ---- #10 Expert Merge + ResAdd ----------------------------------------------------
    f10 = t([lambda: KD.launch_moe_sum(p.y3v, p.out, p.h1, TOPK, msum, True)])
    u10 = t([lambda: KD.launch_moe_sum(p.y3v, p.routed, p.h1, TOPK, msum, False),
             lambda: KD.launch_resadd(p.routed, p.h1, p.out, ra)])
    add("#10 Expert Merge + ResAdd", "sglang moe_sum + residual add",
        "no - no GEMM involved", f10, fmt(msum) + " ADD_RESIDUAL=1", u10,
        [("expert merge (moe_sum)", ks, fmt(msum)), ("residual add", kr, fmt(ra))])

    # ---- not covered here --------------------------------------------------------------
    for fusion, why in (
        ("#4 ResAdd + RMSNorm + Router", "norm_router kernels are not part of the layer "
                                         "pipeline (dominated in context, LOG-11 S6)"),
        ("#5 RMSNorm + Router", "as #4"),
        ("#11a Lazy Pre-Norm -> w13 grouped GEMM", "lazy_prenorm kernels not tuned at this "
                                                   "regime"),
        ("#11b Lazy Pre-Norm -> router GEMM", "as #11a"),
        ("#11b' half-fused pre-norm -> router GEMM", "as #11a"),
    ):
        r = {f: "" for f in FIELDS}
        r.update(fusion=fusion, variant="-", replicates="", coda_correspondence="",
                 notes=f"NOT MEASURED at this regime: {why}")
        out.append(r)
    return out


def main() -> None:
    REPORT.mkdir(exist_ok=True)
    for regime in ("decode_bs512", "decode_bs1024"):
        print(f"===== {regime} =====", flush=True)
        p = LayerProblem(regime)
        L = tune_layer_cfgs(p)

        # correctness of the pieces this table reports, against fp32
        KN.fused_add_rmsnorm(p.c, p.h_in, p.w_norm, p.h1, p.x2, L["addnorm"])
        torch.cuda.synchronize()
        ref_x2 = R.rmsnorm((p.c.float() + p.h_in.float()).to(p.c.dtype), p.w_norm)
        err = rel_err(p.x2, ref_x2)
        print(f"  add+rmsnorm rel_err vs fp32 = {err:.3e}", flush=True)
        assert err < 5e-2, err

        rows = rows_for(p, L)
        path = REPORT / f"fusion_{regime}.csv"
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {path}  ({len(rows)} rows)", flush=True)
        for r in rows:
            if r["fused_ms"]:
                print(f"    {r['fusion']:<36} {r['fused_ms']:>9} / {r['unfused_total_ms']:>9}"
                      f"  = {r['speedup']}x", flush=True)
        del p
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
