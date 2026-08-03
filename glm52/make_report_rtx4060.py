"""Per-regime fusion CSVs for the RTX 4060 port, in the same schema as report_glm52_c500/.

Only the five families measurable on 8 GB are present: #1, #3, #4, #5, #10, #11b (+#11b').
#6, #8, #9 and #11a need the 256-expert w13/w2 weights (12.0 / 6.0 GB) and cannot run here;
they are absent rather than estimated.

Descriptive columns (`replicates`, `coda_correspondence`) are properties of the fusion and are
carried over verbatim from the C500 report; every number is re-measured on this device.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "rtx4060"
OUT = ROOT / "report_glm52_rtx4060"
C500 = ROOT / "report_glm52_c500"
REGIMES = ["decode_bs1", "decode_bs32", "decode_bs256", "prefill_t2048", "prefill_t8192"]

FIELDS = ["fusion", "variant", "replicates", "coda_correspondence", "fused_ms",
          "fused_mapping", "unfused_total_ms", "speedup", "n_unfused_kernels",
          "unfused_k1_name", "unfused_k1_ms", "unfused_k1_mapping",
          "unfused_k2_name", "unfused_k2_ms", "unfused_k2_mapping",
          "unfused_k3_name", "unfused_k3_ms", "unfused_k3_mapping", "notes"]

def annot() -> dict:
    """(fusion, variant) -> (replicates, coda_correspondence) from the C500 report."""
    out = {}
    for p in C500.glob("fusion_*.csv"):
        for r in csv.DictReader(p.open()):
            out.setdefault((r["fusion"], r["variant"]), (r["replicates"], r["coda_correspondence"]))
    return out

def m(cfg: dict | None) -> str:
    """cfg dict -> the compact mapping string the C500 report uses."""
    if not cfg:
        return ""
    k = {"BLOCK_M": "BM", "BLOCK_N": "BN", "BLOCK_K": "BK", "GROUP_M": "GM", "SPLIT_K": "SK",
         "BLOCK_E": "BE", "NORM_BK": "NBK", "BLOCK": "BLK", "ROWS": "ROWS", "KVEC": "KVEC",
         "UNROLL": "UNROLL", "EVICT": "EVICT", "grid_cap": "CAP"}
    parts = [f"{v}{cfg[a]}" for a, v in k.items() if cfg.get(a) not in (None, False)]
    if cfg.get("num_warps"): parts.append(f"w{cfg['num_warps']}")
    if cfg.get("num_stages"): parts.append(f"s{cfg['num_stages']}")
    return " ".join(parts)

def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None

# --- caveats established by the adversarial review (LOG-13 S9) -------------------------
TICK = ("decode timings are 9-17 ticks of this device's 1.024 us CUDA-event granularity "
        "(4x coarser than C500's 0.256 us); ratio resolvable to about +-8%")
LAUNCH = ("win is dominated by eliminating one kernel launch (measured L=3.36 us vs a "
          "4.10 us gap); at T=1 the working set is 12 KB, so almost none of it is bandwidth")
F01_T8192 = ("campaign reported 1.0267, which exceeds this cell's physical ceiling; that run "
             "drifted 22% thermally (coarse 137.25 ms vs final 167.20 ms) and times the fused "
             "arm entirely before the unfused. Value here is the interleaved A/B re-measurement "
             "(n=120, paired p50 1.0143); primitive decomposition at the shared config gives "
             "gemm(FUSE=True) 134.969 ms vs gemm(FUSE=False) 135.050 ms + epilogue 1.956 ms "
             "-> 1.0151. Fusing the residual costs +0.06% of GEMM time here vs +22.8% on C500.")
F0405_DEC = ("unfused router GEMM is under-tuned at decode: bench_f04f05 offers BLOCK_K in "
             "{32,64} with no GROUP_M axis and no refine, while bench_f11 tunes the "
             "byte-identical op over BLOCK_K in {32,64,128} + GROUP_M + refine and reaches "
             "0.0522 ms vs 0.1116 ms at bs1. The true speedup is WORSE than shown.")

def rows_for(regime: str) -> list[dict]:
    A = annot(); out = []
    def add(fusion, variant, **kw):
        rep, coda = A.get((fusion, variant), A.get((fusion, "-"), ("", "")))
        r = {f: "" for f in FIELDS}
        r.update(fusion=fusion, variant=variant, replicates=rep, coda_correspondence=coda)
        r.update({k: v for k, v in kw.items() if v is not None})
        out.append(r)
    dec = regime.startswith("decode")
    tiny = regime in ("decode_bs1", "decode_bs32")

    # ---- #1 -------------------------------------------------------------------------
    d = load("f01_oproj_resadd.json")
    if d:
        r = next((x for x in d["rows"] if x["regime"] == regime), None)
        if r:
            sp, fm, note = r["speedup"], r["fused_ms"], ""
            if regime == "prefill_t8192":
                e = json.loads((RES / "exp1_f01_interleaved.json").read_text())
                sp, fm, note = e["paired_p50"], e["fused_p50"], F01_T8192
                um = e["unfused_p50"]
            else:
                um = r["unfused_ms"]
            k1 = k2 = None
            if regime == "prefill_t8192":
                p = json.loads((RES / "exp1b_f01_primitives.json").read_text())
                k1, k2 = p["gemm FUSE_RESADD=False"], p["epilogue add-bf16"]
            add("#1 o_proj + ResAdd", "triton", fused_ms=round(fm, 4),
                fused_mapping=m(r["fused_cfg"]), unfused_total_ms=round(um, 4),
                speedup=round(sp, 4), n_unfused_kernels=2,
                unfused_k1_name="o_proj GEMM", unfused_k1_ms=round(k1, 4) if k1 else "",
                unfused_k1_mapping=m(r["unfused_cfg"]),
                unfused_k2_name="residual add (elementwise)",
                unfused_k2_ms=round(k2, 4) if k2 else "", notes=note)

    # ---- #3 -------------------------------------------------------------------------
    d = load("f03_resadd_rmsnorm.json")
    if d:
        r = next((x for x in d["rows"] if x["regime"] == regime), None)
        if r:
            u = r["unfused_cfg"]
            note = f"{LAUNCH}; {TICK}" if tiny else ""
            if regime == "decode_bs256":
                note = ("h1 fits this device's 32 MB L2 (it spilled C500's 8 MB), so the pass "
                        "the fusion removes is not DRAM traffic here -- fusion turns negative")
            add("#3 ResAdd + RMSNorm", "-", fused_ms=round(r["fused_ms"], 4),
                fused_mapping=m(r["fused_cfg"]), unfused_total_ms=round(r["unfused_ms"], 4),
                speedup=round(r["speedup"], 4), n_unfused_kernels=2,
                unfused_k1_name="residual add", unfused_k1_ms=round(r["add_only_ms"], 4),
                unfused_k1_mapping=m(u.get("add")), unfused_k2_name="rmsnorm",
                unfused_k2_ms=round(r["norm_only_ms"], 4), unfused_k2_mapping=m(u.get("norm")),
                notes=note)

    # ---- #4 / #5 --------------------------------------------------------------------
    d = load("f04f05_norm_router.json")
    if d:
        names = {"F5": "#5 RMSNorm + Router", "F5_topk": "#5 RMSNorm + Router + TopK",
                 "F4": "#4 ResAdd + RMSNorm + Router", "F4_topk": "#4 ResAdd + RMSNorm + Router + TopK"}
        for v in ("F5", "F5_topk", "F4", "F4_topk"):
            r = next((x for x in d["rows"] if x["regime"] == regime and x["variant"] == v), None)
            if not r: continue
            add(names[v], v, fused_ms=round(r["fused_ms"], 4), fused_mapping=m(r["fused_cfg"]),
                unfused_total_ms=round(r["unfused_ms"], 4), speedup=round(r["speedup"], 4),
                n_unfused_kernels=2, unfused_k1_name="rmsnorm (or add+rmsnorm)",
                unfused_k2_name="router GEMM (fp32)", unfused_k2_mapping=m(r.get("unfused_cfg")),
                notes=F0405_DEC if dec else "")

    # ---- #10 ------------------------------------------------------------------------
    d = load("f10_merge_resadd.json")
    if d:
        r = next((x for x in d["rows"] if x["regime"] == regime), None)
        if r:
            note = f"{LAUNCH}; {TICK}" if tiny else ""
            add("#10 Expert Merge + ResAdd", "-", fused_ms=round(r["fused_ms"], 4),
                fused_mapping=m(r["fused_cfg"]), unfused_total_ms=round(r["unfused_ms"], 4),
                speedup=round(r["speedup"], 4), n_unfused_kernels=2,
                unfused_k1_name="expert merge (moe_sum)", unfused_k1_ms=round(r["merge_only_ms"], 4),
                unfused_k2_name="residual add", unfused_k2_ms=round(r["resadd_only_ms"], 4),
                notes=note)

    # ---- #11b and #11b' -------------------------------------------------------------
    d = load("f11_lazy_prenorm.json")
    if d:
        r = next((x for x in d["rows"] if x["regime"] == regime), None)
        if r and r.get("f11b_router"):
            b = r["f11b_router"]
            add("#11b Lazy Pre-Norm -> router GEMM", "lazy pre-norm (prologue)",
                fused_ms=round(b["fused_ms"], 4), fused_mapping=m(b["fused_cfg"]),
                unfused_total_ms=round(b["unfused_ms"], 4), speedup=round(b["speedup"], 4),
                n_unfused_kernels=2, unfused_k1_name="rmsnorm (writes x2)",
                unfused_k1_ms=round(b["norm_only_ms"], 4), unfused_k1_mapping=m(b.get("unfused_norm_cfg")),
                unfused_k2_name="router GEMM", unfused_k2_ms=round(b["unfused_gemm_only_ms"], 4),
                unfused_k2_mapping=m(b.get("unfused_gemm_cfg")),
                notes=f"sum-of-squares redundancy {b.get('sq_redundancy','?')}x; "
                      f"valid only if ALL K=6144 consumers are fused (x2 never materialised)")
            h = r.get("half_fused")
            if h:
                add("#11b' half-fused pre-norm -> router GEMM", "rstd + epilogue scale",
                    fused_ms=round(h["router_ms"], 4),
                    fused_mapping=f"rstd: {m(h.get('rstd_cfg'))} | gemm: {m(h.get('router_cfg'))}",
                    unfused_total_ms=round(b["unfused_ms"], 4),
                    speedup=round(h["router_speedup_vs_unfused"], 4), n_unfused_kernels=2,
                    unfused_k1_name="rmsnorm (writes x2)", unfused_k1_ms=round(b["norm_only_ms"], 4),
                    unfused_k2_name="router GEMM", unfused_k2_ms=round(b["unfused_gemm_only_ms"], 4),
                    notes=f"fused side is itself 2 kernels (rstd {h.get('rstd_only_ms',0):.4f} ms "
                          f"+ GEMM); router-only. {h.get('note','')}")
    return out

def main() -> None:
    OUT.mkdir(exist_ok=True)
    for reg in REGIMES:
        rows = rows_for(reg)
        p = OUT / f"fusion_{reg}.csv"
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
        print(f"  wrote {p.relative_to(ROOT)}  ({len(rows)} rows)")

if __name__ == "__main__":
    main()
