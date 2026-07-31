"""Emit one CSV per benchmark regime into `report/`.

Each CSV lists every *measured* fusion variant with: the production reference the Triton
kernel actually replicates, whether it corresponds to CODA's GEMM-plus-epilogue abstraction,
the fused time and mapping, and the unfused chain broken down per kernel with each kernel's
own time and mapping.

Column definitions (see report/README.md):
  fused_ms          p50 of the fused variant, timed as one chain with one L2 flush before it
  unfused_total_ms  p50 of the whole unfused chain, timed the same way -- AUTHORITATIVE
  unfused_kN_ms     kernel N timed ALONE. The per-kernel times need not sum exactly to
                    unfused_total_ms: in the chain the second kernel's input is still L2-hot,
                    and launches partially overlap. The total is the number to trust.
  *_mapping         the mapping that variant/kernel actually ran at in the reported timing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPORT = ROOT / "report"

REGIMES = ["decode_bs1", "decode_bs32", "decode_bs256", "prefill_t2048", "prefill_t8192"]

ABBREV = [
    ("BLOCK_M", "BM"), ("BLOCK_N", "BN"), ("BLOCK_K", "BK"), ("BLOCK_E", "BE"),
    ("BLOCK_DIM", "BD"), ("BLOCK", "BLK"), ("GROUP_M", "GM"), ("SPLIT_K", "SK"),
    ("ROWS", "ROWS"), ("NORM_BK", "NBK"), ("KVEC", "KVEC"), ("UNROLL", "UNROLL"),
    ("EVICT", "EVICT"), ("num_warps", "w"), ("num_stages", "s"), ("impl", "impl"),
]
SKIP = {"eps", "grid_cap", "EPI"}


def fmt_cfg(cfg) -> str:
    """Render a config dict as a compact, stable mapping string."""
    if cfg is None:
        return ""
    if not isinstance(cfg, dict):
        return str(cfg)
    parts = []
    for key, short in ABBREV:
        if key in cfg and cfg[key] is not None:
            parts.append(f"{short}{cfg[key]}")
    for k, v in cfg.items():  # anything not in the abbreviation table
        if k in SKIP or v is None or any(k == a for a, _ in ABBREV):
            continue
        parts.append(f"{k}={v}")
    return " ".join(parts)


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def rows_by_regime(payload: dict, variant: str | None = None) -> dict:
    out = {}
    for r in payload.get("rows", []):
        if variant is not None and r.get("variant") != variant:
            continue
        out[r["regime"]] = r
    return out


def build() -> dict[str, list[dict]]:
    f01 = load("f01_oproj_resadd")
    f01split = json.loads((RESULTS / "_f01_perkernel_split.json").read_text())
    f03 = load("f03_resadd_rmsnorm")
    f0405 = load("f04f05_norm_router")
    f06 = load("f06_upgate_swiglu")
    f0809 = load("f08f09_down_merge_resadd")
    f10 = load("f10_merge_resadd")
    f11 = load("f11_lazy_prenorm")

    f11_rows = {r["regime"]: r for r in f11["rows"]}
    per_regime: dict[str, list[dict]] = {g: [] for g in REGIMES}

    def emit(regime, **kw):
        base = {
            "fusion": "", "variant": "", "replicates": "", "coda_correspondence": "",
            "fused_ms": "", "fused_mapping": "", "unfused_total_ms": "", "speedup": "",
            "n_unfused_kernels": "",
            "unfused_k1_name": "", "unfused_k1_ms": "", "unfused_k1_mapping": "",
            "unfused_k2_name": "", "unfused_k2_ms": "", "unfused_k2_mapping": "",
            "unfused_k3_name": "", "unfused_k3_ms": "", "unfused_k3_mapping": "",
            "notes": "",
        }
        base.update(kw)
        for k in ("fused_ms", "unfused_total_ms", "speedup",
                  "unfused_k1_ms", "unfused_k2_ms", "unfused_k3_ms"):
            if isinstance(base[k], float):
                base[k] = f"{base[k]:.4f}"
        per_regime[regime].append(base)

    for g in REGIMES:
        # ---- #1 o_proj + ResAdd -------------------------------------------------------
        r = rows_by_regime(f01)[g]
        sp = f01split[g]
        ucfg = {k: v for k, v in r["unfused_cfg"].items() if k != "EPI"}
        fcfg = r["fused_cfg"]
        fmap = fmt_cfg({k: v for k, v in fcfg.items() if k != "EPI"})
        if fcfg.get("EPI"):
            fmap += f" | cast-epi: {fmt_cfg(fcfg['EPI'])}"
        note = ("SPLIT_K>1, so both sides also run a zero-init + fp32 cast kernel"
                if sp["split_k"] > 1 else "")
        if "note" in sp:
            note = (note + "; " if note else "") + sp["note"]
        emit(g, fusion="#1 o_proj + ResAdd", variant="triton",
             replicates="vendor cuBLAS-equivalent epilogue (torch.addmm, beta=1)",
             coda_correspondence="yes - GEMM + residual-add epilogue",
             fused_ms=r["fused_ms"], fused_mapping=fmap,
             unfused_total_ms=r["unfused_ms"], speedup=r["speedup"], n_unfused_kernels=2,
             unfused_k1_name="o_proj GEMM", unfused_k1_ms=sp["gemm_ms"],
             unfused_k1_mapping=fmt_cfg(ucfg),
             unfused_k2_name="residual add (elementwise)", unfused_k2_ms=sp["epi_ms"],
             unfused_k2_mapping=fmt_cfg(sp["epi_cfg"]),
             notes=("per-kernel split re-measured by the main session; the original run "
                    "recorded only the chain total. " + note).strip())

        # ---- #3 ResAdd + RMSNorm ------------------------------------------------------
        r = rows_by_regime(f03)[g]
        emit(g, fusion="#3 ResAdd + RMSNorm", variant="-",
             replicates="sglang fused_add_rmsnorm",
             coda_correspondence="no - no GEMM involved",
             fused_ms=r["fused_ms"], fused_mapping=fmt_cfg(r["fused_cfg"]),
             unfused_total_ms=r["unfused_ms"], speedup=r["speedup"], n_unfused_kernels=2,
             unfused_k1_name="residual add", unfused_k1_ms=r["add_only_ms"],
             unfused_k1_mapping=fmt_cfg(r["unfused_cfg"]["add"]),
             unfused_k2_name="rmsnorm", unfused_k2_ms=r["norm_only_ms"],
             unfused_k2_mapping=fmt_cfg(r["unfused_cfg"]["norm"]),
             notes="fused output bitwise-identical to unfused at 4/5 regimes")

        # ---- #4 / #5 (+topk) Norm + Router --------------------------------------------
        tt = f0405["tune_tables"][g]

        def kms(key):
            v = tt.get(key)
            return v["best_ms"] if isinstance(v, dict) and "best_ms" in v else ""

        for variant, k1name, k1key, has_topk in (
            ("F5", "rmsnorm", "norm", False),
            ("F5_topk", "rmsnorm", "norm", True),
            ("F4", "add+rmsnorm (fusion #3 already applied)", "add_norm", False),
            ("F4_topk", "add+rmsnorm (fusion #3 already applied)", "add_norm", True),
        ):
            r = rows_by_regime(f0405, variant)[g]
            u = r["unfused_cfg"]
            label = {"F5": "#5 RMSNorm + Router", "F5_topk": "#5 RMSNorm + Router + TopK",
                     "F4": "#4 ResAdd + RMSNorm + Router",
                     "F4_topk": "#4 ResAdd + RMSNorm + Router + TopK"}[variant]
            kw = dict(
                unfused_k1_name=k1name, unfused_k1_ms=kms(k1key),
                unfused_k1_mapping=fmt_cfg(u.get("norm")),
                unfused_k2_name="router GEMM (fp32)", unfused_k2_ms=kms("router_gemm"),
                unfused_k2_mapping=fmt_cfg(u.get("gemm")),
            )
            if has_topk:
                kw.update(unfused_k3_name="sigmoid + noaux_tc top-8",
                          unfused_k3_ms=kms("topk"),
                          unfused_k3_mapping=fmt_cfg(u.get("topk")))
            emit(g, fusion=label, variant=variant,
                 replicates="none (custom); router semantics follow sglang biased_grouped_topk",
                 coda_correspondence=("no - fuses the GEMM INTO the norm kernel, the inverse "
                                      "of CODA's GEMM+epilogue direction"),
                 fused_ms=r["fused_ms"], fused_mapping=fmt_cfg(r["fused_cfg"]),
                 unfused_total_ms=r["unfused_ms"], speedup=r["speedup"],
                 n_unfused_kernels=3 if has_topk else 2,
                 notes=("per-kernel times are each kernel's own tuned best; the chain ran "
                        "jointly-tuned configs"), **kw)

        # ---- #6 Up_Gate + SwiGLU ------------------------------------------------------
        r = rows_by_regime(f06)[g]
        # The main run's sequential block recorded a corrupt GEMM-alone time at
        # prefill_t8192 (52.2 ms, larger than the whole chain it belongs to). The agent's
        # own interleaved A/B re-measurement is the sound number; prefer it wherever the
        # recorded value is internally impossible.
        gemm_ms, gemm_note = r["unfused_gemm_ms"], ""
        rounds = f06["verification"]["interleaved_rounds"].get(g, {}).get("rounds", [])
        gem = sorted(x["gemm_only"] for x in rounds if "gemm_only" in x)
        if gem and gemm_ms + r["unfused_act_ms"] > 1.10 * r["unfused_ms"]:
            gemm_ms = gem[len(gem) // 2]
            gemm_note = (f"GEMM-alone taken from the interleaved re-measurement "
                         f"(median of {len(gem)}); the sequential block recorded "
                         f"{r['unfused_gemm_ms']:.4f} ms, which exceeds the chain it is "
                         f"part of and is a known one-off outlier. ")
        emit(g, fusion="#6 Up_Gate + SwiGLU", variant="-",
             replicates="sglang fused_moe_kernel (GEMM1) + silu_and_mul",
             coda_correspondence="yes - GEMM + activation epilogue",
             fused_ms=r["fused_ms"], fused_mapping=fmt_cfg(r["fused_cfg"]),
             unfused_total_ms=r["unfused_ms"], speedup=r["speedup"], n_unfused_kernels=2,
             unfused_k1_name="w13 grouped GEMM -> [rows, 2I]",
             unfused_k1_ms=gemm_ms,
             unfused_k1_mapping=fmt_cfg(r["unfused_gemm_cfg"]),
             unfused_k2_name="silu_and_mul", unfused_k2_ms=r["unfused_act_ms"],
             unfused_k2_mapping=fmt_cfg(r["unfused_act_cfg"]),
             notes=(gemm_note + "fused BM128/BN128 is UNCOMPILABLE (96 KB SMEM vs 64 KB "
                    "limit), so the fused side cannot reach the unfused winner's tile"))

        # ---- #8 / #9 Down + Merge (+ ResAdd2) -----------------------------------------
        for variant in ("f8_atomic", "f8_token_major", "f9_atomic", "f9_token_major"):
            r = rows_by_regime(f0809, variant)[g]
            u = r["unfused_cfg"]
            is9 = variant.startswith("f9")
            is_tok = "token_major" in variant
            label = ("#9 Down + Expert Merge + ResAdd2" if is9
                     else "#8 Down + Expert Merge")
            kw = dict(
                unfused_k1_name="w2 grouped down GEMM -> [rows, H]",
                unfused_k1_ms=r["unfused_gemm_only_ms"],
                unfused_k1_mapping=fmt_cfg(u.get("gemm")),
                unfused_k2_name="moe_sum (expert merge over top-8)",
                unfused_k2_ms=r["unfused_sum_only_ms"],
                unfused_k2_mapping=fmt_cfg(u.get("sum")),
            )
            note = []
            if is9:
                kw.update(unfused_k3_name="residual add 2",
                          unfused_k3_ms=r["unfused_resadd_only_ms"],
                          unfused_k3_mapping=fmt_cfg(u.get("resadd")))
                alt = r.get("unfused9_2kernel_ms")
                if alt:
                    note.append(
                        f"unfused_total is the 3-kernel chain; a strictly better 2-kernel "
                        f"baseline (moe_sum with ADD_RESIDUAL) runs {alt:.4f} ms -> "
                        f"speedup {alt / r['fused_ms']:.3f}x, which LOG-09 uses")
            if is_tok:
                note.append("scored against the EXPERT-major baseline; part of any win is the "
                            "grid change, not the fusion (LOG-08 F3)")
            else:
                note.append("bf16 atomics -> non-deterministic output (LOG-08 F6)")
            emit(g, fusion=label,
                 variant="atomic (sglang FUSE_SUM_ALL_REDUCE)" if not is_tok else "token-major",
                 replicates=("sglang fused_moe_kernel, MUL_ROUTED_WEIGHT + FUSE_SUM_ALL_REDUCE"
                             if not is_tok else "none (custom)"),
                 coda_correspondence=("yes - GEMM + scale/accumulate epilogue" if not is_tok
                                      else "partial - fused epilogue, but a different grid"),
                 fused_ms=r["fused_ms"], fused_mapping=fmt_cfg(r["fused_cfg"].get("gemm", r["fused_cfg"])),
                 unfused_total_ms=r["unfused_ms"], speedup=r["speedup"],
                 n_unfused_kernels=3 if is9 else 2, notes="; ".join(note), **kw)

        # ---- #10 Expert Merge + ResAdd ------------------------------------------------
        r = rows_by_regime(f10)[g]
        emit(g, fusion="#10 Expert Merge + ResAdd", variant="-",
             replicates="sglang moe_sum + residual add",
             coda_correspondence="no - no GEMM involved",
             fused_ms=r["fused_ms"], fused_mapping=fmt_cfg(r["fused_cfg"]),
             unfused_total_ms=r["unfused_ms"], speedup=r["speedup"], n_unfused_kernels=2,
             unfused_k1_name="expert merge (moe_sum)", unfused_k1_ms=r["merge_only_ms"],
             unfused_k1_mapping=fmt_cfg(r["unfused_cfg"]["merge"]),
             unfused_k2_name="residual add", unfused_k2_ms=r["resadd_only_ms"],
             unfused_k2_mapping=fmt_cfg(r["unfused_cfg"]["resadd"]),
             notes="fused output bitwise-identical to unfused")

        # ---- #11 Lazy Pre-Norm --------------------------------------------------------
        fr = f11_rows[g]
        for sub, label, gemm_name in (
            ("f11a_w13", "#11a Lazy Pre-Norm -> w13 grouped GEMM", "w13 grouped GEMM"),
            ("f11b_router", "#11b Lazy Pre-Norm -> router GEMM", "router GEMM"),
        ):
            s = fr[sub]
            emit(g, fusion=label, variant="lazy pre-norm (prologue)",
                 replicates="PyTorch/Meta 'Towards Free Normalization' (Zhou et al. 2026) S2",
                 coda_correspondence="partial - prologue fusion; CODA is epilogue-oriented",
                 fused_ms=s["fused_ms"], fused_mapping=fmt_cfg(s["fused_cfg"]),
                 unfused_total_ms=s["unfused_ms"], speedup=s["speedup"], n_unfused_kernels=2,
                 unfused_k1_name="rmsnorm (writes x2)", unfused_k1_ms=s["norm_only_ms"],
                 unfused_k1_mapping=fmt_cfg(s["unfused_norm_cfg"]),
                 unfused_k2_name=gemm_name, unfused_k2_ms=s["unfused_gemm_only_ms"],
                 unfused_k2_mapping=fmt_cfg(s["unfused_gemm_cfg"]),
                 notes=(f"sum-of-squares redundancy {s.get('sq_redundancy', '?')}x; "
                        "valid only if ALL K=6144 consumers are fused (x2 never materialised)"))

        # half-fused: rstd in a tiny kernel, applied as a pure epilogue scale
        h = fr["half_fused"]
        s = fr["f11b_router"]
        emit(g, fusion="#11b' half-fused pre-norm -> router GEMM", variant="rstd + epilogue scale",
             replicates="none (custom; derived from Lazy Pre-Norm)",
             coda_correspondence="yes - GEMM + scale epilogue, reduction kept outside",
             fused_ms=h["router_ms"],
             fused_mapping=f"rstd: {fmt_cfg(h['rstd_cfg'])} | gemm: {fmt_cfg(h['router_cfg'])}",
             unfused_total_ms=s["unfused_ms"], speedup=h["router_speedup_vs_unfused"],
             n_unfused_kernels=2,
             unfused_k1_name="rmsnorm (writes x2)", unfused_k1_ms=s["norm_only_ms"],
             unfused_k1_mapping=fmt_cfg(s["unfused_norm_cfg"]),
             unfused_k2_name="router GEMM", unfused_k2_ms=s["unfused_gemm_only_ms"],
             unfused_k2_mapping=fmt_cfg(s["unfused_gemm_cfg"]),
             notes=(f"the FUSED side is itself 2 kernels (rstd {h['rstd_only_ms']:.4f} ms + "
                    f"GEMM); router-only -- end-to-end over all consumers is "
                    f"{h['combined_speedup_vs_unfused']:.3f}x"))

    return per_regime


FIELDS = ["fusion", "variant", "replicates", "coda_correspondence", "fused_ms",
          "fused_mapping", "unfused_total_ms", "speedup", "n_unfused_kernels",
          "unfused_k1_name", "unfused_k1_ms", "unfused_k1_mapping",
          "unfused_k2_name", "unfused_k2_ms", "unfused_k2_mapping",
          "unfused_k3_name", "unfused_k3_ms", "unfused_k3_mapping", "notes"]


def _consistency_check(rows: list[dict]) -> list[str]:
    """Flag rows where a SINGLE kernel is slower than the whole chain containing it.

    Note we deliberately do NOT flag `sum(parts) > total`. That is normal and expected:
    each kernel timed alone pays its own launch overhead and its own L2 flush, whereas
    inside the chain the second kernel's input is still L2-hot and launches partly overlap.
    For sub-0.1 ms kernels this routinely makes the parts sum to 1.1-1.4x the total.
    One kernel exceeding the total, however, is impossible and always means a bad sample.
    """
    warnings = []
    for r in rows:
        try:
            total = float(r["unfused_total_ms"])
        except (TypeError, ValueError):
            continue
        if total <= 0:
            continue
        for k in ("unfused_k1_ms", "unfused_k2_ms", "unfused_k3_ms"):
            try:
                part = float(r[k])
            except (TypeError, ValueError):
                continue
            if part > 1.02 * total:
                msg = (f"{r[k.replace('_ms', '_name')]} alone ({part:.4f} ms) exceeds the "
                       f"chain total ({total:.4f} ms) — impossible, suspect sample")
                r["notes"] = (r["notes"] + "; " if r["notes"] else "") + "WARNING: " + msg
                warnings.append(f"    {r['fusion']} [{r['variant']}]: {msg}")
    return warnings


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    data = build()
    for regime, rows in data.items():
        for w in _consistency_check(rows):
            print(f"  {regime}:\n{w}")
    for regime, rows in data.items():
        path = REPORT / f"fusion_{regime}.csv"
        with path.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(rows)
        print(f"{path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
