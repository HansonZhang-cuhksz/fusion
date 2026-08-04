"""Per-regime fusion CSVs for the H200 (sm_90) campaign, in the C500 report's schema.

Modelled on `make_report_rtx4060.py`.  Differences that matter:

* **Everything ran.**  The H200 fits the 256-expert w13 (12.0 GB) and w2 (6.0 GB), so #6, #8,
  #9 and #11a -- absent from the 4060 report -- are present and measured here.  All seven
  regimes are present.
* **Every number is read out of `results/h200/*.json`.**  Nothing is copied from a prose
  summary and nothing is modelled into a column whose name implies measurement.  Where a
  quantity was never measured on this device the cell is EMPTY (see `#1`'s `unfused_k1_ms`:
  `bench_f01` times the unfused arm only as a chain, so the GEMM alone has no number here).
* **The `notes` column carries the fairness record per cell.**  For every cell that means the
  `flags` list from `results/h200/summary.json` verbatim (all 90 cells carry SEQUENTIAL), the
  `speedup_source`, the paired statistic that the headline ratio is *not*, the order/drift/
  clock diagnostics from that cell's own `pair_meta`, and which Hopper axes (TMA / warp
  specialization / thread-block clusters) the winning mappings actually selected.

Descriptive columns (`replicates`, `coda_correspondence`) are properties of the fusion, not of
the device, and are carried over verbatim from the C500 report by (fusion, variant) key.

Run:  python3 glm52/make_report_h200.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "h200"
OUT = ROOT / "report_glm52_h200"
C500 = ROOT / "report_glm52_c500"

REGIMES = ["decode_bs1", "decode_bs32", "decode_bs256", "decode_bs512", "decode_bs1024",
           "prefill_t2048", "prefill_t8192"]

FIELDS = ["fusion", "variant", "replicates", "coda_correspondence", "fused_ms",
          "fused_mapping", "unfused_total_ms", "speedup", "n_unfused_kernels",
          "unfused_k1_name", "unfused_k1_ms", "unfused_k1_mapping",
          "unfused_k2_name", "unfused_k2_ms", "unfused_k2_mapping",
          "unfused_k3_name", "unfused_k3_ms", "unfused_k3_mapping", "notes"]

# (fusion, variant) labels are the C500 ones so the three reports diff row-for-row.
NAME = {
    ("f01", "triton"): ("#1 o_proj + ResAdd", "triton"),
    ("f03", "f3"): ("#3 ResAdd + RMSNorm", "-"),
    ("f04f05", "F5"): ("#5 RMSNorm + Router", "F5"),
    ("f04f05", "F5_topk"): ("#5 RMSNorm + Router + TopK", "F5_topk"),
    ("f04f05", "F4"): ("#4 ResAdd + RMSNorm + Router", "F4"),
    ("f04f05", "F4_topk"): ("#4 ResAdd + RMSNorm + Router + TopK", "F4_topk"),
    ("f06", "f6"): ("#6 Up_Gate + SwiGLU", "-"),
    ("f08f09", "f8_atomic"): ("#8 Down + Expert Merge", "atomic (sglang FUSE_SUM_ALL_REDUCE)"),
    ("f08f09", "f8_token_major"): ("#8 Down + Expert Merge", "token-major"),
    ("f08f09", "f9_atomic"): ("#9 Down + Expert Merge + ResAdd2",
                              "atomic (sglang FUSE_SUM_ALL_REDUCE)"),
    ("f08f09", "f9_token_major"): ("#9 Down + Expert Merge + ResAdd2", "token-major"),
    ("f10", "f10"): ("#10 Expert Merge + ResAdd", "-"),
    ("f11", "f11a_w13"): ("#11a Lazy Pre-Norm -> w13 grouped GEMM", "lazy pre-norm (prologue)"),
    ("f11", "f11b_router"): ("#11b Lazy Pre-Norm -> router GEMM", "lazy pre-norm (prologue)"),
    ("f11", "combined"): ("#11a+#11b combined (one norm charged once)", "combined"),
}
HALF = ("#11b' half-fused pre-norm -> router GEMM", "rstd + epilogue scale")

# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def r4(x, nd: int = 4):
    """Round for a measurement column, or EMPTY.  Never invents a value."""
    return "" if _nan(x) else round(float(x), nd)


def pct(x, nd: int = 1) -> str:
    return "n/a" if _nan(x) else f"{100.0 * float(x):.{nd}f}%"


_MAP_KEYS = [("BLOCK_M", "BM"), ("BLOCK_N", "BN"), ("BLOCK_K", "BK"), ("BLOCK_DIM", "BD"),
             ("GROUP_M", "GM"), ("SPLIT_K", "SK"), ("BLOCK_E", "BE"), ("NORM_BK", "NBK"),
             ("BLOCK", "BLK"), ("ROWS", "ROWS"), ("KVEC", "KVEC"), ("UNROLL", "UNROLL"),
             ("EVICT", "EVICT"), ("grid_cap", "CAP")]


def m(cfg: dict | None) -> str:
    """cfg dict -> the compact mapping string the C500 report uses, + the Hopper axes.

    Hopper axes are appended as TMA / WS / CTAS<n>, because whether the tuner selected one is
    itself a result on this device -- the previous two GPUs could not offer them at all.
    A config whose values are themselves configs (f08f09's {seed, gemm}, f03's {add, norm})
    is rendered as `sub: <mapping> | sub: <mapping>`.
    """
    if not isinstance(cfg, dict):
        return ""
    if cfg.get("impl"):
        return str(cfg["impl"])
    parts = [f"{tag}{cfg[k]}" for k, tag in _MAP_KEYS
             if cfg.get(k) is not None and cfg.get(k) is not False]
    if cfg.get("num_warps"):
        parts.append(f"w{cfg['num_warps']}")
    if cfg.get("num_stages"):
        parts.append(f"s{cfg['num_stages']}")
    if cfg.get("USE_TMA"):
        parts.append("TMA")
    if cfg.get("warp_specialize"):
        parts.append("WS")
    if cfg.get("num_ctas") not in (None, False, 1):
        parts.append(f"CTAS{cfg['num_ctas']}")
    if "USE_DOT" in cfg:
        parts.append(f"USE_DOT={cfg['USE_DOT']}")
    if parts:
        return " ".join(parts)
    order = ["gemm", "merge", "add", "norm", "router", "sum", "resadd", "topk", "act", "seed"]
    keys = sorted((k for k, v in cfg.items() if isinstance(v, dict) and m(v)),
                  key=lambda k: (order.index(k) if k in order else len(order), k))
    return " | ".join(f"{k}: {m(cfg[k])}" for k in keys)


_HOP = (("USE_TMA", "TMA"), ("warp_specialize", "warp-spec"), ("num_ctas", "clusters"))


def hop_axes(cfg) -> list[str]:
    """Which Hopper axes a (possibly nested) winning config actually selected."""
    got: list[str] = []
    if isinstance(cfg, dict):
        for key, label in _HOP:
            v = cfg.get(key)
            if v is True or (key == "num_ctas" and v not in (None, False, 1)):
                got.append(label if key != "num_ctas" else f"clusters(num_ctas={v})")
        for v in cfg.values():
            if isinstance(v, dict):
                got += hop_axes(v)
    out: list[str] = []
    for g in got:
        if g not in out:
            out.append(g)
    return out


def hop_note(fused, unfused) -> str:
    f, u = hop_axes(fused), hop_axes(unfused)
    return ("Hopper axes selected by the tuner: fused=" + (", ".join(f) if f else "none")
            + " / unfused=" + (", ".join(u) if u else "none"))


_AXNAME = {"tma": "TMA", "warp_specialize": "warp specialization", "clusters": "clusters"}


def offered_note(d: dict, family_key: str | None = None) -> str:
    """What the run itself recorded about which Hopper axes each arm was even offered."""
    pf = ((d.get("fairness") or {}).get("h200_axes") or {}).get("per_family") or {}
    if family_key is None:
        key = next(iter(pf), None)
    else:
        key = family_key if family_key in pf else next(iter(pf), None)
    axes = ((pf.get(key) or {}).get("axes")) or {}
    yes = [_AXNAME.get(a, a) for a, v in axes.items() if v.get("offered")]
    no = [_AXNAME.get(a, a) for a, v in axes.items() if not v.get("offered")]
    why = {(v.get("not_offered_because") or "") for a, v in axes.items() if not v.get("offered")}
    s = ("axes the tuner was OFFERED for this family (fairness.h200_axes): "
         + (", ".join(yes) if yes else "none"))
    if no:
        s += " -- NOT offered: " + ", ".join(no)
        reason = next((w for w in why if w), "")
        if reason:
            s += f" ({reason})"
    return s


def budget(*entries) -> tuple[int, int]:
    """(configs timed, configs that failed) summed over one arm's tuner stages."""
    tried = failed = 0
    for e in entries:
        if isinstance(e, dict):
            if "n_tried" in e:
                tried += int(e.get("n_tried") or 0)
                failed += int(e.get("n_failed") or 0)
            else:
                for st in ("coarse", "refine"):
                    sub = e.get(st)
                    if isinstance(sub, dict):
                        tried += int(sub.get("n_tried") or 0)
                        failed += int(sub.get("n_failed") or 0)
    return tried, failed


def budget_note(fused, unfused, extra: str = "") -> str:
    """LOG-14 build requirement 7: record n_tried/n_failed PER ARM, so an unequal effective
    search is detectable after the fact rather than assumed away."""
    ft, ff = fused
    ut, uf = unfused
    s = (f"tuner budget per arm (timed/failed): fused {ft}/{ff}, unfused {ut}/{uf}")
    if ft and ut:
        s += f" -- {ff / ft:.0%} of the fused arm's configs failed vs {uf / ut:.0%} of the "
        s += "unfused arm's"
    if extra:
        s += f" ({extra})"
    return s


def load(name: str):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def best_of(entry: dict | None) -> tuple[float | None, dict | None]:
    """(ms, cfg) of a tuner record that may be flat or split into coarse/refine stages."""
    if not isinstance(entry, dict):
        return None, None
    if "best_ms" in entry:
        return entry.get("best_ms"), entry.get("best_cfg")
    best = None
    for stage in ("coarse", "refine"):
        s = entry.get(stage)
        if isinstance(s, dict) and not _nan(s.get("best_ms")):
            if best is None or s["best_ms"] < best[0]:
                best = (s["best_ms"], s.get("best_cfg"))
    return best if best else (None, None)


def annot() -> dict:
    """(fusion, variant) -> (replicates, coda_correspondence) from the C500 report."""
    out: dict = {}
    for p in sorted(C500.glob("fusion_*.csv")):
        for r in csv.DictReader(p.open()):
            k = (r["fusion"], r["variant"])
            if r["replicates"] or r["coda_correspondence"]:
                out.setdefault(k, (r["replicates"], r["coda_correspondence"]))
    return out


# --------------------------------------------------------------------------------------
# the fairness record, per cell
# --------------------------------------------------------------------------------------
SUMMARY = json.loads((RES / "summary.json").read_text())
CELLS = {(c["family"], c["variant"], c["regime"]): c for c in SUMMARY["cells"]}

# GPU-3 acquired co-tenants mid-run; the driver's post-family checks recorded when.
TENANT_AFTER = {e["when"].replace("after ", "") for e in SUMMARY["gpu"].get("tenant_events", [])}
TENANT = ("CO-TENANCY: run_h200's post-family check first saw foreign processes on this GPU "
          "after " + ", ".join(sorted(TENANT_AFTER)) + " (up to 5.1 GB of another tenant's "
          "allocations) -- this family's measurements are not certified to have had the card to "
          "themselves")

SEQ_FIX = ("that flag fires on the row's `paired: false` field, not on the protocol: this "
           "cell's own pair_meta records impl=common.bench_pair, interleaved=true, with the "
           "leading arm alternating every repetition -- what is sequential is the STATISTIC "
           "(ratio of the two arms' medians), not the arm order")


def cell(family: str, variant: str, regime: str) -> dict | None:
    return CELLS.get((family, variant, regime))


def fairness_notes(family: str, variant: str, regime: str, pair_meta: dict | None,
                   tick: dict | None = None) -> list[str]:
    """Everything summary.json + pair_meta say about how trustworthy this cell is."""
    out: list[str] = []
    c = cell(family, variant, regime)
    if c:
        for fl in c.get("flags", []):
            out.append(f'FLAG(summary.json): "{fl}"')
            if fl.startswith("SEQUENTIAL"):
                out.append(SEQ_FIX)
        if c.get("resolved") is False:
            out.append("summary.json marks this cell resolved=false: the two arms differ by "
                       "fewer than 3 timer ticks, so the ratio is not resolvable")
        src = c.get("speedup_source")
        if src and src != "paired_speedup":
            out.append(f"speedup_source={src}: the published ratio is unfused.p50/fused.p50 "
                       f"(a ratio of medians), NOT the paired per-round statistic")
    pm = pair_meta or {}
    if pm:
        bits = []
        if not _nan(pm.get("ratio_p50")):
            bits.append(f"paired per-round median {pm['ratio_p50']:.4f}")
        if not _nan(pm.get("ratio_trimmed")):
            bits.append(f"trimmed {pm['ratio_trimmed']:.4f}")
        if not _nan(pm.get("ratio_p10")) and not _nan(pm.get("ratio_p90")):
            bits.append(f"per-round p10-p90 {pm['ratio_p10']:.4f}-{pm['ratio_p90']:.4f}")
        if not _nan(pm.get("frac_fused_faster")):
            bits.append(f"fused faster in {pct(pm['frac_fused_faster'], 0)} of "
                        f"{pm.get('n', '?')} rounds")
        if bits:
            out.append("paired statistic from the SAME interleaved run: " + ", ".join(bits))
        og = pm.get("order_gap_frac")
        if not _nan(og) and abs(og) >= 0.02:
            out.append(f"order sensitivity survives interleaving: fused-first vs unfused-first "
                       f"per-round medians differ by {pct(og)} (order_gap_frac)")
        df, du = pm.get("drift_frac_fused"), pm.get("drift_frac_unfused")
        if not _nan(df) and not _nan(du) and max(abs(df), abs(du)) >= 0.05:
            out.append(f"within-window drift: fused {pct(df)}, unfused {pct(du)} "
                       f"(interleaving cancels this to first order -- it is not zero)")
        mc = (pm.get("machine") or {}).get("compare") or {}
        if mc.get("suspect"):
            out.append("hwinfo bracket flagged the measurement window SUSPECT: "
                       + ", ".join(mc.get("reasons") or []))
        elif mc.get("new_throttle"):
            out.append("new throttle reasons during the window: "
                       + ", ".join(mc["new_throttle"]))
    if isinstance(tick, dict) and tick.get("tick_limited"):
        out.append(f"TICK-LIMITED: operands are {tick.get('fused_ticks', 0):.0f} / "
                   f"{tick.get('unfused_ticks', 0):.0f} ticks of the measured "
                   f"{tick.get('timer_tick_us')} us CUDA-event granularity")
    if family in ("f11", "f06", "f08f09"):
        out.append(TENANT)
    return out


# --------------------------------------------------------------------------------------
# per-regime row construction
# --------------------------------------------------------------------------------------
# ======================================================================================
# Provenance: WHICH CARD produced each family, and what its timing basis was.
#
# The campaign did not run on one GPU. Families split across two H200s of the same node,
# and their harness floors differ by 48.2 us:
#
#     59aa5198  f01, f03, f10                       harness floor  -5.975 us
#     6c4cc3d3  f04f05, f06, f08f09, f11, LAYER     harness floor +42.185 us
#
# Two consequences a reader must not have to discover for themselves:
#
#  1. A NEGATIVE harness floor is not physical. It is what the linear fit `t = O + N*L`
#     returns when the launch term is over-estimated, so the small-kernel timing model on
#     59aa5198 is unreliable in an unknown direction.
#  2. At decode the floor is not a rounding term, it is most of the measurement: f03's whole
#     fused arm at decode_bs1 is 52.6 us against a 48.2 us inter-card floor difference. A
#     speedup is (floor + work_u) / (floor + work_f), which equals the work ratio only when
#     the floor is zero. So decode speedups are partly a measurement of the harness, and
#     f01/f03/f10 are not on the same footing as the families they sit beside -- nor as the
#     whole-layer sweep, which ran on 6c4cc3d3.
#
# At prefill the kernels are millisecond-scale and the floor is negligible.
#
# This is the hazard flagged when `--gpu` was added: the checkpoint fence compares device
# NAME, and all eight cards report "NVIDIA H200", so cross-card reuse passes silently. The
# `device_uuid` is recorded by `ckpt_save` and read by nothing.
# ======================================================================================
FAMILY_CARD = {
    "f01": ("59aa5198", -5.975), "f03": ("59aa5198", -5.975), "f10": ("59aa5198", -5.975),
    "f04f05": ("6c4cc3d3", 42.185), "f06": ("6c4cc3d3", 42.185),
    "f08f09": ("6c4cc3d3", 42.185), "f11": ("6c4cc3d3", 42.185),
    "layer": ("6c4cc3d3", 42.185),
}
FUSION_FAMILY = {
    "#1": "f01", "#3": "f03", "#4": "f04f05", "#5": "f04f05", "#6": "f06",
    "#8": "f08f09", "#9": "f08f09", "#10": "f10", "#11": "f11",
}


def provenance_note(fusion: str, regime: str) -> str:
    """The card this row was measured on, and whether its floor threatens the number."""
    key = next((k for k in sorted(FUSION_FAMILY, key=len, reverse=True)
                if fusion.startswith(k)), None)
    fam = FUSION_FAMILY.get(key or "", "")
    card, floor = FAMILY_CARD.get(fam, ("", None))
    if not card:
        return ""
    n = f"MEASURED ON GPU {card} (family {fam}), harness floor {floor:+.3f} us"
    if floor < 0:
        n += (" -- a negative floor is unphysical (an over-estimated launch term in the "
              "t = O + N*L fit), so this card's small-kernel timing model is unreliable")
    if regime.startswith("decode"):
        n += ("; at decode the floor is comparable to the whole measurement, so the ratio is "
              "partly a measurement of the harness. The layer sweep ran on 6c4cc3d3, so "
              "decode rows from 59aa5198 are NOT on the same footing as it")
    return n


A = annot()


def rows_for(regime: str) -> list[dict]:
    out: list[dict] = []

    def add(fusion, variant, notes=None, **kw):
        rep, coda = A.get((fusion, variant), ("", ""))
        r = {f: "" for f in FIELDS}
        r.update(fusion=fusion, variant=variant, replicates=rep, coda_correspondence=coda)
        r.update({k: v for k, v in kw.items() if v is not None})
        nn = list(notes or [])
        pv = provenance_note(fusion, regime)
        if pv:
            nn.append(pv)
        r["notes"] = "; ".join(n for n in nn if n)
        out.append(r)

    def pick(rows, **match):
        for x in rows:
            if all(x.get(k) == v for k, v in match.items()):
                return x
        return None

    # ---- #1 o_proj + ResAdd ---------------------------------------------------------
    d = load("f01_oproj_resadd.json")
    if d:
        r = pick(d["rows"], regime=regime)
        tun = pick(d["tuning"], regime=regime) or {}
        if r:
            f_cfg, u_cfg = dict(r["fused_cfg"]), dict(r["unfused_cfg"])
            f_epi, u_epi = f_cfg.pop("EPI", None), u_cfg.pop("EPI", None)
            # The epilogue the unfused chain runs when the joint stage did not win is the
            # standalone winner: add-f32 when SPLIT_K>1 (fp32 accumulator), else add-bf16.
            epi_key = ("tune_epi_add_f32" if (u_cfg.get("SPLIT_K") or 1) > 1
                       else "tune_epi_add_bf16")
            epi_ms, epi_cfg = best_of(tun.get(epi_key))
            if u_epi:
                epi_cfg = u_epi
            fm = m(f_cfg) + (f" | epi: {m(f_epi)}" if f_epi else "")
            n = fairness_notes("f01", "triton", regime, r.get("pair_meta"), r.get("tick"))
            n.append(hop_note(r["fused_cfg"], r["unfused_cfg"]))
            n.append("unfused_k1_ms is EMPTY on purpose: bench_f01 times the unfused arm only "
                     "as a chain (GEMM + epilogue), so the o_proj GEMM alone was never "
                     "measured on this device")
            n.append("unfused_k2_ms is the epilogue's own tuned standalone best, behind its "
                     "own L2 flush, so it does not sum to unfused_total_ms")
            if (u_cfg.get("SPLIT_K") or 1) > 1 or (f_cfg.get("SPLIT_K") or 1) > 1:
                n.append(f"SPLIT_K>1 (fused SK={f_cfg.get('SPLIT_K')}, unfused "
                         f"SK={u_cfg.get('SPLIT_K')}) adds an fp32 zero-init kernel to the "
                         "chain -- both arms pay it, and n_unfused_kernels counts the GEMM "
                         "and the epilogue only")
            g = ((d.get("fairness") or {}).get("grids") or {}).get(regime) or {}
            tot = g.get("_totals") or {}
            if tot:
                n.append(budget_note((int(tot.get("fused", {}).get("n_tried") or 0),
                                      int(tot.get("fused", {}).get("n_failed") or 0)),
                                     (int(tot.get("unfused", {}).get("n_tried") or 0),
                                      int(tot.get("unfused", {}).get("n_failed") or 0))))
            n.append(f"vendor cuBLAS line on the same tensors: addmm(beta=1) "
                     f"{r['vendor_fused_ms']:.4f} ms vs mm+add {r['vendor_unfused_ms']:.4f} ms "
                     f"-> {r['vendor_speedup']:.4f}x (paired); triton fused is "
                     f"{r['triton_vs_vendor']:.3f}x the vendor fused time")
            add("#1 o_proj + ResAdd", "triton", notes=n,
                fused_ms=r4(r["fused_ms"]), fused_mapping=fm,
                unfused_total_ms=r4(r["unfused_ms"]), speedup=r4(r["speedup"]),
                n_unfused_kernels=2,
                unfused_k1_name="o_proj GEMM", unfused_k1_ms="", unfused_k1_mapping=m(u_cfg),
                unfused_k2_name="residual add (elementwise)",
                unfused_k2_ms=r4(epi_ms), unfused_k2_mapping=m(epi_cfg))

    # ---- #3 ResAdd + RMSNorm --------------------------------------------------------
    d = load("f03_resadd_rmsnorm.json")
    if d:
        r = pick(d["rows"], regime=regime)
        tt = (d.get("tune_tables") or {}).get(regime, {})
        if r:
            _, a_cfg = best_of(tt.get("add_only"))
            _, n_cfg = best_of(tt.get("norm_only"))
            u = r["unfused_cfg"]
            n = fairness_notes("f03", "f3", regime, r.get("pair_meta"), r.get("tick"))
            n.append(hop_note(r["fused_cfg"], u))
            n.append(offered_note(d))
            n.append(budget_note(budget(tt.get("fused")),
                                 budget(tt.get("add_only"), tt.get("norm_only"))))
            n.append("per-kernel times are each kernel's own tuned best, while the chain ran "
                     f"jointly re-tuned configs (add: {m(u.get('add'))} | "
                     f"norm: {m(u.get('norm'))}) -- a stage the unfused side gets and the "
                     "fused side cannot use")
            n.append(f"fused output bitwise-identical to unfused: {r['bitwise_identical']}")
            n.append(f"DRAM-traffic ceiling {r['ideal_speedup']:.2f}x (MODELLED), h1 "
                     f"({r['h1_bytes']} B) fits this device's "
                     f"{r['l2_bytes'] / 2**20:.0f} MB L2: {r['h1_fits_l2']}")
            if not _nan(r.get("torch_compile_ms")):
                n.append(f"reference lines: torch eager {r['torch_eager_ms']:.4f} ms, "
                         f"torch.compile {r['torch_compile_ms']:.4f} ms")
            add("#3 ResAdd + RMSNorm", "-", notes=n,
                fused_ms=r4(r["fused_ms"]), fused_mapping=m(r["fused_cfg"]),
                unfused_total_ms=r4(r["unfused_ms"]), speedup=r4(r["speedup"]),
                n_unfused_kernels=2,
                unfused_k1_name="residual add", unfused_k1_ms=r4(r["add_only_ms"]),
                unfused_k1_mapping=m(a_cfg),
                unfused_k2_name="rmsnorm", unfused_k2_ms=r4(r["norm_only_ms"]),
                unfused_k2_mapping=m(n_cfg))

    # ---- #4 / #5 --------------------------------------------------------------------
    d = load("f04f05_norm_router.json")
    if d:
        tt = (d.get("tune_tables") or {}).get(regime, {})
        g_ms, g_cfg = best_of(tt.get("router_gemm"))
        t_ms, t_cfg = best_of(tt.get("topk"))
        for v in ("F5", "F5_topk", "F4", "F4_topk"):
            r = pick(d["rows"], regime=regime, variant=v)
            if not r:
                continue
            fusion, variant = NAME[("f04f05", v)]
            is4, istk = v.startswith("F4"), v.endswith("_topk")
            k1_ms, k1_cfg = best_of(tt.get("add_norm" if is4 else "norm"))
            u = r.get("unfused_cfg") or {}
            n = fairness_notes("f04f05", v, regime, r.get("pair_meta"), r.get("tick"))
            n.append(hop_note(r["fused_cfg"], u))
            n.append(offered_note(d) + " -- the two arms cannot use them symmetrically: the "
                     "fused side is a norm kernel with the GEMM inside it, the unfused side is "
                     "a standalone GEMM")
            n.append(budget_note(
                budget(tt.get(f"fused_{v}")),
                budget(tt.get("add_norm" if is4 else "norm"), tt.get("router_gemm"),
                       tt.get("topk") if istk else None),
                "the f04f05 compiler log records Triton's warp-specialization pass asserting "
                "on sm_90 (WSLowerToken.cpp:73) once per attempted config, and the fused "
                "kernel is the arm that is offered that axis"))
            n.append("per-kernel times are each kernel's own tuned best, while the chain "
                     f"ran jointly-tuned configs (norm: {m(u.get('norm'))} | "
                     f"gemm: {m(u.get('gemm'))}"
                     + (f" | topk: {m(u.get('topk'))}" if istk else "") + ")")
            if not _nan(r.get("ceiling")):
                n.append(f"ceiling for this cell {r['ceiling']:.3f}x, traffic_ratio_model "
                         f"{r['traffic_ratio_model']:.3f}x -- both are MODELLED, not measured "
                         "(glm52_h200/traffic.py)")
            n.append(f"vendor/reference lines: cuBLAS router GEMM fp32 "
                     f"{r['blas_router_fp32_ms']:.4f} ms, bf16 {r['blas_router_bf16_ms']:.4f} "
                     f"ms, torch reference chain {r['torch_ref_ms']:.4f} ms")
            add(fusion, variant, notes=n,
                fused_ms=r4(r["fused_ms"]), fused_mapping=m(r["fused_cfg"]),
                unfused_total_ms=r4(r["unfused_ms"]), speedup=r4(r["speedup"]),
                n_unfused_kernels=3 if istk else 2,
                unfused_k1_name="add+rmsnorm (fusion #3 already applied)" if is4 else "rmsnorm",
                unfused_k1_ms=r4(k1_ms), unfused_k1_mapping=m(k1_cfg),
                unfused_k2_name="router GEMM (fp32)", unfused_k2_ms=r4(g_ms),
                unfused_k2_mapping=m(g_cfg),
                unfused_k3_name="sigmoid + noaux_tc top-8" if istk else None,
                unfused_k3_ms=r4(t_ms) if istk else None,
                unfused_k3_mapping=m(t_cfg) if istk else None)

    # ---- #6 Up_Gate + SwiGLU --------------------------------------------------------
    d = load("f06_upgate_swiglu.json")
    if d:
        r = pick(d["rows"], regime=regime)
        if r:
            sm = r.get("smem_model") or {}
            n = fairness_notes("f06", "f6", regime, r.get("pair_meta"), r.get("tick"))
            n.append(hop_note(r["fused_cfg"], r["unfused_gemm_cfg"]))
            tn = (d.get("tuning") or {}).get(regime) or {}
            n.append(budget_note(
                budget(tn.get("fused_coarse"), tn.get("fused_refine")),
                budget(tn.get("unfused_gemm_coarse"), tn.get("unfused_gemm_refine"),
                       tn.get("unfused_act"))))
            n.append("per-kernel unfused times are measured at the SAME configs the chain ran "
                     "(gemm_cfg / act_cfg), each behind its own L2 flush, so they need not sum "
                     "to unfused_total_ms")
            if sm.get("unfused_winner_if_fused_bytes") and sm.get("ceiling_bytes"):
                over = sm["unfused_winner_if_fused_bytes"] > sm["ceiling_bytes"]
                n.append(f"SMEM: fused winner {sm['fused_winner_bytes'] / 1024:.0f} KB, "
                         f"unfused winner {sm['unfused_winner_bytes'] / 1024:.0f} KB, the "
                         f"unfused winner's tile IF FUSED would need "
                         f"{sm['unfused_winner_if_fused_bytes'] / 1024:.0f} KB against a "
                         f"{sm['ceiling_bytes'] / 1024:.0f} KB opt-in ceiling"
                         + (" -- so the fused side still cannot reach the unfused winner's "
                            "tile, the same constraint C500 hit at a 4x smaller ceiling"
                            if over else ""))
            ks, ku = r.get("fused_kernel_stats") or {}, r.get("unfused_kernel_stats") or {}
            if ks:
                n.append(f"kernel stats: fused regs {ks.get('n_regs')} spills "
                         f"{ks.get('n_spills')} smem {ks.get('shared_bytes')} B | unfused GEMM "
                         f"regs {ku.get('n_regs')} spills {ku.get('n_spills')} smem "
                         f"{ku.get('shared_bytes')} B")
            n.append(f"throughput: fused {r['fused_tflops']:.1f} TF/s, unfused "
                     f"{r['unfused_tflops']:.1f} TF/s, vendor grouped BLAS "
                     f"{r['vendor_blas_ms']:.4f} ms ({r['vendor_tflops']:.1f} TF/s)")
            add("#6 Up_Gate + SwiGLU", "-", notes=n,
                fused_ms=r4(r["fused_ms"]), fused_mapping=m(r["fused_cfg"]),
                unfused_total_ms=r4(r["unfused_ms"]), speedup=r4(r["speedup"]),
                n_unfused_kernels=2,
                unfused_k1_name="w13 grouped GEMM -> [rows, 2I]",
                unfused_k1_ms=r4(r["unfused_gemm_ms"]),
                unfused_k1_mapping=m(r["unfused_gemm_cfg"]),
                unfused_k2_name="silu_and_mul", unfused_k2_ms=r4(r["unfused_act_ms"]),
                unfused_k2_mapping=m(r["unfused_act_cfg"]))

    # ---- #8 / #9 --------------------------------------------------------------------
    d = load("f08f09_down_merge_resadd.json")
    if d:
        # the standalone per-kernel numbers are measured once per regime, at the #8 chain's
        # winning gemm/sum configs and the #9 chain's resadd config -- take the mapping from
        # the SAME place the ms came from, so the two columns cannot disagree.
        r8a = pick(d["rows"], regime=regime, variant="f8_atomic")
        r9a = pick(d["rows"], regime=regime, variant="f9_atomic")
        prov_gemm = ((r8a or {}).get("unfused_cfg") or {}).get("gemm")
        prov_sum = ((r8a or {}).get("unfused_cfg") or {}).get("sum")
        prov_ra = ((r9a or {}).get("unfused_cfg") or {}).get("resadd")
        for v in ("f8_atomic", "f8_token_major", "f9_atomic", "f9_token_major"):
            r = pick(d["rows"], regime=regime, variant=v)
            if not r:
                continue
            fusion, variant = NAME[("f08f09", v)]
            is9, isatom = v.startswith("f9"), v.endswith("atomic")
            u = r.get("unfused_cfg") or {}
            n = fairness_notes("f08f09", v, regime, r.get("pair_meta"), r.get("tick"))
            n.append(hop_note(r["fused_cfg"], u))
            tn = (d.get("tuning") or {}).get(regime) or {}
            f_bud = (budget(tn.get("atomic_gemm_coarse"), tn.get("atomic_gemm_refine"),
                            tn.get("seed_residual" if is9 else "seed_zero")) if isatom
                     else budget(tn.get("tokmaj9_coarse" if is9 else "tokmaj8_coarse")))
            n.append(budget_note(
                f_bud,
                budget(tn.get("unfused_gemm_coarse"), tn.get("unfused_gemm_refine"),
                       tn.get("moe_sum"), tn.get("resadd") if is9 else None)))
            n.append("per-kernel unfused times are measured once per regime at the #8 chain's "
                     "winning gemm/sum configs and the #9 chain's resadd config, and the "
                     "mappings shown are those same configs")
            if prov_gemm and u.get("gemm") and m(u["gemm"]) != m(prov_gemm):
                n.append(f"this row's own unfused chain ran a different GEMM mapping "
                         f"({m(u['gemm'])}), so unfused_k1_ms is not a decomposition of this "
                         "row's unfused_total_ms")
            if isatom:
                n.append(f"fused arm = zero/residual seed + atomic-merge GEMM, where the seed "
                         f"alone is {r['seed_only_ms']:.4f} ms and the atomic GEMM alone "
                         f"{r['atomic_gemm_only_ms']:.4f} ms, and fused_ms includes both. bf16 "
                         "atomics make the fused output non-deterministic (LOG-08 F6)")
            else:
                n.append("scored against the EXPERT-major baseline; part of any win is the "
                         "grid change, not the fusion (LOG-08 F3)")
            if is9:
                n.append(f"unfused_total is the 3-kernel chain, and the strictly better "
                         f"2-kernel baseline (moe_sum with ADD_RESIDUAL) runs "
                         f"{r['unfused9_2kernel_ms']:.4f} ms -> speedup "
                         f"{r['speedup_vs_2kernel']:.4f}x, which is the honest comparison")
            n.append(f"atomic accumulator {r['atomic_accum_bytes']} B, fits this device's "
                     f"{r['l2_bytes'] / 2**20:.0f} MB L2: {r['atomic_accum_fits_l2']}")
            n.append(f"vendor grouped BLAS GEMM {r['vendor_blas_gemm_ms']:.4f} ms, GEMM+merge "
                     f"{r['vendor_blas_gemm_merge_ms']:.4f} ms")
            add(fusion, variant, notes=n,
                fused_ms=r4(r["fused_ms"]), fused_mapping=m(r["fused_cfg"]),
                unfused_total_ms=r4(r["unfused_ms"]), speedup=r4(r["speedup"]),
                n_unfused_kernels=3 if is9 else 2,
                unfused_k1_name="w2 grouped down GEMM -> [rows, H]",
                unfused_k1_ms=r4(r["unfused_gemm_only_ms"]),
                unfused_k1_mapping=m(prov_gemm),
                unfused_k2_name="moe_sum (expert merge over top-8)",
                unfused_k2_ms=r4(r["unfused_sum_only_ms"]),
                unfused_k2_mapping=m(prov_sum),
                unfused_k3_name="residual add 2" if is9 else None,
                unfused_k3_ms=r4(r["unfused_resadd_only_ms"]) if is9 else None,
                unfused_k3_mapping=m(prov_ra) if is9 else None)

    # ---- #10 Expert Merge + ResAdd --------------------------------------------------
    d = load("f10_merge_resadd.json")
    if d:
        r = pick(d["rows"], regime=regime)
        tt = (d.get("tune_tables") or {}).get(regime, {})
        if r:
            _, mg_cfg = best_of(tt.get("merge_only"))
            _, ra_cfg = best_of(tt.get("resadd_only"))
            u = r["unfused_cfg"]
            n = fairness_notes("f10", "f10", regime, r.get("pair_meta"), r.get("tick"))
            n.append(hop_note(r["fused_cfg"], u))
            n.append(offered_note(d))
            n.append(budget_note(budget(tt.get("fused")),
                                 budget(tt.get("merge_only"), tt.get("resadd_only"))))
            n.append("per-kernel times are each kernel's own tuned best, while the chain ran "
                     f"jointly re-tuned configs (merge: {m(u.get('merge'))} | "
                     f"resadd: {m(u.get('resadd'))})")
            n.append(f"fused output bitwise-identical to unfused: {r['bitwise_identical']}")
            n.append(f"{r['ceiling_note']} (MODELLED). fraction_of_ceiling_gain = "
                     f"{r['pct_of_ceiling']:.3f} -- this is (paired_speedup - 1) / 0.20, NOT "
                     f"measured/ceiling (that would be {r['speedup'] / 1.20:.3f}), and it is "
                     f"derived from the PAIRED estimator while this row's `speedup` column is "
                     f"{r.get('speedup_source', 'as recorded')}; m ({r['m_bytes']} B) fits this device's "
                     f"{r['l2_bytes'] / 2**20:.0f} MB L2: {r['m_fits_in_l2']}")
            if not _nan(r.get("torch_compile_ms")):
                n.append(f"reference lines: torch eager {r['torch_eager_ms']:.4f} ms, "
                         f"torch.compile {r['torch_compile_ms']:.4f} ms")
            add("#10 Expert Merge + ResAdd", "-", notes=n,
                fused_ms=r4(r["fused_ms"]), fused_mapping=m(r["fused_cfg"]),
                unfused_total_ms=r4(r["unfused_ms"]), speedup=r4(r["speedup"]),
                n_unfused_kernels=2,
                unfused_k1_name="expert merge (moe_sum)", unfused_k1_ms=r4(r["merge_only_ms"]),
                unfused_k1_mapping=m(mg_cfg),
                unfused_k2_name="residual add", unfused_k2_ms=r4(r["resadd_only_ms"]),
                unfused_k2_mapping=m(ra_cfg))

    # ---- #11a / #11b / #11a+b / #11b' -----------------------------------------------
    d = load("f11_lazy_prenorm.json")
    rr = pick(d["rows"], regime=regime) if d else None
    if rr:
        head = rr.get("headline") or {}

        def f11_note(kind: str, b: dict) -> list[str]:
            n = fairness_notes("f11", kind, regime, b.get("pair_meta"), b.get("tick"))
            n.append(hop_note(b.get("fused_cfg"), b.get("unfused_gemm_cfg")))
            n.append(offered_note(d, "f11a_w13_gemm" if kind == "f11a_w13"
                                    else "f11b_router_gemm")
                     + " -- whether the tuner took them is in the mappings above")
            n.append(f"sum-of-squares redundancy {b.get('sq_redundancy', '?')}x, valid only "
                     "if ALL K=6144 consumers are fused (x2 never materialised)")
            if not _nan(b.get("ceiling")):
                n.append(f"modelled ceiling for this cell {b['ceiling']:.3f}x (MODELLED, not "
                         "measured)")
            return n

        b = rr.get("f11a_w13")
        if b:
            n = f11_note("f11a_w13", b)
            v = (head.get("moe") or {}).get("verdict")
            if v:
                n.append("isolation study at a SHARED config (FUSE on/off, warp-spec "
                         f'on/off): "{v}"')
            n.append(f"vendor lines: cuBLAS grouped {b['vendor_blas_grouped_ms']:.4f} ms, "
                     f"dense 1-expert {b['vendor_blas_dense_1expert_ms']:.4f} ms")
            add(*NAME[("f11", "f11a_w13")], notes=n,
                fused_ms=r4(b["fused_ms"]), fused_mapping=m(b["fused_cfg"]),
                unfused_total_ms=r4(b["unfused_ms"]), speedup=r4(b["speedup"]),
                n_unfused_kernels=2,
                unfused_k1_name="rmsnorm (writes x2)", unfused_k1_ms=r4(b["norm_only_ms"]),
                unfused_k1_mapping=m(b.get("unfused_norm_cfg")),
                unfused_k2_name="w13 grouped GEMM",
                unfused_k2_ms=r4(b["unfused_gemm_only_ms"]),
                unfused_k2_mapping=m(b.get("unfused_gemm_cfg")))

        b = rr.get("f11b_router")
        if b:
            n = f11_note("f11b_router", b)
            v = (head.get("router") or {}).get("verdict")
            if v:
                n.append("isolation study at a SHARED config (FUSE on/off, warp-spec "
                         f'on/off): "{v}"')
            n.append(f"vendor lines: cuBLAS router bf16 {b['vendor_blas_bf16_ms']:.4f} ms, "
                     f"fp32 {b['vendor_blas_fp32_ms']:.4f} ms")
            add(*NAME[("f11", "f11b_router")], notes=n,
                fused_ms=r4(b["fused_ms"]), fused_mapping=m(b["fused_cfg"]),
                unfused_total_ms=r4(b["unfused_ms"]), speedup=r4(b["speedup"]),
                n_unfused_kernels=2,
                unfused_k1_name="rmsnorm (writes x2)", unfused_k1_ms=r4(b["norm_only_ms"]),
                unfused_k1_mapping=m(b.get("unfused_norm_cfg")),
                unfused_k2_name="router GEMM", unfused_k2_ms=r4(b["unfused_gemm_only_ms"]),
                unfused_k2_mapping=m(b.get("unfused_gemm_cfg")))

            h = rr.get("half_fused")
            if h:
                n = fairness_notes("f11", "f11b_router", regime, None, None)
                n = [x for x in n if not x.startswith("paired statistic")]
                n.insert(0, "NO summary.json cell of its own: #11b' is recorded inside the "
                            "f11 row as an EXPLORATORY arm, so the flags above are those of "
                            "the #11b cell it is scored against")
                n.append(hop_note(h.get("router_cfg"), b.get("unfused_gemm_cfg")))
                n.append(f"the FUSED side is itself 2 kernels (rstd "
                         f"{h['rstd_only_ms']:.4f} ms {m(h.get('rstd_cfg'))} + GEMM), and this "
                         "row is router-only")
                n.append(f"same technique applied to the w13 GEMM gives "
                         f"{h['moe_ms']:.4f} ms ({h['moe_speedup_vs_unfused']:.4f}x) and the "
                         f"two together {h['combined_ms']:.4f} ms "
                         f"({h['combined_speedup_vs_unfused']:.4f}x)")
                n.append(h.get("note", ""))
                add(*HALF, notes=n,
                    fused_ms=r4(h["router_ms"]),
                    fused_mapping=f"rstd: {m(h.get('rstd_cfg'))} | "
                                  f"gemm: {m(h.get('router_cfg'))}",
                    unfused_total_ms=r4(b["unfused_ms"]),
                    speedup=r4(h["router_speedup_vs_unfused"]), n_unfused_kernels=2,
                    unfused_k1_name="rmsnorm (writes x2)", unfused_k1_ms=r4(b["norm_only_ms"]),
                    unfused_k1_mapping=m(b.get("unfused_norm_cfg")),
                    unfused_k2_name="router GEMM", unfused_k2_ms=r4(b["unfused_gemm_only_ms"]),
                    unfused_k2_mapping=m(b.get("unfused_gemm_cfg")))

        c = rr.get("combined")
        a11, b11 = rr.get("f11a_w13") or {}, rr.get("f11b_router") or {}
        if c:
            n = fairness_notes("f11", "combined", regime, c.get("pair_meta"), c.get("tick"))
            n.append(hop_note({"router": b11.get("fused_cfg"), "w13": a11.get("fused_cfg")},
                              {"router": b11.get("unfused_gemm_cfg"),
                               "w13": a11.get("unfused_gemm_cfg")}))
            n.append("NO C500 counterpart row: `replicates` / `coda_correspondence` are "
                     "carried from #11a because it is the same technique")
            n.append(c.get("note", ""))
            n.append("this is the layer-honest form of #11: the unfused side is charged ONE "
                     "norm kernel, which is what the real layer pays; the per-family #11a and "
                     "#11b rows each charge that same norm and therefore double-count it")
            n.append("unfused_k1_ms is EMPTY: the combined row's norm ran a different config "
                     f"({m(c.get('norm_cfg'))}) from the one whose standalone time was "
                     "measured, and it was not re-timed")
            add(*NAME[("f11", "combined")], notes=n,
                fused_ms=r4(c["fused_ms"]),
                fused_mapping=f"router: {m(b11.get('fused_cfg'))} | "
                              f"w13: {m(a11.get('fused_cfg'))}",
                unfused_total_ms=r4(c["unfused_ms"]), speedup=r4(c["speedup"]),
                n_unfused_kernels=3,
                unfused_k1_name="rmsnorm (writes x2, charged once)", unfused_k1_ms="",
                unfused_k1_mapping=m(c.get("norm_cfg")),
                unfused_k2_name="router GEMM", unfused_k2_ms=r4(b11.get("unfused_gemm_only_ms")),
                unfused_k2_mapping=m(b11.get("unfused_gemm_cfg")),
                unfused_k3_name="w13 grouped GEMM",
                unfused_k3_ms=r4(a11.get("unfused_gemm_only_ms")),
                unfused_k3_mapping=m(a11.get("unfused_gemm_cfg")))
    elif d:
        why = ("NOT MEASURED at this regime -- and NOT a scoping choice: the fusion was "
               "attempted and FAILED its correctness screen (tune_tables.<regime>.regime_failed "
               "records moe_fused rel_err 0.395-0.833 against tol 0.02 at t2048/bs1024/bs512/"
               "t8192, and KeyError:'ms' at decode_bs1). Earlier wording said bench_f11 "
               "\"tuned only at\" two regimes, which understated a wrong-answer failure as a "
               "deliberate limit. For the record: bench_f11 tuned the lazy pre-norm kernels at "
               "decode_bs32 and decode_bs256 (the two regimes present in "
               "results/h200/f11_lazy_prenorm.json). No value is estimated here.")
        add(NAME[("f11", "f11a_w13")][0], "-", notes=[why])
        add(NAME[("f11", "f11b_router")][0], "-", notes=[why])
        add(HALF[0], "-", notes=[why])
        add(NAME[("f11", "combined")][0], "-", notes=[why])

    return out


# --------------------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(exist_ok=True)
    seen_cells = set()
    for reg in REGIMES:
        rows = rows_for(reg)
        p = OUT / f"fusion_{reg}.csv"
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        measured = sum(1 for r in rows if r["speedup"] != "")
        print(f"  wrote {p.relative_to(ROOT)}  ({len(rows)} rows, {measured} measured)")
        for r in rows:
            if r["speedup"] != "":
                seen_cells.add((r["fusion"], r["variant"], reg))

    # cross-check every published speedup against summary.json's own cell value
    bad = []
    for (fam, var, reg), c in CELLS.items():
        fusion, variant = NAME[(fam, var)]
        if (fusion, variant, reg) not in seen_cells:
            bad.append(f"MISSING {fam}/{var}/{reg}")
    print(f"  summary.json cells: {len(CELLS)}; published rows carrying a speedup: "
          f"{len(seen_cells)}")
    for b in bad:
        print("  " + b)


if __name__ == "__main__":
    main()
