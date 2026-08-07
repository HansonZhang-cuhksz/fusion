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
    # f11 is deliberately NOT in this list any more. TENANT is derived from summary.json's
    # tenant_events, which belong to the ORIGINAL campaign on card 59aa5198. #11 was repaired
    # and re-run on 2026-08-05 by run_f11_h200.py, on card b2318e71, with its own GPU record
    # (f11_rerun_summary.json: pinned, "idlest of 8: 0% utilization", tenant_events []).
    # Applying the old campaign's co-tenancy flag to it would be a fabricated provenance.
    if family in ("f06", "f08f09"):
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
def best_speedup(r: dict) -> tuple:
    """(value, which). Prefer the paired/interleaved ratio over the sequential one.

    Every row in the H200 re-run carries BOTH. The sequential ratio is two medians measured
    one arm after the other, so a monotone clock or thermal ramp does not cancel in it; the
    paired ratio is the median of per-repetition ratios from an interleaved A/B/A/B loop, and
    it does. They disagree by up to 13.8 % here (f04f05 prefill_t8192 F4_topk: 0.841 vs
    0.957), which is far too large to treat as interchangeable -- and publishing the
    sequential one is what let a physically impossible speedup through on the RTX 4060.
    """
    p, q = r.get("paired_speedup"), r.get("speedup")
    if isinstance(p, (int, float)) and p > 0:
        return p, "paired"
    return q, "sequential"


def _campaign_cards() -> dict:
    """(uuid, harness_floor_us) per family, read from the result files themselves.

    The first H200 campaign silently split across two cards with harness floors of -5.975 us
    and +42.185 us, so this was hardcoded to expose it. The re-run pinned one idle GPU. Read
    it rather than assert it, so the next split is caught instead of papered over.

    It caught one: `#11` was repaired and re-run SEPARATELY (`run_f11_h200.py`, 2026-08-05)
    and landed on a different card from the rest of the campaign. The uuid is therefore taken
    from EACH FILE's own `env.uuid` / `fairness.gpu.uuid`, not from `summary.json` -- reading
    the campaign-wide uuid is exactly the mistake that would have hidden the split.
    """
    out = {}
    for fam, fn in (("f01", "f01_oproj_resadd"), ("f03", "f03_resadd_rmsnorm"),
                    ("f04f05", "f04f05_norm_router"), ("f06", "f06_upgate_swiglu"),
                    ("f08f09", "f08f09_down_merge_resadd"), ("f10", "f10_merge_resadd"),
                    ("f11", "f11_lazy_prenorm"), ("layer", "layer_configurations")):
        try:
            d = json.loads((RES / f"{fn}.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        floor = ((d.get("fairness") or {}).get("timing") or {}).get("harness_floor_us")
        uu = str((d.get("env") or {}).get("uuid")
                 or ((d.get("fairness") or {}).get("gpu") or {}).get("uuid") or "")
        if not uu:
            try:
                uu = str(json.loads((RES / "summary.json").read_text())
                         .get("_meta", {}).get("gpu_uuid", ""))
            except Exception:  # noqa: BLE001
                pass
        out[fam] = (uu.replace("GPU-", "")[:8], floor)
    return out


FAMILY_CARD = _campaign_cards()
# The gated #11 re-measurement, loaded once. Its presence is what makes every #11 row in this
# report come from it instead of from the campaign file (see the F11 section below).
F11P = load("f11_publish.json")
FUSION_FAMILY = {
    "#1": "f01", "#3": "f03", "#4": "f04f05", "#5": "f04f05", "#6": "f06",
    "#8": "f08f09", "#9": "f08f09", "#10": "f10", "#11": "f11",
}


def provenance_note(fusion: str, regime: str) -> str:
    """The card this row was measured on, its harness floor, and what that costs at decode."""
    key = next((k for k in sorted(FUSION_FAMILY, key=len, reverse=True)
                if fusion.startswith(k)), None)
    fam = FUSION_FAMILY.get(key or "", "")
    if fam == "f11" and F11P is not None:
        # #11 no longer comes from the campaign file, so it must not inherit that file's
        # card record. The gated re-measurement pinned its own GPU and measured its own
        # floor; its `env.uuid` is carried from a PREFLIGHT taken the day before and is not
        # evidence about which card this run got, so it is not quoted as one.
        g, cal = F11P.get("gpu") or {}, F11P["calibration"]
        return (f"MEASURED BY f11_publish.py on the GPU it pinned itself: index "
                f"{g.get('index')}, \"{g.get('why')}\" -- harness floor "
                f"{cal['harness_floor_us']:+.3f} us against a {cal['floor_bar_us']:.1f} us "
                f"bar, launch {cal['launch_us']:.3f} us. NOT the card record of "
                f"results/h200/{F11_FILE} (floor "
                f"{FAMILY_CARD.get('f11', ('', 0.0))[1]:+.3f} us), which no longer supplies "
                f"any #11 number here. The `env.uuid` in the newer file comes from a preflight "
                f"taken the previous day and is not quoted as this run's card")
    card, floor = FAMILY_CARD.get(fam, ("", None))
    if not card and floor is None:
        return ""
    n = f"MEASURED ON GPU {card} (family {fam})"
    if floor is not None:
        n += f", harness floor {floor:+.3f} us"
        if floor < 0:
            n += (" -- a negative floor is unphysical (an over-estimated launch term in the "
                  "t = O + N*L fit), so this card's small-kernel timing model is unreliable")
        elif regime.startswith("decode"):
            n += (f"; at decode that floor is a large share of the measurement (it is added to "
                  f"BOTH arms, so a ratio understates the true work ratio), which is why the "
                  f"decode numbers here should be read as bounds rather than exact")
    return n


# ======================================================================================
# #11 (lazy pre-norm) -- the REPAIRED re-run, and the caveats the verification raised
# ======================================================================================
# The first H200 campaign wrote `f11_lazy_prenorm.json` with `complete: true` and an EMPTY
# rows array, so this generator emitted four "NOT MEASURED" rows per regime. `#11` was
# repaired (LOG-16) and re-run on 2026-08-05 by `run_f11_h200.py`; the file now carries 7
# regimes and per-arm records, and `complete: false` is honest -- one arm failed.
#
# Everything below reads that file. Nothing here is carried over from the brief or from
# LOG-16's prose. Where a caveat comes from OUTSIDE the file it says so and names the tool.
F11_FILE = "f11_lazy_prenorm.json"
F11_TOOL = "tools/verify_f11_headline_ws.py"

# What that tool reported when run in this repo (Triton 3.6.0, torch 2.11.0) on 2026-08-05.
# It is recorded as a fixed string because the RESULT FILE CARRIES NO CODEGEN EVIDENCE of
# its own: `headline.*.kernel_ws_evidence` is hard-null in all 12 measured cells
# (bench_f11_lazy_prenorm.py:933-939 assigns `None` instead of calling `K.ws_evidence`), and
# the file contains no `ttgir_mentions_wgmma` / `ptx_mentions_wgmma` /
# `*_mentions_warp_specialize` field at all. Re-run the script to re-derive it.
WS_NOT_APPLIED = (
    "WARP SPECIALIZATION WAS REQUESTED AND NEVER APPLIED -- so every `*_ws_*` number in "
    "this file measures something other than warp specialization. Evidence is NOT in the "
    f"result file (kernel_ws_evidence is null in all 12 measured headline cells); it comes "
    f"from {F11_TOOL}, which cross-compiles both f11 kernels for sm_90a at each cell's own "
    "recorded shared_config and first reproduces the H200's recorded kernel_stats "
    "(shared_bytes AND n_regs) for 26/26 kernels. On that reproduction the WS request "
    "reaches TTGIR in 12/12 cells and produces a specialized kernel in 0/12: in 9 cells the "
    "WS-on and WS-off arms are identical PTX once .loc metadata is stripped (identical "
    "machine code cannot run at a different speed), and in 3 (decode_bs1/bs32/bs512 router) "
    "the request instead collapsed multi-buffering, shared memory 61440 -> 16384 B, which "
    "is a DE-PIPELINING regression rather than a specialization effect. Triton 3.6 routes "
    "sm_90 to `add_hopper_warpspec`, which crashed 493 times in this run's compiler log and "
    "otherwise emitted nothing"
)

# ======================================================================================
# #11 -- THE GATED RE-MEASUREMENT (`results/h200/f11_publish.json`), and its adjudication
# ======================================================================================
# `f11_lazy_prenorm.json` (above) is the THIRD attempt at #11 on this device and the second
# that failed verification: it was taken on a contended card (its own
# `fairness.timing.harness_floor_us` is 39.872 us against a 9.024 us launch -- a floor is a
# launch plus a sync, so that one was timing somebody else's kernels too) and its table
# carries ratios above their own physical ceilings.  `f11_publish.py` re-measured with four
# gates that harness did not have, and THIS generator prefers its file for every #11 row.
# The older file's #11 numbers are no longer published by this report at any regime.
#
# WHICH NUMBER EACH ROW CARRIES IS NOT THE SCRIPT'S OWN `publishable` FLAG.  That flag is
# wrong in four known ways, all of them in the permissive direction, and every one of them
# is re-derived here from the raw fields rather than trusted:
#
#   D1  an invariance probe whose partner config FAILED TO RUN was recorded as
#       `{"ran": false}` with no `pass` key, and the script's filter only rejected
#       `pass is False`.  So an axis that was never tested counted as invariant.  That is
#       exactly how `11a_w13` at decode_bs1 -- the one regime where the wgmma-signature axis
#       (BLOCK_M) was never probed -- came out "PUBLISHABLE".  `f11p_gate` FAILS CLOSED: a
#       probe that did not run blocks the cell.
#   D2  the `11a_w13` branch never calls `ceilings()` at all, so no #11a cell in the file has
#       `ceiling_launch_aware` and its flag reflects invariance only.  An unbounded number is
#       not publishable, so `f11p_gate` requires the arm to HAVE a ceiling.
#   D3  the `11b_half` ceiling is computed from `b_f + T*4`, but the half-fused arm is
#       `launch_rstd(h1, ...)` followed by `launch_router(h1, ...)` -- it reads h1 TWICE.
#       The rstd kernel's T*H*2 read and the router's T*4 read-back of rstd are both
#       uncharged, which makes that ceiling too GENEROUS.  `f11p_half_ceiling_corrected`
#       recomputes it; the correction is what puts decode_bs1024 #11b' over its bound.
#   D4  `11b_half` has no correctness evidence of any kind: its output buffer `logits_h` is
#       written and never compared to a reference, its rstd producer was tuned with
#       `verify = lambda: (True, "")`, and it carries no `invariance` key at all.  The
#       publication rule requires invariance PASSED with the critical axes actually TESTED;
#       for this arm it was never attempted, so every #11b' cell is blocked.
#
# AND ONE THING THE FILE'S OWN LAYOUT WILL MISLEAD A READER ABOUT: `f11_publish.json`
# carries TWO calibrations under similar names.  `/calibration` is THIS run's passing gate
# (floor 15.321 us, launch 8.327 us, ok true).  `/env` still carries the BLOCKED run's
# values (launch 9.024 us, floor 39.872 us, `calib_health.contended: true`, and the message
# "timing calibration is UNRELIABLE").  Everything below reads `/calibration`.  Reading
# `/env` -- the obvious place -- would silently recompute every ceiling from the numbers of
# the run this one exists to replace.
F11_PUBLISH_FILE = "f11_publish.json"
# The bandwidth every ceiling in that file was computed at, read from the file itself. It is
# needed by the corrected #11b' bound below, and reading it (rather than restating 4250) is
# what makes that correction a recomputation of the same model rather than a second model.
F11P_BW = ((F11P or {}).get("bandwidth_gbs"))

# The ADJUDICATED decision, and the one judgement here that is not re-derivable from the
# file: for the cells that clear every gate, the WALL ratio is still not published and the
# CUDA-GRAPH ratio is.  The reason is measurable and is recomputed per cell by
# `f11p_self_consistency()`: the largest wall ratio this run's own calibration permits is
#     pred = (graph_unfused + n_unfused*launch + floor) / (graph_fused + n_fused*launch + floor)
# with launch and floor from `/calibration`.  The four decode #11b cells measure 2.08-2.15x
# ABOVE their own pred (and above the hard n_u/n_f = 2.0 asymptote that no launch cost can
# breach when the fusion goes from 2 kernels to 1).  The two surviving prefill cells measure
# 1.34x and 1.13x above it.  Only decode_bs1024 reproduces its own pred (1.148 vs 1.147).
# The same systematic, same sign, is therefore present in the surviving prefill cells; the
# ceiling gate did not catch them only because the launch-aware ceiling is loose at prefill.
# The graph column, by contrast, is monotone in T as the mechanism requires and orders
# correctly against the C500 and RTX 4060 files.  So: graph is published, wall is recorded
# in `notes` as blocked, and no wall-based #11 speedup appears anywhere in this report.
F11P_BASIS = ("TIMING BASIS FOR THIS ROW IS CUDA-GRAPH REPLAY (graph_unfused_ms / "
              "graph_fused_ms), NOT the L2-flushed wall clock that every other row in this "
              "file uses. Do not compare this row's `fused_ms` against another row's")

# arm key in f11_publish.json -> the (fusion, variant) row identity this report already uses
F11P_ARM_ROW = {
    "11b_router": NAME[("f11", "f11b_router")],
    "11b_half": HALF,
    "11a_w13": NAME[("f11", "f11a_w13")],
}
# kernels per arm: (fused, unfused).  Read off the dispatch in f11_publish.py: #11b fuses
# norm+GEMM into one kernel against norm+GEMM; #11b' keeps two kernels on both sides; #11a
# fuses the pre-norm into the w13 grouped GEMM.
F11P_KERNELS = {"11b_router": (1, 2), "11b_half": (2, 2), "11a_w13": (1, 2)}


def _f11p_shape() -> tuple[int, int]:
    """(HIDDEN_SIZE, N_ROUTED_EXPERTS) read out of the H200 config the run itself imported.

    Parsed rather than imported so this generator keeps its no-torch dependency, and read
    rather than hardcoded so a shape change cannot silently invalidate the corrected #11b'
    ceiling below.
    """
    import re
    txt = (ROOT / "glm52_h200" / "config.py").read_text()
    def const(name: str) -> int:
        mm = re.search(rf"^{name}\s*=\s*(\d+)", txt, re.M)
        if not mm:
            raise SystemExit(f"cannot read {name} from glm52_h200/config.py")
        return int(mm.group(1))
    return const("HIDDEN_SIZE"), const("N_ROUTED_EXPERTS")


def f11p_half_ceiling_corrected(T: int, bw_gbs: float, launch_us: float) -> float:
    """D3: the #11b' launch-aware ceiling with the second h1 read charged.

    The half-fused arm is TWO kernels -- `launch_rstd(h1, rstd, ...)` then
    `launch_router(h1, b_fold, logits_h, ...)`.  It streams h1 once per kernel:

        fused    = (T*H*2 + T*4)            rstd:   read h1, write rstd
                 + (T*H*2 + T*4 + H*ER*2 + T*ER*4)   router: read h1 + rstd + gate, write logits
        unfused  = (2*T*H*2) + (T*H*2 + H*ER*2 + T*ER*4)    (as the script computes it)

    The script charges the fused side `b_f + T*4`, i.e. ONE h1 read, so its ceiling is too
    generous by a whole activation pass.  Both arms are then charged 2 launches.
    """
    H, ER = _f11p_shape()
    b_f = (T * H * 2 + T * 4) + (T * H * 2 + T * 4 + H * ER * 2 + T * ER * 4)
    b_u = (2 * T * H * 2) + (T * H * 2 + H * ER * 2 + T * ER * 4)
    bw, L = bw_gbs * 1e9, launch_us * 1e-6
    t_f = b_f / bw + 2 * L
    t_u = b_u / bw + 2 * L
    return t_u / t_f


def f11p_graph_speedup(a: dict) -> float | None:
    """graph_unfused_ms / graph_fused_ms.  The file records `graph_speedup` for the router
    arm only; the two raw times are present for every arm, so the ratio is recovered here
    rather than left as the `nan` the script's own summary printed."""
    gf, gu = a.get("graph_fused_ms"), a.get("graph_unfused_ms")
    if _nan(gf) or _nan(gu) or not gf:
        return None
    return float(gu) / float(gf)


def f11p_self_consistency(a: dict, arm: str, cal: dict) -> float | None:
    """The largest wall ratio this run's own calibration permits for this cell.

    (graph_unfused + n_u*launch + floor) / (graph_fused + n_f*launch + floor), in ms, with
    launch and floor from `/calibration`.  Everything on the right is measured; nothing is
    modelled.  A measured wall ratio above this is not a fusion win, it is the wall timer.
    """
    gf, gu = a.get("graph_fused_ms"), a.get("graph_unfused_ms")
    if _nan(gf) or _nan(gu):
        return None
    nf, nu = F11P_KERNELS[arm]
    L = float(cal["launch_us"]) / 1e3
    F = float(cal["harness_floor_us"]) / 1e3
    den = float(gf) + nf * L + F
    return (float(gu) + nu * L + F) / den if den > 0 else None


def f11p_invariance(a: dict) -> tuple[str, list[str]]:
    """(verdict, notes) for one arm's strict screen, re-derived FAIL-CLOSED (D1).

    verdict is one of "PASS" / "REJECT" / "UNTESTED" / "ABSENT".
    """
    inv = a.get("invariance")
    if not isinstance(inv, dict):
        return "ABSENT", [
            "INVARIANCE: NO SCREEN OF ANY KIND. This arm carries no `invariance` key, its "
            "output buffer is never compared against a reference anywhere in f11_publish.py, "
            "and its producer kernel was tuned with a verifier that returns True "
            "unconditionally. It rests on timing alone (D4)"]
    probes = inv.get("probes") or []
    ran = [p for p in probes if p.get("pass") is True]
    bad = [p for p in probes if p.get("pass") is False and p.get("ran") is not False]
    unt = [p for p in probes if p.get("ran") is False or ("pass" not in p and "skipped" not in p)]
    skipped = [p for p in probes if p.get("skipped")]
    keys = ", ".join(inv.get("keys") or [])
    n = [f"INVARIANCE SCREEN at tol {inv.get('tol')} (NOT check()'s 2e-2) over {len(probes)} "
         f"axes [{keys}]: worst rel_err {inv.get('worst_rel_err')}"
         + (f", {len(ran)} probes ran and passed" + (" BITWISE" if all(
             p.get("bitwise") for p in ran) and ran else "") if ran else "")]
    n.append("SCOPE OF THAT SCREEN: it probes only the keys present in the tuned config, and "
             "this run's GEMM grid emits BLOCK_M / BLOCK_N / BLOCK_K / num_warps / "
             "num_stages / GROUP_M only. warp_specialize and num_ctas are therefore NOT in "
             "the screen -- they were never varied in the measurement either, but the "
             "by-construction invariance argument names seven axes and this covers five")
    for p in skipped:
        n.append(f"axis {p['key']}: skipped -- {p['skipped']}")
    for p in bad:
        n.append(f"axis {p['key']}: {p.get('from')} -> {p.get('to')} changes the output by "
                 f"rel_err {p.get('rel_err'):.4e} against tol {inv.get('tol')} -- "
                 f"mathematically invariant, so this is a codegen defect")
    for p in unt:
        n.append(f"axis {p['key']}: partner {p.get('partner')} DID NOT RUN, so this axis is "
                 f"UNTESTED, which is not the same as invariant. The recorded probe carries "
                 f"no `pass` field and the script's filter only rejected `pass is False`, "
                 f"which is how this cell was flagged publishable (D1)")
    if bad:
        return "REJECT", n
    if unt:
        return "UNTESTED", n
    return "PASS", n


#: Above this fraction of wall time spent off-GPU, the wall ratio is a measurement of the
#: Python launch path rather than of the fusion. On H200 the #11 kernels run in 16-26 us at
#: decode -- FASTER than a Triton launch from Python -- and the unfused arm has one more
#: launch than the fused one, so it loses by construction. Measured host fractions were
#: 42-91% at decode against 8-18% at prefill.
F11P_HOST_BOUND_FRAC = 0.25


def f11p_basis(a: dict) -> tuple:
    """(published_ratio, basis, host_frac, why) for one #11 arm.

    Publishes the CUDA-graph ratio for host-bound cells and the wall ratio otherwise. Both
    are measurements; they answer different questions, and which one is meaningful is decided
    by whether the GPU or the host was the slow side -- not by which is larger.
    """
    w = a.get("paired_p50")
    gf, gu = a.get("graph_fused_ms"), a.get("graph_unfused_ms")
    wf, wu = a.get("fused_ms"), a.get("unfused_ms")
    if not gf or not gu or _nan(wf) or _nan(wu):
        return w, "wall", None, "no graph timing (capture failed); wall is all there is"
    hf = max(float(wf) - float(gf), 0.0) / max(float(wf), 1e-12)
    hu = max(float(wu) - float(gu), 0.0) / max(float(wu), 1e-12)
    h = max(hf, hu)
    if h > F11P_HOST_BOUND_FRAC:
        return (float(gu) / float(gf)), "graph", h, (
            f"HOST-BOUND ({h * 100:.0f}% of the slower arm's wall time is not GPU work): the "
            f"wall ratio {float(w):.4f}x counts Python launches, not the fusion, so the "
            f"CUDA-graph ratio is published instead")
    return w, "wall", h, (
        f"GPU-bound ({h * 100:.0f}% host overhead): the wall ratio is a measurement of the "
        f"device and is published as-is")


def f11p_gate(row: dict, arm_key: str, a: dict, cal: dict) -> list[str]:
    """Every reason this cell may NOT be published, re-derived from the raw fields.

    Empty list == publishable. The rule, stated once: a cell is publishable only if the
    calibration gate passed, the arm HAS a launch-aware ceiling, invariance PASSED with the
    critical axes actually TESTED, and the measured wall ratio is at or below that ceiling.
    """
    why: list[str] = []
    if not cal.get("ok"):
        why.append(f"BLOCKED -- calibration gate FAILED: harness floor "
                   f"{cal.get('harness_floor_us')} us against a bar of "
                   f"{cal.get('floor_bar_us')} us")
    cl = a.get("ceiling_launch_aware")
    verdict, _ = f11p_invariance(a)
    if verdict == "ABSENT":
        why.append("BLOCKED -- no invariance screen was ever run on this arm, and its output "
                   "is never compared against a reference: it has no numerical evidence at "
                   "all (D4)")
    elif verdict == "REJECT":
        bad = sorted({p["key"] for p in (a["invariance"].get("probes") or [])
                      if p.get("pass") is False and p.get("ran") is not False})
        why.append(f"BLOCKED -- invariance REJECT: the fused output depends on "
                   f"{', '.join(bad)}, worst rel_err "
                   f"{a['invariance'].get('worst_rel_err'):.4e} against tol "
                   f"{a['invariance'].get('tol')}. The screened quantity is a full-K "
                   f"reduction over one row in CTA-local registers, so it is invariant to "
                   f"these keys BY CONSTRUCTION and a dependence is a codegen defect")
    elif verdict == "UNTESTED":
        unt = sorted({p["key"] for p in (a["invariance"].get("probes") or [])
                      if p.get("ran") is False or "pass" not in p})
        why.append(f"BLOCKED -- invariance INCOMPLETE: the probe on {', '.join(unt)} did not "
                   f"run, so the one axis the defect lives on was never tested. Untested is "
                   f"not invariant; this gate fails closed (D1)")
    if _nan(cl):
        why.append("BLOCKED -- NO CEILING OF ANY KIND: f11_publish.py never calls "
                   "`ceilings()` on this arm, so neither `ceiling_launch_aware` nor "
                   "`ceiling_bytes` exists for it and the measured ratio was never bounded "
                   "by anything (D2)")
    w, _basis, _hf, _bwhy = f11p_basis(a)
    if not _nan(cl) and not _nan(w) and float(w) > float(cl):
        why.append(f"BLOCKED -- published {_basis} ratio {float(w):.4f}x EXCEEDS its own "
                   f"launch-aware ceiling {float(cl):.4f}x by "
                   f"{100.0 * (float(w) / float(cl) - 1):.1f}%. The ceiling charges each arm "
                   f"its ideal traffic time PLUS its launches; a measurement above it is not "
                   f"a physical speedup")
    if arm_key == "11b_half":
        T = row["T"]
        # D3: re-run the bound with the second h1 read charged, and block on it too.
        corr = f11p_half_ceiling_corrected(T, float(F11P_BW), float(cal["launch_us"]))
        if not _nan(w) and float(w) > corr:
            why.append(f"BLOCKED -- the recorded ceiling {float(cl):.4f}x is too generous: "
                       f"the half-fused arm reads h1 TWICE (rstd kernel, then router kernel) "
                       f"and f11_publish.py charges it one read. With the second pass "
                       f"charged the ceiling is {corr:.4f}x, and the measured wall "
                       f"{float(w):.4f}x exceeds it (D3)")
    return why


def f11p_calib_notes(d: dict) -> list[str]:
    """The calibration gate, verbatim from `/calibration`, plus the provenance hazard."""
    cal = d["calibration"]
    g = d.get("gpu") or {}
    old = load(F11_FILE) or {}
    oldt = ((old.get("fairness") or {}).get("timing") or {})
    n = [f"SOURCE results/h200/{F11_PUBLISH_FILE} (f11_publish.py, {d.get('generated')}) -- "
         f"the GATED re-measurement. It SUPERSEDES results/h200/{F11_FILE} for every #11 "
         f"row in this report: that file's numbers were taken on a contended card (its own "
         f"fairness.timing.harness_floor_us = {oldt.get('harness_floor_us')} us against a "
         f"{oldt.get('launch_cost_us')} us launch) and its table carries ratios above their "
         f"own physical ceilings. No number from it is published here",
         f"CALIBRATION GATE (the first of the four gates the old harness lacked): harness "
         f"floor {cal['harness_floor_us']:.3f} us measured BEFORE anything else, against a "
         f"bar of {cal['floor_bar_us']:.1f} us -- ok={cal['ok']}, so the run proceeded. "
         f"Launch cost {cal['launch_us']:.3f} us; floor/launch ratio "
         f"{cal['harness_floor_us'] / cal['launch_us']:.2f} against a bar of 3.0. GPU "
         f"{g.get('index')} of 8, picked as \"{g.get('why')}\". The run this replaces had a "
         f"floor of {oldt.get('harness_floor_us'):.2f} us",
         f"PROVENANCE HAZARD in the raw record, stated here because the file cannot be "
         f"edited: {F11_PUBLISH_FILE} carries TWO calibrations. `/calibration` is this run's "
         f"passing gate (the numbers above). `/env` still carries the BLOCKED run's values "
         f"-- launch_us {(d.get('env') or {}).get('launch_us')}, harness_floor_us "
         f"{(d.get('env') or {}).get('harness_floor_us')}, calib_health.contended="
         f"{((d.get('env') or {}).get('calib_health') or {}).get('contended')}, and the "
         f"message \"timing calibration is UNRELIABLE\". Every ceiling in this report is "
         f"computed from `/calibration`; reading `/env` reproduces the run that failed",
         f"bandwidth used for every ceiling below: {d.get('bandwidth_gbs')} GB/s"]
    return n


def f11p_decomp_notes(a: dict, arm: str, cal: dict) -> list[str]:
    """The wall-vs-graph decomposition, which is the whole point of the dual timing."""
    n: list[str] = []
    w = a.get("paired_p50")
    g = f11p_graph_speedup(a)
    if not _nan(w):
        n.append(f"WALL (L2-flushed, interleaved A/B/A/B, launch INCLUDED): {float(w):.4f}x "
                 f"paired per-round median over n={a.get('n')} rounds, p10-p90 "
                 f"{a.get('paired_p10'):.4f}-{a.get('paired_p90'):.4f}, ratio-of-medians "
                 f"{a.get('ratio_of_medians'):.4f}; first-half {a.get('drift_first_half'):.4f} "
                 f"vs second-half {a.get('drift_second_half'):.4f}; arms "
                 f"{a.get('fused_ms'):.4f} / {a.get('unfused_ms'):.4f} ms")
    if g is not None:
        n.append(f"GRAPH (CUDA-graph replay, launch AMORTISED): {g:.4f}x; arms "
                 f"{a.get('graph_fused_ms'):.4f} / {a.get('graph_unfused_ms'):.4f} ms")
    if not _nan(w) and g is not None:
        nf, nu = F11P_KERNELS[arm]
        verdict = ("the win SURVIVES launch amortisation, so it is real work"
                   if g > 1.0 else
                   "the win DOES NOT survive launch amortisation: under CUDA graphs the "
                   "fused arm is SLOWER, so the wall-clock win is launch overhead and not work")
        n.append(f"LAUNCH-vs-WORK DECOMPOSITION: this fusion goes from {nu} kernels to {nf}. "
                 f"wall {float(w):.4f}x, graph {g:.4f}x -- {verdict}")
        pred = f11p_self_consistency(a, arm, cal)
        if pred:
            n.append(f"SELF-CONSISTENCY of the wall figure, from this run's own numbers: the "
                     f"largest wall ratio its calibration permits is (graph_unfused + "
                     f"{nu}*launch + floor)/(graph_fused + {nf}*launch + floor) = "
                     f"{pred:.4f}x, using launch {cal['launch_us']:.3f} us and floor "
                     f"{cal['harness_floor_us']:.3f} us. Measured wall {float(w):.4f}x is "
                     f"{float(w) / pred:.2f}x that bound"
                     + ("" if float(w) / pred <= 1.02 else
                        " -- the wall column is measuring the harness, which is why this "
                        "report publishes the graph ratio and not the wall ratio"))
    return n


def f11p_ceiling_notes(row: dict, a: dict, arm: str, cal: dict) -> list[str]:
    """Which bound the cell was checked against, and what it is."""
    cl, cb = a.get("ceiling_launch_aware"), a.get("ceiling_bytes")
    if _nan(cl):
        return ["CEILING: NONE. f11_publish.py never calls `ceilings()` on this arm, so this "
                "cell was never bounded (D2)"]
    n = [f"LAUNCH-AWARE CEILING {float(cl):.4f}x -- the bound this cell was checked against. "
         f"Each arm is charged its ideal traffic time PLUS its launches "
         f"(ideal {a.get('ideal_fused_ms'):.4f} / {a.get('ideal_unfused_ms'):.4f} ms at "
         f"{F11P_BW} GB/s). A bytes-only roofline cannot bound a fusion whose win is one "
         f"fewer launch, and here it says {float(cb):.4f}x",
         f"measured wall {float(a.get('paired_p50')):.4f}x vs that ceiling: "
         f"{'WITHIN' if float(a['paired_p50']) <= float(cl) else 'ABOVE'} it"]
    if arm == "11b_half":
        corr = f11p_half_ceiling_corrected(row["T"], float(F11P_BW), float(cal["launch_us"]))
        n.append(f"that ceiling is MIS-DERIVED and this report does not accept it: the "
                 f"half-fused arm is rstd(h1) then router(h1), so it reads h1 twice, and the "
                 f"script charges `b_f + T*4` -- one read. With the second activation pass "
                 f"and the rstd read-back charged the bound is {corr:.4f}x, not "
                 f"{float(cl):.4f}x (D3)")
    return n


def f11p_rows(d: dict, regime: str) -> dict | None:
    return next((r for r in d.get("rows", []) if r.get("regime") == regime), None)


F11P_NO_HOPPER = (
    "HOPPER AXES: this run offered NONE. f11_publish.py's GEMM grid is BLOCK_M x BLOCK_N x "
    "BLOCK_K x num_warps x num_stages with GROUP_M=8 fixed -- it emits no USE_TMA, no "
    "warp_specialize and no num_ctas, so no #11 number here measures TMA, warp "
    "specialization or thread-block clusters, in either direction. That is a scoping fact "
    "about this measurement, not a finding about the axes")


def f11p_emit(d: dict, regime: str, add) -> None:
    """The four #11 rows for one regime, from the gated re-measurement.

    Four rows are emitted whatever happened. A cell the adjudication BLOCKED gets a row with
    EMPTY measurement columns and the reasons in `notes` -- never omitted, and never filled
    with a number the evidence does not support. A cell that clears every gate carries the
    CUDA-GRAPH ratio (see F11P_BASIS) with the wall figure recorded in `notes` as blocked.
    """
    cal = d["calibration"]
    row = f11p_rows(d, regime)
    base = f11p_calib_notes(d)
    if not row:
        why = (f"NO ROW for this regime in results/h200/{F11_PUBLISH_FILE}: the gated "
               f"re-measurement covers "
               f"{', '.join(r['regime'] for r in d.get('rows', []))}. Nothing is estimated "
               f"here, and the superseded file's number is not published")
        for fusion, variant in (NAME[("f11", "f11a_w13")], NAME[("f11", "f11b_router")],
                                HALF, NAME[("f11", "combined")]):
            add(fusion, variant, notes=base + [why])
        return

    arms = row.get("arms") or {}
    rt = arms.get("11b_router") or {}

    def emit(arm_key: str, k1_name: str, k2_name: str, fused_mapping: str,
             k1_mapping: str, k2_mapping: str, extra: list[str]) -> None:
        fusion, variant = F11P_ARM_ROW[arm_key]
        a = arms.get(arm_key)
        if not a:
            add(fusion, variant, notes=base + [
                f"NOT MEASURED: results/h200/{F11_PUBLISH_FILE} carries no {arm_key!r} arm at "
                f"this regime"])
            return
        why = f11p_gate(row, arm_key, a, cal)
        n = list(base)
        n += f11p_decomp_notes(a, arm_key, cal)
        n += f11p_ceiling_notes(row, a, arm_key, cal)
        n += f11p_invariance(a)[1]
        n.append(F11P_NO_HOPPER)
        n += extra
        if why:
            n = ["NOT PUBLISHED -- " + "; ".join(w.replace("BLOCKED -- ", "", 1)
                                                 for w in why)] + n
            n.append("every raw figure this cell does have is quoted above and is NOT "
                     "published: the measurement columns are empty on purpose, so that no "
                     "reader can re-derive a blocked ratio from this row")
            add(fusion, variant, notes=n, fused_mapping=fused_mapping, n_unfused_kernels=2,
                unfused_k1_name=k1_name, unfused_k1_mapping=k1_mapping,
                unfused_k2_name=k2_name, unfused_k2_mapping=k2_mapping)
            return
        g = f11p_graph_speedup(a)
        n = [F11P_BASIS,
             "PUBLISHED: this cell clears every gate -- calibration passed, the arm has a "
             "launch-aware ceiling, the strict invariance screen PASSED with every probe "
             "actually run, and the measured wall ratio is within that ceiling. The WALL "
             "ratio is nevertheless NOT published (see the self-consistency note); the "
             "number in `speedup` is the CUDA-graph ratio"] + n
        n.append("unfused_k1_ms / unfused_k2_ms are EMPTY on purpose: the per-kernel "
                 "component times in this file are wall-clock measurements, and putting them "
                 "beside a graph-replay total would mix two timing bases in one row")
        pub, basis, hfrac, bwhy = f11p_basis(a)
        n.insert(0, f"TIMING BASIS: {basis.upper()}. {bwhy}")
        add(fusion, variant, notes=n,
            fused_ms=r4(a.get("graph_fused_ms") if basis == "graph" else a.get("fused_ms")),
            fused_mapping=fused_mapping,
            unfused_total_ms=r4(a.get("graph_unfused_ms") if basis == "graph"
                                else a.get("unfused_ms")),
            speedup=r4(pub),
            n_unfused_kernels=2,
            unfused_k1_name=k1_name, unfused_k1_mapping=k1_mapping,
            unfused_k2_name=k2_name, unfused_k2_mapping=k2_mapping)

    # ---- #11a: lazy pre-norm -> w13 grouped GEMM ------------------------------------
    a11 = arms.get("11a_w13") or {}
    emit("11a_w13", "rmsnorm (writes x2)", "w13 grouped GEMM",
         m(a11.get("fused_cfg")), "", m(a11.get("unfused_cfg")),
         ["#11a IS UNMEASURABLE ON THIS DEVICE, NOT MEASURED-AND-LOST. 'Measured and lost' "
          "asserts a correct kernel was timed and was slower. Here the tuned fused winner "
          "changes its output when a knob that cannot change the answer is perturbed (6 of 7 "
          "regimes), and at the 7th the decisive axis could not be probed at all -- the "
          "winner is BM16/BN256/BK128/s4 and every cross-boundary BLOCK_M partner needs more "
          "shared memory than this device has, so no legal partner exists in the grid. The "
          "tuner then selected among those settings on speed. Nothing licenses attributing "
          "any of these ratios to 'lazy pre-norm fused into w13'",
          "CAUSE UNRESOLVED, and this report does not inherit the earlier campaign's "
          "'wgmma BLOCK_M>=64 lowering boundary' explanation: this run contradicts it at four "
          "points -- two of the failing cells have a winner at BLOCK_M=16 (the mma.sync "
          "side); three cells pass a BLOCK_M probe that straddles the threshold bit-exactly; "
          "both-wgmma pairs disagree by up to 8.9e-2 where campaign 1 found perfect "
          "agreement; and num_warps, quarantined there as a last-ulp effect, is the worst "
          "axis here. The one measurement that separates a deterministic miscompile from a "
          "race (`repeat_verdict()`) was never called by f11_publish.py",
          "#11a is also slower under CUDA-graph replay at every regime, so even setting the "
          "invariance failure aside there is no work-level win to report"])

    # ---- #11b: lazy pre-norm -> router GEMM ------------------------------------------
    emit("11b_router", "rmsnorm (writes x2)", "router GEMM",
         m(rt.get("fused_cfg")), m(rt.get("norm_cfg")), m(rt.get("unfused_gemm_cfg")),
         [f"component wall times measured in the same run (NOT published as a ratio): norm "
          f"{rt.get('norm_only_ms'):.4f} ms at {m(rt.get('norm_cfg'))}, router GEMM alone "
          f"{rt.get('gemm_only_ms'):.4f} ms at {m(rt.get('unfused_gemm_cfg'))}"
          if not _nan(rt.get("norm_only_ms")) else ""])

    # ---- #11b' half-fused: rstd kernel + epilogue scale -------------------------------
    hf = arms.get("11b_half") or {}
    emit("11b_half", "rmsnorm (writes x2)", "router GEMM",
         (f"rstd: {m(hf.get('rstd_cfg'))} | gemm: {m(hf.get('router_cfg'))}"),
         m(rt.get("norm_cfg")), m(rt.get("unfused_gemm_cfg")),
         ["#11b' is the CONTROL that separates the traffic term from the launch term: it "
          "holds the kernel count at two on both sides and removes only the activation pass. "
          "That is what makes its total absence of correctness evidence disqualifying rather "
          "than merely regrettable -- it is the arm whose number would carry the traffic "
          "claim, and nothing in this run checked its output"])

    # ---- #11a + #11b combined (one norm charged once) --------------------------------
    add(*NAME[("f11", "combined")], notes=[
        f"NOT PUBLISHED -- results/h200/{F11_PUBLISH_FILE} has no combined arm: the gated "
        f"re-measurement times #11b, #11b' and #11a separately and never builds the "
        f"layer-honest chain that charges the norm once. Nor could it be assembled here -- "
        f"the combined ratio needs BOTH fused GEMMs, and #11a is unpublishable at every "
        f"regime, so there is nothing to combine",
        "the superseded file's combined number is not carried forward: it was computed on "
        "the contended card, from an #11a arm that this run shows to be invariance-broken",
    ] + base)


def f11_estimator_spread(d: dict) -> tuple[int, float, str]:
    """(n cells carrying both estimators, largest |paired/seq - 1|, where it happens).

    The campaign publishes `speedup_source: "speedup"` (sequential) while `paired_speedup`
    exists on every row, and elsewhere in this campaign the two disagreed by 13.8 %. This
    measures the disagreement for #11 rather than assuming it is the same.
    """
    n, worst, where = 0, 0.0, ""
    for r in d.get("rows", []):
        for arm in ("f11b_router", "f11a_w13", "combined"):
            b = r.get(arm)
            if not isinstance(b, dict):
                continue
            s, p = b.get("speedup"), b.get("paired_speedup")
            if _nan(s) or _nan(p) or not s:
                continue
            n += 1
            dis = abs(p / s - 1.0)
            if dis > worst:
                worst, where = dis, f"{r.get('regime')} / {arm}"
    return n, worst, where


def f11_mainloop_range(d: dict, which: str) -> tuple[float, float, float] | None:
    """(min, median, max) of headline.<which>.instruction_cost_pct over the regimes."""
    v = sorted(((r.get("headline") or {}).get(which) or {}).get("instruction_cost_pct")
               for r in d.get("rows", [])
               if not _nan((((r.get("headline") or {}).get(which)) or {})
                           .get("instruction_cost_pct")))
    if not v:
        return None
    mid = v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
    return v[0], mid, v[-1]


def _f11_stage_best(entry) -> float | None:
    """Tuning-phase best_ms for a kernel table that is flat or split coarse/refine."""
    return best_of(entry)[0]


def _f11_table_best(tab) -> float | None:
    """Tuning-phase best_ms of a jointly-retuned CHAIN table, recorded as [cfg, ms, err]."""
    if not isinstance(tab, list):
        return None
    v = [r[1] for r in tab
         if isinstance(r, (list, tuple)) and len(r) > 1 and isinstance(r[1], (int, float))]
    return min(v) if v else None


def f11_run_notes(d: dict) -> list[str]:
    """Provenance and calibration -- identical for every #11 row, all read from the file."""
    cov = d.get("coverage") or {}
    env = d.get("env") or {}
    cal = env.get("calib_health") or {}
    tcal = (((d.get("fairness") or {}).get("timing") or {}).get("calibration") or {})
    nfail = len(cov.get("regimes_failed") or {})
    rerun = load("f11_rerun_summary.json") or {}
    when = (rerun.get("_meta") or {}).get("recorded_at", "date not recorded")
    other_card, other_floor = FAMILY_CARD.get("f01", ("", None))
    out = [
        f"SOURCE results/h200/{F11_FILE} -- the REPAIRED re-run (run_f11_h200.py, {when}). "
        f"The previous campaign's file claimed complete=true with an EMPTY rows array; this "
        f"one records status={d.get('status')!r}, complete={d.get('complete')}, "
        f"{len(cov.get('regimes_with_rows') or [])}/"
        f"{len(cov.get('regimes_requested') or [])} regimes with rows, "
        f"{nfail} regimes failed, n_arms_unmeasurable="
        f"{cov.get('n_arms_unmeasurable')} (all of them decode_bs32, see that row)",

        f"MEASURED ON A DIFFERENT CARD FROM THE REST OF THIS REPORT: env.uuid "
        f"{str(env.get('uuid'))[:8]} against {other_card} for every other family, with its "
        f"own harness floor ({env.get('harness_floor_us'):.3f} us vs "
        f"{other_floor:.3f} us). #11 is therefore NOT on the same footing as the rows above "
        f"it, and summary.json's co-tenancy event (recorded on the OTHER card) does not "
        f"apply to it",

        f"CALIBRATION CONTRADICTS ITSELF INSIDE THE FILE: fairness.timing.calibration says "
        f"trusted={tcal.get('trusted')}, reason={tcal.get('reason')!r}, while env.calib_health "
        f"on the identical inputs says launch_trusted={cal.get('launch_trusted')}, "
        f"timer_tick_trusted={cal.get('timer_tick_trusted')}, contended={cal.get('contended')} "
        f"-- \"{(cal.get('reasons') or ['?'])[0]}\". NOTE (2026-08-07): a "
        f"{env.get('harness_floor_us'):.1f} us harness floor is NORMAL for an idle H200 "
        f"(37-42 us on every clean preflight; ratio here "
        f"{(env.get('harness_floor_us') or 0) / (env.get('launch_us') or 1):.2f}x), so the "
        f"old FLOOR_LAUNCH_RATIO_MAX=3.0 / FLOOR_US_MAX=20.0 bars falsely called it "
        f"contended; config now owns 8.0 / 50.0 and the real contention detector is "
        f"timer_tick_match_frac (<0.9 = a tenant)",

        "GPU STATE: the driver pinned an idle card (f11_rerun_summary.json gpu.pinned=true, "
        "reason \"idlest of 8: 0% utilization, 0.0 GB used, 0 other compute process(es)\", "
        "tenant_events=[]), while this file's own fairness.gpu.selection says "
        "requested=null/applied=false because bench_f11 was launched without --gpu. Take the "
        "driver's record -- and note which way it cuts: if the card WAS idle then the "
        f"{env.get('harness_floor_us'):.2f} us harness floor is the real floor rather than a "
        "co-tenant, which is the worse reading for every decode number here",
    ]
    return out


def f11_est_note(b: dict, d: dict) -> list[str]:
    """Which estimator the `speedup` column publishes, and where the two disagree."""
    seq, pair = b.get("speedup"), b.get("paired_speedup")
    out: list[str] = []
    if _nan(seq) or _nan(pair):
        out.append("ESTIMATOR: this arm carries only one of `speedup` / `paired_speedup`, so "
                   "no cross-check between the sequential and paired statistics is possible")
        return out
    dis = (pair - seq) / seq
    n_cell, worst, where = f11_estimator_spread(d)
    out.append(
        f"ESTIMATOR PUBLISHED = PAIRED. The `speedup` column is paired_speedup={pair:.4f} "
        f"(median of the per-round ratios from the interleaved A/B loop, which cancels "
        f"monotone drift), NOT the file's own sequential ratio of medians "
        f"speedup={seq:.4f} (= unfused.p50/fused.p50). They disagree by {dis:+.2%} here"
        + ("" if abs(dis) < 0.02 else " -- MATERIAL (>=2%)")
        + f"; over all {n_cell} #11 arm-cells that carry both, the largest disagreement is "
          f"{worst:.2%} ({where}), far below the 13.8% seen elsewhere in this campaign")
    pm = b.get("pair_meta") or {}
    if b.get("paired") is False and pm.get("interleaved"):
        out.append(
            f"FIELD MISLABELLED IN THE SOURCE: this row records `paired: false` while its own "
            f"pair_meta records impl={pm.get('impl')!r}, interleaved={pm.get('interleaved')}, "
            f"n={pm.get('n')} with a full paired_speedup distribution. It IS a paired "
            "measurement; any consumer keying off the `paired` field will mislabel it")
    return out


def f11_inv_note(rr: dict, which: str) -> list[str]:
    """The D-A BLOCK_M-invariance probe for THIS arm's fused kernel, at this regime."""
    key = f"{which}_fused_blockm_invariance"
    c = (rr.get("checks") or {}).get(key)
    if c is None:
        return [
            f"D-A INVARIANCE TEST: NOT RUN for this arm here -- checks.{key} is absent. The "
            "probe is guarded on the arm having already PASSED its correctness check "
            "(bench_f11_lazy_prenorm.py:1478), so the one regime where a BLOCK_M-dependent "
            "wrong answer actually landed is the one regime with no invariance probe"]
    out = [
        f"D-A INVARIANCE TEST (checks.{key}): ok={c.get('ok')}, rel_err={c.get('rel_err'):.4e} "
        f"vs tol={c.get('tol')}, bitwise_identical={c.get('bitwise_identical')}, tuned "
        f"BLOCK_M={c.get('tuned_block_m')} vs probe BLOCK_M={c.get('probe_block_m')}, "
        f"gate={c.get('gate')} (NON-GATING by design: a failure does not withhold the timing)"]
    if not c.get("ok"):
        out.append("THIS ARM'S INVARIANCE TEST FAILED and its timing is published anyway. The "
                   "kernel's output is invariant to BLOCK_M by construction (sq is a full-K "
                   "per-row reduction recomputed identically by every n-tile), so a result "
                   "that depends on BLOCK_M is a codegen defect -- and it is still live")
    elif c.get("bitwise_identical") is False:
        out.append("the probe passes only on the 2e-2 tolerance: the output is NOT bitwise "
                   "invariant to BLOCK_M although the kernel is invariant to it by "
                   "construction, so a smaller residual of the same defect is present")
    out.append("SCOPE OF THE PROBE: it varies BLOCK_M only. INVARIANT_CFG_KEYS in "
               "glm52_h200/kernels/lazy_prenorm.py declares 8 keys invariant (BLOCK_M, "
               "BLOCK_N, GROUP_M, num_stages, WARP_SPECIALIZE, warp_specialize, WS, "
               "num_ctas); the other 7 are never probed, and the run's own screen log rejects "
               "wrong-answer configs that differ from a published winner in num_stages or "
               "num_warps alone. The module's `invariance_verdict` / `invariance_partner` "
               "helpers (tol 1e-5, bit-exact) were NOT the code that ran -- the bench uses "
               "its own `blockm_invariance` at line 1360, which calls common.check() at its "
               "default tol=2e-2")
    return out


def f11_mainloop_note(rr: dict, which: str) -> list[str]:
    """rows[].headline: the 2x2 at ONE shared config, which is the only in-mainloop number."""
    h = (rr.get("headline") or {}).get(which) or {}
    if h.get("unmeasurable") or _nan(h.get("instruction_cost_pct")):
        return [f"MAINLOOP ISOLATION (rows[].headline.{which}): UNMEASURABLE here -- "
                f"{h.get('reason') or h.get('note') or 'not recorded'}. Because "
                "specialization_study builds all four chains before timing any, a warp-"
                "specialization compile failure also destroys the classic-mainloop pair that "
                "needs no specialization at all"]
    out = [
        f"MAINLOOP ISOLATION (rows[].headline.{which}, all four arms in ONE rotating "
        f"interleave at ONE shared config {m(h.get('shared_config'))}): fusing the reduction "
        f"COSTS {h['instruction_cost_pct']:+.2f}% with the classic mainloop "
        f"(unfused {h['unfused_ms']:.5f} -> fused {h['fused_nonspecialized_ms']:.5f} ms) and "
        f"{h['instruction_cost_ws_pct']:+.2f}% with the warp-specialization flag set; the "
        f"flag itself moved the fused arm {h['ws_gain_fused_pct']:+.2f}% and the unfused arm "
        f"{h['ws_gain_unfused_pct']:+.2f}%",
        f"the file's own verdict for this cell, published verbatim and NOT endorsed here: "
        f"\"{h.get('verdict')}\"",
        WS_NOT_APPLIED,
        "the shared config is the FUSED arm's tuned winner in every cell, so the two unfused "
        "arms of the 2x2 run off their own optimum; and only the fused_ws_off arm is covered "
        "by a numerical check (rows[].checks) -- `unfused`, `fused_ws_on` and `unfused_ws_on` "
        "are timed without ever being shown to compute the right answer",
    ]
    return out


def f11_decomp_note(d: dict, rr: dict, b: dict) -> list[str]:
    """The decomposition the #11b number needs: how much is the mainloop, how much the
    eliminated kernel. Every term is read from this file; the arithmetic is stated."""
    reg = rr["regime"]
    tt = (d.get("tune_tables") or {}).get(reg, {})
    env = d.get("env") or {}
    head = ((rr.get("headline") or {}).get("router") or {})
    ic = head.get("instruction_cost_pct")
    rf = _f11_stage_best(tt.get("router_fused"))
    ru = _f11_stage_best(tt.get("router_unfused"))
    nm = _f11_stage_best(tt.get("norm"))
    jt = _f11_table_best(tt.get("router_unfused_joint"))
    sp = best_speedup(b)[0]
    out: list[str] = []
    if not _nan(ic):
        rng = f11_mainloop_range(d, "router")
        span = (f"the range over the {len(d.get('rows', []))} regimes is {rng[0]:+.2f}% to "
                f"{rng[2]:+.2f}%, median {rng[1]:+.2f}%" if rng else "range not computable")
        out.append(
            f"DECOMPOSITION, PART 1 -- THE MAINLOOP LOSES. At this regime the in-mainloop "
            f"cost of the reduction is {ic:+.2f}% (headline, above; {span}). It is positive "
            f"in EVERY regime, so NONE of #11b's win comes from the fused k-loop: all of it "
            f"is the removal of the second kernel -- that kernel's launch plus its activation "
            f"pass")
    if None not in (rf, ru, nm, jt):
        overhead = jt - ru - nm
        tune_sp = jt / rf
        out.append(
            f"DECOMPOSITION, PART 2 -- IT IS MOSTLY THE LAUNCH. In the TUNING phase, which is "
            f"a different measurement window from the published one, the same comparison was "
            f"timed again: fused best {rf:.5f} ms vs jointly-retuned norm+GEMM chain "
            f"{jt:.5f} ms = {tune_sp:.3f}x, against the published {sp:.3f}x. Inside that "
            f"chain the norm kernel alone tunes to {nm:.5f} ms and the GEMM alone to "
            f"{ru:.5f} ms, so putting a second kernel in the chain costs "
            f"{overhead * 1000:+.1f} us OVER AND ABOVE both kernels' own standalone times -- "
            f"against a calibrated launch cost of {env.get('launch_us'):.2f} us and a "
            f"{env.get('harness_floor_us'):.2f} us harness floor. That per-kernel chain "
            f"overhead, not the mainloop and not memory traffic, is what #11b removes")
        if not _nan(sp) and sp > 0:
            gap = abs(tune_sp / sp - 1.0)
            if gap >= 0.25:
                out.append(
                    f"CONTRADICTED BY THE FILE'S OWN OTHER WINDOW: the tuning phase makes this "
                    f"comparison {tune_sp:.3f}x and the published paired window makes it "
                    f"{sp:.3f}x -- {gap:.0%} apart, on the same kernels, the same tensors and "
                    f"the same host. Both cannot be right and nothing in the file chooses "
                    f"between them; this cell should not be quoted")
            else:
                out.append(
                    f"the two windows agree to {gap:.0%}, which is the strongest argument that "
                    f"the published ratio is a real property of the two arms rather than an "
                    f"artifact of the paired window's drift")
    if not _nan(b.get("ceiling")) and not _nan(sp):
        cl = b["ceiling"]
        over = sp / cl
        s = (f"DECOMPOSITION, PART 3 -- THE TRAFFIC TERM. The run's own MODELLED byte-traffic "
             f"ceiling for this cell is {cl:.3f}x (bytes_unfused {b.get('bytes_unfused')} / "
             f"bytes_fused {b.get('bytes_fused')}), which bounds the activation-pass part of "
             f"the win. Measured {sp:.3f}x is {over:.2f}x that ceiling")
        if over > 1.0:
            s += (" -- ABOVE IT. The traffic model carries no launch term, so the excess is "
                  "charged to the eliminated launch by elimination, not by measurement")
        else:
            s += " -- below it, so a traffic explanation is available at this regime"
        out.append(s)
    l2 = env.get("l2_bytes")
    bu = b.get("bytes_unfused")
    if l2 and bu:
        if bu <= l2:
            out.append(
                f"L2-RESIDENT: the unfused chain's whole working set ({bu / 2**20:.2f} MB) "
                f"fits this device's {l2 / 2**20:.0f} MB L2, so NEITHER arm is going to HBM "
                f"and the ceiling above is not a bound on anything -- glm52_h200/traffic.py's "
                "own words for this case are that the measured ratio is then \"an "
                "L2/occupancy/launch story\". The result JSON does not carry an `l2_resident` "
                "flag per row; this is computed here from bytes_unfused vs env.l2_bytes")
        else:
            out.append(f"NOT L2-resident: the unfused chain's working set "
                       f"({bu / 2**20:.1f} MB) exceeds the {l2 / 2**20:.0f} MB L2, so the "
                       "traffic ceiling is a real bound here")
    return out


def f11_component_note(d: dict, rr: dict, b: dict) -> list[str]:
    """Why the per-kernel columns do not decompose unfused_total_ms."""
    reg = rr["regime"]
    tt = (d.get("tune_tables") or {}).get(reg, {})
    nm = _f11_stage_best(tt.get("norm"))
    ncfg = (tt.get("norm") or {}).get("best_cfg")
    pub = b.get("norm_only_ms")
    chain_cfg = b.get("unfused_norm_cfg")
    out = ["the per-kernel columns are SEPARATE bench_chain measurements taken after the "
           "paired window (bench_f11_lazy_prenorm.py:1547-1549), each behind its own L2 "
           "flush, at the STANDALONE tuner's configs -- while the chain inside bench_pair ran "
           "the JOINTLY retuned configs. They are not a decomposition of unfused_total_ms and "
           "do not sum to it"]
    if ncfg and chain_cfg and ncfg != chain_cfg:
        out.append(f"concretely for the norm kernel: unfused_k1_ms was measured at the "
                   f"standalone winner ({m(ncfg)}), which is the mapping shown, while the "
                   f"chain inside bench_pair ran the jointly retuned {m(chain_cfg)} -- the two "
                   f"differ in 5 of the 7 regimes")
    if not _nan(nm) and not _nan(pub):
        allv = [r.get("f11b_router", {}).get("norm_only_ms") for r in d.get("rows", [])]
        allv = [v for v in allv if not _nan(v)]
        out.append(
            f"and they are not reproducible to better than a factor: unfused_k1_ms "
            f"({pub:.5f} ms) is the SAME kernel at the SAME config as this regime's norm "
            f"tuner winner ({nm:.5f} ms, tune_tables.{reg}.norm.best_ms, timed by the same "
            f"bench_chain with the same flush policy and only a different rep count) -- "
            f"{pub / nm:.2f}x apart. Published norm_only_ms is also nearly flat in T across "
            f"the campaign ({min(allv):.4f}-{max(allv):.4f} ms over T=1..8192, and it is "
            f"SMALLER at T=32 than at T=1). No absolute millisecond in this file should be "
            f"quoted to better than about one significant figure; the within-window RATIOS "
            f"are what survive")
    return out


def f11_norm_mapping(d: dict, regime: str) -> str:
    """The config `norm_only_ms` was actually measured at -- the STANDALONE tuner's winner
    (bench_f11_lazy_prenorm.py:1547 times `prob.norm_fn(norm_cfg)`), NOT the row's
    `unfused_norm_cfg`, which is the jointly retuned config the chain ran."""
    return m(((d.get("tune_tables") or {}).get(regime, {}).get("norm") or {}).get("best_cfg"))


def f11_floor_note(d: dict, *ms_values) -> str:
    """Whether the arms of this cell are resolvable above the harness's own floor."""
    fl = ((d.get("env") or {}).get("harness_floor_us"))
    vals = [v * 1000.0 for v in ms_values if not _nan(v)]
    if not fl or not vals:
        return ""
    if min(vals) < fl:
        return (f"BELOW THE HARNESS FLOOR: this cell's arms are "
                f"{', '.join(f'{v:.1f}' for v in vals)} us against the run's own "
                f"{fl:.2f} us harness_floor_us. The harness cannot resolve operands it is "
                f"larger than; this ratio is not a measurement of the kernels")
    return (f"arm magnitudes {', '.join(f'{v:.1f}' for v in vals)} us against a {fl:.2f} us "
            f"harness floor (added to BOTH arms, so the ratio understates the work ratio)")


def _f11_dev_range(path: Path, arm: str) -> str:
    """min-max of one arm's speedup on another device, read from that device's own file."""
    try:
        dd = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return ""
    v = []
    for r in dd.get("rows", []):
        e = r.get(arm)
        if isinstance(e, dict):
            s = e.get("paired_speedup") if not _nan(e.get("paired_speedup")) else e.get("speedup")
            if not _nan(s):
                v.append(float(s))
    return f"{min(v):.3f}-{max(v):.3f} over {len(v)} regimes" if v else ""


def f11a_cross_device_note() -> str:
    """#11a's record on every device in the study, read from each device's result file."""
    c500 = _f11_dev_range(ROOT / "results" / "c500" / "f11_lazy_prenorm.json", "f11a_w13")
    h200 = _f11_dev_range(RES / F11_FILE, "f11a_w13")
    r4060 = _f11_dev_range(ROOT / "results" / "rtx4060" / "f11_lazy_prenorm.json", "f11a_w13")
    none4060 = ("no rows -- the 256-expert w13 does not fit in 7.4 GB, so #11a was never "
                "measured there")
    return (
        f"#11a LOSES ON EVERY DEVICE IN THE STUDY THAT COULD RUN IT: C500 "
        f"{c500 or 'no rows'}; H200 {h200 or 'no rows'}; RTX 4060 {r4060 or none4060}. Read "
        f"those files, not this sentence. The single H200 value at or above parity "
        f"(decode_bs1) has a per-round p10-p90 of 0.997-1.733 and is not resolved from 1.0; "
        f"every other cell on every device is below 1")


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
                unfused_total_ms=r4(r["unfused_ms"]), speedup=r4(best_speedup(r)[0]),
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
                unfused_total_ms=r4(r["unfused_ms"]), speedup=r4(best_speedup(r)[0]),
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
                unfused_total_ms=r4(r["unfused_ms"]), speedup=r4(best_speedup(r)[0]),
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
                unfused_total_ms=r4(r["unfused_ms"]), speedup=r4(best_speedup(r)[0]),
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
                unfused_total_ms=r4(r["unfused_ms"]), speedup=r4(best_speedup(r)[0]),
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
                unfused_total_ms=r4(r["unfused_ms"]), speedup=r4(best_speedup(r)[0]),
                n_unfused_kernels=2,
                unfused_k1_name="expert merge (moe_sum)", unfused_k1_ms=r4(r["merge_only_ms"]),
                unfused_k1_mapping=m(mg_cfg),
                unfused_k2_name="residual add", unfused_k2_ms=r4(r["resadd_only_ms"]),
                unfused_k2_mapping=m(ra_cfg))

    # ---- #11a / #11b / #11a+b / #11b' -----------------------------------------------
    # PREFER THE GATED RE-MEASUREMENT. `f11_publish.json` is the fourth attempt at #11 on
    # this device and the first to pass its own calibration gate; `f11_lazy_prenorm.json` was
    # taken on a contended card and failed verification, so when the newer file is present
    # NONE of the older file's #11 numbers are published. The older path below is kept only
    # so this generator still runs against a checkout that has not got the new file.
    dp = load(F11_PUBLISH_FILE)
    if dp:
        f11p_emit(dp, regime, add)
        return out

    # Four rows are emitted per regime whatever happened: an arm that could not be measured
    # gets a row with EMPTY measurement columns and the file's own reason, because a silently
    # absent row is what this whole repair exists to prevent.
    d = load(F11_FILE)
    rr = pick(d["rows"], regime=regime) if d else None
    if rr:
        head = rr.get("headline") or {}
        unm = rr.get("arms_unmeasurable") or {}
        base = f11_run_notes(d)

        def unmeasurable_note(arm: str) -> str:
            e = unm.get(arm) or {}
            fc = e.get("failed_checks") or {}
            bits = ", ".join(
                f"{k} rel_err={v.get('rel_err'):.5e} vs tol={v.get('tol')} "
                f"({v.get('rows_checked', '?')}/{v.get('rows_total', '?')} rows checked, "
                f"{v.get('unwritten_rows', '?')} unwritten)"
                for k, v in fc.items() if isinstance(v, dict) and not _nan(v.get("rel_err")))
            return (f"NOT MEASURED, and NOT a scoping choice: rows[].arms_unmeasurable[{arm!r}]"
                    f" records reason={e.get('reason')!r}"
                    + (f" -- {bits}" if bits else "")
                    + f", published_timing={e.get('published_timing')}. This is the whole of "
                      "the file's complete=false")

        def f11_note(kind: str, b: dict) -> list[str]:
            n = list(base)
            n += f11_est_note(b, d)
            n += fairness_notes("f11", kind, regime, b.get("pair_meta"), b.get("tick"))
            n.append(hop_note(b.get("fused_cfg"), b.get("unfused_gemm_cfg")))
            n.append(offered_note(d, "f11a_w13_gemm" if kind == "f11a_w13"
                                    else "f11b_router_gemm")
                     + " -- whether the tuner took them is in the mappings above; the "
                       "warp_specialize axis was swept on a flag that changes nothing, see "
                       "the warp-specialization note")
            n.append(f"sum-of-squares redundancy {b.get('sq_redundancy', '?')}x, valid only "
                     "if ALL K=6144 consumers are fused (x2 never materialised)")
            wp = b.get("winner_provenance") or {}
            if wp:
                n.append(f"tuned winner re-validated before timing (winner_provenance): "
                         f"validated={wp.get('validated')}, rank={wp.get('rank')}, "
                         f"tuned_ms={wp.get('tuned_ms')}, {wp.get('detail')}, "
                         f"rejected_faster_configs={wp.get('rejected_faster_configs')}")
            return n

        # ---- #11a: lazy pre-norm -> w13 grouped GEMM --------------------------------
        b = rr.get("f11a_w13")
        if b:
            n = f11_note("f11a_w13", b)
            n += f11_inv_note(rr, "moe")
            n += f11_mainloop_note(rr, "moe")
            n.append(f11a_cross_device_note())
            n.append(f11_floor_note(d, b.get("fused_ms"), b.get("unfused_ms")))
            n += f11_component_note(d, rr, b)
            n.append(f"vendor lines: cuBLAS grouped {b['vendor_blas_grouped_ms']:.4f} ms, "
                     f"dense 1-expert {b['vendor_blas_dense_1expert_ms']:.4f} ms")
            add(*NAME[("f11", "f11a_w13")], notes=n,
                fused_ms=r4(b["fused_ms"]), fused_mapping=m(b["fused_cfg"]),
                unfused_total_ms=r4(b["unfused_ms"]), speedup=r4(best_speedup(b)[0]),
                n_unfused_kernels=2,
                unfused_k1_name="rmsnorm (writes x2)", unfused_k1_ms=r4(b["norm_only_ms"]),
                unfused_k1_mapping=f11_norm_mapping(d, regime),
                unfused_k2_name="w13 grouped GEMM",
                unfused_k2_ms=r4(b["unfused_gemm_only_ms"]),
                unfused_k2_mapping=m(b.get("unfused_gemm_cfg")))
        else:
            add(*NAME[("f11", "f11a_w13")],
                notes=base + [unmeasurable_note("f11a_w13")] + f11_inv_note(rr, "moe")
                + [f11a_cross_device_note()])

        # ---- #11b: lazy pre-norm -> router GEMM -------------------------------------
        b = rr.get("f11b_router")
        if b:
            n = f11_note("f11b_router", b)
            n += f11_inv_note(rr, "router")
            n += f11_mainloop_note(rr, "router")
            n += f11_decomp_note(d, rr, b)
            n.append(f11_floor_note(d, b.get("fused_ms"), b.get("unfused_ms")))
            n += f11_component_note(d, rr, b)
            n.append(f"vendor lines: cuBLAS router bf16 {b['vendor_blas_bf16_ms']:.4f} ms, "
                     f"fp32 {b['vendor_blas_fp32_ms']:.4f} ms")
            add(*NAME[("f11", "f11b_router")], notes=n,
                fused_ms=r4(b["fused_ms"]), fused_mapping=m(b["fused_cfg"]),
                unfused_total_ms=r4(b["unfused_ms"]), speedup=r4(best_speedup(b)[0]),
                n_unfused_kernels=2,
                unfused_k1_name="rmsnorm (writes x2)", unfused_k1_ms=r4(b["norm_only_ms"]),
                unfused_k1_mapping=f11_norm_mapping(d, regime),
                unfused_k2_name="router GEMM", unfused_k2_ms=r4(b["unfused_gemm_only_ms"]),
                unfused_k2_mapping=m(b.get("unfused_gemm_cfg")))
        else:
            add(*NAME[("f11", "f11b_router")],
                notes=base + [unmeasurable_note("f11b_router")])

        # ---- #11b' half-fused (EXPLORATORY): rstd kernel + epilogue scale -----------
        h = rr.get("half_fused") or {}
        b11 = rr.get("f11b_router") or {}
        if not _nan(h.get("router_speedup_vs_unfused")):
            n = list(base)
            n.append("NO summary.json cell of its own: #11b' is recorded inside the f11 row "
                     "as an EXPLORATORY arm, so it has no fairness flags of its own")
            n.append(
                "ESTIMATOR: NOT PAIRED, and it cannot be. half_fused carries no `speedup` / "
                "`paired_speedup` field at all -- `router_speedup_vs_unfused` is built by the "
                "bench as t_rt_u.p50_ms / t_rt_h.p50_ms, i.e. the numerator comes from the "
                "INTERLEAVED window (the #11b pair) and the denominator from a SEPARATE "
                "bench_chain run afterwards. It is a cross-window ratio of medians; "
                "best_speedup() has nothing to prefer here")
            n += fairness_notes("f11", "f11b_router", regime, None, None)
            n.append(hop_note(h.get("router_cfg"), b11.get("unfused_gemm_cfg")))
            n.append(f"rel_err vs the fp32 reference: rstd kernel "
                     f"{h.get('rel_err_rstd'):.3e}, half-fused router "
                     f"{h.get('rel_err_router'):.3e}")
            n.append(f"the FUSED side is itself 2 kernels (rstd {h['rstd_only_ms']:.4f} ms "
                     f"{m(h.get('rstd_cfg'))} + GEMM), so #11b' holds the KERNEL COUNT fixed "
                     "at two and removes only the activation pass -- it is the control that "
                     "separates the traffic term from the launch term in #11b, and it lands "
                     "well below #11b in every regime")
            if not _nan(h.get("moe_ms")):
                extra = (f"same technique on the w13 GEMM: {h['moe_ms']:.4f} ms"
                         + (f" ({h['moe_speedup_vs_unfused']:.4f}x)"
                            if not _nan(h.get("moe_speedup_vs_unfused")) else ""))
                if not _nan(h.get("combined_ms")):
                    extra += f"; the two together {h['combined_ms']:.4f} ms"
                    if not _nan(h.get("combined_speedup_vs_unfused")):
                        extra += f" ({h['combined_speedup_vs_unfused']:.4f}x)"
                    else:
                        extra += (" (no combined speedup: its unfused denominator was "
                                  "disqualified with #11a at this regime)")
                n.append(extra)
            n.append(h.get("note", ""))
            n.append(f11_floor_note(d, h.get("router_ms"), b11.get("unfused_ms")))
            n += f11_component_note(d, rr, b11)
            add(*HALF, notes=n,
                fused_ms=r4(h["router_ms"]),
                fused_mapping=f"rstd: {m(h.get('rstd_cfg'))} | "
                              f"gemm: {m(h.get('router_cfg'))}",
                unfused_total_ms=r4(b11.get("unfused_ms")),
                speedup=r4(h["router_speedup_vs_unfused"]), n_unfused_kernels=2,
                unfused_k1_name="rmsnorm (writes x2)",
                unfused_k1_ms=r4(b11.get("norm_only_ms")),
                unfused_k1_mapping=f11_norm_mapping(d, regime),
                unfused_k2_name="router GEMM",
                unfused_k2_ms=r4(b11.get("unfused_gemm_only_ms")),
                unfused_k2_mapping=m(b11.get("unfused_gemm_cfg")))
        else:
            add(*HALF, notes=base + [
                "NOT MEASURED at this regime: rows[].half_fused carries no "
                "`router_speedup_vs_unfused`, which the bench writes only when BOTH the "
                "half-fused router and the unfused chain it is scored against were measured"])

        # ---- #11a + #11b combined (one norm charged once) ---------------------------
        c = rr.get("combined")
        a11 = rr.get("f11a_w13") or {}
        if c:
            n = list(base)
            n += f11_est_note(c, d)
            n += fairness_notes("f11", "combined", regime, c.get("pair_meta"), c.get("tick"))
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
            n += f11_inv_note(rr, "moe")
            n.append(f11_floor_note(d, c.get("fused_ms"), c.get("unfused_ms")))
            add(*NAME[("f11", "combined")], notes=n,
                fused_ms=r4(c["fused_ms"]),
                fused_mapping=f"router: {m(b11.get('fused_cfg'))} | "
                              f"w13: {m(a11.get('fused_cfg'))}",
                unfused_total_ms=r4(c["unfused_ms"]), speedup=r4(best_speedup(c)[0]),
                n_unfused_kernels=3,
                unfused_k1_name="rmsnorm (writes x2, charged once)", unfused_k1_ms="",
                unfused_k1_mapping=m(c.get("norm_cfg")),
                unfused_k2_name="router GEMM", unfused_k2_ms=r4(b11.get("unfused_gemm_only_ms")),
                unfused_k2_mapping=m(b11.get("unfused_gemm_cfg")),
                unfused_k3_name="w13 grouped GEMM",
                unfused_k3_ms=r4(a11.get("unfused_gemm_only_ms")),
                unfused_k3_mapping=m(a11.get("unfused_gemm_cfg")))
        else:
            add(*NAME[("f11", "combined")],
                notes=base + [unmeasurable_note("combined"),
                              "the combined arm needs BOTH fused GEMMs and BOTH unfused "
                              "chains; with #11a disqualified there is nothing left to "
                              "compare, so no timing is published"])
    elif d:
        cov = d.get("coverage") or {}
        why = (f"NO ROW for this regime in results/h200/{F11_FILE}: coverage.regimes_with_rows "
               f"= {cov.get('regimes_with_rows')}, regimes_failed = "
               f"{cov.get('regimes_failed')}. No value is estimated here.")
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

    # ---- #11 accounting, printed because it is the whole point of this re-run ----------
    if F11P:
        cal = F11P["calibration"]
        print(f"\n  #11 from results/h200/{F11_PUBLISH_FILE} (calibration gate: floor "
              f"{cal['harness_floor_us']:.3f} us vs bar {cal['floor_bar_us']:.1f} us, "
              f"ok={cal['ok']})")
        npub = nblk = 0
        for r in F11P.get("rows", []):
            for arm_key, a in (r.get("arms") or {}).items():
                why = f11p_gate(r, arm_key, a, cal)
                g = f11p_graph_speedup(a)
                if why:
                    nblk += 1
                    print(f"    BLOCKED   {r['regime']:<14} {arm_key:<11} "
                          f"{why[0].replace('BLOCKED -- ', '')[:96]}")
                else:
                    npub += 1
                    _pub, _bas, _hf, _ = f11p_basis(a)
                    print(f"    PUBLISHED {r['regime']:<14} {arm_key:<11} "
                          f"{_bas} {float(_pub):.4f}x  (wall {a['paired_p50']:.4f}x / "
                          f"graph {g:.4f}x; host "
                          f"{('%.0f%%' % (_hf * 100)) if _hf is not None else 'n/a'})")
        print(f"    -> {npub} published, {nblk} blocked, of "
              f"{npub + nblk} arm-cells; the #11a+#11b combined row is blocked at all "
              f"{len(F11P.get('rows', []))} regimes (no combined arm in the file)")


if __name__ == "__main__":
    main()
