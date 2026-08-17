"""Frozen-config cross-device comparison -- C500 vs H200 at IDENTICAL Triton mappings.

Why this exists (the backends are unfair): the C500 campaign's Triton is a young vendor
fork (`triton 3.0.0+metax`, MACA backend) that reaches ~50 % of its vendor BLAS on dense
GEMMs and collapses a good mainloop schedule to -23 % whenever a GEMM epilogue exists
(LOG-10, a codegen cliff). The H200's Triton is upstream 3.6.0 on its flagship target
(94.4 % of cuBLAS). A fused-vs-unfused *ratio* measured on each device with each device's
own autotuned winner therefore mixes the hardware question with a compiler-quality
question: the same fusion can look bad on C500 because its fastest configuration is the
one the MACA backend breaks, not because the fusion costs anything on the hardware.

The frozen-config comparison removes two confounds at once: the mapping is held fixed
(the SAME tile/warp/stage configuration is read out of both devices' own tuning tables),
so neither "the tuner picked differently" nor "the winner happened to hit the cliff" can
move the ratio. Any remaining difference between the two devices' gains at the same
mapping is hardware (SMEM, registers, warp size, GEMM posture) or compiler quality that
NO mapping on the device can avoid -- i.e. the honest lower bound of what hardware
attribution can say from this data.

Method, per family x regime x shared mapping:
  - fused_ms  : the fused kernel's median time recorded for that mapping in the device's
                fused coarse tuning table
  - chain_ms  : the unfused GEMM's table time for the SAME mapping + the device's own
                separately-tuned best companion kernel (f01: residual-add epilogue;
                f06: SwiGLU activation). The companion kernel is per-device (it is a
                plain elementwise kernel with a tiny tuning space and no GEMM mapping),
                and the times are shown in the CSV so they cannot hide.
  - gain      : chain_ms / fused_ms, the study's own speedup convention
  - delta_pp  : (h200_gain - c500_gain) * 100

Fairness perimeter, stated plainly:
  - Same mapping != same machine code. 3.0.0+metax and 3.6.0 lower the same Triton source
    differently. This comparison isolates mapping and tuning from the device term; it
    cannot remove the compiler term, only stop it from choosing the mapping.
  - Hopper axes are excluded by construction: a config carrying USE_TMA /
    warp_specialize / num_ctas keys never matches a C500 config (whose backend has none
    of them), so only the plain-load mma.sync path appears on the H200 side.
  - f01 rows with SPLIT_K > 1 priced the unfused chain with an extra zero-init + fp32
    cast kernel; the coarse tables do not carry that kernel, so those rows are flagged
    `cast_not_priced` rather than silently compared.
  - f06's fused/unfused tables are the joint-tuned variants of the same source; the
    device's published winner figures come from the refine/joint tier and may differ
    slightly from the coarse-table median quoted here. Cells where this analysis's
    mapping IS the published winner are flagged, and the published row value is included
    so the bookkeeping error is visible.

Reads: results/f01_oproj_resadd.json + results/h200/f01_oproj_resadd.json
       results/f06_upgate_swiglu.json + results/h200/f06_upgate_swiglu.json
Writes: report_glm52_h200/cross_device_frozen/<family>.csv + a printed markdown summary.
Run:   python3 glm52/make_frozen_cfg_comparison.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report_glm52_h200" / "cross_device_frozen"

# family -> (fused tier, unfused GEMM tier, companion kernel tier, companion label)
FAMILIES = {
    "f01": {
        "file": "f01_oproj_resadd.json",
        "fused": "tune_fused_coarse",
        "unfused": "tune_unfused_coarse",
        "companion": "tune_epi_add_bf16",
        "companion_label": "resadd_epi",
        "row_fused_key": "fused_cfg",
        "row_unfused_key": "unfused_gemm_cfg",
    },
    "f06": {
        "file": "f06_upgate_swiglu.json",
        "fused": "fused_coarse",
        "unfused": "unfused_gemm_coarse",
        "companion": "unfused_act",
        "companion_label": "swiglu_act",
        "row_fused_key": "fused_cfg",
        "row_unfused_key": "unfused_gemm_cfg",
    },
    "f08f09": {
        # #9 (Down + ExpertMerge + ResAdd2, atomic): the fused tier vs the 2-kernel
        # chain whose tail is moe_sum_with_residual. The #8-only chain (moe_sum) is the
        # same numbers minus one companion; not modelled separately here.
        "file": "f08f09_down_merge_resadd.json",
        "fused": "atomic_gemm_coarse",
        "unfused": "unfused_gemm_coarse",
        "companion": "moe_sum_with_residual",
        "companion_label": "moesum_res",
        "row_fused_key": "fused_cfg",
        "row_unfused_key": "unfused_gemm_cfg",
        "row_nested": "gemm",
    },
}

# shared GEMM mapping keys that define "the same mapping" (companion kernels excluded)
# f01 also carries SPLIT_K; f06 does not. Matching is on the config dict AS RECORDED,
# minus EPI-style nested sub-dicts, so a Hopper-extended H200 config can never match.
GEMM_EXCLUDE = {"EPI", "epi"}


def _flat(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in GEMM_EXCLUDE and not isinstance(v, dict)}


def _cname(cfg: dict) -> str:
    f = _flat(cfg)
    keys = ["BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "SPLIT_K", "num_warps", "num_stages"]
    return "_".join(f"{k}{f[k]}" for k in keys if k in f)


def _table(entry: dict) -> dict:
    """cfg-dict -> ms for a tuning tier, exact dict keys as recorded."""
    out = {}
    for row in entry.get("table") or []:
        if len(row) >= 2 and row[1] is not None and isinstance(row[0], dict):
            out[json.dumps(_flat(row[0]), sort_keys=True)] = (row[0], row[1])
    return out


def _companion_best(entry) -> "tuple[float | None, dict | None]":
    if not entry:
        return None, None
    return entry.get("best_ms"), entry.get("best_cfg")


def _rows_by_regime(path: Path) -> dict:
    d = json.loads(path.read_text())
    return {r["regime"]: r for r in d["rows"]}


def _entries_by_regime(path: Path) -> dict:
    d = json.loads(path.read_text())
    t = d.get("tuning")
    if isinstance(t, dict):  # f06/f08f09: {regime: {tier: entry}}
        return {reg: tiers for reg, tiers in t.items()}
    return {e["regime"]: e for e in t}  # f01: [{regime, tier_*...}]


def _tier(entries: dict, regime: str, name: str):
    e = entries.get(regime)
    if isinstance(e, dict) and name in e:
        return e[name]
    return None


def _pub_cfg(pub: dict, key: str, spec: dict) -> dict:
    cfg = pub.get(key) or {}
    if spec.get("row_nested") and isinstance(cfg, dict):
        cfg = cfg.get(spec["row_nested"]) or {}
    return cfg


def analyze(family: str, spec: dict, pairs: bool = False) -> list:
    file = spec["file"]
    cpath, hpath = ROOT / "results" / file, ROOT / "results" / "h200" / file
    c_entries, h_entries = _entries_by_regime(cpath), _entries_by_regime(hpath)
    c_rows, h_rows = _rows_by_regime(cpath), _rows_by_regime(hpath)
    regimes = sorted(set(c_entries) & set(h_entries))
    out = []
    for reg in regimes:
        for side, tier_name in (("c500", spec["fused"]), ("h200", spec["fused"])):
            tier = _tier((c_entries if side == "c500" else h_entries), reg, tier_name)
            if not tier:
                print(f"[warn] {family} {reg} {side}: missing fused tier {tier_name}")
        day_cache = {"c500": None, "h200": None}
        comp_label = spec["companion_label"]
        cf = _table(_tier(c_entries, reg, spec["fused"]))
        hf = _table(_tier(h_entries, reg, spec["fused"]))
        cu = _table(_tier(c_entries, reg, spec["unfused"]))
        hu = _table(_tier(h_entries, reg, spec["unfused"]))
        shared_f = sorted(set(cf) & set(hf))
        shared_u = sorted(set(cu) & set(hu))
        # Each arm's best inside the shared grid, per device -- the pruning anchor. A pair
        # is "competitive" only if BOTH of its arms are within 1.5x of that device's own
        # shared-grid best, mirroring the study's coarse->refine-around-winner protocol.
        # This is what stops a junk coarse-table config on one side from fabricating a
        # 26x "gain" for the other.
        minf = {"c500": min(cf[k][1] for k in shared_f),
                "h200": min(hf[k][1] for k in shared_f)}
        minu = {"c500": min(cu[k][1] for k in shared_u),
                "h200": min(hu[k][1] for k in shared_u)}
        if pairs:
            combos = [(fk, uk) for fk in shared_f for uk in shared_u]
        else:
            combos = [(k, k) for k in set(shared_f) & set(shared_u)]
        for fk, uk in combos:
            row = {
                "family": family,
                "regime": reg,
                "fused_mapping": _cname(cf[fk][0]),
                "unfused_mapping": _cname(cu[uk][0]),
            }
            competitive = True
            for side, entries in (("c500", c_entries), ("h200", h_entries)):
                fused_ms = ({"c500": cf, "h200": hf}[side][fk][1])
                gemm_ms = ({"c500": cu, "h200": hu}[side][uk][1])
                if fused_ms > 1.5 * minf[side] or gemm_ms > 1.5 * minu[side]:
                    competitive = False
                if day_cache[side] is None:
                    comp = _tier(entries, reg, spec["companion"])
                    day_cache[side] = _companion_best(comp)
                comp_ms, comp_cfg = day_cache[side]
                rows = c_rows if side == "c500" else h_rows
                pub = rows.get(reg) or {}
                row[f"{side}_fused_ms"] = fused_ms
                row[f"{side}_{comp_label}_ms"] = comp_ms
                row[f"{side}_chain_ms"] = (gemm_ms or 0) + (comp_ms or 0)
                row[f"{side}_gain"] = row[f"{side}_chain_ms"] / fused_ms if fused_ms else None
                win_f = json.dumps(_flat(_pub_cfg(pub, spec["row_fused_key"], spec)), sort_keys=True) == fk
                win_u = json.dumps(_flat(_pub_cfg(pub, spec["row_unfused_key"], spec)), sort_keys=True) == uk
                row[f"{side}_pub_is_fused_winner"] = win_f
                row[f"{side}_pub_is_unfused_winner"] = win_u
                pub_gain = pub.get("speedup") if side == "c500" else pub.get("paired_speedup", pub.get("speedup"))
                row[f"{side}_pub_gain"] = pub_gain
            row["chain_note"] = ("cast_not_priced" if cf[fk][0].get("SPLIT_K", 1) > 1 else "")
            pathological = not competitive
            for side in ("c500", "h200"):
                fused = row[f"{side}_fused_ms"]
                chain = row[f"{side}_chain_ms"]
                pub = (c_rows if side == "c500" else h_rows).get(reg) or {}
                pub_fused = pub.get("fused_ms")
                if fused and chain and fused > 3 * chain:
                    pathological = True
                if fused and pub_fused and fused > 2.5 * pub_fused > 0:
                    pathological = True
            row["pathological"] = 1 if pathological else 0
            out.append(row)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    import sys

    args = [a for a in sys.argv[1:]]
    pairs = "--pairs" in args
    fams = [a for a in args if a in FAMILIES] or list(FAMILIES)
    for fam in fams:
        rows = analyze(fam, FAMILIES[fam], pairs=pairs)
        if not rows:
            print(f"{fam}: no shared mappings")
            continue
        cols = sorted(rows[0].keys())
        path = OUT / (f"{fam}_frozen_pairs.csv" if pairs else f"{fam}_frozen_cfg.csv")
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        by_reg = defaultdict(list)
        for r in rows:
            by_reg[r["regime"]].append(r)
        print(f"== {fam} -> {path}  ({len(rows)} rows over {len(by_reg)} regimes, pairs={pairs})")
        def _regkey(r: str) -> "tuple[int, int]":
            tag = r.split("_", 1)[1]
            val = int(tag.lstrip("bst"))
            return (0, val) if r.startswith("decode") else (1, val)

        for reg in sorted(by_reg, key=_regkey):
            rs = by_reg[reg]
            clean = [r for r in rs if not r["pathological"]]
            gvec = [(r["c500_gain"], r["h200_gain"]) for r in clean]
            deltas = [(h - c) * 100 for c, h in gvec]
            cbest = max(clean, key=lambda r: r["c500_gain"])
            hbest = max(clean, key=lambda r: r["h200_gain"])
            shared_best_delta = (hbest["h200_gain"] - cbest["c500_gain"]) * 100
            npath = len(rs) - len(clean)
            if not gvec:
                continue
            cmed = sorted(c for c, _ in gvec)[len(gvec) // 2]
            hmed = sorted(h for _, h in gvec)[len(gvec) // 2]
            dmed = sorted(deltas)[len(deltas) // 2]
            opt = "best-in-shared-grid" if pairs else "best-same-mapping"
            cast = lambda r: ("!" if r["chain_note"] else "")
            print(f"  {reg:14s} n={len(clean):3d} clean (+{npath} pruned)  "
                  f"c500_gain_p50 {cmed:7.3f}  h200_gain_p50 {hmed:7.3f}  delta_pp {dmed:+9.2f} | "
                  f"{opt}: c500 {cbest['c500_gain']:.3f}{cast(cbest)} @ f:{cbest['fused_mapping']} / "
                  f"u:{cbest['unfused_mapping']} , h200 {hbest['h200_gain']:.3f}{cast(hbest)} @ "
                  f"f:{hbest['fused_mapping']} / u:{hbest['unfused_mapping']} | "
                  f"best-delta {shared_best_delta:+9.2f} pp")


if __name__ == "__main__":
    main()