"""Collect every `results/*.json` into one cross-fusion table, joined against the
analytical roofline ceiling from `traffic.py`.

The join is the point: a measured speedup is only interesting relative to the ceiling that
the fusion's traffic saving allows. `measured/ceiling` near 1 means the kernel is
bandwidth-saturated and the fusion delivered what it structurally could; well under 1 means
the mapping is leaving something behind; *above* the ceiling means the win came from
somewhere other than traffic (launch overhead, occupancy) -- or the measurement is wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import traffic

RESULTS = Path(__file__).resolve().parent.parent / "results"

# (results-file id, lowercased variant substring) -> traffic-model fusion key.
# Rules are tried in order; the first whose substring appears in the variant name wins,
# and `""` is the catch-all default for that file. Explicit rather than prefix-matched,
# because `f11a`/`f11b` and `f8`/`f9` are not separable by a common prefix.
CEILING_RULES = {
    "f01_oproj_resadd": [("", "F1_oproj_resadd")],
    "f03_resadd_rmsnorm": [("", "F3_resadd_rmsnorm")],
    "f04f05_norm_router": [
        ("f4", "F4_addnorm_router"),
        ("f5", "F5_rmsnorm_router"),
        ("", "F5_rmsnorm_router"),
    ],
    "f06_upgate_swiglu": [("", "F6_upgate_swiglu")],
    "f08f09_down_merge_resadd": [
        ("f9", "F9_down_merge_resadd"),
        ("f8", "F8_down_merge"),
        ("", "F8_down_merge"),
    ],
    "f10_merge_resadd": [("", "F10_merge_resadd")],
    "f11_lazy_prenorm": [
        ("router", "F11b_prenorm_router"),
        ("w13", "F11a_prenorm_w13"),
        # `combined` and `half_fused` are dominated by the w13 GEMM, so that is their bound
        ("", "F11a_prenorm_w13"),
    ],
}


def ceiling_key(fid: str, variant: str) -> str | None:
    v = str(variant).lower()
    for sub, key in CEILING_RULES.get(fid, []):
        if sub == "" or sub in v:
            return key
    return None


def _walk_rows(obj, out, parent_key=None):
    """Find every dict that looks like a benchmark row, anywhere in the payload.

    `parent_key` supplies a variant label for families that nest their sub-variants under
    named keys instead of carrying a `variant` field (f11 does this: `f11b_router`,
    `f11a_w13`, `combined`, `half_fused`).
    """
    if isinstance(obj, dict):
        if "regime" in obj and ("speedup" in obj or "fused_ms" in obj):
            row = dict(obj)
            row.setdefault("variant", parent_key or "-")
            out.append(row)
            return  # a row's own sub-dicts are diagnostics, not further rows
        for k, v in obj.items():
            _walk_rows(v, out, k)
    elif isinstance(obj, list):
        for v in obj:
            _walk_rows(v, out, parent_key)


def load() -> dict[str, list[dict]]:
    found = {}
    for p in sorted(RESULTS.glob("*.json")):
        try:
            payload = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"!! {p.name}: unreadable ({exc})")
            continue
        rows: list[dict] = []
        _walk_rows(payload, rows)
        found[p.stem] = rows
    return found


def ceilings() -> dict[tuple[str, str], float]:
    return {
        (r["fusion"], r["regime"]): r["roofline_ceiling"] for r in traffic.all_rows()
    }


REGIME_ORDER = ["decode_bs1", "decode_bs32", "decode_bs256", "prefill_t2048", "prefill_t8192"]


def report() -> str:
    data = load()
    ceil = ceilings()
    lines = ["# Consolidated fusion results", ""]
    if not data:
        return "# Consolidated fusion results\n\n(no results/*.json found yet)\n"

    for fid, rows in sorted(data.items()):
        lines.append(f"## `{fid}`  ({len(rows)} rows)")
        if not rows:
            lines.append("\n_no benchmark rows in this file_\n")
            continue
        lines.append("")
        lines.append(
            "| variant | regime | fused ms | unfused ms | speedup | ceiling | of ceiling | rel_err |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")

        def sort_key(r):
            reg = str(r.get("regime", ""))
            idx = REGIME_ORDER.index(reg) if reg in REGIME_ORDER else 99
            return (str(r.get("variant", "")), idx)

        for r in sorted(rows, key=sort_key):
            reg = str(r.get("regime", ""))
            sp = r.get("speedup")
            if sp is None and r.get("fused_ms") and r.get("unfused_ms"):
                sp = r["unfused_ms"] / r["fused_ms"]
            k = ceiling_key(fid, r.get("variant", ""))
            c = ceil.get((k, reg)) if k else None
            frac = f"{sp / c:.2f}" if (sp and c) else "-"
            lines.append(
                f"| {r.get('variant', '-')} | {reg} | "
                f"{_f(r.get('fused_ms'))} | {_f(r.get('unfused_ms'))} | "
                f"{_f(sp, 'x')} | {_f(c, 'x')} | {frac} | {_f(r.get('rel_err'), '', 2e-5)} |"
            )
        lines.append("")
    return "\n".join(lines)


def _f(v, suffix="", small=1e-4) -> str:
    if v is None:
        return "-"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if suffix == "x":
        return f"{v:.3f}x"
    if abs(v) < small:
        return f"{v:.2e}"
    return f"{v:.4f}"


if __name__ == "__main__":
    print(report())
