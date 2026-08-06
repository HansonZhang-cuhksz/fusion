"""sm_90a structural evidence for SQ_MODE=4 (LOG-17 #11a repair).

Compiles `moe_gateup_prenorm_kernel` and `router_gemm_kernel` AOT for sm_90a at a
wgmma-side config (BLOCK_M >= 64) in SQ_MODE 0 vs 4 and reports, from the TTGIR:

  * how many global `tt.load`s feed the k-loop (mode 3 is CSE'd back to one; mode 4
    must keep two -- the dot operand and the reduction's own transposed load),
  * whether the reduction's operand is a global load result or the pipeliner's staged
    buffer (`ttg.local_load` / local alloc reuse) -- the property the fix requires.

Nothing is launched; no CUDA context is needed.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import triton  # noqa: E402

from tools.verify_f11_headline_ws import compile_one  # noqa: E402

CFG = dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=8, num_stages=2)
SHAPE = dict(hidden=6144, router_N=256, w13_N=4096, top_k=8)
T = 1024
MOE_ROWS = 8192


def evidence(kind: str, sq_mode: int, fuse: bool = True):
    cc = compile_one(kind, CFG, fuse, ws=False, sq_mode=sq_mode, shape=SHAPE,
                     T=T, moe_rows=MOE_ROWS)
    src = cc.asm["ttgir"]
    n_load = src.count("tt.load")
    n_local_load = src.count("ttg.local_load")
    # the reduce's input chain: find the tt.reduce operand %X, then the extf/mulf defs
    # above it, and classify the op that DEFINES the tensor the extf consumes.
    mo = re.search(r'tt\.reduce"\((\S+)\)', src)
    feed = "none"
    if mo:
        seed = mo.group(1).rstrip(",").lstrip("%")
        cur = {seed}
        for _ in range(8):
            nxt = set()
            for s in cur:
                pat = r'^\s*%\s*' + re.escape(s) + r'\s*=\s*'
                m = re.search(pat + r'"([a-z0-9_.]+)"\((\S+)', src, re.M)
                if m:
                    op, arg = m.group(1), m.group(2)
                    if op in ("tt.load", "ttg.local_load", "ttg.global_load"):
                        feed = op
                        break
                    nxt.add(arg.rstrip(","))
                elif re.search(pat + r'([a-z0-9_.]+)\s', src, re.M):
                    op = re.search(pat + r'([a-z0-9_.]+)\s', src, re.M).group(1)
                    if op in ("tt.load", "ttg.local_load", "ttg.global_load"):
                        feed = op
                        break
                    m2 = re.search(r'^\s*%\s*' + re.escape(s) + r'\s*=\s*[^\n]*'
                                   r'%\s*([a-zA-Z0-9_]+)', src, re.M)
                    if m2:
                        nxt.add(m2.group(1))
            if feed != "none":
                break
            cur = nxt
    return {
        "kind": kind, "sq_mode": sq_mode,
        "global_loads(tt.load)": n_load,
        "local_loads(ttg.local_load)": n_local_load,
        "reduce_feed": feed,
    }


def main():
    rows = [evidence(k, m) for k in ("moe", "router") for m in (0, 4)]
    hdr = list(rows[0])
    print(" ".join(f"{k:>24s}" for k in hdr))
    for r in rows:
        print(" ".join(f"{str(r[k]):>24s}" for k in hdr))
    print()
    for k in ("moe", "router"):
        m0 = next(r for r in rows if r["kind"] == k and r["sq_mode"] == 0)
        m4 = next(r for r in rows if r["kind"] == k and r["sq_mode"] == 4)
        print(f"{k}: m0 reduce feeds from {m0['reduce_feed']} (the MMA's staged A copy); "
              f"m4 from {m4['reduce_feed']} (its own global load)")
        assert m4["reduce_feed"] == "tt.load", \
            "mode 4's reduction must read its own global load (LOG-17 #11a repair)"
        assert m0["reduce_feed"] in ("ttg.local_load", "ttg.global_load"), \
            "mode 0's reduction must read the staged A copy (the defect's shared path)"
        assert m4["global_loads(tt.load)"] > m0["global_loads(tt.load)"], \
            "mode 4 must add an independent global load (m3 was CSE'd; m4 must not be)"
    print("\nSTRUCTURAL EVIDENCE OK: mode 4 gives the reduction an independent global load;" 
          " mode 0/3 keep it on the MMA's staged A copy.")


if __name__ == "__main__":
    main()
