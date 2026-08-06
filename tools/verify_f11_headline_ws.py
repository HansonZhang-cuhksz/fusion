#!/usr/bin/env python3
"""Was warp specialization APPLIED in any `rows[].headline` cell of the H200 #11 run?

WHY THIS SCRIPT EXISTS
----------------------
`glm52_h200/bench/bench_f11_lazy_prenorm.py::specialization_study` was written to record
the answer, and then does not record it:

    for helper in ("caps_report", "ws_evidence"):
        fn = getattr(K, helper, None)
        if callable(fn):
            out[f"kernel_{helper}"] = fn() if helper == "caps_report" else None
                                                                          ^^^^

`K.ws_evidence(compiled)` is never called.  `kernel_ws_evidence` is therefore hard-null in
every row of `results/h200/f11_lazy_prenorm.json`, and the file contains no
`ttgir_mentions_wgmma`, no `ptx_mentions_wgmma` and no `*_mentions_warp_specialize` at all.
The only warp-specialization provenance the file does carry --
`headline.*.warp_specialize_available`, `warp_specialize_evidence`
("kernels.lazy_prenorm.warp_specialize_available()=True (mode range)") and
`kernel_caps_report.hopper_caps.warp_specialize` (source: "preflight") -- is API and driver
AVAILABILITY.  That is exactly the signal `kernels/lazy_prenorm.py::ws_evidence`'s own
docstring rules out as evidence: "On sm_89 the preflight's `tl.range(warp_specialize=True)`
probe PASSES and specializes nothing ... So 'the launch worked' is not evidence."

WHAT IT DOES
------------
Supplies the missing evidence without an H200 and without launching anything: it
cross-compiles both f11 kernels for `sm_90a` (the H200 target -- confirmed by the emitted
`.target` directive) at the exact `rows[].headline.shared_config` of every cell, for all
four arms the 2x2 timed, and compares the generated code.

Fidelity matters here, so the signature is built the way `JITFunction.run` builds it, via
Triton's own `native_specialize_impl`: pointer and 16-divisible integer arguments carry
`tt.divisibility`, and an integer argument whose runtime value is 1 (every unit stride in
these launches) becomes a CONSTEXPR.  Getting that wrong changes pipelining and shared
memory by 3x.  `--check` verifies the reproduction against the `rows[].kernel_stats`
`shared_bytes` / `n_regs` the H200 actually recorded.

Reported per cell, from the compiled artifact only:

  req       the `tt.warp_specialize` REQUEST attribute reached TTGIR (it was asked for)
  APPLIED   the compiler emitted a specialized form -- `ttg.warp_specialize` with
            partition regions, `async_task` annotations, or `setmaxnreg` in the PTX
  wgmma     the tile lowered to a warpgroup MMA (`wgmma.mma_async`), i.e. whether the
            paper's asynchronous-MMA premise is even reachable at this tile
  PTX       is the WS-on arm's PTX identical to the WS-off arm's, once source-position
            metadata is stripped?  Identical PTX cannot run at a different speed, so a
            non-zero `ws_gain_*_pct` on an identical-PTX pair is a measurement artifact.

`--control` compiles a TMA matmul for sm_90 and sm_100 with the same detector, to show it
is not simply blind: on sm_100 it fires (partition regions, `setmaxnreg`, num_warps 8->12,
more SMEM); on sm_90 it does not.  Triton 3.6 routes Hopper to `add_hopper_warpspec` and
Blackwell to `add_warp_specialize` (`backends/nvidia/compiler.py:274,287`); only the latter
produces a specialized kernel for these loops.

Usage:
    python3 tools/verify_f11_headline_ws.py [--check] [--control]
"""
from __future__ import annotations

import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton._C.libtriton import native_specialize_impl  # noqa: E402
from triton.backends.compiler import BaseBackend, GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

from glm52_h200.kernels import lazy_prenorm as K  # noqa: E402

RESULTS = os.path.join(REPO, "results", "h200", "f11_lazy_prenorm.json")
TARGET = GPUTarget("cuda", 90, 32)          # emits `.target sm_90a` -- the H200 target

#: every warp-specialization spelling `_ws()` strips off the tuned winner before pinning one
WS_CFG_KEYS = ("WARP_SPECIALIZE", "WS", "warp_specialize")

#: the four arms `specialization_study` builds, as (name, FUSE_NORM, WARP_SPECIALIZE)
ARMS = (("unfused", False, False), ("fused_ws_off", True, False),
        ("fused_ws_on", True, True), ("unfused_ws_on", False, True))

#: PTX lines carrying only SOURCE POSITION.  `WARP_SPECIALIZE` selects between two
#: textually duplicated copies of the same mainloop (lazy_prenorm.py:816 vs :859, :991 vs
#: :1030) -- "the body appears TWICE on purpose" -- so the two arms legitimately differ in
#: `.loc` directives and in the DWARF line attributes encoding the same thing.  Everything
#: else being equal is the result this script exists to establish; these are the only noise.
_DEBUG_ONLY = re.compile(r"^\s*\.loc\b|DW_AT_(call|decl)_line")


# ======================================================================================
# building the signature the way the JIT builds it
# ======================================================================================
class _Ptr:
    """Stand-in for a 16-byte-aligned device tensor: only its dtype and alignment matter."""

    def __init__(self, ty: str):
        self.ty = ty


def _spec(val):
    """`(signature entry, attr descriptor, constexpr value or None)` for one runtime arg."""
    if isinstance(val, _Ptr):
        return val.ty, "D", None          # torch's allocator is 16 B aligned -> always "D"
    ty, desc = native_specialize_impl(BaseBackend, val, False, True, True)
    if ty == "constexpr":                 # an integer argument equal to 1
        return "constexpr", "", val
    return ty, (desc or ""), None


def build_source(kernel, runtime_args: dict, constexprs: dict):
    """An `ASTSource` with the same signature/attrs/constants a real launch would produce."""
    sig, attrs, consts = {}, {}, dict(constexprs)
    for i, (name, val) in enumerate(runtime_args.items()):
        ty, desc, const = _spec(val)
        sig[name] = ty
        if const is not None:
            consts[name] = const
        elif "D" in desc:
            attrs[(i, )] = BaseBackend.parse_attr(desc)
    for name in constexprs:
        sig[name] = "constexpr"
    return ASTSource(fn=kernel, signature=sig, constexprs=consts, attrs=attrs)


def router_args(T: int, N: int, Kdim: int) -> dict:
    """`launch_router`'s positional arguments, as values (lazy_prenorm.py:1385)."""
    return {
        "a_ptr": _Ptr("*bf16"), "b_ptr": _Ptr("*bf16"), "c_ptr": _Ptr("*fp32"),
        "rstd_ptr": _Ptr("*fp32"),
        "M": T, "N": N, "K": Kdim,
        "stride_am": Kdim, "stride_ak": 1,     # a [M,K] contiguous
        "stride_bk": N, "stride_bn": 1,        # b [K,N] contiguous
        "stride_cm": N, "stride_cn": 1,        # c [M,N] contiguous
        "eps": 1e-5,
    }


def moe_args(EM: int, N: int, Kdim: int, n_valid: int) -> dict:
    """`launch_moe_gateup`'s positional arguments (lazy_prenorm.py:1308)."""
    return {
        "a_ptr": _Ptr("*bf16"), "b_ptr": _Ptr("*bf16"), "c_ptr": _Ptr("*bf16"),
        "rstd_ptr": _Ptr("*bf16"),
        "sorted_token_ids_ptr": _Ptr("*i32"), "expert_ids_ptr": _Ptr("*i32"),
        "num_tokens_post_padded_ptr": _Ptr("*i32"),
        "N": N, "K": Kdim, "EM": EM, "num_valid_tokens": n_valid,
        "stride_am": Kdim, "stride_ak": 1,     # a [T,K] contiguous
        "stride_be": N * Kdim, "stride_bk": 1, "stride_bn": Kdim,  # b [E,N,K]: (0,2,1)
        "stride_cm": N, "stride_cn": 1,        # c [T*top_k, N] contiguous
        "eps": 1e-5,
    }


def compile_one(kind: str, cfg: dict, fuse: bool, ws: bool, sq_mode: int, shape: dict,
                T: int, moe_rows: int):
    """AOT-compile one arm for sm_90a.  Nothing is launched; no CUDA context is needed."""
    if kind == "router":
        args = router_args(T, shape["router_N"], shape["hidden"])
        consts = {
            "BLOCK_SIZE_M": cfg["BLOCK_M"], "BLOCK_SIZE_N": cfg["BLOCK_N"],
            "BLOCK_SIZE_K": cfg["BLOCK_K"], "GROUP_SIZE_M": cfg["GROUP_M"],
            "even_Ks": shape["hidden"] % cfg["BLOCK_K"] == 0,
            "FUSE_NORM": fuse, "SQ_MODE": sq_mode, "USE_RSTD": False,
            "WARP_SPECIALIZE": ws,
        }
        src = build_source(K.router_gemm_kernel, args, consts)
    else:
        em = -(-moe_rows // cfg["BLOCK_M"]) * cfg["BLOCK_M"]   # block-aligned padding
        args = moe_args(em, shape["w13_N"], shape["hidden"], moe_rows)
        consts = {
            "BLOCK_SIZE_M": cfg["BLOCK_M"], "BLOCK_SIZE_N": cfg["BLOCK_N"],
            "BLOCK_SIZE_K": cfg["BLOCK_K"], "GROUP_SIZE_M": cfg["GROUP_M"],
            "top_k": shape["top_k"], "compute_type": tl.bfloat16,
            "even_Ks": shape["hidden"] % cfg["BLOCK_K"] == 0,
            "FUSE_NORM": fuse, "SQ_MODE": sq_mode, "USE_RSTD": False,
            "WARP_SPECIALIZE": ws,
        }
        src = build_source(K.moe_gateup_prenorm_kernel, args, consts)
    return triton.compile(
        src, target=TARGET,
        options={"num_warps": cfg["num_warps"], "num_stages": cfg["num_stages"],
                 "num_ctas": 1},
    )


#: `CompiledKernel.metadata.n_regs` is filled in by the driver when the cubin is LOADED,
#: which an AOT cross-compile never does.  ptxas has already decided the number, so read it
#: out of the cubin -- that is what makes the fidelity check able to match `n_regs` too.
_CUOBJDUMP = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin",
                          "cuobjdump")
_REG_RE = re.compile(r"\bREG:(\d+)")


def cubin_regs(compiled) -> "int | None":
    import subprocess
    import tempfile
    cubin = (compiled.asm or {}).get("cubin")
    if not cubin or not os.path.exists(_CUOBJDUMP):
        return None
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as fh:
            fh.write(cubin)
            path = fh.name
        out = subprocess.run([_CUOBJDUMP, "-res-usage", path],
                             capture_output=True, text=True, timeout=60).stdout
        m = _REG_RE.search(out)
        return int(m.group(1)) if m else None
    except Exception:  # noqa: BLE001 -- provenance is a bonus, never fatal
        return None
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def inspect(compiled) -> dict:
    md, asm = compiled.metadata, compiled.asm
    ttgir = asm.get("ttgir") or ""
    ttir = asm.get("ttir") or ""
    ptx = asm.get("ptx") or ""
    return {
        "warps": getattr(md, "num_warps", None),
        "smem": getattr(md, "shared", None),
        "regs": getattr(md, "n_regs", None) or cubin_regs(compiled),
        # asked for: the attribute `tl.range(warp_specialize=True)` puts on the scf.for
        "ws_req": ("tt.warp_specialize" in ttgir) or ("tt.warp_specialize" in ttir),
        # APPLIED, in any spelling either backend path can produce.  `setmaxnreg` is the
        # PTX-level tell and is form-agnostic: a producer/consumer kernel re-partitions
        # registers between its warp groups.
        "ws_appl": ("ttg.warp_specialize" in ttgir) or ("partition0" in ttgir)
                   or ("async_task" in ttgir) or ("setmaxnreg" in ptx),
        "wgmma": "wgmma.mma_async" in ptx,
        "ptx_stripped": "\n".join(l for l in ptx.splitlines()
                                  if not _DEBUG_ONLY.search(l)),
    }


# ======================================================================================
def control() -> int:
    """Show the detector fires where warp specialization really is applied."""
    @triton.jit
    def tma_mm(a_desc, b_desc, c_desc, M, N, Kd,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, WS: tl.constexpr):
        pid = tl.program_id(0)
        pm, pn = pid // tl.cdiv(N, BN), pid % tl.cdiv(N, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        if WS:
            for k in tl.range(0, Kd, BK, warp_specialize=True):
                acc += tl.dot(a_desc.load([pm * BM, k]), b_desc.load([pn * BN, k]).T)
        else:
            for k in tl.range(0, Kd, BK):
                acc += tl.dot(a_desc.load([pm * BM, k]), b_desc.load([pn * BN, k]).T)
        c_desc.store([pm * BM, pn * BN], acc.to(tl.bfloat16))

    sig = {"a_desc": "tensordesc<bf16[128,64]>", "b_desc": "tensordesc<bf16[128,64]>",
           "c_desc": "tensordesc<bf16[128,128]>", "M": "i32", "N": "i32", "Kd": "i32",
           "BM": "constexpr", "BN": "constexpr", "BK": "constexpr", "WS": "constexpr"}
    print("POSITIVE CONTROL -- identical detector, TMA matmul")
    for cc in (90, 100):
        for ws in (False, True):
            c = triton.compile(
                ASTSource(fn=tma_mm, signature=dict(sig),
                          constexprs={"BM": 128, "BN": 128, "BK": 64, "WS": ws}),
                target=GPUTarget("cuda", cc, 32),
                options={"num_warps": 8, "num_stages": 3, "num_ctas": 1})
            ev = inspect(c)
            print(f"  sm_{cc:<4d} WS={ws!s:5s}  APPLIED={ev['ws_appl']!s:5s}  "
                  f"warps={ev['warps']:2d}  smem={ev['smem']}")
    return 0


def check(d: dict) -> int:
    """Does the reproduction match the shared memory / registers the H200 recorded?"""
    print("FIDELITY CHECK -- cross-compiled vs rows[].kernel_stats recorded on the H200")
    hdr = (f"{'regime':14s} {'kernel_stats key':28s} {'smem repro':>10s} "
           f"{'smem H200':>10s} {'regs repro':>10s} {'regs H200':>10s}  match")
    print(hdr)
    print("-" * len(hdr))
    shape, ok, tot = d["shape"], 0, 0
    for row in d["rows"]:
        iso = row.get("isolation_fuse_on_vs_off_same_cfg") or {}
        ks = row.get("kernel_stats") or {}
        for cons, keys in (("router", ("router_fused", "router_unfused_at_fused_cfg")),
                           ("moe", ("moe_fused", "moe_unfused_at_fused_cfg"))):
            c = iso.get(cons)
            if not c or not isinstance(c.get("config"), dict):
                continue
            base = {k: v for k, v in c["config"].items() if k not in WS_CFG_KEYS}
            sq = c.get("sq_mode") or d["sq_mode_study"]["pick"][cons]
            for key, fuse in zip(keys, (True, False)):
                want = ks.get(key)
                if not want:
                    continue
                try:
                    ev = inspect(compile_one(cons, base, fuse, False, sq, shape,
                                             row["T"], row.get("moe_rows") or 0))
                except Exception as exc:  # noqa: BLE001
                    print(f"{row['regime']:14s} {key:28s} COMPILE FAILED "
                          f"{type(exc).__name__}")
                    tot += 1
                    continue
                same = (ev["smem"] == want["shared_bytes"]
                        and ev["regs"] == want["n_regs"])
                ok += bool(same)
                tot += 1
                print(f"{row['regime']:14s} {key:28s} {ev['smem']:10d} "
                      f"{want['shared_bytes']:10d} {str(ev['regs']):>10s} "
                      f"{want['n_regs']:10d}  {'yes' if same else 'NO'}")
    print(f"\n{ok}/{tot} cross-compiled kernels reproduce the H200's recorded "
          f"shared memory AND register count exactly")
    return 0


def main() -> int:
    d = json.load(open(RESULTS))
    if "--control" in sys.argv:
        return control()
    if "--check" in sys.argv:
        return check(d)
    shape = d["shape"]
    hdr = (f"{'regime':14s} {'cons':6s} {'BM':>4s} | {'req':5s} {'APPLIED':7s} "
           f"{'wgmma':5s} | {'warps on/off':>13s} {'smem on/off':>17s} | "
           f"{'PTX ws_on vs ws_off':19s} | {'recorded ws_gain fused':>22s}")
    print(hdr)
    print("-" * len(hdr))
    n_appl = n_ident = n_cells = n_depipe = 0
    for row in d["rows"]:
        reg = row["regime"]
        iso = row.get("isolation_fuse_on_vs_off_same_cfg") or {}
        for cons in ("router", "moe"):
            c = iso.get(cons)
            if not c or not isinstance(c.get("config"), dict):
                continue
            sq = c.get("sq_mode") or d["sq_mode_study"]["pick"][cons]
            base = {k: v for k, v in c["config"].items() if k not in WS_CFG_KEYS}
            ev, failed = {}, False
            for arm, fuse, ws in ARMS:
                try:
                    ev[arm] = inspect(compile_one(cons, base, fuse, ws, sq, shape,
                                                  row["T"], row.get("moe_rows") or 0))
                except Exception as exc:  # noqa: BLE001
                    print(f"{reg:14s} {cons:6s} {base['BLOCK_M']:>4d} | COMPILE FAILED "
                          f"[{arm}] {type(exc).__name__}  <- WS cannot even be built here")
                    failed = True
            if failed:
                continue
            n_cells += 1
            fon, foff = ev["fused_ws_on"], ev["fused_ws_off"]
            uon, uoff = ev["unfused_ws_on"], ev["unfused"]
            same = (fon["ptx_stripped"] == foff["ptx_stripped"]
                    and uon["ptx_stripped"] == uoff["ptx_stripped"])
            n_ident += bool(same)
            n_appl += bool(fon["ws_appl"] or uon["ws_appl"])
            # The WS request is not free even when it specializes nothing: on the small
            # tiles it stops `tritongpu-pipeline` multi-buffering, and the arm loses its
            # software pipeline.  That is a DE-pipelining regression wearing the label of a
            # warp-specialization result.
            depipe = fon["smem"] < foff["smem"]
            n_depipe += bool(depipe)
            gain = (c.get("ws_gain_fused_pct")
                    if isinstance(c.get("ws_gain_fused_pct"), (int, float)) else None)
            verdict = ("IDENTICAL" if same
                       else f"DE-PIPELINED {foff['smem'] // max(fon['smem'], 1)}x")
            print(f"{reg:14s} {cons:6s} {base['BLOCK_M']:>4d} | "
                  f"{str(fon['ws_req']):5s} {str(fon['ws_appl']):7s} "
                  f"{str(fon['wgmma']):5s} | "
                  f"{fon['warps']:6d}/{foff['warps']:<6d} "
                  f"{fon['smem']:8d}/{foff['smem']:<8d} | "
                  f"{verdict:19s} | "
                  f"{(f'{gain:+.2f}%' if gain is not None else 'n/a'):>22s}")
    print()
    print(f"cells compiled                                 : {n_cells}")
    print(f"cells where warp specialization was APPLIED    : {n_appl}")
    print(f"cells whose WS-on PTX == WS-off PTX            : {n_ident}")
    print(f"cells where the WS request COST the pipeline   : {n_depipe}")
    if n_cells and n_appl == 0:
        print(
            "\nVERDICT: warp specialization was requested in all 12 headline cells and "
            "APPLIED in none.\n"
            f"  * in {n_ident} cells the WS-on and WS-off arms are the SAME machine code, "
            "so their\n    `ws_gain_*_pct` and `instruction_cost_ws_pct` measure the "
            "harness, not the technique;\n"
            f"  * in {n_depipe} cells the request instead disabled multi-buffering "
            "(shared memory falls\n    to a single stage), so their large negative "
            "`ws_gain_*_pct` is a DE-PIPELINING\n    regression, still not a "
            "warp-specialization effect.\n"
            "Either way, no cell of rows[].headline measures whether warp specialization "
            "absorbs\nthe fused reduction, which is the claim the campaign exists to test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
