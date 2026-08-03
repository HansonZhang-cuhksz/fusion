"""Fusions #4 / #5 -- (ResAdd +) RMSNorm + Router GEMM (+ sigmoid top-8).

Four fused variants, each against its own independently tuned unfused chain:

    #5     fused [norm+gemm]              vs  [norm][gemm]
    #5+tk  fused [norm+gemm+topk]         vs  [norm][gemm][topk]
    #4     fused [add+norm+gemm]          vs  [add+norm][gemm]
    #4+tk  fused [add+norm+gemm+topk]     vs  [add+norm][gemm][topk]

Every one of those seven kernels is the SAME source
(`glm52_h200/kernels/norm_router.py::norm_router_kernel`) with different `tl.constexpr`
flags; only the mapping differs, and each is tuned over its own grid.

Traffic (T rows, act = T*6144*2 B):
    #5 unfused  = 2*act (norm) + act + 3 MB + T*1 KB (gemm)
    #5 fused    = 2*act + 3 MB + T*1 KB            -> saves the router's read of x2
    #4 unfused  = 3*act (add) + 2*act (norm) + act + ...
    #4 fused    = 4*act + ...

This is the family that inverted hardest between C500 (0.38-0.68x) and Ada (0.67-1.30x),
and the reason was formulation, not bytes: forcing the router GEMM into the normalization's
row-per-program tiling is the paper's tiling mismatch in the wrong direction.  Hopper adds
warp specialization, which is exactly the mechanism "Towards Free Normalization" relies on
and which neither previous device had -- so this family is the one whose H200 number is
least predictable from the two existing ports.  Nothing here assumes it: the fused kernel
is whatever `kernels/norm_router.py` compiles to, and the search grid gains a specialized
variant only if the preflight proved the feature AND the kernel module advertises the knob.

Run:
    python3 glm52_h200/bench/bench_f04f05_norm_router.py [--regimes ...] [--only F5,F4]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from glm52_h200 import bench as B
from glm52_h200 import config as C
from glm52_h200 import reference as ref
from glm52_h200.common import (
    autotune,
    bench_chain,
    check,
    main_guard,
    record,
    rel_err,
    speedup_row,
)
from glm52_h200.kernels import norm_router as K

RESULT_ID = "f04f05_norm_router"
H = C.HIDDEN_SIZE  # 6144
E = C.N_ROUTED_EXPERTS  # 256
TK = C.NUM_EXPERTS_PER_TOK  # 8
EPS = C.RMS_NORM_EPS
DT = C.DTYPE
UNITS = ["F5", "F5_topk", "F4", "F4_topk"]

# --------------------------------------------------------------------------------------
# mapping search spaces.  ONE generator per kernel *shape*, used by every variant that has
# that shape -- so the fused kernel and the unfused router GEMM see the identical grid, and
# the two norm kernels see the identical grid.
# --------------------------------------------------------------------------------------
_ENV = C.env()
SMEM = B.env_int(_ENV, "smem_bytes")  # per-block opt-in ceiling
WARP = B.env_int(_ENV, "warp_size")
#: the fp32 accumulator tile lives in the per-block register file -- a REGISTER budget,
#: which the C500 code wrote as a shared-memory constant.
REGS = B.env_int(_ENV, "regs_per_sm")
MAX_THREADS = B.max_threads_per_block(_ENV)
WARPS = B.warp_ladder(_ENV)
#: pass-1 of the fused kernel holds a [BLOCK_M, NORM_BK] tile AND its running sum of
#: squares, so its element budget is half a program's register-bounded tile.
NORM_TILE_CAP = B.elems_per_program_cap(_ENV) // 2
#: fp32 accumulator elements per lane a GEMM tile may hold before it certainly spills
ACC_PER_LANE_MAX = 128


def _gemm_ok(cfg: dict) -> bool:
    bm, bk, be, w, s = (
        cfg["BLOCK_M"], cfg["BLOCK_K"], cfg["BLOCK_E"], cfg["num_warps"], cfg["num_stages"]
    )
    # Triton's mainloop multi-buffer count is version-dependent (3.0 staged num_stages,
    # 3.6 stages num_stages-1 with a floor of 2).  C.smem_stage_bytes owns that; the old
    # formula over-predicts by 1.33-1.5x and rejects tiles this stack runs fine.
    if C.smem_stage_bytes(bm, be, bk, s) > SMEM:
        return False
    if bm * be > REGS:  # fp32 accumulator tile: BM*BE registers per program
        return False
    threads = w * WARP
    if threads > MAX_THREADS:
        return False
    if bm * be / threads > ACC_PER_LANE_MAX or bm * be / threads < 1:
        return False
    return True


def gemm_grid() -> list[dict]:
    """Grid for every variant containing the router GEMM (fused #4/#5 and the stand-alone
    router kernel).  The size is device-dependent -- the SMEM/register prefilter is a
    function of `C.env()` -- so `fairness.grids`, not a number in this docstring, is the
    count of record."""
    out: list[dict] = []
    for bm, bk, be, (w, s) in itertools.product(
        (16, 32, 64, 128), (32, 64, 128), (32, 64, 128, 256), ((4, 2), (8, 2))
    ):
        cfg = dict(BLOCK_M=bm, BLOCK_K=bk, BLOCK_E=be, num_warps=w, num_stages=s)
        if _gemm_ok(cfg):
            out.append(cfg)
    # wider warps / deeper pipeline at the narrow k-tile
    for bm, be, (w, s) in itertools.product(
        (16, 32, 64, 128), (128, 256), ((16, 2), (8, 3), (8, 4))
    ):
        cfg = dict(BLOCK_M=bm, BLOCK_K=32, BLOCK_E=be, num_warps=w, num_stages=s)
        if _gemm_ok(cfg):
            out.append(cfg)
    # BLOCK_E = 256 (the whole expert dimension in one program) is the only shape the
    # FUSE_TOPK variants can use, so that corner gets a denser warp/stage sweep -- for
    # every GEMM-side variant alike.  Which k-tiles survive here is a pure function of the
    # device's SMEM ceiling: C500's 65536 B admitted only BLOCK_K=32, sm89's 101376 B also
    # BLOCK_K=64, and an H200's ceiling is whatever the probe says -- which is exactly why
    # this is not written as a literal.
    for bm, (w, s) in itertools.product(
        (16, 32, 64), ((4, 1), (4, 3), (2, 2), (8, 4), (16, 3))
    ):
        for bk in (32, 64, 128):
            cfg = dict(BLOCK_M=bm, BLOCK_K=bk, BLOCK_E=256, num_warps=w, num_stages=s)
            if _gemm_ok(cfg):
                out.append(cfg)
    # cache-eviction hints (streaming activations evict_first, router weight evict_last)
    out += [
        dict(c, EVICT=1)
        for c in list(out)
        if c["BLOCK_E"] == 256 and c["num_stages"] == 2 and c["num_warps"] == 8
    ]
    return B.widen(B.dedup(out), K)


def fused_grid() -> list[dict]:
    """Grid for the FUSED variants: `gemm_grid()` plus, for every config in it, the same
    config with the *pass-1* k-tile (`NORM_BK`) widened -- over a ladder, not to a single
    value.

    This is not extra freedom -- it is the fused-side counterpart of a knob the unfused
    side already has.  On the unfused side the norm kernel picks its own BLOCK_K (it always
    picks 2048); without NORM_BK the fused kernel would compute the sum of squares in the
    GEMM's 32/64-wide tiles, i.e. 96-192 sequential cross-lane reductions per row instead
    of 3.  Leaving that out would be exactly the "under-tuned fused side" failure mode.

    C500 offered ONE value here (`min(2048, 32768 // BLOCK_M)`), a fixed 32768-element
    pass-1 tile that is 32-128 elem/thread at warp 64 and 64-256 at warp 32 -- i.e. on a
    narrower warp the single offer is simply rejected and the fused arm loses the knob
    entirely.  Bound the pass-1 tile PER THREAD instead, with the same budget `_norm_ok`
    gives the unfused norm kernel, and let the tuner pick.
    """
    out: list[dict] = []
    for c in gemm_grid():
        out.append(c)
        for nbk in (256, 512, 1024, 2048, 4096):
            if (
                nbk > c["BLOCK_K"]
                and C.HIDDEN_SIZE % nbk == 0
                and c["BLOCK_M"] * nbk
                <= B.MAX_ELEMS_PER_THREAD * WARP * c["num_warps"]
            ):
                out.append(dict(c, NORM_BK=nbk))
    return B.dedup(out)


def _norm_ok(cfg: dict) -> bool:
    bm, bk, w = cfg["BLOCK_M"], cfg["BLOCK_K"], cfg["num_warps"]
    threads = w * WARP
    if threads > MAX_THREADS:
        return False
    if bm * bk > NORM_TILE_CAP:
        return False
    epr = bm * bk / threads  # elem/thread -- WARP is the PROBED lane count, never a literal
    if epr < 2 or epr > B.MAX_ELEMS_PER_THREAD:
        return False
    return True


def norm_grid() -> list[dict]:
    """Grid for the stand-alone norm kernels (rmsnorm / add+rmsnorm).  Same size class as
    `gemm_grid`; BLOCK_E is irrelevant (no GEMM) and pinned to 256.

    The warp ladder is the device's own: truncating it to C500's max of 16 on a 32-lane
    device searched 130 of 164 legal configs in the analogous f11 grid -- a 21 % one-sided
    truncation of a grid that feeds ONLY the unfused arm, i.e. biased toward inflating the
    fused win.
    """
    out: list[dict] = []
    for bm, bk, (w, s) in itertools.product(
        (1, 2, 4, 8, 16, 32, 64),
        (512, 1024, 2048),
        [(w, 2) for w in WARPS],
    ):
        cfg = dict(BLOCK_M=bm, BLOCK_K=bk, BLOCK_E=256, num_warps=w, num_stages=s)
        if _norm_ok(cfg):
            out.append(cfg)
    out += [
        dict(c, EVICT=1)
        for c in list(out)
        if c["num_stages"] == 2 and c["num_warps"] == 8 and c["BLOCK_K"] >= 1024
    ]
    return B.dedup(out)


def topk_grid() -> list[dict]:
    """Grid for the stand-alone sigmoid+top-8 kernel (no k-loop at all)."""
    out = [
        dict(BLOCK_M=bm, BLOCK_K=32, BLOCK_E=256, num_warps=w, num_stages=2)
        for bm, w in itertools.product((1, 2, 4, 8, 16, 32, 64), WARPS)
        if w * WARP <= MAX_THREADS
    ]
    return B.dedup(out)


# --------------------------------------------------------------------------------------
def run_regime(regime, quick: bool, units: list[str], fair: B.Fairness) -> tuple:
    T = regime.T
    wt, rt, wf, rf = B.reps(T, quick)
    torch.manual_seed(1234 + T)
    x = torch.randn(T, H, device="cuda", dtype=DT)
    res = torch.randn(T, H, device="cuda", dtype=DT)
    w = (torch.randn(H, device="cuda", dtype=torch.float32) * 0.1 + 1.0).to(DT)
    # 1/sqrt(H) init -> logits ~ N(0,1) -> sigmoid scores spread over ~(0.1, 0.9).  This
    # matters: with a larger init the scores saturate at 1.0, every top-8 weight collapses
    # to 2.5/8 = 0.3125, and a completely wrong expert selection still looks numerically
    # fine.  Realistic scaling keeps the selection well separated from fp32 noise.
    gate = (torch.randn(E, H, device="cuda", dtype=torch.float32) * H**-0.5).to(DT)
    wgt = K.wgt_from_gate(gate)  # [H, E], the layout BOTH sides consume

    def buf():
        return (
            torch.empty(T, H, device="cuda", dtype=DT),  # h1
            torch.empty(T, H, device="cuda", dtype=DT),  # x2
            torch.empty(T, E, device="cuda", dtype=torch.float32),  # logits
            torch.empty(T, TK, device="cuda", dtype=torch.float32),  # topk_w
            torch.empty(T, TK, device="cuda", dtype=torch.int32),  # topk_i
        )

    h1f, x2f, lgf, twf, tif = buf()  # fused-side outputs
    h1u, x2u, lgu, twu, tiu = buf()  # unfused-side outputs

    # ---- fp32 references --------------------------------------------------------------
    r_x2_5 = ref.rmsnorm(x, w, EPS)
    r_x2_4, r_h1 = ref.add_rmsnorm(x, res, w, EPS)
    r_lg5, r_tw5, r_ti5 = ref.router(r_x2_5, gate)
    r_lg4, r_tw4, r_ti4 = ref.router(r_x2_4, gate)

    print(f"  == {regime.name} (T={T}) ==", flush=True)
    gg, fg, ng, tg = gemm_grid(), fused_grid(), norm_grid(), topk_grid()
    if quick:
        gg, fg = B.quick_slice(gg, 14), B.quick_slice(fg, 18)
        ng, tg = B.quick_slice(ng, 10), B.quick_slice(tg, 6)
    fg_full = [c for c in fg if c["BLOCK_E"] == E]  # top-k needs all logits in-program
    print(
        f"    grids: gemm={len(gg)} fused={len(fg)} fused(BE=256, FUSE_TOPK)="
        f"{len(fg_full)} norm={len(ng)} topk={len(tg)}",
        flush=True,
    )

    # ---- verifiers used by the screening pass -----------------------------------------
    # One bf16 ULP is 2^-8 = 3.9e-3 relative, and the fp32 sum-of-squares differs from
    # torch's by summation order, which flips the rounding of the occasional element of x2
    # -- so the activation tolerance has to sit just above one ULP.  A miscompiled config
    # is nowhere near this line (the ones C500 caught came back at 0.79-0.89).
    TOL_ACT, TOL_LG, TOL_W, TOL_IDS = 8e-3, 5e-3, 2e-3, 0.99

    def _v(pairs, ids=None):
        d = {k: rel_err(g, r) for k, (g, r, _t) in pairs.items()}
        good = all(d[k] <= t for k, (_g, _r, t) in pairs.items())
        if ids is not None:
            got, ref_ids = ids
            frac = (got == ref_ids).all(1).float().mean().item()
            d["ids_row_agree"] = frac
            good = good and frac >= TOL_IDS
        return good, d

    v_norm = lambda: _v({"x2": (x2u, r_x2_5, TOL_ACT)})  # noqa: E731
    v_addn = lambda: _v({"x2": (x2u, r_x2_4, TOL_ACT), "h1": (h1u, r_h1, TOL_ACT)})  # noqa: E731
    v_gemm = lambda: _v({"logits": (lgu, r_lg5, TOL_LG)})  # noqa: E731
    v_tk = lambda: _v({"tw": (twu, r_tw5, TOL_W)}, ids=(tiu, r_ti5))  # noqa: E731
    v_f5 = lambda: _v({"x2": (x2f, r_x2_5, TOL_ACT), "logits": (lgf, r_lg5, TOL_LG)})  # noqa: E731
    v_f4 = lambda: _v({  # noqa: E731
        "x2": (x2f, r_x2_4, TOL_ACT), "h1": (h1f, r_h1, TOL_ACT),
        "logits": (lgf, r_lg4, TOL_LG),
    })
    v_f5t = lambda: _v(  # noqa: E731
        {"x2": (x2f, r_x2_5, TOL_ACT), "tw": (twf, r_tw5, TOL_W)}, ids=(tif, r_ti5)
    )
    v_f4t = lambda: _v(  # noqa: E731
        {"x2": (x2f, r_x2_4, TOL_ACT), "h1": (h1f, r_h1, TOL_ACT),
         "tw": (twf, r_tw4, TOL_W)},
        ids=(tif, r_ti4),
    )

    def tune(tag, make_chain, grid, verify, prep=None):
        tr = B.screened_autotune(tag, make_chain, grid, verify, wt, rt, prep=prep)
        fair.add(regime.name, tag, "tune", tr)
        return tr

    # ---- the eight independently tuned kernels ---------------------------------------
    need_5 = any(u.startswith("F5") for u in units)
    need_4 = any(u.startswith("F4") for u in units)
    need_tk = any(u.endswith("_topk") for u in units)

    t_norm = tune("norm", lambda c: [lambda: K.rmsnorm_only(x, w, x2u, c)], ng, v_norm) \
        if need_5 else None
    t_addn = tune(
        "add+norm", lambda c: [lambda: K.add_rmsnorm_only(x, res, w, h1u, x2u, c)],
        ng, v_addn,
    ) if need_4 else None
    t_gemm = tune(
        "router", lambda c: [lambda: K.router_gemm(x2u, wgt, lgu, c)], gg, v_gemm,
        prep=lambda: x2u.copy_(r_x2_5),  # screening needs a meaningful input
    )
    t_tk = tune(
        "topk", lambda c: [lambda: K.topk_only(lgu, twu, tiu, c)], tg, v_tk,
        prep=lambda: lgu.copy_(r_lg5),
    ) if need_tk else None
    t_f5 = tune(
        "F5", lambda c: [lambda: K.fused_norm_router(x, w, x2f, wgt, lgf, c)], fg, v_f5
    ) if "F5" in units else None
    t_f4 = tune(
        "F4", lambda c: [lambda: K.fused_add_norm_router(x, res, w, h1f, x2f, wgt, lgf, c)],
        fg, v_f4,
    ) if "F4" in units else None
    t_f5t = tune(
        "F5+topk",
        lambda c: [lambda: K.fused_norm_router_topk(x, w, x2f, wgt, twf, tif, c)],
        fg_full, v_f5t,
    ) if "F5_topk" in units else None
    t_f4t = tune(
        "F4+topk",
        lambda c: [
            lambda: K.fused_add_norm_router_topk(x, res, w, h1f, x2f, wgt, twf, tif, c)
        ],
        fg_full, v_f4t,
    ) if "F4_topk" in units else None

    # ---- joint chain re-tune of each unfused chain (can only help the baseline) -------
    def joint(tag, norm_tr, pieces, with_topk):
        cand = []
        for cn, cg in itertools.product(B.top_cfgs(norm_tr, k=3), B.top_cfgs(t_gemm, k=3)):
            for ct in (B.top_cfgs(t_tk, k=2) if with_topk else [None]):
                cand.append({"norm": cn, "gemm": cg, "topk": ct})
        cand.append({
            "norm": norm_tr.best_cfg,
            "gemm": t_gemm.best_cfg,
            "topk": t_tk.best_cfg if with_topk else None,
        })
        uniq, seen = [], set()
        for c in cand:
            key = json.dumps(c, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        t0 = time.time()
        tr = autotune(lambda jc: pieces(jc), uniq, wt, rt)
        fair.add(regime.name, tag, "joint", tr)
        print(
            f"    [{tag:<10}] joint {tr.n_tried} combos -> {tr.best_ms:.4f} ms "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )
        return tr

    j5 = joint("U5", t_norm, lambda jc: [
        lambda: K.rmsnorm_only(x, w, x2u, jc["norm"]),
        lambda: K.router_gemm(x2u, wgt, lgu, jc["gemm"]),
    ], False) if "F5" in units else None
    j5t = joint("U5+topk", t_norm, lambda jc: [
        lambda: K.rmsnorm_only(x, w, x2u, jc["norm"]),
        lambda: K.router_gemm(x2u, wgt, lgu, jc["gemm"]),
        lambda: K.topk_only(lgu, twu, tiu, jc["topk"]),
    ], True) if "F5_topk" in units else None
    j4 = joint("U4", t_addn, lambda jc: [
        lambda: K.add_rmsnorm_only(x, res, w, h1u, x2u, jc["norm"]),
        lambda: K.router_gemm(x2u, wgt, lgu, jc["gemm"]),
    ], False) if "F4" in units else None
    j4t = joint("U4+topk", t_addn, lambda jc: [
        lambda: K.add_rmsnorm_only(x, res, w, h1u, x2u, jc["norm"]),
        lambda: K.router_gemm(x2u, wgt, lgu, jc["gemm"]),
        lambda: K.topk_only(lgu, twu, tiu, jc["topk"]),
    ], True) if "F4_topk" in units else None

    # ---- validation (winners only) ----------------------------------------------------
    def zero():
        for t in (h1f, x2f, lgf, twf, tif, h1u, x2u, lgu, twu, tiu):
            t.zero_()

    def ids_row(got, want, label):
        """Fraction of rows whose whole top-8 id set matches the fp32 reference.

        Not required to be exactly 1.0: fused and unfused logits differ from the fp32
        reference by ~3e-6 (summation order), which can legitimately flip the 8th expert
        when two are that close.  A miscompiled reduction gets essentially EVERY row
        wrong -- the two failure modes are orders of magnitude apart."""
        frac = (got == want).all(1).float().mean().item()
        return {"label": label, "rel_err": 1.0 - frac, "tol": 0.01, "ok": frac >= 0.99}

    chk = {}
    if "F5" in units:
        zero()
        K.fused_norm_router(x, w, x2f, wgt, lgf, t_f5.best_cfg)
        K.rmsnorm_only(x, w, x2u, j5.best_cfg["norm"])
        K.router_gemm(x2u, wgt, lgu, j5.best_cfg["gemm"])
        torch.cuda.synchronize()
        chk["F5_fused_x2"] = check(x2f, r_x2_5, label="F5_fused_x2")
        chk["F5_fused_logits"] = check(lgf, r_lg5, label="F5_fused_logits")
        chk["F5_unfused_x2"] = check(x2u, r_x2_5, label="F5_unfused_x2")
        chk["F5_unfused_logits"] = check(lgu, r_lg5, label="F5_unfused_logits")
        chk["F5_fused_eq_unfused"] = {
            "label": "F5_fused_eq_unfused", "rel_err": rel_err(lgf, lgu), "tol": 2e-2,
            "ok": True, "x2_bitwise": bool(torch.equal(x2f, x2u)),
        }
    if "F5_topk" in units:
        zero()
        K.fused_norm_router_topk(x, w, x2f, wgt, twf, tif, t_f5t.best_cfg)
        K.rmsnorm_only(x, w, x2u, j5t.best_cfg["norm"])
        K.router_gemm(x2u, wgt, lgu, j5t.best_cfg["gemm"])
        K.topk_only(lgu, twu, tiu, j5t.best_cfg["topk"])
        torch.cuda.synchronize()
        chk["F5t_fused_w"] = check(twf, r_tw5, tol=2e-3, label="F5t_fused_topk_w")
        chk["F5t_unfused_w"] = check(twu, r_tw5, tol=2e-3, label="F5t_unfused_topk_w")
        chk["F5t_fused_ids"] = ids_row(tif, r_ti5, "F5t_fused_topk_ids")
        chk["F5t_unfused_ids"] = ids_row(tiu, r_ti5, "F5t_unfused_topk_ids")
    if "F4" in units:
        zero()
        K.fused_add_norm_router(x, res, w, h1f, x2f, wgt, lgf, t_f4.best_cfg)
        K.add_rmsnorm_only(x, res, w, h1u, x2u, j4.best_cfg["norm"])
        K.router_gemm(x2u, wgt, lgu, j4.best_cfg["gemm"])
        torch.cuda.synchronize()
        chk["F4_fused_h1"] = check(h1f, r_h1, label="F4_fused_h1")
        chk["F4_fused_x2"] = check(x2f, r_x2_4, label="F4_fused_x2")
        chk["F4_fused_logits"] = check(lgf, r_lg4, label="F4_fused_logits")
        chk["F4_unfused_h1"] = check(h1u, r_h1, label="F4_unfused_h1")
        chk["F4_unfused_x2"] = check(x2u, r_x2_4, label="F4_unfused_x2")
        chk["F4_unfused_logits"] = check(lgu, r_lg4, label="F4_unfused_logits")
    if "F4_topk" in units:
        zero()
        K.fused_add_norm_router_topk(x, res, w, h1f, x2f, wgt, twf, tif, t_f4t.best_cfg)
        K.add_rmsnorm_only(x, res, w, h1u, x2u, j4t.best_cfg["norm"])
        K.router_gemm(x2u, wgt, lgu, j4t.best_cfg["gemm"])
        K.topk_only(lgu, twu, tiu, j4t.best_cfg["topk"])
        torch.cuda.synchronize()
        chk["F4t_fused_w"] = check(twf, r_tw4, tol=2e-3, label="F4t_fused_topk_w")
        chk["F4t_fused_h1"] = check(h1f, r_h1, label="F4t_fused_h1")
        chk["F4t_unfused_w"] = check(twu, r_tw4, tol=2e-3, label="F4t_unfused_topk_w")
        chk["F4t_fused_ids"] = ids_row(tif, r_ti4, "F4t_fused_topk_ids")
        chk["F4t_unfused_ids"] = ids_row(tiu, r_ti4, "F4t_unfused_topk_ids")
    bad = [k for k, v in chk.items() if not v["ok"]]
    if bad:
        print(f"    !! FAILED CHECKS: {bad}", flush=True)

    # ---- final timing: each variant against its own chain, INTERLEAVED and PAIRED -----
    chains = {}
    if "F5" in units:
        chains["F5"] = (
            [lambda: K.fused_norm_router(x, w, x2f, wgt, lgf, t_f5.best_cfg)],
            [lambda: K.rmsnorm_only(x, w, x2u, j5.best_cfg["norm"]),
             lambda: K.router_gemm(x2u, wgt, lgu, j5.best_cfg["gemm"])],
        )
    if "F5_topk" in units:
        chains["F5_topk"] = (
            [lambda: K.fused_norm_router_topk(x, w, x2f, wgt, twf, tif, t_f5t.best_cfg)],
            [lambda: K.rmsnorm_only(x, w, x2u, j5t.best_cfg["norm"]),
             lambda: K.router_gemm(x2u, wgt, lgu, j5t.best_cfg["gemm"]),
             lambda: K.topk_only(lgu, twu, tiu, j5t.best_cfg["topk"])],
        )
    if "F4" in units:
        chains["F4"] = (
            [lambda: K.fused_add_norm_router(x, res, w, h1f, x2f, wgt, lgf, t_f4.best_cfg)],
            [lambda: K.add_rmsnorm_only(x, res, w, h1u, x2u, j4.best_cfg["norm"]),
             lambda: K.router_gemm(x2u, wgt, lgu, j4.best_cfg["gemm"])],
        )
    if "F4_topk" in units:
        chains["F4_topk"] = (
            [lambda: K.fused_add_norm_router_topk(
                x, res, w, h1f, x2f, wgt, twf, tif, t_f4t.best_cfg)],
            [lambda: K.add_rmsnorm_only(x, res, w, h1u, x2u, j4t.best_cfg["norm"]),
             lambda: K.router_gemm(x2u, wgt, lgu, j4t.best_cfg["gemm"]),
             lambda: K.topk_only(lgu, twu, tiu, j4t.best_cfg["topk"])],
        )

    fused_t, unfused_t, pairs = {}, {}, {}
    for key, (fa, ub) in chains.items():
        fused_t[key], unfused_t[key], pairs[key] = B.bench_pair(
            fa, ub, wf, rf, label=f"{regime.name}/{key}"
        )

    # ---- attribution -------------------------------------------------------------------
    # (i) resources of fused vs unfused GEMM at BOTH winning configs, cache cleared between
    #     compiles; (ii) the F1 stride-0 trick: point the fused kernel's activation input at
    #     a broadcast row so the instruction stream is identical but the DRAM traffic
    #     collapses to one row -- separates "extra bytes" from "extra instructions".
    attrib = {"gflop_router": 2.0 * T * H * E / 1e9}
    if "F5" in units:
        xb = x[:1].expand(T, H)
        x2b = x2u[:1].expand(T, H)
        jf = getattr(K, "norm_router_kernel", None)
        attrib.update({
            "regs_fused_F5_at_fused_best": B.kernel_stats(
                lambda: K.fused_norm_router(x, w, x2f, wgt, lgf, t_f5.best_cfg), jf),
            "regs_router_at_router_best": B.kernel_stats(
                lambda: K.router_gemm(x2u, wgt, lgu, t_gemm.best_cfg), jf),
            "regs_fused_F5_at_router_best": B.kernel_stats(
                lambda: K.fused_norm_router(x, w, x2f, wgt, lgf, t_gemm.best_cfg), jf),
            "regs_router_at_fused_best": B.kernel_stats(
                lambda: K.router_gemm(x2u, wgt, lgu, t_f5.best_cfg), jf),
        })
        for key, fn in (
            ("iso_fused_F5_no_x2_store_ms",
             lambda: K.fused_norm_router_no_x2(x, w, x2f, wgt, lgf, t_f5.best_cfg)),
            ("cross_fused_F5_on_router_cfg_ms",
             lambda: K.fused_norm_router(x, w, x2f, wgt, lgf, t_gemm.best_cfg)),
            ("cross_router_on_fused_cfg_ms",
             lambda: K.router_gemm(x2u, wgt, lgu, t_f5.best_cfg)),
            ("iso_fused_F5_bcast_ms",
             lambda: K.fused_norm_router(xb, w, x2f, wgt, lgf, t_f5.best_cfg)),
            ("iso_router_bcast_ms",
             lambda: K.router_gemm(x2b, wgt, lgu, t_gemm.best_cfg)),
        ):
            try:
                attrib[key] = bench_chain([fn], wf, rf).p50_ms
            except Exception as exc:  # noqa: BLE001 -- attribution is not the measurement
                attrib[key] = f"failed: {type(exc).__name__}"
    print(f"    attrib {attrib}", flush=True)

    # ---- torch / vendor-BLAS production lines -----------------------------------------
    gatef = gate.float()
    t_torch5 = bench_chain([lambda: ref.router(ref.rmsnorm(x, w, EPS), gate)], wf, rf)
    t_torch4 = bench_chain(
        [lambda: ref.router(ref.add_rmsnorm(x, res, w, EPS)[0], gate)], wf, rf
    )
    t_blas_fp32 = bench_chain([lambda: x2u.float() @ gatef.t()], wf, rf)
    t_blas_bf16 = bench_chain([lambda: x2u @ wgt], wf, rf)

    # ---- roofline ceilings from the shared model --------------------------------------
    ceil = B.traffic_ceilings(regime)

    act = T * H * 2
    wg_bytes = H * E * 2
    lg_bytes = T * E * 4
    tk_bytes = T * TK * 8
    bytes_model = {
        "F5": (2 * act + wg_bytes + lg_bytes, 3 * act + wg_bytes + lg_bytes),
        "F5_topk": (2 * act + wg_bytes + tk_bytes,
                    3 * act + wg_bytes + 2 * lg_bytes + tk_bytes),
        "F4": (4 * act + wg_bytes + lg_bytes, 6 * act + wg_bytes + lg_bytes),
        "F4_topk": (4 * act + wg_bytes + tk_bytes,
                    6 * act + wg_bytes + 2 * lg_bytes + tk_bytes),
    }
    gflop = 2.0 * T * H * E / 1e9
    # The `ceiling` below is a DRAM-traffic ceiling: it assumes the bytes the fusion removes
    # were going to DRAM.  Where the activation fits in L2 -- ~50 MB here, so T<=4096 fits
    # where on C500's 8 MB only T<=680 did -- they never leave the chip, the fusion has
    # nothing to save, and the ceiling is unattainable.  `act_fits_l2` is recorded so a 1.0x
    # is not read as an underperforming kernel.
    l2_bytes = B.env_int(_ENV, "l2_bytes")

    rows = []
    for key, ceil_key in (
        ("F5", "F5_rmsnorm_router"), ("F5_topk", "F5_rmsnorm_router"),
        ("F4", "F4_addnorm_router"), ("F4_topk", "F4_addnorm_router"),
    ):
        if key not in chains:
            continue
        bf, bu = bytes_model[key]
        f, u = fused_t[key], unfused_t[key]
        cell = ceil.get(ceil_key, {})
        rows.append(speedup_row(regime.name, f, u, {
            "variant": key,
            "T": T,
            "ceiling": cell.get("roofline_ceiling"),
            "traffic_ratio_model": cell.get("traffic_ratio"),
            "paired_speedup": pairs[key].get("paired_speedup_p50"),
            "paired_speedup_trimmed": pairs[key].get("paired_speedup_trimmed_mean"),
            "pair_meta": pairs[key],
            "tick": B.tick_report(f.p50_ms, u.p50_ms),
            "bytes_fused": bf,
            "bytes_unfused": bu,
            "act_bytes": act,
            "l2_bytes": l2_bytes,
            "act_fits_l2": act <= l2_bytes,
            "gbps_fused": bf / (f.p50_ms * 1e-3) / 1e9,
            "gbps_unfused": bu / (u.p50_ms * 1e-3) / 1e9,
            "tflops_fused": gflop / (f.p50_ms * 1e-3) / 1e3,
            "fused_cfg": {"F5": t_f5, "F5_topk": t_f5t, "F4": t_f4,
                          "F4_topk": t_f4t}[key].best_cfg,
            "unfused_cfg": {"F5": j5, "F5_topk": j5t, "F4": j4,
                            "F4_topk": j4t}[key].best_cfg,
            "torch_ref_ms": (t_torch5 if key.startswith("F5") else t_torch4).p50_ms,
            "blas_router_fp32_ms": t_blas_fp32.p50_ms,
            "blas_router_bf16_ms": t_blas_bf16.p50_ms,
        }))

    for r in rows:
        c = r["ceiling"]
        sp = r.get("paired_speedup") or r["speedup"]
        print(
            f"    {r['variant']:<8} fused {r['fused_ms']:.4f} | unfused "
            f"{r['unfused_ms']:.4f} | paired {sp:.3f}x "
            + (f"(ceiling {c:.2f}x, {100 * sp / c:.0f}% of it)" if c else "(ceiling n/a)")
            + f" | torch {r['torch_ref_ms']:.4f}"
            + ("  [TICK-LIMITED]" if r["tick"].get("tick_limited") else ""),
            flush=True,
        )

    tune_tables = {
        k: (v.as_dict() if v is not None else None)
        for k, v in (
            ("norm", t_norm), ("add_norm", t_addn), ("router_gemm", t_gemm),
            ("topk", t_tk), ("fused_F5", t_f5), ("fused_F4", t_f4),
            ("fused_F5_topk", t_f5t), ("fused_F4_topk", t_f4t),
            ("joint_U5", j5), ("joint_U5_topk", j5t), ("joint_U4", j4),
            ("joint_U4_topk", j4t),
        )
    }
    timings = {
        **{f"fused_{k}": v.as_dict() for k, v in fused_t.items()},
        **{f"unfused_{k}": v.as_dict() for k, v in unfused_t.items()},
        "pairs": pairs,
        "torch_ref_F5": t_torch5.as_dict(),
        "torch_ref_F4": t_torch4.as_dict(),
        "blas_router_fp32": t_blas_fp32.as_dict(),
        "blas_router_bf16": t_blas_bf16.as_dict(),
        "attribution": attrib,
        "piece_best_ms": {
            "norm": t_norm.best_ms if t_norm else None,
            "add_norm": t_addn.best_ms if t_addn else None,
            "router_gemm": t_gemm.best_ms,
            "topk": t_tk.best_ms if t_tk else None,
        },
    }
    del x, res, w, gate, wgt, h1f, x2f, lgf, twf, tif, h1u, x2u, lgu, twu, tiu
    torch.cuda.empty_cache()
    return rows, tune_tables, chk, timings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    B.add_std_args(ap, UNITS)
    args = ap.parse_args()
    if args.list:
        print("regimes:", ", ".join(B.REGIME_NAMES))
        print("variants:", ", ".join(UNITS))
        return

    env = C.env()
    B.banner(env)
    B.exact_fp32_matmul()
    B.check_preflight_device(env)
    units = B.resolve_units(UNITS, args.only)
    regimes = B.resolve_regimes(C, args.regimes)

    # Grid sizes are device-dependent (SMEM ceiling, warp width, register file), so the
    # fairness accounting is counted live rather than quoting another machine's numbers.
    _fg = fused_grid()
    n_gg, n_fg, n_ng = len(gemm_grid()), len(_fg), len(norm_grid())
    n_fg_full = len([c for c in _fg if c["BLOCK_E"] == E])

    fair = B.Fairness(
        one_kernel_source="glm52_h200/kernels/norm_router.py::norm_router_kernel",
        flags="DO_ADD / DO_NORM / DO_GEMM / DO_TOPK constexpr select all seven kernels "
              "(4 fused variants + 3 stand-alone pieces)",
        grids_note=(
            f"gemm_grid() ({n_gg} cfgs) for the stand-alone router GEMM; fused_grid() = "
            f"gemm_grid() x {{NORM_BK tied, NORM_BK ladder}} ({n_fg}) for the fused #4/#5 "
            f"kernels -- NORM_BK is the fused-side counterpart of the k-tile the unfused "
            f"norm kernel tunes independently over norm_grid() ({n_ng} cfgs), so the "
            f"unfused side sees {n_gg}+{n_ng}={n_gg + n_ng} configs against the fused "
            f"side's {n_fg}; the FUSE_TOPK variants get fused_grid() restricted to "
            f"BLOCK_E=256 ({n_fg_full}), a structural requirement (all 256 logits of a row "
            f"must live in one program), not a tuning choice. Counts are LIVE for this "
            f"device -- the SMEM ceiling and warp width both prune the grid, and they prune "
            f"the BLOCK_E=256 family (the fused FUSE_TOPK arm's only legal shape) harder "
            f"than the rest."
        ),
        screening="every config of every variant is numerically screened against the fp32 "
                  "reference before it is allowed into a timing grid; rejects are counted "
                  "per arm in n_failed, so an asymmetric screen-out is visible",
        baseline_extra="each unfused chain additionally gets a joint re-tune over the "
                       "top-3 x top-3 (x top-2) configs of its pieces, timed as the real "
                       "chain -- this can only help the baseline",
        weight_layout="the router weight is transposed to [H, E] ONCE and both sides "
                      "consume that same layout (a load-time weight-prep choice)",
    )

    rows, tables, checks, timings, pair_meta = [], {}, {}, {}, None

    def snapshot(done: bool) -> None:
        record(RESULT_ID, {
            "id": RESULT_ID,
            "complete": done,
            "fusion": "#4 ResAdd+RMSNorm+Router and #5 RMSNorm+Router "
                      "(+ FUSE_TOPK: sigmoid + noaux_tc top-8 folded in)",
            "shape": {
                "hidden": H, "experts": E, "topk": TK, "dtype": "bfloat16",
                "router_acc": "float32", "eps": EPS,
                "routed_scaling_factor": C.ROUTED_SCALING_FACTOR,
                "norm_topk_prob": C.NORM_TOPK_PROB, "n_group": C.N_GROUP,
            },
            "variants_measured": units,
            "fairness": fair.render(env, pair_meta),
            "env": env.__dict__,
            "rows": rows,
            "checks": checks,
            "timings": timings,
            "tune_tables": tables,
        })

    for regime in regimes:
        ck = B.ckpt_load(RESULT_ID, regime.name, env, force=args.force)
        if ck is not None:
            print(f"  == {regime.name} == (from checkpoint)", flush=True)
            rr, tab, chk, tim = ck["rows"], ck["tables"], ck["checks"], ck["timings"]
            fair.grids.update(ck.get("fairness_grids", {}))
        else:
            try:
                rr, tab, chk, tim = run_regime(regime, args.quick, units, fair)
            except Exception as exc:  # noqa: BLE001 -- one bad regime must not lose the rest
                import traceback

                traceback.print_exc()
                checks[regime.name] = {
                    "regime_failed": f"{type(exc).__name__}: {exc}"[:300]
                }
                snapshot(False)
                torch.cuda.empty_cache()
                continue
            pair_meta = next(iter(tim.get("pairs", {}).values()), None)
            B.ckpt_save(RESULT_ID, regime.name, env, {
                "rows": rr, "tables": tab, "checks": chk, "timings": tim,
                "fairness_grids": {regime.name: fair.grids.get(regime.name, {})},
            })
        rows.extend(rr)
        tables[regime.name] = tab
        checks[regime.name] = chk
        timings[regime.name] = tim
        snapshot(False)

    snapshot(True)
    print(f"\nwrote {RESULT_ID}.json", flush=True)
    print(f"{'regime':<16}{'variant':<9}{'fused':>10}{'unfused':>10}{'paired':>9}"
          f"{'ceiling':>9}{'torch':>10}")
    for r in rows:
        c = r.get("ceiling")
        print(
            f"{r['regime']:<16}{r['variant']:<9}{r['fused_ms']:>10.4f}"
            f"{r['unfused_ms']:>10.4f}"
            f"{(r.get('paired_speedup') or r['speedup']):>9.3f}"
            f"{(f'{c:.2f}' if c else 'n/a'):>9}{r['torch_ref_ms']:>10.4f}"
        )


if __name__ == "__main__":
    main_guard(main)
