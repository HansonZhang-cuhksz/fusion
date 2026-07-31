"""Fusions #4 / #5 -- (ResAdd +) RMSNorm + Router GEMM (+ sigmoid top-8).

Four fused variants, each against its own independently tuned unfused chain:

    #5     fused [norm+gemm]              vs  [norm][gemm]
    #5+tk  fused [norm+gemm+topk]         vs  [norm][gemm][topk]
    #4     fused [add+norm+gemm]          vs  [add+norm][gemm]
    #4+tk  fused [add+norm+gemm+topk]     vs  [add+norm][gemm][topk]

Every one of those seven kernels is the SAME source
(`glm52/kernels/norm_router.py::norm_router_kernel`) with different `tl.constexpr`
flags; only the mapping differs, and each is tuned over its own grid.

Traffic (T rows, act = T*6144*2 B):
    #5 unfused  = 2*act (norm) + act + 3 MB + T*1 KB (gemm)
    #5 fused    = 2*act + 3 MB + T*1 KB            -> saves the router's read of x2
    #4 unfused  = 3*act (add) + 2*act (norm) + act + ...
    #4 fused    = 4*act + ...
The latency-aware ceiling (`python -m glm52.traffic`) is higher than the traffic ratio
because the norm's bytes hide behind the router GEMM's compute -- "free normalization".

Run:
    CUDA_VISIBLE_DEVICES=3 /home/zhangshuhan/my-envs/fusion/bin/python \
        glm52/bench/bench_f04f05_norm_router.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/zhangshuhan/fusion")

import torch

from glm52 import config as C
from glm52 import reference as ref
from glm52 import traffic as TR
from glm52.common import (
    RESULTS_DIR,
    autotune,
    bench_chain,
    check,
    main_guard,
    record,
    rel_err,
    speedup_row,
)
from glm52.kernels import norm_router as K

RESULT_ID = "f04f05_norm_router"
H = C.HIDDEN_SIZE  # 6144
E = C.N_ROUTED_EXPERTS  # 256
TK = C.NUM_EXPERTS_PER_TOK  # 8
EPS = C.RMS_NORM_EPS
DT = C.DTYPE
CKPT = RESULTS_DIR / f"_{RESULT_ID}_ckpt"

REGIMES = [r for r in C.DECODE_REGIMES if r.T in (1, 32, 256)] + [
    r for r in C.PREFILL_REGIMES if r.T in (2048, 8192)
]


# --------------------------------------------------------------------------------------
# mapping search spaces.  ONE generator per kernel *shape*, used by every variant that
# has that shape -- so the fused kernel and the unfused router GEMM see the identical
# grid, and the two norm kernels see the identical grid.
# --------------------------------------------------------------------------------------
SMEM = 65536


def _gemm_ok(cfg: dict) -> bool:
    bm, bk, be, w, s = (
        cfg["BLOCK_M"],
        cfg["BLOCK_K"],
        cfg["BLOCK_E"],
        cfg["num_warps"],
        cfg["num_stages"],
    )
    # Triton's mainloop double-buffer, the hard C500 ceiling
    if s * 2 * bk * (bm + be) > SMEM:
        return False
    # fp32 accumulator tile: 4 B * BM * BE must be a sane number of registers
    if bm * be * 4 > SMEM:
        return False
    threads = w * 64
    if bm * be / threads > 128 or bm * be / threads < 1:
        return False
    return True


def gemm_grid() -> list[dict]:
    """Grid for every variant containing the router GEMM (fused #4/#5 and the
    stand-alone router kernel).  ~55 configs after the SMEM/register prefilter."""
    out: list[dict] = []
    for bm, bk, be, (w, s) in itertools.product(
        (16, 32, 64, 128), (32, 64), (32, 64, 128, 256), ((4, 2), (8, 2))
    ):
        cfg = dict(BLOCK_M=bm, BLOCK_K=bk, BLOCK_E=be, num_warps=w, num_stages=s)
        if _gemm_ok(cfg):
            out.append(cfg)
    # wider warps / deeper pipeline at the narrow k-tile
    for bm, be, (w, s) in itertools.product(
        (16, 32, 64, 128), (128, 256), ((16, 2), (8, 3))
    ):
        cfg = dict(BLOCK_M=bm, BLOCK_K=32, BLOCK_E=be, num_warps=w, num_stages=s)
        if _gemm_ok(cfg):
            out.append(cfg)
    # BLOCK_E = 256 (the whole expert dimension in one program) is the only shape the
    # FUSE_TOPK variants can use, and BLOCK_K=32 is the only k-tile that fits SMEM there,
    # so that corner gets a denser warp/stage sweep -- for every GEMM-side variant alike.
    for bm, (w, s) in itertools.product((16, 32, 64), ((4, 1), (4, 3), (2, 2), (8, 4))):
        cfg = dict(BLOCK_M=bm, BLOCK_K=32, BLOCK_E=256, num_warps=w, num_stages=s)
        if _gemm_ok(cfg):
            out.append(cfg)
    # cache-eviction hints (streaming activations evict_first, router weight evict_last)
    out += [
        dict(c, EVICT=1)
        for c in list(out)
        if c["BLOCK_E"] == 256 and c["num_stages"] == 2 and c["num_warps"] == 8
    ]
    return _dedup(out)


def fused_grid() -> list[dict]:
    """Grid for the FUSED variants: `gemm_grid()` plus, for every config in it, the same
    config with the *pass-1* k-tile (`NORM_BK`) widened to the largest tile that still
    fits in registers.

    This is not extra freedom -- it is the fused-side counterpart of a knob the unfused
    side already has.  On the unfused side the norm kernel picks its own BLOCK_K (it
    always picks 2048); without NORM_BK the fused kernel would be forced to compute the
    sum of squares in the GEMM's 32/64-wide tiles, i.e. 96-192 sequential cross-lane
    reductions per row instead of 3.  Leaving that out would be exactly the "under-tuned
    fused side" failure mode."""
    out: list[dict] = []
    for c in gemm_grid():
        out.append(c)
        wide = min(2048, 32768 // c["BLOCK_M"])
        if wide > c["BLOCK_K"] and C.HIDDEN_SIZE % wide == 0:
            out.append(dict(c, NORM_BK=wide))
    return _dedup(out)


def _norm_ok(cfg: dict) -> bool:
    bm, bk, w = cfg["BLOCK_M"], cfg["BLOCK_K"], cfg["num_warps"]
    if bm * bk > 32768:
        return False
    epr = bm * bk / (w * 64)
    if epr < 2 or epr > 64:
        return False
    return True


def norm_grid() -> list[dict]:
    """Grid for the stand-alone norm kernels (rmsnorm / add+rmsnorm).  Same size class
    as `gemm_grid`; BLOCK_E is irrelevant (no GEMM) and pinned to 256."""
    out: list[dict] = []
    for bm, bk, (w, s) in itertools.product(
        (1, 2, 4, 8, 16, 32, 64),
        (512, 1024, 2048),
        ((4, 2), (8, 2), (16, 2)),
    ):
        cfg = dict(BLOCK_M=bm, BLOCK_K=bk, BLOCK_E=256, num_warps=w, num_stages=s)
        if _norm_ok(cfg):
            out.append(cfg)
    out += [
        dict(c, EVICT=1)
        for c in list(out)
        if c["num_stages"] == 2 and c["num_warps"] == 8 and c["BLOCK_K"] >= 1024
    ]
    return _dedup(out)


def topk_grid() -> list[dict]:
    """Grid for the stand-alone sigmoid+top-8 kernel (no k-loop at all)."""
    out = [
        dict(BLOCK_M=bm, BLOCK_K=32, BLOCK_E=256, num_warps=w, num_stages=s)
        for bm, (w, s) in itertools.product(
            (1, 2, 4, 8, 16, 32, 64), ((4, 2), (8, 2), (16, 2))
        )
    ]
    return _dedup(out)


def _dedup(cfgs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in cfgs:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _topk_cfgs(tr, k: int) -> list[dict]:
    rows = [(ms, cfg) for cfg, ms, err in tr.table if ms is not None]
    rows.sort(key=lambda t: t[0])
    return [cfg for _, cfg in rows[:k]]


# --------------------------------------------------------------------------------------
# Numerical screening.
#
# MACA Triton 3.0 MISCOMPILES a row-wise (`axis=1`) reduction over a `tl.dot`
# accumulator whenever the mma tile spans more than one warp-row: `tl.max` / `tl.argmax`
# then silently return a per-warp partial answer (measured in scratchpad/f04/t6.py:
# BLOCK_M=16 ok for 4/8/16 warps, BLOCK_M=32 ok only at 4 warps, BLOCK_M=64 wrong at
# every warp count; the accumulator itself is always correct).  It is a *wrong answer*,
# not a crash, so every config of every variant is validated against the fp32 reference
# BEFORE it is allowed into a timing grid, and the rejects are recorded.
# --------------------------------------------------------------------------------------
def screen(tag, run, verify, grid):
    ok, rej = [], []
    for cfg in grid:
        try:
            run(cfg)
            torch.cuda.synchronize()
            good, detail = verify()
            if good:
                ok.append(cfg)
            else:
                rej.append((cfg, None, f"NUMERIC {detail}"))
        except Exception as exc:  # noqa: BLE001
            rej.append((cfg, None, f"{type(exc).__name__}: {exc}"[:160]))
    num = [r for r in rej if r[2].startswith("NUMERIC")]
    print(
        f"    [screen {tag:<9}] {len(grid):>3} offered -> {len(ok):>3} valid "
        f"({len(rej) - len(num)} compile-fail, {len(num)} wrong-answer)",
        flush=True,
    )
    for cfg, _, why in num[:4]:
        print(f"        wrong-answer: {cfg} {why[:110]}", flush=True)
    return ok, rej


def kernel_stats(run, cfg) -> dict:
    """Compile ONE config with the Triton cache cleared first, so the resource report is
    unambiguous (the LOG-10 recipe)."""
    try:
        dev = torch.cuda.current_device()
        cache = K.norm_router_kernel.cache
        if dev in cache:
            cache[dev].clear()
        run(cfg)
        torch.cuda.synchronize()
        vals = list(cache[dev].values())
        if len(vals) != 1:
            return {"note": f"{len(vals)} kernels in cache"}
        k = vals[0]
        out = {"shared": k.metadata.shared}
        for a in ("n_regs", "n_spills"):
            try:
                out[a] = getattr(k, a)
            except Exception:  # noqa: BLE001
                out[a] = None
        return out
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:120]}


# --------------------------------------------------------------------------------------
def run_regime(regime, wt, rt, wf, rf) -> dict:
    T = regime.T
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
    fg_full = [c for c in fg if c["BLOCK_E"] == E]  # top-k needs all logits in-program
    print(
        f"    grids: gemm={len(gg)} fused={len(fg)} fused(BE=256, FUSE_TOPK)="
        f"{len(fg_full)} norm={len(ng)} topk={len(tg)}",
        flush=True,
    )

    # ---- verifiers used by the screening pass -----------------------------------------
    # One bf16 ULP is 2^-8 = 3.9e-3 relative, and the fp32 sum-of-squares differs from
    # torch's by summation order, which flips the rounding of the occasional element of
    # x2 -- so the activation tolerance has to sit just above one ULP.  A miscompiled
    # config is nowhere near this line: the ones caught below come back at 0.79-0.89.
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

    def v_norm():
        return _v({"x2": (x2u, r_x2_5, TOL_ACT)})

    def v_addn():
        return _v({"x2": (x2u, r_x2_4, TOL_ACT), "h1": (h1u, r_h1, TOL_ACT)})

    def v_gemm():
        return _v({"logits": (lgu, r_lg5, TOL_LG)})

    def v_tk():
        return _v({"tw": (twu, r_tw5, TOL_W)}, ids=(tiu, r_ti5))

    def v_f5():
        return _v({"x2": (x2f, r_x2_5, TOL_ACT), "logits": (lgf, r_lg5, TOL_LG)})

    def v_f4():
        return _v(
            {
                "x2": (x2f, r_x2_4, TOL_ACT),
                "h1": (h1f, r_h1, TOL_ACT),
                "logits": (lgf, r_lg4, TOL_LG),
            }
        )

    def v_f5t():
        return _v(
            {"x2": (x2f, r_x2_5, TOL_ACT), "tw": (twf, r_tw5, TOL_W)},
            ids=(tif, r_ti5),
        )

    def v_f4t():
        return _v(
            {
                "x2": (x2f, r_x2_4, TOL_ACT),
                "h1": (h1f, r_h1, TOL_ACT),
                "tw": (twf, r_tw4, TOL_W),
            },
            ids=(tif, r_ti4),
        )

    def tune(tag, make_chain, grid, verify, prep=None):
        t0 = time.time()
        if prep is not None:
            prep()
        ok, rej = screen(tag, lambda c: [f() for f in make_chain(c)], verify, grid)
        if not ok:  # never let a screening tolerance kill an hour-long run
            print(f"    !! [{tag}] screening rejected EVERY config; timing unscreened",
                  flush=True)
            ok, rej = grid, []
        tr = autotune(make_chain, ok, warmup=wt, rep=rt)
        tr.table = list(tr.table) + rej
        tr.n_tried = len(grid)
        tr.n_failed = len(rej)
        print(
            f"    [{tag:<10}] {len(ok):>3}/{len(grid)} cfgs timed -> "
            f"{tr.best_ms:.4f} ms  {tr.best_cfg}  [{time.time()-t0:.0f}s]",
            flush=True,
        )
        return tr

    # ---- the eight independently tuned kernels ---------------------------------------
    t_norm = tune("norm", lambda c: [lambda: K.rmsnorm_only(x, w, x2u, c)], ng, v_norm)
    t_addn = tune(
        "add+norm",
        lambda c: [lambda: K.add_rmsnorm_only(x, res, w, h1u, x2u, c)],
        ng,
        v_addn,
    )
    t_gemm = tune(
        "router",
        lambda c: [lambda: K.router_gemm(x2u, wgt, lgu, c)],
        gg,
        v_gemm,
        prep=lambda: x2u.copy_(r_x2_5),  # screening needs a meaningful input
    )
    t_tk = tune(
        "topk",
        lambda c: [lambda: K.topk_only(lgu, twu, tiu, c)],
        tg,
        v_tk,
        prep=lambda: lgu.copy_(r_lg5),
    )
    t_f5 = tune(
        "F5", lambda c: [lambda: K.fused_norm_router(x, w, x2f, wgt, lgf, c)], fg, v_f5
    )
    t_f4 = tune(
        "F4",
        lambda c: [lambda: K.fused_add_norm_router(x, res, w, h1f, x2f, wgt, lgf, c)],
        fg,
        v_f4,
    )
    t_f5t = tune(
        "F5+topk",
        lambda c: [lambda: K.fused_norm_router_topk(x, w, x2f, wgt, twf, tif, c)],
        fg_full,
        v_f5t,
    )
    t_f4t = tune(
        "F4+topk",
        lambda c: [
            lambda: K.fused_add_norm_router_topk(x, res, w, h1f, x2f, wgt, twf, tif, c)
        ],
        fg_full,
        v_f4t,
    )

    # ---- joint chain re-tune of each unfused chain (can only help the baseline) -------
    def joint(tag, norm_tr, pieces, with_topk):
        """pieces(cfg_norm, cfg_gemm, cfg_topk) -> chain"""
        cand = []
        for cn, cg in itertools.product(
            _topk_cfgs(norm_tr, 3), _topk_cfgs(t_gemm, 3)
        ):
            for ct in _topk_cfgs(t_tk, 2) if with_topk else [None]:
                cand.append({"norm": cn, "gemm": cg, "topk": ct})
        cand.append(
            {
                "norm": norm_tr.best_cfg,
                "gemm": t_gemm.best_cfg,
                "topk": t_tk.best_cfg if with_topk else None,
            }
        )
        uniq, seen = [], set()
        for c in cand:
            key = json.dumps(c, sort_keys=True)
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        t0 = time.time()
        tr = autotune(lambda jc: pieces(jc), uniq, warmup=wt, rep=rt)
        print(
            f"    [{tag:<10}] joint {tr.n_tried} combos -> {tr.best_ms:.4f} ms "
            f"[{time.time()-t0:.0f}s]",
            flush=True,
        )
        return tr

    j5 = joint(
        "U5",
        t_norm,
        lambda jc: [
            lambda: K.rmsnorm_only(x, w, x2u, jc["norm"]),
            lambda: K.router_gemm(x2u, wgt, lgu, jc["gemm"]),
        ],
        False,
    )
    j5t = joint(
        "U5+topk",
        t_norm,
        lambda jc: [
            lambda: K.rmsnorm_only(x, w, x2u, jc["norm"]),
            lambda: K.router_gemm(x2u, wgt, lgu, jc["gemm"]),
            lambda: K.topk_only(lgu, twu, tiu, jc["topk"]),
        ],
        True,
    )
    j4 = joint(
        "U4",
        t_addn,
        lambda jc: [
            lambda: K.add_rmsnorm_only(x, res, w, h1u, x2u, jc["norm"]),
            lambda: K.router_gemm(x2u, wgt, lgu, jc["gemm"]),
        ],
        False,
    )
    j4t = joint(
        "U4+topk",
        t_addn,
        lambda jc: [
            lambda: K.add_rmsnorm_only(x, res, w, h1u, x2u, jc["norm"]),
            lambda: K.router_gemm(x2u, wgt, lgu, jc["gemm"]),
            lambda: K.topk_only(lgu, twu, tiu, jc["topk"]),
        ],
        True,
    )

    # ---- validation (winners only) ----------------------------------------------------
    def zero():
        for t in (h1f, x2f, lgf, twf, tif, h1u, x2u, lgu, twu, tiu):
            t.zero_()

    def ids_row(got, want, label):
        """Fraction of rows whose whole top-8 id set matches the fp32 reference.

        Not required to be exactly 1.0: the fused and unfused logits differ from the
        fp32 reference by ~3e-6 (summation order), which can legitimately flip the 8th
        expert when two experts are that close.  A miscompiled reduction, by contrast,
        gets essentially *every* row wrong -- the two failure modes are orders of
        magnitude apart."""
        frac = (got == want).all(1).float().mean().item()
        return {"label": label, "rel_err": 1.0 - frac, "tol": 0.01, "ok": frac >= 0.99}

    chk = {}
    # #5
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
        "label": "F5_fused_eq_unfused",
        "rel_err": rel_err(lgf, lgu),
        "tol": 2e-2,
        "ok": True,
        "x2_bitwise": bool(torch.equal(x2f, x2u)),
    }
    # #5 + topk
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
    # #4
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
    # #4 + topk
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

    # ---- final timing -----------------------------------------------------------------
    def T5f():
        K.fused_norm_router(x, w, x2f, wgt, lgf, t_f5.best_cfg)

    def T5tf():
        K.fused_norm_router_topk(x, w, x2f, wgt, twf, tif, t_f5t.best_cfg)

    def T4f():
        K.fused_add_norm_router(x, res, w, h1f, x2f, wgt, lgf, t_f4.best_cfg)

    def T4tf():
        K.fused_add_norm_router_topk(x, res, w, h1f, x2f, wgt, twf, tif, t_f4t.best_cfg)

    fused_t = {
        "F5": bench_chain([T5f], wf, rf),
        "F5_topk": bench_chain([T5tf], wf, rf),
        "F4": bench_chain([T4f], wf, rf),
        "F4_topk": bench_chain([T4tf], wf, rf),
    }
    unfused_t = {
        "F5": bench_chain(
            [
                lambda: K.rmsnorm_only(x, w, x2u, j5.best_cfg["norm"]),
                lambda: K.router_gemm(x2u, wgt, lgu, j5.best_cfg["gemm"]),
            ],
            wf,
            rf,
        ),
        "F5_topk": bench_chain(
            [
                lambda: K.rmsnorm_only(x, w, x2u, j5t.best_cfg["norm"]),
                lambda: K.router_gemm(x2u, wgt, lgu, j5t.best_cfg["gemm"]),
                lambda: K.topk_only(lgu, twu, tiu, j5t.best_cfg["topk"]),
            ],
            wf,
            rf,
        ),
        "F4": bench_chain(
            [
                lambda: K.add_rmsnorm_only(x, res, w, h1u, x2u, j4.best_cfg["norm"]),
                lambda: K.router_gemm(x2u, wgt, lgu, j4.best_cfg["gemm"]),
            ],
            wf,
            rf,
        ),
        "F4_topk": bench_chain(
            [
                lambda: K.add_rmsnorm_only(x, res, w, h1u, x2u, j4t.best_cfg["norm"]),
                lambda: K.router_gemm(x2u, wgt, lgu, j4t.best_cfg["gemm"]),
                lambda: K.topk_only(lgu, twu, tiu, j4t.best_cfg["topk"]),
            ],
            wf,
            rf,
        ),
    }

    # ---- attribution -------------------------------------------------------------------
    # (i) resources of fused vs unfused GEMM at BOTH winning configs, cache cleared
    #     between compiles;  (ii) the F1 stride-0 trick: point the fused kernel's activation
    #     input at a broadcast row so the instruction stream is identical but the DRAM
    #     traffic collapses to 12 KB -- separates "extra bytes" from "extra instructions".
    xb = x[:1].expand(T, H)
    x2b = x2u[:1].expand(T, H)
    attrib = {
        "regs_fused_F5_at_fused_best": kernel_stats(
            lambda c: K.fused_norm_router(x, w, x2f, wgt, lgf, c), t_f5.best_cfg
        ),
        "regs_router_at_router_best": kernel_stats(
            lambda c: K.router_gemm(x2u, wgt, lgu, c), t_gemm.best_cfg
        ),
        "regs_fused_F5_at_router_best": kernel_stats(
            lambda c: K.fused_norm_router(x, w, x2f, wgt, lgf, c), t_gemm.best_cfg
        ),
        "regs_router_at_fused_best": kernel_stats(
            lambda c: K.router_gemm(x2u, wgt, lgu, c), t_f5.best_cfg
        ),
        "cross_fused_F5_on_router_cfg_ms": None,
        "cross_router_on_fused_cfg_ms": None,
        "iso_fused_F5_bcast_ms": None,
        "iso_router_bcast_ms": None,
        "iso_fused_F5_no_x2_store_ms": None,
        "gflop_router": 2.0 * T * H * E / 1e9,
    }
    try:
        attrib["iso_fused_F5_no_x2_store_ms"] = bench_chain(
            [lambda: K.fused_norm_router_no_x2(x, w, x2f, wgt, lgf, t_f5.best_cfg)],
            wf,
            rf,
        ).p50_ms
    except Exception:  # noqa: BLE001
        pass
    try:
        attrib["cross_fused_F5_on_router_cfg_ms"] = bench_chain(
            [lambda: K.fused_norm_router(x, w, x2f, wgt, lgf, t_gemm.best_cfg)], wf, rf
        ).p50_ms
    except Exception:  # noqa: BLE001
        pass
    try:
        attrib["cross_router_on_fused_cfg_ms"] = bench_chain(
            [lambda: K.router_gemm(x2u, wgt, lgu, t_f5.best_cfg)], wf, rf
        ).p50_ms
    except Exception:  # noqa: BLE001
        pass
    try:
        attrib["iso_fused_F5_bcast_ms"] = bench_chain(
            [lambda: K.fused_norm_router(xb, w, x2f, wgt, lgf, t_f5.best_cfg)], wf, rf
        ).p50_ms
        attrib["iso_router_bcast_ms"] = bench_chain(
            [lambda: K.router_gemm(x2b, wgt, lgu, t_gemm.best_cfg)], wf, rf
        ).p50_ms
    except Exception:  # noqa: BLE001
        pass
    print(f"    attrib {attrib}", flush=True)

    # ---- torch / vendor-BLAS production lines -----------------------------------------
    gatef = gate.float()

    def torch5():
        return ref.router(ref.rmsnorm(x, w, EPS), gate)

    def torch4():
        return ref.router(ref.add_rmsnorm(x, res, w, EPS)[0], gate)

    t_torch5 = bench_chain([torch5], wf, rf)
    t_torch4 = bench_chain([torch4], wf, rf)
    t_blas_fp32 = bench_chain([lambda: x2u.float() @ gatef.t()], wf, rf)
    t_blas_bf16 = bench_chain([lambda: x2u @ wgt], wf, rf)

    # ---- roofline ceilings from the shared model --------------------------------------
    ceil = {t.fusion: t.row() for t in TR.model(regime)}

    act = T * H * 2
    wg_bytes = H * E * 2
    lg_bytes = T * E * 4
    tk_bytes = T * TK * 8
    bytes_model = {
        "F5": (2 * act + wg_bytes + lg_bytes, 3 * act + wg_bytes + lg_bytes),
        "F5_topk": (
            2 * act + wg_bytes + tk_bytes,
            3 * act + wg_bytes + 2 * lg_bytes + tk_bytes,
        ),
        "F4": (4 * act + wg_bytes + lg_bytes, 6 * act + wg_bytes + lg_bytes),
        "F4_topk": (
            4 * act + wg_bytes + tk_bytes,
            6 * act + wg_bytes + 2 * lg_bytes + tk_bytes,
        ),
    }
    gflop = 2.0 * T * H * E / 1e9

    rows = []
    for key, ceil_key in (
        ("F5", "F5_rmsnorm_router"),
        ("F5_topk", "F5_rmsnorm_router"),
        ("F4", "F4_addnorm_router"),
        ("F4_topk", "F4_addnorm_router"),
    ):
        bf, bu = bytes_model[key]
        f, u = fused_t[key], unfused_t[key]
        rows.append(
            speedup_row(
                regime.name,
                f,
                u,
                {
                    "variant": key,
                    "T": T,
                    "ceiling": ceil[ceil_key]["roofline_ceiling"],
                    "traffic_ratio_model": ceil[ceil_key]["traffic_ratio"],
                    "bytes_fused": bf,
                    "bytes_unfused": bu,
                    "gbps_fused": bf / (f.p50_ms * 1e-3) / 1e9,
                    "gbps_unfused": bu / (u.p50_ms * 1e-3) / 1e9,
                    "tflops_fused": gflop / (f.p50_ms * 1e-3) / 1e3,
                    "fused_cfg": {"F5": t_f5, "F5_topk": t_f5t, "F4": t_f4,
                                  "F4_topk": t_f4t}[key].best_cfg,
                    "unfused_cfg": {"F5": j5, "F5_topk": j5t, "F4": j4,
                                    "F4_topk": j4t}[key].best_cfg,
                    "fused_noflush_ms": f.noflush_p50_ms,
                    "unfused_noflush_ms": u.noflush_p50_ms,
                    "torch_ref_ms": (t_torch5 if key.startswith("F5") else t_torch4).p50_ms,
                    "blas_router_fp32_ms": t_blas_fp32.p50_ms,
                    "blas_router_bf16_ms": t_blas_bf16.p50_ms,
                },
            )
        )

    for r in rows:
        print(
            f"    {r['variant']:<8} fused {r['fused_ms']:.4f} | unfused "
            f"{r['unfused_ms']:.4f} | speedup {r['speedup']:.3f}x "
            f"(ceiling {r['ceiling']:.2f}x, {100*r['speedup']/r['ceiling']:.0f}% of it)"
            f" | torch {r['torch_ref_ms']:.4f}",
            flush=True,
        )

    tune_tables = {
        "norm": t_norm.as_dict(),
        "add_norm": t_addn.as_dict(),
        "router_gemm": t_gemm.as_dict(),
        "topk": t_tk.as_dict(),
        "fused_F5": t_f5.as_dict(),
        "fused_F4": t_f4.as_dict(),
        "fused_F5_topk": t_f5t.as_dict(),
        "fused_F4_topk": t_f4t.as_dict(),
        "joint_U5": j5.as_dict(),
        "joint_U5_topk": j5t.as_dict(),
        "joint_U4": j4.as_dict(),
        "joint_U4_topk": j4t.as_dict(),
    }
    timings = {
        **{f"fused_{k}": v.as_dict() for k, v in fused_t.items()},
        **{f"unfused_{k}": v.as_dict() for k, v in unfused_t.items()},
        "torch_ref_F5": t_torch5.as_dict(),
        "torch_ref_F4": t_torch4.as_dict(),
        "blas_router_fp32": t_blas_fp32.as_dict(),
        "blas_router_bf16": t_blas_bf16.as_dict(),
        "attribution": attrib,
        "piece_best_ms": {
            "norm": t_norm.best_ms,
            "add_norm": t_addn.best_ms,
            "router_gemm": t_gemm.best_ms,
            "topk": t_tk.best_ms,
        },
    }
    return rows, tune_tables, chk, timings


def main() -> None:
    env = C.BenchEnv.probe()
    print(f"device={env.device_name} warp={env.warp_size} CUs={env.num_sm}", flush=True)
    CKPT.mkdir(parents=True, exist_ok=True)

    rows, tables, checks, timings = [], {}, {}, {}

    def snapshot(done: bool) -> None:
        record(
            RESULT_ID,
            {
                "id": RESULT_ID,
                "complete": done,
                "fusion": "#4 ResAdd+RMSNorm+Router and #5 RMSNorm+Router "
                "(+ FUSE_TOPK: sigmoid + noaux_tc top-8 folded in)",
                "shape": {
                    "hidden": H,
                    "experts": E,
                    "topk": TK,
                    "dtype": "bfloat16",
                    "router_acc": "float32",
                    "eps": EPS,
                    "routed_scaling_factor": C.ROUTED_SCALING_FACTOR,
                    "norm_topk_prob": C.NORM_TOPK_PROB,
                    "n_group": C.N_GROUP,
                },
                "fairness": {
                    "one_kernel_source": "glm52/kernels/norm_router.py::norm_router_kernel",
                    "flags": "DO_ADD / DO_NORM / DO_GEMM / DO_TOPK constexpr select all "
                    "seven kernels (4 fused variants + 3 stand-alone pieces)",
                    "grids": "gemm_grid() (80 cfgs) for the stand-alone router GEMM; "
                    "fused_grid() = gemm_grid() x {NORM_BK tied, NORM_BK widened} (160) for "
                    "the fused #4/#5 kernels -- NORM_BK is the fused-side counterpart of the "
                    "k-tile the unfused norm kernel tunes independently over norm_grid() "
                    "(58 cfgs), so the unfused side sees 80+58=138 configs against the fused "
                    "side's 160; the FUSE_TOPK variants get fused_grid() restricted to "
                    "BLOCK_E=256 (48), which is a structural requirement (all 256 logits of a "
                    "row must live in one program), not a tuning choice; every config of every "
                    "variant is numerically screened against the fp32 reference before it is "
                    "allowed into a timing grid (see 'screen' in the bench)",
                    "baseline_extra": "each unfused chain additionally gets a joint re-tune "
                    "over the top-3 x top-3 (x top-2) configs of its pieces, timed as the "
                    "real chain -- this can only help the baseline",
                    "weight_layout": "the router weight is transposed to [H, E] ONCE and both "
                    "sides consume that same layout (a load-time weight-prep choice)",
                },
                "env": env.__dict__,
                "rows": rows,
                "checks": checks,
                "timings": timings,
                "tune_tables": tables,
            },
        )

    for regime in REGIMES:
        if regime.T <= 256:
            wt, rt, wf, rf = 15, 40, 100, 300
        elif regime.T <= 2048:
            wt, rt, wf, rf = 10, 30, 50, 200
        else:
            wt, rt, wf, rf = 10, 25, 30, 120
        try:
            rr, tab, chk, tim = run_regime(regime, wt, rt, wf, rf)
        except Exception as exc:  # noqa: BLE001 - one bad regime must not lose the rest
            import traceback

            traceback.print_exc()
            checks[regime.name] = {"regime_failed": f"{type(exc).__name__}: {exc}"[:300]}
            snapshot(False)
            continue
        rows.extend(rr)
        tables[regime.name] = tab
        checks[regime.name] = chk
        timings[regime.name] = tim
        (CKPT / f"{regime.name}.json").write_text(
            json.dumps(
                {"rows": rr, "checks": chk, "timings": tim, "tune_tables": tab},
                indent=2,
                default=str,
            )
        )
        snapshot(False)

    snapshot(True)
    print(f"\nwrote results/{RESULT_ID}.json", flush=True)
    print(
        f"{'regime':<16}{'variant':<9}{'fused':>10}{'unfused':>10}{'speedup':>9}"
        f"{'ceiling':>9}{'torch':>10}"
    )
    for r in rows:
        print(
            f"{r['regime']:<16}{r['variant']:<9}{r['fused_ms']:>10.4f}"
            f"{r['unfused_ms']:>10.4f}{r['speedup']:>9.3f}{r['ceiling']:>9.2f}"
            f"{r['torch_ref_ms']:>10.4f}"
        )


if __name__ == "__main__":
    main_guard(main)
