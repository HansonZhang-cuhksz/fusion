"""Fusion #11 -- **Lazy Pre-Norm**: RMSNorm fused into a GEMM as a prologue.

Two consumers of `post_attention_layernorm` have K == hidden == 6144 and both are
benchmarked here:

  F11a  routed-expert w13 grouped GEMM   x2 @ w13[e]^T   ([T*8, 6144] x [6144, 4096])
  F11b  router GEMM                      x2 @ W_gate^T   ([T,   6144] x [6144,  256])

For each:
    UNFUSED : rmsnorm kernel (read h1, write x2)  ->  GEMM reading x2, raw weight
    FUSED   : the SAME GEMM reading h1 directly, with the rmsnorm weight pre-folded into
              the GEMM weight offline; sum-of-squares rides the k-loop and rstd is applied
              as an epilogue.  x2 is never materialized.

**x2 materialization -- explicit choice.**  x2 feeds BOTH consumers (and the shared
expert).  We take option (ii) from the brief: *all* K==6144 consumers are fused, so x2 is
genuinely dead and never written.  That is why the per-family rows below are also rolled
up into a `combined` row, which is the only end-to-end-honest number:

    combined UNFUSED = [norm kernel] + [router GEMM] + [w13 grouped GEMM]
    combined FUSED   =                 [router GEMM fused] + [w13 GEMM fused]

Charging the single norm kernel to F11a and to F11b separately (as glm52/traffic.py does)
double-counts it; the combined row does not.

Run:
    CUDA_VISIBLE_DEVICES=1 /home/zhangshuhan/my-envs/fusion/bin/python \
        glm52/bench/bench_f11_lazy_prenorm.py [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from glm52 import config as C  # noqa: E402
from glm52 import reference as R  # noqa: E402
from glm52 import traffic as TR  # noqa: E402
from glm52.common import (  # noqa: E402
    autotune,
    bench_chain,
    check,
    main_guard,
    record,
    rel_err,
    speedup_row,
)
from glm52.kernels import add_rmsnorm as NK  # noqa: E402
from glm52.kernels import lazy_prenorm as K  # noqa: E402

RESULT_ID = "f11_lazy_prenorm"
SMEM_LIMIT = 65536
H = C.HIDDEN_SIZE            # 6144
I = C.MOE_INTERMEDIATE_SIZE  # 2048
NW13 = C.W13_N               # 4096
E = C.N_ROUTED_EXPERTS       # 256
TOPK = C.NUM_EXPERTS_PER_TOK  # 8
EPS = C.RMS_NORM_EPS
DT = C.DTYPE

REGIMES = [r for r in C.DECODE_REGIMES if r.T in (1, 32, 256)] + [
    r for r in C.PREFILL_REGIMES if r.T in (2048, 8192)
]


# ======================================================================================
# Mapping search spaces.  The SAME generator is used for the fused and the unfused side of
# each family -- `fused` is not even a parameter, because unlike F6 the lazy-prenorm
# kernel stages no extra tile and its SMEM footprint is identical.  So the two coarse
# grids are *literally the same config list*; only the refine neighbourhoods differ,
# because they are centred on each side's own coarse winner.
# ======================================================================================
def _ok(cfg: dict, max_bn: int, max_bm: int, acc_lo=2, acc_hi=128) -> bool:
    if cfg["BLOCK_N"] > max_bn or cfg["BLOCK_M"] > max_bm:
        return False
    if K.smem_bytes(cfg) > SMEM_LIMIT:
        return False
    acc_per_lane = cfg["BLOCK_M"] * cfg["BLOCK_N"] / (cfg["num_warps"] * 64)
    return acc_lo <= acc_per_lane <= acc_hi


def router_grid(T: int) -> list[dict]:
    """Router GEMM: M=T, N=256, K=6144."""
    max_bm = max(16, 1 << (max(T, 1) - 1).bit_length())
    out = []
    for bm, bn, bk, w, s in itertools.product(
        (16, 32, 64, 128), (32, 64, 128, 256), (32, 64, 128), (4, 8), (2, 3)
    ):
        cfg = dict(
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, GROUP_M=8
        )
        if _ok(cfg, max_bn=256, max_bm=max_bm):
            out.append(cfg)
    return out


def moe_grid(big: bool) -> list[dict]:
    """w13 grouped GEMM: M=T*8 (padded), N=4096, K=6144.  Same shape rules F6 used."""
    if big:
        bms, bns, bks, warps = (32, 64, 128), (64, 128, 256), (32, 64, 128), (4, 8, 16)
    else:
        bms, bns, bks, warps = (
            (16, 32, 64, 128),
            (64, 128, 256),
            (32, 64, 128),
            (4, 8, 16),
        )
    out = []
    for bm, bn, bk, w, s in itertools.product(bms, bns, bks, warps, (2, 3)):
        cfg = dict(
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, GROUP_M=8
        )
        if _ok(cfg, max_bn=4096, max_bm=4096, acc_lo=4):
            out.append(cfg)
    return out


def refine(best: dict, max_bn: int, max_bm: int, acc_lo=2) -> list[dict]:
    """Same neighbourhood rule for both sides: half/same/double in BM, BN, warps at the
    winning BK/stages; a BK x stages sweep at the winning shape; a GROUP_M sweep."""

    def nb(v, lo, hi):
        return sorted({max(lo, v // 2), v, min(hi, v * 2)})

    cands = []
    for bm in nb(best["BLOCK_M"], 16, 256):
        for bn in nb(best["BLOCK_N"], 32, 256 if max_bn <= 256 else 256):
            for w in nb(best["num_warps"], 2, 16):
                cands.append((bm, bn, best["BLOCK_K"], w, best["num_stages"], 8))
    for bk in nb(best["BLOCK_K"], 32, 128):
        for s in (2, 3, 4):
            cands.append(
                (best["BLOCK_M"], best["BLOCK_N"], bk, best["num_warps"], s, 8)
            )
    for g in (1, 4, 8, 16):
        cands.append(
            (
                best["BLOCK_M"],
                best["BLOCK_N"],
                best["BLOCK_K"],
                best["num_warps"],
                best["num_stages"],
                g,
            )
        )
    out, seen = [], set()
    for bm, bn, bk, w, s, g in cands:
        cfg = dict(
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, GROUP_M=g
        )
        key = tuple(sorted(cfg.items()))
        if key in seen or not _ok(cfg, max_bn, max_bm, acc_lo=acc_lo):
            continue
        seen.add(key)
        out.append(cfg)
    return out


def rstd_grid() -> list[dict]:
    """Mapping space for the exploratory `rstd`-only reduction kernel."""
    out = []
    for b, r, w, s in itertools.product(
        (1024, 2048, 4096, 8192), (1, 2, 4, 8), (2, 4, 8, 16), (1, 2)
    ):
        if b * r > 65536 or not (2 <= b * r / (w * 64) <= 64):
            continue
        if b >= H and s > 2:
            continue
        out.append(dict(ROWS=r, BLOCK_N=b, num_warps=w, num_stages=s))
    return out


def norm_grid() -> list[dict]:
    """F3's proven RMSNorm mapping space (a bonus search handed to the unfused side)."""
    out = []
    for b, r, w, s in itertools.product(
        (512, 1024, 2048, 4096, 8192), (1, 2, 4, 8), (1, 2, 4, 8, 16), (1, 2)
    ):
        threads = w * 64
        if b * r > 65536:
            continue
        epr = b * r / threads
        if epr < 2 or epr > 64:
            continue
        if b >= H and s > 2:
            continue
        out.append(
            dict(ROWS=r, BLOCK_N=b, num_warps=w, num_stages=s, grid_cap=None, eps=EPS)
        )
    return out


def top_cfgs(*tables, k=3) -> list[dict]:
    rows = [(m, c) for tb in tables for (c, m, err) in tb if m is not None]
    rows.sort(key=lambda t: t[0])
    seen, out = set(), []
    for _, c in rows:
        key = tuple(sorted(c.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) == k:
            break
    return out


# ======================================================================================
# Weights (allocated once for the whole run)
# ======================================================================================
def make_w13(w_norm: torch.Tensor):
    """w13 [E, 2I, H] plus its `w`-folded twin, both with an sglang-style trailing pad.

    The pad is a hard requirement on this MACA backend: Triton's software pipeline issues
    speculative (unpredicated) B-tile loads for the peeled prologue/epilogue stages, so
    the last expert's tile can be fetched one BLOCK_K past the end of the tensor.  Both
    tensors get it; it changes no arithmetic and favours neither side.
    """
    numel = E * NW13 * H
    pad = 1 << 20
    raw_buf = torch.empty(numel + pad, device="cuda", dtype=DT)
    fold_buf = torch.empty(numel + pad, device="cuda", dtype=DT)
    raw = raw_buf[:numel].view(E, NW13, H)
    fold = fold_buf[:numel].view(E, NW13, H)
    wf = w_norm.float()
    for e in range(E):  # chunked: a 12.9 GB fp32 temporary would not fit twice
        raw[e].normal_(0, 0.02)
        fold[e] = (raw[e].float() * wf).to(DT)
    raw_buf[numel:].zero_()
    fold_buf[numel:].zero_()
    return raw_buf, raw, fold_buf, fold


# ======================================================================================
# Per-regime problem
# ======================================================================================
class Problem:
    def __init__(self, regime, w_norm, gate, w13_raw, w13_fold, b_raw, b_fold):
        torch.manual_seed(4242 + regime.T)
        self.regime = regime
        T = self.T = regime.T
        self.rows = T * TOPK
        # h1 = the residual stream entering post_attention_layernorm
        self.h1 = (torch.randn(T, H, device="cuda", dtype=torch.float32) * 0.5).to(DT)
        self.w = w_norm
        self.gate = gate
        self.b_raw, self.b_fold = b_raw, b_fold
        self.w13_raw, self.w13_fold = w13_raw, w13_fold

        # x2 -- the materialized intermediate the unfused side needs.  Seeded with the
        # fp32 reference so the unfused GEMM has valid input during tuning; the real
        # unfused chain overwrites it with the Triton norm kernel every iteration.
        self.x2 = R.rmsnorm(self.h1, w_norm, EPS).contiguous()

        _, _, self.topk_ids = R.router(self.x2, gate)
        self.layouts: dict[int, tuple] = {}

        self.logits_f = torch.zeros(T, E, device="cuda", dtype=torch.float32)
        self.logits_u = torch.zeros(T, E, device="cuda", dtype=torch.float32)
        self.logits_h = torch.zeros(T, E, device="cuda", dtype=torch.float32)
        self.c_f = torch.zeros(self.rows, NW13, device="cuda", dtype=DT)
        self.c_u = torch.zeros(self.rows, NW13, device="cuda", dtype=DT)
        self.c_h = torch.zeros(self.rows, NW13, device="cuda", dtype=DT)
        self.x2_out = torch.empty_like(self.x2)
        self.rstd = torch.ones(T, device="cuda", dtype=torch.float32)

    def layout(self, block_m: int):
        if block_m not in self.layouts:
            self.layouts[block_m] = R.moe_align_block_size(self.topk_ids, block_m, E)
        return self.layouts[block_m]

    # ---- callables ------------------------------------------------------------------
    def norm_fn(self, cfg):
        return lambda: NK.norm_only(self.h1, self.w, self.x2_out, cfg)

    def router_fused(self, cfg, sq_mode):
        return lambda: K.launch_router(
            self.h1, self.b_fold, self.logits_f, cfg, True, EPS, sq_mode
        )

    def router_unfused(self, cfg, src=None):
        a = self.x2 if src is None else src
        return lambda: K.launch_router(a, self.b_raw, self.logits_u, cfg, False, EPS)

    def moe_fused(self, cfg, sq_mode):
        sti, eids, ntp = self.layout(cfg["BLOCK_M"])
        return lambda: K.launch_moe_gateup(
            self.h1, self.w13_fold, self.c_f, sti, eids, ntp, self.rows, TOPK,
            cfg, True, EPS, sq_mode,
        )

    # ---- exploratory "half-fused" variant: rstd from a tiny reduction kernel, applied
    # as a pure epilogue scale; the GEMM k-loop is byte-for-byte the unfused one --------
    def rstd_fn(self, cfg):
        return lambda: K.launch_rstd(self.h1, self.rstd, cfg, EPS)

    def router_half(self, cfg):
        return lambda: K.launch_router(
            self.h1, self.b_fold, self.logits_h, cfg, False, EPS, 0, self.rstd
        )

    def moe_half(self, cfg):
        sti, eids, ntp = self.layout(cfg["BLOCK_M"])
        return lambda: K.launch_moe_gateup(
            self.h1, self.w13_fold, self.c_h, sti, eids, ntp, self.rows, TOPK,
            cfg, False, EPS, 0, self.rstd,
        )

    def moe_unfused(self, cfg, src=None):
        a = self.x2 if src is None else src
        sti, eids, ntp = self.layout(cfg["BLOCK_M"])
        return lambda: K.launch_moe_gateup(
            a, self.w13_raw, self.c_u, sti, eids, ntp, self.rows, TOPK, cfg, False, EPS
        )


# ======================================================================================
# Diagnostics
# ======================================================================================
def kstats(fn):
    try:
        k = fn()
        return {
            "n_regs": getattr(k, "n_regs", None),
            "n_spills": getattr(k, "n_spills", None),
            "shared_bytes": getattr(getattr(k, "metadata", None), "shared", None),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:160]}


def clear_triton_cache():
    dev = torch.cuda.current_device()
    for kern in (K.router_gemm_kernel, K.moe_gateup_prenorm_kernel):
        try:
            kern.cache[dev].clear()
        except Exception:  # noqa: BLE001
            pass


# ======================================================================================
def run_regime(prob: Problem, sq_mode: dict, norm_cfg_hint, quick: bool) -> dict:
    reg = prob.regime
    T = reg.T
    big = T >= 2048
    if quick:
        w_t, r_t = 2, 5
    elif T >= 8192:
        w_t, r_t = 3, 8
    elif T >= 2048:
        w_t, r_t = 4, 12
    elif T >= 256:
        w_t, r_t = 8, 20
    else:
        w_t, r_t = 10, 30
    w_f, r_f = (5, 20) if big else (25, 100)

    print(f"\n===== {reg.name} (T={T}, moe rows={prob.rows}) =====", flush=True)
    out: dict = {"T": T, "regime": reg.name}
    tables: dict = {}

    # ---------------------------------------------------------------- norm kernel ----
    ng = norm_grid()
    if quick:
        ng = ng[::6]
    tn = autotune(prob.norm_fn, ng, warmup=w_t, rep=r_t)
    print(f"  norm: {tn.n_tried} cfgs -> {tn.best_ms:.4f} ms {tn.best_cfg}", flush=True)
    tables["norm"] = tn.as_dict()
    norm_cfg = tn.best_cfg

    # =============================== F11b : ROUTER ===================================
    rg = router_grid(T)
    if quick:
        rg = rg[::5]
    tf_c = autotune(lambda c: [prob.router_fused(c, sq_mode["router"])], rg, w_t, r_t)
    tu_c = autotune(lambda c: [prob.router_unfused(c)], rg, w_t, r_t)
    rf = refine(tf_c.best_cfg, 256, max(16, 1 << (max(T, 1) - 1).bit_length()))
    ru = refine(tu_c.best_cfg, 256, max(16, 1 << (max(T, 1) - 1).bit_length()))
    tf_r = autotune(lambda c: [prob.router_fused(c, sq_mode["router"])], rf, w_t, r_t)
    tu_r = autotune(lambda c: [prob.router_unfused(c)], ru, w_t, r_t)
    rt_f_cfg = tf_c.best_cfg if tf_c.best_ms <= tf_r.best_ms else tf_r.best_cfg
    rt_u_cfg = tu_c.best_cfg if tu_c.best_ms <= tu_r.best_ms else tu_r.best_cfg
    print(
        f"  router fused  : coarse {tf_c.n_tried}({tf_c.n_failed}f) + refine "
        f"{tf_r.n_tried}({tf_r.n_failed}f) -> {min(tf_c.best_ms, tf_r.best_ms):.4f} ms {rt_f_cfg}",
        flush=True,
    )
    print(
        f"  router unfused: coarse {tu_c.n_tried}({tu_c.n_failed}f) + refine "
        f"{tu_r.n_tried}({tu_r.n_failed}f) -> {min(tu_c.best_ms, tu_r.best_ms):.4f} ms {rt_u_cfg}",
        flush=True,
    )
    # joint chain re-tune, in the unfused side's favour
    joint_r, best_r, best_pair_r = [], float("inf"), None
    for gc in top_cfgs(tu_c.table, tu_r.table, k=3):
        for nc in top_cfgs(tn.table, k=3):
            try:
                t = bench_chain(
                    [prob.norm_fn(nc), prob.router_unfused(gc, prob.x2_out)], w_t, r_t
                )
                joint_r.append(({"gemm": gc, "norm": nc}, t.p50_ms, None))
                if t.p50_ms < best_r:
                    best_r, best_pair_r = t.p50_ms, (gc, nc)
            except Exception as exc:  # noqa: BLE001
                joint_r.append(({"gemm": gc, "norm": nc}, None, str(exc)[:160]))
    rt_u_gemm, rt_u_norm = best_pair_r
    tables["router_fused"] = {"coarse": tf_c.as_dict(), "refine": tf_r.as_dict()}
    tables["router_unfused"] = {"coarse": tu_c.as_dict(), "refine": tu_r.as_dict()}
    tables["router_unfused_joint"] = joint_r

    # =============================== F11a : w13 =======================================
    mg = moe_grid(big)
    if quick:
        mg = mg[::7]
    mf_c = autotune(lambda c: [prob.moe_fused(c, sq_mode["moe"])], mg, w_t, r_t)
    mu_c = autotune(lambda c: [prob.moe_unfused(c)], mg, w_t, r_t)
    mrf = refine(mf_c.best_cfg, 4096, 4096, acc_lo=4)
    mru = refine(mu_c.best_cfg, 4096, 4096, acc_lo=4)
    mf_r = autotune(lambda c: [prob.moe_fused(c, sq_mode["moe"])], mrf, w_t, r_t)
    mu_r = autotune(lambda c: [prob.moe_unfused(c)], mru, w_t, r_t)
    mo_f_cfg = mf_c.best_cfg if mf_c.best_ms <= mf_r.best_ms else mf_r.best_cfg
    mo_u_cfg = mu_c.best_cfg if mu_c.best_ms <= mu_r.best_ms else mu_r.best_cfg
    print(
        f"  w13 fused  : coarse {mf_c.n_tried}({mf_c.n_failed}f) + refine "
        f"{mf_r.n_tried}({mf_r.n_failed}f) -> {min(mf_c.best_ms, mf_r.best_ms):.4f} ms {mo_f_cfg}",
        flush=True,
    )
    print(
        f"  w13 unfused: coarse {mu_c.n_tried}({mu_c.n_failed}f) + refine "
        f"{mu_r.n_tried}({mu_r.n_failed}f) -> {min(mu_c.best_ms, mu_r.best_ms):.4f} ms {mo_u_cfg}",
        flush=True,
    )
    joint_m, best_m, best_pair_m = [], float("inf"), None
    for gc in top_cfgs(mu_c.table, mu_r.table, k=3):
        for nc in top_cfgs(tn.table, k=2):
            try:
                t = bench_chain(
                    [prob.norm_fn(nc), prob.moe_unfused(gc, prob.x2_out)], w_t, r_t
                )
                joint_m.append(({"gemm": gc, "norm": nc}, t.p50_ms, None))
                if t.p50_ms < best_m:
                    best_m, best_pair_m = t.p50_ms, (gc, nc)
            except Exception as exc:  # noqa: BLE001
                joint_m.append(({"gemm": gc, "norm": nc}, None, str(exc)[:160]))
    mo_u_gemm, mo_u_norm = best_pair_m
    tables["moe_fused"] = {"coarse": mf_c.as_dict(), "refine": mf_r.as_dict()}
    tables["moe_unfused"] = {"coarse": mu_c.as_dict(), "refine": mu_r.as_dict()}
    tables["moe_unfused_joint"] = joint_m

    # ============ EXPLORATORY: "half-fused" (rstd kernel + epilogue scale) ============
    # Not part of the fused-vs-unfused headline.  It is the third point on the design
    # axis: 2/3 of the byte saving, with a k-loop identical to the unfused GEMM.
    tr_rstd = autotune(prob.rstd_fn, rstd_grid()[:: (6 if quick else 1)], w_t, r_t)
    half = {}
    for tag, mk, gcfgs in (
        ("router", prob.router_half, top_cfgs(tu_c.table, tu_r.table, tf_c.table, k=3)),
        ("moe", prob.moe_half, top_cfgs(mu_c.table, mu_r.table, mf_c.table, k=3)),
    ):
        best, bcfg, tab = float("inf"), None, []
        for gc in gcfgs:
            try:
                t = bench_chain([prob.rstd_fn(tr_rstd.best_cfg), mk(gc)], w_t, r_t)
                tab.append((gc, t.p50_ms, None))
                if t.p50_ms < best:
                    best, bcfg = t.p50_ms, gc
            except Exception as exc:  # noqa: BLE001
                tab.append((gc, None, str(exc)[:160]))
        half[tag] = {"cfg": bcfg, "tune_ms": best, "table": tab}
    tables["rstd_kernel"] = tr_rstd.as_dict()
    tables["half_fused"] = half
    print(
        f"  rstd kernel: {tr_rstd.n_tried} cfgs -> {tr_rstd.best_ms:.4f} ms "
        f"{tr_rstd.best_cfg}",
        flush=True,
    )

    # ================================ validate =======================================
    prob.logits_f.zero_(); prob.logits_u.zero_()
    prob.c_f.zero_(); prob.c_u.zero_(); prob.x2_out.zero_()
    prob.norm_fn(rt_u_norm)()
    prob.router_fused(rt_f_cfg, sq_mode["router"])()
    prob.router_unfused(rt_u_gemm, prob.x2_out)()
    prob.moe_fused(mo_f_cfg, sq_mode["moe"])()
    prob.moe_unfused(mo_u_gemm, prob.x2_out)()
    prob.rstd_fn(tr_rstd.best_cfg)()
    prob.router_half(half["router"]["cfg"])()
    prob.moe_half(half["moe"]["cfg"])()
    torch.cuda.synchronize()

    chk = {}
    chk["x2"] = check(prob.x2_out, prob.x2, label="norm_kernel_x2")
    # router reference: (a) the framework path (bf16 x2 then fp32 matmul) and
    #                   (b) the *exact* fp32 path (no bf16 rounding of x2 at all)
    ref_router = prob.x2.float() @ prob.gate.float().t()
    hf = prob.h1.float()
    rstd = torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + EPS)
    ref_router_exact = (hf * rstd * prob.w.float()) @ prob.gate.float().t()
    chk["router_fused"] = check(prob.logits_f, ref_router, label="router_fused")
    chk["router_unfused"] = check(prob.logits_u, ref_router, label="router_unfused")
    chk["router_fused_vs_exact"] = check(
        prob.logits_f, ref_router_exact, label="router_fused_vs_exact_fp32"
    )
    chk["router_unfused_vs_exact"] = check(
        prob.logits_u, ref_router_exact, label="router_unfused_vs_exact_fp32"
    )
    # does the fusion change the routing decision?
    ids_f = prob.logits_f.sigmoid().topk(TOPK, dim=-1).indices
    ids_u = prob.logits_u.sigmoid().topk(TOPK, dim=-1).indices
    chk["topk_id_agreement"] = float(
        (ids_f.sort(-1).values == ids_u.sort(-1).values).float().mean().item()
    )

    chk["router_half"] = check(prob.logits_h, ref_router, label="router_half_fused")
    idx, ref_moe = reference_rows(prob)
    chk["moe_fused"] = check(prob.c_f[idx], ref_moe, label="moe_fused")
    chk["moe_unfused"] = check(prob.c_u[idx], ref_moe, label="moe_unfused")
    chk["moe_half"] = check(prob.c_h[idx], ref_moe, label="moe_half_fused")
    for kk in (
        "x2", "router_fused", "router_unfused", "router_half",
        "moe_fused", "moe_unfused", "moe_half",
    ):
        if not chk[kk]["ok"]:
            raise RuntimeError(f"validation failed at {reg.name}: {kk} {chk[kk]}")
    print(
        f"  rel_err  router f={chk['router_fused']['rel_err']:.2e} "
        f"u={chk['router_unfused']['rel_err']:.2e} | "
        f"w13 f={chk['moe_fused']['rel_err']:.2e} u={chk['moe_unfused']['rel_err']:.2e} | "
        f"topk agree {chk['topk_id_agreement']*100:.2f}%",
        flush=True,
    )

    # ================================ final timing ===================================
    t_norm = bench_chain([prob.norm_fn(norm_cfg)], w_f, r_f)
    t_rt_f = bench_chain([prob.router_fused(rt_f_cfg, sq_mode["router"])], w_f, r_f)
    t_rt_u = bench_chain(
        [prob.norm_fn(rt_u_norm), prob.router_unfused(rt_u_gemm, prob.x2_out)], w_f, r_f
    )
    t_rt_gemm = bench_chain([prob.router_unfused(rt_u_gemm, prob.x2_out)], w_f, r_f)

    t_mo_f = bench_chain([prob.moe_fused(mo_f_cfg, sq_mode["moe"])], w_f, r_f)
    t_mo_u = bench_chain(
        [prob.norm_fn(mo_u_norm), prob.moe_unfused(mo_u_gemm, prob.x2_out)], w_f, r_f
    )
    t_mo_gemm = bench_chain([prob.moe_unfused(mo_u_gemm, prob.x2_out)], w_f, r_f)

    t_rstd = bench_chain([prob.rstd_fn(tr_rstd.best_cfg)], w_f, r_f)
    t_rt_h = bench_chain(
        [prob.rstd_fn(tr_rstd.best_cfg), prob.router_half(half["router"]["cfg"])],
        w_f, r_f,
    )
    t_mo_h = bench_chain(
        [prob.rstd_fn(tr_rstd.best_cfg), prob.moe_half(half["moe"]["cfg"])], w_f, r_f
    )
    t_comb_h = bench_chain(
        [
            prob.rstd_fn(tr_rstd.best_cfg),
            prob.router_half(half["router"]["cfg"]),
            prob.moe_half(half["moe"]["cfg"]),
        ],
        w_f, r_f,
    )

    # combined end-to-end: ONE norm kernel serves both consumers
    t_comb_f = bench_chain(
        [
            prob.router_fused(rt_f_cfg, sq_mode["router"]),
            prob.moe_fused(mo_f_cfg, sq_mode["moe"]),
        ],
        w_f,
        r_f,
    )
    t_comb_u = bench_chain(
        [
            prob.norm_fn(norm_cfg),
            prob.router_unfused(rt_u_gemm, prob.x2_out),
            prob.moe_unfused(mo_u_gemm, prob.x2_out),
        ],
        w_f,
        r_f,
    )

    # ---- ISOLATION: same config, same buffers, FUSE_NORM on vs off.  There is NO extra
    # input tensor in this fusion, so this is a pure instruction-cost measurement (the
    # analogue of F1's stride-0-broadcast trick, only exact).
    iso = {}
    for tag, cfg, mk in (
        ("router", rt_f_cfg, prob.router_fused),
        ("moe", mo_f_cfg, prob.moe_fused),
    ):
        on = bench_chain([mk(cfg, sq_mode[tag])], w_f, r_f)
        off_fn = (
            (lambda: K.launch_router(prob.h1, prob.b_fold, prob.logits_f, cfg, False, EPS))
            if tag == "router"
            else prob.moe_unfused_same(cfg)
        )
        off = bench_chain([off_fn], w_f, r_f)
        iso[tag] = {
            "fuse_on_ms": on.p50_ms,
            "fuse_off_same_cfg_ms": off.p50_ms,
            "instruction_cost_pct": 100.0 * (on.p50_ms / off.p50_ms - 1.0),
        }

    # ---- register / SMEM report, cache cleared between compiles --------------------
    clear_triton_cache()
    regs = {"router_fused": kstats(prob.router_fused(rt_f_cfg, sq_mode["router"]))}
    clear_triton_cache()
    regs["router_unfused"] = kstats(prob.router_unfused(rt_u_gemm, prob.x2_out))
    clear_triton_cache()
    regs["router_unfused_at_fused_cfg"] = kstats(
        lambda: K.launch_router(prob.x2, prob.b_raw, prob.logits_u, rt_f_cfg, False, EPS)
    )
    clear_triton_cache()
    regs["moe_fused"] = kstats(prob.moe_fused(mo_f_cfg, sq_mode["moe"]))
    clear_triton_cache()
    regs["moe_unfused"] = kstats(prob.moe_unfused(mo_u_gemm, prob.x2_out))
    clear_triton_cache()
    regs["moe_unfused_at_fused_cfg"] = kstats(prob.moe_unfused_same(mo_f_cfg))

    # ---- vendor BLAS reference lines ----------------------------------------------
    t_blas_router = bench_chain(
        [lambda: torch.matmul(prob.x2, prob.b_raw)], w_f, r_f
    )
    t_blas_router_fp32 = bench_chain(
        [lambda: torch.matmul(prob.x2.float(), prob.b_raw.float())],
        max(2, w_f // 3),
        max(5, r_f // 3),
    )
    t_blas_moe = bench_chain(
        vendor_moe_chain(prob), max(2, w_f // 3), max(5, r_f // 3)
    )
    t_blas_moe_dense = bench_chain(
        [lambda: torch.matmul(prob.x2, prob.w13_raw[0].t())],
        max(2, w_f // 3),
        max(5, r_f // 3),
    )

    # ---- redundancy / traffic bookkeeping ------------------------------------------
    rt_ntiles = triton.cdiv(E, rt_f_cfg["BLOCK_N"])
    mo_ntiles = triton.cdiv(NW13, mo_f_cfg["BLOCK_N"])
    act = T * H * 2
    f_router = 2.0 * T * H * E
    f_moe = 2.0 * prob.rows * H * NW13
    tmodel = {t.fusion: t.row() for t in TR.model(reg)}

    row = {
        "regime": reg.name,
        "T": T,
        "moe_rows": prob.rows,
        # ---- F11b router -------------------------------------------------------
        "f11b_router": speedup_row(
            reg.name, t_rt_f, t_rt_u,
            {
                "fused_cfg": rt_f_cfg,
                "unfused_gemm_cfg": rt_u_gemm,
                "unfused_norm_cfg": rt_u_norm,
                "unfused_gemm_only_ms": t_rt_gemm.p50_ms,
                "norm_only_ms": t_norm.p50_ms,
                "ceiling": tmodel["F11b_prenorm_router"]["roofline_ceiling"],
                "rel_err": chk["router_fused"]["rel_err"],
                "rel_err_unfused": chk["router_unfused"]["rel_err"],
                "n_tiles": rt_ntiles,
                "sq_redundancy": rt_ntiles,
                "extra_sq_flops_frac": rt_ntiles / E,
                "fused_tflops": f_router / (t_rt_f.p50_ms * 1e-3) / 1e12,
                "vendor_blas_bf16_ms": t_blas_router.p50_ms,
                "vendor_blas_fp32_ms": t_blas_router_fp32.p50_ms,
                "fused_noflush_ms": t_rt_f.noflush_p50_ms,
                "unfused_noflush_ms": t_rt_u.noflush_p50_ms,
                "bytes_fused": act + H * E * 2 + T * E * 4,
                "bytes_unfused": 2 * act + act + H * E * 2 + T * E * 4,
            },
        ),
        # ---- F11a w13 ----------------------------------------------------------
        "f11a_w13": speedup_row(
            reg.name, t_mo_f, t_mo_u,
            {
                "fused_cfg": mo_f_cfg,
                "unfused_gemm_cfg": mo_u_gemm,
                "unfused_norm_cfg": mo_u_norm,
                "unfused_gemm_only_ms": t_mo_gemm.p50_ms,
                "norm_only_ms": t_norm.p50_ms,
                "ceiling": tmodel["F11a_prenorm_w13"]["roofline_ceiling"],
                "rel_err": chk["moe_fused"]["rel_err"],
                "rel_err_unfused": chk["moe_unfused"]["rel_err"],
                "n_tiles": mo_ntiles,
                "sq_redundancy": mo_ntiles * TOPK,
                "sq_redundancy_ntile_only": mo_ntiles,
                "extra_sq_flops_frac": 1.0 / mo_f_cfg["BLOCK_N"],
                "fused_tflops": f_moe / (t_mo_f.p50_ms * 1e-3) / 1e12,
                "unfused_tflops": f_moe / (t_mo_gemm.p50_ms * 1e-3) / 1e12,
                "vendor_blas_grouped_ms": t_blas_moe.p50_ms,
                "vendor_blas_dense_1expert_ms": t_blas_moe_dense.p50_ms,
                "vendor_blas_dense_tflops": (2.0 * T * H * NW13)
                / (t_blas_moe_dense.p50_ms * 1e-3)
                / 1e12,
                "fused_noflush_ms": t_mo_f.noflush_p50_ms,
                "unfused_noflush_ms": t_mo_u.noflush_p50_ms,
            },
        ),
        # ---- combined (the honest end-to-end number) --------------------------
        "combined": speedup_row(
            reg.name, t_comb_f, t_comb_u,
            {
                "note": "unfused = 1 norm + router GEMM + w13 GEMM; "
                "fused = router GEMM + w13 GEMM (x2 never materialized)",
                "norm_cfg": norm_cfg,
                "fused_noflush_ms": t_comb_f.noflush_p50_ms,
                "unfused_noflush_ms": t_comb_u.noflush_p50_ms,
            },
        ),
        # ---- exploratory half-fused (rstd kernel + epilogue scale) -------------
        "half_fused": {
            "note": "EXPLORATORY, not the headline: rstd from a tiny reduction kernel, "
            "applied as a pure epilogue scale. 2 activation passes vs the unfused "
            "side's 3 and the fused side's 1; GEMM k-loop identical to unfused.",
            "rstd_cfg": tr_rstd.best_cfg,
            "rstd_only_ms": t_rstd.p50_ms,
            "router_cfg": half["router"]["cfg"],
            "router_ms": t_rt_h.p50_ms,
            "router_speedup_vs_unfused": t_rt_u.p50_ms / t_rt_h.p50_ms,
            "moe_cfg": half["moe"]["cfg"],
            "moe_ms": t_mo_h.p50_ms,
            "moe_speedup_vs_unfused": t_mo_u.p50_ms / t_mo_h.p50_ms,
            "combined_ms": t_comb_h.p50_ms,
            "combined_speedup_vs_unfused": t_comb_u.p50_ms / t_comb_h.p50_ms,
            "rel_err_router": chk["router_half"]["rel_err"],
            "rel_err_moe": chk["moe_half"]["rel_err"],
        },
        "isolation_fuse_on_vs_off_same_cfg": iso,
        "kernel_stats": regs,
        "checks": chk,
        "grid_sizes": {
            "router_coarse_fused": tf_c.n_tried,
            "router_coarse_unfused": tu_c.n_tried,
            "router_refine_fused": tf_r.n_tried,
            "router_refine_unfused": tu_r.n_tried,
            "moe_coarse_fused": mf_c.n_tried,
            "moe_coarse_unfused": mu_c.n_tried,
            "moe_refine_fused": mf_r.n_tried,
            "moe_refine_unfused": mu_r.n_tried,
            "norm": tn.n_tried,
        },
    }
    print(
        f"  F11b router : fused {t_rt_f.p50_ms:.4f} | unfused {t_rt_u.p50_ms:.4f} "
        f"-> {row['f11b_router']['speedup']:.3f}x  (ceiling "
        f"{row['f11b_router']['ceiling']:.2f}x)",
        flush=True,
    )
    print(
        f"  F11a w13    : fused {t_mo_f.p50_ms:.4f} | unfused {t_mo_u.p50_ms:.4f} "
        f"-> {row['f11a_w13']['speedup']:.3f}x  (ceiling "
        f"{row['f11a_w13']['ceiling']:.2f}x)",
        flush=True,
    )
    print(
        f"  combined    : fused {t_comb_f.p50_ms:.4f} | unfused {t_comb_u.p50_ms:.4f} "
        f"-> {row['combined']['speedup']:.3f}x",
        flush=True,
    )
    print(
        f"  half-fused  : router {t_rt_h.p50_ms:.4f} "
        f"({t_rt_u.p50_ms/t_rt_h.p50_ms:.3f}x) | w13 {t_mo_h.p50_ms:.4f} "
        f"({t_mo_u.p50_ms/t_mo_h.p50_ms:.3f}x) | combined {t_comb_h.p50_ms:.4f} "
        f"({t_comb_u.p50_ms/t_comb_h.p50_ms:.3f}x)   [exploratory]",
        flush=True,
    )
    print(
        f"  isolation   : router +{iso['router']['instruction_cost_pct']:.2f}% | "
        f"w13 +{iso['moe']['instruction_cost_pct']:.2f}%   (same cfg, FUSE_NORM on/off)",
        flush=True,
    )
    return row, tables, norm_cfg


# --- helper bound onto Problem (kept out of the class body for readability) ----------
def _moe_unfused_same(self, cfg):
    """The unfused GEMM at an ARBITRARY config (used for the isolation measurement)."""
    sti, eids, ntp = self.layout(cfg["BLOCK_M"])
    return lambda: K.launch_moe_gateup(
        self.x2, self.w13_raw, self.c_u, sti, eids, ntp, self.rows, TOPK, cfg, False, EPS
    )


Problem.moe_unfused_same = _moe_unfused_same


def reference_rows(prob: Problem, n_sample: int = 1024):
    """fp32 reference for the w13 GEMM on a sampled row subset (a full fp32 reference at
    T=8192 would be 3.3 TFLOP)."""
    rows = prob.rows
    if rows <= n_sample:
        idx = torch.arange(rows, device="cuda")
    else:
        g = torch.Generator(device="cuda").manual_seed(1234)
        idx = torch.randperm(rows, device="cuda", generator=g)[:n_sample].sort().values
    tok = (idx // TOPK).long()
    kk = (idx % TOPK).long()
    experts = prob.topk_ids.long()[tok, kk]
    ref = torch.empty(idx.numel(), NW13, device="cuda", dtype=torch.float32)
    xs = prob.x2.float()[tok]
    for e in torch.unique(experts).tolist():
        sel = (experts == e).nonzero(as_tuple=True)[0]
        ref[sel] = xs[sel] @ prob.w13_raw[e].float().t()
    return idx, ref


def vendor_moe_chain(prob: Problem):
    """Vendor-BLAS grouped GEMM: rows pre-gathered per expert OUTSIDE the timed region,
    so this is the best case for the vendor path (pure per-expert torch.matmul)."""
    flat = prob.topk_ids.flatten().long()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E)
    starts = torch.cat(
        [torch.zeros(1, dtype=torch.long, device="cuda"), counts.cumsum(0)[:-1]]
    )
    a_sorted = prob.x2[(order // TOPK)].contiguous()
    out = torch.empty(prob.rows, NW13, device="cuda", dtype=DT)
    cs, ss = counts.tolist(), starts.tolist()
    segs = [(e, ss[e], ss[e] + cs[e]) for e in range(E) if cs[e]]
    wt = [prob.w13_raw[e].t() for e, _, _ in segs]

    def run():
        for (e, s, t), w in zip(segs, wt):
            out[s:t] = torch.matmul(a_sorted[s:t], w)

    return [run]


# ======================================================================================
# SQ_MODE pre-study: pick the sum-of-squares implementation ONCE per family, then hold it
# fixed so the fused and unfused tuning grids are the same size (fairness rule 2).
#
#   0  sq += tl.sum(af*af, axis=1)          per k-step, the blog's pseudocode
#   1  sqt += af*af  -> one reduce at the end (a [BM, BK] fp32 tile of extra state)
#   2  sqd += tl.dot(a*a, ones[BK,16])      sum of squares on the TENSOR CORE
#   3  a re-loaded separately, then mode 0  (isolates the dot-operand layout hypothesis)
# ======================================================================================
SQ_NAMES = {0: "per-step tl.sum", 1: "tile-accum", 2: "tensor-core dot", 3: "2nd load"}


def sq_study(prob: Problem) -> tuple[dict, list]:
    cfgs_r = [
        dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2, GROUP_M=8),
        dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2, GROUP_M=8),
        dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=2, GROUP_M=8),
        dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=64, num_warps=8, num_stages=2, GROUP_M=8),
    ]
    cfgs_m = [
        dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=2, GROUP_M=8),
        dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=2, GROUP_M=8),
        dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=64, num_warps=8, num_stages=2, GROUP_M=8),
        dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=32, num_warps=8, num_stages=2, GROUP_M=8),
    ]
    modes = (0, 1, 2, 3)
    tab, pick = [], {}
    for tag, cfgs, mk in (
        ("router", cfgs_r, prob.router_fused),
        ("moe", cfgs_m, prob.moe_fused),
    ):
        times: dict[int, dict[int, float]] = {m: {} for m in modes}
        for ci, cfg in enumerate(cfgs):
            for m in modes:
                try:
                    t = bench_chain([mk(cfg, m)], 4, 12).p50_ms
                    tab.append((tag, cfg, m, t, None))
                    times[m][ci] = t
                except Exception as exc:  # noqa: BLE001
                    tab.append((tag, cfg, m, None, str(exc)[:120]))
        # compare only over configs where EVERY mode compiled, so a mode is never
        # rewarded for having failed on the slow shapes
        common = set.intersection(*[set(times[m]) for m in modes]) or set(times[0])
        tot = {m: sum(times[m].get(c, float("inf")) for c in common) for m in modes}
        pick[tag] = min(tot, key=tot.get)
        print(
            f"  SQ study [{tag}] over {len(common)} common cfgs: "
            + ", ".join(f"m{m}({SQ_NAMES[m]}) {tot[m]:.3f}ms" for m in modes)
            + f"  -> SQ_MODE={pick[tag]}",
            flush=True,
        )
    return pick, tab


# ======================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--regimes", default="")
    args = ap.parse_args()

    env = C.BenchEnv.probe()
    print(f"device={env.device_name} warp={env.warp_size} CUs={env.num_sm}", flush=True)

    torch.manual_seed(7)
    w_norm = (torch.randn(H, device="cuda", dtype=torch.float32) * 0.1 + 1.0).to(DT)
    gate = (torch.randn(E, H, device="cuda", dtype=torch.float32) * 0.02).to(DT)
    b_raw = gate.t().contiguous()                                  # [H, E]
    b_fold = K.fold_weight_nk(gate, w_norm).t().contiguous()       # [H, E], w folded
    print("building w13 (raw + folded, 2 x 12.9 GB)...", flush=True)
    t0 = time.time()
    _rb, w13_raw, _fb, w13_fold = make_w13(w_norm)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    # ---- validate the folding identity itself, once, outside all timing --------------
    with torch.no_grad():
        hh = (torch.randn(64, H, device="cuda", dtype=torch.float32) * 0.5).to(DT)
        x2h = R.rmsnorm(hh, w_norm, EPS)
        lhs = x2h.float() @ b_raw.float()
        hf = hh.float()
        rstd = torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + EPS)
        rhs = (hf @ b_fold.float()) * rstd
        fold_err = rel_err(rhs, lhs)
        lhs13 = x2h.float() @ w13_raw[3].float().t()
        rhs13 = (hf @ w13_fold[3].float().t()) * rstd
        fold_err13 = rel_err(rhs13, lhs13)
    print(
        f"folding identity: router rel_err {fold_err:.3e}, w13 rel_err {fold_err13:.3e}",
        flush=True,
    )

    regimes = REGIMES
    if args.regimes:
        want = set(args.regimes.split(","))
        regimes = [r for r in regimes if r.name in want]

    rows, tables = [], {}
    norm_hint = None

    # ---- SQ_MODE pre-study, run ONCE at prefill_t2048 (a regime where the GEMMs are
    # genuinely compute-bound, so the sum-of-squares implementation is actually visible)
    # and then held FIXED for every regime, so that the fused and unfused tuning grids
    # stay the same size.  Recorded in full in the result JSON.
    print("SQ_MODE pre-study (at prefill_t2048):", flush=True)
    study_reg = [r for r in C.PREFILL_REGIMES if r.T == 2048][0]
    _sp = Problem(study_reg, w_norm, gate, w13_raw, w13_fold, b_raw, b_fold)
    sq_pick, sq_tab = sq_study(_sp)
    del _sp
    torch.cuda.empty_cache()

    def snapshot(done: bool) -> None:
        record(
            RESULT_ID,
            {
                "id": RESULT_ID,
                "complete": done,
                "fusion": "#11 Lazy Pre-Norm -- RMSNorm fused into a GEMM as a prologue "
                "(Zhou et al., PyTorch blog 2026-07-10, section 2)",
                "shape": {
                    "hidden": H,
                    "moe_intermediate": I,
                    "w13_N": NW13,
                    "experts": E,
                    "top_k": TOPK,
                    "router_N": E,
                    "dtype": "bfloat16",
                    "eps": EPS,
                },
                "identity": {
                    "affine_free": "(A*rstd) @ B == (A @ B) * rstd",
                    "affine_handling": "((A*rstd)*w) @ B == (A @ (w[:,None]*B)) * rstd; "
                    "w folded into the GEMM weight OFFLINE (load-time transform)",
                    "fold_rel_err_router": fold_err,
                    "fold_rel_err_w13": fold_err13,
                },
                "x2_materialization": {
                    "choice": "(ii) fuse ALL K==6144 consumers; x2 is never materialized",
                    "why": "x2 feeds the router, the routed-expert w13 GEMM and the shared "
                    "expert's w13 GEMM. Both benchmarked consumers are fused here; the "
                    "shared expert is the identical transform on a 1-expert weight. The "
                    "`combined` row charges ONE norm kernel to the unfused side, which is "
                    "what the real layer pays -- the per-family rows double-count it, "
                    "matching glm52/traffic.py's per-family model.",
                },
                "sq_mode_study": {"pick": sq_pick, "table": sq_tab},
                "fairness": {
                    "one_kernel_source": "glm52/kernels/lazy_prenorm.py :: "
                    "router_gemm_kernel and moe_gateup_prenorm_kernel",
                    "flag": "FUSE_NORM constexpr; the unfused side is the SAME kernel with "
                    "it off, plus glm52.kernels.add_rmsnorm.norm_only for the split-out norm",
                    "grids": "the coarse grid generator takes no `fused` argument, so both "
                    "sides search the IDENTICAL config list (unlike F6, the lazy-prenorm "
                    "kernel stages no extra SMEM tile, so no filter can differ). Refine "
                    "grids are the same neighbourhood rule centred on each side's own "
                    "winner, so their sizes can differ by a few configs; all counts are "
                    "recorded per regime in grid_sizes.",
                    "unfused_bonus": "the unfused side additionally gets (a) an independent "
                    "search over the RMSNorm kernel's own 152-config space and (b) a joint "
                    "chain re-tune over top-3 GEMM x top-3 norm configs",
                },
                "env": env.__dict__,
                "rows": rows,
                "tune_tables": tables,
            },
        )

    for reg in regimes:
        prob = Problem(reg, w_norm, gate, w13_raw, w13_fold, b_raw, b_fold)
        row, tab, norm_hint = run_regime(prob, sq_pick, norm_hint, args.quick)
        rows.append(row)
        tables[reg.name] = tab
        del prob
        torch.cuda.empty_cache()
        snapshot(False)

    snapshot(True)
    print(f"\nwrote results/{RESULT_ID}.json\n", flush=True)
    hdr = f"{'regime':<16}{'F11b rt':>9}{'ceil':>7}{'F11a w13':>10}{'ceil':>7}{'combined':>10}"
    print(hdr)
    for r in rows:
        print(
            f"{r['regime']:<16}{r['f11b_router']['speedup']:>9.3f}"
            f"{r['f11b_router']['ceiling']:>7.2f}"
            f"{r['f11a_w13']['speedup']:>10.3f}"
            f"{r['f11a_w13']['ceiling']:>7.2f}"
            f"{r['combined']['speedup']:>10.3f}"
        )


if __name__ == "__main__":
    main_guard(main)
