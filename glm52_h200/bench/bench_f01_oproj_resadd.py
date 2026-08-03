"""Fusion #1 benchmark: o_proj GEMM + residual add (dense GEMM epilogue fusion).

Run:
    python3 glm52_h200/bench/bench_f01_oproj_resadd.py --gpu auto [--regimes ...] [--quick]

`--gpu auto` picks the idlest of the host's GPUs and masks the process to it before
CUDA initialises; on the 8-GPU measurement host that is the difference between timing an
idle card and timing one another tenant is already using.

Reports FOUR numbers per regime:
  triton-fused    : oproj_gemm_kernel(FUSE_RESADD=True)
  triton-unfused  : oproj_gemm_kernel(FUSE_RESADD=False) + epilogue_kernel(HAS_RES=True)
  vendor-fused    : torch.addmm(residual, a, b)          (cuBLASLt, beta=1)
  vendor-unfused  : torch.mm(a, b) ; torch.add(c, residual)

Tuning protocol (identical, independent budgets for the two Triton sides):
  0. epilogue_kernel tuned on its own grid, separately for each (dtype-in, HAS_RES) pair.
  1. coarse GEMM grid (same grid object for both sides) -> per-side winner
  2. refine grid built around each side's own coarse winner
  3. joint refine: that side's top-3 GEMM configs x every epilogue config, timed as chain.
     A SPLIT_K==1 fused winner has no epilogue kernel at all, so instead of skipping the
     stage it spends the SAME number of trials on fresh GEMM neighbours (see stage 3).
Nothing is ever shared between the fused and unfused searches.

WHY THIS ONE IS THE PORT'S CANARY.  On the 4060 this bench timed the whole fused arm and
then the whole unfused arm; the GPU drifted 22 % thermally inside one run and produced a
speedup ABOVE the cell's own physical ceiling.  The headline number was wrong and the
corrected one (interleaved, paired) was better evidence.  Every final timing here now goes
through `bench_pair`.
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from glm52_h200 import config as C
from glm52_h200 import bench as B
from glm52_h200.common import (
    autotune,
    bench_chain,
    check,
    main_guard,
    record,
    speedup_row,
)
from glm52_h200.kernels import oproj_resadd as K
from glm52_h200.kernels.oproj_resadd import (
    epilogue_launch,
    make_fused_chain,
    make_unfused_chain,
)

RESULT_ID = "f01_oproj_resadd"
DEV = "cuda"
N_OUT = C.HIDDEN_SIZE  # 6144
UNITS = ["triton"]  # this family has a single fused/unfused pair; --only kept for symmetry

_ENV = C.env()
SMEM_LIMIT = B.env_int(_ENV, "smem_bytes")  # per-block opt-in ceiling
WARP = B.env_int(_ENV, "warp_size")
MAX_THREADS = B.max_threads_per_block(_ENV)  # largest CTA a launch may request
# Tile ladders derived from the SMEM ceiling, so the grid reaches every shape THIS device
# can run: BM/BN up to 256 on an H200's 232448 B, up to 128 on sm_89's 101376 B, and
# whatever C500's 65536 B allowed.  Written-down ladders are how a port silently
# under-searches a bigger device -- in both arms, which hides it from the ratio.
TILES = B.tile_ladder(_ENV)
BKS = B.bk_ladder(_ENV, hi=128)
ACC_CAP = B.MAX_ACC_ELEMS_PER_THREAD  # fp32 accumulator elements per lane


# --------------------------------------------------------------------------------------
# Config-space construction
# --------------------------------------------------------------------------------------
def _valid_gemm(cfg: dict, M: int) -> bool:
    bm, bn, bk = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"]
    w, st = cfg["num_warps"], cfg["num_stages"]
    # Triton's mainloop multi-buffering is version- AND stack-dependent: 3.0 staged
    # `num_stages` buffers, 3.6/sm_89 stages `num_stages - 1` with a floor of 2, and the
    # H200 preflight's own `smem_probe` block puts 3.6/sm_90 back at `num_stages` (all five
    # observations, including the one that failed, reproduce to the byte).  `B.smem_predict`
    # FITS that offset to those measurements instead of picking one, so this filter is exact
    # rather than merely conservative -- it neither prunes a tile the device can run nor
    # spends two arms' compile time on one it cannot.
    if B.smem_predict(bm, bn, bk, st) > SMEM_LIMIT:
        return False
    threads = w * WARP
    if threads > MAX_THREADS:  # a CTA this wide cannot be launched on this device
        return False
    per_thread = (bm * bn) / threads  # fp32 accumulator elements per lane
    # ACC_CAP is 128, derived from the 256-entry per-thread register window.  It used to be
    # 64, and that rejected `BM128 BN256 num_warps=8` -- exactly the mapping the preflight
    # measured at 788 TF/s, 96 % of cuBLAS.  A grid that cannot express the device's own
    # peak cannot say how far a fused kernel is from it.
    if per_thread < B.MIN_ELEMS_PER_THREAD or per_thread > ACC_CAP:
        return False
    # a/b tile fragments must be distributable over the lanes
    if (bm * bk) % threads or (bk * bn) % threads:
        return False
    if bm > 16 and bm > 2 * M:  # do not pay for tiles that are >50% padding
        return False
    return True


COARSE_CAP = 120
#: Trial budget for the coarse grid AFTER the sm_90 overlays multiply it.  Widening offers
#: TMA / warp specialization / clusters / TMA+WS on top of every classic config, so the
#: legal space is up to 5x what it was; spending the old 120 trials on the classic subset
#: alone would leave the new axes tried only at whatever tiles a pre-widening cap happened
#: to keep.  Both arms get the same widened, same-sampled list.
COARSE_CAP_WIDENED = 220


def coarse_grid(M: int) -> list[dict]:
    """Valid configs spanning the space, capped at COARSE_CAP by a seeded random sample.

    Sampling (rather than striding a product order) keeps the coarse stage unbiased in
    every axis; the refine stage then walks the local neighbourhood of the winner.  The
    same grid object is handed to the fused and the unfused search.
    """
    if M <= 1:
        bms = [TILES[0]]
        bns = [t for t in TILES if t >= 64]
        bks = [k for k in BKS if k >= 64]
        sks, ws, sts, gms = [1, 2, 4, 8, 16], [2, 4, 8], [2, 3], [8]
    elif M <= 32:
        bms = [t for t in TILES if t <= 32]
        bns = [t for t in TILES if t >= 64]
        bks = [k for k in BKS if k >= 64]
        sks, ws, sts, gms = [1, 2, 4, 8, 16], [2, 4, 8], [2, 3], [8]
    elif M <= 256:
        bms = [t for t in TILES if t <= 128]
        bns = [t for t in TILES if t >= 32]
        bks = list(BKS)
        sks, ws, sts, gms = [1, 2, 4, 8], [4, 8], [2, 3], [8]
    else:
        bms = [t for t in TILES if t >= 64]
        bns = [t for t in TILES if t >= 64]
        bks = list(BKS)
        sks, ws, sts, gms = [1, 2], [4, 8, 16], [2, 3, 4], [1, 8]

    pool = []
    for bm, bn, bk, sk, w, st, gm in itertools.product(bms, bns, bks, sks, ws, sts, gms):
        cfg = dict(
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            GROUP_M=gm,
            SPLIT_K=sk,
            num_warps=w,
            num_stages=st,
        )
        if _valid_gemm(cfg, M):
            pool.append(cfg)

    # always include the calibrated generic-GEMM seed and its split-K/tile relatives
    seeds = []
    for bm in {min(128, max(16, 1 << (M - 1).bit_length())), 128}:
        for sk in (1, 2, 4, 8):
            for bn in (64, 128):
                cfg = dict(
                    BLOCK_M=bm,
                    BLOCK_N=bn,
                    BLOCK_K=32,
                    GROUP_M=8,
                    SPLIT_K=sk,
                    num_warps=8,
                    num_stages=2,
                )
                if _valid_gemm(cfg, M):
                    seeds.append(cfg)

    keys = {tuple(sorted(c.items())) for c in seeds}
    rest = [c for c in pool if tuple(sorted(c.items())) not in keys]
    rng = random.Random(20260727)
    if len(rest) > COARSE_CAP - len(seeds):
        rest = rng.sample(rest, COARSE_CAP - len(seeds))
    # H200-only mapping axes (clusters / warp specialization / TMA / TMA+WS) are added here,
    # to BOTH arms' shared coarse grid, and only when a LIVE capability probe proved them
    # AND the kernel module advertises the cfg key.  On a stack without them this is the
    # identity.  The cap is applied AFTER widening so the trial budget is spent on the
    # widened space, not on the classic subset with a few overlays bolted on.
    return B.widen(seeds + rest, K, cap=COARSE_CAP_WIDENED, tag=f"f01/M{M}")


_AXES = {
    # Tile axes come from the device ladder, so a bigger SMEM ceiling really is reachable by
    # refinement and not just by the coarse grid.
    "BLOCK_M": list(TILES),
    "BLOCK_N": list(TILES),
    "BLOCK_K": sorted({16, *B.bk_ladder(_ENV)}),
    "SPLIT_K": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
    "num_warps": B.warp_ladder(_ENV),
    "num_stages": [1, 2, 3, 4, 5, 6],
    "GROUP_M": [1, 2, 4, 8, 16],
}


def refine_grid(best: dict, M: int, seen: set) -> list[dict]:
    """+-1 step on every axis around `best`, plus a few 2-step moves on the tile axes.

    The warp ladder tops out at whatever this device can hold in one CTA (32 warps at 32
    lanes, 16 at 64), and the stage ladder reaches 6 because Hopper's deeper pipeline can
    use stages an Ada or a C500 could not fit; both are widened for BOTH arms.
    """
    out = []

    def add(cfg):
        key = tuple(sorted((k, str(v)) for k, v in cfg.items()))
        if key in seen or not _valid_gemm(cfg, M):
            return
        seen.add(key)
        out.append(cfg)

    for axis, vals in _AXES.items():
        if best.get(axis) not in vals:
            continue
        i = vals.index(best[axis])
        steps = (-2, -1, 1, 2) if axis.startswith("BLOCK") or axis == "SPLIT_K" else (-1, 1)
        for d in steps:
            j = i + d
            if 0 <= j < len(vals):
                cfg = dict(best)
                cfg[axis] = vals[j]
                add(cfg)
    # a few 2-axis moves around the tile shape
    for da, db in itertools.product((-1, 0, 1), repeat=2):
        if da == 0 and db == 0:
            continue
        cfg = dict(best)
        for axis, d in (("BLOCK_M", da), ("BLOCK_N", db)):
            vals = _AXES[axis]
            if cfg.get(axis) not in vals:
                break
            i = vals.index(cfg[axis])
            j = min(max(i + d, 0), len(vals) - 1)
            cfg[axis] = vals[j]
        else:
            add(cfg)
    return out


def epi_grid() -> list[dict]:
    """Elementwise-epilogue grid.

    This one guard has to be exactly right: the epilogue kernel is the ENTIRE unfused-side
    overhead (a SPLIT_K==1 fused chain has no epilogue at all), so a wrong lane count
    under-tunes only the unfused arm and inflates the fusion win.  `WARP` is the probed
    lane count; the 4060 audit found this exact site computing threads as `num_warps * 64`
    on a 32-lane device.
    """
    out = []
    for blk, w in itertools.product(
        [256, 512, 1024, 2048, 4096, 8192, 16384], B.warp_ladder(_ENV)
    ):
        per_thread = blk / (w * WARP)
        if per_thread < B.MIN_ELEMS_PER_THREAD or per_thread > B.MAX_ELEMS_PER_THREAD:
            continue
        out.append(dict(BLOCK=blk, num_warps=w, num_stages=1))
    return out


# --------------------------------------------------------------------------------------
# Per-regime driver
# --------------------------------------------------------------------------------------
def run_regime(regime, quick: bool, fair: B.Fairness) -> tuple[dict, dict]:
    M, K = regime.T, regime.oproj_k
    print(f"\n===== {regime.name}: M={M} K={K} N={N_OUT} =====", flush=True)

    torch.manual_seed(1234)
    a = (torch.randn(M, K, device=DEV, dtype=torch.float32) * 0.05).bfloat16()
    # o_proj weight.  Production storage is nn.Linear [N, K]; the Triton kernels and the
    # vendor calls all consume the SAME [K, N] contiguous tensor so the four numbers are
    # exactly comparable.
    w_nk = (torch.randn(N_OUT, K, device=DEV, dtype=torch.float32) * 0.02).bfloat16()
    b = w_nk.t().contiguous()
    del w_nk
    r = (torch.randn(M, N_OUT, device=DEV, dtype=torch.float32) * 0.5).bfloat16()

    torch.cuda.empty_cache()
    ref = torch.addmm(r.float(), a.float(), b.float())

    out = torch.empty(M, N_OUT, device=DEV, dtype=torch.bfloat16)
    cmat = torch.empty(M, N_OUT, device=DEV, dtype=torch.bfloat16)
    acc32 = torch.zeros(M, N_OUT, device=DEV, dtype=torch.float32)

    tw, tr, mw, mr = B.reps(M, quick)

    # ---------------- stage 0: epilogue kernel, tuned on its own grid ----------------
    eg = B.quick_slice(epi_grid(), 8) if quick else epi_grid()
    t_epi_add_bf16 = autotune(
        lambda c: [lambda: epilogue_launch(cmat, r, out, c, True)], eg, tw, tr
    )
    t_epi_add_f32 = autotune(
        lambda c: [lambda: epilogue_launch(acc32, r, out, c, True)], eg, tw, tr
    )
    t_epi_cast_f32 = autotune(
        lambda c: [lambda: epilogue_launch(acc32, None, out, c, False)], eg, tw, tr
    )
    print(
        f"  epi add-bf16 {t_epi_add_bf16.best_cfg} {t_epi_add_bf16.best_ms:.4f} | "
        f"add-f32 {t_epi_add_f32.best_cfg} {t_epi_add_f32.best_ms:.4f} | "
        f"cast-f32 {t_epi_cast_f32.best_cfg} {t_epi_cast_f32.best_ms:.4f}",
        flush=True,
    )

    e_cast = t_epi_cast_f32.best_cfg
    e_add_bf16 = t_epi_add_bf16.best_cfg
    e_add_f32 = t_epi_add_f32.best_cfg

    def fused_chain(cfg, epi=None):
        cfg = dict(cfg, EPI=epi or e_cast)
        return make_fused_chain(a, b, r, out, acc32, cfg)

    def unfused_chain(cfg, epi=None):
        if epi is None:
            epi = e_add_f32 if cfg.get("SPLIT_K", 1) > 1 else e_add_bf16
        return make_unfused_chain(a, b, r, out, cmat, acc32, cfg, epi)

    # Numerical screen: run each config once and compare against the fp32 reference before
    # it is allowed to compete on time.  A config that is fast because it computes the
    # wrong thing is the failure this cannot afford -- nobody can re-run it here.
    def _v_fused():
        e = check(out, ref, label="screen-fused")
        return e["ok"], f"rel_err={e['rel_err']:.2e}"

    def _v_unfused():
        e = check(out, ref, label="screen-unfused")
        return e["ok"], f"rel_err={e['rel_err']:.2e}"

    # ---------------- stage 1: coarse, identical grid for both sides ----------------
    grid = coarse_grid(M)
    if quick:
        grid = B.quick_slice(grid, 16)
    print(f"  coarse grid: {len(grid)} configs (identical object for both arms)", flush=True)
    t_f_coarse = B.screened_autotune("fused/coarse", fused_chain, grid, _v_fused, tw, tr)
    t_u_coarse = B.screened_autotune(
        "unfused/coarse", unfused_chain, grid, _v_unfused, tw, tr
    )
    # `grid=` records the LIVE per-axis counts for this arm's stage. Both arms are handed
    # the same object here, so the two counts must be identical -- and if a future edit ever
    # makes them differ, the result file says so instead of the prose.
    fair.add(regime.name, "fused", "coarse", t_f_coarse, grid=grid)
    fair.add(regime.name, "unfused", "coarse", t_u_coarse, grid=grid)

    # ---------------- stage 2: refine, each side around its OWN winner ----------------
    seen_f = {tuple(sorted((k, str(v)) for k, v in c.items())) for c in grid}
    seen_u = set(seen_f)
    rg_f = refine_grid(t_f_coarse.best_cfg, M, seen_f)
    rg_u = refine_grid(t_u_coarse.best_cfg, M, seen_u)
    print(f"  refine grids: fused {len(rg_f)}, unfused {len(rg_u)}", flush=True)
    t_f_ref = (
        B.screened_autotune("fused/refine", fused_chain, rg_f, _v_fused, tw, tr)
        if rg_f else None
    )
    t_u_ref = (
        B.screened_autotune("unfused/refine", unfused_chain, rg_u, _v_unfused, tw, tr)
        if rg_u else None
    )
    fair.add(regime.name, "fused", "refine", t_f_ref, size=len(rg_f), grid=rg_f)
    fair.add(regime.name, "unfused", "refine", t_u_ref, size=len(rg_u), grid=rg_u)

    def merge(coarse, ref_t):
        if ref_t is None or ref_t.best_ms >= coarse.best_ms:
            return coarse.best_cfg, coarse.best_ms
        return ref_t.best_cfg, ref_t.best_ms

    f_cfg, f_ms = merge(t_f_coarse, t_f_ref)
    u_cfg, u_ms = merge(t_u_coarse, t_u_ref)

    # ---------------- stage 3: joint GEMM x epilogue refine ----------------
    epi_top_cast = B.top_cfgs(t_epi_cast_f32, k=5)
    f_top = B.top_cfgs(t_f_coarse, t_f_ref)
    joint_f, joint_u = [], []
    for cfg in f_top:
        if cfg.get("SPLIT_K", 1) == 1:
            continue  # no epilogue kernel in the chain at all
        for epi in epi_top_cast:
            joint_f.append((cfg, epi))
    for cfg in B.top_cfgs(t_u_coarse, t_u_ref):
        tops = B.top_cfgs(
            t_epi_add_f32 if cfg.get("SPLIT_K", 1) > 1 else t_epi_add_bf16, k=5
        )
        for epi in tops:
            joint_u.append((cfg, epi))

    # If the fused side's top configs are all SPLIT_K==1 the loop above leaves joint_f
    # empty while the unfused side still gets 3 GEMM x 5 epilogue = 15 timed chains -- the
    # two sides no longer have the same stage-3 budget, and the shortfall pushes the ratio
    # the same way as the effect under study.  Spend the difference on fresh GEMM
    # neighbours of the fused side's own best configs instead.
    extra_f: list[dict] = []
    if len(joint_f) < len(joint_u):
        for cfg in f_top:
            extra_f += refine_grid(cfg, M, seen_f)
            if len(extra_f) >= len(joint_u) - len(joint_f):
                break
        extra_f = extra_f[: len(joint_u) - len(joint_f)]
    print(
        f"  stage3 trials: fused {len(joint_f)}+{len(extra_f)} extra, "
        f"unfused {len(joint_u)}",
        flush=True,
    )

    t_f_joint = (
        autotune(lambda p: fused_chain(p[0], p[1]), joint_f, tw, tr) if joint_f else None
    )
    t_u_joint = (
        autotune(lambda p: unfused_chain(p[0], p[1]), joint_u, tw, tr) if joint_u else None
    )
    t_f_extra = (
        B.screened_autotune("fused/extra", fused_chain, extra_f, _v_fused, tw, tr)
        if extra_f else None
    )
    fair.add(regime.name, "fused", "joint", t_f_joint, size=len(joint_f))
    fair.add(regime.name, "unfused", "joint", t_u_joint, size=len(joint_u))
    fair.add(regime.name, "fused", "extra", t_f_extra, size=len(extra_f))

    f_epi, u_epi = None, None
    if t_f_joint is not None and t_f_joint.best_ms < f_ms:
        f_cfg, f_epi, f_ms = t_f_joint.best_cfg[0], t_f_joint.best_cfg[1], t_f_joint.best_ms
    if t_f_extra is not None and t_f_extra.best_ms < f_ms:
        f_cfg, f_epi, f_ms = t_f_extra.best_cfg, None, t_f_extra.best_ms
    if t_u_joint is not None and t_u_joint.best_ms < u_ms:
        u_cfg, u_epi, u_ms = t_u_joint.best_cfg[0], t_u_joint.best_cfg[1], t_u_joint.best_ms

    print(f"  BEST fused   {f_ms:.4f} ms  {f_cfg} epi={f_epi}", flush=True)
    print(f"  BEST unfused {u_ms:.4f} ms  {u_cfg} epi={u_epi}", flush=True)

    # ---------------- validate the winners ----------------
    out.zero_()
    for fn in fused_chain(f_cfg, f_epi):
        fn()
    chk_f = check(out, ref, label=f"{regime.name}/triton-fused")
    out_f = out.clone()

    out.zero_()
    cmat.zero_()
    for fn in unfused_chain(u_cfg, u_epi):
        fn()
    chk_u = check(out, ref, label=f"{regime.name}/triton-unfused")
    # not bitwise: the two arms are tuned independently and may differ in SPLIT_K, so the
    # atomic accumulation order differs.  The number itself is still recorded below.
    chk_fu = check(out_f, out.float(), tol=2e-2, label=f"{regime.name}/fused-vs-unfused")
    del out_f

    # ---------------- final timing: INTERLEAVED, PAIRED ----------------
    tim_f, tim_u, pair = B.bench_pair(
        fused_chain(f_cfg, f_epi), unfused_chain(u_cfg, u_epi), mw, mr, label=regime.name
    )

    # vendor BLAS reference lines, same tensors, also interleaved against each other
    vout = torch.empty_like(out)
    vc = torch.empty_like(out)
    tim_vf, tim_vu, pair_v = B.bench_pair(
        [lambda: torch.addmm(r, a, b, out=vout)],
        [lambda: torch.mm(a, b, out=vc), lambda: torch.add(vc, r, out=vout)],
        mw, mr, label=f"{regime.name}/vendor",
    )
    torch.addmm(r, a, b, out=vout)
    chk_v = check(vout, ref, label=f"{regime.name}/vendor-addmm")

    flops = 2.0 * M * N_OUT * K
    row = speedup_row(
        regime.name,
        tim_f,
        tim_u,
        {
            "M": M,
            "K": K,
            "N": N_OUT,
            "variant": "triton",
            "fused_cfg": {**f_cfg, "EPI": f_epi},
            "unfused_cfg": {**u_cfg, "EPI": u_epi},
            "paired_speedup": pair.get("paired_speedup_p50"),
            "paired_speedup_trimmed": pair.get("paired_speedup_trimmed_mean"),
            "pair_meta": pair,
            "tick": B.tick_report(tim_f.p50_ms, tim_u.p50_ms),
            "rel_err": chk_f["rel_err"],
            "rel_err_unfused": chk_u["rel_err"],
            "fused_vs_unfused_maxrel": chk_fu["rel_err"],
            "ok": bool(chk_f["ok"] and chk_u["ok"]),
            "vendor_fused_ms": tim_vf.p50_ms,
            "vendor_unfused_ms": tim_vu.p50_ms,
            "vendor_speedup": pair_v.get("paired_speedup_p50"),
            "vendor_rel_err": chk_v["rel_err"],
            "vendor_fused_p10_p90": [tim_vf.p10_ms, tim_vf.p90_ms],
            "vendor_unfused_p10_p90": [tim_vu.p10_ms, tim_vu.p90_ms],
            "fused_tflops": flops / (tim_f.p50_ms * 1e-3) / 1e12,
            "unfused_tflops": flops / (tim_u.p50_ms * 1e-3) / 1e12,
            "vendor_fused_tflops": flops / (tim_vf.p50_ms * 1e-3) / 1e12,
            "triton_vs_vendor": tim_vf.p50_ms / tim_f.p50_ms,
            "resid_bytes_saved": M * N_OUT * 2,
            "gemm_min_bytes": (M * K + K * N_OUT) * 2,
        },
    )
    print(
        f"  RESULT {regime.name}: triton fused {tim_f.p50_ms:.4f} / unfused "
        f"{tim_u.p50_ms:.4f} -> paired {row['paired_speedup']:.4f}x "
        f"(median-of-medians {row['speedup']:.4f}x) | vendor {tim_vf.p50_ms:.4f} / "
        f"{tim_vu.p50_ms:.4f} -> {row['vendor_speedup']:.4f}x"
        + ("  [TICK-LIMITED]" if row["tick"].get("tick_limited") else ""),
        flush=True,
    )

    tuning = {
        "regime": regime.name,
        "coarse_grid_size": len(grid),
        "refine_grid_size_fused": len(rg_f),
        "refine_grid_size_unfused": len(rg_u),
        "joint_grid_size_fused": len(joint_f),
        "joint_grid_size_unfused": len(joint_u),
        "extra_grid_size_fused": len(extra_f),  # stage-3 budget equaliser, see above
        "tune_fused_coarse": t_f_coarse.as_dict(),
        "tune_unfused_coarse": t_u_coarse.as_dict(),
        "tune_fused_refine": t_f_ref.as_dict() if t_f_ref else None,
        "tune_unfused_refine": t_u_ref.as_dict() if t_u_ref else None,
        "tune_fused_joint": t_f_joint.as_dict() if t_f_joint else None,
        "tune_unfused_joint": t_u_joint.as_dict() if t_u_joint else None,
        "tune_fused_extra": t_f_extra.as_dict() if t_f_extra else None,
        "tune_epi_add_bf16": t_epi_add_bf16.as_dict(),
        "tune_epi_add_f32": t_epi_add_f32.as_dict(),
        "tune_epi_cast_f32": t_epi_cast_f32.as_dict(),
        "timing_fused": tim_f.as_dict(),
        "timing_unfused": tim_u.as_dict(),
        "timing_vendor_fused": tim_vf.as_dict(),
        "timing_vendor_unfused": tim_vu.as_dict(),
        "checks": [chk_f, chk_u, chk_v, chk_fu],
    }

    del a, b, r, out, cmat, acc32, ref, vout, vc
    torch.cuda.empty_cache()
    return row, tuning


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
    B.resolve_units(UNITS, args.only)
    regimes = B.resolve_regimes(C, args.regimes)

    fair = B.Fairness(
        one_kernel_source="glm52_h200/kernels/oproj_resadd.py::oproj_gemm_kernel",
        flags="FUSE_RESADD constexpr selects the epilogue; unfused = the same kernel with "
              "the flag off plus epilogue_kernel(HAS_RES=True)",
        protocol=(
            "Both arms search the SAME coarse grid object, then each refines around its "
            "own winner, then each gets a stage-3 pass of the same size: joint "
            "GEMMxEpilogue where the side has an epilogue, extra GEMM neighbours where it "
            "does not (a SPLIT_K==1 fused chain has none). No config is ever shared. Every "
            "config is checked against the fp32 reference before it is timed."
        ),
        layout="B is [K,N] contiguous; identical tensor for triton and vendor.",
        split_k=(
            "SPLIT_K>1 accumulates into an fp32 buffer with tl.atomic_add, so the chain is "
            "[zero, gemm, epilogue]. Both sides pay that identically; the fused epilogue is "
            "a cast, the unfused one is cast+add."
        ),
        h200_axes=(
            "USE_TMA / warp_specialize / num_ctas are overlaid on the SHARED coarse grid, so "
            "both arms search them. They are structurally symmetric for this pair: the fused "
            "and unfused arms are the same GEMM kernel with one constexpr flipped, consume "
            "the same A and B, and therefore admit exactly the same descriptors and the same "
            "mainloop specialization. Per-arm counts are under grids.*.*.axis_counts."
        ),
    )
    fair.axis("f01_oproj_resadd", B.h200_axis_report(K))

    rows, tune_log, pair_meta = [], [], None
    for regime in regimes:
        ck = B.ckpt_load(RESULT_ID, regime.name, env, force=args.force)
        if ck is not None:
            print(f"[ckpt] reusing {regime.name}", flush=True)
            rows.append(ck["row"])
            tune_log.append(ck["tuning"])
            fair.grids.update(ck.get("fairness_grids", {}))
            continue
        row, tuning = run_regime(regime, args.quick, fair)
        pair_meta = row.get("pair_meta")
        B.ckpt_save(
            RESULT_ID, regime.name, env,
            {"row": row, "tuning": tuning,
             "fairness_grids": {regime.name: fair.grids.get(regime.name, {})}},
        )
        rows.append(row)
        tune_log.append(tuning)

    payload = {
        "id": RESULT_ID,
        "fusion": "o_proj GEMM + residual add (epilogue fusion, cuBLASLt beta=1 pattern)",
        "env": env.__dict__,
        "rows": rows,
        "tuning": tune_log,
        "fairness": fair.render(env, pair_meta),
    }
    path = record(RESULT_ID, payload)
    print(f"\nwrote {path}")
    print(
        f"{'regime':<16}{'fused':>10}{'unfused':>10}{'paired':>9}"
        f"{'vfused':>10}{'vunfused':>10}{'vspeedup':>10}{'rel_err':>10}{'tick':>7}"
    )
    for r in rows:
        print(
            f"{r['regime']:<16}{r['fused_ms']:>10.4f}{r['unfused_ms']:>10.4f}"
            f"{(r.get('paired_speedup') or r['speedup']):>9.4f}{r['vendor_fused_ms']:>10.4f}"
            f"{r['vendor_unfused_ms']:>10.4f}{(r.get('vendor_speedup') or 0):>10.4f}"
            f"{r['rel_err']:>10.2e}"
            f"{('!' if (r.get('tick') or {}).get('tick_limited') else ''):>7}"
        )


if __name__ == "__main__":
    main_guard(main)
