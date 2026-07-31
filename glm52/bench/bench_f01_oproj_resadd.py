"""Fusion #1 benchmark: o_proj GEMM + residual add (dense GEMM epilogue fusion).

Run:
    CUDA_VISIBLE_DEVICES=1 /home/zhangshuhan/my-envs/fusion/bin/python \
        glm52/bench/bench_f01_oproj_resadd.py

Reports FOUR numbers per regime:
  triton-fused    : oproj_gemm_kernel(FUSE_RESADD=True)
  triton-unfused  : oproj_gemm_kernel(FUSE_RESADD=False) + epilogue_kernel(HAS_RES=True)
  vendor-fused    : torch.addmm(residual, a, b)          (MetaX BLAS, beta=1)
  vendor-unfused  : torch.mm(a, b) ; torch.add(c, residual)

Tuning protocol (identical, independent budgets for the two Triton sides):
  0. epilogue_kernel tuned on its own grid, separately for each (dtype-in, HAS_RES) pair.
  1. coarse GEMM grid (same grid object for both sides) -> per-side winner
  2. refine grid built around each side's own coarse winner
  3. joint refine: that side's top-3 GEMM configs x every epilogue config, timed as chain
Nothing is ever shared between the fused and unfused searches.
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

torch.backends.cuda.matmul.allow_tf32 = False  # keep the fp32 reference exact

from glm52 import config as C
from glm52.common import (
    autotune,
    bench_chain,
    check,
    main_guard,
    record,
    speedup_row,
)
from glm52.kernels.oproj_resadd import (
    epilogue_launch,
    gemm_launch,
    make_fused_chain,
    make_unfused_chain,
    smem_bytes,
)

RESULT_ID = "f01_oproj_resadd"
DEV = "cuda"
N_OUT = C.HIDDEN_SIZE  # 6144
SMEM_LIMIT = 65536

REGIMES = [r for r in C.DECODE_REGIMES if r.T in (1, 32, 256)] + [
    r for r in C.PREFILL_REGIMES if r.T in (2048, 8192)
]
# optional subset for iteration: F01_REGIMES=decode_bs1,prefill_t2048
_only = os.environ.get("F01_REGIMES")
if _only:
    keep = set(_only.split(","))
    REGIMES = [r for r in REGIMES if r.name in keep]


# --------------------------------------------------------------------------------------
# Config-space construction
# --------------------------------------------------------------------------------------
def _valid_gemm(cfg: dict, M: int) -> bool:
    bm, bn, bk = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"]
    w, st = cfg["num_warps"], cfg["num_stages"]
    if smem_bytes(cfg) > SMEM_LIMIT:
        return False
    threads = w * 64
    per_thread = (bm * bn) / threads  # fp32 accumulator elements per lane
    if per_thread < 1 or per_thread > 64:
        return False
    # a/b tile fragments must be distributable over the lanes
    if (bm * bk) % threads or (bk * bn) % threads:
        return False
    if bm > 16 and bm > 2 * M:  # do not pay for tiles that are >50% padding
        return False
    return True


COARSE_CAP = 120


def coarse_grid(M: int) -> list[dict]:
    """Valid configs spanning the space, capped at COARSE_CAP by a seeded random sample.

    Sampling (rather than striding a product order) keeps the coarse stage unbiased in
    every axis; the refine stage then walks the local neighbourhood of the winner. The
    same grid object is handed to the fused and the unfused search.
    """
    if M <= 1:
        bms, bns, bks = [16], [64, 128, 256], [64, 128, 256]
        sks, ws, sts, gms = [1, 2, 4, 8, 16], [2, 4, 8], [2, 3], [8]
    elif M <= 32:
        bms, bns, bks = [16, 32], [64, 128, 256], [64, 128, 256]
        sks, ws, sts, gms = [1, 2, 4, 8, 16], [2, 4, 8], [2, 3], [8]
    elif M <= 256:
        bms, bns, bks = [16, 32, 64, 128], [32, 64, 128, 256], [32, 64, 128]
        sks, ws, sts, gms = [1, 2, 4, 8], [4, 8], [2, 3], [8]
    else:
        bms, bns, bks = [64, 128, 256], [64, 128, 256], [32, 64, 128]
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
    import random

    rng = random.Random(20260727)
    if len(rest) > COARSE_CAP - len(seeds):
        rest = rng.sample(rest, COARSE_CAP - len(seeds))
    return seeds + rest


_AXES = {
    "BLOCK_M": [16, 32, 64, 128, 256],
    "BLOCK_N": [16, 32, 64, 128, 256],
    "BLOCK_K": [16, 32, 64, 128, 256],
    "SPLIT_K": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
    "num_warps": [1, 2, 4, 8, 16],
    "num_stages": [1, 2, 3, 4, 5],
    "GROUP_M": [1, 2, 4, 8, 16],
}


def refine_grid(best: dict, M: int, seen: set) -> list[dict]:
    """+-1 step on every axis around `best`, plus a few 2-step moves on the tile axes."""
    out = []

    def add(cfg):
        key = tuple(sorted(cfg.items()))
        if key in seen or not _valid_gemm(cfg, M):
            return
        seen.add(key)
        out.append(cfg)

    for axis, vals in _AXES.items():
        if best[axis] not in vals:
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
            i = vals.index(cfg[axis])
            j = min(max(i + d, 0), len(vals) - 1)
            cfg[axis] = vals[j]
        add(cfg)
    return out


def epi_grid() -> list[dict]:
    out = []
    for blk, w in itertools.product(
        [256, 512, 1024, 2048, 4096, 8192, 16384], [1, 2, 4, 8, 16]
    ):
        per_thread = blk / (w * 64)
        if per_thread < 1 or per_thread > 64:
            continue
        out.append(dict(BLOCK=blk, num_warps=w, num_stages=1))
    return out


# --------------------------------------------------------------------------------------
# Per-regime driver
# --------------------------------------------------------------------------------------
def run_regime(regime, log: list) -> dict:
    M, K = regime.T, regime.oproj_k
    print(f"\n===== {regime.name}: M={M} K={K} N={N_OUT} =====", flush=True)

    torch.manual_seed(1234)
    a = (torch.randn(M, K, device=DEV, dtype=torch.float32) * 0.05).bfloat16()
    # o_proj weight. Production storage is nn.Linear [N, K]; the Triton kernels and the
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

    # tuning effort scales with the cost of one call
    if M >= 8192:
        tw, tr = 3, 8
    elif M >= 2048:
        tw, tr = 5, 12
    else:
        tw, tr = 10, 30

    # ---------------- stage 0: epilogue kernel, tuned on its own grid ----------------
    eg = epi_grid()
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

    # ---------------- stage 1: coarse, identical grid for both sides ----------------
    grid = coarse_grid(M)
    print(f"  coarse grid: {len(grid)} configs", flush=True)
    t_f_coarse = autotune(fused_chain, grid, tw, tr)
    t_u_coarse = autotune(unfused_chain, grid, tw, tr)
    print(
        f"  coarse fused   {t_f_coarse.best_ms:.4f} ms {t_f_coarse.best_cfg} "
        f"(fail {t_f_coarse.n_failed}/{t_f_coarse.n_tried})",
        flush=True,
    )
    print(
        f"  coarse unfused {t_u_coarse.best_ms:.4f} ms {t_u_coarse.best_cfg} "
        f"(fail {t_u_coarse.n_failed}/{t_u_coarse.n_tried})",
        flush=True,
    )

    # ---------------- stage 2: refine, each side around its OWN winner ----------------
    seen_f = {tuple(sorted(c.items())) for c in grid}
    seen_u = set(seen_f)
    rg_f = refine_grid(t_f_coarse.best_cfg, M, seen_f)
    rg_u = refine_grid(t_u_coarse.best_cfg, M, seen_u)
    print(f"  refine grids: fused {len(rg_f)}, unfused {len(rg_u)}", flush=True)
    t_f_ref = autotune(fused_chain, rg_f, tw, tr) if rg_f else None
    t_u_ref = autotune(unfused_chain, rg_u, tw, tr) if rg_u else None

    def merge(coarse, ref):
        if ref is None or ref.best_ms >= coarse.best_ms:
            return coarse.best_cfg, coarse.best_ms
        return ref.best_cfg, ref.best_ms

    f_cfg, f_ms = merge(t_f_coarse, t_f_ref)
    u_cfg, u_ms = merge(t_u_coarse, t_u_ref)

    # ---------------- stage 3: joint GEMM x epilogue refine ----------------
    def top_k_cfgs(*tunes, k=3):
        rows = []
        for t in tunes:
            if t is None:
                continue
            rows += [(ms, cfg) for cfg, ms, err in t.table if ms is not None]
        rows.sort(key=lambda x: x[0])
        picked, keys = [], set()
        for ms, cfg in rows:
            key = tuple(sorted(cfg.items()))
            if key in keys:
                continue
            keys.add(key)
            picked.append(cfg)
            if len(picked) == k:
                break
        return picked

    epi_top_cast = top_k_cfgs(t_epi_cast_f32, k=5)
    joint_f, joint_u = [], []
    for cfg in top_k_cfgs(t_f_coarse, t_f_ref):
        if cfg.get("SPLIT_K", 1) == 1:
            continue  # no epilogue kernel in the chain at all
        for epi in epi_top_cast:
            joint_f.append((cfg, epi))
    for cfg in top_k_cfgs(t_u_coarse, t_u_ref):
        tops = top_k_cfgs(
            t_epi_add_f32 if cfg.get("SPLIT_K", 1) > 1 else t_epi_add_bf16, k=5
        )
        for epi in tops:
            joint_u.append((cfg, epi))

    t_f_joint = (
        autotune(lambda p: fused_chain(p[0], p[1]), joint_f, tw, tr) if joint_f else None
    )
    t_u_joint = (
        autotune(lambda p: unfused_chain(p[0], p[1]), joint_u, tw, tr)
        if joint_u
        else None
    )
    f_epi, u_epi = None, None
    if t_f_joint is not None and t_f_joint.best_ms < f_ms:
        f_cfg, f_epi, f_ms = t_f_joint.best_cfg[0], t_f_joint.best_cfg[1], t_f_joint.best_ms
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
    chk_fu = check(out_f, out.float(), tol=0.0, label=f"{regime.name}/fused-vs-unfused")
    del out_f

    # ---------------- final timing, high rep ----------------
    mw, mr = (5, 30) if M >= 8192 else ((10, 60) if M >= 2048 else (25, 200))
    tim_f = bench_chain(fused_chain(f_cfg, f_epi), mw, mr)
    tim_u = bench_chain(unfused_chain(u_cfg, u_epi), mw, mr)

    # vendor BLAS reference lines, same tensors
    vout = torch.empty_like(out)
    vc = torch.empty_like(out)
    tim_vf = bench_chain([lambda: torch.addmm(r, a, b, out=vout)], mw, mr)
    tim_vu = bench_chain(
        [lambda: torch.mm(a, b, out=vc), lambda: torch.add(vc, r, out=vout)], mw, mr
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
            "rel_err": chk_f["rel_err"],
            "rel_err_unfused": chk_u["rel_err"],
            "fused_vs_unfused_maxrel": chk_fu["rel_err"],
            "ok": bool(chk_f["ok"] and chk_u["ok"]),
            "vendor_fused_ms": tim_vf.p50_ms,
            "vendor_unfused_ms": tim_vu.p50_ms,
            "vendor_speedup": tim_vu.p50_ms / tim_vf.p50_ms,
            "vendor_rel_err": chk_v["rel_err"],
            "vendor_fused_p10_p90": [tim_vf.p10_ms, tim_vf.p90_ms],
            "vendor_unfused_p10_p90": [tim_vu.p10_ms, tim_vu.p90_ms],
            "fused_tflops": flops / (tim_f.p50_ms * 1e-3) / 1e12,
            "unfused_tflops": flops / (tim_u.p50_ms * 1e-3) / 1e12,
            "vendor_fused_tflops": flops / (tim_vf.p50_ms * 1e-3) / 1e12,
            "triton_vs_vendor": tim_vf.p50_ms / tim_f.p50_ms,
            "fused_noflush_ms": tim_f.noflush_p50_ms,
            "unfused_noflush_ms": tim_u.noflush_p50_ms,
            "resid_bytes_saved": M * N_OUT * 2,
            "gemm_min_bytes": (M * K + K * N_OUT) * 2,
        },
    )
    print(
        f"  RESULT {regime.name}: triton fused {tim_f.p50_ms:.4f} / unfused "
        f"{tim_u.p50_ms:.4f} -> {row['speedup']:.4f}x | vendor {tim_vf.p50_ms:.4f} / "
        f"{tim_vu.p50_ms:.4f} -> {tim_vu.p50_ms / tim_vf.p50_ms:.4f}x",
        flush=True,
    )

    log.append(
        {
            "regime": regime.name,
            "coarse_grid_size": len(grid),
            "refine_grid_size_fused": len(rg_f),
            "refine_grid_size_unfused": len(rg_u),
            "joint_grid_size_fused": len(joint_f),
            "joint_grid_size_unfused": len(joint_u),
            "tune_fused_coarse": t_f_coarse.as_dict(),
            "tune_unfused_coarse": t_u_coarse.as_dict(),
            "tune_fused_refine": t_f_ref.as_dict() if t_f_ref else None,
            "tune_unfused_refine": t_u_ref.as_dict() if t_u_ref else None,
            "tune_fused_joint": t_f_joint.as_dict() if t_f_joint else None,
            "tune_unfused_joint": t_u_joint.as_dict() if t_u_joint else None,
            "tune_epi_add_bf16": t_epi_add_bf16.as_dict(),
            "tune_epi_add_f32": t_epi_add_f32.as_dict(),
            "tune_epi_cast_f32": t_epi_cast_f32.as_dict(),
            "timing_fused": tim_f.as_dict(),
            "timing_unfused": tim_u.as_dict(),
            "timing_vendor_fused": tim_vf.as_dict(),
            "timing_vendor_unfused": tim_vu.as_dict(),
            "checks": [chk_f, chk_u, chk_v, chk_fu],
        }
    )

    del a, b, r, out, cmat, acc32, ref, vout, vc
    torch.cuda.empty_cache()
    return row


CKPT_DIR = Path(__file__).resolve().parents[2] / "results" / f"_{RESULT_ID}_ckpt"


def main() -> None:
    import json

    env = C.BenchEnv.probe()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    rows, tune_log = [], []
    for regime in REGIMES:
        ck = CKPT_DIR / f"{regime.name}.json"
        if ck.exists() and not os.environ.get("F01_FORCE"):
            blob = json.loads(ck.read_text())
            print(f"[ckpt] reusing {ck}", flush=True)
            rows.append(blob["row"])
            tune_log.append(blob["tuning"])
            continue
        sub: list = []
        row = run_regime(regime, sub)
        ck.write_text(json.dumps({"row": row, "tuning": sub[0]}, indent=1, default=str))
        rows.append(row)
        tune_log.append(sub[0])
    payload = {
        "id": RESULT_ID,
        "fusion": "o_proj GEMM + residual add (epilogue fusion, cuBLASLt beta=1 pattern)",
        "env": env.__dict__,
        "rows": rows,
        "tuning": tune_log,
        "notes": {
            "layout": "B is [K,N] contiguous; identical tensor for triton and vendor.",
            "fairness": (
                "One kernel source; FUSE_RESADD constexpr selects the epilogue. Unfused = "
                "same kernel with the flag off + epilogue_kernel(HAS_RES=True). Both sides "
                "searched the SAME coarse grid, then each refined around its own winner, "
                "then each got a joint GEMMxEpilogue pass. No config is ever shared."
            ),
            "split_k": (
                "SPLIT_K>1 accumulates into an fp32 buffer with tl.atomic_add, so the chain "
                "is [zero, gemm, epilogue]. Both sides pay that identically; the fused "
                "epilogue is a cast, the unfused one is cast+add."
            ),
        },
    }
    path = record(RESULT_ID, payload)
    print(f"\nwrote {path}")
    print(
        f"{'regime':<16}{'fused':>10}{'unfused':>10}{'speedup':>9}"
        f"{'vfused':>10}{'vunfused':>10}{'vspeedup':>10}{'rel_err':>10}"
    )
    for r in rows:
        print(
            f"{r['regime']:<16}{r['fused_ms']:>10.4f}{r['unfused_ms']:>10.4f}"
            f"{r['speedup']:>9.4f}{r['vendor_fused_ms']:>10.4f}"
            f"{r['vendor_unfused_ms']:>10.4f}{r['vendor_speedup']:>10.4f}"
            f"{r['rel_err']:>10.2e}"
        )


if __name__ == "__main__":
    main_guard(main)
