"""Fusion #6 -- MoE Up/Gate grouped GEMM + SwiGLU epilogue.

FUSED    : one kernel, grid over N = I = 2048, two fp32 accumulators (gate cols and up
           cols) sharing one K-loop over the gathered A tile, `silu(g)*u` in the epilogue,
           writes only [rows, 2048].
UNFUSED  : the SAME kernel with FUSE_ACT=False -- grid over N = 2I = 4096, one
           accumulator, writes the full [rows, 4096] intermediate -- followed by a
           separate element-wise `silu_and_mul` kernel reading [rows, 4096] and writing
           [rows, 2048].  This is exactly what sglang 0.5.10 does today.

Both sides produce the identical downstream tensor `down_input` [rows, 2048] bf16.
The unfused side additionally materialises the [rows, 4096] intermediate -- that extra
traffic IS the fusion opportunity.

Run:
  CUDA_VISIBLE_DEVICES=0 /home/zhangshuhan/my-envs/fusion/bin/python \
      glm52/bench/bench_f06_upgate_swiglu.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from glm52 import config as C  # noqa: E402
from glm52 import reference as R  # noqa: E402
from glm52.common import (  # noqa: E402
    autotune,
    bench_chain,
    check,
    main_guard,
    record,
    speedup_row,
)
from glm52.kernels.moe_gateup import (  # noqa: E402
    launch_gateup,
    launch_silu_and_mul,
    smem_bytes,
)

RESULT_ID = "f06_upgate_swiglu"
SMEM_LIMIT = 65536
H = C.HIDDEN_SIZE          # 6144
I = C.MOE_INTERMEDIATE_SIZE  # 2048
E = C.N_ROUTED_EXPERTS     # 256
TOPK = C.NUM_EXPERTS_PER_TOK  # 8

REGIMES = [r for r in C.DECODE_REGIMES if r.T in (1, 32, 256)] + [
    r for r in C.PREFILL_REGIMES if r.T in (2048, 8192)
]


# --------------------------------------------------------------------------------------
# Config-space generation.  Identical rules for both variants; the SMEM / accumulator
# filters differ only because the fused kernel genuinely stages a second B tile and holds
# a second accumulator.
# --------------------------------------------------------------------------------------
def gemm_grid(fused: bool, big: bool) -> list[dict]:
    if big:  # T >= 2048: drop mappings that are structurally hopeless for a big GEMM
        bms, bns, bks, warps = [32, 64, 128], [64, 128, 256], [32, 64, 128], [4, 8, 16]
    else:
        bms, bns, bks, warps = (
            [16, 32, 64, 128],
            [32, 64, 128, 256],
            [32, 64, 128],
            [2, 4, 8, 16],
        )
    nacc = 2 if fused else 1
    out = []
    for bm, bn, bk, w, s in itertools.product(bms, bns, bks, warps, [2, 3]):
        cfg = dict(
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, GROUP_M=8
        )
        if smem_bytes(cfg, fused) > SMEM_LIMIT:
            continue
        acc_per_lane = nacc * bm * bn / (w * 64)
        if acc_per_lane > 128 or acc_per_lane < 4:
            continue
        out.append(cfg)
    return out


def gemm_refine(best: dict, fused: bool) -> list[dict]:
    def nb(v, lo, hi):
        return sorted({max(lo, v // 2), v, min(hi, v * 2)})

    out, seen = [], set()
    cands = []
    for bm in nb(best["BLOCK_M"], 16, 256):
        for bn in nb(best["BLOCK_N"], 32, 256):
            for w in nb(best["num_warps"], 1, 16):
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
    nacc = 2 if fused else 1
    for bm, bn, bk, w, s, g in cands:
        cfg = dict(
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, GROUP_M=g
        )
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        if smem_bytes(cfg, fused) > SMEM_LIMIT:
            continue
        acc_per_lane = nacc * bm * bn / (w * 64)
        if acc_per_lane > 128 or acc_per_lane < 2:
            continue
        out.append(cfg)
    return out


def act_grid() -> list[dict]:
    out = []
    for bm, bn, w, s in itertools.product(
        [1, 2, 4, 8, 16, 32, 64], [64, 128, 256, 512, 1024, 2048], [1, 2, 4, 8], [1, 2]
    ):
        if bm * bn > 8192 or bm * bn < 256:
            continue
        if bm * bn < w * 64:
            continue
        out.append(dict(BLOCK_M=bm, BLOCK_N=bn, num_warps=w, num_stages=s))
    return out


# --------------------------------------------------------------------------------------
# Per-regime problem setup
# --------------------------------------------------------------------------------------
class Problem:
    def __init__(self, regime, w13, gate_w, seed=0):
        torch.manual_seed(seed + regime.T)
        dev = "cuda"
        self.regime = regime
        self.T = regime.T
        self.rows = regime.T * TOPK
        self.w13 = w13
        # post_attention_layernorm output feeding the expert GEMM
        self.x = (torch.randn(self.T, H, device=dev, dtype=torch.float32) * 0.1).to(
            torch.bfloat16
        )
        _, self.topk_weights, self.topk_ids = R.router(self.x, gate_w)
        self.layouts: dict[int, tuple] = {}
        # output buffers (allocated once, as sglang allocates a workspace)
        self.c_fused = torch.zeros(self.rows, I, device=dev, dtype=torch.bfloat16)
        self.c_inter = torch.zeros(self.rows, 2 * I, device=dev, dtype=torch.bfloat16)
        self.c_unfused = torch.zeros(self.rows, I, device=dev, dtype=torch.bfloat16)

    def layout(self, block_m: int):
        if block_m not in self.layouts:
            self.layouts[block_m] = R.moe_align_block_size(self.topk_ids, block_m, E)
        return self.layouts[block_m]

    # --- callable factories ---------------------------------------------------------
    def fused_fn(self, cfg):
        sti, eids, ntp = self.layout(cfg["BLOCK_M"])
        return lambda: launch_gateup(
            self.x, self.w13, self.c_fused, sti, eids, ntp,
            self.rows, TOPK, I, cfg, fused=True,
        )

    def gemm_fn(self, cfg):
        sti, eids, ntp = self.layout(cfg["BLOCK_M"])
        return lambda: launch_gateup(
            self.x, self.w13, self.c_inter, sti, eids, ntp,
            self.rows, TOPK, I, cfg, fused=False,
        )

    def act_fn(self, cfg):
        return lambda: launch_silu_and_mul(self.c_inter, self.c_unfused, cfg)


# --------------------------------------------------------------------------------------
# fp32 reference on a sampled row subset (a full-size fp32 reference at T=8192 would be
# 3.3 TFLOP of fp32 matmul; sampling keeps validation honest and affordable)
# --------------------------------------------------------------------------------------
def reference_rows(prob: Problem, n_sample: int = 2048):
    rows = prob.rows
    if rows <= n_sample:
        idx = torch.arange(rows, device="cuda")
    else:
        g = torch.Generator(device="cuda").manual_seed(1234)
        idx = torch.randperm(rows, device="cuda", generator=g)[:n_sample].sort().values
    tok = (idx // TOPK).long()
    kk = (idx % TOPK).long()
    experts = prob.topk_ids.long()[tok, kk]
    ref = torch.empty(idx.numel(), I, device="cuda", dtype=torch.float32)
    xs = prob.x.float()[tok]
    for e in torch.unique(experts).tolist():
        sel = (experts == e).nonzero(as_tuple=True)[0]
        h = xs[sel] @ prob.w13[e].float().t()
        ref[sel] = (torch.nn.functional.silu(h[:, :I]) * h[:, I:]).float()
    return idx, ref


# --------------------------------------------------------------------------------------
# Vendor-BLAS production reference: per-expert torch.matmul + torch silu_and_mul.
# A rows are pre-gathered per expert OUTSIDE the timed region (stated in the log), so this
# is the best case for the vendor path -- pure GEMM + activation, no gather cost.
# --------------------------------------------------------------------------------------
def vendor_chain(prob: Problem):
    flat = prob.topk_ids.flatten().long()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E)
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device="cuda"), counts.cumsum(0)[:-1]])
    a_sorted = prob.x[(order // TOPK)].contiguous()
    out = torch.empty(prob.rows, I, device="cuda", dtype=torch.bfloat16)
    segs = []
    cs, ss = counts.tolist(), starts.tolist()
    for e in range(E):
        if cs[e]:
            segs.append((e, ss[e], ss[e] + cs[e]))
    wt = [prob.w13[e].t() for e, _, _ in segs]  # bf16 views, no copy

    def run():
        for (e, s, t), w in zip(segs, wt):
            h = torch.matmul(a_sorted[s:t], w)
            out[s:t] = torch.nn.functional.silu(h[:, :I]) * h[:, I:]

    return [run]


# --------------------------------------------------------------------------------------
def kernel_stats(prob: Problem, cfg: dict, fused: bool):
    try:
        k = prob.fused_fn(cfg)() if fused else prob.gemm_fn(cfg)()
        return {
            "n_regs": getattr(k, "n_regs", None),
            "n_spills": getattr(k, "n_spills", None),
            "shared_bytes": getattr(getattr(k, "metadata", None), "shared", None),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:160]}


def make_weights():
    """w13 with an sglang-style trailing pad.

    sglang guards its `fused_moe` weights with `SGLANG_MOE_PADDING`; here the pad is a
    hard requirement, not an option: Triton's software pipeline on this MACA backend
    issues speculative (unpredicated) B-tile loads for the peeled prologue/epilogue
    stages, so the last expert's `up` tile can be fetched one BLOCK_K past the end of
    the tensor.  Without the pad that is an ATU (page) fault that kills the context.
    The pad is allocated for BOTH variants and is never read into an accumulator, so it
    changes no arithmetic and gives neither side an advantage.
    """
    numel = E * 2 * I * H
    pad = 1 << 20  # 2 MiB of slack, >> any BLOCK_K * BLOCK_N tile
    buf = torch.empty(numel + pad, device="cuda", dtype=torch.bfloat16)
    w13 = buf[:numel].view(E, 2 * I, H)
    for e in range(E):  # chunked init: avoids a 12.9 GB fp32 temporary
        w13[e].normal_(0, 0.02)
    buf[numel:].zero_()
    return buf, w13


def run_regime(regime, w13, gate_w, quick: bool) -> tuple[dict, dict]:
    big = regime.T >= 2048
    if quick:
        w_t, r_t = 2, 5
    elif regime.T >= 8192:
        w_t, r_t = 3, 8
    elif regime.T >= 2048:
        w_t, r_t = 4, 12
    elif regime.T >= 256:
        w_t, r_t = 8, 20
    else:
        w_t, r_t = 10, 30
    w_f, r_f = (3, 10) if big else (25, 100)

    with torch.no_grad():
        print(f"\n===== {regime.name} (T={regime.T}, rows={regime.T*TOPK}) =====", flush=True)
        prob = Problem(regime, w13, gate_w)

        # ---------------- FUSED ----------------
        cg = gemm_grid(True, big)
        if quick:
            cg = cg[::7]
        print(f"  fused coarse: {len(cg)} cfgs", flush=True)
        tf_c = autotune(lambda c: [prob.fused_fn(c)], cg, warmup=w_t, rep=r_t)
        rg = gemm_refine(tf_c.best_cfg, True)
        print(f"  fused coarse best {tf_c.best_cfg} {tf_c.best_ms:.4f} ms; refine {len(rg)}", flush=True)
        tf_r = autotune(lambda c: [prob.fused_fn(c)], rg, warmup=w_t, rep=r_t)
        fused_cfg = tf_c.best_cfg if tf_c.best_ms <= tf_r.best_ms else tf_r.best_cfg
        print(f"  FUSED best {fused_cfg} {min(tf_c.best_ms, tf_r.best_ms):.4f} ms", flush=True)

        # ---------------- UNFUSED: GEMM ----------------
        cg2 = gemm_grid(False, big)
        if quick:
            cg2 = cg2[::7]
        print(f"  unfused-GEMM coarse: {len(cg2)} cfgs", flush=True)
        tu_c = autotune(lambda c: [prob.gemm_fn(c)], cg2, warmup=w_t, rep=r_t)
        rg2 = gemm_refine(tu_c.best_cfg, False)
        print(f"  unfused-GEMM coarse best {tu_c.best_cfg} {tu_c.best_ms:.4f} ms; refine {len(rg2)}", flush=True)
        tu_r = autotune(lambda c: [prob.gemm_fn(c)], rg2, warmup=w_t, rep=r_t)

        # ---------------- UNFUSED: silu_and_mul ----------------
        ag = act_grid()
        if quick:
            ag = ag[::5]
        print(f"  act grid: {len(ag)} cfgs", flush=True)
        ta = autotune(lambda c: [prob.act_fn(c)], ag, warmup=w_t, rep=r_t)
        print(f"  ACT best {ta.best_cfg} {ta.best_ms:.4f} ms", flush=True)

        # top-3 x top-3 joint chain re-time (guards against a separately-tuned optimum
        # that is not the joint optimum)
        def top_k_cfgs(*tables, k=3):
            rowsx = [(m, c) for tb in tables for (c, m, err) in tb if m is not None]
            rowsx.sort(key=lambda t: t[0])
            seen, out = set(), []
            for m, c in rowsx:
                key = tuple(sorted(c.items()))
                if key in seen:
                    continue
                seen.add(key)
                out.append(c)
                if len(out) == k:
                    break
            return out

        best_chain_ms, best_pair = float("inf"), None
        joint = []
        for gc in top_k_cfgs(tu_c.table, tu_r.table):
            for ac in top_k_cfgs(ta.table):
                try:
                    t = bench_chain(
                        [prob.gemm_fn(gc), prob.act_fn(ac)], warmup=w_t, rep=r_t
                    )
                    joint.append(({"gemm": gc, "act": ac}, t.p50_ms, None))
                    if t.p50_ms < best_chain_ms:
                        best_chain_ms, best_pair = t.p50_ms, (gc, ac)
                except Exception as exc:  # noqa: BLE001
                    joint.append(({"gemm": gc, "act": ac}, None, str(exc)[:160]))
        gemm_cfg, act_cfg = best_pair
        print(f"  UNFUSED best chain {gemm_cfg} + {act_cfg} {best_chain_ms:.4f} ms", flush=True)

        # ---------------- validate ----------------
        prob.c_fused.zero_()
        prob.c_inter.zero_()
        prob.c_unfused.zero_()
        prob.fused_fn(fused_cfg)()
        prob.gemm_fn(gemm_cfg)()
        prob.act_fn(act_cfg)()
        torch.cuda.synchronize()
        idx, ref = reference_rows(prob)
        chk_f = check(prob.c_fused[idx], ref, label="fused")
        chk_u = check(prob.c_unfused[idx], ref, label="unfused")
        agree = check(prob.c_fused[idx], prob.c_unfused[idx].float(), label="fused_vs_unfused")
        print(f"  rel_err fused={chk_f['rel_err']:.3e} unfused={chk_u['rel_err']:.3e} "
              f"agree={agree['rel_err']:.3e}", flush=True)
        if not (chk_f["ok"] and chk_u["ok"]):
            raise RuntimeError(f"validation failed at {regime.name}: {chk_f} {chk_u}")

        # ---------------- final timing ----------------
        t_fused = bench_chain([prob.fused_fn(fused_cfg)], warmup=w_f, rep=r_f)
        t_unfused = bench_chain(
            [prob.gemm_fn(gemm_cfg), prob.act_fn(act_cfg)], warmup=w_f, rep=r_f
        )
        t_gemm_only = bench_chain([prob.gemm_fn(gemm_cfg)], warmup=w_f, rep=r_f)
        t_act_only = bench_chain([prob.act_fn(act_cfg)], warmup=w_f, rep=r_f)
        t_vendor = bench_chain(vendor_chain(prob), warmup=max(2, w_f // 3), rep=max(5, r_f // 3))

        rows_n = prob.rows
        flops = 2.0 * rows_n * (2 * I) * H
        row = speedup_row(
            regime.name,
            t_fused,
            t_unfused,
            extra={
                "T": regime.T,
                "moe_rows": rows_n,
                "fused_cfg": fused_cfg,
                "unfused_gemm_cfg": gemm_cfg,
                "unfused_act_cfg": act_cfg,
                "unfused_gemm_ms": t_gemm_only.p50_ms,
                "unfused_act_ms": t_act_only.p50_ms,
                "vendor_blas_ms": t_vendor.p50_ms,
                "vendor_tflops": flops / (t_vendor.p50_ms * 1e-3) / 1e12,
                "fused_tflops": flops / (t_fused.p50_ms * 1e-3) / 1e12,
                "unfused_tflops": flops / (t_unfused.p50_ms * 1e-3) / 1e12,
                "rel_err": chk_f["rel_err"],
                "rel_err_unfused": chk_u["rel_err"],
                "fused_vs_unfused_rel_err": agree["rel_err"],
                "gflop": flops / 1e9,
                "traffic": {
                    "A_bytes": rows_n * H * 2,
                    "unfused_intermediate_bytes": rows_n * 2 * I * 2 * 2 + rows_n * I * 2,
                    "fused_intermediate_bytes": rows_n * I * 2,
                    "saved_bytes": rows_n * 2 * I * 2 * 2,
                    "weight_bytes_min": int(
                        torch.unique(prob.topk_ids).numel() * 2 * I * H * 2
                    ),
                },
                "fused_kernel_stats": kernel_stats(prob, fused_cfg, True),
                "unfused_kernel_stats": kernel_stats(prob, gemm_cfg, False),
                "fused_noflush_ms": t_fused.noflush_p50_ms,
                "unfused_noflush_ms": t_unfused.noflush_p50_ms,
            },
        )
        tuning = {
            "fused_coarse": tf_c.as_dict(),
            "fused_refine": tf_r.as_dict(),
            "unfused_gemm_coarse": tu_c.as_dict(),
            "unfused_gemm_refine": tu_r.as_dict(),
            "unfused_act": ta.as_dict(),
            "unfused_joint_chain": joint,
        }
        print(
            f"  RESULT {regime.name}: fused {t_fused.p50_ms:.4f} ms | "
            f"unfused {t_unfused.p50_ms:.4f} ms (gemm {t_gemm_only.p50_ms:.4f} + act "
            f"{t_act_only.p50_ms:.4f}) | speedup {row['speedup']:.3f}x | "
            f"vendor {t_vendor.p50_ms:.4f} ms",
            flush=True,
        )
        del prob
        torch.cuda.empty_cache()
        return row, tuning


# --------------------------------------------------------------------------------------
# Worker / driver.  Each regime runs in its OWN process: the MACA runtime disables the
# whole context after an ATU fault, so one bad launch would otherwise lose the entire run.
# --------------------------------------------------------------------------------------
def worker(regime_name: str, out_path: str, quick: bool):
    regime = next(r for r in REGIMES if r.name == regime_name)
    torch.manual_seed(0)
    print("allocating w13 [%d, %d, %d] bf16 = %.1f GB" % (E, 2 * I, H, E * 2 * I * H * 2 / 2**30), flush=True)
    _buf, w13 = make_weights()
    gate_w = torch.randn(E, H, device="cuda", dtype=torch.float32) * 0.02
    row, tuning = run_regime(regime, w13, gate_w, quick)
    Path(out_path).write_text(json.dumps({"row": row, "tuning": tuning}, default=str))
    print(f"worker wrote {out_path}", flush=True)


def driver(quick: bool, only: list[str] | None):
    import subprocess

    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"{RESULT_ID}_parts"
    tmp.mkdir(parents=True, exist_ok=True)
    env = C.BenchEnv.probe()
    payload = {
        "id": RESULT_ID,
        "fusion": "#6 MoE Up/Gate grouped GEMM + SwiGLU epilogue",
        "shapes": {"H": H, "I": I, "E": E, "topk": TOPK, "K": H, "N_unfused": 2 * I, "N_fused": I},
        "env": env.__dict__,
        "tuning_protocol": (
            "Per regime, per side: coarse grid (SMEM- and accumulator-prefiltered, "
            "identical generation rules for both sides) then a neighbourhood refine "
            "around the coarse winner. The unfused chain's two kernels are tuned "
            "SEPARATELY (GEMM alone, silu_and_mul alone), then the top-3 x top-3 "
            "combinations are re-timed AS A CHAIN and the best chain is reported, so "
            "the separately-tuned optimum cannot under-sell the unfused side. Each "
            "regime runs in its own process for crash isolation."
        ),
        "rows": [],
        "tuning": {},
        "failed_regimes": {},
    }
    for regime in REGIMES:
        if only and regime.name not in only:
            continue
        part = tmp / f"{regime.name}.json"
        if part.exists():
            part.unlink()
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--worker", regime.name, "--out", str(part)]
        if quick:
            cmd.append("--quick")
        ok = False
        for attempt in range(2):
            rc = subprocess.call(cmd)
            if rc == 0 and part.exists():
                ok = True
                break
            print(f"!! {regime.name} worker failed (rc={rc}), attempt {attempt+1}", flush=True)
        if not ok:
            payload["failed_regimes"][regime.name] = "worker process aborted twice"
            continue
        d = json.loads(part.read_text())
        payload["rows"].append(d["row"])
        payload["tuning"][regime.name] = d["tuning"]

    p = record(RESULT_ID, payload)
    print(f"\nwrote {p}", flush=True)
    print(f"{'regime':16s} {'fused':>10s} {'unfused':>10s} {'speedup':>8s} {'vendor':>10s} {'rel_err':>10s}")
    for r in payload["rows"]:
        print(f"{r['regime']:16s} {r['fused_ms']:10.4f} {r['unfused_ms']:10.4f} "
              f"{r['speedup']:8.3f} {r['vendor_blas_ms']:10.4f} {r['rel_err']:10.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--worker", default=None, help="internal: run one regime")
    ap.add_argument("--out", default=None, help="internal: worker output json")
    ap.add_argument("--only", default=None, help="comma-separated regime names")
    a = ap.parse_args()
    if a.worker:
        worker(a.worker, a.out, a.quick)
    else:
        driver(a.quick, a.only.split(",") if a.only else None)


if __name__ == "__main__":
    main_guard(main)
