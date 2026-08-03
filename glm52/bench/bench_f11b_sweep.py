"""Fusion #11b (Lazy Pre-Norm folded into the router GEMM) — gain vs T.

    UNFUSED : rmsnorm kernel  (read h1, write x2)  ->  router GEMM (read x2)
    FUSED   : ONE router GEMM reading the un-normalized h1, accumulating the row
              sum-of-squares in the same k-loop and applying rstd as an epilogue scale.
              x2 is never materialized; the RMSNorm affine weight is pre-folded into the
              gate weight's rows OFFLINE (outside the timed region), which is what makes the
              affine-free identity (A*rstd) @ B == (A @ B) * rstd apply to GLM-5.2.

Both sides independently retuned at every T, each point the median of 5 interleaved rounds,
both validated against an fp32 reference. Same protocol as the #8/#9 sweeps.

DYNAMIC SCHEDULING: with `--queue`, workers claim tasks from a shared directory by atomic
`os.rename` (atomic on POSIX, so exactly one worker wins each task) and loop until it is
empty. Three workers on three GPUs therefore self-balance — a worker that draws a cheap T
immediately takes another, instead of idling while a neighbour grinds through a big one.
(One GPU here, so one worker; the queue is still what makes a run resumable.)

T is capped by VRAM, not by taste: see `max_safe_T` below. The C500 sweep ran to T=262144,
which needs 21.2 GiB in the fp32 reference and cannot run on an 8 GB card.

    python3 glm52/bench/bench_f11b_sweep.py --make-queue decode:1,2,4 prefill:256,512
    python3 glm52/bench/bench_f11b_sweep.py --queue --worker gpu0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from glm52 import config as C  # noqa: E402
from glm52 import reference as R  # noqa: E402
from glm52.bench.bench_f11_lazy_prenorm import _ok, norm_grid  # noqa: E402
from glm52.bench.bench_f8_sweep import rep_budget  # noqa: E402
from glm52.bench.bench_layer import _tile_then_coord  # noqa: E402
from glm52.common import RESULTS_DIR, autotune, bench_chain, rel_err  # noqa: E402
from glm52.kernels import add_rmsnorm as NK  # noqa: E402
from glm52.kernels import lazy_prenorm as L  # noqa: E402

H, ER = C.HIDDEN_SIZE, C.N_ROUTED_EXPERTS
RESULTS = RESULTS_DIR              # honours $GLM52_RESULTS_DIR; C500's own live in results/c500
QDIR = RESULTS / "_f11b_queue"

SEED_GEMM = dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=3)

# --------------------------------------------------------------------------------------
# VRAM ceiling on T
# --------------------------------------------------------------------------------------
# Peak residency is `reference()`, not the timed chains: R.rmsnorm keeps h1.float() (4*H
# per token) and (xf*rstd) (4*H) live alongside the bf16 result (2*H), on top of the
# persistent h1 + x2 + logits (2*H + 2*H + 4*ER). 87040 B/token at H=6144, ER=256.
PEAK_B_PER_T = 4 * H + 4 * H + 2 * H + 2 * H + 2 * H + 4 * ER
# Room for the CUDA context, common.py's L2 flush buffer and allocator fragmentation.
VRAM_HEADROOM = 0.80


def max_safe_T() -> int:
    """Largest power-of-two T whose fp32 reference still fits this card's VRAM.

    Probed and enforced rather than left implicit, because the failure is silent-ish: the
    sweep tunes for minutes and then dies in `reference()`. On this 8 GB card it returns
    65536 (5.3 GiB at the peak; 131072 would want 10.6 GiB); the C500 ran it to 262144.
    Chunking `reference()` over T would buy one more octave if it is ever needed.
    """
    budget = torch.cuda.get_device_properties(0).total_memory * VRAM_HEADROOM
    t = 1
    while 2 * t * PEAK_B_PER_T <= budget:
        t *= 2
    return t


class RouterProblem:
    def __init__(self, T: int, g: torch.Generator):
        self.T = T
        dev, dt = "cuda", torch.bfloat16
        self.h1 = torch.empty(T, H, device=dev, dtype=dt).normal_(0, 0.1, generator=g)
        self.w = (torch.randn(H, generator=g, device=dev) * 0.1 + 1).to(dt)
        wg = torch.empty(ER, H, device=dev, dtype=dt).normal_(0, 0.02, generator=g)
        self.wg = wg
        self.b_kn = wg.t().contiguous()                       # [K, N] raw
        # offline weight transform: fold the RMSNorm affine into B's rows (load-time cost)
        self.b_kn_folded = (self.w[:, None].float() * self.b_kn.float()).to(dt).contiguous()
        self.x2 = torch.empty(T, H, device=dev, dtype=dt)
        self.logits = torch.empty(T, ER, device=dev, dtype=torch.float32)

    def unfused(self, cn, cg):
        return [lambda: NK.norm_only(self.h1, self.w, self.x2, cn),
                lambda: L.launch_router(self.x2, self.b_kn, self.logits, cg, fuse_norm=False)]

    def fused(self, cg):
        return [lambda: L.launch_router(self.h1, self.b_kn_folded, self.logits, cg,
                                        fuse_norm=True)]

    def reference(self):
        x2 = R.rmsnorm(self.h1, self.w)
        return torch.nn.functional.linear(x2.float(), self.wg.float())


def measure(T: int) -> dict:
    t_max = max_safe_T()
    assert T <= t_max, (f"T={T} needs {T * PEAK_B_PER_T / 2**30:.1f} GB in reference(); "
                        f"this card caps the sweep at T<={t_max}")
    g = torch.Generator(device="cuda").manual_seed(0)
    p = RouterProblem(T, g)
    tw, tr_, mw, mr = rep_budget(T)
    print(f"--- T={T} [tune {tw}/{tr_}, measure {mw}/{mr}, cap {t_max}] ---", flush=True)
    # Same admission rule as bench_f11's own router grid (Triton-3.6 SMEM model against the
    # per-block opt-in ceiling, plus an accumulator bound). The old guard tested SMEM only:
    # BM128/BN256/BK32/s2 passes SMEM but wants 256 fp32 accumulators/lane at warp 32, and
    # ptxas spills rather than erroring -- so autotune kept a spilling config. The fused arm
    # carries the sum-of-squares state and spills EARLIER than the unfused arm at the same
    # config, which is how a missing register bound walks the two arms apart.
    guard = lambda c: _ok(c, max_bn=256, max_bm=128)

    # The unfused RMSNorm gets F3's real mapping space, exhaustively. Coordinate descent
    # from the old BLOCK_N=8192 seed could only reach bench_layer's GEMM widths as well,
    # i.e. {8192, 32, 64, 128, 256}: 1024/2048/4096 were unreachable, so it was pinned to the
    # ONE_SHOT path at 64 fp32/lane on half C500's register file -- a spilling unfused
    # kernel inflates the fused win at every T, which is this sweep's entire deliverable.
    tn = autotune(lambda c: [p.unfused(c, SEED_GEMM)[0]], norm_grid(),
                  warmup=tw, rep=tr_)
    cn = tn.best_cfg
    print(f"  [cfgs] {f'rmsnorm T{T}':<16} {tn.best_ms:9.4f} ms  "
          f"({tn.n_tried} cfgs, {tn.n_failed} failed)  {cn}", flush=True)
    cg_u = _tile_then_coord(f"router unfused T{T}", lambda c: [p.unfused(cn, c)[1]],
                            dict(SEED_GEMM), guard=guard, warmup=tw, rep=tr_)
    cg_f = _tile_then_coord(f"router fused T{T}", lambda c: p.fused(c),
                            dict(SEED_GEMM), guard=guard, warmup=tw, rep=tr_)

    ref = p.reference()
    for fn in p.unfused(cn, cg_u):
        fn()
    torch.cuda.synchronize()
    err_u = rel_err(p.logits, ref)
    p.logits.zero_()
    for fn in p.fused(cg_f):
        fn()
    torch.cuda.synchronize()
    err_f = rel_err(p.logits, ref)
    assert max(err_u, err_f) < 5e-2, (err_u, err_f)

    us, fs, gains = [], [], []
    for _ in range(5):
        a = bench_chain(p.unfused(cn, cg_u), warmup=mw, rep=mr).p50_ms
        b = bench_chain(p.fused(cg_f), warmup=mw, rep=mr).p50_ms
        us.append(a); fs.append(b); gains.append(a / b)
    us_s, fs_s, g_s = sorted(us), sorted(fs), sorted(gains)
    t_norm = bench_chain([p.unfused(cn, cg_u)[0]], warmup=mw, rep=mr).p50_ms
    t_gemm = bench_chain([p.unfused(cn, cg_u)[1]], warmup=mw, rep=mr).p50_ms
    n_tiles = -(-ER // cg_f["BLOCK_N"])

    row = dict(T=T, unfused_ms=us_s[2], fused_ms=fs_s[2], gain=g_s[2],
               gain_min=g_s[0], gain_max=g_s[-1],
               gain_spread_pct=(g_s[-1] - g_s[0]) / g_s[2] * 100,
               rounds_unfused=us, rounds_fused=fs, rounds_gain=gains,
               norm_only_ms=t_norm, gemm_only_ms=t_gemm,
               n_tiles=n_tiles, sq_redundancy=n_tiles,
               norm_cfg=cn, unfused_cfg=cg_u, fused_cfg=cg_f,
               norm_grid_size=tn.n_tried, norm_grid_failed=tn.n_failed,
               rel_err_unfused=err_u, rel_err_fused=err_f,
               act_MB=T * H * 2 / 2**20,
               T_cap=t_max, ref_peak_MB=T * PEAK_B_PER_T / 2**20)
    print(f"  T={T:<8} unfused {us_s[2]:9.4f} (norm {t_norm:.4f} + gemm {t_gemm:.4f})  "
          f"fused {fs_s[2]:9.4f}  gain {g_s[2]:.4f}x  redundancy {n_tiles}x  "
          f"relerr u={err_u:.1e} f={err_f:.1e}", flush=True)
    del p
    torch.cuda.empty_cache()
    return row


# --------------------------------------------------------------------------------------
# dynamic work queue: claim by atomic rename, so three workers self-balance
# --------------------------------------------------------------------------------------
def make_queue(specs: list[str]) -> None:
    for d in ("pending", "running", "done"):
        (QDIR / d).mkdir(parents=True, exist_ok=True)
    for d in ("pending", "running", "done"):
        for f in (QDIR / d).glob("*.json"):
            f.unlink()
    n, t_max = 0, max_safe_T()
    for spec in specs:
        regime, ts = spec.split(":")
        for t in (int(x) for x in ts.split(",")):
            # Refuse here rather than 20 minutes into the worker: the fp32 reference is the
            # peak and it is not the thing under test, so an OOM costs a whole tuning run.
            if t > t_max:
                raise SystemExit(
                    f"T={t} needs {t * PEAK_B_PER_T / 2**30:.1f} GB in reference(); "
                    f"this card caps the sweep at T<={t_max}")
            # descending T in the name so lexical order roughly drains big jobs first
            name = f"{10**9 - t:012d}_{regime}_{t}.json"
            (QDIR / "pending" / name).write_text(json.dumps({"regime": regime, "T": t}))
            n += 1
    print(f"queued {n} tasks in {QDIR}/pending  (T cap {t_max} = "
          f"{t_max * PEAK_B_PER_T / 2**30:.1f} GB peak in reference())")


def run_worker(worker: str) -> None:
    done = 0
    while True:
        claimed = None
        for f in sorted((QDIR / "pending").glob("*.json")):
            try:
                dst = QDIR / "running" / f.name
                os.rename(f, dst)          # atomic: exactly one worker wins
                claimed = dst
                break
            except OSError:
                continue                   # another worker took it; try the next
        if claimed is None:
            print(f"[{worker}] queue empty, {done} tasks done", flush=True)
            return
        task = json.loads(claimed.read_text())
        print(f"[{worker}] claimed {task['regime']} T={task['T']}", flush=True)
        try:
            row = measure(task["T"])
            row["regime_kind"] = task["regime"]
            row["worker"] = worker
            out = RESULTS / f"f11b_{task['regime']}_T{task['T']}.json"
            out.write_text(json.dumps(row, indent=2, default=str))
            claimed.rename(QDIR / "done" / claimed.name)
            done += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[{worker}] FAILED {task}: {type(exc).__name__}: {exc}", flush=True)
            claimed.rename(QDIR / "pending" / (claimed.name + ".failed"))
            raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-queue", nargs="*", default=None,
                    help="specs like decode:1,2,4 prefill:256,512")
    ap.add_argument("--queue", action="store_true")
    ap.add_argument("--worker", default="w")
    args = ap.parse_args()
    if args.make_queue:
        make_queue(args.make_queue)
    elif args.queue:
        run_worker(args.worker)
    else:
        ap.error("pass --make-queue or --queue")


if __name__ == "__main__":
    main()
