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
expert).  We take option (ii): *all* K==6144 consumers are fused, so x2 is genuinely dead
and never written.  That is why the per-family rows are also rolled up into a `combined`
row, which is the only end-to-end-honest number:

    combined UNFUSED = [norm kernel] + [router GEMM] + [w13 grouped GEMM]
    combined FUSED   =                 [router GEMM fused] + [w13 GEMM fused]

Charging the single norm kernel to F11a and to F11b separately (as the per-family traffic
model does) double-counts it; the combined row does not.

**F11a runs by default here, unlike the 4060 port.**  Its two w13 buffers are 12.9 GB each;
on an 8 GB card the process died in `torch.empty` before a single F11b number existed.
H200 has 143 GB, so both families are in scope -- but the fast path still exists:
`--router-only` skips the whole w13 family (grid, timings, checks and result rows alike,
rather than emitting nulls), and if the free-memory probe says the weights do not fit, the
same skip happens automatically with the reason recorded.

**Why the H200 number here is not predictable from the two earlier ports, and what this
bench does about it.**  "Towards Free Normalization" relies on WARP SPECIALIZATION to put
the reduction on dedicated warps so it overlaps the MMA pipeline.  Neither C500 (Triton
3.0/MACA) nor sm_89 has it, and on both the reduction *displaced* MMA work instead of
hiding behind it.  Hopper does have it -- the H200 preflight compiled
`tl.range(..., warp_specialize=True)` -- so this is the first device in the study on which
the paper's mechanism can be tested at all.

That test is the headline experiment, and it is a 2x2 at ONE SHARED CONFIG
(`specialization_study`, reported per regime under `rows[].headline`):

    unfused                  FUSE_NORM=0  WARP_SPECIALIZE=0     the classic baseline
    fused, nonspecialized    FUSE_NORM=1  WARP_SPECIALIZE=0     what C500 and sm_89 measured
    fused, warp-specialized  FUSE_NORM=1  WARP_SPECIALIZE=1     the paper's configuration
    unfused, warp-specialized FUSE_NORM=0 WARP_SPECIALIZE=1     the control

The fourth arm is not padding.  Warp specialization speeds up a plain bf16 mainloop on its
own, and without measuring that, every microsecond it saves would be booked as a fusion win.
All four run on one kernel source, one tile, one launch shape and the same tensors -- only
the two constexprs move -- and all four are timed inside a single rotating interleave, so
thermal drift cancels the way it does in `bench_pair`.

Nothing here assumes the mechanism is reachable.  The specialized arms are timed only when
`kernels/lazy_prenorm.py` says its launcher can actually spell warp specialization on this
stack; otherwise they are omitted with the reason recorded, because
`resolve_warp_specialize` *refuses* an impossible request instead of raising -- and timing
the classic mainloop twice under a "warp-specialized" label would be a fabricated result.

Run:
    python3 glm52_h200/bench/bench_f11_lazy_prenorm.py --gpu auto [--router-only] [--regimes ...]

`--gpu` matters on this host: it has eight H200s and other tenants, and the preflight's own
launch/timer calibration was measured on a card someone else was already using.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import triton

from glm52_h200 import bench as B
from glm52_h200 import config as C
from glm52_h200 import reference as R
from glm52_h200.common import (
    bench_chain,
    check,
    main_guard,
    record,
    rel_err,
    speedup_row,
)
from glm52_h200.kernels import add_rmsnorm as NK
from glm52_h200.kernels import lazy_prenorm as K

RESULT_ID = "f11_lazy_prenorm"
H = C.HIDDEN_SIZE  # 6144
I = C.MOE_INTERMEDIATE_SIZE  # 2048
NW13 = C.W13_N  # 4096
E = C.N_ROUTED_EXPERTS  # 256
TOPK = C.NUM_EXPERTS_PER_TOK  # 8
EPS = C.RMS_NORM_EPS
DT = C.DTYPE
UNITS = ["f11b_router", "f11a_w13", "combined", "half_fused"]

_ENV = C.env()
SMEM_LIMIT = B.env_int(_ENV, "smem_bytes")  # per-block opt-in ceiling
WARP = B.env_int(_ENV, "warp_size")  # every per-lane guard below uses THIS, never a literal
MAX_THREADS = B.max_threads_per_block(_ENV)
WARPS = B.warp_ladder(_ENV)
ELEM_CAP = B.elems_per_program_cap(_ENV)
# Derived from THIS device's shared-memory ceiling: [16..256] on the H200's 232448 B,
# [16..128] on sm_89.  The measured Triton peak on this card is BM128/BN256/BK64/w8/s3
# (788 TF/s, 96 % of cuBLAS) and the grid has to be able to express it.
TILES = B.tile_ladder(_ENV)
BKS = B.bk_ladder(_ENV, hi=128)
ACC_CAP = B.MAX_ACC_ELEMS_PER_THREAD
#: Coarse-grid trial budget AFTER the sm_90 overlays multiply the space.
COARSE_CAP = 180


# ======================================================================================
# Mapping search spaces.  The SAME generator is used for the fused and the unfused side of
# each family -- `fused` is not even a parameter, because unlike F6 the lazy-prenorm kernel
# stages no extra tile and its SMEM footprint is identical.  So the two coarse grids are
# *literally the same config list*; only the refine neighbourhoods differ, because they are
# centred on each side's own coarse winner.
# ======================================================================================
def _ok(cfg: dict, max_bn: int, max_bm: int, acc_lo=2, acc_hi=ACC_CAP) -> bool:
    if cfg["BLOCK_N"] > max_bn or cfg["BLOCK_M"] > max_bm:
        return False
    # `B.smem_predict` fits the multi-buffer count to the preflight's own smem_probe
    # observations rather than assuming one (3.0 staged num_stages, 3.6/sm_89 stages
    # num_stages-1, and this H200 stack is back at num_stages -- all five observations
    # reproduce exactly).  The kernel module's own estimate is the 3.0 formula and
    # over-predicts, rejecting tiles that do fit.
    if B.smem_predict(cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"],
                      cfg["num_stages"]) > SMEM_LIMIT:
        return False
    threads = cfg["num_warps"] * WARP
    if threads > MAX_THREADS:
        return False
    acc_per_lane = cfg["BLOCK_M"] * cfg["BLOCK_N"] / threads
    return acc_lo <= acc_per_lane <= acc_hi


def router_grid(T: int) -> list[dict]:
    """Router GEMM: M=T, N=256, K=6144."""
    max_bm = max(16, 1 << (max(T, 1) - 1).bit_length())
    out = []
    for bm, bn, bk, w, s in itertools.product(
        [t for t in TILES if t <= 128], [t for t in TILES if t >= 32], BKS,
        [w for w in WARPS if w >= 4], (2, 3, 4)
    ):
        cfg = dict(
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, GROUP_M=8
        )
        if _ok(cfg, max_bn=256, max_bm=max_bm):
            out.append(cfg)
    # `K.ROUTER_AXES` is the kernel module's per-kernel axis advertisement: the router GEMM
    # may sweep clusters (its B is the 3 MB gate weight every CTA reads, the one place DSMEM
    # could plausibly pay) while the w13 launcher refuses them.  Fall back to the module
    # itself on a build that predates those objects.
    return B.widen(out, getattr(K, "ROUTER_AXES", K), cap=COARSE_CAP, tag=f"f11b/T{T}")


def moe_grid(big: bool) -> list[dict]:
    """w13 grouped GEMM: M=T*8 (padded), N=4096, K=6144.  Same shape rules F6 used."""
    if big:
        bms = [t for t in TILES if t >= 32]
    else:
        bms = [t for t in TILES if t <= 128]
    bns = [t for t in TILES if t >= 64]
    out = []
    for bm, bn, bk, w, s in itertools.product(
        bms, bns, BKS, [w for w in WARPS if w >= 4], (2, 3, 4)
    ):
        cfg = dict(
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, GROUP_M=8
        )
        if _ok(cfg, max_bn=4096, max_bm=4096, acc_lo=4):
            out.append(cfg)
    return B.widen(out, getattr(K, "MOE_AXES", K), cap=COARSE_CAP,
                   tag=f"f11a/{'big' if big else 'small'}")


def refine(best: dict, max_bn: int, max_bm: int, acc_lo=2) -> list[dict]:
    """Same neighbourhood rule for both sides: half/same/double in BM, BN, warps at the
    winning BK/stages; a BK x stages sweep at the winning shape; a GROUP_M sweep.

    Any sm_90 overlay keys on `best` (USE_TMA / WARP_SPECIALIZE / num_ctas) ride along
    unchanged, so a side whose coarse winner was warp-specialized refines a warp-specialized
    neighbourhood rather than falling back to the classic mainloop at the first refine step.
    """

    def nb(v, lo, hi):
        return sorted({max(lo, v // 2), v, min(hi, v * 2)})

    overlay = {kk: vv for kk, vv in best.items()
               if kk in ("USE_TMA", "TMA_A", "TMA_B", "TMA_MODE", "WARP_SPECIALIZE",
                         "warp_specialize", "num_consumer_groups",
                         "num_buffers_warp_spec", "num_ctas")}
    tile_hi = TILES[-1]
    cands = []
    for bm in nb(best["BLOCK_M"], TILES[0], tile_hi):
        for bn in nb(best["BLOCK_N"], 32, tile_hi):
            for w in nb(best["num_warps"], 2, WARPS[-1]):
                cands.append((bm, bn, best["BLOCK_K"], w, best["num_stages"], 8))
    for bk in nb(best["BLOCK_K"], BKS[0], BKS[-1]):
        for s in (2, 3, 4, 5):
            cands.append((best["BLOCK_M"], best["BLOCK_N"], bk, best["num_warps"], s, 8))
    for g in (1, 4, 8, 16):
        cands.append((best["BLOCK_M"], best["BLOCK_N"], best["BLOCK_K"],
                      best["num_warps"], best["num_stages"], g))
    out, seen = [], set()
    for bm, bn, bk, w, s, g in cands:
        cfg = dict(
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, GROUP_M=g,
            **overlay,
        )
        key = tuple(sorted((kk, str(vv)) for kk, vv in cfg.items()))
        if key in seen or not _ok(cfg, max_bn, max_bm, acc_lo=acc_lo):
            continue
        seen.add(key)
        out.append(cfg)
    return out


def rstd_grid() -> list[dict]:
    """Mapping space for the exploratory `rstd`-only reduction kernel."""
    out = []
    for b, r, w, s in itertools.product(
        (1024, 2048, 4096, 8192), (1, 2, 4, 8), WARPS, (1, 2)
    ):
        threads = w * WARP
        if threads > MAX_THREADS or b * r > ELEM_CAP:
            continue
        if not (2 <= b * r / threads <= B.MAX_ELEMS_PER_THREAD):
            continue
        if b >= H and s > 2:
            continue
        out.append(dict(ROWS=r, BLOCK_N=b, num_warps=w, num_stages=s))
    return out


def norm_grid() -> list[dict]:
    """F3's proven RMSNorm mapping space, handed to the unfused side as a bonus search.

    `threads` uses the REAL warp width.  With 64 baked in, this grid admitted configs at a
    true 128 fp32/lane -- a guaranteed spill in the UNFUSED arm's only kernel, which would
    manufacture a fused win at every regime.  And truncating the warp ladder at 16 searched
    130 of 164 legal configs on a 32-lane device: a 21 % one-sided truncation of a grid
    that feeds only the baseline.  Both bounds now come from the probe.
    """
    out = []
    for b, r, w, s in itertools.product(
        (512, 1024, 2048, 4096, 8192), (1, 2, 4, 8), WARPS, (1, 2)
    ):
        threads = w * WARP
        if threads > MAX_THREADS or b * r > ELEM_CAP:
            continue
        epr = b * r / threads
        if epr < 2 or epr > B.MAX_ELEMS_PER_THREAD:
            continue
        if b >= H and s > 2:
            continue
        out.append(
            dict(ROWS=r, BLOCK_N=b, num_warps=w, num_stages=s, grid_cap=None, eps=EPS)
        )
    return out


# ======================================================================================
# Weights (allocated once for the whole run)
# ======================================================================================
def make_w13(w_norm: torch.Tensor):
    """w13 [E, 2I, H] plus its `w`-folded twin, both with an sglang-style trailing pad.

    Triton's software pipeline issues speculative (unpredicated) B-tile loads for the
    peeled prologue/epilogue stages, so the last expert's tile can be fetched one BLOCK_K
    past the end of the tensor.  Both tensors get the pad; it changes no arithmetic and
    favours neither side.
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
        self.has_w13 = w13_raw is not None

        # x2 -- the materialized intermediate the unfused side needs.  Seeded with the fp32
        # reference so the unfused GEMM has valid input during tuning; the real unfused
        # chain overwrites it with the Triton norm kernel every iteration.
        self.x2 = R.rmsnorm(self.h1, w_norm, EPS).contiguous()

        _, _, self.topk_ids = R.router(self.x2, gate)

        self.logits_f = torch.zeros(T, E, device="cuda", dtype=torch.float32)
        self.logits_u = torch.zeros(T, E, device="cuda", dtype=torch.float32)
        self.logits_h = torch.zeros(T, E, device="cuda", dtype=torch.float32)
        # F11a-only state: three [T*8, 4096] bf16 outputs, 512 MB EACH at T=8192, plus the
        # moe_align layout cache.  Never touched under --router-only.
        if self.has_w13:
            self.layouts: dict[int, tuple] = {}
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

    # ---- exploratory "half-fused" variant: rstd from a tiny reduction kernel, applied as
    # a pure epilogue scale; the GEMM k-loop is byte-for-byte the unfused one ----------
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

    def moe_unfused_same(self, cfg):
        """The unfused GEMM at an ARBITRARY config (for the isolation measurement)."""
        sti, eids, ntp = self.layout(cfg["BLOCK_M"])
        return lambda: K.launch_moe_gateup(
            self.x2, self.w13_raw, self.c_u, sti, eids, ntp, self.rows, TOPK,
            cfg, False, EPS,
        )

    def router_unfused_same(self, cfg):
        """The router GEMM at an ARBITRARY config with FUSE_NORM off (isolation control).

        Reads `h1` and the FOLDED weight -- the same two tensors the fused arm reads -- so
        the byte traffic, the tile shapes and the launch are identical and the only
        difference in the k-loop is the sum-of-squares.  Its output is numerically wrong
        (no rstd is ever applied) and is neither checked nor reported: this arm measures
        instructions, not answers, which is exactly what makes it the right control.
        """
        return lambda: K.launch_router(
            self.h1, self.b_fold, self.logits_f, cfg, False, EPS
        )


def reference_rows(prob: Problem, n_sample: int = 1024):
    """fp32 reference for the w13 GEMM on a sampled row subset (a full fp32 reference at
    T=8192 would be 3.3 TFLOP).  The same rows judge both arms."""
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
    """Vendor-BLAS grouped GEMM: rows pre-gathered per expert OUTSIDE the timed region, so
    this is the best case for the vendor path (pure per-expert torch.matmul)."""
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


# ======================================================================================
# Warp specialization: availability, and the 2x2 that is this campaign's headline
# ======================================================================================
#: Every spelling `kernels/lazy_prenorm.py::cfg_warp_specialize` recognises, and the one
#: this bench writes.  It reaches the compiler as a kernel CONSTEXPR
#: (`tl.range(..., warp_specialize=WS)` is lowered at compile time), not as a launch kwarg
#: -- the launch-kwarg spelling (`num_consumer_groups`) is rejected outright by the measured
#: H200 stack with `KeyError: ... unrecognised`.
WS_CFG_KEYS = ("WARP_SPECIALIZE", "WS", "warp_specialize")
WS_CFG_KEY = "WARP_SPECIALIZE"


def ws_offered() -> tuple:
    """`(available, evidence)` for warp specialization, from the kernel module first.

    `kernels/lazy_prenorm.py` owns the verdict: it knows both whether the capability layer
    says the mechanism exists AND whether its own launcher can spell it.  Falling back to
    `B.axis_available` covers the case where that module has not (yet) grown the helper.

    This gate matters more than it looks.  `resolve_warp_specialize` REFUSES an impossible
    request rather than raising -- it warns and runs the classic mainloop.  So an ungated
    `WARP_SPECIALIZE=True` arm on a stack without the feature would time the classic loop
    twice and report the difference as a warp-specialization effect.  That is a fabricated
    result, and it is the specific fabrication `tl.range(warp_specialize=True)` invites,
    because on sm_89 it compiles, runs, and silently is not the Hopper scheme.
    """
    fn = getattr(K, "warp_specialize_available", None)
    if callable(fn):
        try:
            ok = bool(fn())
            mode = getattr(K, "_ws_mode", lambda: "?")()
            return ok, f"kernels.lazy_prenorm.warp_specialize_available()={ok} (mode {mode})"
        except Exception:  # noqa: BLE001 -- fall through to the harness verdict
            pass
    return B.axis_available("warp_specialize")


def _ws(cfg: dict, on: bool) -> dict:
    """`cfg` with warp specialization pinned on or off.

    EVERY recognised spelling is stripped first, then exactly one is written.  The tuned
    winner reaching this function may already carry `warp_specialize=True` from the coarse
    grid's sm_90 overlay, and leaving that in place would make the arms depend on which key
    the kernel module happens to check first.  Pinning explicitly on BOTH sides matters for
    the same reason: an unset flag means the module's own default decides, and an
    auto-selected arm cannot be the control for a forced one.
    """
    out = {kk: vv for kk, vv in cfg.items() if kk not in WS_CFG_KEYS}
    out[WS_CFG_KEY] = bool(on)
    return out


def specialization_study(tag: str, cfg: dict, mk_fused, mk_unfused, sq_mode: int,
                         w_f: int, r_f: int) -> dict:
    """ONE config, one kernel source, four launches: {FUSE_NORM on/off} x {WS on/off}.

    This is the campaign's headline experiment, and the reason is a claim it can falsify.
    "Towards Free Normalization" argues an RMSNorm reduction is free inside a GEMM because
    WARP SPECIALIZATION puts it on producer warps that are not doing the MMA.  Neither C500
    (Triton 3.0 / MACA) nor sm_89 has the mechanism, and on both the reduction DISPLACED
    MMA work: the C500 study measured the fused arm slower at every regime.  The H200 has
    it, and this suite's whole reason for existing on this device is to find out whether
    that changes the sign.

    Three arms answer the question as posed -- unfused, fused-nonspecialized,
    fused-warp-specialized -- and the fourth (unfused-warp-specialized) is what keeps the
    answer honest: warp specialization speeds up a plain bf16 mainloop too, and without the
    control every microsecond it saves would be credited to the fusion.  All four run at the
    SAME tile, the same warps, the same stages, the same tensors and the same launch, so the
    only differences are the two constexprs.

    They are timed by `B.bench_multi`, one round each with a rotating order, for the same
    reason the pairs are interleaved: a 22 % thermal drift inside one run once produced a
    4060 speedup above the cell's own physical ceiling, and per-round ratios cancel it.
    """
    ws_ok, ws_why = ws_offered()
    chains = {
        "unfused": [mk_unfused(_ws(cfg, False))],
        "fused_ws_off": [mk_fused(_ws(cfg, False), sq_mode)],
    }
    if ws_ok:
        chains["fused_ws_on"] = [mk_fused(_ws(cfg, True), sq_mode)]
        chains["unfused_ws_on"] = [mk_unfused(_ws(cfg, True))]

    tim, meta = B.bench_multi(chains, w_f, r_f, baseline="unfused")
    ms = {k: v.p50_ms for k, v in tim.items()}
    out = {
        "config": cfg,
        "sq_mode": sq_mode,
        "warp_specialize_available": ws_ok,
        "warp_specialize_evidence": ws_why,
        "ms": ms,
        "timings": {k: v.as_dict() for k, v in tim.items()},
        "pair_meta": meta,
        # The C500/Ada number: what fusing the reduction costs with a classic mainloop.
        "instruction_cost_pct": 100.0 * (ms["fused_ws_off"] / ms["unfused"] - 1.0),
        "interleaved": True,
    }
    if not ws_ok:
        out["warp_specialize_skipped"] = (
            "no warp-specialized arm was timed: " + ws_why + ". The fused/unfused numbers "
            "above are the classic-mainloop result, directly comparable with C500 and "
            "sm_89, and the paper's mechanism is simply not present to test."
        )
        return out
    out.update(
        {
            # Like-for-like: what fusion costs once BOTH arms are specialized. This is the
            # number the paper's claim is actually about.
            "instruction_cost_ws_pct":
                100.0 * (ms["fused_ws_on"] / ms["unfused_ws_on"] - 1.0),
            # What a layer adopting both at once would pay against today's classic baseline.
            "fused_ws_on_vs_classic_unfused_pct":
                100.0 * (ms["fused_ws_on"] / ms["unfused"] - 1.0),
            # What specialization bought each arm on its own.
            "ws_gain_fused_pct": 100.0 * (1.0 - ms["fused_ws_on"] / ms["fused_ws_off"]),
            "ws_gain_unfused_pct": 100.0 * (1.0 - ms["unfused_ws_on"] / ms["unfused"]),
        }
    )
    # The headline sentence, computed rather than narrated, so it cannot drift from the
    # numbers it describes.
    cost_classic = out["instruction_cost_pct"]
    cost_ws = out["instruction_cost_ws_pct"]
    out["verdict"] = (
        f"[{tag}] fusing the reduction costs {cost_classic:+.2f}% with a classic mainloop "
        f"and {cost_ws:+.2f}% with warp specialization; specialization itself moved the "
        f"fused arm {out['ws_gain_fused_pct']:+.2f}% and the unfused arm "
        f"{out['ws_gain_unfused_pct']:+.2f}%. "
        + (
            "Specialization absorbs the reduction: the fusion is cheaper once the producer "
            "warps carry it."
            if cost_ws < cost_classic - 0.5 else
            "Specialization does NOT absorb the reduction here -- the fused arm pays as "
            "much or more with it as without, so the k-loop, not the memory traffic, is the "
            "binding cost on this device too."
        )
    )
    # Whether the compiler really emitted the producer/consumer scheme, not just whether the
    # flag was accepted. A row labelled warp-specialized that ran the classic loop is worse
    # than no row.
    for helper in ("caps_report", "ws_evidence"):
        fn = getattr(K, helper, None)
        if callable(fn):
            try:
                out[f"kernel_{helper}"] = fn() if helper == "caps_report" else None
            except Exception as exc:  # noqa: BLE001 -- provenance is a bonus, never fatal
                out[f"kernel_{helper}"] = f"{type(exc).__name__}: {exc}"[:160]
    return out


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
    fams = [("router", cfgs_r, prob.router_fused)]
    if prob.has_w13:
        fams.append(("moe", cfgs_m, prob.moe_fused))
    for tag, cfgs, mk in fams:
        times: dict[int, dict[int, float]] = {m: {} for m in modes}
        for ci, cfg in enumerate(cfgs):
            for m in modes:
                try:
                    t = bench_chain([mk(cfg, m)], 4, 12).p50_ms
                    tab.append((tag, cfg, m, t, None))
                    times[m][ci] = t
                except Exception as exc:  # noqa: BLE001
                    tab.append((tag, cfg, m, None, str(exc)[:120]))
        # compare only over configs where EVERY mode compiled, so a mode is never rewarded
        # for having failed on the slow shapes
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
def run_regime(prob: Problem, sq_mode: dict, quick: bool, fair: B.Fairness) -> tuple:
    reg = prob.regime
    T = reg.T
    big = T >= 2048
    w13 = prob.has_w13  # False under --router-only: no F11a grid, timing, check or row
    w_t, r_t, w_f, r_f = B.reps(T, quick)

    tag_w13 = f", moe rows={prob.rows}" if w13 else ", F11b only"
    print(f"\n===== {reg.name} (T={T}{tag_w13}) =====", flush=True)
    tables: dict = {}

    ref_router = prob.x2.float() @ prob.gate.float().t()
    if w13:
        idx, ref_moe = reference_rows(prob)

    # ---- verifiers.  Every config is checked against the fp32 reference before it is
    # allowed into a timing grid; on this family a wrong-but-fast config would be a
    # published speedup, and nobody can re-run it here.
    def v_norm():
        c = check(prob.x2_out, prob.x2, label="norm")
        return c["ok"], f"rel_err={c['rel_err']:.2e}"

    def v_rt_f():
        c = check(prob.logits_f, ref_router, label="router_fused")
        return c["ok"], f"rel_err={c['rel_err']:.2e}"

    def v_rt_u():
        c = check(prob.logits_u, ref_router, label="router_unfused")
        return c["ok"], f"rel_err={c['rel_err']:.2e}"

    def v_moe_f():
        c = check(prob.c_f[idx], ref_moe, label="moe_fused")
        return c["ok"], f"rel_err={c['rel_err']:.2e}"

    def v_moe_u():
        c = check(prob.c_u[idx], ref_moe, label="moe_unfused")
        return c["ok"], f"rel_err={c['rel_err']:.2e}"

    # ---------------------------------------------------------------- norm kernel ----
    ng = norm_grid()
    if quick:
        ng = B.quick_slice(ng, 14)
    tn = B.screened_autotune("norm", prob.norm_fn, ng, v_norm, w_t, r_t)
    fair.add(reg.name, "unfused_norm", "tune", tn)
    print(f"  norm: {tn.n_tried} cfgs -> {tn.best_ms:.4f} ms {tn.best_cfg}", flush=True)
    tables["norm"] = tn.as_dict()
    norm_cfg = tn.best_cfg

    # =============================== F11b : ROUTER ===================================
    rg = router_grid(T)
    if quick:
        rg = B.quick_slice(rg, 14)
    max_bm = max(16, 1 << (max(T, 1) - 1).bit_length())
    tf_c = B.screened_autotune(
        "routerF/coarse", lambda c: [prob.router_fused(c, sq_mode["router"])], rg,
        v_rt_f, w_t, r_t,
    )
    tu_c = B.screened_autotune(
        "routerU/coarse", lambda c: [prob.router_unfused(c)], rg, v_rt_u, w_t, r_t
    )
    rf = refine(tf_c.best_cfg, 256, max_bm)
    ru = refine(tu_c.best_cfg, 256, max_bm)
    tf_r = B.screened_autotune(
        "routerF/refine", lambda c: [prob.router_fused(c, sq_mode["router"])], rf,
        v_rt_f, w_t, r_t,
    )
    tu_r = B.screened_autotune(
        "routerU/refine", lambda c: [prob.router_unfused(c)], ru, v_rt_u, w_t, r_t
    )
    for arm, (tc, tr, rgrid) in (("router_fused", (tf_c, tf_r, rf)),
                                 ("router_unfused", (tu_c, tu_r, ru))):
        # `grid=` records live per-axis counts. The coarse list is the SAME object for both
        # arms, so their USE_TMA / WARP_SPECIALIZE / num_ctas counts must match exactly; the
        # refine lists differ only because each is centred on that arm's own winner.
        fair.add(reg.name, arm, "coarse", tc, grid=rg)
        fair.add(reg.name, arm, "refine", tr, grid=rgrid)
    rt_f_cfg = tf_c.best_cfg if tf_c.best_ms <= tf_r.best_ms else tf_r.best_cfg
    rt_u_cfg = tu_c.best_cfg if tu_c.best_ms <= tu_r.best_ms else tu_r.best_cfg
    print(
        f"  router fused  : coarse {tf_c.n_tried}({tf_c.n_failed}f) + refine "
        f"{tf_r.n_tried}({tf_r.n_failed}f) -> {min(tf_c.best_ms, tf_r.best_ms):.4f} ms "
        f"{rt_f_cfg}", flush=True,
    )
    print(
        f"  router unfused: coarse {tu_c.n_tried}({tu_c.n_failed}f) + refine "
        f"{tu_r.n_tried}({tu_r.n_failed}f) -> {min(tu_c.best_ms, tu_r.best_ms):.4f} ms "
        f"{rt_u_cfg}", flush=True,
    )
    # joint chain re-tune, in the unfused side's favour
    joint_r, best_r, best_pair_r = [], float("inf"), None
    for gc in B.top_cfgs(tu_c, tu_r, k=3):
        for nc in B.top_cfgs(tn, k=3):
            try:
                t = bench_chain(
                    [prob.norm_fn(nc), prob.router_unfused(gc, prob.x2_out)], w_t, r_t
                )
                joint_r.append(({"gemm": gc, "norm": nc}, t.p50_ms, None))
                if t.p50_ms < best_r:
                    best_r, best_pair_r = t.p50_ms, (gc, nc)
            except Exception as exc:  # noqa: BLE001
                joint_r.append(({"gemm": gc, "norm": nc}, None, str(exc)[:160]))
    if best_pair_r is None:
        raise RuntimeError(f"{reg.name}: no unfused router chain combination ran")
    rt_u_gemm, rt_u_norm = best_pair_r
    fair.add(reg.name, "router_unfused_chain", "joint", size=len(joint_r))
    tables["router_fused"] = {"coarse": tf_c.as_dict(), "refine": tf_r.as_dict()}
    tables["router_unfused"] = {"coarse": tu_c.as_dict(), "refine": tu_r.as_dict()}
    tables["router_unfused_joint"] = joint_r

    # =============================== F11a : w13 =======================================
    if w13:
        mg = moe_grid(big)
        if quick:
            mg = B.quick_slice(mg, 12)
        mf_c = B.screened_autotune(
            "w13F/coarse", lambda c: [prob.moe_fused(c, sq_mode["moe"])], mg,
            v_moe_f, w_t, r_t,
        )
        mu_c = B.screened_autotune(
            "w13U/coarse", lambda c: [prob.moe_unfused(c)], mg, v_moe_u, w_t, r_t
        )
        mrf = refine(mf_c.best_cfg, 4096, 4096, acc_lo=4)
        mru = refine(mu_c.best_cfg, 4096, 4096, acc_lo=4)
        mf_r = B.screened_autotune(
            "w13F/refine", lambda c: [prob.moe_fused(c, sq_mode["moe"])], mrf,
            v_moe_f, w_t, r_t,
        )
        mu_r = B.screened_autotune(
            "w13U/refine", lambda c: [prob.moe_unfused(c)], mru, v_moe_u, w_t, r_t
        )
        for arm, (tc, tr, rgrid) in (("w13_fused", (mf_c, mf_r, mrf)),
                                     ("w13_unfused", (mu_c, mu_r, mru))):
            fair.add(reg.name, arm, "coarse", tc, grid=mg)
            fair.add(reg.name, arm, "refine", tr, grid=rgrid)
        mo_f_cfg = mf_c.best_cfg if mf_c.best_ms <= mf_r.best_ms else mf_r.best_cfg
        mo_u_cfg = mu_c.best_cfg if mu_c.best_ms <= mu_r.best_ms else mu_r.best_cfg
        print(
            f"  w13 fused  : coarse {mf_c.n_tried}({mf_c.n_failed}f) + refine "
            f"{mf_r.n_tried}({mf_r.n_failed}f) -> "
            f"{min(mf_c.best_ms, mf_r.best_ms):.4f} ms {mo_f_cfg}", flush=True,
        )
        print(
            f"  w13 unfused: coarse {mu_c.n_tried}({mu_c.n_failed}f) + refine "
            f"{mu_r.n_tried}({mu_r.n_failed}f) -> "
            f"{min(mu_c.best_ms, mu_r.best_ms):.4f} ms {mo_u_cfg}", flush=True,
        )
        joint_m, best_m, best_pair_m = [], float("inf"), None
        for gc in B.top_cfgs(mu_c, mu_r, k=3):
            for nc in B.top_cfgs(tn, k=2):
                try:
                    t = bench_chain(
                        [prob.norm_fn(nc), prob.moe_unfused(gc, prob.x2_out)], w_t, r_t
                    )
                    joint_m.append(({"gemm": gc, "norm": nc}, t.p50_ms, None))
                    if t.p50_ms < best_m:
                        best_m, best_pair_m = t.p50_ms, (gc, nc)
                except Exception as exc:  # noqa: BLE001
                    joint_m.append(({"gemm": gc, "norm": nc}, None, str(exc)[:160]))
        if best_pair_m is None:
            raise RuntimeError(f"{reg.name}: no unfused w13 chain combination ran")
        mo_u_gemm, mo_u_norm = best_pair_m
        fair.add(reg.name, "w13_unfused_chain", "joint", size=len(joint_m))
        tables["moe_fused"] = {"coarse": mf_c.as_dict(), "refine": mf_r.as_dict()}
        tables["moe_unfused"] = {"coarse": mu_c.as_dict(), "refine": mu_r.as_dict()}
        tables["moe_unfused_joint"] = joint_m

    # ============ EXPLORATORY: "half-fused" (rstd kernel + epilogue scale) ============
    # Not part of the fused-vs-unfused headline.  It is the third point on the design axis:
    # 2/3 of the byte saving, with a k-loop identical to the unfused GEMM.  On C500 it beat
    # full Lazy Pre-Norm at every regime, which is the study's cleanest statement that the
    # cost is the k-loop, not the bytes.
    rsg = rstd_grid()
    if quick:
        rsg = B.quick_slice(rsg, 10)
    tr_rstd = B.screened_autotune(
        "rstd", prob.rstd_fn, rsg,
        lambda: (True, "no independent reference; validated via router_half below"),
        w_t, r_t,
    )
    half = {}
    half_fams = [("router", prob.router_half, B.top_cfgs(tu_c, tu_r, tf_c, k=3))]
    if w13:
        half_fams.append(("moe", prob.moe_half, B.top_cfgs(mu_c, mu_r, mf_c, k=3)))
    for tag, mk, gcfgs in half_fams:
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
    print(f"  rstd kernel: {tr_rstd.n_tried} cfgs -> {tr_rstd.best_ms:.4f} ms "
          f"{tr_rstd.best_cfg}", flush=True)

    # ================================ validate =======================================
    prob.logits_f.zero_(); prob.logits_u.zero_(); prob.x2_out.zero_()
    prob.norm_fn(rt_u_norm)()
    prob.router_fused(rt_f_cfg, sq_mode["router"])()
    prob.router_unfused(rt_u_gemm, prob.x2_out)()
    if w13:
        prob.c_f.zero_(); prob.c_u.zero_()
        prob.moe_fused(mo_f_cfg, sq_mode["moe"])()
        prob.moe_unfused(mo_u_gemm, prob.x2_out)()
    prob.rstd_fn(tr_rstd.best_cfg)()
    prob.router_half(half["router"]["cfg"])()
    if w13:
        prob.moe_half(half["moe"]["cfg"])()
    torch.cuda.synchronize()

    chk = {}
    chk["x2"] = check(prob.x2_out, prob.x2, label="norm_kernel_x2")
    # router reference: (a) the framework path (bf16 x2 then fp32 matmul) and
    #                   (b) the *exact* fp32 path (no bf16 rounding of x2 at all)
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
    must = ["x2", "router_fused", "router_unfused", "router_half"]
    if w13:
        chk["moe_fused"] = check(prob.c_f[idx], ref_moe, label="moe_fused")
        chk["moe_unfused"] = check(prob.c_u[idx], ref_moe, label="moe_unfused")
        chk["moe_half"] = check(prob.c_h[idx], ref_moe, label="moe_half_fused")
        must += ["moe_fused", "moe_unfused", "moe_half"]
    for kk in must:
        if not chk[kk]["ok"]:
            raise RuntimeError(f"validation failed at {reg.name}: {kk} {chk[kk]}")
    w13_err = (
        f"w13 f={chk['moe_fused']['rel_err']:.2e} u={chk['moe_unfused']['rel_err']:.2e} | "
        if w13 else ""
    )
    print(
        f"  rel_err  router f={chk['router_fused']['rel_err']:.2e} "
        f"u={chk['router_unfused']['rel_err']:.2e} | " + w13_err
        + f"topk agree {chk['topk_id_agreement'] * 100:.2f}%", flush=True,
    )

    # ================================ final timing ===================================
    # Headline pairs are INTERLEAVED; the diagnostics beside them are not, and are labelled
    # as such (they are components, not ratios).
    t_rt_f, t_rt_u, pair_rt = B.bench_pair(
        [prob.router_fused(rt_f_cfg, sq_mode["router"])],
        [prob.norm_fn(rt_u_norm), prob.router_unfused(rt_u_gemm, prob.x2_out)],
        w_f, r_f, label=f"{reg.name}/f11b",
    )
    t_norm = bench_chain([prob.norm_fn(norm_cfg)], w_f, r_f)
    t_rt_gemm = bench_chain([prob.router_unfused(rt_u_gemm, prob.x2_out)], w_f, r_f)

    if w13:
        t_mo_f, t_mo_u, pair_mo = B.bench_pair(
            [prob.moe_fused(mo_f_cfg, sq_mode["moe"])],
            [prob.norm_fn(mo_u_norm), prob.moe_unfused(mo_u_gemm, prob.x2_out)],
            w_f, r_f, label=f"{reg.name}/f11a",
        )
        t_mo_gemm = bench_chain([prob.moe_unfused(mo_u_gemm, prob.x2_out)], w_f, r_f)
        # combined end-to-end: ONE norm kernel serves both consumers.  With only one
        # consumer left it degenerates to f11b_router, so it is not measured or reported.
        t_comb_f, t_comb_u, pair_comb = B.bench_pair(
            [prob.router_fused(rt_f_cfg, sq_mode["router"]),
             prob.moe_fused(mo_f_cfg, sq_mode["moe"])],
            [prob.norm_fn(norm_cfg),
             prob.router_unfused(rt_u_gemm, prob.x2_out),
             prob.moe_unfused(mo_u_gemm, prob.x2_out)],
            w_f, r_f, label=f"{reg.name}/combined",
        )

    t_rstd = bench_chain([prob.rstd_fn(tr_rstd.best_cfg)], w_f, r_f)
    t_rt_h = bench_chain(
        [prob.rstd_fn(tr_rstd.best_cfg), prob.router_half(half["router"]["cfg"])], w_f, r_f
    )
    if w13:
        t_mo_h = bench_chain(
            [prob.rstd_fn(tr_rstd.best_cfg), prob.moe_half(half["moe"]["cfg"])], w_f, r_f
        )
        t_comb_h = bench_chain(
            [prob.rstd_fn(tr_rstd.best_cfg), prob.router_half(half["router"]["cfg"]),
             prob.moe_half(half["moe"]["cfg"])], w_f, r_f,
        )

    # ---- ISOLATION: same config, same buffers, FUSE_NORM on vs off, WARP_SPECIALIZE on vs
    # off.  There is NO extra input tensor in this fusion, so this is a pure
    # instruction-cost measurement -- the single number that carried the C500-vs-Ada result,
    # and the one that says whether Hopper's warp specialization actually hides the
    # reduction.  All four arms are timed inside one rotating interleave; see
    # `specialization_study`.
    iso = {}
    iso_fams = [("router", rt_f_cfg, prob.router_fused, prob.router_unfused_same)]
    if w13:
        iso_fams.append(("moe", mo_f_cfg, prob.moe_fused, prob.moe_unfused_same))
    for tag, cfg, mk_f, mk_u in iso_fams:
        # The warp-specialized arm is a constexpr pairing introduced AFTER tuning, so
        # `screen()` never saw it: WARP_SPECIALIZE=True is applied at the tuned winner's
        # tile, and Triton's warp-specialize transform has preconditions the winner need not
        # satisfy (the preflight probed it at num_warps=4; these winners run num_warps=8,
        # num_stages=3, and a second warp group costs registers and SMEM).
        #
        # Letting that raise would propagate out of run_regime and skip ckpt_save, throwing
        # away every tuning result already computed for this regime -- hours of work lost to
        # the one arm that is allowed to fail. A failed specialization study is a RESULT
        # ("warp specialization does not compile at the tuned mapping"), not a fatal error.
        try:
            iso[tag] = specialization_study(
                f"{reg.name}/{tag}", cfg, mk_f, mk_u, sq_mode[tag], w_f, r_f
            )
        except Exception as exc:  # noqa: BLE001
            iso[tag] = {
                "failed": f"{type(exc).__name__}: {exc}"[:400],
                "config": dict(cfg) if isinstance(cfg, dict) else repr(cfg),
                "note": "specialization study aborted; tuning results for this regime kept",
            }
            print(f"  !! specialization_study[{tag}] failed: "
                  f"{type(exc).__name__}: {str(exc)[:160]}", flush=True)
            continue
        print(f"  {iso[tag].get('verdict') or iso[tag].get('warp_specialize_skipped')}",
              flush=True)

    # ---- register / SMEM report, cache cleared between compiles --------------------
    rk = getattr(K, "router_gemm_kernel", None)
    mk_ = getattr(K, "moe_gateup_prenorm_kernel", None)
    regs = {
        "router_fused": B.kernel_stats(
            prob.router_fused(rt_f_cfg, sq_mode["router"]), rk),
        "router_unfused": B.kernel_stats(
            prob.router_unfused(rt_u_gemm, prob.x2_out), rk),
        "router_unfused_at_fused_cfg": B.kernel_stats(
            lambda: K.launch_router(prob.x2, prob.b_raw, prob.logits_u, rt_f_cfg,
                                    False, EPS), rk),
    }
    if w13:
        regs["moe_fused"] = B.kernel_stats(prob.moe_fused(mo_f_cfg, sq_mode["moe"]), mk_)
        regs["moe_unfused"] = B.kernel_stats(
            prob.moe_unfused(mo_u_gemm, prob.x2_out), mk_)
        regs["moe_unfused_at_fused_cfg"] = B.kernel_stats(
            prob.moe_unfused_same(mo_f_cfg), mk_)

    # ---- vendor BLAS reference lines ----------------------------------------------
    t_blas_router = bench_chain([lambda: torch.matmul(prob.x2, prob.b_raw)], w_f, r_f)
    t_blas_router_fp32 = bench_chain(
        [lambda: torch.matmul(prob.x2.float(), prob.b_raw.float())],
        max(2, w_f // 3), max(5, r_f // 3),
    )
    if w13:
        t_blas_moe = bench_chain(vendor_moe_chain(prob), max(2, w_f // 3),
                                 max(5, r_f // 3))
        t_blas_moe_dense = bench_chain(
            [lambda: torch.matmul(prob.x2, prob.w13_raw[0].t())],
            max(2, w_f // 3), max(5, r_f // 3),
        )

    # ---- redundancy / traffic bookkeeping ------------------------------------------
    rt_ntiles = triton.cdiv(E, rt_f_cfg["BLOCK_N"])
    act = T * H * 2
    f_router = 2.0 * T * H * E
    if w13:
        mo_ntiles = triton.cdiv(NW13, mo_f_cfg["BLOCK_N"])
        f_moe = 2.0 * prob.rows * H * NW13
    tmodel = B.traffic_ceilings(reg)

    # Every F11a-derived key is OMITTED under --router-only rather than emitted as null: a
    # null in a speedup column silently corrupts the comparison table downstream.
    row = {
        "regime": reg.name,
        "T": T,
        "family": "f11a+f11b" if w13 else "f11b_only",
        "f11b_router": speedup_row(reg.name, t_rt_f, t_rt_u, {
            "fused_cfg": rt_f_cfg,
            "unfused_gemm_cfg": rt_u_gemm,
            "unfused_norm_cfg": rt_u_norm,
            "paired_speedup": pair_rt.get("paired_speedup_p50"),
            "paired_speedup_trimmed": pair_rt.get("paired_speedup_trimmed_mean"),
            "pair_meta": pair_rt,
            "tick": B.tick_report(t_rt_f.p50_ms, t_rt_u.p50_ms),
            "unfused_gemm_only_ms": t_rt_gemm.p50_ms,
            "norm_only_ms": t_norm.p50_ms,
            "ceiling": (tmodel.get("F11b_prenorm_router") or {}).get("roofline_ceiling"),
            "rel_err": chk["router_fused"]["rel_err"],
            "rel_err_unfused": chk["router_unfused"]["rel_err"],
            "n_tiles": rt_ntiles,
            "sq_redundancy": rt_ntiles,
            "extra_sq_flops_frac": rt_ntiles / E,
            "fused_tflops": f_router / (t_rt_f.p50_ms * 1e-3) / 1e12,
            "vendor_blas_bf16_ms": t_blas_router.p50_ms,
            "vendor_blas_fp32_ms": t_blas_router_fp32.p50_ms,
            "bytes_fused": act + H * E * 2 + T * E * 4,
            "bytes_unfused": 2 * act + act + H * E * 2 + T * E * 4,
        }),
        "half_fused": {
            "note": "EXPLORATORY, not the headline: rstd from a tiny reduction kernel, "
                    "applied as a pure epilogue scale. 2 activation passes vs the unfused "
                    "side's 3 and the fused side's 1; GEMM k-loop identical to unfused.",
            "rstd_cfg": tr_rstd.best_cfg,
            "rstd_only_ms": t_rstd.p50_ms,
            "router_cfg": half["router"]["cfg"],
            "router_ms": t_rt_h.p50_ms,
            "router_speedup_vs_unfused": t_rt_u.p50_ms / t_rt_h.p50_ms,
            "rel_err_router": chk["router_half"]["rel_err"],
        },
        # The 2x2 at one shared config: {FUSE_NORM on/off} x {WARP_SPECIALIZE on/off}, all
        # arms interleaved in one rotating loop. The key name predates the warp-specialized
        # arms and is kept so existing readers do not break; `headline` below is the flat
        # three-number summary the campaign is actually about.
        "isolation_fuse_on_vs_off_same_cfg": iso,
        "headline": {
            fam: {
                "unfused_ms": s["ms"].get("unfused"),
                "fused_nonspecialized_ms": s["ms"].get("fused_ws_off"),
                "fused_warp_specialized_ms": s["ms"].get("fused_ws_on"),
                "unfused_warp_specialized_ms": s["ms"].get("unfused_ws_on"),
                "shared_config": s["config"],
                "instruction_cost_pct": s.get("instruction_cost_pct"),
                "instruction_cost_ws_pct": s.get("instruction_cost_ws_pct"),
                "ws_gain_fused_pct": s.get("ws_gain_fused_pct"),
                "ws_gain_unfused_pct": s.get("ws_gain_unfused_pct"),
                "warp_specialize_available": s.get("warp_specialize_available"),
                "verdict": s.get("verdict") or s.get("warp_specialize_skipped"),
            }
            for fam, s in iso.items()
        },
        "kernel_stats": regs,
        "checks": chk,
        "grid_sizes": {
            "router_coarse_fused": tf_c.n_tried,
            "router_coarse_unfused": tu_c.n_tried,
            "router_refine_fused": tf_r.n_tried,
            "router_refine_unfused": tu_r.n_tried,
            "norm": tn.n_tried,
        },
    }
    if w13:
        row["moe_rows"] = prob.rows
        row["f11a_w13"] = speedup_row(reg.name, t_mo_f, t_mo_u, {
            "fused_cfg": mo_f_cfg,
            "unfused_gemm_cfg": mo_u_gemm,
            "unfused_norm_cfg": mo_u_norm,
            "paired_speedup": pair_mo.get("paired_speedup_p50"),
            "pair_meta": pair_mo,
            "tick": B.tick_report(t_mo_f.p50_ms, t_mo_u.p50_ms),
            "unfused_gemm_only_ms": t_mo_gemm.p50_ms,
            "norm_only_ms": t_norm.p50_ms,
            "ceiling": (tmodel.get("F11a_prenorm_w13") or {}).get("roofline_ceiling"),
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
            / (t_blas_moe_dense.p50_ms * 1e-3) / 1e12,
        })
        row["combined"] = speedup_row(reg.name, t_comb_f, t_comb_u, {
            "note": "unfused = 1 norm + router GEMM + w13 GEMM; fused = router GEMM + w13 "
                    "GEMM (x2 never materialized)",
            "norm_cfg": norm_cfg,
            "paired_speedup": pair_comb.get("paired_speedup_p50"),
            "pair_meta": pair_comb,
            "tick": B.tick_report(t_comb_f.p50_ms, t_comb_u.p50_ms),
        })
        row["half_fused"].update({
            "moe_cfg": half["moe"]["cfg"],
            "moe_ms": t_mo_h.p50_ms,
            "moe_speedup_vs_unfused": t_mo_u.p50_ms / t_mo_h.p50_ms,
            "combined_ms": t_comb_h.p50_ms,
            "combined_speedup_vs_unfused": t_comb_u.p50_ms / t_comb_h.p50_ms,
            "rel_err_moe": chk["moe_half"]["rel_err"],
        })
        row["grid_sizes"].update({
            "moe_coarse_fused": mf_c.n_tried,
            "moe_coarse_unfused": mu_c.n_tried,
            "moe_refine_fused": mf_r.n_tried,
            "moe_refine_unfused": mu_r.n_tried,
        })

    c_b = row["f11b_router"]["ceiling"]
    print(
        f"  F11b router : fused {t_rt_f.p50_ms:.4f} | unfused {t_rt_u.p50_ms:.4f} "
        f"-> {row['f11b_router']['paired_speedup']:.3f}x  "
        + (f"(ceiling {c_b:.2f}x)" if c_b else "(ceiling n/a)"), flush=True,
    )
    if w13:
        c_a = row["f11a_w13"]["ceiling"]
        print(
            f"  F11a w13    : fused {t_mo_f.p50_ms:.4f} | unfused {t_mo_u.p50_ms:.4f} "
            f"-> {row['f11a_w13']['paired_speedup']:.3f}x  "
            + (f"(ceiling {c_a:.2f}x)" if c_a else "(ceiling n/a)"), flush=True,
        )
        print(
            f"  combined    : fused {t_comb_f.p50_ms:.4f} | unfused "
            f"{t_comb_u.p50_ms:.4f} -> {row['combined']['paired_speedup']:.3f}x",
            flush=True,
        )
        print(
            f"  half-fused  : router {t_rt_h.p50_ms:.4f} "
            f"({t_rt_u.p50_ms / t_rt_h.p50_ms:.3f}x) | w13 {t_mo_h.p50_ms:.4f} "
            f"({t_mo_u.p50_ms / t_mo_h.p50_ms:.3f}x) | combined {t_comb_h.p50_ms:.4f} "
            f"({t_comb_u.p50_ms / t_comb_h.p50_ms:.3f}x)   [exploratory]", flush=True,
        )
    else:
        print(
            f"  half-fused  : router {t_rt_h.p50_ms:.4f} "
            f"({t_rt_u.p50_ms / t_rt_h.p50_ms:.3f}x)   [exploratory]", flush=True,
        )
    return row, tables, norm_cfg


# ======================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    B.add_std_args(ap, UNITS)
    # F11a's two w13 buffers are 12.9 GB EACH.  H200 can hold them, but the fast path has
    # to exist: --router-only is the switch that turns a 40-minute weight build and a
    # 26 GB allocation into nothing at all when only F11b is wanted.
    ap.add_argument(
        "--router-only", action="store_true", default=False,
        help="F11b only: skip F11a's w13 GEMM and its 2 x 12.9 GB of weights",
    )
    ap.add_argument(
        "--with-w13", dest="router_only", action="store_false",
        help="also run F11a (the default when the weights fit)",
    )
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

    # Capacity is decided at RUNTIME, not from the datasheet: 143 GB is a property of the
    # card, "free" is a property of the machine as configured (MIG, other tenants).
    need = 2 * (E * NW13 * H * 2)
    cap = B.mem_guard(need, "w13 raw + folded [256, 4096, 6144] bf16 x2")
    # `--only f11b_router` implies the w13 path is dead weight: skip it rather than spend
    # 26 GB and a weight build producing rows nobody asked for.
    only_router = not any(u in units for u in ("f11a_w13", "combined"))
    router_only = args.router_only or only_router or not cap["fits"]
    skip_reason = None
    if args.router_only:
        skip_reason = "--router-only requested"
    elif only_router:
        skip_reason = f"--only {','.join(units)} selects no F11a row"
    elif not cap["fits"]:
        skip_reason = (
            f"only {cap['free_bytes'] / 2**30:.1f} GB free, F11a needs "
            f"{need / 2**30:.1f} GB -- skipped automatically rather than OOMing mid-run"
        )
    if skip_reason:
        print(f"[scope] F11a (w13) skipped: {skip_reason}", flush=True)

    torch.manual_seed(7)
    w_norm = (torch.randn(H, device="cuda", dtype=torch.float32) * 0.1 + 1.0).to(DT)
    gate = (torch.randn(E, H, device="cuda", dtype=torch.float32) * 0.02).to(DT)
    b_raw = gate.t().contiguous()  # [H, E]
    b_fold = K.fold_weight_nk(gate, w_norm).t().contiguous()  # [H, E], w folded
    w13_raw = w13_fold = None
    if not router_only:
        print(f"building w13 (raw + folded, {need / 2**30:.1f} GB)...", flush=True)
        t0 = time.time()
        _rb, w13_raw, _fb, w13_fold = make_w13(w_norm)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)

    # ---- validate the folding identity itself, once, outside all timing --------------
    fold_err13 = None
    with torch.no_grad():
        hh = (torch.randn(64, H, device="cuda", dtype=torch.float32) * 0.5).to(DT)
        x2h = R.rmsnorm(hh, w_norm, EPS)
        lhs = x2h.float() @ b_raw.float()
        hf = hh.float()
        rstd = torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + EPS)
        rhs = (hf @ b_fold.float()) * rstd
        fold_err = rel_err(rhs, lhs)
        if w13_raw is not None:
            lhs13 = x2h.float() @ w13_raw[3].float().t()
            rhs13 = (hf @ w13_fold[3].float().t()) * rstd
            fold_err13 = rel_err(rhs13, lhs13)
    print(
        f"folding identity: router rel_err {fold_err:.3e}"
        + (f", w13 rel_err {fold_err13:.3e}" if fold_err13 is not None else ""),
        flush=True,
    )

    fair = B.Fairness(
        one_kernel_source="glm52_h200/kernels/lazy_prenorm.py :: router_gemm_kernel and "
                          "moe_gateup_prenorm_kernel",
        flags="FUSE_NORM constexpr; the unfused side is the SAME kernel with it off, plus "
              "kernels/add_rmsnorm.py::norm_only for the split-out norm",
        protocol=(
            "the coarse grid generator takes no `fused` argument, so both sides search the "
            "IDENTICAL config list (unlike F6, the lazy-prenorm kernel stages no extra "
            "SMEM tile, so no filter can differ). Refine grids are the same neighbourhood "
            "rule centred on each side's own winner, so their sizes can differ by a few "
            "configs; every count is recorded per regime. Every config is numerically "
            "screened against the fp32 reference before it is timed."
        ),
        unfused_bonus=(
            f"the unfused side additionally gets (a) an independent search over the "
            f"RMSNorm kernel's own {len(norm_grid())}-config space and (b) a joint chain "
            f"re-tune over top-3 GEMM x top-3 norm configs"
        ),
        isolation=(
            "isolation_fuse_on_vs_off_same_cfg is the measurement that carries this family. "
            "One kernel source, one config, one launch, the same tensors, and a 2x2 over the "
            "two constexprs that matter: {FUSE_NORM on/off} x {WARP_SPECIALIZE on/off}, all "
            "four arms timed inside ONE rotating interleave so drift cancels. Three of the "
            "four are the campaign's headline (unfused, fused-nonspecialized, "
            "fused-warp-specialized); the fourth -- unfused WITH specialization -- is the "
            "control that stops warp specialization's own speedup being credited to the "
            "fusion. Flat summary per regime under rows[].headline. When the stack cannot "
            "spell warp specialization, the two specialized arms are NOT timed and say so, "
            "rather than silently re-timing the classic mainloop under a specialized label."
        ),
        h200_axes=(
            "USE_TMA / WARP_SPECIALIZE / num_ctas are overlaid on the SHARED coarse grid, so "
            "both arms of both families search them. They are structurally symmetric here: "
            "the fused and unfused arms are the same kernel with FUSE_NORM flipped, stage "
            "the same SMEM (unlike F6, no extra tile) and read equal-sized operands, so "
            "every axis that is meaningful for one is meaningful for the other. Live per-arm "
            "counts are under grids.<regime>.<arm>.<stage>.axis_counts."
        ),
    )
    fair.axis("f11b_router_gemm", B.h200_axis_report(getattr(K, "ROUTER_AXES", K)))
    fair.axis("f11a_w13_gemm", B.h200_axis_report(getattr(K, "MOE_AXES", K)))
    fair.axis("f11_norm_kernel", B.h200_axis_report(NK))

    rows, tables, sq_pick, sq_tab = [], {}, {}, []

    def snapshot(done: bool) -> None:
        record(RESULT_ID, {
            "id": RESULT_ID,
            "complete": done,
            "fusion": "#11 Lazy Pre-Norm -- RMSNorm fused into a GEMM as a prologue "
                      "(Zhou et al., PyTorch blog 2026-07-10, section 2)",
            "shape": {"hidden": H, "moe_intermediate": I, "w13_N": NW13, "experts": E,
                      "top_k": TOPK, "router_N": E, "dtype": "bfloat16", "eps": EPS},
            "identity": {
                "affine_free": "(A*rstd) @ B == (A @ B) * rstd",
                "affine_handling": "((A*rstd)*w) @ B == (A @ (w[:,None]*B)) * rstd; w "
                                   "folded into the GEMM weight OFFLINE (load-time)",
                "fold_rel_err_router": fold_err,
                **({} if fold_err13 is None else {"fold_rel_err_w13": fold_err13}),
            },
            "scope": {
                "router_only": router_only,
                "skip_reason": skip_reason,
                "capacity": cap,
                "why": "F11a's w13 weights are 2 x 12.9 GB. They fit on an H200 and are "
                       "measured by default; when they are skipped every f11a_w13 / "
                       "combined key is OMITTED (not null) from the rows.",
            },
            "x2_materialization": {
                "choice": "(ii) fuse ALL K==6144 consumers; x2 is never materialized",
                "why": "x2 feeds the router, the routed-expert w13 GEMM and the shared "
                       "expert's w13 GEMM. Both benchmarked consumers are fused here; the "
                       "shared expert is the identical transform on a 1-expert weight. The "
                       "`combined` row charges ONE norm kernel to the unfused side, which "
                       "is what the real layer pays -- the per-family rows double-count it. "
                       "Under --router-only `combined` would restate f11b_router and is "
                       "omitted.",
            },
            "sq_mode_study": {"pick": sq_pick, "table": sq_tab},
            "fairness": fair.render(env),
            "env": env.__dict__,
            "rows": rows,
            "tune_tables": tables,
        })

    # ---- SQ_MODE pre-study, run ONCE at prefill_t2048 (a regime where the GEMMs are
    # genuinely compute-bound, so the sum-of-squares implementation is actually visible)
    # and then held FIXED for every regime, so the fused and unfused tuning grids stay the
    # same size.  Recorded in full in the result JSON.
    print("SQ_MODE pre-study (at prefill_t2048):", flush=True)
    study_reg = next(r for r in B.all_regimes(C) if r.name == "prefill_t2048")
    _sp = Problem(study_reg, w_norm, gate, w13_raw, w13_fold, b_raw, b_fold)
    sq_pick, sq_tab = sq_study(_sp)
    del _sp
    torch.cuda.empty_cache()

    for reg in regimes:
        ck = B.ckpt_load(RESULT_ID, reg.name, env, force=args.force)
        if ck is not None and ck.get("router_only") == router_only:
            print(f"  == {reg.name} == (from checkpoint)", flush=True)
            rows.append(ck["row"])
            tables[reg.name] = ck["tables"]
            fair.grids.update(ck.get("fairness_grids", {}))
            snapshot(False)
            continue
        prob = Problem(reg, w_norm, gate, w13_raw, w13_fold, b_raw, b_fold)
        try:
            row, tab, _ = run_regime(prob, sq_pick, args.quick, fair)
        except Exception as exc:  # noqa: BLE001 -- one regime must not lose the rest
            import traceback

            traceback.print_exc()
            tables[reg.name] = {"regime_failed": f"{type(exc).__name__}: {exc}"[:300]}
            del prob
            torch.cuda.empty_cache()
            snapshot(False)
            continue
        B.ckpt_save(RESULT_ID, reg.name, env, {
            "row": row, "tables": tab, "router_only": router_only,
            "fairness_grids": {reg.name: fair.grids.get(reg.name, {})},
        })
        rows.append(row)
        tables[reg.name] = tab
        del prob
        torch.cuda.empty_cache()
        snapshot(False)

    snapshot(True)
    print(f"\nwrote {RESULT_ID}.json\n", flush=True)
    hdr = f"{'regime':<16}{'F11b rt':>9}{'ceil':>7}"
    if not router_only:
        hdr += f"{'F11a w13':>10}{'ceil':>7}{'combined':>10}"
    print(hdr)
    for r in rows:
        c = r["f11b_router"].get("ceiling")
        line = (
            f"{r['regime']:<16}"
            f"{(r['f11b_router'].get('paired_speedup') or r['f11b_router']['speedup']):>9.3f}"
            f"{(f'{c:.2f}' if c else 'n/a'):>7}"
        )
        if "f11a_w13" in r:
            ca = r["f11a_w13"].get("ceiling")
            line += (
                f"{(r['f11a_w13'].get('paired_speedup') or r['f11a_w13']['speedup']):>10.3f}"
                f"{(f'{ca:.2f}' if ca else 'n/a'):>7}"
                f"{(r['combined'].get('paired_speedup') or r['combined']['speedup']):>10.3f}"
            )
        print(line)


if __name__ == "__main__":
    main_guard(main)
