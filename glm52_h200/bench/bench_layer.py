"""End-to-end GLM-5.2 MoE-layer subgraph benchmark: which COMBINATION of fusions minimises
total layer time, per regime.

Scope: the fusible subgraph S3-S11 plus the **shared expert**.

    o_proj -> ResAdd1 -> post-attn RMSNorm -> router(+top-8)
           -> w13 grouped GEMM -> SwiGLU -> w2 grouped GEMM -> expert merge -> ResAdd2
           -> shared expert (w13_s -> SwiGLU -> w2_s) -> add

Excluded and stated as such: the attention core, the MLA q_a/q_b/kv_a/kv_b projections and
the DSA indexer.  None is touched by any fusion candidate, so excluding them cannot change
which combination wins; it only means the absolute number is a subtotal, not a full layer.

Also excluded from the timed region (identical in every configuration, so it cannot affect
the ranking): construction of the MoE dispatch layout.  Our `moe_align_block_size` is a
torch reference with a Python loop over 256 experts; in production it is a fused CUDA
kernel costing ~10-30 us.  Every per-family benchmark in this study excluded it too.

WHY THIS EXISTS.  The per-fusion numbers do not compose, for two reasons:
  1. A kernel timed alone pays its own launch overhead and a cold L2; in a chain the next
     kernel's input is still resident.  Summing per-kernel times overstates by 10-40 % at
     decode sizes.
  2. Several fusions share a *producer*.  #4/#5/#11b all fuse the normalization into the
     router, but in the real layer that same normalization also feeds the w13 GEMM and the
     shared expert -- so fusing it into the router does not remove it.  Their standalone
     wins evaporate in context.  Only an end-to-end measurement shows this.  Deleting the
     norm entirely requires fusing it into EVERY K=6144 consumer, which is the `prenorm_all`
     axis below.

MEASUREMENT PROTOCOL (LOG-11 3).  The first C500 pass produced "winners" whose margin over
the runner-up was 0.2-0.4 %, on a machine that throws 25-320 % one-off excursions.  That
margin is not resolvable in one pass.  So each regime is measured **twice, independently**,
with R interleaved rounds per pass: within a round every candidate is timed once, in a
fixed order (reversed on odd rounds), so drift affecting a whole round cancels in the
per-config median.  A winner is declared only when its gap to the runner-up exceeds the
round-to-round spread of BOTH, in BOTH passes; otherwise the set is reported as **TIED**.
That protocol is what makes the layer numbers defensible, and it is why this file reports a
tie far more often than a ranking.

Run:
    python3 glm52_h200/bench/bench_layer.py --gpu auto [--regimes ...] [--only A_all_unfused,...]

`--gpu auto` picks the idlest of the host's GPUs and masks the process to it before
CUDA initialises; on the 8-GPU measurement host that is the difference between timing an
idle card and timing one another tenant is already using.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from glm52_h200 import bench as B
from glm52_h200 import config as C
from glm52_h200 import reference as R
from glm52_h200.common import RESULTS_DIR, autotune, bench_chain, check, main_guard, record
from glm52_h200.kernels import add_rmsnorm as KN
from glm52_h200.kernels import lazy_prenorm as KP
from glm52_h200.kernels import moe_down_merge as KD
from glm52_h200.kernels import moe_gateup as KG
from glm52_h200.kernels import norm_router as KR
from glm52_h200.kernels import oproj_resadd as KO

RESULT_ID = "layer_configurations"
H = C.HIDDEN_SIZE
I = C.MOE_INTERMEDIATE_SIZE
E = C.N_ROUTED_EXPERTS
TOPK = C.NUM_EXPERTS_PER_TOK
EPS = C.RMS_NORM_EPS

_ENV = C.env()
SMEM_LIMIT = B.env_int(_ENV, "smem_bytes")
WARP = B.env_int(_ENV, "warp_size")
MAX_THREADS = B.max_threads_per_block(_ENV)


# --------------------------------------------------------------------------------------
# Config-space guards.  Every one is a function of the probe; the C500 version of this file
# still carries `65536` and `* 64` and would build C500-shaped grids on any other device.
# --------------------------------------------------------------------------------------
def _smem_ok(bm, bn, bk, s, mult=1) -> bool:
    # `B.smem_predict` fits the multi-buffer count to the preflight's own smem_probe rows
    # (Triton 3.0 staged num_stages, 3.6/sm_89 stages num_stages-1, this H200 stack is back
    # at num_stages) instead of assuming one of them.
    return B.smem_predict(bm, bn, bk, s, bn_mult=mult) <= SMEM_LIMIT


def _row_guard(c) -> bool:
    """Bound a row kernel's per-thread fp32 state.

    C500 enforced a hard 4 KB/thread private-memory cap and failed the launch with
    `mcErrorMemoryValueTooLarge` AFTER a slow compile.  The cap itself was device-specific;
    the useful, portable form of the guard is elements per lane, computed with the PROBED
    warp width, and it is applied identically to every configuration.
    """
    threads = c.get("num_warps", 4) * WARP
    if threads > MAX_THREADS:
        return False
    return c.get("ROWS", 1) * c.get("BLOCK_N", 1) / threads <= B.MAX_ELEMS_PER_THREAD


#: Alternatives tried per key during the coordinate search, ordered by how much they
#: usually matter.  Only keys present in the seed are searched.
_ALTS = {
    # Tile alternatives come from THIS device's SMEM ceiling, so the whole-layer search can
    # reach the shapes the H200 runs (BM/BN up to 256) and stops where sm_89 stops.
    "BLOCK_M": tuple(t for t in B.tile_ladder(_ENV) if t <= 128),
    "BLOCK_N": tuple(t for t in B.tile_ladder(_ENV) if t >= 32),
    "BLOCK_K": tuple(B.bk_ladder(_ENV, hi=128)),
    "BLOCK_E": (64, 128, 256),
    "BLOCK_DIM": (256, 512, 1024),
    "BLOCK": (256, 512, 1024, 2048),
    "ROWS": (1, 2, 4, 8),
    "GROUP_M": (1, 4, 8, 16),
    "num_warps": tuple(B.warp_ladder(_ENV)),
    "num_stages": (1, 2, 3, 4),
}


def _tile_then_coord(tag, make, seed, guard=None, warmup=4, rep=12):
    """Exhaustive over the tile triple (BLOCK_M, BLOCK_N, BLOCK_K), then coordinate-refine.

    WHY NOT PURE COORDINATE DESCENT: it silently under-tuned the unfused w13 GEMM at T=512.
    Seeded at BM16/BN64/BK128 it could not reach the true optimum BM32/BN32/BK64, because
    no SINGLE key change improves on the way there -- it reported 10.83 ms where an
    exhaustive sweep finds 8.89 ms (18 % faster).  Since the fused side happened to be
    seeded nearer its own optimum, the comparison manufactured a 1.16x "win" for fusion #6
    that exhaustive tuning turns into a 0.95x loss.  The tile dims are coupled, so they
    must be swept jointly; the remaining keys are well behaved under coordinate search.
    """
    tiles = [(bm, bn, bk) for bm in _ALTS["BLOCK_M"] for bn in _ALTS["BLOCK_N"]
             for bk in _ALTS["BLOCK_K"]]
    cands = []
    for bm, bn, bk in tiles:
        c = dict(seed, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
        if guard is None or guard(c):
            cands.append(c)
    tr = autotune(make, cands, warmup, rep)
    print(f"  [cfgs] {tag:<18} tile sweep {tr.best_ms:9.4f} ms ({tr.n_tried} cfgs, "
          f"{tr.n_failed} fail)", flush=True)
    return _coord_tune(tag, make, tr.best_cfg, rounds=2, guard=guard, _seeded=True,
                       warmup=warmup, rep=rep)


def _coord_tune(tag, make, seed, rounds=2, guard=None, _seeded=False, warmup=5, rep=15):
    """Coordinate search: vary one key at a time, keep the winner, repeat.

    Safe for the cheap elementwise/row kernels, whose keys are largely independent.  NOT
    safe on its own for GEMM tile dims -- see `_tile_then_coord`.
    """
    cur = dict(seed)
    best_ms, tried = None, 0
    for _ in range(rounds):
        improved = False
        for key, alts in _ALTS.items():
            if key not in cur:
                continue
            cands = []
            for v in alts:
                if v == cur[key]:
                    continue
                c = dict(cur, **{key: v})
                if guard and not guard(c):
                    continue
                cands.append(c)
            if best_ms is None:
                cands = [dict(cur)] + cands
            if not cands:
                continue
            tr = autotune(make, cands, warmup, rep)
            tried += tr.n_tried
            if best_ms is None or tr.best_ms < best_ms:
                if best_ms is None or tr.best_cfg != cur:
                    improved = True
                best_ms, cur = tr.best_ms, tr.best_cfg
        if not improved:
            break
    print(f"  [cfgs] {tag:<18} {best_ms:9.4f} ms  ({tried} cfgs tried)  {cur}", flush=True)
    return cur


# --------------------------------------------------------------------------------------
class LayerProblem:
    def __init__(self, regime, with_prenorm: bool, seed: int = 0):
        T, Kq = regime.T, regime.oproj_k
        self.regime, self.T, self.Kq = regime.name, T, Kq
        self.rows = T * TOPK
        g = torch.Generator(device="cuda").manual_seed(seed)
        dev, dt = "cuda", torch.bfloat16

        def rnd(*shape, scale=0.05):
            """Allocate bf16 directly and fill in place.

            `torch.randn(...).to(bf16)` materialises an fp32 temporary first -- for w13
            that is a 25.8 GB allocation before the cast.  The expert stacks are filled per
            expert for the same reason.
            """
            t = torch.empty(*shape, device=dev, dtype=dt)
            if t.numel() > 2**28:
                for i in range(t.shape[0]):
                    t[i].normal_(0.0, scale, generator=g)
            else:
                t.normal_(0.0, scale, generator=g)
            return t

        # --- attention output + o_proj -------------------------------------------------
        self.a_attn = rnd(T, Kq)
        self.w_o = rnd(Kq, H, scale=0.02)
        self.h_in = rnd(T, H)  # residual entering the layer
        self.c = torch.empty(T, H, device=dev, dtype=dt)  # o_proj output (unfused)
        self.h1 = torch.empty(T, H, device=dev, dtype=dt)  # new residual
        self.acc32 = torch.zeros(T, H, device=dev, dtype=torch.float32)

        # --- norm + router -------------------------------------------------------------
        self.w_norm = (torch.randn(H, generator=g, device=dev) * 0.1 + 1).to(dt)
        self.x2 = torch.empty(T, H, device=dev, dtype=dt)
        self.wg_t = rnd(H, E, scale=0.02)  # [K, N] for the router GEMM
        self.logits = torch.empty(T, E, device=dev, dtype=torch.float32)
        self.topw = torch.empty(T, TOPK, device=dev, dtype=torch.float32)
        self.topi = torch.empty(T, TOPK, device=dev, dtype=torch.int32)

        # --- routed experts ------------------------------------------------------------
        self.w13 = rnd(E, 2 * I, H, scale=0.02)
        self.w2 = rnd(E, H, I, scale=0.02)
        self.inter = torch.empty(self.rows, 2 * I, device=dev, dtype=dt)
        self.act = torch.empty(self.rows, I, device=dev, dtype=dt)
        self.y3 = torch.zeros(self.rows, H, device=dev, dtype=dt)
        self.y3v = self.y3.view(T, TOPK, H)
        self.routed = torch.zeros(T, H, device=dev, dtype=dt)
        self.out = torch.zeros(T, H, device=dev, dtype=dt)

        # --- shared expert (1 dense expert, same intermediate width) --------------------
        self.w13s = rnd(1, 2 * I, H, scale=0.02)
        self.w2s = rnd(1, H, I, scale=0.02)
        self.s_inter = torch.empty(T, 2 * I, device=dev, dtype=dt)
        self.s_act = torch.empty(T, I, device=dev, dtype=dt)
        self.s_out = torch.empty(T, H, device=dev, dtype=dt)

        # --- lazy pre-norm (#11a/#11b) weights: the rmsnorm gain folded into every
        # K==6144 consumer's weight, offline.  Only allocated when the configuration set
        # actually contains a prenorm variant AND the memory is there; the flag is what
        # `build_chain` consults, so an absent fold can never be silently skipped.
        self.has_prenorm = bool(with_prenorm)
        if self.has_prenorm:
            self.wg_t_fold = KP.fold_weight_rowmajor(self.wg_t, self.w_norm)
            self.w13_fold = torch.empty_like(self.w13)
            wf = self.w_norm.float()
            for e in range(E):
                self.w13_fold[e] = (self.w13[e].float() * wf).to(dt)
            self.w13s_fold = (self.w13s[0].float() * wf).to(dt).unsqueeze(0)

        # Routing must be derived from the value the pipeline actually produces
        # (h1 = o_proj(a_attn) + h_in), not from h_in -- otherwise the dispatch layout and
        # the routing weights do not correspond to the tokens the kernels route.  It is
        # then FROZEN: routing is data-dependent but identical across configurations, so
        # freezing it keeps every config doing exactly the same work.  The pipeline still
        # runs (and pays for) the router GEMM + top-k, writing into scratch buffers.
        with torch.no_grad():
            h1_ref = (self.a_attn.float() @ self.w_o.float() + self.h_in.float()).to(dt)
            x2_ref = R.rmsnorm(h1_ref, self.w_norm)
            _, tw, ti = R.router(x2_ref, self.wg_t.t(), torch.zeros(E, device=dev))
        self.topw.copy_(tw)
        self.topi.copy_(ti)
        self.tw_flat = self.topw.flatten().contiguous()
        # scratch targets for the timed router/top-k, so they cannot perturb the frozen
        # routing
        self.topw_scratch = torch.empty_like(self.topw)
        self.topi_scratch = torch.empty_like(self.topi)
        self.ones = torch.ones(T, device=dev, dtype=torch.float32)

        # fp32 reference for the whole subgraph, so a bug shared by every configuration
        # cannot hide behind "all configs agree with each other"
        with torch.no_grad():
            routed = R.moe_mlp(x2_ref, self.w13, self.w2, self.topw, self.topi).float()
            s_h = torch.nn.functional.linear(x2_ref.float(), self.w13s[0].float())
            s_y = torch.nn.functional.linear(
                R.silu_and_mul(s_h.to(dt)).float(), self.w2s[0].float()
            )
            self.ref_out = (routed + h1_ref.float() + s_y).to(dt)
        del h1_ref, x2_ref, routed, s_h, s_y

        self._layouts: dict[int, tuple] = {}
        self._shared_layouts: dict[int, tuple] = {}

    def layout(self, block_m: int):
        if block_m not in self._layouts:
            self._layouts[block_m] = R.moe_align_block_size(self.topi, block_m, E)
        return self._layouts[block_m]

    def shared_layout(self, block_m: int):
        if block_m not in self._shared_layouts:
            ids = torch.zeros(self.T, 1, dtype=torch.int32, device="cuda")
            self._shared_layouts[block_m] = R.moe_align_block_size(ids, block_m, 1)
        return self._shared_layouts[block_m]


# --------------------------------------------------------------------------------------
# Chain construction.  Each option contributes kernels; only the mapping and the fusion
# flags differ, exactly as in the per-family benchmarks.
# --------------------------------------------------------------------------------------
def build_chain(p: LayerProblem, cfg: dict, sel: dict, shared_cfg: dict) -> list:
    fns = []
    L = cfg
    pre = sel.get("prenorm", "none")  # "none" | "router" | "all"
    if pre != "none" and not p.has_prenorm:
        raise RuntimeError("configuration needs the folded pre-norm weights, which were "
                           "not allocated (see --no-prenorm / memory guard)")
    # #4/#5 fuse the normalization INTO the router; lazy pre-norm fuses it into the router
    # too (and, at `all`, into every other K=6144 consumer).  Selecting both would compute
    # the norm twice and read x2 that nothing produced -- refuse rather than measure it.
    if pre != "none" and sel["norm"] in ("fused4", "fused5"):
        raise RuntimeError(f"norm={sel['norm']} and prenorm={pre} both fuse the norm into "
                           f"the router; they are mutually exclusive")
    if pre == "all" and sel["norm"] == "fused3":
        raise RuntimeError("prenorm=all makes x2 dead, so fusing the norm into the "
                           "residual add (#3) would compute a tensor nothing reads")

    # ---- S3/S4  o_proj (+ ResAdd1) ----------------------------------------------------
    if sel["resadd1"] == "in_oproj":  # fusion #1
        g = L["oproj_gemm_fused"]
        if g.get("SPLIT_K", 1) > 1:
            fns += [lambda: p.acc32.zero_(),
                    lambda: KO.gemm_launch(p.a_attn, p.w_o, p.acc32, p.h_in, g, True, True),
                    lambda: KO.epilogue_launch(
                        p.acc32, None, p.h1,
                        L["oproj_epi_fused"] or L["oproj_epi"], False)]
        else:
            fns += [lambda: KO.gemm_launch(p.a_attn, p.w_o, p.h1, p.h_in, g, True, False)]
    else:
        g = L["oproj_gemm"]
        if g.get("SPLIT_K", 1) > 1:
            fns += [lambda: p.acc32.zero_(),
                    lambda: KO.gemm_launch(p.a_attn, p.w_o, p.acc32, None, g, False, True),
                    lambda: KO.epilogue_launch(p.acc32, None, p.c, L["oproj_epi"], False)]
        else:
            fns += [lambda: KO.gemm_launch(p.a_attn, p.w_o, p.c, None, g, False, False)]

    # ---- S5/S6  ResAdd1 (if not already done), post-attention RMSNorm, router, top-8 ---
    # `pre == "all"` is the one arrangement where x2 is genuinely dead: every K=6144
    # consumer (router, w13, shared w13) reads h1 and applies rstd itself, so the norm
    # kernel is not merely fused into one consumer -- it is deleted.
    norm_sel = sel["norm"]
    if norm_sel == "fused4":  # #4: ResAdd + RMSNorm + router GEMM in one kernel
        fns += [lambda: KR.fused_add_norm_router(
            p.c, p.h_in, p.w_norm, p.h1, p.x2, p.wg_t, p.logits, L["f4"])]
    elif norm_sel == "fused5":  # #5: RMSNorm + router GEMM in one kernel
        fns += [lambda: KN.add_only(p.c, p.h_in, p.h1, L["add"]),
                lambda: KR.fused_norm_router(
                    p.h1, p.w_norm, p.x2, p.wg_t, p.logits, L["f5"])]
    else:
        if sel["resadd1"] == "in_oproj":
            pass  # h1 already produced by the fused o_proj
        elif norm_sel == "fused3":  # fusion #3
            fns += [lambda: KN.fused_add_rmsnorm(
                p.c, p.h_in, p.w_norm, p.h1, p.x2, L["addnorm"])]
        else:
            fns += [lambda: KN.add_only(p.c, p.h_in, p.h1, L["add"])]
        if pre == "all":
            pass  # x2 is never materialized
        elif norm_sel != "fused3" or sel["resadd1"] == "in_oproj":
            fns += [lambda: KN.norm_only(p.h1, p.w_norm, p.x2, L["norm"])]
        # router GEMM (timed; writes to scratch so the frozen routing is untouched)
        if pre in ("router", "all"):  # #11b: lazy pre-norm prologue in the router GEMM
            fns += [lambda: KP.launch_router(
                p.h1, p.wg_t_fold, p.logits, L["router_prenorm"], True, EPS,
                L.get("sq_router", 0))]
        else:
            fns += [lambda: KR.router_gemm(p.x2, p.wg_t, p.logits, L["router"])]
    fns += [lambda: KR.topk_only(p.logits, p.topw_scratch, p.topi_scratch, L["topk"])]

    # ---- S7/S8  w13 grouped GEMM + SwiGLU ---------------------------------------------
    if pre == "all":  # #11a: the routed w13 GEMM reads h1 and applies rstd itself
        gc = L["w13_prenorm"]
        sti, eid, ntp = p.layout(gc["BLOCK_M"])
        fns += [lambda: KP.launch_moe_gateup(
            p.h1, p.w13_fold, p.inter, sti, eid, ntp, p.rows, TOPK, gc, True, EPS,
            L.get("sq_moe", 0)),
            lambda: KG.launch_silu_and_mul(p.inter, p.act, L["act"])]
    elif sel["gateup"] == "fused6":
        gc = L["w13_fused"]
        sti, eid, ntp = p.layout(gc["BLOCK_M"])
        fns += [lambda: KG.launch_gateup(
            p.x2, p.w13, p.act, sti, eid, ntp, p.rows, TOPK, I, gc, True)]
    else:
        gc = L["w13"]
        sti, eid, ntp = p.layout(gc["BLOCK_M"])
        fns += [lambda: KG.launch_gateup(
            p.x2, p.w13, p.inter, sti, eid, ntp, p.rows, TOPK, I, gc, False),
            lambda: KG.launch_silu_and_mul(p.inter, p.act, L["act"])]

    # ---- S9/S10/S11  w2 grouped GEMM + expert merge + ResAdd2 -------------------------
    # NOTE: `launch_down` always sets MUL_ROUTED_WEIGHT=True, i.e. the routing weight is
    # applied INSIDE the down GEMM (as in sglang).  The merge that follows must therefore
    # be an UNWEIGHTED sum over top-k -- `KD.launch_moe_sum` -- not f10's `merge_only`,
    # which applies weights itself and would double-weight the result.  Every non-atomic
    # configuration once shared exactly that bug, agreed with each other at rel_err 0.0,
    # and made the CORRECT atomic configurations look wrong.
    d = sel["down"]
    if d in ("atomic8", "atomic9"):
        wc = L["w2_fused8"] if d == "atomic8" else L["w2_fused9"]
        sti2, eid2, ntp2 = p.layout(wc["BLOCK_M"])
        if d == "atomic8":
            fns += [lambda: p.routed.zero_(),
                    lambda: KD.launch_down(p.act, p.w2, p.routed, p.tw_flat, sti2, eid2,
                                           ntp2, p.rows, TOPK, wc, True),
                    lambda: KD.launch_resadd(p.routed, p.h1, p.out, L["resadd2"])]
        else:  # #9: seed the accumulator with the residual, so ResAdd2 costs nothing extra
            fns += [lambda: p.out.copy_(p.h1),
                    lambda: KD.launch_down(p.act, p.w2, p.out, p.tw_flat, sti2, eid2,
                                           ntp2, p.rows, TOPK, wc, True)]
    else:
        wc = L["w2"]
        sti2, eid2, ntp2 = p.layout(wc["BLOCK_M"])
        fns += [lambda: KD.launch_down(p.act, p.w2, p.y3, p.tw_flat, sti2, eid2, ntp2,
                                       p.rows, TOPK, wc, False)]
        if d == "merge_f10":  # fusion #10: merge + ResAdd2 in one kernel
            fns += [lambda: KD.launch_moe_sum(p.y3v, p.out, p.h1, TOPK, L["moe_sum"], True)]
        else:  # fully split: merge, then ResAdd2
            fns += [lambda: KD.launch_moe_sum(p.y3v, p.routed, p.h1, TOPK, L["moe_sum"],
                                              False),
                    lambda: KD.launch_resadd(p.routed, p.h1, p.out, L["resadd2"])]

    # ---- S12  shared expert (identical in every configuration except the prenorm axis,
    # where it is the third K==6144 consumer that has to be fused for x2 to be dead) -----
    sg, sa, sd = shared_cfg["w13"], shared_cfg["act"], shared_cfg["w2"]
    ssti, seid, sntp = p.shared_layout(sg["BLOCK_M"])
    ssti2, seid2, sntp2 = p.shared_layout(sd["BLOCK_M"])
    if pre == "all":
        fns += [lambda: KP.launch_moe_gateup(
            p.h1, p.w13s_fold, p.s_inter, ssti, seid, sntp, p.T, 1, sg, True, EPS,
            L.get("sq_moe", 0))]
    else:
        fns += [lambda: KG.launch_gateup(
            p.x2, p.w13s, p.s_inter, ssti, seid, sntp, p.T, 1, I, sg, False)]
    fns += [
        lambda: KG.launch_silu_and_mul(p.s_inter, p.s_act, sa),
        lambda: KD.launch_down(p.s_act, p.w2s, p.s_out, p.ones, ssti2, seid2, sntp2,
                               p.T, 1, sd, False),
        lambda: KD.launch_resadd(p.s_out, p.out, p.out, L["resadd2"]),
    ]
    return fns


# --------------------------------------------------------------------------------------
def tune_shared(p: LayerProblem) -> dict:
    """The shared expert is a new shape (T rows, 1 expert) -- tune it briefly.

    Small curated grid: one dense expert over T rows, and the routed-expert winners already
    tell us the right neighbourhood.
    """
    gemm_grid, act_grid = [], []
    for bm in ((16, 32, 64) if p.T <= 256 else (64, 128)):
        for bn in (64, 128):
            for bk in (32, 64):
                for w in (4, 8):
                    for s in (2, 3):
                        if not _smem_ok(bm, bn, bk, s):
                            continue
                        if w * WARP > MAX_THREADS:
                            continue
                        gemm_grid.append(dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                                              GROUP_M=8, num_warps=w, num_stages=s))
    down_grid = list(gemm_grid)
    for bm in (1, 4, 8):
        for bn in (512, 1024, 2048):
            for w in (4, 8):
                if w * WARP <= MAX_THREADS:
                    act_grid.append(dict(BLOCK_M=bm, BLOCK_N=bn, num_warps=w, num_stages=2))

    def gemm_chain(c):
        sti, eid, ntp = p.shared_layout(c["BLOCK_M"])
        return [lambda: KG.launch_gateup(p.x2, p.w13s, p.s_inter, sti, eid, ntp, p.T, 1,
                                         I, c, False)]

    def act_chain(c):
        return [lambda: KG.launch_silu_and_mul(p.s_inter, p.s_act, c)]

    def down_chain(c):
        sti, eid, ntp = p.shared_layout(c["BLOCK_M"])
        return [lambda: KD.launch_down(p.s_act, p.w2s, p.s_out, p.ones, sti, eid, ntp,
                                       p.T, 1, c, False)]

    print(f"  [shared] tuning w13 ({len(gemm_grid)}), act ({len(act_grid)}), "
          f"w2 ({len(down_grid)})", flush=True)
    tg = autotune(gemm_chain, gemm_grid, 5, 15)
    ta = autotune(act_chain, act_grid, 5, 15)
    td = autotune(down_chain, down_grid, 5, 15)
    print(f"  [shared] w13 {tg.best_ms:.4f} {tg.best_cfg} | act {ta.best_ms:.4f} | "
          f"w2 {td.best_ms:.4f} {td.best_cfg}", flush=True)
    return {"w13": tg.best_cfg, "act": ta.best_cfg, "w2": td.best_cfg,
            "ms": {"w13": tg.best_ms, "act": ta.best_ms, "w2": td.best_ms}}


def tune_layer_cfgs(p: LayerProblem, env, quick: bool) -> dict:
    """Tune every kernel the layer pipeline uses, at THIS regime's shape.

    Deliberately self-contained.  The C500 version read each family's tuned winner out of
    `results/*.json`, which made the layer benchmark depend on six other result files being
    present, current AND from this device -- three ways to import another machine's
    mapping.  Here every mapping is tuned in place and cached, device-fenced, under the
    results tree.
    """
    ck = B.ckpt_load("layer_cfgs", p.regime, env)
    if ck is not None:
        print(f"  [cfgs] reusing cached mappings for {p.regime}", flush=True)
        return ck

    w, r = (2, 6) if quick else (4, 12)
    out: dict = {}
    seed_gemm = dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=8,
                     num_stages=3)
    seed_row = dict(ROWS=2, BLOCK_N=2048, num_warps=8, num_stages=2, grid_cap=None,
                    eps=EPS)
    g1 = lambda c: _smem_ok(c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"], c["num_stages"])  # noqa: E731
    g2 = lambda c: _smem_ok(c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"], c["num_stages"], 2)  # noqa: E731

    out["oproj_gemm"] = _tile_then_coord("oproj gemm", lambda c: [
        lambda: KO.gemm_launch(p.a_attn, p.w_o, p.c, None, c, False, False)],
        seed_gemm, guard=g1, warmup=w, rep=r)
    out["oproj_gemm_fused"] = _tile_then_coord("oproj gemm+res", lambda c: [
        lambda: KO.gemm_launch(p.a_attn, p.w_o, p.h1, p.h_in, c, True, False)],
        seed_gemm, guard=g1, warmup=w, rep=r)
    out["oproj_epi"] = _coord_tune("oproj epi", lambda c: [
        lambda: KO.epilogue_launch(p.c, p.h_in, p.h1, c, True)],
        dict(BLOCK=1024, num_warps=4, num_stages=1), rounds=1, warmup=w, rep=r)
    out["oproj_epi_fused"] = out["oproj_epi"]

    out["add"] = _coord_tune("add", lambda c: [
        lambda: KN.add_only(p.c, p.h_in, p.h1, c)], seed_row, rounds=1, guard=_row_guard,
        warmup=w, rep=r)
    out["norm"] = _coord_tune("rmsnorm", lambda c: [
        lambda: KN.norm_only(p.h1, p.w_norm, p.x2, c)], seed_row, rounds=1,
        guard=_row_guard, warmup=w, rep=r)
    out["addnorm"] = _coord_tune("add+rmsnorm", lambda c: [
        lambda: KN.fused_add_rmsnorm(p.c, p.h_in, p.w_norm, p.h1, p.x2, c)], seed_row,
        rounds=1, guard=_row_guard, warmup=w, rep=r)

    seed_router = dict(BLOCK_M=32, BLOCK_K=64, BLOCK_E=256, num_warps=8, num_stages=2)
    gr = lambda c: _smem_ok(c["BLOCK_M"], c["BLOCK_E"], c["BLOCK_K"], c["num_stages"])  # noqa: E731
    out["router"] = _coord_tune("router gemm", lambda c: [
        lambda: KR.router_gemm(p.x2, p.wg_t, p.logits, c)], seed_router, guard=gr,
        warmup=w, rep=r)
    out["topk"] = _coord_tune("topk", lambda c: [
        lambda: KR.topk_only(p.logits, p.topw_scratch, p.topi_scratch, c)],
        dict(BLOCK_M=8, BLOCK_K=32, BLOCK_E=256, num_warps=8, num_stages=2), rounds=1,
        warmup=w, rep=r)
    out["f5"] = _coord_tune("f5 norm+router", lambda c: [
        lambda: KR.fused_norm_router(p.h1, p.w_norm, p.x2, p.wg_t, p.logits, c)],
        seed_router, guard=gr, warmup=w, rep=r)
    out["f4"] = _coord_tune("f4 add+norm+rt", lambda c: [
        lambda: KR.fused_add_norm_router(p.c, p.h_in, p.w_norm, p.h1, p.x2, p.wg_t,
                                         p.logits, c)],
        seed_router, guard=gr, warmup=w, rep=r)

    def gate(c, fused):
        sti, eid, ntp = p.layout(c["BLOCK_M"])
        dst = p.act if fused else p.inter
        return [lambda: KG.launch_gateup(p.x2, p.w13, dst, sti, eid, ntp, p.rows, TOPK, I,
                                         c, fused)]

    out["w13"] = _tile_then_coord("w13 gemm", lambda c: gate(c, False), seed_gemm,
                                  guard=g1, warmup=w, rep=r)
    out["w13_fused"] = _tile_then_coord("w13+swiglu", lambda c: gate(c, True), seed_gemm,
                                        guard=g2, warmup=w, rep=r)
    out["act"] = _coord_tune("silu_and_mul", lambda c: [
        lambda: KG.launch_silu_and_mul(p.inter, p.act, c)],
        dict(BLOCK_M=4, BLOCK_N=1024, num_warps=8, num_stages=2), rounds=1,
        warmup=w, rep=r)

    def down(c, fuse):
        sti, eid, ntp = p.layout(c["BLOCK_M"])
        dst = p.routed if fuse else p.y3
        return [lambda: KD.launch_down(p.act, p.w2, dst, p.tw_flat, sti, eid, ntp,
                                       p.rows, TOPK, c, fuse)]

    out["w2"] = _tile_then_coord("w2 gemm", lambda c: down(c, False), seed_gemm, guard=g1,
                                 warmup=w, rep=r)
    out["w2_fused8"] = _tile_then_coord("w2+merge(atomic)", lambda c: down(c, True),
                                        seed_gemm, guard=g1, warmup=w, rep=r)
    out["w2_fused9"] = out["w2_fused8"]
    out["moe_sum"] = _coord_tune("moe_sum", lambda c: [
        lambda: KD.launch_moe_sum(p.y3v, p.routed, p.h1, TOPK, c, False)],
        dict(BLOCK_M=4, BLOCK_DIM=1024, num_warps=8, num_stages=2), rounds=1,
        warmup=w, rep=r)
    out["resadd2"] = _coord_tune("resadd", lambda c: [
        lambda: KD.launch_resadd(p.routed, p.h1, p.out, c)],
        dict(BLOCK_M=4, BLOCK_N=1024, num_warps=8, num_stages=2), rounds=1,
        warmup=w, rep=r)

    if p.has_prenorm:
        out["router_prenorm"] = _tile_then_coord("router prenorm", lambda c: [
            lambda: KP.launch_router(p.h1, p.wg_t_fold, p.logits, c, True, EPS, 0)],
            dict(BLOCK_M=32, BLOCK_N=256, BLOCK_K=64, GROUP_M=8, num_warps=8,
                 num_stages=3), guard=g1, warmup=w, rep=r)

        def gate_pre(c):
            sti, eid, ntp = p.layout(c["BLOCK_M"])
            return [lambda: KP.launch_moe_gateup(
                p.h1, p.w13_fold, p.inter, sti, eid, ntp, p.rows, TOPK, c, True, EPS, 0)]

        out["w13_prenorm"] = _tile_then_coord("w13 prenorm", gate_pre, seed_gemm,
                                              guard=g1, warmup=w, rep=r)
        out["sq_router"] = 0
        out["sq_moe"] = 0

    B.ckpt_save("layer_cfgs", p.regime, env, out)
    return out


# --------------------------------------------------------------------------------------
# Curated configuration set.  The axes touch disjoint kernels, so a baseline plus
# one-axis-at-a-time variants plus the predicted-best combinations identifies the optimum
# without an exhaustive product.  All eleven fusions in the study appear on some axis.
# --------------------------------------------------------------------------------------
CONFIGS = {
    "A_all_unfused":  dict(resadd1="separate", norm="split", gateup="split", down="split"),
    "B_f3":           dict(resadd1="separate", norm="fused3", gateup="split", down="split"),
    "C_f1":           dict(resadd1="in_oproj", norm="split", gateup="split", down="split"),
    "D_f6":           dict(resadd1="separate", norm="split", gateup="fused6", down="split"),
    "E_f10":          dict(resadd1="separate", norm="split", gateup="split",
                           down="merge_f10"),
    "F_f8":           dict(resadd1="separate", norm="split", gateup="split", down="atomic8"),
    "G_f9":           dict(resadd1="separate", norm="split", gateup="split", down="atomic9"),
    "H_f3_f10":       dict(resadd1="separate", norm="fused3", gateup="split",
                           down="merge_f10"),
    "I_f3_f9":        dict(resadd1="separate", norm="fused3", gateup="split",
                           down="atomic9"),
    "J_greedy_all":   dict(resadd1="in_oproj", norm="split", gateup="fused6",
                           down="atomic9"),
    "K_f3_f8":        dict(resadd1="separate", norm="fused3", gateup="split",
                           down="atomic8"),
    # --- the norm->router family, which only an end-to-end build can price -------------
    "L_f5":           dict(resadd1="separate", norm="fused5", gateup="split", down="split"),
    "M_f4":           dict(resadd1="separate", norm="fused4", gateup="split", down="split"),
    # --- lazy pre-norm.  `router` fuses only the router (the norm kernel still runs for
    # w13 and the shared expert); `all` fuses every K=6144 consumer, which is the only
    # arrangement where x2 is genuinely dead. -------------------------------------------
    "N_f11b":         dict(resadd1="separate", norm="split", gateup="split", down="split",
                           prenorm="router"),
    "O_f11ab":        dict(resadd1="separate", norm="split", gateup="split", down="split",
                           prenorm="all"),
    "P_f10_f11ab":    dict(resadd1="separate", norm="split", gateup="split",
                           down="merge_f10", prenorm="all"),
    "Q_f8_f11ab":     dict(resadd1="separate", norm="split", gateup="split",
                           down="atomic8", prenorm="all"),
    # with x2 dead AND ResAdd1 folded into o_proj, the whole [add][norm] pair disappears --
    # the most aggressive arrangement the layer admits that is not simply greedy
    "R_f1_f10_f11ab": dict(resadd1="in_oproj", norm="split", gateup="split",
                           down="merge_f10", prenorm="all"),
}
PRENORM_CONFIGS = [k for k, v in CONFIGS.items() if v.get("prenorm", "none") != "none"]


# --------------------------------------------------------------------------------------
# The two-pass interleaved protocol
# --------------------------------------------------------------------------------------
def measure_pass(chains: dict, rounds: int, rep: int, tag: str) -> dict:
    """R interleaved rounds.  Within a round every candidate is timed once, in a fixed
    order that reverses on odd rounds; drift affecting a whole round therefore cancels in
    the per-config median, and no candidate systematically inherits another's cache state.
    """
    names = list(chains)
    per: dict[str, list[float]] = {k: [] for k in names}
    for r in range(rounds):
        order = names if r % 2 == 0 else list(reversed(names))
        for name in order:
            t = bench_chain(chains[name], 5, rep)
            per[name].append(t.p50_ms)
        print(f"    {tag} round {r}: "
              + "  ".join(f"{n}={per[n][-1]:.4f}" for n in names), flush=True)
    return per


def summarise(per: dict) -> dict:
    stats = {}
    for name, xs in per.items():
        xs_s = sorted(xs)
        med = statistics.median(xs)
        stats[name] = {
            "median": med, "min": xs_s[0], "max": xs_s[-1],
            "spread_ms": xs_s[-1] - xs_s[0],
            "spread_pct": (xs_s[-1] - xs_s[0]) / med * 100 if med else float("nan"),
            "rounds": xs,
        }
    return stats


def verdict_of(stats: dict) -> dict:
    """Winner only if its gap to the runner-up exceeds the round-to-round spread of both.

    Anything else is a TIE, and the tied set is reported in full.  This is the rule that
    turned a 0.2-0.4 % "clean win" on C500 into an honest "at the edge of what this machine
    can resolve".
    """
    order = sorted(stats, key=lambda k: stats[k]["median"])
    if len(order) < 2:
        return {"best": order[0] if order else None, "separated": False, "tied_set": order}
    best, runner = order[0], order[1]
    gap = stats[runner]["median"] - stats[best]["median"]
    noise = max(stats[best]["spread_ms"], stats[runner]["spread_ms"])
    tied = [n for n in order if stats[n]["median"] - stats[best]["median"] <= noise]
    return {
        "best": best, "runner_up": runner, "gap_ms": gap, "noise_ms": noise,
        "separated": bool(gap > noise), "tied_set": tied, "order": order,
    }


def combine(v1: dict, v2: dict, s1: dict, s2: dict) -> dict:
    """Two independent passes.  A winner survives only if BOTH passes separate it and BOTH
    name the same configuration; otherwise the union of the two tied sets is the answer."""
    same = v1.get("best") == v2.get("best")
    sep = bool(v1.get("separated") and v2.get("separated") and same)
    tied = sorted(set(v1.get("tied_set", [])) | set(v2.get("tied_set", [])))
    return {
        "winner": v1.get("best") if sep else None,
        "status": "SEPARATED" if sep else "TIED",
        "tied_set": [v1.get("best")] if sep else tied,
        "pass1": v1,
        "pass2": v2,
        "agree_on_best": same,
        "median_of_passes": {
            n: statistics.fmean([s1[n]["median"], s2[n]["median"]])
            for n in s1 if n in s2
        },
    }


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    B.add_std_args(ap, list(CONFIGS))
    ap.add_argument("--rounds", type=int, default=8,
                    help="interleaved rounds per pass (LOG-11 used 8)")
    ap.add_argument("--rep", type=int, default=15, help="reps per candidate per round")
    ap.add_argument("--no-prenorm", action="store_true",
                    help="skip the lazy-pre-norm configurations and their folded weights "
                         "(saves ~13 GB and the fold pass)")
    args = ap.parse_args()
    if args.list:
        print("regimes:", ", ".join(B.REGIME_NAMES))
        print("configurations:", ", ".join(CONFIGS))
        return

    env = C.env()
    B.banner(env)
    B.exact_fp32_matmul()
    B.check_preflight_device(env)
    regimes = B.resolve_regimes(C, args.regimes)
    want = B.resolve_units(list(CONFIGS), args.only)

    rounds = 2 if args.quick else args.rounds
    rep = 5 if args.quick else args.rep

    # Whether the pre-norm axis can run at all is decided from FREE memory at runtime: the
    # folded w13 is a second 12.9 GB stack on top of w13 + w2.  If it does not fit, those
    # configurations are dropped by name and the reason is recorded -- they are never
    # silently replaced by their unfused twin.
    base_need = (E * 2 * I * H * 2) + (E * H * I * 2)
    fold_need = E * 2 * I * H * 2
    cap = B.mem_guard(base_need + fold_need, "w13 + w2 + folded w13")
    with_prenorm = (not args.no_prenorm) and cap["fits"]
    prenorm_skip = None
    if args.no_prenorm:
        prenorm_skip = "--no-prenorm requested"
    elif not cap["fits"]:
        prenorm_skip = (
            f"folded w13 needs another {fold_need / 2**30:.1f} GB; only "
            f"{cap['free_bytes'] / 2**30:.1f} GB free"
        )
    if not with_prenorm:
        want = [k for k in want if k not in PRENORM_CONFIGS]
        print(f"[scope] pre-norm configurations dropped: {prenorm_skip}", flush=True)

    out_all: dict = {}
    for regime in regimes:
        print(f"\n===== {regime.name} =====", flush=True)
        p = LayerProblem(regime, with_prenorm)
        cfg = tune_layer_cfgs(p, env, args.quick)
        shared = tune_shared(p)

        # ---- build + validate every candidate against the INDEPENDENT fp32 reference ---
        # not against another configuration: configs sharing a bug agree with each other
        # while all being wrong, and that is exactly how the double-applied routing weight
        # nearly shipped.
        chains, correctness = {}, {}
        for name in want:
            sel = CONFIGS[name]
            try:
                chain = build_chain(p, cfg, sel, shared)
                p.out.zero_(); p.routed.zero_(); p.y3.zero_()
                for fn in chain:
                    fn()
                torch.cuda.synchronize()
                ck = check(p.out.clone(), p.ref_out, tol=5e-2, label=name)
                correctness[name] = {"rel_err": ck["rel_err"], "ok": ck["ok"],
                                     "n_kernels": len(chain), "sel": sel}
                if not ck["ok"]:
                    print(f"  {name:<18} !! FAILS CORRECTNESS relerr={ck['rel_err']:.3e} "
                          f"-- excluded", flush=True)
                    continue
                chains[name] = chain
                print(f"  {name:<18} ok  ({len(chain)} kernels)  "
                      f"relerr {ck['rel_err']:.2e}", flush=True)
            except Exception as exc:  # noqa: BLE001 -- one config must not lose the regime
                correctness[name] = {"error": f"{type(exc).__name__}: {exc}"[:300],
                                     "sel": sel}
                print(f"  {name:<18} FAILED {type(exc).__name__}: {exc}", flush=True)
            finally:
                torch.cuda.empty_cache()

        if not chains:
            raise RuntimeError(f"{regime.name}: no configuration passed correctness")

        # ---- two independent interleaved passes ---------------------------------------
        per1 = measure_pass(chains, rounds, rep, "pass1")
        per2 = measure_pass(chains, rounds, rep, "pass2")
        s1, s2 = summarise(per1), summarise(per2)
        res = combine(verdict_of(s1), verdict_of(s2), s1, s2)

        base = res["median_of_passes"].get("A_all_unfused")
        head = res["winner"] or (res["tied_set"][0] if res["tied_set"] else None)
        if head:
            print(f"  --> {res['status']}: {res['winner'] or ', '.join(res['tied_set'])}",
                  flush=True)
            for tag, v in (("pass1", res["pass1"]), ("pass2", res["pass2"])):
                gap, noise = v.get("gap_ms"), v.get("noise_ms")
                print(f"      {tag}: best {v.get('best')} gap "
                      f"{'n/a' if gap is None else f'{gap:.5f}'} ms vs round noise "
                      f"{'n/a' if noise is None else f'{noise:.5f}'} ms -> "
                      f"{'SEPARATED' if v.get('separated') else 'tied with ' + ', '.join(v.get('tied_set', []))}",
                      flush=True)
            if base:
                print(f"      best/all-unfused = "
                      f"{base / res['median_of_passes'][head]:.4f}x", flush=True)

        # ---- one paired A/B of the head configuration against the all-unfused baseline.
        # The round-robin above is already interleaved, but a direct paired ratio is the
        # statistic that survives monotone drift, and it is what the report quotes.
        paired = None
        if head and head != "A_all_unfused" and "A_all_unfused" in chains:
            tf, tu, pm = B.bench_pair(chains[head], chains["A_all_unfused"],
                                      5, max(20, rep * 2), label=f"{regime.name}/head")
            paired = {
                "head": head,
                "head_ms": tf.p50_ms,
                "all_unfused_ms": tu.p50_ms,
                "paired_speedup_p50": pm.get("paired_speedup_p50"),
                "paired_speedup_trimmed": pm.get("paired_speedup_trimmed_mean"),
                "pair_meta": pm,
                "tick": B.tick_report(tf.p50_ms, tu.p50_ms),
            }
            print(f"      paired head-vs-baseline: "
                  f"{paired['paired_speedup_p50']:.4f}x", flush=True)

        out_all[regime.name] = {
            "T": p.T, "oproj_K": p.Kq, "moe_rows": p.rows,
            "correctness": correctness,
            "pass1": s1, "pass2": s2,
            "verdict": res,
            "paired_head_vs_unfused": paired,
            "speedup_vs_unfused": (
                base / res["median_of_passes"][head] if (base and head) else None
            ),
            "shared_expert_ms": shared["ms"],
            "cfgs": cfg,
        }
        record(RESULT_ID, {
            "id": RESULT_ID,
            "scope": "S3-S11 + shared expert; attention core / MLA projections / DSA "
                     "indexer excluded (untouched by every fusion candidate). MoE "
                     "dispatch-layout construction excluded from the timed region, as in "
                     "every per-family bench.",
            "protocol": {
                "passes": 2,
                "rounds_per_pass": rounds,
                "rep_per_round": rep,
                "rule": "within a round every candidate is timed once, in a fixed order "
                        "that reverses on odd rounds; a winner is declared only when its "
                        "gap to the runner-up exceeds the round-to-round spread of BOTH, "
                        "in BOTH passes, and both passes name the same configuration -- "
                        "otherwise the set is reported as TIED",
                "why": "the C500 first pass produced winners with a 0.2-0.4 % margin on a "
                       "machine documented to throw 25-320 % one-off excursions; that "
                       "margin is not resolvable in one pass (LOG-11 3)",
            },
            "configs": CONFIGS,
            "prenorm": {"enabled": with_prenorm, "skip_reason": prenorm_skip,
                        "capacity": cap},
            "env": env.__dict__,
            "fairness": B.Fairness(
                mapping="every kernel in the pipeline is tuned at THIS regime's shape and "
                        "cached device-fenced; no mapping is imported from another result "
                        "file or another machine",
                routing="frozen after being computed from the pipeline's own h1, so every "
                        "configuration routes identical tokens to identical experts; the "
                        "router GEMM and top-k still run and are still timed, writing to "
                        "scratch",
                correctness="every configuration is validated against an INDEPENDENT fp32 "
                            "reference of the whole subgraph, never against another "
                            "configuration; a failing configuration is excluded outright",
                h200_axes="this bench composes whole pipelines rather than tuning a "
                          "fused/unfused PAIR, so there is no grid to bias: every "
                          "configuration reaches its kernels through the same "
                          "_tile_then_coord search over the same alternatives, and any "
                          "sm_90 axis a kernel module honours is therefore available to all "
                          "of them equally. The per-fusion axis reports live in the "
                          "individual bench_fNN result files.",
                gpu_note="which physical card this ran on is under fairness.gpu; on the "
                         "8-GPU measurement host that is not implied by the device name",
            ).render(env),
            "regimes": out_all,
        })
        del p
        torch.cuda.empty_cache()

    print(f"\nwrote {RESULT_ID}.json")
    print(f"{'regime':<16}{'status':<11}{'winner / tied set':<44}{'vs unfused':>11}")
    for name, d in out_all.items():
        v = d["verdict"]
        who = v["winner"] or ", ".join(v["tied_set"][:3]) + (
            " ..." if len(v["tied_set"]) > 3 else ""
        )
        sp = d.get("speedup_vs_unfused")
        print(f"{name:<16}{v['status']:<11}{who:<44}"
              f"{(f'{sp:.4f}x' if sp else 'n/a'):>11}")


if __name__ == "__main__":
    main_guard(main)
