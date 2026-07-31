"""End-to-end GLM-5.2 MoE-layer subgraph benchmark: which COMBINATION of fusions minimises
total layer time, per regime.

Scope (agreed with the user): the fusible subgraph S3-S11 plus the **shared expert**.

    o_proj -> ResAdd1 -> post-attn RMSNorm -> router(+top-8)
           -> w13 grouped GEMM -> SwiGLU -> w2 grouped GEMM -> expert merge -> ResAdd2
           -> shared expert (w13_s -> SwiGLU -> w2_s) -> add

Excluded and stated as such: the attention core, the MLA q_a/q_b/kv_a/kv_b projections and
the DSA indexer. None is touched by any fusion candidate, so excluding them cannot change
which combination wins; it only means the absolute number is a subtotal, not the full layer.

Also excluded from the timed region (identical in every configuration, so it cannot affect
the ranking): construction of the MoE dispatch layout. Our `moe_align_block_size` is a torch
reference with a Python loop over 256 experts; in production it is a fused CUDA kernel
costing ~10-30 us. Every per-family benchmark in this study excluded it too.

WHY THIS EXISTS. The per-fusion numbers do not compose. Two reasons:
  1. A kernel timed alone pays its own launch overhead and a cold L2; in a chain the next
     kernel's input is still resident. Summing per-kernel times overstates by 10-40 % at
     decode sizes.
  2. More importantly, several fusions share a *producer*. #4/#5/#11b all fuse the
     normalization into the router, but in the real layer that same normalization also feeds
     the w13 GEMM and the shared expert -- so fusing it into the router does not remove it.
     Their standalone wins evaporate in context. Only an end-to-end measurement shows this.

Run:
    CUDA_VISIBLE_DEVICES=<n> python glm52/bench/bench_layer.py [--regimes decode_bs1,...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from glm52 import config as C  # noqa: E402
from glm52 import reference as R  # noqa: E402
from glm52.common import autotune, bench_chain, check, record  # noqa: E402
from glm52.kernels import add_rmsnorm as KN  # noqa: E402
from glm52.kernels import merge_resadd as KM  # noqa: E402
from glm52.kernels import moe_down_merge as KD  # noqa: E402
from glm52.kernels import moe_gateup as KG  # noqa: E402
from glm52.kernels import norm_router as KR  # noqa: E402
from glm52.kernels import oproj_resadd as KO  # noqa: E402

H = C.HIDDEN_SIZE
I = C.MOE_INTERMEDIATE_SIZE
E = C.N_ROUTED_EXPERTS
TOPK = C.NUM_EXPERTS_PER_TOK
RESULTS = ROOT / "results"

REGIMES = {
    "decode_bs1": (1, C.OPROJ_K_DECODE),
    "decode_bs32": (32, C.OPROJ_K_DECODE),
    "decode_bs256": (256, C.OPROJ_K_DECODE),
    # T=512 and T=1024 straddle the predicted #8/#10 crossover: the atomic accumulator is
    # [T, 6144] bf16, which fits C500's 8 MB L2 up to T = 8*2**20/(6144*2) = 683 tokens.
    # T=512 -> 6.3 MB (fits); T=1024 -> 12.6 MB (does not). See log/LOG-12.
    "decode_bs512": (512, C.OPROJ_K_DECODE),
    "decode_bs1024": (1024, C.OPROJ_K_DECODE),
    "prefill_t2048": (2048, C.OPROJ_K_PREFILL),
    "prefill_t8192": (8192, C.OPROJ_K_PREFILL),
}

# Regimes with per-family tuned mappings on disk; others are tuned by `tune_layer_cfgs`.
FAMILY_TUNED = {"decode_bs1", "decode_bs32", "decode_bs256", "prefill_t2048", "prefill_t8192"}


# --------------------------------------------------------------------------------------
# Per-regime winning mappings, taken from each family's own tuned result
# --------------------------------------------------------------------------------------
def _smem_ok(bm, bn, bk, s, mult=1):
    return s * 2 * bk * (bm + mult * bn) <= 65536


# Alternatives tried per key during the coordinate search, ordered by how much they usually
# matter. Only keys present in the seed are searched.
_ALTS = {
    "BLOCK_M": (16, 32, 64, 128),
    "BLOCK_N": (32, 64, 128, 256),
    "BLOCK_K": (32, 64, 128),
    "BLOCK_E": (64, 128, 256),
    "BLOCK_DIM": (256, 512, 1024),
    "BLOCK": (256, 512, 1024, 2048),
    "ROWS": (1, 2, 4, 8),
    "GROUP_M": (1, 4, 8, 16),
    "num_warps": (1, 2, 4, 8, 16),
    "num_stages": (1, 2, 3, 4),
}


def _row_guard(c):
    """C500 enforces a hard 4 KB/thread private-memory cap; exceeding it fails the launch
    with `mcErrorMemoryValueTooLarge` *after* a slow compile, which dominated the first
    tuning attempt. A row kernel holds ROWS*BLOCK_N fp32 values across num_warps*64 threads,
    so bound that per-thread and skip the configs that cannot launch."""
    threads = c.get("num_warps", 4) * 64
    return c.get("ROWS", 1) * c.get("BLOCK_N", 1) * 4 / threads <= 2048


def _tile_then_coord(tag, make, seed, guard=None, warmup=4, rep=12):
    """Exhaustive over the tile triple (BLOCK_M, BLOCK_N, BLOCK_K), then coordinate-refine.

    WHY NOT PURE COORDINATE DESCENT: it silently under-tuned the unfused w13 GEMM at
    T=512. Seeded at BM16/BN64/BK128 it could not reach the true optimum BM32/BN32/BK64,
    because no SINGLE key change improves on the way there — it reported 10.83 ms where an
    exhaustive sweep finds 8.89 ms (18 % faster). Since the fused side happened to be seeded
    nearer its own optimum, the comparison manufactured a 1.16x "win" for fusion #6 that
    exhaustive tuning turns into a 0.95x loss. The tile dims are coupled, so they must be
    swept jointly; the remaining keys are well behaved under coordinate search.
    """
    tiles = [(bm, bn, bk) for bm in _ALTS["BLOCK_M"] for bn in _ALTS["BLOCK_N"]
             for bk in _ALTS["BLOCK_K"]]
    cands = []
    for bm, bn, bk in tiles:
        c = dict(seed, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
        if guard is None or guard(c):
            cands.append(c)
    tr = autotune(make, cands, warmup=warmup, rep=rep)
    print(f"  [cfgs] {tag:<16} tile sweep {tr.best_ms:9.4f} ms ({tr.n_tried} cfgs)",
          flush=True)
    return _coord_tune(tag, make, tr.best_cfg, rounds=2, guard=guard, _seeded=True,
                       warmup=warmup, rep=rep)


def _coord_tune(tag, make, seed, rounds=2, guard=None, _seeded=False,
                warmup=5, rep=15):
    """Coordinate search: vary one key at a time, keep the winner, repeat.

    Safe for the cheap elementwise/row kernels, whose keys are largely independent. NOT safe
    on its own for GEMM tile dims — see `_tile_then_coord`.
    """
    cur = dict(seed)
    best_ms = None
    tried = 0
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
            tr = autotune(make, cands, warmup=warmup, rep=rep)
            tried += tr.n_tried
            if best_ms is None or tr.best_ms < best_ms:
                if best_ms is None or tr.best_cfg != cur:
                    improved = True
                best_ms, cur = tr.best_ms, tr.best_cfg
        if not improved:
            break
    print(f"  [cfgs] {tag:<16} {best_ms:9.4f} ms  ({tried} cfgs tried)  {cur}", flush=True)
    return cur


def tune_layer_cfgs(p: "LayerProblem") -> dict:
    """Tune every kernel the layer pipeline uses, for a regime with no per-family results.

    Seeded from decode_bs256 (the nearest tuned regime) and refined by coordinate search, so
    each mapping is that kernel's own optimum at THIS size rather than a borrowed one.
    Cached to results/_layer_cfgs_<regime>.json.
    """
    cache = RESULTS / f"_layer_cfgs_{p.regime}.json"
    if cache.exists():
        print(f"  [cfgs] reusing {cache.name}", flush=True)
        return json.loads(cache.read_text())

    seed = load_cfgs("decode_bs256")
    out = {}
    g1 = lambda c: _smem_ok(c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"], c["num_stages"])
    g2 = lambda c: _smem_ok(c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"], c["num_stages"], 2)

    out["oproj_gemm"] = _tile_then_coord("oproj gemm", lambda c: [
        lambda: KO.gemm_launch(p.a_attn, p.w_o, p.c, None, c, False, False)],
        {k: v for k, v in seed["oproj_gemm"].items() if k != "SPLIT_K"}, guard=g1)
    out["oproj_gemm_fused"] = _tile_then_coord("oproj gemm+res", lambda c: [
        lambda: KO.gemm_launch(p.a_attn, p.w_o, p.h1, p.h_in, c, True, False)],
        {k: v for k, v in seed["oproj_gemm_fused"].items() if k != "SPLIT_K"}, guard=g1)
    out["oproj_epi"] = _coord_tune("oproj epi", lambda c: [
        lambda: KO.epilogue_launch(p.c, p.h_in, p.h1, c, True)], seed["oproj_epi"], rounds=1)
    out["oproj_epi_fused"] = out["oproj_epi"]

    out["add"] = _coord_tune("add", lambda c: [
        lambda: KN.add_only(p.c, p.h_in, p.h1, c)], seed["add"], rounds=1, guard=_row_guard)
    out["norm"] = _coord_tune("rmsnorm", lambda c: [
        lambda: KN.norm_only(p.h1, p.w_norm, p.x2, c)], seed["norm"], rounds=1,
        guard=_row_guard)
    out["addnorm"] = _coord_tune("add+rmsnorm", lambda c: [
        lambda: KN.fused_add_rmsnorm(p.c, p.h_in, p.w_norm, p.h1, p.x2, c)],
        seed["addnorm"], rounds=1, guard=_row_guard)

    out["router"] = _coord_tune("router gemm", lambda c: [
        lambda: KR.router_gemm(p.x2, p.wg_t, p.logits, c)], seed["router"],
        guard=lambda c: _smem_ok(c["BLOCK_M"], c["BLOCK_E"], c["BLOCK_K"], c["num_stages"]))
    out["topk"] = _coord_tune("topk", lambda c: [
        lambda: KR.topk_only(p.logits, p.topw_scratch, p.topi_scratch, c)], seed["topk"], rounds=1)

    def gate(c, fused):
        sti, eid, ntp = p.layout(c["BLOCK_M"])
        dst = p.act if fused else p.inter
        return [lambda: KG.launch_gateup(p.x2, p.w13, dst, sti, eid, ntp,
                                         p.rows, TOPK, I, c, fused)]
    out["w13"] = _tile_then_coord("w13 gemm", lambda c: gate(c, False), seed["w13"], guard=g1)
    out["w13_fused"] = _tile_then_coord("w13+swiglu", lambda c: gate(c, True), seed["w13_fused"],
                                   guard=g2)
    out["act"] = _coord_tune("silu_and_mul", lambda c: [
        lambda: KG.launch_silu_and_mul(p.inter, p.act, c)], seed["act"], rounds=1)

    def down(c, fuse):
        sti, eid, ntp = p.layout(c["BLOCK_M"])
        dst = p.routed if fuse else p.y3
        return [lambda: KD.launch_down(p.act, p.w2, dst, p.tw_flat, sti, eid, ntp,
                                       p.rows, TOPK, c, fuse)]
    out["w2"] = _tile_then_coord("w2 gemm", lambda c: down(c, False), seed["w2"], guard=g1)
    out["w2_fused8"] = _tile_then_coord("w2+merge(atomic)", lambda c: down(c, True),
                                   seed["w2_fused8"], guard=g1)
    out["w2_fused9"] = out["w2_fused8"]
    out["moe_sum"] = _coord_tune("moe_sum", lambda c: [
        lambda: KD.launch_moe_sum(p.y3v, p.routed, p.h1, TOPK, c, False)], seed["moe_sum"], rounds=1)
    out["resadd2"] = _coord_tune("resadd", lambda c: [
        lambda: KD.launch_resadd(p.routed, p.h1, p.out, c)], seed["resadd2"], rounds=1)

    cache.write_text(json.dumps(out, indent=2))
    print(f"  [cfgs] wrote {cache.name}", flush=True)
    return out


def load_cfgs(regime: str) -> dict:
    def js(n):
        return json.loads((RESULTS / f"{n}.json").read_text())

    def row(payload, variant=None):
        for r in payload["rows"]:
            if r["regime"] == regime and (variant is None or r.get("variant") == variant):
                return r
        raise KeyError(regime)

    f01 = row(js("f01_oproj_resadd"))
    f03 = row(js("f03_resadd_rmsnorm"))
    f05 = row(js("f04f05_norm_router"), "F5")
    f06 = row(js("f06_upgate_swiglu"))
    f08 = row(js("f08f09_down_merge_resadd"), "f8_atomic")
    f09 = row(js("f08f09_down_merge_resadd"), "f9_atomic")
    f10 = row(js("f10_merge_resadd"))
    split = json.loads((RESULTS / "_f01_perkernel_split.json").read_text())[regime]

    return {
        "oproj_gemm": {k: v for k, v in f01["unfused_cfg"].items() if k != "EPI"},
        "oproj_gemm_fused": {k: v for k, v in f01["fused_cfg"].items() if k != "EPI"},
        "oproj_epi": split["epi_cfg"],
        "oproj_epi_fused": f01["fused_cfg"].get("EPI"),
        "add": f03["unfused_cfg"]["add"],
        "norm": f03["unfused_cfg"]["norm"],
        "addnorm": f03["fused_cfg"],
        "router": f05["unfused_cfg"]["gemm"],
        "topk": f05["unfused_cfg"].get("topk") or js("f04f05_norm_router")["tune_tables"][
            regime]["topk"]["best_cfg"],
        "w13": f06["unfused_gemm_cfg"],
        "act": f06["unfused_act_cfg"],
        "w13_fused": f06["fused_cfg"],
        "w2": f08["unfused_cfg"]["gemm"],
        "moe_sum": f08["unfused_cfg"]["sum"],
        "resadd2": f09["unfused_cfg"].get("resadd") or f10["unfused_cfg"]["resadd"],
        "w2_fused8": f08["fused_cfg"]["gemm"],
        "seed8": f08["fused_cfg"].get("seed", {"impl": "torch"}),
        "w2_fused9": f09["fused_cfg"]["gemm"],
        "seed9": f09["fused_cfg"].get("seed", {"impl": "torch"}),
        "merge": f10["unfused_cfg"]["merge"],
        "mr_resadd": f10["unfused_cfg"]["resadd"],
        "merge_fused": f10["fused_cfg"],
    }


# --------------------------------------------------------------------------------------
class LayerProblem:
    def __init__(self, regime: str, seed: int = 0):
        T, Kq = REGIMES[regime]
        self.regime, self.T, self.Kq = regime, T, Kq
        self.rows = T * TOPK
        g = torch.Generator(device="cuda").manual_seed(seed)
        dev, dt = "cuda", torch.bfloat16

        def rnd(*shape, scale=0.05):
            """Allocate bf16 directly and fill in place.

            `torch.randn(...).to(bf16)` would materialise an fp32 temporary first — for
            w13 that is a 25.8 GB allocation before the cast, which OOMs or thrashes. The
            expert weights are filled per-expert for the same reason.
            """
            t = torch.empty(*shape, device=dev, dtype=dt)
            if t.numel() > 2**28:  # fill big expert stacks slice by slice
                for i in range(t.shape[0]):
                    t[i].normal_(0.0, scale, generator=g)
            else:
                t.normal_(0.0, scale, generator=g)
            return t

        # --- attention output + o_proj -------------------------------------------------
        self.a_attn = rnd(T, Kq)
        self.w_o = rnd(Kq, H, scale=0.02)
        self.h_in = rnd(T, H)                      # residual entering the layer
        self.c = torch.empty(T, H, device=dev, dtype=dt)     # o_proj output (unfused)
        self.h1 = torch.empty(T, H, device=dev, dtype=dt)    # new residual
        self.acc32 = torch.zeros(T, H, device=dev, dtype=torch.float32)

        # --- norm + router -------------------------------------------------------------
        self.w_norm = (torch.randn(H, generator=g, device=dev) * 0.1 + 1).to(dt)
        self.x2 = torch.empty(T, H, device=dev, dtype=dt)
        self.wg_t = rnd(H, E, scale=0.02)          # [K, N] for the router GEMM
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

        # Routing must be derived from the value the pipeline actually produces
        # (h1 = o_proj(a_attn) + h_in), not from h_in -- otherwise the dispatch layout and
        # the routing weights do not correspond to the tokens the kernels route.
        # It is then FROZEN: routing is data-dependent but identical across configurations,
        # so freezing it keeps every config doing exactly the same work. The pipeline still
        # runs (and pays for) the router GEMM + top-k, writing into scratch buffers.
        with torch.no_grad():
            h1_ref = (self.a_attn.float() @ self.w_o.float() + self.h_in.float()).to(dt)
            x2_ref = R.rmsnorm(h1_ref, self.w_norm)
            _, tw, ti = R.router(x2_ref, self.wg_t.t(), torch.zeros(E, device=dev))
        self.topw.copy_(tw)
        self.topi.copy_(ti)
        self.tw_flat = self.topw.flatten().contiguous()
        # scratch targets for the timed router/top-k, so they cannot perturb the frozen routing
        self.topw_scratch = torch.empty_like(self.topw)
        self.topi_scratch = torch.empty_like(self.topi)

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
# Chain construction. Each option contributes kernels; only the mapping and the fusion
# flags differ, exactly as in the per-family benchmarks.
# --------------------------------------------------------------------------------------
def build_chain(p: LayerProblem, cfg: dict, sel: dict, shared_cfg: dict) -> list:
    fns = []
    L = cfg

    # ---- S3/S4  o_proj (+ ResAdd1) ----------------------------------------------------
    if sel["resadd1"] == "in_oproj":  # fusion #1
        g = L["oproj_gemm_fused"]
        if g.get("SPLIT_K", 1) > 1:
            fns += [lambda: p.acc32.zero_(),
                    lambda: KO.gemm_launch(p.a_attn, p.w_o, p.acc32, p.h_in, g, True, True),
                    lambda: KO.epilogue_launch(p.acc32, None, p.h1, L["oproj_epi_fused"] or L["oproj_epi"], False)]
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

    # ---- S5  ResAdd1 (if not already done) + post-attention RMSNorm --------------------
    if sel["resadd1"] == "in_oproj":
        fns += [lambda: KN.norm_only(p.h1, p.w_norm, p.x2, L["norm"])]
    elif sel["norm"] == "fused3":  # fusion #3
        fns += [lambda: KN.fused_add_rmsnorm(p.c, p.h_in, p.w_norm, p.h1, p.x2, L["addnorm"])]
    else:
        fns += [lambda: KN.add_only(p.c, p.h_in, p.h1, L["add"]),
                lambda: KN.norm_only(p.h1, p.w_norm, p.x2, L["norm"])]

    # ---- S6  router + top-8 (timed; writes to scratch so routing stays frozen) ---------
    fns += [lambda: KR.router_gemm(p.x2, p.wg_t, p.logits, L["router"]),
            lambda: KR.topk_only(p.logits, p.topw_scratch, p.topi_scratch, L["topk"])]

    # ---- S7/S8  w13 grouped GEMM + SwiGLU ---------------------------------------------
    if sel["gateup"] == "fused6":
        gc = L["w13_fused"]
        sti, eid, ntp = p.layout(gc["BLOCK_M"])
        fns += [lambda: KG.launch_gateup(p.x2, p.w13, p.act, sti, eid, ntp, p.rows, TOPK, I, gc, True)]
    else:
        gc = L["w13"]
        sti, eid, ntp = p.layout(gc["BLOCK_M"])
        fns += [lambda: KG.launch_gateup(p.x2, p.w13, p.inter, sti, eid, ntp, p.rows, TOPK, I, gc, False),
                lambda: KG.launch_silu_and_mul(p.inter, p.act, L["act"])]

    # ---- S9/S10/S11  w2 grouped GEMM + expert merge + ResAdd2 -------------------------
    # NOTE: `launch_down` always sets MUL_ROUTED_WEIGHT=True, i.e. the routing weight is
    # applied INSIDE the down GEMM (as in sglang). The merge that follows must therefore be
    # an UNWEIGHTED sum over top-k -- `KD.launch_moe_sum` -- not f10's `merge_only`, which
    # applies weights itself and would double-weight the result.
    d = sel["down"]
    if d in ("atomic8", "atomic9"):
        wc = L["w2_fused8"] if d == "atomic8" else L["w2_fused9"]
        sti2, eid2, ntp2 = p.layout(wc["BLOCK_M"])
        if d == "atomic8":
            fns += [lambda: p.routed.zero_(),
                    lambda: KD.launch_down(p.act, p.w2, p.routed, p.tw_flat, sti2, eid2, ntp2,
                                           p.rows, TOPK, wc, True),
                    lambda: KD.launch_resadd(p.routed, p.h1, p.out, L["resadd2"])]
        else:  # #9: seed the accumulator with the residual, so ResAdd2 costs nothing extra
            fns += [lambda: p.out.copy_(p.h1),
                    lambda: KD.launch_down(p.act, p.w2, p.out, p.tw_flat, sti2, eid2, ntp2,
                                           p.rows, TOPK, wc, True)]
    else:
        wc = L["w2"]
        sti2, eid2, ntp2 = p.layout(wc["BLOCK_M"])
        fns += [lambda: KD.launch_down(p.act, p.w2, p.y3, p.tw_flat, sti2, eid2, ntp2,
                                       p.rows, TOPK, wc, False)]
        if d == "merge_f10":  # fusion #10: merge + ResAdd2 in one kernel
            fns += [lambda: KD.launch_moe_sum(p.y3v, p.out, p.h1, TOPK, L["moe_sum"], True)]
        else:  # fully split: merge, then ResAdd2
            fns += [lambda: KD.launch_moe_sum(p.y3v, p.routed, p.h1, TOPK, L["moe_sum"], False),
                    lambda: KD.launch_resadd(p.routed, p.h1, p.out, L["resadd2"])]

    # ---- S12  shared expert (identical in every configuration) ------------------------
    sg, sa, sd = shared_cfg["w13"], shared_cfg["act"], shared_cfg["w2"]
    ssti, seid, sntp = p.shared_layout(sg["BLOCK_M"])
    ssti2, seid2, sntp2 = p.shared_layout(sd["BLOCK_M"])
    ones = shared_cfg["ones"]
    fns += [
        lambda: KG.launch_gateup(p.x2, p.w13s, p.s_inter, ssti, seid, sntp, p.T, 1, I, sg, False),
        lambda: KG.launch_silu_and_mul(p.s_inter, p.s_act, sa),
        lambda: KD.launch_down(p.s_act, p.w2s, p.s_out, ones, ssti2, seid2, sntp2, p.T, 1, sd, False),
        lambda: KD.launch_resadd(p.s_out, p.out, p.out, L["resadd2"]),
    ]
    return fns


# --------------------------------------------------------------------------------------
def tune_shared(p: LayerProblem) -> dict:
    """The shared expert is a new shape (T rows, 1 expert) -- tune it briefly."""
    ones = torch.ones(p.T, device="cuda", dtype=torch.float32)
    gemm_grid, act_grid = [], []
    # Small curated grid: the shared expert is one dense expert over T rows, and the
    # routed-expert winners already tell us the right neighbourhood. Seeded from those.
    for bm in ((16, 32, 64) if p.T <= 256 else (64, 128)):
        for bn in (64, 128):
            for bk in (32, 64):
                for w in (4, 8):
                    for s in (2, 3):
                        if s * 2 * bk * (bm + bn) > 65536:
                            continue
                        gemm_grid.append(dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                                              GROUP_M=8, num_warps=w, num_stages=s))
    down_grid = list(gemm_grid)
    for bm in (1, 4, 8):
        for bn in (512, 1024, 2048):
            for w in (4, 8):
                act_grid.append(dict(BLOCK_M=bm, BLOCK_N=bn, num_warps=w, num_stages=2))

    def gemm_chain(c):
        sti, eid, ntp = p.shared_layout(c["BLOCK_M"])
        return [lambda: KG.launch_gateup(p.x2, p.w13s, p.s_inter, sti, eid, ntp, p.T, 1, I, c, False)]

    def act_chain(c):
        return [lambda: KG.launch_silu_and_mul(p.s_inter, p.s_act, c)]

    def down_chain(c):
        sti, eid, ntp = p.shared_layout(c["BLOCK_M"])
        return [lambda: KD.launch_down(p.s_act, p.w2s, p.s_out, ones, sti, eid, ntp, p.T, 1, c, False)]

    print(f"  [shared] tuning w13 ({len(gemm_grid)}), act ({len(act_grid)}), w2 ({len(down_grid)})",
          flush=True)
    tg = autotune(gemm_chain, gemm_grid, warmup=5, rep=15)
    ta = autotune(act_chain, act_grid, warmup=5, rep=15)
    td = autotune(down_chain, down_grid, warmup=5, rep=15)
    print(f"  [shared] w13 {tg.best_ms:.4f} {tg.best_cfg} | act {ta.best_ms:.4f} | "
          f"w2 {td.best_ms:.4f} {td.best_cfg}", flush=True)
    return {"w13": tg.best_cfg, "act": ta.best_cfg, "w2": td.best_cfg, "ones": ones,
            "ms": {"w13": tg.best_ms, "act": ta.best_ms, "w2": td.best_ms}}


# Curated configuration set. The axes touch disjoint kernels, so a baseline plus
# one-axis-at-a-time variants plus the predicted-best combination identifies the optimum
# without an exhaustive product.
CONFIGS = {
    "A_all_unfused":      dict(resadd1="separate", norm="split", gateup="split", down="split"),
    "B_f3":               dict(resadd1="separate", norm="fused3", gateup="split", down="split"),
    "C_f1":               dict(resadd1="in_oproj", norm="split", gateup="split", down="split"),
    "D_f6":               dict(resadd1="separate", norm="split", gateup="fused6", down="split"),
    "E_f10":              dict(resadd1="separate", norm="split", gateup="split", down="merge_f10"),
    "F_f8":               dict(resadd1="separate", norm="split", gateup="split", down="atomic8"),
    "G_f9":               dict(resadd1="separate", norm="split", gateup="split", down="atomic9"),
    "H_f3_f10":           dict(resadd1="separate", norm="fused3", gateup="split", down="merge_f10"),
    "I_f3_f9":            dict(resadd1="separate", norm="fused3", gateup="split", down="atomic9"),
    "J_greedy_all":       dict(resadd1="in_oproj", norm="split", gateup="fused6", down="atomic9"),
    # best-of-axes: #3 on the norm axis + #8 on the down axis. Added after the first pass,
    # which showed those two winning their axes independently but never tested together.
    "K_f3_f8":            dict(resadd1="separate", norm="fused3", gateup="split", down="atomic8"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--rep", type=int, default=40)
    args = ap.parse_args()

    out_all = {}
    for regime in args.regimes.split(","):
        regime = regime.strip()
        print(f"\n===== {regime} =====", flush=True)
        p = LayerProblem(regime)
        cfg = load_cfgs(regime) if regime in FAMILY_TUNED else tune_layer_cfgs(p)
        shared = tune_shared(p)

        rows = {}
        for name, sel in CONFIGS.items():
            try:
                chain = build_chain(p, cfg, sel, shared)
                p.out.zero_(); p.routed.zero_(); p.y3.zero_()
                for fn in chain:
                    fn()
                torch.cuda.synchronize()
                got = p.out.clone()
                # against an independent fp32 reference, NOT against another config --
                # configs sharing a bug agree with each other while all being wrong
                ck = check(got, p.ref_out, tol=5e-2, label=name)
                if not ck["ok"]:
                    print(f"  {name:<18} !! FAILS CORRECTNESS relerr={ck['rel_err']:.3e}",
                          flush=True)
                t = bench_chain(chain, warmup=15, rep=args.rep)
                rows[name] = {"ms": t.p50_ms, "p10_p90": [t.p10_ms, t.p90_ms],
                              "n_kernels": len(chain), "sel": sel,
                              "rel_err_vs_fp32_ref": ck.get("rel_err", 0.0), "correct": ck["ok"]}
                print(f"  {name:<18} {t.p50_ms:9.4f} ms  ({len(chain)} kernels)  "
                      f"relerr {ck.get('rel_err', 0.0):.2e}", flush=True)
            except Exception as exc:  # noqa: BLE001
                rows[name] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                print(f"  {name:<18} FAILED {type(exc).__name__}: {exc}", flush=True)
            finally:
                torch.cuda.empty_cache()

        # a fast wrong configuration is not a candidate
        ok = {k: v for k, v in rows.items() if "ms" in v and v.get("correct")}
        if not ok:
            raise RuntimeError(f"{regime}: no configuration passed correctness")
        best = min(ok, key=lambda k: ok[k]["ms"])
        base = ok.get("A_all_unfused", {}).get("ms")
        print(f"  --> BEST {best} = {ok[best]['ms']:.4f} ms"
              + (f"  ({base / ok[best]['ms']:.3f}x vs all-unfused)" if base else ""), flush=True)
        out_all[regime] = {"rows": rows, "best": best,
                           "speedup_vs_unfused": (base / ok[best]["ms"]) if base else None,
                           "shared_expert_ms": shared["ms"],
                           "T": p.T, "oproj_K": p.Kq, "moe_rows": p.rows}
        del p
        torch.cuda.empty_cache()

    record("layer_configurations", {
        "id": "layer_configurations",
        "scope": "S3-S11 + shared expert; attention core / MLA projections / DSA indexer "
                 "excluded (untouched by every fusion candidate). MoE dispatch-layout "
                 "construction excluded from the timed region, as in every per-family bench.",
        "configs": {k: v for k, v in CONFIGS.items()},
        "regimes": out_all,
    })
    print("\nwrote results/layer_configurations.json")


if __name__ == "__main__":
    main()
