"""Campaign-v2 #11a local verification harness (sm_89 / RTX 4060).

The sm_90 defect never reproduced on sm_89, so this script cannot fail or pass the fix on
its own; it exists to establish the three things the H200 probe will then judge:

  1. CORRECTNESS -- every SQ_MODE in {0,1,2,3,4} agrees with an exact fp32 reference on
     the real router-derived dispatch (this catches a botched transposed load before it
     ever ships to the H200).
  2. INVARIANCE -- for every mapping knob in the published INVARIANT_CFG_KEYS, a partner
     config must agree with the base config.  On sm_89 the modes are expected to pass;
     a failure here is still a hard failure (it means the new branch is not
     value-invariant, and then it has no business on the H200).
  3. REPEAT -- same config, 3 fresh NaN-armed launches: the kernel must agree with
     itself bit-for-bit (tol=0).  This is the race-vs-miscompile discriminator the
     campaign never ran.

The w13 family runs with H shrunk to 512 (E=256 kept, real top-8 dispatch), because the
real [E, 4096, 6144] bf16 weights are 12.9 GB -- the 4060 has 8 GB.  The router family
runs at full H=6144, K=512 and K=6144 respectively; BLOCK_K <= 128 divides both, so
`even_Ks` is true everywhere and the audited path is the one exercised.

Exit code 0 iff every cell is ok.
"""
from __future__ import annotations

import itertools
import sys

import torch

from glm52_h200 import config as C
from glm52_h200 import reference as R
from glm52_h200.kernels import lazy_prenorm as K

DT = C.DTYPE
EPS = C.RMS_NORM_EPS
TOPK = C.NUM_EXPERTS_PER_TOK
E = C.N_ROUTED_EXPERTS
H = C.HIDDEN_SIZE          # 6144, full K for the router family
H2 = 512                   # shrunk K for the w13 family (fits the 4060)
I2 = 2 * H2                # 1024, keeps the 2I/H ratio of the real tensor
PAD = 1 << 20              # bench-style speculative-load pad on the w13 buffers

T = 512                    # tokens; rows = T*TOPK = 4096

#: tolerance for the correctness screen (rel to max|ref|), same order as the campaign's
CORR_TOL = 2e-2
#: tolerance for the invariance screen -- dtype-aware (LOG-17): 1e-5 for fp32 outputs
#: (router; its last-ulp class is 1.8e-7), 2e-2 for bf16 outputs (moe; a repartitioned
#: reduce moves the fp32 partials by a last ulp and one bf16 rounding-boundary flip then
#: reads up to ~2^-7 relative; measured 8.6e-4..1.7e-3; the defect class is 0.37+).
ROUTER_INV_TOL = 1e-5
MOE_INV_TOL = 2e-2

#: configs cover both sides of the wgmma boundary (BLOCK_M < 64 and >= 64) since that is
#: where campaign 1's defect lived
CFGS = [
    dict(BLOCK_M=16,  BLOCK_N=64,  BLOCK_K=64,  GROUP_M=8, num_warps=4, num_stages=2),
    dict(BLOCK_M=32,  BLOCK_N=64,  BLOCK_K=64,  GROUP_M=8, num_warps=4, num_stages=2),
    dict(BLOCK_M=64,  BLOCK_N=64,  BLOCK_K=64,  GROUP_M=8, num_warps=4, num_stages=2),
    dict(BLOCK_M=64,  BLOCK_N=128, BLOCK_K=32,  GROUP_M=8, num_warps=4, num_stages=2),
    dict(BLOCK_M=64,  BLOCK_N=256, BLOCK_K=64,  GROUP_M=8, num_warps=8, num_stages=2),
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,  GROUP_M=8, num_warps=8, num_stages=2),
    dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=64,  GROUP_M=8, num_warps=8, num_stages=3),
    dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=64,  GROUP_M=8, num_warps=8, num_stages=4),
]
MODES = (0, 1, 2, 3, 4)


class Problem:
    """Mirror of the bench's Problem with H2-shrunk w13 (E and dispatch kept real)."""

    def __init__(self):
        torch.manual_seed(4242)
        self.h1 = (torch.randn(T, H, device="cuda", dtype=torch.float32) * 0.5).to(DT)
        self.w = (torch.randn(H, device="cuda") * 0.02 + 1.0).to(DT)
        self.gate = (torch.randn(E, H, device="cuda") * 0.01).to(DT)
        self.x2 = R.rmsnorm(self.h1, self.w, EPS).contiguous()
        _, _, self.topk_ids = R.router(self.x2, self.gate)
        self.rows = T * TOPK
        self.has_w13 = True

        # shrunk w13 family: raw + fold twins with the bench's speculative-load pad
        numel = E * I2 * H2
        raw_buf = torch.empty(numel + PAD, device="cuda", dtype=DT)
        fold_buf = torch.empty(numel + PAD, device="cuda", dtype=DT)
        raw = raw_buf[:numel].view(E, I2, H2)
        fold = fold_buf[:numel].view(E, I2, H2)
        wf = self.w[:H2].float()
        for e in range(E):
            raw[e].normal_(0, 0.02)
            fold[e] = (raw[e].float() * wf).to(DT)
        raw_buf[numel:].zero_()
        fold_buf[numel:].zero_()
        self.w13_raw_buf, self.w13_raw, self.w13_fold_buf, self.w13_fold = (
            raw_buf, raw, fold_buf, fold)

        self.h1_m = self.h1[:, :H2].contiguous()
        self.w_m = self.w[:H2]
        self.x2_m = R.rmsnorm(self.h1_m, self.w_m, EPS).contiguous()

        self.logits = torch.zeros(T, E, device="cuda", dtype=torch.float32)
        self.c = torch.zeros(self.rows, I2, device="cuda", dtype=DT)
        self.layouts: dict[int, tuple] = {}

    def layout(self, block_m: int):
        if block_m not in self.layouts:
            self.layouts[block_m] = R.moe_align_block_size(self.topk_ids, block_m, E)
        return self.layouts[block_m]

    def router(self, cfg, sq_mode, out):
        return lambda: K.launch_router(self.h1, (self.gate.float() * self.w.float()).to(DT).t().contiguous(), out,
                                       cfg, True, EPS, sq_mode)

    def moe(self, cfg, sq_mode, out):
        sti, eids, ntp = self.layout(cfg["BLOCK_M"])
        return lambda: K.launch_moe_gateup(
            self.h1_m, self.w13_fold, out, sti, eids, ntp, self.rows, TOPK,
            cfg, True, EPS, sq_mode)

    def ref_router(self):
        return self.x2.float() @ self.gate.float().t()

    def ref_moe(self):
        tok = torch.arange(self.rows, device="cuda") // TOPK
        kk = torch.arange(self.rows, device="cuda") % TOPK
        experts = self.topk_ids.long()[tok, kk]
        ref = torch.empty(self.rows, I2, device="cuda", dtype=torch.float32)
        xs = self.x2_m.float()[tok]
        for e in torch.unique(experts).tolist():
            sel = (experts == e).nonzero(as_tuple=True)[0]
            ref[sel] = xs[sel] @ self.w13_raw[e].float().t()
        return ref


def run_once(fn, out):
    out.fill_(float("nan"))
    fn()


def check_correctness(out, ref):
    a = out.detach().float()
    if a.shape != ref.shape:
        return f"shape {tuple(a.shape)} vs ref {tuple(ref.shape)}"
    if int(torch.isnan(a).sum()):
        return f"{int(torch.isnan(a).sum())} NaN elems (unwritten rows: "
        f"{int(torch.isnan(a).any(-1).sum())})"
    scale = float(ref.abs().max())
    rel = float((a - ref).abs().max()) / scale
    return None if rel <= CORR_TOL else f"max_rel_err {rel:.3e} > {CORR_TOL}"


def main():
    assert torch.cuda.is_available(), "no CUDA"
    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device: {dev}  sm_{cap[0]}{cap[1]}  torch {torch.__version__}  "
          f"triton {torch.__version__.split('+')[0]}")
    print(f"T={T} H={H} H2={H2} E={E} TOPK={TOPK} EPS={EPS}\n")

    prob = Problem()
    refs = {"router": prob.ref_router(), "moe": prob.ref_moe()}

    n_fail = 0
    for fam, mk, out0, ref, inv_tol in (
        ("router", prob.router, prob.logits, refs["router"], ROUTER_INV_TOL),
        ("moe", prob.moe, prob.c, refs["moe"], MOE_INV_TOL),
    ):
        for cfg, mode in itertools.product(CFGS, MODES):
            out = torch.empty_like(out0)
            tag = f"{fam:6s} m{mode} BM{cfg['BLOCK_M']:3d} BN{cfg['BLOCK_N']:3d} " \
                  f"BK{cfg['BLOCK_K']:3d} w{cfg['num_warps']} s{cfg['num_stages']}"
            try:
                # 1. correctness
                run_once(mk(cfg, mode, out), out)
                err = check_correctness(out, ref)
                if err:
                    print(f"FAIL {tag}  correctness: {err}")
                    n_fail += 1
                    continue

                # 2. invariance across mapping keys (partner config must agree)
                for key in K.INVARIANT_CFG_KEYS + K.INVARIANT_CFG_KEYS_ULP:
                    partner = K.invariance_partner(cfg, key)
                    if partner is None:
                        continue
                    out2 = torch.empty_like(out)
                    try:
                        run_once(mk(partner, mode, out2), out2)
                    except Exception as exc:  # noqa: BLE001 -- legality is the caller's
                        continue
                    v = K.invariance_verdict(out, out2, tol=inv_tol)
                    if not v["ok"]:
                        print(f"FAIL {tag}  invariance key={key}: {v.get('reason')} "
                              f"[max_rel={v.get('max_rel_diff')} bit_exact={v.get('bit_exact')}]")
                        n_fail += 1

                # 3. repeat: 3 fresh NaN-armed launches must agree bit-for-bit
                outs = [torch.empty_like(out) for _ in range(3)]
                for o in outs:
                    run_once(mk(cfg, mode, o), o)
                v = K.repeat_verdict(outs, tol=0.0)
                if not v.get("ok", False):
                    print(f"FAIL {tag}  repeat: {v.get('reason')}")
                    n_fail += 1
                else:
                    print(f"ok   {tag}")
            except Exception as exc:  # noqa: BLE001 -- OOM etc. is a skip, not a pass
                print(f"SKIP {tag}  {type(exc).__name__}: {str(exc)[:100]}")

    print(f"\n{'ALL CELLS OK' if n_fail == 0 else f'{n_fail} FAILED CELLS'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
