"""Analytical HBM-traffic model: the roofline ceiling for each fusion.

For a memory-bound fusion the achievable speedup is bounded by
`bytes_unfused / bytes_fused`. Measuring against that ceiling separates "the fusion
worked" from "the fusion worked *and* the kernel is bandwidth-saturated", and it makes an
under-performing fused kernel visible as a mapping problem rather than a fusion problem.

Every count below is HBM traffic in bytes for ONE invocation. Weight traffic is counted
once per distinct expert touched (weights are far larger than L2, so no reuse across CTAs);
activation re-reads that fit in the 8 MB L2 are annotated where they matter.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config as C

SZ = 2  # bf16
FP32 = 4
L2_BYTES = 8 * 2**20


def expected_distinct_experts(T: int, topk: int = C.NUM_EXPERTS_PER_TOK,
                              E: int = C.N_ROUTED_EXPERTS) -> float:
    """E[#experts receiving >=1 token] under uniform routing = E*(1-(1-1/E)^(T*topk)).

    This is what determines expert-weight HBM traffic, and it saturates fast: at T=32 it is
    already ~162 of 256, at T=256 it is 256. Real routing is less uniform, so this is a
    lower bound on weight traffic at small T and exact at large T.
    """
    n = T * topk
    return E * (1.0 - (1.0 - 1.0 / E) ** n)


# Achievable peaks on C500, measured (see LOG-00 calibration sweep), not vendor-spec:
#   compute: best Triton bf16 GEMM = 106 TF/s (the vendor BLAS reaches 215, but these
#            kernels are Triton, so the Triton ceiling is the honest denominator)
#   bandwidth: ~1.05 TB/s achievable HBM
# Recalibrated 2026-07-27 from this study's own measurements (see LOG-10 §4).
#
# Compute: 107.5 TF/s -- unfused o_proj GEMM, BK32/GM8, M=8192 K=16384 N=6144.
#   NB this is the *dense* Triton GEMM ceiling, ~50 % of the vendor BLAS. That ratio does
#   NOT generalise: for the grouped MoE GEMM the Triton kernel reaches 0.93x the vendor path
#   at prefill_t8192 and 2.2x at decode_bs256 (where the vendor path pays 256 launches).
#
# Bandwidth is access-pattern dependent, and the two measurements we have differ by 25 %:
#   1.29 TB/s -- F3 add+rmsnorm, mixed read+write, 4 x 100 MB in 0.312 ms
#   1.43-1.62 TB/s -- F6 MoE decode, near-pure weight *reads* (7.86 GB in 4.84 ms)
# Read-only streaming beats mixed read/write, so both are real. We keep the conservative
# mixed figure because most memory-bound fusions modelled here are read+write vector
# kernels; it under-predicts absolute MoE decode time by ~20 %, but leaves the fused/unfused
# *ratios* untouched, since both sides of those comparisons pay the same weight traffic.
# (The 1.05 TB/s inherited from the earlier project was simply too low.)
C_PEAK = 107e12
B_PEAK = 1.30e12
B_PEAK_READ_ONLY = 1.60e12  # for reference; not used in the ceiling arithmetic


@dataclass
class Traffic:
    """A fusion's cost, as a per-kernel (flops, bytes) breakdown of each side.

    The traffic-only ratio `unfused_bytes/fused_bytes` is NOT the achievable speedup
    ceiling whenever any kernel in the chain is compute-bound. The o_proj prefill case is
    the clean example: the fusion removes 23 % of the bytes, but the GEMM needs 1649 GFLOP
    (~18 ms at 106 TF/s) against 0.13 ms of saved traffic -- a 0.7 % ceiling, not 1.30x.
    So the ceiling reported here is latency-aware: each kernel costs
    `max(flops/C_PEAK, bytes/B_PEAK)`, and a chain costs the sum over its kernels.
    """

    fusion: str
    regime: str
    # each entry is one kernel launch: (flops, bytes)
    unfused_kernels: list[tuple[float, float]]
    fused_kernels: list[tuple[float, float]]
    note: str = ""

    @staticmethod
    def _time(kernels: list[tuple[float, float]]) -> float:
        return sum(max(f / C_PEAK, b / B_PEAK) for f, b in kernels)

    @property
    def unfused_bytes(self) -> float:
        return sum(b for _, b in self.unfused_kernels)

    @property
    def fused_bytes(self) -> float:
        return sum(b for _, b in self.fused_kernels)

    @property
    def traffic_ratio(self) -> float:
        return self.unfused_bytes / self.fused_bytes if self.fused_bytes else float("nan")

    @property
    def ceiling(self) -> float:
        """Latency-aware achievable-speedup ceiling."""
        tf = self._time(self.fused_kernels)
        return self._time(self.unfused_kernels) / tf if tf else float("nan")

    def row(self) -> dict:
        return {
            "fusion": self.fusion,
            "regime": self.regime,
            "unfused_MB": self.unfused_bytes / 2**20,
            "fused_MB": self.fused_bytes / 2**20,
            "traffic_ratio": self.traffic_ratio,
            "unfused_ms": self._time(self.unfused_kernels) * 1e3,
            "fused_ms": self._time(self.fused_kernels) * 1e3,
            "roofline_ceiling": self.ceiling,
            "compute_bound": self._time(self.fused_kernels) * 1e3 > 0
            and sum(f for f, _ in self.fused_kernels) / C_PEAK
            > sum(b for _, b in self.fused_kernels) / B_PEAK,
            "note": self.note,
        }


def model(regime: C.Regime) -> list[Traffic]:
    T = regime.T
    H = C.HIDDEN_SIZE
    I = C.MOE_INTERMEDIATE_SIZE
    k = C.NUM_EXPERTS_PER_TOK
    Er = C.N_ROUTED_EXPERTS
    R = T * k  # rows entering the expert GEMMs
    K = regime.oproj_k
    act = T * H * SZ  # one full activation tensor
    ne = expected_distinct_experts(T)

    out: list[Traffic] = []
    wg = H * Er * SZ  # router weight, 3.0 MB
    logits = T * Er * FP32
    w13 = ne * C.W13_N * H * SZ
    w2 = ne * H * I * SZ

    # --- #1 o_proj + ResAdd ------------------------------------------------------------
    f_oproj = 2.0 * T * K * H
    b_oproj_in = T * K * SZ + K * H * SZ
    out.append(
        Traffic(
            "F1_oproj_resadd",
            regime.name,
            unfused_kernels=[(f_oproj, b_oproj_in + act), (0.0, 3 * act)],
            fused_kernels=[(f_oproj, b_oproj_in + act + act)],
            note=f"o_proj weight alone is {K*H*SZ/2**20:.0f} MB; at prefill the GEMM is "
                 f"compute-bound so the byte saving barely shows in latency",
        )
    )

    # --- #3 ResAdd + RMSNorm -----------------------------------------------------------
    out.append(
        Traffic(
            "F3_resadd_rmsnorm",
            regime.name,
            unfused_kernels=[(0.0, 3 * act), (0.0, 2 * act)],
            fused_kernels=[(0.0, 4 * act)],
            note="both h1 (new residual) and x2 (normed) are live downstream",
        )
    )

    # --- #4 / #5 Norm(+Add) + Router ---------------------------------------------------
    f_router = 2.0 * T * H * Er
    out.append(
        Traffic(
            "F5_rmsnorm_router",
            regime.name,
            unfused_kernels=[(0.0, 2 * act), (f_router, act + wg + logits)],
            fused_kernels=[(f_router, 2 * act + wg + logits)],
            note=f"router weight {wg/2**20:.1f} MB fits the 8 MB L2, so its re-reads are not HBM",
        )
    )
    out.append(
        Traffic(
            "F4_addnorm_router",
            regime.name,
            unfused_kernels=[(0.0, 3 * act), (0.0, 2 * act), (f_router, act + wg + logits)],
            fused_kernels=[(f_router, 4 * act + wg + logits)],
            note="saves the router's read of x2 on top of the F3 saving",
        )
    )

    # --- #6 Up_Gate + SwiGLU -----------------------------------------------------------
    f_gemm1 = 2.0 * R * H * C.W13_N
    out.append(
        Traffic(
            "F6_upgate_swiglu",
            regime.name,
            unfused_kernels=[
                (f_gemm1, R * H * SZ + w13 + R * 2 * I * SZ),
                (0.0, R * 2 * I * SZ + R * I * SZ),
            ],
            fused_kernels=[(f_gemm1, R * H * SZ + w13 + R * I * SZ)],
            note=f"~{ne:.0f}/{Er} experts touched -> {w13/2**20:.0f} MB of w13 weight",
        )
    )

    # --- #8 / #9 Down + Merge (+ ResAdd2) ----------------------------------------------
    f_gemm2 = 2.0 * R * I * H
    down_in = R * I * SZ
    out.append(
        Traffic(
            "F8_down_merge",
            regime.name,
            unfused_kernels=[(f_gemm2, down_in + w2 + R * H * SZ), (0.0, R * H * SZ + act)],
            fused_kernels=[(f_gemm2, down_in + w2 + act)],
            note="token-major variant (no atomics); the atomic variant pays fp32 RMW instead",
        )
    )
    out.append(
        Traffic(
            "F9_down_merge_resadd",
            regime.name,
            unfused_kernels=[
                (f_gemm2, down_in + w2 + R * H * SZ),
                (0.0, R * H * SZ + act),
                (0.0, 3 * act),
            ],
            fused_kernels=[(f_gemm2, down_in + w2 + act + act)],
            note="ResAdd2 seeds the accumulator; ~free on top of F8",
        )
    )

    # --- #10 Expert Merge + ResAdd -----------------------------------------------------
    out.append(
        Traffic(
            "F10_merge_resadd",
            regime.name,
            unfused_kernels=[(0.0, k * act + act), (0.0, 3 * act)],
            fused_kernels=[(0.0, k * act + act + act)],
            note=f"top-k input ({k}x) dominates, capping the ceiling at {(k+4)/(k+2):.2f}x",
        )
    )

    # --- #11 Lazy Pre-Norm -------------------------------------------------------------
    out.append(
        Traffic(
            "F11a_prenorm_w13",
            regime.name,
            unfused_kernels=[(0.0, 2 * act), (f_gemm1, R * H * SZ + w13 + R * I * SZ)],
            fused_kernels=[(f_gemm1, R * H * SZ + w13 + R * I * SZ)],
            note="fused reads h1 directly; x2 never materialized (valid only if all "
                 "consumers are fused). Sum-of-squares is recomputed by every n-tile.",
        )
    )
    out.append(
        Traffic(
            "F11b_prenorm_router",
            regime.name,
            unfused_kernels=[(0.0, 2 * act), (f_router, act + wg + logits)],
            fused_kernels=[(f_router, act + wg + logits)],
            note="N=256 -> ~1-2 n_tiles, so sum-of-squares redundancy is ~1x (ideal case)",
        )
    )
    return out


def all_rows() -> list[dict]:
    regimes = [r for r in C.ALL_REGIMES if r.T in (1, 32, 256, 2048, 8192)]
    return [t.row() for r in regimes for t in model(r)]


if __name__ == "__main__":
    rows = all_rows()
    w = max(len(r["fusion"]) for r in rows)
    cur = None
    for r in rows:
        if r["regime"] != cur:
            cur = r["regime"]
            print(f"\n=== {cur} ===")
            print(
                f"{'fusion':<{w}}  {'traffic':>8}  {'unfused ms':>10}  {'fused ms':>9}  "
                f"{'CEILING':>8}  bound"
            )
        print(
            f"{r['fusion']:<{w}}  {r['traffic_ratio']:>7.2f}x  {r['unfused_ms']:>10.4f}  "
            f"{r['fused_ms']:>9.4f}  {r['roofline_ceiling']:>7.2f}x  "
            f"{'compute' if r['compute_bound'] else 'memory'}"
        )
