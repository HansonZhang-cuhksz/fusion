"""Analytical HBM-traffic model: the roofline ceiling for each fusion.

For a memory-bound fusion the achievable speedup is bounded by
`bytes_unfused / bytes_fused`. Measuring against that ceiling separates "the fusion
worked" from "the fusion worked *and* the kernel is bandwidth-saturated", and it makes an
under-performing fused kernel visible as a mapping problem rather than a fusion problem.

Every count below is HBM traffic in bytes for ONE invocation. Weight traffic is counted
once per distinct expert touched (weights are far larger than L2, so no reuse across CTAs);
activation re-reads that fit in L2 are annotated where they matter -- and L2 is a device
fact (8 MB on C500, 32 MB on the RTX 4060), so it comes from the profile below, not a
literal: at 32 MB every regime up to T=2048 keeps its activations resident, which makes
several of the ceilings here unattainable rather than merely unmet.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from . import config as C

SZ = 2  # bf16
FP32 = 4


def expected_distinct_experts(T: int, topk: int = C.NUM_EXPERTS_PER_TOK,
                              E: int = C.N_ROUTED_EXPERTS) -> float:
    """E[#experts receiving >=1 token] under uniform routing = E*(1-(1-1/E)^(T*topk)).

    This is what determines expert-weight HBM traffic, and it saturates fast: at T=32 it is
    already ~162 of 256, at T=256 it is 256. Real routing is less uniform, so this is a
    lower bound on weight traffic at small T and exact at large T.
    """
    n = T * topk
    return E * (1.0 - (1.0 - 1.0 / E) ** n)


# --------------------------------------------------------------------------------------
# Achievable peaks -- measured, not vendor-spec, and per device
# --------------------------------------------------------------------------------------
# `_time()` below divides by these, so they have to belong to the device whose measurements
# they are compared against. The C500 study hardcoded its two numbers right here; keeping
# them as a table row instead means reproducing the original model is
# `GLM52_DEVICE_PROFILE=c500`, and the next device is a new row plus a calibration file
# rather than an edit to the arithmetic.
#
# Compute is the best *Triton* bf16 GEMM, not the vendor BLAS: these kernels are Triton, so
# the Triton ceiling is the honest denominator. On C500 that was ~50 % of the vendor BLAS,
# and that ratio does NOT generalise -- for the grouped MoE GEMM the Triton kernel reaches
# 0.93x the vendor path at prefill_t8192 and 2.2x at decode_bs256 (where the vendor path
# pays 256 launches); on sm89 the dense gap is gone entirely (11.81 Triton vs 11.62 cuBLAS).
#
# Bandwidth is access-pattern dependent, and on both devices read-only streaming beats
# mixed read+write -- C500: 1.29 TB/s (F3 add+rmsnorm, 4 x 100 MB in 0.312 ms) against
# 1.43-1.62 TB/s (F6 MoE decode, near-pure weight reads, 7.86 GB in 4.84 ms); RTX 4060:
# 140 GB/s against 159 GB/s. We keep the conservative mixed figure because most
# memory-bound fusions modelled here are read+write vector kernels; it under-predicts
# absolute MoE decode time by ~20 %, but leaves the fused/unfused *ratios* untouched, since
# both sides of those comparisons pay the same weight traffic.
#
# The balance point C_PEAK/B_PEAK moves 82.3 -> 84.3 flop/byte, i.e. 2.5 %, so 49 of the 50
# compute_bound/memory_bound labels carry over from C500 unchanged. The one exception is
# F5_rmsnorm_router at decode_bs256, whose fused arm sits at 83.0 flop/byte -- inside the
# window, hence compute on C500 and memory here. It is within 2 % of the balance point on
# *both* devices, i.e. a knife-edge kernel rather than a changed regime; read it as
# "balanced". Everything else that moves is an absolute ms prediction, by ~10x.
@dataclass(frozen=True)
class DevicePeaks:
    """One device's roofline constants. Measured ceilings only -- never vendor spec."""

    match: str  # substring of the probed device name that selects this profile
    c_peak: float  # best Triton bf16 GEMM, flop/s
    b_peak: float  # mixed read+write stream, byte/s
    b_peak_read_only: float  # for reference; not used in the ceiling arithmetic
    l2_bytes: int
    calib: tuple[str, ...] = ()  # calibration JSONs under RESULTS_DIR; later ones win
    source: str = ""  # provenance, filled in by _resolve()


DEVICE_PEAKS: dict[str, DevicePeaks] = {
    # LOG-00 calibration sweep, recalibrated 2026-07-27 from this study's own measurements
    # (LOG-10 §4): 107 TF/s = unfused o_proj GEMM, BK32/GM8, M=8192 K=16384 N=6144.
    # (The 1.05 TB/s inherited from the earlier project was simply too low.)
    "c500": DevicePeaks("MetaX C500", 107e12, 1.30e12, 1.60e12, 8 * 2**20),
    # RTX 4060 Laptop, clocks locked at 1020 MHz SM / 5501 MHz MEM, measured 2026-07-31.
    # 11.81 TF/s = Triton bf16 GEMM, cfg 64/256/32/w8/sk4/st3 at M=4096 K=16384 N=6144.
    # The JSONs carry the exact figures; this row is what is used if they are missing.
    "rtx4060": DevicePeaks(
        "RTX 4060",
        11.81e12,
        140.0e9,
        159.0e9,
        32 * 2**20,
        calib=("device_4060_calibration.json", "rtx4060_gemm_ceiling.json"),
    ),
}
DEFAULT_PROFILE = "rtx4060"  # used when the device cannot be probed (CPU-only checkout)


def _results_dir() -> Path:
    """Where the calibration JSONs live. Imported lazily: this model is pure arithmetic and
    must stay importable on a box with no GPU (and without paying for a CUDA context)."""
    try:
        from .common import RESULTS_DIR

        return RESULTS_DIR
    except Exception:
        return Path(__file__).resolve().parent.parent / "results"


def _overlay(p: DevicePeaks, d: dict) -> DevicePeaks:
    """Apply one calibration file to a profile. Two shapes are understood: the device
    calibration (`measured.{gemm_bf16_TFs,bw_stream_rw_GBs,bw_read_only_GBs}`, `l2_bytes`)
    and the dedicated GEMM ceiling sweep (`triton_bf16_TFs`), which is the denser sweep and
    therefore supersedes the coarse GEMM number when both are present."""
    m = d.get("measured", {})
    tf = d.get("triton_bf16_TFs", m.get("gemm_bf16_TFs"))
    rw, ro, l2 = m.get("bw_stream_rw_GBs"), m.get("bw_read_only_GBs"), d.get("l2_bytes")
    return replace(
        p,
        c_peak=tf * 1e12 if tf else p.c_peak,
        b_peak=rw * 1e9 if rw else p.b_peak,
        b_peak_read_only=ro * 1e9 if ro else p.b_peak_read_only,
        l2_bytes=int(l2) if l2 else p.l2_bytes,
    )


def _resolve() -> DevicePeaks:
    """This device's profile: name match (or `GLM52_DEVICE_PROFILE`), then the JSONs.

    Order is table row -> calibration files -> live probe, each overriding the last, so a
    checkout with no results/ and no GPU still prints a self-consistent table. Only L2 comes
    from the probe: the peaks are measurements, and no device reports them.
    """
    pin = os.environ.get("GLM52_DEVICE_PROFILE", "")
    e = None
    if pin not in DEVICE_PEAKS:
        try:
            e = C.env()
        except Exception:  # no GPU, or a degraded probe -- fall back to the named default
            e = None
        name = (e.device_name if e else "").lower()
        pin = next(
            (k for k, q in DEVICE_PEAKS.items() if q.match.lower() in name), DEFAULT_PROFILE
        )
    p = DEVICE_PEAKS[pin]
    src = [pin]
    for fname in p.calib:
        try:
            d = json.loads((_results_dir() / fname).read_text())
        except Exception:
            continue
        p = _overlay(p, d)
        src.append(fname)
    if e is not None:
        p = replace(p, l2_bytes=e.l2_bytes)
        src.append("probe")
    return replace(p, source=" + ".join(src))


PEAKS = _resolve()
C_PEAK = PEAKS.c_peak
B_PEAK = PEAKS.b_peak
B_PEAK_READ_ONLY = PEAKS.b_peak_read_only  # for reference; not in the ceiling arithmetic
L2_BYTES = PEAKS.l2_bytes


@dataclass
class Traffic:
    """A fusion's cost, as a per-kernel (flops, bytes) breakdown of each side.

    The traffic-only ratio `unfused_bytes/fused_bytes` is NOT the achievable speedup
    ceiling whenever any kernel in the chain is compute-bound. The o_proj prefill case is
    the clean example (C500 numbers, but both terms scale together): the fusion removes 23 %
    of the bytes, but the GEMM needs 1649 GFLOP (~18 ms at 107 TF/s) against 0.13 ms of
    saved traffic -- a 0.7 % ceiling, not 1.30x.
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
            note=f"router weight {wg/2**20:.1f} MB fits the {L2_BYTES/2**20:.0f} MB L2, so "
                 f"its re-reads are not HBM",
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
    # provenance: an absolute ms here is meaningless without the peaks it was divided by
    print(
        f"peaks [{PEAKS.source}]: {C_PEAK/1e12:.2f} TF/s compute, {B_PEAK/1e9:.0f} GB/s "
        f"mixed r+w, balance {C_PEAK/B_PEAK:.1f} flop/byte, L2 {L2_BYTES/2**20:.0f} MB"
    )
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
