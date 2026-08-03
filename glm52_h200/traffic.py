"""Analytical HBM-traffic model: the roofline ceiling for each fusion.

For a memory-bound fusion the achievable speedup is bounded by
`bytes_unfused / bytes_fused`. Measuring against that ceiling separates "the fusion
worked" from "the fusion worked *and* the kernel is bandwidth-saturated", and it makes an
under-performing fused kernel visible as a mapping problem rather than a fusion problem.

Every count below is HBM traffic in bytes for ONE invocation. Weight traffic is counted
once per distinct expert touched (weights are far larger than L2, so no reuse across CTAs);
activation re-reads that fit in L2 are annotated where they matter -- and L2 is a device
fact (8 MB on C500, 32 MB on the RTX 4060, **62914560 B = 60 MiB measured on the H200**), so
it comes from the probe or the profile below, never from a literal.

**On the H200 that L2 caveat is the dominant one.** One [T,6144] bf16 activation is 12.6 MB
at T=1024 and 25 MB at T=2048, so the whole traffic of the vector fusions (F3/F4/F5/F11b) is
L2-resident through decode_bs512 and still is for the two-tensor chains at decode_bs1024:
what the "saved" HBM traffic buys there is L2 traffic, and the HBM ceilings printed for those
cells are unattainable rather than merely unmet. `row()` computes this per cell as
`l2_resident` rather than asserting it per regime -- F10 reads topk=8 activations and leaves
L2 already at decode_bs512, while F5 is still inside it at bs1024. Read a flagged cell as
"the model does not bound this measurement". The prefill regimes (100 MB per activation at
t8192) are where the roofline is unambiguously an HBM story on this device.

One coincidence worth knowing before it is mistaken for a bug: F3's unfused chain at
decode_bs1024 is 5 x 12582912 = 62914560 B, which is the H200's L2 to the byte. The `<=` in
`l2_resident` therefore decides that single cell, and a 1-token change in either direction
flips it. It is a boundary, not a cliff -- treat bs1024 vector rows as "partly resident"
whichever way the flag lands.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from . import config as C

SZ = 2  # bf16
FP32 = 4

HERE = Path(__file__).resolve().parent


def expected_distinct_experts(T: int, topk: int = C.NUM_EXPERTS_PER_TOK,
                              E: int = C.N_ROUTED_EXPERTS) -> float:
    """E[#experts receiving >=1 token] under uniform routing = E*(1-(1-1/E)^(T*topk)).

    This is what determines expert-weight HBM traffic, and it saturates fast: at T=32 it is
    already ~162 of 256, at T=256 it is 256. Real routing is less uniform, so this is a
    lower bound on weight traffic at small T and exact at large T. Note that the H200 is
    the first device in the study where all 256 experts are resident, so at T>=256 the
    model's `ne == E` is what actually happens rather than an untestable extrapolation.
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
# The H200 measurement settles the Hopper question: **788.35 TF/s Triton against 821.64 TF/s
# cuBLAS at M4096/K16384/N6144 -- 96 %**, so the gap does NOT reopen the way we expected, and
# C_PEAK is 788.35 TF/s. (At M8192 Triton falls to 681.80 against cuBLAS's 796.64, i.e. 86 %;
# `_best` takes the max over shapes because a *peak* is a peak, and the M8192 shortfall is a
# Triton tiling limitation to be measured, not a lower ceiling to be charged for.) Using
# cuBLAS's 821.64 as the denominator instead would deflate every utilization number in the
# study by 4 % against a kernel none of these benches can call, which is why `c_peak_vendor`
# is carried alongside but never enters the arithmetic.
#
# Bandwidth is access-pattern dependent, and on every device measured so far read-only
# streaming beats mixed read+write -- C500: 1.29 TB/s (F3 add+rmsnorm, 4 x 100 MB in
# 0.312 ms) against 1.43-1.62 TB/s (F6 MoE decode, near-pure weight reads, 7.86 GB in
# 4.84 ms); RTX 4060: 139 GB/s copy against 161 GB/s read-only; **H200: 4234-4256 GB/s
# copy/rmw against 4584-4650 GB/s read-only, a 9 % spread**. We keep the conservative
# mixed figure because most memory-bound fusions modelled here are read+write vector
# kernels: they read an activation and write an activation, and a peak measured on a
# read-only stream is a peak those kernels cannot reach. It under-predicts absolute MoE
# decode time by ~20 % (that chain is nearly pure weight reads), but leaves the
# fused/unfused *ratios* untouched, since both sides of those comparisons pay the same
# weight traffic. The read-only figure is carried for reference only.
#
# What moves between devices is the balance point C_PEAK/B_PEAK: 82.3 flop/byte on C500,
# 84.3 on the 4060 (2.5 % apart, so 49 of 50 compute/memory labels carried over unchanged),
# and **185.2 on the measured H200** (788.35e12 / 4256.13e9). That is 2.2x the previous two,
# which agreed with each other -- so unlike the C500 -> 4060 step, the compute/memory labels
# do NOT carry over. Three consequences, and they are the whole reason this study is worth
# re-running on Hopper rather than extrapolating:
#
#   * Every cell whose arithmetic intensity sits between ~84 and ~185 flop/byte was
#     compute-bound on both previous devices and is **memory**-bound here. The H200 has
#     ~6.6x the C500's bandwidth and ~7.4x its compute; the extra compute has to come out of
#     somewhere, and it comes out of the label.
#   * Memory-bound *vector* fusions (F3/F4/F5/F10/F11b -- the ones whose whole job is to not
#     re-touch an activation) are therefore relatively MORE valuable here: a byte saved buys
#     2.2x more compute-time than it did on the C500, and there is no compute in those
#     kernels to displace.
#   * Fusions that inject work into a GEMM mainloop (F1 o_proj+ResAdd, F6 up_gate+SwiGLU,
#     F8/F9 down+merge, F11a) are relatively MORE costly: the mainloop they interrupt is now
#     worth 185 flop per byte of traffic they save, so an epilogue that costs the GEMM even a
#     few percent of its tensor-core occupancy can be a net loss at a byte ratio that looked
#     comfortably profitable at 84. Read those rows' `compute_bound` flag before their
#     `traffic_ratio`.
#
# That is a real device difference, not a modelling change, and the printed balance point is
# the number to quote when a label disagrees with LOG-09.
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
    c_peak_vendor: float = 0.0  # cuBLAS bf16 GEMM; reference only, never the denominator
    launch_s: float = 0.0  # measured kernel launch cost, for `ceiling_with_launch`
    timer_tick_s: float = 0.0  # CUDA-event granularity, for flagging unresolvable cells
    from_calibration: bool = False  # True only if a device-matched calibration was applied
    estimated: bool = False  # True if the row's numbers were never measured on hardware
    measured_on: str = ""  # which physical device/run the row's numbers came from
    # Peaks and timings degrade differently under a co-tenant: a peak is a throughput over
    # ~GB and survives, a per-launch cost is a sub-10-us quantity measured by sampling and
    # does not. These two say whether `launch_s` / `timer_tick_s` may be quoted -- see
    # `config.calibration_health()`. False is not "missing": the numbers are still here.
    launch_trusted: bool = True
    timer_tick_trusted: bool = True
    calib_note: str = ""  # why, when either of the two above is False


DEVICE_PEAKS: dict[str, DevicePeaks] = {
    # LOG-00 calibration sweep, recalibrated 2026-07-27 from this study's own measurements
    # (LOG-10 4): 107 TF/s = unfused o_proj GEMM, BK32/GM8, M=8192 K=16384 N=6144.
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
    # H200 (sm_90), MEASURED 2026-08-03 on GPU b2318e71-edbf-aa1b-dcf3-3cc3e6ea7db0 of an
    # 8-GPU node, torch 2.11.0+cu130 / triton 3.6.0. These four replace the estimates this
    # row carried before the hardware existed on this side (600 TF/s, 3.40/4.00 TB/s, 50 MB
    # -- compute was under by 31 %, bandwidth by 25 %, and the estimated *balance point* of
    # 176 was the one thing that came out close).
    #   c_peak    788.35 TF/s -- best Triton bf16 GEMM, BM128/BN256/BK64/GM8/w8/s3 at
    #                            M4096 K16384 N6144. 96 % of the 821.64 TF/s cuBLAS on the
    #                            same shape. NOT the vendor figure; see the block above.
    #   b_peak    4256.13 GB/s-- best mixed read+write stream (rmw, 2048 MB buffer).
    #   b_peak_ro 4650.42 GB/s-- best read-only stream. Reference only.
    #   l2        62914560 B  -- probed, exact. (Not 50 MB, and not a round 64 MiB either.)
    # This row is the FALLBACK: it is what the model uses when no preflight JSON is present,
    # e.g. analysing a returned result file on a laptop. It is a measurement of *an* H200,
    # not of *this* H200, so `from_calibration` stays False until the JSON overlay confirms
    # the device -- which is what `require_measured_peaks()` gates on. `estimated` is now
    # False because these are no longer guesses.
    "h200": DevicePeaks(
        "H200",
        788.3463741284407e12,
        4256.129063005892e9,
        4650.420562966606e9,
        62914560,
        estimated=False,
        c_peak_vendor=821.6363077328735e12,
        measured_on="NVIDIA H200 b2318e71 (132 SM), preflight 2026-08-03 15:14:07, "
                    "torch 2.11.0+cu130 / triton 3.6.0",
        # launch_s / timer_tick_s are deliberately left at 0 rather than transcribed: the
        # only H200 measurement of them was taken next to a 51 GB co-tenant (launch 8.89 us
        # but harness floor 40.55 us, tick matching 3 % of samples). 0 makes
        # `ceiling_with_launch` equal `ceiling`, which is the documented "never silently
        # invent a difference" behaviour. A device-matched preflight overlay supplies them --
        # flagged -- and re-running preflight on an idle GPU makes them real.
        launch_trusted=False,
        timer_tick_trusted=False,
        calib_note="no launch/tick figure in this fallback row; the only H200 measurement of "
                   "them was taken on a contended GPU. Re-run preflight.py with --gpu auto.",
    ),
}
DEFAULT_PROFILE = "h200"  # used when the device cannot be probed (CPU-only checkout)

def _results_dir() -> Path:
    """Where the legacy calibration JSONs live. Imported lazily: this model is pure
    arithmetic and must stay importable on a box with no GPU (and without paying for a
    CUDA context)."""
    try:
        from .common import RESULTS_DIR

        return RESULTS_DIR
    except Exception:
        return Path(__file__).resolve().parent.parent / "results"


_PREFLIGHT_CACHE: "dict | None" = None
_PREFLIGHT_ERROR: str = ""


def preflight_path() -> Path:
    """Where the preflight JSON lives. `config.PREFLIGHT_PATH` is the suite's one answer
    (and its `GLM52_H200_PREFLIGHT` override); the literal below only exists so this module
    keeps working if it is ever read without the rest of the package -- it is not a second
    convention, and it must not become one."""
    p = getattr(C, "PREFLIGHT_PATH", None)
    if p:
        return Path(p)
    return Path(
        os.environ.get("GLM52_H200_PREFLIGHT", str(HERE / "preflight_h200.json"))
    )


def preflight() -> dict | None:
    """The preflight JSON, parsed once, or None if it is absent/unreadable.

    Delegates to `config.load_preflight()` when it exists, so the whole suite reads one
    file through one code path with one cache -- a second reader would be a second place
    for the device fence to be forgotten.

    Absent is a normal state (nobody has run it yet), so this returns None rather than
    raising -- but the reason is kept in `_PREFLIGHT_ERROR` and surfaced in `PEAKS.source`,
    because "the roofline silently fell back to estimates" is exactly the failure this
    suite is trying not to repeat.
    """
    global _PREFLIGHT_CACHE, _PREFLIGHT_ERROR
    if _PREFLIGHT_CACHE is None:
        _PREFLIGHT_CACHE = {}
        loader = getattr(C, "load_preflight", None)
        if callable(loader):
            try:
                d = loader()
            except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
                d, _PREFLIGHT_ERROR = None, f"loader raised {type(exc).__name__}"
            if isinstance(d, dict) and d:
                _PREFLIGHT_CACHE = d
            elif not _PREFLIGHT_ERROR:
                try:
                    _PREFLIGHT_ERROR = getattr(C, "preflight_status", lambda: "")() or ""
                except Exception:  # noqa: BLE001
                    _PREFLIGHT_ERROR = ""
                _PREFLIGHT_ERROR = _PREFLIGHT_ERROR or "absent"
        else:
            try:
                d = json.loads(preflight_path().read_text())
                _PREFLIGHT_CACHE = d if isinstance(d, dict) else {}
                if not isinstance(d, dict):
                    _PREFLIGHT_ERROR = f"top level is {type(d).__name__}, expected object"
            except FileNotFoundError:
                _PREFLIGHT_ERROR = "absent"
            except Exception as exc:  # noqa: BLE001
                _PREFLIGHT_ERROR = f"unreadable: {type(exc).__name__}"
    return _PREFLIGHT_CACHE or None


def _calibration_health(d: dict) -> dict:
    """`config.calibration_health()`, or a permissive stand-in if config predates it.

    Delegated rather than reimplemented: the thresholds that separate "measured on an idle
    GPU" from "measured next to a tenant" are a judgement call, and a second copy of a
    judgement call is a second answer waiting to be quoted. The stand-in trusts everything,
    which is what this module did before the H200 came back contended -- degrading to loud
    would be worse, since it would condemn the C500/4060 calibrations too."""
    fn = getattr(C, "calibration_health", None)
    if callable(fn):
        try:
            return fn(d)
        except Exception:  # noqa: BLE001 -- a health check must not break the roofline
            pass
    return {"launch_trusted": True, "timer_tick_trusted": True, "contended": False, "msg": ""}


def _best(d: dict, prefix: str, suffix: str = "_GBs") -> float:
    """Best (max) value among `prefix*suffix` keys. The preflight sweeps several buffer
    sizes and several GEMM shapes; a *peak* is the best of them, and taking the max also
    means a partially-failed sweep degrades to the sizes that did complete instead of
    poisoning the number with a zero."""
    vals = [
        v for k, v in d.items()
        if k.startswith(prefix) and k.endswith(suffix) and isinstance(v, (int, float))
        and v > 0
    ]
    return max(vals) if vals else 0.0


def _overlay_preflight(p: DevicePeaks, d: dict) -> tuple[DevicePeaks, list[str]]:
    """Apply `preflight_h200.json` to a profile. Returns (profile, provenance tags).

    Field mapping, with the reasoning for each choice:
      c_peak            <- max over `calibration.gemm.triton_*_TFs`   (Triton, not cuBLAS)
      c_peak_vendor     <- max over `calibration.gemm.cublas_*_TFs`   (reference only)
      b_peak            <- max over `calibration.bandwidth.{copy,rmw}_*_GBs`
                           Both are read+write streams (copy reads a and writes b; rmw
                           reads and writes the same buffer), which is the access pattern
                           the vector fusions have -- see the block comment above.
      b_peak_read_only  <- max over `calibration.bandwidth.read_*_GBs`
      l2_bytes          <- `device.L2_cache_size`
      launch_s          <- `calibration.launch_us`      (flagged, see below)
      timer_tick_s      <- `calibration.timer_tick_us`  (flagged, see below)
    Anything missing leaves the profile's own value alone, and only the fields that were
    actually replaced are reported as provenance.

    The two timing fields are taken but *judged*, by `config.calibration_health()`, which is
    shared with the harness so there is exactly one definition of "this preflight ran next to
    somebody else". They are still applied -- a measured-but-suspect launch cost beats no
    launch cost for ordering two chains -- but `launch_trusted` / `timer_tick_trusted` travel
    with them into every row, so no cell is ever flagged (or cleared) as tick-limited against
    a tick nobody actually detected.
    """
    cal = d.get("calibration", {}) or {}
    bw = cal.get("bandwidth", {}) or {}
    gemm = cal.get("gemm", {}) or {}
    dev = d.get("device", {}) or {}

    tf = _best(gemm, "triton_", "_TFs")
    vendor = _best(gemm, "cublas_", "_TFs")
    rw = max(_best(bw, "copy_"), _best(bw, "rmw_"))
    ro = _best(bw, "read_")
    l2 = dev.get("L2_cache_size") or 0
    launch = cal.get("launch_us") or 0.0
    tick = cal.get("timer_tick_us") or 0.0
    health = _calibration_health(d)

    tags = [k for k, v in (("gemm", tf), ("bw", rw), ("l2", l2)) if v]
    if launch or tick:
        tags.append("timing" + ("!CONTENDED" if health.get("contended") else ""))
    who = f"{dev.get('name', '?')} {str(dev.get('uuid', ''))[:8]}"
    return (
        replace(
            p,
            c_peak=tf * 1e12 if tf else p.c_peak,
            c_peak_vendor=vendor * 1e12 if vendor else p.c_peak_vendor,
            b_peak=rw * 1e9 if rw else p.b_peak,
            b_peak_read_only=ro * 1e9 if ro else p.b_peak_read_only,
            l2_bytes=int(l2) if l2 else p.l2_bytes,
            launch_s=float(launch) * 1e-6 if launch else p.launch_s,
            timer_tick_s=float(tick) * 1e-6 if tick else p.timer_tick_s,
            launch_trusted=bool(health.get("launch_trusted", True)),
            timer_tick_trusted=bool(health.get("timer_tick_trusted", True)),
            calib_note=str(health.get("msg", "")),
            # a number has to be traceable to a physical device, not just to a device model:
            # this node has eight H200s and they are not all idle
            measured_on=f"{who} @ {d.get('timestamp', '?')}" if (tf or rw) else p.measured_on,
            # only claim "measured" if the two numbers the arithmetic divides by are both
            # measured; an L2-only overlay is not a calibration
            from_calibration=bool(tf and rw),
            estimated=p.estimated and not bool(tf and rw),
        ),
        tags,
    )


def _overlay(p: DevicePeaks, d: dict) -> DevicePeaks:
    """Apply one legacy calibration file (C500 / RTX 4060 era) to a profile. Two shapes are
    understood: the device calibration (`measured.{gemm_bf16_TFs,bw_stream_rw_GBs,
    bw_read_only_GBs}`, `l2_bytes`) and the dedicated GEMM ceiling sweep
    (`triton_bf16_TFs`), which is the denser sweep and therefore supersedes the coarse GEMM
    number when both are present. Kept so `GLM52_DEVICE_PROFILE=c500|rtx4060` still
    reproduces the earlier devices' models exactly."""
    m = d.get("measured", {})
    tf = d.get("triton_bf16_TFs", m.get("gemm_bf16_TFs"))
    rw, ro, l2 = m.get("bw_stream_rw_GBs"), m.get("bw_read_only_GBs"), d.get("l2_bytes")
    return replace(
        p,
        c_peak=tf * 1e12 if tf else p.c_peak,
        b_peak=rw * 1e9 if rw else p.b_peak,
        b_peak_read_only=ro * 1e9 if ro else p.b_peak_read_only,
        l2_bytes=int(l2) if l2 else p.l2_bytes,
        from_calibration=p.from_calibration or bool(tf and rw),
        estimated=p.estimated and not bool(tf and rw),
    )


def _match(name: str) -> str:
    """Table key whose `match` substring occurs in `name`, or "" if none does."""
    lo = (name or "").lower()
    return next((k for k, q in DEVICE_PEAKS.items() if q.match.lower() in lo), "")


def _resolve() -> DevicePeaks:
    """This device's profile: the pin, else the live probe, else the preflight JSON.

    Order is table row -> preflight calibration -> legacy calibration files -> live probe,
    each overriding the last, so a checkout with no GPU and no JSON still prints a
    self-consistent (and clearly labelled) table.

    Two fences, both from failures this suite has already had:

    * **Device fence on the calibration.** A `preflight_h200.json` whose `device.name` does
      not belong to the selected profile is REFUSED, not applied. A stale checkpoint from
      another GPU was one call away from being republished as a fresh measurement once
      already; peaks are the same hazard with none of the visibility, because a wrong
      denominator produces a plausible table rather than an error.
    * **L2 from the probe only when nothing better is known.** glm52's version overwrote
      L2 with the live probe unconditionally. That is right on the bench box and wrong the
      moment you analyse a returned H200 JSON from a different machine -- it would silently
      model Hopper's 50 MB L2 as the analysing box's.

    Pinning `GLM52_DEVICE_PROFILE` suppresses the live probe entirely, which is both the
    documented offline-analysis path (`GLM52_DEVICE_PROFILE=h200 python -m
    glm52_h200.traffic`) and the way to keep this module from paying for a CUDA context.
    """
    pin = os.environ.get("GLM52_DEVICE_PROFILE", "")
    pf = preflight()
    pf_name = ((pf or {}).get("device", {}) or {}).get("name", "") or ""

    e = None
    if pin in DEVICE_PEAKS:
        src = [f"{pin}(pinned)"]
    else:
        try:
            e = C.env()
        except Exception:  # no GPU, or a degraded probe -- the JSON/default carries us
            e = None
        live_name = getattr(e, "device_name", "") or ""
        # live device first (it is the machine whose kernels are being modelled), then the
        # JSON (offline analysis of a returned run), then the default -- and the provenance
        # says which, because "h200" from a match and "h200" from a shrug are not the same
        # claim. A live device that matches NO row is the loudest case: the arithmetic is
        # then a different device's.
        hit = _match(live_name) or _match(pf_name)
        seen = live_name or pf_name
        pin = hit or DEFAULT_PROFILE
        src = [pin if hit else f"{pin}(DEFAULT; device {seen or 'unprobed'!r} "
                               f"matches no profile)"]

    p = DEVICE_PEAKS[pin]

    # --- preflight calibration, device-fenced -----------------------------------------
    if pf is not None:
        if not pf_name:
            src.append("preflight-REFUSED(no device.name)")
        elif _match(pf_name) != pin:
            src.append(f"preflight-REFUSED(device={pf_name!r} != profile {pin})")
        else:
            p, tags = _overlay_preflight(p, pf)
            src.append(f"preflight[{','.join(tags) or 'nothing-usable'}]")
    elif _PREFLIGHT_ERROR:
        src.append(f"preflight-{_PREFLIGHT_ERROR}")

    # --- legacy per-device calibration files -------------------------------------------
    for fname in p.calib:
        try:
            d = json.loads((_results_dir() / fname).read_text())
        except Exception:
            continue
        p = _overlay(p, d)
        src.append(fname)

    # --- live probe: L2 only, and only if the calibration did not supply it -------------
    if e is not None:
        probe_l2 = getattr(e, "l2_bytes", 0) or 0
        if probe_l2 and not (pf is not None and _match(pf_name) == pin):
            p = replace(p, l2_bytes=probe_l2)
            src.append("probe(l2)")
    # The table row is itself a measurement of a specific GPU on a specific day; say which,
    # so a number in a report traces to a device even when no JSON was involved.
    if p.measured_on and not p.from_calibration:
        src.append(f"row measured on {p.measured_on}")
    return replace(p, source=" + ".join(src))


PEAKS = _resolve()
C_PEAK = PEAKS.c_peak
B_PEAK = PEAKS.b_peak
B_PEAK_READ_ONLY = PEAKS.b_peak_read_only  # for reference; not in the ceiling arithmetic
L2_BYTES = PEAKS.l2_bytes
LAUNCH_S = PEAKS.launch_s
TIMER_TICK_S = PEAKS.timer_tick_s
# Whether the two above may be quoted. Exported as module constants because that is how the
# benches already reach LAUNCH_S / TIMER_TICK_S, and a flag that is harder to find than the
# number it qualifies does not get read.
LAUNCH_TRUSTED = PEAKS.launch_trusted
TIMER_TICK_TRUSTED = PEAKS.timer_tick_trusted
# The single number that says how this device's arithmetic differs from the previous two:
# 82.3 on C500, 84.3 on the RTX 4060, ~185 on the H200. Quote it whenever a compute/memory
# label disagrees with an earlier study.
BALANCE_FLOP_PER_BYTE = C_PEAK / B_PEAK if B_PEAK else float("nan")


def require_measured_peaks() -> DevicePeaks:
    """Raise unless the peaks came from a device-matched calibration.

    Import stays safe on any box -- this model is arithmetic and must remain importable
    with no GPU -- but *publishing* an absolute millisecond or a utilization percentage
    against peaks that belong to a different device is a different act, and the report
    generators should call this first.

    Note what this does and does not assert now that the h200 row holds real measurements.
    It asserts that a device-matched preflight supplied `c_peak` and `b_peak` on THIS run.
    It does not assert that the timing calibration is clean -- that is `PEAKS.launch_trusted`
    / `PEAKS.timer_tick_trusted`, and they are deliberately not fatal: an absolute ms is
    perfectly publishable against a contended launch cost, a *launch-aware ceiling* is not.
    """
    if not PEAKS.from_calibration:
        raise RuntimeError(
            f"roofline peaks are not measured on this device -- refusing to publish "
            f"absolute times.\n  profile source: {PEAKS.source}\n"
            f"  c_peak={C_PEAK/1e12:.1f} TF/s b_peak={B_PEAK/1e9:.0f} GB/s "
            f"(estimated={PEAKS.estimated})\n"
            f"  fix: run `python3 glm52_h200/preflight.py` and keep its JSON at "
            f"{preflight_path()}, or point GLM52_H200_PREFLIGHT at it.\n"
            f"  (ratios -- traffic_ratio, roofline_ceiling -- are less sensitive: they "
            f"divide out anything both arms share. Absolute ms are not.)"
        )
    return PEAKS


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

    `ceiling` deliberately ignores launch cost, exactly as the C500 and 4060 models did, so
    the column is comparable across all three devices. `ceiling_with_launch` adds the
    measured per-launch cost to every kernel and is the honest bound at decode, where the
    chains are 2-3 launches of a few microseconds each and the unfused arm's extra launch
    is a real part of its cost. Where the two disagree, the fusion's win is a launch win.
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

    @staticmethod
    def _time_launch(kernels: list[tuple[float, float]]) -> float:
        return sum(max(f / C_PEAK, b / B_PEAK) + LAUNCH_S for f, b in kernels)

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

    @property
    def ceiling_with_launch(self) -> float:
        """As `ceiling`, but charging the measured launch cost per kernel. Equal to
        `ceiling` when the launch cost is unknown (no preflight JSON), so it never
        silently invents a difference.

        Check `launch_trusted` in the row before quoting this. A launch cost measured on a
        shared GPU is inflated, and it is inflated *asymmetrically* here: the unfused arm has
        more kernels, so an overstated per-launch cost overstates the fusion's win. That is a
        bias in this study's only output, which is why the flag is carried per row rather
        than mentioned in a footnote."""
        tf = self._time_launch(self.fused_kernels)
        return self._time_launch(self.unfused_kernels) / tf if tf else float("nan")

    @property
    def l2_resident(self) -> bool:
        """True when the whole unfused chain's traffic would fit in L2.

        Then neither arm is actually going to HBM, the ceiling below is not a bound on
        anything, and the measured ratio is an L2/occupancy/launch story. On the H200's
        ~50 MB L2 that is most of the decode half of the sweep for the vector fusions, so
        it is a flag that has to be read, not a footnote. (It is a conservative test: a
        chain can also be *partly* resident, and the flag stays False there.)
        """
        return bool(L2_BYTES) and self.unfused_bytes <= L2_BYTES

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
            # --- added for the H200 port; every key above is unchanged from glm52 --------
            "roofline_ceiling_with_launch": self.ceiling_with_launch,
            "l2_resident": self.l2_resident,
            "n_kernels_unfused": len(self.unfused_kernels),
            "n_kernels_fused": len(self.fused_kernels),
            "peaks_estimated": PEAKS.estimated,
            # provenance and trust travel with the number, not with the run: a row copied
            # out of a result file into a report must still be able to say where its
            # denominators came from and which of them are safe to quote.
            "peaks_from_calibration": PEAKS.from_calibration,
            "peaks_measured_on": PEAKS.measured_on,
            "launch_trusted": PEAKS.launch_trusted,
            "timer_tick_trusted": PEAKS.timer_tick_trusted,
            "balance_flop_per_byte": BALANCE_FLOP_PER_BYTE,
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


# The seven regimes the H200 suite reports. Named rather than filtered by `T`, because the
# H200 adds decode_bs512/bs1024 (which the 4060 could not hold) and the model must not
# silently drop a regime if `config.py` renumbers the sweep. Unknown names are skipped and
# an empty selection degrades to every regime config defines, so this never returns [].
HEADLINE_REGIMES = (
    "decode_bs1", "decode_bs32", "decode_bs256", "decode_bs512", "decode_bs1024",
    "prefill_t2048", "prefill_t8192",
)


def all_rows(regimes: "list[C.Regime] | None" = None) -> list[dict]:
    if regimes is None:
        by_name = {r.name: r for r in C.ALL_REGIMES}
        regimes = [by_name[n] for n in HEADLINE_REGIMES if n in by_name]
        if not regimes:
            regimes = list(C.ALL_REGIMES)
    return [t.row() for r in regimes for t in model(r)]


if __name__ == "__main__":
    rows = all_rows()
    w = max(len(r["fusion"]) for r in rows)
    # provenance: an absolute ms here is meaningless without the peaks it was divided by
    print(
        f"peaks [{PEAKS.source}]: {C_PEAK/1e12:.2f} TF/s compute, {B_PEAK/1e9:.0f} GB/s "
        f"mixed r+w, balance {BALANCE_FLOP_PER_BYTE:.1f} flop/byte, "
        f"L2 {L2_BYTES/2**20:.0f} MB"
    )
    if PEAKS.c_peak_vendor:
        print(
            f"       (vendor BLAS {PEAKS.c_peak_vendor/1e12:.2f} TF/s -- reference only; "
            f"these kernels are Triton)"
        )
    if LAUNCH_S or TIMER_TICK_S:
        mark = "" if (LAUNCH_TRUSTED and TIMER_TICK_TRUSTED) else "   <-- UNTRUSTED"
        print(
            f"       launch {LAUNCH_S*1e6:.2f} us/kernel, timer tick "
            f"{TIMER_TICK_S*1e6:.3f} us{mark}"
        )
    if PEAKS.calib_note:
        print(f"  [!] {PEAKS.calib_note}")
    if PEAKS.estimated:
        print(
            "  *** PEAKS ARE ESTIMATES, NOT MEASUREMENTS -- every absolute ms below is a\n"
            "      prediction. Run `python3 glm52_h200/preflight.py` to replace them. ***"
        )
    elif not PEAKS.from_calibration and os.environ.get(
        "GLM52_DEVICE_PROFILE", ""
    ) not in DEVICE_PEAKS:
        # Real measurements, but of a different physical GPU than this run's. Absolute times
        # are then a prediction about this device, which is a weaker claim than the numbers
        # look -- and on an 8-GPU node "an H200" is not "this H200".
        # Not printed when the profile was pinned: asking for `GLM52_DEVICE_PROFILE=c500` IS
        # asking to model a device you are not sitting on, and warning about a thing the
        # operator just typed trains people to ignore the banner.
        print(
            "  *** peaks are measured but NOT from a preflight matching this run's device;\n"
            "      absolute ms below describe the device named above, not necessarily the\n"
            "      one that produced any measurement you are comparing against. ***"
        )
    cur = None
    for r in rows:
        if r["regime"] != cur:
            cur = r["regime"]
            print(f"\n=== {cur} ===")
            print(
                f"{'fusion':<{w}}  {'traffic':>8}  {'unfused ms':>10}  {'fused ms':>9}  "
                f"{'CEILING':>8}  bound    resident"
            )
        print(
            f"{r['fusion']:<{w}}  {r['traffic_ratio']:>7.2f}x  {r['unfused_ms']:>10.4f}  "
            f"{r['fused_ms']:>9.4f}  {r['roofline_ceiling']:>7.2f}x  "
            f"{'compute' if r['compute_bound'] else 'memory':<7}  "
            f"{'L2-RESIDENT' if r['l2_resident'] else ''}"
        )
    if any(r["l2_resident"] for r in rows):
        print(
            f"\nL2-RESIDENT: the unfused chain's entire traffic fits the "
            f"{L2_BYTES/2**20:.0f} MB L2, so neither arm reaches HBM and the ceiling in "
            f"that row bounds nothing. Read those cells as launch/occupancy, not roofline."
        )
