#!/usr/bin/env python3
"""H200 preflight probe -- RUN THIS FIRST, then send back the JSON it writes.

Self-contained: imports nothing from this project, installs nothing, mutates nothing, and is
designed never to crash. Every probe is individually guarded; a failure is *recorded* and the
run continues, because "this feature raised TypeError: unexpected keyword 'warp_specialize'"
is exactly the information the kernels need to be written correctly.

It answers four questions the benchmark suite cannot be written without:

  1. What is the stack?          torch / triton / CUDA / driver versions.
  2. What is the device?         SMs, shared memory ceiling, registers, L2, clocks, ECC, MIG.
  3. Which H200 features are REALLY usable from this Triton?  TMA, warp specialization,
     thread-block clusters, fp8 dot. Probed by COMPILING AND LAUNCHING a tiny kernel for
     each -- attribute existence is not evidence, as several Triton releases expose symbols
     that fail at compile time.
  4. What are this machine's achievable peaks?  Bandwidth, bf16 GEMM (Triton and cuBLAS),
     kernel launch cost, CUDA-event timer granularity. Every roofline ceiling in the study is
     computed from these, never from vendor spec.

Usage:
    python3 glm52_h200/preflight.py                 # writes glm52_h200/preflight_h200.json
    python3 glm52_h200/preflight.py --quick         # skips the slower calibration sweeps

Send back:  glm52_h200/preflight_h200.json   (and the console output if convenient)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "preflight_h200.json"

R: dict = {"schema": 1, "probe_errors": {}}


def _err(key: str, exc: BaseException) -> None:
    R["probe_errors"][key] = f"{type(exc).__name__}: {exc}"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ======================================================================================
# 1. stack
# ======================================================================================
def probe_stack() -> None:
    section("1. STACK")
    s = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        s["torch"] = torch.__version__
        s["torch_cuda"] = torch.version.cuda
        s["torch_hip"] = getattr(torch.version, "hip", None)
        s["cudnn"] = getattr(torch.backends.cudnn, "version", lambda: None)()
        s["cuda_available"] = torch.cuda.is_available()
        s["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception as e:  # noqa: BLE001
        _err("torch_import", e)
    try:
        import triton

        s["triton"] = triton.__version__
        s["triton_file"] = getattr(triton, "__file__", None)
    except Exception as e:  # noqa: BLE001
        _err("triton_import", e)
    R["stack"] = s
    for k, v in s.items():
        print(f"  {k:<20} {v}")


# ======================================================================================
# 2. device
# ======================================================================================
def probe_device() -> None:
    section("2. DEVICE (torch properties)")
    try:
        import torch

        if not torch.cuda.is_available():
            print("  CUDA not available -- nothing further can be probed.")
            return
        p = torch.cuda.get_device_properties(0)
        want = [
            "name", "major", "minor", "multi_processor_count", "warp_size",
            "shared_memory_per_block", "shared_memory_per_block_optin",
            "shared_memory_per_multiprocessor", "regs_per_multiprocessor",
            "max_threads_per_multi_processor", "L2_cache_size", "total_memory",
            "is_multi_gpu_board", "gcnArchName", "uuid",
        ]
        d = {}
        for a in want:
            try:
                v = getattr(p, a)
                d[a] = str(v) if a == "uuid" else v
            except Exception:
                pass
        free, total = torch.cuda.mem_get_info()
        d["mem_free_bytes"], d["mem_total_bytes"] = free, total
        d["compute_capability"] = f"{d.get('major')}.{d.get('minor')}"
        R["device"] = d
        for k, v in d.items():
            print(f"  {k:<36} {v}")
        if (d.get("major"), d.get("minor")) != (9, 0):
            print(f"\n  !! expected sm_90 (Hopper/H200); got sm_{d.get('major')}{d.get('minor')}")
    except Exception as e:  # noqa: BLE001
        _err("device", e)
        traceback.print_exc()


# ======================================================================================
# 3. nvidia-smi -- clocks, power, ECC, MIG, persistence
# ======================================================================================
def probe_smi() -> None:
    section("3. nvidia-smi (clocks / power / ECC / MIG)")
    if not shutil.which("nvidia-smi"):
        print("  nvidia-smi not on PATH")
        return
    fields = [
        "name", "driver_version", "vbios_version", "pstate",
        "clocks.sm", "clocks.mem", "clocks.gr", "clocks.max.sm", "clocks.max.mem",
        "clocks_throttle_reasons.active",
        "power.draw", "power.limit", "enforced.power.limit", "power.max_limit",
        "temperature.gpu", "memory.total", "memory.used", "memory.free",
        "ecc.mode.current", "mig.mode.current", "persistence_mode",
        "compute_mode", "utilization.gpu", "pcie.link.gen.current", "pcie.link.width.current",
    ]
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        )
        rows = [r.strip() for r in out.stdout.strip().split("\n") if r.strip()]
        gpus = []
        for i, row in enumerate(rows):
            vals = [v.strip() for v in row.split(",")]
            g = dict(zip(fields, vals))
            gpus.append(g)
            print(f"  --- GPU {i} ---")
            for k, v in g.items():
                print(f"    {k:<34} {v}")
        R["nvidia_smi"] = gpus
        if out.stderr.strip():
            R["probe_errors"]["nvidia_smi_stderr"] = out.stderr.strip()[:500]
    except Exception as e:  # noqa: BLE001
        _err("nvidia_smi", e)
    # topology is useful if this turns out to be multi-GPU
    try:
        t = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=60)
        R["nvidia_smi_topo"] = t.stdout[:4000]
    except Exception as e:  # noqa: BLE001
        _err("nvidia_smi_topo", e)


# ======================================================================================
# 4. Triton feature probes -- COMPILE AND LAUNCH, do not trust attributes
# ======================================================================================
PROBE_SRC = r'''
import torch, triton, triton.language as tl

# ---- baseline: does anything compile at all? ----
@triton.jit
def k_baseline(X, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    tl.store(Y + i, tl.load(X + i, mask=m, other=0.0) * 2.0, mask=m)

# ---- wgmma / tl.dot on Hopper ----
@triton.jit
def k_dot(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM); rn = tl.arange(0, BN); rk = tl.arange(0, BK)
    a = tl.load(A + rm[:, None] * K + rk[None, :])
    b = tl.load(B + rk[:, None] * N + rn[None, :])
    acc = tl.dot(a, b, out_dtype=tl.float32)
    tl.store(C + rm[:, None] * N + rn[None, :], acc)

# ---- warp specialization via tl.range(warp_specialize=True)  [Triton >= 3.3-ish] ----
@triton.jit
def k_ws_range(X, Y, N, BLOCK: tl.constexpr):
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.range(0, N, BLOCK, warp_specialize=True):
        i = k + tl.arange(0, BLOCK)
        acc += tl.load(X + i, mask=i < N, other=0.0)
    tl.store(Y + tl.arange(0, BLOCK), acc)

# ---- device-side TMA descriptor  [tl.make_tensor_descriptor] ----
@triton.jit
def k_tma_device(desc_ptr, Out, BM: tl.constexpr, BN: tl.constexpr):
    d = tl.make_tensor_descriptor(desc_ptr, shape=[BM, BN], strides=[BN, 1],
                                  block_shape=[BM, BN])
    t = d.load([0, 0])
    tl.store(Out + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], t)
'''


def probe_triton_features(quick: bool) -> None:
    section("4. TRITON FEATURE PROBES (compile + launch, not attribute sniffing)")
    feats: dict = {}
    try:
        import torch
        import triton
        import triton.language as tl
    except Exception as e:  # noqa: BLE001
        _err("triton_features", e)
        return

    # --- 4a. attribute inventory (cheap, informative, but NOT proof) -------------------
    inv = {}
    for path in [
        "triton.language.make_tensor_descriptor",
        "triton.language._experimental_make_tensor_descriptor",
        "triton.language._experimental_descriptor_load",
        "triton.language._experimental_descriptor_store",
        "triton.tools.tensor_descriptor.TensorDescriptor",
        "triton.tools.experimental_descriptor.create_2d_tma_descriptor",
        "triton.language.async_task",
        "triton.set_allocator",
        "triton.language.range",
    ]:
        mod, _, attr = path.rpartition(".")
        try:
            m = __import__(mod, fromlist=["_"])
            inv[path] = hasattr(m, attr)
        except Exception:
            inv[path] = False
    feats["attribute_inventory"] = inv
    print("  -- attribute inventory (existence only) --")
    for k, v in inv.items():
        print(f"    {'yes' if v else ' no'}  {k}")

    # tl.range signature tells us whether warp_specialize is accepted
    try:
        import inspect
        feats["tl_range_signature"] = str(inspect.signature(tl.range))
    except Exception as e:  # noqa: BLE001
        feats["tl_range_signature"] = f"unavailable: {e}"
    print(f"    tl.range{feats['tl_range_signature']}")

    # --- 4b. real compile+launch probes ------------------------------------------------
    src = HERE / "_probe_kernels.py"
    try:
        src.write_text(PROBE_SRC)
        sys.path.insert(0, str(HERE))
        import importlib
        pk = importlib.import_module("_probe_kernels")
    except Exception as e:  # noqa: BLE001
        _err("probe_kernels_import", e)
        feats["compile_probes"] = {"_module": f"failed: {e}"}
        R["triton_features"] = feats
        return

    probes: dict = {}

    def run(name: str, fn) -> None:
        try:
            fn()
            torch.cuda.synchronize()
            probes[name] = {"ok": True}
            print(f"    OK    {name}")
        except Exception as e:  # noqa: BLE001
            probes[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
            print(f"    FAIL  {name}: {type(e).__name__}: {str(e)[:150]}")

    print("  -- compile + launch --")
    N = 4096
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    run("baseline_elementwise", lambda: pk.k_baseline[(N // 256,)](x, y, N, BLOCK=256))

    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(64, 64, device="cuda", dtype=torch.float32)
    run("tl_dot_bf16", lambda: pk.k_dot[(1,)](a, b, c, 64, 64, 64, BM=64, BN=64, BK=64))

    run("warp_specialize_tl_range",
        lambda: pk.k_ws_range[(1,)](x, y, N, BLOCK=256, num_warps=4))

    # num_consumer_groups style warp spec (older/forked Triton)
    def _ws_kwargs():
        pk.k_baseline[(N // 256,)](x, y, N, BLOCK=256, num_consumer_groups=1,
                                   num_buffers_warp_spec=2)
    run("warp_specialize_num_consumer_groups", _ws_kwargs)

    # TMA: host-side descriptor object + device-side make_tensor_descriptor
    def _tma():
        from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: F401
        if hasattr(triton, "set_allocator"):
            triton.set_allocator(
                lambda size, align, stream: torch.empty(size, device="cuda", dtype=torch.int8)
            )
        M2 = torch.randn(64, 64, device="cuda", dtype=torch.float32)
        o = torch.empty(64, 64, device="cuda", dtype=torch.float32)
        desc = TensorDescriptor.from_tensor(M2, [64, 64])
        pk.k_tma_device[(1,)](desc, o, BM=64, BN=64)
    run("tma_tensor_descriptor", _tma)

    # cluster / DSMEM launch attribute
    def _cluster():
        pk.k_baseline[(N // 256,)](x, y, N, BLOCK=256, num_ctas=2)
    run("thread_block_cluster_num_ctas", _cluster)

    feats["compile_probes"] = probes

    # --- 4c. what the compiler reports it can do ---------------------------------------
    try:
        tgt = triton.runtime.driver.active.get_current_target()
        feats["triton_target"] = {"backend": tgt.backend, "arch": tgt.arch,
                                  "warp_size": getattr(tgt, "warp_size", None)}
        print(f"  triton target: {feats['triton_target']}")
    except Exception as e:  # noqa: BLE001
        _err("triton_target", e)
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(0)
        feats["triton_device_properties"] = props
        print(f"  triton device properties: {props}")
    except Exception as e:  # noqa: BLE001
        _err("triton_device_properties", e)

    R["triton_features"] = feats
    try:
        src.unlink()
        for p in (HERE / "__pycache__").glob("_probe_kernels*"):
            p.unlink()
    except Exception:
        pass


# ======================================================================================
# 5. calibration -- every roofline ceiling comes from these, not from spec sheets
# ======================================================================================
CALIB_SRC = r'''
import triton, triton.language as tl

@triton.jit
def read_only(P, O, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0); i = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(P + i, mask=i < N, other=0.0)
    tl.store(O + pid, tl.sum(v.to(tl.float32)))

@triton.jit
def mm(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0); nm = tl.cdiv(M, BM); nn = tl.cdiv(N, BN)
    ng = GM * nn; g = pid // ng; fm = g * GM; gs = min(nm - fm, GM)
    pm = fm + ((pid % ng) % gs); pn = (pid % ng) // gs
    rm = (pm * BM + tl.arange(0, BM)) % M
    rn = (pn * BN + tl.arange(0, BN)) % N
    rk = tl.arange(0, BK)
    a = A + (rm[:, None] * sam + rk[None, :] * sak)
    b = B + (rk[:, None] * sbk + rn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        am = tl.load(a, mask=rk[None, :] < K - k * BK, other=0.0)
        bm = tl.load(b, mask=rk[:, None] < K - k * BK, other=0.0)
        acc = tl.dot(am, bm, acc)
        a += BK * sak; b += BK * sbk
    c = C + (rm[:, None] * scm + rn[None, :] * scn)
    tl.store(c, acc.to(C.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))

@triton.jit
def nop(P):
    pass
'''


def probe_calibration(quick: bool) -> None:
    section("5. CALIBRATION (measured on this machine)")
    try:
        import torch
        import triton
    except Exception as e:  # noqa: BLE001
        _err("calibration_import", e)
        return
    src = HERE / "_calib_kernels.py"
    try:
        src.write_text(CALIB_SRC)
        sys.path.insert(0, str(HERE))
        import importlib
        ck = importlib.import_module("_calib_kernels")
    except Exception as e:  # noqa: BLE001
        _err("calib_import", e)
        return

    cal: dict = {}

    def t_s(fn, w=10, n=40):
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    # --- bandwidth. buffer must far exceed H200's ~50 MB L2 --------------------------
    print("  -- bandwidth (buffers >> L2) --")
    bw = {}
    for mb in ([1024] if quick else [1024, 2048]):
        try:
            n = mb * 2**20 // 2
            a = torch.randn(n, device="cuda", dtype=torch.bfloat16)
            b = torch.empty_like(a)
            bw[f"copy_{mb}MB_GBs"] = 2 * n * 2 / t_s(lambda: b.copy_(a)) / 1e9
            bw[f"rmw_{mb}MB_GBs"] = 2 * n * 2 / t_s(lambda: a.mul_(1.0001)) / 1e9
            g = triton.cdiv(n, 8192)
            o = torch.empty(g, device="cuda", dtype=torch.float32)
            bw[f"read_{mb}MB_GBs"] = n * 2 / t_s(
                lambda: ck.read_only[(g,)](a, o, n, BLOCK=8192)) / 1e9
            for k in (f"copy_{mb}MB_GBs", f"rmw_{mb}MB_GBs", f"read_{mb}MB_GBs"):
                print(f"    {k:<24} {bw[k]:8.1f} GB/s")
            del a, b, o
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            _err(f"bandwidth_{mb}MB", e)
    cal["bandwidth"] = bw

    # --- bf16 GEMM: cuBLAS and Triton, at the study's o_proj shape -------------------
    print("  -- bf16 GEMM --")
    gemm = {}
    shapes = [(4096, 16384, 6144)] if quick else [(4096, 16384, 6144), (8192, 16384, 6144)]
    for (M, K, N) in shapes:
        try:
            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
            c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            fl = 2 * M * N * K
            gemm[f"cublas_{M}x{K}x{N}_TFs"] = fl / t_s(lambda: torch.mm(a, b), 5, 20) / 1e12
            best, bcfg = 0.0, None
            grid_cfgs = [(128, 128, 64, 8, 8, 3), (128, 256, 64, 8, 8, 3),
                         (64, 256, 64, 8, 4, 4), (128, 128, 64, 8, 4, 4),
                         (256, 128, 64, 8, 8, 3), (64, 128, 64, 8, 4, 4)]
            for (BM, BN, BK, GM, w, s) in grid_cfgs:
                try:
                    g = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
                    f = lambda BM=BM, BN=BN, BK=BK, GM=GM, w=w, s=s, g=g: ck.mm[g](
                        a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK, GM=GM,
                        num_warps=w, num_stages=s)
                    tf = fl / t_s(f, 3, 10) / 1e12
                    if tf > best:
                        best, bcfg = tf, dict(BM=BM, BN=BN, BK=BK, GM=GM, num_warps=w,
                                              num_stages=s)
                except Exception:
                    pass
            gemm[f"triton_{M}x{K}x{N}_TFs"] = best
            gemm[f"triton_{M}x{K}x{N}_cfg"] = bcfg
            print(f"    {M}x{K}x{N}: cuBLAS {gemm[f'cublas_{M}x{K}x{N}_TFs']:7.2f} TF/s | "
                  f"Triton {best:7.2f} TF/s ({best / max(gemm[f'cublas_{M}x{K}x{N}_TFs'], 1e-9) * 100:.0f}% of vendor)")
            del a, b, c
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            _err(f"gemm_{M}x{K}x{N}", e)
    cal["gemm"] = gemm

    # --- SMEM ceiling actually reachable from Triton ----------------------------------
    print("  -- reachable shared-memory ceiling --")
    smem = {}
    try:
        M, K, N = 512, 512, 512
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        for (BM, BN, BK, s) in [(128, 128, 64, 3), (128, 256, 64, 3), (128, 256, 64, 4),
                                (256, 256, 64, 3), (128, 256, 128, 3)]:
            try:
                g = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
                k = ck.mm[g](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0),
                             b.stride(1), c.stride(0), c.stride(1),
                             BM=BM, BN=BN, BK=BK, GM=8, num_warps=8, num_stages=s)
                torch.cuda.synchronize()
                got = getattr(getattr(k, "metadata", None), "shared", None)
                smem[f"BM{BM}_BN{BN}_BK{BK}_s{s}"] = {"ok": True, "shared_bytes": got}
                print(f"    OK    BM{BM} BN{BN} BK{BK} s{s} -> shared={got}")
            except Exception as e:  # noqa: BLE001
                smem[f"BM{BM}_BN{BN}_BK{BK}_s{s}"] = {"ok": False,
                                                      "error": f"{type(e).__name__}: {e}"[:200]}
                print(f"    FAIL  BM{BM} BN{BN} BK{BK} s{s}: {str(e)[:110]}")
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        _err("smem_probe", e)
    cal["smem_probe"] = smem

    # --- launch cost + timer granularity ----------------------------------------------
    print("  -- launch cost and timer granularity --")
    try:
        flush = torch.empty(256 * 2**20 // 4, device="cuda", dtype=torch.int32)

        def once(nlaunch):
            flush.zero_()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(nlaunch):
                ck.nop[(1,)](flush)
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e)

        pts = {}
        for n in (1, 2, 4, 8):
            for _ in range(20):
                once(n)
            v = sorted(once(n) for _ in range(150))
            pts[n] = v[len(v) // 2]
        L = (pts[8] - pts[1]) / 7
        cal["launch_us"] = L * 1000
        cal["harness_floor_us"] = (pts[1] - L) * 1000
        vals = [once(1) for _ in range(200)]
        ticks = sorted({round(v * 1e6, 3) for v in vals})
        # The true granularity is the LARGEST quantum that divides essentially every
        # sample: 0.256 trivially divides anything 1.024 divides, so scanning upward and
        # keeping the first match would always report the finest candidate.
        cands = {}
        for q in (0.256, 0.512, 1.024, 2.048, 4.096):
            cands[q] = sum(1 for v in vals
                           if abs((v * 1000) / q - round((v * 1000) / q)) < 1e-3) / len(vals)
        ok = [q for q, f in cands.items() if f >= 0.98]
        cal["timer_tick_us"] = max(ok) if ok else min(cands, key=lambda q: -cands[q])
        cal["timer_tick_match_frac"] = cands[cal["timer_tick_us"]]
        cal["timer_tick_candidates"] = cands
        best_q, best_hits = cal["timer_tick_us"], int(cal["timer_tick_match_frac"] * len(vals))
        print(f"    launch cost      {L * 1000:7.2f} us")
        print(f"    harness floor    {cal['harness_floor_us']:7.2f} us")
        print(f"    timer tick       {best_q} us  ({best_hits}/{len(vals)} exact multiples)")
        del flush
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        _err("launch_timer", e)

    R["calibration"] = cal
    try:
        src.unlink()
        for p in (HERE / "__pycache__").glob("_calib_kernels*"):
            p.unlink()
    except Exception:
        pass


# ======================================================================================
# 6. can the full GLM-5.2 layer be instantiated here?
# ======================================================================================
def probe_capacity() -> None:
    section("6. GLM-5.2 CAPACITY CHECK")
    H, MI, E, ER = 6144, 2048, 256, 256
    w13 = E * 2 * MI * H * 2
    w2 = E * H * MI * 2
    need = w13 + w2
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
    except Exception as e:  # noqa: BLE001
        _err("capacity", e)
        return
    GB = 2**30
    cap = {
        "w13_bytes": w13, "w2_bytes": w2, "expert_weights_bytes": need,
        "free_bytes": free, "total_bytes": total,
        "expert_weights_fit": need < free * 0.85,
        "whole_layer_feasible": need < free * 0.6,
    }
    R["capacity"] = cap
    print(f"  w13 (256 experts)      {w13 / GB:8.2f} GB")
    print(f"  w2  (256 experts)      {w2 / GB:8.2f} GB")
    print(f"  expert weights total   {need / GB:8.2f} GB")
    print(f"  free / total VRAM      {free / GB:8.2f} / {total / GB:.2f} GB")
    print(f"  expert weights fit     {cap['expert_weights_fit']}")
    print(f"  whole-layer feasible   {cap['whole_layer_feasible']}")
    if not cap["expert_weights_fit"]:
        print("  !! #6/#8/#9/#11a and the whole-layer benchmark cannot run on this device.")


# ======================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the slower calibration sweeps")
    a = ap.parse_args()

    print(textwrap.dedent(f"""
        GLM-5.2 fusion study -- H200 preflight probe
        writing: {OUT_JSON}
        This script only reads; it installs nothing and changes no device state.
    """).strip())

    R["argv"] = sys.argv
    R["cwd"] = os.getcwd()
    R["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    probe_stack()
    probe_device()
    probe_smi()
    probe_triton_features(a.quick)
    probe_calibration(a.quick)
    probe_capacity()

    OUT_JSON.write_text(json.dumps(R, indent=2, default=str))
    section("DONE")
    if R["probe_errors"]:
        print(f"  {len(R['probe_errors'])} probe(s) reported errors (recorded, not fatal):")
        for k, v in R["probe_errors"].items():
            print(f"    {k}: {v[:160]}")
    else:
        print("  all probes completed without error")
    print(f"\n  -> send back: {OUT_JSON}")


if __name__ == "__main__":
    main()
