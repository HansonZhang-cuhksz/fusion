"""Fusion #1 -- o_proj GEMM + residual add (dense GEMM epilogue fusion).

ONE kernel source, `tl.constexpr` flags select the fused epilogue:

    oproj_gemm_kernel(..., FUSE_RESADD=True )   -> h1 = A @ B + residual   (fused)
    oproj_gemm_kernel(..., FUSE_RESADD=False)   -> c  = A @ B              (unfused GEMM)
    epilogue_kernel(..., HAS_RES=True)          -> h1 = c + residual       (split-out add)
    epilogue_kernel(..., HAS_RES=False)         -> h1 = c                  (split-K cast only)

The unfused variant is *the same GEMM kernel* with the flag off, plus `epilogue_kernel`.
Only the mapping (BLOCK_M/N/K, GROUP_M, SPLIT_K, num_warps, num_stages) is allowed to
differ between the two sides, and each side is tuned independently.

Split-K note: with SPLIT_K > 1 the GEMM accumulates into an fp32 buffer with
`tl.atomic_add` (confirmed working on this MACA backend for fp32), so the chain becomes
[zero fp32 buf, gemm, epilogue-cast].  Both sides pay that structure identically; the
fused side's epilogue is a pure cast (2 passes) while the unfused side's also reads the
residual (3 passes).  This is the same accounting as the non-split-K case.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------------------
# The single GEMM source.  FUSE_RESADD selects the epilogue.
# --------------------------------------------------------------------------------------
@triton.jit
def oproj_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    r_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_rm,
    stride_rn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    FUSE_RESADD: tl.constexpr,
    ATOMIC_OUT: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    if SPLIT_K == 1:
        k0 = 0
        num_iters = tl.cdiv(K, BLOCK_K)
    else:
        k0 = pid_k * BLOCK_K
        num_iters = tl.cdiv(K - k0, BLOCK_K * SPLIT_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak
    b_ptrs = b_ptr + (k0 + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    step: tl.constexpr = BLOCK_K * SPLIT_K
    for k in range(0, num_iters):
        kcur = k0 + k * step
        kmask = (kcur + offs_k) < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & kmask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=kmask[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += step * stride_ak
        b_ptrs += step * stride_bk

    cmask = mask_m[:, None] & mask_n[None, :]

    if FUSE_RESADD:
        r_ptrs = r_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn
        if SPLIT_K == 1:
            # plain beta=1 epilogue
            acc += tl.load(r_ptrs, mask=cmask, other=0.0).to(tl.float32)
        else:
            # only the first K-slice folds in the residual
            if pid_k == 0:
                acc += tl.load(r_ptrs, mask=cmask, other=0.0).to(tl.float32)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if ATOMIC_OUT:
        tl.atomic_add(c_ptrs, acc, mask=cmask)
    else:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=cmask)


# --------------------------------------------------------------------------------------
# The split-out elementwise work.  HAS_RES selects add-vs-plain-copy/cast.
# --------------------------------------------------------------------------------------
@triton.jit
def epilogue_kernel(
    c_ptr,
    r_ptr,
    o_ptr,
    n_elements,
    HAS_RES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elements
    v = tl.load(c_ptr + offs, mask=m, other=0.0).to(tl.float32)
    if HAS_RES:
        v += tl.load(r_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + offs, v.to(o_ptr.dtype.element_ty), mask=m)


# --------------------------------------------------------------------------------------
# Launchers
# --------------------------------------------------------------------------------------
def smem_bytes(cfg: dict) -> int:
    """Triton's mainloop double-buffer estimate: stages * 2B * BK * (BM + BN)."""
    return cfg["num_stages"] * 2 * cfg["BLOCK_K"] * (cfg["BLOCK_M"] + cfg["BLOCK_N"])


def gemm_launch(a, b, c, r, cfg, fuse_resadd: bool, atomic_out: bool):
    """a:[M,K] b:[K,N] (strided, any layout) c:[M,N] r:[M,N] or None."""
    M, K = a.shape
    N = c.shape[1]
    sk = cfg.get("SPLIT_K", 1)
    grid = (
        triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),
        sk,
    )
    oproj_gemm_kernel[grid](
        a,
        b,
        c,
        r if r is not None else a,  # unused pointer when FUSE_RESADD is off
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        r.stride(0) if r is not None else 0,
        r.stride(1) if r is not None else 0,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg["GROUP_M"],
        SPLIT_K=sk,
        FUSE_RESADD=fuse_resadd,
        ATOMIC_OUT=atomic_out,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def epilogue_launch(c, r, o, cfg, has_res: bool):
    n = o.numel()
    grid = (triton.cdiv(n, cfg["BLOCK"]),)
    epilogue_kernel[grid](
        c,
        r if r is not None else c,
        o,
        n,
        HAS_RES=has_res,
        BLOCK=cfg["BLOCK"],
        num_warps=cfg["num_warps"],
        num_stages=cfg.get("num_stages", 1),
    )


# --------------------------------------------------------------------------------------
# Chain builders.  These are what `autotune`/`bench_chain` time.
# --------------------------------------------------------------------------------------
def make_fused_chain(a, b, r, out, acc32, cfg):
    """FUSED: h1 = A@B + residual, written straight to `out`.

    SPLIT_K == 1 -> one kernel, no intermediate materialized at all.
    SPLIT_K  > 1 -> [zero acc32, atomic gemm (residual folded at pid_k==0), cast].
    """
    sk = cfg.get("SPLIT_K", 1)
    if sk == 1:
        return [lambda: gemm_launch(a, b, out, r, cfg, True, False)]
    ecfg = cfg["EPI"]
    return [
        lambda: acc32.zero_(),
        lambda: gemm_launch(a, b, acc32, r, cfg, True, True),
        lambda: epilogue_launch(acc32, None, out, ecfg, False),
    ]


def make_unfused_chain(a, b, r, out, cmat, acc32, gcfg, ecfg):
    """UNFUSED: GEMM materializes C, then the elementwise kernel writes h1 = C + r."""
    sk = gcfg.get("SPLIT_K", 1)
    if sk == 1:
        return [
            lambda: gemm_launch(a, b, cmat, None, gcfg, False, False),
            lambda: epilogue_launch(cmat, r, out, ecfg, True),
        ]
    return [
        lambda: acc32.zero_(),
        lambda: gemm_launch(a, b, acc32, None, gcfg, False, True),
        lambda: epilogue_launch(acc32, r, out, ecfg, True),
    ]
