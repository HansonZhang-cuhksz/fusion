"""Fusion #6 -- MoE Up/Gate grouped GEMM with a fused SwiGLU epilogue.

ONE kernel source, one `tl.constexpr` flag (`FUSE_ACT`) selecting the epilogue:

* ``FUSE_ACT=False`` -- exactly sglang 0.5.10's ``fused_moe_kernel`` shape: the grid
  covers the full ``N = 2*I = 4096`` output width, one fp32 accumulator, the block is
  written straight out as bf16.  The chain then needs a second, element-wise
  ``silu_and_mul`` kernel that reads ``[rows, 4096]`` and writes ``[rows, 2048]``.
* ``FUSE_ACT=True``  -- the grid covers ``N = I = 2048``.  Each program keeps **two**
  accumulators, one for the gate columns ``[n, n+BN)`` and one for the up columns
  ``[I+n, I+n+BN)``, sharing a single K-loop over the gathered A tile.  The epilogue
  applies ``silu(gate) * up`` in fp32 and writes only ``[rows, 2048]``.  The 4096-wide
  intermediate is never materialised.

Everything else in the kernel -- the ``sorted_token_ids`` / ``expert_ids`` /
``num_tokens_post_padded`` dispatch, the ``offs_token // top_k`` gather on A, the
per-expert ``stride_be`` weight offset, the ``even_Ks`` fast path, the grouped
``GROUP_SIZE_M`` pid swizzle -- is a direct mirror of sglang's kernel.

Only the *mapping* (BLOCK_M / BLOCK_N / BLOCK_K / num_warps / num_stages / GROUP_M) is
allowed to differ between the two variants, and each is tuned independently.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ======================================================================================
# The one grouped-GEMM kernel.  FUSE_ACT picks the epilogue.
# ======================================================================================
@triton.jit
def moe_gateup_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # dims
    N,  # width of THIS kernel's output: I when FUSE_ACT else 2*I
    K,
    EM,
    num_valid_tokens,
    # strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # meta
    I: tl.constexpr,  # moe_intermediate_size -- gate/up column split in w13
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
    FUSE_ACT: tl.constexpr,
):
    # ---- grouped pid swizzle (sglang) --------------------------------------------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- dispatch ------------------------------------------------------------------
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Padded dispatch slots carry the out-of-range sentinel `num_valid_tokens`.  sglang
    # relies on `token_mask` alone; on this MACA backend the pipeliner emits speculative
    # (unpredicated) prologue loads, so the sentinel row must not even be *addressed*.
    # Clamping to row 0 keeps every generated address inside `a`; the value is discarded
    # by `token_mask` exactly as before.  Shared by both variants.
    safe_token = tl.where(token_mask, offs_token, 0)
    a_ptrs = a_ptr + (
        safe_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )
    if FUSE_ACT:
        # second B tile = the `up` half, I columns further along w13's N axis
        b2_ptrs = b_ptrs + I * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if FUSE_ACT:
        acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
            if FUSE_ACT:
                b2 = tl.load(b2_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)
            if FUSE_ACT:
                b2 = tl.load(b2_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

        acc += tl.dot(a, b)
        if FUSE_ACT:
            acc2 += tl.dot(a, b2)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        if FUSE_ACT:
            b2_ptrs += BLOCK_SIZE_K * stride_bk

    # ---- epilogue -------------------------------------------------------------------
    if FUSE_ACT:
        out = (acc * tl.sigmoid(acc)) * acc2  # silu(gate) * up, fp32
    else:
        out = acc
    out = out.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * safe_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, out, mask=c_mask)


# ======================================================================================
# The split-out element-wise kernel used only by the unfused chain.
# ======================================================================================
@triton.jit
def silu_and_mul_kernel(
    x_ptr,
    y_ptr,
    M,
    I,
    stride_xm,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(I, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < I)

    base = x_ptr + rm[:, None] * stride_xm + rn[None, :]
    g = tl.load(base, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(base + I, mask=mask, other=0.0).to(tl.float32)
    o = (g * tl.sigmoid(g)) * u
    tl.store(
        y_ptr + rm[:, None] * stride_ym + rn[None, :],
        o.to(y_ptr.dtype.element_ty),
        mask=mask,
    )


# ======================================================================================
# Thin python launchers
# ======================================================================================
def smem_bytes(cfg: dict, fused: bool) -> int:
    """Rough Triton pipeline SMEM footprint.  The fused variant stages a second B tile."""
    bn_mult = 2 if fused else 1
    return (
        cfg["num_stages"]
        * 2
        * cfg["BLOCK_K"]
        * (cfg["BLOCK_M"] + bn_mult * cfg["BLOCK_N"])
    )


def launch_gateup(
    a,
    w13,
    c,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    num_valid_tokens,
    top_k,
    I,
    cfg,
    fused: bool,
):
    """`a`: [T, H] bf16.  `w13`: [E, 2I, H] bf16.  `c`: [T*top_k, I or 2I] bf16."""
    N = c.shape[1]
    K = a.shape[1]
    EM = sorted_token_ids.shape[0]
    grid = (
        triton.cdiv(EM, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),
    )
    return moe_gateup_kernel[grid](
        a,
        w13,
        c,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_valid_tokens,
        a.stride(0),
        a.stride(1),
        w13.stride(0),
        w13.stride(2),
        w13.stride(1),
        c.stride(0),
        c.stride(1),
        I=I,
        BLOCK_SIZE_M=cfg["BLOCK_M"],
        BLOCK_SIZE_N=cfg["BLOCK_N"],
        BLOCK_SIZE_K=cfg["BLOCK_K"],
        GROUP_SIZE_M=cfg["GROUP_M"],
        top_k=top_k,
        compute_type=tl.bfloat16,
        even_Ks=(K % cfg["BLOCK_K"] == 0),
        FUSE_ACT=fused,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def launch_silu_and_mul(x, y, cfg):
    M, twoI = x.shape
    I = twoI // 2
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(I, cfg["BLOCK_N"]),)
    return silu_and_mul_kernel[grid](
        x,
        y,
        M,
        I,
        x.stride(0),
        y.stride(0),
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
