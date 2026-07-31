"""Fusions #8 / #9 -- MoE **down** grouped GEMM + expert merge (+ residual add).

What sglang 0.5.10 does *today* (the unfused production path):

    invoke_fused_moe_kernel(intermediate_cache2, w2, intermediate_cache3,
                            ..., mul_routed_weight=True, top_k=1)
    -> intermediate_cache3 : [T, topk, H]        (the routing weight is applied here)
    moe_sum_reduce(intermediate_cache3, out)     -> a SEPARATE reduction kernel

so the ``[T, topk, 6144]`` tensor is fully materialised and immediately re-read.  sglang
*does* carry an atomic variant (``FUSE_SUM_ALL_REDUCE``, kernel line 607) but it is gated
behind a server flag and only used together with a fused all-reduce.  Fusion #8 is exactly
the question "should that be the default?", and #9 adds the post-MoE residual add on top.

ONE grouped-GEMM source, ``tl.constexpr`` flags select the epilogue:

* ``moe_down_kernel``  -- the sglang ``fused_moe_kernel`` shape, expert-major grid over
  ``sorted_token_ids``.
    - ``FUSE_MERGE=False``  : ``tl.store`` into ``c[offs_token, n]`` i.e. ``[rows, H]``.
      This is the unfused GEMM; a separate ``moe_sum_kernel`` then reduces over topk.
    - ``FUSE_MERGE=True``   : ``tl.atomic_add`` into ``c[offs_token // top_k, n]`` i.e.
      ``[T, H]``.  Accumulation strategy **(a)**.  The output buffer must be pre-seeded
      (zeroed for #8, filled with the residual for #9) and that seeding is part of the
      fused chain's cost.
  Everything before the epilogue -- dispatch, gather, K-loop, ``even_Ks``, ``GROUP_SIZE_M``
  swizzle -- is byte-for-byte the same code for both flag values.

* ``moe_down_token_major_kernel`` -- accumulation strategy **(b)**.  Same arithmetic, same
  epilogue algebra, but a different **grid order and loop order** (which fairness rule 1
  explicitly permits): one CTA owns ONE token's ``BLOCK_N`` output columns and loops over
  that token's ``topk`` experts internally, summing in registers.  No atomics, no
  ``[T, topk, H]`` tensor, and the residual add (#9) is a single extra load.  Because the
  token tile is necessarily 1 row, the inner product is a GEMV; ``USE_DOT`` lets the tuner
  choose between a padded ``tl.dot`` (M=16, 15 rows masked off) and a broadcast/reduce
  GEMV, both of which are pure mapping choices.

* ``moe_sum_kernel``   -- the split-out merge, a port of sglang's ``_moe_sum_reduce_kernel``
  (lightllm lineage).  ``ADD_RESIDUAL`` folds the #9 residual add into it, which lets the
  bench also report the "2-kernel" #9 baseline alongside the strict 3-kernel one.
* ``resadd_kernel``    -- the split-out residual add for the strict 3-kernel #9 baseline.

Shapes for GLM-5.2: A = ``[rows=T*8, I=2048]`` (SwiGLU output), B = ``w2 [E, H=6144,
I=2048]``, so the GEMM is ``N = H = 6144``, ``K = I = 2048``.  The routing weight index is
``offs_token`` (flat ``token*topk + k``), exactly as sglang does with ``top_k=1`` for the
A-gather of the second GEMM.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# `do_not_specialize` on the token-count arguments: they are used only in comparisons and
# `cdiv`, never in address arithmetic, so Triton's divisible-by-16 / equal-to-1 hints buy
# nothing -- but WITHOUT this, every regime (rows = 8, 256, 2048, 16384, 65536) is a fresh
# specialization and therefore a fresh 4 s MACA compile of every config in the grid.
# Suppressing it lets all five regimes share one compiled binary per config.  Applied
# identically to every kernel here, so it cannot favour either side of the comparison.


# ======================================================================================
# (1) The one grouped down-GEMM.  FUSE_MERGE picks the epilogue.
# ======================================================================================
@triton.jit(do_not_specialize=["EM", "num_valid_tokens"])
def moe_down_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # dims
    N,  # = H = 6144
    K,  # = I = 2048
    EM,
    num_valid_tokens,  # = rows = T * top_k
    # strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # meta
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
    FUSE_MERGE: tl.constexpr,
):
    # ---- grouped pid swizzle (sglang) ------------------------------------------------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- dispatch --------------------------------------------------------------------
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    # Padded dispatch slots carry the sentinel `num_valid_tokens`.  The MACA pipeliner
    # emits speculative (unpredicated) prologue loads, so the sentinel row must not even
    # be *addressed*; clamping to row 0 keeps every address in range and `token_mask`
    # discards the value exactly as sglang's mask does.
    safe_token = tl.where(token_mask, offs_token, 0)

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A of the down GEMM is the [rows, I] SwiGLU output, one row per (token, k) pair --
    # sglang gathers it with top_k=1, i.e. `offs_token` directly (no // top_k).
    a_ptrs = a_ptr + (safe_token[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # ---- epilogue: routing weight, then either scatter-store or atomic merge ----------
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + safe_token, mask=token_mask, other=0.0)
        acc = acc * moe_weight[:, None]
    out = acc.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    if FUSE_MERGE:
        # strategy (a): accumulate straight into [T, H]; [T, topk, H] never exists.
        rows_out = safe_token // top_k
        c_ptrs = c_ptr + stride_cm * rows_out[:, None] + stride_cn * offs_cn[None, :]
        tl.atomic_add(c_ptrs, out, mask=c_mask)
    else:
        c_ptrs = c_ptr + stride_cm * safe_token[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_ptrs, out, mask=c_mask)


# ======================================================================================
# (2) Strategy (b): token-major -- one token per CTA, topk summed in registers.
# ======================================================================================
@triton.jit(do_not_specialize=["T"])
def moe_down_token_major_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    residual_ptr,
    T,
    N,
    K,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_rm,
    TOPK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # padded dot tile (>=16); unused when USE_DOT=False
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
    USE_DOT: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    tok = (pid // num_pid_n).to(tl.int64)
    pid_n = pid % num_pid_n
    if tok >= T:
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    if USE_DOT:
        # tl.dot needs M >= 16; only row 0 is a real token, the rest are masked to zero.
        offs_m = tl.arange(0, BLOCK_SIZE_M)
        m_mask = offs_m < 1
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for kk in range(TOPK):
            e = tl.load(topk_ids_ptr + tok * TOPK + kk).to(tl.int64)
            row = tok * TOPK + kk
            # every row of the M-tile addresses the SAME A row; rows 1.. are masked off.
            a_ptrs = (
                a_ptr
                + row * stride_am
                + offs_k[None, :] * stride_ak
                + tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.int64)
            )
            b_ptrs = (
                b_ptr
                + e * stride_be
                + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            )
            part = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_SIZE_K):
                if even_Ks:
                    a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
                    b = tl.load(b_ptrs)
                else:
                    a = tl.load(
                        a_ptrs,
                        mask=m_mask[:, None] & (offs_k[None, :] < K - k_start),
                        other=0.0,
                    )
                    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)
                part += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk
            if MUL_ROUTED_WEIGHT:
                w = tl.load(topk_weights_ptr + tok * TOPK + kk)
                part = part * w
            acc += part
        out = tl.sum(acc, 0)
    else:
        # GEMV: fold the routing weight into A (fp32, exact) so a SINGLE [BK, BN] fp32
        # register accumulator can absorb every k-step of every expert; one cross-lane
        # reduction at the very end.
        acc2 = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        for kk in range(TOPK):
            e = tl.load(topk_ids_ptr + tok * TOPK + kk).to(tl.int64)
            row = tok * TOPK + kk
            if MUL_ROUTED_WEIGHT:
                w = tl.load(topk_weights_ptr + tok * TOPK + kk)
            else:
                w = 1.0
            a_ptrs = a_ptr + row * stride_am + offs_k * stride_ak
            b_ptrs = (
                b_ptr
                + e * stride_be
                + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            )
            for k_start in range(0, K, BLOCK_SIZE_K):
                if even_Ks:
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs)
                else:
                    a = tl.load(a_ptrs, mask=offs_k < K - k_start, other=0.0)
                    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)
                acc2 += (a.to(tl.float32) * w)[:, None] * b.to(tl.float32)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk
        out = tl.sum(acc2, 0)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_cn < N
    if ADD_RESIDUAL:
        r = tl.load(residual_ptr + tok * stride_rm + offs_cn, mask=n_mask, other=0.0)
        out = out + r.to(tl.float32)
    tl.store(
        c_ptr + tok * stride_cm + offs_cn * stride_cn,
        out.to(compute_type),
        mask=n_mask,
    )


# ======================================================================================
# (3) The split-out merge kernel used by the unfused chain.
#     Port of sglang `_moe_sum_reduce_kernel`; ADD_RESIDUAL is the #9 variant.
# ======================================================================================
@triton.jit(do_not_specialize=["token_num", "hidden_dim"])
def moe_sum_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    output_ptr,
    output_stride_0,
    residual_ptr,
    residual_stride_0,
    token_num,
    hidden_dim,
    topk_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)
    residual_stride_0 = tl.cast(residual_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim
    mask = mask_token[:, None] & mask_dim[None, :]

    base_ptrs = input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)
    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
        tile = tl.load(base_ptrs + i * input_stride_1, mask=mask, other=0.0)
        acc += tile.to(tl.float32)

    if ADD_RESIDUAL:
        r = tl.load(
            residual_ptr + offs_token[:, None] * residual_stride_0 + offs_dim[None, :],
            mask=mask,
            other=0.0,
        )
        acc += r.to(tl.float32)

    tl.store(
        output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :],
        acc.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


# ======================================================================================
# (4) The split-out residual add, for the strict 3-kernel #9 baseline.
# ======================================================================================
@triton.jit(do_not_specialize=["M", "Nd"])
def resadd_kernel(
    x_ptr,
    r_ptr,
    o_ptr,
    M,
    Nd,
    stride_xm,
    stride_rm,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(Nd, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < Nd)
    x = tl.load(x_ptr + rm[:, None] * stride_xm + rn[None, :], mask=mask, other=0.0)
    r = tl.load(r_ptr + rm[:, None] * stride_rm + rn[None, :], mask=mask, other=0.0)
    tl.store(
        o_ptr + rm[:, None] * stride_om + rn[None, :],
        (x.to(tl.float32) + r.to(tl.float32)).to(o_ptr.dtype.element_ty),
        mask=mask,
    )


# ======================================================================================
# (5) Output seeding for the atomic strategy.  ZEROING/SEEDING IS PART OF THE FUSED COST.
# ======================================================================================
@triton.jit(do_not_specialize=["M", "Nd"])
def seed_kernel(
    o_ptr,
    r_ptr,
    M,
    Nd,
    stride_om,
    stride_rm,
    FROM_RESIDUAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(Nd, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < Nd)
    if FROM_RESIDUAL:
        v = tl.load(r_ptr + rm[:, None] * stride_rm + rn[None, :], mask=mask, other=0.0)
    else:
        v = tl.zeros((BLOCK_M, BLOCK_N), dtype=o_ptr.dtype.element_ty)
    tl.store(o_ptr + rm[:, None] * stride_om + rn[None, :], v, mask=mask)


# ======================================================================================
# Thin python launchers
# ======================================================================================
def smem_bytes(cfg: dict) -> int:
    """Triton pipeline SMEM footprint for the grouped down GEMM (one A + one B tile)."""
    return cfg["num_stages"] * 2 * cfg["BLOCK_K"] * (cfg["BLOCK_M"] + cfg["BLOCK_N"])


def smem_bytes_tokmaj(cfg: dict) -> int:
    """Token-major stages only the B tile (A is a single row / a masked M=16 tile)."""
    m = cfg["BLOCK_M"] if cfg.get("USE_DOT") else 1
    return cfg["num_stages"] * 2 * cfg["BLOCK_K"] * (m + cfg["BLOCK_N"])


def launch_down(
    a,
    w2,
    c,
    topk_weights_flat,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    num_valid_tokens,
    top_k,
    cfg,
    fuse_merge: bool,
):
    """a: [rows, I] bf16.  w2: [E, H, I] bf16.
    c: [rows, H] when fuse_merge=False, else the pre-seeded [T, H] output."""
    N = w2.shape[1]
    K = a.shape[1]
    EM = sorted_token_ids.shape[0]
    grid = (triton.cdiv(EM, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    return moe_down_kernel[grid](
        a,
        w2,
        c,
        topk_weights_flat,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_valid_tokens,
        a.stride(0),
        a.stride(1),
        w2.stride(0),
        w2.stride(2),
        w2.stride(1),
        c.stride(0),
        c.stride(1),
        top_k=top_k,
        BLOCK_SIZE_M=cfg["BLOCK_M"],
        BLOCK_SIZE_N=cfg["BLOCK_N"],
        BLOCK_SIZE_K=cfg["BLOCK_K"],
        GROUP_SIZE_M=cfg["GROUP_M"],
        MUL_ROUTED_WEIGHT=True,
        compute_type=tl.bfloat16,
        even_Ks=(K % cfg["BLOCK_K"] == 0),
        FUSE_MERGE=fuse_merge,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def launch_down_token_major(
    a, w2, out, topk_weights, topk_ids, residual, top_k, cfg, add_residual: bool
):
    """a: [T*top_k, I].  out: [T, H].  topk_ids/topk_weights: [T, top_k]."""
    T = topk_ids.shape[0]
    N = w2.shape[1]
    K = a.shape[1]
    grid = (T * triton.cdiv(N, cfg["BLOCK_N"]),)
    return moe_down_token_major_kernel[grid](
        a,
        w2,
        out,
        topk_weights,
        topk_ids,
        residual,
        T,
        N,
        K,
        a.stride(0),
        a.stride(1),
        w2.stride(0),
        w2.stride(2),
        w2.stride(1),
        out.stride(0),
        out.stride(1),
        residual.stride(0),
        TOPK=top_k,
        BLOCK_SIZE_M=cfg.get("BLOCK_M", 16),
        BLOCK_SIZE_N=cfg["BLOCK_N"],
        BLOCK_SIZE_K=cfg["BLOCK_K"],
        MUL_ROUTED_WEIGHT=True,
        compute_type=tl.bfloat16,
        even_Ks=(K % cfg["BLOCK_K"] == 0),
        USE_DOT=bool(cfg.get("USE_DOT", False)),
        ADD_RESIDUAL=add_residual,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def launch_moe_sum(c3, out, residual, topk, cfg, add_residual: bool):
    """c3: [T, topk, H] (or a [rows, H] view).  out: [T, H]."""
    T = out.shape[0]
    Hd = out.shape[1]
    grid = (triton.cdiv(T, cfg["BLOCK_M"]), triton.cdiv(Hd, cfg["BLOCK_DIM"]))
    return moe_sum_kernel[grid](
        c3,
        c3.stride(0),
        c3.stride(1),
        out,
        out.stride(0),
        residual,
        residual.stride(0),
        T,
        Hd,
        topk_num=topk,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_DIM=cfg["BLOCK_DIM"],
        NUM_STAGE=cfg["num_stages"],
        ADD_RESIDUAL=add_residual,
        num_warps=cfg["num_warps"],
    )


def launch_resadd(x, r, o, cfg):
    M, Nd = x.shape
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(Nd, cfg["BLOCK_N"]),)
    return resadd_kernel[grid](
        x,
        r,
        o,
        M,
        Nd,
        x.stride(0),
        r.stride(0),
        o.stride(0),
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def launch_seed(o, r, cfg, from_residual: bool):
    M, Nd = o.shape
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(Nd, cfg["BLOCK_N"]),)
    return seed_kernel[grid](
        o,
        r,
        M,
        Nd,
        o.stride(0),
        r.stride(0),
        FROM_RESIDUAL=from_residual,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
