"""Reference (unfused, framework-level) implementations of every op in the GLM-5.2 MoE
layer, used as correctness ground truth and as the `torch kernel` production baseline.

Fidelity notes, in the priority order the task specifies (cublas > cutlass > sglang >
torch): the GEMMs go through `torch.matmul`, which dispatches to the vendor BLAS
(MetaX's cuBLAS-equivalent) -- that is the highest-priority reference available on this
box. The MoE dispatch layout (`sorted_token_ids` / `expert_ids` / `num_tokens_post_padded`
with BLOCK_M padding) is a faithful port of sglang's `moe_align_block_size`, so the
Triton kernels here consume exactly the layout sglang's `fused_moe_kernel` consumes.
"""

from __future__ import annotations

import torch

from . import config as C


# --------------------------------------------------------------------------------------
# Norms / activations
# --------------------------------------------------------------------------------------
def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = C.RMS_NORM_EPS) -> torch.Tensor:
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * rstd).to(x.dtype) * w


def add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float = C.RMS_NORM_EPS
) -> tuple[torch.Tensor, torch.Tensor]:
    """sglang's `fused_add_rmsnorm` semantics: returns (normed, new_residual)."""
    h = (x.float() + residual.float()).to(x.dtype)
    return rmsnorm(h, w, eps), h


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """x is [..., 2*I] laid out as [gate | up] -- sglang/vLLM w13 convention."""
    d = x.shape[-1] // 2
    gate, up = x[..., :d], x[..., d:]
    return (torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype)


# --------------------------------------------------------------------------------------
# Router: sigmoid scoring + noaux_tc grouped top-k (GLM-5.2 / DeepSeek-V3 style)
# --------------------------------------------------------------------------------------
def router(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    correction_bias: torch.Tensor | None = None,
    topk: int = C.NUM_EXPERTS_PER_TOK,
    n_group: int = C.N_GROUP,
    topk_group: int = C.TOPK_GROUP,
    norm_topk_prob: bool = C.NORM_TOPK_PROB,
    routed_scaling_factor: float = C.ROUTED_SCALING_FACTOR,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (logits, topk_weights, topk_ids).

    `gate_w` is [n_experts, hidden] (nn.Linear convention). Router math runs in fp32,
    matching `moe_router_dtype: float32` in the GLM-5.2 config.
    """
    logits = torch.nn.functional.linear(x.float(), gate_w.float())
    scores = logits.sigmoid()

    choice = scores if correction_bias is None else scores + correction_bias
    if n_group > 1:
        T, E = choice.shape
        g = choice.view(T, n_group, E // n_group)
        # noaux_tc group score = sum of top-2 within the group
        gscore = g.topk(2, dim=-1).values.sum(-1)
        keep = gscore.topk(topk_group, dim=-1).indices
        mask = torch.zeros_like(gscore).scatter_(1, keep, 1.0)
        choice = (g * mask.unsqueeze(-1)).view(T, E)

    topk_ids = choice.topk(topk, dim=-1).indices
    topk_weights = scores.gather(1, topk_ids)
    if norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True).clamp_min(1e-20)
    topk_weights = topk_weights * routed_scaling_factor
    return logits, topk_weights.float(), topk_ids.int()


# --------------------------------------------------------------------------------------
# MoE dispatch layout -- port of sglang `moe_align_block_size`
# --------------------------------------------------------------------------------------
def moe_align_block_size(
    topk_ids: torch.Tensor, block_m: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort (token,expert) pairs by expert and pad each expert's run up to a multiple of
    `block_m`, so every BLOCK_M row-tile of the grouped GEMM belongs to one expert.

    Returns (sorted_token_ids, expert_ids, num_tokens_post_padded) exactly as sglang's
    `fused_moe_kernel` expects: `sorted_token_ids` holds flattened (token*topk+k) indices,
    padded entries hold `numel` (an out-of-range sentinel the kernel masks off).
    """
    device = topk_ids.device
    flat = topk_ids.flatten()
    numel = flat.numel()

    order = torch.argsort(flat.int(), stable=True)
    sorted_experts = flat[order]
    counts = torch.bincount(flat.int(), minlength=num_experts)
    padded = ((counts + block_m - 1) // block_m) * block_m
    total = int(padded.sum().item())

    sorted_token_ids = torch.full((total,), numel, dtype=torch.int32, device=device)
    expert_ids = torch.zeros(total // block_m, dtype=torch.int32, device=device)

    starts = torch.cat(
        [torch.zeros(1, dtype=padded.dtype, device=device), padded.cumsum(0)[:-1]]
    )
    # one .tolist() each instead of 3*num_experts per-element .item() syncs -- same values
    counts_l, starts_l, padded_l = counts.tolist(), starts.tolist(), padded.tolist()
    src = 0
    for e in range(num_experts):
        c = counts_l[e]
        s = starts_l[e]
        if c:
            sorted_token_ids[s : s + c] = order[src : src + c].int()
            src += c
        nblk = padded_l[e] // block_m
        if nblk:
            expert_ids[s // block_m : s // block_m + nblk] = e
    del sorted_experts
    return (
        sorted_token_ids,
        expert_ids,
        torch.tensor([total], dtype=torch.int32, device=device),
    )


# --------------------------------------------------------------------------------------
# Full MoE MLP reference (fp32 accumulation, per-expert loop)
# --------------------------------------------------------------------------------------
def moe_mlp(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """x: [T, H]; w13: [E, 2I, H]; w2: [E, H, I]. Returns [T, H] merged expert output."""
    T, H = x.shape
    out = torch.zeros(T, H, device=x.device, dtype=torch.float32)
    topk = topk_ids.shape[1]
    for e in range(w13.shape[0]):
        sel = (topk_ids == e).nonzero(as_tuple=False)
        if sel.numel() == 0:
            continue
        rows, ks = sel[:, 0], sel[:, 1]
        xe = x[rows]
        h = torch.nn.functional.linear(xe, w13[e])
        a = silu_and_mul(h)
        y = torch.nn.functional.linear(a, w2[e]).float()
        out.index_add_(0, rows, y * topk_weights[rows, ks].unsqueeze(1))
    del topk
    return out.to(x.dtype)


def expert_merge(
    per_expert_out: torch.Tensor, topk_weights: torch.Tensor, chunk: int = 512
) -> torch.Tensor:
    """per_expert_out: [T, topk, H] (unweighted) -> [T, H].

    Chunked over T. The one-shot form holds two fp32 [T, topk, H] temporaries at once
    (`.float()` and the product, 1536 MiB each at T=8192/H=6144) before `.sum(1)` reduces
    them; a 512-row chunk bounds that transient at 192 MiB, which matters because this
    runs inside a timed arm on an 8 GB card.

    Bit-identical to the one-shot form: same fp32 upcast, same per-row product, same
    sum over the same `topk` values in the same order (the reduction is per output
    element and never split across rows), and the slice assignment *is* the fp32->bf16
    `.to(dtype)` copy, same rounding. Small T takes the original expression verbatim so
    decode timings keep the exact allocation/launch sequence they had.
    """
    T, _, H = per_expert_out.shape
    if T <= chunk:
        return (per_expert_out.float() * topk_weights.unsqueeze(-1)).sum(1).to(
            per_expert_out.dtype
        )
    out = torch.empty(
        T, H, device=per_expert_out.device, dtype=per_expert_out.dtype
    )
    for i in range(0, T, chunk):
        j = min(i + chunk, T)
        out[i:j] = (
            per_expert_out[i:j].float() * topk_weights[i:j].unsqueeze(-1)
        ).sum(1)
    return out
