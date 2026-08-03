"""Reference (unfused, framework-level) implementations of every op in the GLM-5.2 MoE
layer, used as correctness ground truth and as the `torch kernel` production baseline.

Fidelity notes, in the priority order the task specifies (cublas > cutlass > sglang >
torch): the GEMMs go through `torch.matmul`, which dispatches to the vendor BLAS (cuBLAS
on the H200) -- that is the highest-priority reference available on this box. The MoE
dispatch layout (`sorted_token_ids` / `expert_ids` / `num_tokens_post_padded` with BLOCK_M
padding) is a faithful port of sglang's `moe_align_block_size`, so the Triton kernels here
consume exactly the layout sglang's `fused_moe_kernel` consumes.

Ported unchanged from `glm52/reference.py` (C500 + RTX 4060). **The arithmetic below is
frozen on purpose.** Every `rel_err` in the study -- across three devices -- is measured
against these expressions, so re-associating a sum or moving a `.float()` would silently
move every number in the comparison without failing anything. The H200 has 143 GB and could
afford the un-chunked `expert_merge`, and it is still chunked: the chunked form is
bit-identical (see its docstring), cheaper, and identical to what the other two devices ran.

----------------------------------------------------------------------------------------
WHICH REFERENCE FOR WHICH CLAIM  (this matters -- see REPORT-lazy-prenorm.md A6)
----------------------------------------------------------------------------------------
`rmsnorm` computes `(x * rstd).to(bf16) * w`: it **rounds the normalized value to bf16
before applying the weight**. That is what the framework does, because the unfused path
materializes `x2` in bf16 and the next kernel re-reads it. A fused kernel keeps that value
in fp32 registers and never rounds it. So a reference built on `rmsnorm` shares the
*unfused* path's rounding error and flatters the unfused arm.

On C500 that inverted the verdict for F11b (LOG-07 6.1): against `rmsnorm` the fused arm
looked 10x worse (2.6e-3 vs 3e-4); against exact fp32 the ordering reversed and the fused
arm was closer to truth at *every* regime (1.5-2.2e-3 vs 2.5-2.7e-3). The same asymmetry
exists wherever a fusion elides a bf16 round-trip: F3/F4 (the residual `h1`), F6/F8/F9 (the
SwiGLU intermediate), F11a/F11b (`x2`).

  * framework-faithful (`rmsnorm`, `add_rmsnorm`, `silu_and_mul`, `moe_mlp`)
      -> use for the PASS/FAIL correctness gate. The question there is "does this kernel
         reproduce what the framework computes today", and both arms are checked against
         the same reference, so the shared rounding cancels for that purpose.
  * exact fp32 (`*_exact_fp32`)
      -> use for any ACCURACY CLAIM that compares fused rel_err against unfused rel_err.
         Reporting only the framework-faithful column there is not a small bias; it is a
         sign error.

When an accuracy claim is made, report BOTH columns and say which reference each is
against, exactly as `bench_f11_lazy_prenorm` does (`router_fused` /
`router_fused_vs_exact_fp32`).
"""

from __future__ import annotations

import torch

from . import config as C


# --------------------------------------------------------------------------------------
# Norms / activations -- framework-faithful (bf16-rounded intermediates)
# --------------------------------------------------------------------------------------
def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = C.RMS_NORM_EPS) -> torch.Tensor:
    """`.to(x.dtype)` before `* w` is deliberate and load-bearing -- see the module
    docstring. This is the framework path, not the exact one."""
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
# Exact fp32 variants -- no intermediate rounding anywhere
# --------------------------------------------------------------------------------------
# These are NOT alternative implementations of the same thing: they answer a different
# question. The framework path rounds every value that crosses a kernel boundary; a fusion
# removes exactly those boundaries. To ask "is the fused kernel more or less accurate", the
# reference must round at neither. Each function below is the same expression, in the same
# order, with every `.to(dtype)` deleted -- so the difference between it and its
# framework-faithful twin is *only* the rounding.
def rmsnorm_exact_fp32(
    x: torch.Tensor, w: torch.Tensor, eps: float = C.RMS_NORM_EPS
) -> torch.Tensor:
    """Returns fp32. `(xf * rstd) * w` -- same association as `rmsnorm`, no bf16 round."""
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf * rstd * w.float()


def add_rmsnorm_exact_fp32(
    x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float = C.RMS_NORM_EPS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (normed_fp32, h_fp32), neither rounded.

    `h_fp32.to(x.dtype)` is bit-identical to `add_rmsnorm`'s second return value, so the
    *residual* check is unaffected by which variant you use -- bf16 storage of the residual
    is architectural (a real tensor that downstream kernels re-read), not a rounding
    shortcut. What differs is whether the norm consumes the rounded or the unrounded `h`,
    which is precisely what an add+norm fusion changes.
    """
    h = x.float() + residual.float()
    return rmsnorm_exact_fp32(h, w, eps), h


def silu_and_mul_exact_fp32(x: torch.Tensor) -> torch.Tensor:
    """Returns fp32. Feed it the *unrounded* fp32 gate/up GEMM output: a fused
    UpGate+SwiGLU kernel never materializes `h` in bf16, and this is the reference that
    does not either."""
    d = x.shape[-1] // 2
    xf = x.float()
    gate, up = xf[..., :d], xf[..., d:]
    return torch.nn.functional.silu(gate) * up


def router_logits_exact_fp32(
    h: torch.Tensor, w: torch.Tensor, gate_w: torch.Tensor, eps: float = C.RMS_NORM_EPS
) -> torch.Tensor:
    """Lazy-pre-norm reference for F11b: `(h * rstd * w) @ gate_w.T`, entirely in fp32.

    This is the expression LOG-07 6.1 used to establish that the fused router is *closer*
    to truth than the unfused one. `h` is the residual stream (pre-norm), `w` the RMSNorm
    weight, `gate_w` the router weight in nn.Linear layout [n_experts, hidden].
    """
    return torch.nn.functional.linear(rmsnorm_exact_fp32(h, w, eps), gate_w.float())


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
    matching `moe_router_dtype: float32` in the GLM-5.2 config. `x` may be bf16 (framework
    path) or fp32 (feed it `rmsnorm_exact_fp32` output for the exact-reference chain);
    the `.float()` below is a no-op in the second case.
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
    """x: [T, H]; w13: [E, 2I, H]; w2: [E, H, I]. Returns [T, H] merged expert output.

    The two `linear`s run at the activations' dtype (bf16 in this study) with the vendor
    BLAS's fp32 accumulate -- this is the *framework* path, not an exact one, and the
    intermediate `h` is therefore bf16-rounded exactly as the unfused kernel chain rounds
    it. `moe_mlp_exact_fp32` is the variant to use for accuracy claims.

    E=256 on GLM-5.2, so this loop costs 256 `nonzero` syncs per call. It is only ever run
    once per regime inside a correctness check, never inside a timed arm.
    """
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


def moe_mlp_exact_fp32(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """`moe_mlp` with no intermediate rounding: fp32 GEMMs, fp32 SwiGLU, fp32 result.

    Same loop, same row order, same `index_add_` accumulation -- only the dtypes differ,
    so a diff against `moe_mlp` isolates the rounding. Use it for F6/F8/F9/F11a accuracy
    claims, where the fused kernel keeps the SwiGLU intermediate in registers and the
    unfused chain round-trips it through bf16 HBM.

    Cost: it upcasts one expert's weights at a time ([2I,H] fp32 = 96 MB for w13), so the
    transient is bounded per iteration, but 256 fp32 GEMMs are slow. Correctness path only.
    """
    T, H = x.shape
    out = torch.zeros(T, H, device=x.device, dtype=torch.float32)
    for e in range(w13.shape[0]):
        sel = (topk_ids == e).nonzero(as_tuple=False)
        if sel.numel() == 0:
            continue
        rows, ks = sel[:, 0], sel[:, 1]
        xe = x[rows].float()
        h = torch.nn.functional.linear(xe, w13[e].float())
        a = silu_and_mul_exact_fp32(h)
        y = torch.nn.functional.linear(a, w2[e].float())
        out.index_add_(0, rows, y * topk_weights[rows, ks].unsqueeze(1))
    return out


def shared_expert_mlp(
    x: torch.Tensor, w13_s: torch.Tensor, w2_s: torch.Tensor
) -> torch.Tensor:
    """The single shared expert: one dense SwiGLU MLP at the routed width. Returns fp32.

    GLM-5.2 has `n_shared_experts: 1`, and the whole-layer benchmark includes it because
    it is a real part of every MoE layer's cost. `w13_s`/`w2_s` may be [1, 2I, H] / [1, H, I]
    (the FusedMoE layout, with the leading expert axis) or already squeezed.

    Bit-identical to the expression `bench_layer` used on C500: bf16 GEMM -> bf16 `h` ->
    framework `silu_and_mul` -> fp32 accumulate, i.e. the framework path. There is no
    exact variant here because no fusion in this study touches the shared expert.
    """
    w13_s = w13_s[0] if w13_s.dim() == 3 else w13_s
    w2_s = w2_s[0] if w2_s.dim() == 3 else w2_s
    h = torch.nn.functional.linear(x.float(), w13_s.float())
    a = silu_and_mul(h.to(x.dtype))
    return torch.nn.functional.linear(a.float(), w2_s.float())


def expert_merge(
    per_expert_out: torch.Tensor, topk_weights: torch.Tensor, chunk: int = 512
) -> torch.Tensor:
    """per_expert_out: [T, topk, H] (unweighted) -> [T, H].

    Chunked over T. The one-shot form holds two fp32 [T, topk, H] temporaries at once
    (`.float()` and the product, 1536 MiB each at T=8192/H=6144) before `.sum(1)` reduces
    them; a 512-row chunk bounds that transient at 192 MiB. The H200's 143 GB would carry
    the one-shot form, but the chunked one is bit-identical and allocates 16x less, and
    keeping it means the H200 runs the arithmetic C500 and the 4060 ran.

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
