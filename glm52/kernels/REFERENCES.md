# Production references for each fused kernel

Priority order set by the task: **cuBLAS > CUTLASS > sglang > torch > other engines**.
On this box the vendor BLAS is MetaX's cuBLAS-equivalent, reached through `torch.matmul` /
`torch.addmm`; CUTLASS is not available for MACA; sglang is not installed, so its source is
used as a structural reference (0.5.10.post1, extracted under the session scratchpad).

| Fusion | Production reference | Where |
|---|---|---|
| #1 o_proj + ResAdd | cuBLASLt epilogue with `beta=1` — reachable as `torch.addmm(residual, a, b)` | vendor BLAS |
| #3 ResAdd + RMSNorm | `fused_add_rmsnorm` | `sglang/srt/layers/layernorm.py`, `sglang/jit_kernel/norm.py` |
| #4/#5 Norm + Router | no single production kernel; router math from `biased_grouped_topk` | `sglang/srt/layers/moe/topk.py` |
| #6 Up_Gate + SwiGLU | `fused_moe_kernel` GEMM1 + `silu_and_mul` (sglang keeps these **separate**) | `sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py` |
| #8/#9 Down + Merge | **`fused_moe_kernel` with `FUSE_SUM_ALL_REDUCE`** — sglang *does* fuse this | same file, epilogue |
| #10 Merge + ResAdd | `moe_sum` + residual add | `sglang/srt/layers/moe/fused_moe_triton/` |
| #11 Lazy Pre-Norm | Meta/PyTorch "Towards Free Normalization" §2 | `ads_model_kernel_library/multi_cta_norm_*` |

## Correction logged 2026-07-27

The brief given to the F8/F9 agent stated that sglang does **not** fuse the expert merge into
the down GEMM. **That is wrong for sglang 0.5.10.** The kernel has a `FUSE_SUM_ALL_REDUCE`
path doing precisely that fusion, via fp32 atomics:

```python
if MUL_ROUTED_WEIGHT:
    moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
    accumulator *= moe_weight[:, None]
...
if FUSE_SUM_ALL_REDUCE:
    offs_token_out = offs_token // ROUTER_TOPK
    c_ptrs = c_ptr + stride_cm * offs_token_out[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.atomic_add(c_ptrs, accumulator, mask=c_mask)
```

Consequences: the **atomic** formulation of #8 is the production design and should be a
faithful port, not an independent invention. sglang gates it behind a flag rather than
enabling it always — the regimes where atomics lose to a separate `moe_sum` are exactly what
this study should identify.

## Dispatch-layout convention (must match, all MoE kernels)

`glm52/reference.py::moe_align_block_size` reproduces sglang's layout:
`sorted_token_ids` holds **flattened `token*topk + k`** indices sorted by expert and padded
per-expert up to a multiple of `BLOCK_SIZE_M`; padded slots hold `numel` as an out-of-range
sentinel. Therefore, inside the kernel:

- **A** is `[T, H]` and is gathered with `offs_token // top_k` (sglang line ~462).
- **topk_weights** is `[T*topk]` and is indexed with the flat `offs_token` directly.
- `expert_ids[pid_m]` gives the expert owning that whole row-tile.
- `num_tokens_post_padded` bounds the grid; tiles past it return early.
