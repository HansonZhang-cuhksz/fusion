export const meta = {
  name: 'glm52-moe-fusion-kernels',
  description: 'Build, tune and benchmark 9 Triton fused/unfused MoE-layer kernel pairs for GLM-5.2 on MetaX C500',
  phases: [
    { title: 'Build+Tune', detail: '7 kernel families across 3 exclusive GPU lanes' },
    { title: 'Audit', detail: 'adversarial fairness review of every fused/unfused pair' },
  ],
}

const PREAMBLE = `
# Context

You are building Triton kernels for a **GLM-5.2 MoE decoder layer** on a **MetaX C500** GPU
(a domestic Chinese accelerator with a CUDA-compatible "MACA" stack). The goal is to measure
the benefit (or harm) of specific **kernel fusions** by comparing a fused implementation
against an unfused one, each independently tuned.

## Environment — use exactly this

- Python: \`/home/zhangshuhan/my-envs/fusion/bin/python\`  (torch 2.8.0+metax, triton 3.0.0+metax)
- Working dir: \`/home/zhangshuhan/fusion\` — run scripts from here so \`glm52\` imports work,
  or \`sys.path.insert(0, '/home/zhangshuhan/fusion')\`.
- **YOUR GPU IS EXCLUSIVE. Prefix EVERY python command with \`CUDA_VISIBLE_DEVICES=<N>\`**
  (your N is stated below). Never touch another GPU — other agents are benchmarking on them
  concurrently and contention corrupts timings. Do not call \`torch.cuda.set_device\`.
- Scratch: \`/tmp/claude-1025/-home-zhangshuhan-fusion/7adedebc-d917-43bd-937c-3bdb4619d8f2/scratchpad\`

## Hardware facts that constrain your mapping search

- **warp = 64 lanes** (NOT 32). So \`num_warps=4\` is 256 threads, \`num_warps=8\` is 512,
  \`num_warps=16\` is 1024 = the block ceiling. Valid: 1, 2, 4, 8, 16.
- **Shared memory = 65536 B per CTA.** Many NVIDIA-shaped configs will fail to compile here;
  that is expected. Pre-filter with the SMEM estimate
  \`num_stages * 2 bytes * BLOCK_K * (BLOCK_M + BLOCK_N)\` before adding a config to the grid.
- 104 CUs, 131072 registers/SM, **8 MB L2**, ~1.05 TB/s achievable HBM bandwidth.
- Measured calibration: **Triton reaches only ~50% of the vendor BLAS** on this backend
  (o_proj shape M=4096,N=6144,K=16384: vendor 3.83 ms / 215 TF/s vs best Triton 7.80 ms /
  106 TF/s). Best generic GEMM config found: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
  num_warps=8, num_stages=2, GROUP_M=8. Use that as a seed for your grid. Do NOT try to
  close the gap to the vendor BLAS — it is a backend limitation. Your metric is the
  fused-vs-unfused ratio.
- Triton 3.0 on MACA: stick to plain pointer arithmetic (the style sglang's
  \`fused_moe_kernel\` uses). Before building a large grid on any Triton feature you are
  unsure about (\`tl.make_block_ptr\`, \`num_ctas\`, \`tl.atomic_add\` on bf16, warp
  specialization), compile ONE trivial kernel using it and confirm. \`tl.atomic_add\` on
  fp32 is confirmed working. Assume clusters/DSMEM/TMA do NOT exist.

## Model shapes — GLM-5.2 (all constants are in \`glm52/config.py\`, import them, do not retype)

hidden H=6144, moe_intermediate I=2048, experts E=256, top-k=8, shared experts=1,
w13 layout \`[E, 2*I, H]\` (gate then up, sglang convention), w2 layout \`[E, H, I]\`,
router weight \`[256, 6144]\`, sigmoid scoring + noaux_tc top-k, routed_scaling_factor 2.5,
norm_topk_prob true, rms_norm_eps 1e-5, SwiGLU = silu(gate)*up, dtype **bfloat16** with
fp32 accumulate (router math in fp32).

## The 5 benchmark regimes (tune AND report at every one of these)

| name | T (tokens) | o_proj K | moe rows = T*8 |
|---|---|---|---|
| decode_bs1    | 1    | 32768 | 8 |
| decode_bs32   | 32   | 32768 | 256 |
| decode_bs256  | 256  | 32768 | 2048 |
| prefill_t2048 | 2048 | 16384 | 16384 |
| prefill_t8192 | 8192 | 16384 | 65536 |

They are \`glm52.config.DECODE_REGIMES\` / \`PREFILL_REGIMES\` — filter to T in {1,32,256} and
{2048,8192}. Decode uses the **absorbed** MLA path (o_proj K=32768), prefill the
**non-absorbed** path (K=16384).

## Harness API — \`glm52/common.py\` (READ IT FIRST, use it, do not reinvent timing)

- \`bench_chain(fns, warmup, rep, flush=True) -> Timing\` — times a LIST of callables
  back-to-back as one logical op, with ONE L2 flush before each repetition of the whole
  chain (never between its kernels). Returns \`.p50_ms/.p10_ms/.p90_ms/.noflush_p50_ms\`.
- \`autotune(make_chain, configs, warmup, rep) -> TuneResult\` — brute-force search;
  \`make_chain(cfg)\` returns the callables to time. Compile failures are recorded, not fatal.
  Returns \`.best_cfg/.best_ms/.n_tried/.n_failed/.table\`.
- \`check(got, ref, tol=2e-2, label) -> dict\` — relative max-abs error vs an fp32 reference.
- \`record(name, payload) -> Path\` — writes \`results/<name>.json\`.
- \`speedup_row(regime, fused, unfused, extra)\` — builds a result row.
- \`glm52/reference.py\` — \`rmsnorm\`, \`add_rmsnorm\`, \`silu_and_mul\`, \`router\` (sigmoid +
  noaux_tc), \`moe_align_block_size\` (port of sglang's), \`moe_mlp\`, \`expert_merge\`.

## THE FAIRNESS RULES — these decide whether the whole result is worth anything

1. **One kernel source, flags differ.** Write the kernel ONCE with \`tl.constexpr\` flags
   selecting the fused epilogue/prologue. The unfused variant is *the same kernel* with the
   flag off, plus a separate kernel for the split-out work. The ONLY thing allowed to differ
   between fused and unfused is the **mapping** (BLOCK sizes, num_warps, num_stages, loop
   order, grid order, GROUP_M). Never write two structurally different kernels and compare them.
2. **Tune both sides independently and equally.** Run \`autotune\` separately for the fused
   variant and for EACH kernel of the unfused chain (or for the chain jointly — state which
   you did). Never reuse the fused kernel's best config for the unfused one. An unfused
   baseline that is under-tuned is the most common way to fabricate a fusion win, and an
   auditor agent WILL check your grids for this.
3. **Both sides must do the same work.** The fused and unfused variants must produce
   bit-comparable outputs (within bf16 tolerance) AND materialize the same tensors that a
   real layer needs downstream. If the fused version legitimately avoids materializing an
   intermediate, that IS the fusion benefit — say so explicitly; but it must not skip an
   output the next layer needs.
4. **Validate before you time.** Every variant checked against the fp32 \`reference.py\`
   implementation with \`check()\`. Report the rel_err in your results. A fast wrong kernel
   is a failed deliverable.
5. **Report losses honestly.** If the fusion is slower, that is a valid and useful result.
   Do not tune the unfused side less to make the fusion look good, and do not hide a
   regression. Say at which regimes it wins and at which it loses.

## Tuning protocol

Two-stage per variant per regime: (a) coarse grid ~60-120 valid configs spanning the space,
(b) refine ~20-40 neighbours around the coarse winner. Pre-filter invalid configs with the
SMEM formula so the grid is not mostly failures. Record the FULL config table in your JSON
(\`TuneResult.table\`) — the audit needs it.

## Deliverables (all paths relative to /home/zhangshuhan/fusion)

1. \`glm52/kernels/<module>.py\` — the Triton kernel(s) + thin python launchers.
2. \`glm52/bench/<script>.py\` — runnable end-to-end: validates, autotunes both sides at all
   5 regimes, writes \`results/<id>.json\` via \`record()\`. Must run standalone with
   \`CUDA_VISIBLE_DEVICES=<N> /home/zhangshuhan/my-envs/fusion/bin/python glm52/bench/<script>.py\`.
3. \`results/<id>.json\` — actually RUN the benchmark and produce this file. Do not deliver
   untested code.
4. \`log/<LOGFILE>\` — markdown log: your plan, the fusion's memory-traffic analysis
   (bytes moved fused vs unfused), what you implemented, the mapping search space and the
   winning configs for each side, the results table (fused ms / unfused ms / speedup /
   rel_err per regime), the vendor-BLAS reference line where a GEMM is involved, surprises,
   and an honest verdict on whether this fusion is worth it and where.

Write real, working, tuned code. Iterate until the benchmark actually runs and the numbers
are trustworthy. If something is genuinely impossible on this hardware, document precisely
why with numbers rather than silently dropping it.
`

const SCHEMA = {
  type: 'object',
  properties: {
    family: { type: 'string' },
    status: { type: 'string', enum: ['complete', 'partial', 'failed'] },
    files: { type: 'array', items: { type: 'string' }, description: 'paths written' },
    results_json: { type: 'string' },
    log_file: { type: 'string' },
    rows: {
      type: 'array',
      description: 'one row per regime per variant pair',
      items: {
        type: 'object',
        properties: {
          variant: { type: 'string' },
          regime: { type: 'string' },
          fused_ms: { type: 'number' },
          unfused_ms: { type: 'number' },
          speedup: { type: 'number' },
          rel_err: { type: 'number' },
          fused_cfg: { type: 'string' },
          unfused_cfg: { type: 'string' },
        },
        required: ['variant', 'regime', 'fused_ms', 'unfused_ms', 'speedup'],
      },
    },
    verdict: { type: 'string', description: 'is this fusion worth it, and where' },
    issues: { type: 'string', description: 'anything unresolved, unsupported, or suspicious' },
  },
  required: ['family', 'status', 'files', 'rows', 'verdict'],
}

const FAMILIES = [
  {
    gpu: 0,
    lane: 0,
    label: 'F6-upgate-swiglu',
    log: 'LOG-04-F6-upgate-swiglu.md',
    id: 'f06_upgate_swiglu',
    spec: `
## Your fusion: **#6 Up_Gate GEMM + SwiGLU activation** (grouped MoE GEMM, epilogue fusion)

This is the MoE layer's first expert GEMM. Reference structure: sglang 0.5.10
\`sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py::fused_moe_kernel\`.
A copy of the sglang source is extracted at
\`/tmp/claude-1025/-home-zhangshuhan-fusion/7adedebc-d917-43bd-937c-3bdb4619d8f2/scratchpad/sgl/sglang-0.5.10.post1/\`
— READ their kernel and mirror its structure (sorted_token_ids / expert_ids /
num_tokens_post_padded dispatch, the \`offs_token\` gather on A, per-expert weight pointer
offsetting, \`even_Ks\` handling). Use \`reference.moe_align_block_size\` to build the layout.

**UNFUSED:** grouped GEMM \`A[T*topk gathered, 6144] @ w13[e][6144, 4096]\` writing the full
\`[rows, 4096]\` intermediate, then a separate \`silu_and_mul\` elementwise kernel reading
\`[rows, 4096]\` and writing \`[rows, 2048]\`. This is exactly what sglang does today.

**FUSED:** grid over the intermediate dim N=2048; each program keeps **two** accumulators —
one for the gate columns \`[n : n+BN]\` and one for the up columns \`[2048+n : 2048+n+BN]\` —
sharing one K-loop over A, then applies \`silu(gate)*up\` in the epilogue and writes only
\`[rows, 2048]\`. Note the cost: two accumulators double the register pressure versus the
unfused kernel's one, so the fused kernel may be forced to a smaller BLOCK_M/BLOCK_N. That
tension is the interesting result — quantify it (report registers/occupancy if you can get
them from the compiled kernel's \`n_regs\`/\`n_spills\` attributes).

Also report the vendor-BLAS reference: a per-expert loop of \`torch.matmul\` + torch
\`silu_and_mul\` for the same shapes, as the absolute production line.

Expected traffic win (state the real numbers for each regime in your log): unfused moves
the 4096-wide intermediate three times (write, read, write-2048); fused moves it zero times
and writes 2048 wide once.`,
  },
  {
    gpu: 0,
    lane: 0,
    label: 'F8F9-down-merge-resadd',
    log: 'LOG-05-F8F9-down-merge-resadd.md',
    id: 'f08f09_down_merge_resadd',
    spec: `
## Your fusions: **#8 Down GEMM + Expert Merge** and **#9 Down + Expert Merge + ResAdd2**

The MoE layer's second expert GEMM plus the top-k reduction. Reference: sglang's
\`fused_moe_kernel\` with \`MUL_ROUTED_WEIGHT=True\` (source extracted at
\`/tmp/claude-1025/-home-zhangshuhan-fusion/7adedebc-d917-43bd-937c-3bdb4619d8f2/scratchpad/sgl/sglang-0.5.10.post1/\`).
Note what sglang actually does today: GEMM2 with MUL_ROUTED_WEIGHT writes
\`intermediate_cache3 [T, topk, 6144]\`, then a SEPARATE \`moe_sum\` reduces over topk. So the
merge is NOT fused in production — you are testing whether it should be.

**UNFUSED:** grouped down GEMM \`act[rows, 2048] @ w2[e][2048, 6144]\` scaled by the routing
weight, writing \`[T, topk, 6144]\`; then a separate merge kernel summing over topk to
\`[T, 6144]\`; then (for #9) a separate residual-add kernel.

**FUSED #8:** the down GEMM's epilogue multiplies by the routing weight and
**accumulates directly into \`out[T, 6144]\`**, never materializing \`[T, topk, 6144]\`.
Implement BOTH accumulation strategies and report both — they have very different profiles:
  - **(a) atomic**: \`tl.atomic_add\` in fp32 into the output. Works at any BLOCK_M. Costs
    read-modify-write traffic and contention.
  - **(b) token-major**: one CTA owns a token-block and loops over that token's topk experts
    internally, summing in registers — NO atomics. Requires BLOCK_M small enough that the
    tokens in a tile share the loop structure (BLOCK_M=1 is the clean case). Should be
    strong in decode (small T) and poor in prefill. Verify that intuition.
**FUSED #9:** as #8 plus the second residual add — seed the accumulator with the residual
(or add it once on the final contribution). Should be nearly free on top of #8; confirm.

Be careful with correctness in variant (a): the output buffer must be zeroed (or seeded with
the residual for #9) before the kernel, and that zeroing/seeding cost MUST be included in the
fused timing. Do not accidentally exclude it.

Also report the vendor-BLAS reference line for the same shapes.`,
  },
  {
    gpu: 1,
    lane: 1,
    label: 'F1-oproj-resadd',
    log: 'LOG-01-F1-oproj-resadd.md',
    id: 'f01_oproj_resadd',
    spec: `
## Your fusion: **#1 o_proj GEMM + Residual Add** (dense GEMM, epilogue fusion)

The attention output projection. Shapes: \`[T, K] @ [K, 6144]\` where **K=32768 for the
decode regimes** (absorbed MLA: attention emits [T, 64, kv_lora_rank=512] and W_UV is folded
into o_proj) and **K=16384 for prefill** (non-absorbed: [T, 64, v_head_dim=256]).

**UNFUSED:** GEMM writes \`C[T, 6144]\`, then a separate elementwise kernel computes
\`h1 = C + residual\`.
**FUSED:** the GEMM's epilogue adds the residual before the store, writing \`h1\` directly.
This is the cuBLASLt \`beta=1\` / CUTLASS \`LinearCombinationResidualBlock\` pattern.

The production reference here is strong and you should report it: \`torch.addmm(residual, a,
b)\` dispatches to the vendor BLAS with a fused beta-accumulate, and \`torch.mm\` + a separate
add is the unfused vendor line. Report **four** numbers per regime: triton-fused,
triton-unfused, vendor-fused (\`addmm\`), vendor-unfused (\`mm\`+add). That gives an honest
picture of whether the Triton fusion reproduces the production fusion's benefit.

Expected win is small (the epilogue saves one write + one read + one write of T*6144*2 B
against a GEMM that at K=32768 is weight-traffic- or compute-dominated). Quantify it
precisely per regime; a small-but-real or a within-noise result are both fine outcomes —
report the p10/p90 spread so the reader can judge significance. At decode_bs1 the GEMM is
entirely weight-bound (reading a 402 MB weight for 1 token) so expect ~0 relative gain;
say so with numbers.

Note: at T=1 and T=32 a standard tiled GEMM wastes most of its tile. Consider including
split-K configs in the grid for the decode regimes (a GEMV-shaped problem), tuned
independently for both sides.`,
  },
  {
    gpu: 1,
    lane: 1,
    label: 'F11-lazy-prenorm',
    log: 'LOG-07-F11-lazy-prenorm.md',
    id: 'f11_lazy_prenorm',
    spec: `
## Your fusion: **#11 Lazy Pre-Norm** — RMSNorm fused as a GEMM **prologue**

This is the technique from the PyTorch/Meta blog "Towards Free Normalization: Fusing
Normalization into GEMM and Attention Kernels" (Zhou et al., 2026-07-10), section 2. The PDF
is at \`/home/zhangshuhan/fusion/Towards Free Normalization_ Fusing Normalization into GEMM
and Attention Kernels – PyTorch.pdf\` — read pages 7-11 for the algorithm and its limits.

**The identity:** for affine-free RMSNorm, row-scaling commutes with matmul:
\`(A * rstd[:, None]) @ B == (A @ B) * rstd[:, None]\`. So a GEMM CTA can accumulate
\`acc += tile_A @ tile_B\` and \`sq_sum += (tile_A * tile_A).sum(-1)\` in the SAME K-loop, then
apply \`rstd = rsqrt(sq_sum / K + eps)\` as an **epilogue scale**. The cyclic dependency
(needing rstd before the K-loop) disappears.

**Handling GLM-5.2's affine weight** (the paper lists this as a blocker; it is not one at
inference): the RMSNorm weight \`w\` is a column-wise scale of A, and B is a constant weight
matrix, so \`((A * rstd) * w) @ B == (A @ (w[:, None] * B)) * rstd\`. Pre-fold \`w\` into B's
rows **offline** (outside the timed region — it is a weight transform done once at load
time). Validate the folded result against the unfolded \`reference.rmsnorm\` + matmul.

**Apply it to the two consumers of \`post_attention_layernorm\` whose K == hidden == 6144:**
  - **(a) the routed-expert w13 GEMM** — \`rmsnorm(x2) @ w13[e]\`, the grouped MoE GEMM.
    This is the high-value case. Use \`reference.moe_align_block_size\` for dispatch, same as
    sglang's \`fused_moe_kernel\`.
  - **(b) the router GEMM** — \`rmsnorm(x2) @ W_gate.T\`, \`[T,6144] @ [6144,256]\`.

**UNFUSED:** a standalone RMSNorm kernel writing \`x2[T, 6144]\`, then the GEMM reading \`x2\`.
**FUSED:** the GEMM reads the UN-normalized \`h1\` directly and normalizes in its own epilogue;
\`x2\` is never materialized.

**The catch you must quantify:** the sum-of-squares is recomputed redundantly by every CTA
that shares an m_tile but owns a different n_tile (the paper flags this in section 2). For
the router GEMM, N=256 means 1-2 n_tiles, so redundancy is ~1x — nearly ideal. For the w13
GEMM, N=4096 means 4096/BLOCK_N n_tiles, so the redundancy factor is 16-64x on the
sum-of-squares reduction. Measure whether it is hidden behind the MMA pipeline as the paper
claims, or whether it dominates. **Report the redundancy factor for each config you pick.**

CRITICAL correctness note: if \`x2\` is genuinely needed by another consumer downstream (it is
— the shared expert and the router both consume it), then a fused variant that never
materializes \`x2\` is only valid if ALL consumers are fused. Handle this honestly: either
(i) time the fused variant as materializing x2 too (fusion still saves the read), or
(ii) fuse all consumers and state that as the precondition. Pick one, implement it, and be
explicit in the log about which you did and why.

Report the vendor-BLAS reference line too.`,
  },
  {
    gpu: 3,
    lane: 2,
    label: 'F3-resadd-rmsnorm',
    log: 'LOG-02-F3-resadd-rmsnorm.md',
    id: 'f03_resadd_rmsnorm',
    spec: `
## Your fusion: **#3 Residual Add + RMSNorm** (pure memory-bound vector fusion)

The single most standard fusion in the layer — sglang ships it as \`fused_add_rmsnorm\`
(see \`sglang/srt/layers/layernorm.py\` and \`sglang/jit_kernel/norm.py\` in the extracted
source at \`/tmp/claude-1025/-home-zhangshuhan-fusion/7adedebc-d917-43bd-937c-3bdb4619d8f2/scratchpad/sgl/sglang-0.5.10.post1/\`).
Semantics (match them exactly): given \`x\` and \`residual\`, compute \`h1 = x + residual\`
(this becomes the NEW residual, and must be written out — the next block needs it) and
\`x2 = rmsnorm(h1) * w\` (written out too). Both outputs are live.

**UNFUSED:** an add kernel (read x, read residual, write h1), then an RMSNorm kernel
(read h1, write x2) = 3 reads + 2 writes.
**FUSED:** one kernel: read x, read residual, write h1 and x2 = 2 reads + 2 writes.
So the ceiling is 5/4 = **1.25x**, and only if you are perfectly bandwidth-bound. Measure how
close you get and explain the gap (launch overhead at decode_bs1 where the whole tensor is
6144*2 = 12 KB will dominate — report that separately).

Shapes: \`[T, 6144]\` bf16 for T in {1, 32, 256, 2048, 8192}. Accumulate the sum-of-squares in
fp32 (match \`reference.add_rmsnorm\`, which is the ground truth).

Mapping space: rows per program, BLOCK_SIZE over the 6144 hidden dim (6144 = 2048*3, so it
is NOT a power of two — decide between a padded power-of-two BLOCK with masking, versus a
multi-pass loop over the row, and tune both), num_warps, num_stages, and whether one program
handles multiple rows. Include a persistent-grid variant. Tune the fused kernel and EACH of
the two unfused kernels independently.

Also report the torch eager reference (\`(x+residual)\` then \`reference.rmsnorm\`) and, if
\`torch.compile\` works on this backend, the inductor-fused version — that is the "torch
kernel" production line. If torch.compile fails on MACA, note that and move on.`,
  },
  {
    gpu: 3,
    lane: 2,
    label: 'F4F5-norm-router',
    log: 'LOG-03-F4F5-norm-router.md',
    id: 'f04f05_norm_router',
    spec: `
## Your fusions: **#4 ResAdd + RMSNorm + Router** and **#5 RMSNorm + Router**

Fuse the router's tiny GEMM into the normalization kernel. The router is
\`x2[T, 6144] @ W_gate.T[6144, 256]\` in **fp32** (GLM-5.2 sets \`moe_router_dtype: float32\`),
followed by sigmoid scoring and noaux_tc top-8 selection (see \`reference.router\` — match it
exactly, including \`norm_topk_prob\` and \`routed_scaling_factor=2.5\`).

**Why this can work despite being a GEMM:** \`W_gate\` is 6144*256*2 B = **3.0 MB** and C500's
L2 is **8 MB**, so every CTA's re-read of the router weight hits L2 rather than HBM. Verify
this empirically — if the measured HBM traffic or the scaling with T says otherwise, report it.

Build ONE kernel with constexpr flags: \`HAS_RESIDUAL\` (gives you #4 vs #5) and \`FUSE_TOPK\`.
Each program owns a block of rows; it computes the normed row in registers/SMEM, then
multiplies by \`W_gate\` to get all 256 logits for that row. Because all 256 logits for a row
live in one program, sigmoid + top-8 can ALSO be folded in — build that as the \`FUSE_TOPK\`
variant and report it as an extra data point (it is beyond the user's original list but is
the natural production endpoint).

Note \`x2\` must still be written out — the expert GEMMs consume it. The fusion saves the
router kernel's read of \`x2\`, not the write.

**Variants to measure (each independently tuned):**
  - #5 unfused: rmsnorm kernel, then router GEMM kernel [, then topk kernel]
  - #5 fused: rmsnorm+router in one kernel [+ FUSE_TOPK]
  - #4 unfused: add kernel, rmsnorm kernel, router GEMM [, topk]
  - #4 fused: add+rmsnorm+router in one kernel [+ FUSE_TOPK]

Coordinate with the F3 agent's result if useful, but do your own tuning — do not import
their configs.

Report the vendor line: \`reference.router\` on top of a torch add+rmsnorm.

At T=8192 the router GEMM is 25.8 GFLOP; at T=1 it is a GEMV. Expect very different verdicts
across regimes and report each.`,
  },
  {
    gpu: 3,
    lane: 2,
    label: 'F10-merge-resadd',
    log: 'LOG-06-F10-merge-resadd.md',
    id: 'f10_merge_resadd',
    spec: `
## Your fusion: **#10 Expert Merge + Residual Add** (pure memory-bound vector fusion)

The tail of the MoE block, and the baseline that the F8/F9 agent's fused down-GEMM must beat.
Input: per-expert outputs \`[T, topk=8, 6144]\` bf16 (unweighted) plus routing weights
\`[T, 8]\` fp32 plus the residual \`[T, 6144]\`. Output: \`[T, 6144]\`.

**UNFUSED:** a merge kernel computing \`sum_k w_k * y[t,k,:]\` -> \`[T, 6144]\`, then a separate
residual-add kernel.
**FUSED:** one kernel doing the weighted top-k reduction and the residual add before the
single store.

Traffic: unfused reads T*8*6144*2 B + writes T*6144*2, then reads 2*T*6144*2 and writes
T*6144*2. Fused reads T*8*6144*2 + T*6144*2 and writes T*6144*2. With topk=8 the merge
input dominates (8 units vs 1), so the fusion saves 3 of 13 units ~= a **1.3x** ceiling at
best; compute the exact ratio per regime and compare against what you measure.

This is a bandwidth kernel — the mapping search is over rows/program, BLOCK over hidden,
whether to loop over topk in registers or unroll it (topk=8 is a compile-time constant here,
so unrolling is available: make it a constexpr and tune both), num_warps, num_stages, and
vectorization width. Also try a persistent-grid variant.

Report the torch eager reference (\`reference.expert_merge\` + add) and torch.compile /
inductor if it works on this backend.

Also: state clearly in your log how this compares to the F8/F9 agent's fused down-GEMM
result (that agent eliminates this kernel entirely), so the reader can see the two paths
side by side. You may read their results JSON at the end if it exists; do not block on it.`,
  },
]

// ---------------------------------------------------------------------------------
// Phase 1: three exclusive GPU lanes, families run sequentially within a lane so that
// no two benchmark processes ever share a GPU (contention would corrupt every timing).
// ---------------------------------------------------------------------------------
phase('Build+Tune')
log('Launching 7 kernel families across GPU lanes 0, 1, 3 (GPU 2 is unavailable)')

const LANES = [0, 1, 2]
const laneResults = await parallel(
  LANES.map((laneId) => async () => {
    const mine = FAMILIES.filter((f) => f.lane === laneId)
    const out = []
    for (const fam of mine) {
      log(`lane ${laneId} (GPU ${mine[0].gpu}): starting ${fam.label}`)
      const r = await agent(
        PREAMBLE +
          `\n\n# ===== YOUR ASSIGNMENT =====\n` +
          `**Your exclusive GPU: CUDA_VISIBLE_DEVICES=${fam.gpu}**\n` +
          `**Your result id: \`${fam.id}\`** (so: \`results/${fam.id}.json\`, ` +
          `\`glm52/bench/bench_${fam.id}.py\`)\n` +
          `**Your log file: \`log/${fam.log}\`**\n` +
          fam.spec,
        { label: fam.label, phase: 'Build+Tune', schema: SCHEMA, effort: 'high' }
      )
      out.push(r)
    }
    return out
  })
)

const done = laneResults.filter(Boolean).flat().filter(Boolean)
log(`${done.length}/7 families returned`)

// ---------------------------------------------------------------------------------
// Phase 2: adversarial fairness audit. The single biggest risk in a fused-vs-unfused
// study is a rigged baseline, so this reads the actual sources rather than the reports.
// ---------------------------------------------------------------------------------
phase('Audit')

const summary = done
  .map(
    (d) =>
      `- ${d.family} [${d.status}] files=${(d.files || []).join(', ')} ` +
      `verdict="${(d.verdict || '').slice(0, 300)}" issues="${(d.issues || '').slice(0, 300)}"`
  )
  .join('\n')

const audit = await agent(
  PREAMBLE +
    `
# ===== YOUR ASSIGNMENT: ADVERSARIAL FAIRNESS AUDIT =====

**Your exclusive GPU: CUDA_VISIBLE_DEVICES=0** (use it only if you need to re-run something;
the build agents are finished).

Seven agents just built fused/unfused Triton kernel pairs. They reported:

${summary}

Your job is to **try to break their results**. Read the ACTUAL SOURCE in
\`/home/zhangshuhan/fusion/glm52/kernels/\` and \`/home/zhangshuhan/fusion/glm52/bench/\` and
the raw \`/home/zhangshuhan/fusion/results/*.json\` — do NOT trust the summaries above or the
prose in \`log/\`. For every fused/unfused pair, check specifically:

1. **Rigged baseline.** Was the unfused side tuned over a grid of comparable size and quality
   to the fused side? Compare \`n_tried\` and the config tables in the JSON. If the fused grid
   has 120 configs and the unfused has 20, the speedup is an artifact. Flag it.
2. **Missing work.** Does the fused version skip an output the unfused version produces
   (e.g. not writing the new residual, not zeroing an atomic accumulator, not materializing a
   tensor a downstream consumer needs)? Check the launchers, not the kernels alone.
3. **Excluded setup cost.** Is any per-call cost (output zeroing, seeding, dispatch/align
   computation, dtype casts) inside the unfused timing but outside the fused timing, or vice
   versa? Both sides must include or exclude the same auxiliary work.
4. **Timing methodology.** Is \`bench_chain\` used with the whole unfused chain as one list
   (correct) or are per-kernel timings being summed (wrong — that adds a flush and full
   sync between kernels and inflates the unfused number)?
5. **Correctness.** Are the reported \`rel_err\` values actually computed against
   \`reference.py\` in fp32, and are they plausible (a suspiciously perfect 0.0 usually means
   the check compared a tensor against itself; a huge value means the kernel is wrong)?
   Re-run at least the two most surprising benchmarks yourself and confirm the numbers
   reproduce within the reported p10/p90 spread.
6. **Overclaimed verdicts.** Does the log's conclusion match its own numbers?

Then WRITE \`/home/zhangshuhan/fusion/log/LOG-08-fairness-audit.md\` containing: your
methodology, a per-family table (pass / concern / fail with the evidence), every finding with
the file and line, which results you personally re-ran and whether they reproduced, and a
clear statement of which reported speedups you consider trustworthy and which you do not.

Return findings ranked most-severe first. Be genuinely adversarial: a clean bill of health
for a study like this is a suspicious outcome, so look hard. But do not invent problems —
each finding must cite a specific file and line and explain the concrete failure mode.
`,
  {
    label: 'fairness-audit',
    phase: 'Audit',
    effort: 'high',
    schema: {
      type: 'object',
      properties: {
        findings: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              family: { type: 'string' },
              severity: { type: 'string', enum: ['fail', 'concern', 'pass'] },
              file: { type: 'string' },
              summary: { type: 'string' },
              evidence: { type: 'string' },
            },
            required: ['family', 'severity', 'summary'],
          },
        },
        reran: { type: 'string', description: 'which benchmarks were re-run and did they reproduce' },
        trustworthy: { type: 'string' },
        untrustworthy: { type: 'string' },
        log_file: { type: 'string' },
      },
      required: ['findings', 'trustworthy'],
    },
  }
)

return { families: done, audit }
