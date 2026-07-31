export const meta = {
  name: 'glm52-moe-fusion-kernels-relaunch',
  description: 'Relaunch the 3 GLM-5.2 fusion families killed by a session limit: F4/F5, F10, F11',
  phases: [
    { title: 'Build+Tune', detail: 'F11 on GPU1; F4/F5 then F10 on GPU3' },
  ],
}

const PREAMBLE = `
# Context

You are building Triton kernels for a **GLM-5.2 MoE decoder layer** on a **MetaX C500** GPU
(a domestic Chinese accelerator with a CUDA-compatible "MACA" stack). The goal is to measure
the benefit (or harm) of specific **kernel fusions** by comparing a fused implementation
against an unfused one, each independently tuned.

Four sibling families already completed; their kernels and benchmarks are in the repo and are
good models to imitate. **Read \`glm52/kernels/add_rmsnorm.py\` and
\`glm52/bench/bench_f03_resadd_rmsnorm.py\` before you start** — that family scored the
cleanest result in the study and its structure is the house style.

## Environment — use exactly this

- Python: \`/home/zhangshuhan/my-envs/fusion/bin/python\`  (torch 2.8.0+metax, triton 3.0.0+metax)
- Working dir: \`/home/zhangshuhan/fusion\` — run scripts from here so \`glm52\` imports work.
- **YOUR GPU IS EXCLUSIVE. Prefix EVERY python command with \`CUDA_VISIBLE_DEVICES=<N>\`**
  (your N is stated below). Another agent is benchmarking on GPU 0 right now; never touch it.
  Do not call \`torch.cuda.set_device\`. GPU 2 is hardware-dead — do not try it.
- Scratch: \`/tmp/claude-1025/-home-zhangshuhan-fusion/7adedebc-d917-43bd-937c-3bdb4619d8f2/scratchpad\`
- sglang 0.5.10 source (reference kernels) is extracted at
  \`/tmp/claude-1025/-home-zhangshuhan-fusion/7adedebc-d917-43bd-937c-3bdb4619d8f2/scratchpad/sgl/sglang-0.5.10.post1/\`

## Hardware facts that constrain your mapping search

- **warp = 64 lanes** (NOT 32). \`num_warps=4\` is 256 threads, 8 is 512, 16 is 1024 = block ceiling.
- **Shared memory = 65536 B per CTA** — a hard ceiling that kills many NVIDIA-shaped configs.
  Pre-filter with \`num_stages * 2 * BLOCK_K * (BLOCK_M + BLOCK_N)\` before adding to a grid.
- 104 CUs, 131072 registers/SM, **8 MB L2**.
- **Measured peaks for this study** (recalibrated from our own runs — use these, they are in
  \`glm52/traffic.py\`): Triton bf16 compute ceiling **107 TF/s** (the vendor BLAS reaches ~215,
  a backend limitation you should not try to close), achievable HBM bandwidth **1.29 TB/s**.
- Best generic Triton GEMM config found: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8,
  num_stages=2, GROUP_M=8. Seed your grid with it.
- Triton 3.0 on MACA: plain pointer arithmetic (sglang's \`fused_moe_kernel\` style). No
  clusters/DSMEM/TMA. \`tl.atomic_add\` on fp32 is confirmed working. Compile ONE trivial
  kernel before building a grid on any Triton feature you are unsure about.

## Model shapes — GLM-5.2 (constants live in \`glm52/config.py\`; import, do not retype)

hidden H=6144, moe_intermediate I=2048, experts E=256, top-k=8, shared experts=1,
w13 \`[E, 2*I, H]\` (gate then up, sglang convention), w2 \`[E, H, I]\`, router weight
\`[256, 6144]\`, sigmoid scoring + noaux_tc top-k, routed_scaling_factor 2.5, norm_topk_prob
true, rms_norm_eps 1e-5, SwiGLU = silu(gate)*up, dtype **bfloat16**, fp32 accumulate,
**router math in fp32** (\`moe_router_dtype: float32\`).

## The 5 benchmark regimes (tune AND report at every one)

| name | T (tokens) | o_proj K | moe rows = T*8 |
|---|---|---|---|
| decode_bs1 | 1 | 32768 | 8 |
| decode_bs32 | 32 | 32768 | 256 |
| decode_bs256 | 256 | 32768 | 2048 |
| prefill_t2048 | 2048 | 16384 | 16384 |
| prefill_t8192 | 8192 | 16384 | 65536 |

## Harness API — \`glm52/common.py\` (READ IT FIRST; do not reinvent timing)

- \`bench_chain(fns, warmup, rep, flush=True) -> Timing\` — times a LIST of callables
  back-to-back as one logical op, ONE L2 flush before each repetition of the whole chain
  (never between its kernels). \`.p50_ms/.p10_ms/.p90_ms/.noflush_p50_ms\`.
- \`autotune(make_chain, configs, warmup, rep) -> TuneResult\` — brute-force search;
  compile failures recorded, not fatal. \`.best_cfg/.best_ms/.n_tried/.n_failed/.table\`.
- \`check(got, ref, tol=2e-2, label)\`, \`record(name, payload)\`, \`speedup_row(...)\`.
- \`glm52/reference.py\` — \`rmsnorm\`, \`add_rmsnorm\`, \`silu_and_mul\`, \`router\` (sigmoid +
  noaux_tc), \`moe_align_block_size\` (port of sglang's), \`moe_mlp\`, \`expert_merge\`.
- \`glm52/traffic.py\` — run \`python -m glm52.traffic\` to get the **latency-aware roofline
  ceiling** for your fusion at each regime. Compare every measured speedup against it and say
  what fraction of the ceiling you achieved. A measured speedup ABOVE the ceiling means
  something is wrong (or the win came from launch overhead) — investigate, do not celebrate.

## THE FAIRNESS RULES — these decide whether the result is worth anything

1. **One kernel source, flags differ.** Write the kernel ONCE with \`tl.constexpr\` flags
   selecting the fused epilogue/prologue. The unfused variant is *the same kernel* with the
   flag off, plus a separate kernel for the split-out work. Only the **mapping** may differ.
2. **Tune both sides independently AND WITH GRIDS OF THE SAME SIZE.** A sibling agent tuned
   its fused side over 45 configs and its unfused side over 79; that asymmetry did not change
   its answer but it made the result unauditable and cost the main session a full re-run to
   confirm. Generate both grids from the same rules, record \`n_tried\` for each, and if one
   side legitimately has fewer valid configs (e.g. SMEM prefilter), say so explicitly.
3. **Both sides must do the same work** and produce comparable outputs. If the fused version
   legitimately avoids materializing an intermediate, that IS the benefit — state it — but it
   must not skip an output a downstream consumer needs.
4. **Validate before you time**, against the fp32 \`reference.py\`, with \`check()\`. Report
   \`rel_err\` in every row.
5. **Report losses honestly.** Two of the four completed families REGRESS (see below). A
   negative result is a real result. Never under-tune the baseline to manufacture a win.
6. **RECORD THE FULL \`TuneResult.table\` IN YOUR JSON.** One sibling left \`tune_tables\`
   empty and its headline result could not be audited without re-running it. Do not repeat that.

## What the completed families found (context for your own numbers)

| fusion | decode | prefill | mechanism |
|---|---|---|---|
| #3 ResAdd+RMSNorm | 1.08–1.11× | **1.25–1.32×** | bandwidth-bound, hits its 1.25× ceiling |
| #1 o_proj+ResAdd | ~1.00× | **0.85–0.87×** | regresses |
| #6 UpGate+SwiGLU | 0.96–0.99× | **0.55–0.77×** | regresses badly |

Two attribution results you should reuse as diagnostic technique (full detail in
\`log/LOG-10-main-session-findings.md\`):

- **F1's regression is a codegen cliff.** Registers were *identical* (126 vs 126) with zero
  spills, and the epilogue's extra DRAM traffic cost only **+0.1 %** — proven by pointing the
  residual at a stride-0 broadcast (\`r[0:1].expand(M,N)\`), same instructions, no traffic.
  The epilogue's mere presence dropped the kernel 107.5 → 87.4 TF/s by disabling the mainloop
  schedule that made \`BLOCK_K=32, GROUP_M>=8\` fast.
- **F6's regression is register pressure + a hard SMEM ceiling.** Two accumulators doubled
  registers (104 → 214, 160 → 242), halving CTAs/SM; and the unfused winner's tile
  \`BM128 BN128\` is **uncompilable** when fused (needs 96 KB SMEM vs the 64 KB limit).

**So if your fused kernel is slower, diagnose it the same way before concluding anything:**
(i) compile both variants with the Triton cache cleared between compiles
(\`kernel.cache[dev].clear()\`) and compare \`n_regs\`/\`n_spills\`; (ii) point the fused
kernel's extra input at a stride-0 broadcast to separate instruction cost from DRAM traffic
cost. Report both.

## Deliverables (paths relative to /home/zhangshuhan/fusion)

1. \`glm52/kernels/<module>.py\` — Triton kernel(s) + thin launchers.
2. \`glm52/bench/<script>.py\` — runnable standalone with
   \`CUDA_VISIBLE_DEVICES=<N> /home/zhangshuhan/my-envs/fusion/bin/python glm52/bench/<script>.py\`;
   validates, autotunes both sides at all 5 regimes, writes \`results/<id>.json\` via \`record()\`.
   Consider per-regime checkpointing (a sibling did this and it saved a long run).
3. \`results/<id>.json\` — actually RUN it. Do not deliver untested code.
4. \`log/<LOGFILE>\` — plan, traffic analysis (bytes moved fused vs unfused), what you built,
   the mapping search space and both winning configs, results table (fused ms / unfused ms /
   speedup / % of roofline ceiling / rel_err per regime), the vendor-BLAS reference line where
   a GEMM is involved, surprises, and an honest verdict on whether the fusion is worth it.

Write real, working, tuned code and iterate until the benchmark runs and the numbers are
trustworthy. If something is genuinely impossible on this hardware, document precisely why
with numbers.
`

const SCHEMA = {
  type: 'object',
  properties: {
    family: { type: 'string' },
    status: { type: 'string', enum: ['complete', 'partial', 'failed'] },
    files: { type: 'array', items: { type: 'string' } },
    results_json: { type: 'string' },
    log_file: { type: 'string' },
    rows: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          variant: { type: 'string' },
          regime: { type: 'string' },
          fused_ms: { type: 'number' },
          unfused_ms: { type: 'number' },
          speedup: { type: 'number' },
          ceiling: { type: 'number' },
          rel_err: { type: 'number' },
        },
        required: ['variant', 'regime', 'fused_ms', 'unfused_ms', 'speedup'],
      },
    },
    verdict: { type: 'string' },
    issues: { type: 'string' },
  },
  required: ['family', 'status', 'files', 'rows', 'verdict'],
}

const F11 = PREAMBLE + `
# ===== YOUR ASSIGNMENT =====
**Your exclusive GPU: CUDA_VISIBLE_DEVICES=1**
**Result id: \`f11_lazy_prenorm\`** → \`results/f11_lazy_prenorm.json\`, \`glm52/bench/bench_f11_lazy_prenorm.py\`
**Log file: \`log/LOG-07-F11-lazy-prenorm.md\`**

## Your fusion: **#11 Lazy Pre-Norm** — RMSNorm fused as a GEMM **prologue**

From the PyTorch/Meta blog "Towards Free Normalization: Fusing Normalization into GEMM and
Attention Kernels" (Zhou et al., 2026-07-10), section 2. The PDF is at
\`/home/zhangshuhan/fusion/Towards Free Normalization_ Fusing Normalization into GEMM and Attention Kernels – PyTorch.pdf\`
— read pages 7–11 for the algorithm and its stated limits.

**The identity:** for affine-free RMSNorm, row-scaling commutes with matmul:
\`(A * rstd[:, None]) @ B == (A @ B) * rstd[:, None]\`. So a GEMM CTA accumulates
\`acc += tile_A @ tile_B\` and \`sq_sum += (tile_A * tile_A).sum(-1)\` in the SAME K-loop, then
applies \`rstd = rsqrt(sq_sum / K + eps)\` as an **epilogue scale**. The cyclic dependency
(needing rstd before the K-loop) disappears.

**Handling GLM-5.2's affine weight** (the paper calls this a blocker; it is not one at
inference): \`w\` is a column-wise scale of A and B is a constant weight matrix, so
\`((A * rstd) * w) @ B == (A @ (w[:, None] * B)) * rstd\`. Pre-fold \`w\` into B's rows
**offline, outside the timed region** — it is a load-time weight transform. Validate the
folded result against the unfolded \`reference.rmsnorm\` + matmul.

**Apply it to the two consumers of \`post_attention_layernorm\` whose K == hidden == 6144:**
  - **(a) the routed-expert w13 grouped GEMM** — \`rmsnorm(x2) @ w13[e]\`. Use
    \`reference.moe_align_block_size\` for dispatch, same as sglang's \`fused_moe_kernel\`.
    You may reuse \`glm52/kernels/moe_gateup.py\` as the structural base (read it first).
  - **(b) the router GEMM** — \`rmsnorm(x2) @ W_gate.T\`, \`[T,6144] @ [6144,256]\`, fp32 math.

**UNFUSED:** a standalone RMSNorm kernel writing \`x2\`, then the GEMM reading \`x2\`.
**FUSED:** the GEMM reads the UN-normalized \`h1\` directly and normalizes in its epilogue.

**The catch you must quantify:** the sum-of-squares is recomputed by every CTA sharing an
m_tile but owning a different n_tile. For the router GEMM N=256 → 1–2 n_tiles → redundancy
~1× (near-ideal). For the w13 GEMM N=4096 → \`4096/BLOCK_N\` n_tiles → redundancy **16–64×**.
Measure whether it hides behind the MMA pipeline as the paper claims. **Report the redundancy
factor for every config you select.**

**Roofline ceilings** (\`python -m glm52.traffic\`): F11a (w13) is **~1.00×** at every regime —
expert-weight traffic swamps everything, so expect no win there and say so. F11b (router) is
**1.01× decode → 1.79× at prefill**. F11b is where the value is; give it the most attention.

**CRITICAL correctness note:** \`x2\` is consumed by BOTH the router and the expert GEMMs, so a
fused variant that never materializes \`x2\` is only valid if ALL consumers are fused. Handle
this explicitly: either (i) time the fused variant as also materializing \`x2\` (the fusion
still saves the read), or (ii) fuse all consumers and state that as a precondition. Pick one,
implement it, and be explicit in the log about which and why.

Report the vendor-BLAS reference line too.
`

const F4F5 = PREAMBLE + `
# ===== YOUR ASSIGNMENT =====
**Your exclusive GPU: CUDA_VISIBLE_DEVICES=3**
**Result id: \`f04f05_norm_router\`** → \`results/f04f05_norm_router.json\`, \`glm52/bench/bench_f04f05_norm_router.py\`
**Log file: \`log/LOG-03-F4F5-norm-router.md\`**

## Your fusions: **#4 ResAdd + RMSNorm + Router** and **#5 RMSNorm + Router**

Fuse the router's small GEMM into the normalization kernel. The router is
\`x2[T, 6144] @ W_gate.T[6144, 256]\` in **fp32** (\`moe_router_dtype: float32\`), followed by
sigmoid scoring and noaux_tc top-8 selection — match \`reference.router\` exactly, including
\`norm_topk_prob\` and \`routed_scaling_factor=2.5\`.

**Why this can work despite being a GEMM:** \`W_gate\` is 6144*256*2 B = **3.0 MB** against an
**8 MB** L2, so every CTA's re-read of the router weight hits L2, not HBM. Verify that
empirically — if the scaling with T says otherwise, report it.

Build ONE kernel with constexpr flags: \`HAS_RESIDUAL\` (#4 vs #5) and \`FUSE_TOPK\`. Each
program owns a block of rows, computes the normed row in registers/SMEM, then multiplies by
\`W_gate\` to get all 256 logits for that row. Because all 256 logits for a row live in one
program, sigmoid + top-8 can ALSO be folded in — build that as \`FUSE_TOPK\` and report it as
an extra data point (beyond the user's original list, but the natural production endpoint).

Note \`x2\` must still be written out — the expert GEMMs consume it. The fusion saves the
router kernel's *read* of \`x2\`, not the write.

**Variants to measure, each independently tuned:**
  - #5 unfused: rmsnorm kernel → router GEMM kernel [→ topk kernel]
  - #5 fused: rmsnorm+router in one kernel [+ FUSE_TOPK]
  - #4 unfused: add kernel → rmsnorm kernel → router GEMM [→ topk]
  - #4 fused: add+rmsnorm+router in one kernel [+ FUSE_TOPK]

\`glm52/kernels/add_rmsnorm.py\` (family #3) already has a tuned add+rmsnorm kernel — read it
and build on its structure, but **do your own tuning**; do not import its configs.

**Roofline ceilings** (\`python -m glm52.traffic\`): #5 is 1.00× (bs1) → **1.79×** (prefill);
#4 is 1.01× → **1.83×**. Note these ceilings EXCEED the pure traffic ratio (1.45×/1.49×),
because fusing lets the memory-bound norm hide behind the router GEMM's compute — that is the
"free normalization" mechanism and it is the most promising fusion left in the study. At T=1
the router is a GEMV; at T=8192 it is 25.8 GFLOP. Expect and report very different verdicts
across regimes.

Report the vendor line: \`reference.router\` on top of a torch add+rmsnorm.
`

const F10 = PREAMBLE + `
# ===== YOUR ASSIGNMENT =====
**Your exclusive GPU: CUDA_VISIBLE_DEVICES=3**
**Result id: \`f10_merge_resadd\`** → \`results/f10_merge_resadd.json\`, \`glm52/bench/bench_f10_merge_resadd.py\`
**Log file: \`log/LOG-06-F10-merge-resadd.md\`**

## Your fusion: **#10 Expert Merge + Residual Add** (pure memory-bound vector fusion)

The tail of the MoE block. Input: per-expert outputs \`[T, topk=8, 6144]\` bf16 (unweighted),
routing weights \`[T, 8]\` fp32, residual \`[T, 6144]\`. Output: \`[T, 6144]\`.

**UNFUSED:** a merge kernel computing \`sum_k w_k * y[t,k,:]\` → \`[T, 6144]\`, then a separate
residual-add kernel.
**FUSED:** one kernel doing the weighted top-k reduction and the residual add before a single store.

Traffic: unfused reads \`T*8*6144*2\` + writes \`T*6144*2\`, then reads \`2*T*6144*2\` and writes
\`T*6144*2\`. Fused reads \`T*8*6144*2\` + \`T*6144*2\` and writes \`T*6144*2\`. The top-k input
dominates (8 units vs 1), so the ceiling is \`(8+4)/(8+2)\` = **1.20×** at every regime
(confirm with \`python -m glm52.traffic\`). This is a bandwidth kernel and it should get close
to that ceiling — family #3 reached 100 % of its 1.25× ceiling, so 1.20× is the bar.

Mapping search: rows/program, BLOCK over hidden, whether to loop over topk or unroll it
(topk=8 is compile-time constant — make it constexpr and tune both), num_warps, num_stages,
vectorization width, and a persistent-grid variant. \`glm52/kernels/add_rmsnorm.py\` is the
house style for this kind of kernel — read it first.

Report the torch eager reference (\`reference.expert_merge\` + add), and torch.compile /
inductor if it works on this backend (note it if it does not).

Also: the F8/F9 family fuses this reduction directly into the down-GEMM, eliminating your
kernel entirely. Their results are at \`results/f08f09_down_merge_resadd.json\` and
\`log/LOG-05-F8F9-down-merge-resadd.md\`. Read them at the END and state in your log how the
two paths compare, so the reader sees them side by side. Do not block on it.
`

phase('Build+Tune')
log('Relaunching F4/F5, F10, F11 — the 3 families killed by the 12:10 UTC session limit')

const [gpu1, gpu3] = await parallel([
  async () => await agent(F11, { label: 'F11-lazy-prenorm', phase: 'Build+Tune', schema: SCHEMA, effort: 'high' }),
  async () => {
    const a = await agent(F4F5, { label: 'F4F5-norm-router', phase: 'Build+Tune', schema: SCHEMA, effort: 'high' })
    const b = await agent(F10, { label: 'F10-merge-resadd', phase: 'Build+Tune', schema: SCHEMA, effort: 'high' })
    return [a, b]
  },
])

return { f11: gpu1, gpu3_lane: gpu3 }
