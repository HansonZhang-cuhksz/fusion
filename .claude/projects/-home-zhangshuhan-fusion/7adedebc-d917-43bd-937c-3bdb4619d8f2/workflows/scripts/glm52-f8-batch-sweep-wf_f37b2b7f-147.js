export const meta = {
  name: 'glm52-f8-batch-sweep',
  description: 'Fusion #8 kernel-level gain vs decode batch size: 20 points, both sides retuned per point, 3 GPU lanes',
  phases: [{ title: 'Sweep', detail: '20 batch sizes split across GPUs 0, 1, 3' }],
}

const COMMON = `
You are running ONE benchmark command for a GPU kernel-fusion study on a **MetaX C500**, then
reporting its numbers. Do NOT modify any source file, do not change flags, do not "improve"
anything. The script is already validated; the point is to collect data.

Environment:
- Python: /home/zhangshuhan/my-envs/fusion/bin/python
- Working dir: /home/zhangshuhan/fusion (run from here)
- **Your GPU is exclusive — prefix with CUDA_VISIBLE_DEVICES=<N>** (your N below). Two other
  agents are benchmarking on the other GPUs; touching them corrupts every timing. GPU 2 is
  hardware-dead, never use it.

What it measures: fusion #8 (MoE down-projection GEMM + expert merge). For each batch size it
independently tunes BOTH the unfused side (w2 grouped GEMM, then moe_sum over top-8) and the
fused side (w2 GEMM with sglang's FUSE_SUM_ALL_REDUCE atomic-accumulate epilogue, including
the output zero-init it requires), validates both against an fp32 reference, then times each
over 5 interleaved rounds and reports the median gain = unfused/fused.

Your command (ONE line; takes 30-70 min depending on lane — launch with nohup and poll):

    cd /home/zhangshuhan/fusion && CUDA_VISIBLE_DEVICES=<N> nohup \\
      /home/zhangshuhan/my-envs/fusion/bin/python -u glm52/bench/bench_f8_sweep.py \\
      --batches <BATCHES> --tag <TAG> > /tmp/f8_<TAG>.log 2>&1 & disown

Then poll /tmp/f8_<TAG>.log with a SINGLE long sleep between checks (do not chain short
sleeps) until it prints "wrote results/f8_sweep_<TAG>.json".

Note: lines containing "mcErrorMemoryValueTooLarge" / "private memory" are EXPECTED and
harmless — that is C500's hard 4 KB/thread cap rejecting some tuning candidates, which the
autotuner catches and skips. Only report them if they appear OUTSIDE the tuning phase.

When done, report for EVERY batch size in your lane: T, rows_per_expert, unfused_ms, fused_ms,
gain, the gain spread across the 5 rounds, and both rel_err values. Flag anything odd: a gain
spread above ~2 %, a rel_err above 5e-2, a non-monotonic jump, or driver errors outside tuning.

Do not interpret whether the fusion is worth it. Just report the numbers accurately.
`

const SCHEMA = {
  type: 'object',
  properties: {
    tag: { type: 'string' },
    results_json: { type: 'string' },
    complete: { type: 'boolean' },
    points: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          T: { type: 'number' },
          rows_per_expert: { type: 'number' },
          unfused_ms: { type: 'number' },
          fused_ms: { type: 'number' },
          gain: { type: 'number' },
          gain_spread_pct: { type: 'number' },
          rel_err_unfused: { type: 'number' },
          rel_err_fused: { type: 'number' },
        },
        required: ['T', 'gain'],
      },
    },
    anomalies: { type: 'string' },
  },
  required: ['tag', 'results_json', 'complete', 'points'],
}

// Descending round-robin so the three lanes carry comparable total cost (larger T costs more).
const LANES = [
  { gpu: 0, tag: 'lane0', batches: '1024,640,384,192,96,32,4' },
  { gpu: 1, tag: 'lane1', batches: '896,512,320,128,64,16,2' },
  { gpu: 3, tag: 'lane2', batches: '768,448,256,48,8,1' },
]

phase('Sweep')
log('Fusion #8 gain vs batch size: 20 points over 3 GPU lanes, both sides retuned per point')

const out = await parallel(
  LANES.map((l) => () =>
    agent(
      COMMON +
        `\n# YOUR ASSIGNMENT\nGPU: **CUDA_VISIBLE_DEVICES=${l.gpu}**\n` +
        `BATCHES: **${l.batches}**\nTAG: **${l.tag}**\n` +
        `Output file will be results/f8_sweep_${l.tag}.json\n`,
      { label: `f8-${l.tag}-gpu${l.gpu}`, phase: 'Sweep', schema: SCHEMA }
    )
  )
)

return { lanes: out.filter(Boolean) }
