export const meta = {
  name: 'glm52-f9-batch-sweep',
  description: 'Fusion #9 kernel-level gain vs decode batch size: 20 points, both sides retuned per point, two unfused baselines, 3 GPU lanes',
  phases: [{ title: 'Sweep', detail: '20 batch sizes across GPUs 0, 1, 3' }],
}

const COMMON = `
You are running ONE benchmark command for a GPU kernel-fusion study on a **MetaX C500**, then
reporting its numbers. Do NOT modify any source file, do not change flags, do not "improve"
anything. The script is already smoke-tested; your job is to collect data accurately.

Environment:
- Python: /home/zhangshuhan/my-envs/fusion/bin/python
- Working dir: /home/zhangshuhan/fusion (run from here)
- **Your GPU is exclusive — prefix with CUDA_VISIBLE_DEVICES=<N>** (your N below). Two other
  agents benchmark on the other GPUs; touching them corrupts every timing. GPU 2 is
  hardware-dead — never use it.

What it measures: fusion #9 (MoE down-projection GEMM + expert merge + second residual add).
Per batch size it independently tunes the unfused GEMM, two moe_sum variants, the resadd, the
fused GEMM and the seed kernel; validates all three variants against an fp32 reference; then
times them over 5 interleaved rounds. It reports the fused time against TWO unfused
baselines — a 3-kernel chain and a strictly better 2-kernel chain.

Your command (ONE line; expect ~25-45 min — launch with nohup and poll):

    cd /home/zhangshuhan/fusion && CUDA_VISIBLE_DEVICES=<N> nohup \\
      /home/zhangshuhan/my-envs/fusion/bin/python -u glm52/bench/bench_f9_sweep.py \\
      --batches <BATCHES> --tag <TAG> > /tmp/f9_<TAG>.log 2>&1 & disown

Then poll /tmp/f9_<TAG>.log with a SINGLE long sleep between checks (do not chain short
sleeps) until it prints "wrote results/f9_sweep_<TAG>.json".

Lines containing "mcErrorMemoryValueTooLarge" / "private memory" during tuning are EXPECTED
and harmless — C500's hard 4 KB/thread cap rejecting candidates, which the autotuner catches.
Only report them if they appear OUTSIDE the tuning phase.

When done, report for EVERY batch size in your lane: T, rows_per_expert, fused_ms,
unfused2_ms, unfused3_ms, gain_vs_2k, gain_vs_3k, the gain spread across the 5 rounds, and the
max rel_err. Flag anything odd: gain spread above ~2%, rel_err above 5e-2, a non-monotonic
jump, or driver errors outside tuning.

IMPORTANT CONTEXT on interpreting your own numbers: a companion sweep of fusion #8 established
that BETWEEN-run variation on this machine is ~2.2% — roughly 20x the within-run spread these
5 rounds measure. So differences under ~2% between adjacent batch sizes are NOT resolved by
this protocol. Report the numbers; do not claim a trend from sub-2% wiggles.
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
          fused_ms: { type: 'number' },
          unfused2_ms: { type: 'number' },
          unfused3_ms: { type: 'number' },
          gain_vs_2k: { type: 'number' },
          gain_vs_3k: { type: 'number' },
          gain_spread_pct: { type: 'number' },
          max_rel_err: { type: 'number' },
        },
        required: ['T', 'gain_vs_2k', 'gain_vs_3k'],
      },
    },
    anomalies: { type: 'string' },
  },
  required: ['tag', 'results_json', 'complete', 'points'],
}

// Descending round-robin so the three lanes carry comparable total cost.
const LANES = [
  { gpu: 0, tag: 'lane0', batches: '1024,640,384,192,96,32,4' },
  { gpu: 1, tag: 'lane1', batches: '896,512,320,128,64,16,2' },
  { gpu: 3, tag: 'lane2', batches: '768,448,256,48,8,1' },
]

phase('Sweep')
log('Fusion #9 gain vs batch size: 20 points over GPUs 0/1/3, two unfused baselines each')

const out = await parallel(
  LANES.map((l) => () =>
    agent(
      COMMON +
        `\n# YOUR ASSIGNMENT\nGPU: **CUDA_VISIBLE_DEVICES=${l.gpu}**\n` +
        `BATCHES: **${l.batches}**\nTAG: **${l.tag}**\n` +
        `Output: results/f9_sweep_${l.tag}.json\n`,
      { label: `f9-${l.tag}-gpu${l.gpu}`, phase: 'Sweep', schema: SCHEMA }
    )
  )
)

return { lanes: out.filter(Boolean) }
