export const meta = {
  name: 'glm52-f8-f9-prefill-sweep',
  description: 'Fusions #8 and #9 gain vs prefill T, 256 to 262144, both sides retuned per point, 3 GPU lanes',
  phases: [{ title: 'Sweep', detail: '11 prefill sizes x 2 fusions across GPUs 0, 1, 3' }],
}

const COMMON = `
You are running benchmark commands for a GPU kernel-fusion study on a **MetaX C500**, then
reporting the numbers. Do NOT modify any source file, change flags, or "improve" anything.
Both scripts are smoke-tested; your job is to collect data accurately.

Environment:
- Python: /home/zhangshuhan/my-envs/fusion/bin/python
- Working dir: /home/zhangshuhan/fusion (run from here)
- **Your GPU is exclusive — prefix every command with CUDA_VISIBLE_DEVICES=<N>** (your N is
  below). Two other agents benchmark on the other GPUs; touching them corrupts every timing.
  GPU 2 is hardware-dead — never use it.

This extends an existing decode sweep (T=1..1024) into the PREFILL range, T=256..262144.
Fusion #8 = MoE down-projection GEMM + expert merge. Fusion #9 = the same plus the second
residual add, reported against two unfused baselines (2-kernel and 3-kernel).

**MEMORY IS TIGHT AT THE TOP END.** T=262144 allocates ~47 GiB of the 64 GiB card (verified).
Run your commands STRICTLY SEQUENTIALLY — never two python processes at once on your GPU, or
they will OOM each other. Wait for one to write its JSON before starting the next.

Launch each with nohup and poll with a SINGLE long sleep between checks (do not chain short
sleeps):

    cd /home/zhangshuhan/fusion && CUDA_VISIBLE_DEVICES=<N> nohup \\
      /home/zhangshuhan/my-envs/fusion/bin/python -u glm52/bench/<SCRIPT> \\
      --batches <BATCHES> --tag <TAG> > /tmp/<LOGNAME>.log 2>&1 & disown

Poll until the log prints "wrote results/<f8|f9>_sweep_<TAG>.json".

Expected duration: the largest points take ~20-25 min each; your whole assignment is roughly
25-45 min. Measurement effort scales down automatically with T (a single launch is ~1.7 s at
T=262144), so do not be alarmed by low rep counts in the log at large sizes — that is by design.

Lines containing "mcErrorMemoryValueTooLarge" / "private memory" during TUNING are expected
and harmless (C500's 4 KB/thread cap rejecting candidates; the autotuner catches them). Report
them only if they appear outside tuning, or if you see a CUDA OOM.

When done, report for EVERY (script, T) in your assignment the per-point numbers the script
prints. Flag: gain spread above ~2%, rel_err above 5e-2, OOM, non-monotonic jumps, or driver
errors outside tuning.

IMPORTANT on interpretation: between-run variation on this machine is ~2.2%, roughly 20x the
within-run spread these rounds measure. Differences under ~2% are NOT resolved. Report the
numbers; do not claim trends from sub-2% wiggles.
`

const SCHEMA = {
  type: 'object',
  properties: {
    lane: { type: 'string' },
    results_json: { type: 'array', items: { type: 'string' } },
    complete: { type: 'boolean' },
    points: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          fusion: { type: 'string' },
          T: { type: 'number' },
          fused_ms: { type: 'number' },
          unfused_ms: { type: 'number' },
          unfused3_ms: { type: 'number' },
          gain: { type: 'number' },
          gain_vs_3k: { type: 'number' },
          gain_spread_pct: { type: 'number' },
          max_rel_err: { type: 'number' },
        },
        required: ['fusion', 'T', 'gain'],
      },
    },
    anomalies: { type: 'string' },
  },
  required: ['lane', 'results_json', 'complete', 'points'],
}

const LANES = [
  { gpu: 0, lane: 'lane0',
    cmds: [['bench_f8_sweep.py', '262144', 'prefill_lane0'],
           ['bench_f9_sweep.py', '262144', 'prefill_lane0']] },
  { gpu: 1, lane: 'lane1',
    cmds: [['bench_f8_sweep.py', '131072,65536,32768', 'prefill_lane1'],
           ['bench_f9_sweep.py', '131072', 'prefill_lane1']] },
  { gpu: 3, lane: 'lane2',
    cmds: [['bench_f9_sweep.py', '65536,32768,16384,8192,4096,2048,1024,512,256', 'prefill_lane2'],
           ['bench_f8_sweep.py', '16384,8192,4096,2048,1024,512,256', 'prefill_lane2']] },
]

phase('Sweep')
log('Prefill sweep 256..262144 for #8 and #9 across GPUs 0/1/3 (T=262144 uses ~47 GiB — run serially)')

const out = await parallel(
  LANES.map((l) => () =>
    agent(
      COMMON +
        `\n# YOUR ASSIGNMENT\nGPU: **CUDA_VISIBLE_DEVICES=${l.gpu}**\n` +
        `Run these commands ONE AT A TIME, in this order:\n` +
        l.cmds
          .map((c, i) =>
            `  ${i + 1}. script=${c[0]}  --batches ${c[1]}  --tag ${c[2]}` +
            `   (log /tmp/${c[0].includes('f8') ? 'pf8' : 'pf9'}_${l.lane}.log,` +
            ` output results/${c[0].includes('f8') ? 'f8' : 'f9'}_sweep_${c[2]}.json)`)
          .join('\n') +
        `\n\nReport every point from both runs.\n`,
      { label: `prefill-${l.lane}-gpu${l.gpu}`, phase: 'Sweep', schema: SCHEMA }
    )
  )
)

return { lanes: out.filter(Boolean) }
