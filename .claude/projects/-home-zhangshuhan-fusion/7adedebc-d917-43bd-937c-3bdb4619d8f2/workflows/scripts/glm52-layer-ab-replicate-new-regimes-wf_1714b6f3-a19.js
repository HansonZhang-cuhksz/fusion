export const meta = {
  name: 'glm52-layer-ab-replicate-new-regimes',
  description: 'Second independent interleaved A/B run for decode_bs512 and decode_bs1024, one regime per GPU',
  phases: [
    { title: 'Replicate', detail: 'independent A/B re-measurement, bs512 on GPU1, bs1024 on GPU3' },
  ],
}

const COMMON = `
You are running one benchmark command for a GPU kernel-fusion study on a **MetaX C500**, then
sanity-checking its output. Do not modify any source file. Do not tune anything. Do not
"improve" the script. The point is an INDEPENDENT REPLICATE of a measurement that already
exists, so the two runs can be compared — changing anything would defeat that.

Environment:
- Python: /home/zhangshuhan/my-envs/fusion/bin/python
- Working dir: /home/zhangshuhan/fusion  (run from here)
- **Your GPU is exclusive. Prefix the command with CUDA_VISIBLE_DEVICES=<N>** (your N below).
  Another agent is benchmarking on the other GPU; never touch it. GPU 2 is hardware-dead.
- The run takes roughly 15-25 minutes. Launch it with nohup writing to a log file, then poll
  the log rather than blocking a foreground call for that long, e.g.:

    cd /home/zhangshuhan/fusion && CUDA_VISIBLE_DEVICES=<N> nohup \\
      /home/zhangshuhan/my-envs/fusion/bin/python -u glm52/bench/bench_layer_ab.py \\
      --regimes <REGIME> --rounds 8 --rep 15 --top 5 --force K_f3_f8,D_f6 \\
      --out <OUTNAME> > /tmp/ab_<REGIME>.log 2>&1 & disown

  Then check /tmp/ab_<REGIME>.log periodically until it prints "wrote results/<OUTNAME>.json".
  Do NOT chain short sleeps in a loop to poll; use a single long sleep between checks.

What the script does: it times ~7 whole-layer fusion CONFIGURATIONS end to end, over 8
interleaved rounds (within each round every configuration is timed once, in a fixed order, so
drift affecting a whole round cancels in the per-configuration median). It reports a winner
only when its gap to the runner-up exceeds the round-to-round spread of both; otherwise it
reports them as tied.

When it finishes, verify and report:
1. The output JSON exists and parses: results/<OUTNAME>.json
2. Every configuration passed correctness (the script drops any that fail against its fp32
   reference and prints "FAILS CORRECTNESS" — report whether any did).
3. Per configuration: the median ms, the round-to-round spread percentage, and whether it is
   in the reported tied set.
4. The verdict line: best configuration, gap to runner-up, round noise, separated-or-tied,
   and the speedup vs the all-unfused baseline.
5. Anything anomalous: a spread above ~1%, a configuration whose median is wildly different
   from the others, or MACA driver errors in the log (lines containing MCR/MXKW/Xnack).

Report the numbers. Do not interpret whether fusion is "worth it" — that judgement is made
elsewhere with both runs in hand.
`

const SCHEMA = {
  type: 'object',
  properties: {
    regime: { type: 'string' },
    results_json: { type: 'string' },
    all_correct: { type: 'boolean' },
    best: { type: 'string' },
    separated: { type: 'boolean' },
    tied_set: { type: 'array', items: { type: 'string' } },
    speedup_vs_unfused: { type: 'number' },
    configs: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          median_ms: { type: 'number' },
          spread_pct: { type: 'number' },
          in_tied_set: { type: 'boolean' },
        },
        required: ['name', 'median_ms'],
      },
    },
    anomalies: { type: 'string' },
  },
  required: ['regime', 'results_json', 'all_correct', 'best', 'configs'],
}

phase('Replicate')
log('Second independent A/B: decode_bs512 on GPU 1, decode_bs1024 on GPU 3')

const results = await parallel([
  () =>
    agent(
      COMMON +
        `\n# YOUR ASSIGNMENT\nGPU: **CUDA_VISIBLE_DEVICES=1**\nREGIME: **decode_bs512**\n` +
        `OUTNAME: **layer_configurations_ab_new2_bs512**\n`,
      { label: 'ab2-bs512', phase: 'Replicate', schema: SCHEMA }
    ),
  () =>
    agent(
      COMMON +
        `\n# YOUR ASSIGNMENT\nGPU: **CUDA_VISIBLE_DEVICES=3**\nREGIME: **decode_bs1024**\n` +
        `OUTNAME: **layer_configurations_ab_new2_bs1024**\n`,
      { label: 'ab2-bs1024', phase: 'Replicate', schema: SCHEMA }
    ),
])

return { replicates: results.filter(Boolean) }
