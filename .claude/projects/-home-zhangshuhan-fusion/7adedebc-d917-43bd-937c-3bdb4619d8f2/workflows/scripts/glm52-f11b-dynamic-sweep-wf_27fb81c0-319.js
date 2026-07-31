export const meta = {
  name: 'glm52-f11b-dynamic-sweep',
  description: 'Fusion #11b gain vs T across decode and prefill, three GPU workers draining a shared dynamic queue',
  phases: [{ title: 'Sweep', detail: '31 tasks, work-stealing across GPUs 0, 1, 3' }],
}

const COMMON = `
You are one of THREE workers draining a shared task queue for a GPU kernel-fusion study on a
**MetaX C500**. Do NOT modify any source file or change flags. The script is smoke-tested.

Environment:
- Python: /home/zhangshuhan/my-envs/fusion/bin/python
- Working dir: /home/zhangshuhan/fusion (run from here)
- **Your GPU is exclusive — prefix with CUDA_VISIBLE_DEVICES=<N>** (yours is below). The
  other two workers are on the other GPUs. GPU 2 is hardware-dead — never use it.

HOW THE QUEUE WORKS (you do not manage it — the script does): 31 tasks sit in
results/_f11b_queue/pending. Your worker claims one by atomic os.rename into running/, runs
it, writes results/f11b_<regime>_T<n>.json, moves the task to done/, and immediately claims
the next. It exits when pending/ is empty. Because claiming is atomic, the three workers
self-balance — you never wait on another worker, and no task is run twice. Tasks are ordered
so the largest T are claimed first, which is what keeps the tail from stranding one GPU.

Your single command (launch with nohup, then poll):

    cd /home/zhangshuhan/fusion && CUDA_VISIBLE_DEVICES=<N> nohup \\
      /home/zhangshuhan/my-envs/fusion/bin/python -u glm52/bench/bench_f11b_sweep.py \\
      --queue --worker <WORKER> > /tmp/f11b_<WORKER>.log 2>&1 & disown

Poll /tmp/f11b_<WORKER>.log with a SINGLE long sleep between checks (do not chain short
sleeps) until it prints "queue empty". Expect roughly 10-30 min depending on which tasks you
happen to draw — an uneven split across workers is the queue working correctly, not a fault.

What is being measured: fusion #11b, Lazy Pre-Norm folded into the router GEMM.
  UNFUSED = rmsnorm kernel (read h1, write x2) then router GEMM (read x2)
  FUSED   = one router GEMM reading un-normalized h1, accumulating the row sum-of-squares in
            the same k-loop, applying rstd as an epilogue scale; x2 never materialized.
Both sides are independently retuned at every T and validated against an fp32 reference.

Lines with "mcErrorMemoryValueTooLarge" / "private memory" during TUNING are expected and
harmless (C500's 4 KB/thread cap rejecting candidates). Report them only if they appear
outside tuning, or if you see a CUDA OOM or a task marked FAILED.

When your worker exits, report every task YOU completed: regime, T, unfused_ms, fused_ms,
gain, gain_spread_pct, sq_redundancy, and both rel_err values (all printed per task, and in
each results/f11b_*.json). Also report how many tasks you completed.

INTERPRETATION: between-run variation on this machine is ~2.2 % (an upper bound dominated by
tuning-quality differences, not jitter). Differences under ~2 % are NOT resolved. Report
numbers; do not claim trends from sub-2 % wiggles.
`

const SCHEMA = {
  type: 'object',
  properties: {
    worker: { type: 'string' },
    tasks_completed: { type: 'number' },
    points: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          regime: { type: 'string' },
          T: { type: 'number' },
          unfused_ms: { type: 'number' },
          fused_ms: { type: 'number' },
          gain: { type: 'number' },
          gain_spread_pct: { type: 'number' },
          sq_redundancy: { type: 'number' },
          rel_err_unfused: { type: 'number' },
          rel_err_fused: { type: 'number' },
        },
        required: ['regime', 'T', 'gain'],
      },
    },
    anomalies: { type: 'string' },
  },
  required: ['worker', 'tasks_completed', 'points'],
}

const WORKERS = [
  { gpu: 0, name: 'gpu0' },
  { gpu: 1, name: 'gpu1' },
  { gpu: 3, name: 'gpu3' },
]

phase('Sweep')
log('31 tasks (20 decode + 11 prefill) drained dynamically by 3 GPU workers')

const out = await parallel(
  WORKERS.map((w) => () =>
    agent(
      COMMON +
        `\n# YOUR ASSIGNMENT\nGPU: **CUDA_VISIBLE_DEVICES=${w.gpu}**\n` +
        `WORKER name: **${w.name}**  (log /tmp/f11b_${w.name}.log)\n` +
        `Start immediately — the other workers are starting at the same time.\n`,
      { label: `f11b-${w.name}`, phase: 'Sweep', schema: SCHEMA }
    )
  )
)

return { workers: out.filter(Boolean) }
