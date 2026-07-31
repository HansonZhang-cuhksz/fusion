# LOG-01 — Fusion #1: o_proj GEMM + Residual Add

**Status:** IN PROGRESS (numbers filled in at the end of the run)

**Date:** 2026-07-27 · **GPU:** MetaX C500, `CUDA_VISIBLE_DEVICES=1` (exclusive) ·
`torch 2.8.0+metax3.7.1.3`, `triton 3.0.0+metax`

**Deliverables**
- kernel: `glm52/kernels/oproj_resadd.py`
- bench: `glm52/bench/bench_f01_oproj_resadd.py`
- result: `results/f01_oproj_resadd.json`

---

## 1. What is being fused

The attention output projection of a GLM-5.2 decoder layer, followed by the first residual
add:

```
unfused:   C  = A @ B                    (GEMM kernel, writes [T,6144] bf16)
           h1 = C + residual             (elementwise kernel)

fused:     h1 = A @ B + residual         (GEMM kernel, residual folded into the epilogue)
```

This is the cuBLASLt `beta=1` / CUTLASS `LinearCombinationResidualBlock` pattern. The
production line to beat is `torch.addmm(residual, a, b)` (vendor BLAS, fused beta-accumulate)
versus `torch.mm(a, b)` + `torch.add` (vendor unfused).

Shapes (`glm52/config.py`): `N = hidden = 6144`; `K = 64 * kv_lora_rank = 32768` for the
decode regimes (absorbed MLA, `W_UV` folded into o_proj) and `K = 64 * v_head_dim = 16384`
for prefill (non-absorbed). dtype bf16, fp32 accumulate.

---

## 2. Memory-traffic analysis (ideal, no re-reads)

`A` is `[M,K]`, `B` is `[K,6144]`, `C`/`residual`/`h1` are `[M,6144]`, all bf16.

* unfused = read A + read B + **write C** + **read C** + read residual + write h1
* fused   = read A + read B + read residual + write h1
* saving  = one write + one read of `[M,6144]` bf16 = `2 * M * 6144 * 2` bytes

| regime | M | K | read A | read B | one C pass | unfused | fused | saved | saved % | saved @1.55 TB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| decode_bs1    | 1    | 32768 | 0.07 MB | 402.65 MB | 0.012 MB | 402.77 MB | 402.74 MB | 0.025 MB | 0.01 % | 0.00002 ms |
| decode_bs32   | 32   | 32768 | 2.10 MB | 402.65 MB | 0.393 MB | 406.32 MB | 405.54 MB | 0.786 MB | 0.19 % | 0.0005 ms |
| decode_bs256  | 256  | 32768 | 16.78 MB | 402.65 MB | 3.146 MB | 432.01 MB | 425.72 MB | 6.29 MB | 1.46 % | 0.0041 ms |
| prefill_t2048 | 2048 | 16384 | 67.11 MB | 201.33 MB | 25.17 MB | 369.10 MB | 318.77 MB | 50.33 MB | 13.6 % | 0.0325 ms |
| prefill_t8192 | 8192 | 16384 | 268.44 MB | 201.33 MB | 100.66 MB | 872.42 MB | 671.09 MB | 201.33 MB | 23.1 % | 0.130 ms |

The percentages above are of *ideal* traffic, not of runtime. The runtime denominators are
very different:

* **decode**: the GEMM reads a 402.65 MB weight for 1–256 tokens, so it is entirely
  weight-bandwidth-bound. The residual traffic is 0.01 %–1.5 % of the bytes and the
  arithmetic is negligible. Predicted fused gain: ~0 at bs1, ≤1.5 % at bs256.
* **prefill**: FLOPs are 412 GFLOP (t2048) and 1649 GFLOP (t8192). At the ~90–106 TF/s
  ceiling Triton reaches on this backend that is ≈4.6 ms and ≈18.3 ms, against 0.033 ms and
  0.130 ms of saved traffic — i.e. a **0.7 % ceiling** on the fusion gain at both prefill
  points, even though the fusion removes 14–23 % of the *bytes*.

So the honest prior before measuring is: this fusion is worth ≤1.5 % anywhere in this
model's shape range, and is worth measuring mainly to confirm the epilogue costs nothing.

---

*(sections 3–7 written after the run)*
