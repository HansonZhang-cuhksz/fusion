"""GLM-5.2 (zai-org/GLM-5.2, `glm_moe_dsa`) architecture constants and benchmark shapes.

Source of truth: https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json
Fetched 2026-07-27. Every number below is copied verbatim from that config; nothing
is inferred except the DERIVED block, which is annotated with its arithmetic.
"""

from dataclasses import dataclass, field

import torch

# --------------------------------------------------------------------------------------
# Verbatim from zai-org/GLM-5.2 config.json
# --------------------------------------------------------------------------------------
HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 12288  # dense MLP (first 3 layers only)
MOE_INTERMEDIATE_SIZE = 2048  # routed + shared expert MLP
NUM_HIDDEN_LAYERS = 78
FIRST_K_DENSE_REPLACE = 3  # layers [0,3) dense, [3,78) MoE
NUM_ATTENTION_HEADS = 64
NUM_KEY_VALUE_HEADS = 64
N_ROUTED_EXPERTS = 256
N_SHARED_EXPERTS = 1
NUM_EXPERTS_PER_TOK = 8  # top-k
ROUTED_SCALING_FACTOR = 2.5
NORM_TOPK_PROB = True
SCORING_FUNC = "sigmoid"
TOPK_METHOD = "noaux_tc"
N_GROUP = 1
TOPK_GROUP = 1
MOE_ROUTER_DTYPE = torch.float32

# MLA
KV_LORA_RANK = 512
Q_LORA_RANK = 2048
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = 256  # 192 + 64
V_HEAD_DIM = 256

# DSA (DeepSeek Sparse Attention) indexer
INDEX_HEAD_DIM = 128
INDEX_N_HEADS = 32
INDEX_TOPK = 2048

RMS_NORM_EPS = 1e-5
HIDDEN_ACT = "silu"  # SwiGLU = silu(gate) * up
VOCAB_SIZE = 154880
DTYPE = torch.bfloat16
ACC_DTYPE = torch.float32

# --------------------------------------------------------------------------------------
# DERIVED
# --------------------------------------------------------------------------------------
# Non-absorbed MLA (prefill path): attention emits [T, num_heads, v_head_dim].
OPROJ_K_PREFILL = NUM_ATTENTION_HEADS * V_HEAD_DIM  # 64 * 256 = 16384
# Absorbed MLA (FlashMLA decode path): attention emits [T, num_heads, kv_lora_rank]
# and W_UV is folded into o_proj.
OPROJ_K_DECODE = NUM_ATTENTION_HEADS * KV_LORA_RANK  # 64 * 512 = 32768

# Gate+Up are stored fused as w13 = [2 * moe_intermediate, hidden], matching
# sglang's FusedMoE layout (w13_weight / w2_weight).
W13_N = 2 * MOE_INTERMEDIATE_SIZE  # 4096
W2_K = MOE_INTERMEDIATE_SIZE  # 2048


@dataclass(frozen=True)
class Regime:
    """One benchmark point: `T` tokens through one MoE decoder layer."""

    name: str
    T: int  # number of tokens in the batch (batch*seq for prefill, batch for decode)
    oproj_k: int
    kv_len: int = 0  # context length, for attention-core reference timing only

    @property
    def moe_rows(self) -> int:
        """Rows entering the routed-expert GEMMs = T * topk."""
        return self.T * NUM_EXPERTS_PER_TOK


DECODE_REGIMES = [
    Regime(f"decode_bs{t}", t, OPROJ_K_DECODE, kv_len=4096)
    for t in (1, 8, 32, 64, 128, 256)
]
PREFILL_REGIMES = [
    Regime(f"prefill_t{t}", t, OPROJ_K_PREFILL, kv_len=t) for t in (2048, 4096, 8192)
]
ALL_REGIMES = DECODE_REGIMES + PREFILL_REGIMES


@dataclass
class BenchEnv:
    """Device/environment facts probed once, recorded into every result file."""

    device_name: str = ""
    warp_size: int = 0
    num_sm: int = 0
    smem_bytes: int = 0
    smem_per_sm: int = 0
    regs_per_sm: int = 0
    threads_per_sm: int = 0
    l2_bytes: int = 0
    torch_version: str = ""
    triton_version: str = ""
    probe_ok: bool = False
    extras: dict = field(default_factory=dict)

    @staticmethod
    def probe() -> "BenchEnv":
        """Probe the device. Never guess a hardware constant.

        The previous version fell back to C500 literals (`warpSize` 64, `max_shared_mem`
        65536, `max_num_regs` 131072) when Triton's property query raised. That query
        JIT-builds and dlopens a C extension on first use, so it is exactly the call that
        fails on a fresh box or a stale build cache -- and when it did, every autotuning
        grid was silently built at C500 shape while the result file recorded the *real*
        device name. A wrong-but-plausible table is worse than a crash.

        So: torch is the source of truth (it exposes every field we need and needs no JIT),
        Triton is only a cross-check, and `probe_ok` records whether the two agreed.
        """
        import triton

        p = torch.cuda.get_device_properties(0)
        try:
            props = triton.runtime.driver.active.utils.get_device_properties(0)
        except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
            props = {"_probe_error": repr(exc)}

        # torch names differ across versions; smem_per_block_optin is the ceiling a kernel
        # can actually reach (49152 default vs 101376 opt-in on sm89; C500 has no opt-in
        # path and reports 65536 for both).
        smem = getattr(p, "shared_memory_per_block_optin", 0) or getattr(
            p, "shared_memory_per_block", 0
        )
        env = BenchEnv(
            device_name=p.name,
            warp_size=p.warp_size,
            num_sm=p.multi_processor_count,
            smem_bytes=smem,
            smem_per_sm=getattr(p, "shared_memory_per_multiprocessor", smem),
            regs_per_sm=p.regs_per_multiprocessor,
            threads_per_sm=p.max_threads_per_multi_processor,
            l2_bytes=p.L2_cache_size,
            torch_version=torch.__version__,
            triton_version=triton.__version__,
            extras={k: v for k, v in props.items()},
        )
        # Cross-check against Triton where it answered; disagreement means one of them is
        # describing a different device than the kernels will actually run on.
        agree = all(
            props.get(k, v) == v
            for k, v in (
                ("warpSize", env.warp_size),
                ("multiprocessor_count", env.num_sm),
                ("max_shared_mem", env.smem_bytes),
                ("max_num_regs", env.regs_per_sm),
            )
        )
        env.probe_ok = bool(env.warp_size and env.num_sm and env.smem_bytes
                            and env.regs_per_sm and agree)
        return env

    def require_ok(self) -> "BenchEnv":
        """Fail loudly before any grid is built. Call once at bench startup."""
        if not self.probe_ok:
            raise RuntimeError(
                f"device probe degraded or inconsistent -- refusing to autotune.\n"
                f"  {self.device_name}: warp={self.warp_size} sm={self.num_sm} "
                f"smem={self.smem_bytes} regs={self.regs_per_sm}\n"
                f"  triton props: {self.extras}\n"
                f"Building a search grid from wrong hardware constants produces a "
                f"plausible table that is silently wrong; fix the probe instead."
            )
        return self

    def banner(self) -> str:
        return (
            f"[env] {self.device_name} | {self.num_sm} SM | warp {self.warp_size} | "
            f"smem {self.smem_bytes} B | regs/SM {self.regs_per_sm} | "
            f"threads/SM {self.threads_per_sm} | L2 {self.l2_bytes >> 20} MB | "
            f"torch {self.torch_version} triton {self.triton_version}"
        )


_ENV_CACHE: "BenchEnv | None" = None


def env() -> "BenchEnv":
    """Probe the device once and reuse it.

    Every hardware constant a bench needs -- shared-memory ceiling, warp width, SM count,
    register file -- must come from here rather than a literal. The C500 study hardcoded
    `65536` and `* 64` throughout; on any other device those silently prune the autotuning
    grid, and they do not prune both arms of a fused/unfused pair equally, which corrupts
    the ratio that is this study's only output.
    """
    global _ENV_CACHE
    if _ENV_CACHE is None:
        # require_ok() here rather than at each call site: every bench reaches its hardware
        # constants through env(), so one choke point protects all of them, including the
        # two (f01, f11b_sweep) that print no environment banner an operator could check.
        _ENV_CACHE = BenchEnv.probe().require_ok()
    return _ENV_CACHE


def smem_stage_bytes(bm: int, bn: int, bk: int, num_stages: int, bn_mult: int = 1) -> int:
    """Shared memory a Triton GEMM mainloop stages, in bytes.

    Triton 3.0 (the C500 stack) allocated `num_stages` buffers; **Triton 3.6 allocates
    `num_stages - 1`**, with a floor of 2. Verified on sm89 by launching 68 configs and
    reading `CompiledKernel.metadata.shared`: this formula is exact on 64 of them and
    conservative on the other 4 (all `num_stages=2`).

    Using the old formula over-predicts by 1.33-1.5x, which rejects configs the hardware
    can actually run -- e.g. BM128/BN256/BK64/s3 is modelled at 144 KB but really uses
    96 KB and launches fine on a 99 KB ceiling.
    """
    return max(2, num_stages - 1) * 2 * bk * (bm + bn_mult * bn)
