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
    warp_size: int = 64
    num_sm: int = 104
    smem_bytes: int = 65536
    regs_per_sm: int = 131072
    l2_bytes: int = 8 * 2**20
    torch_version: str = ""
    triton_version: str = ""
    extras: dict = field(default_factory=dict)

    @staticmethod
    def probe() -> "BenchEnv":
        import triton

        p = torch.cuda.get_device_properties(0)
        try:
            props = triton.runtime.driver.active.utils.get_device_properties(0)
        except Exception:
            props = {}
        return BenchEnv(
            device_name=p.name,
            warp_size=props.get("warpSize", 64),
            num_sm=props.get("multiprocessor_count", p.multi_processor_count),
            smem_bytes=props.get("max_shared_mem", 65536),
            regs_per_sm=props.get("max_num_regs", 131072),
            l2_bytes=p.L2_cache_size,
            torch_version=torch.__version__,
            triton_version=triton.__version__,
            extras={k: v for k, v in props.items()},
        )
