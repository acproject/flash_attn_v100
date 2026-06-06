"""FlashAttn LLM – high-performance LLM inference library.

Built on top of flash_attn_v100 CUDA kernels for V100-optimized attention.

Sub-packages:
- models: Transformer model structures (CausalLM, Attention, MLP, etc.)
- weights: Weight loading from HuggingFace safetensors/bin formats
- tokenizer: Tokenizer wrapper for text encoding/decoding
- kv_cache: KV cache management (contiguous and paged)
- kernels: Linear layers, GEMM, fused activations
- sampling: Token sampling strategies (greedy, top-k, top-p, etc.)
- engine: Inference engine with continuous batching
- quantization: Weight and KV cache quantization
- parallel: Tensor parallelism support
- server: OpenAI-compatible API server
"""

# Models
from flash_attn_llm.models import (
    ModelConfig,
    LlamaConfig,
    Qwen2Config,
    MistralConfig,
    MODEL_CONFIG_REGISTRY,
    get_config_from_hf_json,
    RMSNorm,
    LayerNorm,
    RotaryEmbedding,
    apply_rotary_emb,
    LlamaAttention,
    LlamaMLP,
    GeGLUMLP,
    GenericMLP,
    FusedSwiGLU as FusedSwiGLUMLP,
    build_mlp,
    TransformerDecoderLayer,
    CausalLM,
)

# Weights
from flash_attn_llm.weights import WeightLoader, WeightMapper, get_mapper

# Tokenizer
from flash_attn_llm.tokenizer import Tokenizer, TokenStreamer

# KV Cache
from flash_attn_llm.kv_cache import KVCacheConfig, KVCache, PagedKVCacheManager

# Kernels
from flash_attn_llm.kernels import (
    Linear,
    ColumnParallelLinear,
    RowParallelLinear,
    FusedSwiGLU,
    QuantizedLinear,
)

# Sampling
from flash_attn_llm.sampling import Sampler, SamplingParams

# Engine
from flash_attn_llm.engine import LLMEngine, InferenceRequest, RequestStatus

# Quantization
from flash_attn_llm.quantization import WeightQuantizer, GPTQQuantizer, AWQQuantizer

# Parallel
from flash_attn_llm.parallel import TensorParallelManager

# Server
from flash_attn_llm.server import LLMServer

__all__ = [
    # Models
    "ModelConfig",
    "LlamaConfig",
    "Qwen2Config",
    "MistralConfig",
    "MODEL_CONFIG_REGISTRY",
    "get_config_from_hf_json",
    "RMSNorm",
    "LayerNorm",
    "RotaryEmbedding",
    "apply_rotary_emb",
    "LlamaAttention",
    "LlamaMLP",
    "GeGLUMLP",
    "GenericMLP",
    "FusedSwiGLUMLP",
    "build_mlp",
    "TransformerDecoderLayer",
    "CausalLM",
    # Weights
    "WeightLoader",
    "WeightMapper",
    "get_mapper",
    # Tokenizer
    "Tokenizer",
    "TokenStreamer",
    # KV Cache
    "KVCacheConfig",
    "KVCache",
    "PagedKVCacheManager",
    # Kernels
    "Linear",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "FusedSwiGLU",
    "QuantizedLinear",
    # Sampling
    "Sampler",
    "SamplingParams",
    # Engine
    "LLMEngine",
    "InferenceRequest",
    "RequestStatus",
    # Quantization
    "WeightQuantizer",
    "GPTQQuantizer",
    "AWQQuantizer",
    # Parallel
    "TensorParallelManager",
    # Server
    "LLMServer",
]
