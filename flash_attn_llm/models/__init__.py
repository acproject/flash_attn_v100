"""flash_attn_llm models package.

Exports all model classes, configurations, and utilities.
"""

from .config import (
    ModelConfig,
    LlamaConfig,
    Qwen2Config,
    Qwen3Config,
    MistralConfig,
    MODEL_CONFIG_REGISTRY,
    get_config_from_hf_json,
)
from .norm import RMSNorm, LayerNorm
from .rope import RotaryEmbedding, apply_rotary_emb
from .attention import LlamaAttention
from .mlp import LlamaMLP, GeGLUMLP, GenericMLP, FusedSwiGLU, build_mlp
from .decoder_layer import TransformerDecoderLayer
from .causal_lm import CausalLM, KVCacheManager

__all__ = [
    # Config
    "ModelConfig",
    "LlamaConfig",
    "Qwen2Config",
    "Qwen3Config",
    "MistralConfig",
    "MODEL_CONFIG_REGISTRY",
    "get_config_from_hf_json",
    # Norm
    "RMSNorm",
    "LayerNorm",
    # RoPE
    "RotaryEmbedding",
    "apply_rotary_emb",
    # Attention
    "LlamaAttention",
    # MLP
    "LlamaMLP",
    "GeGLUMLP",
    "GenericMLP",
    "FusedSwiGLU",
    "build_mlp",
    # Decoder
    "TransformerDecoderLayer",
    # Model
    "CausalLM",
    "KVCacheManager",
]
