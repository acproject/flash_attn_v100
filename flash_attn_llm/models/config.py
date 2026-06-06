"""Model configurations for flash_attn_llm inference library."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Base model configuration.

    Attributes:
        vocab_size: Vocabulary size of the tokenizer.
        hidden_size: Dimensionality of the hidden representations.
        intermediate_size: Dimensionality of the MLP intermediate layer.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads for Q.
        num_key_value_heads: Number of attention heads for K/V (GQA/MQA support).
        head_dim: Dimensionality of each attention head.
        max_position_embeddings: Maximum sequence length the model supports.
        rms_norm_eps: Epsilon for RMSNorm numerical stability.
        rope_theta: Base frequency for rotary position embeddings.
        rope_scaling: Optional dict for RoPE scaling configuration.
        hidden_act: Activation function name (silu, gelu, geglu).
        tie_word_embeddings: Whether input and output embeddings are tied.
        model_type: Model architecture identifier.
        dtype: Default floating-point dtype string.
    """

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 128
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    model_type: str = "llama"
    dtype: str = "float16"
    # Qwen3-specific: per-head QK normalization after Q/K projection
    attention_qk_norm: bool = False


@dataclass
class LlamaConfig(ModelConfig):
    """Configuration for Llama-style models."""

    model_type: str = "llama"


@dataclass
class Qwen2Config(ModelConfig):
    """Configuration for Qwen2-style models."""

    model_type: str = "qwen2"
    vocab_size: int = 151936
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    head_dim: int = 128
    max_position_embeddings: int = 131072
    rope_theta: float = 1000000.0


@dataclass
class MistralConfig(ModelConfig):
    """Configuration for Mistral-style models."""

    model_type: str = "mistral"
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 32768


@dataclass
class Qwen3Config(ModelConfig):
    """Configuration for Qwen3-style models (with QK Norm)."""

    model_type: str = "qwen3"
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 40960
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True
    # Qwen3-specific: per-head QK normalization
    attention_qk_norm: bool = True


# Config registry mapping model_type to config class
MODEL_CONFIG_REGISTRY = {
    "llama": LlamaConfig,
    "qwen2": Qwen2Config,
    "qwen3": Qwen3Config,
    "mistral": MistralConfig,
}


def get_config_from_hf_json(config_dict: dict) -> ModelConfig:
    """Create a ModelConfig from a HuggingFace config.json dictionary.

    Args:
        config_dict: Dictionary loaded from a HuggingFace config.json file.

    Returns:
        An instance of the appropriate ModelConfig subclass.
    """
    model_type = config_dict.get("model_type", "llama")
    config_cls = MODEL_CONFIG_REGISTRY.get(model_type, ModelConfig)

    # Map HuggingFace field names to our field names
    field_map = {
        "hidden_size": "hidden_size",
        "intermediate_size": "intermediate_size",
        "num_hidden_layers": "num_hidden_layers",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "max_position_embeddings": "max_position_embeddings",
        "rms_norm_eps": "rms_norm_eps",
        "rope_theta": "rope_theta",
        "rope_scaling": "rope_scaling",
        "vocab_size": "vocab_size",
        "tie_word_embeddings": "tie_word_embeddings",
        "hidden_act": "hidden_act",
        "attention_qk_norm": "attention_qk_norm",
    }

    kwargs = {}
    for hf_key, our_key in field_map.items():
        if hf_key in config_dict:
            kwargs[our_key] = config_dict[hf_key]

    # Compute head_dim if not provided in the config
    if "head_dim" not in kwargs and "hidden_size" in kwargs and "num_attention_heads" in kwargs:
        kwargs["head_dim"] = kwargs["hidden_size"] // kwargs["num_attention_heads"]

    return config_cls(**kwargs)
