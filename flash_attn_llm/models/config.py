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


@dataclass
class Gemma4Config(ModelConfig):
    """Configuration for Gemma4-style models.

    Gemma4 extends the base architecture with:
    - Per-head QK normalization
    - Pre/post feedforward layer norms (3-norm per layer)
    - Per-layer learnable scalar (layer_scalar)
    - Partial rotary embeddings (only a fraction of head_dim)
    - Final logit softcapping
    - Sliding/full attention layer alternation
    - GeLU tanh activation (geglu)
    """

    model_type: str = "gemma4"
    vocab_size: int = 262144
    hidden_size: int = 5376
    intermediate_size: int = 21504
    num_hidden_layers: int = 60
    num_attention_heads: int = 32
    num_key_value_heads: int = 16
    head_dim: int = 256
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    hidden_act: str = "geglu"
    tie_word_embeddings: bool = True
    attention_qk_norm: bool = True
    # Gemma4-specific
    partial_rotary_factor: float = 1.0
    final_logit_softcapping: Optional[float] = None
    sliding_window: Optional[int] = None
    layer_types: Optional[List[str]] = None
    global_head_dim: Optional[int] = None
    num_global_key_value_heads: Optional[int] = None
    attention_k_eq_v: bool = False
    global_rope_theta: Optional[float] = None
    global_partial_rotary_factor: Optional[float] = None


# Config registry mapping model_type to config class
MODEL_CONFIG_REGISTRY = {
    "llama": LlamaConfig,
    "qwen2": Qwen2Config,
    "qwen3": Qwen3Config,
    "mistral": MistralConfig,
    "gemma4": Gemma4Config,
    "gemma4_text": Gemma4Config,
}


def get_config_from_hf_json(config_dict: dict) -> ModelConfig:
    """Create a ModelConfig from a HuggingFace config.json dictionary.

    Args:
        config_dict: Dictionary loaded from a HuggingFace config.json file.

    Returns:
        An instance of the appropriate ModelConfig subclass.
    """
    model_type = config_dict.get("model_type", "llama")

    # For multimodal models (e.g. gemma4), extract the text config
    if model_type in ("gemma4",) and "text_config" in config_dict:
        text_config = config_dict["text_config"]
        # Merge top-level fields into text_config for unified processing
        text_config.setdefault("model_type", model_type + "_text")
        text_config.setdefault("tie_word_embeddings", config_dict.get("tie_word_embeddings", True))
        config_dict = text_config
        model_type = config_dict.get("model_type", model_type)

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
        "head_dim": "head_dim",
        # Gemma4-specific fields
        "partial_rotary_factor": "partial_rotary_factor",
        "final_logit_softcapping": "final_logit_softcapping",
        "sliding_window": "sliding_window",
        "layer_types": "layer_types",
        "global_head_dim": "global_head_dim",
        "num_global_key_value_heads": "num_global_key_value_heads",
        "attention_k_eq_v": "attention_k_eq_v",
    }

    kwargs = {}
    for hf_key, our_key in field_map.items():
        if hf_key in config_dict:
            kwargs[our_key] = config_dict[hf_key]

    # Gemma4: extract rope_theta and partial_rotary_factor from rope_parameters
    if "rope_parameters" in config_dict and isinstance(config_dict["rope_parameters"], dict):
        rope_params = config_dict["rope_parameters"]
        # Extract sliding_attention RoPE params (used as default rope_theta/partial_rotary_factor)
        for attn_type in ("sliding_attention", "default"):
            if attn_type in rope_params:
                section = rope_params[attn_type]
                if "rope_theta" in section and "rope_theta" not in kwargs:
                    kwargs["rope_theta"] = section["rope_theta"]
                if "partial_rotary_factor" in section and "partial_rotary_factor" not in kwargs:
                    kwargs["partial_rotary_factor"] = section["partial_rotary_factor"]
                break
        # Extract full_attention RoPE params (global layers use different params)
        if "full_attention" in rope_params:
            section = rope_params["full_attention"]
            if "rope_theta" in section:
                kwargs["global_rope_theta"] = section["rope_theta"]
            if "partial_rotary_factor" in section:
                kwargs["global_partial_rotary_factor"] = section["partial_rotary_factor"]

    # Gemma4: map hidden_activation to our hidden_act
    if "hidden_activation" in config_dict and "hidden_act" not in kwargs:
        act = config_dict["hidden_activation"]
        if act in ("gelu_pytorch_tanh", "gelu_tanh"):
            kwargs["hidden_act"] = "geglu"
        elif act in ("silu",):
            kwargs["hidden_act"] = "silu"
        else:
            kwargs["hidden_act"] = act

    # Gemma4: final_logit_softcapping from top-level config
    if "final_logit_softcapping" not in kwargs:
        # Check if it was in the original top-level config
        pass  # Already handled via field_map

    # Compute head_dim if not provided in the config
    if "head_dim" not in kwargs and "hidden_size" in kwargs and "num_attention_heads" in kwargs:
        kwargs["head_dim"] = kwargs["hidden_size"] // kwargs["num_attention_heads"]

    # Filter kwargs to only include fields that the config class accepts
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(config_cls)}
    kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

    return config_cls(**kwargs)
