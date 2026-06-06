"""Weight loading and name mapping for LLM inference.

This package provides utilities for loading model weights from HuggingFace
format (safetensors and PyTorch .bin) and mapping HF weight names to
our internal model parameter names.

Quick start::

    from flash_attn_llm.weights import WeightLoader, get_mapper

    # Load weights into a model
    loader = WeightLoader(tp_rank=0, tp_size=1)
    stats = loader.load_weights(model, "/path/to/hf/model", dtype="fp16")

    # Or use the mapper directly
    mapper = get_mapper("llama")
    internal_name = mapper.map_name("model.layers.0.self_attn.q_proj.weight")
    # -> "layers.0.attention.q_proj.weight"
"""

from .loader import (
    WeightLoader,
    get_weight_files,
    load_all_weights,
    load_pytorch_bin,
    load_safetensors,
    shard_weight,
)
from .mapper import (
    MappingRule,
    ShardSpec,
    WeightMapper,
    get_mapper,
    list_supported_models,
)

__all__ = [
    # Loader
    "WeightLoader",
    "load_safetensors",
    "load_pytorch_bin",
    "get_weight_files",
    "load_all_weights",
    "shard_weight",
    # Mapper
    "WeightMapper",
    "MappingRule",
    "ShardSpec",
    "get_mapper",
    "list_supported_models",
]
