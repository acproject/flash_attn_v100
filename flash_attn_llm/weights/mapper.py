"""Weight name mapping between HuggingFace and our model architecture.

This module provides the WeightMapper class and factory function to translate
HuggingFace weight names to our internal model parameter names, handling
naming differences across multiple architectures (Llama, Qwen2, Mistral, Yi, Gemma).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ShardSpec:
    """Specification for how a weight should be sharded for tensor parallelism.

    Attributes:
        shard_dim: The dimension along which to split the tensor.
        is_column: Whether this is a column-parallel split (True) or row-parallel (False).
    """

    shard_dim: int
    is_column: bool


@dataclass
class MappingRule:
    """A single mapping rule from a HF name pattern to our internal name.

    Attributes:
        pattern: Compiled regex pattern to match HF weight names.
        replacement: Replacement string using regex group references.
        shard_spec: Optional sharding specification for tensor parallelism.
    """

    pattern: re.Pattern
    replacement: str
    shard_spec: Optional[ShardSpec] = None


class WeightMapper:
    """Maps HuggingFace weight names to our model's parameter names.

    Each mapper instance is configured with a set of mapping rules that define
    how to translate HF weight names. The mapper also tracks which weights
    should be treated as tied (shared) parameters.

    Attributes:
        model_type: The architecture identifier (e.g. 'llama', 'qwen2').
        rules: Ordered list of MappingRule instances.
        tied_weights: Dict mapping source weight name -> target weight name for tied params.
    """

    def __init__(
        self,
        model_type: str,
        rules: List[MappingRule],
        tied_weights: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model_type = model_type
        self.rules = rules
        self.tied_weights = tied_weights or {}

    def map_name(self, hf_name: str) -> Optional[str]:
        """Map a single HuggingFace weight name to our internal name.

        Args:
            hf_name: The HuggingFace weight name (e.g. 'model.layers.0.self_attn.q_proj.weight').

        Returns:
            The mapped internal name, or None if no rule matches.
        """
        for rule in self.rules:
            match = rule.pattern.fullmatch(hf_name)
            if match:
                return rule.replacement.format(*match.groups())
        return None

    def map_all(self, hf_names: List[str]) -> Dict[str, str]:
        """Map a list of HuggingFace weight names to internal names.

        Args:
            hf_names: List of HuggingFace weight names.

        Returns:
            Dict mapping HF names to internal names. Unmapped names are skipped.
        """
        result: Dict[str, str] = {}
        for name in hf_names:
            mapped = self.map_name(name)
            if mapped is not None:
                result[name] = mapped
        return result

    def get_shard_spec(self, internal_name: str) -> Optional[ShardSpec]:
        """Get the sharding specification for an internal weight name.

        Args:
            internal_name: Our internal parameter name.

        Returns:
            ShardSpec if the weight should be sharded, None otherwise.
        """
        for rule in self.rules:
            if rule.shard_spec is not None:
                # Convert the replacement template to a regex that matches
                # internal names produced by this rule.
                # e.g. "layers.{}.attention.q_proj.weight" -> r"layers\.\d+\.attention\.q_proj\.weight"
                template_regex = re.escape(rule.replacement)
                template_regex = template_regex.replace(re.escape("{}"), r"\d+")
                if re.fullmatch(template_regex, internal_name):
                    return rule.shard_spec
        return None

    def resolve_tied(self, internal_name: str) -> str:
        """Resolve tied weight references.

        If the given internal name is a tied weight target, return the source
        name. Otherwise return the name unchanged.

        Args:
            internal_name: Our internal parameter name.

        Returns:
            The resolved internal name (source of the tie if applicable).
        """
        return self.tied_weights.get(internal_name, internal_name)


# ---------------------------------------------------------------------------
# Architecture-specific mapper builders
# ---------------------------------------------------------------------------

def _build_llama_mapper() -> WeightMapper:
    """Build the WeightMapper for Llama-style models.

    Covers: Llama, Qwen2, Mistral, Yi (they share the same weight naming).
    """
    rules: List[MappingRule] = [
        # Embedding
        MappingRule(
            pattern=re.compile(r'model\.embed_tokens\.weight'),
            replacement='embed_tokens.weight',
            shard_spec=ShardSpec(shard_dim=1, is_column=False),
        ),
        # Attention projections
        MappingRule(
            pattern=re.compile(r'model\.layers\.(\d+)\.self_attn\.q_proj\.weight'),
            replacement='layers.{}.self_attn.q_proj.weight',
            shard_spec=ShardSpec(shard_dim=0, is_column=True),
        ),
        MappingRule(
            pattern=re.compile(r'model\.layers\.(\d+)\.self_attn\.k_proj\.weight'),
            replacement='layers.{}.self_attn.k_proj.weight',
            shard_spec=ShardSpec(shard_dim=0, is_column=True),
        ),
        MappingRule(
            pattern=re.compile(r'model\.layers\.(\d+)\.self_attn\.v_proj\.weight'),
            replacement='layers.{}.self_attn.v_proj.weight',
            shard_spec=ShardSpec(shard_dim=0, is_column=True),
        ),
        MappingRule(
            pattern=re.compile(r'model\.layers\.(\d+)\.self_attn\.o_proj\.weight'),
            replacement='layers.{}.self_attn.o_proj.weight',
            shard_spec=ShardSpec(shard_dim=1, is_column=False),
        ),
        # MLP projections
        MappingRule(
            pattern=re.compile(r'model\.layers\.(\d+)\.mlp\.gate_proj\.weight'),
            replacement='layers.{}.mlp.gate_proj.weight',
            shard_spec=ShardSpec(shard_dim=0, is_column=True),
        ),
        MappingRule(
            pattern=re.compile(r'model\.layers\.(\d+)\.mlp\.up_proj\.weight'),
            replacement='layers.{}.mlp.up_proj.weight',
            shard_spec=ShardSpec(shard_dim=0, is_column=True),
        ),
        MappingRule(
            pattern=re.compile(r'model\.layers\.(\d+)\.mlp\.down_proj\.weight'),
            replacement='layers.{}.mlp.down_proj.weight',
            shard_spec=ShardSpec(shard_dim=1, is_column=False),
        ),
        # Layer norms
        MappingRule(
            pattern=re.compile(r'model\.layers\.(\d+)\.input_layernorm\.weight'),
            replacement='layers.{}.input_layernorm.weight',
        ),
        MappingRule(
            pattern=re.compile(r'model\.layers\.(\d+)\.post_attention_layernorm\.weight'),
            replacement='layers.{}.post_attention_layernorm.weight',
        ),
        # Final norm
        MappingRule(
            pattern=re.compile(r'model\.norm\.weight'),
            replacement='norm.weight',
        ),
        # LM head
        MappingRule(
            pattern=re.compile(r'lm_head\.weight'),
            replacement='lm_head.weight',
            shard_spec=ShardSpec(shard_dim=0, is_column=True),
        ),
    ]

    # Tied embeddings: lm_head shares weights with embed_tokens
    tied_weights: Dict[str, str] = {
        'lm_head.weight': 'embed_tokens.weight',
    }

    return WeightMapper(
        model_type='llama',
        rules=rules,
        tied_weights=tied_weights,
    )


def _build_qwen2_mapper() -> WeightMapper:
    """Build the WeightMapper for Qwen2 models.

    Qwen2 uses the same naming as Llama, but may have additional biases.
    """
    mapper = _build_llama_mapper()
    mapper.model_type = 'qwen2'

    # Add Qwen2-specific rules (attention biases)
    mapper.rules.extend([
        MappingRule(
            pattern=re.compile(r'model\.layers\.(\d+)\.self_attn\.q_proj\.bias'),
            replacement='layers.{}.self_attn.q_proj.bias',
            shard_spec=ShardSpec(shard_dim=0, is_column=True),
        ),
        MappingRule(
            pattern=re.compile(r'model\.layers\.(\d+)\.self_attn\.k_proj\.bias'),
            replacement='layers.{}.self_attn.k_proj.bias',
            shard_spec=ShardSpec(shard_dim=0, is_column=True),
        ),
        MappingRule(
            pattern=re.compile(r'model\.layers\.(\d+)\.self_attn\.v_proj\.bias'),
            replacement='layers.{}.self_attn.v_proj.bias',
            shard_spec=ShardSpec(shard_dim=0, is_column=True),
        ),
    ])

    return mapper


def _build_qwen3_mapper() -> WeightMapper:
    """Build the WeightMapper for Qwen3 models.

    Qwen3 extends Qwen2 with per-head QK normalization (q_norm, k_norm).
    """
    mapper = _build_llama_mapper()
    mapper.model_type = 'qwen3'

    # Add Qwen3-specific rules: QK normalization
    mapper.rules.extend([
        MappingRule(
            pattern=re.compile(r'model\.layers\.(\d+)\.self_attn\.q_norm\.weight'),
            replacement='layers.{}.self_attn.q_norm.weight',
        ),
        MappingRule(
            pattern=re.compile(r'model\.layers\.(\d+)\.self_attn\.k_norm\.weight'),
            replacement='layers.{}.self_attn.k_norm.weight',
        ),
    ])

    return mapper


def _build_mistral_mapper() -> WeightMapper:
    """Build the WeightMapper for Mistral models.

    Mistral uses the same naming convention as Llama.
    """
    mapper = _build_llama_mapper()
    mapper.model_type = 'mistral'
    return mapper


def _build_yi_mapper() -> WeightMapper:
    """Build the WeightMapper for Yi models.

    Yi uses the same naming convention as Llama.
    """
    mapper = _build_llama_mapper()
    mapper.model_type = 'yi'
    return mapper


def _build_gemma4_mapper() -> WeightMapper:
    """Build the WeightMapper for Gemma4 models.

    Gemma4 weights are prefixed with 'model.language_model.' instead of 'model.'.
    Has QK normalization (q_norm, k_norm), pre/post feedforward layernorms,
    and per-layer scalar (layer_scalar).
    """
    rules: List[MappingRule] = [
        # Embedding
        MappingRule(
            pattern=re.compile(r'model\.language_model\.embed_tokens\.weight'),
            replacement='embed_tokens.weight',
            shard_spec=ShardSpec(shard_dim=1, is_column=False),
        ),
        # Attention projections
        MappingRule(
            pattern=re.compile(r'model\.language_model\.layers\.(\d+)\.self_attn\.q_proj\.weight'),
            replacement='layers.{}.self_attn.q_proj.weight',
            shard_spec=ShardSpec(shard_dim=0, is_column=True),
        ),
        MappingRule(
            pattern=re.compile(r'model\.language_model\.layers\.(\d+)\.self_attn\.k_proj\.weight'),
            replacement='layers.{}.self_attn.k_proj.weight',
            shard_spec=ShardSpec(shard_dim=0, is_column=True),
        ),
        MappingRule(
            pattern=re.compile(r'model\.language_model\.layers\.(\d+)\.self_attn\.v_proj\.weight'),
            replacement='layers.{}.self_attn.v_proj.weight',
            shard_spec=ShardSpec(shard_dim=0, is_column=True),
        ),
        MappingRule(
            pattern=re.compile(r'model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.weight'),
            replacement='layers.{}.self_attn.o_proj.weight',
            shard_spec=ShardSpec(shard_dim=1, is_column=False),
        ),
        # QK normalization
        MappingRule(
            pattern=re.compile(r'model\.language_model\.layers\.(\d+)\.self_attn\.q_norm\.weight'),
            replacement='layers.{}.self_attn.q_norm.weight',
        ),
        MappingRule(
            pattern=re.compile(r'model\.language_model\.layers\.(\d+)\.self_attn\.k_norm\.weight'),
            replacement='layers.{}.self_attn.k_norm.weight',
        ),
        # MLP projections
        MappingRule(
            pattern=re.compile(r'model\.language_model\.layers\.(\d+)\.mlp\.gate_proj\.weight'),
            replacement='layers.{}.mlp.gate_proj.weight',
            shard_spec=ShardSpec(shard_dim=0, is_column=True),
        ),
        MappingRule(
            pattern=re.compile(r'model\.language_model\.layers\.(\d+)\.mlp\.up_proj\.weight'),
            replacement='layers.{}.mlp.up_proj.weight',
            shard_spec=ShardSpec(shard_dim=0, is_column=True),
        ),
        MappingRule(
            pattern=re.compile(r'model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.weight'),
            replacement='layers.{}.mlp.down_proj.weight',
            shard_spec=ShardSpec(shard_dim=1, is_column=False),
        ),
        # Layer norms
        MappingRule(
            pattern=re.compile(r'model\.language_model\.layers\.(\d+)\.input_layernorm\.weight'),
            replacement='layers.{}.input_layernorm.weight',
        ),
        MappingRule(
            pattern=re.compile(r'model\.language_model\.layers\.(\d+)\.post_attention_layernorm\.weight'),
            replacement='layers.{}.post_attention_layernorm.weight',
        ),
        MappingRule(
            pattern=re.compile(r'model\.language_model\.layers\.(\d+)\.pre_feedforward_layernorm\.weight'),
            replacement='layers.{}.pre_feedforward_layernorm.weight',
        ),
        MappingRule(
            pattern=re.compile(r'model\.language_model\.layers\.(\d+)\.post_feedforward_layernorm\.weight'),
            replacement='layers.{}.post_feedforward_layernorm.weight',
        ),
        # Per-layer scalar
        MappingRule(
            pattern=re.compile(r'model\.language_model\.layers\.(\d+)\.layer_scalar'),
            replacement='layers.{}.layer_scalar',
        ),
        # Final norm
        MappingRule(
            pattern=re.compile(r'model\.language_model\.norm\.weight'),
            replacement='norm.weight',
        ),
    ]

    # Tied embeddings: lm_head shares weights with embed_tokens
    # (no lm_head.weight in Gemma4 safetensors since tie_word_embeddings=true)
    tied_weights: Dict[str, str] = {
        'lm_head.weight': 'embed_tokens.weight',
    }

    return WeightMapper(
        model_type='gemma4',
        rules=rules,
        tied_weights=tied_weights,
    )


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

_MAPPER_REGISTRY: Dict[str, type] = {}
_MAPPER_BUILDERS: Dict[str, object] = {
    'llama': _build_llama_mapper,
    'qwen2': _build_qwen2_mapper,
    'qwen3': _build_qwen3_mapper,
    'mistral': _build_mistral_mapper,
    'yi': _build_yi_mapper,
    'gemma': _build_gemma4_mapper,
    'gemma4': _build_gemma4_mapper,
    'gemma4_text': _build_gemma4_mapper,
}

# Common aliases for model architectures
_MODEL_TYPE_ALIASES: Dict[str, str] = {
    'llama': 'llama',
    'qwen2': 'qwen2',
    'qwen': 'qwen2',
    'qwen3': 'qwen3',
    'mistral': 'mistral',
    'mixtral': 'mistral',
    'yi': 'yi',
    'gemma': 'gemma4',
    'gemma2': 'gemma4',
    'gemma4': 'gemma4',
    'gemma4_text': 'gemma4',
}


def get_mapper(model_type: str) -> WeightMapper:
    """Return the appropriate WeightMapper for the given model architecture.

    Args:
        model_type: The model architecture identifier. Supported values:
            'llama', 'qwen2', 'qwen', 'mistral', 'mixtral', 'yi', 'gemma', 'gemma2'.

    Returns:
        A WeightMapper instance configured for the specified architecture.

    Raises:
        ValueError: If the model_type is not supported.
    """
    canonical = _MODEL_TYPE_ALIASES.get(model_type.lower())
    if canonical is None:
        supported = sorted(set(_MODEL_TYPE_ALIASES.keys()))
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Supported types: {supported}"
        )

    builder = _MAPPER_BUILDERS[canonical]
    return builder()  # type: ignore[operator]


def list_supported_models() -> List[str]:
    """Return a sorted list of all supported model type identifiers (including aliases)."""
    return sorted(_MODEL_TYPE_ALIASES.keys())
