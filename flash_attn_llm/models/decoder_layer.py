"""Transformer Decoder Layer for flash_attn_llm models.

Implements the standard pre-norm transformer decoder layer with residual connections:
    x = x + attention(norm(x))
    x = x + mlp(norm(x))
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig
from .attention import LlamaAttention
from .mlp import build_mlp
from .norm import RMSNorm


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with pre-norm residual connections.

    Contains self-attention, MLP, and normalization layers.
    Supports both the standard 2-norm (Llama-style) and 3-norm (Gemma4-style)
    architectures, as well as per-layer learnable scalar (layer_scalar).

    Standard (2-norm): input_layernorm + post_attention_layernorm
    Gemma4 (3-norm): input_layernorm + post_attention_layernorm + pre_feedforward_layernorm

    Args:
        config: ModelConfig instance with layer hyperparameters.
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx=layer_idx)
        self.mlp = build_mlp(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Gemma4-style: pre/post feedforward layernorms (4-norm architecture)
        self.has_pre_feedforward_norm = hasattr(config, 'layer_types') and config.layer_types is not None
        if self.has_pre_feedforward_norm:
            self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Gemma4-style: per-layer learnable scalar (buffer, not parameter)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Prefill forward pass for processing a full prompt sequence.

        Args:
            hidden_states: Input tensor [B, seq_len, hidden_size].
            position_ids: Optional position indices [B, seq_len].

        Returns:
            Tuple of:
                - Output tensor [B, seq_len, hidden_size]
                - KV cache tuple (k, v) from attention for this layer
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, kv_cache = self.self_attn.forward_prefill(hidden_states, position_ids)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Feedforward with residual
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states) if self.has_pre_feedforward_norm else self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.has_pre_feedforward_norm:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Gemma4-style: per-layer scalar
        hidden_states = hidden_states * self.layer_scalar

        return hidden_states, kv_cache

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        cache_len: int,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode forward pass for single-token autoregressive generation.

        Args:
            hidden_states: Input tensor [B, 1, hidden_size].
            kv_cache: Tuple of (k_cache, v_cache) for this layer.
            cache_len: Current length of the KV cache.
            position_ids: Optional position indices [B, 1].

        Returns:
            Tuple of:
                - Output tensor [B, 1, hidden_size]
                - Updated KV cache tuple (k_updated, v_updated)
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, updated_kv_cache = self.self_attn.forward_decode(
            hidden_states, kv_cache, cache_len, position_ids
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Feedforward with residual
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states) if self.has_pre_feedforward_norm else self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.has_pre_feedforward_norm:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Gemma4-style: per-layer scalar
        hidden_states = hidden_states * self.layer_scalar

        return hidden_states, updated_kv_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_len: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Unified forward pass dispatching to prefill or decode.

        Args:
            hidden_states: Input tensor [B, seq_len, hidden_size].
            kv_cache: Optional KV cache for decode mode.
            cache_len: Length of existing KV cache (decode only).
            position_ids: Optional position indices.

        Returns:
            Tuple of:
                - Output tensor [B, seq_len, hidden_size]
                - Updated KV cache (None for prefill, tuple for decode)
        """
        if kv_cache is None:
            return self.forward_prefill(hidden_states, position_ids), None
        else:
            return self.forward_decode(hidden_states, kv_cache, cache_len, position_ids)
