"""Attention module using flash_attn_v100 CUDA kernels.

Supports Multi-Head Attention (MHA), Grouped-Query Attention (GQA),
and Multi-Query Attention (MQA) with both prefill and decode modes.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    import flash_attn_v100
except ImportError:
    flash_attn_v100 = None

from .config import ModelConfig
from .norm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary_emb


class LlamaAttention(nn.Module):
    """Multi-head / Grouped-Query Attention with flash_attn_v100 backend.

    Supports GQA where num_key_value_heads < num_attention_heads.
    Uses flash_attn_v100 CUDA kernels for efficient attention computation:
    - Prefill: forward_prefill_gqa_fp16 (GQA) or forward_fp16 (MHA)
    - Decode: forward_decode_gqa_fp16

    Args:
        config: ModelConfig instance with attention hyperparameters.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        # Qwen3-style per-head QK normalization
        self.qk_norm = getattr(config, 'attention_qk_norm', False)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _reshape_for_flash_attn(
        self, tensor: torch.Tensor, num_heads: int
    ) -> torch.Tensor:
        """Reshape projection output to flash_attn_v100 format [B, H, seq_len, D].

        Args:
            tensor: Output of linear projection [B, seq_len, num_heads * head_dim].
            num_heads: Number of attention heads.

        Returns:
            Reshaped tensor [B, num_heads, seq_len, head_dim].
        """
        B, seq_len, _ = tensor.shape
        tensor = tensor.view(B, seq_len, num_heads, self.head_dim)
        return tensor.transpose(1, 2)  # [B, num_heads, seq_len, head_dim]

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
                - Attention output [B, seq_len, hidden_size]
                - KV cache tuple (k, v) for reuse in decode
        """
        B, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [B, H, seq_len, D]
        q = self._reshape_for_flash_attn(q, self.num_heads)
        k = self._reshape_for_flash_attn(k, self.num_kv_heads)
        v = self._reshape_for_flash_attn(v, self.num_kv_heads)

        # Apply QK normalization (Qwen3-style)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(seq_len, position_ids)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Ensure contiguous for CUDA kernels
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Compute attention using flash_attn_v100
        if flash_attn_v100 is None:
            raise RuntimeError(
                "flash_attn_v100 CUDA extension is not available. "
                "Please build and install it first."
            )
        if self.num_kv_heads < self.num_heads:
            # GQA/MQA: use GQA kernel
            attn_output = flash_attn_v100.forward_prefill_gqa_fp16(q, k, v, True)
        else:
            # MHA: use standard kernel
            attn_output = flash_attn_v100.forward_fp16(q, k, v, True)

        # Reshape back: [B, H, seq_len, D] -> [B, seq_len, H * D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, seq_len, self.num_heads * self.head_dim)

        return self.o_proj(attn_output), (k, v)

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
            kv_cache: Tuple of (k_cache, v_cache) tensors.
                k_cache: [B, H_KV, cache_len, D]
                v_cache: [B, H_KV, cache_len, D]
            cache_len: Current length of the KV cache (before adding new token).
            position_ids: Optional position indices [B, 1].

        Returns:
            Tuple of:
                - Attention output [B, 1, hidden_size]
                - Updated KV cache tuple (k_updated, v_updated)
        """
        B, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k_new = self.k_proj(hidden_states)
        v_new = self.v_proj(hidden_states)

        # Reshape to [B, H, 1, D]
        q = self._reshape_for_flash_attn(q, self.num_heads)
        k_new = self._reshape_for_flash_attn(k_new, self.num_kv_heads)
        v_new = self._reshape_for_flash_attn(v_new, self.num_kv_heads)

        # Apply QK normalization (Qwen3-style)
        if self.qk_norm:
            q = self.q_norm(q)
            k_new = self.k_norm(k_new)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(seq_len, position_ids)
        q, k_new = apply_rotary_emb(q, k_new, cos, sin)

        # Update KV cache: append new K/V to existing cache
        k_cache, v_cache = kv_cache
        k_updated = torch.cat([k_cache, k_new], dim=2)  # [B, H_KV, cache_len+1, D]
        v_updated = torch.cat([v_cache, v_new], dim=2)  # [B, H_KV, cache_len+1, D]

        # Ensure contiguous
        q = q.contiguous()
        k_updated = k_updated.contiguous()
        v_updated = v_updated.contiguous()

        # Compute attention using flash_attn_v100 decode kernel
        if flash_attn_v100 is None:
            raise RuntimeError(
                "flash_attn_v100 CUDA extension is not available. "
                "Please build and install it first."
            )
        if self.num_kv_heads < self.num_heads:
            attn_output = flash_attn_v100.forward_decode_gqa_fp16(
                q, k_updated, v_updated, True, cache_len
            )
        else:
            attn_output = flash_attn_v100.forward_decode_fp16(
                q, k_updated, v_updated, True, cache_len
            )

        # Reshape: [B, H, 1, D] -> [B, 1, H * D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, seq_len, self.num_heads * self.head_dim)

        return self.o_proj(attn_output), (k_updated, v_updated)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_len: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Unified forward pass that dispatches to prefill or decode.

        If kv_cache is None, runs prefill mode. Otherwise runs decode mode.

        Args:
            hidden_states: Input tensor [B, seq_len, hidden_size].
            kv_cache: Optional KV cache tuple for decode mode.
            cache_len: Length of existing KV cache (decode mode only).
            position_ids: Optional position indices.

        Returns:
            Tuple of:
                - Attention output [B, seq_len, hidden_size]
                - Updated KV cache (None for prefill, tuple for decode)
        """
        if kv_cache is None:
            return self.forward_prefill(hidden_states, position_ids), None
        else:
            return self.forward_decode(hidden_states, kv_cache, cache_len, position_ids)
