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

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # Determine if this is a global/full attention layer (Gemma4-style)
        layer_types = getattr(config, 'layer_types', None)
        is_global = False
        if layer_types is not None and layer_idx < len(layer_types):
            is_global = layer_types[layer_idx] in ("global_attention", "full_attention")

        # Global attention layers use different head_dim and num_kv_heads
        if is_global:
            global_head_dim = getattr(config, 'global_head_dim', None)
            num_global_kv_heads = getattr(config, 'num_global_key_value_heads', None)
            self.head_dim = global_head_dim if global_head_dim is not None else config.head_dim
            self.num_kv_heads = num_global_kv_heads if num_global_kv_heads is not None else config.num_key_value_heads
            self.num_heads = config.num_attention_heads
        else:
            self.head_dim = config.head_dim
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = config.num_key_value_heads

        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Gemma4 global attention: attention_k_eq_v means no v_proj, v = k
        self.k_eq_v = is_global and getattr(config, 'attention_k_eq_v', False)

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        if self.k_eq_v:
            # No v_proj when k_eq_v; v values come from k_proj
            self.v_proj = None
        else:
            self.v_proj = nn.Linear(
                self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
            )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # RoPE parameters differ between sliding and global attention layers
        if is_global:
            rope_theta = getattr(config, 'global_rope_theta', config.rope_theta)
            partial_rotary_factor = getattr(config, 'global_partial_rotary_factor',
                                            getattr(config, 'partial_rotary_factor', 1.0))
        else:
            rope_theta = config.rope_theta
            partial_rotary_factor = getattr(config, 'partial_rotary_factor', 1.0)

        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            theta=rope_theta,
            rope_scaling=config.rope_scaling,
            partial_rotary_factor=partial_rotary_factor,
        )

        # Qwen3/Gemma4-style per-head QK normalization
        self.qk_norm = getattr(config, 'attention_qk_norm', False)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            # Gemma4-style: V normalization (scale-free RMSNorm, no learned weight)
            self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

        # Check if flash_attn_v100 can handle this head_dim
        self.use_flash_attn = (flash_attn_v100 is not None) and (self.head_dim <= 256)

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

    def _native_attention_prefill(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """PyTorch native attention fallback for head_dim > 256.

        Handles GQA by expanding K/V to match the number of query heads.

        Args:
            q: Query tensor [B, H_Q, seq_len, D].
            k: Key tensor [B, H_KV, seq_len, D].
            v: Value tensor [B, H_KV, seq_len, D].

        Returns:
            Attention output [B, H_Q, seq_len, D].
        """
        # Expand KV for GQA: repeat K/V heads to match Q heads
        if self.num_kv_groups > 1:
            B, H_KV, seq_len, D = k.shape
            k = k[:, :, None, :, :].expand(B, H_KV, self.num_kv_groups, seq_len, D)
            k = k.reshape(B, self.num_heads, seq_len, D)
            v = v[:, :, None, :, :].expand(B, H_KV, self.num_kv_groups, seq_len, D)
            v = v.reshape(B, self.num_heads, seq_len, D)

        # Use fp32 for attention computation to avoid overflow with large head_dim
        # (head_dim=512 can cause softmax overflow in fp16)
        q_fp32 = q.float()
        k_fp32 = k.float()
        v_fp32 = v.float()
        # Gemma4 with QK norm uses scale=1.0 (norm already absorbs the temperature)
        attn_scale = 1.0 if self.qk_norm else None
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q_fp32, k_fp32, v_fp32, is_causal=True, scale=attn_scale
        )
        return attn_output.to(q.dtype)

    def _native_attention_decode(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """PyTorch native attention fallback for decode (head_dim > 256).

        Args:
            q: Query tensor [B, H_Q, 1, D].
            k: Key tensor [B, H_KV, cache_len+1, D].
            v: Value tensor [B, H_KV, cache_len+1, D].

        Returns:
            Attention output [B, H_Q, 1, D].
        """
        # Expand KV for GQA
        if self.num_kv_groups > 1:
            B, H_KV, S, D = k.shape
            k = k[:, :, None, :, :].expand(B, H_KV, self.num_kv_groups, S, D)
            k = k.reshape(B, self.num_heads, S, D)
            v = v[:, :, None, :, :].expand(B, H_KV, self.num_kv_groups, S, D)
            v = v.reshape(B, self.num_heads, S, D)

        # Decode: no causal mask needed (q has length 1)
        # Use fp32 for attention computation to avoid overflow with large head_dim
        q_fp32 = q.float()
        k_fp32 = k.float()
        v_fp32 = v.float()
        # Gemma4 with QK norm uses scale=1.0 (norm already absorbs the temperature)
        attn_scale = 1.0 if self.qk_norm else None
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q_fp32, k_fp32, v_fp32, is_causal=False, scale=attn_scale
        )
        return attn_output.to(q.dtype)

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
        v = self.v_proj(hidden_states) if self.v_proj is not None else k.clone()

        # Debug: check projections
        _has_nan = q.isnan().any().item() or k.isnan().any().item() or v.isnan().any().item()
        if _has_nan:
            print(f"  [DEBUG] NaN after projection! q_nan={q.isnan().any().item()}, k_nan={k.isnan().any().item()}, v_nan={v.isnan().any().item()}")
            print(f"    q: shape={q.shape}, sum={q.sum().item():.4f}")
            print(f"    k: shape={k.shape}, sum={k.sum().item():.4f}")
            print(f"    v: shape={v.shape}, sum={v.sum().item():.4f}")
            print(f"    hidden_states: has_nan={hidden_states.isnan().any().item()}, sum={hidden_states.sum().item():.4f}")
            print(f"    q_proj.weight: sum={self.q_proj.weight.sum().item():.4f}")
            print(f"    k_proj.weight: sum={self.k_proj.weight.sum().item():.4f}")

        # Reshape to [B, H, seq_len, D]
        q = self._reshape_for_flash_attn(q, self.num_heads)
        k = self._reshape_for_flash_attn(k, self.num_kv_heads)
        v = self._reshape_for_flash_attn(v, self.num_kv_heads)

        # Apply QK normalization (Qwen3/Gemma4-style)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
            v = self.v_norm(v)

        # Debug: check after QK norm
        _has_nan = q.isnan().any().item() or k.isnan().any().item()
        if _has_nan:
            print(f"  [DEBUG] NaN after QK norm! q_nan={q.isnan().any().item()}, k_nan={k.isnan().any().item()}")

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(seq_len, position_ids)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Debug: check after RoPE
        _has_nan = q.isnan().any().item() or k.isnan().any().item()
        if _has_nan:
            print(f"  [DEBUG] NaN after RoPE! q_nan={q.isnan().any().item()}, k_nan={k.isnan().any().item()}")
            print(f"    cos: has_nan={cos.isnan().any().item()}, sin: has_nan={sin.isnan().any().item()}")
            print(f"    rotary_dim={self.rotary_emb.rotary_dim}, head_dim={self.head_dim}")

        # Ensure contiguous for CUDA kernels
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Compute attention
        if self.use_flash_attn:
            # Gemma4 with QK norm: Flash Attention uses 1/sqrt(d_k) internally,
            # but we need scale=1.0. Pre-scale Q by sqrt(d_k) to compensate.
            if self.qk_norm:
                q = q * (self.head_dim ** 0.5)
            if self.num_kv_heads < self.num_heads:
                # GQA/MQA: use GQA kernel
                attn_output = flash_attn_v100.forward_prefill_gqa_fp16(q, k, v, True)
            else:
                # MHA: use standard kernel
                attn_output = flash_attn_v100.forward_fp16(q, k, v, True)
        else:
            # Fallback: PyTorch native attention for head_dim > 256
            attn_output = self._native_attention_prefill(q, k, v)

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
        v_new = self.v_proj(hidden_states) if self.v_proj is not None else k_new.clone()

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

        # Compute attention
        if self.use_flash_attn:
            # Gemma4 with QK norm: pre-scale Q by sqrt(d_k) to compensate
            # Flash Attention's internal 1/sqrt(d_k) scaling
            if self.qk_norm:
                q = q * (self.head_dim ** 0.5)
            if self.num_kv_heads < self.num_heads:
                attn_output = flash_attn_v100.forward_decode_gqa_fp16(
                    q, k_updated, v_updated, True, cache_len
                )
            else:
                attn_output = flash_attn_v100.forward_decode_fp16(
                    q, k_updated, v_updated, True, cache_len
                )
        else:
            # Fallback: PyTorch native attention for head_dim > 256
            attn_output = self._native_attention_decode(q, k_updated, v_updated)

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
