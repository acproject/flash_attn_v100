"""Rotary Position Embedding (RoPE) for flash_attn_llm models.

Supports standard RoPE, NTK-aware scaling (dynamic), linear scaling, and YaRN scaling.
Input format: [B, H, seq_len, D] matching flash_attn_v100 kernel layout.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _compute_default_rope_parameters(
    head_dim: int,
    max_position_embeddings: int,
    theta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute default RoPE frequencies (cos, sin).

    Args:
        head_dim: Dimension of each attention head.
        max_position_embeddings: Maximum sequence length.
        theta: Base frequency for RoPE.

    Returns:
        Tuple of (cos, sin) tensors each of shape [max_position_embeddings, head_dim].
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_position_embeddings, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [max_pos, head_dim // 2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_pos, head_dim]
    return emb.cos(), emb.sin()


def _compute_linear_scaling_rope_parameters(
    head_dim: int,
    max_position_embeddings: int,
    theta: float,
    scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute linearly scaled RoPE frequencies.

    Scales positions by a factor, effectively extending the context window
    by interpolating between positions.

    Args:
        head_dim: Dimension of each attention head.
        max_position_embeddings: Maximum sequence length.
        theta: Base frequency for RoPE.
        scaling_factor: Factor by which to scale positions.

    Returns:
        Tuple of (cos, sin) tensors each of shape [max_position_embeddings, head_dim].
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    scaled_positions = torch.arange(max_position_embeddings, dtype=torch.float32) / scaling_factor
    freqs = torch.outer(scaled_positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _compute_dynamic_scaling_rope_parameters(
    head_dim: int,
    max_position_embeddings: int,
    theta: float,
    scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute NTK-aware dynamically scaled RoPE frequencies.

    Instead of scaling positions, this method scales the base theta to
    preserve high-frequency information while extending the context window.

    Args:
        head_dim: Dimension of each attention head.
        max_position_embeddings: Maximum sequence length.
        theta: Base frequency for RoPE.
        scaling_factor: Factor by which to extend the context window.

    Returns:
        Tuple of (cos, sin) tensors each of shape [max_position_embeddings, head_dim].
    """
    base = theta * (scaling_factor ** (head_dim / (head_dim - 2)))
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_position_embeddings, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _compute_yarn_rope_parameters(
    head_dim: int,
    max_position_embeddings: int,
    theta: float,
    scaling_factor: float,
    original_max_position_embeddings: int = 4096,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    mscale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute YaRN (Yet another RoPE extensioN) scaled RoPE frequencies.

    YaRN combines NTK-aware scaling with attention temperature modulation
    to achieve better extrapolation to longer sequences.

    Args:
        head_dim: Dimension of each attention head.
        max_position_embeddings: Maximum sequence length.
        theta: Base frequency for RoPE.
        scaling_factor: Factor by which to extend the context window.
        original_max_position_embeddings: Original max position the model was trained on.
        beta_fast: Fast dimension cutoff parameter.
        beta_slow: Slow dimension cutoff parameter.
        mscale: Multiplicative scale for attention logits.

    Returns:
        Tuple of (cos, sin) tensors each of shape [max_position_embeddings, head_dim].
    """
    pos_freqs = theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    inv_freq_extrapolation = 1.0 / (scaling_factor * pos_freqs)
    inv_freq_interpolation = 1.0 / pos_freqs

    low_freq_wavelen = original_max_position_embeddings / beta_slow
    high_freq_wavelen = original_max_position_embeddings / beta_fast

    inv_freq = torch.zeros_like(inv_freq_interpolation)
    for i in range(len(inv_freq_interpolation)):
        wavelen = 2 * math.pi / inv_freq_interpolation[i]
        if wavelen < high_freq_wavelen:
            # Keep high frequencies unchanged
            inv_freq[i] = inv_freq_interpolation[i]
        elif wavelen > low_freq_wavelen:
            # Linearly scale low frequencies
            inv_freq[i] = inv_freq_extrapolation[i]
        else:
            # Smooth blend between interpolation and extrapolation
            smooth = (original_max_position_embeddings / wavelen - beta_slow) / (beta_fast - beta_slow)
            inv_freq[i] = (1 - smooth) * inv_freq_extrapolation[i] + smooth * inv_freq_interpolation[i]

    positions = torch.arange(max_position_embeddings, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)

    # Apply mscale to cos/sin
    attention_scale = mscale
    return emb.cos() * attention_scale, emb.sin() * attention_scale


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding module with multiple scaling strategies.

    Precomputes cos and sin frequency tables for efficient application during
    forward pass. Supports standard RoPE, linear scaling, NTK-aware dynamic
    scaling, and YaRN scaling.

    Args:
        head_dim: Dimension of each attention head.
        max_position_embeddings: Maximum sequence length.
        theta: Base frequency for RoPE.
        rope_scaling: Optional dict with scaling configuration.
            Supported types: "linear", "dynamic", "yarn".
        partial_rotary_factor: Fraction of head_dim to apply RoPE to.
            E.g. 0.25 means only the first 1/4 of dimensions get RoPE.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 4096,
        theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        partial_rotary_factor: float = 1.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = int(head_dim * partial_rotary_factor)

        # Precompute inverse frequencies only (much smaller than full cos/sin tables)
        # For large max_position_embeddings, full cos/sin tables would be too large
        # (e.g. 262144 * 256 * 4 bytes * 2 = 537 MB per layer)
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_rope_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE parameters based on the scaling strategy."""
        if self.rope_scaling is None:
            return _compute_default_rope_parameters(
                self.rotary_dim, self.max_position_embeddings, self.theta
            )

        scaling_type = self.rope_scaling.get("type", "default")
        scaling_factor = self.rope_scaling.get("factor", 1.0)

        if scaling_type == "linear":
            return _compute_linear_scaling_rope_parameters(
                self.rotary_dim, self.max_position_embeddings, self.theta, scaling_factor
            )
        elif scaling_type == "dynamic":
            return _compute_dynamic_scaling_rope_parameters(
                self.rotary_dim, self.max_position_embeddings, self.theta, scaling_factor
            )
        elif scaling_type == "yarn":
            original_max_pos = self.rope_scaling.get(
                "original_max_position_embeddings", 4096
            )
            beta_fast = self.rope_scaling.get("beta_fast", 32.0)
            beta_slow = self.rope_scaling.get("beta_slow", 1.0)
            mscale = self.rope_scaling.get("mscale", 1.0)
            return _compute_yarn_rope_parameters(
                self.rotary_dim,
                self.max_position_embeddings,
                self.theta,
                scaling_factor,
                original_max_pos,
                beta_fast,
                beta_slow,
                mscale,
            )
        else:
            return _compute_default_rope_parameters(
                self.rotary_dim, self.max_position_embeddings, self.theta
            )

    def forward(
        self, seq_len: int, position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin values for the given positions.

        Dynamically computes cos/sin from inv_freq to avoid storing huge
        precomputed tables (which would be ~537 MB per layer for large
        max_position_embeddings).

        Args:
            seq_len: Length of the sequence.
            position_ids: Optional position indices [B, seq_len]. If None,
                uses sequential positions 0..seq_len-1.

        Returns:
            Tuple of (cos, sin) tensors. If position_ids is None, shapes are
            [1, 1, seq_len, rotary_dim]. Otherwise [B, 1, seq_len, rotary_dim].
        """
        if position_ids is not None:
            # position_ids: [B, seq_len]
            positions = position_ids.float()  # [B, seq_len]
        else:
            positions = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)  # [seq_len]

        # Compute freqs: positions x inv_freq -> [B, seq_len, rotary_dim//2] or [seq_len, rotary_dim//2]
        freqs = torch.outer(positions.flatten(), self.inv_freq) if position_ids is None else torch.einsum("bs,d->bsd", positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [..., rotary_dim]
        cos = emb.cos()
        sin = emb.sin()

        if position_ids is not None:
            B = position_ids.shape[0]
            cos = cos.view(B, position_ids.shape[1], self.rotary_dim)
            sin = sin.view(B, position_ids.shape[1], self.rotary_dim)
            return cos.unsqueeze(1), sin.unsqueeze(1)  # [B, 1, seq_len, rotary_dim]
        else:
            cos = cos.view(seq_len, self.rotary_dim)
            sin = sin.view(seq_len, self.rotary_dim)
            return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, rotary_dim]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input.

    Splits the last dimension into two halves and rotates them:
    [-x2, x1] from [x1, x2].

    Args:
        x: Input tensor of shape [..., D].

    Returns:
        Rotated tensor of the same shape.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Rotate with interleaved layout (used by some models like GPT-NeoX).

    Rearranges pairs of elements: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]

    Args:
        x: Input tensor of shape [..., D].

    Returns:
        Rotated tensor of the same shape.
    """
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    Supports partial rotary embeddings where only a fraction of the head_dim
    dimensions receive RoPE (the rest pass through unchanged).

    Args:
        q: Query tensor of shape [B, H_Q, seq_len, D].
        k: Key tensor of shape [B, H_KV, seq_len, D].
        cos: Cosine values of shape [B_or_1, 1, seq_len, rotary_dim].
        sin: Sine values of shape [B_or_1, 1, seq_len, rotary_dim].
        interleaved: If True, use interleaved rotary layout (GPT-NeoX style).
            If False, use half-rotation layout (LLaMA style).

    Returns:
        Tuple of (q_rotated, k_rotated) with the same shapes as inputs.
    """
    rotary_dim = cos.shape[-1]

    # Cast cos/sin to match q's dtype to avoid type promotion
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)

    # Full rotary: apply to all dimensions
    if rotary_dim == q.shape[-1]:
        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        return q_embed, k_embed

    # Partial rotary: split into rotary and non-rotary parts
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    q_rot_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)
    return q_embed, k_embed
