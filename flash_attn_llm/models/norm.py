"""Normalization layers for flash_attn_llm models."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Computes: x * rsqrt(mean(x^2) + eps) * weight

    Supports FP16 and BF16 dtypes by performing the variance computation
    in FP32 for numerical stability, then casting back.

    Args:
        hidden_size: The number of features in the input tensor.
        eps: A small float for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LayerNorm(nn.Module):
    """Standard Layer Normalization.

    Computes: (x - mean) / sqrt(var + eps) * weight + bias

    Performs computation in FP32 for numerical stability with FP16/BF16 inputs.

    Args:
        hidden_size: The number of features in the input tensor.
        eps: A small float for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states + self.bias).to(input_dtype)
