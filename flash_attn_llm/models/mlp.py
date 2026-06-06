"""MLP modules for flash_attn_llm models.

Includes SwiGLU (Llama-style), GeGLU, and generic MLP with configurable activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LlamaMLP(nn.Module):
    """Llama-style MLP with SwiGLU activation.

    Computes: down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        hidden_size: Input and output dimensionality.
        intermediate_size: Dimensionality of the gate and up projections.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GeGLUMLP(nn.Module):
    """MLP with GeGLU (GELU-Gated Linear Unit) activation.

    Computes: down_proj(gelu(gate_proj(x)) * up_proj(x))

    Args:
        hidden_size: Input and output dimensionality.
        intermediate_size: Dimensionality of the gate and up projections.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class GenericMLP(nn.Module):
    """Generic MLP with configurable activation function.

    Supports silu, gelu, and geglu activations. For silu/gelu, uses a
    standard two-layer MLP. For geglu, uses a gated architecture.

    Args:
        hidden_size: Input and output dimensionality.
        intermediate_size: Dimensionality of the intermediate layer.
        hidden_act: Activation function name ("silu", "gelu", or "geglu").
    """

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.hidden_act = hidden_act

        if hidden_act == "geglu":
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        else:
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def _get_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.hidden_act == "silu":
            return F.silu(x)
        elif self.hidden_act == "gelu":
            return F.gelu(x)
        else:
            raise ValueError(f"Unsupported activation: {self.hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hidden_act == "geglu":
            return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))
        else:
            return self.down_proj(self._get_activation(self.up_proj(x)))


class FusedSwiGLU(nn.Module):
    """SwiGLU with a fused gate+up projection for future CUDA kernel integration.

    Computes: down_proj(silu(gate) * up) where gate and up come from a single
    fused weight matrix. This module is designed to be easily replaced by a
    fused CUDA kernel in the future.

    Args:
        hidden_size: Input and output dimensionality.
        intermediate_size: Dimensionality of the gate and up projections.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # Fused weight: [2 * intermediate_size, hidden_size]
        # First half is gate, second half is up
        self.gate_up_proj = nn.Linear(
            hidden_size, 2 * intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate = gate_up[..., : self.intermediate_size]
        up = gate_up[..., self.intermediate_size :]
        return self.down_proj(F.silu(gate) * up)


def build_mlp(
    hidden_size: int,
    intermediate_size: int,
    hidden_act: str = "silu",
) -> nn.Module:
    """Factory function to build the appropriate MLP module.

    Args:
        hidden_size: Input and output dimensionality.
        intermediate_size: Dimensionality of the intermediate layer.
        hidden_act: Activation function name ("silu", "gelu", or "geglu").

    Returns:
        An MLP module instance.
    """
    if hidden_act == "silu":
        return LlamaMLP(hidden_size, intermediate_size)
    elif hidden_act == "geglu":
        return GeGLUMLP(hidden_size, intermediate_size)
    else:
        return GenericMLP(hidden_size, intermediate_size, hidden_act)
