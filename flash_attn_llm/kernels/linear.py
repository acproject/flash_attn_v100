"""Linear layer wrappers optimized for LLM inference.

Provides standard linear layers, tensor-parallel variants, fused activation
layers, and weight-only quantized linear layers.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """Standard linear layer with optional bias, optimized for inference.

    Wraps :class:`torch.nn.Linear` but stores weight in the transposed layout
    (``[out_features, in_features]``) that is conventional for LLM weight
    matrices and exposes a few convenience helpers for weight-only
    quantization.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If ``True``, adds a learnable bias to the output.
        dtype: Data type for the weight tensor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weight with Kaiming uniform and bias with uniform."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ``y = x @ weight^T + bias``."""
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


class ColumnParallelLinear(nn.Module):
    """Column-parallel linear layer for tensor parallelism.

    The weight matrix is sharded along the output dimension (columns).  Each
    rank holds a vertical slice of the full weight so that the local matmul
    produces a partial output.  **No all-reduce** is needed after the forward
    pass — each rank independently owns its slice of the output.

    Args:
        in_features: Size of each input sample (global).
        out_features: Size of each output sample (global, before sharding).
        bias: If ``True``, adds a learnable bias (also sharded).
        tp_rank: Current tensor-parallel rank.
        tp_size: Total number of tensor-parallel ranks.
        dtype: Data type for the weight tensor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_rank: int = 0,
        tp_size: int = 1,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        assert out_features % tp_size == 0, (
            f"out_features ({out_features}) must be divisible by tp_size ({tp_size})"
        )
        self.local_out_features = out_features // tp_size

        self.weight = nn.Parameter(
            torch.empty(self.local_out_features, in_features, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.local_out_features, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: ``y_local = x @ weight^T + bias`` (sharded output)."""
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"local_out_features={self.local_out_features}, "
            f"tp_rank={self.tp_rank}, tp_size={self.tp_size}, "
            f"bias={self.bias is not None}"
        )


class RowParallelLinear(nn.Module):
    """Row-parallel linear layer for tensor parallelism.

    The weight matrix is sharded along the input dimension (rows).  Each rank
    holds a horizontal slice of the full weight.  After the local matmul, an
    **all-reduce** is required to sum the partial results across ranks.

    Args:
        in_features: Size of each input sample (global, before sharding).
        out_features: Size of each output sample (global).
        bias: If ``True``, adds a learnable bias (not sharded — only rank 0
            adds it to avoid double-counting).
        tp_rank: Current tensor-parallel rank.
        tp_size: Total number of tensor-parallel ranks.
        dtype: Data type for the weight tensor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_rank: int = 0,
        tp_size: int = 1,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        assert in_features % tp_size == 0, (
            f"in_features ({in_features}) must be divisible by tp_size ({tp_size})"
        )
        self.local_in_features = in_features // tp_size

        self.weight = nn.Parameter(
            torch.empty(out_features, self.local_in_features, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self, x: torch.Tensor, all_reduce: bool = True
    ) -> torch.Tensor:
        """Forward: ``y = all_reduce(x_local @ weight^T) + bias``.

        Args:
            x: Input tensor whose last dimension equals ``local_in_features``.
            all_reduce: If ``True`` and ``tp_size > 1``, perform an all-reduce
                across the TP group.  Set to ``False`` when the caller handles
                the reduction (e.g. fused with a previous operation).
        """
        output = F.linear(x, self.weight)

        if all_reduce and self.tp_size > 1:
            torch.distributed.all_reduce(output, group=self._get_tp_group())

        if self.bias is not None:
            output = output + self.bias
        return output

    def _get_tp_group(self) -> Optional[torch.distributed.ProcessGroup]:
        """Return the TP process group (lazy, cached)."""
        if not hasattr(self, "_tp_group"):
            if torch.distributed.is_initialized():
                self._tp_group = torch.distributed.new_group(
                    ranks=list(range(self.tp_size))
                )
            else:
                self._tp_group = None
        return self._tp_group

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"local_in_features={self.local_in_features}, "
            f"tp_rank={self.tp_rank}, tp_size={self.tp_size}, "
            f"bias={self.bias is not None}"
        )


class FusedSwiGLU(nn.Module):
    """Fused SwiGLU activation layer.

    Computes ``down_proj(silu(gate_proj(x)) * up_proj(x))`` in a single
    module.  The current implementation uses standard PyTorch operations but
    is structured so that a future CUDA kernel can replace the forward
    without changing the external interface.

    Args:
        hidden_size: Input (and output) hidden dimension.
        intermediate_size: Dimension of the gate/up projections.
        bias: If ``True``, add bias to each projection.
        dtype: Data type for the weight tensors.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Parameter(
            torch.empty(intermediate_size, hidden_size, dtype=dtype)
        )
        self.up_proj = nn.Parameter(
            torch.empty(intermediate_size, hidden_size, dtype=dtype)
        )
        self.down_proj = nn.Parameter(
            torch.empty(hidden_size, intermediate_size, dtype=dtype)
        )

        if bias:
            self.gate_bias = nn.Parameter(
                torch.empty(intermediate_size, dtype=dtype)
            )
            self.up_bias = nn.Parameter(
                torch.empty(intermediate_size, dtype=dtype)
            )
            self.down_bias = nn.Parameter(
                torch.empty(hidden_size, dtype=dtype)
            )
        else:
            self.register_parameter("gate_bias", None)
            self.register_parameter("up_bias", None)
            self.register_parameter("down_bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for w in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        for b in (self.gate_bias, self.up_bias, self.down_bias):
            if b is not None:
                nn.init.zeros_(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: ``down_proj(silu(gate_proj(x)) * up_proj(x))``."""
        gate = F.linear(x, self.gate_proj, self.gate_bias)
        up = F.linear(x, self.up_proj, self.up_bias)
        # silu(x) = x * sigmoid(x)
        gate = F.silu(gate)
        hidden = gate * up
        output = F.linear(hidden, self.down_proj, self.down_bias)
        return output

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"bias={self.gate_bias is not None}"
        )


class QuantizedLinear(nn.Module):
    """Weight-only quantized linear layer.

    Supports INT8 weight-only quantization with group-wise scaling.  The
    quantized weights and per-group scales are stored as buffers; during the
    forward pass the weights are dequantized on-the-fly and a standard
    matmul is performed.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If ``True``, adds a learnable bias to the output.
        num_bits: Number of bits for quantization (8 for INT8).
        group_size: Number of weights per quantization group.
        dtype: Data type for the dequantized weight during forward.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        num_bits: int = 8,
        group_size: int = 128,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        self.group_size = group_size
        self.dtype = dtype

        assert num_bits == 8, "Only INT8 weight-only quantization is supported"
        assert in_features % group_size == 0, (
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )

        # Quantized weight stored as int8
        self.register_buffer(
            "quant_weight",
            torch.zeros(out_features, in_features, dtype=torch.int8),
        )
        # Per-group scales: [out_features, in_features / group_size]
        self.register_buffer(
            "scale",
            torch.zeros(
                out_features,
                in_features // group_size,
                dtype=dtype,
            ),
        )
        # Per-group zero points (stored as float for simplicity)
        self.register_buffer(
            "zero_point",
            torch.zeros(
                out_features,
                in_features // group_size,
                dtype=dtype,
            ),
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def quantize(
        self,
        weight: torch.Tensor,
        num_bits: int = 8,
        group_size: int = 128,
    ) -> None:
        """Quantize a full-precision weight tensor in-place.

        Uses symmetric per-group quantization:

        1. Reshape weight into ``[out_features, num_groups, group_size]``.
        2. Compute per-group max absolute value → scale.
        3. Quantize: ``q = round(w / scale)`` clamped to ``[-2^(b-1), 2^(b-1)-1]``.
        4. Store ``quant_weight``, ``scale``, and ``zero_point``.

        Args:
            weight: Full-precision weight tensor of shape
                ``[out_features, in_features]``.
            num_bits: Bits per weight element (default 8).
            group_size: Number of elements per quantization group (default 128).
        """
        assert num_bits == 8, "Only INT8 quantization is supported"
        assert group_size == self.group_size, (
            f"group_size mismatch: got {group_size}, expected {self.group_size}"
        )
        out_features, in_features = weight.shape
        assert out_features == self.out_features
        assert in_features == self.in_features

        weight = weight.to(self.dtype)
        num_groups = in_features // group_size

        # [out_features, num_groups, group_size]
        w_grouped = weight.reshape(out_features, num_groups, group_size)

        # Symmetric quantization: scale = max(|w|) / qmax
        qmax = 2 ** (num_bits - 1) - 1  # 127 for INT8
        w_max = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        scale = w_max / qmax  # [out_features, num_groups, 1]

        # Quantize
        q = torch.round(w_grouped / scale).clamp(-qmax - 1, qmax).to(torch.int8)

        # Store
        self.quant_weight.copy_(q.reshape(out_features, in_features))
        self.scale.copy_(scale.squeeze(-1))
        self.zero_point.zero_()

    def _dequantize(self) -> torch.Tensor:
        """Dequantize the stored quantized weight to full precision.

        Returns:
            Dequantized weight of shape ``[out_features, in_features]``.
        """
        num_groups = self.in_features // self.group_size
        # [out_features, num_groups, group_size]
        q = self.quant_weight.reshape(
            self.out_features, num_groups, self.group_size
        ).to(self.dtype)
        s = self.scale.unsqueeze(-1)  # [out_features, num_groups, 1]
        zp = self.zero_point.unsqueeze(-1)  # [out_features, num_groups, 1]
        w = (q - zp) * s
        return w.reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: dequantize weight then ``y = x @ weight^T + bias``."""
        weight = self._dequantize()
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, num_bits={self.num_bits}, "
            f"group_size={self.group_size}"
        )


# ---------------------------------------------------------------------------
# Helper factory functions
# ---------------------------------------------------------------------------


def create_linear(
    in_features: int,
    out_features: int,
    bias: bool = False,
    dtype: torch.dtype = torch.float16,
) -> Linear:
    """Create a :class:`Linear` layer."""
    return Linear(in_features, out_features, bias=bias, dtype=dtype)


def create_column_parallel(
    in_features: int,
    out_features: int,
    bias: bool = False,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> ColumnParallelLinear:
    """Create a :class:`ColumnParallelLinear` layer."""
    return ColumnParallelLinear(
        in_features, out_features, bias=bias, tp_rank=tp_rank, tp_size=tp_size
    )


def create_row_parallel(
    in_features: int,
    out_features: int,
    bias: bool = False,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> RowParallelLinear:
    """Create a :class:`RowParallelLinear` layer."""
    return RowParallelLinear(
        in_features, out_features, bias=bias, tp_rank=tp_rank, tp_size=tp_size
    )
