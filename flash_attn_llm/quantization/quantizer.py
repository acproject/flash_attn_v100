"""Weight and KV cache quantization utilities for efficient LLM inference.

Provides:
- WeightQuantizer: INT8 / INT4 weight quantization with per-group scaling,
  and INT8 KV cache quantization.
- GPTQQuantizer: Simplified GPTQ-style post-training INT4 quantization.
- AWQQuantizer: Simplified AWQ-style activation-aware INT4 quantization.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


# ======================================================================
# Weight Quantizer
# ======================================================================

class WeightQuantizer:
    """Quantize model weights for efficient inference.

    All quantization methods use per-group symmetric quantization with
    a scale factor.  INT4 values are packed two-per-byte into an INT8
    storage tensor.
    """

    # ------------------------------------------------------------------
    # INT8 weight quantization
    # ------------------------------------------------------------------

    @staticmethod
    def quantize_int8_weight(
        weight: torch.Tensor,
        group_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize weight to INT8 with per-group scale.

        Args:
            weight: FP16/FP32 weight tensor of shape (out_features, in_features).
            group_size: Number of consecutive elements per quantization group.

        Returns:
            Tuple of (quantized_weight, scale) where:
            - quantized_weight: INT8 tensor of same shape as weight.
            - scale: FP16/FP32 per-group scale of shape
                     (out_features, in_features // group_size).
        """
        original_shape = weight.shape
        out_features, in_features = weight.shape

        # Pad in_features to be divisible by group_size
        pad_len = (group_size - in_features % group_size) % group_size
        if pad_len > 0:
            weight = torch.nn.functional.pad(weight, (0, pad_len), value=0.0)

        padded_in = weight.shape[1]
        weight_grouped = weight.reshape(out_features, padded_in // group_size, group_size)

        # Per-group max absolute value -> scale
        group_max = weight_grouped.abs().amax(dim=-1, keepdim=True)  # (out, groups, 1)
        scale = group_max.squeeze(-1) / 127.0  # (out, groups)
        scale = scale.clamp(min=1e-8)

        # Quantize
        scale_expanded = scale.unsqueeze(-1)  # (out, groups, 1)
        quant_weight = torch.round(weight_grouped / scale_expanded).clamp(-128, 127).to(torch.int8)

        quant_weight = quant_weight.reshape(out_features, padded_in)
        # Remove padding if any
        if pad_len > 0:
            quant_weight = quant_weight[:, :in_features]

        return quant_weight, scale.to(weight.dtype)

    @staticmethod
    def dequantize_int8_weight(
        quant_weight: torch.Tensor,
        scale: torch.Tensor,
        group_size: int = 128,
    ) -> torch.Tensor:
        """Dequantize INT8 weight back to FP16/FP32.

        Args:
            quant_weight: INT8 tensor of shape (out_features, in_features).
            scale: Per-group scale of shape (out_features, in_features // group_size).
            group_size: Number of elements per quantization group.

        Returns:
            Dequantized weight tensor of shape (out_features, in_features).
        """
        out_features, in_features = quant_weight.shape

        # Pad if needed
        pad_len = (group_size - in_features % group_size) % group_size
        if pad_len > 0:
            quant_weight = torch.nn.functional.pad(
                quant_weight, (0, pad_len), value=0
            )

        padded_in = quant_weight.shape[1]
        qw_grouped = quant_weight.reshape(out_features, padded_in // group_size, group_size).float()
        scale_expanded = scale.unsqueeze(-1)  # (out, groups, 1)

        dequant = qw_grouped * scale_expanded
        dequant = dequant.reshape(out_features, padded_in)

        if pad_len > 0:
            dequant = dequant[:, :in_features]

        return dequant.to(scale.dtype)

    # ------------------------------------------------------------------
    # INT4 weight quantization (packed)
    # ------------------------------------------------------------------

    @staticmethod
    def quantize_int4_weight(
        weight: torch.Tensor,
        group_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize weight to INT4 with per-group scale.

        Two INT4 values are packed into one INT8 value (low nibble first).

        Args:
            weight: FP16/FP32 weight tensor of shape (out_features, in_features).
            group_size: Number of consecutive elements per quantization group.

        Returns:
            Tuple of (quant_weight_packed, scale) where:
            - quant_weight_packed: INT8 tensor of shape
              (out_features, in_features // 2).
            - scale: Per-group scale of shape
              (out_features, in_features // group_size).
        """
        out_features, in_features = weight.shape
        assert in_features % 2 == 0, "in_features must be even for INT4 packing"

        # Pad for group_size
        pad_len = (group_size - in_features % group_size) % group_size
        if pad_len > 0:
            weight = torch.nn.functional.pad(weight, (0, pad_len), value=0.0)

        padded_in = weight.shape[1]
        weight_grouped = weight.reshape(out_features, padded_in // group_size, group_size)

        # Scale: max abs / 7 (4-bit signed range is -8..7, use 7 for symmetric)
        group_max = weight_grouped.abs().amax(dim=-1, keepdim=True)
        scale = group_max.squeeze(-1) / 7.0
        scale = scale.clamp(min=1e-8)

        scale_expanded = scale.unsqueeze(-1)
        quant_vals = torch.round(weight_grouped / scale_expanded).clamp(-8, 7).to(torch.int8)

        quant_vals = quant_vals.reshape(out_features, padded_in)

        # Pack two INT4 values into one INT8
        # Even-index values go to low nibble, odd-index to high nibble
        even_vals = quant_vals[:, 0::2]  # (out, in/2)
        odd_vals = quant_vals[:, 1::2]   # (out, in/2)

        # Store as unsigned: add 8 to shift from [-8,7] to [0,15]
        even_unsigned = (even_vals.to(torch.int16) + 8).to(torch.uint8)
        odd_unsigned = (odd_vals.to(torch.int16) + 8).to(torch.uint8)

        packed = (odd_unsigned << 4) | even_unsigned  # (out, in/2)
        packed = packed.view(torch.int8)  # Reinterpret as int8 for storage

        # Remove padding from scale
        num_groups = in_features // group_size
        scale = scale[:, :num_groups]

        return packed, scale.to(weight.dtype)

    @staticmethod
    def dequantize_int4_weight(
        quant_weight: torch.Tensor,
        scale: torch.Tensor,
        group_size: int = 128,
    ) -> torch.Tensor:
        """Dequantize INT4 weight back to FP16/FP32.

        Args:
            quant_weight: Packed INT8 tensor of shape (out_features, in_features // 2).
            scale: Per-group scale of shape (out_features, in_features // group_size).
            group_size: Number of elements per quantization group.

        Returns:
            Dequantized weight tensor of shape (out_features, in_features).
        """
        out_features = quant_weight.shape[0]
        in_features = quant_weight.shape[1] * 2

        # Unpack
        packed = quant_weight.view(torch.uint8)
        even_unsigned = packed & 0x0F           # Low nibble
        odd_unsigned = (packed >> 4) & 0x0F     # High nibble

        # Convert back to signed: subtract 8
        even_vals = even_unsigned.to(torch.float16) - 8.0
        odd_vals = odd_unsigned.to(torch.float16) - 8.0

        # Interleave
        dequant = torch.zeros(out_features, in_features, device=quant_weight.device, dtype=scale.dtype)
        dequant[:, 0::2] = even_vals.to(scale.dtype)
        dequant[:, 1::2] = odd_vals.to(scale.dtype)

        # Apply scale
        pad_len = (group_size - in_features % group_size) % group_size
        if pad_len > 0:
            dequant = torch.nn.functional.pad(dequant, (0, pad_len), value=0.0)

        padded_in = dequant.shape[1]
        dequant_grouped = dequant.reshape(out_features, padded_in // group_size, group_size)
        scale_expanded = scale.unsqueeze(-1)
        dequant_grouped = dequant_grouped * scale_expanded
        dequant = dequant_grouped.reshape(out_features, padded_in)

        if pad_len > 0:
            dequant = dequant[:, :in_features]

        return dequant

    # ------------------------------------------------------------------
    # KV cache INT8 quantization
    # ------------------------------------------------------------------

    @staticmethod
    def quantize_kv_cache_int8(
        kv_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize KV cache to INT8 with per-token scale.

        Args:
            kv_tensor: FP16/FP32 KV cache tensor of shape
                       (batch, num_heads, seq_len, head_dim).

        Returns:
            Tuple of (quant_kv, scale) where:
            - quant_kv: INT8 tensor of same shape.
            - scale: Per-token scale of shape (batch, num_heads, seq_len, 1).
        """
        # Per-token (along head_dim) max absolute value
        token_max = kv_tensor.abs().amax(dim=-1, keepdim=True)  # (B, H, S, 1)
        scale = token_max / 127.0
        scale = scale.clamp(min=1e-8)

        quant_kv = torch.round(kv_tensor / scale).clamp(-128, 127).to(torch.int8)
        return quant_kv, scale.squeeze(-1).to(kv_tensor.dtype)

    @staticmethod
    def dequantize_kv_cache_int8(
        quant_kv: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize INT8 KV cache back to FP16/FP32.

        Args:
            quant_kv: INT8 tensor of shape (batch, num_heads, seq_len, head_dim).
            scale: Per-token scale of shape (batch, num_heads, seq_len).

        Returns:
            Dequantized KV cache tensor.
        """
        scale_expanded = scale.unsqueeze(-1)  # (B, H, S, 1)
        return quant_kv.float() * scale_expanded.to(torch.float32)


# ======================================================================
# GPTQ Quantizer
# ======================================================================

class GPTQQuantizer:
    """Simplified GPTQ-style post-training quantization for INT4 weights.

    GPTQ processes weight rows one at a time, using the Hessian inverse
    to minimize the layer-wise quantization error.  This is a simplified
    implementation suitable for understanding the algorithm.
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        actorder: bool = False,
    ):
        """
        Args:
            bits: Target bit width (4 for INT4).
            group_size: Number of elements per quantization group.
            actorder: Whether to reorder columns by Hessian diagonal magnitude.
        """
        self.bits = bits
        self.group_size = group_size
        self.actorder = actorder

    def quantize(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """Quantize all Linear layers in the model using GPTQ.

        Args:
            model: The model to quantize (modified in place).
            calibration_data: Optional calibration input for Hessian computation.
                              Shape: (num_samples, seq_len) or similar.

        Returns:
            The quantized model (same object, modified in place).
        """
        # Collect named linear layers
        linear_layers = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        ]

        for name, module in linear_layers:
            quantized_weight, scale = self._quantize_linear(
                module, calibration_data
            )
            # Replace the module's weight with quantized data
            # In practice, a custom QuantizedLinear module would be used
            module.weight = nn.Parameter(
                WeightQuantizer.dequantize_int4_weight(
                    quantized_weight, scale, self.group_size
                ),
                requires_grad=False,
            )
            # Store quantization metadata
            module._quantized_weight = quantized_weight
            module._quant_scale = scale
            module._quant_group_size = self.group_size
            module._quant_bits = self.bits

        return model

    def _quantize_linear(
        self,
        linear: nn.Linear,
        calibration_data: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a single Linear layer using GPTQ algorithm.

        Args:
            linear: The Linear layer to quantize.
            calibration_data: Calibration inputs for Hessian computation.

        Returns:
            Tuple of (packed_quantized_weight, scale).
        """
        weight = linear.weight.data.clone().float()
        out_features, in_features = weight.shape

        # Compute Hessian (or use identity if no calibration data)
        if calibration_data is not None:
            hessian = self._compute_hessian(linear, calibration_data)
        else:
            hessian = torch.eye(in_features, device=weight.device, dtype=weight.dtype)
            hessian += 0.01 * torch.eye(in_features, device=weight.device, dtype=weight.dtype)

        # GPTQ: process each row
        quantized_weight = weight.clone()
        errors = torch.zeros_like(weight)

        # Column permutation based on Hessian diagonal
        perm = None
        if self.actorder:
            diag = torch.diag(hessian)
            perm = diag.argsort(descending=True)
            quantized_weight = quantized_weight[:, perm]
            hessian = hessian[perm][:, perm]

        # Cholesky decomposition of Hessian inverse
        try:
            hessian_inv = torch.linalg.cholesky(hessian)
            hessian_inv = torch.cholesky_inverse(hessian_inv)
            hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)
        except RuntimeError:
            hessian_inv = torch.eye(in_features, device=weight.device, dtype=weight.dtype)

        for i in range(in_features):
            # Quantize column i
            col = quantized_weight[:, i]
            quant_col = self._quantize_column(col, i, in_features)
            errors[:, i] = (col - quant_col) / hessian_inv[i, i]
            quantized_weight[:, i] = quant_col

            # Propagate error to remaining columns
            if i < in_features - 1:
                quantized_weight[:, i + 1:] -= (
                    errors[:, i:i + 1] * hessian_inv[i, i + 1:].unsqueeze(0)
                )

        # Undo permutation
        if perm is not None:
            inv_perm = perm.argsort()
            quantized_weight = quantized_weight[:, inv_perm]

        # Pack into INT4
        packed, scale = WeightQuantizer.quantize_int4_weight(
            quantized_weight, group_size=self.group_size
        )
        return packed, scale

    def _quantize_column(
        self,
        col: torch.Tensor,
        col_idx: int,
        in_features: int,
    ) -> torch.Tensor:
        """Quantize a single column of the weight matrix."""
        group_idx = col_idx // self.group_size
        max_val = col.abs().max().clamp(min=1e-8)
        scale = max_val / 7.0
        quant_col = torch.round(col / scale).clamp(-8, 7) * scale
        return quant_col

    @staticmethod
    def _compute_hessian(
        linear: nn.Linear,
        calibration_data: torch.Tensor,
    ) -> torch.Tensor:
        """Compute approximate Hessian from calibration data."""
        in_features = linear.in_features
        hessian = torch.zeros(
            in_features, in_features,
            device=linear.weight.device,
            dtype=torch.float32,
        )

        with torch.no_grad():
            for batch in calibration_data:
                # Simple approach: use input activations
                if batch.dim() == 1:
                    batch = batch.unsqueeze(0)
                inp = batch.float().to(linear.weight.device)
                if inp.shape[-1] != in_features:
                    continue
                hessian += inp.T @ inp

        hessian /= max(calibration_data.shape[0], 1)
        return hessian


# ======================================================================
# AWQ Quantizer
# ======================================================================

class AWQQuantizer:
    """Simplified AWQ-style activation-aware weight quantization.

    AWQ observes that a small fraction (≈1%) of weight channels are
    salient for model quality.  It protects these channels by scaling
    them up before quantization and scaling the corresponding activations
    down, preserving the product.
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        version: str = "gemm",
    ):
        """
        Args:
            bits: Target bit width (4 for INT4).
            group_size: Number of elements per quantization group.
            zero_point: Whether to use zero-point quantization.
            version: Quantization kernel version hint.
        """
        self.bits = bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.version = version

    def quantize(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """Quantize all Linear layers in the model using AWQ.

        Args:
            model: The model to quantize (modified in place).
            calibration_data: Optional calibration inputs for activation
                              magnitude analysis.

        Returns:
            The quantized model (same object, modified in place).
        """
        linear_layers = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        ]

        for name, module in linear_layers:
            scale_factor = self._compute_scale_factor(module, calibration_data)
            quantized_weight, q_scale = self._quantize_with_scale(
                module, scale_factor
            )
            # Replace weight with dequantized version for forward compatibility
            module.weight = nn.Parameter(
                WeightQuantizer.dequantize_int4_weight(
                    quantized_weight, q_scale, self.group_size
                ),
                requires_grad=False,
            )
            # Store quantization metadata
            module._quantized_weight = quantized_weight
            module._quant_scale = q_scale
            module._quant_group_size = self.group_size
            module._quant_bits = self.bits
            module._awq_scale_factor = scale_factor

        return model

    def _compute_scale_factor(
        self,
        linear: nn.Linear,
        calibration_data: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-channel scale factor based on activation magnitudes.

        AWQ identifies salient channels by the product of activation
        magnitude and weight magnitude, then applies a scaling factor
        to protect those channels during quantization.

        Returns:
            Scale factor tensor of shape (1, in_features).
        """
        in_features = linear.in_features
        weight = linear.weight.data.float()

        # Compute weight magnitude per input channel
        weight_mag = weight.abs().mean(dim=0)  # (in_features,)

        # Compute activation magnitude from calibration data
        if calibration_data is not None:
            act_mag = self._compute_activation_magnitude(linear, calibration_data)
        else:
            # Fallback: use weight magnitude alone
            act_mag = torch.ones(in_features, device=weight.device, dtype=weight.dtype)

        # Salience score = activation magnitude * weight magnitude
        salience = act_mag * weight_mag  # (in_features,)

        # Determine scaling: protect the top-alpha% salient channels
        alpha = 0.5  # Mix ratio
        scale_factor = (salience.mean() / (salience + 1e-8)).pow(alpha)
        scale_factor = scale_factor.clamp(0.5, 2.0)  # Bound the scaling

        return scale_factor.unsqueeze(0)  # (1, in_features)

    def _quantize_with_scale(
        self,
        linear: nn.Linear,
        scale_factor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a Linear layer's weight with AWQ scaling applied.

        The weight is scaled up before quantization (protecting salient
        channels), and the corresponding activation will be scaled down
        during inference.

        Returns:
            Tuple of (packed_quantized_weight, scale).
        """
        scaled_weight = linear.weight.data.float() * scale_factor
        packed, scale = WeightQuantizer.quantize_int4_weight(
            scaled_weight, group_size=self.group_size
        )
        return packed, scale

    @staticmethod
    def _compute_activation_magnitude(
        linear: nn.Linear,
        calibration_data: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-channel activation magnitude from calibration data.

        Returns:
            Activation magnitude tensor of shape (in_features,).
        """
        in_features = linear.in_features
        act_sum = torch.zeros(in_features, device=linear.weight.device, dtype=torch.float32)
        count = 0

        with torch.no_grad():
            for batch in calibration_data:
                if batch.dim() == 1:
                    batch = batch.unsqueeze(0)
                inp = batch.float().to(linear.weight.device)
                if inp.shape[-1] != in_features:
                    continue
                act_sum += inp.abs().sum(dim=0)
                count += inp.shape[0]

        if count > 0:
            return act_sum / count
        return torch.ones(in_features, device=linear.weight.device, dtype=torch.float32)
