"""Weight loading from HuggingFace safetensors and PyTorch .bin files.

This module provides the WeightLoader class and utility functions for loading
model weights from disk, converting dtypes, mapping HF names to internal names,
and supporting tensor-parallel sharding.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from .mapper import WeightMapper, get_mapper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP: Dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
}


def _resolve_dtype(dtype: Any) -> torch.dtype:
    """Resolve a dtype specification to a torch.dtype.

    Args:
        dtype: A torch.dtype, a string (e.g. 'fp16', 'bf16'), or None.

    Returns:
        The resolved torch.dtype.

    Raises:
        ValueError: If the dtype string is not recognized.
    """
    if dtype is None:
        return torch.float16
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        key = dtype.lower().strip()
        if key in _DTYPE_MAP:
            return _DTYPE_MAP[key]
        raise ValueError(
            f"Unknown dtype string '{dtype}'. Supported: {sorted(_DTYPE_MAP.keys())}"
        )
    raise TypeError(f"Unsupported dtype type: {type(dtype)}")


# ---------------------------------------------------------------------------
# Low-level loading functions
# ---------------------------------------------------------------------------

def load_safetensors(
    path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> Dict[str, torch.Tensor]:
    """Load weights from a single safetensors file.

    Args:
        path: Path to the .safetensors file.
        device: Target device for the loaded tensors.
        dtype: Target dtype for the loaded tensors.

    Returns:
        Dict mapping weight names to tensors.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If the safetensors library is not installed.
    """
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError(
            "safetensors library is required. Install with: pip install safetensors"
        )

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Safetensors file not found: {path}")

    logger.info("Loading safetensors: %s", path)
    state_dict = load_file(path, device=device)

    if dtype is not None:
        state_dict = {
            k: v.to(dtype=dtype) if v.is_floating_point() else v
            for k, v in state_dict.items()
        }

    return state_dict


def load_pytorch_bin(
    path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> Dict[str, torch.Tensor]:
    """Load weights from a single PyTorch .bin file.

    Args:
        path: Path to the .bin file.
        device: Target device for the loaded tensors.
        dtype: Target dtype for the loaded tensors.

    Returns:
        Dict mapping weight names to tensors.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PyTorch bin file not found: {path}")

    logger.info("Loading PyTorch bin: %s", path)
    state_dict = torch.load(path, map_location=device, weights_only=True)

    if dtype is not None:
        state_dict = {
            k: v.to(dtype=dtype) if v.is_floating_point() else v
            for k, v in state_dict.items()
        }

    return state_dict


# ---------------------------------------------------------------------------
# Weight file discovery
# ---------------------------------------------------------------------------

_SAFETENSORS_PATTERN = re.compile(r"model-\d+-of-\d+\.safetensors")
_BIN_PATTERN = re.compile(r"pytorch_model-\d+-of-\d+\.bin")


def get_weight_files(model_path: str) -> List[str]:
    """Find all weight files in the given model directory.

    Looks for safetensors files first (preferred), then falls back to
    PyTorch .bin files. Handles both sharded and single-file formats.

    Args:
        model_path: Path to the model directory.

    Returns:
        Sorted list of absolute paths to weight files.

    Raises:
        FileNotFoundError: If no weight files are found.
    """
    model_path = os.path.abspath(model_path)

    # Try safetensors first (preferred format)
    safetensors_files = _find_files(model_path, "*.safetensors", _SAFETENSORS_PATTERN)
    if safetensors_files:
        return safetensors_files

    # Fall back to PyTorch .bin
    bin_files = _find_files(model_path, "*.bin", _BIN_PATTERN)
    if bin_files:
        return bin_files

    raise FileNotFoundError(
        f"No weight files found in {model_path}. "
        "Expected .safetensors or .bin files."
    )


def _find_files(
    model_path: str,
    glob_pattern: str,
    shard_pattern: re.Pattern,
) -> List[str]:
    """Find and sort weight files matching the given patterns.

    Args:
        model_path: Directory to search in.
        glob_pattern: Glob pattern for initial file discovery.
        shard_pattern: Regex pattern for sorting sharded files.

    Returns:
        Sorted list of file paths, or empty list if none found.
    """
    # Look for sharded files first
    all_files = sorted(glob.glob(os.path.join(model_path, glob_pattern)))

    # Filter to only model weight files (exclude optimizer, etc.)
    weight_files = []
    for f in all_files:
        basename = os.path.basename(f)
        if shard_pattern.match(basename) or basename in (
            "model.safetensors",
            "pytorch_model.bin",
        ) or basename.endswith(".safetensors") or basename.endswith(".bin"):
            weight_files.append(f)

    return weight_files


# ---------------------------------------------------------------------------
# Combined loading
# ---------------------------------------------------------------------------

def load_all_weights(
    model_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> Dict[str, torch.Tensor]:
    """Load all weights from a model directory.

    Automatically detects the file format (safetensors or PyTorch bin) and
    loads all shards, merging them into a single state dict.

    Args:
        model_path: Path to the model directory.
        device: Target device for the loaded tensors.
        dtype: Target dtype for the loaded tensors.

    Returns:
        Dict mapping weight names to tensors.
    """
    weight_files = get_weight_files(model_path)
    logger.info(
        "Found %d weight file(s) in %s", len(weight_files), model_path
    )

    combined: Dict[str, torch.Tensor] = {}
    for filepath in weight_files:
        if filepath.endswith(".safetensors"):
            shard = load_safetensors(filepath, device=device, dtype=dtype)
        elif filepath.endswith(".bin"):
            shard = load_pytorch_bin(filepath, device=device, dtype=dtype)
        else:
            logger.warning("Skipping unknown file format: %s", filepath)
            continue

        overlap = set(combined.keys()) & set(shard.keys())
        if overlap:
            logger.warning("Overlapping keys across shards: %s", overlap)
        combined.update(shard)

    logger.info("Loaded %d weight tensors total", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Tensor parallel sharding
# ---------------------------------------------------------------------------

def shard_weight(
    tensor: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    shard_dim: int,
) -> torch.Tensor:
    """Split a weight tensor for tensor parallelism.

    Args:
        tensor: The full weight tensor to shard.
        tp_rank: The tensor parallel rank of this device (0-indexed).
        tp_size: The total number of tensor parallel devices.
        shard_dim: The dimension along which to split.

    Returns:
        The shard of the tensor for the given tp_rank.
    """
    if tp_size <= 1:
        return tensor

    chunk_size = tensor.shape[shard_dim] // tp_size
    start = tp_rank * chunk_size
    end = start + chunk_size
    slices = [slice(None)] * tensor.ndim
    slices[shard_dim] = slice(start, end)
    return tensor[tuple(slices)].contiguous()


# ---------------------------------------------------------------------------
# Layer filtering
# ---------------------------------------------------------------------------

def _parse_layer_index(name: str) -> Optional[int]:
    """Extract the layer index from a weight name, if present.

    Args:
        name: Weight name (e.g. 'layers.3.attention.q_proj.weight').

    Returns:
        The layer index as an int, or None if not a per-layer weight.
    """
    match = re.search(r'layers\.(\d+)\.', name)
    if match:
        return int(match.group(1))
    return None


def _filter_weights(
    weights: Dict[str, torch.Tensor],
    layers: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Filter weights to only include specific layers.

    Args:
        weights: Full weight dict.
        layers: List of layer indices to keep. None means keep all.

    Returns:
        Filtered weight dict.
    """
    if layers is None:
        return weights

    layer_set = set(layers)
    filtered: Dict[str, torch.Tensor] = {}
    for name, tensor in weights.items():
        idx = _parse_layer_index(name)
        if idx is None or idx in layer_set:
            filtered[name] = tensor
    return filtered


# ---------------------------------------------------------------------------
# Model config helpers
# ---------------------------------------------------------------------------

def _detect_model_type(model_path: str) -> str:
    """Detect the model architecture type from the config.json.

    Args:
        model_path: Path to the model directory.

    Returns:
        The model type string (e.g. 'llama', 'qwen2', 'gemma').
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        logger.warning("config.json not found, defaulting to 'llama'")
        return "llama"

    with open(config_path, "r") as f:
        config = json.load(f)

    model_type = config.get("model_type", "llama")
    return model_type


# ---------------------------------------------------------------------------
# WeightLoader class
# ---------------------------------------------------------------------------

class WeightLoader:
    """High-level weight loader for LLM inference.

    Handles loading weights from HuggingFace format, mapping names,
    dtype conversion, tensor parallel sharding, and tied embeddings.

    Example::

        loader = WeightLoader(tp_rank=0, tp_size=2)
        stats = loader.load_weights(model, "/path/to/model", dtype=torch.float16)
        print(f"Loaded {stats['num_loaded']} weights in {stats['load_time']:.2f}s")
    """

    def __init__(
        self,
        tp_rank: int = 0,
        tp_size: int = 1,
        model_type: Optional[str] = None,
    ) -> None:
        """Initialize the WeightLoader.

        Args:
            tp_rank: Tensor parallel rank of this device (0-indexed).
            tp_size: Total number of tensor parallel devices.
            model_type: Override model architecture type. If None, auto-detected
                from config.json.
        """
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.model_type = model_type
        self._mapper: Optional[WeightMapper] = None

    @property
    def mapper(self) -> WeightMapper:
        """Lazily initialized WeightMapper."""
        if self._mapper is None:
            raise RuntimeError(
                "Mapper not initialized. Call load_weights() first or set model_type."
            )
        return self._mapper

    def load_weights(
        self,
        model: torch.nn.Module,
        model_path: str,
        device: str = "cuda",
        dtype: Any = torch.float16,
        layers: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Load weights into a model from a HuggingFace model directory.

        This method:
        1. Detects the model architecture and creates the appropriate mapper.
        2. Loads all weight files from the directory.
        3. Maps HF weight names to internal parameter names.
        4. Applies dtype conversion.
        5. Shards weights for tensor parallelism if tp_size > 1.
        6. Handles tied embeddings.
        7. Loads the weights into the model.

        Args:
            model: The target model (nn.Module) to load weights into.
            model_path: Path to the HuggingFace model directory.
            device: Target device ('cpu', 'cuda', etc.).
            dtype: Target dtype. Can be torch.dtype or string ('fp16', 'bf16', etc.).
            layers: Optional list of layer indices to load. None loads all layers.

        Returns:
            Dict with loading statistics:
                - num_total: Total number of weights found.
                - num_loaded: Number of weights successfully loaded.
                - num_skipped: Number of weights skipped (unmapped).
                - num_sharded: Number of weights that were sharded for TP.
                - load_time: Time in seconds for the full loading process.
                - model_type: Detected/specified model architecture type.
        """
        start_time = time.time()
        resolved_dtype = _resolve_dtype(dtype)

        # 1. Detect model type and create mapper
        effective_model_type = self.model_type or _detect_model_type(model_path)
        self._mapper = get_mapper(effective_model_type)
        logger.info("Using mapper for model_type='%s'", effective_model_type)

        # 2. Load all weights from disk
        raw_weights = load_all_weights(model_path, device="cpu", dtype=resolved_dtype)

        # 3. Map HF names to internal names
        hf_names = list(raw_weights.keys())
        name_mapping = self._mapper.map_all(hf_names)

        mapped_weights: Dict[str, torch.Tensor] = {}
        num_skipped = 0
        for hf_name, tensor in raw_weights.items():
            if hf_name in name_mapping:
                mapped_weights[name_mapping[hf_name]] = tensor
            else:
                num_skipped += 1
                logger.debug("Skipped unmapped weight: %s", hf_name)

        # 4. Filter to specific layers if requested
        mapped_weights = _filter_weights(mapped_weights, layers)

        # 5. Resolve tied weights
        tied_weights: Dict[str, torch.Tensor] = {}
        for internal_name, tensor in mapped_weights.items():
            resolved_name = self._mapper.resolve_tied(internal_name)
            if resolved_name != internal_name:
                # This weight is tied; we store it under the source name
                if resolved_name not in mapped_weights:
                    tied_weights[resolved_name] = tensor
                # The tied target is not loaded separately
            else:
                tied_weights[internal_name] = tensor

        # 6. Shard for tensor parallelism
        num_sharded = 0
        if self.tp_size > 1:
            sharded: Dict[str, torch.Tensor] = {}
            for name, tensor in tied_weights.items():
                spec = self._mapper.get_shard_spec(name)
                if spec is not None:
                    sharded[name] = shard_weight(
                        tensor, self.tp_rank, self.tp_size, spec.shard_dim
                    )
                    num_sharded += 1
                else:
                    sharded[name] = tensor
            tied_weights = sharded

        # 7. Load into model
        model_state = model.state_dict()
        num_loaded = 0
        loaded_keys: Set[str] = set()

        for param_name, tensor in tied_weights.items():
            if param_name in model_state:
                # Move to target device
                tensor = tensor.to(device=device)
                # Handle size mismatch (e.g. due to TP sharding)
                if tensor.shape == model_state[param_name].shape:
                    model_state[param_name].copy_(tensor)
                    loaded_keys.add(param_name)
                    num_loaded += 1
                else:
                    logger.warning(
                        "Shape mismatch for '%s': weight=%s, model=%s. Skipping.",
                        param_name,
                        tuple(tensor.shape),
                        tuple(model_state[param_name].shape),
                    )
            else:
                logger.debug("Weight '%s' not found in model parameters", param_name)

        # Also handle tied params: if lm_head.weight is tied to embed_tokens.weight,
        # make sure both are set
        for target_name, source_name in self._mapper.tied_weights.items():
            if source_name in loaded_keys and target_name in model_state:
                if target_name not in loaded_keys:
                    model_state[target_name].copy_(model_state[source_name])
                    num_loaded += 1

        load_time = time.time() - start_time
        stats = {
            "num_total": len(raw_weights),
            "num_loaded": num_loaded,
            "num_skipped": num_skipped,
            "num_sharded": num_sharded,
            "load_time": load_time,
            "model_type": effective_model_type,
        }

        logger.info(
            "Weight loading complete: %d/%d loaded, %d skipped, "
            "%d sharded, %.2fs",
            num_loaded,
            len(raw_weights),
            num_skipped,
            num_sharded,
            load_time,
        )

        return stats
