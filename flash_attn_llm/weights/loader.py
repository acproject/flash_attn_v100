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
import torch.nn as nn

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
    device: str = "cuda",
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
    device: str = "cuda",
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
    device: str = "粗大",
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
            device: Target device ('cuda', 'cuda', etc.).
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

        # 2. Discover weight files
        weight_files = get_weight_files(model_path)
        logger.info("Found %d weight file(s) in %s", len(weight_files), model_path)

        # 3. Get model state dict once (references to parameters on their devices)
        model_state = model.state_dict()
        num_loaded = 0
        num_skipped = 0
        num_sharded = 0
        num_total = 0
        num_skipped_no_mapping = 0
        num_skipped_not_in_model = 0
        loaded_keys: Set[str] = set()

        # 4. Load shard by shard, copy to GPU immediately, free cuda memory
        for filepath in weight_files:
            if filepath.endswith(".safetensors"):
                # Load each tensor directly to its target GPU device
                # First pass: determine all target devices for this shard
                from safetensors import safe_open
                
                with safe_open(filepath, framework="pt", device="cuda") as f:
                    shard_keys = list(f.keys())
                    num_total += len(shard_keys)

                    hf_names = shard_keys
                    name_mapping = self._mapper.map_all(hf_names)

                    # Group tensors by target device
                    tensors_by_device = {}
                    for hf_name in shard_keys:
                        if hf_name not in name_mapping:
                            num_skipped += 1
                            continue

                        internal_name = name_mapping[hf_name]

                        # Filter by layer index if requested
                        if layers is not None:
                            idx = _parse_layer_index(internal_name)
                            if idx is not None and idx not in set(layers):
                                num_skipped += 1
                                continue

                        # Resolve tied weights
                        resolved_name = self._mapper.resolve_tied(internal_name)
                        if resolved_name != internal_name:
                            if resolved_name in model_state:
                                internal_name = resolved_name
                            else:
                                num_skipped += 1
                                continue

                        # Determine target device from model parameter
                        if internal_name not in model_state:
                            logger.debug("Weight '%s' not found in model parameters", internal_name)
                            num_skipped += 1
                            num_skipped_not_in_model += 1
                            continue

                        target_device = model_state[internal_name].device
                        
                        if target_device not in tensors_by_device:
                            tensors_by_device[target_device] = []
                        tensors_by_device[target_device].append((hf_name, internal_name))

                # Second pass: load tensors by device to avoid cuda memory
                for target_device, tensor_pairs in tensors_by_device.items():
                    # Open file with target device
                    with safe_open(filepath, framework="pt", device=str(target_device)) as f:
                        for hf_name, internal_name in tensor_pairs:
                            # Load tensor directly to target device
                            tensor = f.get_tensor(hf_name)
                            tensor = tensor.to(dtype=model_state[internal_name].dtype)

                            # Shard for tensor parallelism
                            if self.tp_size > 1:
                                spec = self._mapper.get_shard_spec(internal_name)
                                if spec is not None:
                                    tensor = shard_weight(
                                        tensor, self.tp_rank, self.tp_size, spec.shard_dim
                                    )
                                    num_sharded += 1

                            if tensor.shape == model_state[internal_name].shape:
                                model_state[internal_name].data.copy_(tensor)
                                loaded_keys.add(internal_name)
                                num_loaded += 1
                            else:
                                logger.warning(
                                    "Shape mismatch for '%s': weight=%s, model=%s. Skipping.",
                                    internal_name,
                                    tuple(tensor.shape),
                                    tuple(model_state[internal_name].shape),
                                )
            elif filepath.endswith(".bin"):
                shard = load_pytorch_bin(filepath, device="cuda", dtype=resolved_dtype)
                num_total += len(shard)

                hf_names = list(shard.keys())
                name_mapping = self._mapper.map_all(hf_names)

                for hf_name, tensor in shard.items():
                    if hf_name not in name_mapping:
                        num_skipped += 1
                        num_skipped_no_mapping += 1
                        continue

                    internal_name = name_mapping[hf_name]

                    if layers is not None:
                        idx = _parse_layer_index(internal_name)
                        if idx is not None and idx not in set(layers):
                            num_skipped += 1
                            continue

                    resolved_name = self._mapper.resolve_tied(internal_name)
                    if resolved_name != internal_name:
                        if resolved_name in model_state:
                            internal_name = resolved_name
                        else:
                            num_skipped += 1
                            continue

                    if internal_name not in model_state:
                        num_skipped += 1
                        continue

                    target_device = model_state[internal_name].device
                    tensor = tensor.to(device=target_device, dtype=model_state[internal_name].dtype)

                    if self.tp_size > 1:
                        spec = self._mapper.get_shard_spec(internal_name)
                        if spec is not None:
                            tensor = shard_weight(
                                tensor, self.tp_rank, self.tp_size, spec.shard_dim
                            )
                            num_sharded += 1

                    if tensor.shape == model_state[internal_name].shape:
                        model_state[internal_name].data.copy_(tensor)
                        loaded_keys.add(internal_name)
                        num_loaded += 1
                    else:
                        logger.warning(
                            "Shape mismatch for '%s': weight=%s, model=%s. Skipping.",
                            internal_name,
                            tuple(tensor.shape),
                            tuple(model_state[internal_name].shape),
                        )

                del shard
                del name_mapping
            else:
                logger.warning("Skipping unknown file format: %s", filepath)
                continue

        # 5. Handle tied params: copy source -> target on the same device
        for target_name, source_name in self._mapper.tied_weights.items():
            print(
                f"Tied weight check: target={target_name} (in model={target_name in model_state}, "
                f"loaded={target_name in loaded_keys}), source={source_name} (loaded={source_name in loaded_keys})"
            )
            if source_name in loaded_keys and target_name in model_state:
                if target_name not in loaded_keys:
                    src = model_state[source_name]
                    tgt = model_state[target_name]
                    print(
                        f"Tied weight: copying {source_name} ({src.device}, shape={tuple(src.shape)}) "
                        f"-> {target_name} ({tgt.device}, shape={tuple(tgt.shape)})"
                    )
                    # to_empty() tensors don't support copy_() reliably.
                    # Also, direct .to() cross-device is broken in PyTorch 2.8+V100
                    # for large tensors. Must transfer via CPU.
                    src_param = dict(model.named_parameters())[source_name]
                    print(f"  src from model.named_parameters(): sum={src_param.sum().item():.4f}, device={src_param.device}")
                    src_on_tgt_device = src_param.data.cpu().to(tgt.device)
                    print(f"  src_on_tgt_device sum: {src_on_tgt_device.sum().item():.4f}, device: {src_on_tgt_device.device}")
                    # Find the module that owns this parameter and replace it
                    parts = target_name.rsplit(".", 1)
                    if len(parts) == 2:
                        parent_name, attr_name = parts
                        parent_module = model.get_submodule(parent_name)
                        setattr(parent_module, attr_name, nn.Parameter(src_on_tgt_device))
                    else:
                        # Top-level parameter
                        setattr(model, target_name, nn.Parameter(src_on_tgt_device))
                    # Verify
                    new_param = model.get_submodule(target_name.rsplit(".", 1)[0]) if "." in target_name else getattr(model, target_name)
                    if "." in target_name:
                        new_param = getattr(new_param, target_name.rsplit(".", 1)[1])
                    print(f"  After setattr, {target_name} sum: {new_param.sum().item():.4f}, device: {new_param.device}")
                    loaded_keys.add(target_name)
                    num_loaded += 1

        load_time = time.time() - start_time
        stats = {
            "num_total": num_total,
            "num_loaded": num_loaded,
            "num_skipped": num_skipped,
            "num_sharded": num_sharded,
            "load_time": load_time,
            "model_type": effective_model_type,
        }

        logger.info(
            "Weight loading complete: %d/%d loaded, %d skipped "
            "(no_mapping=%d, not_in_model=%d), "
            "%d sharded, %.2fs",
            num_loaded,
            num_total,
            num_skipped,
            num_skipped_no_mapping,
            num_skipped_not_in_model,
            num_sharded,
            load_time,
        )

        return stats
