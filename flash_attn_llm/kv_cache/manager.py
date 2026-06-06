"""KV Cache management for LLM inference.

Provides contiguous per-request KV cache (``KVCache``) and a paged KV cache
manager (``PagedKVCacheManager``) inspired by vLLM that supports block-level
allocation, prefix caching, sliding-window eviction, and batch decode.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


# ======================================================================
# Configuration
# ======================================================================


@dataclass
class KVCacheConfig:
    """Configuration for the paged KV cache manager.

    Args:
        max_seq_len: Maximum sequence length supported.
        num_layers: Number of transformer layers.
        num_kv_heads: Number of key/value attention heads (may differ from
            query heads in GQA / MQA).
        head_dim: Dimension of each attention head.
        dtype: Data type for cache tensors.
        device: Device on which cache tensors are allocated.
        block_size: Number of tokens per paged-attention block.
        num_blocks: Total number of physical blocks to pre-allocate.
            0 means auto-calculate from available GPU memory.
        sliding_window: Size of the sliding window for local attention.
            0 means no sliding window (full attention).
        enable_prefix_cache: Whether to enable prefix caching.
    """

    max_seq_len: int = 4096
    num_layers: int = 32
    num_kv_heads: int = 32
    head_dim: int = 128
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    block_size: int = 16
    num_blocks: int = 0
    sliding_window: int = 0
    enable_prefix_cache: bool = True


# ======================================================================
# Per-layer KV Cache (contiguous, single request)
# ======================================================================


class KVCache:
    """Contiguous KV cache for a single request at one layer.

    Stores ``cache_k`` and ``cache_v`` tensors in a pre-allocated contiguous
    buffer of shape ``(2, max_seq_len, num_kv_heads, head_dim)`` and supports
    incremental appending of new key/value vectors.

    Args:
        num_kv_heads: Number of KV heads.
        head_dim: Dimension per head.
        max_seq_len: Maximum sequence length.
        dtype: Tensor dtype.
        device: Tensor device.
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> None:
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._max_seq_len = max_seq_len
        self._dtype = dtype
        self._device = device
        self._length: int = 0

        # Shape: (max_seq_len, num_kv_heads, head_dim)
        self.cache_k = torch.zeros(
            (max_seq_len, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.cache_v = torch.zeros(
            (max_seq_len, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )

    @property
    def length(self) -> int:
        """Current number of cached token positions."""
        return self._length

    def append(self, k: Tensor, v: Tensor) -> None:
        """Append new key/value tensors to the cache.

        Args:
            k: Key tensor of shape ``(seq_len, num_kv_heads, head_dim)``.
            v: Value tensor of shape ``(seq_len, num_kv_heads, head_dim)``.
        """
        seq_len = k.shape[0]
        if self._length + seq_len > self._max_seq_len:
            raise ValueError(
                f"KV cache overflow: trying to store {self._length + seq_len} "
                f"tokens but max_seq_len is {self._max_seq_len}"
            )
        self.cache_k[self._length : self._length + seq_len].copy_(k)
        self.cache_v[self._length : self._length + seq_len].copy_(v)
        self._length += seq_len

    def get(self) -> Tuple[Tensor, Tensor]:
        """Return the cached key and value tensors up to the current length.

        Returns:
            Tuple of (cache_k, cache_v), each of shape
            ``(length, num_kv_heads, head_dim)``.
        """
        return self.cache_k[: self._length], self.cache_v[: self._length]

    def reset(self) -> None:
        """Reset the cache to empty."""
        self._length = 0

    def shrink_to(self, new_length: int) -> None:
        """Truncate the cache to a shorter length (for sliding window).

        Args:
            new_length: The desired cache length after truncation.
        """
        if new_length < 0:
            new_length = 0
        if new_length < self._length:
            # Shift the cache contents to the beginning
            start = self._length - new_length
            self.cache_k[:new_length].copy_(self.cache_k[start : self._length])
            self.cache_v[:new_length].copy_(self.cache_v[start : self._length])
            self._length = new_length


# ======================================================================
# Paged KV Cache Manager
# ======================================================================


class PagedKVCacheManager:
    """Paged KV Cache Manager for multiple concurrent requests.

    Implements a vLLM-style paged attention cache that manages block-level
    allocation across multiple requests. Key features:

    - **Block allocation / deallocation** with a free list.
    - **Request-to-block-table mapping** for paged attention kernels.
    - **Prefix caching** – reuse KV blocks for shared prompt prefixes.
    - **Sliding window** – evict blocks that fall outside the window.
    - **Batch management** – efficient handling of variable-length caches.

    The cache pool is a single large tensor per layer, partitioned into
    *num_blocks* physical blocks of *block_size* tokens each.

    Args:
        config: A ``KVCacheConfig`` instance.
    """

    def __init__(self, config: KVCacheConfig) -> None:
        self._config = config
        self._block_size = config.block_size
        self._num_layers = config.num_layers
        self._num_kv_heads = config.num_kv_heads
        self._head_dim = config.head_dim
        self._dtype = config.dtype
        self._device = config.device
        self._sliding_window = config.sliding_window
        self._enable_prefix_cache = config.enable_prefix_cache

        # Determine number of blocks
        if config.num_blocks > 0:
            self._num_blocks = config.num_blocks
        else:
            self._num_blocks = self._auto_calculate_num_blocks()

        # Pre-allocate the KV cache pool.
        # Shape per layer: (num_blocks, 2, block_size, num_kv_heads, head_dim)
        # Index [block_idx, 0, ...] = key cache, [block_idx, 1, ...] = value cache
        self._kv_pool: List[Tensor] = []
        for _ in range(self._num_layers):
            layer_pool = torch.zeros(
                (self._num_blocks, 2, self._block_size, self._num_kv_heads, self._head_dim),
                dtype=self._dtype,
                device=self._device,
            )
            self._kv_pool.append(layer_pool)

        # Free list: stack of available physical block indices
        self._free_blocks: List[int] = list(range(self._num_blocks - 1, -1, -1))

        # Per-request state
        # request_id -> list of physical block indices (block table)
        self._block_tables: Dict[str, List[int]] = {}
        # request_id -> number of valid tokens in the last (possibly partial) block
        self._num_tokens: Dict[str, int] = {}

        # Prefix cache: hash -> list of physical block indices
        self._prefix_cache: Dict[str, List[int]] = {}
        # Reverse mapping: physical block index -> prefix cache key (for ref counting)
        self._block_to_prefix_key: Dict[int, str] = {}
        # Reference counts for prefix-cached blocks
        self._prefix_ref_count: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Auto-calculate num_blocks
    # ------------------------------------------------------------------

    def _auto_calculate_num_blocks(self) -> int:
        """Calculate the number of blocks based on available GPU memory.

        Uses ``torch.cuda.mem_get_info`` to determine free memory and
        allocates ~80 % of it for the KV cache pool.
        """
        if self._device != "cuda" or not torch.cuda.is_available():
            # Fallback for CPU: use a reasonable default
            return 1024

        free_mem, _total = torch.cuda.mem_get_info()
        # Reserve 20 % for other tensors
        usable_mem = int(free_mem * 0.8)

        bytes_per_block_per_layer = (
            2  # k + v
            * self._block_size
            * self._num_kv_heads
            * self._head_dim
            * torch.tensor([], dtype=self._dtype).element_size()
        )
        bytes_per_block = bytes_per_block_per_layer * self._num_layers

        if bytes_per_block == 0:
            return 1024

        num_blocks = usable_mem // bytes_per_block
        # Clamp to a reasonable range
        return max(256, min(num_blocks, 1 << 24))

    # ------------------------------------------------------------------
    # Block allocation primitives
    # ------------------------------------------------------------------

    def _allocate_block(self) -> int:
        """Pop a physical block from the free list.

        Returns:
            The physical block index.

        Raises:
            MemoryError: If no free blocks are available.
        """
        if not self._free_blocks:
            raise MemoryError(
                "KV cache out of memory: no free blocks available. "
                f"Total blocks: {self._num_blocks}, "
                f"Free blocks: {len(self._free_blocks)}"
            )
        return self._free_blocks.pop()

    def _free_block(self, block_idx: int) -> None:
        """Return a physical block to the free list."""
        self._free_blocks.append(block_idx)

    def _allocate_blocks(self, num_blocks: int) -> List[int]:
        """Allocate *num_blocks* contiguous or non-contiguous physical blocks.

        Returns:
            List of physical block indices.
        """
        if len(self._free_blocks) < num_blocks:
            raise MemoryError(
                f"KV cache out of memory: requested {num_blocks} blocks, "
                f"but only {len(self._free_blocks)} are free."
            )
        return [self._free_blocks.pop() for _ in range(num_blocks)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(self, request_id: str, num_tokens: int) -> None:
        """Allocate cache blocks for a new request.

        If prefix caching is enabled, the method first checks whether any
        prefix of the token sequence is already cached and reuses those
        blocks.  New blocks are allocated for the remaining tokens.

        Args:
            request_id: Unique identifier for the request.
            num_tokens: Number of tokens to allocate cache space for.
        """
        if request_id in self._block_tables:
            raise ValueError(f"Request {request_id} already has allocated blocks")

        needed_blocks = (num_tokens + self._block_size - 1) // self._block_size
        block_ids = self._allocate_blocks(needed_blocks)
        self._block_tables[request_id] = block_ids
        self._num_tokens[request_id] = num_tokens

    def append(
        self, request_id: str, k: Tensor, v: Tensor, layer_idx: int
    ) -> None:
        """Append new key/value data to the cache for a request.

        Handles dynamic expansion: if the existing blocks cannot hold the
        new tokens, additional blocks are allocated automatically.

        Args:
            request_id: The request identifier.
            k: Key tensor of shape ``(seq_len, num_kv_heads, head_dim)``.
            v: Value tensor of shape ``(seq_len, num_kv_heads, head_dim)``.
            layer_idx: The transformer layer index.
        """
        if request_id not in self._block_tables:
            raise KeyError(f"Request {request_id} not found in block tables")

        seq_len = k.shape[0]
        current_tokens = self._num_tokens[request_id]
        new_total = current_tokens + seq_len

        # Check if we need more blocks
        current_blocks = len(self._block_tables[request_id])
        needed_blocks = (new_total + self._block_size - 1) // self._block_size

        if needed_blocks > current_blocks:
            extra_blocks = self._allocate_blocks(needed_blocks - current_blocks)
            self._block_tables[request_id].extend(extra_blocks)

        # Write the KV data into the pool
        block_table = self._block_tables[request_id]
        pool = self._kv_pool[layer_idx]

        for i in range(seq_len):
            token_pos = current_tokens + i
            block_idx = token_pos // self._block_size
            offset_in_block = token_pos % self._block_size
            physical_block = block_table[block_idx]

            pool[physical_block, 0, offset_in_block].copy_(k[i])
            pool[physical_block, 1, offset_in_block].copy_(v[i])

        self._num_tokens[request_id] = new_total

        # Sliding window eviction
        if self._sliding_window > 0 and new_total > self._sliding_window:
            self._evict_sliding_window(request_id)

    def get_cache(
        self, request_id: str, layer_idx: int
    ) -> Tuple[Tensor, Tensor]:
        """Retrieve the full key and value cache tensors for a request.

        Gathers data from scattered physical blocks into contiguous tensors.

        Args:
            request_id: The request identifier.
            layer_idx: The transformer layer index.

        Returns:
            Tuple of (keys, values), each of shape
            ``(num_tokens, num_kv_heads, head_dim)``.
        """
        if request_id not in self._block_tables:
            raise KeyError(f"Request {request_id} not found in block tables")

        num_tokens = self._num_tokens[request_id]
        block_table = self._block_tables[request_id]
        pool = self._kv_pool[layer_idx]

        num_full_blocks = num_tokens // self._block_size
        remainder = num_tokens % self._block_size

        # Gather full blocks
        keys_list: List[Tensor] = []
        values_list: List[Tensor] = []

        for i in range(num_full_blocks):
            physical_block = block_table[i]
            keys_list.append(pool[physical_block, 0, :self._block_size])
            values_list.append(pool[physical_block, 1, :self._block_size])

        # Gather partial last block
        if remainder > 0:
            physical_block = block_table[num_full_blocks]
            keys_list.append(pool[physical_block, 0, :remainder])
            values_list.append(pool[physical_block, 1, :remainder])

        if keys_list:
            keys = torch.cat(keys_list, dim=0)
            values = torch.cat(values_list, dim=0)
        else:
            keys = torch.zeros(
                (0, self._num_kv_heads, self._head_dim),
                dtype=self._dtype,
                device=self._device,
            )
            values = keys.clone()

        return keys, values

    def get_block_table(self, request_id: str) -> Tensor:
        """Return the block table (physical block indices) for a request.

        The returned tensor is suitable for passing to paged-attention
        kernels.

        Args:
            request_id: The request identifier.

        Returns:
            A 1-D int tensor of physical block indices.
        """
        if request_id not in self._block_tables:
            raise KeyError(f"Request {request_id} not found in block tables")
        return torch.tensor(
            self._block_tables[request_id], dtype=torch.int32, device=self._device
        )

    def release(self, request_id: str) -> None:
        """Release all cache blocks held by a request.

        Blocks that are part of a cached prefix are only freed when no
        other request references them.

        Args:
            request_id: The request identifier.
        """
        if request_id not in self._block_tables:
            return

        block_ids = self._block_tables.pop(request_id)
        self._num_tokens.pop(request_id, None)

        for block_idx in block_ids:
            # Check if this block is part of a prefix cache
            if block_idx in self._block_to_prefix_key:
                prefix_key = self._block_to_prefix_key[block_idx]
                self._prefix_ref_count[prefix_key] -= 1
                if self._prefix_ref_count[prefix_key] <= 0:
                    # No more references; free the prefix cache entry
                    for bidx in self._prefix_cache.pop(prefix_key, []):
                        self._block_to_prefix_key.pop(bidx, None)
                        self._free_block(bidx)
                    self._prefix_ref_count.pop(prefix_key, None)
            else:
                self._free_block(block_idx)

    def get_cache_len(self, request_id: str) -> int:
        """Return the current number of cached tokens for a request.

        Args:
            request_id: The request identifier.

        Returns:
            Number of cached tokens.
        """
        return self._num_tokens.get(request_id, 0)

    def can_allocate(self, num_tokens: int) -> bool:
        """Check whether there are enough free blocks for *num_tokens*.

        Args:
            num_tokens: Number of tokens to check.

        Returns:
            True if allocation is possible, False otherwise.
        """
        needed = (num_tokens + self._block_size - 1) // self._block_size
        return len(self._free_blocks) >= needed

    def get_num_free_blocks(self) -> int:
        """Return the number of currently free blocks."""
        return len(self._free_blocks)

    # ------------------------------------------------------------------
    # Prefix caching
    # ------------------------------------------------------------------

    def get_prefix_cache_key(self, token_ids: list[int]) -> str:
        """Compute a deterministic cache key for a token sequence.

        Args:
            token_ids: The token ID sequence.

        Returns:
            A hex digest string suitable for use as a dictionary key.
        """
        raw = ",".join(str(t) for t in token_ids)
        return hashlib.sha256(raw.encode()).hexdigest()

    def lookup_prefix(self, token_ids: list[int]) -> Optional[list[int]]:
        """Look up cached blocks for a token prefix.

        Searches for the longest prefix of *token_ids* that is already
        cached and returns the corresponding physical block indices.

        Args:
            token_ids: The full token ID sequence.

        Returns:
            A list of physical block indices if a prefix is found, or None.
        """
        if not self._enable_prefix_cache:
            return None

        # Try progressively shorter prefixes (aligned to block boundaries)
        max_blocks = len(token_ids) // self._block_size
        for num_blocks in range(max_blocks, 0, -1):
            prefix = token_ids[: num_blocks * self._block_size]
            key = self.get_prefix_cache_key(prefix)
            if key in self._prefix_cache:
                return list(self._prefix_cache[key])

        return None

    def cache_prefix(self, token_ids: list[int], request_id: str) -> int:
        """Cache the prefix of a token sequence for future reuse.

        Only full blocks (aligned to *block_size*) are cached.

        Args:
            token_ids: The full token ID sequence.
            request_id: The request that owns these tokens.

        Returns:
            The number of tokens that were cached as a prefix.
        """
        if not self._enable_prefix_cache:
            return 0

        num_full_blocks = len(token_ids) // self._block_size
        if num_full_blocks == 0:
            return 0

        prefix_tokens = token_ids[: num_full_blocks * self._block_size]
        key = self.get_prefix_cache_key(prefix_tokens)

        if key in self._prefix_cache:
            # Already cached; just bump the ref count
            self._prefix_ref_count[key] += 1
            return len(prefix_tokens)

        # Record the mapping from physical blocks to this prefix key
        block_table = self._block_tables.get(request_id, [])
        if len(block_table) < num_full_blocks:
            return 0

        prefix_blocks = block_table[:num_full_blocks]
        self._prefix_cache[key] = list(prefix_blocks)
        self._prefix_ref_count[key] = 1
        for bidx in prefix_blocks:
            self._block_to_prefix_key[bidx] = key

        return len(prefix_tokens)

    # ------------------------------------------------------------------
    # Sliding window
    # ------------------------------------------------------------------

    def _evict_sliding_window(self, request_id: str) -> None:
        """Evict blocks that fall outside the sliding window.

        Blocks whose tokens are entirely before
        ``(current_length - sliding_window)`` are freed.

        Args:
            request_id: The request identifier.
        """
        if self._sliding_window <= 0:
            return

        current_len = self._num_tokens[request_id]
        # Number of tokens that can be evicted
        evictable_tokens = current_len - self._sliding_window
        if evictable_tokens <= 0:
            return

        # Only evict full blocks that are entirely outside the window
        evictable_blocks = evictable_tokens // self._block_size
        if evictable_blocks <= 0:
            return

        block_table = self._block_tables[request_id]

        # Free the oldest blocks
        freed_blocks = block_table[:evictable_blocks]
        self._block_tables[request_id] = block_table[evictable_blocks:]

        for bidx in freed_blocks:
            if bidx in self._block_to_prefix_key:
                # Don't free prefix-cached blocks; just remove from this request
                prefix_key = self._block_to_prefix_key[bidx]
                self._prefix_ref_count[prefix_key] -= 1
                if self._prefix_ref_count[prefix_key] <= 0:
                    for pbidx in self._prefix_cache.pop(prefix_key, []):
                        self._block_to_prefix_key.pop(pbidx, None)
                        self._free_block(pbidx)
                    self._prefix_ref_count.pop(prefix_key, None)
            else:
                self._free_block(bidx)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def get_batch_block_tables(
        self, request_ids: List[str], max_num_blocks: int
    ) -> Tensor:
        """Build a padded 2-D block table for a batch of requests.

        Each row corresponds to one request. Shorter block tables are
        padded with -1.

        Args:
            request_ids: List of request IDs in the batch.
            max_num_blocks: Maximum number of blocks per row (determines
                the number of columns).

        Returns:
            An int32 tensor of shape ``(batch_size, max_num_blocks)``.
        """
        batch_size = len(request_ids)
        table = torch.full(
            (batch_size, max_num_blocks),
            -1,
            dtype=torch.int32,
            device=self._device,
        )
        for i, rid in enumerate(request_ids):
            bt = self._block_tables.get(rid, [])
            num = min(len(bt), max_num_blocks)
            table[i, :num] = torch.tensor(bt[:num], dtype=torch.int32)
        return table

    def get_batch_cache_lens(self, request_ids: List[str]) -> Tensor:
        """Return a tensor of cache lengths for a batch of requests.

        Args:
            request_ids: List of request IDs.

        Returns:
            An int32 tensor of shape ``(batch_size,)``.
        """
        lens = [self._num_tokens.get(rid, 0) for rid in request_ids]
        return torch.tensor(lens, dtype=torch.int32, device=self._device)

    def get_batch_seq_positions(
        self, request_ids: List[str], max_seq_len: int
    ) -> Tensor:
        """Build position IDs for a batch of requests.

        Args:
            request_ids: List of request IDs.
            max_seq_len: Maximum sequence length (number of columns).

        Returns:
            An int64 tensor of shape ``(batch_size, max_seq_len)`` with
            position IDs, padded with 0.
        """
        batch_size = len(request_ids)
        positions = torch.zeros(
            (batch_size, max_seq_len), dtype=torch.int64, device=self._device
        )
        for i, rid in enumerate(request_ids):
            cache_len = self._num_tokens.get(rid, 0)
            positions[i, :cache_len] = torch.arange(cache_len, dtype=torch.int64)
        return positions

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def num_blocks(self) -> int:
        """Total number of physical blocks."""
        return self._num_blocks

    @property
    def total_memory_bytes(self) -> int:
        """Total GPU memory allocated for the KV cache pool in bytes."""
        return sum(p.numel() * p.element_size() for p in self._kv_pool)

    def memory_usage_summary(self) -> str:
        """Return a human-readable summary of cache memory usage."""
        used = self._num_blocks - len(self._free_blocks)
        total = self._num_blocks
        pct = (used / total * 100) if total > 0 else 0
        total_bytes = self.total_memory_bytes
        return (
            f"PagedKVCache: {used}/{total} blocks used ({pct:.1f}%), "
            f"{total_bytes / (1 << 30):.2f} GB total, "
            f"{len(self._block_tables)} active requests, "
            f"{len(self._prefix_cache)} prefix cache entries"
        )
