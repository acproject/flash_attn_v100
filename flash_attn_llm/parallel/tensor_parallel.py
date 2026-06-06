"""Tensor parallelism module for multi-GPU LLM inference.

Provides TensorParallelManager which handles:
- Process group initialization for NCCL communication
- Weight sharding across TP ranks
- All-reduce and all-gather collective operations
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class TensorParallelManager:
    """Manages tensor parallelism across multiple GPUs.

    Provides utilities for:
    - Initializing the NCCL process group
    - Sharding model weights across ranks
    - Performing all-reduce and all-gather collectives
    """

    _instance: Optional[TensorParallelManager] = None

    def __init__(self, tp_size: int = 1):
        """
        Args:
            tp_size: Tensor parallelism degree (number of GPUs).
                     1 means no tensor parallelism.
        """
        self.tp_size = tp_size
        self.tp_rank = 0
        self.device = "cuda"
        self._initialized = False
        self._backend: Optional[str] = None

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, tp_size: int = 1) -> TensorParallelManager:
        """Get or create the global TensorParallelManager singleton."""
        if cls._instance is None:
            cls._instance = cls(tp_size=tp_size)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (mainly for testing)."""
        if cls._instance is not None and cls._instance._initialized:
            cls._instance.shutdown()
        cls._instance = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, backend: str = "nccl", init_method: Optional[str] = None) -> None:
        """Initialize the process group for tensor parallelism.

        Args:
            backend: Communication backend (typically 'nccl' for GPU).
            init_method: URL specifying how to initialize the process group.
                         If None, uses env:// (requires MASTER_ADDR/MASTER_PORT).
        """
        if self._initialized:
            logger.warning("TensorParallelManager is already initialized; skipping.")
            return

        if self.tp_size <= 1:
            # Single-GPU: no distributed setup needed
            self._initialized = True
            self._backend = backend
            logger.info("TensorParallelManager: tp_size=1, no distributed init needed.")
            return

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but tp_size > 1 was requested.")

        if dist.is_initialized():
            # Process group already initialized (e.g., by torchrun)
            self.tp_rank = dist.get_rank()
            if dist.get_world_size() != self.tp_size:
                logger.warning(
                    f"Existing process group world_size={dist.get_world_size()} "
                    f"does not match tp_size={self.tp_size}. Using existing group."
                )
                self.tp_size = dist.get_world_size()
        else:
            # Initialize process group
            if init_method is None:
                init_method = "env://"

            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=self.tp_size,
                rank=None,  # Let torchrun / env set the rank
            )
            self.tp_rank = dist.get_rank()

        self.device = f"cuda:{self.tp_rank}"
        torch.cuda.set_device(self.device)
        self._initialized = True
        self._backend = backend

        logger.info(
            f"TensorParallelManager initialized: rank={self.tp_rank}, "
            f"world_size={self.tp_size}, backend={backend}"
        )

    def shutdown(self) -> None:
        """Clean up the distributed process group."""
        if self._initialized and self.tp_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
        self._initialized = False

    # ------------------------------------------------------------------
    # Weight sharding
    # ------------------------------------------------------------------

    def shard_weight(self, weight: torch.Tensor, dim: int) -> torch.Tensor:
        """Shard a weight tensor for this rank along the given dimension.

        The weight is split into tp_size chunks along dim, and this rank
        receives the chunk at index tp_rank.

        Args:
            weight: Full weight tensor to shard.
            dim: Dimension along which to split.

        Returns:
            The shard of the weight for this rank.
        """
        if not self._initialized or self.tp_size <= 1:
            return weight

        chunks = weight.chunk(self.tp_size, dim=dim)
        return chunks[self.tp_rank].contiguous()

    # ------------------------------------------------------------------
    # Collective operations
    # ------------------------------------------------------------------

    def allreduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce (sum) across all TP ranks.

        The result is broadcast to all ranks.

        Args:
            tensor: Input tensor to reduce.

        Returns:
            The reduced tensor (same shape, summed across ranks).
        """
        if not self._initialized or self.tp_size <= 1:
            return tensor

        if not dist.is_initialized():
            return tensor

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def allgather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """All-gather across all TP ranks, concatenating along dim.

        Args:
            tensor: Input tensor from this rank.
            dim: Dimension along which to concatenate gathered tensors.

        Returns:
            Concatenated tensor from all ranks.
        """
        if not self._initialized or self.tp_size <= 1:
            return tensor

        if not dist.is_initialized():
            return tensor

        gather_list = [torch.empty_like(tensor) for _ in range(self.tp_size)]
        dist.all_gather(gather_list, tensor)
        return torch.cat(gather_list, dim=dim)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """Whether the process group has been initialized."""
        return self._initialized

    @property
    def world_size(self) -> int:
        """Total number of TP ranks."""
        return self.tp_size

    @property
    def rank(self) -> int:
        """This rank's index in the TP group."""
        return self.tp_rank

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast a tensor from src rank to all ranks.

        Args:
            tensor: Tensor to broadcast (significant only on src rank).
            src: Source rank index.

        Returns:
            The broadcast tensor.
        """
        if not self._initialized or self.tp_size <= 1:
            return tensor

        if not dist.is_initialized():
            return tensor

        dist.broadcast(tensor, src=src)
        return tensor

    def reduce_scatter(
        self, tensor: torch.Tensor, dim: int = 0
    ) -> torch.Tensor:
        """Reduce-scatter: sum across ranks, then scatter the result.

        Args:
            tensor: Input tensor to reduce and scatter.
            dim: Dimension along which to scatter.

        Returns:
            This rank's scattered chunk of the reduced tensor.
        """
        if not self._initialized or self.tp_size <= 1:
            return tensor

        if not dist.is_initialized():
            return tensor

        output = torch.empty_like(tensor.chunk(self.tp_size, dim=dim)[self.tp_rank])
        dist.reduce_scatter_tensor(output, tensor)
        return output

    def barrier(self) -> None:
        """Synchronize all TP ranks."""
        if self._initialized and self.tp_size > 1 and dist.is_initialized():
            dist.barrier()

    def get_device(self) -> torch.device:
        """Return the torch.device for this rank."""
        return torch.device(self.device)
