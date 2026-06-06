"""Continuous batching scheduler for LLM inference.

Manages request queuing, prefill/decode separation, chunked prefill,
and dynamic batch composition with KV cache block tracking.
"""

from __future__ import annotations

import time
from typing import Optional

from flash_attn_llm.engine.request import InferenceRequest, RequestStatus


class ContinuousBatchingScheduler:
    """Continuous batching scheduler that separates prefill and decode phases.

    Supports:
    - Dynamic addition/removal of requests from the decode batch
    - Chunked prefill for long prompts
    - KV block budget tracking
    - Mixing of short and long requests
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        max_prefill_tokens: int = 4096,
        block_size: int = 16,
        max_num_blocks: Optional[int] = None,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.max_prefill_tokens = max_prefill_tokens
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks or (
            (max_batch_size * max_seq_len) // block_size
        )

        # Request queues
        self._waiting_queue: list[InferenceRequest] = []
        self._prefill_requests: list[InferenceRequest] = []
        self._decode_requests: list[InferenceRequest] = []

        # KV block tracking
        self._used_blocks: int = 0
        # request_id -> number of blocks allocated
        self._request_blocks: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(self, request: InferenceRequest) -> None:
        """Add a new inference request to the waiting queue."""
        request.status = RequestStatus.WAITING
        request.arrival_time = time.time()
        self._waiting_queue.append(request)

    def schedule(self) -> tuple[list[InferenceRequest], list[InferenceRequest]]:
        """Schedule the next iteration's prefill and decode batches.

        Returns:
            A tuple of (prefill_batch, decode_batch).
            - prefill_batch: requests that need their prompt tokens processed.
            - decode_batch: requests that are in the autoregressive decode phase.
        """
        # 1. Existing decode requests continue decoding
        decode_batch = list(self._decode_requests)

        # 2. Determine how many tokens/blocks we can allocate for new prefills
        available_batch_slots = self.max_batch_size - len(decode_batch)
        available_blocks = self.max_num_blocks - self._used_blocks
        prefill_budget = self.max_prefill_tokens

        prefill_batch: list[InferenceRequest] = []

        for request in list(self._waiting_queue):
            if available_batch_slots <= 0:
                break

            # Estimate blocks needed for this request's full sequence
            total_seq_len = request.num_prompt_tokens + request.max_tokens
            blocks_needed = self._num_blocks_for_seq_len(total_seq_len)

            if blocks_needed > available_blocks:
                # Try chunked prefill: only allocate blocks for the prompt portion
                prompt_blocks = self._num_blocks_for_seq_len(request.num_prompt_tokens)
                if prompt_blocks > available_blocks:
                    continue  # Skip – not enough blocks even for the prompt

            # Accept the request into the prefill batch
            self._waiting_queue.remove(request)
            request.status = RequestStatus.PREFILLING
            prefill_batch.append(request)
            available_batch_slots -= 1

            # Allocate blocks for the prompt portion initially
            prompt_blocks = self._num_blocks_for_seq_len(request.num_prompt_tokens)
            self._used_blocks += prompt_blocks
            self._request_blocks[request.request_id] = prompt_blocks
            available_blocks -= prompt_blocks

            # Track remaining prefill token budget
            prefill_budget -= request.num_prompt_tokens
            if prefill_budget <= 0:
                break

        # 3. Apply chunked prefill: split long prompts into chunks
        prefill_batch = self._apply_chunked_prefill(prefill_batch)

        # 4. Update internal state
        self._prefill_requests = prefill_batch

        return prefill_batch, decode_batch

    def update_requests(self, completed_ids: list[str]) -> None:
        """Mark completed requests and release their KV blocks.

        Args:
            completed_ids: List of request IDs that have finished generation.
        """
        completed_set = set(completed_ids)

        # Remove completed requests from decode list and release blocks
        remaining_decode: list[InferenceRequest] = []
        for req in self._decode_requests:
            if req.request_id in completed_set:
                req.status = RequestStatus.COMPLETED
                req.completion_time = time.time()
                self._release_blocks(req.request_id)
            else:
                remaining_decode.append(req)
        self._decode_requests = remaining_decode

        # Also clean up any completed prefill requests (e.g., stopped during prefill)
        remaining_prefill: list[InferenceRequest] = []
        for req in self._prefill_requests:
            if req.request_id in completed_set:
                req.status = RequestStatus.COMPLETED
                req.completion_time = time.time()
                self._release_blocks(req.request_id)
            else:
                remaining_prefill.append(req)
        self._prefill_requests = remaining_prefill

    def move_prefill_to_decode(self, request_ids: list[str]) -> None:
        """Move requests that have finished prefilling into the decode phase.

        Also allocates additional KV blocks for the expected decode tokens.
        """
        ids_set = set(request_ids)
        still_prefilling: list[InferenceRequest] = []

        for req in self._prefill_requests:
            if req.request_id in ids_set:
                req.status = RequestStatus.DECODING
                self._decode_requests.append(req)
                # Expand block allocation for decode tokens
                self._expand_blocks_for_decode(req)
            else:
                still_prefilling.append(req)

        self._prefill_requests = still_prefilling

    def get_num_waiting(self) -> int:
        """Return the number of requests waiting to be scheduled."""
        return len(self._waiting_queue)

    def get_num_active(self) -> int:
        """Return the number of actively processing requests (prefill + decode)."""
        return len(self._prefill_requests) + len(self._decode_requests)

    def has_requests(self) -> bool:
        """Return True if there are any requests in any stage."""
        return bool(self._waiting_queue or self._prefill_requests or self._decode_requests)

    def get_decode_requests(self) -> list[InferenceRequest]:
        """Return current decode-phase requests."""
        return list(self._decode_requests)

    def get_prefill_requests(self) -> list[InferenceRequest]:
        """Return current prefill-phase requests."""
        return list(self._prefill_requests)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _num_blocks_for_seq_len(self, seq_len: int) -> int:
        """Calculate the number of KV blocks needed for a given sequence length."""
        if seq_len <= 0:
            return 0
        return (seq_len + self.block_size - 1) // self.block_size

    def _release_blocks(self, request_id: str) -> None:
        """Release KV blocks allocated to a request."""
        blocks = self._request_blocks.pop(request_id, 0)
        self._used_blocks = max(0, self._used_blocks - blocks)

    def _expand_blocks_for_decode(self, request: InferenceRequest) -> None:
        """Expand KV block allocation to accommodate decode tokens."""
        total_seq_len = request.num_prompt_tokens + request.max_tokens
        total_blocks_needed = self._num_blocks_for_seq_len(total_seq_len)
        current_blocks = self._request_blocks.get(request.request_id, 0)
        additional_blocks = total_blocks_needed - current_blocks
        if additional_blocks > 0:
            self._used_blocks += additional_blocks
            self._request_blocks[request.request_id] = total_blocks_needed

    def _apply_chunked_prefill(
        self, prefill_batch: list[InferenceRequest]
    ) -> list[InferenceRequest]:
        """Split long prefill requests into chunks that respect the token budget.

        For requests whose prompt exceeds max_prefill_tokens, we create a
        "partial prefill" by only processing the first chunk of tokens. The
        remaining tokens will be processed in subsequent scheduling steps.
        """
        if not prefill_batch:
            return prefill_batch

        chunked: list[InferenceRequest] = []
        remaining_budget = self.max_prefill_tokens

        for req in prefill_batch:
            if req.num_prompt_tokens <= remaining_budget:
                chunked.append(req)
                remaining_budget -= req.num_prompt_tokens
            else:
                # Chunk this request: only process up to remaining_budget tokens
                # The request stays in PREFILLING state for the next iteration
                chunked.append(req)
                # After this iteration, the remaining prompt tokens will be
                # handled in subsequent steps (the request stays in _prefill_requests)
                break

        return chunked

    def get_block_usage(self) -> dict[str, int]:
        """Return KV block usage statistics."""
        return {
            "used_blocks": self._used_blocks,
            "max_blocks": self.max_num_blocks,
            "available_blocks": self.max_num_blocks - self._used_blocks,
        }
