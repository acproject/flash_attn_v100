"""Model runner that wraps a CausalLM for efficient batched inference.

Handles prefill and decode forward passes, KV cache management during
the forward pass, and input preparation for the model.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


class ModelRunner:
    """Wraps a CausalLM model for efficient batched inference.

    Responsible for:
    - Executing prefill (prompt processing) forward passes
    - Executing decode (token-by-token) forward passes
    - Managing KV cache interactions during forward passes
    - Preparing input tensors for the model
    """

    def __init__(
        self,
        model: torch.nn.Module,
        kv_cache_manager: Any,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            model: The causal language model (e.g., HuggingFace PreTrainedModel).
            kv_cache_manager: Manager responsible for KV cache allocation and lookup.
            device: Device to run inference on.
            dtype: Data type for inference.
        """
        self.model = model
        self.kv_cache_manager = kv_cache_manager
        self.device = torch.device(device)
        self.dtype = dtype

    def prefill(
        self,
        token_ids: torch.Tensor,
        request_ids: list[str],
    ) -> torch.Tensor:
        """Run prefill forward pass for a batch of requests.

        Processes the full prompt tokens for each request, populates the KV
        cache, and returns logits for the last token position.

        Args:
            token_ids: Prompt token IDs of shape (batch_size, max_prompt_len).
                       Shorter prompts should be left-padded or use attention masks.
            request_ids: Corresponding request IDs for KV cache tracking.

        Returns:
            Logits tensor of shape (batch_size, vocab_size) for the next-token
            prediction at the end of each prompt.
        """
        token_ids = token_ids.to(self.device)

        # Build position ids (cumulative lengths for each request)
        position_ids = self._build_position_ids_prefill(token_ids, request_ids)

        # Retrieve past KV cache for these requests (should be empty for prefill)
        past_key_values = self.kv_cache_manager.get_kv_cache(request_ids)

        with torch.no_grad():
            outputs = self.model(
                input_ids=token_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

        # Update KV cache with new key/value states
        self.kv_cache_manager.update_kv_cache(request_ids, outputs.past_key_values)

        # Return logits for the last token position of each sequence
        # outputs.logits shape: (batch_size, seq_len, vocab_size)
        last_token_logits = outputs.logits[:, -1, :]
        return last_token_logits

    def decode(
        self,
        token_ids: torch.Tensor,
        request_ids: list[str],
        cache_lens: list[int],
    ) -> torch.Tensor:
        """Run decode forward pass for a batch of requests.

        Each request contributes exactly one new token. The KV cache is
        updated in place.

        Args:
            token_ids: Newly generated token IDs of shape (batch_size, 1).
            request_ids: Corresponding request IDs for KV cache lookup.
            cache_lens: Current cache lengths for each request (i.e., number
                        of tokens already in the KV cache before this step).

        Returns:
            Logits tensor of shape (batch_size, vocab_size) for the next-token
            prediction.
        """
        token_ids = token_ids.to(self.device)

        # Position ids = current cache length (0-indexed position of the new token)
        position_ids = torch.tensor(
            cache_lens, dtype=torch.long, device=self.device
        ).unsqueeze(1)

        # Retrieve existing KV cache
        past_key_values = self.kv_cache_manager.get_kv_cache(request_ids)

        with torch.no_grad():
            outputs = self.model(
                input_ids=token_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

        # Update KV cache
        self.kv_cache_manager.update_kv_cache(request_ids, outputs.past_key_values)

        # Logits shape: (batch_size, 1, vocab_size) -> squeeze
        return outputs.logits.squeeze(1)

    def prepare_inputs(self, batch: Any) -> dict[str, torch.Tensor]:
        """Prepare input tensors from a scheduled batch for model forward.

        Args:
            batch: A scheduled batch object containing requests and metadata.

        Returns:
            Dictionary with keys:
                - "input_ids": Token IDs tensor.
                - "position_ids": Position IDs tensor.
                - "attention_mask": Optional attention mask.
        """
        if batch is None:
            return {}

        inputs: dict[str, torch.Tensor] = {}

        if hasattr(batch, "prefill_batch") and batch.prefill_batch:
            inputs["prefill_input_ids"] = self._prepare_prefill_inputs(
                batch.prefill_batch
            )

        if hasattr(batch, "decode_batch") and batch.decode_batch:
            inputs["decode_input_ids"] = self._prepare_decode_inputs(
                batch.decode_batch
            )

        return inputs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_position_ids_prefill(
        self,
        token_ids: torch.Tensor,
        request_ids: list[str],
    ) -> torch.Tensor:
        """Build position IDs for prefill, accounting for padding.

        For left-padded sequences, position ids should start from 0 for
        each request's actual tokens (not the padding positions).
        """
        batch_size, seq_len = token_ids.shape
        # Simple case: no padding, positions are 0..seq_len-1 for each request
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        return position_ids

    def _prepare_prefill_inputs(
        self, requests: list[Any]
    ) -> dict[str, torch.Tensor]:
        """Build prefill input tensors from a list of requests."""
        max_prompt_len = max(r.num_prompt_tokens for r in requests)
        batch_size = len(requests)

        input_ids = torch.zeros(
            batch_size, max_prompt_len, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            batch_size, max_prompt_len, dtype=torch.long, device=self.device
        )

        for i, req in enumerate(requests):
            prompt_len = req.num_prompt_tokens
            # Left-pad
            input_ids[i, max_prompt_len - prompt_len:] = torch.tensor(
                req.prompt_token_ids, dtype=torch.long
            )
            attention_mask[i, max_prompt_len - prompt_len:] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _prepare_decode_inputs(
        self, requests: list[Any]
    ) -> dict[str, torch.Tensor]:
        """Build decode input tensors from a list of requests."""
        batch_size = len(requests)
        input_ids = torch.zeros(
            batch_size, 1, dtype=torch.long, device=self.device
        )

        for i, req in enumerate(requests):
            input_ids[i, 0] = req.output_token_ids[-1]

        return {"input_ids": input_ids}
