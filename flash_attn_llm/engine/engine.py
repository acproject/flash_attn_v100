"""LLM inference engine that ties together scheduling, model execution, and sampling.

Provides both a simple synchronous ``generate`` API and an asynchronous
``add_request / step`` API for continuous batching.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import torch

from flash_attn_llm.engine.request import InferenceRequest, RequestStatus
from flash_attn_llm.engine.scheduler import ContinuousBatchingScheduler
from flash_attn_llm.engine.model_runner import ModelRunner


# ======================================================================
# Sampler
# ======================================================================

class Sampler:
    """Token sampler supporting temperature, top-p, top-k, and repetition penalty."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)

    def sample(
        self,
        logits: torch.Tensor,
        request_ids: list[str],
        requests: dict[str, InferenceRequest],
    ) -> list[int]:
        """Sample next tokens from logits, one per request in the batch.

        Args:
            logits: Shape (batch_size, vocab_size).
            request_ids: Request IDs corresponding to each row in logits.
            requests: Mapping from request ID to InferenceRequest (for sampling params).

        Returns:
            List of sampled token IDs (one per request).
        """
        batch_size = logits.shape[0]
        sampled_tokens: list[int] = []

        for i in range(batch_size):
            req = requests[request_ids[i]]
            token_logits = logits[i].clone().float()

            # Apply repetition penalty
            if req.repetition_penalty != 1.0 and req.output_token_ids:
                for token_id in set(req.output_token_ids):
                    if token_id < token_logits.shape[0]:
                        if token_logits[token_id] > 0:
                            token_logits[token_id] /= req.repetition_penalty
                        else:
                            token_logits[token_id] *= req.repetition_penalty

            # Apply temperature
            if req.temperature != 1.0 and req.temperature > 0:
                token_logits = token_logits / req.temperature

            # Apply top-k
            if req.top_k > 0:
                top_k = min(req.top_k, token_logits.shape[0])
                indices_to_remove = token_logits < torch.topk(token_logits, top_k)[0][-1]
                token_logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus sampling)
            if req.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(token_logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > req.top_p
                # Shift right to keep at least one token
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    0, sorted_indices, sorted_indices_to_remove
                )
                token_logits[indices_to_remove] = float("-inf")

            # Sample
            if req.temperature == 0:
                token_id = torch.argmax(token_logits).item()
            else:
                probs = torch.softmax(token_logits, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1).item()

            sampled_tokens.append(token_id)

        return sampled_tokens


# ======================================================================
# KV Cache Manager
# ======================================================================

class KVCacheManager:
    """Manages per-request KV cache storage and retrieval.

    Uses a block-based approach where each request's KV cache is stored
    as a list of (key, value) tensor pairs, one per transformer layer.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.dtype = dtype
        self.device = torch.device(device)

        # Per-request KV cache: request_id -> list of (key, value) per layer
        self._kv_caches: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        # Track current sequence length per request
        self._cache_lens: dict[str, int] = {}

    def allocate(self, request_id: str) -> None:
        """Pre-allocate KV cache tensors for a request."""
        max_blocks = (self.max_seq_len + self.block_size - 1) // self.block_size
        kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(self.num_layers):
            key = torch.zeros(
                max_blocks,
                self.block_size,
                self.num_heads,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            value = torch.zeros_like(key)
            kv.append((key, value))
        self._kv_caches[request_id] = kv
        self._cache_lens[request_id] = 0

    def get_kv_cache(self, request_ids: list[str]) -> Optional[Any]:
        """Return KV cache entries for the given request IDs.

        Returns a list (layers) of tuples (key, value) suitable for
        passing as past_key_values to the model, or None if no caches exist.
        """
        if not request_ids:
            return None

        # For simplicity, return the cache for the first request.
        # In a true batched scenario, caches would be concatenated.
        caches = []
        for rid in request_ids:
            if rid in self._kv_caches:
                caches.append(self._kv_caches[rid])

        if not caches:
            return None

        # Reorganize: list of layers, each with (key, value) batched
        batched: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            keys = [c[layer_idx][0] for c in caches]
            values = [c[layer_idx][1] for c in caches]
            batched.append((torch.stack(keys), torch.stack(values)))

        return batched

    def update_kv_cache(
        self, request_ids: list[str], past_key_values: Any
    ) -> None:
        """Update stored KV caches with new values from the model output."""
        if past_key_values is None:
            return

        for i, rid in enumerate(request_ids):
            if rid not in self._kv_caches:
                continue
            new_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
            for layer_idx in range(self.num_layers):
                k, v = past_key_values[layer_idx]
                # Store the per-request slice
                new_kv.append((k[i], v[i]))
            self._kv_caches[rid] = new_kv

    def get_cache_len(self, request_id: str) -> int:
        """Return current KV cache length for a request."""
        return self._cache_lens.get(request_id, 0)

    def set_cache_len(self, request_id: str, length: int) -> None:
        """Set the current KV cache length for a request."""
        self._cache_lens[request_id] = length

    def release(self, request_id: str) -> None:
        """Release KV cache for a completed request."""
        self._kv_caches.pop(request_id, None)
        self._cache_lens.pop(request_id, None)

    @property
    def num_active(self) -> int:
        """Number of requests with active KV caches."""
        return len(self._kv_caches)


# ======================================================================
# LLM Engine
# ======================================================================

class LLMEngine:
    """High-level LLM inference engine with continuous batching support.

    Ties together the model, tokenizer, scheduler, KV cache manager,
    sampler, and model runner to provide both synchronous and
    asynchronous generation APIs.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        block_size: int = 16,
        tp_size: int = 1,
    ):
        """
        Args:
            model_path: Path or HuggingFace model identifier to load.
            device: Device for inference.
            dtype: Model data type.
            max_batch_size: Maximum number of concurrent requests.
            max_seq_len: Maximum sequence length (prompt + output).
            block_size: KV cache block size.
            tp_size: Tensor parallelism degree (1 = no TP).
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.tp_size = tp_size

        # Load tokenizer and model
        self.tokenizer = self._load_tokenizer(model_path)
        self.model = self._load_model(model_path, device, dtype)

        # Derive model dimensions
        config = getattr(self.model, "config", None)
        num_layers = getattr(config, "num_hidden_layers", 1)
        num_heads = getattr(config, "num_attention_heads", 1)
        head_dim = getattr(config, "hidden_size", 1) // max(num_heads, 1)

        # Create components
        self.kv_cache_manager = KVCacheManager(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )
        self.scheduler = ContinuousBatchingScheduler(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            block_size=block_size,
        )
        self.sampler = Sampler(device=device)
        self.model_runner = ModelRunner(
            model=self.model,
            kv_cache_manager=self.kv_cache_manager,
            device=device,
            dtype=dtype,
        )

        # Request tracking
        self._requests: dict[str, InferenceRequest] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        **kwargs: Any,
    ) -> str:
        """Simple synchronous generate API.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability threshold.
            top_k: Top-k sampling parameter.
            **kwargs: Additional sampling parameters.

        Returns:
            Generated text string.
        """
        request_id = self.add_request(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs,
        )

        # Run steps until the request completes
        while not self._requests[request_id].is_finished:
            self.step()

        return self._requests[request_id].output_text

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 128,
        **kwargs: Any,
    ) -> list[str]:
        """Batch generation for multiple prompts.

        Args:
            prompts: List of input text prompts.
            max_tokens: Maximum number of tokens to generate per prompt.
            **kwargs: Additional sampling parameters.

        Returns:
            List of generated text strings.
        """
        request_ids = []
        for prompt in prompts:
            rid = self.add_request(prompt=prompt, max_tokens=max_tokens, **kwargs)
            request_ids.append(rid)

        # Run steps until all requests complete
        while any(not self._requests[rid].is_finished for rid in request_ids):
            self.step()

        return [self._requests[rid].output_text for rid in request_ids]

    def add_request(
        self,
        prompt: str,
        max_tokens: int = 128,
        stream_callback: Optional[callable] = None,
        **kwargs: Any,
    ) -> str:
        """Add a request to the queue.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum number of tokens to generate.
            stream_callback: Optional callback invoked for each generated token.
            **kwargs: Additional sampling parameters (temperature, top_p, top_k, etc.)

        Returns:
            Request ID string.
        """
        prompt_token_ids = self.tokenizer.encode(prompt)

        request = InferenceRequest(
            prompt_token_ids=prompt_token_ids,
            prompt_text=prompt,
            max_tokens=max_tokens,
            stream_callback=stream_callback,
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            stop_token_ids=kwargs.get("stop_token_ids", []),
        )

        self._requests[request.request_id] = request
        self.scheduler.add_request(request)
        self.kv_cache_manager.allocate(request.request_id)

        return request.request_id

    def step(self) -> list[InferenceRequest]:
        """Run one scheduling step: schedule -> prefill/decode -> sample -> update.

        Returns:
            List of requests that completed during this step.
        """
        prefill_batch, decode_batch = self.scheduler.schedule()
        completed: list[InferenceRequest] = []

        # --- Prefill phase ---
        if prefill_batch:
            # Build input tensor
            max_prompt_len = max(r.num_prompt_tokens for r in prefill_batch)
            batch_size = len(prefill_batch)
            input_ids = torch.zeros(
                batch_size, max_prompt_len, dtype=torch.long, device=self.device
            )

            for i, req in enumerate(prefill_batch):
                prompt_len = req.num_prompt_tokens
                input_ids[i, :prompt_len] = torch.tensor(
                    req.prompt_token_ids, dtype=torch.long
                )

            request_ids = [r.request_id for r in prefill_batch]
            logits = self.model_runner.prefill(input_ids, request_ids)

            # Sample next tokens
            sampled_tokens = self.sampler.sample(logits, request_ids, self._requests)

            # Update requests
            for req, token_id in zip(prefill_batch, sampled_tokens):
                req.output_token_ids.append(token_id)
                self.kv_cache_manager.set_cache_len(
                    req.request_id, req.num_prompt_tokens + 1
                )

                # Check stop conditions
                if self._should_stop(req):
                    req.status = RequestStatus.COMPLETED
                    req.completion_time = time.time()
                    completed.append(req)
                else:
                    # Move to decode phase
                    pass

            # Move non-completed prefill requests to decode
            non_completed_ids = [
                r.request_id for r in prefill_batch if not r.is_finished
            ]
            self.scheduler.move_prefill_to_decode(non_completed_ids)

        # --- Decode phase ---
        if decode_batch:
            # Build decode input
            batch_size = len(decode_batch)
            input_ids = torch.zeros(
                batch_size, 1, dtype=torch.long, device=self.device
            )
            cache_lens: list[int] = []

            for i, req in enumerate(decode_batch):
                input_ids[i, 0] = req.output_token_ids[-1]
                cache_lens.append(
                    self.kv_cache_manager.get_cache_len(req.request_id)
                )

            request_ids = [r.request_id for r in decode_batch]
            logits = self.model_runner.decode(input_ids, request_ids, cache_lens)

            # Sample next tokens
            sampled_tokens = self.sampler.sample(logits, request_ids, self._requests)

            # Update requests
            newly_completed_ids: list[str] = []
            for req, token_id in zip(decode_batch, sampled_tokens):
                req.output_token_ids.append(token_id)
                self.kv_cache_manager.set_cache_len(
                    req.request_id,
                    self.kv_cache_manager.get_cache_len(req.request_id) + 1,
                )

                # Invoke stream callback if present
                if req.stream_callback is not None:
                    try:
                        token_text = self.tokenizer.decode([token_id])
                        req.stream_callback(token_text)
                    except Exception:
                        pass

                # Check stop conditions
                if self._should_stop(req):
                    req.status = RequestStatus.COMPLETED
                    req.completion_time = time.time()
                    completed.append(req)
                    newly_completed_ids.append(req.request_id)

            # Release completed requests
            if newly_completed_ids:
                self.scheduler.update_requests(newly_completed_ids)
                for rid in newly_completed_ids:
                    self.kv_cache_manager.release(rid)

        # Decode output text for completed requests
        for req in completed:
            req.output_text = self.tokenizer.decode(
                req.output_token_ids, skip_special_tokens=True
            )

        return completed

    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests in the engine."""
        return self.scheduler.has_requests()

    def get_request(self, request_id: str) -> Optional[InferenceRequest]:
        """Retrieve a request by ID."""
        return self._requests.get(request_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_stop(self, request: InferenceRequest) -> bool:
        """Check if a request should stop generating."""
        # Max tokens reached
        if request.num_output_tokens >= request.max_tokens:
            return True

        # Stop token encountered
        if request.output_token_ids and request.stop_token_ids:
            if request.output_token_ids[-1] in request.stop_token_ids:
                return True

        # Exceed max sequence length
        if request.total_tokens >= self.max_seq_len:
            return True

        return False

    @staticmethod
    def _load_tokenizer(model_path: str) -> Any:
        """Load tokenizer from model path."""
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from {model_path}: {e}") from e

    @staticmethod
    def _load_model(
        model_path: str, device: str, dtype: torch.dtype
    ) -> torch.nn.Module:
        """Load model from model path."""
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
            )
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e
