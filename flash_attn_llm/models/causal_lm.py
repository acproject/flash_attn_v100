"""Causal Language Model for flash_attn_llm inference library.

Implements the full transformer model with embedding, decoder layers,
final normalization, and language model head. Supports both prefill
and decode modes with KV cache management for autoregressive generation.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig
from .decoder_layer import TransformerDecoderLayer
from .norm import RMSNorm


class KVCacheManager:
    """Manages KV caches for all layers during autoregressive generation.

    Allocates and manages pre-allocated KV cache tensors for each decoder
    layer, supporting incremental updates during decode.

    Args:
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV attention heads.
        head_dim: Dimension of each attention head.
        max_batch_size: Maximum batch size.
        max_seq_len: Maximum sequence length.
        dtype: Data type for cache tensors.
        device: Device for cache tensors.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_batch_size: int = 1,
        max_seq_len: int = 4096,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device

        # Pre-allocate KV caches for all layers
        # Each cache entry: [B, H_KV, max_seq_len, D]
        self.k_caches: List[torch.Tensor] = []
        self.v_caches: List[torch.Tensor] = []
        for _ in range(num_layers):
            self.k_caches.append(
                torch.zeros(
                    max_batch_size, num_kv_heads, max_seq_len, head_dim,
                    dtype=dtype, device=device,
                )
            )
            self.v_caches.append(
                torch.zeros(
                    max_batch_size, num_kv_heads, max_seq_len, head_dim,
                    dtype=dtype, device=device,
                )
            )

        # Track current cache length per batch item
        self.cache_lens = torch.zeros(max_batch_size, dtype=torch.int32, device=device)

    def get_kv_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the KV cache tensors for a specific layer.

        Returns the cache tensors sliced to the current cache length.

        Args:
            layer_idx: Index of the decoder layer.

        Returns:
            Tuple of (k_cache, v_cache) each of shape [B, H_KV, cache_len, D].
        """
        # Return full cache; the attention module handles slicing via cache_len
        return self.k_caches[layer_idx], self.v_caches[layer_idx]

    def update_cache_len(self, seq_len: int, batch_indices: Optional[torch.Tensor] = None):
        """Update the cache length after processing tokens.

        Args:
            seq_len: Number of new tokens added to the cache.
            batch_indices: Optional indices of batch items to update.
                If None, updates all items.
        """
        if batch_indices is not None:
            self.cache_lens[batch_indices] += seq_len
        else:
            self.cache_lens += seq_len

    def reset(self):
        """Reset all cache lengths to zero."""
        self.cache_lens.zero_()

    def get_cache_len(self, batch_idx: int = 0) -> int:
        """Get the current cache length for a batch item.

        Args:
            batch_idx: Batch item index.

        Returns:
            Current cache length.
        """
        return int(self.cache_lens[batch_idx].item())


class CausalLM(nn.Module):
    """Causal Language Model with flash_attn_v100 backend.

    Full transformer model consisting of token embeddings, N decoder layers,
    final RMSNorm, and an LM head for vocabulary projection.

    Supports:
    - Prefill mode: Process entire prompt at once
    - Decode mode: Autoregressive single-token generation with KV cache
    - generate(): Convenience method for text generation

    Args:
        config: ModelConfig instance with model hyperparameters.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward_prefill(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        return_kv_cache: bool = False,
    ) -> torch.Tensor:
        """Prefill forward pass for processing a full prompt.

        Args:
            input_ids: Token IDs [B, seq_len].
            position_ids: Optional position indices [B, seq_len].
            return_kv_cache: If True, also return KV caches for each layer.

        Returns:
            If return_kv_cache is False:
                Logits tensor [B, seq_len, vocab_size].
            If return_kv_cache is True:
                Tuple of (logits, kv_caches) where kv_caches is a list of
                (k_cache, v_cache) tuples, one per layer.
        """
        hidden_states = self.embed_tokens(input_ids)

        kv_caches = [] if return_kv_cache else None
        for layer in self.layers:
            hidden_states, kv = layer.forward_prefill(hidden_states, position_ids)
            if return_kv_cache:
                kv_caches.append(kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if return_kv_cache:
            return logits, kv_caches
        return logits

    def forward_decode(
        self,
        input_ids: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        cache_len: int,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Decode forward pass for single-token generation.

        Args:
            input_ids: Token IDs [B, 1].
            kv_caches: List of (k_cache, v_cache) tuples, one per layer.
            cache_len: Current length of the KV cache.
            position_ids: Optional position indices [B, 1].

        Returns:
            Tuple of:
                - Logits tensor [B, 1, vocab_size]
                - Updated KV caches list
        """
        hidden_states = self.embed_tokens(input_ids)

        updated_kv_caches = []
        for i, layer in enumerate(self.layers):
            hidden_states, updated_kv = layer.forward_decode(
                hidden_states, kv_caches[i], cache_len, position_ids
            )
            updated_kv_caches.append(updated_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, updated_kv_caches

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        cache_len: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Unified forward pass dispatching to prefill or decode.

        Args:
            input_ids: Token IDs [B, seq_len] (prefill) or [B, 1] (decode).
            kv_caches: Optional list of KV cache tuples for decode mode.
            cache_len: Current KV cache length (decode only).
            position_ids: Optional position indices.

        Returns:
            Tuple of:
                - Logits [B, seq_len, vocab_size]
                - Updated KV caches (None for prefill, list for decode)
        """
        if kv_caches is None:
            logits, _ = self.forward_prefill(input_ids, position_ids, return_kv_cache=True)
            return logits, None
        else:
            return self.forward_decode(input_ids, kv_caches, cache_len, position_ids)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive text generation.

        Performs prefill on the input prompt, then generates tokens
        one at a time using the decode path with KV caching.

        Args:
            input_ids: Prompt token IDs [B, seq_len].
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature. 0 for greedy decoding.
            top_k: If set, only sample from the top-k logits.
            top_p: Nucleus sampling threshold.
            eos_token_id: If set, stop generation when this token is produced.

        Returns:
            Generated token IDs [B, seq_len + num_generated].
        """
        device = input_ids.device
        B, prompt_len = input_ids.shape

        # Step 1: Prefill with KV cache
        position_ids = torch.arange(prompt_len, device=device).unsqueeze(0).expand(B, -1)
        logits, kv_caches = self.forward_prefill(input_ids, position_ids, return_kv_cache=True)

        # Sample first new token from last position logits
        next_token = self._sample_token(logits[:, -1, :], temperature, top_k, top_p)
        generated = [next_token]

        # Step 2: Autoregressive decode
        cache_len = prompt_len
        current_token = next_token  # [B, 1]

        for _ in range(max_new_tokens - 1):
            position_id = torch.full(
                (B, 1), cache_len, dtype=torch.long, device=device
            )
            logits, kv_caches = self.forward_decode(
                current_token, kv_caches, cache_len, position_id
            )
            cache_len += 1

            next_token = self._sample_token(logits[:, -1, :], temperature, top_k, top_p)
            generated.append(next_token)
            current_token = next_token

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        generated_tokens = torch.cat(generated, dim=1)
        return torch.cat([input_ids, generated_tokens], dim=1)

    def _build_initial_kv_cache(
        self, input_ids: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Build initial KV caches by processing the prompt token-by-token.

        This is a simplified approach for correctness. A production system
        would modify the prefill path to return KV caches directly.

        Args:
            input_ids: Prompt token IDs [B, seq_len].

        Returns:
            List of (k_cache, v_cache) tuples, one per layer.
        """
        B, seq_len = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        # Initialize empty KV caches
        kv_caches = []
        for layer in self.layers:
            k_cache = torch.zeros(
                B, layer.self_attn.num_kv_heads, 0, layer.self_attn.head_dim,
                dtype=dtype, device=device,
            )
            v_cache = torch.zeros_like(k_cache)
            kv_caches.append((k_cache, v_cache))

        # Process each prompt token through decode to build KV cache
        for pos in range(seq_len):
            token = input_ids[:, pos:pos + 1]  # [B, 1]
            position_id = torch.full(
                (B, 1), pos, dtype=torch.long, device=device
            )
            _, kv_caches = self.forward_decode(token, kv_caches, pos, position_id)

        return kv_caches

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Sample a token from logits with temperature, top-k, and top-p.

        Args:
            logits: Logits tensor [B, vocab_size].
            temperature: Sampling temperature. 0 means greedy.
            top_k: If set, only consider top-k logits.
            top_p: Nucleus sampling threshold.

        Returns:
            Sampled token IDs [B, 1].
        """
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / temperature

        # Top-k filtering
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @classmethod
    def from_config(cls, config: ModelConfig) -> "CausalLM":
        """Create a CausalLM instance from a ModelConfig.

        Args:
            config: ModelConfig instance.

        Returns:
            Initialized CausalLM model.
        """
        return cls(config)
