"""Token sampler with various decoding strategies for LLM inference.

Supports greedy, temperature, top-k, top-p (nucleus), min-p, repetition /
frequency / presence penalty, and bad-word masking.  All operations are kept
on GPU to avoid CPU round-trips.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    """Parameters that control token sampling behaviour.

    Attributes:
        temperature: Softmax temperature.  ``1.0`` = no change, ``0.0`` =
            greedy (argmax), ``< 1.0`` = sharper, ``> 1.0`` = flatter.
        top_k: Keep only the top-k highest-probability tokens.  ``-1`` disables.
        top_p: Nucleus sampling threshold.  Keep the smallest set of tokens
            whose cumulative probability >= ``top_p``.  ``1.0`` disables.
        min_p: Minimum probability relative to the most probable token.
            Tokens with ``prob < min_p * max_prob`` are filtered.  ``0.0``
            disables.
        repetition_penalty: Multiplicative penalty on the logits of tokens
            that appeared in ``past_tokens``.  ``1.0`` disables.
        frequency_penalty: Additive penalty proportional to how often a token
            appeared in ``past_tokens``.  ``0.0`` disables.
        presence_penalty: Additive penalty if a token appeared at all in
            ``past_tokens``.  ``0.0`` disables.
        max_tokens: Maximum number of tokens to generate (informational —
            not enforced by the sampler itself).
        stop_token_ids: Token IDs that signal end of generation.
        stop_strings: Strings that signal end of generation (handled by the
            caller, not the sampler).
        bad_words_ids: List of token-ID lists whose first token should be
            masked (set to ``-inf``) in the logits.
        seed: Optional RNG seed for reproducibility.
    """

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 128
    stop_token_ids: List[int] = field(default_factory=list)
    stop_strings: List[str] = field(default_factory=list)
    bad_words_ids: List[List[int]] = field(default_factory=list)
    seed: Optional[int] = None


class Sampler:
    """Token sampler with various decoding strategies.

    All tensor operations stay on the specified device (typically CUDA) to
    avoid expensive CPU round-trips during inference.

    Args:
        vocab_size: Size of the model vocabulary.
        device: Device on which to perform sampling (``'cuda'`` or ``'cpu'``).
    """

    def __init__(self, vocab_size: int, device: str = "cuda") -> None:
        self.vocab_size = vocab_size
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        past_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample tokens from logits with the given parameters.

        Args:
            logits: Unnormalised log-probabilities of shape
                ``[B, vocab_size]`` or ``[B, 1, vocab_size]``.
            params: :class:`SamplingParams` controlling the sampling strategy.
            past_tokens: ``[B, seq_len]`` tensor of previously generated
                token IDs, used for repetition / frequency / presence penalty.

        Returns:
            token_ids: ``[B]`` tensor of sampled token IDs on the same device
                as *logits*.
        """
        # Normalise shape to [B, vocab_size]
        if logits.dim() == 3:
            logits = logits.squeeze(1)
        assert logits.dim() == 2, f"Expected 2-D or 3-D logits, got {logits.dim()}-D"

        # Set RNG seed for reproducibility if requested
        if params.seed is not None:
            torch.manual_seed(params.seed)
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed(params.seed)

        # --- Apply penalties before temperature / filtering ---
        # Repetition / frequency / presence penalty (operates on raw logits)
        if past_tokens is not None and (
            params.repetition_penalty != 1.0
            or params.frequency_penalty != 0.0
            or params.presence_penalty != 0.0
        ):
            logits = self._apply_repetition_penalty(
                logits, past_tokens, params.repetition_penalty,
                params.frequency_penalty, params.presence_penalty,
            )

        # Bad-word masking (operates on raw logits)
        if params.bad_words_ids:
            logits = self._apply_bad_words(logits, params.bad_words_ids)

        # --- Temperature ---
        # temperature == 0 → greedy (handled later)
        if params.temperature != 0.0 and params.temperature != 1.0:
            logits = self._apply_temperature(logits, params.temperature)

        # --- Top-k = 1 → greedy shortcut ---
        if params.top_k == 1 or params.temperature == 0.0:
            return self._greedy_sample(logits)

        # --- Top-k filtering ---
        if params.top_k > 0:
            logits = self._apply_top_k(logits, params.top_k)

        # --- Top-p (nucleus) filtering ---
        if params.top_p < 1.0:
            logits = self._apply_top_p(logits, params.top_p)

        # --- Min-p filtering ---
        if params.min_p > 0.0:
            logits = self._apply_min_p(logits, params.min_p)

        # --- Convert to probabilities and sample ---
        probs = F.softmax(logits, dim=-1)
        return self._random_sample(probs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_temperature(
        logits: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """Apply temperature scaling: ``logits /= temperature``."""
        return logits / temperature

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        past_tokens: torch.Tensor,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> torch.Tensor:
        """Apply repetition, frequency, and presence penalties.

        * **Repetition penalty** (multiplicative): for each token that
          appeared in ``past_tokens``, if its logit > 0 divide by the penalty,
          otherwise multiply by the penalty.
        * **Frequency penalty** (additive): subtract ``frequency_penalty ×
          count(token)`` from the logit.
        * **Presence penalty** (additive): subtract ``presence_penalty`` from
          the logit of every token that appeared at least once.

        Args:
            logits: ``[B, vocab_size]``
            past_tokens: ``[B, seq_len]``
            repetition_penalty: Multiplicative penalty (≥ 1.0).
            frequency_penalty: Additive frequency penalty.
            presence_penalty: Additive presence penalty.

        Returns:
            Modified logits of the same shape.
        """
        B, V = logits.shape
        # Clamp past_tokens to valid vocab range to avoid index-out-of-bounds
        clamped = past_tokens.clamp(0, V - 1)

        # One-hot accumulate: [B, V] count of each token
        counts = torch.zeros(B, V, dtype=logits.dtype, device=logits.device)
        counts.scatter_add_(
            1, clamped, torch.ones_like(clamped, dtype=logits.dtype)
        )

        # Presence mask: 1 if token appeared at least once
        presence = (counts > 0).to(logits.dtype)

        # --- Repetition penalty (multiplicative) ---
        if repetition_penalty != 1.0:
            # Build a mask of which logits to penalise
            token_mask = presence.bool()  # [B, V]
            # For positive logits, divide; for negative logits, multiply
            penalised = torch.where(
                logits > 0,
                logits / repetition_penalty,
                logits * repetition_penalty,
            )
            logits = torch.where(token_mask, penalised, logits)

        # --- Frequency penalty (additive) ---
        if frequency_penalty != 0.0:
            logits = logits - frequency_penalty * counts

        # --- Presence penalty (additive) ---
        if presence_penalty != 0.0:
            logits = logits - presence_penalty * presence

        return logits

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Apply top-k filtering.

        Keep only the *k* highest logits; set the rest to ``-inf``.
        """
        if k <= 0 or k >= logits.size(-1):
            return logits

        # [B, k] — the k-th largest value in each row
        kth_values = torch.topk(logits, k, dim=-1).values[:, -1:]
        return torch.where(logits >= kth_values, logits, torch.finfo(logits.dtype).min)

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
        """Apply nucleus (top-p) filtering.

        Keep the smallest set of tokens whose cumulative probability >= *p*
        and set the rest to ``-inf``.
        """
        if p >= 1.0:
            return logits

        # Sort in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # Cumulative probabilities of the sorted tokens
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Mask tokens whose cumulative probability exceeds p (keep the first
        # token that crosses the threshold).
        sorted_mask = cumulative_probs - sorted_probs > p

        # Set masked logits to -inf
        sorted_logits[sorted_mask] = torch.finfo(sorted_logits.dtype).min

        # Unsort
        _, unsorted_indices = torch.sort(sorted_indices, dim=-1)
        logits = sorted_logits.gather(dim=-1, index=unsorted_indices)
        return logits

    @staticmethod
    def _apply_min_p(logits: torch.Tensor, p: float) -> torch.Tensor:
        """Apply min-p filtering.

        Remove tokens whose probability is less than ``p × max_prob``.
        """
        if p <= 0.0:
            return logits

        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        threshold = p * max_prob
        mask = probs < threshold
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        return logits

    @staticmethod
    def _apply_bad_words(
        logits: torch.Tensor, bad_words_ids: List[List[int]]
    ) -> torch.Tensor:
        """Mask bad words by setting their logits to ``-inf``.

        Each entry in *bad_words_ids* is a list of token IDs.  The first
        token of each entry is masked in the logits.
        """
        V = logits.size(-1)
        for word_ids in bad_words_ids:
            if not word_ids:
                continue
            first_token = word_ids[0]
            if 0 <= first_token < V:
                logits[:, first_token] = torch.finfo(logits.dtype).min
        return logits

    @staticmethod
    def _greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        """Greedy decoding: return ``argmax(logits, dim=-1)``."""
        return logits.argmax(dim=-1)

    def _random_sample(self, probs: torch.Tensor) -> torch.Tensor:
        """Random sampling from a probability distribution.

        Uses :func:`torch.multinomial` which runs on the same device as
        *probs*.
        """
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
