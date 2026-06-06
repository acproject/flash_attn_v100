"""Tokenizer module for LLM inference.

Provides a unified tokenizer interface wrapping HuggingFace transformers,
with support for streaming decoding and chat templates.
"""

from __future__ import annotations

from typing import Optional

from transformers import AutoTokenizer


class Tokenizer:
    """Unified tokenizer wrapper for LLM inference.

    Wraps HuggingFace AutoTokenizer as the backend, supporting SentencePiece,
    BPE, and HF tokenizer JSON formats. Provides encoding, decoding, batch
    processing, chat template rendering, and streaming decode capabilities.

    Args:
        model_path: Path or identifier to load the tokenizer from. Can be a
            local directory or a HuggingFace model ID (e.g. "meta-llama/Llama-2-7b-hf").
    """

    def __init__(self, model_path: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self._vocab: dict[str, int] = dict(self._tokenizer.get_vocab())

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode a single text string into token IDs.

        Args:
            text: The input text to tokenize.
            add_special_tokens: Whether to prepend BOS / append EOS tokens.

        Returns:
            A list of integer token IDs.
        """
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def batch_encode(
        self, texts: list[str], add_special_tokens: bool = True
    ) -> list[list[int]]:
        """Encode a batch of text strings into lists of token IDs.

        Args:
            texts: List of input strings.
            add_special_tokens: Whether to add special tokens to each sequence.

        Returns:
            A list of token-ID lists, one per input string.
        """
        encoded = self._tokenizer.batch_encode_plus(
            texts, add_special_tokens=add_special_tokens
        )
        return encoded["input_ids"]

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back into a string.

        Args:
            token_ids: The token IDs to decode.
            skip_special_tokens: Whether to skip special tokens in the output.

        Returns:
            The decoded string.
        """
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def streaming_decode(self, token_ids: list[int]) -> str:
        """Decode token IDs incrementally for streaming output.

        This method decodes only the newly added tokens since the last call,
        making it suitable for token-by-token streaming scenarios. It handles
        partial UTF-8 character boundaries correctly.

        Args:
            token_ids: The full list of token IDs accumulated so far.

        Returns:
            The decoded text for the newly added tokens (the last token).
        """
        if not token_ids:
            return ""
        # Decode only the last token to get incremental text
        new_text = self._tokenizer.decode([token_ids[-1]], skip_special_tokens=False)
        # Decode the full sequence without special tokens to get clean output
        full_text = self._tokenizer.decode(token_ids, skip_special_tokens=True)
        # Return the incremental part
        prev_text = self._tokenizer.decode(token_ids[:-1], skip_special_tokens=True)
        return full_text[len(prev_text):]

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    def apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
    ) -> list[int]:
        """Apply the model's chat template to a list of messages.

        Args:
            messages: A list of message dicts, each with "role" and "content" keys.
                Example: [{"role": "user", "content": "Hello!"}]
            add_generation_prompt: Whether to append the assistant's generation
                prompt at the end.

        Returns:
            A list of token IDs representing the formatted conversation.
        """
        text = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
        return self._tokenizer.encode(text, add_special_tokens=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size (including added tokens)."""
        return self._tokenizer.vocab_size

    @property
    def bos_token_id(self) -> Optional[int]:
        """Return the beginning-of-sequence token ID, or None if not defined."""
        return self._tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        """Return the end-of-sequence token ID, or None if not defined."""
        return self._tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> Optional[int]:
        """Return the padding token ID, or None if not defined."""
        return self._tokenizer.pad_token_id

    @property
    def eos_token(self) -> Optional[str]:
        """Return the end-of-sequence token string, or None if not defined."""
        return self._tokenizer.eos_token

    @property
    def bos_token(self) -> Optional[str]:
        """Return the beginning-of-sequence token string, or None if not defined."""
        return self._tokenizer.bos_token

    @property
    def pad_token(self) -> Optional[str]:
        """Return the padding token string, or None if not defined."""
        return self._tokenizer.pad_token

    @property
    def unk_token(self) -> Optional[str]:
        """Return the unknown token string, or None if not defined."""
        return self._tokenizer.unk_token

    @property
    def tokenizer(self):
        """Return the underlying HuggingFace tokenizer for advanced usage."""
        return self._tokenizer


class TokenStreamer:
    """Streaming token decoder for real-time output.

    Handles incremental decoding of token IDs into text, correctly buffering
    partial UTF-8 characters that may be split across token boundaries.

    Example::

        streamer = TokenStreamer(tokenizer)
        for token_id in generated_ids:
            text = streamer.add_token(token_id)
            if text:
                print(text, end="", flush=True)
        remaining = streamer.finish()
        if remaining:
            print(remaining, end="")
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._token_ids: list[int] = []
        self._decoded_offset: int = 0  # number of tokens already decoded
        self._byte_buffer: bytearray = bytearray()

    def add_token(self, token_id: int) -> Optional[str]:
        """Add a new token ID and return decoded text if available.

        Decodes incrementally, buffering incomplete UTF-8 characters that
        span token boundaries.

        Args:
            token_id: The newly generated token ID.

        Returns:
            Decoded text string if a complete character is available, or None
            if the current bytes form an incomplete UTF-8 sequence.
        """
        self._token_ids.append(token_id)

        # Decode all tokens from the last decoded offset to get incremental text
        new_ids = self._token_ids[self._decoded_offset:]
        if not new_ids:
            return None

        raw_text = self._tokenizer._tokenizer.decode(new_ids)

        # Append new bytes to the buffer
        self._byte_buffer.extend(raw_text.encode("utf-8"))
        self._decoded_offset = len(self._token_ids)

        # Try to decode as much of the buffer as possible
        text, remaining = self._decode_buffer(self._byte_buffer)
        self._byte_buffer = remaining
        return text if text else None

    def finish(self) -> str:
        """Flush any remaining buffered bytes and return the final text.

        Should be called after the last token has been added to ensure no
        partial characters are lost.

        Returns:
            Any remaining decoded text from the buffer.
        """
        if not self._byte_buffer:
            return ""
        # Force-decode remaining bytes using errors='replace' for safety
        text = self._byte_buffer.decode("utf-8", errors="replace")
        self._byte_buffer = bytearray()
        return text

    def reset(self) -> None:
        """Reset the streamer state for reuse."""
        self._token_ids.clear()
        self._decoded_offset = 0
        self._byte_buffer = bytearray()

    @staticmethod
    def _decode_buffer(buf: bytearray) -> tuple[Optional[str], bytearray]:
        """Attempt to decode the buffer, returning complete text and leftover bytes.

        Finds the longest valid UTF-8 prefix of *buf* and returns the decoded
        string along with any trailing bytes that form an incomplete character.
        """
        if not buf:
            return None, bytearray()

        # Try decoding the entire buffer first
        try:
            text = buf.decode("utf-8")
            return text, bytearray()
        except UnicodeDecodeError:
            pass

        # Find the last valid UTF-8 boundary by scanning backwards
        # A UTF-8 character is at most 4 bytes; check up to 3 trailing bytes
        for trim in range(1, min(4, len(buf)) + 1):
            try:
                text = buf[:-trim].decode("utf-8")
                return text, bytearray(buf[-trim:])
            except UnicodeDecodeError:
                continue

        # If nothing could be decoded, keep the whole buffer
        return None, bytearray(buf)
