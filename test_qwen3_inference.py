"""End-to-end test: load Qwen3 0.6B model and run inference.

Usage:
    /home/acproject/miniconda3/envs/lyra2/bin/python test_qwen3_inference.py
"""

import sys
import time
import torch

# Add project root to path
sys.path.insert(0, ".")

from flash_attn_llm.models.config import Qwen3Config
from flash_attn_llm.models.causal_lm import CausalLM
from flash_attn_llm.weights.loader import WeightLoader
from flash_attn_llm.tokenizer.tokenizer import Tokenizer


def main():
    print("=" * 60)
    print("Qwen3 0.6B End-to-End Inference Test")
    print("=" * 60)

    # 1. Create model config for Qwen3 0.6B
    config = Qwen3Config()
    print(f"\n[1] Model config:")
    print(f"    model_type: {config.model_type}")
    print(f"    vocab_size: {config.vocab_size}")
    print(f"    hidden_size: {config.hidden_size}")
    print(f"    num_layers: {config.num_hidden_layers}")
    print(f"    num_heads: {config.num_attention_heads}")
    print(f"    num_kv_heads: {config.num_key_value_heads}")
    print(f"    head_dim: {config.head_dim}")
    print(f"    intermediate_size: {config.intermediate_size}")
    print(f"    max_position_embeddings: {config.max_position_embeddings}")
    print(f"    rope_theta: {config.rope_theta}")
    print(f"    attention_qk_norm: {config.attention_qk_norm}")
    print(f"    tie_word_embeddings: {config.tie_word_embeddings}")

    # 2. Create model
    print(f"\n[2] Creating model...")
    t0 = time.time()
    model = CausalLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Total parameters: {total_params:,}")
    print(f"    Model created in {time.time() - t0:.2f}s")

    # 3. Load weights
    model_path = "/home/acproject/ComfyUI/models/text_encoders"
    print(f"\n[3] Loading weights from {model_path}...")
    t0 = time.time()
    loader = WeightLoader(model_type="qwen3")
    stats = loader.load_weights(model, model_path, device="cuda", dtype=torch.float16)
    print(f"    Loaded {stats['num_loaded']}/{stats['num_total']} weights")
    print(f"    Skipped: {stats['num_skipped']}, Sharded: {stats['num_sharded']}")
    print(f"    Load time: {stats['load_time']:.2f}s")

    # 4. Move model to GPU
    model = model.cuda().half().eval()
    print(f"\n[4] Model moved to CUDA, dtype=fp16")

    # 5. Load tokenizer
    print(f"\n[5] Loading tokenizer...")
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(
            "./qwen3_tokenizer", trust_remote_code=True, local_files_only=True
        )
        
        class TokenizerWrapper:
            def __init__(self, hf_tok):
                self._tok = hf_tok
                self._vocab_size = hf_tok.vocab_size
                self.bos_token_id = hf_tok.bos_token_id
                self.eos_token_id = hf_tok.eos_token_id
            
            def encode(self, text, add_special_tokens=True):
                return self._tok.encode(text, add_special_tokens=add_special_tokens)
            
            def decode(self, token_ids, skip_special_tokens=True):
                return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)
            
            @property
            def vocab_size(self):
                return self._vocab_size
        
        tokenizer = TokenizerWrapper(hf_tokenizer)
        print(f"    Tokenizer loaded, vocab_size={tokenizer.vocab_size}")
        print(f"    BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}")
    except Exception as e:
        print(f"    Failed to load tokenizer: {e}")
        print(f"    Using manual token IDs for testing")
        tokenizer = None

    # 6. Run inference - prefill test
    print(f"\n[6] Running prefill test...")
    if tokenizer:
        prompt = "Hello, how are you?"
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device="cuda")
    else:
        # Use some token IDs directly
        prompt = "<manual tokens>"
        input_ids = torch.tensor([[151644, 872, 198]], dtype=torch.long, device="cuda")

    print(f"    Prompt: '{prompt}'")
    print(f"    Input IDs shape: {input_ids.shape}")

    with torch.no_grad():
        t0 = time.time()
        logits, kv_caches = model.forward_prefill(input_ids, return_kv_cache=True)
        prefill_time = time.time() - t0

    print(f"    Prefill logits shape: {logits.shape}")
    print(f"    KV caches: {len(kv_caches)} layers")
    if kv_caches:
        k0, v0 = kv_caches[0]
        print(f"    Layer 0 K cache: {k0.shape}, V cache: {v0.shape}")
    print(f"    Prefill time: {prefill_time:.3f}s")

    # 7. Run decode steps
    print(f"\n[7] Running decode steps...")
    # Sample first token
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    print(f"    First generated token: {next_token.item()}")

    cache_len = input_ids.shape[1]
    generated_tokens = [next_token]

    with torch.no_grad():
        for step in range(20):
            position_id = torch.full((1, 1), cache_len, dtype=torch.long, device="cuda")
            logits, kv_caches = model.forward_decode(
                next_token, kv_caches, cache_len, position_id
            )
            cache_len += 1
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)

    gen_ids = torch.cat(generated_tokens, dim=1)
    print(f"    Generated {gen_ids.shape[1]} tokens")

    if tokenizer:
        gen_text = tokenizer.decode(gen_ids[0].tolist())
        print(f"    Generated text: '{gen_text}'")
    else:
        print(f"    Generated token IDs: {gen_ids[0].tolist()}")

    # 8. Full generate test
    print(f"\n[8] Running full generate() test...")
    with torch.no_grad():
        t0 = time.time()
        output_ids = model.generate(
            input_ids,
            max_new_tokens=32,
            temperature=0.0,  # greedy
            eos_token_id=151645 if tokenizer else None,
        )
        gen_time = time.time() - t0

    new_tokens = output_ids.shape[1] - input_ids.shape[1]
    print(f"    Generated {new_tokens} new tokens in {gen_time:.3f}s")
    print(f"    Throughput: {new_tokens / gen_time:.1f} tokens/s")

    if tokenizer:
        full_text = tokenizer.decode(output_ids[0].tolist())
        print(f"    Full output: '{full_text}'")

    print(f"\n{'=' * 60}")
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
