"""Compare our CausalLM output with HuggingFace Qwen3ForCausalLM reference.

This script loads both models with the same weights, runs the same input,
and compares outputs at each layer to identify discrepancies.

Usage:
    /home/acproject/miniconda3/envs/lyra2/bin/python compare_with_hf.py
"""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from flash_attn_llm.models.config import Qwen3Config
from flash_attn_llm.models.causal_lm import CausalLM
from flash_attn_llm.weights.loader import WeightLoader

MODEL_PATH = "/home/acproject/ComfyUI/models/text_encoders"
PYTHON = "/home/acproject/miniconda3/envs/lyra2/bin/python"


def load_our_model():
    """Load our CausalLM model with Qwen3 weights."""
    config = Qwen3Config()
    model = CausalLM(config)
    loader = WeightLoader(model_type="qwen3")
    stats = loader.load_weights(model, MODEL_PATH, device="cuda", dtype=torch.float16)
    print(f"[Our Model] Loaded {stats['num_loaded']}/{stats['num_total']} weights")
    model = model.cuda().half().eval()

    # Verify tie_word_embeddings
    tied = model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()
    print(f"[Our Model] tie_word_embeddings working: {tied}")

    return model


def load_hf_model():
    """Load HuggingFace Qwen3ForCausalLM with same weights."""
    from transformers import Qwen3ForCausalLM, AutoConfig

    # Load config from the safetensors directory
    # We need to construct the HF model manually since we don't have a full HF model dir
    hf_config = AutoConfig.for_model(
        model_type="qwen3",
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=40960,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        attention_qk_norm=True,
    )

    model = Qwen3ForCausalLM(hf_config)

    # Load weights from safetensors
    from safetensors.torch import load_file
    import glob
    import os

    safetensor_files = glob.glob(os.path.join(MODEL_PATH, "*.safetensors"))
    state_dict = {}
    for f in safetensor_files:
        state_dict.update(load_file(f))

    # HF model expects "model." prefix for most weights, "lm_head." for head
    result = model.load_state_dict(state_dict, strict=False)
    print(f"[HF Model] Missing keys: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")
    if result.missing_keys:
        for k in result.missing_keys[:5]:
            print(f"  Missing: {k}")

    model = model.cuda().half().eval()
    return model


def compare_tensors(ours, hf, name):
    """Compare two tensors and print statistics."""
    ours_f = ours.float()
    hf_f = hf.float()

    max_diff = (ours_f - hf_f).abs().max().item()
    mean_diff = (ours_f - hf_f).abs().mean().item()
    cos_sim = F.cosine_similarity(
        ours_f.flatten().unsqueeze(0), hf_f.flatten().unsqueeze(0)
    ).item()

    match = "OK" if max_diff < 0.01 else "MISMATCH"
    if max_diff < 0.1:
        match = "OK" if max_diff < 0.01 else "CLOSE"
    else:
        match = "MISMATCH"

    print(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.6f} [{match}]")
    return max_diff


def main():
    print("=" * 70)
    print("Qwen3 0.6B: Our Model vs HuggingFace Reference Comparison")
    print("=" * 70)

    # 1. Load both models
    print("\n[1] Loading models...")
    our_model = load_our_model()
    hf_model = load_hf_model()

    # 2. Compare weight values for key layers
    print("\n[2] Comparing weight values...")
    # Embedding
    compare_tensors(
        our_model.embed_tokens.weight,
        hf_model.model.embed_tokens.weight,
        "embed_tokens.weight"
    )
    # Layer 0 attention
    compare_tensors(
        our_model.layers[0].self_attn.q_proj.weight,
        hf_model.model.layers[0].self_attn.q_proj.weight,
        "layer0.q_proj.weight"
    )
    compare_tensors(
        our_model.layers[0].self_attn.k_proj.weight,
        hf_model.model.layers[0].self_attn.k_proj.weight,
        "layer0.k_proj.weight"
    )
    # Layer 0 QK norm
    if hasattr(our_model.layers[0].self_attn, 'q_norm'):
        compare_tensors(
            our_model.layers[0].self_attn.q_norm.weight,
            hf_model.model.layers[0].self_attn.q_norm.weight,
            "layer0.q_norm.weight"
        )
        compare_tensors(
            our_model.layers[0].self_attn.k_norm.weight,
            hf_model.model.layers[0].self_attn.k_norm.weight,
            "layer0.k_norm.weight"
        )
    # Layer 0 norms
    compare_tensors(
        our_model.layers[0].input_layernorm.weight,
        hf_model.model.layers[0].input_layernorm.weight,
        "layer0.input_layernorm.weight"
    )
    compare_tensors(
        our_model.layers[0].post_attention_layernorm.weight,
        hf_model.model.layers[0].post_attention_layernorm.weight,
        "layer0.post_attention_layernorm.weight"
    )
    # Final norm
    compare_tensors(
        our_model.norm.weight,
        hf_model.model.norm.weight,
        "final_norm.weight"
    )
    # LM head (tied)
    compare_tensors(
        our_model.lm_head.weight,
        hf_model.lm_head.weight,
        "lm_head.weight"
    )

    # 3. Run forward pass and compare layer-by-layer
    print("\n[3] Layer-by-layer forward comparison...")

    # Use a simple input
    input_ids = torch.tensor([[151644, 872, 198]], dtype=torch.long, device="cuda")  # <|im_start|>user\n
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)

    # --- Our model: manual layer-by-layer ---
    with torch.no_grad():
        our_hidden = our_model.embed_tokens(input_ids)
        print(f"  After embedding: norm={our_hidden.float().norm().item():.4f}")

        for i, layer in enumerate(our_model.layers):
            residual = our_hidden
            normed = layer.input_layernorm(our_hidden)
            attn_out, kv = layer.self_attn.forward_prefill(normed, position_ids)
            our_hidden = residual + attn_out

            residual = our_hidden
            normed = layer.post_attention_layernorm(our_hidden)
            mlp_out = layer.mlp(normed)
            our_hidden = residual + mlp_out

            if i < 3 or i == 13 or i == 27:
                print(f"  Our layer {i}: hidden norm={our_hidden.float().norm().item():.4f}")

        our_hidden = our_model.norm(our_hidden)
        our_logits = our_model.lm_head(our_hidden)

    # --- HF model: use full model forward for simplicity ---
    with torch.no_grad():
        hf_outputs = hf_model(
            input_ids,
            position_ids=position_ids,
            use_cache=False,
        )
        hf_logits = hf_outputs.logits

    # 4. Compare final logits
    print("\n[4] Final logits comparison...")
    max_diff = compare_tensors(our_logits, hf_logits, "logits")

    # Top-5 tokens comparison
    our_top5 = our_logits[0, -1].float().topk(5)
    hf_top5 = hf_logits[0, -1].float().topk(5)
    print(f"\n  Our top-5 tokens: IDs={our_top5.indices.tolist()}, probs={F.softmax(our_top5.values, dim=-1).tolist()}")
    print(f"  HF  top-5 tokens: IDs={hf_top5.indices.tolist()}, probs={F.softmax(hf_top5.values, dim=-1).tolist()}")

    # 5. Generate text comparison
    print("\n[5] Generation comparison (greedy, 30 tokens)...")

    # Our model generate
    with torch.no_grad():
        our_output = our_model.generate(
            input_ids, max_new_tokens=30, temperature=0.0
        )

    # HF model generate
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "./qwen3_tokenizer", trust_remote_code=True, local_files_only=True
    )

    with torch.no_grad():
        hf_output = hf_model.generate(
            input_ids,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    our_text = tokenizer.decode(our_output[0], skip_special_tokens=True)
    hf_text = tokenizer.decode(hf_output[0], skip_special_tokens=True)

    print(f"  Our output: {our_text[:200]}")
    print(f"  HF  output: {hf_text[:200]}")

    our_gen_ids = our_output[0, seq_len:].tolist()
    hf_gen_ids = hf_output[0, seq_len:].tolist()
    print(f"\n  Our token IDs: {our_gen_ids}")
    print(f"  HF  token IDs: {hf_gen_ids}")

    match_count = sum(a == b for a, b in zip(our_gen_ids, hf_gen_ids))
    print(f"  Matching tokens: {match_count}/{min(len(our_gen_ids), len(hf_gen_ids))}")

    # 6. If logits differ significantly, do detailed attention comparison
    if max_diff > 0.1:
        print("\n[6] Detailed attention comparison (layer 0)...")
        with torch.no_grad():
            # Get our attention output
            our_emb = our_model.embed_tokens(input_ids)
            our_normed = our_model.layers[0].input_layernorm(our_emb)

            # Our Q/K/V projections
            our_q = our_model.layers[0].self_attn.q_proj(our_normed)
            our_k = our_model.layers[0].self_attn.k_proj(our_normed)
            our_v = our_model.layers[0].self_attn.v_proj(our_normed)

            # HF Q/K/V projections
            hf_emb = hf_model.model.embed_tokens(input_ids)
            hf_normed = hf_model.model.layers[0].input_layernorm(hf_emb)

            hf_q = hf_model.model.layers[0].self_attn.q_proj(hf_normed)
            hf_k = hf_model.model.layers[0].self_attn.k_proj(hf_normed)
            hf_v = hf_model.model.layers[0].self_attn.v_proj(hf_normed)

            compare_tensors(our_q, hf_q, "layer0.q_proj output")
            compare_tensors(our_k, hf_k, "layer0.k_proj output")
            compare_tensors(our_v, hf_v, "layer0.v_proj output")

            # After reshape + QK norm
            B, S, _ = our_q.shape
            head_dim = 128
            num_heads = 16
            num_kv_heads = 8

            our_q_r = our_q.view(B, S, num_heads, head_dim).transpose(1, 2)
            our_k_r = our_k.view(B, S, num_kv_heads, head_dim).transpose(1, 2)
            hf_q_r = hf_q.view(B, S, num_heads, head_dim).transpose(1, 2)
            hf_k_r = hf_k.view(B, S, num_kv_heads, head_dim).transpose(1, 2)

            # Apply QK norm
            our_q_n = our_model.layers[0].self_attn.q_norm(our_q_r)
            our_k_n = our_model.layers[0].self_attn.k_norm(our_k_r)
            hf_q_n = hf_model.model.layers[0].self_attn.q_norm(hf_q_r)
            hf_k_n = hf_model.model.layers[0].self_attn.k_norm(hf_k_r)

            compare_tensors(our_q_n, hf_q_n, "layer0.q after QK norm")
            compare_tensors(our_k_n, hf_k_n, "layer0.k after QK norm")

            # After RoPE
            cos, sin = our_model.layers[0].self_attn.rotary_emb(S, position_ids)
            our_q_rot, our_k_rot = apply_rotary_emb_for_compare(our_q_n, our_k_n, cos, sin)

            # HF RoPE
            hf_cos, hf_sin = hf_model.model.layers[0].self_attn.rotary_emb(S, position_ids)
            hf_q_rot, hf_k_rot = hf_rope_apply(hf_q_n, hf_k_n, hf_cos, hf_sin)

            compare_tensors(our_q_rot, hf_q_rot, "layer0.q after RoPE")
            compare_tensors(our_k_rot, hf_k_rot, "layer0.k after RoPE")

    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)


def apply_rotary_emb_for_compare(q, k, cos, sin):
    """Our RoPE application for comparison."""
    from flash_attn_llm.models.rope import _rotate_half
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def hf_rope_apply(q, k, cos, sin):
    """HF RoPE application for comparison."""
    # HF uses the same _rotate_half approach
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


if __name__ == "__main__":
    main()
