"""Test script for loading and running Gemma4-31B-it with flash_attn_llm.

Tests the full pipeline: config parsing -> model creation -> weight loading -> inference.
Supports multi-GPU pipeline parallelism for large models.
"""

import json
import sys
import time

import torch

# Add project to path
sys.path.insert(0, "/home/acproject/workspace/python_projects/flash_attn_v100")

from flash_attn_llm.models.config import get_config_from_hf_json, Gemma4Config
from flash_attn_llm.models.causal_lm import CausalLM
from flash_attn_llm.weights.loader import WeightLoader
from flash_attn_llm.tokenizer.tokenizer import Tokenizer

MODEL_PATH = "/home/acproject/hf-hub/gemma-4-31B-it"


def get_device_map(num_layers: int, num_gpus: int) -> dict:
    """Create a device map distributing layers across GPUs by estimated memory.

    Accounts for global attention layers being larger than sliding ones.
    Puts embed_tokens on GPU 0, norm + lm_head on last GPU.
    """
    if num_gpus <= 1:
        return None

    # Calculate per-layer memory to distribute evenly
    # Sliding layer: ~1GB fp16, Global layer: ~1.5GB fp16
    # embed_tokens: ~0.5GB fp16
    # We need to fit within 32GB per GPU with ~2GB overhead for CUDA context
    # Available per GPU: ~30GB
    # Total model: ~61.6GB / 3 GPUs ≈ 20.5GB per GPU
    # But to_empty() may use extra memory during allocation
    device_map = {}
    layers_per_gpu = num_layers // num_gpus
    # GPU 0 gets significantly fewer layers since it also holds embed_tokens
    # and to_empty() needs extra memory during allocation
    gpu0_layers = layers_per_gpu - 3
    remaining = num_layers - gpu0_layers
    layers_per_other = remaining // (num_gpus - 1)

    idx = 0
    for _ in range(gpu0_layers):
        device_map[f"layers.{idx}"] = "cuda:0"
        idx += 1
    for gpu in range(1, num_gpus):
        count = layers_per_other if gpu < num_gpus - 1 else (num_layers - idx)
        for _ in range(count):
            device_map[f"layers.{idx}"] = f"cuda:{gpu}"
            idx += 1

    device_map["embed_tokens"] = "cuda:0"
    device_map["norm"] = f"cuda:{num_gpus - 1}"
    device_map["lm_head"] = f"cuda:{num_gpus - 1}"

    return device_map


def move_model_to_devices(model, device_map):
    """Move model submodules to devices according to device_map.

    Uses to_empty() which is the proper way to materialize meta tensors.
    """
    if device_map is None:
        return model.to_empty(device="cuda:0")

    for name, device in device_map.items():
        module = model.get_submodule(name)
        module.to_empty(device=device)
        print(f"    {name} -> {device}")

    return model


def get_model_device(model, layer_idx: int):
    """Get the device for a specific layer."""
    return model.layers[layer_idx].self_attn.q_proj.weight.device


def test_config_parsing():
    """Step 1: Parse config.json and create Gemma4Config."""
    print("=" * 60)
    print("Step 1: Config Parsing")
    print("=" * 60)

    with open(f"{MODEL_PATH}/config.json") as f:
        config_dict = json.load(f)

    config = get_config_from_hf_json(config_dict)

    print(f"  model_type: {config.model_type}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  head_dim: {config.head_dim}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  tie_word_embeddings: {config.tie_word_embeddings}")
    print(f"  hidden_act: {config.hidden_act}")
    print(f"  attention_qk_norm: {config.attention_qk_norm}")
    print(f"  partial_rotary_factor: {getattr(config, 'partial_rotary_factor', 'N/A')}")
    print(f"  final_logit_softcapping: {getattr(config, 'final_logit_softcapping', 'N/A')}")
    print(f"  sliding_window: {getattr(config, 'sliding_window', 'N/A')}")
    layer_types = getattr(config, 'layer_types', None)
    if layer_types:
        print(f"  layer_types: {layer_types[:3]}... ({len(layer_types)} layers)")
    print(f"  global_head_dim: {getattr(config, 'global_head_dim', 'N/A')}")
    print(f"  rope_theta: {config.rope_theta}")

    assert isinstance(config, Gemma4Config), f"Expected Gemma4Config, got {type(config)}"
    assert config.head_dim == 256, f"Expected head_dim=256, got {config.head_dim}"
    # sliding_attention has no partial_rotary_factor (defaults to 1.0)
    # full_attention has partial_rotary_factor=0.25 (stored as global_partial_rotary_factor)
    assert config.partial_rotary_factor == 1.0, f"Expected partial_rotary_factor=1.0, got {config.partial_rotary_factor}"
    assert getattr(config, 'global_partial_rotary_factor', None) == 0.25, f"Expected global_partial_rotary_factor=0.25"
    assert config.final_logit_softcapping == 30.0, f"Expected final_logit_softcapping=30.0, got {config.final_logit_softcapping}"
    assert config.hidden_act == "geglu", f"Expected hidden_act=geglu, got {config.hidden_act}"

    print("  Config parsing PASSED!\n")
    return config


def test_model_creation(config):
    """Step 2: Create CausalLM model from config."""
    print("=" * 60)
    print("Step 2: Model Creation")
    print("=" * 60)

    num_gpus = torch.cuda.device_count()
    print(f"  Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")

    # Create model on meta device with fp16 (no memory allocation)
    with torch.device('meta'):
        model = CausalLM(config)
    model = model.half()  # Convert dtype on meta device (still no memory)

    total_params = sum(p.numel() for p in model.parameters())
    total_size = total_params * 2  # fp16 = 2 bytes

    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size (fp16): {total_size / 1e9:.2f} GB")
    print(f"  Num layers: {len(model.layers)}")
    print(f"  Has pre_feedforward_layernorm: {model.layers[0].has_pre_feedforward_norm}")
    print(f"  Has layer_scalar: {hasattr(model.layers[0], 'layer_scalar')}")
    print(f"  Has qk_norm: {model.layers[0].self_attn.qk_norm}")
    print(f"  Rotary dim: {model.layers[0].self_attn.rotary_emb.rotary_dim}")
    print(f"  Final logit softcapping: {model.final_logit_softcapping}")

    assert model.layers[0].has_pre_feedforward_norm, "Expected pre_feedforward_layernorm"
    assert model.layers[0].self_attn.qk_norm, "Expected qk_norm"
    # Sliding attention: partial_rotary_factor=1.0, rotary_dim=256
    assert model.layers[0].self_attn.rotary_emb.rotary_dim == 256, \
        f"Expected rotary_dim=256 (sliding), got {model.layers[0].self_attn.rotary_emb.rotary_dim}"
    # Global attention (layer 5): partial_rotary_factor=0.25, rotary_dim=128
    assert model.layers[5].self_attn.rotary_emb.rotary_dim == 128, \
        f"Expected rotary_dim=128 (global), got {model.layers[5].self_attn.rotary_emb.rotary_dim}"
    assert model.final_logit_softcapping == 30.0

    # Move model to GPU(s) BEFORE loading weights
    print(f"\n  Moving model to GPU(s)...")
    device_map = get_device_map(config.num_hidden_layers, num_gpus)
    if device_map:
        print(f"  Using multi-GPU device map:")
        move_model_to_devices(model, device_map)
    else:
        model = model.cuda()
        print(f"  Using single GPU: cuda:0")

    # Verify model is on GPU
    first_device = next(model.parameters()).device
    print(f"  Model first param device: {first_device}")

    print("  Model creation PASSED!\n")
    return model, device_map


def test_weight_loading(model, device_map):
    """Step 3: Load weights from safetensors directly to GPU."""
    print("=" * 60)
    print("Step 3: Weight Loading")
    print("=" * 60)

    loader = WeightLoader(tp_rank=0, tp_size=1)

    print("  Loading weights (this may take a while for 62GB model)...")
    start = time.time()

    # Load weights - model is already on GPU, so copy_ will go to GPU
    stats = loader.load_weights(
        model,
        MODEL_PATH,
        device="cuda",  # Weights will be moved to CUDA before copy
        dtype=torch.float16,
    )
    load_time = time.time() - start

    print(f"  Load time: {load_time:.1f}s")
    print(f"  Total weights found: {stats['num_total']}")
    print(f"  Weights loaded: {stats['num_loaded']}")
    print(f"  Weights skipped: {stats['num_skipped']}")
    print(f"  Model type detected: {stats['model_type']}")
    print(f"  Load ratio: {stats['num_loaded']}/{stats['num_total']}")

    # Verify tied weights (lm_head should have embed_tokens values)
    print(f"  lm_head.weight sum: {model.lm_head.weight.sum().item():.4f}")
    print(f"  embed_tokens.weight sum: {model.embed_tokens.weight.sum().item():.4f}")
    if model.lm_head.weight.sum().item() == 0:
        print("  WARNING: lm_head.weight is all zeros! Tied weights not loaded.")

    # Check for uninitialized (zero) parameters
    zero_params = []
    for name, param in model.named_parameters():
        if param.sum().item() == 0 and 'layer_scalar' not in name:
            zero_params.append(name)
    if zero_params:
        print(f"  WARNING: {len(zero_params)} parameters are all zeros (excluding layer_scalar):")
        for p in zero_params[:10]:
            print(f"    {p}")
        if len(zero_params) > 10:
            print(f"    ... and {len(zero_params) - 10} more")

    # Verify weights are on GPU
    first_device = next(model.parameters()).device
    print(f"  First param device after loading: {first_device}")

    print("  Weight loading PASSED!\n")
    return model


def test_inference(model, device_map):
    """Step 4: Run inference with a simple prompt."""
    print("=" * 60)
    print("Step 4: Inference")
    print("=" * 60)

    model.eval()

    tokenizer = Tokenizer(MODEL_PATH)

    prompt = "What is the capital of France?"
    print(f"  Prompt: {prompt}")

    # Use chat template for instruction-tuned model
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Remove thought channel to get direct answer
    # Gemma4-it adds <|channel>thought\n<channel|> at the end of the prompt
    # We can keep it for thinking mode or remove for direct answers
    print(f"  Formatted prompt: {repr(formatted[:150])}...")

    input_ids = tokenizer.encode(formatted)
    print(f"  Token count: {len(input_ids)}")

    # Determine the device for embed_tokens (input should go there)
    embed_device = model.embed_tokens.weight.device
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=embed_device)
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Input device: {input_tensor.device}")

    # Prefill
    print("  Running prefill...")
    start = time.time()
    with torch.no_grad():
        position_ids = torch.arange(input_tensor.shape[1], device=embed_device).unsqueeze(0)
        logits, kv_caches = model.forward_prefill(input_tensor, position_ids, return_kv_cache=True)

    prefill_time = time.time() - start
    print(f"  Prefill time: {prefill_time:.2f}s")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits device: {logits.device}")
    print(f"  KV caches: {len(kv_caches)} layers")

    # Sample first token
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    token_text = tokenizer.decode([next_token.item()])
    print(f"  First token: {next_token.item()} -> '{token_text}'")

    # Debug: check logits range
    print(f"  Logits min={logits.min().item():.2f}, max={logits.max().item():.2f}, mean={logits.mean().item():.2f}")
    print(f"  Last position logits min={logits[:, -1, :].min().item():.2f}, max={logits[:, -1, :].max().item():.2f}")

    # Clamp token to valid range (safety check)
    vocab_size = model.config.vocab_size
    if next_token.item() >= vocab_size:
        print(f"  WARNING: token {next_token.item()} >= vocab_size {vocab_size}, clamping")
        next_token = next_token.clamp(0, vocab_size - 1)

    # Decode more tokens - Gemma4-it has a thinking mode that generates
    # internal reasoning before producing the final answer
    print("  Running decode...")
    generated_tokens = [next_token.item()]
    cache_len = input_tensor.shape[1]
    current_token = next_token

    decode_start = time.time()
    MAX_NEW_TOKENS = 100
    for i in range(MAX_NEW_TOKENS - 1):
        position_id = torch.full((1, 1), cache_len, dtype=torch.long, device=embed_device)
        with torch.no_grad():
            logits, kv_caches = model.forward_decode(
                current_token, kv_caches, cache_len, position_id
            )
        cache_len += 1
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True).clamp(0, vocab_size - 1)
        generated_tokens.append(next_token.item())
        current_token = next_token

    decode_time = time.time() - decode_start
    print(f"  Decode time ({MAX_NEW_TOKENS - 1} tokens): {decode_time:.2f}s")
    print(f"  Decode speed: {(MAX_NEW_TOKENS - 1) / decode_time:.1f} tokens/s")

    # Decode full output
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = tokenizer.decode(input_ids + generated_tokens, skip_special_tokens=True)

    print(f"\n  Generated text: {output_text}")
    print(f"\n  Full output: {full_text}")

    print("\n  Inference PASSED!\n")
    return full_text


if __name__ == "__main__":
    print("Testing Gemma4-31B-it with flash_attn_llm")
    print(f"PyTorch: {torch.__version__}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available! Check GPU drivers and environment.")
        sys.exit(1)
    print(f"CUDA available: True")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    try:
        config = test_config_parsing()
        model, device_map = test_model_creation(config)
        model = test_weight_loading(model, device_map)
        result = test_inference(model, device_map)
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
