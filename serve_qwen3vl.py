"""Qwen3-VL-8B FP8 Serving Script.

Loads ComfyUI FP8 quantized safetensors, dequantizes to FP16,
and serves via OpenAI-compatible API using FastAPI.

Usage:
    /home/acproject/miniconda3/envs/lyra2/bin/python serve_qwen3vl.py --port 8000
"""

import argparse
import io
import json
import os
import sys
import time
import uuid
from typing import Optional, List, Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# FP8 Dequantization
# ---------------------------------------------------------------------------

def load_fp8_safetensors_dequantized(
    path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> Dict[str, torch.Tensor]:
    """Load ComfyUI FP8 safetensors and dequantize to target dtype.

    ComfyUI FP8 format stores:
    - weight: float8_e4m3fn tensor
    - weight_scale: scalar float32 per-tensor scale
    - comfy_quant: JSON metadata (uint8)

    Dequantization: weight_fp16 = weight_fp8.to(float16) * weight_scale

    Args:
        path: Path to the .safetensors file.
        device: Target device.
        dtype: Target dtype (float16 recommended for V100).

    Returns:
        Dict mapping weight names to dequantized tensors.
    """
    from safetensors import safe_open

    result = {}
    skip_suffixes = ('.comfy_quant', '.weight_scale')

    with safe_open(path, framework='pt', device=device) as f:
        keys = list(f.keys())

        # Build scale map: param_name -> scale value
        scale_map = {}
        for key in keys:
            if key.endswith('.weight_scale'):
                param_name = key.replace('.weight_scale', '')
                scale_val = f.get_tensor(key).item()
                scale_map[param_name] = scale_val

        total = len(keys)
        loaded = 0
        for key in keys:
            # Skip metadata and scale keys
            if key.endswith(skip_suffixes):
                continue

            tensor = f.get_tensor(key)

            # Dequantize FP8 weights
            if tensor.dtype == torch.float8_e4m3fn:
                param_name = key.rsplit('.weight', 1)[0] if '.weight' in key else key
                scale = scale_map.get(param_name, 1.0)
                tensor = tensor.to(dtype) * scale
            elif tensor.is_floating_point() and tensor.dtype != dtype:
                tensor = tensor.to(dtype)

            result[key] = tensor
            loaded += 1

            if loaded % 100 == 0:
                print(f"  Loaded {loaded}/{total} tensors...")

    print(f"  Dequantized {loaded} tensors from FP8")
    return result


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def _remap_safetensors_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap ComfyUI safetensors keys to HF Qwen3VL model keys.

    The ComfyUI safetensors uses flat key names (e.g. model.layers.*),
    but HF Qwen3VLForConditionalGeneration wraps the text model inside
    model.language_model.*.

    Key mappings:
    - model.embed_tokens.* -> model.language_model.embed_tokens.*
    - model.layers.* -> model.language_model.layers.*
    - model.norm.* -> model.language_model.norm.*
    - model.visual.* -> model.visual.* (unchanged)
    - lm_head.* -> lm_head.* (unchanged)
    """
    remapped = {}
    for key, tensor in state_dict.items():
        if key.startswith('model.layers.'):
            new_key = key.replace('model.layers.', 'model.language_model.layers.', 1)
        elif key.startswith('model.embed_tokens.'):
            new_key = key.replace('model.embed_tokens.', 'model.language_model.embed_tokens.', 1)
        elif key.startswith('model.norm.'):
            new_key = key.replace('model.norm.', 'model.language_model.norm.', 1)
        else:
            new_key = key
        remapped[new_key] = tensor
    return remapped


def load_qwen3vl_model(
    safetensors_path: str,
    config_dir: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Load Qwen3-VL model from ComfyUI FP8 safetensors.

    Args:
        safetensors_path: Path to the FP8 safetensors file.
        config_dir: Directory with config.json, tokenizer, etc.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tuple of (model, processor).
    """
    from transformers import (
        Qwen3VLForConditionalGeneration,
        AutoProcessor,
        AutoConfig,
    )

    # 1. Load and dequantize weights
    print(f"[1/3] Loading FP8 weights from {safetensors_path}...")
    t0 = time.time()
    state_dict = load_fp8_safetensors_dequantized(
        safetensors_path, device="cpu", dtype=dtype
    )
    print(f"  Loaded {len(state_dict)} tensors in {time.time()-t0:.1f}s")

    # 2. Remap keys for HF model structure
    print(f"[2/3] Remapping keys for HF Qwen3VL model...")
    state_dict = _remap_safetensors_keys(state_dict)

    # 3. Create model and load weights
    print(f"[3/3] Creating and loading model...")
    config = AutoConfig.from_pretrained(config_dir, trust_remote_code=True)

    # Use accelerate to create empty model (no random init, saves time & memory)
    import accelerate
    with accelerate.init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)

    # Get model's expected keys
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())

    # Report mismatches
    missing = model_keys - loaded_keys
    unexpected = loaded_keys - model_keys
    matched = model_keys & loaded_keys
    print(f"  Matched: {len(matched)}, Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        for k in sorted(missing)[:5]:
            print(f"    Missing: {k}")
        if len(missing) > 5:
            print(f"    ... and {len(missing)-5} more")
    if unexpected:
        for k in sorted(unexpected)[:5]:
            print(f"    Unexpected: {k}")

    # Load state dict into the model using accelerate dispatch
    print(f"  Dispatching model to {device}...")
    t1 = time.time()

    # Filter to only matched keys
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

    # Use accelerate's dispatch_model to load and move to device
    # First, load state dict (assign=True replaces meta tensors with real ones)
    model.load_state_dict(filtered_state_dict, strict=False, assign=True)

    # Now move to device - since assign=True was used, real tensors exist
    # but some unmatched params may still be meta, so use to_empty first
    device_obj = torch.device(device)
    for name, param in model.named_parameters():
        if param.device == torch.device('meta'):
            # This param wasn't loaded, initialize it on device
            param.materialize(device_obj, dtype=dtype)

    model = model.to(device=device_obj, dtype=dtype).eval()

    print(f"  Model loaded in {time.time()-t1:.1f}s")

    # 4. Load processor
    print(f"  Loading processor...")
    processor = AutoProcessor.from_pretrained(config_dir, trust_remote_code=True)

    return model, processor


# ---------------------------------------------------------------------------
# API Models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: Any  # str or list of content parts


class ChatCompletionRequest(BaseModel):
    model: str = "qwen3-vl-8b"
    messages: List[Message]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="Qwen3-VL-8B API Server")

# Global model and processor
_model = None
_processor = None
_device = None


@app.on_event("startup")
async def startup():
    global _model, _processor, _device
    # Model is loaded before app starts


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-vl-8b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    from qwen_vl_utils import process_vision_info

    # Build conversation from messages
    conversation = []
    for msg in request.messages:
        content = msg.content
        if isinstance(content, str):
            conversation.append({"role": msg.role, "content": content})
        elif isinstance(content, list):
            # Handle multimodal content
            processed_content = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        processed_content.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        processed_content.append({"type": "image", "image": url})
                    elif part.get("type") == "image":
                        processed_content.append(part)
                    else:
                        processed_content.append(part)
                else:
                    processed_content.append({"type": "text", "text": str(part)})
            conversation.append({"role": msg.role, "content": processed_content})
        else:
            conversation.append({"role": msg.role, "content": str(content)})

    # Apply chat template
    try:
        text = _processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
    except (ValueError, AttributeError):
        # Fallback: manual Qwen-style chat template
        text_parts = []
        for msg in conversation:
            role = msg["role"]
            content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        text_parts.append("<|im_start|>assistant\n")
        text = "".join(text_parts)

    # Process vision info (images/videos)
    has_vision = any(
        isinstance(msg.get("content"), list) and
        any(isinstance(p, dict) and p.get("type") in ("image", "image_url") for p in msg["content"])
        for msg in conversation
    )

    if has_vision:
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = _processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(_device)
    else:
        # Text-only: use tokenizer directly (faster)
        inputs = _processor.tokenizer(
            text, return_tensors="pt"
        ).to(_device)

    input_len = inputs["input_ids"].shape[1]

    # Generate
    gen_kwargs = {
        "max_new_tokens": request.max_tokens,
        "top_p": request.top_p,
        "do_sample": request.temperature > 0,
    }
    if request.temperature > 0:
        gen_kwargs["temperature"] = request.temperature

    with torch.no_grad():
        output_ids = _model.generate(**inputs, **gen_kwargs)

    # Decode only new tokens
    generated_ids = output_ids[0, input_len:]
    output_text = _processor.tokenizer.decode(
        generated_ids, skip_special_tokens=True
    )

    completion_tokens = len(generated_ids)

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=output_text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=input_len,
            completion_tokens=completion_tokens,
            total_tokens=input_len + completion_tokens,
        ),
    )
    return response


@app.get("/health")
async def health():
    return {"status": "ok", "model": "qwen3-vl-8b"}


# ---------------------------------------------------------------------------
# Text-only completion endpoint (simpler, no vision)
# ---------------------------------------------------------------------------

class CompletionRequest(BaseModel):
    model: str = "qwen3-vl-8b"
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inputs = _processor.tokenizer(
        request.prompt, return_tensors="pt"
    ).to(_device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature if request.temperature > 0 else 1.0,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
        )

    generated_ids = output_ids[0, input_len:]
    output_text = _processor.tokenizer.decode(
        generated_ids, skip_special_tokens=True
    )

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "text": output_text,
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_len,
            "completion_tokens": len(generated_ids),
            "total_tokens": input_len + len(generated_ids),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL-8B FP8 API Server")
    parser.add_argument(
        "--safetensors",
        type=str,
        default="/home/acproject/ComfyUI/models/text_encoders/qwen3vl_8b_fp8_scaled.safetensors",
        help="Path to FP8 safetensors file",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="./qwen3vl_8b",
        help="Directory with config.json and tokenizer files",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device (e.g. cuda:0)"
    )
    parser.add_argument(
        "--dtype", type=str, default="float16", help="Model dtype"
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.float16)

    # Load model
    global _model, _processor, _device
    _device = torch.device(args.device)

    print("=" * 60)
    print("Qwen3-VL-8B FP8 API Server")
    print("=" * 60)
    print(f"  Safetensors: {args.safetensors}")
    print(f"  Config dir:  {args.config_dir}")
    print(f"  Device:      {args.device}")
    print(f"  Dtype:       {dtype}")
    print()

    _model, _processor = load_qwen3vl_model(
        args.safetensors, args.config_dir, device=args.device, dtype=dtype
    )

    print(f"\nModel ready! Starting server on port {args.port}...")
    print(f"  Chat API:  http://localhost:{args.port}/v1/chat/completions")
    print(f"  Text API:  http://localhost:{args.port}/v1/completions")
    print(f"  Health:    http://localhost:{args.port}/health")
    print()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
