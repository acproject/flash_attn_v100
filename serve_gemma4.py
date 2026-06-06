"""Gemma 4 31B IT Serving Script.

Loads Gemma4ForConditionalGeneration with multi-GPU support (3x V100 32GB)
and serves via OpenAI-compatible API.

Usage:
    /home/acproject/workspace/python_projects/flash_attn_v100/venv/bin/python serve_gemma4.py --port 8001
"""

import argparse
import time
import uuid
from typing import Optional, List, Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_gemma4_model(
    model_path: str,
    dtype: torch.dtype = torch.float16,
):
    """Load Gemma4 31B model with multi-GPU device_map=auto.

    Args:
        model_path: Path to the model directory with config.json & safetensors.
        dtype: Target dtype (float16 for V100).

    Returns:
        Tuple of (model, processor).
    """
    from transformers import (
        Gemma4ForConditionalGeneration,
        AutoProcessor,
        AutoConfig,
    )

    print(f"[1/3] Loading config from {model_path}...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    print(f"[2/3] Loading model with device_map=auto (dtype={dtype})...")
    t0 = time.time()

    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Print device map summary
    if hasattr(model, 'hf_device_map'):
        devices = set(str(v) for v in model.hf_device_map.values())
        print(f"  Devices used: {devices}")

    print(f"[3/3] Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    return model, processor


# ---------------------------------------------------------------------------
# API Models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: Any  # str or list of content parts


class ChatCompletionRequest(BaseModel):
    model: str = "gemma-4-31b-it"
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

app = FastAPI(title="Gemma 4 31B IT API Server")

_model = None
_processor = None


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "gemma-4-31b-it",
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

    # Build conversation from messages
    conversation = []
    for msg in request.messages:
        content = msg.content
        if isinstance(content, str):
            conversation.append({"role": msg.role, "content": content})
        elif isinstance(content, list):
            processed_content = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        processed_content.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        processed_content.append({"type": "image", "image": url})
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
        # Fallback: Gemma-style chat template
        text_parts = ["<start_of_turn>user\n"]
        for msg in conversation:
            role = msg["role"]
            content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
            text_parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>\n")
        text_parts.append("<start_of_turn>model\n")
        text = "".join(text_parts)

    # Check for vision inputs
    has_vision = any(
        isinstance(msg.get("content"), list) and
        any(isinstance(p, dict) and p.get("type") in ("image", "image_url") for p in msg["content"])
        for msg in conversation
    )

    if has_vision:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = _processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(_model.device)
    else:
        # Text-only: use tokenizer directly
        inputs = _processor.tokenizer(
            text, return_tensors="pt"
        ).to(_model.device)

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
    return {"status": "ok", "model": "gemma-4-31b-it"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gemma 4 31B IT API Server")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/acproject/hf-hub/gemma-4-31B-it",
        help="Path to model directory",
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="Server port"
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

    global _model, _processor

    print("=" * 60)
    print("Gemma 4 31B IT API Server")
    print("=" * 60)
    print(f"  Model path: {args.model_path}")
    print(f"  Dtype:      {dtype}")
    print(f"  Port:       {args.port}")
    print()

    _model, _processor = load_gemma4_model(args.model_path, dtype=dtype)

    print(f"\nModel ready! Starting server on port {args.port}...")
    print(f"  Chat API:  http://localhost:{args.port}/v1/chat/completions")
    print(f"  Health:    http://localhost:{args.port}/health")
    print()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
