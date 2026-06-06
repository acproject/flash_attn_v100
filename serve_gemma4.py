"""Gemma 4 31B IT Serving Script.

Loads Gemma4ForConditionalGeneration with multi-GPU support (3x V100 32GB)
and serves via OpenAI-compatible API.

Usage:
    /home/acproject/workspace/python_projects/flash_attn_v100/venv/bin/python serve_gemma4.py --port 8001
"""

import argparse
import time
import uuid
import json
from typing import Optional, List, Dict, Any, AsyncIterator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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

class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolDefinition(BaseModel):
    type: str = "function"
    function: FunctionDefinition


class Message(BaseModel):
    role: str
    content: Any = None  # str, list of content parts, or None for tool calls
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "gemma-4-31b-it"
    messages: List[Message]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    stop: Optional[List[str]] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Any] = None  # "auto", "none", or {"type":"function","function":{"name":"..."}}


class CompletionRequest(BaseModel):
    model: str = "gemma-4-31b-it"
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    stop: Optional[List[str]] = None
    echo: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class CompletionChoice(BaseModel):
    index: int
    text: str
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


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="Gemma 4 31B IT API Server")

_model = None
_processor = None


# ---- Helper: apply stop sequences during generation ----

def _check_stop(text: str, stop_sequences: List[str]) -> Optional[str]:
    """Check if any stop sequence appears in text. Returns the matched stop or None."""
    for stop in stop_sequences:
        if stop in text:
            return stop
    return None


def _build_tool_system_prompt(tools: List[ToolDefinition]) -> str:
    """Build a system prompt describing available tools for Gemma models."""
    tool_descs = []
    for tool in tools:
        f = tool.function
        desc = f"- {f.name}: {f.description or 'No description'}"
        if f.parameters:
            desc += f"\n  Parameters: {json.dumps(f.parameters, ensure_ascii=False)}"
        tool_descs.append(desc)
    return (
        "You have access to the following tools. When you need to call a tool, "
        "respond with a JSON block in the following format:\n"
        "```json\n"
        '{"tool_calls": [{"name": "<tool_name>", "arguments": {<args>}}]}\n'
        "```\n\n"
        "Available tools:\n" + "\n".join(tool_descs)
    )


def _parse_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    """Try to extract tool calls from model output text."""
    # Look for ```json ... ``` blocks
    import re
    pattern = r'```json\s*(\{.*?\})\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
            if "tool_calls" in data:
                return data["tool_calls"]
        except json.JSONDecodeError:
            continue
    # Also try plain JSON without code block
    pattern2 = r'\{"tool_calls"\s*:\s*\[.*?\]\}'
    matches2 = re.findall(pattern2, text, re.DOTALL)
    for match in matches2:
        try:
            data = json.loads(match)
            if "tool_calls" in data:
                return data["tool_calls"]
        except json.JSONDecodeError:
            continue
    return None


# ---- Streaming helper ----

async def _stream_generate(input_ids, attention_mask, gen_kwargs, stop_sequences, request_id, model_name, is_chat=True):
    """Async generator that yields SSE chunks for streaming responses."""
    from transformers import TextIteratorStreamer
    import threading

    streamer = TextIteratorStreamer(
        _processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs["streamer"] = streamer

    # Run generation in a background thread
    thread = threading.Thread(
        target=_model.generate,
        kwargs={**{"input_ids": input_ids, "attention_mask": attention_mask}, **gen_kwargs},
    )
    thread.start()

    completion_tokens = 0
    accumulated = ""

    try:
        for new_text in streamer:
            accumulated += new_text

            # Check stop sequences
            finish_reason = None
            output_text = new_text
            if stop_sequences:
                matched_stop = _check_stop(accumulated, stop_sequences)
                if matched_stop:
                    # Truncate the output at the stop sequence
                    stop_idx = accumulated.index(matched_stop)
                    # Only output text up to the stop, minus what we already sent
                    already_sent = len(accumulated) - len(new_text)
                    remaining = accumulated[already_sent:stop_idx]
                    output_text = remaining
                    finish_reason = "stop"

            completion_tokens += 1
            created = int(time.time())

            if is_chat:
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": output_text} if output_text else {},
                        "finish_reason": finish_reason,
                    }],
                }
            else:
                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "text": output_text,
                        "finish_reason": finish_reason,
                    }],
                }

            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            if finish_reason == "stop":
                break

        # Send final chunk with finish_reason
        if is_chat:
            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        else:
            final_chunk = {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
            }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    finally:
        thread.join()


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
        elif content is None and msg.tool_calls:
            # Assistant message with tool calls
            conversation.append({"role": msg.role, "content": str(msg.tool_calls)})
        elif content is None and msg.tool_call_id:
            # Tool result message
            conversation.append({"role": msg.role, "content": str(msg.content or "")})
        else:
            conversation.append({"role": msg.role, "content": str(content)})

    # Inject tool descriptions as system message if tools provided
    if request.tools:
        tool_prompt = _build_tool_system_prompt(request.tools)
        # Prepend tool system message
        conversation.insert(0, {"role": "system", "content": tool_prompt})

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
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    # Generate
    gen_kwargs = {
        "max_new_tokens": request.max_tokens,
        "top_p": request.top_p,
        "do_sample": request.temperature > 0,
    }
    if request.temperature > 0:
        gen_kwargs["temperature"] = request.temperature

    # Streaming
    if request.stream:
        return StreamingResponse(
            _stream_generate(
                inputs["input_ids"],
                inputs.get("attention_mask"),
                gen_kwargs,
                request.stop or [],
                request_id,
                request.model,
                is_chat=True,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming
    with torch.no_grad():
        output_ids = _model.generate(**inputs, **gen_kwargs)

    # Decode only new tokens
    generated_ids = output_ids[0, input_len:]
    output_text = _processor.tokenizer.decode(
        generated_ids, skip_special_tokens=True
    )

    # Apply stop sequences
    finish_reason = "stop"
    if request.stop:
        matched_stop = _check_stop(output_text, request.stop)
        if matched_stop:
            output_text = output_text[:output_text.index(matched_stop)]

    # Check for tool calls
    tool_calls_result = None
    if request.tools and request.tool_choice != "none":
        tool_calls_result = _parse_tool_calls(output_text)
        if tool_calls_result:
            finish_reason = "tool_calls"

    completion_tokens = len(generated_ids)

    # Build response message
    if tool_calls_result:
        response_message = Message(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                    },
                }
                for tc in tool_calls_result
            ],
        )
    else:
        response_message = Message(role="assistant", content=output_text)

    response = ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=response_message,
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=input_len,
            completion_tokens=completion_tokens,
            total_tokens=input_len + completion_tokens,
        ),
    )
    return response


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Tokenize prompt
    inputs = _processor.tokenizer(
        request.prompt, return_tensors="pt"
    ).to(_model.device)

    input_len = inputs["input_ids"].shape[1]
    request_id = f"cmpl-{uuid.uuid4().hex[:8]}"

    # Generate
    gen_kwargs = {
        "max_new_tokens": request.max_tokens,
        "top_p": request.top_p,
        "do_sample": request.temperature > 0,
    }
    if request.temperature > 0:
        gen_kwargs["temperature"] = request.temperature

    # Streaming
    if request.stream:
        return StreamingResponse(
            _stream_generate(
                inputs["input_ids"],
                inputs.get("attention_mask"),
                gen_kwargs,
                request.stop or [],
                request_id,
                request.model,
                is_chat=False,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming
    with torch.no_grad():
        output_ids = _model.generate(**inputs, **gen_kwargs)

    # Decode only new tokens
    generated_ids = output_ids[0, input_len:]
    output_text = _processor.tokenizer.decode(
        generated_ids, skip_special_tokens=True
    )

    # Apply stop sequences
    finish_reason = "stop"
    if request.stop:
        matched_stop = _check_stop(output_text, request.stop)
        if matched_stop:
            output_text = output_text[:output_text.index(matched_stop)]

    # Echo prompt if requested
    if request.echo:
        prompt_text = request.prompt
        output_text = prompt_text + output_text

    completion_tokens = len(generated_ids)

    response = CompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=request.model,
        choices=[
            CompletionChoice(
                index=0,
                text=output_text,
                finish_reason=finish_reason,
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
    print(f"  Chat API:       http://localhost:{args.port}/v1/chat/completions")
    print(f"  Completions API: http://localhost:{args.port}/v1/completions")
    print(f"  Models API:     http://localhost:{args.port}/v1/models")
    print(f"  Health:         http://localhost:{args.port}/health")
    print(f"  Features:       Streaming, Stop Sequences, Tool Calling")
    print()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
