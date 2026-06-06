"""OpenAI-compatible LLM inference server built on FastAPI.

Provides:
- POST /v1/chat/completions   – Chat completion (streaming + non-streaming)
- POST /v1/completions        – Text completion (streaming + non-streaming)
- GET  /v1/models             – List available models
- GET  /health                – Health check
- GET  /metrics               – Basic metrics (optional)

Supports SSE streaming, request cancellation, timeout handling,
and an async request queue.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from flash_attn_llm.engine.engine import LLMEngine
from flash_attn_llm.engine.request import InferenceRequest, RequestStatus


# ======================================================================
# Pydantic request / response models
# ======================================================================

class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: list[dict]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 128
    stream: bool = False
    stop: Optional[list[str]] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict


class CompletionRequest(BaseModel):
    """OpenAI-compatible text completion request."""
    model: str
    prompt: str
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 128
    stream: bool = False
    stop: Optional[list[str]] = None


class CompletionResponse(BaseModel):
    """OpenAI-compatible text completion response."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict


class ModelInfo(BaseModel):
    """Model metadata."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "flash-attn-llm"


class ModelsResponse(BaseModel):
    """Response for GET /v1/models."""
    object: str = "list"
    data: list[ModelInfo]


# ======================================================================
# LLMServer
# ======================================================================

class LLMServer:
    """OpenAI-compatible LLM inference server.

    Wraps an LLMEngine and exposes REST endpoints for text and chat
    completions, with support for SSE streaming, request cancellation,
    and timeout handling.
    """

    def __init__(
        self,
        engine: LLMEngine,
        host: str = "0.0.0.0",
        port: int = 8000,
        request_timeout: float = 120.0,
    ):
        """
        Args:
            engine: The LLMEngine instance to serve.
            host: Bind address.
            port: Bind port.
            request_timeout: Maximum seconds to wait for a generation to complete.
        """
        self.engine = engine
        self.host = host
        self.port = port
        self.request_timeout = request_timeout

        self.app = FastAPI(title="FlashAttn LLM Server")
        self._setup_middleware()
        self._setup_routes()

        # Async request queue and result store
        self._pending_queue: asyncio.Queue[str] = asyncio.Queue()
        self._results: dict[str, InferenceRequest] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}

        # Background task reference
        self._engine_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_middleware(self) -> None:
        """Configure CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        """Register all API routes."""

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest) -> Any:
            return await self._handle_chat_completions(request)

        @self.app.post("/v1/completions")
        async def completions(request: CompletionRequest) -> Any:
            return await self._handle_completions(request)

        @self.app.get("/v1/models")
        async def list_models() -> ModelsResponse:
            return self._handle_list_models()

        @self.app.get("/health")
        async def health() -> JSONResponse:
            return JSONResponse({"status": "ok"})

        @self.app.get("/metrics")
        async def metrics() -> JSONResponse:
            return JSONResponse(self._get_metrics())

        @self.app.on_event("startup")
        async def on_startup() -> None:
            self._engine_task = asyncio.create_task(self._engine_loop())

        @self.app.on_event("shutdown")
        async def on_shutdown() -> None:
            if self._engine_task is not None:
                self._engine_task.cancel()
                try:
                    await self._engine_task
                except asyncio.CancelledError:
                    pass

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    async def _handle_chat_completions(self, req: ChatCompletionRequest) -> Any:
        """Handle a chat completion request."""
        prompt = self._messages_to_prompt(req.messages)
        request_id = str(uuid.uuid4())

        if req.stream:
            return StreamingResponse(
                self._stream_chat_completion(request_id, prompt, req),
                media_type="text/event-stream",
            )

        # Non-streaming: synchronous generation
        try:
            result = await asyncio.wait_for(
                self._generate_async(request_id, prompt, req),
                timeout=self.request_timeout,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Generation timed out")

        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=req.model,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop",
            }],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )

    async def _stream_chat_completion(
        self,
        request_id: str,
        prompt: str,
        req: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens via SSE."""
        cancel_event = asyncio.Event()
        self._cancel_events[request_id] = cancel_event

        # Queue the request with a stream callback
        token_queue: asyncio.Queue[str] = asyncio.Queue()

        def stream_callback(token_text: str) -> None:
            try:
                token_queue.put_nowait(token_text)
            except Exception:
                pass

        self.engine.add_request(
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stream_callback=stream_callback,
        )

        try:
            while not cancel_event.is_set():
                try:
                    token = await asyncio.wait_for(token_queue.get(), timeout=self.request_timeout)
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'error': 'timeout'})}\n\n"
                    break

                # Check if the engine finished this request
                inference_req = self.engine.get_request(request_id)
                if inference_req and inference_req.is_finished:
                    # Send final chunk
                    final_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    break
        finally:
            self._cancel_events.pop(request_id, None)

    # ------------------------------------------------------------------
    # Text completions
    # ------------------------------------------------------------------

    async def _handle_completions(self, req: CompletionRequest) -> Any:
        """Handle a text completion request."""
        request_id = str(uuid.uuid4())

        if req.stream:
            return StreamingResponse(
                self._stream_completion(request_id, req),
                media_type="text/event-stream",
            )

        try:
            result = await asyncio.wait_for(
                self._generate_async(request_id, req.prompt, req),
                timeout=self.request_timeout,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Generation timed out")

        return CompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=req.model,
            choices=[{
                "index": 0,
                "text": result,
                "finish_reason": "stop",
            }],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )

    async def _stream_completion(
        self,
        request_id: str,
        req: CompletionRequest,
    ) -> AsyncIterator[str]:
        """Stream text completion tokens via SSE."""
        cancel_event = asyncio.Event()
        self._cancel_events[request_id] = cancel_event

        token_queue: asyncio.Queue[str] = asyncio.Queue()

        def stream_callback(token_text: str) -> None:
            try:
                token_queue.put_nowait(token_text)
            except Exception:
                pass

        self.engine.add_request(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stream_callback=stream_callback,
        )

        try:
            while not cancel_event.is_set():
                try:
                    token = await asyncio.wait_for(token_queue.get(), timeout=self.request_timeout)
                    chunk = {
                        "id": request_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{
                            "index": 0,
                            "text": token,
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'error': 'timeout'})}\n\n"
                    break

                inference_req = self.engine.get_request(request_id)
                if inference_req and inference_req.is_finished:
                    final_chunk = {
                        "id": request_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{
                            "index": 0,
                            "text": "",
                            "finish_reason": "stop",
                        }],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    break
        finally:
            self._cancel_events.pop(request_id, None)

    # ------------------------------------------------------------------
    # Model listing
    # ------------------------------------------------------------------

    def _handle_list_models(self) -> ModelsResponse:
        """Return available models."""
        model_name = getattr(self.engine, "model_path", "default")
        return ModelsResponse(
            data=[ModelInfo(id=model_name, created=int(time.time()))]
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _get_metrics(self) -> dict[str, Any]:
        """Return basic server metrics."""
        scheduler = self.engine.scheduler
        return {
            "scheduler": {
                "waiting": scheduler.get_num_waiting(),
                "active": scheduler.get_num_active(),
                "block_usage": scheduler.get_block_usage(),
            },
            "kv_cache": {
                "num_active": self.engine.kv_cache_manager.num_active,
            },
        }

    # ------------------------------------------------------------------
    # Async generation helpers
    # ------------------------------------------------------------------

    async def _generate_async(
        self,
        request_id: str,
        prompt: str,
        req: Any,
    ) -> str:
        """Run generation asynchronously by stepping the engine in a thread."""
        max_tokens = getattr(req, "max_tokens", 128)
        temperature = getattr(req, "temperature", 1.0)
        top_p = getattr(req, "top_p", 1.0)

        # Add request to engine
        rid = self.engine.add_request(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Run engine steps in an executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        while not self.engine.get_request(rid).is_finished:
            await loop.run_in_executor(None, self.engine.step)

        result = self.engine.get_request(rid)
        return result.output_text

    # ------------------------------------------------------------------
    # Background engine loop
    # ------------------------------------------------------------------

    async def _engine_loop(self) -> None:
        """Background loop that continuously steps the engine."""
        loop = asyncio.get_event_loop()
        try:
            while True:
                if self.engine.has_unfinished_requests():
                    await loop.run_in_executor(None, self.engine.step)
                else:
                    await asyncio.sleep(0.01)  # Idle sleep
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _messages_to_prompt(messages: list[dict]) -> str:
        """Convert chat messages to a single prompt string.

        Uses a simple concatenation format.  Production deployments would
        use the model's chat template.
        """
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the server (blocking)."""
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)
