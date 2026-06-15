"""Interactive chat server for Gemma4-31B-it model.

Starts a FastAPI server with a web UI for manual testing.
Usage: python chat_server.py [--port 8000]
"""

import argparse
import json
import sys
import time
import uuid
from typing import Optional

import torch

sys.path.insert(0, "/home/acproject/workspace/python_projects/flash_attn_v100")

from flash_attn_llm.models.config import get_config_from_hf_json
from flash_attn_llm.models.causal_lm import CausalLM
from flash_attn_llm.weights.loader import WeightLoader
from flash_attn_llm.tokenizer.tokenizer import Tokenizer

MODEL_PATH = "/home/acproject/hf-hub/gemma-4-31B-it"

# Global model state
model = None
tokenizer = None
sessions = {}


def get_device_map(num_layers: int, num_gpus: int) -> dict:
    if num_gpus <= 1:
        return None
    device_map = {}
    layers_per_gpu = num_layers // num_gpus
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
    if device_map is None:
        return model.to_empty(device="cuda:0")
    for name, device in device_map.items():
        module = model.get_submodule(name)
        module.to_empty(device=device)
    return model


def load_model():
    global model, tokenizer

    print("Loading config...")
    with open(f"{MODEL_PATH}/config.json") as f:
        config_dict = json.load(f)
    config = get_config_from_hf_json(config_dict)

    print("Creating model on meta device...")
    with torch.device('meta'):
        model = CausalLM(config)
    model = model.half()

    num_gpus = torch.cuda.device_count()
    device_map = get_device_map(config.num_hidden_layers, num_gpus)
    print(f"Distributing across {num_gpus} GPUs...")
    move_model_to_devices(model, device_map)

    print("Loading weights...")
    loader = WeightLoader(tp_rank=0, tp_size=1)
    stats = loader.load_weights(model, MODEL_PATH, device="cuda", dtype=torch.float16)
    print(f"Loaded {stats['num_loaded']}/{stats['num_total']} weights")

    model.eval()

    print("Loading tokenizer...")
    tokenizer = Tokenizer(MODEL_PATH)

    print("Model ready!")


def generate_response(prompt: str, max_new_tokens: int = 256, temperature: float = 0.7,
                      top_p: float = 0.9, top_k: int = 50) -> str:
    """Generate a response for the given prompt."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    input_ids = tokenizer.encode(formatted)
    embed_device = model.embed_tokens.weight.device
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=embed_device)

    with torch.no_grad():
        position_ids = torch.arange(input_tensor.shape[1], device=embed_device).unsqueeze(0)
        logits, kv_caches = model.forward_prefill(input_tensor, position_ids, return_kv_cache=True)

    vocab_size = model.config.vocab_size

    # Sample first token
    next_token = _sample_token(logits[:, -1, :], temperature, top_k, top_p)
    generated_tokens = [next_token.item()]

    cache_len = input_tensor.shape[1]
    current_token = next_token

    for _ in range(max_new_tokens - 1):
        position_id = torch.full((1, 1), cache_len, dtype=torch.long, device=embed_device)
        with torch.no_grad():
            logits, kv_caches = model.forward_decode(
                current_token, kv_caches, cache_len, position_id
            )
        cache_len += 1
        next_token = _sample_token(logits[:, -1, :], temperature, top_k, top_p)
        generated_tokens.append(next_token.item())
        current_token = next_token

        # Check for end-of-turn tokens
        token_id = next_token.item()
        if token_id in (106, 1):  # <eos> or end-of-turn
            break

    # Decode only the generated part
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return text


def _sample_token(logits, temperature=0.7, top_k=50, top_p=0.9):
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float("-inf")

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# --- FastAPI app ---

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Gemma4 Chat Server")


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


class ChatResponse(BaseModel):
    response: str
    tokens_generated: int
    time_seconds: float


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Gemma4 Chat</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #1a1a2e; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
.header { background: #16213e; padding: 12px 20px; border-bottom: 1px solid #0f3460;
          font-size: 18px; font-weight: 600; }
.header span { color: #e94560; }
#chat { flex: 1; overflow-y: auto; padding: 20px; }
.msg { margin-bottom: 16px; max-width: 80%; padding: 12px 16px; border-radius: 12px;
       line-height: 1.6; white-space: pre-wrap; word-wrap: break-word; }
.msg.user { background: #0f3460; margin-left: auto; border-bottom-right-radius: 4px; }
.msg.assistant { background: #16213e; border-bottom-left-radius: 4px; }
.msg .role { font-size: 12px; color: #888; margin-bottom: 4px; }
.msg .meta { font-size: 11px; color: #666; margin-top: 4px; }
.input-area { background: #16213e; padding: 16px; border-top: 1px solid #0f3460;
              display: flex; gap: 12px; }
#input { flex: 1; padding: 12px 16px; border-radius: 8px; border: 1px solid #0f3460;
         background: #1a1a2e; color: #e0e0e0; font-size: 15px; resize: none;
         font-family: inherit; }
#input:focus { outline: none; border-color: #e94560; }
button { padding: 12px 24px; border-radius: 8px; border: none; background: #e94560;
         color: white; font-size: 15px; cursor: pointer; font-weight: 600; }
button:hover { background: #c73652; }
button:disabled { background: #555; cursor: not-allowed; }
.settings { display: flex; gap: 16px; padding: 8px 20px; background: #16213e;
            border-top: 1px solid #0f3460; font-size: 13px; align-items: center; }
.settings label { color: #888; }
.settings input { width: 60px; padding: 4px 8px; border-radius: 4px; border: 1px solid #0f3460;
                  background: #1a1a2e; color: #e0e0e0; }
</style>
</head>
<body>
<div class="header">Gemma4 <span>31B-it</span> Chat</div>
<div id="chat"></div>
<div class="settings">
  <label>Max tokens: <input type="number" id="max_tokens" value="256" min="1" max="1024"></label>
  <label>Temperature: <input type="number" id="temperature" value="0.7" min="0" max="2" step="0.1"></label>
  <label>Top-p: <input type="number" id="top_p" value="0.9" min="0" max="1" step="0.05"></label>
  <label>Top-k: <input type="number" id="top_k" value="50" min="1" max="200"></label>
</div>
<div class="input-area">
  <textarea id="input" rows="2" placeholder="Type your message..."></textarea>
  <button id="send" onclick="send()">Send</button>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

function addMsg(role, text, meta) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = '<div class="role">' + role + '</div>' +
    '<div>' + text.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</div>' +
    (meta ? '<div class="meta">' + meta + '</div>' : '');
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function send() {
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  document.getElementById('send').disabled = true;

  addMsg('user', msg);

  const params = {
    message: msg,
    max_tokens: parseInt(document.getElementById('max_tokens').value),
    temperature: parseFloat(document.getElementById('temperature').value),
    top_p: parseFloat(document.getElementById('top_p').value),
    top_k: parseInt(document.getElementById('top_k').value),
  };

  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(params)
    });
    const data = await resp.json();
    const meta = data.tokens_generated + ' tokens, ' + data.time_seconds.toFixed(1) + 's, ' +
                 (data.tokens_generated / data.time_seconds).toFixed(1) + ' tok/s';
    addMsg('assistant', data.response, meta);
  } catch(e) {
    addMsg('assistant', 'Error: ' + e.message);
  }
  document.getElementById('send').disabled = false;
  input.focus();
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    start = time.time()
    try:
        response = generate_response(
            prompt=req.message,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
        )
        elapsed = time.time() - start
        # Rough token count
        tok_count = len(response.split()) * 2  # approximate
        return ChatResponse(
            response=response,
            tokens_generated=tok_count,
            time_seconds=elapsed,
        )
    except Exception as e:
        return ChatResponse(
            response=f"Error: {str(e)}",
            tokens_generated=0,
            time_seconds=time.time() - start,
        )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print("=" * 60)
    print("Gemma4-31B-it Chat Server")
    print("=" * 60)

    load_model()

    import uvicorn
    print(f"\nStarting server at http://{args.host}:{args.port}")
    print("Open the URL in your browser to chat!\n")
    uvicorn.run(app, host=args.host, port=args.port)
