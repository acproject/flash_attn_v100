"""
自回归生成示例 - 使用 KV Cache / Decode Kernel
演示如何在实际的 LLM 推理中使用优化的 decode kernel
"""

import torch
import flash_attn_v100
import time

from tensor_parallel import TPDecodeRunner, TensorParallelConfig


class KVCacheManager:
    """KV Cache 管理器"""
    
    def __init__(self, max_seq_len, num_heads, head_dim, batch_size=1, device='cuda'):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.device = device
        
        # 预分配 KV cache 内存
        self.cache_k = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=torch.float16
        )
        self.cache_v = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=torch.float16
        )
        self.current_len = 0
    
    def reset(self):
        """重置 cache"""
        self.current_len = 0
    
    def append(self, k, v):
        """
        追加新的 K/V 到 cache
        
        Args:
            k: [B, H, 1, D] 当前 token 的 K
            v: [B, H, 1, D] 当前 token 的 V
        """
        append_len = k.size(2)
        if self.current_len + append_len > self.max_seq_len:
            raise RuntimeError("KV cache is full")
        
        self.cache_k[:, :, self.current_len:self.current_len + append_len] = k
        self.cache_v[:, :, self.current_len:self.current_len + append_len] = v
        self.current_len += append_len
    
    def get_cache(self):
        """
        获取当前的 K/V cache
        
        Returns:
            k: [B, H, current_len, D]
            v: [B, H, current_len, D]
        """
        return (
            self.cache_k[:, :, :self.current_len].contiguous(),
            self.cache_v[:, :, :self.current_len].contiguous()
        )


class SimpleTransformerDecoder:
    """简化的 Transformer Decoder（单头注意力示例）"""
    
    def __init__(self, vocab_size, embed_dim, num_heads, head_dim, device='cuda'):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        
        # 简化的模型参数（实际应该使用预训练权重）
        self.token_embedding = torch.randn(vocab_size, embed_dim, device=device, dtype=torch.float16)
        self.W_q = torch.randn(embed_dim, num_heads * head_dim, device=device, dtype=torch.float16)
        self.W_k = torch.randn(embed_dim, num_heads * head_dim, device=device, dtype=torch.float16)
        self.W_v = torch.randn(embed_dim, num_heads * head_dim, device=device, dtype=torch.float16)
        self.W_o = torch.randn(num_heads * head_dim, embed_dim, device=device, dtype=torch.float16)

    def eval(self):
        return self
    
    def embed_tokens(self, token_ids):
        """Token embedding"""
        return self.token_embedding[token_ids]
    
    def compute_qkv(self, x, _is_prompt=True):
        """
        计算 Q, K, V
        
        Args:
            x: [B, seq_len, embed_dim]
            is_prompt: 是否为 prompt 阶段
        
        Returns:
            q, k, v: [B, H, seq_len, D]
        """
        # 线性变换
        q = torch.matmul(x, self.W_q)
        k = torch.matmul(x, self.W_k)
        v = torch.matmul(x, self.W_v)
        
        # reshape 为 [B, seq_len, H, D] -> [B, H, seq_len, D]
        B, seq_len, _ = q.shape
        q = q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        
        return q, k, v
    
    def attention(self, q, k, v, _cache, causal=True):
        """
        使用 Flash Attention 计算注意力
        
        Args:
            q: [B, H, q_len, D]
            k: [B, H, k_len, D]
            v: [B, H, k_len, D]
            cache: KVCacheManager
            causal: 是否使用 causal mask
        
        Returns:
            output: [B, H, q_len, D]
        """
        _, _, q_len, _ = q.shape
        k_len = k.shape[2]
        
        if q_len == 1:
            # Decode 模式：使用优化的 decode kernel
            cache_len = k_len - 1
            output = flash_attn_v100.forward_decode_fp16(q, k, v, causal, cache_len)
        else:
            # Prefill 模式：使用标准 kernel
            output = flash_attn_v100.forward_fp16(q, k, v, causal)
        
        return output
    
    def forward(self, x, cache, is_decode=False):
        """
        前向传播
        
        Args:
            x: [B, seq_len, embed_dim]
            cache: KVCacheManager
            is_decode: 是否为 decode 模式
        
        Returns:
            output: [B, seq_len, embed_dim]
        """
        # 计算 Q, K, V
        q, k, v = self.compute_qkv(x)
        
        if is_decode:
            # Decode 模式：追加到 cache
            cache.append(k, v)
            k_cache, v_cache = cache.get_cache()
            
            # 只取最后一个 token 的 Q
            q_last = q[:, :, -1:, :]
            
            # 使用 decode kernel
            attn_out = self.attention(q_last, k_cache, v_cache, cache, causal=True)
        else:
            # Prefill 模式：一次性处理所有 token
            cache.append(k, v)  # 将整个序列加入 cache
            k_cache, v_cache = cache.get_cache()
            
            # 使用标准 kernel
            attn_out = self.attention(q, k_cache, v_cache, cache, causal=True)
        
        # 输出投影
        B, H, seq_len, D = attn_out.shape
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_len, H * D)
        output = torch.matmul(attn_out, self.W_o)
        
        return output


def generate_text(model, prompt_tokens, max_new_tokens, cache, temperature=1.0, top_k=50):
    """
    自回归文本生成
    
    Args:
        model: SimpleTransformerDecoder
        prompt_tokens: [seq_len] 提示 token IDs
        max_new_tokens: 最大生成 token 数
        cache: KVCacheManager
        temperature: 温度参数
        top_k: top-k 采样
    
    Returns:
        generated_tokens: [seq_len + new_tokens]
    """
    model.eval()
    cache.reset()
    _ = temperature, top_k
    
    # Prefill 阶段：处理 prompt
    print("Prefill stage...")
    prompt_embeds = model.embed_tokens(prompt_tokens.unsqueeze(0))
    
    with torch.no_grad():
        _ = model.forward(prompt_embeds, cache, is_decode=False)
    
    # 获取最后一个 token 的 logits（简化处理）
    generated = prompt_tokens.clone()
    
    # Decode 阶段：逐个生成新 token
    print(f"Decode stage (generating {max_new_tokens} tokens)...")
    start_time = time.time()
    
    for i in range(max_new_tokens):
        # 使用最新的 token 生成下一个 token
        last_token = generated[-1].unsqueeze(0).unsqueeze(0)  # [1, 1]
        last_embed = model.embed_tokens(last_token)
        
        with torch.no_grad():
            _ = model.forward(last_embed, cache, is_decode=True)
        
        # 简化的采样（实际应该使用 linear + softmax）
        # 这里随机采样作为示例
        next_token = torch.randint(0, model.vocab_size, (1,), device=model.device)
        generated = torch.cat([generated, next_token])
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Generated {i+1}/{max_new_tokens} tokens ({elapsed:.2f}s)")
    
    total_time = time.time() - start_time
    print(f"\nDecode completed: {max_new_tokens} tokens in {total_time:.2f}s")
    print(f"Throughput: {max_new_tokens/total_time:.2f} tokens/sec")
    
    return generated


def benchmark_decode_performance():
    """Benchmark decode kernel 在自回归场景的性能"""
    print("=" * 80)
    print("Benchmark: Autoregressive Decoding with KV Cache")
    print("=" * 80)
    
    # 配置
    B = 1
    H = 8
    D = 64
    embed_dim = H * D
    vocab_size = 1000
    
    # 创建模型和 cache
    cache = KVCacheManager(
        max_seq_len=2048,
        num_heads=H,
        head_dim=D,
        batch_size=B
    )
    
    model = SimpleTransformerDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=H,
        head_dim=D
    )
    
    # 测试不同 cache 长度下的性能
    prompt_len = 128
    gen_len = 100
    
    print(f"\nConfig:")
    print(f"  Batch size: {B}")
    print(f"  Num heads: {H}")
    print(f"  Head dim: {D}")
    print(f"  Prompt length: {prompt_len}")
    print(f"  Generation length: {gen_len}")
    
    # 创建随机 prompt
    prompt_tokens = torch.randint(0, vocab_size, (prompt_len,), device='cuda')
    
    # 生成
    print("\n" + "-" * 80)
    generated = generate_text(
        model, 
        prompt_tokens, 
        gen_len, 
        cache,
        temperature=0.7
    )
    
    print("\nGenerated token IDs (first 50):")
    print(generated[:50].tolist())


def demo_kv_cache_usage():
    """演示 KV Cache 的基本使用"""
    print("=" * 80)
    print("Demo: KV Cache Usage")
    print("=" * 80)
    
    B, H, D = 1, 4, 64
    
    # 创建 KV cache
    cache = KVCacheManager(
        max_seq_len=1024,
        num_heads=H,
        head_dim=D,
        batch_size=B
    )
    
    # 模拟自回归生成
    print("\nSimulating autoregressive generation...")
    
    for step in range(10):
        # 生成当前 token 的 K/V
        k = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16)
        
        # 追加到 cache
        cache.append(k, v)
        
        # 获取当前 cache
        k_cache, v_cache = cache.get_cache()
        
        # 生成当前 token 的 Q
        q = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16)
        
        # 使用 decode kernel
        cache_len = cache.current_len - 1
        out = flash_attn_v100.forward_decode_fp16(
            q, k_cache, v_cache, True, cache_len
        )
        
        if step % 3 == 0:
            print(f"  Step {step}: cache_len={cache.current_len}, output shape={out.shape}")
    
    print(f"\nFinal cache length: {cache.current_len}")


def demo_tensor_parallel_kv_cache_usage():
    """演示 2-GPU Tensor Parallel KV cache 生命周期"""
    print("\n" + "=" * 80)
    print("Demo: Tensor Parallel KV Cache Lifecycle")
    print("=" * 80)

    if torch.cuda.device_count() < 2:
        print("Skip: need at least 2 GPUs")
        return

    B, H_Q, H_KV, D = 1, 16, 4, 64
    prompt_len = 32
    gen_steps = 6
    max_seq_len = prompt_len + gen_steps + 8

    runner = TPDecodeRunner(
        TensorParallelConfig(H_Q=H_Q, H_KV=H_KV, D=D, world_size=2, causal=True)
    )
    kv_cache = runner.create_kv_cache(B, max_seq_len)

    prompt_q = torch.randn(B, H_Q, prompt_len, D, device='cuda:0', dtype=torch.float16)
    prompt_k = torch.randn(B, H_KV, prompt_len, D, device='cuda:0', dtype=torch.float16)
    prompt_v = torch.randn(B, H_KV, prompt_len, D, device='cuda:0', dtype=torch.float16)

    print(f"Prefill prompt length: {prompt_len}")
    prefill_out = runner.prefill_to_kv_cache(prompt_q, prompt_k, prompt_v, kv_cache, return_output=True)
    print(f"  Prefill output shape: {prefill_out.shape}")
    print(f"  Cache length after prefill: {kv_cache.current_len}")

    history_k = prompt_k.clone()
    history_v = prompt_v.clone()
    start = time.time()

    for step in range(gen_steps):
        q_step = torch.randn(B, H_Q, 1, D, device='cuda:0', dtype=torch.float16)
        k_step = torch.randn(B, H_KV, 1, D, device='cuda:0', dtype=torch.float16)
        v_step = torch.randn(B, H_KV, 1, D, device='cuda:0', dtype=torch.float16)

        out_tp = runner.decode_with_kv_cache(q_step, kv_cache, new_k=k_step, new_v=v_step)
        history_k = torch.cat([history_k, k_step], dim=2)
        history_v = torch.cat([history_v, v_step], dim=2)
        out_ref = flash_attn_v100.forward_decode_gqa_fp16(
            q_step, history_k, history_v, True, history_k.size(2) - 1
        )
        diff = (out_tp.float() - out_ref.float()).abs().max().item()
        print(f"  Step {step}: cache_len={kv_cache.current_len}, max_diff={diff:.8f}")

    elapsed = time.time() - start
    print(f"Generated {gen_steps} decode steps in {elapsed*1000:.2f} ms")
    print(f"Average latency: {elapsed / gen_steps * 1000:.2f} ms/token")
    print(f"Cache metadata before reset: {runner.get_kv_cache_metadata(kv_cache)}")
    print(f"Runtime stats: {runner.get_runtime_stats()}")

    runner.reset_kv_cache(kv_cache)
    print(f"Cache length after reset: {kv_cache.current_len}")


if __name__ == '__main__':
    # 演示 KV Cache 使用
    demo_kv_cache_usage()
    demo_tensor_parallel_kv_cache_usage()
    
    # Benchmark 性能
    benchmark_decode_performance()
