"""
KV Cache / Decode Kernel 使用示例
展示如何在自回归推理中使用优化的 decode kernel
"""

import torch
import flash_attn_v100
import time


def simple_decode_example():
    """简单的 decode 示例"""
    print("=" * 80)
    print("Simple Decode Example")
    print("=" * 80)
    
    B, H, D = 1, 8, 64
    cache_len = 10
    total_len = cache_len + 1  # 11
    
    # 创建输入
    q = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16).contiguous()
    k = torch.randn(B, H, total_len, D, device='cuda', dtype=torch.float16).contiguous()
    v = torch.randn(B, H, total_len, D, device='cuda', dtype=torch.float16).contiguous()
    
    print(f"Input shapes:")
    print(f"  Q: {q.shape}")
    print(f"  K: {k.shape}")
    print(f"  V: {v.shape}")
    
    # 使用 decode kernel
    out = flash_attn_v100.forward_decode_fp16(q, k, v, True, cache_len)
    
    print(f"\nOutput shape: {out.shape}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    print(f"Has NaN: {torch.isnan(out).any().item()}")
    print("✓ Success!")


def autoregressive_simulation():
    """模拟自回归生成过程"""
    print("\n" + "=" * 80)
    print("Autoregressive Generation Simulation")
    print("=" * 80)
    
    B, H, D = 1, 8, 64
    max_seq_len = 20
    num_steps = 10
    
    # 预分配 KV cache
    cache_k = torch.zeros(B, H, max_seq_len, D, device='cuda', dtype=torch.float16)
    cache_v = torch.zeros(B, H, max_seq_len, D, device='cuda', dtype=torch.float16)
    current_len = 0
    
    print(f"Config: B={B}, H={H}, D={D}, max_seq_len={max_seq_len}")
    print(f"Generating {num_steps} tokens...\n")
    
    start_time = time.time()
    
    for step in range(num_steps):
        # 生成当前 token 的 Q, K, V
        q = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16).contiguous()
        k_new = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16).contiguous()
        v_new = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16).contiguous()
        
        # 追加到 cache
        cache_k[:, :, current_len:current_len+1] = k_new
        cache_v[:, :, current_len:current_len+1] = v_new
        current_len += 1
        
        # 获取当前的 K/V cache
        k_cache = cache_k[:, :, :current_len].contiguous()
        v_cache = cache_v[:, :, :current_len].contiguous()
        
        # 使用 decode kernel 计算 attention
        cache_len = current_len - 1
        out = flash_attn_v100.forward_decode_fp16(
            q, k_cache, v_cache,
            True,
            cache_len
        )
        
        if step % 3 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:2d}: cache_len={current_len:2d}, "
                  f"time={elapsed*1000:.2f}ms, "
                  f"out_range=[{out.min().item():.3f}, {out.max().item():.3f}]")
    
    total_time = time.time() - start_time
    print(f"\nCompleted {num_steps} steps in {total_time*1000:.2f}ms")
    print(f"Average: {total_time/num_steps*1000:.2f}ms per token")
    print(f"Throughput: {num_steps/total_time:.1f} tokens/sec")


def batch_decode_example():
    """Batch decode 示例"""
    print("\n" + "=" * 80)
    print("Batch Decode Example")
    print("=" * 80)
    
    B, H, D = 4, 8, 64  # Batch size = 4
    cache_len = 50
    total_len = cache_len + 1
    
    # 创建 batch 输入
    q = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16).contiguous()
    k = torch.randn(B, H, total_len, D, device='cuda', dtype=torch.float16).contiguous()
    v = torch.randn(B, H, total_len, D, device='cuda', dtype=torch.float16).contiguous()
    
    print(f"Input shapes:")
    print(f"  Q: {q.shape}")
    print(f"  K: {k.shape}")
    print(f"  V: {v.shape}")
    
    # Benchmark
    warmup = 10
    iterations = 50
    
    for _ in range(warmup):
        _ = flash_attn_v100.forward_decode_fp16(q, k, v, True, cache_len)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        out = flash_attn_v100.forward_decode_fp16(q, k, v, True, cache_len)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / iterations
    
    print(f"\nPerformance (B={B}):")
    print(f"  Time: {elapsed_ms:.3f} ms")
    print(f"  Throughput: {B * 1000 / elapsed_ms:.1f} tokens/sec")
    print(f"  Output shape: {out.shape}")


def compare_causal_vs_noncausal():
    """对比 causal 和 non-causal 模式"""
    print("\n" + "=" * 80)
    print("Causal vs Non-Causal Comparison")
    print("=" * 80)
    
    B, H, D = 1, 4, 64
    cache_len = 20
    total_len = cache_len + 1
    
    torch.manual_seed(42)
    q = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16).contiguous()
    k = torch.randn(B, H, total_len, D, device='cuda', dtype=torch.float16).contiguous()
    v = torch.randn(B, H, total_len, D, device='cuda', dtype=torch.float16).contiguous()
    
    # Causal 模式
    out_causal = flash_attn_v100.forward_decode_fp16(q, k, v, True, cache_len)
    
    # Non-causal 模式
    out_noncausal = flash_attn_v100.forward_decode_fp16(q, k, v, False, cache_len)
    
    print(f"Causal output (first 5 values):")
    print(f"  {out_causal[0, 0, 0, :5].tolist()}")
    
    print(f"\nNon-causal output (first 5 values):")
    print(f"  {out_noncausal[0, 0, 0, :5].tolist()}")
    
    diff = (out_causal.float() - out_noncausal.float()).abs().max().item()
    print(f"\nMax difference: {diff:.6f}")
    print(f"Outputs are {'same' if diff < 1e-5 else 'different'}")


if __name__ == '__main__':
    # 运行所有示例
    simple_decode_example()
    autoregressive_simulation()
    batch_decode_example()
    compare_causal_vs_noncausal()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
