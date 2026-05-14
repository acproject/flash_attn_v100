"""
GQA/MQA 优势可视化
展示 GQA 相比 MHA 的内存和性能优势
"""

import torch
import flash_attn_v100
import time


def visualize_kv_cache_savings():
    """可视化 KV Cache 内存节省"""
    print("=" * 80)
    print("KV Cache Memory Savings Visualization")
    print("=" * 80)
    
    # 配置
    B = 1
    seq_len = 2048
    D = 128
    
    configs = [
        ("MHA (Traditional)", 32, 32),
        ("GQA (Llama 3)", 32, 8),
        ("GQA (Mistral)", 32, 8),
        ("GQA (Aggressive)", 32, 4),
        ("MQA (Extreme)", 32, 1),
    ]
    
    print(f"\nParameters:")
    print(f"  Batch Size: {B}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Head Dimension: {D}")
    print(f"  Data Type: FP16 (2 bytes)")
    
    print(f"\n{'='*70}")
    print(f"{'Configuration':25s} | {'H_Q':4s} | {'H_KV':4s} | {'KV Cache':10s} | {'Savings':10s}")
    print(f"{'-'*70}")
    
    mha_memory = None
    
    for name, h_q, h_kv in configs:
        # 计算 KV cache 大小 (bytes)
        # K 和 V 各需要: B * H_KV * seq_len * D * 2 bytes
        kv_memory = B * h_kv * seq_len * D * 2 * 2  # K + V
        kv_memory_mb = kv_memory / (1024 * 1024)
        
        if mha_memory is None:
            mha_memory = kv_memory_mb
            savings = 0.0
        else:
            savings = (1 - kv_memory_mb / mha_memory) * 100
        
        # 可视化条
        bar_len = int(savings / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        
        print(f"{name:25s} | {h_q:4d} | {h_kv:4d} | {kv_memory_mb:8.2f} MB | {bar} {savings:5.1f}%")
    
    print(f"{'='*70}")
    
    print(f"\n💡 Key Insights:")
    print(f"  • GQA reduces KV cache by 75% compared to MHA")
    print(f"  • MQA reduces KV cache by 97% compared to MHA")
    print(f"  • This allows 4x larger batch size or sequence length")


def visualize_performance_comparison():
    """可视化性能对比"""
    print("\n" + "=" * 80)
    print("Performance Comparison")
    print("=" * 80)
    
    # 测试配置
    B = 1
    cache_len = 512
    D = 64
    
    configs = [
        ("MHA (8:8)", 8, 8),
        ("GQA (8:4)", 8, 4),
        ("GQA (8:2)", 8, 2),
        ("MQA (8:1)", 8, 1),
        ("GQA (32:8) - Llama 3", 32, 8),
        ("MQA (32:1)", 32, 1),
    ]
    
    results = []
    
    for name, h_q, h_kv in configs:
        total_len = cache_len + 1
        
        q = torch.randn(B, h_q, 1, D, device='cuda', dtype=torch.float16).contiguous()
        k = torch.randn(B, h_kv, total_len, D, device='cuda', dtype=torch.float16).contiguous()
        v = torch.randn(B, h_kv, total_len, D, device='cuda', dtype=torch.float16).contiguous()
        
        # Benchmark
        for _ in range(10):
            _ = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)
        torch.cuda.synchronize()
        
        iterations = 50
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            out = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)
        end.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / iterations
        tokens_per_sec = B * 1000.0 / elapsed_ms
        
        results.append((name, h_q, h_kv, elapsed_ms, tokens_per_sec))
    
    # 打印表格
    print(f"\n{'='*75}")
    print(f"{'Configuration':25s} | {'H_Q':4s} | {'H_KV':4s} | {'Time (ms)':10s} | {'Tokens/sec':12s}")
    print(f"{'-'*75}")
    
    baseline_time = results[0][3]
    
    for name, h_q, h_kv, time_ms, tokens in results:
        speedup = baseline_time / time_ms
        print(f"{name:25s} | {h_q:4d} | {h_kv:4d} | {time_ms:8.3f} ms | {tokens:10.1f} t/s")
    
    print(f"{'='*75}")
    
    print(f"\n💡 Key Insights:")
    print(f"  • GQA/MQA achieves similar or better throughput than MHA")
    print(f"  • Memory bandwidth is significantly reduced")
    print(f"  • Overall system throughput increases due to larger batch sizes")


def visualize_head_mapping():
    """可视化 Head 映射"""
    print("\n" + "=" * 80)
    print("GQA Head Mapping Visualization")
    print("=" * 80)
    
    configs = [
        ("MHA (8:8)", 8, 8),
        ("GQA (8:4)", 8, 4),
        ("GQA (8:2)", 8, 2),
        ("MQA (8:1)", 8, 1),
        ("GQA (32:8) - Llama 3", 32, 8),
    ]
    
    for name, h_q, h_kv in configs:
        print(f"\n{name}")
        print(f"{'-'*60}")
        
        group_size = h_q // h_kv
        
        if h_q <= 16:
            # 打印完整映射
            for h_q_idx in range(h_q):
                h_kv_idx = h_q_idx // group_size
                print(f"  Q head {h_q_idx:2d} ──→ KV head {h_kv_idx}")
        else:
            # 打印摘要
            print(f"  Q heads 0-{group_size-1:2d}        ──→ KV head 0")
            print(f"  Q heads {group_size}-{2*group_size-1:2d}      ──→ KV head 1")
            print(f"  ...")
            print(f"  Q heads {h_q-group_size}-{h_q-1:2d}  ──→ KV head {h_kv-1}")
        
        print(f"  (Each KV head shared by {group_size} Q heads)")


def visualize_real_world_impact():
    """可视化实际影响"""
    print("\n" + "=" * 80)
    print("Real-World Impact on LLM Inference")
    print("=" * 80)
    
    print("\nScenario: Llama 3 8B Inference")
    print(f"{'='*60}")
    print(f"  Model: Llama 3 8B")
    print(f"  Heads: 32 Q, 8 KV (GQA)")
    print(f"  Head Dim: 128")
    print(f"  Sequence Length: 4096")
    print()
    
    # 计算不同 batch size 的内存
    batch_sizes = [1, 2, 4, 8, 16]
    
    print(f"  {'Batch Size':12s} | {'MHA KV Cache':15s} | {'GQA KV Cache':15s} | {'Savings':10s}")
    print(f"  {'-'*60}")
    
    for B in batch_sizes:
        # MHA: 32 heads
        mha_kv = B * 32 * 4096 * 128 * 2 * 2 / (1024 * 1024)  # MB
        # GQA: 8 heads
        gqa_kv = B * 8 * 4096 * 128 * 2 * 2 / (1024 * 1024)  # MB
        savings = (1 - gqa_kv / mha_kv) * 100
        
        print(f"  {B:12d} | {mha_kv:12.2f} MB | {gqa_kv:12.2f} MB | {savings:8.1f}%")
    
    print(f"\n  💡 With GQA, you can:")
    print(f"     • Process 4x larger batch size with same memory")
    print(f"     • Support 4x longer sequences")
    print(f"     • Reduce GPU memory requirements by 75%")
    
    print(f"\n{'='*60}")
    print(f"\nScenario: Serving Multiple Users")
    print(f"{'='*60}")
    
    print(f"  GPU: A100 80GB")
    print(f"  Model: Llama 3 8B")
    print(f"  Sequence Length: 2048")
    print()
    
    # MHA 可以服务的用户数
    gpu_memory_gb = 80
    model_weights_gb = 16  # 8B 模型约 16GB
    available_gb = gpu_memory_gb - model_weights_gb
    
    # 每个用户的 KV cache
    mha_per_user_mb = 1 * 32 * 2048 * 128 * 2 * 2 / (1024 * 1024)
    gqa_per_user_mb = 1 * 8 * 2048 * 128 * 2 * 2 / (1024 * 1024)
    
    mha_users = int(available_gb * 1024 / mha_per_user_mb)
    gqa_users = int(available_gb * 1024 / gqa_per_user_mb)
    
    print(f"  Available GPU Memory: {available_gb} GB")
    print(f"  KV Cache per User (MHA): {mha_per_user_mb:.2f} MB")
    print(f"  KV Cache per User (GQA): {gqa_per_user_mb:.2f} MB")
    print()
    print(f"  Concurrent Users (MHA): {mha_users}")
    print(f"  Concurrent Users (GQA): {gqa_users}")
    print(f"  Improvement: {gqa_users/mha_users:.1f}x more users!")


if __name__ == '__main__':
    visualize_kv_cache_savings()
    visualize_performance_comparison()
    visualize_head_mapping()
    visualize_real_world_impact()
    
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
