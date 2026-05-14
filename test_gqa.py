"""
GQA/MQA (Grouped-Query Attention / Multi-Query Attention) 测试和 Benchmark
"""

import torch
import flash_attn_v100
import time
from tabulate import tabulate


def test_gqa_correctness():
    """测试 GQA 正确性"""
    print("=" * 80)
    print("Testing GQA/MQA Correctness")
    print("=" * 80)
    
    # 测试配置
    test_cases = [
        # (B, H_Q, H_KV, D, cache_len, name)
        (1, 8, 8, 64, 32, "MHA (H_Q=H_KV)"),      # Multi-Head Attention
        (1, 8, 4, 64, 32, "GQA (8Q:4KV)"),         # GQA: 2个Q共享1个KV
        (1, 8, 2, 64, 32, "GQA (8Q:2KV)"),         # GQA: 4个Q共享1个KV
        (1, 8, 1, 64, 32, "MQA (8Q:1KV)"),         # MQA: 所有Q共享1个KV
        (1, 32, 8, 64, 32, "GQA (32Q:8KV)"),       # GQA: 类似 Llama
        (1, 32, 4, 64, 32, "GQA (32Q:4KV)"),       # GQA: 类似 Mistral
        (2, 8, 4, 64, 32, "GQA (Batch=2)"),        # Batch test
    ]
    
    for B, H_Q, H_KV, D, cache_len, name in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print(f"{'='*60}")
        
        total_len = cache_len + 1
        
        # 创建输入
        torch.manual_seed(42)
        q = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16).contiguous()
        k = torch.randn(B, H_KV, total_len, D, device='cuda', dtype=torch.float16).contiguous()
        v = torch.randn(B, H_KV, total_len, D, device='cuda', dtype=torch.float16).contiguous()
        
        try:
            # 运行 GQA kernel
            out_gqa = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)
            
            print(f"  ✓ GQA kernel executed successfully")
            print(f"    Q shape: {q.shape}")
            print(f"    K/V shape: {k.shape}")
            print(f"    Output shape: {out_gqa.shape}")
            
            # 验证输出
            has_nan = torch.isnan(out_gqa).any().item()
            has_inf = torch.isinf(out_gqa).any().item()
            out_min = out_gqa.min().item()
            out_max = out_gqa.max().item()
            
            print(f"    Output range: [{out_min:.4f}, {out_max:.4f}]")
            print(f"    Has NaN: {has_nan}, Has Inf: {has_inf}")
            
            # 验证正确性（与逐头计算对比）
            if B == 1:  # 只验证单 batch
                out_ref = torch.zeros_like(out_gqa)
                for h_q in range(H_Q):
                    h_kv = h_q // (H_Q // H_KV)
                    
                    # 使用标准 decode kernel
                    q_h = q[:, h_q:h_q+1]
                    k_h = k[:, h_kv:h_kv+1]
                    v_h = v[:, h_kv:h_kv+1]
                    
                    out_ref[:, h_q:h_q+1] = flash_attn_v100.forward_decode_fp16(
                        q_h, k_h, v_h, True, cache_len
                    )
                
                max_diff = (out_gqa.float() - out_ref.float()).abs().max().item()
                mean_diff = (out_gqa.float() - out_ref.float()).abs().mean().item()
                all_close = torch.allclose(out_gqa, out_ref, atol=1e-2, rtol=1e-2)
                
                print(f"    Max diff vs reference: {max_diff:.6f}")
                print(f"    Mean diff: {mean_diff:.6f}")
                print(f"    All close: {all_close}")
                
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            import traceback
            traceback.print_exc()


def benchmark_gqa_performance():
    """Benchmark GQA 性能"""
    print("\n" + "=" * 80)
    print("Benchmarking GQA/MQA Performance")
    print("=" * 80)
    
    configs = [
        # (B, H_Q, H_KV, D, cache_len, name)
        (1, 8, 8, 64, 256, "MHA (baseline)"),
        (1, 8, 4, 64, 256, "GQA 8:4"),
        (1, 8, 2, 64, 256, "GQA 8:2"),
        (1, 8, 1, 64, 256, "MQA 8:1"),
        (1, 8, 8, 64, 512, "MHA (len=512)"),
        (1, 8, 4, 64, 512, "GQA 8:4 (len=512)"),
        (1, 8, 1, 64, 512, "MQA 8:1 (len=512)"),
        (1, 32, 8, 64, 512, "GQA 32:8 (Llama-like)"),
        (1, 32, 4, 64, 512, "GQA 32:4 (Mistral-like)"),
        (1, 32, 1, 64, 512, "MQA 32:1"),
    ]
    
    results = []
    
    for B, H_Q, H_KV, D, cache_len, name in configs:
        print(f"\n{'='*60}")
        print(f"Config: {name}")
        print(f"{'='*60}")
        
        total_len = cache_len + 1
        
        # 创建数据
        q = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16).contiguous()
        k = torch.randn(B, H_KV, total_len, D, device='cuda', dtype=torch.float16).contiguous()
        v = torch.randn(B, H_KV, total_len, D, device='cuda', dtype=torch.float16).contiguous()
        
        # Warmup
        for _ in range(10):
            _ = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)
        torch.cuda.synchronize()
        
        # Benchmark
        iterations = 100
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            out = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)
        end.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / iterations
        
        # 计算吞吐量
        tokens_per_sec = B * 1000.0 / elapsed_ms
        
        # 计算 FLOPs (只计算实际执行的)
        # QK^T: B * H_Q * 1 * N * D
        # PV: B * H_Q * 1 * N * D
        # 总计: 2 * B * H_Q * N * D
        flops = 2 * B * H_Q * total_len * D
        time_s = elapsed_ms / 1000.0
        gflops = flops / time_s / 1e9
        
        # 内存带宽
        bytes_per_element = 2  # FP16
        # Q: B*H_Q*1*D, K/V: B*H_KV*N*D, O: B*H_Q*1*D
        total_bytes = (B * H_Q * 1 * D + 2 * B * H_KV * total_len * D + 
                      B * H_Q * 1 * D) * bytes_per_element
        bandwidth_gb_s = total_bytes / time_s / 1e9
        
        # KV cache 内存节省
        kv_reduction = (1 - H_KV / H_Q) * 100
        
        print(f"  Time: {elapsed_ms:.3f} ms")
        print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
        print(f"  GFLOPs: {gflops:.2f}")
        print(f"  Memory Bandwidth: {bandwidth_gb_s:.2f} GB/s")
        print(f"  KV Cache Reduction: {kv_reduction:.1f}%")
        
        results.append({
            'Config': name,
            'H_Q': H_Q,
            'H_KV': H_KV,
            'Cache': cache_len,
            'Time (ms)': f"{elapsed_ms:.3f}",
            'Tokens/sec': f"{tokens_per_sec:.1f}",
            'GFLOPs': f"{gflops:.2f}",
            'BW (GB/s)': f"{bandwidth_gb_s:.2f}",
            'KV Save': f"{kv_reduction:.1f}%"
        })
        
        torch.cuda.empty_cache()
    
    # 打印汇总表
    print("\n" + "=" * 80)
    print("GQA/MQA Benchmark Summary")
    print("=" * 80)
    
    headers = ['Config', 'H_Q', 'H_KV', 'Cache', 'Time (ms)', 'Tokens/sec', 'GFLOPs', 'BW (GB/s)', 'KV Save']
    table_data = []
    
    for r in results:
        table_data.append([
            r['Config'], r['H_Q'], r['H_KV'], r['Cache'],
            r['Time (ms)'], r['Tokens/sec'], r['GFLOPs'], 
            r['BW (GB/s)'], r['KV Save']
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))


def demo_gqa_usage():
    """演示 GQA 使用"""
    print("\n" + "=" * 80)
    print("GQA/MQA Usage Demo")
    print("=" * 80)
    
    # 模拟 Llama 3 配置
    print("\n1. Llama 3-like Configuration (GQA)")
    print("-" * 60)
    
    B = 1
    H_Q = 32       # 32 个 Q heads
    H_KV = 8       # 8 个 KV heads (每 4 个 Q 共享 1 个 KV)
    D = 128        # Head dimension
    cache_len = 100
    
    total_len = cache_len + 1
    
    q = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16).contiguous()
    k = torch.randn(B, H_KV, total_len, D, device='cuda', dtype=torch.float16).contiguous()
    v = torch.randn(B, H_KV, total_len, D, device='cuda', dtype=torch.float16).contiguous()
    
    print(f"  Q heads: {H_Q}")
    print(f"  KV heads: {H_KV}")
    print(f"  Q per KV: {H_Q // H_KV}")
    print(f"  Q shape: {q.shape}")
    print(f"  K/V shape: {k.shape}")
    
    out = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ Success!")
    
    # 模拟 Mistral 配置
    print("\n2. Mistral-like Configuration (GQA)")
    print("-" * 60)
    
    H_Q = 32
    H_KV = 8
    D = 128
    
    q = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16).contiguous()
    k = torch.randn(B, H_KV, total_len, D, device='cuda', dtype=torch.float16).contiguous()
    v = torch.randn(B, H_KV, total_len, D, device='cuda', dtype=torch.float16).contiguous()
    
    print(f"  Q heads: {H_Q}")
    print(f"  KV heads: {H_KV}")
    print(f"  Q per KV: {H_Q // H_KV}")
    
    out = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ Success!")
    
    # 模拟 MQA 配置
    print("\n3. MQA Configuration (Extreme GQA)")
    print("-" * 60)
    
    H_Q = 16
    H_KV = 1  # 所有 Q 共享同一个 KV
    
    q = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16).contiguous()
    k = torch.randn(B, H_KV, total_len, D, device='cuda', dtype=torch.float16).contiguous()
    v = torch.randn(B, H_KV, total_len, D, device='cuda', dtype=torch.float16).contiguous()
    
    print(f"  Q heads: {H_Q}")
    print(f"  KV heads: {H_KV}")
    print(f"  Q per KV: {H_Q}")
    
    out = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ Success!")
    
    # KV Cache 内存对比
    print("\n4. KV Cache Memory Comparison")
    print("-" * 60)
    
    seq_len = 2048
    configs = [
        ("MHA (H_Q=H_KV=32)", 32, 32),
        ("GQA (32Q:8KV)", 32, 8),
        ("GQA (32Q:4KV)", 32, 4),
        ("MQA (32Q:1KV)", 32, 1),
    ]
    
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dimension: {D}")
    print(f"  Batch size: {B}")
    print()
    
    for name, h_q, h_kv in configs:
        # KV cache 大小 (bytes)
        kv_bytes = B * h_kv * seq_len * D * 2  # FP16 = 2 bytes
        kv_mb = kv_bytes / (1024 * 1024)
        reduction = (1 - h_kv / 32) * 100
        
        print(f"  {name:25s}: {kv_mb:8.2f} MB  (saved {reduction:5.1f}%)")


if __name__ == '__main__':
    # 测试正确性
    test_gqa_correctness()
    
    # 性能 benchmark
    benchmark_gqa_performance()
    
    # 使用示例
    demo_gqa_usage()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
