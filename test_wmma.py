"""
WMMA Tensor Core 优化版本测试和 Benchmark
"""

import torch
import flash_attn_v100
import time
from tabulate import tabulate


def reference_attention(q, k, v, causal=True):
    """PyTorch 原生 Attention 实现（参考标准）"""
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)
    
    if causal:
        n = q.size(-2)
        mask = torch.triu(torch.ones(n, n, device=q.device, dtype=q.dtype), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float('-inf'))
    
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def test_wmma_correctness():
    """验证 WMMA 版本的正确性"""
    print("=" * 80)
    print("Testing WMMA Correctness")
    print("=" * 80)
    
    test_cases = [
        (1, 8, 128, 64, "Small"),
        (1, 8, 256, 64, "Medium"),
        (2, 8, 256, 64, "Batch=2"),
        (1, 8, 512, 64, "Long seq"),
        (1, 8, 256, 128, "D=128"),
        (2, 8, 512, 128, "Large"),
    ]
    
    for B, H, N, D, name in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {name} (B={B}, H={H}, N={N}, D={D})")
        print(f"{'='*60}")
        
        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16).contiguous()
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16).contiguous()
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16).contiguous()
        
        try:
            # 参考实现 (FP16 baseline)
            out_ref = flash_attn_v100.forward_fp16(q, k, v, True)
            
            # WMMA 实现
            out_wmma = flash_attn_v100.forward_fp16_wmma(q, k, v, True)
            
            # 验证正确性
            max_diff = (out_wmma.float() - out_ref.float()).abs().max().item()
            mean_diff = (out_wmma.float() - out_ref.float()).abs().mean().item()
            all_close = torch.allclose(out_wmma, out_ref, atol=1e-2, rtol=1e-2)
            
            # 检查 NaN/Inf
            has_nan = torch.isnan(out_wmma).any().item()
            has_inf = torch.isinf(out_wmma).any().item()
            
            print(f"  ✓ WMMA kernel executed successfully")
            print(f"    Max diff: {max_diff:.6f}")
            print(f"    Mean diff: {mean_diff:.6f}")
            print(f"    All close (atol=1e-2): {all_close}")
            print(f"    Has NaN: {has_nan}, Has Inf: {has_inf}")
            
            if not all_close:
                print(f"    ⚠ Warning: Results differ from baseline!")
                
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            import traceback
            traceback.print_exc()


def benchmark_wmma_performance():
    """Benchmark WMMA 性能"""
    print("\n" + "=" * 80)
    print("Benchmarking WMMA Performance")
    print("=" * 80)
    
    configs = [
        (1, 8, 128, 64),
        (1, 8, 256, 64),
        (1, 8, 512, 64),
        (1, 8, 1024, 64),
        (1, 8, 2048, 64),
        (2, 8, 256, 64),
        (2, 8, 512, 64),
        (2, 8, 1024, 64),
        (1, 8, 256, 128),
        (1, 8, 512, 128),
        (1, 8, 1024, 128),
        (2, 8, 256, 128),
        (2, 8, 512, 128),
        (2, 8, 1024, 128),
    ]
    
    results = []
    
    for B, H, N, D in configs:
        print(f"\n{'='*60}")
        print(f"Config: B={B}, H={H}, N={N}, D={D}")
        print(f"{'='*60}")
        
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16).contiguous()
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16).contiguous()
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16).contiguous()
        
        # Benchmark FP16 baseline
        warmup = 10
        iterations = 100
        
        for _ in range(warmup):
            _ = flash_attn_v100.forward_fp16(q, k, v, True)
        torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            _ = flash_attn_v100.forward_fp16(q, k, v, True)
        end.record()
        
        torch.cuda.synchronize()
        time_fp16 = start.elapsed_time(end) / iterations
        
        # Benchmark WMMA
        for _ in range(warmup):
            _ = flash_attn_v100.forward_fp16_wmma(q, k, v, True)
        torch.cuda.synchronize()
        
        start.record()
        for _ in range(iterations):
            _ = flash_attn_v100.forward_fp16_wmma(q, k, v, True)
        end.record()
        
        torch.cuda.synchronize()
        time_wmma = start.elapsed_time(end) / iterations
        
        # 计算加速比
        speedup = time_fp16 / time_wmma if time_wmma > 0 else 0
        
        # 计算 TFLOPs
        flops = 2 * B * H * N * N * D
        tflops_fp16 = flops / (time_fp16 / 1000.0) / 1e12
        tflops_wmma = flops / (time_wmma / 1000.0) / 1e12
        
        print(f"  FP16 Baseline: {time_fp16:.3f} ms, {tflops_fp16:.2f} TFLOPs")
        print(f"  WMMA:          {time_wmma:.3f} ms, {tflops_wmma:.2f} TFLOPs")
        print(f"  Speedup:       {speedup:.2f}x")
        
        results.append({
            'B': B, 'H': H, 'N': N, 'D': D,
            'Time FP16 (ms)': f"{time_fp16:.3f}",
            'Time WMMA (ms)': f"{time_wmma:.3f}",
            'Speedup': f"{speedup:.2f}x",
            'TFLOPs FP16': f"{tflops_fp16:.2f}",
            'TFLOPs WMMA': f"{tflops_wmma:.2f}"
        })
        
        torch.cuda.empty_cache()
    
    # 打印汇总表
    print("\n" + "=" * 80)
    print("WMMA Benchmark Summary")
    print("=" * 80)
    
    headers = ['B', 'H', 'N', 'D', 'Time FP16 (ms)', 'Time WMMA (ms)', 'Speedup', 'TFLOPs FP16', 'TFLOPs WMMA']
    table_data = []
    
    for r in results:
        table_data.append([
            r['B'], r['H'], r['N'], r['D'],
            r['Time FP16 (ms)'], r['Time WMMA (ms)'],
            r['Speedup'], r['TFLOPs FP16'], r['TFLOPs WMMA']
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # 保存结果
    with open('benchmark_wmma_results.txt', 'w') as f:
        f.write("WMMA Tensor Core Benchmark Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    print("\nResults saved to benchmark_wmma_results.txt")


def demo_wmma_usage():
    """演示 WMMA 使用"""
    print("\n" + "=" * 80)
    print("WMMA Usage Demo")
    print("=" * 80)
    
    B, H, N, D = 2, 8, 512, 64
    
    print(f"\nConfig: B={B}, H={H}, N={N}, D={D}")
    
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16).contiguous()
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16).contiguous()
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16).contiguous()
    
    # 测试 causal 和 non-causal
    for causal in [True, False]:
        causal_str = "Causal" if causal else "Non-Causal"
        print(f"\n{causal_str} Mode:")
        
        out = flash_attn_v100.forward_fp16_wmma(q, k, v, causal)
        print(f"  ✓ Output shape: {out.shape}")
        print(f"  Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    
    print("\n✓ WMMA demo completed successfully!")


if __name__ == '__main__':
    # 测试正确性
    test_wmma_correctness()
    
    # 性能 benchmark
    benchmark_wmma_performance()
    
    # 使用示例
    demo_wmma_usage()
    
    print("\n" + "=" * 80)
    print("All WMMA tests completed!")
    print("=" * 80)
