import torch
import flash_attn_v100
import time
import numpy as np
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


def benchmark_function(func, *args, warmup=10, iterations=100):
    """Benchmark 函数执行时间"""
    # Warmup
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()
    
    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        func(*args)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    
    return elapsed_ms / iterations  # 平均时间 (ms)


def calculate_tflops(B, H, N, D, time_ms, causal=True):
    """
    计算 TFLOPs
    Flash Attention 计算复杂度: O(B * H * N^2 * D)
    - QK^T: B * H * N^2 * D
    - Softmax: 忽略不计
    - PV: B * H * N^2 * D
    总计: 2 * B * H * N^2 * D FLOPs
    """
    flops = 2 * B * H * N * N * D
    time_s = time_ms / 1000.0
    tflops = flops / time_s / 1e12
    return tflops


def calculate_memory_bandwidth(B, H, N, D, time_ms):
    """
    计算内存带宽 (GB/s)
    每次迭代需要读取: Q, K, V, 写入: O
    总计: 4 * B * H * N * D * sizeof(dtype) bytes
    """
    bytes_per_element = 2  # FP16: 2 bytes, FP32: 4 bytes
    total_bytes = 4 * B * H * N * D * bytes_per_element
    time_s = time_ms / 1000.0
    bandwidth_gb_s = total_bytes / time_s / 1e9
    return bandwidth_gb_s


def run_benchmark():
    """运行完整的 benchmark 测试"""
    print("=" * 100)
    print("Flash Attention V100 Benchmark")
    print("=" * 100)
    
    # 测试配置
    configs = [
        # (B, H, N, D)
        (1, 8, 128, 64),
        (1, 8, 256, 64),
        (1, 8, 512, 64),
        (1, 8, 1024, 64),
        (1, 8, 2048, 64),
        (2, 8, 256, 64),
        (2, 8, 512, 64),
        (2, 8, 1024, 64),
        (2, 8, 2048, 64),
        (1, 8, 256, 128),
        (1, 8, 512, 128),
        (1, 8, 1024, 128),
        (2, 8, 256, 128),
        (2, 8, 512, 128),
        (2, 8, 1024, 128),
    ]
    
    results = []
    
    for B, H, N, D in configs:
        print(f"\nTesting: B={B}, H={H}, N={N}, D={D}")
        print("-" * 100)
        
        # 创建测试数据 (FP16)
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        
        # 确保连续性
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        for causal in [False, True]:
            causal_str = "Causal" if causal else "Non-Causal"
            
            # Benchmark Flash Attention FP16
            try:
                time_fa = benchmark_function(
                    flash_attn_v100.forward_fp16,
                    q, k, v, causal,
                    warmup=10,
                    iterations=100
                )
                
                # 验证正确性
                out_fa = flash_attn_v100.forward_fp16(q, k, v, causal)
                out_ref = reference_attention(q, k, v, causal)
                
                max_diff = (out_fa.float() - out_ref.float()).abs().max().item()
                all_close = torch.allclose(out_fa, out_ref, atol=1e-2, rtol=1e-2)
                
                # 计算性能指标
                tflops = calculate_tflops(B, H, N, D, time_fa, causal)
                bandwidth = calculate_memory_bandwidth(B, H, N, D, time_fa)
                
                results.append({
                    'B': B, 'H': H, 'N': N, 'D': D,
                    'Mode': causal_str,
                    'Time (ms)': f"{time_fa:.3f}",
                    'TFLOPs': f"{tflops:.2f}",
                    'BW (GB/s)': f"{bandwidth:.2f}",
                    'Max Diff': f"{max_diff:.6f}",
                    'Correct': '✓' if all_close else '✗'
                })
                
                print(f"  {causal_str:12s} | Time: {time_fa:.3f} ms | "
                      f"TFLOPs: {tflops:.2f} | BW: {bandwidth:.2f} GB/s | "
                      f"Max Diff: {max_diff:.6f} | Correct: {all_close}")
                
            except Exception as e:
                print(f"  {causal_str:12s} | ERROR: {str(e)}")
                results.append({
                    'B': B, 'H': H, 'N': N, 'D': D,
                    'Mode': causal_str,
                    'Time (ms)': 'ERROR',
                    'TFLOPs': 'N/A',
                    'BW (GB/s)': 'N/A',
                    'Max Diff': 'N/A',
                    'Correct': '✗'
                })
        
        torch.cuda.empty_cache()
    
    # 打印汇总表
    print("\n" + "=" * 100)
    print("Benchmark Summary")
    print("=" * 100)
    
    headers = ['B', 'H', 'N', 'D', 'Mode', 'Time (ms)', 'TFLOPs', 'BW (GB/s)', 'Max Diff', 'Correct']
    table_data = []
    
    for r in results:
        table_data.append([
            r['B'], r['H'], r['N'], r['D'], r['Mode'],
            r['Time (ms)'], r['TFLOPs'], r['BW (GB/s)'],
            r['Max Diff'], r['Correct']
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # 保存结果到文件
    with open('benchmark_results.txt', 'w') as f:
        f.write("Flash Attention V100 Benchmark Results\n")
        f.write("=" * 100 + "\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    print("\nResults saved to benchmark_results.txt")
    
    # Roofline 分析提示
    print("\n" + "=" * 100)
    print("Roofline Analysis Notes")
    print("=" * 100)
    print("""
V100 GPU 理论峰值:
  - FP16 Tensor Core: ~120 TFLOPs (with mixed precision)
  - FP16 CUDA Core: ~15.7 TFLOPs
  - Memory Bandwidth: 900 GB/s (HBM2)

性能优化建议:
  1. 如果 TFLOPs 较低 → 计算密集型，需要优化计算（如使用 Tensor Core）
  2. 如果带宽接近 900 GB/s → 内存密集型，需要优化内存访问
  3. 当前实现使用 FP16 + 向量化，应能充分利用内存带宽
  4. 要进一步优化可考虑:
     - 使用 WMMA (Warp-level Matrix Multiply-Accumulate) 指令
     - 使用 Tensor Core (需要特殊的矩阵布局)
     - 优化 shared memory bank conflicts
    """)


if __name__ == '__main__':
    run_benchmark()
