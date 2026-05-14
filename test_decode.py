import torch
import flash_attn_v100
import time

def test_decode_kernel():
    """测试 KV Cache / Decode Kernel"""
    print("=" * 80)
    print("Testing KV Cache / Decode Kernel")
    print("=" * 80)
    
    # 配置参数
    B = 1       # batch size
    H = 8       # num heads
    D = 64      # head dim
    cache_len = 1024  # 历史 cache 长度
    current_len = 1   # 当前 token (Q_len = 1)
    total_len = cache_len + current_len  # K/V 总长度
    
    print(f"\nConfig: B={B}, H={H}, D={D}, cache_len={cache_len}, total_len={total_len}")
    
    # 创建测试数据
    # Q: [B, H, 1, D] - 当前 token 的 query
    q = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16)
    
    # K/V: [B, H, total_len, D] - 包含历史 cache 和当前 token
    k = torch.randn(B, H, total_len, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, total_len, D, device='cuda', dtype=torch.float16)
    
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # 测试 causal 模式
    for causal in [True, False]:
        causal_str = "Causal" if causal else "Non-Causal"
        print(f"\n{'='*60}")
        print(f"Testing {causal_str} Mode")
        print(f"{'='*60}")
        
        # 运行 decode kernel
        try:
            out_decode = flash_attn_v100.forward_decode_fp16(q, k, v, causal, cache_len)
            print(f"✓ Decode kernel executed successfully")
            print(f"  Output shape: {out_decode.shape}")
            print(f"  Output dtype: {out_decode.dtype}")
            
            # 验证输出范围
            out_min = out_decode.min().item()
            out_max = out_decode.max().item()
            out_mean = out_decode.mean().item()
            print(f"  Output range: [{out_min:.4f}, {out_max:.4f}], mean: {out_mean:.4f}")
            
            # 检查 NaN/Inf
            has_nan = torch.isnan(out_decode).any().item()
            has_inf = torch.isinf(out_decode).any().item()
            print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
            
        except Exception as e:
            print(f"✗ Decode kernel failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Decode Kernel Test Completed")
    print("=" * 80)


def benchmark_decode_kernel():
    """Benchmark Decode Kernel 性能"""
    print("\n" + "=" * 80)
    print("Benchmarking KV Cache / Decode Kernel")
    print("=" * 80)
    
    configs = [
        # (cache_len, D)
        (256, 64),
        (512, 64),
        (1024, 64),
        (2048, 64),
        (4096, 64),
        (256, 128),
        (512, 128),
        (1024, 128),
        (2048, 128),
        (4096, 128),
    ]
    
    B = 1
    H = 8
    
    results = []
    
    for cache_len, D in configs:
        print(f"\n{'='*60}")
        print(f"Config: cache_len={cache_len}, D={D}")
        print(f"{'='*60}")
        
        total_len = cache_len + 1
        
        # 创建数据
        q = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16).contiguous()
        k = torch.randn(B, H, total_len, D, device='cuda', dtype=torch.float16).contiguous()
        v = torch.randn(B, H, total_len, D, device='cuda', dtype=torch.float16).contiguous()
        
        # Warmup
        for _ in range(10):
            _ = flash_attn_v100.forward_decode_fp16(q, k, v, True, cache_len)
        torch.cuda.synchronize()
        
        # Benchmark
        iterations = 100
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            out = flash_attn_v100.forward_decode_fp16(q, k, v, True, cache_len)
        end.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / iterations
        
        # 计算吞吐量 (tokens/sec)
        # 每次处理 1 个 token，所以 throughput = 1 / time_per_token
        tokens_per_sec = 1000.0 / elapsed_ms
        
        # 计算 FLOPs
        # QK^T: B * H * 1 * N * D
        # PV: B * H * 1 * N * D
        # 总计: 2 * B * H * N * D
        flops = 2 * B * H * total_len * D
        time_s = elapsed_ms / 1000.0
        gflops = flops / time_s / 1e9
        
        # 内存带宽
        # 读取: Q + K + V, 写入: O
        # Q: B*H*1*D, K: B*H*N*D, V: B*H*N*D, O: B*H*1*D
        bytes_per_element = 2  # FP16
        total_bytes = (B * H * 1 * D + B * H * total_len * D + 
                      B * H * total_len * D + B * H * 1 * D) * bytes_per_element
        bandwidth_gb_s = total_bytes / time_s / 1e9
        
        print(f"  Time: {elapsed_ms:.3f} ms")
        print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
        print(f"  GFLOPs: {gflops:.2f}")
        print(f"  Memory Bandwidth: {bandwidth_gb_s:.2f} GB/s")
        
        results.append({
            'cache_len': cache_len,
            'D': D,
            'Time (ms)': f"{elapsed_ms:.3f}",
            'Tokens/sec': f"{tokens_per_sec:.1f}",
            'GFLOPs': f"{gflops:.2f}",
            'BW (GB/s)': f"{bandwidth_gb_s:.2f}"
        })
        
        torch.cuda.empty_cache()
    
    # 打印汇总表
    print("\n" + "=" * 80)
    print("Decode Kernel Benchmark Summary")
    print("=" * 80)
    
    from tabulate import tabulate
    headers = ['cache_len', 'D', 'Time (ms)', 'Tokens/sec', 'GFLOPs', 'BW (GB/s)']
    table_data = []
    
    for r in results:
        table_data.append([
            r['cache_len'], r['D'], r['Time (ms)'], 
            r['Tokens/sec'], r['GFLOPs'], r['BW (GB/s)']
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))


def test_correctness():
    """验证 Decode Kernel 的正确性（与 PyTorch 原生实现对比）"""
    print("\n" + "=" * 80)
    print("Testing Decode Kernel Correctness")
    print("=" * 80)
    
    B = 1
    H = 4
    D = 64
    cache_len = 128
    total_len = cache_len + 1
    
    # 创建数据
    q = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16).contiguous()
    k = torch.randn(B, H, total_len, D, device='cuda', dtype=torch.float16).contiguous()
    v = torch.randn(B, H, total_len, D, device='cuda', dtype=torch.float16).contiguous()
    
    # PyTorch 参考实现
    def reference_decode(q, k, v, cache_len, causal=True):
        d = q.size(-1)
        # q: [B, H, 1, D], k: [B, H, N, D]
        scores = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)  # [B, H, 1, N]
        
        if causal:
            # causal mask: Q 的位置是 cache_len
            mask = torch.ones(1, total_len, device=q.device, dtype=q.dtype)
            # mask out future tokens
            mask[:, cache_len+1:] = float('-inf')
            scores = scores + mask
        
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)  # [B, H, 1, D]
        return out
    
    # 对比
    for causal in [True, False]:
        causal_str = "Causal" if causal else "Non-Causal"
        print(f"\n{causal_str} Mode:")
        
        out_ref = reference_decode(q, k, v, cache_len, causal)
        out_decode = flash_attn_v100.forward_decode_fp16(q, k, v, causal, cache_len)
        
        max_diff = (out_decode.float() - out_ref.float()).abs().max().item()
        mean_diff = (out_decode.float() - out_ref.float()).abs().mean().item()
        all_close = torch.allclose(out_decode, out_ref, atol=1e-2, rtol=1e-2)
        
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        print(f"  All close (atol=1e-2): {all_close}")


if __name__ == '__main__':
    test_decode_kernel()
    benchmark_decode_kernel()
    test_correctness()
