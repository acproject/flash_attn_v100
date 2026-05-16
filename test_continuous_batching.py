"""
Continuous Batching 测试和演示
Continuous Batching 允许不同序列长度的请求动态加入/离开 batch
"""

import torch
import time
from tabulate import tabulate
import flash_attn_v100


def simulate_continuous_batching():
    """模拟 Continuous Batching 场景"""
    print("=" * 80)
    print("Continuous Batching Simulation")
    print("=" * 80)
    
    # 场景：多个用户同时请求，但序列长度不同
    print("\nScenario: Multiple users with different sequence lengths")
    print("-" * 60)
    
    # 模拟 4 个用户的请求
    requests = [
        {"user_id": 1, "prompt_len": 10, "gen_len": 20},  # 短对话
        {"user_id": 2, "prompt_len": 50, "gen_len": 100}, # 中等对话
        {"user_id": 3, "prompt_len": 100, "gen_len": 50}, # 长 prompt，短回复
        {"user_id": 4, "prompt_len": 20, "gen_len": 200}, # 短 prompt，长回复
    ]
    
    print(f"{'User':6s} | {'Prompt Len':10s} | {'Gen Len':10s} | {'Total Len':10s}")
    print("-" * 60)
    for req in requests:
        total = req["prompt_len"] + req["gen_len"]
        print(f"{req['user_id']:6d} | {req['prompt_len']:10d} | {req['gen_len']:10d} | {total:10d}")
    
    print("\n" + "=" * 60)
    print("Static Batching vs Continuous Batching")
    print("=" * 60)
    
    # Static Batching: 所有请求必须 padding 到最长序列
    max_total_len = max(r["prompt_len"] + r["gen_len"] for r in requests)
    static_tokens = len(requests) * max_total_len
    
    # Continuous Batching: 每个请求只使用实际需要的长度
    continuous_tokens = sum(r["prompt_len"] + r["gen_len"] for r in requests)
    
    print(f"\nStatic Batching:")
    print(f"  Max sequence length: {max_total_len}")
    print(f"  Total tokens processed: {static_tokens}")
    print(f"  Padding overhead: {(1 - continuous_tokens/static_tokens)*100:.1f}%")
    
    print(f"\nContinuous Batching:")
    print(f"  Total tokens processed: {continuous_tokens}")
    print(f"  No padding overhead!")
    print(f"  Efficiency gain: {static_tokens/continuous_tokens:.2f}x")


def test_continuous_batching_kernel():
    """测试 Continuous Batching 内核"""
    print("\n" + "=" * 80)
    print("Testing Continuous Batching Kernel")
    print("=" * 80)
    
    # 注意：这里我们需要先编译内核
    # 目前先演示概念，实际测试需要编译
    
    print("\nContinuous Batching 核心概念:")
    print("-" * 60)
    print("""
Traditional Static Batching:
  • 所有请求必须同时开始和结束
  • 短的请求要等待长的请求
  • 大量 padding 浪费计算资源
  
Continuous Batching:
  • 请求可以随时加入 batch
  • 完成的请求立即离开 batch
  • 新请求立即填补空位
  • 每个序列使用实际长度，无 padding
  
优势:
  • 提高 GPU 利用率 2-5x
  • 降低延迟（短请求不用等待）
  • 提高吞吐量
    """)
    
    # 演示数据结构
    print("\nContinuous Batching 数据结构:")
    print("-" * 60)
    
    B = 4  # batch size
    H_Q, H_KV, D = 32, 8, 128
    cache_lens = torch.tensor([10, 50, 100, 20], dtype=torch.int32)  # 每个序列的实际长度
    max_N = cache_lens.max().item() + 1
    
    print(f"Batch size: {B}")
    print(f"Sequence lengths: {cache_lens.tolist()}")
    print(f"Max sequence length: {max_N}")
    print(f"Q shape: [{B}, {H_Q}, 1, {D}]")
    print(f"K/V shape: [{B}, {H_KV}, {max_N}, {D}]")
    print(f"cache_lens shape: [{B}]")
    
    print("\n内存对比:")
    print("-" * 60)
    
    # Static Batching 需要 padding 到 max
    static_kv = B * H_KV * max_N * D * 2 * 2 / (1024*1024)  # MB
    
    # Continuous Batching 只使用实际长度
    continuous_kv = sum((cl + 1) for cl in cache_lens.tolist()) * H_KV * D * 2 * 2 / (1024*1024)
    
    print(f"Static Batching KV Cache: {static_kv:.2f} MB")
    print(f"Continuous Batching KV Cache: {continuous_kv:.2f} MB")
    print(f"Savings: {(1 - continuous_kv/static_kv)*100:.1f}%")


def benchmark_continuous_batching():
    """Benchmark Continuous Batching vs Static Batching"""
    print("\n" + "=" * 80)
    print("Benchmark: Continuous Batching vs Static Batching")
    print("=" * 80)
    
    # 配置
    B = 8
    H_Q, H_KV, D = 32, 8, 128
    
    # 模拟不同的序列长度分布
    scenarios = [
        ("Uniform (all same)", [50] * B),
        ("Mixed (short)", [10, 20, 30, 40, 50, 60, 70, 80]),
        ("Mixed (long)", [100, 150, 200, 250, 300, 350, 400, 450]),
        ("Skewed (mostly short)", [10, 10, 10, 10, 10, 10, 10, 200]),
        ("Skewed (mostly long)", [200, 200, 200, 200, 200, 200, 200, 50]),
    ]
    
    results = []
    
    for name, cache_lens_list in scenarios:
        cache_lens = torch.tensor(cache_lens_list, dtype=torch.int32)
        max_N = max(cache_lens_list) + 1
        total_tokens = sum(cl + 1 for cl in cache_lens_list)
        avg_len = total_tokens / B
        
        # Static Batching 的 token 数（全部 padding 到 max）
        static_tokens = B * max_N
        
        # 计算效率
        efficiency = total_tokens / static_tokens * 100
        
        # KV Cache 对比
        static_kv = B * H_KV * max_N * D * 2 * 2 / (1024*1024)
        continuous_kv = total_tokens * H_KV * D * 2 * 2 / (1024*1024)
        
        results.append({
            'Scenario': name,
            'Avg Len': f"{avg_len:.0f}",
            'Max Len': max_N,
            'Total Tokens': total_tokens,
            'Static Tokens': static_tokens,
            'Efficiency': f"{efficiency:.1f}%",
            'Static KV (MB)': f"{static_kv:.2f}",
            'Continuous KV (MB)': f"{continuous_kv:.2f}",
            'KV Savings': f"{(1 - continuous_kv/static_kv)*100:.1f}%"
        })
    
    # 打印表格
    print("\n" + "=" * 100)
    headers = ['Scenario', 'Avg Len', 'Max Len', 'Total Tokens', 'Static Tokens', 
               'Efficiency', 'Static KV', 'Continuous KV', 'KV Savings']
    table_data = []
    
    for r in results:
        table_data.append([
            r['Scenario'], r['Avg Len'], r['Max Len'], 
            r['Total Tokens'], r['Static Tokens'],
            r['Efficiency'], r['Static KV (MB)'], 
            r['Continuous KV (MB)'], r['KV Savings']
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    print("\n💡 Key Insights:")
    print("  • Continuous Batching eliminates padding overhead")
    print("  • More efficient when sequence lengths vary significantly")
    print("  • Can improve GPU utilization by 2-5x")
    print("  • Reduces memory usage by avoiding padding")


def demo_serving_scenario():
    """演示真实的 serving 场景"""
    print("\n" + "=" * 80)
    print("Real-World Serving Scenario")
    print("=" * 80)
    
    print("\nScenario: LLM API Server with Concurrent Requests")
    print("-" * 60)
    
    # 模拟 10 秒内的请求
    import random
    random.seed(42)
    
    requests = []
    time_step = 0
    
    print(f"{'Time':6s} | {'Event':20s} | {'Batch Size':10s} | {'Max Len':10s} | {'Total Tokens':12s}")
    print("-" * 70)
    
    batch_cache_lens = []
    completed = 0
    
    for t in range(10):
        # 新请求到达
        if random.random() < 0.6:  # 60% 概率有新请求
            new_len = random.randint(20, 100)
            batch_cache_lens.append({"len": new_len, "remaining": new_len})
            print(f"{t:6d} | {'New request':20s} | {len(batch_cache_lens):10d} | {max(r['len'] for r in batch_cache_lens):10d} | {sum(r['remaining']+1 for r in batch_cache_lens):12d}")
        
        # 处理 batch
        if batch_cache_lens:
            # 每个请求减少 1 个 token
            for req in batch_cache_lens:
                req["remaining"] -= 1
            
            # 检查完成的请求
            before_count = len(batch_cache_lens)
            batch_cache_lens = [r for r in batch_cache_lens if r["remaining"] > 0]
            newly_completed = before_count - len(batch_cache_lens)
            completed += newly_completed
            
            if newly_completed > 0:
                print(f"{t:6d} | {'Completed: '+str(newly_completed):20s} | {len(batch_cache_lens):10d} | {max(r['len'] for r in batch_cache_lens) if batch_cache_lens else 0:10d} | {sum(r['remaining']+1 for r in batch_cache_lens):12d}")
    
    print("-" * 70)
    print(f"\nTotal requests completed: {completed}")
    print(f"Average batch size: {completed / 10:.1f}")
    
    print("\n" + "=" * 60)
    print("Static vs Continuous Batching Comparison")
    print("=" * 60)
    print("""
Static Batching:
  • Wait for all requests to finish before starting new batch
  • GPU idle time between batches
  • High latency for short requests
  
Continuous Batching:
  • Immediately fill vacated slots with new requests
  • Near 100% GPU utilization
  • Low latency for all requests
  • Higher throughput
    """)


def show_implementation_details():
    """展示实现细节"""
    print("\n" + "=" * 80)
    print("Implementation Details")
    print("=" * 80)
    
    print("\n1. Key Data Structures")
    print("-" * 60)
    print("""
# 每个序列的 cache 长度
cache_lens: Tensor[B]  # [10, 50, 100, 20]

# Q, K, V tensors
Q: [B, H_Q, 1, D]           # 每个序列 1 个 token
K: [B, H_KV, max_N, D]      # padding 到最大长度
V: [B, H_KV, max_N, D]

# 内核自动处理不同长度
for each batch item b:
    current_N = cache_lens[b] + 1
    # 只处理前 current_N 个 token
    """)
    
    print("\n2. CUDA Kernel 核心逻辑")
    print("-" * 60)
    print("""
__global__ void flash_attn_kernel_continuous_batching(
    ...,
    const int* cache_lens,  // 每个序列的实际长度
    ...
) {
    int b = blockIdx.x / H_Q;
    int current_cache_len = cache_lens[b];  // 获取实际长度
    int current_N = current_cache_len + 1;
    
    // 只处理实际长度，不处理 padding
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int k_len = min(TILE_K, current_N - k_start);
        // ... 处理 k_len 个 token
    }
}
    """)
    
    print("\n3. Scheduling Strategy")
    print("-" * 60)
    print("""
while True:
    # 1. 移除完成的序列
    completed = [s for s in batch if s.done()]
    batch.remove(completed)
    
    # 2. 添加新序列
    while can_add_more():
        new_request = request_queue.pop()
        batch.append(new_request)
    
    # 3. 执行 forward pass
    # 构建 batched tensors with different lengths
    q, k, v, cache_lens = build_batch(batch)
    
    # 4. 调用 Continuous Batching kernel
    out = flash_attn_forward_continuous_batching(q, k, v, cache_lens)
    
    # 5. 处理输出
    for i, seq in enumerate(batch):
        seq.process_output(out[i])
    """)


def reference_continuous_batching_decode(q, k, v, cache_lens, causal=True):
    B, H_Q, _, D = q.shape
    H_KV = k.size(1)
    max_N = k.size(2)
    group = H_Q // H_KV
    scale = 1.0 / (D ** 0.5)

    out = torch.empty((B, H_Q, 1, D), device=q.device, dtype=torch.float16)
    for b in range(B):
        cur_len = int(cache_lens[b].item())
        current_N = min(cur_len + 1, max_N)
        for h_q in range(H_Q):
            h_kv = h_q // group
            q_vec = q[b, h_q, 0].float()
            k_mat = k[b, h_kv, :current_N].float()
            v_mat = v[b, h_kv, :current_N].float()
            scores = (k_mat @ q_vec) * scale
            if causal:
                mask = torch.arange(current_N, device=q.device) > cur_len
                scores = scores.masked_fill(mask, float("-inf"))
            probs = torch.softmax(scores, dim=0)
            out_vec = probs @ v_mat
            out[b, h_q, 0] = out_vec.half()
    return out


def test_continuous_batching_fp16_correctness():
    print("\n" + "=" * 80)
    print("Testing forward_continuous_batching_fp16 Correctness")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return True

    torch.manual_seed(0)
    B, H_Q, H_KV, D = 3, 8, 2, 64
    max_N = 17
    cache_lens = torch.tensor([0, 3, 15], device="cuda", dtype=torch.int32)

    q = torch.randn(B, H_Q, 1, D, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(B, H_KV, max_N, D, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(B, H_KV, max_N, D, device="cuda", dtype=torch.float16).contiguous()

    ok = True
    for causal in (False, True):
        out = flash_attn_v100.forward_continuous_batching_fp16(q, k, v, cache_lens, causal)
        ref = reference_continuous_batching_decode(q, k, v, cache_lens, causal)
        max_diff = (out.float() - ref.float()).abs().max().item()
        all_close = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
        print(f"  causal={causal} | max_diff={max_diff:.6f} | allclose={all_close}")
        ok = ok and all_close

    print(f"Result: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    test_continuous_batching_fp16_correctness()
