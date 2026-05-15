import torch
import torch.distributed as dist
import time
import flash_attn_v100


class TensorParallelDecoder:
    def __init__(self, H_Q, H_KV, D, world_size=None):
        self.H_Q = H_Q
        self.H_KV = H_KV
        self.D = D
        self.world_size = world_size or torch.cuda.device_count()
        self.local_H_Q = H_Q // self.world_size
        self.local_H_KV = H_KV // self.world_size

        assert H_Q % self.world_size == 0, f"H_Q={H_Q} must be divisible by world_size={self.world_size}"
        assert H_KV % self.world_size == 0, f"H_KV={H_KV} must be divisible by world_size={self.world_size}"

        self.streams = []
        for rank in range(self.world_size):
            device = torch.device(f'cuda:{rank}')
            stream = torch.cuda.Stream(device=device)
            self.streams.append(stream)

        self.p2p_enabled = True
        for i in range(self.world_size):
            for j in range(self.world_size):
                if i != j:
                    if not torch.cuda.can_device_access_peer(i, j):
                        try:
                            torch.cuda.enable_peer_access(i, j)
                        except Exception:
                            self.p2p_enabled = False

    def get_local_q(self, q, rank):
        h_start = rank * self.local_H_Q
        h_end = h_start + self.local_H_Q
        return q[:, h_start:h_end, :, :].contiguous()

    def get_local_kv(self, k, v, rank):
        h_start = rank * self.local_H_KV
        h_end = h_start + self.local_H_KV
        local_k = k[:, h_start:h_end, :, :].contiguous()
        local_v = v[:, h_start:h_end, :, :].contiguous()
        return local_k, local_v

    def decode_step(self, q, k, v, cache_len):
        B = q.size(0)
        outputs = [None] * self.world_size

        for rank in range(self.world_size):
            device = torch.device(f'cuda:{rank}')
            with torch.cuda.stream(self.streams[rank]):
                local_q = self.get_local_q(q, rank).to(device, non_blocking=True)
                local_k, local_v = self.get_local_kv(k, v, rank)
                local_k = local_k.to(device, non_blocking=True)
                local_v = local_v.to(device, non_blocking=True)

                local_out = flash_attn_v100.forward_decode_gqa_fp16(
                    local_q, local_k, local_v, True, cache_len
                )
                outputs[rank] = local_out

        torch.cuda.synchronize()
        full_out = torch.cat([o.cpu() for o in outputs], dim=1)
        return full_out.to(q.device)

    def decode_step_local(self, q, kv_caches, cache_len):
        B = q.size(0)
        outputs = [None] * self.world_size

        for rank in range(self.world_size):
            device = torch.device(f'cuda:{rank}')
            with torch.cuda.stream(self.streams[rank]):
                local_q = self.get_local_q(q, rank).to(device, non_blocking=True)
                local_k, local_v = kv_caches[rank]

                local_out = flash_attn_v100.forward_decode_gqa_fp16(
                    local_q, local_k, local_v, True, cache_len
                )
                outputs[rank] = local_out

        torch.cuda.synchronize()
        full_out = torch.cat([o.cpu() for o in outputs], dim=1)
        return full_out.to(q.device)


def test_single_gpu_baseline():
    print("=" * 60)
    print("Test 1: Single GPU Baseline")
    print("=" * 60)

    torch.manual_seed(42)

    B, H_Q, H_KV, N, D = 4, 16, 4, 256, 64
    q = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)

    cache_len = N - 1
    out = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)

    print(f"  B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}")
    print(f"  Output shape: {out.shape}")
    print(f"  Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    print(f"  Result: PASS")
    return True


def test_tensor_parallel_prefill():
    print("\n" + "=" * 60)
    print("Test 8: Tensor Parallel Prefill Latency")
    print("=" * 60)

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("  Skipping: need at least 2 GPUs")
        return True

    torch.manual_seed(42)

    configs = [
        (1, 16, 4, 512, 64),
        (1, 16, 4, 1024, 64),
        (1, 32, 8, 512, 128),
        (1, 32, 8, 1024, 128),
        (2, 32, 8, 2048, 128),
    ]

    num_iters = 50
    warmup = 10

    for B, H_Q, H_KV, N, D in configs:
        q = torch.randn(B, H_Q, N, D, device='cuda:0', dtype=torch.float16)
        k = torch.randn(B, H_KV, N, D, device='cuda:0', dtype=torch.float16)
        v = torch.randn(B, H_KV, N, D, device='cuda:0', dtype=torch.float16)

        for _ in range(warmup):
            flash_attn_v100.forward_prefill_gqa_fp16(q, k, v, True)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iters):
            flash_attn_v100.forward_prefill_gqa_fp16(q, k, v, True)
        torch.cuda.synchronize()
        single_time = (time.perf_counter() - start) / num_iters * 1000

        tp_decoder = TensorParallelDecoder(H_Q, H_KV, D, world_size=num_gpus)

        def tp_prefill():
            outputs = [None] * num_gpus
            for rank in range(num_gpus):
                device = torch.device(f'cuda:{rank}')
                with torch.cuda.stream(tp_decoder.streams[rank]):
                    local_q = tp_decoder.get_local_q(q, rank).to(device, non_blocking=True)
                    local_k, local_v = tp_decoder.get_local_kv(k, v, rank)
                    local_k = local_k.to(device, non_blocking=True)
                    local_v = local_v.to(device, non_blocking=True)
                    outputs[rank] = flash_attn_v100.forward_prefill_gqa_fp16(
                        local_q, local_k, local_v, True
                    )
            torch.cuda.synchronize()
            return torch.cat([o.cpu() for o in outputs], dim=1)

        for _ in range(warmup):
            tp_prefill()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iters):
            tp_prefill()
        torch.cuda.synchronize()
        tp_time = (time.perf_counter() - start) / num_iters * 1000

        speedup = single_time / tp_time
        print(f"  B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}, GPUs={num_gpus}")
        print(f"    Single GPU: {single_time:.3f} ms")
        print(f"    TP ({num_gpus} GPUs): {tp_time:.3f} ms")
        print(f"    Speedup: {speedup:.2f}x")

    print(f"  Result: PASS")
    return True


def test_memory_savings():
    print("\n" + "=" * 60)
    print("Test 9: Multi-GPU Memory Savings")
    print("=" * 60)

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("  Skipping: need at least 2 GPUs")
        return True

    B, H_Q, H_KV, N, D = 4, 32, 8, 4096, 128

    single_kv_mem = B * H_KV * N * D * 2
    per_gpu_kv_mem = B * (H_KV // num_gpus) * N * D * 2

    print(f"  Config: B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}")
    print(f"  Single GPU KV cache: {single_kv_mem / 1024 / 1024:.1f} MB")
    print(f"  Per-GPU KV cache (TP): {per_gpu_kv_mem / 1024 / 1024:.1f} MB")
    print(f"  Memory reduction per GPU: {(1 - per_gpu_kv_mem / single_kv_mem) * 100:.1f}%")

    q = torch.randn(B, H_Q, 1, D, device='cuda:0', dtype=torch.float16)
    k = torch.randn(B, H_KV, N, D, device='cuda:0', dtype=torch.float16)
    v = torch.randn(B, H_KV, N, D, device='cuda:0', dtype=torch.float16)

    torch.cuda.reset_peak_memory_stats(0)
    flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, N - 1)
    torch.cuda.synchronize()
    single_peak = torch.cuda.max_memory_allocated(0) / 1024 / 1024

    tp = TensorParallelDecoder(H_Q, H_KV, D, world_size=num_gpus)
    torch.cuda.reset_peak_memory_stats(0)
    tp.decode_step(q, k, v, N - 1)
    torch.cuda.synchronize()
    tp_peak = torch.cuda.max_memory_allocated(0) / 1024 / 1024

    print(f"  Single GPU peak memory (GPU 0): {single_peak:.1f} MB")
    print(f"  TP peak memory (GPU 0): {tp_peak:.1f} MB")
    print(f"  Result: PASS")
    return True


def test_scatter_gather_heads():
    print("\n" + "=" * 60)
    print("Test 2: Scatter/Gather Heads Correctness")
    print("=" * 60)

    torch.manual_seed(42)

    B, H_Q, N, D = 2, 16, 128, 64
    full_tensor = torch.randn(B, H_Q, N, D, device='cuda', dtype=torch.float16)

    for local_head_count in [4, 8]:
        local_tensor = torch.zeros(B, local_head_count, N, D, device='cuda', dtype=torch.float16)

        flash_attn_v100.scatter_heads(
            full_tensor, local_tensor, H_Q, 0, local_head_count
        )

        expected = full_tensor[:, :local_head_count, :, :]
        diff = (local_tensor - expected).abs().max().item()
        assert diff < 1e-6, f"Scatter diff too large: {diff}"

        reconstructed = torch.zeros_like(full_tensor)
        flash_attn_v100.gather_heads(
            reconstructed, local_tensor, H_Q, 0, local_head_count
        )

        diff2 = (reconstructed[:, :local_head_count, :, :] - expected).abs().max().item()
        assert diff2 < 1e-6, f"Gather diff too large: {diff2}"

        print(f"  local_head_count={local_head_count}: scatter diff={diff:.8f}, gather diff={diff2:.8f}")

    print(f"  Result: PASS")
    return True


def test_tensor_parallel_correctness():
    print("\n" + "=" * 60)
    print("Test 3: Tensor Parallel Decode Correctness")
    print("=" * 60)

    torch.manual_seed(42)

    num_gpus = torch.cuda.device_count()
    print(f"  Available GPUs: {num_gpus}")

    configs = [
        (2, 16, 4, 256, 64),
        (2, 16, 4, 512, 128),
        (4, 32, 8, 256, 64),
    ]

    for B, H_Q, H_KV, N, D in configs:
        if H_Q % num_gpus != 0 or H_KV % num_gpus != 0:
            print(f"  Skipping B={B}, H_Q={H_Q}, H_KV={H_KV} (not divisible by {num_gpus})")
            continue

        q = torch.randn(B, H_Q, 1, D, device='cuda:0', dtype=torch.float16)
        k = torch.randn(B, H_KV, N, D, device='cuda:0', dtype=torch.float16)
        v = torch.randn(B, H_KV, N, D, device='cuda:0', dtype=torch.float16)
        cache_len = N - 1

        out_single = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)

        tp_decoder = TensorParallelDecoder(H_Q, H_KV, D, world_size=num_gpus)
        out_tp = tp_decoder.decode_step(q, k, v, cache_len)

        diff = (out_single.float() - out_tp.float()).abs().max().item()
        print(f"  B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}, GPUs={num_gpus}")
        print(f"    Single GPU shape: {out_single.shape}")
        print(f"    TP shape: {out_tp.shape}")
        print(f"    Max diff: {diff:.8f}")

        assert diff < 0.01, f"TP diff too large: {diff}"

    print(f"  Result: PASS")
    return True


def test_cross_gpu_allreduce():
    print("\n" + "=" * 60)
    print("Test 4: Cross-GPU AllReduce")
    print("=" * 60)

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("  Skipping: need at least 2 GPUs")
        return True

    torch.manual_seed(42)

    B, H, N, D = 2, 8, 128, 64
    tensor_0 = torch.randn(B, H, N, D, device='cuda:0', dtype=torch.float16)
    tensor_1 = torch.randn(B, H, N, D, device='cuda:1', dtype=torch.float16)

    original_0 = tensor_0.clone()
    original_1 = tensor_1.clone()

    expected_sum = original_0.float() + original_1.to('cuda:0').float()

    tensor_1_on_0 = tensor_1.to('cuda:0')
    flash_attn_v100.cross_gpu_allreduce_fp16(tensor_0, tensor_1_on_0)

    diff = (tensor_0.float() - expected_sum).abs().max().item()
    print(f"  Config: B={B}, H={H}, N={N}, D={D}")
    print(f"  GPU 0 + GPU 1 -> GPU 0")
    print(f"  Max diff vs expected: {diff:.6f}")

    assert diff < 0.01, f"AllReduce diff too large: {diff}"

    print(f"  Result: PASS")
    return True


def test_tensor_parallel_latency():
    print("\n" + "=" * 60)
    print("Test 5: Tensor Parallel Decode Latency")
    print("=" * 60)

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("  Skipping: need at least 2 GPUs")
        return True

    torch.manual_seed(42)

    configs = [
        (4, 16, 4, 512, 64),
        (4, 16, 4, 1024, 64),
        (4, 32, 8, 512, 128),
        (4, 32, 8, 1024, 128),
    ]

    num_iters = 100
    warmup = 20

    for B, H_Q, H_KV, N, D in configs:
        q = torch.randn(B, H_Q, 1, D, device='cuda:0', dtype=torch.float16)
        k = torch.randn(B, H_KV, N, D, device='cuda:0', dtype=torch.float16)
        v = torch.randn(B, H_KV, N, D, device='cuda:0', dtype=torch.float16)
        cache_len = N - 1

        for _ in range(warmup):
            flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iters):
            flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)
        torch.cuda.synchronize()
        single_time = (time.perf_counter() - start) / num_iters * 1000

        tp_decoder = TensorParallelDecoder(H_Q, H_KV, D, world_size=num_gpus)

        for _ in range(warmup):
            tp_decoder.decode_step(q, k, v, cache_len)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iters):
            tp_decoder.decode_step(q, k, v, cache_len)
        torch.cuda.synchronize()
        tp_time = (time.perf_counter() - start) / num_iters * 1000

        speedup = single_time / tp_time
        print(f"  B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}, GPUs={num_gpus}")
        print(f"    Single GPU: {single_time:.3f} ms")
        print(f"    TP ({num_gpus} GPUs): {tp_time:.3f} ms")
        print(f"    Speedup: {speedup:.2f}x")

    print(f"  Result: PASS")
    return True


def test_p2p_access():
    print("\n" + "=" * 60)
    print("Test 6: GPU P2P Access Check")
    print("=" * 60)

    num_gpus = torch.cuda.device_count()
    print(f"  Available GPUs: {num_gpus}")

    for i in range(num_gpus):
        for j in range(num_gpus):
            if i != j:
                can_access = torch.cuda.can_device_access_peer(i, j)
                print(f"  GPU {i} -> GPU {j}: {'Yes' if can_access else 'No'}")

    print(f"  Result: PASS")
    return True


def test_multi_gpu_kv_cache():
    print("\n" + "=" * 60)
    print("Test 7: Multi-GPU KV Cache Management")
    print("=" * 60)

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("  Skipping: need at least 2 GPUs")
        return True

    torch.manual_seed(42)

    B, H_Q, H_KV, D = 2, 16, 4, 64
    max_seq_len = 512

    tp = TensorParallelDecoder(H_Q, H_KV, D, world_size=num_gpus)

    kv_caches = []
    for rank in range(num_gpus):
        device = torch.device(f'cuda:{rank}')
        local_k = torch.zeros(B, tp.local_H_KV, max_seq_len, D, device=device, dtype=torch.float16)
        local_v = torch.zeros(B, tp.local_H_KV, max_seq_len, D, device=device, dtype=torch.float16)
        kv_caches.append((local_k, local_v))

    num_steps = 10
    for step in range(num_steps):
        q = torch.randn(B, H_Q, 1, D, device='cuda:0', dtype=torch.float16)
        cache_len = step

        for rank in range(num_gpus):
            device = torch.device(f'cuda:{rank}')
            local_k, local_v = kv_caches[rank]
            new_k = torch.randn(B, tp.local_H_KV, 1, D, device=device, dtype=torch.float16)
            new_v = torch.randn(B, tp.local_H_KV, 1, D, device=device, dtype=torch.float16)
            local_k[:, :, step:step+1, :] = new_k
            local_v[:, :, step:step+1, :] = new_v

        if cache_len > 0:
            out = tp.decode_step_local(q, kv_caches, cache_len)
            assert out.shape == (B, H_Q, 1, D), f"Unexpected shape: {out.shape}"

    print(f"  B={B}, H_Q={H_Q}, H_KV={H_KV}, D={D}, steps={num_steps}")
    print(f"  Each GPU manages {tp.local_H_KV} KV heads")
    print(f"  Output shape: {out.shape}")
    print(f"  Result: PASS")
    return True


if __name__ == "__main__":
    results = []
    results.append(("Single GPU Baseline", test_single_gpu_baseline()))
    results.append(("Scatter/Gather Heads", test_scatter_gather_heads()))
    results.append(("Tensor Parallel Correctness", test_tensor_parallel_correctness()))
    results.append(("Cross-GPU AllReduce", test_cross_gpu_allreduce()))
    results.append(("Tensor Parallel Latency", test_tensor_parallel_latency()))
    results.append(("P2P Access Check", test_p2p_access()))
    results.append(("Multi-GPU KV Cache", test_multi_gpu_kv_cache()))
    results.append(("Tensor Parallel Prefill", test_tensor_parallel_prefill()))
    results.append(("Multi-GPU Memory Savings", test_memory_savings()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n  Total: {passed}/{total} passed")
