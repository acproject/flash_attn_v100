import time

import torch

import flash_attn_v100


TARGET_WORLD_SIZE = 2


def require_two_v100():
    num_gpus = torch.cuda.device_count()
    if num_gpus < TARGET_WORLD_SIZE:
        print(f"  Skipping: need at least {TARGET_WORLD_SIZE} GPUs")
        return False

    gpu_names = [torch.cuda.get_device_name(i) for i in range(TARGET_WORLD_SIZE)]
    print(f"  Using GPUs: {gpu_names}")
    return True


def synchronize_devices(devices):
    for device in devices:
        torch.cuda.synchronize(device)


def benchmark_ms(fn, devices, warmup=10, iterations=50):
    for _ in range(warmup):
        fn()
    synchronize_devices(devices)

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    synchronize_devices(devices)
    return (time.perf_counter() - start) / iterations * 1000.0


class TensorParallelDecoder:
    def __init__(self, H_Q, H_KV, D, world_size=TARGET_WORLD_SIZE, root_rank=0):
        self.H_Q = H_Q
        self.H_KV = H_KV
        self.D = D
        self.world_size = world_size
        self.root_rank = root_rank
        self.devices = [torch.device(f"cuda:{rank}") for rank in range(world_size)]
        self.root_device = self.devices[root_rank]
        self.local_H_Q = H_Q // world_size
        self.local_H_KV = H_KV // world_size

        assert torch.cuda.device_count() >= world_size, "not enough GPUs"
        assert H_Q % world_size == 0, f"H_Q={H_Q} must be divisible by world_size={world_size}"
        assert H_KV % world_size == 0, f"H_KV={H_KV} must be divisible by world_size={world_size}"

        self.streams = [torch.cuda.Stream(device=device) for device in self.devices]
        self.p2p_matrix = [
            [i == j or torch.cuda.can_device_access_peer(i, j) for j in range(world_size)]
            for i in range(world_size)
        ]

    def _head_range_q(self, rank):
        start = rank * self.local_H_Q
        return start, start + self.local_H_Q

    def _head_range_kv(self, rank):
        start = rank * self.local_H_KV
        return start, start + self.local_H_KV

    def shard_q(self, q):
        shards = []
        for rank, device in enumerate(self.devices):
            h_start, h_end = self._head_range_q(rank)
            shards.append(q[:, h_start:h_end].contiguous().to(device, non_blocking=True))
        return shards

    def shard_kv(self, k, v):
        shards = []
        for rank, device in enumerate(self.devices):
            h_start, h_end = self._head_range_kv(rank)
            local_k = k[:, h_start:h_end].contiguous().to(device, non_blocking=True)
            local_v = v[:, h_start:h_end].contiguous().to(device, non_blocking=True)
            shards.append((local_k, local_v))
        return shards

    def shard_qkv(self, q, k, v):
        return self.shard_q(q), self.shard_kv(k, v)

    def allocate_local_kv_cache(self, B, max_seq_len, dtype=torch.float16):
        caches = []
        for device in self.devices:
            local_k = torch.zeros(B, self.local_H_KV, max_seq_len, self.D, device=device, dtype=dtype)
            local_v = torch.zeros(B, self.local_H_KV, max_seq_len, self.D, device=device, dtype=dtype)
            caches.append((local_k, local_v))
        return caches

    def gather_local_outputs(self, local_outputs):
        B, _, seq_len, D = local_outputs[0].shape
        full_out = torch.empty(B, self.H_Q, seq_len, D, device=self.root_device, dtype=local_outputs[0].dtype)
        for rank, local_out in enumerate(local_outputs):
            h_start, h_end = self._head_range_q(rank)
            full_out[:, h_start:h_end].copy_(local_out.to(self.root_device, non_blocking=True))
        return full_out

    def decode_step_sharded(self, local_q_shards, local_kv_shards, cache_len, causal=True):
        outputs = [None] * self.world_size
        for rank, _ in enumerate(self.devices):
            local_q = local_q_shards[rank]
            local_k, local_v = local_kv_shards[rank]
            with torch.cuda.stream(self.streams[rank]):
                outputs[rank] = flash_attn_v100.forward_decode_gqa_fp16(
                    local_q, local_k, local_v, causal, cache_len
                )

        synchronize_devices(self.devices)
        return self.gather_local_outputs(outputs)

    def decode_step(self, q, k, v, cache_len, causal=True):
        local_q_shards, local_kv_shards = self.shard_qkv(q, k, v)
        return self.decode_step_sharded(local_q_shards, local_kv_shards, cache_len, causal)

    def decode_step_local(self, q, kv_caches, cache_len, causal=True):
        local_q_shards = self.shard_q(q)
        return self.decode_step_sharded(local_q_shards, kv_caches, cache_len, causal)

    def prefill_step_sharded(self, local_q_shards, local_kv_shards, causal=True, gather_output=True):
        outputs = [None] * self.world_size
        for rank, _ in enumerate(self.devices):
            local_q = local_q_shards[rank]
            local_k, local_v = local_kv_shards[rank]
            with torch.cuda.stream(self.streams[rank]):
                outputs[rank] = flash_attn_v100.forward_prefill_gqa_fp16(
                    local_q, local_k, local_v, causal
                )

        synchronize_devices(self.devices)
        if not gather_output:
            return outputs
        return self.gather_local_outputs(outputs)

    def prefill_step(self, q, k, v, causal=True):
        local_q_shards, local_kv_shards = self.shard_qkv(q, k, v)
        return self.prefill_step_sharded(local_q_shards, local_kv_shards, causal=causal, gather_output=True)

    def kv_cache_bytes_per_gpu(self, B, seq_len, dtype_bytes=2):
        return B * self.local_H_KV * seq_len * self.D * 2 * dtype_bytes


def test_scatter_gather_heads():
    print("\n" + "=" * 60)
    print("Test 1: Scatter/Gather Heads Correctness")
    print("=" * 60)

    torch.manual_seed(42)
    B, H_Q, N, D = 2, 16, 128, 64
    full_tensor = torch.randn(B, H_Q, N, D, device="cuda:0", dtype=torch.float16)

    for local_head_count in [4, 8]:
        local_tensor = torch.zeros(B, local_head_count, N, D, device="cuda:0", dtype=torch.float16)
        flash_attn_v100.scatter_heads(full_tensor, local_tensor, H_Q, 0, local_head_count)

        expected = full_tensor[:, :local_head_count]
        diff = (local_tensor - expected).abs().max().item()
        assert diff < 1e-6, f"scatter diff too large: {diff}"

        reconstructed = torch.zeros_like(full_tensor)
        flash_attn_v100.gather_heads(reconstructed, local_tensor, H_Q, 0, local_head_count)
        diff2 = (reconstructed[:, :local_head_count] - expected).abs().max().item()
        assert diff2 < 1e-6, f"gather diff too large: {diff2}"

        print(f"  local_head_count={local_head_count}: scatter diff={diff:.8f}, gather diff={diff2:.8f}")

    print("  Result: PASS")
    return True


def test_tensor_parallel_decode_correctness():
    print("\n" + "=" * 60)
    print("Test 2: Tensor Parallel Decode Correctness")
    print("=" * 60)

    if not require_two_v100():
        return True

    torch.manual_seed(42)
    configs = [
        (2, 16, 4, 512, 64),
        (2, 32, 8, 1024, 128),
    ]

    for B, H_Q, H_KV, N, D in configs:
        tp = TensorParallelDecoder(H_Q, H_KV, D, world_size=TARGET_WORLD_SIZE)
        q = torch.randn(B, H_Q, 1, D, device="cuda:0", dtype=torch.float16)
        k = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
        v = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
        cache_len = N - 1

        out_single = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)
        out_tp = tp.decode_step(q, k, v, cache_len, causal=True)
        diff = (out_single.float() - out_tp.float()).abs().max().item()

        print(f"  B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}")
        print(f"    Max diff: {diff:.8f}")
        assert diff < 1e-2, f"TP decode diff too large: {diff}"

    print("  Result: PASS")
    return True


def test_multi_gpu_kv_cache():
    print("\n" + "=" * 60)
    print("Test 3: Tensor Parallel Local KV Cache")
    print("=" * 60)

    if not require_two_v100():
        return True

    torch.manual_seed(42)
    B, H_Q, H_KV, N, D = 2, 16, 4, 512, 64
    tp = TensorParallelDecoder(H_Q, H_KV, D, world_size=TARGET_WORLD_SIZE)

    q = torch.randn(B, H_Q, 1, D, device="cuda:0", dtype=torch.float16)
    k = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
    v = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
    cache_len = N - 1

    local_kv_caches = tp.shard_kv(k, v)
    out_single = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)
    out_local = tp.decode_step_local(q, local_kv_caches, cache_len, causal=True)
    diff = (out_single.float() - out_local.float()).abs().max().item()

    print(f"  Each GPU manages {tp.local_H_KV} KV heads")
    print(f"  Max diff vs single GPU: {diff:.8f}")
    assert diff < 1e-2, f"TP local KV diff too large: {diff}"
    print("  Result: PASS")
    return True


def test_tensor_parallel_prefill_correctness():
    print("\n" + "=" * 60)
    print("Test 4: Tensor Parallel Prefill Correctness")
    print("=" * 60)

    if not require_two_v100():
        return True

    torch.manual_seed(42)
    B, H_Q, H_KV, N, D = 1, 32, 8, 1024, 128
    tp = TensorParallelDecoder(H_Q, H_KV, D, world_size=TARGET_WORLD_SIZE)

    q = torch.randn(B, H_Q, N, D, device="cuda:0", dtype=torch.float16)
    k = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
    v = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)

    out_single = flash_attn_v100.forward_prefill_gqa_fp16(q, k, v, True)
    out_tp = tp.prefill_step(q, k, v, causal=True)
    diff = (out_single.float() - out_tp.float()).abs().max().item()

    print(f"  Max diff: {diff:.8f}")
    assert diff < 1e-2, f"TP prefill diff too large: {diff}"
    print("  Result: PASS")
    return True


def test_memory_savings():
    print("\n" + "=" * 60)
    print("Test 5: 2xV100 KV Cache Memory Savings")
    print("=" * 60)

    if not require_two_v100():
        return True

    B, H_Q, H_KV, N, D = 4, 32, 8, 4096, 128
    tp = TensorParallelDecoder(H_Q, H_KV, D, world_size=TARGET_WORLD_SIZE)

    single_gpu_kv_bytes = B * H_KV * N * D * 2 * 2
    per_gpu_kv_bytes = tp.kv_cache_bytes_per_gpu(B, N)
    reduction = 1.0 - (per_gpu_kv_bytes / single_gpu_kv_bytes)

    print(f"  Config: B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}")
    print(f"  Single GPU KV cache: {single_gpu_kv_bytes / 1024 / 1024:.1f} MB")
    print(f"  Per-GPU KV cache (2-way TP): {per_gpu_kv_bytes / 1024 / 1024:.1f} MB")
    print(f"  Per-GPU memory reduction: {reduction * 100:.1f}%")

    assert abs(reduction - 0.5) < 1e-6, f"expected 50% KV reduction, got {reduction * 100:.2f}%"
    print("  Result: PASS")
    return True


def test_tensor_parallel_prefill_scaling():
    print("\n" + "=" * 60)
    print("Test 6: Tensor Parallel Prefill Scaling")
    print("=" * 60)

    if not require_two_v100():
        return True

    torch.manual_seed(42)
    configs = [
        (1, 32, 8, 1024, 128),
        (1, 32, 8, 2048, 128),
        (2, 32, 8, 2048, 128),
    ]

    best_efficiency = 0.0
    for B, H_Q, H_KV, N, D in configs:
        tp = TensorParallelDecoder(H_Q, H_KV, D, world_size=TARGET_WORLD_SIZE)
        q = torch.randn(B, H_Q, N, D, device="cuda:0", dtype=torch.float16)
        k = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
        v = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)

        local_q_shards, local_kv_shards = tp.shard_qkv(q, k, v)

        single_time = benchmark_ms(
            lambda: flash_attn_v100.forward_prefill_gqa_fp16(q, k, v, True),
            [torch.device("cuda:0")],
            warmup=5,
            iterations=20,
        )
        tp_compute_time = benchmark_ms(
            lambda: tp.prefill_step_sharded(local_q_shards, local_kv_shards, causal=True, gather_output=False),
            tp.devices,
            warmup=5,
            iterations=20,
        )
        tp_end_to_end_time = benchmark_ms(
            lambda: tp.prefill_step_sharded(local_q_shards, local_kv_shards, causal=True, gather_output=True),
            tp.devices,
            warmup=5,
            iterations=20,
        )

        compute_speedup = single_time / tp_compute_time
        end_to_end_speedup = single_time / tp_end_to_end_time
        scaling_efficiency = compute_speedup / TARGET_WORLD_SIZE
        best_efficiency = max(best_efficiency, scaling_efficiency)

        print(f"  B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}")
        print(f"    Single GPU: {single_time:.3f} ms")
        print(f"    TP compute-only: {tp_compute_time:.3f} ms | speedup={compute_speedup:.2f}x | efficiency={scaling_efficiency:.2f}x")
        print(f"    TP end-to-end: {tp_end_to_end_time:.3f} ms | speedup={end_to_end_speedup:.2f}x")

    print(f"  Best 2-GPU compute scaling efficiency: {best_efficiency:.2f}x")
    assert best_efficiency >= 0.85, f"expected prefill scaling efficiency near 0.85x, got {best_efficiency:.2f}x"
    print("  Result: PASS")
    return True


if __name__ == "__main__":
    results = []
    results.append(("Scatter/Gather Heads", test_scatter_gather_heads()))
    results.append(("TP Decode Correctness", test_tensor_parallel_decode_correctness()))
    results.append(("TP Local KV Cache", test_multi_gpu_kv_cache()))
    results.append(("TP Prefill Correctness", test_tensor_parallel_prefill_correctness()))
    results.append(("2xV100 KV Memory Savings", test_memory_savings()))
    results.append(("TP Prefill Scaling", test_tensor_parallel_prefill_scaling()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n  Total: {passed}/{total} passed")
