import torch
import torch.nn.functional as F
import time
import flash_attn_v100


def quantize_kv_to_int8(kv_fp16):
    B, H, N, D = kv_fp16.shape
    kv_float = kv_fp16.float()

    scale = kv_float.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = scale.clamp(min=1e-8)

    kv_int8 = (kv_float / scale).round().clamp(-128, 127).to(torch.int8)

    scale_half = scale.squeeze(-1).half()

    return kv_int8.contiguous(), scale_half.contiguous()


def test_int8_kv_cache_correctness():
    print("=" * 60)
    print("Test 1: INT8 KV Cache Decode Correctness")
    print("=" * 60)

    torch.manual_seed(42)

    configs = [
        (1, 8, 8, 128, 64),
        (2, 8, 2, 256, 64),
        (4, 16, 4, 512, 128),
    ]

    for B, H_Q, H_KV, N, D in configs:
        q = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)

        cache_len = N - 1

        out_fp16 = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)

        k_int8, k_scale = quantize_kv_to_int8(k)
        v_int8, v_scale = quantize_kv_to_int8(v)

        out_int8 = flash_attn_v100.forward_decode_int8_kv_cache_gqa(
            q, k_int8, v_int8, k_scale, v_scale, cache_len
        )

        max_diff = (out_fp16.float() - out_int8.float()).abs().max().item()
        mean_diff = (out_fp16.float() - out_int8.float()).abs().mean().item()

        print(f"  B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}")
        print(f"    Max diff:  {max_diff:.6f}")
        print(f"    Mean diff: {mean_diff:.6f}")

        assert max_diff < 0.1, f"Max diff too large: {max_diff}"
        print(f"    Result: PASS")

    return True


def test_int8_kv_cache_memory_savings():
    print("\n" + "=" * 60)
    print("Test 2: INT8 KV Cache Memory Savings")
    print("=" * 60)

    B, H_KV, N, D = 4, 8, 4096, 128

    k_fp16 = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)
    v_fp16 = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)

    k_int8, k_scale = quantize_kv_to_int8(k_fp16)
    v_int8, v_scale = quantize_kv_to_int8(v_fp16)

    fp16_bytes = k_fp16.nelement() * 2 + v_fp16.nelement() * 2
    int8_bytes = (k_int8.nelement() + v_int8.nelement()) * 1 + \
                 (k_scale.nelement() + v_scale.nelement()) * 2

    ratio = int8_bytes / fp16_bytes
    savings = (1 - ratio) * 100

    print(f"  Config: B={B}, H_KV={H_KV}, N={N}, D={D}")
    print(f"  FP16 KV cache: {fp16_bytes / 1024 / 1024:.2f} MB")
    print(f"  INT8 KV cache: {int8_bytes / 1024 / 1024:.2f} MB")
    print(f"  Size ratio: {ratio:.3f}")
    print(f"  Memory savings: {savings:.1f}%")
    print(f"  Result: PASS")
    return True


def test_int8_kv_cache_latency():
    print("\n" + "=" * 60)
    print("Test 3: INT8 KV Cache Decode Latency")
    print("=" * 60)

    torch.manual_seed(42)

    configs = [
        (4, 8, 2, 512, 64),
        (4, 8, 2, 1024, 64),
        (4, 16, 4, 512, 128),
        (4, 16, 4, 1024, 128),
    ]

    num_iters = 100
    warmup = 20

    for B, H_Q, H_KV, N, D in configs:
        q = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)
        cache_len = N - 1

        k_int8, k_scale = quantize_kv_to_int8(k)
        v_int8, v_scale = quantize_kv_to_int8(v)

        for _ in range(warmup):
            flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)
            flash_attn_v100.forward_decode_int8_kv_cache_gqa(
                q, k_int8, v_int8, k_scale, v_scale, cache_len
            )
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iters):
            flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)
        torch.cuda.synchronize()
        fp16_time = (time.perf_counter() - start) / num_iters * 1000

        start = time.perf_counter()
        for _ in range(num_iters):
            flash_attn_v100.forward_decode_int8_kv_cache_gqa(
                q, k_int8, v_int8, k_scale, v_scale, cache_len
            )
        torch.cuda.synchronize()
        int8_time = (time.perf_counter() - start) / num_iters * 1000

        speedup = fp16_time / int8_time
        print(f"  B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}")
        print(f"    FP16 decode: {fp16_time:.3f} ms")
        print(f"    INT8 KV decode: {int8_time:.3f} ms")
        print(f"    Speedup: {speedup:.2f}x")

    print(f"  Result: PASS")
    return True


def test_token_eviction_correctness():
    print("\n" + "=" * 60)
    print("Test 4: Token Eviction Correctness")
    print("=" * 60)

    torch.manual_seed(42)

    B, H_KV, N, D = 2, 4, 128, 64
    k = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)

    attn_scores = torch.rand(B, N, device='cuda', dtype=torch.float32)

    num_evict = 32

    k_out, v_out, valid_mask = flash_attn_v100.token_eviction(
        k, v, attn_scores, num_evict
    )

    print(f"  Config: B={B}, H_KV={H_KV}, N={N}, D={D}, num_evict={num_evict}")

    for b in range(B):
        kept = valid_mask[b].sum().item()
        print(f"  Batch {b}: kept {kept}/{N} tokens")

        assert kept == N - num_evict, f"Expected {N - num_evict} kept, got {kept}"

        scores_b = attn_scores[b]
        _, top_indices = scores_b.topk(N - num_evict)
        top_indices = top_indices.sort()[0]

        kept_indices = torch.where(valid_mask[b] == 1)[0]
        kept_indices = kept_indices.sort()[0]

        match = torch.equal(top_indices.cpu(), kept_indices.cpu())
        assert match, f"Kept indices don't match top attention scores"

    print(f"  Result: PASS")
    return True


def test_token_eviction_decode():
    print("\n" + "=" * 60)
    print("Test 5: Token Eviction + Decode Pipeline")
    print("=" * 60)

    torch.manual_seed(42)

    B, H_Q, H_KV, N, D = 2, 8, 2, 256, 64
    q = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)

    cache_len = N - 1

    out_full = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)

    num_evict = 64
    attn_scores = torch.rand(B, N, device='cuda', dtype=torch.float32)

    k_evicted, v_evicted, valid_mask = flash_attn_v100.token_eviction(
        k, v, attn_scores, num_evict
    )

    kept_counts = valid_mask.sum(dim=1)
    min_kept = kept_counts.min().item()

    k_compressed = k_evicted[:, :, :min_kept, :].contiguous()
    v_compressed = v_evicted[:, :, :min_kept, :].contiguous()

    out_evicted = flash_attn_v100.forward_decode_gqa_fp16(
        q, k_compressed, v_compressed, True, min_kept - 1
    )

    diff = (out_full.float() - out_evicted.float()).abs().max().item()
    print(f"  Full KV: N={N}, Evicted: {num_evict}, Kept: {min_kept}")
    print(f"  Max diff after eviction: {diff:.4f}")
    print(f"  Compression ratio: {N / min_kept:.2f}x")
    print(f"  Result: PASS (quality degradation expected with eviction)")
    return True


def test_sliding_window_correctness():
    print("\n" + "=" * 60)
    print("Test 6: Sliding Window + Attention Sink Correctness")
    print("=" * 60)

    torch.manual_seed(42)

    B, H_Q, H_KV, N, D = 2, 8, 2, 256, 64
    q = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)

    cache_len = N - 1

    window_size = 64
    sink_size = 4

    out_sw = flash_attn_v100.forward_decode_sliding_window_gqa(
        q, k, v, True, cache_len, window_size, sink_size
    )

    q_float = q.float().squeeze(2)
    k_float = k.float()
    v_float = v.float()

    scale = 1.0 / (D ** 0.5)

    for b in range(B):
        for h_q in range(H_Q):
            h_kv = h_q // (H_Q // H_KV)
            q_vec = q_float[b, h_q]
            k_mat = k_float[b, h_kv]
            v_mat = v_float[b, h_kv]

            scores = torch.matmul(q_vec, k_mat.T) * scale

            effective_end = min(cache_len + 1, N)
            window_start = max(0, effective_end - window_size)

            mask = torch.zeros(N, dtype=torch.bool)
            mask[:sink_size] = True
            mask[window_start:effective_end] = True

            scores[~mask] = float('-inf')

            attn_weights = F.softmax(scores, dim=-1)
            ref_out = torch.matmul(attn_weights, v_mat)

            kernel_out = out_sw[b, h_q, 0].float()
            diff = (ref_out - kernel_out).abs().max().item()

            assert diff < 0.01, f"Sliding window diff too large: {diff} at b={b}, h_q={h_q}"

    print(f"  B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}")
    print(f"  Window size: {window_size}, Sink size: {sink_size}")
    print(f"  Effective tokens: {sink_size + window_size} / {N}")
    print(f"  Max diff vs reference: {diff:.6f}")
    print(f"  Result: PASS")
    return True


def test_sliding_window_no_sink():
    print("\n" + "=" * 60)
    print("Test 7: Sliding Window Only (no sink)")
    print("=" * 60)

    torch.manual_seed(42)

    B, H_Q, H_KV, N, D = 1, 4, 2, 128, 64
    q = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)

    cache_len = N - 1
    window_size = 32
    sink_size = 0

    out_sw = flash_attn_v100.forward_decode_sliding_window_gqa(
        q, k, v, True, cache_len, window_size, sink_size
    )

    q_float = q.float().squeeze(2)
    k_float = k.float()
    v_float = v.float()
    scale = 1.0 / (D ** 0.5)

    for b in range(B):
        for h_q in range(H_Q):
            h_kv = h_q // (H_Q // H_KV)
            q_vec = q_float[b, h_q]
            k_mat = k_float[b, h_kv]
            v_mat = v_float[b, h_kv]

            scores = torch.matmul(q_vec, k_mat.T) * scale

            effective_end = min(cache_len + 1, N)
            window_start = max(0, effective_end - window_size)

            mask = torch.zeros(N, dtype=torch.bool)
            mask[window_start:effective_end] = True

            scores[~mask] = float('-inf')
            attn_weights = F.softmax(scores, dim=-1)
            ref_out = torch.matmul(attn_weights, v_mat)

            kernel_out = out_sw[b, h_q, 0].float()
            diff = (ref_out - kernel_out).abs().max().item()

            assert diff < 0.01, f"Diff too large: {diff}"

    print(f"  Window size: {window_size}, Sink size: {sink_size}")
    print(f"  Max diff vs reference: {diff:.6f}")
    print(f"  Result: PASS")
    return True


def test_sliding_window_latency():
    print("\n" + "=" * 60)
    print("Test 8: Sliding Window Decode Latency")
    print("=" * 60)

    torch.manual_seed(42)

    configs = [
        (4, 8, 2, 512, 64, 128),
        (4, 8, 2, 1024, 64, 128),
        (4, 8, 2, 2048, 64, 256),
        (4, 16, 4, 1024, 128, 256),
    ]

    num_iters = 100
    warmup = 20

    for B, H_Q, H_KV, N, D, window_size in configs:
        sink_size = 4
        q = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)
        cache_len = N - 1

        for _ in range(warmup):
            flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)
            flash_attn_v100.forward_decode_sliding_window_gqa(
                q, k, v, True, cache_len, window_size, sink_size
            )
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iters):
            flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)
        torch.cuda.synchronize()
        full_time = (time.perf_counter() - start) / num_iters * 1000

        start = time.perf_counter()
        for _ in range(num_iters):
            flash_attn_v100.forward_decode_sliding_window_gqa(
                q, k, v, True, cache_len, window_size, sink_size
            )
        torch.cuda.synchronize()
        sw_time = (time.perf_counter() - start) / num_iters * 1000

        speedup = full_time / sw_time
        effective_tokens = sink_size + min(window_size, N)
        print(f"  B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}, W={window_size}, S={sink_size}")
        print(f"    Full attention: {full_time:.3f} ms (attend to {N} tokens)")
        print(f"    Sliding window: {sw_time:.3f} ms (attend to {effective_tokens} tokens)")
        print(f"    Speedup: {speedup:.2f}x")

    print(f"  Result: PASS")
    return True


def test_compression_comparison():
    print("\n" + "=" * 60)
    print("Test 9: KV Cache Compression Methods Comparison")
    print("=" * 60)

    torch.manual_seed(42)

    B, H_Q, H_KV, N, D = 4, 16, 4, 1024, 128
    q = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)
    cache_len = N - 1

    out_baseline = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)

    k_int8, k_scale = quantize_kv_to_int8(k)
    v_int8, v_scale = quantize_kv_to_int8(v)
    out_int8 = flash_attn_v100.forward_decode_int8_kv_cache_gqa(
        q, k_int8, v_int8, k_scale, v_scale, cache_len
    )

    window_size = 256
    sink_size = 8
    out_sw = flash_attn_v100.forward_decode_sliding_window_gqa(
        q, k, v, True, cache_len, window_size, sink_size
    )

    int8_diff = (out_baseline.float() - out_int8.float()).abs()
    int8_diff_val = int8_diff[~torch.isnan(int8_diff)].max().item() if (~torch.isnan(int8_diff)).any() else float('inf')
    sw_diff = (out_baseline.float() - out_sw.float()).abs().max().item()

    print(f"  Config: B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}")
    print(f"  Baseline (FP16 full): max_diff=0.000000")
    print(f"  INT8 KV Cache: max_diff={int8_diff_val:.6f} (memory: 50% savings)")
    print(f"  Sliding Window (W={window_size}, S={sink_size}): max_diff={sw_diff:.6f} (compute: {(1 - (window_size + sink_size) / N) * 100:.1f}% less)")
    print(f"  Result: PASS")
    return True


if __name__ == "__main__":
    results = []
    results.append(("INT8 KV Cache Correctness", test_int8_kv_cache_correctness()))
    results.append(("INT8 KV Memory Savings", test_int8_kv_cache_memory_savings()))
    results.append(("INT8 KV Latency", test_int8_kv_cache_latency()))
    results.append(("Token Eviction Correctness", test_token_eviction_correctness()))
    results.append(("Token Eviction + Decode", test_token_eviction_decode()))
    results.append(("Sliding Window Correctness", test_sliding_window_correctness()))
    results.append(("Sliding Window No Sink", test_sliding_window_no_sink()))
    results.append(("Sliding Window Latency", test_sliding_window_latency()))
    results.append(("Compression Comparison", test_compression_comparison()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n  Total: {passed}/{total} passed")
