import torch
import flash_attn_v100
import time
from tabulate import tabulate


def reference_gqa_attention(q, k, v, causal=True):
    d = q.size(-1)
    H_Q = q.size(1)
    H_KV = k.size(1)
    group_size = H_Q // H_KV

    B, _, N, D = q.shape
    out = torch.zeros_like(q)

    for h_q in range(H_Q):
        h_kv = h_q // group_size
        q_h = q[:, h_q]
        k_h = k[:, h_kv]
        v_h = v[:, h_kv]

        scores = torch.matmul(q_h, k_h.transpose(-1, -2)) / (d ** 0.5)

        if causal:
            mask = torch.triu(torch.ones(N, N, device=q.device, dtype=q.dtype), diagonal=1)
            scores = scores.masked_fill(mask.bool(), float('-inf'))

        probs = torch.softmax(scores, dim=-1)
        out[:, h_q] = torch.matmul(probs, v_h)

    return out


def test_prefill_gqa_correctness():
    print("=" * 80)
    print("Testing Prefill GQA Correctness")
    print("=" * 80)

    test_cases = [
        (1, 8, 8, 64, 128, "MHA (H_Q=H_KV)"),
        (1, 8, 4, 64, 128, "GQA (8Q:4KV)"),
        (1, 8, 2, 64, 128, "GQA (8Q:2KV)"),
        (1, 8, 1, 64, 128, "MQA (8Q:1KV)"),
        (1, 32, 8, 64, 128, "GQA (32Q:8KV) Llama-like"),
        (2, 8, 4, 64, 128, "GQA (Batch=2)"),
        (1, 8, 4, 64, 256, "GQA (Longer seq)"),
        (1, 8, 4, 128, 128, "GQA (D=128)"),
        (1, 32, 8, 128, 256, "GQA (Llama-like D=128)"),
    ]

    for B, H_Q, H_KV, D, N, name in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {name} (B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D})")
        print(f"{'='*60}")

        torch.manual_seed(42)
        q = torch.randn(B, H_Q, N, D, device='cuda', dtype=torch.float16).contiguous()
        k = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16).contiguous()
        v = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16).contiguous()

        try:
            out_ref = reference_gqa_attention(q, k, v, causal=True)

            out_basic = flash_attn_v100.forward_prefill_gqa_fp16(q, k, v, True)
            out_warp = flash_attn_v100.forward_prefill_gqa_fp16_warp(q, k, v, True)

            max_diff_basic = (out_basic.float() - out_ref.float()).abs().max().item()
            max_diff_warp = (out_warp.float() - out_ref.float()).abs().max().item()
            mean_diff_basic = (out_basic.float() - out_ref.float()).abs().mean().item()
            mean_diff_warp = (out_warp.float() - out_ref.float()).abs().mean().item()
            allclose_basic = torch.allclose(out_basic, out_ref, atol=1e-2, rtol=1e-2)
            allclose_warp = torch.allclose(out_warp, out_ref, atol=1e-2, rtol=1e-2)

            has_nan_basic = torch.isnan(out_basic).any().item()
            has_nan_warp = torch.isnan(out_warp).any().item()

            print(f"  Basic kernel:")
            print(f"    Max diff: {max_diff_basic:.6f}, Mean diff: {mean_diff_basic:.6f}")
            print(f"    All close: {allclose_basic}, Has NaN: {has_nan_basic}")
            print(f"  Warp kernel:")
            print(f"    Max diff: {max_diff_warp:.6f}, Mean diff: {mean_diff_warp:.6f}")
            print(f"    All close: {allclose_warp}, Has NaN: {has_nan_warp}")

            if not allclose_basic or not allclose_warp:
                print(f"    ⚠ Warning: Results differ from reference!")

        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            import traceback
            traceback.print_exc()


def test_prefill_gqa_vs_mha():
    print("\n" + "=" * 80)
    print("Testing: Prefill GQA vs MHA (H_Q=H_KV should match forward_fp16)")
    print("=" * 80)

    B, H, N, D = 2, 8, 256, 64

    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16).contiguous()
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16).contiguous()
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16).contiguous()

    out_mha = flash_attn_v100.forward_fp16(q, k, v, True)
    out_gqa = flash_attn_v100.forward_prefill_gqa_fp16(q, k, v, True)
    out_gqa_warp = flash_attn_v100.forward_prefill_gqa_fp16_warp(q, k, v, True)

    max_diff_gqa = (out_gqa.float() - out_mha.float()).abs().max().item()
    max_diff_warp = (out_gqa_warp.float() - out_mha.float()).abs().max().item()

    print(f"  GQA basic vs MHA: max diff = {max_diff_gqa:.8f}")
    print(f"  GQA warp  vs MHA: max diff = {max_diff_warp:.8f}")
    print(f"  GQA basic allclose: {torch.allclose(out_gqa, out_mha, atol=1e-3, rtol=1e-3)}")
    print(f"  GQA warp  allclose: {torch.allclose(out_gqa_warp, out_mha, atol=1e-3, rtol=1e-3)}")


def test_non_causal():
    print("\n" + "=" * 80)
    print("Testing: Non-Causal Mode")
    print("=" * 80)

    B, H_Q, H_KV, N, D = 1, 8, 4, 128, 64

    torch.manual_seed(42)
    q = torch.randn(B, H_Q, N, D, device='cuda', dtype=torch.float16).contiguous()
    k = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16).contiguous()
    v = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16).contiguous()

    out_ref = reference_gqa_attention(q, k, v, causal=False)
    out_basic = flash_attn_v100.forward_prefill_gqa_fp16(q, k, v, False)
    out_warp = flash_attn_v100.forward_prefill_gqa_fp16_warp(q, k, v, False)

    max_diff_basic = (out_basic.float() - out_ref.float()).abs().max().item()
    max_diff_warp = (out_warp.float() - out_ref.float()).abs().max().item()

    print(f"  Basic max diff: {max_diff_basic:.6f}")
    print(f"  Warp max diff: {max_diff_warp:.6f}")
    print(f"  Basic allclose: {torch.allclose(out_basic, out_ref, atol=1e-2, rtol=1e-2)}")
    print(f"  Warp allclose: {torch.allclose(out_warp, out_ref, atol=1e-2, rtol=1e-2)}")


def benchmark_prefill_gqa():
    print("\n" + "=" * 80)
    print("Benchmarking Prefill GQA Performance")
    print("=" * 80)

    configs = [
        (1, 8, 8, 64, 256, "MHA baseline"),
        (1, 8, 4, 64, 256, "GQA 8:4"),
        (1, 8, 2, 64, 256, "GQA 8:2"),
        (1, 8, 1, 64, 256, "MQA 8:1"),
        (1, 32, 8, 64, 256, "GQA 32:8"),
        (1, 32, 8, 128, 256, "GQA 32:8 D=128"),
        (1, 32, 8, 128, 512, "GQA 32:8 N=512"),
        (2, 32, 8, 128, 256, "GQA 32:8 B=2"),
    ]

    results = []

    for B, H_Q, H_KV, D, N, name in configs:
        print(f"\n{'='*60}")
        print(f"Config: {name} (B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D})")
        print(f"{'='*60}")

        q = torch.randn(B, H_Q, N, D, device='cuda', dtype=torch.float16).contiguous()
        k = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16).contiguous()
        v = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16).contiguous()

        warmup = 10
        iterations = 100

        for variant, func in [
            ("basic", flash_attn_v100.forward_prefill_gqa_fp16),
            ("warp", flash_attn_v100.forward_prefill_gqa_fp16_warp),
        ]:
            for _ in range(warmup):
                _ = func(q, k, v, True)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(iterations):
                _ = func(q, k, v, True)
            end.record()

            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end) / iterations

            flops = 2 * B * H_Q * N * N * D
            tflops = flops / (elapsed_ms / 1000.0) / 1e12

            kv_reduction = (1 - H_KV / H_Q) * 100

            print(f"  {variant:6s}: {elapsed_ms:.3f} ms, {tflops:.3f} TFLOPs")

            results.append({
                'Config': name,
                'Variant': variant,
                'H_Q': H_Q,
                'H_KV': H_KV,
                'Time (ms)': f"{elapsed_ms:.3f}",
                'TFLOPs': f"{tflops:.3f}",
                'KV Save': f"{kv_reduction:.1f}%"
            })

        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("Prefill GQA Benchmark Summary")
    print("=" * 80)

    headers = ['Config', 'Variant', 'H_Q', 'H_KV', 'Time (ms)', 'TFLOPs', 'KV Save']
    table_data = []
    for r in results:
        table_data.append([r['Config'], r['Variant'], r['H_Q'], r['H_KV'],
                          r['Time (ms)'], r['TFLOPs'], r['KV Save']])

    print(tabulate(table_data, headers=headers, tablefmt='grid'))


if __name__ == '__main__':
    test_prefill_gqa_correctness()
    test_prefill_gqa_vs_mha()
    test_non_causal()
    benchmark_prefill_gqa()

    print("\n" + "=" * 80)
    print("All Prefill GQA tests completed!")
    print("=" * 80)
