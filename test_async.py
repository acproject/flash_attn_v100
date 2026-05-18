import torch
import time
import flash_attn_v100


class PipelineDecodeWorkspace:
    def __init__(self, B, H_Q, H_KV, N, D):
        self.out = torch.empty(B, H_Q, 1, D, device='cuda', dtype=torch.float16)
        self.q_cpu = flash_attn_v100.alloc_pinned_tensor([B, H_Q, 1, D], 5)
        self.k_cpu = flash_attn_v100.alloc_pinned_tensor([B, H_KV, N, D], 5)
        self.v_cpu = flash_attn_v100.alloc_pinned_tensor([B, H_KV, N, D], 5)
        self.q_dst = torch.empty(B, H_Q, 1, D, device='cuda', dtype=torch.float16)
        self.k_dst = torch.empty(B, H_KV, N, D, device='cuda', dtype=torch.float16)
        self.v_dst = torch.empty(B, H_KV, N, D, device='cuda', dtype=torch.float16)

    def load_random_inputs(self, B, H_Q, H_KV, N, D):
        self.q_cpu.copy_(torch.randn(B, H_Q, 1, D, dtype=torch.float16))
        self.k_cpu.copy_(torch.randn(B, H_KV, N, D, dtype=torch.float16))
        self.v_cpu.copy_(torch.randn(B, H_KV, N, D, dtype=torch.float16))

def ref_attention(q, k, v, causal=False):
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    scale = 1.0 / (q.size(-1) ** 0.5)
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
    if causal:
        seq_q = q.size(2)
        seq_k = k.size(2)
        mask = torch.triu(torch.ones(seq_q, seq_k, device=q.device), diagonal=seq_k - seq_q + 1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_f).half()
    return out


def test_stream_decode_correctness():
    print("=" * 60)
    print("Test 1: Stream Decode GQA Correctness")
    print("=" * 60)

    B, H_Q, H_KV, N, D = 2, 8, 2, 512, 64
    torch.manual_seed(42)

    q = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)

    out_default = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, 0)

    stream = torch.cuda.Stream()
    stream_ptr = stream.cuda_stream

    with torch.cuda.stream(stream):
        out_stream = flash_attn_v100.forward_decode_gqa_fp16_stream(
            q, k, v, True, 0, stream_ptr
        )

    torch.cuda.synchronize()

    diff = (out_default.float() - out_stream.float()).abs().max().item()
    print(f"  B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}")
    print(f"  Max diff (stream vs default): {diff:.8f}")
    print(f"  Result: {'PASS' if diff < 1e-5 else 'FAIL'}")
    return diff < 1e-5


def test_stream_varlen_prefill_correctness():
    print("\n" + "=" * 60)
    print("Test 2: Stream Varlen Prefill Correctness")
    print("=" * 60)

    H, D = 4, 64
    seq_lens = [64, 128, 256]
    torch.manual_seed(42)

    total_q = sum(seq_lens)
    q_packed = torch.randn(total_q, H, D, device='cuda', dtype=torch.float16)
    k_packed = torch.randn(total_q, H, D, device='cuda', dtype=torch.float16)
    v_packed = torch.randn(total_q, H, D, device='cuda', dtype=torch.float16)

    cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0)),
                              device='cuda', dtype=torch.int32)
    max_seqlen = max(seq_lens)

    out_default = flash_attn_v100.forward_varlen_prefill_fp16(
        q_packed, k_packed, v_packed, cu_seqlens, max_seqlen, True
    )

    stream = torch.cuda.Stream()
    stream_ptr = stream.cuda_stream

    with torch.cuda.stream(stream):
        out_stream = flash_attn_v100.forward_varlen_prefill_fp16_stream(
            q_packed, k_packed, v_packed, cu_seqlens, max_seqlen, True, stream_ptr
        )

    torch.cuda.synchronize()

    diff = (out_default.float() - out_stream.float()).abs().max().item()
    print(f"  H={H}, D={D}, seq_lens={seq_lens}")
    print(f"  Max diff (stream vs default): {diff:.8f}")
    print(f"  Result: {'PASS' if diff < 1e-5 else 'FAIL'}")
    return diff < 1e-5


def test_async_h2d_transfer():
    print("\n" + "=" * 60)
    print("Test 3: Async H2D Transfer")
    print("=" * 60)

    B, H, D = 2, 4, 64
    torch.manual_seed(42)

    q_cpu_pinned = flash_attn_v100.alloc_pinned_tensor(
        [B, H, 1, D], 5
    )
    q_cpu_pinned.copy_(torch.randn(B, H, 1, D, dtype=torch.float16))

    q_gpu = torch.zeros(B, H, 1, D, device='cuda', dtype=torch.float16)

    stream = torch.cuda.Stream()
    stream_ptr = stream.cuda_stream

    flash_attn_v100.async_h2d_transfer(q_gpu, q_cpu_pinned, stream_ptr)

    torch.cuda.synchronize()

    q_cpu_ref = q_cpu_pinned.cuda()
    diff = (q_gpu.float() - q_cpu_ref.float()).abs().max().item()
    print(f"  Shape: {list(q_gpu.shape)}")
    print(f"  Max diff (after async transfer): {diff:.8f}")
    print(f"  Result: {'PASS' if diff < 1e-5 else 'FAIL'}")
    return diff < 1e-5


def test_pinned_tensor_alloc():
    print("\n" + "=" * 60)
    print("Test 4: Pinned Tensor Allocation")
    print("=" * 60)

    sizes = [2, 8, 512, 64]
    tensor = flash_attn_v100.alloc_pinned_tensor(sizes, 5)

    print(f"  Shape: {list(tensor.shape)}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Is pinned: {tensor.is_pinned()}")

    ok = (list(tensor.shape) == sizes and
          tensor.dtype == torch.float16 and
          tensor.is_pinned())
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def test_pipeline_decode_step():
    print("\n" + "=" * 60)
    print("Test 5: Pipeline Decode Step (Compute + Transfer Overlap)")
    print("=" * 60)

    B, H_Q, H_KV, N, D = 2, 8, 2, 512, 64
    torch.manual_seed(42)

    q_compute = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16)
    k_compute = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)
    v_compute = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)

    workspace = PipelineDecodeWorkspace(B, H_Q, H_KV, N, D)
    workspace.load_random_inputs(B, H_Q, H_KV, N, D)

    compute_stream = torch.cuda.Stream()
    transfer_stream = torch.cuda.Stream()

    out_ptr = workspace.out.data_ptr()
    out, q_dst, k_dst = flash_attn_v100.pipeline_decode_step_out(
        workspace.out,
        q_compute, k_compute, v_compute,
        workspace.q_cpu, workspace.k_cpu, workspace.v_cpu,
        workspace.q_dst, workspace.k_dst, workspace.v_dst,
        True, 0,
        compute_stream.cuda_stream,
        transfer_stream.cuda_stream
    )

    torch.cuda.synchronize()

    ref_out = flash_attn_v100.forward_decode_gqa_fp16(q_compute, k_compute, v_compute, True, 0)
    out_diff = (out.float() - ref_out.float()).abs().max().item()

    q_ref = workspace.q_cpu.cuda()
    q_diff = (q_dst.float() - q_ref.float()).abs().max().item()

    k_ref = workspace.k_cpu.cuda()
    k_diff = (k_dst.float() - k_ref.float()).abs().max().item()

    print(f"  Compute output max diff: {out_diff:.8f}")
    print(f"  Q transfer max diff: {q_diff:.8f}")
    print(f"  K transfer max diff: {k_diff:.8f}")
    print(f"  Output buffer reused: {out.data_ptr() == out_ptr}")

    ok = out_diff < 1e-5 and q_diff < 1e-5 and k_diff < 1e-5 and out.data_ptr() == out_ptr
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def test_pipeline_latency():
    print("\n" + "=" * 60)
    print("Test 6: Pipeline vs Sequential Latency Comparison")
    print("=" * 60)

    B, H_Q, H_KV, N, D = 4, 8, 2, 1024, 64
    num_iters = 100
    warmup = 20

    torch.manual_seed(42)
    q = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)

    workspace = PipelineDecodeWorkspace(B, H_Q, H_KV, N, D)
    workspace.load_random_inputs(B, H_Q, H_KV, N, D)

    for _ in range(warmup):
        flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, 0)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, 0)
        workspace.q_dst.copy_(workspace.q_cpu)
        workspace.k_dst.copy_(workspace.k_cpu)
        workspace.v_dst.copy_(workspace.v_cpu)
    torch.cuda.synchronize()
    seq_time = (time.perf_counter() - start) / num_iters * 1000

    compute_stream = torch.cuda.Stream()
    transfer_stream = torch.cuda.Stream()

    for _ in range(warmup):
        flash_attn_v100.pipeline_decode_step_out(
            workspace.out,
            q, k, v, workspace.q_cpu, workspace.k_cpu, workspace.v_cpu,
            workspace.q_dst, workspace.k_dst, workspace.v_dst, True, 0,
            compute_stream.cuda_stream,
            transfer_stream.cuda_stream
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        flash_attn_v100.pipeline_decode_step_out(
            workspace.out,
            q, k, v, workspace.q_cpu, workspace.k_cpu, workspace.v_cpu,
            workspace.q_dst, workspace.k_dst, workspace.v_dst, True, 0,
            compute_stream.cuda_stream,
            transfer_stream.cuda_stream
        )
    torch.cuda.synchronize()
    pipe_time = (time.perf_counter() - start) / num_iters * 1000

    speedup = seq_time / pipe_time
    print(f"  Config: B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}")
    print(f"  Sequential (compute + transfer): {seq_time:.3f} ms")
    print(f"  Pipeline (overlapped):           {pipe_time:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    return True


if __name__ == "__main__":
    results = []
    results.append(("Stream Decode Correctness", test_stream_decode_correctness()))
    results.append(("Stream Varlen Prefill Correctness", test_stream_varlen_prefill_correctness()))
    results.append(("Async H2D Transfer", test_async_h2d_transfer()))
    results.append(("Pinned Tensor Allocation", test_pinned_tensor_alloc()))
    results.append(("Pipeline Decode Step", test_pipeline_decode_step()))
    results.append(("Pipeline Latency", test_pipeline_latency()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n  Total: {passed}/{total} passed")
