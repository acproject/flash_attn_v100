import torch
import flash_attn_v100


def reference(q, k, v, causal=True):
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)

    if causal:
        n = q.size(-2)
        mask = torch.triu(torch.ones(n, n, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float('-inf'))

    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def main():
    torch.manual_seed(0)

    B, H, N, D = 2, 8, 256, 64

    # 测试 FP32 版本
    print("=" * 50)
    print("Testing FP32 Version")
    print("=" * 50)
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)

    out_ref = reference(q, k, v, causal=True)
    out = flash_attn_v100.forward(q.contiguous(),
                                  k.contiguous(),
                                  v.contiguous(),
                                  True)

    max_diff = (out - out_ref).abs().max().item()
    print('max diff =', max_diff)
    print('allclose =', torch.allclose(out, out_ref, atol=1e-4, rtol=1e-4))

    # 测试 FP16 版本
    print("\n" + "=" * 50)
    print("Testing FP16 Version")
    print("=" * 50)
    q_fp16 = q.half()
    k_fp16 = k.half()
    v_fp16 = v.half()

    out_ref_fp16 = reference(q_fp16, k_fp16, v_fp16, causal=True)
    out_fp16 = flash_attn_v100.forward_fp16(q_fp16.contiguous(),
                                            k_fp16.contiguous(),
                                            v_fp16.contiguous(),
                                            True)

    max_diff_fp16 = (out_fp16 - out_ref_fp16).abs().max().item()
    print('max diff (FP16) =', max_diff_fp16)
    print('allclose (FP16) =', torch.allclose(out_fp16, out_ref_fp16, atol=1e-2, rtol=1e-2))
    
    # 对比 FP32 和 FP16 的差异
    print("\n" + "=" * 50)
    print("FP32 vs FP16 Comparison")
    print("=" * 50)
    max_diff_32_vs_16 = (out.float() - out_fp16.float()).abs().max().item()
    print('max diff (FP32 vs FP16) =', max_diff_32_vs_16)

    print("\n" + "=" * 50)
    print("Reliability Checks")
    print("=" * 50)

    try:
        flash_attn_v100.forward_fp16(q_fp16.contiguous(),
                                     k.contiguous(),
                                     v_fp16.contiguous(),
                                     True)
        print("dtype mismatch check = FAILED")
    except RuntimeError as exc:
        print("dtype mismatch check = PASS")
        print("message =", str(exc).splitlines()[0])

    try:
        flash_attn_v100.forward_decode_gqa_fp16(q_fp16[:, :, :1, :].contiguous(),
                                                k_fp16.contiguous(),
                                                v_fp16.contiguous(),
                                                True,
                                                -1)
        print("negative cache_len check = FAILED")
    except RuntimeError as exc:
        print("negative cache_len check = PASS")
        print("message =", str(exc).splitlines()[0])

    if torch.cuda.device_count() >= 2:
        try:
            flash_attn_v100.forward_prefill_gqa_fp16(
                q_fp16.to('cuda:0').contiguous(),
                k_fp16.to('cuda:1').contiguous(),
                v_fp16.to('cuda:1').contiguous(),
                True
            )
            print("cross-device check = FAILED")
        except RuntimeError as exc:
            print("cross-device check = PASS")
            print("message =", str(exc).splitlines()[0])

    print("\n" + "=" * 50)
    print("Output Reuse Checks")
    print("=" * 50)

    B_gqa, H_Q, H_KV, N_gqa, D_gqa = 2, 16, 4, 128, 64
    q_decode = torch.randn(B_gqa, H_Q, 1, D_gqa, device='cuda', dtype=torch.float16).contiguous()
    k_gqa = torch.randn(B_gqa, H_KV, N_gqa, D_gqa, device='cuda', dtype=torch.float16).contiguous()
    v_gqa = torch.randn(B_gqa, H_KV, N_gqa, D_gqa, device='cuda', dtype=torch.float16).contiguous()

    decode_ref = flash_attn_v100.forward_decode_gqa_fp16(q_decode, k_gqa, v_gqa, True, N_gqa - 1)
    decode_out = torch.full_like(decode_ref, 123.0)
    decode_out_ptr = decode_out.data_ptr()
    decode_result = flash_attn_v100.forward_decode_gqa_fp16_out(decode_out, q_decode, k_gqa, v_gqa, True, N_gqa - 1)
    decode_diff = (decode_result - decode_ref).abs().max().item()
    print('decode out max diff =', decode_diff)
    print('decode out reused buffer =', decode_result.data_ptr() == decode_out_ptr)

    q_prefill = torch.randn(B_gqa, H_Q, N_gqa, D_gqa, device='cuda', dtype=torch.float16).contiguous()
    k_prefill = torch.randn(B_gqa, H_KV, N_gqa, D_gqa, device='cuda', dtype=torch.float16).contiguous()
    v_prefill = torch.randn(B_gqa, H_KV, N_gqa, D_gqa, device='cuda', dtype=torch.float16).contiguous()

    prefill_ref = flash_attn_v100.forward_prefill_gqa_fp16(q_prefill, k_prefill, v_prefill, True)
    prefill_out = torch.full_like(prefill_ref, -7.0)
    prefill_out_ptr = prefill_out.data_ptr()
    prefill_result = flash_attn_v100.forward_prefill_gqa_fp16_out(prefill_out, q_prefill, k_prefill, v_prefill, True)
    prefill_diff = (prefill_result - prefill_ref).abs().max().item()
    print('prefill out max diff =', prefill_diff)
    print('prefill out reused buffer =', prefill_result.data_ptr() == prefill_out_ptr)

    try:
        bad_out = torch.empty(B_gqa, H_Q, 2, D_gqa, device='cuda', dtype=torch.float16)
        flash_attn_v100.forward_decode_gqa_fp16_out(bad_out, q_decode, k_gqa, v_gqa, True, N_gqa - 1)
        print("bad out shape check = FAILED")
    except RuntimeError as exc:
        print("bad out shape check = PASS")
        print("message =", str(exc).splitlines()[0])


if __name__ == '__main__':
    main()
