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


if __name__ == '__main__':
    main()