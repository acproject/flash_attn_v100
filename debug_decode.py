import torch
import flash_attn_v100

def debug_decode():
    """调试 decode kernel 的正确性"""
    
    # 使用很小的测试用例
    B, H, D = 1, 1, 4  # 简化配置
    cache_len = 3
    total_len = cache_len + 1  # 4
    
    torch.manual_seed(42)
    
    q = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16).contiguous()
    k = torch.randn(B, H, total_len, D, device='cuda', dtype=torch.float16).contiguous()
    v = torch.randn(B, H, total_len, D, device='cuda', dtype=torch.float16).contiguous()
    
    print("Input shapes:")
    print(f"  Q: {q.shape}")
    print(f"  K: {k.shape}")
    print(f"  V: {v.shape}")
    
    # PyTorch 参考实现
    def reference_decode(q, k, v, cache_len, causal=True):
        d = q.size(-1)
        scores = torch.matmul(q.float(), k.transpose(-1, -2).float()) / (d ** 0.5)
        
        if causal:
            # Q 的位置是 cache_len (索引从 0 开始)
            # K 的索引 0, 1, 2, 3 对应位置 0, 1, 2, 3
            # Q 在位置 3，应该只能看到位置 0, 1, 2, 3
            mask = torch.ones(1, total_len, device=q.device)
            mask[:, cache_len+1:] = float('-inf')  # mask out 位置 4+
            scores = scores + mask
        
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v.float())
        return out
    
    # 测试 causal
    print("\n" + "="*60)
    print("Causal Mode")
    print("="*60)
    
    out_ref = reference_decode(q, k, v, cache_len, causal=True)
    out_cuda = flash_attn_v100.forward_decode_fp16(q, k, v, True, cache_len)
    
    print(f"\nReference output:\n{out_ref}")
    print(f"\nCUDA output:\n{out_cuda.float()}")
    
    diff = (out_cuda.float() - out_ref).abs()
    print(f"\nAbsolute diff:\n{diff}")
    print(f"Max diff: {diff.max().item():.6f}")
    print(f"Mean diff: {diff.mean().item():.6f}")
    
    # 测试非 causal
    print("\n" + "="*60)
    print("Non-Causal Mode")
    print("="*60)
    
    out_ref_nc = reference_decode(q, k, v, cache_len, causal=False)
    out_cuda_nc = flash_attn_v100.forward_decode_fp16(q, k, v, False, cache_len)
    
    print(f"\nReference output:\n{out_ref_nc}")
    print(f"\nCUDA output:\n{out_cuda_nc.float()}")
    
    diff_nc = (out_cuda_nc.float() - out_ref_nc).abs()
    print(f"\nAbsolute diff:\n{diff_nc}")
    print(f"Max diff: {diff_nc.max().item():.6f}")
    print(f"Mean diff: {diff_nc.mean().item():.6f}")

if __name__ == '__main__':
    debug_decode()
