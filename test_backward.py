import torch

from flash_attn_autograd import flash_attn


def reference(q, k, v, causal=True):
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)

    if causal:
        n = q.size(-2)
        mask = torch.triu(torch.ones(n, n, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def clone_with_grad(x):
    return x.detach().clone().contiguous().requires_grad_(True)


def compare_backward(dtype, causal):
    torch.manual_seed(7)
    B, H, N, D = 2, 3, 17, 32

    q0 = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    k0 = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    v0 = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    dout = torch.randn(B, H, N, D, device="cuda", dtype=dtype)

    q = clone_with_grad(q0)
    k = clone_with_grad(k0)
    v = clone_with_grad(v0)
    out = flash_attn(q, k, v, causal=causal)
    out.backward(dout)

    q_ref = clone_with_grad(q0)
    k_ref = clone_with_grad(k0)
    v_ref = clone_with_grad(v0)
    out_ref = reference(q_ref, k_ref, v_ref, causal=causal)
    out_ref.backward(dout)

    atol = 2e-4 if dtype == torch.float32 else 3e-2
    rtol = 2e-4 if dtype == torch.float32 else 3e-2

    for name, got, expected in (
        ("dq", q.grad, q_ref.grad),
        ("dk", k.grad, k_ref.grad),
        ("dv", v.grad, v_ref.grad),
    ):
        max_diff = (got.float() - expected.float()).abs().max().item()
        ok = torch.allclose(got.float(), expected.float(), atol=atol, rtol=rtol)
        print(f"{dtype} causal={causal} {name}: max_diff={max_diff:.6f} allclose={ok}")
        assert ok


def main():
    compare_backward(torch.float32, causal=True)
    compare_backward(torch.float32, causal=False)
    compare_backward(torch.float16, causal=True)
    compare_backward(torch.float16, causal=False)


if __name__ == "__main__":
    main()

