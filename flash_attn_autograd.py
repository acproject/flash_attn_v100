import torch
import flash_attn_v100


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=True):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if q.dtype == torch.float16:
            out, logsumexp = flash_attn_v100.forward_fp16_with_lse(q, k, v, causal)
        elif q.dtype == torch.float32:
            out, logsumexp = flash_attn_v100.forward_with_lse(q, k, v, causal)
        else:
            raise TypeError("flash_attn only supports float16 and float32")

        ctx.save_for_backward(q, k, v, out, logsumexp)
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, logsumexp = ctx.saved_tensors
        dq, dk, dv = flash_attn_v100.backward(
            dout.contiguous(), q, k, v, out, logsumexp, ctx.causal
        )
        return dq, dk, dv, None


def flash_attn(q, k, v, causal=True):
    return FlashAttentionFunction.apply(q, k, v, causal)
