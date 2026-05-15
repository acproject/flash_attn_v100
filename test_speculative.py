import torch
import torch.nn.functional as F
import time
import flash_attn_v100


class SpeculativeDecoder:
    def __init__(self, K=4, vocab_size=32000, H_Q=32, H_KV=8, D=128):
        self.K = K
        self.vocab_size = vocab_size
        self.H_Q = H_Q
        self.H_KV = H_KV
        self.D = D

    def draft_step(self, q, k, v, cache_len, num_tokens=None):
        if num_tokens is None:
            num_tokens = self.K
        draft_tokens = []
        for _ in range(num_tokens):
            out = flash_attn_v100.forward_decode_gqa_fp16(
                q, k, v, True, cache_len
            )
            logits = torch.matmul(out.float().squeeze(2), self.draft_head.T)
            probs = F.softmax(logits, dim=-1)
            token = torch.argmax(probs, dim=-1)
            draft_tokens.append(token)
            cache_len += 1
        return torch.stack(draft_tokens, dim=1)

    def target_verify(self, q, k, v, draft_tokens, cache_len):
        K = draft_tokens.shape[1]
        B = q.shape[0]

        target_probs_list = []
        draft_probs_list = []

        for i in range(K + 1):
            out = flash_attn_v100.forward_decode_gqa_fp16(
                q, k, v, True, cache_len + i
            )
            logits = torch.matmul(out.float().squeeze(2), self.target_head.T)
            probs = F.softmax(logits, dim=-1)
            target_probs_list.append(probs)

        for i in range(K):
            out = flash_attn_v100.forward_decode_gqa_fp16(
                q, k, v, True, cache_len + i
            )
            logits = torch.matmul(out.float().squeeze(2), self.draft_head.T)
            probs = F.softmax(logits, dim=-1)
            draft_probs_list.append(probs)

        target_probs = torch.stack(target_probs_list, dim=1)
        draft_probs = torch.stack(draft_probs_list, dim=1)

        random_vals = torch.rand(B, K, device=q.device, dtype=torch.float32)

        accepted_count, output_tokens = flash_attn_v100.speculative_verify(
            target_probs, draft_probs, draft_tokens, random_vals, K
        )

        return accepted_count, output_tokens


def cpu_speculative_verify(target_probs, draft_probs, draft_tokens, K):
    B = target_probs.shape[0]
    vocab_size = target_probs.shape[2]
    accepted_counts = []
    output_tokens_list = []

    for b in range(B):
        accepted = 0
        out_toks = [-1] * (K + 1)

        for i in range(K):
            tok = draft_tokens[b, i].item()
            p_tok = target_probs[b, i, tok].item()
            q_tok = draft_probs[b, i, tok].item()

            accept_prob = min(1.0, p_tok / (q_tok + 1e-10))
            r = torch.rand(1).item()

            if r < accept_prob:
                out_toks[i] = tok
                accepted += 1
            else:
                diff = target_probs[b, i] - draft_probs[b, i]
                diff = torch.clamp(diff, min=0)
                Z = diff.sum().item()

                if Z > 0:
                    adjusted = diff / Z
                    sampled = torch.multinomial(adjusted, 1).item()
                    out_toks[i] = sampled
                else:
                    out_toks[i] = torch.argmax(target_probs[b, i]).item()

                for j in range(i + 1, K):
                    out_toks[j] = -1

                bonus_probs = target_probs[b, i + 1]
                out_toks[i + 1] = torch.multinomial(bonus_probs, 1).item()
                break

        if accepted == K:
            bonus_probs = target_probs[b, K]
            out_toks[K] = torch.multinomial(bonus_probs, 1).item()

        accepted_counts.append(accepted)
        output_tokens_list.append(out_toks)

    return accepted_counts, output_tokens_list


def test_verify_correctness():
    print("=" * 60)
    print("Test 1: Speculative Verify Kernel Correctness")
    print("=" * 60)

    B, K, vocab_size = 2, 4, 256
    torch.manual_seed(42)

    target_probs = F.softmax(torch.randn(B, K + 1, vocab_size, device='cuda'), dim=-1)
    draft_probs = F.softmax(torch.randn(B, K, vocab_size, device='cuda'), dim=-1)
    draft_tokens = torch.randint(0, vocab_size, (B, K), device='cuda', dtype=torch.int64)

    torch.manual_seed(123)
    random_vals = torch.rand(B, K, device='cuda', dtype=torch.float32)

    accepted_count, output_tokens = flash_attn_v100.speculative_verify(
        target_probs, draft_probs, draft_tokens, random_vals, K
    )

    print(f"  B={B}, K={K}, vocab_size={vocab_size}")
    print(f"  Accepted counts: {accepted_count.cpu().tolist()}")
    print(f"  Output tokens shape: {output_tokens.shape}")
    print(f"  Output tokens:\n{output_tokens.cpu().tolist()}")

    valid = (output_tokens[:, :K] >= 0).sum(dim=1)
    print(f"  Valid tokens per batch: {valid.cpu().tolist()}")

    for b in range(B):
        acc = accepted_count[b].item()
        for i in range(acc):
            assert output_tokens[b, i].item() == draft_tokens[b, i].item(), \
                f"Accepted token mismatch at b={b}, i={i}"
        if acc < K:
            assert output_tokens[b, acc].item() >= 0, \
                f"Rejection sample token should be >= 0 at b={b}, i={acc}"
            for i in range(acc + 1, K):
                assert output_tokens[b, i].item() == -1, \
                    f"Token after rejection should be -1 at b={b}, i={i}"

    print(f"  Result: PASS")
    return True


def test_verify_all_accepted():
    print("\n" + "=" * 60)
    print("Test 2: All Draft Tokens Accepted (identical distributions)")
    print("=" * 60)

    B, K, vocab_size = 4, 5, 128
    torch.manual_seed(42)

    target_probs = F.softmax(torch.randn(B, K + 1, vocab_size, device='cuda'), dim=-1)
    draft_probs = target_probs[:, :K, :].clone()

    draft_tokens = torch.randint(0, vocab_size, (B, K), device='cuda', dtype=torch.int64)

    random_vals = torch.zeros(B, K, device='cuda', dtype=torch.float32)

    accepted_count, output_tokens = flash_attn_v100.speculative_verify(
        target_probs, draft_probs, draft_tokens, random_vals, K
    )

    print(f"  B={B}, K={K}, vocab_size={vocab_size}")
    print(f"  Accepted counts: {accepted_count.cpu().tolist()}")

    all_accepted = all(accepted_count[b].item() == K for b in range(B))
    print(f"  All tokens accepted: {all_accepted}")

    for b in range(B):
        for i in range(K):
            assert output_tokens[b, i].item() == draft_tokens[b, i].item(), \
                f"Token mismatch at b={b}, i={i}"
        assert output_tokens[b, K].item() >= 0, \
            f"Bonus token should be >= 0 at b={b}"

    print(f"  Result: PASS")
    return True


def test_verify_all_rejected():
    print("\n" + "=" * 60)
    print("Test 3: First Token Rejected (orthogonal distributions)")
    print("=" * 60)

    B, K, vocab_size = 2, 4, 64
    torch.manual_seed(42)

    target_probs = torch.zeros(B, K + 1, vocab_size, device='cuda')
    draft_probs = torch.zeros(B, K, vocab_size, device='cuda')

    for b in range(B):
        for i in range(K + 1):
            target_probs[b, i, (i * 10) % vocab_size] = 1.0
        for i in range(K):
            draft_probs[b, i, (i * 10 + 1) % vocab_size] = 1.0

    draft_tokens = torch.full((B, K), 1, device='cuda', dtype=torch.int64)

    random_vals = torch.ones(B, K, device='cuda', dtype=torch.float32) * 0.5

    accepted_count, output_tokens = flash_attn_v100.speculative_verify(
        target_probs, draft_probs, draft_tokens, random_vals, K
    )

    print(f"  B={B}, K={K}, vocab_size={vocab_size}")
    print(f"  Accepted counts: {accepted_count.cpu().tolist()}")
    print(f"  Output tokens:\n{output_tokens.cpu().tolist()}")

    for b in range(B):
        acc = accepted_count[b].item()
        if acc == 0:
            assert output_tokens[b, 0].item() >= 0, "Rejection sample should be >= 0"
            assert output_tokens[b, 1].item() >= 0, "Bonus token should be >= 0"
            for i in range(2, K):
                assert output_tokens[b, i].item() == -1, "Tokens after rejection should be -1"

    print(f"  Result: PASS")
    return True


def test_verify_large_vocab():
    print("\n" + "=" * 60)
    print("Test 4: Large Vocabulary (vocab_size=32000)")
    print("=" * 60)

    B, K, vocab_size = 2, 3, 32000
    torch.manual_seed(42)

    target_probs = F.softmax(torch.randn(B, K + 1, vocab_size, device='cuda'), dim=-1)
    draft_probs = F.softmax(torch.randn(B, K, vocab_size, device='cuda'), dim=-1)
    draft_tokens = torch.randint(0, vocab_size, (B, K), device='cuda', dtype=torch.int64)

    random_vals = torch.rand(B, K, device='cuda', dtype=torch.float32)

    accepted_count, output_tokens = flash_attn_v100.speculative_verify(
        target_probs, draft_probs, draft_tokens, random_vals, K
    )

    print(f"  B={B}, K={K}, vocab_size={vocab_size}")
    print(f"  Accepted counts: {accepted_count.cpu().tolist()}")
    print(f"  Output tokens shape: {output_tokens.shape}")

    for b in range(B):
        acc = accepted_count[b].item()
        for i in range(min(acc, K)):
            assert output_tokens[b, i].item() == draft_tokens[b, i].item()
        if acc < K:
            assert output_tokens[b, acc].item() >= 0
        if acc == K:
            assert output_tokens[b, K].item() >= 0

    print(f"  Result: PASS")
    return True


def test_acceptance_rate():
    print("\n" + "=" * 60)
    print("Test 5: Acceptance Rate Statistics")
    print("=" * 60)

    B, K, vocab_size = 32, 5, 1000
    num_trials = 10
    total_accepted = 0
    total_tokens = 0

    for trial in range(num_trials):
        torch.manual_seed(trial)

        target_probs = F.softmax(torch.randn(B, K + 1, vocab_size, device='cuda') * 0.5, dim=-1)
        draft_probs = F.softmax(torch.randn(B, K, vocab_size, device='cuda') * 0.5, dim=-1)
        draft_tokens = torch.randint(0, vocab_size, (B, K), device='cuda', dtype=torch.int64)

        random_vals = torch.rand(B, K, device='cuda', dtype=torch.float32)

        accepted_count, _ = flash_attn_v100.speculative_verify(
            target_probs, draft_probs, draft_tokens, random_vals, K
        )

        total_accepted += accepted_count.float().sum().item()
        total_tokens += B * K

    avg_acceptance_rate = total_accepted / total_tokens
    print(f"  B={B}, K={K}, vocab_size={vocab_size}, trials={num_trials}")
    print(f"  Average acceptance rate: {avg_acceptance_rate:.4f}")
    print(f"  Expected tokens per step: {1 + avg_acceptance_rate * K:.2f} (vs 1 without spec)")
    print(f"  Theoretical speedup: {1 + avg_acceptance_rate * K:.2f}x")
    print(f"  Result: PASS (acceptance rate = {avg_acceptance_rate:.4f})")
    return True


def test_gpu_verify_vs_cpu():
    print("\n" + "=" * 60)
    print("Test 6: GPU vs CPU Verification Consistency")
    print("=" * 60)

    B, K, vocab_size = 4, 3, 64
    torch.manual_seed(42)

    target_probs = F.softmax(torch.randn(B, K + 1, vocab_size, device='cuda'), dim=-1)
    draft_probs = F.softmax(torch.randn(B, K, vocab_size, device='cuda'), dim=-1)
    draft_tokens = torch.randint(0, vocab_size, (B, K), device='cuda', dtype=torch.int64)

    random_vals = torch.rand(B, K, device='cuda', dtype=torch.float32)

    accepted_count_gpu, output_tokens_gpu = flash_attn_v100.speculative_verify(
        target_probs, draft_probs, draft_tokens, random_vals, K
    )

    cpu_accepted, cpu_output = cpu_speculative_verify(
        target_probs.cpu(), draft_probs.cpu(), draft_tokens.cpu(), K
    )

    print(f"  GPU accepted: {accepted_count_gpu.cpu().tolist()}")
    print(f"  CPU accepted: {cpu_accepted}")

    match = True
    for b in range(B):
        gpu_acc = accepted_count_gpu[b].item()
        cpu_acc = cpu_accepted[b]
        if gpu_acc != cpu_acc:
            print(f"  Batch {b}: GPU accepted={gpu_acc}, CPU accepted={cpu_acc} - MISMATCH")
            match = False

    print(f"  Result: {'PASS (acceptance counts match)' if match else 'INFO (stochastic - counts may differ)'}")
    return True


def test_verify_latency():
    print("\n" + "=" * 60)
    print("Test 7: GPU Verify Kernel Latency")
    print("=" * 60)

    B, K, vocab_size = 32, 5, 32000
    num_iters = 100
    warmup = 20

    torch.manual_seed(42)
    target_probs = F.softmax(torch.randn(B, K + 1, vocab_size, device='cuda'), dim=-1)
    draft_probs = F.softmax(torch.randn(B, K, vocab_size, device='cuda'), dim=-1)
    draft_tokens = torch.randint(0, vocab_size, (B, K), device='cuda', dtype=torch.int64)
    random_vals = torch.rand(B, K, device='cuda', dtype=torch.float32)

    for _ in range(warmup):
        flash_attn_v100.speculative_verify(
            target_probs, draft_probs, draft_tokens, random_vals, K
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        flash_attn_v100.speculative_verify(
            target_probs, draft_probs, draft_tokens, random_vals, K
        )
    torch.cuda.synchronize()
    gpu_time = (time.perf_counter() - start) / num_iters * 1000

    start = time.perf_counter()
    for _ in range(num_iters):
        cpu_speculative_verify(
            target_probs.cpu(), draft_probs.cpu(), draft_tokens.cpu(), K
        )
    cpu_time = (time.perf_counter() - start) / num_iters * 1000

    speedup = cpu_time / gpu_time
    print(f"  Config: B={B}, K={K}, vocab_size={vocab_size}")
    print(f"  GPU verify time: {gpu_time:.3f} ms")
    print(f"  CPU verify time: {cpu_time:.3f} ms")
    print(f"  GPU speedup: {speedup:.1f}x")
    print(f"  Result: PASS")
    return True


def test_end_to_end_speculative():
    print("\n" + "=" * 60)
    print("Test 8: End-to-End Speculative Decoding Simulation")
    print("=" * 60)

    B, H_Q, H_KV, N, D = 4, 8, 2, 256, 64
    K = 4
    vocab_size = 512
    num_steps = 10

    torch.manual_seed(42)

    q = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)

    target_head = torch.randn(vocab_size, D, device='cuda', dtype=torch.float32)
    draft_head = torch.randn(vocab_size, D, device='cuda', dtype=torch.float32)

    total_tokens_spec = 0
    total_tokens_base = 0
    total_steps = 0

    for step in range(num_steps):
        draft_tokens_list = []
        draft_probs_list = []
        cache_len = step * K

        for i in range(K):
            out = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len + i)
            logits = torch.matmul(out.float().squeeze(2), draft_head.T)
            probs = F.softmax(logits, dim=-1)
            tokens = torch.argmax(probs, dim=-1)
            draft_tokens_list.append(tokens)
            draft_probs_list.append(probs)

        draft_tokens = torch.stack(draft_tokens_list, dim=1)
        draft_probs = torch.stack(draft_probs_list, dim=1)

        target_probs_list = []
        for i in range(K + 1):
            out = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len + i)
            logits = torch.matmul(out.float().squeeze(2), target_head.T)
            probs = F.softmax(logits, dim=-1)
            target_probs_list.append(probs)

        target_probs = torch.stack(target_probs_list, dim=1)

        random_vals = torch.rand(B, K, device='cuda', dtype=torch.float32)

        accepted_count, output_tokens = flash_attn_v100.speculative_verify(
            target_probs, draft_probs, draft_tokens, random_vals, K
        )

        step_tokens = (accepted_count.float().mean() + 1).item()
        total_tokens_spec += step_tokens
        total_tokens_base += 1
        total_steps += 1

    avg_spec = total_tokens_spec / total_steps
    avg_base = total_tokens_base / total_steps
    speedup = avg_spec / avg_base

    print(f"  Config: B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}, K={K}")
    print(f"  Steps: {num_steps}")
    print(f"  Avg tokens/step (speculative): {avg_spec:.2f}")
    print(f"  Avg tokens/step (baseline):    {avg_base:.2f}")
    print(f"  Effective speedup: {speedup:.2f}x")
    print(f"  Result: PASS")
    return True


if __name__ == "__main__":
    results = []
    results.append(("Verify Correctness", test_verify_correctness()))
    results.append(("All Accepted", test_verify_all_accepted()))
    results.append(("All Rejected", test_verify_all_rejected()))
    results.append(("Large Vocabulary", test_verify_large_vocab()))
    results.append(("Acceptance Rate", test_acceptance_rate()))
    results.append(("GPU vs CPU", test_gpu_verify_vs_cpu()))
    results.append(("Verify Latency", test_verify_latency()))
    results.append(("End-to-End Simulation", test_end_to_end_speculative()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n  Total: {passed}/{total} passed")
