import argparse

import torch

from tensor_parallel import TPDecodeRunner, TensorParallelConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single TP kernel for Nsight Compute profiling")
    parser.add_argument("--mode", choices=["prefill", "decode", "paged_decode"], required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--h-q", type=int, required=True)
    parser.add_argument("--h-kv", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    return parser.parse_args()


def make_runner(args):
    config = TensorParallelConfig(
        H_Q=args.h_q,
        H_KV=args.h_kv,
        D=args.head_dim,
        world_size=args.world_size,
        causal=True,
        auto_grow_kv_cache=True,
        kv_cache_page_block_size=args.block_size,
        kv_cache_layout="blocked",
    )
    return TPDecodeRunner(config)


def run_prefill(args, runner):
    q = torch.randn(args.batch_size, args.h_q, args.seq_len, args.head_dim, device="cuda:0", dtype=torch.float16).contiguous()
    k = torch.randn(args.batch_size, args.h_kv, args.seq_len, args.head_dim, device="cuda:0", dtype=torch.float16).contiguous()
    v = torch.randn(args.batch_size, args.h_kv, args.seq_len, args.head_dim, device="cuda:0", dtype=torch.float16).contiguous()
    runner.prefill(q, k, v, use_workspace=True)


def run_decode(args, runner):
    q = torch.randn(args.batch_size, args.h_q, 1, args.head_dim, device="cuda:0", dtype=torch.float16).contiguous()
    k = torch.randn(args.batch_size, args.h_kv, args.seq_len, args.head_dim, device="cuda:0", dtype=torch.float16).contiguous()
    v = torch.randn(args.batch_size, args.h_kv, args.seq_len, args.head_dim, device="cuda:0", dtype=torch.float16).contiguous()
    runner.decode(q, k, v, cache_len=args.seq_len - 1, use_workspace=True)


def run_paged_decode(args, runner):
    prompt_len = args.seq_len
    q_prompt = torch.randn(args.batch_size, args.h_q, prompt_len, args.head_dim, device="cuda:0", dtype=torch.float16).contiguous()
    k_prompt = torch.randn(args.batch_size, args.h_kv, prompt_len, args.head_dim, device="cuda:0", dtype=torch.float16).contiguous()
    v_prompt = torch.randn(args.batch_size, args.h_kv, prompt_len, args.head_dim, device="cuda:0", dtype=torch.float16).contiguous()
    kv_cache = runner.create_kv_cache(args.batch_size, prompt_len + args.block_size, layout="blocked")
    runner.prefill_to_kv_cache(q_prompt, k_prompt, v_prompt, kv_cache, return_output=False, use_workspace=True)

    q_step = torch.randn(args.batch_size, args.h_q, 1, args.head_dim, device="cuda:0", dtype=torch.float16).contiguous()
    runner.decode_with_paged_kv_cache(q_step, kv_cache, use_workspace=True)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tp_profile_runner")
    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(f"need at least {args.world_size} GPUs")

    torch.manual_seed(0)
    runner = make_runner(args)

    if args.mode == "prefill":
        run_prefill(args, runner)
    elif args.mode == "decode":
        run_decode(args, runner)
    else:
        run_paged_decode(args, runner)

    for device in runner.decoder.devices:
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
