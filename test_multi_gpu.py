import torch

import flash_attn_v100
from benchmark import (
    build_tp_profile_command,
    flash_attn_official_available,
    parse_ncu_csv_metrics,
    profile_tp_kernel,
    run_flash_attn_official,
    run_tp_benchmark,
    run_tp_end_to_end_benchmark,
)
from tensor_parallel import (
    TPDecodeRunner,
    TensorParallelConfig,
    TensorParallelDecoder,
    benchmark_ms,
    cuda_graph_available,
)


TARGET_WORLD_SIZE = 2


def require_n_gpus(target_world_size):
    num_gpus = torch.cuda.device_count()
    if num_gpus < target_world_size:
        print(f"  Skipping: need at least {target_world_size} GPUs")
        return False

    gpu_names = [torch.cuda.get_device_name(i) for i in range(target_world_size)]
    print(f"  Using GPUs: {gpu_names}")
    return True


def require_two_v100():
    return require_n_gpus(TARGET_WORLD_SIZE)


def benchmark_world_size():
    return 3 if torch.cuda.device_count() >= 3 else TARGET_WORLD_SIZE


def test_scatter_gather_heads():
    print("\n" + "=" * 60)
    print("Test 1: Scatter/Gather Heads Correctness")
    print("=" * 60)

    torch.manual_seed(42)
    B, H_Q, N, D = 2, 16, 128, 64
    full_tensor = torch.randn(B, H_Q, N, D, device="cuda:0", dtype=torch.float16)

    for local_head_count in [4, 8]:
        local_tensor = torch.zeros(B, local_head_count, N, D, device="cuda:0", dtype=torch.float16)
        flash_attn_v100.scatter_heads(full_tensor, local_tensor, H_Q, 0, local_head_count)

        expected = full_tensor[:, :local_head_count]
        diff = (local_tensor - expected).abs().max().item()
        assert diff < 1e-6, f"scatter diff too large: {diff}"

        reconstructed = torch.zeros_like(full_tensor)
        flash_attn_v100.gather_heads(reconstructed, local_tensor, H_Q, 0, local_head_count)
        diff2 = (reconstructed[:, :local_head_count] - expected).abs().max().item()
        assert diff2 < 1e-6, f"gather diff too large: {diff2}"

        print(f"  local_head_count={local_head_count}: scatter diff={diff:.8f}, gather diff={diff2:.8f}")

    print("  Result: PASS")
    return True


def test_tensor_parallel_decode_correctness():
    print("\n" + "=" * 60)
    print("Test 2: Tensor Parallel Decode Correctness")
    print("=" * 60)

    if not require_two_v100():
        return True

    torch.manual_seed(42)
    configs = [
        (2, 16, 4, 512, 64),
        (2, 32, 8, 1024, 128),
    ]

    for B, H_Q, H_KV, N, D in configs:
        tp = TensorParallelDecoder(H_Q, H_KV, D, world_size=TARGET_WORLD_SIZE)
        q = torch.randn(B, H_Q, 1, D, device="cuda:0", dtype=torch.float16)
        k = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
        v = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
        cache_len = N - 1

        out_single = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)
        out_tp = tp.decode_step(q, k, v, cache_len, causal=True)
        diff = (out_single.float() - out_tp.float()).abs().max().item()

        print(f"  B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}")
        print(f"    Max diff: {diff:.8f}")
        assert diff < 1e-2, f"TP decode diff too large: {diff}"

    print("  Result: PASS")
    return True


def test_multi_gpu_kv_cache():
    print("\n" + "=" * 60)
    print("Test 3: Tensor Parallel Local KV Cache")
    print("=" * 60)

    if not require_two_v100():
        return True

    torch.manual_seed(42)
    B, H_Q, H_KV, N, D = 2, 16, 4, 512, 64
    tp = TensorParallelDecoder(H_Q, H_KV, D, world_size=TARGET_WORLD_SIZE)

    q = torch.randn(B, H_Q, 1, D, device="cuda:0", dtype=torch.float16)
    k = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
    v = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
    cache_len = N - 1

    local_kv_caches = tp.shard_kv(k, v)
    out_single = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, cache_len)
    out_local = tp.decode_step_local(q, local_kv_caches, cache_len, causal=True)
    diff = (out_single.float() - out_local.float()).abs().max().item()

    print(f"  Each GPU manages {tp.local_H_KV} KV heads")
    print(f"  Max diff vs single GPU: {diff:.8f}")
    assert diff < 1e-2, f"TP local KV diff too large: {diff}"
    print("  Result: PASS")
    return True


def test_tensor_parallel_workspace_reuse():
    print("\n" + "=" * 60)
    print("Test 4: Tensor Parallel Workspace Reuse")
    print("=" * 60)

    if not require_two_v100():
        return True

    torch.manual_seed(42)
    B, H_Q, H_KV, N, D = 2, 16, 4, 512, 64
    tp = TensorParallelDecoder(H_Q, H_KV, D, world_size=TARGET_WORLD_SIZE)
    workspace = tp.create_workspace(B, N, q_seq_len=1, dtype=torch.float16, include_kv=True)

    q = torch.randn(B, H_Q, 1, D, device="cuda:0", dtype=torch.float16)
    k = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
    v = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)

    tp.decode_step(q, k, v, N - 1, causal=True, workspace=workspace)
    first_ptrs = (
        workspace.local_q_shards[0].data_ptr(),
        workspace.local_outs[0].data_ptr(),
        workspace.gathered_out.data_ptr(),
    )

    q2 = torch.randn(B, H_Q, 1, D, device="cuda:0", dtype=torch.float16)
    k2 = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
    v2 = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
    second_out = tp.decode_step(q2, k2, v2, N - 1, causal=True, workspace=workspace)
    second_ptrs = (
        workspace.local_q_shards[0].data_ptr(),
        workspace.local_outs[0].data_ptr(),
        workspace.gathered_out.data_ptr(),
    )

    ref_out = flash_attn_v100.forward_decode_gqa_fp16(q2, k2, v2, True, N - 1)
    diff = (second_out.float() - ref_out.float()).abs().max().item()

    print(f"  Buffer pointers reused: {first_ptrs == second_ptrs}")
    print(f"  Max diff vs single GPU: {diff:.8f}")
    assert first_ptrs == second_ptrs, "workspace buffers should be reused"
    assert diff < 1e-2, f"workspace decode diff too large: {diff}"
    print("  Result: PASS")
    return True


def test_tensor_parallel_workspace_resize():
    print("\n" + "=" * 60)
    print("Test 5: Tensor Parallel Workspace Resize")
    print("=" * 60)

    if not require_two_v100():
        return True

    torch.manual_seed(42)
    H_Q, H_KV, D = 16, 4, 64
    tp = TensorParallelDecoder(H_Q, H_KV, D, world_size=TARGET_WORLD_SIZE)
    workspace = tp.create_workspace(1, 256, q_seq_len=1, dtype=torch.float16, include_kv=True)

    first_ptr = workspace.gathered_out.data_ptr()
    resized = workspace.resize_if_needed(2, 512, q_seq_len=1)
    second_ptr = workspace.gathered_out.data_ptr()
    resized_again = workspace.resize_if_needed(2, 512, q_seq_len=1)
    third_ptr = workspace.gathered_out.data_ptr()

    q = torch.randn(2, H_Q, 1, D, device="cuda:0", dtype=torch.float16)
    k = torch.randn(2, H_KV, 512, D, device="cuda:0", dtype=torch.float16)
    v = torch.randn(2, H_KV, 512, D, device="cuda:0", dtype=torch.float16)
    out = tp.decode_step(q, k, v, 511, causal=True, workspace=workspace)
    ref = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, True, 511)
    diff = (out.float() - ref.float()).abs().max().item()

    print(f"  Resized on larger shape: {resized}")
    print(f"  Pointer changed after resize: {first_ptr != second_ptr}")
    print(f"  Reused after same-shape request: {not resized_again and second_ptr == third_ptr}")
    print(f"  Max diff vs single GPU: {diff:.8f}")
    assert resized, "workspace should resize on larger shape"
    assert first_ptr != second_ptr, "workspace pointer should change after resize"
    assert not resized_again and second_ptr == third_ptr, "workspace should not reallocate for same shape"
    assert diff < 1e-2, f"workspace resize decode diff too large: {diff}"
    print("  Result: PASS")
    return True


def test_tensor_parallel_runner_api():
    print("\n" + "=" * 60)
    print("Test 6: Tensor Parallel Runner API")
    print("=" * 60)

    if not require_two_v100():
        return True

    torch.manual_seed(42)
    B, H_Q, H_KV, N, D = 2, 16, 4, 512, 64
    runner = TPDecodeRunner(TensorParallelConfig(H_Q=H_Q, H_KV=H_KV, D=D, world_size=TARGET_WORLD_SIZE))

    q_decode = torch.randn(B, H_Q, 1, D, device="cuda:0", dtype=torch.float16)
    k_decode = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
    v_decode = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
    out_decode = runner.decode(q_decode, k_decode, v_decode, cache_len=N - 1, use_workspace=True)
    ref_decode = flash_attn_v100.forward_decode_gqa_fp16(q_decode, k_decode, v_decode, True, N - 1)
    decode_diff = (out_decode.float() - ref_decode.float()).abs().max().item()

    q_prefill = torch.randn(B, H_Q, 128, D, device="cuda:0", dtype=torch.float16)
    k_prefill = torch.randn(B, H_KV, 128, D, device="cuda:0", dtype=torch.float16)
    v_prefill = torch.randn(B, H_KV, 128, D, device="cuda:0", dtype=torch.float16)
    out_prefill = runner.prefill(q_prefill, k_prefill, v_prefill, use_workspace=True)
    ref_prefill = flash_attn_v100.forward_prefill_gqa_fp16(q_prefill, k_prefill, v_prefill, True)
    prefill_diff = (out_prefill.float() - ref_prefill.float()).abs().max().item()

    print(f"  Decode max diff: {decode_diff:.8f}")
    print(f"  Prefill max diff: {prefill_diff:.8f}")
    print(f"  Decode workspace allocated: {runner.workspace is not None}")
    print(f"  Prefill workspace allocated: {runner.prefill_workspace is not None}")
    assert decode_diff < 1e-2, f"runner decode diff too large: {decode_diff}"
    assert prefill_diff < 1e-2, f"runner prefill diff too large: {prefill_diff}"
    assert runner.workspace is not None and runner.prefill_workspace is not None, "runner should own workspaces"
    print("  Result: PASS")
    return True


def test_tensor_parallel_kv_cache_lifecycle():
    print("\n" + "=" * 60)
    print("Test 7: Tensor Parallel KV Cache Lifecycle")
    print("=" * 60)

    if not require_two_v100():
        return True

    torch.manual_seed(42)
    B, H_Q, H_KV, D = 2, 16, 4, 64
    prompt_len = 64
    max_seq_len = 96
    runner = TPDecodeRunner(TensorParallelConfig(H_Q=H_Q, H_KV=H_KV, D=D, world_size=TARGET_WORLD_SIZE))
    kv_cache = runner.create_kv_cache(B, max_seq_len)

    q_prompt = torch.randn(B, H_Q, prompt_len, D, device="cuda:0", dtype=torch.float16)
    k_prompt = torch.randn(B, H_KV, prompt_len, D, device="cuda:0", dtype=torch.float16)
    v_prompt = torch.randn(B, H_KV, prompt_len, D, device="cuda:0", dtype=torch.float16)
    out_prefill = runner.prefill_to_kv_cache(q_prompt, k_prompt, v_prompt, kv_cache, return_output=True)
    ref_prefill = flash_attn_v100.forward_prefill_gqa_fp16(q_prompt, k_prompt, v_prompt, True)
    prefill_diff = (out_prefill.float() - ref_prefill.float()).abs().max().item()

    history_k = k_prompt.clone()
    history_v = v_prompt.clone()
    decode_diffs = []
    for _ in range(3):
        q_step = torch.randn(B, H_Q, 1, D, device="cuda:0", dtype=torch.float16)
        k_step = torch.randn(B, H_KV, 1, D, device="cuda:0", dtype=torch.float16)
        v_step = torch.randn(B, H_KV, 1, D, device="cuda:0", dtype=torch.float16)

        out_step = runner.decode_with_kv_cache(q_step, kv_cache, new_k=k_step, new_v=v_step)
        history_k = torch.cat([history_k, k_step], dim=2)
        history_v = torch.cat([history_v, v_step], dim=2)
        ref_step = flash_attn_v100.forward_decode_gqa_fp16(q_step, history_k, history_v, True, history_k.size(2) - 1)
        decode_diffs.append((out_step.float() - ref_step.float()).abs().max().item())

    runner.reset_kv_cache(kv_cache)

    print(f"  Prefill diff: {prefill_diff:.8f}")
    print(f"  Decode diffs: {[round(x, 8) for x in decode_diffs]}")
    print(f"  Cache length after reset: {kv_cache.current_len}")
    assert prefill_diff < 1e-2, f"kv cache prefill diff too large: {prefill_diff}"
    assert max(decode_diffs) < 1e-2, f"kv cache decode diff too large: {max(decode_diffs)}"
    assert kv_cache.current_len == 0, "kv cache reset should set current_len to 0"
    print("  Result: PASS")
    return True


def test_tensor_parallel_kv_cache_auto_grow_and_stats():
    print("\n" + "=" * 60)
    print("Test 8: TP KV Cache Auto Grow And Stats")
    print("=" * 60)

    if not require_two_v100():
        return True

    torch.manual_seed(42)
    B, H_Q, H_KV, D = 1, 16, 4, 64
    runner = TPDecodeRunner(
        TensorParallelConfig(
            H_Q=H_Q,
            H_KV=H_KV,
            D=D,
            world_size=TARGET_WORLD_SIZE,
            auto_grow_kv_cache=True,
            kv_cache_growth_factor=2.0,
            kv_cache_page_block_size=16,
        )
    )
    kv_cache = runner.create_kv_cache(B, 8)

    q_prompt = torch.randn(B, H_Q, 6, D, device="cuda:0", dtype=torch.float16)
    k_prompt = torch.randn(B, H_KV, 6, D, device="cuda:0", dtype=torch.float16)
    v_prompt = torch.randn(B, H_KV, 6, D, device="cuda:0", dtype=torch.float16)
    runner.prefill_to_kv_cache(q_prompt, k_prompt, v_prompt, kv_cache, return_output=False)

    q_step = torch.randn(B, H_Q, 1, D, device="cuda:0", dtype=torch.float16)
    k_step = torch.randn(B, H_KV, 4, D, device="cuda:0", dtype=torch.float16)
    v_step = torch.randn(B, H_KV, 4, D, device="cuda:0", dtype=torch.float16)
    runner.decode_with_kv_cache(q_step, kv_cache, new_k=k_step, new_v=v_step)

    metadata = runner.get_kv_cache_metadata(kv_cache)
    stats = runner.get_runtime_stats()

    print(f"  Cache metadata: {metadata}")
    print(
        f"  Stats: append_calls={stats['kv_cache_append_calls']}, grow_count={stats['kv_cache_grow_count']}, "
        f"decode_local_calls={stats['decode_local_calls']}, last_op={stats['last_operation']}"
    )
    assert metadata["capacity"] >= 16, f"expected grown capacity >= 16, got {metadata['capacity']}"
    assert metadata["grow_count"] >= 1, "kv cache should grow at least once"
    assert stats["kv_cache_append_calls"] >= 2, "expected prompt append + decode append"
    assert stats["kv_cache_grow_count"] >= 1, "runner stats should record kv cache growth"
    assert stats["decode_local_calls"] >= 1, "decode_local should be used by kv cache path"
    assert stats["last_operation"] == "decode_local", "last operation should be decode_local after decode_with_kv_cache"
    print("  Result: PASS")
    return True


def test_tensor_parallel_block_metadata():
    print("\n" + "=" * 60)
    print("Test 9: TP KV Cache Block Metadata")
    print("=" * 60)

    if not require_two_v100():
        return True

    runner = TPDecodeRunner(
        TensorParallelConfig(
            H_Q=16,
            H_KV=4,
            D=64,
            world_size=TARGET_WORLD_SIZE,
            kv_cache_layout="blocked",
            kv_cache_page_block_size=16,
        )
    )
    kv_cache = runner.create_kv_cache(batch_size=2, max_seq_len=48)
    meta = runner.get_kv_cache_metadata(kv_cache)
    block_meta = runner.get_kv_cache_block_metadata(kv_cache)

    q = torch.randn(2, 16, 20, 64, device="cuda:0", dtype=torch.float16)
    k = torch.randn(2, 4, 20, 64, device="cuda:0", dtype=torch.float16)
    v = torch.randn(2, 4, 20, 64, device="cuda:0", dtype=torch.float16)
    runner.prefill_to_kv_cache(q, k, v, kv_cache, return_output=False)
    block_meta_after = runner.get_kv_cache_block_metadata(kv_cache)

    print(f"  Initial metadata: {meta}")
    print(f"  Block metadata after prefill: {block_meta_after}")
    assert meta["layout"] == "blocked", "kv cache should use blocked layout"
    assert block_meta["tokens_per_block"] == 16, "unexpected tokens_per_block"
    assert block_meta_after["allocated_block_count"] == 2, "20 tokens should allocate 2 blocks"
    assert len(block_meta_after["block_table"]) == 2, "batch block table size mismatch"
    assert all(len(row) == 2 for row in block_meta_after["block_table"]), "each batch entry should map 2 blocks"
    print("  Result: PASS")
    return True


def test_tensor_parallel_paged_decode_correctness():
    print("\n" + "=" * 60)
    print("Test 10: TP Paged Decode Correctness")
    print("=" * 60)

    if not require_two_v100():
        return True

    torch.manual_seed(42)
    B, H_Q, H_KV, prompt_len, D = 2, 16, 4, 32, 64
    runner = TPDecodeRunner(
        TensorParallelConfig(
            H_Q=H_Q,
            H_KV=H_KV,
            D=D,
            world_size=TARGET_WORLD_SIZE,
            kv_cache_layout="blocked",
            kv_cache_page_block_size=16,
        )
    )
    kv_cache = runner.create_kv_cache(B, 64, layout="blocked")

    q_prompt = torch.randn(B, H_Q, prompt_len, D, device="cuda:0", dtype=torch.float16)
    k_prompt = torch.randn(B, H_KV, prompt_len, D, device="cuda:0", dtype=torch.float16)
    v_prompt = torch.randn(B, H_KV, prompt_len, D, device="cuda:0", dtype=torch.float16)
    runner.prefill_to_kv_cache(q_prompt, k_prompt, v_prompt, kv_cache, return_output=False)

    q_step = torch.randn(B, H_Q, 1, D, device="cuda:0", dtype=torch.float16)
    out_paged = runner.decode_with_paged_kv_cache(q_step, kv_cache, use_workspace=True)
    out_ref = flash_attn_v100.forward_decode_gqa_fp16(q_step, k_prompt, v_prompt, True, prompt_len - 1)
    diff = (out_paged.float() - out_ref.float()).abs().max().item()
    stats = runner.get_runtime_stats()
    block_meta = runner.get_kv_cache_block_metadata(kv_cache)

    print(f"  Max diff vs single GPU decode: {diff:.8f}")
    print(f"  Runtime stats last op: {stats['last_operation']}")
    print(f"  Block metadata: {block_meta}")
    assert diff < 1e-2, f"TP paged decode diff too large: {diff}"
    assert stats["decode_paged_calls"] >= 1, "decode_paged_calls should be tracked"
    assert stats["last_operation"] == "decode_paged", "last operation should be decode_paged"
    assert block_meta["allocated_block_count"] == 2, "32 tokens with block size 16 should use 2 blocks"
    print("  Result: PASS")
    return True


def test_tensor_parallel_paged_kv_storage_reuse():
    print("\n" + "=" * 60)
    print("Test 10B: TP Paged KV Storage Reuse")
    print("=" * 60)

    if not require_two_v100():
        return True

    torch.manual_seed(42)
    B, H_Q, H_KV, D = 2, 16, 4, 64
    runner = TPDecodeRunner(
        TensorParallelConfig(
            H_Q=H_Q,
            H_KV=H_KV,
            D=D,
            world_size=TARGET_WORLD_SIZE,
            kv_cache_layout="blocked",
            kv_cache_page_block_size=16,
        )
    )
    kv_cache = runner.create_kv_cache(B, 48, layout="blocked")

    q_prompt = torch.randn(B, H_Q, 20, D, device="cuda:0", dtype=torch.float16)
    k_prompt = torch.randn(B, H_KV, 20, D, device="cuda:0", dtype=torch.float16)
    v_prompt = torch.randn(B, H_KV, 20, D, device="cuda:0", dtype=torch.float16)
    runner.prefill_to_kv_cache(q_prompt, k_prompt, v_prompt, kv_cache, return_output=False)

    first_paged = kv_cache.build_paged_cache_tensors(compiled_block_size=16)
    first_ptrs = [(k_pages.data_ptr(), v_pages.data_ptr()) for k_pages, v_pages, _, _, _ in first_paged]
    first_tables = [block_table.clone() for _, _, block_table, _, _ in first_paged]

    q_step = torch.randn(B, H_Q, 1, D, device="cuda:0", dtype=torch.float16)
    k_step = torch.randn(B, H_KV, 16, D, device="cuda:0", dtype=torch.float16)
    v_step = torch.randn(B, H_KV, 16, D, device="cuda:0", dtype=torch.float16)
    runner.append_to_kv_cache(kv_cache, k_step, v_step)

    second_paged = kv_cache.build_paged_cache_tensors(compiled_block_size=16)
    second_ptrs = [(k_pages.data_ptr(), v_pages.data_ptr()) for k_pages, v_pages, _, _, _ in second_paged]
    second_tables = [block_table.clone() for _, _, block_table, _, _ in second_paged]

    active_kv = kv_cache.get_active_kv()
    out_local = runner.decode_local(q_step, active_kv, kv_cache.current_len - 1, use_workspace=True)
    out_paged = runner.decode_with_paged_kv_cache(q_step, kv_cache, use_workspace=True)
    diff = (out_local.float() - out_paged.float()).abs().max().item()

    print(f"  First page pointers: {first_ptrs}")
    print(f"  Second page pointers: {second_ptrs}")
    print(f"  Block table before append: {[table.tolist() for table in first_tables]}")
    print(f"  Block table after append: {[table.tolist() for table in second_tables]}")
    print(f"  Max diff vs reconstructed contiguous KV: {diff:.8f}")
    assert first_ptrs == second_ptrs, "paged KV storage should be persistent across decode steps"
    assert any(
        not torch.equal(before, after) for before, after in zip(first_tables, second_tables)
    ), "block table should update in-place as new pages are allocated"
    assert diff < 1e-2, f"paged KV storage decode diff too large: {diff}"
    print("  Result: PASS")
    return True


def test_tensor_parallel_benchmark_smoke():
    print("\n" + "=" * 60)
    print("Test 11: TP Benchmark Smoke")
    print("=" * 60)

    world_size = benchmark_world_size()
    if not require_n_gpus(world_size):
        return True

    configs = [(1, 24, 6, 256, 64, 16)] if world_size == 3 else [(1, 16, 4, 256, 64, 16)]
    results = run_tp_benchmark(
        configs=configs,
        world_size=world_size,
        prefill_warmup=1,
        prefill_iterations=2,
        decode_warmup=1,
        decode_iterations=3,
    )
    assert len(results) == 1, "expected one TP benchmark result"
    row = results[0]
    print(f"  Benchmark row: {row}")
    assert row["TP"] == world_size, "TP benchmark should record world size"
    assert row["KV layout"] == "blocked", "TP benchmark should use blocked layout"
    assert "Sweep tag" in row, "TP benchmark should include sweep tag"
    assert float(row["Prefill lat/token (us)"]) > 0.0, "prefill latency/token should be positive"
    assert float(row["Decode lat/token (us)"]) > 0.0, "decode latency/token should be positive"
    assert float(row["Prefill TFLOPS"]) > 0.0, "prefill TFLOPS should be positive"
    assert "Prefill occ (%)" in row, "TP benchmark should include prefill occupancy"
    assert "Decode occ (%)" in row, "TP benchmark should include decode occupancy"
    assert "Official prefill (ms)" in row, "TP benchmark should include official prefill baseline"
    assert "Official decode (ms)" in row, "TP benchmark should include official decode baseline"
    assert "Official baseline" in row, "TP benchmark should include official baseline status"
    assert "Occ profiler" in row, "TP benchmark should include profiler status"
    assert float(row["Decode diff"]) < 1e-2, "decode diff too large in TP benchmark"
    assert float(row["Paged diff"]) < 1e-2, "paged decode diff too large in TP benchmark"
    assert float(row["Prefill diff"]) < 1e-2, "prefill diff too large in TP benchmark"
    print("  Result: PASS")
    return True


def test_tensor_parallel_end_to_end_benchmark_smoke():
    print("\n" + "=" * 60)
    print("Test 12: TP End-to-End Benchmark Smoke")
    print("=" * 60)

    world_size = benchmark_world_size()
    if not require_n_gpus(world_size):
        return True

    configs = [
        (1, 24, 6, 64, 64, 8, 16),
        (1, 24, 6, 64, 64, 4, 32),
    ] if world_size == 3 else [
        (1, 16, 4, 64, 64, 8, 16),
        (1, 16, 4, 64, 64, 4, 32),
    ]
    results = run_tp_end_to_end_benchmark(
        configs=configs,
        world_size=world_size,
        prefill_warmup=1,
        prefill_iterations=2,
        decode_warmup=0,
        decode_iterations=1,
    )
    assert len(results) == 2, "expected two TP end-to-end benchmark results"
    paged_row, contiguous_row = results
    print(f"  Paged row: {paged_row}")
    print(f"  Contiguous row: {contiguous_row}")
    assert paged_row["TP"] == world_size, "paged row should record benchmark world size"
    assert contiguous_row["TP"] == world_size, "contiguous row should record benchmark world size"
    assert paged_row["Decode path"] == "paged", "block_size=16 should use paged path"
    assert contiguous_row["Decode path"] == "contiguous", "non-16 block_size should use contiguous path"
    for row in results:
        assert float(row["TTFT total (ms)"]) > 0.0, "TTFT total should be positive"
        assert float(row["ITL (ms)"]) > 0.0, "ITL should be positive"
        assert float(row["tok/s"]) > 0.0, "tok/s should be positive"
        assert "GPU util (%)" in row, "end-to-end benchmark should include GPU utilization"
        assert "NVLink est. (GB/s)" in row, "end-to-end benchmark should include NVLink estimate"
    print("  Result: PASS")
    return True


def test_tensor_parallel_world_size_correctness():
    print("\n" + "=" * 60)
    print("Test 13: TP World Size Correctness")
    print("=" * 60)

    torch.manual_seed(42)
    configs = [
        (1, 2, 16, 4, 256, 64),
        (2, 2, 16, 4, 512, 64),
        (3, 1, 24, 6, 256, 64),
        (3, 2, 24, 6, 512, 64),
    ]

    for world_size, B, H_Q, H_KV, N, D in configs:
        if torch.cuda.device_count() < world_size:
            print(f"  Skip TP={world_size}: need at least {world_size} GPUs")
            continue

        runner = TPDecodeRunner(
            TensorParallelConfig(H_Q=H_Q, H_KV=H_KV, D=D, world_size=world_size, kv_cache_page_block_size=16)
        )
        q_decode = torch.randn(B, H_Q, 1, D, device="cuda:0", dtype=torch.float16)
        k_decode = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
        v_decode = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
        out_decode = runner.decode(q_decode, k_decode, v_decode, cache_len=N - 1, use_workspace=True)
        ref_decode = flash_attn_v100.forward_decode_gqa_fp16(q_decode, k_decode, v_decode, True, N - 1)
        decode_diff = (out_decode.float() - ref_decode.float()).abs().max().item()

        q_prefill = torch.randn(B, H_Q, 64, D, device="cuda:0", dtype=torch.float16)
        k_prefill = torch.randn(B, H_KV, 64, D, device="cuda:0", dtype=torch.float16)
        v_prefill = torch.randn(B, H_KV, 64, D, device="cuda:0", dtype=torch.float16)
        out_prefill = runner.prefill(q_prefill, k_prefill, v_prefill, use_workspace=True)
        ref_prefill = flash_attn_v100.forward_prefill_gqa_fp16(q_prefill, k_prefill, v_prefill, True)
        prefill_diff = (out_prefill.float() - ref_prefill.float()).abs().max().item()

        print(f"  TP={world_size} | decode_diff={decode_diff:.8f}, prefill_diff={prefill_diff:.8f}")
        assert decode_diff < 1e-2, f"TP={world_size} decode diff too large: {decode_diff}"
        assert prefill_diff < 1e-2, f"TP={world_size} prefill diff too large: {prefill_diff}"

    print("  Result: PASS")
    return True


def test_tensor_parallel_prefill_correctness():
    print("\n" + "=" * 60)
    print("Test 14: Tensor Parallel Prefill Correctness")
    print("=" * 60)

    if not require_two_v100():
        return True

    torch.manual_seed(42)
    B, H_Q, H_KV, N, D = 1, 32, 8, 1024, 128
    tp = TensorParallelDecoder(H_Q, H_KV, D, world_size=TARGET_WORLD_SIZE)

    q = torch.randn(B, H_Q, N, D, device="cuda:0", dtype=torch.float16)
    k = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
    v = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)

    out_single = flash_attn_v100.forward_prefill_gqa_fp16(q, k, v, True)
    out_tp = tp.prefill_step(q, k, v, causal=True)
    diff = (out_single.float() - out_tp.float()).abs().max().item()

    print(f"  Max diff: {diff:.8f}")
    assert diff < 1e-2, f"TP prefill diff too large: {diff}"
    print("  Result: PASS")
    return True


def test_memory_savings():
    print("\n" + "=" * 60)
    print("Test 15: 2xV100 KV Cache Memory Savings")
    print("=" * 60)

    if not require_two_v100():
        return True

    B, H_Q, H_KV, N, D = 4, 32, 8, 4096, 128
    tp = TensorParallelDecoder(H_Q, H_KV, D, world_size=TARGET_WORLD_SIZE)

    single_gpu_kv_bytes = B * H_KV * N * D * 2 * 2
    per_gpu_kv_bytes = tp.kv_cache_bytes_per_gpu(B, N)
    reduction = 1.0 - (per_gpu_kv_bytes / single_gpu_kv_bytes)

    print(f"  Config: B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}")
    print(f"  Single GPU KV cache: {single_gpu_kv_bytes / 1024 / 1024:.1f} MB")
    print(f"  Per-GPU KV cache (2-way TP): {per_gpu_kv_bytes / 1024 / 1024:.1f} MB")
    print(f"  Per-GPU memory reduction: {reduction * 100:.1f}%")

    assert abs(reduction - 0.5) < 1e-6, f"expected 50% KV reduction, got {reduction * 100:.2f}%"
    print("  Result: PASS")
    return True


def test_tensor_parallel_prefill_scaling():
    print("\n" + "=" * 60)
    print("Test 16: Tensor Parallel Prefill Scaling")
    print("=" * 60)

    if not require_two_v100():
        return True

    torch.manual_seed(42)
    configs = [
        (1, 32, 8, 1024, 128),
        (1, 32, 8, 2048, 128),
        (2, 32, 8, 2048, 128),
    ]

    best_efficiency = 0.0
    for B, H_Q, H_KV, N, D in configs:
        tp = TensorParallelDecoder(H_Q, H_KV, D, world_size=TARGET_WORLD_SIZE)
        q = torch.randn(B, H_Q, N, D, device="cuda:0", dtype=torch.float16)
        k = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)
        v = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16)

        workspace = tp.create_workspace(B, N, q_seq_len=N, dtype=torch.float16, include_kv=True)
        local_q_shards, local_kv_shards = tp.shard_qkv(q, k, v, workspace=workspace)

        single_time = benchmark_ms(
            lambda: flash_attn_v100.forward_prefill_gqa_fp16(q, k, v, True),
            [torch.device("cuda:0")],
            warmup=5,
            iterations=20,
        )
        tp_compute_time = benchmark_ms(
            lambda: tp.prefill_step_sharded(local_q_shards, local_kv_shards, causal=True, gather_output=False, workspace=workspace),
            tp.devices,
            warmup=5,
            iterations=20,
        )
        tp_end_to_end_time = benchmark_ms(
            lambda: tp.prefill_step_sharded(local_q_shards, local_kv_shards, causal=True, gather_output=True, workspace=workspace),
            tp.devices,
            warmup=5,
            iterations=20,
        )

        compute_speedup = single_time / tp_compute_time
        end_to_end_speedup = single_time / tp_end_to_end_time
        scaling_efficiency = compute_speedup / TARGET_WORLD_SIZE
        best_efficiency = max(best_efficiency, scaling_efficiency)

        print(f"  B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}")
        print(f"    Single GPU: {single_time:.3f} ms")
        print(f"    TP compute-only: {tp_compute_time:.3f} ms | speedup={compute_speedup:.2f}x | efficiency={scaling_efficiency:.2f}x")
        print(f"    TP end-to-end: {tp_end_to_end_time:.3f} ms | speedup={end_to_end_speedup:.2f}x")

    print(f"  Best 2-GPU compute scaling efficiency: {best_efficiency:.2f}x")
    assert best_efficiency >= 0.85, f"expected prefill scaling efficiency near 0.85x, got {best_efficiency:.2f}x"
    print("  Result: PASS")
    return True


def test_tp_profiler_api_smoke():
    print("\n" + "=" * 60)
    print("Test 17: TP Profiler API Smoke")
    print("=" * 60)

    config = (1, 16, 4, 256, 64, 16)
    command = build_tp_profile_command(config, mode="decode", world_size=2)
    assert "ncu" in command[0], "profiler command should launch ncu"
    assert "--metrics" in command, "profiler command should include metrics"

    sample_csv = 'ID,Kernel Name,Metric Name,Metric Unit,Metric Value\n1,kernel,"sm__warps_active.avg.pct_of_peak_sustained_active",%,62.5\n'
    metrics = parse_ncu_csv_metrics(sample_csv, ["sm__warps_active.avg.pct_of_peak_sustained_active"])
    assert abs(metrics["sm__warps_active.avg.pct_of_peak_sustained_active"] - 62.5) < 1e-6, "failed to parse occupancy metric"

    profile = profile_tp_kernel(config, mode="decode", world_size=2, timeout_s=1)
    print(f"  Profile result: {profile}")
    assert profile["status"] in {"ok", "cuda_unavailable", "insufficient_gpus", "ncu_unavailable", "timeout", "launch_error"} or profile["status"].startswith("ncu_error_"), "unexpected profiler status"
    print("  Result: PASS")
    return True


def test_flash_attn_official_baseline_api_smoke():
    print("\n" + "=" * 60)
    print("Test 18: FlashAttention Official Baseline API Smoke")
    print("=" * 60)

    available = flash_attn_official_available()
    print(f"  flash_attn official available: {available}")
    assert isinstance(available, bool), "availability flag should be bool"

    if not available:
        try:
            q = torch.randn(1, 4, 16, 64, dtype=torch.float16)
            run_flash_attn_official(q, q, q, causal=True)
            raise AssertionError("expected missing flash_attn official to raise")
        except RuntimeError as exc:
            assert "not installed" in str(exc), "missing official baseline should raise a clear error"
    print("  Result: PASS")
    return True


def test_tp_cuda_graph_api_smoke():
    print("\n" + "=" * 60)
    print("Test 19: TP CUDA Graph API Smoke")
    print("=" * 60)

    available = cuda_graph_available()
    print(f"  CUDA Graph available: {available}")
    assert isinstance(available, bool), "CUDA Graph availability flag should be bool"
    print("  Result: PASS")
    return True


def test_tensor_parallel_paged_decode_cuda_graph():
    print("\n" + "=" * 60)
    print("Test 20: TP Paged Decode CUDA Graph")
    print("=" * 60)

    if not require_two_v100():
        return True
    if not cuda_graph_available():
        print("  Skipping: CUDA Graph is not available")
        return True

    torch.manual_seed(42)
    B, H_Q, H_KV, D = 2, 16, 4, 64
    config = TensorParallelConfig(
        H_Q=H_Q,
        H_KV=H_KV,
        D=D,
        world_size=TARGET_WORLD_SIZE,
        kv_cache_layout="blocked",
        kv_cache_page_block_size=16,
    )
    runner = TPDecodeRunner(config)
    kv_cache_eager = runner.create_kv_cache(B, 64, layout="blocked")
    kv_cache_graph = runner.create_kv_cache(B, 64, layout="blocked")

    q_prompt = torch.randn(B, H_Q, 24, D, device="cuda:0", dtype=torch.float16)
    k_prompt = torch.randn(B, H_KV, 24, D, device="cuda:0", dtype=torch.float16)
    v_prompt = torch.randn(B, H_KV, 24, D, device="cuda:0", dtype=torch.float16)
    runner.prefill_to_kv_cache(q_prompt, k_prompt, v_prompt, kv_cache_eager, return_output=False)
    runner.prefill_to_kv_cache(q_prompt, k_prompt, v_prompt, kv_cache_graph, return_output=False)

    bucket = runner.create_paged_decode_cuda_graph_bucket(kv_cache_graph)
    q_step = torch.randn(B, H_Q, 1, D, device="cuda:0", dtype=torch.float16)
    out_eager = runner.decode_with_paged_kv_cache(q_step, kv_cache_eager, use_workspace=True)
    out_graph = runner.decode_with_paged_kv_cache_graph(q_step, kv_cache_graph, bucket)
    diff = (out_eager.float() - out_graph.float()).abs().max().item()
    print(f"  Max diff eager vs graph: {diff:.8f}")
    assert diff < 1e-2, f"graph decode diff too large: {diff}"

    k_next = torch.randn(B, H_KV, 1, D, device="cuda:0", dtype=torch.float16)
    v_next = torch.randn(B, H_KV, 1, D, device="cuda:0", dtype=torch.float16)
    q_next = torch.randn(B, H_Q, 1, D, device="cuda:0", dtype=torch.float16)
    out_eager_next = runner.decode_with_paged_kv_cache(q_next, kv_cache_eager, new_k=k_next, new_v=v_next, use_workspace=True)
    out_graph_next = runner.decode_with_paged_kv_cache_graph(q_next, kv_cache_graph, bucket, new_k=k_next, new_v=v_next)
    diff_next = (out_eager_next.float() - out_graph_next.float()).abs().max().item()
    stats = runner.get_runtime_stats()
    print(f"  Max diff eager vs graph after append: {diff_next:.8f}")
    print(f"  Runtime stats: graph_replay_calls={stats['decode_graph_replay_calls']}, graph_captures={stats['decode_graph_capture_count']}")
    assert diff_next < 1e-2, f"graph decode diff after append too large: {diff_next}"
    assert stats["decode_graph_replay_calls"] >= 2, "graph replay count should be tracked"
    assert stats["decode_graph_capture_count"] >= 1, "graph capture count should be tracked"
    print("  Result: PASS")
    return True


if __name__ == "__main__":
    results = []
    results.append(("Scatter/Gather Heads", test_scatter_gather_heads()))
    results.append(("TP Decode Correctness", test_tensor_parallel_decode_correctness()))
    results.append(("TP Local KV Cache", test_multi_gpu_kv_cache()))
    results.append(("TP Workspace Reuse", test_tensor_parallel_workspace_reuse()))
    results.append(("TP Workspace Resize", test_tensor_parallel_workspace_resize()))
    results.append(("TP Runner API", test_tensor_parallel_runner_api()))
    results.append(("TP KV Cache Lifecycle", test_tensor_parallel_kv_cache_lifecycle()))
    results.append(("TP KV Cache Auto Grow/Stats", test_tensor_parallel_kv_cache_auto_grow_and_stats()))
    results.append(("TP KV Block Metadata", test_tensor_parallel_block_metadata()))
    results.append(("TP Paged Decode", test_tensor_parallel_paged_decode_correctness()))
    results.append(("TP Paged KV Storage Reuse", test_tensor_parallel_paged_kv_storage_reuse()))
    results.append(("TP Benchmark Smoke", test_tensor_parallel_benchmark_smoke()))
    results.append(("TP End-to-End Benchmark Smoke", test_tensor_parallel_end_to_end_benchmark_smoke()))
    results.append(("TP World Size Correctness", test_tensor_parallel_world_size_correctness()))
    results.append(("TP Prefill Correctness", test_tensor_parallel_prefill_correctness()))
    results.append(("2xV100 KV Memory Savings", test_memory_savings()))
    results.append(("TP Prefill Scaling", test_tensor_parallel_prefill_scaling()))
    results.append(("TP Profiler API Smoke", test_tp_profiler_api_smoke()))
    results.append(("FlashAttention Official Baseline API Smoke", test_flash_attn_official_baseline_api_smoke()))
    results.append(("TP CUDA Graph API Smoke", test_tp_cuda_graph_api_smoke()))
    results.append(("TP Paged Decode CUDA Graph", test_tensor_parallel_paged_decode_cuda_graph()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n  Total: {passed}/{total} passed")
