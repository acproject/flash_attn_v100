import time
from dataclasses import asdict, dataclass

import torch

import flash_attn_v100


DEFAULT_WORLD_SIZE = 2


@dataclass
class TensorParallelConfig:
    H_Q: int
    H_KV: int
    D: int
    world_size: int = DEFAULT_WORLD_SIZE
    root_rank: int = 0
    dtype: torch.dtype = torch.float16
    causal: bool = True
    auto_grow_kv_cache: bool = True
    kv_cache_growth_factor: float = 2.0
    kv_cache_page_block_size: int = 256
    kv_cache_layout: str = "contiguous"


@dataclass
class TPRuntimeStats:
    decode_calls: int = 0
    decode_local_calls: int = 0
    decode_paged_calls: int = 0
    prefill_calls: int = 0
    kv_cache_append_calls: int = 0
    kv_cache_reset_calls: int = 0
    kv_cache_grow_count: int = 0
    decode_workspace_resize_count: int = 0
    prefill_workspace_resize_count: int = 0
    last_cache_len: int = 0
    last_kv_cache_capacity: int = 0
    last_decode_ms: float = 0.0
    last_paged_decode_ms: float = 0.0
    last_prefill_ms: float = 0.0
    last_kv_append_ms: float = 0.0
    decode_graph_replay_calls: int = 0
    decode_graph_capture_count: int = 0
    last_graph_bucket_seq_len: int = 0
    last_operation: str = "init"


def cuda_graph_available():
    return (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "CUDAGraph")
        and hasattr(torch.cuda, "graph")
    )


def available_devices(world_size=None):
    count = torch.cuda.device_count() if world_size is None else world_size
    return [torch.device(f"cuda:{rank}") for rank in range(count)]


def synchronize_devices(devices):
    for device in devices:
        torch.cuda.synchronize(device)


def benchmark_ms(fn, devices, warmup=10, iterations=50):
    for _ in range(warmup):
        fn()
    synchronize_devices(devices)

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    synchronize_devices(devices)
    return (time.perf_counter() - start) / iterations * 1000.0


class TensorParallelWorkspace:
    def __init__(self, decoder, B, kv_seq_len, q_seq_len=1, dtype=torch.float16, include_kv=True):
        self.decoder = decoder
        self.include_kv = include_kv
        self.resize(B, kv_seq_len, q_seq_len=q_seq_len, dtype=dtype)

    def resize(self, B, kv_seq_len, q_seq_len=1, dtype=None):
        self.B = B
        self.kv_seq_len = kv_seq_len
        self.q_seq_len = q_seq_len
        self.dtype = self.dtype if dtype is None and hasattr(self, "dtype") else (dtype or torch.float16)

        self.local_q_shards = [
            torch.empty(B, self.decoder.local_H_Q, q_seq_len, self.decoder.D, device=device, dtype=self.dtype)
            for device in self.decoder.devices
        ]
        self.local_outs = [
            torch.empty(B, self.decoder.local_H_Q, q_seq_len, self.decoder.D, device=device, dtype=self.dtype)
            for device in self.decoder.devices
        ]
        self.gathered_out = torch.empty(
            B, self.decoder.H_Q, q_seq_len, self.decoder.D, device=self.decoder.root_device, dtype=self.dtype
        )
        self.local_kv_shards = None
        if self.include_kv:
            self.local_kv_shards = [
                (
                    torch.empty(B, self.decoder.local_H_KV, kv_seq_len, self.decoder.D, device=device, dtype=self.dtype),
                    torch.empty(B, self.decoder.local_H_KV, kv_seq_len, self.decoder.D, device=device, dtype=self.dtype),
                )
                for device in self.decoder.devices
            ]

    def resize_if_needed(self, B, kv_seq_len, q_seq_len=1, dtype=None):
        requested_dtype = self.dtype if dtype is None else dtype
        if (
            self.B == B
            and self.kv_seq_len == kv_seq_len
            and self.q_seq_len == q_seq_len
            and self.dtype == requested_dtype
        ):
            return False
        self.resize(B, kv_seq_len, q_seq_len=q_seq_len, dtype=requested_dtype)
        return True


@dataclass
class TPCUDAGraphDecodeBucket:
    kv_cache: "TensorParallelKVCache"
    batch_size: int
    max_seq_len: int
    dtype: torch.dtype
    workspace: TensorParallelWorkspace
    graphs: list
    local_outputs: list
    storage_ptrs: list
    metadata_ptrs: list
    layout: str = "blocked"


class TensorParallelKVCache:
    def __init__(
        self,
        decoder,
        batch_size,
        max_seq_len,
        dtype=torch.float16,
        auto_grow=True,
        growth_factor=2.0,
        page_block_size=256,
        layout="contiguous",
    ):
        self.decoder = decoder
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.auto_grow = auto_grow
        self.growth_factor = growth_factor
        self.page_block_size = page_block_size
        self.layout = layout
        self.current_len = 0
        self.grow_count = 0
        self.tokens_per_block = page_block_size if page_block_size > 0 else max_seq_len
        self.num_blocks = self._capacity_to_blocks(max_seq_len)
        self.total_physical_blocks = batch_size * self.num_blocks if layout == "blocked" else self.num_blocks
        if layout == "blocked":
            self.local_kv = decoder.allocate_blocked_local_kv_cache(
                self.total_physical_blocks,
                self.tokens_per_block,
                dtype=dtype,
            )
            self.block_table_tensors = [
                torch.full((batch_size, self.num_blocks), -1, device=device, dtype=torch.int32)
                for device in decoder.devices
            ]
            self.seq_lens_tensors = [
                torch.zeros((batch_size,), device=device, dtype=torch.int32)
                for device in decoder.devices
            ]
            self.free_block_ids = list(range(self.total_physical_blocks))
        else:
            self.local_kv = decoder.allocate_local_kv_cache(batch_size, max_seq_len, dtype=dtype)
            self.block_table_tensors = None
            self.seq_lens_tensors = None
            self.free_block_ids = []
        self.block_table = [[] for _ in range(batch_size)]
        self.physical_block_ids = list(range(self.total_physical_blocks))
        self.allocated_block_count = 0
        self._refresh_block_table()

    def reset(self):
        self.current_len = 0
        if self.layout == "blocked":
            self.block_table = [[] for _ in range(self.batch_size)]
            self.free_block_ids = list(range(self.total_physical_blocks))
        self._refresh_block_table()

    def _capacity_to_blocks(self, seq_len):
        if self.tokens_per_block <= 0:
            return 1
        return max(1, (seq_len + self.tokens_per_block - 1) // self.tokens_per_block)

    def _used_blocks(self, seq_len):
        if seq_len <= 0:
            return 0
        return self._capacity_to_blocks(seq_len)

    def _refresh_block_table(self):
        used_blocks = self._used_blocks(self.current_len)
        if self.layout == "blocked":
            self.block_table = [row[:used_blocks] for row in self.block_table]
            allocated_ids = sorted({block_id for row in self.block_table for block_id in row})
            self.allocated_block_count = len(allocated_ids)
        else:
            self.allocated_block_count = used_blocks
            self.block_table = [[] for _ in range(self.batch_size)]
        self._sync_paged_metadata_tensors()

    def _sync_paged_metadata_tensors(self):
        if self.layout != "blocked":
            return
        for block_table_tensor, seq_lens_tensor in zip(self.block_table_tensors, self.seq_lens_tensors):
            block_table_tensor.fill_(-1)
            for batch_idx, row in enumerate(self.block_table):
                if row:
                    block_table_tensor[batch_idx, :len(row)].copy_(
                        torch.tensor(row, device=block_table_tensor.device, dtype=torch.int32)
                    )
            seq_lens_tensor.fill_(self.current_len)

    def _next_capacity(self, required_len):
        grown = max(required_len, int(max(self.max_seq_len, 1) * self.growth_factor))
        if self.page_block_size > 0:
            grown = ((grown + self.page_block_size - 1) // self.page_block_size) * self.page_block_size
        return grown

    def grow(self, new_max_seq_len):
        if new_max_seq_len <= self.max_seq_len:
            return False

        if self.layout == "blocked":
            new_num_blocks = self._capacity_to_blocks(new_max_seq_len)
            new_total_physical_blocks = self.batch_size * new_num_blocks
            new_local_kv = self.decoder.allocate_blocked_local_kv_cache(
                new_total_physical_blocks,
                self.tokens_per_block,
                dtype=self.dtype,
            )
            for rank, (new_k, new_v) in enumerate(new_local_kv):
                old_k, old_v = self.local_kv[rank]
                new_k[:self.total_physical_blocks].copy_(old_k)
                new_v[:self.total_physical_blocks].copy_(old_v)
            self.local_kv = new_local_kv
            old_total_physical_blocks = self.total_physical_blocks
            self.max_seq_len = new_max_seq_len
            self.grow_count += 1
            self.num_blocks = new_num_blocks
            self.total_physical_blocks = new_total_physical_blocks
            self.physical_block_ids = list(range(self.total_physical_blocks))
            self.free_block_ids.extend(range(old_total_physical_blocks, new_total_physical_blocks))
            self.block_table_tensors = [
                torch.full((self.batch_size, self.num_blocks), -1, device=device, dtype=torch.int32)
                for device in self.decoder.devices
            ]
            self.seq_lens_tensors = [
                torch.full((self.batch_size,), self.current_len, device=device, dtype=torch.int32)
                for device in self.decoder.devices
            ]
        else:
            new_local_kv = self.decoder.allocate_local_kv_cache(self.batch_size, new_max_seq_len, dtype=self.dtype)
            for rank, (new_k, new_v) in enumerate(new_local_kv):
                old_k, old_v = self.local_kv[rank]
                if self.current_len > 0:
                    new_k[:, :, :self.current_len].copy_(old_k[:, :, :self.current_len])
                    new_v[:, :, :self.current_len].copy_(old_v[:, :, :self.current_len])
            self.local_kv = new_local_kv
            self.max_seq_len = new_max_seq_len
            self.grow_count += 1
            self.num_blocks = self._capacity_to_blocks(new_max_seq_len)
            self.total_physical_blocks = self.num_blocks
            self.physical_block_ids = list(range(self.total_physical_blocks))

        self._refresh_block_table()
        return True

    def append_sharded(self, local_kv_shards):
        if not local_kv_shards:
            raise ValueError("local_kv_shards must not be empty")
        step_len = local_kv_shards[0][0].size(2)
        next_len = self.current_len + step_len
        if next_len > self.max_seq_len:
            if not self.auto_grow:
                raise RuntimeError("KV cache is full")
            self.grow(self._next_capacity(next_len))

        if self.layout == "blocked":
            used_blocks_after = self._used_blocks(next_len)
            for batch_idx in range(self.batch_size):
                while len(self.block_table[batch_idx]) < used_blocks_after:
                    if not self.free_block_ids:
                        raise RuntimeError("paged KV cache has no free blocks")
                    self.block_table[batch_idx].append(self.free_block_ids.pop(0))

            token_offset = 0
            while token_offset < step_len:
                absolute_pos = self.current_len + token_offset
                logical_block = absolute_pos // self.tokens_per_block
                block_offset = absolute_pos % self.tokens_per_block
                chunk_len = min(step_len - token_offset, self.tokens_per_block - block_offset)

                for rank, (src_k, src_v) in enumerate(local_kv_shards):
                    k_pages, v_pages = self.local_kv[rank]
                    for batch_idx in range(self.batch_size):
                        physical_block = self.block_table[batch_idx][logical_block]
                        k_pages[physical_block, :, block_offset:block_offset + chunk_len].copy_(
                            src_k[batch_idx, :, token_offset:token_offset + chunk_len]
                        )
                        v_pages[physical_block, :, block_offset:block_offset + chunk_len].copy_(
                            src_v[batch_idx, :, token_offset:token_offset + chunk_len]
                        )
                token_offset += chunk_len
        else:
            for rank, (src_k, src_v) in enumerate(local_kv_shards):
                dst_k, dst_v = self.local_kv[rank]
                dst_k[:, :, self.current_len:next_len].copy_(src_k)
                dst_v[:, :, self.current_len:next_len].copy_(src_v)
        self.current_len = next_len
        self._refresh_block_table()

    def get_active_kv(self):
        if self.current_len <= 0:
            raise RuntimeError("KV cache is empty")
        if self.layout == "blocked":
            active_kv = []
            for k_pages, v_pages in self.local_kv:
                local_h_kv = k_pages.size(1)
                D = k_pages.size(3)
                local_k = torch.empty(
                    self.batch_size, local_h_kv, self.current_len, D, device=k_pages.device, dtype=self.dtype
                )
                local_v = torch.empty_like(local_k)
                for batch_idx in range(self.batch_size):
                    for logical_block, physical_block in enumerate(self.block_table[batch_idx]):
                        start = logical_block * self.tokens_per_block
                        end = min(start + self.tokens_per_block, self.current_len)
                        valid_tokens = end - start
                        if valid_tokens <= 0:
                            continue
                        local_k[batch_idx, :, start:end].copy_(k_pages[physical_block, :, :valid_tokens])
                        local_v[batch_idx, :, start:end].copy_(v_pages[physical_block, :, :valid_tokens])
                active_kv.append((local_k, local_v))
            return active_kv
        return [
            (
                local_k[:, :, :self.current_len].contiguous(),
                local_v[:, :, :self.current_len].contiguous(),
            )
            for local_k, local_v in self.local_kv
        ]

    def capacity_tokens(self):
        return self.max_seq_len

    def metadata(self):
        return {
            "batch_size": self.batch_size,
            "current_len": self.current_len,
            "capacity": self.max_seq_len,
            "dtype": str(self.dtype),
            "layout": self.layout,
            "page_block_size": self.page_block_size,
            "tokens_per_block": self.tokens_per_block,
            "num_blocks": self.num_blocks,
            "total_physical_blocks": self.total_physical_blocks,
            "allocated_block_count": self.allocated_block_count,
            "auto_grow": self.auto_grow,
            "grow_count": self.grow_count,
        }

    def block_metadata(self):
        return {
            "layout": self.layout,
            "tokens_per_block": self.tokens_per_block,
            "num_blocks": self.num_blocks,
            "total_physical_blocks": self.total_physical_blocks,
            "allocated_block_count": self.allocated_block_count,
            "physical_block_ids": list(self.physical_block_ids),
            "block_table": [list(row) for row in self.block_table],
        }

    def supports_paged_decode(self, compiled_block_size=16):
        return self.layout == "blocked" and self.tokens_per_block == compiled_block_size

    def build_paged_cache_tensors(self, compiled_block_size=16):
        if not self.supports_paged_decode(compiled_block_size=compiled_block_size):
            raise ValueError(
                f"paged decode requires blocked layout with page_block_size={compiled_block_size}, "
                f"got layout={self.layout}, page_block_size={self.tokens_per_block}"
            )
        if self.current_len <= 0:
            raise RuntimeError("KV cache is empty")
        self._sync_paged_metadata_tensors()
        return [
            (k_pages, v_pages, block_table, seq_lens, self.num_blocks)
            for (k_pages, v_pages), block_table, seq_lens in zip(
                self.local_kv, self.block_table_tensors, self.seq_lens_tensors
            )
        ]

    def paged_storage_ptrs(self):
        if self.layout != "blocked":
            return []
        return [
            (k_pages.data_ptr(), v_pages.data_ptr())
            for k_pages, v_pages in self.local_kv
        ]

    def paged_metadata_ptrs(self):
        if self.layout != "blocked":
            return []
        return [
            (block_table.data_ptr(), seq_lens.data_ptr())
            for block_table, seq_lens in zip(self.block_table_tensors, self.seq_lens_tensors)
        ]


class TensorParallelDecoder:
    def __init__(self, H_Q, H_KV, D, world_size=DEFAULT_WORLD_SIZE, root_rank=0):
        self.H_Q = H_Q
        self.H_KV = H_KV
        self.D = D
        self.world_size = world_size
        self.root_rank = root_rank
        self.devices = available_devices(world_size)
        self.root_device = self.devices[root_rank]
        self.local_H_Q = H_Q // world_size
        self.local_H_KV = H_KV // world_size

        assert torch.cuda.device_count() >= world_size, "not enough GPUs"
        assert H_Q % world_size == 0, f"H_Q={H_Q} must be divisible by world_size={world_size}"
        assert H_KV % world_size == 0, f"H_KV={H_KV} must be divisible by world_size={world_size}"

        self.streams = [torch.cuda.Stream(device=device) for device in self.devices]
        self.p2p_matrix = [
            [src == dst or torch.cuda.can_device_access_peer(src, dst) for dst in range(world_size)]
            for src in range(world_size)
        ]

    def create_workspace(self, B, kv_seq_len, q_seq_len=1, dtype=torch.float16, include_kv=True):
        return TensorParallelWorkspace(self, B, kv_seq_len, q_seq_len=q_seq_len, dtype=dtype, include_kv=include_kv)

    def _head_range_q(self, rank):
        start = rank * self.local_H_Q
        return start, start + self.local_H_Q

    def _head_range_kv(self, rank):
        start = rank * self.local_H_KV
        return start, start + self.local_H_KV

    def shard_q(self, q, workspace=None):
        if workspace is not None:
            workspace.resize_if_needed(q.size(0), workspace.kv_seq_len, q_seq_len=q.size(2), dtype=q.dtype)
            for rank, device in enumerate(self.devices):
                h_start, h_end = self._head_range_q(rank)
                workspace.local_q_shards[rank].copy_(q[:, h_start:h_end].contiguous().to(device, non_blocking=True))
            return workspace.local_q_shards

        shards = []
        for rank, device in enumerate(self.devices):
            h_start, h_end = self._head_range_q(rank)
            shards.append(q[:, h_start:h_end].contiguous().to(device, non_blocking=True))
        return shards

    def shard_kv(self, k, v, workspace=None):
        if workspace is not None:
            assert workspace.local_kv_shards is not None, "workspace was created without KV shards"
            workspace.resize_if_needed(k.size(0), k.size(2), q_seq_len=workspace.q_seq_len, dtype=k.dtype)
            for rank, device in enumerate(self.devices):
                h_start, h_end = self._head_range_kv(rank)
                local_k, local_v = workspace.local_kv_shards[rank]
                local_k.copy_(k[:, h_start:h_end].contiguous().to(device, non_blocking=True))
                local_v.copy_(v[:, h_start:h_end].contiguous().to(device, non_blocking=True))
            return workspace.local_kv_shards

        shards = []
        for rank, device in enumerate(self.devices):
            h_start, h_end = self._head_range_kv(rank)
            local_k = k[:, h_start:h_end].contiguous().to(device, non_blocking=True)
            local_v = v[:, h_start:h_end].contiguous().to(device, non_blocking=True)
            shards.append((local_k, local_v))
        return shards

    def shard_qkv(self, q, k, v, workspace=None):
        if workspace is not None:
            workspace.resize_if_needed(q.size(0), k.size(2), q_seq_len=q.size(2), dtype=q.dtype)
        return self.shard_q(q, workspace=workspace), self.shard_kv(k, v, workspace=workspace)

    def allocate_local_kv_cache(self, B, max_seq_len, dtype=torch.float16):
        caches = []
        for device in self.devices:
            local_k = torch.zeros(B, self.local_H_KV, max_seq_len, self.D, device=device, dtype=dtype)
            local_v = torch.zeros(B, self.local_H_KV, max_seq_len, self.D, device=device, dtype=dtype)
            caches.append((local_k, local_v))
        return caches

    def allocate_blocked_local_kv_cache(self, total_blocks, tokens_per_block, dtype=torch.float16):
        caches = []
        for device in self.devices:
            local_k = torch.zeros(total_blocks, self.local_H_KV, tokens_per_block, self.D, device=device, dtype=dtype)
            local_v = torch.zeros_like(local_k)
            caches.append((local_k, local_v))
        return caches

    def gather_local_outputs(self, local_outputs, workspace=None):
        B, _, seq_len, D = local_outputs[0].shape
        if workspace is not None:
            workspace.resize_if_needed(B, workspace.kv_seq_len, q_seq_len=seq_len, dtype=local_outputs[0].dtype)
            full_out = workspace.gathered_out
        else:
            full_out = torch.empty(B, self.H_Q, seq_len, D, device=self.root_device, dtype=local_outputs[0].dtype)
        for rank, local_out in enumerate(local_outputs):
            h_start, h_end = self._head_range_q(rank)
            full_out[:, h_start:h_end].copy_(local_out.to(self.root_device, non_blocking=True))
        return full_out

    def decode_step_sharded(self, local_q_shards, local_kv_shards, cache_len, causal=True, workspace=None):
        outputs = workspace.local_outs if workspace is not None else [None] * self.world_size
        for rank, _ in enumerate(self.devices):
            local_q = local_q_shards[rank]
            local_k, local_v = local_kv_shards[rank]
            with torch.cuda.stream(self.streams[rank]):
                if workspace is not None:
                    flash_attn_v100.forward_decode_gqa_fp16_out(
                        outputs[rank], local_q, local_k, local_v, causal, cache_len
                    )
                else:
                    outputs[rank] = flash_attn_v100.forward_decode_gqa_fp16(
                        local_q, local_k, local_v, causal, cache_len
                    )

        synchronize_devices(self.devices)
        return self.gather_local_outputs(outputs, workspace=workspace)

    def decode_step(self, q, k, v, cache_len, causal=True, workspace=None):
        local_q_shards, local_kv_shards = self.shard_qkv(q, k, v, workspace=workspace)
        return self.decode_step_sharded(local_q_shards, local_kv_shards, cache_len, causal, workspace=workspace)

    def decode_step_local(self, q, kv_caches, cache_len, causal=True, workspace=None):
        if workspace is not None:
            workspace.resize_if_needed(q.size(0), workspace.kv_seq_len, q_seq_len=q.size(2), dtype=q.dtype)
        local_q_shards = self.shard_q(q, workspace=workspace)
        return self.decode_step_sharded(local_q_shards, kv_caches, cache_len, causal, workspace=workspace)

    def decode_step_local_paged(self, q, paged_kv_caches, workspace=None):
        if len(paged_kv_caches) != self.world_size:
            raise ValueError("paged_kv_caches size must match world_size")
        if workspace is not None:
            workspace.resize_if_needed(q.size(0), workspace.kv_seq_len, q_seq_len=q.size(2), dtype=q.dtype)

        local_q_shards = self.shard_q(q, workspace=workspace)
        outputs = [None] * self.world_size

        for rank, _ in enumerate(self.devices):
            local_q = local_q_shards[rank]
            k_pages, v_pages, block_table, seq_lens, max_num_blocks = paged_kv_caches[rank]
            with torch.cuda.stream(self.streams[rank]):
                outputs[rank] = flash_attn_v100.forward_paged_decode_gqa_fp16(
                    local_q, k_pages, v_pages, block_table, seq_lens, max_num_blocks
                )

        synchronize_devices(self.devices)
        return self.gather_local_outputs(outputs, workspace=workspace)

    def prefill_step_sharded(self, local_q_shards, local_kv_shards, causal=True, gather_output=True, workspace=None):
        outputs = workspace.local_outs if workspace is not None else [None] * self.world_size
        for rank, _ in enumerate(self.devices):
            local_q = local_q_shards[rank]
            local_k, local_v = local_kv_shards[rank]
            with torch.cuda.stream(self.streams[rank]):
                if workspace is not None:
                    flash_attn_v100.forward_prefill_gqa_fp16_out(
                        outputs[rank], local_q, local_k, local_v, causal
                    )
                else:
                    outputs[rank] = flash_attn_v100.forward_prefill_gqa_fp16(
                        local_q, local_k, local_v, causal
                    )

        synchronize_devices(self.devices)
        if not gather_output:
            return outputs
        return self.gather_local_outputs(outputs, workspace=workspace)

    def prefill_step(self, q, k, v, causal=True, workspace=None):
        local_q_shards, local_kv_shards = self.shard_qkv(q, k, v, workspace=workspace)
        return self.prefill_step_sharded(
            local_q_shards, local_kv_shards, causal=causal, gather_output=True, workspace=workspace
        )

    def kv_cache_bytes_per_gpu(self, B, seq_len, dtype_bytes=2):
        return B * self.local_H_KV * seq_len * self.D * 2 * dtype_bytes


class TPDecodeRunner:
    def __init__(self, config: TensorParallelConfig):
        self.config = config
        self.decoder = TensorParallelDecoder(
            config.H_Q,
            config.H_KV,
            config.D,
            world_size=config.world_size,
            root_rank=config.root_rank,
        )
        self.workspace = None
        self.prefill_workspace = None
        self.decode_graph_buckets = []
        self.stats = TPRuntimeStats()

    def _check_qkv(self, q, k, v, decode_mode):
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("q, k, v must be CUDA tensors")
        if q.dtype != self.config.dtype:
            raise ValueError("q dtype does not match runner config")
        if k.dtype != self.config.dtype or v.dtype != self.config.dtype:
            raise ValueError("k/v dtype does not match runner config")
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError("q, k, v must be 4D tensors")
        if q.size(0) != k.size(0) or q.size(0) != v.size(0):
            raise ValueError("batch size mismatch")
        if q.size(1) != self.config.H_Q:
            raise ValueError("q head count does not match config")
        if k.size(1) != self.config.H_KV or v.size(1) != self.config.H_KV:
            raise ValueError("kv head count does not match config")
        if q.size(3) != self.config.D or k.size(3) != self.config.D or v.size(3) != self.config.D:
            raise ValueError("head dim mismatch")
        if k.size(2) != v.size(2):
            raise ValueError("k and v sequence length must match")
        if decode_mode:
            if q.size(2) != 1:
                raise ValueError("decode mode expects q sequence length == 1")
        else:
            if q.size(2) != k.size(2) or q.size(2) != v.size(2):
                raise ValueError("prefill mode expects q/k/v sequence lengths to match")

    def create_kv_cache(self, batch_size, max_seq_len, dtype=None, layout=None):
        kv_cache = TensorParallelKVCache(
            self.decoder,
            batch_size,
            max_seq_len,
            dtype=dtype or self.config.dtype,
            auto_grow=self.config.auto_grow_kv_cache,
            growth_factor=self.config.kv_cache_growth_factor,
            page_block_size=self.config.kv_cache_page_block_size,
            layout=layout or self.config.kv_cache_layout,
        )
        self.stats.last_kv_cache_capacity = kv_cache.max_seq_len
        self.stats.last_operation = "create_kv_cache"
        return kv_cache

    def reset_kv_cache(self, kv_cache: TensorParallelKVCache):
        kv_cache.reset()
        self.stats.kv_cache_reset_calls += 1
        self.stats.last_cache_len = 0
        self.stats.last_kv_cache_capacity = kv_cache.max_seq_len
        self.stats.last_operation = "reset_kv_cache"

    def _check_kv_cache(self, kv_cache: TensorParallelKVCache, batch_size, dtype):
        if kv_cache.decoder is not self.decoder:
            raise ValueError("kv_cache was created by another decoder")
        if kv_cache.batch_size != batch_size:
            raise ValueError("kv_cache batch size does not match inputs")
        if kv_cache.dtype != dtype:
            raise ValueError("kv_cache dtype does not match inputs")

    def append_to_kv_cache(self, kv_cache: TensorParallelKVCache, k, v):
        if k.dim() != 4 or v.dim() != 4:
            raise ValueError("k and v must be 4D tensors")
        if k.size(1) != self.config.H_KV or v.size(1) != self.config.H_KV:
            raise ValueError("kv head count does not match config")
        if k.size(3) != self.config.D or v.size(3) != self.config.D:
            raise ValueError("kv head dim does not match config")
        if k.size(2) != v.size(2):
            raise ValueError("k and v sequence length must match")
        self._check_kv_cache(kv_cache, k.size(0), k.dtype)
        old_grow_count = kv_cache.grow_count
        start = time.perf_counter()
        local_kv_shards = self.decoder.shard_kv(k, v)
        kv_cache.append_sharded(local_kv_shards)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.stats.kv_cache_append_calls += 1
        self.stats.kv_cache_grow_count += kv_cache.grow_count - old_grow_count
        self.stats.last_cache_len = kv_cache.current_len
        self.stats.last_kv_cache_capacity = kv_cache.max_seq_len
        self.stats.last_kv_append_ms = elapsed_ms
        self.stats.last_operation = "append_to_kv_cache"
        return kv_cache

    def ensure_decode_workspace(self, B, kv_seq_len, q_seq_len=1, dtype=None, include_kv=True):
        target_dtype = dtype or self.config.dtype
        if self.workspace is None:
            self.workspace = self.decoder.create_workspace(
                B, kv_seq_len, q_seq_len=q_seq_len, dtype=target_dtype, include_kv=include_kv
            )
            self.stats.decode_workspace_resize_count += 1
        else:
            resized = self.workspace.resize_if_needed(B, kv_seq_len, q_seq_len=q_seq_len, dtype=target_dtype)
            self.stats.decode_workspace_resize_count += int(resized)
        return self.workspace

    def ensure_prefill_workspace(self, B, seq_len, dtype=None, include_kv=True):
        target_dtype = dtype or self.config.dtype
        if self.prefill_workspace is None:
            self.prefill_workspace = self.decoder.create_workspace(
                B, seq_len, q_seq_len=seq_len, dtype=target_dtype, include_kv=include_kv
            )
            self.stats.prefill_workspace_resize_count += 1
        else:
            resized = self.prefill_workspace.resize_if_needed(B, seq_len, q_seq_len=seq_len, dtype=target_dtype)
            self.stats.prefill_workspace_resize_count += int(resized)
        return self.prefill_workspace

    def decode(self, q, k, v, cache_len=None, causal=None, use_workspace=True):
        self._check_qkv(q, k, v, decode_mode=True)
        causal = self.config.causal if causal is None else causal
        cache_len = (k.size(2) - 1) if cache_len is None else cache_len
        workspace = self.ensure_decode_workspace(q.size(0), k.size(2), q_seq_len=q.size(2), dtype=q.dtype) if use_workspace else None
        start = time.perf_counter()
        out = self.decoder.decode_step(q, k, v, cache_len, causal=causal, workspace=workspace)
        self.stats.decode_calls += 1
        self.stats.last_cache_len = k.size(2)
        self.stats.last_decode_ms = (time.perf_counter() - start) * 1000.0
        self.stats.last_operation = "decode"
        return out

    def decode_local(self, q, kv_caches, cache_len, causal=None, use_workspace=True):
        if len(kv_caches) != self.config.world_size:
            raise ValueError("kv_caches size must match world_size")
        causal = self.config.causal if causal is None else causal
        kv_seq_len = kv_caches[0][0].size(2)
        workspace = self.ensure_decode_workspace(q.size(0), kv_seq_len, q_seq_len=q.size(2), dtype=q.dtype, include_kv=False) if use_workspace else None
        start = time.perf_counter()
        out = self.decoder.decode_step_local(q, kv_caches, cache_len, causal=causal, workspace=workspace)
        self.stats.decode_local_calls += 1
        self.stats.last_cache_len = kv_seq_len
        self.stats.last_decode_ms = (time.perf_counter() - start) * 1000.0
        self.stats.last_operation = "decode_local"
        return out

    def prefill(self, q, k, v, causal=None, use_workspace=True):
        self._check_qkv(q, k, v, decode_mode=False)
        causal = self.config.causal if causal is None else causal
        workspace = self.ensure_prefill_workspace(q.size(0), q.size(2), dtype=q.dtype) if use_workspace else None
        start = time.perf_counter()
        out = self.decoder.prefill_step(q, k, v, causal=causal, workspace=workspace)
        self.stats.prefill_calls += 1
        self.stats.last_cache_len = q.size(2)
        self.stats.last_prefill_ms = (time.perf_counter() - start) * 1000.0
        self.stats.last_operation = "prefill"
        return out

    def shard_kv(self, k, v, use_decode_workspace=False):
        workspace = None
        if use_decode_workspace:
            workspace = self.ensure_decode_workspace(k.size(0), k.size(2), q_seq_len=1, dtype=k.dtype)
        return self.decoder.shard_kv(k, v, workspace=workspace)

    def allocate_local_kv_cache(self, B, max_seq_len, dtype=None):
        return self.decoder.allocate_local_kv_cache(B, max_seq_len, dtype=dtype or self.config.dtype)

    def kv_cache_bytes_per_gpu(self, B, seq_len, dtype_bytes=2):
        return self.decoder.kv_cache_bytes_per_gpu(B, seq_len, dtype_bytes=dtype_bytes)

    def get_kv_cache_metadata(self, kv_cache: TensorParallelKVCache):
        self._check_kv_cache(kv_cache, kv_cache.batch_size, kv_cache.dtype)
        return kv_cache.metadata()

    def get_kv_cache_block_metadata(self, kv_cache: TensorParallelKVCache):
        self._check_kv_cache(kv_cache, kv_cache.batch_size, kv_cache.dtype)
        return kv_cache.block_metadata()

    def prefill_to_kv_cache(self, q, k, v, kv_cache: TensorParallelKVCache, causal=None, return_output=True, use_workspace=True):
        self._check_qkv(q, k, v, decode_mode=False)
        self._check_kv_cache(kv_cache, q.size(0), q.dtype)
        kv_cache.reset()
        self.append_to_kv_cache(kv_cache, k, v)
        if not return_output:
            return None
        return self.prefill(q, k, v, causal=causal, use_workspace=use_workspace)

    def decode_with_kv_cache(self, q, kv_cache: TensorParallelKVCache, new_k=None, new_v=None, causal=None, use_workspace=True):
        if new_k is not None or new_v is not None:
            if new_k is None or new_v is None:
                raise ValueError("new_k and new_v must be provided together")
            self.append_to_kv_cache(kv_cache, new_k, new_v)
        self._check_kv_cache(kv_cache, q.size(0), q.dtype)
        active_kv = kv_cache.get_active_kv()
        cache_len = kv_cache.current_len - 1
        return self.decode_local(q, active_kv, cache_len, causal=causal, use_workspace=use_workspace)

    def decode_with_paged_kv_cache(self, q, kv_cache: TensorParallelKVCache, new_k=None, new_v=None, use_workspace=True):
        if q.dim() != 4 or q.size(2) != 1:
            raise ValueError("paged decode expects q to have shape [B, H_Q, 1, D]")
        if new_k is not None or new_v is not None:
            if new_k is None or new_v is None:
                raise ValueError("new_k and new_v must be provided together")
            self.append_to_kv_cache(kv_cache, new_k, new_v)
        self._check_kv_cache(kv_cache, q.size(0), q.dtype)
        if not kv_cache.supports_paged_decode(compiled_block_size=16):
            raise ValueError("kv_cache is not compatible with the current paged decode CUDA kernel")

        workspace = self.ensure_decode_workspace(
            q.size(0),
            kv_cache.current_len,
            q_seq_len=q.size(2),
            dtype=q.dtype,
            include_kv=False,
        ) if use_workspace else None

        paged_kv_caches = kv_cache.build_paged_cache_tensors(compiled_block_size=16)
        start = time.perf_counter()
        out = self.decoder.decode_step_local_paged(q, paged_kv_caches, workspace=workspace)
        self.stats.decode_paged_calls += 1
        self.stats.last_cache_len = kv_cache.current_len
        self.stats.last_kv_cache_capacity = kv_cache.max_seq_len
        self.stats.last_paged_decode_ms = (time.perf_counter() - start) * 1000.0
        self.stats.last_operation = "decode_paged"
        return out

    def supports_cuda_graph_decode(self, kv_cache: TensorParallelKVCache):
        return (
            cuda_graph_available()
            and kv_cache.decoder is self.decoder
            and kv_cache.layout == "blocked"
            and kv_cache.supports_paged_decode(compiled_block_size=16)
            and kv_cache.current_len > 0
        )

    def create_paged_decode_cuda_graph_bucket(self, kv_cache: TensorParallelKVCache, q_dtype=None):
        q_dtype = q_dtype or self.config.dtype
        self._check_kv_cache(kv_cache, kv_cache.batch_size, q_dtype)
        if not self.supports_cuda_graph_decode(kv_cache):
            raise RuntimeError("paged decode CUDA Graph is not available for the given kv_cache/runtime state")
        if kv_cache.current_len <= 0:
            raise RuntimeError("paged decode CUDA Graph requires a non-empty kv_cache")

        paged_kv_caches = kv_cache.build_paged_cache_tensors(compiled_block_size=16)
        workspace = self.decoder.create_workspace(
            kv_cache.batch_size,
            kv_cache.max_seq_len,
            q_seq_len=1,
            dtype=q_dtype,
            include_kv=False,
        )
        local_q_shards = workspace.local_q_shards
        for local_q in local_q_shards:
            local_q.zero_()

        graphs = []
        local_outputs = []
        for rank, device in enumerate(self.decoder.devices):
            local_q = local_q_shards[rank]
            k_pages, v_pages, block_table, seq_lens, max_num_blocks = paged_kv_caches[rank]
            capture_stream = torch.cuda.Stream(device=device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.device(device):
                torch.cuda.synchronize(device)
                with torch.cuda.stream(capture_stream):
                    _ = flash_attn_v100.forward_paged_decode_gqa_fp16(
                        local_q, k_pages, v_pages, block_table, seq_lens, max_num_blocks
                    )
                capture_stream.synchronize()
                torch.cuda.synchronize(device)
                with torch.cuda.graph(graph, stream=capture_stream):
                    local_out = flash_attn_v100.forward_paged_decode_gqa_fp16(
                        local_q, k_pages, v_pages, block_table, seq_lens, max_num_blocks
                    )
            graphs.append(graph)
            local_outputs.append(local_out)

        bucket = TPCUDAGraphDecodeBucket(
            kv_cache=kv_cache,
            batch_size=kv_cache.batch_size,
            max_seq_len=kv_cache.max_seq_len,
            dtype=q_dtype,
            workspace=workspace,
            graphs=graphs,
            local_outputs=local_outputs,
            storage_ptrs=kv_cache.paged_storage_ptrs(),
            metadata_ptrs=kv_cache.paged_metadata_ptrs(),
            layout=kv_cache.layout,
        )
        self.decode_graph_buckets.append(bucket)
        self.stats.decode_graph_capture_count += 1
        self.stats.last_graph_bucket_seq_len = kv_cache.max_seq_len
        self.stats.last_operation = "create_decode_graph_bucket"
        return bucket

    def _validate_decode_graph_bucket(self, bucket: TPCUDAGraphDecodeBucket, kv_cache: TensorParallelKVCache, q):
        if bucket.kv_cache is not kv_cache:
            raise ValueError("decode graph bucket is bound to a different kv_cache")
        if kv_cache.max_seq_len != bucket.max_seq_len:
            raise ValueError("kv_cache capacity changed after CUDA Graph capture")
        if kv_cache.layout != bucket.layout:
            raise ValueError("kv_cache layout changed after CUDA Graph capture")
        if kv_cache.paged_storage_ptrs() != bucket.storage_ptrs:
            raise ValueError("paged KV storage pointers changed after CUDA Graph capture")
        if kv_cache.paged_metadata_ptrs() != bucket.metadata_ptrs:
            raise ValueError("paged KV metadata pointers changed after CUDA Graph capture")
        if q.size(0) != bucket.batch_size:
            raise ValueError("batch size does not match CUDA Graph bucket")
        if q.dtype != bucket.dtype:
            raise ValueError("q dtype does not match CUDA Graph bucket")

    def decode_with_paged_kv_cache_graph(
        self,
        q,
        kv_cache: TensorParallelKVCache,
        bucket: TPCUDAGraphDecodeBucket,
        new_k=None,
        new_v=None,
        out=None,
    ):
        if q.dim() != 4 or q.size(2) != 1:
            raise ValueError("paged decode graph expects q to have shape [B, H_Q, 1, D]")
        if new_k is not None or new_v is not None:
            if new_k is None or new_v is None:
                raise ValueError("new_k and new_v must be provided together")
            self.append_to_kv_cache(kv_cache, new_k, new_v)
        self._check_kv_cache(kv_cache, q.size(0), q.dtype)
        if not self.supports_cuda_graph_decode(kv_cache):
            raise RuntimeError("paged decode CUDA Graph is not available for the given kv_cache/runtime state")
        self._validate_decode_graph_bucket(bucket, kv_cache, q)
        kv_cache._sync_paged_metadata_tensors()
        self.decoder.shard_q(q, workspace=bucket.workspace)
        start = time.perf_counter()
        for graph in bucket.graphs:
            graph.replay()
        synchronize_devices(self.decoder.devices)
        graph_output = self.decoder.gather_local_outputs(bucket.local_outputs, workspace=bucket.workspace)
        self.stats.decode_graph_replay_calls += 1
        self.stats.decode_paged_calls += 1
        self.stats.last_cache_len = kv_cache.current_len
        self.stats.last_kv_cache_capacity = kv_cache.max_seq_len
        self.stats.last_graph_bucket_seq_len = bucket.max_seq_len
        self.stats.last_paged_decode_ms = (time.perf_counter() - start) * 1000.0
        self.stats.last_operation = "decode_paged_graph"
        if out is not None:
            out.copy_(graph_output)
            return out
        return graph_output

    def get_runtime_stats(self):
        stats_dict = asdict(self.stats)
        stats_dict["config"] = {
            "H_Q": self.config.H_Q,
            "H_KV": self.config.H_KV,
            "D": self.config.D,
            "world_size": self.config.world_size,
            "root_rank": self.config.root_rank,
            "dtype": str(self.config.dtype),
            "causal": self.config.causal,
            "auto_grow_kv_cache": self.config.auto_grow_kv_cache,
            "kv_cache_growth_factor": self.config.kv_cache_growth_factor,
            "kv_cache_page_block_size": self.config.kv_cache_page_block_size,
            "kv_cache_layout": self.config.kv_cache_layout,
        }
        return stats_dict
