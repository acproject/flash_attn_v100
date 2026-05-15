# Flash Attention V100

[English](README_EN.md) [中文](README.md)

针对 NVIDIA V100 GPU 优化的高效 Flash Attention CUDA 实现，支持从训练到生产级 LLM Serving 的完整技术栈。

## 🌟 核心特性

- ✅ **多精度支持**: FP32 和 FP16 实现
- ✅ **KV Cache 优化**: Decode 阶段高效推理
- ✅ **GQA/MQA 支持**: 兼容 Llama 3、Mistral 等现代 LLM
- ✅ **Continuous Batching**: 动态序列长度，消除 padding
- ✅ **WMMA Tensor Core**: 使用 Volta Tensor Core 加速矩阵乘法
- ✅ **生产就绪**: 完整的测试和 benchmark

## 📊 性能表现

### 吞吐量提升

| 优化阶段 | 吞吐量 | 提升倍数 |
|---------|--------|---------|
| FP16 + Warp Reduction | ~800 tokens/s | 2x |
| Decode Kernel | ~1000 tokens/s | 2.5x |
| GQA (32:8) | ~1000 tokens/s | 2.5x |
| Continuous Batching | ~2000-5000 tokens/s | **5-10x** |

### 内存优化

- **FP16 vs FP32**: 节省 50% 内存
- **GQA (32:8)**: KV Cache 节省 75%
- **Continuous Batching**: 消除 padding，节省 50-80% 内存
- **综合效果**: 内存需求降低 **85-95%**

## 🚀 快速开始

### 环境要求

- NVIDIA V100 GPU (Volta 架构, sm_70)
- CUDA 12.8
- Python 3.13+
- PyTorch 2.0+

### 安装

```bash
# 激活虚拟环境
cd /home/acproject/workspace/python_projects/flash_attn_v100
source venv/bin/activate

# 编译
python setup.py build_ext --inplace
```

### 基础使用

#### 1. Prefill 阶段 (MHA)

```python
import torch
import flash_attn_v100

# 输入: [Batch, Heads, SeqLen, HeadDim]
q = torch.randn(2, 8, 256, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 8, 256, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 8, 256, 64, device='cuda', dtype=torch.float16)

# FP16 Flash Attention
out = flash_attn_v100.forward_fp16(q, k, v, causal=True)
```

#### 2. Decode 阶段 (KV Cache)

```python
# Q: [B, H, 1, D], K/V: [B, H, cache_len+1, D]
q = torch.randn(1, 8, 1, 64, device='cuda', dtype=torch.float16)
k = torch.randn(1, 8, 101, 64, device='cuda', dtype=torch.float16)
v = torch.randn(1, 8, 101, 64, device='cuda', dtype=torch.float16)

out = flash_attn_v100.forward_decode_fp16(q, k, v, causal=True, cache_len=100)
```

#### 3. GQA/MQA (Llama 3 风格)

```python
# Llama 3: 32 Q heads, 8 KV heads
q = torch.randn(1, 32, 1, 128, device='cuda', dtype=torch.float16)
k = torch.randn(1, 8, 101, 128, device='cuda', dtype=torch.float16)
v = torch.randn(1, 8, 101, 128, device='cuda', dtype=torch.float16)

out = flash_attn_v100.forward_decode_gqa_fp16(q, k, v, causal=True, cache_len=100)
```

#### 4. Continuous Batching

```python
# 不同序列长度的 batch
B = 4
cache_lens = torch.tensor([10, 50, 100, 20], dtype=torch.int32, device='cuda')
max_N = 101

q = torch.randn(B, 32, 1, 128, device='cuda', dtype=torch.float16)
k = torch.randn(B, 8, max_N, 128, device='cuda', dtype=torch.float16)
v = torch.randn(B, 8, max_N, 128, device='cuda', dtype=torch.float16)

out = flash_attn_v100.forward_continuous_batching_fp16(
    q, k, v, cache_lens, causal=True
)
```

#### 5. WMMA Tensor Core 加速

```python
# 使用 WMMA 优化 QK^T 计算 (D 必须是 16 的倍数)
q = torch.randn(2, 8, 256, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 8, 256, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 8, 256, 64, device='cuda', dtype=torch.float16)

out = flash_attn_v100.forward_fp16_wmma(q, k, v, causal=True)
```

#### 6. Varlen Prefill (动态 batch size)

```python
# 不同长度的序列打包到一起，消除 padding
seq_lens = [64, 128, 256]
total_q = sum(seq_lens)
H, D = 8, 128

# 输入: [total_q, H, D] (所有序列拼接)
q_packed = torch.randn(total_q, H, D, device='cuda', dtype=torch.float16)
k_packed = torch.randn(total_q, H, D, device='cuda', dtype=torch.float16)
v_packed = torch.randn(total_q, H, D, device='cuda', dtype=torch.float16)

# 累积序列长度
cu_seqlens = torch.tensor([0, 64, 192, 448], dtype=torch.int32, device='cuda')

out = flash_attn_v100.forward_varlen_prefill_fp16(
    q_packed, k_packed, v_packed, cu_seqlens, max(seq_lens), causal=True
)
# 输出: [total_q, H, D]
```

#### 7. PagedAttention Decode

```python
# 非连续 KV cache，通过 block_table 映射
B, H_Q, H_KV, D = 4, 32, 8, 128
PAGE_BLOCK_SIZE = 16

q = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16)
k_cache = torch.randn(max_pages, H_KV, PAGE_BLOCK_SIZE, D, device='cuda', dtype=torch.float16)
v_cache = torch.randn(max_pages, H_KV, PAGE_BLOCK_SIZE, D, device='cuda', dtype=torch.float16)
block_table = torch.randint(0, max_pages, (B, max_num_blocks), device='cuda', dtype=torch.int32)
seq_lens = torch.tensor([64, 128, 32, 256], dtype=torch.int32, device='cuda')

out = flash_attn_v100.forward_paged_decode_gqa_fp16(
    q, k_cache, v_cache, block_table, seq_lens, max_num_blocks
)
```

#### 8. INT8 Decode

```python
# INT8 量化加速 decode
q = torch.randn(B, H_Q, 1, D, device='cuda', dtype=torch.float16)
k = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)
v = torch.randn(B, H_KV, N, D, device='cuda', dtype=torch.float16)

# 量化 scale (per-head)
q_scale = torch.max(q.abs(), dim=-1).values.float().squeeze(-1) / 127.0
k_scale = torch.max(k.abs(), dim=-1).values.float() / 127.0

out = flash_attn_v100.forward_decode_int8_gqa(
    q, k, v, q_scale, k_scale, cache_len=100
)
```

## 📁 项目结构

```
flash_attn_v100/
├── 核心实现
│   ├── flash_attn_kernel.cu          # 主 CUDA 内核 (包含 WMMA)
│   ├── flash_attn.cpp                # C++/Python 绑定
│   └── continuous_batching_kernel.cu # Continuous Batching 内核
│
├── 测试脚本
│   ├── test.py                       # 基础正确性测试
│   ├── test_decode.py                # Decode Kernel 测试
│   ├── test_gqa.py                   # GQA/MQA 测试
│   ├── test_continuous_batching.py   # Continuous Batching 测试
│   ├── test_wmma.py                  # WMMA 测试
│   ├── test_async.py                 # 异步执行测试
│   └── benchmark.py                  # 性能基准测试
│
├── 示例
│   ├── example_decode_usage.py       # Decode 使用示例
│   ├── example_autoregressive.py     # 自回归生成示例
│   └── visualize_gqa.py              # GQA 可视化
│
└── 文档
    ├── README.md                     # 本文档
    ├── README_EN.md                  # 英文文档
    ├── FINAL_SUMMARY.md              # 完整优化总结
    ├── DECODE_KERNEL.md              # Decode 技术文档
    ├── GQA_MQA_DOCUMENTATION.md      # GQA 技术文档
    └── CONTINUOUS_BATCHING.md        # Continuous Batching 文档
```

## 🔧 API 参考

### Python 接口

```python
import flash_attn_v100

# Prefill 阶段
flash_attn_v100.forward(q, k, v, causal)              # FP32
flash_attn_v100.forward_fp16(q, k, v, causal)         # FP16
flash_attn_v100.forward_fp16_warp(q, k, v, causal)    # FP16 + Warp 优化
flash_attn_v100.forward_fp16_wmma(q, k, v, causal)    # FP16 + WMMA

# Decode 阶段
flash_attn_v100.forward_decode_fp16(q, k, v, causal, cache_len)
flash_attn_v100.forward_decode_gqa_fp16(q, k, v, causal, cache_len)

# Continuous Batching
flash_attn_v100.forward_continuous_batching_fp16(q, k, v, cache_lens, causal)

# Varlen Prefill (动态 batch size)
flash_attn_v100.forward_varlen_prefill_fp16(q, k, v, cu_seqlens, max_seqlen, causal)

# Paged Decode
flash_attn_v100.forward_paged_decode_gqa_fp16(q, k_cache, v_cache, block_table, seq_lens, max_num_blocks)

# INT8 Decode
flash_attn_v100.forward_decode_int8_gqa(q, k, v, q_scale, k_scale, cache_len)

# Stream-aware (异步执行)
flash_attn_v100.forward_decode_gqa_fp16_stream(q, k, v, causal, cache_len, stream_int)
flash_attn_v100.forward_varlen_prefill_fp16_stream(q, k, v, cu_seqlens, max_seqlen, causal, stream_int)
flash_attn_v100.async_h2d_transfer(dst, src, stream_int)
flash_attn_v100.alloc_pinned_tensor(sizes, dtype_int)
flash_attn_v100.pipeline_decode_step(q_compute, k_compute, v_compute,
    q_transfer, k_transfer, v_transfer, q_h2d_dst, k_h2d_dst, v_h2d_dst,
    causal, cache_len, compute_stream_int, transfer_stream_int)

# Speculative Decoding
flash_attn_v100.speculative_verify(target_probs, draft_probs, draft_tokens, random_vals, K)
# Returns: (accepted_count [B], output_tokens [B, K+1])

# INT8 KV Cache Decode
flash_attn_v100.forward_decode_int8_kv_cache_gqa(q, k_int8, v_int8, k_scale, v_scale, cache_len)
# k_int8/v_int8: [B, H_KV, N, D] int8, k_scale/v_scale: [B, H_KV, N] float16

# Token Eviction
flash_attn_v100.token_eviction(k, v, attn_scores, num_evict)
# Returns: (k_out, v_out, valid_mask)

# Sliding Window + Attention Sink Decode
flash_attn_v100.forward_decode_sliding_window_gqa(q, k, v, causal, cache_len, window_size, sink_size)

# Multi-GPU Tensor Parallelism
flash_attn_v100.cross_gpu_allreduce_fp16(dst_tensor, src_tensor)  # Cross-GPU reduce
flash_attn_v100.scatter_heads(full_tensor, local_tensor, total_heads, head_start, head_count)  # Scatter heads to local GPU
flash_attn_v100.gather_heads(full_tensor, local_tensor, total_heads, head_start, head_count)  # Gather heads from local GPU

# TensorParallelDecoder (Python)
from test_multi_gpu import TensorParallelDecoder
tp = TensorParallelDecoder(H_Q=32, H_KV=8, D=128, world_size=2)
out = tp.decode_step(q, k, v, cache_len)  # Multi-GPU decode
out = tp.decode_step_local(q, kv_caches, cache_len)  # Multi-GPU decode with local KV caches
```

### 参数说明

- `q`: Query tensor, shape `[B, H_Q, N, D]`
- `k`: Key tensor, shape `[B, H_KV, N, D]`
- `v`: Value tensor, shape `[B, H_KV, N, D]`
- `causal`: 是否使用 causal mask
- `cache_len`: 历史 KV cache 长度
- `cache_lens`: 每个序列的 cache 长度 (Continuous Batching)

## 📈 性能基准

### 测试环境

- GPU: NVIDIA V100
- CUDA: 12.8
- Precision: FP16

### Decode 性能 (B=1, H=8, D=64)

| Cache Length | Time (ms) | Tokens/sec |
|-------------|-----------|------------|
| 256 | 1.0 | 1000 |
| 512 | 1.8 | 556 |
| 1024 | 3.3 | 303 |
| 2048 | 6.7 | 149 |
| 4096 | 13.7 | 73 |

### GQA 性能对比 (cache_len=512, D=64)

| 配置 | H_Q | H_KV | Tokens/sec | KV Cache 节省 |
|-----|-----|------|------------|--------------|
| MHA | 8 | 8 | 520 | 0% |
| GQA | 8 | 4 | 520 | 50% |
| GQA | 32 | 8 | 513 | 75% |
| MQA | 32 | 1 | 513 | 97% |

## 🎓 适用场景

### 训练场景
- Transformer 模型训练
- Fine-tuning
- 长序列建模

### 推理场景
- 自回归文本生成
- 文本摘要
- 对话系统

### 现代 LLM 支持
- ✅ Llama 3 (GQA 32:8)
- ✅ Mistral (GQA 32:8)
- ✅ Mixtral (GQA 32:8)
- ✅ DeepSeek V2 (MQA 128:1)

### 生产服务
- LLM API Server
- 多用户并发服务
- 高吞吐场景
- 低延迟要求

## 🔮 未来优化方向

### 短期
- [x] Prefill 阶段 GQA 支持 (Q_len > 1)
- [x] WMMA 性能优化 (N=512时比Basic快10-15%，N=1024时慢12-22%)
- [x] PagedAttention 非连续 KV cache (比连续KV cache decode快20-38%)

### 中期
- [x] INT8 量化支持 (decode阶段D=64快15%, D=128快35%)
- [x] 动态 batch size (Varlen Prefill, padding浪费61%时加速1.5-2.5x)
- [x] 异步执行 (计算和传输重叠, Pipeline模式1.62x加速)

### 长期
- [x] Speculative Decoding (GPU验证内核比CPU快17x, 端到端加速1.9x)
- [x] KV Cache 压缩 (INT8 KV Cache内存省49%/decode快1.6x, 滑动窗口加速3.5-5.7x, Token驱逐)
- [ ] 多 GPU 分布式推理 (Tensor Parallel, 2xV100 KV内存省50%, 长序列prefill接近0.85x)

## 📚 相关资源

### 论文
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [GQA: Training Generalized from Cross-Attention](https://arxiv.org/abs/2305.13245)
- [Orca: A Distributed Serving System with Continuous Batching](https://arxiv.org/abs/2211.05102)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

### 相关项目
- [vLLM](https://github.com/vllm-project/vllm)
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

## ⚠️ 注意事项

1. **WMMA 性能**: 当前 WMMA 实现由于寄存器压力和 shared memory 访问模式问题，性能不如预期。需要进一步优化。
2. **头维度限制**: `D` 必须是 16 的倍数 (64, 128 等) 才能使用 WMMA。
3. **架构要求**: WMMA 需要 Volta (sm_70) 或更新架构。
4. **精度**: FP16 版本与 FP32 参考实现的最大误差约为 1e-2。

## 🙏 致谢

感谢以下开源项目的启发：
- FlashAttention (Tri Dao)
- vLLM (UC Berkeley)
- TGI (Hugging Face)
- TensorRT-LLM (NVIDIA)

## 📄 许可证

本项目用于学习和研究目的。

---

**项目状态**: ✅ 活跃开发中  
**最后更新**: 2026-05-15  
**版本**: 1.0.0
