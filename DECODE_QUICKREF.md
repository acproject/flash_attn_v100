# Decode Kernel 快速参考

## API 接口

### Python
```python
import flash_attn_v100

out = flash_attn_v100.forward_decode_fp16(
    q,          # [B, H, 1, D] - 当前 token 的 Q
    k,          # [B, H, N, D] - 包含 cache 的 K
    v,          # [B, H, N, D] - 包含 cache 的 V
    True,       # causal (bool)
    cache_len   # 历史 cache 长度 (int)
)
# 返回: [B, H, 1, D]
```

## 快速示例

### 基本使用
```python
import torch
import flash_attn_v100

B, H, D = 1, 8, 64
cache_len = 100
total_len = cache_len + 1

q = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16).contiguous()
k = torch.randn(B, H, total_len, D, device='cuda', dtype=torch.float16).contiguous()
v = torch.randn(B, H, total_len, D, device='cuda', dtype=torch.float16).contiguous()

out = flash_attn_v100.forward_decode_fp16(q, k, v, True, cache_len)
```

### KV Cache 管理
```python
# 预分配 cache
max_len = 2048
cache_k = torch.zeros(B, H, max_len, D, device='cuda', dtype=torch.float16)
cache_v = torch.zeros(B, H, max_len, D, device='cuda', dtype=torch.float16)
current_len = 0

# 自回归生成循环
for step in range(num_steps):
    # 生成新 token
    q = ...  # [B, H, 1, D]
    k_new = ...  # [B, H, 1, D]
    v_new = ...  # [B, H, 1, D]
    
    # 更新 cache
    cache_k[:, :, current_len] = k_new.squeeze(2)
    cache_v[:, :, current_len] = v_new.squeeze(2)
    current_len += 1
    
    # 获取 cache
    k_cache = cache_k[:, :, :current_len].contiguous()
    v_cache = cache_v[:, :, :current_len].contiguous()
    
    # 计算 attention
    out = flash_attn_v100.forward_decode_fp16(
        q, k_cache, v_cache, True, current_len - 1
    )
```

## 关键约束

✅ **必须满足**:
- Q shape: `[B, H, 1, D]` (第 2 维必须是 1)
- K/V shape: `[B, H, N, D]` (N = cache_len + 1)
- 数据类型: `torch.float16`
- 内存连续: `.contiguous()`
- `D <= 128`
- `cache_len < N`

❌ **常见错误**:
```python
# 错误 1: Q 的 seq_len 不是 1
q = torch.randn(B, H, 10, D)  # ✗ 必须是 1

# 错误 2: 忘记 contiguous()
k = cache_k[:, :, :current_len]  # ✗ 需要 .contiguous()

# 错误 3: 数据类型错误
q = torch.randn(B, H, 1, D)  # ✗ 默认是 float32
q = q.half()  # ✓ 转换为 float16

# 错误 4: cache_len 不正确
out = forward_decode_fp16(q, k, v, True, N)  # ✗ 应该是 N-1
```

## 性能参考

| Cache Len | Time (ms) | Tokens/sec |
|-----------|-----------|------------|
| 256 | ~1.0 | ~1000 |
| 512 | ~1.8 | ~560 |
| 1024 | ~3.3 | ~300 |
| 2048 | ~6.7 | ~150 |
| 4096 | ~13.7 | ~73 |

*测试环境: V100, B=1, H=8, D=64*

## 调试技巧

### 检查 NaN/Inf
```python
out = flash_attn_v100.forward_decode_fp16(q, k, v, True, cache_len)
assert not torch.isnan(out).any(), "Output contains NaN"
assert not torch.isinf(out).any(), "Output contains Inf"
```

### 验证正确性
```python
# 与 PyTorch 对比
scores = torch.matmul(q.float(), k.transpose(-1, -2).float()) / (D ** 0.5)
mask = torch.ones(1, total_len, device='cuda')
mask[:, cache_len+1:] = float('-inf')
scores = scores + mask
probs = torch.softmax(scores, dim=-1)
out_ref = torch.matmul(probs, v.float())

max_diff = (out.float() - out_ref).abs().max().item()
print(f"Max diff: {max_diff}")  # 应该 < 0.001
```

### 性能分析
```python
import time

torch.cuda.synchronize()
start = time.time()

for _ in range(100):
    out = flash_attn_v100.forward_decode_fp16(q, k, v, True, cache_len)

torch.cuda.synchronize()
elapsed = time.time() - start
print(f"Average time: {elapsed/100*1000:.3f} ms")
print(f"Throughput: {100/elapsed:.1f} tokens/sec")
```

## 与 Prefill 的对比

| 特性 | Prefill | Decode |
|------|---------|--------|
| Q shape | `[B,H,N,D]` | `[B,H,1,D]` |
| K/V shape | `[B,H,N,D]` | `[B,H,N,D]` |
| API | `forward_fp16` | `forward_decode_fp16` |
| 参数 | `causal` | `causal, cache_len` |
| 复杂度 | O(N²) | O(N) |
| 适用 | Prompt 处理 | Token 生成 |

## 编译命令

```bash
# 重新编译
cd /home/acproject/workspace/python_projects/flash_attn_v100
source venv/bin/activate
python setup.py build_ext --inplace

# 运行测试
python test_decode.py
python example_decode_usage.py
```

## 文件位置

- CUDA 内核: `flash_attn_kernel.cu` (第 623-770 行)
- C++ 绑定: `flash_attn.cpp` (第 20-26 行)
- 测试: `test_decode.py`
- 示例: `example_decode_usage.py`
- 文档: `DECODE_KERNEL.md`, `DECODE_SUMMARY.md`
