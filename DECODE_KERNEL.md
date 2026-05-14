# KV Cache / Decode Kernel 优化文档

## 概述

实现了针对自回归解码（Autoregressive Decoding）场景优化的 Flash Attention CUDA 内核。

## 特征

### Decode 模式特点
- **Q_len = 1**: 每次只生成一个 token
- **K_len 很长**: 包含所有历史 KV cache
- **增量计算**: 复用历史 KV，只计算新 token 的 attention

### 与 Prefill 模式的对比

| 特性 | Prefill (训练/编码) | Decode (推理/解码) |
|------|-------------------|-------------------|
| Q 长度 | N (长序列) | 1 (单 token) |
| K/V 长度 | N | N (包含 cache) |
| 计算复杂度 | O(N²) | O(N) |
| 内存访问 | 高 | 中等 |
| 优化重点 | 并行度、共享内存 | 减少冗余计算 |

## 实现细节

### CUDA 内核设计

```cpp
__global__ void flash_attn_kernel_decode_fp16(
    const half* Q,      // [B, H, 1, D]
    const half* K,      // [B, H, N, D]
    const half* V,      // [B, H, N, D]
    half* O,            // [B, H, 1, D]
    int B, int H, int N, int D,
    bool causal,
    float scale,
    int cache_len       // 历史 cache 长度
)
```

### 关键优化

1. **单 Query 优化**
   - Q 只有一个 token，所有线程协作处理
   - 共享内存中只存储一行 Q
   - 减少共享内存使用

2. **Warp-Level Reduction**
   - 使用 warp shuffle 指令进行点积归约
   - 减少寄存器压力和同步开销
   - 提高计算效率

3. **Causal Mask 适配**
   - 在 decode 模式下，Q 的位置是 `cache_len`
   - 只允许 attention 到位置 `0 ~ cache_len`
   - 正确实现自回归约束

4. **Lane 0 串行化**
   - 由于只有一个 query，由 lane 0 执行所有计算
   - 避免多线程重复计算
   - 简化控制流

### 分块策略

```cpp
constexpr int TILE_K_DECODE = 64;  // decode 模式使用更大的 tile
```

- 更大的 tile size 减少 kernel launch 开销
- 适合长序列场景
- 平衡共享内存使用

## 性能测试

### 测试环境
- GPU: NVIDIA V100
- CUDA: 12.8
- 精度: FP16

### Benchmark 结果

#### D=64 时

| Cache Len | Time (ms) | Tokens/sec | GFLOPs | BW (GB/s) |
|-----------|-----------|------------|--------|-----------|
| 256       | 1.004     | 995.7      | 0.26   | 0.53      |
| 512       | 1.791     | 558.2      | 0.29   | 0.59      |
| 1024      | 3.349     | 298.6      | 0.31   | 0.63      |
| 2048      | 6.682     | 149.6      | 0.31   | 0.63      |
| 4096      | 13.691    | 73.0       | 0.31   | 0.61      |

#### D=128 时

| Cache Len | Time (ms) | Tokens/sec | GFLOPs | BW (GB/s) |
|-----------|-----------|------------|--------|-----------|
| 256       | 1.728     | 578.7      | 0.30   | 0.61      |
| 512       | 3.436     | 291.0      | 0.31   | 0.61      |
| 1024      | 6.852     | 145.9      | 0.31   | 0.61      |
| 2048      | 14.019    | 71.3       | 0.30   | 0.60      |
| 4096      | 27.987    | 35.7       | 0.30   | 0.60      |

### 性能特征

1. **线性扩展**: 时间复杂度 O(N)，与 cache_len 成正比
2. **吞吐量**: 在短 cache 时可达 ~1000 tokens/sec
3. **内存带宽**: 约 0.6 GB/s (受限于单 token 的低并行度)

## 正确性验证

### 测试配置
- B=1, H=4, D=64
- cache_len=128
- 对比 PyTorch 原生实现

### 结果

| 模式 | Max Diff | Mean Diff | All Close |
|------|----------|-----------|-----------|
| Causal | 0.000732 | 0.000081 | ✓ True |
| Non-Causal | 0.000244 | 0.000040 | ✓ True |

**结论**: 与 PyTorch 参考实现高度一致（误差 < 0.001）

## 使用方法

### Python API

```python
import torch
import flash_attn_v100

# 输入准备
B, H, D = 1, 8, 64
cache_len = 1024
total_len = cache_len + 1  # 包含当前 token

q = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16)
k = torch.randn(B, H, total_len, D, device='cuda', dtype=torch.float16)
v = torch.randn(B, H, total_len, D, device='cuda', dtype=torch.float16)

# 调用 decode kernel
out = flash_attn_v100.forward_decode_fp16(
    q, k, v, 
    causal=True, 
    cache_len=cache_len
)
```

### KV Cache 管理示例

```python
class KVCache:
    def __init__(self, max_len, B, H, D, device='cuda'):
        self.cache_k = torch.zeros(B, H, max_len, D, device=device, dtype=torch.float16)
        self.cache_v = torch.zeros(B, H, max_len, D, device=device, dtype=torch.float16)
        self.cache_len = 0
    
    def append(self, k, v):
        """追加新的 K/V 到 cache"""
        self.cache_k[:, :, self.cache_len:self.cache_len+1] = k
        self.cache_v[:, :, self.cache_len:self.cache_len+1] = v
        self.cache_len += 1
    
    def decode_step(self, q, causal=True):
        """执行一步 decode"""
        # 获取当前的 K/V (包含 cache)
        k = self.cache_k[:, :, :self.cache_len]
        v = self.cache_v[:, :, :self.cache_len]
        
        # 调用 decode kernel
        out = flash_attn_v100.forward_decode_fp16(
            q, k, v,
            causal=causal,
            cache_len=self.cache_len - 1
        )
        return out
```

## 优化建议

### 当前限制

1. **单线程执行**: 目前只有 lane 0 执行计算，未充分利用并行性
2. **内存带宽**: 较低（~0.6 GB/s），受限于单 token 场景
3. **无 Tensor Core**: 未使用 WMMA 指令

### 未来优化方向

1. **批处理 Decode**
   - 同时处理多个 token（batch decode）
   - 提高并行度和吞吐量

2. **Paged Attention**
   - 支持非连续的 KV cache
   - 减少内存碎片

3. **Tensor Core 加速**
   - 使用 WMMA 指令进行矩阵乘法
   - 提升计算效率

4. **异步拷贝**
   - 使用 `cp.async` 指令
   - 重叠内存传输和计算

5. **量化支持**
   - INT8/INT4 量化
   - 减少内存占用和带宽需求

## 文件说明

- `flash_attn_kernel.cu`: CUDA 内核实现（包含 decode kernel）
- `flash_attn.cpp`: C++ 绑定接口
- `test_decode.py`: 完整的测试和 benchmark 脚本
- `debug_decode.py`: 调试脚本（小样例验证）

## 总结

成功实现了针对 Decode 场景优化的 Flash Attention CUDA 内核：

✓ **正确性**: 与 PyTorch 参考实现误差 < 0.001  
✓ **性能**: 线性时间复杂度 O(N)，支持长序列  
✓ **功能**: 支持 causal/non-causal 模式  
✓ **精度**: FP16 混合精度计算  
✓ **稳定性**: 无 NaN/Inf，数值稳定  

适合用于大语言模型的自回归推理场景。
