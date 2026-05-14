#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cmath>
#include <cfloat>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 配置参数
constexpr int TILE_Q = 32;
constexpr int TILE_K = 32;
constexpr int MAX_D = 128;

// process a block (batch, head, q_tile)
__global__ void flash_attn_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B,
    int H,
    int N,
    int D,
    bool causal,
    float scale
) {
    int q_tile_idx = blockIdx.x;
    int bh = blockIdx.y;

    int b = bh / H;
    int h = bh % H;

    int tid = threadIdx.x;
    int q_start = q_tile_idx * TILE_Q;

    if (q_start >= N) return;
    int q_end = min(TILE_Q, N - q_start);

    // base pointer
    const float* Q_ptr = Q + ((b * H + h) * N * D);
    const float* K_ptr = K + ((b * H + h) * N * D);
    const float* V_ptr = V + ((b * H + h) * N * D);
    float* O_ptr = O + ((b * H + h) * N * D);

    // shared memory
    __shared__ float sQ[TILE_Q][MAX_D];
    __shared__ float sK[TILE_K][MAX_D];
    __shared__ float sV[TILE_K][MAX_D];

    // every thread process a qurey row
    if (tid >= q_end) return;

    // load Q
    for (int d = 0; d < D; ++d) {
        sQ[tid][d] = Q_ptr[(q_start + tid) * D + d];
    }

    // online softmax status
    float m_i = -FLT_MAX;
    float l_i = 0.f;
    float acc[MAX_D];

    for (int d= 0; d< D; ++d) acc[d] = 0.f;
    // iter K/V blocks

    for (int k_start = 0; k_start < N; k_start += TILE_K) {
        int k_len = min(TILE_K, N - k_start);

        // cooperative load K/V
        for (int idx = threadIdx.x; idx < k_len * D; idx += blockDim.x) {
            int row = idx / D;
            int col = idx % D;
            sK[row][col] = K_ptr[(k_start + row) * D + col];
            sV[row][col] = V_ptr[(k_start + row) * D + col];
        }
        __syncthreads();

        // score
        float scores[TILE_K];
        float m_block = -FLT_MAX;

        int q_idx = q_start + tid;

        for (int j = 0; j < k_len; ++j) {
            int k_idx = k_start + j;

            if (causal && k_idx > q_idx) {
                scores[j] = -FLT_MAX;
                continue;
            }

            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += sQ[tid][d] * sK[j][d];
            }

            dot *= scale;
            scores[j] = dot;
            m_block = fmaxf(m_block, dot);
        }

        // all blocks masked
        if (m_block == -FLT_MAX) {
            __syncthreads();
            continue;
        }

        float m_new = fmaxf(m_i, m_block);

        float alpha = (m_i == -FLT_MAX) ? 0.f : expf(m_i - m_new);

        float beta_sum = 0.f;
        float weighted[MAX_D];
        for (int d = 0; d < D; ++d) {
            weighted[d] = 0.f;
        }

        for (int j =0; j < k_len; ++j) {
            if (scores[j] == -FLT_MAX) continue;

            float p = expf(scores[j] - m_new);
            beta_sum += p;

            for (int d = 0; d < D; ++d) {
                weighted[d] += sV[j][d] * p;
            }
        }

        // upadte status
        for (int d = 0; d < D; ++d) {
            acc[d] = alpha * acc[d] + weighted[d];
        }

        l_i = alpha * l_i + beta_sum;
        m_i = m_new;

        __syncthreads();
    }

    // reback write
    for (int d = 0; d < D; ++d) {
        O_ptr[(q_start + tid) * D + d] = acc[d] / l_i;
    }
}

torch::Tensor flash_attn_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    TORCH_CHECK(q.dtype() == torch::kFloat32, "q must be float32");

    auto B = q.size(0);
    auto H = q.size(1);
    auto N = q.size(2);
    auto D = q.size(3);

    TORCH_CHECK(D <= MAX_D, "D must be less than or equal to MAX_D(128)");

    auto out = torch::zeros_like(q);

    dim3 qrid((N + TILE_Q -1) / TILE_Q, B * H);
    dim3 block(TILE_Q);

    float scale = 1.0f / std::sqrt((float)D);

    flash_attn_kernel<<<qrid, block>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        H,
        N,
        D,
        causal,
        scale
    );
    return out;
}


// Warp-level Reduction 辅助函数
__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// 原始 FP16 内核（half2 向量化版本）
__global__ void flash_attn_kernel_fp16(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B,
    int H,
    int N,
    int D,
    bool causal,
    float scale) 
{
    int q_tile_idx = blockIdx.x;
    int bh = blockIdx.y;

    int b = bh / H;
    int h = bh % H;

    int tid = threadIdx.x;
    int q_start = q_tile_idx * TILE_Q;

    if (q_start >= N) return;
    int q_end = min(TILE_Q, N - q_start);

    // base pointer
    const half* Q_ptr = Q + ((b * H + h) * N * D);
    const half* K_ptr = K + ((b * H + h) * N * D);
    const half* V_ptr = V + ((b * H + h) * N * D);
    half* O_ptr = O + ((b * H + h) * N * D);

    // shared memory (使用 half 类型节省空间)
    __shared__ half sQ[TILE_Q][MAX_D];
    __shared__ half sK[TILE_K][MAX_D];
    __shared__ half sV[TILE_K][MAX_D];

    // every thread process a query row
    if (tid >= q_end) return;

    // load Q (使用 half2 向量化加载)
    const half* q_row = Q_ptr + (q_start + tid) * D;
    #pragma unroll
    for (int d = 0; d < D; d += 2) {
        half2 h2 = *reinterpret_cast<const half2*>(&q_row[d]);
        sQ[tid][d] = h2.x;
        sQ[tid][d+1] = h2.y;
    }

    // online softmax status (使用 float 保证精度)
    float m_i = -FLT_MAX;
    float l_i = 0.f;
    float acc[MAX_D];

    for (int d = 0; d < D; ++d) acc[d] = 0.f;
    
    // iter K/V blocks
    for (int k_start = 0; k_start < N; k_start += TILE_K) {
        int k_len = min(TILE_K, N - k_start);

        // cooperative load K/V (使用 half2 向量化加载)
        int total_elements = k_len * D;
        for (int idx = tid * 2; idx < total_elements; idx += blockDim.x * 2) {
            int row = idx / D;
            int col = idx % D;
            half2 h2_k = *reinterpret_cast<const half2*>(&K_ptr[(k_start + row) * D + col]);
            half2 h2_v = *reinterpret_cast<const half2*>(&V_ptr[(k_start + row) * D + col]);
            sK[row][col] = h2_k.x;
            sK[row][col+1] = h2_k.y;
            sV[row][col] = h2_v.x;
            sV[row][col+1] = h2_v.y;
        }
        __syncthreads();

        // score (使用 half2 向量化计算点积)
        float scores[TILE_K];
        float m_block = -FLT_MAX;

        int q_idx = q_start + tid;

        for (int j = 0; j < k_len; ++j) {
            int k_idx = k_start + j;

            if (causal && k_idx > q_idx) {
                scores[j] = -FLT_MAX;
                continue;
            }

            // 使用 half2 向量化计算点积
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; d += 2) {
                half2 a = make_half2(sQ[tid][d], sQ[tid][d+1]);
                half2 b = make_half2(sK[j][d], sK[j][d+1]);
                float2 p = __half22float2(__hmul2(a, b));
                dot += p.x + p.y;
            }

            dot *= scale;
            scores[j] = dot;
            m_block = fmaxf(m_block, dot);
        }

        // Warp-level max reduction
        m_block = warp_reduce_max(m_block);
        m_block = __shfl_sync(0xffffffff, m_block, 0);

        // all blocks masked
        if (m_block == -FLT_MAX) {
            __syncthreads();
            continue;
        }

        float m_new = fmaxf(m_i, m_block);

        float alpha = (m_i == -FLT_MAX) ? 0.f : expf(m_i - m_new);

        float beta_sum = 0.f;
        float weighted[MAX_D];
        for (int d = 0; d < D; ++d) {
            weighted[d] = 0.f;
        }

        for (int j = 0; j < k_len; ++j) {
            if (scores[j] == -FLT_MAX) continue;

            float p = expf(scores[j] - m_new);
            beta_sum += p;

            // 使用 half2 向量化加权累加
            #pragma unroll
            for (int d = 0; d < D; d += 2) {
                half2 v = make_half2(sV[j][d], sV[j][d+1]);
                float2 vf = __half22float2(v);
                weighted[d] += vf.x * p;
                weighted[d+1] += vf.y * p;
            }
        }

        // Warp-level reduction for beta_sum
        beta_sum = warp_reduce_sum(beta_sum);
        beta_sum = __shfl_sync(0xffffffff, beta_sum, 0);

        // update status
        for (int d = 0; d < D; ++d) {
            acc[d] = alpha * acc[d] + weighted[d];
        }

        l_i = alpha * l_i + beta_sum;
        m_i = m_new;

        __syncthreads();
    }

// write back (使用 half2 向量化存储)
    int q_idx_write = q_start + tid;
    #pragma unroll
    for (int d = 0; d < D; d += 2) {
        half2 out_h2 = make_half2(
            __float2half(acc[d] / l_i),
            __float2half(acc[d+1] / l_i)
        );
        *reinterpret_cast<half2*>(&O_ptr[q_idx_write * D + d]) = out_h2;
    }
}

// Warp-level Reduction 优化版本 + Shared Memory Double Buffering
__global__ void flash_attn_kernel_fp16_warp(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B,
    int H,
    int N,
    int D,
    bool causal,
    float scale) 
{
    int q_tile_idx = blockIdx.x;
    int bh = blockIdx.y;

    int b = bh / H;
    int h = bh % H;

    int tid = threadIdx.x;
    int warp_id = tid / 32;  // 当前 warp ID
    int lane_id = tid % 32;  // 当前 lane ID
    
    int q_start = q_tile_idx * TILE_Q;

    if (q_start >= N) return;
    int q_end = min(TILE_Q, N - q_start);

    // 每个 warp 处理一个 query
    if (warp_id >= q_end) return;

    // base pointer
    const half* Q_ptr = Q + ((b * H + h) * N * D);
    const half* K_ptr = K + ((b * H + h) * N * D);
    const half* V_ptr = V + ((b * H + h) * N * D);
    half* O_ptr = O + ((b * H + h) * N * D);

    // 双缓冲共享内存 (ping-pong buffers)
    // buffer[0] 和 buffer[1] 交替使用
    __shared__ half sQ[TILE_Q][MAX_D];
    __shared__ half sK[2][TILE_K][MAX_D];  // 双缓冲 K
    __shared__ half sV[2][TILE_K][MAX_D];  // 双缓冲 V

    // 当前 warp 处理的 query 行
    int q_idx = q_start + warp_id;

    // load Q (使用 half2 向量化加载)
    const half* q_row = Q_ptr + q_idx * D;
    #pragma unroll
    for (int d = 0; d < D; d += 2) {
        half2 h2 = *reinterpret_cast<const half2*>(&q_row[d]);
        sQ[warp_id][d] = h2.x;
        sQ[warp_id][d+1] = h2.y;
    }

    // online softmax status (使用 float 保证精度)
    float m_i = -FLT_MAX;
    float l_i = 0.f;
    float acc[MAX_D];

    for (int d = 0; d < D; ++d) acc[d] = 0.f;
    
    // 计算总 block 数
    int num_kv_blocks = (N + TILE_K - 1) / TILE_K;
    
    // 预加载第一个 K/V block 到 buffer[0]
    int first_k_start = 0;
    int first_k_len = min(TILE_K, N - first_k_start);
    int total_elements_first = first_k_len * D;
    for (int idx = tid * 2; idx < total_elements_first; idx += blockDim.x * 2) {
        int row = idx / D;
        int col = idx % D;
        half2 h2_k = *reinterpret_cast<const half2*>(&K_ptr[(first_k_start + row) * D + col]);
        half2 h2_v = *reinterpret_cast<const half2*>(&V_ptr[(first_k_start + row) * D + col]);
        sK[0][row][col] = h2_k.x;
        sK[0][row][col+1] = h2_k.y;
        sV[0][row][col] = h2_v.x;
        sV[0][row][col+1] = h2_v.y;
    }
    __syncthreads();
    
    // iter K/V blocks (使用双缓冲)
    for (int block_idx = 0; block_idx < num_kv_blocks; ++block_idx) {
        int k_start = block_idx * TILE_K;
        int k_len = min(TILE_K, N - k_start);
        
        // 当前使用的 buffer (ping-pong)
        int current_buffer = block_idx % 2;
        int next_buffer = 1 - current_buffer;
        
        // 异步预加载下一个 block 到另一个 buffer
        int next_k_start = (block_idx + 1) * TILE_K;
        if (next_k_start < N) {
            int next_k_len = min(TILE_K, N - next_k_start);
            int total_elements_next = next_k_len * D;
            for (int idx = tid * 2; idx < total_elements_next; idx += blockDim.x * 2) {
                int row = idx / D;
                int col = idx % D;
                half2 h2_k = *reinterpret_cast<const half2*>(&K_ptr[(next_k_start + row) * D + col]);
                half2 h2_v = *reinterpret_cast<const half2*>(&V_ptr[(next_k_start + row) * D + col]);
                sK[next_buffer][row][col] = h2_k.x;
                sK[next_buffer][row][col+1] = h2_k.y;
                sV[next_buffer][row][col] = h2_v.x;
                sV[next_buffer][row][col+1] = h2_v.y;
            }
        }
        
        // 使用当前 buffer 进行计算
        // score (使用 warp-level reduction 优化点积计算)
        float scores[TILE_K];
        float m_block = -FLT_MAX;

        for (int j = 0; j < k_len; ++j) {
            int k_idx = k_start + j;

            if (causal && k_idx > q_idx) {
                scores[j] = -FLT_MAX;
                continue;
            }

            // 使用 warp-level reduction 计算点积
            // 每个 lane 处理部分维度
            float dot = 0.0f;
            #pragma unroll
            for (int d = lane_id; d < D; d += 32) {
                dot += __half2float(sQ[warp_id][d]) * __half2float(sK[current_buffer][j][d]);
            }
            
            // Warp-level reduction
            dot = warp_reduce_sum(dot);
            
            // lane 0 广播结果
            dot = __shfl_sync(0xffffffff, dot, 0);

            dot *= scale;
            scores[j] = dot;
            m_block = fmaxf(m_block, dot);
        }

        // Warp-level max reduction
        m_block = warp_reduce_max(m_block);
        m_block = __shfl_sync(0xffffffff, m_block, 0);

        // all blocks masked
        if (m_block == -FLT_MAX) {
            __syncthreads();
            continue;
        }

        float m_new = fmaxf(m_i, m_block);

        float alpha = (m_i == -FLT_MAX) ? 0.f : expf(m_i - m_new);

        float beta_sum = 0.f;
        float weighted[MAX_D];
        for (int d = 0; d < D; ++d) {
            weighted[d] = 0.f;
        }

        for (int j = 0; j < k_len; ++j) {
            if (scores[j] == -FLT_MAX) continue;

            float p = expf(scores[j] - m_new);
            beta_sum += p;

            // 使用 half2 向量化加权累加
            #pragma unroll
            for (int d = 0; d < D; d += 2) {
                half2 v = make_half2(sV[current_buffer][j][d], sV[current_buffer][j][d+1]);
                float2 vf = __half22float2(v);
                weighted[d] += vf.x * p;
                weighted[d+1] += vf.y * p;
            }
        }

        // Warp-level reduction for beta_sum
        beta_sum = warp_reduce_sum(beta_sum);
        beta_sum = __shfl_sync(0xffffffff, beta_sum, 0);

        // update status
        for (int d = 0; d < D; ++d) {
            acc[d] = alpha * acc[d] + weighted[d];
        }

        l_i = alpha * l_i + beta_sum;
        m_i = m_new;

        __syncthreads();
    }

    // write back (使用 half2 向量化存储)
    #pragma unroll
    for (int d = 0; d < D; d += 2) {
        half2 out_h2 = make_half2(
            __float2half(acc[d] / l_i),
            __float2half(acc[d+1] / l_i)
        );
        *reinterpret_cast<half2*>(&O_ptr[q_idx * D + d]) = out_h2;
    }
}

torch::Tensor flash_attn_forward_fp16_warp(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    TORCH_CHECK(q.dtype() == torch::kHalf, "q must be float16");

    auto B = q.size(0);
    auto H = q.size(1);
    auto N = q.size(2);
    auto D = q.size(3);

    TORCH_CHECK(D <= MAX_D, "D must be less than or equal to MAX_D(64)");

    auto out = torch::zeros_like(q);

    // 每个 block 处理一个 (batch, head, q_tile)
    // 每个 warp 处理一个 query
    dim3 qrid((N + TILE_Q - 1) / TILE_Q, B * H);
    dim3 block(TILE_Q);  // 32 threads = 1 warp per query, 最多处理 32 queries

    float scale = 1.0f / std::sqrt((float)D);

    flash_attn_kernel_fp16_warp<<<qrid, block>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        B,
        H,
        N,
        D,
        causal,
        scale
    );
    
    return out;
}

// KV Cache / Decode Kernel 优化版本
// 特征: Q_len = 1, K_len 很长, 使用历史 KV cache
__global__ void flash_attn_kernel_decode_fp16(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B,
    int H,
    int N,      // 当前 K/V 的长度（包含 cache）
    int D,
    bool causal,
    float scale,
    int cache_len  // 历史 cache 长度（不包含当前 token）
) {
    // 每个 block 处理一个 (batch, head)
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;

    int tid = threadIdx.x;
    int lane_id = tid % 32;

    // base pointer
    const half* Q_ptr = Q + ((b * H + h) * 1 * D);  // Q: [B, H, 1, D]
    const half* K_ptr = K + ((b * H + h) * N * D);  // K: [B, H, N, D]
    const half* V_ptr = V + ((b * H + h) * N * D);  // V: [B, H, N, D]
    half* O_ptr = O + ((b * H + h) * 1 * D);        // O: [B, H, 1, D]

    // 在线 softmax 状态 (所有线程共享)
    float m_i = -FLT_MAX;
    float l_i = 0.f;
    float acc[MAX_D];

    for (int d = 0; d < D; ++d) acc[d] = 0.f;

    // 加载 Q (只有一个 token) - 所有线程都加载完整的 Q
    __shared__ half sQ[MAX_D];
    if (tid < D) {
        sQ[tid] = Q_ptr[tid];
    }
    __syncthreads();

    // 分块处理 K/V (tile size 可以更大，因为 Q 只有一个 token)
    constexpr int TILE_K_DECODE = 64;  // decode 模式可以用更大的 tile
    int num_kv_blocks = (N + TILE_K_DECODE - 1) / TILE_K_DECODE;

    __shared__ half sK[TILE_K_DECODE][MAX_D];
    __shared__ half sV[TILE_K_DECODE][MAX_D];

    for (int block_idx = 0; block_idx < num_kv_blocks; ++block_idx) {
        int k_start = block_idx * TILE_K_DECODE;
        int k_len = min(TILE_K_DECODE, N - k_start);

        // 协作加载 K/V
        int total_elements = k_len * D;
        for (int idx = tid; idx < total_elements; idx += blockDim.x) {
            int row = idx / D;
            int col = idx % D;
            sK[row][col] = K_ptr[(k_start + row) * D + col];
            sV[row][col] = V_ptr[(k_start + row) * D + col];
        }
        __syncthreads();

        // 计算 attention scores (Q 只有一个 token，所以只需要计算一行)
        float scores[TILE_K_DECODE];
        float m_block = -FLT_MAX;

        for (int j = 0; j < k_len; ++j) {
            int k_idx = k_start + j;

            // causal mask: 在 decode 模式下，Q 的位置是 cache_len
            if (causal && k_idx > cache_len) {
                scores[j] = -FLT_MAX;
                continue;
            }

            // 计算点积 (使用 warp-level reduction)
            float dot = 0.0f;
            #pragma unroll
            for (int d = lane_id; d < D; d += 32) {
                dot += __half2float(sQ[d]) * __half2float(sK[j][d]);
            }
            
            // Warp-level reduction
            dot = warp_reduce_sum(dot);
            
            // lane 0 有正确的点积结果
            if (lane_id == 0) {
                dot *= scale;
                scores[j] = dot;
            }
            
            // 广播到所有线程
            scores[j] = __shfl_sync(0xffffffff, scores[j], 0);
            m_block = fmaxf(m_block, scores[j]);
        }

        // Warp-level max reduction
        m_block = warp_reduce_max(m_block);
        m_block = __shfl_sync(0xffffffff, m_block, 0);

        // 所有位置都被 mask
        if (m_block == -FLT_MAX) {
            __syncthreads();
            continue;
        }

        float m_new = fmaxf(m_i, m_block);
        float alpha = (m_i == -FLT_MAX) ? 0.f : expf(m_i - m_new);

        // 计算加权和 (只有 lane 0 执行)
        if (lane_id == 0) {
            float beta_sum = 0.f;
            float weighted[MAX_D];
            for (int d = 0; d < D; ++d) {
                weighted[d] = 0.f;
            }

            for (int j = 0; j < k_len; ++j) {
                if (scores[j] == -FLT_MAX) continue;

                float p = expf(scores[j] - m_new);
                beta_sum += p;

                // 加权累加 V
                #pragma unroll
                for (int d = 0; d < D; ++d) {
                    weighted[d] += __half2float(sV[j][d]) * p;
                }
            }

            // 更新状态
            for (int d = 0; d < D; ++d) {
                acc[d] = alpha * acc[d] + weighted[d];
            }

            l_i = alpha * l_i + beta_sum;
            m_i = m_new;
        }
        
        __syncthreads();
    }

    // 写回结果 (只有 lane 0 执行)
    if (tid == 0) {
        for (int d = 0; d < D; ++d) {
            O_ptr[d] = __float2half(acc[d] / l_i);
        }
    }
}

// Decode 模式的 FP32 版本
torch::Tensor flash_attn_forward_decode_fp16(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    int cache_len
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    TORCH_CHECK(q.dtype() == torch::kHalf, "q must be float16");
    TORCH_CHECK(q.size(2) == 1, "q sequence length must be 1 for decode mode");

    auto B = q.size(0);
    auto H = q.size(1);
    auto N = k.size(2);  // K/V 的长度（包含 cache）
    auto D = q.size(3);

    TORCH_CHECK(D <= MAX_D, "D must be less than or equal to MAX_D(128)");
    TORCH_CHECK(cache_len < N, "cache_len must be less than N");

    auto out = torch::zeros({B, H, 1, D}, q.options());

    // 每个 block 处理一个 (batch, head)
    // 使用 128 或 256 个线程（根据 D 的大小调整）
    int threads = 128;
    dim3 grid(B * H);
    dim3 block(threads);

    float scale = 1.0f / std::sqrt((float)D);

    flash_attn_kernel_decode_fp16<<<grid, block>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        B,
        H,
        N,
        D,
        causal,
        scale,
        cache_len
    );
    
    return out;
}

torch::Tensor flash_attn_forward_fp16(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    TORCH_CHECK(q.dtype() == torch::kHalf, "q must be float16");

    auto B = q.size(0);
    auto H = q.size(1);
    auto N = q.size(2);
    auto D = q.size(3);

    TORCH_CHECK(D <= MAX_D, "D must be less than or equal to MAX_D(64)");

    auto out = torch::zeros_like(q);

    dim3 qrid((N + TILE_Q - 1) / TILE_Q, B * H);
    dim3 block(TILE_Q);

    float scale = 1.0f / std::sqrt((float)D);

    flash_attn_kernel_fp16<<<qrid, block>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        B,
        H,
        N,
        D,
        causal,
        scale
    );
    
    return out;
} 
