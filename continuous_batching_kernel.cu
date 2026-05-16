// Continuous Batching 优化版本
// 特征: 支持不同序列长度的 batch，每个序列有自己的 cache_len
// 适用于 LLM serving 场景，提高 GPU 利用率

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

constexpr int MAX_D_CB = 128;

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

__global__ void flash_attn_kernel_continuous_batching_fp16(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int* __restrict__ cache_lens,  // [B] 每个序列的实际 cache 长度
    int B,
    int H_Q,
    int H_KV,
    int max_N,    // max sequence length in batch
    int D,
    bool causal,
    float scale
) {
    // 每个 block 处理一个 (batch, q_head)
    int bqh = blockIdx.x;
    int b = bqh / H_Q;
    int h_q = bqh % H_Q;
    
    // GQA 映射
    int h_kv = h_q / (H_Q / H_KV);

    int tid = threadIdx.x;
    int lane_id = tid % 32;

    // 获取当前 batch 的实际长度
    int current_cache_len = cache_lens[b];
    int current_N = current_cache_len + 1;  // 包含当前 token

    if (current_N > max_N) current_N = max_N;

    // base pointer
    // Q: [B, H_Q, 1, D]
    const half* Q_ptr = Q + ((b * H_Q + h_q) * 1 * D);
    // K/V: [B, H_KV, max_N, D] - 使用 max_N 对齐
    const half* K_ptr = K + ((b * H_KV + h_kv) * max_N * D);
    const half* V_ptr = V + ((b * H_KV + h_kv) * max_N * D);
    // O: [B, H_Q, 1, D]
    half* O_ptr = O + ((b * H_Q + h_q) * 1 * D);

    // 在线 softmax 状态
    float m_i = -FLT_MAX;
    float l_i = 0.f;
    float acc[MAX_D_CB];

    for (int d = 0; d < D; ++d) acc[d] = 0.f;

    // 加载 Q
    __shared__ half sQ[MAX_D_CB];
    if (tid < D) {
        sQ[tid] = Q_ptr[tid];
    }
    __syncthreads();

    // 分块处理 K/V
    constexpr int TILE_K_DECODE = 64;
    int num_kv_blocks = (current_N + TILE_K_DECODE - 1) / TILE_K_DECODE;

    __shared__ half sK[TILE_K_DECODE][MAX_D_CB];
    __shared__ half sV[TILE_K_DECODE][MAX_D_CB];

    for (int block_idx = 0; block_idx < num_kv_blocks; ++block_idx) {
        int k_start = block_idx * TILE_K_DECODE;
        int k_len = min(TILE_K_DECODE, current_N - k_start);

        // 协作加载 K/V
        int total_elements = k_len * D;
        for (int idx = tid; idx < total_elements; idx += blockDim.x) {
            int row = idx / D;
            int col = idx % D;
            sK[row][col] = K_ptr[(k_start + row) * D + col];
            sV[row][col] = V_ptr[(k_start + row) * D + col];
        }
        __syncthreads();

        // 计算 attention scores
        float scores[TILE_K_DECODE];
        float m_block = -FLT_MAX;

        for (int j = 0; j < k_len; ++j) {
            int k_idx = k_start + j;

            // causal mask: Q 的位置是 current_cache_len
            if (causal && k_idx > current_cache_len) {
                scores[j] = -FLT_MAX;
                continue;
            }

            // 计算点积
            float dot = 0.0f;
            #pragma unroll
            for (int d = lane_id; d < D; d += 32) {
                dot += __half2float(sQ[d]) * __half2float(sK[j][d]);
            }
            
            // Warp-level reduction
            dot = warp_reduce_sum(dot);
            
            if (lane_id == 0) {
                dot *= scale;
                scores[j] = dot;
            }
            
            scores[j] = __shfl_sync(0xffffffff, scores[j], 0);
            m_block = fmaxf(m_block, scores[j]);
        }

        // Warp-level max reduction
        m_block = warp_reduce_max(m_block);
        m_block = __shfl_sync(0xffffffff, m_block, 0);

        if (m_block == -FLT_MAX) {
            __syncthreads();
            continue;
        }

        float m_new = fmaxf(m_i, m_block);
        float alpha = (m_i == -FLT_MAX) ? 0.f : expf(m_i - m_new);

        // 计算加权和 (只有 lane 0 执行)
        if (lane_id == 0) {
            float beta_sum = 0.f;
            float weighted[MAX_D_CB];
            for (int d = 0; d < D; ++d) {
                weighted[d] = 0.f;
            }

            for (int j = 0; j < k_len; ++j) {
                if (scores[j] == -FLT_MAX) continue;

                float p = expf(scores[j] - m_new);
                beta_sum += p;

                #pragma unroll
                for (int d = 0; d < D; ++d) {
                    weighted[d] += __half2float(sV[j][d]) * p;
                }
            }

            for (int d = 0; d < D; ++d) {
                acc[d] = alpha * acc[d] + weighted[d];
            }

            l_i = alpha * l_i + beta_sum;
            m_i = m_new;
        }
        
        __syncthreads();
    }

    // 写回结果
    if (tid == 0) {
        for (int d = 0; d < D; ++d) {
            O_ptr[d] = __float2half(acc[d] / l_i);
        }
    }
}

// Continuous Batching 接口
torch::Tensor flash_attn_forward_continuous_batching_fp16(
    torch::Tensor q,          // [B, H_Q, 1, D]
    torch::Tensor k,          // [B, H_KV, max_N, D]
    torch::Tensor v,          // [B, H_KV, max_N, D]
    torch::Tensor cache_lens, // [B] 每个序列的 cache 长度
    bool causal
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(cache_lens);

    TORCH_CHECK(q.dtype() == torch::kHalf, "q must be float16");
    TORCH_CHECK(q.size(2) == 1, "q sequence length must be 1 for decode mode");
    TORCH_CHECK(cache_lens.dim() == 1, "cache_lens must be 1D tensor");
    TORCH_CHECK(cache_lens.size(0) == q.size(0), "cache_lens size must match batch size");
    
    auto B = q.size(0);
    auto H_Q = q.size(1);
    auto H_KV = k.size(1);
    auto max_N = k.size(2);
    auto D = q.size(3);

    TORCH_CHECK(H_Q >= H_KV, "H_Q must be >= H_KV");
    TORCH_CHECK(H_Q % H_KV == 0, "H_Q must be divisible by H_KV");
    TORCH_CHECK(D <= MAX_D_CB, "D must be <= MAX_D_CB(128)");

    auto out = torch::zeros({B, H_Q, 1, D}, q.options());

    // 每个 block 处理一个 (batch, q_head)
    int threads = 128;
    dim3 grid(B * H_Q);
    dim3 block(threads);

    float scale = 1.0f / std::sqrt((float)D);

    flash_attn_kernel_continuous_batching_fp16<<<grid, block>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        cache_lens.data_ptr<int>(),
        B,
        H_Q,
        H_KV,
        max_N,
        D,
        causal,
        scale
    );
    
    return out;
}
