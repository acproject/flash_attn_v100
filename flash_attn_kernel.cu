#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cmath>
#include <cfloat>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 配置参数
constexpr int TILE_Q = 32;
constexpr int TILE_K = 32;
constexpr int MAX_D = 128;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int QUERIES_PER_BLOCK = WARPS_PER_BLOCK;
constexpr int WARP_BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;

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
    bool is_active = tid < q_end;

    // load Q
    if (is_active) {
        for (int d = 0; d < D; ++d) {
            sQ[tid][d] = Q_ptr[(q_start + tid) * D + d];
        }
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

        if (is_active) {
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
        if (m_block != -FLT_MAX) {
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
        }
        }

        __syncthreads();
    }

    // reback write
    if (is_active) {
    for (int d = 0; d < D; ++d) {
        O_ptr[(q_start + tid) * D + d] = acc[d] / l_i;
    }
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

__global__ void allreduce_sum_kernel(
    half* __restrict__ output,
    const half* __restrict__ input,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        output[idx] = __float2half(__half2float(output[idx]) + __half2float(input[idx]));
    }
}

void multi_gpu_allreduce_fp16(
    torch::Tensor tensor,
    int src_device,
    int dst_device
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(tensor.dtype() == torch::kHalf, "tensor must be float16");

    auto total_elements = tensor.numel();

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    cudaSetDevice(dst_device);

    allreduce_sum_kernel<<<blocks, threads>>>(
        reinterpret_cast<half*>(tensor.data_ptr<at::Half>()),
        reinterpret_cast<half*>(tensor.data_ptr<at::Half>()),
        total_elements
    );
}

__global__ void cross_gpu_reduce_kernel(
    half* __restrict__ dst,
    const half* __restrict__ src,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        dst[idx] = __float2half(__half2float(dst[idx]) + __half2float(src[idx]));
    }
}

void cross_gpu_allreduce_fp16(
    torch::Tensor dst_tensor,
    torch::Tensor src_tensor
) {
    CHECK_INPUT(dst_tensor);
    CHECK_INPUT(src_tensor);
    TORCH_CHECK(dst_tensor.dtype() == torch::kHalf, "dst must be float16");
    TORCH_CHECK(src_tensor.dtype() == torch::kHalf, "src must be float16");
    TORCH_CHECK(dst_tensor.numel() == src_tensor.numel(), "size mismatch");

    auto total_elements = dst_tensor.numel();

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    cross_gpu_reduce_kernel<<<blocks, threads>>>(
        reinterpret_cast<half*>(dst_tensor.data_ptr<at::Half>()),
        reinterpret_cast<half*>(src_tensor.data_ptr<at::Half>()),
        total_elements
    );
}

__global__ void scatter_heads_kernel(
    const half* __restrict__ full_tensor,
    half* __restrict__ local_tensor,
    int total_heads,
    int local_head_start,
    int local_head_count,
    int B,
    int N,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elements_per_head = N * D;
    int local_elements = B * local_head_count * elements_per_head;

    if (idx >= local_elements) return;

    int b = idx / (local_head_count * elements_per_head);
    int remainder1 = idx % (local_head_count * elements_per_head);
    int local_h = remainder1 / elements_per_head;
    int remainder2 = remainder1 % elements_per_head;
    int global_h = local_h + local_head_start;

    if (global_h < total_heads) {
        int full_idx = b * total_heads * elements_per_head + global_h * elements_per_head + remainder2;
        local_tensor[idx] = full_tensor[full_idx];
    }
}

__global__ void gather_heads_kernel(
    half* __restrict__ full_tensor,
    const half* __restrict__ local_tensor,
    int total_heads,
    int local_head_start,
    int local_head_count,
    int B,
    int N,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elements_per_head = N * D;
    int local_elements = B * local_head_count * elements_per_head;

    if (idx >= local_elements) return;

    int b = idx / (local_head_count * elements_per_head);
    int remainder1 = idx % (local_head_count * elements_per_head);
    int local_h = remainder1 / elements_per_head;
    int remainder2 = remainder1 % elements_per_head;
    int global_h = local_h + local_head_start;

    if (global_h < total_heads) {
        int full_idx = b * total_heads * elements_per_head + global_h * elements_per_head + remainder2;
        full_tensor[full_idx] = local_tensor[idx];
    }
}

void scatter_heads_forward(
    torch::Tensor full_tensor,
    torch::Tensor local_tensor,
    int total_heads,
    int local_head_start,
    int local_head_count
) {
    CHECK_INPUT(full_tensor);
    CHECK_INPUT(local_tensor);
    TORCH_CHECK(full_tensor.dtype() == torch::kHalf, "full_tensor must be float16");
    TORCH_CHECK(local_tensor.dtype() == torch::kHalf, "local_tensor must be float16");

    auto B = full_tensor.size(0);
    auto N = full_tensor.size(2);
    auto D = full_tensor.size(3);
    auto local_elements = local_tensor.numel();

    int threads = 256;
    int blocks = (local_elements + threads - 1) / threads;

    scatter_heads_kernel<<<blocks, threads>>>(
        reinterpret_cast<half*>(full_tensor.data_ptr<at::Half>()),
        reinterpret_cast<half*>(local_tensor.data_ptr<at::Half>()),
        total_heads,
        local_head_start,
        local_head_count,
        B,
        N,
        D
    );
}

void gather_heads_forward(
    torch::Tensor full_tensor,
    torch::Tensor local_tensor,
    int total_heads,
    int local_head_start,
    int local_head_count
) {
    CHECK_INPUT(full_tensor);
    CHECK_INPUT(local_tensor);
    TORCH_CHECK(full_tensor.dtype() == torch::kHalf, "full_tensor must be float16");
    TORCH_CHECK(local_tensor.dtype() == torch::kHalf, "local_tensor must be float16");

    auto B = full_tensor.size(0);
    auto N = full_tensor.size(2);
    auto D = full_tensor.size(3);
    auto local_elements = local_tensor.numel();

    int threads = 256;
    int blocks = (local_elements + threads - 1) / threads;

    gather_heads_kernel<<<blocks, threads>>>(
        reinterpret_cast<half*>(full_tensor.data_ptr<at::Half>()),
        reinterpret_cast<half*>(local_tensor.data_ptr<at::Half>()),
        total_heads,
        local_head_start,
        local_head_count,
        B,
        N,
        D
    );
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

// WMMA QK^T 计算：一个 warp 处理 16x16 score tile
// 计算 Q[16,D] @ K[16,D]^T = scores[16,16]
__device__ void wmma_compute_qk(
    const half* Q_16xD,      // Q 的 16xD 块 (row-major)
    const half* K_16xD,      // K 的 16xD 块 (row-major)
    float scores[16][16],    // 输出 scores
    int D,
    float scale,
    int q_global_row,        // Q 的起始全局行号
    int k_global_row,        // K 的起始全局行号
    bool causal
) {
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // 初始化累加器为 0
    wmma::fill_fragment(c_frag, 0.0f);
    
    // 沿着 D 维度迭代 (D 必须是 16 的倍数)
    for (int d = 0; d < D; d += 16) {
        // 加载 Q[16, 16] (row-major)
        wmma::load_matrix_sync(a_frag, Q_16xD + d, D);
        
        // 加载 K[16, 16] (col_major)
        // col_major 会将 K[16,16] 视为 K^T[16,16]
        // 因此 WMMA 计算的是 Q @ K^T ✓
        wmma::load_matrix_sync(b_frag, K_16xD + d, D);
        
        // 执行矩阵乘法: C = A * B + C
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 存储到 shared memory (每个 warp 使用不同的 offset)
    __shared__ float temp_buffers[4][256];  // 4 warps
    int warp_id = threadIdx.x / 32;
    float* temp_buffer = temp_buffers[warp_id];
    
    wmma::store_matrix_sync(temp_buffer, c_frag, 16, wmma::mem_row_major);
    
    // 应用 scale 和 causal mask
    // 每个 thread 处理部分元素
    int tid_in_warp = threadIdx.x % 32;
    for (int idx = tid_in_warp; idx < 256; idx += 32) {
        int i = idx / 16;
        int j = idx % 16;
        int qi = q_global_row + i;
        int kj = k_global_row + j;
        float val = temp_buffer[idx] * scale;
        
        // 应用 causal mask
        if (causal && kj > qi) {
            scores[i][j] = -FLT_MAX;
        } else {
            scores[i][j] = val;
        }
    }
    
    // 等待所有 thread 完成写入 scores
    __syncwarp();
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
    bool is_active = tid < q_end;

    // load Q (使用 half2 向量化加载)
    if (is_active) {
    const half* q_row = Q_ptr + (q_start + tid) * D;
    #pragma unroll
    for (int d = 0; d < D; d += 2) {
        half2 h2 = *reinterpret_cast<const half2*>(&q_row[d]);
        sQ[tid][d] = h2.x;
        sQ[tid][d+1] = h2.y;
    }
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

        if (is_active) {
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
                half2 b2 = make_half2(sK[j][d], sK[j][d+1]);
                float2 p = __half22float2(__hmul2(a, b2));
                dot += p.x + p.y;
            }

            dot *= scale;
            scores[j] = dot;
            m_block = fmaxf(m_block, dot);
        }

        // all blocks masked
        if (m_block != -FLT_MAX) {
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

        // update status
        for (int d = 0; d < D; ++d) {
            acc[d] = alpha * acc[d] + weighted[d];
        }

        l_i = alpha * l_i + beta_sum;
        m_i = m_new;
        }
        }

        __syncthreads();
    }

// write back (使用 half2 向量化存储)
    if (is_active) {
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
}

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
    int q_start = q_tile_idx * TILE_Q;

    if (q_start >= N) return;
    int q_end = min(TILE_Q, N - q_start);

    const half* Q_ptr = Q + ((b * H + h) * N * D);
    const half* K_ptr = K + ((b * H + h) * N * D);
    const half* V_ptr = V + ((b * H + h) * N * D);
    half* O_ptr = O + ((b * H + h) * N * D);

    __shared__ half sQ[TILE_Q][MAX_D];
    __shared__ half sK[TILE_K][MAX_D];
    __shared__ half sV[TILE_K][MAX_D];

    bool is_active = tid < q_end;

    if (is_active) {
        const half* q_row = Q_ptr + (q_start + tid) * D;
        #pragma unroll
        for (int d = 0; d < D; d += 2) {
            half2 h2 = *reinterpret_cast<const half2*>(&q_row[d]);
            sQ[tid][d] = h2.x;
            sQ[tid][d+1] = h2.y;
        }
    }

    float m_i = -FLT_MAX;
    float l_i = 0.f;
    float acc[MAX_D];
    for (int d = 0; d < D; ++d) acc[d] = 0.f;

    __syncthreads();

    for (int k_start = 0; k_start < N; k_start += TILE_K) {
        int k_len = min(TILE_K, N - k_start);

        int total_elements = k_len * D;
        for (int idx = tid * 2; idx < total_elements; idx += TILE_Q * 2) {
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

        if (is_active) {
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
                #pragma unroll
                for (int d = 0; d < D; d += 2) {
                    half2 a = make_half2(sQ[tid][d], sQ[tid][d+1]);
                    half2 b2 = make_half2(sK[j][d], sK[j][d+1]);
                    float2 p = __half22float2(__hmul2(a, b2));
                    dot += p.x + p.y;
                }
                dot *= scale;
                scores[j] = dot;
                m_block = fmaxf(m_block, dot);
            }

            if (m_block != -FLT_MAX) {
                float m_new = fmaxf(m_i, m_block);
                float alpha = (m_i == -FLT_MAX) ? 0.f : expf(m_i - m_new);

                float beta_sum = 0.f;
                float weighted[MAX_D];
                for (int d = 0; d < D; ++d) weighted[d] = 0.f;

                for (int j = 0; j < k_len; ++j) {
                    if (scores[j] == -FLT_MAX) continue;
                    float p = expf(scores[j] - m_new);
                    beta_sum += p;
                    #pragma unroll
                    for (int d = 0; d < D; d += 2) {
                        half2 v = make_half2(sV[j][d], sV[j][d+1]);
                        float2 vf = __half22float2(v);
                        weighted[d] += vf.x * p;
                        weighted[d+1] += vf.y * p;
                    }
                }

                for (int d = 0; d < D; ++d) {
                    acc[d] = alpha * acc[d] + weighted[d];
                }
                l_i = alpha * l_i + beta_sum;
                m_i = m_new;
            }
        }

        __syncthreads();
    }

    if (is_active) {
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

    dim3 qrid((N + TILE_Q - 1) / TILE_Q, B * H);
    dim3 block(TILE_Q);

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
// GQA/MQA Decode Kernel 优化版本
// 特征: 支持 Grouped-Query Attention 和 Multi-Query Attention
// GQA: H_Q > H_KV, 多个 Q heads 共享一个 KV head
// MQA: H_Q > 1, H_KV = 1, 所有 Q heads 共享同一个 KV head
__global__ void flash_attn_kernel_decode_gqa_fp16(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B,
    int H_Q,      // Q 的 head 数量
    int H_KV,     // K/V 的 head 数量 (H_KV <= H_Q)
    int N,        // 当前 K/V 的长度（包含 cache）
    int D,
    bool causal,
    float scale,
    int cache_len  // 历史 cache 长度（不包含当前 token）
) {
    // 每个 block 处理一个 (batch, q_head)
    int bqh = blockIdx.x;
    int b = bqh / H_Q;
    int h_q = bqh % H_Q;
    
    // GQA 映射: 计算对应的 KV head
    // 每个 KV head 被 (H_Q / H_KV) 个 Q heads 共享
    int h_kv = h_q / (H_Q / H_KV);

    int tid = threadIdx.x;
    int lane_id = tid % 32;

    // base pointer
    // Q: [B, H_Q, 1, D]
    const half* Q_ptr = Q + ((b * H_Q + h_q) * 1 * D);
    // K/V: [B, H_KV, N, D] - 注意是 H_KV 不是 H_Q
    const half* K_ptr = K + ((b * H_KV + h_kv) * N * D);
    const half* V_ptr = V + ((b * H_KV + h_kv) * N * D);
    // O: [B, H_Q, 1, D]
    half* O_ptr = O + ((b * H_Q + h_q) * 1 * D);

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

// GQA/MQA Decode 模式的 FP32 版本
torch::Tensor flash_attn_forward_decode_gqa_fp16(
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
    
    // GQA 验证
    auto H_Q = q.size(1);
    auto H_KV = k.size(1);
    TORCH_CHECK(H_KV == v.size(1), "k and v must have same number of heads");
    TORCH_CHECK(H_Q >= H_KV, "H_Q must be >= H_KV");
    TORCH_CHECK(H_Q % H_KV == 0, "H_Q must be divisible by H_KV");

    auto B = q.size(0);
    auto N = k.size(2);  // K/V 的长度（包含 cache）
    auto D = q.size(3);

    TORCH_CHECK(D <= MAX_D, "D must be less than or equal to MAX_D(128)");
    TORCH_CHECK(cache_len < N, "cache_len must be less than N");

    auto out = torch::zeros({B, H_Q, 1, D}, q.options());

    // 每个 block 处理一个 (batch, q_head)
    // 使用 128 或 256 个线程（根据 D 的大小调整）
    int threads = 128;
    dim3 grid(B * H_Q);
    dim3 block(threads);

    float scale = 1.0f / std::sqrt((float)D);

    flash_attn_kernel_decode_gqa_fp16<<<grid, block>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        B,
        H_Q,
        H_KV,
        N,
        D,
        causal,
        scale,
        cache_len
    );
    
    return out;
}

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

constexpr int WMMA_Q = 32;
constexpr int WMMA_KV = 32;
constexpr int WMMA_WARPS = 2;
constexpr int WMMA_THREADS = WMMA_WARPS * WARP_SIZE;

__global__ void flash_attn_kernel_fp16_wmma(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
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
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int q_start = q_tile_idx * WMMA_Q;
    if (q_start >= N) return;

    int q_len = min(WMMA_Q, N - q_start);

    const half* Q_ptr = Q + ((b * H + h) * N * D);
    const half* K_ptr = K + ((b * H + h) * N * D);
    const half* V_ptr = V + ((b * H + h) * N * D);
    half* O_ptr = O + ((b * H + h) * N * D);

    __shared__ half sQ[WMMA_Q][MAX_D];
    __shared__ half sK[WMMA_KV][MAX_D];
    __shared__ half sV[WMMA_KV][MAX_D];
    __shared__ float sScores[WMMA_Q][WMMA_KV];

    for (int idx = tid; idx < q_len * D; idx += WMMA_THREADS) {
        int row = idx / D;
        int col = idx % D;
        sQ[row][col] = Q_ptr[(q_start + row) * D + col];
    }
    __syncthreads();

    int my_q_base = warp_id * 16;
    int my_q_idx = my_q_base + lane_id;
    bool is_active = my_q_idx < q_len;

    float m_i = -FLT_MAX;
    float l_i = 0.f;
    float acc[MAX_D];
    for (int d = 0; d < D; ++d) acc[d] = 0.f;

    for (int k_start = 0; k_start < N; k_start += WMMA_KV) {
        int k_len = min(WMMA_KV, N - k_start);

        for (int idx = tid; idx < k_len * D; idx += WMMA_THREADS) {
            int row = idx / D;
            int col = idx % D;
            sK[row][col] = K_ptr[(k_start + row) * D + col];
            sV[row][col] = V_ptr[(k_start + row) * D + col];
        }
        __syncthreads();

        {
            for (int k_block = 0; k_block < 2; ++k_block) {
                int k_block_start = k_block * 16;
                int k_block_len = min(16, k_len - k_block_start);
                if (k_block_len <= 0) continue;

                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

                wmma::fill_fragment(c_frag, 0.0f);

                for (int d = 0; d < D; d += 16) {
                    wmma::load_matrix_sync(a_frag, &sQ[my_q_base][d], MAX_D);
                    wmma::load_matrix_sync(b_frag, &sK[k_block_start][d], MAX_D);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }

                wmma::store_matrix_sync(&sScores[my_q_base][k_block_start], c_frag, WMMA_KV, wmma::mem_row_major);
            }
        }

        for (int idx = lane_id; idx < 16 * k_len; idx += WARP_SIZE) {
            int qi = my_q_base + idx / k_len;
            int kj = idx % k_len;
            if (qi >= q_len) continue;
            int q_global = q_start + qi;
            int k_global = k_start + kj;
            float val = sScores[qi][kj] * scale;
            sScores[qi][kj] = (causal && k_global > q_global) ? -FLT_MAX : val;
        }

        __syncthreads();

        if (is_active) {
            float scores[WMMA_KV];
            float m_block = -FLT_MAX;

            int q_idx = q_start + my_q_idx;

            for (int j = 0; j < k_len; ++j) {
                scores[j] = sScores[my_q_idx][j];
                m_block = fmaxf(m_block, scores[j]);
            }

            if (m_block != -FLT_MAX) {
                float m_new = fmaxf(m_i, m_block);
                float alpha = (m_i == -FLT_MAX) ? 0.f : expf(m_i - m_new);

                float beta_sum = 0.f;
                float weighted[MAX_D];
                for (int d = 0; d < D; ++d) weighted[d] = 0.f;

                for (int j = 0; j < k_len; ++j) {
                    if (scores[j] == -FLT_MAX) continue;
                    float p = expf(scores[j] - m_new);
                    beta_sum += p;
                    #pragma unroll
                    for (int d = 0; d < D; d += 2) {
                        half2 v = make_half2(sV[j][d], sV[j][d+1]);
                        float2 vf = __half22float2(v);
                        weighted[d] += vf.x * p;
                        weighted[d+1] += vf.y * p;
                    }
                }

                for (int d = 0; d < D; ++d) {
                    acc[d] = alpha * acc[d] + weighted[d];
                }
                l_i = alpha * l_i + beta_sum;
                m_i = m_new;
            }
        }

        __syncthreads();
    }

    if (is_active) {
        int q_idx_write = q_start + my_q_idx;
        #pragma unroll
        for (int d = 0; d < D; d += 2) {
            half2 out_h2 = make_half2(
                __float2half(acc[d] / l_i),
                __float2half(acc[d+1] / l_i)
            );
            *reinterpret_cast<half2*>(&O_ptr[q_idx_write * D + d]) = out_h2;
        }
    }
}

torch::Tensor flash_attn_forward_fp16_wmma(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    TORCH_CHECK(q.dtype() == torch::kHalf, "q must be float16");
    TORCH_CHECK(q.size(3) % 16 == 0, "D must be multiple of 16 for WMMA");

    auto B = q.size(0);
    auto H = q.size(1);
    auto N = q.size(2);
    auto D = q.size(3);

    TORCH_CHECK(D <= MAX_D, "D must be less than or equal to MAX_D(128)");

    auto out = torch::zeros_like(q);

    dim3 qrid((N + WMMA_Q - 1) / WMMA_Q, B * H);
    dim3 block(WMMA_THREADS);

    float scale = 1.0f / std::sqrt((float)D);

    flash_attn_kernel_fp16_wmma<<<qrid, block>>>(
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


__global__ void flash_attn_kernel_prefill_gqa_fp16(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B,
    int H_Q,
    int H_KV,
    int N,
    int D,
    bool causal,
    float scale
) {
    int q_tile_idx = blockIdx.x;
    int bhq = blockIdx.y;

    int b = bhq / H_Q;
    int h_q = bhq % H_Q;
    int h_kv = h_q / (H_Q / H_KV);

    int tid = threadIdx.x;
    int q_start = q_tile_idx * TILE_Q;
    int q_end_count = min(TILE_Q, N - q_start);

    bool is_active = (tid < q_end_count) && (q_start < N);

    const half* Q_ptr = Q + ((b * H_Q + h_q) * N * D);
    const half* K_ptr = K + ((b * H_KV + h_kv) * N * D);
    const half* V_ptr = V + ((b * H_KV + h_kv) * N * D);
    half* O_ptr = O + ((b * H_Q + h_q) * N * D);

    __shared__ half sQ[TILE_Q][MAX_D];
    __shared__ half sK[TILE_K][MAX_D];
    __shared__ half sV[TILE_K][MAX_D];

    if (is_active) {
        const half* q_row = Q_ptr + (q_start + tid) * D;
        #pragma unroll
        for (int d = 0; d < D; d += 2) {
            half2 h2 = *reinterpret_cast<const half2*>(&q_row[d]);
            sQ[tid][d] = h2.x;
            sQ[tid][d+1] = h2.y;
        }
    }

    float m_i = -FLT_MAX;
    float l_i = 0.f;
    float acc[MAX_D];
    for (int d = 0; d < D; ++d) acc[d] = 0.f;

    for (int k_start = 0; k_start < N; k_start += TILE_K) {
        int k_len = min(TILE_K, N - k_start);

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

        if (is_active) {
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
                #pragma unroll
                for (int d = 0; d < D; d += 2) {
                    half2 a = make_half2(sQ[tid][d], sQ[tid][d+1]);
                    half2 b2 = make_half2(sK[j][d], sK[j][d+1]);
                    float2 p = __half22float2(__hmul2(a, b2));
                    dot += p.x + p.y;
                }
                dot *= scale;
                scores[j] = dot;
                m_block = fmaxf(m_block, dot);
            }

            if (m_block != -FLT_MAX) {
                float m_new = fmaxf(m_i, m_block);
                float alpha = (m_i == -FLT_MAX) ? 0.f : expf(m_i - m_new);

                float beta_sum = 0.f;
                float weighted[MAX_D];
                for (int d = 0; d < D; ++d) weighted[d] = 0.f;

                for (int j = 0; j < k_len; ++j) {
                    if (scores[j] == -FLT_MAX) continue;
                    float p = expf(scores[j] - m_new);
                    beta_sum += p;
                    #pragma unroll
                    for (int d = 0; d < D; d += 2) {
                        half2 v2 = make_half2(sV[j][d], sV[j][d+1]);
                        float2 vf2 = __half22float2(v2);
                        weighted[d] += vf2.x * p;
                        weighted[d+1] += vf2.y * p;
                    }
                }

                for (int d = 0; d < D; ++d) {
                    acc[d] = alpha * acc[d] + weighted[d];
                }
                l_i = alpha * l_i + beta_sum;
                m_i = m_new;
            }
        }

        __syncthreads();
    }

    if (is_active) {
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
}


__global__ void flash_attn_kernel_prefill_gqa_fp16_warp(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B,
    int H_Q,
    int H_KV,
    int N,
    int D,
    bool causal,
    float scale
) {
    int q_tile_idx = blockIdx.x;
    int bhq = blockIdx.y;

    int b = bhq / H_Q;
    int h_q = bhq % H_Q;
    int h_kv = h_q / (H_Q / H_KV);

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int q_start = q_tile_idx * QUERIES_PER_BLOCK;
    int q_end_count = min(QUERIES_PER_BLOCK, N - q_start);

    bool is_active = (warp_id < q_end_count) && (q_start < N);

    int q_idx = q_start + warp_id;

    const half* Q_ptr = Q + ((b * H_Q + h_q) * N * D);
    const half* K_ptr = K + ((b * H_KV + h_kv) * N * D);
    const half* V_ptr = V + ((b * H_KV + h_kv) * N * D);
    half* O_ptr = O + ((b * H_Q + h_q) * N * D);

    __shared__ half sQ[QUERIES_PER_BLOCK][MAX_D];
    __shared__ half sK[TILE_K][MAX_D];
    __shared__ half sV[TILE_K][MAX_D];

    if (is_active) {
        const half* q_row = Q_ptr + q_idx * D;
        for (int d = lane_id * 2; d < D; d += WARP_SIZE * 2) {
            half2 h2 = *reinterpret_cast<const half2*>(&q_row[d]);
            sQ[warp_id][d] = h2.x;
            sQ[warp_id][d+1] = h2.y;
        }
    }

    float m_i = -FLT_MAX;
    float l_i = 0.f;
    float acc[MAX_D];
    for (int d = 0; d < D; ++d) acc[d] = 0.f;

    __syncthreads();

    for (int k_start = 0; k_start < N; k_start += TILE_K) {
        int k_len = min(TILE_K, N - k_start);

        int total_elements = k_len * D;
        for (int idx = tid * 2; idx < total_elements; idx += WARP_BLOCK_SIZE * 2) {
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

        if (is_active) {
            float scores[TILE_K];
            float m_block = -FLT_MAX;

            for (int j = 0; j < k_len; ++j) {
                int k_idx = k_start + j;
                if (causal && k_idx > q_idx) {
                    scores[j] = -FLT_MAX;
                    continue;
                }

                float dot = 0.0f;
                for (int d = lane_id; d < D; d += WARP_SIZE) {
                    dot += __half2float(sQ[warp_id][d]) * __half2float(sK[j][d]);
                }
                dot = warp_reduce_sum(dot);
                dot = __shfl_sync(0xffffffff, dot, 0);
                dot *= scale;
                scores[j] = dot;
                m_block = fmaxf(m_block, dot);
            }

            m_block = warp_reduce_max(m_block);
            m_block = __shfl_sync(0xffffffff, m_block, 0);

            if (m_block != -FLT_MAX) {
                float m_new = fmaxf(m_i, m_block);
                float alpha = (m_i == -FLT_MAX) ? 0.f : expf(m_i - m_new);

                float beta_sum = 0.f;
                float weighted[MAX_D];
                for (int d = 0; d < D; ++d) weighted[d] = 0.f;

                for (int j = 0; j < k_len; ++j) {
                    if (scores[j] == -FLT_MAX) continue;
                    float p = expf(scores[j] - m_new);
                    beta_sum += p;
                    for (int d = lane_id; d < D; d += WARP_SIZE) {
                        weighted[d] += __half2float(sV[j][d]) * p;
                    }
                }

                for (int d = lane_id; d < D; d += WARP_SIZE) {
                    acc[d] = alpha * acc[d] + weighted[d];
                }
                l_i = alpha * l_i + beta_sum;
                m_i = m_new;
            }
        }

        __syncthreads();
    }

    if (is_active) {
        for (int d = lane_id; d < D; d += WARP_SIZE) {
            O_ptr[q_idx * D + d] = __float2half(acc[d] / l_i);
        }
    }
}

torch::Tensor flash_attn_forward_prefill_gqa_fp16(
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
    auto H_Q = q.size(1);
    auto N = q.size(2);
    auto D = q.size(3);
    auto H_KV = k.size(1);

    TORCH_CHECK(H_Q >= H_KV, "H_Q must be >= H_KV");
    TORCH_CHECK(H_Q % H_KV == 0, "H_Q must be divisible by H_KV");
    TORCH_CHECK(D <= MAX_D, "D must be <= MAX_D(128)");
    TORCH_CHECK(k.size(2) == N, "k sequence length must match q");
    TORCH_CHECK(v.size(2) == N, "v sequence length must match q");

    auto out = torch::zeros({B, H_Q, N, D}, q.options());

    dim3 grid((N + TILE_Q - 1) / TILE_Q, B * H_Q);
    dim3 block(TILE_Q);

    float scale = 1.0f / std::sqrt((float)D);

    flash_attn_kernel_prefill_gqa_fp16<<<grid, block>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        B,
        H_Q,
        H_KV,
        N,
        D,
        causal,
        scale
    );

    return out;
}

__global__ void flash_attn_kernel_decode_int8_gqa(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const float* __restrict__ Q_scale,
    const float* __restrict__ K_scale,
    int B,
    int H_Q,
    int H_KV,
    int N,
    int D,
    float scale,
    int cache_len
) {
    int bqh = blockIdx.x;
    int b = bqh / H_Q;
    int h_q = bqh % H_Q;
    int h_kv = h_q / (H_Q / H_KV);

    int tid = threadIdx.x;
    int lane_id = tid % 32;

    const half* Q_ptr = Q + ((b * H_Q + h_q) * D);
    const half* K_ptr = K + ((b * H_KV + h_kv) * N * D);
    const half* V_ptr = V + ((b * H_KV + h_kv) * N * D);
    half* O_ptr = O + ((b * H_Q + h_q) * D);

    float q_s = Q_scale[b * H_Q + h_q];
    float k_s = K_scale[b * H_KV + h_kv];

    float m_i = -FLT_MAX;
    float l_i = 0.f;
    float acc[MAX_D];
    for (int d = 0; d < D; ++d) acc[d] = 0.f;

    __shared__ int8_t sQ_int8[MAX_D];
    __shared__ half sV_row[MAX_D];

    if (tid < D) {
        float q_val = __half2float(Q_ptr[tid]);
        sQ_int8[tid] = static_cast<int8_t>(fmaxf(-127.f, fminf(127.f, roundf(q_val / q_s))));
    }
    __syncthreads();

    constexpr int TILE_K_DECODE = 64;
    int num_kv_blocks = (N + TILE_K_DECODE - 1) / TILE_K_DECODE;

    __shared__ int8_t sK_int8[TILE_K_DECODE][MAX_D];
    __shared__ half sV[TILE_K_DECODE][MAX_D];

    for (int block_idx = 0; block_idx < num_kv_blocks; ++block_idx) {
        int k_start = block_idx * TILE_K_DECODE;
        int k_len = min(TILE_K_DECODE, N - k_start);

        int total_elements = k_len * D;
        for (int idx = tid; idx < total_elements; idx += blockDim.x) {
            int row = idx / D;
            int col = idx % D;
            float k_val = __half2float(K_ptr[(k_start + row) * D + col]);
            sK_int8[row][col] = static_cast<int8_t>(fmaxf(-127.f, fminf(127.f, roundf(k_val / k_s))));
            sV[row][col] = V_ptr[(k_start + row) * D + col];
        }
        __syncthreads();

        float scores[TILE_K_DECODE];
        float m_block = -FLT_MAX;

        for (int j = 0; j < k_len; ++j) {
            int k_idx = k_start + j;
            if (k_idx > cache_len) {
                scores[j] = -FLT_MAX;
                continue;
            }

            int int_dot = 0;
            #pragma unroll
            for (int d = lane_id * 4; d < D; d += 32 * 4) {
                int8_t a0 = sQ_int8[d];
                int8_t a1 = (d + 1 < D) ? sQ_int8[d + 1] : (int8_t)0;
                int8_t a2 = (d + 2 < D) ? sQ_int8[d + 2] : (int8_t)0;
                int8_t a3 = (d + 3 < D) ? sQ_int8[d + 3] : (int8_t)0;
                int8_t b0 = sK_int8[j][d];
                int8_t b1 = (d + 1 < D) ? sK_int8[j][d + 1] : (int8_t)0;
                int8_t b2 = (d + 2 < D) ? sK_int8[j][d + 2] : (int8_t)0;
                int8_t b3 = (d + 3 < D) ? sK_int8[j][d + 3] : (int8_t)0;
                int_dot += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
            }

            int_dot = warp_reduce_sum(int_dot);

            if (lane_id == 0) {
                float dot = static_cast<float>(int_dot) * q_s * k_s * scale;
                scores[j] = dot;
            }

            scores[j] = __shfl_sync(0xffffffff, scores[j], 0);
            m_block = fmaxf(m_block, scores[j]);
        }

        m_block = warp_reduce_max(m_block);
        m_block = __shfl_sync(0xffffffff, m_block, 0);

        if (m_block == -FLT_MAX) {
            __syncthreads();
            continue;
        }

        float m_new = fmaxf(m_i, m_block);
        float alpha = (m_i == -FLT_MAX) ? 0.f : expf(m_i - m_new);

        if (lane_id == 0) {
            float beta_sum = 0.f;
            float weighted[MAX_D];
            for (int d = 0; d < D; ++d) weighted[d] = 0.f;

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

    if (tid == 0) {
        for (int d = 0; d < D; ++d) {
            O_ptr[d] = __float2half(acc[d] / l_i);
        }
    }
}

torch::Tensor flash_attn_forward_decode_int8_gqa(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor q_scale,
    torch::Tensor k_scale,
    int cache_len
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(q_scale);
    CHECK_INPUT(k_scale);

    TORCH_CHECK(q.dtype() == torch::kHalf, "q must be float16");
    TORCH_CHECK(q.size(2) == 1, "q sequence length must be 1 for decode mode");

    auto B = q.size(0);
    auto H_Q = q.size(1);
    auto H_KV = k.size(1);
    auto N = k.size(2);
    auto D = q.size(3);

    TORCH_CHECK(H_Q >= H_KV, "H_Q must be >= H_KV");
    TORCH_CHECK(H_Q % H_KV == 0, "H_Q must be divisible by H_KV");
    TORCH_CHECK(D <= MAX_D, "D must be <= MAX_D(128)");

    auto out = torch::zeros({B, H_Q, 1, D}, q.options());

    int threads = 128;
    dim3 grid(B * H_Q);
    dim3 block(threads);

    float scale = 1.0f / std::sqrt((float)D);

    flash_attn_kernel_decode_int8_gqa<<<grid, block>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        q_scale.data_ptr<float>(),
        k_scale.data_ptr<float>(),
        B,
        H_Q,
        H_KV,
        N,
        D,
        scale,
        cache_len
    );

    return out;
}

__global__ void flash_attn_kernel_varlen_prefill_fp16(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int* __restrict__ cu_seqlens,
    int H,
    int total_q,
    int D,
    bool causal,
    float scale
) {
    int q_tile_idx = blockIdx.x;
    int h = blockIdx.y;
    int batch_idx = blockIdx.z;

    int seq_start = cu_seqlens[batch_idx];
    int seq_end = cu_seqlens[batch_idx + 1];
    int seq_len = seq_end - seq_start;

    if (seq_len == 0) return;

    int tid = threadIdx.x;
    int q_start = seq_start + q_tile_idx * TILE_Q;

    if (q_start >= seq_end) return;
    int q_end_count = min(TILE_Q, seq_end - q_start);

    bool is_active = tid < q_end_count;

    const half* Q_ptr = Q + h * D;
    const half* K_ptr = K + h * D;
    const half* V_ptr = V + h * D;
    half* O_ptr = O + h * D;

    __shared__ half sQ[TILE_Q][MAX_D];
    __shared__ half sK[TILE_K][MAX_D];
    __shared__ half sV[TILE_K][MAX_D];

    if (is_active) {
        const half* q_row = Q_ptr + (q_start + tid) * H * D;
        #pragma unroll
        for (int d = 0; d < D; d += 2) {
            half2 h2 = *reinterpret_cast<const half2*>(&q_row[d]);
            sQ[tid][d] = h2.x;
            sQ[tid][d+1] = h2.y;
        }
    }

    float m_i = -FLT_MAX;
    float l_i = 0.f;
    float acc[MAX_D];
    for (int d = 0; d < D; ++d) acc[d] = 0.f;

    __syncthreads();

    for (int k_start = seq_start; k_start < seq_end; k_start += TILE_K) {
        int k_len = min(TILE_K, seq_end - k_start);

        int total_elements = k_len * D;
        for (int idx = tid * 2; idx < total_elements; idx += TILE_Q * 2) {
            int row = idx / D;
            int col = idx % D;
            if (row < k_len) {
                half2 h2_k = *reinterpret_cast<const half2*>(&K_ptr[(k_start + row) * H * D + col]);
                half2 h2_v = *reinterpret_cast<const half2*>(&V_ptr[(k_start + row) * H * D + col]);
                sK[row][col] = h2_k.x;
                sK[row][col+1] = h2_k.y;
                sV[row][col] = h2_v.x;
                sV[row][col+1] = h2_v.y;
            }
        }
        __syncthreads();

        if (is_active) {
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
                #pragma unroll
                for (int d = 0; d < D; d += 2) {
                    half2 a = make_half2(sQ[tid][d], sQ[tid][d+1]);
                    half2 b2 = make_half2(sK[j][d], sK[j][d+1]);
                    float2 p = __half22float2(__hmul2(a, b2));
                    dot += p.x + p.y;
                }
                dot *= scale;
                scores[j] = dot;
                m_block = fmaxf(m_block, dot);
            }

            if (m_block != -FLT_MAX) {
                float m_new = fmaxf(m_i, m_block);
                float alpha = (m_i == -FLT_MAX) ? 0.f : expf(m_i - m_new);

                float beta_sum = 0.f;
                float weighted[MAX_D];
                for (int d = 0; d < D; ++d) weighted[d] = 0.f;

                for (int j = 0; j < k_len; ++j) {
                    if (scores[j] == -FLT_MAX) continue;
                    float p = expf(scores[j] - m_new);
                    beta_sum += p;
                    #pragma unroll
                    for (int d = 0; d < D; d += 2) {
                        half2 v = make_half2(sV[j][d], sV[j][d+1]);
                        float2 vf = __half22float2(v);
                        weighted[d] += vf.x * p;
                        weighted[d+1] += vf.y * p;
                    }
                }

                for (int d = 0; d < D; ++d) {
                    acc[d] = alpha * acc[d] + weighted[d];
                }
                l_i = alpha * l_i + beta_sum;
                m_i = m_new;
            }
        }

        __syncthreads();
    }

    if (is_active) {
        int q_idx_write = q_start + tid;
        #pragma unroll
        for (int d = 0; d < D; d += 2) {
            half2 out_h2 = make_half2(
                __float2half(acc[d] / l_i),
                __float2half(acc[d+1] / l_i)
            );
            *reinterpret_cast<half2*>(&O_ptr[q_idx_write * H * D + d]) = out_h2;
        }
    }
}

torch::Tensor flash_attn_forward_varlen_prefill_fp16(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor cu_seqlens,
    int max_seqlen,
    bool causal
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(cu_seqlens);

    TORCH_CHECK(q.dtype() == torch::kHalf, "q must be float16");
    TORCH_CHECK(q.dim() == 3, "q must be [total_q, H, D] for varlen mode");
    TORCH_CHECK(k.dim() == 3, "k must be [total_kv, H, D] for varlen mode");
    TORCH_CHECK(v.dim() == 3, "v must be [total_kv, H, D] for varlen mode");

    auto total_q = q.size(0);
    auto H = q.size(1);
    auto D = q.size(2);
    auto batch_size = cu_seqlens.size(0) - 1;

    TORCH_CHECK(D <= MAX_D, "D must be <= MAX_D(128)");
    TORCH_CHECK(k.size(0) == total_q, "k and q must have same total length");
    TORCH_CHECK(v.size(0) == total_q, "v and q must have same total length");

    auto out = torch::zeros_like(q);

    int num_q_tiles = (max_seqlen + TILE_Q - 1) / TILE_Q;

    dim3 grid(num_q_tiles, H, batch_size);
    dim3 block(TILE_Q);

    float scale = 1.0f / std::sqrt((float)D);

    flash_attn_kernel_varlen_prefill_fp16<<<grid, block>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        cu_seqlens.data_ptr<int>(),
        H,
        total_q,
        D,
        causal,
        scale
    );

    return out;
}

constexpr int PAGE_BLOCK_SIZE = 16;

__global__ void flash_attn_kernel_paged_decode_gqa_fp16(
    const half* __restrict__ Q,
    const half* __restrict__ K_cache,
    const half* __restrict__ V_cache,
    half* __restrict__ O,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    int B,
    int H_Q,
    int H_KV,
    int D,
    int max_num_blocks,
    float scale
) {
    int bqh = blockIdx.x;
    int b = bqh / H_Q;
    int h_q = bqh % H_Q;
    int h_kv = h_q / (H_Q / H_KV);

    int tid = threadIdx.x;
    int lane_id = tid % 32;

    int seq_len = seq_lens[b];

    const half* Q_ptr = Q + ((b * H_Q + h_q) * D);
    half* O_ptr = O + ((b * H_Q + h_q) * D);

    const half* K_head_ptr = K_cache + (h_kv * PAGE_BLOCK_SIZE * D);
    const half* V_head_ptr = V_cache + (h_kv * PAGE_BLOCK_SIZE * D);

    float m_i = -FLT_MAX;
    float l_i = 0.f;
    float acc[MAX_D];
    for (int d = 0; d < D; ++d) acc[d] = 0.f;

    __shared__ half sQ[MAX_D];
    if (tid < D) {
        sQ[tid] = Q_ptr[tid];
    }
    __syncthreads();

    __shared__ half sK[PAGE_BLOCK_SIZE][MAX_D];
    __shared__ half sV[PAGE_BLOCK_SIZE][MAX_D];

    int num_blocks = (seq_len + PAGE_BLOCK_SIZE - 1) / PAGE_BLOCK_SIZE;

    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int physical_block = block_table[b * max_num_blocks + block_idx];
        int block_start = block_idx * PAGE_BLOCK_SIZE;
        int k_len = min(PAGE_BLOCK_SIZE, seq_len - block_start);

        int total_elements = k_len * D;
        for (int idx = tid; idx < total_elements; idx += blockDim.x) {
            int row = idx / D;
            int col = idx % D;
            sK[row][col] = K_head_ptr[(physical_block * H_KV * PAGE_BLOCK_SIZE * D) + row * D + col];
            sV[row][col] = V_head_ptr[(physical_block * H_KV * PAGE_BLOCK_SIZE * D) + row * D + col];
        }
        __syncthreads();

        float scores[PAGE_BLOCK_SIZE];
        float m_block = -FLT_MAX;

        for (int j = 0; j < k_len; ++j) {
            float dot = 0.0f;
            #pragma unroll
            for (int d = lane_id; d < D; d += 32) {
                dot += __half2float(sQ[d]) * __half2float(sK[j][d]);
            }

            dot = warp_reduce_sum(dot);

            if (lane_id == 0) {
                dot *= scale;
                scores[j] = dot;
            }

            scores[j] = __shfl_sync(0xffffffff, scores[j], 0);
            m_block = fmaxf(m_block, scores[j]);
        }

        m_block = warp_reduce_max(m_block);
        m_block = __shfl_sync(0xffffffff, m_block, 0);

        if (m_block == -FLT_MAX) {
            __syncthreads();
            continue;
        }

        float m_new = fmaxf(m_i, m_block);
        float alpha = (m_i == -FLT_MAX) ? 0.f : expf(m_i - m_new);

        if (lane_id == 0) {
            float beta_sum = 0.f;
            float weighted[MAX_D];
            for (int d = 0; d < D; ++d) weighted[d] = 0.f;

            for (int j = 0; j < k_len; ++j) {
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

    if (tid == 0) {
        for (int d = 0; d < D; ++d) {
            O_ptr[d] = __float2half(acc[d] / l_i);
        }
    }
}

torch::Tensor flash_attn_forward_paged_decode_gqa_fp16(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int max_num_blocks
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(block_table);
    CHECK_INPUT(seq_lens);

    TORCH_CHECK(q.dtype() == torch::kHalf, "q must be float16");
    TORCH_CHECK(k_cache.dtype() == torch::kHalf, "k_cache must be float16");
    TORCH_CHECK(v_cache.dtype() == torch::kHalf, "v_cache must be float16");
    TORCH_CHECK(q.size(2) == 1, "q sequence length must be 1 for decode mode");

    auto B = q.size(0);
    auto H_Q = q.size(1);
    auto H_KV = k_cache.size(1);
    auto D = q.size(3);

    TORCH_CHECK(H_Q >= H_KV, "H_Q must be >= H_KV");
    TORCH_CHECK(H_Q % H_KV == 0, "H_Q must be divisible by H_KV");
    TORCH_CHECK(D <= MAX_D, "D must be <= MAX_D(128)");
    TORCH_CHECK(k_cache.size(2) == PAGE_BLOCK_SIZE, "k_cache block size mismatch");
    TORCH_CHECK(v_cache.size(2) == PAGE_BLOCK_SIZE, "v_cache block size mismatch");

    auto out = torch::zeros({B, H_Q, 1, D}, q.options());

    int threads = 128;
    dim3 grid(B * H_Q);
    dim3 block(threads);

    float scale = 1.0f / std::sqrt((float)D);

    flash_attn_kernel_paged_decode_gqa_fp16<<<grid, block>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        block_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(),
        B,
        H_Q,
        H_KV,
        D,
        max_num_blocks,
        scale
    );

    return out;
}

torch::Tensor flash_attn_forward_prefill_gqa_fp16_warp(
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
    auto H_Q = q.size(1);
    auto N = q.size(2);
    auto D = q.size(3);
    auto H_KV = k.size(1);

    TORCH_CHECK(H_Q >= H_KV, "H_Q must be >= H_KV");
    TORCH_CHECK(H_Q % H_KV == 0, "H_Q must be divisible by H_KV");
    TORCH_CHECK(D <= MAX_D, "D must be <= MAX_D(128)");
    TORCH_CHECK(k.size(2) == N, "k sequence length must match q");
    TORCH_CHECK(v.size(2) == N, "v sequence length must match q");

    auto out = torch::zeros({B, H_Q, N, D}, q.options());

    dim3 grid((N + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK, B * H_Q);
    dim3 block(WARP_BLOCK_SIZE);

    float scale = 1.0f / std::sqrt((float)D);

    flash_attn_kernel_prefill_gqa_fp16_warp<<<grid, block>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        B,
        H_Q,
        H_KV,
        N,
        D,
        causal,
        scale
    );

    return out;
}

torch::Tensor flash_attn_forward_decode_gqa_fp16_stream(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    int cache_len,
    int64_t stream_int
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    TORCH_CHECK(q.dtype() == torch::kHalf, "q must be float16");
    TORCH_CHECK(q.size(2) == 1, "q sequence length must be 1 for decode mode");

    auto H_Q = q.size(1);
    auto H_KV = k.size(1);
    TORCH_CHECK(H_KV == v.size(1), "k and v must have same number of heads");
    TORCH_CHECK(H_Q >= H_KV, "H_Q must be >= H_KV");
    TORCH_CHECK(H_Q % H_KV == 0, "H_Q must be divisible by H_KV");

    auto B = q.size(0);
    auto N = k.size(2);
    auto D = q.size(3);

    TORCH_CHECK(D <= MAX_D, "D must be less than or equal to MAX_D(128)");
    TORCH_CHECK(cache_len < N, "cache_len must be less than N");

    auto out = torch::zeros({B, H_Q, 1, D}, q.options());

    int threads = 128;
    dim3 grid(B * H_Q);
    dim3 block(threads);

    float scale = 1.0f / std::sqrt((float)D);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_int);

    flash_attn_kernel_decode_gqa_fp16<<<grid, block, 0, stream>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        B,
        H_Q,
        H_KV,
        N,
        D,
        causal,
        scale,
        cache_len
    );

    return out;
}

torch::Tensor flash_attn_forward_varlen_prefill_fp16_stream(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor cu_seqlens,
    int max_seqlen,
    bool causal,
    int64_t stream_int
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(cu_seqlens);

    TORCH_CHECK(q.dtype() == torch::kHalf, "q must be float16");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");

    auto total_q = q.size(0);
    auto H = q.size(1);
    auto D = q.size(2);
    auto batch_size = cu_seqlens.size(0) - 1;

    TORCH_CHECK(D <= MAX_D, "D must be <= MAX_D(128)");
    TORCH_CHECK(k.size(0) == total_q, "k and q must have same total length");
    TORCH_CHECK(v.size(0) == total_q, "v and q must have same total length");

    auto out = torch::zeros_like(q);

    int num_q_tiles = (max_seqlen + TILE_Q - 1) / TILE_Q;

    dim3 grid(num_q_tiles, H, batch_size);
    dim3 block(TILE_Q);

    float scale = 1.0f / std::sqrt((float)D);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_int);

    flash_attn_kernel_varlen_prefill_fp16<<<grid, block, 0, stream>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        cu_seqlens.data_ptr<int>(),
        H,
        total_q,
        D,
        causal,
        scale
    );

    return out;
}

void async_h2d_transfer(
    torch::Tensor dst,
    torch::Tensor src,
    int64_t stream_int
) {
    TORCH_CHECK(dst.is_cuda(), "dst must be a CUDA tensor");
    TORCH_CHECK(!src.is_cuda(), "src must be a CPU tensor");
    TORCH_CHECK(dst.dtype() == src.dtype(), "dtype must match");
    TORCH_CHECK(dst.numel() == src.numel(), "size must match");
    TORCH_CHECK(src.is_pinned(), "src must be pinned memory for async transfer");

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_int);
    size_t nbytes = dst.numel() * dst.element_size();

    cudaMemcpyAsync(
        dst.data_ptr(),
        src.data_ptr(),
        nbytes,
        cudaMemcpyHostToDevice,
        stream
    );
}

torch::Tensor alloc_pinned_tensor(
    std::vector<int64_t> sizes,
    int dtype_int
) {
    auto options = torch::TensorOptions()
        .dtype(torch::ScalarType(dtype_int))
        .device(torch::kCPU)
        .pinned_memory(true);

    return torch::empty(sizes, options);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pipeline_decode_step(
    torch::Tensor q_compute,
    torch::Tensor k_compute,
    torch::Tensor v_compute,
    torch::Tensor q_transfer,
    torch::Tensor k_transfer,
    torch::Tensor v_transfer,
    torch::Tensor q_h2d_dst,
    torch::Tensor k_h2d_dst,
    torch::Tensor v_h2d_dst,
    bool causal,
    int cache_len,
    int64_t compute_stream_int,
    int64_t transfer_stream_int
) {
    CHECK_INPUT(q_compute);
    CHECK_INPUT(k_compute);
    CHECK_INPUT(v_compute);

    auto H_Q = q_compute.size(1);
    auto H_KV = k_compute.size(1);
    auto B = q_compute.size(0);
    auto N = k_compute.size(2);
    auto D = q_compute.size(3);

    TORCH_CHECK(D <= MAX_D, "D must be <= MAX_D(128)");

    auto out = torch::zeros({B, H_Q, 1, D}, q_compute.options());

    cudaStream_t compute_stream = reinterpret_cast<cudaStream_t>(compute_stream_int);
    cudaStream_t transfer_stream = reinterpret_cast<cudaStream_t>(transfer_stream_int);

    int threads = 128;
    dim3 grid(B * H_Q);
    dim3 block(threads);
    float scale = 1.0f / std::sqrt((float)D);

    flash_attn_kernel_decode_gqa_fp16<<<grid, block, 0, compute_stream>>>(
        reinterpret_cast<half*>(q_compute.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k_compute.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_compute.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        B,
        H_Q,
        H_KV,
        N,
        D,
        causal,
        scale,
        cache_len
    );

    if (q_transfer.numel() > 0 && q_h2d_dst.numel() > 0) {
        size_t q_bytes = q_h2d_dst.numel() * q_h2d_dst.element_size();
        size_t k_bytes = k_h2d_dst.numel() * k_h2d_dst.element_size();
        size_t v_bytes = v_h2d_dst.numel() * v_h2d_dst.element_size();

        cudaMemcpyAsync(q_h2d_dst.data_ptr(), q_transfer.data_ptr(), q_bytes,
                        cudaMemcpyHostToDevice, transfer_stream);
        cudaMemcpyAsync(k_h2d_dst.data_ptr(), k_transfer.data_ptr(), k_bytes,
                        cudaMemcpyHostToDevice, transfer_stream);
        cudaMemcpyAsync(v_h2d_dst.data_ptr(), v_transfer.data_ptr(), v_bytes,
                        cudaMemcpyHostToDevice, transfer_stream);
    }

    return std::make_tuple(out, q_h2d_dst, k_h2d_dst);
}

constexpr int SPEC_MAX_K = 8;

__global__ void speculative_verify_kernel(
    const float* __restrict__ target_probs,
    const float* __restrict__ draft_probs,
    const int64_t* __restrict__ draft_tokens,
    float* __restrict__ random_vals,
    int* __restrict__ accepted_count,
    int64_t* __restrict__ output_tokens,
    int B,
    int K,
    int vocab_size
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    if (bid >= B) return;

    const float* target_b = target_probs + bid * (K + 1) * vocab_size;
    const float* draft_b = draft_probs + bid * K * vocab_size;
    const int64_t* draft_tok_b = draft_tokens + bid * K;
    int64_t* out_tok_b = output_tokens + bid * (K + 1);
    float* rand_b = random_vals + bid * K;

    __shared__ int s_accepted;
    __shared__ int s_rejected;

    if (tid == 0) {
        s_accepted = 0;
        s_rejected = 0;
    }
    __syncthreads();

    for (int i = 0; i < K; i++) {
        if (s_rejected) break;

        int64_t tok = draft_tok_b[i];
        float p_tok = target_b[i * vocab_size + tok];
        float q_tok = draft_b[i * vocab_size + tok];

        float accept_prob = fminf(1.0f, p_tok / (q_tok + 1e-10f));
        float r = rand_b[i];

        if (r < accept_prob) {
            if (tid == 0) {
                out_tok_b[i] = tok;
                s_accepted++;
            }
        } else {
            if (tid == 0) {
                s_rejected = 1;

                float Z = 0.0f;
                for (int v = 0; v < vocab_size; v++) {
                    float diff = target_b[i * vocab_size + v] - draft_b[i * vocab_size + v];
                    if (diff > 0.0f) Z += diff;
                }

                if (Z > 0.0f) {
                    float r2 = rand_b[i];
                    float cumsum = 0.0f;
                    int sampled_tok = 0;
                    for (int v = 0; v < vocab_size; v++) {
                        float diff = target_b[i * vocab_size + v] - draft_b[i * vocab_size + v];
                        if (diff > 0.0f) {
                            cumsum += diff / Z;
                            if (r2 < cumsum && sampled_tok == 0) {
                                sampled_tok = v;
                            }
                        }
                    }
                    out_tok_b[i] = sampled_tok;
                } else {
                    float r2 = rand_b[i];
                    float cum = 0.0f;
                    int sampled_tok = 0;
                    for (int v = 0; v < vocab_size; v++) {
                        cum += target_b[i * vocab_size + v];
                        if (r2 < cum && sampled_tok == 0) {
                            sampled_tok = v;
                        }
                    }
                    out_tok_b[i] = sampled_tok;
                }

                for (int j = i + 1; j < K; j++) {
                    out_tok_b[j] = -1;
                }

                float bonus_r = rand_b[min(i, K - 1)];
                float cum = 0.0f;
                int bonus_tok = 0;
                for (int v = 0; v < vocab_size; v++) {
                    cum += target_b[(i + 1) * vocab_size + v];
                    if (cum >= bonus_r && bonus_tok == 0) {
                        bonus_tok = v;
                    }
                }
                out_tok_b[i + 1] = bonus_tok;
            }
            __syncthreads();
            break;
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (!s_rejected) {
            float bonus_r = rand_b[K - 1];
            float cum = 0.0f;
            int bonus_tok = 0;
            for (int v = 0; v < vocab_size; v++) {
                cum += target_b[K * vocab_size + v];
                if (cum >= bonus_r && bonus_tok == 0) {
                    bonus_tok = v;
                }
            }
            out_tok_b[K] = bonus_tok;
        }
        accepted_count[bid] = s_accepted;
    }
}

std::tuple<torch::Tensor, torch::Tensor> speculative_verify_forward(
    torch::Tensor target_probs,
    torch::Tensor draft_probs,
    torch::Tensor draft_tokens,
    torch::Tensor random_vals,
    int K
) {
    CHECK_INPUT(target_probs);
    CHECK_INPUT(draft_probs);
    CHECK_INPUT(draft_tokens);
    CHECK_INPUT(random_vals);

    TORCH_CHECK(target_probs.dtype() == torch::kFloat32, "target_probs must be float32");
    TORCH_CHECK(draft_probs.dtype() == torch::kFloat32, "draft_probs must be float32");
    TORCH_CHECK(draft_tokens.dtype() == torch::kInt64, "draft_tokens must be int64");

    auto B = target_probs.size(0);
    auto vocab_size = target_probs.size(2);

    TORCH_CHECK(K <= SPEC_MAX_K, "K must be <= SPEC_MAX_K(8)");
    TORCH_CHECK(target_probs.size(1) == K + 1, "target_probs must have K+1 positions");
    TORCH_CHECK(draft_probs.size(1) == K, "draft_probs must have K positions");
    TORCH_CHECK(draft_tokens.size(1) == K, "draft_tokens must have K tokens");

    auto accepted_count = torch::zeros({B}, torch::TensorOptions().dtype(torch::kInt32).device(target_probs.device()));
    auto output_tokens = torch::full({B, K + 1}, -1, torch::TensorOptions().dtype(torch::kInt64).device(target_probs.device()));

    int threads = 256;
    dim3 grid(B);
    dim3 block(threads);

    speculative_verify_kernel<<<grid, block>>>(
        target_probs.data_ptr<float>(),
        draft_probs.data_ptr<float>(),
        draft_tokens.data_ptr<int64_t>(),
        random_vals.data_ptr<float>(),
        accepted_count.data_ptr<int>(),
        output_tokens.data_ptr<int64_t>(),
        B,
        K,
        vocab_size
    );

    return std::make_tuple(accepted_count, output_tokens);
}

__global__ void flash_attn_kernel_decode_int8_kv_cache_gqa(
    const half* __restrict__ Q,
    const int8_t* __restrict__ K_int8,
    const int8_t* __restrict__ V_int8,
    const half* __restrict__ K_scale,
    const half* __restrict__ V_scale,
    half* __restrict__ O,
    int B,
    int H_Q,
    int H_KV,
    int N,
    int D,
    float scale,
    int cache_len
) {
    int bqh = blockIdx.x;
    int b = bqh / H_Q;
    int h_q = bqh % H_Q;
    int h_kv = h_q / (H_Q / H_KV);

    int tid = threadIdx.x;
    int lane_id = tid % 32;

    const half* Q_ptr = Q + ((b * H_Q + h_q) * D);
    const int8_t* K_ptr = K_int8 + ((b * H_KV + h_kv) * N * D);
    const int8_t* V_ptr = V_int8 + ((b * H_KV + h_kv) * N * D);
    const half* Ks_ptr = K_scale + ((b * H_KV + h_kv) * N);
    const half* Vs_ptr = V_scale + ((b * H_KV + h_kv) * N);
    half* O_ptr = O + ((b * H_Q + h_q) * D);

    float m_i = -FLT_MAX;
    float l_i = 0.f;
    float acc[MAX_D];
    for (int d = 0; d < D; ++d) acc[d] = 0.f;

    __shared__ half sQ[MAX_D];
    if (tid < D) {
        sQ[tid] = Q_ptr[tid];
    }
    __syncthreads();

    constexpr int TILE_K_DECODE = 64;
    int num_kv_blocks = (N + TILE_K_DECODE - 1) / TILE_K_DECODE;

    __shared__ half sK[TILE_K_DECODE][MAX_D];
    __shared__ half sV[TILE_K_DECODE][MAX_D];
    __shared__ half sK_scale[TILE_K_DECODE];
    __shared__ half sV_scale[TILE_K_DECODE];

    for (int block_idx = 0; block_idx < num_kv_blocks; ++block_idx) {
        int k_start = block_idx * TILE_K_DECODE;
        int k_len = min(TILE_K_DECODE, N - k_start);

        for (int j = tid; j < k_len; j += blockDim.x) {
            sK_scale[j] = Ks_ptr[k_start + j];
            sV_scale[j] = Vs_ptr[k_start + j];
        }

        int total_elements = k_len * D;
        for (int idx = tid; idx < total_elements; idx += blockDim.x) {
            int row = idx / D;
            int col = idx % D;
            float k_val = static_cast<float>(K_ptr[(k_start + row) * D + col]) * __half2float(sK_scale[row]);
            float v_val = static_cast<float>(V_ptr[(k_start + row) * D + col]) * __half2float(sV_scale[row]);
            sK[row][col] = __float2half(k_val);
            sV[row][col] = __float2half(v_val);
        }
        __syncthreads();

        float scores[TILE_K_DECODE];
        float m_block = -FLT_MAX;

        for (int j = 0; j < k_len; ++j) {
            int k_idx = k_start + j;
            if (k_idx > cache_len) {
                scores[j] = -FLT_MAX;
                continue;
            }

            float dot = 0.0f;
            #pragma unroll
            for (int d = lane_id; d < D; d += 32) {
                dot += __half2float(sQ[d]) * __half2float(sK[j][d]);
            }

            dot = warp_reduce_sum(dot);

            if (lane_id == 0) {
                dot *= scale;
                scores[j] = dot;
            }

            scores[j] = __shfl_sync(0xffffffff, scores[j], 0);
            m_block = fmaxf(m_block, scores[j]);
        }

        m_block = warp_reduce_max(m_block);
        m_block = __shfl_sync(0xffffffff, m_block, 0);

        if (m_block == -FLT_MAX) {
            __syncthreads();
            continue;
        }

        float m_new = fmaxf(m_i, m_block);
        float alpha = (m_i == -FLT_MAX) ? 0.f : expf(m_i - m_new);

        if (lane_id == 0) {
            float beta_sum = 0.f;
            float weighted[MAX_D];
            for (int d = 0; d < D; ++d) weighted[d] = 0.f;

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

    if (tid == 0) {
        for (int d = 0; d < D; ++d) {
            O_ptr[d] = __float2half(acc[d] / l_i);
        }
    }
}

torch::Tensor flash_attn_forward_decode_int8_kv_cache_gqa(
    torch::Tensor q,
    torch::Tensor k_int8,
    torch::Tensor v_int8,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    int cache_len
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k_int8);
    CHECK_INPUT(v_int8);
    CHECK_INPUT(k_scale);
    CHECK_INPUT(v_scale);

    TORCH_CHECK(q.dtype() == torch::kHalf, "q must be float16");
    TORCH_CHECK(k_int8.dtype() == torch::kInt8, "k_int8 must be int8");
    TORCH_CHECK(v_int8.dtype() == torch::kInt8, "v_int8 must be int8");
    TORCH_CHECK(k_scale.dtype() == torch::kHalf, "k_scale must be float16");
    TORCH_CHECK(v_scale.dtype() == torch::kHalf, "v_scale must be float16");
    TORCH_CHECK(q.size(2) == 1, "q sequence length must be 1 for decode mode");

    auto B = q.size(0);
    auto H_Q = q.size(1);
    auto H_KV = k_int8.size(1);
    auto N = k_int8.size(2);
    auto D = q.size(3);

    TORCH_CHECK(H_Q >= H_KV, "H_Q must be >= H_KV");
    TORCH_CHECK(H_Q % H_KV == 0, "H_Q must be divisible by H_KV");
    TORCH_CHECK(D <= MAX_D, "D must be <= MAX_D(128)");
    TORCH_CHECK(k_int8.size(3) == D, "k_int8 last dim must match D");
    TORCH_CHECK(v_int8.size(3) == D, "v_int8 last dim must match D");
    TORCH_CHECK(k_scale.size(2) == N, "k_scale size mismatch");
    TORCH_CHECK(v_scale.size(2) == N, "v_scale size mismatch");

    auto out = torch::zeros({B, H_Q, 1, D}, q.options());

    int threads = 128;
    dim3 grid(B * H_Q);
    dim3 block(threads);

    float scale = 1.0f / std::sqrt((float)D);

    flash_attn_kernel_decode_int8_kv_cache_gqa<<<grid, block>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        k_int8.data_ptr<int8_t>(),
        v_int8.data_ptr<int8_t>(),
        reinterpret_cast<half*>(k_scale.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_scale.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        B,
        H_Q,
        H_KV,
        N,
        D,
        scale,
        cache_len
    );

    return out;
}

__global__ void token_eviction_kernel(
    const half* __restrict__ K,
    const half* __restrict__ V,
    const float* __restrict__ attn_scores,
    half* __restrict__ K_out,
    half* __restrict__ V_out,
    int* __restrict__ valid_mask,
    int B,
    int H_KV,
    int N,
    int D,
    int num_evict
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    if (bid >= B) return;

    const float* scores_b = attn_scores + bid * N;
    int* mask_b = valid_mask + bid * N;

    __shared__ int s_indices[1024];
    __shared__ float s_scores[1024];

    if (N > 1024) return;

    for (int i = tid; i < N; i += blockDim.x) {
        s_indices[i] = i;
        s_scores[i] = scores_b[i];
    }
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < N - 1; i++) {
            for (int j = 0; j < N - 1 - i; j++) {
                if (s_scores[j] > s_scores[j + 1]) {
                    float tmp_s = s_scores[j];
                    s_scores[j] = s_scores[j + 1];
                    s_scores[j + 1] = tmp_s;
                    int tmp_i = s_indices[j];
                    s_indices[j] = s_indices[j + 1];
                    s_indices[j + 1] = tmp_i;
                }
            }
        }

        for (int i = 0; i < N; i++) {
            mask_b[i] = 1;
        }
        for (int i = 0; i < num_evict && i < N; i++) {
            mask_b[s_indices[i]] = 0;
        }
    }
    __syncthreads();

    for (int h = 0; h < H_KV; h++) {
        const half* K_b = K + (bid * H_KV + h) * N * D;
        const half* V_b = V + (bid * H_KV + h) * N * D;
        half* K_out_b = K_out + (bid * H_KV + h) * N * D;
        half* V_out_b = V_out + (bid * H_KV + h) * N * D;

        int write_idx = 0;
        for (int i = 0; i < N; i++) {
            if (mask_b[i]) {
                for (int d = tid; d < D; d += blockDim.x) {
                    K_out_b[write_idx * D + d] = K_b[i * D + d];
                    V_out_b[write_idx * D + d] = V_b[i * D + d];
                }
                write_idx++;
                __syncthreads();
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> token_eviction_forward(
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor attn_scores,
    int num_evict
) {
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(attn_scores);

    TORCH_CHECK(k.dtype() == torch::kHalf, "k must be float16");
    TORCH_CHECK(v.dtype() == torch::kHalf, "v must be float16");
    TORCH_CHECK(attn_scores.dtype() == torch::kFloat32, "attn_scores must be float32");

    auto B = k.size(0);
    auto H_KV = k.size(1);
    auto N = k.size(2);
    auto D = k.size(3);

    TORCH_CHECK(attn_scores.size(0) == B, "attn_scores batch mismatch");
    TORCH_CHECK(attn_scores.size(1) == N, "attn_scores length mismatch");
    TORCH_CHECK(N <= 1024, "N must be <= 1024 for token eviction");

    auto k_out = torch::zeros_like(k);
    auto v_out = torch::zeros_like(v);
    auto valid_mask = torch::ones({B, N}, torch::TensorOptions().dtype(torch::kInt32).device(k.device()));

    int threads = min((int)N, 256);
    dim3 grid(B);
    dim3 block(threads);

    token_eviction_kernel<<<grid, block>>>(
        reinterpret_cast<half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v.data_ptr<at::Half>()),
        attn_scores.data_ptr<float>(),
        reinterpret_cast<half*>(k_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_out.data_ptr<at::Half>()),
        valid_mask.data_ptr<int>(),
        B,
        H_KV,
        N,
        D,
        num_evict
    );

    return std::make_tuple(k_out, v_out, valid_mask);
}

__global__ void flash_attn_kernel_decode_sliding_window_gqa(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B,
    int H_Q,
    int H_KV,
    int N,
    int D,
    float scale,
    int cache_len,
    int window_size,
    int sink_size
) {
    int bqh = blockIdx.x;
    int b = bqh / H_Q;
    int h_q = bqh % H_Q;
    int h_kv = h_q / (H_Q / H_KV);

    int tid = threadIdx.x;
    int lane_id = tid % 32;

    const half* Q_ptr = Q + ((b * H_Q + h_q) * D);
    const half* K_ptr = K + ((b * H_KV + h_kv) * N * D);
    const half* V_ptr = V + ((b * H_KV + h_kv) * N * D);
    half* O_ptr = O + ((b * H_Q + h_q) * D);

    float m_i = -FLT_MAX;
    float l_i = 0.f;
    float acc[MAX_D];
    for (int d = 0; d < D; ++d) acc[d] = 0.f;

    __shared__ half sQ[MAX_D];
    if (tid < D) {
        sQ[tid] = Q_ptr[tid];
    }
    __syncthreads();

    int effective_end = min(cache_len + 1, N);
    int window_start = max(0, effective_end - window_size);

    constexpr int TILE_K_DECODE = 64;
    int num_kv_blocks = (N + TILE_K_DECODE - 1) / TILE_K_DECODE;

    __shared__ half sK[TILE_K_DECODE][MAX_D];
    __shared__ half sV[TILE_K_DECODE][MAX_D];

    for (int block_idx = 0; block_idx < num_kv_blocks; ++block_idx) {
        int k_start = block_idx * TILE_K_DECODE;
        int k_len = min(TILE_K_DECODE, N - k_start);

        int total_elements = k_len * D;
        for (int idx = tid; idx < total_elements; idx += blockDim.x) {
            int row = idx / D;
            int col = idx % D;
            sK[row][col] = K_ptr[(k_start + row) * D + col];
            sV[row][col] = V_ptr[(k_start + row) * D + col];
        }
        __syncthreads();

        float scores[TILE_K_DECODE];
        float m_block = -FLT_MAX;

        for (int j = 0; j < k_len; ++j) {
            int k_idx = k_start + j;

            bool in_sink = (k_idx < sink_size);
            bool in_window = (k_idx >= window_start && k_idx <= cache_len);
            bool in_range = in_sink || in_window;

            if (!in_range) {
                scores[j] = -FLT_MAX;
                continue;
            }

            float dot = 0.0f;
            #pragma unroll
            for (int d = lane_id; d < D; d += 32) {
                dot += __half2float(sQ[d]) * __half2float(sK[j][d]);
            }

            dot = warp_reduce_sum(dot);

            if (lane_id == 0) {
                dot *= scale;
                scores[j] = dot;
            }

            scores[j] = __shfl_sync(0xffffffff, scores[j], 0);
            m_block = fmaxf(m_block, scores[j]);
        }

        m_block = warp_reduce_max(m_block);
        m_block = __shfl_sync(0xffffffff, m_block, 0);

        if (m_block == -FLT_MAX) {
            __syncthreads();
            continue;
        }

        float m_new = fmaxf(m_i, m_block);
        float alpha = (m_i == -FLT_MAX) ? 0.f : expf(m_i - m_new);

        if (lane_id == 0) {
            float beta_sum = 0.f;
            float weighted[MAX_D];
            for (int d = 0; d < D; ++d) weighted[d] = 0.f;

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

    if (tid == 0) {
        for (int d = 0; d < D; ++d) {
            O_ptr[d] = __float2half(acc[d] / l_i);
        }
    }
}

torch::Tensor flash_attn_forward_decode_sliding_window_gqa(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    int cache_len,
    int window_size,
    int sink_size
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    TORCH_CHECK(q.dtype() == torch::kHalf, "q must be float16");
    TORCH_CHECK(q.size(2) == 1, "q sequence length must be 1 for decode mode");

    auto B = q.size(0);
    auto H_Q = q.size(1);
    auto H_KV = k.size(1);
    auto N = k.size(2);
    auto D = q.size(3);

    TORCH_CHECK(H_Q >= H_KV, "H_Q must be >= H_KV");
    TORCH_CHECK(H_Q % H_KV == 0, "H_Q must be divisible by H_KV");
    TORCH_CHECK(D <= MAX_D, "D must be <= MAX_D(128)");

    auto out = torch::zeros({B, H_Q, 1, D}, q.options());

    int threads = 128;
    dim3 grid(B * H_Q);
    dim3 block(threads);

    float scale = 1.0f / std::sqrt((float)D);

    flash_attn_kernel_decode_sliding_window_gqa<<<grid, block>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        B,
        H_Q,
        H_KV,
        N,
        D,
        scale,
        cache_len,
        window_size,
        sink_size
    );

    return out;
}
