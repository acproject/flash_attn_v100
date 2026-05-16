#include <torch/extension.h>
torch::Tensor flash_attn_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal);

torch::Tensor flash_attn_forward_fp16(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal);

torch::Tensor flash_attn_forward_fp16_warp(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal);

torch::Tensor flash_attn_forward_decode_fp16(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    int cache_len);

torch::Tensor flash_attn_forward_decode_gqa_fp16(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    int cache_len);

torch::Tensor flash_attn_forward_fp16_wmma(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal);

torch::Tensor flash_attn_forward_prefill_gqa_fp16(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal);

torch::Tensor flash_attn_forward_prefill_gqa_fp16_warp(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal);

torch::Tensor flash_attn_forward_paged_decode_gqa_fp16(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int max_num_blocks);

torch::Tensor flash_attn_forward_decode_int8_gqa(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor q_scale,
    torch::Tensor k_scale,
    int cache_len);

torch::Tensor flash_attn_forward_varlen_prefill_fp16(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor cu_seqlens,
    int max_seqlen,
    bool causal);

torch::Tensor flash_attn_forward_decode_gqa_fp16_stream(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    int cache_len,
    int64_t stream_int);

torch::Tensor flash_attn_forward_varlen_prefill_fp16_stream(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor cu_seqlens,
    int max_seqlen,
    bool causal,
    int64_t stream_int);

torch::Tensor flash_attn_forward_continuous_batching_fp16(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor cache_lens,
    bool causal);

void async_h2d_transfer(
    torch::Tensor dst,
    torch::Tensor src,
    int64_t stream_int);

torch::Tensor alloc_pinned_tensor(
    std::vector<int64_t> sizes,
    int dtype_int);

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
    int64_t transfer_stream_int);

std::tuple<torch::Tensor, torch::Tensor> speculative_verify_forward(
    torch::Tensor target_probs,
    torch::Tensor draft_probs,
    torch::Tensor draft_tokens,
    torch::Tensor random_vals,
    int K);

torch::Tensor flash_attn_forward_decode_int8_kv_cache_gqa(
    torch::Tensor q,
    torch::Tensor k_int8,
    torch::Tensor v_int8,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    int cache_len);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> token_eviction_forward(
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor attn_scores,
    int num_evict);

torch::Tensor flash_attn_forward_decode_sliding_window_gqa(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    int cache_len,
    int window_size,
    int sink_size);

void cross_gpu_allreduce_fp16(
    torch::Tensor dst_tensor,
    torch::Tensor src_tensor);

void scatter_heads_forward(
    torch::Tensor full_tensor,
    torch::Tensor local_tensor,
    int total_heads,
    int local_head_start,
    int local_head_count);

void gather_heads_forward(
    torch::Tensor full_tensor,
    torch::Tensor local_tensor,
    int total_heads,
    int local_head_start,
    int local_head_count);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attn_forward, "Flash Attention Forward FP32 (V100)");
    m.def("forward_fp16", &flash_attn_forward_fp16, "Flash Attention Forward FP16 (V100)");
    m.def("forward_fp16_warp", &flash_attn_forward_fp16_warp, "Flash Attention Forward FP16 with Warp Reduction (V100)");
    m.def("forward_decode_fp16", &flash_attn_forward_decode_fp16, "Flash Attention Decode FP16 with KV Cache (V100)");
    m.def("forward_decode_gqa_fp16", &flash_attn_forward_decode_gqa_fp16, "Flash Attention GQA/MQA Decode FP16 with KV Cache (V100)");
    m.def("forward_fp16_wmma", &flash_attn_forward_fp16_wmma, "Flash Attention Forward FP16 with WMMA Tensor Core (V100)");
    m.def("forward_prefill_gqa_fp16", &flash_attn_forward_prefill_gqa_fp16, "Flash Attention Prefill GQA/MQA FP16 (V100)");
    m.def("forward_prefill_gqa_fp16_warp", &flash_attn_forward_prefill_gqa_fp16_warp, "Flash Attention Prefill GQA/MQA FP16 with Warp Reduction (V100)");
    m.def("forward_paged_decode_gqa_fp16", &flash_attn_forward_paged_decode_gqa_fp16, "Flash Attention Paged Decode GQA/MQA FP16 (V100)");
    m.def("forward_decode_int8_gqa", &flash_attn_forward_decode_int8_gqa, "Flash Attention Decode INT8 GQA/MQA (V100)");
    m.def("forward_varlen_prefill_fp16", &flash_attn_forward_varlen_prefill_fp16, "Flash Attention Varlen Prefill FP16 (V100)");
    m.def("forward_decode_gqa_fp16_stream", &flash_attn_forward_decode_gqa_fp16_stream, "Flash Attention GQA Decode FP16 with CUDA Stream (V100)");
    m.def("forward_varlen_prefill_fp16_stream", &flash_attn_forward_varlen_prefill_fp16_stream, "Flash Attention Varlen Prefill FP16 with CUDA Stream (V100)");
    m.def("forward_continuous_batching_fp16", &flash_attn_forward_continuous_batching_fp16, "Flash Attention Continuous Batching FP16 (V100)");
    m.def("async_h2d_transfer", &async_h2d_transfer, "Async Host-to-Device Transfer");
    m.def("alloc_pinned_tensor", &alloc_pinned_tensor, "Allocate Pinned Memory Tensor");
    m.def("pipeline_decode_step", &pipeline_decode_step, "Pipeline Decode Step (compute + transfer overlap)");
    m.def("speculative_verify", &speculative_verify_forward, "Speculative Decoding Verify (GPU acceptance/rejection)");
    m.def("forward_decode_int8_kv_cache_gqa", &flash_attn_forward_decode_int8_kv_cache_gqa, "Flash Attention Decode with INT8 KV Cache GQA/MQA (V100)");
    m.def("token_eviction", &token_eviction_forward, "Token Eviction based on attention scores");
    m.def("forward_decode_sliding_window_gqa", &flash_attn_forward_decode_sliding_window_gqa, "Flash Attention Decode Sliding Window + Attention Sink GQA/MQA (V100)");
    m.def("cross_gpu_allreduce_fp16", &cross_gpu_allreduce_fp16, "Cross GPU AllReduce FP16");
    m.def("scatter_heads", &scatter_heads_forward, "Scatter heads for tensor parallelism");
    m.def("gather_heads", &gather_heads_forward, "Gather heads for tensor parallelism");
}
