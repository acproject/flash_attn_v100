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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attn_forward, "Flash Attention Forward FP32 (V100)");
    m.def("forward_fp16", &flash_attn_forward_fp16, "Flash Attention Forward FP16 (V100)");
    m.def("forward_fp16_warp", &flash_attn_forward_fp16_warp, "Flash Attention Forward FP16 with Warp Reduction (V100)");
    m.def("forward_decode_fp16", &flash_attn_forward_decode_fp16, "Flash Attention Decode FP16 with KV Cache (V100)");
    m.def("forward_decode_gqa_fp16", &flash_attn_forward_decode_gqa_fp16, "Flash Attention GQA/MQA Decode FP16 with KV Cache (V100)");
    m.def("forward_fp16_wmma", &flash_attn_forward_fp16_wmma, "Flash Attention Forward FP16 with WMMA Tensor Core (V100)");
}
