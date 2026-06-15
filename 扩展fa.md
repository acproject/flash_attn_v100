目前这个仓库已经有了 LLM 推理底层注意力/解码加速模块，包括：

FlashAttention prefill
decode kernel
GQA/MQA
continuous batching
paged decode
INT8 KV cache
sliding window / token eviction
speculative verify
一些 tensor parallel 辅助函数
但它还不是一个“完整可推理 LLM 模型的库”。现在更像是 attention serving kernel 库。要变成完整 LLM inference library，还需要补这些模块：

1. 模型结构模块

需要实现完整 Transformer Decoder，而不只是 attention。

包括：

token embedding
RMSNorm / LayerNorm
QKV projection
RoPE / ALiBi 等位置编码
attention block 调用现有 flash attention kernel
MLP / FFN / SwiGLU / GeGLU
residual connection
final norm
LM head
支持不同模型架构配置：Llama、Mistral、Qwen、Yi、Gemma 等
也就是说，需要有类似：

LlamaForCausalLM
LlamaDecoderLayer
LlamaAttention
LlamaMLP
RMSNorm
RotaryEmbedding
当前 example_autoregressive.py 只是简化 demo，还不能加载真实模型权重直接推理。

2. 权重加载模块

要能从真实模型文件加载权重。

需要支持：

HuggingFace safetensors
PyTorch .bin
GGUF，如果想兼容 llama.cpp 生态
权重名映射，比如 HF 的 model.layers.0.self_attn.q_proj.weight
分片模型加载
dtype 转换：FP16、BF16、INT8、INT4
tensor parallel 下的权重切分
这是完整推理库非常关键的一层。没有它，只能跑随机权重 demo。

3. Tokenizer 模块

完整 LLM 推理必须包含 tokenizer。

需要支持：

SentencePiece
BPE tokenizer
HuggingFace tokenizer JSON
special tokens：BOS、EOS、PAD、UNK
chat template
encode / decode
batch encode
streaming decode
否则只能输入 token id，不能输入自然语言 prompt。

4. RoPE / 位置编码模块

现代 LLM 基本都需要 RoPE。

需要支持：

标准 RoPE
Llama RoPE
Qwen RoPE
NTK scaling
YaRN scaling
long context scaling
interleaved / non-interleaved rotary layout
这个模块最好放在 Q/K projection 后、attention 前。

5. KV Cache 管理模块

你现在有 KV cache kernel，但还需要完整 runtime 管理层。

需要：

contiguous KV cache
paged KV cache
block allocator
free list
request 到 block table 的映射
cache append
cache eviction
prefix cache
sliding window cache
batch 内不同请求的 cache lens 管理
decode 时动态扩容/回收
这部分决定它能不能从“单次 demo”变成真正 serving。

6. Sampling / Logits Processor 模块

LLM 生成不只是 forward，还需要采样。

需要支持：

greedy decoding
temperature
top-k
top-p / nucleus sampling
min-p
repetition penalty
frequency penalty
presence penalty
bad words mask
stop tokens
grammar / constrained decoding，可选
logits softmax / argmax CUDA kernel，最好别每步都回 CPU
当前 example 里生成逻辑很简化，离真实可用还差这一层。

7. 推理引擎 Runtime

需要一个统一的 engine 把模型、cache、batching、sampling 串起来。

典型结构：

Engine
Request
Scheduler
Batch
KVCacheManager
ModelRunner
Sampler
它要负责：

接收 prompt
prefill
decode loop
更新 KV cache
调用 attention kernel
logits sampling
判断 EOS / stop
返回 token stream
处理多请求 batch
这是从“模型能 forward”到“库能推理”的核心。

8. Continuous Batching Scheduler

你已经有 continuous batching kernel，但还需要调度器。

需要支持：

请求入队
prefill/decode 分离
decode batch 动态增删
max tokens budget
max batch size
max prefill tokens
请求完成后释放 KV block
长短请求混合调度
chunked prefill
这部分可以参考 vLLM / TensorRT-LLM 的设计。

9. Quantization 模块

目前有 INT8 decode / INT8 KV cache，但完整库还需要权重量化。

建议优先级：

FP16 baseline
INT8 weight-only
INT8 KV cache
GPTQ / AWQ INT4
per-channel scale
group-wise quant
fused dequant + matmul kernel
注意：完整 LLM 推理瓶颈不只在 attention，MLP 和线性层也很重，所以只优化 attention 不够。

10. GEMM / Linear Kernel 模块

这是当前最大缺口之一。

完整 LLM 每层主要计算包括：

QKV projection
O projection
gate/up/down projection
LM head
这些都是 GEMM。现在项目主要是 attention kernel，还缺：

FP16 GEMM
INT8 GEMM
INT4 GEMM
fused bias / activation
fused SwiGLU
tensor parallel linear
batched GEMM 或调用 cuBLASLt
如果不做 GEMM 层，这个库仍然只能当 attention extension 用。

11. 多 GPU / Tensor Parallel 完整实现

现在有 tensor_parallel.py 和一些 scatter/gather/allreduce 辅助，但还需要完整模型级 TP。

包括：

column parallel linear
row parallel linear
attention head sharding
KV head sharding
MLP sharding
vocab parallel LM head
NCCL allreduce/allgather
rank/world size 初始化
多 GPU 权重加载切分
否则大模型放不下或吞吐上不去。

12. API / Server 模块

如果目标是“可推理 LLM 模型的库”，至少需要 Python API。

如果目标是 serving，还需要：

HTTP server
OpenAI-compatible API
streaming SSE
request cancellation
timeout
metrics
health check
async queue
例如：

engine = LLMEngine(model_path="...")
output = engine.generate("你好", max_new_tokens=128)
或者：

curl /v1/chat/completions
13. 测试和 Benchmark 模块

目前 kernel 测试不少，但完整 LLM 还需要：

模型结构 correctness test
与 HuggingFace 输出对齐
tokenizer encode/decode 测试
单 token logits 对齐
prefill/decode 一致性测试
sampling 测试
KV cache correctness
多 batch correctness
多 GPU correctness
端到端吞吐/延迟 benchmark
建议路线

我建议按这个顺序补：

先做 Llama-like 模型结构：RMSNorm、RoPE、Attention、MLP、DecoderLayer、CausalLM。
接入 safetensors 权重加载：先支持 Llama/Qwen 其中一个。
接入 tokenizer 和 generate API：能输入字符串，输出字符串。
完善 KV cache runtime：把现有 attention kernel 真正挂到模型 decode。
补 GEMM/MLP 路径：短期可先用 PyTorch/cuBLAS，后续再写自定义 fused kernel。
做 scheduler + continuous batching runtime：让多请求 serving 跑起来。
再做量化、多 GPU、server API。
一句话总结：
这个项目现在已经有了很不错的 attention 推理内核地基，但完整 LLM 推理库还缺 模型层、权重加载、tokenizer、GEMM/MLP、KV cache runtime、采样、调度器和 serving API。其中最优先的是 真实模型结构 + 权重加载 + tokenizer + generate loop。


### TODO
要正确支持 D=256 需要对 kernel 进行更深入的重构（如分块计算、动态共享内存等），这是一个较大的工程。

sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm