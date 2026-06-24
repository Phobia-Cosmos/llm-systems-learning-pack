# LLM Systems Learning Pack

这个资料包覆盖你列出的方向：模型训练与对齐、Transformer 基础、KV cache 和推理服务、并行训练、MoE/MHA 变体、CUDA 算子与低精度、通信硬件、2025-2026 前沿论文，以及可读代码项目。

当前状态：

- 论文清单：`paper_list.tsv`，共 135 条。
- 已下载 PDF：`papers/` 下 134 个，已用 `pdfinfo` 校验。
- 唯一未下载：AMD CDNA3 官方白皮书，AMD 站点多次返回 HTTP 524；官方链接保留在 `paper_list.tsv` 和 `PAPER_INDEX.md`。
- 代码项目：`projects/minimind`、`projects/vllm`、`projects/sglang`、`projects/LLaMA-Factory`、`projects/verl`、`projects/trl`、`projects/Megatron-LM`、`projects/flash-attention`。

## 先建立总图

大模型系统可以按三层理解：

1. 模型层：decoder-only Transformer、attention、MLP、Norm、RoPE、MHA/MQA/GQA/MLA、MoE。
2. 训练层：pretrain、SFT、DPO/RLHF/RL、并行策略、通信、checkpoint、低精度。
3. 推理层：KV cache、prefill/decode、continuous batching、chunked prefill、PD 分离、投机解码、服务调度、CUDA/Triton/CUTLASS 算子。

你的学习目标如果是“向 LLM systems / inference engine / training infra 靠齐”，不要只读模型论文；至少要把模型结构、GPU 内存层次、并行通信、服务调度这四条线同时推进。

## 核心概念速记

Pretrain 是在大规模无标注语料上做 next-token prediction，学语言和知识；SFT 是用指令-答案数据把基座模型调成会按任务回答；DPO 是用偏好对直接优化策略模型，避免显式训练 reward model，是 RLHF 的一种更简单替代路线。MiniMind 是一个适合入门的“小模型全流程”代码项目，可以看 tokenizer、pretrain、full SFT、LoRA、DPO、PPO/GRPO、MoE 的最小实现。

Decoder-only Transformer 是 GPT/LLaMA 这类自回归结构：输入 token 只能看左侧历史 token，用 causal self-attention 逐 token 预测下一个 token。典型 decode layer 是：RMSNorm -> self-attention -> residual -> RMSNorm -> MLP/SwiGLU -> residual。现代 LLM 多用 Pre-Norm；Post-Norm 是早期 Transformer 常见结构，深层训练更不稳定。

Attention 公式：

```text
Q = X Wq, K = X Wk, V = X Wv
Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k) + mask) V
```

MHA 是多个 Q/K/V head 并行；MQA 让多个 query head 共享一组 KV，降低 KV cache；GQA 把 query head 分组共享 KV，是 MHA 和 MQA 的折中；MLA 用低秩 latent KV 压缩 KV cache，是 DeepSeek-V2/V3 的关键设计之一。

RMSNorm 只按均方根缩放，不减均值；LayerNorm 减均值再除标准差；BatchNorm 跨 batch 统计，序列模型和变长推理里不如 LayerNorm/RMSNorm 合适。Norm 的作用是稳定激活尺度和梯度传播，让深层网络更容易训练。

KV cache 是推理时保存历史 token 的 K/V，避免每生成一个 token 都重算所有历史。常见方案包括连续内存 KV、PagedAttention 的分页 KV、prefix/radix cache 的共享前缀复用、滑动窗口 KV、KV offload、KV quantization、PD 分离时跨节点传输 KV。

Continuous batching 是服务端按 token 迭代动态合批；chunked prefill 把长 prompt 的 prefill 切块，避免长请求阻塞 decode；PD 分离把 prefill 和 decode 放到不同 GPU/节点；投机解码用 draft model/多头预测多个候选 token，再由 target model 验证；MTP 是 multi-token prediction；EPLB 通常指 Expert Parallel Load Balancing，用于 MoE 专家负载均衡。

MoE 由 router/gating network 给 token 选择 top-k experts，常见约束是容量、负载均衡 loss、expert parallel 和 all-to-all 通信。MoE 计算量稀疏，但通信和调度复杂。

CUDA C++ 算子就是直接写 GPU kernel 或调用 CUTLASS/CUDA primitives 实现矩阵乘、attention、Norm、RoPE、cache 写入、MoE dispatch/combine 等。节省空间和提升利用率的核心手段是 kernel fusion、减少 HBM 读写、共享内存/寄存器复用、coalesced access、向量化 load/store、online softmax、低精度、KV cache 压缩和分页管理。

NCCL 是 NVIDIA GPU collective 通信库；RDMA 是绕过 CPU 让网卡直接访问远端内存；InfiniBand 是 HPC/AI 集群常用低延迟高带宽网络；RoCE 是基于以太网的 RDMA。A 卡通常指 AMD GPU；H 卡在不同语境里可能指华为昇腾，也可能指 NVIDIA H100/H200/H20 系列，具体要看集群环境。

## 推荐学习顺序

基础模型结构要从前往后学：Attention is All You Need -> GPT-2/GPT-3 -> LLaMA/LLaMA2/LLaMA3 -> DeepSeek-V2/V3。原因是后面的 MLA、MoE、RMSNorm、RoPE、SwiGLU 都是在基础 Transformer 上的工程化演进。

训练与对齐先前后结合：先读 Scaling Laws、Chinchilla、InstructGPT、DPO，建立 pretrain/SFT/RLHF/DPO 主线；再读 DeepSeek-R1、Kimi k1.5、LIMO、Qwen3 等 2025 前沿。对齐方向论文更新快，最新论文适合“先看结论，再回补基础”。

KV cache 和推理服务建议从前往后：Orca -> vLLM/PagedAttention -> SGLang/RadixAttention -> Sarathi/Splitwise/DistServe/Mooncake -> 2025-2026 的 Jenga、vLLM-Omni、KVServe。这个方向强依赖系统问题演化，按时间线最清楚。

CUDA/算子建议从底往上：先写 naive softmax/matmul/RMSNorm，再读 online softmax、FlashAttention、FlashAttention-2/3、SageAttention、KIVI/TurboQuant，最后看 vLLM/SGLang/FlashAttention 源码。只读论文不写 kernel，很难理解 HBM traffic 为什么是瓶颈。

并行训练建议从概念到系统：Data Parallel -> Tensor Parallel -> Pipeline Parallel -> ZeRO/FSDP -> Sequence/Context Parallel -> Expert Parallel -> MegaScale/MegaScale-MoE。这个方向需要配合通信原语一起学。

通信硬件建议从硬件和 collective 原语开始：PCIe/NVLink/NVSwitch/IB/RDMA -> all-reduce/all-gather/reduce-scatter/all-to-all -> NCCL/Blink/TACCL -> 大模型训练系统。这里要从底层往上。

2025-2026 前沿建议从后往前：先读最新技术报告/系统论文的摘要、架构图和实验结论，找出它解决的瓶颈，再回到对应基础论文。前沿论文通常默认你已经知道 PagedAttention、MoE、FlashAttention、RLHF 等背景。

## 代码阅读入口

MiniMind：

- `projects/minimind/model/model_minimind.py`：RMSNorm、Attention、DecoderLayer、MoE。
- `projects/minimind/trainer/train_pretrain.py`：pretrain。
- `projects/minimind/trainer/train_full_sft.py`：SFT。
- `projects/minimind/trainer/train_dpo.py`：DPO loss 和训练 loop。
- `projects/minimind/trainer/train_lora.py`、`model/model_lora.py`：LoRA。

vLLM：

- `projects/vllm/vllm/v1/core/kv_cache_manager.py`、`block_pool.py`、`kv_cache_utils.py`：PagedAttention/KV block 管理。
- `projects/vllm/vllm/v1/core/sched/scheduler.py`：continuous batching 和调度。
- `projects/vllm/vllm/config/scheduler.py`、`config/speculative.py`、`config/cache.py`：服务端配置。
- `projects/vllm/vllm/v1/attention/ops/paged_attn.py`、`chunked_prefill_paged_decode.py`：attention/KV 算子入口。
- `projects/vllm/vllm/distributed/kv_transfer/`：PD 分离和 KV transfer。

SGLang：

- `projects/sglang/python/sglang/srt/mem_cache/radix_cache.py`、`base_prefix_cache.py`、`chunk_cache.py`：prefix/radix cache。
- `projects/sglang/python/sglang/srt/managers/scheduler.py`：服务调度。
- `projects/sglang/python/sglang/srt/disaggregation/prefill.py`、`decode.py`：PD 分离。
- `projects/sglang/python/sglang/srt/layers/radix_attention.py`：RadixAttention。
- `projects/sglang/python/sglang/srt/layers/attention/linear/lightning_attn.py`：Lightning Attention 相关实现。

Megatron-LM：

- `projects/Megatron-LM/megatron/core/tensor_parallel/`：TP 通信映射和并行线性层。
- `projects/Megatron-LM/megatron/core/pipeline_parallel/`：PP。
- `projects/Megatron-LM/megatron/core/parallel_state.py`：DP/TP/PP/EP/CP 进程组。
- `projects/Megatron-LM/megatron/core/transformer/attention.py`：训练侧 attention。
- `projects/Megatron-LM/megatron/core/transformer/moe/`：MoE/EP。

FlashAttention：

- `projects/flash-attention/csrc/flash_attn/src/softmax.h`：online softmax 相关实现。
- `projects/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`：Ampere 路径。
- `projects/flash-attention/hopper/flash_fwd_kernel_sm90.h`、`hopper/paged_kv.h`：Hopper/paged KV。
- `projects/flash-attention/flash_attn/ops/rms_norm.py`、`csrc/layer_norm/`：Norm/fused op。

训练/对齐框架：

- `projects/LLaMA-Factory`：工程化 SFT/LoRA/DPO/PPO/导出流程。
- `projects/trl`：Hugging Face DPO/PPO/GRPO 等 trainer。
- `projects/verl`：大规模 RLHF/RL rollout 和训练编排。

## 每个方向的论文目录

- Transformer/Norm/Attention：`papers/01_transformer_norm_attention`
- Pretrain/SFT/DPO/RLHF：`papers/02_pretrain_sft_alignment`
- KV cache/推理服务：`papers/03_kv_cache_serving_inference`
- 并行训练系统：`papers/04_parallel_training_systems`
- MoE/MHA/MLA/Lightning/Sparse Attention：`papers/05_moe_mha_variants`
- CUDA/FlashAttention/低精度/量化：`papers/06_cuda_kernels_precision`
- 通信与硬件：`papers/07_communication_hardware`
- 2025-2026 前沿：`papers/08_2025_2026_frontier`
- 综述：`papers/09_surveys`

完整索引见 `PAPER_INDEX.md`。

## 12 周实践路线

第 1-2 周：读 `01_transformer_norm_attention` 中 Attention、RMSNorm、Pre-LN/Post-LN、LLaMA；在 MiniMind 里画出 forward 数据流，手算一个小 attention。

第 3-4 周：跑 MiniMind 的 pretrain/SFT/DPO 小数据流程，理解 loss、mask、label shift、LoRA 参数冻结。对应读 InstructGPT、DPO、LoRA、QLoRA、LLaMA-Factory。

第 5-6 周：读 vLLM 和 SGLang，重点看 KV block、prefix/radix cache、scheduler。用小模型对比普通 Transformers generate、vLLM、SGLang 的吞吐和延迟。

第 7-8 周：读 FlashAttention 和 online softmax，实现一个 PyTorch naive attention、一个 tiled attention，再对照 FlashAttention 源码理解为什么 IO-aware。

第 9 周：读 MQA/GQA/MLA/MoE，修改 MiniMind 的 attention head 配置或 MoE top-k，观察 KV cache 大小、参数量、速度变化。

第 10 周：读 Megatron、ZeRO、Alpa、MegaScale；画出 TP/DP/PP/EP/CP 的通信图，写出每一步需要 all-reduce、all-gather、reduce-scatter 还是 all-to-all。

第 11 周：读 NCCL/RDMA/硬件论文和白皮书，理解 GPU、HBM、PCIe、NVLink、IB/RoCE 的瓶颈边界；用 `nvidia-smi topo -m` 或集群拓扑图做一次拓扑分析。

第 12 周：读 `08_2025_2026_frontier`，选一个主题复现小实验：比如 KV quantization、PD 分离、speculative decoding、MLA decode、MoE expert load balance。

## 如何与模型互动

读论文时不要问“总结这篇论文”，而是带着任务问：

```text
我正在读 vLLM/PagedAttention。请只根据论文解释：
1. 它解决的具体内存碎片问题是什么？
2. block table 在 serving 中承担什么角色？
3. prefill 和 decode 阶段的瓶颈分别是什么？
4. 对应到 vllm/v1/core/kv_cache_manager.py，我应该重点看哪些函数？
```

看代码时让模型做“约束式解释”：

```text
下面是 MiniMind 的 Attention.forward。请按 tensor shape 逐行解释，
指出哪里产生 Q/K/V，哪里使用 RoPE，哪里写 KV cache，
不要泛泛解释 Transformer。
```

做实验时让模型先给可验证假设：

```text
我想比较 MHA、GQA、MQA 的 KV cache 显存。请给出公式、最小实验脚本结构、
需要记录的指标，以及预期结果。如果实际结果相反，应该排查哪些实现问题？
```

追前沿时让模型反向补基础：

```text
我读 DeepSeek-V3 里的 MLA/MTP/EPLB 不清楚。请把每个词映射到：
1. 依赖的基础论文；
2. 对应的系统瓶颈；
3. 可读的开源代码位置；
4. 一个最小复现实验。
```

这种互动方式会比让模型给大而全教程有效，因为它会把论文、代码和实验绑在一起。

## 注意事项

2025-2026 论文中不少仍是 arXiv 技术报告或刚发表的系统工作，不要把它们都当成已经完成多年检验的“定论”。阅读时优先看问题定义、瓶颈分析、实验设置和消融，不要只记新名词。

“全部论文”没有严格边界。这个包按你列出的关键词做了覆盖式代表集合：每个方向都有基础论文、系统论文或近两年前沿论文。后续新增论文时，把条目加到 `paper_list.tsv`，再运行 `./generate_paper_index.sh` 更新索引。
