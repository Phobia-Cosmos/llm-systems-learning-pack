# SGLang, FlashInfer, CUDA Graph 与 Serving 笔记

更新时间：2026-07-02

这份笔记把当前本地环境里的 SGLang/vLLM/nano-vLLM 实验、FlashInfer attention/sampler、CUDA graph、Hugging Face 模型格式、logits 与采样、serving scheduler 串起来。它是学习索引，不替代完整论文阅读。

## 当前本地结论

同一台 RTX 4070 SUPER、同一个 Qwen3-0.6B 模型、同一组 3 个 prompt x 128 output tokens 的本地实验中，结果记录在 `benchmarks/qwen_compare/FULL_BENCHMARK_RESULTS_2026_07_01.md`：

| 路径 | 完整配置 | 输出吞吐 |
| --- | --- | ---: |
| vLLM | FlashAttention v2 + FlashInfer sampler | 725.34 tok/s |
| SGLang | FlashInfer attention + FlashInfer sampler + CUDA graph | 497.69 tok/s |
| nano-vLLM | FlashAttention + CUDA graph | 406.45 tok/s |
| Transformers | Hugging Face `generate()` baseline | 239.67 tok/s |

这个结果的主要意义不是排名，而是说明 serving engine 的优化点在哪里：更高效的 attention/KV cache 管理、持续批处理、采样 kernel、CUDA graph 复用，以及更好的调度策略。

## FlashInfer Attention, Sampler, CUDA Graph 是什么

FlashInfer attention 是面向 LLM 推理的 GPU kernel 后端，重点解决 decode 阶段对 KV cache 的高频访问。传统 attention 每一步都要把当前 token 的 query 和历史 key/value 做计算；serving 场景下历史 KV 不再是一个简单连续大矩阵，而是由很多并发请求、不同长度、分页 KV cache 组成。FlashInfer 的作用是在这种 paged/ragged KV 形状下尽量把访存、索引、softmax、V 加权合并成高效 kernel。

FlashInfer sampler 是把 logits 后处理和采样放到更适合 GPU 的路径里。模型最后一层输出的是 logits，后面通常要做 temperature、top-k、top-p、softmax、随机抽样。小 batch 时这些操作看起来不大，但在服务中每个 decode step 都会执行一次，且每个请求的 sampling 参数可能不同。FlashInfer sampler 的价值是减少 CPU/PyTorch 逐步调度和小 kernel 开销。

CUDA graph 是 CUDA 提供的“捕获并重放一串 GPU 操作”的机制。LLM decode 阶段经常反复执行类似的图：embedding/attention/MLP/lm_head/sampling。如果 batch bucket、tensor shape、显存地址、控制流足够稳定，就可以先 capture，后续 replay，减少 Python、PyTorch dispatcher、CUDA launch 的固定开销。

“稳定形状下的一串 GPU 操作”指的是：同一组 kernel launch 顺序、相同或兼容的 tensor shape、可复用的输入输出 buffer 地址、相同的执行分支。例如 SGLang 可以对 batch size 1/2/4/8 这类 bucket 分别 capture；真实请求不足 bucket 时用 padding 或元数据控制，让 replay 的形状保持稳定。

## 为什么要保存为 Hugging Face 格式

Hugging Face 格式本质上是一个标准模型目录约定，常见内容包括：

| 文件 | 作用 |
| --- | --- |
| `config.json` | 模型结构、hidden size、层数、head 数、rope 参数等 |
| `model.safetensors` 或分片 | 权重本体，通常比 pickle 风格的 `.bin` 更安全 |
| `tokenizer.json`、`tokenizer_config.json` | tokenizer 的词表、规则、特殊 token |
| `generation_config.json` | 默认生成参数，如 eos、pad、temperature 等 |
| `modeling_*.py` 等可选文件 | `trust_remote_code` 模型的自定义结构代码 |

保存为 HF 格式的主要原因是互操作性。Transformers 可以 `AutoModel.from_pretrained()`，vLLM/SGLang 可以直接读取同一目录，训练脚本、量化脚本、模型上传下载工具也能共享同一套元数据。对学习 AI infra 来说，这相当于把“模型资产”从某个训练仓库里解耦出来。

在 HF 格式普及前，常见保存方式包括：

| 旧格式 | 问题 |
| --- | --- |
| PyTorch `.pt/.pth` 或 `state_dict` | 只保存参数名和 tensor，结构、tokenizer、生成配置经常靠代码约定 |
| TensorFlow checkpoint | 依赖 TF 变量名和图结构 |
| JAX/Flax params | 依赖 JAX tree 结构和加载代码 |
| Megatron-LM/DeepSpeed ZeRO 分片 | 训练并行友好，但服务前经常要 merge/convert |
| fairseq、research repo 自定义 checkpoint | 每个项目各有转换脚本，迁移成本高 |

所以“保存为 Hugging Face 格式”不是为了训练本身更快，而是为了让模型能被训练、评测、量化、serving、发布系统共同识别。

## 普通 PyTorch Attention 是什么

普通 PyTorch attention 可以理解为直接用 PyTorch 张量算子表达 scaled dot-product attention：

```python
scores = q @ k.transpose(-2, -1) / sqrt(d_head)
scores = scores + mask
probs = softmax(scores, dim=-1)
out = probs @ v
```

PyTorch 也提供 `torch.nn.functional.scaled_dot_product_attention`，它会根据硬件和输入选择 math、memory efficient、FlashAttention 等实现。但在 serving engine 语境里，“普通 PyTorch attention”通常指不使用 vLLM/SGLang 的 paged KV cache、专用 decode kernel、调度器和 CUDA graph 的路径。

区别在这里：

| 路径 | 适合场景 | 局限 |
| --- | --- | --- |
| 手写 PyTorch attention | 教学、baseline、简单实验 | 小 kernel 多，KV 管理弱，并发服务效率低 |
| PyTorch SDPA | 单次 forward 更优化 | 不等同于完整 serving engine |
| FlashInfer/vLLM/SGLang attention | 多请求 decode、paged KV、低延迟 serving | 环境和 shape/缓存管理更复杂 |

## Logits, Temperature, Top-p, Top-k

模型最后一层通常先得到最后 token 的 hidden state `h`，再经过语言模型头：

```text
logits = h @ W_vocab^T + b
```

`logits` 是词表上每个 token 的未归一化分数，shape 通常是 `[batch, vocab_size]` 或 `[batch, seq, vocab_size]`。它还不是概率；经过 softmax 之后才是概率分布。

Temperature、top-p、top-k 是采样前对 logits/probability 分布做控制：

| 参数 | 公式/行为 | 作用 |
| --- | --- | --- |
| temperature | `prob = softmax(logits / T)` | `T < 1` 更确定，`T > 1` 更发散 |
| top-k | 只保留概率最高的 k 个 token | 限制候选空间，避免低概率 token |
| top-p | 保留累计概率达到 p 的最小 token 集 | 动态候选空间，也叫 nucleus sampling |

如果不做这些控制，直接 greedy 或纯 softmax sample 要么太死板，要么容易采到长尾低质量 token。服务系统把这些步骤做成 sampler，是因为它们每个 decode step 都会执行，吞吐和延迟都会受影响。

## SGLang Attention 后端

当前本地 `.venv-sglang` 的 SGLang 版本是 0.5.9，`server_args.py` 中 `ATTENTION_BACKEND_CHOICES` 的实际值如下：

| 后端 | 类型 | 备注 |
| --- | --- | --- |
| `triton` | 通用 GPU | SGLang 自带 Triton attention 路径 |
| `torch_native` | PyTorch | 更接近普通 PyTorch 后端，通常用于兼容/调试 |
| `flex_attention` | PyTorch FlexAttention | 依赖 PyTorch FlexAttention 能力 |
| `nsa` | Sparse attention | Native Sparse Attention 相关 |
| `cutlass_mla` | NVIDIA/MLA | 面向 MLA 模型的 CUTLASS 后端 |
| `fa3` | NVIDIA | FlashAttention-3 后端 |
| `fa4` | NVIDIA | FlashAttention-4 后端 |
| `flashinfer` | NVIDIA/通用推理 | 本机 Qwen3-0.6B benchmark 使用的完整后端 |
| `flashmla` | NVIDIA/MLA | MLA 相关 FlashMLA 后端 |
| `trtllm_mla` | NVIDIA/TensorRT-LLM | MLA 模型后端 |
| `trtllm_mha` | NVIDIA/TensorRT-LLM | MHA 模型后端 |
| `dual_chunk_flash_attn` | 长上下文 | dual chunk attention 相关 |
| `aiter` | AMD | AMD AITER 后端 |
| `wave` | AMD | AMD Wave 后端 |
| `intel_amx` | Intel CPU | AMX 后端 |
| `ascend` | Ascend NPU | 华为 NPU 后端 |
| `intel_xpu` | Intel XPU | Intel XPU 后端 |

采样后端 `SAMPLING_BACKEND_CHOICES` 是：`flashinfer`、`pytorch`、`ascend`。

对当前这台 NVIDIA RTX 4070 SUPER 来说，学习优先级建议是：

1. `flashinfer` attention + `flashinfer` sampler + CUDA graph：完整 SGLang 推理路径。
2. `flashinfer` attention + `flashinfer` sampler + 禁用 CUDA graph：观察 CUDA graph 的收益。
3. `triton` attention + `pytorch` sampler：观察非 FlashInfer 路径。
4. `torch_native`：作为兼容/理解 baseline，不作为性能目标。

## Serving Engine Scheduler 策略来源

SGLang 当前源码 `schedule_policy.py` 中的策略分两类：

| 策略 | 含义 | 来源/关联 |
| --- | --- | --- |
| `fcfs` | first come first serve，先来先服务 | 经典队列基线，不是某篇 LLM 论文专属 |
| `lof` | longest output first，优先预计输出更长的请求 | SGLang 实现里的启发式策略 |
| `random` | 随机排序 | baseline/消融用 |
| `routing-key` | 按 running batch 中 routing key 频率排序 | 面向应用路由/缓存亲和性的工程策略 |
| `lpm` | longest prefix match，优先最长 prefix cache 命中 | 与 SGLang/RadixAttention 的 prefix cache 机制相关 |
| `dfs-weight` | 基于 radix tree 的 DFS 权重排序 | 与 prefix tree/cache-aware batching 相关 |
| priority scheduling | 在部分策略上叠加请求 priority | 通用优先级调度思想，SGLang 工程实现 |

这些 scheduler 不是每一个都来自一篇独立论文。更准确的论文脉络是：

| 论文/系统 | 和 scheduler 的关系 |
| --- | --- |
| Orca | 提出面向生成式模型服务的 iteration-level scheduling/continuous batching 思路 |
| vLLM/PagedAttention | 提出 paged KV cache 管理，支撑高效连续批处理 |
| SGLang/RadixAttention | 用 radix tree 管理 prefix cache，让相同前缀请求更容易复用 KV |
| FlashAttention/FlashInfer | 主要是 attention/kernel 层优化，不直接定义请求调度策略 |

因此学习时可以这样分层：Orca 解释为什么需要连续批处理；vLLM 解释为什么 KV cache 要分页；SGLang 解释为什么 prefix/cache-aware scheduling 重要；FlashInfer/FlashAttention 解释单个 attention/sampling kernel 如何更快。

## `scripts/use_sglang.sh` 环境变量标注

脚本路径：`scripts/use_sglang.sh`

下面是 2026-07-02 在 `/home/undefined/Desktop/ai` 下执行 `source scripts/use_sglang.sh` 后的实际值。

| 变量 | 当前值 | 作用 |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0` | 只让 SGLang 看到第 0 张 GPU |
| `TOKENIZERS_PARALLELISM` | `false` | 关闭 tokenizer 多线程提示/竞争，减少日志干扰 |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | 让 PyTorch CUDA allocator 使用可扩展 segment，缓解显存碎片 |
| `HF_HOME` | `/home/undefined/Desktop/ai/.model_cache/huggingface` | Hugging Face 模型和缓存目录；当前是指向 Disk 的软链接 |
| `HF_XET_HIGH_PERFORMANCE` | `1` | 开启 Hugging Face Xet 下载高性能路径 |
| `MODELSCOPE_CACHE` | `/home/undefined/Desktop/ai/.model_cache/modelscope` | ModelScope 缓存目录 |
| `FLASHINFER_WORKSPACE_BASE` | `/home/undefined/Disk/cache/flashinfer-system-cuda-release` | FlashInfer JIT 编译缓存目录，避免复用错误 CUDA header/cache |
| `SGLANG_VENV` | `/home/undefined/Desktop/ai/.venv-sglang` | SGLang Python 虚拟环境路径；当前是指向 Disk 的软链接 |
| `CUDA_HOME` | `/usr/local/cuda-13.0` | 完整系统 CUDA Toolkit 路径，提供 nvcc/header/lib |
| `FLASHINFER_NVCC` | `/usr/local/cuda-13.0/bin/nvcc` | FlashInfer JIT 编译扩展时显式使用的 nvcc |
| `CUDA_LIB_PATH` | `/usr/local/cuda-13.0/lib64` | 脚本内部辅助变量，不是 `export` 变量；用于拼接 `LD_LIBRARY_PATH` |
| `LD_LIBRARY_PATH` | `/usr/local/cuda-13.0/lib64:/home/undefined/Desktop/ai/.venv-sglang/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/home/undefined/Desktop/ai/.venv-sglang/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/undefined/Desktop/ai/.venv-sglang/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/cuda-13.0/lib64:` | 动态库搜索路径，确保系统 CUDA 和 pip NVIDIA wheel 的运行库能被找到 |
| `PATH` | 见下方完整值 | 命令搜索路径，确保当前 Python 来自 `.venv-sglang`，`nvcc` 来自系统 CUDA |

`PATH` 的完整值：

```text
/home/undefined/Desktop/ai/.venv-sglang/bin:/usr/local/cuda-13.0/bin:/home/undefined/.local/bin:/usr/local/cuda-13.0/bin:/home/undefined/.nvm/versions/node/v24.13.0/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/codex-path:/home/undefined/.local/bin:/home/undefined/.codex/tmp/arg0/codex-arg0Oq4gzC:/home/undefined/.nvm/versions/node/v24.13.0/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/codex-path:/home/undefined/.local/bin:/home/undefined/.nvm/versions/node/v24.13.0/bin:/home/undefined/.opencode/bin:/opt/riscv/bin:/home/undefined/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin
```

实际开头最关键：`.venv-sglang/bin` 在前，`/usr/local/cuda-13.0/bin` 也在前。这样 `python` 会进入 SGLang 虚拟环境，`nvcc` 会使用系统 CUDA Toolkit。

脚本里的这段判断：

```bash
if [ "$CUDA_HOME" = "$SGLANG_VENV/lib/python3.12/site-packages/nvidia/cu13" ]; then
  ...
fi
```

是在系统没有 `/usr/local/cuda-13.0` 时才走的 fallback。pip 安装的 NVIDIA CUDA wheel 有时目录叫 `lib` 而不是传统 Toolkit 的 `lib64`，运行库也可能只有 `libcudart.so.13` 而没有通用名字 `libcudart.so`。这段逻辑通过软链接补齐 JIT 编译器/链接器常见的查找路径。当前机器已经有完整系统 CUDA 13.0，所以这段 fallback 不会生效。

## SGLang 复现命令

离线 Python Engine 路径：

```bash
cd /home/undefined/Desktop/ai
source scripts/use_sglang.sh
python benchmarks/qwen_compare/run_sglang_full.py
```

对比 attention/sampler/CUDA graph：

```bash
source scripts/use_sglang.sh
python benchmarks/qwen_compare/run_sglang_variant.py --attention-backend flashinfer --sampling-backend flashinfer
python benchmarks/qwen_compare/run_sglang_variant.py --attention-backend flashinfer --sampling-backend flashinfer --disable-cuda-graph
python benchmarks/qwen_compare/run_sglang_variant.py --attention-backend triton --sampling-backend pytorch --disable-cuda-graph
```

OpenAI-compatible server 路径：

```bash
source scripts/use_sglang.sh
python -m sglang.launch_server \
  --model-path /home/undefined/Desktop/ai/.model_cache/huggingface/Qwen3-0.6B \
  --trust-remote-code \
  --context-length 2048 \
  --mem-fraction-static 0.45 \
  --attention-backend flashinfer \
  --sampling-backend flashinfer \
  --host 127.0.0.1 \
  --port 30000
```

SGLang 主要是 serving/inference engine，不是常规预训练框架。训练一个小模型通常用 Transformers、Megatron-LM、DeepSpeed、FSDP、torchtune 等；RLHF/RL rollout 阶段可以让 SGLang/vLLM 负责高吞吐生成，训练器如 verl/TRL 负责优化。

## 学习顺序

1. 先用 Transformers 跑通 `model.generate()`，理解 logits、attention、KV cache。
2. 用 nano-vLLM 看最小化 serving engine：paged KV、prefill/decode、CUDA graph。
3. 用 vLLM 看生产级 PagedAttention、continuous batching、OpenAI API server。
4. 用 SGLang 看 FlashInfer、RadixAttention/prefix cache、structured generation、scheduler。
5. 再向下看 CUDA/Triton kernel：softmax、RMSNorm、GEMM、FlashAttention、sampling。
6. 最后把训练系统接上：Megatron/DeepSpeed/FSDP 负责训练，vLLM/SGLang 负责推理或 RL rollout。

## 参考来源

- SGLang local source: `.venv-sglang/lib/python3.12/site-packages/sglang/srt/server_args.py`
- SGLang local source: `.venv-sglang/lib/python3.12/site-packages/sglang/srt/layers/attention/attention_registry.py`
- SGLang local source: `.venv-sglang/lib/python3.12/site-packages/sglang/srt/managers/schedule_policy.py`
- SGLang paper: https://arxiv.org/abs/2312.07104
- vLLM/PagedAttention paper: https://arxiv.org/abs/2309.06180
- Orca OSDI 2022: https://www.usenix.org/conference/osdi22/presentation/yu
- Nucleus sampling/top-p: https://arxiv.org/abs/1904.09751
- Hugging Face Transformers model API: https://huggingface.co/docs/transformers/en/main_classes/model
- safetensors documentation: https://huggingface.co/docs/safetensors/en/index
- PyTorch scaled dot-product attention: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- NVIDIA CUDA Graphs programming guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
- FlashInfer documentation: https://docs.flashinfer.ai/
