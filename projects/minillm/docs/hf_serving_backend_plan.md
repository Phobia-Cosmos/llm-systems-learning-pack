# MiniLLM, Hugging Face 格式与推理引擎接入计划

## 1. 什么是 decoder-only Transformer

Transformer 不只有一种。常见有三类：

| 类型 | 注意力方式 | 代表任务 | 代表模型 |
| --- | --- | --- | --- |
| encoder-only | 双向 self-attention，可以看左右上下文 | 理解、分类、检索 embedding | BERT、RoBERTa |
| encoder-decoder | encoder 理解输入，decoder 自回归生成输出 | 翻译、摘要、seq2seq | 原始 Transformer、T5、BART |
| decoder-only | causal self-attention，只看当前位置之前 | 文本生成、ChatGPT 类模型 | GPT、Llama、Qwen、Mistral |

MiniLLM 是 decoder-only，因为它的训练目标是 next-token prediction：给定前面的 token，预测下一个 token。

## 2. 为什么训练少时输出随机

MiniLLM 的权重一开始是随机初始化。训练步数少时，模型还没学会这些东西：

- 语料中哪些字符经常跟在一起。
- `用户:` 后面通常是问题，`助手:` 后面通常是回答。
- attention 应该从前文哪些位置取信息。
- lm_head 应该把隐藏状态映射到哪个 token。

所以早期 logits 接近随机分数，采样出来就像随机字符。让输出更有意义，至少需要：更清晰的数据、更多 step、足够模型容量、较稳定解码方式。对这个 toy char-level corpus，建议先用 `500-1500` step 观察格式；想明显记住问答内容，用 `2000-5000` step 更稳。

## 3. 怎么让输出更有意义

本项目新增了 `data/teaching_corpus.txt`，它比原来的 `tiny_corpus.txt` 更结构化，重复了 MiniLLM/Transformer 问答格式，便于小字符模型快速学会模式。

CPU 推荐先跑：

```bash
cd /home/undefined/Desktop/ai/projects/minillm
/home/undefined/Desktop/ai/.venv/bin/python train.py \
  --data data/teaching_corpus.txt \
  --device cpu \
  --max-steps 1500 \
  --eval-interval 300 \
  --eval-iters 5 \
  --batch-size 32 \
  --block-size 128 \
  --n-layer 2 \
  --n-head 4 \
  --n-embd 128 \
  --out-dir checkpoints
```

GPU 可以跑：

```bash
cd /home/undefined/Desktop/ai/projects/minillm
/home/undefined/Desktop/ai/.venv-sglang/bin/python train.py \
  --data data/teaching_corpus.txt \
  --device cuda \
  --max-steps 1500 \
  --eval-interval 300 \
  --eval-iters 5 \
  --batch-size 64 \
  --block-size 128 \
  --n-layer 2 \
  --n-head 4 \
  --n-embd 128 \
  --out-dir checkpoints
```

生成时先用 greedy 看模型是否学会稳定格式：

```bash
/home/undefined/Desktop/ai/.venv/bin/python generate.py \
  --checkpoint checkpoints/minillm.pt \
  --device cpu \
  --prompt "用户: 什么是 attention？\n助手:" \
  --max-new-tokens 120 \
  --greedy
```

再用采样看多样性：

```bash
/home/undefined/Desktop/ai/.venv/bin/python generate.py \
  --checkpoint checkpoints/minillm.pt \
  --device cpu \
  --prompt "用户: 什么是 decoder-only Transformer？\n助手:" \
  --max-new-tokens 120 \
  --temperature 0.7 \
  --top-k 20 \
  --seed 1
```

## 4. 一定需要 GPU 吗

不一定。MiniLLM 这种几十万到百万级参数的小模型可以用 CPU 训练和推理，只是慢一些。GPU 的价值在于大规模矩阵乘法并行和高显存带宽。真正的大模型训练通常需要 GPU；但小模型教学、短语料过拟合实验，CPU 完全可以完成。

如果 GPU 上训练好了一个 checkpoint，后续 CPU 可以直接加载这个 checkpoint 推理，不需要重新训练。训练阶段更新参数；推理阶段只读取参数并做前向计算。

## 5. 训练哪些参数能让模型变强

从零训练时，所有参数都会更新：

- token embedding：学习 token 的向量表示。
- position embedding：学习位置表示。
- attention 的 Q/K/V/O 线性层：学习如何从上下文取信息。
- MLP 的线性层：学习非线性特征变换。
- LayerNorm 的 scale/bias：稳定激活分布。
- lm_head：把隐藏状态映射到词表 logits；本项目和 token embedding 权重共享。

让模型更强不只是训练更多 step，还包括更大更干净的数据、更合理 tokenizer、更大模型容量、更好的优化器设置、更长上下文、更多训练 token 和更好的解码策略。

## 6. Qwen、Llama、Mistral 是什么架构

这些都是现代 decoder-only causal language model 家族，基本主线类似 GPT：token embedding、RoPE、多个 decoder block、causal self-attention、MLP、RMSNorm/LayerNorm、lm_head。

它们的差异主要在工程细节：

- Llama：Meta 的开源 decoder-only LLM 家族，常见结构是 RoPE + RMSNorm + SwiGLU + GQA/MHA。
- Qwen：阿里通义千问系列，也是 decoder-only；不同版本有 dense/MoE、GQA、长上下文、推理模型等变化。
- Mistral：Mistral AI 的 decoder-only 家族，早期 7B 强调 sliding-window attention，Mixtral 是 MoE 版本。

## 7. PreTrainedModel 风格是什么

Hugging Face `PreTrainedModel` 风格指模型类继承 Transformers 的基类，并实现标准接口：

- `config_class` 指向对应 `PretrainedConfig`。
- `forward()` 返回 Transformers 约定的输出结构，如 logits/loss/past_key_values。
- 支持 `save_pretrained()` 和 `from_pretrained()`。
- 权重命名、tie weights、generation 相关方法符合 Transformers 生态。

MiniLLM 当前只是普通 `torch.nn.Module`，所以能 `torch.save/torch.load`，但还不是完整 HF Transformers 模型。

## 8. config.json 保存什么

`config.json` 保存模型结构元数据，不保存权重。典型字段包括：

- `model_type`：模型类型，比如 `llama`、`qwen3`、`minigpt`。
- `architectures`：模型类名，比如 `Qwen3ForCausalLM`。
- `vocab_size`：词表大小。
- `hidden_size` / `n_embd`：隐藏维度。
- `num_hidden_layers` / `n_layer`：层数。
- `num_attention_heads` / `n_head`：attention 头数。
- `max_position_embeddings` / `block_size`：最大上下文长度。
- RoPE、RMSNorm、MoE、dtype、tie_word_embeddings 等结构参数。

vLLM 本地文档 `projects/vllm/docs/design/huggingface_integration.md` 也说明：vLLM 会先找 `config.json`，再根据 `model_type` 和 `architectures` 决定用哪个模型类加载。

## 9. 为什么保存为 model.safetensors

`model.safetensors` 保存权重张量。相比 PyTorch pickle 风格的 `.bin/.pt`，它有几个特点：

- 不执行任意 Python pickle 代码，安全性更好。
- 文件有明确张量元数据，便于快速读取。
- 更适合分片权重和分布式推理加载。
- vLLM/SGLang/HF Transformers 都优先支持它。

MiniLLM 新增 `export_hf_like.py`，可以导出 `config.json`、tokenizer 元数据、`generation_config.json` 和权重文件。

## 10. HF tokenizer 有什么特点

Hugging Face tokenizer 不只是一个 `encode/decode` 函数，通常还包括：

- 标准文件结构：`tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json` 等。
- fast tokenizer：Rust 实现，速度快，支持 offset mapping。
- special tokens：bos/eos/pad/unk/chat template。
- 和 `AutoTokenizer.from_pretrained()` 兼容。

MiniLLM 的 `CharTokenizer` 是教学用字符 tokenizer。它很清楚，但不是标准 HF fast tokenizer。后续若要进入 vLLM/SGLang，建议换成 HF tokenizers 的 WordLevel/BPE，或者实现完整自定义 tokenizer 文件与加载代码。

## 11. 引擎支持哪些模型类型

推理引擎支持的是“架构实现”，不是任意 checkpoint。SGLang 本地支持列表在 `projects/sglang/docs/supported_models/text_generation/generative_models.md`，包括 Qwen、Llama、Mistral、DeepSeek、Gemma、Phi、MiniCPM、ChatGLM、Baichuan、Mixtral/MoE 等大量模型家族。

vLLM 的加载链路通常是：读取 HF `config.json` -> 看 `model_type`/`architectures` -> 在 vLLM model registry 中找到对应实现 -> 加载 tokenizer 和 safetensors/bin 权重。若 registry 没有这个 architecture，就需要新增模型实现或走 Transformers backend/remote code 路径。

## 12. 注册架构有什么要求

对 vLLM/nano-vLLM/SGLang 来说，注册一个新模型通常要满足：

- 有明确 config class 和 architecture 名称。
- 模型 forward 签名符合引擎要求。
- 能处理 input_ids、positions、KV cache、attention metadata。
- lm_head/logits 输出符合 sampler 预期。
- 权重命名能从 HF/safetensors 加载到模型参数。
- tokenizer 能把文本转成 token id，并能 decode 输出。
- 若要高性能，还要适配 paged KV cache、CUDA graph、FlashAttention/FlashInfer/Triton attention。

## 13. MiniLLM 放到 nano-vLLM 的计划

当前 `projects/nano-vllm` 是 Qwen3 专用路径：`model_runner.py` 直接构造 `Qwen3ForCausalLM`，权重 loader 读 HF `*.safetensors`。要加 MiniLLM 后端，建议按下面做：

1. 在 `nanovllm/models/minigpt.py` 新增 `MiniGPTForCausalLM`，先复用 MiniLLM 的 embedding、LayerNorm、attention、MLP、lm_head。
2. 在 `model_runner.py` 中根据 `hf_config.model_type == "minigpt"` 选择 `MiniGPTForCausalLM`，否则走 Qwen3。
3. 适配权重 loader，让 `model.safetensors` 的参数名能对应 MiniGPT 参数。
4. 第一阶段先用 eager full forward，不做 paged KV cache，只验证输出一致。
5. 第二阶段再给 MiniGPT attention 加 KV cache 接口。
6. 第三阶段接入 nano-vLLM scheduler/block_manager，让它真正成为 serving engine 后端。

这是一个很好的练习，但不要一开始就追求完整高性能。先做到“nano-vLLM 能根据 config 识别 minigpt，并跑通离线 generate”。

## 14. MiniLLM 放到 SGLang 的计划

SGLang 本地文档 `projects/sglang/docs/supported_models/extending/support_new_models.md` 说明：新语言模型通常要在 `python/sglang/srt/models` 下新增一个模型文件，或用 `ModelRegistry` 注册外部实现。

Mini-SGLang 路线可以这样做：

1. 先写一个外部 `MiniGPTForCausalLM` wrapper。
2. 实现 SGLang 要求的 forward/forward_batch 入口。
3. 用 SGLang 的 logits processor 输出 logits。
4. 第一阶段不做多模态、不做量化、不做分布式。
5. 对比 Hugging Face/PyTorch MiniLLM 输出，确保 prefill logits 对齐。
6. 再考虑 RadixAttention/KV cache。

相比 nano-vLLM，SGLang 接入面更大；建议先做 nano-vLLM 后端练习。
