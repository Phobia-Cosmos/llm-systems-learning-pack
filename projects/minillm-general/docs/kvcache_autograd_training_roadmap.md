# KV Cache、Autograd、训练到推理的完整路线

这份文档回答两个核心问题：

1. MiniLLM 里 KV cache、autograd、训练、推理到底是什么关系。
2. 从 MiniLLM 走到“训练、微调、推理引擎、量化压缩”的完整学习路线应该怎么补齐。

## 1. 自回归生成阶段是什么

decoder-only language model 的目标是按顺序生成 token。数学上把一句话的概率拆成：

```text
P(x_1, x_2, ..., x_T) = Π_t P(x_t | x_<t)
```

生成时已经有 prompt：

```text
用户: 什么是 attention？
助手:
```

模型先计算“下一个 token 是谁”的分布：

```text
logits_t = model(x_1, ..., x_t)[最后一个位置]
p(next_token) = softmax(logits_t / temperature)
```

采样或 argmax 得到新 token 后，把它接回输入：

```text
x_{t+1} ~ p(next_token)
输入变成 x_1, ..., x_t, x_{t+1}
```

然后继续预测下一个 token。因此生成 token 的外层循环是串行的：第 10 个新 token 不出来，第 11 个新 token 的输入就还不存在。

但是这不代表 GPU 不能并行。每一步内部仍然有大量并行：

- batch 内多个请求并行。
- 多个 attention head 并行。
- 矩阵乘法内部并行。
- 一次 prefill 可以并行计算 prompt 中所有位置的 logits。

## 2. MiniLLM 训练时也能并行计算 logits 吗

可以。训练阶段使用 teacher forcing：正确答案序列已经在数据里，所以可以一次性把整段输入给模型。

训练 batch 里通常有：

```text
x = token[0:T]
y = token[1:T+1]
```

模型一次 forward 输出：

```text
logits = model(x)        # shape: [B, T, vocab_size]
loss = cross_entropy(logits, y)
```

虽然每个位置只能看左边上下文，但 causal mask 已经保证第 `t` 个位置不能偷看未来 token。因此训练时不需要一个 token 一个 token 地跑，可以一次性并行计算所有位置的 logits。

## 3. 为什么无 KV cache 的 MiniLLM 每一步都重算完整 forward

当前普通 `generate()` 的核心逻辑是：

```python
idx_cond = idx[:, -self.config.block_size:]
logits, _ = self(idx_cond)
logits = logits[:, -1, :]
```

也就是说，每生成一个 token，都把最近 `block_size` 个 token 重新送入模型。

如果当前上下文长度是 `T`，一次完整 forward 会重新做：

- token embedding。
- position embedding。
- 每一层的 Q/K/V 投影。
- 每一层的 attention 分数 `QK^T`。
- 每一层的 MLP。
- 最后的 lm_head。

但生成阶段真正需要的只是“最后一个位置”的 logits。前面历史 token 的 K/V、隐藏状态相关计算已经算过很多次，所以重复计算明显浪费。

## 4. KV cache 存储的是什么

在 self-attention 中，每层都会算：

```text
Q = XW_Q
K = XW_K
V = XW_V
Attention(Q,K,V) = softmax(QK^T / sqrt(d_head) + mask) V
```

生成阶段，新 token 的 query `Q_new` 每次都不同，必须现算。但历史 token 的 `K_old` 和 `V_old` 在推理时不会变，因为：

- 模型参数固定。
- 历史 token 固定。
- 推理阶段没有 optimizer 更新参数。

所以 KV cache 保存的是每一层历史 token 的 K 和 V：

```text
past_key_values[layer] = (K_cache, V_cache)
K_cache shape: [batch, num_key_value_heads, past_len, head_dim]
V_cache shape: [batch, num_key_value_heads, past_len, head_dim]
```

MHA 中 `num_key_value_heads=n_head`；GQA/MQA 只保存紧凑 KV heads，做 attention matmul 时才映射到 query heads，因此 cache 元素数按 `num_key_value_heads/n_head` 缩小。

下一个 token 到来时，只算新 token 的：

```text
q_new, k_new, v_new
```

然后拼到缓存：

```text
K_all = concat(K_cache, k_new)
V_all = concat(V_cache, v_new)
output_new = softmax(q_new K_all^T / sqrt(d_head)) V_all
```

KV cache 不保存 Q，一般也不保存 logits。Q 是当前 query，用完即可；K/V 是未来每一步都会被新 query 查询的历史信息。

## 5. KV cache 会保存历史所有 token 吗

概念上会保存当前上下文窗口内所有历史 token 的 K/V。真实推理引擎不会简单放一个无限增长的大 tensor，而是会受 `max_model_len` 限制，并用更复杂的管理方式：

- vLLM/nano-vLLM：paged KV cache，把 KV 按 block/page 管理。
- SGLang：RadixAttention/prefix cache 复用公共前缀。
- 长上下文模型：可能配合 sliding window、chunked attention、prefix cache、offload。

MiniLLM 现在支持 learned absolute position 和 RoPE。当前教学 KV cache 路径对两种模式都要求：

```text
prompt_len + max_new_tokens <= block_size
```

learned 模式超过 `block_size` 时不能简单左移 cache，因为旧 K/V 已混入绝对位置向量。RoPE 模式已经消除了这个结构障碍，但可靠长上下文仍需扩展训练长度、cache eviction/sliding-window 策略和专门评测；仅加入 RoPE 不等于自动获得长上下文能力。

## 6. 当前 MiniLLM 已经加入的 KV cache 路径

当前代码已经新增：

- `CausalSelfAttention.forward_with_cache()`：计算当前 token 的 Q/K/V，并拼接历史 K/V。
- `TransformerBlock.forward_with_cache()`：每个 block 返回自己的 present KV。
- `MiniGPT.forward_with_cache()`：整模型返回 logits 和每层 KV cache。
- `MiniGPT.generate_with_kv_cache()`：生成时复用历史 K/V。
- `generate.py --kv-cache`：命令行入口。

运行：

```bash
cd /home/undefined/Desktop/ai/projects/minillm
/home/undefined/Desktop/ai/.venv-sglang/bin/python generate.py \
  --device cuda \
  --checkpoint artifacts/checkpoints/minillm.pt \
  --prompt "用户: 什么是 decoder-only Transformer？\n助手:" \
  --max-new-tokens 30 \
  --greedy \
  --kv-cache
```

我已经做过一致性验证：在 greedy 解码、总长度不超过 `block_size` 时，无 cache 和 KV cache 输出一致。

## 7. 为什么 CPU 训练时不使用 KV cache

训练和推理要优化的问题不同。

训练阶段：

- 所有 token 的目标答案都已知。
- 一次 forward 可以并行计算所有位置的 logits。
- 之后要 backward，计算参数梯度。
- 每次 `optimizer.step()` 后参数会变。

KV cache 的价值是“复用历史 token 在固定参数下算出的 K/V”。训练时参数会更新，上一轮 batch 或上一轮 step 的 K/V 立刻过期。即使在同一个 forward 内，训练也不需要逐 token 生成，因为 teacher forcing 已经让整段序列并行计算更快。

所以训练不靠 KV cache 加速，而是靠：

- 大 batch。
- 矩阵乘法并行。
- mixed precision。
- FlashAttention。
- gradient checkpointing。
- 数据并行、张量并行、流水并行、FSDP/ZeRO。

## 8. Autograd 保存的中间激活是什么，存在何地

PyTorch autograd 是为了反向传播服务的。只要 tensor 参与计算并且需要梯度，PyTorch 会动态记录计算图：

```text
loss
  -> cross_entropy
  -> lm_head
  -> ln_f
  -> TransformerBlock N
  -> ...
  -> embedding
```

为了计算梯度，很多 backward 公式需要 forward 时的中间值。例如：

- Linear backward 需要输入 `x` 和权重 `W`。
- LayerNorm backward 需要均值、方差或归一化后的中间结果。
- GELU backward 需要激活输入。
- Attention backward 需要 Q/K/V、softmax 权重或足够重建它们的信息。

这些中间激活保存在当前设备的内存里：

- CPU 训练：主要在 CPU RAM。
- CUDA 训练：主要在 GPU VRAM。

它们通常只活到一次 `loss.backward()` 结束。除非设置 `retain_graph=True`，否则计算图和大部分 saved tensors 会被释放。参数梯度保存在每个参数的 `.grad` 上；AdamW 的动量状态保存在 optimizer 对象里。

因此：

```text
autograd 中间激活 = 为了求梯度临时保存的 forward 计算图数据
KV cache = 为了推理复用历史 token 的 K/V
```

两者都“存中间结果”，但目的完全不同。

## 9. Autograd 和 KV cache 的产生原理有什么区别

Autograd 是 PyTorch 自动产生的：

```python
logits, loss = model(x, y)
loss.backward()
```

只要没有 `torch.no_grad()`，PyTorch 会记录张量操作，自动按链式法则求导。

KV cache 是模型代码显式产生的：

```python
logits, past_key_values = model.forward_with_cache(input_ids)
logits, past_key_values = model.forward_with_cache(next_token, past_key_values)
```

推理时通常会包在：

```python
@torch.no_grad()
```

这样不会构建 autograd graph，只保存我们主动返回的 K/V cache。

## 10. 训练时各轮次之间 autograd 会变化吗

会。每个 batch、每次 forward 都会创建新的动态图。因为：

- 输入 batch 不同。
- dropout 等训练行为可能不同。
- 参数经过 optimizer 更新后不同。
- loss 不同，梯度也不同。

训练不会把上一轮 autograd graph 存下来复用。checkpoint 保存的是模型参数、config、tokenizer、训练参数，不保存 autograd 中间激活。

## 11. 为什么存 KV cache 可以加速推理

假设 prompt 长度为 `P`，要生成 `N` 个 token。

无 cache 时，第 `i` 步通常重新跑长度约 `P+i` 的完整 forward。attention 里会重复计算历史 token 的 Q/K/V 和 `QK^T`。

有 cache 时：

1. prefill：对 prompt 做一次完整 forward，得到每层 K/V cache。
2. decode：每一步只输入 1 个新 token，算它的 q/k/v，然后让 `q_new` 查历史 `K_all/V_all`。

直观差别：

```text
无 cache：每一步重算整段历史。
有 cache：历史 K/V 只算一次，之后不断追加新 token 的 K/V。
```

对于小模型、短序列，Python 调度和 kernel launch 开销可能掩盖收益；对于大模型、长上下文、多并发请求，KV cache 是推理引擎必须具备的能力。

## 12. 推理时的采样状态是什么

一次请求在推理引擎里不只是一个字符串，还包含状态：

- 已生成的 token ids。
- 当前序列长度。
- KV cache 或 block table。
- temperature、top-p、top-k、repetition penalty 等采样参数。
- RNG 状态。
- eos 是否出现。
- 请求状态：waiting/running/finished。
- 对 serving engine 来说，还包括 batch 调度、prefix cache 命中、显存 block 分配。

采样本身通常是：

```text
logits -> logits processor -> softmax -> sample/argmax -> next token
```

MiniLLM 当前支持 temperature、top-k、greedy。vLLM/SGLang 会有更完整的 logits processor 和并发调度。

## 13. 训练时梯度如何计算

MiniLLM 的训练目标是 next-token prediction。

输入：

```text
x_{b,t} = 第 b 条样本第 t 个 token
y_{b,t} = x_{b,t+1}
```

模型：

```text
h = Transformer(x; θ)
z = h W_lm^T
p = softmax(z)
```

交叉熵损失：

```text
L = - 1/(B*T) Σ_b Σ_t log p_{b,t,y_{b,t}}
```

对 logits 的梯度核心形式是：

```text
∂L/∂z = p - one_hot(y)
```

然后 autograd 按链式法则继续往前传：

```text
lm_head -> ln_f -> MLP -> attention -> embedding
```

每个参数都会得到梯度：

```text
param.grad = ∂L/∂param
```

AdamW 更新大致是：

```text
m_t = β1 m_{t-1} + (1-β1) g_t
v_t = β2 v_{t-1} + (1-β2) g_t^2
θ_t = θ_{t-1} - lr * m_hat / (sqrt(v_hat) + eps) - lr * weight_decay * θ_{t-1}
```

对应到 `train.py`：

```python
_, loss = model(x, y)
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

## 14. MiniLLM 和真实 GPT 类大模型的差距

除了参数量，差距主要在这些方面：

- tokenizer：MiniLLM 已支持 Char、Byte-BPE、SentencePiece 和 HF adapter；真实 LLM 还会使用大词表、成熟 normalizer/chat template 与大量语料训练 tokenizer。
- 数据：MiniLLM 使用小教学语料；真实 LLM 使用大规模、多源、清洗、去重、配比的数据。
- 位置编码：MiniLLM 已支持 learned absolute position 与 RoPE；现代 LLM 还常配合 RoPE scaling 和长上下文训练。
- Norm：MiniLLM 用 LayerNorm；Llama/Qwen/Mistral 常用 RMSNorm。
- MLP：MiniLLM 用 GELU MLP；现代 LLM 常用 SwiGLU/GEGLU。
- Attention：MiniLLM 已可选 MHA/GQA/MQA；现代 LLM 还常用 FlashAttention、滑动窗口和长上下文 attention。
- 训练系统：MiniLLM 原生单进程训练；大模型需要分布式、混合精度、checkpoint resume、数据流式加载。
- 对齐阶段：MiniLLM 没有 SFT、DPO/RLHF、安全对齐、偏好数据。
- 格式生态：MiniLLM 只是 PyTorch Module；生产模型通常是完整 Hugging Face `PreTrainedModel` + tokenizer + safetensors。
- 推理系统：MiniLLM 原生 generate；生产系统需要 paged KV cache、continuous batching、CUDA graph、FlashInfer/FlashAttention、量化。
- 评估：MiniLLM 没有系统 benchmark、perplexity 报告、指令跟随评估、回归测试。

但核心主线是一样的：

```text
token ids -> embedding -> decoder blocks -> logits -> next token loss/generation
```

## 15. 是否先原生训练，再用训练框架

建议顺序是：

1. 先用原生 PyTorch 把训练完整跑通。
2. 再引入训练框架。

原因是：原生训练能让你看清楚 batch、forward、loss、backward、optimizer、checkpoint 的完整闭环。训练框架解决的是规模化问题，不应该在你还没理解基本训练循环时先引入。

MiniLLM 当前阶段先补齐原生训练功能：

- Dataset/DataLoader。
- checkpoint resume。
- eval/perplexity。
- learning rate schedule。
- gradient accumulation。
- mixed precision。
- logging。

之后再接：

- Hugging Face Trainer/Accelerate：适合学习标准训练流程。
- DeepSpeed ZeRO/FSDP：适合学习显存切分和分布式训练。
- Megatron-LM 风格：适合理解张量并行/流水并行。

如果只有一张 GPU，真正的“分布式加速”不会凭空变快。可以学习分布式 API 和模拟多进程，但实际收益主要来自 mixed precision、gradient accumulation、torch.compile、FlashAttention 和更好的数据 pipeline。

## 16. 完整 LLM 工程流程

建议把整个 AI infra 路线整理成下面这条链：

```text
数据 -> tokenizer -> 预训练 -> 评估 -> checkpoint/HF 格式
    -> 指令微调/SFT -> 偏好优化/DPO/RLHF -> 再评估
    -> 推理基线 -> 推理引擎 -> 量化/压缩/算子优化
    -> serving/API -> 监控与回归测试
```

对应到当前项目：

1. 数据：`data/tiny_corpus.txt`、`data/teaching_corpus.txt`。
2. tokenizer：Char、Byte-BPE、SentencePiece 或 HF adapter。
3. 模型：`MiniGPT`。
4. 预训练：`train.py`。
5. 推理基线：`generate.py`。
6. HF-like 导出：`export_hf_like.py`。
7. nano-vLLM 后端：`projects/nano-vllm/nanovllm/models/minigpt.py`。
8. KV cache：`generate.py --kv-cache`。
9. 下一步：训练 resume/LR schedule/AMP/gradient accumulation，然后是 LoRA/SFT、真实 HF `PreTrainedModel`、SGLang backend、量化。

## 17. 初学者视角的后续任务规划

### 阶段 A：把 MiniLLM 训练闭环补完整

- 给 `generate_with_kv_cache()` 加测试：已完成，greedy 模式下和普通 `generate()` 输出一致，并覆盖 MHA/GQA/MQA。
- 加 `--resume`，能从 checkpoint 继续训练。
- 加 `--save-interval`，训练中定期保存。
- 加 `perplexity = exp(loss)` 打印。
- 加学习率 warmup/cosine decay。
- 加 gradient accumulation。

### 阶段 B：把 MiniLLM 改得更像现代 GPT

- 字符 tokenizer 换成 BPE 或 SentencePiece。
- learned position embedding 与 RoPE 对照：已完成。
- LayerNorm 换成 RMSNorm：已完成可选实现。
- GELU MLP 换成 SwiGLU：已完成可选实现。
- attention 加 GQA/MQA 选项：已完成，并保留默认 MHA checkpoint 兼容。
- KV cache 支持超过 `block_size` 的长上下文策略。

### 阶段 C：训练框架与微调

- 用原生 PyTorch 实现 SFT 数据格式：`prompt -> response`。
- 加 LoRA，只训练 adapter 参数。
- 对比 full finetune 和 LoRA finetune。
- 再用 Hugging Face/Accelerate 复现同样流程。
- 学习 DeepSpeed/FSDP 主要解决什么显存问题。

### 阶段 D：格式与生态

- 把 MiniLLM 包成真正的 Transformers `PreTrainedModel`。
- 支持 `save_pretrained()` 和 `from_pretrained()`。
- tokenizer 换成 HF tokenizer 文件结构。
- 导出标准 `config.json`、`model.safetensors`、`generation_config.json`。

### 阶段 E：推理引擎

- 当前已经完成：nano-vLLM 和 vLLM 能加载 MiniLLM learned/RoPE HF-like export，并走各自 KV-cache attention 路径生成。
- 已对齐原生 PyTorch、MiniLLM KV cache、nano-vLLM 和 vLLM；SGLang native backend 仍待实现。
- 学 scheduler、continuous batching、prefix cache、CUDA graph、FlashAttention/FlashInfer sampler。

### 阶段 F：量化和算子优化

- 先做权重量化：FP32 -> FP16/BF16 -> INT8 -> INT4。
- 对比精度、速度、显存。
- 写 RMSNorm、RoPE、softmax、attention 的小 CUDA/Triton 算子。
- 把自定义算子接回 MiniLLM 或 nano-vLLM。
- 观察瓶颈从计算受限变成内存带宽受限的过程。

## 18. 现在最应该做什么

当前最合适的下一批任务是：

1. 固定当前 RoPE+BPE checkpoint 的 val loss、perplexity、生成和性能基线。
2. 可选 RMSNorm、SwiGLU、GQA/MQA 已实现；下一步用正式 MHA/GQA checkpoint 补齐 GPU 引擎回归数据。
3. 下一代码迭代给训练脚本加 checkpoint resume、warmup/cosine learning rate、AMP 和 gradient accumulation。
4. 完成 SFT response loss mask 和 LoRA，再包成真正 HF `PreTrainedModel`。
5. 实现 SGLang native MiniGPT backend，然后进入量化和 Triton/CUDA 算子实验。

这条路线比较稳：先把小模型的每个环节讲清楚、跑通，再把同样概念映射到 vLLM/SGLang/真实大模型。
