# 从 MiniLLM 代码理解 Transformer

这份笔记按 `projects/minillm` 的代码路径，把一个 decoder-only Transformer 从数据到生成串起来。先抓住主线：LLM 不是直接理解文字，而是反复做“给定前面的 token，预测下一个 token”。

## 1. 文本先变成 token id

入口是 `CharTokenizer`。它做两件事：

1. `encode(text)`：把字符串变成整数列表，例如每个字符对应一个 id。
2. `decode(ids)`：把模型生成的整数 id 还原成字符串。

模型不能直接处理字符，因为神经网络的输入是张量。token id 是离散编号，后面会通过 `nn.Embedding` 查表变成连续向量。

## 2. 训练样本是 x 预测 y

`get_batch()` 从长文本 token 序列里随机切片：

```text
x = [t0, t1, t2, ..., t63]
y = [t1, t2, t3, ..., t64]
```

也就是说，模型在每个位置都学习“看到当前位置及其以前的 token，预测下一个 token”。这叫 next-token prediction，是 GPT 类模型的基本训练目标。

## 3. token embedding 加 position embedding

`MiniGPT.forward()` 里先做：

```python
x = token_embedding(idx) + position_embedding(positions)
```

`token_embedding` 回答“这个 token 是什么”，`position_embedding` 回答“它在序列第几个位置”。self-attention 本身对顺序不敏感，所以必须加入位置信息。

shape 主线是：

```text
idx:              [batch, seq_len]
token embedding:  [batch, seq_len, n_embd]
position emb:     [seq_len, n_embd] -> broadcast 到 [batch, seq_len, n_embd]
```

## 4. TransformerBlock 的核心结构

一个 block 是：

```text
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

这是 pre-norm decoder block。残差连接 `x + ...` 的作用是让信息和梯度更容易穿过很多层；LayerNorm 稳定每层输入分布；Attention 负责从上下文取信息；MLP 负责对每个位置做非线性变换。

## 5. Attention 到底在算什么

代码里的公式是：

```text
Q, K, V = Linear(x).split(3)
score = QK^T / sqrt(head_dim)
score = causal_mask(score)
weight = softmax(score)
y = weight V
```

直觉解释：

- Q 是 query，表示“当前位置想找什么信息”。
- K 是 key，表示“每个历史位置能提供什么索引”。
- V 是 value，表示“真正被取走的信息内容”。
- `QK^T` 得到每个 token 对其他 token 的匹配分数。
- causal mask 禁止看未来 token，因为训练目标是预测未来，不能泄题。
- softmax 把分数变成权重。
- 权重乘 V 就是从上下文中加权汇总信息。

多头 attention 是把 `n_embd` 切成多个 head。不同 head 可以学习不同关系，例如局部相邻、长距离依赖、标点结构、问答边界等。head 不是越多越好，因为总维度固定时 head 越多，每个 head 的 `head_dim` 越小。

## 6. MLP 为什么要扩到 4 倍

Attention 混合不同 token 的信息，MLP 则在每个 token 位置上独立处理特征。先从 `n_embd` 扩到 `4*n_embd`，再压回 `n_embd`，是 Transformer 里的经典设计：中间维度更大，表达能力更强。GELU 提供非线性，否则多层 Linear 合在一起仍然只是线性变换。

## 7. logits 和 loss

最后：

```python
logits = lm_head(x)
```

`logits` 的 shape 是 `[batch, seq_len, vocab_size]`。它不是 token，也不是概率，而是每个位置对词表中每个 token 的原始分数。

训练时有 targets，所以计算：

```python
loss = cross_entropy(logits, targets)
```

cross entropy 会让正确下一个 token 的概率变高，让错误 token 的概率变低。`loss.backward()` 根据 loss 自动计算梯度，`optimizer.step()` 根据梯度更新参数。

## 8. 训练循环在做什么

每一步训练大致是：

```text
抽一批 x,y
前向：model(x,y) -> logits, loss
清空旧梯度
反向：loss.backward()
裁剪梯度，避免梯度爆炸
优化器更新参数
每隔 eval_interval 评估 train/val loss
```

MiniGPT 本身只定义模型计算，优化器才负责“根据错误改参数”。这就是为什么只有模型不够，还需要 AdamW。

## 9. 生成时为什么一个 token 一个 token 地来

生成没有 targets。流程是：

```text
prompt -> encode -> idx
重复 max_new_tokens 次：
  截取最后 block_size 个 token
  前向得到 logits
  只取最后一个位置 logits[:, -1, :]
  temperature/top-k 调整分布
  softmax 得到概率
  multinomial 采样下一个 token
  拼回 idx
idx -> decode -> 文本
```

GPT 是自回归模型：下一个 token 生成后会成为后续上下文的一部分，所以必须逐步生成。

## 10. 你应该先记住的一张图

```text
文本
  -> tokenizer.encode
  -> token ids [B,T]
  -> token embedding + position embedding [B,T,C]
  -> N 个 TransformerBlock
       -> LayerNorm
       -> causal multi-head self-attention
       -> residual add
       -> LayerNorm
       -> MLP
       -> residual add
  -> final LayerNorm
  -> lm_head
  -> logits [B,T,V]
  -> cross entropy 训练 / sampling 生成
  -> tokenizer.decode
  -> 文本
```

理解 Transformer 时不要先陷进所有工程细节。先把三件事抓稳：

1. 训练目标：根据前文预测下一个 token。
2. Attention：每个 token 用 Q/K/V 从历史 token 里取信息。
3. 输出：hidden state 经过 lm_head 变成 logits，再训练用 loss，生成用采样。

## 11. 第 8 组问题：shape、embedding、logits、tokenizer 与后续迭代路线

### vocab_size 是什么

`vocab_size` 是整个词表里的 token 数量，也就是模型能识别和预测的 token id 总数。

在当前 MiniLLM 里，`CharTokenizer` 会从训练文本里收集出现过的字符，再额外加一个 `<unk>`。所以当前的 `vocab_size` 不是“中文词语数量”或“英文单词数量”，而是“字符 token 数量 + `<unk>`”。

如果 tokenizer 换成 BPE/SentencePiece，`vocab_size` 就会变成子词词表大小，例如 32K、50K、100K 等。`vocab_size` 一变，`token_embedding` 和 `lm_head` 的参数形状都会变，旧 checkpoint 通常不能直接复用。

### input_ids 的 B、T 分别是什么

`input_ids` 是 token id 张量，shape 是：

```text
input_ids: [B, T]
```

含义是：

| 符号 | 含义 | 在 MiniLLM 里的对应 |
| --- | --- | --- |
| B | batch size，一次并行处理多少条样本 | `batch` |
| T | sequence length，每条样本有多少个 token | `seq_len`，不能超过 `block_size` |
| V | vocab size，词表大小 | `config.vocab_size` |
| C | channel / hidden size / embedding dim | `config.n_embd` |

例如 `B=4, T=64` 表示一次喂给模型 4 条训练片段，每条片段 64 个 token。

### 为什么 token embedding 是 vocab 相关的

token id 本身只是整数，例如：

```text
[12, 45, 7, 99]
```

神经网络不能直接从这些编号里理解语义，所以要先查 embedding 表：

```text
token_embedding.weight: [V, C]
input_ids:              [B, T]
token_embedding(ids):   [B, T, C]
```

`V` 是词表大小，因为每一个 token id 都需要一行向量。`C` 是每个 token 被表示成多少维向量。

当前 `embedding=128` 的意思就是 `n_embd=128`：每个 token 会被映射成一个 128 维连续向量。这个 128 维不是 128 个 token，而是每个 token 的隐藏表示宽度。后续 attention、MLP、LayerNorm 都主要在这个 128 维空间里计算。

### position embedding 是什么

你写的 `positic` 可以理解为 `position embedding`，也就是位置编码。

如果只看 token embedding，模型知道“这个 token 是什么”，但不知道“它在第几个位置”。self-attention 本身对顺序不敏感，所以需要额外给每个位置一个向量：

```text
position ids:              [0, 1, 2, ..., T-1]
position_embedding.weight: [block_size, C]
position_embedding(pos):   [T, C]
```

然后：

```text
x = token_embedding(input_ids) + position_embedding(positions)
```

相加后，`x` 仍然是 `[B, T, C]`，但它同时包含 token 信息和位置信息。

当前 MiniLLM 用的是 learned absolute position embedding，也就是每个位置学一个向量。现代 GPT 类模型更常用 RoPE，因为它更适合长上下文扩展，也更接近 Llama/Qwen/Mistral 的路线。

### ln_f 是什么

`ln_f` 是 final LayerNorm，放在所有 Transformer block 之后：

```text
x -> blocks -> ln_f -> lm_head
```

它的作用是把最终 hidden state 的尺度稳定下来，再交给输出层。没有它模型也能跑，但训练稳定性和输出质量通常会变差。GPT-2 这类 pre-norm decoder-only 结构也有最后的 `ln_f`。

### lm_head 是什么

`lm_head` 是 language modeling head，负责把每个位置的 hidden state 映射成词表分数：

```text
hidden state: [B, T, C]
lm_head:      C -> V
logits:       [B, T, V]
```

也就是说，模型在每一个位置都输出一个长度为 `vocab_size` 的分数向量，用来预测“下一个 token 最可能是词表里的哪一个”。

当前 MiniLLM 里：

```python
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
self.lm_head.weight = self.token_embedding.weight
```

这叫 weight tying，也就是输入 embedding 和输出 head 共享同一张权重表。好处是减少参数，也让“读 token 的向量”和“预测 token 的向量”处在同一空间里。

### 为什么 logits 多了一维 vocab_size

`input_ids` 是 `[B, T]`，但 `logits` 是 `[B, T, V]`，多出来的 `V` 是因为每一个位置都要对整个词表打分。

例如某个位置的 hidden state 是 128 维：

```text
x[b, t]: [128]
```

经过 `lm_head` 后：

```text
logits[b, t]: [vocab_size]
```

这个向量里的第 `i` 个数表示“下一个 token 是 token i 的原始分数”。训练时用 `cross_entropy(logits, targets)` 让正确 token 的分数变高；生成时对最后一个位置的 logits 做 temperature、top-k、top-p 和采样，得到下一个 token id。

### MiniLLM 和真正 GPT 还差什么

当前 MiniLLM 已经有 GPT 主干骨架：

```text
tokenizer -> token embedding -> position embedding
-> causal self-attention + MLP blocks
-> ln_f -> lm_head -> logits
-> loss / generate
```

但和真实 GPT/Llama/Qwen/Mistral 还差很多工程和模型细节：

| 方向 | MiniLLM 当前状态 | 真实 GPT 类模型常见做法 |
| --- | --- | --- |
| tokenizer | 字符级 `CharTokenizer` | BPE、Byte-level BPE、SentencePiece、tiktoken/HF fast tokenizer |
| special tokens | 很少 | BOS、EOS、PAD、UNK、system/user/assistant、chat template |
| position | learned absolute position | RoPE、ALiBi 或长上下文位置扩展 |
| norm | LayerNorm | RMSNorm 或 LayerNorm 变体 |
| MLP | Linear + GELU + Linear | SwiGLU/GEGLU 等 gated MLP |
| attention | 教学版 causal attention | FlashAttention、PagedAttention、GQA/MQA、KV cache、prefix cache |
| 训练数据 | tiny corpus / teaching corpus | 大规模清洗语料、代码、数学、多语言、指令数据 |
| 训练系统 | 单机 PyTorch loop | mixed precision、distributed training、checkpoint sharding、监控评测 |
| 后训练 | 初步/待完善 | SFT、偏好优化、RLHF/DPO、安全对齐 |
| HF 生态 | HF-like export | 标准 `PreTrainedModel`、标准 tokenizer、标准 generation config |
| 推理服务 | 已接教学 engine，vLLM MiniGPT 骨架可跑 | OpenAI API server、continuous batching、paged KV cache、量化、并发调度 |

所以 MiniLLM 不是“错误版本”，而是“保留主干、去掉复杂工程”的学习版本。后续每补一个模块，都应该尽量保持主线接口不变：

```text
input_ids [B,T] -> logits [B,T,V]
```

### 为什么当前 tokenizer 不是通用的

当前 `CharTokenizer` 不通用，原因有四个：

1. 它是从当前训练文本临时构建的，换一份语料，token id 映射就会变。
2. 它是字符级，压缩率低；同样一段话会变成更多 token，训练和推理都更慢。
3. 它不是 Hugging Face fast tokenizer，没有标准的 normalizer、pre-tokenizer、decoder、special token 配置。
4. 它没有成熟的 chat template，不适合直接接入 vLLM/SGLang 的标准 OpenAI 服务流程。

真实模型必须固定 tokenizer。训练、SFT、推理必须使用同一个 tokenizer，否则 token id 的语义就变了，权重也对不上。

常用 tokenizer 类型：

| 类型 | 代表 | 特点 |
| --- | --- | --- |
| Char tokenizer | 教学项目常用 | 最容易理解，但序列长、效率低 |
| Word tokenizer | 早期 NLP | 词表容易爆炸，遇到新词麻烦 |
| BPE | GPT-2、很多 LLM | 通过合并高频片段得到子词 |
| Byte-level BPE | GPT-2/tiktoken 路线 | 覆盖能力强，几乎不怕未知字符 |
| WordPiece | BERT | 常见于 encoder-only 模型 |
| SentencePiece BPE/Unigram | T5/Llama/Qwen 等生态常见 | 不依赖空格分词，适合多语言 |
| HF fast tokenizer | Transformers 生态 | Rust 实现，速度快，格式标准 |

### MiniLLM 后续如何最小化迭代

后续开发不要一次性把 tokenizer、RoPE、RMSNorm、KV cache、vLLM、SGLang、量化全混在一起。更好的方式是每次只替换一个模块，并保持其他模块接口稳定。

建议路线：

| 迭代 | 目标 | 改动边界 | 验证方式 |
| --- | --- | --- | --- |
| 0 | 固定 baseline | 保持当前 CharTokenizer + MiniGPT | 能训练、生成、导出 HF-like |
| 1 | 加 special tokens 和 EOS stop | 只改 tokenizer/generate | 生成能在 EOS 停止 |
| 2 | 抽象 tokenizer 接口 | 增加 `BaseTokenizer` 或 adapter | CharTokenizer 行为不变 |
| 3 | 接入 HF/BPE tokenizer | 新增 tokenizer 实现，重训模型 | `input_ids [B,T]` 仍能进模型 |
| 4 | RoPE 替代 position embedding | 只改 position/attention 内部 | logits shape 不变 |
| 5 | RMSNorm 替代 LayerNorm | 只改 norm 模块 | loss 正常下降 |
| 6 | SwiGLU 替代 GELU MLP | 只改 MLP | 参数量和 loss 可对比 |
| 7 | 标准化 KV cache API | 只改 generate/attention cache | 生成结果一致，速度更快 |
| 8 | SFT 数据格式与 `train_sft.py` | 只改数据和 loss mask | 能学习问答格式 |
| 9 | 标准 HF `PreTrainedModel` | 新增 HF wrapper，不破坏原模型 | `from_pretrained` 能加载 |
| 10 | nano-vLLM/vLLM/SGLang 后端 | 只做 engine adapter | engine 能加载权重并生成 |
| 11 | 量化实验 | 新增 inference-only 权重量化路径 | 比较 FP32/FP16/INT8/INT4 |

每次迭代都要守住三条线：

1. 模型主接口稳定：`input_ids [B,T] -> logits [B,T,V]`。
2. 配置和权重形状清楚：改 `vocab_size/n_embd/n_layer/n_head/block_size` 时知道哪些 checkpoint 会失效。
3. 训练、导出、推理分层：训练代码不直接依赖 vLLM/SGLang；推理引擎通过 adapter/后端认识模型结构。

这样你可以单独学习某一部分：今天只看 tokenizer，明天只看 RoPE，后天只看 KV cache，而不会因为所有模块互相耦合导致调试困难。

## 12. 如何打印一个真实输入的完整流转

项目里新增了一个观察脚本：

```bash
python scripts/inspect_flow.py --prompt "用户: 什么是 embedding?
助手:" --vocab-limit 80 --top-k 8
```

它会打印：

1. 当前 checkpoint 中的 `stoi` 和 `itos`。
2. prompt 中每个字符如何被 encode 成 token id。
3. `input_ids [B,T]`。
4. `token_embedding`、`position_embedding`、每个 Transformer block、`ln_f`、`lm_head` 的输出 shape。
5. 最后一个位置 `logits[0,-1,:]` 对应的 top-k 下一个 token 候选。

如果要打印完整词表：

```bash
python scripts/inspect_flow.py --full-vocab
```

这能把抽象的主线落到真实数据上：

```text
文本 -> stoi/encode -> input_ids [B,T]
-> token_embedding + position_embedding -> hidden states [B,T,C]
-> Transformer blocks -> ln_f -> lm_head
-> logits [B,T,V] -> top-k token 候选 -> itos/decode
```

## 13. 第 9 组问题：标准 tokenizer、embedding、batch 与 logits

### 标准 HF tokenizer 的几个组件是什么

Hugging Face tokenizer 不是只有一个 `stoi/itos` 字典。一个标准 tokenizer 通常包含这几层：

| 组件 | 作用 | 例子 |
| --- | --- | --- |
| normalizer | 在正式切 token 前清洗/规范化文本 | Unicode 规范化、大小写处理、去重音、空白符规范化 |
| pre-tokenizer | 先把文本切成粗粒度片段 | 按空格/标点切分、byte-level 切分、正则切分 |
| model / vocab | 把片段继续切成 subword 并映射成 token id | BPE、WordPiece、Unigram、SentencePiece |
| decoder | 把 token id 还原成文本 | 合并 BPE 片段、处理空格标记、byte 到 unicode |
| special tokens | 有特殊控制含义的 token | `<bos>`、`<eos>`、`<pad>`、`<unk>`、`<|system|>`、`<|user|>`、`<|assistant|>` |
| chat template | 把多轮消息格式化成模型训练时见过的 prompt | system/user/assistant 消息转成带特殊 token 的字符串或 token ids |

MiniLLM 当前的 `CharTokenizer` 只有最简单的两张表：

```text
stoi: token -> id
itos: id -> token
```

它没有 normalizer、pre-tokenizer、标准 decoder、完整 special tokens 和 chat template，所以它适合教学，但还不是标准 HF tokenizer。

### token id 如何变成 128 维向量

token id 本身只是整数，不包含可计算的语义。比如：

```text
"用" -> id 252
```

进入模型后，`nn.Embedding(vocab_size, 128)` 会维护一张可训练表：

```text
embedding.weight: [vocab_size, 128]
```

当输入 token id 是 252 时，它会取出第 252 行：

```text
embedding.weight[252] -> [128]
```

所以变化是：

```text
离散整数 id -> 128 个 float 组成的连续向量
```

这个过程本质是查表，不是把数字 252 直接按数学公式变成 128 维。开始训练前，这一行向量通常是随机初始化的；训练过程中，反向传播会不断更新它。随着训练进行，经常出现在相似上下文里的 token，它们的向量会逐渐学到有用的关系。

对整个 batch 来说：

```text
input_ids:       [B,T]
token_embedding: [B,T,128]
```

### 用户问题变长时发生什么

如果只有一个用户输入，但问题更长，主要变化是 `T` 变大：

```text
短问题: input_ids [1, 20]
长问题: input_ids [1, 200]
```

`B` 仍然是 1，因为还是一个 prompt；`T` 变大，因为 token 数更多。

注意 attention 的 prefill 计算会随 `T` 明显变重。普通全量 self-attention 的注意力矩阵是 `[T,T]`，所以单条 prompt 变长时，attention 部分大致会按 `T^2` 增长。这就是长上下文推理更吃显存和算力的原因之一。

### 一次输入 50 个和 100 个 prompt 有什么不同

如果一次处理多个 prompt，主要变化是 `B` 变大：

```text
50 个 prompt:  input_ids [50,T]
100 个 prompt: input_ids [100,T]
```

如果这些 prompt 长度不同，标准 HF 流程通常会 padding 到同一个 batch 内的最大长度，并用 attention mask 告诉模型哪些位置是真的 token，哪些是 padding。vLLM/SGLang 这类推理引擎为了效率，通常不会简单粗暴地按最大长度 padding，而是用 sequence metadata、KV cache block 和 scheduler 管理不同长度请求。

batch size 在两个场景里含义略有不同：

| 场景 | batch 代表什么 |
| --- | --- |
| 训练 | 一次并行处理多少条训练样本 |
| 普通 PyTorch 推理 | 一次并行处理多少条 prompt |
| serving engine | 可以理解为多个用户请求被 scheduler 动态合批处理 |

所以，是的，在 serving 场景里，batch size 可以模拟“不同用户的输入一起处理”。但真实 vLLM/SGLang 不只是固定 `[B,T]`；它还会动态合批、拆分 prefill/decode、管理 KV cache。

### 既然只预测最后一个位置，为什么 logits 还输出所有位置

这个问题要分训练和生成两个阶段看。

训练阶段不是只预测最后一个位置。训练样本通常是：

```text
input:  [t0, t1, t2, t3]
target: [t1, t2, t3, t4]
```

模型会在每个位置都预测下一个 token：

```text
位置 0: 看 t0       -> 预测 t1
位置 1: 看 t0,t1    -> 预测 t2
位置 2: 看 t0,t1,t2 -> 预测 t3
位置 3: 看 t0..t3   -> 预测 t4
```

因此训练时需要：

```text
logits:  [B,T,V]
targets: [B,T]
```

这样一个长度为 `T` 的样本能提供 `T` 个预测监督。如果只输出最后一个位置 `[B,V]`，那每条训练序列只能贡献 1 个监督信号，数据利用率会低很多。

生成阶段确实只需要最后一个位置：

```python
next_token_logits = logits[:, -1, :]
```

也就是：

```text
logits [B,T,V] -> last logits [B,V]
```

所以你说的 `[B, vocab_size]` 在生成阶段是可以的。很多高性能推理引擎内部也会尽量只计算/保留真正需要采样的位置，尤其是 decode 阶段使用 KV cache 后，每次新输入通常只有一个 token，形状接近：

```text
decode input_ids: [B,1]
decode logits:    [B,1,V] -> [B,V]
```

但 MiniLLM 的 `forward()` 同时服务训练、评估和简单生成，所以返回完整 `[B,T,V]` 更通用、更容易理解。后续如果专门做 inference fast path，可以再增加只返回最后 logits 的接口。

## 14. 第 10 组问题：embedding 表、隐藏层宽度和模型参数

### 为什么每个 token 要映射成 embedding

token id 是离散编号，本身没有可计算的语义。比如：

```text
"用" -> 252
"e"  -> 36
```

数字 252 并不比 36 “更大、更重要、更接近某个意思”。如果直接把 id 当数字喂给神经网络，模型会误以为 token 之间有大小关系。

embedding 的作用是给每个 token 一组可训练的连续特征：

```text
token id -> 128 维 float 向量
```

这样模型后续的 attention、MLP、LayerNorm 就可以在连续向量空间里做矩阵乘法、相似度计算和梯度更新。

### embedding 大小如何选择

`n_embd=128` 是模型隐藏层宽度，也叫 hidden size / channel size。它表示每个 token 在模型内部用多少个 float 表示。

它不是由 `vocab_size` 自动决定的，而是一个超参数：

```text
vocab_size = 有多少种 token
n_embd     = 每个 token 用多少维表示
```

选择原则是工程权衡：

| n_embd 较小 | n_embd 较大 |
| --- | --- |
| 参数少，训练快，显存低 | 参数多，表达能力强 |
| 容易欠拟合 | 更吃数据、算力、显存 |
| 适合教学/小语料 | 适合大模型/大语料 |

还要满足结构约束：

```text
n_embd % n_head == 0
head_dim = n_embd / n_head
```

当前 MiniLLM 用 `n_embd=128`、`n_head=4`，所以每个 attention head 的维度是：

```text
head_dim = 128 / 4 = 32
```

真实大模型的 hidden size 通常更大，例如几千维，但代价也会大很多。很多层里的线性层参数量会近似随 `n_embd^2` 增长。

### embedding 内部的 float 会变化吗

会，但只在训练时变化。

训练时：

```text
forward -> loss -> backward -> optimizer.step()
```

如果某个 token 出现在 batch 中，它对应的 embedding 行会收到梯度，优化器会更新这 128 个 float。

推理时：

```text
model.eval()
torch.no_grad()
```

embedding 参数不再更新，只是被查表读取。

所以要区分两类东西：

| 名称 | 是否是参数 | 是否保存到 checkpoint | 是否每次输入都变 |
| --- | --- | --- | --- |
| `token_embedding.weight` | 是 | 是 | 训练时被优化器更新，推理时固定 |
| hidden state / activation | 否 | 否 | 每次 forward 根据输入临时计算 |

### 128 维里的数字是随机的吗

一开始是随机初始化的，但不是随便乱填。MiniLLM 里 `_init_weights()` 使用小方差正态分布：

```python
nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

这样做有两个目的：

1. 打破对称性：不同 token、不同神经元不能一开始完全一样。
2. 控制数值尺度：初始值太大会让激活和梯度不稳定。

训练之后，这些数字就不再是纯随机数，而是被数据和 loss 调整过的参数。每一维通常没有人工命名的含义，不应该理解成“第 1 维表示语法、第 2 维表示情绪”。它们是模型自己学出来的分布式特征。

当前 checkpoint 中真实的几行 embedding 例子：

```text
token "用" id=252 前 8 维:
[0.088637, -0.138615, -0.099947, 0.023800, -0.046104, 0.084862, 0.103014, -0.101036]

token "e" id=36 前 8 维:
[-0.017501, -0.114939, 0.090685, 0.101131, 0.065660, 0.091073, -0.138648, -0.071040]
```

### nn.Embedding(vocab_size, 128) 的表注册在哪里

在 `MiniGPT.__init__()` 里：

```python
self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
```

因为 `MiniGPT` 继承自 `nn.Module`，当你把一个子模块赋值给 `self.token_embedding` 时，PyTorch 会自动注册它。

`nn.Embedding` 内部有一个参数：

```text
token_embedding.weight: [vocab_size, n_embd]
```

当前 checkpoint 中是：

```text
token_embedding.weight: [339, 128]
```

它会出现在：

```python
model.state_dict()["token_embedding.weight"]
```

也会被保存到 checkpoint：

```text
checkpoints/minillm.pt
```

当前 MiniLLM 还做了 weight tying：

```python
self.lm_head.weight = self.token_embedding.weight
```

这表示输入 embedding 表和输出 `lm_head` 共享同一块参数。模型输入时用它把 token id 变成向量；模型输出时也用它把 hidden state 投回词表 logits。

### 什么时候把 token 映射成 128 维向量

发生在 `MiniGPT.forward()` 的开头：

```python
positions = torch.arange(seq_len, device=idx.device)
x = self.token_embedding(idx) + self.position_embedding(positions)
```

其中：

```text
idx:                   [B,T]
self.token_embedding:  [B,T,128]
self.position_embedding: [T,128]
x:                     [B,T,128]
```

从这一行开始，模型就不再直接处理 token id，而是处理 128 维 hidden state。

### 后续计算会改变隐藏层吗

会。这里的“改变隐藏层”要理解成改变临时激活 `x`：

```text
x0 = token embedding + position embedding
x1 = block0(x0)
x2 = block1(x1)
x3 = ln_f(x2)
logits = lm_head(x3)
```

每一层都会生成新的 hidden state。它们是 forward 过程中临时产生的张量，不是模型长期保存的参数。推理时这些 hidden state 会随着不同输入而变化，但模型参数不变。

训练时，forward 产生 hidden state，backward 用这些中间结果计算梯度，最后 optimizer 更新参数。

### 模型参数到底指什么

模型参数不是“某一次输入产生的 hidden state”，而是模型长期保存、训练会更新的权重。

MiniLLM 的参数包括：

```text
token_embedding.weight
position_embedding.weight
blocks.*.ln_*.weight / bias
blocks.*.attn.c_attn.weight / bias
blocks.*.attn.c_proj.weight / bias
blocks.*.mlp.*.weight / bias
ln_f.weight / bias
lm_head.weight  # 当前和 token_embedding.weight 共享
```

一句话区分：

```text
参数 parameters：模型学到并保存的东西。
激活 activations / hidden states：一次 forward 根据输入临时算出来的东西。
梯度 gradients：训练时根据 loss 算出来，用来更新参数的东西。
```
