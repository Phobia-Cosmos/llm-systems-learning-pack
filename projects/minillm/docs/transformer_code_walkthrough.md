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
