# Tokenizer、GPT 训练目标、解码方式与深度学习框架问题记录

本文记录当前关于 MiniLLM、Tokenizer、decoder-only GPT、解码方式、模型容量、数据质量、优化器，以及 PyTorch / TensorFlow 的问题和回答。

## 1. 除了 CharTokenizer，还有哪些 tokenizer

### 问题

除了 `CharTokenizer` 应该还有其他种类的分词器吧？分词器都有哪些种类？

### 回答

有。`CharTokenizer` 只是最简单的一类：每个字符一个 token。真实 LLM 通常不用纯字符级 tokenizer，而是使用 subword tokenizer。

常见 tokenizer 类型如下：

| 类型 | 基本单位 | 代表方法 | 优点 | 缺点 |
| --- | --- | --- | --- | --- |
| Character | 字符 | MiniLLM 当前实现 | 最简单，不容易遇到完全无法切分的字符 | 序列很长，效率低，语义粒度太细 |
| Word | 单词 | 传统 NLP 词表 | 直观，英文场景容易理解 | 词表巨大，未登录词问题严重，中文还要分词 |
| Subword | 子词片段 | BPE、WordPiece、Unigram | 真实 LLM 主流方案，兼顾词表大小和序列长度 | 训练和实现比字符级复杂 |
| Byte-level | 字节或字节子词 | GPT-2 BPE、ByteLevel BPE | 几乎所有文本都能表示，不容易 OOV | token 和人类直觉不完全一致 |
| SentencePiece | 原始文本直接训练 | Unigram/BPE | 不强依赖空格，适合多语言和中文 | 需要额外训练和配置 |
| Regex + BPE | 先正则切块再 BPE | GPT-2、tiktoken 类路线 | 工程成熟，压缩率和兼容性好 | 规则和特殊 token 需要谨慎维护 |

`CharTokenizer` 的核心优点是透明：你能一眼看懂 `encode()` 和 `decode()`。它适合教学，但不适合真实 LLM。

真实 LLM 更常见的是：

- GPT-2 / GPT-3：Byte-level BPE 路线。
- BERT：WordPiece。
- T5 / LLaMA 早期：SentencePiece。
- Qwen / GPT-4 类现代模型：更复杂的 BPE / byte fallback / 特殊 token 体系。

## 2. tokenizer 应该如何挑选

### 问题

分词器该如何挑选？

### 回答

选 tokenizer 要看目标，而不是只看“哪种高级”。

如果目标是学习 Transformer 代码：

- 选 `CharTokenizer`。
- 原因是最清楚，最少依赖，方便观察 token id、embedding、loss、decode 的关系。

如果目标是训练一个稍微像样的小 LLM：

- 选 BPE / SentencePiece / Hugging Face tokenizers。
- 词表可以从几千到几万。
- 中文小模型可以用较小 BPE 词表，例如 6k、16k、32k。

如果目标是接入 Hugging Face、vLLM、SGLang、llama.cpp：

- 优先使用标准 tokenizer 文件结构。
- 至少要有 `tokenizer.json`、`tokenizer_config.json`、special tokens、chat template。
- 不建议临时自定义一个只有 Python dict 的 tokenizer。

如果目标是兼容已有模型：

- 必须使用原模型 tokenizer。
- tokenizer 一变，token id 语义就变了，旧权重基本不能直接用。

挑选时重点看这些指标：

- 词表大小：越大 embedding/lm_head 参数越多，但序列更短。
- 压缩率：同一段文本被切成多少 token。
- OOV 能力：遇到新字符、新语言、符号是否能表示。
- 多语言支持：中文、英文、代码、数学符号混合是否稳定。
- 特殊 token：是否支持 BOS/EOS/PAD/UNK、system/user/assistant、tool call、think 标签。
- 生态兼容：是否能被 `AutoTokenizer`、vLLM、SGLang、llama.cpp 加载。

## 3. CharTokenizer 的编码是否有问题

### 问题

我们现在的编码是否有问题？

### 回答

作为教学代码，没有问题。作为真实 LLM，有明显限制。

MiniLLM 当前 `CharTokenizer` 的特点：

- 词表来自训练文本里的不同字符。
- 每个字符一个 token。
- 没见过的字符变成 `<unk>`。
- 没有标准 BOS/EOS/PAD。
- 没有 chat template。
- 不是 Hugging Face fast tokenizer。

它的问题主要是：

- 序列过长。同样一句话，字符级 token 数通常远多于 BPE。
- 泛化弱。训练语料没见过的字符会变成 `<unk>`。
- 不利于真实语义压缩。模型要从字符拼出词，再从词拼出句子。
- 不利于接入推理生态。vLLM/SGLang 通常期待标准 HF tokenizer。
- 对代码、数学、多语言、特殊格式的处理很弱。

所以：`CharTokenizer` 适合第一阶段学习，不适合作为长期路线。

## 4. 为什么 GPT 不需要双向注意力

### 问题

为什么 GPT 不需要双向注意力？

### 回答

GPT 的目标是生成文本。生成时模型只能看到已经生成的前文，不能看到未来 token。

例如正在生成：

```text
用户: 什么是 attention？
助手: attention 是
```

模型要预测下一个 token。它不能提前知道后面完整答案是什么。因此 GPT 使用 causal self-attention，只允许当前位置看自己和过去位置。

双向注意力适合理解任务，例如 BERT：

```text
我喜欢 [MASK] 苹果
```

BERT 可以同时看 `[MASK]` 左边和右边，用来填空、分类、抽取信息。它不是天然为从左到右生成设计的。

GPT 不用双向注意力，不是因为双向注意力没用，而是因为生成任务的约束决定了它不能看未来。

## 5. 为什么只需要预测下一个 token 的概率

### 问题

为什么只需要预测下一个词的概率？

### 回答

因为任意一段文本的联合概率可以按链式法则拆成一连串 next-token 条件概率：

```text
P(x1, x2, x3, ..., xT)
= P(x1) * P(x2 | x1) * P(x3 | x1,x2) * ... * P(xT | x1,...,xT-1)
```

所以只要模型学会：

```text
P(next_token | previous_tokens)
```

就能给整段文本建模，也能逐 token 生成新文本。

训练时并不是每次只训练一个位置。MiniLLM 会一次输入长度为 `T` 的序列，同时训练所有位置：

```text
input:  t0 t1 t2 t3
target: t1 t2 t3 t4
```

causal mask 保证第 2 个位置不能偷看第 3、4 个位置。这样既符合生成约束，又能并行训练所有位置。

## 6. 为什么可以这样训练

### 问题

为什么可以用 `x` 预测右移一位的 `y` 来训练？

### 回答

这是 teacher forcing。

训练时我们已经有完整文本，因此可以构造：

```text
x = 原文从第 0 个 token 到倒数第 2 个 token
y = 原文从第 1 个 token 到最后一个 token
```

模型看到 `x` 的每个前缀，预测对应位置的下一个 token。正确答案来自真实文本。

这和生成不同：

- 训练：真实前文已经给定，可以并行算所有位置 loss。
- 生成：没有真实后文，只能一步一步采样，把生成结果接回上下文。

这种训练目标简单、数据便宜、可扩展，所以 GPT 类模型能从海量普通文本中学习。

## 7. 模型开始为什么是随机权重

### 问题

我们的 MiniLLM 开始是随机权重，难道其他大模型训练时使用的是确定的权重？

### 回答

从零预训练的大模型一开始也是随机权重。

区别在于：

- 从零预训练：随机初始化，然后用海量数据训练。
- 微调：不是随机初始化，而是加载已有预训练模型权重，再在小数据上继续训练。
- 继续预训练：加载已有 checkpoint，再用新语料继续训练。

“随机”不等于完全不可控。通常会设置随机种子：

```python
torch.manual_seed(1337)
```

这样每次初始化可以复现。但权重值本质仍然是从某个随机分布采样出来的。

为什么不能全初始化成 0？

因为如果所有神经元参数一样，它们会得到相同梯度，永远学成相同功能。随机初始化能打破对称性。

## 8. 模型容量是什么意思

### 问题

模型容量指的是参数多少吗？

### 回答

参数量是模型容量的重要部分，但不是唯一部分。

模型容量可以理解为模型能表达复杂函数和存储统计规律的能力。它受这些因素影响：

- 参数量：总参数越多，通常容量越大。
- hidden size：每个 token 向量维度。
- layer 数：网络深度。
- attention head 数和 head_dim。
- MLP intermediate size。
- context length：能利用多长上下文。
- tokenizer 粒度：同样文本被切成多少 token。
- 架构设计：RMSNorm、RoPE、SwiGLU、GQA、MoE 等。

MiniLLM 参数少、字符级 token 序列长、语料小，所以容量和训练信号都很有限。它适合学习结构，不适合真实问答。

## 9. 足够模型容量代表什么

### 问题

“足够模型容量”具体代表什么？

### 回答

对一个任务来说，模型至少要有足够能力去表示数据中的规律。

例如训练一个小字符模型学会：

```text
用户: 什么是 attention？
助手: attention 是一种让 token 从上下文取信息的机制。
```

它需要学会：

- 字符组合成词。
- `用户:` 和 `助手:` 的格式。
- 问句和答句的对应关系。
- 标点和换行模式。
- 一些概念解释。

如果模型太小，它可能只能学会冒号、换行、重复字符，而学不会完整问答模式。

容量不够的表现包括：

- loss 降不下去。
- 输出重复字符。
- 格式学不稳。
- 回答经常断裂。
- 只能死记硬背少量片段。

容量太大但数据太少也有问题，会过拟合，记住训练集却不能泛化。

## 10. 隐藏状态是什么

### 问题

隐藏状态指的到底是什么？

### 回答

隐藏状态就是模型内部每个 token 的向量表示。

在 MiniLLM 中，token id 先经过 embedding：

```text
idx: [B, T]
embedding 后: [B, T, C]
```

这个 `[B, T, C]` 张量就是一组隐藏状态。每个 token 位置都有一个长度为 `C` 的向量。

经过每一层 TransformerBlock 后，隐藏状态都会更新：

```text
第 0 层输入 hidden states
第 1 层输出 hidden states
第 2 层输出 hidden states
...
最后 hidden states -> lm_head -> logits
```

直觉上，早期隐藏状态更接近字符/词形信息，后面层的隐藏状态会混入上下文关系、语法、任务格式、语义信息。但它不是人类可读的字符串，而是模型学出来的连续向量。

## 11. 结构化输入为什么能帮小字符模型

### 问题

为什么结构化输入可以让小字符模型快速学会模式？

### 回答

因为结构化输入降低了学习难度。

例如非结构化语料：

```text
attention transformer 模型 机制 用户 可以 文本 复杂 ...
```

模型很难知道哪些是问题、哪些是答案、哪里开始回复。

结构化语料：

```text
用户: 什么是 attention？
助手: attention 是让 token 从上下文取信息的机制。
```

重复出现后，小模型会更容易学到：

- `用户:` 后面通常是问题。
- `助手:` 后面通常是答案。
- 问答之间有换行。
- 回答常以解释句式开始。
- 相同问题附近经常出现相同关键词。

对于字符级 tokenizer，这尤其重要。因为它要先从字符层面学格式。清晰、重复、稳定的模板能让 loss 更快下降，输出更早出现像样结构。

## 12. 稳定解码方式是什么意思

### 问题

较稳定解码方式代表什么？

### 回答

解码方式决定如何从 logits 选择下一个 token。

模型输出的是每个候选 token 的 logits。解码策略把 logits 变成实际 token。稳定解码方式通常指更少随机、更少低概率 token、更少发散的策略。

稳定通常意味着：

- temperature 较低。
- top-k 或 top-p 截断低概率候选。
- greedy 或 beam search 不随机采样。
- 有 EOS / stop token 控制停止。
- 有 repetition penalty 减少重复。
- 有格式约束或结构化输出约束。

模型越弱，越需要稳定解码。因为弱模型的概率分布不可靠，随机采样很容易跑偏。

## 13. greedy 是什么意思

### 问题

`greedy` 是什么意思？

### 回答

greedy decoding 就是每一步都选择 logits 最高的 token：

```python
idx_next = torch.argmax(logits, dim=-1)
```

它不随机。

优点：

- 输出稳定。
- 每次结果相同。
- 适合检查模型是否学会基本格式。

缺点：

- 容易重复。
- 缺少多样性。
- 如果一步选错，后面也会被错误上下文带偏。

MiniLLM 早期训练后，建议先用 greedy 检查是否学会格式，再用采样看多样性。

## 14. 有哪些解码方式

### 问题

有哪些解码方式？

### 回答

常见解码方式如下：

| 解码方式 | 核心思想 | 优点 | 缺点 |
| --- | --- | --- | --- |
| Greedy | 每步选最高分 token | 稳定、简单 | 容易重复、无多样性 |
| Temperature sampling | 用温度调节分布后采样 | 可控制随机性 | 温度高会跑偏 |
| Top-k sampling | 只从最高 k 个 token 采样 | 排除低质长尾 token | k 难固定，可能过窄 |
| Top-p / nucleus | 只保留累计概率达到 p 的候选 | 自适应候选数量 | 实现稍复杂 |
| Beam search | 保留多个候选路径 | 翻译/摘要中常用 | 对开放聊天常死板 |
| Repetition penalty | 惩罚已出现 token | 减少复读 | 可能误伤必要重复 |
| Contrastive search | 同时考虑概率和退化惩罚 | 可减少无意义重复 | 参数和实现更复杂 |
| Typical sampling | 保留信息量典型的 token | 有时比 top-p 更自然 | 不如 top-p 常用 |
| Constrained decoding | 按语法/JSON/schema 约束输出 | 适合结构化输出 | 需要约束解析器 |
| Speculative decoding | 小模型草稿，大模型验证 | 加速推理 | 不是改变目标分布的普通采样策略 |

MiniLLM 当前支持：

- greedy
- temperature sampling
- top-k sampling

不支持：

- top-p
- repetition penalty
- EOS 停止
- beam search
- constrained decoding
- KV cache

## 15. 我们现在的解码方式有什么问题

### 问题

我们现在的解码方式有何问题？

### 回答

MiniLLM 当前解码方式作为教学够用，但有明显限制：

1. 没有 EOS / stop token。
   生成只能按 `max_new_tokens` 强制停止，模型不会自然知道什么时候结束。

2. 没有 repetition penalty。
   弱模型很容易输出重复字符，例如连续冒号、连续同一个字。

3. 没有 top-p。
   top-k 固定候选数量，不会根据分布尖锐程度自适应。

4. 没有 KV cache。
   每生成一个 token 都重新计算最近 `block_size` 的完整上下文，速度慢。

5. 使用字符级 tokenizer。
   输出以字符为单位采样，弱模型更容易出现局部重复。

6. 没有标准 chat template。
   只能靠语料中的 `用户:`、`助手:` 字符模式来学格式。

7. 没有非法 token 或格式约束。
   不能强制生成 JSON、工具调用、停止词等。

所以当前解码适合观察学习效果，不适合真实产品。

## 16. 如何算干净的数据

### 问题

如何算是干净的数据？

### 回答

干净数据不是指“内容高大上”，而是指训练信号可靠、格式一致、噪声少。

对 LLM 来说，干净数据通常满足：

- 编码正常，没有乱码、截断、HTML 残片、重复控制字符。
- 去重充分，没有大量重复样本。
- 格式一致，例如 role 字段、换行、标点规则稳定。
- 内容可信，低事实错误、低胡编。
- 任务和答案匹配，问答不串行、不错位。
- 语言质量好，语法和表达稳定。
- 长度分布合理，不全是极短或极长样本。
- 没有大量广告、导航、版权噪声、脚本残留。
- 尽量去除隐私、敏感信息、不可分发内容。
- 数据协议清楚，能合法训练和发布。

对 MiniLLM 这种小模型，数据干净尤其重要。因为容量小，噪声样本会直接占用它有限的学习能力。

## 17. 优化器有哪些

### 问题

优化器都有哪些？

### 回答

优化器负责根据梯度更新参数。常见优化器包括：

| 优化器 | 特点 | 常见用途 |
| --- | --- | --- |
| SGD | 最基础，沿负梯度方向更新 | 教学、小模型、部分视觉任务 |
| SGD + Momentum | 加动量，减少震荡 | CNN、传统深度学习 |
| Nesterov Momentum | 提前看一步的 momentum | 一些经典训练配置 |
| Adagrad | 高频参数学习率下降更快 | 稀疏特征、早期 NLP |
| RMSprop | 用梯度平方滑动平均调整学习率 | RNN、早期深度学习 |
| Adam | 自适应一阶/二阶动量 | 通用深度学习 |
| AdamW | decoupled weight decay 的 Adam | Transformer/LLM 主流 |
| Adafactor | 节省优化器状态内存 | 大模型、T5 系列 |
| Lion | 使用符号动量更新 | 一些大模型实验 |
| Muon | 面向矩阵参数的优化方向改进 | 现代 LLM 训练实验中出现 |

MiniLLM 用的是：

```python
torch.optim.AdamW(model.parameters(), lr=args.lr)
```

AdamW 是 Transformer 训练中非常常见的默认选择。

## 18. PyTorch 和 TensorFlow 有什么区别

### 问题

PyTorch 和 TensorFlow 有何区别？

### 回答

两者都是深度学习框架，都能做 tensor 计算、自动求导、模型训练、GPU 加速和部署。但设计气质不同。

| 维度 | PyTorch | TensorFlow |
| --- | --- | --- |
| 编程风格 | 更 Pythonic，动态图直观 | 早期偏静态图，后来也支持 eager |
| 调试体验 | 像普通 Python 一样逐行调试 | `tf.function` 图模式下调试更绕 |
| 模型接口 | `torch.nn.Module` | `tf.Module` / Keras `Model` |
| 自动求导 | autograd 动态记录计算图 | `tf.GradientTape` / graph gradients |
| 高性能编译 | `torch.compile`、TorchDynamo、Inductor | `tf.function`、AutoGraph、XLA |
| 部署生态 | TorchScript、ONNX、ExecuTorch、服务框架 | SavedModel、TF Serving、TFLite、TF.js |
| 研究使用 | 非常流行，改模型方便 | 工业部署历史强，Keras 易用 |
| LLM 生态 | 当前开源 LLM 训练/推理更偏 PyTorch | 仍可用，但主流 LLM repo 较少用原生 TF |

简化理解：

- PyTorch 更像“直接写 Python 程序，边运行边建计算图”。
- TensorFlow 更强调“把计算组织成可优化、可部署的图”。

现代 TensorFlow 也支持 eager，现代 PyTorch 也支持编译优化，所以差距比早年小。但开源 LLM 生态目前明显偏 PyTorch。

## 19. 为什么 PyTorch 中有 torch

### 问题

为什么 PyTorch 中会有 `torch`？

### 回答

`torch` 是 PyTorch 的主 Python 包名。

历史上 Torch 是一个较早的科学计算和深度学习框架，后来 PyTorch 继承了 Torch 的很多思想，并提供 Python-first 的接口。因此安装 PyTorch 后，代码里通常写：

```python
import torch
```

`torch` 里包含：

- Tensor 创建：`torch.tensor`、`torch.zeros`、`torch.randn`
- 数学运算：`torch.matmul`、`torch.softmax`
- 自动求导：`torch.autograd`
- 神经网络：`torch.nn`
- 优化器：`torch.optim`
- 数据：`torch.utils.data`
- 保存加载：`torch.save`、`torch.load`
- 设备管理：`torch.cuda`、`torch.backends`

所以 `torch` 是入口，`torch.nn`、`torch.optim` 是它下面的子模块。

## 20. 为什么其中会有 tensor

### 问题

为什么 PyTorch 中会有 tensor？

### 回答

深度学习的核心计算几乎都是张量计算。

例如：

- 一条 token 序列是 1D tensor。
- 一个 batch 是 2D tensor。
- embedding 后是 3D tensor。
- attention score 是 4D tensor。
- 模型参数也是 tensor。
- 梯度也是 tensor。

神经网络训练本质上是在做大量矩阵乘法、加法、归一化、softmax、loss、反向传播。这些都需要统一的数据结构。

Tensor 比 Python list 强很多：

- 支持 GPU / CPU 设备。
- 支持 dtype，例如 float32、bfloat16、int64。
- 支持自动求导。
- 支持高性能底层 kernel。
- 支持广播、矩阵乘法、reshape、transpose。
- 支持和 C++/CUDA 后端连接。

所以 tensor 是 PyTorch 的地基。没有 tensor，`nn.Linear`、`nn.Embedding`、attention、loss、optimizer 都无从谈起。

## 21. PyTorch 的架构大概是什么

### 问题

PyTorch 的架构是怎么样的？

### 回答

可以粗略分成几层：

```text
Python API
  torch / torch.nn / torch.optim / torch.utils.data
        |
Autograd
  动态记录计算图，执行反向传播
        |
Dispatcher
  根据 op、dtype、device 选择后端 kernel
        |
ATen / C++ Backend
  Tensor 实现、算子定义、CPU/CUDA 调用
        |
Kernels
  CPU kernels / CUDA kernels / cuDNN / cuBLAS / MPS / XLA 等
```

更现代的 PyTorch 还包括编译链路：

```text
torch.compile
  -> TorchDynamo 捕获 Python 程序片段
  -> AOTAutograd 处理前向/反向图
  -> TorchInductor 生成优化代码
  -> Triton/C++/CUDA kernel 执行
```

日常写 MiniLLM 时主要接触：

- `torch.Tensor`
- `torch.nn.Module`
- `torch.optim.AdamW`
- `loss.backward()`
- `model.to(device)`
- `torch.save/load`

底层 dispatcher、ATen、kernel 通常不用直接写，但它们负责真正执行计算。

## 22. TensorFlow 的架构大概是什么

### 问题

TensorFlow 的架构是怎么样的？

### 回答

TensorFlow 可以粗略理解为：

```text
Python API
  tf / tf.keras / tf.data
        |
Eager Execution 或 tf.function
  立即执行，或把 Python 函数转成计算图
        |
Graph / AutoGraph
  表示计算节点和依赖关系
        |
Runtime
  优化图，调度设备执行
        |
Kernels / XLA
  CPU/GPU/TPU kernel，或 XLA 编译优化
        |
SavedModel / TF Serving / TFLite / TF.js
  部署格式和服务生态
```

TensorFlow 早期最核心的思想是静态计算图：

1. 先定义计算图。
2. 再把数据喂进去执行。

这样有利于优化和部署，但调试不如普通 Python 直观。后来 TensorFlow 2 默认 eager execution，写法更接近 PyTorch；需要性能和部署时，再用 `tf.function` 转图。

Keras 是 TensorFlow 里的高级建模接口。很多用户实际写的是：

```python
model = tf.keras.Model(...)
model.compile(...)
model.fit(...)
```

而不是手写底层训练循环。

## 23. PyTorch 和 TensorFlow 与 MiniLLM 的关系

### 问题

这些框架和我们的 MiniLLM 有什么关系？

### 回答

MiniLLM 是用 PyTorch 写的，所以它依赖 PyTorch 的三件核心能力：

1. Tensor 计算。
2. `nn.Module` 组织模型。
3. autograd 自动反向传播。

MiniLLM 的代码：

```python
class MiniGPT(nn.Module):
    ...
```

说明它是一个 PyTorch 模型。

如果用 TensorFlow 重写，也可以实现同样结构，但写法会变成：

- `tf.Tensor`
- `tf.keras.layers.Dense`
- `tf.keras.Model`
- `tf.GradientTape`

模型数学可以一样，框架 API 和执行机制不同。

目前学习 LLM，更建议先坚持 PyTorch。原因是：

- MiniLLM 已经是 PyTorch。
- Hugging Face Transformers 主线偏 PyTorch。
- vLLM、SGLang、Megatron-LM、verl、TRL 都主要围绕 PyTorch 生态。
- 你后续要理解 `state_dict`、KV cache、推理引擎注册，PyTorch 路线更直接。

## 24. 当前阶段建议

### 问题

当前阶段该怎么学？

### 回答

建议按下面顺序：

1. 继续使用 `CharTokenizer` 理解 encode/decode。
2. 把 next-token prediction、causal mask、logits、cross entropy 串起来。
3. 用 greedy 观察模型是否先学会格式。
4. 再理解 temperature、top-k、top-p、repetition penalty。
5. 之后把 tokenizer 换成 BPE 或 Hugging Face tokenizer。
6. 给生成加入 EOS / stop token。
7. 再给 MiniLLM 加 KV cache。
8. 最后考虑 HF Transformers、vLLM、SGLang 接入。

不要太早把 tokenizer、KV cache、推理引擎、RL 后训练全部混在一起。先把“文本到 token，token 到 hidden state，hidden state 到 logits，logits 到 loss/采样”这条线吃透。
