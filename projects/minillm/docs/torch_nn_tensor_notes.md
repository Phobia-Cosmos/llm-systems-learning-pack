# PyTorch、nn.Module、Tensor 与 KV Cache 问题记录

本文记录当前关于本地 MiniLLM、Agent、KV cache、`torch.Tensor`、`torch.nn`、`nn.Module` 的问题和回答。它偏向学习笔记，不是 API 手册。

## 1. Agent 在没有 GPU 的设备上如何运行

### 问题

既然 LLM 推理需要使用 KV cache，那么在没有 GPU 的设备上，我们的 agent 是如何运行的？是不是这个 agent 只是请求 API 服务，并没有在本地部署模型？

### 回答

要先区分三件事：

- Agent 逻辑：负责组织任务、维护消息历史、调用工具、决定下一步动作。
- 模型推理：把 prompt 输入模型，生成下一个 token。
- 推理服务：模型可能运行在本机，也可能运行在远程 API 后面。

如果一个 agent 只是调用 OpenAI、DeepSeek、Qwen、Claude 或其他远程 API，那么本地没有真正跑大模型。本地只是在做这些事：

- 保存对话历史。
- 拼接 prompt 或 messages。
- 发 HTTP 请求。
- 解析模型返回。
- 调用本地工具，例如搜索、读写文件、运行命令。

这种情况下，KV cache 在远程服务商或远程推理服务器内部产生和管理，本地机器不需要有 GPU，也不需要保存模型的 KV cache。

如果 agent 使用本地模型，例如本地 Transformers、vLLM、SGLang、llama.cpp 或 Ollama，那么模型推理就在本地发生。此时 KV cache 会存在于本地设备上：

- 模型在 GPU 上运行，KV cache 通常在 GPU 显存里。
- 模型在 CPU 上运行，KV cache 通常在 CPU 内存里。
- 模型被量化或 offload，KV cache 的位置取决于推理框架的实现。

所以“没有 GPU 还能运行 agent”并不矛盾。因为很多 agent 本质上是“控制程序 + API 调用”，不是“本地模型本体”。

## 2. CPU 上训练模型时会产生 KV cache 吗

### 问题

如果我们要在 CPU 上训练模型，KV cache 也还会产生吗？

### 回答

通常不会。至少 MiniLLM 当前的训练代码不会使用 KV cache。

KV cache 主要服务于自回归生成阶段。生成时模型一次只新增一个 token，如果每一步都重新计算整段上下文的 K/V，会非常浪费。KV cache 会保存过去 token 的 key/value，下一步只计算新 token 的 Q/K/V，然后和旧 K/V 拼起来继续生成。

训练阶段通常不同。以 decoder-only LLM 的 next-token prediction 为例，训练时会一次性输入整段序列：

```text
input:  x0 x1 x2 x3
target: x1 x2 x3 x4
```

模型对整段序列并行计算 logits，然后用交叉熵比较每个位置的预测和目标。因为所有位置都在同一次 forward 里并行计算，训练时一般不需要“逐 token 增量生成”，因此 KV cache 没有推理阶段那种价值。

但是训练会产生另一类缓存：autograd 为反向传播保存的中间激活。这不是 KV cache。

训练时内存主要消耗在：

- 模型参数。
- 前向传播中间激活。
- 梯度。
- 优化器状态，例如 AdamW 的一阶、二阶动量。
- batch 数据。

推理时内存主要消耗在：

- 模型参数。
- 当前输入。
- KV cache。
- 少量采样状态。

一句话：KV cache 是推理生成的优化；训练时更重要的是 autograd 保存的中间激活。

## 3. MiniLLM 当前是否使用 KV cache

### 问题

MiniLLM 当前代码里有没有 KV cache？

### 回答

当前没有。

MiniLLM 的 `generate()` 每生成一个 token，都会取最近 `block_size` 个 token，然后重新跑一次完整 forward：

```python
idx_cond = idx[:, -self.config.block_size :]
logits, _ = self(idx_cond)
logits = logits[:, -1, :]
```

这段逻辑在 `minillm/model.py` 的 `MiniGPT.generate()` 中。它更容易理解，但效率不高。

如果有 KV cache，生成逻辑会变成：

1. 第一步对完整 prompt 做 prefill，得到所有层的 past key/value。
2. 后续每一步只输入最新 token。
3. 每层 attention 复用过去的 key/value。
4. 每生成一个 token，就把新的 key/value 追加到 cache。

教学项目一开始不加 KV cache 是合理的。因为 KV cache 会让 attention 的 forward 签名、shape、mask、position 处理都变复杂。

## 4. torch.Tensor 是什么

### 问题

`torch.tensor` 和 `torch.Tensor` 到底是什么？为什么文本 token 要变成 tensor 才能进入模型？

### 回答

`torch.Tensor` 是 PyTorch 的核心数据结构，可以理解为带有设备、类型和自动求导能力的多维数组。

一个 Tensor 至少包含这些信息：

- `shape`：形状，例如 `[batch, seq_len]` 或 `[batch, seq_len, hidden]`。
- `dtype`：数据类型，例如 `torch.long`、`torch.float32`、`torch.bfloat16`。
- `device`：所在设备，例如 `cpu`、`cuda:0`、`mps`。
- `requires_grad`：是否需要梯度。
- `grad_fn`：如果它是计算结果，记录反向传播需要的信息。
- `stride`：张量在内存里的步长，影响 `view()`、`transpose()`、`contiguous()` 等操作。

Python 的 `list[int]` 只是普通数据，不能直接进入 `nn.Embedding` 或矩阵乘法。模型需要的是 Tensor，因为 PyTorch 需要知道：

- 数据在哪个设备上算。
- 用什么 dtype 算。
- 形状如何广播和矩阵乘法。
- 是否记录计算图用于反向传播。

MiniLLM 生成脚本中有这段：

```python
token_ids = tokenizer.encode(prompt)
idx = torch.tensor([token_ids], dtype=torch.long, device=device)
```

这里 `tokenizer.encode(prompt)` 得到 `list[int]`，例如：

```python
[12, 8, 31, 44]
```

外层再包一层 `[]` 是为了加 batch 维：

```python
[[12, 8, 31, 44]]
```

最终变成 shape 为 `[1, 4]` 的 LongTensor。`nn.Embedding` 要求输入 token id 是整数类型，所以这里用 `torch.long`。

## 5. Tensor 的 shape 怎么看

### 问题

MiniLLM 里经常看到 `[B, T, C]`、`[B, H, T, D]`，它们代表什么？

### 回答

在语言模型里常见约定是：

- `B`：batch size，一次处理多少条序列。
- `T`：sequence length，一条序列有多少 token。
- `C`：channel 或 hidden size，每个 token 的向量维度。
- `H`：attention head 数量。
- `D`：每个 head 的维度，通常 `D = C // H`。

在 MiniLLM 的 attention 里，输入 `x` 是：

```text
[B, T, C]
```

经过 Q/K/V 线性层后仍然是 `[B, T, C]`，然后拆成多头：

```text
[B, T, H, D]
```

再转置成：

```text
[B, H, T, D]
```

这样每个 batch、每个 head 都可以独立做：

```text
softmax(QK^T / sqrt(D)) V
```

## 6. torch.nn 是什么

### 问题

为什么 `nn` 中会有如此多东西？它到底是什么？

### 回答

`torch.nn` 是 PyTorch 的神经网络模块库。它不是一个单独的模型，而是一组用于搭建模型的标准组件。

它里面东西很多，是因为神经网络训练需要反复处理这些共性问题：

- 怎么声明有参数的层。
- 怎么把子模块组合成大模型。
- 怎么保存和加载权重。
- 怎么把模型移动到 CPU/GPU。
- 怎么区分训练模式和推理模式。
- 怎么实现常见层、激活函数、损失函数和容器。

常见类别包括：

- 基类：`nn.Module`
- 可训练参数：`nn.Parameter`
- 线性层：`nn.Linear`
- Embedding：`nn.Embedding`
- 归一化：`nn.LayerNorm`、`nn.BatchNorm1d`、`nn.GroupNorm`
- 正则化：`nn.Dropout`
- 激活函数：`nn.GELU`、`nn.ReLU`、`nn.SiLU`
- 容器：`nn.Sequential`、`nn.ModuleList`、`nn.ModuleDict`
- 损失函数：`nn.CrossEntropyLoss`、`nn.MSELoss`

MiniLLM 里用到的主要是：

```python
import torch.nn as nn
import torch.nn.functional as F
```

`nn` 里更多是“带状态或可注册的模块”，例如 `nn.Linear`、`nn.Dropout`、`nn.LayerNorm`。

`F` 里更多是“函数式操作”，例如 `F.softmax`、`F.cross_entropy`。

两者不是绝对分离。例如 `nn.GELU()` 是模块，`F.gelu(x)` 是函数。模块形式适合放进模型结构里，函数形式适合在 forward 里临时调用。

## 7. nn.Module 是什么

### 问题

`nn.Module` 是什么？为什么我们的模型类要继承它？

### 回答

`nn.Module` 是 PyTorch 中所有神经网络模块的基类。只要一个类继承了 `nn.Module`，它就能获得 PyTorch 模型需要的一组能力。

这些能力包括：

- 自动注册参数。
- 自动注册子模块。
- 自动注册 buffer。
- 支持 `model.parameters()`。
- 支持 `model.state_dict()`。
- 支持 `model.load_state_dict()`。
- 支持 `model.to(device)`。
- 支持 `model.train()` 和 `model.eval()`。
- 支持通过 `model(x)` 自动调用 `forward(x)`。

一个最小例子：

```python
import torch
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)

model = TinyModel()
x = torch.randn(3, 4)
y = model(x)
```

关键点是 `self.linear = nn.Linear(4, 2)`。因为 `linear` 是一个 `nn.Module`，PyTorch 会自动把它注册为子模块。于是：

```python
list(model.parameters())
```

能找到 `linear.weight` 和 `linear.bias`。

如果不继承 `nn.Module`，PyTorch 就不知道你的类里面有哪些参数，也无法自动保存、加载、移动设备和训练。

## 8. 为什么要写 super().__init__()

### 问题

继承 `nn.Module` 后为什么要在 `__init__` 里写 `super().__init__()`？

### 回答

`nn.Module` 自己需要初始化一些内部容器，用来记录：

- `_parameters`
- `_modules`
- `_buffers`
- hooks
- train/eval 状态

当你写：

```python
super().__init__()
```

这些内部结构才会准备好。之后你再写：

```python
self.linear = nn.Linear(4, 2)
```

PyTorch 才能把它注册进去。

如果忘记调用 `super().__init__()`，通常会出现参数无法注册、模型行为异常或直接报错。

## 9. Parameter、Buffer、普通 Tensor 的区别

### 问题

模型里有些东西是 `nn.Parameter`，有些是 `register_buffer`，有些只是普通 Tensor，它们区别是什么？

### 回答

三者区别如下：

| 类型 | 会被训练 | 会进 state_dict | 会跟随 to(device) | 典型用途 |
| --- | --- | --- | --- | --- |
| `nn.Parameter` | 是 | 是 | 是 | 权重、bias |
| buffer | 否 | 默认是 | 是 | mask、running stats、位置编码缓存 |
| 普通 Tensor 属性 | 否 | 否 | 通常否 | 临时值、非模型状态 |

MiniLLM 中 causal mask 是 buffer：

```python
self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size), persistent=False)
```

它不是训练出来的参数，不需要梯度。但它需要跟着模型移动到同一设备，否则 CPU tensor 和 GPU tensor 混用会报错。

`persistent=False` 表示它不保存进 `state_dict`。因为 causal mask 可以根据 `block_size` 重新生成，不是模型学到的知识。

## 10. model(x) 为什么会调用 forward

### 问题

为什么代码里可以写 `model(x)`，而不是 `model.forward(x)`？

### 回答

因为 `nn.Module` 实现了 `__call__`。当你写：

```python
y = model(x)
```

PyTorch 实际会走 `nn.Module.__call__`，再调用你的 `forward()`。

这样做不是多此一举。`__call__` 在调用 `forward()` 前后还会处理：

- hooks。
- mixed precision/autocast 相关上下文。
- module call 追踪。
- 一些框架集成逻辑。

因此推荐总是写：

```python
y = model(x)
```

而不是手动写：

```python
y = model.forward(x)
```

## 11. train() 和 eval() 做什么

### 问题

`model.train()` 和 `model.eval()` 为什么不用我们自己写？

### 回答

它们来自 `nn.Module`。

`model.train()` 会把模型切到训练模式。`model.eval()` 会把模型切到评估模式。

它们主要影响某些层的行为：

- `Dropout`：训练时随机丢弃一部分激活；eval 时关闭。
- `BatchNorm`：训练时使用当前 batch 统计；eval 时使用保存的 running statistics。

MiniLLM 里有 `Dropout`，所以训练和推理时需要切换模式：

```python
model.train()
model.eval()
```

注意：`eval()` 不等于“不计算梯度”。如果要推理时不保存梯度，还要使用：

```python
with torch.no_grad():
    ...
```

或者：

```python
@torch.no_grad()
def generate(...):
    ...
```

## 12. state_dict 是什么

### 问题

checkpoint、`state_dict`、模型参数之间是什么关系？

### 回答

`state_dict` 是一个字典，保存模型中所有参数和持久 buffer 的名字到 Tensor 的映射。

例如可能包含：

```text
token_embedding.weight
blocks.0.attn.c_attn.weight
blocks.0.attn.c_proj.weight
blocks.0.ln_1.weight
lm_head.weight
```

MiniLLM 训练脚本保存的是一个更大的 checkpoint：

```python
torch.save(
    {
        "model": model.state_dict(),
        "config": asdict(config),
        "tokenizer": tokenizer.to_dict(),
        "args": vars(args),
    },
    ckpt_path,
)
```

这里：

- `"model"` 是权重。
- `"config"` 是模型结构参数。
- `"tokenizer"` 是字符表。
- `"args"` 是训练参数记录。

推理时需要先按 config 重建模型结构，再用 `load_state_dict()` 把权重装进去。

## 13. 为什么需要 config

### 问题

既然 checkpoint 里有权重，为什么还需要 config？

### 回答

权重只是一堆 Tensor。PyTorch 需要先知道模型结构，才能知道这些 Tensor 应该放在哪里。

例如 `c_attn.weight` 的 shape 取决于 `n_embd`：

```python
nn.Linear(n_embd, 3 * n_embd)
```

如果训练时 `n_embd=128`，推理时却用 `n_embd=256` 创建模型，权重 shape 就对不上。

所以推理流程通常是：

1. 加载 checkpoint。
2. 从 checkpoint 读取 config。
3. 用 config 创建同结构模型。
4. 加载 `state_dict`。
5. 加载 tokenizer。
6. 输入 prompt，生成 token。

## 14. CPU 和 GPU 对 PyTorch 模型的影响

### 问题

GPU 上训练的模型能不能在 CPU 上运行？

### 回答

可以。模型参数本质上是 Tensor。只要模型结构一致，CPU 可以加载 GPU 训练出的权重。

关键是加载时要指定：

```python
torch.load(path, map_location="cpu")
```

MiniLLM 的 `generate.py` 已经根据 `--device cpu` 选择 device，并传给 `torch.load(..., map_location=device)`。

差别主要是速度：

- CPU 可以运行，但矩阵乘法慢。
- GPU 更适合大规模并行矩阵计算。
- 小模型和教学实验可以 CPU 跑。
- 大模型训练通常需要 GPU。

## 15. 为什么 nn.Linear、Embedding、LayerNorm 都是 Module

### 问题

为什么这些层都要做成 `nn.Module`？

### 回答

因为它们通常有状态。

`nn.Linear` 有：

- `weight`
- `bias`

`nn.Embedding` 有：

- embedding table，也就是 `weight`

`nn.LayerNorm` 有：

- `weight`
- `bias`
- 归一化逻辑

这些参数需要：

- 被优化器找到。
- 保存进 checkpoint。
- 加载回来。
- 跟随 `.to(device)` 移动。
- 参与 `train/eval` 状态管理。

这正是 `nn.Module` 提供的能力。

## 16. nn.Sequential 和 ModuleList 的区别

### 问题

为什么 `nn` 中还有 `Sequential`、`ModuleList`？

### 回答

它们是模块容器。

`nn.Sequential` 适合简单的顺序网络：

```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),
    nn.GELU(),
    nn.Linear(4 * n_embd, n_embd),
    nn.Dropout(dropout),
)
```

调用时：

```python
y = self.net(x)
```

会自动按顺序执行每一层。

`nn.ModuleList` 只是保存一组模块，不规定 forward 顺序。MiniLLM 的 block 用它：

```python
self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
```

然后自己写循环：

```python
for block in self.blocks:
    x = block(x)
```

为什么不能直接用 Python list？

因为普通 list 里的模块不会被 PyTorch 自动注册。这样 `model.parameters()` 可能找不到它们，优化器也就不会更新这些层。

## 17. torch.nn.functional 是什么

### 问题

为什么有 `torch.nn.functional as F`，它和 `nn` 有什么区别？

### 回答

`torch.nn.functional` 里是函数式接口。它通常不保存自己的参数，只负责执行一次计算。

例如 MiniLLM 用：

```python
weights = F.softmax(scores, dim=-1)
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```

这些操作没有需要注册的权重，因此用函数式接口很自然。

如果一个操作有可训练参数或需要长期状态，通常用 `nn.Module`。如果只是一次数学运算，通常可以用 `F`。

## 18. MiniLLM 中一次训练 step 发生了什么

### 问题

`train.py` 中一次训练到底发生了什么？

### 回答

核心流程是：

```python
x, y = get_batch(...)
_, loss = model(x, y)
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

含义如下：

1. `get_batch()` 取一批 token 序列。
2. `model(x, y)` 前向传播，计算 logits 和 loss。
3. `zero_grad()` 清掉上一步残留梯度。
4. `loss.backward()` 反向传播，计算每个参数的梯度。
5. `clip_grad_norm_()` 限制梯度范数，避免训练不稳定。
6. `optimizer.step()` 用梯度更新参数。

模型本身只负责“给定输入如何计算输出”。优化器负责“根据梯度如何修改参数”。

## 19. MiniLLM 中一次生成 step 发生了什么

### 问题

`generate.py` 中一次生成到底发生了什么？

### 回答

生成流程是：

1. tokenizer 把 prompt 变成 token id。
2. token id 变成 LongTensor。
3. 模型计算最后一个位置的 logits。
4. 根据 greedy、temperature、top-k 选择下一个 token。
5. 把新 token 拼到序列末尾。
6. 重复直到达到 `max_new_tokens`。
7. tokenizer 把 token id 解码成文本。

当前 MiniLLM 每一步都会重新计算最近 `block_size` 个 token，没有 KV cache。

这让代码更简单，但推理复杂度更高。后续如果要接入 vLLM/SGLang/nano-vLLM，KV cache 是必须理解的扩展点。

## 20. 当前结论

### 问题

当前在本地没有 GPU、MiniLLM 参数也暂时缺失的情况下，应该如何理解整个链路？

### 回答

当前结论是：

- Agent 可以不在本地跑模型，只请求远程 API。
- 如果请求远程 API，本地不管理模型 KV cache。
- 如果本地 CPU 跑模型，推理阶段也可以有 KV cache，只是存在 CPU 内存里。
- CPU 训练通常不用 KV cache，而是保存 autograd 中间激活用于反向传播。
- MiniLLM 当前代码没有 KV cache，训练和生成都可以在 CPU 上跑。
- 缺失 MiniLLM checkpoint 不影响继续学习代码结构。
- `torch.Tensor` 是模型计算的基础数据结构。
- `torch.nn` 是神经网络组件库。
- `nn.Module` 是 PyTorch 模型对象的核心协议。

后续建议学习顺序：

1. 熟悉 `torch.Tensor` 的 shape、dtype、device。
2. 理解 `nn.Module` 如何注册参数和子模块。
3. 看懂 MiniLLM 的 `MiniGPT.forward()`。
4. 看懂训练循环里的 `loss.backward()` 和 `optimizer.step()`。
5. 看懂 `generate()` 为什么可以逐 token 生成。
6. 再给 MiniLLM 增加 KV cache。
7. 最后再考虑 HF Transformers、vLLM、SGLang 接入。
