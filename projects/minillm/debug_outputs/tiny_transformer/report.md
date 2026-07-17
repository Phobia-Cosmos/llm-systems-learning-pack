# MiniLLM 极小 Transformer：从语料到生成的完整流转

> [!info] 怎么阅读
> 这份主报告只解释“为什么从 A 变成 B”。完整浮点 Tensor 已移到 [tensor_dump.md](tensor_dump.md)，需要核对某个具体数值时再打开。

> [!question] 本轮问题（已回答）
> - [x] 随机截取 16 个长度 8 的窗口、training step、next-token 样本分别是什么？
> - [x] `[B,T,C]` 中的 C、QKV Linear、`Linear(C,4C)` 是什么？
> - [x] `c_proj`、`mlp.net.0`、`mlp.net.2` 分别做什么？
> - [x] Q/K/V 能否不融合，以及每一步对应哪条公式？
> - [x] 为什么逐 token 生成默认只有 5 个 step？
> - [x] 同一个 prompt 在 nano-vLLM 与 mini-sglang 中怎样完成 prefill、KV-cache decode、调度和返回？

**先直接回答最容易混淆的两个 step：**

| 名称 | 一次 step 做什么 | 本次默认值 |
| --- | --- | --- |
| optimization/training step | 随机取一个 batch → forward → loss → backward → AdamW 更新一次参数 | 总共 300 步 |
| generation token step | 用当前上下文 forward → 只取最后位置 logits → 选 1 个 token → 拼回上下文 | `max_new_tokens=5`，所以每个 prompt 正好 5 步 |

这次运行里 `step 0` 只是训练前评估，不更新参数；`step 1` 特意使用固定的 `B=1,T=5` 样本，方便精确比较第一次更新前后；`step 2～300` 才使用随机的 `B=16,T=8` batch。16 个窗口可以重叠，甚至可以重复，它们不是把语料平均切成 16 份。

“随机截取 `16` 个长度 `8` 的窗口”是指：每个 optimization step 从 96-token 语料中随机选择 16 个起点；从每个起点连续取 8 个 token 组成 x，再右移一位组成 y。因此 x/y shape 都是 `[16,8]`，一个 step 同时监督 `16×8=128` 个 next-token 预测。窗口不是把一个词切成 8 份，而是连续 8 个字符 token。

一个长度 8 窗口的实际 x/y 对齐如下：

| 窗口内位置 t | 输入 x[t] | 目标 y[t]=下一个 token |
| --- | --- | --- |
| 0 | 小 | 猫 |
| 1 | 猫 | 吃 |
| 2 | 吃 | 鱼 |
| 3 | 鱼 | 。 |
| 4 | 。 | \n |
| 5 | \n | 小 |
| 6 | 小 | 狗 |
| 7 | 狗 | 吃 |

必须构造 next-token 样本，是因为 decoder-only LLM 把整段文本概率分解为：

$$p(z_0,\ldots,z_{N-1})=\prod_{t=0}^{N-2}p_\theta(z_{t+1}\mid z_0,\ldots,z_t).$$

文本右移一位就自动提供了标签，不需要人工标注。同一个窗口的 8 个位置可以并行训练；causal mask 保证第 t 个位置看不到未来答案。若目标仍是输入本身，模型只需复制当前 token，学不到续写。生成时则把 next-token 规则逐 token 使用。

## 1. 语料、词表、训练样本

语料共有 `96` 个字符；CharTokenizer 去重后得到 `11` 个 token（含 `<unk>`）。**语料是训练内容，词表只是 token 与整数 id 的双向映射。**

| id | token | Unicode | 出现次数 |
| --- | --- | --- | --- |
| 0 | <unk> | special | 0 |
| 1 | \n | U+000A | 16 |
| 2 | 。 | U+3002 | 16 |
| 3 | 吃 | U+5403 | 8 |
| 4 | 喝 | U+559D | 8 |
| 5 | 小 | U+5C0F | 16 |
| 6 | 水 | U+6C34 | 8 |
| 7 | 狗 | U+72D7 | 8 |
| 8 | 猫 | U+732B | 8 |
| 9 | 肉 | U+8089 | 4 |
| 10 | 鱼 | U+9C7C | 4 |

固定追踪样本（为了训练前后始终比较同一输入）是：

| 位置 | 输入 token | 输入 id | 目标 token | 目标 id |
| --- | --- | --- | --- | --- |
| 0 | 小 | 5 | 猫 | 8 |
| 1 | 猫 | 8 | 吃 | 3 |
| 2 | 吃 | 3 | 鱼 | 10 |
| 3 | 鱼 | 10 | 。 | 2 |
| 4 | 。 | 2 | \n | 1 |

## 2. B、T、C、H、D 到底是什么

| 符号 | 含义 | 训练时 | 固定 trace 时 |
| --- | --- | --- | --- |
| B | batch size，一次并行多少条窗口 | 16 | 1 |
| T | 每条窗口有多少个 token 位置 | 8 | 5 |
| C | channel/hidden size；**每个 token 用多少个连续特征数表示** | 8 | 8 |
| H | attention head 数 | 2 | 2 |
| D | 每个 head 的特征维度，D=C/H | 4 | 4 |
| Vocab | 模型可以预测多少种 token | 11 | 11 |

所以 `[B,T,C]` 不是三种数据：它表示一个三维张量。训练时 `[16,8,8]` 中共有 16 条窗口，每条 8 个 token，每个 token 已从整数 id 变成 8 维向量。C=8 只是为了教学可观察；真实模型常是几千维。

## 3. 模块名称先对上含义

| 代码名 | 真实作用 |
| --- | --- |
| blocks.0 | 第 0 个（也是唯一一个）Transformer block；`.0` 是 Python 从 0 开始的索引。 |
| blocks.0.attn.q_proj / k_proj / v_proj（三个独立 Linear） | 把 LN1 输出分别变成 Q、K、V。教学默认不融合。 |
| blocks.0.attn.c_proj | attention 的输出投影 $W_O$；作用于 `weights @ V` 拼头后的结果，不生成 Q/K/V。 |
| blocks.0.mlp.net.0 | MLP 的第一个 Linear：$C=8 \rightarrow 4C=32$，扩大逐 token 特征空间。 |
| blocks.0.mlp.net.1 | GELU 激活；没有参数，所以参数表里看不到 weight。 |
| blocks.0.mlp.net.2 | MLP 的第二个 Linear：$4C=32 \rightarrow C=8$，恢复残差主干宽度。 |
| blocks.0.mlp.net.3 | Dropout；本调试配置 dropout=0，因此数值不变。 |

`mlp.net` 是 `nn.Sequential`，数字只是执行顺序下标：0→1→2→3。只有 0 和 2 是 Linear，所以参数表只出现 `.0.weight` 与 `.2.weight`。`Linear(C,4C)` 的目的不是跨 token 交流；它对每个 token 单独扩展特征，再用 GELU 产生非线性组合，最后降回 C。跨 token 交流发生在 attention。

## 4. 一次 forward 的逐步流转与公式

下面每一行的输出，就是下一行的输入；这才是整个 Transformer 主线：

```text
LN1(hidden)
 ├─ q_proj → Q → 分头 → RoPE ─┐
 ├─ k_proj → K → 分头 → RoPE ─┼→ QKᵀ → mask → softmax = A
 └─ v_proj → V → 分头 ─────────┘                       │
                                                       A @ V
                                                         ↓
                                                 拼头 → W_O → 残差
```

| 步骤 | 计算 | 公式 | shape 变化 | 与下一步的关系 |
| --- | --- | --- | --- | --- |
| 0 | 原始文本 → token id | $i_t=\operatorname{tokenizer}(text_t)$ | 文本 → `[B,T]`；训练时 B=16, T=8 | 离散文本先变成可查表的整数。 |
| 1 | token id → embedding | $H_0=E[i]$ | `[B,T] → [B,T,C]`，C=8 | 每个 token id 查出一个 C 维向量。 |
| 2 | Attention 前归一化 | $X_1=\operatorname{LN}_1(H_0)$ | `[B,T,C] → [B,T,C]` | 只调整每个 token 向量的尺度，shape 不变。 |
| 3 | 生成 Q/K/V | $Q=X_1W_Q^{\top}+b_Q$；$K=X_1W_K^{\top}+b_K$；$V=X_1W_V^{\top}+b_V$ | 三份 `[B,T,C]` | 三者读取同一个 $X_1$，但使用不同参数，因此含义不同。 |
| 4 | 拆成多个 head | $C=H\times D$ | 每份 `[B,T,8] → [B,2,T,4]` | 这里只 reshape/transpose，不学习、不改变元素值。 |
| 5 | 加入 RoPE | $Q_m'=R_mQ_m,\ K_n'=R_nK_n$ | Q/K shape 不变；V 不变 | 位置只改变 Q/K 的方向，使内积含相对距离。 |
| 6 | Q 和 K 匹配 | $S=Q'K'^{\top}/\sqrt D$ | `[B,H,T,D] × [B,H,D,T] → [B,H,T,T]` | 每个 query 位置得到对每个 key 位置的分数。 |
| 7 | causal mask | $S_{m,n}=-\infty\ (n>m)$ | `[B,H,T,T]` | 未来位置变成 -∞，所以 softmax 后权重为 0。 |
| 8 | 分数变权重 | $A=\operatorname{softmax}(S)$ | `[B,H,T,T]` | 每一行和为 1，表示当前位置怎样分配读取比例。 |
| 9 | 读取 V | $Z=AV$ | `[B,H,T,T] × [B,H,T,4] → [B,H,T,4]` | A 决定读多少，V 提供真正被读取的内容。 |
| 10 | 拼头并通过 $W_O$ | $O=\operatorname{Concat}(Z_1,\ldots,Z_H)W_O^{\top}+b_O$ | `[B,H,T,D] → [B,T,C]` | `c_proj` 混合各 head，并恢复残差所需的 C 维。 |
| 11 | 第一个残差 | $H_1=H_0+O$ | `[B,T,C]` | 原信息与 attention 读取的信息相加。 |
| 12 | MLP 前归一化 | $X_2=\operatorname{LN}_2(H_1)$ | `[B,T,C]` | 为逐 token MLP 稳定尺度。 |
| 13 | MLP 扩维 | $U=X_2W_{up}^{\top}+b_{up}$ | `[B,T,8] → [B,T,32]` | Linear(C,4C) 给每个 token 更多中间特征。 |
| 14 | MLP 非线性 | $G=\operatorname{GELU}(U)$ | `[B,T,32]` | 若没有 GELU，两个 Linear 合起来仍只是一个 Linear。 |
| 15 | MLP 降维 | $M=GW_{down}^{\top}+b_{down}$ | `[B,T,32] → [B,T,8]` | 恢复 C 维，才能与残差主干相加。 |
| 16 | 第二个残差 | $H_2=H_1+M$ | `[B,T,C]` | 这就是一个完整 Transformer block 的输出。 |
| 17 | 最终归一化与词表投影 | $L=\operatorname{LN}_f(H_2)E^{\top}$ | `[B,T,C] → [B,T,Vocab=11]` | 每个位置得到词表中每个 token 的 logit。 |
| 18 | Next-token loss | $\mathcal L=-\frac1{BT}\sum_{b,t}\log p(y_{b,t}\mid x_{b,\le t})$ | 一个标量 | 一次 batch 同时产生 B×T 个预测训练信号。 |
| 19 | 反向传播与更新 | $\nabla_\theta\mathcal L\rightarrow\operatorname{AdamW.step}()$ | 参数 shape 不变、数值改变 | 更新 W/embedding/norm；下一次 forward 才产生变化后的 Q/K/V。 |

## 5. 为什么默认不用 fused QKV

当前报告对应的真实模型参数就是三个独立模块：`q_proj.weight`、`k_proj.weight`、`v_proj.weight`。这与论文公式一一对应，便于学习。

不融合完全可以：分别调用三个 `Linear(C,C)` 即可。生产模型常融合成一个 `Linear(C,3C)`，是为了让 GPU 少读取两次相同的 X，并减少 kernel launch；它不是 Transformer 理论要求，也不改变 Q/K/V 的值。想对比时运行：`python scripts/debug_tiny_transformer.py --fused-qkv`。

## 6. Q/K/V 在训练中究竟怎样变化

```text
持久参数 θ(step s)
    ↓ forward
临时激活 Q/K/V → attention → logits → loss
    ↓ backward
参数梯度 ∂loss/∂θ
    ↓ optimizer.step()
持久参数 θ(step s+1)
    ↓ 下一次 forward
重新计算出新的 Q/K/V
```

Q/K/V 是 **activation（一次 forward 的中间结果）**，不是 optimizer 直接更新的长期参数。对第 s 个 optimization step：

$$Q^{(s)}=X_1^{(s)}(W_Q^{(s)})^\top+b_Q^{(s)}.$$

AdamW 更新的是 embedding、$W_Q/W_K/W_V$、$W_O$、MLP、Norm 等参数；下一次 forward 中，输入 $X_1$ 和投影参数都变了，于是重新算出的 Q/K/V 才发生变化。

| 中间量 | 第 1 次更新后的 L2 变化 | 训练完成后的 L2 变化 | 变化链条 |
| --- | --- | --- | --- |
| q_flat | 0.297521 | 12.229281 | $X_1,W_Q,b_Q$ 共同决定 |
| k_flat | 0.296556 | 11.445688 | $X_1,W_K,b_K$ 共同决定 |
| v_flat | 0.410147 | 3.077194 | $X_1,W_V,b_V$ 共同决定 |
| q_heads | 0.297521 | 12.229281 | Q reshape 后再经 RoPE |
| k_heads | 0.296556 | 11.445689 | K reshape 后再经 RoPE |
| attention_weights | 0.003935 | 1.959946 | 由变化后的 Q/K 经 score、mask、softmax 得到 |

关系链应这样读：`参数/上游 hidden 改变 → Q/K/V 改变 → scores 改变 → attention weights 改变 → 读取的 V 改变 → block 输出改变 → logits/loss 改变`。完整 before/after 数字见 [tensor_dump.md](tensor_dump.md)。

## 7. Loss 是否真的让预测变好

| optimization step | 全语料平均 loss |
| --- | --- |
| 0 | 2.431446 |
| 1 | 2.379675 |
| 25 | 1.138771 |
| 50 | 0.501263 |
| 75 | 0.330016 |
| 100 | 0.306658 |
| 125 | 0.300941 |
| 150 | 0.296943 |
| 175 | 0.296295 |
| 200 | 0.298371 |
| 225 | 0.290272 |
| 250 | 0.255013 |
| 275 | 0.226439 |
| 300 | 0.196331 |

| 位置 | 输入 | 正确下一个 token | 最终 top-1 | 概率 |
| --- | --- | --- | --- | --- |
| 0 | 小 | 猫 | 猫 | 0.694 |
| 1 | 猫 | 吃 | 吃 | 0.518 |
| 2 | 吃 | 鱼 | 鱼 | 0.974 |
| 3 | 鱼 | 。 | 。 | 0.996 |
| 4 | 。 | \n | \n | 0.997 |

## 8. 为什么生成是 5 个 step

脚本默认 `--max-new-tokens 5`，所以每个 prompt 只追加 5 个 token。**生成 step 数等于要求新增的 token 数，与训练的 300 个 optimization step 完全不是一回事。**例如 `小猫吃` 后依次生成 `鱼 → 。 → \n → 小 → 猫`，所以正好 5 步。prompt 有 3 个 token，`3+5=block_size=8`，也恰好满足教学 KV-cache 的总长度限制。当前极小词表没有 `<eos>`，模型不会自行停止，5 是人为停止条件。可改成 `--max-new-tokens 20`；普通 generate 会使用最近 block_size 个 token 的滑动窗口，而教学 KV-cache 路径需要更大的 block_size 或滑动 cache 才能继续。

| prompt | 完整 greedy 输出 |
| --- | --- |
| '小猫吃' | '小猫吃鱼。\n小猫' |
| '小狗吃' | '小狗吃肉。\n小猫' |
| '小猫喝' | '小猫喝水。\n小猫' |
| '小狗喝' | '小狗喝水。\n小猫' |

| prompt | token step | 本步输入 context | top-3 | 本步选中 |
| --- | --- | --- | --- | --- |
| '小猫吃' | 0 | '小猫吃' | 鱼:0.974, 肉:0.017, \n:0.004 | 鱼 |
| '小猫吃' | 1 | '小猫吃鱼' | 。:0.996, 小:0.002, 吃:0.001 | 。 |
| '小猫吃' | 2 | '小猫吃鱼。' | \n:0.997, 鱼:0.001, 小:0.001 | \n |
| '小猫吃' | 3 | '小猫吃鱼。\n' | 小:0.994, 。:0.002, 吃:0.002 | 小 |
| '小猫吃' | 4 | '小猫吃鱼。\n小' | 猫:0.542, 狗:0.453, 鱼:0.002 | 猫 |
| '小狗吃' | 0 | '小狗吃' | 肉:0.962, 鱼:0.031, 吃:0.005 | 肉 |
| '小狗吃' | 1 | '小狗吃肉' | 。:0.995, 肉:0.004, \n:0.001 | 。 |
| '小狗吃' | 2 | '小狗吃肉。' | \n:0.996, 。:0.003, 小:0.001 | \n |
| '小狗吃' | 3 | '小狗吃肉。\n' | 小:0.994, 。:0.002, 吃:0.002 | 小 |
| '小狗吃' | 4 | '小狗吃肉。\n小' | 猫:0.542, 狗:0.453, 鱼:0.002 | 猫 |
| '小猫喝' | 0 | '小猫喝' | 水:0.997, 喝:0.001, \n:0.000 | 水 |
| '小猫喝' | 1 | '小猫喝水' | 。:0.996, 水:0.001, 肉:0.001 | 。 |
| '小猫喝' | 2 | '小猫喝水。' | \n:0.996, 。:0.002, 小:0.001 | \n |
| '小猫喝' | 3 | '小猫喝水。\n' | 小:0.994, 。:0.002, 吃:0.002 | 小 |
| '小猫喝' | 4 | '小猫喝水。\n小' | 猫:0.543, 狗:0.452, 鱼:0.002 | 猫 |
| '小狗喝' | 0 | '小狗喝' | 水:0.989, 喝:0.008, 猫:0.001 | 水 |
| '小狗喝' | 1 | '小狗喝水' | 。:0.994, 水:0.003, 肉:0.002 | 。 |
| '小狗喝' | 2 | '小狗喝水。' | \n:0.996, 。:0.002, 小:0.001 | \n |
| '小狗喝' | 3 | '小狗喝水。\n' | 小:0.994, 。:0.002, 吃:0.002 | 小 |
| '小狗喝' | 4 | '小狗喝水。\n小' | 猫:0.543, 狗:0.452, 鱼:0.002 | 猫 |

## 9. 同一个 prompt 如何流过 nano-vLLM 与 mini-sglang

这一节固定追踪同一个请求：prompt 是 `小猫吃`（3 个 token），最多新增 5 个 token，模型续写为 `鱼 → 。 → \n → 小 → 猫`。这里不再展示浮点 Tensor，只保留 shape、公式和“上一步输出为什么会成为下一步输入”。

> [!question] 这一节回答什么？
> - [x] 一个 prompt 进入推理服务后，在模型计算之前还经历什么？
> - [x] prefill 和 decode 分别是什么，为什么不是每步都重算整个 prompt？
> - [x] Q/K/V 在首次输入和逐 token 生成时分别怎样产生、使用和缓存？
> - [x] nano-vLLM 的 `slot_mapping`、`block_table`、paged KV cache 和调度器怎样配合？
> - [x] mini-sglang 的普通 KV cache 与 nano-vLLM 的分页缓存有什么本质差别？
> - [x] 明明生成 5 个 token，为什么实际是 1 次 prefill 加 4 次 decode forward？

> [!info] 先分清“模型”和“推理引擎”
> Transformer 决定 `embedding → Q/K/V → attention → MLP → logits` 的数学计算；推理引擎负责请求排队、batch、显存中的 KV 存放、选择执行 kernel、停止与返回结果。两个运行时使用的是同一套训练权重，所以模型学到的“`小猫吃` 后面更可能是 `鱼`”没有变化，变化的是组织计算与缓存的方式。

教学 checkpoint 中 Q/K/V 是三个独立的 `q_proj/k_proj/v_proj`。mini-sglang 直接加载这份 `.pt`；nano-vLLM 的 MiniGPT 后端使用融合的 `c_attn`，所以导出时把三组参数沿输出维拼成：

$$
W_{QKV}=\begin{bmatrix}W_Q\\W_K\\W_V\end{bmatrix},\qquad
b_{QKV}=\begin{bmatrix}b_Q\\b_K\\b_V\end{bmatrix}.
$$

于是一次 `Linear(C,3C)` 再切三段，与三次 `Linear(C,C)` 分别计算完全等价。**这是权重布局和执行效率的变化，不是注意力公式的变化。**

整个请求可以先压缩成这一条主线：

```text
HTTP/Python 请求
  ↓ tokenizer.encode
prompt token ids
  ↓ 请求状态与缓存空间准备
prefill：并行处理 prompt 的 3 个 token
  ↓ 最后位置 logits → 选出“鱼”
decode：只输入新 token“鱼”，历史 K/V 从 cache 读取
  ↓ logits → 选出“。”并把它作为下一轮输入
decode：“。” → “\n”
  ↓
decode：“\n” → “小”
  ↓
decode：“小” → “猫”
  ↓ 达到 max_tokens=5
释放请求缓存并返回 completion
```

五次 next-token 决策与五次模型 forward 的对应关系如下。注意最后采样出的 `猫` 已经是调用者要求的第 5 个新 token；除非还要预测第 6 个 token，否则无需再把 `猫` 喂进模型。

| 引擎 forward | 阶段 | 本轮真正送入模型的 token | 本轮写入 cache 后可见的历史 | 用本轮最后 logits 选出 |
| ---: | --- | --- | --- | --- |
| 0 | prefill | `小 猫 吃` | `小 猫 吃` | `鱼` |
| 1 | decode | `鱼` | `小 猫 吃 鱼` | `。` |
| 2 | decode | `。` | `小 猫 吃 鱼 。` | `\n` |
| 3 | decode | `\n` | `小 猫 吃 鱼 。 \n` | `小` |
| 4 | decode | `小` | `小 猫 吃 鱼 。 \n 小` | `猫` |

### 9.1 nano-vLLM：从请求队列到 paged KV cache

在任何 prompt 到来前，构造 `LLM` 已经完成一次性的启动工作：读取导出的 `config.json`，由 model registry 选择 MiniGPT 后端，在 CUDA 上构造模型并加载 `model.safetensors`，用 warmup 测量峰值显存，再按 `gpu_memory_utilization` 预分配全局 KV pool，最后创建 tokenizer、`Scheduler` 和 `BlockManager`。本例词表只有 11 项，当前 tensor-parallel 词表切分要求它能被 TP 数整除，所以实跑使用 `tensor_parallel_size=1`；`enforce_eager=True` 跳过 CUDA Graph capture，但 prefill/decode 仍全部在 GPU 上执行。

`LLM.generate()` 收到字符串和 `SamplingParams` 后，并不会直接调用 Transformer。它先执行下面的控制流：

```text
LLM.generate
  → add_request
  → tokenizer.encode
  → Sequence(status=WAITING)
  → Scheduler.schedule
  → BlockManager.allocate
  → ModelRunner.prepare_prefill / prepare_decode
  → GPU 上的 MiniGPT forward
  → Sampler 选一个 token
  → Scheduler.postprocess
  → 未结束：回到 schedule；结束：deallocate
```

`Sequence` 是 CPU 上的请求记录，保存 prompt/completion token、状态、温度、最大生成数以及逻辑页到物理页的映射。它**不是** KV Tensor；真正的 K/V 在 GPU 上由所有请求共享的预分配缓存池里。该池可概念化为：

$$[2,L,P,S,H_{kv},D]$$

其中 `2` 表示 K 与 V，`L` 是层数，`P` 是物理 block 数，`S` 是每个 block 的 token 槽位数，$H_{kv}$ 是 KV head 数，$D$ 是 head dimension。nano-vLLM 当前将 `S` 设为 256；本例总长度只有 8，仍会占用一个 256 槽物理页。浪费一点页尾空间，换来固定大小分配、回收、共享和较少显存碎片。

首次调度发生的是 **prefill**：

1. `Scheduler` 从 `waiting` 取出请求，检查本轮 token 预算和空闲 KV blocks；`BlockManager` 分配一个物理页，`Sequence` 从 `WAITING` 变为 `RUNNING`。
2. `prepare_prefill` 把同一批请求的 token 去掉补齐空位并压平成一条紧凑 token 流。单请求时相当于 `[小, 猫, 吃]`；多请求时则是多条序列首尾相接的 `[N]`，而不是训练讲解中的 `[B,T]`。
3. `cu_seqlens_q` 保存每条 packed 序列的累计边界。它告诉 FlashAttention “第几段属于哪条请求”，所以压平后不会让一条请求错误地注意到另一条请求。
4. `positions` 给出每个 token 的位置，供 RoPE 使用；`slot_mapping` 给出每个本轮 token 应写到哪个物理 KV 槽位。关系是 `物理槽位 = physical_block_id × block_size + 页内偏移`。
5. `block_table` 则从另一个方向回答“某个请求的第 i 个逻辑页实际在哪个物理页”。`slot_mapping` 用于**本轮新 K/V 的写入地址**，`block_table` 用于**读取该请求的整段历史 K/V**，二者不是重复字段。

> [!info] 为什么本例的 prefill 没有获得 prefix-cache 加速？
> nano-vLLM 只对填满的完整 block 建立可复用哈希。本例的 3-token prompt 远小于 256，连一个完整 block 都没有，因此即便系统具备 prefix cache，也没有可命中的完整页。这里仍会正常使用 KV cache 加速后续 decode，只是不能从另一个相同 prompt 请求复用已有前缀页。

进入 GPU 后，prefill 的模型内部仍对应第 4 节的 Transformer 公式，只是批次布局从 `[B,T,C]` 变为 packed 的 `[N,C]`：

| 模型内步骤 | nano-vLLM 中做什么 | 为什么交给下一步 |
| --- | --- | --- |
| Embedding | token ids 查表得到 `[N,C]` hidden states | Linear、Norm 只能处理连续向量，不能直接处理文字或 id |
| LN1 → fused QKV | `c_attn` 一次产生 `[Q\|K\|V]`，再切为三份 `[N,H,D]` | Q 用来提问，K 用来匹配，V 是匹配后读取的内容 |
| RoPE | 按各自 `positions` 旋转 Q 和 K，V 不旋转 | 让后续 QK 内积包含相对位置 |
| 写 KV cache | K/V 按 `slot_mapping` 散写进每层的 paged cache | decode 时可直接读取 prompt 历史，不再重算它们 |
| FlashAttention prefill | 依据 `cu_seqlens` 做 causal scaled dot-product attention | 每个 prompt 位置只能读取自己及之前位置的 V |
| 输出投影、残差、MLP | 与教学模型相同：`c_proj → residual → LN2 → MLP → residual` | 形成这一层处理后的 hidden states |
| Final Norm → LM head | prefill 只选每条请求的最后一个 hidden state 投影到词表 | 当前只需要“整个 prompt 之后的下一个 token”，无需返回前两个位置的 logits |

本机实跑时 prefill 进入 `flash_attn_varlen_func`，decode 进入 `flash_attn_with_kvcache`。FlashAttention 仍计算同一个 scaled dot-product attention，但用分块矩阵乘法与 online softmax 避免长期物化完整的 `[T,T]` score/weight 矩阵；这是内存访问方式的优化，不会改变 Q/K/V 的含义或公式。

prefill 中第 $i$ 个位置的核心公式仍是：

$$
q_i'=R_iq_i,\qquad k_j'=R_jk_j,\qquad
o_i=\sum_{j\le i}\operatorname{softmax}_j\!\left(\frac{q_i'{k_j'}^\top}{\sqrt D}\right)v_j.
$$

`ParallelLMHead` 只取 prompt 最后位置的 hidden state，得到一行词表 logits。nano-vLLM 的 `Sampler` 再用温度缩放、softmax 和 categorical sampling 选出 `鱼`。当前实现明确禁止严格的 `temperature=0` greedy，因此实测用极小正温度近似 greedy；这会让最高 logit 几乎占满概率，但原理上仍是随机采样，不应误称为严格 `argmax`。

采样结果从 GPU 转回 CPU 后，`Scheduler.postprocess` 才能更新 Python 中的请求状态；这是模型数据流重新进入引擎控制流的边界。

随后 `Scheduler.postprocess` 把 `鱼` 追加到 `Sequence.token_ids`。这个“刚刚选出的输出 token”立即成为下一次 forward 的输入，这就是自回归生成的闭环。

下一轮进入 **decode**，它与 prefill 最大的不同是：模型只接收 `[鱼]`，不再接收 `[小,猫,吃,鱼]` 全序列。

1. `Scheduler` 为每条 running 请求各安排 1 个 token；若刚好跨过页边界，`BlockManager.may_append` 再分配新页。本例始终在第一页内。
2. `prepare_decode` 读取 `Sequence.last_token=鱼`，给它位置 3；同时产生该位置的 `slot_mapping`、当前完整上下文长度和 `block_table`。
3. 模型只为 `鱼` 计算新的 $q_t,k_t,v_t$，并对 $q_t,k_t$ 应用位置 3 的 RoPE。$k_t,v_t$ 原地写入新的缓存槽，prompt 的 K/V 不动。
4. decode FlashAttention 根据 `block_table` 找到 `小、猫、吃、鱼` 的全部 K/V，但 query 只有当前 `鱼` 的 $q_t$，因此输出也只有一个 token 的 hidden state。
5. LM head 和 sampler 得到 `。`；它又成为下一轮唯一的新输入。后续 `。→\n→小→猫` 完全重复这个闭环。

这可以写成统一的增量公式：

$$
K_{\le t}=\operatorname{Concat}(K_{<t},k_t),\qquad
V_{\le t}=\operatorname{Concat}(V_{<t},v_t),
$$

$$
o_t=\operatorname{softmax}\!\left(\frac{q_t' {K_{\le t}'}^\top}{\sqrt D}\right)V_{\le t},\qquad
z_{t+1}=\operatorname{Select}(\operatorname{LMHead}(o_t)).
$$

公式写 `Concat` 是逻辑含义；nano-vLLM 的物理实现不会每轮复制整段历史，而是把新 K/V 原地写到分页缓存的空槽中。也只有 K/V 被缓存：未来 token 的新 Q 需要和所有历史 K 比较，历史 V 需要被加权读取；历史 Q 完成本位置的查询后，未来计算不会再使用，所以没有缓存价值。

RoPE 与 KV cache 并不冲突。历史 key 在产生时已经按它自己的绝对位置旋转，当前 query 按当前位置旋转，因此：

$$
(R_tq_t)^\top(R_jk_j)=q_t^\top R_t^\top R_jk_j=q_t^\top R_{j-t}k_j.
$$

内积最终依赖相对位移 $j-t$。因此缓存的是“已经带上正确位置的 K”；decode 不应把历史 K 再旋转一次。

达到第 5 个 completion token 后，`postprocess` 将状态设为 `FINISHED`，`BlockManager.deallocate` 降低页的引用计数并把空闲页归还池中。最后的 `猫` 已加入返回序列，但还没有作为模型输入产生 K/V；若请求第 6 个 token，下一轮才会为 `猫` 生成 Q/K/V。最终 `LLM.generate` 只 decode completion 部分并返回。

> [!info] CPU 与 GPU 的边界
> tokenizer、`Sequence`、请求队列、调度决策和 block 元数据主要在 CPU；embedding、Q/K/V、RoPE、paged KV Tensor、FlashAttention、MLP、LM head 和采样在 GPU。`enforce_eager=True` 只表示 decode 不走 CUDA Graph，并不表示模型可以在 CPU 运行；这个 nano-vLLM 后端仍是 CUDA/NCCL 推理引擎。

### 9.2 mini-sglang：同一数学过程，最朴素的服务与缓存

> [!question] mini-sglang 是缩小版 SGLang Runtime 吗？
> 不是。本仓库的 `mini-sglang` 是一个教学用的单文件 HTTP 包装器：`ThreadingHTTPServer + MiniGPT`。它模仿了 `/v1/completions` 和 `/v1/chat/completions` 接口，但没有 SGLang 的调度器、continuous batching、RadixAttention、paged KV cache 或流式执行状态机。理解这一点很重要，否则会把“HTTP 接口长得像”误认为“推理引擎内部也相同”。

服务启动时，`load_minillm` 从 `.pt` 读取 config、tokenizer 状态和模型权重，构造 `MiniGPT` 并切换到 `eval()`。本次实际使用 CPU。一个 completion 请求的真实控制流是：

```text
POST /v1/completions
  → BaseHTTPRequestHandler.do_POST
  → 读取并解析 JSON
  → _send_completion
  → _generate
  → tokenizer.encode(prompt)
  → MiniGPT.generate_with_kv_cache
  → tokenizer.decode(完整 ids)
  → 去掉 prompt 文本
  → 一次性返回 JSON
```

请求中 `greedy=true, kv_cache=true, max_tokens=5` 时，`_generate` 先把 3 个 prompt ids 组成 `[B,T]`，再调用 `generate_with_kv_cache`。它的首次 `forward_with_cache(prompt)` 就是语义上的 prefill：

1. token embedding 得到 `[B,T,C]`，这里没有为了跨请求 batch 而压成 `[N,C]`。
2. 唯一的 Transformer block 分别调用 `q_proj/k_proj/v_proj`，产生三份 Q/K/V，再拆成 `[B,H,T,D]` 并给 Q/K 应用 RoPE。
3. causal attention 同时计算 prompt 的所有位置；该层返回 hidden states，并把 `(K,V)` 元组放入 Python list。若有多层，list 中每层各有一对 K/V。
4. Final Norm 与 LM head 得到所有 prompt 位置的 logits；生成函数只读取 `logits[:, -1, :]`，严格 `argmax` 选出 `鱼`。

decode 时，生成函数把 `鱼` 作为 shape `[B,1]` 的 `idx_next` 传给：

```text
forward_with_cache(idx_next, past_key_values)
```

模型依据 `past_len` 自动把位置设为 3，只为 `鱼` 计算新 Q/K/V。每层随后执行逻辑上的：

```text
K = torch.cat((past_K, new_K), dim=token_axis)
V = torch.cat((past_V, new_V), dim=token_axis)
```

当前 `鱼` 的 Q 与拼接后的全部 K 做匹配，权重再读取全部 V，得到一个位置的输出与下一 token logits。`。` 被选中后再次作为 `[B,1]` 输入，如此重复。最后一次选出的 `猫` 同样不会再 forward，因为没有第 6 个 token 请求。

这里的 `torch.cat` 很直观，适合教学，但每增长一个 token 都要为更长 Tensor 重新分配空间并复制旧 K/V。nano-vLLM 则预先分配固定页，仅写入一个新槽位，并能让多个请求的逻辑序列映射到共享物理池；这正是“数学相同，缓存工程不同”。

如果设置 `kv_cache=false`，mini-sglang 会改用普通 `generate`：第 1 轮输入 `小猫吃`，第 2 轮输入 `小猫吃鱼`，第 3 轮输入 `小猫吃鱼。`……每一轮都重新计算所有历史 token 的 Q/K/V。输出可以相同，但重复计算随上下文增长越来越多。KV cache 的目的不是改变预测，而是避免这部分重复工作。

mini-sglang 还没有请求队列、显存页分配、prefix cache、请求抢占、batch 合并、取消或真正 streaming。`ThreadingHTTPServer` 可以为连接创建 handler 线程，但这些线程不会被一个模型调度器合并为 continuous batch；请求里的 `stream=true` 当前也没有执行分支，仍会在生成完成后返回一个完整 JSON。

### 9.3 把两条链放在一起理解

| 观察点 | nano-vLLM | mini-sglang |
| --- | --- | --- |
| 服务入口 | Python `LLM.generate` | HTTP `/v1/completions` |
| 请求状态 | `WAITING → RUNNING → FINISHED` | 一次 handler 函数调用，没有显式状态机 |
| 模型权重 | 导出后 fused QKV，数学上等价 | 直接加载 separate Q/K/V 教学 checkpoint |
| prompt 布局 | 多请求可 packed 为 `[N,C]`，用 `cu_seqlens` 分界 | 单次调用保持 `[B,T,C]` |
| prefill | FlashAttention varlen，并把 prompt K/V 写入全局分页池 | PyTorch attention，把各层 K/V 放入 Python list |
| decode 输入 | 每个 running 请求只交一个 `last_token` | 开启 cache 时同样只交 `idx_next` |
| KV 增长方式 | `slot_mapping` 指定空槽，原地写入 paged cache | `torch.cat(old,new)` 生成更长 Tensor |
| 历史定位 | `block_table + context_lens` | list 中 Tensor 的 token 轴顺序就是历史 |
| 跨请求能力 | 调度、batch、分页分配、完整块 prefix reuse、抢占 | 没有模型级调度与缓存共享 |
| 采样 | 极小温度近似 greedy，仍属于 categorical sampling | `greedy=true` 时严格 argmax |
| 停止与回收 | Scheduler 判断上限/EOS并归还物理页 | Python for-loop 到上限，局部 Tensor 随请求结束释放 |

> [!info] 已用真实运行而不是只读代码推测
> 导出的极小模型已在本机 RTX 4070 SUPER 上通过 nano-vLLM 实际生成 `鱼。\n小猫`；同一 `.pt` checkpoint 也已由 mini-sglang 在 CPU 上以 `kv_cache=true` 和 `kv_cache=false` 分别运行，两条路径得到相同文本。这验证了 fused/separate QKV 与两种 KV 存储方式没有改变模型语义。mini-sglang 的 `stream=true` 也实测仍返回一次完整响应。

最后可以用一句话记忆：**Transformer 负责算“下一个 token 是谁”，KV cache 负责不重复算历史，调度器负责决定“此刻替哪些请求算”，paged cache 负责决定“它们的历史放在哪里”。** mini-sglang 只展示第一、二件事的朴素版本；nano-vLLM 把第三、四件事也显式实现出来。

## 10. 自动校验

| 阶段 | 检查 | 结果 |
| --- | --- | --- |
| 训练前 | trace_logits_match_model_forward | PASS |
| 训练前 | trace_loss_matches_model_forward | PASS |
| 训练前 | qkv_outputs_match_projection_parameters | PASS |
| 训练前 | causal_future_attention_is_zero | PASS |
| 训练前 | attention_rows_sum_to_one | PASS |
| 第 1 步后 | trace_logits_match_model_forward | PASS |
| 第 1 步后 | trace_loss_matches_model_forward | PASS |
| 第 1 步后 | qkv_outputs_match_projection_parameters | PASS |
| 第 1 步后 | causal_future_attention_is_zero | PASS |
| 第 1 步后 | attention_rows_sum_to_one | PASS |
| 训练完成 | trace_logits_match_model_forward | PASS |
| 训练完成 | trace_loss_matches_model_forward | PASS |
| 训练完成 | qkv_outputs_match_projection_parameters | PASS |
| 训练完成 | causal_future_attention_is_zero | PASS |
| 训练完成 | attention_rows_sum_to_one | PASS |
| 保存/生成 | manual greedy trace matches MiniGPT.generate | PASS |
| 保存/生成 | ordinary greedy matches KV-cache greedy | PASS |
| 保存/生成 | default corpus patterns are learned | PASS |
| 保存/生成 | checkpoint reload reproduces final logits | PASS |
| 保存/生成 | checkpoint tokenizer roundtrip is identical | PASS |

完整浮点矩阵、逐 head Q/K/V、mask、softmax、梯度和参数更新值都保存在 [tensor_dump.md](tensor_dump.md)。主报告刻意不再用这些数字打断流程理解。
