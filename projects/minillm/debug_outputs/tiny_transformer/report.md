# MiniLLM 极小 Transformer：从语料到生成的完整流转

> [!info] 怎么阅读
> 这份主报告只解释“为什么从 A 变成 B”。完整浮点 Tensor 已移到 [tensor_dump.md](tensor_dump.md)，需要核对某个具体数值时再打开。

> [!question] 本轮问题（已回答）
> - [x] 随机截取 16 个长度 8 的窗口、training step、next-token 样本分别是什么？
> - [x] `[B,T,C]` 中的 C、QKV Linear、`Linear(C,4C)` 是什么？
> - [x] `c_proj`、`mlp.net.0`、`mlp.net.2` 分别做什么？
> - [x] Q/K/V 能否不融合，以及每一步对应哪条公式？
> - [x] 为什么逐 token 生成默认只有 5 个 step？

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

## 9. 自动校验

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
