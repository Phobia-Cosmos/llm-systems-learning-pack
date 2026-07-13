# MiniLLM RoPE 实现、引擎对齐与后续路线

## 1. 当前实现状态

MiniLLM 现在支持两种位置编码：

```text
position_encoding = "learned" | "rope"
rope_theta = 10000.0
```

- `learned` 是默认值，用于无损加载旧 checkpoint。
- `rope` 不创建 `position_embedding.weight`，而是在每层 attention 内旋转 Q/K。
- native forward 与 native KV-cache decode 使用绝对 `positions`，逐 token logits 已验证一致。
- nano-vLLM 和 vLLM 的 MiniGPT backend 使用同一种 NeoX/Llama half-split RoPE。
- `mini-sglang` 直接加载 MiniLLM 模型，因此自动支持 RoPE checkpoint。
- 上游 SGLang 仍需新增 MiniGPT native backend，不能只靠 config 自动识别自定义结构。

## 2. RoPE 的数学关系

RoPE 来自 *RoFormer: Enhanced Transformer with Rotary Position Embedding*：
https://arxiv.org/abs/2104.09864

对位置 `m`、维度对 `i`，角频率为：

```math
\theta_i = \mathrm{base}^{-2i/d}, \qquad i=0,1,\ldots,d/2-1
```

旋转角度为：

```math
\phi_{m,i} = m\theta_i
```

将一个 head 的向量分成两半 `x=[x_1,x_2]`，NeoX/Llama 风格旋转是：

```math
\operatorname{RoPE}(x,m)=
\begin{bmatrix}
x_1\cos\phi_m-x_2\sin\phi_m\\
x_2\cos\phi_m+x_1\sin\phi_m
\end{bmatrix}
```

attention 仍然计算：

```math
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V
```

区别是先变换：

```math
Q_m' = R_m Q_m, \qquad K_n' = R_n K_n
```

于是点积包含相对位置 `m-n`：

```math
(Q_m')^\top K_n' = Q_m^\top R_{n-m}K_n
```

这解释了为什么 RoPE 加在 Q/K 而不是 V 上，也解释了它为何比单纯的绝对位置向量更自然地表达相对距离。

## 3. 代码数据流

```text
input_ids
  -> token_embedding
  -> LayerNorm
  -> fused c_attn Linear
  -> split Q, K, V
  -> reshape [B,H,T,D]
  -> RoPE(Q, K, absolute_positions)
  -> causal attention
  -> output projection
  -> residual + MLP
  -> ln_f -> lm_head -> logits
```

实现位置：

| 部分 | 文件 |
| --- | --- |
| RoPE reference | `minillm/rope.py` |
| native forward/KV cache | `minillm/model.py` |
| config/CLI | `minillm/config.py`, `train.py` |
| HF-like export | `export_hf_like.py` |
| nano-vLLM backend | `projects/nano-vllm/nanovllm/models/minigpt.py` |
| vLLM backend | `projects/vllm/vllm/model_executor/models/minigpt.py` |

KV cache 保存的是已经应用 RoPE 的历史 K 和未旋转的 V。decode 第 `t` 步只对新 Q/K 使用位置 `t`，再让新 Q 查询缓存中的所有历史 K/V。

## 4. 兼容性与约束

- `head_dim` 必须为偶数，因为旋转需要成对维度。
- `rope_theta` 必须大于 0。
- 旧 checkpoint 没有 `position_encoding` 字段时仍默认为 `learned`。
- learned checkpoint 包含 `position_embedding.weight`；RoPE checkpoint 不包含它，二者不能只改 config 后强行互换。
- tokenizer 与 RoPE 相互独立，但 tokenizer 决定 token 序列长度，从而影响位置范围和 attention 成本。
- 当前训练长度仍是 `block_size=128`。RoPE 消除了 learned position table，但不会自动让模型获得可靠的超训练长度泛化能力。

## 5. 已完成验证

| 验证 | 结果 |
| --- | --- |
| position 0 为恒等旋转 | 通过 |
| 旋转前后向量 L2 norm 不变 | 通过 |
| odd head_dim 配置拒绝 | 通过 |
| learned checkpoint 兼容 | 通过 |
| RoPE 模型无 absolute position 参数 | 通过 |
| full forward 与 token-by-token KV cache logits | 通过 |
| MiniLLM 测试 | 11 项通过 |
| nano-vLLM MiniGPT/RoPE 测试 | 7 项通过 |
| native 与 nano-vLLM 生成 | 通过 |
| native 与 vLLM greedy token IDs | 完全一致 |

验证 prompt：

```text
embedding 是把离散 token id 映射成
```

16 个生成 token 的结果：

```text
连续向量的查表过程
```

vLLM 排查时还修复了一个已有 loader 问题：checkpoint 使用 `mlp.net.0/net.2`，vLLM backend 参数名是 `mlp.fc_in/fc_out`。缺少映射会让 MLP 权重未加载；现在 loader 已显式映射并加入回归测试。

## 6. 推荐的最小化迭代路线

每次只替换一个结构，通过 config 保留旧实现，并要求新旧模式都能训练、保存、加载和生成。

| 顺序 | 迭代 | 主要学习目标 | 验收标准 |
| --- | --- | --- | --- |
| 1 | RMSNorm 可选后端 | normalization 与内存访问 | LayerNorm/RMSNorm 两种 checkpoint 均可回放 |
| 2 | SwiGLU 可选 MLP | gated MLP、参数量与 FLOPs | GELU/SwiGLU loss 和速度对比 |
| 3 | GQA 可选 attention | Q heads 与 KV heads 分离 | KV cache 显存下降，MHA 结果不回归 |
| 4 | 训练基础设施 | resume、LR schedule、AMP、grad accumulation | 中断后可复现续训，记录 val loss/ppl |
| 5 | SFT + response loss mask | 预训练与指令微调的目标差异 | prompt token 不计 loss，response 可稳定生成 |
| 6 | LoRA | 参数高效微调 | adapter 保存/合并，和 full fine-tune 对比 |
| 7 | 真正 HF PreTrainedModel | 标准生态契约 | `save_pretrained/from_pretrained` 往返一致 |
| 8 | SGLang native backend | RadixAttention/prefix cache | engine 能加载同一 export 并生成一致 token |
| 9 | 量化 | FP16/INT8/INT4 权重与 kernel | 报告质量、显存、首 token 延迟、吞吐 |
| 10 | Triton/CUDA 算子 | kernel 与端到端收益的关系 | microbenchmark 和 engine benchmark 都有数据 |

## 7. 近期优先级

下一批最合适的工作不是继续增加很多功能开关，而是先建立稳定基线：

1. 为当前 RoPE+BPE checkpoint 固定 validation loss、perplexity、生成样例和速度结果。
2. 实现可选 RMSNorm，并在 native、nano-vLLM、vLLM 三处保持权重与数值对齐。
3. 实现可选 SwiGLU，明确 intermediate size，避免参数量比较失真。
4. 补 checkpoint resume、cosine LR、warmup、AMP 和 gradient accumulation。
5. 再开始 SFT/loss mask。模型是否“更会回答问题”主要由数据和训练目标决定，不是只靠 RoPE/RMSNorm/SwiGLU。

性能评测至少记录：模型配置、dtype、prompt/output token 数、batch/concurrency、TTFT、ITL、tokens/s、峰值显存和输出 token IDs。只有 token IDs 或质量可接受时，速度对比才有意义。
