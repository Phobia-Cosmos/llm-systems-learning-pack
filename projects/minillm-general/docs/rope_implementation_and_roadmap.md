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
| MiniLLM 测试 | 54 项通过、1 项 CUDA 测试跳过，另有 48 个 subtests 通过 |
| nano-vLLM MiniGPT 测试 | 15 项通过，另有 3 个 subtests 通过 |
| vLLM MiniGPT 直接单元测试 | 5 项通过（含融合 GQA loader）；当前机器未做 GQA CUDA engine 端到端实测 |
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

## 6. 结构组件公平基准

位置、Norm、MLP 和 MHA/GQA/MQA attention 已接入 `scripts/benchmark_components.py`。默认使用 3 个 seed、每个变体 100 次更新，输出逐 seed CSV/JSON 和 Markdown 聚合。公平性约束包括：train-only tokenizer、相同 raw split 与 batch schedule、相同公共参数初值、显式 AdamW 配置、完整 validation target 覆盖，以及固定 greedy prompt。

当前记录位于 `benchmarks/results/components_cpu_100step.{json,csv,md}`。这份 checked-in 记录早于 attention suite，共 36 条逐 seed 结果，普通生成和 KV-cache 生成 token IDs 全部一致。新默认矩阵为 15 个变体、45 条逐 seed 结果。由于模型约 21K 参数、语料约 1K 字符、训练仅 100 步，这组结果的用途是回归与教学，不是宣称某种结构在大模型上普遍更优。

## 7. 推荐的最小化迭代路线

每次只替换一个结构，通过 config 保留旧实现，并要求新旧模式都能训练、保存、加载和生成。

| 顺序 | 迭代 | 状态 | 主要学习目标 | 验收标准 |
| --- | --- | --- | --- | --- |
| 1 | RMSNorm 可选后端 | 已完成 | normalization 与内存访问 | LayerNorm/RMSNorm 可训练、回放并进入三 seed benchmark |
| 2 | SwiGLU 可选 MLP | 已完成 | gated MLP、参数量与 FLOPs | Dense/SwiGLU/GEGLU/ReGLU 有等预算 loss 和速度记录 |
| 3 | GQA 可选 attention | 已完成 | Q heads 与 KV heads 分离 | native CPU 全模式、nano-vLLM/vLLM loader 与构造单测通过，KV cache 按 Hkv/H 缩小，旧 checkpoint 兼容；GPU engine 实测待补 |
| 4 | 训练基础设施 | 未开始 | resume、LR schedule、AMP、grad accumulation | 中断后可复现续训，记录 val loss/ppl |
| 5 | SFT + response loss mask | 未开始 | 预训练与指令微调的目标差异 | prompt token 不计 loss，response 可稳定生成 |
| 6 | LoRA | 未开始 | 参数高效微调 | adapter 保存/合并，和 full fine-tune 对比 |
| 7 | 真正 HF PreTrainedModel | 未开始 | 标准生态契约 | `save_pretrained/from_pretrained` 往返一致 |
| 8 | SGLang native backend | 未开始 | RadixAttention/prefix cache | engine 能加载同一 export 并生成一致 token |
| 9 | 量化 | 未开始 | FP16/INT8/INT4 权重与 kernel | 报告质量、显存、首 token 延迟、吞吐 |
| 10 | Triton/CUDA 算子 | 未开始 | kernel 与端到端收益的关系 | microbenchmark 和 engine benchmark 都有数据 |

## 8. 近期优先级

结构模块和 CPU 公平基准已经建立，下一批工作按以下顺序推进：

1. 训练 RoPE + Byte-BPE + RMSNorm + SwiGLU 的 MHA/GQA checkpoint，在 native、nano-vLLM、vLLM 三条路径比较 logits、KV cache 和 greedy token IDs。
2. GQA/MQA 配置、紧凑投影/cache、HF-like export、attention benchmark 与三条 loader 路径已经实现；GPU engine 实测随正式 checkpoint 一起补齐。
3. 下一代码迭代是 checkpoint resume、cosine LR、warmup、AMP 和 gradient accumulation。
4. 把 benchmark 扩展到更长预算、多个数据 seed、GPU 峰值显存及 engine 吞吐。
5. 再开始 SFT/loss mask。模型是否“更会回答问题”主要由数据和训练目标决定，不是只靠结构开关。

性能评测至少记录：模型配置、dtype、prompt/output token 数、batch/concurrency、TTFT、ITL、tokens/s、峰值显存和输出 token IDs。只有 token IDs 或质量可接受时，速度对比才有意义。
