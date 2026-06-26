# RMSNorm 论文分析与本地实验记录

生成日期: 2026-06-26

论文: `Root Mean Square Layer Normalization`, Biao Zhang and Rico Sennrich, NeurIPS 2019.

本地 PDF:

- `/home/undefined/Desktop/ai/llm-systems-learning-pack/papers/01_transformer_norm_attention/2019NeurIPS-Root mean square layer normalization.pdf`

外部来源:

- 论文页面: https://papers.nips.cc/paper_files/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html
- 作者代码: https://github.com/bzhangGo/rmsnorm
- PyTorch normalization docs: https://docs.pytorch.org/docs/stable/nn.html#normalization-layers
- PyTorch RMSNorm docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html

## 1. 论文核心问题

LayerNorm 的常见形式是:

```text
LN(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

它同时做两件事:

1. re-centering: 减去均值。
2. re-scaling: 除以标准差，稳定向量尺度。

RMSNorm 的核心假设是: LayerNorm 成功的主要原因不是 re-centering，而是 re-scaling。也就是说，训练稳定性主要来自激活尺度被控制住，而不是来自每层激活均值被强制归零。

RMSNorm 因此删除均值计算，只保留 root mean square:

```text
RMS(x) = sqrt(mean(x^2))
RMSNorm(x) = x / RMS(x) * gamma
```

作者还提出 pRMSNorm，用部分维度估计 RMS:

```text
pRMSNorm(x) = x / sqrt(mean(x_first_p_percent^2)) * gamma
```

## 2. 主要贡献

### 2.1 把 LayerNorm 的有效性拆成两个不变性

论文不是简单提出一个更短公式，而是把 LayerNorm 的性质拆成:

- re-centering invariance: 输入或权重整体平移后输出不变。
- re-scaling invariance: 输入或权重整体缩放后输出不变。

作者认为后者更关键。RMSNorm 保留 re-scaling invariance，放弃 re-centering invariance。

这一点很重要，因为它把 Norm 的讨论从“公式替换”推进到“哪些不变性真正影响训练稳定性”。

### 2.2 提出 RMSNorm 作为 LayerNorm 的 drop-in replacement

RMSNorm 不需要计算均值和方差，只需要平方、均值、开方、缩放。相对 LayerNorm，它减少了一次均值中心化和方差计算相关开销。

它适合替换 Transformer、RNN、MLP 里的 LayerNorm。现代 LLM 里常见的实现通常类似:

```text
y = x * rsqrt(mean(x^2, dim=-1) + eps) * weight
```

### 2.3 提出 pRMSNorm

pRMSNorm 用少量维度估计 RMS。论文实验常用 p = 6.25%，CIFAR-10 中因模型较小用 p = 12.5%。

理论上 pRMSNorm 计算更少，但论文也承认实际速度不一定总更快，因为切片和框架实现可能引入额外开销。

### 2.4 覆盖多任务实验

论文实验横跨:

- WMT14 En-De 机器翻译: RNNSearch, Transformer。
- CNN/Daily Mail 阅读理解: attentive reader。
- MS COCO 图文检索: order-embedding。
- CIFAR-10 分类: ConvPool-CNN-C。

结论是: RMSNorm 质量基本接近 LayerNorm，训练时间相对 LayerNorm 加速 7% 到 64%，具体依赖模型、框架、硬件和实现。

### 2.5 现代 LLM 的后续影响

RMSNorm 后来成为 LLaMA 系列等 decoder-only LLM 的典型选择。本地 LLaMA PDF 中也能看到:

- LLaMA 1: 使用 Pre-normalization，并采用 RMSNorm。
- LLaMA 2: 继承 LLaMA 1 架构，使用 RMSNorm、SwiGLU、RoPE。

因此这篇论文的实际工程影响很大: 它把 Transformer block 中高频出现的归一化算子变得更便宜，也更利于 fused kernel 优化。

## 3. RMSNorm 与 LayerNorm 的关键差异

| 项 | LayerNorm | RMSNorm |
|---|---|---|
| 是否减均值 | 是 | 否 |
| 统计量 | mean 和 variance | mean(x^2) 的平方根 |
| affine 参数 | gamma, beta | 通常只有 gamma，beta 可选 |
| 缩放不变性 | 有 | 有 |
| 平移不变性 | 有 | 无 |
| 零均值输入时 | 基准 | 近似等价于 LayerNorm |
| 计算开销 | 更高 | 更低 |
| LLM 应用 | BERT/GPT-2/GPT-3/OPT 等常见 | LLaMA/Mistral/Qwen/Gemma 类模型常见 |

注意: RMSNorm 删除 beta 是自然选择，因为它不追求 re-centering invariance。实际实现里也有带 bias 的变体，但主流 LLM 通常只保留 weight。

## 4. 论文实验结果概要

### 4.1 RNNSearch, TensorFlow Nematus

WMT14 En-De:

| Model | Test14 | Test17 | Time per 1k steps |
|---|---:|---:|---:|
| Baseline | 21.7 | 23.4 | 399s |
| LayerNorm | 22.6 | 23.6 | 665s |
| RMSNorm | 22.4 | 23.7 | 501s, 24.7% faster than LN |
| pRMSNorm | 22.6 | 23.1 | 493s, 25.9% faster than LN |

含义:

- LayerNorm/RMSNorm 都显著提升收敛速度和 BLEU。
- LayerNorm 在 TensorFlow RNN 中开销很大，比 baseline 慢很多。
- RMSNorm 质量接近 LayerNorm，但训练时间明显更低。

### 4.2 RNNSearch, Theano/PyTorch

Theano 版 RMSNorm 比 LayerNorm 快约 34%。PyTorch 版 RMSNorm 比 LayerNorm 快约 11%。

论文同时观察到 pRMSNorm 不总是比 RMSNorm 快，原因可能是框架切片开销。

### 4.3 Transformer

WMT14 En-De Transformer base:

| Model | Test14 | Test17 | Time per 1k steps |
|---|---:|---:|---:|
| Baseline without norm | failed | failed | 210s |
| LayerNorm | 26.6 | 27.7 | 248s |
| RMSNorm | 26.8 | 27.7 | 231s, 6.9% faster |
| pRMSNorm | 26.5 | 27.8 | 225s, 9.3% faster |

含义:

- Transformer 里 Norm 是必要组件，不加 Norm 训练失败。
- RMSNorm 在 BLEU 上基本不损失，速度更好。
- Transformer 的 Norm 开销占比低于 RNN，所以加速百分比低于 RNNSearch。

### 4.4 CNN/Daily Mail 阅读理解

Attentive reader:

| Model | Time per 0.1k steps |
|---|---:|
| Baseline | 315s |
| BatchNorm-Everywhere | 348s |
| BatchNorm-LSTM | 345s |
| LayerNorm | 392s |
| RMSNorm | 333s, 15.1% faster than LN |
| pRMSNorm | 330s, 15.8% faster than LN |

RMSNorm 与 LayerNorm 验证误差接近，但更快。

### 4.5 MS COCO 图文检索

Order-embedding:

| Model | Time per 0.1k steps |
|---|---:|
| Baseline | 2.11s |
| LayerNorm | 12.02s |
| RMSNorm | 7.12s, 40.8% faster |
| pRMSNorm | 4.34s, 63.9% faster |

该任务中 LayerNorm 开销非常明显，pRMSNorm 加速最大。

### 4.6 CIFAR-10

ConvPool-CNN-C:

| Model | Test Error | Time per epoch |
|---|---:|---:|
| Baseline | 8.96% | 21s |
| BatchNorm | 8.25% | 38s |
| WeightNorm | 8.28% | 23s |
| LayerNorm | 10.49% | 39s |
| RMSNorm | 8.83% | 31s |
| pRMSNorm | 10.37% | 30s |

含义:

- 图像 CNN 中 BatchNorm/WeightNorm 更强。
- LayerNorm 对 CNN 泛化不佳。
- RMSNorm 比 LayerNorm 更好且更快，但不是 CIFAR-10 最优选择。

## 5. 本地代码克隆结果

已克隆作者代码:

| 路径 | 说明 | commit |
|---|---|---|
| `/home/undefined/Desktop/ai/llm-systems-learning-pack/projects/rmsnorm` | 作者 RMSNorm 实现 | `2e726f1` |

已克隆论文 README 指向的实验相关仓库:

| 路径 | 分支/标签 | commit | 对应实验 |
|---|---|---|---|
| `projects/rmsnorm_experiments/nematus` | `tags/v0.3` detached | `813cc8b` | TensorFlow Nematus RNNSearch |
| `projects/rmsnorm_experiments/nematus-theano` | `theano` | `5727727` | Theano Nematus RNNSearch |
| `projects/rmsnorm_experiments/Attentive_reader` | `bn` | `7c1ea8a` | CNN/Daily Mail |
| `projects/rmsnorm_experiments/order-embedding` | `master` | `2057e60` | MS COCO 图文检索 |
| `projects/rmsnorm_experiments/weight_norm` | `master` | `e625840` | CIFAR-10 |
| `projects/rmsnorm_experiments/zero` | `master` | `d97e2c2` | 作者 Transformer/NMT 代码 |

本地运行日志:

- `/home/undefined/Desktop/ai/llm-systems-learning-pack/analysis_outputs/rmsnorm_smoke_benchmark.log`
- `/home/undefined/Desktop/ai/llm-systems-learning-pack/analysis_outputs/rmsnorm_experiment_env_checks.log`

## 6. 本地运行了什么

当前机器环境:

```text
Python 3.12.3
python2: not found
GPU: 未使用
```

我创建了隔离环境:

```text
/home/undefined/Desktop/ai/llm-systems-learning-pack/projects/rmsnorm/.venv
```

安装:

```text
torch 2.12.1+cpu
numpy 2.5.0
```

运行作者 `rmsnorm_torch.py` 的结果:

```text
formula_max_abs_diff 0.0
zero_mean_ln_rms_max_abs_diff 4.440892098500626e-16
rms_rescale_diff 3.5762786865234375e-07
prms_rescale_diff 7.152557373046875e-07
ln_rescale_diff 4.76837158203125e-07
rms_recenter_shift_diff 0.9177456498146057
ln_recenter_shift_diff 1.1639031072263606e-07
```

解释:

- 作者实现与 RMSNorm 公式完全一致。
- 输入逐样本零均值时，RMSNorm 与 LayerNorm 数值等价。
- RMSNorm/pRMSNorm/LayerNorm 都对缩放基本不变。
- RMSNorm 对平移明显敏感，LayerNorm 对平移基本不变。这验证了论文的不变性分析。

CPU 前向+反向 micro-benchmark:

```text
shape (128, 512):  LayerNorm 0.180 ms, RMSNorm 0.422 ms, pRMSNorm25 0.343 ms
shape (128, 1024): LayerNorm 0.345 ms, RMSNorm 0.809 ms, pRMSNorm25 0.655 ms
shape (128, 4096): LayerNorm 1.363 ms, RMSNorm 3.210 ms, pRMSNorm25 2.662 ms
```

这个 micro-benchmark 不是论文速度复现。它只说明作者的纯 PyTorch Python 实现在当前 CPU 上能跑通。PyTorch 内置 LayerNorm 在 CPU 上高度优化，所以这里 RMSNorm 反而慢。论文的速度结论来自 2019 年老框架、GPU、完整模型训练路径，不能用这个 CPU micro-benchmark 直接推翻或复现。

## 7. 完整论文实验为什么没有直接跑起来

论文实验依赖非常老:

- 作者 README 明确要求 Python 2.7。
- TensorFlow Nematus/Zero 依赖 TensorFlow 1.x。
- Theano Nematus、Attentive Reader、Order Embedding、WeightNorm 依赖 Theano/Lasagne 或 Python2 语法。
- WMT14、MS COCO、CNN/Daily Mail、CIFAR-10 完整训练还需要数据准备和 GPU。

本机入口检查结果:

```text
python2: not found

nematus v0.3:
SyntaxError: Lambda expression parameters cannot be parenthesized

nematus-theano:
SyntaxError: Missing parentheses in call to 'print'

Attentive_reader:
SyntaxError: Missing parentheses in call to 'print'

order-embedding:
SyntaxError: Missing parentheses in call to 'print'

weight_norm:
ModuleNotFoundError: No module named 'cPickle'

zero:
ModuleNotFoundError: No module named 'tensorflow'
```

所以当前环境完成的是:

- 克隆作者仓库和实验依赖仓库。
- 跑通作者 PyTorch RMSNorm 实现的数值验证。
- 记录完整论文训练的环境阻塞点。

若要完整复现论文训练，建议使用独立 Docker/conda 环境，而不是污染当前 Python 3.12:

```text
python 2.7
tensorflow <= 1.13.2
theano 0.8.x/1.0.x
lasagne
numpy/scipy compatible with Python2
CUDA/cuDNN matching old TF/Theano
```

其中 TensorFlow 1.x 和 Python2 对现代系统支持很差，实际更稳的办法是找旧 CUDA 镜像或单独容器。

## 8. 相关 Norm 体系

### 8.1 激活归一化

这些 Norm 直接对激活张量做统计归一化。

| Norm | 统计维度 | 常见用途 | 重要性 |
|---|---|---|---|
| BatchNorm | batch 维和空间维 | CNN 分类、视觉 backbone | 非常重要，CNN 经典默认 |
| SyncBatchNorm | 多卡同步 batch 统计 | 分布式 CNN 训练 | 重要，视觉多卡常用 |
| LayerNorm | 单样本 hidden/channel 维 | Transformer, RNN, NLP | 非常重要，Transformer 经典默认 |
| RMSNorm | 单样本 hidden 维 RMS | 现代 decoder-only LLM | 非常重要，LLM 高频 |
| pRMSNorm | hidden 子集估计 RMS | 研究型，少量工程使用 | 概念重要，应用少 |
| InstanceNorm | 单样本单通道空间维 | 风格迁移、生成模型 | 中等重要 |
| GroupNorm | 单样本 channel 分组 | 小 batch 视觉、检测、分割、扩散 U-Net | 重要 |
| LocalResponseNorm | 局部响应竞争 | 早期 AlexNet 类 CNN | 历史重要，现在少 |

### 8.2 权重/参数归一化

这些 Norm 不直接归一化激活，而是重参数化或约束权重。

| Norm | 思路 | 常见用途 | 重要性 |
|---|---|---|---|
| WeightNorm | 把权重拆成方向和长度 | 早期 RNN/CNN、生成模型 | 历史重要，现在较少作为默认 |
| SpectralNorm | 限制权重谱范数 | GAN 判别器、Lipschitz 约束 | 特定领域重要 |
| Weight Standardization | 对卷积权重标准化 | 与 GroupNorm 配合的视觉模型 | 中等重要 |

### 8.3 Transformer/LLM 结构相关 Norm

| 名称 | 含义 | 常见程度 |
|---|---|---|
| Post-LN | `sublayer -> residual -> norm` | 原始 Transformer 常见，深层训练更难 |
| Pre-LN | `norm -> sublayer -> residual` | GPT/LLaMA 等深层 Transformer 常见 |
| Sandwich-LN | 子层前后都加 Norm | 研究/特定模型使用 |
| DeepNorm | 残差缩放和初始化配合 Norm | 深层 encoder-decoder Transformer 研究常见 |
| NormFormer | 在 attention/FFN 内额外加 Norm | 研究常见，非主流默认 |
| QK-Norm | 对 query/key 做 per-head Norm | 长上下文/稳定 attention 的现代变体，正在增多 |

## 9. 哪些 Norm 最重要

按深度学习通用重要性排序:

1. BatchNorm: CNN 时代最重要的训练稳定化技术之一，仍然是大量视觉分类/检测模型的基础。
2. LayerNorm: Transformer 时代的核心组件，BERT、GPT 系列早期模型、ViT、很多多模态模型都依赖它。
3. RMSNorm: 现代 LLM 中最关键的 Norm 之一，LLaMA 系列把它推成 decoder-only Transformer 的常见默认。
4. GroupNorm: 小 batch 或 batch 统计不稳定时非常有用，扩散模型和检测/分割里常见。
5. WeightNorm/SpectralNorm: 更偏特定场景，历史和理论上重要，但不是通用 Transformer 默认。

## 10. 应用最多的 Norm

按领域看:

| 领域 | 最常见 Norm |
|---|---|
| CNN 图像分类 | BatchNorm, SyncBatchNorm |
| 小 batch 视觉/检测/分割 | GroupNorm, LayerNorm |
| Diffusion U-Net | GroupNorm 很常见，Transformer block 中用 LayerNorm/RMSNorm |
| NLP Transformer encoder | LayerNorm |
| Decoder-only LLM | RMSNorm 或 LayerNorm，现代开源 LLM 更偏 RMSNorm |
| RNN/序列老模型 | LayerNorm/RMSNorm，可减少 batch 依赖 |
| GAN | BatchNorm, SpectralNorm, InstanceNorm, AdaIN |
| 风格迁移 | InstanceNorm, AdaIN |
| 推理系统/LLM serving | RMSNorm 很重要，因为算子频繁且易融合 |

## 11. 对 LLM 系统学习的实际结论

1. 对现代 LLM 来说，最需要掌握的是 LayerNorm、RMSNorm、Pre-LN、QK-Norm。
2. RMSNorm 的价值不仅是省一点 FLOPs，而是省掉均值中心化后更容易做高效 fused kernel，例如 residual-add + RMSNorm、RMSNorm + quant、QK RMSNorm + RoPE。
3. BatchNorm 在 LLM 主干里不常用，因为它依赖 batch 统计，训练/推理行为不一致，也不适合变长 autoregressive decoding。
4. GroupNorm/InstanceNorm 更偏视觉和生成模型，不是 LLM 主干重点。
5. pRMSNorm 是理解 RMSNorm 统计性质的好实验，但现在主流大模型实现更常用完整 RMSNorm，因为 kernel 优化后完整 RMS 的代价已经可控。

## 12. 建议继续做的复现实验

如果目标是学习而不是完整复现论文表格，推荐下一步做两个现代环境实验:

1. 用 PyTorch 写一个小 Transformer block，对比 LayerNorm/RMSNorm 在相同初始化、相同 batch 下的 loss 曲线和速度。
2. 用 Triton 或 CUDA 写一个 RMSNorm kernel，对比 naive PyTorch、torch.compile、FlashAttention/vLLM/SGLang fused RMSNorm 的吞吐和内存带宽。

如果目标是严格复现论文表格，建议单独建立旧环境容器:

```text
Ubuntu 16.04/18.04
Python 2.7
TensorFlow 1.13.2
Theano 0.8.2 或 1.0
Lasagne
CUDA 9/10 级别环境
```

然后按作者 README 对 Nematus、Attentive Reader、Order Embedding、WeightNorm 打 RMSNorm patch，并下载作者提供的数据/脚本或对应公开数据集。

## 13. 当前环境兼容代码修改

按后续要求，我已经把作者 PyTorch 代码改成当前 Python 3.12 + PyTorch 可直接验证的版本。

修改文件:

- `/home/undefined/Desktop/ai/llm-systems-learning-pack/projects/rmsnorm/rmsnorm_torch.py`
- `/home/undefined/Desktop/ai/llm-systems-learning-pack/projects/rmsnorm/experiments/verify_rmsnorm_paper.py`
- `/home/undefined/Desktop/ai/llm-systems-learning-pack/projects/rmsnorm/README_CURRENT_ENV.md`
- `/home/undefined/Desktop/ai/llm-systems-learning-pack/projects/rmsnorm/.gitignore`

核心修改:

```text
old PyTorch author code:
    x / (norm(x) / sqrt(d) + eps)

modern paper-style code:
    x * rsqrt(mean(x^2) + eps)
```

这让 PyTorch 实现与论文公式和作者 TensorFlow 片段更一致。新增脚本不依赖 Python2、TensorFlow1、Theano 或 Lasagne。

新增验证命令:

```bash
/home/undefined/Desktop/ai/llm-systems-learning-pack/projects/rmsnorm/.venv/bin/python \
  /home/undefined/Desktop/ai/llm-systems-learning-pack/projects/rmsnorm/experiments/verify_rmsnorm_paper.py \
  --steps 160 --depth 24 --hidden-size 128 --threads 1
```

输出保存:

- `/home/undefined/Desktop/ai/llm-systems-learning-pack/analysis_outputs/rmsnorm_modern_verify.log`
- `/home/undefined/Desktop/ai/llm-systems-learning-pack/analysis_outputs/rmsnorm_modern_verify_deep.log`

深层设置结果:

```text
invariance_check:
layernorm_rescale_max_abs 4.77e-07
rmsnorm_rescale_max_abs   7.15e-07
prmsnorm_rescale_max_abs  4.77e-07
layernorm_shift_mean_abs  1.27e-07
rmsnorm_shift_mean_abs    1.0003

training_check:
none       final_loss 0.2922, final_acc 0.9033
layernorm  final_loss 0.2441, final_acc 0.9165
rmsnorm    final_loss 0.2649, final_acc 0.9097
prmsnorm   final_loss 0.2346, final_acc 0.9194
```

验证结论:

- LayerNorm、RMSNorm、pRMSNorm 都保持输入缩放不变性。
- RMSNorm 明确不保持平移不变性，这符合论文理论。
- 在当前 PyTorch 小型深层残差 MLP 任务中，RMSNorm 与 LayerNorm 同级，pRMSNorm 也能正常收敛。
- 这验证的是论文核心思想: 去掉 mean-centering 后，保留 RMS 缩放仍然足以稳定训练；不是严格复现原论文四套大型训练实验。
