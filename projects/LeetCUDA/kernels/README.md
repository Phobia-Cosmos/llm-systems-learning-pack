# LeetCUDA Triton 与 torch.compile 后端

本目录现在提供两套可直接导入的统一实现：`kernels/triton_backend` 使用显式 Triton kernel，`kernels/torch_compile_backend` 使用纯 PyTorch 表达式并交给 `torch.compile(..., backend="inductor")`。两套后端按“独立数学语义”覆盖原目录中的 CUDA/CUTLASS/CuTe/Triton 示例；原仓库里标量、`x2/x4/x8`、pack、shared-memory、double-buffer、WMMA/MMA/WGMMA、swizzle、split-Q/KV 等同义优化变体不会机械复制成多个 Python API，而是映射到同一个算子，由 Triton 编译器、autotune 或 TorchInductor 负责具体代码生成。

## 快速开始

从仓库根目录运行：

```bash
PYTHONPATH=. /home/undefined/Disk/python-envs/brainuicl/bin/python kernels/validate_backends.py
PYTHONPATH=. /home/undefined/Disk/python-envs/brainuicl/bin/python kernels/benchmark_backends.py --operator all
```

调用方式：

```python
import torch
import kernels.triton_backend as triton_ops
import kernels.torch_compile_backend as compiled_ops

x = torch.randn(4096, 1024, device="cuda", dtype=torch.float16)
y_triton = triton_ops.rms_norm(x)
y_compiled = compiled_ops.rms_norm(x)
```

`torch.compile` 的第一次调用包含编译过程，计时时必须先 warmup。`benchmark_backends.py` 已经把编译放在预热阶段，不把首次编译时间计入稳态 kernel 时间。

## 覆盖范围

| 原目录或文件 | 统一语义 | Triton / torch.compile API |
| --- | --- | --- |
| `dot-product/`、`interview/notes-v2.cu` 的 dot 示例 | 点积 | `dot_product` |
| `elementwise/`、`cutlass/cute/vector_add.cu`、`openai-triton/vector-add/`、`nvidia-nsight/elementwise.cu` | 向量逐元素加法 | `elementwise_add` |
| `relu/`、`nvidia-nsight/relu.cu`、`interview/notes-v2.cu` 的 ReLU 示例 | ReLU | `relu` |
| `sigmoid/` | Sigmoid | `sigmoid` |
| `elu/` | ELU | `elu` |
| `gelu/` | tanh 近似 GELU | `gelu` |
| `swish/` | Swish / SiLU | `swish` |
| `hardswish/` | HardSwish | `hardswish` |
| `hardshrink/` | HardShrink | `hardshrink` |
| `embedding/` | Embedding gather | `embedding` |
| `reduce/`、`interview/notes-v2.cu` 的 block reduce 示例 | 全局求和归约 | `reduce_sum` |
| `softmax/`、`openai-triton/fused-softmax/`、`interview/notes-v2.cu` 的 softmax 示例 | 行方向安全 Softmax | `softmax` |
| `layer-norm/`、`openai-triton/layer-norm/`、`interview/notes-v2.cu` 的 LayerNorm 示例 | scalar affine LayerNorm、向量 affine forward/backward | `layer_norm`、`layer_norm_affine`、`layer_norm_backward` |
| `rms-norm/`、`interview/notes-v2.cu` 的 RMSNorm 示例 | RMSNorm | `rms_norm` |
| `rope/`、`interview/notes-v2.cu` 的 RoPE 示例 | 旋转位置编码 | `rope` |
| `histogram/`、`interview/notes-v2.cu` 的 histogram 示例 | int32 直方图 | `histogram` |
| `mat-transpose/`、`swizzle/mat_trans_swizzle.cu`、CuTe transpose 示例 | 物理矩阵转置 | `matrix_transpose` |
| `sgemv/`、`hgemv/`、`interview/notes-v2.cu` 的 GEMV 示例 | GEMV | `gemv` |
| `sgemm/`、`hgemm/`、`ws-hgemm/`、`cutlass/cute_dsl/sgemm/`、`swizzle/*gemm*`、`interview/notes-v2.cu` 的 GEMM 示例 | GEMM | `gemm` |
| `flash-attn/`、`interview/notes-v2.cu` 的 attention 示例 | 前向 FlashAttention | `flash_attention` |
| `openai-triton/merge-attn-states/`、`interview/notes-v2.cu` 的 merge 示例 | Split-KV attention state merge | `merge_attention_states` |
| `nms/` | Non-Maximum Suppression | `nms` |

`nvidia-nsight/` 和 `swizzle/` 中的文件主要用于观察 SASS、bank conflict 或地址置换，不代表新的数学算子；它们分别归入逐元素、转置和 GEMM 家族。`interview/notes-v2.cu` 是教学汇总文件，其中可执行语义也都落入上表已有家族。`flash-attn/` 和 `hgemm/` 下的大量文件是同一语义的不同调度、精度或架构实验，而不是几十个不同的模型算子。

## 为什么需要每个 kernel，以及 naive 的差距

这里的 “naive” 指算法层面的直接 CUDA 实现，例如一个线程串行处理一个输出、每次都从全局内存读取、用单一 atomic 聚合或显式生成中间张量。它不等于表格里的 PyTorch eager：`torch.matmul`、`torch.softmax`、`torchvision.ops.nms` 本身已经是高度优化的生产实现，因此 eager 有时会快于教学 Triton kernel。

### 逐元素与激活

| Kernel | 为什么需要 | naive 的主要问题 | Triton 实现 | PyTorch + torch.compile 实现 |
| --- | --- | --- | --- | --- |
| `elementwise_add` | 残差连接、bias、并行分支合并都需要逐元素加法。 | 标量线程仍可达到较高带宽，但手写 `float4/half2/int4` 才能显式扩大访存粒度，尾部处理也容易出错。 | 一维 program + mask；Triton 根据连续地址自动合并访存，不需要暴露 `f32x4/f16x8_pack` API。 | `a + b`；单独调用通常不会优于原生 `torch.add`，价值在于与前后算子跨算子融合。 |
| `relu` | 引入稀疏非线性，广泛用于 CNN/MLP。 | 算术只有一次 `max`，完全受内存带宽和 launch 延迟限制，手工 pack 只改善访存。 | 单次 load、`maximum`、store。 | `torch.relu(x)`；单算子已经很优化，compile 的优势通常只在更大融合图中出现。 |
| `sigmoid` | 门控、概率映射和 logistic 输出需要。 | 每元素需要指数函数，naive 标量实现吞吐低；多 kernel 表达式还会产生临时张量。 | fp32 内部计算 `1 / (1 + exp(-x))` 后按输入 dtype 写回。 | `torch.sigmoid(x)`，由 Inductor 生成 pointwise kernel。 |
| `elu` | 负半轴保持平滑梯度并降低均值偏移。 | 分支与 `exp` 增加算术成本，若拆成 `where`、`exp`、乘加会多次读写显存。 | 一个 kernel 内完成比较、指数和缩放。 | `F.elu(x)`，Inductor 可与邻接逐元素表达式融合。 |
| `gelu` | Transformer MLP 的常用激活。 | 直接 erf 或 tanh 近似包含多次乘加和超越函数，拆分 eager 表达式会产生多个中间结果。 | 使用 tanh 等价的 logistic 形式实现常见近似，整个表达式融合到一次访存。 | `F.gelu(x, approximate="tanh")`。两套后端使用相同近似定义。 |
| `swish` | 现代 CNN/LLM 中常见的平滑门控激活。 | naive 的 `sigmoid(x)` 再乘 `x` 至少需要中间张量或两次 kernel。 | 同一 program 中计算 sigmoid 和乘法，一次读写。 | `F.silu(x)`；Inductor 可生成融合 kernel。 |
| `hardswish` | 移动端模型用分段线性近似替代昂贵 sigmoid。 | 分成 add、clamp、mul、div 会有多次 launch 和内存往返。 | 单 kernel 完成 `x * clamp(x + 3, 0, 6) / 6`。 | `F.hardswish(x)`。 |
| `hardshrink` | 稀疏化与阈值收缩需要把阈值区间内的元素置零。 | naive 分支简单但仍是带宽受限；多个比较与 `where` 若未融合会多次遍历。 | 两个比较与一次 `where` 融合。 | `F.hardshrink(x, lambd)`。 |

### 归约、归一化与索引

| Kernel | 为什么需要 | naive 的主要问题 | Triton 实现 | PyTorch + torch.compile 实现 |
| --- | --- | --- | --- | --- |
| `dot_product` | 相似度、投影和许多归约公式的基本构件。 | 单线程串行是 $O(N)$ 且无并行；每线程 atomic 到同一标量会形成严重竞争。 | 每个 program 先做局部乘加与树归约，再用多阶段 Triton 归约合并 partial，fp16/bf16 输入使用 fp32 累加。 | `(a.float() * b.float()).sum()`；Inductor 融合乘法与第一阶段归约。 |
| `reduce_sum` | loss、统计量、归一化和 collective 前处理都依赖求和。 | 全局 atomic 热点、低占用或单 block 尺寸上限都会限制扩展性。 | 每个 program 处理 4096 个元素，产生 partial 后递归归约；支持原目录的 fp32/fp16/bf16/FP8 E4M3/FP8 E5M2/int8，浮点统一 fp32 累加，int8 输出 int32。 | `x.float().sum()` 或 `torch.sum(x, dtype=torch.int32)`。 |
| `softmax` | 将 logits 转为归一化权重，是分类与注意力核心。 | 不安全版本直接 `exp(x)` 会溢出；三遍 max/sum/div 会重复访问全局内存；生成多个中间张量代价高。 | 一行一个 program，在寄存器中完成 max、指数、sum 和归一化，等价于 safe fused softmax；这吸收了原目录的 scalar、x4、online 变体。 | `torch.softmax(x, dim=-1)`；Inductor 或 PyTorch 专用 kernel 负责稳定实现。 |
| `layer_norm` / `layer_norm_affine` / `layer_norm_backward` | 消除每个 token 特征尺度漂移，是 Transformer 的标准组件；训练还需要对输入、gamma、beta 求梯度。 | naive forward 先 mean、再 variance、再 affine，backward 再拆成多个投影与列归约，产生多次遍历；fp16 直接累加误差较大。 | scalar 版本匹配原 CUDA API；向量 affine forward 保存 mean/rstd，自定义 autograd backward 用一个行 kernel 计算 dx、一个列 kernel 计算 dweight/dbias，内部均为 fp32 累加。 | `F.layer_norm` 提供可微 forward；另有显式解析 backward 的纯 PyTorch 表达式交给 Inductor。注意原 `layer_norm.py` 的 `torch.std` 默认 correction 与 CUDA 的除以 $K$ 不一致，新实现以 CUDA/标准 LayerNorm 语义为准。 |
| `rms_norm` | LLaMA 等模型用它减少 LayerNorm 的均值计算并保持尺度稳定。 | 两遍读取完成平方均值和缩放；拆成 `pow/mean/rsqrt/mul` 会创建临时张量。 | 一行一个 program，fp32 累加平方和并在一次 kernel 中缩放。 | `F.rms_norm(...)*scale`，Inductor 融合尾部缩放。 |
| `embedding` | token id 到隐藏向量的随机 gather 是语言模型输入的核心。 | 每个标量独立线程会产生细碎访问；手写 pack 需要对 embedding dim 对齐。 | 每个索引一个 program，连续读取整行，mask 处理非 2 的幂维度。 | `F.embedding(indices, weight)`；通常 PyTorch 原生 gather 已很强，单独 compile 不一定更快。 |
| `histogram` | 频次统计、量化校准和数据分布分析需要。 | 所有线程对少量 bin 做全局 atomic，分布集中时竞争极重。 | 当前版本使用全局 `atomic_add`，语义完整但仍受热点竞争；传 `num_bins` 可避免为推导输出大小而同步。 | `zeros + scatter_add`，Inductor 也会生成 atomic；生产场景的 `torch.bincount` 通常有更成熟的分层策略。 |

### 布局与线性代数

| Kernel | 为什么需要 | naive 的主要问题 | Triton 实现 | PyTorch + torch.compile 实现 |
| --- | --- | --- | --- | --- |
| `matrix_transpose` | GEMM 布局转换、attention K 转置和算子接口适配需要物理转置。 | 一种方向合并读取就会导致另一方向跨步写入；naive 直接映射无法同时合并读写。 | 32×32 tile 读入寄存器后转置并交换 program 维度写回；等价吸收 shared-memory、padding、swizzle 与 diagonal block-order 实验。 | `x.transpose(0, 1).contiguous()`，Inductor 生成转置 copy kernel。 |
| `gemv` | 解码阶段 batch/token 很小时，矩阵乘向量比 GEMM 更常见。 | 一个线程串行遍历 $K$，并行度不足；通用 GEMM 对 $N=1$ 可能有调度开销。 | 每行一个 program，向量化 load 后在 program 内归约，针对 $N=1$。 | `torch.matmul(matrix, vector)`，通常调度 cuBLAS GEMV/GEMM 路径。 |
| `gemm` | 线性层、MLP、投影和 attention 的主要计算量都在矩阵乘。 | naive 每个输出线程重复从全局内存加载 A/B，算术强度低；没有 tile、共享复用、流水或 Tensor Core。 | block tiling + `tl.dot` + autotune，fp16/bf16 使用 Tensor Core 并以 fp32 accumulator 累加；原目录的 vectorized、shared-memory、double-buffer、WMMA/MMA/WGMMA、CuTe、swizzle、warp-specialization 版本都属于这一优化阶梯。 | `torch.matmul`；Inductor 通常不是自己生成最底层 GEMM，而是选择 cuBLAS/cuBLASLt/CUTLASS 等库并融合可融合的前后处理。 |
| `rope` | 把相对位置信息编码进 Q/K，避免显式位置 embedding 相加。 | eager 的 reshape、频率生成、sin/cos、两个旋转分量和 stack 会产生多个 kernel 与临时张量。 | 每个 token 一个 program，同时生成频率并旋转偶/奇维，一次读写。 | 纯 PyTorch `arange/exp/sin/cos/stack` 表达式由 Inductor 融合。 |

### 注意力与检测

| Kernel | 为什么需要 | naive 的主要问题 | Triton 实现 | PyTorch + torch.compile 实现 |
| --- | --- | --- | --- | --- |
| `flash_attention` | 注意力是 Transformer 的核心，长序列下中间 score 矩阵主导显存。 | naive 先计算并保存 $QK^T$，再 softmax，再乘 V，显存复杂度为 $O(N^2)$，并多次读写巨大的 score 矩阵。 | 按 Q/KV tile 计算，使用 online softmax 维护 running max、running sum 和输出 accumulator，不物化完整 score；支持 causal，head dim 为 16/32/64/128。 | `F.scaled_dot_product_attention`；Inductor/PyTorch 根据硬件和输入选择 Flash、memory-efficient 或 math backend。 |
| `merge_attention_states` | Split-KV、chunked prefill 和分段 attention 需要按各段 LSE 权重合并 partial output。 | eager 要单独计算 `logaddexp`、两个 scale 和两次乘加，产生多个中间张量。 | 每个 token/head 一个 program，稳定地以最大 LSE 为基准计算权重并合并整个 head。 | `logaddexp + exp + broadcast multiply/add`，由 Inductor 融合；`+inf` 按原实现转换为 `-inf`。 |
| `nms` | 目标检测需要按 score 保留代表框并删除高 IoU 重复框。 | 算法存在顺序依赖：只有“已保留”的高分框才能抑制后续框。原 `nms.cu` 在多个 block 间读取 `keep[i]`，但 CUDA kernel 内没有全局同步，结果依赖 block 调度。 | 当前正确性优先版本先稳定排序，再由主机按 score 顺序选择框，每次用 Triton 并行计算剩余框 IoU；语义正确，但每个保留框都有同步和 launch，性能明显不如生产 NMS。 | 编译 `torchvision.ops.nms`，并开启 dynamic-output-shape 捕获；核心仍是 torchvision 自定义 CUDA op，Inductor 负责图捕获而不是重写顺序算法。 |

## CUDA 优化变体如何映射到 Triton

| 原 CUDA 术语 | 作用 | Triton 中的对应方式 |
| --- | --- | --- |
| `f32x4`、`f16x2`、`f16x8_pack` | 128-bit 合并访存、减少指令数 | 连续 pointer tensor + 编译器向量化；mask 处理尾部，不公开 pack 专用 API。 |
| block/warp reduce | 用 shuffle/shared memory 做层级归约 | `tl.sum`、`tl.max` 在 program 内生成层级归约，多 program 时显式 partial pass。 |
| shared-memory tiling | 重用 A/B tile 或转置 tile | `tl.dot` 和 tile pointer 表达数据复用，编译器决定 shared memory/register placement。 |
| double buffer / async copy / multi-stage | 隐藏 global-to-shared 延迟 | `num_stages` 和软件流水；GEMM autotune 配置选择 stage 数。 |
| WMMA/MMA/WGMMA | 使用 Tensor Core | `tl.dot` 根据 dtype、tile 和目标架构降低到合适的矩阵指令。 |
| bank-conflict padding / swizzle | 改变 shared-memory 地址映射 | Triton 编译器 layout encoding 与 lowering 负责大部分布局；需要时可进一步使用 block pointer/layout API。 |
| thread-block / warp swizzle | 改变 tile 调度次序以改善 L2 与负载均衡 | grid 映射与 autotune；本版 GEMM 使用二维 tile 的线性 program 映射。 |
| split-Q / split-KV | 提高 attention 并行度或适应长序列 | `flash_attention` 采用 Q block × KV loop；分段 KV 的结果由 `merge_attention_states` 合并。 |
| fp16 accumulate / fp32 accumulate | 性能与数值精度权衡 | 点积、归约、归一化、attention 和 GEMM accumulator 默认 fp32，最终按输出 dtype 写回。 |

## 实测性能

以下数据于 2026-07-21 在 NVIDIA GeForce RTX 4070 SUPER、PyTorch 2.9.1+cu130、Triton 3.5.1 上运行 `benchmark_backends.py --operator all --warmup 3 --iterations 20` 得到；新增的 LayerNorm 训练项随后用相同参数单独补测。单位为毫秒；`T/eager` 和 `C/eager` 大于 1 表示 Triton 或 compile 更快。首次编译时间不计入。数字只代表表中脚本的固定形状，不能直接外推到其他 GPU、dtype 或尺寸。

| operator | eager/ms | Triton/ms | compile/ms | Triton speedup | compile speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| elementwise_add | 0.00978 | 0.01116 | 0.02463 | 0.88× | 0.40× |
| relu | 0.00814 | 0.01095 | 0.03036 | 0.74× | 0.27× |
| sigmoid | 0.00870 | 0.01162 | 0.02330 | 0.75× | 0.37× |
| elu | 0.01290 | 0.01032 | 0.02499 | 1.25× | 0.52× |
| gelu | 0.00875 | 0.01141 | 0.02285 | 0.77× | 0.38× |
| swish | 0.00833 | 0.01047 | 0.02258 | 0.80× | 0.37× |
| hardswish | 0.00745 | 0.01010 | 0.02307 | 0.74× | 0.32× |
| hardshrink | 0.00737 | 0.00973 | 0.02806 | 0.76× | 0.26× |
| dot_product | 0.15593 | 0.01889 | 0.03948 | 8.25× | 3.95× |
| gemv | 0.07376 | 0.01659 | 0.07332 | 4.45× | 1.01× |
| gemm | 0.26932 | 0.24161 | 0.25400 | 1.11× | 1.06× |
| softmax | 0.01336 | 0.00943 | 0.03369 | 1.42× | 0.40× |
| layer_norm | 0.01638 | 0.00947 | 0.03307 | 1.73× | 0.50× |
| layer_norm_affine | 0.01946 | 0.01736 | 0.04077 | 1.12× | 0.48× |
| layer_norm_backward | 0.60093 | 0.19584 | 0.08366 | 3.07× | 7.18× |
| rms_norm | 0.00947 | 0.00887 | 0.02903 | 1.07× | 0.33× |
| reduce_sum | 0.02806 | 0.01756 | 0.02517 | 1.60× | 1.11× |
| embedding | 0.00748 | 0.01300 | 0.02557 | 0.57× | 0.29× |
| matrix_transpose | 0.85815 | 0.31985 | 0.31245 | 2.68× | 2.75× |
| rope | 0.26363 | 0.02990 | 0.05365 | 8.82× | 4.91× |
| histogram | 0.08337 | 1.05820 | 1.05803 | 0.08× | 0.08× |
| flash_attention | 0.99242 | 0.13896 | 0.13506 | 7.14× | 7.35× |
| merge_attention_states | 0.49130 | 0.08991 | 0.18949 | 5.46× | 2.59× |
| nms | 0.16426 | 8.66564 | 0.19036 | 0.02× | 0.86× |

这些结果不能解读为 “Triton 总是快” 或 “torch.compile 总是快”。单个 `relu`、`add`、`embedding` 已有极低延迟的原生 PyTorch kernel，额外 wrapper 和动态调度使单算子 compile 更慢；在真实模型里，`torch.compile` 的主要价值是跨多个逐元素算子融合。点积、RoPE、状态合并和 naive attention 包含多个 eager kernel 或大中间张量，因此融合收益明显。直方图的少量 bin 导致 atomic 热点，当前 Triton 与 scatter 版本都不如专用 `torch.bincount`。NMS 的 Triton 版本为了保证顺序语义使用 host-driven loop，属于正确性参考实现，不是生产性能实现。

## 接口与限制

- Triton 后端大多数算子是 forward-only 教学/基准实现；`layer_norm_affine` 已注册自定义 autograd 并使用 Triton backward。其他算子需要训练反向时应使用 `torch_compile_backend` 或继续添加对应 backward。
- Triton 后端要求输入是 contiguous CUDA tensor。这样与原目录多数 pybind wrapper 的假设一致，也让 stride 语义明确。
- `layer_norm` 和 `rms_norm` 使用 scalar scale/bias，匹配原 CUDA API；标准可学习向量 gamma/beta 使用 `layer_norm_affine`。
- `rope` 匹配原实现的二维 `[sequence_length, hidden_size]`、相邻偶奇维配对布局。
- `flash_attention` 匹配 `[batch, heads, sequence, head_dim]`，支持 fp16/bf16、causal 与 head dim 16/32/64/128；当前只实现前向且 Q/K/V 序列长度相同。
- `histogram` 只接受非负 int32；生产调用建议显式传 `num_bins`，避免 `max().item()` 带来的 CPU/GPU 同步。
- `nms` 返回原始 box 索引，行为与 `torchvision.ops.nms` 对齐，而不是原 `nms.cu` 返回排序后位置的行为。
- GEMM 的 fp32 路径允许 Triton/NVIDIA 使用目标架构支持的快速矩阵精度策略；对严格 IEEE fp32 有要求时应显式设计独立配置并按误差预算测试。

## 文件结构

```text
kernels/
├── triton_backend/
│   ├── elementwise.py       # add 与七种激活
│   ├── linear.py            # dot、GEMV、GEMM
│   ├── normalization.py     # softmax、LayerNorm、RMSNorm
│   ├── reduction.py         # 多阶段 sum
│   ├── indexing.py          # embedding、transpose、RoPE、histogram
│   └── attention.py         # FlashAttention、state merge、NMS
├── torch_compile_backend/
│   └── ops.py               # 同构纯 PyTorch + Inductor 实现
├── tests/test_accelerated_backends.py
├── validate_backends.py
└── benchmark_backends.py
```

`validate_backends.py` 不依赖 pytest，当前共享环境可直接运行；`tests/test_accelerated_backends.py` 供安装 pytest 的开发环境做参数化回归测试。
