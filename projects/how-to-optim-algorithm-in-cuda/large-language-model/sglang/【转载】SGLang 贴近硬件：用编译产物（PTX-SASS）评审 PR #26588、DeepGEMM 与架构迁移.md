> 原文：SGLang Close to the Metal
> 原文地址：https://gestell.ai/article/sglang-compiled-output-review
> 作者：Gestell
> 发布时间：2026 年 6 月 20 日（约 11 分钟阅读）
>
> 说明：本文转载并整理自 Gestell 的技术博客，保留原文的技术主线、关键数字、表格和指令证据。原文以「编译产物（compiled output）」评审为视角，分析了 SGLang 的 PR #26588、FlashInfer DeepGEMM，以及 Ampere→Hopper 架构迁移三个案例。

# SGLang 贴近硬件：用编译产物（PTX/SASS）评审 PR #26588、DeepGEMM 与架构迁移

> **太长不看版：** 这篇文章讨论的是一种「编译产物评审」视角：不只看源码里算法是否等价，也看 Triton/PTX/SASS 最后发出了什么执行路径。PR #26588 的例子说明，两条源码层面数学等价的融合路径，在 lowering 之后可能改变 BF16/FP32 的实际计算顺序，进而影响 MoE 路由和下游精度。
>
> 这类分析不应该替代端到端精度、CI、benchmark 或 profiler，而更适合作为 kernel/backend PR 的辅助评审产物：提前标出哪些 kernel family 出现或消失、哪些融合数值路径改变、哪些硬件 primitive 或内存搬运模式符合预期。换句话说，它的价值不是“人工读 SASS”，而是把编译后的结构变化整理成维护者能快速判断的信号。

我们选择 SGLang 来做这次「贴近硬件（close to the metal）」的分析。它正好处在一个很有意思的交叉点上，编译产物的行为在这里非常重要：融合 kernel、MoE 路由、多种 config 与 backend 路径，以及 GPU 上架构相关的执行模式。

> 在 lowering（下译）之后到底变了什么？这些变化能不能变成「可评审」的东西？

我们并不打算去 benchmark SGLang 的性能，也不去给各种 backend 和 config 选项排名。源码评审、profiler 和 benchmark 已经把那一层覆盖得很好了。这篇分析揭示的是这些选择在编译之后**真正发出的指令结构**，并由此挖出 SGLang 执行路径里的一些根本特征。

主要案例是 SGLang PR #26588（https://github.com/sgl-project/sglang/pull/26588）：它在下游 Gemma4 GSM8K 出现精度回归之后，回滚（revert）了两条「数学上等价」的融合路径。有意思的问题不在于源码层面的数学是不是等价，而在于 lowering 之后下游到底发生了什么变化。

最后还有一个相关案例，能帮助说明这种分析风格：*跨架构的 Ampere vs. Hopper 执行对比，使用一个 Triton 归一化（normalized）的 backend*。

每一个分析对象都意在说明：为什么用 PTX 和 SASS 作为调查手段，能揭示 SGLang 执行模式中的重要信息。即便是像 SGLang 这种大型项目里很小的 config 改动，也可能产生本质上不同的执行模式，而我们相信这些信息对维护者和用户都有价值。

**方法论：** 对每个案例，我们在相似的 SGLang serving config 下收集编译产物，静态阅读发出的 PTX 和 SASS，按 kernel 家族（family）分组，并跨不同状态对比 kernel 的出现/消失、指令级 delta，以及配置的评审策略。分析器**不执行 kernel、不跑模型、不采集 profiler 计数器、不做运行时 benchmark**。它产出的是一份「编译产物 diff」：哪些 kernel 家族出现、消失或改变，以及哪些指令级的执行结构发生了移动。

---

## PR #26588：等价的数学，不同的浮点执行

### PR #26588 对比 config

在 baseline、pre-revert（回滚前）和被接受的 PR（accepted）三种状态下共享的 SGLang 配置：

```
model-path: google/gemma-4-26B-A4B-it
dtype: bfloat16
kv-cache-dtype: bf16
tensor-parallel-size: 1
expert-parallel-size: 1
attention-backend: triton
fp8-gemm-backend: triton
moe-runner-backend: triton
mem-fraction-static: 0.82
```

PR #26588 有意思的地方不在于 SGLang 出了个 bug，而在于：被回滚的那两条路径在**源码层面是数学等价**的，但作为 lowering 之后的**浮点执行却不等价**。这两个被回滚的 commit，和它们所替换的 eager 代码路径数学上完全等价，路径本身没有 bug。真正的「浮点顺序 bug」发生在 lowering 过程中。用 PR 自己的话说：

> `d72d246a3`：在 `Gemma4Router.forward` 里的 `rmsnorm(x, fused_scale, eps)` 快速路径。它的数学等价于 `(self.norm(x) with weight=1) * fused_scale`，但当 `fused_scale ≈ hidden_size**-0.5 ≈ 0.022` 时，融合 kernel 里的 BF16 累加顺序与两步路径产生了偏离。

> `03826cdd9`：`_gemma4_topk_softmax_scale_kernel`，在一趟（one pass）里同时做 topk + 稳定 softmax + per-expert scale。对 `topk ≤ 8` 来说，它等价于 `softmax(topk_logits) * scale[topk_ids]`，但它内部用 FP32 做的 `exp(top_logit - top1) / sum_top_exp` 顺序，无法和 `torch.nn.functional.softmax` 做到逐 bit 一致。

这就是 failure mode：融合后的 Triton kernel 在做和它们所替换的 eager 版本**相同的数学**，但**顺序不同**。BF16 不满足结合律（associative），所以 bit 会发散。在一个用 `0.022` 去做 scale、并据此喂给专家选择（expert selection）的 router 里，这种 bit 漂移会传播开来：足够多的 token 被路由到了不同的专家，导致 GSM8K 掉了 5 分。源码层面的数学始终等价，但数值执行却在悄无声息地变化。

「出了问题」的信号，既不是数学上的不等价，也不是 profiler 里某个 kernel 变慢，而是**下游精度回归**——而这个信号并不会告诉你回归到底藏在哪。

---

## 两个评审面：源码 vs. 编译产物

GPU 代码有两个评审面（review surface）。

第一个是**源码**。你写的，你能读、能改、能拿它去对照你想表达的意图来测试。SGLang 在源码、config、测试、benchmark 和 profiler trace 这些面上已经有相当成熟的评审手段。这些就是维护者通常用来理解一个 PR 的面：代码改了什么、选了哪条 backend 路径、benchmark 有没有动、profiler 是不是更好看了、eval 还过不过。

第二个面是**编译器吐出来的东西**：真正跑在 GPU 上的 PTX 和 SASS。寄存器分配、指令选择、reduction tree 的形状等等，都是在这里才变得具体。这些决定不是你做的，是编译器做的。Triton 把你的源码 lower 成 PTX，`ptxas` 把 PTX lower 成 SASS，最终跑在芯片上的二进制就是这个结果。

这两个面相关，但在评审意义上是**功能独立**的。一个干净的算法可能在架构移植时编译成丢掉 tensor core 的东西；一个混乱的算法也可能编译得很干净。源码评审和 benchmark 只看第一个面，第二个面传统上根本没人读。Profiler 显示的是编译产物面的**症状**，但不是这个面本身。我们相信，PTX 和 SASS 一旦被评审，就能展现这第二层关键信息。在编译产物 diff 里，可以回答这些问题：

* 哪些 kernel 家族出现了、消失了、或改变了？
* 一条融合的数值路径，是否改变了 conversion、reduction 或 rounding 的结构？
* 一次 backend 切换，是否发出了预期的内存搬运路径？
* 一次架构迁移，是否引入了预期的 tensor-core 和 warpgroup 指令？
* 按项目策略，这个改动应该判定为 PASS、REVIEW，还是 FAIL？

PR #26588（https://github.com/sgl-project/sglang/pull/26588）正好展示了这个模式。 源码层面的恒等式本意是等价的，但融合 kernel 发出了不同的数值执行路径。要找到 bug 的根源，需要调查的恰恰是编译器这一面。因此一份编译产物 diff 可以**增强** eval 流程：它会在 eval 结果成为第一个有用信号之前，先给维护者一个更小的评审产物，明确指出哪些融合路径发生了变化。

---

## 编译产物揭示了什么

我们对 3 个 SGLang 状态做了静态读取：baseline（7ed53d15f，https://github.com/sgl-project/sglang/commit/7ed53d15f357ea4d722c1980c2cb35e8367d8bb0 ，PR 前的 main）、pre-revert（256e1d6c6，https://github.com/sgl-project/sglang/pull/26588/commits/256e1d6c6d739770d1805b7e7ae418bc4698c091 ，四个 commit 全部生效）、以及移除了精度敏感融合路径之后被接受的 PR 状态（847ce14，https://github.com/sgl-project/sglang/pull/26588/commits/847ce14af3a8d5c19694668f767d334622bb216c）。

相同的模型、相同的 dtype、相同的 Triton backend、相同的 MoE config。三份缓存，逐一 diff。我们只是读了编译产物里发出的 PTX/SASS。

在源码层面，PR 把这次 router RMSNorm 改动描述成 `Gemma4Router.forward` 里一条融合的 `rmsnorm(x, fused_scale, eps)` 快速路径。编译产物层面的关键问题是：这个「源码等价」的改动，在 lowering 之后是不是留下了一条**不同的执行路径**。

相关的转换（conversion）边界在 PTX 里清晰可见：

```
cvt.rn.bf16.f32   %rs1, %f1;
```

把一个 FP32 值按 round-to-nearest-even 舍入成 BF16（7 位尾数）。把一条很长的 FP32 链累加进一个 BF16 存储，再把这条链重排后做一遍，你就会得到两个量级接近、但 bit 不同的答案。

### PR #26588 的验证结果

| 状态 | commit | 说明 | 结果 | GSM8K (MTP topk=1) | 平均接受长度 |
|---|---|---|---|---|---|
| Baseline | 376635c1e | all reverts = main | pass | 0.445 | 4.494 |
| Accepted | 27cb94c4 | 5c1+c4 kept | pass | 0.445 | 4.494 |
| Pre Revert | 10ab189e3 | all 4 commits | **fail** | **0.360** | 4.475 |

### RMSNorm 指令构成（按类别求总和）

下表对比 RMSNorm kernel 家族在 base、pre-revert、accepted 三种构建下的类别总量：

| 类别 | Base（合计 **5,367**） | Pre Revert（合计 **6,231**） | Accepted（合计 **6,761**） | Δ |
|---|---|---|---|---|
| 标量计算（Scalar compute） | 2,623 | 3,123 | 3,497 | +874 |
| 同步（Synchronization） | 1,036 | 1,144 | 1,144 | +108 |
| 内存搬运（Memory movement） | 654 | 760 | 840 | +186 |
| 共享内存（Shared memory） | 416 | 464 | 464 | +48 |
| 宽共享内存（Wide shared memory） | 416 | 464 | 464 | +48 |
| 宽内存搬运（Wide memory movement） | 216 | 248 | 324 | +108 |
| 控制流（Control flow） | 6 | 28 | 28 | +22 |

Pre Revert 加入了 by-head（逐头）的 RMSNorm 路径。Accepted 移除了那条路径，但保留了更大的标准 RMSNorm 足迹（footprint），所以 RMSNorm 的总指令数仍然高于 base。

### RMSNorm 指令均值（每 kernel 平均）

下表每一行独立缩放，用来看「每 kernel 平均」在三种构建之间如何移动：

| 类别 | Base | Pre Revert | Accepted | Δ |
|---|---|---|---|---|
| 标量计算 | 1,311.5 | 780.75 | 874.25 | -437.25 |
| 同步 | 518 | 286 | 286 | -232 |
| 内存搬运 | 327 | 190 | 210 | -117 |
| 共享内存 | 208 | 116 | 116 | -92 |
| 宽共享内存 | 208 | 116 | 116 | -92 |
| 宽内存搬运 | 108 | 62 | 81 | -27 |
| 控制流 | 3 | 7 | 7 | +4 |

Accepted 形成了「四 kernel」的 RMSNorm 足迹，所以即便 RMSNorm 总指令数上升，每 kernel 平均仍低于 base——更大的标准路径被拆分到了更多专用的发出 kernel 上。

在编译产物分析里，Pre-Revert 与 Accepted 两个状态之间，整个 kernel 舰队（fleet）的大部分保持不变。Pre-revert 跑出来观测到 59 个 kernel，accepted 是 58 个。在这次 diff 中，56 个 kernel 被对比，54 个未变，2 个改变，2 个新增，3 个被移除。

### 编译 kernel 对比

跨 baseline、accepted、pre-revert 三种构建的 kernel 计数：

| | Baseline | Accepted | Pre Revert |
|---|---|---|---|
| 观测到的 kernel | 44 | 58 | 59 |
| fused_moe_kernel | 12 | 24 | 24 |
| _gemma_qkv_rmsnorm_kernel | 2 | 4 | 2 |

Accepted 移除了 by-head 的 RMSNorm 路径和融合的 top-k softmax-scale 路径，同时保留了更广的 MoE 展开。

这就是这个 PR 在编译形态下的样子。被接受的 patch 并不是说「撤销这个优化」，而是说：保留更广的 MoE 与 attention 改动，移除那些改变了路由行为的融合数值路径。

top-k 这个点展示了这一模式最干净的版本。PR #26588 描述 `_gemma4_topk_softmax_scale_kernel` 把 top-k、稳定 softmax 和 per-expert scale 融合进了一趟。在实数算术里，这和 softmax 一样；但在浮点执行里，顺序不同。

编译 diff 直接把那条融合路径暴露出来。在 pre-revert 构建里，这个融合 top-k kernel 存在；在 accepted 构建里，它消失了。Triton 的 IR 只要维持实数算术语义，就把 FP32 工作的重排视为等价，因此它的 pass 可以按优化器的偏好随意调度这些工作——而它挑的调度方案，恰恰不是 PyTorch 两步路径产生的那个。

### 被移除的融合 top-k softmax-scale 路径

| 信号 | PTX/SASS 指令 | Pre-revert | Accepted |
|---|---|---|---|
| Kernel 家族 | _gemma4_topk_softmax_scale_kernel | 10 | — |
| 该路径里的 BF16 转换 | cvt.f32.bf16 | present | removed |
| 指数运算 | ex2.approx.f32 | present | removed |
| 倒数路径 | MUFU.RCP | present | removed |

实数算术里是同一个 softmax，编译出来却是不同的数值路径。kernel 计数和指令全都动了。算法面说「等价」，编译面说「执行不同」。

RMSNorm 这边形状一样。pre-revert 构建里包含两个 `_gemma_qkv_rmsnorm_by_head_kernel` 实例，它们在 accepted 构建里消失；取而代之，accepted 构建发出了四个 `_gemma_qkv_rmsnorm_kernel` 实例。

### RMSNorm 路径迁移

baseline 的形状被带入了 pre-revert 与 accepted 两个 PR 结果：

**Base + delta**

| | 数量 |
|---|---|
| base 标准 RMSNorm | **2** |
| PR delta 路径 | **2** |

**Pre Revert**

| | 数量 |
|---|---|
| 标准 RMSNorm | **2** |
| by-head RMSNorm | **2** |

**Accepted**

| | 数量 |
|---|---|
| 标准 RMSNorm | **4** |

Base 贡献两个标准 RMSNorm kernel。PR 又加了两个 RMSNorm 路径 kernel；Pre Revert 把它们物化（materialize）成 by-head kernel，而 Accepted 把它们物化成标准 kernel。

编译产物 diff 把发生改变的精度敏感执行路径展示了出来。

SGLang 其实拥有正确的验证机制：kernel 作者理解这次改动，profiler 显示融合路径更快，确定性精度 CI 也确实报警了。但这些工具暴露的是不同的面。源码评审看到的是等价的数学；单测通过，因为数值接近；profiler 很开心，因为 kernel 更快了。唯一指向这次回归的信号是 GSM8K 分数，而分数并不会说回归在哪。

于是该 PR 的验证过程做了 bisect：四个 commit、多种配置、每个配置 200 道 GSM8K 题目，在一台 TP=2 的 H200 上跑。这个 bisect 是对的。但这也正是「当没有任何东西替你读编译器这一面时，你不得不亲自去做的工作」。上面那份编译产物 diff，在**不做 bisect** 的情况下得到了和这次 bisect 相同的答案，直接指出执行路径在悄无声息地改变，并为那段精确的源码提供了一个评审产物。这份 diff 不会取代验证，但它能**更早**地缩小嫌疑集合——通过展示出改动后的 PR 状态引入并移除了哪些具体的融合数值路径，以及相关的 conversion 和 reduction 指令。

---

## 这在 PR 评审里会是什么样子

### 编译产物 PR 评审产物

一份紧凑的策略产物，可以在不要求维护者去看完整 PTX/SASS diff 的前提下，把发出的 kernel 家族变化标记出来：

```
version: v1
rendering:
default: delta
rules:
- id: review-kernel-family-deltas
  effect: review
  match:
    family: "*"
  reason: New specialized kernels have been emitted, flagged for review
```

### 评审产物的决策流程

报告始终可供评审。团队可以在这份 config 之上，针对有问题的 PTX/SASS 变化构建更锐利的标记规则。

- **编译产物 diff** —— lowering 之后的 PTX 与 SASS。
- **kernel 家族 delta 检查** —— 对比发出的 kernel 家族。
  - **PASS 路径**：没有新建 kernel，编译面停留在已知家族集合内 → **通过**。评审产物在通过后仍然可查阅。
  - **REVIEW 路径**：新建了 kernel，diff 里出现了专用的发出 kernel → **需评审**。报告标记出这次 kernel 家族 delta。

这件事的价值，不在于要求维护者逐行读 PTX 或 SASS，而在于产出一份**结构化的评审产物**。kernel 的 owner 随后就可以决定：这个改动应该 pass、应该走 review，还是应该绑定一个精度或 profiler 的 gate。

---

## 把 DeepGEMM kernel 当作编译产物 CI 门禁

PR #26588 展示了数值执行在 lowering 之后发生变化。DeepGEMM 展示了第二种用法：把 backend 的「预期」变成一条编译产物评审策略。

### FlashInfer DeepGEMM config

使用 `flashinfer_deepgemm` 搭配 `deep_gemm` MoE runner 的 FP8 路径：

```
model-path: DeepSeek-V2-Lite-Chat-FP8
dtype: bfloat16
kv-cache-dtype: bf16
tensor-parallel-size: 1
expert-parallel-size: 1
attention-backend: triton
fp8-gemm-backend: flashinfer_deepgemm
moe-runner-backend: deep_gemm
mem-fraction-static: 0.82
```

### SGLang 内置 DeepGEMM config

使用 SGLang 自带 `deep_gemm` backend 和 `deep_gemm` MoE runner 的 FP8 路径：

```
model-path: DeepSeek-V2-Lite-Chat-FP8
dtype: bfloat16
kv-cache-dtype: bf16
tensor-parallel-size: 1
expert-parallel-size: 1
attention-backend: fa3
fp8-gemm-backend: deep_gemm
moe-runner-backend: deep_gemm
mem-fraction-static: 0.82
```

TMA 专门设计用来在 Hopper 级 GPU 上卸载 global memory 与 shared memory 之间的大块搬运。如果一条 backend 路径在配置时就预期「tile 搬运应该走 TMA」，那么编译产物就应该能确认这个预期在 lowering 之后是否依然成立。

这就给了我们一个很自然的策略问题：

> 如果一条 backend 路径被配置成「tile 搬运应该用 TMA」，那么发出的 PTX/SASS 真的符合这个预期吗？

我们用一条严格的评审策略（https://github.com/deepseek-ai/DeepGEMM#notices）跑了 DeepGEMM 对比：

> 如果一条 GEMM 路径预期使用 TMA，那么对反复出现的「手动 global→shared 暂存」标记为需评审。

### 编译产物策略规则

当一个 DeepGEMM kernel 发出匹配的 PTX shared-store 指令时，触发评审：

```
version: v1
rules:
- id: review-gemm-when-tma-expected
  effect: review
  match:
    all:
      - kernel: deep*gemm*
      - operation: st.shared.*
        layer: ptx
        exclude:
          operation: st.shared.f32
  when:
    count:
      gte: 1
  reason: GEMM kernel used repeated manual global-to-shared copies. Check whether this tile movement should have lowered to TMA.
```

在这条策略下，这个 config 在 SGLang FP8 路径和 DeepGEMM 路径上**都**触发了：

### DeepGEMM 策略评审信号

两种配置只在 FP8 GEMM backend 上不同，却命中了同一条 PTX shared-store 评审规则：

| | FlashInfer DeepGEMM（**8**） | SGLang 内置 DeepGEMM（**13**） |
|---|---|---|
| fp8-gemm-backend | flashinfer_deepgemm | deep_gemm |
| moe-runner-backend | deep_gemm | deep_gemm |
| 策略信号 | PTX st.shared.\* | PTX st.shared.\* |
| 受影响的 kernel | `deepgemm_compute_src2dst_triton_kernel` | `deepgemm_compute_src2dst_triton_kernel`、`deep_gemm::transpose_fp32` |
| 评审阈值 ≥ 1 | ✓ | ✓ |

表中计数是匹配到的 PTX shared-store 指令数。策略阈值是 ≥ 1，所以两种配置在这条规则下都需要评审。

这个门禁在 DeepGEMM 集成路径上触发：`flashinfer_deepgemm` 命中 `deepgemm_compute_src2dst_triton_kernel`（https://github.com/sgl-project/sglang/blob/d271de64fe92678ad3236ec965c0edd9cfe1b505/python/sglang/srt/layers/moe/ep_moe/kernels.py#L1111 ，8 条指令）；`deep_gemm` 命中同一个 kernel，还额外命中 `deep_gemm::transpose_fp32`（https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/csrc/jit_kernels/impls/smxx_layout.hpp#L31 ，5 条指令）。这里是不同的 kernel 却有相同的手动 shared 暂存模式——门禁抓的是**一类行为**，而不是只盯着某个库的选择。

那个受影响的 helper kernel 是很具体的。在普通的 FP8 路径里它不出现；在 DeepGEMM TMA-config 的运行里，它出现了两次，并随之带来内存搬运、共享内存活动、同步和标量计算。

### src2dst helper 的指令证据

在 `deepgemm_compute_src2dst_triton_kernel` 里观测到的 PTX 与 SASS 指令：

| 指令 | 计数 |
|---|---|
| PTX ld.global.v2.b64 | 1 |
| PTX ld.global.b32 | 4 |
| PTX ld.global.b64 | 6 |
| PTX ld.shared.b32 | 4 |
| PTX ld.shared.b64 | 4 |
| PTX st.shared.b32 | 4 |
| PTX st.shared.b64 | 4 |
| SASS LDG.E | 8 |

| 指令 | 计数 |
|---|---|
| SASS LDG.E.128 | 1 |
| SASS LDG.E.64 | 2 |
| SASS LDS | 4 |
| SASS LDS.64 | 4 |
| SASS STS | 4 |
| SASS STS.64 | 4 |
| SASS STG.E | 4 |

### transpose helper 的指令证据

在 `deep_gemm::transpose_fp32` 里观测到的 PTX 与 SASS 指令：

| 指令 | 计数 |
|---|---|
| PTX st.shared.u32 | 5 |
| SASS LDS | 6 |

| 指令 | 计数 |
|---|---|
| SASS STS | 40 |
| SASS STG.E | 6 |

源码和 config 评审说「DeepGEMM 路径已启用」。编译产物门禁问的是一个更窄的问题：发出的 kernel 是否匹配这条策略所期望的搬运模式？在上面的抓取里，答案不是「backend 错了」，而是「需要评审」。kernel owner 随后可以接受这条发出路径、调整策略，或者去调查这个 helper 路径是否应该使用不同的搬运结构。

---

## 架构变化会移动编译产物面

第三个案例是一次架构迁移：相同的 SGLang serving 路径，Ampere → Hopper。这正是「源码/config 相等」可能掩盖大量二进制层面移动的地方。

### Ampere 到 Hopper 对比 config

用相同的 SGLang serving 路径来对比跨 GPU 架构发出的指令：

```
model-path: DeepSeek-V2-Lite-Chat
dtype: bfloat16
kv-cache-dtype: bf16
tensor-parallel-size: 1
expert-parallel-size: 1
attention-backend: triton
fp8-gemm-backend: triton
moe-runner-backend: triton
mem-fraction-static: 0.82
```

kernel 家族是稳定的：对比了 13 个家族、13 个 kernel，0 新增、0 移除。但**每一个**被对比的 kernel 都变了。跨架构迁移移动了发出的执行结构。

### Ampere 到 Hopper 的指令移动

相同的 SGLang serving 路径，不同的 GPU 架构：总量展示了 lowering 之后哪些发出指令类别发生了移动：

| 类别 | Ampere | Hopper | Δ |
|---|---|---|---|
| 标量计算 | 18,408 | 18,118 | -290 |
| 内存搬运 | 3,399 | 3,645 | +246 |
| 共享内存 | 898 | 814 | -84 |
| HMMA tensor-core 路径 | 808 | 744 | -64 |
| Hopper WGMMA tensor 计算 | 0 | 8 | +8 |
| Warpgroup / 寄存器控制 | 0 | 14 | +14 |
| 同步 | 680 | 708 | +28 |
| 异步搬运（Async movement） | 200 | 200 | 0 |

`fused_moe_kernel` 家族展示了同样的现象。两种架构上这个家族的计数都固定在 10 个 kernel，但发出的执行结构在移动：

### `fused_moe_kernel` 家族的移动

在 Hopper 上，`fused_moe_kernel` 现在会发出 tensor-core、warpgroup、内存和标量指令：

| 类别 | Ampere | Hopper | Δ |
|---|---|---|---|
| Kernel 计数 | 10 | 10 | 0 |
| HMMA tensor-core 路径 | 128 | 64 | -64 |
| Hopper WGMMA tensor 计算 | 0 | 8 | +8 |
| Warpgroup / 寄存器控制 | 0 | 14 | +14 |
| 内存搬运 | 574 | 627 | +53 |
| 共享内存 | 218 | 170 | -48 |
| 标量计算 | 4,515 | 4,479 | -36 |
| 异步搬运 | 192 | 192 | 0 |

### Hopper 专有指令证据

Hopper 专有的 tensor-core 和 warpgroup 指令出现在发出的 PTX 与 SASS 里：

| 信号 | 层 | 观测计数 |
|---|---|---|
| wgmma.mma_async... | PTX | 12 |
| wgmma.commit_group.sync.aligned | PTX | 4 |
| wgmma.fence.sync.aligned | PTX | 4 |
| wgmma.wait_group.sync.aligned | PTX | 8 |
| HGMMA | SASS | 8 |
| WARPGROUP.ARRIVE | SASS | 4 |
| WARPGROUP.DEPBAR.LE | SASS | 8 |

源码评审说「同一个 kernel 家族」；config 评审说「同一条 serving 路径，不同架构」。而编译产物 diff 展示的是真正变了的东西：哪条 tensor-core 路径出现了、哪种共享内存模式移动了、哪些同步/控制结构发生了改变。

---

## 这意味着什么

算法面和编译产物面会发生分歧（diverge），而这种分歧对源码评审、benchmark 和 profiler 来说可能是不可见的。PR #26588 就是那个「分歧被发布出去、又在下游被抓到」的实际样例。DeepGEMM 是同样的形状，但被做成了一条维护者可以在每个 commit 上跑的 CI 策略。Ampere-到-Hopper 案例则是同样的分析放进一次部署迁移里。

编译产物 diff 的意义在于：它产出的是一份小而结构化的产物，维护者或 agent 可以读它，而不必去读原始 SASS。在 PR #26588 里，这份产物会在 bisect 之前就标记出两个被移除的融合 kernel 和一个被扩展的 kernel，并把精度敏感的指令模式内联列出来。在一次 DeepGEMM CI 运行里，同样的产物会回答一个配置好的策略问题，并据此 gate。在一次架构迁移里，这份产物会展示哪些 kernel 家族迁移到了新的 tensor 和内存路径，哪些没有。

这套分析当然还有可以改进的方向：更好的 source-to-SASS 归因（attribution）、更丰富的策略预设，以及与 profiler trace 的集成。但核心产物是有用的：一份编译产物 diff，能在执行路径的变化「只通过 benchmark、profiler 或下游 eval 才显现」之前，就让它们可见。

在 Gestell，我们很期待编译产物评审能为像 SGLang 这样的开源项目带来怎样的增益。

如需上述编译产物分析的完整报告，请联系 hello@gestell.ai。
