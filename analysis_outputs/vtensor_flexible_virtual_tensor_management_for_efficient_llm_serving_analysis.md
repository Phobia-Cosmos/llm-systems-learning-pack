# Paper Analysis

论文：vTensor: Flexible Virtual Tensor Management for Efficient LLM Serving

作者：Jiale Xu, Rui Zhang, Cong Guo, Weiming Hu, Zihan Liu, Feiyang Wu, Yu Feng, Shixuan Sun, Changxu Shao, Yuhong Guo, Junping Zhao, Ke Zhang, Minyi Guo, Jingwen Leng

会议/期刊：arXiv preprint, arXiv:2407.15309。未核实到正式会议/期刊发表版本。

年份：2024。注：本地文件名包含 `2025arXiv`，但论文正文和 arXiv 页面显示提交日期为 2024-07-22。

源码/项目：

- 官方论文页：https://arxiv.org/abs/2407.15309
- 官方代码：未在论文、arXiv 页面或轻量搜索中找到 vTensor/FlexInfer 官方公开仓库。
- 相关开源实现：Microsoft vAttention, https://github.com/microsoft/vattention
- vLLM feature request：https://github.com/vllm-project/vllm/issues/6687

本报告依据：

- 本地 PDF：`/home/undefined/Desktop/ai/papers/03_kv_cache_serving_inference/2025arXiv-vTensor Flexible Virtual Tensor Management for Efficient LLM Serving.pdf`
- arXiv 官方页面、vLLM issue、Microsoft vAttention 页面、KV cache survey 与后续 eLLM 论文页面。网页状态核对日期：2026-07-02。

论文速览：

| Item | Content |
| --- | --- |
| 研究对象 | LLM serving 中 KV cache 的 GPU 内存管理与 attention kernel 解耦 |
| 核心问题 | vLLM/PagedAttention 解决了 KV cache 碎片，但把分页地址翻译耦合进 attention kernel，带来开发复杂度、tensor core 利用受限和静态预留内存浪费 |
| 方法摘要 | 用 CUDA GPU Virtual Memory Management 构造 vTensor：计算侧看到连续 tensor pointer，物理侧可映射到非连续 GPU chunks；FlexInfer 在 CPU 侧调度 VMM 操作并与 GPU 计算重叠 |
| 任务/场景 | 单轮长上下文生成、多轮 chat、prefix sharing/prefix caching、decode kernel、prefix-prefill kernel |
| 数据集 | SGLang synthetic multi-turn/prefix-sharing 数据；LV-Eval 截断构造长输入长输出单轮生成数据 |
| 主要指标 | kernel latency、end-to-end throughput、GPU memory footprint、memory trace |
| 关键结论 | 论文报告 end-to-end 平均 1.86x speedup、最高 2.42x，多种 kernel 最高 3.92x/3.27x，并相对 vLLM 平均释放约 71.25% 即 57GB A100 GPU 内存 |
| 开放资源 | 论文开放；代码未找到官方发布；相关思想在 vAttention/eLLM 等后续或并发工作中继续发展 |

## 第一部分 研究背景

LLM serving 的核心矛盾是：请求到达是动态的，序列长度差异很大，KV cache 随 decode 逐 token 增长；但 GPU 高性能 kernel 往往希望输入是规则、连续、易向量化的 tensor。为了提高吞吐，服务系统会做 batching/continuous batching，但不同请求长度不同会造成严重内存碎片。如果预留每个请求的最大长度，短请求浪费大量 KV cache 空间；如果不预留，decode 阶段又需要频繁扩展内存。

vLLM 的 PagedAttention 是这个方向的关键系统：它借鉴操作系统分页，把每个请求的 KV cache 切成 blocks，用 block table 管理逻辑块到物理块的映射，从而显著降低碎片并提高 batch size。但分页方案有一个系统代价：attention kernel 不再能把 KV cache 当成普通连续 tensor，需要在 kernel 内部处理 page table、地址翻译和非连续访存。这带来三类压力：

1. 性能压力：分页地址翻译和非连续访问会让 kernel 更难使用 tensor core 或新 attention kernel 的优化路径。
2. 开发压力：每出现一种新的 attention/kernel/模型结构，都可能要重写 paged 版本。
3. 内存压力：vLLM 需要为 KV cache 预留大块 GPU memory；在多实例、动态负载或与 activation/其他任务共享时，这些被 reserved 的内存不够弹性。

vTensor 的问题定义就是：能否保留 PagedAttention 的“物理内存按需分配/减少碎片”优势，同时让计算 kernel 仍然看到普通连续 tensor？如果能做到，就可以把内存管理从 attention kernel 中剥离出来，让 FlashAttention、FlashInfer、cuBLAS 风格的高性能实现更容易复用。

## 第二部分 历史发展

| 时间 | 代表工作/阶段 | 核心思想 | 尚未解决的问题 | 与本文关系 |
| --- | --- | --- | --- | --- |
| 早期 Transformer serving | 静态 KV cache 预分配、padding、固定 batch | 简单，兼容标准 kernel | 序列长度差异导致大量碎片；batch size 被内存限制 | vTensor 要解决的原始痛点 |
| FlashAttention / FlashInfer | 通过 IO-aware attention 和高效 kernel 提升 attention 性能 | 减少 HBM 访问，提升 kernel 性能 | 本身不解决动态 KV cache 管理；若接入分页 cache 需要额外适配 | vTensor 希望让这些 kernel 无需分页改造即可使用 |
| Continuous batching / Orca / Sarathi-Serve | iteration-level scheduling，提高 GPU 利用率 | 请求完成后可立即插入新请求 | KV cache 动态增长和碎片仍是瓶颈 | 提供 serving 负载背景 |
| vLLM / PagedAttention | 用 block table 管理 KV cache，减少碎片和重复复制 | 大幅提升吞吐，成为事实基线 | page table 与 attention kernel 耦合；需要预留 KV cache 内存 | vTensor 的直接对比对象 |
| Prefix caching / Prompt cache / SGLang | 复用 shared prefix 的 KV cache，降低 prefill 成本 | 多轮对话和 shared system prompt 中很有价值 | prefix cache 进一步增加内存管理复杂度 | vTensor 用 radix tree 支持 prefix record/match |
| GMLake | 用 CUDA VMM 做 GPU memory defragmentation，主要面向训练/大规模 DNN | 证明 GPU VMM 可用于深度学习内存管理 | 不是专门为 LLM serving/KV cache 设计 | vTensor 沿用 VMM 思路到 serving |
| vAttention, 2024 | 用 CUDA VMM 保持 KV cache 虚拟连续、物理按需分配 | 支持未修改 attention kernel，开源实现 | 与 vTensor 高度相关/并发；具体调度、prefix 支持、系统评测不同 | 是最接近的并发/后续比较对象 |
| vTensor, 2024 | 用 vTensor + FlexInfer 解耦 KV cache 内存管理和计算 kernel | 兼顾连续虚拟地址、动态物理映射、prefix cache 和 CPU-GPU 异构调度 | 官方代码未公开；评测覆盖有限 | 本文 |
| eLLM, 2025 | 同一作者线进一步提出 elastic memory，把 KV cache 和 activation 放入统一虚拟 tensor/弹性内存框架 | 试图突破 KV cache 与 runtime activation 分层管理的隔离 | 更复杂，需要 SLO-aware scheduling 和 CPU buffer | 可视作 vTensor 思想的扩展 |

## 第三部分 本文创新

### Novelty

本文的新意是把“KV cache 分页/碎片整理”从 attention kernel 中移走，放到 GPU VMM 管理的虚拟内存层。计算 kernel 只看到一个连续 virtual address，对它来说 vTensor 像普通 CUDA tensor；真实物理内存可以是不连续的 GPU chunks，并由 CPU 侧 vTensor Manager 调度 cuMemCreate、cuMemAddressReserve、cuMemMap、Unmap 等 VMM 操作。

这与 vLLM 的关键差异是抽象边界不同：

- vLLM：分页抽象在 serving runtime 和 attention kernel 之间，kernel 要懂 page table。
- vTensor：分页/映射抽象在 GPU virtual memory 层，kernel 不需要懂 page table。

### Contributions

1. 指出 PagedAttention 的双重代价：虽然减少 KV cache 碎片，但把内存管理耦合进 kernel，限制计算灵活性和内存弹性。
2. 提出 vTensor 抽象：连续虚拟地址 + 非连续物理 chunks + CPU 可管理的 physical handles。
3. 设计 vTensor Manager，包括 vTensor Pool、vTensor Operations、vTensor Scheduler，并提供 allocation、deallocation、prefix tree 操作。
4. 实现 FlexInfer：基于 vLLM 修改约 3000 行 Python/C++，集成 flash-attn-2.5.8，用 vTensor 支撑 decode、prefix-prefill、multi-turn chat 等场景。
5. 在 A100 80GB 上评测 Yi-6B-200K、Yi-9B-32K、Yi-34B-32K 等模型，报告 kernel 和 end-to-end 提升，以及显著 GPU memory 释放。

### Core Insight

核心洞察：KV cache “逻辑上连续”与“物理上连续”不是同一件事。只要 GPU VMM 能把连续虚拟地址映射到非连续物理 chunks，attention kernel 就可以继续使用高性能连续 tensor 接口，而 serving runtime 仍能按需扩展/回收物理内存。

## 第四部分 方法详解

逐模块说明 Why / How / Result / Failure Mode。

| 模块 | Why | How | 为什么有效 | 可能问题 |
| --- | --- | --- | --- | --- |
| vTensor abstraction | 让 kernel 不感知分页，同时消除物理碎片 | 为每个请求保留连续 virtual address；物理 chunks 按需创建并映射 | 虚拟连续满足标准 CUDA kernel；物理非连续减少碎片 | 依赖 CUDA VMM；跨 GPU/旧硬件/不同驱动行为可能复杂 |
| vTensor Pool (vSet/pSet/rTree) | 管理虚拟地址、物理 chunk 和 prefix cache | vSet 存 virtual address；pSet 存 physical handles/ref count；rTree 做 prefix match | 把 KV cache 生命周期和 prefix 复用显式建模 | 元数据一致性、并发调度和 ref count 错误会很危险 |
| vTensor Operation | 封装内存动作 | vAlloc, pAlloc, Map, Unmap, vFree, pFree, rPush, rPrefixMatch | 将 VMM primitive 转成 serving 可用操作 | cuMem* 调用开销可能高；chunk size 选择影响碎片与开销 |
| Dynamic extension | decode 阶段 KV cache 逐步增长 | 预先为下一步 token 分配/映射 physical chunk，并与已有计算重叠 | 利用 autoregressive 规律隐藏内存分配开销 | 如果输出突增、负载抖动或 preemption 频繁，重叠可能不充分 |
| Prefix record/match | 多轮对话或 shared prompt 避免重复 prefill | 完成请求后把 vTensor 插入 radix tree；新请求匹配 prefix 后复制 page table/metadata 并共享 physical chunks | prefix cache 不必复制真实 KV 数据，只增加映射关系 | prefix pattern 管理复杂；共享 chunks 的释放和隔离需要谨慎 |
| Lazy deallocation | 减少频繁释放/重建的开销 | 多数情况下 Unmap 并放回可复用集合，任务结束或显式操作时再释放 | 减少 VMM API 调用和 GPU 分配抖动 | 可能造成暂时性内存占用偏高；需要好的 release 策略 |
| FlexInfer Scheduler | 把请求状态转成 memory instructions，并与 GPU kernel 调度协调 | prefill 时 Create/Prefix_Match；decode 时 Extend；资源不足时 preempt low-priority request；finished 时 Prefix_Record | 让内存操作在 CPU 侧异步执行并尽量被 GPU 计算隐藏 | 论文没有充分展开 SLO、公平性、复杂生产调度策略 |

一个关键技术细节是 chunk size：论文使用 2MB physical chunk。这个选择是在 VMM 调用开销、page table/metadata 开销和内存碎片之间折中。chunk 太小会增加映射和元数据开销；chunk 太大又会损失按需分配的细粒度。

## 第五部分 实验

| Item | Content |
| --- | --- |
| Datasets | SGLang synthetic multi-turn chatbot/prefix-sharing 数据；LV-Eval 中截断构造 6K-8K prompt、2K-4K output 的单轮长上下文数据 |
| Baselines | vLLM v0.4.2；vLLM PagedAttention；Paged FlashAttention；native FlashAttention；SGLang Triton prefix-prefilling kernel |
| Metrics | kernel latency；end-to-end throughput/token/s；GPU memory usage；memory trace |
| Main Results | end-to-end 平均 1.86x speedup，最高 2.42x；decode kernel 平均 2.12x，最高 3.27x；prefix-prefill kernel 平均 3.15x，最高 3.92x；A100 上相对 vLLM 平均释放约 57GB |
| Ablations | batch size、sequence length、KV head 数、prefix ratio、single-generation、多轮 chat、prefix sharing、memory trace |
| Efficiency/Cost | 8x A100 80GB + Intel Xeon Platinum 8369B；模型包括 Yi-6B-200K、Yi-9B-32K、Yi-34B-32K；不同 TP 设置 |
| Reproducibility | 说明基于 vLLM 修改约 3000 行并集成 flash-attn-2.5.8，但未找到官方公开代码，复现难度较高 |
| Experimental Weaknesses | baseline 版本较旧；未与 vAttention 做直接实验比较；没有真实线上 trace；主要 GPU 为 A100；缺少端到端 tail latency/SLO 分析 |

### 5.1 Kernel 结果

decode kernel 评测中，FlexInfer attention 在 batch size 变化时平均比 vLLM PagedAttention 快 2.78x，峰值 3.08x；在 sequence length 变化时平均比 PagedAttention 快 2.67x，峰值 3.27x。论文解释是：PagedAttention 的页表翻译在 CUDA core 上执行，难以使用 tensor core 路径；而 vTensor 让 kernel 保持连续 tensor 视角，可以直接利用高性能 FlashAttention 风格实现。

prefix-prefilling kernel 中，FlexInfer 相比 SGLang Triton prefix-prefill 平均 3.40x，峰值 3.49x；prefix/prompt ratio 变化时最高 3.92x。这里的核心证据支持“计算灵活性”主张：如果 memory management 不侵入 kernel，就更容易复用或接入更快的 attention kernel。

### 5.2 End-to-end 结果

single-generation 场景中，Yi-6B-200K、Yi-9B-32K、Yi-34B-32K 的平均 throughput 提升分别约 1.8x、1.3x、1.4x，batch size=64 时峰值分别约 2.02x、1.5x、1.53x。

prefix-caching 场景中，FlexInfer 在 multi-turn chat 中最高约 2.42x，在 fork/prefix-sharing 场景中最高约 2.0x。论文认为主要收益来自更高效的 prefix-prefill kernel 和 prefix cache 映射复用。

### 5.3 Memory 结果

论文报告 FlexInfer 可随 batch size 动态调整 KV cache memory usage；当 batch size 小时，可相对 vLLM 释放几乎全部预留 KV cache 空间。相比之下，vLLM 静态预留 KV cache pool，在请求率波动时会出现 GPU memory 被保留但未充分使用的问题。

## 第六部分 与已有工作的比较

| Work | Key Idea | Difference From This Paper | Limitation |
| --- | --- | --- | --- |
| 静态 KV cache / padding | 每个请求预分配最大长度 KV cache，保证 tensor 连续 | vTensor 保留虚拟连续但物理按需分配 | 静态方式碎片严重，长短请求混合时浪费大 |
| FlashAttention / FlashInfer | IO-aware attention kernel，提高计算效率 | vTensor 不直接发明新 attention，而是让这些 kernel 更容易在动态 KV cache 场景复用 | 不单独解决 KV cache 生命周期、prefix 复用和 serving 调度 |
| vLLM / PagedAttention | block table + paged KV cache，降低碎片，提高 batch size | vTensor 把地址映射下沉到 GPU VMM，kernel 不需要处理 page table | PagedAttention 需要 paged kernel，开发和优化成本高 |
| SGLang prefix caching | prefix cache 和高效 serving runtime | vTensor 用 rTree 和 virtual mapping 复用 prefix physical chunks | prefix kernel/内存管理仍需适配具体 runtime |
| GMLake | CUDA VMM 用于 GPU memory defragmentation，主要面向 DNN training | vTensor 面向 LLM serving/KV cache，结合 prefix cache 和 decode extension | GMLake 不是 serving runtime，也不处理 autoregressive KV 生命周期 |
| vAttention | CUDA VMM 解耦 virtual/physical KV cache，支持未修改 attention kernel，Microsoft 开源 | 与 vTensor 思路高度接近；vTensor更强调 vTensor Pool、prefix tree、FlexInfer 调度和多个 serving 场景 | vTensor 论文未直接与 vAttention 实验比较；vAttention 需要特定 runtime/driver 支持 |
| eLLM | 进一步统一 activation 与 KV cache 的弹性内存管理 | 可视为 vTensor 思路从 KV cache 扩展到 runtime memory/KV cache 统一池 | 系统更复杂，需要 SLO-aware scheduling 和 CPU memory buffer |
| KV compression/offloading 系列 | 通过量化、压缩、CPU/SSD offload 降低 KV cache 占用 | vTensor 主要解决 GPU 侧分配/映射和 kernel 解耦，可与压缩/offload 正交 | offload/压缩可能带来精度、PCIe/NVLink 传输或调度成本 |

## 第七部分 局限性

### 作者提到的局限

论文没有单独的 Limitations section，但从正文和实验可以看出作者承认或隐含以下边界：

1. VMM 操作本身有开销，因此设计必须把 virtual allocation 与 physical allocation 解耦，并通过异步调度隐藏开销。
2. FlexInfer 需要 runtime-level 调度配合；不是只替换一个 kernel 就能完整获得收益。
3. 系统收益依赖 workload 形态。prefix sharing、多轮对话、长上下文和高 batch 场景更容易体现优势。

### 作者没有充分展开但值得注意的局限

1. 没有官方开源代码。论文说基于 vLLM 改 3000 行并有 C++ vTensor manager，但当前未找到官方仓库，复现实验成本较高。
2. 与 vAttention 的直接比较缺失。vAttention 在 2024-05 已提出非常相似的 CUDA VMM 思路，并有 Microsoft 官方开源实现；vTensor 只在 Related Work 中称其为 concurrent work，没有端到端对照。
3. baseline 时效性有限。vLLM v0.4.2、FlashAttention 2.5.8 和当时的 SGLang kernel 到 2026 已不是最新状态，性能差距可能变化。
4. 硬件泛化不足。主要在 A100 80GB 上评测，CUDA VMM 的开销、驱动行为、multi-GPU mapping、H100/B200/不同 MIG 配置下表现未充分展开。
5. 生产指标不足。论文主要看 throughput 和 memory usage，缺少 p95/p99 latency、SLO violation、preemption fairness、multi-tenant isolation 等生产 serving 指标。
6. 数据负载偏合成。多轮 chat/prefix-sharing 主要用 SGLang synthetic 数据，单轮数据由 LV-Eval 截断构造，未覆盖真实线上 arrival process 和 prompt distribution。
7. 安全性和稳定性风险。把物理 chunks 共享给多个 virtual addresses 依赖 ref count 和 metadata 正确性；runtime bug 可能导致内存泄漏、错误复用或隔离问题。

### 未来仍未解决的问题

- 如何把 VMM-based KV cache 管理合入主流 serving runtime，而不破坏已有 scheduler、allocator 和 kernel ecosystem？
- 如何自动选择 chunk size、pre-extension 时机和 release 策略，以适配不同模型、上下文长度和请求率？
- VMM-based memory management 与 KV compression、CPU/NVMe offloading、disaggregated prefill/decode 如何组合？
- 如何在多租户场景下同时优化 throughput、tail latency、memory isolation 和成本？

## 第八部分 最新发展

截至日期：2026-07-02

| 方向/工作 | 最新状态 | 与本文关系 | 证据来源 |
| --- | --- | --- | --- |
| vTensor/FlexInfer | arXiv v1 提交于 2024-07-22，16 pages/12 figures；未找到官方代码 | 本文主体 | arXiv 页面 |
| vLLM 集成 | vLLM issue #6687 提到 vTensor，但状态为 closed as not planned，无 assignee/PR | 说明主流 runtime 未直接吸收该实现 | GitHub issue |
| vAttention | Microsoft 官方页面显示为 ASPLOS 2025，GitHub 开源；同样使用 CUDA VMM，目标是不用 PagedAttention 也能动态 KV cache 管理 | 最接近的并发/竞品路线，且开源程度更高 | Microsoft Research 与 GitHub |
| KV cache survey | 2024/2025 survey 将 vTensor/FlexInfer 归入 OS-inspired KV cache memory management 或 heterogeneous design | 说明该工作被后续综述纳入方向图谱 | arXiv/OpenReview survey |
| eLLM | 2025-06 arXiv，2026 DAC 页面显示 eLLM 扩展 virtual tensor 思想，统一 KV cache 与 activation 的弹性内存管理 | 可视作同一作者线的后续演进 | arXiv 与 DAC program |

工业/开源采用情况：

- OpenAI/Google/Anthropic/DeepSeek 是否采用 vTensor：未找到公开证据。
- vLLM：有社区 feature request，但 issue closed as not planned，未见直接集成证据。
- Microsoft：公开实现的是 vAttention，不是 vTensor/FlexInfer；但技术路线同样是 CUDA VMM + virtual/physical allocation 解耦。
- 主流开源 serving 系统仍以 vLLM/PagedAttention、FlashAttention/FlashInfer、prefix cache、KV compression/offload、disaggregated serving 等多路线并行演进。

Open Problems：

- VMM API 的跨硬件/跨驱动稳定性和开销建模。
- 与 PyTorch/CUDA allocator、vLLM block manager、FlashInfer kernel ecosystem 的统一接口。
- 多实例/多租户环境下的弹性 memory pool、隔离和 SLO-aware scheduling。
- Prefix cache、RAG cache、long-context KV eviction 与 VMM remapping 的组合优化。

## 第九部分 科研价值

vTensor 的长期价值在于把 LLM serving 的 KV cache 问题重新表述为一个系统抽象问题：不要把内存碎片整理塞进每个 attention kernel，而是给 kernel 一个稳定的连续 tensor 接口，把动态性放在虚拟内存层。这种抽象边界如果成立，就能显著降低 kernel 生态适配成本，也能让 serving runtime 更像操作系统一样管理 GPU memory。

如果继续写下一篇论文：

| 方向 | 具体问题 | 为什么有价值 | 难度/资源 | 可能做法 |
| --- | --- | --- | --- | --- |
| 与 vAttention 的系统对比 | 在同一 vLLM/Sarathi/SGLang runtime、同一模型/trace 上比较 vTensor 与 vAttention | 解决目前最关键的证据缺口 | 中等，需要实现或复现 vTensor | 用公开 vAttention 作为基线，复现 vTensor manager 的核心抽象 |
| VMM cost model | 建立 cuMemCreate/Map/Unmap 在不同 GPU/driver/chunk size 下的开销模型 | 决定何时 VMM 比 PagedAttention 更优 | 中等 | 做 microbenchmark + analytical model + runtime adaptive policy |
| Unified memory pool | 统一 KV cache、activation、temporary buffers，而不是只管 KV cache | eLLM 已沿这个方向推进，仍有大量系统空间 | 高 | memory ballooning、SLO-aware scheduler、PyTorch allocator interop |
| Production trace evaluation | 用真实或更接近真实的 arrival trace、prompt length、multi-turn pattern 评测 | 合成负载很难说明线上收益 | 中等到高 | 收集脱敏 trace 或构造公开 benchmark |
| Tail-latency-aware scheduling | 在 throughput 之外优化 p95/p99 latency 与 SLO violation | 工业部署比平均吞吐更重视尾延迟 | 中等 | 将 Extend/Release/Preempt 策略和 SLO-aware priority queue 结合 |
| VMM + KV compression | 把 virtual mapping 与 token/head/layer-level compression 结合 | 同时减少物理内存和提升复用弹性 | 高 | chunk-level compression metadata + on-demand decompress |
| Multi-GPU / disaggregated serving | prefill/decode 分离、多 GPU KV migration、remote KV pool | 当前 serving 趋势是 disaggregation 和长上下文 | 高 | NVLink/PCIe-aware remapping、KV ownership protocol |
| Safety/isolation verification | 验证 ref count、shared chunks、prefix cache release 的正确性 | 内存管理 bug 代价很高 | 中等 | 形式化状态机、stress test、fault injection |

最容易创新的切入点不是再提出一个抽象名词，而是补齐 vTensor 论文缺少的工程证据：开源实现、与 vAttention 的公平比较、真实 trace、tail latency 和跨硬件 VMM cost model。

## 第十部分 Roadmap

### 推荐阅读顺序

| Order | Paper/Topic | Why Read It |
| ---: | --- | --- |
| 1 | Transformer KV cache / autoregressive decoding 基础 | 理解为什么 decode 阶段 memory-bound |
| 2 | FlashAttention / FlashInfer | 理解高性能 attention kernel 为什么依赖规则访存和硬件利用 |
| 3 | Orca / continuous batching / Sarathi-Serve | 理解 serving scheduler 如何动态组 batch |
| 4 | vLLM / PagedAttention | 理解 KV cache 分页管理的标准基线 |
| 5 | Prefix caching / SGLang | 理解 shared prefix 场景为什么需要更复杂的 KV 生命周期管理 |
| 6 | GMLake | 理解 CUDA VMM 在深度学习 memory defragmentation 中的系统背景 |
| 7 | vTensor | 重点读 abstraction boundary、VTM/VTS、prefix tree 和实验 |
| 8 | vAttention | 与 vTensor 做横向对比，重点看开源实现和 ASPLOS 2025 版本 |
| 9 | eLLM | 看 vTensor 思想如何扩展到 KV cache + activation 的统一弹性管理 |

### 知识树

```text
LLM serving memory management
├── KV cache basics
│   ├── prefill / decode
│   ├── MHA / GQA / MQA
│   └── long-context memory pressure
├── Serving scheduler
│   ├── continuous batching
│   ├── prefill-decode disaggregation
│   └── SLO-aware scheduling
├── KV cache allocation
│   ├── static allocation
│   ├── PagedAttention / block table
│   ├── CUDA VMM / vTensor / vAttention
│   └── unified elastic memory / eLLM
├── KV reuse and reduction
│   ├── prefix caching
│   ├── KV compression
│   ├── KV eviction
│   └── CPU/SSD offloading
└── Evaluation
    ├── kernel latency
    ├── throughput
    ├── tail latency / SLO
    ├── memory footprint
    └── production trace robustness
```

### 时间线

| 时间 | 主题 | 推荐关注点 |
| --- | --- | --- |
| 2022-2023 | FlashAttention、continuous batching、vLLM/PagedAttention | 从 kernel 优化走向 serving memory virtualization |
| 2024-05 | vAttention | CUDA VMM 保持 KV cache virtual contiguity，开源实现 |
| 2024-07 | vTensor/FlexInfer | vTensor abstraction + CPU-GPU heterogeneous VMM scheduling |
| 2024-2025 | KV cache survey、offloading/compression/disaggregation | KV cache optimization 分化成多条系统路线 |
| 2025-2026 | eLLM / elastic memory | 从 KV cache 管理扩展到 activation + KV cache 统一弹性管理 |

### 后续论文/主题

- vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention
- eLLM: Elastic Memory Management Framework for Efficient LLM Serving
- Efficient Memory Management for Large Language Model Serving with PagedAttention
- GMLake: Efficient and Transparent GPU Memory Defragmentation for Large-scale DNN Training with Virtual Memory Stitching
- A Survey on Large Language Model Acceleration based on KV Cache Management
- Prefix caching / Prompt cache / SGLang 相关工作
- Disaggregated prefill-decode serving、KV offloading、KV compression、tail-latency-aware scheduling

## Sources

- 本地论文：`/home/undefined/Desktop/ai/papers/03_kv_cache_serving_inference/2025arXiv-vTensor Flexible Virtual Tensor Management for Efficient LLM Serving.pdf`
- 官方论文页：https://arxiv.org/abs/2407.15309
- arXiv HTML：https://arxiv.org/html/2407.15309v1
- vLLM issue：https://github.com/vllm-project/vllm/issues/6687
- Microsoft vAttention project：https://www.microsoft.com/en-us/research/publication/vattention-dynamic-memory-management-for-serving-llms-without-pagedattention/
- vAttention GitHub：https://github.com/microsoft/vattention
- KV cache survey：https://arxiv.org/html/2412.19442v2
- eLLM arXiv：https://arxiv.org/abs/2506.15155
- DAC 2026 eLLM page：https://63dac.conference-program.com/presentation/?id=RESEARCH1351&sess=sess109
