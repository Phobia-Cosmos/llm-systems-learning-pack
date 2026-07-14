# CUTLASS 方向综述与学习路线

> 核验日期：2026-07-13  
> 主题：NVIDIA CUTLASS、CuTe、GPU GEMM、Tensor Core 与 AI Kernel 性能工程  
> 适用对象：希望从 CUDA 入门，最终能开发或研究高性能 AI/HPC 算子的学习者

## 报告元数据

| 项目 | 内容 |
| --- | --- |
| 研究对象 | NVIDIA CUTLASS / CuTe 及其相关 GPU Kernel 方向 |
| 核心问题 | 如何把 GEMM、卷积、Attention、MoE 等计算高效映射到 NVIDIA GPU |
| 官方项目 | [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass) |
| 稳定版本口径 | 截至核验日，最高正式 Git tag 为 [v4.5.3](https://github.com/NVIDIA/cutlass/releases/tag/v4.5.3) |
| 开发版本口径 | 官方 main、version.h 与 latest 文档已写 4.6.0，但尚无 v4.6.0 tag，本文称其为 4.6 开发态 |
| 主要语言 | CUDA C++17 模板；CUTLASS 4.x 另提供 Python-native CuTe DSL |
| 最重要的先修 | CUDA 编程、GPU 存储/线程层次、GEMM、性能分析、基础 C++ 模板 |
| 典型出口 | GPU 算子工程、ML Systems、编译器/自动调优、LLM 推理训练优化、HPC 库 |

## 一页结论

CUTLASS 不是一种模型，也不是一篇论文。它是一套 NVIDIA 开源的高性能 CUDA 内核构建组件。它最擅长的不是“调用标准矩阵乘”，而是：

- 自定义 GEMM、batched/grouped GEMM、隐式 GEMM 卷积；
- 把 bias、activation、quantization、dequantization、reduction 等操作融合进 mainloop 或 epilogue；
- 使用 Tensor Core、异步复制、TMA、warp specialization、persistent scheduling 等硬件能力；
- 为 Attention、MoE、低精度计算、推理框架构建白盒且可定制的底层 kernel；
- 研究线程和数据布局、流水线、调度、自动调优及跨架构性能。

如果只需要标准 GEMM，优先用 cuBLAS/cuBLASLt。如果需要源码级控制、特殊数据类型/布局、深度融合或研究新调度，CUTLASS 才真正有价值。

推荐的总体学习顺序是：

~~~text
CUDA 基础
→ 手写 tiled GEMM
→ Tensor Core 与性能分析
→ CUTLASS Profiler / 现成 GEMM
→ CuTe Layout 与 Tensor
→ CUTLASS 3.x mainloop / epilogue / scheduler
→ CuTe DSL
→ Attention / MoE / 低精度 / 编译器研究
~~~

不要一开始就钻进大型 Hopper 或 Blackwell kernel。先建立“数据在哪里、谁搬、谁算、如何同步、瓶颈是什么”的可验证心智模型。

## 第一部分 研究背景

### 1. CUTLASS 是什么

CUTLASS 原名 CUDA Templates for Linear Algebra Subroutines。当前官方描述更宽泛：它是一组用于在 CUDA 中实现高性能矩阵乘及相关计算的抽象。

C++ 部分是 header-only 模板库。CUTLASS 4.x 在此基础上加入了 Python-native 的 CuTe DSL；Python 语法降低了 C++ 元编程负担，但它仍然是能显式控制 layout、thread/data hierarchy 和硬件 atom 的低层 DSL。

CUTLASS 围绕经典 GEMM：

$$
D = \alpha A B + \beta C
$$

一次 M×N×K GEMM 约有 2MNK 次浮点运算。高性能的关键通常不是公式，而是减少与隐藏数据搬运，并让 Tensor Core 持续得到可计算的数据。

### 2. 它解决的核心矛盾

GPU 的计算吞吐量增长很快，但数据需要经过多层存储与线程层次：

~~~text
HBM / Global Memory
        ↓
       L2
        ↓
Shared Memory / Tensor Memory
        ↓
Registers
        ↓
CUDA Core / Tensor Core
~~~

同时，计算任务还要映射到：

~~~text
Grid / Cluster
    ↓
CTA / Thread Block
    ↓
Warp Group / Warp
    ↓
Thread / MMA instruction
~~~

CUTLASS 把 tile、layout、copy、MMA、pipeline、scheduler、epilogue 拆成可组合组件，帮助程序员表达“哪组线程在何时搬哪块数据，并用哪条矩阵指令计算”。

### 3. 这个方向实际做什么

一个 GPU Kernel / ML Systems 工程师的典型工作流是：

1. 用 profiler 找到端到端瓶颈；
2. 明确计算量、数据量、精度和实际 shape 分布；
3. 设计分块、线程映射和数据布局；
4. 在 GMEM、L2、SMEM、寄存器之间建立流水；
5. 选择 CUDA Core、Tensor Core、TMA、异步拷贝和同步机制；
6. 融合相邻计算，降低中间张量读写和 launch 开销；
7. 验证数值正确性、边界 shape 和并发行为；
8. 与 cuBLAS/cuBLASLt、PyTorch、Triton 或现有 CUTLASS kernel 做公平基准；
9. 集成到 PyTorch、JAX、vLLM、SGLang 或其他系统中。

典型岗位包括 GPU Kernel Engineer、ML Compiler Engineer、Inference/Training Performance Engineer、HPC Library Engineer 和 ML Systems Researcher。

## 第二部分 历史发展

### 1. 方向时间线

| 时间 | 代表工作/阶段 | 核心思想 | 与 CUTLASS 的关系 |
| --- | --- | --- | --- |
| 2008 | Goto GEMM、Volkov GPU GEMM | 分层 blocking、寄存器复用、GPU GEMM 调优 | CUTLASS 层次分块的算法前史 |
| 2009 | Roofline | 用算术强度判断 memory-bound 或 compute-bound | 决定优化数据搬运还是算术流水 |
| 2017 | Volta Tensor Core；CUTLASS 首次开源 | 可编程矩阵乘指令与模板化线性代数组件 | CUTLASS 项目起点 |
| 2018 | Tensor Core 研究、TVM/AutoTVM、CUTLASS GTC talk | 硬件矩阵指令、张量编译、自动调优 | 硬件原语与编译路线开始分化 |
| 2019–2020 | CUTLASS 2.x、Triton、Fireiron、Ansor | 模板配置、tile DSL、schedule 与自动搜索 | 形成几条主流 kernel 开发路线 |
| 2022 | BOLT、FlashAttention | 硬件原生模板搜索；IO-aware 融合 | CUTLASS 用于自动模板生成和真实 AI 算子 |
| 2023 | CUTLASS 3.0 / CuTe；TensorIR；FA2 Hopper case study | layout algebra、TMA、WGMMA、warp specialization | 现代 CUTLASS 的核心形态 |
| 2024 | FlashAttention-2/3、ThunderKittens | 工作划分、异步流水、低精度、tile primitives | CUTLASS/CuTe 的重要应用与对照 |
| 2025 | CUTLASS 4.0 CuTe DSL、TileLang、Hexcute、DeepGEMM | Python DSL、layout 自动合成、低精度 LLM GEMM | 更低开发成本与更多自动化 |
| 2026 | CUTLASS 4.5.x；4.6 开发态 | Operator API、编译流水接口、细粒度 tracing 等 | 正在降低 kernel 发现、集成和调试成本 |

### 2. 一个重要的论文口径

没有一篇公认的“CUTLASS 原始同行评审论文”。

项目最初主要通过代码、NVIDIA 技术博客、设计文档和 GTC 演讲发布。最接近项目首发材料的是：

- Andrew Kerr 等人的 [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)（2017，技术博客，不是论文）；
- [CUTLASS: Software Primitives for Dense Linear Algebra at All Levels and Scales within CUDA](https://www.nvidia.com/en-us/on-demand/session/gtcsiliconvalley2018-s8854/)（GTC 2018 演讲，不是同行评审论文）；
- [官方仓库](https://github.com/NVIDIA/cutlass)与[官方文档](https://docs.nvidia.com/cutlass/)。

因此，下面的清单是 GEMM、Tensor Core、调度语言、编译器和下游 AI kernel 的论文谱系，不应把其中任一篇称为“the CUTLASS paper”。

### 3. 核心论文：优先读这些

| 顺序 | 论文 | 年份 / Venue | 为什么读 |
| ---: | --- | --- | --- |
| 1 | [Anatomy of High-Performance Matrix Multiplication](https://doi.org/10.1145/1356052.1356053) | 2008, ACM TOMS | 理解分层 blocking 和数据复用；虽以 CPU 为主，思想直接迁移到 GEMM |
| 2 | [Benchmarking GPUs to Tune Dense Linear Algebra](https://doi.org/10.1109/SC.2008.5214359) | 2008, SC | GPU GEMM、寄存器 blocking、延迟隐藏及“occupancy 不是越高越好” |
| 3 | [NVIDIA Tensor Core Programmability, Performance & Precision](https://arxiv.org/abs/1803.04014) | 2018, IPDPSW | Tensor Core、WMMA、CUTLASS、cuBLAS 与精度行为的早期系统研究 |
| 4 | [Demystifying Tensor Cores to Optimize Half-Precision Matrix Multiply](https://doi.org/10.1109/IPDPS47924.2020.00071) | 2020, IPDPS | 从微架构角度理解 Tensor Core GEMM 的瓶颈 |
| 5 | [Triton: an Intermediate Language and Compiler for Tiled Neural Network Computations](https://doi.org/10.1145/3315508.3329973) | 2019, MAPL | 最重要的对照路线：编译器 tile DSL 与 CUTLASS 低层控制的取舍 |
| 6 | [BOLT: Bridging the Gap between Auto-tuners and Hardware-native Performance](https://proceedings.mlsys.org/paper_files/paper/2022/hash/1f8053a67ec8e0b57455713cefdd8218-Abstract.html) | 2022, MLSys | 直接在 CUTLASS 模板空间中做硬件原生自动调优 |
| 7 | [FlashAttention](https://arxiv.org/abs/2205.14135) | 2022, NeurIPS | 把 IO-aware、tiling 和融合应用到真实 LLM kernel |
| 8 | [A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on Hopper using CUTLASS](https://arxiv.org/abs/2312.11918) | 2023, arXiv preprint | 最直接的现代 CUTLASS/CuTe、TMA、WGMMA、两次 GEMM 融合教材 |
| 9 | [FlashAttention-2](https://arxiv.org/abs/2307.08691) | 2024, ICLR | 学习 threadblock/warp 工作划分和非 matmul FLOPs 优化 |
| 10 | [FlashAttention-3](https://arxiv.org/abs/2407.08608) | 2024, NeurIPS | Hopper warp specialization、异步流水与 FP8 |

### 4. 按研究分支扩展阅读

| 分支 | 论文 | 年份 / Venue | 与 CUTLASS 的关系 |
| --- | --- | --- | --- |
| 微架构 | [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/abs/1804.06826) | 2018, arXiv | 理解 SM、cache、Tensor Core 和指令行为 |
| 直接 CUTLASS 实例 | [Implementing Strassen’s Algorithm with CUTLASS on NVIDIA Volta GPUs](https://arxiv.org/abs/1808.07984) | 2018, arXiv preprint | 展示如何扩展 CUTLASS；注意它不是 SC 正式论文 |
| 编译器 | [TVM](https://www.usenix.org/conference/osdi18/presentation/chen) | 2018, OSDI | vendor kernel 与端到端张量编译的取舍 |
| 自动调优 | [Learning to Optimize Tensor Programs](https://arxiv.org/abs/1805.08166) | 2018, NeurIPS | AutoTVM 的 cost model 与 schedule search |
| 调度 DSL | [Fireiron: A Data-Movement-Aware Scheduling Language for GPUs](https://doi.org/10.1145/3410463.3414632) | 2020, PACT | 显式表达数据移动；适合与 CUTLASS 模板分层对照 |
| 自动生成 | [Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng) | 2020, OSDI | 自动构造 schedule 搜索空间 |
| 搜索空间 | [Tensor Program Optimization with Probabilistic Programs](https://arxiv.org/abs/2205.13603) | 2022, NeurIPS | MetaSchedule 的模块化搜索思想 |
| 快速编译 | [ROLLER](https://www.usenix.org/conference/osdi22/presentation/zhu) | 2022, OSDI | 以硬件约束缩小 schedule 空间 |
| Tensor IR | [TensorIR](https://doi.org/10.1145/3575693.3576933) | 2023, ASPLOS | 把 tensorization 和硬件张量指令作为一等抽象 |
| 融合调度 | [Welder](https://www.usenix.org/conference/osdi23/presentation/shi) | 2023, OSDI | 用 tile graph 联合优化计算和内存访问 |
| Tile primitives | [ThunderKittens](https://arxiv.org/abs/2410.20399) | 2024, arXiv preprint | 另一套 CUDA C++ tile/warp primitives |
| Tiled DSL | [TileLang](https://arxiv.org/abs/2504.17577) | 2025, arXiv preprint | 更 Python 化的 tile 数据流与调度表达 |
| Layout 自动化 | [Hexcute](https://arxiv.org/abs/2504.16214) | 2025, arXiv preprint | 直接针对 CuTe 手工 layout 难题做自动合成 |
| LLM 系统 | [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | 2023, SOSP | 从单 kernel 扩展到动态 batching 和分页 KV cache |
| 低精度系统 | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 2024, technical report | 了解 FP8、MoE 和系统级效率需求；不是 CUTLASS 论文 |

## 第三部分 核心创新

### Novelty

CUTLASS 的长期技术特点不是发明 GEMM，而是把高性能 GEMM 的“moving parts”模块化，并让抽象贴近算法和硬件能力。

CUTLASS 3.x 的 GEMM API 有五层：

| 层 | 主要对象 | 作用 |
| --- | --- | --- |
| Device | GemmUniversalAdapter | 主机端初始化、参数检查、launch |
| Kernel | GemmUniversal | 网格级 tile 调度，组合 mainloop 与 epilogue |
| Collective | CollectiveMma / collective epilogue | 协作线程组的数据搬运、流水、同步和 MMA |
| Tiled operation | TiledMma / TiledCopy | 把 MMA 或 Copy atom 铺到线程和数据 tile 上 |
| Atom | Mma_Atom / Copy_Atom | 最小硬件矩阵乘或复制操作 |

CuTe 的中心抽象是：

- Layout = Shape + Stride：一个层次化的坐标到索引映射；
- Tensor = Engine/指针 + Layout：统一描述 GMEM、SMEM、寄存器或 tensor memory 中的数据；
- layout algebra：composition、tiling、partition、coalesce 等操作；
- atom/tiled operation：显式描述“哪些线程操作哪些数据”。

### Contributions

- 大量可组合的高性能 GEMM、卷积、Attention 和相关原语；
- 支持 FP64、FP32、TF32、FP16、BF16、FP8、窄整数、FP4/FP6 及 block-scaled 类型，具体依赖硬件；
- 暴露异步拷贝、TMA、MMA/WGMMA/TCGen05、cluster、persistent scheduler 等能力；
- CUTLASS Profiler、kernel generator、示例、单元测试及框架集成资源；
- 4.x 加入 CuTe DSL，让 Python 代码仍能表达低层 layout 和硬件控制。

### Core Insight

最简洁的核心思想是：

> 高性能张量计算的本质，是设计可复用的数据 tile，并让搬运、计算、同步和写回形成与硬件层次匹配的流水线。

## 第四部分 方法详解

### 1. 一个 GEMM kernel 中发生什么

| 模块 | Why | How | 为什么有效 | 常见失败 |
| --- | --- | --- | --- | --- |
| Problem tiling | 把大矩阵分给并行 CTA | 按 M/N/K 选择 CTA tile | 提供并行性与局部复用 | tile 不适合实际 shape，尾块浪费 |
| GMEM load | HBM 延迟高、带宽有限 | 向量化、合并访存、cp.async 或 TMA | 降低事务数并隐藏延迟 | 未对齐、非合并、错误边界处理 |
| SMEM layout | CTA 内共享 A/B tile | swizzle/layout algebra | 提高复用并避免 bank conflict | layout 与 copy/MMA 不匹配 |
| Pipeline | 搬运和计算需要重叠 | 多 stage、双缓冲、barrier | 隐藏内存延迟 | stage 太多导致 SMEM/寄存器压力 |
| MMA | 使用 Tensor Core | TiledMma + hardware atom | 获得高矩阵吞吐 | dtype、alignment、shape 不满足 |
| Scheduling | tile 数和 SM 数常不整齐 | raster、persistent、Stream-K、cluster | 提高负载均衡和 cache locality | 小 shape launch-bound；大 shape 尾波 |
| Epilogue | accumulator 要缩放并写回 | bias、activation、quantize、aux output fusion | 避免额外 kernel 和中间张量读写 | fusion 增加寄存器压力，反而变慢 |
| Verification | 低精度和异步代码易错 | reference、随机/边界 shape、sanitizer | 保证结果和同步正确 | 只测整齐方阵或只测一次 |

### 2. 真正需要调的参数

- 输入、输出和 accumulator dtype；
- A/B/C/D 的 layout、stride 和 alignment；
- CTA tile 的 M/N/K；
- warp 或 warp-group 的布局；
- MMA instruction / atom；
- pipeline stage 数；
- shared-memory layout 与 swizzle；
- cluster shape；
- kernel schedule、raster order、persistent 或 Stream-K；
- split-K、batched/grouped 策略；
- epilogue fusion；
- 实际 workload 的 shape 分布。

这些参数彼此制约。更大的 tile 会增加数据复用，也会增加寄存器和共享内存压力；更多 stage 能隐藏延迟，也可能降低 resident CTA 数；更高 occupancy 不一定意味着更高性能。

## 第五部分 实验

### 1. 正确的性能验证框架

每个学习项目至少报告：

| 项目 | 建议 |
| --- | --- |
| Correctness | 对比 PyTorch/cuBLAS 或高精度 reference；覆盖随机、零值、极值与尾块 |
| Shapes | 同时测试方阵、长瘦矩阵、小 M decode、非 tile 整数倍和真实模型 shape |
| Dtypes | 明确输入、累加、输出类型，以及是否允许 TF32 |
| Warm-up | 先预热，分开报告首次编译/JIT 和 steady-state |
| Timing | CUDA Event；显式同步；报告 median，必要时 p95 |
| Throughput | GEMM 用 2MNK / time 计算 TFLOP/s |
| Bandwidth | 对 memory-bound kernel 报有效带宽 |
| Baselines | cuBLAS/cuBLASLt、PyTorch、CUTLASS 默认 kernel、Triton 或目标框架 |
| Resources | registers/thread、SMEM/CTA、occupancy、编译时间、二进制大小 |
| End-to-end | 除 microbenchmark 外，报告真实模型或服务收益 |
| Reproducibility | GPU、SM、driver、CUDA、CUTLASS commit、clock/power、命令和 seed |

### 2. 工具分工

- [CUTLASS Profiler](https://docs.nvidia.com/cutlass/4.5.3/media/docs/cpp/profiler.html)：枚举、验证和比较 CUTLASS kernel；
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/)：观察 launch、CPU/GPU timeline、通信和 kernel 间空隙；
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/)：观察 Tensor Core 利用、吞吐、memory traffic、warp stall、bank conflict、寄存器和 occupancy；
- ptxas verbose、cuobjdump、nvdisasm：检查寄存器、spill、PTX/SASS；
- compute-sanitizer：检查越界、竞争和同步问题。

不要只看 occupancy。先用 Roofline 和 profiler 判断是算力、带宽、延迟、launch、负载不均还是资源压力。

## 第六部分 与已有工作的比较

| 工具 | 抽象层级 | 适合什么 | 优点 | 主要限制 |
| --- | --- | --- | --- | --- |
| cuBLAS/cuBLASLt | 高层预编译库 | 标准 GEMM、常见 epilogue | 使用最简单，通常是强基线 | 闭源，深度定制有限 |
| CUDA C++ / PTX | 最低层通用编程 | 新算法、非规则 kernel、极致控制 | 最灵活 | 开发和维护成本最高 |
| CUTLASS C++ | 低层可组合模板 | 自定义 GEMM、融合、架构特化 | 白盒、组件丰富、接近硬件 | C++ 模板复杂，编译慢，版本敏感 |
| CuTe C++ | 布局与原语层 | 自定义数据/线程映射 | layout algebra 表达力强 | 概念抽象，报错和调试较难 |
| CuTe DSL | 低层 Python DSL | 快速原型、现代 NVIDIA kernel | Python 入口但保留硬件控制 | 当前仍在快速演进，兼容面较窄 |
| Triton | 较高层 Python tile DSL | 快速开发常见 AI kernel | 易写、框架集成成熟 | 极端架构特化时控制力或编译器支持可能受限 |
| TVM / TensorIR | 编译器栈 | 跨平台 codegen、自动调优 | 自动化和可移植性强 | 系统复杂，峰值性能依赖搜索与后端 |
| TileLang | Python tiled DSL | 显式数据流 + 编译器调度 | 兼顾可写性和 tile 控制 | 较新，生态和稳定性仍发展中 |
| ThunderKittens | CUDA C++ tile primitives | 研究型 AI kernel | 代码紧凑、tile 抽象直观 | 生态和覆盖度小于 CUTLASS |
| DeepGEMM | 专用 JIT kernel 库 | Hopper/Blackwell 的 LLM GEMM、MoE | 代码聚焦真实低精度场景 | 硬件和 workload 范围更窄 |

选择原则：

- 只调用标准 GEMM：cuBLAS/cuBLASLt；
- 两三天内写出常见 fused op：先 Triton；
- 需要 NVIDIA 新硬件细节、定制 mainloop/epilogue：CUTLASS/CuTe；
- 研究自动调优或可移植编译：TVM/TensorIR/TileLang；
- 研究 LLM 生产 kernel：同时读 CUTLASS、FlashAttention、DeepGEMM、FlashInfer 和框架调用代码。

## 第七部分 局限性

### 官方或工程上明确可见的局限

- 主要面向 NVIDIA CUDA，不是跨厂商通用抽象；
- 模板实例化和全量 kernel 构建可能非常慢、占用大量内存和二进制空间；
- API、示例和硬件特性强相关，升级版本可能需要迁移；
- 性能高度依赖 shape、dtype、layout、alignment 和 GPU；
- 低精度与异步流水容易产生隐蔽的数值或同步错误；
- CUTLASS 3.x/4.x 的 Windows 支持仍有官方已知问题；
- CuTe DSL 当前仍处于快速演进阶段。

### 学习上的主要风险

- 一开始只背模板类型，不理解 CUDA 数据流；
- 只在 4096³ 方阵上测试，忽略真实模型的小 M、长瘦和 grouped shape；
- 把高 occupancy 当目标；
- 只和自己上一版比较，不和 cuBLAS/cuBLASLt 或成熟实现比较；
- 混用不同数学模式，例如把 TF32 与严格 FP32 当成同一基准；
- 不记录 CUTLASS commit、SM 架构和 CUDA 版本；
- 一次构建所有 kernel，浪费数小时甚至触发链接失败。

## 第八部分 最新发展

### 1. 版本状态

截至 2026-07-13：

- 最高正式 Git tag 是 v4.5.3；
- main 的 README、CHANGELOG、version.h 和 docs latest 已显示 4.6.0；
- 远端尚无 v4.6.0 tag，release URL 仍为 404；
- 因此，复现实验建议固定 v4.5.3；需要 4.6 开发功能时固定具体 commit；
- 4.0 起加入 CuTe DSL，但 C++ 2.x/3.x API 并未被它替代。

C++ 的官方总体最低口径是 Volta、C++17、CUDA 11.4；CuTe DSL 的工具链要求更窄且变化更快，应以所固定 tag/commit 对应的 versioned Quick Start 和 setup.sh 为准。

Hopper 的架构加速特性要使用 sm_90a；数据中心 Blackwell 的 sm_100a 与 RTX 50 系列的 sm_120 不是同一目标，不能把架构条件二进制混用。

### 2. 值得读源码的开源项目

| 项目 | 与 CUTLASS 的关系 | 推荐阶段 |
| --- | --- | --- |
| [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass) | 官方实现、文档、profiler、tests、C++ 和 DSL examples | 从第一天开始 |
| [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | CUDA 基础与设备能力示例 | CUDA 入门 |
| [GPU MODE lectures](https://github.com/gpu-mode/lectures) | Lecture 15/36/57/86/103 覆盖 CUTLASS、FA3、CuTe、CuTe DSL 与 layout algebra | 全程伴随 |
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | CUDA/CUTLASS/CuTe 生态；当前 README 还包含 CuTe DSL 的 FlashAttention-4 | 高级 |
| [Dao-AILab/quack](https://github.com/Dao-AILab/quack) | 直接用 CuTe DSL 写 RMSNorm、Softmax、GEMM 等 | CuTe DSL 阶段 |
| [DeepSeek-AI/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | 借鉴 CUTLASS/CuTe，依赖 CUTLASS 头文件，但刻意避免重模板；覆盖低精度 GEMM/MoE | 高级 |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | CUTLASS 后端和 CuTe DSL kernel，面向 LLM serving | 高级 |
| [xFormers](https://github.com/facebookresearch/xformers) | 包含 CUTLASS Attention 等后端 | 集成阶段 |
| [NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine) | NVIDIA 生产级低精度训练/推理组件，vendoring CUTLASS | 高级 |
| [PyTorch](https://github.com/pytorch/pytorch) | Inductor 中有 CUTLASS GEMM template 与 autotuning 集成 | 编译器阶段 |
| [Triton](https://github.com/triton-lang/triton) | 主要对照 DSL | CUDA 基础后 |
| [Apache TVM](https://github.com/apache/tvm) | 张量编译和自动调优主线 | 编译器方向 |
| [TileLang](https://github.com/tile-ai/tilelang) | TVM 上的 tile 编程模型 | 编译器方向 |
| [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) | 另一套 CUDA tile primitives | 对比研究 |

## 第九部分 科研价值

### 1. 三条主要研究路线

#### A. 手写高性能 kernel / 架构协同设计

研究问题包括：

- 小 M、长瘦矩阵和不规则 shape 的调度；
- grouped GEMM / MoE 负载不均；
- epilogue 或 mainloop fusion 的收益与寄存器压力；
- FP8、FP4、block scaling 的性能—精度协同；
- Hopper/Blackwell 的 TMA、warp specialization、persistent kernel；
- Attention、MLA、稀疏或动态 kernel。

这是最贴近 CUTLASS 的路线。

#### B. 编译器、DSL 与自动调优

研究问题包括：

- 从计算图或 Python 自动生成 CuTe layout；
- tile、stage、cluster、scheduler 的 cost model；
- 缩小模板搜索空间并降低编译时间；
- 从 Ampere 到 Hopper/Blackwell 的性能可移植性；
- 自动验证 layout、alignment、barrier 和 pipeline；
- CUTLASS template、CuTe DSL、Triton、TileLang 之间的 IR 映射。

Hexcute、BOLT、TensorIR、TileLang 是很好的起点。

#### C. LLM 训练/推理系统

研究问题包括：

- decode 小 batch / 小 M GEMM；
- MoE grouped GEMM 与 expert imbalance；
- paged KV cache、GQA/MLA、FlashAttention；
- 量化/反量化与 GEMM 融合；
- 多 GPU distributed GEMM 和通信—计算重叠；
- kernel 选择对端到端吞吐、延迟和显存的影响。

这条路线需要同时理解 kernel 和 serving/training workload。

### 2. 可做成项目或论文的题目

| 方向 | 具体问题 | 难度 | 最低可交付成果 |
| --- | --- | ---: | --- |
| Shape autotuning | 针对一组真实 LLM shape 搜索 tile/stage/schedule | 低—中 | 可复现 benchmark + cost 特征 + 强 baseline |
| Fused epilogue | GEMM + bias + SiLU/GELU/quantize | 中 | 正确性、流量分析、register ablation、端到端收益 |
| Small-M GEMM | decode 场景的窄矩阵调度 | 中 | 与 cuBLASLt/CUTLASS/Triton 的 shape sweep |
| Grouped GEMM | skewed MoE token 分布的负载均衡 | 中—高 | 分布建模、scheduler、真实 MoE trace |
| Layout synthesis | 自动生成或验证 CuTe layout | 高 | 类型/约束系统 + 多架构验证 |
| Low precision | FP8/FP4 scaling 粒度和融合策略 | 高 | 精度—性能联合评估 |
| Distributed GEMM | NVLink 上通信计算重叠 | 很高 | 多 GPU 原型、timeline 和扩展性实验 |

一篇可信的性能论文不能只展示一个大方阵上的峰值。至少需要真实 shape 分布、强 baseline、正确性、资源/瓶颈分析、关键 ablation 和端到端结果。

## 第十部分 Roadmap

下面按每周 8–12 小时设计。已有 CUDA 基础可压缩为 8–10 周；完全新手建议 14–18 周。

### 第 0 周：环境与基线

学习：

- Linux、CMake、基础 C++17、Python；
- 确认 GPU、compute capability、driver、CUDA；
- 固定 CUTLASS tag 或 commit。

动作：

~~~bash
nvidia-smi
nvcc --version
git clone --branch v4.5.3 --depth 1 https://github.com/NVIDIA/cutlass.git
~~~

构建时只指定自己的架构，只编译需要的 example/profiler subset，不要构建 all kernels。

验收：

- 记录 GPU/SM、driver、CUDA、compiler、CUTLASS commit；
- 能编译并运行一个 CUDA sample；
- 能解释 sm_80、sm_90a、sm_100a、sm_120 的区别。

### 第 1–2 周：CUDA 基础

必学：

- grid、block、warp、thread；
- global/shared/register/constant memory；
- coalescing、bank conflict、同步；
- CUDA Event、错误检查、occupancy 的含义。

练习：

1. vector add；
2. reduction；
3. tiled transpose；
4. naive SGEMM；
5. shared-memory tiled SGEMM。

验收：

- 每个 kernel 都有 CPU/PyTorch reference；
- 能解释为什么 transpose 会发生非合并访存或 bank conflict；
- 能用 Nsight Compute 指出一个真实瓶颈。

资料：

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [GPU MODE lectures](https://github.com/gpu-mode/lectures)

### 第 3 周：GEMM 性能方法论

学习：

- GEMM 的 2MNK FLOPs；
- arithmetic intensity 和 Roofline；
- CTA/warp/thread blocking；
- vectorized load、double buffering；
- registers、SMEM、occupancy 的权衡。

阅读：

- Goto 2008；
- Volkov & Demmel 2008；
- [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/4.5.3/media/docs/cpp/efficient_gemm.html)。

验收：

- 写一份 naive → tiled GEMM 的优化日志；
- 至少扫 10 个不同类型 shape；
- 报 TFLOP/s、带宽、registers、SMEM 和正确性。

### 第 4 周：Tensor Core 与数值

学习：

- FP16/BF16/TF32/FP8 输入与 FP32 accumulation；
- WMMA、mma.sync 的基本概念；
- 对齐、layout 和 tile 约束；
- 误差容忍、溢出和缩放。

阅读：

- Markidis 2018；
- Yan 等 2020；
- PTX ISA 中相应 MMA 章节。

验收：

- 运行一个 Tensor Core GEMM；
- 与 CUDA Core / cuBLAS 比较；
- 说明性能差异和数值差异来自哪里。

### 第 5 周：先用 CUTLASS，不急着改源码

学习：

- C++ Quick Start；
- CUTLASS Profiler；
- basic GEMM example；
- dtype、layout、alignment、tile、stage、epilogue 参数。

动作：

- 构建自己的单一 SM 架构；
- 用 profiler 搜索一个小 kernel 集合；
- 与 cuBLAS 做验证；
- 对真实和非整齐 shape 做 sweep。

验收：

- 能从 profiler kernel 名和参数还原主要配置；
- 能解释一个 kernel 为什么在某 shape 快、换 shape 后变慢。

### 第 6–7 周：CUTLASS 3.x 组合层

学习：

- Device → Kernel → Collective → TiledMma/Copy → Atom；
- CollectiveBuilder、mainloop、epilogue；
- StageCountAuto、KernelScheduleAuto；
- persistent kernel、Stream-K、grouped GEMM 的概念。

项目：

- 从现有 GEMM 改成 fused bias + activation；
- 比较单独三个 kernel 与融合 kernel；
- 测量中间内存流量、launch 数、register pressure。

验收：

- 正确处理 beta 非零、尾块、不同 alignment；
- 端到端快于未融合路径，而不只 kernel 时间好看。

### 第 8–9 周：CuTe Layout Algebra

按顺序学习：

1. IntTuple / Shape / Stride；
2. Layout：坐标到索引；
3. composition、coalesce、logical divide/product；
4. Tensor；
5. local_tile、local_partition；
6. Copy Atom / TiledCopy；
7. MMA Atom / TiledMma。

练习原则：

- 先在纸上或 host 端打印 1D/2D layout；
- 再解释 thread-to-data mapping；
- 最后进入完整 GEMM。

验收：

- 给定一个 Shape + Stride，能手算坐标映射；
- 能画出每个 thread 负责的数据；
- 能解释某个 SMEM swizzle 如何避免 bank conflict。

### 第 10 周：CuTe GEMM 与架构特性

Ampere 路线：

- cp.async；
- mma.sync；
- 多 stage pipeline。

Hopper 路线：

- TMA；
- WGMMA；
- warp-group specialization；
- mbarrier、cluster、persistent schedule。

Blackwell 路线：

- 先区分 SM100 与 SM120；
- 再学习 TCGen05、tensor memory、block scaling。

只有对应硬件时才做高级架构实验；否则先读代码和论文，不要假装完成性能验证。

### 第 11 周：CuTe DSL

建议：

- 使用与 tag/commit 对应的官方 setup.sh；
- 先跑官方 notebook 和 Ampere GEMM；
- 再看 QuACK 的 memory-bound kernel；
- 不要因为是 Python 就把它当成高层 Triton。

项目：

- 写 softmax 或 RMSNorm；
- 再写 GEMM + 简单 epilogue；
- 对比 PyTorch、Triton 和 C++ CUTLASS。

验收：

- 分开报告 JIT 编译时间、首次调用和 steady-state；
- 能查看生成的 PTX/SASS；
- 能说明 layout、copy、MMA 和 pipeline 分别在哪里定义。

### 第 12–14 周：选一个方向做完整项目

三个推荐项目：

1. **工程型**：GEMM + bias + SiLU/quantize 融合；
2. **LLM 型**：small-M 或 MoE grouped GEMM；
3. **研究型**：shape-aware autotuner 或 CuTe layout verifier。

完整交付物：

- 问题定义；
- strong baselines；
- correctness tests；
- shape/dtype sweep；
- Nsight 报告；
- ablation；
- PyTorch/JAX 接口；
- 可复现脚本；
- 一篇 5–10 页技术报告。

### 最小阅读集合

如果时间很少，只读：

1. CUDA Programming Guide 的执行/内存模型；
2. Efficient GEMM in CUDA；
3. Volkov & Demmel 2008；
4. Markidis 2018；
5. CUTLASS 3.x Design 与 GEMM API；
6. CuTe 00–03 文档；
7. Triton 2019；
8. BOLT 2022；
9. FlashAttention 1/2；
10. CUTLASS Hopper case study；
11. FlashAttention-3；
12. TileLang 或 Hexcute 二选一。

### 你的工作区已有材料

你已经有不少可直接接入这条路线的本地材料：

- [CUTLASS 2.x & 3.x Intro 学习笔记](</home/undefined/Desktop/ai/projects/how-to-optim-algorithm-in-cuda/cutlass/CUTLASS 2.x & CUTLASS 3.x Intro 学习笔记.md>)：适合第 5–7 周，但其中“非 Hopper 优先 2.x”等建议具有历史版本背景，应用到 2026 年环境时需结合 3.x/4.x 官方文档；
- [CUTLASS 笔记系列—杨远航](</home/undefined/Desktop/ai/projects/how-to-optim-algorithm-in-cuda/cutlass/cute/CUTLASS笔记系列-杨远航.md>)：适合第 8–10 周；
- [CUTLASS TMA 教程](</home/undefined/Desktop/ai/projects/how-to-optim-algorithm-in-cuda/cutlass/tma/CUTLASS Tutorial: Mastering the NVIDIA® Tensor Memory Accelerator (TMA).md>)：适合有 Hopper 后的第 10 周；
- [LeetCUDA](/home/undefined/Desktop/ai/projects/LeetCUDA)：适合 CUDA/PTX 基础练习；
- [vLLM CUTLASS 扩展](/home/undefined/Desktop/ai/projects/vllm/csrc/cutlass_extensions)：适合第 12–14 周观察生产集成；
- [SGLang CUTLASS kernel](/home/undefined/Desktop/ai/projects/sglang/sgl-kernel/csrc/cutlass_extensions)：适合第 12–14 周做框架对照。

当前执行环境没有 nvidia-smi，因此本报告无法替你确认实际训练机器的 GPU/SM。开始实践前，应在真正运行 kernel 的机器上完成第 0 周的环境记录。

## Sources

### 官方 CUTLASS

- [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass)
- [v4.5.3 文档](https://docs.nvidia.com/cutlass/4.5.3/)
- [CUTLASS 3.x Design](https://docs.nvidia.com/cutlass/4.5.3/media/docs/cpp/cutlass_3x_design.html)
- [CUTLASS 3.x GEMM API](https://docs.nvidia.com/cutlass/4.5.3/media/docs/cpp/gemm_api_3x.html)
- [CuTe C++ Quick Start](https://docs.nvidia.com/cutlass/4.5.3/media/docs/cpp/cute/00_quickstart.html)
- [C++ Quick Start](https://docs.nvidia.com/cutlass/4.5.3/media/docs/cpp/quickstart.html)
- [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/4.5.3/media/docs/cpp/efficient_gemm.html)
- [CUTLASS Profiler](https://docs.nvidia.com/cutlass/4.5.3/media/docs/cpp/profiler.html)
- [官方 examples](https://github.com/NVIDIA/cutlass/tree/v4.5.3/examples)
- [CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)

### 官方 CUDA 与分析工具

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/)
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

### 证据说明

- 版本、兼容性和 API 以 NVIDIA 官方仓库和版本化文档为准；
- 论文 venue 以 DOI、正式 proceedings、USENIX/MLSys/ICLR/NeurIPS 页面或 arXiv 元数据核验；
- 标为 arXiv preprint/technical report 的工作没有被写成同行评审正式论文；
- “学习顺序、方向选择和研究题目”是基于上述资料给出的分析建议，不是 NVIDIA 官方结论。
