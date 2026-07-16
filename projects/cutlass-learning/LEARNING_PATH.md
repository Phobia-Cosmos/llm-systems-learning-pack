# 针对 RTX 4070 SUPER 的 CUTLASS 学习路线

这份安排从原路线图的 14–18 周方案收敛而来。完全新手建议每周 8–12 小时、
用 12–14 周；已有 CUDA 基础可以压缩到 8–10 周。

## 始终使用同一个闭环

每个 kernel 都按下面六步做，不要只追一个更高的数字：

1. 写下假设：慢在哪里，准备改什么。
2. 先做 reference 和尾块测试。
3. 只改一个变量，例如 tile 或 stage。
4. 预热后用 CUDA Event 计时。
5. 用 sanitizer / Nsight / SASS 找证据。
6. 记录 GPU、版本、shape、dtype、误差和结果。

一个最小实验记录建议包含：

```text
日期：
GPU / SM / driver / CUDA / CUTLASS commit：
问题 shape、dtype、layout、alpha、beta：
改动与假设：
reference 与容差：
warm-up / iterations / 统计量：
时间、TFLOP/s 或有效带宽：
registers / SMEM / profiler 证据：
结论与下一个实验：
```

## 第 0 周：环境与基线（已经完成）

学习：Linux 命令、CMake/Ninja、基础 C++17；能解释 compute capability 和
`sm_89` 的关系。

实操：

```bash
./scripts/check_env.sh
./scripts/build.sh
./build/bin/device_info
ctest --test-dir build --output-on-failure
```

验收：知道当前是 Ada/SM89；知道为什么不能运行 Hopper 的 TMA/WGMMA；知道
实验为何固定在已验证的 v4.5.3，而不是在同一 target 混用 vLLM 的 4.4.2 或
当前已正式发布的 v4.6.0。升级应单独进行并重跑全部验证。

## 第 1–2 周：CUDA 执行与内存模型

学习：grid、block、warp、thread；global/shared/register memory；合并访存、
同步、尾块、CUDA Event。

先读：

- CUDA Programming Guide 的 Programming Model 与 Memory Hierarchy；
- CUDA Best Practices 的 coalesced access 和 shared memory；
- [`01_vector_add/main.cu`](01_vector_add/main.cu)。

项目 1 已完成，可先一键复现：

```bash
./scripts/run_vector_add_project.sh
./scripts/inspect_vector_add_sass.sh
```

然后按以下顺序理解和扩展：

1. 把 block size 改成 64、128、256、512，记录吞吐和 grid 数。
2. 测 `N=1`、`255`、`256`、`257`、`1,000,003`。
3. 写 `float4` 向量化版本，同时保留尾部标量处理。
4. 新写 reduction 和 tiled transpose；transpose 分别做 naive、coalesced、
   shared-memory padding 三版。
5. 每版都运行 `compute-sanitizer`。

验收：能用线程索引手算某个元素由谁处理；能解释为什么小数组的有效带宽可能
超过显存标称带宽；能识别非合并访存和 bank conflict。

进入手写 CUDA 前，先运行 [`05_python_operators`](05_python_operators/README.md) 建立 PyTorch reference、Python 教学公式、`torch.compile` 和 Triton 的共同 correctness/benchmark 口径。这样后续每个 C++/CUDA kernel 都有同输入、同误差、同计时协议的 baseline，不会把 Python 循环、PyTorch C++ backend 和 GPU kernel 混成同一个比较对象。

```bash
./scripts/run_python_operator_project.sh
```

## 第 3 周：从 naive GEMM 到 tiled GEMM

学习：CTA tile、算术强度、数据复用、Roofline、共享内存、同步与资源权衡。

代码：[`02_tiled_gemm/main.cu`](02_tiled_gemm/main.cu)。

实操：

```bash
./build/bin/tiled_gemm 129 131 127 20
./build/bin/tiled_gemm 512 512 512 50
./scripts/sanitize.sh
```

接着把模板 tile 从 16 分别改为 8 和 32。每次只改 tile，比较：

- 正确性与尾块；
- CUDA Event 时间；
- shared memory/CTA；
- 是否因 thread 数、寄存器或 occupancy 反而变慢。

阅读原路线图列出的 Volkov & Demmel 2008，以及 CUTLASS 官方
`Efficient GEMM in CUDA`。

验收：能画出一个 CTA 如何把 A/B tile 搬进 shared memory；能解释两次
`__syncthreads()` 分别保护什么；不再把高 occupancy 当作唯一目标。

## 第 4 周：Tensor Core 与数值

学习：FP16/BF16/TF32 输入、FP32 accumulation；对齐约束；`mma.sync`；误差
容忍、缩放与溢出。

代码：

- [`03_cutlass_sgemm/main.cu`](03_cutlass_sgemm/main.cu)：FP32 CUDA Core；
- [`03_cutlass_sgemm/tensorop.cu`](03_cutlass_sgemm/tensorop.cu)：FP16 Tensor Core。

实操：

```bash
./build/bin/cutlass_sgemm 1024 1024 1024 100
./build/bin/cutlass_tensorop 1024 1024 1024 100
./scripts/inspect_sass.sh
```

然后尝试不满足对齐的 `N/K`，观察 `can_implement`/前置检查如何失败。把输入
范围逐步扩大，记录 FP16 误差和溢出行为。

验收：能区分输入 dtype、accumulator dtype 与输出 dtype；能在 SASS 中指出
HMMA；能解释 Tensor Core 更快不等于任何 shape 都更快。

## 第 5 周：先会用 CUTLASS 与 Profiler

学习 `device::Gemm` 的生命周期：

```text
Arguments
→ can_implement
→ get_workspace_size
→ initialize
→ run
```

先运行轻量目标和 shape sweep：

```bash
./scripts/shape_sweep.sh
```

再在时间充足时构建官方 Profiler 子集：

```bash
./scripts/build_profiler.sh
./scripts/run_profiler.sh
```

验收：能从 dtype、layout、alignment、CTA/warp/instruction tile 解释一个 kernel；
能说明同一 kernel 在 1024³ 和 `M=32` 时为什么差异很大。

## 第 6–7 周：mainloop、epilogue 与融合

学习 CUTLASS 五层：

```text
Device → Kernel → Collective → TiledMma/TiledCopy → Atom
```

第一个完整工程建议做 `GEMM + bias + ReLU`。以
`third_party/cutlass/examples/12_gemm_bias_relu/` 为参考，比较：

- GEMM 后再启动 bias/ReLU kernel；
- 使用 `LinearCombinationRelu` 的 fused epilogue。

验收不只是 kernel 时间：还要报告 launch 数、中间张量流量、寄存器压力、
尾块、非零 beta 和端到端时间。

## 第 8–9 周：CuTe Layout Algebra

先运行：

```bash
./build/bin/cute_layout
```

学习顺序：

1. IntTuple / Shape / Stride；
2. Layout 的坐标到索引映射；
3. hierarchical layout 与 `coalesce`；
4. composition、logical divide/product；
5. Tensor、`local_tile`、`local_partition`；
6. Copy Atom / TiledCopy；
7. MMA Atom / TiledMma。

先在 host 打印、手算和画图，再进入完整 GEMM。当前程序能直接验证
row-major `(2,3) → 19`、padding 与层次布局。

本地材料：

- [`CUTLASS 笔记系列—杨远航`](../how-to-optim-algorithm-in-cuda/cutlass/cute/CUTLASS笔记系列-杨远航.md)
- [`cutlass-notes`](../how-to-optim-algorithm-in-cuda/cutlass/code/cutlass-notes/README.md)

验收：给一个 Shape+Stride 能手算映射；能画 thread-to-data mapping；能解释
shared-memory swizzle 为什么能减少 bank conflict。

## 第 10 周：只学本机能验证的架构特性

RTX 4070 SUPER 路线：`cp.async`、`mma.sync`、多 stage pipeline、Ada Tensor
Core。可以阅读 Hopper/Blackwell 代码，但实验报告必须明确写“未在对应硬件
验证”。

不要在当前机器编译/运行 `cutlass-notes` 的 11–14 章 TMA/WGMMA 示例。

## 第 11 周：CuTe DSL（可选、独立环境）

工作区的共享 vLLM 环境已有 `nvidia-cutlass-dsl 4.5.2`，可以只读 import
探测，但它属于 vLLM，不能为了本课程升级或修改。

```bash
/home/undefined/Disk/python-envs/vllm/bin/python -c \
  'from cutlass import cute; print(cute.__file__)'
```

注意正确入口是 `from cutlass import cute`，不是顶层 `import cute`；而且该共享
DSL 是 4.5.2，本项目 C++ 头文件是 4.5.3，所以这里只做可用性探测，不把两个
小版本混成同一套可复现实验环境。

真正进入 DSL 时，按照固定 CUTLASS tag 对应的官方 `setup.sh`，在
`/home/undefined/Disk/python-envs/` 新建专用环境。分别报告 JIT 编译、首次调用
和 steady-state 时间。先做 Softmax/RMSNorm，再做 GEMM + 简单 epilogue。

## 第 12–14 周：做一个完整项目

按兴趣选一条：

- 工程：GEMM + bias + SiLU/GELU/quantize fusion；
- LLM：small-M 或 MoE grouped GEMM；
- 研究：shape-aware autotuner 或 CuTe layout verifier。

最低交付：问题定义、强 baseline、correctness、shape/dtype sweep、Nsight 证据、
ablation、复现脚本和技术报告。具体拆解见 [PROJECT_IDEAS.md](PROJECT_IDEAS.md)。
