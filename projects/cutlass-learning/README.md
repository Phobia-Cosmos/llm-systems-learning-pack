# CUTLASS Learning Lab

这个目录是根据工作区根目录的
[`CUTLASS_direction_survey_and_learning_roadmap.md`](../../CUTLASS_direction_survey_and_learning_roadmap.md)
搭建的可运行练习场。它不是一个“大而全”的 CUTLASS 构建，而是只针对当前
RTX 4070 SUPER 编译 `sm_89`，先把 CUDA、GEMM、Tensor Core、CUTLASS 调用和
CuTe Layout 五条最重要的链路跑通。

当前状态：21 个自动测试全部通过；`compute-sanitizer` 的 memcheck/racecheck
通过；Tensor Core 二进制中已确认存在 `HMMA.16816.F32` 指令。

详细材料：

- [学习路线](LEARNING_PATH.md)
- [练习项目与扩展题](PROJECT_IDEAS.md)
- [本机验证记录](VALIDATION.md)

## 快速开始

```bash
cd /home/undefined/Desktop/ai/projects/cutlass-learning

# 检测 GPU、CUDA 和编译工具
./scripts/check_env.sh

# 固定/检查 CUTLASS v4.5.3，配置并编译轻量练习
./scripts/build.sh

# 跑自动测试和代表性样例
./scripts/run_all.sh
```

`build.sh` 会在缺少源码时浅克隆官方 CUTLASS v4.5.3。当前源码已经准备好，
固定 commit 为：

```text
4552152794e8bd3bcfd63cf9b44369e590420dba
```

CUTLASS 是 header-only 依赖；这批 C++ 练习不需要 Python 虚拟环境，也没有
修改工作区中的共享 Python 环境。

## 当前机器与正确路线

| 项目 | 当前值 | 对学习的影响 |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 4070 SUPER，56 SM | Ada 消费卡，单 GPU |
| Compute Capability | 8.9 | CMake/nvcc 目标固定为 `89` / `sm_89` |
| Driver | 580.159.03 | CUDA 13.0 可用 |
| CUDA Toolkit | 13.0.88 | 已真实编译 CUTLASS v4.5.3 |
| Host compiler | GCC/G++ 13.3 | CUDA 13 支持范围内 |
| Build tools | CMake 3.28.3，Ninja 1.11.1 | 满足 CUTLASS C++17 要求 |
| 分析工具 | Nsight Systems/Compute、Compute Sanitizer | Systems/sanitizer 可用；NCU 已完成一次性管理员采集，普通用户仍受限 |

这张卡适合学习：

- CUDA Core 与 Tensor Core GEMM；
- `mma.sync` / SASS 中的 HMMA；
- Ampere 风格的 `cp.async`、多 stage pipeline 和 CuTe；
- FP16、BF16、TF32，以及后续的 Ada FP8 实验；
- small-M、非整齐 shape、epilogue fusion 和 shape tuning。

这张卡不能实机验证：

- Hopper 的 TMA、WGMMA、warp-group specialization、`sm_90a`；
- Blackwell 的 TCGen05、tensor memory、`sm_100a` / `sm_120`；
- NVLink、多 GPU distributed GEMM。

路线图末尾“当前执行环境没有 `nvidia-smi`”是调研当时的限制，不符合现在的
机器状态；本项目已经现场检测并纠正为 `sm_89`。

## 已搭好的练习

| 目录/目标 | 你会练到什么 | 当前验收 |
| --- | --- | --- |
| `00_device_info/device_info` | 设备属性、kernel launch、CUDA 错误检查 | 设备和 kernel smoke test 通过 |
| `01_vector_add/vector_add` | block sweep、尾块、`float4`、alignment fallback、Event | 六种边界及 A/B/C 未对齐测试通过 |
| `01_vector_add/vector_add_advanced` | 任意 warp multiple、10 轮 median/p95、half2/int4、NVTX、曲线 | aligned/unaligned、CSV/PNG/SVG、Nsight Systems 通过 |
| `02_tiled_gemm/tiled_gemm` | naive GEMM、shared-memory tiling、同步、边界 | CPU reference 与两个 kernel 均通过 |
| `03_cutlass_sgemm/cutlass_sgemm` | CUTLASS `Arguments → can_implement → initialize → run` | FP32 与 cuBLAS 对齐 |
| `03_cutlass_sgemm/cutlass_tensorop` | FP16 输入、FP32 累加/输出、Tensor Core | 与 cuBLAS 对齐，SASS 有 HMMA |
| `04_cute_layout/cute_layout` | Shape、Stride、Layout、层次结构、coalesce | row/column/padded/hierarchical 映射可视化 |

所有 GEMM 都按 row-major 输入理解。测试故意包含不是 CTA/tile 整数倍的 shape；
Tensor Core 版本仍要求 `N` 和 `K` 是 8 个 half 元素的整数倍，这是 128-bit
向量访问的对齐约束。

## 最常用的实操命令

单独运行：

```bash
./build/bin/device_info
./build/bin/vector_add 16777219 100
./build/bin/vector_add_advanced --n 16777219 --rounds 10 --min-block 32 --max-block 1024 --step 32 --csv results/vector_add_advanced.csv
./build/bin/tiled_gemm 512 512 512 50
./build/bin/cutlass_sgemm 1024 1024 1024 100
./build/bin/cutlass_tensorop 1024 1024 1024 100
./build/bin/cute_layout
```

参数约定：

```text
vector_add       N iterations [A_offset B_offset C_offset block_size]
vector_add_advanced 使用 --help 查看命名参数
tiled_gemm       M N K iterations
cutlass_sgemm    M N K iterations
cutlass_tensorop M N K iterations
```

扫方阵、尾块和 LLM decode 风格 small-M shape：

```bash
./scripts/shape_sweep.sh
less results/shape_sweep.txt
```

检查内存越界和 shared-memory 竞争：

```bash
./scripts/sanitize.sh
```

单独完成 Vector Add 项目的边界、block sweep、大小 working set 与未对齐实验：

```bash
./scripts/run_vector_add_project.sh
./scripts/run_vector_add_advanced.sh
./scripts/profile_vector_add_advanced_nsys.sh
./scripts/inspect_vector_add_sass.sh
```

高级曲线结果默认写入 `results/vector_add_advanced/`。原始基础版 `main.cu` 保留，
高级实验独立放在 `advanced.cu`。

确认编译器真的生成 Tensor Core 指令：

```bash
./scripts/inspect_sass.sh
```

生成无需硬件 counter 权限的时间线：

```bash
./scripts/profile_nsys.sh
nsys stats profiles/cutlass_tensorop.nsys-rep
```

Nsight Compute 的硬件计数器当前被驱动限制为管理员访问；一次性管理员采集已
完成，报告位于 `profiles/vector_add_advanced/ncu/`，汇总位于
`results/vector_add_advanced/ncu/{summary,comparison}.csv`。复现命令：

```bash
./scripts/profile_ncu.sh
sudo ./scripts/profile_vector_add_advanced_ncu.sh  # 一次性管理员采集

# 或在可信个人开发机持久开放，重启后普通用户采集
sudo ./scripts/enable_nvidia_performance_counters.sh
sudo reboot
./scripts/profile_vector_add_advanced_ncu.sh
```

持久开放会让所有本地用户读取 GPU 计数器；可用
`sudo ./scripts/disable_nvidia_performance_counters.sh` 后重启恢复。

## CUTLASS Profiler 是可选的重构建

下面的脚本只生成两个明确命名的 kernel：一个 FP32 SIMT SGEMM 和一个 FP16
Tensor Core GEMM。它不会构建 `all kernels`。

```bash
./scripts/build_profiler.sh
./scripts/run_profiler.sh
```

即使 kernel 只选两个，官方 profiler 本体仍会编译很多 reference provider。
本机首次尝试在 10 分钟后只到 13/66 个非 unity 编译单元，因此它被放在第 5
周，而不是入门环境的阻塞步骤。脚本使用 unity build 和 `JOBS=1` 来控制内存；
首次构建仍可能需要十几分钟。轻量 `cutlass_sgemm` 和 `cutlass_tensorop` 已经完整
验证 CUTLASS 头文件、cuBLAS、Tensor Core 和 `sm_89` 工具链。

## 构建选项

默认构建目录是 `build/`。可用环境变量覆盖：

```bash
CUDA_ARCH=89 JOBS=2 BUILD_DIR=/tmp/cutlass-learning-build ./scripts/build.sh
```

Profiler 单独使用：

```bash
CUDA_ARCH=89 JOBS=1 \
PROFILER_BUILD_DIR=/tmp/cutlass-profiler-build \
./scripts/build_profiler.sh
```

不要在这台机器上使用 `CUTLASS_LIBRARY_KERNELS=all`，也不要把目标改成
`sm_90a`、`sm_100a` 或 `sm_120`。

## 怎样看性能数字

GEMM 每次大约执行 $2MNK$ 次浮点运算；代码按
$\mathrm{TFLOP/s}=2MNK/(t_{ms}\times 10^9)$ 报告吞吐。

这些可执行文件当前报告预热后的多次平均 CUDA Event 时间，适合学习和快速
回归，不应直接当论文数据。正式实验还要补：

- 多轮独立采样，报告 median 和 p95；
- 固定/记录 GPU clock、power、温度和桌面负载；
- 用大于 L2 的数据区分 HBM 与 cache；
- 同时记录寄存器、SMEM、occupancy、memory traffic 和 warp stalls；
- 覆盖真实模型的 shape 分布，而不是只跑 4096³。

`vector_add` 的 `effective_GB/s` 特意标成 cache-sensitive；小数组反复运行会被
L2 缓存放大，不能把它直接解释成 HBM 带宽。
