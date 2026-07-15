# 本机验证记录

验证日期：2026-07-15。下面是一次环境/正确性快照，不是锁频后的正式论文基准。

## 环境

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 SUPER，CC 8.9，56 SM |
| Driver | 580.159.03 |
| CUDA | Toolkit 13.0.88 |
| cuBLAS | CUDA 13 安装自带；项目链接成功 |
| Compiler | GCC/G++ 13.3.0 |
| CMake / Ninja | 3.28.3 / 1.11.1 |
| CUTLASS | v4.5.3，commit `4552152794e8bd3bcfd63cf9b44369e590420dba` |
| 编译目标 | 只编 `sm_89` |

CUTLASS v4.5.3 的部分 README 兼容性文字仍主要列 CUDA 12.x，但 changelog 已有
CUDA 13 相关改动。本机实际完成了 CUDA 13 + v4.5.3 的配置、编译、运行和
Tensor Core 验证，因此当前组合可用。

## 自动测试

命令：

```bash
ctest --test-dir build --output-on-failure
```

结果：

```text
device_info                 PASS
vector_add_n1/n3/n255       PASS
vector_add_n256/n257        PASS
vector_add_n1000003_sweep   PASS
vector_add_unaligned_a/b/c  PASS
vector_add_unaligned_all    PASS
vector_add_advanced_aligned PASS
vector_add_advanced_small_n PASS
vector_add_advanced_tail2/tail3 PASS
vector_add_advanced_unaligned PASS
vector_add_advanced_reject_nonwarp PASS (expected rejection)
tiled_gemm_tail             PASS
cutlass_sgemm_tail          PASS
cutlass_tensorop_tail       PASS
cute_layout_mapping         PASS

100% tests passed, 0 tests failed out of 21
```

其中 16 个 Vector Add 测试覆盖 `N=1,3,255,256,257,258,259,1,000,003`、基础
block sizes、advanced small-N fallback、全部 pack 余数、非 2 次幂 warp
multiples、block=1024、A/B/C 未对齐，以及非法 block=33 的预期拒绝。其余测试
覆盖三个 GEMM 维度的尾块和满足 128-bit 输入对齐、但不是 CTA tile 整数倍的
Tensor Core shape。

## Sanitizer

命令：

```bash
./scripts/sanitize.sh
```

结果：

```text
Vector Add aligned float4 + scalar tail: 0 errors
Vector Add unaligned scalar fallback:    0 errors
Vector Add advanced float4/half2/int4:   0 errors
Vector Add advanced alignment fallback:  0 errors
Tiled GEMM memcheck:                     0 errors
Tiled GEMM racecheck:                    0 hazards, 0 errors, 0 warnings
```

## 一次性能快照

以下是预热后的平均 CUDA Event 时间。桌面 GPU 未锁频、未固定 power，数字会随
温度和后台图形负载波动。

| 实验 | Shape | CUTLASS/自写 | cuBLAS/对照 | 正确性 |
| --- | --- | ---: | ---: | --- |
| Vector Add scalar | N=16,777,219 | 最好约 0.455 ms，443 effective GB/s | CPU ref | guard/结果通过 |
| Vector Add float4 | N=16,777,219 | 最好约 0.457 ms，440 effective GB/s | scalar/CPU ref | guard/结果通过 |
| Advanced float scalar / float4 | N=16,777,219，10 rounds | 453 / 448 median effective GB/s | typed CPU ref | 通过 |
| Advanced half scalar / half2 | N=16,777,219，10 rounds | 456 / 455 median effective GB/s | typed CPU ref | 通过 |
| Advanced int scalar / CUDA int4 | N=16,777,219，10 rounds | 454 / 447 median effective GB/s | typed CPU ref | 通过 |
| Naive SGEMM | 512³ | 0.130 ms，2.06 TFLOP/s | CPU ref | 通过 |
| Tiled SGEMM | 512³ | 0.116 ms，2.31 TFLOP/s | CPU ref | 通过 |
| CUTLASS FP32 SIMT | 1024³ | 0.159–0.183 ms，11.7–13.5 TFLOP/s | 0.122–0.155 ms，13.8–17.6 TFLOP/s | 通过 |
| CUTLASS FP16 TensorOp | 1024³ | 0.056–0.060 ms，35.5–38.4 TFLOP/s | 0.044–0.047 ms，45.8–48.6 TFLOP/s | 通过 |
| CUTLASS FP16 TensorOp | 520×504×264 | 0.0097–0.0105 ms，13.1–14.3 TFLOP/s | 0.0066–0.0073 ms，19.1–20.9 TFLOP/s | 通过 |

Small-M sweep 的代表结果：

| Kernel | Shape | CUTLASS | cuBLAS |
| --- | --- | ---: | ---: |
| FP32 SIMT | 32×4096×4096 | 3.32 TFLOP/s | 6.23 TFLOP/s |
| FP16 TensorOp | 32×4096×4096 | 8.83 TFLOP/s | 19.34 TFLOP/s |

它直观展示了“同一 kernel 在方阵快，不代表 decode small-M 也高效”。完整输出由
`./scripts/shape_sweep.sh` 生成到 `results/shape_sweep.txt`。

Vector Add 数据总量大于 L2，但反复访问仍受缓存、写策略和计时方式影响，所以
只称为 effective bandwidth，不称为 HBM 实测峰值。大 working set 上 `float4`
没有稳定超过合并访存良好的 scalar kernel，这也是有效的实验结论；小 working
set 的一次 block=64 快照曾达到约 1.53x，但主要受 L2 和 launch/调度影响。

Vector Add 的 SASS 已通过 `inspect_vector_add_sass.sh` 确认包含：

```text
LDG.E.128
STG.E.128
```

因此手写 `float4` 确实落成 128-bit global-memory 指令。扩展后的同一检查脚本还
确认 advanced `float4`/CUDA `int4` 主体都有 `LDG.E.128`/`STG.E.128`，`half2`
主体则有 32-bit packed load/store 和 `HADD2`；尾部仍按元素宽度使用标量指令。

Advanced 原始样本和曲线位于 `results/vector_add_advanced/`。每行 CSV 保存 10 个
round samples、type-7 median/p95、实际 alignment path 和 correctness。PNG/SVG
曲线覆盖 block=32..1024 的全部 warp multiples，并用阴影表示 median 到
p95-latency-derived bandwidth。

## Tensor Core 证据

命令：

```bash
./scripts/inspect_sass.sh
```

`cuobjdump` 在 `cutlass_tensorop` 中找到了多条：

```text
HMMA.16816.F32
```

这证明 `Sm80` 的 `16×8×16` MMA 模板在 `sm_89` 二进制中落成了 Ada 可执行的
Tensor Core 指令，而不是仅凭源码类型名推断。

## 分析工具

Nsight Systems 已成功生成：

```text
profiles/cutlass_tensorop.nsys-rep
profiles/vector_add_advanced/nvtx_timeline.nsys-rep
```

一次 summary 中可区分 CUTLASS kernel 与 cuBLAS 的 Ampere FP16 kernel，说明
CUDA timeline 路径可用。

Advanced timeline 的 `nvtx_gpu_proj_sum` 已识别
`dtype/variant/path/block/round` ranges，并分别统计六个 scalar/packed kernel，
说明 NVTX 标注与 Systems 脚本可用。

Nsight Compute CLI 已安装，当前普通用户运行仍得到：

```text
ERR_NVGPUCTRPERM
```

原因是 `/proc/driver/nvidia/params` 中 `RmProfilingAdminOnly: 1`。本轮经用户授权，
使用 `sudo profile_vector_add_advanced_ncu.sh` 完成了一次性管理员采集；没有写入
modprobe 配置，也没有向其他普通用户永久开放 counter。六份报告和两张数据表位于：

```text
profiles/vector_add_advanced/ncu/{float,half,int}_{scalar,vector}.ncu-rep
results/vector_add_advanced/ncu/summary.csv
results/vector_add_advanced/ncu/comparison.csv
```

`N=16,777,219`、block=256、单个被 NCU 捕获的 kernel 结果如下。load/store 是
warp-level SASS 指令执行数；L2/DRAM 是硬件计数器报告的实际流量率，不是逻辑
`3*N*sizeof(T)` 人工换算值。

| dtype | global load inst S→V | global store inst S→V | L2 GB/s S/V | DRAM GB/s S/V | DRAM peak % S/V | GPU time µs S/V |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 / `float4` | 1,048,578 → 262,146 | 524,289 → 131,073 | 432.2 / 436.0 | 441.9 / 436.4 | 89.89 / 88.78 | 466.1 / 465.4 |
| FP16 / `half2` | 1,048,578 → 524,292 | 524,289 → 262,146 | 510.5 / 499.4 | 460.5 / 408.8 | 93.68 / 83.18 | 222.6 / 228.1 |
| INT32 / CUDA `int4` | 1,048,578 → 262,146 | 524,289 → 131,073 | 435.4 / 441.9 | 445.0 / 443.5 | 90.52 / 90.23 | 462.7 / 455.9 |

因此 `float4/int4` 确实把 global load/store 指令减少约 75%，`half2` 减少约 50%，
但 kernel 时间分别只约 -0.1%、+2.5%、-1.5%。物理 DRAM byte 还会受压缩、写回
和事务粒度影响，不能拿它反推逻辑元素数。NCU 会 replay kernel，程序在 profiler
内打印的 CUDA Event 带宽不具代表性；稳定性能结论仍以独立 10-round 曲线为准。

## 官方 CUTLASS Profiler 构建状态

Profiler 的 kernel filter 已核对，只生成：

```text
cutlass_simt_sgemm_128x128_8x2_nn_align1
cutlass_tensorop_s1688gemm_f16_256x128_32x2_nt_align8
```

对应两个 kernel 动态库曾成功生成。但官方 profiler 本体还固定编译大量 reference
provider；非 unity、`JOBS=1` 的首次尝试在 10 分钟时为 13/66，因资源/时间成本
主动停止。它不是入门构建的完成条件。

保留的 `build_profiler.sh` 改为 unity + `JOBS=1`，可在第 5 周按需续建。主项目、
cuBLAS 对照、Tensor Core、SASS、sanitizer 和 Nsight Systems 均已独立验证完成。
