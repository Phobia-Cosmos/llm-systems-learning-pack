# 项目 1：Vector Add 的边界与 `float4` 向量化

本项目同时保留标量 baseline 和手写 `float4` 路径，并让 host dispatcher 根据
实际地址与问题大小选择安全实现。

## 已实现内容

- block size 64、128、256、512 自动 sweep；
- scalar kernel：每个线程处理一个 `float`；
- `float4` kernel：每个线程处理四个连续 `float`；
- `N % 4` 的 0–3 个元素在同一个 kernel 中用标量访问处理；
- `N < 4` 时回退 scalar，避免非法的 0-block launch；
- A、B、C 任一地址不满足 16-byte alignment 时回退 scalar；
- C 使用 NaN poison 和尾部 guard；offset>0 时前置区也充当 prefix guard，用于
  检测漏写与 allocation 内的逻辑越界写；越界读/写另由 Compute Sanitizer 检测；
- CPU reference、绝对/相对误差与 guard 检查；
- CTest 边界矩阵与 Compute Sanitizer。

## 编译与一键验收

```bash
cd /home/undefined/Desktop/ai/projects/cutlass-learning
./scripts/build.sh
./scripts/run_vector_add_project.sh
./scripts/sanitize.sh
./scripts/inspect_vector_add_sass.sh
```

## 命令行

```text
vector_add [N] [iterations] [a_offset] [b_offset] [c_offset] [block_size]
```

- offset 的单位是 `float` 元素，不是 byte；
- `cudaMalloc` 基址至少满足 `float4` 对齐，`offset=1` 会把地址移动 4 byte，
  从而稳定制造“不满足 16-byte alignment”的测试；
- 不给 `block_size` 时自动测试 64/128/256/512；
- offset 默认都是 0。

示例：

```bash
# 对齐，N%4=1：float4 主体 + 1 个标量尾元素
./build/bin/vector_add 257 20 0 0 0 256

# 只有 A 未对齐：requested-vector 路径整体回退 scalar
./build/bin/vector_add 257 20 1 0 0 256

# A/B/C 都未对齐并 sweep 四种 block size
./build/bin/vector_add 1000003 20 1 1 1
```

## Dispatcher 为什么这样设计

只有同时满足下列条件才 reinterpret 成 `float4*`：

1. A、B、C 三个 device pointer 都是 16-byte aligned；
2. `N >= 4`，至少存在一个完整的四元素 pack。

`N % 4 != 0` 并不要求整个问题回退。`floor(N/4)` 个完整 pack 仍走
`float4`，剩下 0–3 个元素由前几个线程用 `float` 标量访问处理。任一指针未
对齐时才整体回退，因为对未对齐地址做 `float4` load/store 在 C++/CUDA 内存
模型中不合法，不能指望硬件“勉强工作”。

## 正确性矩阵

CTest 覆盖：

```text
N = 1, 3, 255, 256, 257, 1,000,003
A-only、B-only、C-only、A/B/C-all unaligned
block = 64, 128, 256, 512
```

每次 variant 运行前把整个 C allocation poison 成 NaN。如果向量 kernel 忘了
写尾部，旧的 scalar 结果不会替它掩盖错误。逻辑输出后固定有四个 guard 元素；
`c_offset>0` 时，逻辑输出前的 offset 区也作为 prefix guard。它们必须保持 NaN。

## 如何理解性能

一次 Vector Add 的逻辑流量是两次读加一次写。程序只计 kernel，不计 allocation、
H2D 或 D2H。

- small case 反复访问后容易驻留 L2，数字是 cache-sensitive bandwidth；
- 极小 N 通常被 launch latency 主导；
- large case 的三个向量总量超过本机 48 MiB L2，更接近显存流量实验；
- `float4` 不会自动获得四倍带宽：标量 warp 本来也能形成合并访存。它主要减少
  thread 数以及 load/store 指令数量，是否更快必须实测；
- 桌面 GPU 未锁频，因此一次 sweep 中的小差异不能当成稳定结论。

`inspect_vector_add_sass.sh` 已确认当前 `sm_89` 二进制含有
`LDG.E.128` / `STG.E.128`，证明手写向量访问确实落成 128-bit 全局访存指令。

## 高级扩展（已完成，未覆盖基础版）

基础版仍保留在 `main.cu`；所有后续实验位于新的 `advanced.cu`，可执行文件是
`vector_add_advanced`。

已经实现：

- 任意 warp 整数倍 block range，默认 32、64、…、1024；
- 每配置独立 10 轮，每轮多次 launch，保留原始样本并报告 type-7 median/p95；
- `float`/`float4`、`__half`/`half2`、`int32`/CUDA `int4`；
- aligned 与 element offset=1 的 scalar alignment fallback 对照；
- CSV、PNG、SVG，以及 median 到 p95-latency-derived bandwidth 阴影；
- 每个测量 batch 的 NVTX range 和 Nsight Systems timeline；
- Nsight Compute 的 global load/store、DRAM 与 L2 指标脚本；
- SASS 检查确认 `float4/int4` 的 128-bit 访存以及 `half2` 的 packed `HADD2`；
- 输出前后固定 guard、typed CPU reference、CTest 和 Compute Sanitizer。

注意 CUDA `int4` 是四个 32-bit `int`，总计 16 byte，不是量化中的 4-bit INT4；
`half2` 是两个 FP16，总计 4 byte。

一键跑完整 10 轮曲线：

```bash
./scripts/run_vector_add_advanced.sh
```

结果位于：

```text
results/vector_add_advanced/aligned.csv
results/vector_add_advanced/unaligned_offset1.csv
results/vector_add_advanced/vector_add_bandwidth.{png,svg}
results/vector_add_advanced/vector_add_alignment.{png,svg}
```

自定义 sweep 示例：

```bash
./build/bin/vector_add_advanced \
  --n 16777219 --iterations 20 --rounds 10 --warmup 3 \
  --min-block 96 --max-block 960 --step 96 \
  --types all --variants all --offset 0 \
  --csv results/vector_add_advanced/custom.csv
```

`min/max/step` 都必须构成 warp（本机 32 threads）的整数倍；`--block 256` 可只
测一个 block size。CSV 同时记录 `requested_variant` 和 `actual_path`，因此未对齐
时不会把 scalar fallback 误标为 packed 性能。

NVTX/Nsight Systems 已验证：

```bash
./scripts/profile_vector_add_advanced_nsys.sh
```

输出位于 `profiles/vector_add_advanced/nvtx_timeline.nsys-rep`，其中可以直接看到
`float32/float4/path=.../block=256/round=...` 等 range。

Nsight Compute 对普通用户仍受系统管理员权限限制；本机已用下面的一次性管理员
方式成功采集六个 scalar/packed kernel，没有永久修改驱动权限：

```bash
sudo ./scripts/profile_vector_add_advanced_ncu.sh
```

输出包括六份 `.ncu-rep`、19 项原始指标的 `summary.csv`，以及直接比较 scalar 与
packed 的 `comparison.csv`：

```text
profiles/vector_add_advanced/ncu/
results/vector_add_advanced/ncu/summary.csv
results/vector_add_advanced/ncu/comparison.csv
```

实测中 `float4`/CUDA `int4` 把 warp-level global load/store 指令数降至 scalar 的
约 1/4，`half2` 降至约 1/2；但三组 kernel 时间只变化约 -1.5% 到 +2.5%。这说明
指令减少是真实的，但 DRAM 已接近高利用率，Vector Add 不会因此获得 2–4 倍加速。

若确实需要普通用户反复采集，也可仅在可信个人开发机持久开放，重启后再运行：

```bash
sudo ./scripts/enable_nvidia_performance_counters.sh
sudo reboot
./scripts/profile_vector_add_advanced_ncu.sh
```

恢复 admin-only：

```bash
sudo ./scripts/disable_nvidia_performance_counters.sh
sudo reboot
```

本机最终二进制重跑后的最佳 median 有效带宽约为 447–456 GB/s：float
scalar/float4 约 453/448，half scalar/half2 约 456/455，int scalar/CUDA int4
约 454/447 GB/s。六条路径接近，说明这个 workload 主要受 memory subsystem
限制。曲线有明显桌面负载/频率噪声，不能只看单个最高点。
