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
- A/B 输入区与 C 前后 guard 使用 NaN canary，检测漏写、越界读和逻辑越界写；
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
写尾部，旧的 scalar 结果不会替它掩盖错误。逻辑输出前后还有四个 guard 元素；
它们必须始终保持 NaN。

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

## 推荐继续修改

1. 把 block size 扩展为任意 warp 整数倍并画曲线；
2. 独立重复 10 轮，报告 median/p95，而不是只看单轮平均；
3. 加 `half2`、`int4` 等不同向量类型，比较 alignment 与有效带宽；
4. 用 NVTX 标出 scalar/vectorized 区间，再用 Nsight Systems 看 timeline；
5. 管理员开放 performance counter 后，用 Nsight Compute 对比 global load/store
   指令数与 DRAM/L2 throughput。
