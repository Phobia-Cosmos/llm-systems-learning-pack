# 练习项目与扩展题

前五个项目已经有可编译代码。建议先“改现有项目”，再新建更复杂目录；每一题
都要求保留 reference、ragged shape 和复现命令。

## 项目 1：Vector Add 的边界与向量化（已完成）

起点：[`01_vector_add/main.cu`](01_vector_add/main.cu)

已经完成：

1. block size 64/128/256/512 sweep；
2. scalar 与 `float4` 两个 kernel；
3. `N % 4` 标量尾部和 `N < 4` fallback；
4. A/B/C 任一地址未 16-byte aligned 时整体回退 scalar；
5. small/L2-sensitive 与大于 L2 的数据集；
6. NaN poison、前后 guard、CPU reference、CTest 与 memcheck；
7. SASS 中的 `LDG.E.128` / `STG.E.128` 证据。

一键复现见 [`01_vector_add/README.md`](01_vector_add/README.md) 和：

```bash
./scripts/run_vector_add_project.sh
./scripts/inspect_vector_add_sass.sh
```

## 项目 2：Tiled Transpose 与 Bank Conflict（入门）

建议新建 `05_tiled_transpose/`，实现三版：

- naive transpose；
- shared-memory coalesced transpose；
- `tile[T][T+1]` padding 版本。

建议 shape：`1024×1024`、`1023×1001`、`32×4096`。

验收：CPU reference 通过；解释为什么读或写不合并；用 padding 前后结果证明
bank conflict 的影响。当前 Nsight Compute counter 权限未开放时，先用 Event
和代码推理，管理员开放后补 counter 证据。

## 项目 3：GEMM Tile Sweep（入门到中级）

起点：[`02_tiled_gemm/main.cu`](02_tiled_gemm/main.cu)

把 8、16、32 三种 tile 同时编译成三个 kernel。对以下 shape 建表：

```text
129 × 131 × 127    # 三个维度都有尾块
512 × 512 × 512    # 快速方阵
1024 × 1024 × 1024
32 × 4096 × 4096   # decode/small-M 风格
4096 × 32 × 4096   # 长瘦形状
```

报告 shared memory/CTA、threads/CTA、时间与 TFLOP/s。预测哪个 tile 最好，再
用数据验证。进阶版加入 register blocking，让每个 thread 计算多个 C 元素。

## 项目 4：Tensor Core Shape 与数值实验（中级）

起点：[`03_cutlass_sgemm/tensorop.cu`](03_cutlass_sgemm/tensorop.cu)

任务：

- 扫 `M=8,16,32,64,128,512`，固定 `N=K=4096`；
- 对比 FP32 CUDA Core、FP16 Tensor Core 和 cuBLAS；
- 改输入范围，记录最大绝对/相对误差；
- 比较 beta 为 0 和非 0；
- 用 `inspect_sass.sh` 保留 HMMA 证据。

验收：解释 small-M 下 CTA 数、launch 开销和 tile 浪费；区分数学模式，不能
把严格 FP32 与 FP16 输入当成相同精度基准。

现成 sweep：

```bash
./scripts/shape_sweep.sh
```

## 项目 5：GEMM + Bias + ReLU 融合（中级）

建议新建 `05_fused_bias_relu/`。参考：

```text
third_party/cutlass/examples/12_gemm_bias_relu/gemm_bias_relu.cu
```

实现两条路径：

1. CUTLASS GEMM，然后独立 bias+ReLU kernel；
2. 使用 `LinearCombinationRelu` 的 fused epilogue。

先沿用官方 column-major 输出和 per-row bias 语义，避免把广播方向搞反；理解后
再改 row-major。至少测 `520×504×264` 这种满足向量对齐但不是 CTA tile 整数倍
的 shape。

验收：CPU reference；bias 广播正确；Nsight Systems 显示 fused 少一次 launch；
报告中间张量流量、端到端时间和 register 变化。

## 项目 6：Shape-aware Mini Autotuner（中级到研究入门）

在项目 3/4 的多个 kernel 配置上建立自动选择器：

```text
输入特征：M、N、K、dtype、alignment
候选配置：tile、stage、SIMT/TensorOp
输出：最快的合法 kernel
```

训练集和测试集必须按 shape 分开，至少包含方阵、长瘦、small-M、ragged shape。
先用穷举表和最近邻规则，不必立刻上机器学习。

验收：与“永远固定一个 kernel”和 cuBLAS 做 baseline；报告搜索成本、编译成本、
命中率和端到端收益，避免只展示一个 4096³ 方阵。

## 项目 7：CuTe Layout Verifier（研究入门）

起点：[`04_cute_layout/main.cu`](04_cute_layout/main.cu)

给定 Shape/Stride 与 thread/value layout，自动检查：

- 坐标是否覆盖完整；
- 是否有重复索引或洞；
- vector access 是否对齐；
- 每个 thread 的元素数是否均衡；
- shared-memory bank 映射是否存在明显冲突。

先支持静态小 layout 并打印反例，再扩展 composition/local_partition。该项目不
追求 GPU 峰值，重点是把 CuTe 最难的 layout 推理变成可验证工具。

## 推荐选择

- 完全新手：项目 1 → 2 → 3。
- 想做 GPU kernel 工程：项目 3 → 4 → 5。
- 想做 LLM 推理优化：项目 4 → 5 → 6，重点加入 small-M/MoE shape。
- 想做编译器/研究：项目 3 → 6 → 7。
