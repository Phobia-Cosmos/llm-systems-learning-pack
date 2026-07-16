# Python/PyTorch 算子教学与基准

这个目录先把 PyTorch 官方实现当作 reference/baseline，再用可读的 Python 张量表达式展开算子数学语义，最后可选择把同一教学函数交给 `torch.compile`，或对一个明确热点改写为 Triton。当前阶段不写 C++/CUDA，后续 extension 仍应接入同一个 benchmark harness，而不是另写一套不可比较的计时程序。

PyTorch 的 dispatcher 中有数千个 schema、overload、in-place/out/不同 backend 组合，因此“列出并重写全部 PyTorch 算子”没有稳定边界。这里把 MiniLLM、nano-vLLM 和常见深度学习中真正高频的算子族列全，并为其中 30 个代表算子提供可运行教学实现；同一族的 dtype、shape 和 overload 由统一实现与 dispatcher 覆盖，而不是复制几十份源码。

## 常见算子族

| 算子族 | 常见 PyTorch API | 在模型中的用途 | 本目录状态 |
| --- | --- | --- | --- |
| 张量创建 | `empty/zeros/ones/full/arange/rand/randn/randint` | 参数、mask、位置、临时张量 | 列入清单；它们主要定义数据，不单独复刻 allocator/RNG |
| 布局与视图 | `view/reshape/flatten/squeeze/unsqueeze/transpose/permute/contiguous/cat/stack/split/chunk` | head 拆分、QKV 切分、layout 变换 | 在 RoPE、attention、Linear 等实现中使用 |
| 逐元素算术 | `add/sub/mul/div/pow/maximum/minimum/clamp/where` | residual、bias、scale、mask | `vector_add`、`saxpy` |
| 比较与逻辑 | `eq/ne/lt/le/gt/ge/logical_and/or/not` | causal mask、边界和路由条件 | 在 ReLU、attention mask 中使用 |
| 数学函数 | `exp/log/sqrt/rsqrt/sin/cos/tanh/erf` | 激活、归一化、RoPE、概率 | 在多个教学公式中使用 |
| 激活 | `relu/sigmoid/tanh/silu/gelu/softplus` | MLP 与门控 | ReLU、sigmoid、SiLU、精确/近似 GELU、bias+SiLU |
| 归约与统计 | `sum/mean/amax/amin/argmax/var/std/logsumexp` | loss、norm、softmax、采样 | tree sum/mean/max、softmax、log-softmax |
| 矩阵与线性 | `dot/mv/mm/matmul/bmm/addmm/einsum/F.linear/nn.Linear` | QKV、attention、MLP、lm-head | GEMM 语义、BMM、decode/prefill Linear |
| 卷积与池化 | `conv1d/2d/3d/max_pool/avg_pool/adaptive_pool` | CNN、视觉/音频前端 | unfold+matmul Conv2d、unfold+max MaxPool2d |
| 索引与重排 | `embedding/index_select/gather/scatter/scatter_add/masked_select/sort/topk` | token embedding、KV page、MoE、采样 | one-hot 语义版 embedding/gather/scatter-add、sort 版 top-k |
| 归一化 | `layer_norm/rms_norm/batch_norm/group_norm` | Transformer/CNN 数值稳定 | LayerNorm、RMSNorm、BatchNorm inference |
| 位置编码 | `sin/cos` 加 pair rotation，或专用 fused RoPE | Transformer token 位置 | pairwise RoPE |
| Attention | `scaled_dot_product_attention`、显式 `QK^T→softmax→PV` | prefill/decode attention | causal SDPA reference 与显式教学版 |
| Loss | `cross_entropy/nll_loss/mse_loss/binary_cross_entropy/kl_div` | 训练目标 | cross entropy、MSE |
| 随机与采样 | `multinomial/topk/argmax/sort` | token generation | deterministic top-k；随机采样需额外统一 RNG 状态 |
| 稀疏/MoE | sparse tensor ops、segment/grouped GEMM、routing/scatter | expert dispatch 和稀疏计算 | 当前用 gather/scatter 展示地址语义，grouped GEMM 留到 scheduler 阶段 |
| 量化 | `quantize_per_tensor/per_channel`、fake quant、backend-specific INT8/FP8/INT4 | 降低权重/激活容量与流量 | 先记录 dtype/scale 语义，等 FP32/FP16 baseline 稳定后实现 |
| 通信 | `torch.distributed` 的 `all_reduce/all_gather/reduce_scatter/all_to_all` | tensor/data/expert parallel | 多 GPU 算子，不在当前单 GPU correctness 集合中 |

`operators.py` 中的教学版并不都追求高性能。例如 `matmul_from_broadcast` 会显式形成 `M×K×N` 临时张量，one-hot embedding/gather/scatter 也会制造大中间量；它们用于看清数学依赖，性能差正是 baseline 应展示的结果。较完整的 Python 优化路径是先用同一公式验证，再尝试 `torch.compile` 融合，最后只把 profiler 证明的热点写成 Triton。

## 已实现层级

```text
PyTorch reference
    ↓ 相同输入做 correctness
Python teaching composition
    ↓ torch.compile，编译时间不计入 steady-state
Inductor generated kernel(s)
    ↓ 仍不满足 shape/layout/fusion 目标时
Triton teaching kernel + eligibility checks + PyTorch fallback
    ↓ 需要 CUTLASS、显式 ABI、复杂调度或更低层控制时
C++/CUDA extension（后续阶段）
```

`triton_ops.py` 已把 `F.silu(x+bias)` 展开为一个 Triton JIT kernel，并检查 device、dtype、shape、stride、16-byte base alignment 与 autograd 范围。不满足条件时 `fused_bias_silu_dispatch` 回退到 PyTorch；当前 Triton 版本是 forward-only，因此输入需要梯度时必须 fallback，不能悄悄返回无法反向传播的结果。

## 运行

CPU correctness 使用共享 CPU 环境：

```bash
cd /home/undefined/Desktop/ai/projects/cutlass-learning/05_python_operators
/home/undefined/Disk/python-envs/ai-core-py312/bin/python -m unittest -v test_operators.py
/home/undefined/Disk/python-envs/ai-core-py312/bin/python benchmark.py --device cpu --check-only
```

CUDA correctness 与基础计时使用现有 CUDA PyTorch 环境，不修改该共享环境：

```bash
PYTHON=/home/undefined/Disk/python-envs/vllm/bin/python
$PYTHON benchmark.py --device cuda --dtype float32 --check-only
$PYTHON benchmark.py --device cuda --dtype float32 --profile smoke --variants reference,teaching --output ../results/python_operators/gpu_smoke
```

对一个算子比较四个 Python 入口层级：

```bash
$PYTHON benchmark.py \
  --operators fused_bias_silu \
  --device cuda \
  --dtype float16 \
  --profile llm \
  --variants reference,teaching,compiled,triton \
  --warmup 10 --repeats 20 --inner 20 \
  --output ../results/python_operators/fused_bias_silu_llm
```

第一次 `compiled`/`triton` 调用会触发编译，harness 明确在计时前完成这一步；结果是 steady-state kernel/dispatch 时间。首次编译延迟仍应在部署实验中单独记录，不能把它与 steady-state median 混成一个数字。

## C、C++、CUDA 与 Python 怎样公平比较

普通 C 循环通常远快于逐元素 Python `for` 循环，因为 C 没有每个元素的 Python object/bytecode 开销；但 PyTorch Python API 本来就在一次调用中进入高度优化的 C++/CUDA kernel，所以“把 `torch.add` 用 C 重写”通常不会自动更快。普通 C 只能优化 CPU；要在 NVIDIA GPU 上实现 kernel，需要 CUDA C++、Triton、CuTe DSL 或调用 cuBLAS/CUTLASS 等 GPU 后端。

后续 C++/CUDA 版本应通过 `torch.library` 或 PyTorch extension 接受现有 `torch.Tensor`，然后由当前 `benchmark.py` 作为新增 variant 调用。这样 reference、Python、Triton、C++/CUDA 共用同一进程中的输入张量，计时区间不会混入文件 I/O、Host↔Device copy、进程启动或随机数据生成。

公平口径必须固定：相同 GPU/CPU、PyTorch/CUDA 版本、shape、dtype、layout/stride、alignment、输入数值、输出语义和误差容差；明确 inference 还是 forward+backward；固定 TF32/autocast 等精度开关；所有 JIT/build 在计时外；统一 warm-up、stream 与同步；报告多轮 median/p95 而非单次最小值；GPU 输入预先驻留 VRAM；同时记录 correctness、峰值显存和真实 prefill/decode shape。若 C++ 程序另行分配数据或用另一套计时方法，得到的数字不能直接与 Python API 结果比较。
