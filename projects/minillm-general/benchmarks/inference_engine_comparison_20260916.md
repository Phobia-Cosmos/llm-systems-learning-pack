# MiniLLM 本机推理对照（2026-09-16）

本次对照使用同一个仓库内训练好的 `minillm-rope.pt`（L=2、hidden=128、4 heads、vocab=512），同一 prompt `embedding 是把离散 token id 映射成`，贪心生成 8 tokens。结果不是开发板部署结果，全部在本机 RTX 4070 SUPER 或 CPU 上执行。

## 可复现实验命令

原生 MiniLLM（不经过 serving engine）CUDA：

```sh
cd /home/undefined/llm-systems-learning-pack/projects/minillm-general
/home/undefined/Disk/python-envs/sglang/bin/python scripts/benchmark_minillm_generate.py \
  --checkpoint artifacts/checkpoints/minillm-rope.pt --device cuda \
  --max-new-tokens 8 --warmup 1 --samples 5 \
  --output benchmarks/results/native_minillm_cuda_20260916.json
```

原生 MiniLLM CPU：

```sh
/home/undefined/Disk/python-envs/brainuicl/bin/python scripts/benchmark_minillm_generate.py \
  --checkpoint artifacts/checkpoints/minillm-rope.pt --device cpu \
  --max-new-tokens 8 --warmup 1 --samples 5 \
  --output benchmarks/results/native_minillm_cpu_20260916.json
```

原生 MiniLLM CUDA KV-cache 变体：

```sh
/home/undefined/Disk/python-envs/sglang/bin/python scripts/benchmark_minillm_generate.py \
  --checkpoint artifacts/checkpoints/minillm-rope.pt --device cuda --kv-cache \
  --max-new-tokens 8 --warmup 1 --samples 5 \
  --output benchmarks/results/native_minillm_cuda_kv_20260916.json
```

逐算子/Prefill/Decode 对照（原生 MiniLLM 与 nano-vLLM 的真实 PyTorch fallback）：

```sh
/home/undefined/Disk/python-envs/sglang/bin/python scripts/benchmark_inference_shapes.py \
  --checkpoint artifacts/checkpoints/minillm-rope.pt --batch-sizes 1 \
  --dtypes float32,float16 --generated-tokens 8 --warmup 3 --samples 10 \
  --target-sample-ms 2 --max-inner-loops 64 --device cuda \
  --out-dir benchmarks/results --run-name inference_engine_compare_minillm_rope_20260916
```

同一逐算子对照的 CPU 版本（CPU 使用同步单调时钟，不伪称 CUDA Event）：

```sh
/home/undefined/Disk/python-envs/brainuicl/bin/python scripts/benchmark_inference_shapes.py \
  --checkpoint artifacts/checkpoints/minillm-rope.pt --batch-sizes 1 \
  --dtypes float32 --generated-tokens 8 --warmup 2 --samples 5 \
  --target-sample-ms 20 --max-inner-loops 8 --device cpu \
  --out-dir benchmarks/results --run-name inference_cpu_minillm_rope_20260916
```

实际 nano-vLLM engine（包含 scheduler、paged KV cache 和 engine 初始化）：

```sh
/home/undefined/Disk/python-envs/sglang/bin/python scripts/benchmark_nanovllm_minigpt.py \
  --model artifacts/hf_exports/minillm-rope --max-tokens 8 --warmup 1 --samples 5 \
  --gpu-memory-utilization 0.7 \
  --output benchmarks/results/nanovllm_minillm_20260916.json
```

`gpu-memory-utilization` 必须设为 0.7：当时 GPU 上已有约 3.1 GiB 进程占用，默认 0.2 无法分配正数 KV blocks，engine 会在 `allocate_kv_cache` 处阻塞。这是资源配置错误，不是模型不支持。

## 结果

| 路径 | 设备/运行时 | 首次加载/初始化 | 单次 8-token 生成中位数 | 输出 |
| --- | --- | ---: | ---: | --- |
| Native MiniLLM | CUDA eager，无 KV cache | 1817 ms | 4.760 ms | `连续向量的` |
| Native MiniLLM | CUDA eager，KV cache | 1249 ms | 5.406 ms | `连续向量的` |
| Native MiniLLM | CPU eager，无 KV cache | 3245 ms | 2074.012 ms | `连续向量的` |
| nano-vLLM | CUDA engine，eager + paged KV | 1065 ms | 6.445 ms | `连续向量的` |
| mini-sglang server | CUDA HTTP wrapper + native MiniLLM KV | 常驻进程 | 12.585 ms（HTTP round trip） | `连续向量的` |

Native 与 nano-vLLM 生成的 token suffix 一致（`[328,259,282,262,292,246,340,275]`），说明本次 engine 路径没有改变贪心结果。nano-vLLM 相对 native CUDA eager 的单请求中位延迟约慢 1.35 倍，但它额外提供 scheduler、paged KV cache 和批处理接口；本次小模型/单请求不能代表大模型吞吐。mini-sglang HTTP wrapper 又引入 Python HTTP 和 JSON 开销，约为 native eager 的 2.6 倍。

逐算子 CUDA 对照的 full-pass 结果如下：

| 运行时 | Prefill GPU median | Decode GPU median |
| --- | ---: | ---: |
| MiniLLM native float32 | 0.525 ms | 0.492 ms |
| nano-vLLM PyTorch fallback float32 | 0.596 ms | 0.566 ms |
| MiniLLM native float16 | 0.563 ms | 0.538 ms |
| nano-vLLM PyTorch fallback float16 | 0.654 ms | 0.624 ms |

同一对照的 CPU float32 full-pass 为 native `0.550 ms` prefill / `0.333 ms` decode，nano fallback `0.513 ms` / `0.354 ms`（设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`）；这是单次很小张量的算子调用时间，不能与“完整 8-token CPU 生成中位数 2074 ms”混为一谈，后者包含逐 token Python 循环和完整序列重新计算。

## 对 MiniSGLang/NanoVLLM 的开发结论

- `nano-vllm` 已能直接加载本项目的 MiniGPT HF-like export，并且其 paged KV cache 与 native MiniLLM logits 在本次 prompt 上一致；下一步应补充统一 benchmark schema、真实多请求 batch、prefix-cache 命中率和 scheduler trace。
- `mini-sglang-official` 当前 registry 只注册 Qwen2/Qwen3 架构，不能直接加载本项目 MiniGPT；不要把官方 Mini-SGLang 的 Qwen 支持误写成自研模型支持。
- 仓库内 `projects/mini-sglang/mini_sglang_server.py` 是一个 OpenAI-like 教学 HTTP wrapper，不是官方 SGLang runtime。它可以服务本项目 checkpoint，但当前没有 RadixAttention、continuous batching 或 paged KV；本次 HTTP 结果用于量化 API 层开销。
- 下一步应先让 NanoVLLM 和 MiniSGLang 共用一个 `ModelAdapter`/`GenerationRequest`/`GenerationResult` 契约，再实现 prefix cache、batch scheduler、streaming 和 metrics；每项优化都必须与 native MiniLLM 做 logits/token parity。
- Agent 系统先使用本机 native/nano-vLLM 的自训练 MiniLLM 作为 planner/tool-calling 实验模型，工具执行、状态和安全策略放在 engine 之外；等模型输出与调度 trace 契约稳定后，再接 EdgeForge Worker/Artifact。

## 统一生成契约回归

已把三条路径的最小接口收敛为 `GenerationRequest`、`GenerationResult` 和 `GenerationBackend`。native MiniLLM 与 nano-vLLM 返回 completion token ids，HTTP wrapper 明确标记 token ids 未由当前 API 暴露，因此 Agent 层不依赖某个引擎的返回字典，也不会把 prompt token 误计入 completion。

可复现实验命令：

```sh
cd /home/undefined/llm-systems-learning-pack
PYTHONPATH=projects/minillm-general /home/undefined/Disk/python-envs/brainuicl/bin/python projects/minillm-general/scripts/contract_smoke.py --backend native --device cuda --output projects/minillm-general/benchmarks/results/contract_native_20260916.json
PYTHONPATH=projects/minillm-general:projects/nano-vllm /home/undefined/Disk/python-envs/brainuicl/bin/python projects/minillm-general/scripts/contract_smoke.py --backend nano --output projects/minillm-general/benchmarks/results/contract_nano_20260916.json
```

本次回归两条路径生成完全相同的后缀：`连续向量的`，token ids 为 `[328, 259, 282, 262, 292, 246, 340, 275]`。这只是单请求 greedy token parity，不代表两条路径延迟相同；调度、paged KV 和多请求吞吐仍需单独 benchmark。

## 四请求 batch 回归

为验证统一契约不是只服务单请求，又对 4 个等长请求执行了 warmup 后 3 次 CUDA batch。native MiniLLM KV-cache 的中位数为 `3.755 ms / 32 completion tokens`，nano-vLLM scheduler + paged KV 的中位数为 `5.268 ms / 32 completion tokens`；两者四条输出的 token ids 均相同。当前模型很小，nano 的单机调度开销仍高于 native，但 batch 结果已经能作为后续 prefix cache、continuous batching 和 admission control 的基线。等长请求是为了避免教学 MiniGPT 当前没有 attention mask 时把 padding 当作上下文。

命令和证据：

```sh
PYTHONPATH=projects/minillm-general /home/undefined/Disk/python-envs/brainuicl/bin/python projects/minillm-general/scripts/benchmark_batch_contract.py --backend native --device cuda --warmup 1 --samples 3 --output projects/minillm-general/benchmarks/results/batch_native_20260916.json
PYTHONPATH=projects/minillm-general:projects/nano-vllm /home/undefined/Disk/python-envs/brainuicl/bin/python projects/minillm-general/scripts/benchmark_batch_contract.py --backend nano --warmup 1 --samples 3 --output projects/minillm-general/benchmarks/results/batch_nano_20260916.json
```
