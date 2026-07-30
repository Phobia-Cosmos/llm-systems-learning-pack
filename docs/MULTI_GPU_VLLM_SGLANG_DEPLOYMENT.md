# A100 上的 vLLM、SGLang 单卡基线与多卡部署

记录日期：2026-07-29

## 当前环境与结论

测试环境为单张 NVIDIA A100-SXM4-40GB、驱动 550.54.15、Python 3.11.11、PyTorch 2.6.0+cu124、vLLM 0.8.5、SGLang 0.4.6.post5。模型使用 BF16 的 Qwen3-8B；服务上下文上限为 8192 tokens，显存利用率上限为 85%。

Qwen3-8B 在 vLLM 和 SGLang 上均完成短上下文、约 2.7K tokens、约 7.3K tokens、近 8K tokens 以及并发 1–32 的测试，所有请求均成功，没有 OOM、超时或服务崩溃。短上下文并发 32 时，vLLM 与 SGLang 的输出吞吐分别约为 1409 和 1455 tokens/s。近 8K 上下文并发 16 时，vLLM 的输入吞吐约为 23.6K tokens/s、P95 延迟约为 5.42 秒；SGLang 的输入吞吐约为 54.3K tokens/s、P95 延迟约为 2.36 秒。

单卡双模型测试使用 Qwen3-8B 和 Qwen3-0.6B 两个独立 vLLM 实例。两个服务同时在线时显存约为 37.6 GiB；2.7K 上下文合计并发 24、7.3K 上下文合计并发 12 均无失败。共享同一张 GPU 会明显增加 8B 模型的首 token 延迟，生产部署更适合每个模型独占一张 GPU。

完整原始结果位于 `ai/benchmarks/gpu-serving/20260728-155836/`。

## 多卡申请要求

优先申请同一台服务器、同一容器内可见的同型号 GPU。张量并行需要 GPU 之间频繁通信，A100 SXM 之间有 NVLink/NVSwitch 时效果最好；如果拓扑显示 `SYS`，通信需要跨 CPU/PCIe，扩展效率会明显下降。

建议的资源规格：

| 用途 | GPU | CPU | 内存 | 说明 |
| --- | ---: | ---: | ---: | --- |
| Qwen3-8B 多副本吞吐测试 | 2×A100 40GB | ≥32 核 | ≥128 GiB | 每卡一个副本，前端负载均衡 |
| Qwen3-14B 张量并行与长上下文 | 2×A100 40GB | ≥32 核 | ≥160 GiB | `TP_SIZE=2` |
| 8B 与 14B 多模型同时服务 | 2×A100 40GB | ≥32 核 | ≥160 GiB | 每个模型独占一张卡 |
| 32B/更长上下文测试 | 4×A100 40GB | ≥64 核 | ≥256 GiB | 根据模型结构使用 TP=4 |
| 70B BF16 | 8×A100 40GB 或更大显存 GPU | ≥96 核 | ≥512 GiB | 还需为 KV cache 和激活保留空间 |

模型能否部署不能只比较权重大小与总显存。每张卡还需要 CUDA context、计算图、临时激活和 KV cache；建议把总显存上限设为 85%–90%，不要把权重刚好塞满所有显存。

## 申请到多卡后的检查

先挂载已经准备好的持久化环境：

```bash
/public/home/u43077/lzh/scripts/install-serving-cu124.sh
/public/home/u43077/lzh/scripts/verify-serving.sh
```

确认所有 GPU 在同一个 Notebook 容器内可见：

```bash
nvidia-smi -L
nvidia-smi topo -m
/opt/venvs/vllm/bin/python - <<'PY'
import torch

print("device_count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))
PY
```

如果申请了两张卡，`torch.cuda.device_count()` 必须返回 2。`nvidia-smi topo -m` 中 GPU 间显示 `NV#` 最好，`PIX/PXB/PHB` 次之，`SYS` 最慢。多卡启动失败时使用 `NCCL_DEBUG=INFO` 检查 NCCL 初始化，而不是先重新安装 PyTorch。

## 单模型张量并行

vLLM 的两卡 Qwen3-14B：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export TP_SIZE=2
export MAX_MODEL_LEN=32768
export GPU_MEMORY_UTILIZATION=0.90
export NCCL_DEBUG=WARN
/public/home/u43077/lzh/scripts/serve-vllm.sh Qwen3-14B 8000
```

SGLang 的两卡 Qwen3-14B：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export TP_SIZE=2
export MAX_MODEL_LEN=32768
export GPU_MEMORY_UTILIZATION=0.90
export NCCL_DEBUG=WARN
/public/home/u43077/lzh/scripts/serve-sglang.sh Qwen3-14B 30000
```

`TP_SIZE` 必须等于参与张量并行的 GPU 数量，并且通常应能整除模型的 attention heads/KV heads。先使用 TP=2；申请四卡后再验证模型结构是否适合 TP=4。单机张量并行不要为每个进程分别设置不同的 `CUDA_VISIBLE_DEVICES`，整个服务进程需要看到所有参与的 GPU。

## 多副本提高吞吐

对于已经能放进单卡的 Qwen3-8B，两个独立副本通常比 TP=2 更适合大量互不相关的短请求：

```bash
CUDA_VISIBLE_DEVICES=0 GPU_MEMORY_UTILIZATION=0.90 \
  /public/home/u43077/lzh/scripts/serve-vllm.sh Qwen3-8B 8000 \
  > /public/home/u43077/lzh/run/qwen3-8b-gpu0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 GPU_MEMORY_UTILIZATION=0.90 \
  /public/home/u43077/lzh/scripts/serve-vllm.sh Qwen3-8B 8001 \
  > /public/home/u43077/lzh/run/qwen3-8b-gpu1.log 2>&1 &
```

确认两个端点分别健康：

```bash
curl -f http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8001/health
```

然后由 Nginx、HAProxy 或调用端以 round-robin/least-connections 方式把请求分配到两个端口。多副本会复制模型权重，但避免 TP 的卡间通信，通常具有更好的短请求总吞吐。

## 多模型同时部署

多卡环境中不要让多个 vLLM 实例争抢同一张 GPU，直接按 GPU 隔离：

```bash
CUDA_VISIBLE_DEVICES=0 GPU_MEMORY_UTILIZATION=0.90 \
  /public/home/u43077/lzh/scripts/serve-vllm.sh Qwen3-8B 8000 \
  > /public/home/u43077/lzh/run/qwen3-8b.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 GPU_MEMORY_UTILIZATION=0.90 \
  /public/home/u43077/lzh/scripts/serve-vllm.sh Qwen3-14B 8001 \
  > /public/home/u43077/lzh/run/qwen3-14b.log 2>&1 &
```

这样每个服务拥有独立 CUDA context 和 KV cache，性能和故障隔离都优于单卡多实例。若一个模型需要两卡 TP、另一个模型也需要两卡 TP，则应申请四卡，并分别使用 `CUDA_VISIBLE_DEVICES=0,1` 与 `CUDA_VISIBLE_DEVICES=2,3`。

## 多卡压力测试

先预热，再运行至少三类负载：

1. 短上下文：约 300 输入 tokens、64 输出 tokens，并发 1/8/32/64。
2. 中等上下文：约 2.7K 输入 tokens、64 输出 tokens，并发 1/8/16/32。
3. 长上下文：约 7.3K 或接近服务上限的输入，并发 1/4/8/16。

示例：

```bash
/public/home/u43077/lzh/scripts/benchmark-openai.py \
  --base-url http://127.0.0.1:8000 \
  --model Qwen3-14B \
  --num-requests 64 \
  --concurrency 16 \
  --prompt-words 2800 \
  --max-tokens 64 \
  --timeout 900 \
  --output /public/home/u43077/lzh/benchmarks/qwen3-14b-tp2-long-c16.json
```

同时采集每张卡的数据：

```bash
nvidia-smi dmon -s pucvmet -d 1 -o DT \
  > /public/home/u43077/lzh/benchmarks/nvidia-smi-dmon.log
```

需要记录的指标包括成功率、请求吞吐、输入/输出 token 吞吐、P50/P95/P99 延迟、TTFT、每张卡的显存占用和利用率、卡间负载是否均衡、功耗与温度。TP 服务中某张卡长期低利用率通常意味着拓扑、NCCL、分片或调度存在问题。

## 选择建议

- 大模型无法放进单卡或需要扩大单请求 KV cache：使用 TP。
- 模型能放进单卡，目标是提高独立请求总吞吐：使用每卡一个副本。
- 多个不同模型同时在线：每个模型独占 GPU 或独占一组 GPU。
- 当前 A100 上的近 8K 长提示批处理：优先测试 SGLang。
- 当前环境的 SGLang 0.4.6.post5 不包含完整 HiCache；不要在 CUDA 12.4 环境中直接混装需要 CUDA 12.6+ 的 HiCache 依赖。
