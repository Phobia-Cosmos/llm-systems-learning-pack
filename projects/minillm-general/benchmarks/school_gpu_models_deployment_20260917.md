# School GPU 训练模型本地部署矩阵

## 本地模型

三组学校 GPU 训练产物已从 `/public/home/u43077/lzh/outputs/minillm-general/inference-exports/` 拉取到 `/home/undefined/Disk/models/school-gpu/minillm-general/`。远端与本地 checkpoint/tokenizer 的 SHA-256 逐项一致；完整来源、step、训练 token 数和 digest 见该目录的 `manifest.json`。

| 模型 | 训练进度 | 原始推理 checkpoint | FP16 HF-like export |
| --- | ---: | ---: | ---: |
| MiniLLM 90M final | step 20,000 | 352 MB | 352 MB |
| MiniLLM 160M 4K final | step 211,345 / 10.00B tokens | 655 MB | 654 MB |
| MiniLLM 160M 8K final | step 218,975 / 11.00B tokens | 655 MB | 654 MB |

HF-like export 使用 `model.safetensors`，保留 MiniGPT、RoPE、12Q/4KV GQA、QK-Norm、RMSNorm、SwiGLU 和 tokenizer 配置。FP16 是 serving 副本；原始 FP32 checkpoint 没有覆盖。

## 本机验证结果

统一 prompt 为 `embedding 是把离散 token id 映射成`，greedy 生成 4 tokens。

| 模型 | Native CUDA KV | vLLM | nano-vLLM | mini-sglang HTTP | 官方 SGLang |
| --- | --- | --- | --- | --- | --- |
| 90M final | ` qwen`，14.290 ms | ` qwen`，token parity | ` qwen`，token parity | ` qwen` | 不支持 MiniGPT |
| 160M 4K final | `连续的 2`，26.820 ms | `连续的 2`，token parity | `连续的 2`，token parity | `连续的 2` | 不支持 MiniGPT |
| 160M 8K final | `连续的 2`，25.710 ms | `连续的 2`，token parity | `连续的 2`，token parity | `连续的 2` | 不支持 MiniGPT |

Native 延迟是 warmup 1 次后 3 次完整 4-token 生成的中位数。vLLM/nano-vLLM/mini-sglang 本轮以加载、生成和 token parity 为门禁，尚未把冷启动、HTTP 和稳态请求混为一个性能数字。

vLLM 和 nano-vLLM 的 MiniGPT adapter 已补齐训练模型需要的 per-head QK-Norm；否则 loader 会忽略或拒绝 `attn.q_norm/k_norm.weight`，结果不能作为正确部署。官方 SGLang 0.5.9 在 `AutoConfig.from_pretrained` 阶段明确报 `model type minigpt` 未注册，因此当前不是可选 Agent 后端。仓库的 mini-sglang 是 native MiniLLM 的 OpenAI-like HTTP wrapper，可用但没有 continuous batching、paged KV 或 prefix cache。

## 可复现命令

Native CUDA：

```sh
PYTHONPATH=projects/minillm-general /home/undefined/Disk/python-envs/brainuicl/bin/python projects/minillm-general/scripts/benchmark_minillm_generate.py \
  --checkpoint /home/undefined/Disk/models/school-gpu/minillm-general/minillm-160m-8k-final/minillm-160m-8k-final.pt \
  --tokenizer /home/undefined/Disk/models/school-gpu/minillm-general/minillm-160m-8k-final/tokenizer.json \
  --device cuda --kv-cache --max-new-tokens 4 --warmup 1 --samples 3 \
  --output /home/undefined/Disk/models/school-gpu/minillm-general/minillm-160m-8k-final/native_cuda_kv.json
```

vLLM：

```sh
/home/undefined/Disk/python-envs/vllm/bin/python projects/minillm-general/scripts/run_vllm_minigpt.py \
  --model /home/undefined/Disk/models/school-gpu/minillm-general/minillm-160m-8k-final/hf \
  --prompt 'embedding 是把离散 token id 映射成' --max-tokens 4 --temperature 0 --gpu-memory-utilization 0.25
```

nano-vLLM：

```sh
PYTHONPATH=projects/minillm-general:projects/nano-vllm /home/undefined/Disk/python-envs/sglang/bin/python projects/minillm-general/scripts/run_nanovllm_minigpt.py \
  --model /home/undefined/Disk/models/school-gpu/minillm-general/minillm-160m-8k-final/hf \
  --prompt 'embedding 是把离散 token id 映射成' --max-tokens 4 --temperature 0
```

mini-sglang：

```sh
PYTHONPATH=projects/minillm-general /home/undefined/Disk/python-envs/sglang/bin/python projects/mini-sglang/mini_sglang_server.py \
  --checkpoint /home/undefined/Disk/models/school-gpu/minillm-general/minillm-160m-8k-final/minillm-160m-8k-final.pt \
  --host 127.0.0.1 --port 18093 --device cuda
curl http://127.0.0.1:18093/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"embedding 是把离散 token id 映射成","max_tokens":4,"temperature":0,"greedy":true,"kv_cache":true}'
```

官方 SGLang 兼容性探测：

```sh
SGLANG_ENV=/home/undefined/Disk/python-envs/sglang
LD_LIBRARY_PATH="$SGLANG_ENV/lib/python3.12/site-packages/nvidia/cublas/lib:$SGLANG_ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib" \
  "$SGLANG_ENV/bin/python" -m sglang.launch_server \
  --model-path /home/undefined/Disk/models/school-gpu/minillm-general/minillm-90m-final/hf \
  --host 127.0.0.1 --port 31090 --tp-size 1 --mem-fraction-static 0.2 --disable-cuda-graph
```

当前选择建议：Agent 后端的近期主线使用 nano-vLLM，因为它在自研范围内已支持 MiniGPT、scheduler、paged KV，并能与 native/vLLM 做 token parity；vLLM 保留为成熟性能和服务基线；mini-sglang 保留为 API/Agent 协议试验入口；官方 SGLang 等 MiniGPT model adapter 落地后再进入正式对比。模型能力方面优先使用 160M 8K final，但它只是预训练模型，真实 tool calling 仍需 SFT 与 Agent eval，不能仅凭本次续写样例判定可直接担任可靠 planner。

vLLM OpenAI server 的本轮服务证据为 `minillm-90m-final/vllm-server.json`，请求返回 `qwen`，usage 显示 prompt 17 tokens、completion 4 tokens。mini-sglang 的 HTTP 证据保存在各模型目录的 `minisglang.json`；nano-vLLM 的输出保存在 `nanovllm.txt`。这些文件是功能门禁，不是跨引擎最终性能排名；正式排名需要同一模型、同一 dtype、同一 max context、相同 warmup 和多请求并发压测。
