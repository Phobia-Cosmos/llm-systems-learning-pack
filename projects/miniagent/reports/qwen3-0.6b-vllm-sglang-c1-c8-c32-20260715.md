# Qwen 在 vLLM / SGLang 上的并发性能与 TTFT 退化报告

运行 ID：`qwen3-0.6b-vllm-sglang-c1-c8-c32-20260715`

## 结论摘要

本报告的数据校验已通过。

- vLLM pooled p95 TTFT：c1: 24.25 ms；c8: 105.95 ms；c32: 449.46 ms。
- SGLang pooled p95 TTFT：c1: 26.17 ms；c8: 158.11 ms；c32: 608.40 ms。
- c32 吞吐/E2E 权衡：SGLang 1845.88 token/s、E2E p95 2265.14 ms；vLLM 1835.90 token/s、E2E p95 2499.89 ms。

TTFT 是客户端从发出请求到收到第一个流式 token 的时间，包含同机 HTTP/前端处理、服务端排队、tokenization/prefill 以及首 token decode；它不是纯 GPU prefill 时间。

## 测量结果

下表的延迟百分位由所有 repetition 的逐请求原始样本合并后重新计算。“重复 p95”给出各次运行 p95 的中位数 `[最小值, 最大值]`，不会用运行级 p95 的平均值代替 pooled p95。
吞吐按所有 repetition 的总 token 数除以总持续时间计算，因此是 duration-weighted 聚合。

| 引擎 | 并发 | 请求样本 | TTFT p50 / p95 / p99 (ms) | 重复 p95 中位数 [min, max] (ms) | TPOT p95 (ms/token) | E2E p95 (ms) | 输出吞吐 (token/s) | max waiting / running | queue mean (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vLLM | 1 | 768 | 22.42 / 24.25 / 25.20 | 24.35 [23.93, 24.37] | 3.71 | 495.01 | 268.16 | 0 / 1 | 0.00 |
| vLLM | 8 | 768 | 66.33 / 105.95 / 149.64 | 106.05 [105.95, 106.42] | 6.84 | 924.50 | 1143.50 | 3 / 8 | 0.96 |
| vLLM | 32 | 768 | 141.30 / 449.46 / 688.30 | 437.44 [426.78, 439.87] | 16.97 | 2499.89 | 1835.90 | 26 / 32 | 31.72 |
| SGLang | 1 | 768 | 23.84 / 26.17 / 27.08 | 26.24 [25.80, 26.49] | 3.59 | 481.00 | 276.48 | 0 / 1 | 0.07 |
| SGLang | 8 | 768 | 103.97 / 158.11 / 164.66 | 158.55 [156.97, 158.68] | 6.78 | 916.51 | 1155.12 | 4 / 8 | 26.71 |
| SGLang | 32 | 768 | 340.23 / 608.40 / 626.17 | 610.36 [594.67, 615.44] | 16.97 | 2265.14 | 1845.88 | 27 / 32 | 232.23 |

## p95 TTFT 为什么退化

下面每一阶先列测量事实，再列因果证据，最后才给解释。`max_waiting` 用于确认队列是否出现；只有 mean queue time ≥ 1 ms（或 mean 缺失而 waiting 非零）时，才把排队称为退化的实质贡献。

- **vLLM，并发 1 → 8**：测量事实：pooled p95 TTFT 从 24.25 ms 上升到 105.95 ms（+336.9%）。
  - 因果证据：服务端仅观测到轻微/短暂排队（max_waiting=3，queue mean=0.96 ms，低于 1 ms 诊断阈值）；max_running=8；KV/token usage max=14.4%；GPU utilization max=100%；prefill mean=46.33 ms。
  - 证据支持的解释：排队时间不足 1 ms，但服务端 prefill mean 从 19.90 ms 增至 46.33 ms；主要退化与更大的并发 prefill batch / prefill-decode 计算竞争一致。它仍不是排除其他因素后的唯一因果证明。
- **vLLM，并发 8 → 32**：测量事实：pooled p95 TTFT 从 105.95 ms 上升到 449.46 ms（+324.2%）。
  - 因果证据：服务端直接观测到排队且达到实质幅度（max_waiting=26，queue mean=31.72 ms）；max_running=32；KV/token usage max=57.4%；GPU utilization max=100%；prefill mean=102.93 ms。
  - 证据支持的解释：请求在开始 prefill 前已经等待，排队至少是 TTFT 退化的一个贡献因素；仅凭这些指标不能断言它是唯一原因。
- **SGLang，并发 1 → 8**：测量事实：pooled p95 TTFT 从 26.17 ms 上升到 158.11 ms（+504.1%）。
  - 因果证据：服务端直接观测到排队且达到实质幅度（max_waiting=4，queue mean=26.71 ms）；max_running=8；KV/token usage max=16.1%；GPU utilization max=100%。
  - 证据支持的解释：请求在开始 prefill 前已经等待，排队至少是 TTFT 退化的一个贡献因素；仅凭这些指标不能断言它是唯一原因。
- **SGLang，并发 8 → 32**：测量事实：pooled p95 TTFT 从 158.11 ms 上升到 608.40 ms（+284.8%）。
  - 因果证据：服务端直接观测到排队且达到实质幅度（max_waiting=27，queue mean=232.23 ms）；max_running=32；KV/token usage max=64.2%；GPU utilization max=100%。
  - 证据支持的解释：请求在开始 prefill 前已经等待，排队至少是 TTFT 退化的一个贡献因素；仅凭这些指标不能断言它是唯一原因。

## 引擎间比较

- 并发 1：TTFT 尾延迟由 vLLM 领先，pooled p95=24.25 ms，对比 SGLang 的 26.17 ms（后者相对高 7.9%）；输出吞吐分别为 268.16 / 276.48 token/s；E2E p95 则是 SGLang 481.00 ms、vLLM 495.01 ms；mean queue time 为 0.00 / 0.07 ms。
- 并发 8：TTFT 尾延迟由 vLLM 领先，pooled p95=105.95 ms，对比 SGLang 的 158.11 ms（后者相对高 49.2%）；输出吞吐分别为 1143.50 / 1155.12 token/s；E2E p95 则是 SGLang 916.51 ms、vLLM 924.50 ms；mean queue time 为 0.96 / 26.71 ms。
- 并发 32：TTFT 尾延迟由 vLLM 领先，pooled p95=449.46 ms，对比 SGLang 的 608.40 ms（后者相对高 35.4%）；输出吞吐分别为 1835.90 / 1845.88 token/s；E2E p95 则是 SGLang 2265.14 ms、vLLM 2499.89 ms；mean queue time 为 31.72 / 232.23 ms。

## 综合根因判断与公平性检查

- 配置机制：每个 prompt 固定 1024 tokens，而单轮 prefill/batched budget 是 2048。c1: 1024/2048，至少 1 个 token-budget waves；c8: 8192/2048，至少 4 个 token-budget waves；c32: 32768/2048，至少 16 个 token-budget waves。因此并发越高，尾部请求经历更多 prefill admission/scheduling waves；这是配置与 waiting/queue 指标共同支持的机制，不只是对客户端曲线的猜测。
- vLLM 的 p95 TPOT 从 3.71 增至 16.97 ms/token，高并发 GPU utilization max=100%；这表明首 token 之外的 decode 也受到 batch 扩大和 GPU 计算竞争影响。
- SGLang 的 p95 TPOT 从 3.59 增至 16.97 ms/token，高并发 GPU utilization max=100%；这表明首 token 之外的 decode 也受到 batch 扩大和 GPU 计算竞争影响。
- KV 公平性：最高并发的 KV/token usage max 为 vLLM=57.4%、SGLang=64.2%；vLLM preemption=0、SGLang max_retracted=0。本次没有接近 100% 的 KV 使用或换出事件，因此不能把主要退化归因于 KV cache 耗尽。

## 实验上下文

- `config.client_timeout_seconds`: `1800.0`
- `config.concurrencies`: `1, 8, 32`
- `config.cuda_home`: `/usr/local/cuda-13.0`
- `config.engines`: `vllm, sglang`
- `config.flashinfer_workspace`: `/home/undefined/Disk/cache/flashinfer-system-cuda-release`
- `config.gpu_interval_seconds`: `0.5`
- `config.hf_home`: `/home/undefined/Disk/cache/models/huggingface`
- `config.host`: `127.0.0.1`
- `config.input_tokens`: `1024`
- `config.max_batched_tokens`: `2048`
- `config.max_model_len`: `2048`
- `config.max_running_requests`: `32`
- `config.metrics_interval_seconds`: `0.1`
- `config.model_path`: `/home/undefined/Disk/cache/models/huggingface/Qwen3-0.6B`
- `config.num_prompts`: `256`
- `config.output_root`: `/home/undefined/Desktop/ai/projects/miniagent/artifacts/runs`
- `config.output_tokens`: `128`
- `config.repeats`: `3`
- `config.seed`: `42`
- `config.served_model_name`: `qwen3-0.6b`
- `config.server_start_timeout_seconds`: `600.0`
- `config.server_stop_timeout_seconds`: `30.0`
- `config.sglang_memory_fraction`: `0.7`
- `config.sglang_port`: `18001`
- `config.sglang_python`: `/home/undefined/Disk/python-envs/sglang/bin/python`
- `config.vllm_cli`: `/home/undefined/Disk/python-envs/vllm/bin/vllm`
- `config.vllm_gpu_memory_fraction`: `0.7`
- `config.vllm_port`: `18000`
- `config.vllm_python`: `/home/undefined/Disk/python-envs/vllm/bin/python`
- `config.warmup_requests`: `8`
- `created_at`: `2026-07-15T02:29:45.133+00:00`
- `environment.captured_at`: `2026-07-15T02:29:45.134+00:00`
- `environment.cuda_compiler.command`: `/usr/local/cuda-13.0/bin/nvcc, --version`
- `environment.cuda_compiler.output`: `nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0`
- `environment.cuda_compiler.returncode`: `0`
- `environment.git_commit.command`: `git, rev-parse, HEAD`
- `environment.git_commit.output`: `36df5d75b6fa01138a44ec573c98843c046618e5`
- `environment.git_commit.returncode`: `0`
- `environment.git_status_at_start.command`: `git, status, --short`
- `environment.git_status_at_start.output`: `?? ./`
- `environment.git_status_at_start.returncode`: `0`
- `environment.gpu.command`: `nvidia-smi, --query-gpu=name,uuid,driver_version,memory.total,compute_cap, --format=csv,noheader,nounits`
- `environment.gpu.output`: `NVIDIA GeForce RTX 4070 SUPER, GPU-e8366400-0166-5f13-aaac-47dbad44e029, 580.159.03, 12282, 8.9`
- `environment.gpu.returncode`: `0`
- `environment.machine`: `x86_64`
- `environment.platform`: `Linux-6.14.0-36-generic-x86_64-with-glibc2.39`
- `environment.sglang_environment.command`: `/home/undefined/Disk/python-envs/sglang/bin/python, -c, import json,platform; d={'python':platform.python_version()}; import torch; d.update(torch=torch.__version__,cuda=torch.version.cuda); 
try:
 import vllm; d['vllm']=vllm.__version__
except Exception: pass
try:
 import sglang; d['sglang']=getattr(sglang,'__version__','unknown')
except Exception: pass
print(json.dumps(d,sort_keys=True))`
- `environment.sglang_environment.output`: `{"cuda": "13.0", "python": "3.12.3", "sglang": "0.5.9", "torch": "2.9.1+cu130"}`
- `environment.sglang_environment.returncode`: `0`
- `environment.vllm_environment.command`: `/home/undefined/Disk/python-envs/vllm/bin/python, -c, import json,platform; d={'python':platform.python_version()}; import torch; d.update(torch=torch.__version__,cuda=torch.version.cuda); 
try:
 import vllm; d['vllm']=vllm.__version__
except Exception: pass
try:
 import sglang; d['sglang']=getattr(sglang,'__version__','unknown')
except Exception: pass
print(json.dumps(d,sort_keys=True))`
- `environment.vllm_environment.output`: `{"cuda": "13.0", "python": "3.12.3", "torch": "2.11.0+cu130", "vllm": "0.11.2.dev0+local"}`
- `environment.vllm_environment.returncode`: `0`
- `experiment_id`: `qwen3-0.6b-vllm-sglang-c1-c8-c32-20260715`
- `protocol.api`: `OpenAI-compatible streaming /v1/completions`
- `protocol.client`: `vllm bench serve`
- `protocol.load_model`: `closed-loop saturation; request-rate=inf with client max-concurrency`
- `protocol.metrics_note`: `ready-check is disabled during measured commands; deltas cover measured requests only`
- `protocol.percentiles`: `50, 90, 95, 99`
- `protocol.prefix_cache`: `disabled on both engines`
- `protocol.sampling`: `temperature=0, top_p=1, ignore_eos=true`
- `protocol.warmup`: `separate unmeasured run before each measured cell`

## 数据校验

- 状态：通过
- 没有错误或警告。

## 证据索引

每个文件的 SHA-256 固定了本报告所引用的字节内容。客户端 raw JSON 是延迟结论的主证据；metrics JSON 是排队、运行请求数和 preemption/retraction 的辅助因果证据；server log 固定了实际生效配置、KV 容量与 CUDA Graph 记录。

| Evidence ID | 类型 | 引擎 / 并发 / 重复 | 路径 | SHA-256 |
|---|---|---|---|---|
| MANIFEST | manifest | - / - / - | `manifest.json` | `8b1b9bab4c3a65f316833e0d88c9c39eb53b8363b820d257fb6322953bee026a` |
| WORKLOAD | exact_token_workload | - / - / - | `workload/qwen_exact_tokens.json` | `267867634e9fb770bc4bf27f681768dd4fdcb950545afb53d58bb222af765e8d` |
| RAW-VLLM-C1-R1 | client_raw | vllm / 1 / 1 | `raw/vllm-c1-r1.json` | `639d544c1cca984f6701b7553bff0eca7d7c7f3732c6da3ebb80a1b3897956c7` |
| METRICS-VLLM-C1-R1 | server_metrics | vllm / 1 / 1 | `metrics/vllm-c1-r1.json` | `1afc66eede1464f46b56bdc482066eb1878e33c6b58ff0e377408c2b3353beec` |
| RAW-VLLM-C8-R1 | client_raw | vllm / 8 / 1 | `raw/vllm-c8-r1.json` | `855e9896a907a4e0d619dc65839fc1c6d756cce8d83fc859b8b95a0cd1f1a8f6` |
| METRICS-VLLM-C8-R1 | server_metrics | vllm / 8 / 1 | `metrics/vllm-c8-r1.json` | `33da664bc904fd9bf3ebba689a3fec2b89023d4c942916eb1a8a0f483039d9e8` |
| RAW-VLLM-C32-R1 | client_raw | vllm / 32 / 1 | `raw/vllm-c32-r1.json` | `36aea559700af7bf9b7366b8a520a66b866f455b446cfa107ee01457a21b3169` |
| METRICS-VLLM-C32-R1 | server_metrics | vllm / 32 / 1 | `metrics/vllm-c32-r1.json` | `ea51ea2b7fa62e407d12362f1333e5bed324e6769cc6eedb08662a7405bd8d23` |
| RAW-VLLM-C8-R2 | client_raw | vllm / 8 / 2 | `raw/vllm-c8-r2.json` | `4e17acdfa686d19eab045de8d3555872891c23dedec3dc5310cb3a39bb4b352b` |
| METRICS-VLLM-C8-R2 | server_metrics | vllm / 8 / 2 | `metrics/vllm-c8-r2.json` | `2fbce12939e508565693c0d0aac1c951e4fb34059820ff50db703d28aa0d54d0` |
| RAW-VLLM-C32-R2 | client_raw | vllm / 32 / 2 | `raw/vllm-c32-r2.json` | `e942cb328eab13355ae880e50db2e0da083c971c75439bfe6e396955332547f7` |
| METRICS-VLLM-C32-R2 | server_metrics | vllm / 32 / 2 | `metrics/vllm-c32-r2.json` | `eadfe76cf9c8398bc96a30739c4e3a3b176b7d0521303b3e91f2aed516a8702b` |
| RAW-VLLM-C1-R2 | client_raw | vllm / 1 / 2 | `raw/vllm-c1-r2.json` | `c735f4e07b5b42378b57ce188771acc575c1938989bacaea3dc4d9a1c0ca4a41` |
| METRICS-VLLM-C1-R2 | server_metrics | vllm / 1 / 2 | `metrics/vllm-c1-r2.json` | `2f9ecf1026d614cb8c72d5fa23fa7269e647d34e4955dd24081c5a9e3479428e` |
| RAW-VLLM-C32-R3 | client_raw | vllm / 32 / 3 | `raw/vllm-c32-r3.json` | `43d89e4de81a746dd0fe36cadef8a7ed399bbf02a8959925e2214b3018d0fc0a` |
| METRICS-VLLM-C32-R3 | server_metrics | vllm / 32 / 3 | `metrics/vllm-c32-r3.json` | `5a4a43fbb441a8a27b7e919dcb71c3de41686f6fbd68e1f3ebce0c4707d27d97` |
| RAW-VLLM-C1-R3 | client_raw | vllm / 1 / 3 | `raw/vllm-c1-r3.json` | `f5a7d4d7eb0b2bd3335db42e1f8b9dd4319d4c7ba1c6698f2fcac67f8ee40089` |
| METRICS-VLLM-C1-R3 | server_metrics | vllm / 1 / 3 | `metrics/vllm-c1-r3.json` | `df4a1c6642e136d29fce3eb33c335188dd6bb336c7223f7871679de7b03a9f2a` |
| RAW-VLLM-C8-R3 | client_raw | vllm / 8 / 3 | `raw/vllm-c8-r3.json` | `d2b828b716212b56fabbfcdcc7cc637dbdedf150da69d840506edef5ece63508` |
| METRICS-VLLM-C8-R3 | server_metrics | vllm / 8 / 3 | `metrics/vllm-c8-r3.json` | `f716c0a5493bba7b951440de60567a901615f1de22e22272370fe1a44b8bc3b1` |
| RAW-SGLANG-C1-R1 | client_raw | sglang / 1 / 1 | `raw/sglang-c1-r1.json` | `68fa00032e11dc54501a99ccaa1ab55a04488951adfc1e7e2a6bc6fa8e6e0176` |
| METRICS-SGLANG-C1-R1 | server_metrics | sglang / 1 / 1 | `metrics/sglang-c1-r1.json` | `95b7a20d52f745459b9b8e47169fae43c16778d3b654813644918013517c4b20` |
| RAW-SGLANG-C8-R1 | client_raw | sglang / 8 / 1 | `raw/sglang-c8-r1.json` | `5278547f8c21e76e92227e5da9f57035167527159708438b323049110d6ee5ab` |
| METRICS-SGLANG-C8-R1 | server_metrics | sglang / 8 / 1 | `metrics/sglang-c8-r1.json` | `3715ec4bd0c217488fec35c71ecc4688db93cf932025963a7a8d23a812abc0e7` |
| RAW-SGLANG-C32-R1 | client_raw | sglang / 32 / 1 | `raw/sglang-c32-r1.json` | `2ffaf103bdea91ef85bd1a1978fe666fba61f7da8b799f9ad98deba0504980a6` |
| METRICS-SGLANG-C32-R1 | server_metrics | sglang / 32 / 1 | `metrics/sglang-c32-r1.json` | `0922daf509ccfea7726974df185bf2de7aeea5367f07f40682b9437d96220e69` |
| RAW-SGLANG-C8-R2 | client_raw | sglang / 8 / 2 | `raw/sglang-c8-r2.json` | `b6e1df1892f2762df52d3886e56c5c4c1637d8201244aae6e2a3e491e162c302` |
| METRICS-SGLANG-C8-R2 | server_metrics | sglang / 8 / 2 | `metrics/sglang-c8-r2.json` | `88058200177df524c59b6b66c6f2303f7bf1fddbb4218a1e240872ed88115ea2` |
| RAW-SGLANG-C32-R2 | client_raw | sglang / 32 / 2 | `raw/sglang-c32-r2.json` | `42143a3d109c63103c4b93cf11d442754b24cf1f4aa4cea56e8dbc9fb0e28734` |
| METRICS-SGLANG-C32-R2 | server_metrics | sglang / 32 / 2 | `metrics/sglang-c32-r2.json` | `8e2593b2b78e8248baca5e7457257e77f445e2f902c0c151cedacd67af67e2f3` |
| RAW-SGLANG-C1-R2 | client_raw | sglang / 1 / 2 | `raw/sglang-c1-r2.json` | `c6061c3961bac517b8f58e062708aa4378a23de0a030134931592ea9065184b8` |
| METRICS-SGLANG-C1-R2 | server_metrics | sglang / 1 / 2 | `metrics/sglang-c1-r2.json` | `8e34f01ce44d2480db95a18e1632a37bc004a258f76ccc340b9922d2ce4462d3` |
| RAW-SGLANG-C32-R3 | client_raw | sglang / 32 / 3 | `raw/sglang-c32-r3.json` | `3eea6c39300d20da062a80ae2f7b5f14fdb217be69b667532951ca24d2552feb` |
| METRICS-SGLANG-C32-R3 | server_metrics | sglang / 32 / 3 | `metrics/sglang-c32-r3.json` | `57e4a08f50916ad07f3c265828c40111047b2f88c4364582d652d04fc0a0e05d` |
| RAW-SGLANG-C1-R3 | client_raw | sglang / 1 / 3 | `raw/sglang-c1-r3.json` | `8c43349b855251b79914213303d63279c24dd863e5f3f0b78c6ef4b618ab8fd3` |
| METRICS-SGLANG-C1-R3 | server_metrics | sglang / 1 / 3 | `metrics/sglang-c1-r3.json` | `8244feb459042a1a9854986ab1cebaf61c92fe363349c206cb704ba5b32616a5` |
| RAW-SGLANG-C8-R3 | client_raw | sglang / 8 / 3 | `raw/sglang-c8-r3.json` | `b18e89ea153ca73b86e1447700a8dfa142d2eb51c9099ecb66d36f274d3260b6` |
| METRICS-SGLANG-C8-R3 | server_metrics | sglang / 8 / 3 | `metrics/sglang-c8-r3.json` | `0d487eb8e887a44ffde267255b07ec8e476b3702b2e62c25951d19bb1ea7dcf0` |
| SERVER-SGLANG | server_log | sglang / - / - | `server/sglang.log` | `bce42ccdf70005bd4f244987214b668fcb7246eaeb533cb20ed034d43a6ee2f7` |
| SERVER-VLLM | server_log | vllm / - / - | `server/vllm.log` | `2e06180fe6b9f0f84ad4b04142e2007f4a6988833f6230df08d53e1502586a3c` |

## 局限性

- 这是固定输入/输出长度、无限 request-rate 的饱和压测；它回答容量边界问题，不代表生产到达过程。
- 同机客户端减少网络噪声，但没有覆盖真实跨机网络和网关开销。
- 单 GPU、单模型和当前软件版本的结论不能直接外推到其他模型、量化方式或硬件。
- Prometheus 轮询是离散采样；短于采样间隔的瞬时队列可能未被 gauge 最大值捕获。
- 观察相关性不能单独证明唯一因果；报告已把服务端直接指标与架构推断分开。
