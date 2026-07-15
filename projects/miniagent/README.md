# MiniAgent：用证据回答 LLM Serving 性能问题

MiniAgent 的第一个可运行工具用于比较同一个 Qwen 模型在 vLLM 和
SGLang 上的服务性能。它不让 LLM 计算统计量或猜根因：实验编排、校验、
百分位计算和证据索引全部是确定性代码；未来再把这套 workflow 暴露成
Agent Tool，由 LLM 负责选择工具、追问和解释已编号的 evidence。

## 已实现的实验

Canonical suite 固定如下：

| 项目 | 配置 |
|---|---|
| 模型 | 本地 `Qwen3-0.6B`，BF16，TP=1 |
| API / client | 同一个 `vllm bench serve` client，OpenAI `/v1/completions` SSE |
| 输入 / 输出 | 每请求严格 1024 / 128 token |
| 负载 | `request-rate=inf` 的 closed-loop saturation |
| 并发 | 1、8、32 |
| 样本 | 每格 256 请求，3 次 repetition，共 4608 个 measured requests |
| 调度上限 | 最多 32 个 running requests，prefill/batched token budget=2048 |
| 缓存 | 两边均关闭 prefix/radix cache |
| 采样 | temperature=0、top-p=1、ignore-EOS |
| 证据 | 请求级 TTFT/ITL、服务端 Prometheus、GPU 时序、命令、日志、SHA-256 |

`request-rate=inf + max-concurrency=C` 表示客户端最多保持 C 个在途请求；
它测的是饱和容量，不是生产环境的 Poisson 到达流量。

## 本机正式结果（2026-07-15）

环境是 RTX 4070 SUPER 12 GB、Qwen3-0.6B、vLLM
`0.11.2.dev0+local`、SGLang `0.5.9`。18 个 cell 共 4608 个 measured
requests 全部成功。

| 引擎 | 并发 | pooled p95 TTFT (ms) | p95 TPOT (ms/token) | p95 E2E (ms) | 输出吞吐 (token/s) |
|---|---:|---:|---:|---:|---:|
| vLLM | 1 | 24.25 | 3.71 | 495.01 | 268.16 |
| vLLM | 8 | 105.95 | 6.84 | 924.50 | 1143.50 |
| vLLM | 32 | 449.46 | 16.97 | 2499.89 | 1835.90 |
| SGLang | 1 | 26.17 | 3.59 | 481.00 | 276.48 |
| SGLang | 8 | 158.11 | 6.78 | 916.51 | 1155.12 |
| SGLang | 32 | 608.40 | 16.97 | 2265.14 | 1845.88 |

本机这组 workload 下，两边输出吞吐只差约 0.5%–3.1%，vLLM 的 p95 TTFT
在三档都更低；SGLang 的 p95 E2E 则略低，c32 时低约 9.4%。这说明“首 token
响应快”和“整条请求完成快”不是同一个排名。

完整数字、逐阶根因分析和 SHA-256 证据索引见
[正式报告](reports/qwen3-0.6b-vllm-sglang-c1-c8-c32-20260715.md)。

## 为什么不用 random dataset

当前两个注册环境分别加载 `Qwen2Tokenizer` 与 `Qwen2TokenizerFast`。
直接让 client 随机生成 token 再解码，服务端重新编码后可能不再是同一长度。
MiniAgent 只生成一次 ShareGPT JSON，其中每条 prompt 都用 `" x"` 补齐，随后
在两个环境中对全部记录重新 tokenize；只要任意一边不是严格 1024 token，
实验就会在启动服务前失败。ShareGPT carrier 还避开了 vLLM `custom JSONL`
loader 对可选 pandas benchmark extra 的依赖。

## 运行

先查看完整命令，不启动 GPU 服务：

```bash
cd /home/undefined/Desktop/ai/projects/miniagent
PYTHONPATH=. /home/undefined/Disk/python-envs/ai-tools-py312/bin/python \
  -m miniagent benchmark qwen-compare --dry-run
```

短集成冒烟：

```bash
PYTHONPATH=. /home/undefined/Disk/python-envs/ai-tools-py312/bin/python \
  -m miniagent benchmark qwen-compare --smoke
```

完整实验（默认参数就是 canonical suite）：

```bash
PYTHONPATH=. /home/undefined/Disk/python-envs/ai-tools-py312/bin/python \
  -m miniagent benchmark qwen-compare
```

从已有 raw evidence 重新生成确定性报告：

```bash
PYTHONPATH=. /home/undefined/Disk/python-envs/ai-tools-py312/bin/python \
  -m miniagent report artifacts/runs/<run-id>
```

MiniAgent 本身只有 Python 标准库依赖。服务和共同 benchmark client 分别复用：

- `/home/undefined/Disk/python-envs/vllm`（本地 editable vLLM）
- `/home/undefined/Disk/python-envs/sglang`（SGLang）

不会创建新虚拟环境，也不会改动共享环境。

## 输出与证据链

每次运行生成：

```text
artifacts/runs/<run-id>/
├── manifest.json          # 环境、精确 argv、公平性配置、运行状态
├── workload/              # 一次生成、双 tokenizer 全量验证的数据
├── raw/                   # vllm bench 的逐请求原始数组
├── metrics/               # Prometheus + nvidia-smi 原始时序与摘要
├── logs/                  # 每个 warmup / measured client 的输出
├── server/                # 两个服务的完整启动与调度日志
├── summary.json           # 从 raw 重新计算的确定性统计
└── report.md              # 中文证据报告
```

报告中的 pooled p95 从三次 repetition 的所有逐请求 TTFT 重新计算；同时单独
给出三次 run-level p95 的中位数和 `[min,max]`，不会平均三个 p95 冒充总体
p95。吞吐则使用 `总 token / 总 duration`，不是简单平均每次吞吐。

TTFT 是从客户端发出请求到收到第一个非空流式 token 的时间，包含同机 HTTP、
前端处理、服务端排队、tokenization/prefill 和首 token decode。诊断遵循：

- `max_waiting > 0` 或 mean queue time 至少 1 ms：排队是直接证据；
- queue 很小而 server prefill time 上升：prefill batching/争用有直接支持；
- TTFT 与 TPOT 同涨且 GPU 利用率高：与 GPU 计算竞争一致；
- KV/token usage 接近 100% 且出现 preemption/retraction：KV 容量压力；
- 只有客户端延迟而缺少对应服务端指标时，只报告候选解释，不宣称根因。

## 测试

```bash
cd /home/undefined/Desktop/ai/projects/miniagent
PYTHONPATH=. /home/undefined/Disk/python-envs/ai-tools-py312/bin/python \
  -m unittest discover -s tests -v
```

单元测试不启动 GPU，覆盖线性百分位、E2E/TPOT 重建、weighted throughput、
Prometheus histogram delta、证据报告措辞以及两套命令的公平性约束。
