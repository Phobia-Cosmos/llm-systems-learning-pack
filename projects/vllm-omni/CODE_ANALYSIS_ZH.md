# vLLM-Omni 代码分析与论文知识点映射

> 本文是对本地仓库 `/home/undefined/Desktop/ai/vllm-omni` 的静态代码阅读记录，重点对应论文概念、核心代码路径和后续学习路线。

## 0. 元信息

- 仓库：<https://github.com/vllm-project/vllm-omni>
- 本地路径：`/home/undefined/Desktop/ai/vllm-omni`
- 当前 commit：`c39ac9f [Bugfix] Stream Qwen3-TTS WebSocket input as one request (#4731)`
- 论文：`vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models`
- arXiv：<https://arxiv.org/abs/2602.02204>
- 论文引用入口也写在 [README.md](README.md) 第 80 行附近。

## 1. 一句话架构

vLLM-Omni 可以理解为：

> 在 vLLM 的高性能 AR/LLM 推理能力上，加入多模态 stage graph、DiT/diffusion runtime、OmniConnector 跨 stage 数据传输、stage 级资源隔离和并行，从而服务 any-to-any multimodal models。

README 中给出的定位很清楚：

- [README.md](README.md) 第 32-35 行：vLLM 原本主要服务文本 AR 生成，vLLM-Omni 扩展到 omni-modality 与 non-autoregressive/DiT 架构。
- [README.md](README.md) 第 46-54 行：核心性能来源包括 vLLM KV cache、pipeline overlap、fully disaggregation、parallelism。
- [README.md](README.md) 第 80-86 行：论文引用为 `arXiv:2602.02204`。

## 2. 目录地图

| 目录/文件 | 作用 | 阅读重点 |
| --- | --- | --- |
| `vllm_omni/engine/` | 多 stage 总控运行时 | `AsyncOmniEngine`、`Orchestrator`、`StagePool`、`StageRuntime` |
| `vllm_omni/core/sched/` | AR/generation scheduler 扩展 | 继承 vLLM scheduler，加入 omni payload、chunk/full-payload 输入协调、KV transfer 标记 |
| `vllm_omni/distributed/omni_connectors/` | 跨 stage 传输抽象 | `OmniConnectorBase`、factory、KV transfer、chunk transfer |
| `vllm_omni/diffusion/` | DiT/diffusion 推理 runtime | `DiffusionEngine`、request/step scheduler、model runner、parallelism/cache |
| `vllm_omni/config/` | pipeline/stage 配置 | deploy YAML、stage schema、pipeline registry |
| `vllm_omni/model_executor/` | 具体模型和 stage input processor | Qwen/BAGEL/TTS 等模型如何把一个 stage 输出转成下一个 stage 输入 |
| `docs/design/` | 设计文档 | AR module、DiT module、async chunk、disaggregated inference、prefix caching |
| `docs/user_guide/examples/` | 用户例子 | Qwen3-Omni、BAGEL 等端到端 topology |
| `vllm_omni/deploy/` | 内置部署拓扑 | stage 划分、connector、TP/DP/PP、async_chunk |

## 3. 端到端请求链路

典型请求路径如下：

```text
Client / OpenAI-compatible API
  -> AsyncOmni / AsyncOmniEngine
  -> Orchestrator
  -> StagePool.submit_initial()
  -> 某个 stage runtime / replica
  -> AR LLM engine 或 DiffusionEngine
  -> 输出回 Orchestrator
  -> 输出给前端，或转发到下一个 stage
  -> OmniConnector / KV transfer / chunk transfer
  -> 最终 streaming/final response
```

关键代码：

- [vllm_omni/engine/async_omni_engine.py](vllm_omni/engine/async_omni_engine.py) 第 187 行：`AsyncOmniEngine`。
- [vllm_omni/engine/async_omni_engine.py](vllm_omni/engine/async_omni_engine.py) 第 342 行：`_initialize_stages()` 初始化 stage runtime、stage pools、processors。
- [vllm_omni/engine/async_omni_engine.py](vllm_omni/engine/async_omni_engine.py) 第 400 行：`_bootstrap_orchestrator()` 构造 orchestrator。
- [vllm_omni/engine/orchestrator.py](vllm_omni/engine/orchestrator.py) 第 424 行：`_handle_add_request()` 接收新请求，向 stage 0 提交。
- [vllm_omni/engine/stage_pool.py](vllm_omni/engine/stage_pool.py) 第 902 行：`submit_initial()` 选择 replica 并提交初始请求。
- [vllm_omni/engine/stage_pool.py](vllm_omni/engine/stage_pool.py) 第 969 行：`submit_update()` 给已经存在的 downstream request 追加 chunk/update。

## 4. 核心代码精读

### 4.1 AsyncOmniEngine：对外入口和后台 orchestrator

`AsyncOmniEngine` 是外部 API 和内部多 stage runtime 之间的桥。

重点：

- 它不是每一步都自己执行模型，而是启动和管理后台 `Orchestrator`。
- stage 初始化、processor 构建、runtime 创建都在这里完成。
- 对用户侧暴露 `add_request()` / `add_request_async()` 等 API。

对应代码：

- [vllm_omni/engine/async_omni_engine.py](vllm_omni/engine/async_omni_engine.py) 第 187 行：类定义。
- [vllm_omni/engine/async_omni_engine.py](vllm_omni/engine/async_omni_engine.py) 第 342-430 行：初始化 stages 与 orchestrator。
- [vllm_omni/engine/async_omni_engine.py](vllm_omni/engine/async_omni_engine.py) 第 1238 行：同步 `add_request()`。
- [vllm_omni/engine/async_omni_engine.py](vllm_omni/engine/async_omni_engine.py) 第 1290 行：异步 `add_request_async()`。

论文关联：

- 对应论文里的 serving framework/runtime 层。
- 它把用户请求变成内部 stage graph request，而不是单模型单进程调用。

### 4.2 Orchestrator：多 stage 图的调度中枢

`Orchestrator` 是这套系统最值得精读的文件之一。它负责：

- 接收新请求。
- 维护 request state。
- 决定某个 stage 的输出是否应该返回给前端。
- 决定是否 forward 到下一个 stage。
- 处理 async_chunk 的下游预热。
- 处理 PD disaggregation、CFG companion、KV sender info。

关键入口：

- [vllm_omni/engine/orchestrator.py](vllm_omni/engine/orchestrator.py) 第 424 行：`_handle_add_request()`，创建 `OrchestratorRequestState` 并提交 stage 0。
- [vllm_omni/engine/orchestrator.py](vllm_omni/engine/orchestrator.py) 第 861 行附近：`_route_output()`，处理某个 stage 的输出。
- [vllm_omni/engine/orchestrator.py](vllm_omni/engine/orchestrator.py) 第 1158 行：`_forward_to_next_stage()`，把上游输出转成下游输入。
- [vllm_omni/engine/orchestrator.py](vllm_omni/engine/orchestrator.py) 第 1419 行：`_prewarm_async_chunk_stages()`，async_chunk 模式下提前启动下游 stages。
- [vllm_omni/engine/orchestrator.py](vllm_omni/engine/orchestrator.py) 第 1505 行：`_build_kv_sender_info()`，为 diffusion KV receiver 生成 sender 信息。

特别重要的逻辑：

- 非 async_chunk 模式下，`_route_output()` 通常等上游 stage finished 或 segment finished 后再 forward。
- async_chunk 模式下，下游 stage 会提前预提交，后续 chunk 通过 connector 到达，不需要每个 chunk 都绕 orchestrator。
- AR -> diffusion 的桥接一般通过 `custom_process_input_func` 完成，它把 AR hidden states、token ids、KV metadata 等整理成 DiT stage 可以理解的 prompt。

论文关联：

- `fully disaggregated serving` 的核心落点。
- `stage graph`、`heterogeneous stages`、`dynamic resource allocation` 都需要 orchestrator 维护状态和转发边。

### 4.3 StagePool / StageRuntime：stage 副本和进程边界

`StagePool` 是一个逻辑 stage 的提交入口；`StageRuntime` 负责真实 runtime/进程/worker 的生命周期。

关键代码：

- [vllm_omni/engine/stage_pool.py](vllm_omni/engine/stage_pool.py) 第 47 行：`StagePool`。
- [vllm_omni/engine/stage_pool.py](vllm_omni/engine/stage_pool.py) 第 902 行：`submit_initial()`。
- [vllm_omni/engine/stage_pool.py](vllm_omni/engine/stage_pool.py) 第 969 行：`submit_update()`。
- [vllm_omni/engine/stage_runtime.py](vllm_omni/engine/stage_runtime.py) 第 112 行：`StageRuntime`。
- [vllm_omni/engine/stage_runtime.py](vllm_omni/engine/stage_runtime.py) 第 713 行：`DistStageRuntime`。
- [vllm_omni/engine/stage_runtime.py](vllm_omni/engine/stage_runtime.py) 第 1059 行：`create_stage_runtime()`。

设计意义：

- 一个 stage 可以是 LLM/AR，也可以是 diffusion，也可以是其他 generation engine。
- 一个 stage 可以有 replica，用于并发或负载均衡。
- stage 可以 colocated，也可以是独立进程/节点。

论文关联：

- stage-level resource allocation。
- stage-level disaggregation。
- heterogeneous engine composition。

### 4.4 AR 模块：继承 vLLM，但暴露 hidden states 和多模态 payload

AR 模块的设计文档在 [docs/design/module/ar_module.md](docs/design/module/ar_module.md)。

重点：

- 第 5 行：AR stage 用于 text、CoT、audio latent tokens 等顺序生成。
- 第 19 行：AR module 基于 vLLM，通过继承保持 scheduling、batching、KV cache 管理兼容。
- 第 101-105 行：Scheduler、ModelRunner、OutputProcessor 都是 vLLM 基础上的 omni 扩展。
- 第 371-387 行：核心差异包括 hidden state exposure、多模态 routing、minimal API changes。

核心代码：

- [vllm_omni/core/sched/omni_ar_scheduler.py](vllm_omni/core/sched/omni_ar_scheduler.py) 第 803 行：`_mark_request_for_kv_transfer()`。
- [vllm_omni/core/sched/omni_ar_scheduler.py](vllm_omni/core/sched/omni_ar_scheduler.py) 第 846 行：`_should_transfer_kv_for_request()`。
- [vllm_omni/core/sched/omni_ar_scheduler.py](vllm_omni/core/sched/omni_ar_scheduler.py) 第 887 行：`get_finished_requests_needing_kv_transfer()`。
- [vllm_omni/core/sched/omni_generation_scheduler.py](vllm_omni/core/sched/omni_generation_scheduler.py) 第 42 行：`OmniGenerationScheduler`。
- [vllm_omni/core/sched/omni_generation_scheduler.py](vllm_omni/core/sched/omni_generation_scheduler.py) 第 59 行：`schedule()`。
- [vllm_omni/core/sched/omni_generation_scheduler.py](vllm_omni/core/sched/omni_generation_scheduler.py) 第 122 和 182 行：通过 `kv_cache_manager.allocate_slots()` 分配 KV/cache slots。

设计意义：

- vLLM 原本的强项是 LLM token generation：continuous batching、KV cache、PagedAttention、sampling。
- vLLM-Omni 保留这套能力，同时让 AR stage 输出 hidden states、多模态张量、KV 元数据，供下游 stage 使用。

论文关联：

- AR extension。
- vLLM inheritance。
- KV cache management inherited from vLLM。

相关论文/知识点：

- vLLM / PagedAttention：<https://arxiv.org/abs/2309.06180>
- FlashAttention：<https://arxiv.org/abs/2205.14135>
- FlashAttention-2：<https://arxiv.org/abs/2307.08691>

### 4.5 OmniConnector：stage 间传输抽象

connector 是 vLLM-Omni disaggregation 的基础设施。

核心代码：

- [vllm_omni/distributed/omni_connectors/connectors/base.py](vllm_omni/distributed/omni_connectors/connectors/base.py) 第 12 行：`OmniConnectorBase`。
- [vllm_omni/distributed/omni_connectors/connectors/base.py](vllm_omni/distributed/omni_connectors/connectors/base.py) 第 21-52 行：抽象 `put()` / `get()`。
- [vllm_omni/distributed/omni_connectors/connectors/base.py](vllm_omni/distributed/omni_connectors/connectors/base.py) 第 59-69 行：`cleanup()` / `health()` / `close()`。
- [vllm_omni/distributed/omni_connectors/factory.py](vllm_omni/distributed/omni_connectors/factory.py) 第 24 行：`OmniConnectorFactory`。
- [vllm_omni/distributed/omni_connectors/factory.py](vllm_omni/distributed/omni_connectors/factory.py) 第 129-136 行：注册 `MooncakeStoreConnector`、`MooncakeTransferEngineConnector`、`SharedMemoryConnector`、`YuanrongConnector`、`MoriTransferEngineConnector` 等。

KV transfer：

- [vllm_omni/distributed/omni_connectors/kv_transfer_manager.py](vllm_omni/distributed/omni_connectors/kv_transfer_manager.py) 第 89 行：`OmniKVCacheConfig`，包含 `need_recv_cache`、`need_send_cache`、`from_stage`、`to_stage` 等。
- [vllm_omni/distributed/omni_connectors/kv_transfer_manager.py](vllm_omni/distributed/omni_connectors/kv_transfer_manager.py) 第 108 行：`KVCacheTransferData`，负责 KV 数据打包。

chunk transfer：

- [vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py](vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py) 第 210 行附近：接收 chunk 后更新 request prompt/additional_information。
- [vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py](vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py) 第 294 行附近：发送单个 request chunk。
- [vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py](vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py) 第 443 行附近：处理 pending chunks。

论文关联：

- unified connector。
- inter-stage transfer。
- local shared memory / cross-node transfer。
- KV cache transfer from AR stage to DiT stage。

### 4.6 Async Chunk：跨 stage 流式重叠

设计文档：[docs/design/feature/async_chunk.md](docs/design/feature/async_chunk.md)。

关键点：

- 第 13 行：`async_chunk` 允许多 stage pipeline 按 chunk 异步处理，而不是等完整上游输出。
- 第 21-23 行：典型 Qwen3-Omni 链路是 Thinker -> Talker -> Code2Wav。
- 第 30 行：chunk IO 通过后台线程与 compute 重叠。
- 第 167-178 行：`OmniConnector` 只做 inter-stage data transport，`OmniChunkTransferAdapter` 负责 background recv/save loop。

用户例子：

- [docs/user_guide/examples/offline_inference/qwen3_omni.md](docs/user_guide/examples/offline_inference/qwen3_omni.md) 第 69-82 行：下游 Talker/Code2Wav 可以在 Thinker 结束前启动，chunk 直接通过 connector 在 stage workers 之间传输，不经过 orchestrator。

代码落点：

- orchestrator 预提交下游 stage：[vllm_omni/engine/orchestrator.py](vllm_omni/engine/orchestrator.py) 第 1419 行。
- scheduler 处理 pending chunk：[vllm_omni/core/sched/omni_generation_scheduler.py](vllm_omni/core/sched/omni_generation_scheduler.py) 第 46-57 行、第 59 行。
- chunk adapter 负责异步 get/put：[vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py](vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py)。

论文关联：

- pipeline overlap。
- stage-level concurrency。
- low-latency streaming multimodal output。

### 4.7 Diffusion / DiT runtime：非自回归生成

设计文档：[docs/design/module/dit_module.md](docs/design/module/dit_module.md)。

重点：

- 第 49-55 行：`DiffusionEngine` 是 diffusion 推理系统的 orchestrator。
- 第 134 行：scheduler 管 request lifecycle 和 scheduling decisions。
- 第 218 行：`RequestScheduler` request-mode 一次 dispatch 完成完整 request，因此当前通常单 active request。
- 第 595-648 行：Ring/Ulysses 是 sequence parallel 的通信模式，不是单纯 attention kernel。
- 第 821 行附近：支持 DP/TP/PP/CFG/SP 等并行策略。

核心代码：

- [vllm_omni/diffusion/diffusion_engine.py](vllm_omni/diffusion/diffusion_engine.py) 第 101 行：`DiffusionEngine`。
- [vllm_omni/diffusion/diffusion_engine.py](vllm_omni/diffusion/diffusion_engine.py) 第 138 行：按 `step_execution` 选择 `StepScheduler` 或 `RequestScheduler`。
- [vllm_omni/diffusion/diffusion_engine.py](vllm_omni/diffusion/diffusion_engine.py) 第 163-165 行：选择 `execute_step` 或 `execute_request`。
- [vllm_omni/diffusion/sched/base_scheduler.py](vllm_omni/diffusion/sched/base_scheduler.py) 第 43 行：`_BaseScheduler`。
- [vllm_omni/diffusion/sched/base_scheduler.py](vllm_omni/diffusion/sched/base_scheduler.py) 第 86 行：`schedule()`。
- [vllm_omni/diffusion/sched/request_scheduler.py](vllm_omni/diffusion/sched/request_scheduler.py) 第 19 行：`RequestScheduler`。
- [vllm_omni/diffusion/sched/step_scheduler.py](vllm_omni/diffusion/sched/step_scheduler.py) 第 30 行：`StepScheduler`。
- [vllm_omni/diffusion/worker/diffusion_model_runner.py](vllm_omni/diffusion/worker/diffusion_model_runner.py) 第 286 行：`execute_model()`。
- [vllm_omni/diffusion/worker/diffusion_model_runner.py](vllm_omni/diffusion/worker/diffusion_model_runner.py) 第 319 行：接收 AR KV cache。
- [vllm_omni/diffusion/worker/diffusion_model_runner.py](vllm_omni/diffusion/worker/diffusion_model_runner.py) 第 509 行：`execute_stepwise()`。
- [vllm_omni/diffusion/worker/diffusion_model_runner.py](vllm_omni/diffusion/worker/diffusion_model_runner.py) 第 532 行：调用 `pipeline.denoise_step()`。

论文关联：

- non-autoregressive architectures。
- Diffusion Transformers。
- heterogeneous output modalities：image/video/audio 等。
- diffusion runtime 不同于 AR token scheduler，需要 request/denoise step 级调度。

相关论文/知识点：

- DiT：<https://arxiv.org/abs/2212.09748>
- DDPM：<https://arxiv.org/abs/2006.11239>
- Classifier-Free Guidance：<https://arxiv.org/abs/2207.12598>

### 4.8 Prefix cache：多模态 hidden/mm outputs 的复用

核心代码：

- [vllm_omni/core/prefix_cache.py](vllm_omni/core/prefix_cache.py) 第 33 行：`OmniTensorPrefixCache`。
- [vllm_omni/core/prefix_cache.py](vllm_omni/core/prefix_cache.py) 第 119 行：CPU cache tensor 分配。
- [vllm_omni/core/prefix_cache.py](vllm_omni/core/prefix_cache.py) 第 171 行：非阻塞 GPU -> CPU 拷贝。
- [vllm_omni/core/prefix_cache.py](vllm_omni/core/prefix_cache.py) 第 614 行：合并 prefix cached multimodal states。
- [vllm_omni/core/prefix_cache.py](vllm_omni/core/prefix_cache.py) 第 649 行：合并 hidden states。

设计意义：

- vLLM 的 prefix cache 主要服务 token/KV cache。
- vLLM-Omni 需要额外处理 hidden states、multimodal outputs、audio/image latent 等张量。
- 这里能看到 GPU/CPU staging、async copy、block layout 对齐等工程细节。

论文关联：

- efficient memory/cache management。
- multimodal prefix reuse。

### 4.9 Stage config：论文里的 stage graph 落到 YAML

设计文档：[docs/configuration/stage_configs.md](docs/configuration/stage_configs.md)。

关键点：

- 第 3 行：目标模型被拆成多个 stages，可由 LLMEngines、DiffusionEngines 或其他 engines 处理。
- 第 15-18 行：`async_chunk`、`connectors`、`edges`、`stages` 是 deploy YAML 的核心字段。
- 第 42 行：每个 stage 可单独设置 `tensor_parallel_size`。
- 第 68-69 行：connector schema 支持 `SharedMemoryConnector`、`MooncakeStoreConnector`。
- 第 96-100 行：stage-based CLI 支持 stage 0 API/orchestrator 与 headless worker stages。
- 第 213-218 行：stage config 的主要作用包括 stage partition、disaggregated topology、engine args、input/output dependencies。

这就是论文里 stage-level disaggregation 在代码里的配置入口。

### 4.10 BAGEL/Qwen3-Omni：两个典型拓扑

Qwen3-Omni：

- [docs/user_guide/examples/offline_inference/qwen3_omni.md](docs/user_guide/examples/offline_inference/qwen3_omni.md) 第 69-82 行：async_chunk 让 Talker/Code2Wav 在 Thinker 结束前启动，stage workers 直接通过 connector 传数据。

BAGEL：

- [docs/user_guide/examples/offline_inference/bagel.md](docs/user_guide/examples/offline_inference/bagel.md) 第 11-15 行：BAGEL 有 two-stage 和 single-stage topology。
- [docs/user_guide/examples/offline_inference/bagel.md](docs/user_guide/examples/offline_inference/bagel.md) 第 15 行：two-stage 是 Stage 0 Thinker AR + Stage 1 DiT Diffusion，KV cache 在 stages 间传输。
- [docs/user_guide/examples/offline_inference/bagel.md](docs/user_guide/examples/offline_inference/bagel.md) 第 94 行：Thinker 解码 think tokens 后，把 augmented KV cache 传给 DiT stage。

这两个例子分别适合学习：

- Qwen3-Omni：async_chunk、音频流式输出、stage overlap。
- BAGEL：AR -> DiT、KV transfer、single-stage vs two-stage tradeoff。

## 5. 论文概念 ↔ 代码位置 ↔ 学习知识点

| 论文/知识点 | 代码/文档位置 | 要理解的问题 |
| --- | --- | --- |
| Fully disaggregated serving | `engine/orchestrator.py`、`engine/stage_runtime.py`、`docs/configuration/stage_configs.md` | 为什么要把一个 multimodal model 拆成多个 stage？stage 间如何通信？ |
| Any-to-any multimodal serving | `docs/serving/`、`model_executor/`、`diffusion/models/` | 输入/输出模态如何被 route、serialize、accumulate？ |
| Heterogeneous stage graph | `vllm_omni/deploy/*.yaml`、`config/pipeline_registry.py` | AR、DiT、TTS decoder、vocoder 怎样被放进同一 pipeline？ |
| vLLM AR inheritance | `core/sched/omni_ar_scheduler.py`、`worker/gpu_ar_model_runner.py`、`docs/design/module/ar_module.md` | vLLM 的 scheduler/KV cache/sampling 如何复用？ |
| KV cache transfer | `distributed/omni_connectors/kv_transfer_manager.py`、`core/sched/omni_ar_scheduler.py` | 为什么 AR stage 的 KV 能变成下游 DiT 的条件？block ids 如何标记？ |
| OmniConnector | `distributed/omni_connectors/connectors/base.py`、`factory.py` | 为什么要抽象 shared memory、TCP/RDMA、平台 connector？ |
| Async chunk | `transfer_adapter/chunk_transfer_adapter.py`、`docs/design/feature/async_chunk.md` | 为什么 streaming 不能等上游 stage 全结束？如何减少 TTFP？ |
| Diffusion / DiT runtime | `diffusion/diffusion_engine.py`、`diffusion/sched/`、`diffusion/worker/diffusion_model_runner.py` | 非 AR 生成为什么不能直接套 token scheduler？ |
| Diffusion continuous batching | `diffusion/sched/step_scheduler.py`、`docs/design/feature/diffusion_continuous_batching.md` | 为什么按 denoise step 才更容易 batch 多个 diffusion request？ |
| Prefix caching | `core/prefix_cache.py`、`docs/design/feature/prefix_caching.md` | 多模态 hidden states/mm outputs 如何复用？ |
| Parallelism | `docs/design/module/dit_module.md`、`docs/user_guide/diffusion/parallelism/` | TP/PP/DP/CFG/SP/EP 分别解决什么瓶颈？ |
| MoE / Expert Parallelism | `docs/user_guide/diffusion/parallelism/expert_parallel.md` | sparse MoE 为什么适合 shard experts，而 dense 模型不适合 EP？ |
| OpenAI-compatible serving | `docs/serving/`、`examples/online_serving/` | 怎么把 image/audio/video/chat API 映射到内部 Omni request？ |
| Metrics/production | `docs/design/metrics.md`、`vllm_omni/metrics/` | stage/replica 粒度如何观测 TTFT、TPOT、KV cache、吞吐？ |

## 6. 和你前面几个问题的直接关联

### 6.1 为什么 softmax 计算内存受限？

标准 attention 是：

```text
Q = X W_Q
K = X W_K
V = X W_V
scores = Q K^T / sqrt(d_k)
P = softmax(scores)
O = P V
```

其中：

- `X` 是输入 hidden states，形状通常是 `[batch, seq_len, hidden_dim]`。
- `W_Q/W_K/W_V` 是可学习投影矩阵。
- `Q` 是 query matrix，表示“当前位置想查什么信息”。
- `K` 是 key matrix，表示“每个历史/上下文 token 提供什么索引”。
- `V` 是 value matrix，表示“真正要聚合的信息内容”。

为什么 attention 是 `QK^T`？

- 每个 query 要和所有 key 计算相似度。
- 一个 query 与一个 key 的点积，就是该 query 对该 key/value 的注意力分数。
- 把所有 query 和所有 key 一次性矩阵乘，就是 `QK^T`。

为什么 softmax 常被说成 memory-bound？

- `scores` 形状是 `[batch, heads, seq_len, seq_len]`，序列越长越爆炸。
- softmax 需要读 scores、做 max/reduce/sum、写概率，再给 `PV` 使用。
- 对每个元素的浮点计算不算特别多，但读写中间矩阵很多，长序列时瓶颈更偏显存带宽和中间激活存储。
- FlashAttention 的核心价值就是减少 materialized attention matrix，把 attention 变成 tiled/online softmax，降低 HBM 读写压力。

和 vLLM-Omni 的关系：

- AR stage 继承 vLLM 的高效 attention/KV cache 路径。
- diffusion 模块有自己的 attention backend 和 sequence parallel 设计。
- vLLM-Omni 本身最突出的不是重新发明 attention kernel，而是把 AR/DiT/multimodal stages 组合起来，并处理 KV/cache/connector/parallelism。

### 6.2 为什么需要推理引擎？

如果只有模型 forward，你只能“跑一次模型”。推理引擎要解决的是“高并发、低延迟、可流式、可控资源”的服务问题。

vLLM/vLLM-Omni 这类推理引擎负责：

- 请求队列和调度：waiting/running/pending 状态。
- batching：把多个请求合并到一次 GPU 执行。
- KV cache 管理：减少重复计算，避免显存碎片。
- sampling：从 logits 做 temperature/top-p/top-k 等采样策略。
- streaming：边生成边返回。
- parallelism：TP/PP/DP/EP/CFG/SP 等。
- disaggregation：prefill/decode 或 AR/DiT 不同 stage 拆开部署。
- observability：TTFT、TPOT、吞吐、KV cache 使用率。

vLLM-Omni 的新增点：

- 不只服务文本 LLM。
- 能服务 AR + DiT + TTS/vocoder 等异构 pipeline。
- 能跨 stage 传 hidden states、KV cache、chunked payload。

### 6.3 vLLM 并行和采样在哪里体现？

采样主要在 vLLM AR 路径里，vLLM-Omni 通过继承保留：

- `OmniARScheduler` / `OmniGenerationScheduler` 仍然围绕 vLLM scheduler 输出构造调度结果。
- AR model runner 保持 vLLM 的 execute/sample 分离。
- sampling params 在 stage config 和 request 中传入。

并行主要分三层：

- vLLM inherited：TP/PP/DP、KV cache、continuous batching。
- vLLM-Omni stage-level：不同 stage 使用不同 GPU、不同 memory budget、不同 TP。
- diffusion-specific：CFG parallel、sequence parallel、expert parallel、VAE parallel 等。

### 6.4 MoE sparse 架构和 dense 架构是什么？

Dense 模型：

- 每个 token 基本都会经过同一套 FFN/MLP 参数。
- 参数使用密集，工程路径相对规整。
- tensor parallel 往往对每层都切分。

Sparse MoE 模型：

- 模型有多个 expert。
- router/gate 对每个 token 选择 top-k experts。
- 每个 token 只激活部分 expert，所以是 sparse activation。
- 总参数量可以很大，但单 token 计算量不一定等比例增加。
- 需要 all-to-all dispatch/combine，把 token 发到对应 expert 所在 rank。

vLLM-Omni 的 EP 文档：

- [docs/user_guide/diffusion/parallelism/expert_parallel.md](docs/user_guide/diffusion/parallelism/expert_parallel.md) 第 16 行：EP 只 shard MoE expert MLP blocks，不像 TP 每层都切。
- 第 18 行：forward 时 gate 路由 token 到 expert，需要 all-to-all。
- 第 79 行：dense/no-MoE 模型启用 EP 没有效果。

相关论文/知识点：

- GShard：<https://arxiv.org/abs/2006.16668>
- Switch Transformer：<https://arxiv.org/abs/2101.03961>

## 7. 如果要学习 vLLM / SGLang / vLLM-Omni，建议路径

### Step 1：先把 vLLM 基础补齐

重点不是先看全部代码，而是先理解这几个概念：

- prefill vs decode。
- KV cache。
- PagedAttention。
- continuous batching。
- scheduler waiting/running。
- logits sampling。
- tensor parallel / pipeline parallel。

建议论文：

- vLLM / PagedAttention：<https://arxiv.org/abs/2309.06180>
- FlashAttention：<https://arxiv.org/abs/2205.14135>

### Step 2：读 vLLM-Omni 的 stage graph

阅读顺序：

1. [README.md](README.md)
2. [docs/configuration/stage_configs.md](docs/configuration/stage_configs.md)
3. [vllm_omni/deploy/qwen3_omni_moe.yaml](vllm_omni/deploy/qwen3_omni_moe.yaml)
4. [vllm_omni/deploy/bagel.yaml](vllm_omni/deploy/bagel.yaml)
5. [vllm_omni/engine/orchestrator.py](vllm_omni/engine/orchestrator.py)

目标：能画出某个模型的 stage 拓扑。

### Step 3：读 AR 路径

阅读顺序：

1. [docs/design/module/ar_module.md](docs/design/module/ar_module.md)
2. [vllm_omni/core/sched/omni_ar_scheduler.py](vllm_omni/core/sched/omni_ar_scheduler.py)
3. [vllm_omni/core/sched/omni_generation_scheduler.py](vllm_omni/core/sched/omni_generation_scheduler.py)
4. [vllm_omni/worker/gpu_ar_model_runner.py](vllm_omni/worker/gpu_ar_model_runner.py)

目标：理解 vLLM 原生调度如何被扩展为 omni 调度。

### Step 4：读 connector 和 KV transfer

阅读顺序：

1. [vllm_omni/distributed/omni_connectors/connectors/base.py](vllm_omni/distributed/omni_connectors/connectors/base.py)
2. [vllm_omni/distributed/omni_connectors/factory.py](vllm_omni/distributed/omni_connectors/factory.py)
3. [vllm_omni/distributed/omni_connectors/kv_transfer_manager.py](vllm_omni/distributed/omni_connectors/kv_transfer_manager.py)
4. [vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py](vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py)

目标：理解 stage 间到底传什么，以及为什么 connector 要独立抽象。

### Step 5：读 diffusion / DiT 路径

阅读顺序：

1. [docs/design/module/dit_module.md](docs/design/module/dit_module.md)
2. [vllm_omni/diffusion/diffusion_engine.py](vllm_omni/diffusion/diffusion_engine.py)
3. [vllm_omni/diffusion/sched/base_scheduler.py](vllm_omni/diffusion/sched/base_scheduler.py)
4. [vllm_omni/diffusion/sched/request_scheduler.py](vllm_omni/diffusion/sched/request_scheduler.py)
5. [vllm_omni/diffusion/sched/step_scheduler.py](vllm_omni/diffusion/sched/step_scheduler.py)
6. [vllm_omni/diffusion/worker/diffusion_model_runner.py](vllm_omni/diffusion/worker/diffusion_model_runner.py)

目标：理解 non-AR runtime 为什么要单独实现。

### Step 6：用一个模型例子串起来

优先建议：

- Qwen3-Omni：学习 async_chunk、TTS/audio streaming。
- BAGEL：学习 AR -> DiT、KV transfer、two-stage vs single-stage。

对应文档：

- [docs/user_guide/examples/offline_inference/qwen3_omni.md](docs/user_guide/examples/offline_inference/qwen3_omni.md)
- [docs/user_guide/examples/offline_inference/bagel.md](docs/user_guide/examples/offline_inference/bagel.md)

## 8. 建议做的三个代码实验

### 实验 1：画出 stage graph

选一个 deploy YAML，例如 `vllm_omni/deploy/bagel.yaml`，手动画：

- stage id。
- stage type。
- input/output connectors。
- engine args。
- sampling params。
- GPU/resource 参数。

### 实验 2：跟踪一个 request 的生命周期

从 `AsyncOmniEngine.add_request()` 开始，沿着：

```text
add_request
  -> _handle_add_request
  -> StagePool.submit_initial
  -> stage output
  -> _route_output
  -> _forward_to_next_stage
  -> output_async_queue
```

把 request id、stage id、finished、segment_finished、final_output 都标出来。

### 实验 3：对比 sync vs async_chunk

看 Qwen3-Omni：

- sync：上游完成后再转发。
- async_chunk：下游预提交，chunk 通过 connector 流动。

重点看：

- TTFP 为什么下降。
- orchestrator 为什么不处理每个 chunk。
- scheduler 为什么要处理 `WAITING_FOR_CHUNK`。

## 9. 当前分析边界

- 本文是静态代码分析，没有安装依赖，也没有运行模型或测试。
- vLLM-Omni 依赖 GPU/CUDA/torch/vLLM 环境，完整运行需要单独配置。
- 论文映射主要基于 README、仓库设计文档、源码结构和 arXiv 论文主题；没有在本文中逐页复述论文图表。

