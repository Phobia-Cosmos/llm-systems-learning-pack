# MiniLLM Agent runtime（初版）

`minillm_agent.py` 把训练好的 MiniLLM、nano-vLLM/mini-sglang 可替换后端和工具执行层分开。工具调用使用显式 `<tool_call>{...}</tool_call>` 契约，当前只开放无 `eval()` 的安全四则运算；每次请求保存 JSON trace，后续可上传 EdgeForge Artifact。

先用确定性计划验证 Agent 基础设施：

```sh
cd /home/undefined/llm-systems-learning-pack/projects/minillm-general
/home/undefined/Disk/python-envs/brainuicl/bin/python agents/minillm_agent.py \
  --plan --output benchmarks/results/agent_plan_smoke_20260916.json
```

使用自训练模型生成（模型是否遵循工具格式仍需专门 tool-call 数据训练）：

```sh
/home/undefined/Disk/python-envs/sglang/bin/python agents/minillm_agent.py \
  --backend native --device cuda --max-tokens 32 \
  --output benchmarks/results/agent_native_minillm_20260916.json
```

`mini-sglang` HTTP 后端只需先启动 `projects/mini-sglang/mini_sglang_server.py`，再将 `--backend http --url ...` 传入。Agent 不把工具执行、状态或安全策略塞进推理引擎；后续 NanoVLLM/MiniSGLang 的 scheduler、prefix cache 和 streaming 优化都可以复用同一请求/结果/trace 契约。

后端契约位于 `minillm/contracts.py` 和 `minillm/adapters.py`：`GenerationRequest` 描述 prompt 与采样预算，`GenerationResult` 明确区分 prompt token ids 和 completion token ids，并携带后端、耗时及可扩展 metadata。`scripts/contract_smoke.py` 验证 native/nano token parity，`scripts/benchmark_batch_contract.py` 提供四请求 scheduler 基线；这两个 smoke 的 JSON 可直接作为 Agent trace 接入 EdgeForge Artifact 的输入。

当前后端目录见 `backend_catalog.json`。Agent 的默认候选是本地 nano-vLLM + `minillm-160m-8k-final`，native MiniLLM 作为 correctness reference，vLLM 作为成熟 server baseline，mini-sglang 作为协议层 reference，官方 SGLang 暂时保持 blocked，直到补上真正的 MiniGPT model adapter。
