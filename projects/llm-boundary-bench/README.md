# Local LLM Boundary Bench

这组脚本已经固定到 `/home/undefined/Disk/python-envs/sglang`，使用本地 MiniLLM checkpoint/HF export 和 `/home/undefined/Disk/cache/models/huggingface/Qwen3-0.6B`，不下载模型、不改源权重。第一轮全矩阵位于 `/home/undefined/Disk/build-tmp/llm-boundary-20260726/`，本轮饱和网格、greedy/graph/prefix/语义 A/B 与 Nsight Compute 报告位于 `/home/undefined/Disk/build-tmp/llm-boundary-20260727/`；完整矩阵只采计时与显存，Nsight 只采指定 case，避免 trace 占满磁盘。

默认随机 ID 模式中，`N` 是同批请求数，`P` 是每条 prompt 的精确 token 数，`D` 是每条请求精确生成的 token 数。prefill 已经预测第一个输出，所以实际 decode forward 次数是 `D-1`，最终逻辑 KV 长度是 `实际 prompt tokens+D-1`。随机 ID 模式使用彼此不同的确定性 token ID 和 `ignore_eos=True`，排除 tokenizer、EOS 提前结束和意外共享前缀。真实文本模式则先走模型目录中的本地 tokenizer：默认不补齐，`P` 只是每条输入的截断上限，因此必须看 JSON 的实际 prompt token 分布，不能假定每条输入恰好有 `P` 个 token。离线 `TTFT` 是所有请求同时入队时的首 token 延迟，不等同于带到达过程的在线服务 TTFT。

直接运行：

```bash
cd /home/undefined/Desktop/ai/projects/llm-boundary-bench

./native_minillm_bench.py --matrix all --device cuda --dtype float16 \
  --cache-mode static \
  --warmup 2 --repeat 5 --output /tmp/native-minillm.jsonl --overwrite

# 同 shape 运行 legacy，才能判断固定地址 KV 是降低分配还是已经带来端到端加速。
./native_minillm_bench.py -N 32 -P 8 -D 120 --device cuda --dtype float16 \
  --cache-mode legacy --warmup 2 --repeat 5 --output /tmp/minillm-legacy.jsonl --overwrite

./engine_pressure_bench.py --engine nano --model-kind minillm \
  --preset request --mode graph --warmup-each-case 1 --repeats 3 \
  --output /tmp/nano-minillm.jsonl

./engine_pressure_bench.py --engine nano --model-kind qwen \
  --preset request --mode graph --warmup-each-case 1 --repeats 3 \
  --output /tmp/nano-qwen.jsonl

./engine_pressure_bench.py --engine minisgl --model-kind qwen \
  --preset request --mode graph --cache-type naive --page-size 1 \
  --warmup-each-case 1 --repeats 3 --output /tmp/minisgl-qwen.jsonl

./compress_models.py --model minillm --threads 1 --iterations 50 \
  --output /tmp/minillm-int8.json

./compress_models.py --model qwen --threads 1 --iterations 50 \
  --output /tmp/qwen-int8.json
```

系统化查找 batch 饱和点时，用 `--grid-batches` 与 `--grid-contexts` 生成笛卡尔积。下面会对每个 context 依次测试 `N=1,2,4,8,16,32`，对重复结果取中位数，并在 JSONL 末尾写入一条 `saturation_analysis`：当 `当前吞吐/前一批次吞吐-1 < 5%` 时，`first_low_gain_batch` 标出第一次边际收益不足的 batch。`points` 同时保存实际 total prefill tokens、最终逻辑 cache tokens、prefill/decode 有效 batch，方便区分“请求数增加了”与“调度器实际没有同时运行这么多请求”。

```bash
./engine_pressure_bench.py --engine nano --model-kind qwen --mode graph \
  --grid-batches 1,2,4,8,16,32 --grid-contexts 128,512,2048 \
  --grid-decode-tokens 16 --repeats 3 --warmup-each-case 1 \
  --saturation-metric output_tokens_per_s_wall --saturation-threshold 0.05 \
  --output /tmp/nano-qwen-saturation.jsonl
```

prefill-only 方向的扫描可将 `--grid-decode-tokens` 设为 `1`，并将判断指标改成 `input_tokens_per_s_wall`。不同 context 之间独立找拐点；阈值是相邻 batch 的相对吞吐增益，不是 GPU 利用率，也不单独证明已经达到硬件 roofline。

内置真实文本覆盖中文、英文、代码和较长说明文。`truncate` 只截断超过 `P` 的输入，不补齐短输入；`repeat-truncate` 会先重复原文再截断到 `P`，适合控制 context 压力，但它是合成长度分布，不应当用来评价自然语义质量。tokenizer 加载使用 `local_files_only=True`，不会下载模型或数据。生成 token checksum 总会记录；只有显式传 `--record-generated-text` 才会把完整生成文本写入 JSONL。若要定位第几个 token 开始分歧，再临时加 `--record-output-ids`；全量 token ID 会显著放大长矩阵结果，性能扫描默认不保存。

```bash
# 自然长度：P=512 只是上限，实际长度看 prompt_preparation.prompt_tokens。
./engine_pressure_bench.py --engine nano --model-kind qwen --mode eager \
  --cases 'semantic:8:512:32' --prompt-source builtin --prompt-suite all \
  --text-length-policy truncate --record-generated-text --record-output-ids \
  --output /tmp/qwen-real-prompts.jsonl

# 控制长度压力：每条文本重复后截到目标 P，但仍不做 batch padding。
./engine_pressure_bench.py --engine nano --model-kind qwen --mode eager \
  --grid-batches 1,4,16 --grid-contexts 128,512,2048 \
  --grid-decode-tokens 8 --prompt-source builtin --prompt-suite long \
  --text-length-policy repeat-truncate --output /tmp/qwen-text-pressure.jsonl

# 普通 UTF-8 文件每个非空行是一条 prompt；.jsonl 可用 JSON 字符串或 prompt/text 字段。
./engine_pressure_bench.py --engine nano --model-kind qwen --mode eager \
  --cases 'file-prompts:4:1024:32' --prompt-source file \
  --prompt-file /absolute/path/to/prompts.jsonl --output /tmp/qwen-file-prompts.jsonl
```

每条 case 的 `prompt_preparation` 记录 tokenizer 耗时、截断请求数、token 数分布，以及对“实际送入模型且可 decode 的文本”计算的 chars/token 和 UTF-8 bytes/token；它们用于解释同样字符数为何产生不同 prefill 负载。`metrics.prompt.total_prefill_tokens` 是实际输入 token 总数，`metrics.cache.final_logical_length_per_request` 是每请求最终 KV 长度分布，`metrics.batch.effective_prefill/effective_decode` 是 scheduler step 中的有效 batch。`semantic_regression.output_token_sha256` 可做输出回归，但 checksum 只回答“是否完全相同”：一次接近并列的 argmax 就会改变后续上下文并级联出完全不同的 hash。严格正确性门禁还应比较固定真实语料上的逐 token 一致率、第一处分歧、参考 logits 最大误差以及分歧位置的 top-1/top-2 margin。

`--temperature auto`（默认）保留原有语义：nano-vLLM 使用 `0.1`，Mini-SGLang 使用 `0.0`。run record 的 `config.sampling` 会同时写入请求值、最终温度、greedy/stochastic 策略和 `ignore_eos`。比较两个引擎的 sampler 或做严格输出 checksum A/B 时，应显式给两边相同的 `--temperature 0`；比较真实随机采样则显式给相同的正温度。不要拿 nano 的 auto 随机路径和 Mini-SGLang 的 auto greedy 路径归因成纯引擎差异。

```bash
./engine_pressure_bench.py --engine nano --model-kind qwen --mode eager \
  --cases 'greedy:8:128:32' --temperature 0 --output /tmp/nano-greedy.jsonl

./engine_pressure_bench.py --engine minisgl --model-kind qwen --mode eager \
  --cases 'greedy:8:128:32' --temperature 0 --output /tmp/minisgl-greedy.jsonl
```

自定义压力点用 `NAME:N:P:D`，多个 case 以分号分隔。一个 engine 常驻完成全部 case，避免把模型加载和 CUDA Graph capture 重复算进延迟；每个新 shape 应设置 `--warmup-each-case 1`。高风险超长上下文或接近 OOM 的 case 应放在独立进程中：

```bash
./engine_pressure_bench.py --engine nano --model-kind qwen --mode eager \
  --cases 'context-16k:1:16384:1;context-max:1:40959:1' \
  --max-model-len 40960 --max-num-seqs 1 --max-batched-tokens 8192 \
  --output /tmp/qwen-context.jsonl
```

Nsight Systems 应只捕获一个已 warmup case。脚本用 `cudaProfilerStart/Stop` 标出区间，当前系统必须禁用无权限的 CPU sampling：

```bash
/usr/local/cuda-13.0/bin/nsys profile \
  --trace=cuda,nvtx,osrt,cublas --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true -o /tmp/minisgl-prefill \
  ./engine_pressure_bench.py --engine minisgl --model-kind qwen --mode eager \
  --cases 'profile:1:8192:1' --max-model-len 8193 --max-num-seqs 1 \
  --max-batched-tokens 8192 --cache-type naive --cuda-profiler-case 0 \
  --output /tmp/minisgl-prefill.jsonl

/usr/local/cuda-13.0/bin/nsys stats \
  --report nvtx_gpu_proj_sum,cuda_gpu_kern_sum,cuda_api_sum \
  /tmp/minisgl-prefill.nsys-rep
```

分析顺序是：先看端到端 `wall_ms/TTFT/ITL`，再看 prefill/decode CUDA Event 时间和 scheduler step 数，随后在 Nsight 中查看 NVTX 的 MHA/MLP/LMHead/Sampler 占比、kernel 总时间、kernel 之间的空洞、launch/memcpy/synchronize。若 NVTX 区间远大于 kernel 总时间，瓶颈通常是 CPU 调度或 launch 空洞；若 MHA 随 `P` 快速增长，则是长上下文 attention；若跨 KV page/block 后 step 数跳变，则是 KV 容量或碎片。

nano-vLLM 现已支持 `temperature=0` 的 argmax fast path 和同批 greedy/random 请求，严格跨引擎 A/B 应显式传相同温度。本机 `N=16,P=128,D=64,graph` 的中位结果中，greedy 相比 temperature 0.1 随机采样把总时间从 328.6 ms 降到 310.7 ms、输出吞吐从 3116 提高到 3296 tok/s；这约 5.8% 是当前 shape 的 sampler 路径差异，不代表所有 batch/context 都有相同比例。

Nsight Compute 2025.3.1 已经通过管理员权限完成真实采集，报告保存在 `/home/undefined/Disk/build-tmp/llm-boundary-20260727/profiles/`。8192-token Mini-SGLang prefill attention 约 5.80 ms，SM throughput 42.0%、DRAM 4.1%、occupancy 16.2%，每线程 248 个 registers，主要受寄存器/共享内存容量与 math-pipeline stall 限制；代表性 prefill GEMM 约 1.26 ms，SM 48.1%、DRAM 15.5%、occupancy 16.5%，同样不是 HBM 饱和。nano 单请求 LM-head GEMV 约 662.7 μs，DRAM throughput 95.7%、470.6 GB/s、L2 hit 0.66%，则是明确的权重带宽瓶颈。NCU 不应抓整场服务，只从 Systems 热点中按 kernel 名限制一次 launch：

```bash
sudo /usr/local/cuda-13.0/bin/ncu --target-processes all --set full \
  --kernel-name 'regex:BatchPrefillWithPagedKVCacheKernel' --launch-count 1 \
  -o /tmp/qwen-prefill-kernel \
  ./engine_pressure_bench.py ...
```

本轮固定 shape 结果也说明瓶颈不能只用“模型大小”概括：`P=128` 时 nano CUDA Graph 将 `N=1` 输出吞吐从 96.8 提高到 266.0 tok/s、`N=16` 从 1363.8 提高到 3070.7 tok/s，说明 decode launch 明显；`P=2048,N=16` 时 graph 只剩几个百分点，因为 32768 个输入 token 被 2048-token 调度上限拆成 16 个 prefill step。相同长上下文下 `N=8→16` 的吞吐增益只剩 nano 14.2%、Mini-SGLang 7.1%，TTFT 却接近翻倍，实用饱和区已经在 N=8–16。

Mini-SGLang 的 warm shared-prefix A/B 必须使用相同请求、同一常驻进程并先 warm cache。本机 `N=8,P=512,D=32,graph` 中，Naive 实际 prefill 4096 token、TTFT 55.1 ms、输出 1167 tok/s；Radix 只重新 prefill 每请求最后 1 个 token，共 8 token，TTFT 7.1 ms、输出 1976 tok/s。该结果不代表冷请求、随机前缀或多租户都能获得同样收益，服务监控必须拆分 cold/warm、matched/evicted/protected tokens。

TorchAO 实验是 CPU 内存中的 INT8 weight-only 教学基线，不覆盖 checkpoint，也不代表 nano-vLLM/Mini-SGLang 已有对应 GPU 量化 kernel。评估压缩必须同时看参数载荷、真实目标硬件延迟、logit/top-1 和 PPL；只看文件变小不足以判断服务会更快。

纯 CPU 的参数与统计逻辑不需要加载模型或占用 GPU，可以单独回归：

```bash
/home/undefined/Disk/python-envs/sglang/bin/python -m unittest -v \
  /home/undefined/Desktop/ai/projects/llm-boundary-bench/test_engine_pressure_bench.py
```

非 root NVIDIA performance counters 配置已经写入系统并更新 initramfs，下次正常重启后生效；当前驱动会话仍显示 `RmProfilingAdminOnly=1`，所以本轮 NCU 使用 `sudo`。CPU 的 `kernel.perf_event_paranoid=-1` 已立即生效。脚本不会卸载驱动或自动重启，重启前后都可以用检查脚本确认真实运行时状态：

```bash
cd /home/undefined/Desktop/ai/projects/llm-boundary-bench
sudo ./setup_profiling_permissions.sh --cpu-perf-event-paranoid -1
# 在方便时手动重启，随后检查运行时参数是否已经生效。
./check_profiling_permissions.sh
```
