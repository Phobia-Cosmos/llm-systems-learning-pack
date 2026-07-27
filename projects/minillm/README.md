# MiniLLM: 一个最小可扩展的教学 LLM

这个项目实现了一个最小 decoder-only GPT/LLM。它不是 ChatGPT 那种大模型，而是用来学习 LLM 基本结构、训练流程和后续扩展点的代码骨架。

## 它包含什么

- `CharTokenizer`: 最小字符级 tokenizer。
- `HFByteBPETokenizer`: 可训练的 Byte-level BPE 教学实现。
- `HFTokenizerAdapter`: 复用标准 Hugging Face fast tokenizer。
- `MiniGPT`: decoder-only Transformer。
- `CausalSelfAttention`: 带 causal mask 的 MHA/GQA/MQA 自注意力与紧凑 KV cache。
- learned、sinusoidal、RoPE、ALiBi 与 NoPE 位置模式，可通过配置切换。
- `TransformerBlock`: LayerNorm、attention、MLP、残差连接。
- `train.py`: next-token prediction 训练脚本。
- `generate.py`: 从 checkpoint 采样生成文本。
- `data/tiny_corpus.txt`: 一个很小的中文教学语料。

## 环境要求

最低配置：

- Python 3.10+，当前机器是 Python 3.12.3。
- PyTorch 2.3+。
- CPU 可运行，2GB 内存足够跑默认 toy 配置。
- GPU 不是必须；安装 CUDA-enabled PyTorch 后可在同一份代码中选择 CPU 或 CUDA。

默认模型大约几十万到百万级参数，主要用于理解结构，不用于真实生产。

## 安装

`requirements.txt` 不再固定 `+cpu` wheel；安装时只选择一个与机器匹配的 PyTorch index。CUDA wheel 本身也能执行 CPU 路径，不需要维护两份 MiniLLM 源码。

```bash
python3 -m venv .venv
source .venv/bin/activate

# CPU-only 环境
pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 当前机器 CUDA 13 对应环境；若官方 index 变化，以 PyTorch 安装页为准
pip install --index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
```

本机也可直接复用已登记的 `/home/undefined/Disk/python-envs/vllm/bin/python`（PyTorch 2.11.0+cu130）运行 CUDA 教学基准；不要为了本项目修改这个共享环境。`train.py`/`generate.py` 已支持 `--device auto|cpu|cuda|mps`，模型参数、buffer 和输入会由现有 `.to(device)` 路径迁移。

## 训练

CPU 上运行：

```bash
python train.py --device cpu --max-steps 500
```

更小更快的调试配置：

```bash
python train.py --device cpu --max-steps 100 --n-layer 1 --n-head 2 --n-embd 64 --batch-size 16
```

如果有 CUDA：

```bash
python train.py --device cuda --max-steps 1000
```

现有 MiniMind JSONL 的可复现数据准备使用流式脚本，不把 1.24 GiB 原文件复制进仓库。它会做 UTF-8/JSON/字段检查、Unicode 规范化、按规范化正文 SHA-256 精确去重，再按文档或来源 hash 稳定切成 train/validation/test；同一内容不会跨 split，若数据带可靠的来源/文档 ID，还应传 `--group-field` 让同一来源留在同一 split。先用 10000 行 smoke 验证输出和 manifest：

```bash
/home/undefined/Disk/python-envs/sglang/bin/python scripts/prepare_jsonl_corpus.py \
  --input /home/undefined/Disk/datasets/minimind/pretrain_t2t_mini.jsonl \
  --output-dir /home/undefined/Disk/build-tmp/minillm-corpus-smoke \
  --max-records 10000 --validation-fraction 0.1 --test-fraction 0.1
```

输出目录包含 `train.txt`、`validation.txt`、`test.txt` 与记录输入扫描 hash、过滤计数、split 统计和输出 hash 的 `manifest.json`。输出目录必须是新目录，避免把不同运行静默混在一起；完整处理时删掉 `--max-records` 并换一个新 run 目录。MiniLLM 可以直接使用已经切好的 train/validation，且新 tokenizer 只从 `train.txt` 学习：

```bash
/home/undefined/Disk/python-envs/sglang/bin/python train.py \
  --data /home/undefined/Disk/build-tmp/minillm-corpus-smoke/train.txt \
  --val-data /home/undefined/Disk/build-tmp/minillm-corpus-smoke/validation.txt \
  --tokenizer byte-bpe --tokenizer-vocab-size 4096 \
  --tokenizer-output-dir artifacts/tokenizers/minimind-smoke \
  --device cuda --max-steps 100
```

当前 `train.py` 仍会把整个文本 split 和 token ids 读入内存，所以完整 1.24 GiB 语料应在下一步接入 mmap/packed-token Dataset、按 shard 迭代、checkpoint resume 与分布式 sampler 后再正式长训；数据切分脚本已经先固定了不会泄漏的边界和可复现 manifest。

训练 RoPE 版本：

```bash
python train.py --device cuda --position-encoding rope --rope-theta 10000 --checkpoint-name minillm-rope.pt
```

训练 GQA 或 MQA 版本：

```bash
# 4 个 query heads、2 个 KV heads：GQA
python train.py --n-head 4 --num-key-value-heads 2 --position-encoding rope --checkpoint-name minillm-gqa.pt

# KV heads=1：MQA；省略该参数或设为 --n-head 则保持旧 MHA
python train.py --n-head 4 --num-key-value-heads 1 --position-encoding rope --checkpoint-name minillm-mqa.pt
```

选择 tokenizer：

```bash
# 默认字符 tokenizer
python train.py --tokenizer char

# 项目内训练/保存的 Byte-level BPE
python train.py \
  --tokenizer byte-bpe \
  --tokenizer-output-dir artifacts/tokenizers/byte_bpe \
  --checkpoint-name minillm-byte-bpe.pt

# 导入任意标准 HF fast tokenizer
python train.py \
  --tokenizer hf-auto \
  --tokenizer-path artifacts/tokenizers/byte_bpe \
  --checkpoint-name minillm-hf-auto.pt

# 独立训练 SentencePiece BPE 或 Unigram
python train.py --tokenizer sentencepiece-bpe --tokenizer-vocab-size 512 --checkpoint-name minillm-sentencepiece-bpe.pt
python train.py --tokenizer sentencepiece-unigram --tokenizer-vocab-size 512 --checkpoint-name minillm-sentencepiece-unigram.pt
```

公共接口、SentencePiece 论文与技术演进见：

```text
docs/tokenizer_interface_hf_adapter_and_evolution.md
```

训练完成后会保存：

```text
artifacts/checkpoints/minillm.pt
```

所有模型产物统一放在 `artifacts/` 下，但按格式分层：PyTorch checkpoint 平铺在 `artifacts/checkpoints/`，HF-like 多文件导出放在 `artifacts/hf_exports/<variant>/`，tokenizer bundle 放在 `artifacts/tokenizers/<variant>/`。现有 checkpoint 分别是 `minillm.pt`、`minillm-byte-bpe.pt` 和 `minillm-rope.pt`；训练新变体时用 `--checkpoint-name` 指定不会覆盖现有文件的名称。

## 生成

```bash
python generate.py --device cpu --prompt "用户: LLM 可以做什么？"
```

可调参数：

```bash
python generate.py --prompt "MiniGPT" --max-new-tokens 200 --temperature 0.8 --top-k 40
```

使用教学版 KV cache 生成：

```bash
python generate.py --prompt "MiniGPT" --max-new-tokens 40 --greedy --kv-cache
```

## 完整数值调试：从极小语料到每个 Q/K/V

如果目标是先完整看懂一次 LLM 训练，而不是立即追求生成质量，运行专用的极小调试流水线：

```bash
/home/undefined/Desktop/ai/.venv/bin/python scripts/debug_tiny_transformer.py --device cpu
```

它使用 `data/debug_corpus.txt`，默认模型只有 1 个 Transformer block、2 个 attention head、8 维 hidden state、976 个可训练参数。为便于第一次学习，默认实际创建独立的 `q_proj/k_proj/v_proj`；传入 `--fused-qkv` 才切换到生产式融合 QKV。脚本会真实执行：

```text
corpus -> CharTokenizer -> 完整词表 -> shifted x/y
-> embedding -> LN -> Wq/Wk/Wv -> 分头 -> RoPE(Q,K)
-> QK^T -> causal mask -> softmax -> weights@V -> Wo
-> residual -> MLP -> logits -> cross entropy
-> backward -> dL/dQ,dL/dK,dL/dV -> AdamW update
-> 训练前/第 1 步后/训练完成的 QKV 对比 -> greedy generation
```

输出目录是 `debug_outputs/tiny_transformer/`：

- `report.md`：以公式、shape 和上下游关系为主的完整中文流程讲解。
- `tensor_dump.md`：训练前、第一步后和训练完成后的全部小张量真实数值。
- `vocab.json`：字符 token 的 `token <-> id` 完整词表。
- `loss.csv`：训练过程的全语料 loss。
- `checkpoint.pt`：可由 MiniLLM 恢复的模型、config 和 tokenizer。

报告还会自动验证 trace logits/loss 与真实 `MiniGPT.forward` 一致、Q/K/V 输出与各自投影公式一致、未来 attention 权重为 0、softmax 每行之和为 1。调试实现位于 `minillm/debug.py`，对应回归测试为 `tests/test_debug_flow.py`。



## 更稳定的教学输出

原始 `data/tiny_corpus.txt` 很小，训练步数少时输出会像随机字符。现在提供了更结构化的教学语料：

```bash
/home/undefined/Desktop/ai/.venv/bin/python train.py   --data data/teaching_corpus.txt   --device cpu   --max-steps 1500   --eval-interval 300   --eval-iters 5   --batch-size 32   --block-size 128   --n-layer 2   --n-head 4   --n-embd 128
```

训练后先用 greedy 解码检查模型是否学会稳定格式：

```bash
/home/undefined/Desktop/ai/.venv/bin/python generate.py   --device cpu   --prompt "用户: 什么是 attention？
助手:"   --max-new-tokens 120   --greedy
```

导出教学用 HF-like 目录：

```bash
/home/undefined/Desktop/ai/.venv-sglang/bin/python export_hf_like.py   --checkpoint artifacts/checkpoints/minillm.pt   --out-dir artifacts/hf_exports/minillm   --safe-serialization
```

这个导出目录用于学习 Hugging Face 文件结构。当前同级 `nano-vLLM` 和本地 vLLM 源码都已注册 MiniGPT 后端；原版 SGLang 仍需单独实现 native backend。

## 这个 LLM 可以做什么

在默认 tiny corpus 上，它只能学到很小语料里的字符模式，输出可能不稳定。它适合做这些事情：

- 学习 LLM 的基本组成部分。
- 观察 loss 如何下降。
- 理解 token、embedding、attention、MLP、采样之间的关系。
- 作为读论文时的实验底座。
- 对比 learned/sinusoidal/RoPE/ALiBi/NoPE、LayerNorm/RMSNorm/ScaleNorm、多种 dense/gated MLP，以及 MHA/GQA/MQA。
- 后续扩展训练 resume、LoRA、量化、RAG、指令微调。

它不适合：

- 当真实问答系统。
- 当可靠知识库。
- 评估真实 LLM 能力。

## 结构组件公平 benchmark

位置编码、Norm、MLP 和 attention 已有可复现的四组消融基准：

```bash
/home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/benchmark_components.py \
  --run-name components_cpu_100step
```

新运行默认比较 15 个变体（包含 MHA/GQA/MQA）、3 个模型随机种子，每个变体严格执行 100 次参数更新。tokenizer 只在 train split 上建立；所有变体复用相同的训练 batch、完整验证目标、optimizer、token 数和 greedy prompt；同名同 shape 的公共参数也具有相同初值。

输出为：

- `benchmarks/results/components_cpu_100step.json`：完整 config、逐 seed 原始结果、hash 与配对统计。
- `benchmarks/results/components_cpu_100step.csv`：适合后续画图或数据分析的扁平记录。
- `benchmarks/results/components_cpu_100step.md`：便于阅读的均值、标准差与相对基线差。

仓库中已有的历史 CPU 小基准仍是加入 attention suite 前的 12 变体、36 条逐 seed 记录；新默认矩阵会产生 45 条，并额外记录 Q/KV heads、每 token 每层 KV cache bytes 和相对 MHA 压缩比。loss 和 tokens/s 只用于教学回归，不能据此给真实大模型的结构优劣排名。详细契约见 `benchmarks/README.md`。

## 后续扩展路线

建议按这个顺序改：

1. 已完成 position、Norm、MLP 模块化与三 seed CSV/JSON benchmark，并保留旧 checkpoint 兼容。
2. 为 RoPE + Byte-BPE + RMSNorm + SwiGLU 训练正式 checkpoint，补齐 native/nano-vLLM/vLLM 端到端对齐。
3. 已完成可选 GQA/MQA、紧凑 KV cache、attention benchmark 与 nano-vLLM/vLLM loader，并保持 MHA/旧 checkpoint 兼容。
4. 加 `Dataset/DataLoader`、checkpoint resume、学习率调度、warmup、梯度累积和混合精度。
5. 扩展为更长训练预算、更多语料与 GPU/engine 性能 benchmark。
6. 完成 SFT loss mask，再加入 LoRA。
7. 包装为真正的 Transformers `PreTrainedModel`。
8. 完成 SGLang native backend、OpenAI 服务与并发测试。
9. 加 INT8/INT4 量化并对比精度、显存、吞吐。
10. 接入自定义 Triton/CUDA 算子并做端到端 benchmark。

## 代码入口

- 模型结构: `minillm/model.py`
- 配置: `minillm/config.py`
- tokenizer: `minillm/tokenizer.py`
- 数据 batch: `minillm/data.py`
- 训练: `train.py`
- 生成: `generate.py`
- 结构组件 benchmark: `scripts/benchmark_components.py`、`minillm/benchmark.py`
- KV cache、autograd、训练到推理路线: `docs/kvcache_autograd_training_roadmap.md`
- 模型结构、推理引擎接入、AI Infra 表格: `docs/minillm_ai_infra_engine_requirements.md`
- RoPE 原理、实现、验证与后续路线: `docs/rope_implementation_and_roadmap.md`

### 通过 nano-vLLM 教学后端运行 MiniLLM

先训练并导出 HF-like 目录后，可以通过 nano-vLLM 的最小 MiniGPT 后端运行：

```bash
/home/undefined/Desktop/ai/.venv-sglang/bin/python export_hf_like.py \
  --checkpoint artifacts/checkpoints/minillm.pt \
  --out-dir artifacts/hf_exports/minillm \
  --safe-serialization

/home/undefined/Desktop/ai/.venv-sglang/bin/python scripts/run_nanovllm_minigpt.py
```

当前 MiniGPT 通过独立模型模块注册，走 nano-vLLM 的 `LLM.generate()`、scheduler、sampler、FlashAttention 和 paged KV cache。learned/RoPE 与 MHA/GQA/MQA export 都可由后端配置和 loader 识别；教学模型的总序列长度仍受训练配置中的 `block_size` 限制。

### 通过 vLLM 运行 RoPE MiniLLM

```bash
source /home/undefined/Desktop/ai/use_vllm.sh
VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/run_vllm_minigpt.py
```

当前 RoPE checkpoint 的验证输出为 `embedding 是把离散 token id 映射成连续向量的查表过程`。native MiniLLM、native KV cache、nano-vLLM 和 vLLM 使用相同 greedy token 序列。

### 通过 mini-sglang 教学服务调用 MiniLLM

```bash
cd /home/undefined/Desktop/ai
source /home/undefined/Desktop/ai/use_disk_ai_env.sh
python projects/mini-sglang/mini_sglang_server.py \
  --checkpoint projects/minillm/artifacts/checkpoints/minillm.pt \
  --host 127.0.0.1 \
  --port 8011 \
  --device cpu
```

然后请求 OpenAI-like completion：

```bash
curl http://127.0.0.1:8011/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"用户: 什么是 decoder-only Transformer？\n助手:","max_tokens":80}'
```
