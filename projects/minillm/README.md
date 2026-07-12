# MiniLLM: 一个最小可扩展的教学 LLM

这个项目实现了一个最小 decoder-only GPT/LLM。它不是 ChatGPT 那种大模型，而是用来学习 LLM 基本结构、训练流程和后续扩展点的代码骨架。

## 它包含什么

- `CharTokenizer`: 最小字符级 tokenizer。
- `HFByteBPETokenizer`: 可训练的 Byte-level BPE 教学实现。
- `HFTokenizerAdapter`: 复用标准 Hugging Face fast tokenizer。
- `MiniGPT`: decoder-only Transformer。
- `CausalSelfAttention`: 带 causal mask 的多头自注意力。
- `TransformerBlock`: LayerNorm、attention、MLP、残差连接。
- `train.py`: next-token prediction 训练脚本。
- `generate.py`: 从 checkpoint 采样生成文本。
- `data/tiny_corpus.txt`: 一个很小的中文教学语料。

## 环境要求

最低配置：

- Python 3.10+，当前机器是 Python 3.12.3。
- PyTorch 2.3+。
- CPU 可运行，2GB 内存足够跑默认 toy 配置。
- GPU 不是必须；如果有 CUDA GPU，训练会更快。

默认模型大约几十万到百万级参数，主要用于理解结构，不用于真实生产。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果你要安装指定 CUDA 版本的 PyTorch，建议按 PyTorch 官网给出的命令安装，再运行本项目。

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

选择 tokenizer：

```bash
# 默认字符 tokenizer
python train.py --tokenizer char

# 项目内训练/保存的 Byte-level BPE
python train.py --tokenizer byte-bpe --tokenizer-output-dir tokenizer_variants/byte_bpe

# 导入任意标准 HF fast tokenizer
python train.py \
  --tokenizer hf-auto \
  --tokenizer-path tokenizer_variants/byte_bpe

# 独立训练 SentencePiece BPE 或 Unigram
python train.py --tokenizer sentencepiece-bpe --tokenizer-vocab-size 512
python train.py --tokenizer sentencepiece-unigram --tokenizer-vocab-size 512
```

公共接口、SentencePiece 论文与技术演进见：

```text
docs/tokenizer_interface_hf_adapter_and_evolution.md
```

训练完成后会保存：

```text
checkpoints/minillm.pt
```

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
/home/undefined/Desktop/ai/.venv-sglang/bin/python export_hf_like.py   --checkpoint checkpoints/minillm.pt   --out-dir hf_exports/minillm   --safe-serialization
```

这个导出目录用于学习 Hugging Face 文件结构。当前同级 `nano-vllm` 项目已经注册 MiniGPT 后端；原版 vLLM/SGLang 仍需各自实现并注册该架构。

## 这个 LLM 可以做什么

在默认 tiny corpus 上，它只能学到很小语料里的字符模式，输出可能不稳定。它适合做这些事情：

- 学习 LLM 的基本组成部分。
- 观察 loss 如何下降。
- 理解 token、embedding、attention、MLP、采样之间的关系。
- 作为读论文时的实验底座。
- 后续扩展 LoRA、RoPE、RMSNorm、SwiGLU、量化、RAG、指令微调。

它不适合：

- 当真实问答系统。
- 当可靠知识库。
- 评估真实 LLM 能力。

## 后续扩展路线

建议按这个顺序改：

1. 对比已实现的 Char、Byte-BPE、HF adapter，并继续做 SentencePiece BPE/Unigram 实验。
2. 把 `position_embedding` 换成 RoPE。
3. 把 `LayerNorm + GELU MLP` 换成 `RMSNorm + SwiGLU`。
4. 继续完善 KV cache：当前已有教学版 `--kv-cache`，下一步支持 RoPE 和更长上下文。
5. 加 `torch.utils.data.Dataset/DataLoader`，支持大文件和多 worker。
6. 加验证集 perplexity、生成样例、checkpoint resume。
7. 加 LoRA，只训练 adapter。
8. 加 INT8/INT4 量化推理。
9. 加 instruction tuning 数据格式。
10. 加 RAG，把外部文档检索结果拼到 prompt。

## 代码入口

- 模型结构: `minillm/model.py`
- 配置: `minillm/config.py`
- tokenizer: `minillm/tokenizer.py`
- 数据 batch: `minillm/data.py`
- 训练: `train.py`
- 生成: `generate.py`
- KV cache、autograd、训练到推理路线: `docs/kvcache_autograd_training_roadmap.md`
- 模型结构、推理引擎接入、AI Infra 表格: `docs/minillm_ai_infra_engine_requirements.md`

### 通过 nano-vLLM 教学后端运行 MiniLLM

先训练并导出 HF-like 目录后，可以通过 nano-vLLM 的最小 MiniGPT 后端运行：

```bash
/home/undefined/Desktop/ai/.venv-sglang/bin/python export_hf_like.py \
  --checkpoint checkpoints/minillm.pt \
  --out-dir hf_exports/minillm \
  --safe-serialization

/home/undefined/Desktop/ai/.venv-sglang/bin/python scripts/run_nanovllm_minigpt.py
```

当前 MiniGPT 通过独立模型模块注册，走 nano-vLLM 的 `LLM.generate()`、scheduler、sampler、FlashAttention 和 paged KV cache。模型仍使用训练时的 learned absolute position embedding，因此 prompt 与生成 token 总数不能超过导出配置中的 `block_size`。

### 通过 mini-sglang 教学服务调用 MiniLLM

```bash
cd /home/undefined/Desktop/ai
source scripts/use_disk_ai_env.sh
python projects/mini-sglang/mini_sglang_server.py \
  --checkpoint projects/minillm/checkpoints/minillm.pt \
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
