# MiniLLM: 一个最小可扩展的教学 LLM

这个项目实现了一个最小 decoder-only GPT/LLM。它不是 ChatGPT 那种大模型，而是用来学习 LLM 基本结构、训练流程和后续扩展点的代码骨架。

## 它包含什么

- `CharTokenizer`: 最小字符级 tokenizer。
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

## 这个 LLM 可以做什么

在默认 tiny corpus 上，它只能学到很小语料里的字符模式，输出可能不稳定。它适合做这些事情：

- 学习 LLM 的基本组成部分。
- 观察 loss 如何下降。
- 理解 token、embedding、attention、MLP、采样之间的关系。
- 作为读论文时的实验底座。
- 后续扩展 LoRA、RoPE、RMSNorm、SwiGLU、KV cache、量化、RAG、指令微调。

它不适合：

- 当真实问答系统。
- 当可靠知识库。
- 评估真实 LLM 能力。

## 后续扩展路线

建议按这个顺序改：

1. 把 `CharTokenizer` 换成 BPE/SentencePiece tokenizer。
2. 把 `position_embedding` 换成 RoPE。
3. 把 `LayerNorm + GELU MLP` 换成 `RMSNorm + SwiGLU`。
4. 给 `generate()` 加 KV cache，减少重复计算。
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

