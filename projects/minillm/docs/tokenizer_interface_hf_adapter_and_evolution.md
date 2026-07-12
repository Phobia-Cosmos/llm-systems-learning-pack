# MiniLLM Tokenizer 公共接口、HF Adapter 与技术演进

## 1. 当前实现状态

MiniLLM 现在有一个稳定公共接口和五种选择：

| `--tokenizer` | 实现 | 用途 |
| --- | --- | --- |
| `char` | `CharTokenizer` | 最小教学实现；不依赖 Transformers/tokenizers |
| `byte-bpe` | `HFByteBPETokenizer` | 自己训练 Byte-level BPE，观察 vocab/merge/special token |
| `hf-auto` | `HFTokenizerAdapter` | 直接复用标准 Hugging Face fast tokenizer，包括由 SentencePiece 转换出的 fast tokenizer |
| `sentencepiece-bpe` | `SentencePieceTokenizer` | 自己训练 SentencePiece BPE，学习 raw-text subword 流程 |
| `sentencepiece-unigram` | `SentencePieceTokenizer` | 自己训练 Unigram LM，比较概率式词表与 BPE merge |

公共接口在：

```text
minillm/tokenizer_base.py
```

变体在：

```text
minillm/tokenizer_variants/byte_bpe.py
minillm/tokenizer_variants/hf_adapter.py
minillm/tokenizer_variants/sentencepiece_tokenizer.py
```

CharTokenizer 保持轻量；Byte-BPE/HF 依赖采用懒加载，因此只学习字符模型时不强制安装 Transformers。

## 2. 公共接口包含什么

`MiniTokenizer` 定义：

```text
vocab_size
bos_token_id / eos_token_id / pad_token_id / unk_token_id
encode / decode
token_to_id / id_to_token
batch_encode -> input_ids + attention_mask
apply_chat_template
save_pretrained
to_dict
```

`TokenizerBatch` 返回：

```text
input_ids:      [B, T] 的 Python 列表形式
attention_mask: 真实 token 为 1，padding 为 0
```

公共 batch 逻辑要求调用者显式选择：

- 是否 padding。
- 是否 truncation。
- `max_length` 是多少。
- 是否加入 BOS/EOS。

如果 tokenizer 没有 PAD（旧 CharTokenizer），却要求给不同长度样本 padding，会明确报错，不会偷偷用 `<unk>` 代替 PAD。

## 3. HFTokenizerAdapter 的边界

加载本地或 Hugging Face tokenizer：

```python
from minillm.tokenizer_variants import HFTokenizerAdapter

tokenizer = HFTokenizerAdapter.from_pretrained("path/or/repo")
```

训练入口：

```bash
python train.py \
  --tokenizer hf-auto \
  --tokenizer-path tokenizer_variants/byte_bpe \
  --data data/teaching_corpus.txt \
  --out-dir checkpoints-hf
```

当前 adapter 要求 fast tokenizer。原因是 fast tokenizer 的 Rust backend 可以序列化成完整 `tokenizer_json`，直接写进 checkpoint：

```text
checkpoint
  tokenizer_type = hf-auto
  tokenizer.tokenizer_json
  tokenizer.special_tokens
  tokenizer.chat_template
```

因此移动 checkpoint 后不需要依赖原始 tokenizer 下载目录。

如果某个生产模型的 SentencePiece tokenizer 只有 slow Python 实现，建议先转换成 HF fast tokenizer；本项目自带的
`SentencePieceTokenizer` 主要用于算法实验，并用 base64 model proto 实现自包含 checkpoint。

## 4. SentencePiece 是算法还是工具

SentencePiece 更准确地说是“从原始文本训练和执行 subword tokenizer 的框架”。它主要支持两种模型：

1. BPE：反复合并高频相邻片段。
2. Unigram Language Model：从大候选词表中逐步删减，使语料似然损失尽量小。

它的关键价值是：

- 不要求语言必须用空格分词。
- 把空格也当普通符号处理。
- 适合中文、日文及多语言语料。
- 同时提供训练、encode、decode 和可移植模型文件。

核心论文：

- SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing, Kudo and Richardson, EMNLP 2018: https://arxiv.org/abs/1808.06226
- Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates, Kudo, ACL 2018: https://arxiv.org/abs/1804.10959

MiniLLM 有两条 SentencePiece 路线：

```bash
# 学习 SentencePiece BPE
python train.py --tokenizer sentencepiece-bpe --tokenizer-vocab-size 512

# 学习 SentencePiece Unigram LM
python train.py --tokenizer sentencepiece-unigram --tokenizer-vocab-size 512

# 使用真实模型已经发布的标准 HF/SentencePiece tokenizer
python train.py --tokenizer hf-auto --tokenizer-path /path/to/tokenizer
```

自训练版本保存 `tokenizer.model`、`tokenizer.vocab` 和配置，并将序列化 model proto 写入 checkpoint。HF adapter
和自训练 SentencePiece 仍保持独立，因为前者强调生态兼容，后者强调算法可观察性。

## 5. 三种“编码”不要混在一起

```text
文本
  -> tokenizer / text encoding
  -> token ids [B,T]
  -> token embedding
  -> hidden states [B,T,C]
  -> position encoding（当前 learned absolute；未来可选 RoPE）
  -> attention / Transformer blocks
```

### 文本编码

决定文本怎样切成 token，以及 token 对应哪个整数 id。

### Token embedding

根据 token id 查询模型权重：

```text
embedding.weight: [vocab_size, hidden_size]
```

这属于模型参数，不属于 tokenizer。

### 位置编码

告诉 attention token 在什么位置。当前 MiniLLM 使用 learned absolute position embedding；RoPE 属于这一层，与 Char/BPE/SentencePiece 无关。

## 6. 为什么推理引擎也要实现 embedding

引擎端不是学习一套不同语义的 embedding。训练模型和引擎必须使用同一个 checkpoint 中的：

```text
token_embedding.weight
```

区别是执行实现：

| 场景 | 可能的 embedding 实现 |
| --- | --- |
| MiniLLM 教学训练 | 单卡 `nn.Embedding` |
| nano-vLLM 单卡 | 与模型结构同形的 embedding，加载相同权重 |
| vLLM/SGLang tensor parallel | `VocabParallelEmbedding`，按 vocab 行切到多张 GPU |
| 量化推理 | INT8/INT4 或特定 kernel/layout 的 embedding |
| kernel 对齐 | vocab 可能 padding 到硬件友好倍数，但有效 token id 范围不变 |

推理引擎通常不会直接调用训练项目的 `MiniGPT.forward()`，而是重新构造一个适合 scheduler、paged KV cache、tensor parallel 和 CUDA graph 的执行图。因此它必须定义 embedding 层，并把 checkpoint 的同一份权重加载进去。

必须保持三者一致：

```text
tokenizer 的 token id 语义
config.vocab_size
embedding/lm_head 权重行
```

任何一个不一致都会导致越界或生成乱码。

## 7. 主流文本 tokenization 方法

| 方法 | 基本单位 | 优点 | 主要问题 |
| --- | --- | --- | --- |
| Word | 单词 | 人类可读 | 词表爆炸、OOV、多语言分词困难 |
| Character | Unicode 字符 | 简单、几乎无词表训练 | 序列长，计算成本高 |
| WordPiece | 概率/似然驱动的子词构造 | BERT 生态成熟 | 实现和预切分规则依赖较强 |
| BPE | 高频相邻片段合并 | 简单、压缩率好 | merge 是贪心频率规则 |
| Unigram LM | 候选子词概率模型 | 可做多种切分和 subword regularization | 训练算法更复杂 |
| SentencePiece | 原始文本上的 BPE/Unigram 框架 | 语言无关、多语言友好 | 模型格式和 HF 集成需处理 |
| Byte-level BPE | byte alphabet + BPE | 基本没有 OOV | token piece 对人不直观 |
| Pure byte/character | 字节或字符直接建模 | 无固定 subword 词表偏差 | 序列更长，需要更强架构 |

## 8. 研究发展时间线

| 时间 | 工作 | 发展意义 |
| --- | --- | --- |
| 2012 | WordPiece, Schuster and Nakajima | 用子词缓解语音搜索中的大词表/OOV |
| 2016 | Neural Machine Translation of Rare Words with Subword Units | 把 BPE 系统引入现代 NMT；https://arxiv.org/abs/1508.07909 |
| 2018 | SentencePiece | 从原始文本训练语言无关 tokenizer；https://arxiv.org/abs/1808.06226 |
| 2018 | Subword Regularization / Unigram LM | 训练时采样多种切分，提高鲁棒性；https://arxiv.org/abs/1804.10959 |
| 2018 | BERT / WordPiece | WordPiece 成为 encoder 模型主流接口；https://arxiv.org/abs/1810.04805 |
| 2019 | GPT-2 Byte-level BPE | byte 覆盖 + BPE 成为 GPT 路线的重要方案 |
| 2020 | BPE-Dropout | 随机丢 merge，增强子词切分鲁棒性；https://arxiv.org/abs/1910.13267 |
| 2021 | ByT5 / CANINE / Charformer | 重新研究 byte/character 和可学习切分，减少固定 tokenizer 偏差 |
| 2023 | MegaByte | 用分层结构降低长 byte 序列成本；https://arxiv.org/abs/2305.07185 |
| 2024 | Byte Latent Transformer | 根据 byte 序列信息动态形成 patch，探索动态 tokenization；https://arxiv.org/abs/2412.09871 |

整体趋势：

```text
固定 word vocab
  -> 固定 subword vocab（WordPiece/BPE/Unigram）
  -> byte fallback 和多语言 tokenizer
  -> subword regularization
  -> pure byte/character
  -> 内容感知的动态 patch/tokenization
```

主流生产 LLM 目前仍大量使用固定 subword/byte-level BPE，因为训练、缓存、服务和生态最成熟；研究则在尝试减少固定 tokenizer 对语言、拼写、数字、代码和多模态数据带来的偏差。

## 9. 下一步与 RoPE 的边界

Tokenizer 本轮已形成独立稳定层。下一步可分两条实验线：

1. Tokenizer 研究线：对比已实现的 SentencePiece BPE/Unigram，并继续做 subword regularization、压缩率与多语言公平性。
2. 模型结构线：learned absolute position 与 RoPE 对比。

RoPE 实现时不修改 tokenizer 接口；只修改模型 config、Q/K 旋转、KV-cache position 和各推理引擎的 MiniGPT backend。
