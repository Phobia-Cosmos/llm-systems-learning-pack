# MiniLLM Tokenizer Variants: CharTokenizer vs Byte-level BPE

## 1. 本次实现了什么

没有覆盖当前 `CharTokenizer`。新增内容如下：

```text
minillm/tokenizer_variants/
  __init__.py
  byte_bpe.py

artifacts/tokenizers/
  README.md
  requirements.txt
  byte_bpe/
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json

scripts/
  train_byte_bpe_tokenizer.py
  compare_tokenizers.py
```

当前 `CharTokenizer` 仍然在：

```text
minillm/tokenizer.py
```

## 2. 新 tokenizer 版本：Byte-level BPE

这个版本更接近 Hugging Face tokenizer 生态。它包含：

| 组件 | 当前实现 | 作用 |
| --- | --- | --- |
| normalizer | NFC | 做保守 Unicode 规范化，尽量保留原始字符形态 |
| pre-tokenizer | ByteLevel | 把文本变成 byte-level 片段，减少未知字符 |
| model/vocab | BPE | 学习常见 byte/subword 合并 |
| decoder | ByteLevel decoder | 把 token ids 还原成原始文本 |
| special tokens | `<unk>`、`<pad>`、`<bos>`、`<eos>`、`<|system|>`、`<|user|>`、`<|assistant|>` | 控制未知词、padding、序列边界和 chat role |
| chat template | 简单 role/content 模板 | 把 system/user/assistant 消息拼成模型输入 |

训练命令：

```bash
cd /home/undefined/Desktop/ai/projects/minillm
/home/undefined/Disk/ai-storage/.venv-vllm/bin/python scripts/train_byte_bpe_tokenizer.py
```

对比命令：

```bash
cd /home/undefined/Desktop/ai/projects/minillm
/home/undefined/Disk/ai-storage/.venv-vllm/bin/python scripts/compare_tokenizers.py --show-tokens
```

## 3. 验证结果

训练后：

```text
byte_bpe_vocab_size = 512
special tokens:
  '<unk>'          -> 0
  '<pad>'          -> 1
  '<bos>'          -> 2
  '<eos>'          -> 3
  '<|system|>'     -> 4
  '<|user|>'       -> 5
  '<|assistant|>'  -> 6
```

样例：

```text
输入:
用户: 什么是 embedding?
助手:

CharTokenizer:
22 tokens，包含 1 个 <unk>

Byte-level BPE:
10 tokens，可以 decode 回原文
```

对比表：

| sample | CharTokenizer tokens | CharTokenizer unk | Char roundtrip | Byte-BPE tokens | Byte-BPE roundtrip |
| --- | ---: | ---: | --- | ---: | --- |
| `用户: 什么是 embedding?\n助手:` | 22 | 1 | True | 10 | True |
| `Hello, CUDA 13.0 + vLLM?` | 24 | 5 | False | 20 | True |
| `今天天气怎么样？` | 8 | 5 | False | 16 | True |
| `🚀 emoji and unseen text` | 23 | 1 | False | 24 | True |

chat template 验证：

```text
<|system|>
你是 MiniLLM 助手。
<|user|>
解释 embedding。
<|assistant|>
```

其中 `<|system|>`、`<|user|>`、`<|assistant|>` 能保持为独立 special token，并且可以 decode 回原文。

## 4. 两个版本的区别

| 对比项 | CharTokenizer | Byte-level BPE |
| --- | --- | --- |
| 粒度 | 每个字符一个 token | byte/subword token |
| 词表来源 | 当前训练文本中的字符集合 | 从语料中训练 BPE merge |
| 未见字符 | 变成 `<unk>` | 通常可用 byte-level token 表示 |
| token 数 | 中文/英文都按字符切，序列可能更长 | 常见片段会合并，序列可能更短 |
| 可读性 | 极高，适合初学 | token 可能显示为 byte-level 片段，不直观 |
| HF 兼容性 | 低 | 高，保存为 `tokenizer.json` 等标准文件 |
| chat template | 无 | 有基础 role 模板 |
| 当前 checkpoint 可直接使用 | 是 | 否，需要重训或做 embedding 迁移 |

## 5. 效果有什么不同

tokenizer 层面，Byte-level BPE 的效果更接近真实 LLM：

1. 更少 `<unk>`：英文标点、未见中文、emoji 不会轻易丢失。
2. 常见片段更短：例如 `embedding` 被切成 `embed` + `ding`，不是逐字符。
3. 更适合服务生态：有 `tokenizer.json`、special tokens 和 chat template。

但它不会自动让当前 MiniLLM 生成质量变好。原因是当前 checkpoint 是用 `CharTokenizer` 训练的：

```text
旧模型 token_embedding.weight: [339, 128]
新 tokenizer vocab_size:       512
```

不仅形状不同，token id 的语义也不同。比如旧的 id 36 是字符 `e`，新 tokenizer 的 id 36 不一定是同一个含义。

所以要真正比较模型效果，需要下一步：

1. 修改训练入口支持选择 tokenizer。
2. 用 Byte-level BPE 重新训练 MiniLLM。
3. 对比 loss、生成文本、token 数、训练速度和推理速度。

当前本轮完成的是 tokenizer 层面的独立实现和验证。

## 6. 训练脚本现在如何选择 tokenizer

`train.py` 已支持选择 tokenizer：

```bash
# 旧路线：字符级 tokenizer，默认不变
python train.py --tokenizer char

# 新路线：Byte-level BPE tokenizer
python train.py \
  --data data/teaching_corpus.txt \
  --out-dir artifacts/checkpoints \
  --checkpoint-name minillm-byte-bpe.pt \
  --tokenizer byte-bpe \
  --tokenizer-output-dir artifacts/tokenizers/byte_bpe \
  --tokenizer-vocab-size 512 \
  --block-size 128
```

checkpoint 中会保存：

```text
tokenizer_type
tokenizer
config
model
args
```

所以 `generate.py`、`export_hf_like.py`、`inspect_flow.py` 都可以从 checkpoint 自动恢复对应 tokenizer。

## 7. Byte-BPE 模型训练与部署验证

本次训练命令：

```bash
/home/undefined/Disk/ai-storage/.venv-vllm/bin/python train.py \
  --data data/teaching_corpus.txt \
  --out-dir artifacts/checkpoints \
  --checkpoint-name minillm-byte-bpe.pt \
  --tokenizer byte-bpe \
  --tokenizer-output-dir artifacts/tokenizers/byte_bpe \
  --tokenizer-vocab-size 512 \
  --block-size 128 \
  --batch-size 32 \
  --max-steps 300 \
  --eval-interval 100 \
  --eval-iters 10 \
  --device cuda
```

训练结果：

```text
step 0000: train loss 6.2705, val loss 6.2752
step 0100: train loss 3.3582, val loss 3.3655
step 0200: train loss 1.5213, val loss 1.5325
step 0300: train loss 0.4612, val loss 0.4628
saved checkpoint to artifacts/checkpoints/minillm-byte-bpe.pt
```

模型对比：

| 模型 | tokenizer | vocab_size | 参数量 |
| --- | --- | ---: | ---: |
| `artifacts/checkpoints/minillm.pt` | legacy char | 339 | 456,576 |
| `artifacts/checkpoints/minillm-byte-bpe.pt` | byte-bpe | 512 | 478,720 |

Byte-BPE 参数更多，主要是因为 `token_embedding.weight` / tied `lm_head.weight` 的第一维从 339 变成 512。

导出命令：

```bash
python export_hf_like.py \
  --checkpoint artifacts/checkpoints/minillm-byte-bpe.pt \
  --out-dir artifacts/hf_exports/minillm-byte-bpe \
  --safe-serialization
```

导出目录包含：

```text
config.json
generation_config.json
model.safetensors
README.md
special_tokens_map.json
tokenizer_config.json
tokenizer.json
```

部署验证结果：

| 路径 | 结果 | 说明 |
| --- | --- | --- |
| `generate.py` | 通过 | checkpoint 自动恢复 Byte-BPE tokenizer 并生成 |
| `transformers.AutoTokenizer` | 通过 | 可加载 `artifacts/hf_exports/minillm-byte-bpe`，prompt roundtrip 正常 |
| vLLM Python engine | 通过 | 直接传字符串 prompt，vLLM 使用标准 tokenizer 编码 |
| vLLM OpenAI-compatible server | 通过 | `/health` 和 `/v1/completions` 返回成功 |
| nano-vLLM | 通过 | MiniGPT tokenizer loader 已扩展为支持 HF tokenizer |
| mini-sglang 教学服务 | 通过 | checkpoint tokenizer loader 已扩展为自动识别 tokenizer 类型 |

注意：这只是小语料 300 step 的功能验证，不代表模型质量已经好。生成仍可能重复、混杂或输出乱码片段。

## 8. 除了 Byte-level BPE，还有哪些 tokenizer

| 类型 | 代表模型/工具 | 特点 | 适合学习什么 |
| --- | --- | --- | --- |
| Char tokenizer | MiniLLM 当前 legacy 版本 | 每个字符一个 token，最直观 | token/id/embedding/logits 的基本关系 |
| Word tokenizer | 早期 NLP | 按词切分，词表容易爆炸 | 为什么需要 subword |
| BPE | GPT-2、很多 LLM | 从字符/byte 开始合并高频片段 | merge 规则、压缩率 |
| Byte-level BPE | GPT-2/tiktoken 路线 | byte fallback 强，未知字符少 | 真实 GPT tokenizer 思路 |
| WordPiece | BERT | 常见于 encoder-only 模型 | masked LM 生态 |
| SentencePiece BPE | Llama/Qwen 等常见 | 不依赖空格分词，多语言友好 | 多语言 tokenizer |
| SentencePiece Unigram | T5 等 | 基于概率选择子词 | tokenizer 训练目标差异 |
| tiktoken | OpenAI GPT 系列常用 | 工程速度快，BPE 规则成熟 | 服务侧高性能编码 |

建议后续 tokenizer 迭代顺序：

1. 当前 Byte-level BPE：已经完成。
2. SentencePiece BPE：更接近 Llama/Qwen 生态。
3. tiktoken 风格 BPE：更接近 OpenAI/GPT 服务生态。
4. 对比不同 tokenizer 的 token 数、unk 数、训练 loss、生成质量和推理速度。

## 9. 当前 MiniLLM 是否实现了 RoPE

已经实现，并保留 learned absolute position 作为旧 checkpoint 兼容模式：

```text
position_encoding = "learned" | "rope"
rope_theta = 10000.0
```

RoPE 的特点是：不再把一个位置向量加到 token embedding 上，而是在 attention 内部对 Q/K 做旋转位置编码。也就是说 RoPE 的改动边界主要在 attention：

```text
learned:
token_embedding + position_embedding -> Q/K/V

RoPE:
token_embedding -> Q/K/V -> apply_rotary_pos_emb(Q,K,positions)
```

已同步完成 MiniLLM、nano-vLLM 和 vLLM MiniGPT backend。`mini-sglang` 直接加载 MiniLLM checkpoint，因此也能运行 RoPE；上游 SGLang native backend 尚未实现。实现与验证细节见 `docs/rope_implementation_and_roadmap.md`。
