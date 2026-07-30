# MiniLLM Tokenizer Artifacts

这个目录用于保存不同 tokenizer 的训练产物，方便和当前 `CharTokenizer` 对比。

当前已有变体：

| 目录 | 类型 | 作用 |
| --- | --- | --- |
| `byte_bpe/` | Byte-level BPE / HF-compatible tokenizer | 更接近 Hugging Face tokenizer 生态，包含 `tokenizer.json`、special tokens 和 chat template 配置 |

## 当前 CharTokenizer

代码位置：

```text
minillm/tokenizer.py
```

特点：

- 每个字符一个 token。
- 只有 `stoi` 和 `itos`。
- 遇到训练词表外字符会变成 `<unk>`。
- 最适合教学，因为 encode/decode 极其直观。

## Byte-level BPE tokenizer

代码位置：

```text
minillm/tokenizer_variants/byte_bpe.py
```

训练命令：

```bash
python scripts/train_byte_bpe_tokenizer.py
```

对比命令：

```bash
python scripts/compare_tokenizers.py --show-tokens
```

它包含：

- `normalizer`: NFC 文本规范化，保留中文全角标点等字符形态，避免破坏 roundtrip。
- `pre-tokenizer`: byte-level 预切分。
- `model/vocab`: BPE 子词词表。
- `decoder`: byte-level decoder，把 token ids 还原为文本。
- `special tokens`: `<unk>`、`<pad>`、`<bos>`、`<eos>`、`<|system|>`、`<|user|>`、`<|assistant|>`。
- `chat template`: 把多轮 role/content 消息格式化为模型输入文本。

## 为什么不能直接替换当前 checkpoint

当前 `artifacts/checkpoints/minillm.pt` 是用 `CharTokenizer` 训练的。它的 `token_embedding.weight` 第一维对应旧词表：

```text
CharTokenizer vocab_size -> token_embedding.weight.shape[0]
```

如果换成 Byte-level BPE：

1. token id 的含义会变化。
2. `vocab_size` 会变化。
3. `token_embedding.weight` 和 `lm_head.weight` 的形状/语义都不匹配。

所以新 tokenizer 要真正接入模型，需要重新训练，或者做明确的 embedding resize 和迁移实验。当前目录先用于 tokenizer 层面的独立对比。
