from __future__ import annotations

from pathlib import Path

from .tokenizer import CharTokenizer
from .tokenizer_base import MiniTokenizer

# 问题（已回答）：registry 文件需要负责训练吗？
# 回答：它只负责选择、构造、加载和序列化 tokenizer；byte-BPE 不存在时可调用其 trainer 生成词表，
# 但不会训练 MiniGPT 模型参数，模型训练仍在 train.py。

SUPPORTED_TOKENIZERS = (
    "char",
    "byte-bpe",
    "hf-auto",
    "sentencepiece-bpe",
    "sentencepiece-unigram",
)


# 问题（已回答）：def、全局函数、* 和 tokenizer_vocab_size 分别是什么？
# 回答：def 在模块顶层定义可导入的全局函数；* 之后的参数必须按名字传入，避免多个可选参数位置混淆。
# tokenizer_vocab_size 是训练词表时的目标上限/超参数，不是永远固定；一旦模型开始训练则必须固定并与 embedding 行数一致。
def build_tokenizer(
    tokenizer_name: str,
    text: str,
    *,
    training_file: str | None = None,
    tokenizer_path: str | None = None,
    tokenizer_output_dir: str | None = None,
    tokenizer_vocab_size: int = 512,
    retrain_tokenizer: bool = False,
    trust_remote_code: bool = False,
) -> MiniTokenizer:
    if tokenizer_name == "char":
        return CharTokenizer.from_text(text)

    if tokenizer_name == "byte-bpe":
        from .tokenizer_variants import HFByteBPETokenizer

        output_dir = Path(tokenizer_output_dir or "artifacts/tokenizers/byte_bpe")

        # 问题（已回答）：为什么这里用 if/elif/else 选择路径？
        # 回答：三个来源按优先级互斥：显式 tokenizer_path 最高，其次 output_dir 下的默认文件，最后项目默认路径；
        # 一次只能加载一个 tokenizer.json，所以命中后无需继续判断。
        if tokenizer_path is not None:
            path = Path(tokenizer_path)
        else:
            path = output_dir / "tokenizer.json"

        if path.exists() and not retrain_tokenizer:
            return HFByteBPETokenizer.from_file(path)

        if training_file is None:
            # 问题（已回答）：training_file 是什么，为什么 Byte-BPE 训练需要它？
            # 回答：它是用于统计 byte/subword 频率并学习 BPE merge 的纯文本语料路径；内容可以与 CharTokenizer/模型预训练 corpus 相同。
            # CharTokenizer 只需 text 中不同字符即可建表；BPE trainer 还需反复统计片段频率，因此接口接收文件。
            raise ValueError("training_file is required when training a byte-bpe tokenizer")
        tokenizer = HFByteBPETokenizer.train([training_file], vocab_size=tokenizer_vocab_size)
        tokenizer.save_pretrained(output_dir)
        return tokenizer

    if tokenizer_name == "hf-auto":
        from .tokenizer_variants import HFTokenizerAdapter

        if tokenizer_path is None:
            raise ValueError("--tokenizer-path is required for --tokenizer hf-auto")
        return HFTokenizerAdapter.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
        )

    if tokenizer_name in {"sentencepiece-bpe", "sentencepiece-unigram"}:
        from .tokenizer_variants import SentencePieceTokenizer

        model_type = tokenizer_name.removeprefix("sentencepiece-")
        output_dir = Path(tokenizer_output_dir or f"artifacts/tokenizers/sentencepiece_{model_type}")
        path = Path(tokenizer_path) if tokenizer_path is not None else output_dir / "tokenizer.model"
        if path.is_dir():
            path = path / "tokenizer.model"
        if path.exists() and not retrain_tokenizer:
            return SentencePieceTokenizer.from_file(path, model_type=model_type)
        if training_file is None:
            raise ValueError("training_file is required when training a SentencePiece tokenizer")
        tokenizer = SentencePieceTokenizer.train(
            [training_file],
            model_type=model_type,
            vocab_size=tokenizer_vocab_size,
        )
        tokenizer.save_pretrained(output_dir)
        return tokenizer

    raise ValueError(f"Unsupported tokenizer {tokenizer_name!r}")


def tokenizer_from_checkpoint(checkpoint: dict) -> MiniTokenizer:
    tokenizer_type = checkpoint.get("tokenizer_type")
    payload = checkpoint["tokenizer"]

    if tokenizer_type is None:
        tokenizer_type = payload.get("type", "char") if isinstance(payload, dict) else "char"

    if tokenizer_type == "char":
        return CharTokenizer.from_dict(payload)
    if tokenizer_type == "byte-bpe":
        from .tokenizer_variants import HFByteBPETokenizer

        return HFByteBPETokenizer.from_dict(payload)
    if tokenizer_type == "hf-auto":
        from .tokenizer_variants import HFTokenizerAdapter

        return HFTokenizerAdapter.from_dict(payload)
    if tokenizer_type in {"sentencepiece-bpe", "sentencepiece-unigram"}:
        from .tokenizer_variants import SentencePieceTokenizer

        return SentencePieceTokenizer.from_dict(payload)

    raise ValueError(f"Unsupported tokenizer type in checkpoint: {tokenizer_type!r}")


def tokenizer_to_checkpoint_payload(tokenizer_name: str, tokenizer: MiniTokenizer) -> dict:
    payload = tokenizer.to_dict()
    if tokenizer_name == "char" and isinstance(payload, dict) and "type" not in payload:
        payload = {"type": "char", **payload}
    return payload
