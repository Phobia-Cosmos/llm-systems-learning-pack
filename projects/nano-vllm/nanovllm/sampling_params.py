from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    # 问题（已回答）：采样发生在 logits 之后，为什么还要 max_tokens？
    # 回答：temperature 决定如何从当前 logits 选下一个 token；max_tokens 是请求级停止预算，限制最多重复
    # “forward -> logits -> sample”多少次，防止请求无限生成，也供 scheduler 预先检查最大序列长度。
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
