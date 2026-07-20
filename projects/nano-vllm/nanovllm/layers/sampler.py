import torch
from torch import nn


class Sampler(nn.Module):

    # 问题（已回答）：为什么这里使用 @torch.compile？
    # 回答：采样会在每轮 decode 重复执行，compile 可把类型转换、温度缩放、softmax 和随机采样组织成
    # 优化后的计算图，减少 Python 与 kernel 调度开销；它是性能优化，不是保证采样正确性的必要条件。
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 问题（已回答）：为什么 temperatures 需要 unsqueeze？
        # 回答：logits 的形状是 [B,V]，temperatures 原本是 [B]；变成 [B,1] 后才能沿词表维 V 广播，
        # 让每个请求的一个温度除以该请求的全部 logits，而不会与 V 维错误对齐。
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # 问题（已回答）：这里使用什么采样公式，为什么需要 empty_like 和 clamp_min_？
        # 回答：为每个概率 p_i 生成独立的 E_i~Exp(1)，再取 argmax(p_i/E_i)；它等价于指数竞赛
        # （也等价于 Gumbel-max）并按 probs 的 categorical 分布采样。empty_like 创建同形状、同设备、同 dtype
        # 的临时 Tensor，exponential_(1) 原地填入指数随机数，clamp_min_(1e-10) 防止分母过小或为零。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
