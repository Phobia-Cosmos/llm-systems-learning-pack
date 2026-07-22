from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # TODO：non_blocking=True是什么意思？
    # 解答：它允许把这次 H2D 拷贝排入 CUDA stream 后尽快返回；这里源张量是 pinned memory，因此拷贝具备真正异步并与其他工作重叠的条件。
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    import flashinfer.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    # TODO：这个是词表的大小是吗？
    # 解答：是模型完整词表的 token 数，必须与 LM Head 产生的 logits 最后一维一致；Sampler 本身不决定 logits shape，只用它表示“top-k 不裁剪”的 k 值。
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        # TODO：这个函数是什么意思？all是用来判断什么的？为什么greedy就可以直接返回？
        # 解答：all 检查批内每个请求是否都为 greedy；若全是，就用 temperatures=None 标记整批直接 argmax，省去采样参数张量、softmax 和随机采样。
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        # TODO：为什么greedy就让t为0？然后在和MIN-T比较？
        # 解答：混合批不能走整批 argmax，greedy 请求便先映射为 0 再钳到极小正数 MIN_T；这样既避免除零，又让概率分布近似 one-hot。
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]

        # TODO：这又是在判断什么？为什么要top k大于1？为什么可以让ks变为vocab size？
        # 解答：代码接受所有 k >= 1；k <= 0 表示关闭 top-k，此时令 k=vocab_size 等价于保留词表中的全部候选。
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]

        # TODO：这里为什么又要先max在min 这个top ps又是用来做什么的？
        # 解答：top-p 只保留累计概率达到阈值的最小候选集合；把值钳到 [MIN_P, 1] 可避免非法的零/越界值，其中 1 表示不裁剪。
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]

        # TODO：为什么要把这些值全部放到self.device?
        # 解答：FlashInfer 采样 kernel 在 GPU 上逐请求读取这些参数，因此它们必须与 logits 位于同一 CUDA device，且用一次批量 H2D 拷贝可避免逐项同步。
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None

        # TODO：为什么要求存在k != self.vocab_size以及p < 1.0？
        # 解答：只有至少一个请求真正启用相应过滤时才传参数张量；全为 vocab_size 或 1 时传 None 可选择更简单的 kernel 并省去无效过滤。
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        # TODO：为什么要这样一行？
        # 解答：这个 NVTX range 只给性能分析器标记采样区间，方便在 Nsight 中观察耗时和 CUDA 时序，不改变采样结果。
        # 给 GPU/CPU 性能分析工具添加一个名为 Sampler 的时间区间标记（NVTX Range），方便你在 Nsight Systems 等工具中观察这段代码到底执行了多久、和其他 CUDA 操作是什么关系。
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
