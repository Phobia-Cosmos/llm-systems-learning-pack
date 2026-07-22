from abc import ABC, abstractmethod

import torch


class BaseMoeBackend(ABC):
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        # TODO：w指的是不同专家的参数矩阵是吗？gating_output的作用是什么？
        # 解答：是；w1 堆叠所有专家合并后的 gate+up 投影 [E, 2I_tp, H]，w2 是 down 投影 [E, H, I_tp]。gating_output 是每个 token 对各专家的 router logits。
        w1: torch.Tensor,
        w2: torch.Tensor,

        gating_output: torch.Tensor,

        # TODO：这个是选择前k个专家的输出的意思吗？为什么需要renormalize？
        # 解答：topk 是每个 token 激活的专家数，最终会组合这些专家的输出；renormalize 将入选专家权重重新缩放到和为 1，是否启用必须遵循模型训练时的路由定义。
        topk: int,
        renormalize: bool,
        activation: str,

        # TODO：这个属性又是什么东西？发挥什么作用？
        # 解答：它决定 router 权重在专家第一层投影结果（激活前）还是最终输出处相乘；接口允许模型选择，但当前 MoEMLP 不传该参数，默认值为 False（在最终专家输出处相乘）。
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor: ...
