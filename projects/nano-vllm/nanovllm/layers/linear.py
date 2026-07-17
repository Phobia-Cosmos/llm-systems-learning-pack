import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# 问题（已回答）：为什么有多种 Linear，weight_loader 有什么区别？
# 回答：F.linear 的数学形式相同，区别在 TP 下权重沿哪一维切、输出何时通信以及多个逻辑矩阵是否打包。
# Replicated 保存完整权重；ColumnParallel 切输出行；MergedColumn 先按逻辑矩阵分组再切行；QKV 还处理 Q/K/V
# 和 GQA 的不同宽度；RowParallel 切输入列并 all-reduce 部分和。loader 必须把完整 HF checkpoint 转成各自本地布局。
class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 问题（已回答）：这里传入的是 tp_size 还是 tp_dim？
        # 回答：不是把 tp_size 传给 tp_dim。divide(output_size,tp_size) 先计算本 rank 的输出行数，最后一个参数 0
        # 才是 tp_dim，表示 checkpoint weight 沿第 0 维（out_features/行）切分。
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # 问题（已回答）：shard_size 和 start_idx 在计算什么？
        # 回答：param_data 已是本 rank 的局部矩阵；其第 tp_dim 维长度就是每 rank shard 大小，
        # start_idx=rank*shard_size 定位当前 rank 在完整 checkpoint 对应维度上的连续区间。
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        # 问题（已回答）：为什么要 narrow loaded_weight？
        # 回答：loaded_weight 是完整权重，而本地 Parameter 只能容纳当前 rank 的行；narrow 返回该连续区间的 view，
        # 不产生额外完整拷贝，再 copy_ 到本地。也可用 split/chunk，但 narrow 可直接表达起点和精确长度。
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

# 问题（已回答）：MergedColumnParallelLinear 为什么没有 forward，何时使用？
# 回答：它继承 ColumnParallelLinear.forward，一次 F.linear 就输出按最后一维拼接的多个逻辑投影，因此无需重写。
# 它用于 gate+up、以及其他共享同一输入且可打包的投影：减少输入读取和 kernel launch，之后再 split/chunk 结果。
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        # 问题（已回答）：loaded_shard_id 的作用是什么？
        # 回答：它标识当前 loaded_weight 属于 packed 参数中的第几个逻辑矩阵，例如 gate=0、up=1。
        # loader 读取分开的 HF 权重时会传 id；若读到已经打包的完整权重则为 None，由下面先拆分。
        loaded_shard_id: int | None = None,
    ):
        if loaded_shard_id is None:
            # 问题（已回答）：为什么先 split 再递归加载？
            # 回答：完整 packed checkpoint 的第 0 维依次放着 output_sizes 指定的多个逻辑矩阵；split 还原这些矩阵。
            # 每个逻辑矩阵都要先单独按 TP rank 切分，再放入本地 packed 参数的对应区域；若直接整体均分，
            # 不同大小矩阵或每段边界可能被切错。递归复用“已知 shard_id”的统一装载分支。
            for shard_id, shard in enumerate(loaded_weight.split(self.output_sizes, self.tp_dim)):
                self.weight_loader(param, shard, shard_id)
            return
        param_data = param.data
        # 问题（已回答）：shard_offset 和 shard_size 为什么这样计算？
        # 回答：前面逻辑矩阵在本地各只保留全局宽度的 1/tp_size，所以其本地长度之和给出 destination offset；
        # 当前逻辑矩阵的全局 output_size/tp_size 则是本地 destination 长度，narrow 定位该段 Parameter view。
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 问题（已回答）：loaded_weight 为什么按 TP chunk？
        # 回答：此时 loaded_weight 是某一个完整逻辑矩阵；Column Parallel 要按输出行均分给所有 rank，
        # chunk(...)[tp_rank] 选出当前 rank 的那一份，再写入上面定位的本地 packed 区域。
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

# 问题（已回答）：QKVParallelLinear 为什么继承 ColumnParallelLinear？
# 回答：Q/K/V 都是从同一 hidden_states 投影出的输出通道，天然适合沿 head/输出行分到 TP rank，forward 可复用
# ColumnParallel 的单次 F.linear。它只需定制 packed 输出总宽度和 loader，因为 GQA 下 Q head 数与 KV head 数不同。
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # 问题（已回答）：QKV fused Linear 的 output_size 为什么这样计算？
        # 回答：每个 head 有 head_size 个标量；Q 需要 total_num_heads*D，K 和 V 各需要
        # total_num_kv_heads*D，所以拼接宽度是 (Hq+2*Hkv)*D。MHA 时 Hq=Hkv，退化为 3*hidden_size。
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        # 问题（已回答）：QKV loaded_shard_id 何时为 None？
        # 回答：MiniGPT 等 checkpoint 若已经用一个 c_attn/qkv_proj 保存 [Q|K|V] packed 权重，通用 loader
        # 会直接调用且不传 shard id；Qwen HF checkpoint 通常分开保存 q_proj/k_proj/v_proj，packed_modules_mapping
        # 会分别传 "q"、"k"、"v"。
        if loaded_shard_id is None:
            # 问题（已回答）：为什么 KV head 宽度出现两次？
            # 回答：K 和 V head 数相同，但它们是参数完全不同的两套投影矩阵，所以 packed 权重中各占一段 Hkv*D；
            # 这不是重复加载同一权重，而是分别描述 [Q|K|V] 三段的长度。
            shard_sizes = [
                self.total_num_heads * self.head_size,
                self.total_num_kv_heads * self.head_size,
                self.total_num_kv_heads * self.head_size,
            ]
            # 问题（已回答）：这里 zip 的作用是什么？
            # 回答：split 按三个长度返回 Q/K/V Tensor，zip 将语义标签 ("q","k","v") 与对应 Tensor 一一配对，
            # 循环即可调用同一 loader 分支；比依赖数字 0/1/2 更清楚，也与 packed_modules_mapping 的 shard id 一致。
            for shard_id, shard in zip(("q", "k", "v"), loaded_weight.split(shard_sizes, self.tp_dim)):
                self.weight_loader(param, shard, shard_id)
            return
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        # 问题（已回答）：Q/K/V 是否顺序存放在同一片参数中？
        # 回答：是，本地 param_data 的第 0 维布局为 [Q_local | K_local | V_local]。下面根据 shard_id 计算
        # 每段 offset/size，再从对应完整 Q/K/V 权重取本 rank shard 写入；forward 后可按相同顺序 split 输出。
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # 问题（已回答）：RowParallel loader 为什么判断 param_data.ndim==1？
        # 回答：二维 weight 要沿输入列 tp_dim=1 切分；一维 bias 没有输入维，而且最终输出在各 rank all-reduce 后共享，
        # 因此 bias 应完整复制。forward 只让 rank 0 加一次 bias，避免 all-reduce 后被重复 TP 次。
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        # 问题（已回答）：RowParallel forward 为什么 all_reduce？
        # 回答：输入和 weight 都沿 input_features 切片，每个 rank 只能计算 y_r=x_r W_r^T 的部分和；
        # 完整线性层满足 y=sum_r y_r，因此必须 all_reduce(sum)。bias 只在 rank 0 加入一次，求和后所有 rank 得到相同完整 y。
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
