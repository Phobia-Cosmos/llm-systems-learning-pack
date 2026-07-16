from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

Tensor = torch.Tensor
# 这些是类型别名，不会创建 Tensor 或函数对象。Tensor 让后续标注比反复写 torch.Tensor 更简洁；
# Callable[[参数类型...], 返回类型] 的方括号是泛型类型参数，不是数组。Operator 中的 ... 表示各算子参数列表可以不同，
# 返回值可以是一个 Tensor，也可以是由任意个 Tensor 组成的 tuple。InputFactory 固定接收 device、dtype、profile：
# 第二个参数 torch.dtype 决定生成 float32/float16/bfloat16 等输入，最后返回可直接用 *inputs 解包给算子的参数 tuple。
Operator = Callable[..., Tensor | tuple[Tensor, ...]]
InputFactory = Callable[[torch.device, torch.dtype, str], tuple[object, ...]]


@dataclass(frozen=True)
class OperatorCase:
    """保存一项可复现的算子对照实验配置。

    dataclass 自动生成 __init__/__repr__/__eq__；frozen=True 表示实例创建后字段不能重新赋值，避免 benchmark
    运行中意外更换 reference、输入工厂或容差。它只冻结字段引用，不会把字段内部的 Tensor 深度冻结。
    """

    name: str
    # family 用于按 elementwise/reduction/matrix 等依赖模式分组、筛选和汇总结果，不参与数值计算。
    family: str
    pytorch_reference: Operator
    teaching_python: Operator
    make_inputs: InputFactory
    description: str
    # torch.testing.assert_close 使用 |actual-expected| <= atol + rtol*|expected|：atol 是接近 0 时仍允许的绝对误差，
    # rtol 是随参考值幅度增长的相对误差。浮点归约顺序不同会产生舍入差，所以不能一律要求逐 bit 相等。
    atol: float = 1e-5
    rtol: float = 1e-4

# TODO:这里实现的这么多的函数目的是什么？这里不是自定义的python版本kernel 算子吧？这些函数会放到GPU上执行是吗？
def vector_add(a: Tensor, b: Tensor) -> Tensor:
    """逐元素向量/张量加法。

    数学公式：C_i = A_i + B_i；对多维 Tensor，i 表示同一逻辑坐标并遵循 PyTorch broadcasting。
    原理：每个输出只依赖相同位置的两个输入，没有跨元素归约，典型 GPU kernel 让每个 thread 处理一个或多个元素。
    场景：Transformer residual add、bias 前后的逐元素组合、梯度累加和通用数值计算。
    类型说明：这里的 Tensor 只是上方 torch.Tensor 的类型别名；C++ 通常直接在形参写 Tensor/指针类型，Python 为了让多个函数标注简洁才定义别名，并非 Python 额外需要创建一个 Tensor 变量。
    """

    return a + b


def saxpy(x: Tensor, y: Tensor, alpha: float) -> Tensor:
    """计算缩放向量加法 SAXPY。

    数学公式：Z_i = αX_i + Y_i，其中 alpha=α 是对整条 x 使用的标量缩放系数。
    原理：先逐元素乘 alpha，再与 y 相加；BLAS 名称 SAXPY 原指 Single-precision A·X Plus Y，本教学函数也可接受其他浮点 dtype。
    场景：优化器参数更新、线性插值、残差缩放、数值线性代数；CUDA 上常用 FMA 把乘法和加法合成一次舍入。
    """

    return alpha * x + y


def relu_from_where(x: Tensor) -> Tensor:
    """用条件选择展开 ReLU，而不是调用 F.relu。

    数学公式：ReLU(x) = max(0,x) = x（x>0），否则为 0。
    原理：torch.where 根据逐元素 predicate 选择 x 或同 device/dtype 的标量零；输出 shape 与 x 相同。
    场景：CNN/MLP 激活、稀疏化负激活；函数用于教学比较，生产中优先使用已经优化的 F.relu 或融合 epilogue。
    """

    return torch.where(x > 0, x, torch.zeros((), dtype=x.dtype, device=x.device))


def sigmoid_from_exp(x: Tensor) -> Tensor:
    """从指数函数实现 Logistic sigmoid。

    数学公式：σ(x) = 1 / (1 + exp(-x))，输出范围为 (0,1)。
    原理：把任意实数平滑压缩成概率/门值；极大绝对值会使梯度接近 0，朴素公式对极端负值还可能发生 exp overflow。
    场景：二分类概率、LSTM/GRU gate、SiLU/GLU 等门控结构；教学版展示公式，稳定性和融合性能通常不及专用 torch.sigmoid。
    """

    return 1.0 / (1.0 + torch.exp(-x))


def silu_from_primitives(x: Tensor) -> Tensor:
    """用乘法与 sigmoid 实现 SiLU/Swish 激活。

    数学公式：SiLU(x) = x·σ(x) = x/(1+exp(-x))。
    原理：保留正值并平滑抑制负值，处处可导；本 eager 版本的 sigmoid 和 multiply 可能分别 launch kernel。
    场景：现代 Transformer MLP、SwiGLU 门控和视觉网络；热点路径通常使用 torch.compile/Triton/CUDA 做融合。
    """

    return x * sigmoid_from_exp(x)


def fused_bias_silu_from_primitives(x: Tensor, bias: Tensor) -> Tensor:
    """先按最后一维广播 bias，再执行 SiLU。

    数学公式：Z_{r,c}=X_{r,c}+b_c，Y_{r,c}=Z_{r,c}·σ(Z_{r,c})。
    原理：bias shape 为 [columns]，广播到 x 的所有前导行；eager 教学版仍是多个 PyTorch primitive，torch.compile/Triton 可把 add、exp/sigmoid、mul 合成一个 kernel 并减少中间 GMEM 流量。
    场景：Linear/卷积输出的 bias+activation epilogue、Transformer MLP；也是本项目从 Python callable 走向 Triton 的完整示例。
    """

    z = x + bias
    return z * sigmoid_from_exp(z)


def gelu_exact_from_erf(x: Tensor) -> Tensor:
    """用误差函数实现精确形式的 GELU。

    数学公式：GELU(x) = x·Φ(x) = 0.5x[1+erf(x/√2)]，Φ 是标准正态分布 CDF。
    原理：按输入在高斯分布下的累计概率平滑缩放 x，不像 ReLU 那样在 0 处硬截断。
    场景：BERT、GPT 等 Transformer MLP；精确 erf 版通常比 tanh 近似计算更贵。
    """

    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_tanh_from_primitives(x: Tensor) -> Tensor:
    """用 tanh 多项式近似 GELU。

    数学公式：GELU(x) ≈ 0.5x{1+tanh[√(2/π)(x+0.044715x³)]}。
    原理：用 tanh 与低阶多项式近似标准正态 CDF，减少 erf 实现成本，但结果不会与精确 GELU 逐 bit 相同。
    场景：允许近似误差、追求较快激活的 Transformer MLP；必须让 reference 也选择 approximate='tanh' 才是同一语义。
    """

    coefficient = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(coefficient * (x + 0.044715 * x * x * x)))


# TODO：这是在求倒数第二层网络的参数和吗？
def tree_sum_last_dim(x: Tensor) -> Tensor:
    """用成对树形结构求最后一维之和。

    数学公式：y_{i_1,...,i_{d-1}} = Σ_{k=0}^{K-1} x_{i_1,...,i_{d-1},k}。
    原理：每轮把偶数位与相邻奇数位相加，使 K 约减半，经过 ceil(log₂K) 层得到一个值；K 为奇数时补加法单位元 0，保证末元素不丢失。GPU 高性能 reduction 会在线程/warp/CTA 内做类似树形合并，但本 eager 教学版每一层都是独立 Tensor op/launch，因此不以性能为目标。
    场景：loss 汇总、统计量、softmax 分母、norm 平方和与矩阵乘 K 归约；不同加法顺序会产生浮点舍入差异。
    """

    values = x
    while values.shape[-1] > 1:
        if values.shape[-1] % 2:
            # TODO：这几个函数的作用分别是什么？
            padding = torch.zeros_like(values[..., :1])
            values = torch.cat((values, padding), dim=-1)
            # TODO：这个是什么语法？
        values = values[..., 0::2] + values[..., 1::2]
    return values.squeeze(-1)


def tree_mean_last_dim(x: Tensor) -> Tensor:
    """用树形求和实现最后一维算术平均值。

    数学公式：mean(x) = (1/K)Σ_{k=0}^{K-1}x_k。
    原理：复用 tree_sum_last_dim 得到总和，再除以原始 K；必须在 reduction 前记住 x.shape[-1]，不能除以补零后的长度。
    场景：LayerNorm/BatchNorm 统计、loss 平均和数据分析；生产实现通常把 sum 与除法融合或采用专用 reduction kernel。
    """

    return tree_sum_last_dim(x) / x.shape[-1]


def tree_max_last_dim(x: Tensor) -> Tensor:
    """用成对树形结构求最后一维最大值。

    数学公式：y = max_{0≤k<K} x_k。
    原理：每轮对相邻元素做 maximum；奇数长度补 max 运算的单位元 -∞，因为 max(x,-∞)=x，最终在 O(log K) 层后得到最大值。
    场景：稳定 softmax 的减最大值、max pooling、argmax/top-k 的子步骤和统计；若还要索引，必须同时传播 winner index，本函数只返回 value。
    """

    values = x
    while values.shape[-1] > 1:
        if values.shape[-1] % 2:
            padding = torch.full_like(values[..., :1], -torch.inf)
            values = torch.cat((values, padding), dim=-1)
        values = torch.maximum(values[..., 0::2], values[..., 1::2])
    return values.squeeze(-1)


def stable_softmax(x: Tensor) -> Tensor:
    """实现数值稳定的最后一维 Softmax。

    数学公式：p_i = exp(x_i-m)/Σ_j exp(x_j-m)，m=max_j x_j；减同一常数不改变 softmax 比例。
    原理：先转 FP32，再减最大值，使最大指数为 exp(0)=1，避免大正数 exp overflow；归一化后转回输入 dtype。所有输出非负且每行和约为 1。
    场景：attention probability、分类概率和采样 logits；生产 kernel 通常把 max reduction、exp、sum reduction、divide 融合，避免多次读写。
    """

    work = x.float()
    shifted = work - torch.amax(work, dim=-1, keepdim=True)
    numerator = torch.exp(shifted)
    return (numerator / torch.sum(numerator, dim=-1, keepdim=True)).to(x.dtype)


def stable_log_softmax(x: Tensor) -> Tensor:
    """实现数值稳定的最后一维 LogSoftmax。

    数学公式：log p_i = x_i - logΣ_j exp(x_j)；稳定形式为 (x_i-m)-logΣ_j exp(x_j-m)。
    原理：在 log domain 直接得到对数概率，避免先算很小的 softmax 再取 log 造成 underflow；中间使用 FP32。
    场景：NLLLoss/CrossEntropy、概率模型与 beam score；通常与负对数似然融合。
    """

    work = x.float()
    maximum = torch.amax(work, dim=-1, keepdim=True)
    shifted = work - maximum
    return (shifted - torch.log(torch.sum(torch.exp(shifted), dim=-1, keepdim=True))).to(x.dtype)


def layer_norm_from_formula(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    """按最后一维展开 LayerNorm 公式。

    数学公式：μ=(1/H)Σ_jx_j，σ²=(1/H)Σ_j(x_j-μ)²，y_j=γ_j(x_j-μ)/√(σ²+ε)+β_j。
    原理：每一行/token 独立统计 hidden 维；FP32 中间量降低低精度归约误差，eps 防止方差为 0 时除零，weight=γ 与 bias=β 是长度 H 的可学习仿射参数。
    场景：Transformer pre-norm/post-norm、MLP 和序列模型；高性能实现会融合 mean/variance、normalize 和 affine。
    """

    work = x.float()
    mean = torch.mean(work, dim=-1, keepdim=True)
    variance = torch.mean((work - mean) ** 2, dim=-1, keepdim=True)
    normalized = (work - mean) * torch.rsqrt(variance + eps)
    return (normalized * weight.float() + bias.float()).to(x.dtype)


def rms_norm_from_formula(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """按最后一维展开 RMSNorm。

    数学公式：rms=√[(1/H)Σ_jx_j²+ε]，y_j=γ_jx_j/rms。
    原理：只用平方均值衡量尺度，不减去均值，也通常没有 bias；比 LayerNorm 少一组中心化依赖，中间转 FP32 后再转回输入 dtype。
    场景：LLaMA、Qwen 等现代 LLM；常与 residual add 融合成 add-RMSNorm kernel。
    """

    work = x.float()
    inverse_rms = torch.rsqrt(torch.mean(work * work, dim=-1, keepdim=True) + eps)
    return (work * inverse_rms * weight.float()).to(x.dtype)


def matmul_from_broadcast(a: Tensor, b: Tensor) -> Tensor:
    """用 broadcasting multiply 加 K 归约展示二维矩阵乘语义。

    数学公式：若 A∈R^{M×K}、B∈R^{K×N}，则 C_{m,n}=Σ_{k=0}^{K-1}A_{m,k}B_{k,n}。
    原理：A.unsqueeze(-1) 变为 [M,K,1]，B.unsqueeze(-3) 变为 [1,K,N]，广播乘法显式产生 [M,K,N]，再对 K 所在的 -2 维求和。
    场景：教学验证 GEMM 的乘积与归约依赖；它会物化巨大的 M×K×N 临时张量，真实 Linear/GEMM 必须用 torch.matmul/cuBLAS/CUTLASS 等 tiled kernel，不能把本函数当高性能实现。
    """

    return torch.sum(a.unsqueeze(-1) * b.unsqueeze(-3), dim=-2)


def linear_from_matmul(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """用矩阵乘与广播加法展开全连接层。

    数学公式：Y = XWᵀ + b；逐元素写作 y_{m,n}=Σ_k x_{m,k}w_{n,k}+b_n。
    原理：PyTorch Linear 的 weight layout 是 [out_features=N,in_features=K]，所以先 transpose 成 [K,N]；bias [N] 沿所有 M 行广播。
    场景：Transformer Q/K/V、attention output、MLP up/down/gate、lm-head；prefill 与 decode 数学相同但 M 大小不同，可能由 backend 选择不同 kernel。
    """

    return x @ weight.transpose(-2, -1) + bias


def batched_matmul_from_broadcast(a: Tensor, b: Tensor) -> Tensor:
    """用广播与归约展示 batched matrix multiplication。

    数学公式：C_{b,m,n}=Σ_k A_{b,m,k}B_{b,k,n}，每个 batch b 独立做一次 GEMM。
    原理：在 batch 维保持一一对应，只扩展 M/K/N 位置形成 [B,M,K,N] 临时结果，再沿 K 求和。
    场景：多头 attention 的 QKᵀ/PV、小批量矩阵运算；教学版临时量大，生产使用 torch.bmm、strided-batched GEMM 或 fused attention。
    """

    return torch.sum(a.unsqueeze(-1) * b.unsqueeze(-3), dim=-2)


def embedding_from_one_hot(indices: Tensor, weight: Tensor) -> Tensor:
    """用 one-hot 矩阵乘展示 Embedding 查表语义。

    数学公式：若 e_i 是 token id i 的 one-hot 行向量，则 embedding(i)=e_iW=W[i,:]。
    原理：把每个整数 index 编成 vocab 长度 one-hot，再乘 [vocab,hidden] 权重；结果 shape 为 indices.shape+[hidden]。
    场景：token/position/expert embedding。真实 F.embedding 直接 gather 所需行，复杂度与流量远小于创建 one-hot，本函数只用于说明“查表等价于稀疏矩阵乘”。
    """

    encoded = F.one_hot(indices, num_classes=weight.shape[0]).to(weight.dtype)
    return encoded @ weight


def gather_from_one_hot(x: Tensor, indices: Tensor) -> Tensor:
    """用 one-hot selector 展示最后一维 gather。

    数学公式：Y_{r,j}=X_{r,I_{r,j}}，I 是运行时索引。
    原理：为每个 index 构造 columns 长度的 one-hot，与对应 x 行相乘并对 columns 求和，只留下被选择位置。
    场景：paged KV/稀疏索引、候选选择和表查找；真实 torch.gather 做间接读，不应创建 one-hot，性能重点是地址局部性与合并访存。
    """

    selector = F.one_hot(indices, num_classes=x.shape[-1]).to(x.dtype)
    return torch.sum(selector * x.unsqueeze(-2), dim=-1)


def scatter_add_from_one_hot(src: Tensor, indices: Tensor, output_size: int) -> Tensor:
    """用 one-hot destination 展示最后一维 scatter-add。

    数学公式：Y_{r,c}=Σ_{j:I_{r,j}=c}src_{r,j}；多个 j 指向同一 c 时必须相加。
    原理：每个 index 生成 output_size 长度目的 one-hot，乘 src 后沿输入 values 维求和，天然展示重复 index 的归约语义。
    场景：MoE token 回填、histogram、图聚合和稀疏梯度；真实实现使用 scatter/atomic/reduction，one-hot 中间量只适合教学小 shape。
    """

    destinations = F.one_hot(indices, num_classes=output_size).to(src.dtype)
    return torch.sum(destinations * src.unsqueeze(-1), dim=-2)


def apply_rope_from_pairs(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """对最后一维相邻偶/奇通道执行 Rotary Position Embedding。

    数学公式：[y_{2i},y_{2i+1}]ᵀ = [[cosθ_i,-sinθ_i],[sinθ_i,cosθ_i]]·[x_{2i},x_{2i+1}]ᵀ。
    原理：把 head_dim 拆成 even/odd pair，按 token position 与频率 θ 做二维旋转；旋转保持每对向量范数，并把相对位置信息编码进 Q/K 相位。
    场景：LLaMA/Qwen 等 Transformer 的 Q/K 位置编码；head_dim 必须为偶数，cos/sin 需能广播到 [batch,heads,tokens,head_dim/2]。
    """

    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def causal_attention_from_primitives(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """用基础算子展开 causal scaled dot-product attention。

    数学公式：S=QKᵀ/√D，P=softmax(S+M)，O=PV；causal mask M_{i,j}=0（j≤i），否则 -∞。
    原理：Q/K 点积衡量每个 query-key 对，相除 √D 控制方差，下三角 mask 阻止看到未来 token，softmax 得权重后加权求和 V。
    场景：decoder-only Transformer prefill；本函数会显式物化 [B,H,T,T] scores/probabilities，生产优先用 FlashAttention/SDPA 在线分块避免 O(T²) 中间显存。这里只构造方形 self-attention，非方形 decode KV-cache 还要按绝对 query/key position 构造 mask。
    """

    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-2, -1)) * scale
    query_length = q.shape[-2]
    key_length = k.shape[-2]
    mask = torch.ones((query_length, key_length), dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~mask, -torch.inf)
    probabilities = stable_softmax(scores)
    return probabilities @ v


def cross_entropy_from_logsumexp(logits: Tensor, targets: Tensor) -> Tensor:
    """用 logsumexp 与 gather 展开多分类交叉熵。

    数学公式：L=-(1/R)Σ_r log[exp(z_{r,y_r})/Σ_c exp(z_{r,c})] = -(1/R)Σ_r[z_{r,y_r}-logΣ_c exp(z_{r,c})]。
    原理：logsumexp 稳定计算归一化项，gather 选择每行 target 类的 log probability，最后对 rows 求平均；FP32 中间量降低低精度 overflow/underflow。
    场景：LLM next-token prediction、分类训练；真实 F.cross_entropy 常融合 log-softmax 与 NLL，并支持 ignore_index、class weight、label smoothing，本教学版只覆盖默认语义。
    """

    work = logits.float()
    log_probabilities = work - torch.logsumexp(work, dim=-1, keepdim=True)
    selected = torch.gather(log_probabilities, -1, targets.unsqueeze(-1)).squeeze(-1)
    return (-torch.mean(selected)).to(logits.dtype)


def mse_from_formula(prediction: Tensor, target: Tensor) -> Tensor:
    """实现均方误差 Mean Squared Error。

    数学公式：MSE=(1/N)Σ_i(ŷ_i-y_i)²。
    原理：先求逐元素残差，再平方并对全部元素取平均；平方会更重地惩罚大误差。
    场景：回归、重建、蒸馏特征匹配；对 outlier 敏感，分类 logits 通常更适合 cross entropy。
    """

    difference = prediction - target
    return torch.mean(difference * difference)


def topk_from_sort(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """用完整降序排序实现最后一维 Top-K。

    数学定义：返回集合中最大的 k 个 value 及其原 index，满足 values_0≥...≥values_{k-1}。
    原理：torch.sort 先排列全部 vocab/columns，再切前 k；语义直观但做了比 Top-K 必需更多的排序工作。
    场景：LLM top-k sampling、beam search、检索候选；生产 torch.topk 可使用 selection/partial sort，k≪vocab 时通常更高效。
    """

    values, indices = torch.sort(x, dim=-1, descending=True)
    return values[..., :k], indices[..., :k]


def batch_norm_eval_from_formula(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    eps: float,
) -> Tensor:
    """展开 BatchNorm 的 inference/eval 公式。

    数学公式：y_{n,c,...}=γ_c(x_{n,c,...}-μ_c)/√(v_c+ε)+β_c，其中 μ/v 是训练期累计的 running statistics。
    原理：每个 channel 使用一组固定 mean/variance/weight/bias；shape=(1,C,1,...) 让 channel 参数沿 batch 与空间维广播。这里只实现 eval，训练版还要从当前 mini-batch 统计均值/方差并更新 running state。
    场景：CNN/视觉模型推理；部署时常把 BatchNorm 的 affine 变换折叠进前一层 convolution 权重与 bias。
    """

    shape = (1, -1) + (1,) * (x.ndim - 2)
    mean = running_mean.reshape(shape)
    variance = running_var.reshape(shape)
    scale = weight.reshape(shape)
    offset = bias.reshape(shape)
    return (x - mean) * torch.rsqrt(variance + eps) * scale + offset


def conv2d_from_unfold(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    stride: int,
    padding: int,
) -> Tensor:
    """用 im2col/unfold 加矩阵乘实现二维卷积。

    数学公式：Y_{n,o,h,w}=b_o+Σ_{c,r,s}X_{n,c,h·stride+r-padding,w·stride+s-padding}W_{o,c,r,s}。
    原理：F.unfold 把每个滑动窗口摊成长度 Cin·Kh·Kw 的 column，卷积核摊成 [Cout,Cin·Kh·Kw]，einsum/GEMM 一次计算所有窗口，再 reshape 回 NCHW。
    场景：CNN、视觉/音频前端；im2col 说明卷积为何可 lower 成 GEMM，但会复制重叠 patch，cuDNN/Inductor 还可能选择 direct、implicit GEMM、Winograd 等更优算法。
    """

    batch, _channels, height, width = x.shape
    out_channels, _in_channels, kernel_h, kernel_w = weight.shape
    columns = F.unfold(x, (kernel_h, kernel_w), padding=padding, stride=stride)
    flattened_weight = weight.reshape(out_channels, -1)
    output = torch.einsum("oc,ncl->nol", flattened_weight, columns)
    output = output + bias.reshape(1, -1, 1)
    output_h = (height + 2 * padding - kernel_h) // stride + 1
    output_w = (width + 2 * padding - kernel_w) // stride + 1
    return output.reshape(batch, out_channels, output_h, output_w)


def max_pool2d_from_unfold(x: Tensor, kernel_size: int, stride: int) -> Tensor:
    """用 unfold 与 max reduction 实现二维最大池化。

    数学公式：Y_{n,c,h,w}=max_{0≤r,s<K}X_{n,c,h·stride+r,w·stride+s}。
    原理：把每个 K×K 窗口展开后对窗口元素维求最大值；本版本没有 padding/dilation/ceil_mode，输出尺寸按 floor 规则计算。
    场景：CNN 下采样、局部显著特征选择；生产 F.max_pool2d 直接读取窗口并归约，不会物化完整 unfold 临时张量。
    """

    batch, channels, height, width = x.shape
    columns = F.unfold(x, kernel_size=(kernel_size, kernel_size), stride=stride)
    columns = columns.reshape(batch, channels, kernel_size * kernel_size, -1)
    pooled = torch.amax(columns, dim=2)
    output_h = (height - kernel_size) // stride + 1
    output_w = (width - kernel_size) // stride + 1
    return pooled.reshape(batch, channels, output_h, output_w)

def _shape(profile: str, smoke: tuple[int, ...], llm: tuple[int, ...]) -> tuple[int, ...]:
    """根据 benchmark profile 在快速验证 shape 与代表性大 shape 之间选择。

    smoke 用小且经常非 2 次幂/非整齐的维度快速暴露 broadcasting、奇数尾部和 ragged shape 错误；llm 使用更接近 hidden=4096、vocab=32000、T=512 等 workload 的尺寸做性能测试。该函数只返回预先给定的维度 tuple，不分配 Tensor；未知 profile 立即报错，避免静默测错规模。
    """

    if profile == "smoke":
        return smoke
    if profile == "llm":
        return llm
    raise ValueError(f"unknown profile: {profile}")


def _randn(shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> Tensor:
    """在目标 device/dtype 上直接生成标准正态输入。

    数学分布：每个元素独立采样 X_i~N(0,1)。直接在目标 device 创建可避免把 Host→Device copy 混入 benchmark；统一 dtype 让 reference/teaching 使用完全相同的精度。随机值同时含正负数，适合激活、归一化和一般数值正确性测试。
    """

    return torch.randn(shape, device=device, dtype=dtype)


def build_cases() -> tuple[OperatorCase, ...]:
    """构造所有算子实验的不可变 tuple。

    返回标注 tuple[OperatorCase, ...] 中的 ... 表示“任意数量的 OperatorCase”，不是省略了待填写代码；若写 tuple[OperatorCase, OperatorCase] 才表示固定两个元素。每个内部 input factory 统一接收 device/dtype/profile 并返回参数 tuple，使 reference 与 teaching 能用相同的 function(*inputs) 调用和相同底层 Tensor。
    """

    def unary_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为单输入逐元素/激活算子创建 x。

        smoke=[128,257] 用 257 这个非 2 次幂宽度覆盖尾部；llm=[2048,4096] 可理解为 2048 个 token/row、hidden=4096。返回 `(x,)` 的尾逗号表示单元素 tuple，保证后续 *inputs 正确解包，而不是直接返回 Tensor。
        """

        return (_randn(_shape(profile, (128, 257), (2048, 4096)), device, dtype),)

    def binary_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为 A+B、MSE 等二输入逐元素算子创建两个同 shape 独立张量。

        相同 shape 隔离核心逐元素语义，不把 broadcasting 额外变量混进基准；smoke 的 257 检查 ragged tail，llm 的 [2048,4096] 模拟 token×hidden 激活。
        """

        shape = _shape(profile, (128, 257), (2048, 4096))
        return _randn(shape, device, dtype), _randn(shape, device, dtype)

    def saxpy_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """在 binary_inputs 后追加固定标量 alpha=0.25。

        `(*binary_inputs(...),0.25)` 中星号会把 `(x,y)` 的两个元素展开到新 tuple，再追加 alpha，最终得到 `(x,y,0.25)`；固定 alpha 保证 reference/teaching 语义一致且结果可复现。
        """

        return (*binary_inputs(device, dtype, profile), 0.25)

    def bias_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为 bias+activation 创建矩阵 x 与一维 bias。

        x shape=[rows,columns] 表示多行 token/样本特征，bias shape=[columns] 表示每个输出 channel 一份参数；PyTorch 会把同一 bias 广播到所有 rows。必须分别创建二维 x 和一维 bias，而不是再创建 [rows,columns] bias，否则既浪费内存，也不再模拟 Linear bias 的参数共享语义。257 检查非整齐宽度，4096 模拟 LLM hidden。
        """

        rows, columns = _shape(profile, (128, 257), (2048, 4096))
        return _randn((rows, columns), device, dtype), _randn((columns,), device, dtype)

    def reduction_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为 sum/mean/max/softmax 等最后一维归约创建输入。

        smoke=[64,257] 让每行 K=257 为奇数，强制 tree reduction 走补单位元的尾部逻辑；llm=[2048,4096] 模拟大量 token 沿 hidden/vocab 子维归约。64/2048 是独立 reduction rows，可并行处理。
        """

        return (_randn(_shape(profile, (64, 257), (2048, 4096)), device, dtype),)

    def norm_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为 LayerNorm 创建 x、gamma、beta 与 eps。

        x=[rows,columns] 表示每行独立沿 hidden 归一化；weight=gamma 和 bias=beta 都是 [columns]，在 rows 上共享；eps=1e-5 防止方差为零时除零。smoke 的 257 测非整齐 hidden，llm 的 4096 是常见大模型 hidden。
        """

        rows, columns = _shape(profile, (64, 257), (2048, 4096))
        return (
            _randn((rows, columns), device, dtype),
            _randn((columns,), device, dtype),
            _randn((columns,), device, dtype),
            1e-5,
        )

    def rms_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为 RMSNorm 创建 x、gamma 与 eps。

        shape 理由与 LayerNorm 相同，但 RMSNorm 公式通常没有 beta，因此只生成一个 [columns] weight；这也验证教学函数没有误加中心化/bias 语义。
        """

        rows, columns = _shape(profile, (64, 257), (2048, 4096))
        return _randn((rows, columns), device, dtype), _randn((columns,), device, dtype), 1e-5

    def small_matmul_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为显式 broadcast GEMM 语义版创建 A[M,K] 与 B[K,N]。

        smoke=(31,29,27) 全是非整齐维度，用来检查 M/N/K 次序和尾部；llm 仍只用 64³，因为教学函数会物化 M×K×N 临时张量，若直接给 4096³ 会耗费不可接受的显存。真实大 GEMM 由 linear cases 测试，不用这个故意低效实现。
        """

        m, n, k = _shape(profile, (31, 29, 27), (64, 64, 64))
        return _randn((m, k), device, dtype), _randn((k, n), device, dtype)

    def linear_decode_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为 decode small-M Linear 创建 x[M,K]、weight[N,K]、bias[N]。

        smoke=(M=3,N=65,K=63) 快速覆盖 ragged 维度；llm=(1,4096,4096) 表示 batch=1 decode 每次只新增一个 token，M=1 而输入/输出 hidden 都很大。weight 按 PyTorch F.linear 约定使用 [out=N,in=K]。
        """

        m, n, k = _shape(profile, (3, 65, 63), (1, 4096, 4096))
        return (
            _randn((m, k), device, dtype),
            _randn((n, k), device, dtype),
            _randn((n,), device, dtype),
        )

    def linear_prefill_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为 prefill large-M Linear 创建与 decode 相同语义、不同 M 的输入。

        smoke M=32 保持快速；llm M=512 表示一次处理 512 个 prompt token，N=K=4096。与 decode 共用 weight/bias layout，使基准能隔离 M 变化对库算法、并行度和吞吐的影响。
        """

        m, n, k = _shape(profile, (32, 65, 63), (512, 4096, 4096))
        return (
            _randn((m, k), device, dtype),
            _randn((n, k), device, dtype),
            _randn((n,), device, dtype),
        )

    def bmm_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为 batched matmul 创建 A[B,M,K] 与 B[B,K,N]。

        smoke=(2,17,19,13) 用两个 batch 和 ragged M/N/K 验证 batch 不会互相混合；llm=(8,64,64,64) 模拟多 head/多组规则小矩阵，同时限制 broadcast 教学临时量。
        """

        batch, m, n, k = _shape(profile, (2, 17, 19, 13), (8, 64, 64, 64))
        return _randn((batch, m, k), device, dtype), _randn((batch, k, n), device, dtype)

    def embedding_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为 Embedding 创建合法 token ids 与词表权重。

        weight=[vocab,hidden]，indices=[tokens] 且每个整数均在 [0,vocab)；smoke 采用 127×31 与 23 tokens 检查非整齐维度，llm 用 vocab=4096、hidden=256、tokens=512。教学 one-hot 临时量随 tokens×vocab 增长，所以没有直接使用 32000 词表。
        """

        vocab, hidden, tokens = _shape(profile, (127, 31, 23), (4096, 256, 512))
        indices = torch.randint(vocab, (tokens,), device=device)
        return indices, _randn((vocab, hidden), device, dtype)

    def gather_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为每行 gather 创建源矩阵与合法索引。

        x=[rows,columns]，indices=[rows,selected]，每个 index 从 [0,columns) 随机采样，因此输出是 [rows,selected]。selected 远小于 columns，模拟从大表/特征行中选少数位置；llm 512×4096 选 16 个值。
        """

        rows, columns, selected = _shape(profile, (11, 29, 7), (512, 4096, 16))
        x = _randn((rows, columns), device, dtype)
        indices = torch.randint(columns, (rows, selected), device=device)
        return x, indices

    def scatter_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为 scatter-add 创建 values、目的索引和输出宽度。

        src/indices 都是 [rows,values]，indices 落在 [0,output_size)；output_size 大于单行 values 时既有空目的，也可能随机出现重复目的，从而测试“重复 index 必须求和”。llm 用 512 行、每行 64 个值散到 256 个槽位，近似稀疏/MoE 回填模式。
        """

        rows, values, output_size = _shape(profile, (11, 17, 23), (512, 64, 256))
        src = _randn((rows, values), device, dtype)
        indices = torch.randint(output_size, (rows, values), device=device)
        return src, indices, output_size

    def rope_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为 RoPE 创建 x 与可广播的 cos/sin 表。

        x=[batch,heads,tokens,head_dim]，head_dim 取偶数以形成 D/2 个 pair；positions=[1,1,T,1]，frequencies=[1,1,1,D/2]，广播后 angles/cos/sin=[1,1,T,D/2]，在 batch/head 间共享。smoke 测小多头与奇数 token 数，llm 使用 B=1、H=32、T=512、D=128。
        """

        batch, heads, tokens, head_dim = _shape(profile, (2, 4, 17, 32), (1, 32, 512, 128))
        x = _randn((batch, heads, tokens, head_dim), device, dtype)
        positions = torch.arange(tokens, device=device, dtype=torch.float32).reshape(1, 1, tokens, 1)
        frequencies = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
        angles = positions / (10000.0**frequencies.reshape(1, 1, 1, -1))
        return x, torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)

    def attention_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为方形 causal self-attention 创建独立 Q/K/V。

        三者 shape 都是 [B,H,T,D]，因为每个 query 与同一序列全部 key 做 T×T 打分，再用相同长度 V 聚合。独立随机值比令 Q=K=V 更能暴露转置或参数混用错误；smoke 为 2×4×32×32，llm prefill 为 1×8×512×64。
        """

        batch, heads, tokens, head_dim = _shape(profile, (2, 4, 32, 32), (1, 8, 512, 64))
        shape = (batch, heads, tokens, head_dim)
        return _randn(shape, device, dtype), _randn(shape, device, dtype), _randn(shape, device, dtype)

    def classification_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为 CrossEntropy 创建 logits[R,C] 与每行一个 target class。

        targets=[rows] 的整数均在 [0,classes)，对应 logits 每行的正确类别；smoke 37×101 覆盖非整齐 class 数，llm 2048×32000 模拟大量 token 对大词表的 next-token loss。
        """

        rows, classes = _shape(profile, (37, 101), (2048, 32000))
        return _randn((rows, classes), device, dtype), torch.randint(classes, (rows,), device=device)

    def topk_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为 Top-K 创建 batch×vocab scores 与满足 0<k≤vocab 的 k。

        smoke 用 7×127、k=5 快速检查 values/indices；llm 用 batch=32、vocab=32000、k=50 模拟并行 token sampling。随机连续 logits 几乎不会精确 tie，使 reference 与 full-sort 教学版索引可稳定比较。
        """

        batch, vocab, k = _shape(profile, (7, 127, 5), (32, 32000, 50))
        return _randn((batch, vocab), device, dtype), k

    def batch_norm_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为 NCHW BatchNorm eval 创建激活和每 channel 参数/统计量。

        x=[N,C,H,W]；weight、bias、running_mean、running_var 都是 [C] 并沿 N/H/W 广播。running_var 用 rand+0.5 保证严格为正，避免随机负方差使 sqrt 产生 NaN；smoke 使用非方形 9×7 检查空间广播，llm 使用常见 56×56 feature map。
        """

        batch, channels, height, width = _shape(profile, (2, 8, 9, 7), (32, 64, 56, 56))
        return (
            _randn((batch, channels, height, width), device, dtype),
            _randn((channels,), device, dtype),
            _randn((channels,), device, dtype),
            _randn((channels,), device, dtype),
            torch.rand((channels,), device=device, dtype=dtype) + 0.5,
            1e-5,
        )

    def conv_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为 3×3 Conv2d 创建 NCHW 输入、OIHW 权重、bias、stride=1、padding=1。

        padding=1 配合 kernel=3/stride=1 使输出 H/W 与输入相同，便于 reference 对齐；smoke 用 RGB-like Cin=3、Cout=5、16×16，llm 用 N=16、Cin=32、Cout=64、56×56 模拟视觉中间层。weight [Cout,Cin,3,3] 遵循 PyTorch layout。
        """

        batch, in_channels, out_channels, size = _shape(profile, (2, 3, 5, 16), (16, 32, 64, 56))
        return (
            _randn((batch, in_channels, size, size), device, dtype),
            _randn((out_channels, in_channels, 3, 3), device, dtype),
            _randn((out_channels,), device, dtype),
            1,
            1,
        )

    def pool_inputs(device: torch.device, dtype: torch.dtype, profile: str) -> tuple[object, ...]:
        """为无 padding 的 2×2、stride=2 MaxPool 创建 NCHW 输入。

        H/W 都能被 2 整除，所以输出恰为一半，先隔离核心窗口 max 语义；smoke 2×4×16×16，llm 16×64×56×56。奇数边界、padding、ceil_mode 应在扩展测试中单独加入，不能混进第一条性能基线。
        """

        shape = _shape(profile, (2, 4, 16, 16), (16, 64, 56, 56))
        return _randn(shape, device, dtype), 2, 2

    return (
        OperatorCase("vector_add", "elementwise", torch.add, vector_add, binary_inputs, "C=A+B"),
        OperatorCase("saxpy", "elementwise", lambda x, y, alpha: torch.add(y, x, alpha=alpha), saxpy, saxpy_inputs, "alpha*x+y"),
        OperatorCase("relu", "activation", F.relu, relu_from_where, unary_inputs, "ReLU from where"),
        OperatorCase("sigmoid", "activation", torch.sigmoid, sigmoid_from_exp, unary_inputs, "sigmoid from exp"),
        OperatorCase("silu", "activation", F.silu, silu_from_primitives, unary_inputs, "SiLU=x*sigmoid(x)"),
        OperatorCase("fused_bias_silu", "fusion", lambda x, bias: F.silu(x + bias), fused_bias_silu_from_primitives, bias_inputs, "bias add plus SiLU"),
        OperatorCase("gelu_exact", "activation", lambda x: F.gelu(x, approximate="none"), gelu_exact_from_erf, unary_inputs, "exact GELU from erf"),
        OperatorCase("gelu_tanh", "activation", lambda x: F.gelu(x, approximate="tanh"), gelu_tanh_from_primitives, unary_inputs, "approximate GELU"),
        OperatorCase("reduce_sum", "reduction", lambda x: torch.sum(x, dim=-1), tree_sum_last_dim, reduction_inputs, "pairwise tree sum", atol=2e-4, rtol=2e-4),
        OperatorCase("reduce_mean", "reduction", lambda x: torch.mean(x, dim=-1), tree_mean_last_dim, reduction_inputs, "pairwise tree mean", atol=2e-5, rtol=2e-4),
        OperatorCase("reduce_max", "reduction", lambda x: torch.amax(x, dim=-1), tree_max_last_dim, reduction_inputs, "pairwise tree max"),
        OperatorCase("softmax", "normalization", lambda x: F.softmax(x, dim=-1), stable_softmax, reduction_inputs, "numerically stable softmax", atol=2e-5, rtol=2e-4),
        OperatorCase("log_softmax", "normalization", lambda x: F.log_softmax(x, dim=-1), stable_log_softmax, reduction_inputs, "numerically stable log-softmax", atol=2e-5, rtol=2e-4),
        OperatorCase("layer_norm", "normalization", lambda x, w, b, eps: F.layer_norm(x, (x.shape[-1],), w, b, eps), layer_norm_from_formula, norm_inputs, "LayerNorm formula", atol=2e-5, rtol=2e-4),
        OperatorCase("rms_norm", "normalization", lambda x, w, eps: F.rms_norm(x, (x.shape[-1],), w, eps), rms_norm_from_formula, rms_inputs, "RMSNorm formula", atol=2e-5, rtol=2e-4),
        OperatorCase("matmul_semantics", "matrix", torch.matmul, matmul_from_broadcast, small_matmul_inputs, "broadcast multiply plus K reduction", atol=3e-4, rtol=3e-4),
        OperatorCase("linear_decode", "matrix", F.linear, linear_from_matmul, linear_decode_inputs, "small-M Linear", atol=3e-4, rtol=3e-4),
        OperatorCase("linear_prefill", "matrix", F.linear, linear_from_matmul, linear_prefill_inputs, "prefill Linear", atol=3e-4, rtol=3e-4),
        OperatorCase("batched_matmul", "matrix", torch.bmm, batched_matmul_from_broadcast, bmm_inputs, "batched broadcast multiply plus reduction", atol=3e-4, rtol=3e-4),
        OperatorCase("embedding", "indexing", F.embedding, embedding_from_one_hot, embedding_inputs, "one-hot embedding semantics"),
        OperatorCase("gather", "indexing", lambda x, indices: torch.gather(x, -1, indices), gather_from_one_hot, gather_inputs, "one-hot gather semantics"),
        OperatorCase("scatter_add", "indexing", lambda src, indices, size: torch.zeros((*src.shape[:-1], size), dtype=src.dtype, device=src.device).scatter_add(-1, indices, src), scatter_add_from_one_hot, scatter_inputs, "one-hot scatter-add semantics", atol=3e-5, rtol=3e-4),
        OperatorCase("rope", "position", apply_rope_from_pairs, apply_rope_from_pairs, rope_inputs, "pairwise rotary embedding"),
        OperatorCase("causal_attention", "attention", lambda q, k, v: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True), causal_attention_from_primitives, attention_inputs, "explicit QK-softmax-PV", atol=5e-4, rtol=5e-4),
        OperatorCase("cross_entropy", "loss", F.cross_entropy, cross_entropy_from_logsumexp, classification_inputs, "cross entropy from logsumexp", atol=2e-5, rtol=2e-4),
        OperatorCase("mse_loss", "loss", F.mse_loss, mse_from_formula, binary_inputs, "mean squared error"),
        OperatorCase("topk", "sampling", torch.topk, topk_from_sort, topk_inputs, "top-k from full sort"),
        OperatorCase("batch_norm_eval", "normalization", lambda x, w, b, mean, var, eps: F.batch_norm(x, mean, var, w, b, training=False, eps=eps), batch_norm_eval_from_formula, batch_norm_inputs, "BatchNorm inference formula", atol=2e-5, rtol=2e-4),
        OperatorCase("conv2d", "convolution", lambda x, w, b, stride, padding: F.conv2d(x, w, b, stride=stride, padding=padding), conv2d_from_unfold, conv_inputs, "im2col/unfold plus matrix multiplication", atol=5e-4, rtol=5e-4),
        OperatorCase("max_pool2d", "pooling", lambda x, kernel, stride: F.max_pool2d(x, kernel, stride), max_pool2d_from_unfold, pool_inputs, "unfold plus max reduction"),
    )


CASES = build_cases()
CASE_BY_NAME = {case.name: case for case in CASES}


def select_cases(names: str) -> tuple[OperatorCase, ...]:
    """把 CLI 的名称字符串解析成有序 OperatorCase tuple。

    names='all' 返回完整注册表；否则按逗号拆分并保留用户指定顺序，例如 'relu,softmax'。集合差检查未知名称并尽早报错，避免 benchmark 因拼写错误静默漏测；返回 tuple 防止调用方意外修改全局 CASES。
    """

    if names == "all":
        return CASES
    requested = tuple(part.strip() for part in names.split(",") if part.strip())
    unknown = sorted(set(requested) - CASE_BY_NAME.keys())
    if unknown:
        raise ValueError(f"unknown operators: {', '.join(unknown)}")
    return tuple(CASE_BY_NAME[name] for name in requested)
