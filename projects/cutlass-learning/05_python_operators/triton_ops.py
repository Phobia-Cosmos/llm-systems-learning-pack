from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # Optional dependency: CPU-only teaching remains usable.
    # 没有 GPU 不等于一定无法 import Triton：只要包已安装，Python import 可能成功；但当前 kernel 的 JIT/launch
    # 需要 Triton 支持的 accelerator 和对应 driver。CPU-only 环境仍能导入本模块，是因为 ImportError 时把
    # triton/tl 记为 None，eligibility 会拒绝 Triton 路径并让 dispatcher 使用 PyTorch CPU fallback。
    triton = None
    tl = None


@dataclass(frozen=True)
class Eligibility:
    """记录优化 kernel 是否支持当前输入，以及不支持时可读的原因。"""

    supported: bool
    reason: str


# 这个 kernel 把原本的 bias add、sigmoid 和 multiply 放进同一个 Triton program：每个 program 从 x/bias
# 各读取一次，在寄存器中算 z=x+bias 与 y=z*sigmoid(z)，最后只写一次 output。之所以能融合，是因为这些操作
# 都是逐元素/按列广播，某个 y 元素不依赖其他 y 元素，不需要跨 program 全局同步。相对 eager 的多个 kernel，
# 它减少了 launch 次数，也避免把中间 z 和 sigmoid(z) 写回再读出 global memory。
if triton is not None:

    @triton.jit
    def _fused_bias_silu_kernel(
        x_ptr,
        bias_ptr,
        output_ptr,
        element_count,
        columns: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # tl.program_id(0) 是当前 Triton program 在 grid 第 0 维的标量编号，类似 CUDA 的 blockIdx.x，
        # 不是数组；tl.arange(0, BLOCK_SIZE) 才生成一个包含 BLOCK_SIZE 个 lane offset 的编译期向量。
        # 二者组合后 offsets 是本 program 负责的一段全局元素索引向量。
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < element_count
        column_offsets = offsets % columns
        # tl.load(..., mask=mask, other=0.0) 只对 mask=True 的 lane 访问内存；越过 element_count 的尾部 lane
        # 不发起非法 load，并在寄存器中得到 other=0.0。other 只是 masked-off lane 的替代值，不会写回输入。
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + column_offsets, mask=mask, other=0.0)
        z = x + bias
        # 当前使用的 Triton 3.6 tl.sigmoid/math.exp 路径要求 FP32/FP64 输入；显式转 FP32 也让 exp/sigmoid
        # 的范围与精度比直接 FP16 更稳。它不是“所有 Triton 算子永远必须转 FP32”的规则；若目标 primitive
        # 原生支持低精度且误差契约允许，可以保留低精度。tl.store 会按 output_ptr dtype 转回 FP16/BF16/FP32。
        z_fp32 = z.to(tl.float32)
        output = z_fp32 * tl.sigmoid(z_fp32)
        tl.store(output_ptr + offsets, output, mask=mask)


def fused_bias_silu_eligibility(x: torch.Tensor, bias: torch.Tensor) -> Eligibility:
    """检查当前 forward-only Triton kernel 的完整输入契约。"""

    if triton is None:
        return Eligibility(False, "Triton is not installed")
    # torch.Tensor 是跨 backend 抽象，可以位于 CPU、CUDA、MPS、XLA、meta 等 device；只有用户显式创建在
    # CUDA 上或调用 tensor.to("cuda") 后 is_cuda 才为 True。本 kernel 写的是 CUDA/Triton device 路径。
    if not x.is_cuda or not bias.is_cuda:
        return Eligibility(False, "Triton implementation requires CUDA tensors")
    # 多 GPU 机器上可能出现 x 在 cuda:0、bias 在 cuda:1，或调用方只迁移了部分模型参数。一个 kernel
    # launch 绑定一个当前 device，不能直接解引用另一张 GPU 的普通指针，所以两者必须在同一 CUDA device。
    if x.device != bias.device:
        return Eligibility(False, "x and bias must be on the same CUDA device")
    if x.dtype != bias.dtype:
        return Eligibility(False, "x and bias must have the same dtype")
    # 不应承诺支持“torch 所有 dtype”：bool、整数、复数、量化、float8 等拥有不同数学语义，SiLU 本身也
    # 只对浮点输入合理。生产库通常先声明每个 op/backend 的支持矩阵，再用 dispatcher、C++ type-dispatch
    # 宏、模板或 JIT specialization 复用实现并实例化有限 dtype；不支持的组合走其他 kernel/fallback 或报错。
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return Eligibility(False, "supported dtypes are float16, bfloat16, and float32")

    # x.shape[-1] 是 x 最后一维的整数长度，即 columns；bias.numel() 返回 bias 的总元素个数，也是 Python int。
    # 数学语义是 y[...,c]=SiLU(x[...,c]+bias[c])，所以一维 bias 必须恰有 columns 个值才能按列广播。
    if x.ndim < 1 or bias.ndim != 1 or x.shape[-1] != bias.numel():
        return Eligibility(False, "bias must be one-dimensional and match x.shape[-1]")
    # fallback 是覆盖更多输入的已知正确替代路径。本 kernel 不 launch 0 个 program，空 Tensor 直接交给
    # F.silu(x+bias)，由 PyTorch 正确产生同 shape 的空输出；fallback 不是失败，而是 dispatcher 的正常分支。
    if x.numel() == 0:
        return Eligibility(False, "empty tensors use the PyTorch fallback")

    # x.stride(-1) 返回“最后一维坐标加 1 时，storage offset 增加多少个元素”的单个整数，不是数组；
    # x.stride() 才返回所有维度的 tuple，也可用 stride(dim) 查询任意合法维度。此 kernel 用扁平连续地址
    # x_ptr+offsets，因此要求整体 contiguous 且最后一维 stride=1；transpose view 等需 fallback 或另写 stride kernel。
    if x.stride(-1) != 1 or not bias.is_contiguous() or not x.is_contiguous():
        return Eligibility(False, "teaching kernel requires contiguous last-dimension storage")
    if x.data_ptr() % 16 or bias.data_ptr() % 16:
        return Eligibility(False, "teaching kernel requires 16-byte-aligned base addresses")
    # 当前 Triton 调用没有向 PyTorch autograd 注册 backward。若要支持训练，需要实现
    # dSiLU(z)/dz = sigmoid(z) * [1 + z*(1-sigmoid(z))]，令 grad_z=grad_out*dSiLU/dz；
    # grad_x=grad_z，grad_bias=沿 x 的所有前导/row 维对 grad_z 求和。随后用 torch.autograd.Function 或
    # torch.library 注册 forward/backward，并决定保存 z 还是 backward 时重算；在此之前 requires_grad 输入必须 fallback。
    if x.requires_grad or bias.requires_grad:
        return Eligibility(False, "this forward-only teaching kernel has no registered backward")
    return Eligibility(True, "supported")


def triton_fused_bias_silu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """严格调用 Triton forward；不满足契约时抛错而不静默 fallback。"""

    eligibility = fused_bias_silu_eligibility(x, bias)
    if not eligibility.supported:
        raise ValueError(eligibility.reason)
    if triton is None:
        raise RuntimeError("unreachable: eligibility accepted without Triton")
    output = torch.empty_like(x)
    block_size = 256
    # triton.cdiv(element_count,256) 是向上整除，得到覆盖全部元素所需的 program 数；grid 是一维 tuple，
    # 例如 1000 个元素得到 (4,)。最后一个 program 的越界 lane 由 mask 屏蔽。
    grid = (triton.cdiv(x.numel(), block_size),)
    with torch.cuda.device(x.device):
        # Triton 与 CUDA 一样不能用一个 program instance 串行处理任意大 Tensor；grid 定义启动多少个并行
        # program instance，每个负责 BLOCK_SIZE 个元素。kernel[grid](...) 是 Triton 的 launch 语法，
        # with torch.cuda.device(...) 确保 JIT/launch 使用 x 所在 GPU。
        _fused_bias_silu_kernel[grid](
            x,
            bias,
            output,
            x.numel(),
            columns=x.shape[-1],
            BLOCK_SIZE=block_size,
        )
    return output

def fused_bias_silu_dispatch(
    x: torch.Tensor,
    bias: torch.Tensor,
    *,
    allow_fallback: bool = True,
) -> torch.Tensor:
    """面向调用方的安全入口，在 Triton 优化路径与 PyTorch 通用路径之间分发。

    不是每个内部 Triton kernel 都必须各写一份 dispatcher：若上层已经严格保证契约，可直接调用；多个 kernel
    也可共享一个 registry/dispatcher。但公开 custom op 通常需要某种分发层来处理 device/dtype/layout/shape、
    autograd 和 fallback，否则一次非连续输入或 CPU 调用就会在低层难以理解地失败。
    """

    eligibility = fused_bias_silu_eligibility(x, bias)
    # Triton 实现是受限的 forward-only CUDA 单 kernel，目标是减少 launch 与中间 GMEM 流量；fallback
    # F.silu(x+bias) 覆盖 CPU、空/非连续/未对齐和 autograd 等更多语义，通常由 PyTorch 分别执行或再被
    # torch.compile 融合。两者数学输出应一致，但适用范围、kernel 数、性能和反向传播能力不同。
    if eligibility.supported:
        return triton_fused_bias_silu(x, bias)
    if allow_fallback:
        return F.silu(x + bias)
    raise ValueError(eligibility.reason)
