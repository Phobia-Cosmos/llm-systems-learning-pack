import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

# TODO:这个使用来做什么的？不要使用反向传播？
# 关闭 autograd 的默认梯度记录，适合这里只做前向性能基准的脚本；它不会关闭 CUDA，也不会改变 kernel 的数值计算。
torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="elementwise_lib",
    sources=["elementwise.cu"],
    # TODO:各个flags分别代表什么意思？
    # -O3 开启高等级优化；-U... 取消 PyTorch/CUDA 对 half、half2、bfloat16 运算的禁用宏；--expt-* 开启扩展 constexpr/lambda；--use_fast_math 用更快但可能略低精度的数学指令；C++ 侧使用 C++17。
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)


# TODO:为什么要有iters，难道不是计算就好了么 还需要迭代吗？perf_func是什么？callable？
# warmup 后重复 iters 次是为了摊平一次 kernel launch、计时器和缓存抖动；perf_func 是可调用对象，callable 是这里的类型标注。
def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    # TODO：这个tag传入的是什么？
    # tag 是调用方传入的可读标签，例如 f32、f16x8 或 f32_th，用于组成输出名称。
    tag: str,
    # TODO：这个Optional是什么意思？
    # Optional[torch.Tensor] 表示 out 可以是 Tensor，也可以是 None；默认 None 让函数创建并返回新输出，传入 Tensor 则复用调用方缓冲区。
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    # torch.dot vs custom dot_prod kernel
    # TODO:这个是什么意思？out何时会变成None？为什么要fill_(0)?
    # out=None 表示函数返回新 tensor；传入 out 表示原地写入预分配缓冲区。fill_(0) 清掉上一次基准结果，避免残留值影响未覆盖位置。
    if out is not None:
        out.fill_(0)
    # warmup
    # TODO:为什么一个要传入out一个不传入？请你给我一个具体例子 为什么会有这两种不同的情况？
    # 例如 torch.add(a, b) 返回新 tensor，而自定义 elementwise_add(a, b, c) 直接写 c；前者方便组合，后者减少重复分配。
    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)

    # TODO:torch和cuda的关系是什么？为什么这里使用的是CPU时间而不是GPU时间？
    # PyTorch 是张量运行时，CUDA 是 GPU 执行后端；time.time() 测 CPU 墙钟时间，靠前后的 synchronize 等待 GPU 完成，因此会包含同步开销。
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b)
    torch.cuda.synchronize()
    end = time.time()

    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"

    # TODO：这一些列操作分别在做什么？各个函数的作用分别是什么？round的作用是什么？
    # flatten 把输出展平成一维，detach 切断 autograd，cpu 把 GPU 数据复制到 CPU，numpy 转 NumPy，tolist 转 Python list，切片取前两个值，round(v, 8) 保留 8 位小数以便打印比较。
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]

    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]

for S, K in SKs:
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")
    # TODO:这里在做什么？为什么c是0？randn是生成S*K的tensor张量是吗？为什么一定要contiguous？如果不连续会有什么情况 这里也需要对比实现吧？
    # 这里生成 [S,K] 输入并用零初始化输出 c；contiguous 保证按行连续，满足 CUDA 指针按连续地址访问的假设。不连续输入可能被错误解释或需要额外复制。
    a = torch.randn((S, K)).cuda().float().contiguous()
    b = torch.randn((S, K)).cuda().float().contiguous()
    c = torch.zeros_like(a).cuda().float().contiguous()

    run_benchmark(lib.elementwise_add_f32, a, b, "f32", c)
    run_benchmark(lib.elementwise_add_f32x4, a, b, "f32x4", c)
    # TODO:partial是什么意思？为什么最后都要运行一个partial？torch.add是naive实现吗？
    # partial(torch.add, out=c) 预先固定 out 参数，得到只接收 a、b 的 callable。torch.add 是 PyTorch 已优化的原生实现，不等于简单 naive CUDA。
    run_benchmark(partial(torch.add, out=c), a, b, "f32_th")

    print("-" * 85)
    # TODO：half就是float16是吗？
    # 是；PyTorch 的 half 通常就是 IEEE float16，每个元素 16 bit = 2 byte。
    a_f16 = a.half().contiguous()
    b_f16 = b.half().contiguous()
    c_f16 = c.half().contiguous()
    run_benchmark(lib.elementwise_add_f16, a_f16, b_f16, "f16", c_f16)
    run_benchmark(lib.elementwise_add_f16x2, a_f16, b_f16, "f16x2", c_f16)
    run_benchmark(lib.elementwise_add_f16x8, a_f16, b_f16, "f16x8", c_f16)
    run_benchmark(
        lib.elementwise_add_f16x8_pack, a_f16, b_f16, "f16x8pack", c_f16
    )
    run_benchmark(partial(torch.add, out=c_f16), a_f16, b_f16, "f16_th")
    print("-" * 85)
