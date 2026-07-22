from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, List, NamedTuple, Tuple, TypeAlias, Union

if TYPE_CHECKING:
    from tvm_ffi import Module

# TODO：这里在我们的情境中指向的是那一个文件？
# 解答：它是当前 utils.py 同级的 csrc 目录，即 python/minisgl/kernel/csrc；后续再分别拼接 include、src 或 jit 下的文件。
KERNEL_PATH = pathlib.Path(__file__).parent / "csrc"
DEFAULT_INCLUDE = [str(KERNEL_PATH / "include")]
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_CUDA_CFLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]
DEFAULT_LDFLAGS = []

# TODO：这个是什么东西？类型是什么？
# 解答：这是类型别名，表示 C++ 模板实参在 Python 侧只接受 int、float 或 bool；TypeAlias 本身不创建新运行时类型。
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, bool]


class CppArgList(list[str]):
    # TODO：这个返回的是什么？把这一系列list[str]拼起来吗？
    # 解答：是；对象仍是 list[str]，只有调用 str(obj) 时才把元素用 ", " 连接，便于嵌入 C++ 模板参数文本。
    def __str__(self) -> str:
        return ", ".join(self)


class KernelConfig(NamedTuple):
    num_threads: int
    # TODO：这个属性代表什么？use_pdl是什么？
    # 解答：字段虽名为 max_occupancy，实际会成为 __launch_bounds__ 的第二参数 minBlocksPerMultiprocessor，用来约束寄存器分配以争取每个 SM 至少驻留这些 block；use_pdl 控制 CUDA Programmatic Dependent Launch。
    max_occupancy: int
    use_pdl: bool

    @property
    def template_args(self) -> str:
        pdl = "true" if self.use_pdl else "false"
        return f"{self.num_threads},{self.max_occupancy},{pdl}"


def _make_name(*args: str) -> str:
    return "minisgl__" + "_".join(str(arg) for arg in args)


def _make_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"

# TODO：这一般返回的是什么 举一个例子给我
# 解答：它把 Python 模板值转成 C++ 字面量列表；例如 (2048, 2, True) 得到 ["2048", "2", "true"]，转成字符串则是 "2048, 2, true"。
def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CppArgList:
    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, float)):
            return str(arg)
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CppArgList(_convert(arg) for arg in args)


# TODO：这个函数的作用是什么？
# 解答：它解析 csrc/src 下的源文件，附加统一编译/链接参数，调用 tvm_ffi.cpp.load 编译并加载共享库，返回可从 Python 调用其导出符号的 Module。
def load_aot(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    from tvm_ffi.cpp import load

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    cpp_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cpp_files]
    cuda_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cuda_files]

    return load(
        _make_name(*args),
        cpp_files=cpp_files,
        cuda_files=cuda_files,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,
    )


def load_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,

    build_directory: str | None = None,
) -> Module:
    from tvm_ffi.cpp import load_inline

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    # TODO：这两个包装类作用是什么？
    # 解答：它们不是包装类，而是 (Python 导出名, C++ 可调用符号) 元组；代码生成器据此追加 TVM-FFI 导出宏，让编译后的函数以 module.<导出名> 调用。
    cpp_wrappers = cpp_wrappers or []
    cuda_wrappers = cuda_wrappers or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    # include cpp files
    cpp_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cpp_files]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

    # include cuda files
    cuda_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]

    return load_inline(
        _make_name(*args),
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,
    )
