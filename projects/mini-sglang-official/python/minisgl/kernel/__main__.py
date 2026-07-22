assert __name__ == "__main__"


def generate_clangd():
    import os
    import subprocess

    from minisgl.kernel.utils import DEFAULT_INCLUDE
    from minisgl.utils import init_logger
    from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path

    logger = init_logger(__name__)
    logger.info("Generating .clangd file...")
    include_paths = [find_include_path(), find_dlpack_include_path()] + DEFAULT_INCLUDE
    # TODO：capture_output和check的作用是什么？
    # 解答：capture_output=True 把子进程的 stdout/stderr 收进 CompletedProcess；check=True 让非零退出码直接抛出 CalledProcessError，避免继续使用无效结果。
    status = subprocess.run(
        args=["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        check=True,
    )
    compute_cap = status.stdout.decode("utf-8").strip().split("\n")[0]
    major, minor = compute_cap.split(".")
    compile_flags = ",\n    ".join(
        [
            "-xcuda",
            f"--cuda-gpu-arch=sm_{major}{minor}",
            "-std=c++20",
            "-Wall",
            "-Wextra",
        ]
        + [f"-isystem{path}" for path in include_paths]
    )
    # TODO：这个是什么？
    # 解答：这是要写入 .clangd 的 YAML 内容，告诉 clangd 按 CUDA、当前 GPU 架构、C++20 和这些头文件目录解析源码。
    clangd_content = f"""
CompileFlags:
  Add: [
    {compile_flags}
  ]
"""
    if os.path.exists(".clangd"):
        logger.warning(".clangd file already exists, nothing done.")
        logger.warning(f"suggested content: {clangd_content}")
    else:
        # TODO：为什么要写这一个文件？
        # 解答：.clangd 只服务于编辑器的补全、跳转和静态诊断，不参与 Mini-SGLang 的编译或运行；写文件是为了让 clangd 找到 CUDA/TVM-FFI 头文件。
        with open(".clangd", "w") as f:
            f.write(clangd_content)
        logger.info(".clangd file generated.")


generate_clangd()
