from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

_LOG_LEVEL = None

# TODO：为什么一个函数内部可以存在class？
# 解答：Python 的 class 是可执行语句，因此能在函数中定义；这使 ColorFormatter 只在初始化时可见，并能闭包访问 suffix、use_tp_rank 等局部状态。
def init_logger(
    name: str,
    suffix: str = "",
    *,
    strip_file: bool = True,
    level: str | None = None,
    use_pid: bool | None = None,
    use_tp_rank: bool | None = None,
):
    """Initialize the logger for the module with colors and pretty formatting."""
    import logging
    import os
    import sys

    global _LOG_LEVEL
    if _LOG_LEVEL is None:
        LEVEL_MAP = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        level = level or os.getenv("LOG_LEVEL", "").upper()
        # TODO：第二个代表默认INFO？
        # 解答：是的，dict.get(key, default) 在 level 不是有效键时返回 logging.INFO。
        _LOG_LEVEL = LEVEL_MAP.get(level, logging.INFO)

    if strip_file:
        suffix = os.path.basename(suffix)

    # TODO：这个是在做什么？
    # 解答：非空 suffix 前加 "|"，便于后面直接拼成如 [date|time|worker.py] 的日志时间戳。
    if suffix:
        suffix = f"|{suffix}"

    if use_pid is None:
        # TODO：LOG_PID这个是我们自定义的吗？
        # 解答：是项目约定的自定义环境变量，os.getenv 只负责读取；值为 1/true/yes 时在日志中加入进程 PID。
        use_pid = os.getenv("LOG_PID", "0").lower() in ("1", "true", "yes")

    if use_pid:
        pid = os.getpid()
        suffix = f"|pid={pid}{suffix}"

    tp_info = None

    # Color formatter class
    class ColorFormatter(logging.Formatter):
        """Formatter with colors and pretty output"""

        # ANSI color codes
        COLORS = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        RESET = "\033[0m"
        BOLD = "\033[1m"

        # TODO：record是什么类型？
        # 解答：record 是 logging.LogRecord，包含 levelname、消息参数、创建时间、模块名等一次日志事件的元数据。
        def format(self, record):
            from minisgl.distributed import try_get_tp_info

            # Format timestamp like SGLang: [YYYY-MM-DD|HH:MM:SS|pid=1234]
            timestamp = self.formatTime(record, "[%Y-%m-%d|%H:%M:%S{suffix}]")
            nonlocal tp_info
            tp_info = tp_info or try_get_tp_info()
            if tp_info is not None and use_tp_rank is not False:
                real_suffix = f"{suffix}|core|rank={tp_info.rank}"
            else:
                real_suffix = suffix
            timestamp = timestamp.format(suffix=real_suffix)

            # Get color for log level
            level_color = self.COLORS.get(record.levelname, "")

            # Format the message
            colored_level = f"{level_color}{record.levelname:<8}{self.RESET}"
            message = record.getMessage()

            # Pretty format: [timestamp] LEVEL message
            return f"{self.BOLD}{timestamp}{self.RESET} {colored_level} {message}"

    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = ColorFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    def _call_rank0(msg, *args, _which, **kwargs):
        from minisgl.distributed import get_tp_info

        nonlocal tp_info
        tp_info = tp_info or get_tp_info()
        assert tp_info is not None, "TP info not set yet"
        if tp_info.is_primary():
            getattr(logger, _which)(msg, *args, **kwargs)

    # TODO：这里是在做什么？
    # 解答：TYPE_CHECKING 仅在静态类型检查时为 True，此分支用 WrapperLogger 告诉 IDE/检查器动态添加的 rank0 方法；真实运行总走 else。
    if TYPE_CHECKING:

        class WrapperLogger(logging.Logger):
            """Custom logger to handle the color formatter."""

            # TODO：这里定义的这些函数如何使用 会有什么效果？
            # 解答：它们是类型声明用的方法 stub；logger.info_rank0(...) 等调用在运行时由下方 partial 实现，只让 TP rank 0 输出对应级别的日志。
            def info_rank0(self, msg, *args, **kwargs): ...
            def warning_rank0(self, msg, *args, **kwargs): ...
            def debug_rank0(self, msg, *args, **kwargs): ...
            def critical_rank0(self, msg, *args, **kwargs): ...

        return WrapperLogger(name)
    else:
        logger.info_rank0 = partial(_call_rank0, _which="info")
        logger.debug_rank0 = partial(_call_rank0, _which="debug")
        logger.critical_rank0 = partial(_call_rank0, _which="critical")
        logger.warning_rank0 = partial(_call_rank0, _which="warning")
        return logger
