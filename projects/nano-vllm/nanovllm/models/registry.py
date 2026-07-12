from __future__ import annotations

import importlib
import json
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from torch import nn
from transformers import AutoTokenizer, PretrainedConfig

# 问题（已回答）：TokenizerLoader 需要自己生成吗？
# 回答：它只是类型别名，表示“模型目录 -> tokenizer”的可调用接口；特殊模型自行提供 loader，普通模型用 AutoTokenizer。
TokenizerLoader = Callable[[str], object]

# 问题（已回答）：dataclass(frozen=True) 的 frozen 是什么？
# 回答：实例创建后字段不可重新赋值，使注册记录只读，避免运行中误改映射。
@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    architectures: tuple[str, ...]
    # 问题（已回答）：type[...] 表示什么，model_class 为什么是 nn.Module？
    # 回答：type[PretrainedConfig] 表示类对象而非实例；model_class 必须是 nn.Module 子类，
    # 因为 engine 依赖其参数注册、device/dtype、state_dict 和 forward 约定。
    config_class: type[PretrainedConfig]
    model_class: type[nn.Module]
    tokenizer_loader: TokenizerLoader | None = None


_MODEL_SPECS: list[ModelSpec] = []
_builtins_loaded = False

# 问题（已回答）：为什么内部定义 decorator 并返回它？
# 回答：这是带参数装饰器：register_model(...) 先保存元数据并返回 decorator，Python 再把模型类传给它；
# decorator 登记 ModelSpec 后原样返回模型类。
def register_model(
    *,
    model_type: str,
    architectures: tuple[str, ...],
    config_class: type[PretrainedConfig],
    tokenizer_loader: TokenizerLoader | None = None,
):
    """Register a model implementation without coupling it to the engine."""

    def decorator(model_class: type[nn.Module]) -> type[nn.Module]:
        spec = ModelSpec(
            model_type=model_type,
            architectures=architectures,
            config_class=config_class,
            model_class=model_class,
            tokenizer_loader=tokenizer_loader,
        )
        for existing in _MODEL_SPECS:
            same_model_type = existing.model_type == model_type
            shared_architecture = bool(set(existing.architectures) & set(architectures))
            if same_model_type or shared_architecture:
                raise ValueError(f"Model registration conflicts with {existing}")
        _MODEL_SPECS.append(spec)
        return model_class

    return decorator

# 问题（已回答）：_load_builtin_models 是列出所有模型吗？
# 回答：它扫描并 import 内置模型模块；import 会执行 @register_model 填充 registry，但不会加载权重或实例化所有模型。
def _load_builtin_models() -> None:
    global _builtins_loaded
    if _builtins_loaded:
        return
    models_package = importlib.import_module("nanovllm.models")
    # 问题（已回答）：为什么过滤模块名？
    # 回答：下划线模块通常是内部实现，registry 自身也不能递归导入；这里只加载公开模型模块。
    for module in pkgutil.iter_modules(models_package.__path__):
        if not module.name.startswith("_") and module.name != "registry":
            importlib.import_module(f"{models_package.__name__}.{module.name}")
    _builtins_loaded = True

# 问题（已回答）：为什么同时匹配 model_type 和 architectures？
# 回答：不同 HF/自定义 config 可能只可靠填写其中一个；双通道提高兼容性，多重命中则报歧义以避免选错实现。
def _resolve_model(model_type: str | None, architectures: list[str] | tuple[str, ...] | None) -> ModelSpec:
    _load_builtin_models()
    architecture_set = set(architectures or ())
    matches = [
        spec
        for spec in _MODEL_SPECS
        if spec.model_type == model_type or architecture_set.intersection(spec.architectures)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous model configuration: model_type={model_type!r}, architectures={sorted(architecture_set)!r}"
        )
    supported = ", ".join(sorted(spec.model_type for spec in _MODEL_SPECS))
    raise ValueError(
        f"Unsupported model: model_type={model_type!r}, architectures={sorted(architecture_set)!r}. "
        f"Supported model types: {supported}"
    )


def _read_config(model_path: str) -> dict:
    config_path = Path(model_path) / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing model config: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid model config: {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Model config must contain a JSON object: {config_path}")
    return payload


def load_model_config(model_path: str) -> PretrainedConfig:
    payload = _read_config(model_path)
    spec = _resolve_model(payload.get("model_type"), payload.get("architectures"))
    return spec.config_class.from_pretrained(model_path)


def create_model(config: PretrainedConfig) -> nn.Module:
    spec = _resolve_model(getattr(config, "model_type", None), getattr(config, "architectures", None))
    return spec.model_class(config)


def load_tokenizer(model_path: str, config: PretrainedConfig):
    spec = _resolve_model(getattr(config, "model_type", None), getattr(config, "architectures", None))
    if spec.tokenizer_loader is not None:
        return spec.tokenizer_loader(model_path)
    # 问题（已回答）：这里使用默认分词器吗？
    # 回答：是。未注册专用 loader 时用 HF AutoTokenizer 加载 fast tokenizer；字符词表等非标准格式需专用 loader。
    return AutoTokenizer.from_pretrained(model_path, use_fast=True)


def supported_model_types() -> tuple[str, ...]:
    _load_builtin_models()
    return tuple(sorted(spec.model_type for spec in _MODEL_SPECS))
