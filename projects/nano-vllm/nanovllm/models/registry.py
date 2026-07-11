from __future__ import annotations

import importlib
import json
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from torch import nn
from transformers import AutoTokenizer, PretrainedConfig

# TODO：这个是需要我们自己生成还是？
TokenizerLoader = Callable[[str], object]

# TODO：frozen是什么意思？
@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    architectures: tuple[str, ...]
    # TODO：这里定义type的作用是什么？为什么model class是nn.Module？
    config_class: type[PretrainedConfig]
    model_class: type[nn.Module]
    tokenizer_loader: TokenizerLoader | None = None


_MODEL_SPECS: list[ModelSpec] = []
_builtins_loaded = False

# TODO：为什么要在函数内部定义def？为什么返回的是一个decorator？
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

# TODO：这里是把符合的model全部列出来是吗？
def _load_builtin_models() -> None:
    global _builtins_loaded
    if _builtins_loaded:
        return
    models_package = importlib.import_module("nanovllm.models")
    # TODO：为什么要判断module name符合的的条件？
    for module in pkgutil.iter_modules(models_package.__path__):
        if not module.name.startswith("_") and module.name != "registry":
            importlib.import_module(f"{models_package.__name__}.{module.name}")
    _builtins_loaded = True

# TODO：为什么要匹配model type和architectures？
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
    # TODO：这里是使用默认的分词器是吗？
    return AutoTokenizer.from_pretrained(model_path, use_fast=True)


def supported_model_types() -> tuple[str, ...]:
    _load_builtin_models()
    return tuple(sorted(spec.model_type for spec in _MODEL_SPECS))
