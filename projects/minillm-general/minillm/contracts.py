"""Small backend-neutral generation contract used by the local experiments.

The contract intentionally contains text and token ids.  Text is convenient for
an agent, while token ids make native/nano-vLLM parity testable without relying
on tokenizer string slicing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_k: int | None = None
    greedy: bool = True
    kv_cache: bool = True
    request_id: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive or None")


@dataclass(slots=True)
class GenerationResult:
    backend: str
    prompt: str
    text: str
    token_ids: list[int]
    prompt_token_ids: list[int] = field(default_factory=list)
    elapsed_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GenerationBackend(Protocol):
    name: str

    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...

