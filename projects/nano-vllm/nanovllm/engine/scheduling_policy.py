from abc import ABC, abstractmethod
from enum import Enum


class BatchPhase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class SchedulingPolicy(ABC):
    """Selects which scheduler phase gets the next model invocation."""

    name: str

    @property
    @abstractmethod
    def phase_order(self) -> tuple[BatchPhase, BatchPhase]:
        """Return phases in the order in which they should be attempted."""


class PrefillFirstPolicy(SchedulingPolicy):
    name = "prefill_first"
    phase_order = (BatchPhase.PREFILL, BatchPhase.DECODE)


class DecodeFirstPolicy(SchedulingPolicy):
    name = "decode_first"
    phase_order = (BatchPhase.DECODE, BatchPhase.PREFILL)


_POLICIES: dict[str, type[SchedulingPolicy]] = {
    PrefillFirstPolicy.name: PrefillFirstPolicy,
    DecodeFirstPolicy.name: DecodeFirstPolicy,
}


def normalize_scheduling_policy(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if normalized not in _POLICIES:
        choices = ", ".join(sorted(_POLICIES))
        raise ValueError(f"Unknown scheduling_policy={name!r}; expected one of: {choices}")
    return normalized


def create_scheduling_policy(name: str) -> SchedulingPolicy:
    return _POLICIES[normalize_scheduling_policy(name)]()
