from __future__ import annotations

import pytest

from minillm.contracts import GenerationRequest, GenerationResult


def test_generation_request_validation_and_defaults() -> None:
    request = GenerationRequest("hello")
    assert request.max_new_tokens == 32
    assert request.greedy is True
    assert request.kv_cache is True

    with pytest.raises(ValueError):
        GenerationRequest("")
    with pytest.raises(ValueError):
        GenerationRequest("hello", max_new_tokens=-1)
    with pytest.raises(ValueError):
        GenerationRequest("hello", top_k=0)


def test_generation_result_serializes_completion_ids() -> None:
    result = GenerationResult("native-minillm", "hello", " world", [4, 5], [1, 2], 1.5)
    payload = result.to_dict()
    assert result.token_count == 2
    assert payload["token_ids"] == [4, 5]
    assert payload["prompt_token_ids"] == [1, 2]
