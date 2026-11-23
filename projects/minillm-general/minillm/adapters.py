"""Adapters that make the three local runtimes implement one contract."""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import torch

from .contracts import GenerationBackend, GenerationRequest, GenerationResult
from . import GPTConfig, MiniGPT
from .tokenizer_registry import tokenizer_from_checkpoint


class NativeMiniLLMBackend:
    name = "native-minillm"

    def __init__(self, checkpoint: str | Path, device: str = "cuda") -> None:
        self.checkpoint_path = Path(checkpoint)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        checkpoint_data = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.tokenizer = tokenizer_from_checkpoint(checkpoint_data)
        self.model = MiniGPT(GPTConfig(**checkpoint_data["config"])).to(self.device)
        self.model.load_state_dict(checkpoint_data["model"])
        self.model.eval()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        prompt_ids = self.tokenizer.encode(request.prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generate_fn = self.model.generate_with_kv_cache if request.kv_cache else self.model.generate
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = generate_fn(
                input_ids,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                greedy=request.greedy,
            )[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        output_ids = output.tolist()
        completion_ids = output_ids[len(prompt_ids) :]
        return GenerationResult(
            backend=self.name,
            prompt=request.prompt,
            text=self.tokenizer.decode(completion_ids),
            token_ids=completion_ids,
            prompt_token_ids=prompt_ids,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            metadata={"device": str(self.device), "kv_cache": request.kv_cache},
        )


class NanoVLLMBackend:
    name = "nano-vllm"

    def __init__(self, model_dir: str | Path, **engine_kwargs) -> None:
        nano_root = Path(__file__).resolve().parents[2] / "nano-vllm"
        if str(nano_root) not in sys.path:
            sys.path.insert(0, str(nano_root))
        from nanovllm import LLM, SamplingParams  # noqa: WPS433

        self._SamplingParams = SamplingParams
        self.engine = LLM(str(model_dir), **engine_kwargs)
        self.tokenizer = self.engine.tokenizer

    def generate(self, request: GenerationRequest) -> GenerationResult:
        sampling = self._SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_new_tokens,
            ignore_eos=True,
        )
        started = time.perf_counter()
        output = self.engine.generate([request.prompt], sampling, use_tqdm=False)[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        token_ids = [int(token_id) for token_id in output["token_ids"]]
        return GenerationResult(
            backend=self.name,
            prompt=request.prompt,
            text=str(output["text"]),
            token_ids=token_ids,
            prompt_token_ids=[int(token_id) for token_id in self.tokenizer.encode(request.prompt)],
            elapsed_ms=elapsed_ms,
            metadata={"engine_output": "completion-only"},
        )

    def close(self) -> None:
        self.engine.exit()


class MiniSGLangHTTPBackend:
    name = "mini-sglang-http"

    def __init__(self, url: str) -> None:
        self.url = url

    def generate(self, request: GenerationRequest) -> GenerationResult:
        body = json.dumps(
            {
                "prompt": request.prompt,
                "max_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_k": request.top_k or 0,
                "greedy": request.greedy,
                "kv_cache": request.kv_cache,
            }
        ).encode()
        http_request = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        with urllib.request.urlopen(http_request, timeout=120) as response:
            payload = json.loads(response.read())
        text = str(payload["choices"][0]["text"])
        choice = payload["choices"][0]
        token_ids = [int(token_id) for token_id in choice.get("token_ids", payload.get("completion_token_ids", []))]
        prompt_token_ids = [int(token_id) for token_id in payload.get("prompt_token_ids", [])]
        return GenerationResult(
            backend=self.name,
            prompt=request.prompt,
            text=text,
            token_ids=token_ids,
            prompt_token_ids=prompt_token_ids,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            metadata={"url": self.url},
        )


__all__ = ["GenerationBackend", "GenerationRequest", "GenerationResult", "NativeMiniLLMBackend", "NanoVLLMBackend", "MiniSGLangHTTPBackend"]
