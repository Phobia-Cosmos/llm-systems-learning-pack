from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch


AI_ROOT = Path(__file__).resolve().parents[2]
MINILLM_ROOT = AI_ROOT / "projects" / "minillm"
sys.path.insert(0, str(MINILLM_ROOT))

from minillm import GPTConfig, MiniGPT  # noqa: E402
from minillm.tokenizer_registry import MiniTokenizer, tokenizer_from_checkpoint  # noqa: E402


def load_minillm(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = tokenizer_from_checkpoint(checkpoint)
    model = MiniGPT(GPTConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer


def messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = str(message.get("content", ""))
        if role == "assistant":
            lines.append(f"助手: {content}")
        else:
            lines.append(f"用户: {content}")
    lines.append("助手:")
    return "\n".join(lines)


class MiniSGLangHandler(BaseHTTPRequestHandler):
    model: MiniGPT
    tokenizer: MiniTokenizer
    device: str

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": f"unknown route: {self.path}"})

    def do_POST(self) -> None:
        try:
            if self.path == "/v1/completions":
                self._send_completion(self._read_json())
            elif self.path == "/v1/chat/completions":
                self._send_chat_completion(self._read_json())
            else:
                self._send_json(404, {"error": f"unknown route: {self.path}"})
        except Exception as exc:
            self._send_json(500, {"error": type(exc).__name__, "message": str(exc)})

    def _generate(self, prompt: str, request: dict[str, Any]) -> str:
        max_new_tokens = int(request.get("max_tokens", request.get("max_new_tokens", 80)))
        temperature = float(request.get("temperature", 0.8))
        top_k = request.get("top_k", 40)
        greedy = bool(request.get("greedy", False))
        use_kv_cache = bool(request.get("kv_cache", False))
        token_ids = self.tokenizer.encode(prompt)
        idx = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        generate_fn = self.model.generate_with_kv_cache if use_kv_cache else self.model.generate
        with torch.no_grad():
            output_ids = generate_fn(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=None if top_k is None or int(top_k) <= 0 else int(top_k),
                greedy=greedy,
            )[0].tolist()
        text = self.tokenizer.decode(output_ids)
        return text[len(prompt) :]

    def _send_completion(self, request: dict[str, Any]) -> None:
        prompt = str(request.get("prompt", ""))
        generated = self._generate(prompt, request)
        now = int(time.time())
        self._send_json(
            200,
            {
                "id": f"cmpl-minillm-{now}",
                "object": "text_completion",
                "created": now,
                "model": "minillm",
                "choices": [{"index": 0, "text": generated, "finish_reason": "length"}],
            },
        )

    def _send_chat_completion(self, request: dict[str, Any]) -> None:
        prompt = messages_to_prompt(request.get("messages", []))
        generated = self._generate(prompt, request)
        now = int(time.time())
        self._send_json(
            200,
            {
                "id": f"chatcmpl-minillm-{now}",
                "object": "chat.completion",
                "created": now,
                "model": "minillm",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": generated},
                        "finish_reason": "length",
                    }
                ],
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny OpenAI-like server for MiniLLM.")
    parser.add_argument("--checkpoint", default=str(MINILLM_ROOT / "checkpoints" / "minillm.pt"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MiniSGLangHandler.model, MiniSGLangHandler.tokenizer = load_minillm(args.checkpoint, args.device)
    MiniSGLangHandler.device = args.device
    server = ThreadingHTTPServer((args.host, args.port), MiniSGLangHandler)
    print(f"mini-sglang serving MiniLLM on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
