"""Minimal tool-calling agent runtime over the trained MiniLLM checkpoint.

The model remains replaceable (native MiniLLM, nano-vLLM, or mini-sglang HTTP);
tool execution and safety limits stay outside the model/engine.
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import re
import sys
import time
import urllib.request
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from minillm import GPTConfig, MiniGPT  # noqa: E402
from minillm.tokenizer_registry import tokenizer_from_checkpoint  # noqa: E402


TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculator(expression: str) -> float:
    """Evaluate a deliberately tiny arithmetic grammar; never call eval()."""
    tree = ast.parse(expression, mode="eval").body

    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError("only numeric + - * / expressions are allowed")

    return float(visit(tree))


def parse_tool_call(text: str) -> dict | None:
    match = TOOL_RE.search(text)
    if not match:
        return None
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict) or payload.get("name") != "calculator":
        raise ValueError("unsupported tool; only calculator is enabled")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict) or not isinstance(arguments.get("expression"), str):
        raise ValueError("calculator requires string argument 'expression'")
    return payload


def native_generate(checkpoint: Path, prompt: str, device: str, max_tokens: int) -> str:
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = GPTConfig(**checkpoint_data["config"])
    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint_data["model"])
    model.eval()
    tokenizer = tokenizer_from_checkpoint(checkpoint_data)
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model.generate_with_kv_cache(ids, max_new_tokens=max_tokens, temperature=0.0, top_k=None, greedy=True)[0]
    return tokenizer.decode(output.tolist())[len(prompt):]


def http_generate(url: str, prompt: str, max_tokens: int) -> str:
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "top_k": 0, "greedy": True, "kv_cache": True}).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read())
    return str(payload["choices"][0]["text"])


def run_agent(args: argparse.Namespace) -> dict:
    prompt = args.prompt
    if args.plan:
        model_text = '<tool_call>{"name":"calculator","arguments":{"expression":"2+2"}}</tool_call>'
        backend = "scripted-plan"
    elif args.backend == "native":
        model_text = native_generate(Path(args.checkpoint), prompt, args.device, args.max_tokens)
        backend = "native-minillm"
    else:
        model_text = http_generate(args.url, prompt, args.max_tokens)
        backend = "mini-sglang-http"
    trace = {"schema_version": 1, "backend": backend, "prompt": prompt, "model_text": model_text, "events": []}
    call = parse_tool_call(model_text)
    if call is None:
        trace["status"] = "no-tool-call"
        return trace
    expression = call["arguments"]["expression"]
    started = time.perf_counter()
    result = calculator(expression)
    trace["events"].append({"type": "tool_call", "name": "calculator", "arguments": {"expression": expression}, "result": result, "latency_ms": (time.perf_counter() - started) * 1000.0})
    trace["status"] = "tool-completed"
    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("native", "http"), default="native")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "artifacts/checkpoints/minillm-rope.pt"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--url", default="http://127.0.0.1:8013/v1/completions")
    parser.add_argument("--prompt", default="请计算 2+2，并严格输出 calculator 工具调用。")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--plan", action="store_true", help="run a deterministic tool-call plan without model generation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = run_agent(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(trace, ensure_ascii=False))


if __name__ == "__main__":
    main()
