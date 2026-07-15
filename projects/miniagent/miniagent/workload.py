"""Generate and independently verify fixed-token ShareGPT workloads."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


_GENERATOR = r'''
import json
import sys
from transformers import AutoTokenizer

model, target_text, count_text, output_text = sys.argv[1:]
target = int(target_text)
count = int(count_text)
output_tokens = int(output_text)
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
for index in range(count):
    prefix = f"Request {index}:"
    prefix_len = len(tokenizer(prefix, add_special_tokens=False).input_ids)
    if prefix_len >= target:
        raise RuntimeError(f"prefix is already {prefix_len} tokens")
    # For Qwen, a leading-space ASCII word is one token. Recheck every record
    # rather than trusting that tokenizer property silently.
    prompt = prefix + (" x" * (target - prefix_len))
    actual = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    if actual != target:
        raise RuntimeError(
            f"cannot construct exact prompt {index}: expected {target}, got {actual}"
        )
    print(json.dumps({
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": " x" * output_tokens},
        ]
    }, ensure_ascii=False))
'''


_VERIFIER = r'''
import json
import sys
from transformers import AutoTokenizer

model, path, target_text = sys.argv[1:]
target = int(target_text)
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
lengths = []
with open(path, encoding="utf-8") as handle:
    records = json.load(handle)
for record in records:
    prompt = record["conversations"][0]["value"]
    lengths.append(len(tokenizer(prompt).input_ids))
bad = [{"index": i, "tokens": n} for i, n in enumerate(lengths) if n != target]
print(json.dumps({
    "tokenizer_class": type(tokenizer).__name__,
    "count": len(lengths),
    "min_tokens": min(lengths) if lengths else None,
    "max_tokens": max(lengths) if lengths else None,
    "mismatch_count": len(bad),
    "first_mismatches": bad[:10],
}, sort_keys=True))
'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def generate_workload(
    output_path: Path,
    python: Path,
    model_path: Path,
    input_tokens: int,
    output_tokens: int,
    num_prompts: int,
) -> dict[str, Any]:
    """Generate ShareGPT JSON through the registered tokenizer environment.

    vLLM's custom JSONL loader imports the optional pandas benchmark extra,
    which is intentionally absent from the shared serving environment.  Its
    ShareGPT JSON loader is dependency-free and sends the first turn verbatim,
    so it is the portable carrier for our exact-token prompt text.
    """

    command = [
        str(python),
        "-c",
        _GENERATOR,
        str(model_path),
        str(input_tokens),
        str(num_prompts),
        str(output_tokens),
    ]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_tokenizer_env(),
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != num_prompts:
        raise RuntimeError(
            f"tokenizer generated {len(lines)} records, expected {num_prompts}: "
            f"{completed.stderr[-1000:]}"
        )
    records = [json.loads(line) for line in lines]
    for record in records:
        conversations = record.get("conversations")
        if (
            not isinstance(conversations, list)
            or len(conversations) < 2
            or not isinstance(conversations[0].get("value"), str)
        ):
            raise RuntimeError("generated workload contains an invalid prompt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "records": len(lines),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "generator_python": str(python),
        "generator_command": command[:2] + ["<embedded tokenizer generator>"] + command[3:],
    }


def verify_workload(
    workload_path: Path,
    python: Path,
    model_path: Path,
    input_tokens: int,
) -> dict[str, Any]:
    command = [
        str(python),
        "-c",
        _VERIFIER,
        str(model_path),
        str(workload_path),
        str(input_tokens),
    ]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_tokenizer_env(),
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    result["python"] = str(python)
    if result["mismatch_count"]:
        raise RuntimeError(
            f"{python} tokenizer does not preserve exact workload length: {result}"
        )
    return result
