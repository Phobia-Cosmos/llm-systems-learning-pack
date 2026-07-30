from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from minillm import GPTConfig, MiniGPT


TASKS = ("ceval", "cmmlu", "arc_easy", "arc_challenge", "hellaswag")
LABELS = ("A", "B", "C", "D", "E")


@dataclass(frozen=True)
class MultipleChoiceExample:
    task: str
    subject: str
    identifier: str
    prompt: str
    choices: tuple[str, ...]
    answer: int


@dataclass(frozen=True)
class EncodedCandidate:
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    score_start: int
    score_tokens: int


class ModelBackend(Protocol):
    name: str
    max_length: int
    pad_token_id: int
    parameters: int

    def encode(self, text: str) -> list[int]: ...

    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MiniLLM or a Hugging Face causal LM with one local MC harness."
    )
    parser.add_argument("--backend", required=True, choices=("minillm", "hf"))
    parser.add_argument("--model", required=True, help="MiniLLM checkpoint or Hugging Face model directory.")
    parser.add_argument("--tokenizer", default=None, help="MiniLLM tokenizer.json; defaults beside checkpoint.")
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--task", action="append", choices=TASKS)
    parser.add_argument("--limit-per-task", type=int, default=500, help="0 evaluates every example.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--score-content",
        action="store_true",
        help="Also score option text for letter-answer tasks; HellaSwag always scores endings.",
    )
    return parser.parse_args()


def stable_sample(
    examples: list[MultipleChoiceExample], limit: int, seed: int, task: str
) -> list[MultipleChoiceExample]:
    if limit <= 0 or len(examples) <= limit:
        return examples
    task_seed = int.from_bytes(hashlib.sha256(task.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed ^ task_seed)
    indices = sorted(rng.sample(range(len(examples)), limit))
    return [examples[index] for index in indices]


def format_letter_prompt(question: str, choices: list[str], language: str) -> str:
    lines = [question]
    lines.extend(f"{LABELS[index]}. {choice}" for index, choice in enumerate(choices))
    lines.append("答案：" if language == "zh" else "Answer:")
    return "\n".join(lines)


def load_ceval(root: Path) -> list[MultipleChoiceExample]:
    result: list[MultipleChoiceExample] = []
    for path in sorted((root / "ceval").glob("*/val-*.parquet")):
        subject = path.parent.name
        for row in pq.read_table(path).to_pylist():
            choices = [str(row[label]) for label in LABELS[:4]]
            result.append(
                MultipleChoiceExample(
                    task="ceval",
                    subject=subject,
                    identifier=f"{subject}:{row['id']}",
                    prompt=format_letter_prompt(str(row["question"]), choices, "zh"),
                    choices=tuple(choices),
                    answer=LABELS.index(str(row["answer"]).strip()),
                )
            )
    return result


def load_cmmlu(root: Path) -> list[MultipleChoiceExample]:
    archive = root / "cmmlu" / "cmmlu_v1_0_1.zip"
    result: list[MultipleChoiceExample] = []
    with zipfile.ZipFile(archive) as handle:
        test_names = sorted(
            name for name in handle.namelist() if name.startswith("test/") and name.endswith(".csv")
        )
        names = test_names or sorted(
            name for name in handle.namelist() if name.startswith("dev/") and name.endswith(".csv")
        )
        for name in names:
            subject = Path(name).stem
            text = handle.read(name).decode("utf-8-sig")
            for row_number, row in enumerate(csv.DictReader(text.splitlines())):
                choices = [str(row[label]) for label in LABELS[:4]]
                result.append(
                    MultipleChoiceExample(
                        task="cmmlu",
                        subject=subject,
                        identifier=f"{subject}:{row_number}",
                        prompt=format_letter_prompt(str(row["Question"]), choices, "zh"),
                        choices=tuple(choices),
                        answer=LABELS.index(str(row["Answer"]).strip()),
                    )
                )
    return result


def load_arc(root: Path, subset: str) -> list[MultipleChoiceExample]:
    task = "arc_easy" if subset == "ARC-Easy" else "arc_challenge"
    result: list[MultipleChoiceExample] = []
    for path in sorted((root / "arc" / subset).glob("validation-*.parquet")):
        for row in pq.read_table(path).to_pylist():
            labels = [str(value) for value in row["choices"]["label"]]
            choices = [str(value) for value in row["choices"]["text"]]
            answer_label = str(row["answerKey"])
            if answer_label not in labels:
                continue
            result.append(
                MultipleChoiceExample(
                    task=task,
                    subject=subset,
                    identifier=str(row["id"]),
                    prompt=format_letter_prompt(str(row["question"]), choices, "en"),
                    choices=tuple(choices),
                    answer=labels.index(answer_label),
                )
            )
    return result


def load_hellaswag(root: Path) -> list[MultipleChoiceExample]:
    result: list[MultipleChoiceExample] = []
    for path in sorted((root / "hellaswag" / "data").glob("validation-*.parquet")):
        for row in pq.read_table(path).to_pylist():
            result.append(
                MultipleChoiceExample(
                    task="hellaswag",
                    subject=str(row["activity_label"]),
                    identifier=str(row["ind"]),
                    prompt=str(row["ctx"]).rstrip(),
                    choices=tuple(str(value) for value in row["endings"]),
                    answer=int(row["label"]),
                )
            )
    return result


def load_task(root: Path, task: str) -> list[MultipleChoiceExample]:
    loaders = {
        "ceval": lambda: load_ceval(root),
        "cmmlu": lambda: load_cmmlu(root),
        "arc_easy": lambda: load_arc(root, "ARC-Easy"),
        "arc_challenge": lambda: load_arc(root, "ARC-Challenge"),
        "hellaswag": lambda: load_hellaswag(root),
    }
    examples = loaders[task]()
    if not examples:
        raise RuntimeError(f"no examples loaded for {task} from {root}")
    return examples


class MiniLLMBackend:
    def __init__(self, checkpoint_path: Path, tokenizer_path: Path, device: torch.device):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = GPTConfig(**checkpoint["config"])
        model = MiniGPT(config)
        model.load_state_dict(checkpoint["model"])
        self.model = model.to(device).eval()
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.device = device
        self.name = str(checkpoint_path)
        self.max_length = config.block_size
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>") or 0
        self.parameters = model.parameter_count()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            logits, _ = self.model(input_ids)
        return logits


class HuggingFaceBackend:
    def __init__(self, model_path: Path, device: torch.device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation="sdpa" if device.type == "cuda" else "eager",
            trust_remote_code=False,
        ).to(device)
        self.model.eval()
        self.device = device
        self.name = str(model_path)
        config_limit = getattr(self.model.config, "max_position_embeddings", None)
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        sane_limits = [
            int(value)
            for value in (config_limit, tokenizer_limit)
            if isinstance(value, int) and 1 < value < 10_000_000
        ]
        self.max_length = min(sane_limits) if sane_limits else 2048
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        self.pad_token_id = 0 if pad is None else int(pad)
        self.parameters = sum(parameter.numel() for parameter in self.model.parameters())

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        count += 1
    return count


def encode_candidate(
    backend: ModelBackend, prompt: str, continuation: str
) -> EncodedCandidate:
    prompt_ids = backend.encode(prompt)
    full_ids = backend.encode(prompt + continuation)
    boundary = common_prefix_length(prompt_ids, full_ids)
    if boundary == len(full_ids):
        raise ValueError("continuation encoded to no scoreable tokens")
    if boundary == 0:
        raise ValueError("prompt and prompt+continuation have no common token prefix")

    if len(full_ids) > backend.max_length:
        drop = len(full_ids) - backend.max_length
        if drop >= boundary:
            raise ValueError(
                f"continuation leaves no prompt token in context window {backend.max_length}"
            )
        full_ids = full_ids[drop:]
        boundary -= drop

    # Logits position j predicts token j+1. The first continuation token is at
    # token index `boundary`, hence its first scoring logit is boundary - 1.
    return EncodedCandidate(
        input_ids=tuple(full_ids[:-1]),
        target_ids=tuple(full_ids[1:]),
        score_start=boundary - 1,
        score_tokens=len(full_ids) - boundary,
    )


@torch.inference_mode()
def score_candidates(
    backend: ModelBackend,
    encoded: list[EncodedCandidate],
    batch_size: int,
    device: torch.device,
) -> list[tuple[float, float]]:
    scores: list[tuple[float, float]] = []
    for offset in range(0, len(encoded), batch_size):
        batch = encoded[offset : offset + batch_size]
        max_tokens = max(len(item.input_ids) for item in batch)
        inputs = torch.full(
            (len(batch), max_tokens),
            backend.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention = torch.zeros_like(inputs)
        for row, item in enumerate(batch):
            length = len(item.input_ids)
            inputs[row, :length] = torch.tensor(item.input_ids, dtype=torch.long, device=device)
            attention[row, :length] = 1

        logits = backend.logits(inputs, attention)
        for row, item in enumerate(batch):
            start = item.score_start
            stop = start + item.score_tokens
            selected_logits = logits[row, start:stop].float()
            targets = torch.tensor(
                item.target_ids[start:stop], dtype=torch.long, device=device
            )
            token_log_probs = F.log_softmax(selected_logits, dim=-1).gather(
                1, targets.unsqueeze(1)
            )
            total = float(token_log_probs.sum().item())
            scores.append((total, total / item.score_tokens))
    return scores


def candidate_schemes(
    example: MultipleChoiceExample, score_content: bool
) -> dict[str, tuple[str, ...]]:
    if example.task == "hellaswag":
        return {"ending": tuple(" " + choice.lstrip() for choice in example.choices)}
    schemes = {"label": tuple(LABELS[: len(example.choices)])}
    if score_content:
        schemes["content"] = tuple(" " + choice.lstrip() for choice in example.choices)
    return schemes


def evaluate_task(
    backend: ModelBackend,
    examples: list[MultipleChoiceExample],
    batch_size: int,
    device: torch.device,
    score_content: bool,
) -> dict[str, object]:
    flattened: list[EncodedCandidate] = []
    layout: list[tuple[str, int, int, int]] = []
    for example_index, example in enumerate(examples):
        for scheme, continuations in candidate_schemes(example, score_content).items():
            start = len(flattened)
            flattened.extend(
                encode_candidate(backend, example.prompt, continuation)
                for continuation in continuations
            )
            layout.append((scheme, example_index, start, len(continuations)))

    started = time.perf_counter()
    candidate_scores = score_candidates(backend, flattened, batch_size, device)
    elapsed = time.perf_counter() - started
    metrics: dict[str, dict[str, float | int]] = {}
    for scheme, example_index, start, count in layout:
        scores = candidate_scores[start : start + count]
        raw_prediction = max(range(count), key=lambda index: scores[index][0])
        normalized_prediction = max(range(count), key=lambda index: scores[index][1])
        answer = examples[example_index].answer
        metric = metrics.setdefault(
            scheme,
            {
                "examples": 0,
                "correct": 0,
                "correct_normalized": 0,
            },
        )
        metric["examples"] += 1
        metric["correct"] += int(raw_prediction == answer)
        metric["correct_normalized"] += int(normalized_prediction == answer)

    result_metrics: dict[str, dict[str, float | int]] = {}
    for scheme, metric in metrics.items():
        count = int(metric["examples"])
        accuracy = int(metric["correct"]) / count
        normalized = int(metric["correct_normalized"]) / count
        result_metrics[scheme] = {
            **metric,
            "accuracy": accuracy,
            "accuracy_normalized": normalized,
            "standard_error": math.sqrt(accuracy * (1.0 - accuracy) / count),
            "standard_error_normalized": math.sqrt(
                normalized * (1.0 - normalized) / count
            ),
        }
    return {
        "examples": len(examples),
        "subjects": len({example.subject for example in examples}),
        "random_accuracy": sum(1.0 / len(example.choices) for example in examples)
        / len(examples),
        "candidate_sequences": len(flattened),
        "scored_tokens": sum(item.score_tokens for item in flattened),
        "elapsed_seconds": elapsed,
        "metrics": result_metrics,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.limit_per_task < 0:
        raise ValueError("--limit-per-task must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model_path = Path(args.model).expanduser().resolve()
    if args.backend == "minillm":
        tokenizer_path = (
            Path(args.tokenizer).expanduser().resolve()
            if args.tokenizer
            else model_path.parent / "tokenizer.json"
        )
        backend: ModelBackend = MiniLLMBackend(model_path, tokenizer_path, device)
    else:
        backend = HuggingFaceBackend(model_path, device)

    root = Path(args.datasets_dir).expanduser().resolve()
    selected_tasks = args.task or list(TASKS)
    results: dict[str, object] = {}
    for task in selected_tasks:
        all_examples = load_task(root, task)
        examples = stable_sample(all_examples, args.limit_per_task, args.seed, task)
        result = evaluate_task(
            backend,
            examples,
            args.batch_size,
            device,
            args.score_content,
        )
        result["available_examples"] = len(all_examples)
        results[task] = result
        print(json.dumps({"task": task, **result}, ensure_ascii=False), flush=True)

    payload = {
        "schema_version": 1,
        "harness": {
            "choice_method": "conditional_log_likelihood",
            "few_shot": 0,
            "seed": args.seed,
            "limit_per_task": args.limit_per_task,
            "score_content": args.score_content,
        },
        "model": {
            "backend": args.backend,
            "path": str(model_path),
            "name": backend.name,
            "parameters": backend.parameters,
            "max_length": backend.max_length,
        },
        "tasks": results,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
