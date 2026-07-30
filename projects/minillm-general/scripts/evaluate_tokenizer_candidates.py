#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "8")

from tokenizers import Tokenizer

from train_tokenizer_candidates import PROBES


@dataclass
class DomainStats:
    documents: int = 0
    characters: int = 0
    bytes: int = 0
    tokens: int = 0
    unknown_tokens: int = 0
    roundtrip_failures: int = 0
    seconds: float = 0.0
    tokens_per_kib: list[float] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tokenizer size, coverage and held-out compression.")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tokenizer",
        action="append",
        required=True,
        help="NAME=/path/to/tokenizer.json; repeat for each candidate/reference.",
    )
    parser.add_argument("--max-documents-per-domain", type=int, default=2000)
    parser.add_argument("--hidden-size", type=int, default=768)
    return parser.parse_args()


def parse_tokenizers(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid tokenizer argument {value!r}; expected NAME=/path/tokenizer.json")
        name, path = value.split("=", 1)
        resolved = Path(path).expanduser().resolve()
        if not name or not resolved.is_file():
            raise FileNotFoundError(f"invalid tokenizer {value!r}")
        result[name] = resolved
    return result


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_validation(path: Path, max_per_domain: int) -> dict[str, list[str]]:
    documents: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            domain = payload["domain"]
            if len(documents[domain]) < max_per_domain:
                documents[domain].append(payload["text"])
    return dict(documents)


def evaluate_tokenizer(tokenizer_path: Path, documents: dict[str, list[str]], hidden_size: int) -> dict:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    unk_id = tokenizer.token_to_id("<|unk|>")
    domain_results: dict[str, dict] = {}
    total = DomainStats()
    for domain, texts in sorted(documents.items()):
        stats = DomainStats()
        for text in texts:
            expected = unicodedata.normalize("NFC", text)
            started = time.perf_counter()
            encoding = tokenizer.encode(text, add_special_tokens=False)
            elapsed = time.perf_counter() - started
            decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
            byte_count = len(expected.encode("utf-8"))
            token_count = len(encoding.ids)
            stats.documents += 1
            stats.characters += len(expected)
            stats.bytes += byte_count
            stats.tokens += token_count
            stats.seconds += elapsed
            stats.unknown_tokens += sum(token_id == unk_id for token_id in encoding.ids) if unk_id is not None else 0
            stats.roundtrip_failures += decoded != expected
            if byte_count:
                stats.tokens_per_kib.append(token_count * 1024 / byte_count)
        domain_results[domain] = summarize(stats)
        accumulate(total, stats)

    probe_failures = []
    for probe in PROBES:
        expected = unicodedata.normalize("NFC", probe)
        encoding = tokenizer.encode(probe, add_special_tokens=False)
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
        unknowns = sum(token_id == unk_id for token_id in encoding.ids) if unk_id is not None else 0
        if decoded != expected or unknowns:
            probe_failures.append(
                {"text": probe, "decoded": decoded, "unknown_tokens": unknowns, "tokens": len(encoding.ids)}
            )

    vocab_size = tokenizer.get_vocab_size()
    return {
        "path": str(tokenizer_path),
        "sha256": sha256(tokenizer_path),
        "file_bytes": tokenizer_path.stat().st_size,
        "vocab_size": vocab_size,
        "embedding_parameters_at_hidden_size": vocab_size * hidden_size,
        "embedding_bf16_mib": vocab_size * hidden_size * 2 / (1024 * 1024),
        "hidden_size": hidden_size,
        "overall": summarize(total),
        "domains": domain_results,
        "probe_failures": probe_failures,
    }


def accumulate(target: DomainStats, source: DomainStats) -> None:
    target.documents += source.documents
    target.characters += source.characters
    target.bytes += source.bytes
    target.tokens += source.tokens
    target.unknown_tokens += source.unknown_tokens
    target.roundtrip_failures += source.roundtrip_failures
    target.seconds += source.seconds
    target.tokens_per_kib.extend(source.tokens_per_kib)


def summarize(stats: DomainStats) -> dict:
    return {
        "documents": stats.documents,
        "characters": stats.characters,
        "bytes": stats.bytes,
        "tokens": stats.tokens,
        "characters_per_token": stats.characters / stats.tokens if stats.tokens else math.nan,
        "bytes_per_token": stats.bytes / stats.tokens if stats.tokens else math.nan,
        "tokens_per_kib_p50": statistics.median(stats.tokens_per_kib) if stats.tokens_per_kib else math.nan,
        "tokens_per_kib_p95": percentile(stats.tokens_per_kib, 0.95),
        "unknown_tokens": stats.unknown_tokens,
        "roundtrip_failures": stats.roundtrip_failures,
        "encoding_megabytes_per_second": (
            stats.bytes / (1024 * 1024) / stats.seconds if stats.seconds > 0 else math.nan
        ),
    }


def markdown_report(results: dict[str, dict]) -> str:
    lines = [
        "# MiniLLM tokenizer lab",
        "",
        "| Tokenizer | Vocab | File MiB | Embedding params @768 | BF16 embedding MiB | Bytes/token | P95 tokens/KiB | `<unk>` | Round-trip failures | Encode MiB/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        overall = result["overall"]
        lines.append(
            f"| {name} | {result['vocab_size']:,} | {result['file_bytes'] / (1024 * 1024):.2f} | "
            f"{result['embedding_parameters_at_hidden_size']:,} | {result['embedding_bf16_mib']:.1f} | "
            f"{overall['bytes_per_token']:.3f} | {overall['tokens_per_kib_p95']:.1f} | "
            f"{overall['unknown_tokens']} | {overall['roundtrip_failures']} | "
            f"{overall['encoding_megabytes_per_second']:.1f} |"
        )
    domains = sorted({domain for result in results.values() for domain in result["domains"]})
    lines.extend(
        [
            "",
            "## Held-out bytes per token by domain",
            "",
            "| Domain | " + " | ".join(results) + " |",
            "|---|" + "|".join("---:" for _ in results) + "|",
        ]
    )
    for domain in domains:
        values = [
            f"{results[name]['domains'][domain]['bytes_per_token']:.3f}"
            if domain in results[name]["domains"]
            else "—"
            for name in results
        ]
        lines.append(f"| {domain} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    validation = Path(args.validation).expanduser().resolve()
    documents = collect_validation(validation, args.max_documents_per_domain)
    if not documents:
        raise RuntimeError(f"no validation documents found in {validation}")
    tokenizer_paths = parse_tokenizers(args.tokenizer)
    results = {
        name: evaluate_tokenizer(path, documents, args.hidden_size)
        for name, path in tokenizer_paths.items()
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "validation_path": str(validation),
        "validation_sha256": sha256(validation),
        "max_documents_per_domain": args.max_documents_per_domain,
        "results": results,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = markdown_report(results)
    (output_dir / "evaluation.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
