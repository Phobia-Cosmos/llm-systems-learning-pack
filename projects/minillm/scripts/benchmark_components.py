from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minillm.benchmark import (  # noqa: E402
    BenchmarkSettings,
    SUPPORTED_BENCHMARK_SUITES,
    environment_summary,
    run_component_benchmark,
    write_benchmark_outputs,
)


def format_run_summary(payload: dict) -> str:
    return (
        f"wrote {len(payload['aggregates'])} variants / "
        f"{len(payload['results'])} result rows"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fairly compare MiniLLM position, normalization, MLP, and "
            "MHA/GQA/MQA attention variants."
        )
    )
    parser.add_argument("--data", default="data/tiny_corpus.txt")
    parser.add_argument("--out-dir", default="benchmarks/results")
    parser.add_argument("--run-name", default="components_cpu")
    parser.add_argument(
        "--suites",
        default=",".join(SUPPORTED_BENCHMARK_SUITES),
        help="Comma-separated subset of: position,norm,mlp,attention",
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=1)
    parser.add_argument(
        "--n-head",
        type=int,
        default=4,
        help=(
            "Number of query heads. The attention suite uses this for MHA, 1 for MQA, "
            "and the largest proper divisor for GQA (so it must be composite)."
        ),
    )
    parser.add_argument("--n-embd", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--model-seeds",
        default="1337,1338,1339",
        help="Comma-separated seeds; use one seed only for a smoke test.",
    )
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--prompt", default="LLM 是")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--generation-repeats", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_name):
        raise ValueError("--run-name may contain only letters, numbers, dot, underscore, and dash")
    suites = tuple(part.strip() for part in args.suites.split(",") if part.strip())
    model_seeds = tuple(int(part.strip()) for part in args.model_seeds.split(",") if part.strip())
    settings = BenchmarkSettings(
        data_path=args.data,
        device=args.device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        eval_batches=args.eval_batches,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        model_seeds=model_seeds,
        data_seed=args.data_seed,
        prompt=args.prompt.replace("\\n", "\n"),
        max_new_tokens=args.max_new_tokens,
        generation_repeats=args.generation_repeats,
        torch_threads=args.torch_threads,
    )
    print(f"{environment_summary()} device={args.device} suites={','.join(suites)}")
    payload = run_component_benchmark(settings, suites=suites)

    out_dir = Path(args.out_dir)
    json_path = out_dir / f"{args.run_name}.json"
    csv_path = out_dir / f"{args.run_name}.csv"
    markdown_path = out_dir / f"{args.run_name}.md"
    write_benchmark_outputs(
        payload,
        json_path=json_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
    )
    print(format_run_summary(payload))
    print(json_path)
    print(csv_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
