from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minillm.inference_baseline import (  # noqa: E402
    TimerSettings,
    run_inference_baseline,
    write_baseline_outputs,
)


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected one or more positive comma-separated integers")
    return parsed


def _comma_separated_dtypes(value: str) -> tuple[torch.dtype, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    try:
        return tuple(DTYPES[name] for name in names)
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"unsupported dtype {exc.args[0]!r}; choose from {','.join(DTYPES)}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MiniLLM and nano-vLLM inference shapes with eager PyTorch/cuBLAS."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "artifacts" / "checkpoints" / "minillm-rope.pt"),
    )
    parser.add_argument("--prompt", default="embedding 是把离散 token id 映射成")
    parser.add_argument("--batch-sizes", type=_comma_separated_ints, default=(1, 8))
    parser.add_argument(
        "--dtypes", type=_comma_separated_dtypes, default=(torch.float32, torch.float16)
    )
    parser.add_argument("--generated-tokens", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--target-sample-ms", type=float, default=3.0)
    parser.add_argument("--max-inner-loops", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "benchmarks" / "results"))
    parser.add_argument("--run-name", default="inference_pytorch_cublas")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_name):
        raise ValueError("--run-name may contain only letters, numbers, dot, underscore, and dash")
    if args.generated_tokens <= 0:
        raise ValueError("--generated-tokens must be positive")
    if args.page_size <= 0:
        raise ValueError("--page-size must be positive")
    if args.warmup < 0 or args.samples <= 0 or args.max_inner_loops <= 0:
        raise ValueError("timer counts must be non-negative, with samples and max-inner-loops positive")
    if args.target_sample_ms <= 0:
        raise ValueError("--target-sample-ms must be positive")

    settings = TimerSettings(
        warmup=args.warmup,
        samples=args.samples,
        target_sample_ms=args.target_sample_ms,
        max_inner_loops=args.max_inner_loops,
    )
    payload = run_inference_baseline(
        args.checkpoint,
        args.prompt.replace("\\n", "\n"),
        batch_sizes=args.batch_sizes,
        dtypes=args.dtypes,
        generated_tokens=args.generated_tokens,
        page_size=args.page_size,
        timer_settings=settings,
        device=args.device,
        project_root=AI_ROOT,
    )

    output_directory = Path(args.out_dir)
    json_path = output_directory / f"{args.run_name}.json"
    csv_path = output_directory / f"{args.run_name}.csv"
    markdown_path = output_directory / f"{args.run_name}.md"
    write_baseline_outputs(
        payload,
        json_path=json_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
    )
    print(f"wrote {len(payload['results'])} timed stage rows")
    print(json_path)
    print(csv_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
