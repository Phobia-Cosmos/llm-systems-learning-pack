from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DATASETS = {
    "ceval": {
        "repo_id": "ceval/ceval-exam",
        "allow_patterns": ["README.md", "*/dev-*.parquet", "*/val-*.parquet"],
    },
    "cmmlu": {
        "repo_id": "haonan-li/cmmlu",
        "allow_patterns": ["README.md", "cmmlu_v1_0_1.zip"],
    },
    "arc": {
        "repo_id": "allenai/ai2_arc",
        "allow_patterns": ["README.md", "*/train-*.parquet", "*/validation-*.parquet"],
    },
    "hellaswag": {
        "repo_id": "Rowan/hellaswag",
        "allow_patterns": ["README.md", "data/validation-*.parquet"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pinned public evaluation dataset snapshots.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = args.dataset or list(DATASETS)
    api = HfApi()
    manifest: dict[str, dict[str, object]] = {"schema_version": 1, "datasets": {}}

    for name in selected:
        spec = DATASETS[name]
        info = api.dataset_info(spec["repo_id"])
        target = output_dir / name
        snapshot_download(
            repo_id=spec["repo_id"],
            repo_type="dataset",
            revision=info.sha,
            local_dir=target,
            allow_patterns=spec["allow_patterns"],
        )
        files = [
            {"path": str(path.relative_to(target)), "bytes": path.stat().st_size}
            for path in sorted(target.rglob("*"))
            if path.is_file() and ".cache" not in path.parts
        ]
        manifest["datasets"][name] = {
            "repo_id": spec["repo_id"],
            "revision": info.sha,
            "files": files,
            "bytes": sum(item["bytes"] for item in files),
        }

    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
