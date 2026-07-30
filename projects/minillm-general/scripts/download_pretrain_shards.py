#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DATASETS = {
    "fineweb2-zh-first": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "allow_patterns": [
            "README.md",
            "data/cmn_Hani/train/000_00000.parquet",
        ],
    },
    "fineweb-edu-en-first": {
        "repo_id": "HuggingFaceFW/fineweb-edu",
        "allow_patterns": [
            "README.md",
            "sample/10BT/000_00000.parquet",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pinned starter shards for bilingual continued pretraining."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = args.dataset or list(DATASETS)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema_version": 1, "datasets": {}}
    api = HfApi()

    for name in selected:
        spec = DATASETS[name]
        revision = api.dataset_info(spec["repo_id"]).sha
        target = output_dir / name
        snapshot_download(
            repo_id=spec["repo_id"],
            repo_type="dataset",
            revision=revision,
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
            "revision": revision,
            "path": str(target),
            "selected_files": spec["allow_patterns"],
            "files": files,
            "bytes": sum(item["bytes"] for item in files),
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
        print(json.dumps(manifest["datasets"][name], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
