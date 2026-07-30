#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


# Revisions are intentionally pinned. Updating a dataset is an explicit experiment
# change, not something that should happen because a repository's main branch moved.
DATASETS = {
    "fineweb2-multilingual": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "revision": "af9c13333eb981300149d5ca60a8e9d659b276b9",
        "allow_patterns": [
            "README.md",
            "data/arb_Arab/train/000_00000.parquet",
            "data/hin_Deva/train/000_00000.parquet",
            "data/jpn_Jpan/train/000_00000.parquet",
            "data/kor_Hang/train/000_00000.parquet",
            "data/rus_Cyrl/train/000_00000.parquet",
            "data/spa_Latn/train/000_00000.parquet",
        ],
    },
    "github-code-starter": {
        "repo_id": "codeparrot/github-code",
        "revision": "b5661e6b17396364b2bcf8e68977b0d28e1ebd19",
        "allow_patterns": [
            "README.md",
            "data/train-00000-of-01126.parquet",
            "data/train-00001-of-01126.parquet",
            "data/train-00002-of-01126.parquet",
            "data/train-00003-of-01126.parquet",
        ],
    },
    "finemath-3plus": {
        "repo_id": "HuggingFaceTB/finemath",
        "revision": "e92b25a616738fe95dc186b64dfb19f9c8525594",
        "allow_patterns": [
            "README.md",
            "finemath-3plus/train-00000-of-00128.parquet",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pinned datasets for the MiniLLM tokenizer lab.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS))
    return parser.parse_args()


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "download-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema_version": 1, "datasets": {}}

    for name in args.dataset or list(DATASETS):
        spec = DATASETS[name]
        target = output_dir / name
        print(f"downloading {name} -> {target}", flush=True)
        snapshot_download(
            repo_id=spec["repo_id"],
            repo_type="dataset",
            revision=spec["revision"],
            local_dir=target,
            allow_patterns=spec["allow_patterns"],
        )
        files = [
            {"path": str(path.relative_to(target)), "bytes": path.stat().st_size}
            for path in sorted(target.rglob("*"))
            if path.is_file() and ".cache" not in path.parts
        ]
        manifest["datasets"][name] = {
            **spec,
            "path": str(target),
            "files": files,
            "bytes": sum(item["bytes"] for item in files),
        }
        write_manifest(manifest_path, manifest)
        print(json.dumps(manifest["datasets"][name], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
