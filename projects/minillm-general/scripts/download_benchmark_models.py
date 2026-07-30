#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODELS = {
    "pythia-70m-deduped": "EleutherAI/pythia-70m-deduped",
    "gpt2-chinese-102m": "uer/gpt2-chinese-cluecorpussmall",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pinned, similar-size public baselines for MiniLLM evaluation."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", action="append", choices=sorted(MODELS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = args.model or list(MODELS)
    api = HfApi()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema_version": 1, "models": {}}

    for name in selected:
        repo_id = MODELS[name]
        revision = api.model_info(repo_id).sha
        target = output_dir / name
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            local_dir=target,
            allow_patterns=[
                "README.md",
                "LICENSE*",
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "pytorch_model.bin",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.*",
                "merges.txt",
            ],
        )
        files = [
            {"path": str(path.relative_to(target)), "bytes": path.stat().st_size}
            for path in sorted(target.rglob("*"))
            if path.is_file() and ".cache" not in path.parts
        ]
        manifest["models"][name] = {
            "repo_id": repo_id,
            "revision": revision,
            "path": str(target),
            "files": files,
            "bytes": sum(item["bytes"] for item in files),
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
        print(json.dumps(manifest["models"][name], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
