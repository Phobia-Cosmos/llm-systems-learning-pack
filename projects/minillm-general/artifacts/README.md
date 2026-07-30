# MiniLLM Artifacts

Generated model assets live under one parent while retaining their distinct formats:

| Path | Contents |
| --- | --- |
| `checkpoints/` | Flat PyTorch training checkpoints (`.pt`) |
| `hf_exports/` | One Hugging Face-compatible directory bundle per model variant |
| `tokenizers/` | Reusable tokenizer bundles and their local documentation |

Checkpoint and HF export files should not be flattened together. A checkpoint contains training state and embedded MiniLLM metadata; an HF export is a multi-file inference bundle with standard filenames such as `config.json`, `tokenizer.json`, and `model.safetensors`.

Use a unique `--checkpoint-name` when training a variant so every checkpoint remains in `checkpoints/` without overwriting another model.
