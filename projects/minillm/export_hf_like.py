from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from minillm import GPTConfig, MiniGPT
from minillm.tokenizer_registry import tokenizer_from_checkpoint


def pack_separate_qkv_for_export(model: MiniGPT, config: GPTConfig) -> tuple[MiniGPT, GPTConfig]:
    """Convert teaching q/k/v modules into the packed inference checkpoint layout."""

    if config.fused_qkv:
        return model, config

    export_config = replace(config, fused_qkv=True)
    export_model = MiniGPT(export_config)
    source_state = model.state_dict()
    packed_state: dict[str, torch.Tensor] = {}
    skip_fragments = (".attn.q_proj.", ".attn.k_proj.", ".attn.v_proj.")
    for name, value in source_state.items():
        if not any(fragment in name for fragment in skip_fragments):
            packed_state[name] = value
    for layer_index in range(config.n_layer):
        prefix = f"blocks.{layer_index}.attn"
        packed_state[f"{prefix}.c_attn.weight"] = torch.cat(
            [
                source_state[f"{prefix}.q_proj.weight"],
                source_state[f"{prefix}.k_proj.weight"],
                source_state[f"{prefix}.v_proj.weight"],
            ],
            dim=0,
        )
        if config.bias:
            packed_state[f"{prefix}.c_attn.bias"] = torch.cat(
                [
                    source_state[f"{prefix}.q_proj.bias"],
                    source_state[f"{prefix}.k_proj.bias"],
                    source_state[f"{prefix}.v_proj.bias"],
                ],
                dim=0,
            )
    export_model.load_state_dict(packed_state)
    export_model.eval()
    return export_model, export_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a MiniLLM checkpoint to an educational HF-like directory.")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/minillm.pt")
    parser.add_argument("--out-dir", default="artifacts/hf_exports/minillm")
    parser.add_argument("--safe-serialization", action="store_true", help="Write model.safetensors; requires safetensors.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = GPTConfig(**checkpoint["config"])
    tokenizer = tokenizer_from_checkpoint(checkpoint)
    tokenizer_type = checkpoint.get("tokenizer_type", checkpoint.get("tokenizer", {}).get("type", "char"))

    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model, config = pack_separate_qkv_for_export(model, config)

    config_json = {
        "model_type": "minigpt",
        "architectures": ["MiniGPTForCausalLM"],
        "vocab_size": config.vocab_size,
        "block_size": config.block_size,
        "max_position_embeddings": config.block_size,
        "n_layer": config.n_layer,
        "num_hidden_layers": config.n_layer,
        "n_head": config.n_head,
        "num_attention_heads": config.n_head,
        "n_embd": config.n_embd,
        "hidden_size": config.n_embd,
        "dropout": config.dropout,
        "bias": config.bias,
        "fused_qkv": config.fused_qkv,
        "position_encoding": config.position_encoding,
        "rope_theta": config.rope_theta,
        "sinusoidal_theta": config.sinusoidal_theta,
        "norm_type": config.norm_type,
        "norm_eps": config.norm_eps,
        "mlp_type": config.mlp_type,
        "activation": config.activation,
        "hidden_act": config.activation,
        "intermediate_size": config.intermediate_size,
        "tie_word_embeddings": True,
        "torch_dtype": "float16",
    }
    (out_dir / "config.json").write_text(json.dumps(config_json, ensure_ascii=False, indent=2) + "\n")

    tokenizer.save_pretrained(out_dir, model_max_length=config.block_size)
    (out_dir / "generation_config.json").write_text(
        json.dumps({"max_new_tokens": 160, "temperature": 0.8, "top_k": 40}, ensure_ascii=False, indent=2) + "\n"
    )

    if args.safe_serialization:
        try:
            from safetensors.torch import save_model
        except ImportError as exc:
            raise SystemExit("safetensors is required for --safe-serialization. Use .venv-sglang or install safetensors.") from exc
        save_model(model, out_dir / "model.safetensors")
    else:
        torch.save(model.state_dict(), out_dir / "pytorch_model.bin")

    (out_dir / "README.md").write_text(
        "# MiniLLM HF-like Export\n\n"
        "This directory is useful for learning the Hugging Face model layout. "
        f"Tokenizer type: `{tokenizer_type}`.\n\n"
        "It can be loaded by the sibling nano-vLLM and vLLM MiniGPT backends. "
        "Upstream SGLang still needs its own MiniGPT implementation and registration.\n"
    )
    print(f"exported to {out_dir}")


if __name__ == "__main__":
    main()
