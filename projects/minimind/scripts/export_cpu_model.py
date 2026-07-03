import argparse
import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from model.model_minimind import MiniMindConfig


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def build_qwen_config(lm_config):
    return Qwen3Config(
        vocab_size=lm_config.vocab_size,
        hidden_size=lm_config.hidden_size,
        intermediate_size=lm_config.intermediate_size,
        num_hidden_layers=lm_config.num_hidden_layers,
        num_attention_heads=lm_config.num_attention_heads,
        num_key_value_heads=lm_config.num_key_value_heads,
        head_dim=lm_config.hidden_size // lm_config.num_attention_heads,
        max_position_embeddings=lm_config.max_position_embeddings,
        rms_norm_eps=lm_config.rms_norm_eps,
        rope_theta=lm_config.rope_theta,
        tie_word_embeddings=lm_config.tie_word_embeddings,
        use_sliding_window=False,
        sliding_window=None,
    )


def main():
    parser = argparse.ArgumentParser(description="Export MiniMind torch weights to a CPU-friendly Transformers directory.")
    parser.add_argument("--torch_path", default="../out/full_sft_512.pth", help="Input native .pth weight path.")
    parser.add_argument("--output_dir", default="../artifacts/minimind-cpu-512", help="Output Transformers model directory.")
    parser.add_argument("--tokenizer_path", default="../model", help="Tokenizer directory.")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="float32", help="Use float32 for the widest CPU compatibility.")
    args = parser.parse_args()

    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=False)
    model = Qwen3ForCausalLM(build_qwen_config(lm_config))
    state_dict = torch.load(args.torch_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(DTYPES[args.dtype]).eval()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.tokenizer_path).save_pretrained(args.output_dir)

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Exported {params_m:.2f}M params to {args.output_dir} ({args.dtype}).")


if __name__ == "__main__":
    main()
