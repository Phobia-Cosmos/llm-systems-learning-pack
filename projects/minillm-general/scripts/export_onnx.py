"""Export a MiniLLM checkpoint for CPU-oriented ONNX Runtime deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minillm import GPTConfig, MiniGPT


class LogitsOnly(torch.nn.Module):
    def __init__(self, model: MiniGPT) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--example-length", type=int, default=8)
    parser.add_argument(
        "--dynamic-sequence",
        action="store_true",
        help="Experimentally mark the sequence dimension dynamic. The static default is safer for traced causal masks.",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_payload = dict(checkpoint["config"])
    # PyTorch 2.0 has SDPA but not the enable_gqa keyword used by this model.
    config_payload["use_sdpa"] = False
    config = GPTConfig(**config_payload)
    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    wrapper = LogitsOnly(model)
    example = torch.zeros((1, args.example_length), dtype=torch.long)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if args.dynamic_sequence:
        dynamic_axes = {"input_ids": {1: "sequence"}, "logits": {1: "sequence"}}
    torch.onnx.export(
        wrapper,
        (example,),
        str(output),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    print(output)


if __name__ == "__main__":
    main()
