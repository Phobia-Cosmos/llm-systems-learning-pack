<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

Install `nano-vllm[flash-attn]` to use an independently installed FlashAttention package. When nano-vLLM shares an environment with a CUDA-enabled vLLM wheel, it can reuse vLLM's bundled FlashAttention extension instead.

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Supported Models

- Qwen3 (`model_type: qwen3`)
- MiniLLM MiniGPT (`model_type: minigpt`), including its character tokenizer and HF-compatible Byte BPE exports

Run the MiniLLM export in this workspace with:

```bash
cd /home/undefined/Desktop/ai/projects/minillm
python export_hf_like.py \
  --checkpoint checkpoints/minillm.pt \
  --out-dir hf_exports/minillm \
  --safe-serialization
python scripts/run_nanovllm_minigpt.py
```

The bundled character model has a vocabulary size of 339, so use `tensor_parallel_size=1` for that checkpoint.

## Adding A Model

Model-specific code lives under `nanovllm/models/`. A model module owns its `PretrainedConfig`, inference model, optional tokenizer loader, and registration:

```python
from nanovllm.models.registry import register_model

@register_model(
    model_type="my_model",
    architectures=("MyModelForCausalLM",),
    config_class=MyModelConfig,
    tokenizer_loader=load_my_tokenizer,  # optional
)
class MyModelForCausalLM(nn.Module):
    ...
```

Built-in modules under `nanovllm.models` are discovered and imported lazily by the registry. External model modules can register themselves before constructing `LLM`. The scheduler, model runner, and request API do not require architecture-specific branches.

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
