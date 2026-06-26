# vLLM / SGLang Local Inference Environment

## Hardware

- GPU: NVIDIA GeForce RTX 4070 SUPER
- VRAM: 12 GB
- Compute capability: 8.9
- Driver: 580.159.03
- CUDA runtime reported by driver: 13.0

## Environments

Two isolated Python environments are used to avoid dependency conflicts:

- vLLM: `/home/undefined/Desktop/ai/.venv-vllm`
- SGLang: `/home/undefined/Desktop/ai/.venv-sglang`

Activate vLLM:

```bash
cd /home/undefined/Desktop/ai
source scripts/use_vllm.sh
```

Activate SGLang:

```bash
cd /home/undefined/Desktop/ai
source scripts/use_sglang.sh
```

## Verified Versions

vLLM environment:

- `torch==2.11.0+cu130`
- `vllm==0.23.0`
- `torch.cuda.is_available() == True`

SGLang environment:

- `torch==2.9.1+cu130`
- `sglang==0.5.9`
- `torch.cuda.is_available() == True`

## Model Size Guidance

This GPU has 12 GB VRAM. For learning, start with small instruction models:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

Full 7B FP16 models are usually too large for this card once KV cache and runtime overhead are included. Use smaller models or quantized variants for 7B-class experiments.

## Quick Checks

```bash
source scripts/use_vllm.sh
python -c "import torch, vllm; print(torch.cuda.is_available(), torch.version.cuda, vllm.__version__)"

deactivate
source scripts/use_sglang.sh
python -c "import torch, sglang; print(torch.cuda.is_available(), torch.version.cuda, sglang.__version__)"
```

## Minimal Server Examples

vLLM OpenAI-compatible server:

```bash
source scripts/use_vllm.sh
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096
```

SGLang server:

```bash
source scripts/use_sglang.sh
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 30000
```
