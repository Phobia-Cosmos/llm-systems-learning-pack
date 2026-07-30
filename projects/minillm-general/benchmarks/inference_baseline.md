# PyTorch / cuBLAS inference baseline

This benchmark establishes the unoptimized reference for MiniLLM inference operators before adding custom CUDA, CUTLASS, Triton, FlashAttention, or CUDA Graph implementations.

## Measured layouts

The same MiniLLM checkpoint is executed through two layouts:

| Runtime | Prefill hidden states | Decode hidden states | KV cache |
| --- | --- | --- | --- |
| MiniLLM | `[B, T, C]` | `[B, 1, C]` | contiguous `[B, H, S, D]`, extended with `cat` |
| nano-vLLM PyTorch baseline | `[sum_tokens, C]` | `[num_sequences, C]` | paged `[num_blocks, 256, H, D]` |

The nano-vLLM baseline mirrors the engine's flattened token and paged-cache contracts. Its attention uses the repository fallback's `einsum` equations and loops over sequences. The paged store uses `index_copy_` here so it remains a PyTorch baseline; nano-vLLM's normal store path uses a Triton kernel.

## Run

Use the shared CUDA environment from the MiniLLM project directory:

```bash
/home/undefined/Disk/python-envs/vllm/bin/python \
  scripts/benchmark_inference_shapes.py \
  --batch-sizes 1,8 \
  --dtypes float32,float16 \
  --generated-tokens 16 \
  --run-name inference_pytorch_cublas_4070super_20260716
```

For a quick plumbing check:

```bash
/home/undefined/Disk/python-envs/vllm/bin/python \
  scripts/benchmark_inference_shapes.py \
  --batch-sizes 1 \
  --dtypes float32 \
  --warmup 1 \
  --samples 2 \
  --target-sample-ms 0.1 \
  --max-inner-loops 2 \
  --run-name inference_smoke
```

The command writes JSON, CSV, and Markdown under `benchmarks/results/`. JSON is the source of truth and includes every measured tensor's shape, dtype, stride, allocation pointer alignment, timing samples summary, and logical call frequency.

## Timing contract

- CUDA Events measure stream execution time; synchronized wall time additionally exposes Python dispatch and launch/synchronization overhead.
- `F.linear` is the PyTorch cuBLAS/cuBLASLt baseline. QK and probability-value products use `matmul`/batched GEMM as the cuBLAS baseline.
- TF32 is disabled and float32 matmul precision is `highest`, so the FP32 and FP16 rows remain distinct references.
- FlashAttention, Triton attention, CUDA Graph, and `torch.compile` are excluded by design.
- The adaptive inner loop makes sub-10-microsecond operators measurable without changing their input shapes.
- After timing, one PyTorch profiler invocation audits linear, matmul, layout, and KV-cache stages for implicit `clone`, `contiguous`, and `copy_` operations. Profiler overhead is excluded from CUDA Event timing.

For `N` requested output tokens, prefill computes the logits used to choose token 1, followed by `N-1` decode model passes. The report records both calls per model pass and calls for that complete generation convention.

Component timings do not add exactly to the full pass. Residual additions, casts, allocations, framework dispatch, and some metadata work are not separate rows, while full-pass execution contains all of them.
