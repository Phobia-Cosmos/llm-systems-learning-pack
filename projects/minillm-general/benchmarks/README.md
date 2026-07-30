# MiniLLM component benchmark

This benchmark compares one architectural axis at a time:

- position: learned, sinusoidal, RoPE, ALiBi, and NoPE;
- normalization: LayerNorm, RMSNorm, and ScaleNorm with RoPE fixed;
- MLP: dense GELU, SwiGLU, GEGLU, and ReGLU with RoPE + RMSNorm fixed;
- attention: MHA, GQA, and MQA with RoPE + RMSNorm + dense GELU fixed.

The attention suite interprets `--n-head` as the query-head count. MHA uses the same
number of KV heads, MQA uses one, and GQA uses the largest proper divisor (for the
default four query heads, the matrix is `Hkv=4/2/1`). Requesting the attention suite
with a prime query-head count fails explicitly because no distinct, evenly grouped
GQA configuration exists.

Run the reproducible CPU baseline from the MiniLLM project root:

```bash
/home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/benchmark_components.py \
  --run-name components_cpu_100step
```

The default run uses three model seeds and 100 optimizer updates per variant. A quick plumbing check can use one seed and two updates:

```bash
/home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/benchmark_components.py \
  --model-seeds 7 \
  --max-steps 2 \
  --eval-batches 1 \
  --generation-repeats 1 \
  --run-name components_smoke
```

## Fairness contract

The script fits one character tokenizer on the training split only, pre-generates one training schedule, and evaluates every validation target exactly once. Every variant receives the same raw split, batches, optimizer settings, token budget, prompt, and greedy decoding parameters. Random matrices use a stable seed derived from `model seed + semantic parameter name`, so common parameters have identical initial values even when one architecture adds an extra table or matrix.

Schema version 2 JSON records the full resolved config, raw per-seed metrics, paired deltas, code/data/tokenizer/schedule hashes, Git state, device details, gradient clipping, generated token IDs, and KV-cache parity. CSV contains one row per seed and variant, including the serialized resolved config. Markdown contains the three-seed summary. New outputs report parameter count plus KV-cache elements and bytes per token per layer, along with the compression ratio relative to MHA at the same query-head count and head dimension.

## Recorded baseline

The checked-in [components_cpu_100step.md](results/components_cpu_100step.md) is a regression baseline from a one-layer, 32-hidden CPU model on `data/tiny_corpus.txt`. It predates the attention suite and contains 12 comparison rows × 3 seeds. New default runs contain 15 rows × 3 seeds. Equivalent baselines shared by two axes are executed once per seed and reused, so they are not silently weighted twice.

The most important result is that all 36 raw rows produced identical greedy token IDs with and without the teaching KV cache. The loss differences are not a production-model ranking: the corpus has only about one thousand characters, each run uses 100 updates, and the model has about 21K parameters. In particular, the weak fixed-sinusoidal result and the near-tie between learned/RoPE/ALiBi/NoPE describe this tiny optimization regime, not a universal ordering of position methods.

CPU peak Torch memory is left blank because PyTorch does not expose a reliable CPU allocator peak counter. CUDA runs populate `peak_cuda_memory_bytes`. Very short generation timings are noisy; use them only as same-run smoke measurements, not as deployment throughput claims.
