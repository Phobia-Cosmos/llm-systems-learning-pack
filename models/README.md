# Trained MiniLLM Models

The final inference-only MiniLLM checkpoints are published as assets on the [`trained-models-2026-09`](https://github.com/Phobia-Cosmos/llm-systems-learning-pack/releases/tag/trained-models-2026-09) release. They contain model weights and configuration, without optimizer state or training RNG state. Each model also has a matching tokenizer and resolved training configuration in the release assets.

| Model | Parameters | Context | Final step | Training tokens | Weight asset | SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| MiniLLM 90M final | 88,101,120 | 1,024 | 20,000 | 655,360,000 | `minillm-90m-final.pt` | `30ad56dd68c95267d220208eacb3790c8be706a1ba2fc55847823f3f4436adf9` |
| MiniLLM 160M 4K final | 163,615,232 | 4,096 | 211,345 | 10,000,007,168 | `minillm-160m-4k-final.pt` | `f876001962a2e153026dfc65b814755601d826205dd97bea931f4cf569ee7ad0` |
| MiniLLM 160M 8K final | 163,615,232 | 8,192 | 218,975 | 11,000,086,528 | `minillm-160m-8k-final.pt` | `f8c46f1c538f03fc4b3a251bb9d4e701561ab09f2c1ef2f3b8a1298c9f5c861f` |

The 90M final checkpoint reached test loss 2.3059 and perplexity 10.03. The 160M 8K final checkpoint reached 8K test loss 2.3560 and perplexity 10.55 over 131,072 evaluated tokens. Static and dynamic KV-cache outputs matched in the recorded GPU evaluation. Greedy passkey retrieval did not pass, so the 8K artifact should be treated as a long-context pretraining checkpoint rather than a fully instruction-tuned assistant.

Download all files for one model and keep its `.pt`, tokenizer, and resolved configuration together. Verify the weight before loading it:

```bash
sha256sum -c SHA256SUMS
```

The checkpoint payload uses the schema expected by `projects/minillm-general`: `schema_version`, `step`, optional `tokens_processed`, `config`, `model`, and `tokenizer_file`. Load it only with code you trust because PyTorch `.pt` files use pickle serialization.

The full training checkpoints remain in persistent school storage. They were not added to Git history or the release because each contains optimizer and restart state and is substantially larger than the corresponding inference artifact.
