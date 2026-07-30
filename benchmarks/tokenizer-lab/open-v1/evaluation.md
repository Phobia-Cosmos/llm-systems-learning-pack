# MiniLLM tokenizer lab

| Tokenizer | Vocab | File MiB | Embedding params @768 | BF16 embedding MiB | Bytes/token | P95 tokens/KiB | `<unk>` | Round-trip failures | Encode MiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| openbpe-32k | 32,768 | 2.36 | 25,165,824 | 48.0 | 3.963 | 351.2 | 0 | 0 | 1.3 |
| openbpe-48k | 49,152 | 3.64 | 37,748,736 | 72.0 | 4.205 | 336.4 | 0 | 0 | 1.3 |
| minillm-current-16k | 16,384 | 1.26 | 12,582,912 | 24.0 | 2.400 | 634.9 | 0 | 0 | 1.3 |
| qwen3-151k | 151,669 | 10.89 | 116,481,792 | 222.2 | 3.922 | 364.8 | 0 | 0 | 1.1 |

## Held-out bytes per token by domain

| Domain | openbpe-32k | openbpe-48k | minillm-current-16k | qwen3-151k |
|---|---:|---:|---:|---:|
| ar-web | 4.500 | 4.861 | 1.646 | 4.520 |
| code-multilingual | 3.243 | 3.397 | 2.146 | 3.830 |
| en-educational | 3.784 | 4.006 | 2.919 | 4.570 |
| es-web | 3.329 | 3.597 | 2.239 | 3.903 |
| hi-web | 4.835 | 4.933 | 2.167 | 2.776 |
| ja-web | 4.341 | 4.681 | 2.191 | 4.153 |
| ko-web | 3.733 | 4.033 | 1.165 | 3.385 |
| math | 3.323 | 3.487 | 2.593 | 3.730 |
| ru-web | 4.965 | 5.447 | 1.896 | 5.381 |
| zh-curated | 4.596 | 4.906 | 4.466 | 4.327 |
| zh-web | 3.896 | 4.125 | 3.421 | 3.921 |
