# MiniLLM component benchmark

This is a controlled teaching benchmark, not a claim about large-model quality.

## Fairness contract

- Data: `data/tiny_corpus.txt`
- Updates per variant: 100
- Batch/sequence: 8 × 32
- Model: 1 layer(s), 4 head(s), hidden size 32
- Seeds: model=(1337, 1338, 1339), data=2026
- Every variant receives the same pre-generated batches; common semantic parameter names use identical initial values.
- CPU peak Torch memory is reported as blank because PyTorch has no reliable allocator peak counter for CPU.

## position

| variant | seeds | params | final train loss | final val loss | paired Δ vs baseline | val ppl | train tok/s | KV tok/s | example ids | cache parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| learned | 3 | 21,920 | 2.2981 ± 0.0197 | 3.3077 ± 0.0787 | +0.0000 ± 0.0000 | 27.38 | 202222.3 | 11397.8 | `[2, 13, 13, 13, 13, 13, 13, 13]` | True |
| sinusoidal | 3 | 20,896 | 4.8471 ± 0.0008 | 4.6565 ± 0.0113 | +1.3488 ± 0.0781 | 105.27 | 205205.4 | 11796.6 | `[2, 2, 2, 2, 2, 2, 2, 2]` | True |
| rope | 3 | 20,896 | 2.1701 ± 0.0070 | 3.2749 ± 0.0999 | -0.0328 ± 0.0799 | 26.53 | 180510.0 | 8345.2 | `[42, 46, 2, 13, 13, 13, 13, 13]` | True |
| alibi | 3 | 20,896 | 2.1641 ± 0.0065 | 3.2919 ± 0.1013 | -0.0159 ± 0.0716 | 26.99 | 203665.7 | 10733.3 | `[2, 13, 13, 13, 13, 13, 13, 13]` | True |
| none | 3 | 20,896 | 2.1847 ± 0.0041 | 3.2840 ± 0.1030 | -0.0237 ± 0.0721 | 26.78 | 207151.7 | 12741.8 | `[2, 13, 13, 13, 13, 13, 13, 13]` | True |

## norm

| variant | seeds | params | final train loss | final val loss | paired Δ vs baseline | val ppl | train tok/s | KV tok/s | example ids | cache parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| layernorm | 3 | 20,896 | 2.1701 ± 0.0070 | 3.2749 ± 0.0999 | +0.0000 ± 0.0000 | 26.53 | 180510.0 | 8345.2 | `[42, 46, 2, 13, 13, 13, 13, 13]` | True |
| rmsnorm | 3 | 20,800 | 2.1993 ± 0.0142 | 3.3348 ± 0.1177 | +0.0598 ± 0.0396 | 28.20 | 173238.3 | 6970.3 | `[42, 46, 156, 118, 168, 107, 107, 107]` | True |
| scalenorm | 3 | 20,707 | 2.4987 ± 0.0164 | 3.4022 ± 0.0889 | +0.1273 ± 0.0382 | 30.11 | 176465.8 | 7785.7 | `[42, 46, 46, 2, 36, 36, 36, 36]` | True |

## mlp

| variant | seeds | params | final train loss | final val loss | paired Δ vs baseline | val ppl | train tok/s | KV tok/s | example ids | cache parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| dense | 3 | 20,800 | 2.1993 ± 0.0142 | 3.3348 ± 0.1177 | +0.0000 ± 0.0000 | 28.20 | 173238.3 | 6970.3 | `[42, 46, 156, 118, 168, 107, 107, 107]` | True |
| swiglu | 3 | 21,104 | 2.1650 ± 0.0173 | 3.4512 ± 0.0250 | +0.1164 ± 0.1195 | 31.54 | 174749.1 | 7512.5 | `[2, 36, 32, 28, 23, 31, 31, 31]` | True |
| geglu | 3 | 21,104 | 2.1520 ± 0.0183 | 3.4243 ± 0.0405 | +0.0895 ± 0.1300 | 30.72 | 171168.9 | 7116.4 | `[2, 36, 32, 28, 23, 31, 31, 31]` | True |
| reglu | 3 | 21,104 | 2.1133 ± 0.0066 | 3.3239 ± 0.0326 | -0.0109 ± 0.0984 | 27.78 | 176426.1 | 7552.9 | `[42, 46, 2, 13, 13, 13, 13, 13]` | True |

## Interpretation limits

Loss and throughput from a tiny corpus and tiny CPU model are useful for regression and learning, but they do not predict the ranking of production-scale LLMs. Timing should only be compared within the same machine/run, and generation text is recorded as a regression artifact rather than a quality score.
