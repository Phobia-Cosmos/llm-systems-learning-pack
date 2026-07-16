# MiniLLM / nano-vLLM PyTorch-cuBLAS inference baseline

Generated: `2026-07-16T16:00:44+0800`

## Scope and method

This is an eager PyTorch baseline on the real MiniLLM checkpoint and the tensor layouts used by MiniLLM and nano-vLLM. Linear layers use `F.linear` (cuBLAS/cuBLASLt); MiniLLM QK/PV use `matmul`, while the nano-vLLM fallback uses its real `einsum` equations; both lower to batched GEMM/cuBLAS for these shapes. FlashAttention, Triton attention, CUDA Graph, and `torch.compile` are intentionally excluded so later optimized kernels have a stable reference.

Timing uses 10 warmups and 30 CUDA Event samples. The adaptive inner loop targets 3.0 ms per sample. Synchronized wall time includes Python dispatch and launch/synchronization overhead; GPU Event time covers work on the CUDA stream.

For `N` generated tokens, prefill produces token 1 and the model executes `N-1` decode passes. Component contribution is `median time for one measured stage invocation x logical calls per pass`.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 SUPER |
| Compute capability | [8, 9] |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 |
| NVIDIA driver | 580.159.03 |
| BLAS | _BlasBackend.Cublas |
| TF32 | matmul=False, cuDNN=False |
| Checkpoint SHA-256 | `85f69b910214c78537c510620f084454e26dba8bbea66506cc94664ac5f31e57` |
| Git | `f2b28dd4c97f6c81427b136de3b75800449e1abd`; dirty=True |

## Model and workloads

Model: `L=2, C=128, H=4, D=32, V=512, block_size=128`. Prompt token count: `20`. nano-vLLM KV page size: `256`.
The decode row measures the first incremental pass: one new token is processed while attention reads the prompt plus that token. The call count for a 16-token generation is exact, but later decode passes have progressively longer KV lengths and are not assigned the first-step latency.

| Workload | MiniLLM prefill | nano prefill | MiniLLM decode | nano decode | dtype |
| --- | --- | --- | --- | --- | --- |
| b1_t20_float32 | `[1, 20, 128]` | `[20, 128]` | `[1, 1, 128]` | `[1, 128]` | float32 |
| b8_t20_float32 | `[8, 20, 128]` | `[160, 128]` | `[8, 1, 128]` | `[8, 128]` | float32 |
| b1_t20_float16 | `[1, 20, 128]` | `[20, 128]` | `[1, 1, 128]` | `[1, 128]` | float16 |
| b8_t20_float16 | `[8, 20, 128]` | `[160, 128]` | `[8, 1, 128]` | `[8, 128]` | float16 |

## Full-pass latency

| Workload | Runtime | Phase | GPU median (ms) | p90 (ms) | Wall median (ms) | Throughput token/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| b1_t20_float32 | minillm | prefill | 0.510854 | 0.515524 | 0.512131 | 39150.1 |
| b1_t20_float32 | minillm | decode | 0.473400 | 0.479240 | 0.474422 | 2112.4 |
| b1_t20_float32 | nano_vllm_torch | prefill | 0.591250 | 0.643344 | 0.596808 | 33826.6 |
| b1_t20_float32 | nano_vllm_torch | decode | 0.511132 | 0.514576 | 0.512211 | 1956.4 |
| b8_t20_float32 | minillm | prefill | 0.494846 | 0.498136 | 0.495945 | 323332.9 |
| b8_t20_float32 | minillm | decode | 0.465878 | 0.473088 | 0.466918 | 17171.9 |
| b8_t20_float32 | nano_vllm_torch | prefill | 1.337036 | 1.339832 | 1.339249 | 119667.7 |
| b8_t20_float32 | nano_vllm_torch | decode | 1.572184 | 1.608112 | 1.576319 | 5088.5 |
| b1_t20_float16 | minillm | prefill | 0.528280 | 0.554028 | 0.529453 | 37858.7 |
| b1_t20_float16 | minillm | decode | 0.529224 | 0.548968 | 0.530346 | 1889.6 |
| b1_t20_float16 | nano_vllm_torch | prefill | 0.616160 | 0.623720 | 0.617356 | 32459.1 |
| b1_t20_float16 | nano_vllm_torch | decode | 0.581024 | 0.591008 | 0.582711 | 1721.1 |
| b8_t20_float16 | minillm | prefill | 0.554552 | 0.571804 | 0.555648 | 288521.2 |
| b8_t20_float16 | minillm | decode | 0.526002 | 0.530724 | 0.527164 | 15209.1 |
| b8_t20_float16 | nano_vllm_torch | prefill | 1.599448 | 1.619664 | 1.603964 | 100034.5 |
| b8_t20_float16 | nano_vllm_torch | decode | 1.811648 | 1.846160 | 1.816231 | 4415.9 |

### Scaling observations

`B=1 -> B=8 throughput scaling` is `8 x latency(B=1) / latency(B=8)`. An ideal eightfold throughput increase is 8.0x. `FP16 / FP32 latency` below 1.0 means FP16 is faster.

| Runtime | Phase | FP32 B1->B8 throughput | FP16 B1->B8 throughput | B1 FP16/FP32 latency | B8 FP16/FP32 latency |
| --- | --- | ---: | ---: | ---: | ---: |
| minillm | prefill | 8.259x | 7.621x | 1.034x | 1.121x |
| minillm | decode | 8.129x | 8.049x | 1.118x | 1.129x |
| nano_vllm_torch | prefill | 3.538x | 3.082x | 1.042x | 1.196x |
| nano_vllm_torch | decode | 2.601x | 2.566x | 1.137x | 1.152x |

## Component hotspots

The ranking multiplies isolated median stage latency by the logical calls in one model pass. It identifies optimization candidates; it is not a claim that these isolated rows sum exactly to full-model latency.

| Workload | Runtime/phase | Top 1 | Top 2 | Top 3 |
| --- | --- | --- | --- | --- |
| b1_t20_float32 | minillm/prefill | rope (174.4 us/pass) | causal_mask (23.4 us/pass) | qk_matmul (19.6 us/pass) |
| b1_t20_float32 | minillm/decode | rope (163.3 us/pass) | causal_mask (23.0 us/pass) | qk_matmul (19.6 us/pass) |
| b1_t20_float32 | nano_vllm_torch/prefill | rope (137.9 us/pass) | qk_matmul (24.9 us/pass) | causal_mask (22.8 us/pass) |
| b1_t20_float32 | nano_vllm_torch/decode | rope (133.9 us/pass) | nano_paged_kv_gather (33.1 us/pass) | qk_matmul (24.7 us/pass) |
| b8_t20_float32 | minillm/prefill | rope (174.3 us/pass) | causal_mask (23.4 us/pass) | qk_matmul (19.5 us/pass) |
| b8_t20_float32 | minillm/decode | rope (161.4 us/pass) | causal_mask (22.7 us/pass) | qk_matmul (19.7 us/pass) |
| b8_t20_float32 | nano_vllm_torch/prefill | qk_matmul (199.2 us/pass) | causal_mask (180.3 us/pass) | rope (137.0 us/pass) |
| b8_t20_float32 | nano_vllm_torch/decode | nano_paged_kv_gather (262.7 us/pass) | qk_matmul (196.8 us/pass) | causal_mask (176.5 us/pass) |
| b1_t20_float16 | minillm/prefill | rope (208.9 us/pass) | causal_mask (23.4 us/pass) | qk_matmul (20.6 us/pass) |
| b1_t20_float16 | minillm/decode | rope (195.3 us/pass) | causal_mask (22.8 us/pass) | qk_matmul (19.6 us/pass) |
| b1_t20_float16 | nano_vllm_torch/prefill | rope (177.9 us/pass) | qk_matmul (34.9 us/pass) | causal_mask (23.3 us/pass) |
| b1_t20_float16 | nano_vllm_torch/decode | rope (164.2 us/pass) | nano_paged_kv_gather (33.6 us/pass) | qk_matmul (33.0 us/pass) |
| b8_t20_float16 | minillm/prefill | rope (211.6 us/pass) | causal_mask (23.5 us/pass) | qk_matmul (20.8 us/pass) |
| b8_t20_float16 | minillm/decode | rope (198.7 us/pass) | causal_mask (23.4 us/pass) | qk_matmul (20.4 us/pass) |
| b8_t20_float16 | nano_vllm_torch/prefill | qk_matmul (277.3 us/pass) | causal_mask (185.7 us/pass) | rope (172.0 us/pass) |
| b8_t20_float16 | nano_vllm_torch/decode | nano_paged_kv_gather (266.6 us/pass) | qk_matmul (265.7 us/pass) | causal_mask (181.4 us/pass) |

### b1_t20_float32: minillm prefill

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 87.213 | 90.745 | 2 | 2 | 174.425 | `q_raw=[1, 4, 20, 32]; k_raw=[1, 4, 20, 32]; positions=[20]` |
| causal_mask | 11.684 | 12.523 | 2 | 2 | 23.368 | `scores=[1, 4, 20, 20]; mask=[1, 1, 20, 20]` |
| qk_matmul | 9.807 | 10.819 | 2 | 2 | 19.615 | `query=[1, 4, 20, 32]; key=[1, 4, 20, 32]` |
| mlp_fc1 | 9.564 | 9.592 | 2 | 2 | 19.128 | `hidden_states=[1, 20, 128]; weight=[512, 128]` |
| mlp_fc2 | 9.345 | 9.460 | 2 | 2 | 18.691 | `hidden_states=[1, 20, 512]; weight=[128, 512]` |
| output_projection | 8.739 | 8.803 | 2 | 2 | 17.478 | `hidden_states=[1, 20, 128]; weight=[128, 128]` |
| mlp_norm | 7.266 | 7.300 | 2 | 2 | 14.533 | `hidden_states=[1, 20, 128]` |
| qkv_linear | 6.825 | 6.846 | 2 | 2 | 13.650 | `hidden_states=[1, 20, 128]; weight=[384, 128]` |
| attention_value_matmul | 5.420 | 5.496 | 2 | 2 | 10.840 | `probabilities=[1, 4, 20, 20]; value=[1, 4, 20, 32]` |
| head_merge | 5.150 | 5.172 | 2 | 2 | 10.300 | `attention_values=[1, 4, 20, 32]` |
| attention_norm | 5.052 | 5.063 | 2 | 2 | 10.103 | `hidden_states=[1, 20, 128]` |
| lm_head | 7.488 | 7.521 | 1 | 1 | 7.488 | `hidden_states=[1, 20, 128]; weight=[512, 128]` |
| softmax | 3.698 | 3.719 | 2 | 2 | 7.395 | `scores=[1, 4, 20, 20]` |
| final_norm | 7.324 | 7.362 | 1 | 1 | 7.324 | `hidden_states=[1, 20, 128]` |
| gelu | 3.395 | 3.535 | 2 | 2 | 6.789 | `hidden_states=[1, 20, 512]` |
| embedding | 5.238 | 5.276 | 1 | 1 | 5.238 | `input_ids=[1, 20]; weight=[512, 128]` |

### b1_t20_float32: minillm decode

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 81.653 | 81.799 | 2 | 30 | 163.305 | `q_raw=[1, 4, 1, 32]; k_raw=[1, 4, 1, 32]; positions=[1]` |
| causal_mask | 11.500 | 11.560 | 2 | 30 | 23.001 | `scores=[1, 4, 1, 21]; mask=[1, 1, 1, 21]` |
| qk_matmul | 9.783 | 9.800 | 2 | 30 | 19.565 | `query=[1, 4, 1, 32]; key=[1, 4, 21, 32]` |
| native_kv_concat | 8.024 | 8.056 | 2 | 30 | 16.048 | `past_key=[1, 4, 20, 32]; new_key=[1, 4, 1, 32]; past_value=[1, 4, 20, 32]; new_value=[1, 4, 1, 32]` |
| mlp_fc2 | 7.810 | 7.835 | 2 | 30 | 15.621 | `hidden_states=[1, 1, 512]; weight=[128, 512]` |
| mlp_fc1 | 7.809 | 7.842 | 2 | 30 | 15.619 | `hidden_states=[1, 1, 128]; weight=[512, 128]` |
| attention_norm | 7.218 | 7.235 | 2 | 30 | 14.437 | `hidden_states=[1, 1, 128]` |
| mlp_norm | 7.179 | 7.208 | 2 | 30 | 14.357 | `hidden_states=[1, 1, 128]` |
| qkv_linear | 7.172 | 7.191 | 2 | 30 | 14.344 | `hidden_states=[1, 1, 128]; weight=[384, 128]` |
| output_projection | 7.058 | 7.089 | 2 | 30 | 14.116 | `hidden_states=[1, 1, 128]; weight=[128, 128]` |
| attention_value_matmul | 5.162 | 5.184 | 2 | 30 | 10.325 | `probabilities=[1, 4, 1, 21]; value=[1, 4, 21, 32]` |
| softmax | 3.745 | 3.759 | 2 | 30 | 7.489 | `scores=[1, 4, 1, 21]` |
| final_norm | 7.260 | 7.277 | 1 | 15 | 7.260 | `hidden_states=[1, 1, 128]` |
| gelu | 3.389 | 3.400 | 2 | 30 | 6.778 | `hidden_states=[1, 1, 512]` |
| embedding | 5.816 | 5.837 | 1 | 15 | 5.816 | `input_ids=[1, 1]; weight=[512, 128]` |
| lm_head | 5.547 | 5.557 | 1 | 15 | 5.547 | `hidden_states=[1, 1, 128]; weight=[512, 128]` |
| head_merge | 0.882 | 0.885 | 2 | 30 | 1.763 | `attention_values=[1, 4, 1, 32]` |

### b1_t20_float32: nano_vllm_torch prefill

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 68.952 | 69.104 | 2 | 2 | 137.904 | `q_raw=[20, 4, 32]; k_raw=[20, 4, 32]; positions=[20]` |
| qk_matmul | 12.448 | 12.532 | 2 | 2 | 24.897 | `query=[20, 4, 32]; key=[20, 4, 32]` |
| causal_mask | 11.385 | 11.475 | 2 | 2 | 22.770 | `scores=[4, 20, 20]; mask=[1, 20, 20]` |
| mlp_fc1 | 9.411 | 9.622 | 2 | 2 | 18.822 | `hidden_states=[20, 128]; weight=[512, 128]` |
| mlp_fc2 | 9.184 | 9.332 | 2 | 2 | 18.369 | `hidden_states=[20, 512]; weight=[128, 512]` |
| qkv_linear | 8.786 | 8.798 | 2 | 2 | 17.572 | `hidden_states=[20, 128]; weight=[384, 128]` |
| output_projection | 8.572 | 9.849 | 2 | 2 | 17.144 | `hidden_states=[20, 128]; weight=[128, 128]` |
| attention_value_matmul | 8.116 | 8.148 | 2 | 2 | 16.231 | `probabilities=[4, 20, 20]; value=[20, 4, 32]` |
| nano_paged_kv_store | 7.809 | 7.811 | 2 | 2 | 15.618 | `key=[20, 4, 32]; value=[20, 4, 32]; key_cache=[1, 256, 4, 32]; slot_mapping=[20]` |
| attention_norm | 7.270 | 7.281 | 2 | 2 | 14.539 | `hidden_states=[20, 128]` |
| mlp_norm | 7.257 | 7.308 | 2 | 2 | 14.514 | `hidden_states=[20, 128]` |
| attention_output_concat | 4.584 | 4.596 | 2 | 2 | 9.168 | `sequence_0=[20, 4, 32]` |
| softmax | 4.117 | 4.138 | 2 | 2 | 8.234 | `scores=[4, 20, 20]` |
| final_norm | 7.332 | 7.610 | 1 | 1 | 7.332 | `hidden_states=[20, 128]` |
| embedding | 6.993 | 7.018 | 1 | 1 | 6.993 | `input_ids=[20]; weight=[512, 128]` |
| gelu | 3.391 | 3.517 | 2 | 2 | 6.781 | `hidden_states=[20, 512]` |
| lm_head | 5.304 | 5.553 | 1 | 1 | 5.304 | `hidden_states=[1, 128]; weight=[512, 128]` |
| head_merge | 0.391 | 0.395 | 2 | 2 | 0.783 | `attention_values=[20, 4, 32]` |

### b1_t20_float32: nano_vllm_torch decode

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 66.966 | 72.590 | 2 | 30 | 133.932 | `q_raw=[1, 4, 32]; k_raw=[1, 4, 32]; positions=[1]` |
| nano_paged_kv_gather | 16.544 | 16.831 | 2 | 30 | 33.088 | `key_cache=[1, 256, 4, 32]; value_cache=[1, 256, 4, 32]; block_table=[1]` |
| qk_matmul | 12.352 | 12.815 | 2 | 30 | 24.703 | `query=[1, 4, 32]; key=[21, 4, 32]` |
| causal_mask | 11.218 | 11.378 | 2 | 30 | 22.435 | `scores=[4, 1, 21]; mask=[1, 1, 21]` |
| attention_value_matmul | 8.001 | 8.283 | 2 | 30 | 16.001 | `probabilities=[4, 1, 21]; value=[21, 4, 32]` |
| nano_paged_kv_store | 7.879 | 7.982 | 2 | 30 | 15.758 | `key=[1, 4, 32]; value=[1, 4, 32]; key_cache=[1, 256, 4, 32]; slot_mapping=[1]` |
| mlp_fc2 | 7.567 | 7.619 | 2 | 30 | 15.134 | `hidden_states=[1, 512]; weight=[128, 512]` |
| mlp_fc1 | 7.495 | 7.594 | 2 | 30 | 14.991 | `hidden_states=[1, 128]; weight=[512, 128]` |
| attention_norm | 7.343 | 7.823 | 2 | 30 | 14.685 | `hidden_states=[1, 128]` |
| qkv_linear | 7.199 | 7.785 | 2 | 30 | 14.399 | `hidden_states=[1, 128]; weight=[384, 128]` |
| mlp_norm | 7.184 | 7.348 | 2 | 30 | 14.368 | `hidden_states=[1, 128]` |
| output_projection | 6.930 | 7.070 | 2 | 30 | 13.861 | `hidden_states=[1, 128]; weight=[128, 128]` |
| attention_output_concat | 4.385 | 4.473 | 2 | 30 | 8.769 | `sequence_0=[1, 4, 32]` |
| softmax | 4.085 | 4.189 | 2 | 30 | 8.170 | `scores=[4, 1, 21]` |
| final_norm | 7.229 | 7.253 | 1 | 15 | 7.229 | `hidden_states=[1, 128]` |
| gelu | 3.379 | 3.420 | 2 | 30 | 6.757 | `hidden_states=[1, 512]` |
| embedding | 5.651 | 6.044 | 1 | 15 | 5.651 | `input_ids=[1]; weight=[512, 128]` |
| lm_head | 5.249 | 5.267 | 1 | 15 | 5.249 | `hidden_states=[1, 128]; weight=[512, 128]` |
| head_merge | 0.386 | 0.392 | 2 | 30 | 0.771 | `attention_values=[1, 4, 32]` |

### b8_t20_float32: minillm prefill

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 87.128 | 88.169 | 2 | 2 | 174.256 | `q_raw=[8, 4, 20, 32]; k_raw=[8, 4, 20, 32]; positions=[20]` |
| causal_mask | 11.695 | 11.737 | 2 | 2 | 23.389 | `scores=[8, 4, 20, 20]; mask=[1, 1, 20, 20]` |
| qk_matmul | 9.759 | 9.786 | 2 | 2 | 19.518 | `query=[8, 4, 20, 32]; key=[8, 4, 20, 32]` |
| mlp_fc2 | 9.401 | 9.493 | 2 | 2 | 18.801 | `hidden_states=[8, 20, 512]; weight=[128, 512]` |
| attention_value_matmul | 9.327 | 9.382 | 2 | 2 | 18.654 | `probabilities=[8, 4, 20, 20]; value=[8, 4, 20, 32]` |
| mlp_fc1 | 7.821 | 7.872 | 2 | 2 | 15.641 | `hidden_states=[8, 20, 128]; weight=[512, 128]` |
| qkv_linear | 7.220 | 7.247 | 2 | 2 | 14.439 | `hidden_states=[8, 20, 128]; weight=[384, 128]` |
| attention_norm | 7.160 | 7.226 | 2 | 2 | 14.321 | `hidden_states=[8, 20, 128]` |
| mlp_norm | 7.094 | 7.250 | 2 | 2 | 14.188 | `hidden_states=[8, 20, 128]` |
| output_projection | 6.949 | 6.989 | 2 | 2 | 13.898 | `hidden_states=[8, 20, 128]; weight=[128, 128]` |
| head_merge | 5.106 | 5.135 | 2 | 2 | 10.212 | `attention_values=[8, 4, 20, 32]` |
| embedding | 7.303 | 7.315 | 1 | 1 | 7.303 | `input_ids=[8, 20]; weight=[512, 128]` |
| softmax | 3.597 | 3.619 | 2 | 2 | 7.193 | `scores=[8, 4, 20, 20]` |
| final_norm | 7.153 | 7.287 | 1 | 1 | 7.153 | `hidden_states=[8, 20, 128]` |
| gelu | 3.288 | 3.347 | 2 | 2 | 6.576 | `hidden_states=[8, 20, 512]` |
| lm_head | 5.618 | 5.689 | 1 | 1 | 5.618 | `hidden_states=[8, 20, 128]; weight=[512, 128]` |

### b8_t20_float32: minillm decode

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 80.695 | 81.647 | 2 | 30 | 161.390 | `q_raw=[8, 4, 1, 32]; k_raw=[8, 4, 1, 32]; positions=[1]` |
| causal_mask | 11.354 | 11.640 | 2 | 30 | 22.708 | `scores=[8, 4, 1, 21]; mask=[1, 1, 1, 21]` |
| qk_matmul | 9.862 | 9.921 | 2 | 30 | 19.723 | `query=[8, 4, 1, 32]; key=[8, 4, 21, 32]` |
| native_kv_concat | 8.000 | 8.382 | 2 | 30 | 15.999 | `past_key=[8, 4, 20, 32]; new_key=[8, 4, 1, 32]; past_value=[8, 4, 20, 32]; new_value=[8, 4, 1, 32]` |
| mlp_fc1 | 7.807 | 7.848 | 2 | 30 | 15.615 | `hidden_states=[8, 1, 128]; weight=[512, 128]` |
| mlp_fc2 | 7.781 | 7.825 | 2 | 30 | 15.562 | `hidden_states=[8, 1, 512]; weight=[128, 512]` |
| attention_norm | 7.193 | 7.208 | 2 | 30 | 14.386 | `hidden_states=[8, 1, 128]` |
| qkv_linear | 7.089 | 7.120 | 2 | 30 | 14.179 | `hidden_states=[8, 1, 128]; weight=[384, 128]` |
| mlp_norm | 7.076 | 7.235 | 2 | 30 | 14.151 | `hidden_states=[8, 1, 128]` |
| output_projection | 7.016 | 7.049 | 2 | 30 | 14.032 | `hidden_states=[8, 1, 128]; weight=[128, 128]` |
| attention_value_matmul | 5.259 | 5.266 | 2 | 30 | 10.518 | `probabilities=[8, 4, 1, 21]; value=[8, 4, 21, 32]` |
| final_norm | 7.178 | 7.220 | 1 | 15 | 7.178 | `hidden_states=[8, 1, 128]` |
| softmax | 3.538 | 4.160 | 2 | 30 | 7.076 | `scores=[8, 4, 1, 21]` |
| gelu | 3.370 | 3.394 | 2 | 30 | 6.739 | `hidden_states=[8, 1, 512]` |
| embedding | 5.853 | 5.873 | 1 | 15 | 5.853 | `input_ids=[8, 1]; weight=[512, 128]` |
| lm_head | 5.506 | 5.543 | 1 | 15 | 5.506 | `hidden_states=[8, 1, 128]; weight=[512, 128]` |
| head_merge | 0.878 | 0.881 | 2 | 30 | 1.755 | `attention_values=[8, 4, 1, 32]` |

### b8_t20_float32: nano_vllm_torch prefill

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| qk_matmul | 12.449 | 12.881 | 16 | 16 | 199.190 | `query=[20, 4, 32]; key=[20, 4, 32]` |
| causal_mask | 11.266 | 11.366 | 16 | 16 | 180.253 | `scores=[4, 20, 20]; mask=[1, 20, 20]` |
| rope | 68.509 | 68.848 | 2 | 2 | 137.018 | `q_raw=[160, 4, 32]; k_raw=[160, 4, 32]; positions=[160]` |
| attention_value_matmul | 8.031 | 8.095 | 16 | 16 | 128.497 | `probabilities=[4, 20, 20]; value=[20, 4, 32]` |
| softmax | 4.146 | 4.223 | 16 | 16 | 66.342 | `scores=[4, 20, 20]` |
| mlp_fc2 | 9.217 | 9.276 | 2 | 2 | 18.433 | `hidden_states=[160, 512]; weight=[128, 512]` |
| nano_paged_kv_store | 7.807 | 7.955 | 2 | 2 | 15.614 | `key=[160, 4, 32]; value=[160, 4, 32]; key_cache=[8, 256, 4, 32]; slot_mapping=[160]` |
| mlp_fc1 | 7.691 | 7.856 | 2 | 2 | 15.382 | `hidden_states=[160, 128]; weight=[512, 128]` |
| mlp_norm | 7.108 | 7.251 | 2 | 2 | 14.216 | `hidden_states=[160, 128]` |
| attention_norm | 7.094 | 7.161 | 2 | 2 | 14.188 | `hidden_states=[160, 128]` |
| qkv_linear | 7.077 | 7.132 | 2 | 2 | 14.154 | `hidden_states=[160, 128]; weight=[384, 128]` |
| output_projection | 6.832 | 6.913 | 2 | 2 | 13.664 | `hidden_states=[160, 128]; weight=[128, 128]` |
| attention_output_concat | 4.374 | 4.433 | 2 | 2 | 8.748 | `sequence_0=[20, 4, 32]; sequence_1=[20, 4, 32]; sequence_2=[20, 4, 32]; sequence_3=[20, 4, 32]; sequence_4=[20, 4, 32]; sequence_5=[20, 4, 32]; sequence_6=[20, 4, 32]; sequence_7=[20, 4, 32]` |
| final_norm | 7.112 | 7.126 | 1 | 1 | 7.112 | `hidden_states=[160, 128]` |
| embedding | 7.009 | 7.042 | 1 | 1 | 7.009 | `input_ids=[160]; weight=[512, 128]` |
| gelu | 3.290 | 3.389 | 2 | 2 | 6.580 | `hidden_states=[160, 512]` |
| lm_head | 5.220 | 5.224 | 1 | 1 | 5.220 | `hidden_states=[8, 128]; weight=[512, 128]` |
| head_merge | 0.388 | 0.390 | 2 | 2 | 0.776 | `attention_values=[160, 4, 32]` |

### b8_t20_float32: nano_vllm_torch decode

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| nano_paged_kv_gather | 16.420 | 16.461 | 16 | 240 | 262.728 | `key_cache=[8, 256, 4, 32]; value_cache=[8, 256, 4, 32]; block_table=[1]` |
| qk_matmul | 12.300 | 12.370 | 16 | 240 | 196.799 | `query=[1, 4, 32]; key=[21, 4, 32]` |
| causal_mask | 11.033 | 11.094 | 16 | 240 | 176.523 | `scores=[4, 1, 21]; mask=[1, 1, 21]` |
| rope | 65.399 | 65.629 | 2 | 30 | 130.798 | `q_raw=[8, 4, 32]; k_raw=[8, 4, 32]; positions=[8]` |
| attention_value_matmul | 8.010 | 8.026 | 16 | 240 | 128.160 | `probabilities=[4, 1, 21]; value=[21, 4, 32]` |
| softmax | 3.997 | 4.003 | 16 | 240 | 63.955 | `scores=[4, 1, 21]` |
| nano_paged_kv_store | 7.803 | 7.811 | 2 | 30 | 15.605 | `key=[8, 4, 32]; value=[8, 4, 32]; key_cache=[8, 256, 4, 32]; slot_mapping=[8]` |
| mlp_fc1 | 7.550 | 7.607 | 2 | 30 | 15.099 | `hidden_states=[8, 128]; weight=[512, 128]` |
| mlp_fc2 | 7.477 | 7.565 | 2 | 30 | 14.955 | `hidden_states=[8, 512]; weight=[128, 512]` |
| mlp_norm | 7.066 | 7.124 | 2 | 30 | 14.131 | `hidden_states=[8, 128]` |
| attention_norm | 7.043 | 7.082 | 2 | 30 | 14.087 | `hidden_states=[8, 128]` |
| qkv_linear | 6.863 | 6.918 | 2 | 30 | 13.726 | `hidden_states=[8, 128]; weight=[384, 128]` |
| output_projection | 6.846 | 6.866 | 2 | 30 | 13.692 | `hidden_states=[8, 128]; weight=[128, 128]` |
| attention_output_concat | 4.372 | 4.459 | 2 | 30 | 8.745 | `sequence_0=[1, 4, 32]; sequence_1=[1, 4, 32]; sequence_2=[1, 4, 32]; sequence_3=[1, 4, 32]; sequence_4=[1, 4, 32]; sequence_5=[1, 4, 32]; sequence_6=[1, 4, 32]; sequence_7=[1, 4, 32]` |
| final_norm | 7.150 | 7.216 | 1 | 15 | 7.150 | `hidden_states=[8, 128]` |
| gelu | 3.398 | 3.408 | 2 | 30 | 6.796 | `hidden_states=[8, 512]` |
| embedding | 5.541 | 5.556 | 1 | 15 | 5.541 | `input_ids=[8]; weight=[512, 128]` |
| lm_head | 5.225 | 5.266 | 1 | 15 | 5.225 | `hidden_states=[8, 128]; weight=[512, 128]` |
| head_merge | 0.391 | 0.394 | 2 | 30 | 0.782 | `attention_values=[8, 4, 32]` |

### b1_t20_float16: minillm prefill

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 104.466 | 107.129 | 2 | 2 | 208.932 | `q_raw=[1, 4, 20, 32]; k_raw=[1, 4, 20, 32]; positions=[20]` |
| causal_mask | 11.720 | 11.959 | 2 | 2 | 23.440 | `scores=[1, 4, 20, 20]; mask=[1, 1, 20, 20]` |
| qk_matmul | 10.275 | 10.370 | 2 | 2 | 20.551 | `query=[1, 4, 20, 32]; key=[1, 4, 20, 32]` |
| mlp_fc2 | 8.189 | 8.241 | 2 | 2 | 16.377 | `hidden_states=[1, 20, 512]; weight=[128, 512]` |
| mlp_fc1 | 8.147 | 8.230 | 2 | 2 | 16.293 | `hidden_states=[1, 20, 128]; weight=[512, 128]` |
| qkv_linear | 7.533 | 7.615 | 2 | 2 | 15.066 | `hidden_states=[1, 20, 128]; weight=[384, 128]` |
| output_projection | 7.520 | 7.598 | 2 | 2 | 15.039 | `hidden_states=[1, 20, 128]; weight=[128, 128]` |
| mlp_norm | 7.218 | 7.298 | 2 | 2 | 14.435 | `hidden_states=[1, 20, 128]` |
| attention_norm | 7.158 | 7.648 | 2 | 2 | 14.316 | `hidden_states=[1, 20, 128]` |
| attention_value_matmul | 5.849 | 5.926 | 2 | 2 | 11.699 | `probabilities=[1, 4, 20, 20]; value=[1, 4, 20, 32]` |
| head_merge | 5.212 | 5.251 | 2 | 2 | 10.424 | `attention_values=[1, 4, 20, 32]` |
| embedding | 7.339 | 7.400 | 1 | 1 | 7.339 | `input_ids=[1, 20]; weight=[512, 128]` |
| softmax | 3.666 | 3.776 | 2 | 2 | 7.333 | `scores=[1, 4, 20, 20]` |
| final_norm | 7.297 | 7.323 | 1 | 1 | 7.297 | `hidden_states=[1, 20, 128]` |
| gelu | 3.303 | 3.318 | 2 | 2 | 6.606 | `hidden_states=[1, 20, 512]` |
| lm_head | 5.876 | 5.895 | 1 | 1 | 5.876 | `hidden_states=[1, 20, 128]; weight=[512, 128]` |

### b1_t20_float16: minillm decode

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 97.665 | 99.365 | 2 | 30 | 195.329 | `q_raw=[1, 4, 1, 32]; k_raw=[1, 4, 1, 32]; positions=[1]` |
| causal_mask | 11.399 | 11.436 | 2 | 30 | 22.798 | `scores=[1, 4, 1, 21]; mask=[1, 1, 1, 21]` |
| qk_matmul | 9.814 | 9.875 | 2 | 30 | 19.627 | `query=[1, 4, 1, 32]; key=[1, 4, 21, 32]` |
| native_kv_concat | 8.003 | 8.096 | 2 | 30 | 16.006 | `past_key=[1, 4, 20, 32]; new_key=[1, 4, 1, 32]; past_value=[1, 4, 20, 32]; new_value=[1, 4, 1, 32]` |
| mlp_fc1 | 7.912 | 8.137 | 2 | 30 | 15.824 | `hidden_states=[1, 1, 128]; weight=[512, 128]` |
| mlp_fc2 | 7.886 | 7.996 | 2 | 30 | 15.773 | `hidden_states=[1, 1, 512]; weight=[128, 512]` |
| qkv_linear | 7.201 | 7.281 | 2 | 30 | 14.403 | `hidden_states=[1, 1, 128]; weight=[384, 128]` |
| attention_norm | 7.120 | 7.176 | 2 | 30 | 14.241 | `hidden_states=[1, 1, 128]` |
| output_projection | 7.104 | 7.134 | 2 | 30 | 14.208 | `hidden_states=[1, 1, 128]; weight=[128, 128]` |
| mlp_norm | 7.050 | 7.159 | 2 | 30 | 14.099 | `hidden_states=[1, 1, 128]` |
| attention_value_matmul | 5.271 | 5.280 | 2 | 30 | 10.542 | `probabilities=[1, 4, 1, 21]; value=[1, 4, 21, 32]` |
| final_norm | 7.356 | 7.447 | 1 | 15 | 7.356 | `hidden_states=[1, 1, 128]` |
| softmax | 3.623 | 3.642 | 2 | 30 | 7.246 | `scores=[1, 4, 1, 21]` |
| gelu | 3.363 | 3.402 | 2 | 30 | 6.726 | `hidden_states=[1, 1, 512]` |
| embedding | 5.758 | 5.794 | 1 | 15 | 5.758 | `input_ids=[1, 1]; weight=[512, 128]` |
| lm_head | 5.738 | 5.933 | 1 | 15 | 5.738 | `hidden_states=[1, 1, 128]; weight=[512, 128]` |
| head_merge | 0.891 | 0.894 | 2 | 30 | 1.782 | `attention_values=[1, 4, 1, 32]` |

### b1_t20_float16: nano_vllm_torch prefill

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 88.957 | 92.331 | 2 | 2 | 177.915 | `q_raw=[20, 4, 32]; k_raw=[20, 4, 32]; positions=[20]` |
| qk_matmul | 17.471 | 17.939 | 2 | 2 | 34.943 | `query=[20, 4, 32]; key=[20, 4, 32]` |
| causal_mask | 11.646 | 12.118 | 2 | 2 | 23.291 | `scores=[4, 20, 20]; mask=[1, 20, 20]` |
| attention_value_matmul | 8.824 | 9.032 | 2 | 2 | 17.648 | `probabilities=[4, 20, 20]; value=[20, 4, 32]` |
| mlp_fc2 | 8.142 | 8.285 | 2 | 2 | 16.284 | `hidden_states=[20, 512]; weight=[128, 512]` |
| mlp_fc1 | 8.128 | 8.271 | 2 | 2 | 16.256 | `hidden_states=[20, 128]; weight=[512, 128]` |
| nano_paged_kv_store | 7.913 | 8.031 | 2 | 2 | 15.826 | `key=[20, 4, 32]; value=[20, 4, 32]; key_cache=[1, 256, 4, 32]; slot_mapping=[20]` |
| softmax | 7.863 | 8.032 | 2 | 2 | 15.727 | `scores=[4, 20, 20]` |
| output_projection | 7.610 | 7.956 | 2 | 2 | 15.219 | `hidden_states=[20, 128]; weight=[128, 128]` |
| qkv_linear | 7.504 | 7.619 | 2 | 2 | 15.008 | `hidden_states=[20, 128]; weight=[384, 128]` |
| mlp_norm | 7.400 | 7.524 | 2 | 2 | 14.801 | `hidden_states=[20, 128]` |
| attention_norm | 7.338 | 7.463 | 2 | 2 | 14.676 | `hidden_states=[20, 128]` |
| attention_output_concat | 4.813 | 4.996 | 2 | 2 | 9.627 | `sequence_0=[20, 4, 32]` |
| final_norm | 7.373 | 7.459 | 1 | 1 | 7.373 | `hidden_states=[20, 128]` |
| embedding | 7.162 | 7.372 | 1 | 1 | 7.162 | `input_ids=[20]; weight=[512, 128]` |
| gelu | 3.347 | 3.434 | 2 | 2 | 6.694 | `hidden_states=[20, 512]` |
| lm_head | 5.478 | 5.608 | 1 | 1 | 5.478 | `hidden_states=[1, 128]; weight=[512, 128]` |
| head_merge | 0.390 | 0.490 | 2 | 2 | 0.780 | `attention_values=[20, 4, 32]` |

### b1_t20_float16: nano_vllm_torch decode

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 82.117 | 82.271 | 2 | 30 | 164.235 | `q_raw=[1, 4, 32]; k_raw=[1, 4, 32]; positions=[1]` |
| nano_paged_kv_gather | 16.789 | 16.864 | 2 | 30 | 33.578 | `key_cache=[1, 256, 4, 32]; value_cache=[1, 256, 4, 32]; block_table=[1]` |
| qk_matmul | 16.520 | 16.807 | 2 | 30 | 33.039 | `query=[1, 4, 32]; key=[21, 4, 32]` |
| causal_mask | 11.380 | 11.761 | 2 | 30 | 22.759 | `scores=[4, 1, 21]; mask=[1, 1, 21]` |
| attention_value_matmul | 8.228 | 8.321 | 2 | 30 | 16.455 | `probabilities=[4, 1, 21]; value=[21, 4, 32]` |
| softmax | 7.817 | 7.907 | 2 | 30 | 15.633 | `scores=[4, 1, 21]` |
| mlp_fc2 | 7.810 | 7.818 | 2 | 30 | 15.620 | `hidden_states=[1, 512]; weight=[128, 512]` |
| nano_paged_kv_store | 7.810 | 7.812 | 2 | 30 | 15.619 | `key=[1, 4, 32]; value=[1, 4, 32]; key_cache=[1, 256, 4, 32]; slot_mapping=[1]` |
| mlp_fc1 | 7.809 | 7.840 | 2 | 30 | 15.618 | `hidden_states=[1, 128]; weight=[512, 128]` |
| mlp_norm | 7.199 | 7.239 | 2 | 30 | 14.398 | `hidden_states=[1, 128]` |
| attention_norm | 7.144 | 7.276 | 2 | 30 | 14.288 | `hidden_states=[1, 128]` |
| qkv_linear | 7.083 | 7.150 | 2 | 30 | 14.165 | `hidden_states=[1, 128]; weight=[384, 128]` |
| output_projection | 7.047 | 7.074 | 2 | 30 | 14.094 | `hidden_states=[1, 128]; weight=[128, 128]` |
| attention_output_concat | 4.436 | 4.446 | 2 | 30 | 8.871 | `sequence_0=[1, 4, 32]` |
| final_norm | 7.254 | 7.275 | 1 | 15 | 7.254 | `hidden_states=[1, 128]` |
| gelu | 3.368 | 3.388 | 2 | 30 | 6.736 | `hidden_states=[1, 512]` |
| embedding | 5.551 | 5.606 | 1 | 15 | 5.551 | `input_ids=[1]; weight=[512, 128]` |
| lm_head | 5.478 | 5.593 | 1 | 15 | 5.478 | `hidden_states=[1, 128]; weight=[512, 128]` |
| head_merge | 0.396 | 0.400 | 2 | 30 | 0.791 | `attention_values=[1, 4, 32]` |

### b8_t20_float16: minillm prefill

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 105.810 | 111.463 | 2 | 2 | 211.619 | `q_raw=[8, 4, 20, 32]; k_raw=[8, 4, 20, 32]; positions=[20]` |
| causal_mask | 11.764 | 12.281 | 2 | 2 | 23.527 | `scores=[8, 4, 20, 20]; mask=[1, 1, 20, 20]` |
| qk_matmul | 10.416 | 10.526 | 2 | 2 | 20.833 | `query=[8, 4, 20, 32]; key=[8, 4, 20, 32]` |
| attention_value_matmul | 9.776 | 9.892 | 2 | 2 | 19.553 | `probabilities=[8, 4, 20, 20]; value=[8, 4, 20, 32]` |
| mlp_fc2 | 8.227 | 8.242 | 2 | 2 | 16.454 | `hidden_states=[8, 20, 512]; weight=[128, 512]` |
| mlp_fc1 | 8.166 | 8.187 | 2 | 2 | 16.332 | `hidden_states=[8, 20, 128]; weight=[512, 128]` |
| qkv_linear | 7.656 | 7.715 | 2 | 2 | 15.312 | `hidden_states=[8, 20, 128]; weight=[384, 128]` |
| output_projection | 7.484 | 7.506 | 2 | 2 | 14.969 | `hidden_states=[8, 20, 128]; weight=[128, 128]` |
| mlp_norm | 7.280 | 7.301 | 2 | 2 | 14.560 | `hidden_states=[8, 20, 128]` |
| attention_norm | 7.254 | 7.279 | 2 | 2 | 14.509 | `hidden_states=[8, 20, 128]` |
| head_merge | 5.234 | 5.245 | 2 | 2 | 10.467 | `attention_values=[8, 4, 20, 32]` |
| softmax | 3.866 | 3.888 | 2 | 2 | 7.731 | `scores=[8, 4, 20, 20]` |
| embedding | 7.427 | 7.511 | 1 | 1 | 7.427 | `input_ids=[8, 20]; weight=[512, 128]` |
| final_norm | 7.259 | 7.346 | 1 | 1 | 7.259 | `hidden_states=[8, 20, 128]` |
| gelu | 3.417 | 3.441 | 2 | 2 | 6.833 | `hidden_states=[8, 20, 512]` |
| lm_head | 5.959 | 5.989 | 1 | 1 | 5.959 | `hidden_states=[8, 20, 128]; weight=[512, 128]` |

### b8_t20_float16: minillm decode

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 99.370 | 100.856 | 2 | 30 | 198.741 | `q_raw=[8, 4, 1, 32]; k_raw=[8, 4, 1, 32]; positions=[1]` |
| causal_mask | 11.721 | 12.186 | 2 | 30 | 23.442 | `scores=[8, 4, 1, 21]; mask=[1, 1, 1, 21]` |
| qk_matmul | 10.184 | 10.323 | 2 | 30 | 20.368 | `query=[8, 4, 1, 32]; key=[8, 4, 21, 32]` |
| mlp_fc1 | 8.218 | 8.234 | 2 | 30 | 16.436 | `hidden_states=[8, 1, 128]; weight=[512, 128]` |
| mlp_fc2 | 8.147 | 8.188 | 2 | 30 | 16.295 | `hidden_states=[8, 1, 512]; weight=[128, 512]` |
| native_kv_concat | 8.129 | 8.287 | 2 | 30 | 16.258 | `past_key=[8, 4, 20, 32]; new_key=[8, 4, 1, 32]; past_value=[8, 4, 20, 32]; new_value=[8, 4, 1, 32]` |
| qkv_linear | 7.607 | 7.702 | 2 | 30 | 15.214 | `hidden_states=[8, 1, 128]; weight=[384, 128]` |
| output_projection | 7.492 | 7.510 | 2 | 30 | 14.984 | `hidden_states=[8, 1, 128]; weight=[128, 128]` |
| attention_norm | 7.248 | 7.262 | 2 | 30 | 14.496 | `hidden_states=[8, 1, 128]` |
| mlp_norm | 7.223 | 7.244 | 2 | 30 | 14.446 | `hidden_states=[8, 1, 128]` |
| attention_value_matmul | 5.362 | 5.445 | 2 | 30 | 10.723 | `probabilities=[8, 4, 1, 21]; value=[8, 4, 21, 32]` |
| softmax | 3.898 | 3.934 | 2 | 30 | 7.796 | `scores=[8, 4, 1, 21]` |
| final_norm | 7.289 | 7.300 | 1 | 15 | 7.289 | `hidden_states=[8, 1, 128]` |
| gelu | 3.340 | 3.348 | 2 | 30 | 6.680 | `hidden_states=[8, 1, 512]` |
| lm_head | 5.954 | 5.974 | 1 | 15 | 5.954 | `hidden_states=[8, 1, 128]; weight=[512, 128]` |
| embedding | 5.851 | 5.865 | 1 | 15 | 5.851 | `input_ids=[8, 1]; weight=[512, 128]` |
| head_merge | 0.890 | 1.005 | 2 | 30 | 1.780 | `attention_values=[8, 4, 1, 32]` |

### b8_t20_float16: nano_vllm_torch prefill

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| qk_matmul | 17.333 | 17.367 | 16 | 16 | 277.323 | `query=[20, 4, 32]; key=[20, 4, 32]` |
| causal_mask | 11.605 | 11.736 | 16 | 16 | 185.681 | `scores=[4, 20, 20]; mask=[1, 20, 20]` |
| rope | 85.994 | 86.243 | 2 | 2 | 171.989 | `q_raw=[160, 4, 32]; k_raw=[160, 4, 32]; positions=[160]` |
| attention_value_matmul | 8.668 | 8.706 | 16 | 16 | 138.690 | `probabilities=[4, 20, 20]; value=[20, 4, 32]` |
| softmax | 7.822 | 7.835 | 16 | 16 | 125.144 | `scores=[4, 20, 20]` |
| mlp_fc2 | 8.083 | 8.203 | 2 | 2 | 16.166 | `hidden_states=[160, 512]; weight=[128, 512]` |
| mlp_fc1 | 8.019 | 8.070 | 2 | 2 | 16.039 | `hidden_states=[160, 128]; weight=[512, 128]` |
| nano_paged_kv_store | 7.810 | 8.029 | 2 | 2 | 15.619 | `key=[160, 4, 32]; value=[160, 4, 32]; key_cache=[8, 256, 4, 32]; slot_mapping=[160]` |
| output_projection | 7.410 | 7.418 | 2 | 2 | 14.820 | `hidden_states=[160, 128]; weight=[128, 128]` |
| qkv_linear | 7.407 | 7.440 | 2 | 2 | 14.814 | `hidden_states=[160, 128]; weight=[384, 128]` |
| mlp_norm | 7.227 | 7.258 | 2 | 2 | 14.454 | `hidden_states=[160, 128]` |
| attention_norm | 7.214 | 7.280 | 2 | 2 | 14.428 | `hidden_states=[160, 128]` |
| attention_output_concat | 4.468 | 4.484 | 2 | 2 | 8.937 | `sequence_0=[20, 4, 32]; sequence_1=[20, 4, 32]; sequence_2=[20, 4, 32]; sequence_3=[20, 4, 32]; sequence_4=[20, 4, 32]; sequence_5=[20, 4, 32]; sequence_6=[20, 4, 32]; sequence_7=[20, 4, 32]` |
| final_norm | 7.355 | 7.378 | 1 | 1 | 7.355 | `hidden_states=[160, 128]` |
| embedding | 7.095 | 7.125 | 1 | 1 | 7.095 | `input_ids=[160]; weight=[512, 128]` |
| gelu | 3.431 | 3.443 | 2 | 2 | 6.862 | `hidden_states=[160, 512]` |
| lm_head | 5.849 | 5.866 | 1 | 1 | 5.849 | `hidden_states=[8, 128]; weight=[512, 128]` |
| head_merge | 0.398 | 0.402 | 2 | 2 | 0.797 | `attention_values=[160, 4, 32]` |

### b8_t20_float16: nano_vllm_torch decode

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| nano_paged_kv_gather | 16.664 | 16.792 | 16 | 240 | 266.629 | `key_cache=[8, 256, 4, 32]; value_cache=[8, 256, 4, 32]; block_table=[1]` |
| qk_matmul | 16.606 | 16.913 | 16 | 240 | 265.704 | `query=[1, 4, 32]; key=[21, 4, 32]` |
| causal_mask | 11.338 | 11.442 | 16 | 240 | 181.411 | `scores=[4, 1, 21]; mask=[1, 1, 21]` |
| rope | 83.510 | 84.739 | 2 | 30 | 167.020 | `q_raw=[8, 4, 32]; k_raw=[8, 4, 32]; positions=[8]` |
| attention_value_matmul | 8.274 | 8.382 | 16 | 240 | 132.391 | `probabilities=[4, 1, 21]; value=[21, 4, 32]` |
| softmax | 7.812 | 7.913 | 16 | 240 | 124.990 | `scores=[4, 1, 21]` |
| mlp_fc1 | 8.067 | 8.196 | 2 | 30 | 16.133 | `hidden_states=[8, 128]; weight=[512, 128]` |
| mlp_fc2 | 7.968 | 8.115 | 2 | 30 | 15.936 | `hidden_states=[8, 512]; weight=[128, 512]` |
| nano_paged_kv_store | 7.810 | 7.954 | 2 | 30 | 15.620 | `key=[8, 4, 32]; value=[8, 4, 32]; key_cache=[8, 256, 4, 32]; slot_mapping=[8]` |
| qkv_linear | 7.447 | 7.470 | 2 | 30 | 14.895 | `hidden_states=[8, 128]; weight=[384, 128]` |
| output_projection | 7.347 | 7.374 | 2 | 30 | 14.695 | `hidden_states=[8, 128]; weight=[128, 128]` |
| mlp_norm | 7.227 | 7.358 | 2 | 30 | 14.454 | `hidden_states=[8, 128]` |
| attention_norm | 7.215 | 7.244 | 2 | 30 | 14.429 | `hidden_states=[8, 128]` |
| attention_output_concat | 4.359 | 4.402 | 2 | 30 | 8.718 | `sequence_0=[1, 4, 32]; sequence_1=[1, 4, 32]; sequence_2=[1, 4, 32]; sequence_3=[1, 4, 32]; sequence_4=[1, 4, 32]; sequence_5=[1, 4, 32]; sequence_6=[1, 4, 32]; sequence_7=[1, 4, 32]` |
| final_norm | 7.303 | 7.366 | 1 | 15 | 7.303 | `hidden_states=[8, 128]` |
| gelu | 3.356 | 3.400 | 2 | 30 | 6.712 | `hidden_states=[8, 512]` |
| lm_head | 5.722 | 5.771 | 1 | 15 | 5.722 | `hidden_states=[8, 128]; weight=[512, 128]` |
| embedding | 5.604 | 5.620 | 1 | 15 | 5.604 | `input_ids=[8]; weight=[512, 128]` |
| head_merge | 0.394 | 0.397 | 2 | 30 | 0.789 | `attention_values=[8, 4, 32]` |

## Canonical tensor layouts

The pointer alignment is the maximum power-of-two alignment observed for that concrete allocation. `all-row alignment` additionally requires every active outer stride to preserve that alignment, so it is the guarantee for all last-dimension row starts. Both values are diagnostic for this run, not permanent allocator guarantees.

| Workload | Runtime/phase | Tensor | Shape | Stride | dtype | Contiguous | Ptr / all-row alignment | mod 128 / 256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_t20_float32 | minillm/prefill | input_ids | `[1, 20]` | `[20, 1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | minillm/prefill | hidden | `[1, 20, 128]` | `[2560, 128, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | minillm/prefill | q_raw | `[1, 4, 20, 32]` | `[7680, 32, 384, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | minillm/prefill | k_raw | `[1, 4, 20, 32]` | `[7680, 32, 384, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | minillm/prefill | v_raw | `[1, 4, 20, 32]` | `[7680, 32, 384, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | minillm/prefill | query | `[1, 4, 20, 32]` | `[2560, 640, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | minillm/prefill | key | `[1, 4, 20, 32]` | `[2560, 640, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | minillm/prefill | scores | `[1, 4, 20, 20]` | `[1600, 400, 20, 1]` | float32 | True | 256 / 16 B | 0 / 0 |
| b1_t20_float32 | minillm/prefill | logits | `[1, 20, 512]` | `[10240, 512, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | minillm/decode | input_ids | `[1, 1]` | `[1, 1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | minillm/decode | hidden | `[1, 1, 128]` | `[128, 128, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | minillm/decode | q_raw | `[1, 4, 1, 32]` | `[128, 32, 128, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | minillm/decode | k_raw | `[1, 4, 1, 32]` | `[128, 32, 128, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | minillm/decode | v_raw | `[1, 4, 1, 32]` | `[128, 32, 128, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | minillm/decode | query | `[1, 4, 1, 32]` | `[128, 32, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | minillm/decode | key | `[1, 4, 21, 32]` | `[2688, 672, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | minillm/decode | scores | `[1, 4, 1, 21]` | `[84, 21, 21, 1]` | float32 | True | 256 / 4 B | 0 / 0 |
| b1_t20_float32 | minillm/decode | logits | `[1, 1, 512]` | `[512, 512, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/prefill | input_ids | `[20]` | `[1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/prefill | hidden | `[20, 128]` | `[128, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/prefill | q_raw | `[20, 4, 32]` | `[384, 32, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/prefill | k_raw | `[20, 4, 32]` | `[384, 32, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/prefill | v_raw | `[20, 4, 32]` | `[384, 32, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/prefill | query | `[20, 4, 32]` | `[128, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/prefill | key | `[20, 4, 32]` | `[128, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/prefill | scores | `[4, 20, 20]` | `[400, 20, 1]` | float32 | True | 256 / 16 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/prefill | key_cache | `[1, 256, 4, 32]` | `[32768, 128, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/prefill | slot_mapping | `[20]` | `[1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/prefill | head_input | `[1, 128]` | `[128, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/prefill | logits | `[1, 512]` | `[512, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/decode | input_ids | `[1]` | `[1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/decode | hidden | `[1, 128]` | `[128, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/decode | q_raw | `[1, 4, 32]` | `[128, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/decode | k_raw | `[1, 4, 32]` | `[128, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/decode | v_raw | `[1, 4, 32]` | `[128, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/decode | query | `[1, 4, 32]` | `[128, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/decode | key | `[1, 4, 32]` | `[128, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/decode | scores | `[4, 1, 21]` | `[21, 21, 1]` | float32 | True | 256 / 4 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/decode | key_cache | `[1, 256, 4, 32]` | `[32768, 128, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/decode | slot_mapping | `[1]` | `[1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/decode | head_input | `[1, 128]` | `[128, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/decode | logits | `[1, 512]` | `[512, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float32 | minillm/prefill | input_ids | `[8, 20]` | `[20, 1]` | int64 | True | 256 / 32 B | 0 / 0 |
| b8_t20_float32 | minillm/prefill | hidden | `[8, 20, 128]` | `[2560, 128, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float32 | minillm/prefill | q_raw | `[8, 4, 20, 32]` | `[7680, 32, 384, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | minillm/prefill | k_raw | `[8, 4, 20, 32]` | `[7680, 32, 384, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | minillm/prefill | v_raw | `[8, 4, 20, 32]` | `[7680, 32, 384, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | minillm/prefill | query | `[8, 4, 20, 32]` | `[2560, 640, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | minillm/prefill | key | `[8, 4, 20, 32]` | `[2560, 640, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | minillm/prefill | scores | `[8, 4, 20, 20]` | `[1600, 400, 20, 1]` | float32 | True | 256 / 16 B | 0 / 0 |
| b8_t20_float32 | minillm/prefill | logits | `[8, 20, 512]` | `[10240, 512, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float32 | minillm/decode | input_ids | `[8, 1]` | `[1, 1]` | int64 | True | 256 / 8 B | 0 / 0 |
| b8_t20_float32 | minillm/decode | hidden | `[8, 1, 128]` | `[128, 128, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float32 | minillm/decode | q_raw | `[8, 4, 1, 32]` | `[384, 32, 128, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | minillm/decode | k_raw | `[8, 4, 1, 32]` | `[384, 32, 128, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | minillm/decode | v_raw | `[8, 4, 1, 32]` | `[384, 32, 128, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | minillm/decode | query | `[8, 4, 1, 32]` | `[128, 32, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | minillm/decode | key | `[8, 4, 21, 32]` | `[2688, 672, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | minillm/decode | scores | `[8, 4, 1, 21]` | `[84, 21, 21, 1]` | float32 | True | 256 / 4 B | 0 / 0 |
| b8_t20_float32 | minillm/decode | logits | `[8, 1, 512]` | `[512, 512, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/prefill | input_ids | `[160]` | `[1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/prefill | hidden | `[160, 128]` | `[128, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/prefill | q_raw | `[160, 4, 32]` | `[384, 32, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/prefill | k_raw | `[160, 4, 32]` | `[384, 32, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/prefill | v_raw | `[160, 4, 32]` | `[384, 32, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/prefill | query | `[160, 4, 32]` | `[128, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/prefill | key | `[160, 4, 32]` | `[128, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/prefill | scores | `[4, 20, 20]` | `[400, 20, 1]` | float32 | True | 256 / 16 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/prefill | key_cache | `[8, 256, 4, 32]` | `[32768, 128, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/prefill | slot_mapping | `[160]` | `[1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/prefill | head_input | `[8, 128]` | `[128, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/prefill | logits | `[8, 512]` | `[512, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/decode | input_ids | `[8]` | `[1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/decode | hidden | `[8, 128]` | `[128, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/decode | q_raw | `[8, 4, 32]` | `[384, 32, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/decode | k_raw | `[8, 4, 32]` | `[384, 32, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/decode | v_raw | `[8, 4, 32]` | `[384, 32, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/decode | query | `[8, 4, 32]` | `[128, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/decode | key | `[8, 4, 32]` | `[128, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/decode | scores | `[4, 1, 21]` | `[21, 21, 1]` | float32 | True | 256 / 4 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/decode | key_cache | `[8, 256, 4, 32]` | `[32768, 128, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/decode | slot_mapping | `[8]` | `[1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/decode | head_input | `[8, 128]` | `[128, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float32 | nano_vllm_torch/decode | logits | `[8, 512]` | `[512, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float16 | minillm/prefill | input_ids | `[1, 20]` | `[20, 1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float16 | minillm/prefill | hidden | `[1, 20, 128]` | `[2560, 128, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float16 | minillm/prefill | q_raw | `[1, 4, 20, 32]` | `[7680, 32, 384, 1]` | float16 | False | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | minillm/prefill | k_raw | `[1, 4, 20, 32]` | `[7680, 32, 384, 1]` | float16 | False | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | minillm/prefill | v_raw | `[1, 4, 20, 32]` | `[7680, 32, 384, 1]` | float16 | False | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | minillm/prefill | query | `[1, 4, 20, 32]` | `[2560, 640, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | minillm/prefill | key | `[1, 4, 20, 32]` | `[2560, 640, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | minillm/prefill | scores | `[1, 4, 20, 20]` | `[1600, 400, 20, 1]` | float16 | True | 256 / 8 B | 0 / 0 |
| b1_t20_float16 | minillm/prefill | logits | `[1, 20, 512]` | `[10240, 512, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float16 | minillm/decode | input_ids | `[1, 1]` | `[1, 1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float16 | minillm/decode | hidden | `[1, 1, 128]` | `[128, 128, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float16 | minillm/decode | q_raw | `[1, 4, 1, 32]` | `[128, 32, 128, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | minillm/decode | k_raw | `[1, 4, 1, 32]` | `[128, 32, 128, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | minillm/decode | v_raw | `[1, 4, 1, 32]` | `[128, 32, 128, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | minillm/decode | query | `[1, 4, 1, 32]` | `[128, 32, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | minillm/decode | key | `[1, 4, 21, 32]` | `[2688, 672, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | minillm/decode | scores | `[1, 4, 1, 21]` | `[84, 21, 21, 1]` | float16 | True | 256 / 2 B | 0 / 0 |
| b1_t20_float16 | minillm/decode | logits | `[1, 1, 512]` | `[512, 512, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/prefill | input_ids | `[20]` | `[1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/prefill | hidden | `[20, 128]` | `[128, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/prefill | q_raw | `[20, 4, 32]` | `[384, 32, 1]` | float16 | False | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/prefill | k_raw | `[20, 4, 32]` | `[384, 32, 1]` | float16 | False | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/prefill | v_raw | `[20, 4, 32]` | `[384, 32, 1]` | float16 | False | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/prefill | query | `[20, 4, 32]` | `[128, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/prefill | key | `[20, 4, 32]` | `[128, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/prefill | scores | `[4, 20, 20]` | `[400, 20, 1]` | float32 | True | 256 / 16 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/prefill | key_cache | `[1, 256, 4, 32]` | `[32768, 128, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/prefill | slot_mapping | `[20]` | `[1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/prefill | head_input | `[1, 128]` | `[128, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/prefill | logits | `[1, 512]` | `[512, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/decode | input_ids | `[1]` | `[1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/decode | hidden | `[1, 128]` | `[128, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/decode | q_raw | `[1, 4, 32]` | `[128, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/decode | k_raw | `[1, 4, 32]` | `[128, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/decode | v_raw | `[1, 4, 32]` | `[128, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/decode | query | `[1, 4, 32]` | `[128, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/decode | key | `[1, 4, 32]` | `[128, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/decode | scores | `[4, 1, 21]` | `[21, 21, 1]` | float32 | True | 256 / 4 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/decode | key_cache | `[1, 256, 4, 32]` | `[32768, 128, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/decode | slot_mapping | `[1]` | `[1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/decode | head_input | `[1, 128]` | `[128, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float16 | nano_vllm_torch/decode | logits | `[1, 512]` | `[512, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float16 | minillm/prefill | input_ids | `[8, 20]` | `[20, 1]` | int64 | True | 256 / 32 B | 0 / 0 |
| b8_t20_float16 | minillm/prefill | hidden | `[8, 20, 128]` | `[2560, 128, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float16 | minillm/prefill | q_raw | `[8, 4, 20, 32]` | `[7680, 32, 384, 1]` | float16 | False | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | minillm/prefill | k_raw | `[8, 4, 20, 32]` | `[7680, 32, 384, 1]` | float16 | False | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | minillm/prefill | v_raw | `[8, 4, 20, 32]` | `[7680, 32, 384, 1]` | float16 | False | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | minillm/prefill | query | `[8, 4, 20, 32]` | `[2560, 640, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | minillm/prefill | key | `[8, 4, 20, 32]` | `[2560, 640, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | minillm/prefill | scores | `[8, 4, 20, 20]` | `[1600, 400, 20, 1]` | float16 | True | 256 / 8 B | 0 / 0 |
| b8_t20_float16 | minillm/prefill | logits | `[8, 20, 512]` | `[10240, 512, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float16 | minillm/decode | input_ids | `[8, 1]` | `[1, 1]` | int64 | True | 256 / 8 B | 0 / 0 |
| b8_t20_float16 | minillm/decode | hidden | `[8, 1, 128]` | `[128, 128, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float16 | minillm/decode | q_raw | `[8, 4, 1, 32]` | `[384, 32, 128, 1]` | float16 | False | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | minillm/decode | k_raw | `[8, 4, 1, 32]` | `[384, 32, 128, 1]` | float16 | False | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | minillm/decode | v_raw | `[8, 4, 1, 32]` | `[384, 32, 128, 1]` | float16 | False | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | minillm/decode | query | `[8, 4, 1, 32]` | `[128, 32, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | minillm/decode | key | `[8, 4, 21, 32]` | `[2688, 672, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | minillm/decode | scores | `[8, 4, 1, 21]` | `[84, 21, 21, 1]` | float16 | True | 256 / 2 B | 0 / 0 |
| b8_t20_float16 | minillm/decode | logits | `[8, 1, 512]` | `[512, 512, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/prefill | input_ids | `[160]` | `[1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/prefill | hidden | `[160, 128]` | `[128, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/prefill | q_raw | `[160, 4, 32]` | `[384, 32, 1]` | float16 | False | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/prefill | k_raw | `[160, 4, 32]` | `[384, 32, 1]` | float16 | False | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/prefill | v_raw | `[160, 4, 32]` | `[384, 32, 1]` | float16 | False | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/prefill | query | `[160, 4, 32]` | `[128, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/prefill | key | `[160, 4, 32]` | `[128, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/prefill | scores | `[4, 20, 20]` | `[400, 20, 1]` | float32 | True | 256 / 16 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/prefill | key_cache | `[8, 256, 4, 32]` | `[32768, 128, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/prefill | slot_mapping | `[160]` | `[1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/prefill | head_input | `[8, 128]` | `[128, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/prefill | logits | `[8, 512]` | `[512, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/decode | input_ids | `[8]` | `[1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/decode | hidden | `[8, 128]` | `[128, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/decode | q_raw | `[8, 4, 32]` | `[384, 32, 1]` | float16 | False | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/decode | k_raw | `[8, 4, 32]` | `[384, 32, 1]` | float16 | False | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/decode | v_raw | `[8, 4, 32]` | `[384, 32, 1]` | float16 | False | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/decode | query | `[8, 4, 32]` | `[128, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/decode | key | `[8, 4, 32]` | `[128, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/decode | scores | `[4, 1, 21]` | `[21, 21, 1]` | float32 | True | 256 / 4 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/decode | key_cache | `[8, 256, 4, 32]` | `[32768, 128, 32, 1]` | float16 | True | 256 / 64 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/decode | slot_mapping | `[8]` | `[1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/decode | head_input | `[8, 128]` | `[128, 1]` | float16 | True | 256 / 256 B | 0 / 0 |
| b8_t20_float16 | nano_vllm_torch/decode | logits | `[8, 512]` | `[512, 1]` | float16 | True | 256 / 256 B | 0 / 0 |

## Numerical layout parity

nano-vLLM serving returns only the last prefill logit per sequence, while MiniLLM returns all positions. These checks compare equivalent last-token/decode logits.

| Workload | Phase | Max abs error | Mean abs error | Argmax match |
| --- | --- | ---: | ---: | ---: |
| b1_t20_float32 | prefill_last_token_logits | 4.76837e-07 | 8.0192e-08 | 1.000 |
| b1_t20_float32 | decode_logits | 0 | 0 | 1.000 |
| b8_t20_float32 | prefill_last_token_logits | 7.15256e-07 | 9.70285e-08 | 1.000 |
| b8_t20_float32 | decode_logits | 7.15256e-07 | 1.61146e-07 | 1.000 |
| b1_t20_float16 | prefill_last_token_logits | 0.00390625 | 0.000530105 | 1.000 |
| b1_t20_float16 | decode_logits | 0.00292969 | 0.000501211 | 1.000 |
| b8_t20_float16 | prefill_last_token_logits | 0.00390625 | 0.000529986 | 1.000 |
| b8_t20_float16 | decode_logits | 0.00292969 | 0.000497929 | 1.000 |

## Profiler materialization audit

Each linear, matmul, layout, and KV-cache stage gets one post-timing PyTorch profiler invocation. `clone` or `contiguous` means the eager operator materialized a new layout before or during the measured operation; profiler overhead is not part of CUDA Event timing.

| Workload | Runtime/phase | Stage | clone | contiguous | copy | bmm | Evidence |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| b1_t20_float32 | minillm/prefill | head_merge | 1 | 1 | 1 | 0 | `aten::contiguous [[1, 20, 4, 32], []]; aten::clone [[1, 20, 4, 32], []]` |
| b8_t20_float32 | minillm/prefill | attention_value_matmul | 1 | 0 | 1 | 1 | `aten::clone [[8, 4, 20, 32], []]` |
| b8_t20_float32 | minillm/prefill | head_merge | 1 | 1 | 1 | 0 | `aten::contiguous [[8, 20, 4, 32], []]; aten::clone [[8, 20, 4, 32], []]` |
| b1_t20_float16 | minillm/prefill | head_merge | 1 | 1 | 1 | 0 | `aten::contiguous [[1, 20, 4, 32], []]; aten::clone [[1, 20, 4, 32], []]` |
| b8_t20_float16 | minillm/prefill | attention_value_matmul | 1 | 0 | 1 | 1 | `aten::clone [[8, 4, 20, 32], []]` |
| b8_t20_float16 | minillm/prefill | head_merge | 1 | 1 | 1 | 0 | `aten::contiguous [[8, 20, 4, 32], []]; aten::clone [[8, 20, 4, 32], []]` |

## Interpretation constraints

- This checkpoint is intentionally tiny. Decode GEMMs have very small `M` and are commonly launch/latency bound rather than Tensor Core throughput bound.
- nano-vLLM's PyTorch attention is a correctness fallback that loops over sequences and gathers paged KV. FlashAttention/Triton/CUDA Graph results must be measured separately, not inferred from this table.
- Stage sums are attribution estimates. They do not include every residual/add/cast/allocation and therefore need not equal full-pass latency.
- Decode timing is for the first incremental step at the reported context. Attention and cache costs increase as more tokens are appended.
- `F.linear` and batched matmul dispatch through PyTorch's CUDA BLAS integration; exact cuBLAS versus cuBLASLt algorithm selection is internal and can change with shape, dtype, and PyTorch version.
