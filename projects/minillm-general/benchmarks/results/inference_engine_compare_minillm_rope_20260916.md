# MiniLLM / nano-vLLM PyTorch-cuBLAS inference baseline

Generated: `2026-09-16T09:02:35+0800`

## Scope and method

This is an eager PyTorch baseline on the real MiniLLM checkpoint and the tensor layouts used by MiniLLM and nano-vLLM. Linear layers use `F.linear` (cuBLAS/cuBLASLt); MiniLLM QK/PV use `matmul`, while the nano-vLLM fallback uses its real `einsum` equations; both lower to batched GEMM/cuBLAS for these shapes. FlashAttention, Triton attention, CUDA Graph, and `torch.compile` are intentionally excluded so later optimized kernels have a stable reference.

Timing uses 3 warmups and 10 CUDA Event samples. The adaptive inner loop targets 2.0 ms per sample. Synchronized wall time includes Python dispatch and launch/synchronization overhead; GPU Event time covers work on the CUDA stream.

For `N` generated tokens, prefill produces token 1 and the model executes `N-1` decode passes. Component contribution is `median time for one measured stage invocation x logical calls per pass`.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 SUPER |
| Compute capability | [8, 9] |
| PyTorch / CUDA | 2.9.1+cu130 / 13.0 |
| NVIDIA driver | 580.173.02 |
| BLAS | _BlasBackend.Cublas |
| TF32 | matmul=False, cuDNN=False |
| Checkpoint SHA-256 | `85f69b910214c78537c510620f084454e26dba8bbea66506cc94664ac5f31e57` |
| Git | `5e85b94a89e2cc77a893b71fcdb3042570ea14a8`; dirty=True |

## Model and workloads

Model: `L=2, C=128, H=4, Hkv=4, D=32, V=512, block_size=128`. Prompt token count: `20`. nano-vLLM KV page size: `256`.
The decode row measures the first incremental pass: one new token is processed while attention reads the prompt plus that token. The call count for a 8-token generation is exact, but later decode passes have progressively longer KV lengths and are not assigned the first-step latency.

| Workload | MiniLLM prefill | nano prefill | MiniLLM decode | nano decode | dtype |
| --- | --- | --- | --- | --- | --- |
| b1_t20_float32 | `[1, 20, 128]` | `[20, 128]` | `[1, 1, 128]` | `[1, 128]` | float32 |
| b1_t20_float16 | `[1, 20, 128]` | `[20, 128]` | `[1, 1, 128]` | `[1, 128]` | float16 |

## Full-pass latency

| Workload | Runtime | Phase | GPU median (ms) | p90 (ms) | Wall median (ms) | Throughput token/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| b1_t20_float32 | minillm | prefill | 0.525036 | 0.534888 | 0.527208 | 38092.6 |
| b1_t20_float32 | minillm | decode | 0.491888 | 0.536096 | 0.494399 | 2033.0 |
| b1_t20_float32 | nano_vllm_torch | prefill | 0.596288 | 0.600312 | 0.598454 | 33540.8 |
| b1_t20_float32 | nano_vllm_torch | decode | 0.565768 | 0.665032 | 0.579514 | 1767.5 |
| b1_t20_float16 | minillm | prefill | 0.563320 | 0.574216 | 0.566520 | 35503.8 |
| b1_t20_float16 | minillm | decode | 0.538472 | 0.600752 | 0.540876 | 1857.1 |
| b1_t20_float16 | nano_vllm_torch | prefill | 0.654452 | 0.663968 | 0.658400 | 30559.9 |
| b1_t20_float16 | nano_vllm_torch | decode | 0.624324 | 0.627440 | 0.627868 | 1601.7 |

## Component hotspots

The ranking multiplies isolated median stage latency by the logical calls in one model pass. It identifies optimization candidates; it is not a claim that these isolated rows sum exactly to full-model latency.

| Workload | Runtime/phase | Top 1 | Top 2 | Top 3 |
| --- | --- | --- | --- | --- |
| b1_t20_float32 | minillm/prefill | rope (179.9 us/pass) | causal_mask (23.6 us/pass) | qk_matmul (20.4 us/pass) |
| b1_t20_float32 | minillm/decode | rope (166.6 us/pass) | causal_mask (23.6 us/pass) | qk_matmul (20.6 us/pass) |
| b1_t20_float32 | nano_vllm_torch/prefill | rope (144.1 us/pass) | qk_matmul (26.1 us/pass) | causal_mask (23.2 us/pass) |
| b1_t20_float32 | nano_vllm_torch/decode | rope (135.0 us/pass) | nano_paged_kv_gather (33.7 us/pass) | qk_matmul (25.9 us/pass) |
| b1_t20_float16 | minillm/prefill | rope (211.8 us/pass) | causal_mask (24.1 us/pass) | qk_matmul (21.8 us/pass) |
| b1_t20_float16 | minillm/decode | rope (200.2 us/pass) | causal_mask (23.7 us/pass) | qk_matmul (21.5 us/pass) |
| b1_t20_float16 | nano_vllm_torch/prefill | rope (173.0 us/pass) | qk_matmul (36.6 us/pass) | causal_mask (23.6 us/pass) |
| b1_t20_float16 | nano_vllm_torch/decode | rope (168.3 us/pass) | qk_matmul (34.8 us/pass) | nano_paged_kv_gather (33.9 us/pass) |

### b1_t20_float32: minillm prefill

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 89.972 | 92.451 | 2 | 2 | 179.943 | `q_raw=[1, 4, 20, 32]; k_raw=[1, 4, 20, 32]; positions=[20]` |
| causal_mask | 11.809 | 11.843 | 2 | 2 | 23.619 | `scores=[1, 4, 20, 20]; mask=[1, 1, 20, 20]` |
| qk_matmul | 10.203 | 10.230 | 2 | 2 | 20.406 | `query=[1, 4, 20, 32]; key=[1, 4, 20, 32]` |
| mlp_fc1 | 10.068 | 10.126 | 2 | 2 | 20.136 | `hidden_states=[1, 20, 128]; weight=[512, 128]` |
| mlp_fc2 | 9.883 | 9.910 | 2 | 2 | 19.766 | `hidden_states=[1, 20, 512]; weight=[128, 512]` |
| output_projection | 9.083 | 9.120 | 2 | 2 | 18.167 | `hidden_states=[1, 20, 128]; weight=[128, 128]` |
| mlp_norm | 7.441 | 7.486 | 2 | 2 | 14.883 | `hidden_states=[1, 20, 128]` |
| qkv_linear | 6.981 | 7.198 | 2 | 2 | 13.962 | `hidden_states=[1, 20, 128]; weight=[384, 128]` |
| attention_value_matmul | 5.739 | 5.763 | 2 | 2 | 11.478 | `probabilities=[1, 4, 20, 20]; value=[1, 4, 20, 32]` |
| head_merge | 5.485 | 5.885 | 2 | 2 | 10.970 | `attention_values=[1, 4, 20, 32]` |
| attention_norm | 5.146 | 5.209 | 2 | 2 | 10.292 | `hidden_states=[1, 20, 128]` |
| lm_head | 8.003 | 11.467 | 1 | 1 | 8.003 | `hidden_states=[1, 20, 128]; weight=[512, 128]` |
| final_norm | 7.464 | 7.496 | 1 | 1 | 7.464 | `hidden_states=[1, 20, 128]` |
| softmax | 3.581 | 3.610 | 2 | 2 | 7.162 | `scores=[1, 4, 20, 20]` |
| gelu | 3.395 | 4.023 | 2 | 2 | 6.789 | `hidden_states=[1, 20, 512]` |
| embedding | 5.407 | 5.456 | 1 | 1 | 5.407 | `input_ids=[1, 20]; weight=[512, 128]` |

### b1_t20_float32: minillm decode

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 83.295 | 86.003 | 2 | 14 | 166.591 | `q_raw=[1, 4, 1, 32]; k_raw=[1, 4, 1, 32]; positions=[1]` |
| causal_mask | 11.815 | 12.776 | 2 | 14 | 23.631 | `scores=[1, 4, 1, 21]; mask=[1, 1, 1, 21]` |
| qk_matmul | 10.318 | 10.350 | 2 | 14 | 20.636 | `query=[1, 4, 1, 32]; key=[1, 4, 21, 32]` |
| native_kv_concat | 8.294 | 10.272 | 2 | 14 | 16.588 | `past_key=[1, 4, 20, 32]; new_key=[1, 4, 1, 32]; past_value=[1, 4, 20, 32]; new_value=[1, 4, 1, 32]` |
| mlp_fc2 | 8.119 | 8.136 | 2 | 14 | 16.237 | `hidden_states=[1, 1, 512]; weight=[128, 512]` |
| mlp_fc1 | 8.061 | 8.543 | 2 | 14 | 16.123 | `hidden_states=[1, 1, 128]; weight=[512, 128]` |
| attention_norm | 7.507 | 7.712 | 2 | 14 | 15.014 | `hidden_states=[1, 1, 128]` |
| mlp_norm | 7.406 | 7.451 | 2 | 14 | 14.811 | `hidden_states=[1, 1, 128]` |
| qkv_linear | 7.330 | 7.351 | 2 | 14 | 14.659 | `hidden_states=[1, 1, 128]; weight=[384, 128]` |
| output_projection | 7.320 | 7.363 | 2 | 14 | 14.640 | `hidden_states=[1, 1, 128]; weight=[128, 128]` |
| attention_value_matmul | 5.416 | 5.457 | 2 | 14 | 10.832 | `probabilities=[1, 4, 1, 21]; value=[1, 4, 21, 32]` |
| softmax | 3.769 | 3.779 | 2 | 14 | 7.538 | `scores=[1, 4, 1, 21]` |
| final_norm | 7.402 | 7.418 | 1 | 7 | 7.402 | `hidden_states=[1, 1, 128]` |
| gelu | 3.495 | 3.569 | 2 | 14 | 6.990 | `hidden_states=[1, 1, 512]` |
| embedding | 5.946 | 6.026 | 1 | 7 | 5.946 | `input_ids=[1, 1]; weight=[512, 128]` |
| lm_head | 5.713 | 5.745 | 1 | 7 | 5.713 | `hidden_states=[1, 1, 128]; weight=[512, 128]` |
| head_merge | 1.024 | 1.064 | 2 | 14 | 2.048 | `attention_values=[1, 4, 1, 32]` |

### b1_t20_float32: nano_vllm_torch prefill

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 72.052 | 75.924 | 2 | 2 | 144.105 | `q_raw=[20, 4, 32]; k_raw=[20, 4, 32]; positions=[20]` |
| qk_matmul | 13.040 | 14.267 | 2 | 2 | 26.081 | `query=[20, 4, 32]; key=[20, 4, 32]` |
| causal_mask | 11.597 | 11.777 | 2 | 2 | 23.194 | `scores=[4, 20, 20]; mask=[1, 20, 20]` |
| mlp_fc1 | 9.883 | 9.901 | 2 | 2 | 19.766 | `hidden_states=[20, 128]; weight=[512, 128]` |
| mlp_fc2 | 9.617 | 9.751 | 2 | 2 | 19.234 | `hidden_states=[20, 512]; weight=[128, 512]` |
| qkv_linear | 9.043 | 9.116 | 2 | 2 | 18.087 | `hidden_states=[20, 128]; weight=[384, 128]` |
| output_projection | 8.868 | 8.886 | 2 | 2 | 17.737 | `hidden_states=[20, 128]; weight=[128, 128]` |
| attention_value_matmul | 8.515 | 8.531 | 2 | 2 | 17.030 | `probabilities=[4, 20, 20]; value=[20, 4, 32]` |
| nano_paged_kv_store | 7.790 | 7.837 | 2 | 2 | 15.580 | `key=[20, 4, 32]; value=[20, 4, 32]; key_cache=[1, 256, 4, 32]; slot_mapping=[20]` |
| attention_norm | 7.465 | 7.623 | 2 | 2 | 14.929 | `hidden_states=[20, 128]` |
| mlp_norm | 7.395 | 7.446 | 2 | 2 | 14.791 | `hidden_states=[20, 128]` |
| attention_output_concat | 4.773 | 4.794 | 2 | 2 | 9.545 | `sequence_0=[20, 4, 32]` |
| softmax | 4.010 | 4.072 | 2 | 2 | 8.020 | `scores=[4, 20, 20]` |
| final_norm | 7.480 | 7.538 | 1 | 1 | 7.480 | `hidden_states=[20, 128]` |
| embedding | 7.236 | 8.141 | 1 | 1 | 7.236 | `input_ids=[20]; weight=[512, 128]` |
| gelu | 3.482 | 3.507 | 2 | 2 | 6.964 | `hidden_states=[20, 512]` |
| lm_head | 5.500 | 5.555 | 1 | 1 | 5.500 | `hidden_states=[1, 128]; weight=[512, 128]` |
| head_merge | 0.484 | 0.522 | 2 | 2 | 0.968 | `attention_values=[20, 4, 32]` |

### b1_t20_float32: nano_vllm_torch decode

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 67.478 | 67.604 | 2 | 14 | 134.956 | `q_raw=[1, 4, 32]; k_raw=[1, 4, 32]; positions=[1]` |
| nano_paged_kv_gather | 16.843 | 16.963 | 2 | 14 | 33.685 | `key_cache=[1, 256, 4, 32]; value_cache=[1, 256, 4, 32]; block_table=[1]` |
| qk_matmul | 12.936 | 12.978 | 2 | 14 | 25.873 | `query=[1, 4, 32]; key=[21, 4, 32]` |
| causal_mask | 11.708 | 11.778 | 2 | 14 | 23.416 | `scores=[4, 1, 21]; mask=[1, 1, 21]` |
| attention_value_matmul | 8.454 | 8.503 | 2 | 14 | 16.908 | `probabilities=[4, 1, 21]; value=[21, 4, 32]` |
| mlp_fc2 | 7.999 | 9.467 | 2 | 14 | 15.997 | `hidden_states=[1, 512]; weight=[128, 512]` |
| mlp_fc1 | 7.963 | 12.543 | 2 | 14 | 15.925 | `hidden_states=[1, 128]; weight=[512, 128]` |
| nano_paged_kv_store | 7.826 | 7.873 | 2 | 14 | 15.652 | `key=[1, 4, 32]; value=[1, 4, 32]; key_cache=[1, 256, 4, 32]; slot_mapping=[1]` |
| attention_norm | 7.458 | 7.512 | 2 | 14 | 14.917 | `hidden_states=[1, 128]` |
| mlp_norm | 7.407 | 9.073 | 2 | 14 | 14.813 | `hidden_states=[1, 128]` |
| output_projection | 7.045 | 7.087 | 2 | 14 | 14.090 | `hidden_states=[1, 128]; weight=[128, 128]` |
| qkv_linear | 7.039 | 8.107 | 2 | 14 | 14.077 | `hidden_states=[1, 128]; weight=[384, 128]` |
| attention_output_concat | 4.598 | 6.088 | 2 | 14 | 9.196 | `sequence_0=[1, 4, 32]` |
| softmax | 4.084 | 4.114 | 2 | 14 | 8.168 | `scores=[4, 1, 21]` |
| final_norm | 7.398 | 10.508 | 1 | 7 | 7.398 | `hidden_states=[1, 128]` |
| gelu | 3.525 | 3.615 | 2 | 14 | 7.050 | `hidden_states=[1, 512]` |
| embedding | 5.719 | 5.779 | 1 | 7 | 5.719 | `input_ids=[1]; weight=[512, 128]` |
| lm_head | 5.502 | 6.572 | 1 | 7 | 5.502 | `hidden_states=[1, 128]; weight=[512, 128]` |
| head_merge | 0.490 | 0.500 | 2 | 14 | 0.979 | `attention_values=[1, 4, 32]` |

### b1_t20_float16: minillm prefill

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 105.882 | 106.279 | 2 | 2 | 211.765 | `q_raw=[1, 4, 20, 32]; k_raw=[1, 4, 20, 32]; positions=[20]` |
| causal_mask | 12.040 | 12.065 | 2 | 2 | 24.080 | `scores=[1, 4, 20, 20]; mask=[1, 1, 20, 20]` |
| qk_matmul | 10.882 | 10.971 | 2 | 2 | 21.765 | `query=[1, 4, 20, 32]; key=[1, 4, 20, 32]` |
| mlp_fc1 | 8.469 | 8.526 | 2 | 2 | 16.937 | `hidden_states=[1, 20, 128]; weight=[512, 128]` |
| mlp_fc2 | 8.446 | 8.495 | 2 | 2 | 16.892 | `hidden_states=[1, 20, 512]; weight=[128, 512]` |
| qkv_linear | 7.623 | 7.682 | 2 | 2 | 15.245 | `hidden_states=[1, 20, 128]; weight=[384, 128]` |
| output_projection | 7.609 | 8.425 | 2 | 2 | 15.219 | `hidden_states=[1, 20, 128]; weight=[128, 128]` |
| mlp_norm | 7.458 | 7.480 | 2 | 2 | 14.916 | `hidden_states=[1, 20, 128]` |
| attention_norm | 7.450 | 7.475 | 2 | 2 | 14.900 | `hidden_states=[1, 20, 128]` |
| attention_value_matmul | 6.121 | 6.154 | 2 | 2 | 12.242 | `probabilities=[1, 4, 20, 20]; value=[1, 4, 20, 32]` |
| head_merge | 5.442 | 5.472 | 2 | 2 | 10.883 | `attention_values=[1, 4, 20, 32]` |
| embedding | 7.568 | 8.549 | 1 | 1 | 7.568 | `input_ids=[1, 20]; weight=[512, 128]` |
| final_norm | 7.484 | 7.527 | 1 | 1 | 7.484 | `hidden_states=[1, 20, 128]` |
| softmax | 3.627 | 3.675 | 2 | 2 | 7.254 | `scores=[1, 4, 20, 20]` |
| gelu | 3.428 | 3.436 | 2 | 2 | 6.855 | `hidden_states=[1, 20, 512]` |
| lm_head | 6.245 | 6.270 | 1 | 1 | 6.245 | `hidden_states=[1, 20, 128]; weight=[512, 128]` |

### b1_t20_float16: minillm decode

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 100.115 | 102.940 | 2 | 14 | 200.229 | `q_raw=[1, 4, 1, 32]; k_raw=[1, 4, 1, 32]; positions=[1]` |
| causal_mask | 11.845 | 11.900 | 2 | 14 | 23.690 | `scores=[1, 4, 1, 21]; mask=[1, 1, 1, 21]` |
| qk_matmul | 10.739 | 11.757 | 2 | 14 | 21.477 | `query=[1, 4, 1, 32]; key=[1, 4, 21, 32]` |
| native_kv_concat | 8.292 | 9.547 | 2 | 14 | 16.585 | `past_key=[1, 4, 20, 32]; new_key=[1, 4, 1, 32]; past_value=[1, 4, 20, 32]; new_value=[1, 4, 1, 32]` |
| mlp_fc1 | 8.140 | 10.188 | 2 | 14 | 16.281 | `hidden_states=[1, 1, 128]; weight=[512, 128]` |
| mlp_fc2 | 8.139 | 8.275 | 2 | 14 | 16.278 | `hidden_states=[1, 1, 512]; weight=[128, 512]` |
| mlp_norm | 7.483 | 7.699 | 2 | 14 | 14.965 | `hidden_states=[1, 1, 128]` |
| attention_norm | 7.432 | 7.474 | 2 | 14 | 14.865 | `hidden_states=[1, 1, 128]` |
| output_projection | 7.289 | 7.708 | 2 | 14 | 14.578 | `hidden_states=[1, 1, 128]; weight=[128, 128]` |
| qkv_linear | 7.251 | 7.285 | 2 | 14 | 14.501 | `hidden_states=[1, 1, 128]; weight=[384, 128]` |
| attention_value_matmul | 5.625 | 5.657 | 2 | 14 | 11.251 | `probabilities=[1, 4, 1, 21]; value=[1, 4, 21, 32]` |
| final_norm | 7.498 | 7.562 | 1 | 7 | 7.498 | `hidden_states=[1, 1, 128]` |
| softmax | 3.638 | 3.700 | 2 | 14 | 7.277 | `scores=[1, 4, 1, 21]` |
| gelu | 3.549 | 4.198 | 2 | 14 | 7.098 | `hidden_states=[1, 1, 512]` |
| lm_head | 5.980 | 6.164 | 1 | 7 | 5.980 | `hidden_states=[1, 1, 128]; weight=[512, 128]` |
| embedding | 5.894 | 5.925 | 1 | 7 | 5.894 | `input_ids=[1, 1]; weight=[512, 128]` |
| head_merge | 1.035 | 1.052 | 2 | 14 | 2.069 | `attention_values=[1, 4, 1, 32]` |

### b1_t20_float16: nano_vllm_torch prefill

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 86.498 | 112.465 | 2 | 2 | 172.997 | `q_raw=[20, 4, 32]; k_raw=[20, 4, 32]; positions=[20]` |
| qk_matmul | 18.318 | 20.984 | 2 | 2 | 36.635 | `query=[20, 4, 32]; key=[20, 4, 32]` |
| causal_mask | 11.824 | 12.919 | 2 | 2 | 23.648 | `scores=[4, 20, 20]; mask=[1, 20, 20]` |
| attention_value_matmul | 9.103 | 9.122 | 2 | 2 | 18.206 | `probabilities=[4, 20, 20]; value=[20, 4, 32]` |
| mlp_fc1 | 8.212 | 8.264 | 2 | 2 | 16.423 | `hidden_states=[20, 128]; weight=[512, 128]` |
| mlp_fc2 | 8.163 | 8.185 | 2 | 2 | 16.326 | `hidden_states=[20, 512]; weight=[128, 512]` |
| softmax | 7.949 | 7.974 | 2 | 2 | 15.898 | `scores=[4, 20, 20]` |
| nano_paged_kv_store | 7.821 | 7.931 | 2 | 2 | 15.641 | `key=[20, 4, 32]; value=[20, 4, 32]; key_cache=[1, 256, 4, 32]; slot_mapping=[20]` |
| attention_norm | 7.472 | 9.537 | 2 | 2 | 14.943 | `hidden_states=[20, 128]` |
| mlp_norm | 7.445 | 7.479 | 2 | 2 | 14.890 | `hidden_states=[20, 128]` |
| qkv_linear | 7.396 | 8.892 | 2 | 2 | 14.793 | `hidden_states=[20, 128]; weight=[384, 128]` |
| output_projection | 7.337 | 8.314 | 2 | 2 | 14.675 | `hidden_states=[20, 128]; weight=[128, 128]` |
| attention_output_concat | 4.847 | 4.916 | 2 | 2 | 9.694 | `sequence_0=[20, 4, 32]` |
| final_norm | 7.426 | 7.465 | 1 | 1 | 7.426 | `hidden_states=[20, 128]` |
| embedding | 7.160 | 8.314 | 1 | 1 | 7.160 | `input_ids=[20]; weight=[512, 128]` |
| gelu | 3.453 | 5.048 | 2 | 2 | 6.906 | `hidden_states=[20, 512]` |
| lm_head | 5.743 | 5.757 | 1 | 1 | 5.743 | `hidden_states=[1, 128]; weight=[512, 128]` |
| head_merge | 0.490 | 0.496 | 2 | 2 | 0.979 | `attention_values=[20, 4, 32]` |

### b1_t20_float16: nano_vllm_torch decode

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 84.136 | 86.881 | 2 | 14 | 168.271 | `q_raw=[1, 4, 32]; k_raw=[1, 4, 32]; positions=[1]` |
| qk_matmul | 17.399 | 17.558 | 2 | 14 | 34.798 | `query=[1, 4, 32]; key=[21, 4, 32]` |
| nano_paged_kv_gather | 16.930 | 16.962 | 2 | 14 | 33.860 | `key_cache=[1, 256, 4, 32]; value_cache=[1, 256, 4, 32]; block_table=[1]` |
| causal_mask | 11.770 | 11.787 | 2 | 14 | 23.540 | `scores=[4, 1, 21]; mask=[1, 1, 21]` |
| attention_value_matmul | 8.722 | 8.754 | 2 | 14 | 17.443 | `probabilities=[4, 1, 21]; value=[21, 4, 32]` |
| mlp_fc1 | 7.885 | 7.938 | 2 | 14 | 15.770 | `hidden_states=[1, 128]; weight=[512, 128]` |
| mlp_fc2 | 7.884 | 7.938 | 2 | 14 | 15.767 | `hidden_states=[1, 512]; weight=[128, 512]` |
| softmax | 7.881 | 7.915 | 2 | 14 | 15.763 | `scores=[4, 1, 21]` |
| nano_paged_kv_store | 7.826 | 7.865 | 2 | 14 | 15.653 | `key=[1, 4, 32]; value=[1, 4, 32]; key_cache=[1, 256, 4, 32]; slot_mapping=[1]` |
| attention_norm | 7.478 | 7.507 | 2 | 14 | 14.956 | `hidden_states=[1, 128]` |
| mlp_norm | 7.453 | 7.495 | 2 | 14 | 14.905 | `hidden_states=[1, 128]` |
| output_projection | 7.075 | 7.115 | 2 | 14 | 14.150 | `hidden_states=[1, 128]; weight=[128, 128]` |
| qkv_linear | 7.010 | 7.051 | 2 | 14 | 14.021 | `hidden_states=[1, 128]; weight=[384, 128]` |
| attention_output_concat | 4.549 | 4.614 | 2 | 14 | 9.098 | `sequence_0=[1, 4, 32]` |
| final_norm | 7.488 | 7.510 | 1 | 7 | 7.488 | `hidden_states=[1, 128]` |
| gelu | 3.533 | 3.547 | 2 | 14 | 7.067 | `hidden_states=[1, 512]` |
| lm_head | 5.713 | 5.740 | 1 | 7 | 5.713 | `hidden_states=[1, 128]; weight=[512, 128]` |
| embedding | 5.629 | 5.654 | 1 | 7 | 5.629 | `input_ids=[1]; weight=[512, 128]` |
| head_merge | 0.486 | 0.490 | 2 | 14 | 0.972 | `attention_values=[1, 4, 32]` |

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

## Numerical layout parity

nano-vLLM serving returns only the last prefill logit per sequence, while MiniLLM returns all positions. These checks compare equivalent last-token/decode logits.

| Workload | Phase | Max abs error | Mean abs error | Argmax match |
| --- | --- | ---: | ---: | ---: |
| b1_t20_float32 | prefill_last_token_logits | 4.76837e-07 | 8.0192e-08 | 1.000 |
| b1_t20_float32 | decode_logits | 0 | 0 | 1.000 |
| b1_t20_float16 | prefill_last_token_logits | 0.00390625 | 0.000530105 | 1.000 |
| b1_t20_float16 | decode_logits | 0.00292969 | 0.000501211 | 1.000 |

## Profiler materialization audit

Each linear, matmul, layout, and KV-cache stage gets one post-timing PyTorch profiler invocation. `clone` or `contiguous` means the eager operator materialized a new layout before or during the measured operation; profiler overhead is not part of CUDA Event timing.

| Workload | Runtime/phase | Stage | clone | contiguous | copy | bmm | Evidence |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| b1_t20_float32 | minillm/prefill | head_merge | 1 | 1 | 1 | 0 | `aten::contiguous [[1, 20, 4, 32], []]; aten::clone [[1, 20, 4, 32], []]` |
| b1_t20_float16 | minillm/prefill | head_merge | 1 | 1 | 1 | 0 | `aten::contiguous [[1, 20, 4, 32], []]; aten::clone [[1, 20, 4, 32], []]` |

## Interpretation constraints

- This checkpoint is intentionally tiny. Decode GEMMs have very small `M` and are commonly launch/latency bound rather than Tensor Core throughput bound.
- nano-vLLM's PyTorch attention is a correctness fallback that loops over sequences and gathers paged KV. FlashAttention/Triton/CUDA Graph results must be measured separately, not inferred from this table.
- Stage sums are attribution estimates. They do not include every residual/add/cast/allocation and therefore need not equal full-pass latency.
- Decode timing is for the first incremental step at the reported context. Attention and cache costs increase as more tokens are appended.
- `F.linear` and batched matmul dispatch through PyTorch's CUDA BLAS integration; exact cuBLAS versus cuBLASLt algorithm selection is internal and can change with shape, dtype, and PyTorch version.
