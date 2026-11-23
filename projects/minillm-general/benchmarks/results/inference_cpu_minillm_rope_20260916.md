# MiniLLM / nano-vLLM PyTorch inference baseline

Generated: `2026-09-16T09:21:07+0800`

## Scope and method

This is an eager PyTorch baseline on the real MiniLLM checkpoint and the tensor layouts used by MiniLLM and nano-vLLM. On CUDA, linear/matmul operations lower through cuBLAS/cuBLASLt; on CPU they use the corresponding PyTorch CPU kernels. MiniLLM QK/PV use `matmul`, while the nano-vLLM fallback uses its real `einsum` equations. FlashAttention, Triton attention, CUDA Graph, and `torch.compile` are intentionally excluded so later optimized kernels have a stable reference.

Timing uses 2 warmups and 5 samples. The adaptive inner loop targets 20.0 ms per sample. On CUDA, GPU Event time covers work on the CUDA stream; on CPU, both timing fields are synchronous monotonic wall time.

For `N` generated tokens, prefill produces token 1 and the model executes `N-1` decode passes. Component contribution is `median time for one measured stage invocation x logical calls per pass`.

## Environment

| Item | Value |
| --- | --- |
| GPU | n/a |
| Compute capability | n/a |
| PyTorch / CUDA | 2.9.1+cu130 / 13.0 |
| NVIDIA driver | n/a |
| BLAS | n/a |
| TF32 | matmul=False, cuDNN=False |
| Checkpoint SHA-256 | `85f69b910214c78537c510620f084454e26dba8bbea66506cc94664ac5f31e57` |
| Git | `5e85b94a89e2cc77a893b71fcdb3042570ea14a8`; dirty=True |

## Model and workloads

Model: `L=2, C=128, H=4, Hkv=4, D=32, V=512, block_size=128`. Prompt token count: `20`. nano-vLLM KV page size: `256`.
The decode row measures the first incremental pass: one new token is processed while attention reads the prompt plus that token. The call count for a 8-token generation is exact, but later decode passes have progressively longer KV lengths and are not assigned the first-step latency.

| Workload | MiniLLM prefill | nano prefill | MiniLLM decode | nano decode | dtype |
| --- | --- | --- | --- | --- | --- |
| b1_t20_float32 | `[1, 20, 128]` | `[20, 128]` | `[1, 1, 128]` | `[1, 128]` | float32 |

## Full-pass latency

| Workload | Runtime | Phase | CPU wall median (ms) | p90 (ms) | Wall median (ms) | Throughput token/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| b1_t20_float32 | minillm | prefill | 0.513364 | 0.515958 | 0.513364 | 38958.7 |
| b1_t20_float32 | minillm | decode | 0.297041 | 0.301270 | 0.297041 | 3366.5 |
| b1_t20_float32 | nano_vllm_torch | prefill | 0.491820 | 0.499370 | 0.491820 | 40665.3 |
| b1_t20_float32 | nano_vllm_torch | decode | 0.323572 | 0.326216 | 0.323572 | 3090.5 |

## Component hotspots

The ranking multiplies isolated median stage latency by the logical calls in one model pass. It identifies optimization candidates; it is not a claim that these isolated rows sum exactly to full-model latency.

| Workload | Runtime/phase | Top 1 | Top 2 | Top 3 |
| --- | --- | --- | --- | --- |
| b1_t20_float32 | minillm/prefill | rope (100.0 us/pass) | mlp_fc1 (59.2 us/pass) | mlp_fc2 (58.9 us/pass) |
| b1_t20_float32 | minillm/decode | rope (83.7 us/pass) | mlp_fc1 (16.9 us/pass) | mlp_fc2 (15.6 us/pass) |
| b1_t20_float32 | nano_vllm_torch/prefill | rope (65.5 us/pass) | mlp_fc2 (56.0 us/pass) | mlp_fc1 (55.5 us/pass) |
| b1_t20_float32 | nano_vllm_torch/decode | rope (55.6 us/pass) | nano_paged_kv_gather (23.3 us/pass) | qk_matmul (19.9 us/pass) |

### b1_t20_float32: minillm prefill

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 50.015 | 50.666 | 2 | 2 | 100.030 | `q_raw=[1, 4, 20, 32]; k_raw=[1, 4, 20, 32]; positions=[20]` |
| mlp_fc1 | 29.620 | 29.876 | 2 | 2 | 59.241 | `hidden_states=[1, 20, 128]; weight=[512, 128]` |
| mlp_fc2 | 29.426 | 30.071 | 2 | 2 | 58.852 | `hidden_states=[1, 20, 512]; weight=[128, 512]` |
| qkv_linear | 22.677 | 25.742 | 2 | 2 | 45.355 | `hidden_states=[1, 20, 128]; weight=[384, 128]` |
| lm_head | 26.090 | 26.558 | 1 | 1 | 26.090 | `hidden_states=[1, 20, 128]; weight=[512, 128]` |
| output_projection | 10.261 | 10.335 | 2 | 2 | 20.521 | `hidden_states=[1, 20, 128]; weight=[128, 128]` |
| qk_matmul | 7.997 | 8.436 | 2 | 2 | 15.995 | `query=[1, 4, 20, 32]; key=[1, 4, 20, 32]` |
| causal_mask | 6.586 | 6.607 | 2 | 2 | 13.172 | `scores=[1, 4, 20, 20]; mask=[1, 1, 20, 20]` |
| gelu | 6.343 | 6.580 | 2 | 2 | 12.686 | `hidden_states=[1, 20, 512]` |
| mlp_norm | 4.968 | 4.976 | 2 | 2 | 9.936 | `hidden_states=[1, 20, 128]` |
| attention_norm | 4.927 | 4.979 | 2 | 2 | 9.853 | `hidden_states=[1, 20, 128]` |
| attention_value_matmul | 4.103 | 4.109 | 2 | 2 | 8.205 | `probabilities=[1, 4, 20, 20]; value=[1, 4, 20, 32]` |
| softmax | 3.435 | 3.459 | 2 | 2 | 6.870 | `scores=[1, 4, 20, 20]` |
| head_merge | 3.265 | 3.693 | 2 | 2 | 6.530 | `attention_values=[1, 4, 20, 32]` |
| final_norm | 5.081 | 5.103 | 1 | 1 | 5.081 | `hidden_states=[1, 20, 128]` |
| embedding | 4.493 | 4.543 | 1 | 1 | 4.493 | `input_ids=[1, 20]; weight=[512, 128]` |

### b1_t20_float32: minillm decode

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 41.867 | 42.101 | 2 | 14 | 83.734 | `q_raw=[1, 4, 1, 32]; k_raw=[1, 4, 1, 32]; positions=[1]` |
| mlp_fc1 | 8.475 | 8.501 | 2 | 14 | 16.949 | `hidden_states=[1, 1, 128]; weight=[512, 128]` |
| mlp_fc2 | 7.823 | 8.224 | 2 | 14 | 15.647 | `hidden_states=[1, 1, 512]; weight=[128, 512]` |
| qkv_linear | 6.705 | 6.734 | 2 | 14 | 13.410 | `hidden_states=[1, 1, 128]; weight=[384, 128]` |
| qk_matmul | 6.114 | 6.123 | 2 | 14 | 12.228 | `query=[1, 4, 1, 32]; key=[1, 4, 21, 32]` |
| causal_mask | 5.152 | 5.495 | 2 | 14 | 10.304 | `scores=[1, 4, 1, 21]; mask=[1, 1, 1, 21]` |
| output_projection | 4.909 | 4.915 | 2 | 14 | 9.819 | `hidden_states=[1, 1, 128]; weight=[128, 128]` |
| native_kv_concat | 4.574 | 4.588 | 2 | 14 | 9.147 | `past_key=[1, 4, 20, 32]; new_key=[1, 4, 1, 32]; past_value=[1, 4, 20, 32]; new_value=[1, 4, 1, 32]` |
| gelu | 4.572 | 4.645 | 2 | 14 | 9.144 | `hidden_states=[1, 1, 512]` |
| attention_norm | 3.739 | 3.757 | 2 | 14 | 7.479 | `hidden_states=[1, 1, 128]` |
| mlp_norm | 3.706 | 4.025 | 2 | 14 | 7.411 | `hidden_states=[1, 1, 128]` |
| lm_head | 5.807 | 5.833 | 1 | 7 | 5.807 | `hidden_states=[1, 1, 128]; weight=[512, 128]` |
| attention_value_matmul | 2.511 | 2.513 | 2 | 14 | 5.022 | `probabilities=[1, 4, 1, 21]; value=[1, 4, 21, 32]` |
| embedding | 4.105 | 4.149 | 1 | 7 | 4.105 | `input_ids=[1, 1]; weight=[512, 128]` |
| final_norm | 3.806 | 3.823 | 1 | 7 | 3.806 | `hidden_states=[1, 1, 128]` |
| head_merge | 1.564 | 1.590 | 2 | 14 | 3.129 | `attention_values=[1, 4, 1, 32]` |
| softmax | 1.071 | 1.093 | 2 | 14 | 2.141 | `scores=[1, 4, 1, 21]` |

### b1_t20_float32: nano_vllm_torch prefill

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 32.748 | 32.859 | 2 | 2 | 65.495 | `q_raw=[20, 4, 32]; k_raw=[20, 4, 32]; positions=[20]` |
| mlp_fc2 | 28.009 | 28.751 | 2 | 2 | 56.017 | `hidden_states=[20, 512]; weight=[128, 512]` |
| mlp_fc1 | 27.759 | 27.981 | 2 | 2 | 55.519 | `hidden_states=[20, 128]; weight=[512, 128]` |
| qkv_linear | 20.895 | 21.053 | 2 | 2 | 41.791 | `hidden_states=[20, 128]; weight=[384, 128]` |
| qk_matmul | 12.320 | 12.546 | 2 | 2 | 24.641 | `query=[20, 4, 32]; key=[20, 4, 32]` |
| output_projection | 9.574 | 9.681 | 2 | 2 | 19.148 | `hidden_states=[20, 128]; weight=[128, 128]` |
| attention_value_matmul | 8.448 | 8.468 | 2 | 2 | 16.897 | `probabilities=[4, 20, 20]; value=[20, 4, 32]` |
| nano_paged_kv_store | 6.758 | 6.992 | 2 | 2 | 13.515 | `key=[20, 4, 32]; value=[20, 4, 32]; key_cache=[1, 256, 4, 32]; slot_mapping=[20]` |
| gelu | 6.130 | 6.182 | 2 | 2 | 12.260 | `hidden_states=[20, 512]` |
| mlp_norm | 4.790 | 4.809 | 2 | 2 | 9.580 | `hidden_states=[20, 128]` |
| attention_norm | 4.751 | 4.774 | 2 | 2 | 9.503 | `hidden_states=[20, 128]` |
| causal_mask | 4.405 | 4.763 | 2 | 2 | 8.809 | `scores=[4, 20, 20]; mask=[1, 20, 20]` |
| softmax | 4.091 | 4.178 | 2 | 2 | 8.183 | `scores=[4, 20, 20]` |
| lm_head | 5.550 | 5.637 | 1 | 1 | 5.550 | `hidden_states=[1, 128]; weight=[512, 128]` |
| final_norm | 4.943 | 4.969 | 1 | 1 | 4.943 | `hidden_states=[20, 128]` |
| attention_output_concat | 2.413 | 2.510 | 2 | 2 | 4.826 | `sequence_0=[20, 4, 32]` |
| embedding | 3.807 | 3.833 | 1 | 1 | 3.807 | `input_ids=[20]; weight=[512, 128]` |
| head_merge | 0.710 | 0.728 | 2 | 2 | 1.420 | `attention_values=[20, 4, 32]` |

### b1_t20_float32: nano_vllm_torch decode

| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rope | 27.823 | 28.046 | 2 | 14 | 55.646 | `q_raw=[1, 4, 32]; k_raw=[1, 4, 32]; positions=[1]` |
| nano_paged_kv_gather | 11.660 | 11.881 | 2 | 14 | 23.321 | `key_cache=[1, 256, 4, 32]; value_cache=[1, 256, 4, 32]; block_table=[1]` |
| qk_matmul | 9.957 | 10.012 | 2 | 14 | 19.915 | `query=[1, 4, 32]; key=[21, 4, 32]` |
| mlp_fc1 | 8.028 | 8.058 | 2 | 14 | 16.055 | `hidden_states=[1, 128]; weight=[512, 128]` |
| mlp_fc2 | 7.445 | 7.456 | 2 | 14 | 14.890 | `hidden_states=[1, 512]; weight=[128, 512]` |
| attention_value_matmul | 7.199 | 7.215 | 2 | 14 | 14.397 | `probabilities=[4, 1, 21]; value=[21, 4, 32]` |
| qkv_linear | 6.249 | 6.285 | 2 | 14 | 12.498 | `hidden_states=[1, 128]; weight=[384, 128]` |
| output_projection | 4.534 | 4.555 | 2 | 14 | 9.067 | `hidden_states=[1, 128]; weight=[128, 128]` |
| gelu | 4.534 | 6.751 | 2 | 14 | 9.067 | `hidden_states=[1, 512]` |
| nano_paged_kv_store | 4.456 | 4.477 | 2 | 14 | 8.911 | `key=[1, 4, 32]; value=[1, 4, 32]; key_cache=[1, 256, 4, 32]; slot_mapping=[1]` |
| attention_norm | 3.758 | 3.777 | 2 | 14 | 7.516 | `hidden_states=[1, 128]` |
| mlp_norm | 3.713 | 3.724 | 2 | 14 | 7.426 | `hidden_states=[1, 128]` |
| causal_mask | 3.570 | 3.580 | 2 | 14 | 7.141 | `scores=[4, 1, 21]; mask=[1, 1, 21]` |
| lm_head | 5.450 | 5.493 | 1 | 7 | 5.450 | `hidden_states=[1, 128]; weight=[512, 128]` |
| final_norm | 3.796 | 3.821 | 1 | 7 | 3.796 | `hidden_states=[1, 128]` |
| embedding | 3.672 | 4.010 | 1 | 7 | 3.672 | `input_ids=[1]; weight=[512, 128]` |
| softmax | 1.771 | 1.785 | 2 | 14 | 3.541 | `scores=[4, 1, 21]` |
| attention_output_concat | 1.154 | 1.175 | 2 | 14 | 2.307 | `sequence_0=[1, 4, 32]` |
| head_merge | 0.715 | 0.721 | 2 | 14 | 1.430 | `attention_values=[1, 4, 32]` |

## Canonical tensor layouts

The pointer alignment is the maximum power-of-two alignment observed for that concrete allocation. `all-row alignment` additionally requires every active outer stride to preserve that alignment, so it is the guarantee for all last-dimension row starts. Both values are diagnostic for this run, not permanent allocator guarantees.

| Workload | Runtime/phase | Tensor | Shape | Stride | dtype | Contiguous | Ptr / all-row alignment | mod 128 / 256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_t20_float32 | minillm/prefill | input_ids | `[1, 20]` | `[20, 1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | minillm/prefill | hidden | `[1, 20, 128]` | `[2560, 128, 1]` | float32 | True | 128 / 128 B | 0 / 128 |
| b1_t20_float32 | minillm/prefill | q_raw | `[1, 4, 20, 32]` | `[7680, 32, 384, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | minillm/prefill | k_raw | `[1, 4, 20, 32]` | `[7680, 32, 384, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | minillm/prefill | v_raw | `[1, 4, 20, 32]` | `[7680, 32, 384, 1]` | float32 | False | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | minillm/prefill | query | `[1, 4, 20, 32]` | `[2560, 640, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | minillm/prefill | key | `[1, 4, 20, 32]` | `[2560, 640, 32, 1]` | float32 | True | 128 / 128 B | 0 / 128 |
| b1_t20_float32 | minillm/prefill | scores | `[1, 4, 20, 20]` | `[1600, 400, 20, 1]` | float32 | True | 256 / 16 B | 0 / 0 |
| b1_t20_float32 | minillm/prefill | logits | `[1, 20, 512]` | `[10240, 512, 1]` | float32 | True | 64 / 64 B | 64 / 192 |
| b1_t20_float32 | minillm/decode | input_ids | `[1, 1]` | `[1, 1]` | int64 | True | 128 / 128 B | 0 / 128 |
| b1_t20_float32 | minillm/decode | hidden | `[1, 1, 128]` | `[128, 128, 1]` | float32 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | minillm/decode | q_raw | `[1, 4, 1, 32]` | `[128, 32, 128, 1]` | float32 | True | 64 / 64 B | 64 / 64 |
| b1_t20_float32 | minillm/decode | k_raw | `[1, 4, 1, 32]` | `[128, 32, 128, 1]` | float32 | True | 64 / 64 B | 64 / 64 |
| b1_t20_float32 | minillm/decode | v_raw | `[1, 4, 1, 32]` | `[128, 32, 128, 1]` | float32 | True | 64 / 64 B | 64 / 64 |
| b1_t20_float32 | minillm/decode | query | `[1, 4, 1, 32]` | `[128, 32, 32, 1]` | float32 | True | 64 / 64 B | 64 / 192 |
| b1_t20_float32 | minillm/decode | key | `[1, 4, 21, 32]` | `[2688, 672, 32, 1]` | float32 | True | 64 / 64 B | 64 / 64 |
| b1_t20_float32 | minillm/decode | scores | `[1, 4, 1, 21]` | `[84, 21, 21, 1]` | float32 | True | 128 / 4 B | 0 / 128 |
| b1_t20_float32 | minillm/decode | logits | `[1, 1, 512]` | `[512, 512, 1]` | float32 | True | 64 / 64 B | 64 / 64 |
| b1_t20_float32 | nano_vllm_torch/prefill | input_ids | `[20]` | `[1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/prefill | hidden | `[20, 128]` | `[128, 1]` | float32 | True | 64 / 64 B | 64 / 64 |
| b1_t20_float32 | nano_vllm_torch/prefill | q_raw | `[20, 4, 32]` | `[384, 32, 1]` | float32 | False | 64 / 64 B | 64 / 192 |
| b1_t20_float32 | nano_vllm_torch/prefill | k_raw | `[20, 4, 32]` | `[384, 32, 1]` | float32 | False | 64 / 64 B | 64 / 192 |
| b1_t20_float32 | nano_vllm_torch/prefill | v_raw | `[20, 4, 32]` | `[384, 32, 1]` | float32 | False | 64 / 64 B | 64 / 192 |
| b1_t20_float32 | nano_vllm_torch/prefill | query | `[20, 4, 32]` | `[128, 32, 1]` | float32 | True | 128 / 128 B | 0 / 128 |
| b1_t20_float32 | nano_vllm_torch/prefill | key | `[20, 4, 32]` | `[128, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/prefill | scores | `[4, 20, 20]` | `[400, 20, 1]` | float32 | True | 128 / 16 B | 0 / 128 |
| b1_t20_float32 | nano_vllm_torch/prefill | key_cache | `[1, 256, 4, 32]` | `[32768, 128, 32, 1]` | float32 | True | 64 / 64 B | 64 / 64 |
| b1_t20_float32 | nano_vllm_torch/prefill | slot_mapping | `[20]` | `[1]` | int64 | True | 64 / 64 B | 64 / 192 |
| b1_t20_float32 | nano_vllm_torch/prefill | head_input | `[1, 128]` | `[128, 1]` | float32 | True | 64 / 64 B | 64 / 192 |
| b1_t20_float32 | nano_vllm_torch/prefill | logits | `[1, 512]` | `[512, 1]` | float32 | True | 64 / 64 B | 64 / 64 |
| b1_t20_float32 | nano_vllm_torch/decode | input_ids | `[1]` | `[1]` | int64 | True | 128 / 128 B | 0 / 128 |
| b1_t20_float32 | nano_vllm_torch/decode | hidden | `[1, 128]` | `[128, 1]` | float32 | True | 128 / 128 B | 0 / 128 |
| b1_t20_float32 | nano_vllm_torch/decode | q_raw | `[1, 4, 32]` | `[128, 32, 1]` | float32 | True | 64 / 64 B | 64 / 64 |
| b1_t20_float32 | nano_vllm_torch/decode | k_raw | `[1, 4, 32]` | `[128, 32, 1]` | float32 | True | 64 / 64 B | 64 / 64 |
| b1_t20_float32 | nano_vllm_torch/decode | v_raw | `[1, 4, 32]` | `[128, 32, 1]` | float32 | True | 64 / 64 B | 64 / 64 |
| b1_t20_float32 | nano_vllm_torch/decode | query | `[1, 4, 32]` | `[128, 32, 1]` | float32 | True | 64 / 64 B | 64 / 192 |
| b1_t20_float32 | nano_vllm_torch/decode | key | `[1, 4, 32]` | `[128, 32, 1]` | float32 | True | 256 / 128 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/decode | scores | `[4, 1, 21]` | `[21, 21, 1]` | float32 | True | 128 / 4 B | 0 / 128 |
| b1_t20_float32 | nano_vllm_torch/decode | key_cache | `[1, 256, 4, 32]` | `[32768, 128, 32, 1]` | float32 | True | 64 / 64 B | 64 / 64 |
| b1_t20_float32 | nano_vllm_torch/decode | slot_mapping | `[1]` | `[1]` | int64 | True | 256 / 256 B | 0 / 0 |
| b1_t20_float32 | nano_vllm_torch/decode | head_input | `[1, 128]` | `[128, 1]` | float32 | True | 128 / 128 B | 0 / 128 |
| b1_t20_float32 | nano_vllm_torch/decode | logits | `[1, 512]` | `[512, 1]` | float32 | True | 256 / 256 B | 0 / 0 |

## Numerical layout parity

nano-vLLM serving returns only the last prefill logit per sequence, while MiniLLM returns all positions. These checks compare equivalent last-token/decode logits.

| Workload | Phase | Max abs error | Mean abs error | Argmax match |
| --- | --- | ---: | ---: | ---: |
| b1_t20_float32 | prefill_last_token_logits | 2.38419e-06 | 2.78032e-07 | 1.000 |
| b1_t20_float32 | decode_logits | 7.7486e-07 | 1.67602e-07 | 1.000 |

## Profiler materialization audit

Each linear, matmul, layout, and KV-cache stage gets one post-timing PyTorch profiler invocation. `clone` or `contiguous` means the eager operator materialized a new layout before or during the measured operation; profiler overhead is not part of CUDA Event timing.

| Workload | Runtime/phase | Stage | clone | contiguous | copy | bmm | Evidence |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| all | all | none observed | 0 | 0 | 0 | 0 | n/a |

## Interpretation constraints

- This checkpoint is intentionally tiny. Decode GEMMs have very small `M` and are commonly launch/latency bound rather than Tensor Core throughput bound.
- nano-vLLM's PyTorch attention is a correctness fallback that loops over sequences and gathers paged KV. FlashAttention/Triton/CUDA Graph results must be measured separately, not inferred from this table.
- Stage sums are attribution estimates. They do not include every residual/add/cast/allocation and therefore need not equal full-pass latency.
- Decode timing is for the first incremental step at the reported context. Attention and cache costs increase as more tokens are appended.
- `F.linear` and batched matmul dispatch through PyTorch's CUDA BLAS integration; exact cuBLAS versus cuBLASLt algorithm selection is internal and can change with shape, dtype, and PyTorch version.
