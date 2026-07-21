from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ._common import check_cuda_contiguous, check_float_tensor


@triton.jit
def _flash_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_on,
    stride_od,
    num_heads: tl.constexpr,
    sequence_length: tl.constexpr,
    head_dim: tl.constexpr,
    scale,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads

    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    dim_offsets = tl.arange(0, head_dim)

    q_base = q_ptr + batch * stride_qb + head * stride_qh
    k_base = k_ptr + batch * stride_kb + head * stride_kh
    v_base = v_ptr + batch * stride_vb + head * stride_vh
    o_base = output_ptr + batch * stride_ob + head * stride_oh

    q = tl.load(
        q_base + query_offsets[:, None] * stride_qn + dim_offsets[None, :] * stride_qd,
        mask=query_offsets[:, None] < sequence_length,
        other=0.0,
    )
    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, head_dim), tl.float32)

    for key_start in range(0, sequence_length, BLOCK_N):
        current_keys = key_start + key_offsets
        k = tl.load(
            k_base + current_keys[None, :] * stride_kn + dim_offsets[:, None] * stride_kd,
            mask=current_keys[None, :] < sequence_length,
            other=0.0,
        )
        scores = tl.dot(q, k) * scale
        score_mask = (query_offsets[:, None] < sequence_length) & (current_keys[None, :] < sequence_length)
        if CAUSAL:
            score_mask &= current_keys[None, :] <= query_offsets[:, None]
        scores = tl.where(score_mask, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        correction = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        block_sum = tl.sum(probabilities, axis=1)
        accumulator *= correction[:, None]

        v = tl.load(
            v_base + current_keys[:, None] * stride_vn + dim_offsets[None, :] * stride_vd,
            mask=current_keys[:, None] < sequence_length,
            other=0.0,
        )
        accumulator += tl.dot(probabilities.to(v.dtype), v)
        running_sum = running_sum * correction + block_sum
        running_max = new_max

    accumulator /= running_sum[:, None]
    output_offsets = query_offsets[:, None] * stride_on + dim_offsets[None, :] * stride_od
    tl.store(o_base + output_offsets, accumulator, mask=query_offsets[:, None] < sequence_length)


@triton.jit
def _nms_suppress_kernel(boxes_ptr, alive_ptr, selected_position, num_boxes, threshold, BLOCK_SIZE: tl.constexpr):
    positions = selected_position + 1 + tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = positions < num_boxes
    selected = boxes_ptr + selected_position * 4
    selected_x1 = tl.load(selected)
    selected_y1 = tl.load(selected + 1)
    selected_x2 = tl.load(selected + 2)
    selected_y2 = tl.load(selected + 3)

    boxes = boxes_ptr + positions * 4
    x1 = tl.load(boxes, mask=mask)
    y1 = tl.load(boxes + 1, mask=mask)
    x2 = tl.load(boxes + 2, mask=mask)
    y2 = tl.load(boxes + 3, mask=mask)
    intersection_width = tl.maximum(0.0, tl.minimum(x2, selected_x2) - tl.maximum(x1, selected_x1))
    intersection_height = tl.maximum(0.0, tl.minimum(y2, selected_y2) - tl.maximum(y1, selected_y1))
    intersection = intersection_width * intersection_height
    selected_area = tl.maximum(0.0, selected_x2 - selected_x1) * tl.maximum(0.0, selected_y2 - selected_y1)
    area = tl.maximum(0.0, x2 - x1) * tl.maximum(0.0, y2 - y1)
    iou = intersection / tl.maximum(selected_area + area - intersection, 1e-12)
    is_alive = tl.load(alive_ptr + positions, mask=mask, other=0)
    tl.store(alive_ptr + positions, tl.where(iou > threshold, 0, is_alive), mask=mask)


@triton.jit
def _merge_attention_states_kernel(
    prefix_output_ptr,
    prefix_lse_ptr,
    suffix_output_ptr,
    suffix_lse_ptr,
    output_ptr,
    output_lse_ptr,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    prefix_lse = tl.load(prefix_lse_ptr + head * num_tokens + token)
    suffix_lse = tl.load(suffix_lse_ptr + head * num_tokens + token)
    prefix_lse = tl.where(prefix_lse == float("inf"), -float("inf"), prefix_lse)
    suffix_lse = tl.where(suffix_lse == float("inf"), -float("inf"), suffix_lse)
    maximum = tl.maximum(prefix_lse, suffix_lse)
    prefix_exp = tl.exp(prefix_lse - maximum)
    suffix_exp = tl.exp(suffix_lse - maximum)
    denominator = prefix_exp + suffix_exp
    tl.store(output_lse_ptr + head * num_tokens + token, tl.log(denominator) + maximum)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < head_dim
    base = token * num_heads * head_dim + head * head_dim
    prefix_output = tl.load(prefix_output_ptr + base + offsets, mask=mask)
    suffix_output = tl.load(suffix_output_ptr + base + offsets, mask=mask)
    merged = prefix_output * (prefix_exp / denominator) + suffix_output * (suffix_exp / denominator)
    tl.store(output_ptr + base + offsets, merged, mask=mask)


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    check_cuda_contiguous(query, key, value)
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have the same [B, H, N, D] shape")
    if query.ndim != 4:
        raise ValueError("flash_attention expects [batch, heads, sequence, head_dim]")
    if query.dtype not in (torch.float16, torch.bfloat16) or key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("flash_attention expects matching float16 or bfloat16 tensors")
    batch, heads, sequence_length, head_dim = query.shape
    if head_dim not in (16, 32, 64, 128):
        raise ValueError("head_dim must be one of 16, 32, 64, or 128")
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    output = torch.empty_like(query)
    block_m = 64
    block_n = 64
    grid = (triton.cdiv(sequence_length, block_m), batch * heads)
    _flash_attention_kernel[grid](
        query,
        key,
        value,
        output,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output.stride(),
        num_heads=heads,
        sequence_length=sequence_length,
        head_dim=head_dim,
        scale=scale,
        CAUSAL=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4 if head_dim <= 64 else 8,
        num_stages=3,
    )
    return output


def merge_attention_states(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    *,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    check_cuda_contiguous(prefix_output, prefix_lse, suffix_output, suffix_lse)
    if prefix_output.shape != suffix_output.shape or prefix_lse.shape != suffix_lse.shape:
        raise ValueError("prefix and suffix states must have matching shapes")
    if prefix_output.ndim != 3:
        raise ValueError("attention outputs must have shape [tokens, heads, head_dim]")
    num_tokens, num_heads, head_dim = prefix_output.shape
    if prefix_lse.shape != (num_heads, num_tokens):
        raise ValueError("LSE tensors must have shape [heads, tokens]")
    output = torch.empty_like(prefix_output)
    output_lse = torch.empty_like(prefix_lse)
    block_size = triton.next_power_of_2(head_dim)
    _merge_attention_states_kernel[(num_tokens, num_heads)](
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        output,
        output_lse,
        num_tokens=num_tokens,
        num_heads=num_heads,
        head_dim=head_dim,
        BLOCK_SIZE=block_size,
        num_warps=min(8, max(1, block_size // 32)),
    )
    return (output, output_lse) if return_lse else output


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    check_cuda_contiguous(boxes, scores)
    check_float_tensor(boxes)
    if boxes.dtype != torch.float32 or scores.dtype != torch.float32:
        raise TypeError("nms expects float32 boxes and scores")
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (boxes.shape[0],):
        raise ValueError("expected boxes [N, 4] and scores [N]")

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_boxes = boxes[order].contiguous()
    alive = torch.ones((boxes.shape[0],), device=boxes.device, dtype=torch.int8)
    selected_positions: list[int] = []
    block_size = 256
    for position in range(boxes.shape[0]):
        if not bool(alive[position].item()):
            continue
        selected_positions.append(position)
        remaining = boxes.shape[0] - position - 1
        if remaining:
            _nms_suppress_kernel[(triton.cdiv(remaining, block_size),)](
                sorted_boxes,
                alive,
                position,
                boxes.shape[0],
                iou_threshold,
                BLOCK_SIZE=block_size,
                num_warps=4,
            )
    positions = torch.tensor(selected_positions, device=boxes.device, dtype=torch.int64)
    return order[positions]
