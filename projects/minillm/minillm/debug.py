from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .model import CausalSelfAttention, MiniGPT


def _snapshot(value: torch.Tensor) -> torch.Tensor:
    """Detach a tensor and move a stable copy to CPU for reports/tests."""

    return value.detach().to(dtype=torch.float32, device="cpu").clone()


def split_qkv_parameters(
    attention: CausalSelfAttention,
) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
    """Expose the three logical Q/K/V Linear parameter pairs.

    This returns the same interface for both MiniLLM layouts: three explicit
    modules in teaching mode, or three row slices of production-style c_attn.
    """

    if attention.c_attn is not None:
        weight = attention.c_attn.weight
        bias = attention.c_attn.bias
        result: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        start = 0
        for name, size in zip(
            ("q", "k", "v"),
            (attention.q_size, attention.kv_size, attention.kv_size),
        ):
            stop = start + size
            part_bias = None if bias is None else bias[start:stop]
            result[name] = (weight[start:stop], part_bias)
            start = stop
        return result

    if attention.q_proj is None or attention.k_proj is None or attention.v_proj is None:
        raise RuntimeError("attention has neither fused nor separate Q/K/V projections")
    return {
        "q": (attention.q_proj.weight, attention.q_proj.bias),
        "k": (attention.k_proj.weight, attention.k_proj.bias),
        "v": (attention.v_proj.weight, attention.v_proj.bias),
    }


@torch.no_grad()
def trace_forward(
    model: MiniGPT,
    input_ids: torch.Tensor,
    targets: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Re-run MiniGPT's eval forward pass while preserving every teaching tensor.

    This deliberately spells out the same operations as ``MiniGPT.forward``.
    The returned checks compare the reconstructed result with the production
    forward method, so the teaching trace cannot silently drift from the model.
    Dropout must be disabled via ``model.eval()`` for a deterministic trace.
    """

    if model.training:
        raise ValueError("trace_forward requires model.eval() so dropout is deterministic")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, seq_len]")

    batch, seq_len = input_ids.shape
    if seq_len > model.config.block_size:
        raise ValueError("trace sequence exceeds model block_size")
    if targets is not None and targets.shape != input_ids.shape:
        raise ValueError("targets must have the same [batch, seq_len] shape as input_ids")

    positions = torch.arange(seq_len, device=input_ids.device)
    token_embedding = model.token_embedding(input_ids)
    position_embedding = None
    x = token_embedding
    if model.position_embedding is not None:
        position_embedding = model.position_embedding(positions)
        x = x + position_embedding
    embedding_sum = x
    x = model.drop(x)

    trace: dict[str, Any] = {
        "input_ids": input_ids.detach().cpu().clone(),
        "targets": None if targets is None else targets.detach().cpu().clone(),
        "positions": positions.detach().cpu().clone(),
        "token_embedding": _snapshot(token_embedding),
        "position_embedding": None if position_embedding is None else _snapshot(position_embedding),
        "embedding_sum": _snapshot(embedding_sum),
        "embedding_after_dropout": _snapshot(x),
        "blocks": [],
        "checks": {},
    }

    all_future_weights_zero = True
    all_attention_rows_sum_to_one = True
    all_qkv_slices_match = True

    for layer_index, block in enumerate(model.blocks):
        residual_before_attention = x
        ln_1 = block.ln_1(x)
        batch_size, current_len, channels = ln_1.shape

        q_flat, k_flat, v_flat = block.attn.project_qkv(ln_1)
        qkv_concatenated = torch.cat((q_flat, k_flat, v_flat), dim=-1)

        qkv_parameters = split_qkv_parameters(block.attn)
        manual_parts: dict[str, torch.Tensor] = {}
        for name, (weight, bias) in qkv_parameters.items():
            manual_parts[name] = F.linear(ln_1, weight, bias)
        all_qkv_slices_match = all_qkv_slices_match and all(
            torch.allclose(actual, manual_parts[name], rtol=1e-6, atol=1e-7)
            for name, actual in (("q", q_flat), ("k", k_flat), ("v", v_flat))
        )

        q_heads_before_position = q_flat.view(
            batch_size, current_len, block.attn.n_head, block.attn.head_dim
        ).transpose(1, 2)
        k_heads_before_position = k_flat.view(
            batch_size,
            current_len,
            block.attn.num_key_value_heads,
            block.attn.head_dim,
        ).transpose(1, 2)
        v_heads_grouped = v_flat.view(
            batch_size,
            current_len,
            block.attn.num_key_value_heads,
            block.attn.head_dim,
        ).transpose(1, 2)

        q_heads = q_heads_before_position
        k_heads_grouped = k_heads_before_position
        q_heads, k_heads_grouped = block.attn.position_encoding.apply_qk(
            q_heads, k_heads_grouped, positions
        )
        k_heads = block.attn.expand_kv_heads(k_heads_grouped)
        v_heads = block.attn.expand_kv_heads(v_heads_grouped)

        raw_scores = q_heads @ k_heads.transpose(-2, -1)
        scaled_scores_without_position = raw_scores / math.sqrt(block.attn.head_dim)
        position_bias = block.attn.position_encoding.attention_bias(
            positions,
            positions,
            dtype=scaled_scores_without_position.dtype,
            device=scaled_scores_without_position.device,
        )
        scaled_scores = scaled_scores_without_position
        if position_bias is not None:
            scaled_scores = scaled_scores + position_bias
        causal_mask = block.attn.causal_mask[:, :, :current_len, :current_len]
        masked_scores = scaled_scores.masked_fill(causal_mask == 0, float("-inf"))
        attention_weights_before_dropout = F.softmax(masked_scores, dim=-1)
        attention_weights = block.attn.attn_dropout(attention_weights_before_dropout)
        context_per_head = attention_weights @ v_heads
        concatenated_heads = (
            context_per_head.transpose(1, 2)
            .contiguous()
            .view(batch_size, current_len, channels)
        )
        attention_projection = block.attn.c_proj(concatenated_heads)
        attention_branch = block.attn.resid_dropout(attention_projection)
        x_after_attention = residual_before_attention + attention_branch

        ln_2 = block.ln_2(x_after_attention)
        mlp_expanded = block.mlp.net[0](ln_2)
        mlp_activated = block.mlp.net[1](mlp_expanded)
        mlp_projected = block.mlp.net[2](mlp_activated)
        mlp_branch = block.mlp.net[3](mlp_projected)
        x_after_mlp = x_after_attention + mlp_branch

        future_mask = torch.triu(
            torch.ones(current_len, current_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        future_weights = attention_weights_before_dropout[..., future_mask]
        all_future_weights_zero = all_future_weights_zero and bool(
            torch.count_nonzero(future_weights).item() == 0
        )
        row_sums = attention_weights_before_dropout.sum(dim=-1)
        all_attention_rows_sum_to_one = all_attention_rows_sum_to_one and bool(
            torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-6, atol=1e-6)
        )

        trace["blocks"].append(
            {
                "layer_index": layer_index,
                "residual_before_attention": _snapshot(residual_before_attention),
                "ln_1": _snapshot(ln_1),
                "qkv_concatenated": _snapshot(qkv_concatenated),
                "q_flat": _snapshot(q_flat),
                "k_flat": _snapshot(k_flat),
                "v_flat": _snapshot(v_flat),
                "q_heads_before_position": _snapshot(q_heads_before_position),
                "k_heads_before_position": _snapshot(k_heads_before_position),
                "q_heads": _snapshot(q_heads),
                "k_heads_grouped": _snapshot(k_heads_grouped),
                "v_heads_grouped": _snapshot(v_heads_grouped),
                "k_heads": _snapshot(k_heads),
                "v_heads": _snapshot(v_heads),
                "raw_scores": _snapshot(raw_scores),
                "scaled_scores_without_position": _snapshot(scaled_scores_without_position),
                "position_bias": None if position_bias is None else _snapshot(position_bias),
                "scaled_scores": _snapshot(scaled_scores),
                "causal_mask": causal_mask.detach().to(dtype=torch.int64, device="cpu").clone(),
                "masked_scores": _snapshot(masked_scores),
                "attention_weights": _snapshot(attention_weights_before_dropout),
                "context_per_head": _snapshot(context_per_head),
                "concatenated_heads": _snapshot(concatenated_heads),
                "attention_projection": _snapshot(attention_projection),
                "attention_branch": _snapshot(attention_branch),
                "x_after_attention": _snapshot(x_after_attention),
                "ln_2": _snapshot(ln_2),
                "mlp_expanded": _snapshot(mlp_expanded),
                "mlp_activated": _snapshot(mlp_activated),
                "mlp_projected": _snapshot(mlp_projected),
                "mlp_branch": _snapshot(mlp_branch),
                "x_after_mlp": _snapshot(x_after_mlp),
            }
        )
        x = x_after_mlp

    final_norm = model.ln_f(x)
    logits = model.lm_head(final_norm)
    probabilities = F.softmax(logits, dim=-1)
    loss = None
    per_token_loss = None
    if targets is not None:
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view(batch, seq_len)
        loss = per_token_loss.mean()

    reference_logits, reference_loss = model(input_ids, targets)
    logits_match = bool(torch.allclose(logits, reference_logits, rtol=1e-6, atol=1e-7))
    loss_match = loss is None or bool(
        reference_loss is not None and torch.allclose(loss, reference_loss, rtol=1e-6, atol=1e-7)
    )

    trace.update(
        {
            "final_norm": _snapshot(final_norm),
            "logits": _snapshot(logits),
            "probabilities": _snapshot(probabilities),
            "per_token_loss": None if per_token_loss is None else _snapshot(per_token_loss),
            "loss": None if loss is None else float(loss.item()),
        }
    )
    trace["checks"] = {
        "trace_logits_match_model_forward": logits_match,
        "trace_loss_matches_model_forward": loss_match,
        "qkv_outputs_match_projection_parameters": all_qkv_slices_match,
        "causal_future_attention_is_zero": all_future_weights_zero,
        "attention_rows_sum_to_one": all_attention_rows_sum_to_one,
    }
    return trace


def tensor_change(before: torch.Tensor, after: torch.Tensor) -> dict[str, float]:
    """Summarize how a traced tensor changed after an optimizer update/training."""

    if before.shape != after.shape:
        raise ValueError(f"shape changed from {tuple(before.shape)} to {tuple(after.shape)}")
    delta = after.float() - before.float()
    return {
        "l2": float(delta.norm().item()),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
    }
