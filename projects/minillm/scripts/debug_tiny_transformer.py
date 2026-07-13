from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minillm import CharTokenizer, GPTConfig, MiniGPT  # noqa: E402
from minillm.data import get_batch, read_text  # noqa: E402
from minillm.debug import tensor_change, trace_forward  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and record a complete tiny MiniLLM training/debug pipeline."
    )
    parser.add_argument("--data", default=str(ROOT / "data" / "debug_corpus.txt"))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "debug_outputs" / "tiny_transformer"),
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--report-interval", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--trace-length", type=int, default=5)
    parser.add_argument("--n-embd", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument(
        "--fused-qkv",
        action="store_true",
        help="Use production-style c_attn; default uses separate q_proj/k_proj/v_proj for teaching.",
    )
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--position-encoding",
        default="rope",
        choices=["learned", "rope"],
    )
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    return parser.parse_args()


def display_token(token: str) -> str:
    return {"\n": "\\n", " ": "<space>", "\t": "\\t"}.get(token, token)


def code_block(text: str, language: str = "text") -> str:
    return f"```{language}\n{text}\n```"


def tensor_stats(tensor: torch.Tensor) -> str:
    value = tensor.detach().float().cpu()
    finite = value[torch.isfinite(value)]
    if finite.numel() == 0:
        return f"shape={tuple(value.shape)}, finite=0/{value.numel()}"
    return (
        f"shape={tuple(value.shape)}, min={finite.min().item():.5f}, "
        f"max={finite.max().item():.5f}, mean={finite.mean().item():.5f}, "
        f"std={finite.std(unbiased=False).item():.5f}, "
        f"finite={finite.numel()}/{value.numel()}"
    )


def tensor_section(title: str, tensor: torch.Tensor, explanation: str) -> str:
    value = tensor.detach().float().cpu()
    return "\n".join(
        [
            f"**{title}** — {explanation}",
            "",
            f"`{tensor_stats(value)}`",
            "",
            code_block(repr(value)),
        ]
    )


def head_tensor_sections(title: str, tensor: torch.Tensor, explanation: str) -> str:
    value = tensor.detach().float().cpu()
    if value.ndim != 4 or value.size(0) != 1:
        return tensor_section(title, value, explanation)
    parts = [f"**{title}** — {explanation}", "", f"`{tensor_stats(value)}`"]
    for head in range(value.size(1)):
        parts.extend(["", f"head {head}:", "", code_block(repr(value[0, head]))])
    return "\n".join(parts)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered = [[str(cell).replace("|", "\\|") for cell in row] for row in rows]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *("| " + " | ".join(row) + " |" for row in rendered),
        ]
    )


def make_all_windows(token_ids: list[int], block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor(token_ids, dtype=torch.long)
    if len(data) <= block_size:
        raise ValueError("debug corpus must contain more than block_size tokens")
    starts = range(len(data) - block_size)
    x = torch.stack([data[start : start + block_size] for start in starts])
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return x, y


@torch.no_grad()
def full_corpus_loss(
    model: MiniGPT,
    all_x: torch.Tensor,
    all_y: torch.Tensor,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    _logits, loss = model(all_x.to(device), all_y.to(device))
    if was_training:
        model.train()
    if loss is None:
        raise RuntimeError("expected a training loss")
    return float(loss.item())


def total_grad_norm(model: MiniGPT) -> float:
    squares = [parameter.grad.detach().float().pow(2).sum() for parameter in model.parameters() if parameter.grad is not None]
    return math.sqrt(sum(float(item.item()) for item in squares)) if squares else 0.0


def snapshot_parameters(model: MiniGPT) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().float().cpu().clone() for name, parameter in model.named_parameters()}


def snapshot_gradients(model: MiniGPT) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def summarize_optimizer_state(
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter, {})
        step = state.get("step", "-")
        if torch.is_tensor(step):
            step = int(step.item())
        first_moment = state.get("exp_avg", state.get("momentum_buffer"))
        second_moment = state.get("exp_avg_sq")
        rows.append(
            [
                name,
                step,
                "-" if first_moment is None else f"{first_moment.detach().float().norm().item():.6f}",
                "-" if second_moment is None else f"{second_moment.detach().float().norm().item():.6f}",
            ]
        )
    return rows


@torch.no_grad()
def greedy_generation_trace(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, torch.Tensor, list[dict[str, Any]]]:
    """Greedy decode while recording each context, top candidate, and choice."""

    token_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    generated = token_ids
    steps: list[dict[str, Any]] = []
    for step in range(max_new_tokens):
        context = generated[:, -model.config.block_size :]
        logits, _loss = model(context)
        last_logits = logits[:, -1]
        probabilities = F.softmax(last_logits, dim=-1)
        top_probabilities, top_ids = torch.topk(
            probabilities[0],
            k=min(3, tokenizer.vocab_size),
        )
        chosen = torch.argmax(last_logits, dim=-1, keepdim=True)
        steps.append(
            {
                "step": step,
                "context_ids": context[0].detach().cpu().tolist(),
                "context": tokenizer.decode(context[0].detach().cpu().tolist()),
                "top_ids": top_ids.detach().cpu().tolist(),
                "top_probabilities": top_probabilities.detach().cpu().tolist(),
                "chosen_id": int(chosen.item()),
                "chosen_token": tokenizer.itos[int(chosen.item())],
            }
        )
        generated = torch.cat((generated, chosen), dim=1)
    return tokenizer.decode(generated[0].detach().cpu().tolist()), generated, steps


def render_trace(
    trace: dict[str, Any],
    tokenizer: CharTokenizer,
    model: MiniGPT,
    *,
    include_pipeline_tensors: bool,
) -> str:
    input_ids = trace["input_ids"][0].tolist()
    target_ids = None if trace["targets"] is None else trace["targets"][0].tolist()
    token_labels = [display_token(tokenizer.itos[token_id]) for token_id in input_ids]
    parts: list[str] = []

    if include_pipeline_tensors:
        parts.extend(
            [
                tensor_section(
                    "token embedding",
                    trace["token_embedding"][0],
                    "根据 input id 从 embedding.weight 查出的内容向量；每行对应一个位置。",
                ),
                "",
            ]
        )
        if trace["position_embedding"] is None:
            parts.extend(
                [
                    "**position embedding** — 当前使用 RoPE，因此这里不向输入 hidden state 加绝对位置向量；位置稍后只旋转 Q/K。",
                    "",
                ]
            )
        else:
            parts.extend(
                [
                    tensor_section(
                        "learned position embedding",
                        trace["position_embedding"],
                        "按位置 0..T-1 查出的可训练绝对位置向量。",
                    ),
                    "",
                ]
            )
        parts.extend(
            [
                tensor_section(
                    "embedding sum / block input",
                    trace["embedding_sum"][0],
                    "token embedding 与可选 position embedding 相加后的 Transformer 输入。",
                ),
                "",
            ]
        )

    for block_trace in trace["blocks"]:
        layer = block_trace["layer_index"]
        parts.extend([f"#### Block {layer}", ""])
        if include_pipeline_tensors:
            parts.extend(
                [
                    tensor_section(
                        "LN1 output / QKV projection input",
                        block_trace["ln_1"][0],
                        "Pre-Norm 后的 X；Q=XW_Q^T+b_Q、K=XW_K^T+b_K、V=XW_V^T+b_V 都从它产生。",
                    ),
                    "",
                    tensor_section(
                        "Q/K/V concatenated for side-by-side display",
                        block_trace["qkv_concatenated"][0],
                        "这里只为报告并排拼接成 3C 维；默认教学模型实际由三个独立 Linear 产生 Q、K、V。",
                    ),
                    "",
                ]
            )

        for name, meaning in (
            ("q_flat", "每个位置的完整 query，尚未拆成多个 head。"),
            ("k_flat", "每个位置的完整 key，尚未拆成多个 head。"),
            ("v_flat", "每个位置的完整 value，尚未拆成多个 head。"),
        ):
            parts.extend([tensor_section(name, block_trace[name][0], meaning), ""])

        if not include_pipeline_tensors:
            parts.extend(
                [
                    head_tensor_sections(
                        "Q used by attention",
                        block_trace["q_heads"],
                        "逐 head 的最终 Q；RoPE 模式下已经旋转。",
                    ),
                    "",
                    head_tensor_sections(
                        "K used by attention",
                        block_trace["k_heads"],
                        "逐 head 的最终 K；RoPE 模式下已经旋转。",
                    ),
                    "",
                    head_tensor_sections(
                        "V per head",
                        block_trace["v_heads"],
                        "逐 head 的 V；RoPE 不旋转 V。",
                    ),
                    "",
                ]
            )

        if include_pipeline_tensors:
            parts.extend(
                [
                    head_tensor_sections(
                        "Q per head before position encoding",
                        block_trace["q_heads_before_position"],
                        "Q 从 [B,T,C] reshape/transpose 为 [B,H,T,D]。",
                    ),
                    "",
                    head_tensor_sections(
                        "K per head before position encoding",
                        block_trace["k_heads_before_position"],
                        "K 使用相同的拆头方式。",
                    ),
                    "",
                    head_tensor_sections(
                        "Q used by attention",
                        block_trace["q_heads"],
                        "RoPE 模式下这是旋转后的 Q；learned 模式下与旋转前相同。",
                    ),
                    "",
                    head_tensor_sections(
                        "K used by attention",
                        block_trace["k_heads"],
                        "RoPE 模式下这是旋转后的 K；V 永远不做 RoPE。",
                    ),
                    "",
                    head_tensor_sections(
                        "V per head",
                        block_trace["v_heads"],
                        "attention 权重最终读取的内容；shape=[B,H,T,D]。",
                    ),
                    "",
                    head_tensor_sections(
                        "scaled scores",
                        block_trace["scaled_scores"],
                        "每个 head 的 QK^T/sqrt(D)，尚未 causal mask。",
                    ),
                    "",
                    head_tensor_sections(
                        "masked scores",
                        block_trace["masked_scores"],
                        "未来位置被替换成 -inf，因此 softmax 后概率严格为 0。",
                    ),
                    "",
                    head_tensor_sections(
                        "attention weights",
                        block_trace["attention_weights"],
                        "对 masked scores 最后一维做 softmax；每行之和为 1。",
                    ),
                    "",
                    head_tensor_sections(
                        "context per head = weights @ V",
                        block_trace["context_per_head"],
                        "每个 query 位置从可见 value 中得到的加权和。",
                    ),
                    "",
                    tensor_section(
                        "concatenated heads",
                        block_trace["concatenated_heads"][0],
                        "把 H 个 head 的 D 维结果拼回 C=H*D 维。",
                    ),
                    "",
                    tensor_section(
                        "attention projection W_O",
                        block_trace["attention_projection"][0],
                        "拼头结果经过 c_proj/W_O 混合不同 head。",
                    ),
                    "",
                    tensor_section(
                        "first residual output",
                        block_trace["x_after_attention"][0],
                        "block 输入 + attention 分支输出。",
                    ),
                    "",
                    tensor_section(
                        "MLP expanded",
                        block_trace["mlp_expanded"][0],
                        "逐 token 从 C 扩到 4C。",
                    ),
                    "",
                    tensor_section(
                        "MLP activated",
                        block_trace["mlp_activated"][0],
                        "GELU 引入非线性。",
                    ),
                    "",
                    tensor_section(
                        "second residual / block output",
                        block_trace["x_after_mlp"][0],
                        "attention residual + MLP 分支，送入下一层或最终 LayerNorm。",
                    ),
                    "",
                ]
            )

    if include_pipeline_tensors:
        parts.extend(
            [
                tensor_section(
                    "final LayerNorm",
                    trace["final_norm"][0],
                    "所有 block 之后、lm_head 之前的 hidden state。",
                ),
                "",
                tensor_section(
                    "logits",
                    trace["logits"][0],
                    "每个位置对完整词表的未归一化分数，shape=[T,Vocab]。",
                ),
                "",
            ]
        )

    prediction_rows: list[list[Any]] = []
    probabilities = trace["probabilities"][0]
    for position, token_id in enumerate(input_ids):
        top_probabilities, top_ids = torch.topk(probabilities[position], k=min(3, tokenizer.vocab_size))
        candidates = ", ".join(
            f"{display_token(tokenizer.itos[int(candidate_id)])}:{probability:.3f}"
            for candidate_id, probability in zip(top_ids.tolist(), top_probabilities.tolist())
        )
        target = "-" if target_ids is None else display_token(tokenizer.itos[target_ids[position]])
        token_loss = "-" if trace["per_token_loss"] is None else f"{trace['per_token_loss'][0, position].item():.4f}"
        prediction_rows.append([position, token_labels[position], target, candidates, token_loss])
    parts.extend(
        [
            "**每个位置的 next-token 预测**",
            "",
            markdown_table(
                ["位置", "输入 token", "正确下一个 token", "概率最高的 3 项", "该位置 CE loss"],
                prediction_rows,
            ),
            "",
            f"平均 loss：`{trace['loss']:.6f}`" if trace["loss"] is not None else "未提供 targets，不计算 loss。",
        ]
    )
    return "\n".join(parts)


def build_tensor_dump(
    *,
    args: argparse.Namespace,
    corpus: str,
    tokenizer: CharTokenizer,
    token_ids: list[int],
    model: MiniGPT,
    trace_x: torch.Tensor,
    trace_y: torch.Tensor,
    before_trace: dict[str, Any],
    after_one_trace: dict[str, Any],
    final_trace: dict[str, Any],
    parameter_shapes: list[tuple[str, tuple[int, ...], int]],
    parameter_before: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor],
    qkv_activation_gradients: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    parameter_after_one: dict[str, torch.Tensor],
    optimizer_state_summary: list[list[Any]],
    raw_grad_norm: float,
    clipped_grad_norm: float,
    first_step_loss: float,
    loss_history: list[tuple[int, float]],
    generations: list[tuple[str, str]],
    generation_steps: list[dict[str, Any]],
    pipeline_checks: dict[str, bool],
) -> str:
    counts = Counter(corpus)
    vocab_rows = []
    for token_id, token in enumerate(tokenizer.itos):
        codepoint = "special" if token == tokenizer.unk_token else f"U+{ord(token):04X}"
        vocab_rows.append([token_id, display_token(token), codepoint, counts.get(token, 0)])

    x_ids = trace_x[0].tolist()
    y_ids = trace_y[0].tolist()
    shift_rows = [
        [
            position,
            x_ids[position],
            display_token(tokenizer.itos[x_ids[position]]),
            y_ids[position],
            display_token(tokenizer.itos[y_ids[position]]),
        ]
        for position in range(len(x_ids))
    ]

    parameter_rows = [
        [name, str(shape), count] for name, shape, count in parameter_shapes
    ]
    update_rows = []
    for name, before in parameter_before.items():
        after = parameter_after_one[name]
        grad = gradients.get(name)
        change = tensor_change(before, after)
        update_rows.append(
            [
                name,
                f"{before.norm().item():.6f}",
                "-" if grad is None else f"{grad.norm().item():.6f}",
                f"{change['l2']:.6f}",
                f"{change['max_abs']:.6f}",
            ]
        )

    qkv_changes = []
    for layer_index, (before_block, after_one_block, final_block) in enumerate(
        zip(before_trace["blocks"], after_one_trace["blocks"], final_trace["blocks"])
    ):
        for name in ("q_flat", "k_flat", "v_flat", "q_heads", "k_heads", "attention_weights"):
            one_change = tensor_change(before_block[name], after_one_block[name])
            final_change = tensor_change(before_block[name], final_block[name])
            qkv_changes.append(
                [
                    layer_index,
                    name,
                    f"{one_change['l2']:.6f}",
                    f"{one_change['max_abs']:.6f}",
                    f"{final_change['l2']:.6f}",
                    f"{final_change['max_abs']:.6f}",
                ]
            )

    config = model.config
    architecture = f"""input ids [B,T]
  -> CharTokenizer vocabulary (V={tokenizer.vocab_size})
  -> token embedding [B,T,C], C={config.n_embd}
  -> position: {config.position_encoding}
  -> 1 x pre-norm Transformer block
       LN -> {'fused QKV Linear' if config.fused_qkv else 'separate q_proj / k_proj / v_proj'} -> split {config.n_head} heads (D={config.n_embd // config.n_head})
       -> {'RoPE(Q,K) -> ' if config.position_encoding == 'rope' else ''}QK^T/sqrt(D) -> causal mask -> softmax -> weights@V
       -> concat heads -> W_O -> residual
       -> LN -> Linear(C,4C) -> GELU -> Linear(4C,C) -> residual
  -> final LayerNorm -> tied lm_head [B,T,V]
  -> cross entropy (training) / argmax (generation)"""

    qkv_parameter_parts: list[str] = []
    qkv_activation_gradient_parts: list[str] = []
    channels = config.n_embd
    if args.optimizer == "sgd":
        parameter_update_explanation = "SGD 直接使用裁剪后的梯度：$W_{new}=W_{old}-lr\\,dW$。"
        optimizer_explanation = (
            f"默认调试选择了 SGD，更新公式是 $\\theta_{{new}}=\\theta_{{old}}-\\eta\\nabla_\\theta L$，"
            f"这里 $\\eta={args.learning_rate}$。"
        )
    else:
        parameter_update_explanation = f"AdamW 使用一阶矩 m、二阶矩 v 和 bias correction 计算更新；这里 weight_decay={args.weight_decay}。"
        optimizer_explanation = (
            f"默认使用与 `train.py` 同类的 AdamW，`lr={args.learning_rate}`、`weight_decay={args.weight_decay}`。"
            "它先维护 $m_t=\\beta_1m_{t-1}+(1-\\beta_1)g_t$ 与 "
            "$v_t=\\beta_2v_{t-1}+(1-\\beta_2)g_t^2$，bias correction 后按 "
            "$\\theta_t=\\theta_{t-1}-\\eta\\hat m_t/(\\sqrt{\\hat v_t}+\\epsilon)$ 更新。"
        )

    for index, name in enumerate(("Q", "K", "V")):
        projection_name = name.lower()
        if config.fused_qkv:
            weight_parameter_name = "blocks.0.attn.c_attn.weight"
            bias_parameter_name = "blocks.0.attn.c_attn.bias"
            start, stop = index * channels, (index + 1) * channels
            before_weight = parameter_before[weight_parameter_name][start:stop]
            gradient_weight = gradients[weight_parameter_name][start:stop]
            after_weight = parameter_after_one[weight_parameter_name][start:stop]
            full_bias = parameter_before.get(bias_parameter_name)
            before_bias = None if full_bias is None else full_bias[start:stop]
            storage_explanation = f"融合 c_attn.weight 的第 {start}:{stop} 行"
        else:
            weight_parameter_name = f"blocks.0.attn.{projection_name}_proj.weight"
            bias_parameter_name = f"blocks.0.attn.{projection_name}_proj.bias"
            before_weight = parameter_before[weight_parameter_name]
            gradient_weight = gradients[weight_parameter_name]
            after_weight = parameter_after_one[weight_parameter_name]
            before_bias = parameter_before.get(bias_parameter_name)
            storage_explanation = f"独立模块 {projection_name}_proj.weight"
        qkv_parameter_parts.extend(
            [
                tensor_section(
                    f"W_{name} before first update",
                    before_weight,
                    f"{storage_explanation}；PyTorch 用 X @ W_{name}^T。",
                ),
                "",
                tensor_section(
                    f"gradient dW_{name}",
                    gradient_weight,
                    "loss.backward() 根据链式法则得到，随后被全局范数裁剪。",
                ),
                "",
                tensor_section(
                    f"W_{name} after first optimizer update",
                    after_weight,
                    parameter_update_explanation,
                ),
                "",
            ]
        )
        if before_bias is not None:
            qkv_parameter_parts.extend(
                [
                    tensor_section(
                        f"b_{name} before first update",
                        before_bias,
                        "对应 Q/K/V 线性投影的 bias。",
                    ),
                    "",
                ]
            )

    for name, gradient in zip(("Q", "K", "V"), qkv_activation_gradients):
        qkv_activation_gradient_parts.extend(
            [
                tensor_section(
                    f"activation gradient dL/d{name}",
                    gradient[0],
                    f"loss 对本次 forward 产生的 {name} 激活的梯度；它是传回 W_{name} 与上游 hidden state 的学习信号。",
                ),
                "",
            ]
        )

    check_rows = []
    for stage_name, trace in (
        ("训练前", before_trace),
        ("第 1 步后", after_one_trace),
        ("训练完成", final_trace),
    ):
        for check, passed in trace["checks"].items():
            check_rows.append([stage_name, check, "PASS" if passed else "FAIL"])
    for check, passed in pipeline_checks.items():
        check_rows.append(["保存/生成", check, "PASS" if passed else "FAIL"])

    generation_rows = [[repr(prompt), repr(output)] for prompt, output in generations]
    generation_step_rows: list[list[Any]] = []
    for record in generation_steps:
        candidates = ", ".join(
            f"{display_token(tokenizer.itos[token_id])}:{probability:.3f}"
            for token_id, probability in zip(record["top_ids"], record["top_probabilities"])
        )
        generation_step_rows.append(
            [
                repr(record["prompt"]),
                record["step"],
                repr(record["context"]),
                candidates,
                record["chosen_id"],
                display_token(record["chosen_token"]),
            ]
        )
    loss_rows = [[step, f"{loss:.6f}"] for step, loss in loss_history]

    return "\n\n".join(
        [
            "# MiniLLM 极小 Transformer 完整数值附录",
            f"> 生成命令：`python scripts/debug_tiny_transformer.py --device {args.device}`  \n> 随机种子：`{args.seed}`；报告中的值可在相同 PyTorch/设备上复现。",
            "## 先看结论",
            "这不是一个有通用知识的 LLM，而是一个约一千参数的 decoder-only Transformer 教具。它使用真实的 MiniLLM 模型、真实 autograd 和真实优化器，只把语料、维度和层数缩小到可以完整打印。真正的最小骨架只需 1 个 head；这里保留 2 个 head 是为了把 reshape、逐 head attention 和拼头过程也展示出来。默认 RoPE 用来衔接位置编码学习，也可用 `--position-encoding learned --n-head 1` 进一步简化。阅读顺序建议是：词表 → x/y → embedding → QKV → mask/softmax → V 加权 → residual/MLP → logits/loss → backward/update → generation。",
            "## 1. 极小语料与词表不是一回事",
            f"语料是训练文本，共 `{len(corpus)}` 个 Unicode 字符；词表是从语料中去重得到的 token 集合，共 `{tokenizer.vocab_size}` 项（含 `<unk>`）。当前 CharTokenizer 的一个中文字符、标点、空格或换行各算一个 token。",
            code_block(corpus.rstrip("\n")),
            "**完整词表**",
            markdown_table(["id", "token", "Unicode", "在语料中出现次数"], vocab_rows),
            f"完整语料编码后共有 `{len(token_ids)}` 个 token id。token id 只是查表编号，本身没有大小语义。",
            "## 2. Next-token 训练样本如何构造",
            "固定追踪样本采用右移一位的 x/y。位置 t 输入 x[t]，正确答案是原语料的下一个 token y[t]：",
            markdown_table(["位置", "x id", "x token", "y id", "应预测 token"], shift_rows),
            f"`x shape={tuple(trace_x.shape)}`，`y shape={tuple(trace_y.shape)}`。训练其它 step 使用同一语料中随机截取的 `{args.batch_size}` 个长度 `{args.block_size}` 窗口。",
            "## 3. 最小模型结构与 shape",
            code_block(architecture),
            markdown_table(["参数名", "shape", "元素数"], parameter_rows),
            f"去重后的可训练参数总数：`{model.parameter_count():,}`。`token_embedding.weight` 与 `lm_head.weight` 权重共享，所以不会重复计数。",
            f"shape 记号：B=batch，T=序列长度，V=词表大小，C=隐藏维度，H=head 数，D=C/H。这里追踪时 B=1、T={trace_x.size(1)}、C={config.n_embd}、H={config.n_head}、D={config.n_embd // config.n_head}。",
            "## 4. Q、K、V 究竟怎样生成",
            ("教学配置实际使用三个独立模块：`q_proj: Linear(C,C)`、`k_proj: Linear(C,C)`、`v_proj: Linear(C,C)`。"
             if not config.fused_qkv else
             "当前显式启用了生产式 `c_attn: Linear(C,3C)`，其输出再切成 Q/K/V。")
            + "\n\n"
            + r"$$Q=XW_Q^{\top}+b_Q,\qquad K=XW_K^{\top}+b_K,\qquad V=XW_V^{\top}+b_V.$$"
            + "\n\n"
            + "融合与不融合数学完全相同；融合只是少读两次 X、减少 kernel launch。第一次学习建议使用默认独立模式。之后每份 `[B,T,C]` reshape 成 `[B,H,T,D]`。",
            *qkv_parameter_parts,
            "## 5. 训练前完整 forward trace",
            render_trace(before_trace, tokenizer, model, include_pipeline_tensors=True),
            "## 6. Backward 与第一个 optimizer step",
            f"固定追踪样本训练前的 loss 是 `{first_step_loss:.6f}`。`loss.backward()` 先得到原始全局梯度范数 `{raw_grad_norm:.6f}`；`clip_grad_norm_(..., 1.0)` 后范数为 `{clipped_grad_norm:.6f}`。{optimizer_explanation} Q/K/V 本身是本次 forward 的临时激活，不由 optimizer 直接保存或更新；反向传播先得到 $dL/dQ,dL/dK,dL/dV$，再据此计算参数梯度并更新 $W_Q,W_K,W_V$。",
            *qkv_activation_gradient_parts,
            "**第一个 optimizer step 后的状态**",
            markdown_table(["参数", "state step", "一阶矩/动量 L2", "二阶矩 L2"], optimizer_state_summary),
            markdown_table(
                ["参数", "更新前参数 L2", "裁剪后梯度 L2", "参数更新 L2", "最大绝对更新"],
                update_rows,
            ),
            "**同一个固定样本在第一个 update 后，内部张量怎样变化**",
            markdown_table(
                ["block", "张量", "第 1 步变化 L2", "第 1 步最大变化", "最终变化 L2", "最终最大变化"],
                qkv_changes,
            ),
            "### 同一固定样本在第 1 步后的 Q/K/V",
            render_trace(after_one_trace, tokenizer, model, include_pipeline_tensors=False),
            "## 7. 训练完成后的 Q/K/V 与预测",
            render_trace(final_trace, tokenizer, model, include_pipeline_tensors=False),
            "上面再次完整打印了最终 Q/K/V。它们变化有两个来源：生成 QKV 的 $W_Q/W_K/W_V$ 已更新；它们的输入 embedding、LayerNorm、前级参数也已更新。RoPE 本身没有训练参数，但输入 Q/K 变化后，旋转结果也随之变化。",
            "## 8. Loss 曲线与生成",
            markdown_table(["优化 step", "全语料滑窗平均 loss"], loss_rows),
            markdown_table(["prompt", "greedy 生成结果（含 prompt）"], generation_rows),
            "**逐 token 生成决策**",
            markdown_table(["prompt", "step", "本步 context", "top-3", "选中 id", "选中 token"], generation_step_rows),
            "生成循环每次只取最后位置的 logits，argmax 得到一个新 token id，拼回上下文后再次 forward。脚本还用 KV cache 路径重跑并验证 token 序列完全一致。真实应用常用 temperature/top-k 采样；这里用 greedy 是为了结果可复现。",
            "## 9. 自动正确性检查",
            markdown_table(["阶段", "检查", "结果"], check_rows),
            "这些检查验证：教学 trace 重算的 logits/loss 与 MiniGPT.forward 一致；Q/K/V 输出与各自投影参数公式一致；causal mask 后未来权重为 0；softmax 每行和为 1。任何一项 FAIL 都意味着不要相信报告。",
            "## 10. 后续添加功能时从哪里观察",
            markdown_table(
                ["准备添加的功能", "先观察/修改的位置", "它会改变什么"],
                [
                    ["更换 BPE/SentencePiece tokenizer", "tokenizer、vocab.json、embedding/lm_head shape", "token 粒度、V 和序列长度"],
                    ["记忆/RAG", "送入 tokenizer 前的 prompt 或额外 memory tokens", "输入序列与 attention 可见信息"],
                    ["KV cache", "每层旋转后的 K 与 V", "避免生成时重复计算历史 token"],
                    ["GQA/MQA", "Q/K/V 的 head 数与 reshape", "多个 Q head 共享较少 K/V head"],
                    ["LoRA", "W_Q/W_K/W_V/W_O 或 MLP Linear", "冻结原权重，仅学习低秩增量"],
                    ["RMSNorm/SwiGLU", "LN1/LN2 与 MLP trace", "归一化方式与非线性分支"],
                    ["更长上下文", "position encoding、causal mask、T×T scores", "位置外推与 attention 成本"],
                ],
            ),
            "最重要的调试习惯：每加一个功能，先固定 seed 和同一条 x/y，比较该功能前后的 shape、数值、loss、梯度和生成；一次只改变一个变量。",
        ]
    ) + "\n"


def build_guide_report(
    *,
    args: argparse.Namespace,
    corpus: str,
    tokenizer: CharTokenizer,
    token_ids: list[int],
    model: MiniGPT,
    trace_x: torch.Tensor,
    trace_y: torch.Tensor,
    before_trace: dict[str, Any],
    after_one_trace: dict[str, Any],
    final_trace: dict[str, Any],
    loss_history: list[tuple[int, float]],
    generations: list[tuple[str, str]],
    generation_steps: list[dict[str, Any]],
    pipeline_checks: dict[str, bool],
) -> str:
    """Render the relationship-first tutorial; exact numbers live in tensor_dump.md."""

    config = model.config
    channels = config.n_embd
    heads = config.n_head
    head_dim = channels // heads
    train_batch = args.batch_size
    train_length = args.block_size
    trace_length = trace_x.size(1)
    counts = Counter(corpus)

    vocab_rows = []
    for token_id, token in enumerate(tokenizer.itos):
        codepoint = "special" if token == tokenizer.unk_token else f"U+{ord(token):04X}"
        vocab_rows.append([token_id, display_token(token), codepoint, counts.get(token, 0)])

    trace_rows = []
    for position, (input_id, target_id) in enumerate(zip(trace_x[0].tolist(), trace_y[0].tolist())):
        trace_rows.append(
            [
                position,
                display_token(tokenizer.itos[input_id]),
                input_id,
                display_token(tokenizer.itos[target_id]),
                target_id,
            ]
        )

    window_x = token_ids[:train_length]
    window_y = token_ids[1 : train_length + 1]
    window_rows = [
        [
            position,
            display_token(tokenizer.itos[input_id]),
            display_token(tokenizer.itos[target_id]),
        ]
        for position, (input_id, target_id) in enumerate(zip(window_x, window_y))
    ]

    qkv_module_names = (
        "blocks.0.attn.c_attn（融合，随后 split）"
        if config.fused_qkv
        else "blocks.0.attn.q_proj / k_proj / v_proj（三个独立 Linear）"
    )
    component_rows = [
        ["blocks.0", "第 0 个（也是唯一一个）Transformer block；`.0` 是 Python 从 0 开始的索引。"],
        [qkv_module_names, "把 LN1 输出分别变成 Q、K、V。教学默认不融合。"],
        ["blocks.0.attn.c_proj", "attention 的输出投影 $W_O$；作用于 `weights @ V` 拼头后的结果，不生成 Q/K/V。"],
        ["blocks.0.mlp.net.0", f"MLP 的第一个 Linear：$C={channels} \\rightarrow 4C={4 * channels}$，扩大逐 token 特征空间。"],
        ["blocks.0.mlp.net.1", "GELU 激活；没有参数，所以参数表里看不到 weight。"],
        ["blocks.0.mlp.net.2", f"MLP 的第二个 Linear：$4C={4 * channels} \\rightarrow C={channels}$，恢复残差主干宽度。"],
        ["blocks.0.mlp.net.3", "Dropout；本调试配置 dropout=0，因此数值不变。"],
    ]

    qkv_layout_formula = (
        r"$[Q\;K\;V]=X_1W_{QKV}^{\top}+b_{QKV}$，再沿最后一维切三份"
        if config.fused_qkv
        else r"$Q=X_1W_Q^{\top}+b_Q$；$K=X_1W_K^{\top}+b_K$；$V=X_1W_V^{\top}+b_V$"
    )
    flow_rows = [
        ["0", "原始文本 → token id", r"$i_t=\operatorname{tokenizer}(text_t)$", f"文本 → `[B,T]`；训练时 B={train_batch}, T={train_length}", "离散文本先变成可查表的整数。"],
        ["1", "token id → embedding", r"$H_0=E[i]$", f"`[B,T] → [B,T,C]`，C={channels}", "每个 token id 查出一个 C 维向量。"],
        ["2", "Attention 前归一化", r"$X_1=\operatorname{LN}_1(H_0)$", "`[B,T,C] → [B,T,C]`", "只调整每个 token 向量的尺度，shape 不变。"],
        ["3", "生成 Q/K/V", qkv_layout_formula, "三份 `[B,T,C]`", "三者读取同一个 $X_1$，但使用不同参数，因此含义不同。"],
        ["4", "拆成多个 head", r"$C=H\times D$", f"每份 `[B,T,{channels}] → [B,{heads},T,{head_dim}]`", "这里只 reshape/transpose，不学习、不改变元素值。"],
        ["5", "加入 RoPE", r"$Q_m'=R_mQ_m,\ K_n'=R_nK_n$", "Q/K shape 不变；V 不变", "位置只改变 Q/K 的方向，使内积含相对距离。"],
        ["6", "Q 和 K 匹配", r"$S=Q'K'^{\top}/\sqrt D$", f"`[B,H,T,D] × [B,H,D,T] → [B,H,T,T]`", "每个 query 位置得到对每个 key 位置的分数。"],
        ["7", "causal mask", r"$S_{m,n}=-\infty\ (n>m)$", "`[B,H,T,T]`", "未来位置变成 -∞，所以 softmax 后权重为 0。"],
        ["8", "分数变权重", r"$A=\operatorname{softmax}(S)$", "`[B,H,T,T]`", "每一行和为 1，表示当前位置怎样分配读取比例。"],
        ["9", "读取 V", r"$Z=AV$", f"`[B,H,T,T] × [B,H,T,{head_dim}] → [B,H,T,{head_dim}]`", "A 决定读多少，V 提供真正被读取的内容。"],
        ["10", "拼头并通过 $W_O$", r"$O=\operatorname{Concat}(Z_1,\ldots,Z_H)W_O^{\top}+b_O$", f"`[B,H,T,D] → [B,T,C]`", "`c_proj` 混合各 head，并恢复残差所需的 C 维。"],
        ["11", "第一个残差", r"$H_1=H_0+O$", "`[B,T,C]`", "原信息与 attention 读取的信息相加。"],
        ["12", "MLP 前归一化", r"$X_2=\operatorname{LN}_2(H_1)$", "`[B,T,C]`", "为逐 token MLP 稳定尺度。"],
        ["13", "MLP 扩维", r"$U=X_2W_{up}^{\top}+b_{up}$", f"`[B,T,{channels}] → [B,T,{4 * channels}]`", "Linear(C,4C) 给每个 token 更多中间特征。"],
        ["14", "MLP 非线性", r"$G=\operatorname{GELU}(U)$", f"`[B,T,{4 * channels}]`", "若没有 GELU，两个 Linear 合起来仍只是一个 Linear。"],
        ["15", "MLP 降维", r"$M=GW_{down}^{\top}+b_{down}$", f"`[B,T,{4 * channels}] → [B,T,{channels}]`", "恢复 C 维，才能与残差主干相加。"],
        ["16", "第二个残差", r"$H_2=H_1+M$", "`[B,T,C]`", "这就是一个完整 Transformer block 的输出。"],
        ["17", "最终归一化与词表投影", r"$L=\operatorname{LN}_f(H_2)E^{\top}$", f"`[B,T,C] → [B,T,Vocab={tokenizer.vocab_size}]`", "每个位置得到词表中每个 token 的 logit。"],
        ["18", "Next-token loss", r"$\mathcal L=-\frac1{BT}\sum_{b,t}\log p(y_{b,t}\mid x_{b,\le t})$", "一个标量", "一次 batch 同时产生 B×T 个预测训练信号。"],
        ["19", "反向传播与更新", r"$\nabla_\theta\mathcal L\rightarrow\operatorname{AdamW.step}()$", "参数 shape 不变、数值改变", "更新 W/embedding/norm；下一次 forward 才产生变化后的 Q/K/V。"],
    ]

    change_rows = []
    for name in ("q_flat", "k_flat", "v_flat", "q_heads", "k_heads", "attention_weights"):
        first_change = tensor_change(before_trace["blocks"][0][name], after_one_trace["blocks"][0][name])
        final_change = tensor_change(before_trace["blocks"][0][name], final_trace["blocks"][0][name])
        change_rows.append(
            [
                name,
                f"{first_change['l2']:.6f}",
                f"{final_change['l2']:.6f}",
                {
                    "q_flat": "$X_1,W_Q,b_Q$ 共同决定",
                    "k_flat": "$X_1,W_K,b_K$ 共同决定",
                    "v_flat": "$X_1,W_V,b_V$ 共同决定",
                    "q_heads": "Q reshape 后再经 RoPE",
                    "k_heads": "K reshape 后再经 RoPE",
                    "attention_weights": "由变化后的 Q/K 经 score、mask、softmax 得到",
                }[name],
            ]
        )

    final_prediction_rows = []
    probabilities = final_trace["probabilities"][0]
    for position, (input_id, target_id) in enumerate(zip(trace_x[0].tolist(), trace_y[0].tolist())):
        top_probability, top_id = torch.max(probabilities[position], dim=-1)
        final_prediction_rows.append(
            [
                position,
                display_token(tokenizer.itos[input_id]),
                display_token(tokenizer.itos[target_id]),
                display_token(tokenizer.itos[int(top_id.item())]),
                f"{top_probability.item():.3f}",
            ]
        )

    loss_rows = [[step, f"{loss:.6f}"] for step, loss in loss_history]
    generation_rows = [[repr(prompt), repr(output)] for prompt, output in generations]
    generation_step_rows = []
    for record in generation_steps:
        candidates = ", ".join(
            f"{display_token(tokenizer.itos[token_id])}:{probability:.3f}"
            for token_id, probability in zip(record["top_ids"], record["top_probabilities"])
        )
        generation_step_rows.append(
            [
                repr(record["prompt"]),
                record["step"],
                repr(record["context"]),
                candidates,
                display_token(record["chosen_token"]),
            ]
        )

    check_rows = []
    for stage, trace in (("训练前", before_trace), ("第 1 步后", after_one_trace), ("训练完成", final_trace)):
        for check, passed in trace["checks"].items():
            check_rows.append([stage, check, "PASS" if passed else "FAIL"])
    for check, passed in pipeline_checks.items():
        check_rows.append(["保存/生成", check, "PASS" if passed else "FAIL"])

    return "\n\n".join(
        [
            "# MiniLLM 极小 Transformer：从语料到生成的完整流转",
            "> [!info] 怎么阅读\n> 这份主报告只解释“为什么从 A 变成 B”。完整浮点 Tensor 已移到 [tensor_dump.md](tensor_dump.md)，需要核对某个具体数值时再打开。",
            f"> [!question] 本轮问题（已回答）\n> - [x] 随机截取 {train_batch} 个长度 {train_length} 的窗口、training step、next-token 样本分别是什么？\n> - [x] `[B,T,C]` 中的 C、QKV Linear、`Linear(C,4C)` 是什么？\n> - [x] `c_proj`、`mlp.net.0`、`mlp.net.2` 分别做什么？\n> - [x] Q/K/V 能否不融合，以及每一步对应哪条公式？\n> - [x] 为什么逐 token 生成默认只有 {args.max_new_tokens} 个 step？",
            "**先直接回答最容易混淆的两个 step：**",
            markdown_table(
                ["名称", "一次 step 做什么", "本次默认值"],
                [
                    ["optimization/training step", "随机取一个 batch → forward → loss → backward → AdamW 更新一次参数", f"总共 {args.train_steps} 步"],
                    ["generation token step", "用当前上下文 forward → 只取最后位置 logits → 选 1 个 token → 拼回上下文", f"`max_new_tokens={args.max_new_tokens}`，所以每个 prompt 正好 5 步"],
                ],
            ),
            f"这次运行里 `step 0` 只是训练前评估，不更新参数；`step 1` 特意使用固定的 `B=1,T={trace_length}` 样本，方便精确比较第一次更新前后；`step 2～{args.train_steps}` 才使用随机的 `B={train_batch},T={train_length}` batch。{train_batch} 个窗口可以重叠，甚至可以重复，它们不是把语料平均切成 {train_batch} 份。",
            f"“随机截取 `{train_batch}` 个长度 `{train_length}` 的窗口”是指：每个 optimization step 从 96-token 语料中随机选择 {train_batch} 个起点；从每个起点连续取 {train_length} 个 token 组成 x，再右移一位组成 y。因此 x/y shape 都是 `[{train_batch},{train_length}]`，一个 step 同时监督 `{train_batch}×{train_length}={train_batch * train_length}` 个 next-token 预测。窗口不是把一个词切成 8 份，而是连续 8 个字符 token。",
            "一个长度 8 窗口的实际 x/y 对齐如下：",
            markdown_table(["窗口内位置 t", "输入 x[t]", "目标 y[t]=下一个 token"], window_rows),
            "必须构造 next-token 样本，是因为 decoder-only LLM 把整段文本概率分解为：\n\n"
            + r"$$p(z_0,\ldots,z_{N-1})=\prod_{t=0}^{N-2}p_\theta(z_{t+1}\mid z_0,\ldots,z_t).$$"
            + "\n\n文本右移一位就自动提供了标签，不需要人工标注。同一个窗口的 8 个位置可以并行训练；causal mask 保证第 t 个位置看不到未来答案。若目标仍是输入本身，模型只需复制当前 token，学不到续写。生成时则把 next-token 规则逐 token 使用。",
            "## 1. 语料、词表、训练样本",
            f"语料共有 `{len(corpus)}` 个字符；CharTokenizer 去重后得到 `{tokenizer.vocab_size}` 个 token（含 `<unk>`）。**语料是训练内容，词表只是 token 与整数 id 的双向映射。**",
            markdown_table(["id", "token", "Unicode", "出现次数"], vocab_rows),
            "固定追踪样本（为了训练前后始终比较同一输入）是：",
            markdown_table(["位置", "输入 token", "输入 id", "目标 token", "目标 id"], trace_rows),
            "## 2. B、T、C、H、D 到底是什么",
            markdown_table(
                ["符号", "含义", "训练时", "固定 trace 时"],
                [
                    ["B", "batch size，一次并行多少条窗口", train_batch, 1],
                    ["T", "每条窗口有多少个 token 位置", train_length, trace_length],
                    ["C", "channel/hidden size；**每个 token 用多少个连续特征数表示**", channels, channels],
                    ["H", "attention head 数", heads, heads],
                    ["D", "每个 head 的特征维度，D=C/H", head_dim, head_dim],
                    ["Vocab", "模型可以预测多少种 token", tokenizer.vocab_size, tokenizer.vocab_size],
                ],
            ),
            f"所以 `[B,T,C]` 不是三种数据：它表示一个三维张量。训练时 `[16,8,{channels}]` 中共有 16 条窗口，每条 8 个 token，每个 token 已从整数 id 变成 {channels} 维向量。C=8 只是为了教学可观察；真实模型常是几千维。",
            "## 3. 模块名称先对上含义",
            markdown_table(["代码名", "真实作用"], component_rows),
            "`mlp.net` 是 `nn.Sequential`，数字只是执行顺序下标：0→1→2→3。只有 0 和 2 是 Linear，所以参数表只出现 `.0.weight` 与 `.2.weight`。`Linear(C,4C)` 的目的不是跨 token 交流；它对每个 token 单独扩展特征，再用 GELU 产生非线性组合，最后降回 C。跨 token 交流发生在 attention。",
            "## 4. 一次 forward 的逐步流转与公式",
            "下面每一行的输出，就是下一行的输入；这才是整个 Transformer 主线：",
            code_block(
                "LN1(hidden)\n"
                " ├─ q_proj → Q → 分头 → RoPE ─┐\n"
                " ├─ k_proj → K → 分头 → RoPE ─┼→ QKᵀ → mask → softmax = A\n"
                " └─ v_proj → V → 分头 ─────────┘                       │\n"
                "                                                       A @ V\n"
                "                                                         ↓\n"
                "                                                 拼头 → W_O → 残差"
            ),
            markdown_table(["步骤", "计算", "公式", "shape 变化", "与下一步的关系"], flow_rows),
            "## 5. 为什么默认不用 fused QKV",
            ("当前报告对应的真实模型参数就是三个独立模块：`q_proj.weight`、`k_proj.weight`、`v_proj.weight`。这与论文公式一一对应，便于学习。"
             if not config.fused_qkv else
             "当前命令显式传入了 `--fused-qkv`，所以真实模型使用 `c_attn`，但报告仍按等价的三条公式解释。"),
            "不融合完全可以：分别调用三个 `Linear(C,C)` 即可。生产模型常融合成一个 `Linear(C,3C)`，是为了让 GPU 少读取两次相同的 X，并减少 kernel launch；它不是 Transformer 理论要求，也不改变 Q/K/V 的值。想对比时运行：`python scripts/debug_tiny_transformer.py --fused-qkv`。",
            "## 6. Q/K/V 在训练中究竟怎样变化",
            code_block(
                "持久参数 θ(step s)\n"
                "    ↓ forward\n"
                "临时激活 Q/K/V → attention → logits → loss\n"
                "    ↓ backward\n"
                "参数梯度 ∂loss/∂θ\n"
                "    ↓ optimizer.step()\n"
                "持久参数 θ(step s+1)\n"
                "    ↓ 下一次 forward\n"
                "重新计算出新的 Q/K/V"
            ),
            "Q/K/V 是 **activation（一次 forward 的中间结果）**，不是 optimizer 直接更新的长期参数。对第 s 个 optimization step：\n\n"
            + r"$$Q^{(s)}=X_1^{(s)}(W_Q^{(s)})^\top+b_Q^{(s)}.$$"
            + "\n\nAdamW 更新的是 embedding、$W_Q/W_K/W_V$、$W_O$、MLP、Norm 等参数；下一次 forward 中，输入 $X_1$ 和投影参数都变了，于是重新算出的 Q/K/V 才发生变化。",
            markdown_table(["中间量", "第 1 次更新后的 L2 变化", "训练完成后的 L2 变化", "变化链条"], change_rows),
            "关系链应这样读：`参数/上游 hidden 改变 → Q/K/V 改变 → scores 改变 → attention weights 改变 → 读取的 V 改变 → block 输出改变 → logits/loss 改变`。完整 before/after 数字见 [tensor_dump.md](tensor_dump.md)。",
            "## 7. Loss 是否真的让预测变好",
            markdown_table(["optimization step", "全语料平均 loss"], loss_rows),
            markdown_table(["位置", "输入", "正确下一个 token", "最终 top-1", "概率"], final_prediction_rows),
            "## 8. 为什么生成是 5 个 step",
            f"脚本默认 `--max-new-tokens {args.max_new_tokens}`，所以每个 prompt 只追加 {args.max_new_tokens} 个 token。**生成 step 数等于要求新增的 token 数，与训练的 {args.train_steps} 个 optimization step 完全不是一回事。**例如 `小猫吃` 后依次生成 `鱼 → 。 → \\n → 小 → 猫`，所以正好 {args.max_new_tokens} 步。prompt 有 3 个 token，`3+{args.max_new_tokens}=block_size={config.block_size}`，也恰好满足教学 KV-cache 的总长度限制。当前极小词表没有 `<eos>`，模型不会自行停止，{args.max_new_tokens} 是人为停止条件。可改成 `--max-new-tokens 20`；普通 generate 会使用最近 block_size 个 token 的滑动窗口，而教学 KV-cache 路径需要更大的 block_size 或滑动 cache 才能继续。",
            markdown_table(["prompt", "完整 greedy 输出"], generation_rows),
            markdown_table(["prompt", "token step", "本步输入 context", "top-3", "本步选中"], generation_step_rows),
            "## 9. 自动校验",
            markdown_table(["阶段", "检查", "结果"], check_rows),
            "完整浮点矩阵、逐 head Q/K/V、mask、softmax、梯度和参数更新值都保存在 [tensor_dump.md](tensor_dump.md)。主报告刻意不再用这些数字打断流程理解。",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    if args.train_steps < 1:
        raise ValueError("train_steps must be at least 1")
    if args.trace_length < 1 or args.trace_length > args.block_size:
        raise ValueError("trace_length must be in [1, block_size]")

    torch.manual_seed(args.seed)
    torch.set_printoptions(precision=5, sci_mode=False, linewidth=180, threshold=100_000)
    device = torch.device(args.device)
    corpus = read_text(args.data)
    tokenizer = CharTokenizer.from_text(corpus)
    token_ids = tokenizer.encode(corpus)
    if len(token_ids) <= args.trace_length:
        raise ValueError("corpus is too short for trace_length + target")

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=1,
        n_head=args.n_head,
        n_embd=args.n_embd,
        fused_qkv=args.fused_qkv,
        dropout=0.0,
        position_encoding=args.position_encoding,
        rope_theta=args.rope_theta,
    )
    model = MiniGPT(config).to(device)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    trace_x = torch.tensor([token_ids[: args.trace_length]], dtype=torch.long, device=device)
    trace_y = torch.tensor([token_ids[1 : args.trace_length + 1]], dtype=torch.long, device=device)
    training_data = torch.tensor(token_ids, dtype=torch.long)
    all_x, all_y = make_all_windows(token_ids, args.block_size)

    parameter_shapes = [
        (name, tuple(parameter.shape), parameter.numel())
        for name, parameter in model.named_parameters()
    ]
    parameter_before = snapshot_parameters(model)
    model.eval()
    before_trace = trace_forward(model, trace_x, trace_y)
    loss_history: list[tuple[int, float]] = [(0, full_corpus_loss(model, all_x, all_y, device))]

    model.train()
    optimizer.zero_grad(set_to_none=True)
    retained_qkv: dict[str, torch.Tensor] = {}
    qkv_hooks: list[torch.utils.hooks.RemovableHandle] = []

    def make_retain_qkv_gradient(name: str):
        def retain_qkv_gradient(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            output.retain_grad()
            retained_qkv[name] = output

        return retain_qkv_gradient

    attention = model.blocks[0].attn
    if attention.c_attn is not None:
        qkv_hooks.append(
            attention.c_attn.register_forward_hook(make_retain_qkv_gradient("fused"))
        )
    else:
        for name, projection in (
            ("q", attention.q_proj),
            ("k", attention.k_proj),
            ("v", attention.v_proj),
        ):
            if projection is None:
                raise RuntimeError(f"missing {name}_proj in separate QKV mode")
            qkv_hooks.append(projection.register_forward_hook(make_retain_qkv_gradient(name)))

    _logits, first_loss_tensor = model(trace_x, trace_y)
    if first_loss_tensor is None:
        raise RuntimeError("expected first-step loss")
    first_step_loss = float(first_loss_tensor.item())
    first_loss_tensor.backward()
    for hook in qkv_hooks:
        hook.remove()
    if config.fused_qkv:
        fused_qkv_activation = retained_qkv.get("fused")
        if fused_qkv_activation is None or fused_qkv_activation.grad is None:
            raise RuntimeError("failed to retain fused QKV activation gradient")
        qkv_activation_gradients = tuple(
            part.detach().float().cpu().clone()
            for part in fused_qkv_activation.grad.split(config.n_embd, dim=-1)
        )
    else:
        gradient_parts = []
        for name in ("q", "k", "v"):
            activation = retained_qkv.get(name)
            if activation is None or activation.grad is None:
                raise RuntimeError(f"failed to retain {name.upper()} activation gradient")
            gradient_parts.append(activation.grad.detach().float().cpu().clone())
        qkv_activation_gradients = tuple(gradient_parts)
    raw_grad_norm = total_grad_norm(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    clipped_grad_norm = total_grad_norm(model)
    gradients = snapshot_gradients(model)
    optimizer.step()
    optimizer_state_summary = summarize_optimizer_state(model, optimizer)
    parameter_after_one = snapshot_parameters(model)

    model.eval()
    after_one_trace = trace_forward(model, trace_x, trace_y)
    loss_history.append((1, full_corpus_loss(model, all_x, all_y, device)))

    for step in range(2, args.train_steps + 1):
        model.train()
        x, y = get_batch(training_data, args.block_size, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        _logits, loss = model(x, y)
        if loss is None:
            raise RuntimeError("expected training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.report_interval == 0 or step == args.train_steps:
            loss_history.append((step, full_corpus_loss(model, all_x, all_y, device)))
            print(f"step={step:04d} full_corpus_loss={loss_history[-1][1]:.6f}")

    model.eval()
    final_trace = trace_forward(model, trace_x, trace_y)
    generations: list[tuple[str, str]] = []
    generation_steps: list[dict[str, Any]] = []
    pipeline_checks = {
        "manual greedy trace matches MiniGPT.generate": True,
        "ordinary greedy matches KV-cache greedy": True,
    }
    for prompt in ("小猫吃", "小狗吃", "小猫喝", "小狗喝"):
        prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        output, traced_ids, step_records = greedy_generation_trace(
            model,
            tokenizer,
            prompt,
            args.max_new_tokens,
        )
        ordinary_ids = model.generate(
            prompt_ids.clone(),
            max_new_tokens=args.max_new_tokens,
            greedy=True,
        )
        cached_ids = model.generate_with_kv_cache(
            prompt_ids.clone(),
            max_new_tokens=args.max_new_tokens,
            greedy=True,
        )
        pipeline_checks["manual greedy trace matches MiniGPT.generate"] &= bool(
            torch.equal(traced_ids, ordinary_ids)
        )
        pipeline_checks["ordinary greedy matches KV-cache greedy"] &= bool(
            torch.equal(ordinary_ids, cached_ids)
        )
        for record in step_records:
            record["prompt"] = prompt
            generation_steps.append(record)
        generations.append((prompt, output))

    if Path(args.data).resolve() == (ROOT / "data" / "debug_corpus.txt").resolve():
        expected_next = {"小猫吃": "鱼", "小狗吃": "肉", "小猫喝": "水", "小狗喝": "水"}
        pipeline_checks["default corpus patterns are learned"] = all(
            len(output) > len(prompt) and output[len(prompt)] == expected_next[prompt]
            for prompt, output in generations
        )

    if not all(
        bool(value)
        for trace in (before_trace, after_one_trace, final_trace)
        for value in trace["checks"].values()
    ):
        raise AssertionError("one or more trace correctness checks failed")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_payload = {
        "tokenizer_type": "char",
        "vocab_size": tokenizer.vocab_size,
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
        "note": "Each ordinary entry is one Unicode character; <unk> is id 0.",
    }
    (out_dir / "vocab.json").write_text(
        json.dumps(vocab_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_buffer = StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["step", "full_corpus_loss"])
    writer.writerows(loss_history)
    (out_dir / "loss.csv").write_text(csv_buffer.getvalue(), encoding="utf-8")

    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "tokenizer_type": "char",
            "tokenizer": tokenizer.to_dict(),
            "args": vars(args),
        },
        checkpoint_path,
    )
    restored_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    restored_model = MiniGPT(GPTConfig(**restored_checkpoint["config"])).to(device).eval()
    restored_model.load_state_dict(restored_checkpoint["model"])
    restored_tokenizer = CharTokenizer.from_dict(restored_checkpoint["tokenizer"])
    with torch.no_grad():
        restored_logits, _restored_loss = restored_model(trace_x, trace_y)
    pipeline_checks["checkpoint reload reproduces final logits"] = bool(
        torch.allclose(restored_logits.detach().cpu(), final_trace["logits"], rtol=1e-6, atol=1e-7)
    )
    pipeline_checks["checkpoint tokenizer roundtrip is identical"] = bool(
        restored_tokenizer.to_dict() == tokenizer.to_dict()
    )

    if not all(pipeline_checks.values()):
        raise AssertionError("one or more save/generation pipeline checks failed")

    tensor_dump = build_tensor_dump(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        token_ids=token_ids,
        model=model,
        trace_x=trace_x.detach().cpu(),
        trace_y=trace_y.detach().cpu(),
        before_trace=before_trace,
        after_one_trace=after_one_trace,
        final_trace=final_trace,
        parameter_shapes=parameter_shapes,
        parameter_before=parameter_before,
        gradients=gradients,
        qkv_activation_gradients=qkv_activation_gradients,
        parameter_after_one=parameter_after_one,
        optimizer_state_summary=optimizer_state_summary,
        raw_grad_norm=raw_grad_norm,
        clipped_grad_norm=clipped_grad_norm,
        first_step_loss=first_step_loss,
        loss_history=loss_history,
        generations=generations,
        generation_steps=generation_steps,
        pipeline_checks=pipeline_checks,
    )
    guide_report = build_guide_report(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        token_ids=token_ids,
        model=model,
        trace_x=trace_x.detach().cpu(),
        trace_y=trace_y.detach().cpu(),
        before_trace=before_trace,
        after_one_trace=after_one_trace,
        final_trace=final_trace,
        loss_history=loss_history,
        generations=generations,
        generation_steps=generation_steps,
        pipeline_checks=pipeline_checks,
    )
    (out_dir / "report.md").write_text(guide_report, encoding="utf-8")
    (out_dir / "tensor_dump.md").write_text(tensor_dump, encoding="utf-8")

    print(f"vocab_size={tokenizer.vocab_size}, parameters={model.parameter_count():,}")
    print(f"before_loss={before_trace['loss']:.6f}, final_loss={final_trace['loss']:.6f}")
    for prompt, output in generations:
        print(f"generate {prompt!r} -> {output!r}")
    print(f"wrote {out_dir / 'report.md'}")
    print(f"wrote {out_dir / 'tensor_dump.md'}")
    print(f"wrote {out_dir / 'vocab.json'}")
    print(f"wrote {out_dir / 'loss.csv'}")
    print(f"wrote {out_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
