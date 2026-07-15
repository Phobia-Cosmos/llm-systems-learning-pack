"""Chinese evidence report generation for MiniAgent benchmark summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    number = _number(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}{suffix}"


def _engine_label(engine: Any) -> str:
    value = str(engine)
    return {"vllm": "vLLM", "sglang": "SGLang"}.get(value.lower(), value)


def _pct_change(previous: float | None, current: float | None) -> float | None:
    if previous is None or current is None or previous == 0:
        return None
    return (current / previous - 1.0) * 100.0


def _stat(group: Mapping[str, Any], metric: str, percentile: str) -> float | None:
    stats = group.get("stats", {})
    if not isinstance(stats, Mapping):
        return None
    metric_stats = stats.get(metric, {})
    if not isinstance(metric_stats, Mapping):
        return None
    return _number(metric_stats.get(percentile))


def _metrics(group: Mapping[str, Any]) -> Mapping[str, Any]:
    value = group.get("metrics", {})
    return value if isinstance(value, Mapping) else {}


def _engine_groups(summary: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    groups = summary.get("groups", [])
    if not isinstance(groups, Sequence):
        return result
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        engine = str(group.get("engine", "unknown"))
        result.setdefault(engine, []).append(group)
    for engine_groups in result.values():
        engine_groups.sort(key=lambda value: int(value.get("concurrency", 0)))
    return result


def _repeat_range(group: Mapping[str, Any]) -> str:
    value = group.get("repeat_p95_ttft_ms", {})
    if not isinstance(value, Mapping):
        return "N/A"
    return (
        f"{_fmt(value.get('median'))} "
        f"[{_fmt(value.get('min'))}, {_fmt(value.get('max'))}]"
    )


QUEUE_MATERIALITY_SECONDS = 0.001


def _queue_evidence(
    group: Mapping[str, Any],
) -> tuple[bool, bool, bool, str]:
    """Return (available, present, material, concise description)."""

    metrics = _metrics(group)
    runs_with_metrics = _number(metrics.get("runs_with_metrics")) or 0
    waiting = _number(metrics.get("max_waiting"))
    queue_mean_s = _number(metrics.get("queue_time_mean_seconds"))
    available = runs_with_metrics > 0 and (waiting is not None or queue_mean_s is not None)
    # Queue histograms include unavoidable microsecond-scale admission
    # bookkeeping even at c1. Treating any nonzero float as the explanation
    # for a tens-of-milliseconds TTFT change is numerically true but causally
    # misleading. A sampled waiting request or >=1 ms mean is material here.
    present = (waiting is not None and waiting > 0) or (
        queue_mean_s is not None and queue_mean_s > 0
    )
    # If a mean exists, use its magnitude to judge contribution; a transient
    # nonzero waiting gauge alone does not explain a large cell-wide p95 jump.
    material = (
        queue_mean_s >= QUEUE_MATERIALITY_SECONDS
        if queue_mean_s is not None
        else waiting is not None and waiting > 0
    )
    details: list[str] = []
    if waiting is not None:
        details.append(f"max_waiting={waiting:.0f}")
    if queue_mean_s is not None:
        details.append(f"queue mean={queue_mean_s * 1000.0:.2f} ms")
        if queue_mean_s < QUEUE_MATERIALITY_SECONDS:
            details.append("低于 1 ms 诊断阈值")
    return (
        available,
        present,
        material,
        "，".join(details) if details else "无可用排队指标",
    )


def _diagnose_step(
    engine: str, previous: Mapping[str, Any], current: Mapping[str, Any]
) -> list[str]:
    previous_c = int(previous.get("concurrency", 0))
    current_c = int(current.get("concurrency", 0))
    previous_ttft = _stat(previous, "ttft_ms", "p95")
    current_ttft = _stat(current, "ttft_ms", "p95")
    change = _pct_change(previous_ttft, current_ttft)
    prefix = f"- **{_engine_label(engine)}，并发 {previous_c} → {current_c}**："
    if change is None:
        return [prefix + "TTFT 样本不足，不能判断退化。"]
    direction = "上升" if change > 0 else "下降"
    measured = (
        f"测量事实：pooled p95 TTFT 从 {_fmt(previous_ttft)} ms "
        f"{direction}到 {_fmt(current_ttft)} ms（{change:+.1f}%）。"
    )
    lines = [prefix + measured]

    available, queue_present, queue_material, queue_description = _queue_evidence(
        current
    )
    metrics = _metrics(current)
    running = _number(metrics.get("max_running"))
    preemptions = _number(metrics.get("preemptions_delta"))
    retractions = _number(metrics.get("retractions_delta"))
    max_retracted = _number(metrics.get("max_retracted"))
    kv_usage = _number(metrics.get("kv_usage_max"))
    gpu_max = _number(metrics.get("gpu_utilization_percent_max"))
    prefill_mean = _number(metrics.get("prefill_time_mean_seconds"))
    previous_prefill_mean = _number(
        _metrics(previous).get("prefill_time_mean_seconds")
    )
    current_tpot = _stat(current, "tpot_ms", "p95")
    previous_tpot = _stat(previous, "tpot_ms", "p95")
    tpot_change = _pct_change(previous_tpot, current_tpot)

    causal_evidence: list[str] = []
    if queue_material:
        causal_evidence.append(
            f"服务端直接观测到排队且达到实质幅度（{queue_description}）"
        )
    elif queue_present:
        causal_evidence.append(f"服务端仅观测到轻微/短暂排队（{queue_description}）")
    elif available:
        causal_evidence.append(f"本次采样未观测到排队（{queue_description}）")
    else:
        causal_evidence.append("服务端排队指标缺失，不能直接验证排队")
    if running is not None:
        causal_evidence.append(f"max_running={running:.0f}")
    if preemptions is not None and preemptions > 0:
        causal_evidence.append(f"preemption 增量={preemptions:.0f}")
    if retractions is not None and retractions > 0:
        causal_evidence.append(f"retraction 增量={retractions:.0f}")
    if max_retracted is not None and max_retracted > 0:
        causal_evidence.append(f"max_retracted={max_retracted:.0f}")
    if kv_usage is not None:
        causal_evidence.append(f"KV/token usage max={kv_usage * 100.0:.1f}%")
    if gpu_max is not None:
        causal_evidence.append(f"GPU utilization max={gpu_max:.0f}%")
    if prefill_mean is not None:
        causal_evidence.append(f"prefill mean={prefill_mean * 1000.0:.2f} ms")
    lines.append("  - 因果证据：" + "；".join(causal_evidence) + "。")

    if change <= 0:
        lines.append("  - 解释边界：这一阶没有 p95 TTFT 退化，无需为其构造退化原因。")
    elif queue_material:
        lines.append(
            "  - 证据支持的解释：请求在开始 prefill 前已经等待，排队至少是 TTFT "
            "退化的一个贡献因素；仅凭这些指标不能断言它是唯一原因。"
        )
    elif (
        prefill_mean is not None
        and previous_prefill_mean is not None
        and prefill_mean > previous_prefill_mean * 1.2
    ):
        lines.append(
            "  - 证据支持的解释：排队时间不足 1 ms，但服务端 prefill mean 从 "
            f"{previous_prefill_mean * 1000.0:.2f} ms 增至 "
            f"{prefill_mean * 1000.0:.2f} ms；主要退化与更大的并发 prefill batch / "
            "prefill-decode 计算竞争一致。它仍不是排除其他因素后的唯一因果证明。"
        )
    elif available and tpot_change is not None and tpot_change > 0:
        lines.append(
            f"  - 架构推断：没有实质排队，但 p95 TPOT 同时变化 {tpot_change:+.1f}%；"
            "更像批处理后的 GPU 计算竞争或调度开销。该句是推断，不是排队指标直接证明。"
        )
    elif not available:
        lines.append(
            "  - 解释边界：缺少排队指标，客户端 TTFT 只能证明退化存在，不能把原因归结为排队；"
            "批处理、prefill 争用与前端开销都仍是候选。"
        )
    else:
        lines.append(
            "  - 解释边界：已有指标没有给出排队证据，也没有足够的同步 TPOT 变化；"
            "当前数据不足以定位原因。"
        )
    return lines


def _flatten_context(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_flatten_context(value[key], child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        scalar = all(not isinstance(item, (Mapping, Sequence)) or isinstance(item, str) for item in value)
        if scalar:
            result.append((prefix, ", ".join(str(item) for item in value)))
    elif value is not None:
        result.append((prefix, str(value)))
    return result


def render_report(summary: Mapping[str, Any], title: str | None = None) -> str:
    """Render a deterministic Chinese Markdown evidence report."""

    context = summary.get("context", {})
    if not isinstance(context, Mapping):
        context = {}
    run_id = context.get("run_id", context.get("experiment_id"))
    report_title = title or "Qwen 在 vLLM / SGLang 上的并发性能与 TTFT 退化报告"
    lines = [f"# {report_title}", ""]
    if run_id:
        lines.extend([f"运行 ID：`{run_id}`", ""])

    valid = bool(summary.get("validation", {}).get("valid", False))
    lines.extend(
        [
            "## 结论摘要",
            "",
            (
                "本报告的数据校验已通过。"
                if valid
                else "本报告存在数据校验错误；错误修复前，数字只能用于排障，不能作为最终结论。"
            ),
            "",
        ]
    )
    engine_groups = _engine_groups(summary)
    for engine, groups in engine_groups.items():
        if not groups:
            continue
        pieces = [
            f"c{int(group.get('concurrency', 0))}: {_fmt(_stat(group, 'ttft_ms', 'p95'))} ms"
            for group in groups
        ]
        lines.append(
            f"- {_engine_label(engine)} pooled p95 TTFT："
            + "；".join(pieces)
            + "。"
        )
    if len(engine_groups) >= 2:
        highest_concurrency = max(
            int(group.get("concurrency", 0))
            for groups in engine_groups.values()
            for group in groups
        )
        high_groups = [
            group
            for groups in engine_groups.values()
            for group in groups
            if int(group.get("concurrency", 0)) == highest_concurrency
        ]
        if len(high_groups) >= 2:
            high_groups.sort(key=lambda group: str(group.get("engine")))
            descriptions = []
            for group in high_groups:
                throughput = group.get("throughput", {})
                output_rate = (
                    throughput.get("output_tokens_per_second")
                    if isinstance(throughput, Mapping)
                    else None
                )
                descriptions.append(
                    f"{_engine_label(group.get('engine'))} {_fmt(output_rate)} token/s、"
                    f"E2E p95 {_fmt(_stat(group, 'e2e_ms', 'p95'))} ms"
                )
            lines.append(
                f"- c{highest_concurrency} 吞吐/E2E 权衡："
                + "；".join(descriptions)
                + "。"
            )
    if engine_groups:
        lines.append("")

    lines.extend(
        [
            "TTFT 是客户端从发出请求到收到第一个流式 token 的时间，包含同机 HTTP/前端处理、"
            "服务端排队、tokenization/prefill 以及首 token decode；它不是纯 GPU prefill 时间。",
            "",
            "## 测量结果",
            "",
            "下表的延迟百分位由所有 repetition 的逐请求原始样本合并后重新计算。"
            "“重复 p95”给出各次运行 p95 的中位数 `[最小值, 最大值]`，不会用运行级 p95 的平均值代替 pooled p95。",
            "吞吐按所有 repetition 的总 token 数除以总持续时间计算，因此是 duration-weighted 聚合。",
            "",
            "| 引擎 | 并发 | 请求样本 | TTFT p50 / p95 / p99 (ms) | 重复 p95 中位数 [min, max] (ms) | "
            "TPOT p95 (ms/token) | E2E p95 (ms) | 输出吞吐 (token/s) | max waiting / running | queue mean (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for engine, groups in engine_groups.items():
        for group in groups:
            metrics = _metrics(group)
            queue_mean = _number(metrics.get("queue_time_mean_seconds"))
            waiting = _fmt(metrics.get("max_waiting"), 0)
            running = _fmt(metrics.get("max_running"), 0)
            throughput = group.get("throughput", {})
            output_rate = (
                throughput.get("output_tokens_per_second")
                if isinstance(throughput, Mapping)
                else None
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        _engine_label(engine),
                        str(group.get("concurrency", "N/A")),
                        str(group.get("request_samples", "N/A")),
                        " / ".join(
                            _fmt(_stat(group, "ttft_ms", percentile))
                            for percentile in ("p50", "p95", "p99")
                        ),
                        _repeat_range(group),
                        _fmt(_stat(group, "tpot_ms", "p95")),
                        _fmt(_stat(group, "e2e_ms", "p95")),
                        _fmt(output_rate),
                        f"{waiting} / {running}",
                        _fmt(queue_mean * 1000.0 if queue_mean is not None else None),
                    )
                )
                + " |"
            )
    lines.append("")

    lines.extend(["## p95 TTFT 为什么退化", ""])
    lines.append(
        "下面每一阶先列测量事实，再列因果证据，最后才给解释。`max_waiting` 用于确认队列是否出现；"
        "只有 mean queue time ≥ 1 ms（或 mean 缺失而 waiting 非零）时，才把排队称为退化的实质贡献。"
    )
    lines.append("")
    any_steps = False
    for engine, groups in engine_groups.items():
        for previous, current in zip(groups, groups[1:]):
            any_steps = True
            lines.extend(_diagnose_step(engine, previous, current))
    if not any_steps:
        lines.append("没有足够的并发阶梯可做退化分析。")
    lines.append("")

    lines.extend(["## 引擎间比较", ""])
    by_concurrency: dict[int, list[Mapping[str, Any]]] = {}
    for groups in engine_groups.values():
        for group in groups:
            by_concurrency.setdefault(int(group.get("concurrency", 0)), []).append(group)
    comparisons = 0
    for concurrency in sorted(by_concurrency):
        candidates = [
            (group, _stat(group, "ttft_ms", "p95"))
            for group in by_concurrency[concurrency]
        ]
        candidates = [(group, value) for group, value in candidates if value is not None]
        if len(candidates) < 2:
            continue
        comparisons += 1
        candidates.sort(key=lambda item: item[1])
        winner, winner_value = candidates[0]
        runner_up, runner_value = candidates[1]
        gap = _pct_change(winner_value, runner_value)
        first, second = candidates[0][0], candidates[1][0]
        first_rate = _number(
            first.get("throughput", {}).get("output_tokens_per_second")
            if isinstance(first.get("throughput"), Mapping)
            else None
        )
        second_rate = _number(
            second.get("throughput", {}).get("output_tokens_per_second")
            if isinstance(second.get("throughput"), Mapping)
            else None
        )
        throughput_text = ""
        if first_rate is not None and second_rate is not None:
            throughput_text = (
                f"；输出吞吐分别为 {_fmt(first_rate)} / {_fmt(second_rate)} token/s"
            )
        e2e_candidates = [
            (group, _stat(group, "e2e_ms", "p95"))
            for group in by_concurrency[concurrency]
        ]
        e2e_candidates = [
            (group, value) for group, value in e2e_candidates if value is not None
        ]
        e2e_text = ""
        if len(e2e_candidates) >= 2:
            e2e_candidates.sort(key=lambda item: item[1])
            e2e_winner, e2e_value = e2e_candidates[0]
            e2e_runner, e2e_runner_value = e2e_candidates[1]
            e2e_text = (
                f"；E2E p95 则是 {_engine_label(e2e_winner.get('engine'))} {_fmt(e2e_value)} ms、"
                f"{_engine_label(e2e_runner.get('engine'))} {_fmt(e2e_runner_value)} ms"
            )
        first_queue = _number(_metrics(first).get("queue_time_mean_seconds"))
        second_queue = _number(_metrics(second).get("queue_time_mean_seconds"))
        queue_text = ""
        if first_queue is not None and second_queue is not None:
            queue_text = (
                f"；mean queue time 为 {_fmt(first_queue * 1000.0)} / "
                f"{_fmt(second_queue * 1000.0)} ms"
            )
        lines.append(
            f"- 并发 {concurrency}：TTFT 尾延迟由 {_engine_label(winner.get('engine'))} 领先，pooled p95="
            f"{_fmt(winner_value)} ms，对比 {_engine_label(runner_up.get('engine'))} 的 {_fmt(runner_value)} ms"
            f"（后者相对高 {_fmt(gap, 1, '%')}）{throughput_text}{e2e_text}{queue_text}。"
        )
    if comparisons == 0:
        lines.append("缺少同并发、跨引擎的完整数据，暂不排名。")
    lines.append("")

    lines.extend(["## 综合根因判断与公平性检查", ""])
    config = context.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    input_tokens = _number(config.get("input_tokens"))
    prefill_budget = _number(config.get("max_batched_tokens"))
    if input_tokens and prefill_budget:
        concurrency_values = sorted(
            {
                int(group.get("concurrency", 0))
                for groups in engine_groups.values()
                for group in groups
            }
        )
        wave_descriptions = []
        for concurrency in concurrency_values:
            prompt_demand = concurrency * input_tokens
            waves = math.ceil(prompt_demand / prefill_budget)
            wave_descriptions.append(
                f"c{concurrency}: {prompt_demand:.0f}/{prefill_budget:.0f}，至少 {waves} 个 token-budget waves"
            )
        lines.append(
            "- 配置机制：每个 prompt 固定 "
            f"{input_tokens:.0f} tokens，而单轮 prefill/batched budget 是 {prefill_budget:.0f}。"
            + "；".join(wave_descriptions)
            + "。因此并发越高，尾部请求经历更多 prefill admission/scheduling waves；"
            "这是配置与 waiting/queue 指标共同支持的机制，不只是对客户端曲线的猜测。"
        )
    for engine, groups in engine_groups.items():
        if len(groups) < 2:
            continue
        low, high = groups[0], groups[-1]
        low_tpot = _stat(low, "tpot_ms", "p95")
        high_tpot = _stat(high, "tpot_ms", "p95")
        gpu_max = _number(_metrics(high).get("gpu_utilization_percent_max"))
        lines.append(
            f"- {_engine_label(engine)} 的 p95 TPOT 从 {_fmt(low_tpot)} 增至 {_fmt(high_tpot)} ms/token"
            + (
                f"，高并发 GPU utilization max={gpu_max:.0f}%"
                if gpu_max is not None
                else ""
            )
            + "；这表明首 token 之外的 decode 也受到 batch 扩大和 GPU 计算竞争影响。"
        )
    highest_groups = [groups[-1] for groups in engine_groups.values() if groups]
    if highest_groups:
        kv_parts = []
        pressure_events = []
        for group in highest_groups:
            engine = _engine_label(group.get("engine"))
            metrics = _metrics(group)
            kv = _number(metrics.get("kv_usage_max"))
            if kv is not None:
                kv_parts.append(f"{engine}={kv * 100.0:.1f}%")
            preemptions = _number(metrics.get("preemptions_delta"))
            retractions = _number(metrics.get("retractions_delta"))
            max_retracted = _number(metrics.get("max_retracted"))
            if preemptions is not None:
                pressure_events.append(f"{engine} preemption={preemptions:.0f}")
            if retractions is not None:
                pressure_events.append(f"{engine} retraction={retractions:.0f}")
            elif max_retracted is not None:
                pressure_events.append(f"{engine} max_retracted={max_retracted:.0f}")
        if kv_parts:
            lines.append(
                "- KV 公平性：最高并发的 KV/token usage max 为 "
                + "、".join(kv_parts)
                + ("；" + "、".join(pressure_events) if pressure_events else "")
                + "。本次没有接近 100% 的 KV 使用或换出事件，因此不能把主要退化归因于 KV cache 耗尽。"
            )
    lines.append("")

    lines.extend(["## 实验上下文", ""])
    flattened = _flatten_context(context)
    if flattened:
        for key, value in flattened:
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("manifest 未提供实验上下文。")
    lines.append("")

    validation = summary.get("validation", {})
    if isinstance(validation, Mapping):
        errors = validation.get("errors", [])
        warnings = validation.get("warnings", [])
    else:
        errors, warnings = [], []
    lines.extend(["## 数据校验", ""])
    lines.append(f"- 状态：{'通过' if valid else '失败'}")
    if errors:
        for message in errors:
            lines.append(f"- 错误：{message}")
    if warnings:
        for message in warnings:
            lines.append(f"- 警告：{message}")
    if not errors and not warnings:
        lines.append("- 没有错误或警告。")
    lines.append("")

    lines.extend(
        [
            "## 证据索引",
            "",
            "每个文件的 SHA-256 固定了本报告所引用的字节内容。客户端 raw JSON 是延迟结论的主证据；"
            "metrics JSON 是排队、运行请求数和 preemption/retraction 的辅助因果证据；"
            "server log 固定了实际生效配置、KV 容量与 CUDA Graph 记录。",
            "",
            "| Evidence ID | 类型 | 引擎 / 并发 / 重复 | 路径 | SHA-256 |",
            "|---|---|---|---|---|",
        ]
    )
    evidence = summary.get("evidence", [])
    if isinstance(evidence, Sequence):
        for item in evidence:
            if not isinstance(item, Mapping):
                continue
            cell = " / ".join(
                str(item.get(key, "-")) for key in ("engine", "concurrency", "repeat")
            )
            lines.append(
                f"| {item.get('id', '-')} | {item.get('kind', '-')} | {cell} | "
                f"`{item.get('path', '-')}` | `{item.get('sha256', '-')}` |"
            )
    lines.append("")

    lines.extend(
        [
            "## 局限性",
            "",
            "- 这是固定输入/输出长度、无限 request-rate 的饱和压测；它回答容量边界问题，不代表生产到达过程。",
            "- 同机客户端减少网络噪声，但没有覆盖真实跨机网络和网关开销。",
            "- 单 GPU、单模型和当前软件版本的结论不能直接外推到其他模型、量化方式或硬件。",
            "- Prometheus 轮询是离散采样；短于采样间隔的瞬时队列可能未被 gauge 最大值捕获。",
            "- 观察相关性不能单独证明唯一因果；报告已把服务端直接指标与架构推断分开。",
            "",
        ]
    )
    return "\n".join(lines)


def generate_report(
    summary: Mapping[str, Any] | str | Path,
    output_path: str | Path | None = None,
    *,
    title: str | None = None,
) -> str:
    """Load (if needed), render, optionally write, and return Markdown."""

    if isinstance(summary, Mapping):
        data = summary
    else:
        with Path(summary).open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, Mapping):
            raise ValueError("summary JSON root must be an object")
        data = loaded
    markdown = render_report(data, title=title)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return markdown
