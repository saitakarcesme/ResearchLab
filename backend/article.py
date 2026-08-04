from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

_SETTING_PATTERNS: tuple[tuple[str, str], ...] = (
    ("window_pattern", r"\bwindow_pattern\b|attention window pattern"),
    (
        "short_attention_window",
        r"short[-_ ]windows?|short[-_ ]attention(?:[-_ ](?:windows?|layers?))?",
    ),
    ("warmdown", r"\bwarmdown(?:_ratio)?\b"),
    ("warmup", r"\bwarmup(?:_ratio)?\b"),
    ("matrix_lr", r"\bmatrix_lr\b|matrix learning rate"),
    ("weight_decay", r"\bweight_decay\b|weight decay"),
    ("adam_betas", r"\badam_betas\b|adam betas?|\bbeta[12]\b"),
    ("momentum_ramp", r"momentum ramp"),
    ("bos_targets", r"\bbos\b.*(?:target|token)|(?:target|token).*\bbos\b"),
    ("batch_size", r"\btotal_batch_size\b|total batch size|batch[- ]size"),
)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _format_metric(value: Any) -> str:
    number = _number(value)
    return "not recorded" if number is None else f"{number:.6f}"


def _one_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _prediction_loss_language(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"`?val_bpb`?", "prediction loss", text, flags=re.IGNORECASE)
    return re.sub(
        r"validation bits per byte", "prediction loss", text, flags=re.IGNORECASE
    )


def _experiment_number(item: Mapping[str, Any]) -> int:
    try:
        return int(item.get("experiment_number") or 0)
    except (TypeError, ValueError):
        return 0


def _is_lower_better(research: Mapping[str, Any]) -> bool:
    return str(research.get("metric_direction", "lower_is_better")) != (
        "higher_is_better"
    )


def _is_better(candidate: float, reference: float, *, lower_is_better: bool) -> bool:
    return candidate < reference if lower_is_better else candidate > reference


def _baseline_value(
    research: Mapping[str, Any], experiments: Sequence[Mapping[str, Any]]
) -> float | None:
    persisted = _number(research.get("baseline_value"))
    if persisted is not None:
        return persisted
    for item in sorted(experiments, key=_experiment_number):
        if item.get("accepted") is True and item.get("previous_best") is None:
            measured = _number(item.get("metric_value"))
            if measured is not None:
                return measured
    return None


def _best_value(
    research: Mapping[str, Any],
    experiments: Sequence[Mapping[str, Any]],
    *,
    lower_is_better: bool,
) -> float | None:
    persisted = _number(research.get("best_value"))
    if persisted is not None:
        return persisted
    measured = [
        value
        for item in experiments
        if item.get("accepted") is True
        for value in [_number(item.get("metric_value"))]
        if value is not None
    ]
    if not measured:
        return None
    return min(measured) if lower_is_better else max(measured)


def _is_baseline_experiment(item: Mapping[str, Any], baseline: float | None) -> bool:
    if item.get("accepted") is not True or item.get("previous_best") is not None:
        return False
    measured = _number(item.get("metric_value"))
    text = f"{item.get('hypothesis', '')} {item.get('change_summary', '')}".lower()
    return (
        "baseline" in text
        or baseline is None
        or (
            measured is not None
            and math.isclose(measured, baseline, rel_tol=0, abs_tol=1e-12)
        )
    )


def _best_experiment(
    research: Mapping[str, Any],
    experiments: Sequence[Mapping[str, Any]],
    best: float | None,
) -> Mapping[str, Any] | None:
    accepted = [
        item
        for item in experiments
        if item.get("accepted") is True and not item.get("error")
    ]
    requested_commit = _one_line(research.get("best_git_commit"))
    if requested_commit:
        matching_commit = [
            item
            for item in accepted
            if _one_line(item.get("git_commit")) == requested_commit
        ]
        if matching_commit:
            return max(matching_commit, key=_experiment_number)
    if best is not None:
        matching_score = [
            item
            for item in accepted
            if (value := _number(item.get("metric_value"))) is not None
            and math.isclose(value, best, rel_tol=0, abs_tol=1e-12)
        ]
        if matching_score:
            return max(matching_score, key=_experiment_number)
    return max(accepted, key=_experiment_number) if accepted else None


def _setting_key(item: Mapping[str, Any]) -> str:
    summary = _one_line(item.get("change_summary"))
    lowered = summary.lower()
    for key, pattern in _SETTING_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            return key
    constant = re.search(r"\b[A-Z][A-Z0-9_]{2,}\b", summary)
    if constant and constant.group(0) not in {"GPT", "CUDA"}:
        return f"constant:{constant.group(0)}"
    stem = re.sub(r"\d+(?:\.\d+)?", "#", lowered)
    stem = re.sub(r"[^a-z_#]+", " ", stem).strip()
    return f"change:{stem[:80]}"


def _last_match(pattern: str, text: str) -> str | None:
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = next((part for part in reversed(value) if part), "")
    return str(value).strip().strip('.,;"') or None


def _clean_change_summary(value: Any) -> str:
    text = _one_line(value).replace(" | ", ". ")
    if not text:
        return "No code-change summary was recorded"
    text = re.sub(
        r"\s*;\s*(?:the\s+)?(?:model\s+)?(?:forward|evaluation|GPT\.forward).*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s+and (?:the\s+)?(?:model\s+)?forward/evaluation contracts?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*\.\s*(?:The\s+)?(?:model\s+)?(?:forward|evaluation|GPT\.forward).*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.strip().rstrip(".;")
    if len(text) > 260:
        text = text[:257].rstrip() + "…"
    return text


def _setting_description(item: Mapping[str, Any]) -> str:
    summary = _one_line(item.get("change_summary"))
    key = _setting_key(item)

    if key == "batch_size":
        value = _last_match(
            r"TOTAL_BATCH_SIZE\s+(?:from\s+[^\s;,.]+\s+to|to)\s+([0-9][0-9*,_]*)",
            summary,
        )
        if value:
            return f"Total batch size: `{value}`"
    elif key == "warmdown":
        value = _last_match(
            r"WARMDOWN(?:_RATIO)?\s+from\s+[0-9.]+\s+to\s+([0-9.]+)", summary
        )
        if value:
            return f"Warmdown ratio: `{value}`"
    elif key == "warmup":
        value = _last_match(
            r"WARMUP(?:_RATIO)?\s+from\s+[0-9.]+\s+to\s+([0-9.]+)", summary
        )
        if value:
            return f"Warmup ratio: `{value}`"
    elif key == "matrix_lr":
        value = _last_match(r"MATRIX_LR\s+from\s+[0-9.]+\s+to\s+([0-9.]+)", summary)
        if value:
            return f"Matrix learning rate: `{value}`"
    elif key == "weight_decay":
        value = _last_match(r"WEIGHT_DECAY\s+from\s+[0-9.]+\s+to\s+([0-9.]+)", summary)
        if value:
            return f"Weight decay: `{value}`"
    elif key == "adam_betas":
        value = _last_match(
            r"ADAM_BETAS\s+from\s+\([^)]*\)\s+to\s+(\([^)]*\))", summary
        )
        if value:
            return f"Adam betas: `{value}`"
    elif key == "momentum_ramp":
        value = _last_match(
            r"momentum ramp denominator\s+from\s+\d+\s+to\s+(\d+)", summary
        )
        if value:
            return f"Muon momentum-ramp denominator: `{value}` steps"
    elif key == "bos_targets" and re.search(
        r"mask(?:ed|ing)?.*BOS|BOS.*mask", summary, re.IGNORECASE
    ):
        return "BOS training targets are excluded from the loss"
    elif key == "window_pattern":
        value = _last_match(
            r"WINDOW_PATTERN\s+from\s+[\"'][^\"']+[\"']\s+to\s+[\"']([^\"']+)[\"']",
            summary,
        )
        if value:
            return f"Attention-window pattern: `{value}`"
    elif key == "short_attention_window":
        value = _last_match(
            r"(?:to|producing|yielding(?:\s+a)?)\s+([0-9][0-9,]*)-token\s+(?:short(?:[- ]attention)?\s+)?windows?",
            summary,
        )
        if value:
            final_context = _last_match(
                r"final\s+([0-9][0-9,]*)-token\s+full-context\s+layer", summary
            )
            seven_layers = bool(
                re.search(
                    r"seven short-attention layers|layers\s+0\s*[–-]\s*6",
                    summary,
                    re.IGNORECASE,
                )
            )
            description = f"Short-attention window: `{value}` tokens"
            if seven_layers:
                description += " for the first seven layers"
            if final_context:
                description += (
                    f"; the final layer keeps `{final_context}`-token full context"
                )
            return description

    return _clean_change_summary(summary)


def _candidate_description(item: Mapping[str, Any]) -> str:
    description = _setting_description(item).rstrip(".")
    if ": " in description:
        label, value = description.split(": ", 1)
        return f"{label.lower()} {value}"
    return description[0].lower() + description[1:] if description else "the candidate"


def _accepted_gain(item: Mapping[str, Any], *, lower_is_better: bool) -> float:
    measured = _number(item.get("metric_value"))
    previous = _number(item.get("previous_best"))
    if measured is None or previous is None:
        return 0.0
    return max(previous - measured if lower_is_better else measured - previous, 0.0)


def _final_settings(
    experiments: Sequence[Mapping[str, Any]],
    baseline: float | None,
    best_item: Mapping[str, Any] | None,
    *,
    lower_is_better: bool,
    limit: int = 8,
) -> list[Mapping[str, Any]]:
    best_number = _experiment_number(best_item) if best_item is not None else math.inf
    final_by_area: dict[str, Mapping[str, Any]] = {}
    for item in sorted(experiments, key=_experiment_number):
        if (
            item.get("accepted") is not True
            or item.get("error")
            or _experiment_number(item) > best_number
            or _is_baseline_experiment(item, baseline)
        ):
            continue
        final_by_area[_setting_key(item)] = item
    settings = list(final_by_area.values())
    if len(settings) > limit:
        settings = sorted(
            settings,
            key=lambda item: (
                _accepted_gain(item, lower_is_better=lower_is_better),
                _experiment_number(item),
            ),
            reverse=True,
        )[:limit]
    return sorted(settings, key=_experiment_number)


def _rejection_cost(item: Mapping[str, Any], *, lower_is_better: bool) -> float | None:
    measured = _number(item.get("metric_value"))
    previous = _number(item.get("previous_best"))
    if measured is None or previous is None:
        return None
    return measured - previous if lower_is_better else previous - measured


def _useful_rejections(
    experiments: Sequence[Mapping[str, Any]], *, lower_is_better: bool, limit: int = 3
) -> list[Mapping[str, Any]]:
    latest_by_area: dict[str, Mapping[str, Any]] = {}
    for item in sorted(experiments, key=_experiment_number):
        if (
            item.get("accepted") is False
            and not item.get("error")
            and _number(item.get("metric_value")) is not None
            and _number(item.get("previous_best")) is not None
        ):
            latest_by_area[_setting_key(item)] = item
    ranked = sorted(
        latest_by_area.values(),
        key=lambda item: (
            _rejection_cost(item, lower_is_better=lower_is_better) or 0.0,
            _experiment_number(item),
        ),
        reverse=True,
    )
    worse = [
        item
        for item in ranked
        if (_rejection_cost(item, lower_is_better=lower_is_better) or 0.0) > 0
    ]
    return (worse or ranked)[:limit]


def _is_candidate_timeout(message: Any) -> bool:
    lowered = _one_line(message).lower()
    return "candidate generation" in lowered and (
        "time limit" in lowered or "timed out" in lowered or "timeout" in lowered
    )


def _error_key(message: Any) -> tuple[str, str]:
    text = _one_line(message)
    lowered = text.lower()
    if _is_candidate_timeout(text):
        return ("candidate_timeout", "")
    if "expected exactly one full terminal summary block" in lowered:
        return ("missing_metric_summary", "")
    if "out of memory" in lowered or "cuda oom" in lowered:
        return ("gpu_memory", "")
    if re.search(r"(?:^|\s)uv(?::|\s).*not found", lowered):
        return ("missing_uv", "")
    if "stopped by user" in lowered:
        return ("stopped_by_user", "")
    normalized = re.sub(
        r"^experiment\s+\d+\s+failed:\s*", "", text, flags=re.IGNORECASE
    )
    return ("other", normalized[:180])


def _counted_noun(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def _experiment_error_note(key: tuple[str, str], count: int) -> str:
    kind, detail = key
    subject = _counted_noun(count, "experiment")
    if kind == "missing_metric_summary":
        return f"{subject.capitalize()} did not produce a complete terminal metric summary, so no score was used."
    if kind == "gpu_memory":
        return f"{subject.capitalize()} ran out of GPU memory and produced no usable score."
    if kind == "missing_uv":
        return f"{subject.capitalize()} could not start because the `uv` runtime was unavailable."
    if kind == "stopped_by_user":
        verb = "was" if count == 1 else "were"
        return f"{subject.capitalize()} {verb} stopped by the user before producing a score."
    return f"{subject.capitalize()} ended without a usable score: {detail or 'no error detail was recorded'}."


def _operational_notes(
    experiments: Sequence[Mapping[str, Any]], logs: Sequence[Mapping[str, Any]]
) -> list[str]:
    timeout_logs = [log for log in logs if _is_candidate_timeout(log.get("message"))]
    timeout_experiments = [
        item for item in experiments if _is_candidate_timeout(item.get("error"))
    ]
    timeout_count = len(timeout_logs) or len(timeout_experiments)
    notes: list[str] = []
    if timeout_count:
        attempts = _counted_noun(timeout_count, "candidate-generation attempt")
        notes.append(
            f"Codex timed out during {attempts}. Those attempts produced no metric and do not change the recommendation."
        )

    experiment_groups: dict[tuple[str, str], int] = {}
    for item in experiments:
        if not item.get("error"):
            continue
        key = _error_key(item.get("error"))
        if key[0] == "candidate_timeout":
            continue
        experiment_groups[key] = experiment_groups.get(key, 0) + 1

    # Some research-level failures have no experiment row. Include those once, while
    # avoiding the duplicate log emitted for an experiment failure.
    log_groups: dict[tuple[str, str], int] = {}
    for log in logs:
        if str(log.get("level", "")).lower() != "error":
            continue
        key = _error_key(log.get("message"))
        if key[0] == "candidate_timeout" or key in experiment_groups:
            continue
        log_groups[key] = log_groups.get(key, 0) + 1

    grouped = sorted(
        (*experiment_groups.items(), *log_groups.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )
    for key, count in grouped[:2]:
        notes.append(_experiment_error_note(key, count))
    return notes


def generate_article_markdown(
    research: Mapping[str, Any],
    experiments: Sequence[Mapping[str, Any]],
    logs: Sequence[Mapping[str, Any]],
) -> str:
    """Turn persisted run facts into a concise, evidence-backed recommendation."""
    ordered_experiments = sorted(experiments, key=_experiment_number)
    lower_is_better = _is_lower_better(research)
    metric = _one_line(research.get("metric_name")) or "the tracked metric"
    is_prediction_loss = metric.lower() == "val_bpb"
    metric_label = "prediction loss" if is_prediction_loss else f"`{metric}`"
    baseline = _baseline_value(research, ordered_experiments)
    best = _best_value(research, ordered_experiments, lower_is_better=lower_is_better)
    best_item = _best_experiment(research, ordered_experiments, best)
    improved = (
        baseline is not None
        and best is not None
        and _is_better(best, baseline, lower_is_better=lower_is_better)
    )
    unresolved = [
        item
        for item in ordered_experiments
        if item.get("accepted") is None and not item.get("error")
    ]

    opening: list[str] = []
    if improved and best_item is not None:
        experiment_number = _experiment_number(best_item)
        opening.append(
            f"Use the accepted configuration from experiment {experiment_number}. It reached "
            f"the best recorded {metric_label} of **{_format_metric(best)}**."
        )
    elif improved:
        opening.append(
            f"The best persisted {metric_label} is **{_format_metric(best)}**, but the saved "
            "records do not link that score to an accepted experiment. Treat it as a result "
            "to verify, not yet as a configuration recommendation."
        )
    elif baseline is not None:
        opening.append(
            f"Keep the baseline as the reference for now. It measured **{_format_metric(baseline)}** "
            f"on {metric_label}, and no persisted change has beaten it."
        )
    else:
        opening.append(
            "There is not enough measured evidence to recommend a configuration yet. No valid "
            f"{metric_label} score has been persisted."
        )

    if improved and baseline is not None and best is not None:
        absolute_gain = baseline - best if lower_is_better else best - baseline
        relative_gain = (
            absolute_gain / abs(baseline) * 100
            if not math.isclose(baseline, 0.0)
            else None
        )
        movement = "fell" if lower_is_better else "rose"
        comparison_subject = "prediction loss" if is_prediction_loss else "the score"
        comparison = (
            f"From the **{_format_metric(baseline)}** baseline, {comparison_subject} {movement} by "
            f"**{absolute_gain:.6f}**"
        )
        if relative_gain is not None:
            comparison += f", or **{relative_gain:.2f}%**"
        opening.append(comparison + ".")

    lines = [" ".join(opening)]

    if unresolved:
        subject = _counted_noun(len(unresolved), "newer candidate")
        verb = "has" if len(unresolved) == 1 else "have"
        lines.extend(
            [
                "",
                f"{subject.capitalize()} {verb} no completed measurement and is not part of this recommendation."
                if len(unresolved) == 1
                else f"{subject.capitalize()} {verb} no completed measurements and are not part of this recommendation.",
            ]
        )

    objective = _one_line(research.get("objective") or research.get("original_prompt"))
    if is_prediction_loss:
        objective = _prediction_loss_language(objective)
    lines.extend(["", "## What this study was trying to do", ""])
    if objective:
        lines.extend([objective, ""])
    if is_prediction_loss:
        lines.append(
            "`val_bpb` is the raw name for **validation bits per byte**, the prediction loss "
            "used here. In plain language, it estimates how many bits the model needs to "
            "encode each byte of held-out validation text. Lower prediction loss is better, "
            "and comparisons are meaningful when the evaluation setup is the same."
        )
    else:
        direction = "lower" if lower_is_better else "higher"
        lines.append(
            f"The saved comparison rule says **{direction} is better** for `{metric}`. "
            "This summary uses that rule and does not infer any broader quality claim."
        )

    settings = _final_settings(
        ordered_experiments,
        baseline,
        best_item,
        lower_is_better=lower_is_better,
    )
    if improved and best_item is not None:
        lines.extend(["", "## Recommended configuration", ""])
        commit = _one_line(best_item.get("git_commit")) or _one_line(
            research.get("best_git_commit")
        )
        if commit:
            lines.append(
                f"The reproducible checkpoint is accepted commit `{commit}`. Its key persisted settings are:"
            )
        else:
            lines.append(
                "The key persisted settings in the best accepted configuration are:"
            )
        if settings:
            lines.append("")
            for item in settings:
                description = _setting_description(item).rstrip(".")
                if is_prediction_loss:
                    description = _prediction_loss_language(description)
                lines.append(f"- {description}.")
        else:
            summary = _clean_change_summary(best_item.get("change_summary")).rstrip(".")
            if is_prediction_loss:
                summary = _prediction_loss_language(summary)
            lines.extend(
                [
                    "",
                    f"- {summary}.",
                ]
            )

    rejections = _useful_rejections(
        ordered_experiments, lower_is_better=lower_is_better
    )
    if rejections:
        lines.extend(["", "## What did not help", ""])
        lines.append(
            "These completed comparisons are the useful boundaries; unmeasured failures are not treated as model evidence:"
        )
        lines.append("")
        for item in rejections:
            number = _experiment_number(item)
            candidate = _candidate_description(item)
            if is_prediction_loss:
                candidate = _prediction_loss_language(candidate)
            lines.append(
                f"- Experiment {number} tested {candidate}. It scored "
                f"**{_format_metric(item.get('metric_value'))}** against the then-best "
                f"**{_format_metric(item.get('previous_best'))}**, so it was not retained."
            )

    notes = _operational_notes(ordered_experiments, logs)
    if notes:
        lines.extend(["", "## Run notes", ""])
        if is_prediction_loss:
            notes = [_prediction_loss_language(note) for note in notes]
        lines.extend(f"- {note}" for note in notes)

    return "\n".join(lines).rstrip() + "\n"
