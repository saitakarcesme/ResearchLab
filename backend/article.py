from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _format_metric(value: float | None) -> str:
    return "not available" if value is None else f"{value:.6f}"


def generate_article_markdown(
    research: Mapping[str, Any],
    experiments: Sequence[Mapping[str, Any]],
    logs: Sequence[Mapping[str, Any]],
) -> str:
    """Render persisted facts as an editorial research journal without invented claims."""
    metric = str(research["metric_name"])
    direction = str(research["metric_direction"]).replace("_", " ")
    accepted = [item for item in experiments if item.get("accepted") is True]
    rejected = [
        item
        for item in experiments
        if item.get("accepted") is False and not item.get("error")
    ]
    failures = [item for item in experiments if item.get("error")]
    unresolved = [
        item
        for item in experiments
        if item.get("accepted") is None and not item.get("error")
    ]
    completed_count = len(accepted) + len(rejected) + len(failures)

    lines = [
        "The starting point was straightforward.",
        "",
        f"> {research['objective']}",
        "",
        (
            "This journal stays close to the evidence: saved hypotheses, actual code "
            "changes, measured outcomes, and recorded errors. An unfinished run is left "
            "unfinished rather than presented as a result."
        ),
        "",
        "## The question behind the run",
        "",
        (
            f"The lab tracked `{metric}` ({direction}). The persisted baseline was "
            f"**{_format_metric(research.get('baseline_value'))}**."
        ),
        "",
        (
            "That made the rule for progress unambiguous: a candidate had to improve the "
            "recorded metric in the configured direction. Everything else remained useful "
            "as evidence, but it did not become the new best configuration."
        ),
        "",
        "## What we tried, one run at a time",
        "",
    ]
    if experiments:
        for item in experiments:
            if item.get("error"):
                outcome = "failed"
            elif item.get("accepted") is True:
                outcome = "accepted"
            elif item.get("accepted") is False:
                outcome = "rejected"
            else:
                outcome = "running or interrupted"
            metric_text = _format_metric(item.get("metric_value"))
            lines.extend(
                [
                    f"### Experiment {item['experiment_number']} — {outcome}",
                    "",
                    f"**Hypothesis.** {item['hypothesis']}",
                    "",
                    f"**Change.** {item['change_summary']}",
                    "",
                ]
            )
            if item.get("error"):
                lines.append(f"**Outcome.** The run failed: {item['error']}")
            elif item.get("accepted") is None:
                lines.append("**Outcome.** No completed measurement was persisted.")
            else:
                lines.append(
                    f"**Outcome.** `{metric}` finished at **{metric_text}** and the "
                    f"candidate was {outcome}."
                )
            lines.append("")
    else:
        lines.extend(
            [
                (
                    "No experiments have been persisted yet. The question is defined, but "
                    "the evidence trail has not started."
                ),
                "",
            ]
        )

    lines.extend(["## What held up", ""])
    if accepted:
        lines.extend(
            [
                "These changes cleared the metric rule and became part of the accepted line:",
                "",
            ]
        )
        for item in accepted:
            lines.append(
                f"- Experiment {item['experiment_number']}: {item['change_summary']} "
                f"(`{metric}` {_format_metric(item.get('metric_value'))}, commit "
                f"`{item.get('git_commit') or 'not recorded'}`)."
            )
    else:
        lines.append(
            "No accepted improvement is recorded yet. That is a result too: the baseline "
            "still stands."
        )

    lines.extend(["", "## What did not hold up", ""])
    if rejected or failures or unresolved:
        for item in rejected:
            lines.append(
                f"- Experiment {item['experiment_number']} was rejected: "
                f"{item['change_summary']} did not strictly improve the previous best "
                f"({_format_metric(item.get('previous_best'))})."
            )
        for item in failures:
            lines.append(
                f"- Experiment {item['experiment_number']} failed: {item['error']}."
            )
        for item in unresolved:
            lines.append(
                f"- Experiment {item['experiment_number']} remained running or interrupted: "
                f"{item['change_summary']}."
            )
    else:
        lines.append("No rejected, failed, or unresolved experiment is recorded.")

    lines.extend(
        [
            "",
            "## Where the run landed",
            "",
            (
                f"After {completed_count} completed "
                f"{'experiment' if completed_count == 1 else 'experiments'}, the best "
                f"recorded `{metric}` is "
                f"**{_format_metric(research.get('best_value'))}**. "
                f"There are {len(accepted)} accepted, {len(rejected)} rejected, and "
                f"{len(failures)} failed experiments."
            ),
            "",
            (
                f"{len(unresolved)} additional "
                f"{'run is' if len(unresolved) == 1 else 'runs are'} still unresolved."
                if unresolved
                else "Every persisted experiment in this snapshot has a recorded outcome."
            ),
            "",
            "## The configuration we kept",
            "",
        ]
    )
    best = next((item for item in reversed(accepted) if item.get("git_commit")), None)
    if best:
        lines.append(
            f"The current accepted head is commit `{best['git_commit']}` from experiment "
            f"{best['experiment_number']}: {best['change_summary']}"
        )
    else:
        lines.append("No accepted code commit is recorded.")

    lines.extend(["", "## The metric trail", ""])
    measured = [item for item in experiments if item.get("metric_value") is not None]
    if measured:
        for item in measured:
            if item.get("accepted") is True:
                outcome = "accepted"
            elif item.get("accepted") is False:
                outcome = "rejected"
            else:
                outcome = "unresolved"
            lines.append(
                f"- Experiment {item['experiment_number']}: "
                f"{_format_metric(item['metric_value'])} ({outcome})"
            )
    else:
        lines.append("No measured metric values are recorded.")

    meaningful_errors = [log for log in logs if log.get("level") == "error"]
    if meaningful_errors:
        error_counts: dict[str, int] = {}
        for log in meaningful_errors:
            message = str(log["message"])
            error_counts[message] = error_counts.get(message, 0) + 1
        lines.extend(["", "## Errors recorded along the way", ""])
        for message, count in error_counts.items():
            suffix = f" — recorded {count} times" if count > 1 else ""
            lines.append(f"- {message}{suffix}")

    lines.extend(
        [
            "",
            (
                "The boundary of the story is simple: it reports what the persisted "
                "measurements show, and no more."
            ),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
