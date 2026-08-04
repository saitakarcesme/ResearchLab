from __future__ import annotations

from backend.article import generate_article_markdown


def test_article_leads_with_a_plain_recommendation_and_measured_gain() -> None:
    research = {
        "title": "Single-3090 Harness Efficiency",
        "original_prompt": "Build the most efficient single-3090 harness.",
        "objective": (
            "Minimize `val_bpb` in the fixed five-minute evaluation while editing only "
            "`train.py`."
        ),
        "metric_name": "val_bpb",
        "metric_direction": "lower_is_better",
        "baseline_value": 1.275720,
        "best_value": 1.113376,
        "best_git_commit": "best-commit",
    }
    experiments = [
        {
            "experiment_number": 1,
            "hypothesis": "Establish the baseline",
            "change_summary": "Baseline with the runner-managed batch profile.",
            "metric_value": 1.275720,
            "previous_best": None,
            "accepted": True,
            "git_commit": "baseline-commit",
            "error": None,
        },
        {
            "experiment_number": 2,
            "hypothesis": "Use less accumulation",
            "change_summary": "Changed TOTAL_BATCH_SIZE from 2**19 to 2**16.",
            "metric_value": 1.129309,
            "previous_best": 1.275720,
            "accepted": True,
            "git_commit": "batch-commit",
            "error": None,
        },
        {
            "experiment_number": 3,
            "hypothesis": "Reduce short attention",
            "change_summary": (
                "Changed the short-window calculation to produce 128-token windows for "
                "layers 0-6 while preserving full-context attention in layer 7."
            ),
            "metric_value": 1.113808,
            "previous_best": 1.129309,
            "accepted": True,
            "git_commit": "window-128",
            "error": None,
        },
        {
            "experiment_number": 4,
            "hypothesis": "Find the final short-window size",
            "change_summary": (
                "Changed the seven short-attention layers from 64-token to 96-token "
                "windows while preserving the final 2,048-token full-context layer and "
                "the model forward/evaluation contract."
            ),
            "metric_value": 1.113376,
            "previous_best": 1.113808,
            "accepted": True,
            "git_commit": "best-commit",
            "error": None,
        },
        {
            "experiment_number": 5,
            "hypothesis": "Check the lower window boundary",
            "change_summary": (
                "Changed the divisor to 64, producing 32-token short windows. The model "
                "forward and evaluation contract remain unchanged."
            ),
            "metric_value": 1.116608,
            "previous_best": 1.113376,
            "accepted": False,
            "git_commit": "too-short",
            "error": None,
        },
        {
            "experiment_number": 6,
            "hypothesis": "Try higher beta1",
            "change_summary": "Changed ADAM_BETAS from (0.8, 0.95) to (0.9, 0.95).",
            "metric_value": 1.134752,
            "previous_best": 1.113376,
            "accepted": False,
            "git_commit": "beta-test",
            "error": None,
        },
        {
            "experiment_number": 7,
            "hypothesis": "Try 88 tokens",
            "change_summary": "Changed the short-attention layers to 88-token windows.",
            "metric_value": None,
            "previous_best": 1.113376,
            "accepted": None,
            "git_commit": None,
            "error": None,
        },
    ]
    logs = [
        {
            "level": "error",
            "message": "Codex candidate generation exceeded its time limit",
        }
        for _ in range(56)
    ]

    markdown = generate_article_markdown(research, experiments, logs)

    assert markdown.startswith(
        "If I were setting this up, I would start with the configuration saved after "
        "experiment **4**."
    )
    assert not markdown.startswith("#")
    assert "validation bits per byte" in markdown
    assert "Think of it as how surprised the model is" in markdown
    assert markdown.count("val_bpb") == 1
    assert "practical takeaway is simple" in markdown
    assert "model became less uncertain" in markdown
    assert "1.275720" in markdown
    assert "1.113376" in markdown
    assert "0.162344" in markdown
    assert "12.73%" in markdown
    assert "experiment **4**" in markdown
    assert "`best-commit`" in markdown
    assert (
        "**Training scale and schedule:** Set total batch size to `2**16`" in markdown
    )
    assert "how much training data is combined before each optimizer update" in markdown
    assert (
        "**Context and targets:** Set short-attention window to `96` tokens for the "
        "first seven layers" in markdown
    )
    assert "how much nearby text the early layers inspect" in markdown
    assert "`2,048`-token full context" in markdown
    assert "128-token windows" not in markdown
    assert "88-token windows" not in markdown
    assert "Skip adam betas `(0.9, 0.95)`" in markdown
    assert "experiment **5**" in markdown
    assert "56 candidate-generation attempts" in markdown
    assert markdown.lower().count("codex timed out") == 1
    assert "## The setup I would copy" in markdown
    assert "## Why I would trust this result" in markdown
    assert "## What I would leave out" in markdown
    assert "In experiment 2, this step lowered" not in markdown
    assert "What we tried, one run at a time" not in markdown
    assert "state of the art" not in markdown.lower()


def test_article_collapses_unmeasured_failures_without_treating_them_as_evidence() -> (
    None
):
    research = {
        "title": "Parser retries",
        "objective": "Measure val_bpb truthfully.",
        "metric_name": "val_bpb",
        "metric_direction": "lower_is_better",
        "baseline_value": None,
        "best_value": None,
        "best_git_commit": None,
    }
    experiments = [
        {
            "experiment_number": number,
            "hypothesis": "Retry the harness",
            "change_summary": "No measured change was produced.",
            "metric_value": None,
            "previous_best": None,
            "accepted": False,
            "git_commit": None,
            "error": "expected exactly one full terminal summary block, found 0.",
        }
        for number in range(1, 4)
    ]
    logs = [
        {
            "level": "error",
            "message": (
                f"Experiment {number} failed: expected exactly one full terminal summary "
                "block, found 0."
            ),
        }
        for number in range(1, 4)
    ]

    markdown = generate_article_markdown(research, experiments, logs)

    assert "not have enough measured evidence" in markdown
    assert (
        "3 experiments did not produce a complete terminal metric summary" in markdown
    )
    assert "expected exactly one full terminal summary block" not in markdown
    assert "## What I would leave out" not in markdown
    assert "not available" not in markdown


def test_article_does_not_present_the_baseline_as_an_improvement() -> None:
    research = {
        "title": "Baseline only",
        "objective": "Establish a comparable baseline.",
        "metric_name": "val_bpb",
        "metric_direction": "lower_is_better",
        "baseline_value": 1.318191,
        "best_value": 1.318191,
        "best_git_commit": "baseline",
    }
    experiments = [
        {
            "experiment_number": 1,
            "hypothesis": "Establish baseline",
            "change_summary": "Pinned baseline configuration.",
            "metric_value": 1.318191,
            "previous_best": None,
            "accepted": True,
            "git_commit": "baseline",
            "error": None,
        }
    ]
    logs = [
        {
            "level": "error",
            "message": "prepare failed: sh: 1: uv: not found",
        }
    ]

    markdown = generate_article_markdown(research, experiments, logs)

    assert "The honest answer is to keep the baseline for now" in markdown
    assert "none of the completed, persisted changes beat it" in markdown
    assert "## The setup I would copy" not in markdown
    assert "could not start because the `uv` runtime was unavailable" in markdown
    assert "Across this run" not in markdown


def test_article_handles_a_higher_is_better_metric_without_inverting_the_gain() -> None:
    research = {
        "title": "Accuracy run",
        "objective": "Increase held-out accuracy.",
        "metric_name": "accuracy",
        "metric_direction": "higher_is_better",
        "baseline_value": 0.5,
        "best_value": 0.6,
        "best_git_commit": "accuracy-best",
    }
    experiments = [
        {
            "experiment_number": 1,
            "hypothesis": "Baseline",
            "change_summary": "Baseline.",
            "metric_value": 0.5,
            "previous_best": None,
            "accepted": True,
            "git_commit": "accuracy-baseline",
            "error": None,
        },
        {
            "experiment_number": 2,
            "hypothesis": "Try a persisted change",
            "change_summary": "Enabled feature fusion.",
            "metric_value": 0.6,
            "previous_best": 0.5,
            "accepted": True,
            "git_commit": "accuracy-best",
            "error": None,
        },
    ]

    markdown = generate_article_markdown(research, experiments, [])

    assert (
        "It raised `accuracy` from **0.500000** to **0.600000** by "
        "**0.100000** (**20.00%**)" in markdown
    )
    assert "higher as better" in markdown.lower()


def test_article_keeps_the_changed_setting_despite_boilerplate_mentions() -> None:
    research = {
        "title": "Setting summary",
        "objective": "Minimize val_bpb.",
        "metric_name": "val_bpb",
        "metric_direction": "lower_is_better",
        "baseline_value": 1.2,
        "best_value": 1.1,
        "best_git_commit": "warmdown-best",
    }
    experiments = [
        {
            "experiment_number": 1,
            "hypothesis": "Baseline",
            "change_summary": "Baseline.",
            "metric_value": 1.2,
            "previous_best": None,
            "accepted": True,
            "git_commit": "baseline",
            "error": None,
        },
        {
            "experiment_number": 2,
            "hypothesis": "Reduce accumulation",
            "change_summary": "Changed TOTAL_BATCH_SIZE from 2**19 to 2**16.",
            "metric_value": 1.11,
            "previous_best": 1.2,
            "accepted": True,
            "git_commit": "batch",
            "error": None,
        },
        {
            "experiment_number": 3,
            "hypothesis": "Lengthen warmdown",
            "change_summary": (
                "Changed WARMDOWN_RATIO from 0.5 to 0.7; model, batch size, optimizer "
                "peak rates, forward method, and evaluation contract remain unchanged."
            ),
            "metric_value": 1.1,
            "previous_best": 1.11,
            "accepted": True,
            "git_commit": "warmdown-best",
            "error": None,
        },
    ]

    markdown = generate_article_markdown(research, experiments, [])

    assert (
        "**Training scale and schedule:** Set total batch size to `2**16`" in markdown
    )
    assert "Set warmdown ratio to `0.7`" in markdown
    assert (
        "how much of the run is spent gradually reducing the learning rate" in markdown
    )
