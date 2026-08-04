from __future__ import annotations

from backend.article import generate_article_markdown


def test_article_uses_only_persisted_experiments_and_logs() -> None:
    research = {
        "title": "Coding Harness",
        "objective": "Minimize measured validation bits per byte.",
        "metric_name": "val_bpb",
        "metric_direction": "lower_is_better",
        "baseline_value": 1.200000,
        "best_value": 1.100000,
    }
    experiments = [
        {
            "experiment_number": 1,
            "hypothesis": "Establish baseline",
            "change_summary": "Pinned baseline",
            "metric_value": 1.2,
            "previous_best": None,
            "accepted": True,
            "git_commit": "aaaaaaa",
            "error": None,
        },
        {
            "experiment_number": 2,
            "hypothesis": "Tune optimizer",
            "change_summary": "Changed matrix LR",
            "metric_value": 1.1,
            "previous_best": 1.2,
            "accepted": True,
            "git_commit": "bbbbbbb",
            "error": None,
        },
        {
            "experiment_number": 3,
            "hypothesis": "Increase depth",
            "change_summary": "Raised depth",
            "metric_value": None,
            "previous_best": 1.1,
            "accepted": False,
            "git_commit": "ccccccc",
            "error": "CUDA out of memory",
        },
    ]
    logs = [{"level": "error", "message": "CUDA out of memory"}]
    markdown = generate_article_markdown(research, experiments, logs)
    for persisted_fact in (
        "Minimize measured validation bits per byte.",
        "Changed matrix LR",
        "1.100000",
        "bbbbbbb",
        "CUDA out of memory",
    ):
        assert persisted_fact in markdown
    assert "state of the art" not in markdown.lower()
    assert "significant breakthrough" not in markdown.lower()
    assert not markdown.startswith("# ")
    assert "## The question behind the run" in markdown
    assert "## What we tried, one run at a time" in markdown
    assert "## Where the run landed" in markdown
    assert "There are 2 accepted, 0 rejected, and 1 failed experiments." in markdown


def test_article_does_not_call_an_incomplete_experiment_rejected() -> None:
    research = {
        "title": "Interrupted",
        "objective": "Measure val_bpb truthfully.",
        "metric_name": "val_bpb",
        "metric_direction": "lower_is_better",
        "baseline_value": None,
        "best_value": None,
    }
    experiment = {
        "experiment_number": 1,
        "hypothesis": "Candidate still running",
        "change_summary": "No outcome yet",
        "metric_value": None,
        "previous_best": None,
        "accepted": None,
        "git_commit": None,
        "error": None,
    }
    markdown = generate_article_markdown(research, [experiment], [])
    assert "running or interrupted" in markdown
    assert "No outcome yet (rejected" not in markdown


def test_article_collapses_repeated_recorded_errors() -> None:
    research = {
        "title": "Retries",
        "objective": "Keep retry evidence readable.",
        "metric_name": "val_bpb",
        "metric_direction": "lower_is_better",
        "baseline_value": 1.2,
        "best_value": 1.2,
    }
    logs = [
        {"level": "error", "message": "Candidate generation timed out"},
        {"level": "error", "message": "Candidate generation timed out"},
    ]
    markdown = generate_article_markdown(research, [], logs)
    assert markdown.count("Candidate generation timed out") == 1
    assert "recorded 2 times" in markdown
