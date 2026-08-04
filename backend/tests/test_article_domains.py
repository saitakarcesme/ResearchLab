from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from backend.article import (
    ResearchArticleGenerator,
    article_kind,
    generate_article_markdown,
    generate_general_article_markdown,
)
from backend.db import Database
from backend.main import create_app


def test_article_kind_uses_the_original_question_instead_of_adapter_boilerplate() -> (
    None
):
    assert (
        article_kind(
            {
                "original_prompt": "Ways to develop a habit of reading",
                "objective": "Modify train.py to minimize val_bpb.",
            }
        )
        == "general"
    )
    assert (
        article_kind(
            {
                "original_prompt": (
                    "Optimize a training harness for one RTX 3090 without exceeding "
                    "24 GB VRAM"
                )
            }
        )
        == "technical"
    )
    assert (
        article_kind(
            {
                "original_prompt": (
                    "The most efficient harness that can be set up with a single 3090"
                )
            }
        )
        == "technical"
    )
    assert article_kind({"original_prompt": "Düzenli okuma alışkanlığı nasıl kurulur?"}) == (
        "general"
    )


def test_local_model_research_is_a_reader_friendly_technical_article() -> None:
    research = {
        "title": "Test Model Local Inference Efficiency",
        "original_prompt": "Measure the speed of this local language model",
        "objective": "Compare reproducible warm generation profiles.",
        "research_type": "local_model_benchmark",
        "model_id": "ollama:test-model:latest",
        "metric_name": "output_tokens_per_second",
        "metric_direction": "higher_is_better",
        "baseline_value": 40.0,
        "best_value": 52.0,
    }
    experiments = [
        {
            "experiment_number": 1,
            "hypothesis": "Establish the compact-context baseline.",
            "change_summary": "Run three times with a 2,048-token context and batch 256.",
            "metric_value": 40.0,
            "previous_best": None,
            "accepted": True,
            "error": None,
        },
        {
            "experiment_number": 2,
            "hypothesis": "A larger processing batch may improve throughput.",
            "change_summary": "Run three times with a 2,048-token context and batch 512.",
            "metric_value": 52.0,
            "previous_best": 40.0,
            "accepted": True,
            "error": None,
        },
    ]

    markdown = generate_article_markdown(research, experiments, [])

    assert article_kind(research) == "technical"
    assert "generation speed" in markdown
    assert "`test-model:latest`" in markdown
    assert "does not compare answer quality" in markdown
    assert "output_tokens_per_second" not in markdown


def test_general_fallback_ignores_unrelated_software_experiments(settings) -> None:
    generator = ResearchArticleGenerator(replace(settings, execution_enabled=False))
    research = {
        "title": "Reading Habit Guidance",
        "original_prompt": "Ways to develop a habit of reading",
        "objective": "Modify train.py to minimize val_bpb.",
    }
    experiments = [
        {
            "experiment_number": 4,
            "change_summary": "Changed optimizer and batch size.",
            "git_commit": "secret-commit",
            "accepted": True,
            "metric_value": 1.1,
        }
    ]

    markdown = generator.generate(research, experiments, [])

    assert markdown.startswith("A reading habit is easier to grow")
    assert "## Make the next page easy to reach" in markdown
    assert "## Try one gentle two-week routine" in markdown
    assert "findings from a study conducted with people" in markdown
    for leaked_term in (
        "train.py",
        "val_bpb",
        "optimizer",
        "batch size",
        "secret-commit",
        "experiment **4**",
    ):
        assert leaked_term not in markdown


def test_turkish_reading_fallback_keeps_the_reader_language(settings) -> None:
    generator = ResearchArticleGenerator(replace(settings, execution_enabled=False))

    markdown = generator.generate(
        {
            "title": "Okuma Alışkanlığı",
            "original_prompt": "Düzenli okuma alışkanlığı nasıl kurulur?",
        },
        [],
        [],
    )

    assert markdown.startswith("Okuma alışkanlığı")
    assert "## Bir sonraki sayfayı yakınlaştırın" in markdown
    assert "Kapsam notu" in markdown
    assert "A reading habit" not in markdown


def test_general_codex_article_is_schema_constrained_and_reader_facing(
    settings, monkeypatch
) -> None:
    observed: dict[str, object] = {}
    generated = """You do not need to force reading into every spare minute. A more inviting start is to give one book a dependable place in your day and let curiosity, rather than guilt, carry the session forward. The aim is to make returning feel ordinary.

## Give the book a natural cue

Put the book beside something you already reach for and choose a moment that is usually calm enough for a few pages. Keep the promise small on busy days. If the book repeatedly feels like homework, choose one that better matches the kind of evening or commute you actually have. That is an adjustment, not a failure.

## Notice what makes you return

For two weeks, note whether you opened the book and one sentence about the experience. Pay attention to the setting, the time, and whether the book itself held your interest. At the end of the week, change only one obstacle. You might move the book, shorten the session, or change the title.

## Let missed days stay small

Do not repay a missed day with an oversized reading target. Simply return at the next cue. The useful pattern is not an unbroken performance; it is learning how to begin again without turning the book into a debt."""

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["input"] = kwargs["input"]
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text(json.dumps({"markdown": generated}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("backend.article.subprocess.run", fake_run)
    generator = ResearchArticleGenerator(replace(settings, execution_enabled=True))
    markdown = generator.generate(
        {
            "title": "Reading Habit Guidance",
            "original_prompt": "Ways to develop a habit of reading",
            "objective": "Do not expose this unrelated train.py objective.",
        },
        [{"git_commit": "do-not-pass-this-commit-to-codex"}],
        [{"message": "do-not-pass-this-log-to-codex"}],
    )

    assert markdown.startswith("You do not need to force reading")
    assert markdown.count("## ") == 3
    assert markdown.endswith("findings from a study conducted with people.\n")
    command = observed["command"]
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--output-schema" in command
    sent = observed["input"]
    assert b"Ways to develop a habit of reading" in sent
    assert b"do-not-pass-this-commit-to-codex" not in sent
    assert b"do-not-pass-this-log-to-codex" not in sent


def test_invalid_general_ai_copy_falls_back_without_leaking_run_details(
    settings, monkeypatch
) -> None:
    def fake_run(command, **kwargs):
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text(
            json.dumps(
                {
                    "markdown": (
                        "Start from commit abc and change train.py.\n\n"
                        "## Setup\n\nTechnical leak.\n\n## Result\n\nStill a leak."
                    )
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("backend.article.subprocess.run", fake_run)
    generator = ResearchArticleGenerator(replace(settings, execution_enabled=True))
    markdown = generator.generate(
        {
            "title": "Reading Habit Guidance",
            "original_prompt": "Ways to develop a habit of reading",
        },
        [],
        [],
    )

    assert markdown == generate_general_article_markdown(
        {
            "title": "Reading Habit Guidance",
            "original_prompt": "Ways to develop a habit of reading",
        }
    )
    assert "commit" not in markdown
    assert "train.py" not in markdown


def test_technical_prompt_keeps_the_measured_article_path(settings, monkeypatch) -> None:
    def unexpected_run(*args, **kwargs):
        raise AssertionError("Technical articles must not use the general prose generator")

    monkeypatch.setattr("backend.article.subprocess.run", unexpected_run)
    generator = ResearchArticleGenerator(replace(settings, execution_enabled=True))
    markdown = generator.generate(
        {
            "title": "RTX Harness",
            "original_prompt": "Optimize an RTX 3090 training harness",
            "objective": "Lower val_bpb.",
            "metric_name": "val_bpb",
            "baseline_value": None,
            "best_value": None,
        },
        [],
        [],
    )

    assert "not have enough measured evidence" in markdown
    assert "validation bits per byte" in markdown


def test_article_endpoint_applies_the_general_reader_contract(
    settings, monkeypatch
) -> None:
    configured = replace(settings, execution_enabled=False)
    monkeypatch.setattr(
        "backend.main.TelemetryService.sample",
        lambda self, source: {"available": False, "gpu_name": "Test GPU"},
    )
    database = Database(configured.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Test GPU")
    research = database.create_research(
        {
            "title": "Reading Habit Guidance",
            "original_prompt": "Ways to develop a habit of reading",
            "objective": "Modify train.py to minimize val_bpb.",
            "gpu_source_id": source["id"],
            "target_gpu_allocation": 100,
        }
    )
    experiment = database.create_experiment(
        research["id"],
        "Change batch size",
        "Changed optimizer and batch size.",
        1.25,
    )
    database.finish_experiment(
        experiment["id"], metric_value=1.1, accepted=True, git_commit="abc123"
    )

    with TestClient(create_app(configured)) as client:
        response = client.post(f"/api/researches/{research['id']}/article")

    assert response.status_code == 200
    assert response.json()["article_kind"] == "general"
    markdown = response.json()["markdown"]
    assert "## Make the next page easy to reach" in markdown
    assert "abc123" not in markdown
    assert "optimizer" not in markdown
