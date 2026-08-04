from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from backend.db import Database
from backend.main import create_app


def test_article_detail_includes_the_research_data_required_by_the_chart(
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
            "title": "Readable result",
            "original_prompt": "Find a better configuration",
            "objective": "Lower prediction loss.",
            "gpu_source_id": source["id"],
            "target_gpu_allocation": 100,
        }
    )
    experiment = database.create_experiment(
        research["id"], "Try the measured change", "Changed one setting", 1.25
    )
    database.finish_experiment(
        experiment["id"], metric_value=1.1, accepted=True, git_commit="abc123"
    )
    database.update_research(
        research["id"],
        {
            "baseline_value": 1.25,
            "best_value": 1.1,
            "best_git_commit": "abc123",
        },
    )
    article = database.upsert_article(
        research["id"], "Readable result", "A persisted article.\n"
    )

    with TestClient(create_app(configured)) as client:
        response = client.get(f"/api/articles/{article['id']}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["baseline_value"] == 1.25
    assert payload["best_value"] == 1.1
    assert payload["metric_direction"] == "lower_is_better"
    assert payload["objective"] == "Lower prediction loss."
    assert len(payload["experiments"]) == 1
    assert payload["experiments"][0]["metric_value"] == 1.1
