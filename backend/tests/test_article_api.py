from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from backend.db import Database, article_title_slug
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
        listing = client.get("/api/articles")

    assert response.status_code == 200
    payload = response.json()
    assert payload["baseline_value"] == 1.25
    assert payload["best_value"] == 1.1
    assert payload["metric_direction"] == "lower_is_better"
    assert payload["article_kind"] == "general"
    assert payload["objective"] == "Lower prediction loss."
    assert len(payload["experiments"]) == 1
    assert payload["experiments"][0]["metric_value"] == 1.1
    assert listing.status_code == 200
    assert listing.json()[0]["article_kind"] == "general"
    assert "original_prompt" not in listing.json()[0]


def test_article_title_slug_is_readable_for_turkish_and_non_latin_titles() -> None:
    assert article_title_slug("Okuma Alışkanlığı: 30 Gün!") == "okuma-aliskanligi-30-gun"
    assert article_title_slug("研究 結果") == "研究-結果"
    assert article_title_slug("---") == "article"


def test_article_detail_accepts_slugs_ids_and_disambiguates_duplicate_titles(
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
    articles = []
    for index in range(2):
        research = database.create_research(
            {
                "title": "Okuma Alışkanlığı",
                "original_prompt": f"Reading prompt {index}",
                "objective": "Build a lasting reading habit.",
                "gpu_source_id": source["id"],
                "target_gpu_allocation": 100,
            }
        )
        articles.append(
            database.upsert_article(
                research["id"], "Okuma Alışkanlığı", "A readable article.\n"
            )
        )

    slugs = {article["slug"] for article in database.list_articles()}
    assert "okuma-aliskanligi" in slugs
    assert len(slugs) == 2
    assert all(slug.startswith("okuma-aliskanligi") for slug in slugs)

    with TestClient(create_app(configured)) as client:
        for article in articles:
            by_id = client.get(f"/api/articles/{article['id']}")
            assert by_id.status_code == 200
            canonical_slug = by_id.json()["slug"]
            by_slug = client.get(f"/api/articles/{canonical_slug}")
            assert by_slug.status_code == 200
            assert by_slug.json()["id"] == article["id"]

        missing = client.get("/api/articles/not-a-real-article")

    assert missing.status_code == 404
