from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from backend.main import create_app
from backend.researcher_models import (
    researcher_model_catalog,
    researcher_model_command_args,
    resolve_researcher_model,
)


def test_tool_capable_ollama_models_are_available_as_research_agents(settings) -> None:
    local_models = [
        {
            "id": "ollama:qwen3:4b",
            "name": "qwen3:4b",
            "parameter_size": "4.0B",
            "capabilities": ["completion", "tools"],
        },
        {
            "id": "ollama:embedding-only",
            "name": "embedding-only",
            "capabilities": ["embedding"],
        },
    ]

    catalog = researcher_model_catalog(settings, local_models)
    local = next(model for model in catalog["models"] if model["provider"] == "ollama")
    assert local["id"] == "ollama:qwen3:4b"
    assert local["parameter_size"] == "4.0B"
    assert resolve_researcher_model(settings, local["id"], local_models) == local["id"]
    assert researcher_model_command_args(local["id"]) == [
        "--oss",
        "--local-provider",
        "ollama",
        "--model",
        "qwen3:4b",
    ]


def test_lab_persists_a_research_agent_separately_from_the_target_model(
    settings, monkeypatch
) -> None:
    configured = replace(settings, execution_enabled=False)
    monkeypatch.setattr(
        "backend.main.TelemetryService.sample",
        lambda self, source: {"available": False, "gpu_name": "Test GPU"},
    )
    selected = configured.researcher_models[-1]
    with TestClient(create_app(configured)) as client:
        catalog = client.get("/api/researcher-models")
        assert catalog.status_code == 200
        assert selected in {item["id"] for item in catalog.json()["models"]}
        source_id = client.get("/api/gpu-sources").json()[0]["id"]
        created = client.post(
            "/api/researches",
            json={
                "original_prompt": "Improve the pinned training harness",
                "gpu_source_id": source_id,
                "target_gpu_allocation": 100,
                "research_type": "training_optimization",
                "researcher_model_id": selected,
                "auto_start": False,
            },
        )
        assert created.status_code == 201, created.text
        payload = created.json()
        assert payload["researcher_model_id"] == selected
        assert payload["model_id"] is None


def test_lab_rejects_an_unconfigured_research_agent(settings, monkeypatch) -> None:
    configured = replace(settings, execution_enabled=False)
    monkeypatch.setattr(
        "backend.main.TelemetryService.sample",
        lambda self, source: {"available": False, "gpu_name": "Test GPU"},
    )
    with TestClient(create_app(configured)) as client:
        source_id = client.get("/api/gpu-sources").json()[0]["id"]
        response = client.post(
            "/api/researches",
            json={
                "original_prompt": "Improve the pinned training harness",
                "gpu_source_id": source_id,
                "researcher_model_id": "not-in-the-catalog",
                "auto_start": False,
            },
        )
    assert response.status_code == 409
    assert "not enabled" in response.json()["detail"]
