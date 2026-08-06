from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.adapters.base import AdapterContext
from backend.adapters.ollama import OllamaBenchmarkAdapter
from backend.db import Database
from backend.local_models import OllamaClient, OllamaModel
from backend.main import create_app
from backend.supervisor import ResearchSupervisor
from backend.telemetry import TelemetryService


class FakeOllamaClient:
    base_url = "http://127.0.0.1:11434"

    def __init__(self) -> None:
        self.model = OllamaModel(
            id="ollama:test-model:latest",
            name="test-model:latest",
            digest="sha256:test-digest",
            size_bytes=4 * 1024**3,
            parameter_size="8B",
            quantization_level="Q4_K_M",
            family="test",
        )
        self.unloads = 0
        self.residency_checks = 0

    def resolve(self, model_id: str) -> OllamaModel:
        if model_id != self.model.id:
            raise ValueError(model_id)
        return self.model

    def show(self, name: str) -> dict[str, Any]:
        assert name == self.model.name
        return {"capabilities": ["completion"]}

    def completion_models(self) -> dict[str, Any]:
        item = self.model.as_dict()
        item.update(
            {
                "capabilities": ["completion"],
                "context_length": 8192,
                "recommended": True,
            }
        )
        return {
            "runtime": {
                "provider": "ollama",
                "endpoint": self.base_url,
                "version": "test",
            },
            "models": [item],
        }

    def generate(self, name: str, **kwargs: Any) -> dict[str, Any]:
        assert name == self.model.name
        options = kwargs["options"]
        output_tokens = int(options["num_predict"])
        speed = 40 + int(options["num_batch"]) / 64
        return {
            "eval_count": output_tokens,
            "eval_duration": int(output_tokens / speed * 1_000_000_000),
            "prompt_eval_count": 24,
            "prompt_eval_duration": 100_000_000,
            "load_duration": 0,
            "total_duration": 1_000_000_000,
        }

    def running_models(self) -> list[dict[str, Any]]:
        self.residency_checks += 1
        return [
            {
                "name": self.model.name,
                "size": self.model.size_bytes,
                "size_vram": self.model.size_bytes,
                "context_length": 4096,
            }
        ]

    def unload(self, name: str) -> None:
        assert name == self.model.name
        self.unloads += 1


def test_unload_waits_for_ollama_process_list_to_update(monkeypatch) -> None:
    client = OllamaClient()
    states = iter(
        [
            [{"name": "test-model:latest"}],
            [{"name": "test-model:latest"}],
            [],
        ]
    )
    unloaded: list[str] = []
    monkeypatch.setattr(client, "unload", unloaded.append)
    monkeypatch.setattr(client, "running_models", lambda: next(states))

    assert client.unload_and_wait(
        "test-model:latest", timeout_seconds=1, poll_interval_seconds=0
    )
    assert unloaded == ["test-model:latest"]


def _wait_for(predicate, timeout: float = 4.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for local-model research")


def _benchmark(database: Database, source_id: str, title: str) -> dict[str, Any]:
    return database.create_research(
        {
            "title": title,
            "original_prompt": "Measure this installed model",
            "objective": "Maximize reproducible warm output speed.",
            "metric_name": "output_tokens_per_second",
            "metric_direction": "higher_is_better",
            "gpu_source_id": source_id,
            "target_gpu_allocation": 100,
            "adapter_type": "ollama_benchmark",
            "research_type": "local_model_benchmark",
            "model_id": "ollama:test-model:latest",
            "model_digest": "sha256:test-digest",
            "benchmark_profile": "ollama-text-v1",
            "status": "queued",
        }
    )


def test_model_research_continues_after_calibration_until_stopped(settings) -> None:
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    first = _benchmark(database, source["id"], "First model benchmark")
    second = _benchmark(database, source["id"], "Second model benchmark")
    fake = FakeOllamaClient()

    def planner(research: dict[str, Any]) -> dict[str, Any]:
        experiment_number = len(research.get("experiments") or []) + 1
        return {
            "hypothesis": f"Agent hypothesis {experiment_number}",
            "num_ctx": 8192 + (experiment_number % 4) * 256,
            "num_batch": 2048,
            "expected_improvement": "A larger processing batch may improve throughput.",
        }

    supervisor = ResearchSupervisor(
        settings,
        database,
        TelemetryService(),
        adapter_factories={
            "ollama_benchmark": lambda context: OllamaBenchmarkAdapter(
                context, client=fake, candidate_planner=planner
            )
        },
    )
    try:
        _wait_for(
            lambda: (
                len(database.get_research(first["id"], detail=True)["experiments"])
                >= 6
            )
        )
        first_detail = database.get_research(first["id"], detail=True)
        assert first_detail["status"] == "running"
        assert database.get_research(second["id"])["status"] == "queued"
        assert len(first_detail["experiments"]) >= 6
        assert first_detail["best_value"] > first_detail["baseline_value"]

        supervisor.stop(first["id"])
        _wait_for(
            lambda: (
                len(database.get_research(second["id"], detail=True)["experiments"])
                >= 5
            )
        )
        assert database.get_research(first["id"])["status"] == "stopped"
        assert database.get_research(second["id"])["status"] == "running"
        supervisor.stop(second["id"])
        assert fake.unloads >= 2
    finally:
        supervisor.shutdown()


def test_failed_profiles_do_not_end_the_autonomous_research_loop(
    settings,
) -> None:
    class AlwaysFailingClient(FakeOllamaClient):
        def generate(self, name: str, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("synthetic Ollama failure")

    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    research = _benchmark(database, source["id"], "Failing model benchmark")
    fake = AlwaysFailingClient()

    def planner(research: dict[str, Any]) -> dict[str, Any]:
        return {
            "hypothesis": "Try another safe context and batch combination.",
            "num_ctx": 8192,
            "num_batch": 2048,
            "expected_improvement": "The research loop should continue after failures.",
        }

    supervisor = ResearchSupervisor(
        settings,
        database,
        TelemetryService(),
        adapter_factories={
            "ollama_benchmark": lambda context: OllamaBenchmarkAdapter(
                context, client=fake, candidate_planner=planner
            )
        },
    )
    try:
        _wait_for(
            lambda: (
                len(database.get_research(research["id"], detail=True)["experiments"])
                >= 10
            )
        )
        detail = database.get_research(research["id"], detail=True)
        assert detail["status"] == "running"
        assert all(item["metric_value"] is None for item in detail["experiments"])
        assert not any(
            log["event_type"] in {"research_failed", "research_completed"}
            for log in detail["logs"]
        )
        supervisor.stop(research["id"])
        assert database.get_research(research["id"])["status"] == "stopped"
    finally:
        supervisor.shutdown()


def test_completed_ollama_research_can_reenter_the_continuous_loop(settings) -> None:
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    research = _benchmark(database, source["id"], "Legacy completed benchmark")
    database.update_research(research["id"], {"status": "completed"})
    fake = FakeOllamaClient()
    supervisor = ResearchSupervisor(
        settings,
        database,
        TelemetryService(),
        adapter_factories={
            "ollama_benchmark": lambda context: OllamaBenchmarkAdapter(
                context,
                client=fake,
                candidate_planner=lambda _research: {
                    "hypothesis": "Continue beyond the legacy finite benchmark.",
                    "num_ctx": 8192,
                    "num_batch": 2048,
                    "expected_improvement": "The old completed state must be restartable.",
                },
            )
        },
    )
    try:
        supervisor.start(research["id"])
        _wait_for(
            lambda: (
                len(database.get_research(research["id"], detail=True)["experiments"])
                >= 5
            )
        )
        assert database.get_research(research["id"])["status"] == "running"
        supervisor.stop(research["id"])
    finally:
        supervisor.shutdown()


def test_stopped_benchmark_unloads_the_selected_model(settings) -> None:
    class StoppedRuntime:
        def __init__(self) -> None:
            self.stop_event = threading.Event()
            self.stop_event.set()
            self.run_event = threading.Event()

        def set_process(self, process: Any | None) -> None:
            return None

        def set_phase(self, phase: str) -> None:
            return None

        def wait_until_running(self) -> bool:
            return False

    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    research = _benchmark(database, source["id"], "Stopped model benchmark")
    fake = FakeOllamaClient()
    adapter = OllamaBenchmarkAdapter(
        AdapterContext(
            settings=settings,
            database=database,
            telemetry=TelemetryService(),
            research=research,
            gpu_source=source,
            runtime=StoppedRuntime(),
            gpu_lock=threading.Lock(),
            repository_lock=threading.Lock(),
        ),
        client=fake,
    )

    with pytest.raises(InterruptedError):
        adapter.run_iteration()

    assert fake.unloads == 0


def test_model_catalog_and_create_api_pin_the_selected_digest(
    settings, monkeypatch
) -> None:
    configured = replace(settings, execution_enabled=False)
    fake = FakeOllamaClient()
    monkeypatch.setattr("backend.main.OllamaClient", lambda: fake)
    monkeypatch.setattr(
        "backend.main.TelemetryService.sample",
        lambda self, source: {"available": False, "gpu_name": "Test GPU"},
    )
    with TestClient(create_app(configured)) as client:
        source = client.get("/api/gpu-sources").json()[0]
        catalog = client.get(f"/api/gpu-sources/{source['id']}/models")
        assert catalog.status_code == 200
        models = catalog.json()["models"]
        assert models[0]["id"] == "huggingface:Ibrahimsait/Beyefendi-v2"
        assert next(item for item in models if item["id"] == fake.model.id)

        invalid = client.post(
            "/api/researches",
            json={
                "original_prompt": "Benchmark an installed local model",
                "research_type": "local_model_benchmark",
            },
        )
        assert invalid.status_code == 422

        created = client.post(
            "/api/researches",
            json={
                "original_prompt": "Benchmark an installed local model",
                "research_type": "local_model_benchmark",
                "model_id": fake.model.id,
                "gpu_source_id": source["id"],
            },
        )
        assert created.status_code == 201
        body = created.json()
        assert body["status"] == "queued"
        assert body["adapter_type"] == "ollama_benchmark"
        assert body["metric_name"] == "output_tokens_per_second"
        assert body["model_id"] == fake.model.id
        assert body["model_digest"] == fake.model.digest


def test_create_api_pins_beyefendi_v2_and_requires_full_gpu(
    settings, monkeypatch
) -> None:
    configured = replace(settings, execution_enabled=False)
    monkeypatch.setattr(
        "backend.main.TelemetryService.sample",
        lambda self, source: {"available": False, "gpu_name": "Test GPU"},
    )
    with TestClient(create_app(configured)) as client:
        source = client.get("/api/gpu-sources").json()[0]
        payload = {
            "original_prompt": "Measure my Beyefendi-v2 model's real generation speed",
            "research_type": "local_model_benchmark",
            "model_id": "huggingface:Ibrahimsait/Beyefendi-v2",
            "benchmark_profile": "hf-transformers-text-v1",
            "gpu_source_id": source["id"],
        }

        invalid = client.post(
            "/api/researches",
            json={**payload, "target_gpu_allocation": 95},
        )
        assert invalid.status_code == 422

        created = client.post(
            "/api/researches",
            json={**payload, "target_gpu_allocation": 100},
        )
        assert created.status_code == 201
        body = created.json()
        assert body["status"] == "queued"
        assert body["adapter_type"] == "huggingface_beyefendi_benchmark"
        assert body["metric_name"] == "output_tokens_per_second"
        assert body["model_runtime"] == "huggingface"
        assert body["benchmark_profile"] == "hf-transformers-text-v1"
        assert body["target_gpu_allocation"] == 100
