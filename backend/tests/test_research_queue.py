from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import replace
from typing import ClassVar

from fastapi.testclient import TestClient

from backend.adapters.base import ResearchAdapter
from backend.db import Database
from backend.main import create_app
from backend.supervisor import ResearchSupervisor
from backend.telemetry import TelemetryService


class QueueWaitingAdapter(ResearchAdapter):
    lock = threading.Lock()
    started: ClassVar[list[str]] = []

    def prepare(self) -> None:
        with type(self).lock:
            type(self).started.append(self.context.research["id"])

    def run_iteration(self) -> None:
        self.context.runtime.stop_event.wait(0.02)


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for queue scheduler")


def _queued_research(database: Database, source_id: str, title: str, allocation: int):
    return database.create_research(
        {
            "title": title,
            "original_prompt": f"Queue {title}",
            "objective": "Exercise the persistent GPU queue.",
            "gpu_source_id": source_id,
            "target_gpu_allocation": allocation,
            "adapter_type": "queue_waiting",
            "status": "queued",
        }
    )


def test_legacy_database_migrates_queue_status_without_losing_future_columns(
    settings,
) -> None:
    with sqlite3.connect(settings.database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE gpu_sources (
                id TEXT PRIMARY KEY, name TEXT NOT NULL,
                type TEXT NOT NULL CHECK (type IN ('local', 'remote')),
                host TEXT, port INTEGER NOT NULL DEFAULT 22,
                username TEXT, auth_method TEXT NOT NULL DEFAULT 'agent',
                workspace_path TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE researches (
                id TEXT PRIMARY KEY, title TEXT NOT NULL,
                original_prompt TEXT NOT NULL, objective TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('running', 'paused', 'completed', 'stopped', 'failed')),
                metric_name TEXT NOT NULL,
                metric_direction TEXT NOT NULL CHECK (metric_direction IN ('lower_is_better', 'higher_is_better')),
                baseline_value REAL, best_value REAL, best_git_commit TEXT,
                gpu_source_id TEXT NOT NULL REFERENCES gpu_sources(id),
                target_gpu_allocation INTEGER NOT NULL,
                workspace_path TEXT, adapter_type TEXT NOT NULL,
                future_profile TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            INSERT INTO gpu_sources VALUES
                ('local', 'Test GPU', 'local', NULL, 22, NULL, 'agent', NULL, 't0', 't0');
            INSERT INTO researches VALUES
                ('legacy', 'Legacy', 'prompt', 'objective', 'stopped', 'val_bpb',
                 'lower_is_better', NULL, NULL, NULL, 'local', 100, NULL,
                 'karpathy_autoresearch', 'keep-me', 't0', 't0');
            """
        )

    database = Database(settings.database_path)
    database.initialize()
    assert database.transition_research("legacy", ["stopped"], "queued")

    reopened = Database(settings.database_path)
    reopened.initialize()
    research = reopened.get_research("legacy")
    assert research["status"] == "queued"
    assert research["queued_at"] is not None
    assert research["queue_order"] == 1
    with reopened.connect() as connection:
        assert connection.execute(
            "SELECT future_profile FROM researches WHERE id = 'legacy'"
        ).fetchone()[0] == "keep-me"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_exclusive_scheduler_runs_one_queued_research_per_gpu(settings) -> None:
    QueueWaitingAdapter.started = []
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    first = _queued_research(database, source["id"], "First", 60)
    second = _queued_research(database, source["id"], "Second", 40)
    assert [item["id"] for item in database.list_queued_researches()] == [
        first["id"],
        second["id"],
    ]

    supervisor = ResearchSupervisor(
        settings,
        database,
        TelemetryService(),
        adapter_factories={"queue_waiting": QueueWaitingAdapter},
    )
    try:
        _wait_for(lambda: len(QueueWaitingAdapter.started) == 1)
        assert QueueWaitingAdapter.started == [first["id"]]
        assert database.get_research(first["id"])["status"] == "running"
        assert database.get_research(second["id"])["status"] == "queued"

        supervisor.stop(first["id"])
        _wait_for(lambda: len(QueueWaitingAdapter.started) == 2)
        assert QueueWaitingAdapter.started == [first["id"], second["id"]]
        assert database.get_research(second["id"])["status"] == "running"
        supervisor.stop(second["id"])
    finally:
        supervisor.shutdown()


def test_mps_scheduler_respects_sum_of_target_allocations(settings) -> None:
    QueueWaitingAdapter.started = []
    configured = replace(settings, use_cuda_mps=True)
    database = Database(configured.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    first = _queued_research(database, source["id"], "Sixty", 60)
    second = _queued_research(database, source["id"], "Forty", 40)
    third = _queued_research(database, source["id"], "Twenty", 20)

    supervisor = ResearchSupervisor(
        configured,
        database,
        TelemetryService(),
        adapter_factories={"queue_waiting": QueueWaitingAdapter},
    )
    try:
        _wait_for(lambda: len(QueueWaitingAdapter.started) == 2)
        assert QueueWaitingAdapter.started == [first["id"], second["id"]]
        assert database.get_research(third["id"])["status"] == "queued"

        supervisor.stop(first["id"])
        _wait_for(lambda: len(QueueWaitingAdapter.started) == 3)
        assert QueueWaitingAdapter.started[-1] == third["id"]
        supervisor.stop(second["id"])
        supervisor.stop(third["id"])
    finally:
        supervisor.shutdown()


def test_paused_research_reserves_gpu_across_supervisor_restart(settings) -> None:
    QueueWaitingAdapter.started = []
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    paused = _queued_research(database, source["id"], "Paused", 100)
    assert database.transition_research(paused["id"], ["queued"], "paused")
    for index in range(101):
        filler = _queued_research(database, source["id"], f"Finished {index}", 100)
        assert database.transition_research(filler["id"], ["queued"], "stopped")
    waiting = _queued_research(database, source["id"], "Waiting", 100)

    supervisor = ResearchSupervisor(
        settings,
        database,
        TelemetryService(),
        adapter_factories={"queue_waiting": QueueWaitingAdapter},
    )
    try:
        assert supervisor.start_next() is None
        time.sleep(0.05)
        assert QueueWaitingAdapter.started == []
        assert database.get_research(waiting["id"])["status"] == "queued"
    finally:
        supervisor.shutdown()


def test_research_api_queues_by_default_and_supports_queue_controls(
    settings, monkeypatch
) -> None:
    configured = replace(settings, execution_enabled=False)
    monkeypatch.setattr(
        "backend.main.TelemetryService.sample",
        lambda self, source: {"available": False, "gpu_name": "Test GPU"},
    )
    payload = {
        "original_prompt": "Queue this measurable research",
        "title": "Queued API research",
        "objective": "Measure queue API behavior without starting execution.",
    }
    with TestClient(create_app(configured)) as client:
        created = client.post("/api/researches", json=payload)
        assert created.status_code == 201
        research_id = created.json()["id"]
        assert created.json()["status"] == "queued"
        assert created.json()["queue_position"] == 1

        listing = client.get("/api/researches/queue")
        assert listing.status_code == 200
        assert listing.json()[0]["id"] == research_id
        assert listing.json()[0]["queue_position"] == 1

        dequeued = client.post(f"/api/researches/{research_id}/dequeue")
        assert dequeued.status_code == 200
        assert dequeued.json()["status"] == "stopped"
        enqueued = client.post(f"/api/researches/{research_id}/enqueue")
        assert enqueued.status_code == 200
        assert enqueued.json()["status"] == "queued"
        assert client.patch(
            f"/api/researches/{research_id}", json={"target_gpu_allocation": 50}
        ).status_code == 409
        assert client.post("/api/researches/queue/start-next").status_code == 409

        immediate = client.post(
            "/api/researches", json={**payload, "title": "Immediate", "auto_start": True}
        )
        assert immediate.status_code == 409

    database = Database(configured.database_path)
    statuses = {item["title"]: item["status"] for item in database.list_researches()}
    assert statuses["Queued API research"] == "queued"
    assert statuses["Immediate"] == "stopped"
