from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from backend.adapters.base import ResearchAdapter
from backend.db import Database
from backend.supervisor import ResearchSupervisor
from backend.telemetry import TelemetryService


class WaitingAdapter(ResearchAdapter):
    prepared = threading.Event()

    def prepare(self) -> None:
        type(self).prepared.set()

    def run_iteration(self) -> None:
        self.context.runtime.stop_event.wait(0.05)


class RepositoryLockAdapter(ResearchAdapter):
    guard = threading.Lock()
    active_prepares = 0
    max_active_prepares = 0
    prepared_count = 0
    both_prepared = threading.Event()

    def prepare(self) -> None:
        with self.context.repository_lock:
            with type(self).guard:
                type(self).active_prepares += 1
                type(self).max_active_prepares = max(
                    type(self).max_active_prepares, type(self).active_prepares
                )
            threading.Event().wait(0.05)
            with type(self).guard:
                type(self).active_prepares -= 1
                type(self).prepared_count += 1
                if type(self).prepared_count == 2:
                    type(self).both_prepared.set()

    def run_iteration(self) -> None:
        self.context.runtime.stop_event.wait(0.05)


class MpsConcurrencyAdapter(ResearchAdapter):
    guard = threading.Lock()
    barrier = threading.Barrier(2)
    active = 0
    max_active = 0
    both_running = threading.Event()

    def __init__(self, context) -> None:
        super().__init__(context)
        self.ran = False

    def prepare(self) -> None:
        pass

    def run_iteration(self) -> None:
        if self.ran:
            self.context.runtime.stop_event.wait(0.05)
            return
        self.ran = True
        with self.context.gpu_lock:
            type(self).barrier.wait(timeout=1)
            with type(self).guard:
                type(self).active += 1
                type(self).max_active = max(type(self).max_active, type(self).active)
                if type(self).active == 2:
                    type(self).both_running.set()
            threading.Event().wait(0.1)
            with type(self).guard:
                type(self).active -= 1


def test_real_supervisor_state_machine_start_pause_resume_stop(settings) -> None:
    WaitingAdapter.prepared.clear()
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    research = database.create_research(
        {
            "title": "Transitions",
            "original_prompt": "test transitions",
            "objective": "Test state transitions",
            "gpu_source_id": source["id"],
            "target_gpu_allocation": 100,
            "adapter_type": "waiting",
        }
    )
    supervisor = ResearchSupervisor(
        settings,
        database,
        TelemetryService(),
        adapter_factories={"waiting": WaitingAdapter},
    )

    assert supervisor.start(research["id"])["status"] == "running"
    assert WaitingAdapter.prepared.wait(1)
    assert supervisor.pause(research["id"])["status"] == "paused"
    snapshot = supervisor.runtime_snapshot(research["id"])
    assert snapshot["paused"] is True
    assert snapshot["phase"] == "ready"
    assert snapshot["phase_changed_at"].endswith("Z")
    assert snapshot["allocation"] == {
        "target_percent": 100,
        "mode": "exclusive_serialized",
        "target_role": "scheduling_metadata",
        "target_enforced": False,
        "is_hard_utilization_target": False,
        "utilization_source": "nvidia-smi_device_sample",
    }
    assert supervisor.resume(research["id"])["status"] == "running"
    assert supervisor.stop(research["id"])["status"] == "stopped"
    assert supervisor.runtime_snapshot(research["id"])["attached"] is False

    event_types = [
        item["event_type"] for item in database.list_logs(research["id"], limit=50)
    ]
    assert event_types == [
        "research_started",
        "research_ready",
        "research_paused",
        "research_resumed",
        "research_stopped",
    ]
    with pytest.raises(RuntimeError, match="running or paused"):
        supervisor.stop(research["id"])


def test_execution_must_be_opted_in(settings) -> None:
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    research = database.create_research(
        {
            "title": "Safe",
            "original_prompt": "do not auto start",
            "objective": "Stay stopped",
            "gpu_source_id": source["id"],
            "target_gpu_allocation": 100,
        }
    )
    supervisor = ResearchSupervisor(
        replace(settings, execution_enabled=False), database, TelemetryService()
    )
    with pytest.raises(PermissionError, match="AUTORESEARCH_ENABLE_EXECUTION"):
        supervisor.start(research["id"])
    assert database.get_research(research["id"])["status"] == "stopped"


def test_concurrent_supervisors_share_one_repository_preparation_lock(settings) -> None:
    RepositoryLockAdapter.active_prepares = 0
    RepositoryLockAdapter.max_active_prepares = 0
    RepositoryLockAdapter.prepared_count = 0
    RepositoryLockAdapter.both_prepared.clear()
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    research_ids = []
    for index in range(2):
        research = database.create_research(
            {
                "title": f"Concurrent {index}",
                "original_prompt": "test shared repo preparation",
                "objective": "Serialize shared repository mutations",
                "gpu_source_id": source["id"],
                "target_gpu_allocation": 50,
                "adapter_type": "repo_lock",
            }
        )
        research_ids.append(research["id"])
    supervisor = ResearchSupervisor(
        settings,
        database,
        TelemetryService(),
        adapter_factories={"repo_lock": RepositoryLockAdapter},
    )
    for research_id in research_ids:
        supervisor.start(research_id)
    assert RepositoryLockAdapter.both_prepared.wait(2)
    assert RepositoryLockAdapter.max_active_prepares == 1
    for research_id in research_ids:
        supervisor.stop(research_id)


def test_explicit_cuda_mps_mode_allows_concurrent_research_processes(settings) -> None:
    MpsConcurrencyAdapter.barrier = threading.Barrier(2)
    MpsConcurrencyAdapter.active = 0
    MpsConcurrencyAdapter.max_active = 0
    MpsConcurrencyAdapter.both_running.clear()
    mps_settings = replace(settings, use_cuda_mps=True)
    database = Database(mps_settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    research_ids = []
    for index in range(2):
        research = database.create_research(
            {
                "title": f"MPS {index}",
                "original_prompt": "test explicit MPS concurrency",
                "objective": "Run concurrent workers only with configured CUDA MPS",
                "gpu_source_id": source["id"],
                "target_gpu_allocation": 50,
                "adapter_type": "mps",
            }
        )
        research_ids.append(research["id"])
    supervisor = ResearchSupervisor(
        mps_settings,
        database,
        TelemetryService(),
        adapter_factories={"mps": MpsConcurrencyAdapter},
    )
    for research_id in research_ids:
        supervisor.start(research_id)
    assert MpsConcurrencyAdapter.both_running.wait(2)
    assert MpsConcurrencyAdapter.max_active == 2
    snapshot = supervisor.runtime_snapshot(research_ids[0])
    assert snapshot["allocation"]["mode"] == "cuda_mps_active_thread_percentage"
    assert snapshot["allocation"]["target_enforced"] is True
    assert snapshot["allocation"]["is_hard_utilization_target"] is False
    for research_id in research_ids:
        supervisor.stop(research_id)
