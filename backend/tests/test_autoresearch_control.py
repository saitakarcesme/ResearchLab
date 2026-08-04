from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest

from backend.adapters.autoresearch import KarpathyAutoresearchAdapter
from backend.adapters.base import AdapterContext, ResearchFailed
from backend.db import Database
from backend.process_control import ProcessResult
from backend.supervisor import ResearchRuntime
from backend.telemetry import TelemetryService


def test_evaluation_control_error_survives_cleanup_and_persistence_failures(
    settings, monkeypatch
) -> None:
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    research = database.create_research(
        {
            "title": "Fail-safe evaluation",
            "original_prompt": "Verify process-control failure handling",
            "objective": "Keep the GPU reserved after an unconfirmed termination.",
            "gpu_source_id": source["id"],
            "target_gpu_allocation": 100,
            "adapter_type": "karpathy_autoresearch",
            "status": "stopped",
        }
    )
    runtime = ResearchRuntime(research["id"])
    gpu_lock = threading.Lock()
    adapter = KarpathyAutoresearchAdapter(
        AdapterContext(
            settings=settings,
            database=database,
            telemetry=TelemetryService(),
            research=research,
            gpu_source=source,
            runtime=runtime,
            gpu_lock=gpu_lock,
            repository_lock=threading.Lock(),
        )
    )
    output = settings.log_dir / research["id"] / "evaluation.log"
    output.write_text("synthetic evaluation", encoding="utf-8")
    monkeypatch.setattr(
        adapter,
        "_git",
        lambda *args, **kwargs: SimpleNamespace(stdout="base-sha\n"),
    )
    monkeypatch.setattr(adapter, "_assert_prepare_unchanged", lambda: None)
    monkeypatch.setattr(adapter, "_capture_evaluation_state", lambda: ())
    monkeypatch.setattr(
        adapter,
        "_run_evaluation",
        lambda experiment_number, experiment_id: (
            ProcessResult(
                returncode=1,
                timed_out=False,
                stopped=False,
                duration_seconds=1.0,
                active_duration_seconds=1.0,
                control_error="process group still exists",
            ),
            output,
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_clean_candidate",
        lambda base_sha: (_ for _ in ()).throw(RuntimeError("cleanup unavailable")),
    )
    def fail_finish(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(database, "finish_experiment", fail_finish)

    with pytest.raises(ResearchFailed) as failure:
        adapter.run_iteration()

    message = str(failure.value)
    assert "process group still exists" in message
    assert "candidate cleanup failed: cleanup unavailable" in message
    assert "result persistence failed: database unavailable" in message
    assert runtime.control_error == "process group still exists"
    assert gpu_lock.acquire(blocking=False)
    gpu_lock.release()
