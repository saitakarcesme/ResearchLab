from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest
from fastapi.testclient import TestClient

from backend.adapters.base import ResearchAdapter
from backend.db import Database
from backend.main import create_app
from backend.supervisor import ResearchSupervisor
from backend.telemetry import TelemetryService


def _create_directory_link(link: Path, target: Path) -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            pytest.skip(
                "Directory junctions are unavailable: "
                + (result.stderr or result.stdout).strip()
            )
        return
    os.symlink(target, link, target_is_directory=True)


class AttachedProcess:
    def __init__(self) -> None:
        self.last_control_error: str | None = None
        self.paused = False
        self.terminated = threading.Event()

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def terminate(self) -> None:
        self.terminated.set()


class AttachedProcessAdapter(ResearchAdapter):
    prepared = threading.Event()
    process: AttachedProcess | None = None

    def prepare(self) -> None:
        process = AttachedProcess()
        type(self).process = process
        self.context.runtime.set_process(process)  # type: ignore[arg-type]
        type(self).prepared.set()

    def run_iteration(self) -> None:
        self.context.runtime.stop_event.wait(0.05)


class UnconfirmedSetupAdapter(ResearchAdapter):
    def prepare(self) -> None:
        self.context.runtime.control_error = "setup termination was not confirmed"
        raise RuntimeError("synthetic setup timeout")

    def run_iteration(self) -> None:
        raise AssertionError("setup failure must prevent an iteration")


def _create_research(
    database: Database, source_id: str, *, status: str = "stopped"
) -> dict:
    return database.create_research(
        {
            "title": "Disposable research",
            "original_prompt": "Measure a disposable research",
            "objective": "Verify complete deletion.",
            "gpu_source_id": source_id,
            "target_gpu_allocation": 100,
            "adapter_type": "attached",
            "status": status,
        }
    )


def test_delete_endpoint_cascades_records_and_only_removes_managed_workspace(
    settings, monkeypatch, tmp_path
) -> None:
    configured = replace(settings, execution_enabled=False)
    monkeypatch.setattr(
        "backend.main.TelemetryService.sample",
        lambda self, source: {"available": False, "gpu_name": "Test GPU"},
    )

    with TestClient(create_app(configured)) as client:
        database = client.app.state.database
        source = database.list_gpu_sources()[0]
        research = _create_research(database, source["id"], status="queued")
        research_id = research["id"]
        workspace = configured.workspace_dir / research_id
        workspace.mkdir(parents=True)
        (workspace / "artifact.txt").write_text("managed", encoding="utf-8")
        log_dir = configured.log_dir / research_id
        log_dir.mkdir(parents=True)
        (log_dir / "raw.log").write_text("managed log", encoding="utf-8")
        direct_runtime = configured.runtime_dir / research_id
        direct_runtime.mkdir(parents=True)
        (direct_runtime / "sample.json").write_text("{}", encoding="utf-8")
        nested_runtime = configured.runtime_dir / "gpu-source" / research_id
        nested_runtime.mkdir(parents=True)
        (nested_runtime / "cache.txt").write_text("managed", encoding="utf-8")
        sandbox = configured.data_dir / "agent-sandboxes" / research_id
        sandbox.mkdir(parents=True)
        (sandbox / "candidate.txt").write_text("managed", encoding="utf-8")
        article_log_root = configured.log_dir / "articles"
        article_log_root.mkdir(parents=True)
        article_log = article_log_root / f"{research_id}-generation.log"
        article_log.write_text("managed", encoding="utf-8")
        unrelated_article_log = article_log_root / "unrelated.log"
        unrelated_article_log.write_text("keep", encoding="utf-8")
        database.update_research(research_id, {"workspace_path": str(workspace)})

        experiment = database.create_experiment(
            research_id, "Try one change", "Changed one setting", None
        )
        database.add_log(
            research_id,
            "experiment_started",
            "Started",
            experiment_id=experiment["id"],
        )
        article = database.upsert_article(
            research_id, "Disposable article", "Temporary result"
        )

        notifications: list[bool] = []
        monkeypatch.setattr(
            client.app.state.supervisor,
            "notify_queue",
            lambda: notifications.append(True),
        )
        response = client.delete(f"/api/researches/{research_id}")

        assert response.status_code == 204
        assert notifications == [True]
        assert database.get_research(research_id) is None
        assert database.get_experiment(experiment["id"]) is None
        assert database.list_logs(research_id) == []
        assert database.get_article(article["id"]) is None
        assert not workspace.exists()
        assert not log_dir.exists()
        assert not direct_runtime.exists()
        assert not nested_runtime.exists()
        assert not sandbox.exists()
        assert not article_log.exists()
        assert unrelated_article_log.read_text(encoding="utf-8") == "keep"
        assert client.delete(f"/api/researches/{research_id}").status_code == 404

        unsafe = tmp_path / "outside-workspaces"
        unsafe.mkdir()
        (unsafe / "keep.txt").write_text("keep", encoding="utf-8")
        other = _create_research(database, source["id"])
        database.update_research(other["id"], {"workspace_path": str(unsafe)})

        assert client.delete(f"/api/researches/{other['id']}").status_code == 204
        assert (unsafe / "keep.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("pause_first", [False, True])
def test_delete_stops_attached_process_before_removing_active_research(
    settings, pause_first
) -> None:
    AttachedProcessAdapter.prepared.clear()
    AttachedProcessAdapter.process = None
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    research = _create_research(database, source["id"])
    workspace = settings.workspace_dir / research["id"]
    workspace.mkdir(parents=True)
    database.update_research(research["id"], {"workspace_path": str(workspace)})
    supervisor = ResearchSupervisor(
        settings,
        database,
        TelemetryService(),
        adapter_factories={"attached": AttachedProcessAdapter},
    )

    try:
        assert supervisor.start(research["id"])["status"] == "running"
        assert AttachedProcessAdapter.prepared.wait(1)
        process = AttachedProcessAdapter.process
        assert process is not None
        if pause_first:
            assert supervisor.pause(research["id"])["status"] == "paused"
            assert process.paused is True

        supervisor.delete(research["id"])

        assert process.terminated.wait(1)
        assert database.get_research(research["id"]) is None
        assert supervisor.runtime_snapshot(research["id"])["attached"] is False
        assert not workspace.exists()
    finally:
        supervisor.shutdown()


def test_delete_keeps_record_when_process_termination_cannot_be_confirmed(
    settings,
) -> None:
    AttachedProcessAdapter.prepared.clear()
    AttachedProcessAdapter.process = None
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    research = _create_research(database, source["id"])
    workspace = settings.workspace_dir / research["id"]
    workspace.mkdir(parents=True)
    database.update_research(research["id"], {"workspace_path": str(workspace)})
    supervisor = ResearchSupervisor(
        settings,
        database,
        TelemetryService(),
        adapter_factories={"attached": AttachedProcessAdapter},
    )

    try:
        supervisor.start(research["id"])
        assert AttachedProcessAdapter.prepared.wait(1)
        process = AttachedProcessAdapter.process
        assert process is not None
        process.last_control_error = "termination was not confirmed"

        with pytest.raises(RuntimeError, match="termination was not confirmed"):
            supervisor.delete(research["id"])

        retained = database.get_research(research["id"])
        assert retained is not None
        assert retained["status"] == "failed"
        assert workspace.exists()
    finally:
        supervisor.shutdown()


def test_delete_keeps_record_when_managed_workspace_cannot_be_removed(
    settings, monkeypatch
) -> None:
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    research = _create_research(database, source["id"], status="queued")
    workspace = settings.workspace_dir / research["id"]
    workspace.mkdir(parents=True)
    database.update_research(research["id"], {"workspace_path": str(workspace)})
    supervisor = ResearchSupervisor(settings, database, TelemetryService())

    monkeypatch.setattr(
        ResearchSupervisor,
        "_remove_managed_directory",
        staticmethod(lambda root, name, candidate: False),
    )
    try:
        with pytest.raises(RuntimeError, match="workspace could not be removed"):
            supervisor.delete(research["id"])

        assert database.get_research(research["id"]) is not None
        assert workspace.exists()
    finally:
        supervisor.shutdown()


def test_unconfirmed_setup_termination_keeps_gpu_reserved(settings) -> None:
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    research = _create_research(database, source["id"])
    supervisor = ResearchSupervisor(
        settings,
        database,
        TelemetryService(),
        adapter_factories={"attached": UnconfirmedSetupAdapter},
    )

    try:
        supervisor.start(research["id"])
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            snapshot = database.get_research(research["id"], detail=True)
            if snapshot["status"] == "paused" and any(
                log["event_type"] == "research_control_error"
                for log in snapshot["logs"]
            ):
                break
            time.sleep(0.01)

        retained = database.get_research(research["id"], detail=True)
        assert retained is not None
        assert retained["status"] == "paused"
        assert any(
            log["event_type"] == "research_control_error"
            for log in retained["logs"]
        )
        queued = _create_research(database, source["id"], status="queued")
        assert supervisor.start_next() is None
        assert database.get_research(queued["id"])["status"] == "queued"
    finally:
        supervisor.shutdown()


def test_failed_startup_cleanup_is_conservatively_reserved(
    settings, monkeypatch
) -> None:
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    research = _create_research(database, source["id"])
    database.update_research(research["id"], {"status": "failed"})
    failed = database.get_research(research["id"])
    assert failed is not None

    monkeypatch.setattr(
        ResearchSupervisor,
        "_terminate_research_process_groups",
        classmethod(
            lambda cls, settings, source, research: (_ for _ in ()).throw(
                RuntimeError("cleanup unavailable")
            )
        ),
    )
    ResearchSupervisor.terminate_stale_process_groups(
        settings, database, [failed]
    )

    retained = database.get_research(research["id"], detail=True)
    assert retained is not None
    assert retained["status"] == "paused"
    assert any(
        log["event_type"] == "stale_process_warning"
        for log in retained["logs"]
    )


def test_cleanup_candidates_exclude_ordinary_failed_researches(settings) -> None:
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    ordinary_failed = _create_research(database, source["id"])
    uncertain_failed = _create_research(database, source["id"])
    running = _create_research(database, source["id"])
    paused = _create_research(database, source["id"])
    database.update_research(ordinary_failed["id"], {"status": "failed"})
    database.update_research(uncertain_failed["id"], {"status": "failed"})
    database.update_research(running["id"], {"status": "running"})
    database.update_research(paused["id"], {"status": "paused"})
    database.add_log(
        uncertain_failed["id"],
        "research_control_error",
        "Termination was not confirmed.",
        level="error",
    )

    candidate_ids = {
        item["id"] for item in database.list_process_cleanup_candidates()
    }

    assert candidate_ids == {
        uncertain_failed["id"],
        running["id"],
        paused["id"],
    }
    assert ordinary_failed["id"] not in candidate_ids


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (Path(r"C:\managed\research"), r"\\?\C:\managed\research"),
        (
            Path(r"\\server\share\managed\research"),
            r"\\?\UNC\server\share\managed\research",
        ),
        (Path(r"\\?\C:\managed\research"), r"\\?\C:\managed\research"),
    ],
)
def test_windows_extended_path_handles_drive_unc_and_existing_prefix(
    path: Path, expected: str
) -> None:
    assert ResearchSupervisor._windows_extended_path(path) == expected


def test_posix_artifact_cleanup_is_bounded_to_one_exact_child() -> None:
    root = PurePosixPath("/srv/researchlab/runtimes/source")
    script = ResearchSupervisor._posix_remove_child_script(root, "research-id")

    assert "python3 - /srv/researchlab/runtimes/source research-id" in script
    assert "os.O_DIRECTORY | os.O_NOFOLLOW" in script
    assert "src_dir_fd=root_fd" in script
    assert "shutil.rmtree(quarantine, dir_fd=root_fd)" in script

    for unsafe_root, unsafe_name in (
        (PurePosixPath("/"), "research-id"),
        (PurePosixPath("relative"), "research-id"),
        (root, ""),
        (root, "."),
        (root, ".."),
        (root, "nested/research-id"),
    ):
        with pytest.raises(ValueError, match="absolute and bounded"):
            ResearchSupervisor._posix_remove_child_script(
                unsafe_root, unsafe_name
            )


def test_managed_delete_does_not_follow_nested_directory_symlink(tmp_path) -> None:
    root = tmp_path / "managed"
    candidate = root / "research-id"
    outside = tmp_path / "outside"
    candidate.mkdir(parents=True)
    outside.mkdir()
    protected = outside / "keep.txt"
    protected.write_text("keep", encoding="utf-8")
    try:
        _create_directory_link(candidate / "outside-link", outside)
    except OSError as exc:
        pytest.skip(f"Directory links are unavailable: {exc}")

    assert ResearchSupervisor._remove_managed_directory(
        root, "research-id", candidate
    )
    assert not candidate.exists()
    assert protected.read_text(encoding="utf-8") == "keep"


def test_managed_delete_rejects_a_symlinked_root(tmp_path) -> None:
    outside_root = tmp_path / "outside-root"
    outside_candidate = outside_root / "research-id"
    outside_candidate.mkdir(parents=True)
    protected = outside_candidate / "keep.txt"
    protected.write_text("keep", encoding="utf-8")
    linked_root = tmp_path / "managed-link"
    try:
        _create_directory_link(linked_root, outside_root)
    except OSError as exc:
        pytest.skip(f"Directory links are unavailable: {exc}")

    assert not ResearchSupervisor._remove_managed_directory(
        linked_root, "research-id", linked_root / "research-id"
    )
    assert protected.read_text(encoding="utf-8") == "keep"


def test_article_log_cleanup_rejects_a_matching_reparse_entry(tmp_path) -> None:
    article_root = tmp_path / "articles"
    outside = tmp_path / "outside-article-logs"
    article_root.mkdir()
    outside.mkdir()
    protected = outside / "keep.txt"
    protected.write_text("keep", encoding="utf-8")
    try:
        _create_directory_link(article_root / "research-id-linked", outside)
    except OSError as exc:
        pytest.skip(f"Directory links are unavailable: {exc}")

    assert not ResearchSupervisor._remove_managed_files_by_prefix(
        article_root, "research-id-"
    )
    assert protected.read_text(encoding="utf-8") == "keep"


def test_article_log_cleanup_rejects_a_reparse_root(tmp_path) -> None:
    real_root = tmp_path / "real-article-logs"
    real_root.mkdir()
    protected = real_root / "research-id-generation.log"
    protected.write_text("keep", encoding="utf-8")
    linked_root = tmp_path / "linked-article-logs"
    try:
        _create_directory_link(linked_root, real_root)
    except OSError as exc:
        pytest.skip(f"Directory links are unavailable: {exc}")

    assert not ResearchSupervisor._remove_managed_files_by_prefix(
        linked_root, "research-id-"
    )
    assert protected.read_text(encoding="utf-8") == "keep"
