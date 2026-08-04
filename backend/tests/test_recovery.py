from __future__ import annotations

import subprocess

from backend.adapters.autoresearch import KarpathyAutoresearchAdapter
from backend.db import Database


def _git(workspace, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_resume_resets_candidate_to_persisted_best_commit(settings, tmp_path) -> None:
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    research = database.create_research(
        {
            "title": "Recovery",
            "original_prompt": "recover Git state",
            "objective": "Resume only from a validated commit",
            "gpu_source_id": source["id"],
            "target_gpu_allocation": 100,
        }
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _git(workspace, "init")
    _git(workspace, "config", "user.name", "Test")
    _git(workspace, "config", "user.email", "test@example.invalid")
    for name, value in {
        "prepare.py": "FIXED = True\n",
        "program.md": "objective\n",
        "train.py": "VALUE = 'safe'\n",
    }.items():
        (workspace / name).write_text(value, encoding="utf-8")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-m", "safe baseline")
    safe_commit = _git(workspace, "rev-parse", "HEAD")
    database.update_research(research["id"], {"best_git_commit": safe_commit})

    experiment = database.create_experiment(
        research["id"], "Unsafe candidate", "Change not yet evaluated", None
    )
    (workspace / "train.py").write_text("VALUE = 'candidate'\n", encoding="utf-8")
    _git(workspace, "add", "train.py")
    _git(workspace, "commit", "-m", "experiment 1: unsafe")
    assert _git(workspace, "rev-parse", "HEAD") != safe_commit

    recovered = database.close_incomplete_experiments(
        research["id"], "Interrupted before validation"
    )
    adapter = object.__new__(KarpathyAutoresearchAdapter)
    adapter.settings = settings
    adapter.db = database
    adapter.research_id = research["id"]
    adapter.workspace = workspace
    restored = adapter._restore_persisted_best(
        database.get_research(research["id"]), recovered
    )

    assert restored == safe_commit
    assert _git(workspace, "rev-parse", "HEAD") == safe_commit
    assert (workspace / "train.py").read_text(encoding="utf-8") == "VALUE = 'safe'\n"
    assert database.get_experiment(experiment["id"])["accepted"] is False
