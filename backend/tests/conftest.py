from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from backend.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    data = tmp_path / "data"
    configured = replace(
        Settings.from_env(),
        data_dir=data,
        database_path=data / "test.sqlite3",
        workspace_dir=data / "workspaces",
        log_dir=data / "logs",
        upstream_cache_dir=data / "upstream",
        runtime_dir=data / "runtimes",
        execution_enabled=True,
        wsl_distro=None,
    )
    configured.ensure_directories()
    return configured
