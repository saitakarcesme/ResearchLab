from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.adapters.autoresearch import (
    AgentGenerationTimeout,
    KarpathyAutoresearchAdapter,
    agent_retry_delay_seconds,
)
from backend.config import Settings


class RecordingDatabase:
    def __init__(self) -> None:
        self.logs: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def add_log(self, *args, **kwargs) -> None:
        self.logs.append((args, kwargs))


class RecordingEvent:
    def __init__(self) -> None:
        self.waits: list[float] = []

    def wait(self, timeout: float) -> bool:
        self.waits.append(timeout)
        return False


def test_agent_retry_backoff_is_exponential_bounded_and_unlimited() -> None:
    assert [agent_retry_delay_seconds(attempt) for attempt in range(1, 9)] == [
        5,
        10,
        20,
        40,
        80,
        160,
        300,
        300,
    ]
    assert agent_retry_delay_seconds(100) == 300
    with pytest.raises(ValueError, match="positive"):
        agent_retry_delay_seconds(0)


def test_candidate_command_overrides_personal_reasoning_effort(tmp_path: Path) -> None:
    adapter = object.__new__(KarpathyAutoresearchAdapter)
    adapter.settings = SimpleNamespace(
        codex_binary="codex", agent_reasoning_effort="high"
    )
    command = adapter._agent_command(tmp_path / "scratch", tmp_path / "result.json")
    override = command[command.index("--config") + 1]
    assert override == 'model_reasoning_effort="high"'
    assert command[:2] == ["codex", "exec"]


def test_agent_timeout_records_attempt_and_waits_stop_aware() -> None:
    adapter = object.__new__(KarpathyAutoresearchAdapter)
    database = RecordingDatabase()
    stop_event = RecordingEvent()
    adapter.db = database
    adapter.context = SimpleNamespace(runtime=SimpleNamespace(stop_event=stop_event))
    adapter.settings = SimpleNamespace(
        agent_timeout_seconds=600, agent_reasoning_effort="high"
    )
    adapter.research_id = "research-id"
    adapter._agent_failure_streak = 2

    delay = adapter._record_agent_failure(
        AgentGenerationTimeout("timed out after 600 active seconds"),
        experiment_number=16,
        attempt_number=3,
    )

    assert delay == 20
    assert stop_event.waits == [20]
    args, kwargs = database.logs[-1]
    assert args[:2] == ("research-id", "agent_error")
    assert "attempt 4" in args[2]
    assert kwargs["data"] == {
        "experiment_number": 16,
        "attempt_number": 3,
        "next_attempt_number": 4,
        "consecutive_failures": 3,
        "failure_kind": "timeout",
        "retry_delay_seconds": 20,
        "agent_timeout_seconds": 600,
        "reasoning_effort": "high",
        "diagnostic_log": "agent-16-attempt-3.log",
    }


def test_agent_settings_have_service_owned_defaults(monkeypatch) -> None:
    monkeypatch.delenv("AUTORESEARCH_AGENT_TIMEOUT", raising=False)
    monkeypatch.delenv("AUTORESEARCH_AGENT_REASONING_EFFORT", raising=False)
    settings = Settings.from_env()
    assert settings.agent_timeout_seconds == 600
    assert settings.agent_reasoning_effort == "high"

    monkeypatch.setenv("AUTORESEARCH_AGENT_REASONING_EFFORT", "impossible")
    with pytest.raises(ValueError, match="AUTORESEARCH_AGENT_REASONING_EFFORT"):
        Settings.from_env()
