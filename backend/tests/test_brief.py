from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from backend.brief import CodexResearchBriefGenerator, validate_brief_payload


def test_brief_generation_uses_fast_fallback_when_execution_disabled(settings) -> None:
    generator = CodexResearchBriefGenerator(replace(settings, execution_enabled=False))
    brief = generator.generate("Improve a coding harness on one RTX 3090")
    assert brief.title == "Improve A Coding Harness On One"
    assert "val_bpb" in brief.objective
    assert brief.used_ai is False
    assert "fallback" in brief.warning


def test_structured_brief_validation_is_strict() -> None:
    title, objective = validate_brief_payload(
        {
            "title": "Coding Harness",
            "objective": "Lower val_bpb is better under the fixed five-minute train.py evaluation.",
        }
    )
    assert title == "Coding Harness"
    assert "val_bpb" in objective
    with pytest.raises(ValueError, match="exactly"):
        validate_brief_payload({"title": "X", "objective": "val_bpb", "extra": True})
    with pytest.raises(ValueError, match="val_bpb"):
        validate_brief_payload(
            {
                "title": "Coding Harness",
                "objective": "Improve a metric without naming it explicitly.",
            }
        )


def test_enabled_brief_generation_invokes_read_only_schema_constrained_codex(
    settings, monkeypatch
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["input"] = kwargs["input"]
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text(
            json.dumps(
                {
                    "title": "Coding Harness",
                    "objective": "Make val_bpb lower under the fixed five-minute train.py evaluation for the requested coding harness.",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("backend.brief.subprocess.run", fake_run)
    generator = CodexResearchBriefGenerator(replace(settings, execution_enabled=True))
    brief = generator.generate("Build the best coding harness possible")
    assert brief.used_ai is True
    assert brief.title == "Coding Harness"
    command = observed["command"]
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--output-schema" in command
    assert b"Build the best coding harness possible" in observed["input"]
