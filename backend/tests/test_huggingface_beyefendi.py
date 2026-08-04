from __future__ import annotations

import json
import threading
from dataclasses import replace
from typing import Any

import pytest

from backend.adapters.base import AdapterContext, ResearchComplete
from backend.adapters.huggingface_beyefendi import (
    ADAPTER_CONFIG_SHA256,
    ADAPTER_FILE_SHA256,
    ADAPTER_REVISION,
    BASE_MODEL,
    BASE_REVISION,
    BENCHMARK_PROFILE,
    MODEL_ID,
    HuggingFaceBeyefendiBenchmarkAdapter,
    parse_benchmark_output,
    validate_benchmark_payload,
)
from backend.db import Database
from backend.runners.beyefendi_v2_benchmark import (
    PROFILES,
    RESULT_PREFIX,
    _render_prompt,
)
from backend.telemetry import TelemetryService


class Runtime:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.run_event = threading.Event()
        self.run_event.set()
        self.phases: list[str] = []

    def set_process(self, process: Any | None) -> None:
        return None

    def set_phase(self, phase: str) -> None:
        self.phases.append(phase)

    def wait_until_running(self) -> bool:
        return not self.stop_event.is_set()


def _payload(run_id: str, *, failed_profile: int | None = None) -> dict[str, Any]:
    profiles = []
    for index, profile in enumerate(PROFILES):
        if index == failed_profile:
            profiles.append(
                {
                    "id": profile.id,
                    "label": profile.label,
                    "target_input_tokens": profile.target_input_tokens,
                    "batch_size": profile.batch_size,
                    "output_tokens_per_request": profile.output_tokens_per_request,
                    "error": "synthetic profile failure",
                    "repeats": [],
                }
            )
            continue
        total_output_tokens = (
            profile.batch_size * profile.output_tokens_per_request
        )
        repeats = []
        for repeat in range(3):
            speed = 40.0 + index + repeat
            elapsed = total_output_tokens / speed
            repeats.append(
                {
                    "elapsed_seconds": elapsed,
                    "output_tokens_per_second": speed,
                    "request_latency_seconds": elapsed,
                    "input_tokens_per_request": profile.target_input_tokens,
                    "output_tokens_per_request": profile.output_tokens_per_request,
                    "total_output_tokens": total_output_tokens,
                }
            )
        profiles.append(
            {
                "id": profile.id,
                "label": profile.label,
                "target_input_tokens": profile.target_input_tokens,
                "batch_size": profile.batch_size,
                "output_tokens_per_request": profile.output_tokens_per_request,
                "error": None,
                "warmup": {"output_tokens_per_second": 10.0},
                "repeats": repeats,
                "median_output_tokens_per_second": 41.0 + index,
                "median_request_latency_seconds": total_output_tokens
                / (41.0 + index),
            }
        )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "model": {
            "id": MODEL_ID,
            "adapter_repository": "Ibrahimsait/Beyefendi-v2",
            "adapter_revision": ADAPTER_REVISION,
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
        },
        "runtime": {
            "scheduler_share_percent": 100,
            "quantization": "nf4",
            "compute_dtype": "bfloat16",
            "double_quantization": True,
            "device_map": {"": 0},
            "warmup_runs_per_profile": 1,
            "measured_repeats_per_profile": 3,
        },
        "gpu": {
            "fully_gpu_resident": True,
            "adapter_file_sha256": ADAPTER_FILE_SHA256,
            "adapter_config_sha256": ADAPTER_CONFIG_SHA256,
            "load_seconds": 12.5,
        },
        "profiles": profiles,
    }


def _research(database: Database, source_id: str) -> dict[str, Any]:
    return database.create_research(
        {
            "title": "Beyefendi-v2 speed research",
            "original_prompt": "Measure my pinned Beyefendi-v2 model at full GPU share",
            "objective": "Find the fastest understandable usage profile.",
            "metric_name": "output_tokens_per_second",
            "metric_direction": "higher_is_better",
            "gpu_source_id": source_id,
            "target_gpu_allocation": 100,
            "adapter_type": "huggingface_beyefendi_benchmark",
            "research_type": "local_model_benchmark",
            "model_id": MODEL_ID,
            "model_digest": ADAPTER_REVISION,
            "model_runtime": "huggingface",
            "benchmark_profile": BENCHMARK_PROFILE,
            "status": "queued",
        }
    )


def test_parse_runner_result_requires_pins_and_terminal_json() -> None:
    run_id = "a" * 32
    payload = _payload(run_id)
    raw = (
        "loader diagnostic\n"
        + RESULT_PREFIX
        + json.dumps(payload)
        + "\nlate buffered loader progress\n"
    )
    assert parse_benchmark_output(raw, expected_run_id=run_id) == payload

    payload["model"]["adapter_revision"] = "moving-main"
    with pytest.raises(ValueError, match="pins"):
        validate_benchmark_payload(payload, expected_run_id=run_id)

    payload = _payload(run_id)
    payload["gpu"]["adapter_file_sha256"] = "wrong-weights"
    with pytest.raises(ValueError, match="SHA-256"):
        validate_benchmark_payload(payload, expected_run_id=run_id)

    payload = _payload(run_id)
    payload["profiles"][0]["median_output_tokens_per_second"] = 999_999.0
    with pytest.raises(ValueError, match="inconsistent median speed"):
        validate_benchmark_payload(payload, expected_run_id=run_id)


def test_render_prompt_accepts_transformers_batch_encoding() -> None:
    class FakeTensor:
        ndim = 2

        def __init__(self, tokens: list[int]) -> None:
            self.tokens = tokens

        @property
        def shape(self) -> tuple[int, int]:
            return (1, len(self.tokens))

        def __getitem__(self, key):
            if key == 0:
                return type(
                    "FakeRow",
                    (),
                    {"tolist": lambda row: list(self.tokens)},
                )()
            rows, columns = key
            assert rows == slice(None)
            return FakeTensor(self.tokens[columns])

        def to(self, device: str) -> FakeTensor:
            assert device == "cuda"
            return self

    prefix = [1, 2, 3]
    suffix = [97, 98, 99]
    full = FakeTensor(prefix + list(range(10, 84)) + suffix)
    empty = FakeTensor(prefix + suffix)

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[-1]["role"] == "user"
            assert kwargs["return_tensors"] == "pt"
            return {
                "input_ids": full if messages[-1]["content"] else empty
            }

    class FakeTorch:
        @staticmethod
        def cat(segments, dim: int):
            assert dim == 1
            return FakeTensor(
                [token for segment in segments for token in segment.tokens]
            )

    rendered = _render_prompt(FakeTokenizer(), FakeTorch(), 64)
    assert rendered.shape == (1, 64)
    assert rendered.tokens[:3] == prefix
    assert rendered.tokens[-3:] == suffix


def test_adapter_persists_all_profiles_from_one_runner_result(
    settings, monkeypatch
) -> None:
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    research = _research(database, source["id"])
    calls: list[str] = []

    def execute(run_id: str) -> dict[str, Any]:
        calls.append(run_id)
        return _payload(run_id)

    runtime = Runtime()
    adapter = HuggingFaceBeyefendiBenchmarkAdapter(
        AdapterContext(
            settings=replace(settings, wsl_distro="Ubuntu-24.04"),
            database=database,
            telemetry=TelemetryService(),
            research=research,
            gpu_source=source,
            runtime=runtime,
            gpu_lock=threading.Lock(),
            repository_lock=threading.Lock(),
        ),
        benchmark_executor=execute,
    )
    if __import__("os").name == "nt":
        monkeypatch.setattr(adapter, "_resolve_windows_runtime", lambda: None)

    with pytest.raises(ResearchComplete):
        adapter.run_iteration()

    detail = database.get_research(research["id"], detail=True)
    assert len(calls) == 1
    assert len(detail["experiments"]) == len(PROFILES)
    assert [item["metric_value"] for item in detail["experiments"]] == [
        41.0,
        42.0,
        43.0,
        44.0,
    ]
    assert [item["previous_best"] for item in detail["experiments"]] == [
        None,
        41.0,
        42.0,
        43.0,
    ]
    assert detail["baseline_value"] == 41.0
    assert detail["best_value"] == 44.0
    assert adapter._completed_single_load_run(detail) is True


def test_native_windows_without_wsl_has_an_explicit_error(
    settings, monkeypatch
) -> None:
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    research = _research(database, source["id"])
    adapter = HuggingFaceBeyefendiBenchmarkAdapter(
        AdapterContext(
            settings=replace(settings, wsl_distro=None),
            database=database,
            telemetry=TelemetryService(),
            research=research,
            gpu_source=source,
            runtime=Runtime(),
            gpu_lock=threading.Lock(),
            repository_lock=threading.Lock(),
        ),
        benchmark_executor=lambda run_id: _payload(run_id),
    )

    if __import__("os").name == "nt":
        monkeypatch.setenv("AUTORESEARCH_HF_WSL_DISTRO", "Missing-Distro")
        monkeypatch.setattr(
            adapter,
            "_probe_wsl",
            lambda distro, command: __import__("subprocess").CompletedProcess(
                [], 1, "", "not found"
            ),
        )
        with pytest.raises(RuntimeError, match="Native Windows"):
            adapter.prepare()
