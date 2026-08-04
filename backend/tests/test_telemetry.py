from __future__ import annotations

import subprocess

from backend.telemetry import TelemetryService, parse_gpu_csv, parse_process_csv

GPU_OUTPUT = """0, GPU-aaa, NVIDIA GeForce RTX 3090, 39, 37, 1953, 24576
1, GPU-bbb, NVIDIA A10, 10 %, 41 C, 100 MiB, 23028 MiB
"""


def test_telemetry_csv_parsers_handle_units_and_wddm_na() -> None:
    gpus = parse_gpu_csv(GPU_OUTPUT)
    assert gpus[0] == {
        "index": 0,
        "uuid": "GPU-aaa",
        "name": "NVIDIA GeForce RTX 3090",
        "utilization_percent": 39.0,
        "temperature_c": 37.0,
        "memory_used_mb": 1953.0,
        "memory_total_mb": 24576.0,
    }
    processes = parse_process_csv("GPU-aaa, 1234, C:\\ollama.exe, [N/A]\n")
    assert processes[0]["pid"] == 1234
    assert processes[0]["memory_used_mb"] is None


def test_telemetry_sample_returns_aggregate_and_processes(monkeypatch) -> None:
    service = TelemetryService()
    calls = iter(
        [
            subprocess.CompletedProcess([], 0, GPU_OUTPUT, ""),
            subprocess.CompletedProcess([], 0, "GPU-aaa, 42, python.exe, 512\n", ""),
        ]
    )
    monkeypatch.setattr(service, "_local", lambda _args: next(calls))
    sample = service.sample({"id": "local", "type": "local"})
    assert sample["available"] is True
    assert sample["selected_gpu_indices"] == [0]
    assert sample["utilization_percent"] == 39.0
    assert sample["temperature_c"] == 37.0
    assert sample["memory_total_mb"] == 24576.0
    assert sample["aggregate_all"]["utilization_percent"] == 24.5
    assert sample["aggregate_all"]["memory_total_mb"] == 47604.0
    assert sample["processes"][0]["process_name"] == "python.exe"


def test_telemetry_unavailable_is_graceful(monkeypatch) -> None:
    service = TelemetryService()
    monkeypatch.setattr(
        service,
        "_local",
        lambda _args: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    sample = service.sample({"id": "local", "type": "local"})
    assert sample["available"] is False
    assert sample["gpus"] == []
    assert "missing" in sample["error"]
