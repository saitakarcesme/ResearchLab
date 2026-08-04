from __future__ import annotations

import csv
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

GPU_QUERY = (
    "--query-gpu=index,uuid,name,utilization.gpu,temperature.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
)
PROCESS_QUERY = (
    "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
    "--format=csv,noheader,nounits",
)


def _number(value: str, cast: type[int | float] = float) -> int | float | None:
    value = value.strip()
    if not value or value.lower() in {"n/a", "[n/a]", "not supported", "-"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if not match:
        return None
    return cast(float(match.group(0)))


def parse_gpu_csv(output: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for row in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(row) != 7:
            continue
        index, uuid, name, utilization, temperature, memory_used, memory_total = (
            part.strip() for part in row
        )
        gpus.append(
            {
                "index": _number(index, int),
                "uuid": uuid,
                "name": name,
                "utilization_percent": _number(utilization),
                "temperature_c": _number(temperature),
                "memory_used_mb": _number(memory_used),
                "memory_total_mb": _number(memory_total),
            }
        )
    return gpus


def parse_process_csv(output: str) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for row in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(row) != 4:
            continue
        gpu_uuid, pid, process_name, memory = (part.strip() for part in row)
        processes.append(
            {
                "gpu_uuid": gpu_uuid,
                "pid": _number(pid, int),
                "process_name": process_name,
                "memory_used_mb": _number(memory),
            }
        )
    return processes


def _aggregate(gpus: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not gpus:
        return {
            "gpu_name": None,
            "utilization_percent": None,
            "temperature_c": None,
            "memory_used_mb": None,
            "memory_total_mb": None,
        }
    utilizations = [
        float(g["utilization_percent"])
        for g in gpus
        if g.get("utilization_percent") is not None
    ]
    temperatures = [
        float(g["temperature_c"]) for g in gpus if g.get("temperature_c") is not None
    ]
    used = [
        float(g["memory_used_mb"]) for g in gpus if g.get("memory_used_mb") is not None
    ]
    total = [
        float(g["memory_total_mb"])
        for g in gpus
        if g.get("memory_total_mb") is not None
    ]
    return {
        "gpu_name": ", ".join(str(g["name"]) for g in gpus),
        "utilization_percent": sum(utilizations) / len(utilizations)
        if utilizations
        else None,
        "temperature_c": max(temperatures) if temperatures else None,
        "memory_used_mb": sum(used) if used else None,
        "memory_total_mb": sum(total) if total else None,
    }


@dataclass(slots=True)
class SSHCommandRunner:
    source: Mapping[str, Any]

    def base_command(self) -> list[str]:
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-p",
            str(self.source.get("port") or 22),
        ]
        if self.source.get("auth_method") == "key_env":
            normalized = re.sub(r"[^A-Za-z0-9]", "_", str(self.source["id"])).upper()
            identity = os.getenv(f"AUTORESEARCH_SSH_KEY_{normalized}") or os.getenv(
                "AUTORESEARCH_SSH_KEY_DEFAULT"
            )
            if not identity:
                raise RuntimeError(
                    f"Set AUTORESEARCH_SSH_KEY_{normalized} to the SSH private-key path; key contents are never stored"
                )
            command.extend(["-i", identity])
        command.append(f"{self.source['username']}@{self.source['host']}")
        return command

    def run(
        self,
        remote_command: str,
        timeout: int = 15,
        *,
        stdin_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*self.base_command(), remote_command],
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace",
        )


class TelemetryService:
    def __init__(self, nvidia_smi_binary: str = "nvidia-smi"):
        self.nvidia_smi_binary = nvidia_smi_binary

    @staticmethod
    def _sampled_at() -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    def _local(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.nvidia_smi_binary, *args],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    def _remote(
        self, source: Mapping[str, Any], args: Sequence[str]
    ) -> subprocess.CompletedProcess[str]:
        safe_args = " ".join(args)
        return SSHCommandRunner(source).run(f"nvidia-smi {safe_args}")

    def sample(self, source: Mapping[str, Any]) -> dict[str, Any]:
        sampled_at = self._sampled_at()
        try:
            run = (
                self._local
                if source["type"] == "local"
                else lambda args: self._remote(source, args)
            )
            gpu_result = run(GPU_QUERY)
            if gpu_result.returncode != 0:
                error = (
                    gpu_result.stderr or gpu_result.stdout or "nvidia-smi failed"
                ).strip()
                raise RuntimeError(error)
            process_result = run(PROCESS_QUERY)
            gpus = parse_gpu_csv(gpu_result.stdout)
            if not gpus:
                raise RuntimeError("nvidia-smi returned no GPU rows")
            processes = (
                parse_process_csv(process_result.stdout)
                if process_result.returncode == 0
                else []
            )
            requested_indices = {
                int(value.strip())
                for value in os.getenv("AUTORESEARCH_CUDA_VISIBLE_DEVICES", "0").split(
                    ","
                )
                if value.strip().isdigit()
            }
            selected_gpus = [
                gpu for gpu in gpus if gpu.get("index") in requested_indices
            ]
            if not selected_gpus:
                selected_gpus = gpus
            selected_uuids = {str(gpu["uuid"]) for gpu in selected_gpus}
            selected_processes = [
                process
                for process in processes
                if process.get("gpu_uuid") in selected_uuids
            ]
            return {
                "available": True,
                "source_id": source["id"],
                "sampled_at": sampled_at,
                "error": None,
                **_aggregate(selected_gpus),
                "selected_gpu_indices": [gpu["index"] for gpu in selected_gpus],
                "aggregate_all": _aggregate(gpus),
                "gpus": gpus,
                "processes": selected_processes,
                "all_processes": processes,
            }
        except (
            FileNotFoundError,
            subprocess.TimeoutExpired,
            OSError,
            RuntimeError,
        ) as exc:
            return {
                "available": False,
                "source_id": source.get("id"),
                "sampled_at": sampled_at,
                "error": str(exc)[:500],
                **_aggregate([]),
                "selected_gpu_indices": [],
                "aggregate_all": _aggregate([]),
                "gpus": [],
                "processes": [],
                "all_processes": [],
            }
