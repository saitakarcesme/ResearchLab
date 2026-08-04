from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from backend.adapters.autoresearch import (
    build_managed_posix_script,
    build_posix_group_signal_script,
    build_posix_group_terminate_script,
    build_setsid_wait_argv,
)
from backend.adapters.base import (
    AdapterContext,
    ResearchAdapter,
    ResearchComplete,
    ResearchFailed,
)
from backend.process_control import run_managed_process
from backend.runners.beyefendi_v2_benchmark import (
    ADAPTER_CONFIG_SHA256,
    ADAPTER_FILE_SHA256,
    ADAPTER_REPOSITORY,
    ADAPTER_REVISION,
    BASE_MODEL,
    BASE_REVISION,
    MODEL_ID,
    PROFILES,
    RESULT_PREFIX,
)
from backend.telemetry import SSHCommandRunner

BENCHMARK_PROFILE = "hf-transformers-text-v1"
RUNTIME_PROVIDER = "huggingface"
RUNTIME_REQUIREMENTS = (
    "torch==2.13.0",
    "transformers==5.14.1",
    "peft==0.19.1",
    "accelerate==1.14.0",
    "bitsandbytes==0.49.2",
    "safetensors>=0.6.2",
)
BENCHMARK_TIMEOUT_SECONDS = 7200
PROFILE_MARKER = re.compile(r"^Beyefendi-v2 run ([0-9a-f]{32}) profile (\d+)/(\d+):")
BenchmarkExecutor = Callable[[str], Mapping[str, Any]]


def _windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    remainder = resolved.as_posix().split(":", 1)[-1]
    return f"/mnt/{drive}{remainder}"


def _tail(path: Path, limit: int = 3000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:].strip()
    except OSError:
        return ""


def _finite_positive(value: Any, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric") from exc
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{field} must be a finite positive number")
    return converted


def _validated_profile_metrics(
    profile: Mapping[str, Any], expected: Any
) -> tuple[float, float]:
    repeats = profile.get("repeats")
    if not isinstance(repeats, list) or len(repeats) != 3:
        raise ValueError(f"{expected.id} must contain three measured repeats")
    speeds: list[float] = []
    latencies: list[float] = []
    expected_total = expected.batch_size * expected.output_tokens_per_request
    for index, repeat in enumerate(repeats, start=1):
        if not isinstance(repeat, Mapping):
            raise TypeError(f"{expected.id} repeat {index} is invalid")
        speed = _finite_positive(
            repeat.get("output_tokens_per_second"),
            f"{expected.id}.repeat_{index}.output_tokens_per_second",
        )
        elapsed = _finite_positive(
            repeat.get("elapsed_seconds"),
            f"{expected.id}.repeat_{index}.elapsed_seconds",
        )
        latency = _finite_positive(
            repeat.get("request_latency_seconds"),
            f"{expected.id}.repeat_{index}.request_latency_seconds",
        )
        if repeat.get("input_tokens_per_request") != expected.target_input_tokens:
            raise ValueError(
                f"{expected.id} repeat {index} did not use the fixed input length"
            )
        if (
            repeat.get("output_tokens_per_request")
            != expected.output_tokens_per_request
        ):
            raise ValueError(
                f"{expected.id} repeat {index} did not use the fixed output length"
            )
        if repeat.get("total_output_tokens") != expected_total:
            raise ValueError(
                f"{expected.id} repeat {index} returned the wrong batch token total"
            )
        if not math.isclose(latency, elapsed, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"{expected.id} repeat {index} latency does not match elapsed time"
            )
        derived_speed = expected_total / elapsed
        if not math.isclose(speed, derived_speed, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"{expected.id} repeat {index} throughput is not derived from tokens/time"
            )
        speeds.append(speed)
        latencies.append(latency)
    median_speed = statistics.median(speeds)
    median_latency = statistics.median(latencies)
    reported_speed = _finite_positive(
        profile.get("median_output_tokens_per_second"),
        f"{expected.id}.median_output_tokens_per_second",
    )
    reported_latency = _finite_positive(
        profile.get("median_request_latency_seconds"),
        f"{expected.id}.median_request_latency_seconds",
    )
    if not math.isclose(
        reported_speed, median_speed, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError(f"{expected.id} reported an inconsistent median speed")
    if not math.isclose(
        reported_latency, median_latency, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError(f"{expected.id} reported an inconsistent median latency")
    return median_speed, median_latency


def parse_benchmark_output(output: str, *, expected_run_id: str) -> dict[str, Any]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    result_lines = [line for line in lines if line.startswith(RESULT_PREFIX)]
    if len(result_lines) != 1:
        raise ValueError(
            f"expected one terminal benchmark JSON result, found {len(result_lines)}"
        )
    try:
        payload = json.loads(result_lines[0][len(RESULT_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise ValueError("benchmark runner returned invalid JSON") from exc
    validate_benchmark_payload(payload, expected_run_id=expected_run_id)
    return payload


def validate_benchmark_payload(
    payload: Mapping[str, Any], *, expected_run_id: str
) -> None:
    if payload.get("schema_version") != 1 or payload.get("run_id") != expected_run_id:
        raise ValueError("benchmark schema or run id does not match")
    model = payload.get("model")
    expected_model = {
        "id": MODEL_ID,
        "adapter_repository": ADAPTER_REPOSITORY,
        "adapter_revision": ADAPTER_REVISION,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
    }
    if not isinstance(model, Mapping) or any(
        model.get(key) != value for key, value in expected_model.items()
    ):
        raise ValueError("benchmark model pins do not match the requested checkpoint")
    runtime = payload.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("scheduler_share_percent") != 100
    ):
        raise ValueError("benchmark did not preserve its 100% scheduler share")
    if payload.get("fatal_error"):
        raise RuntimeError(str(payload["fatal_error"]))
    if (
        runtime.get("quantization") != "nf4"
        or runtime.get("compute_dtype") != "bfloat16"
        or runtime.get("double_quantization") is not True
        or runtime.get("device_map") != {"": 0}
        or runtime.get("warmup_runs_per_profile") != 1
        or runtime.get("measured_repeats_per_profile") != 3
    ):
        raise ValueError(
            "benchmark runtime does not match the pinned full-GPU protocol"
        )
    gpu = payload.get("gpu")
    if not isinstance(gpu, Mapping) or gpu.get("fully_gpu_resident") is not True:
        raise ValueError("benchmark did not confirm full GPU model residency")
    if gpu.get("adapter_file_sha256") != ADAPTER_FILE_SHA256:
        raise ValueError("benchmark adapter weights do not match the pinned SHA-256")
    if gpu.get("adapter_config_sha256") != ADAPTER_CONFIG_SHA256:
        raise ValueError("benchmark adapter config does not match the pinned SHA-256")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != len(PROFILES):
        raise ValueError("benchmark did not return every fixed profile")
    for expected, profile in zip(PROFILES, profiles, strict=True):
        expected_profile = {
            "id": expected.id,
            "label": expected.label,
            "target_input_tokens": expected.target_input_tokens,
            "batch_size": expected.batch_size,
            "output_tokens_per_request": expected.output_tokens_per_request,
        }
        if not isinstance(profile, Mapping) or any(
            profile.get(key) != value for key, value in expected_profile.items()
        ):
            raise ValueError("benchmark profiles are missing or out of order")
        if profile.get("error"):
            continue
        _validated_profile_metrics(profile, expected)


class HuggingFaceBeyefendiBenchmarkAdapter(ResearchAdapter):
    """Benchmark the pinned Beyefendi-v2 adapter once across fixed GPU profiles."""

    def __init__(
        self,
        context: AdapterContext,
        *,
        benchmark_executor: BenchmarkExecutor | None = None,
    ):
        super().__init__(context)
        self.settings = context.settings
        self.db = context.database
        self.research_id = str(context.research["id"])
        self.research_log_dir = self.settings.log_dir / self.research_id
        self.research_log_dir.mkdir(parents=True, exist_ok=True)
        self.runner_path = (
            Path(__file__).resolve().parents[1]
            / "runners"
            / "beyefendi_v2_benchmark.py"
        )
        self.benchmark_executor = benchmark_executor
        self._wsl_runtime_root: PurePosixPath | None = None
        self._resolved_wsl_distro: str | None = None
        self._resolved_wsl_home: str | None = None
        self._prebuilt_python: str | None = None

    def prepare(self) -> None:
        research = self.db.get_research(self.research_id) or self.context.research
        self._validate_research(research)
        recovered = self.db.close_incomplete_experiments(
            self.research_id,
            "Backend stopped before the single-load Beyefendi-v2 benchmark completed.",
        )
        if recovered:
            self.db.add_log(
                self.research_id,
                "experiment_recovered",
                "An interrupted speed run was closed; all profiles will be measured together again.",
                level="warning",
                data={"experiment_numbers": recovered},
            )
        if self.benchmark_executor is None:
            self._prepare_runtime()
        telemetry = self.context.telemetry.sample(self.context.gpu_source)
        self.db.add_log(
            self.research_id,
            "model_verified",
            "Beyefendi-v2 and its Qwen3.5-9B base are pinned for a full-GPU speed run.",
            data={
                "model_id": MODEL_ID,
                "adapter_revision": ADAPTER_REVISION,
                "adapter_file_sha256": ADAPTER_FILE_SHA256,
                "adapter_config_sha256": ADAPTER_CONFIG_SHA256,
                "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION,
                "benchmark_profile": BENCHMARK_PROFILE,
                "scheduler_share_percent": 100,
                "gpu_name": telemetry.get("gpu_name"),
                "requirements": list(RUNTIME_REQUIREMENTS),
            },
        )

    def _validate_research(self, research: Mapping[str, Any]) -> None:
        if research.get("model_id") != MODEL_ID:
            raise RuntimeError(f"This adapter only runs the pinned model {MODEL_ID}")
        if research.get("model_digest") != ADAPTER_REVISION:
            raise RuntimeError(
                "The queued Beyefendi-v2 adapter revision is not pinned correctly"
            )
        if research.get("benchmark_profile") != BENCHMARK_PROFILE:
            raise RuntimeError(
                "The queued benchmark profile is not Beyefendi-v2 speed v1"
            )
        runtime = str(research.get("model_runtime") or RUNTIME_PROVIDER).lower()
        if runtime not in {
            RUNTIME_PROVIDER,
            "huggingface.co",
            "https://huggingface.co",
        }:
            raise RuntimeError("The queued model runtime is not Hugging Face")
        if int(research.get("target_gpu_allocation") or 0) != 100:
            raise RuntimeError(
                "Beyefendi-v2 speed research requires a 100% GPU scheduler share"
            )
        source = self.context.gpu_source
        if source.get("type") not in {"local", "remote"}:
            raise RuntimeError(
                "Beyefendi-v2 requires a local WSL/POSIX or remote GPU source"
            )
        if source.get("type") == "local" and os.name == "nt":
            self._resolve_windows_runtime()

    def _source_key(self) -> str:
        return hashlib.sha256(
            str(self.context.gpu_source["id"]).encode("utf-8")
        ).hexdigest()[:16]

    @staticmethod
    def _probe_wsl(distro: str, command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["wsl.exe", "-d", distro, "--", "sh", "-lc", command],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    def _resolve_windows_runtime(self) -> None:
        if self._resolved_wsl_distro is not None:
            return
        explicit_distro = os.getenv("AUTORESEARCH_HF_WSL_DISTRO", "").strip()
        candidates = (
            [explicit_distro]
            if explicit_distro
            else ["Ubuntu", str(self.settings.wsl_distro or "")]
        )
        candidates = list(dict.fromkeys(value for value in candidates if value))
        configured_python = os.getenv("AUTORESEARCH_HF_PYTHON", "").strip()
        available_distros: list[str] = []
        validation = (
            "import accelerate,bitsandbytes,peft,torch,transformers;"
            "assert torch.__version__=='2.13.0+cu130';"
            "assert transformers.__version__=='5.14.1';"
            "assert peft.__version__=='0.19.1';"
            "assert accelerate.__version__=='1.14.0';"
            "assert bitsandbytes.__version__=='0.49.2';"
            "assert torch.cuda.is_available();"
            "assert torch.cuda.device_count()==1"
        )
        for distro in candidates:
            try:
                home_result = self._probe_wsl(distro, "printf '%s' \"$HOME\"")
            except (OSError, subprocess.SubprocessError):
                continue
            home = home_result.stdout.strip()
            if home_result.returncode != 0 or not home.startswith("/"):
                continue
            available_distros.append(distro)
            python = configured_python or str(
                PurePosixPath(home) / ".cache" / "beyefendi-v2-venv" / "bin" / "python"
            )
            probe = self._probe_wsl(
                distro,
                f"test -x {shlex.quote(python)} && "
                f"{shlex.quote(python)} -c {shlex.quote(validation)}",
            )
            if probe.returncode == 0:
                self._resolved_wsl_distro = distro
                self._resolved_wsl_home = home
                self._prebuilt_python = python
                return
        if configured_python:
            raise RuntimeError(
                "AUTORESEARCH_HF_PYTHON did not pass the pinned CUDA runtime preflight "
                "in AUTORESEARCH_HF_WSL_DISTRO or an available WSL distribution."
            )
        if explicit_distro and explicit_distro not in available_distros:
            raise RuntimeError(
                "Native Windows execution is unsupported for this CUDA benchmark. "
                "AUTORESEARCH_HF_WSL_DISTRO is not an available WSL distribution; "
                "configure a valid WSL runtime or select a remote GPU source."
            )
        fallback = explicit_distro or (
            str(self.settings.wsl_distro)
            if self.settings.wsl_distro in available_distros
            else None
        )
        fallback = fallback or (available_distros[0] if available_distros else None)
        if fallback is None:
            raise RuntimeError(
                "Native Windows execution is unsupported for this CUDA benchmark. "
                "Set AUTORESEARCH_HF_WSL_DISTRO and optionally AUTORESEARCH_HF_PYTHON, "
                "or select a remote GPU source."
            )
        self._resolved_wsl_distro = fallback
        if fallback in available_distros:
            home_result = self._probe_wsl(fallback, "printf '%s' \"$HOME\"")
            self._resolved_wsl_home = home_result.stdout.strip()

    def _wsl_distro(self) -> str:
        self._resolve_windows_runtime()
        if self._resolved_wsl_distro is None:
            raise RuntimeError("A WSL distribution is required")
        return self._resolved_wsl_distro

    def _wsl_root(self) -> PurePosixPath:
        if self._wsl_runtime_root is not None:
            return self._wsl_runtime_root
        configured = os.getenv("AUTORESEARCH_WSL_RUNTIME_ROOT", "").strip()
        if configured:
            base = PurePosixPath(configured)
        else:
            result = subprocess.run(
                ["wsl.exe", "-d", self._wsl_distro(), "--", "sh", "-s"],
                input="printf '%s' \"$HOME\"",
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            home = result.stdout.strip()
            if result.returncode != 0 or not home.startswith("/"):
                raise RuntimeError(
                    "Could not resolve the configured WSL home directory"
                )
            base = PurePosixPath(home) / ".cache" / "researchlab" / "runtimes"
        if not base.is_absolute():
            raise RuntimeError("The WSL runtime root must be an absolute POSIX path")
        self._wsl_runtime_root = base / self._source_key() / "huggingface-beyefendi-v2"
        return self._wsl_runtime_root

    def _remote_root(self) -> PurePosixPath:
        return (
            PurePosixPath(str(self.context.gpu_source["workspace_path"]))
            / ".researchlab-runtimes"
            / self._source_key()
            / "huggingface-beyefendi-v2"
        )

    def _native_root(self) -> Path:
        return (
            self.settings.runtime_dir / self._source_key() / "huggingface-beyefendi-v2"
        )

    def _runtime_python(self) -> str:
        if self.context.gpu_source["type"] == "remote":
            return str(self._remote_root() / "venv" / "bin" / "python")
        if os.name == "nt":
            self._resolve_windows_runtime()
            if self._prebuilt_python:
                return self._prebuilt_python
            return str(self._wsl_root() / "venv" / "bin" / "python")
        return str(self._native_root() / "venv" / "bin" / "python")

    def _runtime_cache(self) -> str:
        if self.context.gpu_source["type"] == "remote":
            return os.getenv("AUTORESEARCH_HF_REMOTE_HOME", "").strip() or str(
                self._remote_root() / "huggingface-cache"
            )
        if os.name == "nt":
            self._resolve_windows_runtime()
            configured = os.getenv("AUTORESEARCH_HF_HOME", "").strip()
            if configured:
                return configured
            if self._resolved_wsl_home:
                return str(
                    PurePosixPath(self._resolved_wsl_home) / ".cache" / "huggingface"
                )
            return str(self._wsl_root() / "huggingface-cache")
        return os.getenv("AUTORESEARCH_HF_HOME", "").strip() or str(
            self._native_root() / "huggingface-cache"
        )

    def _runtime_runner(self) -> str:
        if self.context.gpu_source["type"] == "remote":
            return str(self._remote_root() / self.runner_path.name)
        if os.name == "nt":
            return _windows_to_wsl(self.runner_path)
        return str(self.runner_path)

    def _local_adapter_path(self) -> str | None:
        source = self.context.gpu_source
        if source["type"] == "remote":
            return (
                os.getenv("AUTORESEARCH_BEYEFENDI_REMOTE_ADAPTER_PATH", "").strip()
                or None
            )
        configured = os.getenv("AUTORESEARCH_BEYEFENDI_ADAPTER_PATH", "").strip()
        if configured:
            if os.name == "nt" and Path(configured).is_dir():
                return _windows_to_wsl(Path(configured))
            return configured
        return None

    def _fingerprint(self) -> str:
        source = self.runner_path.read_bytes()
        payload = json.dumps(
            {
                "requirements": RUNTIME_REQUIREMENTS,
                "adapter_revision": ADAPTER_REVISION,
                "base_revision": BASE_REVISION,
                "runner_sha256": hashlib.sha256(source).hexdigest(),
                "source_id": self.context.gpu_source["id"],
                "wsl_distro": self._resolved_wsl_distro or self.settings.wsl_distro,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _prepare_runtime(self) -> None:
        if os.name == "nt":
            self._resolve_windows_runtime()
            if self._prebuilt_python:
                self.db.add_log(
                    self.research_id,
                    "environment_ready",
                    "Using the preflighted Beyefendi-v2 CUDA runtime in WSL.",
                    data={
                        "wsl_distro": self._wsl_distro(),
                        "python": self._prebuilt_python,
                        "torch": "2.13.0+cu130",
                    },
                )
                return
        marker = self.research_log_dir / ".beyefendi-v2-runtime"
        fingerprint = self._fingerprint()
        if (
            marker.exists()
            and marker.read_text(encoding="utf-8").strip() == fingerprint
            and self._runtime_exists()
        ):
            if self.context.gpu_source["type"] == "remote":
                self._copy_remote_runner()
            return
        self.db.add_log(
            self.research_id,
            "environment_started",
            "Preparing the pinned Beyefendi-v2 benchmark environment.",
        )
        runtime_python = self._runtime_python()
        runtime_root = str(PurePosixPath(runtime_python).parents[2])
        version_check = (
            "import accelerate, bitsandbytes, peft, safetensors, torch, transformers; "
            "assert torch.__version__ == '2.13.0+cu130'; "
            "assert torch.cuda.is_available(); "
            "assert transformers.__version__ == '5.14.1'; "
            "assert peft.__version__ == '0.19.1'; "
            "assert accelerate.__version__ == '1.14.0'; "
            "assert bitsandbytes.__version__ == '0.49.2'"
        )
        requirements = " ".join(shlex.quote(item) for item in RUNTIME_REQUIREMENTS)
        body = (
            f"mkdir -p {shlex.quote(runtime_root)} && "
            f"test -x {shlex.quote(runtime_python)} || "
            f"{shlex.quote(self.settings.uv_binary)} venv --python 3.12 --system-site-packages "
            f"{shlex.quote(str(PurePosixPath(runtime_root) / 'venv'))}; "
            f"{shlex.quote(self.settings.uv_binary)} pip install --python "
            f"{shlex.quote(runtime_python)} {requirements} && "
            f"{shlex.quote(runtime_python)} -c {shlex.quote(version_check)}"
        )
        self._run_posix_setup(body)
        if self.context.gpu_source["type"] == "remote":
            self._copy_remote_runner()
        marker.write_text(fingerprint + "\n", encoding="utf-8")
        self.db.add_log(
            self.research_id,
            "environment_ready",
            "Pinned Transformers, PEFT and 4-bit inference dependencies are ready.",
        )

    def _runtime_exists(self) -> bool:
        python = self._runtime_python()
        source = self.context.gpu_source
        if source["type"] == "remote":
            result = SSHCommandRunner(source).run(
                f"test -x {shlex.quote(python)}", timeout=10
            )
            return result.returncode == 0
        if os.name == "nt":
            result = subprocess.run(
                [
                    "wsl.exe",
                    "-d",
                    self._wsl_distro(),
                    "--",
                    "sh",
                    "-lc",
                    f"test -x {shlex.quote(python)}",
                ],
                capture_output=True,
                timeout=10,
                check=False,
            )
            return result.returncode == 0
        return Path(python).is_file()

    def _run_posix_setup(self, body: str) -> None:
        pid_file = f"/tmp/autoresearch-setup-{self.research_id}.pid"
        script = build_managed_posix_script(pid_file, body)
        source = self.context.gpu_source
        if source["type"] == "remote":
            runner = SSHCommandRunner(source)
            command = [
                *runner.base_command(),
                shlex.join(build_setsid_wait_argv(self.research_id)),
            ]
            pause = lambda: self._signal_remote(runner, pid_file, "STOP")
            resume = lambda: self._signal_remote(runner, pid_file, "CONT")
            terminate = lambda: self._terminate_remote(runner, pid_file)
        elif os.name == "nt":
            command = [
                "wsl.exe",
                "-d",
                self._wsl_distro(),
                "--",
                *build_setsid_wait_argv(self.research_id),
            ]
            pause = lambda: self._signal_wsl(pid_file, "STOP")
            resume = lambda: self._signal_wsl(pid_file, "CONT")
            terminate = lambda: self._terminate_wsl(pid_file)
        else:
            command = ["sh", "-s"]
            pause = resume = terminate = None
            script = body
        output = self.research_log_dir / "setup-beyefendi-v2.log"
        result = run_managed_process(
            command,
            cwd=None,
            env=os.environ.copy(),
            output=output,
            timeout_seconds=1800,
            runtime=self.context.runtime,
            stdin_text=script,
            pause_callback=pause,
            resume_callback=resume,
            terminate_callback=terminate,
        )
        if result.control_error:
            self.context.runtime.control_error = result.control_error
            raise RuntimeError(
                "Could not confirm benchmark environment process termination: "
                f"{result.control_error}"
            )
        if result.stopped:
            raise InterruptedError("research stopped during environment preparation")
        if result.timed_out or result.returncode != 0:
            raise RuntimeError(
                f"Beyefendi-v2 environment setup failed: {_tail(output)}"
            )

    def _copy_remote_runner(self) -> None:
        source = self.context.gpu_source
        root = self._remote_root()
        runner = SSHCommandRunner(source)
        mkdir = runner.run(f"mkdir -p {shlex.quote(str(root))}", timeout=15)
        if mkdir.returncode != 0:
            raise RuntimeError((mkdir.stderr or mkdir.stdout).strip())
        command = ["scp", "-q", "-P", str(source.get("port") or 22)]
        ssh_base = runner.base_command()
        if "-i" in ssh_base:
            index = ssh_base.index("-i")
            command.extend(["-i", ssh_base[index + 1]])
        command.extend(
            [
                str(self.runner_path),
                f"{source['username']}@{source['host']}:{shlex.quote(self._runtime_runner())}",
            ]
        )
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())

    def _signal_remote(
        self, runner: SSHCommandRunner, pid_file: str, signal_name: str
    ) -> None:
        script = build_posix_group_signal_script(
            pid_file, self.research_id, signal_name
        )
        result = runner.run("sh -s", timeout=10, stdin_text=script)
        if result.returncode != 0:
            raise RuntimeError(f"Remote process signal {signal_name} failed")

    def _terminate_remote(self, runner: SSHCommandRunner, pid_file: str) -> None:
        script = build_posix_group_terminate_script(pid_file, self.research_id)
        result = runner.run("sh -s", timeout=10, stdin_text=script)
        if result.returncode != 0:
            raise RuntimeError("Remote benchmark process termination failed")

    def _wsl_control(self, script: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["wsl.exe", "-d", self._wsl_distro(), "--", "sh", "-s"],
            input=script.encode("utf-8"),
            capture_output=True,
            timeout=10,
            check=False,
        )

    def _signal_wsl(self, pid_file: str, signal_name: str) -> None:
        result = self._wsl_control(
            build_posix_group_signal_script(pid_file, self.research_id, signal_name)
        )
        if result.returncode != 0:
            raise RuntimeError(f"WSL process signal {signal_name} failed")

    def _terminate_wsl(self, pid_file: str) -> None:
        result = self._wsl_control(
            build_posix_group_terminate_script(pid_file, self.research_id)
        )
        if result.returncode != 0:
            raise RuntimeError("WSL benchmark process termination failed")

    def _acquire_gpu(self) -> bool:
        while not self.context.runtime.stop_event.is_set():
            if not self.context.runtime.wait_until_running():
                return False
            if self.context.gpu_lock.acquire(timeout=0.5):
                return True
        return False

    def _completed_single_load_run(self, research: Mapping[str, Any]) -> bool:
        grouped: dict[str, set[int]] = {}
        for experiment in research.get("experiments") or []:
            if experiment.get("metric_value") is None or experiment.get("error"):
                continue
            match = PROFILE_MARKER.match(str(experiment.get("hypothesis") or ""))
            if not match or int(match.group(3)) != len(PROFILES):
                continue
            grouped.setdefault(match.group(1), set()).add(int(match.group(2)))
        return any(
            indices == set(range(1, len(PROFILES) + 1)) for indices in grouped.values()
        )

    def run_iteration(self) -> None:
        research = self.db.get_research(self.research_id, detail=True)
        if research is None:
            raise InterruptedError
        self._validate_research(research)
        if self._completed_single_load_run(research):
            raise ResearchComplete(
                "The pinned Beyefendi-v2 model completed every single-load speed profile."
            )
        if not self._acquire_gpu():
            raise InterruptedError
        try:
            self.context.runtime.set_phase("evaluating")
            run_id = uuid.uuid4().hex
            experiments = self._create_profile_experiments(research, run_id)
            if self.benchmark_executor is not None:
                payload = dict(self.benchmark_executor(run_id))
                validate_benchmark_payload(payload, expected_run_id=run_id)
            else:
                payload = self._execute_benchmark(run_id)
            failed = self._persist_profiles(research, experiments, payload)
            if failed:
                raise ResearchFailed(
                    f"{len(failed)} of {len(PROFILES)} Beyefendi-v2 speed profiles failed: "
                    + "; ".join(failed)
                )
            raise ResearchComplete(
                "Beyefendi-v2 completed four full-GPU profiles with one model load, "
                "one warmup and three deterministic measurements per profile."
            )
        except InterruptedError:
            self._close_open_run(experiments if "experiments" in locals() else [])
            raise
        except (ResearchComplete, ResearchFailed):
            raise
        except Exception as exc:
            self._close_open_run(
                experiments if "experiments" in locals() else [], error=str(exc)
            )
            raise ResearchFailed(
                f"Beyefendi-v2 benchmark could not finish: {exc}"
            ) from exc
        finally:
            self.context.gpu_lock.release()
            self.context.runtime.set_phase("ready")

    def _create_profile_experiments(
        self, research: Mapping[str, Any], run_id: str
    ) -> list[dict[str, Any]]:
        experiments = []
        previous_best = research.get("best_value")
        for index, profile in enumerate(PROFILES, start=1):
            experiments.append(
                self.db.create_experiment(
                    self.research_id,
                    (
                        f"Beyefendi-v2 run {run_id} profile {index}/{len(PROFILES)}: "
                        f"Measure {profile.label.lower()} generation speed."
                    ),
                    (
                        "Load the pinned NF4 adapter/base pair once for the full run; "
                        f"warm this profile and measure three deterministic repeats "
                        f"at batch {profile.batch_size}."
                    ),
                    previous_best,
                )
            )
        self.db.add_log(
            self.research_id,
            "benchmark_started",
            "Beyefendi-v2 speed measurement started with the GPU reserved exclusively.",
            data={
                "run_id": run_id,
                "scheduler_share_percent": 100,
                "adapter_revision": ADAPTER_REVISION,
                "base_revision": BASE_REVISION,
            },
        )
        return experiments

    def _execute_benchmark(self, run_id: str) -> dict[str, Any]:
        output = self.research_log_dir / f"beyefendi-v2-{run_id}.log"
        environment = {
            "CUDA_VISIBLE_DEVICES": os.getenv("AUTORESEARCH_CUDA_VISIBLE_DEVICES", "0"),
            "HF_HOME": self._runtime_cache(),
            "TOKENIZERS_PARALLELISM": "false",
            "AUTORESEARCH_TARGET_GPU_ALLOCATION": "100",
        }
        arguments = [
            self._runtime_python(),
            self._runtime_runner(),
            "--run-id",
            run_id,
            "--scheduler-share",
            "100",
        ]
        adapter_path = self._local_adapter_path()
        if adapter_path:
            arguments.extend(["--adapter-path", adapter_path])
        pid_file = f"/tmp/autoresearch-{self.research_id}.pid"
        body = (
            "env "
            + " ".join(
                f"{key}={shlex.quote(value)}" for key, value in environment.items()
            )
            + " "
            + shlex.join(arguments)
        )
        script = build_managed_posix_script(pid_file, body)
        source = self.context.gpu_source
        if source["type"] == "remote":
            runner = SSHCommandRunner(source)
            command = [
                *runner.base_command(),
                shlex.join(build_setsid_wait_argv(self.research_id)),
            ]
            pause = lambda: self._signal_remote(runner, pid_file, "STOP")
            resume = lambda: self._signal_remote(runner, pid_file, "CONT")
            terminate = lambda: self._terminate_remote(runner, pid_file)
            process_env = None
        elif os.name == "nt":
            command = [
                "wsl.exe",
                "-d",
                self._wsl_distro(),
                "--",
                *build_setsid_wait_argv(self.research_id),
            ]
            pause = lambda: self._signal_wsl(pid_file, "STOP")
            resume = lambda: self._signal_wsl(pid_file, "CONT")
            terminate = lambda: self._terminate_wsl(pid_file)
            process_env = None
        else:
            command = arguments
            pause = resume = terminate = None
            process_env = {**os.environ, **environment}
            script = None
        result = run_managed_process(
            command,
            cwd=None,
            env=process_env,
            output=output,
            timeout_seconds=max(
                BENCHMARK_TIMEOUT_SECONDS,
                self.settings.experiment_timeout_seconds,
            ),
            runtime=self.context.runtime,
            stdin_text=script,
            pause_callback=pause,
            resume_callback=resume,
            terminate_callback=terminate,
        )
        if result.control_error:
            self.context.runtime.control_error = result.control_error
            raise RuntimeError(
                "Could not confirm benchmark process termination: "
                f"{result.control_error}"
            )
        if result.stopped:
            raise InterruptedError("Beyefendi-v2 benchmark was stopped")
        if result.timed_out:
            raise RuntimeError("Beyefendi-v2 loading or measurement exceeded two hours")
        raw = output.read_text(encoding="utf-8", errors="replace")
        try:
            payload = parse_benchmark_output(raw, expected_run_id=run_id)
        except Exception as exc:
            detail = _tail(output, 1200)
            raise RuntimeError(
                f"Benchmark result was invalid: {exc}. {detail}"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"Beyefendi-v2 runner exited with status {result.returncode}"
            )
        return payload

    def _persist_profiles(
        self,
        research: Mapping[str, Any],
        experiments: Sequence[Mapping[str, Any]],
        payload: Mapping[str, Any],
    ) -> list[str]:
        failed: list[str] = []
        best = (
            float(research["best_value"])
            if research.get("best_value") is not None
            else None
        )
        baseline = research.get("baseline_value")
        for expected, experiment, result in zip(
            PROFILES,
            experiments,
            payload["profiles"],
            strict=True,
        ):
            error = result.get("error")
            if error:
                failed.append(f"{expected.label}: {error}")
                self.db.finish_experiment(
                    str(experiment["id"]),
                    metric_value=None,
                    accepted=False,
                    git_commit=None,
                    error=str(error),
                )
                continue
            metric, _ = _validated_profile_metrics(result, expected)
            self.db.update_experiment_previous_best(
                str(experiment["id"]), best
            )
            accepted = best is None or metric > best
            updates: dict[str, float] = {}
            if baseline is None:
                baseline = metric
                updates["baseline_value"] = metric
            if accepted:
                best = metric
                updates["best_value"] = metric
            token_samples = list(result["repeats"])
            warmup = result.get("warmup")
            if (
                isinstance(warmup, Mapping)
                and warmup.get("input_tokens_per_request") is not None
                and warmup.get("total_output_tokens") is not None
            ):
                token_samples.append(warmup)
            token_count = sum(
                int(sample["input_tokens_per_request"]) * expected.batch_size
                + int(sample["total_output_tokens"])
                for sample in token_samples
            )
            self.db.finish_experiment(
                str(experiment["id"]),
                metric_value=metric,
                accepted=accepted,
                git_commit=None,
                token_count=token_count,
                research_updates=updates or None,
            )
            self.db.add_log(
                self.research_id,
                "experiment_accepted" if accepted else "experiment_rejected",
                (
                    f"{expected.label} produced {metric:.1f} output tokens per second "
                    "across three deterministic measurements."
                ),
                experiment_id=str(experiment["id"]),
                data={
                    **dict(result),
                    "model_load_seconds": payload["gpu"].get("load_seconds"),
                    "scheduler_share_percent": 100,
                    "adapter_revision": ADAPTER_REVISION,
                    "base_revision": BASE_REVISION,
                },
            )
        return failed

    def _close_open_run(
        self,
        experiments: Sequence[Mapping[str, Any]],
        *,
        error: str = "Benchmark stopped before all profiles completed",
    ) -> None:
        for experiment in experiments:
            current = self.db.get_experiment(str(experiment["id"]))
            if current is None or current.get("completed_at") is not None:
                continue
            self.db.finish_experiment(
                str(experiment["id"]),
                metric_value=None,
                accepted=False,
                git_commit=None,
                error=error[:3000],
            )
