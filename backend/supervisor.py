from __future__ import annotations

import os
import shlex
import subprocess
import threading
from collections.abc import Callable
from typing import Any

from backend.adapters import (
    AdapterContext,
    KarpathyAutoresearchAdapter,
    ResearchAdapter,
)
from backend.config import Settings
from backend.db import Database, utc_now
from backend.process_control import ManagedProcess
from backend.telemetry import SSHCommandRunner, TelemetryService


class ResearchRuntime:
    def __init__(self, research_id: str):
        self.research_id = research_id
        self.stop_event = threading.Event()
        self.run_event = threading.Event()
        self.run_event.set()
        self._process: ManagedProcess | None = None
        self._lock = threading.RLock()
        self.thread: threading.Thread | None = None
        self.control_error: str | None = None
        self.phase = "starting"
        self.phase_changed_at = utc_now()

    def set_process(self, process: ManagedProcess | None) -> None:
        with self._lock:
            self._process = process
            if process is not None and not self.run_event.is_set():
                process.pause()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            if self.phase != phase:
                self.phase = phase
                self.phase_changed_at = utc_now()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "phase": self.phase,
                "phase_changed_at": self.phase_changed_at,
                "subprocess_attached": self._process is not None,
            }

    def wait_until_running(self) -> bool:
        while not self.stop_event.is_set():
            if self.run_event.wait(0.5):
                return True
        return False

    def pause(self) -> None:
        self.run_event.clear()
        try:
            with self._lock:
                if self._process:
                    self._process.pause()
        except Exception:
            self.run_event.set()
            raise

    def resume(self) -> None:
        with self._lock:
            if self._process:
                self._process.resume()
        self.run_event.set()

    def stop(self) -> None:
        self.stop_event.set()
        self.run_event.set()
        self.set_phase("stopping")
        with self._lock:
            if self._process:
                self._process.terminate()
                self.control_error = self._process.last_control_error


AdapterFactory = Callable[[AdapterContext], ResearchAdapter]


class ResearchSupervisor:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        telemetry: TelemetryService,
        *,
        adapter_factories: dict[str, AdapterFactory] | None = None,
    ):
        self.settings = settings
        self.db = database
        self.telemetry = telemetry
        self.adapter_factories = adapter_factories or {
            "karpathy_autoresearch": KarpathyAutoresearchAdapter
        }
        self._runtimes: dict[str, ResearchRuntime] = {}
        self._gpu_locks: dict[str, threading.Lock] = {}
        self._repository_lock = threading.Lock()
        self._lock = threading.RLock()

    @staticmethod
    def terminate_stale_process_groups(
        settings: Settings, database: Database, researches: list[dict[str, Any]]
    ) -> None:
        for research in researches:
            source = database.get_gpu_source(research["gpu_source_id"])
            if source is None:
                continue
            research_id = str(research["id"])
            pid_files = [
                f"/tmp/autoresearch-{research_id}.pid",
                f"/tmp/autoresearch-setup-{research_id}.pid",
            ]
            files = " ".join(shlex.quote(path) for path in pid_files)
            marker = shlex.quote(f"AUTORESEARCH_RUN_ID={research_id}")
            script = (
                f"for f in {files}; do "
                'if [ -f "$f" ]; then p=$(cat "$f"); '
                'case "$p" in \'\'|*[!0-9]*) rm -f "$f"; continue;; esac; '
                f"if [ -r \"/proc/$p/environ\" ] && tr '\\0' '\\n' < \"/proc/$p/environ\" | grep -Fqx -- {marker}; then "
                '/bin/kill -TERM -- -"$p" 2>/dev/null || true; sleep 1; '
                '/bin/kill -0 -- -"$p" 2>/dev/null && /bin/kill -KILL -- -"$p" 2>/dev/null || true; '
                'fi; rm -f "$f"; fi; done'
            )
            try:
                if source["type"] == "remote":
                    result = SSHCommandRunner(source).run(
                        "sh -s", timeout=20, stdin_text=script
                    )
                elif settings.wsl_distro and os.name == "nt":
                    result = subprocess.run(
                        [
                            "wsl.exe",
                            "-d",
                            settings.wsl_distro,
                            "--",
                            "sh",
                            "-s",
                        ],
                        input=script,
                        capture_output=True,
                        text=True,
                        timeout=20,
                        check=False,
                    )
                else:
                    continue
                if result.returncode != 0:
                    raise RuntimeError((result.stderr or result.stdout).strip())
                database.add_log(
                    research_id,
                    "stale_process_cleanup",
                    "Startup checked and terminated any matching stale WSL/SSH process group.",
                    level="warning",
                )
            except Exception as exc:  # noqa: BLE001 - startup recovery must continue and report uncertainty
                database.add_log(
                    research_id,
                    "stale_process_warning",
                    f"Could not confirm stale WSL/SSH process-group cleanup: {str(exc)[:500]}",
                    level="error",
                )

    def _gpu_lock(self, source_id: str) -> threading.Lock:
        with self._lock:
            return self._gpu_locks.setdefault(source_id, threading.Lock())

    @staticmethod
    def _physical_gpu_key(source: dict[str, Any]) -> str:
        device = os.getenv("AUTORESEARCH_CUDA_VISIBLE_DEVICES", "0")
        if source["type"] == "local":
            return f"local:{device}"
        return f"remote:{source.get('host')}:{device}"

    def _launch(self, research: dict[str, Any]) -> ResearchRuntime:
        source = self.db.get_gpu_source(research["gpu_source_id"])
        if source is None:
            raise ValueError("GPU source no longer exists")
        factory = self.adapter_factories.get(research["adapter_type"])
        if factory is None:
            raise ValueError(f"Unknown research adapter: {research['adapter_type']}")
        runtime = ResearchRuntime(research["id"])
        context = AdapterContext(
            settings=self.settings,
            database=self.db,
            telemetry=self.telemetry,
            research=research,
            gpu_source=source,
            runtime=runtime,
            gpu_lock=(
                threading.Lock()
                if self.settings.use_cuda_mps
                else self._gpu_lock(self._physical_gpu_key(source))
            ),
            repository_lock=self._repository_lock,
        )
        adapter = factory(context)
        thread = threading.Thread(
            target=self._worker,
            args=(research["id"], runtime, adapter),
            name=f"research-{research['id'][:8]}",
            daemon=True,
        )
        runtime.thread = thread
        with self._lock:
            if research["id"] in self._runtimes:
                raise RuntimeError("Research already has an active supervisor")
            self._runtimes[research["id"]] = runtime
        thread.start()
        return runtime

    def _worker(
        self, research_id: str, runtime: ResearchRuntime, adapter: ResearchAdapter
    ) -> None:
        try:
            runtime.set_phase("preparing")
            adapter.prepare()
            runtime.set_phase("ready")
            self.db.add_log(
                research_id, "research_ready", "Research workspace is ready."
            )
            while not runtime.stop_event.is_set():
                if not runtime.wait_until_running():
                    break
                try:
                    adapter.run_iteration()
                except InterruptedError:
                    break
                except Exception as exc:  # noqa: BLE001 - an iteration failure must not end autonomous research
                    self.db.add_log(
                        research_id,
                        "iteration_error",
                        f"Iteration error; the last accepted workspace remains active: {exc}",
                        level="error",
                    )
                    runtime.stop_event.wait(2)
        except InterruptedError:
            pass
        except Exception as exc:  # noqa: BLE001 - adapter startup failures are persisted as failed state
            if not runtime.stop_event.is_set():
                self.db.update_research(research_id, {"status": "failed"})
                self.db.add_log(research_id, "research_failed", str(exc), level="error")
        finally:
            runtime.set_phase("detached")
            with self._lock:
                self._runtimes.pop(research_id, None)

    def start(self, research_id: str) -> dict[str, Any]:
        if not self.settings.execution_enabled:
            raise PermissionError(
                "Real execution is disabled. Set AUTORESEARCH_ENABLE_EXECUTION=1 and restart the backend."
            )
        research = self.db.get_research(research_id, detail=True)
        if research is None:
            raise KeyError(research_id)
        with self._lock:
            existing = self._runtimes.get(research_id)
        if existing:
            if research["status"] == "paused":
                return self.resume(research_id)
            raise RuntimeError("Research is already running")
        if research["status"] not in {"stopped", "paused", "failed"}:
            raise RuntimeError(f"Research cannot start from {research['status']}")
        if not self.db.transition_research(
            research_id, [research["status"]], "running"
        ):
            raise RuntimeError("Research state changed concurrently")
        updated = self.db.get_research(research_id, detail=True) or research
        self.db.add_log(
            research_id,
            "research_started",
            "Research started on a real execution worker.",
        )
        try:
            self._launch(updated)
        except Exception:
            self.db.update_research(research_id, {"status": "failed"})
            raise
        return self.db.get_research(research_id, detail=True) or updated

    def pause(self, research_id: str) -> dict[str, Any]:
        research = self.db.get_research(research_id)
        if research is None:
            raise KeyError(research_id)
        if research["status"] != "running":
            raise RuntimeError("Only a running research can be paused")
        with self._lock:
            runtime = self._runtimes.get(research_id)
        if runtime is None:
            raise RuntimeError("No live process supervisor is attached")
        runtime.pause()
        if not self.db.transition_research(research_id, ["running"], "paused"):
            runtime.resume()
            raise RuntimeError("Research state changed concurrently")
        self.db.add_log(
            research_id,
            "research_paused",
            "Research and its active subprocess were paused.",
        )
        return self.db.get_research(research_id, detail=True) or research

    def resume(self, research_id: str) -> dict[str, Any]:
        if not self.settings.execution_enabled:
            raise PermissionError(
                "Real execution is disabled. Set AUTORESEARCH_ENABLE_EXECUTION=1 and restart the backend."
            )
        research = self.db.get_research(research_id, detail=True)
        if research is None:
            raise KeyError(research_id)
        if research["status"] != "paused":
            raise RuntimeError("Only a paused research can be resumed")
        with self._lock:
            runtime = self._runtimes.get(research_id)
        if not self.db.transition_research(research_id, ["paused"], "running"):
            raise RuntimeError("Research state changed concurrently")
        if runtime is None:
            updated = self.db.get_research(research_id, detail=True) or research
            try:
                runtime = self._launch(updated)
            except Exception:
                self.db.update_research(research_id, {"status": "failed"})
                raise
        else:
            runtime.resume()
        self.db.add_log(
            research_id, "research_resumed", "Research and its subprocess resumed."
        )
        return self.db.get_research(research_id, detail=True) or research

    def stop(self, research_id: str) -> dict[str, Any]:
        research = self.db.get_research(research_id)
        if research is None:
            raise KeyError(research_id)
        if research["status"] not in {"running", "paused"}:
            raise RuntimeError("Only a running or paused research can be stopped")
        with self._lock:
            runtime = self._runtimes.get(research_id)
        if runtime:
            runtime.stop()
        target_status = "failed" if runtime and runtime.control_error else "stopped"
        if not self.db.transition_research(
            research_id, ["running", "paused"], target_status
        ):
            raise RuntimeError("Research state changed concurrently")
        if runtime and runtime.control_error:
            self.db.add_log(
                research_id,
                "research_control_error",
                f"Stop could not confirm remote process-group termination: {runtime.control_error}",
                level="error",
            )
        else:
            self.db.add_log(
                research_id, "research_stopped", "Research stopped by the user."
            )
        if runtime and runtime.thread:
            runtime.thread.join(timeout=10)
        result = self.db.get_research(research_id, detail=True) or research
        if runtime and runtime.control_error:
            raise RuntimeError(runtime.control_error)
        return result

    def shutdown(self) -> None:
        with self._lock:
            runtimes = list(self._runtimes.values())
        for runtime in runtimes:
            research = self.db.get_research(runtime.research_id)
            if research and research["status"] in {"running", "paused"}:
                runtime.stop()
                shutdown_status = "failed" if runtime.control_error else "paused"
                self.db.update_research(
                    runtime.research_id, {"status": shutdown_status}
                )
                self.db.add_log(
                    runtime.research_id,
                    "research_control_error"
                    if runtime.control_error
                    else "service_shutdown",
                    (
                        f"Backend shutdown could not confirm process termination: {runtime.control_error}"
                        if runtime.control_error
                        else "Backend shutdown interrupted execution; research is paused for safe resume."
                    ),
                    level="error" if runtime.control_error else "warning",
                )
        for runtime in runtimes:
            if runtime.thread:
                runtime.thread.join(timeout=5)

    def runtime_snapshot(self, research_id: str) -> dict[str, Any]:
        with self._lock:
            runtime = self._runtimes.get(research_id)
        research = self.db.get_research(research_id)
        target = research.get("target_gpu_allocation") if research else None
        mps = self.settings.use_cuda_mps
        result = {
            "attached": runtime is not None,
            "thread_alive": bool(
                runtime and runtime.thread and runtime.thread.is_alive()
            ),
            "paused": bool(runtime and not runtime.run_event.is_set()),
            "phase": "detached",
            "phase_changed_at": None,
            "subprocess_attached": False,
            "allocation": {
                "target_percent": target,
                "mode": (
                    "cuda_mps_active_thread_percentage"
                    if mps
                    else "exclusive_serialized"
                ),
                "target_role": (
                    "cuda_mps_active_thread_percentage"
                    if mps
                    else "scheduling_metadata"
                ),
                "target_enforced": mps,
                "is_hard_utilization_target": False,
                "utilization_source": "nvidia-smi_device_sample",
            },
        }
        if runtime:
            result.update(runtime.snapshot())
        return result
