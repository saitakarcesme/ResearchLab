from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import stat
import subprocess
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import psutil

from backend.adapters import (
    AdapterContext,
    HuggingFaceBeyefendiBenchmarkAdapter,
    KarpathyAutoresearchAdapter,
    OllamaBenchmarkAdapter,
    ResearchAdapter,
    ResearchComplete,
    ResearchFailed,
)
from backend.config import Settings
from backend.db import Database, utc_now
from backend.process_control import MANAGED_RUN_ID_ENV, ManagedProcess
from backend.telemetry import SSHCommandRunner, TelemetryService

_POSIX_SAFE_DELETE_PROGRAM = r"""
import os
import shutil
import stat
import sys
import uuid


def open_directory_chain(root):
    normalized = os.path.normpath(root)
    if not normalized.startswith("/") or normalized == "/" or normalized != root:
        raise ValueError("unsafe root")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptors = [os.open("/", flags)]
    try:
        for component in normalized.split("/")[1:]:
            descriptors.append(os.open(component, flags, dir_fd=descriptors[-1]))
    except Exception:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise
    return descriptors


def main():
    root, child_name = sys.argv[1:]
    if (
        not child_name
        or child_name in {".", ".."}
        or "/" in child_name
        or "\x00" in child_name
        or not shutil.rmtree.avoids_symlink_attacks
    ):
        return 76
    try:
        descriptors = open_directory_chain(root)
    except FileNotFoundError:
        return 0
    except (OSError, ValueError):
        return 77
    root_fd = descriptors[-1]
    try:
        try:
            original = os.stat(child_name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            return 0
        if not stat.S_ISDIR(original.st_mode):
            return 78
        quarantine = f".autoresearch-delete-{child_name}-{uuid.uuid4().hex}"
        os.rename(child_name, quarantine, src_dir_fd=root_fd, dst_dir_fd=root_fd)
        moved = os.stat(quarantine, dir_fd=root_fd, follow_symlinks=False)
        if (moved.st_dev, moved.st_ino) != (original.st_dev, original.st_ino):
            return 79
        shutil.rmtree(quarantine, dir_fd=root_fd)
        try:
            os.stat(quarantine, dir_fd=root_fd, follow_symlinks=False)
            return 80
        except FileNotFoundError:
            pass
        try:
            os.stat(child_name, dir_fd=root_fd, follow_symlinks=False)
            return 81
        except FileNotFoundError:
            return 0
    except OSError:
        return 82
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


raise SystemExit(main())
""".strip()


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
            "karpathy_autoresearch": KarpathyAutoresearchAdapter,
            "huggingface_beyefendi_benchmark": HuggingFaceBeyefendiBenchmarkAdapter,
            "ollama_benchmark": OllamaBenchmarkAdapter,
        }
        self._runtimes: dict[str, ResearchRuntime] = {}
        self._gpu_locks: dict[str, threading.Lock] = {}
        self._repository_lock = threading.Lock()
        self._lock = threading.RLock()
        self._dispatch_lock = threading.RLock()
        self._queue_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._queue_thread = threading.Thread(
            target=self._queue_worker,
            name="research-queue",
            daemon=True,
        )
        self._queue_thread.start()
        self._queue_event.set()

    def notify_queue(self) -> None:
        """Wake the scheduler after a durable queue change or capacity release."""

        self._queue_event.set()

    def _queue_worker(self) -> None:
        while not self._shutdown_event.is_set():
            # Scheduled work must wake up when its daily window opens even when
            # no new queue event occurs.
            self._queue_event.wait(timeout=20)
            self._queue_event.clear()
            if self._shutdown_event.is_set():
                break
            if not self.settings.execution_enabled:
                continue
            while not self._shutdown_event.is_set():
                try:
                    started = self.start_next()
                except Exception:  # noqa: BLE001 - start() persists the individual failure
                    break
                if started is None:
                    break

    @staticmethod
    def _process_group_cleanup_script(research_id: str) -> str:
        pid_files = [
            f"/tmp/autoresearch-{research_id}.pid",
            f"/tmp/autoresearch-setup-{research_id}.pid",
        ]
        files = " ".join(shlex.quote(path) for path in pid_files)
        marker = shlex.quote(f"AUTORESEARCH_RUN_ID={research_id}")
        return (
            "groups=''; "
            "add_group() { case \"$1\" in ''|*[!0-9]*) return;; esac; "
            "case \" $groups \" in *\" $1 \"*) ;; *) groups=\"$groups $1\";; esac; }; "
            f"for f in {files}; do "
            '[ -f "$f" ] || continue; p=$(cat "$f"); '
            'case "$p" in \'\'|*[!0-9]*) continue;; esac; '
            f"if [ -r \"/proc/$p/environ\" ] && tr '\\0' '\\n' < \"/proc/$p/environ\" | grep -Fqx -- {marker}; then "
            'g=$(ps -o pgid= -p "$p") || true; set -- $g; add_group "${1:-}"; fi; done; '
            "for e in /proc/[0-9]*/environ; do [ -r \"$e\" ] || continue; "
            f"if tr '\\0' '\\n' < \"$e\" | grep -Fqx -- {marker}; then "
            'p=${e#/proc/}; p=${p%/environ}; g=$(ps -o pgid= -p "$p") || true; '
            'set -- $g; add_group "${1:-}"; fi; done; '
            'for p in $groups; do /bin/kill -TERM -- -"$p" 2>/dev/null || true; done; '
            '[ -z "$groups" ] || sleep 1; '
            'for p in $groups; do /bin/kill -0 -- -"$p" 2>/dev/null && '
            '/bin/kill -KILL -- -"$p" 2>/dev/null || true; done; '
            '[ -z "$groups" ] || sleep 0.2; '
            'for p in $groups; do /bin/kill -0 -- -"$p" 2>/dev/null && exit 76; done; '
            f"rm -f -- {files}; exit 0"
        )

    @staticmethod
    def _wsl_cleanup_distros(
        settings: Settings, research: dict[str, Any]
    ) -> list[str]:
        explicit_hf = os.getenv("AUTORESEARCH_HF_WSL_DISTRO", "").strip()
        if research.get("adapter_type") == "huggingface_beyefendi_benchmark":
            candidates = (
                [explicit_hf]
                if explicit_hf
                else ["Ubuntu", str(settings.wsl_distro or "")]
            )
        else:
            candidates = [str(settings.wsl_distro or "")]
        candidates = list(dict.fromkeys(item for item in candidates if item))
        if not candidates:
            return []
        result = subprocess.run(
            ["wsl.exe", "-l", "-q"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("Could not list WSL distributions for process cleanup")
        available = {
            line.strip().lstrip("\ufeff")
            for line in result.stdout.decode("utf-16-le", errors="replace").splitlines()
            if line.strip()
        }
        return [item for item in candidates if item in available]

    @staticmethod
    def _terminate_native_process_trees(research_id: str) -> None:
        """Terminate same-user Windows processes carrying one exact run marker."""

        matches: dict[int, psutil.Process] = {}
        for process in psutil.process_iter(["pid", "ppid"]):
            if process.pid == os.getpid():
                continue
            try:
                if process.environ().get(MANAGED_RUN_ID_ENV) == research_id:
                    matches[process.pid] = process
            except (OSError, psutil.Error):
                continue
        roots = [
            process
            for process in matches.values()
            if process.ppid() not in matches
        ]
        targets: dict[int, psutil.Process] = {}
        for root in roots:
            try:
                descendants = root.children(recursive=True)
            except psutil.Error:
                descendants = []
            for process in [*reversed(descendants), root]:
                targets[process.pid] = process
        for process in targets.values():
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(list(targets.values()), timeout=3)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(alive, timeout=3)
        if alive:
            raise RuntimeError(
                "Could not confirm native managed process-tree termination"
            )

    @classmethod
    def _terminate_research_process_groups(
        cls,
        settings: Settings,
        source: dict[str, Any],
        research: dict[str, Any],
    ) -> None:
        script = cls._process_group_cleanup_script(str(research["id"]))
        if os.name == "nt":
            cls._terminate_native_process_trees(str(research["id"]))
        if source["type"] == "remote":
            results = [
                SSHCommandRunner(source).run("sh -s", timeout=25, stdin_text=script)
            ]
        elif os.name == "nt":
            distros = cls._wsl_cleanup_distros(settings, research)
            if research.get("adapter_type") == "huggingface_beyefendi_benchmark" and not distros:
                raise RuntimeError(
                    "Could not find the Hugging Face benchmark's WSL distribution"
                )
            results = [
                subprocess.run(
                    ["wsl.exe", "-d", distro, "--", "sh", "-s"],
                    input=script,
                    capture_output=True,
                    text=True,
                    timeout=25,
                    check=False,
                )
                for distro in distros
            ]
        elif os.name == "posix":
            results = [
                subprocess.run(
                    ["sh", "-s"],
                    input=script,
                    capture_output=True,
                    text=True,
                    timeout=25,
                    check=False,
                )
            ]
        else:
            results = []
        failures = [
            (result.stderr or result.stdout).strip()
            for result in results
            if result.returncode != 0
        ]
        if failures:
            raise RuntimeError(
                "Could not confirm managed process-group termination: "
                + "; ".join(item or "cleanup command failed" for item in failures)
            )

    @classmethod
    def terminate_stale_process_groups(
        cls, settings: Settings, database: Database, researches: list[dict[str, Any]]
    ) -> None:
        for research in researches:
            source = database.get_gpu_source(research["gpu_source_id"])
            if source is None:
                continue
            research_id = str(research["id"])
            try:
                cls._terminate_research_process_groups(settings, source, research)
                database.add_log(
                    research_id,
                    "stale_process_cleanup",
                    "Startup confirmed cleanup in every runtime that could own this research.",
                    level="warning",
                )
            except Exception as exc:  # noqa: BLE001 - startup recovery must continue and report uncertainty
                current = database.get_research(research_id)
                if current and current["status"] == "failed":
                    database.update_research(research_id, {"status": "paused"})
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

    def _has_gpu_capacity(self, candidate: dict[str, Any]) -> bool:
        source = self.db.get_gpu_source(candidate["gpu_source_id"])
        if source is None:
            return False
        candidate_key = self._physical_gpu_key(source)
        allocations = 0
        # A paused research keeps its place on the GPU, including across a service
        # restart. Looking only at attached runtimes would let a queued job jump
        # ahead during the short window before the paused research is resumed.
        for active in self.db.list_gpu_reservations():
            active_source = self.db.get_gpu_source(active["gpu_source_id"])
            if (
                active_source is None
                or self._physical_gpu_key(active_source) != candidate_key
            ):
                continue
            if not self.settings.use_cuda_mps:
                return False
            allocations += int(active["target_gpu_allocation"])
        if not self.settings.use_cuda_mps:
            return True
        return allocations + int(candidate["target_gpu_allocation"]) <= 100

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
        try:
            thread.start()
        except Exception:
            with self._lock:
                self._runtimes.pop(research["id"], None)
            raise
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
                except (ResearchComplete, ResearchFailed):
                    raise
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
        except ResearchComplete as complete:
            if self.db.transition_research(research_id, ["running"], "completed"):
                self.db.add_log(
                    research_id,
                    "research_completed",
                    complete.message,
                )
        except ResearchFailed as failed:
            target_status = "paused" if runtime.control_error else "failed"
            if self.db.transition_research(
                research_id, ["running"], target_status
            ):
                self.db.add_log(
                    research_id,
                    "research_control_error"
                    if runtime.control_error
                    else "research_failed",
                    failed.message,
                    level="error",
                )
        except InterruptedError:
            pass
        except Exception as exc:  # noqa: BLE001 - adapter startup failures are persisted as failed state
            if not runtime.stop_event.is_set():
                target_status = "paused" if runtime.control_error else "failed"
                self.db.update_research(research_id, {"status": target_status})
                self.db.add_log(
                    research_id,
                    "research_control_error"
                    if runtime.control_error
                    else "research_failed",
                    (
                        "Process termination could not be confirmed; the GPU remains "
                        f"reserved until cleanup succeeds: {runtime.control_error}"
                        if runtime.control_error
                        else str(exc)
                    ),
                    level="error",
                )
        finally:
            runtime.set_phase("detached")
            with self._lock:
                self._runtimes.pop(research_id, None)
            self.notify_queue()

    def start(self, research_id: str) -> dict[str, Any]:
        if not self.settings.execution_enabled:
            raise PermissionError(
                "Real execution is disabled. Set AUTORESEARCH_ENABLE_EXECUTION=1 and restart the backend."
            )
        with self._dispatch_lock:
            research = self.db.get_research(research_id, detail=True)
            if research is None:
                raise KeyError(research_id)
            with self._lock:
                existing = self._runtimes.get(research_id)
            if existing:
                if research["status"] == "paused":
                    return self.resume(research_id)
                raise RuntimeError("Research is already running")
            if research["status"] not in {"queued", "stopped", "paused", "failed"}:
                raise RuntimeError(f"Research cannot start from {research['status']}")
            previous_status = research["status"]
            if not self.db.transition_research(
                research_id, [previous_status], "running"
            ):
                raise RuntimeError("Research state changed concurrently")
            updated = self.db.get_research(research_id, detail=True) or research
            self.db.add_log(
                research_id,
                "research_started",
                (
                    "Queued research started after its GPU source became available."
                    if previous_status == "queued"
                    else "Research started on a real execution worker."
                ),
                data={"from_queue": previous_status == "queued"},
            )
            try:
                self._launch(updated)
            except Exception as exc:
                self.db.update_research(research_id, {"status": "failed"})
                self.db.add_log(
                    research_id,
                    "research_start_failed",
                    f"Research worker could not start: {exc}",
                    level="error",
                )
                self.notify_queue()
                raise
            return self.db.get_research(research_id, detail=True) or updated

    def start_next(self, gpu_source_id: str | None = None) -> dict[str, Any] | None:
        """Start the oldest queued research whose GPU has available capacity."""

        if not self.settings.execution_enabled:
            raise PermissionError(
                "Real execution is disabled. Set AUTORESEARCH_ENABLE_EXECUTION=1 and restart the backend."
            )
        with self._dispatch_lock:
            for research in self.db.list_queued_researches(gpu_source_id):
                if not self._schedule_is_open(research):
                    continue
                if self._has_gpu_capacity(research):
                    return self.start(research["id"])
        return None

    @staticmethod
    def _schedule_is_open(
        research: dict[str, Any], now: datetime | None = None
    ) -> bool:
        start_value = research.get("schedule_start_time")
        end_value = research.get("schedule_end_time")
        timezone_name = research.get("schedule_timezone")
        if not start_value or not end_value or not timezone_name:
            return True
        try:
            zone = ZoneInfo(str(timezone_name))
        except ZoneInfoNotFoundError:
            offset = research.get("schedule_utc_offset_minutes")
            if offset is None:
                return False
            zone = timezone(timedelta(minutes=int(offset)))
        local_now = (now or datetime.now(tz=zone)).astimezone(zone)
        current = local_now.strftime("%H:%M")
        start, end = str(start_value), str(end_value)
        if start < end:
            return start <= current < end
        return current >= start or current < end

    def enqueue(self, research_id: str) -> dict[str, Any]:
        with self._dispatch_lock:
            research = self.db.get_research(research_id, detail=True)
            if research is None:
                raise KeyError(research_id)
            if research["status"] == "queued":
                return research
            if research["status"] not in {"stopped", "failed"}:
                raise RuntimeError("Only a stopped or failed research can be queued")
            if not self.db.transition_research(
                research_id, [research["status"]], "queued"
            ):
                raise RuntimeError("Research state changed concurrently")
            self.db.add_log(
                research_id,
                "research_queued",
                "Research added to the GPU queue.",
            )
            result = self.db.get_research(research_id, detail=True) or research
        self.notify_queue()
        return result

    def dequeue(self, research_id: str) -> dict[str, Any]:
        with self._dispatch_lock:
            research = self.db.get_research(research_id, detail=True)
            if research is None:
                raise KeyError(research_id)
            if research["status"] != "queued":
                raise RuntimeError("Only a queued research can be removed from the queue")
            if not self.db.transition_research(research_id, ["queued"], "stopped"):
                raise RuntimeError("Research state changed concurrently")
            self.db.add_log(
                research_id,
                "research_dequeued",
                "Research removed from the GPU queue.",
            )
            return self.db.get_research(research_id, detail=True) or research

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
        if runtime is None:
            source = self.db.get_gpu_source(research["gpu_source_id"])
            if source is None:
                raise RuntimeError("GPU source no longer exists")
            self._terminate_research_process_groups(
                self.settings, source, research
            )
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
        if research["status"] == "queued":
            return self.dequeue(research_id)
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
        self.notify_queue()
        result = self.db.get_research(research_id, detail=True) or research
        if runtime and runtime.control_error:
            raise RuntimeError(runtime.control_error)
        return result

    def delete(self, research_id: str) -> None:
        """Stop any live worker, delete its durable data, then clean its workspace.

        Holding the dispatch lock closes the race with the queue worker: a queued
        research cannot be selected between the initial read and the database
        deletion. Filesystem cleanup happens before the cascading database
        transaction so a locked managed directory cannot be hidden by deleting
        its durable owner first.
        """

        workspace_path: str | None = None
        with self._dispatch_lock:
            research = self.db.get_research(research_id, detail=True)
            if research is None:
                raise KeyError(research_id)
            workspace_path = research.get("workspace_path")
            source = self.db.get_gpu_source(research["gpu_source_id"])
            if source is None:
                raise RuntimeError("GPU source no longer exists")

            with self._lock:
                runtime = self._runtimes.get(research_id)

            if runtime is None and research["status"] != "queued":
                self._terminate_research_process_groups(
                    self.settings, source, research
                )

            if research["status"] in {"running", "paused"}:
                try:
                    self.stop(research_id)
                except RuntimeError:
                    # Completion can win the race with a delete request between
                    # get_research() and stop().  Proceed only if the durable state
                    # proves the job is no longer active.
                    if runtime and runtime.control_error:
                        raise
                    current = self.db.get_research(research_id)
                    if current is None or current["status"] in {"running", "paused"}:
                        raise

            if runtime and runtime.thread and runtime.thread.is_alive():
                runtime.thread.join(timeout=10)
                if runtime.thread.is_alive():
                    raise RuntimeError(
                        "Research worker did not detach after its process was stopped"
                    )

            workspace_removed = self._remove_managed_workspace(
                research_id, workspace_path
            )
            if not workspace_removed:
                raise RuntimeError(
                    "The managed research workspace could not be removed safely"
                )
            if not self._remove_local_research_artifacts(research_id):
                raise RuntimeError(
                    "Managed research runtime artifacts could not be removed safely"
                )
            if not self._remove_external_research_artifacts(research, source):
                raise RuntimeError(
                    "Managed WSL or remote research artifacts could not be removed safely"
                )
            if not self._remove_generated_article_logs(research_id):
                raise RuntimeError(
                    "Generated article logs could not be removed safely"
                )
            logs_removed = self._remove_managed_directory(
                self.settings.log_dir,
                research_id,
                self.settings.log_dir / research_id,
            )
            if not logs_removed:
                raise RuntimeError(
                    "The managed research logs could not be removed safely"
                )

            if not self.db.delete_research(research_id):
                current = self.db.get_research(research_id)
                if current is None:
                    raise KeyError(research_id)
                raise RuntimeError(
                    "Research is still active and could not be deleted safely"
                )
        self.notify_queue()

    def _remove_managed_workspace(
        self, research_id: str, workspace_path: str | None
    ) -> bool:
        """Remove only the exact workspace allocated for this research id.

        A corrupted or user-edited database path must never turn a research delete
        into an arbitrary recursive filesystem delete.  Resolving both paths also
        rejects workspace symlinks/junctions that escape the managed root.
        """

        if not workspace_path:
            return True
        try:
            root = self.settings.workspace_dir.resolve(strict=False)
            expected = (root / research_id).resolve(strict=False)
            candidate = Path(workspace_path).resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            return True
        # An external/corrupted path is not owned by ResearchLab. Preserve it and
        # remove only the durable reference to it.
        if candidate != expected:
            return True
        return self._remove_managed_directory(
            self.settings.workspace_dir, research_id, candidate
        )

    def _remove_local_research_artifacts(self, research_id: str) -> bool:
        sandbox_root = self.settings.data_dir / "agent-sandboxes"
        if sandbox_root.is_symlink():
            return False
        direct_candidates = (
            (
                sandbox_root,
                sandbox_root / research_id,
            ),
            (self.settings.runtime_dir, self.settings.runtime_dir / research_id),
        )
        for root, candidate in direct_candidates:
            if not self._remove_managed_directory(root, research_id, candidate):
                return False

        try:
            runtime_root = self.settings.runtime_dir.resolve(strict=False)
            source_roots = list(self.settings.runtime_dir.iterdir())
        except FileNotFoundError:
            return True
        except OSError:
            return False
        for source_root in source_roots:
            try:
                if source_root.is_symlink() or not source_root.is_dir():
                    continue
                resolved_source = source_root.resolve(strict=False)
                resolved_source.relative_to(runtime_root)
            except (OSError, RuntimeError, ValueError):
                continue
            if not self._remove_managed_directory(
                resolved_source,
                research_id,
                resolved_source / research_id,
            ):
                return False
        return True

    def _remove_generated_article_logs(self, research_id: str) -> bool:
        article_root = self.settings.log_dir / "articles"
        try:
            lexical_log_root = Path(os.path.abspath(self.settings.log_dir))
            lexical_article_root = Path(os.path.abspath(article_root))
            lexical_article_root.relative_to(lexical_log_root)
        except (OSError, ValueError):
            return False
        return self._remove_managed_files_by_prefix(
            lexical_article_root, f"{research_id}-"
        )

    @staticmethod
    def _safe_posix_child(root: PurePosixPath, child_name: str) -> PurePosixPath:
        if (
            not root.is_absolute()
            or root == PurePosixPath("/")
            or ".." in root.parts
            or not child_name
            or child_name in {".", ".."}
            or "/" in child_name
            or "\x00" in child_name
        ):
            raise ValueError("Managed POSIX artifact path is not absolute and bounded")
        child = root / child_name
        if child.parent != root:
            raise ValueError("Managed POSIX artifact escaped its root")
        return child

    @classmethod
    def _posix_remove_child_script(
        cls, root: PurePosixPath, child_name: str
    ) -> str:
        cls._safe_posix_child(root, child_name)
        root_value = shlex.quote(str(root))
        child_value = shlex.quote(child_name)
        return (
            "command -v python3 >/dev/null 2>&1 || exit 75; "
            f"python3 - {root_value} {child_value} <<'PY'\n"
            f"{_POSIX_SAFE_DELETE_PROGRAM}\n"
            "PY"
        )

    def _remove_external_research_artifacts(
        self, research: dict[str, Any], source: dict[str, Any]
    ) -> bool:
        if (
            research.get("adapter_type") != "karpathy_autoresearch"
            or not research.get("workspace_path")
        ):
            return True
        research_id = str(research["id"])
        source_key = hashlib.sha256(
            str(source["id"]).encode("utf-8")
        ).hexdigest()[:16]
        commands: list[tuple[str, str]] = []
        try:
            if source["type"] == "remote":
                workspace_root = PurePosixPath(str(source.get("workspace_path") or ""))
                commands = [
                    (
                        "remote workspace",
                        self._posix_remove_child_script(workspace_root, research_id),
                    ),
                    (
                        "remote runtime",
                        self._posix_remove_child_script(
                            workspace_root / ".autoresearch-runtimes" / source_key,
                            research_id,
                        ),
                    ),
                ]
                runner = SSHCommandRunner(source)
                results = [
                    runner.run("sh -s", timeout=120, stdin_text=script)
                    for _, script in commands
                ]
            elif os.name == "nt" and self.settings.wsl_distro:
                configured = os.getenv(
                    "AUTORESEARCH_WSL_RUNTIME_ROOT", ""
                ).strip()
                if configured:
                    runtime_root = PurePosixPath(configured)
                else:
                    home_result = subprocess.run(
                        [
                            "wsl.exe",
                            "-d",
                            self.settings.wsl_distro,
                            "--",
                            "sh",
                            "-s",
                        ],
                        input='printf %s "$HOME"',
                        capture_output=True,
                        text=True,
                        timeout=15,
                        check=False,
                    )
                    home = home_result.stdout.strip()
                    if home_result.returncode != 0 or not home.startswith("/"):
                        return False
                    runtime_root = (
                        PurePosixPath(home)
                        / ".cache"
                        / "autoresearch-lab"
                        / "runtimes"
                    )
                script = self._posix_remove_child_script(
                    runtime_root / source_key, research_id
                )
                results = [
                    subprocess.run(
                        [
                            "wsl.exe",
                            "-d",
                            self.settings.wsl_distro,
                            "--",
                            "sh",
                            "-s",
                        ],
                        input=script,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                    )
                ]
            else:
                return True
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
            return False
        return all(result.returncode == 0 for result in results)

    @staticmethod
    def _windows_extended_path(path: Path) -> str:
        value = str(path)
        if value.startswith("\\\\?\\"):
            return value
        if value.startswith("\\\\"):
            return "\\\\?\\UNC\\" + value.lstrip("\\")
        return "\\\\?\\" + value

    @staticmethod
    def _remove_posix_child(root: Path, child_name: str) -> bool:
        """Delete one child through held dirfds without following symlinks."""

        if not shutil.rmtree.avoids_symlink_attacks:
            return False
        root = Path(os.path.abspath(root))
        try:
            ResearchSupervisor._safe_posix_child(
                PurePosixPath(root.as_posix()), child_name
            )
        except ValueError:
            return False
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        descriptors: list[int] = []
        try:
            descriptors.append(os.open("/", flags))
            for component in root.parts[1:]:
                descriptors.append(
                    os.open(component, flags, dir_fd=descriptors[-1])
                )
            root_fd = descriptors[-1]
            try:
                original = os.stat(
                    child_name, dir_fd=root_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                return True
            if not stat.S_ISDIR(original.st_mode):
                return False
            quarantine = (
                f".autoresearch-delete-{child_name}-{uuid.uuid4().hex}"
            )
            os.rename(
                child_name,
                quarantine,
                src_dir_fd=root_fd,
                dst_dir_fd=root_fd,
            )
            moved = os.stat(
                quarantine, dir_fd=root_fd, follow_symlinks=False
            )
            if (moved.st_dev, moved.st_ino) != (
                original.st_dev,
                original.st_ino,
            ):
                return False
            shutil.rmtree(quarantine, dir_fd=root_fd)
            try:
                os.stat(quarantine, dir_fd=root_fd, follow_symlinks=False)
                return False
            except FileNotFoundError:
                pass
            try:
                os.stat(child_name, dir_fd=root_fd, follow_symlinks=False)
                return False
            except FileNotFoundError:
                return True
        except OSError:
            return False
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    @staticmethod
    def _windows_open_reparse_handle(
        path: Path | str, *, desired_access: int, share_mode: int
    ) -> tuple[int, int, tuple[int, int, int]]:
        import ctypes
        from ctypes import wintypes

        class FileAttributeTagInfo(ctypes.Structure):
            _fields_ = [
                ("file_attributes", wintypes.DWORD),
                ("reparse_tag", wintypes.DWORD),
            ]

        class ByHandleFileInformation(ctypes.Structure):
            _fields_ = [
                ("file_attributes", wintypes.DWORD),
                ("creation_time", wintypes.FILETIME),
                ("last_access_time", wintypes.FILETIME),
                ("last_write_time", wintypes.FILETIME),
                ("volume_serial_number", wintypes.DWORD),
                ("file_size_high", wintypes.DWORD),
                ("file_size_low", wintypes.DWORD),
                ("number_of_links", wintypes.DWORD),
                ("file_index_high", wintypes.DWORD),
                ("file_index_low", wintypes.DWORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        get_attribute_info = kernel32.GetFileInformationByHandleEx
        get_attribute_info.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        get_attribute_info.restype = wintypes.BOOL
        get_file_info = kernel32.GetFileInformationByHandle
        get_file_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ByHandleFileInformation),
        ]
        get_file_info.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        value = (
            str(path)
            if str(path).startswith("\\\\?\\")
            else ResearchSupervisor._windows_extended_path(Path(path))
        )
        handle = create_file(
            value,
            desired_access,
            share_mode,
            None,
            3,  # OPEN_EXISTING
            0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
            None,
        )
        invalid_handle = wintypes.HANDLE(-1).value
        if handle == invalid_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            attribute_info = FileAttributeTagInfo()
            if not get_attribute_info(
                handle,
                9,  # FileAttributeTagInfo
                ctypes.byref(attribute_info),
                ctypes.sizeof(attribute_info),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            file_info = ByHandleFileInformation()
            if not get_file_info(handle, ctypes.byref(file_info)):
                raise ctypes.WinError(ctypes.get_last_error())
            identity = (
                int(file_info.volume_serial_number),
                int(file_info.file_index_high),
                int(file_info.file_index_low),
            )
            return int(handle), int(attribute_info.file_attributes), identity
        except Exception:
            close_handle(handle)
            raise

    @staticmethod
    def _windows_close_handle(handle: int) -> None:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        close_handle(wintypes.HANDLE(handle))

    @staticmethod
    def _windows_mark_handle_for_delete(handle: int) -> None:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        set_information = kernel32.SetFileInformationByHandle
        set_information.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        set_information.restype = wintypes.BOOL
        flags = wintypes.DWORD(
            0x00000001 | 0x00000010
        )  # DELETE | IGNORE_READONLY_ATTRIBUTE
        if not set_information(
            wintypes.HANDLE(handle),
            21,  # FileDispositionInfoEx
            ctypes.byref(flags),
            ctypes.sizeof(flags),
        ):
            raise ctypes.WinError(ctypes.get_last_error())

    @classmethod
    def _windows_delete_entry(
        cls,
        path: Path | str,
        *,
        expected_identity: tuple[int, int, int] | None = None,
    ) -> bool:
        file_read_attributes = 0x00000080
        delete_access = 0x00010000
        share_read_write = 0x00000001 | 0x00000002
        file_attribute_directory = 0x00000010
        file_attribute_reparse_point = 0x00000400
        handle: int | None = None
        try:
            handle, attributes, identity = cls._windows_open_reparse_handle(
                path,
                desired_access=file_read_attributes | delete_access,
                share_mode=share_read_write,
            )
            if expected_identity is not None and identity != expected_identity:
                return False
            is_directory = bool(attributes & file_attribute_directory)
            is_reparse_point = bool(attributes & file_attribute_reparse_point)
            if is_directory and not is_reparse_point:
                path_value = (
                    str(path)
                    if str(path).startswith("\\\\?\\")
                    else cls._windows_extended_path(Path(path))
                )
                for _ in range(4):
                    entries = list(os.scandir(path_value))
                    if not entries:
                        break
                    for entry in entries:
                        if not cls._windows_delete_entry(entry.path):
                            return False
                if list(os.scandir(path_value)):
                    return False
            cls._windows_mark_handle_for_delete(handle)
        except FileNotFoundError:
            return True
        except OSError:
            return False
        finally:
            if handle is not None:
                cls._windows_close_handle(handle)
        return not os.path.lexists(str(path))

    @classmethod
    def _remove_windows_child(
        cls, root: Path, child_name: str
    ) -> bool:
        file_read_attributes = 0x00000080
        share_read_write = 0x00000001 | 0x00000002
        file_attribute_directory = 0x00000010
        file_attribute_reparse_point = 0x00000400
        guards: list[int] = []
        root = Path(os.path.abspath(root))
        candidate = root / child_name
        try:
            current = Path(root.anchor)
            prefixes = [current]
            for component in root.parts[1:]:
                current /= component
                prefixes.append(current)
            for prefix in prefixes:
                handle, attributes, _ = cls._windows_open_reparse_handle(
                    prefix,
                    desired_access=file_read_attributes,
                    share_mode=share_read_write,
                )
                guards.append(handle)
                if not attributes & file_attribute_directory:
                    return False
                if attributes & file_attribute_reparse_point:
                    return False
            try:
                original_handle, original_attributes, original_identity = (
                    cls._windows_open_reparse_handle(
                        candidate,
                        desired_access=file_read_attributes,
                        share_mode=(share_read_write | 0x00000004),
                    )
                )
            except FileNotFoundError:
                return True
            try:
                if not original_attributes & file_attribute_directory:
                    return False
                if original_attributes & file_attribute_reparse_point:
                    return False
            finally:
                cls._windows_close_handle(original_handle)
            quarantine = root / (
                f".autoresearch-delete-{child_name}-{uuid.uuid4().hex}"
            )
            os.rename(
                cls._windows_extended_path(candidate),
                cls._windows_extended_path(quarantine),
            )
            if not cls._windows_delete_entry(
                quarantine, expected_identity=original_identity
            ):
                return False
            return not os.path.lexists(cls._windows_extended_path(candidate))
        except FileNotFoundError:
            return True
        except OSError:
            return False
        finally:
            for handle in reversed(guards):
                cls._windows_close_handle(handle)

    @staticmethod
    def _remove_posix_files_by_prefix(root: Path, prefix: str) -> bool:
        if (
            not prefix
            or "/" in prefix
            or "\x00" in prefix
            or not shutil.rmtree.avoids_symlink_attacks
        ):
            return False
        root = Path(os.path.abspath(root))
        try:
            ResearchSupervisor._safe_posix_child(
                PurePosixPath(root.as_posix()), f"{prefix}probe"
            )
        except ValueError:
            return False
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        descriptors: list[int] = []
        try:
            descriptors.append(os.open("/", flags))
            for component in root.parts[1:]:
                descriptors.append(
                    os.open(component, flags, dir_fd=descriptors[-1])
                )
            root_fd = descriptors[-1]
            for name in sorted(os.listdir(root_fd)):
                if not name.startswith(prefix):
                    continue
                original = os.stat(
                    name, dir_fd=root_fd, follow_symlinks=False
                )
                if not stat.S_ISREG(original.st_mode):
                    return False
                quarantine = (
                    f".autoresearch-delete-{name}-{uuid.uuid4().hex}"
                )
                os.rename(
                    name,
                    quarantine,
                    src_dir_fd=root_fd,
                    dst_dir_fd=root_fd,
                )
                moved = os.stat(
                    quarantine, dir_fd=root_fd, follow_symlinks=False
                )
                if (moved.st_dev, moved.st_ino) != (
                    original.st_dev,
                    original.st_ino,
                ):
                    return False
                os.unlink(quarantine, dir_fd=root_fd)
                try:
                    os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                    return False
                except FileNotFoundError:
                    pass
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    @classmethod
    def _remove_windows_files_by_prefix(
        cls, root: Path, prefix: str
    ) -> bool:
        if not prefix or "/" in prefix or "\\" in prefix or "\x00" in prefix:
            return False
        file_read_attributes = 0x00000080
        share_read_write = 0x00000001 | 0x00000002
        share_delete = 0x00000004
        file_attribute_directory = 0x00000010
        file_attribute_reparse_point = 0x00000400
        guards: list[int] = []
        root = Path(os.path.abspath(root))
        try:
            current = Path(root.anchor)
            prefixes = [current]
            for component in root.parts[1:]:
                current /= component
                prefixes.append(current)
            for path_prefix in prefixes:
                handle, attributes, _ = cls._windows_open_reparse_handle(
                    path_prefix,
                    desired_access=file_read_attributes,
                    share_mode=share_read_write,
                )
                guards.append(handle)
                if not attributes & file_attribute_directory:
                    return False
                if attributes & file_attribute_reparse_point:
                    return False
            root_value = cls._windows_extended_path(root)
            for entry in list(os.scandir(root_value)):
                if not entry.name.startswith(prefix):
                    continue
                entry_handle: int | None = None
                try:
                    entry_handle, attributes, identity = (
                        cls._windows_open_reparse_handle(
                            entry.path,
                            desired_access=file_read_attributes,
                            share_mode=share_read_write | share_delete,
                        )
                    )
                    if attributes & file_attribute_directory:
                        return False
                    if attributes & file_attribute_reparse_point:
                        return False
                finally:
                    if entry_handle is not None:
                        cls._windows_close_handle(entry_handle)
                quarantine = root / (
                    f".autoresearch-delete-{entry.name}-{uuid.uuid4().hex}"
                )
                os.rename(entry.path, cls._windows_extended_path(quarantine))
                if not cls._windows_delete_entry(
                    quarantine, expected_identity=identity
                ):
                    return False
                if os.path.lexists(entry.path):
                    return False
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False
        finally:
            for handle in reversed(guards):
                cls._windows_close_handle(handle)

    @classmethod
    def _remove_managed_files_by_prefix(
        cls, root: Path, prefix: str
    ) -> bool:
        if os.name == "nt":
            return cls._remove_windows_files_by_prefix(root, prefix)
        return cls._remove_posix_files_by_prefix(root, prefix)

    @staticmethod
    def _remove_managed_directory(
        root_path: Path, expected_name: str, candidate_path: Path
    ) -> bool:
        """Recursively remove only one exact child of a configured managed root."""

        try:
            lexical_root = Path(os.path.abspath(root_path))
            lexical_expected = lexical_root / expected_name
            lexical_candidate = Path(os.path.abspath(candidate_path))
            if lexical_candidate != lexical_expected or candidate_path.is_symlink():
                return False
            root = root_path.resolve(strict=False)
            expected = root / expected_name
            candidate = candidate_path.resolve(strict=False)
            expected.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            return False
        if candidate != expected:
            return False
        if os.name == "nt":
            return ResearchSupervisor._remove_windows_child(
                lexical_root, expected_name
            )
        return ResearchSupervisor._remove_posix_child(
            lexical_root, expected_name
        )

    def shutdown(self) -> None:
        self._shutdown_event.set()
        self._queue_event.set()
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
        self._queue_thread.join(timeout=5)

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
