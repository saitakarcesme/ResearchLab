from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import psutil

MANAGED_RUN_ID_ENV = "AUTORESEARCH_RUN_ID"


class RuntimeControl(Protocol):
    stop_event: threading.Event

    def set_process(self, process: ManagedProcess | None) -> None: ...


@dataclass(frozen=True, slots=True)
class ProcessResult:
    returncode: int
    timed_out: bool
    stopped: bool
    duration_seconds: float
    active_duration_seconds: float
    control_error: str | None = None


class ManagedProcess:
    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        pause_callback: Callable[[], None] | None = None,
        resume_callback: Callable[[], None] | None = None,
        terminate_callback: Callable[[], None] | None = None,
    ):
        self.process = process
        self.pause_callback = pause_callback
        self.resume_callback = resume_callback
        self.terminate_callback = terminate_callback
        self._paused = False
        self._lock = threading.RLock()
        self.last_control_error: str | None = None

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def paused(self) -> bool:
        return self._paused

    def _tree(self) -> tuple[psutil.Process, list[psutil.Process]]:
        parent = psutil.Process(self.process.pid)
        return parent, parent.children(recursive=True)

    def pause(self) -> None:
        with self._lock:
            if self.process.poll() is not None or self._paused:
                return
            if self.pause_callback:
                self.pause_callback()
            elif os.name == "posix":
                os.killpg(os.getpgid(self.process.pid), signal.SIGSTOP)
            else:
                parent, children = self._tree()
                for child in reversed(children):
                    child.suspend()
                parent.suspend()
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            if self.process.poll() is not None or not self._paused:
                return
            if self.resume_callback:
                self.resume_callback()
            elif os.name == "posix":
                os.killpg(os.getpgid(self.process.pid), signal.SIGCONT)
            else:
                parent, children = self._tree()
                parent.resume()
                for child in children:
                    child.resume()
            self._paused = False

    def terminate(self, grace_seconds: float = 8.0) -> None:
        with self._lock:
            if self.process.poll() is not None:
                return
            if self._paused:
                try:
                    self.resume()
                except (OSError, psutil.Error):
                    pass
            if self.terminate_callback:
                try:
                    self.terminate_callback()
                except Exception as exc:  # noqa: BLE001 - local tree termination remains the fallback
                    self.last_control_error = str(exc)
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                else:
                    parent, children = self._tree()
                    for child in children:
                        child.terminate()
                    parent.terminate()
            except (OSError, psutil.Error):
                self.process.terminate()
        try:
            self.process.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            else:
                parent, children = self._tree()
                for child in children:
                    child.kill()
                parent.kill()
        except (OSError, psutil.Error):
            try:
                self.process.kill()
            except OSError:
                pass


def run_managed_process(
    command: Sequence[str],
    *,
    cwd: Path | str | None,
    env: Mapping[str, str] | None,
    output: Path | str,
    timeout_seconds: int,
    runtime: RuntimeControl,
    stdin_text: str | None = None,
    pause_callback: Callable[[], None] | None = None,
    resume_callback: Callable[[], None] | None = None,
    terminate_callback: Callable[[], None] | None = None,
) -> ProcessResult:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_tick = started
    active_seconds = 0.0
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process_env = dict(os.environ if env is None else env)
    research_id = getattr(runtime, "research_id", None)
    if research_id:
        process_env[MANAGED_RUN_ID_ENV] = str(research_id)
    with output_path.open("wb") as sink:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd) if cwd else None,
            env=process_env,
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=sink,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
            creationflags=creationflags,
        )
        managed = ManagedProcess(
            process,
            pause_callback=pause_callback,
            resume_callback=resume_callback,
            terminate_callback=terminate_callback,
        )
        try:
            runtime.set_process(managed)
        except Exception:
            managed.terminate()
            runtime.set_process(None)
            raise
        if stdin_text is not None and process.stdin:
            process.stdin.write(stdin_text.encode("utf-8"))
            process.stdin.close()
        timed_out = False
        stopped = False
        try:
            while process.poll() is None:
                if runtime.stop_event.wait(0.2):
                    stopped = True
                    managed.terminate()
                    break
                now = time.monotonic()
                if not managed.paused:
                    active_seconds += now - last_tick
                last_tick = now
                if active_seconds >= timeout_seconds:
                    timed_out = True
                    managed.terminate()
                    break
            now = time.monotonic()
            if not managed.paused:
                active_seconds += now - last_tick
            returncode = process.wait()
        finally:
            runtime.set_process(None)
    return ProcessResult(
        returncode=returncode,
        timed_out=timed_out,
        stopped=stopped,
        duration_seconds=time.monotonic() - started,
        active_duration_seconds=active_seconds,
        control_error=managed.last_control_error,
    )
