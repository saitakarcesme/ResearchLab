from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from backend.config import Settings
from backend.db import Database
from backend.telemetry import TelemetryService


class ResearchRuntime(Protocol):
    stop_event: threading.Event
    run_event: threading.Event

    def set_process(self, process: Any | None) -> None: ...
    def wait_until_running(self) -> bool: ...


@dataclass(slots=True)
class AdapterContext:
    settings: Settings
    database: Database
    telemetry: TelemetryService
    research: Mapping[str, Any]
    gpu_source: Mapping[str, Any]
    runtime: ResearchRuntime
    gpu_lock: threading.Lock
    repository_lock: threading.Lock


class ResearchAdapter(ABC):
    def __init__(self, context: AdapterContext):
        self.context = context

    @abstractmethod
    def prepare(self) -> None:
        """Create and validate the isolated research workspace."""

    @abstractmethod
    def run_iteration(self) -> None:
        """Run one measured experiment and persist its outcome."""
