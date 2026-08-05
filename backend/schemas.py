from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class GpuSourceCreate(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    type: Literal["local", "remote"]
    host: str | None = Field(default=None, max_length=255)
    port: int = Field(default=22, ge=1, le=65535)
    username: str | None = Field(default=None, max_length=120)
    auth_method: Literal["agent", "key_env"] = "agent"
    workspace_path: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_remote(self) -> GpuSourceCreate:
        if self.type == "remote" and not (
            self.host and self.username and self.workspace_path
        ):
            raise ValueError(
                "Remote sources require host, username, and workspace_path"
            )
        return self


class GpuSourceUpdate(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    host: str | None = Field(default=None, max_length=255)
    port: int | None = Field(default=None, ge=1, le=65535)
    username: str | None = Field(default=None, max_length=120)
    auth_method: Literal["agent", "key_env"] | None = None
    workspace_path: str | None = Field(default=None, max_length=1000)


class ResearchCreate(StrictModel):
    original_prompt: str = Field(min_length=3, max_length=4000)
    gpu_source_id: str | None = None
    target_gpu_allocation: int = Field(default=100, ge=1, le=100)
    title: str | None = Field(default=None, min_length=1, max_length=100)
    objective: str | None = Field(default=None, min_length=1, max_length=4000)
    research_type: Literal[
        "training_optimization", "local_model_benchmark"
    ] = "training_optimization"
    model_id: str | None = Field(default=None, min_length=1, max_length=300)
    benchmark_profile: Literal[
        "ollama-text-v1", "hf-transformers-text-v1"
    ] | None = None
    researcher_model_id: str | None = Field(default=None, min_length=1, max_length=120)
    schedule_start_time: str | None = Field(default=None, pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    schedule_end_time: str | None = Field(default=None, pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    schedule_timezone: str | None = Field(default=None, min_length=1, max_length=80)
    schedule_utc_offset_minutes: int | None = Field(default=None, ge=-840, le=840)
    auto_start: bool = False

    @model_validator(mode="after")
    def validate_research_type(self) -> ResearchCreate:
        schedule_values = (
            self.schedule_start_time,
            self.schedule_end_time,
            self.schedule_timezone,
            self.schedule_utc_offset_minutes,
        )
        if any(value is not None for value in schedule_values) and not all(
            value is not None for value in schedule_values
        ):
            raise ValueError("A schedule requires a start time, end time, and timezone")
        if self.schedule_start_time == self.schedule_end_time and self.schedule_start_time:
            raise ValueError("The schedule start and end times must be different")
        if self.schedule_timezone:
            try:
                ZoneInfo(self.schedule_timezone)
            except ZoneInfoNotFoundError:
                # Windows Python installations do not always ship the IANA
                # database. The browser-provided UTC offset remains a durable
                # fallback for local scheduling.
                pass
        if self.research_type == "local_model_benchmark":
            if not self.model_id:
                raise ValueError("A local model must be selected for this benchmark")
            if self.model_id.startswith("ollama:"):
                self.benchmark_profile = self.benchmark_profile or "ollama-text-v1"
                if self.benchmark_profile != "ollama-text-v1":
                    raise ValueError("Ollama models require the ollama-text-v1 profile")
            elif self.model_id == "huggingface:Ibrahimsait/Beyefendi-v2":
                self.benchmark_profile = (
                    self.benchmark_profile or "hf-transformers-text-v1"
                )
                if self.benchmark_profile != "hf-transformers-text-v1":
                    raise ValueError(
                        "Beyefendi-v2 requires the hf-transformers-text-v1 profile"
                    )
                if self.target_gpu_allocation != 100:
                    raise ValueError(
                        "Beyefendi-v2 speed research requires a 100% GPU scheduling share"
                    )
            else:
                raise ValueError("The selected model provider is not supported")
        elif self.model_id is not None or self.benchmark_profile is not None:
            raise ValueError(
                "Model fields are only valid for a local-model benchmark"
            )
        return self


class ResearchUpdate(StrictModel):
    title: str | None = Field(default=None, min_length=1, max_length=100)
    objective: str | None = Field(default=None, min_length=1, max_length=4000)
    gpu_source_id: str | None = None
    target_gpu_allocation: int | None = Field(default=None, ge=1, le=100)
