from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    auto_start: bool = False


class ResearchUpdate(StrictModel):
    title: str | None = Field(default=None, min_length=1, max_length=100)
    objective: str | None = Field(default=None, min_length=1, max_length=4000)
    gpu_source_id: str | None = None
    target_gpu_allocation: int | None = Field(default=None, ge=1, le=100)
