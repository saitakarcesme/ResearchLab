from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    database_path: Path
    workspace_dir: Path
    log_dir: Path
    upstream_cache_dir: Path
    runtime_dir: Path
    upstream_url: str
    upstream_revision: str
    execution_enabled: bool
    codex_binary: str
    uv_binary: str
    git_binary: str
    nvidia_smi_binary: str
    experiment_timeout_seconds: int
    agent_timeout_seconds: int
    agent_reasoning_effort: str
    researcher_models: tuple[str, ...]
    default_researcher_model: str
    wsl_distro: str | None
    use_cuda_mps: bool
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> Settings:
        backend_dir = Path(__file__).resolve().parent
        data_dir = Path(
            os.getenv("AUTORESEARCH_DATA_DIR", backend_dir / "data")
        ).resolve()
        origins = os.getenv(
            "AUTORESEARCH_CORS_ORIGINS",
            ",".join(
                f"{scheme}://{host}:{port}"
                for scheme in ("http", "https")
                for host in ("localhost", "127.0.0.1")
                for port in (3000, 3001, 4173, 5173)
            ),
        )
        wsl = os.getenv("AUTORESEARCH_WSL_DISTRO", "").strip() or None
        codex_binary = os.getenv("AUTORESEARCH_CODEX_BIN", "codex")
        if os.name == "nt":
            codex_binary = shutil.which(codex_binary) or codex_binary
        return cls(
            data_dir=data_dir,
            database_path=Path(
                os.getenv("AUTORESEARCH_DATABASE", data_dir / "autoresearch.sqlite3")
            ).resolve(),
            workspace_dir=Path(
                os.getenv("AUTORESEARCH_WORKSPACES", data_dir / "workspaces")
            ).resolve(),
            log_dir=Path(os.getenv("AUTORESEARCH_LOGS", data_dir / "logs")).resolve(),
            upstream_cache_dir=Path(
                os.getenv("AUTORESEARCH_UPSTREAM_CACHE", data_dir / "upstream")
            ).resolve(),
            runtime_dir=Path(
                os.getenv("AUTORESEARCH_RUNTIMES", data_dir / "runtimes")
            ).resolve(),
            upstream_url=os.getenv(
                "AUTORESEARCH_UPSTREAM", "https://github.com/karpathy/autoresearch.git"
            ),
            upstream_revision=os.getenv(
                "AUTORESEARCH_UPSTREAM_REVISION",
                "228791fb499afffb54b46200aca536f79142f117",
            ),
            execution_enabled=_as_bool(os.getenv("AUTORESEARCH_ENABLE_EXECUTION")),
            codex_binary=codex_binary,
            uv_binary=os.getenv("AUTORESEARCH_UV_BIN", "uv"),
            git_binary=os.getenv("AUTORESEARCH_GIT_BIN", "git"),
            nvidia_smi_binary=os.getenv("AUTORESEARCH_NVIDIA_SMI_BIN", "nvidia-smi"),
            experiment_timeout_seconds=max(
                60, int(os.getenv("AUTORESEARCH_EXPERIMENT_TIMEOUT", "600"))
            ),
            agent_timeout_seconds=max(
                60, int(os.getenv("AUTORESEARCH_AGENT_TIMEOUT", "600"))
            ),
            agent_reasoning_effort=_agent_reasoning_effort(),
            researcher_models=_researcher_models(),
            default_researcher_model=_default_researcher_model(),
            wsl_distro=wsl,
            use_cuda_mps=_as_bool(os.getenv("AUTORESEARCH_USE_CUDA_MPS")),
            cors_origins=tuple(
                origin.strip() for origin in origins.split(",") if origin.strip()
            ),
        )

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.workspace_dir,
            self.log_dir,
            self.upstream_cache_dir,
            self.runtime_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


def _agent_reasoning_effort() -> str:
    effort = os.getenv("AUTORESEARCH_AGENT_REASONING_EFFORT", "high").strip().lower()
    allowed = {"low", "medium", "high", "xhigh", "max", "ultra"}
    if effort not in allowed:
        values = ", ".join(sorted(allowed))
        raise ValueError(
            f"AUTORESEARCH_AGENT_REASONING_EFFORT must be one of: {values}"
        )
    return effort


def _researcher_models() -> tuple[str, ...]:
    configured = os.getenv(
        "AUTORESEARCH_RESEARCHER_MODELS",
        "gpt-5.6-sol,gpt-5.6-terra,gpt-5.3-codex-spark",
    )
    models = tuple(
        dict.fromkeys(item.strip() for item in configured.split(",") if item.strip())
    )
    if not models:
        raise ValueError("AUTORESEARCH_RESEARCHER_MODELS must contain at least one model")
    for model in models:
        if len(model) > 120 or not all(
            character.isalnum() or character in "._:-" for character in model
        ):
            raise ValueError(
                "AUTORESEARCH_RESEARCHER_MODELS contains an invalid model identifier"
            )
    return models


def _default_researcher_model() -> str:
    models = _researcher_models()
    selected = os.getenv("AUTORESEARCH_DEFAULT_RESEARCHER_MODEL", models[0]).strip()
    if selected not in models:
        raise ValueError(
            "AUTORESEARCH_DEFAULT_RESEARCHER_MODEL must be listed in "
            "AUTORESEARCH_RESEARCHER_MODELS"
        )
    return selected
