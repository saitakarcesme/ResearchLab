from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from backend.config import Settings
from backend.local_models import model_name_from_id

_MODEL_METADATA: dict[str, tuple[str, str]] = {
    "gpt-5.6-sol": (
        "GPT-5.6 Sol",
        "Deepest analysis for difficult experiment planning and source synthesis.",
    ),
    "gpt-5.6-terra": (
        "GPT-5.6 Terra",
        "Balanced reasoning speed for most Lab work.",
    ),
    "gpt-5.3-codex-spark": (
        "GPT-5.3 Codex Spark",
        "Faster iteration for narrow technical experiments and quick topic scans.",
    ),
}


def researcher_model_catalog(
    settings: Settings,
    local_models: Sequence[Mapping[str, Any]] = (),
    *,
    local_error: str | None = None,
) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for model_id in settings.researcher_models:
        label, description = _MODEL_METADATA.get(
            model_id,
            (model_id, "Configured Codex research model."),
        )
        models.append(
            {
                "id": model_id,
                "label": label,
                "description": description,
                "provider": "codex",
                "capabilities": ["lab_agent"],
                "recommended": model_id == settings.default_researcher_model,
            }
        )
    for model in local_models:
        capabilities = {
            str(value) for value in model.get("capabilities", []) if value
        }
        model_id = str(model.get("id") or "")
        if (
            not model_id.startswith("ollama:")
            or "completion" not in capabilities
            or "tools" not in capabilities
        ):
            continue
        name = str(model.get("name") or model_id.removeprefix("ollama:"))
        parameter_size = str(model.get("parameter_size") or "").strip() or None
        models.append(
            {
                "id": model_id,
                "label": name,
                "description": "Runs on this computer"
                + (f" ({parameter_size})" if parameter_size else ""),
                "provider": "ollama",
                "parameter_size": parameter_size,
                "capabilities": ["lab_agent", "local"],
                "recommended": False,
            }
        )
    return {
        "default_model_id": settings.default_researcher_model,
        "models": models,
        "local_error": local_error,
    }


def resolve_researcher_model(
    settings: Settings,
    requested: str | None,
    local_models: Sequence[Mapping[str, Any]] = (),
) -> str:
    model_id = (requested or settings.default_researcher_model).strip()
    enabled = {
        str(model.get("id"))
        for model in researcher_model_catalog(settings, local_models)["models"]
    }
    if model_id not in enabled:
        raise ValueError(
            "The selected research model is not enabled. Refresh the model list and choose an available model."
        )
    return model_id


def researcher_model_command_args(model_id: str) -> list[str]:
    """Return safe top-level Codex CLI arguments for one persisted agent."""

    if model_id.startswith("ollama:"):
        return [
            "--oss",
            "--local-provider",
            "ollama",
            "--model",
            model_name_from_id(model_id),
        ]
    return ["--model", model_id]
