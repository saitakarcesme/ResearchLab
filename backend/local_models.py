from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


class OllamaConnectionError(RuntimeError):
    """Raised when the configured local Ollama runtime cannot be queried."""


@dataclass(frozen=True, slots=True)
class OllamaModel:
    id: str
    name: str
    digest: str
    size_bytes: int
    parameter_size: str | None
    quantization_level: str | None
    family: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": "ollama",
            "name": self.name,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "parameter_size": self.parameter_size,
            "quantization_level": self.quantization_level,
            "family": self.family,
        }


class OllamaClient:
    def __init__(self, base_url: str | None = None, *, timeout_seconds: float = 15):
        self.base_url = (
            base_url
            or os.getenv("AUTORESEARCH_OLLAMA_URL")
            or DEFAULT_OLLAMA_BASE_URL
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request(
        self,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        method = "GET"
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
            method = "POST"
        request = Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with urlopen(
                request,
                timeout=timeout_seconds or self.timeout_seconds,
            ) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise OllamaConnectionError(
                f"Ollama returned HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaConnectionError(
                f"Could not reach the Ollama runtime at {self.base_url}: {exc}"
            ) from exc
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaConnectionError("Ollama returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise OllamaConnectionError("Ollama returned an unexpected response")
        if value.get("error"):
            raise OllamaConnectionError(str(value["error"]))
        return value

    def list_models(self) -> list[OllamaModel]:
        payload = self._request("/api/tags")
        models: list[OllamaModel] = []
        for item in payload.get("models", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "").strip()
            digest = str(item.get("digest") or "").strip()
            if not name or not digest:
                continue
            details = item.get("details") if isinstance(item.get("details"), dict) else {}
            models.append(
                OllamaModel(
                    id=f"ollama:{name}",
                    name=name,
                    digest=digest,
                    size_bytes=int(item.get("size") or 0),
                    parameter_size=_optional_text(details.get("parameter_size")),
                    quantization_level=_optional_text(
                        details.get("quantization_level")
                    ),
                    family=_optional_text(details.get("family")),
                )
            )
        return sorted(models, key=lambda item: (item.size_bytes, item.name.lower()))

    def resolve(self, model_id: str) -> OllamaModel:
        if not model_id.startswith("ollama:"):
            raise ValueError("Only Ollama model identifiers are supported")
        match = next((item for item in self.list_models() if item.id == model_id), None)
        if match is None:
            raise ValueError(
                f"The selected model is not installed in {self.base_url}: {model_id[7:]}"
            )
        return match

    def show(self, name: str) -> dict[str, Any]:
        return self._request("/api/show", {"model": name})

    def version(self) -> str | None:
        return _optional_text(self._request("/api/version").get("version"))

    def running_models(self) -> list[dict[str, Any]]:
        payload = self._request("/api/ps")
        return [item for item in payload.get("models", []) if isinstance(item, dict)]

    def completion_models(self) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for model in self.list_models():
            shown = self.show(model.name)
            capabilities = [
                str(value)
                for value in shown.get("capabilities", [])
                if isinstance(value, str)
            ]
            if "completion" not in capabilities:
                continue
            model_info = (
                shown.get("model_info")
                if isinstance(shown.get("model_info"), dict)
                else {}
            )
            context_lengths = [
                int(value)
                for key, value in model_info.items()
                if key.endswith(".context_length")
                and isinstance(value, (int, float))
            ]
            item = model.as_dict()
            item.update(
                {
                    "capabilities": capabilities,
                    "context_length": max(context_lengths, default=None),
                    "recommended": model.size_bytes <= 12 * 1024**3,
                }
            )
            items.append(item)
        return {
            "runtime": {
                "provider": "ollama",
                "endpoint": self.base_url,
                "version": self.version(),
            },
            "models": items,
        }

    def generate(
        self,
        name: str,
        *,
        prompt: str,
        options: Mapping[str, Any],
        keep_alive: str | int = "5m",
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "/api/generate",
            {
                "model": name,
                "prompt": prompt,
                "stream": False,
                "raw": True,
                "think": False,
                "keep_alive": keep_alive,
                "options": dict(options),
            },
            timeout_seconds=timeout_seconds,
        )

    def unload(self, name: str) -> None:
        self._request(
            "/api/generate",
            {"model": name, "keep_alive": 0},
            timeout_seconds=30,
        )


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def model_name_from_id(model_id: str) -> str:
    if not model_id.startswith("ollama:"):
        raise ValueError("Only Ollama model identifiers are supported")
    name = model_id.removeprefix("ollama:").strip()
    if not name:
        raise ValueError("Ollama model identifier is missing a model name")
    # Keep model names opaque, but reject query/control-like values before logs.
    if any(character in name for character in ("\\", "?", "#", "\n", "\r")):
        raise ValueError("Invalid Ollama model name")
    return name
