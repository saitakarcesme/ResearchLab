from __future__ import annotations

from typing import Any

from backend.runners.beyefendi_v2_benchmark import (
    ADAPTER_CONFIG_SHA256,
    ADAPTER_REPOSITORY,
    ADAPTER_REVISION,
    BASE_MODEL,
    BASE_REVISION,
    MODEL_ID,
)

BEYEFENDI_V2_SOURCE_URL = f"https://huggingface.co/{ADAPTER_REPOSITORY}"
BEYEFENDI_V2_ADAPTER_SIZE_BYTES = 86_624_424
BEYEFENDI_V2_BASE_DOWNLOAD_SIZE_BYTES = 19_329_393_661
BEYEFENDI_V2_TOTAL_DOWNLOAD_SIZE_BYTES = 19_416_019_265
BEYEFENDI_V2_RUNTIME_FOOTPRINT_BYTES = 7_702_963_456
BEYEFENDI_V2_MODEL_ID = MODEL_ID
BEYEFENDI_V2_REVISION = ADAPTER_REVISION
BEYEFENDI_V2_BASE_MODEL = BASE_MODEL
BEYEFENDI_V2_BASE_REVISION = BASE_REVISION


def beyefendi_v2_catalog_model() -> dict[str, Any]:
    """Return the immutable Hugging Face checkpoint exposed by this runtime."""

    return {
        "id": MODEL_ID,
        "provider": "huggingface",
        "name": "Beyefendi-v2",
        "digest": ADAPTER_REVISION,
        "size_bytes": BEYEFENDI_V2_TOTAL_DOWNLOAD_SIZE_BYTES,
        "adapter_size_bytes": BEYEFENDI_V2_ADAPTER_SIZE_BYTES,
        "base_download_size_bytes": BEYEFENDI_V2_BASE_DOWNLOAD_SIZE_BYTES,
        "runtime_memory_footprint_bytes": BEYEFENDI_V2_RUNTIME_FOOTPRINT_BYTES,
        "parameter_size": "9.7B base + 43.3M LoRA",
        "quantization_level": "NF4 (runtime)",
        "family": "qwen3.5-peft",
        "capabilities": ["completion", "thinking"],
        "context_length": 262_144,
        "recommended": True,
        "source_url": BEYEFENDI_V2_SOURCE_URL,
        "adapter_repository": ADAPTER_REPOSITORY,
        "adapter_revision": ADAPTER_REVISION,
        "adapter_config_sha256": ADAPTER_CONFIG_SHA256,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
    }
