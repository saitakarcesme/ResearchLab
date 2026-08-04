#!/usr/bin/env python3
"""Pinned, single-load speed benchmark for the Beyefendi-v2 PEFT adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "huggingface:Ibrahimsait/Beyefendi-v2"
ADAPTER_REPOSITORY = "Ibrahimsait/Beyefendi-v2"
ADAPTER_REVISION = "498a7d2b30a60b05ccbcd00424dc2be1d2537cb1"
BASE_MODEL = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ADAPTER_FILE_SHA256 = "ce3e946660cb27dca3979558b11d1ea138ce6ac3e49d9ed88f75ba0e7dc897c2"
ADAPTER_CONFIG_SHA256 = "acc9b726a54e3ac5413934c731fdafe3560f69938f5cd182b6e7a111e8d0bfa6"
RESULT_PREFIX = "RESEARCHLAB_BENCHMARK_RESULT="
SYSTEM_PROMPT = (
    "Sen Beyefendi'sin. Türkçe sorulara açık, doğrudan ve dürüst cevap ver. "
    "Bilmediğin bir şeyi uydurma; belirsizliği açıkça belirt."
)
PROMPT_SEED = (
    "Yerel bir dil modelinin hızını değerlendiriyoruz. Bağlam uzunluğu, aynı anda "
    "işlenen istek sayısı ve ekran kartı belleği arasındaki ilişkiyi anlaşılır bir "
    "Türkçeyle açıkla. Sonunda gündelik kullanım için kısa bir öneri ver. "
)


@dataclass(frozen=True, slots=True)
class BenchmarkProfile:
    id: str
    label: str
    target_input_tokens: int
    batch_size: int
    output_tokens_per_request: int


PROFILES = (
    BenchmarkProfile("interactive", "Tek kullanıcı", 256, 1, 128),
    BenchmarkProfile("long_context", "Uzun bağlam", 1024, 1, 128),
    BenchmarkProfile("dual_request", "İki eşzamanlı istek", 256, 2, 128),
    BenchmarkProfile("gpu_saturation", "GPU odaklı toplu üretim", 256, 4, 128),
)


def _finite_positive(value: Any, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise RuntimeError(f"{name} must be a finite positive number")
    return converted


def _physical_gpu_identifier() -> str:
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0].strip()
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not identify a physical GPU")
    return visible


class GpuMonitor:
    """Sample utilization while generation is active without importing NVML."""

    def __init__(self, gpu_identifier: str) -> None:
        self.gpu_identifier = gpu_identifier
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.utilization: list[float] = []
        self.memory_used_mb: list[float] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | int | None]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return {
            "sample_count": len(self.utilization),
            "average_gpu_utilization_percent": (
                statistics.fmean(self.utilization) if self.utilization else None
            ),
            "peak_gpu_utilization_percent": (
                max(self.utilization) if self.utilization else None
            ),
            "peak_memory_used_mb": (
                max(self.memory_used_mb) if self.memory_used_mb else None
            ),
        }

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.gpu_identifier}",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                if result.returncode == 0:
                    fields = result.stdout.strip().splitlines()[0].split(",")
                    if len(fields) == 2:
                        self.utilization.append(float(fields[0].strip()))
                        self.memory_used_mb.append(float(fields[1].strip()))
            except (IndexError, OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(0.2)


def _local_kwargs() -> dict[str, Any]:
    return {"local_files_only": False}


def _load_supported_model(
    transformers_module: Any,
    auto_model_class: Any,
    model_kwargs: dict[str, Any],
) -> Any:
    try:
        return auto_model_class.from_pretrained(BASE_MODEL, **model_kwargs)
    except ValueError as auto_error:
        for class_name in (
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5ForCausalLM",
        ):
            model_class = getattr(transformers_module, class_name, None)
            if model_class is not None:
                return model_class.from_pretrained(BASE_MODEL, **model_kwargs)
        raise RuntimeError(
            "Transformers cannot load the pinned Qwen3.5 base model"
        ) from auto_error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_adapter(adapter_path: str | None) -> Path:
    if adapter_path:
        snapshot = Path(adapter_path).expanduser().resolve(strict=True)
    else:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=ADAPTER_REPOSITORY,
                revision=ADAPTER_REVISION,
                allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
            )
        )
    weights = snapshot / "adapter_model.safetensors"
    config = snapshot / "adapter_config.json"
    if not weights.is_file() or not config.is_file():
        raise RuntimeError("The pinned adapter snapshot is incomplete")
    if _sha256(weights) != ADAPTER_FILE_SHA256:
        raise RuntimeError(
            "Beyefendi-v2 adapter_model.safetensors does not match the pinned hash"
        )
    if _sha256(config) != ADAPTER_CONFIG_SHA256:
        raise RuntimeError(
            "Beyefendi-v2 adapter_config.json does not match the pinned hash"
        )
    return snapshot


def load_model(adapter_path: str | None) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    import transformers as transformers_module
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Beyefendi-v2 speed benchmark")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU must support bfloat16 computation")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        trust_remote_code=False,
        use_fast=True,
        **_local_kwargs(),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise RuntimeError("The pinned base tokenizer has no chat template")

    adapter_snapshot = _resolve_adapter(adapter_path)
    adapter_config = PeftConfig.from_pretrained(
        str(adapter_snapshot), local_files_only=True
    )
    if str(adapter_config.base_model_name_or_path) != BASE_MODEL:
        raise RuntimeError("The pinned adapter no longer names the expected base model")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_kwargs: dict[str, Any] = {
        "revision": BASE_REVISION,
        "trust_remote_code": False,
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "device_map": {"": 0},
        "quantization_config": quantization,
        "attn_implementation": "sdpa",
        **_local_kwargs(),
    }
    load_started = time.perf_counter()
    base = _load_supported_model(
        transformers_module,
        AutoModelForCausalLM,
        model_kwargs,
    )
    model = PeftModel.from_pretrained(
        base,
        str(adapter_snapshot),
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    device_map = getattr(model, "hf_device_map", None) or getattr(
        base, "hf_device_map", {}
    )
    devices = {str(device).lower() for device in device_map.values()}
    if device_map and any(
        device in {"cpu", "disk", "meta"} for device in devices
    ):
        rendered_map = {
            str(module): str(device) for module, device in device_map.items()
        }
        raise RuntimeError(
            "The model is not fully resident on the selected GPU; "
            f"device map: {json.dumps(rendered_map, sort_keys=True)[:1500]}"
        )
    non_cuda_parameters = sorted(
        {
            str(parameter.device)
            for parameter in model.parameters()
            if parameter.device.type != "cuda"
        }
    )
    if non_cuda_parameters:
        joined = ", ".join(non_cuda_parameters)
        raise RuntimeError(f"Model parameters were offloaded from CUDA: {joined}")

    properties = torch.cuda.get_device_properties(0)
    metadata = {
        "load_seconds": load_seconds,
        "device_map": (
            {str(key): str(value) for key, value in device_map.items()}
            if device_map
            else {"": "cuda:0 (verified from parameters)"}
        ),
        "fully_gpu_resident": True,
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": int(properties.total_memory),
        "model_memory_footprint_bytes": int(model.get_memory_footprint()),
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
        "adapter_file_sha256": ADAPTER_FILE_SHA256,
        "adapter_config_sha256": ADAPTER_CONFIG_SHA256,
        "physical_gpu_identifier": _physical_gpu_identifier(),
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": (
            [int(value) for value in tokenizer.eos_token_id]
            if isinstance(tokenizer.eos_token_id, (list, tuple))
            else int(tokenizer.eos_token_id)
        ),
    }
    return model, tokenizer, torch, metadata


def _render_messages(tokenizer: Any, messages: list[dict[str, str]]) -> Any:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    input_ids = rendered.get("input_ids") if isinstance(rendered, Mapping) else rendered
    if input_ids is None:
        raise RuntimeError("Tokenizer did not return input_ids")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError("Tokenizer returned an unexpected prompt tensor")
    return input_ids


def _render_prompt(tokenizer: Any, torch: Any, target_tokens: int) -> Any:
    content = PROMPT_SEED * max(1, target_tokens // 48)
    full_input_ids = _render_messages(
        tokenizer,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    empty_input_ids = _render_messages(
        tokenizer,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ""},
        ],
    )
    full_tokens = full_input_ids[0].tolist()
    empty_tokens = empty_input_ids[0].tolist()
    prefix_length = 0
    while (
        prefix_length < len(full_tokens)
        and prefix_length < len(empty_tokens)
        and full_tokens[prefix_length] == empty_tokens[prefix_length]
    ):
        prefix_length += 1
    suffix_length = 0
    while (
        suffix_length < len(full_tokens) - prefix_length
        and suffix_length < len(empty_tokens) - prefix_length
        and full_tokens[-(suffix_length + 1)] == empty_tokens[-(suffix_length + 1)]
    ):
        suffix_length += 1
    framing_length = prefix_length + suffix_length
    if framing_length >= target_tokens:
        raise RuntimeError("The fixed profile is too short for the chat framing")
    if len(full_tokens) < target_tokens:
        raise RuntimeError("Rendered prompt is shorter than its fixed token profile")
    content_length = target_tokens - framing_length
    available_content = len(full_tokens) - framing_length
    if available_content < content_length:
        raise RuntimeError("Rendered user content cannot fill the fixed token profile")
    segments = [
        full_input_ids[:, :prefix_length],
        full_input_ids[:, prefix_length : prefix_length + content_length],
    ]
    if suffix_length:
        segments.append(full_input_ids[:, -suffix_length:])
    input_ids = torch.cat(segments, dim=1)
    if input_ids.shape[1] != target_tokens:
        raise RuntimeError("Fixed prompt construction produced the wrong token length")
    return input_ids.to("cuda")


def _generate_once(
    model: Any,
    tokenizer: Any,
    torch: Any,
    profile: BenchmarkProfile,
    *,
    output_tokens: int,
    monitor_gpu: bool,
) -> dict[str, Any]:
    input_ids = _render_prompt(tokenizer, torch, profile.target_input_tokens)
    input_ids = input_ids.repeat(profile.batch_size, 1)
    attention_mask = torch.ones_like(input_ids)
    monitor = GpuMonitor(_physical_gpu_identifier()) if monitor_gpu else None
    torch.manual_seed(20260731)
    torch.cuda.manual_seed_all(20260731)
    torch.cuda.synchronize()
    if monitor:
        monitor.start()
    try:
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=output_tokens,
                min_new_tokens=output_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    finally:
        gpu_samples = monitor.stop() if monitor else {}
    generated_per_request = int(output.shape[1] - input_ids.shape[1])
    generated_total = generated_per_request * profile.batch_size
    throughput = generated_total / _finite_positive(elapsed, "elapsed_seconds")
    return {
        "elapsed_seconds": elapsed,
        "input_tokens_per_request": int(input_ids.shape[1]),
        "output_tokens_per_request": generated_per_request,
        "total_output_tokens": generated_total,
        "output_tokens_per_second": throughput,
        "request_latency_seconds": elapsed,
        **gpu_samples,
    }


def run_profile(
    model: Any,
    tokenizer: Any,
    torch: Any,
    profile: BenchmarkProfile,
) -> dict[str, Any]:
    try:
        warmup = _generate_once(
            model,
            tokenizer,
            torch,
            profile,
            output_tokens=16,
            monitor_gpu=False,
        )
        repeats = [
            _generate_once(
                model,
                tokenizer,
                torch,
                profile,
                output_tokens=profile.output_tokens_per_request,
                monitor_gpu=True,
            )
            for _ in range(3)
        ]
        speeds = [float(item["output_tokens_per_second"]) for item in repeats]
        latencies = [float(item["request_latency_seconds"]) for item in repeats]
        return {
            **asdict(profile),
            "warmup": warmup,
            "repeats": repeats,
            "median_output_tokens_per_second": statistics.median(speeds),
            "median_request_latency_seconds": statistics.median(latencies),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - preserve other profile measurements
        torch.cuda.empty_cache()
        return {
            **asdict(profile),
            "warmup": None,
            "repeats": [],
            "median_output_tokens_per_second": None,
            "median_request_latency_seconds": None,
            "error": f"{type(exc).__name__}: {exc}"[:1000],
        }


def benchmark(
    run_id: str, scheduler_share: int, adapter_path: str | None
) -> dict[str, Any]:
    model, tokenizer, torch, model_metadata = load_model(adapter_path)
    try:
        profile_results = [
            run_profile(model, tokenizer, torch, profile) for profile in PROFILES
        ]
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    finally:
        del model
        torch.cuda.empty_cache()
    return {
        "schema_version": 1,
        "run_id": run_id,
        "model": {
            "id": MODEL_ID,
            "adapter_repository": ADAPTER_REPOSITORY,
            "adapter_revision": ADAPTER_REVISION,
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
        },
        "runtime": {
            "scheduler_share_percent": scheduler_share,
            "quantization": "nf4",
            "compute_dtype": "bfloat16",
            "double_quantization": True,
            "device_map": {"": 0},
            "deterministic_generation": True,
            "warmup_runs_per_profile": 1,
            "measured_repeats_per_profile": 3,
        },
        "gpu": {
            **model_metadata,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
        },
        "profiles": profile_results,
    }


def _fatal_payload(
    run_id: str, scheduler_share: int, error: Exception
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "model": {
            "id": MODEL_ID,
            "adapter_repository": ADAPTER_REPOSITORY,
            "adapter_revision": ADAPTER_REVISION,
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
        },
        "runtime": {"scheduler_share_percent": scheduler_share},
        "gpu": {},
        "profiles": [],
        "fatal_error": str(error)[:2000],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scheduler-share", type=int, required=True)
    parser.add_argument("--adapter-path")
    args = parser.parse_args()
    if args.scheduler_share != 100:
        payload = _fatal_payload(
            args.run_id,
            args.scheduler_share,
            RuntimeError("Beyefendi-v2 benchmark requires a 100% scheduler share"),
        )
        print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False, allow_nan=False))
        return 2
    try:
        payload = benchmark(args.run_id, args.scheduler_share, args.adapter_path)
        status = 0
    except Exception as exc:  # noqa: BLE001 - emit a terminal machine-readable result
        payload = _fatal_payload(args.run_id, args.scheduler_share, exc)
        status = 2
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False, allow_nan=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
