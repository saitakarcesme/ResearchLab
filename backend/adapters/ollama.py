from __future__ import annotations

import json
import re
import shutil
import statistics
from dataclasses import dataclass
from typing import Any

from backend.adapters.base import (
    AdapterContext,
    ResearchAdapter,
    ResearchComplete,
    ResearchFailed,
)
from backend.local_models import OllamaClient, model_name_from_id
from backend.process_control import run_managed_process

BENCHMARK_PROMPT = (
    "Explain, in clear practical language, how batching and context length affect "
    "the speed and memory use of a local language model. Give one concrete example."
)


@dataclass(frozen=True, slots=True)
class BenchmarkProfile:
    label: str
    num_ctx: int
    num_batch: int


PROFILES = (
    BenchmarkProfile("Compact context", 2048, 256),
    BenchmarkProfile("Compact context, larger batch", 2048, 512),
    BenchmarkProfile("Everyday context", 4096, 512),
    BenchmarkProfile("Everyday context, throughput focus", 4096, 1024),
)
MAX_PROFILE_ATTEMPTS = 2
PROFILE_MARKER = re.compile(r"^Profile (\d+)/(\d+):")


class OllamaBenchmarkAdapter(ResearchAdapter):
    """Measure a selected installed model with reproducible Ollama profiles."""

    def __init__(self, context: AdapterContext, *, client: OllamaClient | None = None):
        super().__init__(context)
        self.db = context.database
        self.research_id = str(context.research["id"])
        self.model_id = str(context.research.get("model_id") or "")
        self.model_name = model_name_from_id(self.model_id)
        self.client = client or OllamaClient()
        self._managed_http = client is None

    def prepare(self) -> None:
        if self.context.gpu_source.get("type") != "local":
            raise RuntimeError(
                "Local-model benchmarks currently require the local GPU source. "
                "Remote model caches are separate and must be configured explicitly."
            )
        expected_runtime = str(self.context.research.get("model_runtime") or "")
        if expected_runtime and expected_runtime.rstrip("/") != self.client.base_url:
            raise RuntimeError(
                "The Ollama runtime changed after this research was queued. Re-create "
                "the research against the currently configured local catalog."
            )
        installed = self.client.resolve(self.model_id)
        expected_digest = str(self.context.research.get("model_digest") or "")
        if expected_digest and installed.digest != expected_digest:
            raise RuntimeError(
                "The installed model changed after this research was queued; create a "
                "new research to benchmark the new digest."
            )
        details = self.client.show(self.model_name)
        capabilities = details.get("capabilities") or []
        if "completion" not in capabilities:
            raise RuntimeError(
                f"{self.model_name} does not expose Ollama's completion capability"
            )
        self.db.add_log(
            self.research_id,
            "model_verified",
            f"Verified the installed {self.model_name} model and pinned its digest for this benchmark.",
            data={
                "model_id": installed.id,
                "digest": installed.digest,
                "size_bytes": installed.size_bytes,
                "parameter_size": installed.parameter_size,
                "quantization_level": installed.quantization_level,
            },
        )

    def run_iteration(self) -> None:
        research = self.db.get_research(self.research_id, detail=True)
        if research is None:
            raise InterruptedError
        profile_index, profile, attempt = self._next_profile(research)
        previous_best = research.get("best_value")
        experiment = self.db.create_experiment(
            self.research_id,
            (
                f"Profile {profile_index + 1}/{len(PROFILES)}: Test whether "
                f"{profile.label.lower()} improves measured generation speed for "
                f"{self.model_name}."
            ),
            (
                f"Attempt {attempt}: warm the model, confirm full-GPU residency, then "
                f"run the same prompt three times with a {profile.num_ctx:,}-token "
                f"context and a {profile.num_batch}-token processing batch."
            ),
            previous_best,
        )
        experiment_id = str(experiment["id"])
        number = int(experiment["experiment_number"])
        self.db.add_log(
            self.research_id,
            "experiment_started",
            (
                f"Profile {profile_index + 1}/{len(PROFILES)} started: {profile.label} "
                f"({profile.num_ctx:,}-token context, batch {profile.num_batch}, "
                f"attempt {attempt}/{MAX_PROFILE_ATTEMPTS})."
            ),
            experiment_id=experiment_id,
        )

        acquired = False
        try:
            self.context.runtime.set_phase("waiting_for_gpu")
            while not self.context.runtime.stop_event.is_set():
                if self.context.gpu_lock.acquire(timeout=0.5):
                    acquired = True
                    break
            if not acquired:
                raise InterruptedError
            self.context.runtime.set_phase("evaluating")
            self._generate(
                profile,
                num_predict=16,
                sample_label=f"profile-{profile_index + 1}-warmup",
            )
            residency = self._full_gpu_residency()
            self.db.add_log(
                self.research_id,
                "model_warmed",
                (
                    f"{profile.label} is warm and {self.model_name} is fully resident "
                    f"on the GPU ({int(residency['size_vram']) / 1024**3:.1f} GB)."
                ),
                experiment_id=experiment_id,
                data=residency,
            )
            measurements = [
                self._measure(profile, repeat=index + 1) for index in range(3)
            ]
            speeds = [item["tokens_per_second"] for item in measurements]
            metric = statistics.median(speeds)
            accepted = previous_best is None or metric > float(previous_best)
            updates: dict[str, Any] | None = None
            if accepted:
                updates = {"best_value": metric}
                if research.get("baseline_value") is None:
                    updates["baseline_value"] = metric
            self.db.finish_experiment(
                experiment_id,
                metric_value=metric,
                accepted=accepted,
                git_commit=None,
                error=None,
                research_updates=updates,
            )
            event = "experiment_accepted" if accepted else "experiment_rejected"
            comparison = (
                "This is the first measured baseline."
                if previous_best is None
                else (
                    "It is the fastest profile so far."
                    if accepted
                    else f"The best remains {float(previous_best):.1f} tokens per second."
                )
            )
            self.db.add_log(
                self.research_id,
                event,
                (
                    f"{profile.label} produced {metric:.1f} output tokens per second. "
                    f"{comparison}"
                ),
                experiment_id=experiment_id,
                data={
                    "profile": profile.label,
                    "num_ctx": profile.num_ctx,
                    "num_batch": profile.num_batch,
                    "median_tokens_per_second": metric,
                    "measurements": measurements,
                    "gpu_residency": residency,
                },
            )
        except InterruptedError:
            self.db.finish_experiment(
                experiment_id,
                metric_value=None,
                accepted=False,
                git_commit=None,
                error="Benchmark stopped before the profile completed",
            )
            self._unload_safely()
            raise
        except Exception as exc:  # noqa: BLE001 - persist a failed profile and continue the queue
            self.db.finish_experiment(
                experiment_id,
                metric_value=None,
                accepted=False,
                git_commit=None,
                error=str(exc)[:3000],
            )
            self.db.add_log(
                self.research_id,
                "experiment_error",
                f"Profile {number} could not be measured: {exc}",
                level="error",
                experiment_id=experiment_id,
            )
        finally:
            if acquired:
                self.context.gpu_lock.release()
            self.context.runtime.set_phase("processing_results")

        self.context.runtime.set_phase("ready")

    def _next_profile(
        self, research: dict[str, Any]
    ) -> tuple[int, BenchmarkProfile, int]:
        experiments = research.get("experiments") or []
        for profile_index, profile in enumerate(PROFILES):
            attempts = [
                item
                for item in experiments
                if self._experiment_profile_index(item) == profile_index
            ]
            if any(
                item.get("metric_value") is not None and not item.get("error")
                for item in attempts
            ):
                continue
            if len(attempts) >= MAX_PROFILE_ATTEMPTS:
                self._finish_research(research)
            return profile_index, profile, len(attempts) + 1
        self._finish_research(research)

    @staticmethod
    def _experiment_profile_index(experiment: dict[str, Any]) -> int | None:
        match = PROFILE_MARKER.match(str(experiment.get("hypothesis") or ""))
        if not match or int(match.group(2)) != len(PROFILES):
            return None
        index = int(match.group(1)) - 1
        return index if 0 <= index < len(PROFILES) else None

    def _measure(
        self, profile: BenchmarkProfile, *, repeat: int
    ) -> dict[str, float | int]:
        response = self._generate(
            profile, num_predict=128, sample_label=f"repeat-{repeat}"
        )
        count = int(response.get("eval_count") or 0)
        duration_ns = int(response.get("eval_duration") or 0)
        if count <= 0 or duration_ns <= 0:
            raise RuntimeError("Ollama did not return usable output-token timing data")
        return {
            "tokens_per_second": count / (duration_ns / 1_000_000_000),
            "output_tokens": count,
            "eval_duration_ns": duration_ns,
            "prompt_tokens": int(response.get("prompt_eval_count") or 0),
            "prompt_duration_ns": int(response.get("prompt_eval_duration") or 0),
            "load_duration_ns": int(response.get("load_duration") or 0),
            "total_duration_ns": int(response.get("total_duration") or 0),
        }

    def _generate(
        self,
        profile: BenchmarkProfile,
        *,
        num_predict: int,
        sample_label: str,
    ) -> dict[str, Any]:
        options = {
            "num_ctx": profile.num_ctx,
            "num_batch": profile.num_batch,
            "num_predict": num_predict,
            "temperature": 0,
            "seed": 42,
        }
        if not self._managed_http:
            return self.client.generate(
                self.model_name,
                prompt=BENCHMARK_PROMPT,
                options=options,
                timeout_seconds=300,
            )

        curl = shutil.which("curl")
        if not curl:
            raise RuntimeError("curl is required for controllable Ollama benchmarks")
        output = (
            self.context.settings.runtime_dir
            / self.research_id
            / f"ollama-{sample_label}.json"
        )
        body = json.dumps(
            {
                "model": self.model_name,
                "prompt": BENCHMARK_PROMPT,
                "stream": False,
                "raw": True,
                "think": False,
                "keep_alive": "5m",
                "options": options,
            },
            ensure_ascii=False,
        )
        result = run_managed_process(
            [
                curl,
                "--silent",
                "--show-error",
                "--fail-with-body",
                "--max-time",
                "300",
                "--header",
                "Content-Type: application/json",
                "--data-binary",
                "@-",
                f"{self.client.base_url}/api/generate",
            ],
            cwd=None,
            env=None,
            output=output,
            timeout_seconds=310,
            runtime=self.context.runtime,
            stdin_text=body,
        )
        if result.stopped:
            raise InterruptedError
        if result.timed_out:
            raise RuntimeError("Ollama generation exceeded the five-minute limit")
        raw = output.read_text(encoding="utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama returned invalid JSON: {raw[-300:]}") from exc
        if result.returncode != 0 or payload.get("error"):
            detail = payload.get("error") or raw[-500:]
            raise RuntimeError(f"Ollama generation failed: {detail}")
        return payload

    def _full_gpu_residency(self) -> dict[str, Any]:
        running = self.client.running_models()
        item = next(
            (
                model
                for model in running
                if str(model.get("name") or model.get("model")) == self.model_name
            ),
            None,
        )
        if item is None:
            raise RuntimeError("Ollama did not report the warmed model as loaded")
        size = int(item.get("size") or 0)
        size_vram = int(item.get("size_vram") or 0)
        if size <= 0 or size_vram < size * 0.95:
            raise RuntimeError(
                "The model is partly CPU-offloaded, so its speed would not be comparable "
                "to a full-GPU run. Choose a smaller model or context."
            )
        return {
            "size": size,
            "size_vram": size_vram,
            "context_length": int(item.get("context_length") or 0),
            "fully_gpu_resident": True,
        }

    def _unload_safely(self) -> None:
        try:
            self.client.unload(self.model_name)
        except Exception as exc:  # noqa: BLE001 - cleanup must not turn measurements into a failure
            self.db.add_log(
                self.research_id,
                "model_unload_warning",
                f"Measurements are complete, but Ollama could not unload the model: {exc}",
                level="warning",
            )

    def _finish_research(self, research: dict[str, Any]) -> None:
        valid_profiles = {
            profile_index
            for item in research.get("experiments") or []
            if item.get("metric_value") is not None and not item.get("error")
            for profile_index in [self._experiment_profile_index(item)]
            if profile_index is not None
        }
        self._unload_safely()
        if len(valid_profiles) != len(PROFILES):
            raise ResearchFailed(
                f"Only {len(valid_profiles)} of {len(PROFILES)} profiles produced a "
                "valid full-GPU measurement after two attempts."
            )
        raise ResearchComplete(
            f"Completed all {len(PROFILES)} reproducible full-GPU profiles for "
            f"{self.model_name}."
        )
