from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.article import ResearchArticleGenerator, article_kind
from backend.brief import CodexResearchBriefGenerator
from backend.cloud_gpu import CloudGpuManager
from backend.cloud_providers import CloudProviderError, VastProvider
from backend.config import Settings
from backend.db import Database
from backend.huggingface_models import (
    BEYEFENDI_V2_BASE_MODEL,
    BEYEFENDI_V2_BASE_REVISION,
    BEYEFENDI_V2_MODEL_ID,
    BEYEFENDI_V2_REVISION,
    BEYEFENDI_V2_SOURCE_URL,
    beyefendi_v2_catalog_model,
)
from backend.local_models import OllamaClient, OllamaConnectionError
from backend.payments import PaymentConfigurationError, StripeCheckoutService
from backend.researcher_models import researcher_model_catalog, resolve_researcher_model
from backend.schemas import (
    CloudAccountCreate,
    CloudCheckoutCreate,
    CloudRentCreate,
    GpuSourceCreate,
    GpuSourceUpdate,
    ResearchCreate,
    ResearchUpdate,
)
from backend.secret_store import SecretStore
from backend.supervisor import ResearchSupervisor
from backend.telemetry import TelemetryService


def _database(request: Request) -> Database:
    return request.app.state.database


def _supervisor(request: Request) -> ResearchSupervisor:
    return request.app.state.supervisor


def _telemetry(request: Request) -> TelemetryService:
    return request.app.state.telemetry


def _ollama(request: Request) -> OllamaClient:
    return request.app.state.ollama


def _cloud(request: Request) -> CloudGpuManager:
    return request.app.state.cloud_gpu


def _payments(request: Request) -> StripeCheckoutService:
    return request.app.state.payments


def _local_researcher_models(
    request: Request,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        return list(_ollama(request).completion_models().get("models", [])), None
    except OllamaConnectionError as exc:
        return [], str(exc)


def _resolve_researcher_model(request: Request, requested: str | None) -> str:
    local_models: list[dict[str, Any]] = []
    if requested and requested.startswith("ollama:"):
        local_models, error = _local_researcher_models(request)
        if error:
            raise ValueError(error)
    return resolve_researcher_model(
        request.app.state.settings,
        requested,
        local_models,
    )


def _not_found(kind: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{kind} not found")


def _control_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return _not_found("Research")
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=409, detail=str(exc))


def create_app(
    settings: Settings | None = None,
    brief_generator: CodexResearchBriefGenerator | None = None,
    article_generator: ResearchArticleGenerator | None = None,
) -> FastAPI:
    configured = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configured.ensure_directories()
        database = Database(configured.database_path)
        database.initialize()
        telemetry = TelemetryService(configured.nvidia_smi_binary)
        provisional = {
            "id": "local",
            "type": "local",
            "name": "Local NVIDIA GPU",
            "host": None,
            "port": 22,
            "username": None,
            "auth_method": "agent",
            "workspace_path": None,
        }
        sample = telemetry.sample(provisional)
        local_name = sample.get("gpu_name") or "Local NVIDIA GPU"
        database.ensure_local_gpu_source(str(local_name))
        # A hard crash can leave a managed process behind while its durable row is
        # running, paused, or failed after an unconfirmed control operation.
        interrupted = database.list_process_cleanup_candidates()
        ResearchSupervisor.terminate_stale_process_groups(
            configured, database, interrupted
        )
        database.recover_interrupted_researches()
        supervisor = ResearchSupervisor(configured, database, telemetry)
        cloud_gpu = CloudGpuManager(database, SecretStore(configured.data_dir / "secrets"))
        app.state.settings = configured
        app.state.database = database
        app.state.telemetry = telemetry
        app.state.ollama = OllamaClient()
        app.state.supervisor = supervisor
        app.state.cloud_gpu = cloud_gpu
        app.state.payments = StripeCheckoutService(database, cloud_gpu)
        app.state.brief_generator = brief_generator or CodexResearchBriefGenerator(
            configured
        )
        app.state.article_generator = article_generator or ResearchArticleGenerator(
            configured, usage_recorder=database.record_codex_token_usage
        )

        async def cloud_sync_loop() -> None:
            while True:
                await asyncio.sleep(15)
                await asyncio.to_thread(cloud_gpu.sync_all)

        cloud_task = asyncio.create_task(cloud_sync_loop())
        try:
            yield
        finally:
            cloud_task.cancel()
            with suppress(asyncio.CancelledError):
                await cloud_task
            supervisor.shutdown()

    app = FastAPI(
        title="Autoresearch Lab API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(configured.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )
    app.mount(
        "/assets",
        StaticFiles(
            directory=Path(__file__).resolve().parents[1] / "dist" / "client" / "assets",
            check_dir=False,
        ),
        name="production-assets",
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return {
            "status": "ok",
            "execution_enabled": request.app.state.settings.execution_enabled,
            "database": str(request.app.state.settings.database_path),
        }

    @app.get("/api/gpu-sources")
    def list_gpu_sources(request: Request) -> list[dict[str, Any]]:
        return _database(request).list_gpu_sources()

    @app.get("/api/cloud/accounts")
    def list_cloud_accounts(request: Request) -> list[dict[str, Any]]:
        return [_cloud(request).public_account(value) for value in _database(request).list_cloud_accounts()]

    @app.get("/api/cloud/vast-offers")
    def list_public_vast_offers() -> list[dict[str, Any]]:
        try:
            return VastProvider.public_offers()
        except CloudProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/cloud/accounts", status_code=status.HTTP_201_CREATED)
    def create_cloud_account(payload: CloudAccountCreate, request: Request) -> dict[str, Any]:
        secret_ref = _cloud(request).secrets.put(payload.api_key)
        try:
            account = _database(request).create_cloud_account(
                {"provider": payload.provider, "name": payload.name, "budget_usd": payload.budget_usd,
                 "settings": {"ssh_key_id": payload.ssh_key_id, "workspace_path": payload.workspace_path, "image_name": payload.image_name}},
                secret_ref,
            )
        except Exception:
            _cloud(request).secrets.delete(secret_ref)
            raise
        return _cloud(request).public_account(account)

    @app.delete("/api/cloud/accounts/{account_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_cloud_account(account_id: str, request: Request) -> Response:
        try:
            account = _database(request).delete_cloud_account(account_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not account:
            raise _not_found("Cloud account")
        _cloud(request).secrets.delete(account["secret_ref"])
        return Response(status_code=204)

    @app.get("/api/cloud/accounts/{account_id}/offers")
    def list_cloud_offers(account_id: str, request: Request) -> list[dict[str, Any]]:
        try:
            return _cloud(request).offers(account_id)
        except KeyError as exc:
            raise _not_found("Cloud account") from exc
        except CloudProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/api/cloud/instances")
    def list_cloud_instances(request: Request) -> list[dict[str, Any]]:
        return _database(request).list_cloud_instances()

    @app.get("/api/cloud/payment-config")
    def cloud_payment_config(request: Request) -> dict[str, Any]:
        return _payments(request).public_config()

    @app.post("/api/cloud/checkout", status_code=status.HTTP_201_CREATED)
    def create_cloud_checkout(payload: CloudCheckoutCreate, request: Request) -> dict[str, Any]:
        try:
            return _payments(request).create(payload.account_id, payload.offer_id, payload.hours)
        except KeyError as exc:
            raise _not_found("Cloud account") from exc
        except CloudProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except (PaymentConfigurationError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/cloud/orders/{order_id}")
    def get_cloud_order(order_id: str, request: Request) -> dict[str, Any]:
        try:
            return _payments(request).sync(order_id)
        except KeyError as exc:
            raise _not_found("Cloud rental order") from exc
        except (PaymentConfigurationError, CloudProviderError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/cloud/stripe-webhook")
    async def stripe_cloud_webhook(request: Request, stripe_signature: str | None = Header(default=None)) -> dict[str, bool]:
        if not stripe_signature:
            raise HTTPException(status_code=400, detail="Missing Stripe-Signature header")
        raw = await request.body()
        try:
            _payments(request).handle_webhook(raw, stripe_signature)
        except (PaymentConfigurationError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"received": True}

    @app.post("/api/cloud/instances", status_code=status.HTTP_201_CREATED)
    def rent_cloud_instance(payload: CloudRentCreate, request: Request) -> dict[str, Any]:
        try:
            return _cloud(request).rent(payload.account_id, payload.offer_id, name=payload.name, max_hours=payload.max_hours)
        except KeyError as exc:
            raise _not_found("Cloud account") from exc
        except (ValueError, CloudProviderError) as exc:
            raise HTTPException(status_code=409 if isinstance(exc, ValueError) else 502, detail=str(exc)) from exc

    @app.post("/api/cloud/instances/{instance_id}/refresh")
    def refresh_cloud_instance(instance_id: str, request: Request) -> dict[str, Any]:
        try:
            return _cloud(request).sync(instance_id)
        except KeyError as exc:
            raise _not_found("Cloud instance") from exc
        except CloudProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/cloud/instances/{instance_id}/terminate")
    def terminate_cloud_instance(instance_id: str, request: Request) -> dict[str, Any]:
        try:
            return _cloud(request).terminate(instance_id)
        except KeyError as exc:
            raise _not_found("Cloud instance") from exc
        except CloudProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/gpu-sources", status_code=status.HTTP_201_CREATED)
    def create_gpu_source(payload: GpuSourceCreate, request: Request) -> dict[str, Any]:
        if payload.type == "local":
            raise HTTPException(
                status_code=409,
                detail="A real local GPU source is created automatically at startup",
            )
        return _database(request).create_gpu_source(payload.model_dump())

    @app.get("/api/gpu-sources/{source_id}")
    def get_gpu_source(source_id: str, request: Request) -> dict[str, Any]:
        source = _database(request).get_gpu_source(source_id)
        if source is None:
            raise _not_found("GPU source")
        return source

    @app.patch("/api/gpu-sources/{source_id}")
    def update_gpu_source(
        source_id: str, payload: GpuSourceUpdate, request: Request
    ) -> dict[str, Any]:
        try:
            source = _database(request).update_gpu_source(
                source_id, payload.model_dump(exclude_unset=True)
            )
        except Exception as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if source is None:
            raise _not_found("GPU source")
        return source

    @app.delete("/api/gpu-sources/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_gpu_source(source_id: str, request: Request) -> Response:
        try:
            deleted = _database(request).delete_gpu_source(source_id)
        except Exception as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise _not_found("GPU source")
        return Response(status_code=204)

    @app.get("/api/gpu-sources/{source_id}/telemetry")
    def gpu_telemetry(source_id: str, request: Request) -> dict[str, Any]:
        source = _database(request).get_gpu_source(source_id)
        if source is None:
            raise _not_found("GPU source")
        return _telemetry(request).sample(source)

    @app.get("/api/gpu-sources/{source_id}/models")
    def list_local_models(source_id: str, request: Request) -> dict[str, Any]:
        source = _database(request).get_gpu_source(source_id)
        if source is None:
            raise _not_found("GPU source")
        if source["type"] != "local":
            raise HTTPException(
                status_code=409,
                detail=(
                    "Remote model catalogs are separate from this computer. Add a "
                    "remote runtime before selecting models on that source."
                ),
            )
        pinned_model = beyefendi_v2_catalog_model()
        try:
            catalog = _ollama(request).completion_models()
        except OllamaConnectionError as exc:
            catalog = {
                "runtime": {
                    "provider": "ollama",
                    "endpoint": _ollama(request).base_url,
                    "version": None,
                    "error": str(exc),
                },
                "models": [],
            }
        catalog["models"] = [
            pinned_model,
            *[
                model
                for model in catalog.get("models", [])
                if model.get("id") != pinned_model["id"]
            ],
        ]
        return catalog

    @app.get("/api/telemetry/local")
    def local_telemetry(request: Request) -> dict[str, Any]:
        source = next(
            (
                item
                for item in _database(request).list_gpu_sources()
                if item["type"] == "local"
            ),
            None,
        )
        if source is None:
            raise _not_found("Local GPU source")
        return _telemetry(request).sample(source)

    @app.get("/api/researcher-models")
    def list_researcher_models(request: Request) -> dict[str, Any]:
        local_models, local_error = _local_researcher_models(request)
        return researcher_model_catalog(
            request.app.state.settings,
            local_models,
            local_error=local_error,
        )

    @app.get("/api/researches")
    def list_researches(
        request: Request,
        research_status: str | None = Query(default=None, alias="status"),
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> list[dict[str, Any]]:
        return _database(request).list_researches(research_status, limit, offset)

    @app.post("/api/researches", status_code=status.HTTP_201_CREATED)
    def create_research(payload: ResearchCreate, request: Request) -> dict[str, Any]:
        database = _database(request)
        try:
            researcher_model_id = _resolve_researcher_model(
                request, payload.researcher_model_id
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        source_id = payload.gpu_source_id
        if source_id is None:
            local = next(
                (
                    item
                    for item in database.list_gpu_sources()
                    if item["type"] == "local"
                ),
                None,
            )
            if local is None:
                raise HTTPException(
                    status_code=409, detail="No GPU source is configured"
                )
            source_id = local["id"]
        source = database.get_gpu_source(source_id)
        if source is None:
            raise _not_found("GPU source")
        brief = None
        values: dict[str, Any]
        if payload.research_type == "local_model_benchmark":
            if source["type"] != "local":
                raise HTTPException(
                    status_code=409,
                    detail="Local-model benchmarks currently require the local GPU source",
                )
            assert payload.model_id is not None
            if payload.model_id == BEYEFENDI_V2_MODEL_ID:
                values = {
                    "title": payload.title or "Beyefendi-v2 Hugging Face Speed",
                    "original_prompt": payload.original_prompt,
                    "objective": payload.objective
                    or (
                        "Measure the pinned Beyefendi-v2 adapter and Qwen3.5-9B base "
                        "on the RTX 3090 across interactive, long-context, dual-request, "
                        "and GPU-focused profiles. Exclude model loading and warm-up, "
                        "then maximize median generated tokens per second while keeping "
                        "the quantized model fully resident on the GPU."
                    ),
                    "metric_name": "output_tokens_per_second",
                    "metric_direction": "higher_is_better",
                    "gpu_source_id": source_id,
                    "target_gpu_allocation": 100,
                    "adapter_type": "huggingface_beyefendi_benchmark",
                    "research_type": payload.research_type,
                    "model_id": BEYEFENDI_V2_MODEL_ID,
                    "model_digest": BEYEFENDI_V2_REVISION,
                    "model_runtime": "huggingface",
                    "benchmark_profile": "hf-transformers-text-v1",
                    "researcher_model_id": researcher_model_id,
                    "status": "stopped" if payload.auto_start and not payload.schedule_start_time else "queued",
                }
            else:
                try:
                    installed = _ollama(request).resolve(payload.model_id)
                    shown = _ollama(request).show(installed.name)
                except (OllamaConnectionError, ValueError) as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                if "completion" not in (shown.get("capabilities") or []):
                    raise HTTPException(
                        status_code=409,
                        detail="The selected Ollama model cannot generate text",
                    )
                display_name = installed.name.removesuffix(":latest")
                values = {
                    "title": payload.title
                    or f"{display_name} Local Inference Efficiency",
                    "original_prompt": payload.original_prompt,
                    "objective": payload.objective
                    or (
                        f"Continuously optimize {installed.name} with agent-proposed context "
                        "and batch hypotheses under a fixed prompt and output length. Keep "
                        "strictly faster reproducible results, require full-GPU residency, "
                        "and continue until the user stops the research."
                    ),
                    "metric_name": "output_tokens_per_second",
                    "metric_direction": "higher_is_better",
                    "gpu_source_id": source_id,
                    "target_gpu_allocation": payload.target_gpu_allocation,
                    "adapter_type": "ollama_benchmark",
                    "research_type": payload.research_type,
                    "model_id": installed.id,
                    "model_digest": installed.digest,
                    "model_runtime": _ollama(request).base_url,
                    "benchmark_profile": payload.benchmark_profile
                    or "ollama-text-v1",
                    "researcher_model_id": researcher_model_id,
                    "status": "stopped" if payload.auto_start and not payload.schedule_start_time else "queued",
                }
        else:
            brief = request.app.state.brief_generator.generate(
                payload.original_prompt,
                supplied_title=payload.title,
                supplied_objective=payload.objective,
                researcher_model_id=researcher_model_id,
            )
            values = {
                "title": brief.title,
                "original_prompt": payload.original_prompt,
                "objective": brief.objective,
                "metric_name": "val_bpb",
                "metric_direction": "lower_is_better",
                "gpu_source_id": source_id,
                "target_gpu_allocation": payload.target_gpu_allocation,
                "adapter_type": "karpathy_autoresearch",
                "research_type": payload.research_type,
                "researcher_model_id": researcher_model_id,
                "status": "stopped" if payload.auto_start and not payload.schedule_start_time else "queued",
            }
        values.update(
            {
                "schedule_start_time": payload.schedule_start_time,
                "schedule_end_time": payload.schedule_end_time,
                "schedule_timezone": payload.schedule_timezone,
                "schedule_utc_offset_minutes": payload.schedule_utc_offset_minutes,
            }
        )
        research = database.create_research(values)
        if brief and brief.usage.has_usage:
            database.record_codex_token_usage(
                research["id"],
                "brief",
                call_id=f"{research['id']}:brief",
                input_tokens=brief.usage.input_tokens,
                cached_input_tokens=brief.usage.cached_input_tokens,
                output_tokens=brief.usage.output_tokens,
                reasoning_output_tokens=brief.usage.reasoning_output_tokens,
            )
        database.add_log(
            research["id"],
            "research_created",
            (
                "Research created for an explicit immediate start."
                if payload.auto_start and not payload.schedule_start_time
                else "Research created and added to the persistent GPU queue."
            ),
        )
        database.add_log(
            research["id"],
            "researcher_model_selected",
            f"{researcher_model_id} will plan and interpret this Lab research.",
            data={
                "researcher_model_id": researcher_model_id,
                "provider": (
                    "ollama" if researcher_model_id.startswith("ollama:") else "codex"
                ),
            },
        )
        if payload.research_type == "local_model_benchmark":
            model_log_data = {
                "model_id": values["model_id"],
                "model_digest": values["model_digest"],
                "benchmark_profile": values["benchmark_profile"],
                "provider": values["model_runtime"],
            }
            if values["model_id"] == BEYEFENDI_V2_MODEL_ID:
                model_log_data.update(
                    {
                        "source_url": BEYEFENDI_V2_SOURCE_URL,
                        "base_model": BEYEFENDI_V2_BASE_MODEL,
                        "base_revision": BEYEFENDI_V2_BASE_REVISION,
                    }
                )
            else:
                model_log_data["ollama_endpoint"] = _ollama(request).base_url
            database.add_log(
                research["id"],
                "model_selected",
                (
                    f"Pinned {values['model_id']} at digest "
                    f"{str(values['model_digest'])[:12]} for a reproducible local benchmark."
                ),
                data={
                    **model_log_data,
                },
            )
        elif brief and brief.warning:
            database.add_log(
                research["id"],
                "research_brief_fallback",
                brief.warning,
                level="warning",
            )
        elif brief and brief.used_ai:
            database.add_log(
                research["id"],
                "research_brief_generated",
                "Codex converted the original prompt into the persisted title and measurable objective.",
            )
        if payload.auto_start and not payload.schedule_start_time:
            try:
                return _supervisor(request).start(research["id"])
            except Exception as exc:
                raise _control_error(exc) from exc
        database.add_log(
            research["id"],
            "research_queued",
            "Research is waiting for its GPU source to become available.",
        )
        _supervisor(request).notify_queue()
        return database.get_research(research["id"], detail=True) or research

    @app.get("/api/researches/queue")
    def list_research_queue(
        request: Request,
        gpu_source_id: str | None = Query(default=None),
    ) -> list[dict[str, Any]]:
        database = _database(request)
        if gpu_source_id is not None and database.get_gpu_source(gpu_source_id) is None:
            raise _not_found("GPU source")
        return database.list_queued_researches(gpu_source_id)

    @app.post("/api/researches/queue/start-next")
    def start_next_research(
        request: Request,
        gpu_source_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        database = _database(request)
        if gpu_source_id is not None and database.get_gpu_source(gpu_source_id) is None:
            raise _not_found("GPU source")
        try:
            return {"started": _supervisor(request).start_next(gpu_source_id)}
        except Exception as exc:
            raise _control_error(exc) from exc

    @app.get("/api/researches/{research_id}")
    def get_research(
        research_id: str,
        request: Request,
        history: bool = Query(default=True),
    ) -> dict[str, Any]:
        database = _database(request)
        research = database.get_research_snapshot(
            research_id, include_history=history
        )
        if research is None:
            raise _not_found("Research")
        research["gpu_source"] = database.get_gpu_source(research["gpu_source_id"])
        research["runtime"] = _supervisor(request).runtime_snapshot(research_id)
        return research

    @app.patch("/api/researches/{research_id}")
    def update_research(
        research_id: str, payload: ResearchUpdate, request: Request
    ) -> dict[str, Any]:
        database = _database(request)
        current = database.get_research(research_id)
        if current is None:
            raise _not_found("Research")
        if current["status"] in {"queued", "running", "paused"}:
            raise HTTPException(
                status_code=409,
                detail="Dequeue or stop the research before changing its execution settings",
            )
        values = payload.model_dump(exclude_unset=True)
        if (
            values.get("gpu_source_id")
            and database.get_gpu_source(values["gpu_source_id"]) is None
        ):
            raise _not_found("GPU source")
        return database.update_research(research_id, values) or current

    @app.delete("/api/researches/{research_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_research(research_id: str, request: Request) -> Response:
        try:
            _supervisor(request).delete(research_id)
        except Exception as exc:
            raise _control_error(exc) from exc
        return Response(status_code=204)

    @app.post("/api/researches/{research_id}/start")
    def start_research(research_id: str, request: Request) -> dict[str, Any]:
        try:
            return _supervisor(request).start(research_id)
        except Exception as exc:
            raise _control_error(exc) from exc

    @app.post("/api/researches/{research_id}/enqueue")
    def enqueue_research(research_id: str, request: Request) -> dict[str, Any]:
        try:
            return _supervisor(request).enqueue(research_id)
        except Exception as exc:
            raise _control_error(exc) from exc

    @app.post("/api/researches/{research_id}/dequeue")
    def dequeue_research(research_id: str, request: Request) -> dict[str, Any]:
        try:
            return _supervisor(request).dequeue(research_id)
        except Exception as exc:
            raise _control_error(exc) from exc

    @app.post("/api/researches/{research_id}/pause")
    def pause_research(research_id: str, request: Request) -> dict[str, Any]:
        try:
            return _supervisor(request).pause(research_id)
        except Exception as exc:
            raise _control_error(exc) from exc

    @app.post("/api/researches/{research_id}/resume")
    def resume_research(research_id: str, request: Request) -> dict[str, Any]:
        try:
            return _supervisor(request).resume(research_id)
        except Exception as exc:
            raise _control_error(exc) from exc

    @app.post("/api/researches/{research_id}/stop")
    def stop_research(research_id: str, request: Request) -> dict[str, Any]:
        try:
            return _supervisor(request).stop(research_id)
        except Exception as exc:
            raise _control_error(exc) from exc

    @app.get("/api/researches/{research_id}/experiments")
    def list_experiments(
        research_id: str,
        request: Request,
        after_number: int = Query(default=0, ge=0),
        limit: int = Query(default=80, ge=1, le=300),
    ) -> list[dict[str, Any]]:
        database = _database(request)
        if database.get_research(research_id) is None:
            raise _not_found("Research")
        return database.list_experiments_after(
            research_id, after_number=after_number, limit=limit
        )

    @app.get("/api/researches/{research_id}/logs")
    def list_logs(
        research_id: str,
        request: Request,
        after_id: int = Query(default=0, ge=0),
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        database = _database(request)
        if database.get_research(research_id) is None:
            raise _not_found("Research")
        return database.list_logs(research_id, after_id=after_id, limit=limit)

    @app.get("/api/researches/{research_id}/events")
    async def research_events(
        research_id: str,
        request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
        after_id: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        database = _database(request)
        if database.get_research(research_id) is None:
            raise _not_found("Research")
        try:
            cursor = max(after_id, int(last_event_id or 0))
        except ValueError:
            cursor = after_id

        async def stream() -> AsyncIterator[str]:
            nonlocal cursor
            quiet_ticks = 0
            while not await request.is_disconnected():
                events = await asyncio.to_thread(
                    database.list_logs, research_id, after_id=cursor, limit=200
                )
                if events:
                    quiet_ticks = 0
                    for event in events:
                        cursor = int(event["id"])
                        payload = json.dumps(
                            database.compact_log(event),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        yield f"id: {cursor}\ndata: {payload}\n\n"
                else:
                    quiet_ticks += 1
                    if quiet_ticks >= 15:
                        quiet_ticks = 0
                        yield ": keep-alive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/telemetry/stream")
    async def telemetry_stream(source_id: str, request: Request) -> StreamingResponse:
        source = _database(request).get_gpu_source(source_id)
        if source is None:
            raise _not_found("GPU source")

        async def stream() -> AsyncIterator[str]:
            while not await request.is_disconnected():
                sample = await asyncio.to_thread(_telemetry(request).sample, source)
                payload = json.dumps(sample, ensure_ascii=False, separators=(",", ":"))
                yield f"data: {payload}\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/articles")
    def list_articles(request: Request) -> list[dict[str, Any]]:
        articles = _database(request).list_articles()
        for article in articles:
            article["article_kind"] = article_kind(article)
            article.pop("original_prompt", None)
        return articles

    @app.get("/api/articles/{article_identifier}")
    def get_article(article_identifier: str, request: Request) -> dict[str, Any]:
        database = _database(request)
        article = database.get_article(article_identifier)
        if article is None:
            raise _not_found("Article")
        research = database.get_research(article["research_id"], detail=True)
        if research is not None:
            article.update(
                {
                    "objective": research["objective"],
                    "baseline_value": research["baseline_value"],
                    "metric_direction": research["metric_direction"],
                    "research_type": research.get("research_type"),
                    "model_id": research.get("model_id"),
                    "experiments": research["experiments"],
                    "article_kind": article_kind(research),
                }
            )
        return article

    @app.post("/api/researches/{research_id}/article")
    def generate_article(research_id: str, request: Request) -> dict[str, Any]:
        database = _database(request)
        research = database.get_research(research_id, detail=False)
        if research is None:
            raise _not_found("Research")
        experiments = database.list_experiments(research_id)
        logs = database.list_logs(research_id, limit=1_000_000)
        markdown = request.app.state.article_generator.generate(
            research, experiments, logs
        )
        article = database.upsert_article(research_id, research["title"], markdown)
        article["article_kind"] = article_kind(research)
        database.add_log(
            research_id,
            "article_generated",
            (
                "Reader-focused article regenerated from the original question and "
                "relevant persisted evidence."
            ),
        )
        return article

    return app


app = create_app()
