from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.article import generate_article_markdown
from backend.brief import CodexResearchBriefGenerator
from backend.config import Settings
from backend.db import Database
from backend.schemas import (
    GpuSourceCreate,
    GpuSourceUpdate,
    ResearchCreate,
    ResearchUpdate,
)
from backend.supervisor import ResearchSupervisor
from backend.telemetry import TelemetryService


def _database(request: Request) -> Database:
    return request.app.state.database


def _supervisor(request: Request) -> ResearchSupervisor:
    return request.app.state.supervisor


def _telemetry(request: Request) -> TelemetryService:
    return request.app.state.telemetry


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
        interrupted = database.list_researches(status="running")
        ResearchSupervisor.terminate_stale_process_groups(
            configured, database, interrupted
        )
        database.recover_interrupted_researches()
        supervisor = ResearchSupervisor(configured, database, telemetry)
        app.state.settings = configured
        app.state.database = database
        app.state.telemetry = telemetry
        app.state.supervisor = supervisor
        app.state.brief_generator = brief_generator or CodexResearchBriefGenerator(
            configured
        )
        try:
            yield
        finally:
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
        if database.get_gpu_source(source_id) is None:
            raise _not_found("GPU source")
        brief = request.app.state.brief_generator.generate(
            payload.original_prompt,
            supplied_title=payload.title,
            supplied_objective=payload.objective,
        )
        research = database.create_research(
            {
                "title": brief.title,
                "original_prompt": payload.original_prompt,
                "objective": brief.objective,
                "metric_name": "val_bpb",
                "metric_direction": "lower_is_better",
                "gpu_source_id": source_id,
                "target_gpu_allocation": payload.target_gpu_allocation,
                "adapter_type": "karpathy_autoresearch",
            }
        )
        database.add_log(
            research["id"],
            "research_created",
            "Research created. Real execution begins only after an explicit start request.",
        )
        if brief.warning:
            database.add_log(
                research["id"],
                "research_brief_fallback",
                brief.warning,
                level="warning",
            )
        elif brief.used_ai:
            database.add_log(
                research["id"],
                "research_brief_generated",
                "Codex converted the original prompt into the persisted title and measurable objective.",
            )
        if payload.auto_start:
            try:
                return _supervisor(request).start(research["id"])
            except Exception as exc:
                raise _control_error(exc) from exc
        return database.get_research(research["id"], detail=True) or research

    @app.get("/api/researches/{research_id}")
    def get_research(research_id: str, request: Request) -> dict[str, Any]:
        database = _database(request)
        research = database.get_research(research_id, detail=True)
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
        if current["status"] in {"running", "paused"}:
            raise HTTPException(
                status_code=409,
                detail="Stop the research before changing its execution settings",
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
        database = _database(request)
        if database.get_research(research_id) is None:
            raise _not_found("Research")
        if not database.delete_research(research_id):
            raise HTTPException(
                status_code=409,
                detail="Stop the research before deleting its database record",
            )
        return Response(status_code=204)

    @app.post("/api/researches/{research_id}/start")
    def start_research(research_id: str, request: Request) -> dict[str, Any]:
        try:
            return _supervisor(request).start(research_id)
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
    def list_experiments(research_id: str, request: Request) -> list[dict[str, Any]]:
        database = _database(request)
        if database.get_research(research_id) is None:
            raise _not_found("Research")
        return database.list_experiments(research_id)

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
    ) -> StreamingResponse:
        database = _database(request)
        if database.get_research(research_id) is None:
            raise _not_found("Research")
        try:
            cursor = max(0, int(last_event_id or 0))
        except ValueError:
            cursor = 0

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
                            event, ensure_ascii=False, separators=(",", ":")
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
        return _database(request).list_articles()

    @app.get("/api/articles/{article_id}")
    def get_article(article_id: str, request: Request) -> dict[str, Any]:
        article = _database(request).get_article(article_id)
        if article is None:
            raise _not_found("Article")
        return article

    @app.post("/api/researches/{research_id}/article")
    def generate_article(research_id: str, request: Request) -> dict[str, Any]:
        database = _database(request)
        research = database.get_research(research_id, detail=True)
        if research is None:
            raise _not_found("Research")
        markdown = generate_article_markdown(
            research, research["experiments"], research["logs"]
        )
        article = database.upsert_article(research_id, research["title"], markdown)
        database.add_log(
            research_id,
            "article_generated",
            "Article regenerated from persisted experiments and logs.",
        )
        return article

    return app


app = create_app()
