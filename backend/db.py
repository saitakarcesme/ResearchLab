from __future__ import annotations

import json
import re
import sqlite3
import threading
import unicodedata
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


_TURKISH_ASCII_TRANSLATION = str.maketrans(
    {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "Ç": "C",
        "Ğ": "G",
        "İ": "I",
        "Ö": "O",
        "Ş": "S",
        "Ü": "U",
    }
)


def article_title_slug(title: str) -> str:
    """Return a readable, URL-safe slug while retaining non-Latin scripts."""

    normalized = unicodedata.normalize(
        "NFKD", title.translate(_TURKISH_ASCII_TRANSLATION)
    )
    parts: list[str] = []
    needs_separator = False
    for character in normalized.casefold():
        if unicodedata.combining(character):
            continue
        if character.isalnum():
            if needs_separator and parts:
                parts.append("-")
            parts.append(character)
            needs_separator = False
        else:
            needs_separator = True
    return "".join(parts).strip("-") or "article"


def article_slug_map(articles: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Assign stable slugs in creation order, disambiguating later collisions."""

    ordered = sorted(
        articles,
        key=lambda article: (str(article.get("created_at", "")), str(article["id"])),
    )
    assigned: dict[str, str] = {}
    used: set[str] = set()
    for article in ordered:
        article_id = str(article["id"])
        base = article_title_slug(str(article.get("title") or ""))
        candidate = base
        if candidate in used:
            identity = "".join(
                character
                for character in article_id.casefold()
                if character.isalnum()
            )
            identity = identity or "article"
            prefix_length = min(8, len(identity))
            while True:
                candidate = f"{base}-{identity[:prefix_length]}"
                if candidate not in used:
                    break
                if prefix_length < len(identity):
                    prefix_length = min(prefix_length + 4, len(identity))
                    continue
                suffix = 2
                while f"{candidate}-{suffix}" in used:
                    suffix += 1
                candidate = f"{candidate}-{suffix}"
                break
        assigned[article_id] = candidate
        used.add(candidate)
    return assigned


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS gpu_sources (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('local', 'remote')),
    host TEXT,
    port INTEGER NOT NULL DEFAULT 22 CHECK (port BETWEEN 1 AND 65535),
    username TEXT,
    auth_method TEXT NOT NULL DEFAULT 'agent' CHECK (auth_method IN ('agent', 'key_env')),
    workspace_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (
      (type = 'local') OR
      (type = 'remote' AND host IS NOT NULL AND username IS NOT NULL AND workspace_path IS NOT NULL)
    )
);

CREATE TABLE IF NOT EXISTS researches (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    original_prompt TEXT NOT NULL,
    objective TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'paused', 'completed', 'stopped', 'failed')),
    metric_name TEXT NOT NULL,
    metric_direction TEXT NOT NULL CHECK (metric_direction IN ('lower_is_better', 'higher_is_better')),
    baseline_value REAL,
    best_value REAL,
    best_git_commit TEXT,
    gpu_source_id TEXT NOT NULL REFERENCES gpu_sources(id) ON DELETE RESTRICT,
    target_gpu_allocation INTEGER NOT NULL CHECK (target_gpu_allocation BETWEEN 1 AND 100),
    workspace_path TEXT,
    adapter_type TEXT NOT NULL DEFAULT 'karpathy_autoresearch',
    research_type TEXT NOT NULL DEFAULT 'training_optimization',
    model_id TEXT,
    model_digest TEXT,
    model_runtime TEXT,
    benchmark_profile TEXT,
    researcher_model_id TEXT,
    schedule_start_time TEXT,
    schedule_end_time TEXT,
    schedule_timezone TEXT,
    schedule_utc_offset_minutes INTEGER,
    queued_at TEXT,
    queue_order INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    research_id TEXT NOT NULL REFERENCES researches(id) ON DELETE CASCADE,
    experiment_number INTEGER NOT NULL,
    hypothesis TEXT NOT NULL,
    change_summary TEXT NOT NULL,
    metric_value REAL,
    previous_best REAL,
    accepted INTEGER CHECK (accepted IN (0, 1) OR accepted IS NULL),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    git_commit TEXT,
    error TEXT,
    token_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(research_id, experiment_number)
);

CREATE TABLE IF NOT EXISTS research_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    research_id TEXT NOT NULL REFERENCES researches(id) ON DELETE CASCADE,
    experiment_id TEXT REFERENCES experiments(id) ON DELETE SET NULL,
    level TEXT NOT NULL CHECK (level IN ('info', 'warning', 'error')),
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    data_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    research_id TEXT NOT NULL REFERENCES researches(id) ON DELETE CASCADE,
    call_id TEXT NOT NULL UNIQUE,
    phase TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_output_tokens INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    research_id TEXT NOT NULL UNIQUE REFERENCES researches(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    markdown TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_researches_status ON researches(status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_experiments_research ON experiments(research_id, experiment_number);
CREATE INDEX IF NOT EXISTS idx_logs_research ON research_logs(research_id, id);
CREATE INDEX IF NOT EXISTS idx_token_usage_research ON research_token_usage(research_id, id);
CREATE INDEX IF NOT EXISTS idx_articles_updated ON articles(updated_at DESC);
"""


class ClosingConnection(sqlite3.Connection):
    """sqlite3's default context manager commits but does not close the file."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc_value, traceback))
        finally:
            self.close()


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._write_lock = threading.RLock()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=30,
            check_same_thread=False,
            factory=ClosingConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock, self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(SCHEMA)
            research_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(researches)"
                ).fetchall()
            }
            if "best_git_commit" not in research_columns:
                connection.execute(
                    "ALTER TABLE researches ADD COLUMN best_git_commit TEXT"
                )
                research_columns.add("best_git_commit")
            table_sql_row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'researches'"
            ).fetchone()
            table_sql = str(table_sql_row["sql"] if table_sql_row else "")
            if "'queued'" not in table_sql:
                self._migrate_researches_for_queue(
                    connection, research_columns, table_sql
                )
            else:
                if "queued_at" not in research_columns:
                    connection.execute(
                        "ALTER TABLE researches ADD COLUMN queued_at TEXT"
                    )
                if "queue_order" not in research_columns:
                    connection.execute(
                        "ALTER TABLE researches ADD COLUMN queue_order INTEGER"
                    )
            research_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(researches)"
                ).fetchall()
            }
            model_columns = {
                "research_type": "TEXT NOT NULL DEFAULT 'training_optimization'",
                "model_id": "TEXT",
                "model_digest": "TEXT",
                "model_runtime": "TEXT",
                "benchmark_profile": "TEXT",
                "researcher_model_id": "TEXT",
                "schedule_start_time": "TEXT",
                "schedule_end_time": "TEXT",
                "schedule_timezone": "TEXT",
                "schedule_utc_offset_minutes": "INTEGER",
            }
            for column, definition in model_columns.items():
                if column not in research_columns:
                    connection.execute(
                        f"ALTER TABLE researches ADD COLUMN {column} {definition}"
                    )
            experiment_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(experiments)"
                ).fetchall()
            }
            if "token_count" not in experiment_columns:
                connection.execute(
                    "ALTER TABLE experiments ADD COLUMN token_count INTEGER NOT NULL DEFAULT 0"
                )
            self._backfill_experiment_token_counts(connection)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_researches_queue "
                "ON researches(status, queue_order, queued_at)"
            )
            connection.execute("PRAGMA optimize")

    @staticmethod
    def _token_count_from_log_data(data: Mapping[str, Any]) -> int:
        total_tokens_m = data.get("total_tokens_M")
        if total_tokens_m is not None:
            try:
                return max(0, round(float(total_tokens_m) * 1_000_000))
            except (TypeError, ValueError):
                return 0

        measurements = data.get("measurements")
        if isinstance(measurements, list):
            return sum(
                max(0, int(item.get("prompt_tokens") or 0))
                + max(0, int(item.get("output_tokens") or 0))
                for item in measurements
                if isinstance(item, Mapping)
            )

        repeats = data.get("repeats")
        if not isinstance(repeats, list):
            return 0
        try:
            batch_size = max(1, int(data.get("batch_size") or 1))
        except (TypeError, ValueError):
            batch_size = 1

        samples = [item for item in repeats if isinstance(item, Mapping)]
        warmup = data.get("warmup")
        if isinstance(warmup, Mapping):
            samples.append(warmup)
        total = 0
        for sample in samples:
            try:
                input_tokens = max(
                    0, int(sample.get("input_tokens_per_request") or 0)
                )
                output_tokens = max(0, int(sample.get("total_output_tokens") or 0))
            except (TypeError, ValueError):
                continue
            total += input_tokens * batch_size + output_tokens
        return total

    @classmethod
    def _backfill_experiment_token_counts(
        cls, connection: sqlite3.Connection
    ) -> None:
        rows = connection.execute(
            """SELECT experiment_id, data_json FROM research_logs
               WHERE experiment_id IS NOT NULL
                 AND event_type IN ('experiment_completed', 'experiment_accepted', 'experiment_rejected')
               ORDER BY created_at"""
        ).fetchall()
        totals: dict[str, int] = {}
        for row in rows:
            try:
                data = json.loads(row["data_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(data, Mapping):
                continue
            count = cls._token_count_from_log_data(data)
            experiment_id = str(row["experiment_id"])
            totals[experiment_id] = max(totals.get(experiment_id, 0), count)
        connection.executemany(
            """UPDATE experiments SET token_count = ?
               WHERE id = ? AND COALESCE(token_count, 0) = 0""",
            ((count, experiment_id) for experiment_id, count in totals.items() if count),
        )

    @staticmethod
    def _migrate_researches_for_queue(
        connection: sqlite3.Connection,
        research_columns: set[str],
        table_sql: str,
    ) -> None:
        """Rebuild the table while retaining columns added by other features."""

        migration_sql = re.sub(
            r"^\s*CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:[\"`]researches[\"`]|researches)",
            "CREATE TABLE researches_queue_migration",
            table_sql,
            count=1,
            flags=re.IGNORECASE,
        )
        migration_sql = migration_sql.replace(
            "status IN ('running'",
            "status IN ('queued', 'running'",
            1,
        )
        if (
            not migration_sql.startswith("CREATE TABLE researches_queue_migration")
            or "'queued'" not in migration_sql
        ):
            raise RuntimeError("Could not prepare the research queue schema migration")

        if connection.in_transaction:
            connection.commit()
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(migration_sql)
            if "queued_at" not in research_columns:
                connection.execute(
                    "ALTER TABLE researches_queue_migration ADD COLUMN queued_at TEXT"
                )
            if "queue_order" not in research_columns:
                connection.execute(
                    "ALTER TABLE researches_queue_migration ADD COLUMN queue_order INTEGER"
                )
            copied_columns = ", ".join(
                f'"{column.replace(chr(34), chr(34) * 2)}"'
                for column in research_columns
            )
            connection.execute(
                f"INSERT INTO researches_queue_migration ({copied_columns}) "
                f"SELECT {copied_columns} FROM researches"
            )
            connection.execute("DROP TABLE researches")
            connection.execute(
                "ALTER TABLE researches_queue_migration RENAME TO researches"
            )
            connection.execute(
                "CREATE INDEX idx_researches_status "
                "ON researches(status, updated_at DESC)"
            )
            connection.execute(
                "CREATE INDEX idx_researches_queue "
                "ON researches(status, queue_order, queued_at)"
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.execute("PRAGMA foreign_keys = ON")
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError("Research queue migration left invalid foreign keys")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock, self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    @staticmethod
    def _record(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        record = dict(row)
        if "accepted" in record and record["accepted"] is not None:
            record["accepted"] = bool(record["accepted"])
        if "data_json" in record:
            raw = record.pop("data_json")
            record["data"] = json.loads(raw) if raw else None
        return record

    def ensure_local_gpu_source(self, name: str) -> dict[str, Any]:
        now = utc_now()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM gpu_sources WHERE type = 'local' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                source_id = "local"
                connection.execute(
                    """INSERT INTO gpu_sources
                       (id, name, type, host, port, username, auth_method, workspace_path, created_at, updated_at)
                       VALUES (?, ?, 'local', NULL, 22, NULL, 'agent', NULL, ?, ?)""",
                    (source_id, name, now, now),
                )
                row = connection.execute(
                    "SELECT * FROM gpu_sources WHERE id = ?", (source_id,)
                ).fetchone()
            elif (
                name
                and row["name"] in {"Local NVIDIA GPU", "Local GPU"}
                and row["name"] != name
            ):
                connection.execute(
                    "UPDATE gpu_sources SET name = ?, updated_at = ? WHERE id = ?",
                    (name, now, row["id"]),
                )
                row = connection.execute(
                    "SELECT * FROM gpu_sources WHERE id = ?", (row["id"],)
                ).fetchone()
        return self._record(row) or {}

    def list_gpu_sources(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM gpu_sources ORDER BY type, created_at"
            ).fetchall()
        return [self._record(row) or {} for row in rows]

    def get_gpu_source(self, source_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM gpu_sources WHERE id = ?", (source_id,)
            ).fetchone()
        return self._record(row)

    def create_gpu_source(self, values: Mapping[str, Any]) -> dict[str, Any]:
        source_id, now = str(uuid.uuid4()), utc_now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO gpu_sources
                   (id, name, type, host, port, username, auth_method, workspace_path, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    source_id,
                    values["name"],
                    values["type"],
                    values.get("host"),
                    values.get("port", 22),
                    values.get("username"),
                    values.get("auth_method", "agent"),
                    values.get("workspace_path"),
                    now,
                    now,
                ),
            )
        return self.get_gpu_source(source_id) or {}

    def update_gpu_source(
        self, source_id: str, values: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        allowed = {"name", "host", "port", "username", "auth_method", "workspace_path"}
        updates = {key: value for key, value in values.items() if key in allowed}
        if not updates:
            return self.get_gpu_source(source_id)
        updates["updated_at"] = utc_now()
        assignments = ", ".join(f"{column} = ?" for column in updates)
        with self.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE gpu_sources SET {assignments} WHERE id = ?",
                (*updates.values(), source_id),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_gpu_source(source_id)

    def delete_gpu_source(self, source_id: str) -> bool:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT type FROM gpu_sources WHERE id = ?", (source_id,)
            ).fetchone()
            if row is None:
                return False
            if row["type"] == "local":
                raise ValueError("The default local GPU source cannot be deleted")
            connection.execute("DELETE FROM gpu_sources WHERE id = ?", (source_id,))
        return True

    def create_research(self, values: Mapping[str, Any]) -> dict[str, Any]:
        research_id, now = str(uuid.uuid4()), utc_now()
        initial_status = str(values.get("status", "stopped"))
        if initial_status not in {"queued", "stopped"}:
            raise ValueError("New researches must start queued or stopped")
        queued_at = now if initial_status == "queued" else None
        with self.transaction() as connection:
            queue_order = (
                connection.execute(
                    "SELECT COALESCE(MAX(queue_order), 0) + 1 FROM researches"
                ).fetchone()[0]
                if initial_status == "queued"
                else None
            )
            connection.execute(
                """INSERT INTO researches
                   (id, title, original_prompt, objective, status, metric_name, metric_direction,
                    baseline_value, best_value, gpu_source_id, target_gpu_allocation, workspace_path,
                    adapter_type, research_type, model_id, model_digest, model_runtime, benchmark_profile,
                    researcher_model_id, schedule_start_time, schedule_end_time, schedule_timezone, schedule_utc_offset_minutes,
                    queued_at, queue_order, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    research_id,
                    values["title"],
                    values["original_prompt"],
                    values["objective"],
                    initial_status,
                    values.get("metric_name", "val_bpb"),
                    values.get("metric_direction", "lower_is_better"),
                    values["gpu_source_id"],
                    values["target_gpu_allocation"],
                    values.get("adapter_type", "karpathy_autoresearch"),
                    values.get("research_type", "training_optimization"),
                    values.get("model_id"),
                    values.get("model_digest"),
                    values.get("model_runtime"),
                    values.get("benchmark_profile"),
                    values.get("researcher_model_id"),
                    values.get("schedule_start_time"),
                    values.get("schedule_end_time"),
                    values.get("schedule_timezone"),
                    values.get("schedule_utc_offset_minutes"),
                    queued_at,
                    queue_order,
                    now,
                    now,
                ),
            )
        return self.get_research(research_id, detail=True) or {}

    def _research_select(self) -> str:
        return """
            SELECT r.*, g.name AS gpu_name, g.type AS gpu_type,
                   COUNT(DISTINCT e.id) AS experiment_count,
                   COALESCE(SUM(e.token_count), 0) AS training_tokens,
                   COALESCE((SELECT SUM(u.input_tokens) FROM research_token_usage u
                             WHERE u.research_id = r.id), 0) AS input_tokens,
                   COALESCE((SELECT SUM(u.cached_input_tokens) FROM research_token_usage u
                             WHERE u.research_id = r.id), 0) AS cached_input_tokens,
                   COALESCE((SELECT SUM(u.output_tokens) FROM research_token_usage u
                             WHERE u.research_id = r.id), 0) AS output_tokens,
                   COALESCE((SELECT SUM(u.reasoning_output_tokens) FROM research_token_usage u
                             WHERE u.research_id = r.id), 0) AS reasoning_output_tokens,
                   COALESCE((SELECT SUM(u.input_tokens + u.output_tokens)
                             FROM research_token_usage u WHERE u.research_id = r.id), 0)
                             AS total_tokens
            FROM researches r
            JOIN gpu_sources g ON g.id = r.gpu_source_id
            LEFT JOIN experiments e ON e.research_id = r.id
        """

    def record_codex_token_usage(
        self,
        research_id: str,
        phase: str,
        *,
        call_id: str,
        input_tokens: int,
        cached_input_tokens: int = 0,
        output_tokens: int,
        reasoning_output_tokens: int = 0,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO research_token_usage
                   (research_id, call_id, phase, input_tokens, cached_input_tokens,
                    output_tokens, reasoning_output_tokens, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    research_id,
                    call_id,
                    phase[:80],
                    max(0, int(input_tokens)),
                    max(0, int(cached_input_tokens)),
                    max(0, int(output_tokens)),
                    max(0, int(reasoning_output_tokens)),
                    utc_now(),
                ),
            )

    def list_researches(
        self, status: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        where, params = ("WHERE r.status = ?", [status]) if status else ("", [])
        query = (
            self._research_select()
            + f" {where} GROUP BY r.id ORDER BY r.updated_at DESC LIMIT ? OFFSET ?"
        )
        with self.connect() as connection:
            rows = connection.execute(query, (*params, limit, offset)).fetchall()
        researches = [self._record(row) or {} for row in rows]
        if any(item.get("status") == "queued" for item in researches):
            positions = {
                item["id"]: item["queue_position"]
                for item in self.list_queued_researches()
            }
            for research in researches:
                if research.get("status") == "queued":
                    research["queue_position"] = positions.get(research["id"])
        return researches

    def list_queued_researches(
        self, gpu_source_id: str | None = None
    ) -> list[dict[str, Any]]:
        where = "WHERE r.status = 'queued'"
        params: list[Any] = []
        if gpu_source_id is not None:
            where += " AND r.gpu_source_id = ?"
            params.append(gpu_source_id)
        query = (
            self._research_select()
            + f" {where} GROUP BY r.id "
            "ORDER BY r.queue_order ASC, r.queued_at ASC, r.created_at ASC, r.id ASC"
        )
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        queued = [self._record(row) or {} for row in rows]
        for position, research in enumerate(queued, start=1):
            research["queue_position"] = position
        return queued

    def list_gpu_reservations(self) -> list[dict[str, Any]]:
        """Return every running or paused research without UI pagination limits."""

        query = (
            self._research_select()
            + " WHERE r.status IN ('running', 'paused')"
            + " GROUP BY r.id ORDER BY r.updated_at DESC"
        )
        with self.connect() as connection:
            rows = connection.execute(query).fetchall()
        return [self._record(row) or {} for row in rows]

    def list_process_cleanup_candidates(self) -> list[dict[str, Any]]:
        """Return records whose last process termination may be uncertain."""

        query = (
            self._research_select()
            + " WHERE r.status IN ('running', 'paused')"
            + " OR (r.status = 'failed' AND EXISTS ("
            + "SELECT 1 FROM research_logs cleanup_log "
            + "WHERE cleanup_log.research_id = r.id "
            + "AND cleanup_log.event_type IN "
            + "('research_control_error', 'stale_process_warning')))"
            + " GROUP BY r.id ORDER BY r.updated_at DESC"
        )
        with self.connect() as connection:
            rows = connection.execute(query).fetchall()
        return [self._record(row) or {} for row in rows]

    def get_research(
        self, research_id: str, *, detail: bool = False
    ) -> dict[str, Any] | None:
        query = self._research_select() + " WHERE r.id = ? GROUP BY r.id"
        with self.connect() as connection:
            row = connection.execute(query, (research_id,)).fetchone()
        record = self._record(row)
        if record is not None and record.get("status") == "queued":
            record["queue_position"] = next(
                (
                    item["queue_position"]
                    for item in self.list_queued_researches()
                    if item["id"] == research_id
                ),
                None,
            )
        if record is not None and detail:
            record["experiments"] = self.list_experiments(research_id)
            record["logs"] = self.list_recent_logs(research_id, limit=200)
        return record

    def update_research(
        self, research_id: str, values: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        allowed = {
            "title",
            "objective",
            "status",
            "metric_name",
            "metric_direction",
            "baseline_value",
            "best_value",
            "best_git_commit",
            "gpu_source_id",
            "target_gpu_allocation",
            "workspace_path",
            "researcher_model_id",
            "schedule_start_time",
            "schedule_end_time",
            "schedule_timezone",
            "schedule_utc_offset_minutes",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        if not updates:
            return self.get_research(research_id, detail=True)
        if "status" in updates:
            updates["queued_at"] = (
                utc_now() if updates["status"] == "queued" else None
            )
            if updates["status"] != "queued":
                updates["queue_order"] = None
        updates["updated_at"] = utc_now()
        assignments = ", ".join(f"{column} = ?" for column in updates)
        with self.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE researches SET {assignments} WHERE id = ?",
                (*updates.values(), research_id),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_research(research_id, detail=True)

    def transition_research(
        self, research_id: str, from_statuses: Sequence[str], to_status: str
    ) -> bool:
        placeholders = ",".join("?" for _ in from_statuses)
        with self.transaction() as connection:
            queued_at = utc_now() if to_status == "queued" else None
            queue_order = (
                connection.execute(
                    "SELECT COALESCE(MAX(queue_order), 0) + 1 FROM researches"
                ).fetchone()[0]
                if to_status == "queued"
                else None
            )
            cursor = connection.execute(
                f"UPDATE researches SET status = ?, queued_at = ?, queue_order = ?, updated_at = ? "
                f"WHERE id = ? AND status IN ({placeholders})",
                (
                    to_status,
                    queued_at,
                    queue_order,
                    utc_now(),
                    research_id,
                    *from_statuses,
                ),
            )
        return cursor.rowcount == 1

    def delete_research(self, research_id: str) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM researches WHERE id = ? AND status NOT IN ('running', 'paused')",
                (research_id,),
            )
        return cursor.rowcount == 1

    def recover_interrupted_researches(self) -> int:
        now = utc_now()
        with self.transaction() as connection:
            rows = connection.execute(
                "SELECT id FROM researches WHERE status = 'running'"
            ).fetchall()
            for row in rows:
                incomplete = connection.execute(
                    """SELECT experiment_number FROM experiments
                       WHERE research_id = ? AND completed_at IS NULL""",
                    (row["id"],),
                ).fetchall()
                connection.execute(
                    """UPDATE experiments
                       SET accepted = 0, completed_at = ?,
                           error = COALESCE(error, 'Backend stopped before the experiment completed')
                       WHERE research_id = ? AND completed_at IS NULL""",
                    (now, row["id"]),
                )
                connection.execute(
                    "UPDATE researches SET status = 'paused', updated_at = ? WHERE id = ?",
                    (now, row["id"]),
                )
                connection.execute(
                    """INSERT INTO research_logs
                       (research_id, experiment_id, level, event_type, message, data_json, created_at)
                       VALUES (?, NULL, 'warning', 'service_restarted', ?, NULL, ?)""",
                    (
                        row["id"],
                        "Backend restarted; research is paused and can be resumed safely.",
                        now,
                    ),
                )
                if incomplete:
                    numbers = [item["experiment_number"] for item in incomplete]
                    connection.execute(
                        """INSERT INTO research_logs
                           (research_id, experiment_id, level, event_type, message, data_json, created_at)
                           VALUES (?, NULL, 'error', 'experiment_recovered', ?, ?, ?)""",
                        (
                            row["id"],
                            "Incomplete experiment records were closed after backend restart.",
                            json.dumps(
                                {"experiment_numbers": numbers}, separators=(",", ":")
                            ),
                            now,
                        ),
                    )
        return len(rows)

    def close_incomplete_experiments(self, research_id: str, reason: str) -> list[int]:
        now = utc_now()
        with self.transaction() as connection:
            rows = connection.execute(
                """SELECT experiment_number FROM experiments
                   WHERE research_id = ? AND completed_at IS NULL
                   ORDER BY experiment_number""",
                (research_id,),
            ).fetchall()
            connection.execute(
                """UPDATE experiments SET accepted = 0, completed_at = ?, error = COALESCE(error, ?)
                   WHERE research_id = ? AND completed_at IS NULL""",
                (now, reason, research_id),
            )
        return [int(row["experiment_number"]) for row in rows]

    def next_experiment_number(self, research_id: str) -> int:
        with self.connect() as connection:
            value = connection.execute(
                "SELECT COALESCE(MAX(experiment_number), 0) + 1 FROM experiments WHERE research_id = ?",
                (research_id,),
            ).fetchone()[0]
        return int(value)

    def create_experiment(
        self,
        research_id: str,
        hypothesis: str,
        change_summary: str,
        previous_best: float | None,
    ) -> dict[str, Any]:
        experiment_id, now = str(uuid.uuid4()), utc_now()
        number = self.next_experiment_number(research_id)
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO experiments
                   (id, research_id, experiment_number, hypothesis, change_summary, metric_value,
                    previous_best, accepted, started_at, completed_at, git_commit, error)
                   VALUES (?, ?, ?, ?, ?, NULL, ?, NULL, ?, NULL, NULL, NULL)""",
                (
                    experiment_id,
                    research_id,
                    number,
                    hypothesis,
                    change_summary,
                    previous_best,
                    now,
                ),
            )
        return self.get_experiment(experiment_id) or {}

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
        return self._record(row)

    def append_experiment_change_summary(
        self, experiment_id: str, summary: str
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """UPDATE experiments
                   SET change_summary = change_summary || ' | ' || ?
                   WHERE id = ?""",
                (summary[:1000], experiment_id),
            )

    def update_experiment_previous_best(
        self, experiment_id: str, previous_best: float | None
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """UPDATE experiments SET previous_best = ?
                   WHERE id = ? AND completed_at IS NULL""",
                (previous_best, experiment_id),
            )

    def update_experiment_token_count(
        self, experiment_id: str, token_count: int
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """UPDATE experiments SET token_count = MAX(token_count, ?)
                   WHERE id = ? AND completed_at IS NULL""",
                (max(0, int(token_count)), experiment_id),
            )

    def finish_experiment(
        self,
        experiment_id: str,
        *,
        metric_value: float | None,
        accepted: bool,
        git_commit: str | None,
        error: str | None = None,
        token_count: int | None = None,
        research_updates: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self.transaction() as connection:
            connection.execute(
                """UPDATE experiments SET metric_value = ?, accepted = ?, completed_at = ?,
                   git_commit = ?, error = ?,
                   token_count = CASE WHEN ? IS NULL THEN token_count ELSE MAX(token_count, ?) END
                   WHERE id = ?""",
                (
                    metric_value,
                    int(accepted),
                    utc_now(),
                    git_commit,
                    error,
                    max(0, int(token_count)) if token_count is not None else None,
                    max(0, int(token_count)) if token_count is not None else None,
                    experiment_id,
                ),
            )
            if research_updates:
                allowed = {"baseline_value", "best_value", "best_git_commit"}
                updates = {
                    key: value
                    for key, value in research_updates.items()
                    if key in allowed
                }
                if updates:
                    updates["updated_at"] = utc_now()
                    assignments = ", ".join(f"{column} = ?" for column in updates)
                    connection.execute(
                        f"""UPDATE researches SET {assignments}
                            WHERE id = (SELECT research_id FROM experiments WHERE id = ?)""",
                        (*updates.values(), experiment_id),
                    )
        return self.get_experiment(experiment_id)

    def list_experiments(self, research_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM experiments WHERE research_id = ? ORDER BY experiment_number",
                (research_id,),
            ).fetchall()
        return [self._record(row) or {} for row in rows]

    def add_log(
        self,
        research_id: str,
        event_type: str,
        message: str,
        *,
        level: str = "info",
        experiment_id: str | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                """INSERT INTO research_logs
                   (research_id, experiment_id, level, event_type, message, data_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    research_id,
                    experiment_id,
                    level,
                    event_type,
                    message[:2000],
                    json.dumps(data, ensure_ascii=False, separators=(",", ":"))
                    if data
                    else None,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM research_logs WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return self._record(row) or {}

    def list_logs(
        self, research_id: str, *, after_id: int = 0, limit: int = 200
    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM research_logs WHERE research_id = ? AND id > ?
                   ORDER BY id ASC LIMIT ?""",
                (research_id, after_id, limit),
            ).fetchall()
        return [self._record(row) or {} for row in rows]

    def list_recent_logs(
        self, research_id: str, *, limit: int = 200
    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM (
                       SELECT * FROM research_logs WHERE research_id = ? ORDER BY id DESC LIMIT ?
                   ) ORDER BY id ASC""",
                (research_id, limit),
            ).fetchall()
        return [self._record(row) or {} for row in rows]

    def upsert_article(
        self, research_id: str, title: str, markdown: str
    ) -> dict[str, Any]:
        now = utc_now()
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT id, created_at FROM articles WHERE research_id = ?",
                (research_id,),
            ).fetchone()
            if existing:
                article_id = existing["id"]
                connection.execute(
                    "UPDATE articles SET title = ?, markdown = ?, updated_at = ? WHERE id = ?",
                    (title, markdown, now, article_id),
                )
            else:
                article_id = str(uuid.uuid4())
                connection.execute(
                    """INSERT INTO articles (id, research_id, title, markdown, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (article_id, research_id, title, markdown, now, now),
                )
        return self.get_article(article_id) or {}

    def _publication_identities(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at FROM articles ORDER BY created_at ASC, id ASC"
            ).fetchall()
        return [self._record(row) or {} for row in rows]

    def _publication_slugs(self) -> dict[str, str]:
        return article_slug_map(self._publication_identities())

    def list_articles(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT a.id, a.research_id, a.title, a.created_at, a.updated_at,
                          r.status AS research_status, r.metric_name, r.best_value,
                          r.original_prompt, r.research_type, r.model_id
                   FROM articles a JOIN researches r ON r.id = a.research_id
                   ORDER BY a.updated_at DESC"""
            ).fetchall()
        records = [self._record(row) or {} for row in rows]
        slugs = self._publication_slugs()
        for record in records:
            record["slug"] = slugs[record["id"]]
        return records

    def get_article(self, article_identifier: str) -> dict[str, Any] | None:
        slugs = self._publication_slugs()
        article_id = article_identifier
        if article_identifier not in slugs:
            article_id = next(
                (
                    identifier
                    for identifier, slug in slugs.items()
                    if slug == article_identifier
                ),
                "",
            )
        with self.connect() as connection:
            row = connection.execute(
                """SELECT a.*, r.status AS research_status, r.metric_name, r.best_value,
                          r.research_type, r.model_id
                   FROM articles a JOIN researches r ON r.id = a.research_id WHERE a.id = ?""",
                (article_id,),
            ).fetchone()
        record = self._record(row)
        if record is not None:
            record["slug"] = slugs[record["id"]]
        return record

    def get_article_by_research(self, research_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id FROM articles WHERE research_id = ?", (research_id,)
            ).fetchone()
        return self.get_article(row["id"]) if row else None
