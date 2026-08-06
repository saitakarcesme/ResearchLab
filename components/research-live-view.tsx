"use client";

import { AnimatePresence, motion } from "framer-motion";
import { BookOpen, ListX, LoaderCircle, Pause, Play, Square, Trash2, X } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  controlResearch,
  createArticle,
  deleteResearch,
  getGpuTelemetry,
  getResearch,
  listResearchExperiments,
  normalizeTelemetryEvent,
  researchEventsUrl,
  telemetryEventsUrl,
} from "@/lib/api";
import { articlePath } from "@/lib/article-url";
import { formatMetric, formatTokenCount, metricDescription, metricLabel } from "@/lib/format";
import { modelDisplayName } from "@/lib/model-label";
import type { Article, Experiment, GpuTelemetry, Research, ResearchLog } from "@/lib/types";
import { GpuPerformance } from "./gpu-performance";
import { LiveLogs } from "./live-logs";
import { ProgressChart } from "./progress-chart";
import { AnimatedTokenCount, ElapsedResearchTime } from "./research-runtime-metrics";
import { StatusPill } from "./status-pill";

function exactTokens(value: number | null | undefined): string {
  return Math.max(0, Math.round(value ?? 0)).toLocaleString();
}

function tokenBreakdown(research: Research): string {
  const cached = Math.max(0, research.cached_input_tokens ?? 0);
  return `${exactTokens(research.input_tokens)} input${cached ? ` (${exactTokens(cached)} cached)` : ""} · ${exactTokens(research.output_tokens)} output`;
}

const MAX_LIVE_EXPERIMENTS = 160;
const MAX_LIVE_LOGS = 60;

function mergeExperiments(current: Experiment[], incoming: Experiment[]): Experiment[] {
  const merged = new Map(current.map((item) => [item.id, item]));
  for (const item of incoming) merged.set(item.id, item);
  return [...merged.values()]
    .sort((a, b) => a.experiment_number - b.experiment_number)
    .slice(-MAX_LIVE_EXPERIMENTS);
}

function mergeLogs(current: ResearchLog[], incoming: ResearchLog[]): ResearchLog[] {
  const merged = new Map(current.map((item) => [String(item.id), item]));
  for (const item of incoming) merged.set(String(item.id), item);
  return [...merged.values()]
    .sort((a, b) => Number(a.id) - Number(b.id))
    .slice(-MAX_LIVE_LOGS);
}

export function ResearchLiveView({
  researchId,
  initialResearch,
  monitorMode = false,
}: {
  researchId: string;
  initialResearch?: Research | null;
  monitorMode?: boolean;
}) {
  const router = useRouter();
  const [research, setResearch] = useState<Research | null>(initialResearch ?? null);
  const [telemetry, setTelemetry] = useState<GpuTelemetry | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [article, setArticle] = useState<Article | null>(null);
  const [error, setError] = useState<string | null>(null);
  const researchRef = useRef<Research | null>(initialResearch ?? null);
  const refreshController = useRef<AbortController | null>(null);
  const eventRefreshTimer = useRef<number | null>(null);
  const sseConnected = useRef(false);
  const deleting = useRef(false);
  const deleteTriggerRef = useRef<HTMLButtonElement>(null);
  const deleteDialogRef = useRef<HTMLElement>(null);

  const [pageVisible, setPageVisible] = useState(true);
  const [historyReady, setHistoryReady] = useState(Boolean(initialResearch?.logs));
  const nativePageVisible = useRef(true);

  useEffect(() => {
    researchRef.current = research;
  }, [research]);

  useEffect(() => {
    const updateVisibility = () => setPageVisible(
      document.visibilityState === "visible" && nativePageVisible.current,
    );
    const updateNativeVisibility = (event: Event) => {
      nativePageVisible.current = (event as CustomEvent<boolean>).detail !== false;
      updateVisibility();
    };
    updateVisibility();
    document.addEventListener("visibilitychange", updateVisibility);
    window.addEventListener("researchlab-visibility", updateNativeVisibility);
    return () => {
      document.removeEventListener("visibilitychange", updateVisibility);
      window.removeEventListener("researchlab-visibility", updateNativeVisibility);
    };
  }, []);

  const refresh = useCallback(async (history = false) => {
    refreshController.current?.abort();
    const controller = new AbortController();
    refreshController.current = controller;
    try {
      const next = await getResearch(researchId, { history, signal: controller.signal });
      setResearch((current) => ({
        ...next,
        experiments: next.experiments ?? current?.experiments ?? [],
        logs: next.logs ?? current?.logs ?? [],
      }));
      if (history) setHistoryReady(true);
      setError(null);
    } catch (nextError) {
      if (controller.signal.aborted) return;
      setError(nextError instanceof Error ? nextError.message : "Research data is unavailable.");
    }
  }, [researchId]);

  useEffect(() => {
    if (!pageVisible) return;
    const initial = window.setTimeout(() => void refresh(true), 0);
    const interval = window.setInterval(() => {
      if (!sseConnected.current) void refresh(!historyReady);
    }, 30_000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(interval);
      refreshController.current?.abort();
    };
  }, [historyReady, pageVisible, refresh]);

  useEffect(() => {
    if (!pageVisible || !historyReady) return;
    const lastLogId = Number(researchRef.current?.logs?.at(-1)?.id ?? 0);
    const events = new EventSource(researchEventsUrl(researchId, lastLogId));
    let disposed = false;
    let syncController: AbortController | null = null;
    const syncChanges = async () => {
      eventRefreshTimer.current = null;
      const snapshot = researchRef.current;
      const lastNumber = snapshot?.experiments?.at(-1)?.experiment_number ?? 0;
      syncController?.abort();
      const controller = new AbortController();
      syncController = controller;
      try {
        const [summary, experiments] = await Promise.all([
          getResearch(researchId, { history: false, signal: controller.signal }),
          listResearchExperiments(researchId, Math.max(0, lastNumber - 1), { signal: controller.signal }),
        ]);
        if (disposed) return;
        setResearch((current) => ({
          ...summary,
          experiments: mergeExperiments(current?.experiments ?? [], experiments),
          logs: current?.logs ?? [],
        }));
        setError(null);
      } catch (nextError) {
        if (controller.signal.aborted || disposed) return;
        setError(nextError instanceof Error ? nextError.message : "Live research update failed.");
      }
    };
    const scheduleSync = () => {
      if (eventRefreshTimer.current != null) return;
      eventRefreshTimer.current = window.setTimeout(() => void syncChanges(), 500);
    };
    events.onopen = () => {
      sseConnected.current = true;
      setError(null);
    };
    events.onerror = () => {
      sseConnected.current = false;
    };
    events.onmessage = (event) => {
      try {
        const log = JSON.parse(event.data) as ResearchLog;
        setResearch((current) => current ? {
          ...current,
          logs: mergeLogs(current.logs ?? [], [log]),
        } : current);
      } catch {
        // A summary refresh below still reconciles malformed or future event shapes.
      }
      scheduleSync();
    };
    return () => {
      disposed = true;
      syncController?.abort();
      sseConnected.current = false;
      events.close();
      if (eventRefreshTimer.current != null) {
        window.clearTimeout(eventRefreshTimer.current);
        eventRefreshTimer.current = null;
      }
    };
  }, [historyReady, pageVisible, researchId]);

  useEffect(() => {
    const sourceId = research?.gpu_source_id;
    if (!pageVisible || !sourceId) return;
    const events = new EventSource(telemetryEventsUrl(sourceId));
    let fallbackRequested = false;
    events.onopen = () => {
      fallbackRequested = false;
    };
    events.onmessage = (event) => {
      try {
        setTelemetry(normalizeTelemetryEvent(JSON.parse(event.data)));
      } catch {
        // Retain the last good sample across a transient malformed event.
      }
    };
    events.onerror = () => {
      if (fallbackRequested) return;
      fallbackRequested = true;
      void getGpuTelemetry(sourceId).then(setTelemetry).catch(() => undefined);
    };
    return () => {
      events.close();
    };
  }, [pageVisible, research?.gpu_source_id]);

  const visibleExperiments = useMemo(
    () => (research?.experiments ?? []).slice(monitorMode ? -120 : -MAX_LIVE_EXPERIMENTS),
    [monitorMode, research?.experiments],
  );
  const visibleLogs = useMemo(
    () => (research?.logs ?? []).slice(monitorMode ? -40 : -MAX_LIVE_LOGS),
    [monitorMode, research?.logs],
  );

  useEffect(() => {
    if (!deleteOpen) return;
    const previousOverflow = document.body.style.overflow;
    const returnFocus = deleteTriggerRef.current;
    document.body.style.overflow = "hidden";
    const frame = window.requestAnimationFrame(() => {
      deleteDialogRef.current?.querySelector<HTMLElement>("button:not(:disabled)")?.focus();
    });
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (!deleting.current) {
          event.preventDefault();
          setDeleteOpen(false);
        }
        return;
      }
      if (event.key !== "Tab" || !deleteDialogRef.current) return;
      const focusable = Array.from(
        deleteDialogRef.current.querySelectorAll<HTMLElement>(
          "button:not(:disabled), input:not(:disabled), [href], [tabindex]:not([tabindex='-1'])",
        ),
      );
      if (!focusable.length) {
        event.preventDefault();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      window.cancelAnimationFrame(frame);
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", handleKeyDown);
      returnFocus?.focus();
    };
  }, [deleteOpen]);

  async function runControl(action: "pause" | "resume" | "stop" | "start" | "dequeue") {
    setBusyAction(action);
    setError(null);
    try {
      await controlResearch(researchId, action);
      await refresh(true);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : `Could not ${action} research.`);
    } finally {
      setBusyAction(null);
    }
  }

  async function buildArticle() {
    setBusyAction("article");
    setError(null);
    try {
      setArticle(await createArticle(researchId));
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not build the article.");
    } finally {
      setBusyAction(null);
    }
  }

  async function removeResearch() {
    deleting.current = true;
    setBusyAction("delete");
    setDeleteError(null);
    try {
      await deleteResearch(researchId);
      router.replace("/researches");
      router.refresh();
    } catch (nextError) {
      deleting.current = false;
      setDeleteError(nextError instanceof Error ? nextError.message : "Could not delete the research.");
      setBusyAction(null);
    }
  }

  if (!research) {
    return (
      <div className="research-loader" role="status">
        <LoaderCircle className="spin" size={18} />
        <span>{error ?? "Loading research"}</span>
      </div>
    );
  }

  const canPause = research.status === "running";
  const canResume = research.status === "paused";
  const canRestartCompletedOllama = research.status === "completed" && research.adapter_type === "ollama_benchmark";
  const canStart = research.status === "queued" || research.status === "stopped" || research.status === "failed" || canRestartCompletedOllama;
  const canDequeue = research.status === "queued";
  const canStop = research.status === "running" || research.status === "paused";
  const hasActiveExperiment = research.experiments?.some(
    (experiment) => experiment.started_at && !experiment.completed_at,
  ) ?? false;
  const runtimePhase = research.runtime?.phase;
  const gpuPhase = research.status === "paused"
    ? "paused"
    : research.status !== "running"
      ? "idle"
      : runtimePhase === "gpu_evaluation" || (!runtimePhase && hasActiveExperiment)
        ? "training"
        : "planning";

  return (
    <motion.div
      className={`research-live ${monitorMode ? "research-monitor-mode" : ""}`}
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: "easeOut" }}
    >
      <div className="research-live-content" inert={deleteOpen ? true : undefined} aria-hidden={deleteOpen || undefined}>
      <div className="research-tv-stats" aria-label="Research monitoring summary">
        <div className="research-tv-status"><StatusPill status={research.status} queuePosition={research.queue_position} /></div>
        <div className="research-tv-token">
          <AnimatedTokenCount value={research.total_tokens ?? 0} />
          <span>Research agent token usage</span>
          <small>{tokenBreakdown(research)}</small>
        </div>
        <div className="research-tv-processed">
          <AnimatedTokenCount value={research.training_tokens ?? 0} />
          <span>Processed tokens</span>
        </div>
        <div className="research-tv-runtime">
          <ElapsedResearchTime
            createdAt={research.created_at}
            updatedAt={research.updated_at}
            status={research.status}
            className="research-tv-elapsed"
            showStartedAt={false}
          />
          <span>Total duration</span>
        </div>
      </div>
      <header className="research-header">
        <div>
          <div className="research-title-line">
            <h1>{research.title}</h1>
            <StatusPill status={research.status} queuePosition={research.queue_position} />
          </div>
          <div className="research-metric-line">
            <span>{metricLabel(research.metric_name)}</span>
            <strong>{formatMetric(research.best_value ?? research.baseline_value)}</strong>
            <span>{metricDescription(research.metric_name)}</span>
            {research.model_id ? <span className="research-model-name">{modelDisplayName(research.model_id)}</span> : null}
            {research.researcher_model_id ? <span className="research-model-name">Agent · {research.researcher_model_id}</span> : null}
            {research.schedule_start_time && research.schedule_end_time ? (
              <span className="research-model-name">Window · {research.schedule_start_time}–{research.schedule_end_time}</span>
            ) : null}
            <span className="research-token-count">Codex · {formatTokenCount(research.total_tokens)}</span>
          </div>
        </div>

        <div className="research-actions" aria-label="Research controls">
          {canStart ? (
            <button className="control-button primary-control" type="button" disabled={busyAction != null} onClick={() => void runControl("start")}>
              <Play size={15} fill="currentColor" /> {research.status === "queued" ? "Run now" : "Start"}
            </button>
          ) : null}
          {canDequeue ? (
            <button className="control-button" type="button" disabled={busyAction != null} onClick={() => void runControl("dequeue")}>
              <ListX size={15} /> Remove from queue
            </button>
          ) : null}
          {canPause ? (
            <button className="control-button" type="button" disabled={busyAction != null} onClick={() => void runControl("pause")}>
              <Pause size={15} fill="currentColor" /> Pause
            </button>
          ) : null}
          {canResume ? (
            <button className="control-button primary-control" type="button" disabled={busyAction != null} onClick={() => void runControl("resume")}>
              <Play size={15} fill="currentColor" /> Resume
            </button>
          ) : null}
          {canStop ? (
            <button className="control-button stop-control" type="button" disabled={busyAction != null} onClick={() => void runControl("stop")}>
              <Square size={14} fill="currentColor" /> Stop
            </button>
          ) : null}
          <button className="control-button" type="button" disabled={busyAction != null || !(research.experiments?.length)} onClick={() => void buildArticle()}>
            {busyAction === "article" ? <LoaderCircle className="spin" size={15} /> : <BookOpen size={15} />} Article
          </button>
          <button
            ref={deleteTriggerRef}
            className="control-button delete-control"
            type="button"
            disabled={busyAction != null}
            onClick={() => {
              deleting.current = false;
              setDeleteError(null);
              setDeleteOpen(true);
            }}
          >
            <Trash2 size={15} /> Delete
          </button>
        </div>
      </header>

      <AnimatePresence>
        {error ? <motion.p className="inline-error research-error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} role="alert">{error}</motion.p> : null}
        {article ? (
          <motion.div className="article-ready" initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }}>
            {article.article_kind === "general"
              ? "Reader guide built from the original question. "
              : "Article built from recorded experiments. "}
            <Link href={articlePath(article)}>Open article</Link>
          </motion.div>
        ) : null}
      </AnimatePresence>

      <div className="research-runtime-strip" aria-label="Research totals">
        <div className="research-runtime-tokens">
          <span>Research agent token usage</span>
          <AnimatedTokenCount value={research.total_tokens ?? 0} />
          <small>{tokenBreakdown(research)}</small>
        </div>
        <div className="research-runtime-tokens research-runtime-processed">
          <span>Processed tokens</span>
          <AnimatedTokenCount value={research.training_tokens ?? 0} />
          <small>Measured by completed GPU experiments</small>
        </div>
        <div className="research-runtime-duration">
          <span>Total duration</span>
          <ElapsedResearchTime
            createdAt={research.created_at}
            updatedAt={research.updated_at}
            status={research.status}
            className="research-elapsed-value"
          />
        </div>
      </div>

      <div className="research-grid">
        <ProgressChart
          experiments={visibleExperiments}
          metricName={research.metric_name}
          direction={research.metric_direction}
          baseline={research.baseline_value}
          acceptedCount={research.accepted_experiment_count}
          rejectedCount={research.rejected_experiment_count}
        />
        <GpuPerformance telemetry={telemetry} target={research.target_gpu_allocation} phase={gpuPhase} />
      </div>
      <LiveLogs logs={visibleLogs} status={research.status} metricName={research.metric_name} />
      </div>

      <AnimatePresence>
        {deleteOpen ? (
          <motion.div
            className="dialog-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onMouseDown={(event) => {
              if (event.currentTarget === event.target && busyAction !== "delete") setDeleteOpen(false);
            }}
          >
            <motion.section
              ref={deleteDialogRef}
              className="delete-research-dialog"
              role="dialog"
              aria-modal="true"
              aria-labelledby="delete-research-title"
              aria-describedby="delete-research-description"
              initial={{ opacity: 0, y: 12, scale: 0.985 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.99 }}
              transition={{ duration: 0.18, ease: "easeOut" }}
            >
              <div className="dialog-heading delete-dialog-heading">
                <div>
                  <span className="delete-dialog-eyebrow">Permanent action</span>
                  <h2 id="delete-research-title">Delete this research?</h2>
                </div>
                <button
                  className="icon-button"
                  type="button"
                  aria-label="Cancel deletion"
                  disabled={busyAction === "delete"}
                  onClick={() => setDeleteOpen(false)}
                >
                  <X size={18} />
                </button>
              </div>
              <p id="delete-research-description" className="delete-dialog-copy">
                <strong>{research.title}</strong> and its experiments, logs, and generated article will be removed. This cannot be undone.
              </p>
              {research.status === "running" || research.status === "paused" ? (
                <p className="delete-dialog-note">Its active GPU work will be stopped before removal.</p>
              ) : null}
              {deleteError ? <p className="inline-error delete-dialog-error" role="alert">{deleteError}</p> : null}
              <div className="delete-dialog-actions">
                <button
                  className="control-button"
                  type="button"
                  disabled={busyAction === "delete"}
                  onClick={() => setDeleteOpen(false)}
                >
                  Keep research
                </button>
                <button
                  className="control-button confirm-delete-control"
                  type="button"
                  disabled={busyAction === "delete"}
                  onClick={() => void removeResearch()}
                >
                  {busyAction === "delete" ? <LoaderCircle className="spin" size={15} /> : <Trash2 size={15} />}
                  {busyAction === "delete" ? "Deleting…" : "Delete permanently"}
                </button>
              </div>
            </motion.section>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </motion.div>
  );
}
