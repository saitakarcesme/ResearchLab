"use client";

import { AnimatePresence, motion } from "framer-motion";
import { BookOpen, LoaderCircle, Pause, Play, Square } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  controlResearch,
  createArticle,
  getGpuTelemetry,
  getResearch,
  researchEventsUrl,
} from "@/lib/api";
import { formatMetric, metricDescription, metricLabel } from "@/lib/format";
import type { Article, GpuTelemetry, Research } from "@/lib/types";
import { GpuPerformance } from "./gpu-performance";
import { LiveLogs } from "./live-logs";
import { ProgressChart } from "./progress-chart";
import { StatusPill } from "./status-pill";

export function ResearchLiveView({
  researchId,
  initialResearch,
}: {
  researchId: string;
  initialResearch?: Research | null;
}) {
  const [research, setResearch] = useState<Research | null>(initialResearch ?? null);
  const [telemetry, setTelemetry] = useState<GpuTelemetry | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [article, setArticle] = useState<Article | null>(null);
  const [error, setError] = useState<string | null>(null);
  const refreshing = useRef(false);

  const refresh = useCallback(async () => {
    if (refreshing.current) return;
    refreshing.current = true;
    try {
      setResearch(await getResearch(researchId));
      setError(null);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Research data is unavailable.");
    } finally {
      refreshing.current = false;
    }
  }, [researchId]);

  useEffect(() => {
    const initial = window.setTimeout(() => void refresh(), 0);
    const interval = window.setInterval(() => void refresh(), 5000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(interval);
    };
  }, [refresh]);

  useEffect(() => {
    const events = new EventSource(researchEventsUrl(researchId));
    const handleEvent = () => void refresh();
    events.onmessage = handleEvent;
    [
      "research_created",
      "research_started",
      "research_ready",
      "research_paused",
      "research_resumed",
      "research_stopped",
      "research_failed",
      "environment_started",
      "environment_ready",
      "gpu_profile_applied",
      "research_brief_generated",
      "research_brief_fallback",
      "agent_started",
      "agent_error",
      "gpu_evaluation_started",
      "experiment_started",
      "change_applied",
      "oom_retry",
      "experiment_error",
      "experiment_completed",
      "experiment_accepted",
      "experiment_rejected",
      "article_generated",
      "service_restarted",
      "service_shutdown",
      "iteration_error",
    ].forEach((name) =>
      events.addEventListener(name, handleEvent),
    );
    return () => events.close();
  }, [refresh, researchId]);

  useEffect(() => {
    let mounted = true;
    async function sample() {
      try {
        const next = await getGpuTelemetry(research?.gpu_source_id);
        if (mounted) setTelemetry(next);
      } catch {
        if (mounted) setTelemetry(null);
      }
    }
    void sample();
    const interval = window.setInterval(() => void sample(), 2200);
    return () => {
      mounted = false;
      window.clearInterval(interval);
    };
  }, [research?.gpu_source_id]);

  async function runControl(action: "pause" | "resume" | "stop" | "start") {
    setBusyAction(action);
    setError(null);
    try {
      await controlResearch(researchId, action);
      await refresh();
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
  const canStart = research.status === "stopped" || research.status === "failed";
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
      className="research-live"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: "easeOut" }}
    >
      <header className="research-header">
        <div>
          <div className="research-title-line">
            <h1>{research.title}</h1>
            <StatusPill status={research.status} />
          </div>
          <div className="research-metric-line">
            <span>{metricLabel(research.metric_name)}</span>
            <strong>{formatMetric(research.best_value ?? research.baseline_value)}</strong>
            <span>{metricDescription(research.metric_name)}</span>
          </div>
        </div>

        <div className="research-actions" aria-label="Research controls">
          {canStart ? (
            <button className="control-button primary-control" type="button" disabled={busyAction != null} onClick={() => void runControl("start")}>
              <Play size={15} fill="currentColor" /> Start
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
        </div>
      </header>

      <AnimatePresence>
        {error ? <motion.p className="inline-error research-error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} role="alert">{error}</motion.p> : null}
        {article ? (
          <motion.div className="article-ready" initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }}>
            Article built from recorded experiments. <Link href={`/articles/${article.id}`}>Open article</Link>
          </motion.div>
        ) : null}
      </AnimatePresence>

      <div className="research-grid">
        <ProgressChart
          experiments={research.experiments ?? []}
          metricName={research.metric_name}
          direction={research.metric_direction}
          baseline={research.baseline_value}
        />
        <GpuPerformance telemetry={telemetry} target={research.target_gpu_allocation} phase={gpuPhase} />
      </div>
      <LiveLogs logs={research.logs ?? []} />
    </motion.div>
  );
}
