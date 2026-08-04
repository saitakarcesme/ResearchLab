"use client";

import {
  AlertTriangle,
  Check,
  Circle,
  LoaderCircle,
  PlayCircle,
  RotateCcw,
} from "lucide-react";
import { memo, useMemo } from "react";
import { formatMetric, formatTime, metricLabel, titleCase } from "@/lib/format";
import type { ResearchLog, ResearchStatus } from "@/lib/types";

type ActivityTone = "working" | "success" | "warning" | "error" | "neutral";

interface ActivityGroup {
  key: string;
  experimentNumber: number | null;
  logs: ResearchLog[];
}

interface ActivityCopy {
  title: string;
  detail: string;
  tone: ActivityTone;
}

const SETUP_EVENTS = new Set([
  "research_created",
  "researcher_model_selected",
  "research_brief_generated",
  "research_brief_fallback",
  "research_started",
  "gpu_profile_applied",
  "model_selected",
  "model_verified",
  "environment_started",
  "environment_ready",
  "research_ready",
  "benchmark_started",
]);

const VISIBLE_EVENTS = new Set([
  ...SETUP_EVENTS,
  "agent_started",
  "agent_error",
  "experiment_started",
  "gpu_evaluation_started",
  "oom_retry",
  "experiment_completed",
  "experiment_accepted",
  "experiment_rejected",
  "experiment_error",
  "iteration_error",
  "research_paused",
  "research_resumed",
  "research_completed",
  "research_stopped",
  "research_failed",
  "research_control_error",
  "research_start_failed",
  "article_generated",
]);

function readableLogMessage(message: string): string {
  return message
    .replace(/\bval_bpb\b/gi, "text prediction loss")
    .replace(/\s+/g, " ")
    .trim();
}

function numericData(log: ResearchLog, key: string): number | null {
  const value = log.data?.[key];
  if (value == null) return null;
  const converted = typeof value === "number" ? value : Number(value);
  return Number.isFinite(converted) ? converted : null;
}

function experimentNumberFromLog(log: ResearchLog): number | null {
  const fromData = numericData(log, "experiment_number");
  if (fromData != null) return fromData;
  const match = log.message.match(/(?:experiment|profile)\s+(\d+)/i);
  return match ? Number(match[1]) : null;
}

function groupLogs(logs: ResearchLog[]): ActivityGroup[] {
  const experimentNumbers = new Map<string, number>();
  for (const log of logs) {
    const number = experimentNumberFromLog(log);
    if (log.experiment_id && number != null) experimentNumbers.set(log.experiment_id, number);
  }

  const groups = new Map<string, ActivityGroup>();
  for (const log of logs) {
    const experimentNumber = experimentNumberFromLog(log)
      ?? (log.experiment_id ? experimentNumbers.get(log.experiment_id) ?? null : null);
    const key = experimentNumber != null
      ? `experiment-${experimentNumber}`
      : SETUP_EVENTS.has(log.event_type)
        ? "setup"
        : `event-${log.id}`;
    const group = groups.get(key);
    if (group) {
      group.logs.push(log);
    } else {
      groups.set(key, { key, experimentNumber, logs: [log] });
    }
  }
  return [...groups.values()];
}

function modelName(log: ResearchLog): string {
  const value = String(log.data?.researcher_model_id ?? log.data?.model_id ?? "the selected model");
  return value.replace(/^ollama:/, "");
}

function eventCopy(log: ResearchLog, experimentNumber: number | null): ActivityCopy {
  const number = experimentNumber != null ? ` ${experimentNumber}` : "";
  const detail = readableLogMessage(log.message);
  switch (log.event_type) {
    case "research_created":
      return { title: "Research created", detail: "The request has been saved and prepared for the Lab.", tone: "success" };
    case "researcher_model_selected":
      return { title: "Research model selected", detail: `${modelName(log)} will plan and interpret the run.`, tone: "success" };
    case "research_brief_generated":
      return { title: "Goal prepared", detail: "The original request was converted into a measurable research objective.", tone: "success" };
    case "research_brief_fallback":
      return { title: "Goal prepared with fallback", detail: "The original request was retained without blocking the run.", tone: "warning" };
    case "research_started":
      return { title: "Research started", detail: "The execution worker is active.", tone: "working" };
    case "gpu_profile_applied":
      return { title: "GPU profile applied", detail, tone: "success" };
    case "model_selected":
    case "model_verified":
      return { title: "Model verified", detail, tone: "success" };
    case "environment_started":
      return { title: "Preparing the environment", detail: "Loading the pinned code, runtime and dependencies.", tone: "working" };
    case "environment_ready":
    case "research_ready":
      return { title: "Workspace ready", detail: "The Lab can now run measured experiments.", tone: "success" };
    case "benchmark_started":
      return { title: "Benchmark started", detail: "The model is loaded once before the measured profiles begin.", tone: "working" };
    case "agent_started":
      return { title: `Planning experiment${number}`, detail: "The research model is reviewing earlier results and proposing the next test.", tone: "working" };
    case "agent_error":
      return { title: `Experiment${number} will retry`, detail: "The candidate was not ready in time; the next attempt is scheduled automatically.", tone: "warning" };
    case "local_researcher_released":
      return { title: "GPU memory released", detail: "The local research model was unloaded before measurement.", tone: "success" };
    case "experiment_started":
      return { title: `Running experiment${number}`, detail, tone: "working" };
    case "change_applied":
      return { title: "Testing a code change", detail, tone: "working" };
    case "gpu_evaluation_started":
      return { title: "Measuring on the GPU", detail: "The fixed evaluation is running; setup periods may briefly show lower utilization.", tone: "working" };
    case "model_warmed":
      return { title: "Model warmed up", detail, tone: "working" };
    case "oom_retry":
      return { title: "Adjusting GPU memory use", detail, tone: "warning" };
    case "experiment_completed":
      return { title: `Experiment${number} measured`, detail, tone: "success" };
    case "experiment_accepted":
      return { title: `Experiment${number} improved the result`, detail, tone: "success" };
    case "experiment_rejected":
      return { title: `Experiment${number} tested — best unchanged`, detail, tone: "neutral" };
    case "experiment_error":
    case "iteration_error":
      return {
        title: `Experiment${number} needs attention`,
        detail: detail.includes("flash-attn3") || detail.includes("Cannot install kernel")
          ? "The required GPU kernel is not available in the current runtime; the saved result was kept."
          : "The test did not produce a valid measurement; the saved best result was kept.",
        tone: "error",
      };
    case "experiment_recovered":
      return { title: `Experiment${number} safely closed`, detail, tone: "warning" };
    case "research_paused":
      return { title: "Research paused", detail: "Current state and the best accepted result are preserved.", tone: "warning" };
    case "research_resumed":
      return { title: "Research resumed", detail: "The worker is continuing from the saved best result.", tone: "working" };
    case "research_completed":
      return { title: "Research completed", detail, tone: "success" };
    case "research_stopped":
      return { title: "Research stopped", detail: "The best accepted result remains saved.", tone: "neutral" };
    case "research_failed":
    case "research_control_error":
    case "research_start_failed":
      return { title: "Research needs attention", detail, tone: "error" };
    case "article_generated":
      return { title: "Article updated", detail: "The readable article was rebuilt from persisted results.", tone: "success" };
    default:
      return {
        title: titleCase(log.event_type),
        detail,
        tone: log.level === "error" ? "error" : log.level === "warning" ? "warning" : "neutral",
      };
  }
}

function groupCopy(group: ActivityGroup): ActivityCopy {
  const last = group.logs.at(-1)!;
  const copy = eventCopy(last, group.experimentNumber);
  if (group.key !== "setup") return copy;
  if (last.event_type === "research_ready" || last.event_type === "environment_ready") {
    return { title: "Research workspace ready", detail: "Model, GPU profile and execution environment are prepared.", tone: "success" };
  }
  return copy;
}

function groupMetadata(group: ActivityGroup, metricName: string): string[] {
  const metadata: string[] = [];
  if (group.experimentNumber != null) metadata.push(`Experiment ${group.experimentNumber}`);
  const resultLog = [...group.logs].reverse().find((log) => numericData(log, "metric_value") != null);
  const metricValue = resultLog ? numericData(resultLog, "metric_value") : null;
  if (metricValue != null) metadata.push(`${metricLabel(metricName)} ${formatMetric(metricValue)}`);
  const completedLog = [...group.logs].reverse().find((log) => log.event_type === "experiment_completed");
  const seconds = completedLog ? numericData(completedLog, "training_seconds") ?? numericData(completedLog, "total_seconds") : null;
  if (seconds != null) metadata.push(`${Math.round(seconds)}s measured`);
  const latest = group.logs.at(-1)!;
  const retryDelay = numericData(latest, "retry_delay_seconds");
  if (retryDelay != null) metadata.push(`Retry in ${Math.round(retryDelay)}s`);
  const retryCount = group.logs.filter((log) => log.event_type === "agent_error").length;
  if (retryCount > 1) metadata.push(`${retryCount} retries grouped`);
  return metadata;
}

function ActivityIcon({ tone, active = false }: { tone: ActivityTone; active?: boolean }) {
  if (active && tone === "working") return <LoaderCircle className="spin" size={16} />;
  if (tone === "error") return <AlertTriangle size={15} />;
  if (tone === "warning") return <RotateCcw size={15} />;
  if (tone === "success") return <Check size={15} />;
  if (tone === "working") return <PlayCircle size={15} />;
  return <Circle size={10} />;
}

function LiveLogsComponent({
  logs,
  status,
  metricName,
}: {
  logs: ResearchLog[];
  status: ResearchStatus;
  metricName: string;
}) {
  const ordered = useMemo(
    () => [...logs].sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()),
    [logs],
  );
  const groups = useMemo(() => groupLogs(ordered), [ordered]);
  const visibleGroups = useMemo(
    () => groups.filter((group) => group.logs.some((log) => VISIBLE_EVENTS.has(log.event_type) || log.level === "error")),
    [groups],
  );
  const latest = visibleGroups.at(-1) ?? null;
  const recent = visibleGroups.slice(0, -1).reverse().slice(0, 6);
  const running = status === "running";
  const latestCopy = latest ? groupCopy(latest) : null;

  return (
    <section className="surface-card logs-card activity-card" aria-labelledby="logs-title">
      <div className="card-heading activity-heading">
        <h2 id="logs-title">Activity</h2>
        <span className={`activity-state ${running ? "activity-state-live" : ""}`}>
          <span />{running ? "Live" : titleCase(status)}
        </span>
      </div>

      {latest && latestCopy ? (
        <div className={`activity-current activity-tone-${latestCopy.tone}`} aria-live="polite">
          <div className="activity-current-icon"><ActivityIcon tone={latestCopy.tone} active={running} /></div>
          <div className="activity-current-copy">
            <strong>{latestCopy.title}</strong>
            <div className="activity-meta">
              {groupMetadata(latest, metricName).map((item) => <span key={item}>{item}</span>)}
            </div>
          </div>
          <time dateTime={latest.logs.at(-1)!.created_at}>{formatTime(latest.logs.at(-1)!.created_at)}</time>
        </div>
      ) : (
        <div className="activity-empty"><LoaderCircle className="spin" size={15} /> Waiting for the first activity.</div>
      )}

      {recent.length ? (
        <div className="activity-history" aria-label="Recent activity">
          {recent.map((group) => {
            const copy = groupCopy(group);
            const metadata = groupMetadata(group, metricName);
            return (
              <div className="activity-history-row" key={group.key}>
                <span className={`activity-history-icon activity-tone-${copy.tone}`}><ActivityIcon tone={copy.tone} /></span>
                <strong>{copy.title}</strong>
                {metadata.length ? <small>{metadata.join(" · ")}</small> : null}
                <time dateTime={group.logs.at(-1)!.created_at}>{formatTime(group.logs.at(-1)!.created_at)}</time>
              </div>
            );
          })}
        </div>
      ) : null}
    </section>
  );
}

export const LiveLogs = memo(LiveLogsComponent);
