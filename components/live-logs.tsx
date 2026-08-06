"use client";

import {
  AlertTriangle,
  ArrowRight,
  Check,
  ChevronDown,
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
  "local_researcher_released",
  "research_plan_created",
  "research_plan_repeated",
  "research_plan_fallback",
  "experiment_started",
  "change_applied",
  "gpu_evaluation_started",
  "model_warmed",
  "oom_retry",
  "experiment_completed",
  "experiment_accepted",
  "experiment_rejected",
  "experiment_error",
  "iteration_error",
  "experiment_recovered",
  "research_paused",
  "research_resumed",
  "research_completed",
  "research_stopped",
  "research_failed",
  "research_control_error",
  "research_start_failed",
  "article_generated",
]);

const OUTCOME_EVENTS = new Set([
  "experiment_completed",
  "experiment_accepted",
  "experiment_rejected",
  "experiment_error",
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
    if (group) group.logs.push(log);
    else groups.set(key, { key, experimentNumber, logs: [log] });
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
      return { title: "Research created", detail: "The request is saved and ready for the execution worker.", tone: "success" };
    case "researcher_model_selected":
      return { title: "Research model selected", detail: `${modelName(log)} will plan and interpret every experiment.`, tone: "success" };
    case "research_brief_generated":
      return { title: "Measurable goal prepared", detail: "The request was converted into a fixed objective and evaluation contract.", tone: "success" };
    case "research_brief_fallback":
      return { title: "Goal preserved", detail: "The original request was retained without blocking the run.", tone: "warning" };
    case "research_started":
      return { title: "Research worker started", detail: "The autonomous experiment loop is active.", tone: "working" };
    case "gpu_profile_applied":
      return { title: "GPU profile applied", detail, tone: "success" };
    case "model_selected":
    case "model_verified":
      return { title: "Model verified", detail, tone: "success" };
    case "environment_started":
      return { title: "Preparing the environment", detail: "Loading the pinned repository, runtime and dependencies.", tone: "working" };
    case "environment_ready":
    case "research_ready":
      return { title: "Workspace ready", detail: "The fixed evaluation harness is ready for measured experiments.", tone: "success" };
    case "benchmark_started":
      return { title: "Benchmark started", detail: "The selected model is being loaded before measurement.", tone: "working" };
    case "agent_started":
      return { title: `Planning experiment${number}`, detail: "The research model is reading previous results and preparing one train.py change.", tone: "working" };
    case "agent_error":
      return { title: `Planning experiment${number} again`, detail: "The candidate was not ready; the loop retained the best result and scheduled another attempt.", tone: "warning" };
    case "research_plan_created":
      return { title: "Next hypothesis selected", detail, tone: "working" };
    case "research_plan_repeated":
    case "research_plan_fallback":
      return { title: "Search path adjusted", detail, tone: "warning" };
    case "local_researcher_released":
      return { title: "GPU handed to the experiment", detail: "The local research model left GPU memory before the measured run.", tone: "success" };
    case "experiment_started":
      return { title: `Experiment${number} started`, detail, tone: "working" };
    case "change_applied":
      return { title: "Candidate change ready", detail, tone: "working" };
    case "gpu_evaluation_started":
      return { title: "Measuring on the GPU", detail: "The fixed five-minute evaluation is running now.", tone: "working" };
    case "model_warmed":
      return { title: "Model warmed up", detail, tone: "working" };
    case "oom_retry":
      return { title: "GPU memory plan adjusted", detail, tone: "warning" };
    case "experiment_completed":
      return { title: `Experiment${number} measured`, detail, tone: "success" };
    case "experiment_accepted":
      return { title: `Experiment${number} improved the best result`, detail, tone: "success" };
    case "experiment_rejected":
      return { title: `Experiment${number} tested — best unchanged`, detail, tone: "neutral" };
    case "experiment_error":
    case "iteration_error":
      return { title: `Experiment${number} could not be measured`, detail: "The saved best result was preserved; the failure details are available below.", tone: "error" };
    case "experiment_recovered":
      return { title: `Experiment${number} safely recovered`, detail, tone: "warning" };
    case "research_paused":
      return { title: "Research paused", detail: "The active state and best result are preserved.", tone: "warning" };
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
      return { title: "Article updated", detail: "The readable article was rebuilt from persisted evidence.", tone: "success" };
    default:
      return { title: titleCase(log.event_type), detail, tone: log.level === "error" ? "error" : log.level === "warning" ? "warning" : "neutral" };
  }
}

function groupCopy(group: ActivityGroup): ActivityCopy {
  const last = group.logs.at(-1)!;
  const copy = eventCopy(last, group.experimentNumber);
  if (group.key !== "setup") return copy;
  if (last.event_type === "research_ready" || last.event_type === "environment_ready") {
    return { title: "Research workspace ready", detail: "Model, GPU profile and execution environment passed setup.", tone: "success" };
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
  const retryDelay = numericData(group.logs.at(-1)!, "retry_delay_seconds");
  if (retryDelay != null) metadata.push(`Retry in ${Math.round(retryDelay)}s`);
  return metadata;
}

function nextStep(status: ResearchStatus, latest: ResearchLog | null): ActivityCopy {
  if (status === "paused") return { title: "Waiting for Resume", detail: "No process will advance until the research is resumed.", tone: "warning" };
  if (status === "stopped" || status === "completed") return { title: "No next experiment scheduled", detail: "Start the research again to continue from the saved best result.", tone: "neutral" };
  if (status === "failed") return { title: "Review the latest failure", detail: "The saved result is intact; restart after the blocking condition is resolved.", tone: "error" };
  if (status === "queued") return { title: "Waiting for GPU capacity", detail: "The worker will start automatically when the selected GPU is available.", tone: "neutral" };
  if (latest?.event_type === "gpu_evaluation_started") return { title: "Compare the measured result", detail: "The runner will keep an improvement and discard a regression.", tone: "working" };
  if (latest?.event_type === "agent_started") return { title: "Validate one candidate", detail: "Only train.py may change before the fixed GPU evaluation begins.", tone: "working" };
  if (OUTCOME_EVENTS.has(latest?.event_type ?? "")) return { title: "Propose the next experiment", detail: "The local research model will use this result to choose the next change.", tone: "working" };
  return { title: "Advance the autonomous loop", detail: "The next meaningful transition will appear here automatically.", tone: "working" };
}

function ActivityIcon({ tone, active = false }: { tone: ActivityTone; active?: boolean }) {
  if (active && tone === "working") return <LoaderCircle className="spin" size={18} />;
  if (tone === "error") return <AlertTriangle size={17} />;
  if (tone === "warning") return <RotateCcw size={17} />;
  if (tone === "success") return <Check size={17} />;
  if (tone === "working") return <PlayCircle size={17} />;
  return <Circle size={11} />;
}

function ActivitySummary({ label, copy }: { label: string; copy: ActivityCopy }) {
  return (
    <div className="activity-summary-item">
      <span>{label}</span>
      <strong>{copy.title}</strong>
      <p>{copy.detail}</p>
    </div>
  );
}

function ActivityHistoryItem({ group, metricName }: { group: ActivityGroup; metricName: string }) {
  const copy = groupCopy(group);
  const metadata = groupMetadata(group, metricName);
  const last = group.logs.at(-1)!;
  const summary = (
    <>
      <span className={`activity-history-icon activity-tone-${copy.tone}`}><ActivityIcon tone={copy.tone} /></span>
      <div className="activity-history-copy">
        <strong>{copy.title}</strong>
        <p>{copy.detail}</p>
        {metadata.length ? <small>{metadata.join(" · ")}</small> : null}
      </div>
      <time dateTime={last.created_at}>{formatTime(last.created_at)}</time>
    </>
  );

  if (group.logs.length < 2) return <div className="activity-history-row">{summary}</div>;
  return (
    <details className="activity-history-group">
      <summary>{summary}<ChevronDown className="activity-disclosure" size={15} /></summary>
      <div className="activity-steps">
        {group.logs.map((log) => {
          const step = eventCopy(log, group.experimentNumber);
          return (
            <div className="activity-step" key={log.id}>
              <ArrowRight size={12} />
              <div><strong>{step.title}</strong><span>{readableLogMessage(log.message)}</span></div>
              <time dateTime={log.created_at}>{formatTime(log.created_at)}</time>
            </div>
          );
        })}
      </div>
    </details>
  );
}

function LiveLogsComponent({ logs, status, metricName }: { logs: ResearchLog[]; status: ResearchStatus; metricName: string }) {
  const ordered = useMemo(() => [...logs].sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()), [logs]);
  const visibleGroups = useMemo(
    () => groupLogs(ordered).filter((group) => group.logs.some((log) => VISIBLE_EVENTS.has(log.event_type) || log.level === "error")),
    [ordered],
  );
  const latest = visibleGroups.at(-1) ?? null;
  const recent = visibleGroups.slice(0, -1).reverse().slice(0, 24);
  const lastOutcome = [...visibleGroups].reverse().find((group) => group.logs.some((log) => OUTCOME_EVENTS.has(log.event_type))) ?? null;
  const running = status === "running";
  const latestCopy = latest ? groupCopy(latest) : null;
  const outcomeCopy = lastOutcome ? groupCopy(lastOutcome) : { title: "No measured result yet", detail: "The first completed GPU evaluation will appear here.", tone: "neutral" as const };
  const nextCopy = nextStep(status, latest?.logs.at(-1) ?? null);

  return (
    <section className="surface-card logs-card activity-card" aria-labelledby="logs-title">
      <div className="card-heading activity-heading">
        <div><span className="eyebrow">Research flow</span><h2 id="logs-title">Activity</h2></div>
        <span className={`activity-state ${running ? "activity-state-live" : ""}`}><span />{running ? "Live" : titleCase(status)}</span>
      </div>

      {latest && latestCopy ? (
        <div className={`activity-current activity-tone-${latestCopy.tone}`} aria-live="polite">
          <div className="activity-current-icon"><ActivityIcon tone={latestCopy.tone} active={running} /></div>
          <div className="activity-current-copy">
            <span>Working now</span>
            <strong>{latestCopy.title}</strong>
            <p>{latestCopy.detail}</p>
            <div className="activity-meta">{groupMetadata(latest, metricName).map((item) => <span key={item}>{item}</span>)}</div>
          </div>
          <time dateTime={latest.logs.at(-1)!.created_at}>{formatTime(latest.logs.at(-1)!.created_at)}</time>
        </div>
      ) : (
        <div className="activity-empty"><LoaderCircle className="spin" size={17} /> Waiting for the first meaningful event.</div>
      )}

      <div className="activity-summary" aria-label="Research context">
        <ActivitySummary label="Last measured result" copy={outcomeCopy} />
        <ActivitySummary label="Next" copy={nextCopy} />
      </div>

      {recent.length ? (
        <div className="activity-trail">
          <div className="activity-trail-heading"><span>Research trail</span><small>{recent.length} grouped updates</small></div>
          <div className="activity-history" aria-label="Recent activity">
            {recent.map((group) => <ActivityHistoryItem group={group} metricName={metricName} key={group.key} />)}
          </div>
        </div>
      ) : null}
    </section>
  );
}

export const LiveLogs = memo(LiveLogsComponent);
