"use client";

import { AlertTriangle, CheckCircle2, Circle, PauseCircle, PlayCircle, StopCircle } from "lucide-react";
import { memo, useEffect, useMemo, useRef } from "react";
import { formatTime } from "@/lib/format";
import type { ResearchLog } from "@/lib/types";

type DisplayLog = ResearchLog & { repeatCount?: number };

function readableLogMessage(message: string): string {
  return message.replace(/\bval_bpb\b/gi, "text prediction loss");
}

function collapseRepeatedTimeouts(logs: ResearchLog[]): DisplayLog[] {
  const output: DisplayLog[] = [];
  let timeoutCount = 0;
  for (const log of logs) {
    const isCandidateTimeout = log.message.toLowerCase().includes("candidate generation")
      && log.message.toLowerCase().includes("time limit");
    if (!isCandidateTimeout) {
      output.push(log);
      continue;
    }
    timeoutCount += 1;
    const previousSummary = output.findIndex((item) => item.repeatCount != null);
    if (previousSummary >= 0) output.splice(previousSummary, 1);
    output.push({ ...log, repeatCount: timeoutCount });
  }
  return output;
}

function LogIcon({ type }: { type: string }) {
  const normalized = type.toLowerCase();
  if (normalized.includes("error") || normalized.includes("fail")) return <AlertTriangle size={15} />;
  if (normalized.includes("accept") || normalized.includes("complete")) return <CheckCircle2 size={15} />;
  if (normalized.includes("pause")) return <PauseCircle size={15} />;
  if (normalized.includes("resume") || normalized.includes("start")) return <PlayCircle size={15} />;
  if (normalized.includes("stop")) return <StopCircle size={15} />;
  return <Circle size={11} />;
}

function LiveLogsComponent({ logs }: { logs: ResearchLog[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const followsTail = useRef(true);
  const ordered = useMemo(
    () => [...logs].sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()),
    [logs],
  );
  const displayed = useMemo(() => collapseRepeatedTimeouts(ordered), [ordered]);

  useEffect(() => {
    const container = containerRef.current;
    if (container && followsTail.current) {
      container.scrollTop = container.scrollHeight;
    }
  }, [ordered.length]);

  return (
    <section className="surface-card logs-card" aria-labelledby="logs-title">
      <div className="card-heading">
        <div>
          <span className="eyebrow">Live logs</span>
          <h2 id="logs-title">Meaningful events</h2>
        </div>
        <span className="log-count">{displayed.length}</span>
      </div>
      <div
        className="logs-list"
        aria-live="polite"
        ref={containerRef}
        onScroll={(event) => {
          const target = event.currentTarget;
          followsTail.current = target.scrollHeight - target.scrollTop - target.clientHeight < 44;
        }}
      >
        {displayed.length ? displayed.slice(-80).map((log) => (
          <div className={`log-row log-${log.event_type.toLowerCase()}`} key={log.id}>
            <div className="log-icon"><LogIcon type={log.event_type} /></div>
            <span className="log-time">{formatTime(log.created_at)}</span>
            <p>{readableLogMessage(log.message)}{log.repeatCount && log.repeatCount > 1 ? <span className="log-repeat">×{log.repeatCount}</span> : null}</p>
          </div>
        )) : (
          <div className="quiet-row">No events yet.</div>
        )}
      </div>
    </section>
  );
}

export const LiveLogs = memo(LiveLogsComponent);
