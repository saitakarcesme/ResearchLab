"use client";

import { AlertTriangle, CheckCircle2, Circle, PauseCircle, PlayCircle, StopCircle } from "lucide-react";
import { memo, useEffect, useMemo, useRef } from "react";
import { formatTime } from "@/lib/format";
import type { ResearchLog } from "@/lib/types";

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
        <span className="log-count">{ordered.length}</span>
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
        {ordered.length ? ordered.slice(-80).map((log) => (
          <div className={`log-row log-${log.event_type.toLowerCase()}`} key={log.id}>
            <div className="log-icon"><LogIcon type={log.event_type} /></div>
            <span className="log-time">{formatTime(log.created_at)}</span>
            <p>{log.message}</p>
          </div>
        )) : (
          <div className="quiet-row">No events yet.</div>
        )}
      </div>
    </section>
  );
}

export const LiveLogs = memo(LiveLogsComponent);
