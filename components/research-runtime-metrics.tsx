"use client";

import { useEffect, useRef, useState } from "react";
import type { ResearchStatus } from "@/lib/types";

const terminalStatuses = new Set<ResearchStatus>(["completed", "stopped", "failed"]);

function easeOutCubic(value: number): number {
  return 1 - (1 - value) ** 3;
}

function formatExactTokens(value: number): string {
  return Math.max(0, Math.round(value)).toLocaleString();
}

function formatDuration(milliseconds: number): string {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return [hours, minutes, seconds].map((part) => String(part).padStart(2, "0")).join(":");
}

function formatStartedAt(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat(undefined, {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function AnimatedTokenCount({ value, className = "" }: { value: number; className?: string }) {
  const displayValueRef = useRef(0);
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    const target = Math.max(0, value);
    const from = displayValueRef.current;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      const frame = window.requestAnimationFrame(() => {
        displayValueRef.current = target;
        setDisplayValue(target);
      });
      return () => window.cancelAnimationFrame(frame);
    }
    const startedAt = performance.now();
    const duration = from === 0 ? 1100 : 650;
    let frame = 0;
    const update = (now: number) => {
      const progress = Math.min(1, (now - startedAt) / duration);
      const nextValue = from + (target - from) * easeOutCubic(progress);
      displayValueRef.current = nextValue;
      setDisplayValue(nextValue);
      if (progress < 1) {
        frame = window.requestAnimationFrame(update);
      }
    };
    frame = window.requestAnimationFrame(update);
    return () => window.cancelAnimationFrame(frame);
  }, [value]);

  return <strong className={className}>{formatExactTokens(displayValue)}</strong>;
}

export function ElapsedResearchTime({
  createdAt,
  updatedAt,
  status,
  className = "",
  showStartedAt = true,
}: {
  createdAt: string;
  updatedAt: string;
  status: ResearchStatus;
  className?: string;
  showStartedAt?: boolean;
}) {
  const terminal = terminalStatuses.has(status);
  const [clock, setClock] = useState(Date.now);

  useEffect(() => {
    if (terminal) return;
    const timer = window.setInterval(() => setClock(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [terminal]);

  const started = new Date(createdAt).getTime();
  const now = terminal ? new Date(updatedAt).getTime() : clock;
  const elapsed = Number.isFinite(started) && Number.isFinite(now) ? now - started : 0;
  return (
    <span className={className}>
      <strong>{formatDuration(elapsed)}</strong>
      {showStartedAt ? <small>Started {formatStartedAt(createdAt)}</small> : null}
    </span>
  );
}

export function LabTotalTokenUsage({
  value,
  input,
  output,
}: {
  value: number;
  input: number;
  output: number;
}) {
  return (
    <div className="lab-total-token-usage" aria-label={`${formatExactTokens(value)} research agent tokens used`}>
      <AnimatedTokenCount value={value} />
      <span>Research agent token usage</span>
      <small>{formatExactTokens(input)} input · {formatExactTokens(output)} output</small>
    </div>
  );
}
