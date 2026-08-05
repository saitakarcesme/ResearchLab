"use client";

import { useEffect, useRef, useState } from "react";
import { Area, AreaChart, ResponsiveContainer } from "recharts";
import { formatMemory } from "@/lib/format";
import type { GpuTelemetry } from "@/lib/types";

interface UsagePoint { sample: number; value: number }

export type GpuActivityPhase = "planning" | "training" | "paused" | "idle";

const phaseCopy: Record<GpuActivityPhase, string> = {
  planning: "Planning next test",
  training: "GPU test running",
  paused: "Research paused",
  idle: "Research idle",
};

export function GpuPerformance({
  telemetry,
  target,
  phase,
}: {
  telemetry: GpuTelemetry | null;
  target: number;
  phase: GpuActivityPhase;
}) {
  const cursor = useRef(0);
  const [history, setHistory] = useState<UsagePoint[]>([]);

  useEffect(() => {
    if (telemetry?.utilization == null) return;
    cursor.current += 1;
    setHistory((current) => [
      ...current.slice(-29),
      { sample: cursor.current, value: telemetry.utilization as number },
    ]);
  }, [telemetry?.sampled_at, telemetry?.utilization]);

  const used = telemetry?.memory_used_mb ?? null;
  const total = telemetry?.memory_total_mb ?? null;
  const memoryPercent = used != null && total ? Math.min(100, (used / total) * 100) : 0;

  return (
    <section className="surface-card gpu-card" aria-labelledby="gpu-title">
      <div className="card-heading gpu-heading">
        <div>
          <span className="eyebrow">GPU performance</span>
          <h2 id="gpu-title">{telemetry?.name ?? "GPU unavailable"}</h2>
        </div>
        <span className={`live-indicator ${phase === "training" ? "online" : ""}`}>
          <i /> {phaseCopy[phase]}
        </span>
      </div>

      <div className="gpu-hero-metric">
        <strong>{telemetry?.utilization == null ? "â€”" : `${Math.round(telemetry.utilization)}%`}</strong>
        <span>live GPU utilization</span>
      </div>

      <div className="gpu-sparkline" aria-hidden="true">
        {history.length > 1 ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={history} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="gpuCardFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#d8d8d8" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#d8d8d8" stopOpacity={0} />
                </linearGradient>
              </defs>
              <Area type="monotone" dataKey="value" stroke="#f2f2f2" strokeWidth={2} fill="url(#gpuCardFill)" dot={false} isAnimationActive animationDuration={400} />
            </AreaChart>
          </ResponsiveContainer>
        ) : <div className="quiet-sparkline" />}
      </div>

      <div className="gpu-stats">
        <div><span>Temperature</span><strong>{telemetry?.temperature == null ? "—" : `${Math.round(telemetry.temperature)}°C`}</strong></div>
        <div><span>VRAM</span><strong>{formatMemory(used)} / {formatMemory(total)}</strong></div>
        <div><span>Scheduler share</span><strong>{target}%</strong></div>
      </div>

      <div className="memory-track" aria-label={total ? `VRAM ${Math.round(memoryPercent)} percent used` : "VRAM unavailable"}>
        <span style={{ width: `${memoryPercent}%` }} />
      </div>
      <p className="allocation-note">
        {phase === "planning"
          ? `${target}% is the requested scheduling share. The GPU can rest while the next test is being planned.`
          : phase === "training"
            ? `${target}% is the requested scheduling share; the live number above is the measured load.`
            : "The scheduler share applies again when this research resumes GPU work."}
      </p>
    </section>
  );
}
