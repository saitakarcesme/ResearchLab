"use client";

import { Cloud, Cpu, LoaderCircle, Radio, Server } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { getGpuTelemetry, listGpuSources } from "@/lib/api";
import type { GpuSource, GpuTelemetry } from "@/lib/types";
import { CloudGpuRental } from "./cloud-gpu-rental";
import { SettingsPanel } from "./settings-panel";

function formatMemory(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${(value / 1024).toFixed(1)} GB`;
}

export function GpusView() {
  const [sources, setSources] = useState<GpuSource[]>([]);
  const [telemetry, setTelemetry] = useState<Record<string, GpuTelemetry>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const nextSources = await listGpuSources();
      setSources(nextSources);
      const samples = await Promise.allSettled(
        nextSources.map(async (source) => [source.id, await getGpuTelemetry(source.id)] as const),
      );
      setTelemetry(Object.fromEntries(
        samples.flatMap((sample) => sample.status === "fulfilled" ? [sample.value] : []),
      ));
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "GPU inventory is unavailable.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => void refresh(), 0);
    const timer = window.setInterval(() => void refresh(), 4000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [refresh]);

  return (
    <div className="gpus-page">
      <header className="gpus-page-heading">
        <h1>GPUs</h1>
        <span><Radio size={12} aria-hidden="true" /> Live inventory</span>
      </header>

      {loading ? (
        <div className="gpu-inventory-loading" role="status"><LoaderCircle className="spin" size={16} /> Reading GPUs</div>
      ) : (
        <section className="gpu-inventory" aria-label="Available GPU sources">
          {sources.map((source) => {
            const sample = telemetry[source.id];
            const Icon = source.type === "local" ? Cpu : Server;
            return (
              <article className="gpu-inventory-card" key={source.id}>
                <div className="gpu-inventory-card-title">
                  <Icon size={17} aria-hidden="true" />
                  <div><strong>{sample?.name ?? source.name}</strong><span>{source.type === "local" ? "This computer" : source.host}</span></div>
                  <em>{sample?.available ? "Live" : "Offline"}</em>
                </div>
                <div className="gpu-inventory-utilization">
                  <strong>{sample?.utilization == null ? "—" : `${Math.round(sample.utilization)}%`}</strong>
                  <span>GPU load</span>
                </div>
                <dl>
                  <div><dt>Temperature</dt><dd>{sample?.temperature == null ? "—" : `${Math.round(sample.temperature)}°C`}</dd></div>
                  <div><dt>VRAM</dt><dd>{formatMemory(sample?.memory_used_mb)} / {formatMemory(sample?.memory_total_mb)}</dd></div>
                  <div><dt>Processes</dt><dd>{sample?.processes.length ?? 0}</dd></div>
                </dl>
              </article>
            );
          })}
        </section>
      )}
      {error ? <p className="inline-error" role="alert">{error}</p> : null}

      <section className="gpu-page-section">
        <div className="gpu-page-section-title"><Server size={15} /><div><strong>Sources</strong><span>Connect another computer over SSH.</span></div></div>
        <SettingsPanel />
      </section>

      <section className="gpu-page-section">
        <div className="gpu-page-section-title"><Cloud size={15} /><div><strong>Cloud rental</strong><span>Choose a live Vast.ai offer and pay from your phone.</span></div></div>
        <CloudGpuRental />
      </section>
    </div>
  );
}
