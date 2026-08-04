"use client";

import { ChevronRight, LoaderCircle } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { listGpuSources, listResearches } from "@/lib/api";
import { formatMetric, metricLabel } from "@/lib/format";
import type { GpuSource, Research } from "@/lib/types";
import { StatusPill } from "./status-pill";

function researchMethod(research: Research): string {
  if (!research.model_id) return "Training optimization";
  return research.model_id.replace(/^ollama:/, "").replace(/:latest$/, "");
}

export function ResearchesView() {
  const [researches, setResearches] = useState<Research[]>([]);
  const [sources, setSources] = useState<GpuSource[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [nextResearches, nextSources] = await Promise.all([listResearches(), listGpuSources()]);
      setResearches(nextResearches);
      setSources(nextSources);
      setError(null);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Researches are unavailable.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => void load(), 0);
    const timer = window.setInterval(() => void load(), 6000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [load]);

  const sourceNames = useMemo(() => new Map(sources.map((source) => [source.id, source.name])), [sources]);
  const ordered = [...researches].sort((a, b) => {
    if (a.status === "queued" && b.status === "queued") {
      return (a.queue_position ?? Number.MAX_SAFE_INTEGER) - (b.queue_position ?? Number.MAX_SAFE_INTEGER);
    }
    if (a.status === "queued") return -1;
    if (b.status === "queued") return 1;
    return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
  });

  return (
    <div className="collection-page">
      <header className="compact-page-header">
        <h1>Researches</h1>
        <span>{researches.length}</span>
      </header>
      {error ? <p className="inline-error" role="alert">{error}</p> : null}
      <div className="research-table" aria-live="polite">
        <div className="research-table-head" aria-hidden="true">
          <span>Research</span><span>Status</span><span>GPU</span><span>Method / model</span><span>Tests</span><span>Best metric</span><span />
        </div>
        {loading ? (
          <div className="quiet-row centered table-loading"><LoaderCircle className="spin" size={16} /> Loading researches</div>
        ) : error ? null : ordered.length ? ordered.map((research) => (
          <Link className="research-table-row" href={`/researches/${research.id}`} key={research.id}>
            <strong>{research.title}</strong>
            <StatusPill status={research.status} queuePosition={research.queue_position} />
            <span data-label="GPU">{research.gpu_source_name ?? sourceNames.get(research.gpu_source_id ?? "") ?? "—"}</span>
            <span className="method-cell" data-label="Method">{researchMethod(research)}</span>
            <span data-label="Tests">{research.experiment_count ?? research.experiments?.length ?? 0}</span>
            <span className="metric-cell" data-label="Best metric">{metricLabel(research.metric_name)} <strong>{formatMetric(research.best_value)}</strong></span>
            <ChevronRight size={16} />
          </Link>
        )) : (
          <div className="empty-collection">
            <span>No researches yet.</span>
            <Link href="/lab">Start in Lab</Link>
          </div>
        )}
      </div>
    </div>
  );
}
