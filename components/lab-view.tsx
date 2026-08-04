"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ArrowUp, ChevronRight, LoaderCircle } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { controlResearch, createResearch, getGpuTelemetry, listGpuSources, listResearches } from "@/lib/api";
import { formatMetric, metricLabel } from "@/lib/format";
import type { GpuSource, GpuTelemetry, Research } from "@/lib/types";
import { GpuLineBackdrop } from "./gpu-line-backdrop";
import { ResearchLiveView } from "./research-live-view";
import { StatusPill } from "./status-pill";

export function LabView() {
  const [prompt, setPrompt] = useState("");
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null);
  const [researches, setResearches] = useState<Research[]>([]);
  const [sources, setSources] = useState<GpuSource[]>([]);
  const [selectedSource, setSelectedSource] = useState("");
  const [allocation, setAllocation] = useState(100);
  const [telemetry, setTelemetry] = useState<GpuTelemetry | null>(null);
  const [activeResearch, setActiveResearch] = useState<Research | null>(null);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const liveRef = useRef<HTMLDivElement>(null);
  const sourceSelectRef = useRef<HTMLSelectElement>(null);
  const wasConfiguringRef = useRef(false);

  const load = useCallback(async () => {
    try {
      const [nextResearches, nextSources] = await Promise.all([listResearches(), listGpuSources()]);
      setResearches(nextResearches);
      setSources(nextSources);
      setSelectedSource((current) => current || nextSources[0]?.id || "");
      setError(null);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Research service is unavailable.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (activeResearch) return;
    const initial = window.setTimeout(() => void load(), 0);
    const timer = window.setInterval(() => void load(), 8000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [activeResearch, load]);

  useEffect(() => {
    if (activeResearch) return;
    let mounted = true;
    async function sample() {
      try {
        const next = await getGpuTelemetry(selectedSource || null);
        if (mounted) setTelemetry(next);
      } catch {
        if (mounted) setTelemetry(null);
      }
    }
    void sample();
    const timer = window.setInterval(() => void sample(), 2200);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, [activeResearch, selectedSource]);

  useEffect(() => {
    if (!pendingPrompt) {
      wasConfiguringRef.current = false;
      return;
    }
    if (wasConfiguringRef.current) return;
    wasConfiguringRef.current = true;
    const frame = window.requestAnimationFrame(() => sourceSelectRef.current?.focus());
    return () => window.cancelAnimationFrame(frame);
  }, [pendingPrompt]);

  const visibleResearches = useMemo(
    () => [...researches].sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()).slice(0, 6),
    [researches],
  );

  function prepareLaunch(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = prompt.trim();
    if (!value) return;
    setPendingPrompt(value);
    setError(null);
  }

  async function launch() {
    if (!pendingPrompt || !selectedSource) return;
    setStarting(true);
    setError(null);
    try {
      const created = await createResearch({
        original_prompt: pendingPrompt,
        gpu_source_id: selectedSource,
        target_gpu_allocation: allocation,
        auto_start: false,
      });
      setActiveResearch(created);
      setPrompt("");
      setPendingPrompt(null);
      let startFailure: string | null = null;
      try {
        const started = await controlResearch(created.id, "start");
        setActiveResearch(started);
      } catch (startError) {
        startFailure = startError instanceof Error ? startError.message : "Research was created but could not start.";
      }
      await load();
      if (startFailure) setError(startFailure);
      window.requestAnimationFrame(() => liveRef.current?.scrollIntoView({
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
        block: "start",
      }));
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not start research.");
    } finally {
      setStarting(false);
    }
  }

  return (
    <div className={`lab-page ${activeResearch ? "lab-page-active" : ""}`}>
      <h1 className="sr-only">Autoresearch Lab</h1>
      <section className="lab-composer-section">
        <GpuLineBackdrop utilization={telemetry?.utilization ?? null} />
        <form className="research-composer" onSubmit={prepareLaunch}>
          <label className="sr-only" htmlFor="research-prompt">What do you want to research?</label>
          <textarea
            id="research-prompt"
            rows={1}
            minLength={3}
            maxLength={4000}
            value={prompt}
            onChange={(event) => {
              const value = event.target.value;
              setPrompt(value);
              setPendingPrompt((current) => current == null ? null : value.trim() || null);
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder="What do you want to research?"
          />
          <button className="composer-send" type="submit" aria-label="Configure research" disabled={prompt.trim().length < 3}>
            <ArrowUp size={17} strokeWidth={2} />
          </button>
        </form>

        <AnimatePresence mode="wait">
          {pendingPrompt ? (
            <motion.div
              className="launch-config"
              key="launch-config"
              initial={{ opacity: 0, y: -8, height: 0 }}
              animate={{ opacity: 1, y: 0, height: "auto" }}
              exit={{ opacity: 0, y: -6, height: 0 }}
              transition={{ duration: 0.22 }}
              role="region"
              aria-label="Research run configuration"
            >
              <label>
                <span>GPU</span>
                <select ref={sourceSelectRef} value={selectedSource} onChange={(event) => setSelectedSource(event.target.value)}>
                  {sources.map((source) => <option value={source.id} key={source.id}>{source.name}</option>)}
                </select>
              </label>
              <label className="allocation-control">
                <span>GPU scheduling share <strong>{allocation}%</strong></span>
                <input type="range" min={10} max={100} step={5} value={allocation} onChange={(event) => setAllocation(Number(event.target.value))} />
                <small>{allocation}% requests this scheduling share while a GPU test is running.</small>
              </label>
              <button className="primary-button" type="button" disabled={starting || !selectedSource} onClick={() => void launch()}>
                {starting ? <LoaderCircle className="spin" size={16} /> : <ArrowUp size={16} />}
                Start research
              </button>
            </motion.div>
          ) : null}
        </AnimatePresence>

        {error ? <p className="inline-error lab-error" role="alert">{error}</p> : null}

        <AnimatePresence initial={false}>
          {!prompt.trim() && !pendingPrompt && !activeResearch ? (
            <motion.div
              className="recent-researches"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10, filter: "blur(5px)" }}
              transition={{ duration: 0.22, ease: "easeOut" }}
            >
              {loading ? (
                <div className="quiet-row centered"><LoaderCircle className="spin" size={15} /> Loading</div>
              ) : visibleResearches.length ? visibleResearches.map((research) => (
                <Link className="recent-row" href={`/researches/${research.id}`} key={research.id}>
                  <strong>{research.title}</strong>
                  <StatusPill status={research.status} />
                  <span>
                    {research.experiment_count ?? research.experiments?.length ?? 0} experiments
                    {research.best_value != null ? ` · ${metricLabel(research.metric_name)} ${formatMetric(research.best_value)}` : ""}
                  </span>
                  <ChevronRight size={15} aria-hidden="true" />
                </Link>
              )) : (
                <div className="quiet-row centered">No research has been started.</div>
              )}
            </motion.div>
          ) : null}
        </AnimatePresence>
      </section>

      {activeResearch ? (
        <div ref={liveRef} className="active-research-section">
          <ResearchLiveView researchId={activeResearch.id} initialResearch={activeResearch} />
        </div>
      ) : null}
    </div>
  );
}
