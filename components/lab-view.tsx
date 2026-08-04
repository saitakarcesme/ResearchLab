"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ArrowUp, ChevronRight, Gauge, ListPlus, LoaderCircle, Play, Wrench } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createResearch, getGpuTelemetry, listGpuSources, listLocalModels, listResearches } from "@/lib/api";
import { formatMetric, metricLabel } from "@/lib/format";
import type { GpuSource, GpuTelemetry, LocalModel, Research, ResearchType } from "@/lib/types";
import { GpuLineBackdrop } from "./gpu-line-backdrop";
import { ResearchLiveView } from "./research-live-view";
import { StatusPill } from "./status-pill";

function modelName(modelId: string | null | undefined): string | null {
  return modelId?.replace(/^ollama:/, "").replace(/:latest$/, "") ?? null;
}

function modelOption(model: LocalModel): string {
  const details = [model.parameter_size, model.quantization_level].filter(Boolean).join(" · ");
  return `${model.name}${details ? ` — ${details}` : ""}${model.recommended ? " · recommended" : ""}`;
}

export function LabView() {
  const [prompt, setPrompt] = useState("");
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null);
  const [researches, setResearches] = useState<Research[]>([]);
  const [sources, setSources] = useState<GpuSource[]>([]);
  const [selectedSource, setSelectedSource] = useState("");
  const [researchType, setResearchType] = useState<ResearchType>("training_optimization");
  const [models, setModels] = useState<LocalModel[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [allocation, setAllocation] = useState(100);
  const [telemetry, setTelemetry] = useState<GpuTelemetry | null>(null);
  const [activeResearch, setActiveResearch] = useState<Research | null>(null);
  const [loading, setLoading] = useState(true);
  const [launching, setLaunching] = useState<"queue" | "run" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
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

  const selectedSourceType = sources.find((source) => source.id === selectedSource)?.type;

  useEffect(() => {
    if (!pendingPrompt || researchType !== "local_model_benchmark" || selectedSourceType !== "local" || !selectedSource) return;

    let cancelled = false;
    void listLocalModels(selectedSource)
      .then((catalog) => {
        if (cancelled) return;
        setModels(catalog.models);
        setModelError(null);
        setSelectedModel((current) => {
          if (catalog.models.some((model) => model.id === current)) return current;
          return catalog.models.find((model) => model.recommended)?.id ?? catalog.models[0]?.id ?? "";
        });
        if (!catalog.models.length) {
          setModelError("No completion-capable Ollama models are installed.");
        }
      })
      .catch((nextError) => {
        if (cancelled) return;
        setModels([]);
        setSelectedModel("");
        setModelError(nextError instanceof Error ? nextError.message : "Could not read the installed models.");
      })
      .finally(() => {
        if (!cancelled) setModelsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [pendingPrompt, researchType, selectedSource, selectedSourceType]);

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
    setNotice(null);
    if (researchType === "local_model_benchmark") setModelsLoading(true);
  }

  function chooseResearchType(nextType: ResearchType) {
    setResearchType(nextType);
    setError(null);
    setModelError(null);
    setModels([]);
    setSelectedModel("");
    setModelsLoading(nextType === "local_model_benchmark");
    if (nextType === "local_model_benchmark" && selectedSourceType !== "local") {
      const localSource = sources.find((source) => source.type === "local");
      if (localSource) {
        setSelectedSource(localSource.id);
      } else {
        setModelsLoading(false);
        setModelError("No local GPU source is configured.");
      }
    }
  }

  async function launch(autoStart: boolean) {
    if (!pendingPrompt || !selectedSource) return;
    if (researchType === "local_model_benchmark" && !selectedModel) {
      setError("Choose an installed local model before continuing.");
      return;
    }
    setLaunching(autoStart ? "run" : "queue");
    setError(null);
    setNotice(null);
    try {
      const created = await createResearch({
        original_prompt: pendingPrompt,
        gpu_source_id: selectedSource,
        target_gpu_allocation: allocation,
        auto_start: autoStart,
        research_type: researchType,
        ...(researchType === "local_model_benchmark"
          ? { model_id: selectedModel, benchmark_profile: "ollama-text-v1" as const }
          : {}),
      });
      setPrompt("");
      setPendingPrompt(null);
      await load();
      if (autoStart) {
        setActiveResearch(created);
        window.requestAnimationFrame(() => liveRef.current?.scrollIntoView({
          behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
          block: "start",
        }));
      } else {
        setNotice(created.queue_position
          ? `Added to queue at position ${created.queue_position}.`
          : "Added to the research queue.");
      }
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not create research.");
    } finally {
      setLaunching(null);
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
              <fieldset className="launch-methods">
                <legend>What should the lab test?</legend>
                <label className={researchType === "training_optimization" ? "launch-method-active" : ""}>
                  <input
                    type="radio"
                    name="research-method"
                    value="training_optimization"
                    checked={researchType === "training_optimization"}
                    onChange={() => chooseResearchType("training_optimization")}
                  />
                  <Wrench size={16} aria-hidden="true" />
                  <span><strong>Training optimization</strong><small>Improve the built-in training run.</small></span>
                </label>
                <label className={researchType === "local_model_benchmark" ? "launch-method-active" : ""}>
                  <input
                    type="radio"
                    name="research-method"
                    value="local_model_benchmark"
                    checked={researchType === "local_model_benchmark"}
                    onChange={() => chooseResearchType("local_model_benchmark")}
                  />
                  <Gauge size={16} aria-hidden="true" />
                  <span><strong>Local model benchmark</strong><small>Measure a real installed Ollama model.</small></span>
                </label>
              </fieldset>

              <div className="launch-fields">
                <label>
                  <span>GPU</span>
                  <select
                    ref={sourceSelectRef}
                    value={selectedSource}
                    onChange={(event) => {
                      setSelectedSource(event.target.value);
                      if (researchType === "local_model_benchmark") {
                        setModelsLoading(true);
                        setModelError(null);
                      }
                    }}
                  >
                    {sources.map((source) => (
                      <option
                        value={source.id}
                        key={source.id}
                        disabled={researchType === "local_model_benchmark" && source.type !== "local"}
                      >
                        {source.name}{source.type === "remote" ? " · remote" : " · this computer"}
                      </option>
                    ))}
                  </select>
                </label>

                {researchType === "local_model_benchmark" ? (
                  <label>
                    <span>Installed model</span>
                    <select
                      value={selectedModel}
                      disabled={modelsLoading || !models.length}
                      onChange={(event) => setSelectedModel(event.target.value)}
                    >
                      {modelsLoading ? <option value="">Reading Ollama models…</option> : null}
                      {!modelsLoading && !models.length ? <option value="">No compatible models found</option> : null}
                      {models.map((model) => <option value={model.id} key={model.id}>{modelOption(model)}</option>)}
                    </select>
                    <small className="launch-field-note">Four repeatable profiles; speed is measured in output tokens per second.</small>
                    {modelError ? <small className="launch-field-error" role="alert">{modelError}</small> : null}
                  </label>
                ) : null}

                <label className="allocation-control">
                  <span>GPU scheduling share <strong>{allocation}%</strong></span>
                  <input type="range" min={10} max={100} step={5} value={allocation} onChange={(event) => setAllocation(Number(event.target.value))} />
                  <small>{allocation}% requests this scheduling share while a GPU test is running.</small>
                </label>
              </div>

              <div className="launch-actions">
                <button
                  className="queue-button"
                  type="button"
                  disabled={launching != null || !selectedSource || (researchType === "local_model_benchmark" && !selectedModel)}
                  onClick={() => void launch(false)}
                >
                  {launching === "queue" ? <LoaderCircle className="spin" size={16} /> : <ListPlus size={16} />}
                  Add to queue
                </button>
                <button
                  className="primary-button"
                  type="button"
                  disabled={launching != null || !selectedSource || (researchType === "local_model_benchmark" && !selectedModel)}
                  onClick={() => void launch(true)}
                >
                  {launching === "run" ? <LoaderCircle className="spin" size={16} /> : <Play size={15} fill="currentColor" />}
                  Run now
                </button>
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>

        {error ? <p className="inline-error lab-error" role="alert">{error}</p> : null}
        {notice ? <p className="launch-notice" role="status">{notice}</p> : null}

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
                  <StatusPill status={research.status} queuePosition={research.queue_position} />
                  <span>
                    {research.status === "queued"
                      ? `${modelName(research.model_id) ?? "Training run"} · waiting for GPU`
                      : `${research.experiment_count ?? research.experiments?.length ?? 0} experiments${research.best_value != null ? ` · ${metricLabel(research.metric_name)} ${formatMetric(research.best_value)}` : ""}`}
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
