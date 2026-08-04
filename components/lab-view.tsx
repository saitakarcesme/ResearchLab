"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ArrowUp, CalendarClock, ChevronRight, Gauge, ListPlus, LoaderCircle, Play, Wrench } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createResearch, getGpuTelemetry, listGpuSources, listLocalModels, listResearcherModels, listResearches } from "@/lib/api";
import { formatMetric, formatTokenCount, metricLabel } from "@/lib/format";
import { modelDisplayName, modelProviderLabel } from "@/lib/model-label";
import type { GpuSource, GpuTelemetry, LocalModel, Research, ResearcherModel, ResearchType } from "@/lib/types";
import { GpuLineBackdrop } from "./gpu-line-backdrop";
import { ResearchLiveView } from "./research-live-view";
import { ResearcherModelSelect } from "./researcher-model-select";
import { LabTotalTokenUsage } from "./research-runtime-metrics";
import { StatusPill } from "./status-pill";

function modelOption(model: LocalModel): string {
  const details = [model.parameter_size, model.quantization_level].filter(Boolean).join(" · ");
  const pinned = model.provider === "huggingface" ? " · pinned" : "";
  return `${modelProviderLabel(model.provider)} · ${model.name}${details ? ` — ${details}` : ""}${pinned}${model.recommended ? " · recommended" : ""}`;
}

export function LabView() {
  const router = useRouter();
  const [prompt, setPrompt] = useState("");
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null);
  const [researches, setResearches] = useState<Research[]>([]);
  const [sources, setSources] = useState<GpuSource[]>([]);
  const [selectedSource, setSelectedSource] = useState("");
  const [researchType, setResearchType] = useState<ResearchType>("training_optimization");
  const [researcherModels, setResearcherModels] = useState<ResearcherModel[]>([]);
  const [selectedResearcherModel, setSelectedResearcherModel] = useState("");
  const [researcherModelsLoading, setResearcherModelsLoading] = useState(true);
  const [researcherModelError, setResearcherModelError] = useState<string | null>(null);
  const [models, setModels] = useState<LocalModel[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [modelsLoading, setModelsLoading] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [allocation, setAllocation] = useState(100);
  const [scheduleEnabled, setScheduleEnabled] = useState(false);
  const [scheduleStart, setScheduleStart] = useState("22:00");
  const [scheduleEnd, setScheduleEnd] = useState("07:00");
  const [telemetry, setTelemetry] = useState<GpuTelemetry | null>(null);
  const [loading, setLoading] = useState(true);
  const [launching, setLaunching] = useState<"queue" | "run" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const sourceSelectRef = useRef<HTMLSelectElement>(null);
  const wasConfiguringRef = useRef(false);
  const requiresFullGpu = selectedModel.startsWith("huggingface:");
  const effectiveAllocation = requiresFullGpu ? 100 : allocation;
  const scheduleInvalid = scheduleEnabled && scheduleStart === scheduleEnd;

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
    const syncFullscreen = () => setFullscreen(document.fullscreenElement != null);
    document.addEventListener("fullscreenchange", syncFullscreen);
    return () => document.removeEventListener("fullscreenchange", syncFullscreen);
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => void load(), 0);
    const timer = window.setInterval(() => void load(), 2000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [load]);

  useEffect(() => {
    let mounted = true;
    void listResearcherModels()
      .then((catalog) => {
        if (!mounted) return;
        setResearcherModels(catalog.models);
        setSelectedResearcherModel(
          catalog.models.some((model) => model.id === catalog.default_model_id)
            ? catalog.default_model_id
            : catalog.models.find((model) => model.recommended)?.id ?? catalog.models[0]?.id ?? "",
        );
        setResearcherModelError(catalog.models.length ? null : "No research agent is currently available.");
      })
      .catch((nextError) => {
        if (!mounted) return;
        setResearcherModels([]);
        setSelectedResearcherModel("");
        setResearcherModelError(nextError instanceof Error ? nextError.message : "Could not read research agents.");
      })
      .finally(() => {
        if (mounted) setResearcherModelsLoading(false);
      });
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
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
  }, [selectedSource]);

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
          setModelError("No compatible Ollama or pinned Hugging Face models are available.");
        }
      })
      .catch((nextError) => {
        if (cancelled) return;
        setModels([]);
        setSelectedModel("");
        setModelError(nextError instanceof Error ? nextError.message : "Could not read the model catalog.");
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
  const totalTokenUsage = useMemo(
    () => researches.reduce((total, research) => total + Math.max(0, research.total_tokens ?? 0), 0),
    [researches],
  );
  const totalInputTokens = useMemo(
    () => researches.reduce((total, research) => total + Math.max(0, research.input_tokens ?? 0), 0),
    [researches],
  );
  const totalOutputTokens = useMemo(
    () => researches.reduce((total, research) => total + Math.max(0, research.output_tokens ?? 0), 0),
    [researches],
  );
  const monitorResearch = researches.find((research) => research.status === "running")
    ?? researches.find((research) => research.status === "paused")
    ?? visibleResearches[0]
    ?? null;

  function prepareLaunch(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = prompt.trim();
    if (!value) return;
    setPendingPrompt(value);
    setError(null);
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
    if (!selectedResearcherModel) {
      setError("Choose the research agent that should run this lab work.");
      return;
    }
    if (researchType === "local_model_benchmark" && !selectedModel) {
      setError("Choose a benchmark model before continuing.");
      return;
    }
    setLaunching(autoStart ? "run" : "queue");
    setError(null);
    try {
      const scheduled = scheduleEnabled;
      const shouldStartNow = autoStart && !scheduled;
      const created = await createResearch({
        original_prompt: pendingPrompt,
        gpu_source_id: selectedSource,
        target_gpu_allocation: effectiveAllocation,
        auto_start: shouldStartNow,
        research_type: researchType,
        researcher_model_id: selectedResearcherModel,
        ...(scheduled
          ? {
              schedule_start_time: scheduleStart,
              schedule_end_time: scheduleEnd,
              schedule_timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC",
              schedule_utc_offset_minutes: -new Date().getTimezoneOffset(),
            }
          : {}),
        ...(researchType === "local_model_benchmark"
          ? {
              model_id: selectedModel,
              benchmark_profile: selectedModel.startsWith("huggingface:")
                ? "hf-transformers-text-v1" as const
                : "ollama-text-v1" as const,
            }
          : {}),
      });
      setPrompt("");
      setPendingPrompt(null);
      router.push(`/researches/${created.id}`);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not create research.");
    } finally {
      setLaunching(null);
    }
  }

  return (
    <div className="lab-page">
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
          <ResearcherModelSelect
            id="lab-researcher-model"
            models={researcherModels}
            value={selectedResearcherModel}
            loading={researcherModelsLoading}
            error={researcherModelError}
            compact
            onChange={setSelectedResearcherModel}
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
                  <span><strong>Local model benchmark</strong><small>Measure an Ollama or pinned Hugging Face model.</small></span>
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
                    <span>Model under test</span>
                    <select
                      value={selectedModel}
                      disabled={modelsLoading || !models.length}
                      onChange={(event) => setSelectedModel(event.target.value)}
                    >
                      {modelsLoading ? <option value="">Reading model catalog…</option> : null}
                      {!modelsLoading && !models.length ? <option value="">No compatible models found</option> : null}
                      {models.map((model) => <option value={model.id} key={model.id}>{modelOption(model)}</option>)}
                    </select>
                    <small className="launch-field-note">Provider and pinned model revision are recorded; speed is measured in output tokens per second.</small>
                    {modelError ? <small className="launch-field-error" role="alert">{modelError}</small> : null}
                  </label>
                ) : null}

                <label className="allocation-control">
                  <span>GPU scheduling share <strong>{effectiveAllocation}%</strong></span>
                  <input
                    type="range"
                    min={10}
                    max={100}
                    step={5}
                    value={effectiveAllocation}
                    disabled={requiresFullGpu}
                    onChange={(event) => setAllocation(Number(event.target.value))}
                  />
                  <small>
                    {requiresFullGpu
                      ? "The pinned Hugging Face benchmark reserves the full scheduler share."
                      : `${effectiveAllocation}% requests this scheduling share while a GPU test is running.`}
                  </small>
                </label>

                <div className="schedule-window-control">
                  <span className="schedule-window-heading">
                    <CalendarClock size={14} aria-hidden="true" />
                    <input
                      type="checkbox"
                      checked={scheduleEnabled}
                      onChange={(event) => setScheduleEnabled(event.target.checked)}
                    />
                    Run in a time window
                  </span>
                  {scheduleEnabled ? (
                    <span className="schedule-time-inputs">
                      <input aria-label="Schedule start time" type="time" value={scheduleStart} onChange={(event) => setScheduleStart(event.target.value)} />
                      <i aria-hidden="true">to</i>
                      <input aria-label="Schedule end time" type="time" value={scheduleEnd} onChange={(event) => setScheduleEnd(event.target.value)} />
                    </span>
                  ) : null}
                  <small className={scheduleInvalid ? "launch-field-error" : undefined}>
                    {scheduleInvalid
                      ? "Choose two different times."
                      : scheduleEnabled
                        ? "The queued run starts only inside this daily local-time window."
                        : "Optional: keep heavy local work for quieter hours."}
                  </small>
                </div>
              </div>

              {researcherModels.find((model) => model.id === selectedResearcherModel)?.provider === "ollama" ? (
                <p className="launch-resource-note">Local research agents share this computer&apos;s RAM, CPU and GPU. Large models can make the desktop less responsive; use the time window for unattended runs.</p>
              ) : null}

              <div className="launch-actions">
                <button
                  className="queue-button"
                  type="button"
                  disabled={scheduleInvalid || launching != null || !selectedSource || !selectedResearcherModel || (researchType === "local_model_benchmark" && !selectedModel)}
                  onClick={() => void launch(false)}
                >
                  {launching === "queue" ? <LoaderCircle className="spin" size={16} /> : <ListPlus size={16} />}
                  Add to queue
                </button>
                <button
                  className="primary-button"
                  type="button"
                  disabled={scheduleInvalid || launching != null || !selectedSource || !selectedResearcherModel || (researchType === "local_model_benchmark" && !selectedModel)}
                  onClick={() => void launch(true)}
                >
                  {launching === "run" ? <LoaderCircle className="spin" size={16} /> : <Play size={15} fill="currentColor" />}
                  {scheduleEnabled ? "Schedule" : "Run now"}
                </button>
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>

        {error || researcherModelError ? <p className="inline-error lab-error" role="alert">{error ?? researcherModelError}</p> : null}
        <AnimatePresence initial={false}>
          {!prompt.trim() && !pendingPrompt ? (
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
                      ? `${modelDisplayName(research.model_id) ?? "Training run"} · waiting for GPU`
                      : `${research.experiment_count ?? research.experiments?.length ?? 0} experiments · ${formatTokenCount(research.total_tokens)} agent tokens${research.best_value != null ? ` · ${metricLabel(research.metric_name)} ${formatMetric(research.best_value)}` : ""}`}
                  </span>
                  <ChevronRight size={15} aria-hidden="true" />
                </Link>
              )) : (
                <div className="quiet-row centered">No research has been started.</div>
              )}
            </motion.div>
          ) : null}
        </AnimatePresence>
        {!loading && !prompt.trim() && !pendingPrompt ? (
          <LabTotalTokenUsage value={totalTokenUsage} input={totalInputTokens} output={totalOutputTokens} />
        ) : null}
      </section>

      {fullscreen && monitorResearch ? (
        <div className="lab-fullscreen-monitor">
          <ResearchLiveView researchId={monitorResearch.id} initialResearch={monitorResearch} />
        </div>
      ) : null}
    </div>
  );
}
