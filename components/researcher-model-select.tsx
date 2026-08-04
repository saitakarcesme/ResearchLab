"use client";

import { Check, ChevronDown, Cloud, HardDrive } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ResearcherModel } from "@/lib/types";

export function ResearcherModelSelect({
  id,
  models,
  value,
  loading = false,
  error = null,
  compact = false,
  onChange,
}: {
  id: string;
  models: ResearcherModel[];
  value: string;
  loading?: boolean;
  error?: string | null;
  compact?: boolean;
  onChange: (modelId: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const fieldRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const selected = models.find((model) => model.id === value);
  const groupedModels = useMemo(() => [
    {
      id: "cloud",
      label: "GPT models",
      icon: Cloud,
      models: models.filter((model) => model.provider === "codex"),
    },
    {
      id: "local",
      label: "Local · Ollama",
      icon: HardDrive,
      models: models.filter((model) => model.provider === "ollama"),
    },
  ].filter((group) => group.models.length), [models]);
  const flatModels = useMemo(
    () => groupedModels.flatMap((group) => group.models),
    [groupedModels],
  );
  const optionIndexById = useMemo(
    () => new Map(flatModels.map((model, index) => [model.id, index])),
    [flatModels],
  );
  const menuId = `${id}-menu`;

  useEffect(() => {
    if (!open) return;
    const closeOnOutsidePress = (event: PointerEvent) => {
      if (!fieldRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setOpen(false);
      triggerRef.current?.focus();
    };
    document.addEventListener("pointerdown", closeOnOutsidePress);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsidePress);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  const focusOption = (direction: 1 | -1) => {
    if (!flatModels.length) return;
    const currentIndex = optionRefs.current.findIndex((option) => option === document.activeElement);
    const selectedIndex = Math.max(0, flatModels.findIndex((model) => model.id === value));
    const nextIndex = currentIndex < 0
      ? selectedIndex
      : (currentIndex + direction + flatModels.length) % flatModels.length;
    optionRefs.current[nextIndex]?.focus();
  };

  return (
    <div
      ref={fieldRef}
      className={`researcher-model-field ${compact ? "researcher-model-field-compact" : ""} ${open ? "researcher-model-field-open" : ""}`}
    >
      <span className={compact ? "sr-only" : undefined}>Research agent</span>
      <button
        ref={triggerRef}
        id={id}
        className="researcher-model-trigger"
        type="button"
        disabled={loading || !models.length}
        aria-label="Research agent"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        onClick={() => setOpen((current) => !current)}
        onKeyDown={(event) => {
          if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
          event.preventDefault();
          if (!open) {
            setOpen(true);
            window.requestAnimationFrame(() => focusOption(event.key === "ArrowDown" ? 1 : -1));
          }
        }}
      >
        <span>{loading ? "Loading models…" : selected?.label ?? "Choose model"}</span>
        <ChevronDown size={14} aria-hidden="true" />
      </button>

      {open ? (
        <div
          id={menuId}
          className="researcher-model-menu"
          role="listbox"
          aria-label="Research models"
          onKeyDown={(event) => {
            if (event.key === "ArrowDown" || event.key === "ArrowUp") {
              event.preventDefault();
              focusOption(event.key === "ArrowDown" ? 1 : -1);
            }
          }}
        >
          <div className="researcher-model-menu-heading">
            <span>Research model</span>
            <small>{flatModels.length} available</small>
          </div>
          {groupedModels.map((group) => {
            const Icon = group.icon;
            return (
              <div className="researcher-model-group" key={group.id}>
                <div className="researcher-model-group-label">
                  <Icon size={13} aria-hidden="true" />
                  <span>{group.label}</span>
                </div>
                {group.models.map((model) => {
                  const optionIndex = optionIndexById.get(model.id) ?? 0;
                  const active = model.id === value;
                  return (
                    <button
                      ref={(node) => { optionRefs.current[optionIndex] = node; }}
                      className={`researcher-model-option ${active ? "researcher-model-option-active" : ""}`}
                      type="button"
                      role="option"
                      aria-selected={active}
                      key={model.id}
                      onClick={() => {
                        onChange(model.id);
                        setOpen(false);
                        triggerRef.current?.focus();
                      }}
                    >
                      <span>
                        <strong>{model.label}</strong>
                        <small>{model.description}</small>
                      </span>
                      <span className="researcher-model-option-meta">
                        {model.recommended ? <em>Suggested</em> : null}
                        {model.parameter_size ? <small>{model.parameter_size}</small> : null}
                        {active ? <Check size={14} aria-hidden="true" /> : null}
                      </span>
                    </button>
                  );
                })}
              </div>
            );
          })}
          {error ? <p className="researcher-model-menu-error" role="alert">{error}</p> : null}
        </div>
      ) : null}

      {!compact ? (
        <small className={error ? "launch-field-error" : "launch-field-note"} role={error ? "alert" : undefined}>
          {error ?? selected?.description ?? "Choose the model that plans and runs the research."}
        </small>
      ) : null}
    </div>
  );
}
