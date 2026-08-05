"use client";

import { Check, ChevronDown, Cloud, HardDrive } from "lucide-react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { ResearcherModel } from "@/lib/types";

type MenuPosition = {
  left: number;
  top: number;
  width: number;
  maxHeight: number;
};

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
  const [menuPosition, setMenuPosition] = useState<MenuPosition | null>(null);
  const fieldRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
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
      const target = event.target as Node;
      if (!fieldRef.current?.contains(target) && !menuRef.current?.contains(target)) {
        setOpen(false);
      }
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

  useLayoutEffect(() => {
    if (!open) {
      return;
    }

    const placeMenu = () => {
      const trigger = triggerRef.current;
      if (!trigger) return;
      const rect = trigger.getBoundingClientRect();
      const gutter = 16;
      const width = Math.min(340, window.innerWidth - gutter * 2);
      const left = Math.max(gutter, Math.min(rect.right - width, window.innerWidth - width - gutter));
      const roomBelow = window.innerHeight - rect.bottom - gutter - 9;
      const roomAbove = rect.top - gutter - 9;
      const useAbove = roomBelow < 240 && roomAbove > roomBelow;
      const maxHeight = Math.max(180, Math.min(430, useAbove ? roomAbove : roomBelow));
      const top = useAbove
        ? Math.max(gutter, rect.top - maxHeight - 9)
        : rect.bottom + 9;
      setMenuPosition({ left, top, width, maxHeight });
    };

    placeMenu();
    window.addEventListener("resize", placeMenu);
    window.addEventListener("scroll", placeMenu, true);
    return () => {
      window.removeEventListener("resize", placeMenu);
      window.removeEventListener("scroll", placeMenu, true);
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

      {open && typeof document !== "undefined" ? createPortal(
        <div
          ref={menuRef}
          id={menuId}
          className="researcher-model-menu"
          role="listbox"
          aria-label="Research models"
          style={menuPosition ? {
            left: menuPosition.left,
            top: menuPosition.top,
            width: menuPosition.width,
            maxHeight: menuPosition.maxHeight,
          } : { visibility: "hidden" }}
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
        </div>,
        document.body,
      ) : null}

      {!compact ? (
        <small className={error ? "launch-field-error" : "launch-field-note"} role={error ? "alert" : undefined}>
          {error ?? selected?.description ?? "Choose the model that plans and runs the research."}
        </small>
      ) : null}
    </div>
  );
}
