"use client";

import { AnimatePresence, MotionConfig, motion } from "framer-motion";
import { BookOpen, FlaskConical, Maximize2, Microscope, Minimize2, Settings, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { SettingsPanel } from "./settings-panel";

const navigation = [
  { href: "/lab", label: "Lab", icon: FlaskConical, match: "prefix" },
  { href: "/researches", label: "Researches", icon: Microscope, match: "prefix" },
  { href: "/articles", label: "Articles", icon: BookOpen, match: "prefix" },
] as const;

function isNavigationActive(pathname: string, item: (typeof navigation)[number]): boolean {
  return pathname === item.href || pathname.startsWith(`${item.href}/`);
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const settingsTriggerRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLElement>(null);
  const mainRef = useRef<HTMLElement>(null);
  const closeSettings = useCallback(() => setSettingsOpen(false), []);

  useEffect(() => {
    if (!settingsOpen) return;
    const previousOverflow = document.body.style.overflow;
    const returnFocus = settingsTriggerRef.current;
    document.body.style.overflow = "hidden";
    const frame = window.requestAnimationFrame(() => {
      dialogRef.current?.querySelector<HTMLElement>("button, input, select, textarea, [href], [tabindex]:not([tabindex='-1'])")?.focus();
    });
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeSettings();
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = Array.from(
        dialogRef.current.querySelectorAll<HTMLElement>(
          "button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [href], [tabindex]:not([tabindex='-1'])",
        ),
      );
      if (!focusable.length) {
        event.preventDefault();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      window.cancelAnimationFrame(frame);
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", handleKeyDown);
      returnFocus?.focus();
    };
  }, [closeSettings, settingsOpen]);

  useEffect(() => {
    const syncFullscreenState = () => setFullscreen(document.fullscreenElement === mainRef.current);
    document.addEventListener("fullscreenchange", syncFullscreenState);
    return () => document.removeEventListener("fullscreenchange", syncFullscreenState);
  }, []);

  const toggleFullscreen = useCallback(async () => {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
        return;
      }
      await mainRef.current?.requestFullscreen();
    } catch {
      setFullscreen(false);
    }
  }, []);

  return (
    <MotionConfig reducedMotion="user">
      <div className="app-shell">
        <div className="app-background" inert={settingsOpen ? true : undefined} aria-hidden={settingsOpen || undefined}>
        <nav className="floating-nav" aria-label="Primary navigation">
          {navigation.map((item) => {
            const { href, label, icon: Icon } = item;
            const active = isNavigationActive(pathname, item);
            return (
              <Link
                href={href}
                key={href}
                className={`nav-pill ${active ? "nav-pill-active" : ""}`}
                aria-current={active ? "page" : undefined}
              >
                <Icon size={16} strokeWidth={1.9} aria-hidden="true" />
                <span>{label}</span>
              </Link>
            );
          })}
        </nav>

        <button
          ref={settingsTriggerRef}
          className="settings-trigger"
          type="button"
          aria-label="GPU source settings"
          aria-expanded={settingsOpen}
          onClick={() => setSettingsOpen(true)}
        >
          <Settings size={18} strokeWidth={1.8} />
        </button>

        <main ref={mainRef} className={`app-main ${fullscreen ? "app-main-fullscreen" : ""}`}>
          {children}
          <button
            className="fullscreen-trigger"
            type="button"
            aria-label={fullscreen ? "Exit fullscreen monitoring" : "Open fullscreen monitoring"}
            title={fullscreen ? "Exit fullscreen" : "Fullscreen monitoring"}
            onClick={() => void toggleFullscreen()}
          >
            {fullscreen ? <Minimize2 size={17} aria-hidden="true" /> : <Maximize2 size={17} aria-hidden="true" />}
          </button>
        </main>
        </div>

        <AnimatePresence>
          {settingsOpen ? (
            <motion.div
            className="dialog-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onMouseDown={(event) => {
              if (event.currentTarget === event.target) closeSettings();
            }}
          >
            <motion.section
              ref={dialogRef}
              className="settings-dialog"
              role="dialog"
              aria-modal="true"
              aria-labelledby="gpu-settings-title"
              initial={{ opacity: 0, y: 14, scale: 0.985 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.99 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
            >
              <div className="dialog-heading">
                <h2 id="gpu-settings-title">GPU sources</h2>
                <button
                  className="icon-button"
                  type="button"
                  aria-label="Close settings"
                  onClick={closeSettings}
                >
                  <X size={18} />
                </button>
              </div>
              <SettingsPanel />
            </motion.section>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
    </MotionConfig>
  );
}
