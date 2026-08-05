"use client";

import { Check, LoaderCircle, Plus } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { createGpuSource, listGpuSources } from "@/lib/api";
import type { GpuSource } from "@/lib/types";

const emptyForm = {
  name: "",
  type: "remote" as "local" | "remote",
  host: "",
  port: 22,
  username: "",
  auth_method: "agent" as "agent" | "key_env",
  workspace_path: "",
};

export function SettingsPanel() {
  const [sources, setSources] = useState<GpuSource[]>([]);
  const [form, setForm] = useState(emptyForm);
  const [showForm, setShowForm] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setSources(await listGpuSources());
      setError(null);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "GPU sources are unavailable.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(initial);
  }, [load]);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      await createGpuSource({
        ...form,
        host: form.type === "local" ? null : form.host || null,
        username: form.type === "local" ? null : form.username || null,
      });
      setForm(emptyForm);
      setShowForm(false);
      await load();
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not save the GPU source.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="settings-content">
      <div className="source-list" aria-live="polite">
        {loading ? (
          <div className="quiet-row"><LoaderCircle className="spin" size={16} /> Reading sources</div>
        ) : sources.length ? (
          sources.map((source) => (
            <div className="source-row" key={source.id}>
              <span className="source-dot" aria-hidden="true" />
              <div>
                <strong>{source.name}</strong>
                <span>{source.type === "local" ? "Local" : `${source.username ?? "ssh"}@${source.host}`}</span>
              </div>
              <code>{source.workspace_path}</code>
            </div>
          ))
        ) : (
          <div className="quiet-row">No GPU source is configured.</div>
        )}
      </div>

      {error ? <p className="inline-error" role="alert">{error}</p> : null}

      {showForm ? (
        <form className="source-form" onSubmit={submit}>
          <div className="field-grid two-columns">
            <label>
              <span>Name</span>
              <input
                required
                value={form.name}
                onChange={(event) => setForm({ ...form, name: event.target.value })}
                placeholder="Studio 3090"
              />
            </label>
            <label>
              <span>Connection</span>
              <select
                value={form.type}
                onChange={(event) => setForm({ ...form, type: event.target.value as "local" | "remote" })}
              >
                <option value="local" disabled>Local (auto-detected)</option>
                <option value="remote">Remote SSH</option>
              </select>
            </label>
          </div>

          {form.type === "remote" ? (
            <div className="field-grid remote-grid">
              <label>
                <span>Host</span>
                <input required value={form.host} onChange={(event) => setForm({ ...form, host: event.target.value })} placeholder="192.168.1.42" />
              </label>
              <label>
                <span>Port</span>
                <input type="number" min={1} max={65535} value={form.port} onChange={(event) => setForm({ ...form, port: Number(event.target.value) })} />
              </label>
              <label>
                <span>Username</span>
                <input required value={form.username} onChange={(event) => setForm({ ...form, username: event.target.value })} placeholder="research" />
              </label>
              <label>
                <span>Authentication</span>
                <select value={form.auth_method} onChange={(event) => setForm({ ...form, auth_method: event.target.value as "agent" | "key_env" })}>
                  <option value="agent">SSH agent</option>
                  <option value="key_env">Key path from environment</option>
                </select>
              </label>
            </div>
          ) : null}

          <label>
            <span>Workspace path</span>
            <input required value={form.workspace_path} onChange={(event) => setForm({ ...form, workspace_path: event.target.value })} placeholder={form.type === "local" ? "C:\\research\\workspaces" : "/home/research/workspaces"} />
          </label>

          <p className="form-note">Keys and passwords stay in your SSH agent or environment; they are never stored here.</p>
          <div className="form-actions">
            <button className="text-button" type="button" onClick={() => setShowForm(false)}>Cancel</button>
            <button className="primary-button compact" type="submit" disabled={saving}>
              {saving ? <LoaderCircle className="spin" size={15} /> : <Check size={15} />}
              Save source
            </button>
          </div>
        </form>
      ) : (
        <button className="add-source-button" type="button" onClick={() => setShowForm(true)}>
          <Plus size={16} /> Add GPU source
        </button>
      )}
    </div>
  );
}
