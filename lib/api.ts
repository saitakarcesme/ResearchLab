import type {
  Article,
  BenchmarkProfile,
  GpuSource,
  GpuTelemetry,
  LocalModelCatalog,
  Research,
  ResearchType,
} from "./types";

type ResearchPayload = Research & { gpu_name?: string | null };

interface TelemetryPayload {
  available: boolean;
  sampled_at?: string;
  error?: string | null;
  gpu_name?: string | null;
  name?: string | null;
  utilization_percent?: number | null;
  utilization?: number | null;
  temperature_c?: number | null;
  temperature?: number | null;
  memory_used_mb?: number | null;
  memory_total_mb?: number | null;
  processes?: Array<{
    pid?: number | null;
    process_name?: string;
    name?: string;
    memory_used_mb?: number | null;
    used_memory_mb?: number | null;
  }>;
}

export const API_BASE = (
  process.env.NEXT_PUBLIC_RESEARCH_API_URL ?? "http://127.0.0.1:7331"
).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (init?.body != null && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
    cache: "no-store",
  });

  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as {
        detail?: string | Array<{ msg?: string; loc?: Array<string | number> }>;
        error?: string;
      };
      if (Array.isArray(body.detail)) {
        message = body.detail
          .map((item) => {
            const field = item.loc?.at(-1);
            return `${field ? `${String(field)}: ` : ""}${item.msg ?? "Invalid value"}`;
          })
          .join(" · ");
      } else {
        message = body.detail ?? body.error ?? message;
      }
    } catch {
      // Keep the status-based fallback when the service returns no JSON body.
    }
    throw new ApiError(message, response.status);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function unwrapList<T>(value: T[] | { items?: T[]; data?: T[] }): T[] {
  if (Array.isArray(value)) return value;
  return value.items ?? value.data ?? [];
}

function normalizeResearch(value: ResearchPayload): Research {
  return {
    ...value,
    gpu_source_name: value.gpu_source_name ?? value.gpu_name ?? null,
  };
}

function normalizeTelemetry(value: TelemetryPayload): GpuTelemetry {
  return {
    available: value.available,
    name: value.name ?? value.gpu_name ?? null,
    utilization: value.utilization ?? value.utilization_percent ?? null,
    temperature: value.temperature ?? value.temperature_c ?? null,
    memory_used_mb: value.memory_used_mb ?? null,
    memory_total_mb: value.memory_total_mb ?? null,
    sampled_at: value.sampled_at,
    error: value.error ?? null,
    processes: (value.processes ?? []).map((process) => ({
      pid: process.pid ?? 0,
      name: process.name ?? process.process_name ?? "GPU process",
      used_memory_mb: process.used_memory_mb ?? process.memory_used_mb ?? null,
    })),
  };
}

export async function listResearches(): Promise<Research[]> {
  const values = unwrapList(await request<ResearchPayload[] | { items: ResearchPayload[] }>("/api/researches"));
  return values.map(normalizeResearch);
}

export async function getResearch(id: string): Promise<Research> {
  return normalizeResearch(await request<ResearchPayload>(`/api/researches/${encodeURIComponent(id)}`));
}

export async function createResearch(input: {
  original_prompt: string;
  gpu_source_id: string;
  target_gpu_allocation: number;
  auto_start: boolean;
  research_type: ResearchType;
  model_id?: string;
  benchmark_profile?: BenchmarkProfile;
}): Promise<Research> {
  return normalizeResearch(await request<ResearchPayload>("/api/researches", {
    method: "POST",
    body: JSON.stringify(input),
  }));
}

export async function controlResearch(
  id: string,
  action: "start" | "pause" | "resume" | "stop" | "enqueue" | "dequeue",
): Promise<Research> {
  return normalizeResearch(await request<ResearchPayload>(`/api/researches/${encodeURIComponent(id)}/${action}`, {
    method: "POST",
    body: "{}",
  }));
}

export function createArticle(researchId: string): Promise<Article> {
  return request(`/api/researches/${encodeURIComponent(researchId)}/article`, {
    method: "POST",
    body: "{}",
  });
}

export async function listGpuSources(): Promise<GpuSource[]> {
  return unwrapList(await request<GpuSource[] | { items: GpuSource[] }>("/api/gpu-sources"));
}

export function listLocalModels(sourceId: string): Promise<LocalModelCatalog> {
  return request(`/api/gpu-sources/${encodeURIComponent(sourceId)}/models`);
}

export function createGpuSource(input: {
  name: string;
  type: "local" | "remote";
  host: string | null;
  port: number;
  username: string | null;
  auth_method: "agent" | "key_env";
  workspace_path: string;
}): Promise<GpuSource> {
  return request("/api/gpu-sources", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function getGpuTelemetry(sourceId?: string | null): Promise<GpuTelemetry> {
  const path = sourceId
    ? `/api/gpu-sources/${encodeURIComponent(sourceId)}/telemetry`
    : "/api/telemetry/local";
  return normalizeTelemetry(await request<TelemetryPayload>(path));
}

export async function listArticles(): Promise<Article[]> {
  return unwrapList(await request<Article[] | { items: Article[] }>("/api/articles"));
}

export function getArticle(identifier: string): Promise<Article> {
  return request(`/api/articles/${encodeURIComponent(identifier)}`);
}

export function researchEventsUrl(id: string): string {
  return `${API_BASE}/api/researches/${encodeURIComponent(id)}/events`;
}
