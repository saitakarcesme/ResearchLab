export type ResearchStatus =
  | "queued"
  | "running"
  | "paused"
  | "completed"
  | "stopped"
  | "failed";

export type MetricDirection = "lower_is_better" | "higher_is_better";

export type ResearchType = "training_optimization" | "local_model_benchmark";

export type BenchmarkProfile = "ollama-text-v1" | "hf-transformers-text-v1";

export interface Experiment {
  id: string;
  research_id: string;
  experiment_number: number;
  hypothesis: string;
  change_summary: string;
  metric_value: number | null;
  previous_best: number | null;
  accepted: boolean | null;
  started_at: string | null;
  completed_at: string | null;
  git_commit: string | null;
  error: string | null;
  token_count?: number;
}

export interface ResearchLog {
  id: string | number;
  research_id: string;
  experiment_id?: string | null;
  level?: "info" | "warning" | "error";
  event_type: string;
  message: string;
  created_at: string;
  data?: Record<string, unknown> | null;
}

export interface Research {
  id: string;
  title: string;
  original_prompt: string;
  objective?: string;
  status: ResearchStatus;
  metric_name: string;
  metric_direction: MetricDirection;
  research_type: ResearchType;
  researcher_model_id: string | null;
  model_id: string | null;
  model_digest: string | null;
  model_runtime: string | null;
  benchmark_profile: BenchmarkProfile | null;
  queue_position?: number | null;
  schedule_start_time?: string | null;
  schedule_end_time?: string | null;
  schedule_timezone?: string | null;
  schedule_utc_offset_minutes?: number | null;
  baseline_value: number | null;
  best_value: number | null;
  gpu_source_id: string | null;
  gpu_source_name?: string | null;
  target_gpu_allocation: number;
  workspace_path: string | null;
  created_at: string;
  updated_at: string;
  experiment_count?: number;
  total_tokens?: number;
  input_tokens?: number;
  cached_input_tokens?: number;
  output_tokens?: number;
  reasoning_output_tokens?: number;
  training_tokens?: number;
  experiments?: Experiment[];
  logs?: ResearchLog[];
  runtime?: {
    phase?: string;
    phase_changed_at?: string | null;
    allocation?: {
      target_percent?: number | null;
      mode?: string;
      target_role?: string;
      target_enforced?: boolean;
      is_hard_utilization_target?: boolean;
    };
  };
}

export interface ResearcherModel {
  id: string;
  label: string;
  description: string;
  provider: "codex" | "ollama";
  parameter_size?: string | null;
  capabilities?: string[];
  recommended: boolean;
}

export interface ResearcherModelCatalog {
  default_model_id: string;
  models: ResearcherModel[];
  local_error?: string | null;
}

export interface LocalModel {
  id: string;
  provider: "ollama" | "huggingface";
  name: string;
  digest: string;
  size_bytes: number;
  adapter_size_bytes?: number;
  base_download_size_bytes?: number;
  runtime_memory_footprint_bytes?: number;
  parameter_size: string | null;
  quantization_level: string | null;
  family: string | null;
  capabilities: string[];
  context_length: number | null;
  recommended: boolean;
}

export interface LocalModelCatalog {
  runtime: {
    provider: "ollama";
    endpoint: string;
    version: string | null;
  };
  models: LocalModel[];
}

export interface GpuProcess {
  pid: number;
  name: string;
  used_memory_mb: number | null;
}

export interface GpuTelemetry {
  available: boolean;
  name: string | null;
  utilization: number | null;
  temperature: number | null;
  memory_used_mb: number | null;
  memory_total_mb: number | null;
  processes: GpuProcess[];
  sampled_at?: string;
  error?: string | null;
}

export interface GpuSource {
  id: string;
  name: string;
  type: "local" | "remote";
  host: string | null;
  port: number;
  username: string | null;
  auth_method?: "agent" | "key_env";
  workspace_path: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface CloudAccount {
  id: string;
  provider: "shadeform" | "runpod";
  name: string;
  budget_usd: number;
  enabled: boolean;
  settings: { ssh_key_id?: string | null; workspace_path?: string; image_name?: string | null };
  billing_url: string;
  created_at: string;
  updated_at: string;
}

export interface CloudOffer {
  id: string;
  name: string;
  gpu: string;
  vram_gb: number | null;
  region: string;
  hourly_price_usd: number;
  available: boolean;
}

export interface CloudGpuInstance {
  id: string;
  provider_account_id: string;
  provider: "shadeform" | "runpod";
  provider_account_name: string;
  gpu_source_id: string | null;
  name: string;
  status: string;
  hourly_price_usd: number;
  max_spend_usd: number;
  estimated_spend_usd: number;
  host: string | null;
  created_at: string;
}

export interface Article {
  id: string;
  slug: string;
  research_id: string;
  title: string;
  markdown?: string;
  article_kind?: "general" | "technical";
  metric_name?: string;
  metric_direction?: MetricDirection;
  objective?: string;
  baseline_value?: number | null;
  best_value?: number | null;
  experiments?: Experiment[];
  created_at: string;
  updated_at: string;
}
