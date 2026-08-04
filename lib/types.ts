export type ResearchStatus =
  | "running"
  | "paused"
  | "completed"
  | "stopped"
  | "failed";

export type MetricDirection = "lower_is_better" | "higher_is_better";

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
}

export interface ResearchLog {
  id: string | number;
  research_id: string;
  event_type: string;
  message: string;
  created_at: string;
}

export interface Research {
  id: string;
  title: string;
  original_prompt: string;
  objective?: string;
  status: ResearchStatus;
  metric_name: string;
  metric_direction: MetricDirection;
  baseline_value: number | null;
  best_value: number | null;
  gpu_source_id: string | null;
  gpu_source_name?: string | null;
  target_gpu_allocation: number;
  workspace_path: string | null;
  created_at: string;
  updated_at: string;
  experiment_count?: number;
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

export interface Article {
  id: string;
  research_id: string;
  title: string;
  markdown?: string;
  metric_name?: string;
  metric_direction?: MetricDirection;
  objective?: string;
  baseline_value?: number | null;
  best_value?: number | null;
  experiments?: Experiment[];
  created_at: string;
  updated_at: string;
}
