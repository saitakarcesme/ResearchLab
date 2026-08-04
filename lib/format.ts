export function formatMetric(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

export function formatMemory(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  if (value >= 1024) return `${(value / 1024).toFixed(1)} GB`;
  return `${Math.round(value)} MB`;
}

export function formatTokenCount(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const count = Math.max(0, value);
  if (count >= 1_000_000_000) {
    return `${(count / 1_000_000_000).toLocaleString(undefined, { maximumFractionDigits: 2 })}B`;
  }
  if (count >= 1_000_000) {
    return `${(count / 1_000_000).toLocaleString(undefined, { maximumFractionDigits: 2 })}M`;
  }
  if (count >= 1_000) {
    return `${(count / 1_000).toLocaleString(undefined, { maximumFractionDigits: 1 })}K`;
  }
  return Math.round(count).toLocaleString();
}

export function formatTime(value: string | null | undefined): string {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function titleCase(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function metricLabel(name: string): string {
  if (name === "val_bpb") return "Text prediction loss";
  if (name === "output_tokens_per_second") return "Generation speed";
  return titleCase(name);
}

export function metricDescription(name: string): string {
  if (name === "val_bpb") {
    return "How much the model struggles to predict text it has not seen. Lower is better.";
  }
  if (name === "output_tokens_per_second") {
    return "How many answer tokens the model produces each second. Higher is better.";
  }
  return "The score used to compare each completed test.";
}
