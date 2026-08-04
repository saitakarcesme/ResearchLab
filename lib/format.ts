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
  return titleCase(name);
}

export function metricDescription(name: string): string {
  if (name === "val_bpb") {
    return "How much the model struggles to predict text it has not seen. Lower is better.";
  }
  return "The score used to compare each completed test.";
}
