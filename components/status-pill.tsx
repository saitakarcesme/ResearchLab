import type { ResearchStatus } from "@/lib/types";

export function StatusPill({
  status,
  queuePosition,
}: {
  status: ResearchStatus;
  queuePosition?: number | null;
}) {
  const label = status === "queued" && queuePosition
    ? `queued #${queuePosition}`
    : status;
  return (
    <span className={`status-pill status-${status}`} aria-label={status === "queued" && queuePosition ? `Queued, position ${queuePosition}` : status}>
      <span aria-hidden="true" />
      {label}
    </span>
  );
}
