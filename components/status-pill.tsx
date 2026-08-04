import type { ResearchStatus } from "@/lib/types";
import {
  AlertTriangle,
  CheckCircle2,
  Clock3,
  LoaderCircle,
  Pause,
  Square,
} from "lucide-react";

const statusIcons = {
  queued: Clock3,
  running: LoaderCircle,
  paused: Pause,
  completed: CheckCircle2,
  stopped: Square,
  failed: AlertTriangle,
} satisfies Record<ResearchStatus, typeof Clock3>;

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
  const Icon = statusIcons[status];
  return (
    <span className={`status-pill status-${status}`} aria-label={status === "queued" && queuePosition ? `Queued, position ${queuePosition}` : status}>
      <Icon className={status === "running" ? "spin" : undefined} size={13} strokeWidth={1.8} aria-hidden="true" />
      {label}
    </span>
  );
}
