import type { ResearchStatus } from "@/lib/types";

export function StatusPill({ status }: { status: ResearchStatus }) {
  return (
    <span className={`status-pill status-${status}`}>
      <span aria-hidden="true" />
      {status}
    </span>
  );
}
