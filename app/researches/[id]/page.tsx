import type { Metadata } from "next";
import { ResearchLiveView } from "@/components/research-live-view";

export const metadata: Metadata = { title: "Research detail" };

export default async function ResearchDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return (
    <div className="detail-page">
      <ResearchLiveView researchId={id} />
    </div>
  );
}
