import type { Metadata } from "next";
import { ResearchLiveView } from "@/components/research-live-view";

export const metadata: Metadata = { title: "Research detail" };

export default async function ResearchDetailPage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ monitor?: string }>;
}) {
  const { id } = await params;
  const { monitor } = await searchParams;
  return (
    <div className="detail-page">
      <ResearchLiveView researchId={id} monitorMode={monitor === "1"} />
    </div>
  );
}
