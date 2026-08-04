import type { Metadata } from "next";
import { ResearchesView } from "@/components/researches-view";

export const metadata: Metadata = { title: "Researches" };

export default function ResearchesPage() {
  return <ResearchesView />;
}
