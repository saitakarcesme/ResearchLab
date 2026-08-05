import type { Metadata } from "next";
import { GpusView } from "@/components/gpus-view";

export const metadata: Metadata = { title: "GPUs" };

export default function GpusPage() {
  return <GpusView />;
}
