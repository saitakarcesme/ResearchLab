import type { Metadata } from "next";
import { ArticlesView } from "@/components/articles-view";

export const metadata: Metadata = { title: "Articles" };

export default function ArticlesPage() {
  return <ArticlesView />;
}
