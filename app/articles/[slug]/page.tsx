import type { Metadata } from "next";
import { ArticleReader } from "@/components/article-reader";

export const metadata: Metadata = { title: "Research article" };

export default async function ArticleDetailPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  return <ArticleReader articleIdentifier={slug} />;
}
