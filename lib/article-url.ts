import type { Article } from "./types";

export function articlePath(article: Pick<Article, "id" | "slug">): string {
  return `/articles/${encodeURIComponent(article.slug || article.id)}`;
}
