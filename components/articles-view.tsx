"use client";

import { ArrowUpRight, BookOpen, LoaderCircle } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { listArticles } from "@/lib/api";
import { articlePath } from "@/lib/article-url";
import { formatMetric, metricLabel } from "@/lib/format";
import type { Article } from "@/lib/types";

export function ArticlesView() {
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    listArticles()
      .then((items) => mounted && setArticles(items))
      .catch((nextError) => mounted && setError(nextError instanceof Error ? nextError.message : "Articles are unavailable."))
      .finally(() => mounted && setLoading(false));
    return () => { mounted = false; };
  }, []);

  return (
    <div className="collection-page articles-page">
      <header className="compact-page-header">
        <h1>Articles</h1>
        <span>{articles.length}</span>
      </header>
      {error ? <p className="inline-error" role="alert">{error}</p> : null}
      {loading ? (
        <div className="quiet-row centered collection-loading"><LoaderCircle className="spin" size={16} /> Loading articles</div>
      ) : error ? null : articles.length ? (
        <div className="article-shelf">
          {articles.map((article, index) => (
            <Link className={`article-cover cover-${index % 4}`} href={articlePath(article)} key={article.id}>
              <div className="cover-sheen" aria-hidden="true" />
              <BookOpen size={18} strokeWidth={1.6} />
              <div className="cover-title">
                <span>{article.article_kind === "general" ? "Practical guide" : "Lab report"}</span>
                <h2>{article.title}</h2>
              </div>
              <div className="cover-footer">
                <span>
                  {article.article_kind === "general"
                    ? "Reader guide"
                    : `${article.metric_name ? metricLabel(article.metric_name) : "Research result"}${article.best_value != null ? ` · ${formatMetric(article.best_value)}` : ""}`}
                </span>
                <ArrowUpRight size={16} />
              </div>
            </Link>
          ))}
        </div>
      ) : (
        <div className="empty-collection article-empty">
          <span>No articles have been generated.</span>
          <Link href="/researches">Open researches</Link>
        </div>
      )}
    </div>
  );
}
