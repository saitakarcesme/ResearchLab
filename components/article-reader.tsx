"use client";

import { ArrowLeft, LoaderCircle } from "lucide-react";
import Link from "next/link";
import { Children, isValidElement, useEffect, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getArticle } from "@/lib/api";
import { formatMetric, metricLabel } from "@/lib/format";
import type { Article } from "@/lib/types";
import { ArticleResultChart } from "./article-result-chart";

function stripLeadingTitle(markdown: string, title: string): string {
  const match = markdown.match(/^\s*#\s+([^\r\n]+)\r?\n(?:\s*\r?\n)?/);
  if (!match || match[1].trim().toLocaleLowerCase() !== title.trim().toLocaleLowerCase()) {
    return markdown.trim();
  }
  return markdown.slice(match[0].length).trim();
}

function plainText(node: ReactNode): string {
  return Children.toArray(node)
    .map((child) => {
      if (typeof child === "string" || typeof child === "number") return String(child);
      return isValidElement<{ children?: ReactNode }>(child) ? plainText(child.props.children) : "";
    })
    .join("");
}

function headingId(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}

function markdownHeadings(markdown: string): Array<{ id: string; label: string }> {
  return markdown
    .split(/\r?\n/)
    .map((line) => line.match(/^##\s+(.+)$/)?.[1]?.replace(/[*_`]/g, "").trim())
    .filter((label): label is string => Boolean(label))
    .map((label) => ({ id: headingId(label), label }));
}

function removeInternalErrorSection(markdown: string): string {
  return markdown
    .replace(/\n## (?:Recorded errors|Errors recorded along the way)\s*\n[\s\S]*?(?=\n## |$)/gi, "")
    .trim();
}

function splitIntroduction(markdown: string): { introduction: string; body: string } {
  const sectionIndex = markdown.search(/^##\s+/m);
  if (sectionIndex < 0) return { introduction: markdown, body: "" };
  return {
    introduction: markdown.slice(0, sectionIndex).trim(),
    body: markdown.slice(sectionIndex).trim(),
  };
}

function ArticleMarkdown({ markdown }: { markdown: string }) {
  if (!markdown) return null;
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        h2: ({ children }) => {
          const id = headingId(plainText(children));
          return <h2 id={id}>{children}</h2>;
        },
      }}
    >
      {markdown}
    </ReactMarkdown>
  );
}

export function ArticleReader({ articleId }: { articleId: string }) {
  const [article, setArticle] = useState<Article | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retryKey, setRetryKey] = useState(0);

  useEffect(() => {
    let mounted = true;
    getArticle(articleId)
      .then((value) => mounted && setArticle(value))
      .catch((nextError) => mounted && setError(nextError instanceof Error ? nextError.message : "Article is unavailable."));
    return () => { mounted = false; };
  }, [articleId, retryKey]);

  if (!article) {
    if (error) {
      return (
        <div className="research-loader article-load-error" role="alert">
          <span>{error}</span>
          <button className="control-button" type="button" onClick={() => {
            setError(null);
            setRetryKey((value) => value + 1);
          }}>Retry</button>
        </div>
      );
    }
    return <div className="research-loader" role="status"><LoaderCircle className="spin" size={18} /> Loading article</div>;
  }

  const markdown = removeInternalErrorSection(stripLeadingTitle(article.markdown ?? "", article.title));
  const { introduction, body } = splitIntroduction(markdown);
  const headings = markdownHeadings(markdown);
  const wordCount = markdown.replace(/[#>*_`|\[\]-]/g, " ").trim().split(/\s+/).filter(Boolean).length;
  const readingMinutes = Math.max(1, Math.ceil(wordCount / 210));
  const publishedAt = new Intl.DateTimeFormat("en", {
    month: "long",
    day: "numeric",
    year: "numeric",
  }).format(new Date(article.updated_at || article.created_at));
  const metricSummary = article.metric_name
    ? `${metricLabel(article.metric_name)}${article.best_value != null ? ` ${formatMetric(article.best_value)}` : ""}`
    : null;

  return (
    <div className="article-page">
      <Link className="back-link" href="/articles"><ArrowLeft size={15} /> Articles</Link>
      <div className="article-layout">
        {headings.length ? (
          <nav className="article-toc" aria-label="Article contents">
            <span>Contents</span>
            <ol>
              {headings.map((heading, index) => (
                <li key={`${heading.id}-${index}`}>
                  <a href={`#${heading.id}`}>{heading.label}</a>
                </li>
              ))}
            </ol>
          </nav>
        ) : <div aria-hidden="true" />}

        <article className="article-reader">
          <header className="article-header">
            <div className="article-meta">
              <span>Research journal</span>
              <span aria-hidden="true">·</span>
              <time dateTime={article.updated_at || article.created_at}>{publishedAt}</time>
              <span aria-hidden="true">·</span>
              <span>{readingMinutes} min read</span>
              {metricSummary ? <><span aria-hidden="true">·</span><code>{metricSummary}</code></> : null}
            </div>
            <h1>{article.title}</h1>
          </header>
          <div className="markdown-body">
            <ArticleMarkdown markdown={introduction} />
            <ArticleResultChart article={article} />
            <ArticleMarkdown markdown={body} />
          </div>
        </article>
      </div>
    </div>
  );
}
