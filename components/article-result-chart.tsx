"use client";

import { useReducedMotion } from "framer-motion";
import { useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatMetric, metricDescription, metricLabel } from "@/lib/format";
import type { Article, Experiment, MetricDirection } from "@/lib/types";

interface ImprovementPoint {
  experiment: number;
  improvement: number;
  best: number;
}

function improvementPercent(
  baseline: number,
  value: number,
  direction: MetricDirection,
): number {
  if (baseline === 0) return 0;
  const delta = direction === "lower_is_better"
    ? baseline - value
    : value - baseline;
  return (delta / Math.abs(baseline)) * 100;
}

function buildImprovementPoints(article: Article): ImprovementPoint[] {
  const direction = article.metric_direction ?? "lower_is_better";
  const measured = [...(article.experiments ?? [])]
    .filter((experiment): experiment is Experiment & { metric_value: number } => experiment.metric_value != null)
    .sort((left, right) => left.experiment_number - right.experiment_number);
  const baseline = article.baseline_value ?? measured[0]?.metric_value;
  if (baseline == null) return [];

  let runningBest = baseline;
  const points: ImprovementPoint[] = [{ experiment: 0, improvement: 0, best: baseline }];
  for (const experiment of measured) {
    if (experiment.accepted === true) runningBest = experiment.metric_value;
    points.push({
      experiment: experiment.experiment_number,
      improvement: improvementPercent(baseline, runningBest, direction),
      best: runningBest,
    });
  }
  return points;
}

export function ArticleResultChart({ article }: { article: Article }) {
  const reduceMotion = useReducedMotion();
  const points = useMemo(() => buildImprovementPoints(article), [article]);
  if (points.length < 2 || article.baseline_value == null || article.best_value == null) return null;

  const improvement = improvementPercent(
    article.baseline_value,
    article.best_value,
    article.metric_direction ?? "lower_is_better",
  );
  const completedTests = article.experiments?.filter((experiment) => experiment.completed_at).length ?? 0;
  const directionVerb = article.metric_direction === "higher_is_better" ? "rose" : "fell";

  return (
    <figure className="article-result-chart" aria-labelledby="article-chart-title">
      <header>
        <div>
          <span>Measured improvement</span>
          <strong id="article-chart-title">{Math.max(0, improvement).toFixed(1)}% better</strong>
        </div>
        <dl>
          <div><dt>Started</dt><dd>{formatMetric(article.baseline_value)}</dd></div>
          <div><dt>Best</dt><dd>{formatMetric(article.best_value)}</dd></div>
        </dl>
      </header>
      <div className="article-chart-canvas" aria-hidden="true">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={points} margin={{ top: 12, right: 8, bottom: 0, left: -8 }}>
            <defs>
              <linearGradient id="articleImprovementFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#d77ab4" stopOpacity={0.28} />
                <stop offset="100%" stopColor="#d77ab4" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid vertical={false} stroke="rgba(255,255,255,.055)" />
            <XAxis
              dataKey="experiment"
              axisLine={false}
              tickLine={false}
              tick={{ fill: "#858087", fontSize: 10 }}
              tickFormatter={(value) => value === 0 ? "Start" : String(value)}
              minTickGap={20}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fill: "#858087", fontSize: 10 }}
              tickFormatter={(value) => `${Number(value).toFixed(0)}%`}
              width={44}
              domain={[0, "auto"]}
            />
            <Tooltip
              cursor={{ stroke: "rgba(255,255,255,.12)", strokeDasharray: "3 5" }}
              contentStyle={{
                background: "rgba(18,16,21,.96)",
                border: "1px solid rgba(255,255,255,.08)",
                borderRadius: 10,
                color: "#eee9e1",
                fontFamily: "var(--font-geist-sans), Arial, sans-serif",
                fontSize: 11,
              }}
              formatter={(value) => [`${Number(value).toFixed(1)}%`, "Improvement"]}
              labelFormatter={(value) => value === 0 ? "Starting point" : `After test ${value}`}
            />
            <Area
              type="monotone"
              dataKey="improvement"
              stroke="#e08abc"
              strokeWidth={2.2}
              fill="url(#articleImprovementFill)"
              dot={false}
              activeDot={{ r: 4, fill: "#f2a7cf", stroke: "#fff", strokeWidth: 1 }}
              isAnimationActive={!reduceMotion}
              animationDuration={450}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <figcaption>
        Best-so-far improvement across {completedTests} completed tests. {metricDescription(article.metric_name ?? "val_bpb")} {metricLabel(article.metric_name ?? "val_bpb")} {directionVerb} from {formatMetric(article.baseline_value)} to {formatMetric(article.best_value)}.
      </figcaption>
    </figure>
  );
}
