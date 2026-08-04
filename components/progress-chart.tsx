"use client";

import { Check, X } from "lucide-react";
import { memo, useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatMetric, metricDescription, metricLabel } from "@/lib/format";
import type { Experiment, MetricDirection } from "@/lib/types";

interface ChartPoint {
  experiment: number;
  metric: number;
  best: number;
  accepted: boolean;
  summary: string;
}

function buildPoints(
  experiments: Experiment[],
  baseline: number | null,
): ChartPoint[] {
  let runningBest: number | null = baseline;
  return experiments
    .filter((experiment) => experiment.metric_value != null)
    .sort((a, b) => a.experiment_number - b.experiment_number)
    .map((experiment) => {
      const metric = experiment.metric_value as number;
      if (experiment.accepted === true) {
        runningBest = metric;
      }
      return {
        experiment: experiment.experiment_number,
        metric,
        best: runningBest ?? experiment.previous_best ?? metric,
        accepted: experiment.accepted === true,
        summary: experiment.change_summary || experiment.hypothesis,
      };
    });
}

function ExperimentDot({ cx, cy, payload }: { cx?: number; cy?: number; payload?: ChartPoint }) {
  if (cx == null || cy == null || !payload) return null;
  return (
    <circle
      cx={cx}
      cy={cy}
      r={payload.accepted ? 5 : 4}
      fill={payload.accepted ? "#ef8bc3" : "#25212c"}
      stroke={payload.accepted ? "#ffd0e9" : "#756d7e"}
      strokeWidth={payload.accepted ? 2 : 1.5}
    />
  );
}

function ProgressChartComponent({
  experiments,
  metricName,
  direction,
  baseline,
}: {
  experiments: Experiment[];
  metricName: string;
  direction: MetricDirection;
  baseline: number | null;
}) {
  const points = useMemo(
    () => buildPoints(experiments, baseline),
    [baseline, experiments],
  );
  const accepted = useMemo(
    () => experiments
      .filter((experiment) => experiment.accepted)
      .sort((a, b) => b.experiment_number - a.experiment_number)
      .slice(0, 4),
    [experiments],
  );
  const displayMetric = metricLabel(metricName);

  return (
    <section className="surface-card progress-card" aria-labelledby="progress-title">
      <div className="card-heading">
        <div>
          <span className="eyebrow">Research progress</span>
          <h2 id="progress-title">{displayMetric}</h2>
          <p className="chart-metric-description">{metricDescription(metricName)}</p>
        </div>
        <div className="chart-key" aria-label="Chart legend">
          <span><i className="key-accepted" /> accepted</span>
          <span><i className="key-rejected" /> rejected</span>
          <span><i className="key-best" /> running best</span>
        </div>
      </div>

      {points.length ? (
        <>
          <div className="progress-chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={points} margin={{ top: 12, right: 14, left: -8, bottom: 0 }}>
                <defs>
                  <linearGradient id="progressFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#ec78b9" stopOpacity={0.22} />
                    <stop offset="100%" stopColor="#ec78b9" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid vertical={false} stroke="rgba(255,255,255,.045)" />
                <XAxis
                  dataKey="experiment"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: "#77707e", fontSize: 11 }}
                  tickMargin={10}
                />
                <YAxis
                  reversed={direction === "lower_is_better"}
                  domain={["auto", "auto"]}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: "#77707e", fontSize: 11 }}
                  tickFormatter={(value) => formatMetric(Number(value))}
                  width={58}
                />
                <Tooltip
                  cursor={{ stroke: "rgba(255,255,255,.12)", strokeDasharray: "3 5" }}
                  contentStyle={{
                    background: "rgba(21,18,27,.96)",
                    border: "1px solid rgba(255,255,255,.07)",
                    borderRadius: 14,
                    color: "#f5f1f6",
                    boxShadow: "0 16px 50px rgba(0,0,0,.35)",
                  }}
                  formatter={(value, name) => [formatMetric(Number(value)), name === "best" ? "Best so far" : displayMetric]}
                  labelFormatter={(value) => `Experiment ${value}`}
                />
                <Area type="monotone" dataKey="metric" fill="url(#progressFill)" stroke="none" />
                <Line
                  type="monotone"
                  dataKey="metric"
                  stroke="#ec78b9"
                  strokeWidth={2.2}
                  dot={<ExperimentDot />}
                  activeDot={{ r: 6, fill: "#f4a0ce", stroke: "#fff", strokeWidth: 1.5 }}
                  isAnimationActive
                  animationDuration={450}
                />
                <Line
                  type="stepAfter"
                  dataKey="best"
                  stroke="#9c83ed"
                  strokeWidth={1.8}
                  strokeDasharray="5 5"
                  dot={false}
                  isAnimationActive
                  animationDuration={450}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {accepted.length ? (
            <div className="accepted-steps">
              {accepted.map((experiment) => (
                <div className="accepted-step" key={experiment.id}>
                  <Check size={13} />
                  <span><strong>#{experiment.experiment_number}</strong> {experiment.change_summary || experiment.hypothesis}</span>
                </div>
              ))}
            </div>
          ) : null}
        </>
      ) : (
        <div className="chart-empty">
          <div className="empty-trace" aria-hidden="true" />
          <span>Waiting for the first measured experiment</span>
        </div>
      )}

      <div className="experiment-counts">
        <span><Check size={13} /> {experiments.filter((item) => item.accepted).length} accepted</span>
        <span><X size={13} /> {experiments.filter((item) => item.accepted === false).length} rejected</span>
      </div>
    </section>
  );
}

export const ProgressChart = memo(ProgressChartComponent);
