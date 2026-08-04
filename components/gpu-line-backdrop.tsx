"use client";

import { useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { Line, LineChart, ResponsiveContainer } from "recharts";

type Point = { index: number; value: number };

function placeholderValue(index: number): number {
  return 35 + Math.sin(index / 2.8) * 8 + Math.sin(index / 7.2) * 5;
}

export function GpuLineBackdrop({ utilization }: { utilization: number | null }) {
  const reduceMotion = useReducedMotion();
  const cursor = useRef(32);
  const [points, setPoints] = useState<Point[]>(() =>
    Array.from({ length: 32 }, (_, index) => ({ index, value: placeholderValue(index) })),
  );

  useEffect(() => {
    const timer = window.setInterval(() => {
      cursor.current += 1;
      const value = utilization == null ? placeholderValue(cursor.current) : utilization;
      setPoints((current) => [
        ...current.slice(-31),
        { index: cursor.current, value },
      ]);
    }, utilization == null ? 1800 : 1200);
    return () => window.clearInterval(timer);
  }, [utilization]);

  return (
    <div className="gpu-line-backdrop" aria-hidden="true">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points} margin={{ top: 20, right: 0, bottom: 20, left: 0 }}>
          <defs>
            <linearGradient id="ambientGpuStroke" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#a56cf4" stopOpacity={0.42} />
              <stop offset="52%" stopColor="#f47bc0" stopOpacity={1} />
              <stop offset="100%" stopColor="#8c63e8" stopOpacity={0.46} />
            </linearGradient>
          </defs>
          <Line
            type="monotone"
            dataKey="value"
            stroke="#dc69b0"
            strokeWidth={14}
            strokeOpacity={0.12}
            isAnimationActive={false}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke="url(#ambientGpuStroke)"
            strokeWidth={3.4}
            isAnimationActive={!reduceMotion}
            animationDuration={900}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
