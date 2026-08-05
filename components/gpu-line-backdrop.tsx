"use client";

import { motion, useReducedMotion } from "framer-motion";
import { useEffect, useMemo, useRef, useState } from "react";

type Point = { index: number; value: number };

function placeholderValue(index: number): number {
  // Node and Chromium can differ in the last floating-point digits of trig
  // functions. Rounding keeps the server and client SVG path byte-identical.
  return Number((35 + Math.sin(index / 2.8) * 8 + Math.sin(index / 7.2) * 5).toFixed(4));
}

function smoothPath(points: Point[]): string {
  const width = 1000;
  const height = 300;
  const padding = 24;
  const values = points.map((point) => point.value);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const center = (minimum + maximum) / 2;
  const range = Math.max(20, maximum - minimum);
  const lower = center - range / 2;
  const usableHeight = height - padding * 2;
  const coordinates = points.map((point, index) => ({
    x: (index / Math.max(1, points.length - 1)) * width,
    y: padding + (1 - (point.value - lower) / range) * usableHeight,
  }));

  return coordinates.slice(1).reduce((path, coordinate, index) => {
    const previous = coordinates[index];
    const midpoint = (previous.x + coordinate.x) / 2;
    return `${path} C ${midpoint} ${previous.y}, ${midpoint} ${coordinate.y}, ${coordinate.x} ${coordinate.y}`;
  }, `M ${coordinates[0].x} ${coordinates[0].y}`);
}

export function GpuLineBackdrop({ utilization }: { utilization: number | null }) {
  const reduceMotion = useReducedMotion();
  const cursor = useRef(32);
  const latestUtilization = useRef(utilization);
  const [points, setPoints] = useState<Point[]>(() =>
    Array.from({ length: 32 }, (_, index) => ({ index, value: placeholderValue(index) })),
  );

  useEffect(() => {
    latestUtilization.current = utilization;
  }, [utilization]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      cursor.current += 1;
      setPoints((current) => {
        const target = latestUtilization.current ?? placeholderValue(cursor.current);
        const previous = current.at(-1)?.value ?? target;
        const value = previous + (target - previous) * 0.52;
        return [...current.slice(-31), { index: cursor.current, value }];
      });
    }, reduceMotion ? 2200 : 1150);
    return () => window.clearInterval(timer);
  }, [reduceMotion]);

  const path = useMemo(() => smoothPath(points), [points]);
  const transition = reduceMotion
    ? { duration: 0 }
    : { duration: 1.15, ease: "linear" as const };

  return (
    <div className="gpu-line-backdrop" aria-hidden="true">
      <svg viewBox="0 0 1000 300" preserveAspectRatio="none">
        <defs>
          <linearGradient id="ambientGpuStroke" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#888888" stopOpacity={0.68} />
            <stop offset="50%" stopColor="#ffffff" stopOpacity={1} />
            <stop offset="100%" stopColor="#a8a8a8" stopOpacity={0.72} />
          </linearGradient>
        </defs>
        <motion.path
          initial={false}
          animate={{ d: path }}
          transition={transition}
          fill="none"
          stroke="#ffffff"
          strokeWidth={26}
          strokeOpacity={0.1}
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
        <motion.path
          initial={false}
          animate={{ d: path }}
          transition={transition}
          fill="none"
          stroke="url(#ambientGpuStroke)"
          strokeWidth={9}
          strokeOpacity={0.32}
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
        <motion.path
          initial={false}
          animate={{ d: path }}
          transition={transition}
          fill="none"
          stroke="url(#ambientGpuStroke)"
          strokeWidth={3.8}
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
        <motion.path
          className="gpu-trace-flow"
          initial={false}
          animate={{ d: path }}
          transition={transition}
          fill="none"
          stroke="rgba(220, 220, 220, 0.42)"
          strokeWidth={1.25}
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
    </div>
  );
}
