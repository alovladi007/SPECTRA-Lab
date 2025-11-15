"use client";

/**
 * Wafer Map Component
 *
 * Displays:
 * - 2D wafer visualization (circular)
 * - Thickness/stress/parameter overlay as heatmap
 * - Measurement point indicators
 * - Color scale legend
 * - Interactive tooltips
 */

import React, { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

export interface WaferPoint {
  x: number; // mm from center (-radius to +radius)
  y: number; // mm from center (-radius to +radius)
  value: number; // measured value (thickness, stress, etc.)
  label?: string;
}

interface WaferMapProps {
  points: WaferPoint[];
  waferDiameter?: number; // mm (default 300)
  parameter?: string; // "Thickness", "Stress", etc.
  unit?: string; // "nm", "MPa", etc.
  colorScale?: "viridis" | "rdylgn" | "thermal" | "blues";
  showLegend?: boolean;
  showGrid?: boolean;
  valueRange?: [number, number]; // [min, max] for color scale
  highlightOutliers?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}

export function WaferMap({
  points,
  waferDiameter = 300,
  parameter = "Thickness",
  unit = "nm",
  colorScale = "viridis",
  showLegend = true,
  showGrid = true,
  valueRange,
  highlightOutliers = true,
  size = "md",
  className = "",
}: WaferMapProps) {
  const radius = waferDiameter / 2;

  // Calculate value statistics
  const stats = useMemo(() => {
    if (points.length === 0) {
      return { min: 0, max: 0, mean: 0, stdDev: 0, range: 0 };
    }

    const values = points.map(p => p.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);
    const range = max - min;

    return { min, max, mean, stdDev, range };
  }, [points]);

  // Use provided range or calculated range
  const [minValue, maxValue] = valueRange || [stats.min, stats.max];
  const valueSpan = maxValue - minValue;

  // Color scale functions
  const getColor = (value: number): string => {
    if (valueSpan === 0) return getColorScaleColor(colorScale, 0.5);

    const normalized = (value - minValue) / valueSpan;

    return getColorScaleColor(colorScale, normalized);
  };

  const getColorScaleColor = (scale: string, t: number): string => {
    // Clamp t to [0, 1]
    t = Math.max(0, Math.min(1, t));

    switch (scale) {
      case "viridis":
        // Simplified viridis-like scale
        if (t < 0.25) return interpolateColor("#440154", "#31688e", t * 4);
        if (t < 0.5) return interpolateColor("#31688e", "#35b779", (t - 0.25) * 4);
        if (t < 0.75) return interpolateColor("#35b779", "#fde724", (t - 0.5) * 4);
        return interpolateColor("#fde724", "#fde724", (t - 0.75) * 4);

      case "rdylgn":
        // Red-Yellow-Green scale
        if (t < 0.5) return interpolateColor("#d73027", "#fee08b", t * 2);
        return interpolateColor("#fee08b", "#1a9850", (t - 0.5) * 2);

      case "thermal":
        // Blue-Red thermal scale
        if (t < 0.33) return interpolateColor("#0000ff", "#00ffff", t * 3);
        if (t < 0.66) return interpolateColor("#00ffff", "#ffff00", (t - 0.33) * 3);
        return interpolateColor("#ffff00", "#ff0000", (t - 0.66) * 3);

      case "blues":
        // Blue monochrome
        return interpolateColor("#f7fbff", "#08519c", t);

      default:
        return "#999999";
    }
  };

  const interpolateColor = (color1: string, color2: string, t: number): string => {
    const hex2rgb = (hex: string) => {
      const r = parseInt(hex.slice(1, 3), 16);
      const g = parseInt(hex.slice(3, 5), 16);
      const b = parseInt(hex.slice(5, 7), 16);
      return [r, g, b];
    };

    const [r1, g1, b1] = hex2rgb(color1);
    const [r2, g2, b2] = hex2rgb(color2);

    const r = Math.round(r1 + (r2 - r1) * t);
    const g = Math.round(g1 + (g2 - g1) * t);
    const b = Math.round(b1 + (b2 - b1) * t);

    return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
  };

  // Detect outliers (> 2 std dev from mean)
  const isOutlier = (value: number): boolean => {
    if (!highlightOutliers) return false;
    return Math.abs(value - stats.mean) > 2 * stats.stdDev;
  };

  // Size variants
  const sizeConfig = {
    sm: { svg: 200, pointRadius: 4, fontSize: 10 },
    md: { svg: 300, pointRadius: 6, fontSize: 12 },
    lg: { svg: 400, pointRadius: 8, fontSize: 14 },
  };

  const config = sizeConfig[size];
  const scale = config.svg / (waferDiameter * 1.1); // 10% margin

  // Convert wafer coordinates to SVG coordinates
  const toSVG = (x: number, y: number) => {
    return {
      x: config.svg / 2 + x * scale,
      y: config.svg / 2 - y * scale, // Invert Y axis
    };
  };

  return (
    <Card className={className}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">
            {parameter} Map ({waferDiameter}mm wafer)
          </CardTitle>
          <Badge variant="outline">
            {points.length} points
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* SVG Wafer Map */}
          <div className="flex justify-center">
            <svg
              width={config.svg}
              height={config.svg}
              viewBox={`0 0 ${config.svg} ${config.svg}`}
              className="border rounded-lg bg-gray-50"
            >
              {/* Wafer outline */}
              <circle
                cx={config.svg / 2}
                cy={config.svg / 2}
                r={radius * scale}
                fill="white"
                stroke="#d1d5db"
                strokeWidth="2"
              />

              {/* Wafer flat (orientation marker) */}
              <line
                x1={config.svg / 2 - radius * scale * 0.3}
                y1={config.svg / 2 + radius * scale}
                x2={config.svg / 2 + radius * scale * 0.3}
                y2={config.svg / 2 + radius * scale}
                stroke="#d1d5db"
                strokeWidth="3"
              />

              {/* Grid lines */}
              {showGrid && (
                <>
                  {/* Horizontal center line */}
                  <line
                    x1={config.svg / 2 - radius * scale}
                    y1={config.svg / 2}
                    x2={config.svg / 2 + radius * scale}
                    y2={config.svg / 2}
                    stroke="#e5e7eb"
                    strokeWidth="1"
                    strokeDasharray="4 2"
                  />
                  {/* Vertical center line */}
                  <line
                    x1={config.svg / 2}
                    y1={config.svg / 2 - radius * scale}
                    x2={config.svg / 2}
                    y2={config.svg / 2 + radius * scale}
                    stroke="#e5e7eb"
                    strokeWidth="1"
                    strokeDasharray="4 2"
                  />
                  {/* Radial circles */}
                  {[0.33, 0.67].map((fraction, i) => (
                    <circle
                      key={i}
                      cx={config.svg / 2}
                      cy={config.svg / 2}
                      r={radius * scale * fraction}
                      fill="none"
                      stroke="#e5e7eb"
                      strokeWidth="1"
                      strokeDasharray="4 2"
                    />
                  ))}
                </>
              )}

              {/* Data points */}
              <TooltipProvider>
                {points.map((point, idx) => {
                  const pos = toSVG(point.x, point.y);
                  const color = getColor(point.value);
                  const outlier = isOutlier(point.value);

                  return (
                    <Tooltip key={idx}>
                      <TooltipTrigger asChild>
                        <g>
                          {/* Point circle */}
                          <circle
                            cx={pos.x}
                            cy={pos.y}
                            r={config.pointRadius}
                            fill={color}
                            stroke={outlier ? "#ef4444" : "#374151"}
                            strokeWidth={outlier ? 2 : 1}
                            className="cursor-help transition-transform hover:scale-125"
                          />
                          {/* Label */}
                          {point.label && (
                            <text
                              x={pos.x}
                              y={pos.y - config.pointRadius - 4}
                              textAnchor="middle"
                              fontSize={config.fontSize - 2}
                              fill="#6b7280"
                              className="pointer-events-none"
                            >
                              {point.label}
                            </text>
                          )}
                        </g>
                      </TooltipTrigger>
                      <TooltipContent>
                        <div className="space-y-1">
                          {point.label && (
                            <div className="font-semibold">{point.label}</div>
                          )}
                          <div className="text-sm">
                            {parameter}: <span className="font-semibold">{point.value.toFixed(2)} {unit}</span>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Position: ({point.x.toFixed(0)}, {point.y.toFixed(0)}) mm
                          </div>
                          {outlier && (
                            <div className="text-xs text-red-600 font-medium">
                              Outlier (&gt;2σ from mean)
                            </div>
                          )}
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  );
                })}
              </TooltipProvider>

              {/* Center marker */}
              <circle
                cx={config.svg / 2}
                cy={config.svg / 2}
                r="2"
                fill="#9ca3af"
              />
            </svg>
          </div>

          {/* Statistics */}
          <div className="grid grid-cols-4 gap-2 text-xs">
            <div>
              <div className="text-muted-foreground">Min</div>
              <div className="font-semibold">{stats.min.toFixed(1)} {unit}</div>
            </div>
            <div>
              <div className="text-muted-foreground">Max</div>
              <div className="font-semibold">{stats.max.toFixed(1)} {unit}</div>
            </div>
            <div>
              <div className="text-muted-foreground">Mean</div>
              <div className="font-semibold">{stats.mean.toFixed(1)} {unit}</div>
            </div>
            <div>
              <div className="text-muted-foreground">Uniformity</div>
              <div className="font-semibold">
                ±{stats.mean > 0 ? ((stats.stdDev / stats.mean) * 100).toFixed(2) : 0}%
              </div>
            </div>
          </div>

          {/* Color legend */}
          {showLegend && (
            <div className="space-y-2">
              <div className="text-xs text-muted-foreground font-medium">Color Scale</div>
              <div className="relative h-6">
                {/* Gradient bar */}
                <div className="absolute inset-0 rounded overflow-hidden flex">
                  {Array.from({ length: 20 }).map((_, i) => {
                    const t = i / 19;
                    const color = getColorScaleColor(colorScale, t);
                    return (
                      <div
                        key={i}
                        style={{ backgroundColor: color, width: `${100 / 20}%` }}
                      />
                    );
                  })}
                </div>
                {/* Labels */}
                <div className="absolute inset-0 flex justify-between items-center px-1 text-[10px] font-semibold text-white drop-shadow">
                  <span>{minValue.toFixed(0)}</span>
                  <span>{maxValue.toFixed(0)} {unit}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Generate standard measurement point patterns
 */
export const generateWaferPoints = {
  // 5-point pattern (center + 4 cardinal points)
  fivePoint: (radius: number = 150): Omit<WaferPoint, "value">[] => {
    const r = radius * 0.7; // 70% of radius
    return [
      { x: 0, y: 0, label: "C" },
      { x: 0, y: r, label: "N" },
      { x: r, y: 0, label: "E" },
      { x: 0, y: -r, label: "S" },
      { x: -r, y: 0, label: "W" },
    ];
  },

  // 9-point pattern
  ninePoint: (radius: number = 150): Omit<WaferPoint, "value">[] => {
    const r = radius * 0.7;
    return [
      { x: 0, y: 0, label: "C" },
      { x: 0, y: r, label: "N" },
      { x: r * 0.707, y: r * 0.707, label: "NE" },
      { x: r, y: 0, label: "E" },
      { x: r * 0.707, y: -r * 0.707, label: "SE" },
      { x: 0, y: -r, label: "S" },
      { x: -r * 0.707, y: -r * 0.707, label: "SW" },
      { x: -r, y: 0, label: "W" },
      { x: -r * 0.707, y: r * 0.707, label: "NW" },
    ];
  },

  // 49-point pattern (7x7 grid)
  fortyNinePoint: (radius: number = 150): Omit<WaferPoint, "value">[] => {
    const points: Omit<WaferPoint, "value">[] = [];
    const step = radius * 2 / 7;

    for (let i = 0; i < 7; i++) {
      for (let j = 0; j < 7; j++) {
        const x = -radius + i * step + step / 2;
        const y = -radius + j * step + step / 2;

        // Only include points within wafer radius
        if (Math.sqrt(x * x + y * y) <= radius * 0.95) {
          points.push({ x, y });
        }
      }
    }

    return points;
  },
};

export default WaferMap;
