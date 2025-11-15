"use client";

/**
 * Thickness Gauge Component
 *
 * Displays:
 * - Target vs actual thickness
 * - Uniformity ring (color-coded)
 * - Deviation indicator
 */

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { TrendingUp, TrendingDown, Minus, AlertCircle } from "lucide-react";

interface ThicknessGaugeProps {
  actual: number; // Actual thickness (nm)
  target: number; // Target thickness (nm)
  uniformity: number; // WIW uniformity (%)
  tolerance?: number; // Tolerance (%) - default 5%
  showUniformityRing?: boolean;
  className?: string;
}

export function ThicknessGauge({
  actual,
  target,
  uniformity,
  tolerance = 5.0,
  showUniformityRing = true,
  className = "",
}: ThicknessGaugeProps) {
  // Calculate deviation
  const deviation = ((actual - target) / target) * 100;
  const absDeviation = Math.abs(deviation);

  // Determine status
  const isInSpec = absDeviation <= tolerance;
  const statusColor = isInSpec ? "text-green-600" : "text-red-600";
  const statusBg = isInSpec ? "bg-green-50" : "bg-red-50";

  // Determine uniformity status
  const getUniformityColor = (uniformity: number) => {
    if (uniformity < 2) return "text-green-600";
    if (uniformity < 5) return "text-yellow-600";
    return "text-red-600";
  };

  const getUniformityRingColor = (uniformity: number) => {
    if (uniformity < 2) return "stroke-green-500";
    if (uniformity < 5) return "stroke-yellow-500";
    return "stroke-red-500";
  };

  // Calculate progress (0-100%)
  const progress = Math.min(100, (actual / target) * 100);

  return (
    <Card className={className}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">Film Thickness</CardTitle>
          <Badge variant={isInSpec ? "default" : "destructive"} className={statusBg}>
            {isInSpec ? "In Spec" : "Out of Spec"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Gauge Visualization */}
          <div className="relative flex items-center justify-center">
            {/* SVG Gauge */}
            <svg width="180" height="180" viewBox="0 0 180 180" className="transform -rotate-90">
              {/* Background circle */}
              <circle
                cx="90"
                cy="90"
                r="70"
                fill="none"
                stroke="#e5e7eb"
                strokeWidth="12"
              />

              {/* Progress arc */}
              <circle
                cx="90"
                cy="90"
                r="70"
                fill="none"
                stroke={isInSpec ? "#22c55e" : "#ef4444"}
                strokeWidth="12"
                strokeDasharray={`${(progress / 100) * 440} 440`}
                strokeLinecap="round"
                className="transition-all duration-500"
              />

              {/* Uniformity ring (outer) */}
              {showUniformityRing && (
                <circle
                  cx="90"
                  cy="90"
                  r="82"
                  fill="none"
                  className={getUniformityRingColor(uniformity)}
                  strokeWidth="4"
                  strokeDasharray="4 2"
                />
              )}
            </svg>

            {/* Center content */}
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <div className="text-3xl font-bold">{actual.toFixed(1)}</div>
              <div className="text-xs text-muted-foreground">nm</div>
            </div>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 gap-3 text-sm">
            {/* Target */}
            <div>
              <div className="text-muted-foreground">Target</div>
              <div className="font-medium">{target.toFixed(1)} nm</div>
            </div>

            {/* Deviation */}
            <div>
              <div className="text-muted-foreground">Deviation</div>
              <div className={`font-medium flex items-center ${statusColor}`}>
                {deviation > 0 ? (
                  <TrendingUp className="h-3 w-3 mr-1" />
                ) : deviation < 0 ? (
                  <TrendingDown className="h-3 w-3 mr-1" />
                ) : (
                  <Minus className="h-3 w-3 mr-1" />
                )}
                {absDeviation.toFixed(1)}%
              </div>
            </div>

            {/* Uniformity */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="col-span-2 cursor-help">
                    <div className="text-muted-foreground flex items-center">
                      WIW Uniformity
                      <AlertCircle className="h-3 w-3 ml-1" />
                    </div>
                    <div className={`font-medium ${getUniformityColor(uniformity)}`}>
                      ±{uniformity.toFixed(2)}%
                    </div>
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="text-xs">
                    Within-Wafer Uniformity
                    <br />
                    &lt; 2%: Excellent
                    <br />
                    2-5%: Good
                    <br />
                    &gt; 5%: Poor
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>

            {/* Tolerance */}
            <div className="col-span-2 text-xs text-muted-foreground pt-2 border-t">
              Tolerance: ±{tolerance}% ({(target * (1 - tolerance / 100)).toFixed(1)} - {(target * (1 + tolerance / 100)).toFixed(1)} nm)
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default ThicknessGauge;
