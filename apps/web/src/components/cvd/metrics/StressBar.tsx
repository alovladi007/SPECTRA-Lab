"use client";

/**
 * Stress Bar Component
 *
 * Displays:
 * - Horizontal stress axis (compressive ← → tensile)
 * - "Safe zone" highlighted region
 * - Current stress value indicator
 * - Color-coded based on risk level
 */

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { AlertCircle, ArrowLeft, ArrowRight } from "lucide-react";

interface StressBarProps {
  stress: number; // Current stress (MPa)
  safeZoneMin?: number; // Safe zone minimum (MPa) - default -400
  safeZoneMax?: number; // Safe zone maximum (MPa) - default 300
  rangeMin?: number; // Display range minimum - default -800
  rangeMax?: number; // Display range maximum - default 600
  showLabels?: boolean;
  className?: string;
}

export function StressBar({
  stress,
  safeZoneMin = -400,
  safeZoneMax = 300,
  rangeMin = -800,
  rangeMax = 600,
  showLabels = true,
  className = "",
}: StressBarProps) {
  // Determine stress type and risk level
  const isCompressive = stress < 0;
  const isTensile = stress > 0;
  const isInSafeZone = stress >= safeZoneMin && stress <= safeZoneMax;

  // Calculate position on bar (0-100%)
  const calculatePosition = (value: number) => {
    const normalized = ((value - rangeMin) / (rangeMax - rangeMin)) * 100;
    return Math.max(0, Math.min(100, normalized));
  };

  const stressPosition = calculatePosition(stress);
  const safeZoneStartPercent = calculatePosition(safeZoneMin);
  const safeZoneWidthPercent = calculatePosition(safeZoneMax) - safeZoneStartPercent;

  // Determine status and color
  const getStressStatus = () => {
    if (stress < -500) return { label: "Critical Compressive", color: "destructive", severity: "CRITICAL" };
    if (stress < safeZoneMin) return { label: "High Compressive", color: "destructive", severity: "WARNING" };
    if (stress > 500) return { label: "Critical Tensile", color: "destructive", severity: "CRITICAL" };
    if (stress > safeZoneMax) return { label: "High Tensile", color: "destructive", severity: "WARNING" };
    return { label: "Normal", color: "default", severity: "OK" };
  };

  const status = getStressStatus();

  const getStressColor = () => {
    if (!isInSafeZone) {
      if (Math.abs(stress) > 500) return "bg-red-600"; // Critical
      return "bg-orange-500"; // Warning
    }
    return isCompressive ? "bg-blue-500" : "bg-green-500"; // Safe zone
  };

  const getIndicatorColor = () => {
    if (!isInSafeZone) {
      if (Math.abs(stress) > 500) return "border-red-600 bg-red-100";
      return "border-orange-500 bg-orange-100";
    }
    return isCompressive ? "border-blue-500 bg-blue-100" : "border-green-500 bg-green-100";
  };

  return (
    <Card className={className}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">Film Stress</CardTitle>
          <Badge
            variant={status.color as "default" | "destructive"}
            className={isInSafeZone ? "bg-green-50 text-green-700" : ""}
          >
            {status.label}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Current stress value */}
          <div className="text-center">
            <div className="text-3xl font-bold">
              {stress.toFixed(0)}
              <span className="text-lg text-muted-foreground ml-1">MPa</span>
            </div>
            <div className="text-xs text-muted-foreground flex items-center justify-center gap-1 mt-1">
              {isCompressive && (
                <>
                  <ArrowLeft className="h-3 w-3" />
                  <span>Compressive</span>
                </>
              )}
              {isTensile && (
                <>
                  <span>Tensile</span>
                  <ArrowRight className="h-3 w-3" />
                </>
              )}
              {stress === 0 && <span>Neutral</span>}
            </div>
          </div>

          {/* Stress bar visualization */}
          <div className="relative">
            {/* Bar container */}
            <div className="relative h-12 bg-gray-100 rounded-lg overflow-hidden">
              {/* Safe zone highlight */}
              <div
                className="absolute top-0 bottom-0 bg-green-100 border-l-2 border-r-2 border-green-300"
                style={{
                  left: `${safeZoneStartPercent}%`,
                  width: `${safeZoneWidthPercent}%`,
                }}
              />

              {/* Center line (zero stress) */}
              <div
                className="absolute top-0 bottom-0 w-0.5 bg-gray-400"
                style={{ left: `${calculatePosition(0)}%` }}
              />

              {/* Stress indicator */}
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div
                      className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 cursor-help"
                      style={{ left: `${stressPosition}%` }}
                    >
                      {/* Indicator marker */}
                      <div className="relative">
                        <div
                          className={`w-3 h-10 rounded-sm border-2 ${getIndicatorColor()}`}
                        />
                        {/* Triangle pointer */}
                        <div
                          className={`absolute -bottom-1 left-1/2 -translate-x-1/2 w-0 h-0
                            border-l-4 border-l-transparent
                            border-r-4 border-r-transparent
                            border-t-4 ${getStressColor().replace('bg-', 'border-t-')}`}
                        />
                      </div>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="text-xs font-semibold">{stress.toFixed(1)} MPa</p>
                    <p className="text-xs">
                      {isCompressive ? "Compressive stress" : isTensile ? "Tensile stress" : "Neutral"}
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>

            {/* Scale labels */}
            {showLabels && (
              <div className="flex justify-between text-xs text-muted-foreground mt-2">
                <div className="text-left">
                  <div className="font-medium">{rangeMin} MPa</div>
                  <div className="text-[10px]">Compressive</div>
                </div>
                <div className="text-center">
                  <div className="font-medium">0</div>
                  <div className="text-[10px]">Neutral</div>
                </div>
                <div className="text-right">
                  <div className="font-medium">+{rangeMax} MPa</div>
                  <div className="text-[10px]">Tensile</div>
                </div>
              </div>
            )}
          </div>

          {/* Safe zone info */}
          <div className="flex items-center justify-between text-xs">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex items-center gap-1 text-muted-foreground cursor-help">
                    <div className="w-3 h-3 bg-green-100 border border-green-300 rounded" />
                    <span>Safe Zone</span>
                    <AlertCircle className="h-3 w-3" />
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="text-xs">
                    Safe stress range:
                    <br />
                    {safeZoneMin} to {safeZoneMax} MPa
                    <br />
                    <br />
                    &lt; {safeZoneMin} MPa: High compressive
                    <br />
                    &gt; {safeZoneMax} MPa: High tensile
                    <br />
                    |σ| &gt; 500 MPa: Critical (adhesion risk)
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>

            <div className="text-muted-foreground">
              {isInSafeZone ? (
                <span className="text-green-600 font-medium">Within safe zone</span>
              ) : (
                <span className="text-orange-600 font-medium">
                  {Math.abs(stress - (stress < 0 ? safeZoneMin : safeZoneMax)).toFixed(0)} MPa from safe zone
                </span>
              )}
            </div>
          </div>

          {/* Risk warnings */}
          {!isInSafeZone && (
            <div className={`text-xs p-2 rounded ${
              Math.abs(stress) > 500 ? "bg-red-50 text-red-700" : "bg-orange-50 text-orange-700"
            }`}>
              <div className="flex items-start gap-2">
                <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                <div>
                  {Math.abs(stress) > 500 ? (
                    <>
                      <div className="font-semibold">Critical Stress Level</div>
                      <div className="mt-1">
                        High risk of film delamination or cracking. Consider adjusting temperature or deposition rate.
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="font-semibold">Elevated Stress</div>
                      <div className="mt-1">
                        {isCompressive
                          ? "Compressive stress may cause buckling or delamination."
                          : "Tensile stress may cause cracking or poor adhesion."}
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default StressBar;
