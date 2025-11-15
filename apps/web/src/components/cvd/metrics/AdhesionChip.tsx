"use client";

/**
 * Adhesion Chip Component
 *
 * Displays:
 * - Color-coded adhesion class badge
 * - Tooltip with test method and score details
 * - Risk indicators
 */

import React from "react";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { AlertCircle, CheckCircle, Info } from "lucide-react";

export type AdhesionClass = "excellent" | "good" | "fair" | "poor" | "unknown";

export interface AdhesionTestMethod {
  name: string;
  description: string;
  standard?: string;
}

interface AdhesionChipProps {
  score: number; // Adhesion score (0-100)
  adhesionClass?: AdhesionClass; // Override class determination
  testMethod?: AdhesionTestMethod;
  showScore?: boolean;
  showIcon?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}

// Common test methods
export const TEST_METHODS = {
  TAPE_TEST: {
    name: "Tape Test",
    description: "Cross-hatch adhesion test using pressure-sensitive tape",
    standard: "ASTM D3359",
  },
  SCRATCH_TEST: {
    name: "Scratch Test",
    description: "Progressive load scratch test to determine critical load",
    standard: "ASTM C1624",
  },
  PULL_OFF: {
    name: "Pull-Off Test",
    description: "Perpendicular tensile force to measure adhesion strength",
    standard: "ASTM D4541",
  },
  FOUR_POINT_BEND: {
    name: "Four-Point Bend",
    description: "Interfacial fracture toughness measurement",
    standard: "ASTM E290",
  },
  AUTOMATED_OPTICAL: {
    name: "Automated Optical",
    description: "Optical inspection for delamination and defects",
    standard: "In-house",
  },
} as const;

export function AdhesionChip({
  score,
  adhesionClass,
  testMethod = TEST_METHODS.TAPE_TEST,
  showScore = true,
  showIcon = true,
  size = "md",
  className = "",
}: AdhesionChipProps) {
  // Determine adhesion class from score if not provided
  const getAdhesionClass = (score: number): AdhesionClass => {
    if (score >= 80) return "excellent";
    if (score >= 60) return "good";
    if (score >= 40) return "fair";
    if (score >= 0) return "poor";
    return "unknown";
  };

  const finalClass = adhesionClass || getAdhesionClass(score);

  // Get display properties based on class
  const getClassProperties = (adhesionClass: AdhesionClass) => {
    switch (adhesionClass) {
      case "excellent":
        return {
          label: "Excellent",
          color: "bg-green-100 text-green-800 border-green-300",
          icon: CheckCircle,
          iconColor: "text-green-600",
          description: "Strong adhesion, no risk of delamination",
        };
      case "good":
        return {
          label: "Good",
          color: "bg-blue-100 text-blue-800 border-blue-300",
          icon: CheckCircle,
          iconColor: "text-blue-600",
          description: "Acceptable adhesion, low risk",
        };
      case "fair":
        return {
          label: "Fair",
          color: "bg-yellow-100 text-yellow-800 border-yellow-400",
          icon: Info,
          iconColor: "text-yellow-600",
          description: "Marginal adhesion, monitor for delamination",
        };
      case "poor":
        return {
          label: "Poor",
          color: "bg-red-100 text-red-800 border-red-400",
          icon: AlertCircle,
          iconColor: "text-red-600",
          description: "Weak adhesion, high risk of failure",
        };
      default:
        return {
          label: "Unknown",
          color: "bg-gray-100 text-gray-800 border-gray-300",
          icon: Info,
          iconColor: "text-gray-600",
          description: "Adhesion not tested",
        };
    }
  };

  const props = getClassProperties(finalClass);
  const Icon = props.icon;

  // Size variants
  const sizeClasses = {
    sm: {
      badge: "text-xs px-2 py-0.5",
      icon: "h-3 w-3",
      score: "text-xs",
    },
    md: {
      badge: "text-sm px-3 py-1",
      icon: "h-4 w-4",
      score: "text-sm",
    },
    lg: {
      badge: "text-base px-4 py-1.5",
      icon: "h-5 w-5",
      score: "text-base",
    },
  };

  const sizeClass = sizeClasses[size];

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            variant="outline"
            className={`${props.color} ${sizeClass.badge} font-medium border cursor-help inline-flex items-center gap-1.5 ${className}`}
          >
            {showIcon && <Icon className={`${sizeClass.icon} ${props.iconColor}`} />}
            <span>{props.label}</span>
            {showScore && (
              <span className={`${sizeClass.score} font-semibold`}>
                ({score.toFixed(0)})
              </span>
            )}
          </Badge>
        </TooltipTrigger>
        <TooltipContent className="max-w-xs">
          <div className="space-y-2">
            {/* Adhesion score */}
            <div>
              <div className="font-semibold text-sm">Adhesion Score: {score.toFixed(1)}/100</div>
              <div className="text-xs text-muted-foreground mt-0.5">
                {props.description}
              </div>
            </div>

            {/* Score ranges */}
            <div className="text-xs border-t pt-2">
              <div className="font-medium mb-1">Classification:</div>
              <div className="space-y-0.5 text-muted-foreground">
                <div>80-100: Excellent</div>
                <div>60-80: Good</div>
                <div>40-60: Fair</div>
                <div>0-40: Poor</div>
              </div>
            </div>

            {/* Test method */}
            {testMethod && (
              <div className="text-xs border-t pt-2">
                <div className="font-medium">{testMethod.name}</div>
                <div className="text-muted-foreground mt-0.5">
                  {testMethod.description}
                </div>
                {testMethod.standard && (
                  <div className="text-muted-foreground mt-1 text-[10px] font-mono">
                    Standard: {testMethod.standard}
                  </div>
                )}
              </div>
            )}

            {/* Risk indicator for poor adhesion */}
            {finalClass === "poor" && (
              <div className="text-xs bg-red-50 text-red-700 p-2 rounded border-t border-red-200">
                <div className="flex items-start gap-1">
                  <AlertCircle className="h-3 w-3 mt-0.5 flex-shrink-0" />
                  <div>
                    <span className="font-semibold">High Risk:</span> Film may delaminate during processing or use. Consider surface treatment or process optimization.
                  </div>
                </div>
              </div>
            )}
            {finalClass === "fair" && (
              <div className="text-xs bg-yellow-50 text-yellow-700 p-2 rounded border-t border-yellow-200">
                <div className="flex items-start gap-1">
                  <Info className="h-3 w-3 mt-0.5 flex-shrink-0" />
                  <div>
                    <span className="font-semibold">Monitor:</span> Marginal adhesion may lead to issues under stress or thermal cycling.
                  </div>
                </div>
              </div>
            )}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

/**
 * Compact variant without score display
 */
export function AdhesionBadge({
  score,
  adhesionClass,
  className = "",
}: {
  score: number;
  adhesionClass?: AdhesionClass;
  className?: string;
}) {
  return (
    <AdhesionChip
      score={score}
      adhesionClass={adhesionClass}
      showScore={false}
      showIcon={true}
      size="sm"
      className={className}
    />
  );
}

/**
 * Large variant for detail views
 */
export function AdhesionDetail({
  score,
  adhesionClass,
  testMethod,
  className = "",
}: {
  score: number;
  adhesionClass?: AdhesionClass;
  testMethod?: AdhesionTestMethod;
  className?: string;
}) {
  return (
    <AdhesionChip
      score={score}
      adhesionClass={adhesionClass}
      testMethod={testMethod}
      showScore={true}
      showIcon={true}
      size="lg"
      className={className}
    />
  );
}

export default AdhesionChip;
