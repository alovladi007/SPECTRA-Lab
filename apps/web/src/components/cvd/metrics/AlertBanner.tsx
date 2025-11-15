"use client";

/**
 * Alert Banner Component
 *
 * Displays:
 * - Out-of-spec warnings and alerts
 * - Multiple severity levels (info, warning, error, critical)
 * - Dismissible alerts
 * - Action buttons
 * - Alert grouping and filtering
 */

import React, { useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  AlertCircle,
  AlertTriangle,
  Info,
  XCircle,
  X,
  ChevronDown,
  ChevronUp,
  ExternalLink,
} from "lucide-react";

export type AlertSeverity = "info" | "warning" | "error" | "critical";

export interface AlertItem {
  id: string;
  severity: AlertSeverity;
  title: string;
  message: string;
  timestamp?: string;
  source?: string; // "thickness", "stress", "adhesion", "process", etc.
  details?: Record<string, any>;
  actionLabel?: string;
  actionUrl?: string;
  onAction?: () => void;
  dismissible?: boolean;
}

interface AlertBannerProps {
  alerts: AlertItem[];
  onDismiss?: (alertId: string) => void;
  maxVisible?: number;
  groupBySeverity?: boolean;
  showTimestamp?: boolean;
  className?: string;
}

export function AlertBanner({
  alerts,
  onDismiss,
  maxVisible = 5,
  groupBySeverity = false,
  showTimestamp = true,
  className = "",
}: AlertBannerProps) {
  const [dismissedAlerts, setDismissedAlerts] = useState<Set<string>>(new Set());
  const [expanded, setExpanded] = useState(false);

  // Filter out dismissed alerts
  const visibleAlerts = alerts.filter(alert => !dismissedAlerts.has(alert.id));

  // Handle dismiss
  const handleDismiss = (alertId: string) => {
    setDismissedAlerts(prev => new Set(prev).add(alertId));
    onDismiss?.(alertId);
  };

  // Group alerts by severity if requested
  const groupedAlerts = groupBySeverity
    ? {
        critical: visibleAlerts.filter(a => a.severity === "critical"),
        error: visibleAlerts.filter(a => a.severity === "error"),
        warning: visibleAlerts.filter(a => a.severity === "warning"),
        info: visibleAlerts.filter(a => a.severity === "info"),
      }
    : { all: visibleAlerts };

  // Limit visible alerts
  const shouldShowMore = visibleAlerts.length > maxVisible;
  const displayAlerts = expanded ? visibleAlerts : visibleAlerts.slice(0, maxVisible);

  if (visibleAlerts.length === 0) {
    return null;
  }

  return (
    <div className={`space-y-3 ${className}`}>
      {/* Alert count header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <AlertCircle className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">
            {visibleAlerts.length} {visibleAlerts.length === 1 ? "Alert" : "Alerts"}
          </span>
        </div>
        {shouldShowMore && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setExpanded(!expanded)}
            className="text-xs"
          >
            {expanded ? (
              <>
                Show Less <ChevronUp className="ml-1 h-3 w-3" />
              </>
            ) : (
              <>
                Show All ({visibleAlerts.length}) <ChevronDown className="ml-1 h-3 w-3" />
              </>
            )}
          </Button>
        )}
      </div>

      {/* Alert list */}
      <div className="space-y-2">
        {displayAlerts.map(alert => (
          <AlertBannerItem
            key={alert.id}
            alert={alert}
            onDismiss={handleDismiss}
            showTimestamp={showTimestamp}
          />
        ))}
      </div>
    </div>
  );
}

/**
 * Individual Alert Item
 */
function AlertBannerItem({
  alert,
  onDismiss,
  showTimestamp,
}: {
  alert: AlertItem;
  onDismiss: (id: string) => void;
  showTimestamp: boolean;
}) {
  const severityConfig = getSeverityConfig(alert.severity);
  const Icon = severityConfig.icon;

  return (
    <Alert className={`${severityConfig.className} relative`}>
      <Icon className={`h-4 w-4 ${severityConfig.iconColor}`} />

      <AlertTitle className="flex items-center justify-between pr-8">
        <div className="flex items-center gap-2">
          <span>{alert.title}</span>
          <Badge variant="outline" className="text-xs">
            {alert.source || alert.severity}
          </Badge>
        </div>
      </AlertTitle>

      <AlertDescription>
        <div className="space-y-2">
          {/* Message */}
          <div className="text-sm">{alert.message}</div>

          {/* Details */}
          {alert.details && Object.keys(alert.details).length > 0 && (
            <div className="text-xs bg-white/50 dark:bg-black/20 p-2 rounded border">
              <div className="font-medium mb-1">Details:</div>
              <div className="space-y-0.5">
                {Object.entries(alert.details).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-muted-foreground">{key}:</span>
                    <span className="font-mono">{String(value)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Footer */}
          <div className="flex items-center justify-between text-xs">
            {/* Timestamp */}
            {showTimestamp && alert.timestamp && (
              <span className="text-muted-foreground">
                {new Date(alert.timestamp).toLocaleString()}
              </span>
            )}

            {/* Action button */}
            {(alert.actionLabel || alert.onAction) && (
              <Button
                variant="outline"
                size="sm"
                onClick={alert.onAction}
                className="h-7 text-xs"
              >
                {alert.actionLabel || "View Details"}
                {alert.actionUrl && <ExternalLink className="ml-1 h-3 w-3" />}
              </Button>
            )}
          </div>
        </div>
      </AlertDescription>

      {/* Dismiss button */}
      {alert.dismissible !== false && (
        <Button
          variant="ghost"
          size="icon"
          className="absolute top-2 right-2 h-6 w-6"
          onClick={() => onDismiss(alert.id)}
        >
          <X className="h-3 w-3" />
        </Button>
      )}
    </Alert>
  );
}

/**
 * Get severity configuration
 */
function getSeverityConfig(severity: AlertSeverity) {
  switch (severity) {
    case "critical":
      return {
        icon: XCircle,
        iconColor: "text-red-600",
        className: "border-red-500 bg-red-50 dark:bg-red-950",
      };
    case "error":
      return {
        icon: AlertCircle,
        iconColor: "text-red-600",
        className: "border-red-400 bg-red-50 dark:bg-red-950",
      };
    case "warning":
      return {
        icon: AlertTriangle,
        iconColor: "text-yellow-600",
        className: "border-yellow-400 bg-yellow-50 dark:bg-yellow-950",
      };
    case "info":
      return {
        icon: Info,
        iconColor: "text-blue-600",
        className: "border-blue-400 bg-blue-50 dark:bg-blue-950",
      };
  }
}

/**
 * Compact alert list for dashboards
 */
export function AlertList({
  alerts,
  maxVisible = 3,
  className = "",
}: {
  alerts: AlertItem[];
  maxVisible?: number;
  className?: string;
}) {
  const displayAlerts = alerts.slice(0, maxVisible);

  if (alerts.length === 0) {
    return (
      <div className={`text-sm text-muted-foreground ${className}`}>
        No active alerts
      </div>
    );
  }

  return (
    <div className={`space-y-2 ${className}`}>
      {displayAlerts.map(alert => {
        const severityConfig = getSeverityConfig(alert.severity);
        const Icon = severityConfig.icon;

        return (
          <div
            key={alert.id}
            className="flex items-start gap-2 text-xs p-2 rounded bg-muted/50"
          >
            <Icon className={`h-3 w-3 mt-0.5 ${severityConfig.iconColor}`} />
            <div className="flex-1 min-w-0">
              <div className="font-medium truncate">{alert.title}</div>
              <div className="text-muted-foreground truncate">{alert.message}</div>
            </div>
            {alert.timestamp && (
              <div className="text-muted-foreground text-[10px] whitespace-nowrap">
                {new Date(alert.timestamp).toLocaleTimeString()}
              </div>
            )}
          </div>
        );
      })}

      {alerts.length > maxVisible && (
        <div className="text-xs text-center text-muted-foreground">
          +{alerts.length - maxVisible} more
        </div>
      )}
    </div>
  );
}

/**
 * Single inline alert (for immediate feedback)
 */
export function InlineAlert({
  severity,
  message,
  className = "",
}: {
  severity: AlertSeverity;
  message: string;
  className?: string;
}) {
  const severityConfig = getSeverityConfig(severity);
  const Icon = severityConfig.icon;

  return (
    <div className={`flex items-center gap-2 text-sm p-3 rounded-lg ${severityConfig.className} ${className}`}>
      <Icon className={`h-4 w-4 ${severityConfig.iconColor} flex-shrink-0`} />
      <span>{message}</span>
    </div>
  );
}

export default AlertBanner;
