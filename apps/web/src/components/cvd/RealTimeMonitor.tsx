"use client";

/**
 * Real-Time CVD Run Monitor
 *
 * Displays live updates from WebSocket connection for a running CVD process.
 * Can be integrated into run detail pages or dashboards.
 */

import React, { useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { AlertList } from "./metrics/AlertBanner";
import { useCVDWebSocket, useCVDEventFilter, useLatestCVDEvent } from "@/hooks/useCVDWebSocket";
import { Wifi, WifiOff, Activity, TrendingUp } from "lucide-react";

interface RealTimeMonitorProps {
  runId: string;
  onProgressUpdate?: (progress: number) => void;
  onThicknessUpdate?: (thickness: number) => void;
  onStressUpdate?: (stress: number) => void;
  className?: string;
}

export function RealTimeMonitor({
  runId,
  onProgressUpdate,
  onThicknessUpdate,
  onStressUpdate,
  className = "",
}: RealTimeMonitorProps) {
  const { connectionState, events, isConnected } = useCVDWebSocket({
    runId,
    onEvent: (event) => {
      console.log("[RealTimeMonitor] New event:", event);

      // Trigger callbacks
      if (event.event_type === "progress_update" && event.data.progress) {
        onProgressUpdate?.(event.data.progress);
      }
      if (event.event_type === "thickness_update" && event.data.current_thickness_nm) {
        onThicknessUpdate?.(event.data.current_thickness_nm);
      }
      if (event.event_type === "metrics_update" && event.data.current_stress_mpa) {
        onStressUpdate?.(event.data.current_stress_mpa);
      }
    },
    autoReconnect: true,
  });

  // Get latest progress
  const latestProgress = useLatestCVDEvent(events, "progress_update");
  const latestMetrics = useLatestCVDEvent(events, "metrics_update");

  // Filter alerts
  const alerts = useCVDEventFilter(events, [
    "warning",
    "error",
    "stress_risk",
    "adhesion_risk",
    "rate_anomaly",
  ]).map((event, index) => ({
    id: `${event.event_type}-${index}`,
    severity: event.event_type === "error" ? "error" as const : "warning" as const,
    title: event.event_type.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase()),
    message: event.data.message || JSON.stringify(event.data),
    timestamp: event.timestamp,
    source: event.event_type,
  }));

  const getConnectionBadge = () => {
    const configs = {
      connected: { icon: Wifi, label: "Live", className: "bg-green-100 text-green-800" },
      connecting: { icon: Activity, label: "Connecting...", className: "bg-yellow-100 text-yellow-800" },
      disconnected: { icon: WifiOff, label: "Offline", className: "bg-gray-100 text-gray-800" },
      error: { icon: WifiOff, label: "Error", className: "bg-red-100 text-red-800" },
    };

    const config = configs[connectionState];
    const Icon = config.icon;

    return (
      <Badge variant="outline" className={config.className}>
        <Icon className="h-3 w-3 mr-1" />
        {config.label}
      </Badge>
    );
  };

  return (
    <Card className={className}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">Real-Time Updates</CardTitle>
          {getConnectionBadge()}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Connection Status */}
        {!isConnected && (
          <div className="text-xs text-muted-foreground p-2 bg-muted rounded">
            Waiting for connection to run {runId}...
          </div>
        )}

        {/* Progress */}
        {latestProgress && (
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">Deposition Progress</span>
              <span className="font-medium">{latestProgress.data.progress?.toFixed(1)}%</span>
            </div>
            <Progress value={latestProgress.data.progress || 0} className="h-2" />
          </div>
        )}

        {/* Metrics */}
        {latestMetrics && (
          <div className="grid grid-cols-2 gap-3 text-xs">
            {latestMetrics.data.current_thickness_nm && (
              <div>
                <div className="text-muted-foreground">Thickness</div>
                <div className="font-medium flex items-center gap-1">
                  <TrendingUp className="h-3 w-3 text-blue-600" />
                  {latestMetrics.data.current_thickness_nm.toFixed(1)} nm
                </div>
              </div>
            )}

            {latestMetrics.data.deposition_rate_nm_min && (
              <div>
                <div className="text-muted-foreground">Rate</div>
                <div className="font-medium">
                  {latestMetrics.data.deposition_rate_nm_min.toFixed(1)} nm/min
                </div>
              </div>
            )}

            {latestMetrics.data.current_stress_mpa && (
              <div>
                <div className="text-muted-foreground">Stress</div>
                <div className="font-medium">
                  {latestMetrics.data.current_stress_mpa.toFixed(0)} MPa
                </div>
              </div>
            )}

            {latestMetrics.data.uniformity && (
              <div>
                <div className="text-muted-foreground">Uniformity</div>
                <div className="font-medium">
                  ±{latestMetrics.data.uniformity.toFixed(2)}%
                </div>
              </div>
            )}
          </div>
        )}

        {/* Alerts */}
        {alerts.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs font-medium text-muted-foreground">Active Alerts</div>
            <AlertList alerts={alerts} maxVisible={3} />
          </div>
        )}

        {/* Event Count */}
        <div className="text-xs text-muted-foreground pt-2 border-t">
          {events.length} event{events.length !== 1 ? "s" : ""} received
          {latestProgress && (
            <span className="ml-2">
              • Last update: {new Date(latestProgress.timestamp).toLocaleTimeString()}
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Compact version for sidebars or dashboards
 */
export function RealTimeIndicator({ runId, className = "" }: { runId: string; className?: string }) {
  const { connectionState, isConnected } = useCVDWebSocket({
    runId,
    onEvent: (event) => {
      console.log(`[${runId}] Event:`, event.event_type);
    },
  });

  const Icon = isConnected ? Wifi : WifiOff;
  const color = isConnected ? "text-green-600" : "text-gray-400";

  return (
    <div className={`flex items-center gap-1 text-xs ${className}`}>
      <Icon className={`h-3 w-3 ${color}`} />
      <span className="text-muted-foreground">
        {connectionState === "connected" && "Live"}
        {connectionState === "connecting" && "Connecting..."}
        {connectionState === "disconnected" && "Offline"}
        {connectionState === "error" && "Error"}
      </span>
    </div>
  );
}

export default RealTimeMonitor;
