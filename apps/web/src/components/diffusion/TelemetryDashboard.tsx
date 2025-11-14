"use client";

/**
 * Diffusion Manufacturing Real-Time Telemetry Dashboard
 * WebSocket-based live monitoring of diffusion furnace parameters
 */

import React, { useState, useEffect, useCallback, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import {
  Activity,
  Thermometer,
  Wind,
  Gauge,
  Clock,
  AlertTriangle,
  CheckCircle2,
} from "lucide-react";
import { diffusionApi } from "@/lib/api/diffusion";

// Types
interface TelemetryData {
  timestamp: string;
  elapsed_time_s: number;
  temperatures: Record<string, number>; // zone1, zone2, zone3
  pressures?: Record<string, number>;
  ambient_flow_sccm: number;
  ambient_gas: string;
}

interface TelemetryHistory {
  timestamp: number;
  [key: string]: number | string;
}

interface TelemetryDashboardProps {
  runId: string;
  recipeName?: string;
  targetTemp?: number;
  onDisconnect?: () => void;
}

export default function TelemetryDashboard({
  runId,
  recipeName,
  targetTemp,
  onDisconnect,
}: TelemetryDashboardProps) {
  const [isConnected, setIsConnected] = useState(false);
  const [currentData, setCurrentData] = useState<TelemetryData | null>(null);
  const [temperatureHistory, setTemperatureHistory] = useState<TelemetryHistory[]>([]);
  const [alarms, setAlarms] = useState<string[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const maxHistoryPoints = 300; // 5 minutes at 1 Hz

  // WebSocket connection
  useEffect(() => {
    if (!runId) return;

    const ws = diffusionApi.connectTelemetryStream(runId);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("Diffusion WebSocket connected");
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data: TelemetryData = JSON.parse(event.data);
        handleTelemetryUpdate(data);
      } catch (error) {
        console.error("Error parsing telemetry data:", error);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      setAlarms((prev) => [...prev, "WebSocket connection error"]);
    };

    ws.onclose = () => {
      console.log("WebSocket disconnected");
      setIsConnected(false);
      onDisconnect?.();
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    };
  }, [runId]);

  const handleTelemetryUpdate = useCallback((data: TelemetryData) => {
    setCurrentData(data);

    const timestamp = new Date(data.timestamp).getTime();

    // Update temperature history
    setTemperatureHistory((prev) => {
      const newPoint: TelemetryHistory = {
        timestamp,
        elapsed_time_s: data.elapsed_time_s,
      };

      Object.entries(data.temperatures).forEach(([key, value]) => {
        newPoint[key] = value;
      });

      const updated = [...prev, newPoint];
      return updated.slice(-maxHistoryPoints);
    });

    // Check for alarms
    checkAlarms(data);
  }, []);

  const checkAlarms = (data: TelemetryData) => {
    const newAlarms: string[] = [];

    // Temperature alarms
    Object.entries(data.temperatures).forEach(([zone, temp]) => {
      if (temp > 1200) {
        newAlarms.push(`Critical temperature in ${zone}: ${temp.toFixed(1)}°C`);
      } else if (targetTemp && Math.abs(temp - targetTemp) > 20) {
        newAlarms.push(`Temperature deviation in ${zone}: ${Math.abs(temp - targetTemp).toFixed(1)}°C`);
      }
    });

    // Flow alarm
    if (data.ambient_flow_sccm < 10) {
      newAlarms.push(`Low ambient flow: ${data.ambient_flow_sccm.toFixed(1)} sccm`);
    }

    if (newAlarms.length > 0) {
      setAlarms((prev) => [...prev.slice(-4), ...newAlarms].slice(-5));
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getAvgTemperature = () => {
    if (!currentData) return 0;
    const temps = Object.values(currentData.temperatures);
    return temps.reduce((a, b) => a + b, 0) / temps.length;
  };

  const getTempUniformity = () => {
    if (!currentData) return 0;
    const temps = Object.values(currentData.temperatures);
    const avg = getAvgTemperature();
    const maxDev = Math.max(...temps.map(t => Math.abs(t - avg)));
    return (maxDev / avg) * 100;
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Real-Time Telemetry</h2>
          {recipeName && (
            <p className="text-sm text-muted-foreground">Recipe: {recipeName}</p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={isConnected ? "default" : "destructive"}>
            <Activity className="mr-1 h-3 w-3" />
            {isConnected ? "Live" : "Disconnected"}
          </Badge>
          <Badge variant="outline">Run ID: {runId.slice(0, 8)}</Badge>
        </div>
      </div>

      {/* Alarms */}
      {alarms.length > 0 && (
        <Card className="border-orange-500 bg-orange-50 dark:bg-orange-950">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <AlertTriangle className="h-4 w-4 text-orange-500" />
              Active Alarms
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-1 text-sm">
              {alarms.map((alarm, idx) => (
                <li key={idx} className="text-orange-700 dark:text-orange-300">
                  • {alarm}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Current Values Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Average Temperature */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Thermometer className="h-4 w-4 text-red-500" />
              Avg Temperature
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {currentData ? getAvgTemperature().toFixed(1) : "--"}°C
            </div>
            {targetTemp && (
              <p className="text-xs text-muted-foreground mt-1">
                Target: {targetTemp}°C
              </p>
            )}
          </CardContent>
        </Card>

        {/* Temperature Uniformity */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-green-500" />
              Uniformity
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {currentData ? getTempUniformity().toFixed(2) : "--"}%
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Temperature variation
            </p>
          </CardContent>
        </Card>

        {/* Ambient Gas Flow */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Wind className="h-4 w-4 text-blue-500" />
              {currentData?.ambient_gas || "Gas"} Flow
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {currentData ? currentData.ambient_flow_sccm.toFixed(0) : "--"}
            </div>
            <p className="text-xs text-muted-foreground mt-1">sccm</p>
          </CardContent>
        </Card>

        {/* Elapsed Time */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Clock className="h-4 w-4 text-purple-500" />
              Elapsed Time
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {currentData ? formatTime(currentData.elapsed_time_s) : "--:--"}
            </div>
            <p className="text-xs text-muted-foreground mt-1">mm:ss</p>
          </CardContent>
        </Card>
      </div>

      {/* Individual Zone Temperatures */}
      {currentData && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Zone Temperatures</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
              {Object.entries(currentData.temperatures).map(([zone, temp]) => (
                <div key={zone} className="text-center">
                  <div className="text-xs text-muted-foreground uppercase mb-1">
                    {zone}
                  </div>
                  <div className="text-xl font-bold">{temp.toFixed(1)}°C</div>
                  {targetTemp && (
                    <div className="text-xs text-muted-foreground mt-1">
                      Δ {(temp - targetTemp).toFixed(1)}°C
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Temperature History Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Temperature Profile</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={temperatureHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="elapsed_time_s"
                  tickFormatter={(value) => formatTime(value)}
                  label={{ value: "Time (mm:ss)", position: "insideBottom", offset: -5 }}
                />
                <YAxis
                  label={{ value: "Temperature (°C)", angle: -90, position: "insideLeft" }}
                />
                <Tooltip
                  labelFormatter={(value) => `Time: ${formatTime(value as number)}`}
                  formatter={(value: any) => [`${Number(value).toFixed(1)}°C`, ""]}
                />
                <Legend />
                {targetTemp && (
                  <ReferenceLine
                    y={targetTemp}
                    stroke="#22c55e"
                    strokeDasharray="5 5"
                    label="Target"
                  />
                )}
                {currentData &&
                  Object.keys(currentData.temperatures).map((zone, idx) => (
                    <Line
                      key={zone}
                      type="monotone"
                      dataKey={zone}
                      stroke={`hsl(${idx * 60}, 70%, 50%)`}
                      strokeWidth={2}
                      dot={false}
                      name={zone}
                    />
                  ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Process Status */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Process Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Ambient Gas:</span>
              <Badge variant="outline">{currentData?.ambient_gas || "N/A"}</Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Flow Rate:</span>
              <span className="font-medium">
                {currentData?.ambient_flow_sccm.toFixed(1) || "--"} sccm
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Data Points:</span>
              <span className="font-medium">{temperatureHistory.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Connection:</span>
              <Badge variant={isConnected ? "default" : "destructive"}>
                {isConnected ? "Connected" : "Disconnected"}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
