"use client";

/**
 * CVD Real-Time Telemetry Dashboard
 * WebSocket-based live monitoring of CVD process parameters
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
  Gauge,
  Thermometer,
  Wind,
  Zap,
  Droplets,
  RotateCw,
  Power,
  AlertTriangle,
} from "lucide-react";
import { cvdApi } from "@/lib/api/cvd";

// Types
interface TelemetryData {
  timestamp: string;
  temperatures: Record<string, number>;
  pressures: Record<string, number>;
  gas_flows: Record<string, number>;
  plasma_parameters?: Record<string, number>;
  rotation_speed_rpm?: number;
  valve_positions?: Record<string, number>;
  heater_powers?: Record<string, number>;
}

interface TelemetryHistory {
  timestamp: number;
  [key: string]: number;
}

interface TelemetryDashboardProps {
  runId: string;
  recipeName?: string;
  onDisconnect?: () => void;
}

export default function TelemetryDashboard({
  runId,
  recipeName,
  onDisconnect,
}: TelemetryDashboardProps) {
  const [isConnected, setIsConnected] = useState(false);
  const [currentData, setCurrentData] = useState<TelemetryData | null>(null);
  const [temperatureHistory, setTemperatureHistory] = useState<TelemetryHistory[]>([]);
  const [pressureHistory, setPressureHistory] = useState<TelemetryHistory[]>([]);
  const [flowHistory, setFlowHistory] = useState<TelemetryHistory[]>([]);
  const [alarms, setAlarms] = useState<string[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const maxHistoryPoints = 300; // 5 minutes at 1 Hz

  // WebSocket connection
  useEffect(() => {
    if (!runId) return;

    const ws = cvdApi.connectTelemetryStream(runId);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("WebSocket connected");
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
      // Only close if the WebSocket is still open
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    };
  }, [runId]); // Removed onDisconnect from dependencies to prevent unnecessary reconnections

  const handleTelemetryUpdate = useCallback((data: TelemetryData) => {
    setCurrentData(data);

    const timestamp = new Date(data.timestamp).getTime();

    // Update temperature history
    setTemperatureHistory((prev) => {
      const newPoint: TelemetryHistory = { timestamp };
      Object.entries(data.temperatures).forEach(([key, value]) => {
        newPoint[key] = value;
      });

      const updated = [...prev, newPoint];
      return updated.slice(-maxHistoryPoints);
    });

    // Update pressure history
    setPressureHistory((prev) => {
      const newPoint: TelemetryHistory = { timestamp };
      Object.entries(data.pressures).forEach(([key, value]) => {
        newPoint[key] = value;
      });

      const updated = [...prev, newPoint];
      return updated.slice(-maxHistoryPoints);
    });

    // Update flow history
    setFlowHistory((prev) => {
      const newPoint: TelemetryHistory = { timestamp };
      Object.entries(data.gas_flows).forEach(([key, value]) => {
        newPoint[key] = value;
      });

      const updated = [...prev, newPoint];
      return updated.slice(-maxHistoryPoints);
    });

    // Check for alarms (simplified)
    checkAlarms(data);
  }, []);

  const checkAlarms = (data: TelemetryData) => {
    const newAlarms: string[] = [];

    // Temperature alarms
    Object.entries(data.temperatures).forEach(([zone, temp]) => {
      if (temp > 1000) {
        newAlarms.push(`High temperature in ${zone}: ${temp.toFixed(1)}°C`);
      }
    });

    // Pressure alarms
    Object.entries(data.pressures).forEach(([location, pressure]) => {
      if (location === "chamber" && pressure > 500) {
        newAlarms.push(`High chamber pressure: ${pressure.toFixed(1)} Pa`);
      }
    });

    if (newAlarms.length > 0) {
      setAlarms((prev) => [...prev.slice(-4), ...newAlarms].slice(-5));
    }
  };

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
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

      <Separator />

      {/* Alarms */}
      {alarms.length > 0 && (
        <Card className="border-red-500">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center text-red-600">
              <AlertTriangle className="mr-2 h-4 w-4" />
              Active Alarms
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1">
              {alarms.map((alarm, idx) => (
                <div key={idx} className="text-sm text-red-600">
                  • {alarm}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Current Values Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {/* Temperature Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Temperature</CardTitle>
            <Thermometer className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {currentData?.temperatures ? (
              <div className="space-y-1">
                {Object.entries(currentData.temperatures).map(([zone, temp]) => (
                  <div key={zone} className="flex justify-between text-sm">
                    <span className="text-muted-foreground">{zone}:</span>
                    <span className="font-bold">{temp.toFixed(1)}°C</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-muted-foreground">No data</div>
            )}
          </CardContent>
        </Card>

        {/* Pressure Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pressure</CardTitle>
            <Gauge className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {currentData?.pressures ? (
              <div className="space-y-1">
                {Object.entries(currentData.pressures).map(([location, pressure]) => (
                  <div key={location} className="flex justify-between text-sm">
                    <span className="text-muted-foreground">{location}:</span>
                    <span className="font-bold">{pressure.toFixed(1)} Pa</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-muted-foreground">No data</div>
            )}
          </CardContent>
        </Card>

        {/* Gas Flows Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Gas Flows</CardTitle>
            <Wind className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {currentData?.gas_flows ? (
              <div className="space-y-1">
                {Object.entries(currentData.gas_flows)
                  .slice(0, 3)
                  .map(([gas, flow]) => (
                    <div key={gas} className="flex justify-between text-sm">
                      <span className="text-muted-foreground">{gas}:</span>
                      <span className="font-bold">{flow.toFixed(1)} sccm</span>
                    </div>
                  ))}
              </div>
            ) : (
              <div className="text-muted-foreground">No data</div>
            )}
          </CardContent>
        </Card>

        {/* Plasma/Other Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Additional</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="space-y-1">
              {currentData?.plasma_parameters?.rf_power_w && (
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">RF Power:</span>
                  <span className="font-bold">
                    {currentData.plasma_parameters.rf_power_w.toFixed(0)} W
                  </span>
                </div>
              )}
              {currentData?.rotation_speed_rpm !== undefined && (
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Rotation:</span>
                  <span className="font-bold">
                    {currentData.rotation_speed_rpm.toFixed(0)} RPM
                  </span>
                </div>
              )}
              {!currentData?.plasma_parameters && !currentData?.rotation_speed_rpm && (
                <div className="text-muted-foreground">No data</div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid gap-4 lg:grid-cols-2">
        {/* Temperature Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Thermometer className="mr-2 h-4 w-4" />
              Temperature Trend
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={temperatureHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={formatTime}
                  domain={["dataMin", "dataMax"]}
                />
                <YAxis label={{ value: "°C", angle: -90, position: "insideLeft" }} />
                <Tooltip labelFormatter={formatTime} />
                <Legend />
                {currentData?.temperatures &&
                  Object.keys(currentData.temperatures).map((zone, idx) => (
                    <Line
                      key={zone}
                      type="monotone"
                      dataKey={zone}
                      stroke={`hsl(${idx * 60}, 70%, 50%)`}
                      dot={false}
                      strokeWidth={2}
                    />
                  ))}
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Pressure Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Gauge className="mr-2 h-4 w-4" />
              Pressure Trend
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={pressureHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={formatTime}
                  domain={["dataMin", "dataMax"]}
                />
                <YAxis label={{ value: "Pa", angle: -90, position: "insideLeft" }} />
                <Tooltip labelFormatter={formatTime} />
                <Legend />
                {currentData?.pressures &&
                  Object.keys(currentData.pressures).map((location, idx) => (
                    <Line
                      key={location}
                      type="monotone"
                      dataKey={location}
                      stroke={`hsl(${idx * 120}, 70%, 50%)`}
                      dot={false}
                      strokeWidth={2}
                    />
                  ))}
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Gas Flow Chart */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Wind className="mr-2 h-4 w-4" />
              Gas Flow Trends
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={flowHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={formatTime}
                  domain={["dataMin", "dataMax"]}
                />
                <YAxis label={{ value: "sccm", angle: -90, position: "insideLeft" }} />
                <Tooltip labelFormatter={formatTime} />
                <Legend />
                {currentData?.gas_flows &&
                  Object.keys(currentData.gas_flows).map((gas, idx) => (
                    <Line
                      key={gas}
                      type="monotone"
                      dataKey={gas}
                      stroke={`hsl(${idx * 45}, 70%, 50%)`}
                      dot={false}
                      strokeWidth={2}
                    />
                  ))}
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
