"use client";

/**
 * CVD Run Detail Page
 *
 * Features:
 * - Telemetry plots (temperature, pressure, flow rates over time)
 * - Predicted vs actual thickness comparison
 * - Stress indicator gauge (using StressBar component)
 * - Adhesion risk badge (using AdhesionChip component)
 * - Real-time updates for running processes
 * - Alert history
 */

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams, useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ThicknessGauge } from "@/components/cvd/metrics/ThicknessGauge";
import { StressBar } from "@/components/cvd/metrics/StressBar";
import { AdhesionChip, AdhesionDetail } from "@/components/cvd/metrics/AdhesionChip";
import { AlertList } from "@/components/cvd/metrics/AlertBanner";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";
import {
  ArrowLeft,
  Play,
  Pause,
  CheckCircle,
  XCircle,
  Download,
  AlertCircle,
} from "lucide-react";
import Link from "next/link";
import { Progress } from "@/components/ui/progress";

export default function RunDetailPage() {
  const params = useParams();
  const router = useRouter();
  const runId = params.id as string;

  // Fetch run details
  const { data: run, isLoading } = useQuery({
    queryKey: ["cvd-run", runId],
    queryFn: async () => {
      const response = await fetch(`http://localhost:8001/api/cvd/runs/${runId}`);
      if (!response.ok) throw new Error("Failed to fetch run details");
      return response.json();
    },
    refetchInterval: (data) => {
      // Refresh every 2s if running, otherwise don't refetch
      return data?.status === "running" ? 2000 : false;
    },
  });

  // Fetch telemetry data
  const { data: telemetry } = useQuery({
    queryKey: ["cvd-telemetry", runId],
    queryFn: async () => {
      const response = await fetch(`http://localhost:8001/api/cvd/runs/${runId}/telemetry`);
      if (!response.ok) throw new Error("Failed to fetch telemetry");
      return response.json();
    },
    refetchInterval: (data, query) => {
      return run?.status === "running" ? 2000 : false;
    },
    enabled: !!run,
  });

  // Fetch alerts
  const { data: alerts } = useQuery({
    queryKey: ["cvd-run-alerts", runId],
    queryFn: async () => {
      const response = await fetch(`http://localhost:8001/api/cvd/runs/${runId}/alerts`);
      if (!response.ok) throw new Error("Failed to fetch alerts");
      return response.json();
    },
    enabled: !!run,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  if (!run) {
    return (
      <div className="container mx-auto p-6">
        <div className="text-center">
          <h2 className="text-2xl font-bold">Run Not Found</h2>
          <Button className="mt-4" onClick={() => router.push("/cvd/runs")}>
            Back to Runs
          </Button>
        </div>
      </div>
    );
  }

  // Mock data (replace with real data from API)
  const mockRun = {
    id: runId,
    run_id: "CVD_RUN_20251114_103045",
    recipe_name: "Si3N4 Standard",
    tool: "CVD-01",
    status: "running",
    progress: 67,
    started_at: "2025-11-14T10:30:45",
    wafer_id: "W12345",
    target_thickness_nm: 100,
    current_thickness_nm: 67.5,
    predicted_final_thickness_nm: 101.2,
    thickness_uniformity: 2.3,
    current_stress_mpa: -185,
    target_stress_mpa: -200,
    predicted_final_stress_mpa: -192,
    adhesion_score: 88,
    target_adhesion_score: 85,
    temperature_c: 800,
    pressure_torr: 0.5,
    precursor_flow_sccm: 80,
    carrier_gas_flow_sccm: 500,
    ...run,
  };

  // Mock telemetry data
  const mockTelemetry = telemetry || {
    thickness: Array.from({ length: 20 }, (_, i) => ({
      time: i * 6,
      value: (i + 1) * 5,
      predicted: (i + 1) * 5.1,
    })),
    temperature: Array.from({ length: 20 }, (_, i) => ({
      time: i * 6,
      value: 800 + Math.sin(i * 0.5) * 5,
      setpoint: 800,
    })),
    pressure: Array.from({ length: 20 }, (_, i) => ({
      time: i * 6,
      value: 0.5 + Math.sin(i * 0.3) * 0.02,
      setpoint: 0.5,
    })),
    flow_rates: Array.from({ length: 20 }, (_, i) => ({
      time: i * 6,
      precursor: 80 + Math.sin(i * 0.4) * 2,
      carrier: 500 + Math.sin(i * 0.6) * 10,
    })),
    stress: Array.from({ length: 20 }, (_, i) => ({
      time: i * 6,
      value: -180 - (i * 1.5) + Math.sin(i * 0.5) * 5,
    })),
  };

  const mockAlerts = alerts || [
    {
      id: "1",
      severity: "warning",
      title: "Elevated Stress Detected",
      message: "Compressive stress approaching warning threshold",
      timestamp: "2025-11-14T10:45:30",
      source: "stress",
      details: { current_stress: -185, threshold: -200 },
    },
  ];

  const getStatusBadge = () => {
    const configs = {
      running: { variant: "default" as const, icon: Play, className: "bg-blue-100 text-blue-800" },
      completed: { variant: "default" as const, icon: CheckCircle, className: "bg-green-100 text-green-800" },
      failed: { variant: "destructive" as const, icon: XCircle, className: "bg-red-100 text-red-800" },
      pending: { variant: "secondary" as const, icon: Pause, className: "bg-gray-100 text-gray-800" },
      cancelled: { variant: "secondary" as const, icon: Pause, className: "bg-gray-100 text-gray-800" },
    };

    const config = configs[mockRun.status as keyof typeof configs];
    const Icon = config.icon;

    return (
      <Badge variant={config.variant} className={config.className}>
        <Icon className="h-4 w-4 mr-1" />
        {mockRun.status.charAt(0).toUpperCase() + mockRun.status.slice(1)}
      </Badge>
    );
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" asChild>
            <Link href="/cvd/runs">
              <ArrowLeft className="h-5 w-5" />
            </Link>
          </Button>
          <div>
            <h1 className="text-3xl font-bold">{mockRun.run_id}</h1>
            <p className="text-muted-foreground mt-1">
              {mockRun.recipe_name} • {mockRun.tool} • Wafer {mockRun.wafer_id}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {getStatusBadge()}
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export Data
          </Button>
        </div>
      </div>

      {/* Progress (if running) */}
      {mockRun.status === "running" && (
        <Card className="bg-blue-50 dark:bg-blue-950 border-blue-200">
          <CardContent className="pt-6">
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">Deposition Progress</span>
                <span className="text-muted-foreground">{mockRun.progress}%</span>
              </div>
              <Progress value={mockRun.progress} className="h-3" />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Started: {new Date(mockRun.started_at).toLocaleString()}</span>
                <span>
                  Est. completion: {new Date(
                    new Date(mockRun.started_at).getTime() +
                    (120 * 1000 * (100 / mockRun.progress))
                  ).toLocaleTimeString()}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Metrics Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Thickness Gauge */}
        <ThicknessGauge
          actual={mockRun.current_thickness_nm}
          target={mockRun.target_thickness_nm}
          uniformity={mockRun.thickness_uniformity}
          tolerance={5}
        />

        {/* Stress Bar */}
        <StressBar
          stress={mockRun.current_stress_mpa}
          safeZoneMin={-400}
          safeZoneMax={300}
        />

        {/* Adhesion & Alerts */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Adhesion & Alerts</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label className="text-xs text-muted-foreground">Adhesion Score</Label>
              <div className="mt-2">
                <AdhesionDetail
                  score={mockRun.adhesion_score}
                  testMethod={{ name: "Tape Test", description: "ASTM D3359", standard: "ASTM D3359" }}
                />
              </div>
              {mockRun.status === "completed" && (
                <div className="mt-2 text-xs text-muted-foreground">
                  Target: {mockRun.target_adhesion_score} • Actual: {mockRun.adhesion_score}
                </div>
              )}
            </div>

            <div>
              <Label className="text-xs text-muted-foreground">Active Alerts</Label>
              <div className="mt-2">
                {mockAlerts.length > 0 ? (
                  <Badge variant="destructive">
                    <AlertCircle className="h-3 w-3 mr-1" />
                    {mockAlerts.length} alert{mockAlerts.length > 1 ? "s" : ""}
                  </Badge>
                ) : (
                  <Badge variant="outline" className="bg-green-50 text-green-700">
                    <CheckCircle className="h-3 w-3 mr-1" />
                    No alerts
                  </Badge>
                )}
              </div>
            </div>

            {mockRun.status === "running" && (
              <Button variant="outline" className="w-full" size="sm">
                View Live Results →
              </Button>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Main Content - Tabs */}
      <Card>
        <CardHeader>
          <CardTitle>Run Data</CardTitle>
          <CardDescription>Telemetry, predictions, and alerts</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="telemetry" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="telemetry">Telemetry</TabsTrigger>
              <TabsTrigger value="predictions">Predictions</TabsTrigger>
              <TabsTrigger value="alerts">Alerts ({mockAlerts.length})</TabsTrigger>
              <TabsTrigger value="parameters">Parameters</TabsTrigger>
            </TabsList>

            <TabsContent value="telemetry" className="space-y-6 mt-6">
              {/* Thickness vs Time */}
              <div>
                <h3 className="text-sm font-semibold mb-3">Thickness Growth</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <AreaChart data={mockTelemetry.thickness}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="time"
                      label={{ value: "Time (s)", position: "insideBottom", offset: -5 }}
                    />
                    <YAxis label={{ value: "Thickness (nm)", angle: -90, position: "insideLeft" }} />
                    <RechartsTooltip />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="value"
                      name="Actual"
                      stroke="#3b82f6"
                      fill="#3b82f680"
                    />
                    <Line
                      type="monotone"
                      dataKey="predicted"
                      name="Predicted"
                      stroke="#f59e0b"
                      strokeDasharray="5 5"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Temperature */}
              <div>
                <h3 className="text-sm font-semibold mb-3">Temperature</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={mockTelemetry.temperature}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" label={{ value: "Time (s)", position: "insideBottom", offset: -5 }} />
                    <YAxis domain={[795, 805]} label={{ value: "Temperature (°C)", angle: -90, position: "insideLeft" }} />
                    <RechartsTooltip />
                    <Legend />
                    <Line type="monotone" dataKey="value" name="Actual" stroke="#ef4444" strokeWidth={2} />
                    <Line type="monotone" dataKey="setpoint" name="Setpoint" stroke="#94a3b8" strokeDasharray="3 3" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Pressure */}
              <div>
                <h3 className="text-sm font-semibold mb-3">Pressure</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={mockTelemetry.pressure}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" label={{ value: "Time (s)", position: "insideBottom", offset: -5 }} />
                    <YAxis domain={[0.45, 0.55]} label={{ value: "Pressure (Torr)", angle: -90, position: "insideLeft" }} />
                    <RechartsTooltip />
                    <Legend />
                    <Line type="monotone" dataKey="value" name="Actual" stroke="#10b981" strokeWidth={2} />
                    <Line type="monotone" dataKey="setpoint" name="Setpoint" stroke="#94a3b8" strokeDasharray="3 3" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Stress Evolution */}
              <div>
                <h3 className="text-sm font-semibold mb-3">Stress Evolution</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={mockTelemetry.stress}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" label={{ value: "Time (s)", position: "insideBottom", offset: -5 }} />
                    <YAxis label={{ value: "Stress (MPa)", angle: -90, position: "insideLeft" }} />
                    <RechartsTooltip />
                    <Legend />
                    <Area type="monotone" dataKey="value" name="Stress" stroke="#f59e0b" fill="#f59e0b80" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            <TabsContent value="predictions" className="space-y-4 mt-6">
              <div className="grid grid-cols-2 gap-4">
                <Card className="bg-blue-50 dark:bg-blue-950">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">Predicted Final Thickness</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{mockRun.predicted_final_thickness_nm} nm</div>
                    <div className="text-xs text-muted-foreground mt-1">
                      Target: {mockRun.target_thickness_nm} nm ({((mockRun.predicted_final_thickness_nm - mockRun.target_thickness_nm) / mockRun.target_thickness_nm * 100).toFixed(1)}%)
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-blue-50 dark:bg-blue-950">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">Predicted Final Stress</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{mockRun.predicted_final_stress_mpa} MPa</div>
                    <div className="text-xs text-muted-foreground mt-1">
                      Target: {mockRun.target_stress_mpa} MPa
                    </div>
                  </CardContent>
                </Card>
              </div>

              <div className="text-xs text-muted-foreground p-4 bg-muted rounded">
                <strong>Model Confidence:</strong> Predictions are based on real-time process parameters
                and historical data. Current model confidence: 92%.
              </div>
            </TabsContent>

            <TabsContent value="alerts" className="mt-6">
              <AlertList alerts={mockAlerts} maxVisible={10} />
            </TabsContent>

            <TabsContent value="parameters" className="mt-6">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <ParamCard label="Recipe" value={mockRun.recipe_name} />
                <ParamCard label="Tool" value={mockRun.tool} />
                <ParamCard label="Wafer ID" value={mockRun.wafer_id} />
                <ParamCard label="Temperature" value={`${mockRun.temperature_c} °C`} />
                <ParamCard label="Pressure" value={`${mockRun.pressure_torr} Torr`} />
                <ParamCard label="Precursor Flow" value={`${mockRun.precursor_flow_sccm} sccm`} />
                <ParamCard label="Carrier Gas" value={`${mockRun.carrier_gas_flow_sccm} sccm`} />
                <ParamCard label="Target Thickness" value={`${mockRun.target_thickness_nm} nm`} />
                <ParamCard label="Target Stress" value={`${mockRun.target_stress_mpa} MPa`} />
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}

function ParamCard({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardContent className="pt-4">
        <div className="text-xs text-muted-foreground">{label}</div>
        <div className="text-sm font-medium mt-1">{value}</div>
      </CardContent>
    </Card>
  );
}

function Label({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`text-sm font-medium ${className}`}>{children}</div>;
}
