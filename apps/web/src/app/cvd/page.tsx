"use client";

/**
 * CVD Overview Dashboard
 *
 * Displays:
 * - Average thickness per tool/recipe
 * - Stress distribution
 * - Adhesion class distribution per tool/recipe
 * - Recent alerts
 */

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
} from "recharts";
import { AlertList } from "@/components/cvd/metrics/AlertBanner";
import { AdhesionBadge } from "@/components/cvd/metrics/AdhesionChip";
import { ArrowRight, TrendingUp, TrendingDown, AlertCircle, Activity } from "lucide-react";
import Link from "next/link";

export default function CVDOverviewPage() {
  // Fetch overview data
  const { data: overview, isLoading } = useQuery({
    queryKey: ["cvd-overview"],
    queryFn: async () => {
      const response = await fetch("http://localhost:8001/api/cvd/overview");
      if (!response.ok) throw new Error("Failed to fetch overview");
      return response.json();
    },
    refetchInterval: 30000, // Refresh every 30s
  });

  // Fetch recent alerts
  const { data: alerts } = useQuery({
    queryKey: ["cvd-alerts"],
    queryFn: async () => {
      const response = await fetch("http://localhost:8001/api/cvd/alerts?limit=10");
      if (!response.ok) throw new Error("Failed to fetch alerts");
      return response.json();
    },
    refetchInterval: 10000, // Refresh every 10s
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  // Mock data for demonstration (replace with real data from overview)
  const thicknessData = overview?.thickness_by_tool || [
    { tool: "CVD-01", avg_thickness: 98.5, target: 100, runs: 45 },
    { tool: "CVD-02", avg_thickness: 102.3, target: 100, runs: 38 },
    { tool: "CVD-03", avg_thickness: 99.1, target: 100, runs: 52 },
  ];

  const stressDistribution = overview?.stress_distribution || [
    { range: "< -400", count: 5, severity: "high_compressive" },
    { range: "-400 to -200", count: 12, severity: "compressive" },
    { range: "-200 to 0", count: 28, severity: "low_compressive" },
    { range: "0 to 200", count: 35, severity: "low_tensile" },
    { range: "200 to 300", count: 15, severity: "tensile" },
    { range: "> 300", count: 8, severity: "high_tensile" },
  ];

  const adhesionByTool = overview?.adhesion_by_tool || [
    { tool: "CVD-01", excellent: 35, good: 8, fair: 2, poor: 0 },
    { tool: "CVD-02", excellent: 28, good: 9, fair: 1, poor: 0 },
    { tool: "CVD-03", excellent: 42, good: 7, fair: 3, poor: 0 },
  ];

  const recentTrends = overview?.recent_trends || [
    { date: "2025-11-08", avg_thickness: 99.2, avg_stress: -180, avg_adhesion: 87 },
    { date: "2025-11-09", avg_thickness: 100.1, avg_stress: -165, avg_adhesion: 89 },
    { date: "2025-11-10", avg_thickness: 98.8, avg_stress: -175, avg_adhesion: 88 },
    { date: "2025-11-11", avg_thickness: 101.2, avg_stress: -155, avg_adhesion: 90 },
    { date: "2025-11-12", avg_thickness: 99.5, avg_stress: -170, avg_adhesion: 86 },
    { date: "2025-11-13", avg_thickness: 100.3, avg_stress: -160, avg_adhesion: 91 },
    { date: "2025-11-14", avg_thickness: 99.8, avg_stress: -168, avg_adhesion: 89 },
  ];

  const stressColors = {
    high_compressive: "#ef4444",
    compressive: "#f97316",
    low_compressive: "#3b82f6",
    low_tensile: "#10b981",
    tensile: "#f59e0b",
    high_tensile: "#dc2626",
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">CVD Overview</h1>
          <p className="text-muted-foreground mt-1">
            Real-time monitoring of thickness, stress, and adhesion metrics
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/cvd/recipes">
              Recipes <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
          <Button asChild>
            <Link href="/cvd/runs">
              View Runs <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Active Runs
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{overview?.active_runs || 12}</div>
            <div className="flex items-center text-xs text-muted-foreground mt-1">
              <Activity className="h-3 w-3 mr-1" />
              <span>Across 3 tools</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Avg Thickness
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {overview?.avg_thickness?.toFixed(1) || "99.8"} nm
            </div>
            <div className="flex items-center text-xs text-green-600 mt-1">
              <TrendingUp className="h-3 w-3 mr-1" />
              <span>+0.5% vs target</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Avg Stress
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {overview?.avg_stress?.toFixed(0) || "-168"} MPa
            </div>
            <div className="flex items-center text-xs text-blue-600 mt-1">
              <TrendingDown className="h-3 w-3 mr-1" />
              <span>Within safe zone</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Alerts (24h)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{alerts?.length || 0}</div>
            <div className="flex items-center text-xs text-muted-foreground mt-1">
              <AlertCircle className="h-3 w-3 mr-1" />
              <span>{alerts?.filter((a: any) => a.severity === "critical").length || 0} critical</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Charts */}
        <div className="lg:col-span-2 space-y-6">
          {/* Thickness by Tool */}
          <Card>
            <CardHeader>
              <CardTitle>Average Thickness by Tool</CardTitle>
              <CardDescription>Last 30 days</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={thicknessData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="tool" />
                  <YAxis label={{ value: "Thickness (nm)", angle: -90, position: "insideLeft" }} />
                  <RechartsTooltip />
                  <Legend />
                  <Bar dataKey="avg_thickness" name="Actual" fill="#3b82f6" />
                  <Bar dataKey="target" name="Target" fill="#94a3b8" />
                </BarChart>
              </ResponsiveContainer>
              <div className="flex justify-around mt-4 text-xs text-muted-foreground">
                {thicknessData.map(tool => (
                  <div key={tool.tool} className="text-center">
                    <div className="font-medium text-foreground">{tool.tool}</div>
                    <div>{tool.runs} runs</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Stress Distribution */}
          <Card>
            <CardHeader>
              <CardTitle>Stress Distribution</CardTitle>
              <CardDescription>All runs (last 30 days)</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={stressDistribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" />
                  <YAxis label={{ value: "Count", angle: -90, position: "insideLeft" }} />
                  <RechartsTooltip />
                  <Bar dataKey="count" name="Runs">
                    {stressDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={stressColors[entry.severity as keyof typeof stressColors]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="flex items-center justify-center gap-4 mt-4 text-xs">
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded bg-green-500" />
                  <span>Safe Zone</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded bg-orange-500" />
                  <span>Warning</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded bg-red-500" />
                  <span>Critical</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Recent Trends */}
          <Card>
            <CardHeader>
              <CardTitle>7-Day Trends</CardTitle>
              <CardDescription>Average metrics over time</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="thickness">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="thickness">Thickness</TabsTrigger>
                  <TabsTrigger value="stress">Stress</TabsTrigger>
                  <TabsTrigger value="adhesion">Adhesion</TabsTrigger>
                </TabsList>

                <TabsContent value="thickness">
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={recentTrends}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" tickFormatter={(date) => new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} />
                      <YAxis domain={[95, 105]} />
                      <RechartsTooltip />
                      <Line type="monotone" dataKey="avg_thickness" stroke="#3b82f6" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </TabsContent>

                <TabsContent value="stress">
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={recentTrends}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" tickFormatter={(date) => new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} />
                      <YAxis domain={[-200, -150]} />
                      <RechartsTooltip />
                      <Line type="monotone" dataKey="avg_stress" stroke="#f59e0b" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </TabsContent>

                <TabsContent value="adhesion">
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={recentTrends}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" tickFormatter={(date) => new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} />
                      <YAxis domain={[80, 95]} />
                      <RechartsTooltip />
                      <Line type="monotone" dataKey="avg_adhesion" stroke="#10b981" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>

        {/* Right Column - Alerts & Adhesion */}
        <div className="space-y-6">
          {/* Recent Alerts */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Recent Alerts</span>
                <Badge variant="destructive">{alerts?.length || 0}</Badge>
              </CardTitle>
              <CardDescription>Last 24 hours</CardDescription>
            </CardHeader>
            <CardContent>
              <AlertList alerts={alerts || []} maxVisible={5} />
              <Button variant="outline" className="w-full mt-4" size="sm" asChild>
                <Link href="/cvd/runs?filter=alerts">View All Alerts</Link>
              </Button>
            </CardContent>
          </Card>

          {/* Adhesion by Tool */}
          <Card>
            <CardHeader>
              <CardTitle>Adhesion by Tool</CardTitle>
              <CardDescription>Last 30 days</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {adhesionByTool.map(tool => {
                const total = tool.excellent + tool.good + tool.fair + tool.poor;
                return (
                  <div key={tool.tool} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium">{tool.tool}</span>
                      <span className="text-muted-foreground">{total} runs</span>
                    </div>
                    <div className="flex gap-1 h-4">
                      <div
                        className="bg-green-500 rounded-l"
                        style={{ width: `${(tool.excellent / total) * 100}%` }}
                        title={`Excellent: ${tool.excellent}`}
                      />
                      <div
                        className="bg-blue-500"
                        style={{ width: `${(tool.good / total) * 100}%` }}
                        title={`Good: ${tool.good}`}
                      />
                      <div
                        className="bg-yellow-500"
                        style={{ width: `${(tool.fair / total) * 100}%` }}
                        title={`Fair: ${tool.fair}`}
                      />
                      {tool.poor > 0 && (
                        <div
                          className="bg-red-500 rounded-r"
                          style={{ width: `${(tool.poor / total) * 100}%` }}
                          title={`Poor: ${tool.poor}`}
                        />
                      )}
                    </div>
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>{((tool.excellent / total) * 100).toFixed(0)}% excellent</span>
                      <span>{((tool.good / total) * 100).toFixed(0)}% good</span>
                    </div>
                  </div>
                );
              })}
              <div className="flex items-center justify-center gap-3 pt-2 border-t text-xs">
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded bg-green-500" />
                  <span>Excellent</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded bg-blue-500" />
                  <span>Good</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded bg-yellow-500" />
                  <span>Fair</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded bg-red-500" />
                  <span>Poor</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Quick Links */}
          <Card>
            <CardHeader>
              <CardTitle>Quick Links</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button variant="outline" className="w-full justify-start" asChild>
                <Link href="/cvd/recipes">
                  <ArrowRight className="mr-2 h-4 w-4" />
                  Manage Recipes
                </Link>
              </Button>
              <Button variant="outline" className="w-full justify-start" asChild>
                <Link href="/cvd/runs">
                  <ArrowRight className="mr-2 h-4 w-4" />
                  View All Runs
                </Link>
              </Button>
              <Button variant="outline" className="w-full justify-start" asChild>
                <Link href="/cvd/workspace">
                  <ArrowRight className="mr-2 h-4 w-4" />
                  Full Workspace
                </Link>
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
