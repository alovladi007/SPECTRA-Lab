"use client";

/**
 * CVD Results Deep-Dive Page
 *
 * Features:
 * - Wafer thickness map (2D visualization)
 * - Histograms of thickness and stress distributions
 * - Adhesion test results (table and plots)
 * - SPC summary charts
 * - VM (Virtual Metrology) residual plots
 * - Detailed statistical analysis
 */

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams, useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { WaferMap, generateWaferPoints, type WaferPoint } from "@/components/cvd/metrics/WaferMap";
import { AdhesionDetail, TEST_METHODS } from "@/components/cvd/metrics/AdhesionChip";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  LineChart,
  Line,
  ReferenceLine,
} from "recharts";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ArrowLeft, Download, CheckCircle, AlertCircle } from "lucide-react";
import Link from "next/link";

export default function ResultsDeepDivePage() {
  const params = useParams();
  const router = useRouter();
  const resultId = params.id as string;

  // Fetch results
  const { data: results, isLoading } = useQuery({
    queryKey: ["cvd-results", resultId],
    queryFn: async () => {
      const response = await fetch(`http://localhost:8001/api/cvd/results/${resultId}`);
      if (!response.ok) throw new Error("Failed to fetch results");
      return response.json();
    },
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  // Mock data for demonstration
  const mockResults = results || {
    run_id: "CVD_RUN_20251114_103045",
    recipe_name: "Si3N4 Standard",
    tool: "CVD-01",
    wafer_id: "W12345",
    completed_at: "2025-11-14T11:35:00",

    // Wafer map data (9-point measurement)
    wafer_measurements: generateWaferPoints.ninePoint(150).map((p, i) => ({
      ...p,
      value: 100 + (Math.random() - 0.5) * 4, // Thickness variation
    })),

    // Thickness statistics
    thickness_stats: {
      mean: 100.2,
      std_dev: 1.8,
      min: 96.5,
      max: 103.1,
      uniformity: 1.8,
      cpk: 1.45,
    },

    // Stress statistics
    stress_stats: {
      mean: -192,
      std_dev: 18,
      min: -225,
      max: -165,
    },

    // Adhesion test results
    adhesion_tests: [
      { location: "Center", score: 92, test_method: "ASTM D3359", result: "5B" },
      { location: "North", score: 89, test_method: "ASTM D3359", result: "5B" },
      { location: "East", score: 90, test_method: "ASTM D3359", result: "5B" },
      { location: "South", score: 87, test_method: "ASTM D3359", result: "4B" },
      { location: "West", score: 91, test_method: "ASTM D3359", result: "5B" },
    ],
    adhesion_mean: 89.8,

    // SPC data
    spc_data: Array.from({ length: 30 }, (_, i) => ({
      run_number: i + 1,
      thickness: 100 + (Math.random() - 0.5) * 5,
      ucl: 105,
      lcl: 95,
      target: 100,
    })),

    // VM residuals
    vm_residuals: Array.from({ length: 50 }, (_, i) => ({
      predicted: 100 + (Math.random() - 0.5) * 8,
      actual: 100 + (Math.random() - 0.5) * 8,
    })).map(p => ({
      ...p,
      residual: p.actual - p.predicted,
    })),
  };

  // Generate histogram data
  const thicknessHistogram = Array.from({ length: 15 }, (_, i) => {
    const binCenter = 95 + i * 1;
    const count = Math.exp(-Math.pow((binCenter - mockResults.thickness_stats.mean) / mockResults.thickness_stats.std_dev, 2) / 2) * 10;
    return {
      bin: `${binCenter.toFixed(0)}-${(binCenter + 1).toFixed(0)}`,
      count: Math.round(count),
    };
  });

  const stressHistogram = Array.from({ length: 15 }, (_, i) => {
    const binCenter = -250 + i * 10;
    const count = Math.exp(-Math.pow((binCenter - mockResults.stress_stats.mean) / mockResults.stress_stats.std_dev, 2) / 2) * 8;
    return {
      bin: `${binCenter}`,
      count: Math.round(count),
    };
  });

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
            <h1 className="text-3xl font-bold">Results: {mockResults.run_id}</h1>
            <p className="text-muted-foreground mt-1">
              {mockResults.recipe_name} • {mockResults.tool} • Wafer {mockResults.wafer_id}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="bg-green-50 text-green-700">
            <CheckCircle className="h-3 w-3 mr-1" />
            Completed
          </Badge>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export Report
          </Button>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Mean Thickness
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{mockResults.thickness_stats.mean} nm</div>
            <div className="text-xs text-muted-foreground mt-1">
              ±{mockResults.thickness_stats.std_dev} nm (σ)
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Uniformity
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">±{mockResults.thickness_stats.uniformity}%</div>
            <div className="text-xs text-green-600 mt-1">
              <CheckCircle className="h-3 w-3 inline mr-1" />
              Excellent
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Mean Stress
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{mockResults.stress_stats.mean} MPa</div>
            <div className="text-xs text-muted-foreground mt-1">
              ±{mockResults.stress_stats.std_dev} MPa (σ)
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Adhesion Score
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{mockResults.adhesion_mean.toFixed(1)}/100</div>
            <div className="text-xs text-green-600 mt-1">
              <CheckCircle className="h-3 w-3 inline mr-1" />
              Excellent
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content - Tabs */}
      <Tabs defaultValue="wafermap" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="wafermap">Wafer Map</TabsTrigger>
          <TabsTrigger value="distributions">Distributions</TabsTrigger>
          <TabsTrigger value="adhesion">Adhesion Tests</TabsTrigger>
          <TabsTrigger value="spc">SPC</TabsTrigger>
          <TabsTrigger value="vm">VM Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="wafermap" className="space-y-6 mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Thickness Map */}
            <WaferMap
              points={mockResults.wafer_measurements}
              parameter="Thickness"
              unit="nm"
              colorScale="viridis"
              showLegend={true}
              showGrid={true}
              highlightOutliers={true}
              size="lg"
            />

            {/* Stress Map (generate from thickness with correlation) */}
            <WaferMap
              points={mockResults.wafer_measurements.map(p => ({
                ...p,
                value: -192 + (p.value - 100) * -5, // Stress correlation
              }))}
              parameter="Stress"
              unit="MPa"
              colorScale="rdylgn"
              showLegend={true}
              showGrid={true}
              size="lg"
            />
          </div>

          {/* Measurement Details */}
          <Card>
            <CardHeader>
              <CardTitle>Measurement Points</CardTitle>
              <CardDescription>9-point thickness measurement data</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Location</TableHead>
                    <TableHead>Position (mm)</TableHead>
                    <TableHead>Thickness (nm)</TableHead>
                    <TableHead>Deviation from Mean</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {mockResults.wafer_measurements.map((point: WaferPoint, i: number) => {
                    const deviation = point.value - mockResults.thickness_stats.mean;
                    return (
                      <TableRow key={i}>
                        <TableCell className="font-medium">{point.label || `Point ${i + 1}`}</TableCell>
                        <TableCell className="text-xs font-mono">
                          ({point.x.toFixed(0)}, {point.y.toFixed(0)})
                        </TableCell>
                        <TableCell>{point.value.toFixed(2)} nm</TableCell>
                        <TableCell className={deviation > 0 ? "text-green-600" : "text-blue-600"}>
                          {deviation > 0 ? "+" : ""}{deviation.toFixed(2)} nm
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="distributions" className="space-y-6 mt-6">
          {/* Thickness Histogram */}
          <Card>
            <CardHeader>
              <CardTitle>Thickness Distribution</CardTitle>
              <CardDescription>
                Normal distribution: μ = {mockResults.thickness_stats.mean} nm, σ = {mockResults.thickness_stats.std_dev} nm
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={thicknessHistogram}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="bin" label={{ value: "Thickness (nm)", position: "insideBottom", offset: -5 }} />
                  <YAxis label={{ value: "Count", angle: -90, position: "insideLeft" }} />
                  <RechartsTooltip />
                  <Bar dataKey="count" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
              <div className="grid grid-cols-4 gap-4 mt-4 text-sm">
                <div>
                  <div className="text-muted-foreground">Min</div>
                  <div className="font-semibold">{mockResults.thickness_stats.min} nm</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Max</div>
                  <div className="font-semibold">{mockResults.thickness_stats.max} nm</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Range</div>
                  <div className="font-semibold">{(mockResults.thickness_stats.max - mockResults.thickness_stats.min).toFixed(1)} nm</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Cpk</div>
                  <div className="font-semibold text-green-600">{mockResults.thickness_stats.cpk}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Stress Histogram */}
          <Card>
            <CardHeader>
              <CardTitle>Stress Distribution</CardTitle>
              <CardDescription>
                Normal distribution: μ = {mockResults.stress_stats.mean} MPa, σ = {mockResults.stress_stats.std_dev} MPa
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={stressHistogram}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="bin" label={{ value: "Stress (MPa)", position: "insideBottom", offset: -5 }} />
                  <YAxis label={{ value: "Count", angle: -90, position: "insideLeft" }} />
                  <RechartsTooltip />
                  <Bar dataKey="count" fill="#f59e0b" />
                </BarChart>
              </ResponsiveContainer>
              <div className="grid grid-cols-3 gap-4 mt-4 text-sm">
                <div>
                  <div className="text-muted-foreground">Min</div>
                  <div className="font-semibold">{mockResults.stress_stats.min} MPa</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Max</div>
                  <div className="font-semibold">{mockResults.stress_stats.max} MPa</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Range</div>
                  <div className="font-semibold">{(mockResults.stress_stats.max - mockResults.stress_stats.min).toFixed(0)} MPa</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="adhesion" className="space-y-6 mt-6">
          {/* Adhesion Test Results Table */}
          <Card>
            <CardHeader>
              <CardTitle>Adhesion Test Results</CardTitle>
              <CardDescription>Cross-hatch tape test (ASTM D3359)</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Location</TableHead>
                    <TableHead>Test Method</TableHead>
                    <TableHead>Result</TableHead>
                    <TableHead>Score</TableHead>
                    <TableHead>Classification</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {mockResults.adhesion_tests.map((test: any, i: number) => (
                    <TableRow key={i}>
                      <TableCell className="font-medium">{test.location}</TableCell>
                      <TableCell className="text-xs">{test.test_method}</TableCell>
                      <TableCell>
                        <Badge variant="outline">{test.result}</Badge>
                      </TableCell>
                      <TableCell className="font-semibold">{test.score}/100</TableCell>
                      <TableCell>
                        <AdhesionDetail
                          score={test.score}
                          testMethod={TEST_METHODS.TAPE_TEST}
                          className="inline-flex"
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

              <div className="mt-6 p-4 bg-green-50 dark:bg-green-950 rounded-lg border border-green-200">
                <div className="flex items-start gap-2">
                  <CheckCircle className="h-5 w-5 text-green-600 mt-0.5" />
                  <div>
                    <div className="font-semibold text-green-900 dark:text-green-100">
                      All Tests Passed
                    </div>
                    <div className="text-sm text-green-700 dark:text-green-200 mt-1">
                      Mean adhesion score: {mockResults.adhesion_mean.toFixed(1)}/100 (Excellent).
                      No delamination or adhesion failures detected.
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Adhesion Score by Location */}
          <Card>
            <CardHeader>
              <CardTitle>Adhesion Score by Location</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={mockResults.adhesion_tests}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="location" />
                  <YAxis domain={[0, 100]} label={{ value: "Score", angle: -90, position: "insideLeft" }} />
                  <RechartsTooltip />
                  <Bar dataKey="score" fill="#10b981" />
                  <ReferenceLine y={80} stroke="#22c55e" strokeDasharray="3 3" label="Excellent" />
                  <ReferenceLine y={60} stroke="#eab308" strokeDasharray="3 3" label="Good" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="spc" className="space-y-6 mt-6">
          {/* SPC Control Chart */}
          <Card>
            <CardHeader>
              <CardTitle>SPC Control Chart - Thickness</CardTitle>
              <CardDescription>Last 30 runs (Cpk = {mockResults.thickness_stats.cpk})</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={mockResults.spc_data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="run_number" label={{ value: "Run Number", position: "insideBottom", offset: -5 }} />
                  <YAxis domain={[90, 110]} label={{ value: "Thickness (nm)", angle: -90, position: "insideLeft" }} />
                  <RechartsTooltip />
                  <Legend />
                  <Line type="monotone" dataKey="thickness" name="Measured" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="ucl" name="UCL" stroke="#ef4444" strokeDasharray="5 5" strokeWidth={2} />
                  <Line type="monotone" dataKey="target" name="Target" stroke="#22c55e" strokeWidth={2} />
                  <Line type="monotone" dataKey="lcl" name="LCL" stroke="#ef4444" strokeDasharray="5 5" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>

              <div className="grid grid-cols-3 gap-4 mt-6 text-sm">
                <Card className="bg-blue-50 dark:bg-blue-950">
                  <CardContent className="pt-4">
                    <div className="text-muted-foreground">Process Capability</div>
                    <div className="text-2xl font-bold mt-1">{mockResults.thickness_stats.cpk}</div>
                    <div className="text-xs text-green-600 mt-1">Capable (Cpk &gt; 1.33)</div>
                  </CardContent>
                </Card>

                <Card className="bg-green-50 dark:bg-green-950">
                  <CardContent className="pt-4">
                    <div className="text-muted-foreground">In Control</div>
                    <div className="text-2xl font-bold mt-1">100%</div>
                    <div className="text-xs text-green-600 mt-1">All points within limits</div>
                  </CardContent>
                </Card>

                <Card className="bg-gray-50 dark:bg-gray-950">
                  <CardContent className="pt-4">
                    <div className="text-muted-foreground">Sigma Level</div>
                    <div className="text-2xl font-bold mt-1">4.3σ</div>
                    <div className="text-xs text-muted-foreground mt-1">Based on Cpk</div>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="vm" className="space-y-6 mt-6">
          {/* VM Residuals Plot */}
          <Card>
            <CardHeader>
              <CardTitle>Virtual Metrology Residuals</CardTitle>
              <CardDescription>Predicted vs Actual Thickness</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="predicted" name="Predicted" unit=" nm" label={{ value: "Predicted Thickness (nm)", position: "insideBottom", offset: -5 }} />
                  <YAxis dataKey="actual" name="Actual" unit=" nm" label={{ value: "Actual Thickness (nm)", angle: -90, position: "insideLeft" }} />
                  <RechartsTooltip cursor={{ strokeDasharray: "3 3" }} />
                  <Scatter name="Measurements" data={mockResults.vm_residuals} fill="#3b82f6" />
                  <ReferenceLine y={100} stroke="#94a3b8" strokeDasharray="3 3" />
                  <ReferenceLine x={100} stroke="#94a3b8" strokeDasharray="3 3" />
                </ScatterChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Residuals Distribution */}
          <Card>
            <CardHeader>
              <CardTitle>Residuals Distribution</CardTitle>
              <CardDescription>Difference between predicted and actual values</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={Array.from({ length: 11 }, (_, i) => {
                  const bin = -5 + i;
                  const count = mockResults.vm_residuals.filter((r: any) =>
                    r.residual >= bin - 0.5 && r.residual < bin + 0.5
                  ).length;
                  return { bin: bin.toFixed(0), count };
                })}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="bin" label={{ value: "Residual (nm)", position: "insideBottom", offset: -5 }} />
                  <YAxis label={{ value: "Count", angle: -90, position: "insideLeft" }} />
                  <RechartsTooltip />
                  <Bar dataKey="count" fill="#10b981" />
                </BarChart>
              </ResponsiveContainer>

              <div className="grid grid-cols-3 gap-4 mt-6 text-sm">
                <div className="text-center">
                  <div className="text-muted-foreground">RMSE</div>
                  <div className="text-xl font-bold">1.2 nm</div>
                </div>
                <div className="text-center">
                  <div className="text-muted-foreground">R² Score</div>
                  <div className="text-xl font-bold text-green-600">0.94</div>
                </div>
                <div className="text-center">
                  <div className="text-muted-foreground">MAE</div>
                  <div className="text-xl font-bold">0.9 nm</div>
                </div>
              </div>

              <div className="mt-4 text-xs text-muted-foreground p-3 bg-muted rounded">
                <strong>Model Performance:</strong> The virtual metrology model shows excellent predictive accuracy
                with R² = 0.94 and RMSE = 1.2 nm. Residuals are normally distributed around zero, indicating
                no systematic bias.
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
