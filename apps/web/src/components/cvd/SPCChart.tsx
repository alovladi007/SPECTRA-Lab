"use client";

import React, { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
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
  Scatter,
  ScatterChart,
  ZAxis,
} from "recharts";
import {
  AlertCircle,
  CheckCircle,
  TrendingUp,
  TrendingDown,
  Info,
  RefreshCw,
  Loader2,
} from "lucide-react";
import { cvdApi, SPCSeries, SPCPoint } from "@/lib/api/cvd";

interface SPCChartProps {
  seriesId?: string;
  organizationId: string;
  recipeId?: string;
  processModId?: string;
  metricName?: string;
  chartType?: "xbar-r" | "i-mr" | "ewma" | "cusum" | "p" | "np" | "c" | "u";
  autoRefresh?: boolean;
  refreshInterval?: number; // in seconds
}

interface SPCDataPoint {
  timestamp: string;
  value: number;
  outOfControl: boolean;
  violations: string[];
  displayTime: string;
  runId?: string;
  subgroupId?: string;
}

const CHART_TYPE_LABELS: Record<string, string> = {
  "xbar-r": "X-bar R Chart (Subgroup Mean)",
  "i-mr": "I-MR Chart (Individual & Moving Range)",
  ewma: "EWMA Chart (Exponentially Weighted)",
  cusum: "CUSUM Chart (Cumulative Sum)",
  p: "P Chart (Proportion Defective)",
  np: "NP Chart (Number Defective)",
  c: "C Chart (Count of Defects)",
  u: "U Chart (Defects per Unit)",
};

const WESTERN_ELECTRIC_RULES = [
  "Rule 1: One point beyond 3σ",
  "Rule 2: Two of three consecutive points beyond 2σ on same side",
  "Rule 3: Four of five consecutive points beyond 1σ on same side",
  "Rule 4: Eight consecutive points on same side of center line",
  "Rule 5: Obvious trend (6+ points in a row increasing/decreasing)",
  "Rule 6: Two of three points near control limits",
  "Rule 7: Fifteen points in a row within 1σ of center line",
  "Rule 8: Eight consecutive points with none within 1σ of center line",
];

export default function SPCChart({
  seriesId,
  organizationId,
  recipeId,
  processModId,
  metricName,
  chartType = "xbar-r",
  autoRefresh = false,
  refreshInterval = 30,
}: SPCChartProps) {
  const [series, setSeries] = useState<SPCSeries | null>(null);
  const [dataPoints, setDataPoints] = useState<SPCDataPoint[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedSeries, setSelectedSeries] = useState<string>("");
  const [availableSeries, setAvailableSeries] = useState<SPCSeries[]>([]);
  const [stats, setStats] = useState<{
    totalPoints: number;
    outOfControlCount: number;
    cpk?: number;
    sigma?: number;
  }>({
    totalPoints: 0,
    outOfControlCount: 0,
  });

  // Load available series on mount
  useEffect(() => {
    loadAvailableSeries();
  }, [organizationId, recipeId, processModId, metricName]);

  // Load data when series is selected
  useEffect(() => {
    if (seriesId || selectedSeries) {
      loadSeriesData(seriesId || selectedSeries);
    }
  }, [seriesId, selectedSeries]);

  // Auto-refresh
  useEffect(() => {
    if (autoRefresh && (seriesId || selectedSeries)) {
      const interval = setInterval(() => {
        loadSeriesData(seriesId || selectedSeries);
      }, refreshInterval * 1000);

      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval, seriesId, selectedSeries]);

  const loadAvailableSeries = async () => {
    try {
      const params: any = {};

      // Only include org_id if it's a valid UUID (not "default-org" or "undefined")
      const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
      if (organizationId && uuidRegex.test(organizationId)) {
        params.org_id = organizationId;
      }

      if (recipeId) params.recipe_id = recipeId;
      if (processModId) params.process_mode_id = processModId;
      if (metricName) params.metric_name = metricName;

      const data = await cvdApi.getSPCSeries(params);
      setAvailableSeries(data);

      // Auto-select first series if seriesId not provided
      if (!seriesId && data.length > 0) {
        setSelectedSeries(data[0].id);
      }
    } catch (err: any) {
      console.error("Failed to load SPC series:", err);
    }
  };

  const loadSeriesData = async (id: string) => {
    try {
      setIsLoading(true);
      setError(null);

      // Load series metadata
      const seriesData = availableSeries.find((s) => s.id === id);
      if (seriesData) {
        setSeries(seriesData);
      }

      // Load points (last 100 points)
      const points = await cvdApi.getSPCPoints(id, 100);

      // Transform data
      const transformedData: SPCDataPoint[] = points.map((point) => ({
        timestamp: point.timestamp,
        value: point.value,
        outOfControl: point.out_of_control,
        violations: point.violation_rules,
        displayTime: new Date(point.timestamp).toLocaleString(),
        runId: point.run_id,
        subgroupId: point.subgroup_id,
      }));

      setDataPoints(transformedData);

      // Calculate statistics
      const outOfControlCount = transformedData.filter(
        (p) => p.outOfControl
      ).length;

      setStats({
        totalPoints: transformedData.length,
        outOfControlCount,
      });
    } catch (err: any) {
      setError(`Failed to load SPC data: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRefresh = () => {
    if (seriesId || selectedSeries) {
      loadSeriesData(seriesId || selectedSeries);
    }
  };

  const renderTooltip = ({ active, payload }: any) => {
    if (!active || !payload || payload.length === 0) return null;

    const data = payload[0].payload as SPCDataPoint;

    return (
      <div className="bg-white p-3 border rounded shadow-lg">
        <div className="text-sm font-semibold">{data.displayTime}</div>
        <div className="text-sm mt-1">
          <strong>Value:</strong> {data.value.toFixed(3)}
        </div>
        {data.runId && (
          <div className="text-xs text-gray-600">Run: {data.runId}</div>
        )}
        {data.outOfControl && (
          <div className="mt-2 pt-2 border-t">
            <div className="text-xs font-semibold text-red-600">
              Out of Control
            </div>
            {data.violations.length > 0 && (
              <div className="text-xs text-gray-600 mt-1">
                Violations:
                <ul className="list-disc list-inside">
                  {data.violations.map((v, i) => (
                    <li key={i}>{v}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  const getPointColor = (point: SPCDataPoint) => {
    if (point.outOfControl) return "#ef4444"; // red
    return "#3b82f6"; // blue
  };

  const renderChart = () => {
    if (!series) return null;

    const chartData = dataPoints.map((point, index) => ({
      ...point,
      index: index + 1,
      color: getPointColor(point),
    }));

    return (
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="index"
            label={{ value: "Sample Number", position: "insideBottom", offset: -5 }}
          />
          <YAxis
            label={{ value: series.metric_name, angle: -90, position: "insideLeft" }}
          />
          <Tooltip content={renderTooltip} />
          <Legend />

          {/* Control Limits */}
          <ReferenceLine
            y={series.ucl}
            stroke="#ef4444"
            strokeDasharray="3 3"
            label={{ value: "UCL", position: "right" }}
          />
          <ReferenceLine
            y={series.center_line}
            stroke="#22c55e"
            strokeWidth={2}
            label={{ value: "CL", position: "right" }}
          />
          <ReferenceLine
            y={series.lcl}
            stroke="#ef4444"
            strokeDasharray="3 3"
            label={{ value: "LCL", position: "right" }}
          />

          {/* Specification Limits (if available) */}
          {series.usl && (
            <ReferenceLine
              y={series.usl}
              stroke="#f59e0b"
              strokeDasharray="5 5"
              label={{ value: "USL", position: "right" }}
            />
          )}
          {series.lsl && (
            <ReferenceLine
              y={series.lsl}
              stroke="#f59e0b"
              strokeDasharray="5 5"
              label={{ value: "LSL", position: "right" }}
            />
          )}

          {/* Data Line */}
          <Line
            type="monotone"
            dataKey="value"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={(props: any) => {
              const { cx, cy, payload } = props;
              return (
                <circle
                  cx={cx}
                  cy={cy}
                  r={payload.outOfControl ? 6 : 4}
                  fill={payload.color}
                  stroke={payload.outOfControl ? "#dc2626" : "#3b82f6"}
                  strokeWidth={payload.outOfControl ? 2 : 1}
                />
              );
            }}
            name={series.metric_name}
          />
        </LineChart>
      </ResponsiveContainer>
    );
  };

  const processStability =
    stats.totalPoints > 0
      ? ((stats.totalPoints - stats.outOfControlCount) / stats.totalPoints) *
        100
      : 0;

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>SPC Chart</CardTitle>
            <CardDescription>
              {series
                ? `${CHART_TYPE_LABELS[series.chart_type]} - ${series.metric_name}`
                : "Select a chart to view"}
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {autoRefresh && (
              <Badge variant="outline" className="gap-1">
                <RefreshCw className="h-3 w-3 animate-spin" />
                Auto
              </Badge>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              disabled={isLoading}
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Series Selection */}
        {!seriesId && availableSeries.length > 0 && (
          <div>
            <Select value={selectedSeries} onValueChange={setSelectedSeries}>
              <SelectTrigger>
                <SelectValue placeholder="Select SPC chart..." />
              </SelectTrigger>
              <SelectContent>
                {availableSeries.map((s) => (
                  <SelectItem key={s.id} value={s.id}>
                    {s.metric_name} ({CHART_TYPE_LABELS[s.chart_type]})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Statistics Cards */}
        {series && dataPoints.length > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="p-4">
                <CardDescription className="text-xs">
                  Total Points
                </CardDescription>
                <CardTitle className="text-2xl">
                  {stats.totalPoints}
                </CardTitle>
              </CardHeader>
            </Card>

            <Card>
              <CardHeader className="p-4">
                <CardDescription className="text-xs">
                  Out of Control
                </CardDescription>
                <CardTitle className="text-2xl text-red-600">
                  {stats.outOfControlCount}
                </CardTitle>
              </CardHeader>
            </Card>

            <Card>
              <CardHeader className="p-4">
                <CardDescription className="text-xs">
                  Process Stability
                </CardDescription>
                <CardTitle
                  className={`text-2xl ${
                    processStability >= 99
                      ? "text-green-600"
                      : processStability >= 95
                      ? "text-yellow-600"
                      : "text-red-600"
                  }`}
                >
                  {processStability.toFixed(1)}%
                </CardTitle>
              </CardHeader>
            </Card>

            <Card>
              <CardHeader className="p-4">
                <CardDescription className="text-xs">Status</CardDescription>
                <CardTitle className="text-sm flex items-center gap-2">
                  {stats.outOfControlCount === 0 ? (
                    <>
                      <CheckCircle className="h-5 w-5 text-green-600" />
                      <span className="text-green-600">In Control</span>
                    </>
                  ) : (
                    <>
                      <AlertCircle className="h-5 w-5 text-red-600" />
                      <span className="text-red-600">Out of Control</span>
                    </>
                  )}
                </CardTitle>
              </CardHeader>
            </Card>
          </div>
        )}

        {/* Control Limits Info */}
        {series && (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-sm">
            <div>
              <span className="text-gray-600">UCL:</span>{" "}
              <strong>{series.ucl.toFixed(3)}</strong>
            </div>
            <div>
              <span className="text-gray-600">CL:</span>{" "}
              <strong>{series.center_line.toFixed(3)}</strong>
            </div>
            <div>
              <span className="text-gray-600">LCL:</span>{" "}
              <strong>{series.lcl.toFixed(3)}</strong>
            </div>
            {series.usl && (
              <div>
                <span className="text-gray-600">USL:</span>{" "}
                <strong>{series.usl.toFixed(3)}</strong>
              </div>
            )}
            {series.lsl && (
              <div>
                <span className="text-gray-600">LSL:</span>{" "}
                <strong>{series.lsl.toFixed(3)}</strong>
              </div>
            )}
          </div>
        )}

        {/* Chart */}
        {isLoading ? (
          <div className="flex items-center justify-center h-96">
            <Loader2 className="h-8 w-8 animate-spin" />
          </div>
        ) : dataPoints.length > 0 ? (
          renderChart()
        ) : (
          <Alert>
            <Info className="h-4 w-4" />
            <AlertDescription>
              No data points available. Data will appear as runs are processed.
            </AlertDescription>
          </Alert>
        )}

        {/* Western Electric Rules Info */}
        {series && dataPoints.length > 0 && (
          <details className="text-sm">
            <summary className="cursor-pointer font-semibold text-gray-700">
              Control Chart Rules (Western Electric)
            </summary>
            <ul className="mt-2 space-y-1 text-gray-600 list-disc list-inside">
              {WESTERN_ELECTRIC_RULES.map((rule, index) => (
                <li key={index}>{rule}</li>
              ))}
            </ul>
          </details>
        )}

        {/* Recent Violations */}
        {stats.outOfControlCount > 0 && (
          <div>
            <div className="font-semibold text-sm mb-2">
              Recent Violations
            </div>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {dataPoints
                .filter((p) => p.outOfControl)
                .slice(-5)
                .reverse()
                .map((point, index) => (
                  <Alert key={index} variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      <div className="text-xs">
                        <strong>{point.displayTime}</strong> - Value:{" "}
                        {point.value.toFixed(3)}
                      </div>
                      {point.violations.length > 0 && (
                        <div className="text-xs mt-1">
                          {point.violations.join(", ")}
                        </div>
                      )}
                    </AlertDescription>
                  </Alert>
                ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
