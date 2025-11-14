"use client";

import React, { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  AlertCircle,
  CheckCircle,
  TrendingUp,
  Activity,
  BarChart3,
  RefreshCw,
  Loader2,
  Download,
} from "lucide-react";
import SPCChart from "./SPCChart";
import { cvdApi, SPCSeries } from "@/lib/api/cvd";

interface SPCDashboardProps {
  organizationId: string;
  recipeId?: string;
  processModId?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

interface MetricCategory {
  name: string;
  metrics: string[];
  icon: React.ElementType;
}

const METRIC_CATEGORIES: MetricCategory[] = [
  {
    name: "Thickness",
    metrics: [
      "thickness_nm",
      "thickness_uniformity_pct",
      "thickness_range_nm",
      "thickness_std_nm",
    ],
    icon: Activity,
  },
  {
    name: "Process Parameters",
    metrics: [
      "temperature_avg_c",
      "pressure_avg_pa",
      "flow_total_sccm",
      "deposition_rate_nm_min",
    ],
    icon: TrendingUp,
  },
  {
    name: "Quality",
    metrics: [
      "refractive_index",
      "stress_mpa",
      "defect_count",
      "uniformity_pct",
    ],
    icon: CheckCircle,
  },
];

export default function SPCDashboard({
  organizationId,
  recipeId,
  processModId,
  autoRefresh = true,
  refreshInterval = 30,
}: SPCDashboardProps) {
  const [allSeries, setAllSeries] = useState<SPCSeries[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [selectedRecipe, setSelectedRecipe] = useState<string>(recipeId || "");
  const [availableRecipes, setAvailableRecipes] = useState<
    { id: string; name: string }[]
  >([]);
  const [dashboardStats, setDashboardStats] = useState({
    totalCharts: 0,
    activeCharts: 0,
    chartsInControl: 0,
    chartsOutOfControl: 0,
  });

  useEffect(() => {
    loadRecipes();
  }, [organizationId]);

  useEffect(() => {
    loadAllSeries();
  }, [organizationId, selectedRecipe, processModId]);

  const loadRecipes = async () => {
    try {
      const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
      const params: any = {};

      // Only include org_id if it's a valid UUID
      if (organizationId && uuidRegex.test(organizationId)) {
        params.org_id = organizationId;
      }

      const recipes = await cvdApi.getRecipes(params);

      setAvailableRecipes(
        recipes.map((r) => ({
          id: r.id,
          name: r.name,
        }))
      );

      if (recipes.length > 0 && !selectedRecipe) {
        setSelectedRecipe(recipes[0].id);
      }
    } catch (err) {
      console.error("Failed to load recipes:", err);
    }
  };

  const loadAllSeries = async () => {
    try {
      setIsLoading(true);

      const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
      const params: any = {};

      // Only include org_id if it's a valid UUID
      if (organizationId && uuidRegex.test(organizationId)) {
        params.org_id = organizationId;
      }

      if (selectedRecipe) params.recipe_id = selectedRecipe;
      if (processModId) params.process_mode_id = processModId;

      const series = await cvdApi.getSPCSeries(params);
      setAllSeries(series);

      // Calculate statistics
      // In a real implementation, we would load points for each series
      // and calculate actual in-control status
      setDashboardStats({
        totalCharts: series.length,
        activeCharts: series.filter((s) => s.is_active).length,
        chartsInControl: Math.floor(series.length * 0.85), // Mock: 85% in control
        chartsOutOfControl: Math.ceil(series.length * 0.15), // Mock: 15% out of control
      });
    } catch (err) {
      console.error("Failed to load SPC series:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const getFilteredSeries = () => {
    if (selectedCategory === "all") {
      return allSeries;
    }

    const category = METRIC_CATEGORIES.find((c) => c.name === selectedCategory);
    if (!category) return allSeries;

    return allSeries.filter((s) =>
      category.metrics.some((m) => s.metric_name.includes(m))
    );
  };

  const filteredSeries = getFilteredSeries();

  const handleRefresh = () => {
    loadAllSeries();
  };

  const handleExport = () => {
    // In a real implementation, this would export SPC data to CSV/Excel
    alert("Export functionality would generate CSV/Excel with SPC data");
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>SPC Dashboard</CardTitle>
              <CardDescription>
                Statistical Process Control - Monitor process stability and
                capability
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              {autoRefresh && (
                <Badge variant="outline" className="gap-1">
                  <RefreshCw className="h-3 w-3 animate-spin" />
                  Auto-refresh {refreshInterval}s
                </Badge>
              )}
              <Button variant="outline" size="sm" onClick={handleExport}>
                <Download className="h-4 w-4 mr-1" />
                Export
              </Button>
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

        <CardContent>
          {/* Recipe Selection */}
          {!recipeId && availableRecipes.length > 0 && (
            <div className="mb-4">
              <Select value={selectedRecipe} onValueChange={setSelectedRecipe}>
                <SelectTrigger className="w-64">
                  <SelectValue placeholder="Select recipe..." />
                </SelectTrigger>
                <SelectContent>
                  {availableRecipes.map((recipe) => (
                    <SelectItem key={recipe.id} value={recipe.id}>
                      {recipe.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Summary Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="p-4">
                <div className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5 text-blue-600" />
                  <CardDescription className="text-xs">
                    Total Charts
                  </CardDescription>
                </div>
                <CardTitle className="text-2xl">
                  {dashboardStats.totalCharts}
                </CardTitle>
              </CardHeader>
            </Card>

            <Card>
              <CardHeader className="p-4">
                <div className="flex items-center gap-2">
                  <Activity className="h-5 w-5 text-green-600" />
                  <CardDescription className="text-xs">
                    Active Charts
                  </CardDescription>
                </div>
                <CardTitle className="text-2xl">
                  {dashboardStats.activeCharts}
                </CardTitle>
              </CardHeader>
            </Card>

            <Card>
              <CardHeader className="p-4">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-5 w-5 text-green-600" />
                  <CardDescription className="text-xs">
                    In Control
                  </CardDescription>
                </div>
                <CardTitle className="text-2xl text-green-600">
                  {dashboardStats.chartsInControl}
                </CardTitle>
              </CardHeader>
            </Card>

            <Card>
              <CardHeader className="p-4">
                <div className="flex items-center gap-2">
                  <AlertCircle className="h-5 w-5 text-red-600" />
                  <CardDescription className="text-xs">
                    Out of Control
                  </CardDescription>
                </div>
                <CardTitle className="text-2xl text-red-600">
                  {dashboardStats.chartsOutOfControl}
                </CardTitle>
              </CardHeader>
            </Card>
          </div>
        </CardContent>
      </Card>

      {/* Alerts */}
      {dashboardStats.chartsOutOfControl > 0 && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {dashboardStats.chartsOutOfControl} chart
            {dashboardStats.chartsOutOfControl > 1 ? "s are" : " is"} currently
            showing out-of-control conditions. Review charts below for details.
          </AlertDescription>
        </Alert>
      )}

      {/* Category Tabs */}
      <Tabs value={selectedCategory} onValueChange={setSelectedCategory}>
        <TabsList>
          <TabsTrigger value="all">All Metrics</TabsTrigger>
          {METRIC_CATEGORIES.map((category) => {
            const Icon = category.icon;
            return (
              <TabsTrigger key={category.name} value={category.name}>
                <Icon className="h-4 w-4 mr-1" />
                {category.name}
              </TabsTrigger>
            );
          })}
        </TabsList>

        <TabsContent value={selectedCategory} className="space-y-4 mt-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
          ) : filteredSeries.length === 0 ? (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                No SPC charts found for this category. Charts will be
                automatically created as runs are processed.
              </AlertDescription>
            </Alert>
          ) : (
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              {filteredSeries.map((series) => (
                <SPCChart
                  key={series.id}
                  seriesId={series.id}
                  organizationId={organizationId}
                  autoRefresh={autoRefresh}
                  refreshInterval={refreshInterval}
                />
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>

      {/* Process Capability Summary */}
      {filteredSeries.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Process Capability Summary</CardTitle>
            <CardDescription>
              Key metrics for process performance assessment
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {filteredSeries.slice(0, 6).map((series) => {
                // Mock Cpk calculation (would be calculated from actual data)
                const cpk = 1.33 + Math.random() * 0.5;
                const cpkColor =
                  cpk >= 1.67
                    ? "text-green-600"
                    : cpk >= 1.33
                    ? "text-blue-600"
                    : cpk >= 1.0
                    ? "text-yellow-600"
                    : "text-red-600";

                return (
                  <div
                    key={series.id}
                    className="border rounded p-3 space-y-2"
                  >
                    <div className="font-semibold text-sm">
                      {series.metric_name}
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <span className="text-gray-600">Target:</span>{" "}
                        {series.center_line.toFixed(2)}
                      </div>
                      <div>
                        <span className="text-gray-600">Cpk:</span>{" "}
                        <span className={cpkColor}>{cpk.toFixed(2)}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">UCL:</span>{" "}
                        {series.ucl.toFixed(2)}
                      </div>
                      <div>
                        <span className="text-gray-600">LCL:</span>{" "}
                        {series.lcl.toFixed(2)}
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      {cpk >= 1.33 ? (
                        <>
                          <CheckCircle className="h-3 w-3 text-green-600" />
                          <span className="text-xs text-green-600">
                            Capable
                          </span>
                        </>
                      ) : (
                        <>
                          <AlertCircle className="h-3 w-3 text-yellow-600" />
                          <span className="text-xs text-yellow-600">
                            Needs Improvement
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="mt-4 text-xs text-gray-600">
              <strong>Cpk Interpretation:</strong> ≥1.67 (Excellent), ≥1.33
              (Capable), ≥1.0 (Marginal), &lt;1.0 (Incapable)
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
