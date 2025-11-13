"use client";

/**
 * CVD Workspace Page
 * Main workspace for CVD operations - process modes, recipes, runs, and monitoring
 */

import React, { useState, useEffect, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { toast } from "sonner";
import {
  Activity,
  Beaker,
  ChevronRight,
  Database,
  FileText,
  Play,
  Plus,
  RefreshCw,
  Settings,
  TrendingUp,
  Zap,
} from "lucide-react";

// API client (to be implemented)
import { cvdApi } from "@/lib/api/cvd";

// Types
interface CVDProcessMode {
  id: string;
  pressure_mode: string;
  energy_mode: string;
  reactor_type: string;
  chemistry_type: string;
  variant?: string;
  description?: string;
  is_active: boolean;
  created_at: string;
}

interface CVDRecipe {
  id: string;
  name: string;
  description?: string;
  process_mode_id: string;
  temperature_profile: any;
  gas_flows: any;
  pressure_profile: any;
  plasma_settings?: any;
  recipe_steps: any[];
  process_time_s: number;
  target_thickness_nm?: number;
  is_baseline: boolean;
  is_golden: boolean;
  is_active: boolean;
  run_count: number;
  created_at: string;
}

interface CVDRun {
  id: string;
  recipe_id: string;
  status: string;
  lot_id?: string;
  wafer_ids: string[];
  start_time?: string;
  end_time?: string;
  duration_s?: number;
  created_at: string;
}

export default function CVDWorkspacePage() {
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedProcessMode, setSelectedProcessMode] = useState<string | null>(null);
  const [selectedRecipe, setSelectedRecipe] = useState<string | null>(null);
  const [isCreateRecipeOpen, setIsCreateRecipeOpen] = useState(false);
  const [isLaunchRunOpen, setIsLaunchRunOpen] = useState(false);

  const queryClient = useQueryClient();

  // Fetch process modes
  const {
    data: processModes,
    isLoading: isLoadingProcessModes,
    error: processModesError,
  } = useQuery({
    queryKey: ["cvd", "process-modes"],
    queryFn: () => cvdApi.getProcessModes(),
  });

  // Fetch recipes
  const {
    data: recipes,
    isLoading: isLoadingRecipes,
    error: recipesError,
  } = useQuery({
    queryKey: ["cvd", "recipes", selectedProcessMode],
    queryFn: () => cvdApi.getRecipes({ process_mode_id: selectedProcessMode }),
    enabled: !!selectedProcessMode,
  });

  // Fetch runs
  const {
    data: runs,
    isLoading: isLoadingRuns,
    error: runsError,
  } = useQuery({
    queryKey: ["cvd", "runs"],
    queryFn: () => cvdApi.getRuns({ limit: 50 }),
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  // Create run mutation
  const createRunMutation = useMutation({
    mutationFn: (runData: any) => cvdApi.createRun(runData),
    onSuccess: () => {
      toast.success("Run created successfully");
      queryClient.invalidateQueries({ queryKey: ["cvd", "runs"] });
      setIsLaunchRunOpen(false);
    },
    onError: (error: any) => {
      toast.error(`Failed to create run: ${error.message}`);
    },
  });

  // Get status badge color
  const getStatusColor = (status: string) => {
    const colors: Record<string, string> = {
      QUEUED: "bg-gray-500",
      INITIALIZING: "bg-blue-500",
      PUMPING_DOWN: "bg-cyan-500",
      HEATING: "bg-orange-500",
      STABILIZING: "bg-yellow-500",
      PROCESSING: "bg-green-500",
      COOLING: "bg-purple-500",
      VENTING: "bg-indigo-500",
      COMPLETED: "bg-green-600",
      ABORTED: "bg-red-500",
      ERROR: "bg-red-600",
    };
    return colors[status] || "bg-gray-500";
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">CVD Workspace</h1>
          <p className="text-muted-foreground">
            Chemical Vapor Deposition - Process Management & Monitoring
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            <Settings className="mr-2 h-4 w-4" />
            Settings
          </Button>
          <Button variant="default" size="sm" onClick={() => setIsLaunchRunOpen(true)}>
            <Play className="mr-2 h-4 w-4" />
            Launch Run
          </Button>
        </div>
      </div>

      <Separator />

      {/* Main Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">
            <Activity className="mr-2 h-4 w-4" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="process-modes">
            <Database className="mr-2 h-4 w-4" />
            Process Modes
          </TabsTrigger>
          <TabsTrigger value="recipes">
            <Beaker className="mr-2 h-4 w-4" />
            Recipes
          </TabsTrigger>
          <TabsTrigger value="runs">
            <Play className="mr-2 h-4 w-4" />
            Runs
          </TabsTrigger>
          <TabsTrigger value="analytics">
            <TrendingUp className="mr-2 h-4 w-4" />
            Analytics
          </TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active Runs</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {runs?.filter((r: CVDRun) => r.status === "PROCESSING").length || 0}
                </div>
                <p className="text-xs text-muted-foreground">Currently processing</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Process Modes</CardTitle>
                <Database className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {processModes?.filter((pm: CVDProcessMode) => pm.is_active).length || 0}
                </div>
                <p className="text-xs text-muted-foreground">
                  {processModes?.length || 0} total
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Recipes</CardTitle>
                <Beaker className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {
                    processModes
                      ?.map((pm: CVDProcessMode) =>
                        cvdApi.getRecipes({ process_mode_id: pm.id })
                      )
                      .flat().length
                  }
                </div>
                <p className="text-xs text-muted-foreground">Across all modes</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Completed Today</CardTitle>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {
                    runs?.filter(
                      (r: CVDRun) =>
                        r.status === "COMPLETED" &&
                        new Date(r.created_at).toDateString() ===
                          new Date().toDateString()
                    ).length || 0
                  }
                </div>
                <p className="text-xs text-muted-foreground">Successful runs</p>
              </CardContent>
            </Card>
          </div>

          {/* Recent Runs */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Runs</CardTitle>
              <CardDescription>Latest CVD process runs</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Run ID</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Lot ID</TableHead>
                    <TableHead>Wafers</TableHead>
                    <TableHead>Duration</TableHead>
                    <TableHead>Created</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {runs?.slice(0, 5).map((run: CVDRun) => (
                    <TableRow key={run.id}>
                      <TableCell className="font-mono text-sm">
                        {run.id.slice(0, 8)}...
                      </TableCell>
                      <TableCell>
                        <Badge className={getStatusColor(run.status)}>
                          {run.status}
                        </Badge>
                      </TableCell>
                      <TableCell>{run.lot_id || "-"}</TableCell>
                      <TableCell>{run.wafer_ids.length}</TableCell>
                      <TableCell>
                        {run.duration_s
                          ? `${Math.round(run.duration_s / 60)}m`
                          : "-"}
                      </TableCell>
                      <TableCell>
                        {new Date(run.created_at).toLocaleString()}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Process Modes Tab */}
        <TabsContent value="process-modes" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>CVD Process Modes</CardTitle>
                  <CardDescription>
                    Available CVD variants and configurations
                  </CardDescription>
                </div>
                <Button size="sm">
                  <Plus className="mr-2 h-4 w-4" />
                  Add Process Mode
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {isLoadingProcessModes ? (
                <div className="flex items-center justify-center p-8">
                  <RefreshCw className="h-6 w-6 animate-spin" />
                </div>
              ) : (
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {processModes?.map((mode: CVDProcessMode) => (
                    <Card
                      key={mode.id}
                      className={`cursor-pointer transition-all hover:shadow-md ${
                        selectedProcessMode === mode.id
                          ? "ring-2 ring-primary"
                          : ""
                      }`}
                      onClick={() => setSelectedProcessMode(mode.id)}
                    >
                      <CardHeader>
                        <div className="flex items-start justify-between">
                          <div>
                            <CardTitle className="text-lg">
                              {mode.variant || `${mode.pressure_mode}-${mode.energy_mode}`}
                            </CardTitle>
                            <CardDescription className="text-xs">
                              {mode.description}
                            </CardDescription>
                          </div>
                          {mode.is_active ? (
                            <Badge variant="outline" className="bg-green-50">
                              Active
                            </Badge>
                          ) : (
                            <Badge variant="outline" className="bg-gray-50">
                              Inactive
                            </Badge>
                          )}
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Pressure:</span>
                            <span className="font-medium">{mode.pressure_mode}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Energy:</span>
                            <span className="font-medium">{mode.energy_mode}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Reactor:</span>
                            <span className="font-medium">{mode.reactor_type}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Chemistry:</span>
                            <span className="font-medium">{mode.chemistry_type}</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Recipes Tab */}
        <TabsContent value="recipes" className="space-y-4">
          {!selectedProcessMode ? (
            <Alert>
              <Beaker className="h-4 w-4" />
              <AlertTitle>Select a Process Mode</AlertTitle>
              <AlertDescription>
                Please select a process mode from the Process Modes tab to view and
                manage recipes.
              </AlertDescription>
            </Alert>
          ) : (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Recipes</CardTitle>
                    <CardDescription>
                      Manage process recipes for selected mode
                    </CardDescription>
                  </div>
                  <Button size="sm" onClick={() => setIsCreateRecipeOpen(true)}>
                    <Plus className="mr-2 h-4 w-4" />
                    Create Recipe
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {isLoadingRecipes ? (
                  <div className="flex items-center justify-center p-8">
                    <RefreshCw className="h-6 w-6 animate-spin" />
                  </div>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Target Thickness</TableHead>
                        <TableHead>Process Time</TableHead>
                        <TableHead>Runs</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {recipes?.map((recipe: CVDRecipe) => (
                        <TableRow key={recipe.id}>
                          <TableCell>
                            <div>
                              <div className="font-medium">{recipe.name}</div>
                              {recipe.description && (
                                <div className="text-xs text-muted-foreground">
                                  {recipe.description}
                                </div>
                              )}
                            </div>
                          </TableCell>
                          <TableCell>
                            {recipe.target_thickness_nm
                              ? `${recipe.target_thickness_nm} nm`
                              : "-"}
                          </TableCell>
                          <TableCell>
                            {Math.round(recipe.process_time_s / 60)} min
                          </TableCell>
                          <TableCell>{recipe.run_count}</TableCell>
                          <TableCell>
                            <div className="flex gap-1">
                              {recipe.is_golden && (
                                <Badge variant="outline" className="bg-yellow-50">
                                  Golden
                                </Badge>
                              )}
                              {recipe.is_baseline && (
                                <Badge variant="outline" className="bg-blue-50">
                                  Baseline
                                </Badge>
                              )}
                              {!recipe.is_active && (
                                <Badge variant="outline" className="bg-gray-50">
                                  Inactive
                                </Badge>
                              )}
                            </div>
                          </TableCell>
                          <TableCell>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => setSelectedRecipe(recipe.id)}
                            >
                              View
                              <ChevronRight className="ml-1 h-4 w-4" />
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Runs Tab */}
        <TabsContent value="runs" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Process Runs</CardTitle>
              <CardDescription>Monitor and manage CVD runs</CardDescription>
            </CardHeader>
            <CardContent>
              {/* Run list implementation */}
              <div className="text-muted-foreground">
                Run monitoring interface - implementation in progress
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="analytics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Process Analytics</CardTitle>
              <CardDescription>
                Statistical analysis and process capability
              </CardDescription>
            </CardHeader>
            <CardContent>
              {/* Analytics implementation */}
              <div className="text-muted-foreground">
                Analytics dashboard - implementation in progress
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Launch Run Dialog */}
      <Dialog open={isLaunchRunOpen} onOpenChange={setIsLaunchRunOpen}>
        <DialogContent className="sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle>Launch CVD Run</DialogTitle>
            <DialogDescription>
              Configure and launch a new CVD process run
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            {/* Run configuration form - simplified */}
            <div className="text-sm text-muted-foreground">
              Run configuration interface - implementation in progress
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsLaunchRunOpen(false)}>
              Cancel
            </Button>
            <Button onClick={() => {}}>
              <Play className="mr-2 h-4 w-4" />
              Launch
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
