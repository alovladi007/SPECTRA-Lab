"use client";

/**
 * CVD Runs List Page
 *
 * Features:
 * - List of all CVD runs with status
 * - Filter by status, recipe, tool, date range
 * - Real-time status updates
 * - Quick metrics preview
 */

import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { AdhesionBadge } from "@/components/cvd/metrics/AdhesionChip";
import {
  Play,
  Pause,
  CheckCircle,
  XCircle,
  Clock,
  AlertCircle,
  Search,
  Filter,
  TrendingUp,
  TrendingDown,
} from "lucide-react";
import Link from "next/link";
import { Progress } from "@/components/ui/progress";

interface Run {
  id: string;
  run_id: string;
  recipe_name: string;
  tool: string;
  status: "running" | "completed" | "failed" | "pending" | "cancelled";
  progress?: number;
  started_at: string;
  completed_at?: string;
  current_thickness_nm?: number;
  target_thickness_nm: number;
  current_stress_mpa?: number;
  adhesion_score?: number;
  alerts_count?: number;
}

export default function CVDRunsPage() {
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");

  // Fetch runs list
  const { data: runs, isLoading } = useQuery({
    queryKey: ["cvd-runs", statusFilter],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (statusFilter !== "all") params.set("status", statusFilter);

      const response = await fetch(`http://localhost:8001/api/cvd/runs?${params}`);
      if (!response.ok) throw new Error("Failed to fetch runs");
      return response.json();
    },
    refetchInterval: 5000, // Refresh every 5s for real-time updates
  });

  // Mock data
  const mockRuns: Run[] = runs || [
    {
      id: "1",
      run_id: "CVD_RUN_20251114_103045",
      recipe_name: "Si3N4 Standard",
      tool: "CVD-01",
      status: "running",
      progress: 67,
      started_at: "2025-11-14T10:30:45",
      current_thickness_nm: 67.5,
      target_thickness_nm: 100,
      current_stress_mpa: -185,
    },
    {
      id: "2",
      run_id: "CVD_RUN_20251114_093022",
      recipe_name: "TiN Barrier Layer",
      tool: "CVD-02",
      status: "completed",
      progress: 100,
      started_at: "2025-11-14T09:30:22",
      completed_at: "2025-11-14T10:15:30",
      current_thickness_nm: 20.3,
      target_thickness_nm: 20,
      current_stress_mpa: -105,
      adhesion_score: 91,
    },
    {
      id: "3",
      run_id: "CVD_RUN_20251114_082015",
      recipe_name: "W Fill",
      tool: "CVD-03",
      status: "completed",
      progress: 100,
      started_at: "2025-11-14T08:20:15",
      completed_at: "2025-11-14T09:05:45",
      current_thickness_nm: 198.7,
      target_thickness_nm: 200,
      current_stress_mpa: 142,
      adhesion_score: 74,
      alerts_count: 1,
    },
    {
      id: "4",
      run_id: "CVD_RUN_20251113_163012",
      recipe_name: "Si3N4 Standard",
      tool: "CVD-01",
      status: "failed",
      progress: 42,
      started_at: "2025-11-13T16:30:12",
      current_thickness_nm: 42.8,
      target_thickness_nm: 100,
      alerts_count: 3,
    },
  ];

  // Filter runs
  const filteredRuns = mockRuns.filter(run => {
    const matchesStatus = statusFilter === "all" || run.status === statusFilter;
    const matchesSearch = searchQuery === "" ||
      run.run_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      run.recipe_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      run.tool.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesStatus && matchesSearch;
  });

  // Status badge configuration
  const getStatusBadge = (status: Run["status"]) => {
    const configs = {
      running: { variant: "default" as const, icon: Play, className: "bg-blue-100 text-blue-800" },
      completed: { variant: "default" as const, icon: CheckCircle, className: "bg-green-100 text-green-800" },
      failed: { variant: "destructive" as const, icon: XCircle, className: "bg-red-100 text-red-800" },
      pending: { variant: "secondary" as const, icon: Clock, className: "bg-gray-100 text-gray-800" },
      cancelled: { variant: "secondary" as const, icon: Pause, className: "bg-gray-100 text-gray-800" },
    };

    const config = configs[status];
    const Icon = config.icon;

    return (
      <Badge variant={config.variant} className={config.className}>
        <Icon className="h-3 w-3 mr-1" />
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </Badge>
    );
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">CVD Runs</h1>
          <p className="text-muted-foreground mt-1">
            Monitor and review all deposition runs
          </p>
        </div>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search by run ID, recipe, or tool..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-48">
                <Filter className="h-4 w-4 mr-2" />
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="running">Running</SelectItem>
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="failed">Failed</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Runs Table */}
      <Card>
        <CardHeader>
          <CardTitle>Runs ({filteredRuns.length})</CardTitle>
          <CardDescription>
            Real-time status and metrics for all runs
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Run ID</TableHead>
                <TableHead>Recipe</TableHead>
                <TableHead>Tool</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Progress</TableHead>
                <TableHead>Thickness</TableHead>
                <TableHead>Stress</TableHead>
                <TableHead>Adhesion</TableHead>
                <TableHead>Started</TableHead>
                <TableHead>Alerts</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredRuns.map((run) => {
                const thicknessDeviation = run.current_thickness_nm
                  ? ((run.current_thickness_nm - run.target_thickness_nm) / run.target_thickness_nm) * 100
                  : 0;

                const isThicknessOnTarget = Math.abs(thicknessDeviation) <= 5;

                return (
                  <TableRow key={run.id} className="cursor-pointer hover:bg-muted/50">
                    <TableCell className="font-mono text-xs">
                      <Link href={`/cvd/runs/${run.id}`} className="hover:underline">
                        {run.run_id}
                      </Link>
                    </TableCell>
                    <TableCell>{run.recipe_name}</TableCell>
                    <TableCell>
                      <Badge variant="outline">{run.tool}</Badge>
                    </TableCell>
                    <TableCell>{getStatusBadge(run.status)}</TableCell>
                    <TableCell>
                      {run.status === "running" ? (
                        <div className="space-y-1">
                          <Progress value={run.progress} className="w-20" />
                          <div className="text-xs text-muted-foreground">{run.progress}%</div>
                        </div>
                      ) : (
                        <span className="text-sm text-muted-foreground">
                          {run.progress === 100 ? "Complete" : `${run.progress}%`}
                        </span>
                      )}
                    </TableCell>
                    <TableCell>
                      {run.current_thickness_nm !== undefined ? (
                        <div className="text-sm">
                          <div className="flex items-center gap-1">
                            <span className="font-medium">{run.current_thickness_nm.toFixed(1)} nm</span>
                            {isThicknessOnTarget ? (
                              <TrendingUp className="h-3 w-3 text-green-600" />
                            ) : (
                              <TrendingDown className="h-3 w-3 text-orange-600" />
                            )}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Target: {run.target_thickness_nm} nm
                          </div>
                        </div>
                      ) : (
                        <span className="text-sm text-muted-foreground">-</span>
                      )}
                    </TableCell>
                    <TableCell>
                      {run.current_stress_mpa !== undefined ? (
                        <div className="text-sm">
                          <span className={
                            Math.abs(run.current_stress_mpa) > 400
                              ? "text-red-600 font-medium"
                              : "font-medium"
                          }>
                            {run.current_stress_mpa.toFixed(0)} MPa
                          </span>
                        </div>
                      ) : (
                        <span className="text-sm text-muted-foreground">-</span>
                      )}
                    </TableCell>
                    <TableCell>
                      {run.adhesion_score !== undefined ? (
                        <AdhesionBadge score={run.adhesion_score} />
                      ) : (
                        <span className="text-sm text-muted-foreground">Pending</span>
                      )}
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {new Date(run.started_at).toLocaleString()}
                      {run.completed_at && (
                        <div className="text-[10px]">
                          Duration: {Math.round((new Date(run.completed_at).getTime() - new Date(run.started_at).getTime()) / 60000)} min
                        </div>
                      )}
                    </TableCell>
                    <TableCell>
                      {run.alerts_count && run.alerts_count > 0 ? (
                        <Badge variant="destructive" className="text-xs">
                          <AlertCircle className="h-3 w-3 mr-1" />
                          {run.alerts_count}
                        </Badge>
                      ) : (
                        <span className="text-sm text-muted-foreground">-</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      <Button variant="outline" size="sm" asChild>
                        <Link href={`/cvd/runs/${run.id}`}>View Details</Link>
                      </Button>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
