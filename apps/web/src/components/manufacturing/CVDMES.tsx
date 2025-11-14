/**
 * CVD Manufacturing Execution System
 * Chemical Vapor Deposition process control and monitoring
 */

"use client"

import React, { useState, useEffect } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Button } from '@/components/ui/button'
import { Flame, Beaker, PlayCircle, BarChart3, Settings, RefreshCw, CheckCircle, Clock, AlertCircle } from 'lucide-react'
import { RoleDisplay } from '@/components/shared/RBACActionButtons'
import {
  cvdApi,
  type CVDProcessMode,
  type CVDRecipe,
  type CVDRun,
  type CVDDashboard
} from '@/lib/api/cvd'

const MOCK_ORG_ID = '173cf517-1566-4906-a2db-0ce023d7b378' // Demo organization UUID

export const CVDMES: React.FC = () => {
  const [processModes, setProcessModes] = useState<CVDProcessMode[]>([])
  const [recipes, setRecipes] = useState<CVDRecipe[]>([])
  const [runs, setRuns] = useState<CVDRun[]>([])
  const [dashboard, setDashboard] = useState<CVDDashboard | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      const [modesData, recipesData, runsData] = await Promise.all([
        cvdApi.getProcessModes({ org_id: MOCK_ORG_ID }),
        cvdApi.getRecipes({ org_id: MOCK_ORG_ID, limit: 50 }),
        cvdApi.getRuns({ org_id: MOCK_ORG_ID, limit: 50 }),
      ])
      setProcessModes(modesData)
      setRecipes(recipesData)
      setRuns(runsData)
      setDashboard(null)
    } catch (error) {
      console.error('Error loading CVD data:', error)
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'completed': return 'default'
      case 'running': return 'secondary'
      case 'failed': return 'destructive'
      case 'aborted': return 'outline'
      case 'queued': return 'outline'
      default: return 'outline'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'running': return <Clock className="w-4 h-4 text-blue-500 animate-pulse" />
      case 'failed': return <AlertCircle className="w-4 h-4 text-red-500" />
      case 'aborted': return <AlertCircle className="w-4 h-4 text-gray-500" />
      case 'queued': return <Clock className="w-4 h-4 text-gray-400" />
      default: return null
    }
  }

  const formatPressureMode = (mode: string) => {
    const modes: Record<string, string> = {
      'LPCVD': 'Low Pressure',
      'APCVD': 'Atmospheric Pressure',
      'PECVD': 'Plasma Enhanced',
      'UHVCVD': 'Ultra-High Vacuum',
      'MOCVD': 'Metal-Organic'
    }
    return modes[mode] || mode
  }

  const formatEnergyMode = (mode: string) => {
    const modes: Record<string, string> = {
      'THERMAL': 'Thermal',
      'PLASMA': 'Plasma',
      'PHOTO': 'Photo-Assisted',
      'LASER': 'Laser-Assisted'
    }
    return modes[mode] || mode
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold">CVD Manufacturing</h1>
          <p className="text-muted-foreground mt-1">
            Chemical Vapor Deposition process control and monitoring
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <div className="flex items-center gap-2">
            <RoleDisplay showPermissions={false} />
            <Button onClick={loadData} variant="outline" size="sm" disabled={loading}>
              <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
          {dashboard && (
            <div className="flex gap-2">
              <Badge variant="outline" className="text-xs">{dashboard.total_recipes} Recipes</Badge>
              <Badge variant="outline" className="text-xs">{dashboard.total_runs} Total Runs</Badge>
            </div>
          )}
        </div>
      </div>

      <Separator />

      {/* Dashboard Cards */}
      {loading ? (
        <div className="text-center py-12 text-muted-foreground">Loading CVD data...</div>
      ) : !dashboard ? (
        <div className="text-center py-12 text-muted-foreground">
          No CVD data available. Configure process modes and recipes to start.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Settings className="w-4 h-4 text-blue-500" />
                Process Modes
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{dashboard.active_process_modes}</div>
              <p className="text-xs text-muted-foreground">Active configurations</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Beaker className="w-4 h-4 text-purple-500" />
                Recipes
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{dashboard.total_recipes}</div>
              <p className="text-xs text-muted-foreground">{dashboard.golden_recipes} golden, {dashboard.baseline_recipes} baseline</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <PlayCircle className="w-4 h-4 text-green-500" />
                Active Runs
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">{dashboard.running_runs}</div>
              <p className="text-xs text-muted-foreground">{dashboard.queued_runs} queued</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-orange-500" />
                Completion Rate
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{dashboard.completion_rate?.toFixed(1) || 0}%</div>
              <p className="text-xs text-muted-foreground">Last 30 days</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Main Tabs */}
      <Tabs defaultValue="recipes" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="recipes">
            <Beaker className="w-4 h-4 mr-2" />
            Recipes
          </TabsTrigger>
          <TabsTrigger value="runs">
            <PlayCircle className="w-4 h-4 mr-2" />
            Runs
          </TabsTrigger>
          <TabsTrigger value="modes">
            <Settings className="w-4 h-4 mr-2" />
            Process Modes
          </TabsTrigger>
          <TabsTrigger value="monitor">
            <Flame className="w-4 h-4 mr-2" />
            Live Monitor
          </TabsTrigger>
          <TabsTrigger value="results">
            <BarChart3 className="w-4 h-4 mr-2" />
            Results
          </TabsTrigger>
        </TabsList>

        {/* Recipes Tab */}
        <TabsContent value="recipes" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>CVD Recipes</CardTitle>
              <CardDescription>Process recipes with temperature, gas flows, and plasma settings</CardDescription>
            </CardHeader>
            <CardContent>
              {recipes.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground">
                  No CVD recipes found. Create a recipe to start processing.
                </div>
              ) : (
                <div className="space-y-3">
                  {recipes.map((recipe) => (
                    <div key={recipe.id} className="flex items-center justify-between p-4 border rounded-lg hover:bg-accent">
                      <div className="flex items-center gap-3 flex-1">
                        <Beaker className="w-5 h-5 text-purple-500" />
                        <div>
                          <div className="flex items-center gap-2">
                            <h4 className="font-medium">{recipe.name}</h4>
                            {recipe.is_golden && <Badge variant="default" className="text-xs">GOLDEN</Badge>}
                            {recipe.is_baseline && <Badge variant="secondary" className="text-xs">BASELINE</Badge>}
                          </div>
                          <p className="text-sm text-muted-foreground">
                            {recipe.process_time_s}s process • {recipe.run_count} runs
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right text-sm">
                          <p className="font-medium">{recipe.recipe_steps?.length || 0} steps</p>
                          <p className="text-xs text-muted-foreground">Multi-step process</p>
                        </div>
                        <Button size="sm" variant="outline">
                          <PlayCircle className="w-4 h-4 mr-1" />
                          Run
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Runs Tab */}
        <TabsContent value="runs" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>CVD Process Runs</CardTitle>
              <CardDescription>Active and historical CVD process executions</CardDescription>
            </CardHeader>
            <CardContent>
              {runs.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground">
                  No CVD runs found. Execute a recipe to start processing wafers.
                </div>
              ) : (
                <div className="space-y-3">
                  {runs.map((run) => (
                    <div key={run.id} className="flex items-center justify-between p-4 border rounded-lg hover:bg-accent">
                      <div className="flex items-center gap-3 flex-1">
                        {getStatusIcon(run.status)}
                        <div>
                          <h4 className="font-medium">
                            {run.recipe?.name || `Run ${run.id.substring(0, 8)}`}
                          </h4>
                          <p className="text-sm text-muted-foreground">
                            {run.wafer_ids?.length || 0} wafers
                            {run.actual_temperature_c && ` • ${run.actual_temperature_c}°C`}
                            {run.actual_pressure_pa && ` • ${(run.actual_pressure_pa / 133.322).toFixed(2)} Torr`}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        {run.job_progress !== undefined && run.status?.toLowerCase() === 'running' && (
                          <div className="text-right">
                            <p className="text-sm font-medium">{run.job_progress}% complete</p>
                            <div className="w-24 h-2 bg-gray-200 rounded-full mt-1">
                              <div
                                className="h-2 bg-blue-500 rounded-full transition-all"
                                style={{ width: `${run.job_progress}%` }}
                              />
                            </div>
                          </div>
                        )}
                        {run.start_time && (
                          <div className="text-right text-sm">
                            <p className="text-xs text-muted-foreground">Started</p>
                            <p className="font-medium">{new Date(run.start_time).toLocaleString()}</p>
                          </div>
                        )}
                        <Badge variant={getStatusColor(run.status)}>
                          {run.status?.toUpperCase() || 'UNKNOWN'}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Process Modes Tab */}
        <TabsContent value="modes" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>CVD Process Modes</CardTitle>
              <CardDescription>Available CVD configurations and reactor settings</CardDescription>
            </CardHeader>
            <CardContent>
              {processModes.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground">
                  No process modes configured.
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {processModes.map((mode) => (
                    <div key={mode.id} className="p-4 border rounded-lg space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Settings className="w-5 h-5 text-blue-500" />
                          <h4 className="font-medium">{mode.pressure_mode}</h4>
                        </div>
                        <Badge variant={mode.is_active ? 'default' : 'outline'}>
                          {mode.is_active ? 'Active' : 'Inactive'}
                        </Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                          <span className="text-muted-foreground">Pressure:</span>
                          <p className="font-medium">{formatPressureMode(mode.pressure_mode)}</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Energy:</span>
                          <p className="font-medium">{formatEnergyMode(mode.energy_mode)}</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Reactor:</span>
                          <p className="font-medium">{mode.reactor_type}</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Chemistry:</span>
                          <p className="font-medium">{mode.chemistry_type}</p>
                        </div>
                      </div>
                      {mode.materials && mode.materials.length > 0 && (
                        <div>
                          <span className="text-xs text-muted-foreground">Materials:</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {mode.materials.map((material, idx) => (
                              <Badge key={idx} variant="outline" className="text-xs">
                                {material}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                      <div className="text-xs text-muted-foreground">
                        Temp: {mode.temperature_range_c?.min}–{mode.temperature_range_c?.max}°C
                        {' • '}
                        Pressure: {(mode.pressure_range_pa?.min / 133.322).toFixed(1)}–{(mode.pressure_range_pa?.max / 133.322).toFixed(1)} Torr
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Live Monitor Tab */}
        <TabsContent value="monitor">
          <Card>
            <CardHeader>
              <CardTitle>Live Process Monitor</CardTitle>
              <CardDescription>Real-time CVD process monitoring and control</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {runs.filter(r => r.status?.toLowerCase() === 'running').length === 0 ? (
                  <div className="text-center py-12 text-muted-foreground">
                    No active CVD processes running.
                  </div>
                ) : (
                  runs.filter(r => r.status?.toLowerCase() === 'running').map((run) => (
                    <div key={run.id} className="p-4 border rounded-lg space-y-3">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium flex items-center gap-2">
                          <Clock className="w-4 h-4 text-blue-500 animate-pulse" />
                          {run.recipe?.name || `Run ${run.id.substring(0, 8)}`}
                        </h4>
                        <Badge variant="secondary">RUNNING</Badge>
                      </div>
                      {run.job_progress !== undefined && (
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span className="text-muted-foreground">Progress</span>
                            <span className="font-medium">{run.job_progress}%</span>
                          </div>
                          <div className="w-full h-3 bg-gray-200 rounded-full">
                            <div
                              className="h-3 bg-blue-500 rounded-full transition-all"
                              style={{ width: `${run.job_progress}%` }}
                            />
                          </div>
                        </div>
                      )}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                        {run.actual_temperature_c && (
                          <div className="p-2 bg-accent rounded">
                            <span className="text-muted-foreground">Temperature</span>
                            <p className="font-bold text-lg">{run.actual_temperature_c}°C</p>
                          </div>
                        )}
                        {run.actual_pressure_pa && (
                          <div className="p-2 bg-accent rounded">
                            <span className="text-muted-foreground">Pressure</span>
                            <p className="font-bold text-lg">{(run.actual_pressure_pa / 133.322).toFixed(2)} Torr</p>
                          </div>
                        )}
                        <div className="p-2 bg-accent rounded">
                          <span className="text-muted-foreground">Wafers</span>
                          <p className="font-bold text-lg">{run.wafer_ids?.length || 0}</p>
                        </div>
                        {run.process_mode && (
                          <div className="p-2 bg-accent rounded">
                            <span className="text-muted-foreground">Mode</span>
                            <p className="font-bold text-sm">{run.process_mode.pressure_mode}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value="results">
          <Card>
            <CardHeader>
              <CardTitle>Process Results & Analytics</CardTitle>
              <CardDescription>CVD process outcomes and statistical analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {dashboard && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-2">Run Statistics</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Total Runs:</span>
                          <span className="font-bold">{dashboard.total_runs}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Completed:</span>
                          <span className="font-bold text-green-600">{dashboard.completed_runs}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Failed:</span>
                          <span className="font-bold text-red-600">{dashboard.failed_runs}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Completion Rate:</span>
                          <span className="font-bold">{dashboard.completion_rate?.toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>

                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-2">Recipe Performance</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Golden Recipes:</span>
                          <span className="font-bold">{dashboard.golden_recipes}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Baseline Recipes:</span>
                          <span className="font-bold">{dashboard.baseline_recipes}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Total Recipes:</span>
                          <span className="font-bold">{dashboard.total_recipes}</span>
                        </div>
                      </div>
                    </div>

                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-2">Equipment Status</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Process Modes:</span>
                          <span className="font-bold">{dashboard.active_process_modes}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Running:</span>
                          <span className="font-bold text-blue-600">{dashboard.running_runs}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Queued:</span>
                          <span className="font-bold text-gray-600">{dashboard.queued_runs}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-3">Recent Completed Runs</h4>
                  <div className="space-y-2">
                    {runs
                      .filter(r => r.status?.toLowerCase() === 'completed')
                      .slice(0, 5)
                      .map((run) => (
                        <div key={run.id} className="flex items-center justify-between p-2 border rounded text-sm">
                          <div className="flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-green-500" />
                            <span className="font-medium">{run.recipe?.name || `Run ${run.id.substring(0, 8)}`}</span>
                          </div>
                          <div className="flex items-center gap-4">
                            <span className="text-muted-foreground">{run.wafer_ids?.length || 0} wafers</span>
                            {run.end_time && (
                              <span className="text-xs text-muted-foreground">
                                {new Date(run.end_time).toLocaleDateString()}
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    {runs.filter(r => r.status?.toLowerCase() === 'completed').length === 0 && (
                      <p className="text-center text-muted-foreground py-4">No completed runs yet.</p>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default CVDMES
