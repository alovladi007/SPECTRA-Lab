/**
 * Diffusion Manufacturing Execution System
 * Dopant diffusion process control and monitoring
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
  diffusionApi,
  type DiffusionFurnace,
  type DiffusionRecipe,
  type DiffusionRun,
  type DiffusionDashboard
} from '@/lib/api/diffusion'

const MOCK_ORG_ID = '00000000-0000-0000-0000-000000000001' // Demo organization UUID

export const DiffusionMES: React.FC = () => {
  const [furnaces, setFurnaces] = useState<DiffusionFurnace[]>([])
  const [recipes, setRecipes] = useState<DiffusionRecipe[]>([])
  const [runs, setRuns] = useState<DiffusionRun[]>([])
  const [dashboard, setDashboard] = useState<DiffusionDashboard | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      const [furnacesData, recipesData, runsData, dashboardData] = await Promise.all([
        diffusionApi.getFurnaces({ org_id: MOCK_ORG_ID, is_active: true }),
        diffusionApi.getRecipes({ org_id: MOCK_ORG_ID, limit: 50 }),
        diffusionApi.getRuns({ org_id: MOCK_ORG_ID, limit: 50 }),
        diffusionApi.getDashboard({ org_id: MOCK_ORG_ID }),
      ])
      setFurnaces(furnacesData)
      setRecipes(recipesData)
      setRuns(runsData)
      setDashboard(dashboardData)
    } catch (error) {
      console.error('Error loading diffusion data:', error)
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

  const getRecipeStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'approved': return 'default'
      case 'draft': return 'secondary'
      case 'deprecated': return 'outline'
      default: return 'outline'
    }
  }

  const formatDiffusionType = (type: string) => {
    const types: Record<string, string> = {
      'PREDEPOSITION': 'Pre-Deposition',
      'DRIVE_IN': 'Drive-In',
      'CONTINUOUS': 'Continuous'
    }
    return types[type] || type
  }

  const formatDopant = (dopant: string) => {
    const dopants: Record<string, string> = {
      'BORON': 'Boron (p-type)',
      'PHOSPHORUS': 'Phosphorus (n-type)',
      'ARSENIC': 'Arsenic (n-type)',
      'ANTIMONY': 'Antimony (n-type)'
    }
    return dopants[dopant] || dopant
  }

  const formatFurnaceType = (type: string) => {
    const types: Record<string, string> = {
      'TUBE': 'Horizontal Tube',
      'BATCH': 'Batch',
      'VERTICAL': 'Vertical',
      'RTP_HYBRID': 'RTP Hybrid'
    }
    return types[type] || type
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold">Diffusion Manufacturing</h1>
          <p className="text-muted-foreground mt-1">
            Dopant diffusion process control and monitoring
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
        <div className="text-center py-12 text-muted-foreground">Loading diffusion data...</div>
      ) : !dashboard ? (
        <div className="text-center py-12 text-muted-foreground">
          No diffusion data available. Configure furnaces and recipes to start.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Flame className="w-4 h-4 text-orange-500" />
                Furnaces
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{dashboard.active_furnaces}</div>
              <p className="text-xs text-muted-foreground">Active furnaces</p>
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
              <p className="text-xs text-muted-foreground">{dashboard.approved_recipes} approved</p>
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
                <BarChart3 className="w-4 h-4 text-blue-500" />
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
          <TabsTrigger value="furnaces">
            <Flame className="w-4 h-4 mr-2" />
            Furnaces
          </TabsTrigger>
          <TabsTrigger value="monitor">
            <Settings className="w-4 h-4 mr-2" />
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
              <CardTitle>Diffusion Recipes</CardTitle>
              <CardDescription>Process recipes with dopant, temperature profiles, and ambient settings</CardDescription>
            </CardHeader>
            <CardContent>
              {recipes.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground">
                  No diffusion recipes found. Create a recipe to start processing.
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
                            <Badge variant={getRecipeStatusColor(recipe.status)}>
                              {recipe.status?.toUpperCase()}
                            </Badge>
                          </div>
                          <p className="text-sm text-muted-foreground">
                            {formatDiffusionType(recipe.diffusion_type)} • {formatDopant(recipe.dopant)}
                            {recipe.dopant_source && ` • ${recipe.dopant_source} source`}
                            {recipe.ambient_gas && ` • ${recipe.ambient_gas} ambient`}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right text-sm">
                          <p className="font-medium">{recipe.total_time_min} min</p>
                          <p className="text-xs text-muted-foreground">
                            {recipe.recipe_steps?.length || 0} steps • {recipe.run_count} runs
                          </p>
                        </div>
                        {recipe.target_junction_depth_nm && (
                          <div className="text-right text-xs text-muted-foreground">
                            <p>Target Xj: {recipe.target_junction_depth_nm} nm</p>
                            {recipe.target_sheet_resistance_ohm_sq && (
                              <p>Rs: {recipe.target_sheet_resistance_ohm_sq} Ω/sq</p>
                            )}
                          </div>
                        )}
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
              <CardTitle>Diffusion Process Runs</CardTitle>
              <CardDescription>Active and historical diffusion process executions</CardDescription>
            </CardHeader>
            <CardContent>
              {runs.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground">
                  No diffusion runs found. Execute a recipe to start processing wafers.
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
                            {run.furnace && ` • ${run.furnace.name}`}
                            {run.actual_peak_temp_c && ` • ${run.actual_peak_temp_c}°C peak`}
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
                        {run.actual_total_time_min && (
                          <div className="text-right text-sm">
                            <p className="text-xs text-muted-foreground">Actual time</p>
                            <p className="font-medium">{run.actual_total_time_min} min</p>
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

        {/* Furnaces Tab */}
        <TabsContent value="furnaces" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Diffusion Furnaces</CardTitle>
              <CardDescription>Available diffusion furnaces and their capabilities</CardDescription>
            </CardHeader>
            <CardContent>
              {furnaces.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground">
                  No furnaces configured.
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {furnaces.map((furnace) => (
                    <div key={furnace.id} className="p-4 border rounded-lg space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Flame className="w-5 h-5 text-orange-500" />
                          <h4 className="font-medium">{furnace.name}</h4>
                        </div>
                        <Badge variant={furnace.is_active ? 'default' : 'outline'}>
                          {furnace.is_active ? 'Active' : 'Inactive'}
                        </Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                          <span className="text-muted-foreground">Type:</span>
                          <p className="font-medium">{formatFurnaceType(furnace.furnace_type)}</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Temp Zones:</span>
                          <p className="font-medium">{furnace.num_temperature_zones}</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Capacity:</span>
                          <p className="font-medium">{furnace.max_wafer_capacity} wafers</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Max Temp:</span>
                          <p className="font-medium">{furnace.max_temperature_c}°C</p>
                        </div>
                      </div>
                      {furnace.supported_dopants && furnace.supported_dopants.length > 0 && (
                        <div>
                          <span className="text-xs text-muted-foreground">Supported Dopants:</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {furnace.supported_dopants.map((dopant, idx) => (
                              <Badge key={idx} variant="outline" className="text-xs">
                                {dopant}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                      {furnace.supported_sources && furnace.supported_sources.length > 0 && (
                        <div>
                          <span className="text-xs text-muted-foreground">Dopant Sources:</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {furnace.supported_sources.map((source, idx) => (
                              <Badge key={idx} variant="outline" className="text-xs">
                                {source}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                      {furnace.supported_ambients && furnace.supported_ambients.length > 0 && (
                        <div>
                          <span className="text-xs text-muted-foreground">Ambient Gases:</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {furnace.supported_ambients.map((ambient, idx) => (
                              <Badge key={idx} variant="outline" className="text-xs">
                                {ambient}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
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
              <CardDescription>Real-time diffusion process monitoring and control</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {runs.filter(r => r.status?.toLowerCase() === 'running').length === 0 ? (
                  <div className="text-center py-12 text-muted-foreground">
                    No active diffusion processes running.
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
                        {run.actual_peak_temp_c && (
                          <div className="p-2 bg-accent rounded">
                            <span className="text-muted-foreground">Peak Temp</span>
                            <p className="font-bold text-lg">{run.actual_peak_temp_c}°C</p>
                          </div>
                        )}
                        {run.actual_total_time_min && (
                          <div className="p-2 bg-accent rounded">
                            <span className="text-muted-foreground">Process Time</span>
                            <p className="font-bold text-lg">{run.actual_total_time_min} min</p>
                          </div>
                        )}
                        <div className="p-2 bg-accent rounded">
                          <span className="text-muted-foreground">Wafers</span>
                          <p className="font-bold text-lg">{run.wafer_ids?.length || 0}</p>
                        </div>
                        {run.furnace && (
                          <div className="p-2 bg-accent rounded">
                            <span className="text-muted-foreground">Furnace</span>
                            <p className="font-bold text-sm">{run.furnace.name}</p>
                          </div>
                        )}
                      </div>
                      {run.recipe && (
                        <div className="text-xs text-muted-foreground">
                          {formatDiffusionType(run.recipe.diffusion_type)} • {formatDopant(run.recipe.dopant)}
                          {run.recipe.dopant_source && ` • ${run.recipe.dopant_source} source`}
                        </div>
                      )}
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
              <CardDescription>Diffusion process outcomes and statistical analysis</CardDescription>
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
                          <span className="text-muted-foreground">Approved Recipes:</span>
                          <span className="font-bold">{dashboard.approved_recipes}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Draft Recipes:</span>
                          <span className="font-bold">{dashboard.draft_recipes}</span>
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
                          <span className="text-muted-foreground">Active Furnaces:</span>
                          <span className="font-bold">{dashboard.active_furnaces}</span>
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
                            {run.actual_total_time_min && (
                              <span className="text-xs text-muted-foreground">
                                {run.actual_total_time_min} min
                              </span>
                            )}
                            {run.recipe?.dopant && (
                              <Badge variant="outline" className="text-xs">
                                {run.recipe.dopant}
                              </Badge>
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

export default DiffusionMES
