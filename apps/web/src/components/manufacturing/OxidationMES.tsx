/**
 * Oxidation Manufacturing Execution System (MES)
 * Production-grade thermal oxidation control and monitoring
 */

"use client"

import React, { useState, useEffect } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Button } from '@/components/ui/button'
import { Layers, Activity, BarChart3, Settings, PlayCircle, Database, RefreshCw } from 'lucide-react'
import { RoleDisplay } from '@/components/shared/RBACActionButtons'
import CalibrationBadges from '@/components/shared/CalibrationBadges'
import { oxidationApi, type OxidationFurnace, type OxidationRecipe, type OxidationRun } from '@/lib/api/oxidation'

const MOCK_ORG_ID = '00000000-0000-0000-0000-000000000001' // Demo organization UUID

export const OxidationMES: React.FC = () => {
  const [activeRun, setActiveRun] = useState<string | null>(null)
  const [furnaces, setFurnaces] = useState<OxidationFurnace[]>([])
  const [recipes, setRecipes] = useState<OxidationRecipe[]>([])
  const [runs, setRuns] = useState<OxidationRun[]>([])
  const [loading, setLoading] = useState(true)

  // Load data on mount
  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      const [furnacesData, recipesData, runsData] = await Promise.all([
        oxidationApi.getFurnaces({ org_id: MOCK_ORG_ID }),
        oxidationApi.getRecipes({ org_id: MOCK_ORG_ID }),
        oxidationApi.getRuns({ org_id: MOCK_ORG_ID, limit: 20 }),
      ])
      setFurnaces(furnacesData)
      setRecipes(recipesData)
      setRuns(runsData)
    } catch (error) {
      console.error('Error loading oxidation data:', error)
    } finally {
      setLoading(false)
    }
  }

  const getStatusBadge = (status: string) => {
    const variants: Record<string, 'default' | 'destructive' | 'outline' | 'secondary'> = {
      QUEUED: 'outline',
      RUNNING: 'destructive',
      COMPLETED: 'secondary',
      FAILED: 'destructive',
      ABORTED: 'destructive',
      APPROVED: 'default',
      DRAFT: 'secondary',
    }
    return <Badge variant={variants[status] || 'outline'}>{status}</Badge>
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold">Thermal Oxidation MES</h1>
          <p className="text-muted-foreground mt-1">
            Production oxidation furnace control and process monitoring
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
          {furnaces.length > 0 && (
            <CalibrationBadges equipmentIds={furnaces.slice(0, 2).map(f => f.id)} />
          )}
        </div>
      </div>

      <Separator />

      {/* Equipment Status Overview */}
      {loading ? (
        <div className="text-center py-12 text-muted-foreground">Loading...</div>
      ) : furnaces.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          No furnaces found. Create your first oxidation furnace to get started.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {furnaces.slice(0, 3).map((furnace) => (
            <Card key={furnace.id}>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium">{furnace.name}</CardTitle>
                  <Badge variant={furnace.is_active ? 'default' : 'outline'}>
                    {furnace.is_active ? 'ACTIVE' : 'INACTIVE'}
                  </Badge>
                </div>
                <CardDescription className="text-xs">{furnace.manufacturer} {furnace.model}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Max Temperature</span>
                    <span className="font-medium">{furnace.max_temperature_c}°C</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Type</span>
                    <span className="font-medium text-xs">{furnace.furnace_type}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Capacity</span>
                    <span className="font-medium text-xs">{furnace.max_wafer_capacity} wafers</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Main Tabs */}
      <Tabs defaultValue="recipes" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="recipes">
            <Settings className="w-4 h-4 mr-2" />
            Recipes
          </TabsTrigger>
          <TabsTrigger value="runs">
            <Database className="w-4 h-4 mr-2" />
            Runs
          </TabsTrigger>
          <TabsTrigger value="monitor" disabled={!activeRun}>
            <Activity className="w-4 h-4 mr-2" />
            Live Monitor
            {activeRun && <Badge variant="destructive" className="ml-2 animate-pulse">LIVE</Badge>}
          </TabsTrigger>
          <TabsTrigger value="results">
            <BarChart3 className="w-4 h-4 mr-2" />
            Results
          </TabsTrigger>
          <TabsTrigger value="furnaces">
            <Layers className="w-4 h-4 mr-2" />
            Furnaces
          </TabsTrigger>
        </TabsList>

        {/* Recipes Tab */}
        <TabsContent value="recipes" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Oxidation Recipes</CardTitle>
              <CardDescription>Manage thermal oxidation process recipes and parameters</CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="text-center py-8 text-muted-foreground">Loading...</div>
              ) : recipes.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  No recipes found. Create your first oxidation recipe to get started.
                </div>
              ) : (
                <div className="space-y-4">
                  {recipes.map((recipe) => (
                    <div key={recipe.id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div>
                        <h4 className="font-medium">{recipe.name}</h4>
                        <p className="text-sm text-muted-foreground">
                          {recipe.temperature_c}°C • {recipe.time_minutes} min • {recipe.oxidation_type} •
                          {recipe.target_thickness_nm && ` Target: ${recipe.target_thickness_nm}nm`}
                        </p>
                        {recipe.description && (
                          <p className="text-xs text-muted-foreground mt-1">{recipe.description}</p>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        {getStatusBadge(recipe.status)}
                        <Button size="sm" variant="outline" disabled={recipe.status !== 'APPROVED'}>
                          <PlayCircle className="w-4 h-4 mr-1" />
                          Start Run
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
              <CardTitle>Process Runs</CardTitle>
              <CardDescription>Track oxidation process runs and execution history</CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="text-center py-8 text-muted-foreground">Loading...</div>
              ) : runs.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  No runs found. Start your first oxidation run to begin processing.
                </div>
              ) : (
                <div className="space-y-2">
                  {runs.map((run) => (
                    <div key={run.id} className="flex items-center justify-between p-3 border rounded-lg hover:bg-accent">
                      <div className="flex-1">
                        <p className="font-medium text-sm">{run.run_number || run.id.slice(0, 13)}</p>
                        <p className="text-xs text-muted-foreground">
                          {run.recipe?.name || 'N/A'} • {run.furnace?.name || 'N/A'} • {run.wafer_ids.length} wafers
                        </p>
                        {run.status === 'RUNNING' && run.job_progress > 0 && (
                          <div className="mt-1">
                            <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-blue-500"
                                style={{ width: `${run.job_progress * 100}%` }}
                              />
                            </div>
                          </div>
                        )}
                      </div>
                      {getStatusBadge(run.status)}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Monitor Tab */}
        <TabsContent value="monitor">
          <Card>
            <CardHeader>
              <CardTitle>Real-Time Process Monitoring</CardTitle>
              <CardDescription>Live telemetry streaming from oxidation furnace</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12 text-muted-foreground">
                No active run selected. Start a run to see live monitoring.
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value="results">
          <Card>
            <CardHeader>
              <CardTitle>Process Results & SPC</CardTitle>
              <CardDescription>Oxide thickness measurements and statistical process control</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12 text-muted-foreground">
                Results and SPC charts will be displayed here
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Furnaces Tab */}
        <TabsContent value="furnaces">
          <Card>
            <CardHeader>
              <CardTitle>Furnace Management</CardTitle>
              <CardDescription>Configure and maintain oxidation furnace equipment</CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="text-center py-8 text-muted-foreground">Loading...</div>
              ) : furnaces.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  No furnaces found. Add oxidation furnace equipment to get started.
                </div>
              ) : (
                <div className="space-y-4">
                  {furnaces.map((furnace) => (
                    <div key={furnace.id} className="p-4 border rounded-lg space-y-2">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium">{furnace.name}</h4>
                        <div className="flex gap-2">
                          <Badge variant={furnace.is_active ? 'default' : 'outline'}>
                            {furnace.is_active ? 'ACTIVE' : 'INACTIVE'}
                          </Badge>
                          {!furnace.is_calibrated && (
                            <Badge variant="destructive">UNCALIBRATED</Badge>
                          )}
                        </div>
                      </div>
                      <div className="grid grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-muted-foreground">Type:</span>
                          <p className="font-medium">{furnace.furnace_type}</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Max Temp:</span>
                          <p className="font-medium">{furnace.max_temperature_c}°C</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Capacity:</span>
                          <p className="font-medium">{furnace.max_wafer_capacity} wafers</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Zones:</span>
                          <p className="font-medium">{furnace.num_temperature_zones}</p>
                        </div>
                      </div>
                      <div className="flex gap-2 text-xs text-muted-foreground">
                        {furnace.supports_dry_oxidation && <Badge variant="outline">Dry</Badge>}
                        {furnace.supports_wet_oxidation && <Badge variant="outline">Wet</Badge>}
                        {furnace.supports_steam_oxidation && <Badge variant="outline">Steam</Badge>}
                        {furnace.supports_pyrogenic && <Badge variant="outline">Pyrogenic</Badge>}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default OxidationMES
