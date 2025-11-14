'use client'

/**
 * Diffusion Manufacturing Platform
 * Production-grade thermal diffusion MES interface
 */

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Activity, Plus, Play, Square, Thermometer, Beaker, BarChart3, Settings } from 'lucide-react'
import TelemetryDashboard from '@/components/diffusion/TelemetryDashboard'
import { diffusionApi, type DiffusionFurnace, type DiffusionRecipe, type DiffusionRun } from '@/lib/api/diffusion'

const MOCK_ORG_ID = 'org_demo_001'

export default function DiffusionManufacturingPage() {
  const [activeTab, setActiveTab] = useState('furnaces')
  const [furnaces, setFurnaces] = useState<DiffusionFurnace[]>([])
  const [recipes, setRecipes] = useState<DiffusionRecipe[]>([])
  const [runs, setRuns] = useState<DiffusionRun[]>([])
  const [selectedRun, setSelectedRun] = useState<DiffusionRun | null>(null)
  const [loading, setLoading] = useState(true)

  // Load data on mount
  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      const [furnacesData, recipesData, runsData] = await Promise.all([
        diffusionApi.getFurnaces({ org_id: MOCK_ORG_ID }),
        diffusionApi.getRecipes({ org_id: MOCK_ORG_ID }),
        diffusionApi.getRuns({ org_id: MOCK_ORG_ID, limit: 20 }),
      ])
      setFurnaces(furnacesData)
      setRecipes(recipesData)
      setRuns(runsData)
    } catch (error) {
      console.error('Error loading data:', error)
    } finally {
      setLoading(false)
    }
  }

  const getStatusBadge = (status: string) => {
    const variants: Record<string, 'default' | 'destructive' | 'outline' | 'secondary'> = {
      QUEUED: 'outline',
      RUNNING: 'default',
      COMPLETED: 'secondary',
      FAILED: 'destructive',
      ABORTED: 'destructive',
    }
    return <Badge variant={variants[status] || 'outline'}>{status}</Badge>
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg">
                <Thermometer className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                  Diffusion Manufacturing
                </h1>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Production thermal diffusion platform with real-time monitoring
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              <Button onClick={loadData} variant="outline" size="sm">
                <Activity className="w-4 h-4 mr-2" />
                Refresh
              </Button>
              <Button size="sm">
                <Plus className="w-4 h-4 mr-2" />
                New Run
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Real-time Telemetry (if run selected) */}
        {selectedRun && selectedRun.status === 'RUNNING' && (
          <div className="mb-6">
            <TelemetryDashboard
              runId={selectedRun.id}
              recipeName={selectedRun.recipe?.name}
              targetTemp={selectedRun.recipe?.temperature_profile?.peak_temp_c}
              onDisconnect={() => setSelectedRun(null)}
            />
          </div>
        )}

        {/* Main Content Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList>
            <TabsTrigger value="furnaces">
              <Thermometer className="w-4 h-4 mr-2" />
              Furnaces
            </TabsTrigger>
            <TabsTrigger value="recipes">
              <Beaker className="w-4 h-4 mr-2" />
              Recipes
            </TabsTrigger>
            <TabsTrigger value="runs">
              <Play className="w-4 h-4 mr-2" />
              Runs
            </TabsTrigger>
            <TabsTrigger value="spc">
              <BarChart3 className="w-4 h-4 mr-2" />
              SPC
            </TabsTrigger>
          </TabsList>

          {/* Furnaces Tab */}
          <TabsContent value="furnaces" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Diffusion Furnaces</CardTitle>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="text-center py-8 text-gray-500">Loading...</div>
                ) : furnaces.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    No furnaces found. Create your first furnace to get started.
                  </div>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Zones</TableHead>
                        <TableHead>Capacity</TableHead>
                        <TableHead>Dopants</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {furnaces.map((furnace) => (
                        <TableRow key={furnace.id}>
                          <TableCell className="font-medium">{furnace.name}</TableCell>
                          <TableCell>{furnace.furnace_type}</TableCell>
                          <TableCell>{furnace.num_temperature_zones}</TableCell>
                          <TableCell>{furnace.max_wafer_capacity} wafers</TableCell>
                          <TableCell>
                            <div className="flex gap-1 flex-wrap">
                              {furnace.supported_dopants.slice(0, 3).map((dopant) => (
                                <Badge key={dopant} variant="outline" className="text-xs">
                                  {dopant}
                                </Badge>
                              ))}
                            </div>
                          </TableCell>
                          <TableCell>
                            <Badge variant={furnace.is_active ? 'default' : 'secondary'}>
                              {furnace.is_active ? 'Active' : 'Inactive'}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <Button variant="outline" size="sm">
                              <Settings className="w-4 h-4" />
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Recipes Tab */}
          <TabsContent value="recipes" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Diffusion Recipes</CardTitle>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="text-center py-8 text-gray-500">Loading...</div>
                ) : recipes.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    No recipes found. Create your first recipe to get started.
                  </div>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Dopant</TableHead>
                        <TableHead>Source</TableHead>
                        <TableHead>Time</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Runs</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {recipes.map((recipe) => (
                        <TableRow key={recipe.id}>
                          <TableCell className="font-medium">{recipe.name}</TableCell>
                          <TableCell>{recipe.diffusion_type}</TableCell>
                          <TableCell>{recipe.dopant}</TableCell>
                          <TableCell>{recipe.dopant_source}</TableCell>
                          <TableCell>{recipe.total_time_min} min</TableCell>
                          <TableCell>{getStatusBadge(recipe.status)}</TableCell>
                          <TableCell>{recipe.run_count}</TableCell>
                          <TableCell>
                            <Button variant="outline" size="sm">
                              View
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Runs Tab */}
          <TabsContent value="runs" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Recent Runs</CardTitle>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="text-center py-8 text-gray-500">Loading...</div>
                ) : runs.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    No runs found. Start a new run to begin processing.
                  </div>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Run Number</TableHead>
                        <TableHead>Recipe</TableHead>
                        <TableHead>Furnace</TableHead>
                        <TableHead>Wafers</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Progress</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {runs.map((run) => (
                        <TableRow key={run.id}>
                          <TableCell className="font-medium">
                            {run.run_number || run.id.slice(0, 8)}
                          </TableCell>
                          <TableCell>{run.recipe?.name || 'N/A'}</TableCell>
                          <TableCell>{run.furnace?.name || 'N/A'}</TableCell>
                          <TableCell>{run.wafer_ids.length}</TableCell>
                          <TableCell>{getStatusBadge(run.status)}</TableCell>
                          <TableCell>
                            {run.status === 'RUNNING' && run.job_progress ? (
                              <div className="flex items-center gap-2">
                                <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-blue-500"
                                    style={{ width: `${run.job_progress * 100}%` }}
                                  />
                                </div>
                                <span className="text-xs text-gray-500">
                                  {(run.job_progress * 100).toFixed(0)}%
                                </span>
                              </div>
                            ) : (
                              <span className="text-xs text-gray-500">--</span>
                            )}
                          </TableCell>
                          <TableCell>
                            {run.status === 'RUNNING' ? (
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => setSelectedRun(run)}
                              >
                                <Activity className="w-4 h-4 mr-1" />
                                Monitor
                              </Button>
                            ) : (
                              <Button variant="outline" size="sm">
                                View
                              </Button>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* SPC Tab */}
          <TabsContent value="spc" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Statistical Process Control</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-12 text-gray-500">
                  SPC charts and analysis will be displayed here.
                  <br />
                  <span className="text-sm">Junction depth, sheet resistance, uniformity trending</span>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">
                Active Furnaces
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {furnaces.filter((f) => f.is_active).length}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">
                Approved Recipes
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {recipes.filter((r) => r.status === 'APPROVED').length}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">
                Running Jobs
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {runs.filter((r) => r.status === 'RUNNING').length}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">
                Today's Wafers
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {runs
                  .filter((r) => {
                    const today = new Date().toDateString()
                    return new Date(r.created_at).toDateString() === today
                  })
                  .reduce((sum, r) => sum + r.wafer_ids.length, 0)}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
