/**
 * Oxidation Manufacturing Execution System (MES)
 * Production-grade thermal oxidation control and monitoring
 */

"use client"

import React, { useState } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Button } from '@/components/ui/button'
import { Layers, Activity, BarChart3, Settings, PlayCircle, Database } from 'lucide-react'
import { RoleDisplay } from '@/components/shared/RBACActionButtons'
import CalibrationBadges from '@/components/shared/CalibrationBadges'

export const OxidationMES: React.FC = () => {
  const [activeRun, setActiveRun] = useState<string | null>(null)
  const [furnaces] = useState([
    { id: 'OX-FURN-001', name: 'Horizontal Tube 1', status: 'IDLE', temp: 950, type: 'DRY_WET' },
    { id: 'OX-FURN-002', name: 'Horizontal Tube 2', status: 'RUNNING', temp: 1100, type: 'DRY_ONLY' },
    { id: 'OX-FURN-003', name: 'Vertical Tube', status: 'IDLE', temp: 800, type: 'WET_ONLY' }
  ])

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
          <RoleDisplay showPermissions={false} />
          <CalibrationBadges equipmentIds={['OX-FURN-001', 'OX-FURN-002']} />
        </div>
      </div>

      <Separator />

      {/* Equipment Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {furnaces.map((furnace) => (
          <Card key={furnace.id}>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-medium">{furnace.name}</CardTitle>
                <Badge variant={furnace.status === 'RUNNING' ? 'destructive' : 'outline'}>
                  {furnace.status}
                </Badge>
              </div>
              <CardDescription className="text-xs">{furnace.id}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Temperature</span>
                  <span className="font-medium">{furnace.temp}°C</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Type</span>
                  <span className="font-medium text-xs">{furnace.type}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

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
              <div className="space-y-4">
                {[
                  { name: 'Dry Oxide 50nm', temp: 1000, time: 120, ambient: 'DRY', thickness: 50, status: 'APPROVED' },
                  { name: 'Wet Oxide 200nm', temp: 1000, time: 60, ambient: 'WET', thickness: 200, status: 'APPROVED' },
                  { name: 'Gate Oxide 10nm', temp: 900, time: 30, ambient: 'DRY', thickness: 10, status: 'DRAFT' }
                ].map((recipe, idx) => (
                  <div key={idx} className="flex items-center justify-between p-4 border rounded-lg">
                    <div>
                      <h4 className="font-medium">{recipe.name}</h4>
                      <p className="text-sm text-muted-foreground">
                        {recipe.temp}°C • {recipe.time} min • {recipe.ambient} • Target: {recipe.thickness}nm
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant={recipe.status === 'APPROVED' ? 'default' : 'secondary'}>
                        {recipe.status}
                      </Badge>
                      <Button size="sm" variant="outline">
                        <PlayCircle className="w-4 h-4 mr-1" />
                        Start Run
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
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
              <div className="space-y-2">
                {[
                  { id: 'OX-RUN-2024-001', recipe: 'Dry Oxide 50nm', status: 'COMPLETED', furnace: 'OX-FURN-001', wafers: 25 },
                  { id: 'OX-RUN-2024-002', recipe: 'Wet Oxide 200nm', status: 'RUNNING', furnace: 'OX-FURN-002', wafers: 25 },
                  { id: 'OX-RUN-2024-003', recipe: 'Gate Oxide 10nm', status: 'QUEUED', furnace: 'OX-FURN-001', wafers: 25 }
                ].map((run) => (
                  <div key={run.id} className="flex items-center justify-between p-3 border rounded-lg hover:bg-accent">
                    <div className="flex-1">
                      <p className="font-medium text-sm">{run.id}</p>
                      <p className="text-xs text-muted-foreground">{run.recipe} • {run.furnace} • {run.wafers} wafers</p>
                    </div>
                    <Badge variant={
                      run.status === 'COMPLETED' ? 'default' :
                      run.status === 'RUNNING' ? 'destructive' : 'secondary'
                    }>
                      {run.status}
                    </Badge>
                  </div>
                ))}
              </div>
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
              <div className="space-y-4">
                {furnaces.map((furnace) => (
                  <div key={furnace.id} className="p-4 border rounded-lg space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className="font-medium">{furnace.name}</h4>
                      <Badge variant={furnace.status === 'RUNNING' ? 'destructive' : 'outline'}>
                        {furnace.status}
                      </Badge>
                    </div>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Equipment ID:</span>
                        <p className="font-medium">{furnace.id}</p>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Temperature:</span>
                        <p className="font-medium">{furnace.temp}°C</p>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Configuration:</span>
                        <p className="font-medium">{furnace.type}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default OxidationMES
