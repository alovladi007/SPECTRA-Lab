/**
 * Ion Implantation Control Dashboard
 *
 * Integrates all Ion Implantation components:
 * - Equipment Mimic Panel
 * - Recipe Builder
 * - Run Monitor
 * - Results View
 * - Global components (Wafer Explorer, Calibration Badges, RBAC, FAIR Export)
 */

"use client"

import React, { useState } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Settings, Activity, BarChart3, Radio } from 'lucide-react'

// Import all Ion components
import IonEquipmentMimic from '@/components/process-control/ion/IonEquipmentMimic'
import IonRecipeBuilder from '@/components/process-control/ion/IonRecipeBuilder'
import IonRunMonitor from '@/components/process-control/ion/IonRunMonitor'
import IonResultsView from '@/components/process-control/ion/IonResultsView'

// Import global components
import WaferLotExplorer from '@/components/shared/WaferLotExplorer'
import CalibrationBadges from '@/components/shared/CalibrationBadges'
import { RoleDisplay } from '@/components/shared/RBACActionButtons'
import FAIRExportButton from '@/components/shared/FAIRExportButton'

// Import Section 8 components
import AutoReportGenerator from '@/components/shared/AutoReportGenerator'
import FAIRDataPackageExporter from '@/components/shared/FAIRDataPackageExporter'
import ELNLIMSIntegration from '@/components/shared/ELNLIMSIntegration'

export const IonImplantationControl: React.FC = () => {
  const [currentRunId, setCurrentRunId] = useState<string | null>(null)
  const [selectedWaferId, setSelectedWaferId] = useState<string | null>(null)

  const handleRecipeSubmit = (recipe: any) => {
    // This would be called after recipe submission
    // The run_id would come from the API response
    console.log('Recipe submitted:', recipe)
    // Simulate run ID generation
    const runId = `ION-${Date.now()}`
    setCurrentRunId(runId)
  }

  const handleRunComplete = () => {
    console.log('Run completed:', currentRunId)
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold">Ion Implantation Control</h1>
          <p className="text-muted-foreground mt-1">
            Configure, monitor, and analyze ion implantation processes
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <RoleDisplay showPermissions={false} />
          <CalibrationBadges equipmentIds={['ION-001']} />
        </div>
      </div>

      <Separator />

      {/* Main Tabs */}
      <Tabs defaultValue="equipment" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="equipment">
            <Radio className="w-4 h-4 mr-2" />
            Equipment
          </TabsTrigger>
          <TabsTrigger value="recipe">
            <Settings className="w-4 h-4 mr-2" />
            Recipe Builder
          </TabsTrigger>
          <TabsTrigger value="monitor" disabled={!currentRunId}>
            <Activity className="w-4 h-4 mr-2" />
            Run Monitor
            {currentRunId && <Badge variant="destructive" className="ml-2 animate-pulse">LIVE</Badge>}
          </TabsTrigger>
          <TabsTrigger value="results" disabled={!currentRunId}>
            <BarChart3 className="w-4 h-4 mr-2" />
            Results
          </TabsTrigger>
          <TabsTrigger value="wafers">
            Wafer Explorer
          </TabsTrigger>
        </TabsList>

        {/* Equipment Mimic Tab */}
        <TabsContent value="equipment">
          <IonEquipmentMimic />
        </TabsContent>

        {/* Recipe Builder Tab */}
        <TabsContent value="recipe">
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Create New Implantation Recipe</CardTitle>
              </CardHeader>
            </Card>
            <IonRecipeBuilder onSubmit={handleRecipeSubmit} />
          </div>
        </TabsContent>

        {/* Run Monitor Tab */}
        <TabsContent value="monitor">
          {currentRunId ? (
            <IonRunMonitor
              runId={currentRunId}
              onComplete={handleRunComplete}
            />
          ) : (
            <Card>
              <CardContent className="py-12 text-center">
                <div className="text-muted-foreground">
                  No active run. Please submit a recipe from the Recipe Builder tab.
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value="results">
          {currentRunId ? (
            <div className="space-y-4">
              <div className="flex justify-between items-center gap-2">
                <Card className="flex-1">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base">Run Analysis & Export Options</CardTitle>
                  </CardHeader>
                </Card>
                <div className="flex gap-2">
                  <FAIRExportButton
                    datasetId={currentRunId}
                    datasetType="ion_run"
                    defaultTitle={`Ion Implantation Run ${currentRunId}`}
                  />
                  <FAIRDataPackageExporter
                    runId={currentRunId}
                    processType="ion"
                  />
                  <ELNLIMSIntegration
                    runId={currentRunId}
                    processType="ion"
                    waferId={selectedWaferId || undefined}
                  />
                </div>
              </div>

              <IonResultsView runId={currentRunId} />

              {/* Section 8: Reports & Documentation */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Reports & Documentation</CardTitle>
                </CardHeader>
                <CardContent>
                  <AutoReportGenerator
                    runId={currentRunId}
                    processType="ion"
                  />
                </CardContent>
              </Card>
            </div>
          ) : (
            <Card>
              <CardContent className="py-12 text-center">
                <div className="text-muted-foreground">
                  No run data available. Complete a run to view results.
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Wafer Explorer Tab */}
        <TabsContent value="wafers">
          <WaferLotExplorer
            filterProcessType="ion"
            onWaferSelect={(wafer) => {
              setSelectedWaferId(wafer.wafer_id)
              console.log('Selected wafer:', wafer)
            }}
          />
          {selectedWaferId && (
            <Card className="mt-4 border-2 border-primary">
              <CardHeader>
                <CardTitle>Selected: {selectedWaferId}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Wafer selected for next implantation run
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default IonImplantationControl
