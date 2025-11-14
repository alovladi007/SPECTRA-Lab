/**
 * Predictive Maintenance System (PdM)
 * Equipment health monitoring and predictive maintenance scheduling
 */

"use client"

import React, { useState } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Button } from '@/components/ui/button'
import { Gauge, TrendingDown, Calendar, AlertTriangle, CheckCircle, Activity } from 'lucide-react'
import { RoleDisplay } from '@/components/shared/RBACActionButtons'

export const PredictiveMaintenanceMES: React.FC = () => {
  const [equipment] = useState([
    { id: 'ION-IMP-001', name: 'Ion Implanter 1', health: 92, mtbf: 4200, nextPM: '2025-01-10', risk: 'LOW', prediction: 'HEALTHY' },
    { id: 'RTP-001', name: 'RTP Chamber 1', health: 78, mtbf: 3100, nextPM: '2024-12-28', risk: 'MEDIUM', prediction: 'DEGRADING' },
    { id: 'DIFF-001', name: 'Diffusion Furnace', health: 65, mtbf: 2400, nextPM: '2024-12-20', risk: 'HIGH', prediction: 'AT_RISK' },
    { id: 'OX-FURN-001', name: 'Oxidation Furnace 1', health: 95, mtbf: 5100, nextPM: '2025-02-05', risk: 'LOW', prediction: 'HEALTHY' }
  ])

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'LOW': return 'default'
      case 'MEDIUM': return 'secondary'
      case 'HIGH': return 'destructive'
      default: return 'outline'
    }
  }

  const getHealthColor = (health: number) => {
    if (health >= 90) return 'text-green-600'
    if (health >= 75) return 'text-yellow-600'
    return 'text-red-600'
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold">Predictive Maintenance</h1>
          <p className="text-muted-foreground mt-1">
            AI-powered equipment health monitoring and failure prediction
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <RoleDisplay showPermissions={false} />
          <Badge variant="outline" className="text-xs">ML Model v2.3.1</Badge>
        </div>
      </div>

      <Separator />

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-500" />
              Healthy
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">38</div>
            <p className="text-xs text-muted-foreground">79.2% of fleet</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingDown className="w-4 h-4 text-yellow-500" />
              Degrading
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-yellow-600">8</div>
            <p className="text-xs text-muted-foreground">Requires monitoring</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-red-500" />
              At Risk
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">2</div>
            <p className="text-xs text-muted-foreground">Action needed</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Activity className="w-4 h-4 text-blue-500" />
              Avg MTBF
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">3,850h</div>
            <p className="text-xs text-muted-foreground">Mean time between failures</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Tabs */}
      <Tabs defaultValue="equipment" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="equipment">
            <Gauge className="w-4 h-4 mr-2" />
            Equipment Health
          </TabsTrigger>
          <TabsTrigger value="predictions">
            <TrendingDown className="w-4 h-4 mr-2" />
            Predictions
          </TabsTrigger>
          <TabsTrigger value="schedule">
            <Calendar className="w-4 h-4 mr-2" />
            PM Schedule
          </TabsTrigger>
          <TabsTrigger value="analytics">
            <Activity className="w-4 h-4 mr-2" />
            Analytics
          </TabsTrigger>
        </TabsList>

        {/* Equipment Health Tab */}
        <TabsContent value="equipment" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Equipment Health Dashboard</CardTitle>
              <CardDescription>Real-time health scores and failure risk assessment</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {equipment.map((eq) => (
                  <div key={eq.id} className="p-4 border rounded-lg space-y-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium">{eq.name}</h4>
                        <p className="text-sm text-muted-foreground">{eq.id}</p>
                      </div>
                      <Badge variant={getRiskColor(eq.risk)}>{eq.risk} RISK</Badge>
                    </div>

                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Health Score</span>
                        <p className={`font-bold text-lg ${getHealthColor(eq.health)}`}>{eq.health}%</p>
                      </div>
                      <div>
                        <span className="text-muted-foreground">MTBF</span>
                        <p className="font-bold text-lg">{eq.mtbf}h</p>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Next PM</span>
                        <p className="font-medium">{eq.nextPM}</p>
                      </div>
                    </div>

                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-muted-foreground">Health Trend</span>
                        <span className="font-medium">Status: {eq.prediction}</span>
                      </div>
                      <div className="w-full bg-secondary rounded-full h-2">
                        <div
                          className={`h-2 rounded-full transition-all ${
                            eq.health >= 90 ? 'bg-green-500' :
                            eq.health >= 75 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${eq.health}%` }}
                        />
                      </div>
                    </div>

                    <Button size="sm" variant="outline" className="w-full">
                      View Detailed Diagnostics
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Predictions Tab */}
        <TabsContent value="predictions">
          <Card>
            <CardHeader>
              <CardTitle>Failure Predictions</CardTitle>
              <CardDescription>ML-based failure forecasting and anomaly detection</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {[
                  { equipment: 'DIFF-001', component: 'Heating Element', probability: 68, window: '14 days', confidence: 82 },
                  { equipment: 'RTP-001', component: 'Lamp Array', probability: 45, window: '30 days', confidence: 75 },
                  { equipment: 'ION-IMP-002', component: 'Beam Optics', probability: 32, window: '45 days', confidence: 71 }
                ].map((pred, idx) => (
                  <div key={idx} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex-1">
                      <h4 className="font-medium text-sm">{pred.equipment} - {pred.component}</h4>
                      <p className="text-xs text-muted-foreground">
                        Failure probability: {pred.probability}% within {pred.window} • Confidence: {pred.confidence}%
                      </p>
                    </div>
                    <Badge variant={pred.probability > 50 ? 'destructive' : 'secondary'}>
                      {pred.probability > 50 ? 'HIGH' : 'MEDIUM'}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* PM Schedule Tab */}
        <TabsContent value="schedule">
          <Card>
            <CardHeader>
              <CardTitle>Preventive Maintenance Schedule</CardTitle>
              <CardDescription>Optimized PM scheduling based on health predictions</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {equipment.map((eq) => (
                  <div key={eq.id} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-3">
                      <Calendar className="w-5 h-5 text-blue-500" />
                      <div>
                        <p className="font-medium text-sm">{eq.name}</p>
                        <p className="text-xs text-muted-foreground">Scheduled PM</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <Badge variant="outline">{eq.nextPM}</Badge>
                      <Button size="sm" variant="outline">Reschedule</Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="analytics">
          <Card>
            <CardHeader>
              <CardTitle>Maintenance Analytics</CardTitle>
              <CardDescription>Historical trends, cost analysis, and downtime optimization</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">Downtime Reduction</h4>
                  <p className="text-2xl font-bold text-green-600">32%</p>
                  <p className="text-xs text-muted-foreground">vs. reactive maintenance</p>
                </div>
                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">Cost Savings</h4>
                  <p className="text-2xl font-bold text-green-600">$245K</p>
                  <p className="text-xs text-muted-foreground">Annual savings (FY2024)</p>
                </div>
                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">Prediction Accuracy</h4>
                  <p className="text-2xl font-bold">87.3%</p>
                  <p className="text-xs text-muted-foreground">Model performance</p>
                </div>
                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">MTBF Improvement</h4>
                  <p className="text-2xl font-bold text-green-600">+18%</p>
                  <p className="text-xs text-muted-foreground">Year over year</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default PredictiveMaintenanceMES
