/**
 * Predictive Maintenance System (PdM)
 * Equipment health monitoring and predictive maintenance scheduling
 */

"use client"

import React, { useState, useEffect } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Button } from '@/components/ui/button'
import { Gauge, TrendingDown, Calendar, AlertTriangle, CheckCircle, Activity, RefreshCw } from 'lucide-react'
import { RoleDisplay } from '@/components/shared/RBACActionButtons'
import {
  predictiveMaintenanceApi,
  type EquipmentHealth,
  type PredictiveMaintenanceDashboard,
  type FailurePrediction,
  type MaintenanceEvent
} from '@/lib/api/predictive-maintenance'

const MOCK_ORG_ID = '00000000-0000-0000-0000-000000000001' // Demo organization UUID

export const PredictiveMaintenanceMES: React.FC = () => {
  const [healthRecords, setHealthRecords] = useState<EquipmentHealth[]>([])
  const [dashboard, setDashboard] = useState<PredictiveMaintenanceDashboard | null>(null)
  const [predictions, setPredictions] = useState<FailurePrediction[]>([])
  const [events, setEvents] = useState<MaintenanceEvent[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      const [healthData, dashboardData, predictionsData, eventsData] = await Promise.all([
        predictiveMaintenanceApi.getHealthRecords({ org_id: MOCK_ORG_ID, limit: 50 }),
        predictiveMaintenanceApi.getDashboard({ org_id: MOCK_ORG_ID }),
        predictiveMaintenanceApi.getFailurePredictions({ org_id: MOCK_ORG_ID, days_ahead: 90, limit: 10 }),
        predictiveMaintenanceApi.getMaintenanceEvents({ org_id: MOCK_ORG_ID, limit: 50 }),
      ])
      setHealthRecords(healthData)
      setDashboard(dashboardData)
      setPredictions(predictionsData)
      setEvents(eventsData)
    } catch (error) {
      console.error('Error loading predictive maintenance data:', error)
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'default'
      case 'warning': return 'secondary'
      case 'critical': return 'destructive'
      default: return 'outline'
    }
  }

  const getStatusLabel = (status: string) => {
    return status.toUpperCase()
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
          <div className="flex items-center gap-2">
            <RoleDisplay showPermissions={false} />
            <Button onClick={loadData} variant="outline" size="sm" disabled={loading}>
              <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
          {dashboard && (
            <Badge variant="outline" className="text-xs">{dashboard.total_equipment} Equipment Monitored</Badge>
          )}
        </div>
      </div>

      <Separator />

      {/* Status Cards */}
      {loading ? (
        <div className="text-center py-12 text-muted-foreground">Loading predictive maintenance data...</div>
      ) : !dashboard ? (
        <div className="text-center py-12 text-muted-foreground">
          No predictive maintenance data available. Add equipment health records to start monitoring.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                Healthy
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{dashboard.healthy_count}</div>
              <p className="text-xs text-muted-foreground">
                {dashboard.total_equipment > 0 ? ((dashboard.healthy_count / dashboard.total_equipment) * 100).toFixed(1) : 0}% of fleet
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <TrendingDown className="w-4 h-4 text-yellow-500" />
                Warning
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-yellow-600">{dashboard.warning_count}</div>
              <p className="text-xs text-muted-foreground">Requires monitoring</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-red-500" />
                Critical
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-600">{dashboard.critical_count}</div>
              <p className="text-xs text-muted-foreground">Action needed</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Activity className="w-4 h-4 text-blue-500" />
                Avg Health
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{dashboard.avg_health_score.toFixed(1)}%</div>
              <p className="text-xs text-muted-foreground">Overall health score</p>
            </CardContent>
          </Card>
        </div>
      )}

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
              {loading ? (
                <div className="text-center py-8 text-muted-foreground">Loading...</div>
              ) : healthRecords.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  No equipment health records found. Add health monitoring data to get started.
                </div>
              ) : (
                <div className="space-y-4">
                  {healthRecords.map((health) => (
                    <div key={health.id} className="p-4 border rounded-lg space-y-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="font-medium">{health.equipment_name}</h4>
                          <p className="text-sm text-muted-foreground">
                            {health.equipment_id.substring(0, 13)} • {health.equipment_type || 'N/A'}
                          </p>
                        </div>
                        <Badge variant={getStatusColor(health.status)}>{getStatusLabel(health.status)}</Badge>
                      </div>

                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <span className="text-muted-foreground">Health Score</span>
                          <p className={`font-bold text-lg ${getHealthColor(health.health_score)}`}>
                            {health.health_score.toFixed(1)}%
                          </p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">MTBF</span>
                          <p className="font-bold text-lg">
                            {health.mtbf_hours ? `${health.mtbf_hours.toFixed(0)}h` : 'N/A'}
                          </p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Next PM</span>
                          <p className="font-medium">
                            {health.next_maintenance_date ? new Date(health.next_maintenance_date).toLocaleDateString() : 'N/A'}
                          </p>
                        </div>
                      </div>

                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-muted-foreground">Health Trend</span>
                          <span className="font-medium">
                            {health.predicted_failure_date ?
                              `Predicted: ${new Date(health.predicted_failure_date).toLocaleDateString()}` :
                              'No prediction'}
                          </span>
                        </div>
                        <div className="w-full bg-secondary rounded-full h-2">
                          <div
                            className={`h-2 rounded-full transition-all ${
                              health.health_score >= 90 ? 'bg-green-500' :
                              health.health_score >= 75 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${health.health_score}%` }}
                          />
                        </div>
                      </div>

                      {health.recommendations && (
                        <div className="p-2 bg-secondary/50 rounded text-xs">
                          <p className="text-muted-foreground">Recommendations:</p>
                          <p className="font-medium">
                            {JSON.stringify(health.recommendations).substring(0, 100)}...
                          </p>
                        </div>
                      )}

                      <Button size="sm" variant="outline" className="w-full">
                        View Detailed Diagnostics
                      </Button>
                    </div>
                  ))}
                </div>
              )}
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
              {loading ? (
                <div className="text-center py-8 text-muted-foreground">Loading...</div>
              ) : predictions.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  No upcoming failure predictions found.
                </div>
              ) : (
                <div className="space-y-3">
                  {predictions.map((pred) => (
                    <div key={pred.equipment_id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex-1">
                        <h4 className="font-medium text-sm">
                          {pred.equipment_name} ({pred.equipment_type || 'N/A'})
                        </h4>
                        <p className="text-xs text-muted-foreground">
                          Failure probability: {(pred.failure_probability * 100).toFixed(1)}% in {pred.days_until_failure} days
                          • Confidence: {(pred.confidence * 100).toFixed(1)}%
                          • Health: {pred.health_score.toFixed(1)}%
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">
                          Predicted: {new Date(pred.predicted_failure_date).toLocaleDateString()}
                        </p>
                      </div>
                      <div className="flex flex-col gap-2 items-end">
                        <Badge variant={pred.failure_probability > 0.5 ? 'destructive' : pred.failure_probability > 0.3 ? 'secondary' : 'default'}>
                          {pred.failure_probability > 0.5 ? 'HIGH' : pred.failure_probability > 0.3 ? 'MEDIUM' : 'LOW'}
                        </Badge>
                        <Badge variant={getStatusColor(pred.status)} className="text-xs">
                          {getStatusLabel(pred.status)}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              )}
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
              {loading ? (
                <div className="text-center py-8 text-muted-foreground">Loading...</div>
              ) : healthRecords.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  No maintenance schedules found.
                </div>
              ) : (
                <div className="space-y-2">
                  {healthRecords
                    .filter(health => health.next_maintenance_date)
                    .sort((a, b) => new Date(a.next_maintenance_date!).getTime() - new Date(b.next_maintenance_date!).getTime())
                    .map((health) => (
                      <div key={health.id} className="flex items-center justify-between p-3 border rounded-lg">
                        <div className="flex items-center gap-3">
                          <Calendar className="w-5 h-5 text-blue-500" />
                          <div>
                            <p className="font-medium text-sm">{health.equipment_name}</p>
                            <p className="text-xs text-muted-foreground">
                              Last PM: {health.last_maintenance_date ? new Date(health.last_maintenance_date).toLocaleDateString() : 'N/A'}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <Badge variant="outline">
                            {new Date(health.next_maintenance_date!).toLocaleDateString()}
                          </Badge>
                          <Badge variant={getStatusColor(health.status)} className="text-xs">
                            {getStatusLabel(health.status)}
                          </Badge>
                        </div>
                      </div>
                    ))}
                  {healthRecords.filter(health => health.next_maintenance_date).length === 0 && (
                    <div className="text-center py-8 text-muted-foreground">
                      No scheduled maintenance dates found.
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="analytics">
          <Card>
            <CardHeader>
              <CardTitle>Maintenance Analytics</CardTitle>
              <CardDescription>Historical trends, cost analysis, and downtime optimization (Last 30 days)</CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="text-center py-8 text-muted-foreground">Loading...</div>
              ) : !dashboard ? (
                <div className="text-center py-8 text-muted-foreground">
                  No analytics data available.
                </div>
              ) : (
                <div className="space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-2">Total Downtime</h4>
                      <p className="text-2xl font-bold">{dashboard.total_downtime_hours.toFixed(1)}h</p>
                      <p className="text-xs text-muted-foreground">Last 30 days</p>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-2">Maintenance Cost</h4>
                      <p className="text-2xl font-bold">${(dashboard.total_maintenance_cost / 1000).toFixed(1)}K</p>
                      <p className="text-xs text-muted-foreground">Last 30 days</p>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-2">Avg Health Score</h4>
                      <p className="text-2xl font-bold">{dashboard.avg_health_score.toFixed(1)}%</p>
                      <p className="text-xs text-muted-foreground">Fleet average</p>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-2">Predicted Failures</h4>
                      <p className="text-2xl font-bold">{dashboard.predictions_this_month}</p>
                      <p className="text-xs text-muted-foreground">Next 30 days</p>
                    </div>
                  </div>

                  {events.length > 0 && (
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-3">Recent Maintenance Events ({events.length})</h4>
                      <div className="space-y-2">
                        {events.slice(0, 5).map((event) => (
                          <div key={event.id} className="flex items-center justify-between text-sm p-2 bg-secondary/30 rounded">
                            <div className="flex-1">
                              <p className="font-medium">{event.event_type.toUpperCase()}</p>
                              <p className="text-xs text-muted-foreground">
                                {new Date(event.performed_date).toLocaleDateString()} •
                                {event.technician ? ` ${event.technician} • ` : ' '}
                                {event.downtime_hours}h downtime
                              </p>
                            </div>
                            <Badge variant="outline">${event.cost.toFixed(0)}</Badge>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default PredictiveMaintenanceMES
