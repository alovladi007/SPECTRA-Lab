/**
 * Calibration Management System (CMS)
 * Equipment calibration tracking, scheduling, and compliance
 */

"use client"

import React, { useState, useEffect } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Button } from '@/components/ui/button'
import { Wrench, Calendar, FileCheck, AlertCircle, CheckCircle, Clock, RefreshCw } from 'lucide-react'
import { RoleDisplay } from '@/components/shared/RBACActionButtons'
import { calibrationApi, type Calibration, type CalibrationDashboard } from '@/lib/api/calibration'

const MOCK_ORG_ID = '00000000-0000-0000-0000-000000000001' // Demo organization UUID

export const CalibrationMES: React.FC = () => {
  const [calibrations, setCalibrations] = useState<Calibration[]>([])
  const [dashboard, setDashboard] = useState<CalibrationDashboard | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      const [calibrationsData, dashboardData] = await Promise.all([
        calibrationApi.getCalibrations({ org_id: MOCK_ORG_ID, limit: 100 }),
        calibrationApi.getDashboard({ org_id: MOCK_ORG_ID }),
      ])
      setCalibrations(calibrationsData)
      setDashboard(dashboardData)
    } catch (error) {
      console.error('Error loading calibration data:', error)
    } finally {
      setLoading(false)
    }
  }

  const getDaysUntilDue = (nextCalDate: string): number => {
    const next = new Date(nextCalDate)
    const now = new Date()
    const diff = next.getTime() - now.getTime()
    return Math.floor(diff / (1000 * 60 * 60 * 24))
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'valid': return 'default'
      case 'due_soon': return 'secondary'
      case 'expired': return 'destructive'
      default: return 'outline'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'valid': return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'due_soon': return <Clock className="w-4 h-4 text-yellow-500" />
      case 'expired': return <AlertCircle className="w-4 h-4 text-red-500" />
      default: return null
    }
  }

  const getStatusLabel = (status: string) => {
    return status === 'due_soon' ? 'DUE SOON' : status.toUpperCase()
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold">Calibration Management</h1>
          <p className="text-muted-foreground mt-1">
            Equipment calibration tracking, scheduling, and compliance management
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
            <Badge variant="outline" className="text-xs">{dashboard.total} Equipment Total</Badge>
          )}
        </div>
      </div>

      <Separator />

      {/* Status Cards */}
      {loading ? (
        <div className="text-center py-12 text-muted-foreground">Loading calibration data...</div>
      ) : !dashboard ? (
        <div className="text-center py-12 text-muted-foreground">
          No calibration data available. Add equipment calibration records to start tracking.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                Valid
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{dashboard.valid}</div>
              <p className="text-xs text-muted-foreground">{dashboard.compliance_rate.toFixed(1)}% compliant</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Clock className="w-4 h-4 text-yellow-500" />
                Due Soon
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-yellow-600">{dashboard.due_soon}</div>
              <p className="text-xs text-muted-foreground">Within 30 days</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-red-500" />
                Expired
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-600">{dashboard.expired}</div>
              <p className="text-xs text-muted-foreground">Requires immediate action</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Calendar className="w-4 h-4 text-blue-500" />
                Scheduled
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{dashboard.upcoming_this_month}</div>
              <p className="text-xs text-muted-foreground">This month</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Main Tabs */}
      <Tabs defaultValue="equipment" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="equipment">
            <Wrench className="w-4 h-4 mr-2" />
            Equipment
          </TabsTrigger>
          <TabsTrigger value="schedule">
            <Calendar className="w-4 h-4 mr-2" />
            Schedule
          </TabsTrigger>
          <TabsTrigger value="certificates">
            <FileCheck className="w-4 h-4 mr-2" />
            Certificates
          </TabsTrigger>
          <TabsTrigger value="compliance">
            <CheckCircle className="w-4 h-4 mr-2" />
            Compliance
          </TabsTrigger>
        </TabsList>

        {/* Equipment Tab */}
        <TabsContent value="equipment" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Equipment Calibration Status</CardTitle>
              <CardDescription>Monitor calibration status across all manufacturing equipment</CardDescription>
            </CardHeader>
            <CardContent>
              {calibrations.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground">
                  No equipment calibration records found.
                </div>
              ) : (
                <div className="space-y-3">
                  {calibrations.map((cal) => {
                    const daysLeft = getDaysUntilDue(cal.next_calibration_date)
                    return (
                      <div key={cal.id} className="flex items-center justify-between p-4 border rounded-lg hover:bg-accent">
                        <div className="flex items-center gap-3 flex-1">
                          {getStatusIcon(cal.status)}
                          <div>
                            <h4 className="font-medium">{cal.equipment_name}</h4>
                            <p className="text-sm text-muted-foreground">
                              {cal.equipment_id.substring(0, 8)} • {cal.equipment_type || 'N/A'}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <div className="text-right">
                            <p className="text-sm font-medium">Next: {new Date(cal.next_calibration_date).toLocaleDateString()}</p>
                            <p className="text-xs text-muted-foreground">
                              {daysLeft > 0 ? `${daysLeft} days left` : `${Math.abs(daysLeft)} days overdue`}
                            </p>
                          </div>
                          <Badge variant={getStatusColor(cal.status)}>
                            {getStatusLabel(cal.status)}
                          </Badge>
                          <Button size="sm" variant="outline">
                            <Wrench className="w-4 h-4 mr-1" />
                            Calibrate
                          </Button>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Schedule Tab */}
        <TabsContent value="schedule">
          <Card>
            <CardHeader>
              <CardTitle>Calibration Schedule</CardTitle>
              <CardDescription>Planned and upcoming calibration events</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {[
                  { date: '2024-12-15', equipment: 'RTP-002', type: 'Temperature', tech: 'Smith, J.' },
                  { date: '2024-12-18', equipment: 'OX-FURN-001', type: 'Temperature', tech: 'Johnson, M.' },
                  { date: '2024-12-20', equipment: 'DIFF-001', type: 'Multi-zone Temp', tech: 'Smith, J.' },
                  { date: '2024-12-22', equipment: 'ION-IMP-002', type: 'Beam Current', tech: 'Lee, S.' }
                ].map((cal, idx) => (
                  <div key={idx} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-3">
                      <Calendar className="w-5 h-5 text-blue-500" />
                      <div>
                        <p className="font-medium text-sm">{cal.equipment} - {cal.type}</p>
                        <p className="text-xs text-muted-foreground">Technician: {cal.tech}</p>
                      </div>
                    </div>
                    <Badge variant="outline">{cal.date}</Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Certificates Tab */}
        <TabsContent value="certificates">
          <Card>
            <CardHeader>
              <CardTitle>Calibration Certificates</CardTitle>
              <CardDescription>Certificate repository and documentation</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12 text-muted-foreground">
                Certificate management and PDF viewer
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Compliance Tab */}
        <TabsContent value="compliance">
          <Card>
            <CardHeader>
              <CardTitle>Compliance Reports</CardTitle>
              <CardDescription>ISO 9001, ISO/IEC 17025 compliance tracking</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">Monthly Compliance Summary</h4>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Overall Compliance:</span>
                      <p className="font-bold text-lg text-green-600">95.8%</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">On-Time Calibrations:</span>
                      <p className="font-bold text-lg">46/48</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Audit Ready:</span>
                      <p className="font-bold text-lg text-green-600">Yes</p>
                    </div>
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

export default CalibrationMES
