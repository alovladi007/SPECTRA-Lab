/**
 * RTP Run Monitor Component
 *
 * Real-time monitoring of RTP process with:
 * - WebSocket connection to live telemetry stream
 * - Multi-zone temperature display (4 zones)
 * - Setpoint vs measured temperature tracking with error ribbons
 * - Lamp power visualization
 * - Thermal budget accumulator
 * - Ramp error tracking and overshoot detection
 * - Segment transition indicators
 * - SPC alerts and warnings
 * - Run control (pause/resume/cancel)
 */

"use client"

import React, { useState, useEffect, useCallback, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Progress } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'
import {
  Thermometer,
  Zap,
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Pause,
  Play,
  TrendingUp,
  TrendingDown,
  Radio,
  Clock,
  Gauge,
  Calculator
} from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  AreaChart,
  Legend
} from 'recharts'

interface TelemetryData {
  time_s: number
  zone_temps_c: {
    zone1: number
    zone2: number
    zone3: number
    zone4: number
  }
  setpoint_c: number
  lamp_power_pct: number
  thermal_budget_c_s: number
  ramp_error_c: number
  current_segment: number
  segment_progress_pct: number
}

interface SPCAlert {
  timestamp: string
  parameter: string
  message: string
  severity: 'info' | 'warning' | 'critical'
  value?: number
  limit?: number
}

interface JobStatus {
  run_id: string
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  current_step: string
  started_at?: string
  completed_at?: string
  error_message?: string
}

interface SegmentInfo {
  segment_id: number
  target_temp_c: number
  duration_s: number
  segment_type: string
  start_time_s: number
  end_time_s: number
}

interface RTPRunMonitorProps {
  runId: string
  apiEndpoint?: string
  wsEndpoint?: string
  onComplete?: () => void
}

export const RTPRunMonitor: React.FC<RTPRunMonitorProps> = ({
  runId,
  apiEndpoint = 'http://localhost:8003',
  wsEndpoint = 'ws://localhost:8003',
  onComplete
}) => {
  const [jobStatus, setJobStatus] = useState<JobStatus>({
    run_id: runId,
    job_id: '',
    status: 'pending',
    progress: 0,
    current_step: 'Initializing...'
  })

  const [currentTelemetry, setCurrentTelemetry] = useState<TelemetryData>({
    time_s: 0,
    zone_temps_c: { zone1: 25, zone2: 25, zone3: 25, zone4: 25 },
    setpoint_c: 25,
    lamp_power_pct: 0,
    thermal_budget_c_s: 0,
    ramp_error_c: 0,
    current_segment: 1,
    segment_progress_pct: 0
  })

  const [telemetryHistory, setTelemetryHistory] = useState<TelemetryData[]>([])
  const [segments, setSegments] = useState<SegmentInfo[]>([])
  const [spcAlerts, setSpcAlerts] = useState<SPCAlert[]>([])
  const [overshootDetected, setOvershootDetected] = useState(false)
  const [maxOvershoot, setMaxOvershoot] = useState(0)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const [isConnected, setIsConnected] = useState(false)

  // Calculate average temperature across all zones
  const avgZoneTemp = (currentTelemetry.zone_temps_c.zone1 +
    currentTelemetry.zone_temps_c.zone2 +
    currentTelemetry.zone_temps_c.zone3 +
    currentTelemetry.zone_temps_c.zone4) / 4

  // Calculate temperature error
  const tempError = avgZoneTemp - currentTelemetry.setpoint_c

  // Detect overshoot
  useEffect(() => {
    if (Math.abs(tempError) > 10) {
      setOvershootDetected(true)
      setMaxOvershoot(prev => Math.max(prev, Math.abs(tempError)))
    }
  }, [tempError])

  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    const token = localStorage.getItem('token') || 'demo-token'
    const ws = new WebSocket(`${wsEndpoint}/api/rtp/stream/${runId}?token=${token}`)

    ws.onopen = () => {
      console.log('RTP WebSocket connected')
      setIsConnected(true)
    }

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data)

      switch (message.type) {
        case 'connected':
          console.log('RTP stream connected:', message.data)
          break

        case 'progress':
          setJobStatus(prev => ({
            ...prev,
            status: message.data.status,
            progress: message.data.progress,
            current_step: message.data.current_step,
          }))
          break

        case 'telemetry':
          const telemetry = message.data as TelemetryData
          setCurrentTelemetry(telemetry)
          setTelemetryHistory(prev => {
            const updated = [...prev, telemetry]
            return updated.slice(-200) // Keep last 200 points
          })
          break

        case 'segments':
          setSegments(message.data as SegmentInfo[])
          break

        case 'alert':
          const alert: SPCAlert = {
            parameter: message.data.parameter,
            message: message.data.message,
            severity: message.data.severity,
            timestamp: new Date().toISOString(),
            value: message.data.value,
            limit: message.data.limit
          }
          setSpcAlerts(prev => [alert, ...prev.slice(0, 19)])
          break

        case 'completed':
          setJobStatus(prev => ({
            ...prev,
            status: 'completed',
            progress: 100,
            current_step: 'Process completed',
            completed_at: new Date().toISOString()
          }))
          if (onComplete) onComplete()
          break

        case 'error':
          setJobStatus(prev => ({
            ...prev,
            status: 'failed',
            error_message: message.data.error
          }))
          break

        case 'cancelled':
          setJobStatus(prev => ({
            ...prev,
            status: 'cancelled',
            current_step: 'Run cancelled by operator'
          }))
          break
      }
    }

    ws.onerror = (error) => {
      console.error('RTP WebSocket error:', error)
      setIsConnected(false)
    }

    ws.onclose = () => {
      console.log('RTP WebSocket disconnected')
      setIsConnected(false)

      // Auto-reconnect if job is still running
      if (jobStatus.status === 'running' || jobStatus.status === 'pending') {
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('Attempting to reconnect...')
          connectWebSocket()
        }, 3000)
      }
    }

    wsRef.current = ws
  }, [runId, wsEndpoint, jobStatus.status, onComplete])

  // Initial connection
  useEffect(() => {
    connectWebSocket()

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [connectWebSocket])

  // Cancel run
  const handleCancel = async () => {
    try {
      const response = await fetch(`${apiEndpoint}/api/rtp/runs/${runId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token') || 'demo-token'}`,
        },
      })

      if (response.ok) {
        console.log('Run cancelled successfully')
      }
    } catch (error) {
      console.error('Failed to cancel run:', error)
    }
  }

  // Get status badge
  const getStatusBadge = () => {
    const statusConfig = {
      pending: { variant: 'secondary' as const, icon: Clock, label: 'Pending' },
      running: { variant: 'default' as const, icon: Activity, label: 'Running' },
      completed: { variant: 'default' as const, icon: CheckCircle, label: 'Completed' },
      failed: { variant: 'destructive' as const, icon: XCircle, label: 'Failed' },
      cancelled: { variant: 'secondary' as const, icon: XCircle, label: 'Cancelled' },
    }

    const config = statusConfig[jobStatus.status]
    const Icon = config.icon

    return (
      <Badge variant={config.variant} className="flex items-center gap-1">
        <Icon className="w-3 h-3" />
        {config.label}
      </Badge>
    )
  }

  // Current segment info
  const currentSegment = segments.find(s => s.segment_id === currentTelemetry.current_segment)

  return (
    <div className="space-y-4">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-start">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5" />
                RTP Run Monitor
              </CardTitle>
              <CardDescription className="mt-1">
                Run ID: {runId}
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              {getStatusBadge()}
              {isConnected ? (
                <Badge variant="outline" className="flex items-center gap-1">
                  <Radio className="w-3 h-3 text-green-500" />
                  Live
                </Badge>
              ) : (
                <Badge variant="outline" className="flex items-center gap-1">
                  <Radio className="w-3 h-3 text-red-500" />
                  Disconnected
                </Badge>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">{jobStatus.current_step}</span>
              <span className="font-semibold">{jobStatus.progress.toFixed(0)}%</span>
            </div>
            <Progress value={jobStatus.progress} />
          </div>

          {currentSegment && (
            <div className="mt-4 p-3 border rounded-lg bg-muted/30">
              <div className="flex justify-between items-center">
                <div>
                  <Badge variant="outline">Segment {currentSegment.segment_id}</Badge>
                  <span className="ml-2 text-sm text-muted-foreground">
                    {currentSegment.segment_type} to {currentSegment.target_temp_c}°C
                  </span>
                </div>
                <div className="text-sm font-mono">
                  {currentTelemetry.segment_progress_pct.toFixed(0)}% complete
                </div>
              </div>
              <Progress value={currentTelemetry.segment_progress_pct} className="mt-2 h-1" />
            </div>
          )}
        </CardContent>
      </Card>

      {/* Overshoot Warning */}
      {overshootDetected && Math.abs(tempError) > 5 && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Temperature overshoot detected: {tempError > 0 ? '+' : ''}{tempError.toFixed(1)}°C
            (Max: {maxOvershoot.toFixed(1)}°C)
          </AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Real-time Telemetry */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Thermometer className="w-5 h-5" />
              Temperature Zones
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Zone Temperature Cards */}
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(currentTelemetry.zone_temps_c).map(([zone, temp]) => {
                const error = temp - currentTelemetry.setpoint_c
                const isInSpec = Math.abs(error) < 5

                return (
                  <div key={zone} className={`p-3 border rounded-lg ${isInSpec ? 'bg-green-50 border-green-300' : 'bg-orange-50 border-orange-300'}`}>
                    <div className="text-xs font-medium text-muted-foreground uppercase">
                      {zone.replace('zone', 'Zone ')}
                    </div>
                    <div className="text-2xl font-bold font-mono mt-1">
                      {temp.toFixed(1)}°C
                    </div>
                    <div className={`text-xs mt-1 flex items-center gap-1 ${error > 0 ? 'text-orange-600' : 'text-blue-600'}`}>
                      {error > 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                      {error > 0 ? '+' : ''}{error.toFixed(1)}°C
                    </div>
                  </div>
                )
              })}
            </div>

            <Separator />

            {/* Setpoint and Average */}
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 border rounded-lg bg-blue-50">
                <div className="text-xs font-medium text-muted-foreground">Setpoint</div>
                <div className="text-2xl font-bold font-mono text-blue-600">
                  {currentTelemetry.setpoint_c.toFixed(1)}°C
                </div>
              </div>

              <div className="p-3 border rounded-lg bg-purple-50">
                <div className="text-xs font-medium text-muted-foreground">Average</div>
                <div className="text-2xl font-bold font-mono text-purple-600">
                  {avgZoneTemp.toFixed(1)}°C
                </div>
              </div>
            </div>

            {/* Ramp Error */}
            <div className="p-3 border rounded-lg bg-muted/30">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-muted-foreground">Tracking Error</span>
                <Badge variant={Math.abs(currentTelemetry.ramp_error_c) < 2 ? 'default' : 'destructive'}>
                  {currentTelemetry.ramp_error_c > 0 ? '+' : ''}{currentTelemetry.ramp_error_c.toFixed(2)}°C
                </Badge>
              </div>
              <div className="mt-2 h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all ${Math.abs(currentTelemetry.ramp_error_c) < 2 ? 'bg-green-500' : 'bg-red-500'}`}
                  style={{ width: `${Math.min(100, Math.abs(currentTelemetry.ramp_error_c) * 10)}%` }}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Lamp Power and Metrics */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5" />
              System Metrics
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Lamp Power */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium">Lamp Power</span>
                <span className="text-2xl font-bold font-mono">
                  {currentTelemetry.lamp_power_pct.toFixed(1)}%
                </span>
              </div>
              <div className="h-4 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-yellow-400 via-orange-500 to-red-500 transition-all"
                  style={{ width: `${currentTelemetry.lamp_power_pct}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>

            <Separator />

            {/* Thermal Budget */}
            <div className="p-3 border rounded-lg bg-gradient-to-br from-orange-50 to-red-50">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Calculator className="w-4 h-4 text-orange-600" />
                  <span className="text-sm font-medium text-muted-foreground">Thermal Budget</span>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold font-mono text-orange-600">
                    {(currentTelemetry.thermal_budget_c_s / 1000).toFixed(1)}k
                  </div>
                  <div className="text-xs text-muted-foreground">°C·s accumulated</div>
                </div>
              </div>
            </div>

            {/* Process Time */}
            <div className="p-3 border rounded-lg bg-muted/30">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4" />
                  <span className="text-sm font-medium text-muted-foreground">Elapsed Time</span>
                </div>
                <div className="text-2xl font-bold font-mono">
                  {Math.floor(currentTelemetry.time_s / 60)}:{(currentTelemetry.time_s % 60).toFixed(0).padStart(2, '0')}
                </div>
              </div>
            </div>

            {/* Zone Uniformity */}
            <div className="p-3 border rounded-lg bg-muted/30">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-muted-foreground">Zone Uniformity (σ)</span>
                <div className="text-lg font-bold font-mono">
                  {(() => {
                    const temps = Object.values(currentTelemetry.zone_temps_c)
                    const mean = temps.reduce((a, b) => a + b, 0) / temps.length
                    const variance = temps.reduce((sum, t) => sum + Math.pow(t - mean, 2), 0) / temps.length
                    const stdDev = Math.sqrt(variance)
                    return `±${stdDev.toFixed(2)}°C`
                  })()}
                </div>
              </div>
            </div>

            {/* Overshoot Indicator */}
            {maxOvershoot > 0 && (
              <div className="p-3 border rounded-lg bg-orange-50 border-orange-300">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-orange-600" />
                    <span className="text-sm font-medium text-orange-800">Max Overshoot</span>
                  </div>
                  <div className="text-lg font-bold font-mono text-orange-600">
                    {maxOvershoot.toFixed(1)}°C
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Temperature Tracking Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Temperature Tracking (Setpoint vs Measured)</CardTitle>
          <CardDescription>Real-time temperature trajectory with error ribbons</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={telemetryHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="time_s"
                  label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }}
                />
                <YAxis
                  label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload
                      const avgTemp = (data.zone_temps_c.zone1 + data.zone_temps_c.zone2 +
                        data.zone_temps_c.zone3 + data.zone_temps_c.zone4) / 4
                      return (
                        <div className="bg-background border rounded p-3 shadow-lg">
                          <p className="text-xs font-semibold mb-1">Time: {data.time_s.toFixed(1)} s</p>
                          <p className="text-xs text-blue-600">Setpoint: {data.setpoint_c.toFixed(1)}°C</p>
                          <p className="text-xs text-purple-600">Average: {avgTemp.toFixed(1)}°C</p>
                          <p className="text-xs text-muted-foreground mt-1">
                            Error: {(avgTemp - data.setpoint_c).toFixed(2)}°C
                          </p>
                        </div>
                      )
                    }
                    return null
                  }}
                />
                <Legend />

                {/* Error ribbon (±5°C bands) */}
                <Area
                  type="monotone"
                  dataKey={(data: TelemetryData) => data.setpoint_c + 5}
                  stroke="none"
                  fill="#fca5a5"
                  fillOpacity={0.2}
                  isAnimationActive={false}
                />
                <Area
                  type="monotone"
                  dataKey={(data: TelemetryData) => data.setpoint_c - 5}
                  stroke="none"
                  fill="#fca5a5"
                  fillOpacity={0.2}
                  isAnimationActive={false}
                />

                {/* Setpoint */}
                <Line
                  type="stepAfter"
                  dataKey="setpoint_c"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={false}
                  name="Setpoint"
                  isAnimationActive={false}
                />

                {/* Average measured temperature */}
                <Line
                  type="monotone"
                  dataKey={(data: TelemetryData) => {
                    return (data.zone_temps_c.zone1 + data.zone_temps_c.zone2 +
                      data.zone_temps_c.zone3 + data.zone_temps_c.zone4) / 4
                  }}
                  stroke="#a855f7"
                  strokeWidth={3}
                  dot={false}
                  name="Measured (Avg)"
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Lamp Power History */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Gauge className="w-5 h-5" />
            Lamp Power History
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={telemetryHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time_s" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey="lamp_power_pct"
                  stroke="#f97316"
                  fill="#fed7aa"
                  strokeWidth={2}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* SPC Alerts */}
      {spcAlerts.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" />
              Process Alerts
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {spcAlerts.map((alert, idx) => (
                <Alert
                  key={idx}
                  variant={alert.severity === 'critical' ? 'destructive' : 'default'}
                  className="py-2"
                >
                  <AlertDescription className="text-xs">
                    <div className="flex justify-between items-start">
                      <div>
                        <Badge variant={alert.severity === 'critical' ? 'destructive' : 'secondary'} className="text-xs">
                          {alert.severity.toUpperCase()}
                        </Badge>
                        <span className="ml-2 font-semibold">{alert.parameter}</span>
                        <p className="mt-1">{alert.message}</p>
                        {alert.value !== undefined && alert.limit !== undefined && (
                          <p className="mt-1 text-muted-foreground">
                            Value: {alert.value.toFixed(2)} | Limit: {alert.limit.toFixed(2)}
                          </p>
                        )}
                      </div>
                      <span className="text-muted-foreground text-xs">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Control Buttons */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex justify-between items-center">
            <div className="text-sm text-muted-foreground">
              {jobStatus.status === 'running' && (
                <span>Process is running... Monitor temperature and lamp power closely.</span>
              )}
              {jobStatus.status === 'completed' && (
                <span className="text-green-600 font-semibold">✓ Process completed successfully</span>
              )}
              {jobStatus.status === 'failed' && (
                <span className="text-red-600 font-semibold">✗ Process failed: {jobStatus.error_message}</span>
              )}
              {jobStatus.status === 'cancelled' && (
                <span className="text-orange-600 font-semibold">Process cancelled by operator</span>
              )}
            </div>
            <div className="flex gap-2">
              {(jobStatus.status === 'running' || jobStatus.status === 'pending') && (
                <Button
                  variant="destructive"
                  onClick={handleCancel}
                >
                  <XCircle className="w-4 h-4 mr-2" />
                  Cancel Run
                </Button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default RTPRunMonitor
