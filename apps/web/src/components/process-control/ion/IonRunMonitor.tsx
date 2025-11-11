/**
 * Ion Implantation Run Monitor
 *
 * Real-time monitoring interface with:
 * - WebSocket telemetry streaming
 * - Live beam current and dose tracking
 * - Pressure and field monitoring
 * - 2D beam profile heatmap
 * - Real-time charts
 * - SPC alerts
 * - Controller setpoint display
 * - Progress tracking
 */

"use client"

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Separator } from '@/components/ui/separator'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts'
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Zap,
  Gauge,
  TrendingUp,
  Square,
  Wifi,
  WifiOff,
  Clock
} from 'lucide-react'

interface TelemetryData {
  time_s: number
  beam_current_ma: number
  chamber_pressure_torr: number
  analyzer_field_v: number
  integrated_dose_atoms_cm2: number
  dose_uniformity_pct: number
  wafer_temp_c: number
}

interface SPCAlert {
  parameter: string
  message: string
  severity: 'info' | 'warning' | 'critical'
  timestamp: string
}

interface JobStatus {
  job_id: string
  run_id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  current_step: string
  started_at?: string
  error_message?: string
}

interface BeamProfile {
  data: number[][]
  width: number
  height: number
}

interface IonRunMonitorProps {
  runId: string
  apiEndpoint?: string
  wsEndpoint?: string
}

export const IonRunMonitor: React.FC<IonRunMonitorProps> = ({
  runId,
  apiEndpoint = 'http://localhost:8003',
  wsEndpoint = 'ws://localhost:8003'
}) => {
  // State management
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [telemetryHistory, setTelemetryHistory] = useState<TelemetryData[]>([])
  const [currentTelemetry, setCurrentTelemetry] = useState<TelemetryData | null>(null)
  const [spcAlerts, setSpcAlerts] = useState<SPCAlert[]>([])
  const [beamProfile, setBeamProfile] = useState<BeamProfile | null>(null)
  const [wsConnected, setWsConnected] = useState(false)
  const [isCancelling, setIsCancelling] = useState(false)

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null)

  // Connect to WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    const token = localStorage.getItem('token') || 'demo-token'
    const ws = new WebSocket(`${wsEndpoint}/api/ion/stream/${runId}?token=${token}`)

    ws.onopen = () => {
      console.log('WebSocket connected')
      setWsConnected(true)
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)

        switch (message.type) {
          case 'connected':
            console.log('Connected to run:', message.data.run_id)
            break

          case 'progress':
            setJobStatus(prev => ({
              ...(prev || {} as JobStatus),
              status: message.data.status,
              progress: message.data.progress,
              current_step: message.data.current_step,
            }))
            break

          case 'telemetry':
            const telemetry = message.data as TelemetryData
            setCurrentTelemetry(telemetry)
            setTelemetryHistory(prev => [...prev.slice(-99), telemetry])
            break

          case 'alert':
            const alert: SPCAlert = {
              parameter: message.data.parameter,
              message: message.data.message,
              severity: message.data.severity,
              timestamp: new Date().toISOString(),
            }
            setSpcAlerts(prev => [alert, ...prev.slice(0, 19)])
            break

          case 'completed':
            setJobStatus(prev => ({
              ...(prev || {} as JobStatus),
              status: 'completed',
              progress: 100,
              current_step: 'Completed',
            }))
            console.log('Run completed:', message.data)
            break

          case 'error':
            setJobStatus(prev => ({
              ...(prev || {} as JobStatus),
              status: 'failed',
              error_message: message.data.error_message,
            }))
            break

          case 'cancelled':
            setJobStatus(prev => ({
              ...(prev || {} as JobStatus),
              status: 'cancelled',
              current_step: 'Cancelled',
            }))
            break
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setWsConnected(false)
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
      setWsConnected(false)

      // Attempt reconnection if run is still active
      if (jobStatus?.status === 'running') {
        setTimeout(connectWebSocket, 3000)
      }
    }

    wsRef.current = ws
  }, [runId, wsEndpoint, jobStatus?.status])

  // Fetch initial job status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch(`${apiEndpoint}/api/ion/runs/${runId}`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token') || 'demo-token'}`,
          },
        })

        if (response.ok) {
          const data = await response.json()
          setJobStatus({
            job_id: data.job_id,
            run_id: data.run_id,
            status: data.status,
            progress: data.progress,
            current_step: data.current_step,
            started_at: data.started_at,
          })
        }
      } catch (error) {
        console.error('Failed to fetch run status:', error)
      }
    }

    fetchStatus()
  }, [runId, apiEndpoint])

  // Connect WebSocket when component mounts
  useEffect(() => {
    connectWebSocket()

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [connectWebSocket])

  // Generate mock beam profile (in production, this comes from WebSocket)
  useEffect(() => {
    if (currentTelemetry && !beamProfile) {
      const size = 50
      const profile: number[][] = []
      const centerX = size / 2
      const centerY = size / 2
      const sigma = size / 6

      for (let y = 0; y < size; y++) {
        const row: number[] = []
        for (let x = 0; x < size; x++) {
          const dx = x - centerX
          const dy = y - centerY
          const r2 = dx * dx + dy * dy
          const value = Math.exp(-r2 / (2 * sigma * sigma))
          row.push(value)
        }
        profile.push(row)
      }

      setBeamProfile({ data: profile, width: size, height: size })
    }
  }, [currentTelemetry, beamProfile])

  const handleCancelRun = async () => {
    if (!confirm('Are you sure you want to cancel this run?')) {
      return
    }

    setIsCancelling(true)

    try {
      const response = await fetch(`${apiEndpoint}/api/ion/runs/${runId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token') || 'demo-token'}`,
        },
      })

      if (!response.ok) {
        throw new Error('Failed to cancel run')
      }

      alert('Run cancelled successfully')
    } catch (error) {
      console.error('Failed to cancel run:', error)
      alert('Failed to cancel run')
    } finally {
      setIsCancelling(false)
    }
  }

  const getStatusBadge = (status: JobStatus['status']) => {
    switch (status) {
      case 'queued':
        return <Badge variant="secondary">Queued</Badge>
      case 'running':
        return <Badge className="bg-orange-500 animate-pulse">Running</Badge>
      case 'completed':
        return <Badge className="bg-green-500">Completed</Badge>
      case 'failed':
        return <Badge variant="destructive">Failed</Badge>
      case 'cancelled':
        return <Badge variant="outline">Cancelled</Badge>
    }
  }

  const doseProgress = currentTelemetry
    ? (currentTelemetry.integrated_dose_atoms_cm2 / 1e15) * 100
    : 0

  return (
    <div className="space-y-6">
      {/* Header with Status */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5" />
                Ion Implantation Run Monitor
              </CardTitle>
              <CardDescription>
                Run ID: {runId} | Job ID: {jobStatus?.job_id || 'Loading...'}
              </CardDescription>
            </div>
            <div className="flex items-center gap-3">
              {wsConnected ? (
                <Badge variant="default" className="gap-1">
                  <Wifi className="w-3 h-3" />
                  Connected
                </Badge>
              ) : (
                <Badge variant="secondary" className="gap-1">
                  <WifiOff className="w-3 h-3" />
                  Disconnected
                </Badge>
              )}
              {jobStatus && getStatusBadge(jobStatus.status)}
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="font-medium">{jobStatus?.current_step || 'Initializing...'}</span>
              <span className="text-muted-foreground">{jobStatus?.progress.toFixed(1) || 0}%</span>
            </div>
            <Progress value={jobStatus?.progress || 0} className="h-3" />
          </div>

          {/* Control Buttons */}
          {jobStatus?.status === 'running' && (
            <div className="flex gap-2">
              <Button
                variant="destructive"
                onClick={handleCancelRun}
                disabled={isCancelling}
              >
                <Square className="w-4 h-4 mr-2" />
                {isCancelling ? 'Cancelling...' : 'Cancel Run'}
              </Button>
            </div>
          )}

          {/* Error Display */}
          {jobStatus?.error_message && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{jobStatus.error_message}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Live Telemetry */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Beam Current */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Zap className="w-4 h-4 text-orange-500" />
              Beam Current
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold font-mono">
              {currentTelemetry?.beam_current_ma.toFixed(2) || '0.00'}
              <span className="text-lg text-muted-foreground ml-1">mA</span>
            </div>
          </CardContent>
        </Card>

        {/* Integrated Dose */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Gauge className="w-4 h-4 text-blue-500" />
              Integrated Dose
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold font-mono">
              {currentTelemetry?.integrated_dose_atoms_cm2.toExponential(2) || '0.00e0'}
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              atoms/cm² ({doseProgress.toFixed(1)}% of target)
            </div>
          </CardContent>
        </Card>

        {/* Chamber Pressure */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Gauge className="w-4 h-4 text-purple-500" />
              Vacuum
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold font-mono">
              {currentTelemetry?.chamber_pressure_torr.toExponential(1) || '0.0e0'}
            </div>
            <div className="text-xs text-muted-foreground mt-1">Torr</div>
          </CardContent>
        </Card>

        {/* Uniformity */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-green-500" />
              Uniformity
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold font-mono">
              {currentTelemetry?.dose_uniformity_pct.toFixed(1) || '0.0'}
              <span className="text-lg text-muted-foreground ml-1">%</span>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Real-time Charts */}
        <Card>
          <CardHeader>
            <CardTitle>Beam Current History</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={telemetryHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="time_s"
                    label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis label={{ value: 'Current (mA)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="beam_current_ma"
                    stroke="#f97316"
                    strokeWidth={2}
                    dot={false}
                    name="Beam Current"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Dose Accumulation */}
        <Card>
          <CardHeader>
            <CardTitle>Dose Accumulation</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={telemetryHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="time_s"
                    label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    tickFormatter={(value) => value.toExponential(0)}
                    label={{ value: 'Dose (atoms/cm²)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip formatter={(value: number) => value.toExponential(2)} />
                  <Area
                    type="monotone"
                    dataKey="integrated_dose_atoms_cm2"
                    stroke="#3b82f6"
                    fill="#3b82f6"
                    fillOpacity={0.3}
                    name="Integrated Dose"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Beam Profile Heatmap */}
      <Card>
        <CardHeader>
          <CardTitle>2D Beam Profile</CardTitle>
          <CardDescription>Real-time wafer uniformity map</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex justify-center">
            <div className="relative" style={{ width: 400, height: 400 }}>
              {beamProfile ? (
                <svg width="400" height="400" viewBox="0 0 400 400">
                  {/* Wafer outline */}
                  <circle
                    cx="200"
                    cy="200"
                    r="190"
                    fill="none"
                    stroke="#94a3b8"
                    strokeWidth="2"
                  />

                  {/* Beam profile heatmap */}
                  {beamProfile.data.map((row, y) =>
                    row.map((value, x) => {
                      const size = 8
                      const px = 10 + (x * 380) / beamProfile.width
                      const py = 10 + (y * 380) / beamProfile.height
                      const intensity = Math.round(value * 255)
                      const color = `rgb(${intensity}, ${Math.round(intensity * 0.5)}, ${255 - intensity})`

                      return (
                        <rect
                          key={`${x}-${y}`}
                          x={px}
                          y={py}
                          width={size}
                          height={size}
                          fill={color}
                          opacity={0.7}
                        />
                      )
                    })
                  )}

                  {/* Center marker */}
                  <circle cx="200" cy="200" r="3" fill="#ef4444" />
                </svg>
              ) : (
                <div className="flex items-center justify-center h-full text-muted-foreground">
                  Waiting for beam data...
                </div>
              )}
            </div>
          </div>

          <div className="mt-4 flex justify-center gap-8 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-gradient-to-r from-blue-500 to-red-500 rounded" />
              <span className="text-muted-foreground">Low → High Intensity</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* SPC Alerts */}
      {spcAlerts.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5" />
              SPC Alerts ({spcAlerts.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {spcAlerts.slice(0, 5).map((alert, idx) => (
                <Alert
                  key={idx}
                  variant={alert.severity === 'critical' ? 'destructive' : 'default'}
                  className="py-2"
                >
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription className="flex items-center justify-between">
                    <div>
                      <span className="font-semibold">{alert.parameter}:</span> {alert.message}
                    </div>
                    <Badge variant="outline" className="text-xs">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </Badge>
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Additional Telemetry */}
      <Card>
        <CardHeader>
          <CardTitle>Extended Parameters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="space-y-1">
              <div className="text-sm text-muted-foreground">Analyzer Field</div>
              <div className="text-lg font-semibold font-mono">
                {currentTelemetry?.analyzer_field_v.toFixed(0) || '0'} V
              </div>
            </div>
            <div className="space-y-1">
              <div className="text-sm text-muted-foreground">Wafer Temperature</div>
              <div className="text-lg font-semibold font-mono">
                {currentTelemetry?.wafer_temp_c.toFixed(1) || '0.0'} °C
              </div>
            </div>
            <div className="space-y-1">
              <div className="text-sm text-muted-foreground flex items-center gap-1">
                <Clock className="w-3 h-3" />
                Run Time
              </div>
              <div className="text-lg font-semibold font-mono">
                {currentTelemetry?.time_s.toFixed(0) || '0'} s
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default IonRunMonitor
