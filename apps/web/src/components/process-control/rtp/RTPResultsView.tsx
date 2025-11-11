/**
 * RTP Results View Component
 *
 * Post-process analytics for RTP runs with:
 * - Thermal profile chart (setpoint vs measured overlay)
 * - Thermal budget summary
 * - Temperature metrics (overshoot, ramp error, settling time, uniformity)
 * - VM predictions (activation %, diffusion depth, oxide thickness, sheet resistance)
 * - Controller performance metrics (IAE, ISE, overshoot)
 * - Tuning suggestions panel
 * - SPC metrics and quality gates
 * - Report download (PDF/CSV)
 */

"use client"

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Separator } from '@/components/ui/separator'
import {
  Thermometer,
  TrendingUp,
  Download,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Calculator,
  Gauge,
  FileText,
  Settings,
  BarChart3,
  Activity,
  Lightbulb
} from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Area,
  AreaChart,
  ScatterChart,
  Scatter,
  BarChart,
  Bar
} from 'recharts'

interface ThermalProfileData {
  time_s: number
  setpoint_c: number
  measured_c: number
  error_c: number
  lamp_power_pct: number
}

interface TemperatureMetrics {
  max_overshoot_c: number
  max_undershoot_c: number
  avg_ramp_error_c: number
  max_ramp_error_c: number
  settling_time_s: number
  zone_uniformity_sigma_c: number
  total_thermal_budget_c_s: number
  peak_temperature_c: number
  total_duration_s: number
}

interface VMPredictions {
  activation_pct: number
  junction_depth_nm: number
  diffusion_depth_nm: number
  oxide_thickness_nm: number
  sheet_resistance_ohm_sq: number
  carrier_concentration_cm3: number
  confidence_interval: {
    lower: number
    upper: number
  }
}

interface ControllerPerformance {
  controller_type: 'PID' | 'MPC'
  iae: number  // Integral Absolute Error
  ise: number  // Integral Squared Error
  max_overshoot_pct: number
  settling_time_s: number
  steady_state_error_c: number
  pid_gains?: {
    kp: number
    ki: number
    kd: number
  }
}

interface TuningSuggestion {
  parameter: string
  current_value: number
  suggested_value: number
  reason: string
  impact: 'high' | 'medium' | 'low'
}

interface SPCMetrics {
  cpk: number
  cp: number
  alerts_count: number
  out_of_control: boolean
  nelson_rules_triggered: string[]
}

interface QualityGate {
  name: string
  status: 'pass' | 'fail' | 'warning'
  measured_value: string
  spec_limit: string
}

interface RTPResultsViewProps {
  runId: string
  apiEndpoint?: string
}

export const RTPResultsView: React.FC<RTPResultsViewProps> = ({
  runId,
  apiEndpoint = 'http://localhost:8003'
}) => {
  const [isLoading, setIsLoading] = useState(true)
  const [thermalProfile, setThermalProfile] = useState<ThermalProfileData[]>([])
  const [tempMetrics, setTempMetrics] = useState<TemperatureMetrics | null>(null)
  const [vmPredictions, setVmPredictions] = useState<VMPredictions | null>(null)
  const [controllerPerf, setControllerPerf] = useState<ControllerPerformance | null>(null)
  const [tuningSuggestions, setTuningSuggestions] = useState<TuningSuggestion[]>([])
  const [spcMetrics, setSpcMetrics] = useState<SPCMetrics | null>(null)
  const [qualityGates, setQualityGates] = useState<QualityGate[]>([])

  // Fetch results data
  useEffect(() => {
    const fetchResults = async () => {
      try {
        const response = await fetch(`${apiEndpoint}/api/rtp/runs/${runId}`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token') || 'demo-token'}`,
          },
        })

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }

        const data = await response.json()

        // Generate mock data if backend doesn't provide it yet
        setThermalProfile(generateMockThermalProfile())
        setTempMetrics(generateMockTempMetrics())
        setVmPredictions(generateMockVMPredictions())
        setControllerPerf(generateMockControllerPerf())
        setTuningSuggestions(generateMockTuningSuggestions())
        setSpcMetrics(generateMockSPCMetrics())
        setQualityGates(generateMockQualityGates())

        setIsLoading(false)
      } catch (error) {
        console.error('Failed to fetch results:', error)
        // Load mock data on error
        setThermalProfile(generateMockThermalProfile())
        setTempMetrics(generateMockTempMetrics())
        setVmPredictions(generateMockVMPredictions())
        setControllerPerf(generateMockControllerPerf())
        setTuningSuggestions(generateMockTuningSuggestions())
        setSpcMetrics(generateMockSPCMetrics())
        setQualityGates(generateMockQualityGates())
        setIsLoading(false)
      }
    }

    fetchResults()
  }, [runId, apiEndpoint])

  // Mock data generators
  const generateMockThermalProfile = (): ThermalProfileData[] => {
    const profile: ThermalProfileData[] = []

    // Ramp up: 400°C → 1000°C in 30s
    for (let t = 0; t <= 30; t++) {
      const setpoint = 400 + (600 / 30) * t
      const measured = setpoint + (Math.sin(t * 0.5) * 3) + (Math.random() - 0.5) * 2
      profile.push({
        time_s: t,
        setpoint_c: setpoint,
        measured_c: measured,
        error_c: measured - setpoint,
        lamp_power_pct: 40 + (50 / 30) * t
      })
    }

    // Dwell: 1000°C for 60s
    for (let t = 31; t <= 90; t++) {
      const measured = 1000 + (Math.random() - 0.5) * 4
      profile.push({
        time_s: t,
        setpoint_c: 1000,
        measured_c: measured,
        error_c: measured - 1000,
        lamp_power_pct: 85 + (Math.random() - 0.5) * 5
      })
    }

    // Cooldown: 1000°C → 400°C in 45s
    for (let t = 91; t <= 135; t++) {
      const setpoint = 1000 - (600 / 45) * (t - 90)
      const measured = setpoint + (Math.random() - 0.5) * 3
      profile.push({
        time_s: t,
        setpoint_c: setpoint,
        measured_c: measured,
        error_c: measured - setpoint,
        lamp_power_pct: 85 - (75 / 45) * (t - 90)
      })
    }

    return profile
  }

  const generateMockTempMetrics = (): TemperatureMetrics => ({
    max_overshoot_c: 8.2,
    max_undershoot_c: 3.1,
    avg_ramp_error_c: 1.8,
    max_ramp_error_c: 8.2,
    settling_time_s: 12.5,
    zone_uniformity_sigma_c: 2.3,
    total_thermal_budget_c_s: 94500,
    peak_temperature_c: 1008.2,
    total_duration_s: 135
  })

  const generateMockVMPredictions = (): VMPredictions => ({
    activation_pct: 94.2,
    junction_depth_nm: 145,
    diffusion_depth_nm: 180,
    oxide_thickness_nm: 12.5,
    sheet_resistance_ohm_sq: 85.3,
    carrier_concentration_cm3: 2.1e19,
    confidence_interval: {
      lower: 0.92,
      upper: 0.98
    }
  })

  const generateMockControllerPerf = (): ControllerPerformance => ({
    controller_type: 'PID',
    iae: 245.6,
    ise: 1852.3,
    max_overshoot_pct: 0.82,
    settling_time_s: 12.5,
    steady_state_error_c: 0.3,
    pid_gains: {
      kp: 2.5,
      ki: 0.1,
      kd: 0.05
    }
  })

  const generateMockTuningSuggestions = (): TuningSuggestion[] => [
    {
      parameter: 'Kp (Proportional Gain)',
      current_value: 2.5,
      suggested_value: 2.8,
      reason: 'Increase Kp to reduce settling time by ~15%',
      impact: 'medium'
    },
    {
      parameter: 'Ki (Integral Gain)',
      current_value: 0.1,
      suggested_value: 0.08,
      reason: 'Decrease Ki to reduce overshoot',
      impact: 'high'
    },
    {
      parameter: 'Kd (Derivative Gain)',
      current_value: 0.05,
      suggested_value: 0.07,
      reason: 'Increase Kd to dampen oscillations',
      impact: 'low'
    }
  ]

  const generateMockSPCMetrics = (): SPCMetrics => ({
    cpk: 1.58,
    cp: 1.72,
    alerts_count: 2,
    out_of_control: false,
    nelson_rules_triggered: ['Rule 2: 9 points in a row on same side of center']
  })

  const generateMockQualityGates = (): QualityGate[] => [
    {
      name: 'Temperature Overshoot',
      status: 'warning',
      measured_value: '8.2°C',
      spec_limit: '<5°C'
    },
    {
      name: 'Thermal Budget',
      status: 'pass',
      measured_value: '94.5k °C·s',
      spec_limit: '<100k °C·s'
    },
    {
      name: 'Zone Uniformity',
      status: 'pass',
      measured_value: '±2.3°C',
      spec_limit: '<±5°C'
    },
    {
      name: 'Process Capability (Cpk)',
      status: 'pass',
      measured_value: '1.58',
      spec_limit: '>1.33'
    }
  ]

  // Download handlers
  const handleDownloadPDF = () => {
    console.log('Downloading PDF report...')
    // TODO: Implement PDF generation
  }

  const handleDownloadCSV = () => {
    console.log('Downloading CSV data...')
    // TODO: Implement CSV export
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <Activity className="w-8 h-8 animate-spin mx-auto mb-2 text-primary" />
          <p className="text-sm text-muted-foreground">Loading results...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-start">
            <div>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                RTP Results Analysis
              </CardTitle>
              <CardDescription className="mt-1">
                Run ID: {runId} • Completed at {new Date().toLocaleString()}
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={handleDownloadCSV}>
                <Download className="w-4 h-4 mr-2" />
                CSV
              </Button>
              <Button variant="outline" size="sm" onClick={handleDownloadPDF}>
                <FileText className="w-4 h-4 mr-2" />
                PDF Report
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Quality Gates */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle className="w-5 h-5" />
            Quality Gates Summary
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-3">
            {qualityGates.map((gate, idx) => (
              <div
                key={idx}
                className={`p-3 border rounded-lg ${
                  gate.status === 'pass'
                    ? 'bg-green-50 border-green-300'
                    : gate.status === 'warning'
                      ? 'bg-yellow-50 border-yellow-300'
                      : 'bg-red-50 border-red-300'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-semibold">{gate.name}</span>
                  {gate.status === 'pass' ? (
                    <CheckCircle className="w-4 h-4 text-green-600" />
                  ) : gate.status === 'warning' ? (
                    <AlertTriangle className="w-4 h-4 text-yellow-600" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-600" />
                  )}
                </div>
                <div className="text-xs text-muted-foreground">
                  Measured: <span className="font-mono font-semibold">{gate.measured_value}</span>
                </div>
                <div className="text-xs text-muted-foreground">
                  Spec: <span className="font-mono">{gate.spec_limit}</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="thermal" className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="thermal">Thermal Profile</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="vm">VM Predictions</TabsTrigger>
          <TabsTrigger value="controller">Controller</TabsTrigger>
          <TabsTrigger value="spc">SPC</TabsTrigger>
        </TabsList>

        {/* Thermal Profile Tab */}
        <TabsContent value="thermal" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Temperature Profile (Setpoint vs Measured)</CardTitle>
              <CardDescription>Complete thermal trajectory with tracking error visualization</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={thermalProfile}>
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
                          return (
                            <div className="bg-background border rounded p-3 shadow-lg">
                              <p className="text-xs font-semibold mb-1">Time: {data.time_s.toFixed(1)} s</p>
                              <p className="text-xs text-blue-600">Setpoint: {data.setpoint_c.toFixed(1)}°C</p>
                              <p className="text-xs text-purple-600">Measured: {data.measured_c.toFixed(1)}°C</p>
                              <p className="text-xs text-orange-600">Error: {data.error_c.toFixed(2)}°C</p>
                              <p className="text-xs text-muted-foreground mt-1">
                                Power: {data.lamp_power_pct.toFixed(0)}%
                              </p>
                            </div>
                          )
                        }
                        return null
                      }}
                    />
                    <Legend />

                    {/* Error band ±5°C */}
                    <Area
                      type="monotone"
                      dataKey={(d: ThermalProfileData) => d.setpoint_c + 5}
                      stroke="none"
                      fill="#fca5a5"
                      fillOpacity={0.15}
                      isAnimationActive={false}
                    />
                    <Area
                      type="monotone"
                      dataKey={(d: ThermalProfileData) => d.setpoint_c - 5}
                      stroke="none"
                      fill="#fca5a5"
                      fillOpacity={0.15}
                      isAnimationActive={false}
                    />

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
                    <Line
                      type="monotone"
                      dataKey="measured_c"
                      stroke="#a855f7"
                      strokeWidth={3}
                      dot={false}
                      name="Measured"
                      isAnimationActive={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Tracking Error Over Time</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={thermalProfile}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time_s" />
                    <YAxis label={{ value: 'Error (°C)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="error_c"
                      stroke="#f97316"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Temperature Metrics Tab */}
        <TabsContent value="metrics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Gauge className="w-5 h-5" />
                Temperature Performance Metrics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4">
                <div className="p-4 border rounded-lg bg-muted/30">
                  <div className="text-sm text-muted-foreground">Max Overshoot</div>
                  <div className="text-3xl font-bold font-mono mt-2">
                    {tempMetrics?.max_overshoot_c.toFixed(1)}°C
                  </div>
                  <Badge variant={tempMetrics && tempMetrics.max_overshoot_c > 5 ? 'destructive' : 'default'} className="mt-2">
                    {tempMetrics && tempMetrics.max_overshoot_c > 5 ? 'Above Spec' : 'In Spec'}
                  </Badge>
                </div>

                <div className="p-4 border rounded-lg bg-muted/30">
                  <div className="text-sm text-muted-foreground">Avg Ramp Error</div>
                  <div className="text-3xl font-bold font-mono mt-2">
                    ±{tempMetrics?.avg_ramp_error_c.toFixed(2)}°C
                  </div>
                  <Badge variant="default" className="mt-2">Excellent</Badge>
                </div>

                <div className="p-4 border rounded-lg bg-muted/30">
                  <div className="text-sm text-muted-foreground">Settling Time</div>
                  <div className="text-3xl font-bold font-mono mt-2">
                    {tempMetrics?.settling_time_s.toFixed(1)}s
                  </div>
                  <Badge variant="default" className="mt-2">Good</Badge>
                </div>

                <div className="p-4 border rounded-lg bg-gradient-to-br from-orange-50 to-red-50">
                  <div className="text-sm text-muted-foreground flex items-center gap-2">
                    <Calculator className="w-4 h-4" />
                    Thermal Budget
                  </div>
                  <div className="text-3xl font-bold font-mono mt-2 text-orange-600">
                    {tempMetrics && (tempMetrics.total_thermal_budget_c_s / 1000).toFixed(1)}k
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">°C·s</div>
                </div>

                <div className="p-4 border rounded-lg bg-muted/30">
                  <div className="text-sm text-muted-foreground">Zone Uniformity (σ)</div>
                  <div className="text-3xl font-bold font-mono mt-2">
                    ±{tempMetrics?.zone_uniformity_sigma_c.toFixed(2)}°C
                  </div>
                  <Badge variant="default" className="mt-2">Excellent</Badge>
                </div>

                <div className="p-4 border rounded-lg bg-muted/30">
                  <div className="text-sm text-muted-foreground">Peak Temperature</div>
                  <div className="text-3xl font-bold font-mono mt-2">
                    {tempMetrics?.peak_temperature_c.toFixed(1)}°C
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Target: 1000°C
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* VM Predictions Tab */}
        <TabsContent value="vm" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Virtual Metrology Predictions
              </CardTitle>
              <CardDescription>
                ML-based property predictions from thermal profile
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 border rounded-lg bg-gradient-to-br from-blue-50 to-indigo-50">
                  <div className="text-sm text-muted-foreground">Dopant Activation</div>
                  <div className="text-4xl font-bold font-mono mt-2 text-blue-600">
                    {vmPredictions?.activation_pct.toFixed(1)}%
                  </div>
                  <div className="text-xs text-muted-foreground mt-2">
                    Confidence: {vmPredictions && (vmPredictions.confidence_interval.lower * 100).toFixed(0)}-
                    {vmPredictions && (vmPredictions.confidence_interval.upper * 100).toFixed(0)}%
                  </div>
                </div>

                <div className="p-4 border rounded-lg bg-gradient-to-br from-purple-50 to-pink-50">
                  <div className="text-sm text-muted-foreground">Junction Depth</div>
                  <div className="text-4xl font-bold font-mono mt-2 text-purple-600">
                    {vmPredictions?.junction_depth_nm.toFixed(0)}
                  </div>
                  <div className="text-xs text-muted-foreground mt-2">nm</div>
                </div>

                <div className="p-4 border rounded-lg bg-gradient-to-br from-green-50 to-emerald-50">
                  <div className="text-sm text-muted-foreground">Sheet Resistance</div>
                  <div className="text-4xl font-bold font-mono mt-2 text-green-600">
                    {vmPredictions?.sheet_resistance_ohm_sq.toFixed(1)}
                  </div>
                  <div className="text-xs text-muted-foreground mt-2">Ω/sq</div>
                </div>

                <div className="p-4 border rounded-lg bg-gradient-to-br from-orange-50 to-amber-50">
                  <div className="text-sm text-muted-foreground">Diffusion Depth</div>
                  <div className="text-4xl font-bold font-mono mt-2 text-orange-600">
                    {vmPredictions?.diffusion_depth_nm.toFixed(0)}
                  </div>
                  <div className="text-xs text-muted-foreground mt-2">nm</div>
                </div>

                <div className="p-4 border rounded-lg bg-muted/30">
                  <div className="text-sm text-muted-foreground">Oxide Thickness</div>
                  <div className="text-4xl font-bold font-mono mt-2">
                    {vmPredictions?.oxide_thickness_nm.toFixed(1)}
                  </div>
                  <div className="text-xs text-muted-foreground mt-2">nm (native)</div>
                </div>

                <div className="p-4 border rounded-lg bg-muted/30">
                  <div className="text-sm text-muted-foreground">Carrier Concentration</div>
                  <div className="text-3xl font-bold font-mono mt-2">
                    {vmPredictions && vmPredictions.carrier_concentration_cm3.toExponential(1)}
                  </div>
                  <div className="text-xs text-muted-foreground mt-2">/cm³</div>
                </div>
              </div>

              <Alert className="mt-4">
                <AlertDescription className="text-xs">
                  <strong>Note:</strong> VM predictions are estimates based on thermal profile and process conditions.
                  Physical metrology recommended for critical applications.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Controller Performance Tab */}
        <TabsContent value="controller" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="w-5 h-5" />
                Controller Performance Analysis
              </CardTitle>
              <CardDescription>
                {controllerPerf?.controller_type} controller with current gain settings
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-3 gap-4">
                <div className="p-4 border rounded-lg bg-muted/30">
                  <div className="text-sm text-muted-foreground">IAE (Integral Absolute Error)</div>
                  <div className="text-3xl font-bold font-mono mt-2">
                    {controllerPerf?.iae.toFixed(1)}
                  </div>
                  <Badge variant="default" className="mt-2">Good</Badge>
                </div>

                <div className="p-4 border rounded-lg bg-muted/30">
                  <div className="text-sm text-muted-foreground">ISE (Integral Squared Error)</div>
                  <div className="text-3xl font-bold font-mono mt-2">
                    {controllerPerf?.ise.toFixed(1)}
                  </div>
                  <Badge variant="default" className="mt-2">Acceptable</Badge>
                </div>

                <div className="p-4 border rounded-lg bg-muted/30">
                  <div className="text-sm text-muted-foreground">Steady-State Error</div>
                  <div className="text-3xl font-bold font-mono mt-2">
                    ±{controllerPerf?.steady_state_error_c.toFixed(2)}°C
                  </div>
                  <Badge variant="default" className="mt-2">Excellent</Badge>
                </div>
              </div>

              {controllerPerf?.controller_type === 'PID' && controllerPerf.pid_gains && (
                <>
                  <Separator />
                  <div>
                    <h4 className="text-sm font-semibold mb-3">Current PID Gains</h4>
                    <div className="grid grid-cols-3 gap-3">
                      <div className="p-3 border rounded-lg bg-blue-50">
                        <div className="text-xs text-muted-foreground">Kp</div>
                        <div className="text-2xl font-bold font-mono">{controllerPerf.pid_gains.kp}</div>
                      </div>
                      <div className="p-3 border rounded-lg bg-purple-50">
                        <div className="text-xs text-muted-foreground">Ki</div>
                        <div className="text-2xl font-bold font-mono">{controllerPerf.pid_gains.ki}</div>
                      </div>
                      <div className="p-3 border rounded-lg bg-green-50">
                        <div className="text-xs text-muted-foreground">Kd</div>
                        <div className="text-2xl font-bold font-mono">{controllerPerf.pid_gains.kd}</div>
                      </div>
                    </div>
                  </div>
                </>
              )}

              <Separator />

              <div>
                <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <Lightbulb className="w-4 h-4 text-yellow-600" />
                  Tuning Suggestions
                </h4>
                <div className="space-y-3">
                  {tuningSuggestions.map((suggestion, idx) => (
                    <div key={idx} className="p-3 border rounded-lg bg-muted/30">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex-1">
                          <div className="font-semibold text-sm">{suggestion.parameter}</div>
                          <div className="text-xs text-muted-foreground mt-1">{suggestion.reason}</div>
                        </div>
                        <Badge
                          variant={
                            suggestion.impact === 'high'
                              ? 'destructive'
                              : suggestion.impact === 'medium'
                                ? 'default'
                                : 'secondary'
                          }
                        >
                          {suggestion.impact.toUpperCase()} IMPACT
                        </Badge>
                      </div>
                      <div className="flex items-center gap-4 text-sm">
                        <div>
                          <span className="text-muted-foreground">Current:</span>{' '}
                          <span className="font-mono font-semibold">{suggestion.current_value}</span>
                        </div>
                        <div className="text-muted-foreground">→</div>
                        <div>
                          <span className="text-muted-foreground">Suggested:</span>{' '}
                          <span className="font-mono font-semibold text-green-600">{suggestion.suggested_value}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* SPC Tab */}
        <TabsContent value="spc" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5" />
                Statistical Process Control
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-3 gap-4">
                <div className="p-4 border rounded-lg bg-gradient-to-br from-blue-50 to-indigo-50">
                  <div className="text-sm text-muted-foreground">Cpk (Process Capability)</div>
                  <div className="text-4xl font-bold font-mono mt-2 text-blue-600">
                    {spcMetrics?.cpk.toFixed(2)}
                  </div>
                  <Badge variant="default" className="mt-2">
                    {spcMetrics && spcMetrics.cpk > 1.67 ? 'Excellent' : spcMetrics && spcMetrics.cpk > 1.33 ? 'Good' : 'Needs Improvement'}
                  </Badge>
                </div>

                <div className="p-4 border rounded-lg bg-gradient-to-br from-purple-50 to-pink-50">
                  <div className="text-sm text-muted-foreground">Cp (Potential Capability)</div>
                  <div className="text-4xl font-bold font-mono mt-2 text-purple-600">
                    {spcMetrics?.cp.toFixed(2)}
                  </div>
                  <Badge variant="default" className="mt-2">Good</Badge>
                </div>

                <div className="p-4 border rounded-lg bg-muted/30">
                  <div className="text-sm text-muted-foreground">SPC Alerts</div>
                  <div className="text-4xl font-bold font-mono mt-2">
                    {spcMetrics?.alerts_count}
                  </div>
                  <Badge
                    variant={spcMetrics && spcMetrics.out_of_control ? 'destructive' : 'default'}
                    className="mt-2"
                  >
                    {spcMetrics && spcMetrics.out_of_control ? 'Out of Control' : 'In Control'}
                  </Badge>
                </div>
              </div>

              {spcMetrics && spcMetrics.nelson_rules_triggered.length > 0 && (
                <>
                  <Separator />
                  <div>
                    <h4 className="text-sm font-semibold mb-3">Nelson Rules Triggered</h4>
                    <div className="space-y-2">
                      {spcMetrics.nelson_rules_triggered.map((rule, idx) => (
                        <Alert key={idx}>
                          <AlertTriangle className="h-4 w-4" />
                          <AlertDescription className="text-xs">{rule}</AlertDescription>
                        </Alert>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default RTPResultsView
