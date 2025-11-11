/**
 * RTP Ramp Editor Component
 *
 * Thermal profile editor with:
 * - Segment list management (add/remove/reorder)
 * - Per-segment configuration (target temp, duration, ramp rate)
 * - Real-time thermal profile visualization
 * - Constraint validation (max ramp rates)
 * - Gas flow configuration (N2, O2)
 * - Chamber pressure and emissivity settings
 * - Controller selection (PID/MPC)
 * - PID gain tuning interface
 * - Thermal budget calculator
 * - API integration with recipe submission
 */

"use client"

import React, { useState, useMemo, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Separator } from '@/components/ui/separator'
import {
  Plus,
  Trash2,
  ChevronUp,
  ChevronDown,
  AlertTriangle,
  CheckCircle,
  Thermometer,
  Clock,
  TrendingUp,
  Wind,
  Gauge,
  Settings,
  Play,
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
  AreaChart
} from 'recharts'

interface ThermalSegment {
  id: string
  target_temp_c: number
  duration_s: number
  ramp_rate_c_per_s?: number
  dwell_time_s?: number
  segment_type: 'ramp' | 'dwell' | 'cooldown'
}

interface GasConfig {
  n2_flow_slm: number
  o2_flow_slm: number
  ambient: 'N2' | 'O2' | 'Forming Gas' | 'Vacuum'
}

interface ControllerConfig {
  type: 'PID' | 'MPC'
  pid_gains?: {
    kp: number
    ki: number
    kd: number
  }
  mpc_horizon?: number
}

interface RTPRecipe {
  recipe_name: string
  wafer_id: string
  lot_id: string
  segments: ThermalSegment[]
  gas_config: GasConfig
  chamber_pressure_torr: number
  emissivity: number
  controller: ControllerConfig
  max_ramp_up_rate: number
  max_ramp_down_rate: number
  operator_notes?: string
}

interface ValidationError {
  segment_id?: string
  field: string
  message: string
  severity: 'error' | 'warning'
}

interface RTPRampEditorProps {
  initialRecipe?: Partial<RTPRecipe>
  onSubmit?: (recipe: RTPRecipe) => void
  apiEndpoint?: string
}

const CONSTRAINTS = {
  MAX_RAMP_UP_RATE: 100, // °C/s
  MAX_RAMP_DOWN_RATE: 50, // °C/s
  MIN_TEMP: 400, // °C
  MAX_TEMP: 1200, // °C
  MIN_DURATION: 0, // seconds
  MAX_DURATION: 300, // seconds
  MAX_THERMAL_BUDGET: 100000, // °C·s
}

const DEFAULT_SEGMENT: ThermalSegment = {
  id: '',
  target_temp_c: 800,
  duration_s: 30,
  ramp_rate_c_per_s: 50,
  segment_type: 'ramp'
}

const DEFAULT_GAS_CONFIG: GasConfig = {
  n2_flow_slm: 10,
  o2_flow_slm: 0,
  ambient: 'N2'
}

const DEFAULT_CONTROLLER: ControllerConfig = {
  type: 'PID',
  pid_gains: {
    kp: 2.5,
    ki: 0.1,
    kd: 0.05
  }
}

export const RTPRampEditor: React.FC<RTPRampEditorProps> = ({
  initialRecipe,
  onSubmit,
  apiEndpoint = 'http://localhost:8003'
}) => {
  const [recipe, setRecipe] = useState<RTPRecipe>({
    recipe_name: initialRecipe?.recipe_name || '',
    wafer_id: initialRecipe?.wafer_id || '',
    lot_id: initialRecipe?.lot_id || '',
    segments: initialRecipe?.segments || [
      { ...DEFAULT_SEGMENT, id: '1', target_temp_c: 400, duration_s: 10, segment_type: 'ramp' },
      { ...DEFAULT_SEGMENT, id: '2', target_temp_c: 1000, duration_s: 60, segment_type: 'dwell' },
      { ...DEFAULT_SEGMENT, id: '3', target_temp_c: 400, duration_s: 30, segment_type: 'cooldown' }
    ],
    gas_config: initialRecipe?.gas_config || DEFAULT_GAS_CONFIG,
    chamber_pressure_torr: initialRecipe?.chamber_pressure_torr || 1.0,
    emissivity: initialRecipe?.emissivity || 0.7,
    controller: initialRecipe?.controller || DEFAULT_CONTROLLER,
    max_ramp_up_rate: CONSTRAINTS.MAX_RAMP_UP_RATE,
    max_ramp_down_rate: CONSTRAINTS.MAX_RAMP_DOWN_RATE,
    operator_notes: initialRecipe?.operator_notes || ''
  })

  const [validationErrors, setValidationErrors] = useState<ValidationError[]>([])
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitSuccess, setSubmitSuccess] = useState(false)
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null)

  // Calculate thermal budget
  const thermalBudget = useMemo(() => {
    let budget = 0
    let currentTemp = recipe.segments[0]?.target_temp_c || 400

    recipe.segments.forEach(segment => {
      const avgTemp = (currentTemp + segment.target_temp_c) / 2
      budget += avgTemp * segment.duration_s
      currentTemp = segment.target_temp_c
    })

    return budget
  }, [recipe.segments])

  // Generate thermal profile data for visualization
  const profileData = useMemo(() => {
    const data: Array<{ time_s: number; temp_c: number; segment: string }> = []
    let cumulativeTime = 0
    let currentTemp = recipe.segments[0]?.target_temp_c || 400

    recipe.segments.forEach((segment, idx) => {
      const rampTime = segment.ramp_rate_c_per_s
        ? Math.abs(segment.target_temp_c - currentTemp) / segment.ramp_rate_c_per_s
        : 0

      // Add ramp points
      if (rampTime > 0) {
        const steps = Math.ceil(rampTime)
        for (let i = 0; i <= steps; i++) {
          const t = (i / steps) * rampTime
          const temp = currentTemp + ((segment.target_temp_c - currentTemp) * i / steps)
          data.push({
            time_s: cumulativeTime + t,
            temp_c: temp,
            segment: `Segment ${idx + 1}`
          })
        }
        cumulativeTime += rampTime
        currentTemp = segment.target_temp_c
      }

      // Add dwell points
      if (segment.duration_s > 0) {
        data.push({
          time_s: cumulativeTime,
          temp_c: segment.target_temp_c,
          segment: `Segment ${idx + 1}`
        })
        data.push({
          time_s: cumulativeTime + segment.duration_s,
          temp_c: segment.target_temp_c,
          segment: `Segment ${idx + 1}`
        })
        cumulativeTime += segment.duration_s
      }
    })

    return data
  }, [recipe.segments])

  // Validate recipe
  const validateRecipe = useCallback(() => {
    const errors: ValidationError[] = []

    // Validate basic fields
    if (!recipe.recipe_name.trim()) {
      errors.push({ field: 'recipe_name', message: 'Recipe name is required', severity: 'error' })
    }
    if (!recipe.wafer_id.trim()) {
      errors.push({ field: 'wafer_id', message: 'Wafer ID is required', severity: 'error' })
    }
    if (!recipe.lot_id.trim()) {
      errors.push({ field: 'lot_id', message: 'Lot ID is required', severity: 'error' })
    }

    // Validate segments
    if (recipe.segments.length === 0) {
      errors.push({ field: 'segments', message: 'At least one segment is required', severity: 'error' })
    }

    let prevTemp = recipe.segments[0]?.target_temp_c || 400
    recipe.segments.forEach((segment, idx) => {
      const segmentLabel = `Segment ${idx + 1}`

      // Temperature range
      if (segment.target_temp_c < CONSTRAINTS.MIN_TEMP || segment.target_temp_c > CONSTRAINTS.MAX_TEMP) {
        errors.push({
          segment_id: segment.id,
          field: 'target_temp_c',
          message: `${segmentLabel}: Temperature must be between ${CONSTRAINTS.MIN_TEMP}-${CONSTRAINTS.MAX_TEMP}°C`,
          severity: 'error'
        })
      }

      // Duration range
      if (segment.duration_s < CONSTRAINTS.MIN_DURATION || segment.duration_s > CONSTRAINTS.MAX_DURATION) {
        errors.push({
          segment_id: segment.id,
          field: 'duration_s',
          message: `${segmentLabel}: Duration must be between ${CONSTRAINTS.MIN_DURATION}-${CONSTRAINTS.MAX_DURATION}s`,
          severity: 'error'
        })
      }

      // Ramp rate constraints
      if (segment.ramp_rate_c_per_s) {
        const tempDiff = segment.target_temp_c - prevTemp
        const isHeating = tempDiff > 0
        const maxRate = isHeating ? recipe.max_ramp_up_rate : recipe.max_ramp_down_rate

        if (Math.abs(segment.ramp_rate_c_per_s) > maxRate) {
          errors.push({
            segment_id: segment.id,
            field: 'ramp_rate_c_per_s',
            message: `${segmentLabel}: Ramp rate exceeds ${maxRate}°C/s limit for ${isHeating ? 'heating' : 'cooling'}`,
            severity: 'error'
          })
        }
      }

      prevTemp = segment.target_temp_c
    })

    // Thermal budget check
    if (thermalBudget > CONSTRAINTS.MAX_THERMAL_BUDGET) {
      errors.push({
        field: 'thermal_budget',
        message: `Thermal budget ${thermalBudget.toFixed(0)}°C·s exceeds limit of ${CONSTRAINTS.MAX_THERMAL_BUDGET}°C·s`,
        severity: 'warning'
      })
    }

    // Gas flow validation
    if (recipe.gas_config.n2_flow_slm < 0 || recipe.gas_config.n2_flow_slm > 20) {
      errors.push({ field: 'n2_flow', message: 'N2 flow must be 0-20 SLM', severity: 'error' })
    }
    if (recipe.gas_config.o2_flow_slm < 0 || recipe.gas_config.o2_flow_slm > 5) {
      errors.push({ field: 'o2_flow', message: 'O2 flow must be 0-5 SLM', severity: 'error' })
    }

    // Pressure validation
    if (recipe.chamber_pressure_torr < 0.1 || recipe.chamber_pressure_torr > 760) {
      errors.push({ field: 'pressure', message: 'Pressure must be 0.1-760 Torr', severity: 'error' })
    }

    // Emissivity validation
    if (recipe.emissivity < 0.1 || recipe.emissivity > 1.0) {
      errors.push({ field: 'emissivity', message: 'Emissivity must be 0.1-1.0', severity: 'error' })
    }

    setValidationErrors(errors)
    return errors.filter(e => e.severity === 'error').length === 0
  }, [recipe, thermalBudget])

  // Add segment
  const addSegment = useCallback(() => {
    const lastSegment = recipe.segments[recipe.segments.length - 1]
    const newId = (parseInt(lastSegment?.id || '0') + 1).toString()

    setRecipe(prev => ({
      ...prev,
      segments: [
        ...prev.segments,
        {
          ...DEFAULT_SEGMENT,
          id: newId,
          target_temp_c: lastSegment?.target_temp_c || 800,
          segment_type: 'dwell'
        }
      ]
    }))
  }, [recipe.segments])

  // Remove segment
  const removeSegment = useCallback((id: string) => {
    if (recipe.segments.length <= 1) return
    setRecipe(prev => ({
      ...prev,
      segments: prev.segments.filter(s => s.id !== id)
    }))
  }, [recipe.segments])

  // Move segment
  const moveSegment = useCallback((id: string, direction: 'up' | 'down') => {
    const idx = recipe.segments.findIndex(s => s.id === id)
    if (idx === -1) return
    if (direction === 'up' && idx === 0) return
    if (direction === 'down' && idx === recipe.segments.length - 1) return

    const newSegments = [...recipe.segments]
    const targetIdx = direction === 'up' ? idx - 1 : idx + 1
    ;[newSegments[idx], newSegments[targetIdx]] = [newSegments[targetIdx], newSegments[idx]]

    setRecipe(prev => ({ ...prev, segments: newSegments }))
  }, [recipe.segments])

  // Update segment
  const updateSegment = useCallback((id: string, field: keyof ThermalSegment, value: any) => {
    setRecipe(prev => ({
      ...prev,
      segments: prev.segments.map(s =>
        s.id === id ? { ...s, [field]: value } : s
      )
    }))
  }, [])

  // Submit recipe
  const handleSubmit = async () => {
    if (!validateRecipe()) {
      return
    }

    setIsSubmitting(true)
    setSubmitSuccess(false)

    try {
      const response = await fetch(`${apiEndpoint}/api/rtp/runs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token') || 'demo-token'}`,
        },
        body: JSON.stringify(recipe),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const result = await response.json()
      setSubmitSuccess(true)

      if (onSubmit) {
        onSubmit(recipe)
      }

      console.log('RTP run created:', result)
    } catch (error) {
      console.error('Failed to submit recipe:', error)
      setValidationErrors(prev => [
        ...prev,
        { field: 'submit', message: `Submission failed: ${error}`, severity: 'error' }
      ])
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Thermometer className="w-5 h-5" />
            RTP Thermal Ramp Editor
          </CardTitle>
          <CardDescription>
            Configure thermal profile segments with automated validation and thermal budget calculation
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Validation Alerts */}
      {validationErrors.length > 0 && (
        <Alert variant={validationErrors.some(e => e.severity === 'error') ? 'destructive' : 'default'}>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            <div className="font-semibold mb-2">
              {validationErrors.filter(e => e.severity === 'error').length} Error(s), {' '}
              {validationErrors.filter(e => e.severity === 'warning').length} Warning(s)
            </div>
            <ul className="text-xs space-y-1 list-disc list-inside">
              {validationErrors.slice(0, 5).map((error, idx) => (
                <li key={idx}>{error.message}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {/* Success Message */}
      {submitSuccess && (
        <Alert className="bg-green-50 border-green-500">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800">
            Recipe submitted successfully! Run is starting...
          </AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="segments" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="segments">Segments</TabsTrigger>
          <TabsTrigger value="profile">Thermal Profile</TabsTrigger>
          <TabsTrigger value="environment">Environment</TabsTrigger>
          <TabsTrigger value="controller">Controller</TabsTrigger>
        </TabsList>

        {/* Segments Tab */}
        <TabsContent value="segments" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle>Recipe Information</CardTitle>
                  <CardDescription>Basic recipe metadata and identification</CardDescription>
                </div>
                <Badge variant="outline" className="text-base">
                  {recipe.segments.length} Segments
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <Label htmlFor="recipe_name">Recipe Name *</Label>
                  <Input
                    id="recipe_name"
                    value={recipe.recipe_name}
                    onChange={(e) => setRecipe(prev => ({ ...prev, recipe_name: e.target.value }))}
                    placeholder="e.g., Dopant_Activation_1000C"
                  />
                </div>
                <div>
                  <Label htmlFor="wafer_id">Wafer ID *</Label>
                  <Input
                    id="wafer_id"
                    value={recipe.wafer_id}
                    onChange={(e) => setRecipe(prev => ({ ...prev, wafer_id: e.target.value }))}
                    placeholder="e.g., W12345"
                  />
                </div>
                <div>
                  <Label htmlFor="lot_id">Lot ID *</Label>
                  <Input
                    id="lot_id"
                    value={recipe.lot_id}
                    onChange={(e) => setRecipe(prev => ({ ...prev, lot_id: e.target.value }))}
                    placeholder="e.g., LOT-2025-001"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle>Thermal Segments</CardTitle>
                <Button onClick={addSegment} size="sm">
                  <Plus className="w-4 h-4 mr-2" />
                  Add Segment
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {recipe.segments.map((segment, idx) => {
                  const hasError = validationErrors.some(e => e.segment_id === segment.id && e.severity === 'error')
                  const prevTemp = idx > 0 ? recipe.segments[idx - 1].target_temp_c : segment.target_temp_c
                  const isHeating = segment.target_temp_c > prevTemp

                  return (
                    <div
                      key={segment.id}
                      className={`p-4 border rounded-lg ${
                        hasError ? 'border-red-500 bg-red-50' : 'border-border bg-card'
                      } ${selectedSegmentId === segment.id ? 'ring-2 ring-primary' : ''}`}
                      onClick={() => setSelectedSegmentId(segment.id)}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <Badge variant={hasError ? 'destructive' : 'default'}>
                            Segment {idx + 1}
                          </Badge>
                          <Badge variant="outline">
                            {segment.segment_type}
                          </Badge>
                          {isHeating && <TrendingUp className="w-4 h-4 text-orange-500" />}
                        </div>
                        <div className="flex gap-1">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => moveSegment(segment.id, 'up')}
                            disabled={idx === 0}
                          >
                            <ChevronUp className="w-4 h-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => moveSegment(segment.id, 'down')}
                            disabled={idx === recipe.segments.length - 1}
                          >
                            <ChevronDown className="w-4 h-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => removeSegment(segment.id)}
                            disabled={recipe.segments.length <= 1}
                          >
                            <Trash2 className="w-4 h-4 text-red-500" />
                          </Button>
                        </div>
                      </div>

                      <div className="grid grid-cols-4 gap-3">
                        <div>
                          <Label className="text-xs">Segment Type</Label>
                          <Select
                            value={segment.segment_type}
                            onValueChange={(value) => updateSegment(segment.id, 'segment_type', value)}
                          >
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="ramp">Ramp</SelectItem>
                              <SelectItem value="dwell">Dwell</SelectItem>
                              <SelectItem value="cooldown">Cooldown</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div>
                          <Label className="text-xs flex items-center gap-1">
                            <Thermometer className="w-3 h-3" />
                            Target Temp (°C)
                          </Label>
                          <Input
                            type="number"
                            value={segment.target_temp_c}
                            onChange={(e) => updateSegment(segment.id, 'target_temp_c', parseFloat(e.target.value) || 0)}
                            min={CONSTRAINTS.MIN_TEMP}
                            max={CONSTRAINTS.MAX_TEMP}
                          />
                        </div>

                        <div>
                          <Label className="text-xs flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            Duration (s)
                          </Label>
                          <Input
                            type="number"
                            value={segment.duration_s}
                            onChange={(e) => updateSegment(segment.id, 'duration_s', parseFloat(e.target.value) || 0)}
                            min={CONSTRAINTS.MIN_DURATION}
                            max={CONSTRAINTS.MAX_DURATION}
                          />
                        </div>

                        <div>
                          <Label className="text-xs flex items-center gap-1">
                            <TrendingUp className="w-3 h-3" />
                            Ramp Rate (°C/s)
                          </Label>
                          <Input
                            type="number"
                            value={segment.ramp_rate_c_per_s || ''}
                            onChange={(e) => updateSegment(segment.id, 'ramp_rate_c_per_s', parseFloat(e.target.value) || undefined)}
                            step="0.1"
                            placeholder="Auto"
                          />
                        </div>
                      </div>

                      {idx > 0 && (
                        <div className="mt-2 text-xs text-muted-foreground">
                          ΔT from previous: {(segment.target_temp_c - prevTemp).toFixed(0)}°C
                          {segment.ramp_rate_c_per_s && (
                            <span className="ml-2">
                              • Ramp time: {(Math.abs(segment.target_temp_c - prevTemp) / segment.ramp_rate_c_per_s).toFixed(1)}s
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Thermal Profile Tab */}
        <TabsContent value="profile" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Thermal Profile Visualization</CardTitle>
              <CardDescription>
                Real-time preview of temperature vs time trajectory
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={profileData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="time_s"
                      label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis
                      label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft' }}
                      domain={[CONSTRAINTS.MIN_TEMP - 100, CONSTRAINTS.MAX_TEMP + 100]}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          return (
                            <div className="bg-background border rounded p-2 shadow-lg">
                              <p className="text-sm font-semibold">{payload[0].payload.segment}</p>
                              <p className="text-xs">Time: {payload[0].payload.time_s.toFixed(1)} s</p>
                              <p className="text-xs">Temp: {payload[0].payload.temp_c.toFixed(0)} °C</p>
                            </div>
                          )
                        }
                        return null
                      }}
                    />
                    <ReferenceLine y={CONSTRAINTS.MAX_TEMP} stroke="red" strokeDasharray="3 3" label="Max" />
                    <ReferenceLine y={CONSTRAINTS.MIN_TEMP} stroke="blue" strokeDasharray="3 3" label="Min" />
                    <Line
                      type="linear"
                      dataKey="temp_c"
                      stroke="#f97316"
                      strokeWidth={3}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <Separator className="my-4" />

              {/* Thermal Budget Summary */}
              <div className="grid grid-cols-4 gap-4">
                <div className="p-3 border rounded bg-muted/30">
                  <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                    <Calculator className="w-4 h-4" />
                    Thermal Budget
                  </div>
                  <div className="text-2xl font-bold font-mono mt-1">
                    {(thermalBudget / 1000).toFixed(1)}k
                  </div>
                  <div className="text-xs text-muted-foreground">°C·s</div>
                  <div className={`text-xs mt-1 ${thermalBudget > CONSTRAINTS.MAX_THERMAL_BUDGET ? 'text-red-600' : 'text-green-600'}`}>
                    {((thermalBudget / CONSTRAINTS.MAX_THERMAL_BUDGET) * 100).toFixed(0)}% of limit
                  </div>
                </div>

                <div className="p-3 border rounded bg-muted/30">
                  <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                    <Clock className="w-4 h-4" />
                    Total Duration
                  </div>
                  <div className="text-2xl font-bold font-mono mt-1">
                    {recipe.segments.reduce((sum, s) => sum + s.duration_s, 0)}
                  </div>
                  <div className="text-xs text-muted-foreground">seconds</div>
                </div>

                <div className="p-3 border rounded bg-muted/30">
                  <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                    <Thermometer className="w-4 h-4" />
                    Peak Temp
                  </div>
                  <div className="text-2xl font-bold font-mono mt-1">
                    {Math.max(...recipe.segments.map(s => s.target_temp_c))}
                  </div>
                  <div className="text-xs text-muted-foreground">°C</div>
                </div>

                <div className="p-3 border rounded bg-muted/30">
                  <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                    <TrendingUp className="w-4 h-4" />
                    Max Ramp Rate
                  </div>
                  <div className="text-2xl font-bold font-mono mt-1">
                    {Math.max(...recipe.segments.map(s => s.ramp_rate_c_per_s || 0))}
                  </div>
                  <div className="text-xs text-muted-foreground">°C/s</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Environment Tab */}
        <TabsContent value="environment" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Wind className="w-5 h-5" />
                Ambient Environment
              </CardTitle>
              <CardDescription>Gas flows, pressure, and emissivity configuration</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <Label>Ambient Atmosphere</Label>
                <Select
                  value={recipe.gas_config.ambient}
                  onValueChange={(value: any) =>
                    setRecipe(prev => ({
                      ...prev,
                      gas_config: { ...prev.gas_config, ambient: value }
                    }))
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="N2">Nitrogen (N2)</SelectItem>
                    <SelectItem value="O2">Oxygen (O2)</SelectItem>
                    <SelectItem value="Forming Gas">Forming Gas (5% H2 in N2)</SelectItem>
                    <SelectItem value="Vacuum">Vacuum</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="n2_flow">N2 Flow (SLM)</Label>
                  <Input
                    id="n2_flow"
                    type="number"
                    value={recipe.gas_config.n2_flow_slm}
                    onChange={(e) =>
                      setRecipe(prev => ({
                        ...prev,
                        gas_config: { ...prev.gas_config, n2_flow_slm: parseFloat(e.target.value) || 0 }
                      }))
                    }
                    min={0}
                    max={20}
                    step={0.1}
                  />
                  <p className="text-xs text-muted-foreground mt-1">Range: 0-20 SLM</p>
                </div>

                <div>
                  <Label htmlFor="o2_flow">O2 Flow (SLM)</Label>
                  <Input
                    id="o2_flow"
                    type="number"
                    value={recipe.gas_config.o2_flow_slm}
                    onChange={(e) =>
                      setRecipe(prev => ({
                        ...prev,
                        gas_config: { ...prev.gas_config, o2_flow_slm: parseFloat(e.target.value) || 0 }
                      }))
                    }
                    min={0}
                    max={5}
                    step={0.1}
                  />
                  <p className="text-xs text-muted-foreground mt-1">Range: 0-5 SLM</p>
                </div>
              </div>

              <Separator />

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="pressure" className="flex items-center gap-2">
                    <Gauge className="w-4 h-4" />
                    Chamber Pressure (Torr)
                  </Label>
                  <Input
                    id="pressure"
                    type="number"
                    value={recipe.chamber_pressure_torr}
                    onChange={(e) =>
                      setRecipe(prev => ({ ...prev, chamber_pressure_torr: parseFloat(e.target.value) || 0 }))
                    }
                    min={0.1}
                    max={760}
                    step={0.1}
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    Typical: 1 Torr (process), 760 Torr (atmospheric)
                  </p>
                </div>

                <div>
                  <Label htmlFor="emissivity">Wafer Emissivity</Label>
                  <Input
                    id="emissivity"
                    type="number"
                    value={recipe.emissivity}
                    onChange={(e) =>
                      setRecipe(prev => ({ ...prev, emissivity: parseFloat(e.target.value) || 0 }))
                    }
                    min={0.1}
                    max={1.0}
                    step={0.01}
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    Si bare: 0.7, Oxide: 0.8-0.9, Nitride: 0.6
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Controller Tab */}
        <TabsContent value="controller" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="w-5 h-5" />
                Temperature Controller Configuration
              </CardTitle>
              <CardDescription>PID or MPC controller setup and gain tuning</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <Label>Controller Type</Label>
                <Select
                  value={recipe.controller.type}
                  onValueChange={(value: 'PID' | 'MPC') =>
                    setRecipe(prev => ({
                      ...prev,
                      controller: {
                        ...prev.controller,
                        type: value,
                        pid_gains: value === 'PID' ? (prev.controller.pid_gains || DEFAULT_CONTROLLER.pid_gains!) : undefined,
                        mpc_horizon: value === 'MPC' ? 10 : undefined
                      }
                    }))
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="PID">PID (Proportional-Integral-Derivative)</SelectItem>
                    <SelectItem value="MPC">MPC (Model Predictive Control)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {recipe.controller.type === 'PID' && recipe.controller.pid_gains && (
                <>
                  <Separator />
                  <div className="space-y-4">
                    <h4 className="text-sm font-semibold">PID Gain Parameters</h4>

                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <Label htmlFor="kp">Proportional Gain (Kp)</Label>
                        <Input
                          id="kp"
                          type="number"
                          value={recipe.controller.pid_gains.kp}
                          onChange={(e) =>
                            setRecipe(prev => ({
                              ...prev,
                              controller: {
                                ...prev.controller,
                                pid_gains: {
                                  ...prev.controller.pid_gains!,
                                  kp: parseFloat(e.target.value) || 0
                                }
                              }
                            }))
                          }
                          step={0.1}
                        />
                        <p className="text-xs text-muted-foreground mt-1">Typical: 1.0-5.0</p>
                      </div>

                      <div>
                        <Label htmlFor="ki">Integral Gain (Ki)</Label>
                        <Input
                          id="ki"
                          type="number"
                          value={recipe.controller.pid_gains.ki}
                          onChange={(e) =>
                            setRecipe(prev => ({
                              ...prev,
                              controller: {
                                ...prev.controller,
                                pid_gains: {
                                  ...prev.controller.pid_gains!,
                                  ki: parseFloat(e.target.value) || 0
                                }
                              }
                            }))
                          }
                          step={0.01}
                        />
                        <p className="text-xs text-muted-foreground mt-1">Typical: 0.05-0.2</p>
                      </div>

                      <div>
                        <Label htmlFor="kd">Derivative Gain (Kd)</Label>
                        <Input
                          id="kd"
                          type="number"
                          value={recipe.controller.pid_gains.kd}
                          onChange={(e) =>
                            setRecipe(prev => ({
                              ...prev,
                              controller: {
                                ...prev.controller,
                                pid_gains: {
                                  ...prev.controller.pid_gains!,
                                  kd: parseFloat(e.target.value) || 0
                                }
                              }
                            }))
                          }
                          step={0.01}
                        />
                        <p className="text-xs text-muted-foreground mt-1">Typical: 0.01-0.1</p>
                      </div>
                    </div>

                    <Alert>
                      <AlertDescription className="text-xs">
                        <strong>Tuning Guidelines:</strong>
                        <ul className="list-disc list-inside mt-2 space-y-1">
                          <li>Higher Kp → Faster response, risk of overshoot</li>
                          <li>Higher Ki → Eliminates steady-state error, risk of oscillation</li>
                          <li>Higher Kd → Reduces overshoot, sensitive to noise</li>
                          <li>Start with conservative values and tune incrementally</li>
                        </ul>
                      </AlertDescription>
                    </Alert>
                  </div>
                </>
              )}

              {recipe.controller.type === 'MPC' && (
                <>
                  <Separator />
                  <div>
                    <Label htmlFor="mpc_horizon">Prediction Horizon (steps)</Label>
                    <Input
                      id="mpc_horizon"
                      type="number"
                      value={recipe.controller.mpc_horizon || 10}
                      onChange={(e) =>
                        setRecipe(prev => ({
                          ...prev,
                          controller: {
                            ...prev.controller,
                            mpc_horizon: parseInt(e.target.value) || 10
                          }
                        }))
                      }
                      min={5}
                      max={50}
                    />
                    <p className="text-xs text-muted-foreground mt-1">
                      Number of future steps for optimization (typically 10-20)
                    </p>
                  </div>

                  <Alert>
                    <AlertDescription className="text-xs">
                      MPC provides advanced trajectory tracking with constraint handling and
                      feed-forward control. Suitable for complex multi-segment profiles.
                    </AlertDescription>
                  </Alert>
                </>
              )}

              <Separator />

              <div>
                <Label htmlFor="operator_notes">Operator Notes (Optional)</Label>
                <textarea
                  id="operator_notes"
                  className="w-full min-h-[80px] px-3 py-2 text-sm border rounded-md resize-none"
                  value={recipe.operator_notes}
                  onChange={(e) => setRecipe(prev => ({ ...prev, operator_notes: e.target.value }))}
                  placeholder="Additional notes, special instructions, or observations..."
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Submit Button */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex justify-between items-center">
            <div className="text-sm text-muted-foreground">
              <div>Total thermal budget: <strong>{(thermalBudget / 1000).toFixed(1)}k °C·s</strong></div>
              <div>Total process time: <strong>{recipe.segments.reduce((sum, s) => sum + s.duration_s, 0)} s</strong></div>
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={() => validateRecipe()}
              >
                <AlertTriangle className="w-4 h-4 mr-2" />
                Validate
              </Button>
              <Button
                onClick={handleSubmit}
                disabled={isSubmitting || validationErrors.some(e => e.severity === 'error')}
              >
                <Play className="w-4 h-4 mr-2" />
                {isSubmitting ? 'Submitting...' : 'Start RTP Run'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default RTPRampEditor
