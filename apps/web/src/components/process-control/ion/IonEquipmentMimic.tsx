/**
 * Ion Implantation Equipment Mimic Panel
 *
 * Visual representation of ion implanter subsystems with real-time status:
 * - Ion Source (extraction, mass analysis)
 * - Accelerator column
 * - Beam line (analyzer magnet, focusing)
 * - End station (scanning, wafer handling)
 * - Hazard states and interlocks
 */

"use client"

import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import {
  AlertTriangle,
  CheckCircle,
  Circle,
  Zap,
  Radio,
  Magnet,
  Target,
  Shield,
  Gauge
} from 'lucide-react'

interface SubsystemStatus {
  name: string
  status: 'safe' | 'hazard' | 'offline' | 'standby'
  parameter?: string
  value?: string
  nominal?: string
}

interface InterlockStatus {
  name: string
  active: boolean
  critical: boolean
}

interface HazardState {
  level: 'safe' | 'caution' | 'warning' | 'danger'
  message: string
}

interface IonEquipmentMimicProps {
  subsystems?: SubsystemStatus[]
  interlocks?: InterlockStatus[]
  hazardState?: HazardState
  beamOn?: boolean
  vacuumPressure?: number
  beamCurrent?: number
  analyzerField?: number
}

const defaultSubsystems: SubsystemStatus[] = [
  { name: 'Ion Source', status: 'safe', parameter: 'Arc Current', value: '15.2 A', nominal: '15 A' },
  { name: 'Extraction', status: 'safe', parameter: 'HV', value: '30.0 kV', nominal: '30 kV' },
  { name: 'Mass Analyzer', status: 'safe', parameter: 'Field', value: '1200 G', nominal: '1200 G' },
  { name: 'Accelerator', status: 'safe', parameter: 'HV', value: '100.0 kV', nominal: '100 kV' },
  { name: 'Focusing', status: 'safe', parameter: 'Quad V', value: '±500 V', nominal: '±500 V' },
  { name: 'Scanner', status: 'safe', parameter: 'Freq', value: '400 Hz', nominal: '400 Hz' },
  { name: 'End Station', status: 'safe', parameter: 'Vacuum', value: '2.1e-6 Torr', nominal: '<1e-5 Torr' },
]

const defaultInterlocks: InterlockStatus[] = [
  { name: 'Vacuum OK', active: true, critical: true },
  { name: 'Cooling Water', active: true, critical: true },
  { name: 'Door Closed', active: true, critical: true },
  { name: 'Beam Stop', active: false, critical: false },
  { name: 'E-Stop', active: false, critical: true },
]

export const IonEquipmentMimic: React.FC<IonEquipmentMimicProps> = ({
  subsystems = defaultSubsystems,
  interlocks = defaultInterlocks,
  hazardState = { level: 'safe', message: 'All systems normal' },
  beamOn = false,
  vacuumPressure = 2.1e-6,
  beamCurrent = 0,
  analyzerField = 0,
}) => {
  const getStatusColor = (status: SubsystemStatus['status']) => {
    switch (status) {
      case 'safe': return 'bg-green-500'
      case 'hazard': return 'bg-red-500 animate-pulse'
      case 'offline': return 'bg-gray-400'
      case 'standby': return 'bg-yellow-500'
    }
  }

  const getStatusIcon = (status: SubsystemStatus['status']) => {
    switch (status) {
      case 'safe': return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'hazard': return <AlertTriangle className="w-4 h-4 text-red-500 animate-pulse" />
      case 'offline': return <Circle className="w-4 h-4 text-gray-400" />
      case 'standby': return <Circle className="w-4 h-4 text-yellow-500" />
    }
  }

  const getHazardColor = (level: HazardState['level']) => {
    switch (level) {
      case 'safe': return 'bg-green-100 border-green-500 text-green-800'
      case 'caution': return 'bg-blue-100 border-blue-500 text-blue-800'
      case 'warning': return 'bg-yellow-100 border-yellow-500 text-yellow-800'
      case 'danger': return 'bg-red-100 border-red-500 text-red-800'
    }
  }

  return (
    <div className="space-y-4">
      {/* Hazard State Banner */}
      <Alert className={`border-2 ${getHazardColor(hazardState.level)}`}>
        <Shield className="h-5 w-5" />
        <AlertDescription className="font-semibold text-base">
          {hazardState.level.toUpperCase()}: {hazardState.message}
        </AlertDescription>
      </Alert>

      {/* Beam Status Indicator */}
      {beamOn && (
        <Alert variant="destructive" className="border-2 animate-pulse">
          <Zap className="h-5 w-5" />
          <AlertDescription className="font-bold text-lg">
            ⚠️ ION BEAM ACTIVE - HIGH VOLTAGE HAZARD
          </AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Equipment Schematic */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Radio className="w-5 h-5" />
              Equipment Mimic
            </CardTitle>
          </CardHeader>
          <CardContent>
            {/* Visual representation of ion implanter */}
            <div className="space-y-4">
              {/* Ion Source */}
              <div className="flex items-center gap-3 p-3 border rounded-lg bg-muted/50">
                <div className={`w-3 h-3 rounded-full ${getStatusColor(subsystems[0]?.status || 'safe')}`} />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <Zap className="w-4 h-4" />
                    <span className="font-semibold">Ion Source</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Arc: {subsystems[0]?.value || '15.2 A'}
                  </div>
                </div>
                {getStatusIcon(subsystems[0]?.status || 'safe')}
              </div>

              {/* Extraction */}
              <div className="ml-8 border-l-2 border-muted-foreground h-8" />
              <div className="flex items-center gap-3 p-3 border rounded-lg bg-muted/50">
                <div className={`w-3 h-3 rounded-full ${getStatusColor(subsystems[1]?.status || 'safe')}`} />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <Zap className="w-4 h-4 text-orange-500" />
                    <span className="font-semibold">Extraction</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    HV: {subsystems[1]?.value || '30.0 kV'}
                  </div>
                </div>
                {getStatusIcon(subsystems[1]?.status || 'safe')}
              </div>

              {/* Mass Analyzer */}
              <div className="ml-8 border-l-2 border-muted-foreground h-8" />
              <div className="flex items-center gap-3 p-3 border rounded-lg bg-muted/50">
                <div className={`w-3 h-3 rounded-full ${getStatusColor(subsystems[2]?.status || 'safe')}`} />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <Magnet className="w-4 h-4 text-purple-500" />
                    <span className="font-semibold">Mass Analyzer</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    B-Field: {analyzerField.toFixed(0)} G
                  </div>
                </div>
                {getStatusIcon(subsystems[2]?.status || 'safe')}
              </div>

              {/* Accelerator */}
              <div className="ml-8 border-l-2 border-muted-foreground h-8" />
              <div className="flex items-center gap-3 p-3 border rounded-lg bg-muted/50">
                <div className={`w-3 h-3 rounded-full ${getStatusColor(subsystems[3]?.status || 'safe')}`} />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <Zap className="w-4 h-4 text-red-500" />
                    <span className="font-semibold">Accelerator</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    HV: {subsystems[3]?.value || '100.0 kV'}
                  </div>
                </div>
                {getStatusIcon(subsystems[3]?.status || 'safe')}
              </div>

              {/* Focusing */}
              <div className="ml-8 border-l-2 border-muted-foreground h-8" />
              <div className="flex items-center gap-3 p-3 border rounded-lg bg-muted/50">
                <div className={`w-3 h-3 rounded-full ${getStatusColor(subsystems[4]?.status || 'safe')}`} />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <Target className="w-4 h-4 text-blue-500" />
                    <span className="font-semibold">Focusing</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Quads: {subsystems[4]?.value || '±500 V'}
                  </div>
                </div>
                {getStatusIcon(subsystems[4]?.status || 'safe')}
              </div>

              {/* Scanner */}
              <div className="ml-8 border-l-2 border-muted-foreground h-8" />
              <div className="flex items-center gap-3 p-3 border rounded-lg bg-muted/50">
                <div className={`w-3 h-3 rounded-full ${getStatusColor(subsystems[5]?.status || 'safe')}`} />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <Radio className="w-4 h-4 text-indigo-500" />
                    <span className="font-semibold">Scanner</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {subsystems[5]?.value || '400 Hz'}
                  </div>
                </div>
                {getStatusIcon(subsystems[5]?.status || 'safe')}
              </div>

              {/* End Station */}
              <div className="ml-8 border-l-2 border-muted-foreground h-8" />
              <div className="flex items-center gap-3 p-3 border rounded-lg bg-muted/50">
                <div className={`w-3 h-3 rounded-full ${getStatusColor(subsystems[6]?.status || 'safe')}`} />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <Gauge className="w-4 h-4 text-teal-500" />
                    <span className="font-semibold">End Station</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Vacuum: {vacuumPressure.toExponential(1)} Torr
                  </div>
                </div>
                {getStatusIcon(subsystems[6]?.status || 'safe')}
              </div>
            </div>

            {/* Beam Current Indicator */}
            <div className="mt-4 p-3 border rounded-lg bg-primary/5">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Beam Current</span>
                <span className="text-2xl font-bold font-mono">
                  {beamCurrent.toFixed(2)} mA
                </span>
              </div>
              <div className="mt-2 h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-300 ${beamOn ? 'bg-orange-500' : 'bg-gray-300'}`}
                  style={{ width: `${Math.min(100, (beamCurrent / 10) * 100)}%` }}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Interlocks and Safety */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="w-5 h-5" />
              Safety Interlocks
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Interlock Status */}
            <div className="space-y-2">
              {interlocks.map((interlock, idx) => (
                <div
                  key={idx}
                  className={`flex items-center justify-between p-2 border rounded ${
                    !interlock.active && interlock.critical
                      ? 'border-red-500 bg-red-50'
                      : interlock.active
                        ? 'border-green-500 bg-green-50'
                        : 'border-gray-300 bg-gray-50'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    {interlock.active ? (
                      <CheckCircle className="w-4 h-4 text-green-600" />
                    ) : (
                      <AlertTriangle className={`w-4 h-4 ${interlock.critical ? 'text-red-600' : 'text-yellow-600'}`} />
                    )}
                    <span className="text-sm font-medium">{interlock.name}</span>
                  </div>
                  <Badge variant={interlock.active ? "default" : "destructive"}>
                    {interlock.active ? 'OK' : 'FAULT'}
                  </Badge>
                </div>
              ))}
            </div>

            {/* Critical Parameters */}
            <div className="pt-4 border-t space-y-3">
              <h4 className="text-sm font-semibold text-muted-foreground">Critical Parameters</h4>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-2 border rounded bg-muted/30">
                  <div className="text-xs text-muted-foreground">Vacuum</div>
                  <div className="text-sm font-mono font-semibold">
                    {vacuumPressure.toExponential(1)}
                  </div>
                  <div className="text-xs text-green-600">Nominal: &lt;1e-5</div>
                </div>

                <div className="p-2 border rounded bg-muted/30">
                  <div className="text-xs text-muted-foreground">Beam Current</div>
                  <div className="text-sm font-mono font-semibold">
                    {beamCurrent.toFixed(2)} mA
                  </div>
                  <div className="text-xs text-green-600">Target: 5.0 mA</div>
                </div>

                <div className="p-2 border rounded bg-muted/30">
                  <div className="text-xs text-muted-foreground">Analyzer Field</div>
                  <div className="text-sm font-mono font-semibold">
                    {analyzerField.toFixed(0)} G
                  </div>
                  <div className="text-xs text-green-600">Nominal: 1200 G</div>
                </div>

                <div className="p-2 border rounded bg-muted/30">
                  <div className="text-xs text-muted-foreground">HV Status</div>
                  <div className="text-sm font-mono font-semibold">
                    {beamOn ? 'ON' : 'OFF'}
                  </div>
                  <div className={`text-xs ${beamOn ? 'text-red-600' : 'text-green-600'}`}>
                    {beamOn ? 'HAZARD' : 'SAFE'}
                  </div>
                </div>
              </div>
            </div>

            {/* Safety Messages */}
            <div className="pt-4 border-t">
              <h4 className="text-sm font-semibold text-muted-foreground mb-2">Safety Notes</h4>
              <ul className="text-xs space-y-1 text-muted-foreground list-disc list-inside">
                <li>High voltage present when beam is on</li>
                <li>X-ray radiation hazard in beam line area</li>
                <li>Chamber under vacuum - do not open door</li>
                <li>Emergency stop buttons located on control console</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default IonEquipmentMimic
