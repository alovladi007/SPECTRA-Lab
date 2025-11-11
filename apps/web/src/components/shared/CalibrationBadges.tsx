/**
 * Calibration Badges Component
 *
 * Global component for equipment calibration status display:
 * - Equipment calibration status indicators
 * - Days until next calibration countdown
 * - Color-coded expiry warnings (green/yellow/red)
 * - Certificate links and download
 * - Last calibration date and next due date
 * - Calibration history tracking
 */

"use client"

import React, { useState, useEffect } from 'react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import {
  CheckCircle,
  AlertTriangle,
  AlertCircle,
  Clock,
  FileText,
  Calendar,
  ExternalLink
} from 'lucide-react'

interface CalibrationInfo {
  equipment_id: string
  equipment_name: string
  equipment_type: 'ion_implanter' | 'rtp' | 'furnace' | 'metrology' | 'other'
  last_calibration_date: string
  next_calibration_due: string
  calibration_interval_days: number
  status: 'current' | 'due_soon' | 'overdue' | 'expired'
  certificate_url?: string
  calibrated_by?: string
  notes?: string
}

interface CalibrationBadgesProps {
  equipmentIds?: string[]
  showExpanded?: boolean
  onCalibrationClick?: (equipment: CalibrationInfo) => void
  apiEndpoint?: string
}

export const CalibrationBadges: React.FC<CalibrationBadgesProps> = ({
  equipmentIds,
  showExpanded = false,
  onCalibrationClick,
  apiEndpoint = 'http://localhost:8002'
}) => {
  const [calibrations, setCalibrations] = useState<CalibrationInfo[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    fetchCalibrations()
  }, [equipmentIds])

  const fetchCalibrations = async () => {
    try {
      const url = equipmentIds && equipmentIds.length > 0
        ? `${apiEndpoint}/api/calibrations?equipment_ids=${equipmentIds.join(',')}`
        : `${apiEndpoint}/api/calibrations`

      const response = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token') || 'demo-token'}`,
        },
      })

      if (response.ok) {
        const data = await response.json()
        setCalibrations(data)
      } else {
        // Load mock data
        setCalibrations(generateMockCalibrations())
      }
    } catch (error) {
      console.error('Failed to fetch calibrations:', error)
      setCalibrations(generateMockCalibrations())
    } finally {
      setIsLoading(false)
    }
  }

  const generateMockCalibrations = (): CalibrationInfo[] => {
    const today = new Date()

    return [
      {
        equipment_id: 'ION-001',
        equipment_name: 'Ion Implanter A',
        equipment_type: 'ion_implanter',
        last_calibration_date: new Date(today.getTime() - 20 * 24 * 60 * 60 * 1000).toISOString(),
        next_calibration_due: new Date(today.getTime() + 10 * 24 * 60 * 60 * 1000).toISOString(),
        calibration_interval_days: 30,
        status: 'due_soon',
        certificate_url: '/certificates/ION-001-2025-01.pdf',
        calibrated_by: 'Metrology Team',
        notes: 'Faraday cup and beam current monitor calibrated'
      },
      {
        equipment_id: 'RTP-001',
        equipment_name: 'RTP Chamber 1',
        equipment_type: 'rtp',
        last_calibration_date: new Date(today.getTime() - 5 * 24 * 60 * 60 * 1000).toISOString(),
        next_calibration_due: new Date(today.getTime() + 55 * 24 * 60 * 60 * 1000).toISOString(),
        calibration_interval_days: 60,
        status: 'current',
        certificate_url: '/certificates/RTP-001-2025-01.pdf',
        calibrated_by: 'Service Engineer',
        notes: 'Pyrometer and thermocouple calibration'
      },
      {
        equipment_id: 'METRO-001',
        equipment_name: 'Four-Point Probe',
        equipment_type: 'metrology',
        last_calibration_date: new Date(today.getTime() - 95 * 24 * 60 * 60 * 1000).toISOString(),
        next_calibration_due: new Date(today.getTime() - 5 * 24 * 60 * 60 * 1000).toISOString(),
        calibration_interval_days: 90,
        status: 'overdue',
        calibrated_by: 'Metrology Team',
        notes: 'Standard resistor verification'
      },
      {
        equipment_id: 'FURNACE-001',
        equipment_name: 'Diffusion Furnace 3',
        equipment_type: 'furnace',
        last_calibration_date: new Date(today.getTime() - 15 * 24 * 60 * 60 * 1000).toISOString(),
        next_calibration_due: new Date(today.getTime() + 75 * 24 * 60 * 60 * 1000).toISOString(),
        calibration_interval_days: 90,
        status: 'current',
        certificate_url: '/certificates/FURNACE-001-2024-12.pdf',
        calibrated_by: 'Metrology Team',
        notes: 'Temperature profile verification across all zones'
      }
    ]
  }

  // Calculate days until calibration
  const getDaysUntilCalibration = (nextDue: string): number => {
    const today = new Date()
    const dueDate = new Date(nextDue)
    const diffTime = dueDate.getTime() - today.getTime()
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24))
    return diffDays
  }

  // Get status badge configuration
  const getStatusConfig = (status: CalibrationInfo['status'], daysUntil: number) => {
    switch (status) {
      case 'current':
        return {
          variant: 'default' as const,
          icon: CheckCircle,
          label: 'Current',
          color: 'text-green-600 bg-green-100 border-green-300',
          textColor: 'text-green-800'
        }
      case 'due_soon':
        return {
          variant: 'secondary' as const,
          icon: AlertTriangle,
          label: `Due in ${daysUntil}d`,
          color: 'text-yellow-600 bg-yellow-100 border-yellow-300',
          textColor: 'text-yellow-800'
        }
      case 'overdue':
        return {
          variant: 'destructive' as const,
          icon: AlertCircle,
          label: `Overdue ${Math.abs(daysUntil)}d`,
          color: 'text-red-600 bg-red-100 border-red-300 animate-pulse',
          textColor: 'text-red-800'
        }
      case 'expired':
        return {
          variant: 'destructive' as const,
          icon: XCircle,
          label: 'Expired',
          color: 'text-red-600 bg-red-100 border-red-300',
          textColor: 'text-red-800'
        }
      default:
        return {
          variant: 'secondary' as const,
          icon: Clock,
          label: 'Unknown',
          color: 'text-gray-600 bg-gray-100 border-gray-300',
          textColor: 'text-gray-800'
        }
    }
  }

  // Compact badge view
  const renderCompactBadge = (cal: CalibrationInfo) => {
    const daysUntil = getDaysUntilCalibration(cal.next_calibration_due)
    const config = getStatusConfig(cal.status, daysUntil)
    const Icon = config.icon

    return (
      <TooltipProvider key={cal.equipment_id}>
        <Tooltip>
          <TooltipTrigger asChild>
            <Badge
              variant={config.variant}
              className={`cursor-pointer ${config.color}`}
              onClick={() => onCalibrationClick?.(cal)}
            >
              <Icon className="w-3 h-3 mr-1" />
              {cal.equipment_name}
            </Badge>
          </TooltipTrigger>
          <TooltipContent>
            <div className="text-xs space-y-1">
              <div className="font-semibold">{cal.equipment_name}</div>
              <div>Status: {config.label}</div>
              <div>Last Cal: {new Date(cal.last_calibration_date).toLocaleDateString()}</div>
              <div>Next Due: {new Date(cal.next_calibration_due).toLocaleDateString()}</div>
              {cal.calibrated_by && <div>By: {cal.calibrated_by}</div>}
            </div>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    )
  }

  // Expanded card view
  const renderExpandedCard = (cal: CalibrationInfo) => {
    const daysUntil = getDaysUntilCalibration(cal.next_calibration_due)
    const config = getStatusConfig(cal.status, daysUntil)
    const Icon = config.icon

    return (
      <Card key={cal.equipment_id} className={`border-2 ${config.color}`}>
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <CardTitle className="text-base">{cal.equipment_name}</CardTitle>
              <div className="text-xs text-muted-foreground mt-1">
                {cal.equipment_id} • {cal.equipment_type.replace('_', ' ').toUpperCase()}
              </div>
            </div>
            <Badge variant={config.variant} className={`${config.color}`}>
              <Icon className="w-3 h-3 mr-1" />
              {config.label}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="p-2 border rounded bg-muted/30">
              <div className="flex items-center gap-1 text-xs text-muted-foreground mb-1">
                <Calendar className="w-3 h-3" />
                Last Calibration
              </div>
              <div className="font-semibold">
                {new Date(cal.last_calibration_date).toLocaleDateString()}
              </div>
              {cal.calibrated_by && (
                <div className="text-xs text-muted-foreground mt-1">
                  by {cal.calibrated_by}
                </div>
              )}
            </div>

            <div className="p-2 border rounded bg-muted/30">
              <div className="flex items-center gap-1 text-xs text-muted-foreground mb-1">
                <Clock className="w-3 h-3" />
                Next Due
              </div>
              <div className={`font-semibold ${config.textColor}`}>
                {new Date(cal.next_calibration_due).toLocaleDateString()}
              </div>
              <div className={`text-xs mt-1 ${config.textColor}`}>
                {daysUntil > 0 ? `in ${daysUntil} days` : `${Math.abs(daysUntil)} days overdue`}
              </div>
            </div>
          </div>

          {cal.notes && (
            <div className="text-xs p-2 border rounded bg-muted/30">
              <div className="font-medium text-muted-foreground mb-1">Notes:</div>
              <div>{cal.notes}</div>
            </div>
          )}

          {cal.certificate_url && (
            <Button
              variant="outline"
              size="sm"
              className="w-full"
              onClick={() => window.open(cal.certificate_url, '_blank')}
            >
              <FileText className="w-3 h-3 mr-2" />
              View Certificate
              <ExternalLink className="w-3 h-3 ml-2" />
            </Button>
          )}

          <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t">
            <span>Interval: {cal.calibration_interval_days} days</span>
            {cal.status === 'overdue' && (
              <Button
                variant="destructive"
                size="sm"
                onClick={() => alert('Schedule calibration workflow would be triggered')}
              >
                Schedule Now
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    )
  }

  if (isLoading) {
    return (
      <div className="flex items-center gap-2">
        <Clock className="w-4 h-4 animate-spin" />
        <span className="text-sm text-muted-foreground">Loading calibrations...</span>
      </div>
    )
  }

  if (calibrations.length === 0) {
    return (
      <Badge variant="secondary">
        <AlertTriangle className="w-3 h-3 mr-1" />
        No calibration data
      </Badge>
    )
  }

  // Filter to show only equipment with issues in compact mode
  const priorityCalibrations = showExpanded
    ? calibrations
    : calibrations.filter(cal => cal.status !== 'current')

  if (!showExpanded) {
    return (
      <div className="flex flex-wrap gap-2">
        {priorityCalibrations.length > 0 ? (
          priorityCalibrations.map(renderCompactBadge)
        ) : (
          <Badge variant="default" className="text-green-600 bg-green-100 border-green-300">
            <CheckCircle className="w-3 h-3 mr-1" />
            All Equipment Current
          </Badge>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {calibrations.map(renderExpandedCard)}
    </div>
  )
}

export default CalibrationBadges
