/**
 * Calibration API Client
 * Equipment calibration tracking, scheduling, and compliance
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_ANALYSIS_API_URL || 'http://localhost:8001'

// Enums
export type CalibrationStatus = 'valid' | 'due_soon' | 'expired'

// Calibration interfaces
export interface Calibration {
  id: string
  org_id: string
  equipment_id: string
  equipment_name: string
  equipment_type?: string
  calibration_date: string
  next_calibration_date: string
  interval_days: number
  calibration_standard?: string
  performed_by: string
  certificate_number?: string
  status: CalibrationStatus
  notes?: string
  metadata_json?: Record<string, any>
  created_at: string
  updated_at?: string
}

export interface CalibrationCreate {
  org_id: string
  equipment_id: string
  equipment_name: string
  equipment_type?: string
  calibration_date: string
  next_calibration_date: string
  interval_days: number
  calibration_standard?: string
  performed_by: string
  certificate_number?: string
  notes?: string
  metadata_json?: Record<string, any>
}

export interface CalibrationUpdate {
  equipment_name?: string
  equipment_type?: string
  calibration_date?: string
  next_calibration_date?: string
  interval_days?: number
  calibration_standard?: string
  performed_by?: string
  certificate_number?: string
  notes?: string
  metadata_json?: Record<string, any>
}

export interface CalibrationStatusCheck {
  equipment_id: string
  equipment_name: string
  status: CalibrationStatus
  calibration_date: string
  next_calibration_date: string
  days_until_due: number
  is_valid: boolean
  is_expired: boolean
  certificate_number?: string
  performed_by: string
}

export interface CalibrationDashboard {
  total: number
  valid: number
  due_soon: number
  expired: number
  scheduled: number
  compliance_rate: number
  upcoming_this_month: number
  upcoming_this_quarter: number
  by_status: Record<string, number>
  by_equipment_type: Record<string, number>
}

class CalibrationAPI {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  // Get all calibrations
  async getCalibrations(params?: {
    org_id?: string
    equipment_id?: string
    status?: CalibrationStatus
    equipment_type?: string
    skip?: number
    limit?: number
  }): Promise<Calibration[]> {
    const queryParams = new URLSearchParams()
    if (params?.org_id) queryParams.append('org_id', params.org_id)
    if (params?.equipment_id) queryParams.append('equipment_id', params.equipment_id)
    if (params?.status) queryParams.append('status', params.status)
    if (params?.equipment_type) queryParams.append('equipment_type', params.equipment_type)
    if (params?.skip !== undefined) queryParams.append('skip', params.skip.toString())
    if (params?.limit) queryParams.append('limit', params.limit.toString())

    const response = await fetch(`${this.baseUrl}/api/v1/calibration/calibrations?${queryParams}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch calibrations: ${response.statusText}`)
    }
    return response.json()
  }

  // Get specific calibration
  async getCalibration(id: string): Promise<Calibration> {
    const response = await fetch(`${this.baseUrl}/api/v1/calibration/calibrations/${id}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch calibration: ${response.statusText}`)
    }
    return response.json()
  }

  // Create calibration
  async createCalibration(data: CalibrationCreate): Promise<Calibration> {
    const response = await fetch(`${this.baseUrl}/api/v1/calibration/calibrations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!response.ok) {
      throw new Error(`Failed to create calibration: ${response.statusText}`)
    }
    return response.json()
  }

  // Update calibration
  async updateCalibration(id: string, data: CalibrationUpdate): Promise<Calibration> {
    const response = await fetch(`${this.baseUrl}/api/v1/calibration/calibrations/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!response.ok) {
      throw new Error(`Failed to update calibration: ${response.statusText}`)
    }
    return response.json()
  }

  // Check equipment calibration status
  async checkEquipmentStatus(equipmentId: string): Promise<CalibrationStatusCheck> {
    const response = await fetch(`${this.baseUrl}/api/v1/calibration/equipment/${equipmentId}/status`)
    if (!response.ok) {
      throw new Error(`Failed to check equipment status: ${response.statusText}`)
    }
    return response.json()
  }

  // Get dashboard statistics
  async getDashboard(params?: { org_id?: string }): Promise<CalibrationDashboard> {
    const queryParams = new URLSearchParams()
    if (params?.org_id) queryParams.append('org_id', params.org_id)

    const response = await fetch(`${this.baseUrl}/api/v1/calibration/dashboard?${queryParams}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch dashboard: ${response.statusText}`)
    }
    return response.json()
  }

  // Helper methods
  getValidCalibrations(calibrations: Calibration[]) {
    return calibrations.filter(c => c.status === 'valid')
  }

  getDueSoonCalibrations(calibrations: Calibration[]) {
    return calibrations.filter(c => c.status === 'due_soon')
  }

  getExpiredCalibrations(calibrations: Calibration[]) {
    return calibrations.filter(c => c.status === 'expired')
  }

  getCalibrationsByEquipmentType(calibrations: Calibration[]) {
    return calibrations.reduce((acc, c) => {
      const type = c.equipment_type || 'Unknown'
      if (!acc[type]) {
        acc[type] = []
      }
      acc[type].push(c)
      return acc
    }, {} as Record<string, Calibration[]>)
  }
}

export const calibrationApi = new CalibrationAPI()
export default calibrationApi
