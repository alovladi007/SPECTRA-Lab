/**
 * Calibration API Client
 * Equipment calibration tracking and management
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_ANALYSIS_API_URL || 'http://localhost:8001'

export interface Calibration {
  id: string
  equipment_id: string
  equipment_name: string
  equipment_type: string
  calibration_type: string
  calibration_date: string
  next_calibration_date: string
  status: 'VALID' | 'DUE_SOON' | 'EXPIRED'
  calibration_standard: string
  performed_by: string
  certificate_url?: string
  notes?: string
  created_at: string
  updated_at: string
}

export interface CalibrationCreate {
  equipment_id: string
  equipment_name: string
  equipment_type: string
  calibration_type: string
  calibration_date: string
  next_calibration_date: string
  calibration_standard: string
  performed_by: string
  certificate_url?: string
  notes?: string
}

export interface CalibrationStatusCheck {
  equipment_id: string
  status: 'VALID' | 'DUE_SOON' | 'EXPIRED'
  days_until_due: number
  next_calibration_date: string
}

class CalibrationAPI {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  async getCalibrations(): Promise<Calibration[]> {
    const response = await fetch(`${this.baseUrl}/api/v1/calibrations/`)
    if (!response.ok) {
      throw new Error(`Failed to fetch calibrations: ${response.statusText}`)
    }
    return response.json()
  }

  async getCalibration(calibrationId: string): Promise<Calibration> {
    const response = await fetch(`${this.baseUrl}/api/v1/calibrations/${calibrationId}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch calibration: ${response.statusText}`)
    }
    return response.json()
  }

  async createCalibration(data: CalibrationCreate): Promise<Calibration> {
    const response = await fetch(`${this.baseUrl}/api/v1/calibrations/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    })
    if (!response.ok) {
      throw new Error(`Failed to create calibration: ${response.statusText}`)
    }
    return response.json()
  }

  async expireCalibration(calibrationId: string): Promise<Calibration> {
    const response = await fetch(`${this.baseUrl}/api/v1/calibrations/${calibrationId}/expire`, {
      method: 'PATCH',
    })
    if (!response.ok) {
      throw new Error(`Failed to expire calibration: ${response.statusText}`)
    }
    return response.json()
  }

  async checkStatus(equipmentId: string): Promise<CalibrationStatusCheck> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/calibrations/status/check?equipment_id=${equipmentId}`
    )
    if (!response.ok) {
      throw new Error(`Failed to check calibration status: ${response.statusText}`)
    }
    return response.json()
  }

  // Helper method to group calibrations by status
  getCalibrationsByStatus(calibrations: Calibration[]) {
    return {
      valid: calibrations.filter(c => c.status === 'VALID'),
      dueSoon: calibrations.filter(c => c.status === 'DUE_SOON'),
      expired: calibrations.filter(c => c.status === 'EXPIRED'),
    }
  }

  // Helper method to get calibrations by equipment type
  getCalibrationsByType(calibrations: Calibration[], type: string) {
    return calibrations.filter(c => c.equipment_type === type)
  }
}

export const calibrationApi = new CalibrationAPI()
export default calibrationApi
