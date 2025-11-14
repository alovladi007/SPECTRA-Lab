/**
 * Predictive Maintenance API Client
 * Equipment health monitoring, failure prediction, and maintenance tracking
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_ANALYSIS_API_URL || 'http://localhost:8001'

// Enums
export type EquipmentStatus = 'healthy' | 'warning' | 'critical'
export type MaintenanceEventType = 'preventive' | 'corrective' | 'predictive' | 'inspection'

// Equipment Health interfaces
export interface EquipmentHealth {
  id: string
  org_id: string
  equipment_id: string
  equipment_name: string
  equipment_type?: string

  // Health metrics
  health_score: number
  predicted_failure_date?: string
  confidence: number
  failure_probability: number

  // Reliability metrics
  mtbf_hours?: number
  mttr_hours?: number

  // Maintenance schedule
  last_maintenance_date?: string
  next_maintenance_date?: string

  // Data and analysis
  sensor_data?: Record<string, any>
  anomalies?: Record<string, any>
  recommendations?: Record<string, any>

  // Status
  status: EquipmentStatus
  created_at: string
  updated_at?: string
}

export interface EquipmentHealthCreate {
  org_id: string
  equipment_id: string
  equipment_name: string
  equipment_type?: string
  health_score: number
  predicted_failure_date?: string
  confidence: number
  failure_probability: number
  mtbf_hours?: number
  mttr_hours?: number
  last_maintenance_date?: string
  next_maintenance_date?: string
  sensor_data?: Record<string, any>
  anomalies?: Record<string, any>
  recommendations?: Record<string, any>
}

export interface EquipmentHealthUpdate {
  equipment_name?: string
  equipment_type?: string
  health_score?: number
  predicted_failure_date?: string
  confidence?: number
  failure_probability?: number
  mtbf_hours?: number
  mttr_hours?: number
  last_maintenance_date?: string
  next_maintenance_date?: string
  sensor_data?: Record<string, any>
  anomalies?: Record<string, any>
  recommendations?: Record<string, any>
}

// Maintenance Event interfaces
export interface MaintenanceEvent {
  id: string
  org_id: string
  equipment_id: string
  event_type: MaintenanceEventType
  performed_date: string
  downtime_hours: number
  cost: number
  technician?: string
  description?: string
  parts_replaced?: Record<string, any>
  notes?: string
  created_at: string
}

export interface MaintenanceEventCreate {
  org_id: string
  equipment_id: string
  event_type: MaintenanceEventType
  performed_date: string
  downtime_hours?: number
  cost?: number
  technician?: string
  description?: string
  parts_replaced?: Record<string, any>
  notes?: string
}

export interface MaintenanceEventUpdate {
  event_type?: MaintenanceEventType
  performed_date?: string
  downtime_hours?: number
  cost?: number
  technician?: string
  description?: string
  parts_replaced?: Record<string, any>
  notes?: string
}

// Dashboard interface
export interface PredictiveMaintenanceDashboard {
  total_equipment: number
  healthy_count: number
  warning_count: number
  critical_count: number
  avg_health_score: number
  predictions_this_month: number
  total_downtime_hours: number
  total_maintenance_cost: number
  by_status: Record<string, number>
  by_equipment_type: Record<string, number>
}

// Failure Prediction interface
export interface FailurePrediction {
  equipment_id: string
  equipment_name: string
  equipment_type?: string
  health_score: number
  predicted_failure_date: string
  days_until_failure: number
  failure_probability: number
  confidence: number
  status: EquipmentStatus
  recommendations?: Record<string, any>
}

class PredictiveMaintenanceAPI {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  // ============================================================================
  // Equipment Health Records
  // ============================================================================

  // Get all equipment health records
  async getHealthRecords(params?: {
    org_id?: string
    status?: EquipmentStatus
    equipment_type?: string
    equipment_id?: string
    min_health_score?: number
    max_health_score?: number
    skip?: number
    limit?: number
    sort_by?: string
    sort_desc?: boolean
  }): Promise<EquipmentHealth[]> {
    const queryParams = new URLSearchParams()
    if (params?.org_id) queryParams.append('org_id', params.org_id)
    if (params?.status) queryParams.append('status', params.status)
    if (params?.equipment_type) queryParams.append('equipment_type', params.equipment_type)
    if (params?.equipment_id) queryParams.append('equipment_id', params.equipment_id)
    if (params?.min_health_score !== undefined) queryParams.append('min_health_score', params.min_health_score.toString())
    if (params?.max_health_score !== undefined) queryParams.append('max_health_score', params.max_health_score.toString())
    if (params?.skip !== undefined) queryParams.append('skip', params.skip.toString())
    if (params?.limit) queryParams.append('limit', params.limit.toString())
    if (params?.sort_by) queryParams.append('sort_by', params.sort_by)
    if (params?.sort_desc !== undefined) queryParams.append('sort_desc', params.sort_desc.toString())

    const response = await fetch(`${this.baseUrl}/api/v1/predictive-maintenance/health?${queryParams}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch health records: ${response.statusText}`)
    }
    return response.json()
  }

  // Get specific health record
  async getHealthRecord(id: string): Promise<EquipmentHealth> {
    const response = await fetch(`${this.baseUrl}/api/v1/predictive-maintenance/health/${id}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch health record: ${response.statusText}`)
    }
    return response.json()
  }

  // Create health record
  async createHealthRecord(data: EquipmentHealthCreate): Promise<EquipmentHealth> {
    const response = await fetch(`${this.baseUrl}/api/v1/predictive-maintenance/health`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!response.ok) {
      throw new Error(`Failed to create health record: ${response.statusText}`)
    }
    return response.json()
  }

  // Update health record
  async updateHealthRecord(id: string, data: EquipmentHealthUpdate): Promise<EquipmentHealth> {
    const response = await fetch(`${this.baseUrl}/api/v1/predictive-maintenance/health/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!response.ok) {
      throw new Error(`Failed to update health record: ${response.statusText}`)
    }
    return response.json()
  }

  // Get latest health for specific equipment
  async getLatestEquipmentHealth(equipmentId: string): Promise<EquipmentHealth> {
    const response = await fetch(`${this.baseUrl}/api/v1/predictive-maintenance/equipment/${equipmentId}/health`)
    if (!response.ok) {
      throw new Error(`Failed to fetch equipment health: ${response.statusText}`)
    }
    return response.json()
  }

  // ============================================================================
  // Maintenance Events
  // ============================================================================

  // Get all maintenance events
  async getMaintenanceEvents(params?: {
    org_id?: string
    equipment_id?: string
    event_type?: MaintenanceEventType
    performed_date_from?: string
    performed_date_to?: string
    skip?: number
    limit?: number
    sort_by?: string
    sort_desc?: boolean
  }): Promise<MaintenanceEvent[]> {
    const queryParams = new URLSearchParams()
    if (params?.org_id) queryParams.append('org_id', params.org_id)
    if (params?.equipment_id) queryParams.append('equipment_id', params.equipment_id)
    if (params?.event_type) queryParams.append('event_type', params.event_type)
    if (params?.performed_date_from) queryParams.append('performed_date_from', params.performed_date_from)
    if (params?.performed_date_to) queryParams.append('performed_date_to', params.performed_date_to)
    if (params?.skip !== undefined) queryParams.append('skip', params.skip.toString())
    if (params?.limit) queryParams.append('limit', params.limit.toString())
    if (params?.sort_by) queryParams.append('sort_by', params.sort_by)
    if (params?.sort_desc !== undefined) queryParams.append('sort_desc', params.sort_desc.toString())

    const response = await fetch(`${this.baseUrl}/api/v1/predictive-maintenance/events?${queryParams}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch maintenance events: ${response.statusText}`)
    }
    return response.json()
  }

  // Create maintenance event
  async createMaintenanceEvent(data: MaintenanceEventCreate): Promise<MaintenanceEvent> {
    const response = await fetch(`${this.baseUrl}/api/v1/predictive-maintenance/events`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!response.ok) {
      throw new Error(`Failed to create maintenance event: ${response.statusText}`)
    }
    return response.json()
  }

  // ============================================================================
  // Dashboard & Analytics
  // ============================================================================

  // Get dashboard statistics
  async getDashboard(params?: { org_id?: string }): Promise<PredictiveMaintenanceDashboard> {
    const queryParams = new URLSearchParams()
    if (params?.org_id) queryParams.append('org_id', params.org_id)

    const response = await fetch(`${this.baseUrl}/api/v1/predictive-maintenance/dashboard?${queryParams}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch dashboard: ${response.statusText}`)
    }
    return response.json()
  }

  // Get failure predictions
  async getFailurePredictions(params?: {
    org_id?: string
    days_ahead?: number
    min_probability?: number
    skip?: number
    limit?: number
  }): Promise<FailurePrediction[]> {
    const queryParams = new URLSearchParams()
    if (params?.org_id) queryParams.append('org_id', params.org_id)
    if (params?.days_ahead) queryParams.append('days_ahead', params.days_ahead.toString())
    if (params?.min_probability !== undefined) queryParams.append('min_probability', params.min_probability.toString())
    if (params?.skip !== undefined) queryParams.append('skip', params.skip.toString())
    if (params?.limit) queryParams.append('limit', params.limit.toString())

    const response = await fetch(`${this.baseUrl}/api/v1/predictive-maintenance/predictions?${queryParams}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch failure predictions: ${response.statusText}`)
    }
    return response.json()
  }

  // ============================================================================
  // Helper Methods
  // ============================================================================

  getHealthyEquipment(healthRecords: EquipmentHealth[]) {
    return healthRecords.filter(h => h.status === 'healthy')
  }

  getWarningEquipment(healthRecords: EquipmentHealth[]) {
    return healthRecords.filter(h => h.status === 'warning')
  }

  getCriticalEquipment(healthRecords: EquipmentHealth[]) {
    return healthRecords.filter(h => h.status === 'critical')
  }

  getEventsByType(events: MaintenanceEvent[]) {
    return events.reduce((acc, e) => {
      if (!acc[e.event_type]) {
        acc[e.event_type] = []
      }
      acc[e.event_type].push(e)
      return acc
    }, {} as Record<MaintenanceEventType, MaintenanceEvent[]>)
  }

  calculateTotalDowntime(events: MaintenanceEvent[]): number {
    return events.reduce((sum, e) => sum + e.downtime_hours, 0)
  }

  calculateTotalCost(events: MaintenanceEvent[]): number {
    return events.reduce((sum, e) => sum + e.cost, 0)
  }
}

export const predictiveMaintenanceApi = new PredictiveMaintenanceAPI()
export default predictiveMaintenanceApi
