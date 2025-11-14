/**
 * Oxidation Manufacturing API Client
 * Thermal oxidation process management with furnaces, recipes, runs, and results
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_ANALYSIS_API_URL || 'http://localhost:8001'

// Enums matching backend
export type OxidationFurnaceType = 'horizontal_tube' | 'vertical_tube' | 'batch' | 'rapid_thermal'
export type OxidationType = 'dry' | 'wet' | 'steam' | 'pyrogenic' | 'anodic'
export type OxideApplication = 'gate_oxide' | 'field_oxide' | 'mask_oxide' | 'passivation' | 'tunnel_oxide'
export type RunStatus = 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'ABORTED'
export type MeasurementType = 'ELLIPSOMETRY' | 'REFLECTOMETRY' | 'CV'

// Furnace interfaces
export interface OxidationFurnace {
  id: string
  org_id: string
  name: string
  furnace_type: OxidationFurnaceType
  manufacturer?: string
  model?: string
  serial_number?: string
  tube_diameter_mm?: number
  tube_length_mm?: number
  num_temperature_zones: number
  max_temperature_c: number
  max_wafer_capacity: number
  supports_dry_oxidation: boolean
  supports_wet_oxidation: boolean
  supports_steam_oxidation: boolean
  supports_pyrogenic: boolean
  is_active: boolean
  is_calibrated: boolean
  last_pm_date?: string
  next_pm_date?: string
  created_at: string
  updated_at?: string
  metadata_json?: Record<string, any>
}

export interface OxidationFurnaceCreate {
  org_id: string
  name: string
  furnace_type: OxidationFurnaceType
  manufacturer?: string
  model?: string
  serial_number?: string
  tube_diameter_mm?: number
  tube_length_mm?: number
  num_temperature_zones?: number
  max_temperature_c: number
  max_wafer_capacity: number
  supports_dry_oxidation?: boolean
  supports_wet_oxidation?: boolean
  supports_steam_oxidation?: boolean
  supports_pyrogenic?: boolean
  metadata_json?: Record<string, any>
}

export interface OxidationFurnaceUpdate {
  name?: string
  is_active?: boolean
  is_calibrated?: boolean
  last_pm_date?: string
  next_pm_date?: string
  metadata_json?: Record<string, any>
}

// Recipe interfaces
export interface OxidationRecipe {
  id: string
  org_id: string
  furnace_id: string
  name: string
  description?: string
  oxidation_type: OxidationType
  application?: OxideApplication
  temperature_c: number
  time_minutes: number
  ramp_rate_c_per_min: number
  o2_flow_sccm: number
  n2_flow_sccm: number
  h2_flow_sccm: number
  target_thickness_nm?: number
  thickness_tolerance_nm?: number
  status: string
  version: string
  run_count: number
  created_at: string
  updated_at?: string
  created_by?: string
  notes?: string
  furnace?: OxidationFurnace
}

export interface OxidationRecipeCreate {
  org_id: string
  furnace_id: string
  name: string
  description?: string
  oxidation_type: OxidationType
  application?: OxideApplication
  temperature_c: number
  time_minutes: number
  ramp_rate_c_per_min?: number
  o2_flow_sccm?: number
  n2_flow_sccm?: number
  h2_flow_sccm?: number
  target_thickness_nm?: number
  thickness_tolerance_nm?: number
  created_by?: string
  notes?: string
}

export interface OxidationRecipeUpdate {
  name?: string
  description?: string
  status?: string
  notes?: string
}

// Run interfaces
export interface OxidationRun {
  id: string
  org_id: string
  furnace_id: string
  recipe_id: string
  run_number?: string
  lot_id?: string
  wafer_ids: string[]
  status: RunStatus
  queued_at: string
  started_at?: string
  completed_at?: string
  actual_temperature_c?: number
  actual_time_minutes?: number
  actual_thickness_nm?: number
  measured_thickness_nm?: number
  thickness_uniformity_percent?: number
  refractive_index?: number
  job_id?: string
  job_progress: number
  error_message?: string
  created_at: string
  updated_at?: string
  operator?: string
  notes?: string
  furnace?: OxidationFurnace
  recipe?: OxidationRecipe
}

export interface OxidationRunCreate {
  org_id: string
  furnace_id: string
  recipe_id: string
  run_number?: string
  lot_id?: string
  wafer_ids?: string[]
  operator?: string
  notes?: string
}

export interface OxidationRunUpdate {
  status?: RunStatus
  started_at?: string
  completed_at?: string
  actual_temperature_c?: number
  actual_time_minutes?: number
  measured_thickness_nm?: number
  thickness_uniformity_percent?: number
  refractive_index?: number
  job_progress?: number
  error_message?: string
  notes?: string
}

// Result interfaces
export interface OxidationResult {
  id: string
  run_id: string
  org_id: string
  wafer_id: string
  measurement_type: MeasurementType
  thickness_nm: number
  thickness_std_dev?: number
  uniformity_percent?: number
  refractive_index?: number
  extinction_coefficient?: number
  breakdown_voltage_v?: number
  dielectric_constant?: number
  interface_state_density?: number
  measurement_points?: Record<string, any>
  measured_at: string
  measured_by?: string
  equipment_id?: string
  created_at: string
  notes?: string
}

export interface OxidationResultCreate {
  run_id: string
  org_id: string
  wafer_id: string
  measurement_type: MeasurementType
  thickness_nm: number
  thickness_std_dev?: number
  uniformity_percent?: number
  refractive_index?: number
  extinction_coefficient?: number
  breakdown_voltage_v?: number
  dielectric_constant?: number
  interface_state_density?: number
  measurement_points?: Record<string, any>
  measured_by?: string
  equipment_id?: string
  notes?: string
}

export interface OxidationBatchRunRequest {
  org_id: string
  furnace_id: string
  recipe_id: string
  run_count: number
  wafers_per_run?: number
  lot_prefix?: string
}

export interface OxidationBatchRunResponse {
  runs: OxidationRun[]
  total_runs: number
  total_wafers: number
}

export interface OxidationAnalyticsRequest {
  org_id: string
  furnace_id?: string
  recipe_id?: string
  start_date?: string
  end_date?: string
}

export interface OxidationAnalyticsResponse {
  total_runs: number
  completed_runs: number
  failed_runs: number
  avg_thickness_nm?: number
  avg_uniformity_percent?: number
  avg_cycle_time_minutes?: number
  furnace_utilization_percent?: number
}

class OxidationAPI {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  // Furnace endpoints
  async getFurnaces(params?: { org_id?: string }): Promise<OxidationFurnace[]> {
    const queryParams = new URLSearchParams()
    if (params?.org_id) queryParams.append('org_id', params.org_id)

    const response = await fetch(`${this.baseUrl}/api/v1/oxidation/furnaces?${queryParams}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch furnaces: ${response.statusText}`)
    }
    return response.json()
  }

  async getFurnace(furnaceId: string): Promise<OxidationFurnace> {
    const response = await fetch(`${this.baseUrl}/api/v1/oxidation/furnaces/${furnaceId}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch furnace: ${response.statusText}`)
    }
    return response.json()
  }

  async createFurnace(data: OxidationFurnaceCreate): Promise<OxidationFurnace> {
    const response = await fetch(`${this.baseUrl}/api/v1/oxidation/furnaces`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!response.ok) {
      throw new Error(`Failed to create furnace: ${response.statusText}`)
    }
    return response.json()
  }

  async updateFurnace(furnaceId: string, data: OxidationFurnaceUpdate): Promise<OxidationFurnace> {
    const response = await fetch(`${this.baseUrl}/api/v1/oxidation/furnaces/${furnaceId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!response.ok) {
      throw new Error(`Failed to update furnace: ${response.statusText}`)
    }
    return response.json()
  }

  // Recipe endpoints
  async getRecipes(params?: { org_id?: string; furnace_id?: string }): Promise<OxidationRecipe[]> {
    const queryParams = new URLSearchParams()
    if (params?.org_id) queryParams.append('org_id', params.org_id)
    if (params?.furnace_id) queryParams.append('furnace_id', params.furnace_id)

    const response = await fetch(`${this.baseUrl}/api/v1/oxidation/recipes?${queryParams}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch recipes: ${response.statusText}`)
    }
    return response.json()
  }

  async getRecipe(recipeId: string): Promise<OxidationRecipe> {
    const response = await fetch(`${this.baseUrl}/api/v1/oxidation/recipes/${recipeId}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch recipe: ${response.statusText}`)
    }
    return response.json()
  }

  async createRecipe(data: OxidationRecipeCreate): Promise<OxidationRecipe> {
    const response = await fetch(`${this.baseUrl}/api/v1/oxidation/recipes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!response.ok) {
      throw new Error(`Failed to create recipe: ${response.statusText}`)
    }
    return response.json()
  }

  async updateRecipe(recipeId: string, data: OxidationRecipeUpdate): Promise<OxidationRecipe> {
    const response = await fetch(`${this.baseUrl}/api/v1/oxidation/recipes/${recipeId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!response.ok) {
      throw new Error(`Failed to update recipe: ${response.statusText}`)
    }
    return response.json()
  }

  // Run endpoints
  async getRuns(params?: { org_id?: string; furnace_id?: string; status?: RunStatus; limit?: number }): Promise<OxidationRun[]> {
    const queryParams = new URLSearchParams()
    if (params?.org_id) queryParams.append('org_id', params.org_id)
    if (params?.furnace_id) queryParams.append('furnace_id', params.furnace_id)
    if (params?.status) queryParams.append('status', params.status)
    if (params?.limit) queryParams.append('limit', params.limit.toString())

    const response = await fetch(`${this.baseUrl}/api/v1/oxidation/runs?${queryParams}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch runs: ${response.statusText}`)
    }
    return response.json()
  }

  async getRun(runId: string): Promise<OxidationRun> {
    const response = await fetch(`${this.baseUrl}/api/v1/oxidation/runs/${runId}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch run: ${response.statusText}`)
    }
    return response.json()
  }

  async createRun(data: OxidationRunCreate): Promise<OxidationRun> {
    const response = await fetch(`${this.baseUrl}/api/v1/oxidation/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!response.ok) {
      throw new Error(`Failed to create run: ${response.statusText}`)
    }
    return response.json()
  }

  async updateRun(runId: string, data: OxidationRunUpdate): Promise<OxidationRun> {
    const response = await fetch(`${this.baseUrl}/api/v1/oxidation/runs/${runId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!response.ok) {
      throw new Error(`Failed to update run: ${response.statusText}`)
    }
    return response.json()
  }

  async createBatchRuns(data: OxidationBatchRunRequest): Promise<OxidationBatchRunResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/oxidation/runs/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!response.ok) {
      throw new Error(`Failed to create batch runs: ${response.statusText}`)
    }
    return response.json()
  }

  // Result endpoints
  async getRunResults(runId: string): Promise<OxidationResult[]> {
    const response = await fetch(`${this.baseUrl}/api/v1/oxidation/results/run/${runId}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch run results: ${response.statusText}`)
    }
    return response.json()
  }

  async createResult(data: OxidationResultCreate): Promise<OxidationResult> {
    const response = await fetch(`${this.baseUrl}/api/v1/oxidation/results`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!response.ok) {
      throw new Error(`Failed to create result: ${response.statusText}`)
    }
    return response.json()
  }

  // Analytics
  async getAnalytics(data: OxidationAnalyticsRequest): Promise<OxidationAnalyticsResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/oxidation/analytics`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!response.ok) {
      throw new Error(`Failed to fetch analytics: ${response.statusText}`)
    }
    return response.json()
  }

  // Helper methods
  getActiveFurnaces(furnaces: OxidationFurnace[]) {
    return furnaces.filter(f => f.is_active)
  }

  getRunsByStatus(runs: OxidationRun[]) {
    return {
      queued: runs.filter(r => r.status === 'QUEUED'),
      running: runs.filter(r => r.status === 'RUNNING'),
      completed: runs.filter(r => r.status === 'COMPLETED'),
      failed: runs.filter(r => r.status === 'FAILED'),
      aborted: runs.filter(r => r.status === 'ABORTED'),
    }
  }

  getApprovedRecipes(recipes: OxidationRecipe[]) {
    return recipes.filter(r => r.status === 'APPROVED')
  }
}

export const oxidationApi = new OxidationAPI()
export default oxidationApi
