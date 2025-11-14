/**
 * SPC (Statistical Process Control) API Client
 * Centralized SPC monitoring across all manufacturing processes
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_ANALYSIS_API_URL || 'http://localhost:8001'

// Enums
export type ProcessType = 'diffusion' | 'cvd' | 'oxidation' | 'ion' | 'rtp'
export type ChartType = 'XBAR_R' | 'I_MR' | 'P' | 'C' | 'EWMA' | 'CUSUM'
export type ControlStatus = 'IN_CONTROL' | 'OUT_OF_CONTROL' | 'WARNING'

//  SPC Series interfaces
export interface UnifiedSPCSeries {
  id: string
  org_id: string
  name: string
  parameter: string
  process_type: ProcessType
  chart_type: ChartType
  control_status: ControlStatus
  ucl?: number
  lcl?: number
  target?: number
  usl?: number
  lsl?: number
  mean?: number
  std_dev?: number
  cp?: number
  cpk?: number
  sample_count: number
  violation_count: number
  created_at: string
  updated_at?: string
  is_active: boolean
}

export interface UnifiedSPCPoint {
  id: string
  series_id: string
  value: number
  ts: string
  process_type: ProcessType
  violates_rule?: boolean
  rule_violations?: string[]
  run_id?: string
}

export interface SPCViolationSummary {
  series_id: string
  series_name: string
  parameter: string
  process_type: ProcessType
  chart_type: ChartType
  violation_count: number
  latest_violation_time?: string
  control_status: ControlStatus
}

export interface SPCDashboardResponse {
  total_series: number
  series_by_process: Record<ProcessType, number>
  in_control_count: number
  out_of_control_count: number
  warning_count: number
  recent_violations: SPCViolationSummary[]
  top_violating_parameters: {
    parameter: string
    process_types: ProcessType[]
    total_violations: number
    series_count: number
  }[]
  process_capability_summary: {
    avg_cpk?: number
    min_cpk?: number
    max_cpk?: number
    capable_series_count: number
  }
}

export interface SPCParameterInfo {
  parameter: string
  process_types: ProcessType[]
  series_count: number
  total_samples: number
}

class SPCAPI {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  // Get all SPC series across processes
  async getSeries(params?: {
    org_id?: string
    process_type?: ProcessType
    status?: ControlStatus
    parameter?: string
    skip?: number
    limit?: number
  }): Promise<UnifiedSPCSeries[]> {
    const queryParams = new URLSearchParams()
    if (params?.org_id) queryParams.append('org_id', params.org_id)
    if (params?.process_type) queryParams.append('process_type', params.process_type)
    if (params?.status) queryParams.append('status', params.status)
    if (params?.parameter) queryParams.append('parameter', params.parameter)
    if (params?.skip !== undefined) queryParams.append('skip', params.skip.toString())
    if (params?.limit) queryParams.append('limit', params.limit.toString())

    const response = await fetch(`${this.baseUrl}/api/v1/spc/series?${queryParams}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch SPC series: ${response.statusText}`)
    }
    return response.json()
  }

  // Get SPC points for a specific series
  async getSeriesPoints(
    seriesId: string,
    params?: {
      limit?: number
      skip?: number
      start_date?: string
      end_date?: string
    }
  ): Promise<UnifiedSPCPoint[]> {
    const queryParams = new URLSearchParams()
    if (params?.limit) queryParams.append('limit', params.limit.toString())
    if (params?.skip !== undefined) queryParams.append('skip', params.skip.toString())
    if (params?.start_date) queryParams.append('start_date', params.start_date)
    if (params?.end_date) queryParams.append('end_date', params.end_date)

    const response = await fetch(
      `${this.baseUrl}/api/v1/spc/series/${seriesId}/points?${queryParams}`
    )
    if (!response.ok) {
      throw new Error(`Failed to fetch SPC points: ${response.statusText}`)
    }
    return response.json()
  }

  // Get SPC violations
  async getViolations(params?: {
    org_id?: string
    start_date?: string
    end_date?: string
    process_type?: ProcessType
    min_violations?: number
    limit?: number
  }): Promise<SPCViolationSummary[]> {
    const queryParams = new URLSearchParams()
    if (params?.org_id) queryParams.append('org_id', params.org_id)
    if (params?.start_date) queryParams.append('start_date', params.start_date)
    if (params?.end_date) queryParams.append('end_date', params.end_date)
    if (params?.process_type) queryParams.append('process_type', params.process_type)
    if (params?.min_violations !== undefined) queryParams.append('min_violations', params.min_violations.toString())
    if (params?.limit) queryParams.append('limit', params.limit.toString())

    const response = await fetch(`${this.baseUrl}/api/v1/spc/violations?${queryParams}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch SPC violations: ${response.statusText}`)
    }
    return response.json()
  }

  // Get SPC dashboard summary
  async getDashboard(orgId: string): Promise<SPCDashboardResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/spc/dashboard?org_id=${orgId}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch SPC dashboard: ${response.statusText}`)
    }
    return response.json()
  }

  // Get list of tracked parameters
  async getParameters(params?: {
    org_id?: string
    process_type?: ProcessType
  }): Promise<SPCParameterInfo[]> {
    const queryParams = new URLSearchParams()
    if (params?.org_id) queryParams.append('org_id', params.org_id)
    if (params?.process_type) queryParams.append('process_type', params.process_type)

    const response = await fetch(`${this.baseUrl}/api/v1/spc/parameters?${queryParams}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch SPC parameters: ${response.statusText}`)
    }
    return response.json()
  }

  // Helper methods
  getSeriesByStatus(series: UnifiedSPCSeries[]) {
    return {
      inControl: series.filter(s => s.control_status === 'IN_CONTROL'),
      outOfControl: series.filter(s => s.control_status === 'OUT_OF_CONTROL'),
      warning: series.filter(s => s.control_status === 'WARNING'),
    }
  }

  getSeriesByProcess(series: UnifiedSPCSeries[]) {
    return series.reduce((acc, s) => {
      if (!acc[s.process_type]) {
        acc[s.process_type] = []
      }
      acc[s.process_type].push(s)
      return acc
    }, {} as Record<ProcessType, UnifiedSPCSeries[]>)
  }

  getCapableSeries(series: UnifiedSPCSeries[], minCpk: number = 1.33) {
    return series.filter(s => s.cpk !== undefined && s.cpk >= minCpk)
  }
}

export const spcApi = new SPCAPI()
export default spcApi
