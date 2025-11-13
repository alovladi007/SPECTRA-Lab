/**
 * CVD Platform API Client
 * TypeScript client for CVD REST API
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

// ============================================================================
// Types
// ============================================================================

export interface CVDProcessMode {
  id: string;
  organization_id: string;
  pressure_mode: string;
  energy_mode: string;
  reactor_type: string;
  chemistry_type: string;
  variant?: string;
  description?: string;
  pressure_range_pa: { min: number; max: number };
  temperature_range_c: { min: number; max: number };
  capabilities: Record<string, any>;
  materials: string[];
  tool_id?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  created_by: string;
}

export interface CVDRecipe {
  id: string;
  name: string;
  description?: string;
  process_mode_id: string;
  organization_id: string;
  temperature_profile: Record<string, any>;
  gas_flows: Record<string, any>;
  pressure_profile: Record<string, any>;
  plasma_settings?: Record<string, any>;
  recipe_steps: Record<string, any>[];
  process_time_s: number;
  target_thickness_nm?: number;
  target_uniformity_pct?: number;
  tags: string[];
  version: string;
  is_baseline: boolean;
  is_golden: boolean;
  is_active: boolean;
  run_count: number;
  last_run_at?: string;
  created_at: string;
  updated_at: string;
  created_by: string;
  process_mode?: CVDProcessMode;
}

export interface CVDRun {
  id: string;
  recipe_id: string;
  process_mode_id: string;
  tool_id: string;
  organization_id: string;
  status: string;
  lot_id?: string;
  wafer_ids: string[];
  operator_id?: string;
  actual_temperature_c?: number;
  actual_pressure_pa?: number;
  actual_time_s?: number;
  run_number?: string;
  notes?: string;
  start_time?: string;
  end_time?: string;
  duration_s?: number;
  created_at: string;
  updated_at: string;
  recipe?: CVDRecipe;
  process_mode?: CVDProcessMode;
}

export interface CVDTelemetry {
  id: string;
  run_id: string;
  timestamp: string;
  temperatures: Record<string, number>;
  pressures: Record<string, number>;
  gas_flows: Record<string, number>;
  plasma_parameters?: Record<string, number>;
  rotation_speed_rpm?: number;
}

export interface CVDResult {
  id: string;
  run_id: string;
  wafer_id: string;
  thickness_nm?: number;
  thickness_std_nm?: number;
  uniformity_pct?: number;
  thickness_map?: Record<string, any>;
  refractive_index?: number;
  stress_mpa?: number;
  composition?: Record<string, number>;
  pass_fail?: boolean;
  defect_count?: number;
  vm_predicted_thickness_nm?: number;
  vm_confidence?: number;
  measurement_timestamp: string;
  created_at: string;
  updated_at: string;
  run?: CVDRun;
}

export interface SPCSeries {
  id: string;
  recipe_id?: string;
  process_mode_id?: string;
  organization_id: string;
  metric_name: string;
  chart_type: string;
  ucl: number;
  lcl: number;
  center_line: number;
  usl?: number;
  lsl?: number;
  subgroup_size: number;
  lambda_ewma?: number;
  k_cusum?: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface SPCPoint {
  id: string;
  series_id: string;
  run_id?: string;
  timestamp: string;
  value: number;
  subgroup_id?: string;
  out_of_control: boolean;
  violation_rules: string[];
  created_at: string;
}

// Query parameters
export interface ProcessModeQuery {
  organization_id?: string;
  pressure_mode?: string;
  energy_mode?: string;
  is_active?: boolean;
  skip?: number;
  limit?: number;
}

export interface RecipeQuery {
  organization_id?: string;
  process_mode_id?: string;
  is_active?: boolean;
  is_baseline?: boolean;
  is_golden?: boolean;
  search?: string;
  tags?: string[];
  skip?: number;
  limit?: number;
}

export interface RunQuery {
  organization_id?: string;
  process_mode_id?: string;
  recipe_id?: string;
  tool_id?: string;
  status?: string;
  lot_id?: string;
  start_date?: string;
  end_date?: string;
  skip?: number;
  limit?: number;
  sort_by?: string;
  sort_desc?: boolean;
}

// ============================================================================
// API Client Class
// ============================================================================

class CVDAPIClient {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  private async request<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;

    const response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  // ========================================================================
  // Process Modes
  // ========================================================================

  async getProcessModes(params?: ProcessModeQuery): Promise<CVDProcessMode[]> {
    const queryString = new URLSearchParams(
      params as Record<string, string>
    ).toString();
    return this.request<CVDProcessMode[]>(
      `/cvd/process-modes${queryString ? `?${queryString}` : ""}`
    );
  }

  async getProcessMode(id: string): Promise<CVDProcessMode> {
    return this.request<CVDProcessMode>(`/cvd/process-modes/${id}`);
  }

  async createProcessMode(data: Partial<CVDProcessMode>): Promise<CVDProcessMode> {
    return this.request<CVDProcessMode>("/cvd/process-modes", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async updateProcessMode(
    id: string,
    data: Partial<CVDProcessMode>
  ): Promise<CVDProcessMode> {
    return this.request<CVDProcessMode>(`/cvd/process-modes/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    });
  }

  // ========================================================================
  // Recipes
  // ========================================================================

  async getRecipes(params?: RecipeQuery): Promise<CVDRecipe[]> {
    const queryString = new URLSearchParams(
      params as Record<string, string>
    ).toString();
    return this.request<CVDRecipe[]>(
      `/cvd/recipes${queryString ? `?${queryString}` : ""}`
    );
  }

  async getRecipe(id: string): Promise<CVDRecipe> {
    return this.request<CVDRecipe>(`/cvd/recipes/${id}`);
  }

  async createRecipe(data: Partial<CVDRecipe>): Promise<CVDRecipe> {
    return this.request<CVDRecipe>("/cvd/recipes", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async updateRecipe(id: string, data: Partial<CVDRecipe>): Promise<CVDRecipe> {
    return this.request<CVDRecipe>(`/cvd/recipes/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    });
  }

  // ========================================================================
  // Runs
  // ========================================================================

  async getRuns(params?: RunQuery): Promise<CVDRun[]> {
    const queryString = new URLSearchParams(
      params as Record<string, string>
    ).toString();
    return this.request<CVDRun[]>(
      `/cvd/runs${queryString ? `?${queryString}` : ""}`
    );
  }

  async getRun(id: string): Promise<CVDRun> {
    return this.request<CVDRun>(`/cvd/runs/${id}`);
  }

  async createRun(data: Partial<CVDRun>): Promise<CVDRun> {
    return this.request<CVDRun>("/cvd/runs", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async updateRun(id: string, data: Partial<CVDRun>): Promise<CVDRun> {
    return this.request<CVDRun>(`/cvd/runs/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    });
  }

  async createBatchRuns(data: {
    recipe_id: string;
    process_mode_id: string;
    tool_id: string;
    organization_id: string;
    lot_id: string;
    wafer_ids: string[];
    operator_id?: string;
  }): Promise<{
    run_ids: string[];
    lot_id: string;
    total_runs: number;
    status: string;
  }> {
    return this.request("/cvd/runs/batch", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // ========================================================================
  // Telemetry
  // ========================================================================

  async getTelemetryForRun(
    runId: string,
    params?: {
      start_time?: string;
      end_time?: string;
      skip?: number;
      limit?: number;
    }
  ): Promise<CVDTelemetry[]> {
    const queryString = new URLSearchParams(
      params as Record<string, string>
    ).toString();
    return this.request<CVDTelemetry[]>(
      `/cvd/telemetry/run/${runId}${queryString ? `?${queryString}` : ""}`
    );
  }

  async createTelemetry(data: Partial<CVDTelemetry>): Promise<CVDTelemetry> {
    return this.request<CVDTelemetry>("/cvd/telemetry", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async createTelemetryBulk(data: {
    run_id: string;
    data_points: Partial<CVDTelemetry>[];
  }): Promise<{ status: string; count: number }> {
    return this.request("/cvd/telemetry/bulk", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // WebSocket for real-time telemetry
  connectTelemetryStream(runId: string): WebSocket {
    const wsURL = this.baseURL.replace("http", "ws");
    return new WebSocket(`${wsURL}/cvd/ws/telemetry/${runId}`);
  }

  // ========================================================================
  // Results
  // ========================================================================

  async getResultsForRun(runId: string): Promise<CVDResult[]> {
    return this.request<CVDResult[]>(`/cvd/results/run/${runId}`);
  }

  async createResult(data: Partial<CVDResult>): Promise<CVDResult> {
    return this.request<CVDResult>("/cvd/results", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // ========================================================================
  // SPC
  // ========================================================================

  async getSPCSeries(params?: {
    organization_id?: string;
    recipe_id?: string;
    metric_name?: string;
    is_active?: boolean;
  }): Promise<SPCSeries[]> {
    const queryString = new URLSearchParams(
      params as Record<string, string>
    ).toString();
    return this.request<SPCSeries[]>(
      `/cvd/spc/series${queryString ? `?${queryString}` : ""}`
    );
  }

  async createSPCSeries(data: Partial<SPCSeries>): Promise<SPCSeries> {
    return this.request<SPCSeries>("/cvd/spc/series", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async getSPCPoints(
    seriesId: string,
    limit?: number
  ): Promise<SPCPoint[]> {
    return this.request<SPCPoint[]>(
      `/cvd/spc/points/${seriesId}${limit ? `?limit=${limit}` : ""}`
    );
  }

  async createSPCPoint(data: Partial<SPCPoint>): Promise<SPCPoint> {
    return this.request<SPCPoint>("/cvd/spc/points", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // ========================================================================
  // Analytics
  // ========================================================================

  async getAnalytics(data: {
    metric: string;
    aggregation?: string;
    organization_id: string;
    process_mode_id?: string;
    recipe_id?: string;
    tool_id?: string;
    start_date: string;
    end_date: string;
    group_by?: string[];
    time_bin?: string;
  }): Promise<{
    metric: string;
    aggregation: string;
    data: any[];
    summary: Record<string, any>;
  }> {
    return this.request("/cvd/analytics", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // ========================================================================
  // Health Check
  // ========================================================================

  async healthCheck(): Promise<{
    status: string;
    service: string;
    timestamp: string;
  }> {
    return this.request("/cvd/health");
  }

  // ========================================================================
  // Tool Status
  // ========================================================================

  async getToolStatus(toolId: string): Promise<{
    tool_id: string;
    state: string;
    current_run_id?: string;
    message: string;
  }> {
    return this.request(`/cvd/tools/${toolId}/status`);
  }
}

// Export singleton instance
export const cvdApi = new CVDAPIClient();

// Export class for custom instances
export { CVDAPIClient };
