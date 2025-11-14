/**
 * Diffusion Manufacturing Platform API Client
 * TypeScript client for Diffusion REST API
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_ANALYSIS_API_URL || "http://localhost:8001/api/v1";

// ============================================================================
// Types
// ============================================================================

export interface DiffusionFurnace {
  id: string;
  org_id: string;
  name: string;
  furnace_type: string; // TUBE | BATCH | VERTICAL | RTP_HYBRID
  num_temperature_zones: number;
  max_wafer_capacity: number;
  max_temperature_c: number;
  min_temperature_c: number;
  supported_dopants: string[]; // BORON, PHOSPHORUS, ARSENIC, ANTIMONY
  supported_sources: string[]; // SOLID, LIQUID, GAS
  supported_ambients: string[]; // N2, O2, AR, H2
  temperature_uniformity_spec_c?: number;
  is_active: boolean;
  last_pm_date?: string;
  next_pm_date?: string;
  created_at: string;
  updated_at: string;
  created_by: string;
}

export interface DiffusionRecipe {
  id: string;
  name: string;
  description?: string;
  furnace_id: string;
  org_id: string;
  diffusion_type: string; // PREDEPOSITION | DRIVE_IN | CONTINUOUS
  dopant: string; // BORON, PHOSPHORUS, ARSENIC, ANTIMONY
  dopant_source: string; // SOLID, LIQUID, GAS
  temperature_profile: Record<string, any>;
  ambient_gas: string; // N2, O2, AR, H2
  ambient_flow_sccm: number;
  recipe_steps: Record<string, any>[];
  total_time_min: number;
  target_junction_depth_nm?: number;
  target_sheet_resistance_ohm_sq?: number;
  tags: string[];
  version: string;
  status: string; // DRAFT, APPROVED, DEPRECATED
  approval_date?: string;
  approved_by?: string;
  is_active: boolean;
  run_count: number;
  last_run_at?: string;
  created_at: string;
  updated_at: string;
  created_by: string;
  furnace?: DiffusionFurnace;
}

export interface DiffusionRun {
  id: string;
  recipe_id: string;
  furnace_id: string;
  org_id: string;
  status: string; // QUEUED, RUNNING, COMPLETED, FAILED, ABORTED
  lot_id?: string;
  wafer_ids: string[];
  operator_id?: string;
  run_number?: string;
  celery_task_id?: string;
  job_progress?: number;
  actual_peak_temp_c?: number;
  actual_total_time_min?: number;
  notes?: string;
  start_time?: string;
  end_time?: string;
  duration_seconds?: number;
  created_at: string;
  updated_at: string;
  recipe?: DiffusionRecipe;
  furnace?: DiffusionFurnace;
}

export interface DiffusionTelemetry {
  id: string;
  run_id: string;
  timestamp: string;
  elapsed_time_s: number;
  temperatures: Record<string, number>; // zone temperatures
  pressures?: Record<string, number>;
  ambient_flow_sccm: number;
  ambient_gas: string;
}

export interface DiffusionResult {
  id: string;
  run_id: string;
  wafer_id: string;
  junction_depth_nm?: number;
  sheet_resistance_ohm_sq?: number;
  sheet_resistance_std_ohm_sq?: number;
  uniformity_pct?: number;
  dopant_profile?: Record<string, any>;
  sims_profile?: Record<string, any>;
  pass_fail?: boolean;
  vm_predicted_junction_depth_nm?: number;
  vm_predicted_sheet_resistance_ohm_sq?: number;
  vm_confidence?: number;
  measurement_timestamp: string;
  created_at: string;
  updated_at: string;
  run?: DiffusionRun;
}

export interface DiffusionSPCSeries {
  id: string;
  recipe_id?: string;
  furnace_id?: string;
  org_id: string;
  metric_name: string;
  chart_type: string; // XBAR_R, I_MR, EWMA, CUSUM
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

export interface DiffusionSPCPoint {
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
export interface FurnaceQuery {
  org_id?: string;
  furnace_type?: string;
  is_active?: boolean;
  skip?: number;
  limit?: number;
}

export interface RecipeQuery {
  org_id?: string;
  furnace_id?: string;
  diffusion_type?: string;
  dopant?: string;
  status?: string;
  is_active?: boolean;
  search?: string;
  tags?: string[];
  skip?: number;
  limit?: number;
}

export interface RunQuery {
  org_id?: string;
  furnace_id?: string;
  recipe_id?: string;
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

class DiffusionAPIClient {
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
  // Furnaces
  // ========================================================================

  async getFurnaces(params?: FurnaceQuery): Promise<DiffusionFurnace[]> {
    const queryString = new URLSearchParams(
      params as Record<string, string>
    ).toString();
    return this.request<DiffusionFurnace[]>(
      `/diffusion/furnaces${queryString ? `?${queryString}` : ""}`
    );
  }

  async getFurnace(id: string): Promise<DiffusionFurnace> {
    return this.request<DiffusionFurnace>(`/diffusion/furnaces/${id}`);
  }

  async createFurnace(data: Partial<DiffusionFurnace>): Promise<DiffusionFurnace> {
    return this.request<DiffusionFurnace>("/diffusion/furnaces", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async updateFurnace(
    id: string,
    data: Partial<DiffusionFurnace>
  ): Promise<DiffusionFurnace> {
    return this.request<DiffusionFurnace>(`/diffusion/furnaces/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    });
  }

  async getFurnaceStatus(furnaceId: string): Promise<{
    furnace_id: string;
    state: string;
    current_run_id?: string;
    message: string;
  }> {
    return this.request(`/diffusion/furnaces/${furnaceId}/status`);
  }

  // ========================================================================
  // Recipes
  // ========================================================================

  async getRecipes(params?: RecipeQuery): Promise<DiffusionRecipe[]> {
    const queryString = new URLSearchParams(
      params as Record<string, string>
    ).toString();
    return this.request<DiffusionRecipe[]>(
      `/diffusion/recipes${queryString ? `?${queryString}` : ""}`
    );
  }

  async getRecipe(id: string): Promise<DiffusionRecipe> {
    return this.request<DiffusionRecipe>(`/diffusion/recipes/${id}`);
  }

  async createRecipe(data: Partial<DiffusionRecipe>): Promise<DiffusionRecipe> {
    return this.request<DiffusionRecipe>("/diffusion/recipes", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async updateRecipe(id: string, data: Partial<DiffusionRecipe>): Promise<DiffusionRecipe> {
    return this.request<DiffusionRecipe>(`/diffusion/recipes/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    });
  }

  async approveRecipe(id: string): Promise<DiffusionRecipe> {
    return this.request<DiffusionRecipe>(`/diffusion/recipes/${id}/approve`, {
      method: "POST",
    });
  }

  // ========================================================================
  // Runs
  // ========================================================================

  async getRuns(params?: RunQuery): Promise<DiffusionRun[]> {
    const queryString = new URLSearchParams(
      params as Record<string, string>
    ).toString();
    return this.request<DiffusionRun[]>(
      `/diffusion/runs${queryString ? `?${queryString}` : ""}`
    );
  }

  async getRun(id: string): Promise<DiffusionRun> {
    return this.request<DiffusionRun>(`/diffusion/runs/${id}`);
  }

  async createRun(data: Partial<DiffusionRun>): Promise<DiffusionRun> {
    return this.request<DiffusionRun>("/diffusion/runs", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async updateRun(id: string, data: Partial<DiffusionRun>): Promise<DiffusionRun> {
    return this.request<DiffusionRun>(`/diffusion/runs/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    });
  }

  async createBatchRuns(data: {
    recipe_id: string;
    furnace_id: string;
    org_id: string;
    lot_id: string;
    wafer_ids: string[];
    operator_id?: string;
  }): Promise<{
    run_ids: string[];
    lot_id: string;
    total_runs: number;
    status: string;
  }> {
    return this.request("/diffusion/runs/batch", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async startRun(id: string): Promise<DiffusionRun> {
    return this.request<DiffusionRun>(`/diffusion/runs/${id}/start`, {
      method: "POST",
    });
  }

  async completeRun(id: string): Promise<DiffusionRun> {
    return this.request<DiffusionRun>(`/diffusion/runs/${id}/complete`, {
      method: "POST",
    });
  }

  async abortRun(id: string, reason?: string): Promise<DiffusionRun> {
    return this.request<DiffusionRun>(`/diffusion/runs/${id}/abort`, {
      method: "POST",
      body: JSON.stringify({ reason }),
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
  ): Promise<DiffusionTelemetry[]> {
    const queryString = new URLSearchParams(
      params as Record<string, string>
    ).toString();
    return this.request<DiffusionTelemetry[]>(
      `/diffusion/telemetry/run/${runId}${queryString ? `?${queryString}` : ""}`
    );
  }

  async createTelemetry(data: Partial<DiffusionTelemetry>): Promise<DiffusionTelemetry> {
    return this.request<DiffusionTelemetry>("/diffusion/telemetry", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async createTelemetryBulk(data: {
    run_id: string;
    data_points: Partial<DiffusionTelemetry>[];
  }): Promise<{ status: string; count: number }> {
    return this.request("/diffusion/telemetry/bulk", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // WebSocket for real-time telemetry
  connectTelemetryStream(runId: string): WebSocket {
    const wsURL = this.baseURL.replace("http", "ws");
    return new WebSocket(`${wsURL}/diffusion/ws/telemetry/${runId}`);
  }

  // ========================================================================
  // Results
  // ========================================================================

  async getResultsForRun(runId: string): Promise<DiffusionResult[]> {
    return this.request<DiffusionResult[]>(`/diffusion/results/run/${runId}`);
  }

  async createResult(data: Partial<DiffusionResult>): Promise<DiffusionResult> {
    return this.request<DiffusionResult>("/diffusion/results", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // ========================================================================
  // SPC
  // ========================================================================

  async getSPCSeries(params?: {
    org_id?: string;
    recipe_id?: string;
    furnace_id?: string;
    metric_name?: string;
    is_active?: boolean;
  }): Promise<DiffusionSPCSeries[]> {
    const queryString = new URLSearchParams(
      params as Record<string, string>
    ).toString();
    return this.request<DiffusionSPCSeries[]>(
      `/diffusion/spc/series${queryString ? `?${queryString}` : ""}`
    );
  }

  async createSPCSeries(data: Partial<DiffusionSPCSeries>): Promise<DiffusionSPCSeries> {
    return this.request<DiffusionSPCSeries>("/diffusion/spc/series", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async getSPCPoints(
    seriesId: string,
    limit?: number
  ): Promise<DiffusionSPCPoint[]> {
    return this.request<DiffusionSPCPoint[]>(
      `/diffusion/spc/points/${seriesId}${limit ? `?limit=${limit}` : ""}`
    );
  }

  async createSPCPoint(data: Partial<DiffusionSPCPoint>): Promise<DiffusionSPCPoint> {
    return this.request<DiffusionSPCPoint>("/diffusion/spc/points", {
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
    org_id: string;
    furnace_id?: string;
    recipe_id?: string;
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
    return this.request("/diffusion/analytics", {
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
    return this.request("/diffusion/health");
  }
}

// Export singleton instance
export const diffusionApi = new DiffusionAPIClient();

// Export class for custom instances
export { DiffusionAPIClient };
