/**
 * API Client with JWT authentication
 */

const LIMS_API_URL = process.env.NEXT_PUBLIC_LIMS_API_URL || 'http://localhost:8002';
const ANALYSIS_API_URL = process.env.NEXT_PUBLIC_ANALYSIS_API_URL || 'http://localhost:8001';

class APIClient {
  private getAuthHeader(): Record<string, string> {
    if (typeof window === 'undefined') return {};

    const tokensStr = localStorage.getItem('auth_tokens');
    if (!tokensStr) return {};

    try {
      const tokens = JSON.parse(tokensStr);
      return {
        Authorization: `Bearer ${tokens.access_token}`,
      };
    } catch {
      return {};
    }
  }

  private async request<T>(
    url: string,
    options: RequestInit = {}
  ): Promise<T> {
    const headers = {
      'Content-Type': 'application/json',
      ...this.getAuthHeader(),
      ...options.headers,
    };

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      if (response.status === 401) {
        // Unauthorized - redirect to login
        if (typeof window !== 'undefined') {
          localStorage.removeItem('auth_tokens');
          localStorage.removeItem('auth_user');
          window.location.href = '/login';
        }
      }

      const error = await response.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(error.detail || `Request failed with status ${response.status}`);
    }

    return response.json();
  }

  // LIMS API methods
  async getSamples() {
    return this.request<any[]>(`${LIMS_API_URL}/api/samples`);
  }

  async getSample(id: string) {
    return this.request<any>(`${LIMS_API_URL}/api/samples/${id}`);
  }

  async createSample(data: any) {
    return this.request<any>(`${LIMS_API_URL}/api/samples`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getRecipes() {
    return this.request<any[]>(`${LIMS_API_URL}/api/recipes`);
  }

  async getRecipe(id: string) {
    return this.request<any>(`${LIMS_API_URL}/api/recipes/${id}`);
  }

  async createRecipe(data: any) {
    return this.request<any>(`${LIMS_API_URL}/api/recipes`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async approveRecipe(id: string, comments: string) {
    return this.request<any>(`${LIMS_API_URL}/api/recipes/${id}/approve`, {
      method: 'POST',
      body: JSON.stringify({ comments }),
    });
  }

  // Analysis API methods
  async getRuns(filters?: { status?: string }) {
    const params = new URLSearchParams(filters as any).toString();
    const url = params ? `${ANALYSIS_API_URL}/api/v1/runs?${params}` : `${ANALYSIS_API_URL}/api/v1/runs`;
    return this.request<any[]>(url);
  }

  async getRun(id: string) {
    return this.request<any>(`${ANALYSIS_API_URL}/api/v1/runs/${id}`);
  }

  async createRun(data: any) {
    return this.request<any>(`${ANALYSIS_API_URL}/api/v1/runs`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async unblockRun(id: string) {
    return this.request<any>(`${ANALYSIS_API_URL}/api/v1/runs/${id}/unblock`, {
      method: 'POST',
    });
  }

  async getCalibrations() {
    return this.request<any[]>(`${ANALYSIS_API_URL}/api/v1/calibrations`);
  }

  async getCalibrationStatus(instrumentId: string) {
    return this.request<any>(`${ANALYSIS_API_URL}/api/v1/calibrations/status/check?instrument_id=${instrumentId}`);
  }

  async createCalibration(data: any) {
    return this.request<any>(`${ANALYSIS_API_URL}/api/v1/calibrations`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }
}

export const api = new APIClient();
