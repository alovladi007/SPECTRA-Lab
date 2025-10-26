/**
 * SPECTRA-Lab API Client
 * Centralized API client for all backend services
 */

const API_BASE_URLS = {
  analysis: process.env.NEXT_PUBLIC_ANALYSIS_API_URL || 'http://localhost:8001',
  lims: process.env.NEXT_PUBLIC_LIMS_API_URL || 'http://localhost:8002',
  instruments: process.env.NEXT_PUBLIC_INSTRUMENTS_API_URL || 'http://localhost:8003',
  platform: process.env.NEXT_PUBLIC_PLATFORM_API_URL || 'http://localhost:8004',
}

class APIError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'APIError'
  }
}

async function fetchAPI<T>(
  service: keyof typeof API_BASE_URLS,
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URLS[service]}${endpoint}`

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    })

    if (!response.ok) {
      throw new APIError(
        response.status,
        `API request failed: ${response.statusText}`
      )
    }

    return await response.json()
  } catch (error) {
    if (error instanceof APIError) throw error
    throw new Error(`Network error: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}

// ==================== Analysis Service API ====================

export const analysisAPI = {
  // Health check
  health: () => fetchAPI('analysis', '/health'),

  // Four-Point Probe
  fourPointProbe: {
    measure: (data: {
      voltage: number
      current: number
      probe_spacing?: number
      sample_thickness?: number
      temperature?: number
    }) => fetchAPI('analysis', '/api/electrical/four-point-probe/measure', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  },

  // Hall Effect
  hallEffect: {
    measure: (data: {
      magnetic_field: number
      hall_voltage: number
      current: number
      sample_thickness: number
      sample_type?: string
    }) => fetchAPI('analysis', '/api/electrical/hall-effect/measure', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  },

  // UV-Vis-NIR
  uvVisNIR: {
    analyze: (data: {
      wavelengths: number[]
      intensities: number[]
      measurement_type?: string
    }) => fetchAPI('analysis', '/api/optical/uv-vis-nir/analyze', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  },

  // SPC
  spc: {
    analyze: (data: {
      data: Array<{ timestamp: string; value: number; sample_id?: string }>
      chart_type?: string
      target?: number
      ucl?: number
      lcl?: number
    }) => fetchAPI('analysis', '/api/spc/analyze', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  },

  // Machine Learning / Virtual Metrology
  ml: {
    predict: (data: {
      process_parameters: Record<string, number>
      equipment_data: Record<string, number>
      target_metric: string
    }) => fetchAPI('analysis', '/api/ml/virtual-metrology/predict', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  },

  // Get status
  status: () => fetchAPI('analysis', '/api/status'),
}

// ==================== LIMS Service API ====================

export const limsAPI = {
  // Health check
  health: () => fetchAPI('lims', '/health'),

  // Sample Management
  samples: {
    create: (data: {
      sample_id?: string
      name: string
      material_type: string
      location: string
      status?: string
    }) => fetchAPI('lims', '/api/lims/samples', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
    list: (params?: { status?: string; material_type?: string; limit?: number }) => {
      const query = new URLSearchParams(params as any).toString()
      return fetchAPI('lims', `/api/lims/samples${query ? `?${query}` : ''}`)
    },
    get: (sampleId: string) => fetchAPI('lims', `/api/lims/samples/${sampleId}`),
    update: (sampleId: string, data: Record<string, any>) =>
      fetchAPI('lims', `/api/lims/samples/${sampleId}`, {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
  },

  // Chain of Custody
  custody: {
    add: (data: {
      sample_id: string
      action: string
      from_user: string
      to_user: string
      from_location: string
      to_location: string
    }) => fetchAPI('lims', '/api/lims/custody', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
    get: (sampleId: string) => fetchAPI('lims', `/api/lims/custody/${sampleId}`),
  },

  // Electronic Lab Notebook
  eln: {
    create: (data: {
      entry_id?: string
      title: string
      content: string
      author: string
      project_id?: string
      linked_samples?: string[]
    }) => fetchAPI('lims', '/api/lims/eln/entries', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
    list: (params?: { author?: string; project_id?: string; limit?: number }) => {
      const query = new URLSearchParams(params as any).toString()
      return fetchAPI('lims', `/api/lims/eln/entries${query ? `?${query}` : ''}`)
    },
    get: (entryId: string) => fetchAPI('lims', `/api/lims/eln/entries/${entryId}`),
  },

  // E-Signatures
  signatures: {
    add: (data: {
      entry_id: string
      user_id: string
      signature_type: string
      reason: string
    }) => fetchAPI('lims', '/api/lims/signatures', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
    get: (entryId: string) => fetchAPI('lims', `/api/lims/signatures/${entryId}`),
  },

  // SOPs
  sops: {
    create: (data: {
      sop_number: string
      title: string
      version: string
      method_name: string
      content: string
      status?: string
    }) => fetchAPI('lims', '/api/lims/sops', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
    list: (params?: { method_name?: string; status?: string }) => {
      const query = new URLSearchParams(params as any).toString()
      return fetchAPI('lims', `/api/lims/sops${query ? `?${query}` : ''}`)
    },
    get: (sopNumber: string) => fetchAPI('lims', `/api/lims/sops/${sopNumber}`),
  },

  // Reports
  reports: {
    generate: (data: {
      run_id: string
      template?: string
      include_plots?: boolean
    }) => fetchAPI('lims', '/api/lims/reports/generate', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  },

  // FAIR Export
  export: {
    fair: (data: {
      run_ids: string[]
      include_raw?: boolean
      include_metadata?: boolean
    }) => fetchAPI('lims', '/api/lims/export/fair', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  },

  // Get status
  status: () => fetchAPI('lims', '/api/status'),
}

// ==================== Helper Functions ====================

export function isAPIError(error: unknown): error is APIError {
  return error instanceof APIError
}

export function getErrorMessage(error: unknown): string {
  if (isAPIError(error)) {
    return `API Error (${error.status}): ${error.message}`
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'An unknown error occurred'
}

export { API_BASE_URLS, APIError }
