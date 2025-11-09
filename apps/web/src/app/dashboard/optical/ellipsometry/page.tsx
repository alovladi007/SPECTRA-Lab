'use client'
import { useState } from 'react'
import { Waves, Plus, Search, Download, TrendingUp, Layers, Eye, Calendar, User, FileText } from 'lucide-react'

interface EllipsometryMeasurement {
  id: string
  sampleId: string
  sampleName: string
  wavelength: number
  psi: number
  delta: number
  thickness: number
  refractiveIndex: number
  material: string
  operator: string
  date: string
  status: 'pending' | 'measuring' | 'completed' | 'failed'
  temperature?: number
  notes?: string
}

const mockMeasurements: EllipsometryMeasurement[] = [
  {
    id: 'ELLIP-001',
    sampleId: 'SMP-WAF-001',
    sampleName: 'Silicon Dioxide on Si Wafer',
    wavelength: 632.8,
    psi: 23.5,
    delta: 156.8,
    thickness: 245.3,
    refractiveIndex: 1.46,
    material: 'SiO2',
    operator: 'Dr. Chen',
    date: '2024-11-09 09:30',
    status: 'completed',
    temperature: 25,
    notes: 'Uniform film, excellent fit quality'
  },
  {
    id: 'ELLIP-002',
    sampleId: 'SMP-FILM-002',
    sampleName: 'TiO2 Thin Film',
    wavelength: 632.8,
    psi: 34.2,
    delta: 142.1,
    thickness: 178.6,
    refractiveIndex: 2.35,
    material: 'TiO2',
    operator: 'Dr. Martinez',
    date: '2024-11-09 10:15',
    status: 'completed',
    temperature: 25
  },
  {
    id: 'ELLIP-003',
    sampleId: 'SMP-GAN-003',
    sampleName: 'GaN on Sapphire',
    wavelength: 632.8,
    psi: 28.7,
    delta: 168.3,
    thickness: 1250,
    refractiveIndex: 2.42,
    material: 'GaN',
    operator: 'Dr. Kim',
    date: '2024-11-09 11:00',
    status: 'measuring',
    temperature: 25
  },
  {
    id: 'ELLIP-004',
    sampleId: 'SMP-POLY-004',
    sampleName: 'Polymer Coating',
    wavelength: 632.8,
    psi: 18.4,
    delta: 134.9,
    thickness: 456.2,
    refractiveIndex: 1.58,
    material: 'PMMA',
    operator: 'Dr. Patel',
    date: '2024-11-09 08:45',
    status: 'completed',
    temperature: 25,
    notes: 'Surface roughness detected'
  },
  {
    id: 'ELLIP-005',
    sampleId: 'SMP-ALN-005',
    sampleName: 'AlN Buffer Layer',
    wavelength: 632.8,
    psi: 31.6,
    delta: 159.2,
    thickness: 89.4,
    refractiveIndex: 2.15,
    material: 'AlN',
    operator: 'Dr. Chen',
    date: '2024-11-09 07:30',
    status: 'completed',
    temperature: 25
  },
  {
    id: 'ELLIP-006',
    sampleId: 'SMP-TEST-006',
    sampleName: 'Test Sample - QC',
    wavelength: 632.8,
    psi: 0,
    delta: 0,
    thickness: 0,
    refractiveIndex: 0,
    material: 'Unknown',
    operator: 'Lab Manager',
    date: '2024-11-09 12:00',
    status: 'failed',
    temperature: 25,
    notes: 'Measurement failed - sample misalignment'
  }
]

const statusConfig = {
  pending: { color: 'bg-yellow-100 text-yellow-800', label: 'Pending' },
  measuring: { color: 'bg-blue-100 text-blue-800', label: 'Measuring' },
  completed: { color: 'bg-green-100 text-green-800', label: 'Completed' },
  failed: { color: 'bg-red-100 text-red-800', label: 'Failed' }
}

const materialColors: { [key: string]: string } = {
  'SiO2': 'bg-blue-100 text-blue-800',
  'TiO2': 'bg-purple-100 text-purple-800',
  'GaN': 'bg-green-100 text-green-800',
  'PMMA': 'bg-orange-100 text-orange-800',
  'AlN': 'bg-teal-100 text-teal-800',
  'Unknown': 'bg-gray-100 text-gray-800'
}

export default function EllipsometryPage() {
  const [measurements, setMeasurements] = useState(mockMeasurements)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [showNewMeasurement, setShowNewMeasurement] = useState(false)

  const filteredMeasurements = measurements.filter(m => {
    const matchesSearch =
      m.sampleName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      m.sampleId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      m.material.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesStatus = statusFilter === 'all' || m.status === statusFilter

    return matchesSearch && matchesStatus
  })

  const avgThickness = measurements
    .filter(m => m.status === 'completed')
    .reduce((sum, m) => sum + m.thickness, 0) / measurements.filter(m => m.status === 'completed').length

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-violet-500 to-purple-500 rounded-xl flex items-center justify-center">
            <Waves className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Spectroscopic Ellipsometry</h1>
            <p className="text-gray-600 mt-1">Thin film thickness and optical properties measurement</p>
          </div>
        </div>
        <button
          onClick={() => setShowNewMeasurement(true)}
          className="flex items-center gap-2 bg-violet-600 text-white px-4 py-2 rounded-lg hover:bg-violet-700 transition-colors"
        >
          <Plus className="w-5 h-5" />
          New Measurement
        </button>
      </div>

      {/* Info Card */}
      <div className="bg-gradient-to-r from-violet-50 to-purple-50 rounded-lg border border-violet-200 p-6">
        <div className="flex items-start gap-3">
          <Eye className="w-6 h-6 text-violet-600 mt-0.5" />
          <div>
            <h3 className="text-lg font-semibold text-violet-900 mb-2">About Ellipsometry</h3>
            <p className="text-sm text-violet-700">
              Ellipsometry measures the change in polarization state of light upon reflection from a sample surface.
              It provides accurate film thickness (Ångström precision) and optical constants (n, k) without contact or damage.
              The technique measures Ψ (psi) and Δ (delta) angles to determine film properties.
            </p>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Measurements</p>
              <p className="text-2xl font-bold text-gray-900">{measurements.length}</p>
            </div>
            <Waves className="w-8 h-8 text-violet-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Completed</p>
              <p className="text-2xl font-bold text-green-600">
                {measurements.filter(m => m.status === 'completed').length}
              </p>
            </div>
            <FileText className="w-8 h-8 text-green-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">In Progress</p>
              <p className="text-2xl font-bold text-blue-600">
                {measurements.filter(m => m.status === 'measuring').length}
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-blue-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Avg Thickness</p>
              <p className="text-2xl font-bold text-violet-600">
                {avgThickness.toFixed(1)}<span className="text-sm ml-1">nm</span>
              </p>
            </div>
            <Layers className="w-8 h-8 text-violet-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Wavelength</p>
              <p className="text-2xl font-bold text-purple-600">
                632.8<span className="text-sm ml-1">nm</span>
              </p>
            </div>
            <Waves className="w-8 h-8 text-purple-500" />
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Search by sample name, ID, or material..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500 focus:border-transparent"
            />
          </div>
          <div className="flex gap-2">
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500 focus:border-transparent"
            >
              <option value="all">All Status</option>
              <option value="pending">Pending</option>
              <option value="measuring">Measuring</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
            </select>
            <button className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
              <Download className="w-5 h-5" />
              Export
            </button>
          </div>
        </div>
      </div>

      {/* Measurements List */}
      <div className="space-y-4">
        {filteredMeasurements.map((measurement) => (
          <div key={measurement.id} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between mb-4">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-2">
                  <h3 className="text-lg font-semibold text-gray-900">{measurement.sampleName}</h3>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${materialColors[measurement.material]}`}>
                    {measurement.material}
                  </span>
                </div>
                <p className="text-sm text-gray-500">
                  Measurement ID: {measurement.id} • Sample: {measurement.sampleId}
                </p>
              </div>
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${statusConfig[measurement.status].color}`}>
                {statusConfig[measurement.status].label}
              </span>
            </div>

            {measurement.status === 'completed' && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                <div className="bg-violet-50 rounded-lg p-3 border border-violet-200">
                  <p className="text-xs text-violet-600 uppercase font-medium mb-1">Ψ (Psi)</p>
                  <p className="text-xl font-bold text-violet-900">{measurement.psi.toFixed(2)}°</p>
                </div>
                <div className="bg-purple-50 rounded-lg p-3 border border-purple-200">
                  <p className="text-xs text-purple-600 uppercase font-medium mb-1">Δ (Delta)</p>
                  <p className="text-xl font-bold text-purple-900">{measurement.delta.toFixed(2)}°</p>
                </div>
                <div className="bg-blue-50 rounded-lg p-3 border border-blue-200">
                  <p className="text-xs text-blue-600 uppercase font-medium mb-1">Thickness</p>
                  <p className="text-xl font-bold text-blue-900">{measurement.thickness.toFixed(1)} nm</p>
                </div>
                <div className="bg-green-50 rounded-lg p-3 border border-green-200">
                  <p className="text-xs text-green-600 uppercase font-medium mb-1">Refractive Index</p>
                  <p className="text-xl font-bold text-green-900">{measurement.refractiveIndex.toFixed(3)}</p>
                </div>
              </div>
            )}

            <div className="flex flex-wrap items-center gap-4 text-sm text-gray-600 mb-3">
              <div className="flex items-center gap-1">
                <User className="w-4 h-4 text-gray-400" />
                {measurement.operator}
              </div>
              <div className="flex items-center gap-1">
                <Calendar className="w-4 h-4 text-gray-400" />
                {measurement.date}
              </div>
              <div className="flex items-center gap-1">
                <Waves className="w-4 h-4 text-gray-400" />
                λ = {measurement.wavelength} nm
              </div>
              {measurement.temperature && (
                <div className="flex items-center gap-1">
                  <TrendingUp className="w-4 h-4 text-gray-400" />
                  {measurement.temperature}°C
                </div>
              )}
            </div>

            {measurement.notes && (
              <div className="bg-gray-50 rounded-lg p-3 mb-3">
                <p className="text-sm text-gray-700">
                  <FileText className="w-4 h-4 inline mr-1 text-gray-400" />
                  {measurement.notes}
                </p>
              </div>
            )}

            <div className="flex gap-2 pt-3 border-t border-gray-200">
              <button className="text-violet-600 hover:text-violet-900 text-sm font-medium">
                View Details
              </button>
              {measurement.status === 'completed' && (
                <>
                  <button className="text-blue-600 hover:text-blue-900 text-sm font-medium">
                    View Spectra
                  </button>
                  <button className="text-green-600 hover:text-green-900 text-sm font-medium">
                    Export Data
                  </button>
                  <button className="text-purple-600 hover:text-purple-900 text-sm font-medium">
                    Fitting Model
                  </button>
                </>
              )}
              {measurement.status === 'failed' && (
                <button className="text-orange-600 hover:text-orange-900 text-sm font-medium">
                  Retry Measurement
                </button>
              )}
            </div>
          </div>
        ))}

        {filteredMeasurements.length === 0 && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
            <Waves className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No measurements found matching your criteria</p>
          </div>
        )}
      </div>

      {/* New Measurement Modal */}
      {showNewMeasurement && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <h2 className="text-2xl font-bold mb-4">New Ellipsometry Measurement</h2>

            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Sample ID</label>
                <input
                  type="text"
                  placeholder="Enter sample identifier"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Sample Name</label>
                <input
                  type="text"
                  placeholder="Descriptive sample name"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Material</label>
                <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500">
                  <option>SiO2 (Silicon Dioxide)</option>
                  <option>TiO2 (Titanium Dioxide)</option>
                  <option>GaN (Gallium Nitride)</option>
                  <option>AlN (Aluminum Nitride)</option>
                  <option>PMMA (Polymer)</option>
                  <option>Other</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Wavelength (nm)</label>
                  <input
                    type="number"
                    defaultValue="632.8"
                    step="0.1"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Temperature (°C)</label>
                  <input
                    type="number"
                    defaultValue="25"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Measurement Type</label>
                <div className="space-y-2">
                  <label className="flex items-center">
                    <input type="radio" name="measurement_type" className="mr-2" defaultChecked />
                    Single wavelength (monochromatic)
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="measurement_type" className="mr-2" />
                    Spectroscopic scan (multi-wavelength)
                  </label>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Notes (Optional)</label>
                <textarea
                  rows={3}
                  placeholder="Additional notes about the measurement..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500"
                />
              </div>
            </div>

            <div className="flex gap-2">
              <button
                onClick={() => setShowNewMeasurement(false)}
                className="flex-1 bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300"
              >
                Cancel
              </button>
              <button
                onClick={() => setShowNewMeasurement(false)}
                className="flex-1 bg-violet-600 text-white px-4 py-2 rounded-lg hover:bg-violet-700"
              >
                Start Measurement
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
