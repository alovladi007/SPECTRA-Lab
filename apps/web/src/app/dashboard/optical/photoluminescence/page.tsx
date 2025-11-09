'use client'
import { useState } from 'react'
import { Zap, Plus, Search, Download, TrendingUp, Activity, Thermometer, Calendar, User, FileText } from 'lucide-react'

interface PLMeasurement {
  id: string
  sampleId: string
  sampleName: string
  peakWavelength: number
  peakIntensity: number
  fwhm: number
  material: string
  excitationWavelength: number
  temperature: number
  operator: string
  date: string
  status: 'pending' | 'measuring' | 'completed' | 'failed'
  quantumEfficiency?: number
  notes?: string
}

const mockMeasurements: PLMeasurement[] = [
  {
    id: 'PL-001',
    sampleId: 'SMP-GAN-001',
    sampleName: 'GaN Quantum Well Structure',
    peakWavelength: 365.2,
    peakIntensity: 12500,
    fwhm: 8.4,
    material: 'GaN',
    excitationWavelength: 266,
    temperature: 300,
    operator: 'Dr. Kim',
    date: '2024-11-09 09:00',
    status: 'completed',
    quantumEfficiency: 68.5,
    notes: 'Excellent material quality, narrow linewidth'
  },
  {
    id: 'PL-002',
    sampleId: 'SMP-INGAAN-002',
    sampleName: 'InGaN/GaN MQW Blue LED',
    peakWavelength: 450.8,
    peakIntensity: 8900,
    fwhm: 15.2,
    material: 'InGaN',
    excitationWavelength: 325,
    temperature: 300,
    operator: 'Dr. Chen',
    date: '2024-11-09 10:30',
    status: 'completed',
    quantumEfficiency: 45.2
  },
  {
    id: 'PL-003',
    sampleId: 'SMP-ALGAAN-003',
    sampleName: 'AlGaN UV Emitter',
    peakWavelength: 285.5,
    peakIntensity: 3400,
    fwhm: 12.8,
    material: 'AlGaN',
    excitationWavelength: 266,
    temperature: 77,
    operator: 'Dr. Patel',
    date: '2024-11-09 11:15',
    status: 'measuring',
    notes: 'Low temperature measurement in progress'
  },
  {
    id: 'PL-004',
    sampleId: 'SMP-PEROV-004',
    sampleName: 'Perovskite Thin Film',
    peakWavelength: 780.2,
    peakIntensity: 15600,
    fwhm: 35.4,
    material: 'CH3NH3PbI3',
    excitationWavelength: 532,
    temperature: 300,
    operator: 'Dr. Martinez',
    date: '2024-11-09 08:30',
    status: 'completed',
    quantumEfficiency: 72.8,
    notes: 'Strong PL emission, promising for solar cells'
  },
  {
    id: 'PL-005',
    sampleId: 'SMP-QDOT-005',
    sampleName: 'CdSe Quantum Dots',
    peakWavelength: 620.5,
    peakIntensity: 22000,
    fwhm: 25.6,
    material: 'CdSe',
    excitationWavelength: 405,
    temperature: 300,
    operator: 'Dr. Kim',
    date: '2024-11-09 07:45',
    status: 'completed',
    quantumEfficiency: 85.3,
    notes: 'Size-tuned quantum dots, excellent quantum yield'
  },
  {
    id: 'PL-006',
    sampleId: 'SMP-TEST-006',
    sampleName: 'Test Sample - Defective',
    peakWavelength: 0,
    peakIntensity: 0,
    fwhm: 0,
    material: 'Unknown',
    excitationWavelength: 325,
    temperature: 300,
    operator: 'Lab Manager',
    date: '2024-11-09 12:00',
    status: 'failed',
    notes: 'No PL signal detected - possible non-radiative defects'
  }
]

const statusConfig = {
  pending: { color: 'bg-yellow-100 text-yellow-800', label: 'Pending' },
  measuring: { color: 'bg-blue-100 text-blue-800', label: 'Measuring' },
  completed: { color: 'bg-green-100 text-green-800', label: 'Completed' },
  failed: { color: 'bg-red-100 text-red-800', label: 'Failed' }
}

const materialColors: { [key: string]: string } = {
  'GaN': 'bg-blue-100 text-blue-800',
  'InGaN': 'bg-indigo-100 text-indigo-800',
  'AlGaN': 'bg-violet-100 text-violet-800',
  'CH3NH3PbI3': 'bg-orange-100 text-orange-800',
  'CdSe': 'bg-red-100 text-red-800',
  'Unknown': 'bg-gray-100 text-gray-800'
}

export default function PhotoluminescencePage() {
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

  const avgQE = measurements
    .filter(m => m.status === 'completed' && m.quantumEfficiency)
    .reduce((sum, m) => sum + (m.quantumEfficiency || 0), 0) /
    measurements.filter(m => m.status === 'completed' && m.quantumEfficiency).length

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-amber-500 to-orange-500 rounded-xl flex items-center justify-center">
            <Zap className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Photoluminescence Spectroscopy</h1>
            <p className="text-gray-600 mt-1">Optical emission and material quality characterization</p>
          </div>
        </div>
        <button
          onClick={() => setShowNewMeasurement(true)}
          className="flex items-center gap-2 bg-amber-600 text-white px-4 py-2 rounded-lg hover:bg-amber-700 transition-colors"
        >
          <Plus className="w-5 h-5" />
          New Measurement
        </button>
      </div>

      {/* Info Card */}
      <div className="bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg border border-amber-200 p-6">
        <div className="flex items-start gap-3">
          <Activity className="w-6 h-6 text-amber-600 mt-0.5" />
          <div>
            <h3 className="text-lg font-semibold text-amber-900 mb-2">About Photoluminescence</h3>
            <p className="text-sm text-amber-700">
              Photoluminescence (PL) spectroscopy measures light emission from materials after optical excitation.
              It reveals bandgap energies, defect states, and material quality through emission spectra.
              PL intensity and linewidth indicate crystal quality, while peak position identifies material composition.
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
            <Zap className="w-8 h-8 text-amber-500" />
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
              <p className="text-sm text-gray-600">Avg QE</p>
              <p className="text-2xl font-bold text-amber-600">
                {avgQE.toFixed(1)}<span className="text-sm ml-1">%</span>
              </p>
            </div>
            <Activity className="w-8 h-8 text-amber-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">RT Measurements</p>
              <p className="text-2xl font-bold text-orange-600">
                {measurements.filter(m => m.temperature === 300).length}
              </p>
            </div>
            <Thermometer className="w-8 h-8 text-orange-500" />
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
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
            />
          </div>
          <div className="flex gap-2">
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
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
                <div className="bg-amber-50 rounded-lg p-3 border border-amber-200">
                  <p className="text-xs text-amber-600 uppercase font-medium mb-1">Peak λ</p>
                  <p className="text-xl font-bold text-amber-900">{measurement.peakWavelength.toFixed(1)} nm</p>
                </div>
                <div className="bg-orange-50 rounded-lg p-3 border border-orange-200">
                  <p className="text-xs text-orange-600 uppercase font-medium mb-1">Intensity</p>
                  <p className="text-xl font-bold text-orange-900">{measurement.peakIntensity.toLocaleString()}</p>
                </div>
                <div className="bg-yellow-50 rounded-lg p-3 border border-yellow-200">
                  <p className="text-xs text-yellow-600 uppercase font-medium mb-1">FWHM</p>
                  <p className="text-xl font-bold text-yellow-900">{measurement.fwhm.toFixed(1)} nm</p>
                </div>
                {measurement.quantumEfficiency && (
                  <div className="bg-green-50 rounded-lg p-3 border border-green-200">
                    <p className="text-xs text-green-600 uppercase font-medium mb-1">QE</p>
                    <p className="text-xl font-bold text-green-900">{measurement.quantumEfficiency.toFixed(1)}%</p>
                  </div>
                )}
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
                <Zap className="w-4 h-4 text-gray-400" />
                Excitation: {measurement.excitationWavelength} nm
              </div>
              <div className="flex items-center gap-1">
                <Thermometer className="w-4 h-4 text-gray-400" />
                {measurement.temperature} K
              </div>
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
              <button className="text-amber-600 hover:text-amber-900 text-sm font-medium">
                View Details
              </button>
              {measurement.status === 'completed' && (
                <>
                  <button className="text-blue-600 hover:text-blue-900 text-sm font-medium">
                    View Spectrum
                  </button>
                  <button className="text-green-600 hover:text-green-900 text-sm font-medium">
                    Export Data
                  </button>
                  <button className="text-purple-600 hover:text-purple-900 text-sm font-medium">
                    Peak Analysis
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
            <Zap className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No measurements found matching your criteria</p>
          </div>
        )}
      </div>

      {/* New Measurement Modal */}
      {showNewMeasurement && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <h2 className="text-2xl font-bold mb-4">New Photoluminescence Measurement</h2>

            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Sample ID</label>
                <input
                  type="text"
                  placeholder="Enter sample identifier"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Sample Name</label>
                <input
                  type="text"
                  placeholder="Descriptive sample name"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Material Type</label>
                <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500">
                  <option>GaN (Gallium Nitride)</option>
                  <option>InGaN (Indium Gallium Nitride)</option>
                  <option>AlGaN (Aluminum Gallium Nitride)</option>
                  <option>CdSe (Cadmium Selenide QDs)</option>
                  <option>Perovskite (CH3NH3PbI3)</option>
                  <option>Other</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Excitation λ (nm)</label>
                  <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500">
                    <option>266 nm (UV)</option>
                    <option>325 nm (HeCd)</option>
                    <option>405 nm (Violet)</option>
                    <option>532 nm (Green)</option>
                    <option>Custom</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Temperature (K)</label>
                  <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500">
                    <option>300 K (RT)</option>
                    <option>77 K (LN2)</option>
                    <option>10 K (Cryostat)</option>
                    <option>Variable</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Measurement Mode</label>
                <div className="space-y-2">
                  <label className="flex items-center">
                    <input type="radio" name="mode" className="mr-2" defaultChecked />
                    Steady-state PL (continuous excitation)
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="mode" className="mr-2" />
                    Time-resolved PL (TRPL)
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="mode" className="mr-2" />
                    Temperature-dependent PL
                  </label>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Scan Parameters</label>
                <div className="grid grid-cols-3 gap-2">
                  <input
                    type="number"
                    placeholder="Start (nm)"
                    className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500"
                  />
                  <input
                    type="number"
                    placeholder="End (nm)"
                    className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500"
                  />
                  <input
                    type="number"
                    placeholder="Step (nm)"
                    defaultValue="0.5"
                    step="0.1"
                    className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Notes (Optional)</label>
                <textarea
                  rows={3}
                  placeholder="Additional notes about the measurement..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500"
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
                className="flex-1 bg-amber-600 text-white px-4 py-2 rounded-lg hover:bg-amber-700"
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
