'use client'
import { useState } from 'react'
import { Waves, Plus, Search, Download, TrendingUp, Activity, Sun, Calendar, User, FileText } from 'lucide-react'

interface UVVisNIRMeasurement {
  id: string
  sampleId: string
  sampleName: string
  wavelengthRange: string
  bandgap: number
  maxAbsorbance: number
  transmission: number
  material: string
  operator: string
  date: string
  status: 'pending' | 'measuring' | 'completed' | 'failed'
  measurementType: 'absorption' | 'transmission' | 'reflectance'
  notes?: string
}

const mockMeasurements: UVVisNIRMeasurement[] = [
  {
    id: 'UV-001',
    sampleId: 'SMP-FILM-001',
    sampleName: 'TiO2 Thin Film on Glass',
    wavelengthRange: '200-800',
    bandgap: 3.2,
    maxAbsorbance: 2.45,
    transmission: 25.3,
    material: 'TiO2',
    operator: 'Dr. Martinez',
    date: '2024-11-09 09:15',
    status: 'completed',
    measurementType: 'absorption',
    notes: 'Clear band edge, excellent transparency in visible range'
  },
  {
    id: 'UV-002',
    sampleId: 'SMP-SEMI-002',
    sampleName: 'Silicon Wafer Reference',
    wavelengthRange: '300-1100',
    bandgap: 1.12,
    maxAbsorbance: 3.85,
    transmission: 12.8,
    material: 'Si',
    operator: 'Dr. Chen',
    date: '2024-11-09 10:00',
    status: 'completed',
    measurementType: 'absorption'
  },
  {
    id: 'UV-003',
    sampleId: 'SMP-POLY-003',
    sampleName: 'Organic Photovoltaic Layer',
    wavelengthRange: '300-900',
    bandgap: 1.85,
    maxAbsorbance: 1.92,
    transmission: 45.6,
    material: 'P3HT:PCBM',
    operator: 'Dr. Patel',
    date: '2024-11-09 11:30',
    status: 'measuring',
    measurementType: 'absorption',
    notes: 'Measuring absorption spectrum for OPV optimization'
  },
  {
    id: 'UV-004',
    sampleId: 'SMP-GAN-004',
    sampleName: 'GaN on Sapphire',
    wavelengthRange: '200-800',
    bandgap: 3.4,
    maxAbsorbance: 2.15,
    transmission: 75.2,
    material: 'GaN',
    operator: 'Dr. Kim',
    date: '2024-11-09 08:45',
    status: 'completed',
    measurementType: 'transmission',
    notes: 'High transparency, sharp absorption edge'
  },
  {
    id: 'UV-005',
    sampleId: 'SMP-PEROV-005',
    sampleName: 'Perovskite Solar Cell',
    wavelengthRange: '300-850',
    bandgap: 1.55,
    maxAbsorbance: 2.68,
    transmission: 18.5,
    material: 'CH3NH3PbI3',
    operator: 'Dr. Martinez',
    date: '2024-11-09 07:20',
    status: 'completed',
    measurementType: 'absorption',
    notes: 'Strong absorption across visible spectrum'
  },
  {
    id: 'UV-006',
    sampleId: 'SMP-TEST-006',
    sampleName: 'Test Sample - QC Failed',
    wavelengthRange: '200-800',
    bandgap: 0,
    maxAbsorbance: 0,
    transmission: 0,
    material: 'Unknown',
    operator: 'Lab Manager',
    date: '2024-11-09 12:00',
    status: 'failed',
    measurementType: 'absorption',
    notes: 'Sample positioning error - measurement aborted'
  }
]

const statusConfig = {
  pending: { color: 'bg-yellow-100 text-yellow-800', label: 'Pending' },
  measuring: { color: 'bg-blue-100 text-blue-800', label: 'Measuring' },
  completed: { color: 'bg-green-100 text-green-800', label: 'Completed' },
  failed: { color: 'bg-red-100 text-red-800', label: 'Failed' }
}

const materialColors: { [key: string]: string } = {
  'TiO2': 'bg-blue-100 text-blue-800',
  'Si': 'bg-gray-100 text-gray-800',
  'P3HT:PCBM': 'bg-orange-100 text-orange-800',
  'GaN': 'bg-green-100 text-green-800',
  'CH3NH3PbI3': 'bg-purple-100 text-purple-800',
  'Unknown': 'bg-gray-100 text-gray-800'
}

const typeColors = {
  absorption: 'bg-indigo-100 text-indigo-800',
  transmission: 'bg-cyan-100 text-cyan-800',
  reflectance: 'bg-teal-100 text-teal-800'
}

export default function UVVisNIRPage() {
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

  const avgBandgap = measurements
    .filter(m => m.status === 'completed' && m.bandgap > 0)
    .reduce((sum, m) => sum + m.bandgap, 0) /
    measurements.filter(m => m.status === 'completed' && m.bandgap > 0).length

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center">
            <Waves className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">UV-Vis-NIR Spectroscopy</h1>
            <p className="text-gray-600 mt-1">Optical absorption, transmission, and reflectance measurements</p>
          </div>
        </div>
        <button
          onClick={() => setShowNewMeasurement(true)}
          className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
        >
          <Plus className="w-5 h-5" />
          New Measurement
        </button>
      </div>

      {/* Info Card */}
      <div className="bg-gradient-to-r from-blue-50 to-cyan-50 rounded-lg border border-blue-200 p-6">
        <div className="flex items-start gap-3">
          <Sun className="w-6 h-6 text-blue-600 mt-0.5" />
          <div>
            <h3 className="text-lg font-semibold text-blue-900 mb-2">About UV-Vis-NIR Spectroscopy</h3>
            <p className="text-sm text-blue-700">
              UV-Vis-NIR spectroscopy measures optical absorption and transmission across ultraviolet (200-400 nm),
              visible (400-700 nm), and near-infrared (700-2500 nm) regions. It determines band gap energies,
              optical density, and material transparency for thin films, semiconductors, and optical coatings.
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
            <Waves className="w-8 h-8 text-blue-500" />
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
              <p className="text-sm text-gray-600">Avg Bandgap</p>
              <p className="text-2xl font-bold text-cyan-600">
                {avgBandgap.toFixed(2)}<span className="text-sm ml-1">eV</span>
              </p>
            </div>
            <Activity className="w-8 h-8 text-cyan-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Range</p>
              <p className="text-2xl font-bold text-indigo-600">
                200-2500<span className="text-sm ml-1">nm</span>
              </p>
            </div>
            <Sun className="w-8 h-8 text-indigo-500" />
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
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          <div className="flex gap-2">
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
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
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${typeColors[measurement.measurementType]}`}>
                    {measurement.measurementType}
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
                <div className="bg-blue-50 rounded-lg p-3 border border-blue-200">
                  <p className="text-xs text-blue-600 uppercase font-medium mb-1">Bandgap</p>
                  <p className="text-xl font-bold text-blue-900">{measurement.bandgap.toFixed(2)} eV</p>
                </div>
                <div className="bg-indigo-50 rounded-lg p-3 border border-indigo-200">
                  <p className="text-xs text-indigo-600 uppercase font-medium mb-1">Max Absorbance</p>
                  <p className="text-xl font-bold text-indigo-900">{measurement.maxAbsorbance.toFixed(2)}</p>
                </div>
                <div className="bg-cyan-50 rounded-lg p-3 border border-cyan-200">
                  <p className="text-xs text-cyan-600 uppercase font-medium mb-1">Transmission</p>
                  <p className="text-xl font-bold text-cyan-900">{measurement.transmission.toFixed(1)}%</p>
                </div>
                <div className="bg-green-50 rounded-lg p-3 border border-green-200">
                  <p className="text-xs text-green-600 uppercase font-medium mb-1">Wavelength Range</p>
                  <p className="text-xl font-bold text-green-900">{measurement.wavelengthRange} nm</p>
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
                <Sun className="w-4 h-4 text-gray-400" />
                {measurement.wavelengthRange} nm range
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
              <button className="text-blue-600 hover:text-blue-900 text-sm font-medium">
                View Details
              </button>
              {measurement.status === 'completed' && (
                <>
                  <button className="text-cyan-600 hover:text-cyan-900 text-sm font-medium">
                    View Spectrum
                  </button>
                  <button className="text-green-600 hover:text-green-900 text-sm font-medium">
                    Export Data
                  </button>
                  <button className="text-indigo-600 hover:text-indigo-900 text-sm font-medium">
                    Tauc Plot
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
            <h2 className="text-2xl font-bold mb-4">New UV-Vis-NIR Measurement</h2>

            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Sample ID</label>
                <input
                  type="text"
                  placeholder="Enter sample identifier"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Sample Name</label>
                <input
                  type="text"
                  placeholder="Descriptive sample name"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Material Type</label>
                <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                  <option>TiO2 (Titanium Dioxide)</option>
                  <option>Si (Silicon)</option>
                  <option>GaN (Gallium Nitride)</option>
                  <option>Organic Polymer</option>
                  <option>Perovskite</option>
                  <option>Other</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Measurement Type</label>
                <div className="space-y-2">
                  <label className="flex items-center">
                    <input type="radio" name="type" className="mr-2" defaultChecked />
                    Absorption (A vs λ)
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="type" className="mr-2" />
                    Transmission (T% vs λ)
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="type" className="mr-2" />
                    Reflectance (R% vs λ)
                  </label>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Wavelength Range</label>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <input
                      type="number"
                      placeholder="Start (nm)"
                      defaultValue="200"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <input
                      type="number"
                      placeholder="End (nm)"
                      defaultValue="800"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Scan Parameters</label>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Scan Speed</label>
                    <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                      <option>Fast (2 nm/s)</option>
                      <option>Medium (1 nm/s)</option>
                      <option>Slow (0.5 nm/s)</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Data Interval</label>
                    <input
                      type="number"
                      defaultValue="1"
                      step="0.5"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                      placeholder="nm"
                    />
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Notes (Optional)</label>
                <textarea
                  rows={3}
                  placeholder="Additional notes about the measurement..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
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
                className="flex-1 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
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
