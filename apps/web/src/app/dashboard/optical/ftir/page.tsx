'use client'
import { useState } from 'react'
import { Radio, Plus, Search, Download, TrendingUp, Activity, Sparkles, Calendar, User, FileText } from 'lucide-react'

interface FTIRMeasurement {
  id: string
  sampleId: string
  sampleName: string
  wavenumberRange: string
  resolution: number
  scans: number
  strongestPeak: number
  material: string
  operator: string
  date: string
  status: 'pending' | 'measuring' | 'completed' | 'failed'
  measurementMode: 'transmission' | 'atr' | 'reflection'
  functionalGroups?: string[]
  notes?: string
}

const mockMeasurements: FTIRMeasurement[] = [
  {
    id: 'FTIR-001',
    sampleId: 'SMP-POLY-001',
    sampleName: 'PMMA Polymer Sample',
    wavenumberRange: '4000-400',
    resolution: 4,
    scans: 32,
    strongestPeak: 1732,
    material: 'PMMA',
    operator: 'Dr. Patel',
    date: '2024-11-09 09:00',
    status: 'completed',
    measurementMode: 'atr',
    functionalGroups: ['C=O stretch', 'C-O stretch', 'C-H bend'],
    notes: 'Clear carbonyl peak, no contamination'
  },
  {
    id: 'FTIR-002',
    sampleId: 'SMP-FILM-002',
    sampleName: 'SiO2 Thin Film',
    wavenumberRange: '4000-400',
    resolution: 2,
    scans: 64,
    strongestPeak: 1080,
    material: 'SiO2',
    operator: 'Dr. Martinez',
    date: '2024-11-09 10:15',
    status: 'completed',
    measurementMode: 'transmission',
    functionalGroups: ['Si-O-Si asymm', 'Si-O-Si symm'],
    notes: 'Strong silicate peaks, good film quality'
  },
  {
    id: 'FTIR-003',
    sampleId: 'SMP-BIO-003',
    sampleName: 'Protein Sample A',
    wavenumberRange: '4000-400',
    resolution: 4,
    scans: 32,
    strongestPeak: 1650,
    material: 'Protein',
    operator: 'Dr. Chen',
    date: '2024-11-09 11:30',
    status: 'measuring',
    measurementMode: 'atr',
    functionalGroups: ['Amide I', 'Amide II'],
    notes: 'Analyzing secondary structure'
  },
  {
    id: 'FTIR-004',
    sampleId: 'SMP-ORG-004',
    sampleName: 'Aspirin Reference',
    wavenumberRange: '4000-400',
    resolution: 4,
    scans: 16,
    strongestPeak: 1755,
    material: 'Aspirin',
    operator: 'Dr. Kim',
    date: '2024-11-09 08:20',
    status: 'completed',
    measurementMode: 'atr',
    functionalGroups: ['C=O ester', 'C=O acid', 'O-H'],
    notes: 'Standard reference spectrum'
  },
  {
    id: 'FTIR-005',
    sampleId: 'SMP-COAT-005',
    sampleName: 'Anti-reflective Coating',
    wavenumberRange: '4000-400',
    resolution: 4,
    scans: 64,
    strongestPeak: 1095,
    material: 'TiO2/SiO2',
    operator: 'Dr. Martinez',
    date: '2024-11-09 07:45',
    status: 'completed',
    measurementMode: 'reflection',
    functionalGroups: ['Ti-O', 'Si-O'],
    notes: 'Multilayer coating characterization'
  },
  {
    id: 'FTIR-006',
    sampleId: 'SMP-TEST-006',
    sampleName: 'Failed Measurement',
    wavenumberRange: '4000-400',
    resolution: 4,
    scans: 0,
    strongestPeak: 0,
    material: 'Unknown',
    operator: 'Lab Manager',
    date: '2024-11-09 12:00',
    status: 'failed',
    measurementMode: 'atr',
    notes: 'Instrument error - detector issue'
  }
]

const statusConfig = {
  pending: { color: 'bg-yellow-100 text-yellow-800', label: 'Pending' },
  measuring: { color: 'bg-blue-100 text-blue-800', label: 'Measuring' },
  completed: { color: 'bg-green-100 text-green-800', label: 'Completed' },
  failed: { color: 'bg-red-100 text-red-800', label: 'Failed' }
}

const modeColors = {
  transmission: 'bg-cyan-100 text-cyan-800',
  atr: 'bg-purple-100 text-purple-800',
  reflection: 'bg-orange-100 text-orange-800'
}

export default function FTIRPage() {
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

  const avgScans = Math.round(
    measurements.filter(m => m.status === 'completed').reduce((sum, m) => sum + m.scans, 0) /
    measurements.filter(m => m.status === 'completed').length
  )

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-red-500 to-pink-500 rounded-xl flex items-center justify-center">
            <Radio className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">FTIR Spectroscopy</h1>
            <p className="text-gray-600 mt-1">Fourier Transform Infrared vibrational spectroscopy</p>
          </div>
        </div>
        <button
          onClick={() => setShowNewMeasurement(true)}
          className="flex items-center gap-2 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors"
        >
          <Plus className="w-5 h-5" />
          New Measurement
        </button>
      </div>

      {/* Info Card */}
      <div className="bg-gradient-to-r from-red-50 to-pink-50 rounded-lg border border-red-200 p-6">
        <div className="flex items-start gap-3">
          <Sparkles className="w-6 h-6 text-red-600 mt-0.5" />
          <div>
            <h3 className="text-lg font-semibold text-red-900 mb-2">About FTIR Spectroscopy</h3>
            <p className="text-sm text-red-700">
              FTIR measures infrared absorption to identify molecular vibrations and functional groups (4000-400 cm⁻¹).
              It provides chemical structure information through characteristic absorption bands for bonds like C=O, O-H, N-H.
              Techniques include transmission, ATR (Attenuated Total Reflectance), and reflectance modes.
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
            <Radio className="w-8 h-8 text-red-500" />
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
              <p className="text-sm text-gray-600">Avg Scans</p>
              <p className="text-2xl font-bold text-pink-600">{avgScans}</p>
            </div>
            <Activity className="w-8 h-8 text-pink-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Range</p>
              <p className="text-2xl font-bold text-purple-600">
                4000-400<span className="text-sm ml-1">cm⁻¹</span>
              </p>
            </div>
            <Sparkles className="w-8 h-8 text-purple-500" />
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
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
            />
          </div>
          <div className="flex gap-2">
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
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
                  <span className="px-2 py-1 bg-gray-100 text-gray-800 rounded-full text-xs font-medium">
                    {measurement.material}
                  </span>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${modeColors[measurement.measurementMode]}`}>
                    {measurement.measurementMode.toUpperCase()}
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
              <>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <div className="bg-red-50 rounded-lg p-3 border border-red-200">
                    <p className="text-xs text-red-600 uppercase font-medium mb-1">Strongest Peak</p>
                    <p className="text-xl font-bold text-red-900">{measurement.strongestPeak} cm⁻¹</p>
                  </div>
                  <div className="bg-pink-50 rounded-lg p-3 border border-pink-200">
                    <p className="text-xs text-pink-600 uppercase font-medium mb-1">Resolution</p>
                    <p className="text-xl font-bold text-pink-900">{measurement.resolution} cm⁻¹</p>
                  </div>
                  <div className="bg-purple-50 rounded-lg p-3 border border-purple-200">
                    <p className="text-xs text-purple-600 uppercase font-medium mb-1">Scans</p>
                    <p className="text-xl font-bold text-purple-900">{measurement.scans}</p>
                  </div>
                  <div className="bg-orange-50 rounded-lg p-3 border border-orange-200">
                    <p className="text-xs text-orange-600 uppercase font-medium mb-1">Range</p>
                    <p className="text-xl font-bold text-orange-900">{measurement.wavenumberRange}</p>
                  </div>
                </div>

                {measurement.functionalGroups && (
                  <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3 mb-3">
                    <p className="text-xs text-indigo-600 uppercase font-medium mb-2">Functional Groups Identified</p>
                    <div className="flex flex-wrap gap-2">
                      {measurement.functionalGroups.map((group, idx) => (
                        <span key={idx} className="px-2 py-1 bg-indigo-100 text-indigo-800 text-xs rounded-full">
                          {group}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </>
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
                <Radio className="w-4 h-4 text-gray-400" />
                {measurement.wavenumberRange} cm⁻¹
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
              <button className="text-red-600 hover:text-red-900 text-sm font-medium">
                View Details
              </button>
              {measurement.status === 'completed' && (
                <>
                  <button className="text-pink-600 hover:text-pink-900 text-sm font-medium">
                    View Spectrum
                  </button>
                  <button className="text-green-600 hover:text-green-900 text-sm font-medium">
                    Export Data
                  </button>
                  <button className="text-purple-600 hover:text-purple-900 text-sm font-medium">
                    Peak Table
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
            <Radio className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No measurements found matching your criteria</p>
          </div>
        )}
      </div>

      {/* New Measurement Modal */}
      {showNewMeasurement && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <h2 className="text-2xl font-bold mb-4">New FTIR Measurement</h2>

            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Sample ID</label>
                <input
                  type="text"
                  placeholder="Enter sample identifier"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Measurement Mode</label>
                <div className="space-y-2">
                  <label className="flex items-center">
                    <input type="radio" name="mode" className="mr-2" defaultChecked />
                    ATR (Attenuated Total Reflectance)
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="mode" className="mr-2" />
                    Transmission
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="mode" className="mr-2" />
                    Reflection
                  </label>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Resolution (cm⁻¹)</label>
                  <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500">
                    <option>4</option>
                    <option>2</option>
                    <option>1</option>
                    <option>0.5</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Number of Scans</label>
                  <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500">
                    <option>16</option>
                    <option>32</option>
                    <option>64</option>
                    <option>128</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Notes (Optional)</label>
                <textarea
                  rows={3}
                  placeholder="Additional notes about the measurement..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500"
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
                className="flex-1 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700"
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
