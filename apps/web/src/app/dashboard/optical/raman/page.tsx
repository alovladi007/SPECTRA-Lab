'use client'

import { useState } from 'react'
import {
  Activity,
  Calendar,
  CheckCircle,
  Clock,
  Filter,
  Plus,
  Search,
  User,
  XCircle,
  Zap,
  Info,
  TrendingUp,
  BarChart3,
} from 'lucide-react'

interface RamanMeasurement {
  id: string
  sampleId: string
  sampleName: string
  laserWavelength: number
  powerMw: number
  exposureTime: number
  accumulations: number
  strongestPeak: number
  peakIntensity: number
  material: string
  operator: string
  date: string
  status: 'pending' | 'measuring' | 'completed' | 'failed'
  ramanShifts?: number[]
  identifiedPeaks?: string[]
  notes?: string
}

const mockMeasurements: RamanMeasurement[] = [
  {
    id: 'RM-001',
    sampleId: 'S-2024-189',
    sampleName: 'Silicon Wafer (100)',
    laserWavelength: 532,
    powerMw: 5.0,
    exposureTime: 10,
    accumulations: 3,
    strongestPeak: 520,
    peakIntensity: 45800,
    material: 'Si',
    operator: 'Dr. Chen',
    date: '2024-01-15',
    status: 'completed',
    ramanShifts: [520, 940, 1450],
    identifiedPeaks: ['Si-Si (F2g)', 'Si-O', '2nd order'],
    notes: 'High quality single crystal Si',
  },
  {
    id: 'RM-002',
    sampleId: 'S-2024-190',
    sampleName: 'Graphene on SiO2',
    laserWavelength: 532,
    powerMw: 1.0,
    exposureTime: 20,
    accumulations: 5,
    strongestPeak: 1580,
    peakIntensity: 38200,
    material: 'Graphene',
    operator: 'Dr. Kim',
    date: '2024-01-14',
    status: 'completed',
    ramanShifts: [1350, 1580, 2700],
    identifiedPeaks: ['D band', 'G band', '2D band'],
    notes: 'Monolayer graphene, low defect density',
  },
  {
    id: 'RM-003',
    sampleId: 'S-2024-191',
    sampleName: 'GaN LED Structure',
    laserWavelength: 532,
    powerMw: 3.0,
    exposureTime: 15,
    accumulations: 4,
    strongestPeak: 568,
    peakIntensity: 29500,
    material: 'GaN',
    operator: 'Dr. Lee',
    date: '2024-01-13',
    status: 'completed',
    ramanShifts: [144, 418, 568],
    identifiedPeaks: ['E2(low)', 'E1(TO)', 'E2(high)'],
    notes: 'High quality wurtzite GaN',
  },
  {
    id: 'RM-004',
    sampleId: 'S-2024-192',
    sampleName: 'Diamond Film CVD',
    laserWavelength: 785,
    powerMw: 10.0,
    exposureTime: 5,
    accumulations: 2,
    strongestPeak: 1332,
    peakIntensity: 52300,
    material: 'Diamond',
    operator: 'Dr. Martinez',
    date: '2024-01-12',
    status: 'measuring',
    ramanShifts: [1332],
    identifiedPeaks: ['C-C sp3'],
  },
  {
    id: 'RM-005',
    sampleId: 'S-2024-193',
    sampleName: 'PMMA Polymer',
    laserWavelength: 785,
    powerMw: 2.0,
    exposureTime: 30,
    accumulations: 6,
    strongestPeak: 2950,
    peakIntensity: 18700,
    material: 'PMMA',
    operator: 'Dr. Wilson',
    date: '2024-01-11',
    status: 'completed',
    ramanShifts: [814, 1450, 2950],
    identifiedPeaks: ['C-O-C stretch', 'CH2 bend', 'C-H stretch'],
    notes: 'Polymer characterization for coating',
  },
  {
    id: 'RM-006',
    sampleId: 'S-2024-194',
    sampleName: 'Carbon Nanotube Array',
    laserWavelength: 532,
    powerMw: 0.5,
    exposureTime: 25,
    accumulations: 8,
    strongestPeak: 1590,
    peakIntensity: 31200,
    material: 'CNT',
    operator: 'Dr. Chen',
    date: '2024-01-10',
    status: 'pending',
    ramanShifts: [186, 1350, 1590],
    identifiedPeaks: ['RBM', 'D band', 'G band'],
  },
]

export default function RamanPage() {
  const [measurements, setMeasurements] = useState<RamanMeasurement[]>(mockMeasurements)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [showNewModal, setShowNewModal] = useState(false)

  const filteredMeasurements = measurements.filter((m) => {
    const matchesSearch =
      m.sampleId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      m.sampleName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      m.material.toLowerCase().includes(searchTerm.toLowerCase()) ||
      m.operator.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesStatus = statusFilter === 'all' || m.status === statusFilter

    return matchesSearch && matchesStatus
  })

  const stats = {
    total: measurements.length,
    completed: measurements.filter((m) => m.status === 'completed').length,
    measuring: measurements.filter((m) => m.status === 'measuring').length,
    pending: measurements.filter((m) => m.status === 'pending').length,
    avgPeakIntensity: Math.round(
      measurements.reduce((sum, m) => sum + m.peakIntensity, 0) / measurements.length
    ),
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800'
      case 'measuring':
        return 'bg-blue-100 text-blue-800'
      case 'pending':
        return 'bg-yellow-100 text-yellow-800'
      case 'failed':
        return 'bg-red-100 text-red-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4" />
      case 'measuring':
        return <Activity className="w-4 h-4 animate-pulse" />
      case 'pending':
        return <Clock className="w-4 h-4" />
      case 'failed':
        return <XCircle className="w-4 h-4" />
      default:
        return null
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
            <Zap className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Raman Spectroscopy</h1>
            <p className="text-gray-600 mt-1">
              Molecular vibration analysis and material characterization
            </p>
          </div>
        </div>
        <button
          onClick={() => setShowNewModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Measurement
        </button>
      </div>

      {/* Stats Dashboard */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
              <BarChart3 className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">{stats.total}</div>
              <div className="text-sm text-gray-600">Total</div>
            </div>
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">{stats.completed}</div>
              <div className="text-sm text-gray-600">Completed</div>
            </div>
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
              <Activity className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">{stats.measuring}</div>
              <div className="text-sm text-gray-600">Measuring</div>
            </div>
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-yellow-100 rounded-lg flex items-center justify-center">
              <Clock className="w-5 h-5 text-yellow-600" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">{stats.pending}</div>
              <div className="text-sm text-gray-600">Pending</div>
            </div>
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-pink-100 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-pink-600" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">
                {stats.avgPeakIntensity.toLocaleString()}
              </div>
              <div className="text-sm text-gray-600">Avg Intensity</div>
            </div>
          </div>
        </div>
      </div>

      {/* Info Card */}
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg shadow-sm border border-purple-200 p-6">
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 bg-purple-600 rounded-lg flex items-center justify-center flex-shrink-0">
            <Info className="w-5 h-5 text-white" />
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              About Raman Spectroscopy
            </h3>
            <p className="text-gray-700 leading-relaxed mb-3">
              Raman spectroscopy is a non-destructive analytical technique based on inelastic
              scattering of monochromatic light (usually laser). It provides information about
              molecular vibrations, crystal structure, and chemical composition by analyzing the
              frequency shift of scattered photons.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-purple-600 rounded-full mt-1.5"></div>
                <div>
                  <span className="font-medium text-gray-900">Molecular fingerprinting:</span>
                  <span className="text-gray-700"> Unique vibrational signatures</span>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-purple-600 rounded-full mt-1.5"></div>
                <div>
                  <span className="font-medium text-gray-900">Non-destructive:</span>
                  <span className="text-gray-700"> No sample preparation needed</span>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-purple-600 rounded-full mt-1.5"></div>
                <div>
                  <span className="font-medium text-gray-900">Crystal quality:</span>
                  <span className="text-gray-700"> Assess defects and strain</span>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-purple-600 rounded-full mt-1.5"></div>
                <div>
                  <span className="font-medium text-gray-900">2D materials:</span>
                  <span className="text-gray-700"> Layer counting and characterization</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search by sample, material, or operator..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-gray-400" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            >
              <option value="all">All Status</option>
              <option value="pending">Pending</option>
              <option value="measuring">Measuring</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
            </select>
          </div>
        </div>
      </div>

      {/* Measurements Table */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Sample
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Material
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Laser
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Strongest Peak
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Intensity
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Identified Peaks
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Operator
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Date
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredMeasurements.map((measurement) => (
                <tr key={measurement.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex flex-col">
                      <div className="text-sm font-medium text-gray-900">{measurement.id}</div>
                      <div className="text-sm text-gray-500">{measurement.sampleName}</div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">{measurement.material}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex flex-col">
                      <div className="text-sm text-gray-900">{measurement.laserWavelength} nm</div>
                      <div className="text-xs text-gray-500">{measurement.powerMw} mW</div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">
                      {measurement.strongestPeak} cm⁻¹
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">
                      {measurement.peakIntensity.toLocaleString()} a.u.
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex flex-wrap gap-1 max-w-xs">
                      {measurement.identifiedPeaks?.slice(0, 2).map((peak, idx) => (
                        <span
                          key={idx}
                          className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800"
                        >
                          {peak}
                        </span>
                      ))}
                      {measurement.identifiedPeaks && measurement.identifiedPeaks.length > 2 && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-600">
                          +{measurement.identifiedPeaks.length - 2}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center gap-2">
                      <User className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-900">{measurement.operator}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center gap-2">
                      <Calendar className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-900">{measurement.date}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(
                        measurement.status
                      )}`}
                    >
                      {getStatusIcon(measurement.status)}
                      {measurement.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {filteredMeasurements.length === 0 && (
          <div className="text-center py-12">
            <Zap className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-sm font-medium text-gray-900 mb-1">No measurements found</h3>
            <p className="text-sm text-gray-500">Try adjusting your search or filters</p>
          </div>
        )}
      </div>

      {/* New Measurement Modal */}
      {showNewModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">New Raman Measurement</h2>
            </div>
            <div className="p-6 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Sample ID
                  </label>
                  <input
                    type="text"
                    placeholder="S-2024-XXX"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Sample Name
                  </label>
                  <input
                    type="text"
                    placeholder="Enter sample name"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Material</label>
                  <input
                    type="text"
                    placeholder="e.g., Si, GaN, Graphene"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Laser Wavelength (nm)
                  </label>
                  <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent">
                    <option>532 nm (Green)</option>
                    <option>633 nm (Red)</option>
                    <option>785 nm (NIR)</option>
                    <option>1064 nm (NIR)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Power (mW)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    placeholder="0.5 - 50"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Exposure Time (s)
                  </label>
                  <input
                    type="number"
                    placeholder="1 - 60"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Accumulations
                  </label>
                  <input
                    type="number"
                    placeholder="1 - 10"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Operator</label>
                  <input
                    type="text"
                    placeholder="Enter operator name"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Notes</label>
                <textarea
                  rows={3}
                  placeholder="Additional notes about the measurement..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                ></textarea>
              </div>
            </div>
            <div className="p-6 border-t border-gray-200 flex justify-end gap-3">
              <button
                onClick={() => setShowNewModal(false)}
                className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowNewModal(false)
                }}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
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
