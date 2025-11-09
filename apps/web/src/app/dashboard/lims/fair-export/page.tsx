'use client'
import { useState } from 'react'
import { Database, Plus, Search, Download, CheckCircle, AlertCircle, Clock, FileText, Shield, Globe } from 'lucide-react'

interface FAIRExport {
  id: string
  datasetName: string
  datasetId: string
  schema: 'JSON-LD' | 'RO-Crate' | 'DataCite' | 'DCAT' | 'Schema.org'
  status: 'validating' | 'ready' | 'exported' | 'failed'
  fairScore: number
  createdBy: string
  createdDate: string
  exportedDate?: string
  size?: string
  repository?: string
  doi?: string
}

const mockExports: FAIRExport[] = [
  {
    id: 'FAIR-001',
    datasetName: 'Silicon Wafer Characterization Dataset Q4 2024',
    datasetId: 'DS-2024-WAF-001',
    schema: 'RO-Crate',
    status: 'exported',
    fairScore: 98,
    createdBy: 'Dr. Chen',
    createdDate: '2024-11-08',
    exportedDate: '2024-11-08',
    size: '45.2 MB',
    repository: 'Zenodo',
    doi: '10.5281/zenodo.1234567'
  },
  {
    id: 'FAIR-002',
    datasetName: 'Thin Film Deposition Process Data',
    datasetId: 'DS-2024-FILM-002',
    schema: 'JSON-LD',
    status: 'ready',
    fairScore: 95,
    createdBy: 'Dr. Martinez',
    createdDate: '2024-11-09',
    size: '12.8 MB'
  },
  {
    id: 'FAIR-003',
    datasetName: 'Device Failure Analysis Results',
    datasetId: 'DS-2024-DEV-003',
    schema: 'DataCite',
    status: 'validating',
    fairScore: 87,
    createdBy: 'Dr. Patel',
    createdDate: '2024-11-09'
  },
  {
    id: 'FAIR-004',
    datasetName: 'Cleanroom Environmental Monitoring 2024',
    datasetId: 'DS-2024-ENV-004',
    schema: 'DCAT',
    status: 'exported',
    fairScore: 92,
    createdBy: 'Lab Manager',
    createdDate: '2024-11-07',
    exportedDate: '2024-11-07',
    size: '8.4 MB',
    repository: 'Figshare',
    doi: '10.6084/m9.figshare.9876543'
  },
  {
    id: 'FAIR-005',
    datasetName: 'GaN Material Properties Dataset',
    datasetId: 'DS-2024-MAT-005',
    schema: 'Schema.org',
    status: 'failed',
    fairScore: 65,
    createdBy: 'Dr. Kim',
    createdDate: '2024-11-09'
  }
]

const statusConfig = {
  validating: { color: 'bg-blue-100 text-blue-800', icon: Clock },
  ready: { color: 'bg-green-100 text-green-800', icon: CheckCircle },
  exported: { color: 'bg-purple-100 text-purple-800', icon: Database },
  failed: { color: 'bg-red-100 text-red-800', icon: AlertCircle }
}

const schemaColors = {
  'JSON-LD': 'bg-blue-100 text-blue-800',
  'RO-Crate': 'bg-green-100 text-green-800',
  'DataCite': 'bg-orange-100 text-orange-800',
  'DCAT': 'bg-purple-100 text-purple-800',
  'Schema.org': 'bg-teal-100 text-teal-800'
}

export default function FAIRExportPage() {
  const [exports, setExports] = useState(mockExports)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [showNewExport, setShowNewExport] = useState(false)

  const filteredExports = exports.filter(exp => {
    const matchesSearch =
      exp.datasetName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      exp.datasetId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      exp.createdBy.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesStatus = statusFilter === 'all' || exp.status === statusFilter

    return matchesSearch && matchesStatus
  })

  const getFairColor = (score: number) => {
    if (score >= 90) return 'text-green-600'
    if (score >= 75) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getFairLabel = (score: number) => {
    if (score >= 90) return 'Excellent'
    if (score >= 75) return 'Good'
    return 'Needs Improvement'
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-xl flex items-center justify-center">
            <Database className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">FAIR Data Export</h1>
            <p className="text-gray-600 mt-1">Export datasets with FAIR principles compliance</p>
          </div>
        </div>
        <button
          onClick={() => setShowNewExport(true)}
          className="flex items-center gap-2 bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition-colors"
        >
          <Plus className="w-5 h-5" />
          New Export
        </button>
      </div>

      {/* FAIR Principles Info */}
      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg border border-indigo-200 p-6">
        <h3 className="text-lg font-semibold text-indigo-900 mb-3">FAIR Data Principles</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="flex items-start gap-2">
            <Shield className="w-5 h-5 text-indigo-600 mt-0.5" />
            <div>
              <p className="font-medium text-indigo-900">Findable</p>
              <p className="text-sm text-indigo-700">Metadata and data should be easy to find</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <Globe className="w-5 h-5 text-indigo-600 mt-0.5" />
            <div>
              <p className="font-medium text-indigo-900">Accessible</p>
              <p className="text-sm text-indigo-700">Once found, know how to access the data</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <FileText className="w-5 h-5 text-indigo-600 mt-0.5" />
            <div>
              <p className="font-medium text-indigo-900">Interoperable</p>
              <p className="text-sm text-indigo-700">Data can be integrated with other data</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <Database className="w-5 h-5 text-indigo-600 mt-0.5" />
            <div>
              <p className="font-medium text-indigo-900">Reusable</p>
              <p className="text-sm text-indigo-700">Data can be used in future research</p>
            </div>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Exports</p>
              <p className="text-2xl font-bold text-gray-900">{exports.length}</p>
            </div>
            <Database className="w-8 h-8 text-indigo-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Exported</p>
              <p className="text-2xl font-bold text-purple-600">
                {exports.filter(e => e.status === 'exported').length}
              </p>
            </div>
            <CheckCircle className="w-8 h-8 text-purple-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Ready</p>
              <p className="text-2xl font-bold text-green-600">
                {exports.filter(e => e.status === 'ready').length}
              </p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Avg FAIR Score</p>
              <p className="text-2xl font-bold text-indigo-600">
                {Math.round(exports.reduce((sum, e) => sum + e.fairScore, 0) / exports.length)}
              </p>
            </div>
            <Shield className="w-8 h-8 text-indigo-500" />
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
              placeholder="Search by dataset name, ID, or creator..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
          </div>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          >
            <option value="all">All Status</option>
            <option value="validating">Validating</option>
            <option value="ready">Ready</option>
            <option value="exported">Exported</option>
            <option value="failed">Failed</option>
          </select>
        </div>
      </div>

      {/* Exports List */}
      <div className="space-y-4">
        {filteredExports.map((exp) => {
          const StatusIcon = statusConfig[exp.status].icon
          return (
            <div key={exp.id} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="text-lg font-semibold text-gray-900">{exp.datasetName}</h3>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${schemaColors[exp.schema]}`}>
                      {exp.schema}
                    </span>
                  </div>
                  <p className="text-sm text-gray-500 mb-3">
                    Export ID: {exp.id} • Dataset: {exp.datasetId}
                  </p>
                </div>
                <span className={`px-3 py-1 rounded-full text-sm font-medium flex items-center gap-1 ${statusConfig[exp.status].color}`}>
                  <StatusIcon className="w-4 h-4" />
                  {exp.status}
                </span>
              </div>

              {/* FAIR Score */}
              <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-4 mb-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Shield className="w-8 h-8 text-indigo-600" />
                    <div>
                      <p className="text-xs text-gray-600 uppercase font-medium">FAIR Compliance Score</p>
                      <p className={`text-2xl font-bold ${getFairColor(exp.fairScore)}`}>
                        {exp.fairScore}/100
                      </p>
                    </div>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${getFairColor(exp.fairScore)} bg-white`}>
                    {getFairLabel(exp.fairScore)}
                  </span>
                </div>
                {/* Progress bar */}
                <div className="mt-3 bg-white rounded-full h-2 overflow-hidden">
                  <div
                    className={`h-full ${exp.fairScore >= 90 ? 'bg-green-500' : exp.fairScore >= 75 ? 'bg-yellow-500' : 'bg-red-500'}`}
                    style={{ width: `${exp.fairScore}%` }}
                  />
                </div>
              </div>

              {/* Metadata */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div className="bg-gray-50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 uppercase font-medium mb-1">Created By</p>
                  <p className="text-sm text-gray-900">{exp.createdBy}</p>
                  <p className="text-xs text-gray-500 mt-1">Date: {exp.createdDate}</p>
                </div>
                {exp.exportedDate && (
                  <div className="bg-gray-50 rounded-lg p-3">
                    <p className="text-xs text-gray-500 uppercase font-medium mb-1">Exported</p>
                    <p className="text-sm text-gray-900">Date: {exp.exportedDate}</p>
                    {exp.size && <p className="text-xs text-gray-500 mt-1">Size: {exp.size}</p>}
                  </div>
                )}
              </div>

              {/* Repository & DOI */}
              {exp.doi && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-3 mb-4">
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-sm font-medium text-green-900 mb-1">
                        <Globe className="w-4 h-4 inline mr-1" />
                        Published to {exp.repository}
                      </p>
                      <p className="text-sm text-green-700">
                        DOI: <a href={`https://doi.org/${exp.doi}`} className="underline hover:text-green-900" target="_blank" rel="noopener noreferrer">
                          {exp.doi}
                        </a>
                      </p>
                    </div>
                    <button className="flex items-center gap-1 bg-green-600 text-white px-3 py-1 rounded text-sm hover:bg-green-700">
                      <Download className="w-4 h-4" />
                      Download
                    </button>
                  </div>
                </div>
              )}

              {exp.status === 'failed' && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
                  <p className="text-sm text-red-800">
                    <AlertCircle className="w-4 h-4 inline mr-1" />
                    Validation failed. FAIR score below threshold. Please review metadata completeness.
                  </p>
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-2 pt-4 border-t border-gray-200">
                <button className="text-indigo-600 hover:text-indigo-900 text-sm font-medium">
                  View Metadata
                </button>
                {exp.status === 'ready' && (
                  <>
                    <button className="text-green-600 hover:text-green-900 text-sm font-medium">
                      Export to Repository
                    </button>
                    <button className="text-blue-600 hover:text-blue-900 text-sm font-medium">
                      Download Package
                    </button>
                  </>
                )}
                {exp.status === 'failed' && (
                  <button className="text-orange-600 hover:text-orange-900 text-sm font-medium">
                    Re-validate
                  </button>
                )}
                <button className="text-purple-600 hover:text-purple-900 text-sm font-medium">
                  Validation Report
                </button>
                <button className="text-gray-600 hover:text-gray-900 text-sm font-medium">
                  Edit
                </button>
              </div>
            </div>
          )
        })}

        {filteredExports.length === 0 && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
            <Database className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No FAIR exports found matching your criteria</p>
          </div>
        )}
      </div>

      {/* New Export Modal */}
      {showNewExport && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <h2 className="text-2xl font-bold mb-4">Create FAIR Data Export</h2>

            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Dataset Name</label>
                <input
                  type="text"
                  placeholder="Enter descriptive dataset name"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Select Data Source</label>
                <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500">
                  <option>Sample Tracking Data</option>
                  <option>Lab Notebook Entries</option>
                  <option>Experimental Results</option>
                  <option>Quality Control Records</option>
                  <option>Equipment Logs</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Metadata Schema</label>
                <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500">
                  <option>RO-Crate (Research Object)</option>
                  <option>JSON-LD (Linked Data)</option>
                  <option>DataCite</option>
                  <option>DCAT (Data Catalog)</option>
                  <option>Schema.org</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Target Repository</label>
                <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500">
                  <option>Zenodo</option>
                  <option>Figshare</option>
                  <option>Dryad</option>
                  <option>OSF (Open Science Framework)</option>
                  <option>Institutional Repository</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">License</label>
                <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500">
                  <option>CC BY 4.0</option>
                  <option>CC BY-SA 4.0</option>
                  <option>CC0 (Public Domain)</option>
                  <option>MIT License</option>
                  <option>Custom</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">FAIR Options</label>
                <div className="space-y-2">
                  <label className="flex items-center">
                    <input type="checkbox" className="mr-2" defaultChecked />
                    Generate persistent identifier (DOI)
                  </label>
                  <label className="flex items-center">
                    <input type="checkbox" className="mr-2" defaultChecked />
                    Include rich metadata
                  </label>
                  <label className="flex items-center">
                    <input type="checkbox" className="mr-2" defaultChecked />
                    Add ontology terms
                  </label>
                  <label className="flex items-center">
                    <input type="checkbox" className="mr-2" />
                    Embargo period (specify duration)
                  </label>
                </div>
              </div>
            </div>

            <div className="flex gap-2">
              <button
                onClick={() => setShowNewExport(false)}
                className="flex-1 bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300"
              >
                Cancel
              </button>
              <button
                onClick={() => setShowNewExport(false)}
                className="flex-1 bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700"
              >
                Create Export
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
