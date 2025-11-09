'use client'
import { useState } from 'react'
import { FileText, Plus, Search, Download, Calendar, User, Clock, CheckCircle, AlertCircle, FileSpreadsheet } from 'lucide-react'

interface Report {
  id: string
  name: string
  type: 'sample_analysis' | 'experiment_summary' | 'compliance' | 'inventory' | 'quality_control'
  format: 'PDF' | 'Excel' | 'CSV' | 'JSON'
  status: 'pending' | 'generating' | 'completed' | 'failed'
  requestedBy: string
  requestDate: string
  completedDate?: string
  parameters: string
  size?: string
}

const mockReports: Report[] = [
  {
    id: 'RPT-2024-015',
    name: 'Weekly Sample Analysis Report',
    type: 'sample_analysis',
    format: 'PDF',
    status: 'completed',
    requestedBy: 'Dr. Chen',
    requestDate: '2024-11-09 08:00',
    completedDate: '2024-11-09 08:15',
    parameters: 'Date Range: Nov 2-9, 2024 | Samples: All Active',
    size: '2.4 MB'
  },
  {
    id: 'RPT-2024-016',
    name: 'Q4 Compliance Audit Report',
    type: 'compliance',
    format: 'PDF',
    status: 'generating',
    requestedBy: 'Lab Manager',
    requestDate: '2024-11-09 09:30',
    parameters: 'Quarter: Q4 2024 | Include: SOPs, Signatures, COC'
  },
  {
    id: 'RPT-2024-017',
    name: 'Experiment Summary - Device Testing',
    type: 'experiment_summary',
    format: 'Excel',
    status: 'completed',
    requestedBy: 'Dr. Patel',
    requestDate: '2024-11-08 14:00',
    completedDate: '2024-11-08 14:25',
    parameters: 'ELN IDs: ELN-2024-001 to ELN-2024-005',
    size: '856 KB'
  },
  {
    id: 'RPT-2024-018',
    name: 'Inventory Status Report',
    type: 'inventory',
    format: 'CSV',
    status: 'completed',
    requestedBy: 'Dr. Martinez',
    requestDate: '2024-11-09 07:00',
    completedDate: '2024-11-09 07:05',
    parameters: 'All Locations | Include: Stock Levels, Expiry Dates',
    size: '124 KB'
  },
  {
    id: 'RPT-2024-019',
    name: 'Quality Control Metrics',
    type: 'quality_control',
    format: 'PDF',
    status: 'pending',
    requestedBy: 'Dr. Kim',
    requestDate: '2024-11-09 10:45',
    parameters: 'Date Range: Oct 1 - Nov 9, 2024 | Include: SPC Charts'
  },
  {
    id: 'RPT-2024-020',
    name: 'Monthly Sample Tracking Export',
    type: 'sample_analysis',
    format: 'JSON',
    status: 'failed',
    requestedBy: 'Lab Manager',
    requestDate: '2024-11-09 06:00',
    parameters: 'Month: October 2024 | Format: JSON-LD'
  }
]

const statusConfig = {
  pending: { color: 'bg-yellow-100 text-yellow-800', icon: Clock },
  generating: { color: 'bg-blue-100 text-blue-800', icon: Clock },
  completed: { color: 'bg-green-100 text-green-800', icon: CheckCircle },
  failed: { color: 'bg-red-100 text-red-800', icon: AlertCircle }
}

const typeColors = {
  sample_analysis: 'bg-teal-100 text-teal-800',
  experiment_summary: 'bg-blue-100 text-blue-800',
  compliance: 'bg-purple-100 text-purple-800',
  inventory: 'bg-orange-100 text-orange-800',
  quality_control: 'bg-green-100 text-green-800'
}

const formatIcons = {
  PDF: FileText,
  Excel: FileSpreadsheet,
  CSV: FileSpreadsheet,
  JSON: FileText
}

export default function ReportsPage() {
  const [reports, setReports] = useState(mockReports)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [showNewReport, setShowNewReport] = useState(false)

  const filteredReports = reports.filter(report => {
    const matchesSearch =
      report.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      report.type.toLowerCase().includes(searchTerm.toLowerCase()) ||
      report.requestedBy.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesStatus = statusFilter === 'all' || report.status === statusFilter

    return matchesSearch && matchesStatus
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-blue-500 rounded-xl flex items-center justify-center">
            <FileText className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Report Generator</h1>
            <p className="text-gray-600 mt-1">Generate and download comprehensive lab reports</p>
          </div>
        </div>
        <button
          onClick={() => setShowNewReport(true)}
          className="flex items-center gap-2 bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-700 transition-colors"
        >
          <Plus className="w-5 h-5" />
          Generate Report
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Reports</p>
              <p className="text-2xl font-bold text-gray-900">{reports.length}</p>
            </div>
            <FileText className="w-8 h-8 text-cyan-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Completed</p>
              <p className="text-2xl font-bold text-green-600">
                {reports.filter(r => r.status === 'completed').length}
              </p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Generating</p>
              <p className="text-2xl font-bold text-blue-600">
                {reports.filter(r => r.status === 'generating').length}
              </p>
            </div>
            <Clock className="w-8 h-8 text-blue-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Pending</p>
              <p className="text-2xl font-bold text-yellow-600">
                {reports.filter(r => r.status === 'pending').length}
              </p>
            </div>
            <Clock className="w-8 h-8 text-yellow-500" />
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
              placeholder="Search by name, type, or requester..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
            />
          </div>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
          >
            <option value="all">All Status</option>
            <option value="pending">Pending</option>
            <option value="generating">Generating</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
        </div>
      </div>

      {/* Reports List */}
      <div className="space-y-4">
        {filteredReports.map((report) => {
          const StatusIcon = statusConfig[report.status].icon
          const FormatIcon = formatIcons[report.format]
          return (
            <div key={report.id} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="text-lg font-semibold text-gray-900">{report.name}</h3>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${typeColors[report.type]}`}>
                      {report.type.replace('_', ' ')}
                    </span>
                    <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded-full text-xs font-medium flex items-center gap-1">
                      <FormatIcon className="w-3 h-3" />
                      {report.format}
                    </span>
                  </div>
                  <p className="text-sm text-gray-500 mb-3">Report ID: {report.id}</p>
                </div>
                <span className={`px-3 py-1 rounded-full text-sm font-medium flex items-center gap-1 ${statusConfig[report.status].color}`}>
                  <StatusIcon className="w-4 h-4" />
                  {report.status}
                </span>
              </div>

              <div className="bg-gray-50 rounded-lg p-4 mb-4">
                <p className="text-xs text-gray-500 uppercase font-medium mb-2">Report Parameters</p>
                <p className="text-sm text-gray-900">{report.parameters}</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                <div className="flex items-center gap-2 text-sm text-gray-600">
                  <User className="w-4 h-4 text-gray-400" />
                  {report.requestedBy}
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-600">
                  <Calendar className="w-4 h-4 text-gray-400" />
                  Requested: {report.requestDate}
                </div>
                {report.completedDate && (
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <CheckCircle className="w-4 h-4 text-gray-400" />
                    Completed: {report.completedDate}
                  </div>
                )}
                {report.size && (
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <FileText className="w-4 h-4 text-gray-400" />
                    Size: {report.size}
                  </div>
                )}
              </div>

              {report.status === 'completed' && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-3 mb-4">
                  <div className="flex items-center justify-between">
                    <p className="text-sm text-green-800">
                      <CheckCircle className="w-4 h-4 inline mr-1" />
                      Report ready for download
                    </p>
                    <button className="flex items-center gap-1 bg-green-600 text-white px-3 py-1 rounded text-sm hover:bg-green-700">
                      <Download className="w-4 h-4" />
                      Download
                    </button>
                  </div>
                </div>
              )}

              {report.status === 'failed' && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
                  <p className="text-sm text-red-800">
                    <AlertCircle className="w-4 h-4 inline mr-1" />
                    Report generation failed. Please try again or contact support.
                  </p>
                </div>
              )}

              <div className="flex gap-2 pt-4 border-t border-gray-200">
                <button className="text-cyan-600 hover:text-cyan-900 text-sm font-medium">
                  View Details
                </button>
                {report.status === 'completed' && (
                  <>
                    <button className="text-blue-600 hover:text-blue-900 text-sm font-medium">
                      Share
                    </button>
                    <button className="text-green-600 hover:text-green-900 text-sm font-medium">
                      Schedule
                    </button>
                  </>
                )}
                {report.status === 'failed' && (
                  <button className="text-orange-600 hover:text-orange-900 text-sm font-medium">
                    Retry Generation
                  </button>
                )}
                <button className="text-gray-600 hover:text-gray-900 text-sm font-medium">
                  Delete
                </button>
              </div>
            </div>
          )
        })}

        {filteredReports.length === 0 && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
            <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No reports found matching your criteria</p>
          </div>
        )}
      </div>

      {/* New Report Modal */}
      {showNewReport && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <h2 className="text-2xl font-bold mb-4">Generate New Report</h2>

            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Report Type</label>
                <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-cyan-500">
                  <option>Sample Analysis Report</option>
                  <option>Experiment Summary Report</option>
                  <option>Compliance Audit Report</option>
                  <option>Inventory Status Report</option>
                  <option>Quality Control Metrics</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Output Format</label>
                <div className="flex gap-4">
                  <label className="flex items-center">
                    <input type="radio" name="format" value="pdf" className="mr-2" defaultChecked />
                    PDF
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="format" value="excel" className="mr-2" />
                    Excel
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="format" value="csv" className="mr-2" />
                    CSV
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="format" value="json" className="mr-2" />
                    JSON
                  </label>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Date Range</label>
                <div className="grid grid-cols-2 gap-4">
                  <input type="date" className="px-3 py-2 border border-gray-300 rounded-lg" />
                  <input type="date" className="px-3 py-2 border border-gray-300 rounded-lg" />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Additional Options</label>
                <div className="space-y-2">
                  <label className="flex items-center">
                    <input type="checkbox" className="mr-2" />
                    Include detailed statistics
                  </label>
                  <label className="flex items-center">
                    <input type="checkbox" className="mr-2" />
                    Include charts and graphs
                  </label>
                  <label className="flex items-center">
                    <input type="checkbox" className="mr-2" />
                    Include raw data appendix
                  </label>
                </div>
              </div>
            </div>

            <div className="flex gap-2">
              <button
                onClick={() => setShowNewReport(false)}
                className="flex-1 bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300"
              >
                Cancel
              </button>
              <button
                onClick={() => setShowNewReport(false)}
                className="flex-1 bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-700"
              >
                Generate Report
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
