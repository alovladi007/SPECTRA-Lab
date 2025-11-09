'use client'
import { useState, useEffect } from 'react'
import { FileText, Plus, Search, Download, Calendar, User, Clock, CheckCircle, AlertCircle, FileSpreadsheet, Trash2 } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter
} from '@/components/ui/dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

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

interface NewReportForm {
  name: string
  type: 'sample_analysis' | 'experiment_summary' | 'compliance' | 'inventory' | 'quality_control'
  format: 'PDF' | 'Excel' | 'CSV' | 'JSON'
  dateFrom: string
  dateTo: string
}

const generateMockReports = (): Report[] => [
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
  },
  {
    id: 'RPT-2024-021',
    name: 'Safety Compliance Summary',
    type: 'compliance',
    format: 'PDF',
    status: 'completed',
    requestedBy: 'Safety Officer',
    requestDate: '2024-11-08 10:00',
    completedDate: '2024-11-08 10:30',
    parameters: 'Month: October 2024 | Include: Training, Incidents',
    size: '1.8 MB'
  },
  {
    id: 'RPT-2024-022',
    name: 'Equipment Utilization Report',
    type: 'inventory',
    format: 'Excel',
    status: 'completed',
    requestedBy: 'Dr. Wilson',
    requestDate: '2024-11-07 15:00',
    completedDate: '2024-11-07 15:20',
    parameters: 'Date Range: Oct 1-31, 2024 | All Equipment',
    size: '645 KB'
  }
]

const statusConfig = {
  pending: { color: 'bg-yellow-100 text-yellow-800', icon: Clock, label: 'Pending' },
  generating: { color: 'bg-blue-100 text-blue-800', icon: Clock, label: 'Generating' },
  completed: { color: 'bg-green-100 text-green-800', icon: CheckCircle, label: 'Completed' },
  failed: { color: 'bg-red-100 text-red-800', icon: AlertCircle, label: 'Failed' }
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
  const [reports, setReports] = useState<Report[]>([])
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [showNewReport, setShowNewReport] = useState(false)
  const [selectedReport, setSelectedReport] = useState<Report | null>(null)
  const [showDetailsDialog, setShowDetailsDialog] = useState(false)

  const [newReport, setNewReport] = useState<NewReportForm>({
    name: '',
    type: 'sample_analysis',
    format: 'PDF',
    dateFrom: '',
    dateTo: ''
  })

  // Generate mock data on client side to prevent hydration errors
  useEffect(() => {
    setReports(generateMockReports())
  }, [])

  const filteredReports = reports.filter(report => {
    const matchesSearch =
      report.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      report.type.toLowerCase().includes(searchTerm.toLowerCase()) ||
      report.requestedBy.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesStatus = statusFilter === 'all' || report.status === statusFilter

    return matchesSearch && matchesStatus
  })

  const handleCreateReport = () => {
    if (!newReport.name || !newReport.dateFrom || !newReport.dateTo) {
      alert('Please fill in all required fields')
      return
    }

    const currentUser = 'Current User'
    const report: Report = {
      id: `RPT-2024-${String(reports.length + 1).padStart(3, '0')}`,
      name: newReport.name,
      type: newReport.type,
      format: newReport.format,
      status: 'pending',
      requestedBy: currentUser,
      requestDate: new Date().toISOString().slice(0, 16).replace('T', ' '),
      parameters: `Date Range: ${newReport.dateFrom} to ${newReport.dateTo}`
    }

    setReports([report, ...reports])
    setShowNewReport(false)
    setNewReport({
      name: '',
      type: 'sample_analysis',
      format: 'PDF',
      dateFrom: '',
      dateTo: ''
    })
  }

  const handleRetryGeneration = (reportId: string) => {
    setReports(reports.map(report => {
      if (report.id === reportId && report.status === 'failed') {
        return { ...report, status: 'pending' as const }
      }
      return report
    }))
  }

  const handleDelete = (reportId: string) => {
    if (confirm('Are you sure you want to delete this report?')) {
      setReports(reports.filter(report => report.id !== reportId))
    }
  }

  const handleViewDetails = (report: Report) => {
    setSelectedReport(report)
    setShowDetailsDialog(true)
  }

  const stats = {
    total: reports.length,
    completed: reports.filter(r => r.status === 'completed').length,
    generating: reports.filter(r => r.status === 'generating').length,
    pending: reports.filter(r => r.status === 'pending').length
  }

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
        <Button
          onClick={() => setShowNewReport(true)}
          className="flex items-center gap-2"
        >
          <Plus className="w-5 h-5" />
          Generate Report
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Reports</p>
              <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
            </div>
            <FileText className="w-8 h-8 text-cyan-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Completed</p>
              <p className="text-2xl font-bold text-green-600">{stats.completed}</p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Generating</p>
              <p className="text-2xl font-bold text-blue-600">{stats.generating}</p>
            </div>
            <Clock className="w-8 h-8 text-blue-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Pending</p>
              <p className="text-2xl font-bold text-yellow-600">{stats.pending}</p>
            </div>
            <Clock className="w-8 h-8 text-yellow-500" />
          </div>
        </Card>
      </div>

      {/* Filters */}
      <Card className="p-4">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <Input
              type="text"
              placeholder="Search by name, type, or requester..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-full md:w-[200px]">
              <SelectValue placeholder="All Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="pending">Pending</SelectItem>
              <SelectItem value="generating">Generating</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="failed">Failed</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </Card>

      {/* Reports List */}
      <div className="space-y-4">
        {filteredReports.map((report) => {
          const StatusIcon = statusConfig[report.status].icon
          const FormatIcon = formatIcons[report.format]
          return (
            <Card key={report.id} className="p-6 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2 flex-wrap">
                    <h3 className="text-lg font-semibold text-gray-900">{report.name}</h3>
                    <Badge className={typeColors[report.type]}>
                      {report.type.replace('_', ' ')}
                    </Badge>
                    <Badge variant="outline" className="flex items-center gap-1">
                      <FormatIcon className="w-3 h-3" />
                      {report.format}
                    </Badge>
                  </div>
                  <p className="text-sm text-gray-500 mb-3">Report ID: {report.id}</p>
                </div>
                <Badge className={statusConfig[report.status].color}>
                  <StatusIcon className="w-3 h-3 mr-1" />
                  {statusConfig[report.status].label}
                </Badge>
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
                    <Button size="sm" className="bg-green-600 hover:bg-green-700">
                      <Download className="w-4 h-4 mr-1" />
                      Download
                    </Button>
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
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleViewDetails(report)}
                  className="text-cyan-600 hover:text-cyan-900"
                >
                  View Details
                </Button>
                {report.status === 'completed' && (
                  <>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-blue-600 hover:text-blue-900"
                    >
                      Share
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-green-600 hover:text-green-900"
                    >
                      Schedule
                    </Button>
                  </>
                )}
                {report.status === 'failed' && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleRetryGeneration(report.id)}
                    className="text-orange-600 hover:text-orange-900"
                  >
                    Retry Generation
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleDelete(report.id)}
                  className="text-red-600 hover:text-red-900"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </div>
            </Card>
          )
        })}

        {filteredReports.length === 0 && (
          <Card className="p-12 text-center">
            <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No reports found matching your criteria</p>
          </Card>
        )}
      </div>

      {/* New Report Dialog */}
      <Dialog open={showNewReport} onOpenChange={setShowNewReport}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Generate New Report</DialogTitle>
            <DialogDescription>
              Configure and generate a new lab report
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Report Name *
              </label>
              <Input
                value={newReport.name}
                onChange={(e) => setNewReport({ ...newReport, name: e.target.value })}
                placeholder="e.g., Weekly Sample Analysis Report"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Report Type *
              </label>
              <Select
                value={newReport.type}
                onValueChange={(value: any) => setNewReport({ ...newReport, type: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="sample_analysis">Sample Analysis Report</SelectItem>
                  <SelectItem value="experiment_summary">Experiment Summary Report</SelectItem>
                  <SelectItem value="compliance">Compliance Audit Report</SelectItem>
                  <SelectItem value="inventory">Inventory Status Report</SelectItem>
                  <SelectItem value="quality_control">Quality Control Metrics</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Output Format *
              </label>
              <Select
                value={newReport.format}
                onValueChange={(value: any) => setNewReport({ ...newReport, format: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="PDF">PDF</SelectItem>
                  <SelectItem value="Excel">Excel</SelectItem>
                  <SelectItem value="CSV">CSV</SelectItem>
                  <SelectItem value="JSON">JSON</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Date Range *
              </label>
              <div className="grid grid-cols-2 gap-4">
                <Input
                  type="date"
                  value={newReport.dateFrom}
                  onChange={(e) => setNewReport({ ...newReport, dateFrom: e.target.value })}
                />
                <Input
                  type="date"
                  value={newReport.dateTo}
                  onChange={(e) => setNewReport({ ...newReport, dateTo: e.target.value })}
                />
              </div>
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowNewReport(false)}
            >
              Cancel
            </Button>
            <Button onClick={handleCreateReport}>
              Generate Report
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Details Dialog */}
      <Dialog open={showDetailsDialog} onOpenChange={setShowDetailsDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Report Details</DialogTitle>
            <DialogDescription>
              {selectedReport?.id}
            </DialogDescription>
          </DialogHeader>

          {selectedReport && (
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-bold text-gray-900 mb-2">{selectedReport.name}</h3>
                <div className="flex flex-wrap gap-2 mb-4">
                  <Badge className={typeColors[selectedReport.type]}>
                    {selectedReport.type.replace('_', ' ')}
                  </Badge>
                  <Badge className={statusConfig[selectedReport.status].color}>
                    {statusConfig[selectedReport.status].label}
                  </Badge>
                  <Badge variant="outline">{selectedReport.format}</Badge>
                </div>
              </div>

              <div className="border-t border-gray-200 pt-4">
                <p className="text-sm font-medium text-gray-500 mb-2">Parameters</p>
                <div className="bg-gray-50 rounded-lg p-3">
                  <p className="text-sm text-gray-900">{selectedReport.parameters}</p>
                </div>
              </div>

              <div className="border-t border-gray-200 pt-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">Requested By</p>
                    <p className="text-base text-gray-900">{selectedReport.requestedBy}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Request Date</p>
                    <p className="text-base text-gray-900">{selectedReport.requestDate}</p>
                  </div>
                  {selectedReport.completedDate && (
                    <div>
                      <p className="text-sm font-medium text-gray-500">Completed Date</p>
                      <p className="text-base text-gray-900">{selectedReport.completedDate}</p>
                    </div>
                  )}
                  {selectedReport.size && (
                    <div>
                      <p className="text-sm font-medium text-gray-500">File Size</p>
                      <p className="text-base text-gray-900">{selectedReport.size}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          <DialogFooter>
            <Button onClick={() => setShowDetailsDialog(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
