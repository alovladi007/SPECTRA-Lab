'use client'
import { useState, useEffect } from 'react'
import { Database, Plus, Search, Download, CheckCircle, AlertCircle, Clock, FileText, Shield, Globe, Trash2 } from 'lucide-react'
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

interface NewExportForm {
  datasetName: string
  dataSource: string
  schema: 'JSON-LD' | 'RO-Crate' | 'DataCite' | 'DCAT' | 'Schema.org'
  repository: string
}

const generateMockExports = (): FAIRExport[] => [
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
  },
  {
    id: 'FAIR-006',
    datasetName: 'Photolithography Process Optimization Data',
    datasetId: 'DS-2024-PHOTO-006',
    schema: 'RO-Crate',
    status: 'ready',
    fairScore: 94,
    createdBy: 'Dr. Wilson',
    createdDate: '2024-11-08',
    size: '23.6 MB'
  }
]

const statusConfig = {
  validating: { color: 'bg-blue-100 text-blue-800', icon: Clock, label: 'Validating' },
  ready: { color: 'bg-green-100 text-green-800', icon: CheckCircle, label: 'Ready' },
  exported: { color: 'bg-purple-100 text-purple-800', icon: Database, label: 'Exported' },
  failed: { color: 'bg-red-100 text-red-800', icon: AlertCircle, label: 'Failed' }
}

const schemaColors = {
  'JSON-LD': 'bg-blue-100 text-blue-800',
  'RO-Crate': 'bg-green-100 text-green-800',
  'DataCite': 'bg-orange-100 text-orange-800',
  'DCAT': 'bg-purple-100 text-purple-800',
  'Schema.org': 'bg-teal-100 text-teal-800'
}

export default function FAIRExportPage() {
  const [exports, setExports] = useState<FAIRExport[]>([])
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [showNewExport, setShowNewExport] = useState(false)
  const [selectedExport, setSelectedExport] = useState<FAIRExport | null>(null)
  const [showDetailsDialog, setShowDetailsDialog] = useState(false)

  const [newExport, setNewExport] = useState<NewExportForm>({
    datasetName: '',
    dataSource: '',
    schema: 'RO-Crate',
    repository: 'Zenodo'
  })

  // Generate mock data on client side to prevent hydration errors
  useEffect(() => {
    setExports(generateMockExports())
  }, [])

  const filteredExports = exports.filter(exp => {
    const matchesSearch =
      exp.datasetName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      exp.datasetId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      exp.createdBy.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesStatus = statusFilter === 'all' || exp.status === statusFilter

    return matchesSearch && matchesStatus
  })

  const handleCreateExport = () => {
    if (!newExport.datasetName || !newExport.dataSource) {
      alert('Please fill in all required fields')
      return
    }

    const currentUser = 'Current User'
    const fairExport: FAIRExport = {
      id: `FAIR-${String(exports.length + 1).padStart(3, '0')}`,
      datasetName: newExport.datasetName,
      datasetId: `DS-2024-${String(exports.length + 1).padStart(3, '0')}`,
      schema: newExport.schema,
      status: 'validating',
      fairScore: 0,
      createdBy: currentUser,
      createdDate: new Date().toISOString().split('T')[0]
    }

    setExports([fairExport, ...exports])
    setShowNewExport(false)
    setNewExport({
      datasetName: '',
      dataSource: '',
      schema: 'RO-Crate',
      repository: 'Zenodo'
    })
  }

  const handleRevalidate = (exportId: string) => {
    setExports(exports.map(exp => {
      if (exp.id === exportId && exp.status === 'failed') {
        return { ...exp, status: 'validating' as const }
      }
      return exp
    }))
  }

  const handleExport = (exportId: string) => {
    setExports(exports.map(exp => {
      if (exp.id === exportId && exp.status === 'ready') {
        return {
          ...exp,
          status: 'exported' as const,
          exportedDate: new Date().toISOString().split('T')[0],
          repository: newExport.repository,
          doi: `10.5281/zenodo.${Math.floor(Math.random() * 9000000 + 1000000)}`
        }
      }
      return exp
    }))
  }

  const handleDelete = (exportId: string) => {
    if (confirm('Are you sure you want to delete this export?')) {
      setExports(exports.filter(exp => exp.id !== exportId))
    }
  }

  const handleViewDetails = (fairExport: FAIRExport) => {
    setSelectedExport(fairExport)
    setShowDetailsDialog(true)
  }

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

  const stats = {
    total: exports.length,
    exported: exports.filter(e => e.status === 'exported').length,
    ready: exports.filter(e => e.status === 'ready').length,
    avgScore: Math.round(exports.reduce((sum, e) => sum + e.fairScore, 0) / exports.length)
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
        <Button
          onClick={() => setShowNewExport(true)}
          className="flex items-center gap-2"
        >
          <Plus className="w-5 h-5" />
          New Export
        </Button>
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
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Exports</p>
              <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
            </div>
            <Database className="w-8 h-8 text-indigo-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Exported</p>
              <p className="text-2xl font-bold text-purple-600">{stats.exported}</p>
            </div>
            <CheckCircle className="w-8 h-8 text-purple-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Ready</p>
              <p className="text-2xl font-bold text-green-600">{stats.ready}</p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Avg FAIR Score</p>
              <p className="text-2xl font-bold text-indigo-600">{stats.avgScore}</p>
            </div>
            <Shield className="w-8 h-8 text-indigo-500" />
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
              placeholder="Search by dataset name, ID, or creator..."
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
              <SelectItem value="validating">Validating</SelectItem>
              <SelectItem value="ready">Ready</SelectItem>
              <SelectItem value="exported">Exported</SelectItem>
              <SelectItem value="failed">Failed</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </Card>

      {/* Exports List */}
      <div className="space-y-4">
        {filteredExports.map((exp) => {
          const StatusIcon = statusConfig[exp.status].icon
          return (
            <Card key={exp.id} className="p-6 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2 flex-wrap">
                    <h3 className="text-lg font-semibold text-gray-900">{exp.datasetName}</h3>
                    <Badge className={schemaColors[exp.schema]}>
                      {exp.schema}
                    </Badge>
                  </div>
                  <p className="text-sm text-gray-500 mb-3">
                    Export ID: {exp.id} • Dataset: {exp.datasetId}
                  </p>
                </div>
                <Badge className={statusConfig[exp.status].color}>
                  <StatusIcon className="w-3 h-3 mr-1" />
                  {statusConfig[exp.status].label}
                </Badge>
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
                  <Badge className={`${getFairColor(exp.fairScore)} bg-white`}>
                    {getFairLabel(exp.fairScore)}
                  </Badge>
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
                    <Button size="sm" className="bg-green-600 hover:bg-green-700">
                      <Download className="w-4 h-4 mr-1" />
                      Download
                    </Button>
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
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleViewDetails(exp)}
                  className="text-indigo-600 hover:text-indigo-900"
                >
                  View Metadata
                </Button>
                {exp.status === 'ready' && (
                  <>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleExport(exp.id)}
                      className="text-green-600 hover:text-green-900"
                    >
                      Export to Repository
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-blue-600 hover:text-blue-900"
                    >
                      Download Package
                    </Button>
                  </>
                )}
                {exp.status === 'failed' && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleRevalidate(exp.id)}
                    className="text-orange-600 hover:text-orange-900"
                  >
                    Re-validate
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-purple-600 hover:text-purple-900"
                >
                  Validation Report
                </Button>
                {exp.status !== 'exported' && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDelete(exp.id)}
                    className="text-red-600 hover:text-red-900"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                )}
              </div>
            </Card>
          )
        })}

        {filteredExports.length === 0 && (
          <Card className="p-12 text-center">
            <Database className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No FAIR exports found matching your criteria</p>
          </Card>
        )}
      </div>

      {/* New Export Dialog */}
      <Dialog open={showNewExport} onOpenChange={setShowNewExport}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Create FAIR Data Export</DialogTitle>
            <DialogDescription>
              Configure a new FAIR-compliant data export
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Dataset Name *
              </label>
              <Input
                value={newExport.datasetName}
                onChange={(e) => setNewExport({ ...newExport, datasetName: e.target.value })}
                placeholder="Enter descriptive dataset name"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Select Data Source *
              </label>
              <Select
                value={newExport.dataSource}
                onValueChange={(value) => setNewExport({ ...newExport, dataSource: value })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select data source" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="samples">Sample Tracking Data</SelectItem>
                  <SelectItem value="notebooks">Lab Notebook Entries</SelectItem>
                  <SelectItem value="experiments">Experimental Results</SelectItem>
                  <SelectItem value="qc">Quality Control Records</SelectItem>
                  <SelectItem value="equipment">Equipment Logs</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Metadata Schema *
              </label>
              <Select
                value={newExport.schema}
                onValueChange={(value: any) => setNewExport({ ...newExport, schema: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="RO-Crate">RO-Crate (Research Object)</SelectItem>
                  <SelectItem value="JSON-LD">JSON-LD (Linked Data)</SelectItem>
                  <SelectItem value="DataCite">DataCite</SelectItem>
                  <SelectItem value="DCAT">DCAT (Data Catalog)</SelectItem>
                  <SelectItem value="Schema.org">Schema.org</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Target Repository *
              </label>
              <Select
                value={newExport.repository}
                onValueChange={(value) => setNewExport({ ...newExport, repository: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Zenodo">Zenodo</SelectItem>
                  <SelectItem value="Figshare">Figshare</SelectItem>
                  <SelectItem value="Dryad">Dryad</SelectItem>
                  <SelectItem value="OSF">OSF (Open Science Framework)</SelectItem>
                  <SelectItem value="Institutional">Institutional Repository</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowNewExport(false)}
            >
              Cancel
            </Button>
            <Button onClick={handleCreateExport}>
              Create Export
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Details Dialog */}
      <Dialog open={showDetailsDialog} onOpenChange={setShowDetailsDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>FAIR Export Metadata</DialogTitle>
            <DialogDescription>
              {selectedExport?.id}
            </DialogDescription>
          </DialogHeader>

          {selectedExport && (
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-bold text-gray-900 mb-2">{selectedExport.datasetName}</h3>
                <div className="flex flex-wrap gap-2 mb-4">
                  <Badge className={schemaColors[selectedExport.schema]}>
                    {selectedExport.schema}
                  </Badge>
                  <Badge className={statusConfig[selectedExport.status].color}>
                    {statusConfig[selectedExport.status].label}
                  </Badge>
                  <Badge className={getFairColor(selectedExport.fairScore)}>
                    FAIR Score: {selectedExport.fairScore}
                  </Badge>
                </div>
              </div>

              <div className="border-t border-gray-200 pt-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">Dataset ID</p>
                    <p className="text-base text-gray-900">{selectedExport.datasetId}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Created By</p>
                    <p className="text-base text-gray-900">{selectedExport.createdBy}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Created Date</p>
                    <p className="text-base text-gray-900">{selectedExport.createdDate}</p>
                  </div>
                  {selectedExport.exportedDate && (
                    <div>
                      <p className="text-sm font-medium text-gray-500">Exported Date</p>
                      <p className="text-base text-gray-900">{selectedExport.exportedDate}</p>
                    </div>
                  )}
                  {selectedExport.size && (
                    <div>
                      <p className="text-sm font-medium text-gray-500">Package Size</p>
                      <p className="text-base text-gray-900">{selectedExport.size}</p>
                    </div>
                  )}
                  {selectedExport.repository && (
                    <div>
                      <p className="text-sm font-medium text-gray-500">Repository</p>
                      <p className="text-base text-gray-900">{selectedExport.repository}</p>
                    </div>
                  )}
                </div>
              </div>

              {selectedExport.doi && (
                <div className="border-t border-gray-200 pt-4">
                  <p className="text-sm font-medium text-gray-500 mb-2">Digital Object Identifier</p>
                  <div className="bg-gray-50 rounded-lg p-3">
                    <a href={`https://doi.org/${selectedExport.doi}`} className="text-sm text-indigo-600 hover:text-indigo-900 underline" target="_blank" rel="noopener noreferrer">
                      {selectedExport.doi}
                    </a>
                  </div>
                </div>
              )}
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
