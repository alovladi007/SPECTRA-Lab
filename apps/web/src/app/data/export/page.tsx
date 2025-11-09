'use client'

import { useState } from 'react'
import {
  Download, FileText, Calendar, CheckCircle2, Clock, AlertCircle,
  Settings, Database, FileJson, Table, FileSpreadsheet, Package
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Alert, AlertDescription } from '@/components/ui/alert'

interface ExportJob {
  id: string
  name: string
  format: string
  dataType: string
  status: 'Completed' | 'In Progress' | 'Failed' | 'Queued'
  created: string
  fileSize?: string
  recordCount?: number
}

// Mock export history
const generateExportHistory = (): ExportJob[] => {
  const formats = ['CSV', 'JSON', 'Excel', 'HDF5', 'FAIR']
  const dataTypes = ['Samples', 'Experiments', 'Results', 'All Data']
  const statuses: ExportJob['status'][] = ['Completed', 'Completed', 'In Progress', 'Failed']

  return Array.from({ length: 12 }, (_, i) => {
    const status = statuses[Math.floor(Math.random() * statuses.length)]
    return {
      id: `EXP-${String(i + 1000).padStart(4, '0')}`,
      name: `Export ${i + 1}`,
      format: formats[Math.floor(Math.random() * formats.length)],
      dataType: dataTypes[Math.floor(Math.random() * dataTypes.length)],
      status,
      created: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
      fileSize: status === 'Completed' ? `${(Math.random() * 500 + 10).toFixed(1)} MB` : undefined,
      recordCount: status === 'Completed' ? Math.floor(Math.random() * 10000 + 100) : undefined
    }
  })
}

export default function DataExportPage() {
  const [exportHistory] = useState<ExportJob[]>(generateExportHistory())
  const [selectedFormat, setSelectedFormat] = useState('csv')
  const [selectedDataType, setSelectedDataType] = useState('results')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [includeMetadata, setIncludeMetadata] = useState(true)
  const [includeParameters, setIncludeParameters] = useState(true)
  const [includeStatistics, setIncludeStatistics] = useState(false)

  const [isExporting, setIsExporting] = useState(false)

  const handleExport = async () => {
    setIsExporting(true)
    // Simulate export
    await new Promise(resolve => setTimeout(resolve, 2000))
    setIsExporting(false)
    alert('Export started! You will be notified when it completes.')
  }

  const getStatusColor = (status: ExportJob['status']) => {
    switch (status) {
      case 'Completed': return 'bg-green-100 text-green-800'
      case 'In Progress': return 'bg-blue-100 text-blue-800'
      case 'Failed': return 'bg-red-100 text-red-800'
      case 'Queued': return 'bg-yellow-100 text-yellow-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getStatusIcon = (status: ExportJob['status']) => {
    switch (status) {
      case 'Completed': return <CheckCircle2 className="w-4 h-4" />
      case 'In Progress': return <Clock className="w-4 h-4" />
      case 'Failed': return <AlertCircle className="w-4 h-4" />
      case 'Queued': return <Clock className="w-4 h-4" />
      default: return null
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
          <Download className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Data Export</h1>
          <p className="text-gray-600 mt-1">Export data in various formats for analysis and archival</p>
        </div>
      </div>

      <Tabs defaultValue="export" className="w-full">
        <TabsList>
          <TabsTrigger value="export">New Export</TabsTrigger>
          <TabsTrigger value="history">Export History</TabsTrigger>
          <TabsTrigger value="templates">Templates</TabsTrigger>
          <TabsTrigger value="scheduled">Scheduled Exports</TabsTrigger>
        </TabsList>

        {/* New Export Tab */}
        <TabsContent value="export" className="space-y-6 mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Export Configuration */}
            <div className="lg:col-span-2 space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Export Configuration</CardTitle>
                  <CardDescription>Configure your data export settings</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Data Type Selection */}
                  <div className="space-y-2">
                    <Label>Data Type</Label>
                    <Select value={selectedDataType} onValueChange={setSelectedDataType}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="samples">Samples</SelectItem>
                        <SelectItem value="experiments">Experiments</SelectItem>
                        <SelectItem value="results">Results</SelectItem>
                        <SelectItem value="all">All Data</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Date Range */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>From Date</Label>
                      <Input
                        type="date"
                        value={dateFrom}
                        onChange={(e) => setDateFrom(e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>To Date</Label>
                      <Input
                        type="date"
                        value={dateTo}
                        onChange={(e) => setDateTo(e.target.value)}
                      />
                    </div>
                  </div>

                  {/* Format Selection */}
                  <div className="space-y-2">
                    <Label>Export Format</Label>
                    <div className="grid grid-cols-2 gap-4">
                      <Card
                        className={`cursor-pointer transition-all ${selectedFormat === 'csv' ? 'border-blue-500 bg-blue-50' : 'hover:border-gray-400'}`}
                        onClick={() => setSelectedFormat('csv')}
                      >
                        <CardContent className="pt-6">
                          <div className="flex items-center gap-3">
                            <Table className="w-8 h-8 text-blue-500" />
                            <div>
                              <p className="font-semibold">CSV</p>
                              <p className="text-xs text-muted-foreground">Comma-separated values</p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      <Card
                        className={`cursor-pointer transition-all ${selectedFormat === 'json' ? 'border-blue-500 bg-blue-50' : 'hover:border-gray-400'}`}
                        onClick={() => setSelectedFormat('json')}
                      >
                        <CardContent className="pt-6">
                          <div className="flex items-center gap-3">
                            <FileJson className="w-8 h-8 text-green-500" />
                            <div>
                              <p className="font-semibold">JSON</p>
                              <p className="text-xs text-muted-foreground">JavaScript Object Notation</p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      <Card
                        className={`cursor-pointer transition-all ${selectedFormat === 'excel' ? 'border-blue-500 bg-blue-50' : 'hover:border-gray-400'}`}
                        onClick={() => setSelectedFormat('excel')}
                      >
                        <CardContent className="pt-6">
                          <div className="flex items-center gap-3">
                            <FileSpreadsheet className="w-8 h-8 text-green-600" />
                            <div>
                              <p className="font-semibold">Excel</p>
                              <p className="text-xs text-muted-foreground">Microsoft Excel format</p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      <Card
                        className={`cursor-pointer transition-all ${selectedFormat === 'hdf5' ? 'border-blue-500 bg-blue-50' : 'hover:border-gray-400'}`}
                        onClick={() => setSelectedFormat('hdf5')}
                      >
                        <CardContent className="pt-6">
                          <div className="flex items-center gap-3">
                            <Database className="w-8 h-8 text-purple-500" />
                            <div>
                              <p className="font-semibold">HDF5</p>
                              <p className="text-xs text-muted-foreground">Hierarchical Data Format</p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      <Card
                        className={`cursor-pointer transition-all col-span-2 ${selectedFormat === 'fair' ? 'border-blue-500 bg-blue-50' : 'hover:border-gray-400'}`}
                        onClick={() => setSelectedFormat('fair')}
                      >
                        <CardContent className="pt-6">
                          <div className="flex items-center gap-3">
                            <Package className="w-8 h-8 text-orange-500" />
                            <div>
                              <p className="font-semibold">FAIR Data Package</p>
                              <p className="text-xs text-muted-foreground">
                                Findable, Accessible, Interoperable, Reusable format with full metadata
                              </p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  </div>

                  {/* Export Options */}
                  <div className="space-y-3">
                    <Label>Export Options</Label>
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="metadata"
                          checked={includeMetadata}
                          onCheckedChange={(checked) => setIncludeMetadata(checked as boolean)}
                        />
                        <label
                          htmlFor="metadata"
                          className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                        >
                          Include metadata
                        </label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="parameters"
                          checked={includeParameters}
                          onCheckedChange={(checked) => setIncludeParameters(checked as boolean)}
                        />
                        <label
                          htmlFor="parameters"
                          className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                        >
                          Include experimental parameters
                        </label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="statistics"
                          checked={includeStatistics}
                          onCheckedChange={(checked) => setIncludeStatistics(checked as boolean)}
                        />
                        <label
                          htmlFor="statistics"
                          className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                        >
                          Include statistical summaries
                        </label>
                      </div>
                    </div>
                  </div>

                  {/* Additional Settings */}
                  {selectedFormat === 'fair' && (
                    <div className="space-y-2">
                      <Label>Data Package Description</Label>
                      <Textarea
                        placeholder="Describe the contents and purpose of this data export..."
                        rows={3}
                      />
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Export Preview */}
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Export Preview</CardTitle>
                  <CardDescription>Summary of your export</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label className="text-muted-foreground">Data Type</Label>
                    <p className="font-medium capitalize">{selectedDataType}</p>
                  </div>

                  <div>
                    <Label className="text-muted-foreground">Format</Label>
                    <p className="font-medium uppercase">{selectedFormat}</p>
                  </div>

                  {(dateFrom || dateTo) && (
                    <div>
                      <Label className="text-muted-foreground">Date Range</Label>
                      <p className="font-medium text-sm">
                        {dateFrom || 'Beginning'} → {dateTo || 'Now'}
                      </p>
                    </div>
                  )}

                  <div>
                    <Label className="text-muted-foreground">Estimated Records</Label>
                    <p className="text-2xl font-bold">~2,450</p>
                  </div>

                  <div>
                    <Label className="text-muted-foreground">Estimated Size</Label>
                    <p className="text-2xl font-bold">~125 MB</p>
                  </div>

                  <Alert>
                    <AlertCircle className="w-4 h-4" />
                    <AlertDescription className="text-sm">
                      Large exports may take several minutes to complete
                    </AlertDescription>
                  </Alert>

                  <Button
                    className="w-full"
                    size="lg"
                    onClick={handleExport}
                    disabled={isExporting}
                  >
                    {isExporting ? (
                      <>
                        <Clock className="w-4 h-4 mr-2 animate-spin" />
                        Exporting...
                      </>
                    ) : (
                      <>
                        <Download className="w-4 h-4 mr-2" />
                        Start Export
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>

              {/* Quick Export Templates */}
              <Card>
                <CardHeader>
                  <CardTitle>Quick Templates</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <Button variant="outline" className="w-full justify-start">
                    <FileText className="w-4 h-4 mr-2" />
                    All Results (CSV)
                  </Button>
                  <Button variant="outline" className="w-full justify-start">
                    <FileJson className="w-4 h-4 mr-2" />
                    Recent Experiments (JSON)
                  </Button>
                  <Button variant="outline" className="w-full justify-start">
                    <Package className="w-4 h-4 mr-2" />
                    Complete Archive (FAIR)
                  </Button>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        {/* Export History Tab */}
        <TabsContent value="history" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Export History</CardTitle>
              <CardDescription>View and download previous exports</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="border-b">
                    <tr className="text-left">
                      <th className="pb-3 font-medium text-muted-foreground">ID</th>
                      <th className="pb-3 font-medium text-muted-foreground">Name</th>
                      <th className="pb-3 font-medium text-muted-foreground">Format</th>
                      <th className="pb-3 font-medium text-muted-foreground">Data Type</th>
                      <th className="pb-3 font-medium text-muted-foreground">Status</th>
                      <th className="pb-3 font-medium text-muted-foreground">Created</th>
                      <th className="pb-3 font-medium text-muted-foreground">Size</th>
                      <th className="pb-3 font-medium text-muted-foreground">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {exportHistory.map((job) => (
                      <tr key={job.id} className="border-b hover:bg-muted/50 transition-colors">
                        <td className="py-3">
                          <span className="font-mono text-sm">{job.id}</span>
                        </td>
                        <td className="py-3 font-medium">{job.name}</td>
                        <td className="py-3">
                          <Badge variant="outline">{job.format}</Badge>
                        </td>
                        <td className="py-3 text-sm">{job.dataType}</td>
                        <td className="py-3">
                          <Badge className={getStatusColor(job.status)}>
                            <div className="flex items-center gap-1">
                              {getStatusIcon(job.status)}
                              {job.status}
                            </div>
                          </Badge>
                        </td>
                        <td className="py-3 text-sm text-muted-foreground">
                          {new Date(job.created).toLocaleDateString()}
                        </td>
                        <td className="py-3 text-sm text-muted-foreground">
                          {job.fileSize || '-'}
                        </td>
                        <td className="py-3">
                          {job.status === 'Completed' && (
                            <Button variant="ghost" size="sm">
                              <Download className="w-4 h-4" />
                            </Button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Templates Tab */}
        <TabsContent value="templates" className="mt-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="cursor-pointer hover:border-blue-500 transition-colors">
              <CardHeader>
                <CardTitle className="text-lg">Complete Results Export</CardTitle>
                <CardDescription>All measurement results with metadata</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Format:</span>
                    <Badge>CSV</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Data Type:</span>
                    <span>Results</span>
                  </div>
                </div>
                <Button className="w-full mt-4" variant="outline">
                  Use Template
                </Button>
              </CardContent>
            </Card>

            <Card className="cursor-pointer hover:border-blue-500 transition-colors">
              <CardHeader>
                <CardTitle className="text-lg">Sample Database</CardTitle>
                <CardDescription>All samples with tracking info</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Format:</span>
                    <Badge>Excel</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Data Type:</span>
                    <span>Samples</span>
                  </div>
                </div>
                <Button className="w-full mt-4" variant="outline">
                  Use Template
                </Button>
              </CardContent>
            </Card>

            <Card className="cursor-pointer hover:border-blue-500 transition-colors">
              <CardHeader>
                <CardTitle className="text-lg">FAIR Data Archive</CardTitle>
                <CardDescription>Complete dataset for publication</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Format:</span>
                    <Badge>FAIR</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Data Type:</span>
                    <span>All Data</span>
                  </div>
                </div>
                <Button className="w-full mt-4" variant="outline">
                  Use Template
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Scheduled Exports Tab */}
        <TabsContent value="scheduled" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Scheduled Exports</CardTitle>
              <CardDescription>Automate regular data exports</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12">
                <Calendar className="w-16 h-16 text-muted-foreground mx-auto mb-4 opacity-20" />
                <p className="text-lg font-medium text-muted-foreground mb-2">No scheduled exports</p>
                <p className="text-sm text-muted-foreground mb-4">
                  Set up automated exports to run daily, weekly, or monthly
                </p>
                <Button>
                  <Settings className="w-4 h-4 mr-2" />
                  Create Schedule
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
