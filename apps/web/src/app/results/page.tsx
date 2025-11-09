'use client'

import { useState, useMemo } from 'react'
import {
  BarChart3, Plus, Search, Filter, Download, Eye, TrendingUp,
  Calendar, FileText, Activity, Zap, Waves, Layers, Beaker, LineChart
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart as RechartsBarChart, Bar } from 'recharts'

interface Result {
  id: string
  experimentId: string
  experimentName: string
  sampleId: string
  technique: string
  type: 'Electrical' | 'Optical' | 'Structural' | 'Chemical' | 'Process Simulation'
  timestamp: string
  operator: string
  parameters: Record<string, any>
  measurements: Record<string, number>
  dataPoints?: number
  fileSize: string
  status: 'Valid' | 'Invalid' | 'Pending Review'
  notes?: string
}

// Mock data generator
const generateMockResults = (): Result[] => {
  const types: Result['type'][] = ['Electrical', 'Optical', 'Structural', 'Chemical', 'Process Simulation']
  const techniques = {
    'Electrical': ['Four-Point Probe', 'Hall Effect', 'I-V Curve', 'C-V Profiling'],
    'Optical': ['UV-Vis', 'FTIR', 'Raman', 'PL Spectroscopy'],
    'Structural': ['XRD', 'SEM', 'TEM', 'AFM'],
    'Chemical': ['XPS', 'SIMS', 'EDX', 'RBS'],
    'Process Simulation': ['Diffusion Profile', 'Oxide Growth', 'Implantation']
  }
  const operators = ['Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown']
  const statuses: Result['status'][] = ['Valid', 'Valid', 'Valid', 'Pending Review', 'Invalid']

  return Array.from({ length: 50 }, (_, i) => {
    const type = types[Math.floor(Math.random() * types.length)]
    const technique = techniques[type][Math.floor(Math.random() * techniques[type].length)]
    const timestamp = new Date(Date.now() - Math.random() * 90 * 24 * 60 * 60 * 1000).toISOString()

    const measurements: Record<string, number> = {}
    if (type === 'Electrical') {
      measurements.resistance = Math.random() * 1000
      measurements.resistivity = Math.random() * 100
      measurements.mobility = Math.random() * 500
    } else if (type === 'Optical') {
      measurements.transmittance = Math.random() * 100
      measurements.absorbance = Math.random() * 3
      measurements.peakWavelength = 400 + Math.random() * 400
    } else if (type === 'Structural') {
      measurements.grainSize = Math.random() * 100
      measurements.roughness = Math.random() * 10
      measurements.thickness = Math.random() * 1000
    } else if (type === 'Chemical') {
      measurements.concentration = Math.random() * 100
      measurements.depth = Math.random() * 500
    }

    return {
      id: `RES-${String(i + 5000).padStart(5, '0')}`,
      experimentId: `EXP-${2000 + Math.floor(Math.random() * 35)}`,
      experimentName: `${technique} Run ${i + 1}`,
      sampleId: `SMP-${1000 + Math.floor(Math.random() * 45)}`,
      technique,
      type,
      timestamp,
      operator: operators[Math.floor(Math.random() * operators.length)],
      parameters: {
        temperature: '300 K',
        pressure: '1e-6 Torr',
        duration: `${Math.floor(Math.random() * 60 + 10)} min`
      },
      measurements,
      dataPoints: Math.floor(Math.random() * 10000 + 100),
      fileSize: `${(Math.random() * 50 + 5).toFixed(1)} MB`,
      status: statuses[Math.floor(Math.random() * statuses.length)],
      notes: Math.random() > 0.7 ? 'Excellent signal-to-noise ratio' : undefined
    }
  })
}

// Generate sample chart data
const generateChartData = (type: string) => {
  if (type === 'Electrical') {
    return Array.from({ length: 20 }, (_, i) => ({
      x: i * 0.1,
      voltage: i * 0.1,
      current: Math.pow(i * 0.1, 2) * 0.5 + Math.random() * 0.1
    }))
  } else if (type === 'Optical') {
    return Array.from({ length: 50 }, (_, i) => ({
      wavelength: 400 + i * 8,
      intensity: Math.sin(i * 0.3) * 50 + 50 + Math.random() * 10
    }))
  } else if (type === 'Structural') {
    return Array.from({ length: 30 }, (_, i) => ({
      angle: 20 + i * 2,
      intensity: Math.exp(-Math.pow(i - 15, 2) / 50) * 1000 + Math.random() * 100
    }))
  }
  return []
}

export default function ResultsPage() {
  const [results, setResults] = useState<Result[]>(generateMockResults())
  const [searchQuery, setSearchQuery] = useState('')
  const [filterType, setFilterType] = useState<string>('all')
  const [filterStatus, setFilterStatus] = useState<string>('all')
  const [selectedResult, setSelectedResult] = useState<Result | null>(null)
  const [isDetailsDialogOpen, setIsDetailsDialogOpen] = useState(false)
  const [chartData, setChartData] = useState<any[]>([])

  // Filtered results
  const filteredResults = useMemo(() => {
    return results.filter(result => {
      const matchesSearch =
        result.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        result.experimentName.toLowerCase().includes(searchQuery.toLowerCase()) ||
        result.sampleId.toLowerCase().includes(searchQuery.toLowerCase()) ||
        result.technique.toLowerCase().includes(searchQuery.toLowerCase()) ||
        result.operator.toLowerCase().includes(searchQuery.toLowerCase())

      const matchesType = filterType === 'all' || result.type === filterType
      const matchesStatus = filterStatus === 'all' || result.status === filterStatus

      return matchesSearch && matchesType && matchesStatus
    })
  }, [results, searchQuery, filterType, filterStatus])

  // Statistics
  const stats = useMemo(() => ({
    total: results.length,
    valid: results.filter(r => r.status === 'Valid').length,
    pending: results.filter(r => r.status === 'Pending Review').length,
    totalSize: results.reduce((acc, r) => acc + parseFloat(r.fileSize), 0).toFixed(1)
  }), [results])

  const handleViewResult = (result: Result) => {
    setSelectedResult(result)
    setChartData(generateChartData(result.type))
    setIsDetailsDialogOpen(true)
  }

  const getTypeIcon = (type: Result['type']) => {
    switch (type) {
      case 'Electrical': return <Zap className="w-4 h-4" />
      case 'Optical': return <Waves className="w-4 h-4" />
      case 'Structural': return <Layers className="w-4 h-4" />
      case 'Chemical': return <Beaker className="w-4 h-4" />
      case 'Process Simulation': return <Activity className="w-4 h-4" />
      default: return <Activity className="w-4 h-4" />
    }
  }

  const getStatusColor = (status: Result['status']) => {
    switch (status) {
      case 'Valid': return 'bg-green-100 text-green-800'
      case 'Pending Review': return 'bg-yellow-100 text-yellow-800'
      case 'Invalid': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
            <BarChart3 className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Results Browser</h1>
            <p className="text-gray-600 mt-1">Browse and analyze measurement results</p>
          </div>
        </div>

        <Button variant="outline" className="flex items-center gap-2">
          <Download className="w-4 h-4" />
          Bulk Export
        </Button>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total Results</p>
                <p className="text-2xl font-bold">{stats.total}</p>
              </div>
              <FileText className="w-8 h-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Valid Results</p>
                <p className="text-2xl font-bold text-green-600">{stats.valid}</p>
              </div>
              <TrendingUp className="w-8 h-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Pending Review</p>
                <p className="text-2xl font-bold text-yellow-600">{stats.pending}</p>
              </div>
              <Activity className="w-8 h-8 text-yellow-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total Data Size</p>
                <p className="text-2xl font-bold">{stats.totalSize} GB</p>
              </div>
              <Download className="w-8 h-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Search and Filters */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search by ID, experiment, sample, technique, or operator..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>

            <Select value={filterType} onValueChange={setFilterType}>
              <SelectTrigger className="w-full md:w-[200px]">
                <div className="flex items-center gap-2">
                  <Filter className="w-4 h-4" />
                  <SelectValue placeholder="Type" />
                </div>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="Electrical">Electrical</SelectItem>
                <SelectItem value="Optical">Optical</SelectItem>
                <SelectItem value="Structural">Structural</SelectItem>
                <SelectItem value="Chemical">Chemical</SelectItem>
                <SelectItem value="Process Simulation">Process Simulation</SelectItem>
              </SelectContent>
            </Select>

            <Select value={filterStatus} onValueChange={setFilterStatus}>
              <SelectTrigger className="w-full md:w-[180px]">
                <div className="flex items-center gap-2">
                  <Filter className="w-4 h-4" />
                  <SelectValue placeholder="Status" />
                </div>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="Valid">Valid</SelectItem>
                <SelectItem value="Pending Review">Pending Review</SelectItem>
                <SelectItem value="Invalid">Invalid</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Results Table */}
      <Card>
        <CardHeader>
          <CardTitle>Results ({filteredResults.length})</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="border-b">
                <tr className="text-left">
                  <th className="pb-3 font-medium text-muted-foreground">ID</th>
                  <th className="pb-3 font-medium text-muted-foreground">Experiment</th>
                  <th className="pb-3 font-medium text-muted-foreground">Sample</th>
                  <th className="pb-3 font-medium text-muted-foreground">Type</th>
                  <th className="pb-3 font-medium text-muted-foreground">Technique</th>
                  <th className="pb-3 font-medium text-muted-foreground">Status</th>
                  <th className="pb-3 font-medium text-muted-foreground">Date</th>
                  <th className="pb-3 font-medium text-muted-foreground">Size</th>
                  <th className="pb-3 font-medium text-muted-foreground">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredResults.map((result) => (
                  <tr key={result.id} className="border-b hover:bg-muted/50 transition-colors">
                    <td className="py-3">
                      <span className="font-mono text-sm">{result.id}</span>
                    </td>
                    <td className="py-3">
                      <div>
                        <p className="font-medium text-sm">{result.experimentName}</p>
                        <p className="text-xs text-muted-foreground">{result.experimentId}</p>
                      </div>
                    </td>
                    <td className="py-3">
                      <Badge variant="outline" className="font-mono text-xs">
                        {result.sampleId}
                      </Badge>
                    </td>
                    <td className="py-3">
                      <Badge variant="outline" className="flex items-center gap-1 w-fit">
                        {getTypeIcon(result.type)}
                        {result.type}
                      </Badge>
                    </td>
                    <td className="py-3 text-sm">{result.technique}</td>
                    <td className="py-3">
                      <Badge className={getStatusColor(result.status)}>
                        {result.status}
                      </Badge>
                    </td>
                    <td className="py-3 text-sm text-muted-foreground">
                      {new Date(result.timestamp).toLocaleDateString()}
                    </td>
                    <td className="py-3 text-sm text-muted-foreground">{result.fileSize}</td>
                    <td className="py-3">
                      <div className="flex items-center gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleViewResult(result)}
                        >
                          <Eye className="w-4 h-4" />
                        </Button>
                        <Button variant="ghost" size="sm">
                          <Download className="w-4 h-4" />
                        </Button>
                        <Button variant="ghost" size="sm">
                          <LineChart className="w-4 h-4" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {filteredResults.length === 0 && (
              <div className="text-center py-12">
                <BarChart3 className="w-16 h-16 text-muted-foreground mx-auto mb-4 opacity-20" />
                <p className="text-lg font-medium text-muted-foreground mb-2">No results found</p>
                <p className="text-sm text-muted-foreground">
                  {searchQuery || filterType !== 'all' || filterStatus !== 'all'
                    ? 'Try adjusting your search or filters'
                    : 'No measurement results available'}
                </p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Result Details Dialog */}
      <Dialog open={isDetailsDialogOpen} onOpenChange={setIsDetailsDialogOpen}>
        <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Result Details</DialogTitle>
            <DialogDescription>
              Detailed information and visualization for {selectedResult?.id}
            </DialogDescription>
          </DialogHeader>

          {selectedResult && (
            <Tabs defaultValue="overview" className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="data">Data</TabsTrigger>
                <TabsTrigger value="visualization">Visualization</TabsTrigger>
                <TabsTrigger value="analysis">Analysis</TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="space-y-6 mt-4">
                {/* Status Badge */}
                <div className="flex items-center gap-4">
                  <Badge className={getStatusColor(selectedResult.status)}>
                    {selectedResult.status}
                  </Badge>
                  <Badge variant="outline" className="flex items-center gap-1">
                    {getTypeIcon(selectedResult.type)}
                    {selectedResult.type}
                  </Badge>
                </div>

                {/* Main Details */}
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div>
                      <Label className="text-muted-foreground">Result ID</Label>
                      <p className="font-mono font-medium">{selectedResult.id}</p>
                    </div>

                    <div>
                      <Label className="text-muted-foreground">Experiment</Label>
                      <p className="font-medium">{selectedResult.experimentName}</p>
                      <p className="text-sm text-muted-foreground">{selectedResult.experimentId}</p>
                    </div>

                    <div>
                      <Label className="text-muted-foreground">Sample ID</Label>
                      <Badge variant="outline" className="font-mono">
                        {selectedResult.sampleId}
                      </Badge>
                    </div>

                    <div>
                      <Label className="text-muted-foreground">Technique</Label>
                      <p className="font-medium">{selectedResult.technique}</p>
                    </div>

                    <div>
                      <Label className="text-muted-foreground">Operator</Label>
                      <p className="font-medium">{selectedResult.operator}</p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <Label className="text-muted-foreground">Timestamp</Label>
                      <div className="flex items-center gap-2 mt-1">
                        <Calendar className="w-4 h-4 text-muted-foreground" />
                        <p className="font-medium">
                          {new Date(selectedResult.timestamp).toLocaleString()}
                        </p>
                      </div>
                    </div>

                    <div>
                      <Label className="text-muted-foreground">Data Points</Label>
                      <p className="font-medium">{selectedResult.dataPoints?.toLocaleString()}</p>
                    </div>

                    <div>
                      <Label className="text-muted-foreground">File Size</Label>
                      <p className="font-medium">{selectedResult.fileSize}</p>
                    </div>

                    <div>
                      <Label className="text-muted-foreground">Parameters</Label>
                      <div className="mt-2 space-y-1">
                        {Object.entries(selectedResult.parameters).map(([key, value]) => (
                          <div key={key} className="flex justify-between text-sm">
                            <span className="text-muted-foreground">{key}:</span>
                            <span className="font-medium">{value}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Notes */}
                {selectedResult.notes && (
                  <div>
                    <Label className="text-muted-foreground">Notes</Label>
                    <div className="mt-2 p-3 bg-muted rounded-lg">
                      <p className="text-sm">{selectedResult.notes}</p>
                    </div>
                  </div>
                )}
              </TabsContent>

              <TabsContent value="data" className="space-y-4 mt-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Measurements Summary</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      {Object.entries(selectedResult.measurements).map(([key, value]) => (
                        <div key={key} className="p-4 border rounded-lg">
                          <Label className="text-muted-foreground text-sm">{key}</Label>
                          <p className="text-2xl font-bold mt-2">
                            {typeof value === 'number' ? value.toFixed(3) : value}
                          </p>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="visualization" className="space-y-4 mt-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Data Visualization</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={400}>
                      {selectedResult.type === 'Electrical' ? (
                        <RechartsLineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="voltage" label={{ value: 'Voltage (V)', position: 'insideBottom', offset: -5 }} />
                          <YAxis label={{ value: 'Current (A)', angle: -90, position: 'insideLeft' }} />
                          <Tooltip />
                          <Legend />
                          <Line type="monotone" dataKey="current" stroke="#3b82f6" strokeWidth={2} dot={false} />
                        </RechartsLineChart>
                      ) : selectedResult.type === 'Optical' ? (
                        <RechartsLineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="wavelength" label={{ value: 'Wavelength (nm)', position: 'insideBottom', offset: -5 }} />
                          <YAxis label={{ value: 'Intensity (a.u.)', angle: -90, position: 'insideLeft' }} />
                          <Tooltip />
                          <Legend />
                          <Line type="monotone" dataKey="intensity" stroke="#8b5cf6" strokeWidth={2} dot={false} />
                        </RechartsLineChart>
                      ) : (
                        <RechartsBarChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="angle" label={{ value: '2θ (degrees)', position: 'insideBottom', offset: -5 }} />
                          <YAxis label={{ value: 'Intensity (counts)', angle: -90, position: 'insideLeft' }} />
                          <Tooltip />
                          <Legend />
                          <Bar dataKey="intensity" fill="#10b981" />
                        </RechartsBarChart>
                      )}
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="analysis" className="space-y-4 mt-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Statistical Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="p-4 border rounded-lg">
                        <Label className="text-muted-foreground text-sm">Mean</Label>
                        <p className="text-xl font-bold mt-2">125.4</p>
                      </div>
                      <div className="p-4 border rounded-lg">
                        <Label className="text-muted-foreground text-sm">Std Dev</Label>
                        <p className="text-xl font-bold mt-2">12.8</p>
                      </div>
                      <div className="p-4 border rounded-lg">
                        <Label className="text-muted-foreground text-sm">R² Value</Label>
                        <p className="text-xl font-bold mt-2">0.998</p>
                      </div>
                    </div>

                    <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                      <p className="text-sm font-medium text-green-800">
                        Data quality: Excellent
                      </p>
                      <p className="text-sm text-green-700 mt-1">
                        Low noise level, good reproducibility, within expected ranges
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          )}

          <div className="flex justify-between pt-4 border-t">
            <div className="flex gap-2">
              <Button variant="outline" size="sm">
                <Download className="w-4 h-4 mr-2" />
                Download Raw Data
              </Button>
              <Button variant="outline" size="sm">
                <FileText className="w-4 h-4 mr-2" />
                Export Report
              </Button>
            </div>
            <Button variant="outline" size="sm" onClick={() => setIsDetailsDialogOpen(false)}>
              Close
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
