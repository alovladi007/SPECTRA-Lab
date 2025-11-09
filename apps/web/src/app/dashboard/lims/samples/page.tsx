'use client'

import { useState, useMemo, useEffect } from 'react'
import {
  ClipboardCheck, Plus, Search, Filter, Download, Eye, Edit2, Trash2,
  QrCode, MapPin, Calendar, User, Tag, AlertCircle, CheckCircle2, Package
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'

interface Sample {
  id: string
  barcode: string
  name: string
  type: string
  location: string
  status: 'received' | 'in_progress' | 'completed' | 'archived'
  owner: string
  received: string
  temperature?: string
  notes?: string
  experiments: number
}

// Mock data generator
const generateMockSamples = (): Sample[] => {
  const types = ['Wafer', 'Film', 'Device', 'Material', 'Substrate', 'Powder']
  const locations = ['Cleanroom-A1', 'Cleanroom-B2', 'Storage-A3', 'Testing-Lab1', 'Archive-B1', 'Receiving']
  const statuses: Sample['status'][] = ['received', 'in_progress', 'completed', 'archived']
  const owners = ['Dr. Chen', 'Dr. Martinez', 'Dr. Patel', 'Dr. Kim', 'Lab Manager']
  const temps = ['RT', '-20°C', '4°C', '-80°C']

  return Array.from({ length: 25 }, (_, i) => {
    const type = types[Math.floor(Math.random() * types.length)]
    const status = statuses[Math.floor(Math.random() * statuses.length)]
    const date = new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000)

    return {
      id: `SMP-${String(i + 1).padStart(3, '0')}`,
      barcode: `${type.substring(0, 3).toUpperCase()}-2024-${String(i + 1).padStart(3, '0')}-${String.fromCharCode(65 + Math.floor(Math.random() * 26))}${Math.floor(Math.random() * 9) + 1}`,
      name: `${type} Sample ${i + 1}`,
      type,
      location: locations[Math.floor(Math.random() * locations.length)],
      status,
      owner: owners[Math.floor(Math.random() * owners.length)],
      received: date.toISOString().split('T')[0],
      temperature: Math.random() > 0.3 ? temps[Math.floor(Math.random() * temps.length)] : undefined,
      notes: Math.random() > 0.5 ? 'Sample prepared for characterization' : '',
      experiments: Math.floor(Math.random() * 8)
    }
  })
}

export default function SamplesPage() {
  const [samples, setSamples] = useState<Sample[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [filterType, setFilterType] = useState<string>('all')
  const [filterStatus, setFilterStatus] = useState<string>('all')
  const [selectedSample, setSelectedSample] = useState<Sample | null>(null)
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false)
  const [isDetailsDialogOpen, setIsDetailsDialogOpen] = useState(false)

  // New sample form state
  const [newSample, setNewSample] = useState({
    name: '',
    type: 'Wafer',
    location: '',
    owner: '',
    temperature: 'RT',
    notes: ''
  })

  // Generate mock data on client side only
  useEffect(() => {
    setSamples(generateMockSamples())
  }, [])

  // Filtered samples
  const filteredSamples = useMemo(() => {
    return samples.filter(sample => {
      const matchesSearch =
        sample.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        sample.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        sample.barcode.toLowerCase().includes(searchQuery.toLowerCase()) ||
        sample.type.toLowerCase().includes(searchQuery.toLowerCase())

      const matchesType = filterType === 'all' || sample.type === filterType
      const matchesStatus = filterStatus === 'all' || sample.status === filterStatus

      return matchesSearch && matchesType && matchesStatus
    })
  }, [samples, searchQuery, filterType, filterStatus])

  // Statistics
  const stats = useMemo(() => ({
    total: samples.length,
    received: samples.filter(s => s.status === 'received').length,
    inProgress: samples.filter(s => s.status === 'in_progress').length,
    completed: samples.filter(s => s.status === 'completed').length
  }), [samples])

  const handleCreateSample = () => {
    const id = `SMP-${String(samples.length + 1).padStart(3, '0')}`
    const now = new Date().toISOString().split('T')[0]

    const sample: Sample = {
      id,
      barcode: `${newSample.type.substring(0, 3).toUpperCase()}-2024-${String(samples.length + 1).padStart(3, '0')}-A1`,
      name: newSample.name,
      type: newSample.type,
      location: newSample.location,
      status: 'received',
      owner: newSample.owner,
      received: now,
      temperature: newSample.temperature,
      notes: newSample.notes,
      experiments: 0
    }

    setSamples(prev => [sample, ...prev])
    setIsCreateDialogOpen(false)

    // Reset form
    setNewSample({
      name: '',
      type: 'Wafer',
      location: '',
      owner: '',
      temperature: 'RT',
      notes: ''
    })
  }

  const handleDeleteSample = (id: string) => {
    if (confirm('Are you sure you want to delete this sample?')) {
      setSamples(prev => prev.filter(s => s.id !== id))
      setIsDetailsDialogOpen(false)
    }
  }

  const getStatusColor = (status: Sample['status']) => {
    switch (status) {
      case 'received': return 'bg-blue-100 text-blue-800'
      case 'in_progress': return 'bg-yellow-100 text-yellow-800'
      case 'completed': return 'bg-green-100 text-green-800'
      case 'archived': return 'bg-gray-100 text-gray-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-teal-500 to-cyan-500 rounded-xl flex items-center justify-center">
            <ClipboardCheck className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Sample Tracking</h1>
            <p className="text-gray-600 mt-1">Manage sample lifecycle with barcode/QR tracking</p>
          </div>
        </div>

        <Button
          className="flex items-center gap-2"
          onClick={() => setIsCreateDialogOpen(true)}
        >
          <Plus className="w-4 h-4" />
          New Sample
        </Button>

        <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
          <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Register New Sample</DialogTitle>
              <DialogDescription>
                Enter the details for the new sample
              </DialogDescription>
            </DialogHeader>

            <div className="grid grid-cols-2 gap-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="name">Sample Name *</Label>
                <Input
                  id="name"
                  placeholder="e.g., Silicon Wafer Batch A"
                  value={newSample.name}
                  onChange={(e) => setNewSample({ ...newSample, name: e.target.value })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="type">Sample Type *</Label>
                <Select
                  value={newSample.type}
                  onValueChange={(value) => setNewSample({ ...newSample, type: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Wafer">Wafer</SelectItem>
                    <SelectItem value="Film">Film</SelectItem>
                    <SelectItem value="Device">Device</SelectItem>
                    <SelectItem value="Material">Material</SelectItem>
                    <SelectItem value="Substrate">Substrate</SelectItem>
                    <SelectItem value="Powder">Powder</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="location">Location *</Label>
                <Input
                  id="location"
                  placeholder="e.g., Cleanroom-A1"
                  value={newSample.location}
                  onChange={(e) => setNewSample({ ...newSample, location: e.target.value })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="owner">Owner *</Label>
                <Input
                  id="owner"
                  placeholder="e.g., Dr. Smith"
                  value={newSample.owner}
                  onChange={(e) => setNewSample({ ...newSample, owner: e.target.value })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="temperature">Storage Temperature</Label>
                <Select
                  value={newSample.temperature}
                  onValueChange={(value) => setNewSample({ ...newSample, temperature: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="RT">Room Temperature (RT)</SelectItem>
                    <SelectItem value="4°C">4°C</SelectItem>
                    <SelectItem value="-20°C">-20°C</SelectItem>
                    <SelectItem value="-80°C">-80°C</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="col-span-2 space-y-2">
                <Label htmlFor="notes">Notes</Label>
                <Textarea
                  id="notes"
                  placeholder="Additional information about the sample..."
                  value={newSample.notes}
                  onChange={(e) => setNewSample({ ...newSample, notes: e.target.value })}
                  rows={3}
                />
              </div>
            </div>

            <DialogFooter>
              <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
                Cancel
              </Button>
              <Button
                onClick={handleCreateSample}
                disabled={!newSample.name || !newSample.location || !newSample.owner}
              >
                Register Sample
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total Samples</p>
                <p className="text-2xl font-bold">{stats.total}</p>
              </div>
              <Package className="w-8 h-8 text-teal-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Received</p>
                <p className="text-2xl font-bold text-blue-600">{stats.received}</p>
              </div>
              <CheckCircle2 className="w-8 h-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">In Progress</p>
                <p className="text-2xl font-bold text-yellow-600">{stats.inProgress}</p>
              </div>
              <AlertCircle className="w-8 h-8 text-yellow-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Completed</p>
                <p className="text-2xl font-bold text-green-600">{stats.completed}</p>
              </div>
              <CheckCircle2 className="w-8 h-8 text-green-500" />
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
                placeholder="Search by ID, name, barcode, or type..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>

            <Select value={filterType} onValueChange={setFilterType}>
              <SelectTrigger className="w-full md:w-[180px]">
                <div className="flex items-center gap-2">
                  <Filter className="w-4 h-4" />
                  <SelectValue placeholder="Type" />
                </div>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="Wafer">Wafer</SelectItem>
                <SelectItem value="Film">Film</SelectItem>
                <SelectItem value="Device">Device</SelectItem>
                <SelectItem value="Material">Material</SelectItem>
                <SelectItem value="Substrate">Substrate</SelectItem>
                <SelectItem value="Powder">Powder</SelectItem>
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
                <SelectItem value="received">Received</SelectItem>
                <SelectItem value="in_progress">In Progress</SelectItem>
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="archived">Archived</SelectItem>
              </SelectContent>
            </Select>

            <Button variant="outline" className="w-full md:w-auto">
              <Download className="w-4 h-4 mr-2" />
              Export
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Samples Table */}
      <Card>
        <CardHeader>
          <CardTitle>Samples ({filteredSamples.length})</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="border-b">
                <tr className="text-left">
                  <th className="pb-3 font-medium text-muted-foreground">ID / Barcode</th>
                  <th className="pb-3 font-medium text-muted-foreground">Name & Type</th>
                  <th className="pb-3 font-medium text-muted-foreground">Location</th>
                  <th className="pb-3 font-medium text-muted-foreground">Status</th>
                  <th className="pb-3 font-medium text-muted-foreground">Owner</th>
                  <th className="pb-3 font-medium text-muted-foreground">Received</th>
                  <th className="pb-3 font-medium text-muted-foreground">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredSamples.map((sample) => (
                  <tr key={sample.id} className="border-b hover:bg-muted/50 transition-colors">
                    <td className="py-3">
                      <div className="flex items-center gap-2">
                        <QrCode className="w-4 h-4 text-muted-foreground" />
                        <div>
                          <div className="font-mono text-sm font-medium">{sample.id}</div>
                          <div className="font-mono text-xs text-muted-foreground">{sample.barcode}</div>
                        </div>
                      </div>
                    </td>
                    <td className="py-3">
                      <div className="font-medium">{sample.name}</div>
                      <div className="text-sm text-muted-foreground">{sample.type}</div>
                    </td>
                    <td className="py-3">
                      <div className="flex items-center gap-1 text-sm">
                        <MapPin className="w-3 h-3 text-muted-foreground" />
                        {sample.location}
                      </div>
                      {sample.temperature && (
                        <div className="text-xs text-muted-foreground mt-1">Temp: {sample.temperature}</div>
                      )}
                    </td>
                    <td className="py-3">
                      <Badge className={getStatusColor(sample.status)}>
                        {sample.status.replace('_', ' ')}
                      </Badge>
                    </td>
                    <td className="py-3">
                      <div className="flex items-center gap-1 text-sm text-muted-foreground">
                        <User className="w-3 h-3" />
                        {sample.owner}
                      </div>
                    </td>
                    <td className="py-3">
                      <div className="flex items-center gap-1 text-sm text-muted-foreground">
                        <Calendar className="w-3 h-3" />
                        {sample.received}
                      </div>
                    </td>
                    <td className="py-3">
                      <div className="flex items-center gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => {
                            setSelectedSample(sample)
                            setIsDetailsDialogOpen(true)
                          }}
                        >
                          <Eye className="w-4 h-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDeleteSample(sample.id)}
                        >
                          <Trash2 className="w-4 h-4 text-red-500" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {filteredSamples.length === 0 && (
              <div className="text-center py-12">
                <Package className="w-16 h-16 text-muted-foreground mx-auto mb-4 opacity-20" />
                <p className="text-lg font-medium text-muted-foreground mb-2">No samples found</p>
                <p className="text-sm text-muted-foreground">
                  {searchQuery || filterType !== 'all' || filterStatus !== 'all'
                    ? 'Try adjusting your search or filters'
                    : 'Register your first sample to get started'}
                </p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Sample Details Dialog */}
      <Dialog open={isDetailsDialogOpen} onOpenChange={setIsDetailsDialogOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Sample Details</DialogTitle>
            <DialogDescription>
              Complete information for {selectedSample?.id}
            </DialogDescription>
          </DialogHeader>

          {selectedSample && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="text-muted-foreground">Sample ID</Label>
                  <p className="font-mono font-medium">{selectedSample.id}</p>
                </div>

                <div className="space-y-2">
                  <Label className="text-muted-foreground">Barcode</Label>
                  <p className="font-mono font-medium">{selectedSample.barcode}</p>
                </div>

                <div className="space-y-2">
                  <Label className="text-muted-foreground">Name</Label>
                  <p className="font-medium">{selectedSample.name}</p>
                </div>

                <div className="space-y-2">
                  <Label className="text-muted-foreground">Type</Label>
                  <Badge variant="outline">{selectedSample.type}</Badge>
                </div>

                <div className="space-y-2">
                  <Label className="text-muted-foreground">Location</Label>
                  <div className="flex items-center gap-2">
                    <MapPin className="w-4 h-4 text-muted-foreground" />
                    <p className="font-medium">{selectedSample.location}</p>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label className="text-muted-foreground">Status</Label>
                  <Badge className={getStatusColor(selectedSample.status)}>
                    {selectedSample.status.replace('_', ' ')}
                  </Badge>
                </div>

                <div className="space-y-2">
                  <Label className="text-muted-foreground">Owner</Label>
                  <div className="flex items-center gap-2">
                    <User className="w-4 h-4 text-muted-foreground" />
                    <p className="font-medium">{selectedSample.owner}</p>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label className="text-muted-foreground">Received</Label>
                  <div className="flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-muted-foreground" />
                    <p className="font-medium">{selectedSample.received}</p>
                  </div>
                </div>

                {selectedSample.temperature && (
                  <div className="space-y-2">
                    <Label className="text-muted-foreground">Storage Temperature</Label>
                    <p className="font-medium">{selectedSample.temperature}</p>
                  </div>
                )}

                <div className="space-y-2">
                  <Label className="text-muted-foreground">Experiments</Label>
                  <p className="font-medium">{selectedSample.experiments} linked experiments</p>
                </div>
              </div>

              {selectedSample.notes && (
                <div className="space-y-2">
                  <Label className="text-muted-foreground">Notes</Label>
                  <div className="p-3 bg-muted rounded-lg">
                    <p className="text-sm">{selectedSample.notes}</p>
                  </div>
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
