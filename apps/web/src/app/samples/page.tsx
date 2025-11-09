'use client'

import { useState, useMemo, useEffect } from 'react'
import {
  Database, Plus, Search, Filter, Download, Eye, Edit2, Trash2,
  QrCode, FileText, Calendar, MapPin, User, Tag, AlertCircle, CheckCircle2
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { Alert, AlertDescription } from '@/components/ui/alert'

interface Sample {
  id: string
  name: string
  type: 'Wafer' | 'Thin Film' | 'Bulk' | 'Nanostructure' | 'Device' | 'Other'
  material: string
  substrate?: string
  status: 'Active' | 'In Process' | 'Completed' | 'Archived' | 'On Hold'
  location: string
  owner: string
  created: string
  modified: string
  dimensions?: string
  thickness?: string
  notes?: string
  experiments: number
  qrCode: string
}

// Mock data generator
const generateMockSamples = (): Sample[] => {
  const types: Sample['type'][] = ['Wafer', 'Thin Film', 'Bulk', 'Nanostructure', 'Device', 'Other']
  const statuses: Sample['status'][] = ['Active', 'In Process', 'Completed', 'Archived', 'On Hold']
  const materials = ['Silicon', 'GaAs', 'SiC', 'GaN', 'InP', 'Germanium', 'AlN', 'SiO2', 'Si3N4']
  const locations = ['Cleanroom A-1', 'Storage B-2', 'Lab C-3', 'Testing D-4', 'Archive E-5']
  const owners = ['Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown', 'Dr. Davis']

  return Array.from({ length: 45 }, (_, i) => {
    const id = `SMP-${String(i + 1000).padStart(4, '0')}`
    const type = types[Math.floor(Math.random() * types.length)]
    const material = materials[Math.floor(Math.random() * materials.length)]
    const created = new Date(Date.now() - Math.random() * 90 * 24 * 60 * 60 * 1000).toISOString()

    return {
      id,
      name: `${material} ${type} ${i + 1}`,
      type,
      material,
      substrate: type === 'Thin Film' ? 'Si(100)' : undefined,
      status: statuses[Math.floor(Math.random() * statuses.length)],
      location: locations[Math.floor(Math.random() * locations.length)],
      owner: owners[Math.floor(Math.random() * owners.length)],
      created,
      modified: new Date(new Date(created).getTime() + Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
      dimensions: type === 'Wafer' ? '100 mm' : undefined,
      thickness: type === 'Thin Film' ? `${Math.floor(Math.random() * 500 + 50)} nm` : undefined,
      notes: Math.random() > 0.5 ? 'Sample prepared for characterization' : '',
      experiments: Math.floor(Math.random() * 12),
      qrCode: `QR-${id}`
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
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false)

  // New sample form state
  const [newSample, setNewSample] = useState({
    name: '',
    type: 'Wafer' as Sample['type'],
    material: '',
    substrate: '',
    location: '',
    owner: '',
    dimensions: '',
    thickness: '',
    notes: ''
  })

  // Generate mock data on client side only to avoid hydration mismatch
  useEffect(() => {
    setSamples(generateMockSamples())
  }, [])

  // Filtered samples
  const filteredSamples = useMemo(() => {
    return samples.filter(sample => {
      const matchesSearch =
        sample.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        sample.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        sample.material.toLowerCase().includes(searchQuery.toLowerCase()) ||
        sample.owner.toLowerCase().includes(searchQuery.toLowerCase())

      const matchesType = filterType === 'all' || sample.type === filterType
      const matchesStatus = filterStatus === 'all' || sample.status === filterStatus

      return matchesSearch && matchesType && matchesStatus
    })
  }, [samples, searchQuery, filterType, filterStatus])

  // Statistics
  const stats = useMemo(() => ({
    total: samples.length,
    active: samples.filter(s => s.status === 'Active').length,
    inProcess: samples.filter(s => s.status === 'In Process').length,
    completed: samples.filter(s => s.status === 'Completed').length
  }), [samples])

  const handleCreateSample = () => {
    const id = `SMP-${String(samples.length + 1000).padStart(4, '0')}`
    const now = new Date().toISOString()

    const sample: Sample = {
      id,
      name: newSample.name,
      type: newSample.type,
      material: newSample.material,
      substrate: newSample.substrate || undefined,
      status: 'Active',
      location: newSample.location,
      owner: newSample.owner,
      created: now,
      modified: now,
      dimensions: newSample.dimensions || undefined,
      thickness: newSample.thickness || undefined,
      notes: newSample.notes || undefined,
      experiments: 0,
      qrCode: `QR-${id}`
    }

    setSamples(prev => [sample, ...prev])
    setIsCreateDialogOpen(false)

    // Reset form
    setNewSample({
      name: '',
      type: 'Wafer',
      material: '',
      substrate: '',
      location: '',
      owner: '',
      dimensions: '',
      thickness: '',
      notes: ''
    })
  }

  const handleDeleteSample = (id: string) => {
    if (confirm('Are you sure you want to delete this sample?')) {
      setSamples(prev => prev.filter(s => s.id !== id))
      setIsDetailsDialogOpen(false)
    }
  }

  const handleUpdateSample = () => {
    if (selectedSample) {
      setSamples(prev => prev.map(s =>
        s.id === selectedSample.id
          ? { ...selectedSample, modified: new Date().toISOString() }
          : s
      ))
      setIsEditDialogOpen(false)
      setIsDetailsDialogOpen(false)
    }
  }

  const getStatusColor = (status: Sample['status']) => {
    switch (status) {
      case 'Active': return 'bg-green-100 text-green-800'
      case 'In Process': return 'bg-blue-100 text-blue-800'
      case 'Completed': return 'bg-purple-100 text-purple-800'
      case 'Archived': return 'bg-gray-100 text-gray-800'
      case 'On Hold': return 'bg-yellow-100 text-yellow-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getStatusIcon = (status: Sample['status']) => {
    switch (status) {
      case 'Active': return <CheckCircle2 className="w-4 h-4" />
      case 'In Process': return <AlertCircle className="w-4 h-4" />
      case 'Completed': return <CheckCircle2 className="w-4 h-4" />
      default: return null
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
            <Database className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Sample Manager</h1>
            <p className="text-gray-600 mt-1">Manage and track all samples</p>
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
              <DialogTitle>Create New Sample</DialogTitle>
              <DialogDescription>
                Enter the details for the new sample
              </DialogDescription>
            </DialogHeader>

            <div className="grid grid-cols-2 gap-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="name">Sample Name *</Label>
                <Input
                  id="name"
                  placeholder="e.g., Silicon Wafer A1"
                  value={newSample.name}
                  onChange={(e) => setNewSample({ ...newSample, name: e.target.value })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="type">Sample Type *</Label>
                <Select
                  value={newSample.type}
                  onValueChange={(value) => setNewSample({ ...newSample, type: value as Sample['type'] })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Wafer">Wafer</SelectItem>
                    <SelectItem value="Thin Film">Thin Film</SelectItem>
                    <SelectItem value="Bulk">Bulk</SelectItem>
                    <SelectItem value="Nanostructure">Nanostructure</SelectItem>
                    <SelectItem value="Device">Device</SelectItem>
                    <SelectItem value="Other">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="material">Material *</Label>
                <Input
                  id="material"
                  placeholder="e.g., Silicon, GaAs"
                  value={newSample.material}
                  onChange={(e) => setNewSample({ ...newSample, material: e.target.value })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="substrate">Substrate</Label>
                <Input
                  id="substrate"
                  placeholder="e.g., Si(100)"
                  value={newSample.substrate}
                  onChange={(e) => setNewSample({ ...newSample, substrate: e.target.value })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="location">Location *</Label>
                <Input
                  id="location"
                  placeholder="e.g., Cleanroom A-1"
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
                <Label htmlFor="dimensions">Dimensions</Label>
                <Input
                  id="dimensions"
                  placeholder="e.g., 100 mm"
                  value={newSample.dimensions}
                  onChange={(e) => setNewSample({ ...newSample, dimensions: e.target.value })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="thickness">Thickness</Label>
                <Input
                  id="thickness"
                  placeholder="e.g., 500 nm"
                  value={newSample.thickness}
                  onChange={(e) => setNewSample({ ...newSample, thickness: e.target.value })}
                />
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
                disabled={!newSample.name || !newSample.material || !newSample.location || !newSample.owner}
              >
                Create Sample
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
              <Database className="w-8 h-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Active</p>
                <p className="text-2xl font-bold text-green-600">{stats.active}</p>
              </div>
              <CheckCircle2 className="w-8 h-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">In Process</p>
                <p className="text-2xl font-bold text-blue-600">{stats.inProcess}</p>
              </div>
              <AlertCircle className="w-8 h-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Completed</p>
                <p className="text-2xl font-bold text-purple-600">{stats.completed}</p>
              </div>
              <CheckCircle2 className="w-8 h-8 text-purple-500" />
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
                placeholder="Search by ID, name, material, or owner..."
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
                <SelectItem value="Thin Film">Thin Film</SelectItem>
                <SelectItem value="Bulk">Bulk</SelectItem>
                <SelectItem value="Nanostructure">Nanostructure</SelectItem>
                <SelectItem value="Device">Device</SelectItem>
                <SelectItem value="Other">Other</SelectItem>
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
                <SelectItem value="Active">Active</SelectItem>
                <SelectItem value="In Process">In Process</SelectItem>
                <SelectItem value="Completed">Completed</SelectItem>
                <SelectItem value="Archived">Archived</SelectItem>
                <SelectItem value="On Hold">On Hold</SelectItem>
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
                  <th className="pb-3 font-medium text-muted-foreground">ID</th>
                  <th className="pb-3 font-medium text-muted-foreground">Name</th>
                  <th className="pb-3 font-medium text-muted-foreground">Type</th>
                  <th className="pb-3 font-medium text-muted-foreground">Material</th>
                  <th className="pb-3 font-medium text-muted-foreground">Status</th>
                  <th className="pb-3 font-medium text-muted-foreground">Location</th>
                  <th className="pb-3 font-medium text-muted-foreground">Owner</th>
                  <th className="pb-3 font-medium text-muted-foreground">Experiments</th>
                  <th className="pb-3 font-medium text-muted-foreground">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredSamples.map((sample) => (
                  <tr key={sample.id} className="border-b hover:bg-muted/50 transition-colors">
                    <td className="py-3">
                      <div className="flex items-center gap-2">
                        <QrCode className="w-4 h-4 text-muted-foreground" />
                        <span className="font-mono text-sm">{sample.id}</span>
                      </div>
                    </td>
                    <td className="py-3 font-medium">{sample.name}</td>
                    <td className="py-3">
                      <Badge variant="outline">{sample.type}</Badge>
                    </td>
                    <td className="py-3">{sample.material}</td>
                    <td className="py-3">
                      <Badge className={getStatusColor(sample.status)}>
                        <div className="flex items-center gap-1">
                          {getStatusIcon(sample.status)}
                          {sample.status}
                        </div>
                      </Badge>
                    </td>
                    <td className="py-3">
                      <div className="flex items-center gap-1 text-sm text-muted-foreground">
                        <MapPin className="w-3 h-3" />
                        {sample.location}
                      </div>
                    </td>
                    <td className="py-3">
                      <div className="flex items-center gap-1 text-sm text-muted-foreground">
                        <User className="w-3 h-3" />
                        {sample.owner}
                      </div>
                    </td>
                    <td className="py-3 text-center">
                      <Badge variant="secondary">{sample.experiments}</Badge>
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
                          onClick={() => {
                            setSelectedSample(sample)
                            setIsEditDialogOpen(true)
                          }}
                        >
                          <Edit2 className="w-4 h-4" />
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
                <Database className="w-16 h-16 text-muted-foreground mx-auto mb-4 opacity-20" />
                <p className="text-lg font-medium text-muted-foreground mb-2">No samples found</p>
                <p className="text-sm text-muted-foreground">
                  {searchQuery || filterType !== 'all' || filterStatus !== 'all'
                    ? 'Try adjusting your search or filters'
                    : 'Create your first sample to get started'}
                </p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Sample Details Dialog */}
      <Dialog open={isDetailsDialogOpen} onOpenChange={setIsDetailsDialogOpen}>
        <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Sample Details</DialogTitle>
            <DialogDescription>
              Complete information for {selectedSample?.id}
            </DialogDescription>
          </DialogHeader>

          {selectedSample && (
            <div className="space-y-6">
              <Alert>
                <QrCode className="w-4 h-4" />
                <AlertDescription>
                  <div className="flex items-center justify-between">
                    <span>QR Code: {selectedSample.qrCode}</span>
                    <Button variant="outline" size="sm">
                      <Download className="w-3 h-3 mr-2" />
                      Download QR
                    </Button>
                  </div>
                </AlertDescription>
              </Alert>

              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <Label className="text-muted-foreground">Sample ID</Label>
                    <p className="font-mono font-medium">{selectedSample.id}</p>
                  </div>

                  <div>
                    <Label className="text-muted-foreground">Name</Label>
                    <p className="font-medium">{selectedSample.name}</p>
                  </div>

                  <div>
                    <Label className="text-muted-foreground">Type</Label>
                    <div className="mt-1">
                      <Badge variant="outline">{selectedSample.type}</Badge>
                    </div>
                  </div>

                  <div>
                    <Label className="text-muted-foreground">Material</Label>
                    <p className="font-medium">{selectedSample.material}</p>
                  </div>

                  {selectedSample.substrate && (
                    <div>
                      <Label className="text-muted-foreground">Substrate</Label>
                      <p className="font-medium">{selectedSample.substrate}</p>
                    </div>
                  )}

                  {selectedSample.dimensions && (
                    <div>
                      <Label className="text-muted-foreground">Dimensions</Label>
                      <p className="font-medium">{selectedSample.dimensions}</p>
                    </div>
                  )}

                  {selectedSample.thickness && (
                    <div>
                      <Label className="text-muted-foreground">Thickness</Label>
                      <p className="font-medium">{selectedSample.thickness}</p>
                    </div>
                  )}
                </div>

                <div className="space-y-4">
                  <div>
                    <Label className="text-muted-foreground">Status</Label>
                    <div className="mt-1">
                      <Badge className={getStatusColor(selectedSample.status)}>
                        <div className="flex items-center gap-1">
                          {getStatusIcon(selectedSample.status)}
                          {selectedSample.status}
                        </div>
                      </Badge>
                    </div>
                  </div>

                  <div>
                    <Label className="text-muted-foreground">Location</Label>
                    <div className="flex items-center gap-2 mt-1">
                      <MapPin className="w-4 h-4 text-muted-foreground" />
                      <p className="font-medium">{selectedSample.location}</p>
                    </div>
                  </div>

                  <div>
                    <Label className="text-muted-foreground">Owner</Label>
                    <div className="flex items-center gap-2 mt-1">
                      <User className="w-4 h-4 text-muted-foreground" />
                      <p className="font-medium">{selectedSample.owner}</p>
                    </div>
                  </div>

                  <div>
                    <Label className="text-muted-foreground">Experiments</Label>
                    <p className="font-medium">{selectedSample.experiments} linked experiments</p>
                  </div>

                  <div>
                    <Label className="text-muted-foreground">Created</Label>
                    <div className="flex items-center gap-2 mt-1">
                      <Calendar className="w-4 h-4 text-muted-foreground" />
                      <p className="font-medium">
                        {new Date(selectedSample.created).toLocaleDateString()}
                      </p>
                    </div>
                  </div>

                  <div>
                    <Label className="text-muted-foreground">Last Modified</Label>
                    <div className="flex items-center gap-2 mt-1">
                      <Calendar className="w-4 h-4 text-muted-foreground" />
                      <p className="font-medium">
                        {new Date(selectedSample.modified).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {selectedSample.notes && (
                <div>
                  <Label className="text-muted-foreground">Notes</Label>
                  <div className="mt-2 p-3 bg-muted rounded-lg">
                    <p className="text-sm">{selectedSample.notes}</p>
                  </div>
                </div>
              )}

              <div className="flex justify-between pt-4 border-t">
                <div className="flex gap-2">
                  <Button variant="outline" size="sm">
                    <FileText className="w-4 h-4 mr-2" />
                    Generate Report
                  </Button>
                  <Button variant="outline" size="sm">
                    <Tag className="w-4 h-4 mr-2" />
                    Print Label
                  </Button>
                </div>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => handleDeleteSample(selectedSample.id)}
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete Sample
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Edit Sample Dialog */}
      <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Edit Sample</DialogTitle>
            <DialogDescription>
              Update sample information
            </DialogDescription>
          </DialogHeader>

          {selectedSample && (
            <div className="grid grid-cols-2 gap-4 py-4">
              <div className="space-y-2">
                <Label>Sample Name</Label>
                <Input
                  value={selectedSample.name}
                  onChange={(e) => setSelectedSample({ ...selectedSample, name: e.target.value })}
                />
              </div>

              <div className="space-y-2">
                <Label>Status</Label>
                <Select
                  value={selectedSample.status}
                  onValueChange={(value) => setSelectedSample({ ...selectedSample, status: value as Sample['status'] })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Active">Active</SelectItem>
                    <SelectItem value="In Process">In Process</SelectItem>
                    <SelectItem value="Completed">Completed</SelectItem>
                    <SelectItem value="Archived">Archived</SelectItem>
                    <SelectItem value="On Hold">On Hold</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Material</Label>
                <Input
                  value={selectedSample.material}
                  onChange={(e) => setSelectedSample({ ...selectedSample, material: e.target.value })}
                />
              </div>

              <div className="space-y-2">
                <Label>Location</Label>
                <Input
                  value={selectedSample.location}
                  onChange={(e) => setSelectedSample({ ...selectedSample, location: e.target.value })}
                />
              </div>

              <div className="space-y-2">
                <Label>Owner</Label>
                <Input
                  value={selectedSample.owner}
                  onChange={(e) => setSelectedSample({ ...selectedSample, owner: e.target.value })}
                />
              </div>

              <div className="space-y-2">
                <Label>Dimensions</Label>
                <Input
                  value={selectedSample.dimensions || ''}
                  onChange={(e) => setSelectedSample({ ...selectedSample, dimensions: e.target.value })}
                />
              </div>

              <div className="col-span-2 space-y-2">
                <Label>Notes</Label>
                <Textarea
                  value={selectedSample.notes || ''}
                  onChange={(e) => setSelectedSample({ ...selectedSample, notes: e.target.value })}
                  rows={3}
                />
              </div>
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsEditDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleUpdateSample}>
              Save Changes
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
