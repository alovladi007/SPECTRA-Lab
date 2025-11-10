'use client'

import { useState, useEffect } from 'react'
import {
  Settings, Plus, Search, AlertCircle, CheckCircle2, Clock,
  Wrench, Calendar, Activity, Edit2, Trash2, Eye, BarChart3
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'

interface Instrument {
  id: string
  name: string
  type: string
  manufacturer: string
  model: string
  serialNumber: string
  location: string
  status: 'operational' | 'maintenance' | 'calibration_needed' | 'offline'
  lastCalibration: string
  nextCalibration: string
  lastMaintenance: string
  nextMaintenance: string
  usageHours: number
  technician: string
  notes?: string
}

function generateMockInstruments(): Instrument[] {
  const instruments = [
    {
      id: 'INST-001',
      name: 'Four-Point Probe System',
      type: 'Electrical',
      manufacturer: 'Keithley',
      model: '4200-SCS',
      serialNumber: 'K4200-2024-001',
      location: 'Lab A - Bay 3',
      status: 'operational' as const,
      lastCalibration: '2024-09-15',
      nextCalibration: '2024-12-15',
      lastMaintenance: '2024-10-01',
      nextMaintenance: '2025-01-01',
      usageHours: 1247,
      technician: 'John Smith',
      notes: 'Regular maintenance scheduled'
    },
    {
      id: 'INST-002',
      name: 'Hall Effect Measurement System',
      type: 'Electrical',
      manufacturer: 'Lakeshore',
      model: '8400',
      serialNumber: 'LS8400-2023-045',
      location: 'Lab A - Bay 5',
      status: 'calibration_needed' as const,
      lastCalibration: '2024-05-10',
      nextCalibration: '2024-11-10',
      lastMaintenance: '2024-08-20',
      nextMaintenance: '2024-12-20',
      usageHours: 2134,
      technician: 'Sarah Johnson'
    },
    {
      id: 'INST-003',
      name: 'UV-Vis-NIR Spectrometer',
      type: 'Optical',
      manufacturer: 'Perkin Elmer',
      model: 'Lambda 1050',
      serialNumber: 'PE1050-2024-012',
      location: 'Lab B - Bay 1',
      status: 'operational' as const,
      lastCalibration: '2024-10-01',
      nextCalibration: '2025-01-01',
      lastMaintenance: '2024-09-15',
      nextMaintenance: '2024-12-15',
      usageHours: 876,
      technician: 'Michael Chen'
    },
    {
      id: 'INST-004',
      name: 'FTIR Spectrometer',
      type: 'Optical',
      manufacturer: 'Bruker',
      model: 'Vertex 80',
      serialNumber: 'BR-V80-2023-028',
      location: 'Lab B - Bay 3',
      status: 'maintenance' as const,
      lastCalibration: '2024-07-20',
      nextCalibration: '2024-12-20',
      lastMaintenance: '2024-11-01',
      nextMaintenance: '2025-02-01',
      usageHours: 1563,
      technician: 'Sarah Johnson',
      notes: 'Laser alignment in progress'
    },
    {
      id: 'INST-005',
      name: 'X-Ray Diffractometer',
      type: 'Structural',
      manufacturer: 'Rigaku',
      model: 'SmartLab SE',
      serialNumber: 'RG-SL-2024-007',
      location: 'Lab C - Bay 2',
      status: 'operational' as const,
      lastCalibration: '2024-08-15',
      nextCalibration: '2024-11-15',
      lastMaintenance: '2024-10-10',
      nextMaintenance: '2025-01-10',
      usageHours: 945,
      technician: 'David Wilson'
    },
    {
      id: 'INST-006',
      name: 'SEM with EDX',
      type: 'Structural',
      manufacturer: 'Zeiss',
      model: 'Gemini 500',
      serialNumber: 'ZS-G500-2023-015',
      location: 'Lab C - Bay 4',
      status: 'operational' as const,
      lastCalibration: '2024-09-01',
      nextCalibration: '2024-12-01',
      lastMaintenance: '2024-10-15',
      nextMaintenance: '2025-01-15',
      usageHours: 1821,
      technician: 'Emily Brown'
    },
    {
      id: 'INST-007',
      name: 'XPS System',
      type: 'Chemical',
      manufacturer: 'Thermo Fisher',
      model: 'K-Alpha+',
      serialNumber: 'TF-KA-2024-003',
      location: 'Lab D - Bay 1',
      status: 'operational' as const,
      lastCalibration: '2024-10-05',
      nextCalibration: '2025-01-05',
      lastMaintenance: '2024-09-20',
      nextMaintenance: '2024-12-20',
      usageHours: 1092,
      technician: 'Michael Chen'
    },
    {
      id: 'INST-008',
      name: 'Ellipsometer',
      type: 'Optical',
      manufacturer: 'J.A. Woollam',
      model: 'M-2000',
      serialNumber: 'JAW-M2K-2023-042',
      location: 'Lab B - Bay 5',
      status: 'offline' as const,
      lastCalibration: '2024-06-10',
      nextCalibration: '2024-12-10',
      lastMaintenance: '2024-11-05',
      nextMaintenance: '2025-02-05',
      usageHours: 1678,
      technician: 'Sarah Johnson',
      notes: 'Light source replacement - parts on order'
    },
    {
      id: 'INST-009',
      name: 'Raman Spectrometer',
      type: 'Optical',
      manufacturer: 'Horiba',
      model: 'LabRAM HR',
      serialNumber: 'HR-LR-2024-009',
      location: 'Lab B - Bay 7',
      status: 'operational' as const,
      lastCalibration: '2024-09-25',
      nextCalibration: '2024-12-25',
      lastMaintenance: '2024-10-12',
      nextMaintenance: '2025-01-12',
      usageHours: 734,
      technician: 'David Wilson'
    },
    {
      id: 'INST-010',
      name: 'AFM System',
      type: 'Structural',
      manufacturer: 'Bruker',
      model: 'Dimension Icon',
      serialNumber: 'BR-DI-2024-001',
      location: 'Lab C - Bay 6',
      status: 'calibration_needed' as const,
      lastCalibration: '2024-04-20',
      nextCalibration: '2024-10-20',
      lastMaintenance: '2024-08-30',
      nextMaintenance: '2024-11-30',
      usageHours: 2301,
      technician: 'Emily Brown',
      notes: 'Calibration overdue - scheduled for next week'
    }
  ]
  return instruments
}

export default function InstrumentsPage() {
  const [instruments, setInstruments] = useState<Instrument[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [filterType, setFilterType] = useState<string>('all')
  const [filterStatus, setFilterStatus] = useState<string>('all')
  const [selectedInstrument, setSelectedInstrument] = useState<Instrument | null>(null)
  const [showNewDialog, setShowNewDialog] = useState(false)
  const [showDetailsDialog, setShowDetailsDialog] = useState(false)
  const [showEditDialog, setShowEditDialog] = useState(false)

  const [newInstrument, setNewInstrument] = useState({
    name: '',
    type: 'Electrical',
    manufacturer: '',
    model: '',
    serialNumber: '',
    location: '',
    technician: '',
    notes: ''
  })

  // Generate mock data on client side only
  useEffect(() => {
    setInstruments(generateMockInstruments())
  }, [])

  // Filter instruments
  const filteredInstruments = instruments.filter(inst => {
    const matchesSearch =
      inst.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      inst.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      inst.manufacturer.toLowerCase().includes(searchQuery.toLowerCase()) ||
      inst.model.toLowerCase().includes(searchQuery.toLowerCase())

    const matchesType = filterType === 'all' || inst.type === filterType
    const matchesStatus = filterStatus === 'all' || inst.status === filterStatus

    return matchesSearch && matchesType && matchesStatus
  })

  // Statistics
  const stats = {
    total: instruments.length,
    operational: instruments.filter(i => i.status === 'operational').length,
    calibrationNeeded: instruments.filter(i => i.status === 'calibration_needed').length,
    maintenance: instruments.filter(i => i.status === 'maintenance').length,
    offline: instruments.filter(i => i.status === 'offline').length
  }

  const handleCreateInstrument = () => {
    if (!newInstrument.name || !newInstrument.manufacturer || !newInstrument.model) {
      alert('Please fill in all required fields')
      return
    }

    const today = new Date().toISOString().split('T')[0]
    const nextCalib = new Date()
    nextCalib.setMonth(nextCalib.getMonth() + 3)
    const nextMaint = new Date()
    nextMaint.setMonth(nextMaint.getMonth() + 3)

    const instrument: Instrument = {
      id: `INST-${String(instruments.length + 1).padStart(3, '0')}`,
      name: newInstrument.name,
      type: newInstrument.type,
      manufacturer: newInstrument.manufacturer,
      model: newInstrument.model,
      serialNumber: newInstrument.serialNumber,
      location: newInstrument.location,
      status: 'operational',
      lastCalibration: today,
      nextCalibration: nextCalib.toISOString().split('T')[0],
      lastMaintenance: today,
      nextMaintenance: nextMaint.toISOString().split('T')[0],
      usageHours: 0,
      technician: newInstrument.technician,
      notes: newInstrument.notes
    }

    setInstruments([instrument, ...instruments])
    setShowNewDialog(false)
    setNewInstrument({
      name: '',
      type: 'Electrical',
      manufacturer: '',
      model: '',
      serialNumber: '',
      location: '',
      technician: '',
      notes: ''
    })
  }

  const handleUpdateInstrument = () => {
    if (!selectedInstrument) return

    setInstruments(instruments.map(inst =>
      inst.id === selectedInstrument.id ? selectedInstrument : inst
    ))
    setShowEditDialog(false)
    setSelectedInstrument(null)
  }

  const handleDeleteInstrument = (id: string) => {
    if (confirm('Are you sure you want to delete this instrument?')) {
      setInstruments(instruments.filter(inst => inst.id !== id))
    }
  }

  const handlePerformCalibration = (instrument: Instrument) => {
    const today = new Date().toISOString().split('T')[0]
    const nextCalib = new Date()
    nextCalib.setMonth(nextCalib.getMonth() + 3)

    setInstruments(instruments.map(inst => {
      if (inst.id === instrument.id) {
        return {
          ...inst,
          lastCalibration: today,
          nextCalibration: nextCalib.toISOString().split('T')[0],
          status: 'operational' as const
        }
      }
      return inst
    }))
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'operational': return 'bg-green-100 text-green-800'
      case 'calibration_needed': return 'bg-yellow-100 text-yellow-800'
      case 'maintenance': return 'bg-blue-100 text-blue-800'
      case 'offline': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'operational': return <CheckCircle2 className="w-4 h-4" />
      case 'calibration_needed': return <AlertCircle className="w-4 h-4" />
      case 'maintenance': return <Wrench className="w-4 h-4" />
      case 'offline': return <AlertCircle className="w-4 h-4" />
      default: return <Clock className="w-4 h-4" />
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center">
            <Settings className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Instrument Management</h1>
            <p className="text-gray-600 mt-1">Monitor and manage laboratory instruments</p>
          </div>
        </div>

        <Button onClick={() => setShowNewDialog(true)}>
          <Plus className="w-5 h-5 mr-2" />
          Add Instrument
        </Button>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-5 gap-4 mb-6">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Instruments</p>
              <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
            </div>
            <Activity className="w-8 h-8 text-gray-400" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Operational</p>
              <p className="text-2xl font-bold text-green-600">{stats.operational}</p>
            </div>
            <CheckCircle2 className="w-8 h-8 text-green-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Calibration Needed</p>
              <p className="text-2xl font-bold text-yellow-600">{stats.calibrationNeeded}</p>
            </div>
            <AlertCircle className="w-8 h-8 text-yellow-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">In Maintenance</p>
              <p className="text-2xl font-bold text-blue-600">{stats.maintenance}</p>
            </div>
            <Wrench className="w-8 h-8 text-blue-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Offline</p>
              <p className="text-2xl font-bold text-red-600">{stats.offline}</p>
            </div>
            <AlertCircle className="w-8 h-8 text-red-500" />
          </div>
        </Card>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 mb-6">
        <div className="grid grid-cols-3 gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <Input
              placeholder="Search instruments..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
          <Select value={filterType} onValueChange={setFilterType}>
            <SelectTrigger>
              <SelectValue placeholder="Filter by type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Types</SelectItem>
              <SelectItem value="Electrical">Electrical</SelectItem>
              <SelectItem value="Optical">Optical</SelectItem>
              <SelectItem value="Structural">Structural</SelectItem>
              <SelectItem value="Chemical">Chemical</SelectItem>
            </SelectContent>
          </Select>
          <Select value={filterStatus} onValueChange={setFilterStatus}>
            <SelectTrigger>
              <SelectValue placeholder="Filter by status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Statuses</SelectItem>
              <SelectItem value="operational">Operational</SelectItem>
              <SelectItem value="calibration_needed">Calibration Needed</SelectItem>
              <SelectItem value="maintenance">Maintenance</SelectItem>
              <SelectItem value="offline">Offline</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Instruments Grid */}
      <div className="grid grid-cols-1 gap-4">
        {filteredInstruments.map((instrument) => (
          <Card key={instrument.id} className="p-6 hover:shadow-lg transition-shadow">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-3">
                  <h3 className="text-lg font-semibold text-gray-900">{instrument.name}</h3>
                  <Badge className={getStatusColor(instrument.status)}>
                    {getStatusIcon(instrument.status)}
                    <span className="ml-1">{instrument.status.replace('_', ' ')}</span>
                  </Badge>
                  <Badge variant="outline">{instrument.type}</Badge>
                </div>

                <div className="grid grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-gray-600">ID</p>
                    <p className="font-medium text-gray-900">{instrument.id}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Manufacturer</p>
                    <p className="font-medium text-gray-900">{instrument.manufacturer}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Model</p>
                    <p className="font-medium text-gray-900">{instrument.model}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Location</p>
                    <p className="font-medium text-gray-900">{instrument.location}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Usage Hours</p>
                    <p className="font-medium text-gray-900">{instrument.usageHours.toLocaleString()} hrs</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Next Calibration</p>
                    <p className="font-medium text-gray-900">{instrument.nextCalibration}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Next Maintenance</p>
                    <p className="font-medium text-gray-900">{instrument.nextMaintenance}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Technician</p>
                    <p className="font-medium text-gray-900">{instrument.technician}</p>
                  </div>
                </div>

                {instrument.notes && (
                  <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-700">{instrument.notes}</p>
                  </div>
                )}
              </div>

              <div className="flex gap-2 ml-4">
                {instrument.status === 'calibration_needed' && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handlePerformCalibration(instrument)}
                  >
                    <CheckCircle2 className="w-4 h-4 mr-1" />
                    Calibrate
                  </Button>
                )}
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    setSelectedInstrument(instrument)
                    setShowDetailsDialog(true)
                  }}
                >
                  <Eye className="w-4 h-4" />
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    setSelectedInstrument(instrument)
                    setShowEditDialog(true)
                  }}
                >
                  <Edit2 className="w-4 h-4" />
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleDeleteInstrument(instrument.id)}
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {filteredInstruments.length === 0 && (
        <div className="text-center py-12">
          <Settings className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No instruments found</h3>
          <p className="text-gray-600">Try adjusting your search or filters</p>
        </div>
      )}

      {/* Add Instrument Dialog */}
      <Dialog open={showNewDialog} onOpenChange={setShowNewDialog}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Add New Instrument</DialogTitle>
            <DialogDescription>Enter the details for the new instrument</DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Instrument Name *</Label>
                <Input
                  value={newInstrument.name}
                  onChange={(e) => setNewInstrument({ ...newInstrument, name: e.target.value })}
                  placeholder="e.g., Four-Point Probe System"
                />
              </div>
              <div>
                <Label>Type *</Label>
                <Select
                  value={newInstrument.type}
                  onValueChange={(value) => setNewInstrument({ ...newInstrument, type: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Electrical">Electrical</SelectItem>
                    <SelectItem value="Optical">Optical</SelectItem>
                    <SelectItem value="Structural">Structural</SelectItem>
                    <SelectItem value="Chemical">Chemical</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Manufacturer *</Label>
                <Input
                  value={newInstrument.manufacturer}
                  onChange={(e) => setNewInstrument({ ...newInstrument, manufacturer: e.target.value })}
                  placeholder="e.g., Keithley"
                />
              </div>
              <div>
                <Label>Model *</Label>
                <Input
                  value={newInstrument.model}
                  onChange={(e) => setNewInstrument({ ...newInstrument, model: e.target.value })}
                  placeholder="e.g., 4200-SCS"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Serial Number</Label>
                <Input
                  value={newInstrument.serialNumber}
                  onChange={(e) => setNewInstrument({ ...newInstrument, serialNumber: e.target.value })}
                  placeholder="e.g., K4200-2024-001"
                />
              </div>
              <div>
                <Label>Location</Label>
                <Input
                  value={newInstrument.location}
                  onChange={(e) => setNewInstrument({ ...newInstrument, location: e.target.value })}
                  placeholder="e.g., Lab A - Bay 3"
                />
              </div>
            </div>

            <div>
              <Label>Technician</Label>
              <Input
                value={newInstrument.technician}
                onChange={(e) => setNewInstrument({ ...newInstrument, technician: e.target.value })}
                placeholder="Assigned technician name"
              />
            </div>

            <div>
              <Label>Notes</Label>
              <Textarea
                value={newInstrument.notes}
                onChange={(e) => setNewInstrument({ ...newInstrument, notes: e.target.value })}
                placeholder="Additional notes"
                rows={3}
              />
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setShowNewDialog(false)}>Cancel</Button>
            <Button onClick={handleCreateInstrument}>Add Instrument</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Details Dialog */}
      <Dialog open={showDetailsDialog} onOpenChange={setShowDetailsDialog}>
        <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Instrument Details</DialogTitle>
          </DialogHeader>

          {selectedInstrument && (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <h2 className="text-2xl font-bold text-gray-900">{selectedInstrument.name}</h2>
                <Badge className={getStatusColor(selectedInstrument.status)}>
                  {getStatusIcon(selectedInstrument.status)}
                  <span className="ml-1">{selectedInstrument.status.replace('_', ' ')}</span>
                </Badge>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-3">
                  <div>
                    <Label className="text-gray-600">Instrument ID</Label>
                    <p className="font-medium">{selectedInstrument.id}</p>
                  </div>
                  <div>
                    <Label className="text-gray-600">Type</Label>
                    <p className="font-medium">{selectedInstrument.type}</p>
                  </div>
                  <div>
                    <Label className="text-gray-600">Manufacturer</Label>
                    <p className="font-medium">{selectedInstrument.manufacturer}</p>
                  </div>
                  <div>
                    <Label className="text-gray-600">Model</Label>
                    <p className="font-medium">{selectedInstrument.model}</p>
                  </div>
                  <div>
                    <Label className="text-gray-600">Serial Number</Label>
                    <p className="font-medium">{selectedInstrument.serialNumber}</p>
                  </div>
                </div>

                <div className="space-y-3">
                  <div>
                    <Label className="text-gray-600">Location</Label>
                    <p className="font-medium">{selectedInstrument.location}</p>
                  </div>
                  <div>
                    <Label className="text-gray-600">Usage Hours</Label>
                    <p className="font-medium">{selectedInstrument.usageHours.toLocaleString()} hours</p>
                  </div>
                  <div>
                    <Label className="text-gray-600">Technician</Label>
                    <p className="font-medium">{selectedInstrument.technician}</p>
                  </div>
                  <div>
                    <Label className="text-gray-600">Last Calibration</Label>
                    <p className="font-medium">{selectedInstrument.lastCalibration}</p>
                  </div>
                  <div>
                    <Label className="text-gray-600">Next Calibration</Label>
                    <p className="font-medium">{selectedInstrument.nextCalibration}</p>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label className="text-gray-600">Last Maintenance</Label>
                  <p className="font-medium">{selectedInstrument.lastMaintenance}</p>
                </div>
                <div>
                  <Label className="text-gray-600">Next Maintenance</Label>
                  <p className="font-medium">{selectedInstrument.nextMaintenance}</p>
                </div>
              </div>

              {selectedInstrument.notes && (
                <div>
                  <Label className="text-gray-600">Notes</Label>
                  <div className="mt-1 p-3 bg-gray-50 rounded-lg">
                    <p className="text-gray-900">{selectedInstrument.notes}</p>
                  </div>
                </div>
              )}
            </div>
          )}

          <DialogFooter>
            <Button onClick={() => setShowDetailsDialog(false)}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog open={showEditDialog} onOpenChange={setShowEditDialog}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Edit Instrument</DialogTitle>
          </DialogHeader>

          {selectedInstrument && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Instrument Name</Label>
                  <Input
                    value={selectedInstrument.name}
                    onChange={(e) => setSelectedInstrument({ ...selectedInstrument, name: e.target.value })}
                  />
                </div>
                <div>
                  <Label>Type</Label>
                  <Select
                    value={selectedInstrument.type}
                    onValueChange={(value) => setSelectedInstrument({ ...selectedInstrument, type: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Electrical">Electrical</SelectItem>
                      <SelectItem value="Optical">Optical</SelectItem>
                      <SelectItem value="Structural">Structural</SelectItem>
                      <SelectItem value="Chemical">Chemical</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Status</Label>
                  <Select
                    value={selectedInstrument.status}
                    onValueChange={(value: any) => setSelectedInstrument({ ...selectedInstrument, status: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="operational">Operational</SelectItem>
                      <SelectItem value="calibration_needed">Calibration Needed</SelectItem>
                      <SelectItem value="maintenance">Maintenance</SelectItem>
                      <SelectItem value="offline">Offline</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Location</Label>
                  <Input
                    value={selectedInstrument.location}
                    onChange={(e) => setSelectedInstrument({ ...selectedInstrument, location: e.target.value })}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Technician</Label>
                  <Input
                    value={selectedInstrument.technician}
                    onChange={(e) => setSelectedInstrument({ ...selectedInstrument, technician: e.target.value })}
                  />
                </div>
                <div>
                  <Label>Usage Hours</Label>
                  <Input
                    type="number"
                    value={selectedInstrument.usageHours}
                    onChange={(e) => setSelectedInstrument({ ...selectedInstrument, usageHours: parseInt(e.target.value) || 0 })}
                  />
                </div>
              </div>

              <div>
                <Label>Notes</Label>
                <Textarea
                  value={selectedInstrument.notes || ''}
                  onChange={(e) => setSelectedInstrument({ ...selectedInstrument, notes: e.target.value })}
                  rows={3}
                />
              </div>
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setShowEditDialog(false)}>Cancel</Button>
            <Button onClick={handleUpdateInstrument}>Save Changes</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
