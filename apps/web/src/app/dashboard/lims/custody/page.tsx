'use client'
import { useState, useEffect } from 'react'
import { Shield, Search, Plus, ArrowRight, User, Calendar, MapPin, FileText, CheckCircle, AlertCircle, X } from 'lucide-react'
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
import { Textarea } from '@/components/ui/textarea'

interface CustodyRecord {
  id: string
  sampleId: string
  sampleName: string
  from: {
    person: string
    location: string
    date: string
  }
  to: {
    person: string
    location: string
    date: string
  }
  purpose: string
  status: 'pending' | 'in_transit' | 'completed' | 'rejected'
  conditions?: string
}

interface NewTransferForm {
  sampleId: string
  sampleName: string
  fromPerson: string
  fromLocation: string
  toPerson: string
  toLocation: string
  purpose: string
  conditions: string
}

const generateMockCustodyRecords = (): CustodyRecord[] => [
  {
    id: 'COC-001',
    sampleId: 'SMP-001',
    sampleName: 'Silicon Wafer Batch A',
    from: {
      person: 'Dr. Chen',
      location: 'Cleanroom-B2',
      date: '2024-11-08 09:00'
    },
    to: {
      person: 'Dr. Patel',
      location: 'Testing-Lab1',
      date: '2024-11-08 14:30'
    },
    purpose: 'Electrical characterization testing',
    status: 'completed',
    conditions: 'Maintained at -20°C during transit'
  },
  {
    id: 'COC-002',
    sampleId: 'SMP-002',
    sampleName: 'Thin Film Sample Set',
    from: {
      person: 'Lab Manager',
      location: 'Receiving',
      date: '2024-11-09 08:15'
    },
    to: {
      person: 'Dr. Martinez',
      location: 'Storage-A3',
      date: '2024-11-09 09:00'
    },
    purpose: 'Initial storage after receiving',
    status: 'in_transit',
    conditions: 'Room temperature'
  },
  {
    id: 'COC-003',
    sampleId: 'SMP-003',
    sampleName: 'Device Prototype Rev3',
    from: {
      person: 'Dr. Patel',
      location: 'Testing-Lab1',
      date: '2024-11-09 10:30'
    },
    to: {
      person: 'Dr. Kim',
      location: 'Analysis-Lab2',
      date: ''
    },
    purpose: 'Failure analysis requested',
    status: 'pending',
    conditions: 'ESD protection required'
  },
  {
    id: 'COC-004',
    sampleId: 'SMP-005',
    sampleName: 'GaN on Sapphire',
    from: {
      person: 'Vendor',
      location: 'External',
      date: '2024-11-07 14:00'
    },
    to: {
      person: 'Lab Manager',
      location: 'Receiving',
      date: '2024-11-07 15:45'
    },
    purpose: 'Initial vendor delivery',
    status: 'completed',
    conditions: 'Sealed package, ambient conditions'
  },
  {
    id: 'COC-005',
    sampleId: 'SMP-007',
    sampleName: 'Optical Component Array',
    from: {
      person: 'Dr. Zhang',
      location: 'Fabrication-C1',
      date: '2024-11-09 11:00'
    },
    to: {
      person: 'Dr. Brown',
      location: 'Optical-Testing',
      date: ''
    },
    purpose: 'Optical characterization',
    status: 'pending',
    conditions: 'Light-sensitive, keep in dark container'
  },
  {
    id: 'COC-006',
    sampleId: 'SMP-008',
    sampleName: 'MEMS Device Series',
    from: {
      person: 'Dr. Lee',
      location: 'MEMS-Lab',
      date: '2024-11-08 13:30'
    },
    to: {
      person: 'Dr. Wilson',
      location: 'Reliability-Test',
      date: '2024-11-08 16:00'
    },
    purpose: 'Environmental stress testing',
    status: 'in_transit',
    conditions: 'Vacuum sealed packaging'
  },
  {
    id: 'COC-007',
    sampleId: 'SMP-010',
    sampleName: 'Biosensor Prototype',
    from: {
      person: 'Dr. Garcia',
      location: 'Bio-Lab',
      date: '2024-11-07 10:00'
    },
    to: {
      person: 'Dr. Anderson',
      location: 'Testing-Lab2',
      date: '2024-11-07 14:30'
    },
    purpose: 'Sensitivity testing',
    status: 'completed',
    conditions: '4°C storage required'
  },
  {
    id: 'COC-008',
    sampleId: 'SMP-012',
    sampleName: 'Quantum Dot Samples',
    from: {
      person: 'Dr. Thompson',
      location: 'Synthesis-Lab',
      date: '2024-11-09 09:45'
    },
    to: {
      person: 'Dr. White',
      location: 'Spectroscopy-Lab',
      date: ''
    },
    purpose: 'Photoluminescence analysis',
    status: 'pending',
    conditions: 'Inert atmosphere required'
  }
]

const statusConfig = {
  pending: { color: 'bg-yellow-100 text-yellow-800', icon: AlertCircle, label: 'Pending' },
  in_transit: { color: 'bg-blue-100 text-blue-800', icon: ArrowRight, label: 'In Transit' },
  completed: { color: 'bg-green-100 text-green-800', icon: CheckCircle, label: 'Completed' },
  rejected: { color: 'bg-red-100 text-red-800', icon: AlertCircle, label: 'Rejected' }
}

export default function CustodyPage() {
  const [records, setRecords] = useState<CustodyRecord[]>([])
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [showNewTransfer, setShowNewTransfer] = useState(false)
  const [selectedRecord, setSelectedRecord] = useState<CustodyRecord | null>(null)
  const [showDetailsDialog, setShowDetailsDialog] = useState(false)

  const [newTransfer, setNewTransfer] = useState<NewTransferForm>({
    sampleId: '',
    sampleName: '',
    fromPerson: '',
    fromLocation: '',
    toPerson: '',
    toLocation: '',
    purpose: '',
    conditions: ''
  })

  // Generate mock data on client side to prevent hydration errors
  useEffect(() => {
    setRecords(generateMockCustodyRecords())
  }, [])

  const filteredRecords = records.filter(record => {
    const matchesSearch =
      record.sampleName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      record.sampleId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      record.from.person.toLowerCase().includes(searchTerm.toLowerCase()) ||
      record.to.person.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesStatus = statusFilter === 'all' || record.status === statusFilter

    return matchesSearch && matchesStatus
  })

  const handleCreateTransfer = () => {
    if (!newTransfer.sampleId || !newTransfer.sampleName || !newTransfer.fromPerson ||
        !newTransfer.fromLocation || !newTransfer.toPerson || !newTransfer.toLocation ||
        !newTransfer.purpose) {
      alert('Please fill in all required fields')
      return
    }

    const newRecord: CustodyRecord = {
      id: `COC-${String(records.length + 1).padStart(3, '0')}`,
      sampleId: newTransfer.sampleId,
      sampleName: newTransfer.sampleName,
      from: {
        person: newTransfer.fromPerson,
        location: newTransfer.fromLocation,
        date: new Date().toISOString().slice(0, 16).replace('T', ' ')
      },
      to: {
        person: newTransfer.toPerson,
        location: newTransfer.toLocation,
        date: ''
      },
      purpose: newTransfer.purpose,
      status: 'pending',
      conditions: newTransfer.conditions
    }

    setRecords([newRecord, ...records])
    setShowNewTransfer(false)
    setNewTransfer({
      sampleId: '',
      sampleName: '',
      fromPerson: '',
      fromLocation: '',
      toPerson: '',
      toLocation: '',
      purpose: '',
      conditions: ''
    })
  }

  const handleCompleteTransfer = (recordId: string) => {
    setRecords(records.map(record => {
      if (record.id === recordId && record.status === 'in_transit') {
        return {
          ...record,
          status: 'completed' as const,
          to: {
            ...record.to,
            date: new Date().toISOString().slice(0, 16).replace('T', ' ')
          }
        }
      }
      return record
    }))
  }

  const handleStartTransfer = (recordId: string) => {
    setRecords(records.map(record => {
      if (record.id === recordId && record.status === 'pending') {
        return {
          ...record,
          status: 'in_transit' as const
        }
      }
      return record
    }))
  }

  const handleViewDetails = (record: CustodyRecord) => {
    setSelectedRecord(record)
    setShowDetailsDialog(true)
  }

  const stats = {
    total: records.length,
    in_transit: records.filter(r => r.status === 'in_transit').length,
    pending: records.filter(r => r.status === 'pending').length,
    completed: records.filter(r => r.status === 'completed').length
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
            <Shield className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Chain of Custody</h1>
            <p className="text-gray-600 mt-1">Track sample transfers with audit trail</p>
          </div>
        </div>
        <Button
          onClick={() => setShowNewTransfer(true)}
          className="flex items-center gap-2"
        >
          <Plus className="w-5 h-5" />
          New Transfer
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Transfers</p>
              <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
            </div>
            <Shield className="w-8 h-8 text-purple-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">In Transit</p>
              <p className="text-2xl font-bold text-blue-600">{stats.in_transit}</p>
            </div>
            <ArrowRight className="w-8 h-8 text-blue-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Pending</p>
              <p className="text-2xl font-bold text-yellow-600">{stats.pending}</p>
            </div>
            <AlertCircle className="w-8 h-8 text-yellow-500" />
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
      </div>

      {/* Filters */}
      <Card className="p-4">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <Input
              type="text"
              placeholder="Search by sample, person, or ID..."
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
              <SelectItem value="in_transit">In Transit</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="rejected">Rejected</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </Card>

      {/* Transfer Records */}
      <div className="space-y-4">
        {filteredRecords.map((record) => {
          const StatusIcon = statusConfig[record.status].icon
          return (
            <Card key={record.id} className="p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                    <Shield className="w-5 h-5 text-purple-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">{record.sampleName}</h3>
                    <p className="text-sm text-gray-500">COC ID: {record.id} • Sample: {record.sampleId}</p>
                  </div>
                </div>
                <Badge className={statusConfig[record.status].color}>
                  <StatusIcon className="w-4 h-4 mr-1" />
                  {statusConfig[record.status].label}
                </Badge>
              </div>

              {/* Transfer Timeline */}
              <div className="flex items-center gap-4">
                {/* From */}
                <div className="flex-1 bg-gray-50 rounded-lg p-4">
                  <p className="text-xs text-gray-500 uppercase font-medium mb-2">From</p>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm">
                      <User className="w-4 h-4 text-gray-400" />
                      <span className="font-medium text-gray-900">{record.from.person}</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      <MapPin className="w-4 h-4 text-gray-400" />
                      {record.from.location}
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      <Calendar className="w-4 h-4 text-gray-400" />
                      {record.from.date}
                    </div>
                  </div>
                </div>

                {/* Arrow */}
                <div className="flex-shrink-0">
                  <ArrowRight className="w-8 h-8 text-purple-500" />
                </div>

                {/* To */}
                <div className="flex-1 bg-gray-50 rounded-lg p-4">
                  <p className="text-xs text-gray-500 uppercase font-medium mb-2">To</p>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm">
                      <User className="w-4 h-4 text-gray-400" />
                      <span className="font-medium text-gray-900">{record.to.person}</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      <MapPin className="w-4 h-4 text-gray-400" />
                      {record.to.location}
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      <Calendar className="w-4 h-4 text-gray-400" />
                      {record.to.date || 'Pending'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Purpose and Conditions */}
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-gray-500 uppercase font-medium mb-1">Purpose</p>
                    <p className="text-sm text-gray-900">{record.purpose}</p>
                  </div>
                  {record.conditions && (
                    <div>
                      <p className="text-xs text-gray-500 uppercase font-medium mb-1">Conditions</p>
                      <p className="text-sm text-gray-900">{record.conditions}</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Actions */}
              <div className="mt-4 pt-4 border-t border-gray-200 flex gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleViewDetails(record)}
                  className="text-purple-600 hover:text-purple-900"
                >
                  View Details
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-blue-600 hover:text-blue-900"
                >
                  Print COC Form
                </Button>
                {record.status === 'pending' && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleStartTransfer(record.id)}
                    className="text-blue-600 hover:text-blue-900"
                  >
                    Start Transfer
                  </Button>
                )}
                {record.status === 'in_transit' && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleCompleteTransfer(record.id)}
                    className="text-green-600 hover:text-green-900"
                  >
                    Complete Transfer
                  </Button>
                )}
              </div>
            </Card>
          )
        })}

        {filteredRecords.length === 0 && (
          <Card className="p-12 text-center">
            <Shield className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No custody records found matching your criteria</p>
          </Card>
        )}
      </div>

      {/* New Transfer Dialog */}
      <Dialog open={showNewTransfer} onOpenChange={setShowNewTransfer}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>New Chain of Custody Transfer</DialogTitle>
            <DialogDescription>
              Create a new custody transfer record for sample tracking
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium text-gray-700 mb-1 block">
                  Sample ID *
                </label>
                <Input
                  value={newTransfer.sampleId}
                  onChange={(e) => setNewTransfer({ ...newTransfer, sampleId: e.target.value })}
                  placeholder="e.g., SMP-001"
                />
              </div>
              <div>
                <label className="text-sm font-medium text-gray-700 mb-1 block">
                  Sample Name *
                </label>
                <Input
                  value={newTransfer.sampleName}
                  onChange={(e) => setNewTransfer({ ...newTransfer, sampleName: e.target.value })}
                  placeholder="e.g., Silicon Wafer"
                />
              </div>
            </div>

            <div className="border-t border-gray-200 pt-4">
              <h4 className="text-sm font-semibold text-gray-900 mb-3">Transfer From</h4>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium text-gray-700 mb-1 block">
                    Person *
                  </label>
                  <Input
                    value={newTransfer.fromPerson}
                    onChange={(e) => setNewTransfer({ ...newTransfer, fromPerson: e.target.value })}
                    placeholder="e.g., Dr. Smith"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700 mb-1 block">
                    Location *
                  </label>
                  <Input
                    value={newTransfer.fromLocation}
                    onChange={(e) => setNewTransfer({ ...newTransfer, fromLocation: e.target.value })}
                    placeholder="e.g., Cleanroom-A1"
                  />
                </div>
              </div>
            </div>

            <div className="border-t border-gray-200 pt-4">
              <h4 className="text-sm font-semibold text-gray-900 mb-3">Transfer To</h4>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium text-gray-700 mb-1 block">
                    Person *
                  </label>
                  <Input
                    value={newTransfer.toPerson}
                    onChange={(e) => setNewTransfer({ ...newTransfer, toPerson: e.target.value })}
                    placeholder="e.g., Dr. Johnson"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700 mb-1 block">
                    Location *
                  </label>
                  <Input
                    value={newTransfer.toLocation}
                    onChange={(e) => setNewTransfer({ ...newTransfer, toLocation: e.target.value })}
                    placeholder="e.g., Testing-Lab1"
                  />
                </div>
              </div>
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Purpose *
              </label>
              <Textarea
                value={newTransfer.purpose}
                onChange={(e) => setNewTransfer({ ...newTransfer, purpose: e.target.value })}
                placeholder="Describe the reason for this transfer"
                className="min-h-[80px]"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Special Conditions
              </label>
              <Textarea
                value={newTransfer.conditions}
                onChange={(e) => setNewTransfer({ ...newTransfer, conditions: e.target.value })}
                placeholder="Any special handling or storage conditions"
                className="min-h-[80px]"
              />
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowNewTransfer(false)}
            >
              Cancel
            </Button>
            <Button onClick={handleCreateTransfer}>
              Create Transfer
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Details Dialog */}
      <Dialog open={showDetailsDialog} onOpenChange={setShowDetailsDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Chain of Custody Details</DialogTitle>
            <DialogDescription>
              Complete transfer information and audit trail
            </DialogDescription>
          </DialogHeader>

          {selectedRecord && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-medium text-gray-500">COC ID</p>
                  <p className="text-base font-semibold text-gray-900">{selectedRecord.id}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Status</p>
                  <Badge className={statusConfig[selectedRecord.status].color}>
                    {statusConfig[selectedRecord.status].label}
                  </Badge>
                </div>
              </div>

              <div className="border-t border-gray-200 pt-4">
                <p className="text-sm font-medium text-gray-500 mb-2">Sample Information</p>
                <div className="bg-gray-50 rounded-lg p-3 space-y-1">
                  <p className="text-sm"><span className="font-medium">ID:</span> {selectedRecord.sampleId}</p>
                  <p className="text-sm"><span className="font-medium">Name:</span> {selectedRecord.sampleName}</p>
                </div>
              </div>

              <div className="border-t border-gray-200 pt-4">
                <p className="text-sm font-medium text-gray-500 mb-2">Transfer From</p>
                <div className="bg-gray-50 rounded-lg p-3 space-y-1">
                  <p className="text-sm"><span className="font-medium">Person:</span> {selectedRecord.from.person}</p>
                  <p className="text-sm"><span className="font-medium">Location:</span> {selectedRecord.from.location}</p>
                  <p className="text-sm"><span className="font-medium">Date:</span> {selectedRecord.from.date}</p>
                </div>
              </div>

              <div className="border-t border-gray-200 pt-4">
                <p className="text-sm font-medium text-gray-500 mb-2">Transfer To</p>
                <div className="bg-gray-50 rounded-lg p-3 space-y-1">
                  <p className="text-sm"><span className="font-medium">Person:</span> {selectedRecord.to.person}</p>
                  <p className="text-sm"><span className="font-medium">Location:</span> {selectedRecord.to.location}</p>
                  <p className="text-sm"><span className="font-medium">Date:</span> {selectedRecord.to.date || 'Pending'}</p>
                </div>
              </div>

              <div className="border-t border-gray-200 pt-4">
                <p className="text-sm font-medium text-gray-500 mb-2">Purpose</p>
                <div className="bg-gray-50 rounded-lg p-3">
                  <p className="text-sm">{selectedRecord.purpose}</p>
                </div>
              </div>

              {selectedRecord.conditions && (
                <div className="border-t border-gray-200 pt-4">
                  <p className="text-sm font-medium text-gray-500 mb-2">Special Conditions</p>
                  <div className="bg-gray-50 rounded-lg p-3">
                    <p className="text-sm">{selectedRecord.conditions}</p>
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
