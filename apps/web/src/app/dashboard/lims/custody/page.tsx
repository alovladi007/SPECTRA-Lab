'use client'
import { useState } from 'react'
import { Shield, Search, Plus, ArrowRight, User, Calendar, MapPin, FileText, CheckCircle, AlertCircle } from 'lucide-react'

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

const mockCustodyRecords: CustodyRecord[] = [
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
  }
]

const statusConfig = {
  pending: { color: 'bg-yellow-100 text-yellow-800', icon: AlertCircle },
  in_transit: { color: 'bg-blue-100 text-blue-800', icon: ArrowRight },
  completed: { color: 'bg-green-100 text-green-800', icon: CheckCircle },
  rejected: { color: 'bg-red-100 text-red-800', icon: AlertCircle }
}

export default function CustodyPage() {
  const [records, setRecords] = useState(mockCustodyRecords)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [showNewTransfer, setShowNewTransfer] = useState(false)

  const filteredRecords = records.filter(record => {
    const matchesSearch =
      record.sampleName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      record.sampleId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      record.from.person.toLowerCase().includes(searchTerm.toLowerCase()) ||
      record.to.person.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesStatus = statusFilter === 'all' || record.status === statusFilter

    return matchesSearch && matchesStatus
  })

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
        <button
          onClick={() => setShowNewTransfer(true)}
          className="flex items-center gap-2 bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition-colors"
        >
          <Plus className="w-5 h-5" />
          New Transfer
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Transfers</p>
              <p className="text-2xl font-bold text-gray-900">{records.length}</p>
            </div>
            <Shield className="w-8 h-8 text-purple-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">In Transit</p>
              <p className="text-2xl font-bold text-blue-600">
                {records.filter(r => r.status === 'in_transit').length}
              </p>
            </div>
            <ArrowRight className="w-8 h-8 text-blue-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Pending</p>
              <p className="text-2xl font-bold text-yellow-600">
                {records.filter(r => r.status === 'pending').length}
              </p>
            </div>
            <AlertCircle className="w-8 h-8 text-yellow-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Completed</p>
              <p className="text-2xl font-bold text-green-600">
                {records.filter(r => r.status === 'completed').length}
              </p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
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
              placeholder="Search by sample, person, or ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            <option value="all">All Status</option>
            <option value="pending">Pending</option>
            <option value="in_transit">In Transit</option>
            <option value="completed">Completed</option>
            <option value="rejected">Rejected</option>
          </select>
        </div>
      </div>

      {/* Transfer Records */}
      <div className="space-y-4">
        {filteredRecords.map((record) => {
          const StatusIcon = statusConfig[record.status].icon
          return (
            <div key={record.id} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
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
                <span className={`px-3 py-1 rounded-full text-sm font-medium flex items-center gap-1 ${statusConfig[record.status].color}`}>
                  <StatusIcon className="w-4 h-4" />
                  {record.status.replace('_', ' ')}
                </span>
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
                <button className="text-purple-600 hover:text-purple-900 text-sm font-medium">
                  View Details
                </button>
                <button className="text-blue-600 hover:text-blue-900 text-sm font-medium">
                  Print COC Form
                </button>
                {record.status === 'in_transit' && (
                  <button className="text-green-600 hover:text-green-900 text-sm font-medium">
                    Complete Transfer
                  </button>
                )}
              </div>
            </div>
          )
        })}

        {filteredRecords.length === 0 && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
            <Shield className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No custody records found matching your criteria</p>
          </div>
        )}
      </div>

      {/* New Transfer Modal */}
      {showNewTransfer && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4">
            <h2 className="text-2xl font-bold mb-4">New Chain of Custody Transfer</h2>
            <p className="text-gray-600 mb-4">COC transfer form will be implemented here</p>
            <button
              onClick={() => setShowNewTransfer(false)}
              className="bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
