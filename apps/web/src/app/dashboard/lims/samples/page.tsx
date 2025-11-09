'use client'
import { useState } from 'react'
import { ClipboardCheck, Barcode, Plus, Search, Filter, Download, QrCode, MapPin, Calendar, User } from 'lucide-react'

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
}

const mockSamples: Sample[] = [
  {
    id: 'SMP-001',
    barcode: 'WAF-2024-001-A1',
    name: 'Silicon Wafer Batch A',
    type: 'Wafer',
    location: 'Cleanroom-B2',
    status: 'in_progress',
    owner: 'Dr. Chen',
    received: '2024-11-08',
    temperature: '-20°C'
  },
  {
    id: 'SMP-002',
    barcode: 'FILM-2024-002-B3',
    name: 'Thin Film Sample Set',
    type: 'Film',
    location: 'Storage-A3',
    status: 'received',
    owner: 'Dr. Martinez',
    received: '2024-11-09',
    temperature: 'RT'
  },
  {
    id: 'SMP-003',
    barcode: 'DEV-2024-003-C2',
    name: 'Device Prototype Rev3',
    type: 'Device',
    location: 'Testing-Lab1',
    status: 'in_progress',
    owner: 'Dr. Patel',
    received: '2024-11-07',
    temperature: 'RT'
  },
  {
    id: 'SMP-004',
    barcode: 'MAT-2024-004-D1',
    name: 'Reference Material Std',
    type: 'Material',
    location: 'Archive-B1',
    status: 'completed',
    owner: 'Lab Manager',
    received: '2024-10-15',
    temperature: '4°C'
  },
  {
    id: 'SMP-005',
    barcode: 'WAF-2024-005-A2',
    name: 'GaN on Sapphire',
    type: 'Wafer',
    location: 'Cleanroom-A1',
    status: 'received',
    owner: 'Dr. Kim',
    received: '2024-11-09',
    temperature: 'RT'
  }
]

const statusColors = {
  received: 'bg-blue-100 text-blue-800',
  in_progress: 'bg-yellow-100 text-yellow-800',
  completed: 'bg-green-100 text-green-800',
  archived: 'bg-gray-100 text-gray-800'
}

export default function SamplesPage() {
  const [samples, setSamples] = useState(mockSamples)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [showNewSampleForm, setShowNewSampleForm] = useState(false)

  const filteredSamples = samples.filter(sample => {
    const matchesSearch =
      sample.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      sample.barcode.toLowerCase().includes(searchTerm.toLowerCase()) ||
      sample.type.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesStatus = statusFilter === 'all' || sample.status === statusFilter

    return matchesSearch && matchesStatus
  })

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
        <button
          onClick={() => setShowNewSampleForm(true)}
          className="flex items-center gap-2 bg-teal-600 text-white px-4 py-2 rounded-lg hover:bg-teal-700 transition-colors"
        >
          <Plus className="w-5 h-5" />
          New Sample
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Samples</p>
              <p className="text-2xl font-bold text-gray-900">{samples.length}</p>
            </div>
            <Barcode className="w-8 h-8 text-teal-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">In Progress</p>
              <p className="text-2xl font-bold text-yellow-600">
                {samples.filter(s => s.status === 'in_progress').length}
              </p>
            </div>
            <ClipboardCheck className="w-8 h-8 text-yellow-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Completed</p>
              <p className="text-2xl font-bold text-green-600">
                {samples.filter(s => s.status === 'completed').length}
              </p>
            </div>
            <ClipboardCheck className="w-8 h-8 text-green-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">New Arrivals</p>
              <p className="text-2xl font-bold text-blue-600">
                {samples.filter(s => s.status === 'received').length}
              </p>
            </div>
            <ClipboardCheck className="w-8 h-8 text-blue-500" />
          </div>
        </div>
      </div>

      {/* Filters and Search */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Search by name, barcode, or type..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-transparent"
            />
          </div>
          <div className="flex gap-2">
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-transparent"
            >
              <option value="all">All Status</option>
              <option value="received">Received</option>
              <option value="in_progress">In Progress</option>
              <option value="completed">Completed</option>
              <option value="archived">Archived</option>
            </select>
            <button className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
              <Download className="w-5 h-5" />
              Export
            </button>
          </div>
        </div>
      </div>

      {/* Samples Table */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Sample ID / Barcode
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Name & Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Location
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Owner
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Received
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredSamples.map((sample) => (
                <tr key={sample.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center gap-2">
                      <QrCode className="w-5 h-5 text-gray-400" />
                      <div>
                        <div className="text-sm font-medium text-gray-900">{sample.id}</div>
                        <div className="text-sm text-gray-500 font-mono">{sample.barcode}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="text-sm font-medium text-gray-900">{sample.name}</div>
                    <div className="text-sm text-gray-500">{sample.type}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center gap-1 text-sm text-gray-900">
                      <MapPin className="w-4 h-4 text-gray-400" />
                      {sample.location}
                    </div>
                    {sample.temperature && (
                      <div className="text-xs text-gray-500 mt-1">Temp: {sample.temperature}</div>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${statusColors[sample.status]}`}>
                      {sample.status.replace('_', ' ')}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center gap-1 text-sm text-gray-900">
                      <User className="w-4 h-4 text-gray-400" />
                      {sample.owner}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center gap-1 text-sm text-gray-500">
                      <Calendar className="w-4 h-4 text-gray-400" />
                      {sample.received}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <div className="flex gap-2">
                      <button className="text-teal-600 hover:text-teal-900">View</button>
                      <button className="text-blue-600 hover:text-blue-900">Edit</button>
                      <button className="text-gray-600 hover:text-gray-900">QR</button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {filteredSamples.length === 0 && (
          <div className="text-center py-12">
            <Barcode className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No samples found matching your criteria</p>
          </div>
        )}
      </div>

      {/* New Sample Form Modal (placeholder) */}
      {showNewSampleForm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4">
            <h2 className="text-2xl font-bold mb-4">Register New Sample</h2>
            <p className="text-gray-600 mb-4">Sample registration form will be implemented here</p>
            <button
              onClick={() => setShowNewSampleForm(false)}
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
