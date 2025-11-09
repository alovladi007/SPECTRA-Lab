'use client'
import { useState } from 'react'
import { PenTool, Search, Clock, CheckCircle, XCircle, AlertCircle, User, Calendar, FileText } from 'lucide-react'

interface SignatureRequest {
  id: string
  documentId: string
  documentTitle: string
  requestedBy: string
  requestedFor: string
  requestDate: string
  dueDate: string
  status: 'pending' | 'signed' | 'rejected' | 'expired'
  signedDate?: string
  comments?: string
  type: 'approval' | 'review' | 'witness' | 'authorization'
}

const mockSignatures: SignatureRequest[] = [
  {
    id: 'SIG-001',
    documentId: 'ELN-2024-003',
    documentTitle: 'Device Failure Analysis - Prototype Rev3',
    requestedBy: 'Dr. Patel',
    requestedFor: 'Dr. Chen',
    requestDate: '2024-11-09 10:00',
    dueDate: '2024-11-12',
    status: 'pending',
    type: 'review'
  },
  {
    id: 'SIG-002',
    documentId: 'SOP-2024-015',
    documentTitle: 'Updated Safety Procedures - Chemical Storage',
    requestedBy: 'Lab Manager',
    requestedFor: 'Dr. Martinez',
    requestDate: '2024-11-08 14:00',
    dueDate: '2024-11-10',
    status: 'signed',
    signedDate: '2024-11-08 16:30',
    type: 'approval',
    comments: 'Approved with noted safety concerns addressed'
  },
  {
    id: 'SIG-003',
    documentId: 'ELN-2024-001',
    documentTitle: 'Silicon Wafer Electrical Characterization',
    requestedBy: 'Dr. Chen',
    requestedFor: 'Dr. Kim',
    requestDate: '2024-11-07 09:00',
    dueDate: '2024-11-09',
    status: 'signed',
    signedDate: '2024-11-08 11:00',
    type: 'witness',
    comments: 'Witnessed measurements as described'
  },
  {
    id: 'SIG-004',
    documentId: 'AUTH-2024-007',
    documentTitle: 'Equipment Authorization - New SEM',
    requestedBy: 'Facilities',
    requestedFor: 'Dr. Patel',
    requestDate: '2024-11-05 13:00',
    dueDate: '2024-11-08',
    status: 'expired',
    type: 'authorization'
  },
  {
    id: 'SIG-005',
    documentId: 'ELN-2024-002',
    documentTitle: 'Thin Film Deposition Protocol v2.1',
    requestedBy: 'Dr. Martinez',
    requestedFor: 'Lab Manager',
    requestDate: '2024-11-09 08:00',
    dueDate: '2024-11-11',
    status: 'pending',
    type: 'approval'
  }
]

const statusConfig = {
  pending: { color: 'bg-yellow-100 text-yellow-800', icon: Clock },
  signed: { color: 'bg-green-100 text-green-800', icon: CheckCircle },
  rejected: { color: 'bg-red-100 text-red-800', icon: XCircle },
  expired: { color: 'bg-gray-100 text-gray-800', icon: AlertCircle }
}

const typeColors = {
  approval: 'bg-blue-100 text-blue-800',
  review: 'bg-purple-100 text-purple-800',
  witness: 'bg-green-100 text-green-800',
  authorization: 'bg-orange-100 text-orange-800'
}

export default function SignaturesPage() {
  const [signatures, setSignatures] = useState(mockSignatures)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')

  const filteredSignatures = signatures.filter(sig => {
    const matchesSearch =
      sig.documentTitle.toLowerCase().includes(searchTerm.toLowerCase()) ||
      sig.requestedBy.toLowerCase().includes(searchTerm.toLowerCase()) ||
      sig.requestedFor.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesStatus = statusFilter === 'all' || sig.status === statusFilter

    return matchesSearch && matchesStatus
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 bg-gradient-to-br from-rose-500 to-pink-500 rounded-xl flex items-center justify-center">
          <PenTool className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">E-Signatures</h1>
          <p className="text-gray-600 mt-1">Digital signature workflow for approvals and reviews</p>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Requests</p>
              <p className="text-2xl font-bold text-gray-900">{signatures.length}</p>
            </div>
            <PenTool className="w-8 h-8 text-rose-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Pending</p>
              <p className="text-2xl font-bold text-yellow-600">
                {signatures.filter(s => s.status === 'pending').length}
              </p>
            </div>
            <Clock className="w-8 h-8 text-yellow-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Signed</p>
              <p className="text-2xl font-bold text-green-600">
                {signatures.filter(s => s.status === 'signed').length}
              </p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Expired</p>
              <p className="text-2xl font-bold text-gray-600">
                {signatures.filter(s => s.status === 'expired').length}
              </p>
            </div>
            <AlertCircle className="w-8 h-8 text-gray-500" />
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
              placeholder="Search by document, requester, or signer..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-rose-500 focus:border-transparent"
            />
          </div>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-rose-500 focus:border-transparent"
          >
            <option value="all">All Status</option>
            <option value="pending">Pending</option>
            <option value="signed">Signed</option>
            <option value="rejected">Rejected</option>
            <option value="expired">Expired</option>
          </select>
        </div>
      </div>

      {/* Signature Requests */}
      <div className="space-y-4">
        {filteredSignatures.map((sig) => {
          const StatusIcon = statusConfig[sig.status].icon
          return (
            <div key={sig.id} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="text-lg font-semibold text-gray-900">{sig.documentTitle}</h3>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${typeColors[sig.type]}`}>
                      {sig.type}
                    </span>
                  </div>
                  <p className="text-sm text-gray-500 mb-3">
                    Signature ID: {sig.id} • Document: {sig.documentId}
                  </p>
                </div>
                <span className={`px-3 py-1 rounded-full text-sm font-medium flex items-center gap-1 ${statusConfig[sig.status].color}`}>
                  <StatusIcon className="w-4 h-4" />
                  {sig.status}
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-xs text-gray-500 uppercase font-medium mb-2">Requested By</p>
                  <div className="flex items-center gap-2 text-sm">
                    <User className="w-4 h-4 text-gray-400" />
                    <span className="font-medium text-gray-900">{sig.requestedBy}</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-gray-600 mt-2">
                    <Calendar className="w-4 h-4 text-gray-400" />
                    {sig.requestDate}
                  </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-xs text-gray-500 uppercase font-medium mb-2">Requested For</p>
                  <div className="flex items-center gap-2 text-sm">
                    <User className="w-4 h-4 text-gray-400" />
                    <span className="font-medium text-gray-900">{sig.requestedFor}</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-gray-600 mt-2">
                    <Calendar className="w-4 h-4 text-gray-400" />
                    Due: {sig.dueDate}
                  </div>
                </div>
              </div>

              {sig.signedDate && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-3 mb-4">
                  <div className="flex items-center gap-2 text-sm text-green-800">
                    <CheckCircle className="w-4 h-4" />
                    <span className="font-medium">Signed on {sig.signedDate}</span>
                  </div>
                  {sig.comments && (
                    <p className="text-sm text-green-700 mt-2">{sig.comments}</p>
                  )}
                </div>
              )}

              <div className="flex gap-2 pt-4 border-t border-gray-200">
                <button className="text-rose-600 hover:text-rose-900 text-sm font-medium">
                  View Document
                </button>
                {sig.status === 'pending' && (
                  <>
                    <button className="text-green-600 hover:text-green-900 text-sm font-medium">
                      Sign Document
                    </button>
                    <button className="text-red-600 hover:text-red-900 text-sm font-medium">
                      Reject
                    </button>
                  </>
                )}
                <button className="text-blue-600 hover:text-blue-900 text-sm font-medium">
                  Audit Trail
                </button>
              </div>
            </div>
          )
        })}

        {filteredSignatures.length === 0 && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
            <PenTool className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No signature requests found matching your criteria</p>
          </div>
        )}
      </div>
    </div>
  )
}
