'use client'
import { useState, useEffect } from 'react'
import { PenTool, Search, Clock, CheckCircle, XCircle, AlertCircle, User, Calendar, FileText, Plus } from 'lucide-react'
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

interface NewRequestForm {
  documentId: string
  documentTitle: string
  requestedFor: string
  dueDate: string
  type: 'approval' | 'review' | 'witness' | 'authorization'
}

const generateMockSignatures = (): SignatureRequest[] => [
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
  },
  {
    id: 'SIG-006',
    documentId: 'ELN-2024-005',
    documentTitle: 'GaN Material Characterization Report',
    requestedBy: 'Dr. Kim',
    requestedFor: 'Dr. Wilson',
    requestDate: '2024-11-06 15:00',
    dueDate: '2024-11-10',
    status: 'signed',
    signedDate: '2024-11-07 10:30',
    type: 'review',
    comments: 'Results verified and approved'
  },
  {
    id: 'SIG-007',
    documentId: 'SOP-2024-018',
    documentTitle: 'Cleanroom Entry Procedures Update',
    requestedBy: 'Safety Officer',
    requestedFor: 'All Staff',
    requestDate: '2024-11-09 11:00',
    dueDate: '2024-11-15',
    status: 'pending',
    type: 'approval'
  },
  {
    id: 'SIG-008',
    documentId: 'AUTH-2024-009',
    documentTitle: 'Hazardous Material Handling Authorization',
    requestedBy: 'Lab Manager',
    requestedFor: 'Dr. Lee',
    requestDate: '2024-11-04 09:00',
    dueDate: '2024-11-07',
    status: 'rejected',
    comments: 'Additional safety training required before authorization',
    type: 'authorization'
  },
  {
    id: 'SIG-009',
    documentId: 'ELN-2024-008',
    documentTitle: 'AFM Surface Analysis Protocol',
    requestedBy: 'Dr. Lee',
    requestedFor: 'Dr. Brown',
    requestDate: '2024-11-08 13:00',
    dueDate: '2024-11-12',
    status: 'pending',
    type: 'witness'
  },
  {
    id: 'SIG-010',
    documentId: 'ELN-2024-009',
    documentTitle: 'Temperature Cycling Test Results',
    requestedBy: 'Dr. Brown',
    requestedFor: 'Quality Manager',
    requestDate: '2024-11-07 16:00',
    dueDate: '2024-11-11',
    status: 'signed',
    signedDate: '2024-11-08 14:00',
    type: 'approval',
    comments: 'Test results meet all quality standards'
  }
]

const statusConfig = {
  pending: { color: 'bg-yellow-100 text-yellow-800', icon: Clock, label: 'Pending' },
  signed: { color: 'bg-green-100 text-green-800', icon: CheckCircle, label: 'Signed' },
  rejected: { color: 'bg-red-100 text-red-800', icon: XCircle, label: 'Rejected' },
  expired: { color: 'bg-gray-100 text-gray-800', icon: AlertCircle, label: 'Expired' }
}

const typeColors = {
  approval: 'bg-blue-100 text-blue-800',
  review: 'bg-purple-100 text-purple-800',
  witness: 'bg-green-100 text-green-800',
  authorization: 'bg-orange-100 text-orange-800'
}

export default function SignaturesPage() {
  const [signatures, setSignatures] = useState<SignatureRequest[]>([])
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [showNewRequest, setShowNewRequest] = useState(false)
  const [showSignDialog, setShowSignDialog] = useState(false)
  const [showRejectDialog, setShowRejectDialog] = useState(false)
  const [showDetailsDialog, setShowDetailsDialog] = useState(false)
  const [selectedSignature, setSelectedSignature] = useState<SignatureRequest | null>(null)
  const [signComments, setSignComments] = useState('')
  const [rejectReason, setRejectReason] = useState('')

  const [newRequest, setNewRequest] = useState<NewRequestForm>({
    documentId: '',
    documentTitle: '',
    requestedFor: '',
    dueDate: '',
    type: 'approval'
  })

  // Generate mock data on client side to prevent hydration errors
  useEffect(() => {
    setSignatures(generateMockSignatures())
  }, [])

  const filteredSignatures = signatures.filter(sig => {
    const matchesSearch =
      sig.documentTitle.toLowerCase().includes(searchTerm.toLowerCase()) ||
      sig.requestedBy.toLowerCase().includes(searchTerm.toLowerCase()) ||
      sig.requestedFor.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesStatus = statusFilter === 'all' || sig.status === statusFilter

    return matchesSearch && matchesStatus
  })

  const handleCreateRequest = () => {
    if (!newRequest.documentId || !newRequest.documentTitle || !newRequest.requestedFor || !newRequest.dueDate) {
      alert('Please fill in all required fields')
      return
    }

    const currentUser = 'Current User' // In real app, get from auth context
    const request: SignatureRequest = {
      id: `SIG-${String(signatures.length + 1).padStart(3, '0')}`,
      documentId: newRequest.documentId,
      documentTitle: newRequest.documentTitle,
      requestedBy: currentUser,
      requestedFor: newRequest.requestedFor,
      requestDate: new Date().toISOString().slice(0, 16).replace('T', ' '),
      dueDate: newRequest.dueDate,
      status: 'pending',
      type: newRequest.type
    }

    setSignatures([request, ...signatures])
    setShowNewRequest(false)
    setNewRequest({
      documentId: '',
      documentTitle: '',
      requestedFor: '',
      dueDate: '',
      type: 'approval'
    })
  }

  const handleSignDocument = () => {
    if (!selectedSignature) return

    setSignatures(signatures.map(sig => {
      if (sig.id === selectedSignature.id && sig.status === 'pending') {
        return {
          ...sig,
          status: 'signed' as const,
          signedDate: new Date().toISOString().slice(0, 16).replace('T', ' '),
          comments: signComments || undefined
        }
      }
      return sig
    }))

    setShowSignDialog(false)
    setSelectedSignature(null)
    setSignComments('')
  }

  const handleRejectDocument = () => {
    if (!selectedSignature || !rejectReason) {
      alert('Please provide a reason for rejection')
      return
    }

    setSignatures(signatures.map(sig => {
      if (sig.id === selectedSignature.id && sig.status === 'pending') {
        return {
          ...sig,
          status: 'rejected' as const,
          comments: rejectReason
        }
      }
      return sig
    }))

    setShowRejectDialog(false)
    setSelectedSignature(null)
    setRejectReason('')
  }

  const handleViewDetails = (signature: SignatureRequest) => {
    setSelectedSignature(signature)
    setShowDetailsDialog(true)
  }

  const stats = {
    total: signatures.length,
    pending: signatures.filter(s => s.status === 'pending').length,
    signed: signatures.filter(s => s.status === 'signed').length,
    expired: signatures.filter(s => s.status === 'expired').length
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-rose-500 to-pink-500 rounded-xl flex items-center justify-center">
            <PenTool className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">E-Signatures</h1>
            <p className="text-gray-600 mt-1">Digital signature workflow for approvals and reviews</p>
          </div>
        </div>
        <Button
          onClick={() => setShowNewRequest(true)}
          className="flex items-center gap-2"
        >
          <Plus className="w-5 h-5" />
          New Request
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Requests</p>
              <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
            </div>
            <PenTool className="w-8 h-8 text-rose-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Pending</p>
              <p className="text-2xl font-bold text-yellow-600">{stats.pending}</p>
            </div>
            <Clock className="w-8 h-8 text-yellow-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Signed</p>
              <p className="text-2xl font-bold text-green-600">{stats.signed}</p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Expired</p>
              <p className="text-2xl font-bold text-gray-600">{stats.expired}</p>
            </div>
            <AlertCircle className="w-8 h-8 text-gray-500" />
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
              placeholder="Search by document, requester, or signer..."
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
              <SelectItem value="signed">Signed</SelectItem>
              <SelectItem value="rejected">Rejected</SelectItem>
              <SelectItem value="expired">Expired</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </Card>

      {/* Signature Requests */}
      <div className="space-y-4">
        {filteredSignatures.map((sig) => {
          const StatusIcon = statusConfig[sig.status].icon
          return (
            <Card key={sig.id} className="p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2 flex-wrap">
                    <h3 className="text-lg font-semibold text-gray-900">{sig.documentTitle}</h3>
                    <Badge className={typeColors[sig.type]}>
                      {sig.type}
                    </Badge>
                    <Badge className={statusConfig[sig.status].color}>
                      <StatusIcon className="w-3 h-3 mr-1" />
                      {statusConfig[sig.status].label}
                    </Badge>
                  </div>
                  <p className="text-sm text-gray-500 mb-3">
                    Signature ID: {sig.id} • Document: {sig.documentId}
                  </p>
                </div>
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

              {sig.status === 'rejected' && sig.comments && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
                  <div className="flex items-center gap-2 text-sm text-red-800">
                    <XCircle className="w-4 h-4" />
                    <span className="font-medium">Rejected</span>
                  </div>
                  <p className="text-sm text-red-700 mt-2">{sig.comments}</p>
                </div>
              )}

              <div className="flex gap-2 pt-4 border-t border-gray-200">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleViewDetails(sig)}
                  className="text-rose-600 hover:text-rose-900"
                >
                  View Document
                </Button>
                {sig.status === 'pending' && (
                  <>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        setSelectedSignature(sig)
                        setShowSignDialog(true)
                      }}
                      className="text-green-600 hover:text-green-900"
                    >
                      Sign Document
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        setSelectedSignature(sig)
                        setShowRejectDialog(true)
                      }}
                      className="text-red-600 hover:text-red-900"
                    >
                      Reject
                    </Button>
                  </>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-blue-600 hover:text-blue-900"
                >
                  Audit Trail
                </Button>
              </div>
            </Card>
          )
        })}

        {filteredSignatures.length === 0 && (
          <Card className="p-12 text-center">
            <PenTool className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No signature requests found matching your criteria</p>
          </Card>
        )}
      </div>

      {/* New Request Dialog */}
      <Dialog open={showNewRequest} onOpenChange={setShowNewRequest}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>New Signature Request</DialogTitle>
            <DialogDescription>
              Request a digital signature for a document
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Document ID *
              </label>
              <Input
                value={newRequest.documentId}
                onChange={(e) => setNewRequest({ ...newRequest, documentId: e.target.value })}
                placeholder="e.g., ELN-2024-001"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Document Title *
              </label>
              <Input
                value={newRequest.documentTitle}
                onChange={(e) => setNewRequest({ ...newRequest, documentTitle: e.target.value })}
                placeholder="e.g., Lab Safety Protocol Update"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Signature Type *
              </label>
              <Select
                value={newRequest.type}
                onValueChange={(value: any) => setNewRequest({ ...newRequest, type: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="approval">Approval</SelectItem>
                  <SelectItem value="review">Review</SelectItem>
                  <SelectItem value="witness">Witness</SelectItem>
                  <SelectItem value="authorization">Authorization</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Request Signature From *
              </label>
              <Input
                value={newRequest.requestedFor}
                onChange={(e) => setNewRequest({ ...newRequest, requestedFor: e.target.value })}
                placeholder="e.g., Dr. Smith"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Due Date *
              </label>
              <Input
                type="date"
                value={newRequest.dueDate}
                onChange={(e) => setNewRequest({ ...newRequest, dueDate: e.target.value })}
              />
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowNewRequest(false)}
            >
              Cancel
            </Button>
            <Button onClick={handleCreateRequest}>
              Create Request
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Sign Document Dialog */}
      <Dialog open={showSignDialog} onOpenChange={setShowSignDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Sign Document</DialogTitle>
            <DialogDescription>
              {selectedSignature?.documentTitle}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <p className="text-sm text-blue-800">
                By signing this document, you confirm that you have reviewed it and agree with its contents.
                This signature will be recorded with a timestamp and your user credentials.
              </p>
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Comments (Optional)
              </label>
              <Textarea
                value={signComments}
                onChange={(e) => setSignComments(e.target.value)}
                placeholder="Add any comments or notes about this signature..."
                className="min-h-[100px]"
              />
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setShowSignDialog(false)
                setSignComments('')
                setSelectedSignature(null)
              }}
            >
              Cancel
            </Button>
            <Button onClick={handleSignDocument} className="bg-green-600 hover:bg-green-700">
              <PenTool className="w-4 h-4 mr-2" />
              Sign Document
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Reject Document Dialog */}
      <Dialog open={showRejectDialog} onOpenChange={setShowRejectDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Reject Document</DialogTitle>
            <DialogDescription>
              {selectedSignature?.documentTitle}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-sm text-red-800">
                Please provide a reason for rejecting this document. This will be recorded in the audit trail.
              </p>
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Reason for Rejection *
              </label>
              <Textarea
                value={rejectReason}
                onChange={(e) => setRejectReason(e.target.value)}
                placeholder="Explain why you are rejecting this document..."
                className="min-h-[100px]"
              />
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setShowRejectDialog(false)
                setRejectReason('')
                setSelectedSignature(null)
              }}
            >
              Cancel
            </Button>
            <Button onClick={handleRejectDocument} className="bg-red-600 hover:bg-red-700">
              <XCircle className="w-4 h-4 mr-2" />
              Reject Document
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Details Dialog */}
      <Dialog open={showDetailsDialog} onOpenChange={setShowDetailsDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Signature Request Details</DialogTitle>
            <DialogDescription>
              Request ID: {selectedSignature?.id}
            </DialogDescription>
          </DialogHeader>

          {selectedSignature && (
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-bold text-gray-900 mb-2">{selectedSignature.documentTitle}</h3>
                <div className="flex flex-wrap gap-2 mb-4">
                  <Badge className={typeColors[selectedSignature.type]}>
                    {selectedSignature.type}
                  </Badge>
                  <Badge className={statusConfig[selectedSignature.status].color}>
                    {statusConfig[selectedSignature.status].label}
                  </Badge>
                </div>
              </div>

              <div className="border-t border-gray-200 pt-4">
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">Document ID</p>
                    <p className="text-base text-gray-900">{selectedSignature.documentId}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Request Date</p>
                    <p className="text-base text-gray-900">{selectedSignature.requestDate}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Requested By</p>
                    <p className="text-base text-gray-900">{selectedSignature.requestedBy}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Requested For</p>
                    <p className="text-base text-gray-900">{selectedSignature.requestedFor}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Due Date</p>
                    <p className="text-base text-gray-900">{selectedSignature.dueDate}</p>
                  </div>
                  {selectedSignature.signedDate && (
                    <div>
                      <p className="text-sm font-medium text-gray-500">Signed Date</p>
                      <p className="text-base text-gray-900">{selectedSignature.signedDate}</p>
                    </div>
                  )}
                </div>
              </div>

              {selectedSignature.comments && (
                <div className="border-t border-gray-200 pt-4">
                  <p className="text-sm font-medium text-gray-500 mb-2">Comments</p>
                  <div className="bg-gray-50 rounded-lg p-3">
                    <p className="text-sm text-gray-900">{selectedSignature.comments}</p>
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
