'use client'
import { useState, useEffect } from 'react'
import { FileText, Plus, Search, GitBranch, CheckCircle, Clock, User, Calendar, Download, Trash2 } from 'lucide-react'
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

interface SOP {
  id: string
  title: string
  category: string
  version: string
  status: 'draft' | 'in_review' | 'active' | 'archived'
  author: string
  effectiveDate: string
  reviewDate: string
  approver?: string
  description: string
  tags: string[]
}

interface NewSOPForm {
  title: string
  category: string
  description: string
  tags: string
}

const generateMockSOPs = (): SOP[] => [
  {
    id: 'SOP-2024-015',
    title: 'Chemical Storage Safety Procedures',
    category: 'Safety',
    version: 'v2.1',
    status: 'active',
    author: 'Lab Manager',
    effectiveDate: '2024-11-01',
    reviewDate: '2025-05-01',
    approver: 'Dr. Martinez',
    description: 'Standard operating procedure for safe storage of hazardous chemicals in lab facilities',
    tags: ['safety', 'chemicals', 'storage']
  },
  {
    id: 'SOP-2024-016',
    title: 'Wafer Cleaning Protocol for Characterization',
    category: 'Sample Preparation',
    version: 'v1.3',
    status: 'active',
    author: 'Dr. Chen',
    effectiveDate: '2024-10-15',
    reviewDate: '2025-04-15',
    approver: 'Lab Manager',
    description: 'Step-by-step procedure for cleaning silicon wafers before electrical characterization',
    tags: ['wafer', 'cleaning', 'preparation']
  },
  {
    id: 'SOP-2024-017',
    title: 'SEM Operation and Maintenance',
    category: 'Equipment',
    version: 'v3.0',
    status: 'in_review',
    author: 'Dr. Patel',
    effectiveDate: '',
    reviewDate: '2024-11-15',
    description: 'Operating procedures and routine maintenance schedule for Scanning Electron Microscope',
    tags: ['SEM', 'equipment', 'maintenance']
  },
  {
    id: 'SOP-2024-018',
    title: 'Data Archival and Backup Procedures',
    category: 'Data Management',
    version: 'v1.0',
    status: 'draft',
    author: 'IT Manager',
    effectiveDate: '',
    reviewDate: '2024-11-20',
    description: 'Guidelines for archiving experimental data and maintaining secure backups',
    tags: ['data', 'backup', 'archival']
  },
  {
    id: 'SOP-2023-089',
    title: 'Legacy Thin Film Deposition Process',
    category: 'Process',
    version: 'v1.5',
    status: 'archived',
    author: 'Dr. Martinez',
    effectiveDate: '2023-01-01',
    reviewDate: '2024-01-01',
    approver: 'Lab Manager',
    description: 'Archived version replaced by SOP-2024-002',
    tags: ['deprecated', 'thin-film', 'deposition']
  },
  {
    id: 'SOP-2024-019',
    title: 'Cleanroom Gowning Procedure',
    category: 'Safety',
    version: 'v1.2',
    status: 'active',
    author: 'Safety Officer',
    effectiveDate: '2024-09-01',
    reviewDate: '2025-03-01',
    approver: 'Lab Manager',
    description: 'Proper procedure for donning cleanroom garments and personal protective equipment',
    tags: ['safety', 'cleanroom', 'PPE']
  },
  {
    id: 'SOP-2024-020',
    title: 'Emergency Spill Response Protocol',
    category: 'Safety',
    version: 'v2.0',
    status: 'active',
    author: 'Safety Officer',
    effectiveDate: '2024-10-01',
    reviewDate: '2025-04-01',
    approver: 'Lab Manager',
    description: 'Immediate response procedures for chemical spills and emergency containment',
    tags: ['safety', 'emergency', 'spill']
  },
  {
    id: 'SOP-2024-021',
    title: 'Photolithography Best Practices',
    category: 'Process',
    version: 'v1.4',
    status: 'in_review',
    author: 'Dr. Wilson',
    effectiveDate: '',
    reviewDate: '2024-11-18',
    description: 'Best practices and troubleshooting guide for photolithography processes',
    tags: ['photolithography', 'process', 'fabrication']
  }
]

const statusConfig = {
  draft: { color: 'bg-gray-100 text-gray-800', icon: FileText, label: 'Draft' },
  in_review: { color: 'bg-yellow-100 text-yellow-800', icon: Clock, label: 'In Review' },
  active: { color: 'bg-green-100 text-green-800', icon: CheckCircle, label: 'Active' },
  archived: { color: 'bg-red-100 text-red-800', icon: FileText, label: 'Archived' }
}

const categoryColors: { [key: string]: string } = {
  'Safety': 'bg-red-100 text-red-800',
  'Sample Preparation': 'bg-blue-100 text-blue-800',
  'Equipment': 'bg-purple-100 text-purple-800',
  'Process': 'bg-green-100 text-green-800',
  'Data Management': 'bg-orange-100 text-orange-800'
}

export default function SOPsPage() {
  const [sops, setSops] = useState<SOP[]>([])
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [showNewSOP, setShowNewSOP] = useState(false)
  const [selectedSOP, setSelectedSOP] = useState<SOP | null>(null)
  const [showDetailsDialog, setShowDetailsDialog] = useState(false)

  const [newSOP, setNewSOP] = useState<NewSOPForm>({
    title: '',
    category: '',
    description: '',
    tags: ''
  })

  // Generate mock data on client side to prevent hydration errors
  useEffect(() => {
    setSops(generateMockSOPs())
  }, [])

  const filteredSOPs = sops.filter(sop => {
    const matchesSearch =
      sop.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      sop.category.toLowerCase().includes(searchTerm.toLowerCase()) ||
      sop.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))

    const matchesStatus = statusFilter === 'all' || sop.status === statusFilter

    return matchesSearch && matchesStatus
  })

  const handleCreateSOP = () => {
    if (!newSOP.title || !newSOP.category || !newSOP.description) {
      alert('Please fill in all required fields')
      return
    }

    const currentUser = 'Current User'
    const sop: SOP = {
      id: `SOP-2024-${String(sops.length + 1).padStart(3, '0')}`,
      title: newSOP.title,
      category: newSOP.category,
      version: 'v1.0',
      status: 'draft',
      author: currentUser,
      effectiveDate: '',
      reviewDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      description: newSOP.description,
      tags: newSOP.tags.split(',').map(t => t.trim()).filter(t => t)
    }

    setSops([sop, ...sops])
    setShowNewSOP(false)
    setNewSOP({
      title: '',
      category: '',
      description: '',
      tags: ''
    })
  }

  const handleSubmitForReview = (sopId: string) => {
    setSops(sops.map(sop => {
      if (sop.id === sopId && sop.status === 'draft') {
        return { ...sop, status: 'in_review' as const }
      }
      return sop
    }))
  }

  const handleApprove = (sopId: string) => {
    setSops(sops.map(sop => {
      if (sop.id === sopId && sop.status === 'in_review') {
        return {
          ...sop,
          status: 'active' as const,
          effectiveDate: new Date().toISOString().split('T')[0],
          approver: 'Current User'
        }
      }
      return sop
    }))
  }

  const handleArchive = (sopId: string) => {
    if (confirm('Are you sure you want to archive this SOP?')) {
      setSops(sops.map(sop => {
        if (sop.id === sopId) {
          return { ...sop, status: 'archived' as const }
        }
        return sop
      }))
    }
  }

  const handleDelete = (sopId: string) => {
    if (confirm('Are you sure you want to delete this SOP? This action cannot be undone.')) {
      setSops(sops.filter(sop => sop.id !== sopId))
    }
  }

  const handleViewDetails = (sop: SOP) => {
    setSelectedSOP(sop)
    setShowDetailsDialog(true)
  }

  const stats = {
    total: sops.length,
    active: sops.filter(s => s.status === 'active').length,
    in_review: sops.filter(s => s.status === 'in_review').length,
    draft: sops.filter(s => s.status === 'draft').length
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-emerald-500 to-teal-500 rounded-xl flex items-center justify-center">
            <FileText className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">SOP Management</h1>
            <p className="text-gray-600 mt-1">Standard Operating Procedures with version control</p>
          </div>
        </div>
        <Button
          onClick={() => setShowNewSOP(true)}
          className="flex items-center gap-2"
        >
          <Plus className="w-5 h-5" />
          New SOP
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total SOPs</p>
              <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
            </div>
            <FileText className="w-8 h-8 text-emerald-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Active</p>
              <p className="text-2xl font-bold text-green-600">{stats.active}</p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">In Review</p>
              <p className="text-2xl font-bold text-yellow-600">{stats.in_review}</p>
            </div>
            <Clock className="w-8 h-8 text-yellow-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Draft</p>
              <p className="text-2xl font-bold text-gray-600">{stats.draft}</p>
            </div>
            <FileText className="w-8 h-8 text-gray-500" />
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
              placeholder="Search by title, category, or tags..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
          <div className="flex gap-2">
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="All Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="draft">Draft</SelectItem>
                <SelectItem value="in_review">In Review</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="archived">Archived</SelectItem>
              </SelectContent>
            </Select>
            <Button variant="outline" className="flex items-center gap-2">
              <Download className="w-4 h-4" />
              Export
            </Button>
          </div>
        </div>
      </Card>

      {/* SOPs List */}
      <div className="space-y-4">
        {filteredSOPs.map((sop) => {
          const StatusIcon = statusConfig[sop.status].icon
          return (
            <Card key={sop.id} className="p-6 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2 flex-wrap">
                    <h3 className="text-lg font-semibold text-gray-900">{sop.title}</h3>
                    <Badge className={categoryColors[sop.category] || 'bg-gray-100 text-gray-800'}>
                      {sop.category}
                    </Badge>
                    <Badge className={statusConfig[sop.status].color}>
                      <StatusIcon className="w-3 h-3 mr-1" />
                      {statusConfig[sop.status].label}
                    </Badge>
                  </div>
                  <p className="text-sm text-gray-600 mb-3">{sop.description}</p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-3">
                <div className="flex items-center gap-2 text-sm">
                  <GitBranch className="w-4 h-4 text-gray-400" />
                  <span className="font-mono font-medium text-gray-900">{sop.version}</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-600">
                  <User className="w-4 h-4 text-gray-400" />
                  {sop.author}
                </div>
                {sop.effectiveDate && (
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <Calendar className="w-4 h-4 text-gray-400" />
                    Effective: {sop.effectiveDate}
                  </div>
                )}
                <div className="flex items-center gap-2 text-sm text-gray-600">
                  <Calendar className="w-4 h-4 text-gray-400" />
                  Review: {sop.reviewDate}
                </div>
              </div>

              {sop.approver && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-2 mb-3">
                  <p className="text-sm text-green-800">
                    <CheckCircle className="w-4 h-4 inline mr-1" />
                    Approved by {sop.approver}
                  </p>
                </div>
              )}

              <div className="flex flex-wrap gap-2 mb-3">
                {sop.tags.map(tag => (
                  <span key={tag} className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full">
                    #{tag}
                  </span>
                ))}
              </div>

              <div className="flex gap-2 pt-3 border-t border-gray-200">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleViewDetails(sop)}
                  className="text-emerald-600 hover:text-emerald-900"
                >
                  View SOP
                </Button>
                {sop.status !== 'archived' && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-blue-600 hover:text-blue-900"
                  >
                    Edit
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-purple-600 hover:text-purple-900"
                >
                  Version History
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-gray-600 hover:text-gray-900"
                >
                  Download PDF
                </Button>
                {sop.status === 'draft' && (
                  <>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleSubmitForReview(sop.id)}
                      className="text-green-600 hover:text-green-900"
                    >
                      Submit for Review
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDelete(sop.id)}
                      className="text-red-600 hover:text-red-900"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </>
                )}
                {sop.status === 'in_review' && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleApprove(sop.id)}
                    className="text-green-600 hover:text-green-900"
                  >
                    Approve
                  </Button>
                )}
                {(sop.status === 'active' || sop.status === 'in_review') && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleArchive(sop.id)}
                    className="text-orange-600 hover:text-orange-900"
                  >
                    Archive
                  </Button>
                )}
              </div>
            </Card>
          )
        })}

        {filteredSOPs.length === 0 && (
          <Card className="p-12 text-center">
            <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No SOPs found matching your criteria</p>
          </Card>
        )}
      </div>

      {/* New SOP Dialog */}
      <Dialog open={showNewSOP} onOpenChange={setShowNewSOP}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Create New SOP</DialogTitle>
            <DialogDescription>
              Create a new Standard Operating Procedure
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Title *
              </label>
              <Input
                value={newSOP.title}
                onChange={(e) => setNewSOP({ ...newSOP, title: e.target.value })}
                placeholder="e.g., Chemical Storage Procedure"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Category *
              </label>
              <Select
                value={newSOP.category}
                onValueChange={(value) => setNewSOP({ ...newSOP, category: value })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select category" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Safety">Safety</SelectItem>
                  <SelectItem value="Sample Preparation">Sample Preparation</SelectItem>
                  <SelectItem value="Equipment">Equipment</SelectItem>
                  <SelectItem value="Process">Process</SelectItem>
                  <SelectItem value="Data Management">Data Management</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Description *
              </label>
              <Textarea
                value={newSOP.description}
                onChange={(e) => setNewSOP({ ...newSOP, description: e.target.value })}
                placeholder="Describe the purpose and scope of this SOP..."
                className="min-h-[120px]"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Tags
              </label>
              <Input
                value={newSOP.tags}
                onChange={(e) => setNewSOP({ ...newSOP, tags: e.target.value })}
                placeholder="e.g., safety, chemicals, storage (comma-separated)"
              />
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowNewSOP(false)}
            >
              Cancel
            </Button>
            <Button onClick={handleCreateSOP}>
              Create SOP
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Details Dialog */}
      <Dialog open={showDetailsDialog} onOpenChange={setShowDetailsDialog}>
        <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>SOP Details</DialogTitle>
            <DialogDescription>
              {selectedSOP?.id}
            </DialogDescription>
          </DialogHeader>

          {selectedSOP && (
            <div className="space-y-4">
              <div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">{selectedSOP.title}</h3>
                <div className="flex flex-wrap gap-2 mb-4">
                  <Badge className={categoryColors[selectedSOP.category]}>
                    {selectedSOP.category}
                  </Badge>
                  <Badge className={statusConfig[selectedSOP.status].color}>
                    {statusConfig[selectedSOP.status].label}
                  </Badge>
                  <Badge variant="outline">{selectedSOP.version}</Badge>
                </div>
              </div>

              <div className="border-t border-gray-200 pt-4">
                <p className="text-sm font-medium text-gray-500 mb-2">Description</p>
                <p className="text-sm text-gray-900">{selectedSOP.description}</p>
              </div>

              <div className="border-t border-gray-200 pt-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">Author</p>
                    <p className="text-base text-gray-900">{selectedSOP.author}</p>
                  </div>
                  {selectedSOP.approver && (
                    <div>
                      <p className="text-sm font-medium text-gray-500">Approver</p>
                      <p className="text-base text-gray-900">{selectedSOP.approver}</p>
                    </div>
                  )}
                  {selectedSOP.effectiveDate && (
                    <div>
                      <p className="text-sm font-medium text-gray-500">Effective Date</p>
                      <p className="text-base text-gray-900">{selectedSOP.effectiveDate}</p>
                    </div>
                  )}
                  <div>
                    <p className="text-sm font-medium text-gray-500">Review Date</p>
                    <p className="text-base text-gray-900">{selectedSOP.reviewDate}</p>
                  </div>
                </div>
              </div>

              <div className="border-t border-gray-200 pt-4">
                <p className="text-sm font-medium text-gray-500 mb-2">Tags</p>
                <div className="flex flex-wrap gap-2">
                  {selectedSOP.tags.map(tag => (
                    <span key={tag} className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full">
                      #{tag}
                    </span>
                  ))}
                </div>
              </div>
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
