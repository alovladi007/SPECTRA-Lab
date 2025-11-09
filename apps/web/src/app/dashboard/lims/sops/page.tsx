'use client'
import { useState } from 'react'
import { FileText, Plus, Search, GitBranch, CheckCircle, Clock, User, Calendar, Download } from 'lucide-react'

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

const mockSOPs: SOP[] = [
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
  }
]

const statusConfig = {
  draft: { color: 'bg-gray-100 text-gray-800', icon: FileText },
  in_review: { color: 'bg-yellow-100 text-yellow-800', icon: Clock },
  active: { color: 'bg-green-100 text-green-800', icon: CheckCircle },
  archived: { color: 'bg-red-100 text-red-800', icon: FileText }
}

const categoryColors: { [key: string]: string } = {
  'Safety': 'bg-red-100 text-red-800',
  'Sample Preparation': 'bg-blue-100 text-blue-800',
  'Equipment': 'bg-purple-100 text-purple-800',
  'Process': 'bg-green-100 text-green-800',
  'Data Management': 'bg-orange-100 text-orange-800'
}

export default function SOPsPage() {
  const [sops, setSops] = useState(mockSOPs)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [showNewSOP, setShowNewSOP] = useState(false)

  const filteredSOPs = sops.filter(sop => {
    const matchesSearch =
      sop.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      sop.category.toLowerCase().includes(searchTerm.toLowerCase()) ||
      sop.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))

    const matchesStatus = statusFilter === 'all' || sop.status === statusFilter

    return matchesSearch && matchesStatus
  })

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
        <button
          onClick={() => setShowNewSOP(true)}
          className="flex items-center gap-2 bg-emerald-600 text-white px-4 py-2 rounded-lg hover:bg-emerald-700 transition-colors"
        >
          <Plus className="w-5 h-5" />
          New SOP
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total SOPs</p>
              <p className="text-2xl font-bold text-gray-900">{sops.length}</p>
            </div>
            <FileText className="w-8 h-8 text-emerald-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Active</p>
              <p className="text-2xl font-bold text-green-600">
                {sops.filter(s => s.status === 'active').length}
              </p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">In Review</p>
              <p className="text-2xl font-bold text-yellow-600">
                {sops.filter(s => s.status === 'in_review').length}
              </p>
            </div>
            <Clock className="w-8 h-8 text-yellow-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Draft</p>
              <p className="text-2xl font-bold text-gray-600">
                {sops.filter(s => s.status === 'draft').length}
              </p>
            </div>
            <FileText className="w-8 h-8 text-gray-500" />
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
              placeholder="Search by title, category, or tags..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
            />
          </div>
          <div className="flex gap-2">
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
            >
              <option value="all">All Status</option>
              <option value="draft">Draft</option>
              <option value="in_review">In Review</option>
              <option value="active">Active</option>
              <option value="archived">Archived</option>
            </select>
            <button className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
              <Download className="w-5 h-5" />
              Export
            </button>
          </div>
        </div>
      </div>

      {/* SOPs List */}
      <div className="space-y-4">
        {filteredSOPs.map((sop) => {
          const StatusIcon = statusConfig[sop.status].icon
          return (
            <div key={sop.id} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="text-lg font-semibold text-gray-900">{sop.title}</h3>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${categoryColors[sop.category] || 'bg-gray-100 text-gray-800'}`}>
                      {sop.category}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mb-3">{sop.description}</p>
                </div>
                <span className={`px-3 py-1 rounded-full text-sm font-medium flex items-center gap-1 ${statusConfig[sop.status].color}`}>
                  <StatusIcon className="w-4 h-4" />
                  {sop.status.replace('_', ' ')}
                </span>
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
                <button className="text-emerald-600 hover:text-emerald-900 text-sm font-medium">
                  View SOP
                </button>
                {sop.status !== 'archived' && (
                  <button className="text-blue-600 hover:text-blue-900 text-sm font-medium">
                    Edit
                  </button>
                )}
                <button className="text-purple-600 hover:text-purple-900 text-sm font-medium">
                  Version History
                </button>
                <button className="text-gray-600 hover:text-gray-900 text-sm font-medium">
                  Download PDF
                </button>
                {sop.status === 'draft' && (
                  <button className="text-green-600 hover:text-green-900 text-sm font-medium">
                    Submit for Review
                  </button>
                )}
              </div>
            </div>
          )
        })}

        {filteredSOPs.length === 0 && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
            <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No SOPs found matching your criteria</p>
          </div>
        )}
      </div>

      {/* New SOP Modal */}
      {showNewSOP && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4">
            <h2 className="text-2xl font-bold mb-4">Create New SOP</h2>
            <p className="text-gray-600 mb-4">SOP creation form will be implemented here</p>
            <button
              onClick={() => setShowNewSOP(false)}
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
