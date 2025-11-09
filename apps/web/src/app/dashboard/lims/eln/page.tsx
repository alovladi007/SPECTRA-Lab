'use client'
import { useState } from 'react'
import { BookOpen, Plus, Search, Calendar, User, FileText, Tag, Lock, Edit3, Eye } from 'lucide-react'

interface LabEntry {
  id: string
  title: string
  author: string
  date: string
  tags: string[]
  type: 'experiment' | 'protocol' | 'observation' | 'analysis'
  status: 'draft' | 'under_review' | 'approved' | 'locked'
  attachments: number
  excerpt: string
}

const mockEntries: LabEntry[] = [
  {
    id: 'ELN-2024-001',
    title: 'Silicon Wafer Electrical Characterization - Batch A',
    author: 'Dr. Chen',
    date: '2024-11-08',
    tags: ['characterization', 'electrical', 'wafer'],
    type: 'experiment',
    status: 'approved',
    attachments: 5,
    excerpt: 'IV measurements performed on 25 sites across wafer. Average resistivity 1.2 Ω·cm...'
  },
  {
    id: 'ELN-2024-002',
    title: 'Thin Film Deposition Protocol v2.1',
    author: 'Dr. Martinez',
    date: '2024-11-07',
    tags: ['protocol', 'deposition', 'thin-film'],
    type: 'protocol',
    status: 'locked',
    attachments: 3,
    excerpt: 'Updated PECVD parameters for uniform deposition. Chamber pressure: 200 mTorr...'
  },
  {
    id: 'ELN-2024-003',
    title: 'Device Failure Analysis - Prototype Rev3',
    author: 'Dr. Patel',
    date: '2024-11-09',
    tags: ['failure-analysis', 'device', 'SEM'],
    type: 'analysis',
    status: 'under_review',
    attachments: 12,
    excerpt: 'SEM imaging reveals gate oxide breakdown at 3 locations. Suspected contamination...'
  },
  {
    id: 'ELN-2024-004',
    title: 'Daily Observation: Cleanroom Environment',
    author: 'Lab Manager',
    date: '2024-11-09',
    tags: ['observation', 'cleanroom', 'maintenance'],
    type: 'observation',
    status: 'approved',
    attachments: 2,
    excerpt: 'Particle count: 100 particles/m³ Class 100. Temperature: 21°C, Humidity: 42%...'
  },
  {
    id: 'ELN-2024-005',
    title: 'GaN on Sapphire Material Inspection',
    author: 'Dr. Kim',
    date: '2024-11-09',
    tags: ['inspection', 'GaN', 'material'],
    type: 'experiment',
    status: 'draft',
    attachments: 1,
    excerpt: 'Initial visual inspection shows good uniformity. XRD analysis pending...'
  }
]

const typeColors = {
  experiment: 'bg-blue-100 text-blue-800',
  protocol: 'bg-purple-100 text-purple-800',
  observation: 'bg-green-100 text-green-800',
  analysis: 'bg-orange-100 text-orange-800'
}

const statusIcons = {
  draft: Edit3,
  under_review: Eye,
  approved: FileText,
  locked: Lock
}

export default function ELNPage() {
  const [entries, setEntries] = useState(mockEntries)
  const [searchTerm, setSearchTerm] = useState('')
  const [typeFilter, setTypeFilter] = useState<string>('all')
  const [showNewEntry, setShowNewEntry] = useState(false)

  const filteredEntries = entries.filter(entry => {
    const matchesSearch =
      entry.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      entry.author.toLowerCase().includes(searchTerm.toLowerCase()) ||
      entry.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))

    const matchesType = typeFilter === 'all' || entry.type === typeFilter

    return matchesSearch && matchesType
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-blue-500 rounded-xl flex items-center justify-center">
            <BookOpen className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Electronic Lab Notebook</h1>
            <p className="text-gray-600 mt-1">Document experiments, protocols, and observations</p>
          </div>
        </div>
        <button
          onClick={() => setShowNewEntry(true)}
          className="flex items-center gap-2 bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition-colors"
        >
          <Plus className="w-5 h-5" />
          New Entry
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Entries</p>
              <p className="text-2xl font-bold text-gray-900">{entries.length}</p>
            </div>
            <BookOpen className="w-8 h-8 text-indigo-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Experiments</p>
              <p className="text-2xl font-bold text-blue-600">
                {entries.filter(e => e.type === 'experiment').length}
              </p>
            </div>
            <FileText className="w-8 h-8 text-blue-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Protocols</p>
              <p className="text-2xl font-bold text-purple-600">
                {entries.filter(e => e.type === 'protocol').length}
              </p>
            </div>
            <FileText className="w-8 h-8 text-purple-500" />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Under Review</p>
              <p className="text-2xl font-bold text-orange-600">
                {entries.filter(e => e.status === 'under_review').length}
              </p>
            </div>
            <Eye className="w-8 h-8 text-orange-500" />
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
              placeholder="Search by title, author, or tags..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
          </div>
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          >
            <option value="all">All Types</option>
            <option value="experiment">Experiments</option>
            <option value="protocol">Protocols</option>
            <option value="observation">Observations</option>
            <option value="analysis">Analysis</option>
          </select>
        </div>
      </div>

      {/* Entries List */}
      <div className="space-y-4">
        {filteredEntries.map((entry) => {
          const StatusIcon = statusIcons[entry.status]
          return (
            <div key={entry.id} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="text-lg font-semibold text-gray-900">{entry.title}</h3>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${typeColors[entry.type]}`}>
                      {entry.type}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mb-3">{entry.excerpt}</p>
                  <div className="flex flex-wrap items-center gap-4 text-sm text-gray-500">
                    <div className="flex items-center gap-1">
                      <User className="w-4 h-4" />
                      {entry.author}
                    </div>
                    <div className="flex items-center gap-1">
                      <Calendar className="w-4 h-4" />
                      {entry.date}
                    </div>
                    <div className="flex items-center gap-1">
                      <FileText className="w-4 h-4" />
                      {entry.attachments} attachments
                    </div>
                    <div className="flex items-center gap-1">
                      <StatusIcon className="w-4 h-4" />
                      <span className="capitalize">{entry.status.replace('_', ' ')}</span>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2 mt-3">
                    {entry.tags.map(tag => (
                      <span key={tag} className="flex items-center gap-1 px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full">
                        <Tag className="w-3 h-3" />
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
              <div className="flex gap-2 pt-3 border-t border-gray-200">
                <button className="text-indigo-600 hover:text-indigo-900 text-sm font-medium">
                  View Entry
                </button>
                {entry.status !== 'locked' && (
                  <button className="text-blue-600 hover:text-blue-900 text-sm font-medium">
                    Edit
                  </button>
                )}
                <button className="text-gray-600 hover:text-gray-900 text-sm font-medium">
                  Export PDF
                </button>
                {entry.status === 'draft' && (
                  <button className="text-green-600 hover:text-green-900 text-sm font-medium">
                    Submit for Review
                  </button>
                )}
              </div>
            </div>
          )
        })}

        {filteredEntries.length === 0 && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
            <BookOpen className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No entries found matching your criteria</p>
          </div>
        )}
      </div>

      {/* New Entry Modal */}
      {showNewEntry && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4">
            <h2 className="text-2xl font-bold mb-4">New Lab Notebook Entry</h2>
            <p className="text-gray-600 mb-4">Entry creation form will be implemented here</p>
            <button
              onClick={() => setShowNewEntry(false)}
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
