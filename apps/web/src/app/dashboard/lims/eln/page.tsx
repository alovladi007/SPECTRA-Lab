'use client'
import { useState, useEffect } from 'react'
import { BookOpen, Plus, Search, Calendar, User, FileText, Tag, Lock, Edit3, Eye, Trash2 } from 'lucide-react'
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
  content?: string
}

interface NewEntryForm {
  title: string
  type: 'experiment' | 'protocol' | 'observation' | 'analysis'
  tags: string
  content: string
}

const generateMockEntries = (): LabEntry[] => [
  {
    id: 'ELN-2024-001',
    title: 'Silicon Wafer Electrical Characterization - Batch A',
    author: 'Dr. Chen',
    date: '2024-11-08',
    tags: ['characterization', 'electrical', 'wafer'],
    type: 'experiment',
    status: 'approved',
    attachments: 5,
    excerpt: 'IV measurements performed on 25 sites across wafer. Average resistivity 1.2 Ω·cm...',
    content: 'Detailed IV measurements were performed on 25 sites across the wafer surface using a 4-point probe configuration. Results show excellent uniformity with average resistivity of 1.2 Ω·cm and standard deviation of 0.08 Ω·cm.'
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
    excerpt: 'Updated PECVD parameters for uniform deposition. Chamber pressure: 200 mTorr...',
    content: 'Updated PECVD parameters for achieving uniform thin film deposition. Chamber pressure maintained at 200 mTorr, RF power at 150W, substrate temperature at 300°C. Deposition rate: 50 nm/min.'
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
    excerpt: 'SEM imaging reveals gate oxide breakdown at 3 locations. Suspected contamination...',
    content: 'SEM imaging performed on failed devices reveals gate oxide breakdown at 3 distinct locations. Cross-sectional analysis indicates possible contamination during gate oxide growth. Recommended cleanroom particle count verification.'
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
    excerpt: 'Particle count: 100 particles/m³ Class 100. Temperature: 21°C, Humidity: 42%...',
    content: 'Daily cleanroom monitoring shows stable conditions. Particle count: 100 particles/m³ maintaining Class 100 specification. Temperature: 21°C, Humidity: 42%. All HEPA filters operating normally.'
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
    excerpt: 'Initial visual inspection shows good uniformity. XRD analysis pending...',
    content: 'Initial visual inspection of GaN on Sapphire wafer shows good uniformity across the surface. No visible defects or discoloration observed. XRD analysis scheduled for detailed crystallographic characterization.'
  },
  {
    id: 'ELN-2024-006',
    title: 'Photolithography Alignment Optimization',
    author: 'Dr. Wilson',
    date: '2024-11-06',
    tags: ['photolithography', 'alignment', 'process'],
    type: 'experiment',
    status: 'approved',
    attachments: 8,
    excerpt: 'Improved alignment marks design reduces overlay error to <50nm...',
    content: 'New alignment mark design implemented using cross-grating pattern. Results show significant improvement with overlay error reduced to <50nm across full wafer. Recommended for production use.'
  },
  {
    id: 'ELN-2024-007',
    title: 'Chemical Safety Protocol Update',
    author: 'Safety Officer',
    date: '2024-11-05',
    tags: ['safety', 'protocol', 'chemicals'],
    type: 'protocol',
    status: 'approved',
    attachments: 4,
    excerpt: 'Updated handling procedures for HF and piranha solution...',
    content: 'Updated safety protocol for handling HF and piranha solution. New PPE requirements include double-glove system and face shield. Emergency shower locations verified and tested.'
  },
  {
    id: 'ELN-2024-008',
    title: 'AFM Surface Roughness Analysis',
    author: 'Dr. Lee',
    date: '2024-11-04',
    tags: ['AFM', 'surface', 'roughness'],
    type: 'analysis',
    status: 'approved',
    attachments: 6,
    excerpt: 'AFM measurements show RMS roughness of 0.8nm for polished samples...',
    content: 'AFM surface analysis performed on CMP-polished silicon wafers. RMS roughness measured at 0.8nm across 10μm x 10μm scan area. Surface quality meets specifications for subsequent processing.'
  },
  {
    id: 'ELN-2024-009',
    title: 'Temperature Cycling Reliability Test',
    author: 'Dr. Brown',
    date: '2024-11-03',
    tags: ['reliability', 'temperature', 'testing'],
    type: 'experiment',
    status: 'under_review',
    attachments: 10,
    excerpt: 'Devices survived 500 thermal cycles (-40°C to 125°C) with minimal degradation...',
    content: 'Temperature cycling reliability testing completed. Devices subjected to 500 cycles between -40°C and 125°C. Electrical parameters show <5% drift. No visual defects or delamination observed.'
  },
  {
    id: 'ELN-2024-010',
    title: 'Weekly Lab Meeting Notes',
    author: 'Dr. Chen',
    date: '2024-11-02',
    tags: ['meeting', 'planning', 'discussion'],
    type: 'observation',
    status: 'approved',
    attachments: 1,
    excerpt: 'Discussed Q4 milestones and equipment maintenance schedule...',
    content: 'Weekly lab meeting covered Q4 project milestones, upcoming equipment maintenance windows, and resource allocation for new research initiatives. Action items distributed to team members.'
  }
]

const typeColors = {
  experiment: 'bg-blue-100 text-blue-800',
  protocol: 'bg-purple-100 text-purple-800',
  observation: 'bg-green-100 text-green-800',
  analysis: 'bg-orange-100 text-orange-800'
}

const statusConfig = {
  draft: { icon: Edit3, color: 'bg-gray-100 text-gray-800', label: 'Draft' },
  under_review: { icon: Eye, color: 'bg-yellow-100 text-yellow-800', label: 'Under Review' },
  approved: { icon: FileText, color: 'bg-green-100 text-green-800', label: 'Approved' },
  locked: { icon: Lock, color: 'bg-red-100 text-red-800', label: 'Locked' }
}

export default function ELNPage() {
  const [entries, setEntries] = useState<LabEntry[]>([])
  const [searchTerm, setSearchTerm] = useState('')
  const [typeFilter, setTypeFilter] = useState<string>('all')
  const [showNewEntry, setShowNewEntry] = useState(false)
  const [selectedEntry, setSelectedEntry] = useState<LabEntry | null>(null)
  const [showDetailsDialog, setShowDetailsDialog] = useState(false)

  const [newEntry, setNewEntry] = useState<NewEntryForm>({
    title: '',
    type: 'experiment',
    tags: '',
    content: ''
  })

  // Generate mock data on client side to prevent hydration errors
  useEffect(() => {
    setEntries(generateMockEntries())
  }, [])

  const filteredEntries = entries.filter(entry => {
    const matchesSearch =
      entry.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      entry.author.toLowerCase().includes(searchTerm.toLowerCase()) ||
      entry.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))

    const matchesType = typeFilter === 'all' || entry.type === typeFilter

    return matchesSearch && matchesType
  })

  const handleCreateEntry = () => {
    if (!newEntry.title || !newEntry.content) {
      alert('Please fill in all required fields')
      return
    }

    const currentUser = 'Current User' // In real app, get from auth context
    const entry: LabEntry = {
      id: `ELN-2024-${String(entries.length + 1).padStart(3, '0')}`,
      title: newEntry.title,
      author: currentUser,
      date: new Date().toISOString().split('T')[0],
      tags: newEntry.tags.split(',').map(t => t.trim()).filter(t => t),
      type: newEntry.type,
      status: 'draft',
      attachments: 0,
      excerpt: newEntry.content.slice(0, 100) + '...',
      content: newEntry.content
    }

    setEntries([entry, ...entries])
    setShowNewEntry(false)
    setNewEntry({
      title: '',
      type: 'experiment',
      tags: '',
      content: ''
    })
  }

  const handleViewDetails = (entry: LabEntry) => {
    setSelectedEntry(entry)
    setShowDetailsDialog(true)
  }

  const handleSubmitForReview = (entryId: string) => {
    setEntries(entries.map(entry => {
      if (entry.id === entryId && entry.status === 'draft') {
        return { ...entry, status: 'under_review' as const }
      }
      return entry
    }))
  }

  const handleApprove = (entryId: string) => {
    setEntries(entries.map(entry => {
      if (entry.id === entryId && entry.status === 'under_review') {
        return { ...entry, status: 'approved' as const }
      }
      return entry
    }))
  }

  const handleDelete = (entryId: string) => {
    if (confirm('Are you sure you want to delete this entry?')) {
      setEntries(entries.filter(entry => entry.id !== entryId))
    }
  }

  const stats = {
    total: entries.length,
    experiments: entries.filter(e => e.type === 'experiment').length,
    protocols: entries.filter(e => e.type === 'protocol').length,
    under_review: entries.filter(e => e.status === 'under_review').length
  }

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
        <Button
          onClick={() => setShowNewEntry(true)}
          className="flex items-center gap-2"
        >
          <Plus className="w-5 h-5" />
          New Entry
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Entries</p>
              <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
            </div>
            <BookOpen className="w-8 h-8 text-indigo-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Experiments</p>
              <p className="text-2xl font-bold text-blue-600">{stats.experiments}</p>
            </div>
            <FileText className="w-8 h-8 text-blue-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Protocols</p>
              <p className="text-2xl font-bold text-purple-600">{stats.protocols}</p>
            </div>
            <FileText className="w-8 h-8 text-purple-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Under Review</p>
              <p className="text-2xl font-bold text-orange-600">{stats.under_review}</p>
            </div>
            <Eye className="w-8 h-8 text-orange-500" />
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
              placeholder="Search by title, author, or tags..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
          <Select value={typeFilter} onValueChange={setTypeFilter}>
            <SelectTrigger className="w-full md:w-[200px]">
              <SelectValue placeholder="All Types" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Types</SelectItem>
              <SelectItem value="experiment">Experiments</SelectItem>
              <SelectItem value="protocol">Protocols</SelectItem>
              <SelectItem value="observation">Observations</SelectItem>
              <SelectItem value="analysis">Analysis</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </Card>

      {/* Entries List */}
      <div className="space-y-4">
        {filteredEntries.map((entry) => {
          const StatusIcon = statusConfig[entry.status].icon
          return (
            <Card key={entry.id} className="p-6 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2 flex-wrap">
                    <h3 className="text-lg font-semibold text-gray-900">{entry.title}</h3>
                    <Badge className={typeColors[entry.type]}>
                      {entry.type}
                    </Badge>
                    <Badge className={statusConfig[entry.status].color}>
                      <StatusIcon className="w-3 h-3 mr-1" />
                      {statusConfig[entry.status].label}
                    </Badge>
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
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleViewDetails(entry)}
                  className="text-indigo-600 hover:text-indigo-900"
                >
                  View Entry
                </Button>
                {entry.status !== 'locked' && (
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
                  className="text-gray-600 hover:text-gray-900"
                >
                  Export PDF
                </Button>
                {entry.status === 'draft' && (
                  <>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleSubmitForReview(entry.id)}
                      className="text-green-600 hover:text-green-900"
                    >
                      Submit for Review
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDelete(entry.id)}
                      className="text-red-600 hover:text-red-900"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </>
                )}
                {entry.status === 'under_review' && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleApprove(entry.id)}
                    className="text-green-600 hover:text-green-900"
                  >
                    Approve
                  </Button>
                )}
              </div>
            </Card>
          )
        })}

        {filteredEntries.length === 0 && (
          <Card className="p-12 text-center">
            <BookOpen className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No entries found matching your criteria</p>
          </Card>
        )}
      </div>

      {/* New Entry Dialog */}
      <Dialog open={showNewEntry} onOpenChange={setShowNewEntry}>
        <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>New Lab Notebook Entry</DialogTitle>
            <DialogDescription>
              Create a new entry to document your work
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Title *
              </label>
              <Input
                value={newEntry.title}
                onChange={(e) => setNewEntry({ ...newEntry, title: e.target.value })}
                placeholder="e.g., Silicon Wafer Characterization"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Entry Type *
              </label>
              <Select
                value={newEntry.type}
                onValueChange={(value: any) => setNewEntry({ ...newEntry, type: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="experiment">Experiment</SelectItem>
                  <SelectItem value="protocol">Protocol</SelectItem>
                  <SelectItem value="observation">Observation</SelectItem>
                  <SelectItem value="analysis">Analysis</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Tags
              </label>
              <Input
                value={newEntry.tags}
                onChange={(e) => setNewEntry({ ...newEntry, tags: e.target.value })}
                placeholder="e.g., characterization, electrical, wafer (comma-separated)"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">
                Content *
              </label>
              <Textarea
                value={newEntry.content}
                onChange={(e) => setNewEntry({ ...newEntry, content: e.target.value })}
                placeholder="Document your experiment, protocol, or observations in detail..."
                className="min-h-[200px]"
              />
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowNewEntry(false)}
            >
              Cancel
            </Button>
            <Button onClick={handleCreateEntry}>
              Create Entry
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Details Dialog */}
      <Dialog open={showDetailsDialog} onOpenChange={setShowDetailsDialog}>
        <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Lab Notebook Entry</DialogTitle>
            <DialogDescription>
              Entry ID: {selectedEntry?.id}
            </DialogDescription>
          </DialogHeader>

          {selectedEntry && (
            <div className="space-y-4">
              <div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">{selectedEntry.title}</h3>
                <div className="flex flex-wrap gap-2 mb-4">
                  <Badge className={typeColors[selectedEntry.type]}>
                    {selectedEntry.type}
                  </Badge>
                  <Badge className={statusConfig[selectedEntry.status].color}>
                    {statusConfig[selectedEntry.status].label}
                  </Badge>
                </div>
              </div>

              <div className="border-t border-gray-200 pt-4">
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">Author</p>
                    <p className="text-base text-gray-900">{selectedEntry.author}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Date</p>
                    <p className="text-base text-gray-900">{selectedEntry.date}</p>
                  </div>
                </div>
              </div>

              <div className="border-t border-gray-200 pt-4">
                <p className="text-sm font-medium text-gray-500 mb-2">Tags</p>
                <div className="flex flex-wrap gap-2">
                  {selectedEntry.tags.map(tag => (
                    <span key={tag} className="flex items-center gap-1 px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full">
                      <Tag className="w-3 h-3" />
                      {tag}
                    </span>
                  ))}
                </div>
              </div>

              <div className="border-t border-gray-200 pt-4">
                <p className="text-sm font-medium text-gray-500 mb-2">Content</p>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-900 whitespace-pre-wrap">{selectedEntry.content}</p>
                </div>
              </div>

              <div className="border-t border-gray-200 pt-4">
                <p className="text-sm font-medium text-gray-500 mb-2">Attachments</p>
                <p className="text-sm text-gray-600">{selectedEntry.attachments} file(s) attached</p>
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
