'use client'

import { useState, useMemo, useEffect } from 'react'
import {
  Activity, Plus, Search, Filter, Download, Eye, Edit2, Trash2,
  Calendar, User, Tag, Clock, CheckCircle2, XCircle, PlayCircle,
  PauseCircle, FileText, Link as LinkIcon, Beaker, Zap, Waves, Layers
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { Progress } from '@/components/ui/progress'

interface Experiment {
  id: string
  name: string
  type: 'Electrical' | 'Optical' | 'Structural' | 'Chemical' | 'Process Simulation'
  status: 'Planned' | 'In Progress' | 'Completed' | 'Failed' | 'Cancelled'
  priority: 'Low' | 'Medium' | 'High' | 'Critical'
  owner: string
  samples: string[]
  technique: string
  created: string
  started?: string
  completed?: string
  duration?: number
  progress: number
  results?: number
  description?: string
  parameters?: Record<string, any>
}

// Mock data generator
const generateMockExperiments = (): Experiment[] => {
  const types: Experiment['type'][] = ['Electrical', 'Optical', 'Structural', 'Chemical', 'Process Simulation']
  const statuses: Experiment['status'][] = ['Planned', 'In Progress', 'Completed', 'Failed', 'Cancelled']
  const priorities: Experiment['priority'][] = ['Low', 'Medium', 'High', 'Critical']
  const owners = ['Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown', 'Dr. Davis']
  const techniques = {
    'Electrical': ['Four-Point Probe', 'Hall Effect', 'MOSFET Analysis', 'BJT Analysis'],
    'Optical': ['UV-Vis-NIR', 'FTIR', 'Raman', 'Ellipsometry'],
    'Structural': ['XRD', 'SEM', 'TEM', 'AFM'],
    'Chemical': ['XPS', 'SIMS', 'EDX', 'RBS'],
    'Process Simulation': ['Diffusion', 'Oxidation', 'Ion Implantation']
  }

  return Array.from({ length: 35 }, (_, i) => {
    const id = `EXP-${String(i + 2000).padStart(4, '0')}`
    const type = types[Math.floor(Math.random() * types.length)]
    const status = statuses[Math.floor(Math.random() * statuses.length)]
    const created = new Date(Date.now() - Math.random() * 60 * 24 * 60 * 60 * 1000).toISOString()
    const technique = techniques[type][Math.floor(Math.random() * techniques[type].length)]

    let progress = 0
    if (status === 'Completed') progress = 100
    else if (status === 'In Progress') progress = Math.floor(Math.random() * 80) + 10
    else if (status === 'Failed' || status === 'Cancelled') progress = Math.floor(Math.random() * 60)

    const started = status !== 'Planned' ? new Date(new Date(created).getTime() + Math.random() * 5 * 24 * 60 * 60 * 1000).toISOString() : undefined
    const completed = status === 'Completed' || status === 'Failed' ? new Date(new Date(started || created).getTime() + Math.random() * 10 * 24 * 60 * 60 * 1000).toISOString() : undefined

    const numSamples = Math.floor(Math.random() * 5) + 1
    const samples = Array.from({ length: numSamples }, (_, j) => `SMP-${1000 + Math.floor(Math.random() * 100)}`)

    return {
      id,
      name: `${technique} Characterization ${i + 1}`,
      type,
      status,
      priority: priorities[Math.floor(Math.random() * priorities.length)],
      owner: owners[Math.floor(Math.random() * owners.length)],
      samples,
      technique,
      created,
      started,
      completed,
      duration: completed && started ? Math.floor((new Date(completed).getTime() - new Date(started).getTime()) / (1000 * 60 * 60)) : undefined,
      progress,
      results: status === 'Completed' ? Math.floor(Math.random() * 20) + 5 : status === 'In Progress' ? Math.floor(Math.random() * 10) : 0,
      description: `${technique} measurement of ${type.toLowerCase()} properties`,
      parameters: {
        temperature: '300 K',
        pressure: '1e-6 Torr',
        'scan_range': '400-800 nm'
      }
    }
  })
}

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [filterType, setFilterType] = useState<string>('all')
  const [filterStatus, setFilterStatus] = useState<string>('all')
  const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null)
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false)
  const [isDetailsDialogOpen, setIsDetailsDialogOpen] = useState(false)

  // New experiment form state
  const [newExperiment, setNewExperiment] = useState({
    name: '',
    type: 'Electrical' as Experiment['type'],
    technique: '',
    priority: 'Medium' as Experiment['priority'],
    owner: '',
    samples: '',
    description: ''
  })

  // Generate mock data on client side only to avoid hydration mismatch
  useEffect(() => {
    setExperiments(generateMockExperiments())
  }, [])

  // Filtered experiments
  const filteredExperiments = useMemo(() => {
    return experiments.filter(exp => {
      const matchesSearch =
        exp.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        exp.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        exp.technique.toLowerCase().includes(searchQuery.toLowerCase()) ||
        exp.owner.toLowerCase().includes(searchQuery.toLowerCase())

      const matchesType = filterType === 'all' || exp.type === filterType
      const matchesStatus = filterStatus === 'all' || exp.status === filterStatus

      return matchesSearch && matchesType && matchesStatus
    })
  }, [experiments, searchQuery, filterType, filterStatus])

  // Statistics
  const stats = useMemo(() => ({
    total: experiments.length,
    planned: experiments.filter(e => e.status === 'Planned').length,
    inProgress: experiments.filter(e => e.status === 'In Progress').length,
    completed: experiments.filter(e => e.status === 'Completed').length,
    failed: experiments.filter(e => e.status === 'Failed').length
  }), [experiments])

  const handleCreateExperiment = () => {
    const id = `EXP-${String(experiments.length + 2000).padStart(4, '0')}`
    const now = new Date().toISOString()

    const experiment: Experiment = {
      id,
      name: newExperiment.name,
      type: newExperiment.type,
      status: 'Planned',
      priority: newExperiment.priority,
      owner: newExperiment.owner,
      samples: newExperiment.samples.split(',').map(s => s.trim()).filter(Boolean),
      technique: newExperiment.technique,
      created: now,
      progress: 0,
      results: 0,
      description: newExperiment.description
    }

    setExperiments(prev => [experiment, ...prev])
    setIsCreateDialogOpen(false)

    // Reset form
    setNewExperiment({
      name: '',
      type: 'Electrical',
      technique: '',
      priority: 'Medium',
      owner: '',
      samples: '',
      description: ''
    })
  }

  const handleDeleteExperiment = (id: string) => {
    if (confirm('Are you sure you want to delete this experiment?')) {
      setExperiments(prev => prev.filter(e => e.id !== id))
      setIsDetailsDialogOpen(false)
    }
  }

  const handleStartExperiment = (id: string) => {
    setExperiments(prev => prev.map(e =>
      e.id === id ? { ...e, status: 'In Progress', started: new Date().toISOString(), progress: 5 } : e
    ))
  }

  const handleCompleteExperiment = (id: string) => {
    setExperiments(prev => prev.map(e =>
      e.id === id ? { ...e, status: 'Completed', completed: new Date().toISOString(), progress: 100 } : e
    ))
  }

  const getStatusColor = (status: Experiment['status']) => {
    switch (status) {
      case 'Planned': return 'bg-gray-100 text-gray-800'
      case 'In Progress': return 'bg-blue-100 text-blue-800'
      case 'Completed': return 'bg-green-100 text-green-800'
      case 'Failed': return 'bg-red-100 text-red-800'
      case 'Cancelled': return 'bg-orange-100 text-orange-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getStatusIcon = (status: Experiment['status']) => {
    switch (status) {
      case 'Planned': return <Clock className="w-4 h-4" />
      case 'In Progress': return <PlayCircle className="w-4 h-4" />
      case 'Completed': return <CheckCircle2 className="w-4 h-4" />
      case 'Failed': return <XCircle className="w-4 h-4" />
      case 'Cancelled': return <PauseCircle className="w-4 h-4" />
      default: return null
    }
  }

  const getPriorityColor = (priority: Experiment['priority']) => {
    switch (priority) {
      case 'Low': return 'bg-gray-100 text-gray-700'
      case 'Medium': return 'bg-blue-100 text-blue-700'
      case 'High': return 'bg-orange-100 text-orange-700'
      case 'Critical': return 'bg-red-100 text-red-700'
      default: return 'bg-gray-100 text-gray-700'
    }
  }

  const getTypeIcon = (type: Experiment['type']) => {
    switch (type) {
      case 'Electrical': return <Zap className="w-4 h-4" />
      case 'Optical': return <Waves className="w-4 h-4" />
      case 'Structural': return <Layers className="w-4 h-4" />
      case 'Chemical': return <Beaker className="w-4 h-4" />
      case 'Process Simulation': return <Activity className="w-4 h-4" />
      default: return <Activity className="w-4 h-4" />
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Experiments</h1>
            <p className="text-gray-600 mt-1">Manage and track all experimental runs</p>
          </div>
        </div>

        <Button
          className="flex items-center gap-2"
          onClick={() => setIsCreateDialogOpen(true)}
        >
          <Plus className="w-4 h-4" />
          New Experiment
        </Button>

        <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
          <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Create New Experiment</DialogTitle>
              <DialogDescription>
                Enter the details for the new experiment
              </DialogDescription>
            </DialogHeader>

            <div className="grid grid-cols-2 gap-4 py-4">
              <div className="col-span-2 space-y-2">
                <Label htmlFor="exp-name">Experiment Name *</Label>
                <Input
                  id="exp-name"
                  placeholder="e.g., Hall Effect Measurement Series A"
                  value={newExperiment.name}
                  onChange={(e) => setNewExperiment({ ...newExperiment, name: e.target.value })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="exp-type">Type *</Label>
                <Select
                  value={newExperiment.type}
                  onValueChange={(value) => setNewExperiment({ ...newExperiment, type: value as Experiment['type'] })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Electrical">Electrical</SelectItem>
                    <SelectItem value="Optical">Optical</SelectItem>
                    <SelectItem value="Structural">Structural</SelectItem>
                    <SelectItem value="Chemical">Chemical</SelectItem>
                    <SelectItem value="Process Simulation">Process Simulation</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="exp-technique">Technique *</Label>
                <Input
                  id="exp-technique"
                  placeholder="e.g., Hall Effect, XRD"
                  value={newExperiment.technique}
                  onChange={(e) => setNewExperiment({ ...newExperiment, technique: e.target.value })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="exp-priority">Priority</Label>
                <Select
                  value={newExperiment.priority}
                  onValueChange={(value) => setNewExperiment({ ...newExperiment, priority: value as Experiment['priority'] })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Low">Low</SelectItem>
                    <SelectItem value="Medium">Medium</SelectItem>
                    <SelectItem value="High">High</SelectItem>
                    <SelectItem value="Critical">Critical</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="exp-owner">Owner *</Label>
                <Input
                  id="exp-owner"
                  placeholder="e.g., Dr. Smith"
                  value={newExperiment.owner}
                  onChange={(e) => setNewExperiment({ ...newExperiment, owner: e.target.value })}
                />
              </div>

              <div className="col-span-2 space-y-2">
                <Label htmlFor="exp-samples">Sample IDs (comma-separated)</Label>
                <Input
                  id="exp-samples"
                  placeholder="e.g., SMP-1001, SMP-1002"
                  value={newExperiment.samples}
                  onChange={(e) => setNewExperiment({ ...newExperiment, samples: e.target.value })}
                />
              </div>

              <div className="col-span-2 space-y-2">
                <Label htmlFor="exp-description">Description</Label>
                <Textarea
                  id="exp-description"
                  placeholder="Describe the experiment objectives and methodology..."
                  value={newExperiment.description}
                  onChange={(e) => setNewExperiment({ ...newExperiment, description: e.target.value })}
                  rows={3}
                />
              </div>
            </div>

            <DialogFooter>
              <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
                Cancel
              </Button>
              <Button
                onClick={handleCreateExperiment}
                disabled={!newExperiment.name || !newExperiment.technique || !newExperiment.owner}
              >
                Create Experiment
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total</p>
                <p className="text-2xl font-bold">{stats.total}</p>
              </div>
              <Activity className="w-8 h-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Planned</p>
                <p className="text-2xl font-bold text-gray-600">{stats.planned}</p>
              </div>
              <Clock className="w-8 h-8 text-gray-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">In Progress</p>
                <p className="text-2xl font-bold text-blue-600">{stats.inProgress}</p>
              </div>
              <PlayCircle className="w-8 h-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Completed</p>
                <p className="text-2xl font-bold text-green-600">{stats.completed}</p>
              </div>
              <CheckCircle2 className="w-8 h-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Failed</p>
                <p className="text-2xl font-bold text-red-600">{stats.failed}</p>
              </div>
              <XCircle className="w-8 h-8 text-red-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Search and Filters */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search by ID, name, technique, or owner..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>

            <Select value={filterType} onValueChange={setFilterType}>
              <SelectTrigger className="w-full md:w-[200px]">
                <div className="flex items-center gap-2">
                  <Filter className="w-4 h-4" />
                  <SelectValue placeholder="Type" />
                </div>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="Electrical">Electrical</SelectItem>
                <SelectItem value="Optical">Optical</SelectItem>
                <SelectItem value="Structural">Structural</SelectItem>
                <SelectItem value="Chemical">Chemical</SelectItem>
                <SelectItem value="Process Simulation">Process Simulation</SelectItem>
              </SelectContent>
            </Select>

            <Select value={filterStatus} onValueChange={setFilterStatus}>
              <SelectTrigger className="w-full md:w-[200px]">
                <div className="flex items-center gap-2">
                  <Filter className="w-4 h-4" />
                  <SelectValue placeholder="Status" />
                </div>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="Planned">Planned</SelectItem>
                <SelectItem value="In Progress">In Progress</SelectItem>
                <SelectItem value="Completed">Completed</SelectItem>
                <SelectItem value="Failed">Failed</SelectItem>
                <SelectItem value="Cancelled">Cancelled</SelectItem>
              </SelectContent>
            </Select>

            <Button variant="outline" className="w-full md:w-auto">
              <Download className="w-4 h-4 mr-2" />
              Export
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Experiments Table */}
      <Card>
        <CardHeader>
          <CardTitle>Experiments ({filteredExperiments.length})</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="border-b">
                <tr className="text-left">
                  <th className="pb-3 font-medium text-muted-foreground">ID</th>
                  <th className="pb-3 font-medium text-muted-foreground">Name</th>
                  <th className="pb-3 font-medium text-muted-foreground">Type</th>
                  <th className="pb-3 font-medium text-muted-foreground">Status</th>
                  <th className="pb-3 font-medium text-muted-foreground">Priority</th>
                  <th className="pb-3 font-medium text-muted-foreground">Progress</th>
                  <th className="pb-3 font-medium text-muted-foreground">Owner</th>
                  <th className="pb-3 font-medium text-muted-foreground">Samples</th>
                  <th className="pb-3 font-medium text-muted-foreground">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredExperiments.map((exp) => (
                  <tr key={exp.id} className="border-b hover:bg-muted/50 transition-colors">
                    <td className="py-3">
                      <span className="font-mono text-sm">{exp.id}</span>
                    </td>
                    <td className="py-3">
                      <div>
                        <p className="font-medium">{exp.name}</p>
                        <p className="text-xs text-muted-foreground">{exp.technique}</p>
                      </div>
                    </td>
                    <td className="py-3">
                      <Badge variant="outline" className="flex items-center gap-1 w-fit">
                        {getTypeIcon(exp.type)}
                        {exp.type}
                      </Badge>
                    </td>
                    <td className="py-3">
                      <Badge className={getStatusColor(exp.status)}>
                        <div className="flex items-center gap-1">
                          {getStatusIcon(exp.status)}
                          {exp.status}
                        </div>
                      </Badge>
                    </td>
                    <td className="py-3">
                      <Badge className={getPriorityColor(exp.priority)} variant="outline">
                        {exp.priority}
                      </Badge>
                    </td>
                    <td className="py-3">
                      <div className="space-y-1">
                        <Progress value={exp.progress} className="h-2 w-20" />
                        <p className="text-xs text-muted-foreground">{exp.progress}%</p>
                      </div>
                    </td>
                    <td className="py-3">
                      <div className="flex items-center gap-1 text-sm text-muted-foreground">
                        <User className="w-3 h-3" />
                        {exp.owner}
                      </div>
                    </td>
                    <td className="py-3">
                      <Badge variant="secondary">{exp.samples.length}</Badge>
                    </td>
                    <td className="py-3">
                      <div className="flex items-center gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => {
                            setSelectedExperiment(exp)
                            setIsDetailsDialogOpen(true)
                          }}
                        >
                          <Eye className="w-4 h-4" />
                        </Button>
                        {exp.status === 'Planned' && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleStartExperiment(exp.id)}
                          >
                            <PlayCircle className="w-4 h-4 text-blue-500" />
                          </Button>
                        )}
                        {exp.status === 'In Progress' && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleCompleteExperiment(exp.id)}
                          >
                            <CheckCircle2 className="w-4 h-4 text-green-500" />
                          </Button>
                        )}
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDeleteExperiment(exp.id)}
                        >
                          <Trash2 className="w-4 h-4 text-red-500" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {filteredExperiments.length === 0 && (
              <div className="text-center py-12">
                <Activity className="w-16 h-16 text-muted-foreground mx-auto mb-4 opacity-20" />
                <p className="text-lg font-medium text-muted-foreground mb-2">No experiments found</p>
                <p className="text-sm text-muted-foreground">
                  {searchQuery || filterType !== 'all' || filterStatus !== 'all'
                    ? 'Try adjusting your search or filters'
                    : 'Create your first experiment to get started'}
                </p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Experiment Details Dialog */}
      <Dialog open={isDetailsDialogOpen} onOpenChange={setIsDetailsDialogOpen}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Experiment Details</DialogTitle>
            <DialogDescription>
              Complete information for {selectedExperiment?.id}
            </DialogDescription>
          </DialogHeader>

          {selectedExperiment && (
            <div className="space-y-6">
              {/* Status and Priority */}
              <div className="flex items-center gap-4">
                <Badge className={getStatusColor(selectedExperiment.status)}>
                  <div className="flex items-center gap-1">
                    {getStatusIcon(selectedExperiment.status)}
                    {selectedExperiment.status}
                  </div>
                </Badge>
                <Badge className={getPriorityColor(selectedExperiment.priority)} variant="outline">
                  {selectedExperiment.priority} Priority
                </Badge>
                <Badge variant="outline" className="flex items-center gap-1">
                  {getTypeIcon(selectedExperiment.type)}
                  {selectedExperiment.type}
                </Badge>
              </div>

              {/* Progress */}
              {selectedExperiment.status === 'In Progress' && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <Label>Progress</Label>
                    <span className="text-muted-foreground">{selectedExperiment.progress}%</span>
                  </div>
                  <Progress value={selectedExperiment.progress} className="h-3" />
                </div>
              )}

              {/* Main Details */}
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <Label className="text-muted-foreground">Experiment ID</Label>
                    <p className="font-mono font-medium">{selectedExperiment.id}</p>
                  </div>

                  <div>
                    <Label className="text-muted-foreground">Name</Label>
                    <p className="font-medium">{selectedExperiment.name}</p>
                  </div>

                  <div>
                    <Label className="text-muted-foreground">Technique</Label>
                    <p className="font-medium">{selectedExperiment.technique}</p>
                  </div>

                  <div>
                    <Label className="text-muted-foreground">Owner</Label>
                    <div className="flex items-center gap-2 mt-1">
                      <User className="w-4 h-4 text-muted-foreground" />
                      <p className="font-medium">{selectedExperiment.owner}</p>
                    </div>
                  </div>

                  <div>
                    <Label className="text-muted-foreground">Samples</Label>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {selectedExperiment.samples.map(sample => (
                        <Badge key={sample} variant="outline" className="font-mono">
                          {sample}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <Label className="text-muted-foreground">Created</Label>
                    <div className="flex items-center gap-2 mt-1">
                      <Calendar className="w-4 h-4 text-muted-foreground" />
                      <p className="font-medium">
                        {new Date(selectedExperiment.created).toLocaleString()}
                      </p>
                    </div>
                  </div>

                  {selectedExperiment.started && (
                    <div>
                      <Label className="text-muted-foreground">Started</Label>
                      <div className="flex items-center gap-2 mt-1">
                        <PlayCircle className="w-4 h-4 text-muted-foreground" />
                        <p className="font-medium">
                          {new Date(selectedExperiment.started).toLocaleString()}
                        </p>
                      </div>
                    </div>
                  )}

                  {selectedExperiment.completed && (
                    <div>
                      <Label className="text-muted-foreground">Completed</Label>
                      <div className="flex items-center gap-2 mt-1">
                        <CheckCircle2 className="w-4 h-4 text-muted-foreground" />
                        <p className="font-medium">
                          {new Date(selectedExperiment.completed).toLocaleString()}
                        </p>
                      </div>
                    </div>
                  )}

                  {selectedExperiment.duration && (
                    <div>
                      <Label className="text-muted-foreground">Duration</Label>
                      <div className="flex items-center gap-2 mt-1">
                        <Clock className="w-4 h-4 text-muted-foreground" />
                        <p className="font-medium">{selectedExperiment.duration} hours</p>
                      </div>
                    </div>
                  )}

                  <div>
                    <Label className="text-muted-foreground">Results</Label>
                    <div className="flex items-center gap-2 mt-1">
                      <FileText className="w-4 h-4 text-muted-foreground" />
                      <p className="font-medium">{selectedExperiment.results} result files</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Description */}
              {selectedExperiment.description && (
                <div>
                  <Label className="text-muted-foreground">Description</Label>
                  <div className="mt-2 p-3 bg-muted rounded-lg">
                    <p className="text-sm">{selectedExperiment.description}</p>
                  </div>
                </div>
              )}

              {/* Actions */}
              <div className="flex justify-between pt-4 border-t">
                <div className="flex gap-2">
                  {selectedExperiment.status === 'Planned' && (
                    <Button
                      variant="default"
                      size="sm"
                      onClick={() => {
                        handleStartExperiment(selectedExperiment.id)
                        setIsDetailsDialogOpen(false)
                      }}
                    >
                      <PlayCircle className="w-4 h-4 mr-2" />
                      Start Experiment
                    </Button>
                  )}
                  {selectedExperiment.status === 'In Progress' && (
                    <Button
                      variant="default"
                      size="sm"
                      onClick={() => {
                        handleCompleteExperiment(selectedExperiment.id)
                        setIsDetailsDialogOpen(false)
                      }}
                    >
                      <CheckCircle2 className="w-4 h-4 mr-2" />
                      Complete Experiment
                    </Button>
                  )}
                  <Button variant="outline" size="sm">
                    <FileText className="w-4 h-4 mr-2" />
                    Generate Report
                  </Button>
                  <Button variant="outline" size="sm">
                    <LinkIcon className="w-4 h-4 mr-2" />
                    View Results
                  </Button>
                </div>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => handleDeleteExperiment(selectedExperiment.id)}
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
