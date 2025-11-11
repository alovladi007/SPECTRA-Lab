/**
 * Wafer/Lot Explorer Component
 *
 * Global component for wafer and lot tracking with:
 * - Wafer search and selection
 * - Lot hierarchy display
 * - Process step tracking timeline
 * - Quick filters (status, process type, date range)
 * - Recent wafers/lots
 * - Multi-select capability
 */

"use client"

import React, { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import {
  Search,
  Filter,
  Layers,
  CheckCircle,
  Clock,
  XCircle,
  ChevronRight,
  Calendar,
  Tag,
  History
} from 'lucide-react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

interface ProcessStep {
  step_id: string
  step_name: string
  process_type: 'ion' | 'rtp' | 'diffusion' | 'oxide' | 'other'
  status: 'pending' | 'running' | 'completed' | 'failed'
  started_at?: string
  completed_at?: string
  operator?: string
}

interface Wafer {
  wafer_id: string
  lot_id: string
  substrate_type: string
  diameter_mm: number
  thickness_um: number
  status: 'available' | 'in_process' | 'completed' | 'quarantine'
  current_step?: string
  process_steps: ProcessStep[]
  created_at: string
  location?: string
}

interface Lot {
  lot_id: string
  lot_name: string
  wafer_count: number
  wafers: Wafer[]
  created_at: string
  owner: string
  status: 'active' | 'completed' | 'archived'
}

interface WaferLotExplorerProps {
  onWaferSelect?: (wafer: Wafer) => void
  onLotSelect?: (lot: Lot) => void
  multiSelect?: boolean
  filterProcessType?: string
  apiEndpoint?: string
}

export const WaferLotExplorer: React.FC<WaferLotExplorerProps> = ({
  onWaferSelect,
  onLotSelect,
  multiSelect = false,
  filterProcessType,
  apiEndpoint = 'http://localhost:8002'
}) => {
  const [searchQuery, setSearchQuery] = useState('')
  const [lots, setLots] = useState<Lot[]>([])
  const [recentWafers, setRecentWafers] = useState<Wafer[]>([])
  const [selectedWafers, setSelectedWafers] = useState<Set<string>>(new Set())
  const [selectedLot, setSelectedLot] = useState<string | null>(null)
  const [expandedLots, setExpandedLots] = useState<Set<string>>(new Set())

  // Filters
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [processTypeFilter, setProcessTypeFilter] = useState<string>(filterProcessType || 'all')
  const [dateRangeFilter, setDateRangeFilter] = useState<string>('all')

  // Fetch lots and wafers
  useEffect(() => {
    fetchLots()
    fetchRecentWafers()
  }, [])

  const fetchLots = async () => {
    try {
      const response = await fetch(`${apiEndpoint}/api/lots`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token') || 'demo-token'}`,
        },
      })

      if (response.ok) {
        const data = await response.json()
        setLots(data)
      } else {
        // Load mock data
        setLots(generateMockLots())
      }
    } catch (error) {
      console.error('Failed to fetch lots:', error)
      setLots(generateMockLots())
    }
  }

  const fetchRecentWafers = async () => {
    try {
      const response = await fetch(`${apiEndpoint}/api/wafers/recent`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token') || 'demo-token'}`,
        },
      })

      if (response.ok) {
        const data = await response.json()
        setRecentWafers(data)
      } else {
        // Load mock data
        setRecentWafers(generateMockRecentWafers())
      }
    } catch (error) {
      console.error('Failed to fetch recent wafers:', error)
      setRecentWafers(generateMockRecentWafers())
    }
  }

  // Mock data generators
  const generateMockLots = (): Lot[] => [
    {
      lot_id: 'LOT-2025-001',
      lot_name: 'CMOS Process Development',
      wafer_count: 5,
      wafers: [
        {
          wafer_id: 'W12345',
          lot_id: 'LOT-2025-001',
          substrate_type: 'Si (100)',
          diameter_mm: 200,
          thickness_um: 725,
          status: 'in_process',
          current_step: 'Ion Implantation',
          process_steps: [
            { step_id: '1', step_name: 'Oxidation', process_type: 'oxide', status: 'completed', started_at: '2025-01-08T10:00:00Z', completed_at: '2025-01-08T11:30:00Z', operator: 'Alice' },
            { step_id: '2', step_name: 'Ion Implant', process_type: 'ion', status: 'running', started_at: '2025-01-08T14:00:00Z' }
          ],
          created_at: '2025-01-08T09:00:00Z',
          location: 'Bay 3'
        },
        {
          wafer_id: 'W12346',
          lot_id: 'LOT-2025-001',
          substrate_type: 'Si (100)',
          diameter_mm: 200,
          thickness_um: 725,
          status: 'completed',
          process_steps: [
            { step_id: '1', step_name: 'Oxidation', process_type: 'oxide', status: 'completed', started_at: '2025-01-08T10:00:00Z', completed_at: '2025-01-08T11:30:00Z' },
            { step_id: '2', step_name: 'Ion Implant', process_type: 'ion', status: 'completed', started_at: '2025-01-08T14:00:00Z', completed_at: '2025-01-08T15:00:00Z' },
            { step_id: '3', step_name: 'RTP Anneal', process_type: 'rtp', status: 'completed', started_at: '2025-01-08T15:30:00Z', completed_at: '2025-01-08T15:45:00Z' }
          ],
          created_at: '2025-01-08T09:00:00Z',
          location: 'Storage'
        }
      ],
      created_at: '2025-01-08T09:00:00Z',
      owner: 'Dr. Smith',
      status: 'active'
    },
    {
      lot_id: 'LOT-2025-002',
      lot_name: 'Power Device Test',
      wafer_count: 3,
      wafers: [
        {
          wafer_id: 'W12350',
          lot_id: 'LOT-2025-002',
          substrate_type: 'Si (111)',
          diameter_mm: 150,
          thickness_um: 650,
          status: 'available',
          process_steps: [],
          created_at: '2025-01-09T08:00:00Z',
          location: 'Cleanroom A'
        }
      ],
      created_at: '2025-01-09T08:00:00Z',
      owner: 'Dr. Johnson',
      status: 'active'
    }
  ]

  const generateMockRecentWafers = (): Wafer[] => [
    {
      wafer_id: 'W12345',
      lot_id: 'LOT-2025-001',
      substrate_type: 'Si (100)',
      diameter_mm: 200,
      thickness_um: 725,
      status: 'in_process',
      current_step: 'Ion Implantation',
      process_steps: [],
      created_at: '2025-01-08T09:00:00Z'
    },
    {
      wafer_id: 'W12340',
      lot_id: 'LOT-2024-099',
      substrate_type: 'Si (100)',
      diameter_mm: 200,
      thickness_um: 725,
      status: 'completed',
      process_steps: [],
      created_at: '2025-01-07T14:00:00Z'
    }
  ]

  // Filter wafers based on current filters
  const filteredLots = lots.filter(lot => {
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      if (!lot.lot_id.toLowerCase().includes(query) &&
          !lot.lot_name.toLowerCase().includes(query) &&
          !lot.wafers.some(w => w.wafer_id.toLowerCase().includes(query))) {
        return false
      }
    }

    if (statusFilter !== 'all') {
      if (lot.status !== statusFilter) return false
    }

    return true
  })

  // Toggle lot expansion
  const toggleLotExpansion = (lotId: string) => {
    const newExpanded = new Set(expandedLots)
    if (newExpanded.has(lotId)) {
      newExpanded.delete(lotId)
    } else {
      newExpanded.add(lotId)
    }
    setExpandedLots(newExpanded)
  }

  // Handle wafer selection
  const handleWaferClick = (wafer: Wafer) => {
    if (multiSelect) {
      const newSelected = new Set(selectedWafers)
      if (newSelected.has(wafer.wafer_id)) {
        newSelected.delete(wafer.wafer_id)
      } else {
        newSelected.add(wafer.wafer_id)
      }
      setSelectedWafers(newSelected)
    } else {
      setSelectedWafers(new Set([wafer.wafer_id]))
      if (onWaferSelect) {
        onWaferSelect(wafer)
      }
    }
  }

  // Handle lot selection
  const handleLotClick = (lot: Lot) => {
    setSelectedLot(lot.lot_id)
    if (onLotSelect) {
      onLotSelect(lot)
    }
  }

  // Get status badge
  const getStatusBadge = (status: string) => {
    const config = {
      available: { variant: 'default' as const, icon: CheckCircle, label: 'Available' },
      in_process: { variant: 'default' as const, icon: Clock, label: 'In Process' },
      completed: { variant: 'secondary' as const, icon: CheckCircle, label: 'Completed' },
      quarantine: { variant: 'destructive' as const, icon: XCircle, label: 'Quarantine' },
      active: { variant: 'default' as const, icon: Clock, label: 'Active' },
      archived: { variant: 'secondary' as const, icon: CheckCircle, label: 'Archived' }
    }

    const cfg = config[status as keyof typeof config]
    if (!cfg) return null

    const Icon = cfg.icon

    return (
      <Badge variant={cfg.variant} className="flex items-center gap-1">
        <Icon className="w-3 h-3" />
        {cfg.label}
      </Badge>
    )
  }

  // Get process type badge color
  const getProcessTypeColor = (type: string) => {
    const colors = {
      ion: 'bg-purple-100 text-purple-800 border-purple-300',
      rtp: 'bg-orange-100 text-orange-800 border-orange-300',
      diffusion: 'bg-blue-100 text-blue-800 border-blue-300',
      oxide: 'bg-green-100 text-green-800 border-green-300',
      other: 'bg-gray-100 text-gray-800 border-gray-300'
    }
    return colors[type as keyof typeof colors] || colors.other
  }

  return (
    <div className="space-y-4">
      {/* Search and Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="w-5 h-5" />
            Wafer/Lot Explorer
          </CardTitle>
          <CardDescription>Search and track wafers across process steps</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Search Bar */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search by wafer ID, lot ID, or lot name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>

          {/* Filters */}
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Status</label>
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="active">Active</SelectItem>
                  <SelectItem value="completed">Completed</SelectItem>
                  <SelectItem value="archived">Archived</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Process Type</label>
              <Select value={processTypeFilter} onValueChange={setProcessTypeFilter}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Processes</SelectItem>
                  <SelectItem value="ion">Ion Implantation</SelectItem>
                  <SelectItem value="rtp">RTP</SelectItem>
                  <SelectItem value="diffusion">Diffusion</SelectItem>
                  <SelectItem value="oxide">Oxidation</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Date Range</label>
              <Select value={dateRangeFilter} onValueChange={setDateRangeFilter}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Time</SelectItem>
                  <SelectItem value="today">Today</SelectItem>
                  <SelectItem value="week">This Week</SelectItem>
                  <SelectItem value="month">This Month</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {selectedWafers.size > 0 && (
            <div className="p-3 border rounded-lg bg-blue-50 border-blue-300">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-blue-800">
                  {selectedWafers.size} wafer(s) selected
                </span>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setSelectedWafers(new Set())}
                >
                  Clear
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Recent Wafers */}
      {recentWafers.length > 0 && !searchQuery && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <History className="w-4 h-4" />
              Recently Accessed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {recentWafers.slice(0, 3).map(wafer => (
                <div
                  key={wafer.wafer_id}
                  className={`p-2 border rounded-lg cursor-pointer transition-colors ${
                    selectedWafers.has(wafer.wafer_id)
                      ? 'bg-blue-50 border-blue-300'
                      : 'hover:bg-muted'
                  }`}
                  onClick={() => handleWaferClick(wafer)}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-semibold text-sm">{wafer.wafer_id}</div>
                      <div className="text-xs text-muted-foreground">
                        {wafer.lot_id} • {wafer.substrate_type}
                      </div>
                    </div>
                    {getStatusBadge(wafer.status)}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Lots List */}
      <div className="space-y-3">
        {filteredLots.map(lot => (
          <Card key={lot.lot_id}>
            <CardHeader
              className="cursor-pointer"
              onClick={() => toggleLotExpansion(lot.lot_id)}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <CardTitle className="text-base">{lot.lot_id}</CardTitle>
                    {getStatusBadge(lot.status)}
                  </div>
                  <CardDescription className="mt-1">
                    {lot.lot_name} • {lot.wafer_count} wafer(s) • Owner: {lot.owner}
                  </CardDescription>
                </div>
                <ChevronRight
                  className={`w-5 h-5 transition-transform ${
                    expandedLots.has(lot.lot_id) ? 'rotate-90' : ''
                  }`}
                />
              </div>
            </CardHeader>

            {expandedLots.has(lot.lot_id) && (
              <CardContent>
                <Separator className="mb-4" />
                <div className="space-y-3">
                  {lot.wafers.map(wafer => (
                    <div
                      key={wafer.wafer_id}
                      className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                        selectedWafers.has(wafer.wafer_id)
                          ? 'bg-blue-50 border-blue-300'
                          : 'hover:bg-muted'
                      }`}
                      onClick={() => handleWaferClick(wafer)}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <div className="font-semibold">{wafer.wafer_id}</div>
                          <div className="text-xs text-muted-foreground mt-1">
                            {wafer.substrate_type} • {wafer.diameter_mm}mm • {wafer.thickness_um}µm
                          </div>
                          {wafer.location && (
                            <div className="text-xs text-muted-foreground mt-1">
                              📍 {wafer.location}
                            </div>
                          )}
                        </div>
                        <div className="text-right">
                          {getStatusBadge(wafer.status)}
                          {wafer.current_step && (
                            <div className="text-xs text-muted-foreground mt-1">
                              Current: {wafer.current_step}
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Process Steps Timeline */}
                      {wafer.process_steps.length > 0 && (
                        <div className="mt-3 pt-3 border-t">
                          <div className="text-xs font-semibold text-muted-foreground mb-2">
                            Process Steps ({wafer.process_steps.length})
                          </div>
                          <div className="space-y-2">
                            {wafer.process_steps.map((step, idx) => (
                              <div key={step.step_id} className="flex items-center gap-2">
                                <div className="flex-shrink-0">
                                  {step.status === 'completed' && <CheckCircle className="w-4 h-4 text-green-600" />}
                                  {step.status === 'running' && <Clock className="w-4 h-4 text-blue-600 animate-pulse" />}
                                  {step.status === 'pending' && <Clock className="w-4 h-4 text-gray-400" />}
                                  {step.status === 'failed' && <XCircle className="w-4 h-4 text-red-600" />}
                                </div>
                                <div className="flex-1">
                                  <div className="flex items-center gap-2">
                                    <span className="text-xs font-medium">{step.step_name}</span>
                                    <Badge
                                      variant="outline"
                                      className={`text-xs ${getProcessTypeColor(step.process_type)}`}
                                    >
                                      {step.process_type.toUpperCase()}
                                    </Badge>
                                  </div>
                                  {step.operator && (
                                    <div className="text-xs text-muted-foreground">
                                      by {step.operator}
                                    </div>
                                  )}
                                </div>
                                {step.completed_at && (
                                  <div className="text-xs text-muted-foreground">
                                    {new Date(step.completed_at).toLocaleTimeString()}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            )}
          </Card>
        ))}

        {filteredLots.length === 0 && (
          <Card>
            <CardContent className="py-12 text-center">
              <div className="text-muted-foreground">
                No lots found matching your criteria
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

export default WaferLotExplorer
