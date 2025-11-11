/**
 * ELN/LIMS Integration Component
 *
 * Provides bidirectional integration with Electronic Lab Notebook (ELN)
 * and Laboratory Information Management System (LIMS):
 * - Auto-create ELN entry summaries for completed runs
 * - Link custody chain for wafers and devices
 * - Reference recipes and e-signatures
 * - Sync sample tracking and metadata
 * - Support for multiple ELN/LIMS platforms (LabArchives, Benchling, LabWare)
 */

"use client"

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { Checkbox } from '@/components/ui/checkbox'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from '@/components/ui/dialog'
import {
  BookOpen,
  Link2,
  CheckCircle2,
  AlertCircle,
  Loader2,
  ExternalLink,
  Users,
  FileCheck,
  ArrowRight,
  Beaker
} from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert'

interface ELNPlatform {
  id: string
  name: string
  endpoint: string
  supported_features: string[]
}

interface ELNEntry {
  entry_id: string
  title: string
  created_at: string
  created_by: string
  url: string
  linked_items: string[]
}

interface CustodyRecord {
  item_id: string
  item_type: 'wafer' | 'device' | 'sample' | 'lot'
  current_custodian: string
  location: string
  status: 'available' | 'in_use' | 'consumed' | 'quarantine'
  last_transferred: string
}

interface RecipeReference {
  recipe_id: string
  recipe_name: string
  version: string
  approved_by: string
  approved_at: string
  signature_hash: string
}

interface ELNLIMSIntegrationProps {
  runId: string
  processType: 'ion' | 'rtp'
  waferId?: string
  onEntryCreated?: (entryId: string) => void
  onCustodyLinked?: (custodyId: string) => void
}

const availablePlatforms: ELNPlatform[] = [
  {
    id: 'labarchives',
    name: 'LabArchives',
    endpoint: 'https://api.labarchives.com',
    supported_features: ['entries', 'attachments', 'custody', 'signatures']
  },
  {
    id: 'benchling',
    name: 'Benchling',
    endpoint: 'https://api.benchling.com',
    supported_features: ['entries', 'samples', 'workflows', 'custody']
  },
  {
    id: 'labware',
    name: 'LabWare LIMS',
    endpoint: 'http://localhost:8002',
    supported_features: ['samples', 'custody', 'qc', 'coa']
  },
  {
    id: 'local',
    name: 'SPECTRA Lab (Local)',
    endpoint: 'http://localhost:8002',
    supported_features: ['entries', 'custody', 'recipes', 'signatures', 'samples']
  }
]

export const ELNLIMSIntegration: React.FC<ELNLIMSIntegrationProps> = ({
  runId,
  processType,
  waferId,
  onEntryCreated,
  onCustodyLinked
}) => {
  const [isOpen, setIsOpen] = useState(false)
  const [selectedPlatform, setSelectedPlatform] = useState<string>('local')
  const [isProcessing, setIsProcessing] = useState(false)
  const [statusMessage, setStatusMessage] = useState<string | null>(null)

  // ELN Entry Configuration
  const [entryConfig, setEntryConfig] = useState({
    autoCreate: true,
    includeParameters: true,
    includeResults: true,
    includeSPC: true,
    includeVM: true,
    includeImages: true,
    notifyCollaborators: false
  })

  const [entryMetadata, setEntryMetadata] = useState({
    title: `${processType.toUpperCase()} Run ${runId}`,
    project: '',
    notebook: '',
    tags: '',
    collaborators: ''
  })

  // Custody Chain Configuration
  const [custodyConfig, setCustodyConfig] = useState({
    linkWafers: true,
    autoTransfer: false,
    updateLocation: true,
    notifyQC: false
  })

  const [custodyData, setCustodyData] = useState({
    currentCustodian: '',
    location: '',
    condition: 'good' as 'good' | 'fair' | 'damaged',
    notes: ''
  })

  // Recipe References
  const [recipeConfig, setRecipeConfig] = useState({
    linkRecipe: true,
    includeSignatures: true,
    includeApprovals: true
  })

  // Mock data - in production, fetch from API
  const [linkedEntries, setLinkedEntries] = useState<ELNEntry[]>([])
  const [custodyRecords, setCustodyRecords] = useState<CustodyRecord[]>([
    {
      item_id: waferId || 'W-2025-001',
      item_type: 'wafer',
      current_custodian: 'sarah.operator',
      location: 'FAB-101, Bay 3',
      status: 'in_use',
      last_transferred: '2025-01-20T14:40:00Z'
    }
  ])
  const [recipeReferences, setRecipeReferences] = useState<RecipeReference[]>([
    {
      recipe_id: 'RCP-2025-001',
      recipe_name: processType === 'ion' ? 'Phosphorus Implant Standard' : 'Dopant Activation Anneal',
      version: '2.1',
      approved_by: 'john.engineer',
      approved_at: '2025-01-15T10:30:00Z',
      signature_hash: 'SHA256:a3f5b8c2d9e1...'
    }
  ])

  const platform = availablePlatforms.find(p => p.id === selectedPlatform)

  const createELNEntry = async () => {
    if (!entryMetadata.title.trim()) {
      alert('Please enter an entry title')
      return
    }

    setIsProcessing(true)
    setStatusMessage('Creating ELN entry...')

    try {
      // Build entry payload
      const entryPayload = {
        platform: selectedPlatform,
        title: entryMetadata.title,
        project: entryMetadata.project,
        notebook: entryMetadata.notebook,
        tags: entryMetadata.tags.split(',').map(t => t.trim()).filter(t => t),
        content: buildEntryContent(),
        collaborators: entryMetadata.collaborators.split(',').map(c => c.trim()).filter(c => c),
        metadata: {
          run_id: runId,
          process_type: processType,
          wafer_id: waferId,
          created_by: 'system',
          timestamp: new Date().toISOString()
        }
      }

      // Simulate API call
      // In production: POST /api/eln/entries
      await new Promise(resolve => setTimeout(resolve, 1500))

      const newEntry: ELNEntry = {
        entry_id: `ELN-${Date.now()}`,
        title: entryMetadata.title,
        created_at: new Date().toISOString(),
        created_by: 'system',
        url: `${platform?.endpoint}/entries/${Date.now()}`,
        linked_items: [runId, ...(waferId ? [waferId] : [])]
      }

      setLinkedEntries(prev => [...prev, newEntry])
      setStatusMessage(`ELN entry created: ${newEntry.entry_id}`)
      onEntryCreated?.(newEntry.entry_id)

      setTimeout(() => setStatusMessage(null), 3000)

    } catch (error) {
      console.error('ELN entry creation failed:', error)
      setStatusMessage('Error: Failed to create ELN entry')
    } finally {
      setIsProcessing(false)
    }
  }

  const buildEntryContent = (): string => {
    let content = `# ${entryMetadata.title}\n\n`
    content += `**Run ID:** ${runId}\n`
    content += `**Process:** ${processType.toUpperCase()}\n`
    if (waferId) content += `**Wafer ID:** ${waferId}\n`
    content += `**Timestamp:** ${new Date().toLocaleString()}\n\n`

    if (entryConfig.includeParameters) {
      content += `## Process Parameters\n\n`
      content += processType === 'ion'
        ? `- Species: Phosphorus (P)\n- Energy: 40 keV\n- Dose: 5×10¹⁴ cm⁻²\n- Tilt: 7°\n`
        : `- Peak Temp: 1050°C\n- Duration: 120s\n- Ramp Rate (up): 75°C/s\n- Thermal Budget: 42,000°C·s\n`
      content += `\n`
    }

    if (entryConfig.includeResults) {
      content += `## Results Summary\n\n`
      content += `- Status: Completed successfully\n`
      content += `- Quality: All SPC metrics in control\n`
      content += `- Uniformity: 98.5%\n\n`
    }

    if (entryConfig.includeSPC) {
      content += `## Statistical Process Control\n\n`
      content += `| Metric | Value | Cpk |\n`
      content += `|--------|-------|-----|\n`
      content += `| Dose Uniformity | 98.5% | 1.67 |\n`
      content += `| Energy Stability | 40.02 keV | 1.89 |\n\n`
    }

    if (entryConfig.includeVM) {
      content += `## Virtual Metrology Predictions\n\n`
      content += `| Parameter | Predicted | Actual | Error |\n`
      content += `|-----------|-----------|--------|-------|\n`
      content += `| Sheet Resistance | 125.3 Ω/□ | 123.8 Ω/□ | +1.2% |\n`
      content += `| Junction Depth | 0.285 μm | 0.291 μm | -2.1% |\n\n`
    }

    if (recipeConfig.linkRecipe && recipeReferences.length > 0) {
      const recipe = recipeReferences[0]
      content += `## Recipe Reference\n\n`
      content += `- **Recipe:** ${recipe.recipe_name} (v${recipe.version})\n`
      content += `- **ID:** ${recipe.recipe_id}\n`
      if (recipeConfig.includeApprovals) {
        content += `- **Approved By:** ${recipe.approved_by}\n`
        content += `- **Approved At:** ${recipe.approved_at}\n`
      }
      if (recipeConfig.includeSignatures) {
        content += `- **Signature:** ${recipe.signature_hash}\n`
      }
      content += `\n`
    }

    if (custodyConfig.linkWafers && custodyRecords.length > 0) {
      content += `## Sample Custody\n\n`
      custodyRecords.forEach(record => {
        content += `- **${record.item_type.toUpperCase()} ${record.item_id}**\n`
        content += `  - Custodian: ${record.current_custodian}\n`
        content += `  - Location: ${record.location}\n`
        content += `  - Status: ${record.status}\n`
      })
    }

    return content
  }

  const linkCustodyChain = async () => {
    if (!custodyData.currentCustodian.trim()) {
      alert('Please enter current custodian')
      return
    }

    setIsProcessing(true)
    setStatusMessage('Updating custody chain...')

    try {
      const custodyPayload = {
        run_id: runId,
        wafer_id: waferId,
        custodian: custodyData.currentCustodian,
        location: custodyData.location,
        condition: custodyData.condition,
        notes: custodyData.notes,
        timestamp: new Date().toISOString()
      }

      // Simulate API call
      // In production: POST /api/lims/custody
      await new Promise(resolve => setTimeout(resolve, 1000))

      // Update custody record
      setCustodyRecords(prev => prev.map(record =>
        record.item_id === waferId
          ? {
              ...record,
              current_custodian: custodyData.currentCustodian,
              location: custodyData.location,
              last_transferred: new Date().toISOString()
            }
          : record
      ))

      setStatusMessage('Custody chain updated successfully')
      onCustodyLinked?.(`CUSTODY-${Date.now()}`)

      setTimeout(() => setStatusMessage(null), 3000)

    } catch (error) {
      console.error('Custody update failed:', error)
      setStatusMessage('Error: Failed to update custody')
    } finally {
      setIsProcessing(false)
    }
  }

  const syncWithLIMS = async () => {
    setIsProcessing(true)
    setStatusMessage('Syncing with LIMS...')

    try {
      // Simulate comprehensive sync
      await new Promise(resolve => setTimeout(resolve, 2000))

      // Would sync: sample tracking, QC results, COA generation, inventory updates
      setStatusMessage('LIMS sync completed successfully')

      setTimeout(() => setStatusMessage(null), 3000)

    } catch (error) {
      console.error('LIMS sync failed:', error)
      setStatusMessage('Error: LIMS sync failed')
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" className="gap-2">
          <BookOpen className="w-4 h-4" />
          ELN/LIMS Integration
        </Button>
      </DialogTrigger>

      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <BookOpen className="w-5 h-5" />
            ELN/LIMS Integration
          </DialogTitle>
          <DialogDescription>
            Automatically create ELN entries, link custody chains, and sync with LIMS for run {runId}
          </DialogDescription>
        </DialogHeader>

        {statusMessage && (
          <Alert className={statusMessage.startsWith('Error') ? 'border-red-500' : 'bg-blue-50 border-blue-500'}>
            {statusMessage.startsWith('Error') ? (
              <AlertCircle className="h-4 w-4 text-red-600" />
            ) : (
              <CheckCircle2 className="h-4 w-4 text-blue-600" />
            )}
            <AlertDescription className={statusMessage.startsWith('Error') ? 'text-red-800' : 'text-blue-800'}>
              {statusMessage}
            </AlertDescription>
          </Alert>
        )}

        <div className="space-y-6">
          {/* Platform Selection */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Select ELN/LIMS Platform</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label>Platform</Label>
                <Select value={selectedPlatform} onValueChange={setSelectedPlatform}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {availablePlatforms.map(p => (
                      <SelectItem key={p.id} value={p.id}>
                        {p.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {platform && (
                <div className="p-3 border rounded bg-muted/30 text-xs space-y-2">
                  <div><strong>Endpoint:</strong> {platform.endpoint}</div>
                  <div>
                    <strong>Supported Features:</strong>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {platform.supported_features.map(feature => (
                        <Badge key={feature} variant="secondary" className="text-xs">
                          {feature}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Separator />

          {/* ELN Entry Creation */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <BookOpen className="w-4 h-4" />
                ELN Entry Creation
              </CardTitle>
              <CardDescription>Auto-generate electronic lab notebook entries</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Entry Title *</Label>
                  <Input
                    value={entryMetadata.title}
                    onChange={(e) => setEntryMetadata(prev => ({ ...prev, title: e.target.value }))}
                  />
                </div>
                <div>
                  <Label>Project</Label>
                  <Input
                    value={entryMetadata.project}
                    onChange={(e) => setEntryMetadata(prev => ({ ...prev, project: e.target.value }))}
                    placeholder="Project name"
                  />
                </div>
                <div>
                  <Label>Notebook</Label>
                  <Input
                    value={entryMetadata.notebook}
                    onChange={(e) => setEntryMetadata(prev => ({ ...prev, notebook: e.target.value }))}
                    placeholder="Notebook ID"
                  />
                </div>
                <div>
                  <Label>Tags (comma-separated)</Label>
                  <Input
                    value={entryMetadata.tags}
                    onChange={(e) => setEntryMetadata(prev => ({ ...prev, tags: e.target.value }))}
                    placeholder="process, ion-implant, production"
                  />
                </div>
              </div>

              <div>
                <Label>Collaborators (comma-separated)</Label>
                <Input
                  value={entryMetadata.collaborators}
                  onChange={(e) => setEntryMetadata(prev => ({ ...prev, collaborators: e.target.value }))}
                  placeholder="john.engineer, sarah.operator"
                />
              </div>

              <div className="space-y-2">
                <Label className="font-semibold">Include in Entry:</Label>
                <div className="grid grid-cols-2 gap-2">
                  <div className="flex items-center gap-2">
                    <Checkbox
                      id="includeParameters"
                      checked={entryConfig.includeParameters}
                      onCheckedChange={(checked) =>
                        setEntryConfig(prev => ({ ...prev, includeParameters: checked as boolean }))
                      }
                    />
                    <Label htmlFor="includeParameters" className="text-sm">Process Parameters</Label>
                  </div>
                  <div className="flex items-center gap-2">
                    <Checkbox
                      id="includeResults"
                      checked={entryConfig.includeResults}
                      onCheckedChange={(checked) =>
                        setEntryConfig(prev => ({ ...prev, includeResults: checked as boolean }))
                      }
                    />
                    <Label htmlFor="includeResults" className="text-sm">Results Summary</Label>
                  </div>
                  <div className="flex items-center gap-2">
                    <Checkbox
                      id="includeSPC"
                      checked={entryConfig.includeSPC}
                      onCheckedChange={(checked) =>
                        setEntryConfig(prev => ({ ...prev, includeSPC: checked as boolean }))
                      }
                    />
                    <Label htmlFor="includeSPC" className="text-sm">SPC Metrics</Label>
                  </div>
                  <div className="flex items-center gap-2">
                    <Checkbox
                      id="includeVM"
                      checked={entryConfig.includeVM}
                      onCheckedChange={(checked) =>
                        setEntryConfig(prev => ({ ...prev, includeVM: checked as boolean }))
                      }
                    />
                    <Label htmlFor="includeVM" className="text-sm">VM Predictions</Label>
                  </div>
                  <div className="flex items-center gap-2">
                    <Checkbox
                      id="includeImages"
                      checked={entryConfig.includeImages}
                      onCheckedChange={(checked) =>
                        setEntryConfig(prev => ({ ...prev, includeImages: checked as boolean }))
                      }
                    />
                    <Label htmlFor="includeImages" className="text-sm">Images/Plots</Label>
                  </div>
                </div>
              </div>

              <Button onClick={createELNEntry} disabled={isProcessing} className="w-full">
                {isProcessing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Creating Entry...
                  </>
                ) : (
                  <>
                    <BookOpen className="w-4 h-4 mr-2" />
                    Create ELN Entry
                  </>
                )}
              </Button>

              {/* Linked Entries */}
              {linkedEntries.length > 0 && (
                <div className="mt-4 space-y-2">
                  <Label className="font-semibold">Linked ELN Entries:</Label>
                  {linkedEntries.map(entry => (
                    <div key={entry.entry_id} className="flex items-center justify-between p-2 border rounded bg-green-50">
                      <div className="flex-1">
                        <div className="font-medium text-sm">{entry.title}</div>
                        <div className="text-xs text-muted-foreground">
                          {entry.entry_id} • Created {new Date(entry.created_at).toLocaleString()}
                        </div>
                      </div>
                      <Button variant="ghost" size="sm" asChild>
                        <a href={entry.url} target="_blank" rel="noopener noreferrer">
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Separator />

          {/* Custody Chain Linking */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Link2 className="w-4 h-4" />
                Custody Chain Linking
              </CardTitle>
              <CardDescription>Track wafer/device custody and location</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Current Custodian *</Label>
                  <Input
                    value={custodyData.currentCustodian}
                    onChange={(e) => setCustodyData(prev => ({ ...prev, currentCustodian: e.target.value }))}
                    placeholder="username or full name"
                  />
                </div>
                <div>
                  <Label>Location</Label>
                  <Input
                    value={custodyData.location}
                    onChange={(e) => setCustodyData(prev => ({ ...prev, location: e.target.value }))}
                    placeholder="FAB-101, Bay 3"
                  />
                </div>
              </div>

              <div>
                <Label>Condition</Label>
                <Select
                  value={custodyData.condition}
                  onValueChange={(value: any) => setCustodyData(prev => ({ ...prev, condition: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="good">Good</SelectItem>
                    <SelectItem value="fair">Fair</SelectItem>
                    <SelectItem value="damaged">Damaged</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Transfer Notes</Label>
                <Textarea
                  value={custodyData.notes}
                  onChange={(e) => setCustodyData(prev => ({ ...prev, notes: e.target.value }))}
                  placeholder="Any notes about the custody transfer..."
                  rows={3}
                />
              </div>

              <Button onClick={linkCustodyChain} disabled={isProcessing} className="w-full">
                {isProcessing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Updating Custody...
                  </>
                ) : (
                  <>
                    <Link2 className="w-4 h-4 mr-2" />
                    Update Custody Chain
                  </>
                )}
              </Button>

              {/* Current Custody Records */}
              <div className="mt-4 space-y-2">
                <Label className="font-semibold">Current Custody Records:</Label>
                {custodyRecords.map(record => (
                  <div key={record.item_id} className="p-3 border rounded bg-muted/30">
                    <div className="flex items-center justify-between mb-2">
                      <div className="font-medium">
                        {record.item_type.toUpperCase()} {record.item_id}
                      </div>
                      <Badge
                        variant={
                          record.status === 'available' ? 'default' :
                          record.status === 'in_use' ? 'secondary' :
                          record.status === 'quarantine' ? 'destructive' : 'outline'
                        }
                      >
                        {record.status}
                      </Badge>
                    </div>
                    <div className="text-xs space-y-1 text-muted-foreground">
                      <div className="flex items-center gap-2">
                        <Users className="w-3 h-3" />
                        Custodian: {record.current_custodian}
                      </div>
                      <div className="flex items-center gap-2">
                        <Beaker className="w-3 h-3" />
                        Location: {record.location}
                      </div>
                      <div>Last Transfer: {new Date(record.last_transferred).toLocaleString()}</div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Separator />

          {/* Recipe & Signature References */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <FileCheck className="w-4 h-4" />
                Recipe & Signature References
              </CardTitle>
              <CardDescription>Link approved recipes and e-signatures</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="linkRecipe"
                    checked={recipeConfig.linkRecipe}
                    onCheckedChange={(checked) =>
                      setRecipeConfig(prev => ({ ...prev, linkRecipe: checked as boolean }))
                    }
                  />
                  <Label htmlFor="linkRecipe">Link recipe to ELN entry</Label>
                </div>
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="includeSignatures"
                    checked={recipeConfig.includeSignatures}
                    onCheckedChange={(checked) =>
                      setRecipeConfig(prev => ({ ...prev, includeSignatures: checked as boolean }))
                    }
                  />
                  <Label htmlFor="includeSignatures">Include e-signature hashes</Label>
                </div>
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="includeApprovals"
                    checked={recipeConfig.includeApprovals}
                    onCheckedChange={(checked) =>
                      setRecipeConfig(prev => ({ ...prev, includeApprovals: checked as boolean }))
                    }
                  />
                  <Label htmlFor="includeApprovals">Include approval chain</Label>
                </div>
              </div>

              <div className="space-y-2 mt-4">
                <Label className="font-semibold">Linked Recipes:</Label>
                {recipeReferences.map(recipe => (
                  <div key={recipe.recipe_id} className="p-3 border rounded bg-blue-50">
                    <div className="font-medium">{recipe.recipe_name} (v{recipe.version})</div>
                    <div className="text-xs text-muted-foreground mt-1 space-y-1">
                      <div>Recipe ID: {recipe.recipe_id}</div>
                      <div>Approved by: {recipe.approved_by} on {new Date(recipe.approved_at).toLocaleString()}</div>
                      {recipeConfig.includeSignatures && (
                        <div className="font-mono text-xs mt-2">
                          Signature: {recipe.signature_hash}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Separator />

          {/* LIMS Sync */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">LIMS Synchronization</CardTitle>
              <CardDescription>Sync sample tracking, QC results, and inventory</CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={syncWithLIMS} disabled={isProcessing} variant="outline" className="w-full">
                {isProcessing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Syncing...
                  </>
                ) : (
                  <>
                    <ArrowRight className="w-4 h-4 mr-2" />
                    Sync with LIMS
                  </>
                )}
              </Button>
              <p className="text-xs text-muted-foreground mt-2">
                Syncs: sample tracking, QC results, COA generation, inventory updates
              </p>
            </CardContent>
          </Card>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setIsOpen(false)} disabled={isProcessing}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default ELNLIMSIntegration
