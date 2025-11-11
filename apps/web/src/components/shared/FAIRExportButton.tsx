/**
 * FAIR Export Button Component
 *
 * FAIR (Findable, Accessible, Interoperable, Reusable) compliant data export:
 * - Format selection (JSON-LD, CSV, HDF5, RDF)
 * - DOI and ORCID metadata fields
 * - License selection (CC-BY, CC0, MIT, etc.)
 * - Provenance chain tracking
 * - Data citation generation
 * - Metadata schema validation
 */

"use client"

import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from '@/components/ui/dialog'
import { Separator } from '@/components/ui/separator'
import {
  Download,
  FileText,
  Database,
  ExternalLink,
  CheckCircle,
  AlertCircle,
  Info
} from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert'

type ExportFormat = 'json-ld' | 'csv' | 'hdf5' | 'rdf' | 'parquet'

interface License {
  id: string
  name: string
  url: string
  description: string
}

interface FAIRMetadata {
  // Findable
  doi?: string
  title: string
  keywords: string[]

  // Accessible
  format: ExportFormat
  license: string
  access_rights: 'open' | 'embargoed' | 'restricted' | 'closed'

  // Interoperable
  schema_version: string
  vocabulary: string[]

  // Reusable
  creator: string
  creator_orcid?: string
  contributor?: string
  created_date: string
  modified_date?: string
  provenance: string[]
  funding?: string
  related_identifiers?: string[]
}

interface FAIRExportButtonProps {
  datasetId: string
  datasetType: 'ion_run' | 'rtp_run' | 'wafer' | 'lot' | 'analysis'
  defaultTitle?: string
  onExport?: (metadata: FAIRMetadata) => void
  apiEndpoint?: string
}

const availableLicenses: License[] = [
  {
    id: 'cc-by-4.0',
    name: 'CC BY 4.0',
    url: 'https://creativecommons.org/licenses/by/4.0/',
    description: 'Attribution required, allows commercial use'
  },
  {
    id: 'cc0-1.0',
    name: 'CC0 1.0',
    url: 'https://creativecommons.org/publicdomain/zero/1.0/',
    description: 'Public domain dedication, no rights reserved'
  },
  {
    id: 'mit',
    name: 'MIT License',
    url: 'https://opensource.org/licenses/MIT',
    description: 'Permissive, allows commercial use'
  },
  {
    id: 'apache-2.0',
    name: 'Apache 2.0',
    url: 'https://www.apache.org/licenses/LICENSE-2.0',
    description: 'Permissive with patent grant'
  }
]

export const FAIRExportButton: React.FC<FAIRExportButtonProps> = ({
  datasetId,
  datasetType,
  defaultTitle,
  onExport,
  apiEndpoint = 'http://localhost:8002'
}) => {
  const [isOpen, setIsOpen] = useState(false)
  const [isExporting, setIsExporting] = useState(false)
  const [exportSuccess, setExportSuccess] = useState(false)

  // FAIR Metadata State
  const [format, setFormat] = useState<ExportFormat>('json-ld')
  const [title, setTitle] = useState(defaultTitle || '')
  const [keywords, setKeywords] = useState<string>('')
  const [doi, setDoi] = useState<string>('')
  const [license, setLicense] = useState<string>('cc-by-4.0')
  const [accessRights, setAccessRights] = useState<FAIRMetadata['access_rights']>('open')
  const [creator, setCreator] = useState<string>('')
  const [creatorOrcid, setCreatorOrcid] = useState<string>('')
  const [contributor, setContributor] = useState<string>('')
  const [funding, setFunding] = useState<string>('')
  const [relatedIdentifiers, setRelatedIdentifiers] = useState<string>('')

  // Validation
  const [validationErrors, setValidationErrors] = useState<string[]>([])

  const validateMetadata = (): boolean => {
    const errors: string[] = []

    if (!title.trim()) {
      errors.push('Title is required')
    }

    if (!creator.trim()) {
      errors.push('Creator/Author is required')
    }

    if (creatorOrcid && !creatorOrcid.match(/^\d{4}-\d{4}-\d{4}-\d{3}[0-9X]$/)) {
      errors.push('Invalid ORCID format (should be XXXX-XXXX-XXXX-XXXX)')
    }

    if (doi && !doi.match(/^10\.\d+\/.+$/)) {
      errors.push('Invalid DOI format (should start with 10.)')
    }

    setValidationErrors(errors)
    return errors.length === 0
  }

  const handleExport = async () => {
    if (!validateMetadata()) {
      return
    }

    setIsExporting(true)

    // Build FAIR metadata
    const metadata: FAIRMetadata = {
      doi: doi || undefined,
      title,
      keywords: keywords.split(',').map(k => k.trim()).filter(k => k),
      format,
      license,
      access_rights: accessRights,
      schema_version: '1.0.0',
      vocabulary: ['schema.org', 'dublin-core', 'datacite'],
      creator,
      creator_orcid: creatorOrcid || undefined,
      contributor: contributor || undefined,
      created_date: new Date().toISOString(),
      provenance: [
        `Generated from ${datasetType} ${datasetId}`,
        'SPECTRA Lab Data Management System',
        `Exported on ${new Date().toLocaleString()}`
      ],
      funding: funding || undefined,
      related_identifiers: relatedIdentifiers
        ? relatedIdentifiers.split(',').map(id => id.trim()).filter(id => id)
        : undefined
    }

    try {
      const response = await fetch(`${apiEndpoint}/api/export/fair`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token') || 'demo-token'}`,
        },
        body: JSON.stringify({
          dataset_id: datasetId,
          dataset_type: datasetType,
          metadata
        }),
      })

      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `${datasetId}_FAIR.${format === 'json-ld' ? 'json' : format}`
        a.click()
        window.URL.revokeObjectURL(url)

        setExportSuccess(true)

        if (onExport) {
          onExport(metadata)
        }

        setTimeout(() => {
          setIsOpen(false)
          setExportSuccess(false)
        }, 2000)
      } else {
        throw new Error('Export failed')
      }
    } catch (error) {
      console.error('FAIR export failed:', error)
      alert('Export failed. Please try again.')
    } finally {
      setIsExporting(false)
    }
  }

  const getFormatDescription = (fmt: ExportFormat): string => {
    const descriptions = {
      'json-ld': 'JSON-LD with schema.org vocabulary - Best for web interoperability',
      'csv': 'CSV with metadata header - Simple tabular format',
      'hdf5': 'HDF5 with embedded metadata - Best for large numerical datasets',
      'rdf': 'RDF/XML semantic triple format - Best for linked data',
      'parquet': 'Apache Parquet columnar format - Best for big data analytics'
    }
    return descriptions[fmt]
  }

  const selectedLicense = availableLicenses.find(l => l.id === license)

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" className="gap-2">
          <Download className="w-4 h-4" />
          FAIR Export
          <Badge variant="secondary" className="ml-1 text-xs">
            F.A.I.R.
          </Badge>
        </Button>
      </DialogTrigger>

      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Database className="w-5 h-5" />
            FAIR-Compliant Data Export
          </DialogTitle>
          <DialogDescription>
            Export dataset with FAIR principles: Findable, Accessible, Interoperable, Reusable
          </DialogDescription>
        </DialogHeader>

        {exportSuccess && (
          <Alert className="bg-green-50 border-green-500">
            <CheckCircle className="h-4 w-4 text-green-600" />
            <AlertDescription className="text-green-800">
              Export successful! Download started.
            </AlertDescription>
          </Alert>
        )}

        {validationErrors.length > 0 && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              <ul className="text-xs list-disc list-inside">
                {validationErrors.map((error, idx) => (
                  <li key={idx}>{error}</li>
                ))}
              </ul>
            </AlertDescription>
          </Alert>
        )}

        <div className="space-y-6">
          {/* Format Selection */}
          <div>
            <Label htmlFor="format" className="font-semibold">Export Format *</Label>
            <Select value={format} onValueChange={(value) => setFormat(value as ExportFormat)}>
              <SelectTrigger id="format">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="json-ld">JSON-LD (Linked Data)</SelectItem>
                <SelectItem value="csv">CSV (Comma-Separated Values)</SelectItem>
                <SelectItem value="hdf5">HDF5 (Hierarchical Data Format)</SelectItem>
                <SelectItem value="rdf">RDF/XML (Resource Description Framework)</SelectItem>
                <SelectItem value="parquet">Apache Parquet</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground mt-1">
              {getFormatDescription(format)}
            </p>
          </div>

          <Separator />

          {/* Findable */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <Info className="w-4 h-4 text-blue-600" />
              Findable
            </h3>

            <div>
              <Label htmlFor="title">Title *</Label>
              <Input
                id="title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Descriptive title for the dataset"
              />
            </div>

            <div>
              <Label htmlFor="doi">DOI (Digital Object Identifier)</Label>
              <Input
                id="doi"
                value={doi}
                onChange={(e) => setDoi(e.target.value)}
                placeholder="10.1234/example"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Optional. Format: 10.XXXX/YYYY
              </p>
            </div>

            <div>
              <Label htmlFor="keywords">Keywords</Label>
              <Input
                id="keywords"
                value={keywords}
                onChange={(e) => setKeywords(e.target.value)}
                placeholder="ion implantation, semiconductors, doping"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Comma-separated list of keywords
              </p>
            </div>
          </div>

          <Separator />

          {/* Accessible */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <Info className="w-4 h-4 text-green-600" />
              Accessible
            </h3>

            <div>
              <Label htmlFor="license">License *</Label>
              <Select value={license} onValueChange={setLicense}>
                <SelectTrigger id="license">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {availableLicenses.map(lic => (
                    <SelectItem key={lic.id} value={lic.id}>
                      {lic.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedLicense && (
                <div className="mt-2 text-xs p-2 border rounded bg-muted/30">
                  <div className="font-medium mb-1">{selectedLicense.name}</div>
                  <div className="text-muted-foreground">{selectedLicense.description}</div>
                  <a
                    href={selectedLicense.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:underline flex items-center gap-1 mt-1"
                  >
                    View License <ExternalLink className="w-3 h-3" />
                  </a>
                </div>
              )}
            </div>

            <div>
              <Label htmlFor="access_rights">Access Rights *</Label>
              <Select value={accessRights} onValueChange={(value) => setAccessRights(value as any)}>
                <SelectTrigger id="access_rights">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="open">Open - Publicly accessible</SelectItem>
                  <SelectItem value="embargoed">Embargoed - Accessible after date</SelectItem>
                  <SelectItem value="restricted">Restricted - Access on request</SelectItem>
                  <SelectItem value="closed">Closed - Private</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <Separator />

          {/* Reusable */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <Info className="w-4 h-4 text-purple-600" />
              Reusable
            </h3>

            <div>
              <Label htmlFor="creator">Creator/Author *</Label>
              <Input
                id="creator"
                value={creator}
                onChange={(e) => setCreator(e.target.value)}
                placeholder="Full name of dataset creator"
              />
            </div>

            <div>
              <Label htmlFor="creator_orcid">Creator ORCID iD</Label>
              <Input
                id="creator_orcid"
                value={creatorOrcid}
                onChange={(e) => setCreatorOrcid(e.target.value)}
                placeholder="0000-0002-1825-0097"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Optional. Format: XXXX-XXXX-XXXX-XXXX
              </p>
            </div>

            <div>
              <Label htmlFor="contributor">Contributors</Label>
              <Input
                id="contributor"
                value={contributor}
                onChange={(e) => setContributor(e.target.value)}
                placeholder="Additional contributors (comma-separated)"
              />
            </div>

            <div>
              <Label htmlFor="funding">Funding Information</Label>
              <Input
                id="funding"
                value={funding}
                onChange={(e) => setFunding(e.target.value)}
                placeholder="Grant number, funding agency"
              />
            </div>

            <div>
              <Label htmlFor="related_identifiers">Related Identifiers</Label>
              <Input
                id="related_identifiers"
                value={relatedIdentifiers}
                onChange={(e) => setRelatedIdentifiers(e.target.value)}
                placeholder="DOIs of related datasets (comma-separated)"
              />
            </div>
          </div>

          <Separator />

          {/* Interoperable Info */}
          <Alert>
            <Info className="h-4 w-4" />
            <AlertDescription className="text-xs">
              <strong>Interoperability:</strong> Data will be exported with standardized schemas
              (schema.org, Dublin Core, DataCite) and controlled vocabularies for maximum compatibility.
            </AlertDescription>
          </Alert>

          {/* Dataset Info */}
          <div className="p-3 border rounded bg-muted/30 text-xs">
            <div className="font-semibold mb-2">Dataset Information</div>
            <div className="space-y-1 text-muted-foreground">
              <div>Dataset ID: <span className="font-mono">{datasetId}</span></div>
              <div>Dataset Type: <span className="font-mono">{datasetType}</span></div>
              <div>Export Date: <span className="font-mono">{new Date().toISOString()}</span></div>
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setIsOpen(false)} disabled={isExporting}>
            Cancel
          </Button>
          <Button onClick={handleExport} disabled={isExporting}>
            {isExporting ? (
              <>
                <FileText className="w-4 h-4 mr-2 animate-pulse" />
                Exporting...
              </>
            ) : (
              <>
                <Download className="w-4 h-4 mr-2" />
                Export FAIR Data
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default FAIRExportButton
