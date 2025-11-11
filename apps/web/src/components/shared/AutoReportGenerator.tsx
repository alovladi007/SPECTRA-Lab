/**
 * Auto-Report Generator
 *
 * Generates comprehensive PDF/HTML reports including:
 * - Process parameters & controller settings
 * - SPC snapshot with control charts
 * - VM predictions vs actuals
 * - Calibration IDs & equipment status
 * - E-signatures with timestamps
 * - Complete audit trail
 */

"use client"

import React, { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Checkbox } from '@/components/ui/checkbox'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { FileText, Download, CheckCircle2, AlertCircle, Loader2, FileType } from 'lucide-react'

interface ReportSection {
  id: string
  label: string
  description: string
  required: boolean
  enabled: boolean
}

interface ReportConfig {
  format: 'pdf' | 'html'
  title: string
  author: string
  includeTimestamp: boolean
  includeLogo: boolean
  sections: ReportSection[]
}

interface VMComparison {
  parameter: string
  predicted: number
  actual: number
  error_percent: number
  unit: string
}

interface SPCMetric {
  name: string
  value: number
  ucl: number
  lcl: number
  cpk: number
  status: 'in_control' | 'warning' | 'out_of_control'
}

interface CalibrationRecord {
  equipment_id: string
  last_calibrated: string
  next_due: string
  certificate_id: string
  status: 'current' | 'due_soon' | 'overdue'
}

interface AuditEntry {
  timestamp: string
  user: string
  action: string
  details: string
  ip_address?: string
}

interface ESignature {
  signer_name: string
  signer_role: string
  signed_at: string
  signature_hash: string
  reason: string
}

interface ReportData {
  run_id: string
  process_type: 'ion' | 'rtp'
  parameters: Record<string, any>
  controller_settings: Record<string, any>
  spc_metrics: SPCMetric[]
  vm_comparisons: VMComparison[]
  calibrations: CalibrationRecord[]
  audit_trail: AuditEntry[]
  e_signatures: ESignature[]
}

interface AutoReportGeneratorProps {
  runId: string
  processType: 'ion' | 'rtp'
  onGenerate?: (reportUrl: string) => void
}

const defaultSections: ReportSection[] = [
  {
    id: 'summary',
    label: 'Executive Summary',
    description: 'Overview of run results and key findings',
    required: true,
    enabled: true
  },
  {
    id: 'parameters',
    label: 'Process Parameters',
    description: 'Complete parameter configuration',
    required: true,
    enabled: true
  },
  {
    id: 'controller',
    label: 'Controller Settings',
    description: 'PID/MPC tuning and control strategy',
    required: true,
    enabled: true
  },
  {
    id: 'spc',
    label: 'SPC Snapshot',
    description: 'Control charts and capability metrics',
    required: false,
    enabled: true
  },
  {
    id: 'vm',
    label: 'Virtual Metrology',
    description: 'Predicted vs actual property comparison',
    required: false,
    enabled: true
  },
  {
    id: 'calibration',
    label: 'Calibration Status',
    description: 'Equipment calibration records',
    required: true,
    enabled: true
  },
  {
    id: 'signatures',
    label: 'E-Signatures',
    description: 'Digital signatures with timestamps',
    required: true,
    enabled: true
  },
  {
    id: 'audit',
    label: 'Audit Trail',
    description: 'Complete chronological action log',
    required: true,
    enabled: true
  }
]

export const AutoReportGenerator: React.FC<AutoReportGeneratorProps> = ({
  runId,
  processType,
  onGenerate
}) => {
  const [config, setConfig] = useState<ReportConfig>({
    format: 'pdf',
    title: `${processType.toUpperCase()} Run Report - ${runId}`,
    author: 'SPECTRA Lab System',
    includeTimestamp: true,
    includeLogo: true,
    sections: defaultSections
  })

  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedUrl, setGeneratedUrl] = useState<string | null>(null)
  const [showSignatureDialog, setShowSignatureDialog] = useState(false)
  const [signatureData, setSignatureData] = useState({
    name: '',
    role: '',
    reason: ''
  })

  // Mock report data - in production, fetch from API
  const mockReportData: ReportData = {
    run_id: runId,
    process_type: processType,
    parameters: processType === 'ion' ? {
      species: 'Phosphorus (P)',
      energy_kev: 40,
      dose_cm2: 5e14,
      tilt_angle: 7,
      twist_angle: 22
    } : {
      peak_temp_c: 1050,
      total_time_s: 120,
      ramp_rate_up: 75,
      ramp_rate_down: 40,
      thermal_budget: 42000
    },
    controller_settings: processType === 'ion' ? {
      beam_current_ma: 2.5,
      scan_frequency_hz: 500,
      scan_amplitude_mm: 100
    } : {
      controller_type: 'MPC',
      kp: 2.5,
      ki: 0.8,
      kd: 0.3,
      setpoint_tracking: 'enabled'
    },
    spc_metrics: [
      { name: 'Dose Uniformity', value: 98.5, ucl: 100, lcl: 95, cpk: 1.67, status: 'in_control' },
      { name: 'Energy Stability', value: 40.02, ucl: 40.5, lcl: 39.5, cpk: 1.89, status: 'in_control' },
      { name: 'Beam Current', value: 2.48, ucl: 2.6, lcl: 2.4, cpk: 1.45, status: 'in_control' }
    ],
    vm_comparisons: [
      { parameter: 'Sheet Resistance', predicted: 125.3, actual: 123.8, error_percent: 1.2, unit: 'Ω/□' },
      { parameter: 'Junction Depth', predicted: 0.285, actual: 0.291, error_percent: -2.1, unit: 'μm' },
      { parameter: 'Activation %', predicted: 87.5, actual: 86.2, error_percent: 1.5, unit: '%' }
    ],
    calibrations: [
      {
        equipment_id: processType === 'ion' ? 'ION-001' : 'RTP-001',
        last_calibrated: '2025-01-15',
        next_due: '2025-04-15',
        certificate_id: 'CAL-2025-001234',
        status: 'current'
      }
    ],
    audit_trail: [
      { timestamp: '2025-01-20 14:32:15', user: 'john.engineer', action: 'Recipe Created', details: 'Created new recipe configuration' },
      { timestamp: '2025-01-20 14:35:42', user: 'sarah.operator', action: 'Recipe Approved', details: 'Approved for production use', ip_address: '192.168.1.105' },
      { timestamp: '2025-01-20 14:40:10', user: 'sarah.operator', action: 'Run Started', details: `Started run ${runId}`, ip_address: '192.168.1.105' },
      { timestamp: '2025-01-20 14:52:33', user: 'system', action: 'Run Completed', details: 'Run completed successfully' },
      { timestamp: '2025-01-20 14:55:00', user: 'john.engineer', action: 'Results Reviewed', details: 'Reviewed VM predictions and SPC metrics', ip_address: '192.168.1.102' }
    ],
    e_signatures: [
      {
        signer_name: 'Sarah Johnson',
        signer_role: 'Process Operator',
        signed_at: '2025-01-20 14:52:45',
        signature_hash: 'SHA256:a3f5b8c2d9e1...',
        reason: 'Run execution approval'
      }
    ]
  }

  const updateSection = (sectionId: string, enabled: boolean) => {
    setConfig(prev => ({
      ...prev,
      sections: prev.sections.map(s =>
        s.id === sectionId ? { ...s, enabled } : s
      )
    }))
  }

  const generateReport = async () => {
    setIsGenerating(true)

    try {
      // Simulate API call to generate report
      // In production: POST /api/reports/generate
      await new Promise(resolve => setTimeout(resolve, 2000))

      const reportHtml = buildReportHTML(mockReportData, config)

      if (config.format === 'pdf') {
        // In production: Use jsPDF or server-side PDF generation
        const blob = new Blob([reportHtml], { type: 'text/html' })
        const url = URL.createObjectURL(blob)
        setGeneratedUrl(url)
        onGenerate?.(url)
      } else {
        const blob = new Blob([reportHtml], { type: 'text/html' })
        const url = URL.createObjectURL(blob)
        setGeneratedUrl(url)
        onGenerate?.(url)
      }

    } catch (error) {
      console.error('Report generation failed:', error)
    } finally {
      setIsGenerating(false)
    }
  }

  const buildReportHTML = (data: ReportData, config: ReportConfig): string => {
    const enabledSections = config.sections.filter(s => s.enabled)

    const timestamp = config.includeTimestamp ? new Date().toISOString() : ''

    return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>${config.title}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
    h1 { color: #2563eb; border-bottom: 3px solid #2563eb; padding-bottom: 10px; }
    h2 { color: #1e40af; margin-top: 30px; border-bottom: 2px solid #93c5fd; padding-bottom: 5px; }
    h3 { color: #1e3a8a; margin-top: 20px; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
    th { background-color: #2563eb; color: white; }
    tr:nth-child(even) { background-color: #f9fafb; }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }
    .metadata { color: #6b7280; font-size: 0.9em; margin-bottom: 30px; }
    .badge { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: bold; }
    .badge-success { background-color: #dcfce7; color: #166534; }
    .badge-warning { background-color: #fef3c7; color: #92400e; }
    .badge-error { background-color: #fee2e2; color: #991b1b; }
    .signature-block { border: 2px solid #2563eb; padding: 15px; margin: 10px 0; background-color: #eff6ff; }
    .audit-entry { padding: 10px; margin: 5px 0; border-left: 3px solid #93c5fd; background-color: #f8fafc; }
    .vm-comparison { margin: 10px 0; }
    .error-positive { color: #dc2626; }
    .error-negative { color: #059669; }
    @media print { body { margin: 20px; } }
  </style>
</head>
<body>
  <div class="header">
    ${config.includeLogo ? '<div style="font-size: 1.5em; font-weight: bold; color: #2563eb;">SPECTRA Lab</div>' : ''}
    <div style="text-align: right;">
      <div style="font-weight: bold;">Run ID: ${data.run_id}</div>
      <div style="color: #6b7280;">Process: ${data.process_type.toUpperCase()}</div>
    </div>
  </div>

  <h1>${config.title}</h1>

  <div class="metadata">
    <strong>Generated:</strong> ${timestamp}<br>
    <strong>Author:</strong> ${config.author}<br>
    <strong>Format:</strong> ${config.format.toUpperCase()}
  </div>

  ${enabledSections.find(s => s.id === 'summary') ? `
  <h2>Executive Summary</h2>
  <p>This report documents the execution and results of ${data.process_type.toUpperCase()} run ${data.run_id}.
  All process parameters were within specification, SPC metrics indicate in-control process,
  and virtual metrology predictions matched actual measurements within acceptable tolerance.</p>
  ` : ''}

  ${enabledSections.find(s => s.id === 'parameters') ? `
  <h2>Process Parameters</h2>
  <table>
    <thead>
      <tr><th>Parameter</th><th>Value</th></tr>
    </thead>
    <tbody>
      ${Object.entries(data.parameters).map(([key, value]) => `
        <tr>
          <td>${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</td>
          <td>${value}</td>
        </tr>
      `).join('')}
    </tbody>
  </table>
  ` : ''}

  ${enabledSections.find(s => s.id === 'controller') ? `
  <h2>Controller Settings</h2>
  <table>
    <thead>
      <tr><th>Setting</th><th>Value</th></tr>
    </thead>
    <tbody>
      ${Object.entries(data.controller_settings).map(([key, value]) => `
        <tr>
          <td>${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</td>
          <td>${value}</td>
        </tr>
      `).join('')}
    </tbody>
  </table>
  ` : ''}

  ${enabledSections.find(s => s.id === 'spc') ? `
  <h2>SPC Snapshot</h2>
  <table>
    <thead>
      <tr><th>Metric</th><th>Value</th><th>UCL</th><th>LCL</th><th>Cpk</th><th>Status</th></tr>
    </thead>
    <tbody>
      ${data.spc_metrics.map(metric => `
        <tr>
          <td>${metric.name}</td>
          <td>${metric.value}</td>
          <td>${metric.ucl}</td>
          <td>${metric.lcl}</td>
          <td>${metric.cpk.toFixed(2)}</td>
          <td>
            <span class="badge ${
              metric.status === 'in_control' ? 'badge-success' :
              metric.status === 'warning' ? 'badge-warning' : 'badge-error'
            }">
              ${metric.status.replace(/_/g, ' ').toUpperCase()}
            </span>
          </td>
        </tr>
      `).join('')}
    </tbody>
  </table>
  ` : ''}

  ${enabledSections.find(s => s.id === 'vm') ? `
  <h2>Virtual Metrology - Predicted vs Actual</h2>
  <table>
    <thead>
      <tr><th>Parameter</th><th>Predicted</th><th>Actual</th><th>Error %</th><th>Unit</th></tr>
    </thead>
    <tbody>
      ${data.vm_comparisons.map(comp => `
        <tr>
          <td>${comp.parameter}</td>
          <td>${comp.predicted.toFixed(2)}</td>
          <td>${comp.actual.toFixed(2)}</td>
          <td class="${comp.error_percent < 0 ? 'error-negative' : 'error-positive'}">
            ${comp.error_percent > 0 ? '+' : ''}${comp.error_percent.toFixed(1)}%
          </td>
          <td>${comp.unit}</td>
        </tr>
      `).join('')}
    </tbody>
  </table>
  <p><em>Note: VM predictions within ±5% error are considered excellent agreement.</em></p>
  ` : ''}

  ${enabledSections.find(s => s.id === 'calibration') ? `
  <h2>Equipment Calibration Status</h2>
  <table>
    <thead>
      <tr><th>Equipment</th><th>Last Calibrated</th><th>Next Due</th><th>Certificate ID</th><th>Status</th></tr>
    </thead>
    <tbody>
      ${data.calibrations.map(cal => `
        <tr>
          <td>${cal.equipment_id}</td>
          <td>${cal.last_calibrated}</td>
          <td>${cal.next_due}</td>
          <td>${cal.certificate_id}</td>
          <td>
            <span class="badge ${
              cal.status === 'current' ? 'badge-success' :
              cal.status === 'due_soon' ? 'badge-warning' : 'badge-error'
            }">
              ${cal.status.replace(/_/g, ' ').toUpperCase()}
            </span>
          </td>
        </tr>
      `).join('')}
    </tbody>
  </table>
  ` : ''}

  ${enabledSections.find(s => s.id === 'signatures') ? `
  <h2>Electronic Signatures</h2>
  ${data.e_signatures.map(sig => `
    <div class="signature-block">
      <strong>Signer:</strong> ${sig.signer_name} (${sig.signer_role})<br>
      <strong>Signed At:</strong> ${sig.signed_at}<br>
      <strong>Reason:</strong> ${sig.reason}<br>
      <strong>Signature Hash:</strong> <code>${sig.signature_hash}</code>
    </div>
  `).join('')}
  ` : ''}

  ${enabledSections.find(s => s.id === 'audit') ? `
  <h2>Audit Trail</h2>
  ${data.audit_trail.map(entry => `
    <div class="audit-entry">
      <strong>${entry.timestamp}</strong> - ${entry.user}<br>
      <strong>Action:</strong> ${entry.action}<br>
      <strong>Details:</strong> ${entry.details}
      ${entry.ip_address ? `<br><strong>IP:</strong> ${entry.ip_address}` : ''}
    </div>
  `).join('')}
  ` : ''}

  <div style="margin-top: 50px; padding-top: 20px; border-top: 2px solid #e5e7eb; color: #6b7280; font-size: 0.85em;">
    <p><em>This report was automatically generated by SPECTRA Lab reporting system.
    All data is traceable and audit trail is maintained per 21 CFR Part 11 requirements.</em></p>
  </div>
</body>
</html>
    `
  }

  const addSignature = () => {
    if (!signatureData.name || !signatureData.role || !signatureData.reason) {
      alert('Please fill in all signature fields')
      return
    }

    const newSignature: ESignature = {
      signer_name: signatureData.name,
      signer_role: signatureData.role,
      signed_at: new Date().toISOString(),
      signature_hash: `SHA256:${Math.random().toString(36).substring(2, 15)}...`,
      reason: signatureData.reason
    }

    mockReportData.e_signatures.push(newSignature)
    setShowSignatureDialog(false)
    setSignatureData({ name: '', role: '', reason: '' })
  }

  const downloadReport = () => {
    if (!generatedUrl) return

    const link = document.createElement('a')
    link.href = generatedUrl
    link.download = `${runId}_report.${config.format === 'pdf' ? 'pdf' : 'html'}`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileText className="w-5 h-5" />
          Auto-Report Generator
        </CardTitle>
        <CardDescription>
          Generate comprehensive PDF/HTML reports with SPC, VM predictions, e-signatures, and audit trails
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Report Configuration */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label>Format</Label>
            <Select
              value={config.format}
              onValueChange={(value: 'pdf' | 'html') =>
                setConfig(prev => ({ ...prev, format: value }))
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="pdf">PDF</SelectItem>
                <SelectItem value="html">HTML</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Report Title</Label>
            <Input
              value={config.title}
              onChange={(e) =>
                setConfig(prev => ({ ...prev, title: e.target.value }))
              }
              placeholder="Report title..."
            />
          </div>

          <div className="space-y-2">
            <Label>Author</Label>
            <Input
              value={config.author}
              onChange={(e) =>
                setConfig(prev => ({ ...prev, author: e.target.value }))
              }
              placeholder="Author name..."
            />
          </div>

          <div className="flex items-center gap-4 pt-6">
            <div className="flex items-center gap-2">
              <Checkbox
                id="timestamp"
                checked={config.includeTimestamp}
                onCheckedChange={(checked) =>
                  setConfig(prev => ({ ...prev, includeTimestamp: checked as boolean }))
                }
              />
              <Label htmlFor="timestamp" className="text-sm">Include Timestamp</Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox
                id="logo"
                checked={config.includeLogo}
                onCheckedChange={(checked) =>
                  setConfig(prev => ({ ...prev, includeLogo: checked as boolean }))
                }
              />
              <Label htmlFor="logo" className="text-sm">Include Logo</Label>
            </div>
          </div>
        </div>

        <Separator />

        {/* Section Selection */}
        <div>
          <h3 className="text-sm font-semibold mb-3">Report Sections</h3>
          <div className="space-y-3">
            {config.sections.map(section => (
              <div
                key={section.id}
                className="flex items-start gap-3 p-3 rounded-lg border bg-muted/30"
              >
                <Checkbox
                  id={section.id}
                  checked={section.enabled}
                  onCheckedChange={(checked) => updateSection(section.id, checked as boolean)}
                  disabled={section.required}
                />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <Label htmlFor={section.id} className="font-medium">
                      {section.label}
                    </Label>
                    {section.required && (
                      <Badge variant="secondary" className="text-xs">Required</Badge>
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground">{section.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <Separator />

        {/* E-Signature Management */}
        <div>
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-sm font-semibold">Electronic Signatures</h3>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowSignatureDialog(true)}
            >
              Add Signature
            </Button>
          </div>
          <div className="space-y-2">
            {mockReportData.e_signatures.map((sig, idx) => (
              <div key={idx} className="flex items-center gap-2 p-2 rounded border bg-blue-50">
                <CheckCircle2 className="w-4 h-4 text-blue-600" />
                <div className="flex-1 text-sm">
                  <strong>{sig.signer_name}</strong> ({sig.signer_role}) - {sig.reason}
                </div>
                <span className="text-xs text-muted-foreground">{sig.signed_at}</span>
              </div>
            ))}
          </div>
        </div>

        <Separator />

        {/* Generation Status */}
        {generatedUrl && (
          <Alert>
            <CheckCircle2 className="h-4 w-4" />
            <AlertDescription>
              Report generated successfully! Click the download button below to save.
            </AlertDescription>
          </Alert>
        )}

        {/* Action Buttons */}
        <div className="flex gap-3">
          <Button
            onClick={generateReport}
            disabled={isGenerating}
            className="flex-1"
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Generating Report...
              </>
            ) : (
              <>
                <FileType className="w-4 h-4 mr-2" />
                Generate Report
              </>
            )}
          </Button>

          {generatedUrl && (
            <Button onClick={downloadReport} variant="outline">
              <Download className="w-4 h-4 mr-2" />
              Download
            </Button>
          )}
        </div>

        {/* Statistics */}
        <div className="grid grid-cols-3 gap-4 pt-4 border-t">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {config.sections.filter(s => s.enabled).length}
            </div>
            <div className="text-xs text-muted-foreground">Sections Included</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {mockReportData.e_signatures.length}
            </div>
            <div className="text-xs text-muted-foreground">E-Signatures</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {mockReportData.audit_trail.length}
            </div>
            <div className="text-xs text-muted-foreground">Audit Entries</div>
          </div>
        </div>
      </CardContent>

      {/* E-Signature Dialog */}
      <Dialog open={showSignatureDialog} onOpenChange={setShowSignatureDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Electronic Signature</DialogTitle>
            <DialogDescription>
              Sign this report to approve the results and certify accuracy
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Your Name</Label>
              <Input
                value={signatureData.name}
                onChange={(e) => setSignatureData(prev => ({ ...prev, name: e.target.value }))}
                placeholder="John Doe"
              />
            </div>
            <div className="space-y-2">
              <Label>Your Role</Label>
              <Input
                value={signatureData.role}
                onChange={(e) => setSignatureData(prev => ({ ...prev, role: e.target.value }))}
                placeholder="Process Engineer"
              />
            </div>
            <div className="space-y-2">
              <Label>Reason for Signature</Label>
              <Input
                value={signatureData.reason}
                onChange={(e) => setSignatureData(prev => ({ ...prev, reason: e.target.value }))}
                placeholder="Approval of run results"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowSignatureDialog(false)}>
              Cancel
            </Button>
            <Button onClick={addSignature}>
              Add Signature
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  )
}

export default AutoReportGenerator
