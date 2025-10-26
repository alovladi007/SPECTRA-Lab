/**
 * SESSION 15: LIMS/ELN & REPORTING - UI Components
 * ==============================================
 * 
 * React/TypeScript components for Sample Management, Electronic Lab Notebook,
 * SOP Management, and Report Generation.
 * 
 * @author SemiconductorLab Platform Team
 * @date October 26, 2025
 * @version 1.0.0
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
  Button, Input, Label, Textarea, Select, SelectContent, SelectItem,
  SelectTrigger, SelectValue, Badge, Tabs, TabsContent, TabsList, TabsTrigger,
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader,
  DialogTitle, Alert, AlertDescription, Checkbox, Progress, Separator
} from '@/components/ui';
import {
  Package, Barcode, QrCode, FileText, CheckSquare, Users,
  Calendar, MapPin, AlertCircle, CheckCircle, Clock, Download,
  Upload, Search, Filter, RefreshCw, Save, Printer, Mail,
  FileSignature, Lock, Unlock, Eye, Edit, Trash2, Plus
} from 'lucide-react';

// ===================================================================
// SAMPLE MANAGEMENT COMPONENTS
// ===================================================================

/**
 * Sample Creation Form with Barcode Generation
 */
export const SampleCreateForm: React.FC<{
  projectId: number;
  onSuccess: (sample: any) => void;
}> = ({ projectId, onSuccess }) => {
  const [formData, setFormData] = useState({
    material_type: '',
    sample_type: 'wafer',
    description: '',
    location: '',
    dimensions: {
      width: 0,
      length: 0,
      thickness: 0,
      units: 'mm'
    },
    weight: 0
  });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch('/api/v1/lims/samples', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...formData,
          project_id: projectId
        })
      });

      const result = await response.json();
      onSuccess(result);
    } catch (error) {
      console.error('Failed to create sample:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="material_type">Material Type *</Label>
          <Select
            value={formData.material_type}
            onValueChange={(value) => 
              setFormData({ ...formData, material_type: value })
            }
          >
            <SelectTrigger>
              <SelectValue placeholder="Select material" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="silicon">Silicon (Si)</SelectItem>
              <SelectItem value="gaas">Gallium Arsenide (GaAs)</SelectItem>
              <SelectItem value="gan">Gallium Nitride (GaN)</SelectItem>
              <SelectItem value="sic">Silicon Carbide (SiC)</SelectItem>
              <SelectItem value="ingap">Indium Gallium Phosphide (InGaP)</SelectItem>
              <SelectItem value="other">Other</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="sample_type">Sample Type *</Label>
          <Select
            value={formData.sample_type}
            onValueChange={(value) => 
              setFormData({ ...formData, sample_type: value })
            }
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="wafer">Wafer</SelectItem>
              <SelectItem value="die">Die</SelectItem>
              <SelectItem value="device">Device</SelectItem>
              <SelectItem value="coupon">Coupon</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="space-y-2">
        <Label htmlFor="description">Description</Label>
        <Textarea
          id="description"
          value={formData.description}
          onChange={(e) => 
            setFormData({ ...formData, description: e.target.value })
          }
          placeholder="Enter sample description..."
          rows={3}
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="location">Storage Location *</Label>
        <Input
          id="location"
          value={formData.location}
          onChange={(e) => 
            setFormData({ ...formData, location: e.target.value })
          }
          placeholder="e.g., Shelf A3, Drawer 2"
        />
      </div>

      <div className="grid grid-cols-4 gap-4">
        <div className="space-y-2">
          <Label htmlFor="width">Width</Label>
          <Input
            id="width"
            type="number"
            step="0.01"
            value={formData.dimensions.width}
            onChange={(e) => 
              setFormData({
                ...formData,
                dimensions: {
                  ...formData.dimensions,
                  width: parseFloat(e.target.value) || 0
                }
              })
            }
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="length">Length</Label>
          <Input
            id="length"
            type="number"
            step="0.01"
            value={formData.dimensions.length}
            onChange={(e) => 
              setFormData({
                ...formData,
                dimensions: {
                  ...formData.dimensions,
                  length: parseFloat(e.target.value) || 0
                }
              })
            }
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="thickness">Thickness</Label>
          <Input
            id="thickness"
            type="number"
            step="0.001"
            value={formData.dimensions.thickness}
            onChange={(e) => 
              setFormData({
                ...formData,
                dimensions: {
                  ...formData.dimensions,
                  thickness: parseFloat(e.target.value) || 0
                }
              })
            }
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="weight">Weight (g)</Label>
          <Input
            id="weight"
            type="number"
            step="0.001"
            value={formData.weight}
            onChange={(e) => 
              setFormData({
                ...formData,
                weight: parseFloat(e.target.value) || 0
              })
            }
          />
        </div>
      </div>

      <Button type="submit" disabled={loading} className="w-full">
        {loading ? (
          <>
            <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
            Creating Sample...
          </>
        ) : (
          <>
            <Plus className="w-4 h-4 mr-2" />
            Create Sample
          </>
        )}
      </Button>
    </form>
  );
};

/**
 * Sample Details with Barcode/QR Code Display
 */
export const SampleDetailsCard: React.FC<{
  sampleId: string;
}> = ({ sampleId }) => {
  const [sample, setSample] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSample();
  }, [sampleId]);

  const fetchSample = async () => {
    try {
      const response = await fetch(`/api/v1/lims/samples/${sampleId}`);
      const data = await response.json();
      setSample(data.sample);
    } catch (error) {
      console.error('Failed to fetch sample:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="animate-pulse">Loading sample...</div>;
  }

  if (!sample) {
    return <Alert><AlertDescription>Sample not found</AlertDescription></Alert>;
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center">
              <Package className="w-5 h-5 mr-2" />
              {sample.sample_id}
            </CardTitle>
            <CardDescription>{sample.description}</CardDescription>
          </div>
          <Badge className={getStatusColor(sample.status)}>
            {sample.status}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Basic Information */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <Label className="text-sm text-gray-500">Material Type</Label>
            <p className="font-medium">{sample.material_type}</p>
          </div>
          <div>
            <Label className="text-sm text-gray-500">Sample Type</Label>
            <p className="font-medium">{sample.sample_type}</p>
          </div>
          <div>
            <Label className="text-sm text-gray-500">Location</Label>
            <p className="font-medium flex items-center">
              <MapPin className="w-4 h-4 mr-1" />
              {sample.location}
            </p>
          </div>
          <div>
            <Label className="text-sm text-gray-500">Received</Label>
            <p className="font-medium">
              {new Date(sample.received_date).toLocaleDateString()}
            </p>
          </div>
        </div>

        <Separator />

        {/* Barcode and QR Code */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label className="flex items-center">
              <Barcode className="w-4 h-4 mr-2" />
              Barcode
            </Label>
            <div className="border rounded-lg p-4 bg-white">
              <img
                src={sample.barcode}
                alt="Sample Barcode"
                className="w-full h-auto"
              />
            </div>
            <Button variant="outline" size="sm" className="w-full">
              <Printer className="w-4 h-4 mr-2" />
              Print Barcode
            </Button>
          </div>

          <div className="space-y-2">
            <Label className="flex items-center">
              <QrCode className="w-4 h-4 mr-2" />
              QR Code
            </Label>
            <div className="border rounded-lg p-4 bg-white">
              <img
                src={sample.qr_code}
                alt="Sample QR Code"
                className="w-full h-auto"
              />
            </div>
            <Button variant="outline" size="sm" className="w-full">
              <Printer className="w-4 h-4 mr-2" />
              Print QR Code
            </Button>
          </div>
        </div>

        {/* Dimensions */}
        {sample.dimensions && (
          <>
            <Separator />
            <div>
              <Label className="text-sm font-semibold mb-2 block">
                Dimensions
              </Label>
              <div className="grid grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-gray-500">Width</p>
                  <p className="font-medium">
                    {sample.dimensions.width} {sample.dimensions.units}
                  </p>
                </div>
                <div>
                  <p className="text-gray-500">Length</p>
                  <p className="font-medium">
                    {sample.dimensions.length} {sample.dimensions.units}
                  </p>
                </div>
                <div>
                  <p className="text-gray-500">Thickness</p>
                  <p className="font-medium">
                    {sample.dimensions.thickness} {sample.dimensions.units}
                  </p>
                </div>
                <div>
                  <p className="text-gray-500">Weight</p>
                  <p className="font-medium">{sample.weight} g</p>
                </div>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
};

/**
 * Chain of Custody Log Viewer
 */
export const CustodyChainViewer: React.FC<{
  sampleId: string;
}> = ({ sampleId }) => {
  const [logs, setLogs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchCustodyLogs();
  }, [sampleId]);

  const fetchCustodyLogs = async () => {
    try {
      const response = await fetch(`/api/v1/lims/samples/${sampleId}`);
      const data = await response.json();
      setLogs(data.custody_chain || []);
    } catch (error) {
      console.error('Failed to fetch custody logs:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <Progress value={50} className="w-full" />;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center">
          <Users className="w-5 h-5 mr-2" />
          Chain of Custody
        </CardTitle>
        <CardDescription>
          Complete history of sample handling and transfers
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {logs.length === 0 ? (
            <Alert>
              <AlertDescription>No custody logs recorded</AlertDescription>
            </Alert>
          ) : (
            logs.map((log, index) => (
              <div
                key={log.id}
                className="flex items-start space-x-4 border-l-2 border-blue-500 pl-4 pb-4"
              >
                <div className="flex-shrink-0 mt-1">
                  {getActionIcon(log.action)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <p className="font-medium">{log.action}</p>
                    <span className="text-sm text-gray-500">
                      {new Date(log.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div className="mt-1 text-sm text-gray-600">
                    {log.from_user_id && (
                      <p>From: User #{log.from_user_id}</p>
                    )}
                    {log.to_user_id && (
                      <p>To: User #{log.to_user_id}</p>
                    )}
                    {log.from_location && (
                      <p>From Location: {log.from_location}</p>
                    )}
                    {log.to_location && (
                      <p>To Location: {log.to_location}</p>
                    )}
                    {log.reason && (
                      <p className="mt-1 italic">{log.reason}</p>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
};

// ===================================================================
// ELECTRONIC LAB NOTEBOOK COMPONENTS
// ===================================================================

/**
 * Rich Text ELN Editor
 */
export const ELNEditor: React.FC<{
  projectId: number;
  initialEntry?: any;
  onSave: (entry: any) => void;
}> = ({ projectId, initialEntry, onSave }) => {
  const [title, setTitle] = useState(initialEntry?.title || '');
  const [content, setContent] = useState(initialEntry?.content || '');
  const [linkedSamples, setLinkedSamples] = useState<string[]>(
    initialEntry?.linked_samples || []
  );
  const [linkedRuns, setLinkedRuns] = useState<string[]>(
    initialEntry?.linked_runs || []
  );
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);

    try {
      const response = await fetch('/api/v1/lims/eln/entries', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_id: projectId,
          title,
          content,
          content_format: 'html',
          linked_samples: linkedSamples,
          linked_runs: linkedRuns
        })
      });

      const result = await response.json();
      onSave(result);
    } catch (error) {
      console.error('Failed to save entry:', error);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center">
            <FileText className="w-5 h-5 mr-2" />
            {initialEntry ? 'Edit Entry' : 'New Lab Notebook Entry'}
          </CardTitle>
          <Button onClick={handleSave} disabled={saving}>
            {saving ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <Save className="w-4 h-4 mr-2" />
                Save Entry
              </>
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="title">Entry Title *</Label>
          <Input
            id="title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Enter entry title..."
            className="text-lg font-medium"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="content">Content</Label>
          <Textarea
            id="content"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Document your experiment, observations, results..."
            rows={15}
            className="font-mono text-sm"
          />
          <p className="text-xs text-gray-500">
            Supports HTML formatting. Use standard HTML tags for formatting.
          </p>
        </div>

        <Separator />

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label>Linked Samples</Label>
            <Input
              placeholder="Add sample IDs..."
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  const value = e.currentTarget.value;
                  if (value && !linkedSamples.includes(value)) {
                    setLinkedSamples([...linkedSamples, value]);
                    e.currentTarget.value = '';
                  }
                }
              }}
            />
            <div className="flex flex-wrap gap-2 mt-2">
              {linkedSamples.map((sample) => (
                <Badge key={sample} variant="secondary">
                  {sample}
                  <button
                    className="ml-2"
                    onClick={() => 
                      setLinkedSamples(linkedSamples.filter(s => s !== sample))
                    }
                  >
                    ×
                  </button>
                </Badge>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <Label>Linked Runs</Label>
            <Input
              placeholder="Add run IDs..."
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  const value = e.currentTarget.value;
                  if (value && !linkedRuns.includes(value)) {
                    setLinkedRuns([...linkedRuns, value]);
                    e.currentTarget.value = '';
                  }
                }
              }}
            />
            <div className="flex flex-wrap gap-2 mt-2">
              {linkedRuns.map((run) => (
                <Badge key={run} variant="secondary">
                  {run}
                  <button
                    className="ml-2"
                    onClick={() => 
                      setLinkedRuns(linkedRuns.filter(r => r !== run))
                    }
                  >
                    ×
                  </button>
                </Badge>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

/**
 * E-Signature Dialog (21 CFR Part 11)
 */
export const SignatureDialog: React.FC<{
  entryId: string;
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
}> = ({ entryId, open, onClose, onSuccess }) => {
  const [signatureType, setSignatureType] = useState('execution');
  const [reason, setReason] = useState('');
  const [password, setPassword] = useState('');
  const [signing, setSigning] = useState(false);

  const handleSign = async () => {
    setSigning(true);

    try {
      const response = await fetch(`/api/v1/lims/eln/entries/${entryId}/sign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          entry_id: entryId,
          signature_type: signatureType,
          reason,
          ip_address: window.location.hostname,
          user_agent: navigator.userAgent
        })
      });

      if (response.ok) {
        onSuccess();
        onClose();
      }
    } catch (error) {
      console.error('Failed to sign entry:', error);
    } finally {
      setSigning(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center">
            <FileSignature className="w-5 h-5 mr-2" />
            Electronic Signature
          </DialogTitle>
          <DialogDescription>
            Sign this entry to certify its accuracy and completeness (21 CFR Part 11)
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="signature_type">Signature Type *</Label>
            <Select value={signatureType} onValueChange={setSignatureType}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="execution">Execution (I performed this)</SelectItem>
                <SelectItem value="review">Review (I reviewed this)</SelectItem>
                <SelectItem value="approval">Approval (I approve this)</SelectItem>
                <SelectItem value="witness">Witness (I witnessed this)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="reason">Meaning/Reason *</Label>
            <Textarea
              id="reason"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="e.g., Approving experimental results for publication"
              rows={3}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="password">Password Confirmation *</Label>
            <Input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
            />
          </div>

          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              By signing, you certify that this information is accurate and you
              have the authority to sign. This action is legally binding and will
              be recorded with timestamp and IP address.
            </AlertDescription>
          </Alert>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button
            onClick={handleSign}
            disabled={signing || !reason || !password}
          >
            {signing ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                Signing...
              </>
            ) : (
              <>
                <FileSignature className="w-4 h-4 mr-2" />
                Sign Entry
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

// ===================================================================
// SOP MANAGEMENT COMPONENTS
// ===================================================================

/**
 * SOP Viewer with Pre-Run Checklist
 */
export const SOPViewer: React.FC<{
  sopNumber: string;
  showChecklist?: boolean;
  onChecklistComplete?: (completed: boolean) => void;
}> = ({ sopNumber, showChecklist = false, onChecklistComplete }) => {
  const [sop, setSop] = useState<any>(null);
  const [checklistState, setChecklistState] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSOP();
  }, [sopNumber]);

  const fetchSOP = async () => {
    // Fetch SOP details
    setLoading(false);
    // Mock data for demo
    setSop({
      sop_number: sopNumber,
      title: 'Four-Point Probe Measurement Procedure',
      version: '2.1',
      effective_date: '2025-01-15',
      content: '# Procedure\n\n1. Safety first...\n2. Sample preparation...',
      checklist_items: [
        { id: 1, item: 'Verify sample is clean and dry', required: true },
        { id: 2, item: 'Check instrument calibration date', required: true },
        { id: 3, item: 'Ensure proper grounding', required: true },
        { id: 4, item: 'Record environmental conditions', required: false }
      ]
    });
  };

  const handleChecklistChange = (itemId: number, checked: boolean) => {
    const newState = { ...checklistState, [itemId]: checked };
    setChecklistState(newState);

    // Check if all required items are completed
    const requiredItems = sop.checklist_items.filter((item: any) => item.required);
    const allComplete = requiredItems.every((item: any) => newState[item.id]);
    
    if (onChecklistComplete) {
      onChecklistComplete(allComplete);
    }
  };

  if (loading || !sop) {
    return <div className="animate-pulse">Loading SOP...</div>;
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center">
              <FileText className="w-5 h-5 mr-2" />
              {sop.title}
            </CardTitle>
            <CardDescription>
              {sop.sop_number} • Version {sop.version} • Effective {sop.effective_date}
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm">
              <Printer className="w-4 h-4 mr-2" />
              Print
            </Button>
            <Button variant="outline" size="sm">
              <Download className="w-4 h-4 mr-2" />
              Download PDF
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="content">
          <TabsList>
            <TabsTrigger value="content">SOP Content</TabsTrigger>
            {showChecklist && (
              <TabsTrigger value="checklist">Pre-Run Checklist</TabsTrigger>
            )}
            <TabsTrigger value="history">Revision History</TabsTrigger>
          </TabsList>

          <TabsContent value="content" className="space-y-4">
            <div className="prose max-w-none">
              {/* Render markdown content */}
              <div dangerouslySetInnerHTML={{ __html: sop.content }} />
            </div>
          </TabsContent>

          {showChecklist && (
            <TabsContent value="checklist" className="space-y-4">
              <Alert>
                <CheckSquare className="h-4 w-4" />
                <AlertDescription>
                  Complete all required items before starting the measurement
                </AlertDescription>
              </Alert>

              <div className="space-y-3">
                {sop.checklist_items.map((item: any) => (
                  <div
                    key={item.id}
                    className="flex items-start space-x-3 p-3 border rounded-lg"
                  >
                    <Checkbox
                      id={`checklist-${item.id}`}
                      checked={checklistState[item.id] || false}
                      onCheckedChange={(checked) =>
                        handleChecklistChange(item.id, checked as boolean)
                      }
                    />
                    <label
                      htmlFor={`checklist-${item.id}`}
                      className="flex-1 cursor-pointer"
                    >
                      <p className="font-medium">{item.item}</p>
                      {item.required && (
                        <Badge variant="destructive" className="mt-1">
                          Required
                        </Badge>
                      )}
                    </label>
                  </div>
                ))}
              </div>
            </TabsContent>
          )}

          <TabsContent value="history">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Version</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead>Changes</TableHead>
                  <TableHead>Author</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                <TableRow>
                  <TableCell>2.1</TableCell>
                  <TableCell>2025-01-15</TableCell>
                  <TableCell>Updated safety procedures</TableCell>
                  <TableCell>John Doe</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>2.0</TableCell>
                  <TableCell>2024-06-01</TableCell>
                  <TableCell>Major revision - new instrument</TableCell>
                  <TableCell>Jane Smith</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

// ===================================================================
// REPORT GENERATION COMPONENTS
// ===================================================================

/**
 * Report Generator Interface
 */
export const ReportGenerator: React.FC<{
  runIds: number[];
}> = ({ runIds }) => {
  const [template, setTemplate] = useState({
    template_name: 'standard',
    title: 'Measurement Report',
    sections: ['summary', 'methods', 'parameters', 'results', 'spc', 'approvals'],
    include_plots: true,
    include_raw_data: false,
    page_size: 'letter'
  });
  const [generating, setGenerating] = useState(false);

  const handleGenerate = async () => {
    setGenerating(true);

    try {
      const response = await fetch('/api/v1/lims/reports/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          run_id: runIds[0], // Single run for now
          template
        })
      });

      const result = await response.json();
      
      // Download PDF
      window.open(result.report_path, '_blank');
    } catch (error) {
      console.error('Failed to generate report:', error);
    } finally {
      setGenerating(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center">
          <FileText className="w-5 h-5 mr-2" />
          Generate Report
        </CardTitle>
        <CardDescription>
          Create a formatted PDF report for selected runs
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <Label htmlFor="title">Report Title</Label>
          <Input
            id="title"
            value={template.title}
            onChange={(e) => setTemplate({ ...template, title: e.target.value })}
          />
        </div>

        <div className="space-y-2">
          <Label>Sections to Include</Label>
          <div className="space-y-2">
            {['summary', 'methods', 'parameters', 'results', 'spc', 'approvals'].map(
              (section) => (
                <div key={section} className="flex items-center space-x-2">
                  <Checkbox
                    id={section}
                    checked={template.sections.includes(section)}
                    onCheckedChange={(checked) => {
                      if (checked) {
                        setTemplate({
                          ...template,
                          sections: [...template.sections, section]
                        });
                      } else {
                        setTemplate({
                          ...template,
                          sections: template.sections.filter(s => s !== section)
                        });
                      }
                    }}
                  />
                  <label htmlFor={section} className="capitalize cursor-pointer">
                    {section}
                  </label>
                </div>
              )
            )}
          </div>
        </div>

        <div className="space-y-2">
          <Label>Options</Label>
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="include_plots"
                checked={template.include_plots}
                onCheckedChange={(checked) =>
                  setTemplate({ ...template, include_plots: checked as boolean })
                }
              />
              <label htmlFor="include_plots" className="cursor-pointer">
                Include plots and figures
              </label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="include_raw_data"
                checked={template.include_raw_data}
                onCheckedChange={(checked) =>
                  setTemplate({ ...template, include_raw_data: checked as boolean })
                }
              />
              <label htmlFor="include_raw_data" className="cursor-pointer">
                Include raw data tables
              </label>
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <Label htmlFor="page_size">Page Size</Label>
          <Select
            value={template.page_size}
            onValueChange={(value) => setTemplate({ ...template, page_size: value })}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="letter">Letter (8.5" × 11")</SelectItem>
              <SelectItem value="A4">A4 (210mm × 297mm)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Button
          onClick={handleGenerate}
          disabled={generating}
          className="w-full"
        >
          {generating ? (
            <>
              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
              Generating Report...
            </>
          ) : (
            <>
              <FileText className="w-4 h-4 mr-2" />
              Generate PDF Report
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  );
};

// ===================================================================
// HELPER FUNCTIONS
// ===================================================================

const getStatusColor = (status: string): string => {
  const colors: Record<string, string> = {
    received: 'bg-blue-100 text-blue-800',
    in_process: 'bg-yellow-100 text-yellow-800',
    measured: 'bg-green-100 text-green-800',
    on_hold: 'bg-orange-100 text-orange-800',
    complete: 'bg-green-100 text-green-800',
    archived: 'bg-gray-100 text-gray-800',
    discarded: 'bg-red-100 text-red-800'
  };
  return colors[status] || 'bg-gray-100 text-gray-800';
};

const getActionIcon = (action: string) => {
  const icons: Record<string, React.ReactNode> = {
    received: <Package className="w-4 h-4 text-blue-500" />,
    transferred: <Users className="w-4 h-4 text-purple-500" />,
    measured: <CheckCircle className="w-4 h-4 text-green-500" />,
    stored: <MapPin className="w-4 h-4 text-gray-500" />,
    disposed: <Trash2 className="w-4 h-4 text-red-500" />
  };
  return icons[action] || <Clock className="w-4 h-4 text-gray-500" />;
};

export default {
  SampleCreateForm,
  SampleDetailsCard,
  CustodyChainViewer,
  ELNEditor,
  SignatureDialog,
  SOPViewer,
  ReportGenerator
};
