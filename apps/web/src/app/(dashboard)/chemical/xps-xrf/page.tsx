// Session 11: XPS/XRF Analysis - React UI Components
// Complete interface for surface and elemental analysis

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import {
  Atom,
  Activity,
  Zap,
  Target,
  Layers,
  TrendingUp,
  Download,
  Upload,
  Play,
  Pause,
  RotateCcw,
  Settings,
  Info,
  AlertCircle,
  ChevronRight,
  Beaker,
  Microscope,
  BarChart3,
  FileText,
  Shield,
  Gauge,
} from 'lucide-react';

// Types
interface XPSPeak {
  id: string;
  position: number;
  area: number;
  fwhm: number;
  element: string;
  orbital: string;
  chemicalState?: string;
  asymmetry: number;
  shape: 'Gaussian' | 'Lorentzian' | 'Voigt' | 'Doniach-Sunjic';
}

interface XRFPeak {
  id: string;
  energy: number;
  intensity: number;
  element: string;
  line: string;
  escapePeak: boolean;
  sumPeak: boolean;
}

interface ElementComposition {
  element: string;
  concentration: number;
  error: number;
  orbital?: string;
  line?: string;
}

interface DepthProfile {
  depth: number[];
  elements: { [key: string]: number[] };
}

interface ChemicalState {
  state: string;
  position: number;
  percentage: number;
  reference: number;
}

interface AcquisitionParams {
  technique: 'XPS' | 'XRF';
  source: string;
  passEnergy?: number;
  dwellTime: number;
  scans: number;
  stepSize: number;
  startEnergy: number;
  endEnergy: number;
  excitationEnergy?: number;
}

// Main XPS/XRF Interface Component
export const ChemicalAnalysisInterface: React.FC = () => {
  const [technique, setTechnique] = useState<'XPS' | 'XRF'>('XPS');
  const [isAcquiring, setIsAcquiring] = useState(false);
  const [currentFile, setCurrentFile] = useState<string>('');
  const [analysisResults, setAnalysisResults] = useState<any>(null);

  return (
    <div className="w-full max-w-7xl mx-auto p-4 space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Atom className="w-6 h-6 text-purple-600" />
              </div>
              <div>
                <CardTitle>Surface & Elemental Analysis</CardTitle>
                <CardDescription>XPS & XRF Spectroscopy System</CardDescription>
              </div>
            </div>
            <Badge variant={isAcquiring ? "default" : "outline"}>
              {isAcquiring ? "Acquiring" : "Ready"}
            </Badge>
          </div>
        </CardHeader>
      </Card>

      <Tabs value={technique} onValueChange={(v) => setTechnique(v as 'XPS' | 'XRF')}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="XPS" className="flex items-center gap-2">
            <Layers className="w-4 h-4" />
            XPS Analysis
          </TabsTrigger>
          <TabsTrigger value="XRF" className="flex items-center gap-2">
            <Zap className="w-4 h-4" />
            XRF Analysis
          </TabsTrigger>
        </TabsList>

        <TabsContent value="XPS" className="space-y-4">
          <XPSInterface />
        </TabsContent>

        <TabsContent value="XRF" className="space-y-4">
          <XRFInterface />
        </TabsContent>
      </Tabs>
    </div>
  );
};

// XPS Interface Component
const XPSInterface: React.FC = () => {
  const [xpsParams, setXpsParams] = useState<AcquisitionParams>({
    technique: 'XPS',
    source: 'Al Kα',
    passEnergy: 20,
    dwellTime: 50,
    scans: 10,
    stepSize: 0.1,
    startEnergy: 0,
    endEnergy: 1200,
  });

  const [selectedPeak, setSelectedPeak] = useState<XPSPeak | null>(null);
  const [showDepthProfile, setShowDepthProfile] = useState(false);
  const [backgroundType, setBackgroundType] = useState<'Shirley' | 'Tougaard' | 'Linear'>('Shirley');

  // Mock data
  const xpsSpectrum = useMemo(() => {
    const data = [];
    for (let be = 0; be <= 1200; be += 0.5) {
      let intensity = Math.random() * 100 + 500;
      
      // Add peaks
      if (Math.abs(be - 284.5) < 5) intensity += 8000 * Math.exp(-((be - 284.5) ** 2) / 2); // C 1s
      if (Math.abs(be - 532.5) < 5) intensity += 5000 * Math.exp(-((be - 532.5) ** 2) / 3); // O 1s
      if (Math.abs(be - 103.4) < 5) intensity += 3000 * Math.exp(-((be - 103.4) ** 2) / 2); // Si 2p
      if (Math.abs(be - 399.5) < 5) intensity += 2000 * Math.exp(-((be - 399.5) ** 2) / 2); // N 1s
      
      data.push({ be, intensity, background: intensity * 0.1 + 400 });
    }
    return data;
  }, []);

  const xpsPeaks: XPSPeak[] = [
    { id: '1', position: 284.5, area: 15000, fwhm: 1.4, element: 'C', orbital: '1s', chemicalState: 'C-C', asymmetry: 0.05, shape: 'Voigt' },
    { id: '2', position: 532.5, area: 12000, fwhm: 1.6, element: 'O', orbital: '1s', chemicalState: 'O-C', asymmetry: 0.0, shape: 'Voigt' },
    { id: '3', position: 103.4, area: 8000, fwhm: 1.5, element: 'Si', orbital: '2p', chemicalState: 'SiO2', asymmetry: 0.0, shape: 'Voigt' },
    { id: '4', position: 399.5, area: 5000, fwhm: 1.7, element: 'N', orbital: '1s', chemicalState: 'N-C', asymmetry: 0.0, shape: 'Gaussian' },
  ];

  const composition: ElementComposition[] = [
    { element: 'C', concentration: 35.2, error: 1.5 },
    { element: 'O', concentration: 42.8, error: 2.0 },
    { element: 'Si', concentration: 15.3, error: 1.2 },
    { element: 'N', concentration: 6.7, error: 0.8 },
  ];

  return (
    <div className="space-y-4">
      {/* Acquisition Parameters */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Acquisition Parameters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <Label>X-ray Source</Label>
              <Select value={xpsParams.source} onValueChange={(v) => setXpsParams({...xpsParams, source: v})}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Al Kα">Al Kα (1486.6 eV)</SelectItem>
                  <SelectItem value="Mg Kα">Mg Kα (1253.6 eV)</SelectItem>
                  <SelectItem value="Monochromatic Al">Monochromatic Al Kα</SelectItem>
                  <SelectItem value="Ag Lα">Ag Lα (2984.3 eV)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label>Pass Energy (eV)</Label>
              <Select value={xpsParams.passEnergy?.toString()} onValueChange={(v) => setXpsParams({...xpsParams, passEnergy: parseInt(v)})}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="10">10 (High Resolution)</SelectItem>
                  <SelectItem value="20">20 (Standard)</SelectItem>
                  <SelectItem value="50">50 (Survey)</SelectItem>
                  <SelectItem value="100">100 (Fast Survey)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label>Number of Scans</Label>
              <Input
                type="number"
                value={xpsParams.scans}
                onChange={(e) => setXpsParams({...xpsParams, scans: parseInt(e.target.value)})}
                min={1}
                max={100}
              />
            </div>

            <div>
              <Label>Start Energy (eV)</Label>
              <Input
                type="number"
                value={xpsParams.startEnergy}
                onChange={(e) => setXpsParams({...xpsParams, startEnergy: parseFloat(e.target.value)})}
              />
            </div>

            <div>
              <Label>End Energy (eV)</Label>
              <Input
                type="number"
                value={xpsParams.endEnergy}
                onChange={(e) => setXpsParams({...xpsParams, endEnergy: parseFloat(e.target.value)})}
              />
            </div>

            <div>
              <Label>Step Size (eV)</Label>
              <Input
                type="number"
                value={xpsParams.stepSize}
                step="0.01"
                onChange={(e) => setXpsParams({...xpsParams, stepSize: parseFloat(e.target.value)})}
              />
            </div>
          </div>

          <div className="flex gap-2 mt-4">
            <Button className="flex items-center gap-2">
              <Play className="w-4 h-4" />
              Start Acquisition
            </Button>
            <Button variant="outline" className="flex items-center gap-2">
              <Upload className="w-4 h-4" />
              Load Spectrum
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Spectrum Display */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">XPS Spectrum</CardTitle>
            <div className="flex gap-2">
              <Select value={backgroundType} onValueChange={(v) => setBackgroundType(v as any)}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Shirley">Shirley</SelectItem>
                  <SelectItem value="Tougaard">Tougaard</SelectItem>
                  <SelectItem value="Linear">Linear</SelectItem>
                </SelectContent>
              </Select>
              <Button size="sm" variant="outline">
                Subtract Background
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={xpsSpectrum}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="be" 
                label={{ value: "Binding Energy (eV)", position: "insideBottom", offset: -5 }}
                reversed
              />
              <YAxis label={{ value: "Intensity (counts/s)", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="intensity" 
                stroke="#8b5cf6" 
                strokeWidth={1.5} 
                dot={false}
                name="Spectrum"
              />
              <Line 
                type="monotone" 
                dataKey="background" 
                stroke="#ef4444" 
                strokeWidth={1}
                strokeDasharray="5 5"
                dot={false}
                name="Background"
              />
              {xpsPeaks.map(peak => (
                <ReferenceLine 
                  key={peak.id}
                  x={peak.position} 
                  stroke="#10b981"
                  strokeDasharray="3 3"
                  label={`${peak.element} ${peak.orbital}`}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 gap-4">
        {/* Peak List */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Identified Peaks</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Element</TableHead>
                  <TableHead>Orbital</TableHead>
                  <TableHead>BE (eV)</TableHead>
                  <TableHead>Area</TableHead>
                  <TableHead>FWHM</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {xpsPeaks.map((peak) => (
                  <TableRow 
                    key={peak.id}
                    className="cursor-pointer hover:bg-gray-50"
                    onClick={() => setSelectedPeak(peak)}
                  >
                    <TableCell className="font-medium">{peak.element}</TableCell>
                    <TableCell>{peak.orbital}</TableCell>
                    <TableCell>{peak.position.toFixed(1)}</TableCell>
                    <TableCell>{peak.area.toFixed(0)}</TableCell>
                    <TableCell>{peak.fwhm.toFixed(2)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        {/* Quantification Results */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Atomic Composition</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={composition}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="element" />
                <YAxis label={{ value: "Atomic %", angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Bar dataKey="concentration" fill="#8b5cf6" />
              </BarChart>
            </ResponsiveContainer>

            <div className="mt-4 space-y-2">
              {composition.map((comp) => (
                <div key={comp.element} className="flex justify-between items-center">
                  <span className="font-medium">{comp.element}:</span>
                  <span>{comp.concentration.toFixed(1)} ± {comp.error.toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Chemical State Analysis */}
      {selectedPeak && (
        <ChemicalStateAnalysis peak={selectedPeak} onClose={() => setSelectedPeak(null)} />
      )}

      {/* Depth Profile */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">Depth Profile Analysis</CardTitle>
            <Switch
              checked={showDepthProfile}
              onCheckedChange={setShowDepthProfile}
            />
          </div>
        </CardHeader>
        {showDepthProfile && (
          <CardContent>
            <DepthProfileViewer />
          </CardContent>
        )}
      </Card>
    </div>
  );
};

// XRF Interface Component
const XRFInterface: React.FC = () => {
  const [xrfParams, setXrfParams] = useState<AcquisitionParams>({
    technique: 'XRF',
    source: 'X-ray Tube',
    excitationEnergy: 50,
    dwellTime: 300,
    scans: 1,
    stepSize: 0.01,
    startEnergy: 0.1,
    endEnergy: 20,
  });

  // Mock XRF spectrum data
  const xrfSpectrum = useMemo(() => {
    const data = [];
    for (let energy = 0.1; energy <= 20; energy += 0.01) {
      let counts = Math.random() * 50 + 100;
      
      // Add element peaks
      if (Math.abs(energy - 1.74) < 0.05) counts += 5000 * Math.exp(-((energy - 1.74) ** 2) / 0.001); // Si Kα
      if (Math.abs(energy - 0.525) < 0.05) counts += 2000 * Math.exp(-((energy - 0.525) ** 2) / 0.001); // O Kα
      if (Math.abs(energy - 6.404) < 0.05) counts += 3000 * Math.exp(-((energy - 6.404) ** 2) / 0.002); // Fe Kα
      if (Math.abs(energy - 9.252) < 0.05) counts += 1500 * Math.exp(-((energy - 9.252) ** 2) / 0.002); // Ga Kα
      
      // Bremsstrahlung background
      counts += (50 - energy) * 5 * (energy > 0 ? 1 : 0);
      
      data.push({ energy, counts });
    }
    return data;
  }, []);

  const xrfPeaks: XRFPeak[] = [
    { id: '1', energy: 1.740, intensity: 5000, element: 'Si', line: 'Kα', escapePeak: false, sumPeak: false },
    { id: '2', energy: 0.525, intensity: 2000, element: 'O', line: 'Kα', escapePeak: false, sumPeak: false },
    { id: '3', energy: 6.404, intensity: 3000, element: 'Fe', line: 'Kα', escapePeak: false, sumPeak: false },
    { id: '4', energy: 9.252, intensity: 1500, element: 'Ga', line: 'Kα', escapePeak: false, sumPeak: false },
  ];

  const xrfComposition: ElementComposition[] = [
    { element: 'Si', concentration: 45.5, error: 2.0, line: 'Kα' },
    { element: 'O', concentration: 30.2, error: 1.5, line: 'Kα' },
    { element: 'Fe', concentration: 15.8, error: 1.0, line: 'Kα' },
    { element: 'Ga', concentration: 8.5, error: 0.5, line: 'Kα' },
  ];

  const detectionLimits = [
    { element: 'Ti', mdl: 25 },
    { element: 'V', mdl: 20 },
    { element: 'Cr', mdl: 18 },
    { element: 'Mn', mdl: 15 },
    { element: 'Co', mdl: 12 },
  ];

  return (
    <div className="space-y-4">
      {/* Acquisition Parameters */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">XRF Parameters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <Label>Excitation Source</Label>
              <Select value={xrfParams.source} onValueChange={(v) => setXrfParams({...xrfParams, source: v})}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="X-ray Tube">X-ray Tube (50 kV)</SelectItem>
                  <SelectItem value="Radioisotope">Radioisotope Source</SelectItem>
                  <SelectItem value="Synchrotron">Synchrotron</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label>Measurement Time (s)</Label>
              <Input
                type="number"
                value={xrfParams.dwellTime}
                onChange={(e) => setXrfParams({...xrfParams, dwellTime: parseInt(e.target.value)})}
                min={10}
                max={3600}
              />
            </div>

            <div>
              <Label>Atmosphere</Label>
              <Select defaultValue="air">
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="air">Air</SelectItem>
                  <SelectItem value="vacuum">Vacuum</SelectItem>
                  <SelectItem value="helium">Helium</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="flex gap-2 mt-4">
            <Button className="flex items-center gap-2">
              <Play className="w-4 h-4" />
              Start Measurement
            </Button>
            <Button variant="outline" className="flex items-center gap-2">
              <Settings className="w-4 h-4" />
              Calibrate
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* XRF Spectrum */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">XRF Spectrum</CardTitle>
            <div className="flex gap-2">
              <Button size="sm" variant="outline">
                Peak Search
              </Button>
              <Button size="sm" variant="outline">
                Auto Identify
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={xrfSpectrum}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="energy" 
                label={{ value: "Energy (keV)", position: "insideBottom", offset: -5 }}
              />
              <YAxis 
                scale="log"
                domain={[10, 10000]}
                label={{ value: "Counts", angle: -90, position: "insideLeft" }} 
              />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="counts" 
                stroke="#ef4444" 
                strokeWidth={1.5} 
                dot={false}
                name="Spectrum"
              />
              {xrfPeaks.map(peak => (
                <ReferenceLine 
                  key={peak.id}
                  x={peak.energy} 
                  stroke="#10b981"
                  strokeDasharray="3 3"
                  label={`${peak.element} ${peak.line}`}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 gap-4">
        {/* Element List */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Detected Elements</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Element</TableHead>
                  <TableHead>Line</TableHead>
                  <TableHead>Energy (keV)</TableHead>
                  <TableHead>Intensity</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {xrfPeaks.map((peak) => (
                  <TableRow key={peak.id}>
                    <TableCell className="font-medium">{peak.element}</TableCell>
                    <TableCell>{peak.line}</TableCell>
                    <TableCell>{peak.energy.toFixed(3)}</TableCell>
                    <TableCell>{peak.intensity.toFixed(0)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        {/* Quantification */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Elemental Composition</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={xrfComposition}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="element" />
                <YAxis label={{ value: "Weight %", angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Bar dataKey="concentration" fill="#ef4444" />
              </BarChart>
            </ResponsiveContainer>

            <div className="mt-4 space-y-2">
              {xrfComposition.map((comp) => (
                <div key={comp.element} className="flex justify-between items-center">
                  <span className="font-medium">{comp.element} ({comp.line}):</span>
                  <span>{comp.concentration.toFixed(1)} ± {comp.error.toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detection Limits */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Method Detection Limits</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-5 gap-4">
            {detectionLimits.map((limit) => (
              <div key={limit.element} className="text-center p-2 border rounded">
                <div className="text-lg font-bold">{limit.element}</div>
                <div className="text-sm text-gray-600">{limit.mdl} ppm</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Chemical State Analysis Dialog
const ChemicalStateAnalysis: React.FC<{ peak: XPSPeak; onClose: () => void }> = ({ peak, onClose }) => {
  const chemicalStates: ChemicalState[] = [
    { state: 'C-C', position: 284.5, percentage: 60, reference: 284.5 },
    { state: 'C-O', position: 286.0, percentage: 25, reference: 286.0 },
    { state: 'C=O', position: 288.0, percentage: 10, reference: 288.0 },
    { state: 'O-C=O', position: 289.0, percentage: 5, reference: 289.0 },
  ];

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Chemical State Analysis - {peak.element} {peak.orbital}</DialogTitle>
          <DialogDescription>
            Peak deconvolution and chemical state identification
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Fitted Peak */}
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={generatePeakFit(peak)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="be" label={{ value: "Binding Energy (eV)", position: "insideBottom", offset: -5 }} />
                <YAxis label={{ value: "Intensity", angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Line type="monotone" dataKey="raw" stroke="#8b5cf6" strokeWidth={2} dot={false} name="Raw Data" />
                <Line type="monotone" dataKey="fitted" stroke="#ef4444" strokeWidth={2} dot={false} name="Fitted" />
                <Line type="monotone" dataKey="component1" stroke="#10b981" strokeWidth={1} strokeDasharray="5 5" dot={false} name="C-C" />
                <Line type="monotone" dataKey="component2" stroke="#f59e0b" strokeWidth={1} strokeDasharray="5 5" dot={false} name="C-O" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Chemical States Table */}
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Chemical State</TableHead>
                <TableHead>Position (eV)</TableHead>
                <TableHead>Percentage</TableHead>
                <TableHead>Reference (eV)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {chemicalStates.map((state, idx) => (
                <TableRow key={idx}>
                  <TableCell className="font-medium">{state.state}</TableCell>
                  <TableCell>{state.position.toFixed(1)}</TableCell>
                  <TableCell>{state.percentage}%</TableCell>
                  <TableCell>{state.reference.toFixed(1)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          {/* Fit Parameters */}
          <div className="grid grid-cols-2 gap-4 p-4 bg-gray-50 rounded">
            <div>
              <div className="text-sm font-medium">Peak Shape</div>
              <div className="text-sm text-gray-600">{peak.shape}</div>
            </div>
            <div>
              <div className="text-sm font-medium">FWHM</div>
              <div className="text-sm text-gray-600">{peak.fwhm.toFixed(2)} eV</div>
            </div>
            <div>
              <div className="text-sm font-medium">Asymmetry</div>
              <div className="text-sm text-gray-600">{peak.asymmetry.toFixed(3)}</div>
            </div>
            <div>
              <div className="text-sm font-medium">R²</div>
              <div className="text-sm text-gray-600">0.998</div>
            </div>
          </div>
        </div>

        <div className="flex justify-end gap-2 mt-4">
          <Button variant="outline" onClick={onClose}>Close</Button>
          <Button>Export Results</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

// Depth Profile Viewer
const DepthProfileViewer: React.FC = () => {
  const depthData = useMemo(() => {
    const data = [];
    for (let depth = 0; depth <= 100; depth += 5) {
      data.push({
        depth,
        Si: 15 + 30 * Math.exp(-depth / 20),
        O: 50 - 20 * Math.exp(-depth / 20),
        C: 30 * Math.exp(-depth / 10),
        N: 5 + 2 * Math.sin(depth / 10),
      });
    }
    return data;
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Label>Etch Rate (nm/s):</Label>
          <Input type="number" defaultValue="0.1" className="w-20" />
        </div>
        <Button size="sm" variant="outline">
          Start Depth Profile
        </Button>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={depthData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="depth" label={{ value: "Depth (nm)", position: "insideBottom", offset: -5 }} />
          <YAxis label={{ value: "Atomic %", angle: -90, position: "insideLeft" }} />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="Si" stroke="#8b5cf6" strokeWidth={2} />
          <Line type="monotone" dataKey="O" stroke="#ef4444" strokeWidth={2} />
          <Line type="monotone" dataKey="C" stroke="#10b981" strokeWidth={2} />
          <Line type="monotone" dataKey="N" stroke="#f59e0b" strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

// Helper function to generate mock peak fit data
function generatePeakFit(peak: XPSPeak) {
  const data = [];
  const center = peak.position;
  const width = peak.fwhm;
  
  for (let be = center - 10; be <= center + 10; be += 0.1) {
    const raw = 1000 + 5000 * Math.exp(-((be - center) ** 2) / (2 * width ** 2)) + Math.random() * 100;
    const component1 = 3000 * Math.exp(-((be - center) ** 2) / (2 * width ** 2));
    const component2 = 1500 * Math.exp(-((be - (center + 1.5)) ** 2) / (2 * width ** 2));
    const fitted = 1000 + component1 + component2;
    
    data.push({ be, raw, fitted, component1, component2 });
  }
  
  return data;
}

export default ChemicalAnalysisInterface;
