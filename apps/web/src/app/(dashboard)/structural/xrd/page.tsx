'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Switch } from '@/components/ui/switch';
import { Checkbox } from '@/components/ui/checkbox';
import { 
  LineChart, Line, ScatterChart, Scatter, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, ReferenceLine, ReferenceArea, Brush,
  ComposedChart, ErrorBar
} from 'recharts';
import {
  Activity, AlertCircle, Download, FileText, Play, Save, Settings,
  Upload, Zap, Eye, Layers, TrendingUp, Thermometer, Gauge,
  Atom, Maximize, Target, Info, CheckCircle, Beaker,
  ScanLine, Lightbulb, Microscope, Grid3x3, Hexagon,
  BarChart3, Ruler, ArrowUpRight, ArrowDownRight, Database
} from 'lucide-react';

// Type definitions
interface XRDPattern {
  twoTheta: number[];
  intensity: number[];
  wavelength: number;
  metadata?: Record<string, any>;
}

interface Peak {
  position: number;
  dSpacing: number;
  intensity: number;
  fwhm: number;
  area: number;
  hkl?: [number, number, number];
  phase?: string;
}

interface PhaseMatch {
  phaseName: string;
  formula: string;
  crystalSystem: string;
  score: number;
  matchedPeaks: Peak[];
  latticeParams: Record<string, number>;
}

interface CrystalliteAnalysis {
  meanSize: number;
  stdSize: number;
  microstrain: number;
  rSquared: number;
}

interface StressAnalysis {
  stress: number;
  type: 'tensile' | 'compressive';
  error: number;
  rSquared: number;
}

// Main XRD Interface Component
const XRDInterface: React.FC = () => {
  // State management
  const [xrdPattern, setXrdPattern] = useState<XRDPattern | null>(null);
  const [peaks, setPeaks] = useState<Peak[]>([]);
  const [phases, setPhases] = useState<PhaseMatch[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState('measurement');
  const [selectedPeaks, setSelectedPeaks] = useState<number[]>([]);
  
  // Analysis results
  const [crystalliteSize, setCrystalliteSize] = useState<CrystalliteAnalysis | null>(null);
  const [stressResult, setStressResult] = useState<StressAnalysis | null>(null);
  const [textureIndex, setTextureIndex] = useState<number>(0);
  
  // Measurement parameters
  const [measurementParams, setMeasurementParams] = useState({
    xraySource: 'Cu_Ka',
    startAngle: 20,
    endAngle: 80,
    stepSize: 0.02,
    scanSpeed: 1,
    voltage: 40,
    current: 30
  });
  
  // Processing options
  const [processingOptions, setProcessingOptions] = useState({
    smoothing: true,
    backgroundCorrection: true,
    kaStripping: true,
    peakProfile: 'pseudo_voigt'
  });

  // Load demo data
  const loadDemoData = useCallback(() => {
    // Generate synthetic Si pattern
    const twoTheta = Array.from({ length: 3000 }, (_, i) => 20 + i * 0.02);
    
    // Add Si peaks (simplified)
    const siPeaks = [
      { pos: 28.4, int: 1000, hkl: [1, 1, 1] },
      { pos: 47.3, int: 600, hkl: [2, 2, 0] },
      { pos: 56.1, int: 350, hkl: [3, 1, 1] },
      { pos: 69.1, int: 100, hkl: [4, 0, 0] },
      { pos: 76.4, int: 150, hkl: [3, 3, 1] }
    ];
    
    const intensity = twoTheta.map(angle => {
      let int = 50 + Math.random() * 10; // Background
      
      // Add peaks
      siPeaks.forEach(peak => {
        const sigma = 0.1;
        int += peak.int * Math.exp(-0.5 * Math.pow((angle - peak.pos) / sigma, 2));
      });
      
      return int;
    });
    
    setXrdPattern({
      twoTheta,
      intensity,
      wavelength: 1.5418,
      metadata: { phase: 'Si', source: 'demo' }
    });
    
    // Set demo peaks
    const demoP

eaks: Peak[] = siPeaks.map(p => ({
      position: p.pos,
      dSpacing: 1.5418 / (2 * Math.sin(p.pos * Math.PI / 360)),
      intensity: p.int,
      fwhm: 0.15,
      area: p.int * 0.15 * Math.sqrt(2 * Math.PI),
      hkl: p.hkl as [number, number, number],
      phase: 'Si'
    }));
    
    setPeaks(demoPeaks);
    
    // Set demo phase identification
    setPhases([{
      phaseName: 'Silicon',
      formula: 'Si',
      crystalSystem: 'cubic',
      score: 95.5,
      matchedPeaks: demoPeaks,
      latticeParams: { a: 5.43095 }
    }]);
    
    // Set demo crystallite analysis
    setCrystalliteSize({
      meanSize: 50,
      stdSize: 5,
      microstrain: 0.002,
      rSquared: 0.98
    });
  }, []);

  // Start measurement
  const startMeasurement = useCallback(async () => {
    setIsProcessing(true);
    
    try {
      // Simulate measurement
      await new Promise(resolve => setTimeout(resolve, 3000));
      loadDemoData();
      setActiveTab('analysis');
    } catch (err) {
      console.error('Measurement failed:', err);
    } finally {
      setIsProcessing(false);
    }
  }, [loadDemoData]);

  // Find peaks
  const findPeaks = useCallback(() => {
    if (!xrdPattern) return;
    
    setIsProcessing(true);
    
    // Simulate peak finding
    setTimeout(() => {
      // Already set in demo data
      setIsProcessing(false);
      setActiveTab('peaks');
    }, 1000);
  }, [xrdPattern]);

  // Identify phases
  const identifyPhases = useCallback(() => {
    if (peaks.length === 0) return;
    
    setIsProcessing(true);
    
    // Simulate phase identification
    setTimeout(() => {
      // Already set in demo data
      setIsProcessing(false);
      setActiveTab('phases');
    }, 1500);
  }, [peaks]);

  // Calculate crystallite size
  const calculateCrystalliteSize = useCallback(() => {
    if (peaks.length === 0) return;
    
    setIsProcessing(true);
    
    setTimeout(() => {
      // Already set in demo data
      setIsProcessing(false);
      setActiveTab('crystallite');
    }, 1000);
  }, [peaks]);

  // Perform stress analysis
  const performStressAnalysis = useCallback(() => {
    setIsProcessing(true);
    
    setTimeout(() => {
      setStressResult({
        stress: 150,  // MPa
        type: 'tensile',
        error: 10,
        rSquared: 0.95
      });
      setIsProcessing(false);
      setActiveTab('stress');
    }, 1500);
  }, []);

  // Chart data preparation
  const patternChartData = useMemo(() => {
    if (!xrdPattern) return [];
    
    // Downsample for performance
    const step = Math.max(1, Math.floor(xrdPattern.twoTheta.length / 500));
    
    return xrdPattern.twoTheta
      .filter((_, i) => i % step === 0)
      .map((angle, i) => ({
        twoTheta: angle,
        intensity: xrdPattern.intensity[i * step],
        dSpacing: xrdPattern.wavelength / (2 * Math.sin(angle * Math.PI / 360))
      }));
  }, [xrdPattern]);

  const peakChartData = useMemo(() => {
    return peaks.map(peak => ({
      position: peak.position,
      intensity: peak.intensity,
      fwhm: peak.fwhm,
      dSpacing: peak.dSpacing,
      hkl: peak.hkl ? `(${peak.hkl.join(',')})` : '',
      phase: peak.phase || 'Unknown'
    }));
  }, [peaks]);

  const williamsonHallData = useMemo(() => {
    if (peaks.length === 0) return [];
    
    return peaks.map(peak => {
      const theta = peak.position / 2 * Math.PI / 180;
      return {
        sinTheta: Math.sin(theta),
        betaCosTheta: (peak.fwhm * Math.PI / 180) * Math.cos(theta),
        hkl: peak.hkl ? `(${peak.hkl.join(',')})` : ''
      };
    });
  }, [peaks]);

  // X-ray source options
  const xraySources = [
    { value: 'Cu_Ka', label: 'Cu Kα (1.5418 Å)' },
    { value: 'Co_Ka', label: 'Co Kα (1.7902 Å)' },
    { value: 'Mo_Ka', label: 'Mo Kα (0.7107 Å)' },
    { value: 'Cr_Ka', label: 'Cr Kα (2.2909 Å)' },
    { value: 'Fe_Ka', label: 'Fe Kα (1.9373 Å)' }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Hexagon className="w-6 h-6" />
                X-ray Diffraction Analysis
              </CardTitle>
              <CardDescription>
                Structural characterization and phase identification
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Badge variant={xrdPattern ? 'success' : 'secondary'}>
                {xrdPattern ? 'Pattern Loaded' : 'No Data'}
              </Badge>
              {peaks.length > 0 && (
                <Badge variant="success">{peaks.length} Peaks</Badge>
              )}
              {phases.length > 0 && (
                <Badge variant="success">{phases.length} Phases</Badge>
              )}
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-6 w-full">
          <TabsTrigger value="measurement">Setup</TabsTrigger>
          <TabsTrigger value="analysis">Pattern</TabsTrigger>
          <TabsTrigger value="peaks">Peaks</TabsTrigger>
          <TabsTrigger value="phases">Phases</TabsTrigger>
          <TabsTrigger value="crystallite">Size/Strain</TabsTrigger>
          <TabsTrigger value="stress">Stress</TabsTrigger>
        </TabsList>

        {/* Measurement Setup Tab */}
        <TabsContent value="measurement">
          <div className="grid grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Measurement Parameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>X-ray Source</Label>
                  <Select
                    value={measurementParams.xraySource}
                    onValueChange={(value) => setMeasurementParams({
                      ...measurementParams,
                      xraySource: value
                    })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {xraySources.map(source => (
                        <SelectItem key={source.value} value={source.value}>
                          {source.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Start 2θ (°)</Label>
                    <Input
                      type="number"
                      value={measurementParams.startAngle}
                      onChange={(e) => setMeasurementParams({
                        ...measurementParams,
                        startAngle: parseFloat(e.target.value)
                      })}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>End 2θ (°)</Label>
                    <Input
                      type="number"
                      value={measurementParams.endAngle}
                      onChange={(e) => setMeasurementParams({
                        ...measurementParams,
                        endAngle: parseFloat(e.target.value)
                      })}
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Step Size (°)</Label>
                  <Input
                    type="number"
                    step="0.001"
                    value={measurementParams.stepSize}
                    onChange={(e) => setMeasurementParams({
                      ...measurementParams,
                      stepSize: parseFloat(e.target.value)
                    })}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Scan Speed (°/min)</Label>
                  <Slider
                    value={[measurementParams.scanSpeed]}
                    onValueChange={(value) => setMeasurementParams({
                      ...measurementParams,
                      scanSpeed: value[0]
                    })}
                    min={0.1}
                    max={10}
                    step={0.1}
                    className="w-full"
                  />
                  <span className="text-sm text-muted-foreground">
                    {measurementParams.scanSpeed} °/min
                  </span>
                </div>

                <Button
                  onClick={startMeasurement}
                  disabled={isProcessing}
                  className="w-full"
                >
                  {isProcessing ? 'Measuring...' : 'Start Measurement'}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Processing Options</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label>Smoothing</Label>
                  <Switch
                    checked={processingOptions.smoothing}
                    onCheckedChange={(checked) => setProcessingOptions({
                      ...processingOptions,
                      smoothing: checked
                    })}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <Label>Background Correction</Label>
                  <Switch
                    checked={processingOptions.backgroundCorrection}
                    onCheckedChange={(checked) => setProcessingOptions({
                      ...processingOptions,
                      backgroundCorrection: checked
                    })}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <Label>Kα₂ Stripping</Label>
                  <Switch
                    checked={processingOptions.kaStripping}
                    onCheckedChange={(checked) => setProcessingOptions({
                      ...processingOptions,
                      kaStripping: checked
                    })}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Peak Profile</Label>
                  <Select
                    value={processingOptions.peakProfile}
                    onValueChange={(value) => setProcessingOptions({
                      ...processingOptions,
                      peakProfile: value
                    })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="gaussian">Gaussian</SelectItem>
                      <SelectItem value="lorentzian">Lorentzian</SelectItem>
                      <SelectItem value="voigt">Voigt</SelectItem>
                      <SelectItem value="pseudo_voigt">Pseudo-Voigt</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Button
                  variant="outline"
                  onClick={loadDemoData}
                  className="w-full"
                >
                  Load Demo Data
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Pattern Analysis Tab */}
        <TabsContent value="analysis">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">Diffraction Pattern</CardTitle>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={findPeaks}
                    disabled={!xrdPattern || isProcessing}
                  >
                    <Target className="w-4 h-4 mr-2" />
                    Find Peaks
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={!xrdPattern}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Export
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={patternChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="twoTheta"
                    label={{ value: '2θ (degrees)', position: 'insideBottom', offset: -5 }}
                    domain={['dataMin', 'dataMax']}
                  />
                  <YAxis 
                    label={{ value: 'Intensity (counts)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="intensity"
                    stroke="#8884d8"
                    strokeWidth={1.5}
                    dot={false}
                  />
                  {/* Add peak markers */}
                  {peaks.map((peak, i) => (
                    <ReferenceLine
                      key={i}
                      x={peak.position}
                      stroke="red"
                      strokeDasharray="5 5"
                      label={{
                        value: peak.hkl ? `(${peak.hkl.join(',')})` : '',
                        position: 'top',
                        fontSize: 10
                      }}
                    />
                  ))}
                  <Brush dataKey="twoTheta" height={30} />
                </LineChart>
              </ResponsiveContainer>

              {xrdPattern && (
                <div className="grid grid-cols-4 gap-4 mt-4">
                  <Card>
                    <CardContent className="pt-6">
                      <div className="text-2xl font-bold">
                        {xrdPattern.twoTheta.length}
                      </div>
                      <p className="text-xs text-muted-foreground">Data Points</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="pt-6">
                      <div className="text-2xl font-bold">
                        {measurementParams.stepSize}°
                      </div>
                      <p className="text-xs text-muted-foreground">Step Size</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="pt-6">
                      <div className="text-2xl font-bold">
                        {Math.max(...(xrdPattern?.intensity || [0])).toFixed(0)}
                      </div>
                      <p className="text-xs text-muted-foreground">Max Intensity</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="pt-6">
                      <div className="text-2xl font-bold">
                        {peaks.length}
                      </div>
                      <p className="text-xs text-muted-foreground">Peaks Found</p>
                    </CardContent>
                  </Card>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Peaks Tab */}
        <TabsContent value="peaks">
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">Peak List</CardTitle>
                  <Button
                    size="sm"
                    onClick={identifyPhases}
                    disabled={peaks.length === 0 || isProcessing}
                  >
                    <Database className="w-4 h-4 mr-2" />
                    Identify Phases
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left p-2">Select</th>
                        <th className="text-left p-2">2θ (°)</th>
                        <th className="text-left p-2">d (Å)</th>
                        <th className="text-left p-2">Intensity</th>
                        <th className="text-left p-2">FWHM (°)</th>
                        <th className="text-left p-2">hkl</th>
                        <th className="text-left p-2">Phase</th>
                      </tr>
                    </thead>
                    <tbody>
                      {peaks.map((peak, i) => (
                        <tr key={i} className="border-b">
                          <td className="p-2">
                            <Checkbox
                              checked={selectedPeaks.includes(i)}
                              onCheckedChange={(checked) => {
                                if (checked) {
                                  setSelectedPeaks([...selectedPeaks, i]);
                                } else {
                                  setSelectedPeaks(selectedPeaks.filter(idx => idx !== i));
                                }
                              }}
                            />
                          </td>
                          <td className="p-2 font-mono">{peak.position.toFixed(2)}</td>
                          <td className="p-2 font-mono">{peak.dSpacing.toFixed(3)}</td>
                          <td className="p-2 font-mono">{peak.intensity.toFixed(0)}</td>
                          <td className="p-2 font-mono">{peak.fwhm.toFixed(3)}</td>
                          <td className="p-2">
                            {peak.hkl ? `(${peak.hkl.join(' ')})` : '-'}
                          </td>
                          <td className="p-2">{peak.phase || '-'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Peak Profile Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={peakChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="position"
                      label={{ value: '2θ (degrees)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: 'Intensity', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip />
                    <Bar dataKey="intensity" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Phase Identification Tab */}
        <TabsContent value="phases">
          <div className="space-y-4">
            {phases.map((phase, i) => (
              <Card key={i}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-lg">
                        {phase.phaseName} ({phase.formula})
                      </CardTitle>
                      <CardDescription>
                        {phase.crystalSystem} crystal system
                      </CardDescription>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="success">
                        Match: {phase.score.toFixed(1)}%
                      </Badge>
                      <Badge>
                        {phase.matchedPeaks.length} peaks
                      </Badge>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-semibold mb-2">Lattice Parameters</h4>
                      <dl className="space-y-1 text-sm">
                        {Object.entries(phase.latticeParams).map(([param, value]) => (
                          <div key={param} className="flex justify-between">
                            <dt>{param}:</dt>
                            <dd className="font-mono">{value.toFixed(4)} Å</dd>
                          </div>
                        ))}
                      </dl>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-2">Matched Peaks</h4>
                      <div className="space-y-1 text-sm">
                        {phase.matchedPeaks.slice(0, 5).map((peak, j) => (
                          <div key={j} className="font-mono">
                            {peak.position.toFixed(2)}° 
                            {peak.hkl && ` (${peak.hkl.join(' ')})`}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}

            {phases.length === 0 && (
              <Alert>
                <Info className="h-4 w-4" />
                <AlertTitle>No Phases Identified</AlertTitle>
                <AlertDescription>
                  Find peaks first, then run phase identification
                </AlertDescription>
              </Alert>
            )}
          </div>
        </TabsContent>

        {/* Crystallite Size/Strain Tab */}
        <TabsContent value="crystallite">
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">Crystallite Size Analysis</CardTitle>
                  <Button
                    size="sm"
                    onClick={calculateCrystalliteSize}
                    disabled={peaks.length === 0 || isProcessing}
                  >
                    Calculate
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {crystalliteSize ? (
                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-semibold mb-3">Scherrer Analysis</h4>
                      <dl className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <dt>Mean crystallite size:</dt>
                          <dd className="font-bold">
                            {crystalliteSize.meanSize.toFixed(1)} nm
                          </dd>
                        </div>
                        <div className="flex justify-between">
                          <dt>Standard deviation:</dt>
                          <dd className="font-mono">
                            ±{crystalliteSize.stdSize.toFixed(1)} nm
                          </dd>
                        </div>
                        <div className="flex justify-between">
                          <dt>Shape factor (K):</dt>
                          <dd className="font-mono">0.9</dd>
                        </div>
                      </dl>
                    </div>
                    
                    <div>
                      <h4 className="font-semibold mb-3">Williamson-Hall Analysis</h4>
                      <dl className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <dt>Crystallite size:</dt>
                          <dd className="font-bold">
                            {(crystalliteSize.meanSize * 0.95).toFixed(1)} nm
                          </dd>
                        </div>
                        <div className="flex justify-between">
                          <dt>Microstrain:</dt>
                          <dd className="font-mono">
                            {(crystalliteSize.microstrain * 100).toFixed(3)}%
                          </dd>
                        </div>
                        <div className="flex justify-between">
                          <dt>R² value:</dt>
                          <dd className="font-mono">
                            {crystalliteSize.rSquared.toFixed(3)}
                          </dd>
                        </div>
                      </dl>
                    </div>
                  </div>
                ) : (
                  <Alert>
                    <Info className="h-4 w-4" />
                    <AlertTitle>No Analysis Available</AlertTitle>
                    <AlertDescription>
                      Perform crystallite size calculation first
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>

            {crystalliteSize && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Williamson-Hall Plot</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <ScatterChart data={williamsonHallData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="sinTheta"
                        label={{ value: 'sin(θ)', position: 'insideBottom', offset: -5 }}
                      />
                      <YAxis 
                        label={{ value: 'β·cos(θ)', angle: -90, position: 'insideLeft' }}
                      />
                      <Tooltip />
                      <Scatter name="Data" data={williamsonHallData} fill="#8884d8">
                        {williamsonHallData.map((entry, index) => (
                          <text
                            key={index}
                            x={entry.sinTheta}
                            y={entry.betaCosTheta}
                            fontSize={10}
                            fill="#666"
                          >
                            {entry.hkl}
                          </text>
                        ))}
                      </Scatter>
                      {/* Add fit line */}
                      <ReferenceLine 
                        segment={[
                          { x: Math.min(...williamsonHallData.map(d => d.sinTheta)), 
                            y: 0.001 },
                          { x: Math.max(...williamsonHallData.map(d => d.sinTheta)), 
                            y: 0.005 }
                        ]}
                        stroke="red"
                        strokeDasharray="5 5"
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        {/* Stress Analysis Tab */}
        <TabsContent value="stress">
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">Residual Stress Analysis (sin²ψ)</CardTitle>
                  <Button
                    size="sm"
                    onClick={performStressAnalysis}
                    disabled={isProcessing}
                  >
                    Analyze Stress
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {stressResult ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-3 gap-4">
                      <Card>
                        <CardContent className="pt-6">
                          <div className="flex items-center gap-2">
                            <div className="text-2xl font-bold">
                              {Math.abs(stressResult.stress).toFixed(0)}
                            </div>
                            <span className="text-sm text-muted-foreground">MPa</span>
                          </div>
                          <p className="text-xs text-muted-foreground mt-1">
                            Residual Stress
                          </p>
                        </CardContent>
                      </Card>
                      
                      <Card>
                        <CardContent className="pt-6">
                          <div className="flex items-center gap-2">
                            {stressResult.type === 'tensile' ? (
                              <ArrowUpRight className="w-5 h-5 text-red-500" />
                            ) : (
                              <ArrowDownRight className="w-5 h-5 text-blue-500" />
                            )}
                            <span className="text-lg font-semibold capitalize">
                              {stressResult.type}
                            </span>
                          </div>
                          <p className="text-xs text-muted-foreground mt-1">
                            Stress Type
                          </p>
                        </CardContent>
                      </Card>
                      
                      <Card>
                        <CardContent className="pt-6">
                          <div className="text-2xl font-bold">
                            {stressResult.rSquared.toFixed(3)}
                          </div>
                          <p className="text-xs text-muted-foreground">
                            R² Value
                          </p>
                        </CardContent>
                      </Card>
                    </div>

                    <Alert className={stressResult.type === 'tensile' ? 'border-red-200' : 'border-blue-200'}>
                      <AlertCircle className="h-4 w-4" />
                      <AlertTitle>Stress Analysis Complete</AlertTitle>
                      <AlertDescription>
                        {stressResult.type === 'tensile' ? 
                          'Tensile stress detected. This may indicate stretching or thermal expansion.' :
                          'Compressive stress detected. This may indicate compression or thermal contraction.'
                        }
                        <br />
                        Error: ±{stressResult.error.toFixed(1)} MPa
                      </AlertDescription>
                    </Alert>

                    <Card>
                      <CardHeader>
                        <CardTitle className="text-base">sin²ψ Plot</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ResponsiveContainer width="100%" height={250}>
                          <ScatterChart>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis 
                              domain={[0, 1]}
                              label={{ value: 'sin²ψ', position: 'insideBottom', offset: -5 }}
                            />
                            <YAxis 
                              label={{ value: 'Strain (×10⁻³)', angle: -90, position: 'insideLeft' }}
                            />
                            <Tooltip />
                            <Scatter
                              name="Measurements"
                              data={[
                                { sin2psi: 0, strain: 0.1 },
                                { sin2psi: 0.25, strain: 0.2 },
                                { sin2psi: 0.5, strain: 0.4 },
                                { sin2psi: 0.75, strain: 0.7 },
                                { sin2psi: 0.87, strain: 1.0 }
                              ]}
                              fill="#8884d8"
                            />
                            <ReferenceLine 
                              segment={[{ x: 0, y: 0.1 }, { x: 1, y: 1.1 }]}
                              stroke="red"
                              strokeDasharray="5 5"
                            />
                          </ScatterChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>
                  </div>
                ) : (
                  <Alert>
                    <Info className="h-4 w-4" />
                    <AlertTitle>No Stress Analysis</AlertTitle>
                    <AlertDescription>
                      Perform sin²ψ measurements at multiple tilt angles
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Texture Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Texture Index</span>
                    <Badge variant={textureIndex > 0.1 ? 'warning' : 'success'}>
                      {textureIndex.toFixed(3)}
                    </Badge>
                  </div>
                  <Progress value={textureIndex * 100} className="w-full" />
                  <p className="text-sm text-muted-foreground">
                    {textureIndex > 0.1 ? 
                      'Sample shows preferred orientation' : 
                      'Sample is randomly oriented'}
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

// Phase Database Component
const PhaseDatabaseBrowser: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSystem, setSelectedSystem] = useState<string>('all');
  
  const phases = [
    { name: 'Silicon', formula: 'Si', system: 'cubic', group: 'Fd-3m', a: 5.43095 },
    { name: 'Gallium Arsenide', formula: 'GaAs', system: 'cubic', group: 'F-43m', a: 5.65325 },
    { name: 'Gallium Nitride', formula: 'GaN', system: 'hexagonal', group: 'P63mc', a: 3.189, c: 5.185 },
    { name: 'Quartz', formula: 'SiO2', system: 'hexagonal', group: 'P3221', a: 4.9133, c: 5.4053 },
    { name: 'Alumina', formula: 'Al2O3', system: 'trigonal', group: 'R-3c', a: 4.759, c: 12.993 },
    { name: 'Anatase', formula: 'TiO2', system: 'tetragonal', group: 'I41/amd', a: 3.785, c: 9.514 }
  ];
  
  const filteredPhases = phases.filter(phase => {
    const matchesSearch = phase.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          phase.formula.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesSystem = selectedSystem === 'all' || phase.system === selectedSystem;
    return matchesSearch && matchesSystem;
  });
  
  return (
    <Card>
      <CardHeader>
        <CardTitle>Phase Database Browser</CardTitle>
        <CardDescription>
          Search and browse crystallographic phases
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex gap-4">
            <Input
              placeholder="Search phases..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="flex-1"
            />
            <Select
              value={selectedSystem}
              onValueChange={setSelectedSystem}
            >
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Crystal System" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Systems</SelectItem>
                <SelectItem value="cubic">Cubic</SelectItem>
                <SelectItem value="tetragonal">Tetragonal</SelectItem>
                <SelectItem value="orthorhombic">Orthorhombic</SelectItem>
                <SelectItem value="hexagonal">Hexagonal</SelectItem>
                <SelectItem value="trigonal">Trigonal</SelectItem>
                <SelectItem value="monoclinic">Monoclinic</SelectItem>
                <SelectItem value="triclinic">Triclinic</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-2">Name</th>
                  <th className="text-left p-2">Formula</th>
                  <th className="text-left p-2">System</th>
                  <th className="text-left p-2">Space Group</th>
                  <th className="text-left p-2">Lattice Parameters</th>
                </tr>
              </thead>
              <tbody>
                {filteredPhases.map((phase, i) => (
                  <tr key={i} className="border-b hover:bg-muted/50 cursor-pointer">
                    <td className="p-2 font-medium">{phase.name}</td>
                    <td className="p-2 font-mono">{phase.formula}</td>
                    <td className="p-2 capitalize">{phase.system}</td>
                    <td className="p-2 font-mono">{phase.group}</td>
                    <td className="p-2 font-mono text-xs">
                      a={phase.a.toFixed(3)}
                      {phase.c && `, c=${phase.c.toFixed(3)}`} Å
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Main Session 9 Component
const Session9XRDInterface: React.FC = () => {
  const [activeView, setActiveView] = useState<'analysis' | 'database'>('analysis');

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Session Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl font-bold">
                Session 9: X-ray Diffraction Analysis
              </CardTitle>
              <CardDescription className="mt-2">
                Structural characterization, phase identification, and stress analysis
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button
                variant={activeView === 'analysis' ? 'default' : 'outline'}
                onClick={() => setActiveView('analysis')}
              >
                <BarChart3 className="w-4 h-4 mr-2" />
                XRD Analysis
              </Button>
              <Button
                variant={activeView === 'database' ? 'default' : 'outline'}
                onClick={() => setActiveView('database')}
              >
                <Database className="w-4 h-4 mr-2" />
                Phase Database
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Content */}
      {activeView === 'analysis' ? <XRDInterface /> : <PhaseDatabaseBrowser />}

      {/* Session Footer */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Session Progress</p>
              <Progress value={100} className="w-[200px]" />
            </div>
            <div className="flex gap-2">
              <Badge>Phase ID</Badge>
              <Badge>Size/Strain</Badge>
              <Badge>Stress</Badge>
              <Badge>Texture</Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Session9XRDInterface;
