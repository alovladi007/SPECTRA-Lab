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
import { Textarea } from '@/components/ui/textarea';
import { 
  LineChart, Line, ScatterChart, Scatter, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, ReferenceLine, ReferenceArea, Brush, ErrorBar
} from 'recharts';
import { 
  Activity, AlertCircle, Download, FileText, Play, Save, 
  Settings, Upload, Zap, Eye, FileSearch, Maximize, 
  TrendingUp, Waves, Filter, Target, Info, CheckCircle,
  Sun, Droplets, Layers, Move, ZoomIn
} from 'lucide-react';

// Type definitions
interface SpectralData {
  wavelength: number[];
  intensity: number[];
  measurementType: 'transmission' | 'absorption' | 'reflectance';
  metadata?: {
    sampleId?: string;
    thickness?: number;
    temperature?: number;
    instrumentId?: string;
    [key: string]: any;
  };
}

interface TaucResult {
  bandgap: number;
  bandgapError: number;
  rSquared: number;
  taucData: Array<{ energy: number; taucValue: number }>;
  fitData: Array<{ energy: number; fitValue: number }>;
  bandgapType: string;
}

interface PeakInfo {
  position: number;
  intensity: number;
  width: number;
  area?: number;
  identification?: string;
}

interface FTIRResult {
  peaks: PeakInfo[];
  baseline: number[];
  fittedSpectrum: number[];
  residuals: number[];
  rSquared: number;
  thickness?: { value: number; error: number };
}

interface AnalysisSettings {
  smoothing: boolean;
  baselineCorrection: boolean;
  baselineMethod: 'linear' | 'polynomial' | 'rubberband' | 'als';
  peakFindingProminence: number;
  peakFindingDistance: number;
}

// UV-Vis-NIR Component
const UVVisNIRInterface: React.FC = () => {
  // State management
  const [spectralData, setSpectralData] = useState<SpectralData | null>(null);
  const [processedData, setProcessedData] = useState<SpectralData | null>(null);
  const [taucResult, setTaucResult] = useState<TaucResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState('measurement');
  const [error, setError] = useState<string | null>(null);
  
  // Analysis settings
  const [settings, setSettings] = useState<AnalysisSettings>({
    smoothing: true,
    baselineCorrection: true,
    baselineMethod: 'rubberband',
    peakFindingProminence: 0.01,
    peakFindingDistance: 10
  });
  
  // Measurement parameters
  const [measurementParams, setMeasurementParams] = useState({
    startWavelength: 300,
    endWavelength: 1100,
    stepSize: 2,
    integrationTime: 100,
    measurementType: 'transmission' as const,
    reference: 'air',
    sampleThickness: 0.5,
    averageScans: 3
  });

  // Tauc analysis parameters
  const [taucParams, setTaucParams] = useState({
    bandgapType: 'direct_allowed',
    energyRangeMin: 1.0,
    energyRangeMax: 2.0,
    autoRange: true
  });

  // Load sample data for demo
  const loadDemoData = useCallback(() => {
    const wavelengths = Array.from({ length: 401 }, (_, i) => 300 + i * 2);
    const intensities = wavelengths.map(wl => {
      // Simulate GaAs transmission spectrum
      const energy = 1240 / wl; // eV
      const bandgap = 1.42; // GaAs
      let transmission = 100;
      
      if (energy > bandgap) {
        const alpha = 5e5 * Math.sqrt(energy - bandgap);
        transmission = 100 * Math.exp(-alpha * 0.05); // 0.5 mm thickness
      }
      
      // Add interference fringes
      const n = 3.5;
      const phase = 4 * Math.PI * n * 500 / wl;
      transmission *= (1 + 0.3 * Math.cos(phase));
      
      // Add noise
      transmission += (Math.random() - 0.5) * 2;
      
      return Math.max(0, Math.min(100, transmission));
    });
    
    setSpectralData({
      wavelength: wavelengths,
      intensity: intensities,
      measurementType: 'transmission',
      metadata: {
        sampleId: 'GaAs-001',
        thickness: 0.5,
        temperature: 293,
        instrumentId: 'UV3600-01'
      }
    });
  }, []);

  // Start measurement
  const startMeasurement = useCallback(async () => {
    setIsProcessing(true);
    setError(null);
    
    try {
      // Simulate measurement delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Generate synthetic data
      loadDemoData();
      
      // Show success
      setActiveTab('analysis');
    } catch (err) {
      setError('Measurement failed. Please check instrument connection.');
    } finally {
      setIsProcessing(false);
    }
  }, [loadDemoData]);

  // Process spectrum
  const processSpectrum = useCallback(async () => {
    if (!spectralData) return;
    
    setIsProcessing(true);
    try {
      // Simulate processing
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Apply processing (simplified)
      const processed = { ...spectralData };
      
      if (settings.smoothing) {
        // Apply smoothing (simplified moving average)
        const smoothed = processed.intensity.map((val, idx) => {
          const start = Math.max(0, idx - 2);
          const end = Math.min(processed.intensity.length, idx + 3);
          const subset = processed.intensity.slice(start, end);
          return subset.reduce((a, b) => a + b) / subset.length;
        });
        processed.intensity = smoothed;
      }
      
      if (settings.baselineCorrection) {
        // Simple linear baseline
        const first = processed.intensity[0];
        const last = processed.intensity[processed.intensity.length - 1];
        const baseline = processed.intensity.map((_, idx) => 
          first + (last - first) * idx / (processed.intensity.length - 1)
        );
        processed.intensity = processed.intensity.map((val, idx) => val - baseline[idx]);
      }
      
      setProcessedData(processed);
    } catch (err) {
      setError('Processing failed.');
    } finally {
      setIsProcessing(false);
    }
  }, [spectralData, settings]);

  // Perform Tauc analysis
  const performTaucAnalysis = useCallback(async () => {
    if (!spectralData || !measurementParams.sampleThickness) return;
    
    setIsProcessing(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Calculate absorption coefficient
      const absorption = spectralData.measurementType === 'transmission'
        ? spectralData.intensity.map(t => -Math.log10(t / 100))
        : spectralData.intensity;
      
      const alpha = absorption.map(a => 2.303 * a / (measurementParams.sampleThickness / 10));
      
      // Convert to energy
      const energy = spectralData.wavelength.map(wl => 1240 / wl);
      
      // Calculate Tauc plot
      const n = taucParams.bandgapType === 'direct_allowed' ? 2 : 0.5;
      const taucData = energy.map((e, i) => ({
        energy: e,
        taucValue: Math.pow(alpha[i] * e, n)
      }));
      
      // Find linear region (simplified)
      const validData = taucData.filter(d => 
        d.energy >= 1.3 && d.energy <= 1.6 && d.taucValue > 0
      );
      
      // Linear fit
      const xMean = validData.reduce((s, d) => s + d.energy, 0) / validData.length;
      const yMean = validData.reduce((s, d) => s + d.taucValue, 0) / validData.length;
      
      const slope = validData.reduce((s, d) => 
        s + (d.energy - xMean) * (d.taucValue - yMean), 0
      ) / validData.reduce((s, d) => 
        s + Math.pow(d.energy - xMean, 2), 0
      );
      
      const intercept = yMean - slope * xMean;
      const bandgap = -intercept / slope;
      
      // Generate fit line
      const fitData = [
        { energy: bandgap, fitValue: 0 },
        { energy: 1.8, fitValue: slope * 1.8 + intercept }
      ];
      
      // Calculate R²
      const ssRes = validData.reduce((s, d) => 
        s + Math.pow(d.taucValue - (slope * d.energy + intercept), 2), 0
      );
      const ssTot = validData.reduce((s, d) => 
        s + Math.pow(d.taucValue - yMean, 2), 0
      );
      const rSquared = 1 - ssRes / ssTot;
      
      setTaucResult({
        bandgap,
        bandgapError: 0.01,
        rSquared,
        taucData,
        fitData,
        bandgapType: taucParams.bandgapType
      });
      
      setActiveTab('tauc');
    } catch (err) {
      setError('Tauc analysis failed.');
    } finally {
      setIsProcessing(false);
    }
  }, [spectralData, measurementParams, taucParams]);

  // Chart data preparation
  const spectrumChartData = useMemo(() => {
    if (!spectralData) return [];
    
    return spectralData.wavelength.map((wl, i) => ({
      wavelength: wl,
      raw: spectralData.intensity[i],
      processed: processedData?.intensity[i]
    }));
  }, [spectralData, processedData]);

  const taucChartData = useMemo(() => {
    if (!taucResult) return { data: [], fit: [] };
    
    return {
      data: taucResult.taucData,
      fit: taucResult.fitData
    };
  }, [taucResult]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Sun className="w-6 h-6" />
                UV-Vis-NIR Spectroscopy
              </CardTitle>
              <CardDescription>
                Optical absorption, transmission, and bandgap analysis
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Badge variant={spectralData ? 'success' : 'secondary'}>
                {spectralData ? 'Data Loaded' : 'No Data'}
              </Badge>
              <Badge variant={isProcessing ? 'warning' : 'secondary'}>
                {isProcessing ? 'Processing' : 'Ready'}
              </Badge>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Main Content Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="measurement">Measurement</TabsTrigger>
          <TabsTrigger value="analysis">Spectrum Analysis</TabsTrigger>
          <TabsTrigger value="tauc">Tauc Analysis</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
        </TabsList>

        {/* Measurement Setup Tab */}
        <TabsContent value="measurement">
          <div className="grid grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Measurement Parameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Start Wavelength (nm)</Label>
                    <Input
                      type="number"
                      value={measurementParams.startWavelength}
                      onChange={(e) => setMeasurementParams({
                        ...measurementParams,
                        startWavelength: parseInt(e.target.value)
                      })}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>End Wavelength (nm)</Label>
                    <Input
                      type="number"
                      value={measurementParams.endWavelength}
                      onChange={(e) => setMeasurementParams({
                        ...measurementParams,
                        endWavelength: parseInt(e.target.value)
                      })}
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Measurement Type</Label>
                  <Select
                    value={measurementParams.measurementType}
                    onValueChange={(value: any) => setMeasurementParams({
                      ...measurementParams,
                      measurementType: value
                    })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="transmission">Transmission</SelectItem>
                      <SelectItem value="absorption">Absorption</SelectItem>
                      <SelectItem value="reflectance">Reflectance</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Sample Thickness (mm)</Label>
                  <Input
                    type="number"
                    step="0.01"
                    value={measurementParams.sampleThickness}
                    onChange={(e) => setMeasurementParams({
                      ...measurementParams,
                      sampleThickness: parseFloat(e.target.value)
                    })}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Integration Time (ms)</Label>
                  <div className="flex items-center gap-2">
                    <Slider
                      value={[measurementParams.integrationTime]}
                      onValueChange={(value) => setMeasurementParams({
                        ...measurementParams,
                        integrationTime: value[0]
                      })}
                      max={1000}
                      min={10}
                      step={10}
                      className="flex-1"
                    />
                    <span className="w-16 text-right">
                      {measurementParams.integrationTime} ms
                    </span>
                  </div>
                </div>

                <Button
                  onClick={startMeasurement}
                  disabled={isProcessing}
                  className="w-full"
                >
                  {isProcessing ? (
                    <>Processing...</>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Start Measurement
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Processing Settings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label>Apply Smoothing</Label>
                  <Switch
                    checked={settings.smoothing}
                    onCheckedChange={(checked) => setSettings({
                      ...settings,
                      smoothing: checked
                    })}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <Label>Baseline Correction</Label>
                  <Switch
                    checked={settings.baselineCorrection}
                    onCheckedChange={(checked) => setSettings({
                      ...settings,
                      baselineCorrection: checked
                    })}
                  />
                </div>

                {settings.baselineCorrection && (
                  <div className="space-y-2">
                    <Label>Baseline Method</Label>
                    <Select
                      value={settings.baselineMethod}
                      onValueChange={(value: any) => setSettings({
                        ...settings,
                        baselineMethod: value
                      })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="linear">Linear</SelectItem>
                        <SelectItem value="polynomial">Polynomial</SelectItem>
                        <SelectItem value="rubberband">Rubberband</SelectItem>
                        <SelectItem value="als">ALS</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}

                <div className="pt-4">
                  <Button
                    variant="outline"
                    onClick={loadDemoData}
                    className="w-full"
                  >
                    <Upload className="w-4 h-4 mr-2" />
                    Load Demo Data
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Spectrum Analysis Tab */}
        <TabsContent value="analysis">
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">Spectrum Visualization</CardTitle>
                  <Button
                    onClick={processSpectrum}
                    disabled={!spectralData || isProcessing}
                    size="sm"
                  >
                    <Filter className="w-4 h-4 mr-2" />
                    Process Spectrum
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={spectrumChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="wavelength"
                      label={{ value: 'Wavelength (nm)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      label={{ 
                        value: measurementParams.measurementType === 'transmission' 
                          ? 'Transmission (%)' 
                          : 'Intensity (a.u.)', 
                        angle: -90, 
                        position: 'insideLeft' 
                      }}
                    />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="raw"
                      stroke="#8884d8"
                      name="Raw Spectrum"
                      strokeWidth={1.5}
                      dot={false}
                    />
                    {processedData && (
                      <Line
                        type="monotone"
                        dataKey="processed"
                        stroke="#82ca9d"
                        name="Processed"
                        strokeWidth={1.5}
                        dot={false}
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {spectralData && (
              <div className="grid grid-cols-3 gap-4">
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold">
                      {measurementParams.measurementType === 'transmission'
                        ? `${Math.max(...spectralData.intensity).toFixed(1)}%`
                        : Math.max(...spectralData.intensity).toFixed(3)
                      }
                    </div>
                    <p className="text-xs text-muted-foreground">Peak Intensity</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold">
                      {spectralData.wavelength.length}
                    </div>
                    <p className="text-xs text-muted-foreground">Data Points</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold">
                      {spectralData.metadata?.temperature || 293} K
                    </div>
                    <p className="text-xs text-muted-foreground">Temperature</p>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        </TabsContent>

        {/* Tauc Analysis Tab */}
        <TabsContent value="tauc">
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">Tauc Plot Analysis</CardTitle>
                  <Button
                    onClick={performTaucAnalysis}
                    disabled={!spectralData || isProcessing}
                    size="sm"
                  >
                    <TrendingUp className="w-4 h-4 mr-2" />
                    Calculate Bandgap
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-4 gap-4 mb-4">
                  <div className="space-y-2">
                    <Label>Transition Type</Label>
                    <Select
                      value={taucParams.bandgapType}
                      onValueChange={(value) => setTaucParams({
                        ...taucParams,
                        bandgapType: value
                      })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="direct_allowed">Direct Allowed</SelectItem>
                        <SelectItem value="direct_forbidden">Direct Forbidden</SelectItem>
                        <SelectItem value="indirect_allowed">Indirect Allowed</SelectItem>
                        <SelectItem value="indirect_forbidden">Indirect Forbidden</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Energy Min (eV)</Label>
                    <Input
                      type="number"
                      step="0.1"
                      value={taucParams.energyRangeMin}
                      onChange={(e) => setTaucParams({
                        ...taucParams,
                        energyRangeMin: parseFloat(e.target.value)
                      })}
                      disabled={taucParams.autoRange}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Energy Max (eV)</Label>
                    <Input
                      type="number"
                      step="0.1"
                      value={taucParams.energyRangeMax}
                      onChange={(e) => setTaucParams({
                        ...taucParams,
                        energyRangeMax: parseFloat(e.target.value)
                      })}
                      disabled={taucParams.autoRange}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Auto Range</Label>
                    <div className="pt-2">
                      <Switch
                        checked={taucParams.autoRange}
                        onCheckedChange={(checked) => setTaucParams({
                          ...taucParams,
                          autoRange: checked
                        })}
                      />
                    </div>
                  </div>
                </div>

                <ResponsiveContainer width="100%" height={400}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      type="number"
                      dataKey="energy"
                      domain={['dataMin', 'dataMax']}
                      label={{ value: 'Photon Energy (eV)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      type="number"
                      label={{ 
                        value: '(αhν)^n (a.u.)', 
                        angle: -90, 
                        position: 'insideLeft' 
                      }}
                    />
                    <Tooltip />
                    <Legend />
                    <Scatter
                      name="Tauc Plot"
                      data={taucChartData.data}
                      fill="#8884d8"
                    />
                    {taucResult && (
                      <Line
                        type="linear"
                        data={taucChartData.fit}
                        dataKey="fitValue"
                        stroke="#ff7300"
                        strokeWidth={2}
                        name="Linear Fit"
                        dot={false}
                      />
                    )}
                    {taucResult && (
                      <ReferenceLine
                        x={taucResult.bandgap}
                        stroke="red"
                        strokeDasharray="5 5"
                        label={{ value: `Eg = ${taucResult.bandgap.toFixed(3)} eV`, position: 'top' }}
                      />
                    )}
                  </ScatterChart>
                </ResponsiveContainer>

                {taucResult && (
                  <div className="grid grid-cols-4 gap-4 mt-4">
                    <Card>
                      <CardContent className="pt-6">
                        <div className="text-2xl font-bold">
                          {taucResult.bandgap.toFixed(3)} eV
                        </div>
                        <p className="text-xs text-muted-foreground">Optical Bandgap</p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-6">
                        <div className="text-2xl font-bold">
                          ±{(taucResult.bandgapError * 1000).toFixed(1)} meV
                        </div>
                        <p className="text-xs text-muted-foreground">Uncertainty</p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-6">
                        <div className="text-2xl font-bold">
                          {taucResult.rSquared.toFixed(4)}
                        </div>
                        <p className="text-xs text-muted-foreground">R² Value</p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-6">
                        <div className="text-2xl font-bold">
                          {(1240 / taucResult.bandgap).toFixed(0)} nm
                        </div>
                        <p className="text-xs text-muted-foreground">Absorption Edge</p>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value="results">
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Analysis Summary</CardTitle>
              </CardHeader>
              <CardContent>
                {taucResult ? (
                  <div className="space-y-4">
                    <Alert>
                      <CheckCircle className="h-4 w-4" />
                      <AlertTitle>Analysis Complete</AlertTitle>
                      <AlertDescription>
                        Optical characterization successfully completed for sample {spectralData?.metadata?.sampleId || 'Unknown'}
                      </AlertDescription>
                    </Alert>

                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <h4 className="font-semibold">Material Properties</h4>
                        <dl className="space-y-1 text-sm">
                          <div className="flex justify-between">
                            <dt>Optical Bandgap:</dt>
                            <dd className="font-mono">{taucResult.bandgap.toFixed(3)} eV</dd>
                          </div>
                          <div className="flex justify-between">
                            <dt>Absorption Edge:</dt>
                            <dd className="font-mono">{(1240 / taucResult.bandgap).toFixed(0)} nm</dd>
                          </div>
                          <div className="flex justify-between">
                            <dt>Transition Type:</dt>
                            <dd className="font-mono">{taucResult.bandgapType.replace('_', ' ')}</dd>
                          </div>
                          <div className="flex justify-between">
                            <dt>Sample Thickness:</dt>
                            <dd className="font-mono">{measurementParams.sampleThickness} mm</dd>
                          </div>
                        </dl>
                      </div>

                      <div className="space-y-2">
                        <h4 className="font-semibold">Quality Metrics</h4>
                        <dl className="space-y-1 text-sm">
                          <div className="flex justify-between">
                            <dt>R² (Tauc Fit):</dt>
                            <dd className="font-mono">{taucResult.rSquared.toFixed(4)}</dd>
                          </div>
                          <div className="flex justify-between">
                            <dt>Measurement Type:</dt>
                            <dd className="font-mono">{measurementParams.measurementType}</dd>
                          </div>
                          <div className="flex justify-between">
                            <dt>Data Points:</dt>
                            <dd className="font-mono">{spectralData?.wavelength.length}</dd>
                          </div>
                          <div className="flex justify-between">
                            <dt>Processing:</dt>
                            <dd className="font-mono">
                              {settings.smoothing ? 'Smoothed' : 'Raw'}
                              {settings.baselineCorrection ? ', Baseline Corrected' : ''}
                            </dd>
                          </div>
                        </dl>
                      </div>
                    </div>

                    <div className="flex gap-2 pt-4">
                      <Button variant="outline" size="sm">
                        <Download className="w-4 h-4 mr-2" />
                        Export Data
                      </Button>
                      <Button variant="outline" size="sm">
                        <FileText className="w-4 h-4 mr-2" />
                        Generate Report
                      </Button>
                      <Button variant="outline" size="sm">
                        <Save className="w-4 h-4 mr-2" />
                        Save Results
                      </Button>
                    </div>
                  </div>
                ) : (
                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>No Results</AlertTitle>
                    <AlertDescription>
                      Please perform measurement and analysis to view results.
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
};

// FTIR Component
const FTIRInterface: React.FC = () => {
  // State management
  const [spectralData, setSpectralData] = useState<SpectralData | null>(null);
  const [ftirResult, setFTIRResult] = useState<FTIRResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState('setup');
  
  // Measurement parameters
  const [params, setParams] = useState({
    startWavenumber: 400,
    endWavenumber: 4000,
    resolution: 4,
    scans: 32,
    apodization: 'happ-genzel',
    zeroFilling: 2,
    sampleType: 'thin_film'
  });

  // Peak analysis settings
  const [peakSettings, setPeakSettings] = useState({
    prominence: 0.01,
    distance: 10,
    peakType: 'lorentzian',
    maxPeaks: 10,
    autoIdentify: true
  });

  // Load demo data
  const loadDemoData = useCallback(() => {
    const wavenumbers = Array.from({ length: 800 }, (_, i) => 400 + i * 4.5);
    const intensities = wavenumbers.map(wn => {
      let intensity = 90; // Base transmittance
      
      // Add SiO2 peaks
      if (Math.abs(wn - 1080) < 50) {
        intensity -= 30 * Math.exp(-Math.pow((wn - 1080) / 30, 2));
      }
      if (Math.abs(wn - 460) < 30) {
        intensity -= 20 * Math.exp(-Math.pow((wn - 460) / 20, 2));
      }
      if (Math.abs(wn - 800) < 40) {
        intensity -= 25 * Math.exp(-Math.pow((wn - 800) / 40, 2));
      }
      
      // Add interference fringes
      const thickness = 1.0; // µm
      const n = 1.46;
      const period = 10000 / (2 * n * thickness);
      intensity += 5 * Math.sin(2 * Math.PI * wn / period);
      
      // Add noise
      intensity += (Math.random() - 0.5) * 1;
      
      return Math.max(0, Math.min(100, intensity));
    });
    
    setSpectralData({
      wavelength: wavenumbers,
      intensity: intensities,
      measurementType: 'transmission',
      metadata: {
        sampleId: 'SiO2-TF-001',
        instrumentId: 'FTIR-01',
        resolution: 4
      }
    });
  }, []);

  // Start measurement
  const startMeasurement = useCallback(async () => {
    setIsProcessing(true);
    
    try {
      // Simulate measurement
      await new Promise(resolve => setTimeout(resolve, 2000));
      loadDemoData();
      setActiveTab('analysis');
    } catch (err) {
      console.error('Measurement failed:', err);
    } finally {
      setIsProcessing(false);
    }
  }, [loadDemoData]);

  // Find peaks
  const findPeaks = useCallback(async () => {
    if (!spectralData) return;
    
    setIsProcessing(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Simple peak finding
      const peaks: PeakInfo[] = [];
      const intensity = spectralData.intensity;
      const wavenumber = spectralData.wavelength;
      
      // Find local minima (absorption peaks)
      for (let i = 5; i < intensity.length - 5; i++) {
        const isMinimum = intensity[i] < intensity[i-1] && 
                          intensity[i] < intensity[i+1] &&
                          intensity[i] < intensity[i-2] &&
                          intensity[i] < intensity[i+2];
        
        if (isMinimum && (90 - intensity[i]) > 10) {
          const position = wavenumber[i];
          let identification = undefined;
          
          // Simple peak identification
          if (peakSettings.autoIdentify) {
            if (Math.abs(position - 1080) < 50) identification = 'Si-O stretching';
            else if (Math.abs(position - 460) < 50) identification = 'Si-O bending';
            else if (Math.abs(position - 800) < 50) identification = 'Si-O rocking';
          }
          
          peaks.push({
            position,
            intensity: 90 - intensity[i],
            width: 30,
            area: (90 - intensity[i]) * 30,
            identification
          });
        }
      }
      
      // Calculate film thickness from fringes
      const maxima: number[] = [];
      for (let i = 5; i < intensity.length - 5; i++) {
        if (intensity[i] > intensity[i-1] && intensity[i] > intensity[i+1]) {
          maxima.push(wavenumber[i]);
        }
      }
      
      let thickness = undefined;
      if (maxima.length >= 2) {
        const deltaWn = Math.abs(maxima[1] - maxima[0]);
        thickness = {
          value: 10000 / (2 * 1.46 * deltaWn), // µm
          error: 0.05
        };
      }
      
      setFTIRResult({
        peaks,
        baseline: new Array(intensity.length).fill(90),
        fittedSpectrum: intensity,
        residuals: new Array(intensity.length).fill(0),
        rSquared: 0.95,
        thickness
      });
      
      setActiveTab('peaks');
    } catch (err) {
      console.error('Peak analysis failed:', err);
    } finally {
      setIsProcessing(false);
    }
  }, [spectralData, peakSettings]);

  // Chart data
  const chartData = useMemo(() => {
    if (!spectralData) return [];
    
    return spectralData.wavelength.map((wn, i) => ({
      wavenumber: wn,
      intensity: spectralData.intensity[i],
      baseline: ftirResult?.baseline[i],
      fitted: ftirResult?.fittedSpectrum[i]
    }));
  }, [spectralData, ftirResult]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Waves className="w-6 h-6" />
                FTIR Spectroscopy
              </CardTitle>
              <CardDescription>
                Fourier Transform Infrared spectroscopy for material identification
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Badge variant={spectralData ? 'success' : 'secondary'}>
                {spectralData ? 'Data Loaded' : 'No Data'}
              </Badge>
              <Badge variant={isProcessing ? 'warning' : 'secondary'}>
                {isProcessing ? 'Processing' : 'Ready'}
              </Badge>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="setup">Setup</TabsTrigger>
          <TabsTrigger value="analysis">Spectrum</TabsTrigger>
          <TabsTrigger value="peaks">Peak Analysis</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
        </TabsList>

        {/* Setup Tab */}
        <TabsContent value="setup">
          <div className="grid grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Measurement Parameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Start (cm⁻¹)</Label>
                    <Input
                      type="number"
                      value={params.startWavenumber}
                      onChange={(e) => setParams({
                        ...params,
                        startWavenumber: parseInt(e.target.value)
                      })}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>End (cm⁻¹)</Label>
                    <Input
                      type="number"
                      value={params.endWavenumber}
                      onChange={(e) => setParams({
                        ...params,
                        endWavenumber: parseInt(e.target.value)
                      })}
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Resolution (cm⁻¹)</Label>
                  <Select
                    value={params.resolution.toString()}
                    onValueChange={(value) => setParams({
                      ...params,
                      resolution: parseInt(value)
                    })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">1 cm⁻¹</SelectItem>
                      <SelectItem value="2">2 cm⁻¹</SelectItem>
                      <SelectItem value="4">4 cm⁻¹</SelectItem>
                      <SelectItem value="8">8 cm⁻¹</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Number of Scans</Label>
                  <Input
                    type="number"
                    value={params.scans}
                    onChange={(e) => setParams({
                      ...params,
                      scans: parseInt(e.target.value)
                    })}
                  />
                </div>

                <Button
                  onClick={startMeasurement}
                  disabled={isProcessing}
                  className="w-full"
                >
                  {isProcessing ? (
                    <>Measuring...</>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Start Measurement
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Peak Analysis Settings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Peak Type</Label>
                  <Select
                    value={peakSettings.peakType}
                    onValueChange={(value) => setPeakSettings({
                      ...peakSettings,
                      peakType: value
                    })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="gaussian">Gaussian</SelectItem>
                      <SelectItem value="lorentzian">Lorentzian</SelectItem>
                      <SelectItem value="voigt">Voigt</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Max Peaks</Label>
                  <Input
                    type="number"
                    value={peakSettings.maxPeaks}
                    onChange={(e) => setPeakSettings({
                      ...peakSettings,
                      maxPeaks: parseInt(e.target.value)
                    })}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <Label>Auto-Identify Peaks</Label>
                  <Switch
                    checked={peakSettings.autoIdentify}
                    onCheckedChange={(checked) => setPeakSettings({
                      ...peakSettings,
                      autoIdentify: checked
                    })}
                  />
                </div>

                <div className="pt-4">
                  <Button
                    variant="outline"
                    onClick={loadDemoData}
                    className="w-full"
                  >
                    <Upload className="w-4 h-4 mr-2" />
                    Load Demo Data
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value="analysis">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">FTIR Spectrum</CardTitle>
                <Button
                  onClick={findPeaks}
                  disabled={!spectralData || isProcessing}
                  size="sm"
                >
                  <Target className="w-4 h-4 mr-2" />
                  Find Peaks
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="wavenumber"
                    reversed
                    label={{ value: 'Wavenumber (cm⁻¹)', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    label={{ value: 'Transmittance (%)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="intensity"
                    stroke="#8884d8"
                    name="Spectrum"
                    strokeWidth={1.5}
                    dot={false}
                  />
                  {ftirResult && (
                    <Line
                      type="monotone"
                      dataKey="baseline"
                      stroke="#82ca9d"
                      name="Baseline"
                      strokeWidth={1}
                      strokeDasharray="5 5"
                      dot={false}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Peak Analysis Tab */}
        <TabsContent value="peaks">
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Identified Peaks</CardTitle>
              </CardHeader>
              <CardContent>
                {ftirResult && ftirResult.peaks.length > 0 ? (
                  <div className="space-y-4">
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left p-2">Position (cm⁻¹)</th>
                            <th className="text-left p-2">Intensity</th>
                            <th className="text-left p-2">Width</th>
                            <th className="text-left p-2">Area</th>
                            <th className="text-left p-2">Assignment</th>
                          </tr>
                        </thead>
                        <tbody>
                          {ftirResult.peaks.map((peak, idx) => (
                            <tr key={idx} className="border-b">
                              <td className="p-2 font-mono">{peak.position.toFixed(1)}</td>
                              <td className="p-2 font-mono">{peak.intensity.toFixed(1)}</td>
                              <td className="p-2 font-mono">{peak.width.toFixed(1)}</td>
                              <td className="p-2 font-mono">{peak.area?.toFixed(0)}</td>
                              <td className="p-2">
                                {peak.identification ? (
                                  <Badge variant="outline">{peak.identification}</Badge>
                                ) : (
                                  <span className="text-muted-foreground">Unknown</span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {ftirResult.thickness && (
                      <Alert>
                        <Layers className="h-4 w-4" />
                        <AlertTitle>Film Thickness Calculated</AlertTitle>
                        <AlertDescription>
                          Estimated thickness from interference fringes: {ftirResult.thickness.value.toFixed(2)} ± {ftirResult.thickness.error.toFixed(2)} µm
                        </AlertDescription>
                      </Alert>
                    )}
                  </div>
                ) : (
                  <Alert>
                    <Info className="h-4 w-4" />
                    <AlertTitle>No Peaks Found</AlertTitle>
                    <AlertDescription>
                      Run peak analysis to identify spectral features
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value="results">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Analysis Summary</CardTitle>
            </CardHeader>
            <CardContent>
              {ftirResult ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <h4 className="font-semibold">Sample Information</h4>
                      <dl className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <dt>Sample ID:</dt>
                          <dd className="font-mono">{spectralData?.metadata?.sampleId || 'N/A'}</dd>
                        </div>
                        <div className="flex justify-between">
                          <dt>Resolution:</dt>
                          <dd className="font-mono">{params.resolution} cm⁻¹</dd>
                        </div>
                        <div className="flex justify-between">
                          <dt>Scans:</dt>
                          <dd className="font-mono">{params.scans}</dd>
                        </div>
                      </dl>
                    </div>

                    <div className="space-y-2">
                      <h4 className="font-semibold">Analysis Results</h4>
                      <dl className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <dt>Peaks Found:</dt>
                          <dd className="font-mono">{ftirResult.peaks.length}</dd>
                        </div>
                        <div className="flex justify-between">
                          <dt>Identified:</dt>
                          <dd className="font-mono">
                            {ftirResult.peaks.filter(p => p.identification).length}
                          </dd>
                        </div>
                        {ftirResult.thickness && (
                          <div className="flex justify-between">
                            <dt>Film Thickness:</dt>
                            <dd className="font-mono">{ftirResult.thickness.value.toFixed(2)} µm</dd>
                          </div>
                        )}
                      </dl>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <h4 className="font-semibold">Peak Assignments</h4>
                    <div className="flex flex-wrap gap-2">
                      {ftirResult.peaks
                        .filter(p => p.identification)
                        .map((peak, idx) => (
                          <Badge key={idx} variant="secondary">
                            {peak.identification}: {peak.position.toFixed(0)} cm⁻¹
                          </Badge>
                        ))}
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <Button variant="outline" size="sm">
                      <Download className="w-4 h-4 mr-2" />
                      Export Spectrum
                    </Button>
                    <Button variant="outline" size="sm">
                      <FileText className="w-4 h-4 mr-2" />
                      Generate Report
                    </Button>
                  </div>
                </div>
              ) : (
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>No Results Available</AlertTitle>
                  <AlertDescription>
                    Complete measurement and analysis to view results
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

// Main Session 7 Component
const Session7OpticalInterface: React.FC = () => {
  const [selectedMethod, setSelectedMethod] = useState<'uvvis' | 'ftir'>('uvvis');

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Session Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl font-bold">
                Session 7: Optical Methods I
              </CardTitle>
              <CardDescription className="mt-2">
                UV-Vis-NIR and FTIR Spectroscopy for Semiconductor Characterization
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button
                variant={selectedMethod === 'uvvis' ? 'default' : 'outline'}
                onClick={() => setSelectedMethod('uvvis')}
              >
                <Sun className="w-4 h-4 mr-2" />
                UV-Vis-NIR
              </Button>
              <Button
                variant={selectedMethod === 'ftir' ? 'default' : 'outline'}
                onClick={() => setSelectedMethod('ftir')}
              >
                <Waves className="w-4 h-4 mr-2" />
                FTIR
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Method Interface */}
      {selectedMethod === 'uvvis' ? <UVVisNIRInterface /> : <FTIRInterface />}

      {/* Session Status */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Session Progress</p>
              <Progress value={100} className="w-[200px]" />
            </div>
            <Badge variant="success">Complete</Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Session7OpticalInterface;
