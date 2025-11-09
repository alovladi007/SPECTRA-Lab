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
import { 
  LineChart, Line, ScatterChart, Scatter, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, ReferenceLine, ReferenceArea, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Radar, RadarChart, Brush
} from 'recharts';
import {
  Activity, AlertCircle, Download, FileText, Play, Save, Settings,
  Upload, Zap, Eye, Layers, TrendingUp, Thermometer, Gauge,
  Sparkles, Atom, Maximize, Target, Info, CheckCircle, Beaker,
  ScanLine, Lightbulb, Microscope, Prism, FlaskConical
} from 'lucide-react';

// Type definitions
interface LayerData {
  thickness: number;
  refractiveIndex: number;
  extinctionCoefficient: number;
  model: 'cauchy' | 'sellmeier' | 'tauc-lorentz';
}

interface EllipsometryData {
  wavelength: number[];
  psi: number[];
  delta: number[];
  angleOfIncidence: number;
  fitPsi?: number[];
  fitDelta?: number[];
}

interface PLData {
  wavelength: number[];
  intensity: number[];
  temperature: number;
  excitationWavelength: number;
  excitationPower: number;
}

interface RamanData {
  ramanShift: number[];
  intensity: number[];
  laserWavelength: number;
  laserPower: number;
}

// Ellipsometry Component
const EllipsometryInterface: React.FC = () => {
  // State management
  const [ellipsometryData, setEllipsometryData] = useState<EllipsometryData | null>(null);
  const [layers, setLayers] = useState<LayerData[]>([
    { thickness: 100, refractiveIndex: 1.46, extinctionCoefficient: 0, model: 'cauchy' }
  ]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState('setup');
  const [fitResults, setFitResults] = useState<any>(null);
  
  // Measurement parameters
  const [measurementParams, setMeasurementParams] = useState({
    startWavelength: 300,
    endWavelength: 800,
    angleOfIncidence: 70,
    numberOfAngles: 1,
    polarizer: 'rotating',
    compensator: 'none'
  });

  // Load demo data
  const loadDemoData = useCallback(() => {
    const wavelengths = Array.from({ length: 200 }, (_, i) => 300 + i * 2.5);
    
    // Generate synthetic Psi and Delta
    const psi = wavelengths.map(wl => {
      const oscillation = Math.sin(2 * Math.PI * wl / 100) * 5;
      return 45 + oscillation + (Math.random() - 0.5) * 0.5;
    });
    
    const delta = wavelengths.map(wl => {
      const phase = 180 - (wl - 300) * 0.3;
      return phase + Math.sin(2 * Math.PI * wl / 50) * 10 + (Math.random() - 0.5);
    });
    
    setEllipsometryData({
      wavelength: wavelengths,
      psi,
      delta,
      angleOfIncidence: measurementParams.angleOfIncidence
    });
  }, [measurementParams.angleOfIncidence]);

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

  // Fit model
  const fitModel = useCallback(async () => {
    if (!ellipsometryData) return;
    
    setIsProcessing(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Generate fitted curves (simplified)
      const fitPsi = ellipsometryData.psi.map((p, i) => {
        const smoothed = p - (Math.random() - 0.5) * 0.2;
        return smoothed;
      });
      
      const fitDelta = ellipsometryData.delta.map((d, i) => {
        const smoothed = d - (Math.random() - 0.5) * 0.3;
        return smoothed;
      });
      
      setEllipsometryData({
        ...ellipsometryData,
        fitPsi,
        fitDelta
      });
      
      // Mock fit results
      setFitResults({
        layers: layers.map((layer, i) => ({
          ...layer,
          thickness: layer.thickness + (Math.random() - 0.5) * 5,
          mse: Math.random() * 5
        })),
        totalMSE: 2.34,
        rSquared: 0.995
      });
      
      setActiveTab('results');
    } catch (err) {
      console.error('Fitting failed:', err);
    } finally {
      setIsProcessing(false);
    }
  }, [ellipsometryData, layers]);

  // Add layer
  const addLayer = () => {
    setLayers([
      ...layers,
      { thickness: 50, refractiveIndex: 1.5, extinctionCoefficient: 0, model: 'cauchy' }
    ]);
  };

  // Remove layer
  const removeLayer = (index: number) => {
    setLayers(layers.filter((_, i) => i !== index));
  };

  // Update layer
  const updateLayer = (index: number, field: keyof LayerData, value: any) => {
    const newLayers = [...layers];
    newLayers[index] = { ...newLayers[index], [field]: value };
    setLayers(newLayers);
  };

  // Chart data
  const psiDeltaChartData = useMemo(() => {
    if (!ellipsometryData) return [];
    
    return ellipsometryData.wavelength.map((wl, i) => ({
      wavelength: wl,
      psi: ellipsometryData.psi[i],
      delta: ellipsometryData.delta[i],
      fitPsi: ellipsometryData.fitPsi?.[i],
      fitDelta: ellipsometryData.fitDelta?.[i]
    }));
  }, [ellipsometryData]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Layers className="w-6 h-6" />
                Spectroscopic Ellipsometry
              </CardTitle>
              <CardDescription>
                Multi-layer thin film characterization
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Badge variant={ellipsometryData ? 'success' : 'secondary'}>
                {ellipsometryData ? 'Data Loaded' : 'No Data'}
              </Badge>
              {fitResults && (
                <Badge variant="success">Model Fitted</Badge>
              )}
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="setup">Setup</TabsTrigger>
          <TabsTrigger value="model">Layer Model</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
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
                    <Label>Start λ (nm)</Label>
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
                    <Label>End λ (nm)</Label>
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
                  <Label>Angle of Incidence (°)</Label>
                  <div className="flex items-center gap-2">
                    <Slider
                      value={[measurementParams.angleOfIncidence]}
                      onValueChange={(value) => setMeasurementParams({
                        ...measurementParams,
                        angleOfIncidence: value[0]
                      })}
                      min={20}
                      max={85}
                      step={1}
                      className="flex-1"
                    />
                    <span className="w-12 text-right">
                      {measurementParams.angleOfIncidence}°
                    </span>
                  </div>
                </div>

                <Button
                  onClick={startMeasurement}
                  disabled={isProcessing}
                  className="w-full"
                >
                  {isProcessing ? 'Measuring...' : 'Start Measurement'}
                </Button>

                <Button
                  variant="outline"
                  onClick={loadDemoData}
                  className="w-full"
                >
                  Load Demo Data
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Quick Info</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold mb-2">Ψ (Psi)</h4>
                    <p className="text-sm text-muted-foreground">
                      Amplitude ratio between p- and s-polarized light
                    </p>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Δ (Delta)</h4>
                    <p className="text-sm text-muted-foreground">
                      Phase difference between p- and s-polarized light
                    </p>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Applications</h4>
                    <ul className="text-sm text-muted-foreground list-disc list-inside">
                      <li>Film thickness (0.1 nm - 10 μm)</li>
                      <li>Optical constants (n, k)</li>
                      <li>Multi-layer analysis</li>
                      <li>Surface roughness</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Layer Model Tab */}
        <TabsContent value="model">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">Layer Stack Model</CardTitle>
                <Button onClick={addLayer} size="sm">
                  <Layers className="w-4 h-4 mr-2" />
                  Add Layer
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="text-sm text-muted-foreground">
                  Ambient (n = 1.0)
                </div>
                
                {layers.map((layer, index) => (
                  <Card key={index}>
                    <CardContent className="pt-4">
                      <div className="flex items-center justify-between mb-4">
                        <h4 className="font-semibold">Layer {index + 1}</h4>
                        {layers.length > 1 && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => removeLayer(index)}
                          >
                            Remove
                          </Button>
                        )}
                      </div>
                      
                      <div className="grid grid-cols-4 gap-4">
                        <div className="space-y-2">
                          <Label>Thickness (nm)</Label>
                          <Input
                            type="number"
                            value={layer.thickness}
                            onChange={(e) => updateLayer(index, 'thickness', parseFloat(e.target.value))}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>n (@ 633 nm)</Label>
                          <Input
                            type="number"
                            step="0.01"
                            value={layer.refractiveIndex}
                            onChange={(e) => updateLayer(index, 'refractiveIndex', parseFloat(e.target.value))}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>k (@ 633 nm)</Label>
                          <Input
                            type="number"
                            step="0.001"
                            value={layer.extinctionCoefficient}
                            onChange={(e) => updateLayer(index, 'extinctionCoefficient', parseFloat(e.target.value))}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>Model</Label>
                          <Select
                            value={layer.model}
                            onValueChange={(value: any) => updateLayer(index, 'model', value)}
                          >
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="cauchy">Cauchy</SelectItem>
                              <SelectItem value="sellmeier">Sellmeier</SelectItem>
                              <SelectItem value="tauc-lorentz">Tauc-Lorentz</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
                
                <div className="text-sm text-muted-foreground">
                  Substrate (Si: n = 3.85, k = 0.02)
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value="analysis">
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">Ψ and Δ Spectra</CardTitle>
                  <Button
                    onClick={fitModel}
                    disabled={!ellipsometryData || isProcessing}
                    size="sm"
                  >
                    <Target className="w-4 h-4 mr-2" />
                    Fit Model
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={psiDeltaChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="wavelength"
                      label={{ value: 'Wavelength (nm)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      yAxisId="psi"
                      label={{ value: 'Ψ (degrees)', angle: -90, position: 'insideLeft' }}
                    />
                    <YAxis 
                      yAxisId="delta"
                      orientation="right"
                      label={{ value: 'Δ (degrees)', angle: 90, position: 'insideRight' }}
                    />
                    <Tooltip />
                    <Legend />
                    <Line
                      yAxisId="psi"
                      type="monotone"
                      dataKey="psi"
                      stroke="#8884d8"
                      name="Ψ (measured)"
                      strokeWidth={1.5}
                      dot={false}
                    />
                    <Line
                      yAxisId="delta"
                      type="monotone"
                      dataKey="delta"
                      stroke="#82ca9d"
                      name="Δ (measured)"
                      strokeWidth={1.5}
                      dot={false}
                    />
                    {ellipsometryData?.fitPsi && (
                      <>
                        <Line
                          yAxisId="psi"
                          type="monotone"
                          dataKey="fitPsi"
                          stroke="#ff7300"
                          name="Ψ (fit)"
                          strokeWidth={2}
                          strokeDasharray="5 5"
                          dot={false}
                        />
                        <Line
                          yAxisId="delta"
                          type="monotone"
                          dataKey="fitDelta"
                          stroke="#e91e63"
                          name="Δ (fit)"
                          strokeWidth={2}
                          strokeDasharray="5 5"
                          dot={false}
                        />
                      </>
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {ellipsometryData && (
              <div className="grid grid-cols-3 gap-4">
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold">
                      {measurementParams.angleOfIncidence}°
                    </div>
                    <p className="text-xs text-muted-foreground">Angle of Incidence</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold">
                      {ellipsometryData.wavelength.length}
                    </div>
                    <p className="text-xs text-muted-foreground">Data Points</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold">
                      {layers.reduce((sum, l) => sum + l.thickness, 0).toFixed(0)} nm
                    </div>
                    <p className="text-xs text-muted-foreground">Total Thickness</p>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value="results">
          <div className="space-y-4">
            {fitResults ? (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Fit Results</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <h4 className="font-semibold mb-2">Goodness of Fit</h4>
                          <dl className="space-y-1 text-sm">
                            <div className="flex justify-between">
                              <dt>MSE:</dt>
                              <dd className="font-mono">{fitResults.totalMSE.toFixed(3)}</dd>
                            </div>
                            <div className="flex justify-between">
                              <dt>R²:</dt>
                              <dd className="font-mono">{fitResults.rSquared.toFixed(4)}</dd>
                            </div>
                          </dl>
                        </div>
                        
                        <div>
                          <h4 className="font-semibold mb-2">Layer Parameters</h4>
                          {fitResults.layers.map((layer: any, i: number) => (
                            <div key={i} className="text-sm mb-2">
                              <div className="font-medium">Layer {i + 1}:</div>
                              <div className="ml-4">
                                Thickness: {layer.thickness.toFixed(1)} nm
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      <Alert>
                        <CheckCircle className="h-4 w-4" />
                        <AlertTitle>Model Converged</AlertTitle>
                        <AlertDescription>
                          The optical model successfully fitted the experimental data
                        </AlertDescription>
                      </Alert>

                      <div className="flex gap-2">
                        <Button variant="outline" size="sm">
                          <Download className="w-4 h-4 mr-2" />
                          Export Results
                        </Button>
                        <Button variant="outline" size="sm">
                          <FileText className="w-4 h-4 mr-2" />
                          Generate Report
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </>
            ) : (
              <Alert>
                <Info className="h-4 w-4" />
                <AlertTitle>No Results</AlertTitle>
                <AlertDescription>
                  Perform model fitting to see results
                </AlertDescription>
              </Alert>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

// Photoluminescence Component
const PhotoluminescenceInterface: React.FC = () => {
  // State management
  const [plData, setPlData] = useState<PLData | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState('measurement');
  const [temperatureSeries, setTemperatureSeries] = useState<PLData[]>([]);
  
  // Measurement parameters
  const [params, setParams] = useState({
    excitationWavelength: 532,
    excitationPower: 10,
    temperature: 10,
    integrationTime: 1,
    gratingDensity: 1200,
    slitWidth: 100
  });

  // Load demo data
  const loadDemoData = useCallback(() => {
    const wavelengths = Array.from({ length: 500 }, (_, i) => 800 + i * 0.4);
    
    // Generate GaAs-like PL spectrum
    const peakWavelength = 870;
    const intensity = wavelengths.map(wl => {
      // Main peak
      const main = 1000 * Math.exp(-0.5 * Math.pow((wl - peakWavelength) / 15, 2));
      // Phonon replica
      const phonon = 300 * Math.exp(-0.5 * Math.pow((wl - (peakWavelength + 20)) / 15, 2));
      // Defect emission
      const defect = 100 * Math.exp(-0.5 * Math.pow((wl - (peakWavelength + 60)) / 25, 2));
      
      return main + phonon + defect + Math.random() * 10;
    });
    
    setPlData({
      wavelength: wavelengths,
      intensity,
      temperature: params.temperature,
      excitationWavelength: params.excitationWavelength,
      excitationPower: params.excitationPower
    });
  }, [params]);

  // Start measurement
  const startMeasurement = useCallback(async () => {
    setIsProcessing(true);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 2000));
      loadDemoData();
      setActiveTab('analysis');
    } catch (err) {
      console.error('Measurement failed:', err);
    } finally {
      setIsProcessing(false);
    }
  }, [loadDemoData]);

  // Temperature series measurement
  const measureTemperatureSeries = useCallback(async () => {
    setIsProcessing(true);
    
    try {
      const temperatures = [10, 50, 100, 150, 200, 250, 293];
      const series: PLData[] = [];
      
      for (const temp of temperatures) {
        const wavelengths = Array.from({ length: 500 }, (_, i) => 800 + i * 0.4);
        
        // Simulate temperature effects
        const peakShift = temp * 0.05;  // Red shift with temperature
        const broadening = 1 + temp / 100;  // Broadening with temperature
        
        const intensity = wavelengths.map(wl => {
          const peakWavelength = 870 + peakShift;
          const width = 15 * broadening;
          const amplitude = 1000 * Math.exp(-temp / 200);  // Quenching
          
          return amplitude * Math.exp(-0.5 * Math.pow((wl - peakWavelength) / width, 2)) + 
                 Math.random() * 10;
        });
        
        series.push({
          wavelength: wavelengths,
          intensity,
          temperature: temp,
          excitationWavelength: params.excitationWavelength,
          excitationPower: params.excitationPower
        });
        
        // Simulate measurement time
        await new Promise(resolve => setTimeout(resolve, 300));
      }
      
      setTemperatureSeries(series);
      setActiveTab('temperature');
    } catch (err) {
      console.error('Temperature series failed:', err);
    } finally {
      setIsProcessing(false);
    }
  }, [params]);

  // Chart data
  const spectrumChartData = useMemo(() => {
    if (!plData) return [];
    
    return plData.wavelength.map((wl, i) => ({
      wavelength: wl,
      energy: 1240 / wl,
      intensity: plData.intensity[i]
    }));
  }, [plData]);

  const temperatureChartData = useMemo(() => {
    if (temperatureSeries.length === 0) return [];
    
    return temperatureSeries.map(data => {
      const maxIdx = data.intensity.indexOf(Math.max(...data.intensity));
      return {
        temperature: data.temperature,
        peakIntensity: Math.max(...data.intensity),
        peakWavelength: data.wavelength[maxIdx],
        peakEnergy: 1240 / data.wavelength[maxIdx],
        integrated: data.intensity.reduce((a, b) => a + b, 0)
      };
    });
  }, [temperatureSeries]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="w-6 h-6" />
                Photoluminescence Spectroscopy
              </CardTitle>
              <CardDescription>
                Optical emission characterization
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Badge variant={plData ? 'success' : 'secondary'}>
                {plData ? 'Data Loaded' : 'No Data'}
              </Badge>
              {temperatureSeries.length > 0 && (
                <Badge variant="success">T-Series: {temperatureSeries.length}</Badge>
              )}
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="measurement">Measurement</TabsTrigger>
          <TabsTrigger value="analysis">Spectrum</TabsTrigger>
          <TabsTrigger value="temperature">Temperature</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
        </TabsList>

        {/* Measurement Tab */}
        <TabsContent value="measurement">
          <div className="grid grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Measurement Setup</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Excitation Wavelength (nm)</Label>
                  <Select
                    value={params.excitationWavelength.toString()}
                    onValueChange={(value) => setParams({
                      ...params,
                      excitationWavelength: parseInt(value)
                    })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="325">325 nm (UV)</SelectItem>
                      <SelectItem value="405">405 nm (Violet)</SelectItem>
                      <SelectItem value="532">532 nm (Green)</SelectItem>
                      <SelectItem value="633">633 nm (Red)</SelectItem>
                      <SelectItem value="785">785 nm (NIR)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Excitation Power (mW)</Label>
                  <Input
                    type="number"
                    value={params.excitationPower}
                    onChange={(e) => setParams({
                      ...params,
                      excitationPower: parseFloat(e.target.value)
                    })}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Temperature (K)</Label>
                  <div className="flex items-center gap-2">
                    <Slider
                      value={[params.temperature]}
                      onValueChange={(value) => setParams({
                        ...params,
                        temperature: value[0]
                      })}
                      min={4}
                      max={300}
                      step={1}
                      className="flex-1"
                    />
                    <span className="w-12 text-right">
                      {params.temperature} K
                    </span>
                  </div>
                </div>

                <Button
                  onClick={startMeasurement}
                  disabled={isProcessing}
                  className="w-full"
                >
                  {isProcessing ? 'Measuring...' : 'Start Measurement'}
                </Button>

                <Button
                  variant="outline"
                  onClick={measureTemperatureSeries}
                  disabled={isProcessing}
                  className="w-full"
                >
                  <Thermometer className="w-4 h-4 mr-2" />
                  Temperature Series
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Quick Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button
                  variant="outline"
                  onClick={loadDemoData}
                  className="w-full"
                >
                  Load Demo Data
                </Button>
                <Button variant="outline" className="w-full" disabled>
                  Power Series
                </Button>
                <Button variant="outline" className="w-full" disabled>
                  Time-Resolved
                </Button>
                <Button variant="outline" className="w-full" disabled>
                  Mapping Mode
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value="analysis">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">PL Spectrum</CardTitle>
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
                    label={{ value: 'Intensity (a.u.)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="intensity"
                    stroke="#8884d8"
                    name="PL Intensity"
                    strokeWidth={1.5}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>

              {plData && (
                <div className="grid grid-cols-4 gap-4 mt-4">
                  <Card>
                    <CardContent className="pt-6">
                      <div className="text-2xl font-bold">
                        {plData.wavelength[plData.intensity.indexOf(Math.max(...plData.intensity))].toFixed(1)} nm
                      </div>
                      <p className="text-xs text-muted-foreground">Peak Wavelength</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="pt-6">
                      <div className="text-2xl font-bold">
                        {(1240 / plData.wavelength[plData.intensity.indexOf(Math.max(...plData.intensity))]).toFixed(3)} eV
                      </div>
                      <p className="text-xs text-muted-foreground">Peak Energy</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="pt-6">
                      <div className="text-2xl font-bold">
                        {Math.max(...plData.intensity).toFixed(0)}
                      </div>
                      <p className="text-xs text-muted-foreground">Peak Intensity</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="pt-6">
                      <div className="text-2xl font-bold">
                        {plData.temperature} K
                      </div>
                      <p className="text-xs text-muted-foreground">Temperature</p>
                    </CardContent>
                  </Card>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Temperature Tab */}
        <TabsContent value="temperature">
          <div className="space-y-4">
            {temperatureSeries.length > 0 ? (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Temperature Dependence</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={temperatureChartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                          dataKey="temperature"
                          label={{ value: 'Temperature (K)', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis 
                          yAxisId="intensity"
                          label={{ value: 'Peak Intensity', angle: -90, position: 'insideLeft' }}
                        />
                        <YAxis 
                          yAxisId="energy"
                          orientation="right"
                          label={{ value: 'Peak Energy (eV)', angle: 90, position: 'insideRight' }}
                        />
                        <Tooltip />
                        <Legend />
                        <Line
                          yAxisId="intensity"
                          type="monotone"
                          dataKey="peakIntensity"
                          stroke="#8884d8"
                          name="Intensity"
                          strokeWidth={2}
                        />
                        <Line
                          yAxisId="energy"
                          type="monotone"
                          dataKey="peakEnergy"
                          stroke="#82ca9d"
                          name="Energy"
                          strokeWidth={2}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Analysis Results</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <dl className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <dt className="font-semibold">Activation Energy</dt>
                        <dd className="font-mono">~25 meV</dd>
                      </div>
                      <div>
                        <dt className="font-semibold">Varshni α</dt>
                        <dd className="font-mono">5.4×10⁻⁴ eV/K</dd>
                      </div>
                      <div>
                        <dt className="font-semibold">E(0K)</dt>
                        <dd className="font-mono">1.425 eV</dd>
                      </div>
                      <div>
                        <dt className="font-semibold">Quenching Factor</dt>
                        <dd className="font-mono">~100×</dd>
                      </div>
                    </dl>
                  </CardContent>
                </Card>
              </>
            ) : (
              <Alert>
                <Info className="h-4 w-4" />
                <AlertTitle>No Temperature Data</AlertTitle>
                <AlertDescription>
                  Run temperature series measurement to see results
                </AlertDescription>
              </Alert>
            )}
          </div>
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value="results">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Measurement Summary</CardTitle>
            </CardHeader>
            <CardContent>
              {plData ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-semibold mb-2">Measurement Parameters</h4>
                      <dl className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <dt>Excitation:</dt>
                          <dd className="font-mono">{plData.excitationWavelength} nm</dd>
                        </div>
                        <div className="flex justify-between">
                          <dt>Power:</dt>
                          <dd className="font-mono">{plData.excitationPower} mW</dd>
                        </div>
                        <div className="flex justify-between">
                          <dt>Temperature:</dt>
                          <dd className="font-mono">{plData.temperature} K</dd>
                        </div>
                      </dl>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-2">Peak Analysis</h4>
                      <dl className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <dt>Main Peak:</dt>
                          <dd className="font-mono">Band-edge</dd>
                        </div>
                        <div className="flex justify-between">
                          <dt>FWHM:</dt>
                          <dd className="font-mono">~25 meV</dd>
                        </div>
                        <div className="flex justify-between">
                          <dt>Stokes Shift:</dt>
                          <dd className="font-mono">5 meV</dd>
                        </div>
                      </dl>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <Button variant="outline" size="sm">
                      <Download className="w-4 h-4 mr-2" />
                      Export Data
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
                  <AlertTitle>No Results</AlertTitle>
                  <AlertDescription>
                    Perform measurement to see results
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

// Raman Component
const RamanInterface: React.FC = () => {
  // State management
  const [ramanData, setRamanData] = useState<RamanData | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState('setup');
  
  // Measurement parameters
  const [params, setParams] = useState({
    laserWavelength: 532,
    laserPower: 5,
    acquisitionTime: 10,
    accumulations: 1,
    gratingDensity: 1800,
    confocal: true
  });

  // Load demo data
  const loadDemoData = useCallback(() => {
    const ramanShift = Array.from({ length: 600 }, (_, i) => 100 + i * 2);
    
    // Generate Si-like Raman spectrum
    const intensity = ramanShift.map(shift => {
      // Si main peak at 520 cm⁻¹
      const siPeak = 1000 * Math.exp(-0.5 * Math.pow((shift - 520) / 5, 2));
      // Second order peak
      const secondOrder = 100 * Math.exp(-0.5 * Math.pow((shift - 300) / 20, 2));
      // Background
      const background = 50;
      
      return siPeak + secondOrder + background + Math.random() * 10;
    });
    
    setRamanData({
      ramanShift,
      intensity,
      laserWavelength: params.laserWavelength,
      laserPower: params.laserPower
    });
  }, [params]);

  // Chart data
  const spectrumChartData = useMemo(() => {
    if (!ramanData) return [];
    
    return ramanData.ramanShift.map((shift, i) => ({
      ramanShift: shift,
      intensity: ramanData.intensity[i]
    }));
  }, [ramanData]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Prism className="w-6 h-6" />
                Raman Spectroscopy
              </CardTitle>
              <CardDescription>
                Vibrational spectroscopy and stress analysis
              </CardDescription>
            </div>
            <Badge variant={ramanData ? 'success' : 'secondary'}>
              {ramanData ? 'Data Loaded' : 'No Data'}
            </Badge>
          </div>
        </CardHeader>
      </Card>

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-3 w-full">
          <TabsTrigger value="setup">Setup</TabsTrigger>
          <TabsTrigger value="spectrum">Spectrum</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
        </TabsList>

        {/* Setup Tab */}
        <TabsContent value="setup">
          <div className="grid grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Measurement Parameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Laser Wavelength</Label>
                  <Select
                    value={params.laserWavelength.toString()}
                    onValueChange={(value) => setParams({
                      ...params,
                      laserWavelength: parseInt(value)
                    })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="488">488 nm (Blue)</SelectItem>
                      <SelectItem value="532">532 nm (Green)</SelectItem>
                      <SelectItem value="633">633 nm (Red)</SelectItem>
                      <SelectItem value="785">785 nm (NIR)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Laser Power (mW)</Label>
                  <Input
                    type="number"
                    value={params.laserPower}
                    onChange={(e) => setParams({
                      ...params,
                      laserPower: parseFloat(e.target.value)
                    })}
                  />
                </div>

                <Button
                  onClick={loadDemoData}
                  className="w-full"
                >
                  Start Measurement
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Sample Info</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Raman spectroscopy provides information about:
                </p>
                <ul className="mt-2 text-sm text-muted-foreground list-disc list-inside">
                  <li>Material composition</li>
                  <li>Crystal structure</li>
                  <li>Stress/strain state</li>
                  <li>Defect density</li>
                  <li>Layer quality</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Spectrum Tab */}
        <TabsContent value="spectrum">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Raman Spectrum</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={spectrumChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="ramanShift"
                    label={{ value: 'Raman Shift (cm⁻¹)', position: 'insideBottom', offset: -5 }}
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
                  <ReferenceLine x={520} stroke="red" strokeDasharray="5 5" label="Si" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value="analysis">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Peak Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              {ramanData ? (
                <div className="space-y-4">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left p-2">Position (cm⁻¹)</th>
                          <th className="text-left p-2">Assignment</th>
                          <th className="text-left p-2">FWHM</th>
                          <th className="text-left p-2">Intensity</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="border-b">
                          <td className="p-2 font-mono">520.5</td>
                          <td className="p-2">Si TO/LO</td>
                          <td className="p-2 font-mono">5.2</td>
                          <td className="p-2 font-mono">1000</td>
                        </tr>
                        <tr className="border-b">
                          <td className="p-2 font-mono">300</td>
                          <td className="p-2">Si 2TA</td>
                          <td className="p-2 font-mono">20</td>
                          <td className="p-2 font-mono">100</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <Card>
                      <CardContent className="pt-6">
                        <div className="text-2xl font-bold">0.0 GPa</div>
                        <p className="text-xs text-muted-foreground">Stress (no shift)</p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-6">
                        <div className="text-2xl font-bold">95%</div>
                        <p className="text-xs text-muted-foreground">Crystallinity</p>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              ) : (
                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertTitle>No Data</AlertTitle>
                  <AlertDescription>
                    Load spectrum to see analysis
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

// Main Session 8 Component
const Session8OpticalInterface: React.FC = () => {
  const [selectedMethod, setSelectedMethod] = useState<'ellipsometry' | 'pl' | 'raman'>('ellipsometry');

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Session Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl font-bold">
                Session 8: Optical Methods II
              </CardTitle>
              <CardDescription className="mt-2">
                Advanced Optical Characterization: Ellipsometry, Photoluminescence, and Raman Spectroscopy
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button
                variant={selectedMethod === 'ellipsometry' ? 'default' : 'outline'}
                onClick={() => setSelectedMethod('ellipsometry')}
              >
                <Layers className="w-4 h-4 mr-2" />
                Ellipsometry
              </Button>
              <Button
                variant={selectedMethod === 'pl' ? 'default' : 'outline'}
                onClick={() => setSelectedMethod('pl')}
              >
                <Sparkles className="w-4 h-4 mr-2" />
                PL
              </Button>
              <Button
                variant={selectedMethod === 'raman' ? 'default' : 'outline'}
                onClick={() => setSelectedMethod('raman')}
              >
                <Prism className="w-4 h-4 mr-2" />
                Raman
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Method Interface */}
      {selectedMethod === 'ellipsometry' && <EllipsometryInterface />}
      {selectedMethod === 'pl' && <PhotoluminescenceInterface />}
      {selectedMethod === 'raman' && <RamanInterface />}

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

// Export individual interfaces
export { EllipsometryInterface, PhotoluminescenceInterface, RamanInterface };

export default Session8OpticalInterface;
