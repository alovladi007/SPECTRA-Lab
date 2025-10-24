// Session 7: Optical I - Frontend UI Components
// UV-Vis-NIR & FTIR Spectroscopy Interfaces

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, Area, AreaChart, Scatter,
  ScatterChart, Brush, ReferenceArea
} from 'recharts';
import {
  Card, CardContent, CardHeader, CardTitle,
  Button, Input, Label, Select, SelectContent, SelectItem,
  SelectTrigger, SelectValue, Slider, Switch, Tabs, TabsContent,
  TabsList, TabsTrigger, Alert, AlertDescription, Badge,
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger,
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
  Progress, Separator, ScrollArea
} from '@/components/ui';
import {
  Activity, AlertCircle, Download, FileText, Filter, 
  Info, Loader2, Play, RefreshCw, Save, Search, Settings,
  TrendingUp, Upload, ZoomIn, Zap, Database, ChevronRight,
  Eye, EyeOff, Layers, Target, Beaker, FileSpreadsheet,
  CheckCircle2, XCircle, AlertTriangle
} from 'lucide-react';

// ============================================================================
// UV-Vis-NIR Spectroscopy Interface
// ============================================================================

const UVVisNIRInterface = () => {
  // State management
  const [spectrumData, setSpectrumData] = useState(null);
  const [processedData, setProcessedData] = useState(null);
  const [taucPlot, setTaucPlot] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState('measurement');
  const [selectedFile, setSelectedFile] = useState(null);
  
  // Measurement parameters
  const [measurementParams, setMeasurementParams] = useState({
    mode: 'transmission',
    wavelengthStart: 200,
    wavelengthEnd: 800,
    stepSize: 1,
    integrationTime: 100,
    averages: 10,
    reference: 'air'
  });
  
  // Processing parameters
  const [processingParams, setProcessingParams] = useState({
    baselineMethod: 'als',
    smoothing: true,
    smoothWindow: 11,
    removeInterference: false,
    transitionType: 'direct',
    fitRange: 'auto',
    filmThickness: null
  });
  
  // Band gap results
  const [bandGapResults, setBandGapResults] = useState(null);
  const [opticalConstants, setOpticalConstants] = useState(null);
  
  // File upload handler
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      // Parse file and load spectrum
      parseSpectrumFile(file);
    }
  };
  
  // Parse spectrum file (CSV/TXT)
  const parseSpectrumFile = async (file) => {
    const text = await file.text();
    const lines = text.split('\n');
    const data = [];
    
    for (const line of lines) {
      const parts = line.split(/[,\t\s]+/);
      if (parts.length >= 2 && !isNaN(parts[0])) {
        data.push({
          wavelength: parseFloat(parts[0]),
          intensity: parseFloat(parts[1])
        });
      }
    }
    
    setSpectrumData(data);
  };
  
  // Process spectrum
  const processSpectrum = async () => {
    if (!spectrumData) return;
    
    setIsProcessing(true);
    
    try {
      // Simulate API call to process spectrum
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Generate processed data (simulation)
      const processed = spectrumData.map(point => ({
        ...point,
        absorbance: -Math.log10(Math.max(point.intensity / 100, 0.001)),
        corrected: -Math.log10(Math.max(point.intensity / 100, 0.001)) - 0.1
      }));
      
      setProcessedData(processed);
      
      // Calculate band gap (simulation)
      calculateBandGap(processed);
      
    } catch (error) {
      console.error('Processing error:', error);
    } finally {
      setIsProcessing(false);
    }
  };
  
  // Calculate band gap from Tauc plot
  const calculateBandGap = (data) => {
    // Convert wavelength to energy (eV)
    const taucData = data.map(point => {
      const energy = 1240 / point.wavelength; // hc/λ in eV
      const alpha = point.corrected * 1e4; // Absorption coefficient
      
      let taucValue;
      if (processingParams.transitionType === 'direct') {
        taucValue = Math.pow(alpha * energy, 2);
      } else {
        taucValue = Math.pow(alpha * energy, 0.5);
      }
      
      return {
        energy,
        taucValue,
        alpha
      };
    });
    
    setTaucPlot(taucData);
    
    // Find linear region and extract band gap (simulation)
    const bandGap = 2.85 + Math.random() * 0.1; // Example value
    const rSquared = 0.98 + Math.random() * 0.02;
    
    setBandGapResults({
      bandGap: bandGap.toFixed(3),
      uncertainty: 0.05,
      rSquared: rSquared.toFixed(4),
      transitionType: processingParams.transitionType,
      fitRange: [2.5, 3.2]
    });
  };
  
  // Export results
  const exportResults = () => {
    const results = {
      spectrum: processedData,
      bandGap: bandGapResults,
      parameters: { ...measurementParams, ...processingParams },
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(results, null, 2)], 
                          { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'uvvisnir_results.json';
    a.click();
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">UV-Vis-NIR Spectroscopy</h1>
        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            <Settings className="w-4 h-4 mr-1" />
            Settings
          </Button>
          <Button variant="outline" size="sm">
            <FileText className="w-4 h-4 mr-1" />
            Report
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="measurement">Measurement</TabsTrigger>
          <TabsTrigger value="processing">Processing</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
        </TabsList>

        {/* Measurement Tab */}
        <TabsContent value="measurement" className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            {/* Measurement Setup */}
            <Card>
              <CardHeader>
                <CardTitle>Measurement Setup</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label>Measurement Mode</Label>
                  <Select 
                    value={measurementParams.mode}
                    onValueChange={(v) => setMeasurementParams({
                      ...measurementParams, mode: v
                    })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="transmission">Transmission</SelectItem>
                      <SelectItem value="absorbance">Absorbance</SelectItem>
                      <SelectItem value="reflectance">Reflectance</SelectItem>
                      <SelectItem value="transmittance_reflectance">T & R</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <Label>Start (nm)</Label>
                    <Input 
                      type="number" 
                      value={measurementParams.wavelengthStart}
                      onChange={(e) => setMeasurementParams({
                        ...measurementParams,
                        wavelengthStart: parseInt(e.target.value)
                      })}
                    />
                  </div>
                  <div>
                    <Label>End (nm)</Label>
                    <Input 
                      type="number" 
                      value={measurementParams.wavelengthEnd}
                      onChange={(e) => setMeasurementParams({
                        ...measurementParams,
                        wavelengthEnd: parseInt(e.target.value)
                      })}
                    />
                  </div>
                </div>

                <div>
                  <Label>Integration Time (ms)</Label>
                  <Slider 
                    value={[measurementParams.integrationTime]}
                    onValueChange={([v]) => setMeasurementParams({
                      ...measurementParams, integrationTime: v
                    })}
                    min={10}
                    max={1000}
                    step={10}
                  />
                  <span className="text-sm text-muted-foreground">
                    {measurementParams.integrationTime} ms
                  </span>
                </div>

                <div>
                  <Label>Averages</Label>
                  <Input 
                    type="number" 
                    value={measurementParams.averages}
                    onChange={(e) => setMeasurementParams({
                      ...measurementParams,
                      averages: parseInt(e.target.value)
                    })}
                  />
                </div>

                <div className="flex gap-2">
                  <Button className="flex-1">
                    <Play className="w-4 h-4 mr-1" />
                    Start Measurement
                  </Button>
                  <Button variant="outline">
                    <RefreshCw className="w-4 h-4 mr-1" />
                    Reference
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* File Upload */}
            <Card>
              <CardHeader>
                <CardTitle>Load Spectrum</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="border-2 border-dashed rounded-lg p-6 text-center">
                  <Upload className="w-12 h-12 mx-auto mb-2 text-muted-foreground" />
                  <Label htmlFor="file-upload" className="cursor-pointer">
                    <span className="text-primary">Upload file</span> or drag and drop
                  </Label>
                  <Input 
                    id="file-upload"
                    type="file" 
                    className="hidden"
                    accept=".csv,.txt,.asc"
                    onChange={handleFileUpload}
                  />
                  <p className="text-sm text-muted-foreground mt-1">
                    CSV, TXT up to 10MB
                  </p>
                </div>

                {selectedFile && (
                  <div className="bg-secondary rounded p-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">
                        {selectedFile.name}
                      </span>
                      <Badge variant="success">Loaded</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">
                      {spectrumData?.length} data points
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Live Spectrum Display */}
          {spectrumData && (
            <Card>
              <CardHeader>
                <CardTitle>Raw Spectrum</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={spectrumData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="wavelength" 
                      label={{ value: 'Wavelength (nm)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: 'Intensity', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey="intensity" 
                      stroke="#8884d8" 
                      strokeWidth={1.5}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Processing Tab */}
        <TabsContent value="processing" className="space-y-4">
          <div className="grid grid-cols-3 gap-4">
            {/* Baseline Correction */}
            <Card>
              <CardHeader>
                <CardTitle>Baseline Correction</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label>Method</Label>
                  <Select 
                    value={processingParams.baselineMethod}
                    onValueChange={(v) => setProcessingParams({
                      ...processingParams, baselineMethod: v
                    })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="als">Asymmetric LS</SelectItem>
                      <SelectItem value="polynomial">Polynomial</SelectItem>
                      <SelectItem value="rubberband">Rubberband</SelectItem>
                      <SelectItem value="manual">Manual</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center space-x-2">
                  <Switch 
                    checked={processingParams.smoothing}
                    onCheckedChange={(v) => setProcessingParams({
                      ...processingParams, smoothing: v
                    })}
                  />
                  <Label>Apply Smoothing</Label>
                </div>

                {processingParams.smoothing && (
                  <div>
                    <Label>Window Size</Label>
                    <Slider 
                      value={[processingParams.smoothWindow]}
                      onValueChange={([v]) => setProcessingParams({
                        ...processingParams, smoothWindow: v
                      })}
                      min={3}
                      max={21}
                      step={2}
                    />
                    <span className="text-sm text-muted-foreground">
                      {processingParams.smoothWindow} points
                    </span>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Advanced Processing */}
            <Card>
              <CardHeader>
                <CardTitle>Advanced Options</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center space-x-2">
                  <Switch 
                    checked={processingParams.removeInterference}
                    onCheckedChange={(v) => setProcessingParams({
                      ...processingParams, removeInterference: v
                    })}
                  />
                  <Label>Remove Interference Fringes</Label>
                </div>

                <div>
                  <Label>Film Thickness (nm)</Label>
                  <Input 
                    type="number"
                    placeholder="Optional"
                    value={processingParams.filmThickness || ''}
                    onChange={(e) => setProcessingParams({
                      ...processingParams,
                      filmThickness: e.target.value ? parseFloat(e.target.value) : null
                    })}
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    For accurate absorption coefficient
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Band Gap Analysis */}
            <Card>
              <CardHeader>
                <CardTitle>Band Gap Analysis</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label>Transition Type</Label>
                  <Select 
                    value={processingParams.transitionType}
                    onValueChange={(v) => setProcessingParams({
                      ...processingParams, transitionType: v
                    })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="direct">Direct</SelectItem>
                      <SelectItem value="indirect">Indirect</SelectItem>
                      <SelectItem value="direct_forbidden">Direct Forbidden</SelectItem>
                      <SelectItem value="indirect_forbidden">Indirect Forbidden</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label>Fitting Range</Label>
                  <Select 
                    value={processingParams.fitRange}
                    onValueChange={(v) => setProcessingParams({
                      ...processingParams, fitRange: v
                    })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">Automatic</SelectItem>
                      <SelectItem value="manual">Manual</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Process Button */}
          <Card>
            <CardContent className="pt-6">
              <Button 
                className="w-full"
                size="lg"
                onClick={processSpectrum}
                disabled={!spectrumData || isProcessing}
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Zap className="w-5 h-5 mr-2" />
                    Process Spectrum
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Processed Spectrum Display */}
          {processedData && (
            <Card>
              <CardHeader>
                <CardTitle>Processed Spectrum</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={processedData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="wavelength" 
                      label={{ value: 'Wavelength (nm)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: 'Absorbance', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="absorbance" 
                      stroke="#8884d8" 
                      strokeWidth={1.5}
                      dot={false}
                      name="Raw"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="corrected" 
                      stroke="#82ca9d" 
                      strokeWidth={1.5}
                      dot={false}
                      name="Corrected"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value="analysis" className="space-y-4">
          {taucPlot && (
            <>
              {/* Tauc Plot */}
              <Card>
                <CardHeader>
                  <CardTitle>Tauc Plot Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={400}>
                    <ScatterChart data={taucPlot}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="energy" 
                        label={{ value: 'Photon Energy (eV)', position: 'insideBottom', offset: -5 }}
                        domain={['dataMin', 'dataMax']}
                      />
                      <YAxis 
                        label={{ 
                          value: processingParams.transitionType === 'direct' 
                            ? '(αhν)² (eV/cm)²' 
                            : '(αhν)^0.5 (eV/cm)^0.5',
                          angle: -90, 
                          position: 'insideLeft' 
                        }}
                      />
                      <Tooltip />
                      <Scatter 
                        dataKey="taucValue" 
                        fill="#8884d8"
                        fillOpacity={0.6}
                      />
                      {/* Add linear fit line */}
                      <ReferenceLine 
                        segment={[
                          { x: 2.5, y: 0 },
                          { x: 3.5, y: 1000 }
                        ]}
                        stroke="red"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Band Gap Results */}
              {bandGapResults && (
                <div className="grid grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Band Gap Extraction</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">Band Gap</span>
                          <span className="text-2xl font-bold">
                            {bandGapResults.bandGap} ± {bandGapResults.uncertainty} eV
                          </span>
                        </div>
                        <Separator />
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">Transition Type</span>
                          <Badge>{bandGapResults.transitionType}</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">R² Value</span>
                          <span className="font-medium">{bandGapResults.rSquared}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">Fit Range</span>
                          <span className="font-medium">
                            {bandGapResults.fitRange[0]} - {bandGapResults.fitRange[1]} eV
                          </span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Material Identification</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        <Alert>
                          <AlertCircle className="h-4 w-4" />
                          <AlertDescription>
                            Based on the extracted band gap, possible materials:
                          </AlertDescription>
                        </Alert>
                        <div className="space-y-2">
                          <div className="flex items-center justify-between p-2 bg-secondary rounded">
                            <span>GaN</span>
                            <span className="text-sm text-muted-foreground">3.4 eV</span>
                          </div>
                          <div className="flex items-center justify-between p-2 bg-secondary rounded">
                            <span>ZnO</span>
                            <span className="text-sm text-muted-foreground">3.37 eV</span>
                          </div>
                          <div className="flex items-center justify-between p-2 bg-secondary rounded">
                            <span>TiO₂ (anatase)</span>
                            <span className="text-sm text-muted-foreground">3.2 eV</span>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}
            </>
          )}
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value="results" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Analysis Summary</CardTitle>
            </CardHeader>
            <CardContent>
              {bandGapResults ? (
                <div className="space-y-6">
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center p-4 bg-secondary rounded">
                      <p className="text-sm text-muted-foreground">Band Gap</p>
                      <p className="text-2xl font-bold">
                        {bandGapResults.bandGap} eV
                      </p>
                    </div>
                    <div className="text-center p-4 bg-secondary rounded">
                      <p className="text-sm text-muted-foreground">Transition</p>
                      <p className="text-2xl font-bold capitalize">
                        {bandGapResults.transitionType}
                      </p>
                    </div>
                    <div className="text-center p-4 bg-secondary rounded">
                      <p className="text-sm text-muted-foreground">Fit Quality</p>
                      <p className="text-2xl font-bold">
                        {(parseFloat(bandGapResults.rSquared) * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>

                  <Separator />

                  <div className="flex gap-2">
                    <Button onClick={exportResults}>
                      <Download className="w-4 h-4 mr-1" />
                      Export Results
                    </Button>
                    <Button variant="outline">
                      <FileText className="w-4 h-4 mr-1" />
                      Generate Report
                    </Button>
                    <Button variant="outline">
                      <Save className="w-4 h-4 mr-1" />
                      Save to Database
                    </Button>
                  </div>
                </div>
              ) : (
                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertDescription>
                    Process a spectrum to view results
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

// ============================================================================
// FTIR Spectroscopy Interface
// ============================================================================

const FTIRInterface = () => {
  // State management
  const [spectrumData, setSpectrumData] = useState(null);
  const [processedData, setProcessedData] = useState(null);
  const [peaks, setPeaks] = useState([]);
  const [functionalGroups, setFunctionalGroups] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState('measurement');
  
  // Measurement parameters
  const [measurementParams, setMeasurementParams] = useState({
    resolution: 4,
    scans: 32,
    apodization: 'happ-genzel',
    zeroFilling: 2,
    phaseCorrection: 'mertz',
    background: 'air'
  });
  
  // Processing parameters
  const [processingParams, setProcessingParams] = useState({
    baselineMethod: 'als',
    smoothing: true,
    atrCorrection: false,
    atrCrystal: 'ZnSe',
    peakThreshold: 0.01,
    minPeakDistance: 10
  });
  
  // Simulated peak library
  const peakLibrary = [
    { position: 1080, name: 'Si-O stretch', group: 'Silicates' },
    { position: 1650, name: 'C=O stretch', group: 'Carbonyl' },
    { position: 2150, name: 'Si-H stretch', group: 'Silicon hydrides' },
    { position: 2925, name: 'C-H stretch', group: 'Alkanes' },
    { position: 3350, name: 'O-H stretch', group: 'Hydroxyl' }
  ];
  
  // Process FTIR spectrum
  const processFTIRSpectrum = async () => {
    setIsProcessing(true);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Simulate processing and peak detection
      const detectedPeaks = [
        { position: 1078, intensity: 85.2, width: 25, assignment: 'Si-O stretch' },
        { position: 1648, intensity: 72.1, width: 20, assignment: 'C=O stretch' },
        { position: 2148, intensity: 45.3, width: 18, assignment: 'Si-H stretch' },
        { position: 2923, intensity: 61.5, width: 30, assignment: 'C-H stretch' },
        { position: 3348, intensity: 68.9, width: 80, assignment: 'O-H stretch' }
      ];
      
      setPeaks(detectedPeaks);
      
      // Identify functional groups
      const groups = [
        { name: 'Silicates', confidence: 0.95, peaks: [1078] },
        { name: 'Carbonyl compounds', confidence: 0.88, peaks: [1648] },
        { name: 'Silicon hydrides', confidence: 0.76, peaks: [2148] },
        { name: 'Alkyl groups', confidence: 0.92, peaks: [2923] },
        { name: 'Hydroxyl groups', confidence: 0.85, peaks: [3348] }
      ];
      
      setFunctionalGroups(groups);
      
    } catch (error) {
      console.error('Processing error:', error);
    } finally {
      setIsProcessing(false);
    }
  };
  
  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">FTIR Spectroscopy</h1>
        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            <Database className="w-4 h-4 mr-1" />
            Library
          </Button>
          <Button variant="outline" size="sm">
            <FileText className="w-4 h-4 mr-1" />
            Report
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="measurement">Measurement</TabsTrigger>
          <TabsTrigger value="processing">Processing</TabsTrigger>
          <TabsTrigger value="peaks">Peak Analysis</TabsTrigger>
          <TabsTrigger value="identification">Identification</TabsTrigger>
        </TabsList>

        {/* Measurement Tab */}
        <TabsContent value="measurement" className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            {/* FTIR Parameters */}
            <Card>
              <CardHeader>
                <CardTitle>FTIR Parameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label>Resolution (cm⁻¹)</Label>
                  <Select 
                    value={measurementParams.resolution.toString()}
                    onValueChange={(v) => setMeasurementParams({
                      ...measurementParams, resolution: parseInt(v)
                    })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">1</SelectItem>
                      <SelectItem value="2">2</SelectItem>
                      <SelectItem value="4">4</SelectItem>
                      <SelectItem value="8">8</SelectItem>
                      <SelectItem value="16">16</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label>Number of Scans</Label>
                  <Slider 
                    value={[measurementParams.scans]}
                    onValueChange={([v]) => setMeasurementParams({
                      ...measurementParams, scans: v
                    })}
                    min={1}
                    max={128}
                    step={1}
                  />
                  <span className="text-sm text-muted-foreground">
                    {measurementParams.scans} scans
                  </span>
                </div>

                <div>
                  <Label>Apodization</Label>
                  <Select 
                    value={measurementParams.apodization}
                    onValueChange={(v) => setMeasurementParams({
                      ...measurementParams, apodization: v
                    })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="happ-genzel">Happ-Genzel</SelectItem>
                      <SelectItem value="blackman-harris">Blackman-Harris</SelectItem>
                      <SelectItem value="norton-beer">Norton-Beer</SelectItem>
                      <SelectItem value="triangle">Triangle</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Button className="w-full">
                  <Play className="w-4 h-4 mr-1" />
                  Start Measurement
                </Button>
              </CardContent>
            </Card>

            {/* Sample Information */}
            <Card>
              <CardHeader>
                <CardTitle>Sample Setup</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label>Sample Type</Label>
                  <Select defaultValue="solid">
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="solid">Solid</SelectItem>
                      <SelectItem value="liquid">Liquid</SelectItem>
                      <SelectItem value="gas">Gas</SelectItem>
                      <SelectItem value="thin_film">Thin Film</SelectItem>
                      <SelectItem value="powder">Powder</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label>Measurement Mode</Label>
                  <Select defaultValue="transmission">
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="transmission">Transmission</SelectItem>
                      <SelectItem value="atr">ATR</SelectItem>
                      <SelectItem value="diffuse_reflectance">DRIFTS</SelectItem>
                      <SelectItem value="specular_reflectance">Specular</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Alert>
                  <Beaker className="h-4 w-4" />
                  <AlertDescription>
                    Ensure sample is properly prepared and dry.
                    Check for CO₂ and H₂O interference.
                  </AlertDescription>
                </Alert>

                <Button variant="outline" className="w-full">
                  <RefreshCw className="w-4 h-4 mr-1" />
                  Collect Background
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Peak Analysis Tab */}
        <TabsContent value="peaks" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Detected Peaks</CardTitle>
            </CardHeader>
            <CardContent>
              {peaks.length > 0 ? (
                <div className="space-y-4">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Position (cm⁻¹)</TableHead>
                        <TableHead>Intensity</TableHead>
                        <TableHead>Width</TableHead>
                        <TableHead>Assignment</TableHead>
                        <TableHead>Action</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {peaks.map((peak, idx) => (
                        <TableRow key={idx}>
                          <TableCell className="font-medium">
                            {peak.position}
                          </TableCell>
                          <TableCell>{peak.intensity.toFixed(1)}</TableCell>
                          <TableCell>{peak.width}</TableCell>
                          <TableCell>
                            <Badge variant="outline">{peak.assignment}</Badge>
                          </TableCell>
                          <TableCell>
                            <Button variant="ghost" size="sm">
                              <Target className="w-4 h-4" />
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>

                  <div className="flex gap-2">
                    <Button 
                      onClick={processFTIRSpectrum}
                      disabled={isProcessing}
                    >
                      {isProcessing ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Search className="w-4 h-4 mr-1" />
                          Find More Peaks
                        </>
                      )}
                    </Button>
                    <Button variant="outline">
                      <Filter className="w-4 h-4 mr-1" />
                      Filter Peaks
                    </Button>
                  </div>
                </div>
              ) : (
                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertDescription>
                    Process a spectrum to detect peaks
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Identification Tab */}
        <TabsContent value="identification" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Functional Group Identification</CardTitle>
            </CardHeader>
            <CardContent>
              {functionalGroups.length > 0 ? (
                <div className="space-y-4">
                  {functionalGroups.map((group, idx) => (
                    <div key={idx} className="p-4 border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-semibold">{group.name}</h3>
                        <div className="flex items-center gap-2">
                          <Progress value={group.confidence * 100} className="w-24" />
                          <span className="text-sm text-muted-foreground">
                            {(group.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        {group.peaks.map((peak) => (
                          <Badge key={peak} variant="secondary">
                            {peak} cm⁻¹
                          </Badge>
                        ))}
                      </div>
                    </div>
                  ))}

                  <Alert className="bg-green-50 dark:bg-green-900/20">
                    <CheckCircle2 className="h-4 w-4 text-green-600" />
                    <AlertDescription className="text-green-600">
                      Sample contains silicate, carbonyl, and hydroxyl functional groups,
                      consistent with an organic-inorganic hybrid material.
                    </AlertDescription>
                  </Alert>
                </div>
              ) : (
                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertDescription>
                    Process a spectrum to identify functional groups
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

// ============================================================================
// Combined Optical Analysis Dashboard
// ============================================================================

const OpticalAnalysisDashboard = () => {
  const [selectedMethod, setSelectedMethod] = useState('uvvisnir');

  return (
    <div className="min-h-screen bg-background">
      <div className="border-b">
        <div className="flex h-16 items-center px-4 gap-4">
          <Layers className="w-6 h-6" />
          <h1 className="text-xl font-semibold">Optical Spectroscopy Suite</h1>
          <Separator orientation="vertical" className="h-6" />
          <div className="flex gap-2">
            <Button 
              variant={selectedMethod === 'uvvisnir' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedMethod('uvvisnir')}
            >
              UV-Vis-NIR
            </Button>
            <Button 
              variant={selectedMethod === 'ftir' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedMethod('ftir')}
            >
              FTIR
            </Button>
          </div>
        </div>
      </div>

      <div className="container mx-auto">
        {selectedMethod === 'uvvisnir' ? (
          <UVVisNIRInterface />
        ) : (
          <FTIRInterface />
        )}
      </div>
    </div>
  );
};

export default OpticalAnalysisDashboard;