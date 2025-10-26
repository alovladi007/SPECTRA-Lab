'use client'

// Session 6: Electrical III - UI Components (Continued)
// Completing EBIC and PCD Measurement Interfaces

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Slider } from '@/components/ui/slider';
import { LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Area, AreaChart } from 'recharts';
import { 
  Play, Square, Download, Settings, TrendingUp, Thermometer, Zap, 
  AlertCircle, CheckCircle, Activity, Cpu, Camera, Clock, Target,
  BarChart, Map, Layers, Sun, Timer
} from 'lucide-react';

// ==========================================
// 2. EBIC Mapping Interface (Continued)
// ==========================================

export const EBICMappingComplete = () => {
  const [isScanning, setIsScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [results, setResults] = useState(null);
  const canvasRef = useRef(null);
  
  const [config, setConfig] = useState({
    scanArea: '100x100',
    pixelSize: '0.5',
    beamEnergy: '20',
    beamCurrent: '100',
    dwellTime: '10',
    temperature: '300',
    biasVoltage: '0'
  });

  const [viewMode, setViewMode] = useState('ebic');
  const [selectedDefect, setSelectedDefect] = useState(null);
  const [lineProfile, setLineProfile] = useState([]);

  const handleStartScan = async () => {
    setIsScanning(true);
    setScanProgress(0);
    
    const interval = setInterval(() => {
      setScanProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          generateEBICMap();
          return 100;
        }
        return prev + 2;
      });
    }, 100);
  };

  const generateEBICMap = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = 256;
    const height = 256;
    canvas.width = width;
    canvas.height = height;
    
    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;
    
    const junctionX = width / 2;
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const distance = Math.abs(x - junctionX);
        
        let signal = Math.exp(-distance / 30) * 255;
        
        if (Math.random() < 0.001) {
          signal *= 0.3;
        }
        
        signal += (Math.random() - 0.5) * 20;
        signal = Math.max(0, Math.min(255, signal));
        
        if (signal < 64) {
          data[idx] = 0;
          data[idx + 1] = 0;
          data[idx + 2] = 4 * signal;
        } else if (signal < 128) {
          data[idx] = 0;
          data[idx + 1] = 4 * (signal - 64);
          data[idx + 2] = 255;
        } else if (signal < 192) {
          data[idx] = 4 * (signal - 128);
          data[idx + 1] = 255;
          data[idx + 2] = 255 - 4 * (signal - 128);
        } else {
          data[idx] = 255;
          data[idx + 1] = 255 - 4 * (signal - 192);
          data[idx + 2] = 0;
        }
        
        data[idx + 3] = 255;
      }
    }
    
    ctx.putImageData(imageData, 0, 0);
    
    // Generate line profile
    const profile = [];
    for (let i = 0; i < 100; i++) {
      const distance = i * 0.5;
      const current = 100 * Math.exp(-distance / 45);
      profile.push({ distance, current });
    }
    setLineProfile(profile);
    
    setResults({
      diffusion_length: {
        mean: 45.2,
        std: 5.3,
        unit: 'µm'
      },
      defects: [
        { id: 1, x: 120, y: 80, contrast: -0.6, area: 25, type: 'Strong recombination' },
        { id: 2, x: 180, y: 150, contrast: -0.4, area: 15, type: 'Moderate recombination' },
        { id: 3, x: 90, y: 200, contrast: -0.7, area: 30, type: 'Strong recombination' }
      ],
      quality_score: 88,
      statistics: {
        mean_current: 52.3,
        max_current: 98.5,
        min_current: 2.1,
        uniformity: 0.42
      }
    });
    
    setIsScanning(false);
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Map className="h-8 w-8 text-purple-600" />
            EBIC Mapping
          </h1>
          <p className="text-gray-600 mt-1">Electron beam induced current imaging for defect analysis</p>
        </div>
        {isScanning && (
          <Badge variant="outline" className="text-lg px-4 py-2">
            <Activity className="h-4 w-4 mr-1 animate-pulse" />
            Scanning: {scanProgress}%
          </Badge>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Configuration Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Scan Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label>Scan Area</Label>
              <Select 
                value={config.scanArea}
                onValueChange={(v) => setConfig({...config, scanArea: v})}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="10x10">10×10 µm</SelectItem>
                  <SelectItem value="50x50">50×50 µm</SelectItem>
                  <SelectItem value="100x100">100×100 µm</SelectItem>
                  <SelectItem value="500x500">500×500 µm</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label>Pixel Size (µm)</Label>
              <Input 
                value={config.pixelSize}
                onChange={(e) => setConfig({...config, pixelSize: e.target.value})}
              />
            </div>

            <div className="space-y-2">
              <Label>Beam Settings</Label>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <Label className="text-xs">Energy (keV)</Label>
                  <Input 
                    value={config.beamEnergy}
                    onChange={(e) => setConfig({...config, beamEnergy: e.target.value})}
                  />
                </div>
                <div>
                  <Label className="text-xs">Current (pA)</Label>
                  <Input 
                    value={config.beamCurrent}
                    onChange={(e) => setConfig({...config, beamCurrent: e.target.value})}
                  />
                </div>
              </div>
            </div>

            <Button 
              onClick={handleStartScan}
              disabled={isScanning}
              className="w-full"
              variant={isScanning ? "secondary" : "default"}
            >
              {isScanning ? (
                <>
                  <Square className="mr-2 h-4 w-4" />
                  Stop Scan
                </>
              ) : (
                <>
                  <Camera className="mr-2 h-4 w-4" />
                  Start Scan
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Map Display */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle>EBIC Map</CardTitle>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant={viewMode === 'ebic' ? 'default' : 'outline'}
                    onClick={() => setViewMode('ebic')}
                  >
                    EBIC
                  </Button>
                  <Button
                    size="sm"
                    variant={viewMode === 'sem' ? 'default' : 'outline'}
                    onClick={() => setViewMode('sem')}
                  >
                    SEM
                  </Button>
                  <Button
                    size="sm"
                    variant={viewMode === 'overlay' ? 'default' : 'outline'}
                    onClick={() => setViewMode('overlay')}
                  >
                    Overlay
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="relative">
                <canvas 
                  ref={canvasRef}
                  className="w-full border rounded"
                  style={{ imageRendering: 'pixelated' }}
                />
                {results && viewMode !== 'sem' && results.defects.map(defect => (
                  <div
                    key={defect.id}
                    className="absolute w-4 h-4 border-2 border-red-500 rounded-full cursor-pointer"
                    style={{
                      left: `${(defect.x / 256) * 100}%`,
                      top: `${(defect.y / 256) * 100}%`,
                      transform: 'translate(-50%, -50%)'
                    }}
                    onClick={() => setSelectedDefect(defect)}
                  />
                ))}
              </div>
              
              <div className="flex justify-between items-center mt-2 text-sm text-gray-600">
                <span>0 µm</span>
                <span>{config.scanArea.split('x')[0]} µm</span>
              </div>
            </CardContent>
          </Card>

          {results && (
            <>
              {/* Line Profile */}
              <Card>
                <CardHeader>
                  <CardTitle>Diffusion Length Extraction</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={lineProfile}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="distance" label={{ value: 'Distance (µm)', position: 'insideBottom', offset: -5 }} />
                      <YAxis label={{ value: 'EBIC Current (nA)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Line type="monotone" dataKey="current" stroke="#9333ea" strokeWidth={2} dot={false} />
                      <ReferenceLine 
                        y={100 * Math.exp(-1)} 
                        stroke="#666" 
                        strokeDasharray="5 5"
                        label="1/e"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <div className="p-3 bg-purple-50 rounded-lg">
                      <div className="text-sm text-gray-600">L_d (mean)</div>
                      <div className="text-xl font-bold text-purple-600">
                        {results.diffusion_length.mean} µm
                      </div>
                    </div>
                    <div className="p-3 bg-blue-50 rounded-lg">
                      <div className="text-sm text-gray-600">Std Dev</div>
                      <div className="text-xl font-bold text-blue-600">
                        ±{results.diffusion_length.std} µm
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Defect Analysis */}
              <Card>
                <CardHeader>
                  <CardTitle>Defect Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {results.defects.map(defect => (
                      <div 
                        key={defect.id}
                        className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                          selectedDefect?.id === defect.id ? 'bg-purple-50 border-purple-300' : ''
                        }`}
                        onClick={() => setSelectedDefect(defect)}
                      >
                        <div className="flex justify-between items-center">
                          <div>
                            <span className="font-semibold">Defect #{defect.id}</span>
                            <p className="text-sm text-gray-600">{defect.type}</p>
                          </div>
                          <div className="text-right">
                            <Badge variant="destructive">
                              {(defect.contrast * 100).toFixed(0)}% contrast
                            </Badge>
                            <p className="text-xs text-gray-600 mt-1">Area: {defect.area} px²</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

// ==========================================
// 3. PCD Lifetime Measurement Interface
// ==========================================

export const PCDMeasurement = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [measurementMode, setMeasurementMode] = useState('transient'); // transient, qsspc
  const [results, setResults] = useState(null);
  
  const [config, setConfig] = useState({
    mode: 'transient',
    excitationWavelength: '904',
    photonFlux: '1e15',
    pulseWidth: '10',
    temperature: '300',
    sampleThickness: '300',
    sampleArea: '1',
    surfaceCondition: 'passivated'
  });

  const [decayData, setDecayData] = useState([]);
  const [injectionData, setInjectionData] = useState([]);

  const handleStartMeasurement = async () => {
    setIsRunning(true);
    
    setTimeout(() => {
      if (measurementMode === 'transient') {
        generateTransientData();
      } else {
        generateQSSPCData();
      }
      setIsRunning(false);
    }, 3000);
  };

  const generateTransientData = () => {
    // Generate photoconductance decay
    const tau = 100e-6; // 100 µs lifetime
    const data = [];
    
    for (let i = 0; i < 200; i++) {
      const time = i * 5e-6; // 5 µs steps
      const photoconductance = 1e-3 * Math.exp(-time / tau);
      const carrierDensity = 1e15 * Math.exp(-time / tau);
      const lifetime = tau * 1e6; // Convert to µs
      
      data.push({
        time: time * 1e6, // Convert to µs
        photoconductance: photoconductance,
        carrierDensity: carrierDensity,
        lifetime: lifetime
      });
    }
    
    setDecayData(data);
    
    setResults({
      tau_effective: 100,
      tau_bulk: 150,
      tau_surface: 300,
      srv_effective: 10,
      srv_front: 10,
      srv_back: 10,
      quality_score: 91,
      mechanisms: {
        srh: 'dominant',
        auger: 'negligible',
        radiative: 'negligible'
      }
    });
  };

  const generateQSSPCData = () => {
    // Generate injection-dependent lifetime
    const data = [];
    
    for (let i = 0; i < 50; i++) {
      const injection = Math.pow(10, 13 + i * 0.08); // Log scale from 1e13 to 1e17
      const tau_srh = 500 / (1 + injection / 1e15);
      const tau_auger = 1 / (1.66e-30 * injection * injection);
      const tau_surface = 300;
      
      const tau_eff = 1 / (1/tau_srh + 1/tau_auger + 1/tau_surface);
      
      data.push({
        injection: injection,
        lifetime: tau_eff * 1e6, // Convert to µs
        bulk: (1 / (1/tau_srh + 1/tau_auger)) * 1e6,
        surface: tau_surface * 1e6
      });
    }
    
    setInjectionData(data);
    
    setResults({
      tau_low_injection: 350,
      tau_high_injection: 15,
      crossover: 1e15,
      srv_effective: 10,
      auger_coefficient: 1.66e-30,
      quality_score: 93
    });
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Clock className="h-8 w-8 text-green-600" />
            PCD Lifetime Measurement
          </h1>
          <p className="text-gray-600 mt-1">Photoconductance decay for minority carrier lifetime</p>
        </div>
        <Badge variant="outline" className="text-lg px-4 py-2">
          {measurementMode === 'transient' ? 'Transient Mode' : 'QSSPC Mode'}
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Configuration Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Measurement Mode */}
            <div>
              <Label>Mode</Label>
              <Select 
                value={measurementMode}
                onValueChange={setMeasurementMode}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="transient">Transient PCD</SelectItem>
                  <SelectItem value="qsspc">Quasi-Steady-State (QSSPC)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Excitation */}
            <div>
              <Label>Wavelength (nm)</Label>
              <Select 
                value={config.excitationWavelength}
                onValueChange={(v) => setConfig({...config, excitationWavelength: v})}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="650">650 (Red)</SelectItem>
                  <SelectItem value="808">808 (NIR)</SelectItem>
                  <SelectItem value="904">904 (NIR)</SelectItem>
                  <SelectItem value="1064">1064 (IR)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Photon Flux */}
            <div>
              <Label>Photon Flux (cm⁻²s⁻¹)</Label>
              <Input 
                value={config.photonFlux}
                onChange={(e) => setConfig({...config, photonFlux: e.target.value})}
                placeholder="1e15"
              />
            </div>

            {measurementMode === 'transient' && (
              <div>
                <Label>Pulse Width (µs)</Label>
                <Input 
                  value={config.pulseWidth}
                  onChange={(e) => setConfig({...config, pulseWidth: e.target.value})}
                />
              </div>
            )}

            {/* Sample Parameters */}
            <div className="space-y-2">
              <Label>Sample Parameters</Label>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <Label className="text-xs">Thickness (µm)</Label>
                  <Input 
                    value={config.sampleThickness}
                    onChange={(e) => setConfig({...config, sampleThickness: e.target.value})}
                  />
                </div>
                <div>
                  <Label className="text-xs">Area (cm²)</Label>
                  <Input 
                    value={config.sampleArea}
                    onChange={(e) => setConfig({...config, sampleArea: e.target.value})}
                  />
                </div>
              </div>
            </div>

            {/* Surface Condition */}
            <div>
              <Label>Surface</Label>
              <Select 
                value={config.surfaceCondition}
                onValueChange={(v) => setConfig({...config, surfaceCondition: v})}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="passivated">Passivated (SiNx)</SelectItem>
                  <SelectItem value="oxide">Thermal Oxide</SelectItem>
                  <SelectItem value="bare">Bare Silicon</SelectItem>
                  <SelectItem value="alox">Al₂O₃</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Temperature */}
            <div>
              <Label>Temperature (K)</Label>
              <Input 
                value={config.temperature}
                onChange={(e) => setConfig({...config, temperature: e.target.value})}
              />
            </div>

            {/* Start Button */}
            <Button 
              onClick={handleStartMeasurement}
              disabled={isRunning}
              className="w-full"
              variant={isRunning ? "secondary" : "default"}
            >
              {isRunning ? (
                <>
                  <Square className="mr-2 h-4 w-4" />
                  Measuring...
                </>
              ) : (
                <>
                  <Sun className="mr-2 h-4 w-4" />
                  Start Measurement
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results Display */}
        <div className="lg:col-span-2 space-y-6">
          {/* Main Plot */}
          <Card>
            <CardHeader>
              <CardTitle>
                {measurementMode === 'transient' ? 'Photoconductance Decay' : 'Injection-Dependent Lifetime'}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {measurementMode === 'transient' ? (
                <Tabs defaultValue="decay" className="w-full">
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="decay">Decay Curve</TabsTrigger>
                    <TabsTrigger value="lifetime">Effective Lifetime</TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="decay">
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={decayData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                          dataKey="time" 
                          label={{ value: 'Time (µs)', position: 'insideBottom', offset: -5 }}
                          scale="log"
                          domain={[1, 1000]}
                        />
                        <YAxis 
                          scale="log"
                          label={{ value: 'Photoconductance (S)', angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip formatter={(value) => value.toExponential(2)} />
                        <Line type="monotone" dataKey="photoconductance" stroke="#10b981" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </TabsContent>
                  
                  <TabsContent value="lifetime">
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={decayData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                          dataKey="carrierDensity" 
                          scale="log"
                          label={{ value: 'Excess Carrier Density (cm⁻³)', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis label={{ value: 'Lifetime (µs)', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Line type="monotone" dataKey="lifetime" stroke="#10b981" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </TabsContent>
                </Tabs>
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={injectionData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="injection" 
                      scale="log"
                      domain={[1e13, 1e17]}
                      label={{ value: 'Injection Level (cm⁻³)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      scale="log"
                      domain={[10, 1000]}
                      label={{ value: 'Lifetime (µs)', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip formatter={(value) => `${value.toFixed(1)} µs`} />
                    <Legend />
                    <Line type="monotone" dataKey="lifetime" stroke="#10b981" strokeWidth={2} dot={false} name="Effective" />
                    <Line type="monotone" dataKey="bulk" stroke="#3b82f6" strokeWidth={2} dot={false} name="Bulk" />
                    <Line type="monotone" dataKey="surface" stroke="#f59e0b" strokeWidth={2} dot={false} name="Surface" />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>

          {/* Lifetime Parameters */}
          {results && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>Extracted Parameters</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4">
                    {measurementMode === 'transient' ? (
                      <>
                        <div className="p-4 bg-green-50 rounded-lg">
                          <div className="text-sm text-gray-600">τ_eff</div>
                          <div className="text-2xl font-bold text-green-600">
                            {results.tau_effective} µs
                          </div>
                        </div>
                        <div className="p-4 bg-blue-50 rounded-lg">
                          <div className="text-sm text-gray-600">τ_bulk</div>
                          <div className="text-2xl font-bold text-blue-600">
                            {results.tau_bulk} µs
                          </div>
                        </div>
                        <div className="p-4 bg-yellow-50 rounded-lg">
                          <div className="text-sm text-gray-600">τ_surface</div>
                          <div className="text-2xl font-bold text-yellow-600">
                            {results.tau_surface} µs
                          </div>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="p-4 bg-green-50 rounded-lg">
                          <div className="text-sm text-gray-600">τ (low inj.)</div>
                          <div className="text-2xl font-bold text-green-600">
                            {results.tau_low_injection} µs
                          </div>
                        </div>
                        <div className="p-4 bg-blue-50 rounded-lg">
                          <div className="text-sm text-gray-600">τ (high inj.)</div>
                          <div className="text-2xl font-bold text-blue-600">
                            {results.tau_high_injection} µs
                          </div>
                        </div>
                        <div className="p-4 bg-purple-50 rounded-lg">
                          <div className="text-sm text-gray-600">Crossover</div>
                          <div className="text-2xl font-bold text-purple-600">
                            {results.crossover.toExponential(1)} cm⁻³
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Surface Recombination */}
              <Card>
                <CardHeader>
                  <CardTitle>Surface Recombination Velocity</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="p-4 bg-indigo-50 rounded-lg">
                      <div className="text-sm text-gray-600">S_eff</div>
                      <div className="text-2xl font-bold text-indigo-600">
                        {results.srv_effective || results.srv_effective} cm/s
                      </div>
                    </div>
                    {measurementMode === 'transient' && (
                      <>
                        <div className="p-4 bg-pink-50 rounded-lg">
                          <div className="text-sm text-gray-600">S_front</div>
                          <div className="text-2xl font-bold text-pink-600">
                            {results.srv_front} cm/s
                          </div>
                        </div>
                        <div className="p-4 bg-cyan-50 rounded-lg">
                          <div className="text-sm text-gray-600">S_back</div>
                          <div className="text-2xl font-bold text-cyan-600">
                            {results.srv_back} cm/s
                          </div>
                        </div>
                      </>
                    )}
                    {measurementMode === 'qsspc' && (
                      <div className="p-4 bg-red-50 rounded-lg">
                        <div className="text-sm text-gray-600">C_Auger</div>
                        <div className="text-2xl font-bold text-red-600">
                          {results.auger_coefficient?.toExponential(2) || '1.66e-30'} cm⁶/s
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Quality Assessment */}
              <Card>
                <CardHeader>
                  <CardTitle>Measurement Quality</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        <span className="text-sm">Signal-to-noise ratio: Excellent</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        <span className="text-sm">Dynamic range: 3.5 decades</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        <span className="text-sm">Injection coverage: Complete</span>
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-4xl font-bold text-green-600">
                        {results.quality_score}
                      </div>
                      <div className="text-sm text-gray-600">Quality Score</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

// Export all Session 6 components
export default {
  DLTSMeasurement,
  EBICMapping: EBICMappingComplete,
  PCDMeasurement
};