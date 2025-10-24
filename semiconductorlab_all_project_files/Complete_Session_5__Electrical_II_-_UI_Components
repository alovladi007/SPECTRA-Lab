// Complete Session 5: Electrical II - UI Components
// MOSFET, C-V Profiling, and BJT Characterization Interfaces

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, ReferenceLine } from 'recharts';
import { Play, Square, Download, Settings, TrendingUp, Cpu, Activity, Zap, AlertCircle, CheckCircle } from 'lucide-react';

// ==========================================
// 1. MOSFET Characterization Interface
// ==========================================

export const MOSFETCharacterization = () => {
  const [deviceType, setDeviceType] = useState('n-mos');
  const [measurementType, setMeasurementType] = useState('transfer');
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState(null);
  
  const [config, setConfig] = useState({
    width: '10',
    length: '1',
    oxideThickness: '10',
    vds: '0.1',
    vgsStart: '-1',
    vgsStop: '3',
    vgsStep: '0.02',
    temperature: '300'
  });

  const [transferData, setTransferData] = useState([]);
  const [outputData, setOutputData] = useState([]);

  const handleRunMeasurement = async () => {
    setIsRunning(true);
    
    // Simulate measurement and data generation
    setTimeout(() => {
      // Generate transfer curve data
      const data = [];
      const vgsStart = parseFloat(config.vgsStart);
      const vgsStop = parseFloat(config.vgsStop);
      const vgsStep = parseFloat(config.vgsStep);
      const vth = deviceType === 'n-mos' ? 0.5 : -0.5;
      
      for (let vgs = vgsStart; vgs <= vgsStop; vgs += vgsStep) {
        const vgsEff = deviceType === 'n-mos' ? vgs - vth : vth - vgs;
        let ids;
        
        if (vgsEff <= 0) {
          // Subthreshold region
          ids = 1e-12 * Math.exp(vgsEff / 0.06);
        } else {
          // Above threshold (square law)
          ids = 1e-4 * vgsEff * vgsEff;
        }
        
        data.push({
          vgs: vgs,
          ids: ids,
          logIds: Math.log10(Math.max(ids, 1e-15))
        });
      }
      
      setTransferData(data);
      
      // Set results
      setResults({
        thresholdVoltage: {
          linearExtrapolation: vth,
          constantCurrent: vth + 0.02,
          maxGm: vth - 0.01
        },
        transconductance: {
          max: 2.5e-3,
          atVgs: 1.5
        },
        mobility: {
          value: 450,
          unit: 'cm²/(V·s)'
        },
        ionIoffRatio: 1e6,
        subthresholdSlope: 65,
        qualityScore: 94
      });
      
      setIsRunning(false);
    }, 2000);
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Cpu className="h-8 w-8 text-blue-600" />
            MOSFET I-V Characterization
          </h1>
          <p className="text-gray-600 mt-1">Transfer and output characteristics analysis</p>
        </div>
        <Badge variant="outline" className="text-lg px-4 py-2">
          {deviceType === 'n-mos' ? 'n-MOS' : 'p-MOS'}
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Configuration Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Device Type */}
            <div>
              <Label>Device Type</Label>
              <Select value={deviceType} onValueChange={setDeviceType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="n-mos">n-MOS</SelectItem>
                  <SelectItem value="p-mos">p-MOS</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Measurement Type */}
            <div>
              <Label>Measurement Type</Label>
              <Select value={measurementType} onValueChange={setMeasurementType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="transfer">Transfer (Id-Vgs)</SelectItem>
                  <SelectItem value="output">Output (Id-Vds)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Device Geometry */}
            <div className="space-y-2">
              <Label>Device Geometry</Label>
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <Label className="text-xs">W (µm)</Label>
                  <Input 
                    value={config.width}
                    onChange={(e) => setConfig({...config, width: e.target.value})}
                  />
                </div>
                <div>
                  <Label className="text-xs">L (µm)</Label>
                  <Input 
                    value={config.length}
                    onChange={(e) => setConfig({...config, length: e.target.value})}
                  />
                </div>
                <div>
                  <Label className="text-xs">tox (nm)</Label>
                  <Input 
                    value={config.oxideThickness}
                    onChange={(e) => setConfig({...config, oxideThickness: e.target.value})}
                  />
                </div>
              </div>
            </div>

            {/* Voltage Configuration */}
            {measurementType === 'transfer' && (
              <div className="space-y-2">
                <Label>Voltage Settings</Label>
                <div>
                  <Label className="text-xs">Vds (V)</Label>
                  <Input 
                    value={config.vds}
                    onChange={(e) => setConfig({...config, vds: e.target.value})}
                  />
                </div>
                <div className="grid grid-cols-3 gap-2">
                  <div>
                    <Label className="text-xs">Vgs Start</Label>
                    <Input 
                      value={config.vgsStart}
                      onChange={(e) => setConfig({...config, vgsStart: e.target.value})}
                    />
                  </div>
                  <div>
                    <Label className="text-xs">Vgs Stop</Label>
                    <Input 
                      value={config.vgsStop}
                      onChange={(e) => setConfig({...config, vgsStop: e.target.value})}
                    />
                  </div>
                  <div>
                    <Label className="text-xs">Step</Label>
                    <Input 
                      value={config.vgsStep}
                      onChange={(e) => setConfig({...config, vgsStep: e.target.value})}
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Run Button */}
            <Button 
              onClick={handleRunMeasurement}
              disabled={isRunning}
              className="w-full"
              variant={isRunning ? "secondary" : "default"}
            >
              {isRunning ? (
                <>
                  <Square className="mr-2 h-4 w-4" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Start Measurement
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results Display */}
        <div className="lg:col-span-2 space-y-6">
          {/* Plots */}
          <Card>
            <CardHeader>
              <CardTitle>Transfer Characteristics</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="linear" className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="linear">Linear Scale</TabsTrigger>
                  <TabsTrigger value="log">Log Scale</TabsTrigger>
                </TabsList>
                <TabsContent value="linear">
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={transferData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="vgs" label={{ value: 'Vgs (V)', position: 'insideBottom', offset: -5 }} />
                      <YAxis label={{ value: 'Ids (A)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip formatter={(value) => value.toExponential(2)} />
                      <Line type="monotone" dataKey="ids" stroke="#2563eb" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </TabsContent>
                <TabsContent value="log">
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={transferData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="vgs" label={{ value: 'Vgs (V)', position: 'insideBottom', offset: -5 }} />
                      <YAxis label={{ value: 'log(Ids)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Line type="monotone" dataKey="logIds" stroke="#2563eb" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          {/* Parameter Results */}
          {results && (
            <Card>
              <CardHeader>
                <CardTitle>Extracted Parameters</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <div className="text-sm text-gray-600">Threshold Voltage</div>
                    <div className="text-2xl font-bold text-blue-600">
                      {results.thresholdVoltage.linearExtrapolation.toFixed(3)} V
                    </div>
                  </div>
                  <div className="p-4 bg-green-50 rounded-lg">
                    <div className="text-sm text-gray-600">Max gm</div>
                    <div className="text-2xl font-bold text-green-600">
                      {(results.transconductance.max * 1000).toFixed(2)} mS
                    </div>
                  </div>
                  <div className="p-4 bg-purple-50 rounded-lg">
                    <div className="text-sm text-gray-600">Mobility</div>
                    <div className="text-2xl font-bold text-purple-600">
                      {results.mobility.value} {results.mobility.unit}
                    </div>
                  </div>
                  <div className="p-4 bg-yellow-50 rounded-lg">
                    <div className="text-sm text-gray-600">Ion/Ioff</div>
                    <div className="text-2xl font-bold text-yellow-700">
                      {results.ionIoffRatio.toExponential(1)}
                    </div>
                  </div>
                  <div className="p-4 bg-red-50 rounded-lg">
                    <div className="text-sm text-gray-600">SS (mV/dec)</div>
                    <div className="text-2xl font-bold text-red-600">
                      {results.subthresholdSlope}
                    </div>
                  </div>
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="text-sm text-gray-600">Quality Score</div>
                    <div className="text-2xl font-bold text-gray-700">
                      {results.qualityScore}/100
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

// ==========================================
// 2. C-V Profiling Interface
// ==========================================

export const CVProfiling = () => {
  const [deviceType, setDeviceType] = useState('mos');
  const [substrateType, setSubstrateType] = useState('n-type');
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState(null);

  const [config, setConfig] = useState({
    frequency: '1e6',
    vStart: '-3',
    vStop: '3',
    vStep: '0.05',
    area: '1e-4',
    temperature: '300'
  });

  const [cvData, setCvData] = useState([]);
  const [mottSchottkyData, setMottSchottkyData] = useState([]);
  const [dopingProfileData, setDopingProfileData] = useState([]);

  const handleRunMeasurement = async () => {
    setIsRunning(true);

    setTimeout(() => {
      // Generate C-V data
      const cv = [];
      const ms = [];
      const vStart = parseFloat(config.vStart);
      const vStop = parseFloat(config.vStop);
      const vStep = parseFloat(config.vStep);
      
      for (let v = vStart; v <= vStop; v += vStep) {
        let c;
        if (deviceType === 'mos') {
          // MOS capacitor model
          const vfb = -0.5;
          const cox = 3.45e-7; // F/cm²
          const cmin = cox * 0.1;
          
          if (v < vfb) {
            c = cmin + (cox - cmin) / (1 + Math.exp((v - vfb) / 0.1));
          } else {
            c = cox / (1 + Math.exp((v - vfb - 0.5) / 0.2));
          }
        } else {
          // Schottky diode model
          const vbi = 0.7;
          const n0 = 1e16;
          c = 1e-7 / Math.sqrt(Math.max(vbi - v, 0.01));
        }
        
        cv.push({ voltage: v, capacitance: c * 1e9 }); // Convert to nF
        ms.push({ voltage: v, invC2: 1 / (c * c) });
      }
      
      setCvData(cv);
      setMottSchottkyData(ms);
      
      // Generate doping profile
      const doping = [];
      for (let depth = 0; depth < 2; depth += 0.05) {
        doping.push({
          depth: depth,
          concentration: 1e16 * Math.exp(-depth / 0.5)
        });
      }
      setDopingProfileData(doping);
      
      // Set results
      if (deviceType === 'mos') {
        setResults({
          cox: { value: 345, unit: 'nF/cm²' },
          tox: { value: 10, unit: 'nm' },
          vfb: { value: -0.5, unit: 'V' },
          vth: { value: 0.5, unit: 'V' },
          dit: { value: 2e11, unit: 'cm⁻²eV⁻¹' },
          substrateDopping: { value: 1e16, unit: 'cm⁻³' },
          qualityScore: 92
        });
      } else {
        setResults({
          vbi: { value: 0.7, unit: 'V' },
          dopingConcentration: { value: 1e16, unit: 'cm⁻³' },
          depletionWidth: { value: 0.35, unit: 'µm' },
          barrierHeight: { value: 0.85, unit: 'eV' },
          qualityScore: 89
        });
      }
      
      setIsRunning(false);
    }, 2000);
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Activity className="h-8 w-8 text-purple-600" />
            C-V Profiling
          </h1>
          <p className="text-gray-600 mt-1">Capacitance-voltage analysis and doping profile extraction</p>
        </div>
        <Badge variant="outline" className="text-lg px-4 py-2">
          {deviceType === 'mos' ? 'MOS Capacitor' : 'Schottky Diode'}
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Configuration Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Device Type */}
            <div>
              <Label>Device Type</Label>
              <Select value={deviceType} onValueChange={setDeviceType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="mos">MOS Capacitor</SelectItem>
                  <SelectItem value="schottky">Schottky Diode</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Substrate Type */}
            <div>
              <Label>Substrate Type</Label>
              <Select value={substrateType} onValueChange={setSubstrateType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="n-type">n-type</SelectItem>
                  <SelectItem value="p-type">p-type</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Frequency */}
            <div>
              <Label>Frequency (Hz)</Label>
              <Select value={config.frequency} onValueChange={(v) => setConfig({...config, frequency: v})}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1e3">1 kHz</SelectItem>
                  <SelectItem value="1e4">10 kHz</SelectItem>
                  <SelectItem value="1e5">100 kHz</SelectItem>
                  <SelectItem value="1e6">1 MHz</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Area */}
            <div>
              <Label>Device Area (cm²)</Label>
              <Input 
                value={config.area}
                onChange={(e) => setConfig({...config, area: e.target.value})}
                placeholder="1e-4"
              />
            </div>

            {/* Voltage Sweep */}
            <div className="space-y-2">
              <Label>Voltage Sweep</Label>
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <Label className="text-xs">Start (V)</Label>
                  <Input 
                    value={config.vStart}
                    onChange={(e) => setConfig({...config, vStart: e.target.value})}
                  />
                </div>
                <div>
                  <Label className="text-xs">Stop (V)</Label>
                  <Input 
                    value={config.vStop}
                    onChange={(e) => setConfig({...config, vStop: e.target.value})}
                  />
                </div>
                <div>
                  <Label className="text-xs">Step (V)</Label>
                  <Input 
                    value={config.vStep}
                    onChange={(e) => setConfig({...config, vStep: e.target.value})}
                  />
                </div>
              </div>
            </div>

            {/* Run Button */}
            <Button 
              onClick={handleRunMeasurement}
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
                  <Play className="mr-2 h-4 w-4" />
                  Start Measurement
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results Display */}
        <div className="lg:col-span-2 space-y-6">
          {/* C-V and Mott-Schottky Plots */}
          <Card>
            <CardHeader>
              <CardTitle>Analysis Plots</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="cv" className="w-full">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="cv">C-V Curve</TabsTrigger>
                  <TabsTrigger value="mott-schottky">Mott-Schottky</TabsTrigger>
                  <TabsTrigger value="doping">Doping Profile</TabsTrigger>
                </TabsList>
                
                <TabsContent value="cv">
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={cvData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="voltage" label={{ value: 'Voltage (V)', position: 'insideBottom', offset: -5 }} />
                      <YAxis label={{ value: 'Capacitance (nF)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Line type="monotone" dataKey="capacitance" stroke="#9333ea" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </TabsContent>
                
                <TabsContent value="mott-schottky">
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={mottSchottkyData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="voltage" label={{ value: 'Voltage (V)', position: 'insideBottom', offset: -5 }} />
                      <YAxis label={{ value: '1/C² (F⁻²)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip formatter={(value) => value.toExponential(2)} />
                      <Line type="monotone" dataKey="invC2" stroke="#9333ea" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </TabsContent>
                
                <TabsContent value="doping">
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={dopingProfileData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="depth" label={{ value: 'Depth (µm)', position: 'insideBottom', offset: -5 }} />
                      <YAxis 
                        scale="log"
                        domain={[1e15, 1e18]}
                        label={{ value: 'N (cm⁻³)', angle: -90, position: 'insideLeft' }} 
                      />
                      <Tooltip formatter={(value) => value.toExponential(2)} />
                      <Line type="monotone" dataKey="concentration" stroke="#9333ea" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          {/* Extracted Parameters */}
          {results && (
            <Card>
              <CardHeader>
                <CardTitle>Extracted Parameters</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {deviceType === 'mos' ? (
                    <>
                      <div className="p-4 bg-purple-50 rounded-lg">
                        <div className="text-sm text-gray-600">Cox</div>
                        <div className="text-2xl font-bold text-purple-600">
                          {results.cox.value} {results.cox.unit}
                        </div>
                      </div>
                      <div className="p-4 bg-blue-50 rounded-lg">
                        <div className="text-sm text-gray-600">tox</div>
                        <div className="text-2xl font-bold text-blue-600">
                          {results.tox.value} {results.tox.unit}
                        </div>
                      </div>
                      <div className="p-4 bg-green-50 rounded-lg">
                        <div className="text-sm text-gray-600">Vfb</div>
                        <div className="text-2xl font-bold text-green-600">
                          {results.vfb.value} {results.vfb.unit}
                        </div>
                      </div>
                      <div className="p-4 bg-yellow-50 rounded-lg">
                        <div className="text-sm text-gray-600">Vth</div>
                        <div className="text-2xl font-bold text-yellow-700">
                          {results.vth.value} {results.vth.unit}
                        </div>
                      </div>
                      <div className="p-4 bg-red-50 rounded-lg">
                        <div className="text-sm text-gray-600">Dit</div>
                        <div className="text-2xl font-bold text-red-600">
                          {results.dit.value.toExponential(1)} {results.dit.unit}
                        </div>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="p-4 bg-purple-50 rounded-lg">
                        <div className="text-sm text-gray-600">Vbi</div>
                        <div className="text-2xl font-bold text-purple-600">
                          {results.vbi.value} {results.vbi.unit}
                        </div>
                      </div>
                      <div className="p-4 bg-blue-50 rounded-lg">
                        <div className="text-sm text-gray-600">Doping</div>
                        <div className="text-2xl font-bold text-blue-600">
                          {results.dopingConcentration.value.toExponential(1)} {results.dopingConcentration.unit}
                        </div>
                      </div>
                      <div className="p-4 bg-green-50 rounded-lg">
                        <div className="text-sm text-gray-600">Depletion Width</div>
                        <div className="text-2xl font-bold text-green-600">
                          {results.depletionWidth.value} {results.depletionWidth.unit}
                        </div>
                      </div>
                      <div className="p-4 bg-yellow-50 rounded-lg">
                        <div className="text-sm text-gray-600">Barrier Height</div>
                        <div className="text-2xl font-bold text-yellow-700">
                          {results.barrierHeight.value} {results.barrierHeight.unit}
                        </div>
                      </div>
                    </>
                  )}
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="text-sm text-gray-600">Quality Score</div>
                    <div className="text-2xl font-bold text-gray-700">
                      {results.qualityScore}/100
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

// ==========================================
// 3. BJT Characterization Interface
// ==========================================

export const BJTCharacterization = () => {
  const [transistorType, setTransistorType] = useState('npn');
  const [measurementType, setMeasurementType] = useState('gummel');
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState(null);

  const [gummelData, setGummelData] = useState([]);
  const [outputData, setOutputData] = useState([]);

  const handleRunMeasurement = async () => {
    setIsRunning(true);

    setTimeout(() => {
      // Generate Gummel plot data
      const gummel = [];
      for (let vbe = 0; vbe <= 0.8; vbe += 0.01) {
        const ic = 1e-3 * Math.exp(vbe / 0.026);
        const ib = ic / 100; // Beta = 100
        
        gummel.push({
          vbe: vbe,
          ic: ic,
          ib: ib,
          logIc: Math.log10(Math.max(ic, 1e-15)),
          logIb: Math.log10(Math.max(ib, 1e-15))
        });
      }
      setGummelData(gummel);

      // Generate output characteristics
      const output = [];
      const ibValues = [10e-6, 20e-6, 30e-6, 40e-6];
      
      for (let ibIndex = 0; ibIndex < ibValues.length; ibIndex++) {
        const ib = ibValues[ibIndex];
        for (let vce = 0; vce <= 5; vce += 0.1) {
          const ic = 100 * ib * (1 + vce / 50); // Simple model with Early effect
          output.push({
            vce: vce,
            ic: ic,
            ib: `${(ib * 1e6).toFixed(0)}µA`
          });
        }
      }
      setOutputData(output);

      // Set results
      setResults({
        currentGain: {
          beta: 100,
          hfe: 98
        },
        earlyVoltage: {
          value: 50,
          unit: 'V'
        },
        idealityFactors: {
          collector: 1.02,
          base: 1.05
        },
        saturationCurrent: {
          is: 1e-14,
          unit: 'A'
        },
        qualityScore: 91
      });

      setIsRunning(false);
    }, 2000);
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Zap className="h-8 w-8 text-yellow-600" />
            BJT Characterization
          </h1>
          <p className="text-gray-600 mt-1">Bipolar junction transistor analysis</p>
        </div>
        <Badge variant="outline" className="text-lg px-4 py-2">
          {transistorType.toUpperCase()}
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Configuration Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Transistor Type */}
            <div>
              <Label>Transistor Type</Label>
              <Select value={transistorType} onValueChange={setTransistorType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="npn">NPN</SelectItem>
                  <SelectItem value="pnp">PNP</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Measurement Type */}
            <div>
              <Label>Measurement Type</Label>
              <Select value={measurementType} onValueChange={setMeasurementType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="gummel">Gummel Plot</SelectItem>
                  <SelectItem value="output">Output Characteristics</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Temperature */}
            <div>
              <Label>Temperature (K)</Label>
              <Input value="300" disabled />
            </div>

            {/* Run Button */}
            <Button 
              onClick={handleRunMeasurement}
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
                  <Play className="mr-2 h-4 w-4" />
                  Start Measurement
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results Display */}
        <div className="lg:col-span-2 space-y-6">
          {/* Plots */}
          <Card>
            <CardHeader>
              <CardTitle>
                {measurementType === 'gummel' ? 'Gummel Plot' : 'Output Characteristics'}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {measurementType === 'gummel' ? (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={gummelData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="vbe" label={{ value: 'Vbe (V)', position: 'insideBottom', offset: -5 }} />
                    <YAxis label={{ value: 'log(I)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="logIc" 
                      stroke="#eab308" 
                      strokeWidth={2} 
                      dot={false} 
                      name="log(Ic)"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="logIb" 
                      stroke="#f97316" 
                      strokeWidth={2} 
                      dot={false} 
                      name="log(Ib)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="vce" 
                      type="number"
                      domain={[0, 5]}
                      label={{ value: 'Vce (V)', position: 'insideBottom', offset: -5 }} 
                    />
                    <YAxis 
                      label={{ value: 'Ic (mA)', angle: -90, position: 'insideLeft' }} 
                    />
                    <Tooltip />
                    <Legend />
                    {['10µA', '20µA', '30µA', '40µA'].map((ib, index) => (
                      <Line
                        key={ib}
                        type="monotone"
                        data={outputData.filter(d => d.ib === ib)}
                        dataKey="ic"
                        stroke={['#eab308', '#f97316', '#ef4444', '#dc2626'][index]}
                        strokeWidth={2}
                        dot={false}
                        name={`Ib = ${ib}`}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>

          {/* Extracted Parameters */}
          {results && (
            <Card>
              <CardHeader>
                <CardTitle>Extracted Parameters</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div className="p-4 bg-yellow-50 rounded-lg">
                    <div className="text-sm text-gray-600">Current Gain (β)</div>
                    <div className="text-2xl font-bold text-yellow-600">
                      {results.currentGain.beta}
                    </div>
                  </div>
                  <div className="p-4 bg-orange-50 rounded-lg">
                    <div className="text-sm text-gray-600">Early Voltage</div>
                    <div className="text-2xl font-bold text-orange-600">
                      {results.earlyVoltage.value} {results.earlyVoltage.unit}
                    </div>
                  </div>
                  <div className="p-4 bg-red-50 rounded-lg">
                    <div className="text-sm text-gray-600">n_c / n_b</div>
                    <div className="text-2xl font-bold text-red-600">
                      {results.idealityFactors.collector} / {results.idealityFactors.base}
                    </div>
                  </div>
                  <div className="p-4 bg-purple-50 rounded-lg">
                    <div className="text-sm text-gray-600">Is</div>
                    <div className="text-2xl font-bold text-purple-600">
                      {results.saturationCurrent.is.toExponential(1)} {results.saturationCurrent.unit}
                    </div>
                  </div>
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="text-sm text-gray-600">Quality Score</div>
                    <div className="text-2xl font-bold text-gray-700">
                      {results.qualityScore}/100
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

// Export all components
export default {
  MOSFETCharacterization,
  CVProfiling,
  BJTCharacterization
};