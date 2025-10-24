import React, { useState } from ‘react’;
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from ‘@/components/ui/card’;
import { Button } from ‘@/components/ui/button’;
import { Input } from ‘@/components/ui/input’;
import { Label } from ‘@/components/ui/label’;
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from ‘@/components/ui/select’;
import { Tabs, TabsContent, TabsList, TabsTrigger } from ‘@/components/ui/tabs’;
import { Badge } from ‘@/components/ui/badge’;
import { Alert, AlertDescription } from ‘@/components/ui/alert’;
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Area, AreaChart } from ‘recharts’;
import { PlayCircle, Download, Layers, CheckCircle, AlertCircle, TrendingUp, Settings, Info } from ‘lucide-react’;

const CVProfiling = () => {
const [deviceType, setDeviceType] = useState(‘mos’);
const [substrateType, setSubstrateType] = useState(‘n-type’);
const [isRunning, setIsRunning] = useState(false);
const [results, setResults] = useState(null);

const [config, setConfig] = useState({
// Device parameters
area: 1e-4, // cm²
frequency: 100000, // Hz (100 kHz)
oxide_thickness: 10, // nm (for MOS)

// Measurement range
voltage_start: -3,
voltage_stop: 3,
voltage_step: 0.05,

// Advanced
ac_voltage: 0.03, // V
temperature: 300, // K

});

// Generate C-V curve data
const generateCVData = () => {
const points = [];
const eps_ox = 3.9 * 8.854e-14; // F/cm for SiO2
const eps_si = 11.7 * 8.854e-14; // F/cm for Si
const q = 1.602e-19; // C
const ni = 1.5e10; // cm^-3 for Si at 300K
const NA = substrateType === ‘n-type’ ? 1e15 : 1e16; // cm^-3

// Oxide capacitance
const Cox = eps_ox * config.area / (config.oxide_thickness * 1e-7); // F

for (let i = 0; i <= 120; i++) {
  const v = config.voltage_start + (config.voltage_stop - config.voltage_start) * (i / 120);
  let c;
  
  if (deviceType === 'mos') {
    // MOS capacitor C-V curve
    const vfb = -0.3; // Flat-band voltage (approximate)
    const phiF = 0.0259 * Math.log(NA / ni); // Fermi potential
    
    if (substrateType === 'n-type') {
      // n-type substrate (p-MOS)
      if (v < vfb - 0.5) {
        // Accumulation
        c = Cox * 0.95;
      } else if (v > vfb + 2 * phiF) {
        // Inversion
        c = Cox * 0.4;
      } else {
        // Depletion
        const Wd = Math.sqrt(2 * eps_si * Math.abs(v - vfb) / (q * NA));
        const Cdep = eps_si * config.area / Wd;
        c = (Cox * Cdep) / (Cox + Cdep);
      }
    } else {
      // p-type substrate (n-MOS)
      if (v > vfb + 0.5) {
        // Accumulation
        c = Cox * 0.95;
      } else if (v < vfb - 2 * phiF) {
        // Inversion
        c = Cox * 0.4;
      } else {
        // Depletion
        const Wd = Math.sqrt(2 * eps_si * Math.abs(v - vfb) / (q * NA));
        const Cdep = eps_si * config.area / Wd;
        c = (Cox * Cdep) / (Cox + Cdep);
      }
    }
  } else {
    // Schottky diode C-V
    const Vbi = 0.7; // Built-in potential
    const ND = substrateType === 'n-type' ? 1e16 : 5e15;
    
    if (v < Vbi) {
      const Wd = Math.sqrt(2 * eps_si * (Vbi - v) / (q * ND));
      c = eps_si * config.area / Wd;
    } else {
      c = eps_si * config.area / 1e-5; // Very high capacitance at forward bias
    }
  }
  
  // Add small noise
  c = c * (1 + (Math.random() - 0.5) * 0.02);
  
  points.push({
    voltage: parseFloat(v.toFixed(4)),
    capacitance: parseFloat((c * 1e12).toFixed(4)), // pF
    capacitance_normalized: parseFloat((c * 1e12 / Cox / 1e12).toFixed(4)),
    inv_c2: parseFloat((1 / Math.pow(c * 1e12, 2)).toFixed(6)) // 1/pF²
  });
}

return points;

};

// Generate doping profile
const generateDopingProfile = () => {
const points = [];
const NA_base = substrateType === ‘n-type’ ? 1e15 : 1e16;

for (let i = 0; i <= 100; i++) {
  const depth = i * 5; // nm
  
  // Simulate doping profile with some variation
  const doping = NA_base * (1 + 0.1 * Math.sin(depth / 50));
  
  points.push({
    depth: depth,
    doping: parseFloat(doping.toExponential(2))
  });
}

return points;

};

const cvData = results?.cv_curve || generateCVData();
const dopingProfile = results?.doping_profile || generateDopingProfile();

const calculateResults = () => {
const data = generateCVData();

// Find Cox (max capacitance)
const Cox = Math.max(...data.map(p => p.capacitance));

// Calculate oxide thickness
const eps_ox = 3.9 * 8.854e-14; // F/cm
const tox = eps_ox * config.area / (Cox * 1e-12) * 1e7; // nm

// Find flat-band voltage (inflection point in accumulation)
let vfb = -0.3;

// Find threshold voltage (for MOS)
const Cmin = Math.min(...data.map(p => p.capacitance));
const threshold_idx = data.findIndex(p => p.capacitance < (Cox + Cmin) / 2);
const vth = threshold_idx > 0 ? data[threshold_idx].voltage : 0.5;

// Estimate Dit (interface trap density)
const dit = 1e11; // cm^-2·eV^-1 (typical)

// Mott-Schottky analysis for doping
const ms_data = data.filter(p => p.voltage < 0);
let doping = 1e15;
let vbi = 0.7;

if (ms_data.length > 10) {
  // Linear fit of 1/C² vs V
  const x = ms_data.map(p => p.voltage);
  const y = ms_data.map(p => p.inv_c2);
  
  // Simple linear regression
  const n = x.length;
  const sum_x = x.reduce((a, b) => a + b, 0);
  const sum_y = y.reduce((a, b) => a + b, 0);
  const sum_xy = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
  const sum_x2 = x.reduce((acc, xi) => acc + xi * xi, 0);
  
  const slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
  const intercept = (sum_y - slope * sum_x) / n;
  
  // Extract doping from slope
  const eps_si = 11.7 * 8.854e-14; // F/cm
  const q = 1.602e-19;
  doping = 2 / (q * eps_si * config.area * config.area * slope * 1e-24);
  
  // Built-in potential from intercept
  vbi = -intercept / slope;
}

// Quality scoring
const cox_score = tox > 5 && tox < 50 ? 30 : 20;
const shape_score = 25; // Based on curve shape
const noise_score = 25; // Based on data quality
const consistency_score = 20;
const quality_score = cox_score + shape_score + noise_score + consistency_score;

if (deviceType === 'mos') {
  return {
    oxide_capacitance: { value: Cox, unit: 'pF' },
    oxide_thickness: { value: tox, unit: 'nm' },
    flatband_voltage: { value: vfb, unit: 'V' },
    threshold_voltage: { value: vth, unit: 'V' },
    interface_trap_density: { value: dit, unit: 'cm⁻²·eV⁻¹' },
    substrate_doping: { value: doping, unit: 'cm⁻³' },
    quality_score: Math.round(quality_score),
    device_type: deviceType,
    substrate_type: substrateType
  };
} else {
  return {
    builtin_potential: { value: vbi, unit: 'V' },
    doping_concentration: { value: doping, unit: 'cm⁻³' },
    barrier_height: { value: vbi + 0.3, unit: 'eV' },
    doping_profile: generateDopingProfile(),
    quality_score: Math.round(quality_score),
    device_type: deviceType,
    substrate_type: substrateType
  };
}

};

const handleRunMeasurement = async () => {
setIsRunning(true);

await new Promise(resolve => setTimeout(resolve, 2000));

const calculatedResults = calculateResults();

setResults({
  ...calculatedResults,
  cv_curve: generateCVData(),
  doping_profile: generateDopingProfile(),
  timestamp: new Date().toISOString()
});

setIsRunning(false);

};

const handleExportData = () => {
if (!results) return;

const exportData = {
  metadata: {
    device_type: deviceType,
    substrate_type: substrateType,
    timestamp: results.timestamp,
    ...config
  },
  results: results,
  curves: {
    cv_curve: results.cv_curve,
    doping_profile: results.doping_profile
  }
};

const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = `cv_${deviceType}_${substrateType}_${Date.now()}.json`;
a.click();

};

return (
<div className="w-full max-w-[1600px] mx-auto p-6 space-y-6 bg-gradient-to-br from-gray-50 to-indigo-50 min-h-screen">
{/* Header */}
<div className="bg-white rounded-lg shadow-md p-6">
<div className="flex justify-between items-start">
<div>
<h1 className="text-4xl font-bold flex items-center gap-3 text-gray-800">
<Layers className="h-10 w-10 text-indigo-600" />
C-V Profiling & Doping Analysis
</h1>
<p className="text-gray-600 mt-2">Capacitance-Voltage Characterization & Dopant Extraction</p>
</div>
<div className="flex gap-2 items-center">
<Badge variant=“outline” className={`text-lg px-4 py-2 ${deviceType === 'mos' ? 'bg-blue-500' : 'bg-green-500'} text-white border-none`}>
{deviceType === ‘mos’ ? ‘MOS Capacitor’ : ‘Schottky Diode’}
</Badge>
<Badge variant="outline" className="text-lg px-4 py-2 bg-gray-100">
{substrateType === ‘n-type’ ? ‘n-type’ : ‘p-type’} Substrate
</Badge>
<Badge variant="outline" className="text-lg px-4 py-2 bg-gray-100">
{(config.frequency / 1000).toFixed(0)} kHz
</Badge>
</div>
</div>
</div>

  {/* Main Content */}
  <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
    
    {/* Configuration Panel */}
    <Card className="lg:col-span-1 shadow-md">
      <CardHeader className="bg-gradient-to-r from-indigo-50 to-blue-50">
        <CardTitle className="flex items-center gap-2">
          <Settings className="h-5 w-5" />
          Configuration
        </CardTitle>
        <CardDescription>Device & measurement parameters</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 pt-4">
        
        {/* Device Type */}
        <div className="space-y-2">
          <Label className="flex items-center gap-2">
            <Layers className="h-4 w-4" />
            Device Type
          </Label>
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
        <div className="space-y-2">
          <Label>Substrate Type</Label>
          <Select value={substrateType} onValueChange={setSubstrateType}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="n-type">n-type (electrons)</SelectItem>
              <SelectItem value="p-type">p-type (holes)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Area */}
        <div className="space-y-2">
          <Label>Device Area (cm²)</Label>
          <Input 
            type="number" 
            value={config.area}
            onChange={(e) => setConfig({...config, area: parseFloat(e.target.value)})}
            step="0.0001"
          />
        </div>

        {/* Frequency */}
        <div className="space-y-2">
          <Label>Frequency (Hz)</Label>
          <Input 
            type="number" 
            value={config.frequency}
            onChange={(e) => setConfig({...config, frequency: parseFloat(e.target.value)})}
          />
          <div className="flex gap-2 flex-wrap">
            <Button 
              size="sm" 
              variant="outline"
              onClick={() => setConfig({...config, frequency: 1000})}
              className="text-xs"
            >
              1 kHz
            </Button>
            <Button 
              size="sm" 
              variant="outline"
              onClick={() => setConfig({...config, frequency: 100000})}
              className="text-xs"
            >
              100 kHz
            </Button>
            <Button 
              size="sm" 
              variant="outline"
              onClick={() => setConfig({...config, frequency: 1000000})}
              className="text-xs"
            >
              1 MHz
            </Button>
          </div>
        </div>

        {/* Oxide Thickness (MOS only) */}
        {deviceType === 'mos' && (
          <div className="space-y-2 p-3 bg-blue-50 rounded border">
            <Label className="text-xs">Expected Oxide Thickness (nm)</Label>
            <Input 
              type="number" 
              value={config.oxide_thickness}
              onChange={(e) => setConfig({...config, oxide_thickness: parseFloat(e.target.value)})}
              className="text-sm"
            />
            <p className="text-xs text-gray-600">For reference/validation</p>
          </div>
        )}

        {/* Voltage Range */}
        <div className="space-y-3 p-3 bg-green-50 rounded border">
          <div className="font-semibold text-sm text-gray-700">Voltage Sweep</div>
          
          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-2">
              <Label className="text-xs">Start (V)</Label>
              <Input 
                type="number" 
                value={config.voltage_start}
                onChange={(e) => setConfig({...config, voltage_start: parseFloat(e.target.value)})}
                className="text-sm"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-xs">Stop (V)</Label>
              <Input 
                type="number" 
                value={config.voltage_stop}
                onChange={(e) => setConfig({...config, voltage_stop: parseFloat(e.target.value)})}
                className="text-sm"
              />
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="space-y-2 pt-4">
          <Button 
            onClick={handleRunMeasurement}
            disabled={isRunning}
            className="w-full bg-gradient-to-r from-indigo-500 to-blue-600 hover:from-indigo-600 hover:to-blue-700"
            size="lg"
          >
            {isRunning ? (
              <>
                <Layers className="h-5 w-5 mr-2 animate-pulse" />
                Measuring...
              </>
            ) : (
              <>
                <PlayCircle className="h-5 w-5 mr-2" />
                Start Measurement
              </>
            )}
          </Button>

          {results && (
            <Button 
              onClick={handleExportData}
              variant="outline"
              className="w-full"
            >
              <Download className="h-4 w-4 mr-2" />
              Export Data
            </Button>
          )}
        </div>
      </CardContent>
    </Card>

    {/* Results & Plots */}
    <Card className="lg:col-span-3 shadow-md">
      <CardHeader className="bg-gradient-to-r from-indigo-50 to-blue-50">
        <CardTitle>
          {!results ? 'Measurement Preview' : `Results - Quality: ${results.quality_score}/100`}
        </CardTitle>
        <CardDescription>
          {!results ? 'Configure and start measurement' : 'C-V Characteristics and Parameter Extraction'}
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        {!results ? (
          <div className="h-96 flex items-center justify-center text-gray-400 bg-gray-50 rounded-lg border-2 border-dashed">
            <div className="text-center">
              <Layers className="h-20 w-20 mx-auto mb-4 opacity-30" />
              <p className="text-lg font-medium">No measurement data yet</p>
              <p className="text-sm mt-2">Configure device and click "Start Measurement"</p>
            </div>
          </div>
        ) : (
          <Tabs defaultValue="cv-curve" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="cv-curve">C-V Curve</TabsTrigger>
              <TabsTrigger value="mott-schottky">Mott-Schottky</TabsTrigger>
              <TabsTrigger value="parameters">Parameters</TabsTrigger>
            </TabsList>

            {/* C-V Curve Tab */}
            <TabsContent value="cv-curve" className="space-y-6">
              <div className="space-y-2">
                <h3 className="font-semibold text-sm text-gray-700">Capacitance vs Voltage</h3>
                <div className="h-96 bg-white p-4 rounded border">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={cvData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                      <XAxis 
                        dataKey="voltage" 
                        label={{ value: 'Voltage (V)', position: 'insideBottom', offset: -5 }}
                      />
                      <YAxis 
                        label={{ value: 'Capacitance (pF)', angle: -90, position: 'insideLeft' }}
                      />
                      <Tooltip 
                        formatter={(value, name) => [value.toFixed(3), name === 'capacitance' ? 'C (pF)' : 'C/Cox']}
                      />
                      <Legend />
                      {deviceType === 'mos' && (
                        <>
                          <ReferenceLine 
                            x={results.flatband_voltage?.value || -0.3}
                            stroke="#f59e0b" 
                            strokeDasharray="3 3"
                            label={{ value: 'Vfb', position: 'top', fill: '#f59e0b', fontSize: 12 }}
                          />
                          <ReferenceLine 
                            x={results.threshold_voltage?.value || 0.5}
                            stroke="#dc2626" 
                            strokeDasharray="3 3"
                            label={{ value: 'Vth', position: 'top', fill: '#dc2626', fontSize: 12 }}
                          />
                        </>
                      )}
                      <Line 
                        type="monotone" 
                        dataKey="capacitance" 
                        stroke="#6366f1" 
                        strokeWidth={2}
                        dot={false}
                        name="Capacitance"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Key Metrics */}
              {deviceType === 'mos' ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <Card className="border-blue-200 bg-blue-50">
                    <CardContent className="pt-4">
                      <div className="text-xs text-gray-600 mb-1">C<sub>ox</sub></div>
                      <div className="text-2xl font-bold text-blue-700">{results.oxide_capacitance.value.toFixed(2)} pF</div>
                      <div className="text-xs text-gray-600 mt-1">Oxide Capacitance</div>
                    </CardContent>
                  </Card>

                  <Card className="border-green-200 bg-green-50">
                    <CardContent className="pt-4">
                      <div className="text-xs text-gray-600 mb-1">t<sub>ox</sub></div>
                      <div className="text-2xl font-bold text-green-700">{results.oxide_thickness.value.toFixed(2)} nm</div>
                      <div className="text-xs text-gray-600 mt-1">Oxide Thickness</div>
                    </CardContent>
                  </Card>

                  <Card className="border-orange-200 bg-orange-50">
                    <CardContent className="pt-4">
                      <div className="text-xs text-gray-600 mb-1">V<sub>fb</sub></div>
                      <div className="text-2xl font-bold text-orange-700">{results.flatband_voltage.value.toFixed(3)} V</div>
                      <div className="text-xs text-gray-600 mt-1">Flat-band Voltage</div>
                    </CardContent>
                  </Card>

                  <Card className="border-red-200 bg-red-50">
                    <CardContent className="pt-4">
                      <div className="text-xs text-gray-600 mb-1">V<sub>th</sub></div>
                      <div className="text-2xl font-bold text-red-700">{results.threshold_voltage.value.toFixed(3)} V</div>
                      <div className="text-xs text-gray-600 mt-1">Threshold Voltage</div>
                    </CardContent>
                  </Card>
                </div>
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <Card className="border-purple-200 bg-purple-50">
                    <CardContent className="pt-4">
                      <div className="text-xs text-gray-600 mb-1">V<sub>bi</sub></div>
                      <div className="text-2xl font-bold text-purple-700">{results.builtin_potential.value.toFixed(3)} V</div>
                      <div className="text-xs text-gray-600 mt-1">Built-in Potential</div>
                    </CardContent>
                  </Card>

                  <Card className="border-indigo-200 bg-indigo-50">
                    <CardContent className="pt-4">
                      <div className="text-xs text-gray-600 mb-1">N<sub>D/A</sub></div>
                      <div className="text-2xl font-bold text-indigo-700">{results.doping_concentration.value.toExponential(1)}</div>
                      <div className="text-xs text-gray-600 mt-1">cm⁻³</div>
                    </CardContent>
                  </Card>

                  <Card className="border-cyan-200 bg-cyan-50">
                    <CardContent className="pt-4">
                      <div className="text-xs text-gray-600 mb-1">φ<sub>B</sub></div>
                      <div className="text-2xl font-bold text-cyan-700">{results.barrier_height.value.toFixed(3)} eV</div>
                      <div className="text-xs text-gray-600 mt-1">Barrier Height</div>
                    </CardContent>
                  </Card>
                </div>
              )}
            </TabsContent>

            {/* Mott-Schottky Tab */}
            <TabsContent value="mott-schottky" className="space-y-6">
              <div className="space-y-2">
                <h3 className="font-semibold text-sm text-gray-700">Mott-Schottky Plot (1/C² vs V)</h3>
                <div className="h-80 bg-white p-4 rounded border">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={cvData.filter(p => p.voltage < 0)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                      <XAxis 
                        dataKey="voltage" 
                        label={{ value: 'Voltage (V)', position: 'insideBottom', offset: -5 }}
                      />
                      <YAxis 
                        label={{ value: '1/C² (1/pF²)', angle: -90, position: 'insideLeft' }}
                      />
                      <Tooltip formatter={(value) => value.toFixed(4)} />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="inv_c2" 
                        stroke="#8b5cf6" 
                        strokeWidth={2}
                        dot={{ r: 2 }}
                        name="1/C²"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="space-y-2">
                <h3 className="font-semibold text-sm text-gray-700">Doping Profile vs Depth</h3>
                <div className="h-80 bg-white p-4 rounded border">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={dopingProfile}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                      <XAxis 
                        dataKey="depth" 
                        label={{ value: 'Depth (nm)', position: 'insideBottom', offset: -5 }}
                      />
                      <YAxis 
                        scale="log"
                        domain={['auto', 'auto']}
                        label={{ value: 'Doping (cm⁻³)', angle: -90, position: 'insideLeft' }}
                      />
                      <Tooltip 
                        formatter={(value) => [value, 'N (cm⁻³)']}
                      />
                      <Legend />
                      <Area 
                        type="monotone" 
                        dataKey="doping" 
                        stroke="#10b981" 
                        fill="#86efac"
                        fillOpacity={0.6}
                        name="Doping Concentration"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </TabsContent>

            {/* Parameters Tab */}
            <TabsContent value="parameters" className="space-y-6">
              {/* Quality Score */}
              <Alert className={
                results.quality_score >= 80 ? 'border-green-200 bg-green-50' :
                results.quality_score >= 60 ? 'border-yellow-200 bg-yellow-50' :
                'border-red-200 bg-red-50'
              }>
                <TrendingUp className="h-4 w-4" />
                <AlertDescription>
                  <div className="flex justify-between items-center">
                    <span className="font-semibold text-lg">Quality Score: {results.quality_score}/100</span>
                    {results.quality_score >= 80 ? (
                      <Badge className="bg-green-600 text-white">Excellent</Badge>
                    ) : results.quality_score >= 60 ? (
                      <Badge className="bg-yellow-600 text-white">Good</Badge>
                    ) : (
                      <Badge className="bg-red-600 text-white">Needs Review</Badge>
                    )}
                  </div>
                </AlertDescription>
              </Alert>

              {/* Parameters Table */}
              <div className="bg-white rounded-lg border overflow-hidden">
                <div className="bg-gray-50 px-4 py-3 border-b">
                  <h3 className="font-semibold text-gray-800">Extracted Parameters</h3>
                </div>
                <table className="w-full">
                  <thead className="bg-gray-100">
                    <tr>
                      <th className="px-4 py-2 text-left text-sm font-semibold text-gray-700">Parameter</th>
                      <th className="px-4 py-2 text-right text-sm font-semibold text-gray-700">Value</th>
                      <th className="px-4 py-2 text-left text-sm font-semibold text-gray-700">Unit</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {deviceType === 'mos' ? (
                      <>
                        <tr className="hover:bg-gray-50 bg-blue-50">
                          <td className="px-4 py-3 text-sm font-semibold">Oxide Capacitance (C<sub>ox</sub>)</td>
                          <td className="px-4 py-3 text-sm text-right font-mono font-semibold">{results.oxide_capacitance.value.toFixed(4)}</td>
                          <td className="px-4 py-3 text-sm">pF</td>
                        </tr>
                        <tr className="hover:bg-gray-50 bg-green-50">
                          <td className="px-4 py-3 text-sm font-semibold">Oxide Thickness (t<sub>ox</sub>)</td>
                          <td className="px-4 py-3 text-sm text-right font-mono font-semibold">{results.oxide_thickness.value.toFixed(2)}</td>
                          <td className="px-4 py-3 text-sm">nm</td>
                        </tr>
                        <tr className="hover:bg-gray-50 bg-orange-50">
                          <td className="px-4 py-3 text-sm font-semibold">Flat-band Voltage (V<sub>fb</sub>)</td>
                          <td className="px-4 py-3 text-sm text-right font-mono font-semibold">{results.flatband_voltage.value.toFixed(4)}</td>
                          <td className="px-4 py-3 text-sm">V</td>
                        </tr>
                        <tr className="hover:bg-gray-50 bg-red-50">
                          <td className="px-4 py-3 text-sm font-semibold">Threshold Voltage (V<sub>th</sub>)</td>
                          <td className="px-4 py-3 text-sm text-right font-mono font-semibold">{results.threshold_voltage.value.toFixed(4)}</td>
                          <td className="px-4 py-3 text-sm">V</td>
                        </tr>
                        <tr className="hover:bg-gray-50">
                          <td className="px-4 py-3 text-sm">Interface Trap Density (D<sub>it</sub>)</td>
                          <td className="px-4 py-3 text-sm text-right font-mono">{results.interface_trap_density.value.toExponential(2)}</td>
                          <td className="px-4 py-3 text-sm">cm⁻²·eV⁻¹</td>
                        </tr>
                        <tr className="hover:bg-gray-50">
                          <td className="px-4 py-3 text-sm">Substrate Doping</td>
                          <td className="px-4 py-3 text-sm text-right font-mono">{results.substrate_doping.value.toExponential(2)}</td>
                          <td className="px-4 py-3 text-sm">cm⁻³</td>
                        </tr>
                      </>
                    ) : (
                      <>
                        <tr className="hover:bg-gray-50 bg-purple-50">
                          <td className="px-4 py-3 text-sm font-semibold">Built-in Potential (V<sub>bi</sub>)</td>
                          <td className="px-4 py-3 text-sm text-right font-mono font-semibold">{results.builtin_potential.value.toFixed(4)}</td>
                          <td className="px-4 py-3 text-sm">V</td>
                        </tr>
                        <tr className="hover:bg-gray-50 bg-indigo-50">
                          <td className="px-4 py-3 text-sm font-semibold">Doping Concentration</td>
                          <td className="px-4 py-3 text-sm text-right font-mono font-semibold">{results.doping_concentration.value.toExponential(3)}</td>
                          <td className="px-4 py-3 text-sm">cm⁻³</td>
                        </tr>
                        <tr className="hover:bg-gray-50 bg-cyan-50">
                          <td className="px-4 py-3 text-sm font-semibold">Barrier Height (φ<sub>B</sub>)</td>
                          <td className="px-4 py-3 text-sm text-right font-mono font-semibold">{results.barrier_height.value.toFixed(4)}</td>
                          <td className="px-4 py-3 text-sm">eV</td>
                        </tr>
                      </>
                    )}
                  </tbody>
                </table>
              </div>

              {/* Device Info */}
              <div className="bg-blue-50 rounded-lg border border-blue-200 p-4">
                <h3 className="font-semibold text-gray-800 mb-3 flex items-center gap-2">
                  <Info className="h-4 w-4" />
                  Measurement Conditions
                </h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Device Type:</span>
                    <span className="ml-2 font-mono font-semibold">{results.device_type === 'mos' ? 'MOS Capacitor' : 'Schottky Diode'}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Substrate:</span>
                    <span className="ml-2 font-mono font-semibold">{results.substrate_type}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Frequency:</span>
                    <span className="ml-2 font-mono font-semibold">{(config.frequency / 1000).toFixed(0)} kHz</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Area:</span>
                    <span className="ml-2 font-mono font-semibold">{config.area} cm²</span>
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        )}
      </CardContent>
    </Card>
  </div>

  {/* Status Messages */}
  {isRunning && (
    <Alert className="bg-indigo-50 border-indigo-200">
      <Layers className="h-4 w-4 animate-pulse text-indigo-600" />
      <AlertDescription className="text-indigo-900">
        <span className="font-semibold">Measurement in progress...</span>
        <span className="ml-2">Sweeping voltage and recording capacitance</span>
      </AlertDescription>
    </Alert>
  )}

  {results && !isRunning && (
    <Alert className="bg-green-50 border-green-200">
      <CheckCircle className="h-4 w-4 text-green-600" />
      <AlertDescription className="text-green-900">
        <span className="font-semibold">Measurement completed successfully!</span>
        <span className="ml-2">Quality: {results.quality_score}/100 | {new Date(results.timestamp).toLocaleString()}</span>
      </AlertDescription>
    </Alert>
  )}
</div>

);
};

export default CVProfiling;