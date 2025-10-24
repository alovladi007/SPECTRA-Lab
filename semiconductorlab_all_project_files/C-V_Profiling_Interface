'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, ScatterChart, Scatter } from 'recharts';
import { Play, Download, AlertCircle, CheckCircle2, TrendingUp, Info } from 'lucide-react';

/**
 * C-V Profiling Interface
 * 
 * Production-ready UI for Capacitance-Voltage profiling
 * - MOS Capacitor analysis (Cox, tox, Vfb, Vth, Dit)
 * - Schottky Diode analysis (Vbi, doping profile)
 * - Mott-Schottky plots
 * - Doping concentration profiles
 */

interface CVConfig {
  deviceType: 'mos' | 'schottky';
  substrateType: 'n-type' | 'p-type';
  frequency: number; // Hz
  area: number; // cm²
  voltage_start: number;
  voltage_stop: number;
  voltage_step: number;
}

interface CVResults {
  // MOS Capacitor Results
  oxide_capacitance?: number;
  oxide_thickness?: number;
  flatband_voltage?: number;
  threshold_voltage?: number;
  interface_trap_density?: number;
  substrate_doping?: number;
  
  // Schottky Diode Results
  builtin_voltage?: number;
  barrier_height?: number;
  doping_concentration?: number;
  
  // Common
  quality_score: number;
  cv_data?: Array<{ voltage: number; capacitance: number; inv_c_squared?: number }>;
  doping_profile?: Array<{ depth: number; concentration: number }>;
}

const CVProfiling: React.FC = () => {
  const [config, setConfig] = useState<CVConfig>({
    deviceType: 'mos',
    substrateType: 'n-type',
    frequency: 100000, // 100 kHz
    area: 1e-4, // 0.01 cm²
    voltage_start: -3,
    voltage_stop: 3,
    voltage_step: 0.05
  });

  const [results, setResults] = useState<CVResults | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    setError(null);

    try {
      const endpoint = config.deviceType === 'mos'
        ? '/api/v1/electrical/cv-profiling/analyze-mos'
        : '/api/v1/electrical/cv-profiling/analyze-schottky';

      const testData = generateTestData(config);

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...testData,
          frequency: config.frequency,
          area: config.area,
          substrate_type: config.substrateType
        })
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const generateTestData = (cfg: CVConfig) => {
    const voltage_array = [];
    const capacitance_array = [];
    
    // Physical constants
    const eps0 = 8.854e-14; // F/cm
    const eps_si = 11.7;
    const eps_ox = 3.9;
    const q = 1.602e-19; // C
    const k = 1.381e-23; // J/K
    const T = 300; // K
    
    if (cfg.deviceType === 'mos') {
      // MOS capacitor C-V simulation
      const tox = 10e-7; // 10 nm
      const Cox = eps0 * eps_ox / tox * cfg.area;
      const NA = cfg.substrateType === 'n-type' ? 1e15 : 5e15; // cm^-3
      const Vfb = cfg.substrateType === 'n-type' ? -0.9 : -1.1;
      
      for (let V = cfg.voltage_start; V <= cfg.voltage_stop; V += cfg.voltage_step) {
        voltage_array.push(V);
        
        // Simplified MOS C-V (high frequency)
        const Vg = V - Vfb;
        const phi_s = Vg; // Simplified
        
        if (Vg < -0.5) {
          // Accumulation
          capacitance_array.push(Cox);
        } else if (Vg < 0.8) {
          // Depletion
          const Cdep = Math.sqrt(eps0 * eps_si * q * NA / (2 * Math.abs(phi_s) + 0.1));
          const Ctotal = 1 / (1/Cox + 1/Cdep);
          capacitance_array.push(Ctotal);
        } else {
          // Inversion
          const Cmin = Cox * 0.3;
          capacitance_array.push(Cmin);
        }
      }
    } else {
      // Schottky diode C-V simulation
      const ND = cfg.substrateType === 'n-type' ? 1e16 : 5e15;
      const Vbi = 0.8;
      
      for (let V = cfg.voltage_start; V <= cfg.voltage_stop; V += cfg.voltage_step) {
        voltage_array.push(V);
        
        if (V < Vbi) {
          const W = Math.sqrt(2 * eps0 * eps_si * (Vbi - V) / (q * ND));
          const C = eps0 * eps_si * cfg.area / W;
          capacitance_array.push(C);
        } else {
          capacitance_array.push(0);
        }
      }
    }
    
    return {
      voltage: voltage_array,
      capacitance: capacitance_array
    };
  };

  const getQualityColor = (score: number): string => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getQualityBadge = (score: number): string => {
    if (score >= 80) return 'bg-green-100 text-green-800';
    if (score >= 60) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const getQualityLabel = (score: number): string => {
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 40) return 'Fair';
    return 'Poor';
  };

  const handleExport = () => {
    const dataStr = JSON.stringify(results, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cv_profiling_${config.deviceType}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold mb-2">C-V Profiling & Doping Analysis</h1>
        <p className="text-gray-600">
          Capacitance-voltage characterization for MOS capacitors and Schottky diodes
        </p>
      </div>

      {/* Configuration Panel */}
      <Card>
        <CardHeader>
          <CardTitle>Measurement Configuration</CardTitle>
          <CardDescription>
            Configure device parameters and measurement settings
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* Device Type */}
            <div className="space-y-2">
              <Label htmlFor="deviceType">Device Type</Label>
              <Select
                value={config.deviceType}
                onValueChange={(value) => setConfig({ ...config, deviceType: value as 'mos' | 'schottky' })}
              >
                <SelectTrigger id="deviceType">
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
              <Label htmlFor="substrateType">Substrate Type</Label>
              <Select
                value={config.substrateType}
                onValueChange={(value) => setConfig({ ...config, substrateType: value as 'n-type' | 'p-type' })}
              >
                <SelectTrigger id="substrateType">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="n-type">n-type</SelectItem>
                  <SelectItem value="p-type">p-type</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Frequency */}
            <div className="space-y-2">
              <Label htmlFor="frequency">Measurement Frequency</Label>
              <Select
                value={config.frequency.toString()}
                onValueChange={(value) => setConfig({ ...config, frequency: parseFloat(value) })}
              >
                <SelectTrigger id="frequency">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1000">1 kHz</SelectItem>
                  <SelectItem value="10000">10 kHz</SelectItem>
                  <SelectItem value="100000">100 kHz (Recommended)</SelectItem>
                  <SelectItem value="1000000">1 MHz</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Area */}
            <div className="space-y-2">
              <Label htmlFor="area">Device Area (cm²)</Label>
              <Input
                id="area"
                type="number"
                value={config.area}
                onChange={(e) => setConfig({ ...config, area: parseFloat(e.target.value) })}
                step="0.0001"
                min="0.0001"
              />
            </div>

            {/* Voltage Start */}
            <div className="space-y-2">
              <Label htmlFor="v_start">Voltage Start (V)</Label>
              <Input
                id="v_start"
                type="number"
                value={config.voltage_start}
                onChange={(e) => setConfig({ ...config, voltage_start: parseFloat(e.target.value) })}
                step="0.1"
              />
            </div>

            {/* Voltage Stop */}
            <div className="space-y-2">
              <Label htmlFor="v_stop">Voltage Stop (V)</Label>
              <Input
                id="v_stop"
                type="number"
                value={config.voltage_stop}
                onChange={(e) => setConfig({ ...config, voltage_stop: parseFloat(e.target.value) })}
                step="0.1"
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3 pt-4">
            <Button
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className="flex items-center gap-2"
              size="lg"
            >
              <Play className="w-4 h-4" />
              {isAnalyzing ? 'Analyzing...' : 'Run Analysis'}
            </Button>
            {results && (
              <Button
                variant="outline"
                onClick={handleExport}
                className="flex items-center gap-2"
                size="lg"
              >
                <Download className="w-4 h-4" />
                Export Data
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="w-4 h-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Results Display */}
      {results && (
        <>
          {/* Quality Score */}
          <Card className="border-2">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Measurement Quality Score</p>
                  <div className="flex items-center gap-3">
                    <span className={`text-5xl font-bold ${getQualityColor(results.quality_score)}`}>
                      {results.quality_score}
                    </span>
                    <div>
                      <Badge className={getQualityBadge(results.quality_score)}>
                        {getQualityLabel(results.quality_score)}
                      </Badge>
                      <p className="text-xs text-gray-500 mt-1">Out of 100</p>
                    </div>
                  </div>
                </div>
                <CheckCircle2 className={`w-20 h-20 ${getQualityColor(results.quality_score)}`} />
              </div>
            </CardContent>
          </Card>

          {/* Results Tabs */}
          <Tabs defaultValue="cv-curves" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="cv-curves">C-V Curves</TabsTrigger>
              <TabsTrigger value="parameters">Extracted Parameters</TabsTrigger>
              <TabsTrigger value="doping">Doping Profile</TabsTrigger>
            </TabsList>

            {/* C-V Curves Tab */}
            <TabsContent value="cv-curves" className="space-y-4">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* C-V Plot */}
                <Card>
                  <CardHeader>
                    <CardTitle>Capacitance vs. Voltage</CardTitle>
                    <CardDescription>
                      {config.deviceType === 'mos'
                        ? 'High-frequency C-V characteristic showing accumulation, depletion, and inversion'
                        : 'Reverse-bias C-V showing depletion width variation'}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {results.cv_data && (
                      <ResponsiveContainer width="100%" height={350}>
                        <LineChart data={results.cv_data}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis
                            dataKey="voltage"
                            label={{ value: 'Voltage (V)', position: 'insideBottom', offset: -5 }}
                          />
                          <YAxis
                            label={{ value: 'Capacitance (F)', angle: -90, position: 'insideLeft' }}
                            tickFormatter={(value) => value.toExponential(1)}
                          />
                          <Tooltip
                            formatter={(value: any) => [(typeof value === 'number' ? value.toExponential(3) : value), 'Capacitance']}
                          />
                          <Legend />
                          <Line
                            type="monotone"
                            dataKey="capacitance"
                            stroke="#3b82f6"
                            strokeWidth={2}
                            dot={false}
                            name="Capacitance"
                          />
                          {config.deviceType === 'mos' && results.flatband_voltage && (
                            <ReferenceLine
                              x={results.flatband_voltage}
                              stroke="#ef4444"
                              strokeDasharray="5 5"
                              label={{ value: 'Vfb', position: 'top' }}
                            />
                          )}
                        </LineChart>
                      </ResponsiveContainer>
                    )}
                  </CardContent>
                </Card>

                {/* Mott-Schottky Plot */}
                <Card>
                  <CardHeader>
                    <CardTitle>Mott-Schottky Plot</CardTitle>
                    <CardDescription>
                      1/C² vs. Voltage for doping extraction
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {results.cv_data && (
                      <ResponsiveContainer width="100%" height={350}>
                        <LineChart data={results.cv_data}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis
                            dataKey="voltage"
                            label={{ value: 'Voltage (V)', position: 'insideBottom', offset: -5 }}
                          />
                          <YAxis
                            label={{ value: '1/C² (F⁻²)', angle: -90, position: 'insideLeft' }}
                            tickFormatter={(value) => value.toExponential(1)}
                          />
                          <Tooltip
                            formatter={(value: any) => [(typeof value === 'number' ? value.toExponential(3) : value), '1/C²']}
                          />
                          <Legend />
                          <Line
                            type="monotone"
                            dataKey="inv_c_squared"
                            stroke="#10b981"
                            strokeWidth={2}
                            dot={false}
                            name="1/C²"
                          />
                          {config.deviceType === 'schottky' && results.builtin_voltage && (
                            <ReferenceLine
                              x={results.builtin_voltage}
                              stroke="#ef4444"
                              strokeDasharray="5 5"
                              label={{ value: 'Vbi', position: 'top' }}
                            />
                          )}
                        </LineChart>
                      </ResponsiveContainer>
                    )}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            {/* Parameters Tab */}
            <TabsContent value="parameters" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {/* MOS-specific parameters */}
                {config.deviceType === 'mos' && (
                  <>
                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-medium text-gray-600">
                          Oxide Capacitance (Cox)
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          <div>
                            <span className="text-3xl font-bold">
                              {results.oxide_capacitance ? (results.oxide_capacitance * 1e12).toFixed(2) : 'N/A'}
                            </span>
                            <span className="text-gray-600 ml-2 text-lg">pF</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-medium text-gray-600">
                          Oxide Thickness (tox)
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          <div>
                            <span className="text-3xl font-bold">
                              {results.oxide_thickness ? (results.oxide_thickness * 1e9).toFixed(1) : 'N/A'}
                            </span>
                            <span className="text-gray-600 ml-2 text-lg">nm</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-medium text-gray-600">
                          Flat-Band Voltage (Vfb)
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          <div>
                            <span className="text-3xl font-bold">
                              {results.flatband_voltage?.toFixed(3) ?? 'N/A'}
                            </span>
                            <span className="text-gray-600 ml-2 text-lg">V</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-medium text-gray-600">
                          Threshold Voltage (Vth)
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          <div>
                            <span className="text-3xl font-bold">
                              {results.threshold_voltage?.toFixed(3) ?? 'N/A'}
                            </span>
                            <span className="text-gray-600 ml-2 text-lg">V</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-medium text-gray-600">
                          Interface Trap Density (Dit)
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          <div>
                            <span className="text-3xl font-bold">
                              {results.interface_trap_density ? results.interface_trap_density.toExponential(1) : 'N/A'}
                            </span>
                            <span className="text-gray-600 ml-2 text-sm">cm⁻²eV⁻¹</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-medium text-gray-600">
                          Substrate Doping
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          <div>
                            <span className="text-3xl font-bold">
                              {results.substrate_doping ? results.substrate_doping.toExponential(1) : 'N/A'}
                            </span>
                            <span className="text-gray-600 ml-2 text-sm">cm⁻³</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </>
                )}

                {/* Schottky-specific parameters */}
                {config.deviceType === 'schottky' && (
                  <>
                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-medium text-gray-600">
                          Built-in Voltage (Vbi)
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          <div>
                            <span className="text-3xl font-bold">
                              {results.builtin_voltage?.toFixed(3) ?? 'N/A'}
                            </span>
                            <span className="text-gray-600 ml-2 text-lg">V</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-medium text-gray-600">
                          Barrier Height (φB)
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          <div>
                            <span className="text-3xl font-bold">
                              {results.barrier_height?.toFixed(3) ?? 'N/A'}
                            </span>
                            <span className="text-gray-600 ml-2 text-lg">eV</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-medium text-gray-600">
                          Doping Concentration
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          <div>
                            <span className="text-3xl font-bold">
                              {results.doping_concentration ? results.doping_concentration.toExponential(1) : 'N/A'}
                            </span>
                            <span className="text-gray-600 ml-2 text-sm">cm⁻³</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </>
                )}
              </div>
            </TabsContent>

            {/* Doping Profile Tab */}
            <TabsContent value="doping" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>Doping Concentration Profile</CardTitle>
                  <CardDescription>
                    Extracted doping vs. depth from depletion region
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {results.doping_profile && results.doping_profile.length > 0 ? (
                    <ResponsiveContainer width="100%" height={450}>
                      <LineChart data={results.doping_profile}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="depth"
                          label={{ value: 'Depth (µm)', position: 'insideBottom', offset: -5 }}
                          tickFormatter={(value) => (value * 1e6).toFixed(2)}
                        />
                        <YAxis
                          scale="log"
                          domain={[1e14, 1e18]}
                          label={{ value: 'Doping Concentration (cm⁻³)', angle: -90, position: 'insideLeft' }}
                          tickFormatter={(value) => value.toExponential(0)}
                        />
                        <Tooltip
                          formatter={(value: any, name: string) => {
                            if (name === 'Doping') {
                              return [typeof value === 'number' ? value.toExponential(2) : value, name];
                            }
                            return [value, name];
                          }}
                          labelFormatter={(label) => `Depth: ${(label * 1e6).toFixed(2)} µm`}
                        />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="concentration"
                          stroke="#8b5cf6"
                          strokeWidth={2}
                          dot={false}
                          name="Doping"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="text-center py-12 text-gray-500">
                      <p>Doping profile data not available</p>
                      <p className="text-sm mt-2">Requires complete C-V sweep with sufficient voltage range</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  );
};

export default CVProfiling;