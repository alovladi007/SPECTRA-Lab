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
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Play, Download, AlertCircle, CheckCircle2, TrendingUp, Info } from 'lucide-react';

/**
 * MOSFET Characterization Interface
 * 
 * Production-ready UI for MOSFET I-V characterization
 * - n-MOS and p-MOS support
 * - Transfer and output characteristics
 * - Real-time parameter extraction
 * - Quality scoring and validation
 */

interface MOSFETConfig {
  deviceType: 'n-mos' | 'p-mos';
  measurementMode: 'transfer' | 'output';
  vgs_start: number;
  vgs_stop: number;
  vgs_step: number;
  vds_fixed: number;
  vds_array: string;
  vgs_array: string;
  width: number;
  length: number;
  oxide_thickness: number;
  compliance: number;
}

interface MOSFETResults {
  threshold_voltage: {
    linear_extrapolation: number;
    constant_current: number;
    transconductance: number;
    average: number;
    std: number;
  };
  transconductance_max: {
    value: number;
    vgs_at_max: number;
  };
  subthreshold_slope: number;
  ion_ioff_ratio: number;
  on_resistance: number;
  mobility: {
    effective: number;
    peak: number;
  };
  lambda?: number;
  quality_score: number;
  transfer_data?: Array<{ vgs: number; id: number; id_log: number; gm?: number }>;
  output_data?: Array<{ vds: number; id: number; vgs_label: string }>;
}

const MOSFETCharacterization: React.FC = () => {
  const [config, setConfig] = useState<MOSFETConfig>({
    deviceType: 'n-mos',
    measurementMode: 'transfer',
    vgs_start: -1,
    vgs_stop: 5,
    vgs_step: 0.1,
    vds_fixed: 0.1,
    vds_array: '0, 1, 2, 3, 4, 5',
    vgs_array: '0, 0.5, 1, 1.5, 2, 2.5, 3',
    width: 10,
    length: 1,
    oxide_thickness: 10,
    compliance: 100
  });

  const [results, setResults] = useState<MOSFETResults | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [plotScale, setPlotScale] = useState<'linear' | 'log'>('linear');

  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    setError(null);

    try {
      const endpoint = config.measurementMode === 'transfer'
        ? '/api/v1/electrical/mosfet/analyze-transfer'
        : '/api/v1/electrical/mosfet/analyze-output';

      const testData = generateTestData(config);

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          device_type: config.deviceType,
          ...testData,
          width: config.width * 1e-6,
          length: config.length * 1e-6,
          oxide_thickness: config.oxide_thickness * 1e-9
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

  const generateTestData = (cfg: MOSFETConfig) => {
    if (cfg.measurementMode === 'transfer') {
      const vgs_array = [];
      const id_array = [];
      
      const Vth = cfg.deviceType === 'n-mos' ? 0.7 : -0.7;
      const k = 1e-3;
      
      for (let vgs = cfg.vgs_start; vgs <= cfg.vgs_stop; vgs += cfg.vgs_step) {
        vgs_array.push(vgs);
        
        if (cfg.deviceType === 'n-mos') {
          if (vgs < Vth) {
            id_array.push(1e-9 * Math.exp((vgs - Vth) / 0.1));
          } else {
            const vds = cfg.vds_fixed;
            if (vds < vgs - Vth) {
              id_array.push(k * (cfg.width / cfg.length) * ((vgs - Vth) * vds - vds ** 2 / 2));
            } else {
              id_array.push(0.5 * k * (cfg.width / cfg.length) * (vgs - Vth) ** 2 * (1 + 0.02 * vds));
            }
          }
        } else {
          if (vgs > Vth) {
            id_array.push(-1e-9 * Math.exp((Vth - vgs) / 0.1));
          } else {
            const vds = -cfg.vds_fixed;
            if (Math.abs(vds) < Math.abs(vgs - Vth)) {
              id_array.push(-k * (cfg.width / cfg.length) * ((vgs - Vth) * vds - vds ** 2 / 2));
            } else {
              id_array.push(-0.5 * k * (cfg.width / cfg.length) * (vgs - Vth) ** 2 * (1 + 0.02 * Math.abs(vds)));
            }
          }
        }
      }
      
      return {
        voltage_gate: vgs_array,
        current_drain: id_array,
        voltage_drain: cfg.vds_fixed
      };
    } else {
      const vds_arr = cfg.vds_array.split(',').map(v => parseFloat(v.trim()));
      const vgs_arr = cfg.vgs_array.split(',').map(v => parseFloat(v.trim()));
      
      return {
        voltage_drain_array: vds_arr,
        voltage_gate_array: vgs_arr
      };
    }
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
    a.download = `mosfet_${config.deviceType}_${config.measurementMode}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold mb-2">MOSFET I-V Characterization</h1>
        <p className="text-gray-600">
          Comprehensive transfer and output characteristic analysis for MOS transistors
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
                onValueChange={(value) => setConfig({ ...config, deviceType: value as 'n-mos' | 'p-mos' })}
              >
                <SelectTrigger id="deviceType">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="n-mos">n-MOS (Enhancement)</SelectItem>
                  <SelectItem value="p-mos">p-MOS (Enhancement)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Measurement Mode */}
            <div className="space-y-2">
              <Label htmlFor="measurementMode">Measurement Mode</Label>
              <Select
                value={config.measurementMode}
                onValueChange={(value) => setConfig({ ...config, measurementMode: value as 'transfer' | 'output' })}
              >
                <SelectTrigger id="measurementMode">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="transfer">Transfer (Id-Vgs)</SelectItem>
                  <SelectItem value="output">Output (Id-Vds)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Width */}
            <div className="space-y-2">
              <Label htmlFor="width">Channel Width (µm)</Label>
              <Input
                id="width"
                type="number"
                value={config.width}
                onChange={(e) => setConfig({ ...config, width: parseFloat(e.target.value) })}
                step="0.1"
                min="0.1"
              />
            </div>

            {/* Length */}
            <div className="space-y-2">
              <Label htmlFor="length">Channel Length (µm)</Label>
              <Input
                id="length"
                type="number"
                value={config.length}
                onChange={(e) => setConfig({ ...config, length: parseFloat(e.target.value) })}
                step="0.01"
                min="0.01"
              />
            </div>

            {/* Oxide Thickness */}
            <div className="space-y-2">
              <Label htmlFor="tox">Oxide Thickness (nm)</Label>
              <Input
                id="tox"
                type="number"
                value={config.oxide_thickness}
                onChange={(e) => setConfig({ ...config, oxide_thickness: parseFloat(e.target.value) })}
                step="0.1"
                min="0.1"
              />
            </div>

            {/* Compliance */}
            <div className="space-y-2">
              <Label htmlFor="compliance">Compliance (mA)</Label>
              <Input
                id="compliance"
                type="number"
                value={config.compliance}
                onChange={(e) => setConfig({ ...config, compliance: parseFloat(e.target.value) })}
                step="1"
                min="1"
              />
            </div>
          </div>

          {/* Transfer-specific settings */}
          {config.measurementMode === 'transfer' && (
            <div className="border-t pt-4 mt-4">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Info className="w-4 h-4" />
                Transfer Curve Settings
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="vgs_start">Vgs Start (V)</Label>
                  <Input
                    id="vgs_start"
                    type="number"
                    value={config.vgs_start}
                    onChange={(e) => setConfig({ ...config, vgs_start: parseFloat(e.target.value) })}
                    step="0.1"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="vgs_stop">Vgs Stop (V)</Label>
                  <Input
                    id="vgs_stop"
                    type="number"
                    value={config.vgs_stop}
                    onChange={(e) => setConfig({ ...config, vgs_stop: parseFloat(e.target.value) })}
                    step="0.1"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="vgs_step">Vgs Step (V)</Label>
                  <Input
                    id="vgs_step"
                    type="number"
                    value={config.vgs_step}
                    onChange={(e) => setConfig({ ...config, vgs_step: parseFloat(e.target.value) })}
                    step="0.01"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="vds_fixed">Vds Fixed (V)</Label>
                  <Input
                    id="vds_fixed"
                    type="number"
                    value={config.vds_fixed}
                    onChange={(e) => setConfig({ ...config, vds_fixed: parseFloat(e.target.value) })}
                    step="0.1"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Output-specific settings */}
          {config.measurementMode === 'output' && (
            <div className="border-t pt-4 mt-4">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Info className="w-4 h-4" />
                Output Curve Settings
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="vds_array">Vds Values (V, comma-separated)</Label>
                  <Input
                    id="vds_array"
                    type="text"
                    value={config.vds_array}
                    onChange={(e) => setConfig({ ...config, vds_array: e.target.value })}
                    placeholder="0, 1, 2, 3, 4, 5"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="vgs_array">Vgs Values (V, comma-separated)</Label>
                  <Input
                    id="vgs_array"
                    type="text"
                    value={config.vgs_array}
                    onChange={(e) => setConfig({ ...config, vgs_array: e.target.value })}
                    placeholder="0, 0.5, 1, 1.5, 2, 2.5, 3"
                  />
                </div>
              </div>
            </div>
          )}

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
          {/* Quality Score Banner */}
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
          <Tabs defaultValue="characteristics" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="characteristics">I-V Characteristics</TabsTrigger>
              <TabsTrigger value="parameters">Extracted Parameters</TabsTrigger>
              <TabsTrigger value="analysis">Detailed Analysis</TabsTrigger>
            </TabsList>

            {/* Characteristics Tab */}
            <TabsContent value="characteristics" className="space-y-4">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>
                      {config.measurementMode === 'transfer' 
                        ? 'Transfer Characteristic (Id-Vgs)'
                        : 'Output Characteristics (Id-Vds)'}
                    </CardTitle>
                    {config.measurementMode === 'transfer' && (
                      <Select
                        value={plotScale}
                        onValueChange={(value) => setPlotScale(value as 'linear' | 'log')}
                      >
                        <SelectTrigger className="w-36">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="linear">Linear Scale</SelectItem>
                          <SelectItem value="log">Log Scale</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  </div>
                  <CardDescription>
                    {config.measurementMode === 'transfer'
                      ? `Drain current vs. gate voltage at Vds = ${config.vds_fixed}V`
                      : 'Drain current vs. drain voltage for multiple gate voltages'}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {config.measurementMode === 'transfer' && results.transfer_data && (
                    <ResponsiveContainer width="100%" height={450}>
                      <LineChart data={results.transfer_data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="vgs"
                          label={{ value: 'Gate Voltage Vgs (V)', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis
                          scale={plotScale === 'log' ? 'log' : 'linear'}
                          domain={plotScale === 'log' ? [1e-12, 'auto'] : ['auto', 'auto']}
                          label={{ value: plotScale === 'log' ? 'log(Id) (A)' : 'Drain Current Id (A)', angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip
                          formatter={(value: any, name: string) => {
                            if (name === 'Drain Current') {
                              return [typeof value === 'number' ? value.toExponential(3) : value, name];
                            }
                            return [value, name];
                          }}
                        />
                        <Legend />
                        {plotScale === 'linear' ? (
                          <Line
                            type="monotone"
                            dataKey="id"
                            stroke="#3b82f6"
                            strokeWidth={2}
                            dot={false}
                            name="Drain Current"
                          />
                        ) : (
                          <Line
                            type="monotone"
                            dataKey="id_log"
                            stroke="#3b82f6"
                            strokeWidth={2}
                            dot={false}
                            name="log(Id)"
                          />
                        )}
                        {results.threshold_voltage && plotScale === 'linear' && (
                          <ReferenceLine
                            x={results.threshold_voltage.average}
                            stroke="#ef4444"
                            strokeDasharray="5 5"
                            label={{ value: `Vth = ${results.threshold_voltage.average.toFixed(2)}V`, position: 'top' }}
                          />
                        )}
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                  {config.measurementMode === 'output' && results.output_data && (
                    <ResponsiveContainer width="100%" height={450}>
                      <LineChart data={results.output_data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="vds"
                          label={{ value: 'Drain Voltage Vds (V)', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis
                          label={{ value: 'Drain Current Id (A)', angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="id" stroke="#3b82f6" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Parameters Tab */}
            <TabsContent value="parameters" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {/* Threshold Voltage */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-gray-600">
                      Threshold Voltage (Vth)
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <span className="text-3xl font-bold">
                          {results.threshold_voltage.average.toFixed(3)}
                        </span>
                        <span className="text-gray-600 ml-2 text-lg">V</span>
                      </div>
                      <div className="text-xs text-gray-600 space-y-1 bg-gray-50 p-2 rounded">
                        <p className="flex justify-between">
                          <span>Linear Extrap:</span>
                          <span className="font-mono">{results.threshold_voltage.linear_extrapolation.toFixed(3)} V</span>
                        </p>
                        <p className="flex justify-between">
                          <span>Constant Current:</span>
                          <span className="font-mono">{results.threshold_voltage.constant_current.toFixed(3)} V</span>
                        </p>
                        <p className="flex justify-between">
                          <span>gm Method:</span>
                          <span className="font-mono">{results.threshold_voltage.transconductance.toFixed(3)} V</span>
                        </p>
                        <p className="flex justify-between border-t pt-1 mt-1">
                          <span>Std Deviation:</span>
                          <span className="font-mono">{results.threshold_voltage.std.toFixed(4)} V</span>
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Transconductance */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-gray-600">
                      Peak Transconductance (gm)
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <span className="text-3xl font-bold">
                          {(results.transconductance_max.value * 1000).toFixed(2)}
                        </span>
                        <span className="text-gray-600 ml-2 text-lg">mS</span>
                      </div>
                      <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                        <p>at Vgs = {results.transconductance_max.vgs_at_max.toFixed(2)} V</p>
                        <p className="mt-1 text-gray-500">
                          Measure of amplification capability
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Mobility */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-gray-600">
                      Effective Mobility (µeff)
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <span className="text-3xl font-bold">
                          {results.mobility.effective.toFixed(0)}
                        </span>
                        <span className="text-gray-600 ml-2 text-lg">cm²/V·s</span>
                      </div>
                      <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                        <p className="flex justify-between">
                          <span>Peak Mobility:</span>
                          <span className="font-mono">{results.mobility.peak.toFixed(0)} cm²/V·s</span>
                        </p>
                        <p className="mt-1 text-gray-500">
                          Typical Si: 300-600 cm²/V·s
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Ion/Ioff Ratio */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-gray-600">
                      Ion/Ioff Ratio
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <span className="text-3xl font-bold">
                          {results.ion_ioff_ratio.toExponential(1)}
                        </span>
                      </div>
                      <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                        <p className="text-gray-500">
                          {results.ion_ioff_ratio > 1e6
                            ? '✓ Excellent switching behavior'
                            : '⚠ May have elevated leakage'}
                        </p>
                        <p className="mt-1 text-gray-500">
                          Target: {'>'}10⁶ for digital applications
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Subthreshold Slope */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-gray-600">
                      Subthreshold Slope (SS)
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <span className="text-3xl font-bold">
                          {(results.subthreshold_slope * 1000).toFixed(1)}
                        </span>
                        <span className="text-gray-600 ml-2 text-lg">mV/dec</span>
                      </div>
                      <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                        <p className="text-gray-500">
                          {results.subthreshold_slope * 1000 < 80
                            ? '✓ Good interface quality'
                            : '⚠ May indicate interface traps'}
                        </p>
                        <p className="mt-1 text-gray-500">
                          Ideal: 60 mV/dec at 300K
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* On-Resistance */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-gray-600">
                      On-Resistance (Ron)
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <span className="text-3xl font-bold">
                          {results.on_resistance.toFixed(1)}
                        </span>
                        <span className="text-gray-600 ml-2 text-lg">Ω</span>
                      </div>
                      <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                        <p className="text-gray-500">
                          Lower values indicate better performance
                        </p>
                        <p className="mt-1 text-gray-500">
                          Important for power applications
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            {/* Detailed Analysis Tab */}
            <TabsContent value="analysis" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="w-5 h-5" />
                    Device Performance Assessment
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Quality Interpretation */}
                  <div>
                    <h4 className="font-semibold mb-2">Overall Device Quality</h4>
                    {results.quality_score >= 80 ? (
                      <Alert className="bg-green-50 border-green-200">
                        <CheckCircle2 className="w-4 h-4 text-green-600" />
                        <AlertDescription className="text-green-800">
                          <strong>Excellent device characteristics.</strong> All parameters are within expected ranges for high-quality devices. This MOSFET shows good interface quality, appropriate threshold voltage, and excellent switching behavior.
                        </AlertDescription>
                      </Alert>
                    ) : results.quality_score >= 60 ? (
                      <Alert className="bg-yellow-50 border-yellow-200">
                        <Info className="w-4 h-4 text-yellow-600" />
                        <AlertDescription className="text-yellow-800">
                          <strong>Good device characteristics.</strong> Most parameters are acceptable, but some may benefit from process optimization. Review the detailed parameters below for specific areas of improvement.
                        </AlertDescription>
                      </Alert>
                    ) : (
                      <Alert className="bg-red-50 border-red-200">
                        <AlertCircle className="w-4 h-4 text-red-600" />
                        <AlertDescription className="text-red-800">
                          <strong>Device quality concerns.</strong> Several parameters are outside expected ranges. This may indicate process issues, measurement problems, or device degradation. Review troubleshooting guide.
                        </AlertDescription>
                      </Alert>
                    )}
                  </div>

                  {/* Parameter-by-Parameter Analysis */}
                  <div className="space-y-3">
                    <h4 className="font-semibold">Parameter Analysis</h4>
                    
                    <div className="bg-gray-50 p-4 rounded-lg space-y-3 text-sm">
                      <div className="border-b pb-2">
                        <p className="font-medium mb-1">Threshold Voltage (Vth)</p>
                        <p className="text-gray-700">
                          {Math.abs(results.threshold_voltage.average) < 1.5
                            ? '✓ Nominal voltage - suitable for low-power applications with standard supply voltages (3.3V or 1.8V).'
                            : '⚠ High threshold voltage - may require higher supply voltage or process adjustments.'}
                        </p>
                        <p className="text-xs text-gray-600 mt-1">
                          Standard deviation of {(results.threshold_voltage.std * 1000).toFixed(1)} mV indicates{' '}
                          {results.threshold_voltage.std < 0.05 ? 'excellent' : 'acceptable'} measurement consistency across extraction methods.
                        </p>
                      </div>

                      <div className="border-b pb-2">
                        <p className="font-medium mb-1">Ion/Ioff Ratio</p>
                        <p className="text-gray-700">
                          {results.ion_ioff_ratio > 1e6
                            ? '✓ Excellent switching behavior with low off-state leakage. Suitable for low-power digital applications.'
                            : results.ion_ioff_ratio > 1e4
                            ? '⚠ Moderate switching ratio. May have elevated off-state leakage current, affecting standby power consumption.'
                            : '⚠ Poor switching ratio. Significant leakage current - check for gate oxide quality, junction integrity, or measurement setup.'}
                        </p>
                      </div>

                      <div className="border-b pb-2">
                        <p className="font-medium mb-1">Subthreshold Slope (SS)</p>
                        <p className="text-gray-700">
                          {results.subthreshold_slope * 1000 < 70
                            ? '✓ Excellent interface quality. SS close to ideal thermal limit (60 mV/dec at 300K).'
                            : results.subthreshold_slope * 1000 < 100
                            ? '✓ Good interface quality with acceptable SS.'
                            : '⚠ Elevated SS may indicate interface traps (Dit), bulk defects, or non-ideal subthreshold behavior. Consider interface passivation or process optimization.'}
                        </p>
                      </div>

                      <div className="border-b pb-2">
                        <p className="font-medium mb-1">Effective Mobility (µeff)</p>
                        <p className="text-gray-700">
                          {results.mobility.effective > 300
                            ? config.deviceType === 'n-mos'
                              ? '✓ Good electron mobility for n-MOS devices. Indicates high-quality gate oxide and channel interface.'
                              : '✓ Excellent hole mobility for p-MOS devices (typically 2-3× lower than electron mobility).'
                            : '⚠ Low carrier mobility may be limited by interface roughness, impurity scattering, or oxide charges. Peak mobility of ' +
                              results.mobility.peak.toFixed(0) + ' cm²/V·s suggests ' +
                              (results.mobility.peak > results.mobility.effective * 1.5 ? 'significant field-dependent degradation.' : 'consistent transport properties.')}
                        </p>
                      </div>

                      <div>
                        <p className="font-medium mb-1">On-Resistance (Ron)</p>
                        <p className="text-gray-700">
                          {results.on_resistance < 100
                            ? '✓ Low on-resistance indicates efficient current delivery in saturation region. Good for power switching applications.'
                            : '⚠ Elevated on-resistance may cause voltage drops and power dissipation in switching applications. Consider increasing W/L ratio or reducing series resistance (contact, source/drain).'}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Recommendations */}
                  <div>
                    <h4 className="font-semibold mb-2">Recommendations</h4>
                    <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                      {results.quality_score >= 80 ? (
                        <>
                          <li>Device parameters are excellent - proceed with circuit integration</li>
                          <li>Document these results as baseline for process control</li>
                          <li>Consider reliability testing (BTI, HCI) for production qualification</li>
                        </>
                      ) : (
                        <>
                          {results.subthreshold_slope * 1000 > 100 && (
                            <li>Investigate interface trap density - consider interface passivation treatments</li>
                          )}
                          {results.ion_ioff_ratio < 1e6 && (
                            <li>Evaluate gate oxide integrity and junction leakage paths</li>
                          )}
                          {results.mobility.effective < 300 && (
                            <li>Optimize oxide growth or deposition conditions to improve interface quality</li>
                          )}
                          {results.on_resistance > 100 && (
                            <li>Reduce series resistance through contact optimization or geometry adjustments</li>
                          )}
                        </>
                      )}
                    </ul>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  );
};

export default MOSFETCharacterization;