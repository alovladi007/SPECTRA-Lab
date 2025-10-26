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
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Play, Download, AlertCircle, CheckCircle2, TrendingUp } from 'lucide-react';

/**
 * BJT Characterization Interface
 * 
 * Production-ready UI for Bipolar Junction Transistor characterization
 * - NPN and PNP transistor support
 * - Gummel plots (IC, IB vs VBE)
 * - Output characteristics (IC vs VCE)
 * - Current gain (β) extraction
 * - Early voltage (VA) extraction
 */

interface BJTConfig {
  transistorType: 'npn' | 'pnp';
  measurementType: 'gummel' | 'output';
  vbe_start: number;
  vbe_stop: number;
  vbe_step: number;
  vce_fixed: number; // For Gummel plot
  vce_start: number; // For output curves
  vce_stop: number;
  vce_step: number;
  ib_array: string; // Comma-separated values for output curves
  compliance_c: number; // mA
  compliance_b: number; // µA
}

interface BJTResults {
  current_gain: {
    beta_dc: number;
    beta_ac?: number;
    ic_at_max: number;
  };
  early_voltage?: number;
  ideality_factors: {
    collector: number;
    base: number;
  };
  saturation_currents: {
    collector: number;
    base: number;
  };
  quality_score: number;
  gummel_data?: Array<{ vbe: number; ic: number; ib: number; log_ic: number; log_ib: number }>;
  output_data?: Array<{ vce: number; ic: number; ib_label: string }>;
}

const BJTCharacterization: React.FC = () => {
  const [config, setConfig] = useState<BJTConfig>({
    transistorType: 'npn',
    measurementType: 'gummel',
    vbe_start: 0.3,
    vbe_stop: 0.9,
    vbe_step: 0.01,
    vce_fixed: 2,
    vce_start: 0,
    vce_stop: 10,
    vce_step: 0.1,
    ib_array: '1, 2, 5, 10, 20, 50',
    compliance_c: 100,
    compliance_b: 1000
  });

  const [results, setResults] = useState<BJTResults | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [plotScale, setPlotScale] = useState<'linear' | 'log'>('log');

  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    setError(null);

    try {
      const endpoint = config.measurementType === 'gummel'
        ? '/api/v1/electrical/bjt/analyze-gummel'
        : '/api/v1/electrical/bjt/analyze-output';

      const testData = generateTestData(config);

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transistor_type: config.transistorType,
          ...testData
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

  const generateTestData = (cfg: BJTConfig) => {
    if (cfg.measurementType === 'gummel') {
      // Gummel plot: IC and IB vs VBE
      const vbe_array = [];
      const ic_array = [];
      const ib_array = [];
      
      const Is = 1e-15;
      const nC = 1.0;
      const nB = 1.5;
      const Vt = 0.026; // Thermal voltage at 300K
      const beta = 200;
      
      for (let vbe = cfg.vbe_start; vbe <= cfg.vbe_stop; vbe += cfg.vbe_step) {
        vbe_array.push(vbe);
        
        if (cfg.transistorType === 'npn') {
          const ic = Is * Math.exp(vbe / (nC * Vt));
          const ib = ic / beta * Math.exp((vbe - 0.65) / (nB * Vt));
          ic_array.push(ic);
          ib_array.push(ib);
        } else {
          const ic = -Is * Math.exp(-vbe / (nC * Vt));
          const ib = ic / beta * Math.exp((-vbe + 0.65) / (nB * Vt));
          ic_array.push(ic);
          ib_array.push(ib);
        }
      }
      
      return {
        voltage_be: vbe_array,
        current_collector: ic_array,
        current_base: ib_array,
        voltage_ce: cfg.vce_fixed
      };
    } else {
      // Output characteristics: IC vs VCE for different IB
      const ib_values = cfg.ib_array.split(',').map(v => parseFloat(v.trim()) * 1e-6);
      
      return {
        voltage_ce_start: cfg.vce_start,
        voltage_ce_stop: cfg.vce_stop,
        voltage_ce_step: cfg.vce_step,
        current_base_array: ib_values
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
    a.download = `bjt_${config.transistorType}_${config.measurementType}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold mb-2">BJT Characterization</h1>
        <p className="text-gray-600">
          Bipolar Junction Transistor analysis - Gummel plots and output characteristics
        </p>
      </div>

      {/* Configuration Panel */}
      <Card>
        <CardHeader>
          <CardTitle>Measurement Configuration</CardTitle>
          <CardDescription>
            Configure transistor type and measurement parameters
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* Transistor Type */}
            <div className="space-y-2">
              <Label htmlFor="transistorType">Transistor Type</Label>
              <Select
                value={config.transistorType}
                onValueChange={(value) => setConfig({ ...config, transistorType: value as 'npn' | 'pnp' })}
              >
                <SelectTrigger id="transistorType">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="npn">NPN</SelectItem>
                  <SelectItem value="pnp">PNP</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Measurement Type */}
            <div className="space-y-2">
              <Label htmlFor="measurementType">Measurement Type</Label>
              <Select
                value={config.measurementType}
                onValueChange={(value) => setConfig({ ...config, measurementType: value as 'gummel' | 'output' })}
              >
                <SelectTrigger id="measurementType">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="gummel">Gummel Plot</SelectItem>
                  <SelectItem value="output">Output Curves</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Collector Compliance */}
            <div className="space-y-2">
              <Label htmlFor="compliance_c">IC Compliance (mA)</Label>
              <Input
                id="compliance_c"
                type="number"
                value={config.compliance_c}
                onChange={(e) => setConfig({ ...config, compliance_c: parseFloat(e.target.value) })}
                step="10"
                min="1"
              />
            </div>

            {/* Base Compliance */}
            <div className="space-y-2">
              <Label htmlFor="compliance_b">IB Compliance (µA)</Label>
              <Input
                id="compliance_b"
                type="number"
                value={config.compliance_b}
                onChange={(e) => setConfig({ ...config, compliance_b: parseFloat(e.target.value) })}
                step="100"
                min="10"
              />
            </div>
          </div>

          {/* Gummel-specific settings */}
          {config.measurementType === 'gummel' && (
            <div className="border-t pt-4 mt-4">
              <h3 className="font-semibold mb-3">Gummel Plot Settings</h3>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="vbe_start">VBE Start (V)</Label>
                  <Input
                    id="vbe_start"
                    type="number"
                    value={config.vbe_start}
                    onChange={(e) => setConfig({ ...config, vbe_start: parseFloat(e.target.value) })}
                    step="0.1"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="vbe_stop">VBE Stop (V)</Label>
                  <Input
                    id="vbe_stop"
                    type="number"
                    value={config.vbe_stop}
                    onChange={(e) => setConfig({ ...config, vbe_stop: parseFloat(e.target.value) })}
                    step="0.1"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="vbe_step">VBE Step (V)</Label>
                  <Input
                    id="vbe_step"
                    type="number"
                    value={config.vbe_step}
                    onChange={(e) => setConfig({ ...config, vbe_step: parseFloat(e.target.value) })}
                    step="0.01"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="vce_fixed">VCE Fixed (V)</Label>
                  <Input
                    id="vce_fixed"
                    type="number"
                    value={config.vce_fixed}
                    onChange={(e) => setConfig({ ...config, vce_fixed: parseFloat(e.target.value) })}
                    step="0.5"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Output-specific settings */}
          {config.measurementType === 'output' && (
            <div className="border-t pt-4 mt-4">
              <h3 className="font-semibold mb-3">Output Characteristics Settings</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="vce_range">VCE Range (V)</Label>
                  <div className="grid grid-cols-3 gap-2">
                    <Input
                      id="vce_start"
                      type="number"
                      value={config.vce_start}
                      onChange={(e) => setConfig({ ...config, vce_start: parseFloat(e.target.value) })}
                      placeholder="Start"
                      step="0.5"
                    />
                    <Input
                      id="vce_stop"
                      type="number"
                      value={config.vce_stop}
                      onChange={(e) => setConfig({ ...config, vce_stop: parseFloat(e.target.value) })}
                      placeholder="Stop"
                      step="0.5"
                    />
                    <Input
                      id="vce_step"
                      type="number"
                      value={config.vce_step}
                      onChange={(e) => setConfig({ ...config, vce_step: parseFloat(e.target.value) })}
                      placeholder="Step"
                      step="0.1"
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="ib_array">IB Values (µA, comma-separated)</Label>
                  <Input
                    id="ib_array"
                    type="text"
                    value={config.ib_array}
                    onChange={(e) => setConfig({ ...config, ib_array: e.target.value })}
                    placeholder="1, 2, 5, 10, 20, 50"
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
          <Tabs defaultValue="characteristics" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="characteristics">Characteristics</TabsTrigger>
              <TabsTrigger value="parameters">Parameters</TabsTrigger>
              <TabsTrigger value="analysis">Analysis</TabsTrigger>
            </TabsList>

            {/* Characteristics Tab */}
            <TabsContent value="characteristics" className="space-y-4">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>
                      {config.measurementType === 'gummel'
                        ? 'Gummel Plot (IC, IB vs VBE)'
                        : 'Output Characteristics (IC vs VCE)'}
                    </CardTitle>
                    {config.measurementType === 'gummel' && (
                      <Select
                        value={plotScale}
                        onValueChange={(value) => setPlotScale(value as 'linear' | 'log')}
                      >
                        <SelectTrigger className="w-32">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="linear">Linear</SelectItem>
                          <SelectItem value="log">Log Scale</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  </div>
                  <CardDescription>
                    {config.measurementType === 'gummel'
                      ? `Collector and base currents vs. base-emitter voltage at VCE = ${config.vce_fixed}V`
                      : 'Collector current vs. collector-emitter voltage for different base currents'}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {config.measurementType === 'gummel' && results.gummel_data && (
                    <ResponsiveContainer width="100%" height={450}>
                      <LineChart data={results.gummel_data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="vbe"
                          label={{ value: 'VBE (V)', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis
                          yAxisId="left"
                          scale={plotScale === 'log' ? 'log' : 'linear'}
                          domain={plotScale === 'log' ? [1e-12, 'auto'] : ['auto', 'auto']}
                          label={{ value: plotScale === 'log' ? 'log(I) (A)' : 'Current (A)', angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip
                          formatter={(value: any) => [(typeof value === 'number' ? value.toExponential(3) : value)]}
                        />
                        <Legend />
                        {plotScale === 'linear' ? (
                          <>
                            <Line
                              yAxisId="left"
                              type="monotone"
                              dataKey="ic"
                              stroke="#3b82f6"
                              strokeWidth={2}
                              dot={false}
                              name="IC (Collector)"
                            />
                            <Line
                              yAxisId="left"
                              type="monotone"
                              dataKey="ib"
                              stroke="#ef4444"
                              strokeWidth={2}
                              dot={false}
                              name="IB (Base)"
                            />
                          </>
                        ) : (
                          <>
                            <Line
                              yAxisId="left"
                              type="monotone"
                              dataKey="log_ic"
                              stroke="#3b82f6"
                              strokeWidth={2}
                              dot={false}
                              name="log(IC)"
                            />
                            <Line
                              yAxisId="left"
                              type="monotone"
                              dataKey="log_ib"
                              stroke="#ef4444"
                              strokeWidth={2}
                              dot={false}
                              name="log(IB)"
                            />
                          </>
                        )}
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                  {config.measurementType === 'output' && results.output_data && (
                    <ResponsiveContainer width="100%" height={450}>
                      <LineChart data={results.output_data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="vce"
                          label={{ value: 'VCE (V)', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis
                          label={{ value: 'IC (A)', angle: -90, position: 'insideLeft' }}
                          tickFormatter={(value) => value.toExponential(1)}
                        />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="ic" stroke="#3b82f6" strokeWidth={2} dot={false} name="IC" />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Parameters Tab */}
            <TabsContent value="parameters" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {/* Current Gain */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-gray-600">
                      DC Current Gain (β/hFE)
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <span className="text-3xl font-bold">
                          {results.current_gain.beta_dc.toFixed(0)}
                        </span>
                      </div>
                      <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                        <p>at IC = {(results.current_gain.ic_at_max * 1000).toFixed(2)} mA</p>
                        <p className="mt-1 text-gray-500">
                          {results.current_gain.beta_dc > 100 ? 'Good amplification' : 'Moderate gain'}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Early Voltage */}
                {results.early_voltage && (
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium text-gray-600">
                        Early Voltage (VA)
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        <div>
                          <span className="text-3xl font-bold">
                            {results.early_voltage.toFixed(0)}
                          </span>
                          <span className="text-gray-600 ml-2 text-lg">V</span>
                        </div>
                        <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                          <p className="text-gray-500">
                            {results.early_voltage > 50 ? 'Good output resistance' : 'Low output resistance'}
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Collector Ideality Factor */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-gray-600">
                      Collector Ideality (nC)
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <span className="text-3xl font-bold">
                          {results.ideality_factors.collector.toFixed(2)}
                        </span>
                      </div>
                      <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                        <p className="text-gray-500">
                          {results.ideality_factors.collector < 1.2 ? 'Ideal diffusion' : 'Non-ideal effects'}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Base Ideality Factor */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-gray-600">
                      Base Ideality (nB)
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <span className="text-3xl font-bold">
                          {results.ideality_factors.base.toFixed(2)}
                        </span>
                      </div>
                      <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                        <p className="text-gray-500">
                          {results.ideality_factors.base < 2 ? 'Good base quality' : 'Elevated recombination'}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Saturation Currents */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-gray-600">
                      Saturation Current (IsC)
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <span className="text-2xl font-bold">
                          {results.saturation_currents.collector.toExponential(2)}
                        </span>
                        <span className="text-gray-600 ml-2 text-sm">A</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-gray-600">
                      Saturation Current (IsB)
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <span className="text-2xl font-bold">
                          {results.saturation_currents.base.toExponential(2)}
                        </span>
                        <span className="text-gray-600 ml-2 text-sm">A</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            {/* Analysis Tab */}
            <TabsContent value="analysis" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="w-5 h-5" />
                    Device Performance Assessment
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="bg-gray-50 p-4 rounded-lg space-y-3 text-sm">
                    <div className="border-b pb-2">
                      <p className="font-medium mb-1">Current Gain (β)</p>
                      <p className="text-gray-700">
                        {results.current_gain.beta_dc > 100
                          ? '✓ Excellent current gain. Device shows good amplification capability suitable for analog applications.'
                          : results.current_gain.beta_dc > 50
                          ? '✓ Good current gain for most applications.'
                          : '⚠ Moderate current gain. May be adequate for switching but limited for high-gain amplification.'}
                      </p>
                    </div>

                    {results.early_voltage && (
                      <div className="border-b pb-2">
                        <p className="font-medium mb-1">Early Voltage (VA)</p>
                        <p className="text-gray-700">
                          {results.early_voltage > 50
                            ? '✓ High Early voltage indicates good output resistance and minimal base-width modulation. Excellent for precision analog circuits.'
                            : '⚠ Low Early voltage may cause significant output conductance and gain variations with VCE.'}
                        </p>
                      </div>
                    )}

                    <div className="border-b pb-2">
                      <p className="font-medium mb-1">Ideality Factors</p>
                      <p className="text-gray-700">
                        Collector nC = {results.ideality_factors.collector.toFixed(2)}, 
                        Base nB = {results.ideality_factors.base.toFixed(2)}.{' '}
                        {results.ideality_factors.collector < 1.2 && results.ideality_factors.base < 2
                          ? '✓ Near-ideal values indicate good junction quality and minimal recombination.'
                          : '⚠ Elevated ideality factors suggest non-ideal effects such as trap-assisted recombination or series resistance.'}
                      </p>
                    </div>

                    <div>
                      <p className="font-medium mb-1">Overall Assessment</p>
                      <p className="text-gray-700">
                        {results.quality_score >= 80
                          ? 'Device shows excellent characteristics across all measured parameters. Suitable for demanding analog and RF applications.'
                          : results.quality_score >= 60
                          ? 'Device has good overall performance with some parameters that could be optimized.'
                          : 'Device shows several areas for improvement. Review process conditions and material quality.'}
                      </p>
                    </div>
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

export default BJTCharacterization;