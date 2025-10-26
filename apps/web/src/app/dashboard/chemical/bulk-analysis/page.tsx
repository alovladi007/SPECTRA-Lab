'use client'

/**
 * Session 12: Chemical II (SIMS/RBS/NAA, Chemical Etch) - UI Components
 * ======================================================================
 * 
 * React/TypeScript components for bulk and chemical analysis:
 * - SIMS depth profiling interface
 * - RBS spectrum fitting
 * - NAA decay curve analysis
 * - Chemical etch mapping
 * 
 * Author: Semiconductor Lab Platform Team
 * Version: 1.0.0
 * Date: October 2024
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  LineChart, Line, ScatterChart, Scatter, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine, ReferenceArea, Label
} from 'recharts';
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label as UILabel } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue
} from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import {
  Activity, Layers, Zap, Target, TrendingDown,
  Info, Play, Download, Settings, AlertCircle, CheckCircle2
} from 'lucide-react';


// ============================================================================
// Type Definitions
// ============================================================================

interface SIMSProfile {
  depth: number[];
  concentration: number[];
  counts: number[];
  element: string;
  matrix: string;
  metadata?: Record<string, any>;
}

interface SIMSInterface {
  depth: number;
  width: number;
  gradient: number;
  concentration: number;
}

interface RBSSpectrum {
  energy: number[];
  counts: number[];
  metadata?: Record<string, any>;
}

interface RBSLayer {
  element: string;
  atomic_fraction: number;
  thickness: number;
  thickness_nm?: number;
}

interface NAADecayCurve {
  time: number[];
  counts: number[];
  live_time: number[];
  element: string;
  isotope: string;
}

interface EtchProfile {
  pattern_density: number[];
  etch_rate: number[];
  chemistry: string;
  temperature: number;
}

interface LoadingEffect {
  nominal_rate: number;
  max_reduction: number;
  critical_density: number;
  model: string;
  r_squared: number;
  coefficients: Record<string, number>;
}


// ============================================================================
// SIMS Analysis Component
// ============================================================================

export const SIMSAnalysisInterface: React.FC = () => {
  const [profile, setProfile] = useState<SIMSProfile | null>(null);
  const [interfaces, setInterfaces] = useState<SIMSInterface[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedElement, setSelectedElement] = useState('B');
  const [matrixElement, setMatrixElement] = useState('Si');
  const [quantificationMethod, setQuantificationMethod] = useState('RSF');
  const [showLog, setShowLog] = useState(true);
  const [detectionLimit, setDetectionLimit] = useState<number | null>(null);
  const [totalDose, setTotalDose] = useState<number | null>(null);

  const elements = ['B', 'P', 'As', 'Sb', 'Ga', 'In', 'Al', 'N', 'O', 'C'];
  const matrices = ['Si', 'GaAs', 'GaN', 'SiC'];
  const methods = ['RSF', 'IMPLANT_STANDARD', 'MCS'];

  const runSimulation = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `/api/simulator/sims?element=${selectedElement}&peak_depth=100&dose=1e15&straggle=30`
      );
      const data = await response.json();
      setProfile(data);

      // Analyze
      const analysisResponse = await fetch('/api/sims/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile_id: 'sim_1',
          method: quantificationMethod
        })
      });
      const analysis = await analysisResponse.json();
      
      setInterfaces(analysis.interfaces || []);
      setDetectionLimit(analysis.detection_limit);
      setTotalDose(analysis.total_dose);

    } catch (error) {
      console.error('SIMS analysis error:', error);
    }
    setLoading(false);
  };

  const chartData = useMemo(() => {
    if (!profile) return [];
    return profile.depth.map((d, i) => ({
      depth: d,
      concentration: profile.concentration[i],
      counts: profile.counts[i],
      log_concentration: Math.log10(Math.max(profile.concentration[i], 1e10))
    }));
  }, [profile]);

  return (
    <div className="space-y-4 p-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="w-5 h-5" />
            SIMS Depth Profiling Analysis
          </CardTitle>
          <CardDescription>
            Secondary Ion Mass Spectrometry - Quantitative depth profiling
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          
          {/* Controls */}
          <div className="grid grid-cols-4 gap-4">
            <div className="space-y-2">
              <UILabel>Element</UILabel>
              <Select value={selectedElement} onValueChange={setSelectedElement}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {elements.map(el => (
                    <SelectItem key={el} value={el}>{el}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <UILabel>Matrix</UILabel>
              <Select value={matrixElement} onValueChange={setMatrixElement}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {matrices.map(mat => (
                    <SelectItem key={mat} value={mat}>{mat}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <UILabel>Quantification</UILabel>
              <Select value={quantificationMethod} onValueChange={setQuantificationMethod}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {methods.map(m => (
                    <SelectItem key={m} value={m}>{m}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <UILabel>Scale</UILabel>
              <div className="flex items-center gap-2 h-10">
                <span className="text-sm">Linear</span>
                <Switch checked={showLog} onCheckedChange={setShowLog} />
                <span className="text-sm">Log</span>
              </div>
            </div>
          </div>

          <Button onClick={runSimulation} disabled={loading} className="w-full">
            <Play className="w-4 h-4 mr-2" />
            {loading ? 'Analyzing...' : 'Run SIMS Analysis'}
          </Button>

          {/* Results Summary */}
          {profile && (
            <div className="grid grid-cols-3 gap-4 p-4 bg-muted rounded-lg">
              <div>
                <div className="text-sm font-medium">Total Dose</div>
                <div className="text-2xl font-bold">
                  {totalDose ? totalDose.toExponential(2) : '—'}
                </div>
                <div className="text-xs text-muted-foreground">atoms/cm²</div>
              </div>
              <div>
                <div className="text-sm font-medium">Detection Limit</div>
                <div className="text-2xl font-bold">
                  {detectionLimit ? detectionLimit.toExponential(2) : '—'}
                </div>
                <div className="text-xs text-muted-foreground">atoms/cm³</div>
              </div>
              <div>
                <div className="text-sm font-medium">Interfaces Found</div>
                <div className="text-2xl font-bold">{interfaces.length}</div>
                <div className="text-xs text-muted-foreground">transitions</div>
              </div>
            </div>
          )}

          {/* Depth Profile Chart */}
          {chartData.length > 0 && (
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="depth" 
                    label={{ value: 'Depth (nm)', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    label={{ 
                      value: showLog ? 'log₁₀(Concentration)' : 'Concentration (atoms/cm³)', 
                      angle: -90, 
                      position: 'insideLeft' 
                    }}
                    domain={showLog ? ['auto', 'auto'] : [0, 'auto']}
                  />
                  <Tooltip 
                    formatter={(value: any) => 
                      showLog ? value.toFixed(2) : value.toExponential(2)
                    }
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey={showLog ? "log_concentration" : "concentration"}
                    stroke="#3b82f6" 
                    strokeWidth={2}
                    dot={false}
                    name={showLog ? "log₁₀(C)" : "Concentration"}
                  />
                  {interfaces.map((intf, idx) => (
                    <ReferenceLine 
                      key={idx}
                      x={intf.depth}
                      stroke="#ef4444"
                      strokeDasharray="3 3"
                      label={`IF${idx + 1}`}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Interface List */}
          {interfaces.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Detected Interfaces</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {interfaces.map((intf, idx) => (
                    <div key={idx} className="flex items-center justify-between p-2 border rounded">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">IF{idx + 1}</Badge>
                        <span className="text-sm font-mono">{intf.depth.toFixed(1)} nm</span>
                      </div>
                      <div className="text-sm text-muted-foreground">
                        Width: {intf.width.toFixed(1)} nm
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

        </CardContent>
      </Card>
    </div>
  );
};


// ============================================================================
// RBS Analysis Component
// ============================================================================

export const RBSAnalysisInterface: React.FC = () => {
  const [spectrum, setSpectrum] = useState<RBSSpectrum | null>(null);
  const [fittedSpectrum, setFittedSpectrum] = useState<number[] | null>(null);
  const [layers, setLayers] = useState<RBSLayer[]>([
    { element: 'Hf', atomic_fraction: 0.5, thickness: 20 },
    { element: 'O', atomic_fraction: 0.5, thickness: 20 }
  ]);
  const [fitResults, setFitResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [fixComposition, setFixComposition] = useState(false);

  const runFitting = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/rbs/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          spectrum_id: 'test_1',
          layers: layers.map(l => ({
            element: l.element,
            fraction: l.atomic_fraction,
            thickness: l.thickness
          })),
          fix_composition: fixComposition
        })
      });
      
      const data = await response.json();
      setFitResults(data);
      setFittedSpectrum(data.simulated_spectrum);
      
      // Update layers with fitted values
      if (data.fitted_layers) {
        setLayers(data.fitted_layers.map((l: any) => ({
          element: l.element,
          atomic_fraction: l.atomic_fraction,
          thickness: l.thickness,
          thickness_nm: l.thickness_nm
        })));
      }

    } catch (error) {
      console.error('RBS fitting error:', error);
    }
    setLoading(false);
  };

  const loadSimulatedSpectrum = async () => {
    try {
      const response = await fetch(
        '/api/simulator/rbs?layer1_element=Hf&layer1_thickness=20&layer2_element=O&layer2_thickness=20'
      );
      const data = await response.json();
      setSpectrum(data);
    } catch (error) {
      console.error('Spectrum loading error:', error);
    }
  };

  useEffect(() => {
    loadSimulatedSpectrum();
  }, []);

  const chartData = useMemo(() => {
    if (!spectrum) return [];
    return spectrum.energy.map((e, i) => ({
      energy: e,
      counts: spectrum.counts[i],
      fitted: fittedSpectrum ? fittedSpectrum[i] : null
    }));
  }, [spectrum, fittedSpectrum]);

  const updateLayer = (index: number, field: keyof RBSLayer, value: any) => {
    const newLayers = [...layers];
    newLayers[index] = { ...newLayers[index], [field]: value };
    setLayers(newLayers);
  };

  return (
    <div className="space-y-4 p-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="w-5 h-5" />
            RBS Spectrum Fitting
          </CardTitle>
          <CardDescription>
            Rutherford Backscattering Spectrometry - Layer structure analysis
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">

          {/* Layer Configuration */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <UILabel className="text-base font-semibold">Layer Structure</UILabel>
              <div className="flex items-center gap-2">
                <span className="text-sm">Fix Composition</span>
                <Switch checked={fixComposition} onCheckedChange={setFixComposition} />
              </div>
            </div>
            
            {layers.map((layer, idx) => (
              <Card key={idx} className="p-4">
                <div className="grid grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <UILabel>Element</UILabel>
                    <Input 
                      value={layer.element}
                      onChange={(e) => updateLayer(idx, 'element', e.target.value)}
                      placeholder="e.g., Hf"
                    />
                  </div>
                  <div className="space-y-2">
                    <UILabel>Atomic Fraction</UILabel>
                    <Input 
                      type="number"
                      step="0.01"
                      min="0"
                      max="1"
                      value={layer.atomic_fraction}
                      onChange={(e) => updateLayer(idx, 'atomic_fraction', parseFloat(e.target.value))}
                    />
                  </div>
                  <div className="space-y-2">
                    <UILabel>Thickness (10¹⁵ at/cm²)</UILabel>
                    <Input 
                      type="number"
                      step="1"
                      min="1"
                      value={layer.thickness}
                      onChange={(e) => updateLayer(idx, 'thickness', parseFloat(e.target.value))}
                    />
                  </div>
                </div>
                {layer.thickness_nm && (
                  <div className="mt-2 text-sm text-muted-foreground">
                    ≈ {layer.thickness_nm.toFixed(1)} nm
                  </div>
                )}
              </Card>
            ))}
          </div>

          <Button onClick={runFitting} disabled={loading} className="w-full">
            <Target className="w-4 h-4 mr-2" />
            {loading ? 'Fitting...' : 'Fit Spectrum'}
          </Button>

          {/* Fit Quality */}
          {fitResults && (
            <div className="grid grid-cols-2 gap-4 p-4 bg-muted rounded-lg">
              <div>
                <div className="text-sm font-medium">χ²</div>
                <div className="text-2xl font-bold">
                  {fitResults.chi_squared.toFixed(2)}
                </div>
              </div>
              <div>
                <div className="text-sm font-medium">R-factor</div>
                <div className="text-2xl font-bold">
                  {(fitResults.r_factor * 100).toFixed(2)}%
                </div>
              </div>
            </div>
          )}

          {/* Spectrum Chart */}
          {chartData.length > 0 && (
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="energy"
                    label={{ value: 'Energy (keV)', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    label={{ value: 'Yield (counts)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone"
                    dataKey="counts"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                    name="Experimental"
                  />
                  {fittedSpectrum && (
                    <Line 
                      type="monotone"
                      dataKey="fitted"
                      stroke="#ef4444"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      dot={false}
                      name="Fitted"
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Fitted Layers */}
          {fitResults && fitResults.fitted_layers && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Fitted Layer Parameters</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {fitResults.fitted_layers.map((layer: any, idx: number) => (
                    <div key={idx} className="p-3 border rounded">
                      <div className="flex items-center justify-between mb-2">
                        <Badge>Layer {idx + 1}</Badge>
                        <span className="font-medium">{layer.element}</span>
                      </div>
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <div className="text-muted-foreground">Fraction</div>
                          <div className="font-mono">{layer.atomic_fraction.toFixed(3)}</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">Areal Density</div>
                          <div className="font-mono">{layer.thickness.toFixed(1)}</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">Thickness</div>
                          <div className="font-mono">{layer.thickness_nm.toFixed(1)} nm</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

        </CardContent>
      </Card>
    </div>
  );
};


// ============================================================================
// NAA Analysis Component
// ============================================================================

export const NAAAnalysisInterface: React.FC = () => {
  const [sampleCurve, setSampleCurve] = useState<NAADecayCurve | null>(null);
  const [standardCurve, setStandardCurve] = useState<NAADecayCurve | null>(null);
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [element, setElement] = useState('Au');
  const [sampleMass, setSampleMass] = useState(0.5);
  const [standardMass, setStandardMass] = useState(0.1);
  const [standardConc, setStandardConc] = useState(100);

  const elements = ['Au', 'Na', 'Mn', 'Cu', 'As', 'Br'];

  const runAnalysis = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/naa/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample_id: 'sample_1',
          standard_id: 'std_1',
          sample_mass: sampleMass,
          standard_mass: standardMass,
          standard_concentration: standardConc,
          element: element
        })
      });
      
      const data = await response.json();
      setResults(data);

    } catch (error) {
      console.error('NAA analysis error:', error);
    }
    setLoading(false);
  };

  return (
    <div className="space-y-4 p-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5" />
            NAA Quantification
          </CardTitle>
          <CardDescription>
            Neutron Activation Analysis - Comparator method
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">

          {/* Parameters */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <UILabel>Element</UILabel>
              <Select value={element} onValueChange={setElement}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {elements.map(el => (
                    <SelectItem key={el} value={el}>{el}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <UILabel>Sample Mass (g)</UILabel>
              <Input 
                type="number"
                step="0.01"
                value={sampleMass}
                onChange={(e) => setSampleMass(parseFloat(e.target.value))}
              />
            </div>

            <div className="space-y-2">
              <UILabel>Standard Mass (g)</UILabel>
              <Input 
                type="number"
                step="0.01"
                value={standardMass}
                onChange={(e) => setStandardMass(parseFloat(e.target.value))}
              />
            </div>

            <div className="space-y-2">
              <UILabel>Standard Conc. (μg/g)</UILabel>
              <Input 
                type="number"
                step="1"
                value={standardConc}
                onChange={(e) => setStandardConc(parseFloat(e.target.value))}
              />
            </div>
          </div>

          <Button onClick={runAnalysis} disabled={loading} className="w-full">
            <Activity className="w-4 h-4 mr-2" />
            {loading ? 'Analyzing...' : 'Run NAA Analysis'}
          </Button>

          {/* Results */}
          {results && (
            <>
              <Alert>
                <CheckCircle2 className="w-4 h-4" />
                <AlertDescription>
                  Analysis complete - {results.isotope} decay measured
                </AlertDescription>
              </Alert>

              <div className="grid grid-cols-3 gap-4 p-4 bg-muted rounded-lg">
                <div>
                  <div className="text-sm font-medium">Concentration</div>
                  <div className="text-2xl font-bold">
                    {results.concentration.toFixed(2)}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    ± {results.uncertainty.toFixed(2)} μg/g
                  </div>
                </div>
                <div>
                  <div className="text-sm font-medium">Detection Limit</div>
                  <div className="text-2xl font-bold">
                    {results.detection_limit.toFixed(3)}
                  </div>
                  <div className="text-xs text-muted-foreground">μg/g</div>
                </div>
                <div>
                  <div className="text-sm font-medium">Activity</div>
                  <div className="text-2xl font-bold">
                    {results.activity.toExponential(2)}
                  </div>
                  <div className="text-xs text-muted-foreground">Bq</div>
                </div>
              </div>
            </>
          )}

        </CardContent>
      </Card>
    </div>
  );
};


// ============================================================================
// Chemical Etch Analysis Component
// ============================================================================

export const ChemicalEtchInterface: React.FC = () => {
  const [profile, setProfile] = useState<EtchProfile | null>(null);
  const [loading, setLoading] = useState<LoadingEffect | null>(null);
  const [uniformity, setUniformity] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [model, setModel] = useState('linear');
  const [nominalRate, setNominalRate] = useState(100);
  const [alpha, setAlpha] = useState(0.3);

  const models = ['linear', 'exponential', 'power'];

  const runAnalysis = async () => {
    setIsAnalyzing(true);
    try {
      const response = await fetch('/api/etch/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const data = await response.json();
      setProfile(data.profile);
      setLoading(data.loading_effect);
      setUniformity(data.uniformity);

    } catch (error) {
      console.error('Etch analysis error:', error);
    }
    setIsAnalyzing(false);
  };

  const chartData = useMemo(() => {
    if (!profile) return [];
    return profile.pattern_density.map((d, i) => ({
      density: d,
      rate: profile.etch_rate[i],
      fitted: loading ? 
        (loading.coefficients.R0 * (1 - loading.coefficients.alpha * d / 100)) : 
        null
    }));
  }, [profile, loading]);

  return (
    <div className="space-y-4 p-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingDown className="w-5 h-5" />
            Chemical Etch Loading Effects
          </CardTitle>
          <CardDescription>
            Pattern density vs etch rate analysis
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">

          {/* Controls */}
          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-2">
              <UILabel>Model</UILabel>
              <Select value={model} onValueChange={setModel}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {models.map(m => (
                    <SelectItem key={m} value={m}>
                      {m.charAt(0).toUpperCase() + m.slice(1)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <UILabel>Nominal Rate (nm/min)</UILabel>
              <Input 
                type="number"
                step="10"
                value={nominalRate}
                onChange={(e) => setNominalRate(parseFloat(e.target.value))}
              />
            </div>

            <div className="space-y-2">
              <UILabel>Loading Coefficient α</UILabel>
              <Input 
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={alpha}
                onChange={(e) => setAlpha(parseFloat(e.target.value))}
              />
            </div>
          </div>

          <Button onClick={runAnalysis} disabled={isAnalyzing} className="w-full">
            <Settings className="w-4 h-4 mr-2" />
            {isAnalyzing ? 'Analyzing...' : 'Analyze Loading Effect'}
          </Button>

          {/* Loading Effect Results */}
          {loading && (
            <div className="grid grid-cols-3 gap-4 p-4 bg-muted rounded-lg">
              <div>
                <div className="text-sm font-medium">Nominal Rate</div>
                <div className="text-2xl font-bold">
                  {loading.nominal_rate.toFixed(1)}
                </div>
                <div className="text-xs text-muted-foreground">nm/min</div>
              </div>
              <div>
                <div className="text-sm font-medium">Max Reduction</div>
                <div className="text-2xl font-bold">
                  {loading.max_reduction.toFixed(1)}%
                </div>
                <div className="text-xs text-muted-foreground">at 100% density</div>
              </div>
              <div>
                <div className="text-sm font-medium">Critical Density</div>
                <div className="text-2xl font-bold">
                  {loading.critical_density.toFixed(1)}%
                </div>
                <div className="text-xs text-muted-foreground">50% reduction</div>
              </div>
            </div>
          )}

          {/* Etch Rate Chart */}
          {chartData.length > 0 && (
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="density"
                    label={{ value: 'Pattern Density (%)', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    label={{ value: 'Etch Rate (nm/min)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip />
                  <Legend />
                  <Scatter 
                    name="Measured"
                    data={chartData}
                    fill="#3b82f6"
                  />
                  {loading && (
                    <Line 
                      type="monotone"
                      dataKey="fitted"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={false}
                      name={`${loading.model} fit (R²=${loading.r_squared.toFixed(3)})`}
                    />
                  )}
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Uniformity Metrics */}
          {uniformity && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Etch Uniformity</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground">Mean Rate</div>
                    <div className="font-mono">{uniformity.mean_rate.toFixed(2)} nm/min</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">CV</div>
                    <div className="font-mono">{uniformity.cv_percent.toFixed(2)}%</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Uniformity (1σ)</div>
                    <div className="font-mono">{uniformity.uniformity_1sigma.toFixed(2)}%</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Uniformity (Range)</div>
                    <div className="font-mono">{uniformity.uniformity_range.toFixed(2)}%</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

        </CardContent>
      </Card>
    </div>
  );
};


// ============================================================================
// Main Session 12 Interface
// ============================================================================

export const Session12ChemicalBulkInterface: React.FC = () => {
  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">
          Session 12: Chemical & Bulk Analysis
        </h1>
        <p className="text-muted-foreground">
          SIMS, RBS, NAA, and Chemical Etch characterization suite
        </p>
      </div>

      <Tabs defaultValue="sims" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="sims">SIMS</TabsTrigger>
          <TabsTrigger value="rbs">RBS</TabsTrigger>
          <TabsTrigger value="naa">NAA</TabsTrigger>
          <TabsTrigger value="etch">Etch Mapping</TabsTrigger>
        </TabsList>

        <TabsContent value="sims">
          <SIMSAnalysisInterface />
        </TabsContent>

        <TabsContent value="rbs">
          <RBSAnalysisInterface />
        </TabsContent>

        <TabsContent value="naa">
          <NAAAnalysisInterface />
        </TabsContent>

        <TabsContent value="etch">
          <ChemicalEtchInterface />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Session12ChemicalBulkInterface;
