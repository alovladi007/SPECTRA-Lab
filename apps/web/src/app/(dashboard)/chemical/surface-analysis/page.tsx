/**
 * Session 11: Surface Analysis (XPS/XRF) - UI Components
 * =======================================================
 * 
 * React/TypeScript components for XPS and XRF analysis:
 * - XPS peak fitting and quantification interface
 * - XRF element identification interface
 * 
 * Author: Semiconductor Lab Platform Team
 * Version: 1.0.0
 * Date: October 2024
 */

import React, { useState, useMemo } from 'react';
import {
  LineChart, Line, ScatterChart, Scatter, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from 'recharts';
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label as UILabel } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Layers, Zap, Play, Settings, CheckCircle2 } from 'lucide-react';


// ============================================================================
// XPS Analysis Component
// ============================================================================

export const XPSAnalysisInterface: React.FC = () => {
  const [spectrum, setSpectrum] = useState<any>(null);
  const [peaks, setPeaks] = useState<any[]>([]);
  const [composition, setComposition] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(false);
  const [element, setElement] = useState('Si');
  const [peakPositions, setPeakPositions] = useState('99, 103');
  const [backgroundType, setBackgroundType] = useState('shirley');

  const elements = ['Si', 'C', 'O', 'N', 'Ti', 'Al', 'Cu', 'Au'];

  const runAnalysis = async () => {
    setLoading(true);
    try {
      // Get simulated spectrum
      const specResponse = await fetch(`/api/simulator/xps?element=${element}&peak_position=99&peak_width=1.5`);
      const specData = await specResponse.json();
      setSpectrum(specData);

      // Analyze
      const positions = peakPositions.split(',').map(p => parseFloat(p.trim()));
      const analysisResponse = await fetch('/api/xps/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          spectrum_id: 'test_1',
          background_type: backgroundType,
          peak_positions: positions,
          charge_reference: true
        })
      });
      
      const analysis = await analysisResponse.json();
      setPeaks(analysis.peaks || []);
      setComposition(analysis.composition || {});

    } catch (error) {
      console.error('XPS analysis error:', error);
    }
    setLoading(false);
  };

  const chartData = useMemo(() => {
    if (!spectrum) return [];
    return spectrum.binding_energy.map((be: number, i: number) => ({
      binding_energy: be,
      intensity: spectrum.intensity[i]
    }));
  }, [spectrum]);

  return (
    <div className="space-y-4 p-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="w-5 h-5" />
            XPS Analysis
          </CardTitle>
          <CardDescription>
            X-ray Photoelectron Spectroscopy - Surface composition
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          
          <div className="grid grid-cols-3 gap-4">
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
              <UILabel>Background</UILabel>
              <Select value={backgroundType} onValueChange={setBackgroundType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="shirley">Shirley</SelectItem>
                  <SelectItem value="linear">Linear</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <UILabel>Peak Positions (eV)</UILabel>
              <Input 
                value={peakPositions}
                onChange={(e) => setPeakPositions(e.target.value)}
                placeholder="99, 103"
              />
            </div>
          </div>

          <Button onClick={runAnalysis} disabled={loading} className="w-full">
            <Play className="w-4 h-4 mr-2" />
            {loading ? 'Analyzing...' : 'Run XPS Analysis'}
          </Button>

          {Object.keys(composition).length > 0 && (
            <div className="grid grid-cols-3 gap-4 p-4 bg-muted rounded-lg">
              {Object.entries(composition).map(([elem, percent]) => (
                <div key={elem}>
                  <div className="text-sm font-medium">{elem}</div>
                  <div className="text-2xl font-bold">{percent.toFixed(1)}%</div>
                  <div className="text-xs text-muted-foreground">atomic</div>
                </div>
              ))}
            </div>
          )}

          {chartData.length > 0 && (
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="binding_energy"
                    reversed
                    label={{ value: 'Binding Energy (eV)', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis label={{ value: 'Intensity (CPS)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone"
                    dataKey="intensity"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                    name="XPS Spectrum"
                  />
                  {peaks.map((peak, idx) => (
                    <ReferenceLine 
                      key={idx}
                      x={peak.position}
                      stroke="#ef4444"
                      strokeDasharray="3 3"
                      label={{ value: `${peak.element}`, position: 'top' }}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {peaks.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Fitted Peaks</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {peaks.map((peak, idx) => (
                    <div key={idx} className="flex items-center justify-between p-2 border rounded">
                      <div>
                        <Badge variant="outline">{peak.element}</Badge>
                        <span className="ml-2 font-mono">{peak.position.toFixed(2)} eV</span>
                      </div>
                      <div className="text-sm">
                        FWHM: {peak.fwhm.toFixed(2)} eV
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
// XRF Analysis Component
// ============================================================================

export const XRFAnalysisInterface: React.FC = () => {
  const [spectrum, setSpectrum] = useState<any>(null);
  const [identifiedPeaks, setIdentifiedPeaks] = useState<any[]>([]);
  const [composition, setComposition] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(false);
  const [element1, setElement1] = useState('Ti');
  const [element2, setElement2] = useState('Cu');
  const [conc1, setConc1] = useState(60);
  const [conc2, setConc2] = useState(40);

  const elements = ['Ti', 'Cr', 'Fe', 'Ni', 'Cu', 'Zn', 'Ga', 'As', 'Mo', 'Ag', 'Au'];

  const runAnalysis = async () => {
    setLoading(true);
    try {
      const specResponse = await fetch(
        `/api/simulator/xrf?element1=${element1}&element2=${element2}&conc1=${conc1}&conc2=${conc2}`
      );
      const specData = await specResponse.json();
      setSpectrum(specData);

      const analysisResponse = await fetch('/api/xrf/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          spectrum_id: 'test_1',
          method: 'fundamental_parameters',
          threshold: 100
        })
      });
      
      const analysis = await analysisResponse.json();
      setIdentifiedPeaks(analysis.identified_peaks || []);
      setComposition(analysis.composition || {});

    } catch (error) {
      console.error('XRF analysis error:', error);
    }
    setLoading(false);
  };

  const chartData = useMemo(() => {
    if (!spectrum) return [];
    return spectrum.energy.map((e: number, i: number) => ({
      energy: e,
      intensity: spectrum.intensity[i]
    }));
  }, [spectrum]);

  return (
    <div className="space-y-4 p-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="w-5 h-5" />
            XRF Analysis
          </CardTitle>
          <CardDescription>
            X-ray Fluorescence - Elemental composition
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          
          <div className="grid grid-cols-4 gap-4">
            <div className="space-y-2">
              <UILabel>Element 1</UILabel>
              <Select value={element1} onValueChange={setElement1}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {elements.map(el => <SelectItem key={el} value={el}>{el}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <UILabel>Conc 1 (wt%)</UILabel>
              <Input 
                type="number"
                value={conc1}
                onChange={(e) => setConc1(parseFloat(e.target.value))}
              />
            </div>

            <div className="space-y-2">
              <UILabel>Element 2</UILabel>
              <Select value={element2} onValueChange={setElement2}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {elements.map(el => <SelectItem key={el} value={el}>{el}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <UILabel>Conc 2 (wt%)</UILabel>
              <Input 
                type="number"
                value={conc2}
                onChange={(e) => setConc2(parseFloat(e.target.value))}
              />
            </div>
          </div>

          <Button onClick={runAnalysis} disabled={loading} className="w-full">
            <Settings className="w-4 h-4 mr-2" />
            {loading ? 'Analyzing...' : 'Run XRF Analysis'}
          </Button>

          {Object.keys(composition).length > 0 && (
            <div className="grid grid-cols-3 gap-4 p-4 bg-muted rounded-lg">
              {Object.entries(composition).map(([elem, percent]) => (
                <div key={elem}>
                  <div className="text-sm font-medium">{elem}</div>
                  <div className="text-2xl font-bold">{percent.toFixed(1)}%</div>
                  <div className="text-xs text-muted-foreground">wt%</div>
                </div>
              ))}
            </div>
          )}

          {chartData.length > 0 && (
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="energy"
                    label={{ value: 'Energy (keV)', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis label={{ value: 'Intensity (counts)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Area 
                    type="monotone"
                    dataKey="intensity"
                    stroke="#3b82f6"
                    fill="#3b82f680"
                    name="XRF Spectrum"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          {identifiedPeaks.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Identified Elements</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {identifiedPeaks.map((peak, idx) => (
                    <div key={idx} className="flex items-center justify-between p-2 border rounded">
                      <div>
                        <Badge>{peak.element}</Badge>
                        <span className="ml-2 text-sm text-muted-foreground">{peak.line}</span>
                      </div>
                      <div className="font-mono text-sm">{peak.energy.toFixed(2)} keV</div>
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
// Main Session 11 Interface
// ============================================================================

export const Session11SurfaceAnalysisInterface: React.FC = () => {
  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">
          Session 11: Surface Analysis
        </h1>
        <p className="text-muted-foreground">
          XPS and XRF characterization suite
        </p>
      </div>

      <Tabs defaultValue="xps" className="space-y-4">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="xps">XPS</TabsTrigger>
          <TabsTrigger value="xrf">XRF</TabsTrigger>
        </TabsList>

        <TabsContent value="xps">
          <XPSAnalysisInterface />
        </TabsContent>

        <TabsContent value="xrf">
          <XRFAnalysisInterface />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Session11SurfaceAnalysisInterface;
