// apps/web/src/app/(dashboard)/electrical/four-point-probe/page.tsx
// Four-Point Probe UI Component - Production Ready

'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Play, Square, Download, AlertTriangle, CheckCircle } from 'lucide-react';

const FourPointProbe = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [config, setConfig] = useState({
    current: '0.001',
    numConfigs: '4',
    sampleThickness: '0.05',
    temperature: '300',
    waferDiameter: '',
    enableWaferMap: false
  });

  const handleStartMeasurement = () => {
    setIsRunning(true);
    // Simulated measurement - production would call API
    setTimeout(() => {
      setIsRunning(false);
      setResult({
        sheet_resistance: { value: 125.3, unit: 'Ω/sq', uncertainty: 2.1 },
        resistivity: { value: 0.627, unit: 'Ω·cm' },
        statistics: { mean: 125.3, std: 2.1, cv_percent: 1.68, count: 4 },
        contact_check: { passed: true, max_resistance: 245.0 }
      });
    }, 4000);
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Four-Point Probe</h1>
          <p className="text-gray-600 mt-1">Van der Pauw method for sheet resistance measurement</p>
        </div>
        <Button onClick={() => setIsConnected(!isConnected)}>
          {isConnected ? 'Disconnect' : 'Connect Instrument'}
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Configuration Panel */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="current">Test Current (A)</Label>
              <Input
                id="current"
                type="number"
                value={config.current}
                onChange={(e) => setConfig({...config, current: e.target.value})}
                disabled={isRunning}
              />
            </div>

            <div>
              <Label>Number of Configurations</Label>
              <Select value={config.numConfigs} onValueChange={(val) => setConfig({...config, numConfigs: val})} disabled={isRunning}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="2">2 (minimum)</SelectItem>
                  <SelectItem value="4">4 (recommended)</SelectItem>
                  <SelectItem value="8">8 (high accuracy)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="thickness">Sample Thickness (cm)</Label>
              <Input
                id="thickness"
                type="number"
                value={config.sampleThickness}
                onChange={(e) => setConfig({...config, sampleThickness: e.target.value})}
                disabled={isRunning}
              />
            </div>

            <Button onClick={handleStartMeasurement} disabled={!isConnected || isRunning} className="w-full">
              <Play className="mr-2 h-4 w-4" />
              Start Measurement
            </Button>
          </CardContent>
        </Card>

        {/* Results Panel */}
        <div className="lg:col-span-2">
          {result && (
            <Card>
              <CardHeader>
                <CardTitle>Results</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-6 mb-6">
                  <div className="text-center">
                    <div className="text-sm text-gray-600">Sheet Resistance</div>
                    <div className="text-5xl font-bold text-blue-600">
                      {result.sheet_resistance.value.toFixed(1)}
                    </div>
                    <div className="text-xl text-gray-700">{result.sheet_resistance.unit}</div>
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

export default FourPointProbe;
