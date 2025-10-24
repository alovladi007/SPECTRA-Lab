import React, { useState } from ‘react’;
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from ‘@/components/ui/card’;
import { Button } from ‘@/components/ui/button’;
import { Input } from ‘@/components/ui/input’;
import { Label } from ‘@/components/ui/label’;
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from ‘@/components/ui/select’;
import { Badge } from ‘@/components/ui/badge’;
import { Alert, AlertDescription } from ‘@/components/ui/alert’;
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from ‘recharts’;
import { Sun, PlayCircle, Download, Zap, TrendingUp, AlertCircle } from ‘lucide-react’;

const SolarCellCharacterization = () => {
const [cellType, setCellType] = useState(‘silicon’);
const [isRunning, setIsRunning] = useState(false);
const [results, setResults] = useState(null);

const [config, setConfig] = useState({
area: 100, // cm²
irradiance: 1000, // W/m²
temperature: 25, // °C
spectrum: ‘AM1.5G’
});

// Mock I-V data
const ivData = results?.ivCurve || Array.from({length: 100}, (_, i) => {
const v = i * 0.007; // 0 to ~0.7V
const isc = 5.0;
const voc = 0.60;
const i = isc * (1 - Math.exp((v - voc) / 0.026));
const p = -v * i;
return { voltage: v, current: i, power: p };
});

const maxPowerPoint = ivData.reduce((max, point) =>
point.power > max.power ? point : max, ivData[0]
);

const handleRunMeasurement = async () => {
setIsRunning(true);

setTimeout(() => {
  setResults({
    isc: { value: 5.12, current_density_ma_cm2: 51.2 },
    voc: { value: 0.605 },
    mpp: { voltage: 0.485, current: 4.85, power: 2.35 },
    fill_factor: { value: 0.758, percent: 75.8 },
    efficiency: { value: 0.235, percent: 23.5 },
    rs: { value: 0.52 },
    rsh: { value: 1250 },
    quality_score: 92
  });
  setIsRunning(false);
}, 2000);

};

const getCellTypeInfo = (type) => {
const info = {
silicon: { name: ‘Crystalline Silicon’, maxEff: ‘26%’, color: ‘blue’ },
gaas: { name: ‘Gallium Arsenide’, maxEff: ‘29%’, color: ‘purple’ },
perovskite: { name: ‘Perovskite’, maxEff: ‘25%’, color: ‘pink’ },
organic: { name: ‘Organic’, maxEff: ‘18%’, color: ‘green’ }
};
return info[type] || info.silicon;
};

const cellInfo = getCellTypeInfo(cellType);

return (
<div className="w-full max-w-7xl mx-auto p-6 space-y-6">
{/* Header */}
<div className="flex justify-between items-center">
<div>
<h1 className="text-3xl font-bold flex items-center gap-2">
<Sun className="h-8 w-8 text-yellow-500" />
Solar Cell I-V Characterization
</h1>
<p className="text-gray-600 mt-1">Photovoltaic performance analysis</p>
</div>
<div className="flex gap-2">
<Badge variant="outline" className="text-lg px-4 py-2">
{cellInfo.name}
</Badge>
<Badge variant="outline" className="text-lg px-4 py-2">
<Sun className="h-4 w-4 mr-1" />
{config.irradiance} W/m²
</Badge>
</div>
</div>

  {/* Main Content */}
  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
    {/* Configuration Panel */}
    <Card className="lg:col-span-1">
      <CardHeader>
        <CardTitle>Configuration</CardTitle>
        <CardDescription>Cell and measurement parameters</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Cell Type */}
        <div className="space-y-2">
          <Label>Cell Type</Label>
          <Select value={cellType} onValueChange={setCellType}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="silicon">Crystalline Silicon</SelectItem>
              <SelectItem value="gaas">Gallium Arsenide</SelectItem>
              <SelectItem value="perovskite">Perovskite</SelectItem>
              <SelectItem value="organic">Organic</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-gray-500">
            Record efficiency: {cellInfo.maxEff}
          </p>
        </div>

        {/* Area */}
        <div className="space-y-2">
          <Label>Active Area (cm²)</Label>
          <Input 
            type="number" 
            value={config.area}
            onChange={(e) => setConfig({...config, area: parseFloat(e.target.value)})}
          />
        </div>

        {/* Irradiance */}
        <div className="space-y-2">
          <Label>Irradiance (W/m²)</Label>
          <Input 
            type="number" 
            value={config.irradiance}
            onChange={(e) => setConfig({...config, irradiance: parseFloat(e.target.value)})}
          />
          <div className="flex gap-2">
            <Button 
              size="sm" 
              variant="outline" 
              className="flex-1"
              onClick={() => setConfig({...config, irradiance: 1000})}
            >
              1 Sun
            </Button>
            <Button 
              size="sm" 
              variant="outline" 
              className="flex-1"
              onClick={() => setConfig({...config, irradiance: 500})}
            >
              0.5 Sun
            </Button>
          </div>
        </div>

        {/* Spectrum */}
        <div className="space-y-2">
          <Label>Spectrum</Label>
          <Select value={config.spectrum} onValueChange={(v) => setConfig({...config, spectrum: v})}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="AM1.5G">AM1.5G (Standard)</SelectItem>
              <SelectItem value="AM0">AM0 (Space)</SelectItem>
              <SelectItem value="AM1.5D">AM1.5D (Direct)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Temperature */}
        <div className="space-y-2">
          <Label>Temperature (°C)</Label>
          <Input 
            type="number" 
            value={config.temperature}
            onChange={(e) => setConfig({...config, temperature: parseFloat(e.target.value)})}
          />
        </div>

        {/* Action Buttons */}
        <div className="pt-4 space-y-2">
          <Button 
            className="w-full" 
            onClick={handleRunMeasurement}
            disabled={isRunning}
          >
            {isRunning ? (
              <>
                <div className="animate-spin mr-2 h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                Measuring...
              </>
            ) : (
              <>
                <PlayCircle className="mr-2 h-4 w-4" />
                Start Measurement
              </>
            )}
          </Button>
          
          {results && (
            <Button variant="outline" className="w-full">
              <Download className="mr-2 h-4 w-4" />
              Export Results
            </Button>
          )}
        </div>

        {/* Info Box */}
        <Alert className="mt-4">
          <Sun className="h-4 w-4" />
          <AlertDescription className="text-xs">
            <strong>Standard Test Conditions (STC):</strong><br />
            1000 W/m², AM1.5G spectrum, 25°C
          </AlertDescription>
        </Alert>
      </CardContent>
    </Card>

    {/* Results and Visualization */}
    <Card className="lg:col-span-2">
      <CardHeader>
        <CardTitle>Measurement Results</CardTitle>
        <CardDescription>
          {results ? `Efficiency: ${results.efficiency.percent.toFixed(2)}%` : 'Configure and start measurement'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {!results ? (
          <div className="h-96 flex items-center justify-center text-gray-400">
            <div className="text-center">
              <Sun className="h-16 w-16 mx-auto mb-4 opacity-50" />
              <p>No measurement data yet</p>
              <p className="text-sm mt-2">Illuminate the cell and start measurement</p>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {/* I-V Curve */}
            <div className="space-y-2">
              <h3 className="font-semibold text-sm">I-V and P-V Characteristics</h3>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={ivData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="voltage" 
                      label={{ value: 'Voltage (V)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      yAxisId="current"
                      label={{ value: 'Current (A)', angle: -90, position: 'insideLeft' }}
                    />
                    <YAxis 
                      yAxisId="power"
                      orientation="right"
                      label={{ value: 'Power (W)', angle: 90, position: 'insideRight' }}
                    />
                    <Tooltip />
                    <Legend />
                    <ReferenceLine 
                      x={maxPowerPoint.voltage} 
                      stroke="red" 
                      strokeDasharray="3 3"
                      label="MPP"
                    />
                    <Line 
                      yAxisId="current"
                      type="monotone" 
                      dataKey="current" 
                      stroke="#2563eb" 
                      dot={false}
                      name="Current"
                    />
                    <Line 
                      yAxisId="power"
                      type="monotone" 
                      dataKey="power" 
                      stroke="#dc2626" 
                      dot={false}
                      name="Power"
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Key Performance Indicators */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Card className="border-blue-200 bg-blue-50">
                <CardContent className="pt-4">
                  <div className="text-xs text-gray-600 mb-1">Isc (Short-Circuit)</div>
                  <div className="text-2xl font-bold text-blue-700">{results.isc.value.toFixed(2)} A</div>
                  <div className="text-xs text-gray-500 mt-1">
                    {results.isc.current_density_ma_cm2.toFixed(1)} mA/cm²
                  </div>
                </CardContent>
              </Card>

              <Card className="border-green-200 bg-green-50">
                <CardContent className="pt-4">
                  <div className="text-xs text-gray-600 mb-1">Voc (Open-Circuit)</div>
                  <div className="text-2xl font-bold text-green-700">{results.voc.value.toFixed(3)} V</div>
                  <div className="text-xs text-gray-500 mt-1">Voltage</div>
                </CardContent>
              </Card>

              <Card className="border-purple-200 bg-purple-50">
                <CardContent className="pt-4">
                  <div className="text-xs text-gray-600 mb-1">Fill Factor</div>
                  <div className="text-2xl font-bold text-purple-700">{results.fill_factor.percent.toFixed(1)}%</div>
                  <div className="text-xs text-gray-500 mt-1">{results.fill_factor.value.toFixed(3)}</div>
                </CardContent>
              </Card>

              <Card className="border-yellow-200 bg-yellow-50">
                <CardContent className="pt-4">
                  <div className="text-xs text-gray-600 mb-1 flex items-center gap-1">
                    <Zap className="h-3 w-3" />
                    Efficiency (η)
                  </div>
                  <div className="text-2xl font-bold text-yellow-700">{results.efficiency.percent.toFixed(2)}%</div>
                  <div className="text-xs text-gray-500 mt-1">PCE</div>
                </CardContent>
              </Card>
            </div>

            {/* Maximum Power Point */}
            <Card className="border-red-200 bg-red-50">
              <CardContent className="pt-4">
                <div className="flex justify-between items-center">
                  <div>
                    <div className="text-sm font-semibold text-red-900 mb-2">Maximum Power Point (MPP)</div>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <div className="text-gray-600">Voltage</div>
                        <div className="font-semibold">{results.mpp.voltage.toFixed(3)} V</div>
                      </div>
                      <div>
                        <div className="text-gray-600">Current</div>
                        <div className="font-semibold">{results.mpp.current.toFixed(3)} A</div>
                      </div>
                      <div>
                        <div className="text-gray-600">Power</div>
                        <div className="font-semibold">{results.mpp.power.toFixed(3)} W</div>
                      </div>
                    </div>
                  </div>
                  <Zap className="h-12 w-12 text-red-500" />
                </div>
              </CardContent>
            </Card>

            {/* Series and Shunt Resistance */}
            <div className="grid grid-cols-2 gap-4">
              <Card>
                <CardContent className="pt-4">
                  <div className="text-sm text-gray-600 mb-1">Series Resistance (Rs)</div>
                  <div className="text-xl font-bold">{results.rs.value.toFixed(2)} Ω</div>
                  <div className="text-xs text-gray-500 mt-1">
                    {results.rs.value < 1 ? '✓ Low resistance' : '⚠ Check contacts'}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-4">
                  <div className="text-sm text-gray-600 mb-1">Shunt Resistance (Rsh)</div>
                  <div className="text-xl font-bold">{results.rsh.value.toFixed(0)} Ω</div>
                  <div className="text-xs text-gray-500 mt-1">
                    {results.rsh.value > 500 ? '✓ High resistance' : '⚠ Check for leakage'}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Quality Score */}
            <Alert className={
              results.quality_score >= 80 ? 'border-green-200 bg-green-50' :
              results.quality_score >= 60 ? 'border-yellow-200 bg-yellow-50' :
              'border-red-200 bg-red-50'
            }>
              <TrendingUp className="h-4 w-4" />
              <AlertDescription>
                <div className="flex justify-between items-center">
                  <span className="font-semibold">Quality Score: {results.quality_score}/100</span>
                  {results.quality_score >= 80 ? (
                    <Badge className="bg-green-600">Excellent</Badge>
                  ) : results.quality_score >= 60 ? (
                    <Badge className="bg-yellow-600">Good</Badge>
                  ) : (
                    <Badge className="bg-red-600">Needs Review</Badge>
                  )}
                </div>
              </AlertDescription>
            </Alert>

            {/* Detailed Parameters */}
            <details className="border rounded-lg">
              <summary className="px-4 py-2 bg-gray-50 cursor-pointer hover:bg-gray-100 font-semibold">
                View All Parameters
              </summary>
              <div className="p-4">
                <table className="w-full text-sm">
                  <tbody className="divide-y">
                    <tr>
                      <td className="py-2">Short-Circuit Current Density</td>
                      <td className="py-2 font-mono text-right">{results.isc.current_density_ma_cm2.toFixed(2)} mA/cm²</td>
                    </tr>
                    <tr>
                      <td className="py-2">Open-Circuit Voltage</td>
                      <td className="py-2 font-mono text-right">{results.voc.value.toFixed(3)} V</td>
                    </tr>
                    <tr>
                      <td className="py-2">MPP Voltage</td>
                      <td className="py-2 font-mono text-right">{results.mpp.voltage.toFixed(3)} V</td>
                    </tr>
                    <tr>
                      <td className="py-2">MPP Current</td>
                      <td className="py-2 font-mono text-right">{results.mpp.current.toFixed(3)} A</td>
                    </tr>
                    <tr>
                      <td className="py-2">Maximum Power</td>
                      <td className="py-2 font-mono text-right">{results.mpp.power.toFixed(3)} W</td>
                    </tr>
                    <tr>
                      <td className="py-2">Fill Factor</td>
                      <td className="py-2 font-mono text-right">{results.fill_factor.percent.toFixed(2)}%</td>
                    </tr>
                    <tr>
                      <td className="py-2">Power Conversion Efficiency</td>
                      <td className="py-2 font-mono text-right">{results.efficiency.percent.toFixed(2)}%</td>
                    </tr>
                    <tr>
                      <td className="py-2">Incident Irradiance</td>
                      <td className="py-2 font-mono text-right">{config.irradiance} W/m²</td>
                    </tr>
                    <tr>
                      <td className="py-2">Cell Area</td>
                      <td className="py-2 font-mono text-right">{config.area} cm²</td>
                    </tr>
                    <tr>
                      <td className="py-2">Temperature</td>
                      <td className="py-2 font-mono text-right">{config.temperature} °C</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </details>
          </div>
        )}
      </CardContent>
    </Card>
  </div>

  {/* Status Bar */}
  {isRunning && (
    <Alert>
      <Sun className="h-4 w-4 animate-pulse" />
      <AlertDescription>
        Measurement in progress... Sweeping voltage and recording I-V curve
      </AlertDescription>
    </Alert>
  )}
</div>

);
};

export default SolarCellCharacterization;