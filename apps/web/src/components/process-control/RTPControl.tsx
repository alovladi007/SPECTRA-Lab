"""React component for Rapid Thermal Processing control and monitoring."""

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  LineChart, Line, Area, AreaChart, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from 'recharts';
import { 
  Play, Square, Thermometer, Wind, Gauge, TrendingUp, 
  AlertTriangle, Flame, Timer, Settings
} from 'lucide-react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useQuery, useMutation } from '@tanstack/react-query';
import { toast } from 'sonner';

interface RecipeSegment {
  time_s: number;
  T_C: number;
  ramp_Cps: number;
  dwell_s: number;
}

interface RTPProfile {
  id: number;
  name: string;
  recipe_curve: RecipeSegment[];
  ambient_gas: string;
  pressure_Torr: number;
  emissivity: number;
}

interface RTPStatus {
  running: boolean;
  recipeActive: boolean;
  currentStep: number;
  temperature: number;
  setpoint: number;
  lampPower: number[];
  pressure: number;
  gasFlows: Record<string, number>;
}

interface TelemetryPoint {
  timestamp: string;
  setpoint_T_C: number;
  pyrometer_T_C: number;
  lamp_power_avg: number;
}

export const RTPControl: React.FC = () => {
  const [profile, setProfile] = useState<RTPProfile>({
    id: 0,
    name: 'Custom Recipe',
    recipe_curve: [
      { time_s: 0, T_C: 25, ramp_Cps: 50, dwell_s: 0 },
      { time_s: 10, T_C: 500, ramp_Cps: 50, dwell_s: 30 },
      { time_s: 40, T_C: 1000, ramp_Cps: 100, dwell_s: 60 },
      { time_s: 100, T_C: 25, ramp_Cps: 20, dwell_s: 0 },
    ],
    ambient_gas: 'N2',
    pressure_Torr: 760,
    emissivity: 0.7,
  });

  const [status, setStatus] = useState<RTPStatus>({
    running: false,
    recipeActive: false,
    currentStep: 0,
    temperature: 25,
    setpoint: 25,
    lampPower: new Array(12).fill(0),
    pressure: 760,
    gasFlows: { N2: 0, O2: 0 },
  });

  const [telemetryData, setTelemetryData] = useState<TelemetryPoint[]>([]);
  const [editingSegment, setEditingSegment] = useState<number | null>(null);

  // WebSocket for real-time data
  const { data: wsData, isConnected } = useWebSocket('/api/v1/rtp/ws');

  // API queries
  const { data: profiles } = useQuery({
    queryKey: ['rtp-profiles'],
    queryFn: async () => {
      const res = await fetch('/api/v1/rtp/profiles');
      return res.json();
    },
  });

  const startRecipe = useMutation({
    mutationFn: async (profileId: number) => {
      const res = await fetch('/api/v1/rtp/control/start-recipe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile_id: profileId,
          run_id: Date.now(),
        }),
      });
      return res.json();
    },
    onSuccess: () => {
      toast.success('Recipe started');
      setStatus(prev => ({ ...prev, running: true, recipeActive: true }));
    },
    onError: () => {
      toast.error('Failed to start recipe');
    },
  });

  const stopRecipe = useMutation({
    mutationFn: async () => {
      const res = await fetch('/api/v1/rtp/control/stop-recipe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: Date.now() }),
      });
      return res.json();
    },
    onSuccess: () => {
      toast.success('Recipe stopped');
      setStatus(prev => ({ ...prev, running: false, recipeActive: false }));
    },
  });

  // Update telemetry from WebSocket
  useEffect(() => {
    if (wsData && wsData.module === 'rtp' && wsData.type === 'telemetry') {
      const point: TelemetryPoint = {
        timestamp: wsData.timestamp,
        setpoint_T_C: wsData.data.setpoint_T_C,
        pyrometer_T_C: wsData.data.pyrometer_T_C,
        lamp_power_avg: wsData.data.lamp_power_pct.reduce((a: number, b: number) => a + b, 0) / 12,
      };

      setTelemetryData(prev => {
        const updated = [...prev, point];
        return updated.slice(-200); // Keep last 200 points
      });

      setStatus(prev => ({
        ...prev,
        temperature: point.pyrometer_T_C,
        setpoint: point.setpoint_T_C,
        lampPower: wsData.data.lamp_power_pct,
        pressure: wsData.data.chamber_pressure_Torr,
        gasFlows: wsData.data.flow_sccm,
      }));
    }
  }, [wsData]);

  const handleStart = () => {
    // Save profile first if needed
    startRecipe.mutate(profile.id || 1);
  };

  const handleStop = () => {
    stopRecipe.mutate();
  };

  const addSegment = () => {
    const lastSegment = profile.recipe_curve[profile.recipe_curve.length - 1];
    const newSegment: RecipeSegment = {
      time_s: lastSegment.time_s + lastSegment.dwell_s + 10,
      T_C: 500,
      ramp_Cps: 50,
      dwell_s: 30,
    };
    setProfile(prev => ({
      ...prev,
      recipe_curve: [...prev.recipe_curve, newSegment],
    }));
  };

  const removeSegment = (index: number) => {
    setProfile(prev => ({
      ...prev,
      recipe_curve: prev.recipe_curve.filter((_, i) => i !== index),
    }));
  };

  const updateSegment = (index: number, field: keyof RecipeSegment, value: number) => {
    setProfile(prev => ({
      ...prev,
      recipe_curve: prev.recipe_curve.map((seg, i) =>
        i === index ? { ...seg, [field]: value } : seg
      ),
    }));
  };

  // Generate recipe visualization data
  const recipeChartData = [];
  let currentTime = 0;
  for (const segment of profile.recipe_curve) {
    // Add ramp
    const rampTime = Math.abs(segment.T_C - (recipeChartData[recipeChartData.length - 1]?.temperature || 25)) / segment.ramp_Cps;
    recipeChartData.push({
      time: currentTime,
      temperature: recipeChartData[recipeChartData.length - 1]?.temperature || 25,
    });
    currentTime += rampTime;
    recipeChartData.push({
      time: currentTime,
      temperature: segment.T_C,
    });
    
    // Add dwell
    if (segment.dwell_s > 0) {
      currentTime += segment.dwell_s;
      recipeChartData.push({
        time: currentTime,
        temperature: segment.T_C,
      });
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold">Rapid Thermal Processing</h2>
          <p className="text-muted-foreground">Configure and monitor RTP temperature profiles</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={isConnected ? "default" : "secondary"}>
            {isConnected ? "Connected" : "Disconnected"}
          </Badge>
          {status.running && (
            <Badge variant="destructive" className="animate-pulse">
              <Flame className="w-4 h-4 mr-1" />
              HEATING
            </Badge>
          )}
        </div>
      </div>

      <Tabs defaultValue="control" className="space-y-4">
        <TabsList>
          <TabsTrigger value="control">Control</TabsTrigger>
          <TabsTrigger value="recipe">Recipe Editor</TabsTrigger>
          <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
          <TabsTrigger value="zones">Zone Control</TabsTrigger>
        </TabsList>

        <TabsContent value="control" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Temperature Control */}
            <Card>
              <CardHeader>
                <CardTitle>Temperature Control</CardTitle>
                <CardDescription>Current thermal status</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-center py-4">
                  <div className="text-5xl font-bold text-primary">
                    {status.temperature.toFixed(1)}°C
                  </div>
                  <div className="text-sm text-muted-foreground mt-2">
                    Setpoint: {status.setpoint.toFixed(1)}°C
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium flex items-center gap-2">
                      <Thermometer className="w-4 h-4" />
                      Temperature Error
                    </span>
                    <span className={`text-sm font-mono ${
                      Math.abs(status.temperature - status.setpoint) > 5 ? 'text-red-500' : 'text-green-500'
                    }`}>
                      {(status.temperature - status.setpoint).toFixed(1)}°C
                    </span>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium flex items-center gap-2">
                      <Flame className="w-4 h-4" />
                      Average Lamp Power
                    </span>
                    <span className="text-sm font-mono">
                      {(status.lampPower.reduce((a, b) => a + b, 0) / 12).toFixed(1)}%
                    </span>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium flex items-center gap-2">
                      <Gauge className="w-4 h-4" />
                      Chamber Pressure
                    </span>
                    <span className="text-sm font-mono">
                      {status.pressure.toFixed(1)} Torr
                    </span>
                  </div>

                  <div className="space-y-2">
                    <Label className="flex items-center gap-2">
                      <Wind className="w-4 h-4" />
                      Gas Flows (sccm)
                    </Label>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(status.gasFlows).map(([gas, flow]) => (
                        <div key={gas} className="flex justify-between p-2 bg-secondary rounded">
                          <span className="text-sm">{gas}</span>
                          <span className="text-sm font-mono">{flow.toFixed(1)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Process Parameters */}
            <Card>
              <CardHeader>
                <CardTitle>Process Parameters</CardTitle>
                <CardDescription>Configure RTP settings</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="ambient-gas">Ambient Gas</Label>
                  <Select
                    value={profile.ambient_gas}
                    onValueChange={(value) => setProfile(prev => ({ ...prev, ambient_gas: value }))}
                    disabled={status.running}
                  >
                    <SelectTrigger id="ambient-gas">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="N2">Nitrogen (N₂)</SelectItem>
                      <SelectItem value="O2">Oxygen (O₂)</SelectItem>
                      <SelectItem value="Ar">Argon (Ar)</SelectItem>
                      <SelectItem value="H2">Hydrogen (H₂)</SelectItem>
                      <SelectItem value="NH3">Ammonia (NH₃)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="pressure">Pressure (Torr)</Label>
                  <div className="flex items-center space-x-4">
                    <Slider
                      id="pressure"
                      min={1}
                      max={760}
                      step={1}
                      value={[profile.pressure_Torr]}
                      onValueChange={([value]) => setProfile(prev => ({ ...prev, pressure_Torr: value }))}
                      disabled={status.running}
                      className="flex-1"
                    />
                    <Input
                      type="number"
                      value={profile.pressure_Torr}
                      onChange={(e) => setProfile(prev => ({ ...prev, pressure_Torr: Number(e.target.value) }))}
                      disabled={status.running}
                      className="w-20"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="emissivity">Emissivity</Label>
                  <div className="flex items-center space-x-4">
                    <Slider
                      id="emissivity"
                      min={0.1}
                      max={1.0}
                      step={0.01}
                      value={[profile.emissivity]}
                      onValueChange={([value]) => setProfile(prev => ({ ...prev, emissivity: value }))}
                      disabled={status.running}
                      className="flex-1"
                    />
                    <Input
                      type="number"
                      value={profile.emissivity}
                      onChange={(e) => setProfile(prev => ({ ...prev, emissivity: Number(e.target.value) }))}
                      disabled={status.running}
                      className="w-20"
                    />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Typical: Si (0.7), SiO₂ (0.9), Metal (0.2-0.4)
                  </p>
                </div>

                <div className="flex gap-2 pt-4">
                  {!status.running ? (
                    <Button
                      onClick={handleStart}
                      className="flex-1"
                      disabled={startRecipe.isPending}
                    >
                      <Play className="w-4 h-4 mr-2" />
                      Start Recipe
                    </Button>
                  ) : (
                    <Button
                      onClick={handleStop}
                      variant="destructive"
                      className="flex-1"
                      disabled={stopRecipe.isPending}
                    >
                      <Square className="w-4 h-4 mr-2" />
                      Stop Recipe
                    </Button>
                  )}
                </div>

                {status.running && (
                  <Alert>
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>
                      High temperature process active. Do not open chamber.
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Recipe Progress */}
          {status.recipeActive && (
            <Card>
              <CardHeader>
                <CardTitle>Recipe Progress</CardTitle>
                <CardDescription>
                  Step {status.currentStep + 1} of {profile.recipe_curve.length}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Progress 
                  value={(status.currentStep + 1) / profile.recipe_curve.length * 100} 
                  className="h-3"
                />
                <div className="mt-4 grid grid-cols-4 gap-4 text-center">
                  <div>
                    <p className="text-sm text-muted-foreground">Current Step</p>
                    <p className="text-xl font-bold">{status.currentStep + 1}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Target Temp</p>
                    <p className="text-xl font-bold">
                      {profile.recipe_curve[status.currentStep]?.T_C || 0}°C
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Ramp Rate</p>
                    <p className="text-xl font-bold">
                      {profile.recipe_curve[status.currentStep]?.ramp_Cps || 0}°C/s
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Dwell Time</p>
                    <p className="text-xl font-bold">
                      {profile.recipe_curve[status.currentStep]?.dwell_s || 0}s
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="recipe" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recipe Editor</CardTitle>
              <CardDescription>Design temperature profile</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Recipe Visualization */}
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={recipeChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }} />
                    <YAxis label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Area
                      type="linear"
                      dataKey="temperature"
                      stroke="#8884d8"
                      fill="#8884d8"
                      fillOpacity={0.3}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Recipe Segments */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label>Recipe Segments</Label>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={addSegment}
                    disabled={status.running}
                  >
                    Add Segment
                  </Button>
                </div>
                
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {profile.recipe_curve.map((segment, index) => (
                    <div
                      key={index}
                      className="p-3 border rounded-lg space-y-2"
                    >
                      <div className="flex justify-between items-center">
                        <span className="font-medium">Segment {index + 1}</span>
                        {profile.recipe_curve.length > 1 && (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => removeSegment(index)}
                            disabled={status.running}
                          >
                            Remove
                          </Button>
                        )}
                      </div>
                      
                      {editingSegment === index ? (
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <Label className="text-xs">Temp (°C)</Label>
                            <Input
                              type="number"
                              value={segment.T_C}
                              onChange={(e) => updateSegment(index, 'T_C', Number(e.target.value))}
                              className="h-8"
                            />
                          </div>
                          <div>
                            <Label className="text-xs">Ramp (°C/s)</Label>
                            <Input
                              type="number"
                              value={segment.ramp_Cps}
                              onChange={(e) => updateSegment(index, 'ramp_Cps', Number(e.target.value))}
                              className="h-8"
                            />
                          </div>
                          <div>
                            <Label className="text-xs">Dwell (s)</Label>
                            <Input
                              type="number"
                              value={segment.dwell_s}
                              onChange={(e) => updateSegment(index, 'dwell_s', Number(e.target.value))}
                              className="h-8"
                            />
                          </div>
                          <div className="flex items-end">
                            <Button
                              size="sm"
                              onClick={() => setEditingSegment(null)}
                              className="w-full"
                            >
                              Done
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <div
                          className="grid grid-cols-3 gap-2 text-sm cursor-pointer hover:bg-accent rounded p-2"
                          onClick={() => setEditingSegment(index)}
                        >
                          <div>
                            <span className="text-muted-foreground">Temp:</span> {segment.T_C}°C
                          </div>
                          <div>
                            <span className="text-muted-foreground">Ramp:</span> {segment.ramp_Cps}°C/s
                          </div>
                          <div>
                            <span className="text-muted-foreground">Dwell:</span> {segment.dwell_s}s
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="monitoring" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Real-time Monitoring</CardTitle>
              <CardDescription>Live temperature and control data</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={telemetryData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="timestamp"
                      tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                    />
                    <YAxis />
                    <Tooltip
                      labelFormatter={(value) => new Date(value).toLocaleString()}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="setpoint_T_C"
                      stroke="#8884d8"
                      name="Setpoint (°C)"
                      dot={false}
                      strokeWidth={2}
                    />
                    <Line
                      type="monotone"
                      dataKey="pyrometer_T_C"
                      stroke="#82ca9d"
                      name="Temperature (°C)"
                      dot={false}
                      strokeWidth={2}
                    />
                    <Line
                      type="monotone"
                      dataKey="lamp_power_avg"
                      stroke="#ffc658"
                      name="Lamp Power (%)"
                      dot={false}
                      yAxisId="right"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="zones" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Multi-Zone Lamp Control</CardTitle>
              <CardDescription>Individual zone power distribution</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4">
                {status.lampPower.map((power, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <Label className="text-sm">Zone {index + 1}</Label>
                      <span className="text-sm font-mono">{power.toFixed(1)}%</span>
                    </div>
                    <Progress value={power} className="h-2" />
                  </div>
                ))}
              </div>
              
              <div className="mt-6 p-4 bg-secondary rounded-lg">
                <h4 className="font-medium mb-2">Zone Configuration</h4>
                <div className="grid grid-cols-3 gap-2 text-sm">
                  <div>
                    <span className="text-muted-foreground">Inner (1-4):</span> Center heating
                  </div>
                  <div>
                    <span className="text-muted-foreground">Middle (5-8):</span> Mid radius
                  </div>
                  <div>
                    <span className="text-muted-foreground">Outer (9-12):</span> Edge compensation
                  </div>
                </div>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Temperature Uniformity</p>
                  <p className="text-2xl font-bold">±1.5°C</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Total Power</p>
                  <p className="text-2xl font-bold">
                    {(status.lampPower.reduce((a, b) => a + b, 0) / 12 * 60).toFixed(1)} kW
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default RTPControl;
