"""React component for Ion Implantation control and monitoring."""

import React, { useState, useEffect, useCallback } from 'react';
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
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Play, Square, AlertCircle, Activity, Zap, Target, TrendingUp } from 'lucide-react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useQuery, useMutation } from '@tanstack/react-query';
import { toast } from 'sonner';

interface BeamParameters {
  ionSpecies: string;
  energy: number;
  current: number;
  dose: number;
  tilt: number;
  twist: number;
}

interface ImplantStatus {
  running: boolean;
  beamOn: boolean;
  currentDose: number;
  targetDose: number;
  beamCurrent: number;
  pressure: number;
  interlocks: Record<string, boolean>;
}

interface TelemetryPoint {
  timestamp: string;
  beamCurrent: number;
  pressure: number;
  dose: number;
  energy: number;
}

export const IonImplantationControl: React.FC = () => {
  const [parameters, setParameters] = useState<BeamParameters>({
    ionSpecies: 'P',
    energy: 100,
    current: 1.0,
    dose: 1e15,
    tilt: 0,
    twist: 0,
  });

  const [status, setStatus] = useState<ImplantStatus>({
    running: false,
    beamOn: false,
    currentDose: 0,
    targetDose: 1e15,
    beamCurrent: 0,
    pressure: 1e-6,
    interlocks: {},
  });

  const [telemetryData, setTelemetryData] = useState<TelemetryPoint[]>([]);
  const [uniformityMap, setUniformityMap] = useState<number[][]>([]);

  // WebSocket connection for real-time data
  const { data: wsData, isConnected } = useWebSocket('/api/v1/implant/ws');

  // API queries
  const { data: profiles } = useQuery({
    queryKey: ['implant-profiles'],
    queryFn: async () => {
      const res = await fetch('/api/v1/implant/profiles');
      return res.json();
    },
  });

  const startImplant = useMutation({
    mutationFn: async (params: BeamParameters) => {
      const res = await fetch('/api/v1/implant/control/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          run_id: Date.now(),
          target_dose_cm2: params.dose,
        }),
      });
      return res.json();
    },
    onSuccess: () => {
      toast.success('Implantation started');
      setStatus(prev => ({ ...prev, running: true, beamOn: true }));
    },
    onError: () => {
      toast.error('Failed to start implantation');
    },
  });

  const stopImplant = useMutation({
    mutationFn: async () => {
      const res = await fetch('/api/v1/implant/control/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: Date.now() }),
      });
      return res.json();
    },
    onSuccess: () => {
      toast.success('Implantation stopped');
      setStatus(prev => ({ ...prev, running: false, beamOn: false }));
    },
  });

  // Update telemetry from WebSocket
  useEffect(() => {
    if (wsData && wsData.module === 'implant' && wsData.type === 'telemetry') {
      const point: TelemetryPoint = {
        timestamp: wsData.timestamp,
        beamCurrent: wsData.data.beam_current_mA,
        pressure: wsData.data.pressure_mTorr,
        dose: wsData.data.dose_count_C_cm2,
        energy: wsData.data.accel_voltage_kV,
      };

      setTelemetryData(prev => {
        const updated = [...prev, point];
        return updated.slice(-100); // Keep last 100 points
      });

      setStatus(prev => ({
        ...prev,
        beamCurrent: point.beamCurrent,
        pressure: point.pressure,
        currentDose: point.dose,
      }));
    }
  }, [wsData]);

  const handleStart = () => {
    startImplant.mutate(parameters);
  };

  const handleStop = () => {
    stopImplant.mutate();
  };

  const doseProgress = (status.currentDose / status.targetDose) * 100;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold">Ion Implantation Control</h2>
          <p className="text-muted-foreground">Configure and monitor ion beam parameters</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={isConnected ? "default" : "secondary"}>
            {isConnected ? "Connected" : "Disconnected"}
          </Badge>
          {status.running && (
            <Badge variant="destructive" className="animate-pulse">
              <Activity className="w-4 h-4 mr-1" />
              BEAM ON
            </Badge>
          )}
        </div>
      </div>

      <Tabs defaultValue="control" className="space-y-4">
        <TabsList>
          <TabsTrigger value="control">Control</TabsTrigger>
          <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
          <TabsTrigger value="uniformity">Uniformity</TabsTrigger>
          <TabsTrigger value="profiles">Profiles</TabsTrigger>
        </TabsList>

        <TabsContent value="control" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Beam Parameters */}
            <Card>
              <CardHeader>
                <CardTitle>Beam Parameters</CardTitle>
                <CardDescription>Configure ion beam settings</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="ion-species">Ion Species</Label>
                  <Select
                    value={parameters.ionSpecies}
                    onValueChange={(value) => setParameters(prev => ({ ...prev, ionSpecies: value }))}
                    disabled={status.running}
                  >
                    <SelectTrigger id="ion-species">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="B">Boron (B)</SelectItem>
                      <SelectItem value="P">Phosphorus (P)</SelectItem>
                      <SelectItem value="As">Arsenic (As)</SelectItem>
                      <SelectItem value="Sb">Antimony (Sb)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="energy">Energy (keV)</Label>
                  <div className="flex items-center space-x-4">
                    <Slider
                      id="energy"
                      min={10}
                      max={500}
                      step={10}
                      value={[parameters.energy]}
                      onValueChange={([value]) => setParameters(prev => ({ ...prev, energy: value }))}
                      disabled={status.running}
                      className="flex-1"
                    />
                    <Input
                      type="number"
                      value={parameters.energy}
                      onChange={(e) => setParameters(prev => ({ ...prev, energy: Number(e.target.value) }))}
                      disabled={status.running}
                      className="w-20"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="current">Beam Current (mA)</Label>
                  <div className="flex items-center space-x-4">
                    <Slider
                      id="current"
                      min={0.1}
                      max={10}
                      step={0.1}
                      value={[parameters.current]}
                      onValueChange={([value]) => setParameters(prev => ({ ...prev, current: value }))}
                      disabled={status.running}
                      className="flex-1"
                    />
                    <Input
                      type="number"
                      value={parameters.current}
                      onChange={(e) => setParameters(prev => ({ ...prev, current: Number(e.target.value) }))}
                      disabled={status.running}
                      className="w-20"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="dose">Target Dose (ions/cm²)</Label>
                  <Input
                    id="dose"
                    type="number"
                    value={parameters.dose}
                    onChange={(e) => setParameters(prev => ({ ...prev, dose: Number(e.target.value) }))}
                    disabled={status.running}
                    placeholder="1e15"
                    className="font-mono"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="tilt">Tilt (deg)</Label>
                    <Input
                      id="tilt"
                      type="number"
                      value={parameters.tilt}
                      onChange={(e) => setParameters(prev => ({ ...prev, tilt: Number(e.target.value) }))}
                      disabled={status.running}
                      min={-10}
                      max={60}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="twist">Twist (deg)</Label>
                    <Input
                      id="twist"
                      type="number"
                      value={parameters.twist}
                      onChange={(e) => setParameters(prev => ({ ...prev, twist: Number(e.target.value) }))}
                      disabled={status.running}
                      min={-180}
                      max={180}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Status and Control */}
            <Card>
              <CardHeader>
                <CardTitle>System Status</CardTitle>
                <CardDescription>Current implanter status</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Beam Current</span>
                    <span className="text-sm font-mono">{status.beamCurrent.toFixed(2)} mA</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Chamber Pressure</span>
                    <span className="text-sm font-mono">{status.pressure.toExponential(2)} Torr</span>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Dose Progress</span>
                      <span className="text-sm font-mono">
                        {status.currentDose.toExponential(2)} / {status.targetDose.toExponential(2)}
                      </span>
                    </div>
                    <Progress value={doseProgress} className="h-2" />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Interlocks</Label>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(status.interlocks).map(([key, value]) => (
                      <Badge
                        key={key}
                        variant={value ? "default" : "destructive"}
                        className="text-xs"
                      >
                        {key}
                      </Badge>
                    ))}
                  </div>
                </div>

                <div className="flex gap-2">
                  {!status.running ? (
                    <Button
                      onClick={handleStart}
                      className="flex-1"
                      disabled={startImplant.isPending}
                    >
                      <Play className="w-4 h-4 mr-2" />
                      Start Implant
                    </Button>
                  ) : (
                    <Button
                      onClick={handleStop}
                      variant="destructive"
                      className="flex-1"
                      disabled={stopImplant.isPending}
                    >
                      <Square className="w-4 h-4 mr-2" />
                      Stop Implant
                    </Button>
                  )}
                </div>

                {status.running && (
                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      Ion beam is active. Do not open chamber door.
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Projected Range Calculation */}
          <Card>
            <CardHeader>
              <CardTitle>SRIM Calculation</CardTitle>
              <CardDescription>Projected range and straggle estimates</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Projected Range (Rp)</p>
                  <p className="text-2xl font-bold">
                    {(parameters.energy * 0.3).toFixed(1)} nm
                  </p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Straggle (ΔRp)</p>
                  <p className="text-2xl font-bold">
                    {(parameters.energy * 0.1).toFixed(1)} nm
                  </p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Peak Concentration</p>
                  <p className="text-2xl font-bold">
                    {(parameters.dose / 1e15).toFixed(1)}e20 /cm³
                  </p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Implant Time</p>
                  <p className="text-2xl font-bold">
                    {(parameters.dose / (parameters.current * 6.24e15)).toFixed(1)} s
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="monitoring" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Real-time Telemetry</CardTitle>
              <CardDescription>Live beam parameters and system metrics</CardDescription>
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
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip
                      labelFormatter={(value) => new Date(value).toLocaleString()}
                    />
                    <Legend />
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey="beamCurrent"
                      stroke="#8884d8"
                      name="Beam Current (mA)"
                      dot={false}
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="dose"
                      stroke="#82ca9d"
                      name="Dose (ions/cm²)"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="uniformity" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Beam Uniformity Map</CardTitle>
              <CardDescription>Spatial distribution across wafer</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="aspect-square max-w-md mx-auto">
                {/* Placeholder for uniformity heatmap */}
                <div className="w-full h-full bg-gradient-radial from-blue-500 to-blue-200 rounded-full opacity-50" />
              </div>
              <div className="mt-4 grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Uniformity</p>
                  <p className="text-xl font-bold">98.5%</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Sigma</p>
                  <p className="text-xl font-bold">1.2%</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="profiles" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Saved Profiles</CardTitle>
              <CardDescription>Load predefined implant recipes</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {profiles?.map((profile: any) => (
                  <div
                    key={profile.id}
                    className="flex justify-between items-center p-3 border rounded-lg hover:bg-accent cursor-pointer"
                    onClick={() => {
                      setParameters({
                        ionSpecies: profile.ion_species,
                        energy: profile.energy_keV,
                        current: 1.0,
                        dose: profile.dose_cm2,
                        tilt: profile.tilt_deg,
                        twist: profile.twist_deg,
                      });
                    }}
                  >
                    <div>
                      <p className="font-medium">
                        {profile.ion_species} @ {profile.energy_keV} keV
                      </p>
                      <p className="text-sm text-muted-foreground">
                        Dose: {profile.dose_cm2.toExponential(1)} ions/cm²
                      </p>
                    </div>
                    <Button size="sm" variant="outline">
                      Load
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default IonImplantationControl;
