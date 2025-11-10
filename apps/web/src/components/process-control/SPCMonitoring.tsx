import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, Dot
} from 'recharts';
import { AlertTriangle, TrendingUp, CheckCircle, XCircle, Bell } from 'lucide-react';

interface SPCPoint {
  id: number;
  timestamp: string;
  value: number;
  moving_range?: number;
  ewma_value?: number;
  violations?: string[];
}

interface SPCSeries {
  id: number;
  name: string;
  parameter: string;
  chart_type: string;
  control_limits: {
    UCL: number;
    CL: number;
    LCL: number;
  };
  spec_limits?: {
    USL: number;
    LSL: number;
  };
}

interface SPCAlert {
  id: number;
  series_id: number;
  alert_type: string;
  severity: string;
  rule_violated: string;
  description: string;
  acknowledged: boolean;
  created_at: string;
}

export const SPCMonitoring: React.FC = () => {
  const [selectedSeries, setSelectedSeries] = useState<number | null>(null);
  const [timeRange, setTimeRange] = useState('1h');
  const [series, setSeries] = useState<SPCSeries[]>([]);
  const [points, setPoints] = useState<SPCPoint[]>([]);
  const [alerts, setAlerts] = useState<SPCAlert[]>([]);

  // Fetch SPC series
  useEffect(() => {
    fetch('/api/v1/spc/series')
      .then(res => res.json())
      .then(data => setSeries(data.series || []))
      .catch(err => console.error('Failed to load series:', err));
  }, []);

  // Fetch SPC points for selected series
  useEffect(() => {
    if (!selectedSeries) return;

    const loadPoints = () => {
      fetch(`/api/v1/spc/points/${selectedSeries}?limit=100`)
        .then(res => res.json())
        .then(data => setPoints(data.points || []))
        .catch(err => console.error('Failed to load points:', err));
    };

    loadPoints();
    const interval = setInterval(loadPoints, 5000); // Auto-refresh every 5 seconds

    return () => clearInterval(interval);
  }, [selectedSeries]);

  // Fetch SPC alerts
  useEffect(() => {
    const loadAlerts = () => {
      fetch('/api/v1/spc/alerts?acknowledged=false&limit=50')
        .then(res => res.json())
        .then(data => setAlerts(data.alerts || []))
        .catch(err => console.error('Failed to load alerts:', err));
    };

    loadAlerts();
    const interval = setInterval(loadAlerts, 10000); // Auto-refresh every 10 seconds

    return () => clearInterval(interval);
  }, []);

  // Acknowledge alert
  const handleAcknowledgeAlert = async (alertId: number) => {
    try {
      await fetch(`/api/v1/spc/alerts/${alertId}/acknowledge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resolution_notes: 'Acknowledged' }),
      });
      // Refresh alerts list
      const res = await fetch('/api/v1/spc/alerts?acknowledged=false&limit=50');
      const data = await res.json();
      setAlerts(data.alerts || []);
    } catch (err) {
      console.error('Failed to acknowledge alert:', err);
    }
  };

  const currentSeries = series?.find((s: SPCSeries) => s.id === selectedSeries);

  // Calculate statistics
  const calculateStats = (data: SPCPoint[]) => {
    if (!data || data.length === 0) return null;

    const values = data.map(p => p.value);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const std = Math.sqrt(
      values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length
    );

    // Check for violations
    const violations = {
      beyond3Sigma: values.filter(v => 
        v > currentSeries?.control_limits.UCL || v < currentSeries?.control_limits.LCL
      ).length,
      beyond2Sigma: values.filter(v => {
        const ucl2 = currentSeries?.control_limits.CL + 2 * std;
        const lcl2 = currentSeries?.control_limits.CL - 2 * std;
        return v > ucl2 || v < lcl2;
      }).length,
      outOfSpec: currentSeries?.spec_limits ? values.filter(v =>
        v > currentSeries.spec_limits.USL || v < currentSeries.spec_limits.LSL
      ).length : 0,
    };

    return { mean, std, violations, count: values.length };
  };

  const stats = calculateStats(points || []);

  // Process capability
  const calculateCpk = () => {
    if (!stats || !currentSeries?.spec_limits) return null;

    const { USL, LSL } = currentSeries.spec_limits;
    const { mean, std } = stats;

    const Cpu = (USL - mean) / (3 * std);
    const Cpl = (mean - LSL) / (3 * std);
    const Cpk = Math.min(Cpu, Cpl);

    return { Cpk, Cpu, Cpl };
  };

  const capability = calculateCpk();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-bold">SPC Monitoring</h2>
        <p className="text-muted-foreground">Statistical Process Control and alerts</p>
      </div>

      <Tabs defaultValue="charts" className="space-y-4">
        <TabsList>
          <TabsTrigger value="charts">Control Charts</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="charts" className="space-y-4">
          {/* Series Selection */}
          <Card>
            <CardHeader>
              <CardTitle>Select Parameter</CardTitle>
            </CardHeader>
            <CardContent>
              <Select
                value={selectedSeries?.toString()}
                onValueChange={(value) => setSelectedSeries(Number(value))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select a parameter to monitor" />
                </SelectTrigger>
                <SelectContent>
                  {series?.map((s: SPCSeries) => (
                    <SelectItem key={s.id} value={s.id.toString()}>
                      {s.name} - {s.parameter}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </CardContent>
          </Card>

          {/* Control Chart */}
          {currentSeries && points && (
            <Card>
              <CardHeader>
                <CardTitle>{currentSeries.name}</CardTitle>
                <CardDescription>
                  {currentSeries.chart_type} Chart - {currentSeries.parameter}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={points}>
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

                      {/* Control Limits */}
                      <ReferenceLine
                        y={currentSeries.control_limits.UCL}
                        label="UCL"
                        stroke="red"
                        strokeDasharray="5 5"
                      />
                      <ReferenceLine
                        y={currentSeries.control_limits.CL}
                        label="CL"
                        stroke="green"
                        strokeDasharray="5 5"
                      />
                      <ReferenceLine
                        y={currentSeries.control_limits.LCL}
                        label="LCL"
                        stroke="red"
                        strokeDasharray="5 5"
                      />

                      {/* Spec Limits */}
                      {currentSeries.spec_limits && (
                        <>
                          <ReferenceLine
                            y={currentSeries.spec_limits.USL}
                            label="USL"
                            stroke="orange"
                            strokeDasharray="3 3"
                          />
                          <ReferenceLine
                            y={currentSeries.spec_limits.LSL}
                            label="LSL"
                            stroke="orange"
                            strokeDasharray="3 3"
                          />
                        </>
                      )}

                      {/* Data Line */}
                      <Line
                        type="monotone"
                        dataKey="value"
                        stroke="#8884d8"
                        name="Value"
                        dot={(props: any) => {
                          const { cx, cy, payload } = props;
                          const hasViolation = payload.violations && payload.violations.length > 0;
                          return (
                            <Dot
                              cx={cx}
                              cy={cy}
                              r={hasViolation ? 5 : 3}
                              fill={hasViolation ? "red" : "#8884d8"}
                            />
                          );
                        }}
                      />

                      {/* EWMA Line if available */}
                      {points[0]?.ewma_value !== undefined && (
                        <Line
                          type="monotone"
                          dataKey="ewma_value"
                          stroke="#82ca9d"
                          name="EWMA"
                          dot={false}
                        />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Statistics */}
                <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Mean</p>
                    <p className="text-xl font-bold">{stats?.mean.toFixed(3)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Std Dev</p>
                    <p className="text-xl font-bold">{stats?.std.toFixed(3)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Points</p>
                    <p className="text-xl font-bold">{stats?.count}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Violations</p>
                    <p className="text-xl font-bold text-red-500">
                      {stats?.violations.beyond3Sigma || 0}
                    </p>
                  </div>
                </div>

                {/* Process Capability */}
                {capability && (
                  <div className="mt-4 p-4 bg-secondary rounded-lg">
                    <h4 className="font-medium mb-2">Process Capability</h4>
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <p className="text-sm text-muted-foreground">Cpk</p>
                        <p className={`text-lg font-bold ${
                          capability.Cpk >= 1.33 ? 'text-green-500' : 
                          capability.Cpk >= 1.0 ? 'text-yellow-500' : 'text-red-500'
                        }`}>
                          {capability.Cpk.toFixed(2)}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Cpu</p>
                        <p className="text-lg font-bold">{capability.Cpu.toFixed(2)}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Cpl</p>
                        <p className="text-lg font-bold">{capability.Cpl.toFixed(2)}</p>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Active Alerts</CardTitle>
              <CardDescription>Unacknowledged SPC violations</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {alerts && alerts.length > 0 ? (
                  alerts.map((alert: SPCAlert) => (
                    <div
                      key={alert.id}
                      className="flex items-center justify-between p-3 border rounded-lg"
                    >
                      <div className="flex items-start gap-3">
                        <AlertTriangle className={`w-5 h-5 mt-0.5 ${
                          alert.severity === 'critical' ? 'text-red-500' : 'text-yellow-500'
                        }`} />
                        <div>
                          <p className="font-medium">{alert.rule_violated}</p>
                          <p className="text-sm text-muted-foreground">
                            {alert.description}
                          </p>
                          <p className="text-xs text-muted-foreground mt-1">
                            {new Date(alert.created_at).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleAcknowledgeAlert(alert.id)}
                      >
                        Acknowledge
                      </Button>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <CheckCircle className="w-12 h-12 mx-auto mb-2" />
                    <p>No active alerts</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analysis" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Statistical Analysis</CardTitle>
              <CardDescription>Detailed process statistics and trends</CardDescription>
            </CardHeader>
            <CardContent>
              {stats && (
                <div className="space-y-6">
                  {/* Distribution Analysis */}
                  <div>
                    <h4 className="font-medium mb-3">Distribution Analysis</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="p-3 bg-secondary rounded">
                        <p className="text-sm text-muted-foreground">Normality</p>
                        <p className="text-lg font-bold">
                          {/* Simplified normality check */}
                          {Math.abs(stats.mean - currentSeries?.control_limits.CL) < stats.std * 0.5 
                            ? 'Normal' : 'Check Required'}
                        </p>
                      </div>
                      <div className="p-3 bg-secondary rounded">
                        <p className="text-sm text-muted-foreground">Stability</p>
                        <p className="text-lg font-bold">
                          {stats.violations.beyond3Sigma === 0 ? 'Stable' : 'Unstable'}
                        </p>
                      </div>
                      <div className="p-3 bg-secondary rounded">
                        <p className="text-sm text-muted-foreground">Trend</p>
                        <p className="text-lg font-bold">
                          <TrendingUp className="inline w-4 h-4 mr-1" />
                          Monitoring
                        </p>
                      </div>
                      <div className="p-3 bg-secondary rounded">
                        <p className="text-sm text-muted-foreground">Performance</p>
                        <p className="text-lg font-bold">
                          {capability && capability.Cpk >= 1.33 ? 'Capable' : 'Review'}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Rule Violations Summary */}
                  <div>
                    <h4 className="font-medium mb-3">Western Electric Rules</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between p-2 bg-secondary rounded">
                        <span className="text-sm">1 point beyond 3σ</span>
                        <Badge variant={stats.violations.beyond3Sigma > 0 ? "destructive" : "default"}>
                          {stats.violations.beyond3Sigma}
                        </Badge>
                      </div>
                      <div className="flex justify-between p-2 bg-secondary rounded">
                        <span className="text-sm">Points beyond 2σ</span>
                        <Badge variant={stats.violations.beyond2Sigma > 2 ? "destructive" : "default"}>
                          {stats.violations.beyond2Sigma}
                        </Badge>
                      </div>
                      <div className="flex justify-between p-2 bg-secondary rounded">
                        <span className="text-sm">Out of specification</span>
                        <Badge variant={stats.violations.outOfSpec > 0 ? "destructive" : "default"}>
                          {stats.violations.outOfSpec}
                        </Badge>
                      </div>
                    </div>
                  </div>

                  {/* Recommendations */}
                  <Alert>
                    <Bell className="h-4 w-4" />
                    <AlertDescription>
                      {capability && capability.Cpk < 1.0 
                        ? 'Process capability is below target. Consider process optimization.'
                        : capability && capability.Cpk < 1.33
                        ? 'Process is marginally capable. Monitor closely.'
                        : 'Process is performing well. Continue monitoring.'}
                    </AlertDescription>
                  </Alert>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SPCMonitoring;