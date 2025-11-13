/**
 * Main Dashboard Component
 * Displays real-time process metrics, SPC status, and equipment health
 */

import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Alert,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface ProcessData {
  timestamp: string;
  temperature: number;
  pressure: number;
  deposition_rate: number;
}

interface SPCSummary {
  total_charts: number;
  total_violations: number;
  active_alarms: number;
}

const DashboardView: React.FC = () => {
  const [processData, setProcessData] = useState<ProcessData[]>([]);
  const [spcSummary, setSPCSummary] = useState<SPCSummary | null>(null);
  const [vmPrediction, setVMPrediction] = useState<any>(null);

  useEffect(() => {
    // Fetch initial data
    fetchSPCSummary();

    // Set up WebSocket for real-time data
    const ws = new WebSocket('ws://localhost:8000/ws/realtime');

    ws.onmessage = (event) => {
      const data: ProcessData = JSON.parse(event.data);
      setProcessData((prev) => {
        const updated = [...prev, data];
        // Keep only last 100 points
        return updated.slice(-100);
      });
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      ws.close();
    };
  }, []);

  const fetchSPCSummary = async () => {
    try {
      const response = await fetch('http://localhost:8000/spc/charts/summary');
      const data = await response.json();
      setSPCSummary(data);
    } catch (error) {
      console.error('Failed to fetch SPC summary:', error);
    }
  };

  const renderMetricCard = (title: string, value: string | number, unit: string = '', color: string = 'primary') => (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography color="textSecondary" gutterBottom variant="overline">
          {title}
        </Typography>
        <Typography variant="h4" component="div" color={color}>
          {value}
          {unit && <Typography component="span" variant="h6"> {unit}</Typography>}
        </Typography>
      </CardContent>
    </Card>
  );

  const latestData = processData.length > 0 ? processData[processData.length - 1] : null;

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Process Dashboard
      </Typography>

      {/* Alert banner for active alarms */}
      {spcSummary && spcSummary.active_alarms > 0 && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          {spcSummary.active_alarms} active alarm(s) detected. Check SPC Charts for details.
        </Alert>
      )}

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          {renderMetricCard(
            'Temperature',
            latestData?.temperature.toFixed(1) || '--',
            '°C',
            'primary'
          )}
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          {renderMetricCard(
            'Pressure',
            latestData?.pressure.toFixed(2) || '--',
            'Torr',
            'secondary'
          )}
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          {renderMetricCard(
            'Deposition Rate',
            latestData?.deposition_rate.toFixed(3) || '--',
            'nm/s',
            'success'
          )}
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          {renderMetricCard(
            'SPC Violations',
            spcSummary?.total_violations || 0,
            '',
            spcSummary && spcSummary.total_violations > 0 ? 'error' : 'success'
          )}
        </Grid>
      </Grid>

      {/* Real-Time Charts */}
      <Grid container spacing={3}>
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Temperature Trend
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={processData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                  />
                  <YAxis domain={['auto', 'auto']} />
                  <Tooltip
                    labelFormatter={(time) => new Date(time).toLocaleTimeString()}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="temperature"
                    stroke="#2196f3"
                    strokeWidth={2}
                    dot={false}
                    name="Temperature (°C)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Pressure Trend
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={processData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                  />
                  <YAxis domain={['auto', 'auto']} />
                  <Tooltip
                    labelFormatter={(time) => new Date(time).toLocaleTimeString()}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="pressure"
                    stroke="#f50057"
                    strokeWidth={2}
                    dot={false}
                    name="Pressure (Torr)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Deposition Rate
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={processData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                  />
                  <YAxis domain={['auto', 'auto']} />
                  <Tooltip
                    labelFormatter={(time) => new Date(time).toLocaleTimeString()}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="deposition_rate"
                    stroke="#4caf50"
                    strokeWidth={2}
                    dot={false}
                    name="Deposition Rate (nm/s)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* SPC Summary */}
      {spcSummary && (
        <Grid container spacing={3} sx={{ mt: 2 }}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  SPC Summary
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={4}>
                    <Typography variant="body2" color="textSecondary">
                      Total Charts
                    </Typography>
                    <Typography variant="h5">
                      {spcSummary.total_charts}
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="body2" color="textSecondary">
                      Total Violations
                    </Typography>
                    <Typography variant="h5" color="warning.main">
                      {spcSummary.total_violations}
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="body2" color="textSecondary">
                      Active Alarms
                    </Typography>
                    <Typography variant="h5" color="error.main">
                      {spcSummary.active_alarms}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default DashboardView;
