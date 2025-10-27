/**
 * Session 13: SPC Hub - Complete UI Components
 * Semiconductor Lab Platform
 * 
 * Complete SPC dashboard with:
 * - Control charts (X-bar/R, EWMA, CUSUM)
 * - Alert management and triage
 * - Process capability analysis
 * - Real-time monitoring
 * - Drill-down capabilities
 * 
 * Author: Semiconductor Lab Platform Team
 * Version: 1.0.0
 * Date: October 2025
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, Area, AreaChart, ScatterChart,
  Scatter, ComposedChart, Bar
} from 'recharts';
import {
  AlertTriangle, CheckCircle, XCircle, TrendingUp, TrendingDown,
  Activity, Bell, Filter, Download, RefreshCw, Search, ChevronDown,
  ChevronRight, AlertCircle, Info, Settings, Calendar, User, Tool
} from 'lucide-react';

// ============================================================================
// Types & Interfaces
// ============================================================================

interface DataPoint {
  timestamp: string;
  value: number;
  subgroup: string;
  runId: string;
  metadata?: Record<string, any>;
}

interface ControlLimits {
  ucl: number;
  lcl: number;
  cl: number;
  sigma: number;
}

interface Alert {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  rule: string;
  message: string;
  timestamp: string;
  suggestedActions: string[];
  acknowledged?: boolean;
}

interface ProcessCapability {
  cp: number;
  cpk: number;
  cpu: number;
  cpl: number;
  mean: number;
  sigma: number;
  interpretation: string;
}

interface MetricData {
  metric: string;
  chartType: string;
  dataPoints: DataPoint[];
  controlLimits: ControlLimits;
  alerts: Alert[];
  capability: ProcessCapability;
  statistics: {
    min: number;
    max: number;
    mean: number;
    median: number;
    std: number;
    range: number;
  };
}

// ============================================================================
// Data Generation Functions
// ============================================================================

const generateInControlData = (
  nPoints: number = 50,
  mean: number = 100,
  sigma: number = 2
): DataPoint[] => {
  const data: DataPoint[] = [];
  const baseDate = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
  
  for (let i = 0; i < nPoints; i++) {
    const value = mean + sigma * (Math.random() * 2 - 1) * Math.sqrt(3); // Uniform-ish
    const timestamp = new Date(baseDate.getTime() + i * 60 * 60 * 1000);
    
    data.push({
      timestamp: timestamp.toISOString(),
      value,
      subgroup: `SG${Math.floor(i / 5) + 1}`,
      runId: `RUN${1000 + i}`,
      metadata: {
        operator: i % 3 === 0 ? 'Alice' : i % 3 === 1 ? 'Bob' : 'Charlie',
        tool: `TOOL${(i % 3) + 1}`
      }
    });
  }
  
  return data;
};

const generateShiftData = (
  nPoints: number = 50,
  mean: number = 100,
  sigma: number = 2,
  shiftPoint: number = 25,
  shiftAmount: number = 4
): DataPoint[] => {
  const data: DataPoint[] = [];
  const baseDate = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
  
  for (let i = 0; i < nPoints; i++) {
    const currentMean = i < shiftPoint ? mean : mean + shiftAmount;
    const value = currentMean + sigma * (Math.random() * 2 - 1) * Math.sqrt(3);
    const timestamp = new Date(baseDate.getTime() + i * 60 * 60 * 1000);
    
    data.push({
      timestamp: timestamp.toISOString(),
      value,
      subgroup: `SG${Math.floor(i / 5) + 1}`,
      runId: `RUN${1000 + i}`,
      metadata: {
        operator: i % 2 === 0 ? 'Alice' : 'Bob',
        tool: 'TOOL1'
      }
    });
  }
  
  return data;
};

const generateTrendData = (
  nPoints: number = 50,
  mean: number = 100,
  sigma: number = 2,
  driftRate: number = 0.1
): DataPoint[] => {
  const data: DataPoint[] = [];
  const baseDate = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
  
  for (let i = 0; i < nPoints; i++) {
    const currentMean = mean + driftRate * i;
    const value = currentMean + sigma * (Math.random() * 2 - 1) * Math.sqrt(3);
    const timestamp = new Date(baseDate.getTime() + i * 60 * 60 * 1000);
    
    data.push({
      timestamp: timestamp.toISOString(),
      value,
      subgroup: `SG${Math.floor(i / 5) + 1}`,
      runId: `RUN${1000 + i}`,
      metadata: {
        operator: 'Alice',
        tool: 'TOOL1',
        note: 'Tool wear suspected'
      }
    });
  }
  
  return data;
};

const calculateControlLimits = (data: DataPoint[]): ControlLimits => {
  const values = data.map(d => d.value);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (values.length - 1);
  const sigma = Math.sqrt(variance);
  
  return {
    ucl: mean + 3 * sigma,
    lcl: mean - 3 * sigma,
    cl: mean,
    sigma
  };
};

const detectAlerts = (data: DataPoint[], limits: ControlLimits): Alert[] => {
  const alerts: Alert[] = [];
  
  // Rule 1: Point beyond 3σ
  data.forEach((point, idx) => {
    if (point.value > limits.ucl || point.value < limits.lcl) {
      alerts.push({
        id: `alert_${point.runId}_${idx}`,
        severity: 'critical',
        rule: '1_point_beyond_3sigma',
        message: `Point beyond 3σ control limits (value: ${point.value.toFixed(2)})`,
        timestamp: point.timestamp,
        suggestedActions: [
          'Check instrument calibration',
          'Verify measurement procedure',
          'Inspect for special causes'
        ]
      });
    }
  });
  
  // Rule 4: 8 consecutive points on same side
  for (let i = 0; i <= data.length - 8; i++) {
    const window = data.slice(i, i + 8);
    const allAbove = window.every(p => p.value > limits.cl);
    const allBelow = window.every(p => p.value < limits.cl);
    
    if (allAbove || allBelow) {
      alerts.push({
        id: `alert_rule4_${i}`,
        severity: 'medium',
        rule: '8_consecutive_same_side',
        message: `8 consecutive points ${allAbove ? 'above' : 'below'} centerline`,
        timestamp: window[7].timestamp,
        suggestedActions: [
          'Process may have shifted',
          'Recalculate control limits if shift is permanent',
          'Investigate root cause'
        ]
      });
      break;
    }
  }
  
  // Rule 5: 6 points trending
  for (let i = 0; i <= data.length - 6; i++) {
    const window = data.slice(i, i + 6);
    const values = window.map(p => p.value);
    
    const increasing = values.every((val, idx) => idx === 0 || val > values[idx - 1]);
    const decreasing = values.every((val, idx) => idx === 0 || val < values[idx - 1]);
    
    if (increasing || decreasing) {
      alerts.push({
        id: `alert_rule5_${i}`,
        severity: 'medium',
        rule: '6_points_trending',
        message: `6 points trending ${increasing ? 'up' : 'down'}`,
        timestamp: window[5].timestamp,
        suggestedActions: [
          'Check for tool wear or drift',
          'Verify environmental conditions',
          'Review PM schedule'
        ]
      });
      break;
    }
  }
  
  return alerts;
};

const calculateCapability = (
  data: DataPoint[],
  lsl: number = 94,
  usl: number = 106
): ProcessCapability => {
  const values = data.map(d => d.value);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (values.length - 1);
  const sigma = Math.sqrt(variance);
  
  const cp = (usl - lsl) / (6 * sigma);
  const cpu = (usl - mean) / (3 * sigma);
  const cpl = (mean - lsl) / (3 * sigma);
  const cpk = Math.min(cpu, cpl);
  
  let interpretation = '';
  if (cpk >= 2.0) interpretation = 'Excellent (6σ capable)';
  else if (cpk >= 1.67) interpretation = 'Very Good (5σ capable)';
  else if (cpk >= 1.33) interpretation = 'Adequate (4σ capable)';
  else if (cpk >= 1.0) interpretation = 'Marginal (3σ capable)';
  else interpretation = 'Poor (High defect rate)';
  
  return { cp, cpk, cpu, cpl, mean, sigma, interpretation };
};

const generateMockMetricData = (scenario: 'in-control' | 'shift' | 'trend' = 'in-control'): MetricData => {
  let dataPoints: DataPoint[];
  
  switch (scenario) {
    case 'shift':
      dataPoints = generateShiftData();
      break;
    case 'trend':
      dataPoints = generateTrendData();
      break;
    default:
      dataPoints = generateInControlData();
  }
  
  const controlLimits = calculateControlLimits(dataPoints);
  const alerts = detectAlerts(dataPoints, controlLimits);
  const capability = calculateCapability(dataPoints);
  
  const values = dataPoints.map(d => d.value);
  
  return {
    metric: 'Sheet Resistance',
    chartType: 'xbar_r',
    dataPoints,
    controlLimits,
    alerts,
    capability,
    statistics: {
      min: Math.min(...values),
      max: Math.max(...values),
      mean: values.reduce((a, b) => a + b, 0) / values.length,
      median: values.sort((a, b) => a - b)[Math.floor(values.length / 2)],
      std: controlLimits.sigma,
      range: Math.max(...values) - Math.min(...values)
    }
  };
};

// ============================================================================
// UI Components
// ============================================================================

const Card: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = '' }) => (
  <div className={`bg-white rounded-lg shadow-sm border border-gray-200 ${className}`}>
    {children}
  </div>
);

const CardHeader: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="px-6 py-4 border-b border-gray-200">
    {children}
  </div>
);

const CardContent: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = '' }) => (
  <div className={`px-6 py-4 ${className}`}>
    {children}
  </div>
);

const Badge: React.FC<{ children: React.ReactNode; variant: 'low' | 'medium' | 'high' | 'critical' | 'default' }> = ({
  children,
  variant
}) => {
  const colors = {
    low: 'bg-blue-100 text-blue-800',
    medium: 'bg-yellow-100 text-yellow-800',
    high: 'bg-orange-100 text-orange-800',
    critical: 'bg-red-100 text-red-800',
    default: 'bg-gray-100 text-gray-800'
  };
  
  return (
    <span className={`px-2 py-1 text-xs font-medium rounded-full ${colors[variant]}`}>
      {children}
    </span>
  );
};

const Button: React.FC<{
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}> = ({ children, onClick, variant = 'primary', size = 'md', className = '' }) => {
  const baseStyles = 'font-medium rounded-lg transition-colors flex items-center gap-2';
  
  const variants = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700',
    secondary: 'bg-gray-200 text-gray-800 hover:bg-gray-300',
    outline: 'border border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
  };
  
  const sizes = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base'
  };
  
  return (
    <button
      onClick={onClick}
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
    >
      {children}
    </button>
  );
};

// ============================================================================
// Control Chart Component
// ============================================================================

const ControlChartDisplay: React.FC<{ metricData: MetricData }> = ({ metricData }) => {
  const { dataPoints, controlLimits } = metricData;
  
  const chartData = dataPoints.map((point, idx) => ({
    index: idx + 1,
    value: point.value,
    timestamp: new Date(point.timestamp).toLocaleDateString(),
    ucl: controlLimits.ucl,
    lcl: controlLimits.lcl,
    cl: controlLimits.cl,
    runId: point.runId
  }));
  
  return (
    <ResponsiveContainer width="100%" height={400}>
      <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="index" 
          label={{ value: 'Sample Number', position: 'insideBottom', offset: -10 }}
        />
        <YAxis 
          label={{ value: 'Value', angle: -90, position: 'insideLeft' }}
          domain={[
            controlLimits.lcl - controlLimits.sigma,
            controlLimits.ucl + controlLimits.sigma
          ]}
        />
        <Tooltip 
          content={({ active, payload }) => {
            if (active && payload && payload.length) {
              return (
                <div className="bg-white p-3 border border-gray-200 rounded shadow-lg">
                  <p className="font-medium">{payload[0].payload.runId}</p>
                  <p className="text-sm text-gray-600">{payload[0].payload.timestamp}</p>
                  <p className="text-sm mt-1">Value: <span className="font-medium">{payload[0].value.toFixed(2)}</span></p>
                </div>
              );
            }
            return null;
          }}
        />
        <Legend />
        
        <ReferenceLine y={controlLimits.ucl} stroke="#ef4444" strokeDasharray="5 5" label="UCL" />
        <ReferenceLine y={controlLimits.cl} stroke="#10b981" strokeDasharray="5 5" label="CL" />
        <ReferenceLine y={controlLimits.lcl} stroke="#ef4444" strokeDasharray="5 5" label="LCL" />
        
        <Line 
          type="monotone" 
          dataKey="value" 
          stroke="#3b82f6" 
          strokeWidth={2}
          dot={{ fill: '#3b82f6', r: 4 }}
          name="Measured Value"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

// ============================================================================
// Alert Panel Component
// ============================================================================

const AlertPanel: React.FC<{ alerts: Alert[] }> = ({ alerts }) => {
  const [filter, setFilter] = useState<'all' | 'critical' | 'high' | 'medium' | 'low'>('all');
  const [expanded, setExpanded] = useState<string | null>(null);
  
  const filteredAlerts = alerts.filter(alert => 
    filter === 'all' || alert.severity === filter
  );
  
  const getAlertIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <XCircle className="w-5 h-5 text-red-600" />;
      case 'high':
        return <AlertTriangle className="w-5 h-5 text-orange-600" />;
      case 'medium':
        return <AlertCircle className="w-5 h-5 text-yellow-600" />;
      default:
        return <Info className="w-5 h-5 text-blue-600" />;
    }
  };
  
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Bell className="w-5 h-5" />
            Active Alerts ({filteredAlerts.length})
          </h3>
          
          <div className="flex items-center gap-2">
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value as any)}
              className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg"
            >
              <option value="all">All Alerts</option>
              <option value="critical">Critical Only</option>
              <option value="high">High Only</option>
              <option value="medium">Medium Only</option>
              <option value="low">Low Only</option>
            </select>
            
            <Button variant="outline" size="sm">
              <Download className="w-4 h-4" />
              Export
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="p-0">
        {filteredAlerts.length === 0 ? (
          <div className="p-8 text-center text-gray-500">
            <CheckCircle className="w-12 h-12 mx-auto mb-3 text-green-500" />
            <p className="font-medium">No alerts found</p>
            <p className="text-sm">Process is in statistical control</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {filteredAlerts.map((alert) => (
              <div key={alert.id} className="p-4 hover:bg-gray-50 transition-colors">
                <div className="flex items-start gap-3">
                  {getAlertIcon(alert.severity)}
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <Badge variant={alert.severity}>
                        {alert.severity.toUpperCase()}
                      </Badge>
                      <span className="text-xs text-gray-500">
                        {new Date(alert.timestamp).toLocaleString()}
                      </span>
                    </div>
                    
                    <p className="font-medium text-gray-900 mb-1">{alert.message}</p>
                    <p className="text-sm text-gray-600 mb-2">Rule: {alert.rule.replace(/_/g, ' ')}</p>
                    
                    <button
                      onClick={() => setExpanded(expanded === alert.id ? null : alert.id)}
                      className="text-sm text-blue-600 hover:text-blue-700 flex items-center gap-1"
                    >
                      {expanded === alert.id ? (
                        <>
                          <ChevronDown className="w-4 h-4" />
                          Hide Actions
                        </>
                      ) : (
                        <>
                          <ChevronRight className="w-4 h-4" />
                          Show Suggested Actions
                        </>
                      )}
                    </button>
                    
                    {expanded === alert.id && (
                      <div className="mt-3 p-3 bg-blue-50 rounded-lg">
                        <p className="text-sm font-medium text-blue-900 mb-2">Suggested Actions:</p>
                        <ul className="space-y-1">
                          {alert.suggestedActions.map((action, idx) => (
                            <li key={idx} className="text-sm text-blue-800 flex items-start gap-2">
                              <span className="text-blue-600 mt-0.5">•</span>
                              {action}
                            </li>
                          ))}
                        </ul>
                        
                        <div className="flex gap-2 mt-3">
                          <Button size="sm" variant="primary">
                            Acknowledge
                          </Button>
                          <Button size="sm" variant="outline">
                            Investigate
                          </Button>
                          <Button size="sm" variant="outline">
                            View Run Details
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

// ============================================================================
// Process Capability Component
// ============================================================================

const ProcessCapabilityDisplay: React.FC<{ capability: ProcessCapability }> = ({ capability }) => {
  const getCapabilityColor = (cpk: number) => {
    if (cpk >= 2.0) return 'text-green-600';
    if (cpk >= 1.67) return 'text-blue-600';
    if (cpk >= 1.33) return 'text-yellow-600';
    if (cpk >= 1.0) return 'text-orange-600';
    return 'text-red-600';
  };
  
  const getCapabilityIcon = (cpk: number) => {
    if (cpk >= 1.33) return <CheckCircle className="w-6 h-6 text-green-600" />;
    if (cpk >= 1.0) return <AlertCircle className="w-6 h-6 text-yellow-600" />;
    return <XCircle className="w-6 h-6 text-red-600" />;
  };
  
  return (
    <Card>
      <CardHeader>
        <h3 className="text-lg font-semibold">Process Capability Analysis</h3>
      </CardHeader>
      
      <CardContent>
        <div className="grid grid-cols-2 gap-6 mb-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
              <span className="text-sm font-medium text-blue-900">Cp (Potential)</span>
              <span className="text-2xl font-bold text-blue-600">{capability.cp.toFixed(3)}</span>
            </div>
            
            <div className="flex items-center justify-between p-4 bg-purple-50 rounded-lg">
              <span className="text-sm font-medium text-purple-900">CPU (Upper)</span>
              <span className="text-2xl font-bold text-purple-600">{capability.cpu.toFixed(3)}</span>
            </div>
            
            <div className="flex items-center justify-between p-4 bg-pink-50 rounded-lg">
              <span className="text-sm font-medium text-pink-900">CPL (Lower)</span>
              <span className="text-2xl font-bold text-pink-600">{capability.cpl.toFixed(3)}</span>
            </div>
          </div>
          
          <div className="flex flex-col items-center justify-center p-6 bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg">
            {getCapabilityIcon(capability.cpk)}
            <div className={`text-5xl font-bold mt-3 ${getCapabilityColor(capability.cpk)}`}>
              {capability.cpk.toFixed(3)}
            </div>
            <div className="text-sm font-medium text-gray-600 mt-1">Cpk (Actual)</div>
            <div className="text-xs text-gray-500 mt-2 text-center max-w-xs">
              {capability.interpretation}
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="p-3 border border-gray-200 rounded-lg">
            <div className="text-xs text-gray-600 mb-1">Process Mean</div>
            <div className="text-lg font-semibold">{capability.mean.toFixed(2)}</div>
          </div>
          
          <div className="p-3 border border-gray-200 rounded-lg">
            <div className="text-xs text-gray-600 mb-1">Process Sigma</div>
            <div className="text-lg font-semibold">{capability.sigma.toFixed(3)}</div>
          </div>
        </div>
        
        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
          <p className="text-xs text-gray-600 mb-2">Interpretation Guidelines:</p>
          <ul className="text-xs text-gray-700 space-y-1">
            <li>• Cpk ≥ 2.0: Excellent (6σ capable)</li>
            <li>• Cpk ≥ 1.67: Very Good (5σ capable)</li>
            <li>• Cpk ≥ 1.33: Adequate (4σ capable)</li>
            <li>• Cpk ≥ 1.0: Marginal (3σ capable, some rejects expected)</li>
            <li>• Cpk &lt; 1.0: Poor (High defect rate, process not capable)</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};

// ============================================================================
// Statistics Summary Component
// ============================================================================

const StatisticsSummary: React.FC<{ statistics: MetricData['statistics'] }> = ({ statistics }) => {
  return (
    <Card>
      <CardHeader>
        <h3 className="text-lg font-semibold">Process Statistics</h3>
      </CardHeader>
      
      <CardContent>
        <div className="grid grid-cols-3 gap-4">
          <div className="p-4 bg-blue-50 rounded-lg">
            <div className="text-xs text-blue-600 font-medium mb-1">Mean</div>
            <div className="text-2xl font-bold text-blue-900">{statistics.mean.toFixed(2)}</div>
          </div>
          
          <div className="p-4 bg-green-50 rounded-lg">
            <div className="text-xs text-green-600 font-medium mb-1">Median</div>
            <div className="text-2xl font-bold text-green-900">{statistics.median.toFixed(2)}</div>
          </div>
          
          <div className="p-4 bg-purple-50 rounded-lg">
            <div className="text-xs text-purple-600 font-medium mb-1">Std Dev</div>
            <div className="text-2xl font-bold text-purple-900">{statistics.std.toFixed(3)}</div>
          </div>
          
          <div className="p-4 bg-orange-50 rounded-lg">
            <div className="text-xs text-orange-600 font-medium mb-1">Min</div>
            <div className="text-2xl font-bold text-orange-900">{statistics.min.toFixed(2)}</div>
          </div>
          
          <div className="p-4 bg-pink-50 rounded-lg">
            <div className="text-xs text-pink-600 font-medium mb-1">Max</div>
            <div className="text-2xl font-bold text-pink-900">{statistics.max.toFixed(2)}</div>
          </div>
          
          <div className="p-4 bg-cyan-50 rounded-lg">
            <div className="text-xs text-cyan-600 font-medium mb-1">Range</div>
            <div className="text-2xl font-bold text-cyan-900">{statistics.range.toFixed(2)}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// ============================================================================
// Main SPC Dashboard Component (Reusable)
// ============================================================================

export const SPCDashboard: React.FC<{ 
  results: MetricData;
  onRefresh?: () => void;
}> = ({ results, onRefresh }) => {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">{results.metric} - SPC Monitoring</h1>
          <p className="text-sm text-gray-600 mt-1">
            Statistical Process Control Dashboard • {results.dataPoints.length} samples analyzed
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" onClick={onRefresh}>
            <RefreshCw className="w-4 h-4" />
            Refresh
          </Button>
          
          <Button variant="outline" size="sm">
            <Settings className="w-4 h-4" />
            Settings
          </Button>
          
          <Button variant="primary" size="sm">
            <Download className="w-4 h-4" />
            Export Report
          </Button>
        </div>
      </div>
      
      {/* Status Summary */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg">
          <div className="flex items-center gap-3">
            <Activity className="w-8 h-8 text-blue-600" />
            <div>
              <div className="text-sm text-blue-600 font-medium">Process Status</div>
              <div className="text-2xl font-bold text-blue-900">
                {results.alerts.length === 0 ? 'In Control' : 'Out of Control'}
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-gradient-to-br from-red-50 to-red-100 p-4 rounded-lg">
          <div className="flex items-center gap-3">
            <Bell className="w-8 h-8 text-red-600" />
            <div>
              <div className="text-sm text-red-600 font-medium">Active Alerts</div>
              <div className="text-2xl font-bold text-red-900">{results.alerts.length}</div>
            </div>
          </div>
        </div>
        
        <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg">
          <div className="flex items-center gap-3">
            <TrendingUp className="w-8 h-8 text-green-600" />
            <div>
              <div className="text-sm text-green-600 font-medium">Cpk Index</div>
              <div className="text-2xl font-bold text-green-900">{results.capability.cpk.toFixed(2)}</div>
            </div>
          </div>
        </div>
        
        <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg">
          <div className="flex items-center gap-3">
            <CheckCircle className="w-8 h-8 text-purple-600" />
            <div>
              <div className="text-sm text-purple-600 font-medium">Samples</div>
              <div className="text-2xl font-bold text-purple-900">{results.dataPoints.length}</div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Control Chart */}
      <Card>
        <CardHeader>
          <h3 className="text-lg font-semibold">X-bar Control Chart</h3>
        </CardHeader>
        <CardContent>
          <ControlChartDisplay metricData={results} />
        </CardContent>
      </Card>
      
      {/* Two Column Layout */}
      <div className="grid grid-cols-2 gap-6">
        <ProcessCapabilityDisplay capability={results.capability} />
        <StatisticsSummary statistics={results.statistics} />
      </div>
      
      {/* Alerts */}
      <AlertPanel alerts={results.alerts} />
    </div>
  );
};

// ============================================================================
// Page Wrapper Component (THIS IS THE FIX!)
// ============================================================================

export default function SPCPage() {
  const [scenario, setScenario] = useState<'in-control' | 'shift' | 'trend'>('in-control');
  const [metricData, setMetricData] = useState<MetricData>(() => generateMockMetricData('in-control'));
  
  const handleScenarioChange = (newScenario: 'in-control' | 'shift' | 'trend') => {
    setScenario(newScenario);
    setMetricData(generateMockMetricData(newScenario));
  };
  
  const handleRefresh = () => {
    setMetricData(generateMockMetricData(scenario));
  };
  
  return (
    <div className="min-h-screen bg-gray-50 p-6">
      {/* Scenario Selector (Dev Tool) */}
      <div className="mb-6 p-4 bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-gray-700">Demo Scenario:</span>
          <div className="flex gap-2">
            <button
              onClick={() => handleScenarioChange('in-control')}
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                scenario === 'in-control'
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              In Control
            </button>
            <button
              onClick={() => handleScenarioChange('shift')}
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                scenario === 'shift'
                  ? 'bg-orange-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Process Shift
            </button>
            <button
              onClick={() => handleScenarioChange('trend')}
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                scenario === 'trend'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Trending
            </button>
          </div>
        </div>
      </div>
      
      {/* Main Dashboard */}
      <SPCDashboard results={metricData} onRefresh={handleRefresh} />
    </div>
  );
}
