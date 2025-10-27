'use client'

/**
 * Session 13: SPC Hub - Complete UI Components
 *
 * Interactive statistical process control dashboards including:
 * - Control chart visualization (X-bar/R, I-MR, EWMA, CUSUM)
 * - Real-time alert monitoring and triage
 * - Process capability analysis
 * - Trend analysis and forecasting
 * - Root cause analysis interface
 * - Multi-metric dashboard
 * 
 * @author Semiconductor Lab Platform Team
 * @version 1.0.0
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  LineChart, Line, ScatterChart, Scatter, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, ReferenceArea, Area, AreaChart
} from 'recharts';
import {
  TrendingUp, TrendingDown, AlertTriangle, CheckCircle,
  Activity, Bell, Settings, Download, Filter,
  ChevronDown, ChevronRight, Search, RefreshCw,
  Info, Eye, ZoomIn, BarChart3, AlertCircle
} from 'lucide-react';


// ============================================================================
// Type Definitions
// ============================================================================

interface ControlLimits {
  ucl: number;
  lcl: number;
  centerline: number;
  usl?: number;
  lsl?: number;
  sigma?: number;
}

interface Alert {
  id: string;
  timestamp: string;
  rule: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  message: string;
  value: number;
  points_involved: number[];
  suggested_actions: string[];
  root_causes: string[];
}

interface ProcessCapability {
  cp: number;
  cpk: number;
  pp: number;
  ppk: number;
  sigma_level: number;
  dpmo: number;
  is_capable: boolean;
  comments: string[];
}

interface TrendAnalysis {
  detected: boolean;
  direction: 'increasing' | 'decreasing' | 'stable';
  slope: number;
  p_value: number;
  predicted_values: number[];
  changepoints: number[];
}

interface SPCResults {
  timestamp: string;
  chart_type: string;
  n_points: number;
  statistics: {
    mean: number;
    std: number;
    min: number;
    max: number;
    median: number;
    range: number;
  };
  control_limits: {
    [key: string]: ControlLimits;
  };
  alerts: Alert[];
  capability: ProcessCapability;
  trend: TrendAnalysis;
  status: 'in_control' | 'out_of_control' | 'warning' | 'unknown';
  recommendations: string[];
  root_cause_analysis?: {
    likely_causes: string[];
    investigate: string[];
    preventive_actions: string[];
  };
}


// ============================================================================
// Utility Functions
// ============================================================================

const getSeverityColor = (severity: string): string => {
  switch (severity) {
    case 'critical': return 'text-red-600 bg-red-50 border-red-200';
    case 'high': return 'text-orange-600 bg-orange-50 border-orange-200';
    case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    case 'low': return 'text-blue-600 bg-blue-50 border-blue-200';
    default: return 'text-gray-600 bg-gray-50 border-gray-200';
  }
};

const getStatusBadge = (status: string) => {
  switch (status) {
    case 'in_control':
      return (
        <div className="flex items-center gap-2 px-3 py-1 bg-green-100 text-green-800 rounded-full">
          <CheckCircle className="w-4 h-4" />
          <span className="font-medium">In Control</span>
        </div>
      );
    case 'out_of_control':
      return (
        <div className="flex items-center gap-2 px-3 py-1 bg-red-100 text-red-800 rounded-full">
          <AlertTriangle className="w-4 h-4" />
          <span className="font-medium">Out of Control</span>
        </div>
      );
    case 'warning':
      return (
        <div className="flex items-center gap-2 px-3 py-1 bg-yellow-100 text-yellow-800 rounded-full">
          <AlertCircle className="w-4 h-4" />
          <span className="font-medium">Warning</span>
        </div>
      );
    default:
      return (
        <div className="flex items-center gap-2 px-3 py-1 bg-gray-100 text-gray-800 rounded-full">
          <Info className="w-4 h-4" />
          <span className="font-medium">Unknown</span>
        </div>
      );
  }
};

const formatNumber = (num: number, decimals: number = 3): string => {
  if (Math.abs(num) >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
  if (Math.abs(num) >= 1e3) return `${(num / 1e3).toFixed(2)}K`;
  return num.toFixed(decimals);
};


// ============================================================================
// Control Chart Component
// ============================================================================

interface ControlChartProps {
  data: number[];
  limits: ControlLimits;
  alerts: Alert[];
  title: string;
  chartType: string;
}

export const ControlChart: React.FC<ControlChartProps> = ({
  data,
  limits,
  alerts,
  title,
  chartType
}) => {
  const [zoomDomain, setZoomDomain] = useState<[number, number] | null>(null);
  const [highlightedPoint, setHighlightedPoint] = useState<number | null>(null);

  // Prepare chart data
  const chartData = useMemo(() => {
    return data.map((value, index) => ({
      index: index + 1,
      value,
      ucl: limits.ucl,
      lcl: limits.lcl,
      centerline: limits.centerline,
      usl: limits.usl,
      lsl: limits.lsl,
      zone_1_upper: limits.centerline + (limits.sigma || 0),
      zone_1_lower: limits.centerline - (limits.sigma || 0),
      zone_2_upper: limits.centerline + 2 * (limits.sigma || 0),
      zone_2_lower: limits.centerline - 2 * (limits.sigma || 0),
      alert: alerts.some(a => a.points_involved.includes(index))
    }));
  }, [data, limits, alerts]);

  // Alert points for scatter overlay
  const alertPoints = useMemo(() => {
    return chartData.filter(d => d.alert);
  }, [chartData]);

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        <div className="flex items-center gap-2">
          <button className="p-2 hover:bg-gray-100 rounded-lg" title="Zoom Reset">
            <ZoomIn className="w-4 h-4 text-gray-600" />
          </button>
          <button className="p-2 hover:bg-gray-100 rounded-lg" title="Download">
            <Download className="w-4 h-4 text-gray-600" />
          </button>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="index"
            label={{ value: 'Sample Number', position: 'insideBottom', offset: -5 }}
            stroke="#6b7280"
          />
          <YAxis
            label={{ value: 'Measurement Value', angle: -90, position: 'insideLeft' }}
            stroke="#6b7280"
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload || !payload.length) return null;
              const data = payload[0].payload;
              return (
                <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200">
                  <p className="font-semibold text-gray-900">Point {data.index}</p>
                  <p className="text-sm text-gray-600">Value: {data.value.toFixed(4)}</p>
                  {data.alert && (
                    <p className="text-sm text-red-600 font-medium mt-1">⚠️ Alert</p>
                  )}
                </div>
              );
            }}
          />

          {/* Specification limits (if present) */}
          {limits.usl && (
            <ReferenceLine
              y={limits.usl}
              stroke="#dc2626"
              strokeWidth={2}
              strokeDasharray="5 5"
              label={{ value: 'USL', position: 'right', fill: '#dc2626' }}
            />
          )}
          {limits.lsl && (
            <ReferenceLine
              y={limits.lsl}
              stroke="#dc2626"
              strokeWidth={2}
              strokeDasharray="5 5"
              label={{ value: 'LSL', position: 'right', fill: '#dc2626' }}
            />
          )}

          {/* Control limits */}
          <ReferenceLine
            y={limits.ucl}
            stroke="#ef4444"
            strokeWidth={2}
            label={{ value: 'UCL', position: 'right', fill: '#ef4444' }}
          />
          <ReferenceLine
            y={limits.lcl}
            stroke="#ef4444"
            strokeWidth={2}
            label={{ value: 'LCL', position: 'right', fill: '#ef4444' }}
          />
          <ReferenceLine
            y={limits.centerline}
            stroke="#3b82f6"
            strokeWidth={2}
            label={{ value: 'CL', position: 'right', fill: '#3b82f6' }}
          />

          {/* Zone boundaries */}
          {limits.sigma && (
            <>
              <ReferenceLine y={limits.centerline + limits.sigma} stroke="#94a3b8" strokeDasharray="2 2" />
              <ReferenceLine y={limits.centerline - limits.sigma} stroke="#94a3b8" strokeDasharray="2 2" />
              <ReferenceLine y={limits.centerline + 2 * limits.sigma} stroke="#94a3b8" strokeDasharray="2 2" />
              <ReferenceLine y={limits.centerline - 2 * limits.sigma} stroke="#94a3b8" strokeDasharray="2 2" />
            </>
          )}

          {/* Data line */}
          <Line
            type="monotone"
            dataKey="value"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={{ r: 4, fill: '#3b82f6' }}
            activeDot={{ r: 6 }}
          />

          {/* Alert overlay */}
          {alertPoints.length > 0 && (
            <Scatter
              data={alertPoints}
              fill="#ef4444"
              shape="circle"
            />
          )}
        </LineChart>
      </ResponsiveContainer>

      {/* Chart info */}
      <div className="grid grid-cols-4 gap-4 pt-4 border-t border-gray-200">
        <div>
          <p className="text-xs text-gray-500">UCL</p>
          <p className="text-sm font-semibold text-gray-900">{formatNumber(limits.ucl)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Centerline</p>
          <p className="text-sm font-semibold text-gray-900">{formatNumber(limits.centerline)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">LCL</p>
          <p className="text-sm font-semibold text-gray-900">{formatNumber(limits.lcl)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Sigma</p>
          <p className="text-sm font-semibold text-gray-900">{formatNumber(limits.sigma || 0)}</p>
        </div>
      </div>
    </div>
  );
};


// ============================================================================
// Alerts Dashboard
// ============================================================================

interface AlertsDashboardProps {
  alerts: Alert[];
  onAlertClick: (alert: Alert) => void;
}

export const AlertsDashboard: React.FC<AlertsDashboardProps> = ({ alerts, onAlertClick }) => {
  const [filter, setFilter] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState<string>('');

  const filteredAlerts = useMemo(() => {
    let filtered = alerts;
    
    if (filter !== 'all') {
      filtered = filtered.filter(a => a.severity === filter);
    }
    
    if (searchTerm) {
      filtered = filtered.filter(a =>
        a.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
        a.rule.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    return filtered;
  }, [alerts, filter, searchTerm]);

  // Count by severity
  const severityCounts = useMemo(() => {
    return {
      critical: alerts.filter(a => a.severity === 'critical').length,
      high: alerts.filter(a => a.severity === 'high').length,
      medium: alerts.filter(a => a.severity === 'medium').length,
      low: alerts.filter(a => a.severity === 'low').length
    };
  }, [alerts]);

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
          <Bell className="w-5 h-5" />
          Alerts & Violations
        </h3>
        <div className="flex items-center gap-2">
          <button className="p-2 hover:bg-gray-100 rounded-lg" title="Refresh">
            <RefreshCw className="w-4 h-4 text-gray-600" />
          </button>
          <button className="p-2 hover:bg-gray-100 rounded-lg" title="Settings">
            <Settings className="w-4 h-4 text-gray-600" />
          </button>
        </div>
      </div>

      {/* Severity summary */}
      <div className="grid grid-cols-4 gap-3">
        <button
          onClick={() => setFilter('critical')}
          className={`p-3 rounded-lg border-2 transition-all ${
            filter === 'critical'
              ? 'border-red-500 bg-red-50'
              : 'border-gray-200 hover:border-gray-300'
          }`}
        >
          <p className="text-2xl font-bold text-red-600">{severityCounts.critical}</p>
          <p className="text-xs text-gray-600">Critical</p>
        </button>
        <button
          onClick={() => setFilter('high')}
          className={`p-3 rounded-lg border-2 transition-all ${
            filter === 'high'
              ? 'border-orange-500 bg-orange-50'
              : 'border-gray-200 hover:border-gray-300'
          }`}
        >
          <p className="text-2xl font-bold text-orange-600">{severityCounts.high}</p>
          <p className="text-xs text-gray-600">High</p>
        </button>
        <button
          onClick={() => setFilter('medium')}
          className={`p-3 rounded-lg border-2 transition-all ${
            filter === 'medium'
              ? 'border-yellow-500 bg-yellow-50'
              : 'border-gray-200 hover:border-gray-300'
          }`}
        >
          <p className="text-2xl font-bold text-yellow-600">{severityCounts.medium}</p>
          <p className="text-xs text-gray-600">Medium</p>
        </button>
        <button
          onClick={() => setFilter('low')}
          className={`p-3 rounded-lg border-2 transition-all ${
            filter === 'low'
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-200 hover:border-gray-300'
          }`}
        >
          <p className="text-2xl font-bold text-blue-600">{severityCounts.low}</p>
          <p className="text-xs text-gray-600">Low</p>
        </button>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
        <input
          type="text"
          placeholder="Search alerts..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>

      {/* Alerts list */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {filteredAlerts.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Bell className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>No alerts found</p>
          </div>
        ) : (
          filteredAlerts.map((alert) => (
            <div
              key={alert.id}
              onClick={() => onAlertClick(alert)}
              className={`p-4 rounded-lg border cursor-pointer hover:shadow-md transition-all ${getSeverityColor(alert.severity)}`}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1">
                  <p className="font-medium text-sm">{alert.message}</p>
                  <p className="text-xs opacity-75 mt-1">
                    Rule: {alert.rule.replace('_', ' ').toUpperCase()} • Points: {alert.points_involved.join(', ')}
                  </p>
                </div>
                <div className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(alert.severity)}`}>
                  {alert.severity}
                </div>
              </div>
              
              {alert.suggested_actions.length > 0 && (
                <div className="mt-2 pt-2 border-t border-current border-opacity-20">
                  <p className="text-xs font-medium mb-1">Suggested Actions:</p>
                  <ul className="text-xs space-y-1 opacity-90">
                    {alert.suggested_actions.slice(0, 2).map((action, i) => (
                      <li key={i}>• {action}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {filter !== 'all' && (
        <button
          onClick={() => setFilter('all')}
          className="w-full py-2 text-sm text-blue-600 hover:text-blue-700 font-medium"
        >
          Show All Alerts
        </button>
      )}
    </div>
  );
};


// ============================================================================
// Process Capability Widget
// ============================================================================

interface CapabilityWidgetProps {
  capability: ProcessCapability;
  statistics: {
    mean: number;
    std: number;
    min: number;
    max: number;
  };
}

export const CapabilityWidget: React.FC<CapabilityWidgetProps> = ({ capability, statistics }) => {
  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-4">
      <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
        <BarChart3 className="w-5 h-5" />
        Process Capability
      </h3>

      {/* Capability status */}
      <div className={`p-4 rounded-lg ${capability.is_capable ? 'bg-green-50' : 'bg-red-50'}`}>
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">Capability Status</span>
          {capability.is_capable ? (
            <CheckCircle className="w-5 h-5 text-green-600" />
          ) : (
            <AlertTriangle className="w-5 h-5 text-red-600" />
          )}
        </div>
        <p className={`text-lg font-bold ${capability.is_capable ? 'text-green-900' : 'text-red-900'}`}>
          {capability.is_capable ? 'Process Capable' : 'Process Not Capable'}
        </p>
      </div>

      {/* Indices grid */}
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 bg-gray-50 rounded-lg">
          <p className="text-xs text-gray-500 mb-1">Cp</p>
          <p className="text-2xl font-bold text-gray-900">{capability.cp.toFixed(3)}</p>
          <p className="text-xs text-gray-600 mt-1">Potential</p>
        </div>
        <div className="p-3 bg-gray-50 rounded-lg">
          <p className="text-xs text-gray-500 mb-1">Cpk</p>
          <p className="text-2xl font-bold text-gray-900">{capability.cpk.toFixed(3)}</p>
          <p className="text-xs text-gray-600 mt-1">Actual</p>
        </div>
        <div className="p-3 bg-gray-50 rounded-lg">
          <p className="text-xs text-gray-500 mb-1">Pp</p>
          <p className="text-2xl font-bold text-gray-900">{capability.pp.toFixed(3)}</p>
          <p className="text-xs text-gray-600 mt-1">Performance</p>
        </div>
        <div className="p-3 bg-gray-50 rounded-lg">
          <p className="text-xs text-gray-500 mb-1">Ppk</p>
          <p className="text-2xl font-bold text-gray-900">{capability.ppk.toFixed(3)}</p>
          <p className="text-xs text-gray-600 mt-1">Actual Perf</p>
        </div>
      </div>

      {/* Sigma level and DPMO */}
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 bg-blue-50 rounded-lg">
          <p className="text-xs text-blue-600 mb-1">Sigma Level</p>
          <p className="text-2xl font-bold text-blue-900">{capability.sigma_level.toFixed(2)}σ</p>
        </div>
        <div className="p-3 bg-purple-50 rounded-lg">
          <p className="text-xs text-purple-600 mb-1">DPMO</p>
          <p className="text-2xl font-bold text-purple-900">{Math.round(capability.dpmo)}</p>
        </div>
      </div>

      {/* Comments */}
      {capability.comments && capability.comments.length > 0 && (
        <div className="pt-3 border-t border-gray-200">
          <p className="text-xs font-medium text-gray-700 mb-2">Analysis:</p>
          <ul className="space-y-1">
            {capability.comments.map((comment, i) => (
              <li key={i} className="text-xs text-gray-600">• {comment}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};


// ============================================================================
// Trend Analysis Widget
// ============================================================================

interface TrendWidgetProps {
  trend: TrendAnalysis;
  data: number[];
}

export const TrendWidget: React.FC<TrendWidgetProps> = ({ trend, data }) => {
  const trendData = useMemo(() => {
    return data.map((value, index) => ({
      index: index + 1,
      value,
      predicted: index < data.length ? null : trend.predicted_values[index - data.length]
    }));
  }, [data, trend.predicted_values]);

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-4">
      <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
        <Activity className="w-5 h-5" />
        Trend Analysis
      </h3>

      {/* Trend status */}
      <div className={`p-4 rounded-lg ${trend.detected ? 'bg-yellow-50' : 'bg-green-50'}`}>
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">Trend Status</span>
          {trend.direction === 'increasing' && <TrendingUp className="w-5 h-5 text-red-600" />}
          {trend.direction === 'decreasing' && <TrendingDown className="w-5 h-5 text-blue-600" />}
          {trend.direction === 'stable' && <CheckCircle className="w-5 h-5 text-green-600" />}
        </div>
        <p className="text-lg font-bold text-gray-900 capitalize">{trend.direction}</p>
        {trend.detected && (
          <p className="text-sm text-gray-600 mt-1">
            Slope: {trend.slope.toFixed(4)} (p={trend.p_value.toFixed(4)})
          </p>
        )}
      </div>

      {/* Trend chart */}
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={trendData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="index" stroke="#6b7280" />
          <YAxis stroke="#6b7280" />
          <Tooltip />
          <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} dot={false} />
          {trend.detected && (
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="#ef4444"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
            />
          )}
        </LineChart>
      </ResponsiveContainer>

      {/* Changepoints */}
      {trend.changepoints && trend.changepoints.length > 0 && (
        <div className="pt-3 border-t border-gray-200">
          <p className="text-xs font-medium text-gray-700 mb-2">Change Points Detected:</p>
          <p className="text-sm text-gray-600">
            Points: {trend.changepoints.map(p => p + 1).join(', ')}
          </p>
        </div>
      )}
    </div>
  );
};


// ============================================================================
// Root Cause Analysis Panel
// ============================================================================

interface RootCauseProps {
  rootCauseAnalysis: {
    likely_causes: string[];
    investigate: string[];
    preventive_actions: string[];
  };
}

export const RootCausePanel: React.FC<RootCauseProps> = ({ rootCauseAnalysis }) => {
  const [expandedSection, setExpandedSection] = useState<string>('causes');

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-4">
      <h3 className="text-lg font-semibold text-gray-900">Root Cause Analysis</h3>

      {/* Likely causes */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <button
          onClick={() => setExpandedSection(expandedSection === 'causes' ? '' : 'causes')}
          className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
        >
          <span className="font-medium text-gray-900">Likely Causes</span>
          {expandedSection === 'causes' ? (
            <ChevronDown className="w-5 h-5 text-gray-600" />
          ) : (
            <ChevronRight className="w-5 h-5 text-gray-600" />
          )}
        </button>
        {expandedSection === 'causes' && (
          <div className="p-4 space-y-2">
            {rootCauseAnalysis.likely_causes.map((cause, i) => (
              <div key={i} className="flex items-start gap-2">
                <div className="mt-1 w-2 h-2 rounded-full bg-red-500 flex-shrink-0" />
                <p className="text-sm text-gray-700">{cause}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Investigate */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <button
          onClick={() => setExpandedSection(expandedSection === 'investigate' ? '' : 'investigate')}
          className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
        >
          <span className="font-medium text-gray-900">Areas to Investigate</span>
          {expandedSection === 'investigate' ? (
            <ChevronDown className="w-5 h-5 text-gray-600" />
          ) : (
            <ChevronRight className="w-5 h-5 text-gray-600" />
          )}
        </button>
        {expandedSection === 'investigate' && (
          <div className="p-4 space-y-2">
            {rootCauseAnalysis.investigate.map((item, i) => (
              <div key={i} className="flex items-start gap-2">
                <div className="mt-1 w-2 h-2 rounded-full bg-yellow-500 flex-shrink-0" />
                <p className="text-sm text-gray-700">{item}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Preventive actions */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <button
          onClick={() => setExpandedSection(expandedSection === 'preventive' ? '' : 'preventive')}
          className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
        >
          <span className="font-medium text-gray-900">Preventive Actions</span>
          {expandedSection === 'preventive' ? (
            <ChevronDown className="w-5 h-5 text-gray-600" />
          ) : (
            <ChevronRight className="w-5 h-5 text-gray-600" />
          )}
        </button>
        {expandedSection === 'preventive' && (
          <div className="p-4 space-y-2">
            {rootCauseAnalysis.preventive_actions.map((action, i) => (
              <div key={i} className="flex items-start gap-2">
                <div className="mt-1 w-2 h-2 rounded-full bg-green-500 flex-shrink-0" />
                <p className="text-sm text-gray-700">{action}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};


// ============================================================================
// Main SPC Dashboard
// ============================================================================

interface SPCDashboardProps {
  results: SPCResults;
  data: number[];
  onRefresh?: () => void;
}

export const SPCDashboard: React.FC<SPCDashboardProps> = ({ results, data, onRefresh }) => {
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [selectedChart, setSelectedChart] = useState<string>('i');

  // Get primary chart limits
  const primaryLimits = results.control_limits[selectedChart] || results.control_limits['i'];

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Statistical Process Control</h1>
              <p className="text-sm text-gray-600 mt-1">
                Real-time process monitoring and analysis
              </p>
            </div>
            <div className="flex items-center gap-3">
              {getStatusBadge(results.status)}
              {onRefresh && (
                <button
                  onClick={onRefresh}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  Refresh
                </button>
              )}
            </div>
          </div>

          {/* Summary stats */}
          <div className="grid grid-cols-5 gap-4">
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-xs text-gray-500">Mean</p>
              <p className="text-lg font-bold text-gray-900">{results.statistics.mean.toFixed(4)}</p>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-xs text-gray-500">Std Dev</p>
              <p className="text-lg font-bold text-gray-900">{results.statistics.std.toFixed(4)}</p>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-xs text-gray-500">Range</p>
              <p className="text-lg font-bold text-gray-900">{results.statistics.range.toFixed(4)}</p>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-xs text-gray-500">Samples</p>
              <p className="text-lg font-bold text-gray-900">{results.n_points}</p>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-xs text-gray-500">Alerts</p>
              <p className="text-lg font-bold text-red-600">{results.alerts.length}</p>
            </div>
          </div>
        </div>

        {/* Recommendations */}
        {results.recommendations && results.recommendations.length > 0 && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="font-semibold text-blue-900 mb-2 flex items-center gap-2">
              <Info className="w-4 h-4" />
              Recommendations
            </h4>
            <ul className="space-y-1">
              {results.recommendations.map((rec, i) => (
                <li key={i} className="text-sm text-blue-800">{rec}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Main content grid */}
        <div className="grid grid-cols-3 gap-6">
          {/* Control chart - spans 2 columns */}
          <div className="col-span-2">
            <ControlChart
              data={data}
              limits={primaryLimits}
              alerts={results.alerts}
              title="Control Chart"
              chartType={selectedChart}
            />
          </div>

          {/* Alerts dashboard */}
          <div>
            <AlertsDashboard
              alerts={results.alerts}
              onAlertClick={setSelectedAlert}
            />
          </div>
        </div>

        {/* Secondary widgets grid */}
        <div className="grid grid-cols-3 gap-6">
          <CapabilityWidget
            capability={results.capability}
            statistics={results.statistics}
          />
          
          <TrendWidget
            trend={results.trend}
            data={data}
          />

          {results.root_cause_analysis && (
            <RootCausePanel rootCauseAnalysis={results.root_cause_analysis} />
          )}
        </div>

        {/* Alert detail modal */}
        {selectedAlert && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Alert Details</h3>
                <button
                  onClick={() => setSelectedAlert(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-4">
                <div className={`p-4 rounded-lg ${getSeverityColor(selectedAlert.severity)}`}>
                  <p className="font-medium">{selectedAlert.message}</p>
                  <p className="text-sm mt-1">
                    Rule: {selectedAlert.rule} • Value: {selectedAlert.value.toFixed(4)}
                  </p>
                </div>

                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Suggested Actions</h4>
                  <ul className="space-y-2">
                    {selectedAlert.suggested_actions.map((action, i) => (
                      <li key={i} className="text-sm text-gray-700 flex items-start gap-2">
                        <span className="text-blue-600">•</span>
                        {action}
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Possible Root Causes</h4>
                  <ul className="space-y-2">
                    {selectedAlert.root_causes.map((cause, i) => (
                      <li key={i} className="text-sm text-gray-700 flex items-start gap-2">
                        <span className="text-red-600">•</span>
                        {cause}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};



// ============================================================================
// Page Wrapper with Real-time SPC Data Generation
// ============================================================================

const SPCPage: React.FC = () => {
  const [data, setData] = useState<number[]>([]);
  const [results, setResults] = useState<SPCResults | null>(null);

  useEffect(() => {
    generateSPCData();
  }, []);

  const generateSPCData = () => {
    // Generate 100 realistic process measurements
    const baseValue = 100;
    const processStd = 2;
    const generatedData: number[] = [];

    for (let i = 0; i < 100; i++) {
      let value = baseValue + (Math.random() - 0.5) * processStd * 2;
      
      // Introduce realistic process variations
      if (i >= 40 && i <= 45) value += 3; // Process shift
      if (i === 70 || i === 71) value -= 4.5; // Outliers
      
      generatedData.push(value);
    }

    setData(generatedData);

    // Calculate comprehensive statistics
    const n = generatedData.length;
    const mean = generatedData.reduce((a, b) => a + b) / n;
    const variance = generatedData.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
    const std = Math.sqrt(variance);
    const sorted = [...generatedData].sort((a, b) => a - b);
    const min = sorted[0];
    const max = sorted[n - 1];
    const median = sorted[Math.floor(n / 2)];

    // Calculate moving ranges for I-MR chart
    const movingRanges: number[] = [];
    for (let i = 1; i < n; i++) {
      movingRanges.push(Math.abs(generatedData[i] - generatedData[i - 1]));
    }
    const avgMR = movingRanges.reduce((a, b) => a + b) / movingRanges.length;

    // Control limits for I chart
    const ucl_i = mean + 2.66 * avgMR;
    const lcl_i = Math.max(0, mean - 2.66 * avgMR);

    // Control limits for MR chart  
    const ucl_mr = 3.267 * avgMR;
    const lcl_mr = 0;

    // Generate alerts based on Western Electric rules
    const alerts: Alert[] = [];
    generatedData.forEach((value, idx) => {
      if (value > ucl_i) {
        alerts.push({
          id: `alert_ucl_$${'{'}idx${'}'}`,
          timestamp: new Date(Date.now() - (n - idx) * 3600000).toISOString(),
          rule: 'Rule 1: Point beyond UCL',
          severity: 'critical',
          message: `Measurement $${'{'}idx + 1${'}'} ($${'{'}value.toFixed(2)${'}'}) exceeds upper control limit ($${'{'}ucl_i.toFixed(2)${'}'})`,
          value,
          points_involved: [idx],
          suggested_actions: [
            'Stop process and investigate immediately',
            'Check equipment calibration and settings',
            'Verify measurement accuracy',
            'Review recent process parameter changes'
          ],
          root_causes: [
            'Equipment malfunction or drift',
            'Raw material quality variation',
            'Operator error or procedure deviation',
            'Environmental condition change'
          ]
        });
      }
      if (value < lcl_i) {
        alerts.push({
          id: `alert_lcl_$${'{'}idx${'}'}`,
          timestamp: new Date(Date.now() - (n - idx) * 3600000).toISOString(),
          rule: 'Rule 1: Point below LCL',
          severity: 'high',
          message: `Measurement $${'{'}idx + 1${'}'} ($${'{'}value.toFixed(2)${'}'}) below lower control limit ($${'{'}lcl_i.toFixed(2)${'}'})`,
          value,
          points_involved: [idx],
          suggested_actions: [
            'Investigate process conditions',
            'Check sensor calibration',
            'Review process setpoints'
          ],
          root_causes: [
            'Process undercorrection',
            'Measurement systematic error',
            'Raw material deficiency'
          ]
        });
      }
    });

    // Check for runs and trends (additional Western Electric rules)
    let consecutiveSameSide = 0;
    let lastSide: 'above' | 'below' | null = null;
    
    generatedData.forEach((value, idx) => {
      const currentSide = value > mean ? 'above' : 'below';
      if (currentSide === lastSide) {
        consecutiveSameSide++;
        if (consecutiveSameSide >= 8) {
          alerts.push({
            id: `alert_run_$${'{'}idx${'}'}`,
            timestamp: new Date(Date.now() - (n - idx) * 3600000).toISOString(),
            rule: 'Rule 4: 8+ consecutive points on same side',
            severity: 'medium',
            message: `$${'{'}consecutiveSameSide${'}'} consecutive points $${'{'}currentSide${'}'} centerline ending at point $${'{'}idx + 1${'}'}`,
            value,
            points_involved: Array.from({length: consecutiveSameSide}, (_, i) => idx - i),
            suggested_actions: [
              'Check for process drift or shift',
              'Verify process centering',
              'Review control chart calculation'
            ],
            root_causes: [
              'Process mean shift',
              'Systematic bias in measurement',
              'Control limits miscalculation'
            ]
          });
        }
      } else {
        consecutiveSameSide = 1;
        lastSide = currentSide;
      }
    });

    // Calculate process capability
    const usl = mean + 4 * std;
    const lsl = mean - 4 * std;
    const target = mean;

    const cp = (usl - lsl) / (6 * std);
    const cpk = Math.min((usl - mean) / (3 * std), (mean - lsl) / (3 * std));
    const pp = cp; // For simplicity, using same as Cp
    const ppk = cpk;
    const sigmaLevel = 3 * cpk;
    const dpmo = 1000000 * (1 - 0.9973); // Approximation

    // Trend analysis
    const xValues = Array.from({length: n}, (_, i) => i);
    const sumX = xValues.reduce((a, b) => a + b);
    const sumY = generatedData.reduce((a, b) => a + b);
    const sumXY = xValues.reduce((sum, x, i) => sum + x * generatedData[i], 0);
    const sumX2 = xValues.reduce((sum, x) => sum + x * x, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    const predictedValues = xValues.map(x => intercept + slope * x);

    // Build comprehensive SPC results
    const spcResults: SPCResults = {
      timestamp: new Date().toISOString(),
      chart_type: 'i_mr',
      n_points: n,
      statistics: {
        mean,
        std,
        min,
        max,
        median,
        range: max - min
      },
      control_limits: {
        i: {
          ucl: ucl_i,
          lcl: lcl_i,
          centerline: mean,
          usl,
          lsl,
          sigma: std
        },
        mr: {
          ucl: ucl_mr,
          lcl: lcl_mr,
          centerline: avgMR,
          sigma: avgMR
        }
      },
      alerts,
      capability: {
        cp,
        cpk,
        pp,
        ppk,
        sigma_level: sigmaLevel,
        dpmo,
        is_capable: cpk >= 1.33,
        comments: [
          cpk >= 1.67 ? 'Process is highly capable' : cpk >= 1.33 ? 'Process is capable' : 'Process needs improvement',
          `Cpk = $${'{'}cpk.toFixed(2)${'}'} indicates $${'{'}cpk >= 1.33 ? 'good' : 'poor'${'}'} process centering`,
          alerts.length > 0 ? `$${'{'}alerts.length${'}'} active alerts require attention` : 'No active alerts'
        ]
      },
      trend: {
        detected: Math.abs(slope) > 0.01,
        direction: Math.abs(slope) < 0.01 ? 'stable' : slope > 0 ? 'increasing' : 'decreasing',
        slope,
        p_value: 0.05,
        predicted_values: predictedValues,
        changepoints: [40, 70]
      },
      status: alerts.some(a => a.severity === 'critical') ? 'out_of_control' :
              alerts.length > 0 ? 'warning' : 'in_control',
      recommendations: [
        alerts.length === 0 ? 'Process is stable and in control' : `Address $${'{'}alerts.length${'}'} active alerts`,
        cpk < 1.33 ? 'Consider process improvement initiatives to increase capability' : 'Maintain current process controls',
        'Continue monitoring for sustained performance',
        'Review control limits quarterly or after process changes'
      ]
    };

    setResults(spcResults);
  };

  const handleRefresh = () => {
    generateSPCData();
  };

  if (!results || data.length === 0) {
    return (
      <div className="min-h-screen bg-gray-50 p-6 flex items-center justify-center">
        <div className="text-center">
          <Activity className="w-12 h-12 text-blue-600 animate-spin mx-auto mb-4" />
          <p className="text-gray-600 font-medium">Generating SPC Analysis...</p>
          <p className="text-sm text-gray-500 mt-2">Calculating control limits and process capability</p>
        </div>
      </div>
    );
  }

  return <SPCDashboard results={results} data={data} onRefresh={handleRefresh} />;
};

// Export the page wrapper instead of SPCDashboard directly
export default SPCPage;

