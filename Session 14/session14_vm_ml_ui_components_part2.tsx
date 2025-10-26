/**
 * SESSION 14: ML/VM HUB - UI COMPONENTS (Part 2)
 * 
 * Anomaly Detection, Drift Monitoring, and Time Series Components
 * 
 * @author Semiconductor Lab Platform Team
 * @date October 2024
 */

import React, { useState, useMemo } from 'react';
import {
  LineChart, Line, ScatterChart, Scatter, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Area, AreaChart, ComposedChart, ReferenceLine, Cell, RadarChart,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import {
  AlertCircle, CheckCircle, TrendingUp, TrendingDown,
  AlertTriangle, Info, Eye, Filter, Calendar, RefreshCw
} from 'lucide-react';

// Import types from part 1
interface AnomalyDetection {
  id: number;
  timestamp: string;
  is_anomaly: boolean;
  anomaly_score: number;
  anomaly_type: string;
  features: Record<string, number>;
  feature_contributions: Record<string, number>;
  likely_causes: string[];
  resolved: boolean;
}

interface DriftReport {
  id: number;
  drift_type: string;
  drift_detected: boolean;
  drift_score: number;
  feature_drifts: Record<string, any>;
  recommended_action: string;
  created_at: string;
}

interface TimeSeriesForecast {
  ds: string;
  yhat: number;
  yhat_lower: number;
  yhat_upper: number;
  trend: number;
}

// ============================================================================
// ANOMALY DETECTION MONITOR
// ============================================================================

interface AnomalyMonitorProps {
  anomalies: AnomalyDetection[];
  onResolve: (id: number) => Promise<void>;
  onInvestigate: (id: number) => void;
}

export const AnomalyMonitor: React.FC<AnomalyMonitorProps> = ({
  anomalies,
  onResolve,
  onInvestigate
}) => {
  const [filter, setFilter] = useState<'all' | 'unresolved' | 'resolved'>('unresolved');
  const [selectedAnomaly, setSelectedAnomaly] = useState<AnomalyDetection | null>(null);

  const filteredAnomalies = useMemo(() => {
    let filtered = anomalies;
    if (filter === 'unresolved') {
      filtered = anomalies.filter(a => !a.resolved);
    } else if (filter === 'resolved') {
      filtered = anomalies.filter(a => a.resolved);
    }
    return filtered.sort((a, b) => 
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  }, [anomalies, filter]);

  const stats = useMemo(() => {
    const total = anomalies.length;
    const unresolved = anomalies.filter(a => !a.resolved).length;
    const last24h = anomalies.filter(a => 
      new Date(a.timestamp).getTime() > Date.now() - 24 * 60 * 60 * 1000
    ).length;

    return { total, unresolved, last24h };
  }, [anomalies]);

  // Anomaly types distribution
  const typeDistribution = useMemo(() => {
    const dist: Record<string, number> = {};
    anomalies.forEach(a => {
      dist[a.anomaly_type] = (dist[a.anomaly_type] || 0) + 1;
    });
    return Object.entries(dist).map(([type, count]) => ({ type, count }));
  }, [anomalies]);

  return (
    <div className="space-y-6">
      {/* Header & Stats */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Anomaly Detection</h2>
        <div className="grid grid-cols-3 gap-4 mt-4">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <div className="text-sm text-gray-600">Total Anomalies</div>
            <div className="text-2xl font-bold text-gray-900 mt-1">{stats.total}</div>
          </div>
          <div className="bg-white rounded-lg shadow-sm border border-amber-200 p-4">
            <div className="text-sm text-amber-700">Unresolved</div>
            <div className="text-2xl font-bold text-amber-600 mt-1">{stats.unresolved}</div>
          </div>
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <div className="text-sm text-gray-600">Last 24 Hours</div>
            <div className="text-2xl font-bold text-gray-900 mt-1">{stats.last24h}</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Anomaly List */}
        <div className="lg:col-span-2 space-y-4">
          {/* Filters */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <div className="flex items-center gap-4">
              <Filter size={16} className="text-gray-400" />
              <div className="flex gap-2">
                {['all', 'unresolved', 'resolved'].map(f => (
                  <button
                    key={f}
                    onClick={() => setFilter(f as any)}
                    className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                      filter === f
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {f.charAt(0).toUpperCase() + f.slice(1)}
                  </button>
                ))}
              </div>
              <div className="ml-auto text-sm text-gray-600">
                {filteredAnomalies.length} anomalies
              </div>
            </div>
          </div>

          {/* Anomaly Cards */}
          <div className="space-y-3">
            {filteredAnomalies.map(anomaly => (
              <div
                key={anomaly.id}
                className={`bg-white rounded-lg shadow-sm border-2 transition-all cursor-pointer ${
                  selectedAnomaly?.id === anomaly.id
                    ? 'border-blue-500'
                    : anomaly.resolved
                    ? 'border-gray-200'
                    : 'border-amber-200'
                }`}
                onClick={() => setSelectedAnomaly(anomaly)}
              >
                <div className="p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <AlertCircle
                          size={18}
                          className={anomaly.resolved ? 'text-gray-400' : 'text-amber-500'}
                        />
                        <span className="font-semibold">Anomaly #{anomaly.id}</span>
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                          anomaly.anomaly_type === 'point' ? 'bg-red-100 text-red-700' :
                          anomaly.anomaly_type === 'contextual' ? 'bg-orange-100 text-orange-700' :
                          'bg-purple-100 text-purple-700'
                        }`}>
                          {anomaly.anomaly_type}
                        </span>
                      </div>
                      <div className="text-xs text-gray-500">
                        {new Date(anomaly.timestamp).toLocaleString()}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-amber-600">
                        {anomaly.anomaly_score.toFixed(2)}
                      </div>
                      <div className="text-xs text-gray-500">Score</div>
                    </div>
                  </div>

                  {/* Top Contributing Features */}
                  <div className="mb-3">
                    <div className="text-xs font-medium text-gray-700 mb-1">
                      Top Contributing Features:
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(anomaly.feature_contributions)
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 3)
                        .map(([feature, contribution]) => (
                          <span
                            key={feature}
                            className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-xs"
                          >
                            {feature}: {contribution.toFixed(2)}
                          </span>
                        ))}
                    </div>
                  </div>

                  {/* Likely Causes */}
                  {anomaly.likely_causes.length > 0 && (
                    <div className="mb-3">
                      <div className="text-xs font-medium text-gray-700 mb-1">
                        Likely Causes:
                      </div>
                      <ul className="text-xs text-gray-600 space-y-1">
                        {anomaly.likely_causes.slice(0, 2).map((cause, idx) => (
                          <li key={idx} className="flex items-start gap-1">
                            <span className="text-gray-400">•</span>
                            <span>{cause}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Actions */}
                  <div className="flex gap-2 pt-3 border-t border-gray-200">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onInvestigate(anomaly.id);
                      }}
                      className="flex-1 px-3 py-1.5 bg-blue-50 text-blue-700 rounded text-xs font-medium hover:bg-blue-100 flex items-center justify-center gap-1"
                    >
                      <Eye size={14} />
                      Investigate
                    </button>
                    {!anomaly.resolved && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onResolve(anomaly.id);
                        }}
                        className="flex-1 px-3 py-1.5 bg-green-50 text-green-700 rounded text-xs font-medium hover:bg-green-100 flex items-center justify-center gap-1"
                      >
                        <CheckCircle size={14} />
                        Resolve
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Anomaly Analysis Panel */}
        <div className="space-y-4">
          {/* Type Distribution */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <h3 className="font-semibold mb-4">Anomaly Types</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={typeDistribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="type" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Selected Anomaly Details */}
          {selectedAnomaly && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
              <h3 className="font-semibold mb-4">Feature Contributions</h3>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={
                  Object.entries(selectedAnomaly.feature_contributions)
                    .slice(0, 6)
                    .map(([feature, value]) => ({
                      feature: feature.length > 15 ? feature.substring(0, 12) + '...' : feature,
                      value: value
                    }))
                }>
                  <PolarGrid stroke="#e5e7eb" />
                  <PolarAngleAxis dataKey="feature" tick={{ fontSize: 10 }} />
                  <PolarRadiusAxis />
                  <Radar
                    name="Contribution"
                    dataKey="value"
                    stroke="#f59e0b"
                    fill="#f59e0b"
                    fillOpacity={0.6}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Timeline */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <h3 className="font-semibold mb-4">Anomaly Timeline</h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={
                anomalies
                  .slice(-20)
                  .map(a => ({
                    timestamp: new Date(a.timestamp).toLocaleTimeString(),
                    score: a.anomaly_score,
                    resolved: a.resolved ? 0 : 1
                  }))
              }>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="timestamp" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="score"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={false}
                />
                <ReferenceLine y={0.5} stroke="#ef4444" strokeDasharray="3 3" label="Threshold" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// DRIFT MONITORING DASHBOARD
// ============================================================================

interface DriftMonitoringProps {
  reports: DriftReport[];
  onRefresh: () => Promise<void>;
}

export const DriftMonitoring: React.FC<DriftMonitoringProps> = ({
  reports,
  onRefresh
}) => {
  const [refreshing, setRefreshing] = useState(false);
  const [selectedReport, setSelectedReport] = useState<DriftReport | null>(null);

  const latestReport = reports[0];

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await onRefresh();
    } finally {
      setRefreshing(false);
    }
  };

  // Calculate drift severity
  const driftSeverity = (score: number) => {
    if (score > 0.5) return { level: 'high', color: 'red' };
    if (score > 0.3) return { level: 'medium', color: 'amber' };
    if (score > 0.1) return { level: 'low', color: 'yellow' };
    return { level: 'none', color: 'green' };
  };

  const severity = latestReport ? driftSeverity(latestReport.drift_score) : null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Model Drift Monitoring</h2>
          <p className="text-sm text-gray-600 mt-1">
            Track distribution shifts and model performance degradation
          </p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:bg-gray-100 flex items-center gap-2"
        >
          <RefreshCw size={16} className={refreshing ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Current Status */}
      {latestReport && severity && (
        <div className={`bg-${severity.color}-50 border-2 border-${severity.color}-200 rounded-lg p-6`}>
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <AlertTriangle
                  size={24}
                  className={`text-${severity.color}-600`}
                />
                <h3 className="text-lg font-semibold">
                  {latestReport.drift_detected ? 'Drift Detected' : 'No Significant Drift'}
                </h3>
              </div>
              <p className="text-sm text-gray-700 mb-4">
                {latestReport.drift_detected
                  ? `Model performance has degraded. ${latestReport.recommended_action}.`
                  : 'Model is performing within expected parameters.'}
              </p>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <div className="text-xs text-gray-600">Drift Score</div>
                  <div className="text-2xl font-bold">{latestReport.drift_score.toFixed(3)}</div>
                </div>
                <div>
                  <div className="text-xs text-gray-600">Severity</div>
                  <div className={`text-2xl font-bold text-${severity.color}-600 capitalize`}>
                    {severity.level}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-600">Affected Features</div>
                  <div className="text-2xl font-bold">
                    {Object.keys(latestReport.feature_drifts || {}).length}
                  </div>
                </div>
              </div>
            </div>
            <div className={`px-4 py-2 bg-${severity.color}-100 rounded-lg`}>
              <div className="text-xs text-gray-600 mb-1">Recommended Action</div>
              <div className={`text-sm font-semibold text-${severity.color}-700 capitalize`}>
                {latestReport.recommended_action.replace('_', ' ')}
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Drift History */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="font-semibold mb-4">Drift Score History</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={reports.slice(0, 20).reverse()}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="created_at"
                tickFormatter={(date) => new Date(date).toLocaleDateString()}
                tick={{ fontSize: 11 }}
              />
              <YAxis />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload[0]) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-white p-3 shadow-lg rounded-lg border border-gray-200">
                        <p className="text-xs text-gray-600">
                          {new Date(data.created_at).toLocaleDateString()}
                        </p>
                        <p className="font-medium mt-1">
                          Score: {data.drift_score.toFixed(3)}
                        </p>
                        <p className="text-sm text-gray-600">
                          Action: {data.recommended_action}
                        </p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <ReferenceLine y={0.3} stroke="#f59e0b" strokeDasharray="3 3" label="Warning" />
              <ReferenceLine y={0.5} stroke="#ef4444" strokeDasharray="3 3" label="Critical" />
              <Area
                type="monotone"
                dataKey="drift_score"
                fill="#3b82f6"
                fillOpacity={0.2}
                stroke="#3b82f6"
                strokeWidth={2}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* Feature Drift Details */}
        {latestReport && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="font-semibold mb-4">Feature Drift Breakdown</h3>
            <div className="space-y-3 max-h-[300px] overflow-y-auto">
              {Object.entries(latestReport.feature_drifts || {})
                .sort(([, a], [, b]) => (b as any).drift_score - (a as any).drift_score)
                .slice(0, 10)
                .map(([feature, drift]) => {
                  const driftData = drift as any;
                  const severity = driftSeverity(driftData.drift_score);
                  return (
                    <div key={feature} className="border border-gray-200 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <div className="font-medium text-sm truncate flex-1">{feature}</div>
                        <div className={`px-2 py-1 rounded text-xs font-medium bg-${severity.color}-100 text-${severity.color}-700`}>
                          {severity.level}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                          <div
                            className={`bg-${severity.color}-500 h-2 rounded-full`}
                            style={{ width: `${Math.min(driftData.drift_score * 100, 100)}%` }}
                          />
                        </div>
                        <div className="text-xs font-medium text-gray-600">
                          {driftData.drift_score.toFixed(3)}
                        </div>
                      </div>
                      {driftData.tests && (
                        <div className="mt-2 text-xs text-gray-600">
                          {driftData.tests.ks && (
                            <div>KS p-value: {driftData.tests.ks.p_value.toFixed(4)}</div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
            </div>
          </div>
        )}
      </div>

      {/* Report History */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="font-semibold mb-4">Drift Reports</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-3 text-left font-medium text-gray-700">Date</th>
                <th className="px-4 py-3 text-left font-medium text-gray-700">Type</th>
                <th className="px-4 py-3 text-left font-medium text-gray-700">Score</th>
                <th className="px-4 py-3 text-left font-medium text-gray-700">Status</th>
                <th className="px-4 py-3 text-left font-medium text-gray-700">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {reports.slice(0, 10).map(report => {
                const severity = driftSeverity(report.drift_score);
                return (
                  <tr
                    key={report.id}
                    className="hover:bg-gray-50 cursor-pointer"
                    onClick={() => setSelectedReport(report)}
                  >
                    <td className="px-4 py-3">
                      {new Date(report.created_at).toLocaleString()}
                    </td>
                    <td className="px-4 py-3 capitalize">{report.drift_type.replace('_', ' ')}</td>
                    <td className="px-4 py-3 font-medium">{report.drift_score.toFixed(3)}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 rounded text-xs font-medium bg-${severity.color}-100 text-${severity.color}-700`}>
                        {report.drift_detected ? 'Drift' : 'Normal'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-xs capitalize">
                      {report.recommended_action.replace('_', ' ')}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// TIME SERIES FORECASTING
// ============================================================================

interface TimeSeriesForecastProps {
  historicalData: Array<{ timestamp: string; value: number }>;
  forecast: TimeSeriesForecast[];
  onReforecast: () => Promise<void>;
}

export const TimeSeriesForecast: React.FC<TimeSeriesForecastProps> = ({
  historicalData,
  forecast,
  onReforecast
}) => {
  const [showConfidence, setShowConfidence] = useState(true);
  const [forecasting, setForecasting] = useState(false);

  const combinedData = useMemo(() => {
    return [
      ...historicalData.map(d => ({
        ...d,
        timestamp: new Date(d.timestamp).getTime(),
        type: 'historical'
      })),
      ...forecast.map(f => ({
        timestamp: new Date(f.ds).getTime(),
        value: f.yhat,
        lower: f.yhat_lower,
        upper: f.yhat_upper,
        trend: f.trend,
        type: 'forecast'
      }))
    ].sort((a, b) => a.timestamp - b.timestamp);
  }, [historicalData, forecast]);

  const handleReforecast = async () => {
    setForecasting(true);
    try {
      await onReforecast();
    } finally {
      setForecasting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Time Series Forecast</h2>
          <p className="text-sm text-gray-600 mt-1">
            Predictive trends and seasonality analysis
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setShowConfidence(!showConfidence)}
            className="px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center gap-2"
          >
            {showConfidence ? <Eye size={16} /> : <EyeOff size={16} />}
            Confidence Band
          </button>
          <button
            onClick={handleReforecast}
            disabled={forecasting}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 flex items-center gap-2"
          >
            <RefreshCw size={16} className={forecasting ? 'animate-spin' : ''} />
            Reforecast
          </button>
        </div>
      </div>

      {/* Forecast Chart */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={combinedData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="timestamp"
              type="number"
              domain={['dataMin', 'dataMax']}
              tickFormatter={(time) => new Date(time).toLocaleDateString()}
              tick={{ fontSize: 11 }}
            />
            <YAxis />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload[0]) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-white p-3 shadow-lg rounded-lg border border-gray-200">
                      <p className="text-xs text-gray-600">
                        {new Date(data.timestamp).toLocaleDateString()}
                      </p>
                      <p className="font-medium mt-1">
                        {data.type === 'historical' ? 'Actual' : 'Forecast'}: {data.value.toFixed(2)}
                      </p>
                      {data.type === 'forecast' && showConfidence && (
                        <>
                          <p className="text-sm text-gray-600">
                            Upper: {data.upper.toFixed(2)}
                          </p>
                          <p className="text-sm text-gray-600">
                            Lower: {data.lower.toFixed(2)}
                          </p>
                        </>
                      )}
                    </div>
                  );
                }}
                return null;
              }}
            />
            <Legend />
            
            {/* Confidence band */}
            {showConfidence && (
              <Area
                type="monotone"
                dataKey="upper"
                fill="#93c5fd"
                fillOpacity={0.2}
                stroke="none"
                name="Confidence Band"
              />
            )}
            
            {/* Historical data */}
            <Line
              type="monotone"
              dataKey="value"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              name="Actual / Forecast"
            />
            
            {/* Trend */}
            <Line
              type="monotone"
              dataKey="trend"
              stroke="#10b981"
              strokeWidth={1}
              strokeDasharray="5 5"
              dot={false}
              name="Trend"
            />
            
            {/* Dividing line between historical and forecast */}
            {historicalData.length > 0 && (
              <ReferenceLine
                x={new Date(historicalData[historicalData.length - 1].timestamp).getTime()}
                stroke="#6b7280"
                strokeDasharray="3 3"
                label={{ value: 'Forecast Start', position: 'top' }}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Forecast Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="text-sm text-gray-600 mb-1">Next Period Forecast</div>
          <div className="text-2xl font-bold text-gray-900">
            {forecast[0]?.yhat.toFixed(2) || 'N/A'}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            ±{forecast[0] ? ((forecast[0].yhat_upper - forecast[0].yhat_lower) / 2).toFixed(2) : 'N/A'}
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="text-sm text-gray-600 mb-1">Trend Direction</div>
          <div className="flex items-center gap-2">
            {forecast.length > 1 && forecast[forecast.length - 1].trend > forecast[0].trend ? (
              <>
                <TrendingUp className="text-green-600" size={24} />
                <span className="text-2xl font-bold text-green-600">Increasing</span>
              </>
            ) : (
              <>
                <TrendingDown className="text-red-600" size={24} />
                <span className="text-2xl font-bold text-red-600">Decreasing</span>
              </>
            )}
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="text-sm text-gray-600 mb-1">Forecast Horizon</div>
          <div className="text-2xl font-bold text-gray-900">
            {forecast.length} periods
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {forecast.length > 0 ? new Date(forecast[forecast.length - 1].ds).toLocaleDateString() : 'N/A'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default {
  AnomalyMonitor,
  DriftMonitoring,
  TimeSeriesForecast
};
