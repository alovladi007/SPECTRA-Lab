/**
 * SESSION 14: ML/VM HUB - UI COMPONENTS
 * 
 * Complete React/TypeScript UI for Machine Learning and Virtual Metrology:
 * - Model training interface
 * - Feature engineering studio
 * - Prediction dashboard
 * - Anomaly detection monitor
 * - Drift analysis
 * - Model registry and versioning
 * - Performance analytics
 * 
 * @author Semiconductor Lab Platform Team
 * @date October 2024
 * @version 1.0.0
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  LineChart, Line, ScatterChart, Scatter, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Area, AreaChart, ComposedChart, ReferenceLine, Cell
} from 'recharts';
import {
  AlertCircle, CheckCircle, TrendingUp, TrendingDown,
  Brain, Activity, AlertTriangle, Info, Settings,
  Play, Pause, RefreshCw, Download, Upload, Save,
  Eye, EyeOff, ChevronDown, ChevronRight, Filter,
  Search, Calendar, BarChart3, Zap, Target
} from 'lucide-react';

// ============================================================================
// TYPES AND INTERFACES
// ============================================================================

interface ModelMetrics {
  r2: number;
  rmse: number;
  mae: number;
  mape: number;
}

interface MLModel {
  id: number;
  name: string;
  version: string;
  type: string;
  algorithm: string;
  status: string;
  metrics: {
    train: ModelMetrics;
    test: ModelMetrics;
    cv: {
      r2_mean: number;
      r2_std: number;
    };
  };
  feature_importance: Record<string, number>;
  deployed_at?: string;
  prediction_count: number;
  drift_detected: boolean;
}

interface FeatureImportance {
  name: string;
  importance: number;
  type: string;
}

interface Prediction {
  id: number;
  timestamp: string;
  features: Record<string, number>;
  prediction: number;
  confidence: number;
  uncertainty: number;
  actual_value?: number;
}

interface AnomalyDetection {
  id: number;
  timestamp: string;
  is_anomaly: boolean;
  anomaly_score: number;
  anomaly_type: string;
  features: Record<string, number>;
  feature_contributions: Record<string, number>;
  likely_causes: string[];
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
// MODEL TRAINING DASHBOARD
// ============================================================================

interface ModelTrainingDashboardProps {
  onTrainModel: (config: any) => Promise<void>;
  models: MLModel[];
}

export const ModelTrainingDashboard: React.FC<ModelTrainingDashboardProps> = ({
  onTrainModel,
  models
}) => {
  const [modelType, setModelType] = useState('virtual_metrology');
  const [algorithm, setAlgorithm] = useState('random_forest');
  const [trainingData, setTrainingData] = useState<File | null>(null);
  const [targetColumn, setTargetColumn] = useState('');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);

  const handleTrain = async () => {
    if (!trainingData || !targetColumn) {
      alert('Please select training data and target column');
      return;
    }

    setIsTraining(true);
    setTrainingProgress(0);

    try {
      const config = {
        model_type: modelType,
        algorithm,
        target_col: targetColumn
      };

      // Simulate progress
      const interval = setInterval(() => {
        setTrainingProgress(prev => Math.min(prev + 10, 90));
      }, 500);

      await onTrainModel(config);

      clearInterval(interval);
      setTrainingProgress(100);

      setTimeout(() => {
        setIsTraining(false);
        setTrainingProgress(0);
      }, 1000);

    } catch (error) {
      console.error('Training error:', error);
      setIsTraining(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Model Training</h2>
          <p className="text-sm text-gray-600 mt-1">
            Train new ML models for virtual metrology and anomaly detection
          </p>
        </div>
        <div className="flex gap-2">
          <button className="px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center gap-2">
            <Upload size={16} />
            Import Config
          </button>
          <button className="px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center gap-2">
            <Save size={16} />
            Save Template
          </button>
        </div>
      </div>

      {/* Training Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Configuration Panel */}
        <div className="lg:col-span-2 bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="p-6 border-b border-gray-200">
            <h3 className="font-semibold text-lg">Training Configuration</h3>
          </div>
          <div className="p-6 space-y-6">
            {/* Model Type */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Model Type
              </label>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { value: 'virtual_metrology', label: 'Virtual Metrology', icon: Target },
                  { value: 'anomaly_detection', label: 'Anomaly Detection', icon: AlertTriangle },
                  { value: 'drift_detection', label: 'Drift Detection', icon: Activity },
                  { value: 'time_series', label: 'Time Series', icon: TrendingUp }
                ].map(type => (
                  <button
                    key={type.value}
                    onClick={() => setModelType(type.value)}
                    className={`p-4 border-2 rounded-lg transition-all ${
                      modelType === type.value
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <type.icon className={`mb-2 ${
                      modelType === type.value ? 'text-blue-600' : 'text-gray-400'
                    }`} size={24} />
                    <div className="text-sm font-medium">{type.label}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Algorithm Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Algorithm
              </label>
              <select
                value={algorithm}
                onChange={(e) => setAlgorithm(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <optgroup label="Regression">
                  <option value="random_forest">Random Forest</option>
                  <option value="gradient_boosting">Gradient Boosting</option>
                  <option value="lightgbm">LightGBM</option>
                </optgroup>
                <optgroup label="Anomaly Detection">
                  <option value="isolation_forest">Isolation Forest</option>
                  <option value="elliptic_envelope">Elliptic Envelope</option>
                  <option value="pca_anomaly">PCA-based</option>
                </optgroup>
                <optgroup label="Time Series">
                  <option value="prophet">Prophet</option>
                </optgroup>
              </select>
            </div>

            {/* Data Upload */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Training Data
              </label>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-gray-400 transition-colors cursor-pointer">
                <input
                  type="file"
                  accept=".csv,.json"
                  onChange={(e) => setTrainingData(e.target.files?.[0] || null)}
                  className="hidden"
                  id="training-data-upload"
                />
                <label htmlFor="training-data-upload" className="cursor-pointer">
                  <Upload className="mx-auto mb-2 text-gray-400" size={32} />
                  <p className="text-sm font-medium text-gray-700">
                    {trainingData ? trainingData.name : 'Click to upload or drag and drop'}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">CSV or JSON (max 100MB)</p>
                </label>
              </div>
            </div>

            {/* Target Column */}
            {modelType === 'virtual_metrology' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Target Column
                </label>
                <input
                  type="text"
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  placeholder="e.g., thickness, resistance, mobility"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            )}

            {/* Advanced Options */}
            <details className="border border-gray-200 rounded-lg">
              <summary className="px-4 py-3 cursor-pointer font-medium text-sm text-gray-700 hover:bg-gray-50">
                Advanced Options
              </summary>
              <div className="p-4 space-y-4 border-t border-gray-200">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">
                      Test Split
                    </label>
                    <input
                      type="number"
                      defaultValue={20}
                      min={10}
                      max={40}
                      className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">
                      CV Folds
                    </label>
                    <input
                      type="number"
                      defaultValue={5}
                      min={2}
                      max={10}
                      className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">
                      Random State
                    </label>
                    <input
                      type="number"
                      defaultValue={42}
                      className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">
                      N Estimators
                    </label>
                    <input
                      type="number"
                      defaultValue={100}
                      min={10}
                      max={1000}
                      className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
                    />
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="feature-engineering"
                    defaultChecked
                    className="rounded border-gray-300"
                  />
                  <label htmlFor="feature-engineering" className="text-sm text-gray-700">
                    Enable automatic feature engineering
                  </label>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="export-onnx"
                    defaultChecked
                    className="rounded border-gray-300"
                  />
                  <label htmlFor="export-onnx" className="text-sm text-gray-700">
                    Export to ONNX format
                  </label>
                </div>
              </div>
            </details>

            {/* Training Progress */}
            {isTraining && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-blue-900">Training in progress...</span>
                  <span className="text-sm font-medium text-blue-900">{trainingProgress}%</span>
                </div>
                <div className="w-full bg-blue-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${trainingProgress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-3 pt-4 border-t border-gray-200">
              <button
                onClick={handleTrain}
                disabled={isTraining || !trainingData || !targetColumn}
                className="flex-1 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed font-medium flex items-center justify-center gap-2"
              >
                <Play size={18} />
                {isTraining ? 'Training...' : 'Start Training'}
              </button>
              <button className="px-4 py-3 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 font-medium">
                Validate Only
              </button>
            </div>
          </div>
        </div>

        {/* Recent Models */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="p-4 border-b border-gray-200">
            <h3 className="font-semibold">Recent Models</h3>
          </div>
          <div className="p-4 space-y-3 max-h-[600px] overflow-y-auto">
            {models.slice(0, 10).map(model => (
              <div key={model.id} className="p-3 border border-gray-200 rounded-lg hover:border-blue-300 transition-colors cursor-pointer">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <div className="font-medium text-sm">{model.name}</div>
                    <div className="text-xs text-gray-500">v{model.version}</div>
                  </div>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    model.status === 'deployed' ? 'bg-green-100 text-green-700' :
                    model.status === 'ready' ? 'bg-blue-100 text-blue-700' :
                    'bg-gray-100 text-gray-700'
                  }`}>
                    {model.status}
                  </span>
                </div>
                <div className="space-y-1 text-xs text-gray-600">
                  <div className="flex justify-between">
                    <span>R²:</span>
                    <span className="font-medium">{model.metrics?.test?.r2?.toFixed(3) || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>RMSE:</span>
                    <span className="font-medium">{model.metrics?.test?.rmse?.toFixed(2) || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Predictions:</span>
                    <span className="font-medium">{model.prediction_count}</span>
                  </div>
                </div>
                {model.drift_detected && (
                  <div className="mt-2 flex items-center gap-1 text-xs text-amber-600">
                    <AlertTriangle size={12} />
                    <span>Drift detected</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// FEATURE IMPORTANCE VISUALIZATION
// ============================================================================

interface FeatureImportanceChartProps {
  features: FeatureImportance[];
  topN?: number;
}

export const FeatureImportanceChart: React.FC<FeatureImportanceChartProps> = ({
  features,
  topN = 20
}) => {
  const sortedFeatures = useMemo(() => {
    return [...features]
      .sort((a, b) => b.importance - a.importance)
      .slice(0, topN);
  }, [features, topN]);

  const featureTypeColors: Record<string, string> = {
    raw: '#3b82f6',
    rolling_stat: '#10b981',
    difference: '#f59e0b',
    ratio: '#ef4444',
    distribution: '#8b5cf6',
    outlier_score: '#ec4899',
    temporal: '#06b6d4',
    interaction: '#84cc16'
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="mb-6">
        <h3 className="text-lg font-semibold">Feature Importance</h3>
        <p className="text-sm text-gray-600 mt-1">
          Top {topN} most important features for model predictions
        </p>
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={sortedFeatures} layout="vertical" margin={{ left: 150 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis type="number" />
          <YAxis dataKey="name" type="category" tick={{ fontSize: 12 }} />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload[0]) {
                const data = payload[0].payload;
                return (
                  <div className="bg-white p-3 shadow-lg rounded-lg border border-gray-200">
                    <p className="font-medium text-sm">{data.name}</p>
                    <p className="text-xs text-gray-600 mt-1">
                      Importance: {(data.importance * 100).toFixed(2)}%
                    </p>
                    <p className="text-xs text-gray-600">
                      Type: {data.type}
                    </p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
            {sortedFeatures.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={featureTypeColors[entry.type] || '#3b82f6'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-3">
        {Object.entries(featureTypeColors).map(([type, color]) => (
          <div key={type} className="flex items-center gap-2">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: color }} />
            <span className="text-xs text-gray-600 capitalize">
              {type.replace('_', ' ')}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

// ============================================================================
// PREDICTION DASHBOARD
// ============================================================================

interface PredictionDashboardProps {
  model: MLModel;
  predictions: Prediction[];
  onPredict: (features: Record<string, number>) => Promise<void>;
}

export const PredictionDashboard: React.FC<PredictionDashboardProps> = ({
  model,
  predictions,
  onPredict
}) => {
  const [features, setFeatures] = useState<Record<string, number>>({});
  const [predicting, setPredicting] = useState(false);
  const [lastPrediction, setLastPrediction] = useState<Prediction | null>(null);

  const handlePredict = async () => {
    setPredicting(true);
    try {
      await onPredict(features);
      // Get last prediction
      if (predictions.length > 0) {
        setLastPrediction(predictions[predictions.length - 1]);
      }
    } finally {
      setPredicting(false);
    }
  };

  // Calculate accuracy metrics
  const accuracyMetrics = useMemo(() => {
    const withActuals = predictions.filter(p => p.actual_value !== undefined);
    if (withActuals.length === 0) return null;

    const errors = withActuals.map(p => Math.abs(p.prediction - p.actual_value!));
    const relErrors = withActuals.map(p => 
      Math.abs((p.prediction - p.actual_value!) / p.actual_value!) * 100
    );

    return {
      mae: errors.reduce((a, b) => a + b, 0) / errors.length,
      mape: relErrors.reduce((a, b) => a + b, 0) / relErrors.length,
      count: withActuals.length
    };
  }, [predictions]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Virtual Metrology Predictions</h2>
        <p className="text-sm text-gray-600 mt-1">
          Model: {model.name} v{model.version}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Prediction Input */}
        <div className="lg:col-span-1 bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="p-4 border-b border-gray-200">
            <h3 className="font-semibold">Input Features</h3>
          </div>
          <div className="p-4 space-y-4">
            {Object.entries(model.feature_importance || {})
              .slice(0, 10)
              .map(([feature, _]) => (
                <div key={feature}>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    {feature}
                  </label>
                  <input
                    type="number"
                    step="any"
                    value={features[feature] || ''}
                    onChange={(e) => setFeatures({
                      ...features,
                      [feature]: parseFloat(e.target.value)
                    })}
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    placeholder="Enter value"
                  />
                </div>
              ))}
            <button
              onClick={handlePredict}
              disabled={predicting || Object.keys(features).length === 0}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed font-medium flex items-center justify-center gap-2"
            >
              <Zap size={16} />
              {predicting ? 'Predicting...' : 'Predict'}
            </button>
          </div>
        </div>

        {/* Results */}
        <div className="lg:col-span-2 space-y-6">
          {/* Latest Prediction */}
          {lastPrediction && (
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200 p-6">
              <h3 className="font-semibold text-lg mb-4">Latest Prediction</h3>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <div className="text-sm text-gray-600 mb-1">Predicted Value</div>
                  <div className="text-3xl font-bold text-blue-600">
                    {lastPrediction.prediction.toFixed(3)}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">Confidence</div>
                  <div className="text-3xl font-bold text-green-600">
                    {(lastPrediction.confidence * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">Uncertainty</div>
                  <div className="text-3xl font-bold text-amber-600">
                    ±{lastPrediction.uncertainty.toFixed(3)}
                  </div>
                </div>
              </div>
              {lastPrediction.actual_value !== undefined && (
                <div className="mt-4 pt-4 border-t border-blue-200">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-700">Actual Value:</span>
                    <span className="font-semibold">{lastPrediction.actual_value.toFixed(3)}</span>
                  </div>
                  <div className="flex items-center justify-between mt-2">
                    <span className="text-sm text-gray-700">Error:</span>
                    <span className={`font-semibold ${
                      Math.abs(lastPrediction.prediction - lastPrediction.actual_value) < lastPrediction.uncertainty
                        ? 'text-green-600'
                        : 'text-red-600'
                    }`}>
                      {((lastPrediction.prediction - lastPrediction.actual_value) / lastPrediction.actual_value * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Accuracy Metrics */}
          {accuracyMetrics && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="font-semibold mb-4">Prediction Accuracy</h3>
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-gray-900">
                    {accuracyMetrics.mae.toFixed(3)}
                  </div>
                  <div className="text-sm text-gray-600 mt-1">MAE</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-gray-900">
                    {accuracyMetrics.mape.toFixed(2)}%
                  </div>
                  <div className="text-sm text-gray-600 mt-1">MAPE</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-gray-900">
                    {accuracyMetrics.count}
                  </div>
                  <div className="text-sm text-gray-600 mt-1">Samples</div>
                </div>
              </div>
            </div>
          )}

          {/* Prediction History */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="font-semibold mb-4">Prediction History</h3>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={predictions.slice(-50)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={(time) => new Date(time).toLocaleTimeString()}
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
                            {new Date(data.timestamp).toLocaleString()}
                          </p>
                          <p className="font-medium mt-1">
                            Predicted: {data.prediction.toFixed(3)}
                          </p>
                          {data.actual_value !== undefined && (
                            <p className="text-sm text-gray-600">
                              Actual: {data.actual_value.toFixed(3)}
                            </p>
                          )}
                          <p className="text-sm text-gray-600">
                            Confidence: {(data.confidence * 100).toFixed(1)}%
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="prediction"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                  name="Predicted"
                />
                {predictions.some(p => p.actual_value !== undefined) && (
                  <Line
                    type="monotone"
                    dataKey="actual_value"
                    stroke="#10b981"
                    strokeWidth={2}
                    dot={false}
                    name="Actual"
                  />
                )}
                <Area
                  type="monotone"
                  dataKey="confidence"
                  stroke="none"
                  fill="#93c5fd"
                  fillOpacity={0.2}
                  name="Confidence Band"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

// Continuing in next message due to length...
