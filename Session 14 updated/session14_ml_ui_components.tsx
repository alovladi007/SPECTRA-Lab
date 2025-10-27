/**
 * Session 14: ML & Virtual Metrology - Complete UI Components
 * ============================================================
 * 
 * Comprehensive React components for machine learning and virtual metrology:
 * 
 * Part 1: Model Training & Prediction
 * - ModelTrainingDashboard: Train ML models with hyperparameter tuning
 * - FeatureImportanceChart: Visualize feature contributions
 * - PredictionDashboard: Real-time predictions and what-if analysis
 * 
 * Part 2: Monitoring & Forecasting
 * - AnomalyMonitor: Real-time anomaly detection and alerts
 * - DriftMonitoring: Data and model drift tracking
 * - TimeSeriesForecast: Predictive maintenance forecasting
 * 
 * Author: Semiconductor Lab Platform Team
 * Date: October 2025
 * Version: 1.0.0
 */

'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Slider } from '@/components/ui/slider';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Brain,
  TrendingUp,
  Activity,
  AlertTriangle,
  CheckCircle,
  Play,
  Save,
  Download,
  RefreshCw,
  BarChart3,
  LineChart,
  Target,
  Zap,
  Clock,
  Database,
  Settings,
  Eye,
  Bell,
} from 'lucide-react';
import {
  LineChart as RechartsLineChart,
  Line,
  BarChart as RechartsBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ReferenceLine,
  Area,
  AreaChart,
  ComposedChart,
} from 'recharts';

// ============================================================================
// Type Definitions
// ============================================================================

interface ModelMetrics {
  r2: number;
  rmse: number;
  mae: number;
  mape: number;
  cv_mean: number;
  cv_std: number;
  training_time: number;
  inference_time: number;
  n_samples: number;
  n_features: number;
}

interface FeatureImportance {
  feature: string;
  importance: number;
  rank: number;
}

interface Prediction {
  id: number;
  timestamp: string;
  actual?: number;
  predicted: number;
  confidence: number;
  features: Record<string, number>;
}

interface Anomaly {
  id: number;
  timestamp: string;
  score: number;
  is_anomaly: boolean;
  confidence: number;
  contributing_features: Array<{ feature: string; score: number }>;
  severity: 'low' | 'medium' | 'high';
}

interface DriftMetric {
  timestamp: string;
  score: number;
  threshold: number;
  method: string;
  severity: 'low' | 'medium' | 'high';
  affected_features: string[];
}

interface ForecastPoint {
  timestamp: string;
  value: number;
  lower_bound: number;
  upper_bound: number;
  is_actual?: boolean;
}

// ============================================================================
// Part 1: Model Training Dashboard
// ============================================================================

export const ModelTrainingDashboard: React.FC = () => {
  const [modelType, setModelType] = useState('random_forest');
  const [targetMetric, setTargetMetric] = useState('thickness');
  const [trainingStatus, setTrainingStatus] = useState<'idle' | 'training' | 'complete' | 'error'>('idle');
  const [progress, setProgress] = useState(0);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [hyperparameters, setHyperparameters] = useState({
    n_estimators: 100,
    max_depth: 10,
    learning_rate: 0.1,
    test_size: 0.2,
    cv_folds: 5,
  });

  const [trainingHistory, setTrainingHistory] = useState<Array<{
    iteration: number;
    train_score: number;
    val_score: number;
    time: number;
  }>>([]);

  // Simulate model training
  const handleTrain = async () => {
    setTrainingStatus('training');
    setProgress(0);

    // Simulate training iterations
    const history: typeof trainingHistory = [];
    for (let i = 1; i <= 100; i++) {
      await new Promise(resolve => setTimeout(resolve, 50));
      setProgress(i);
      
      // Generate synthetic training history
      if (i % 10 === 0) {
        history.push({
          iteration: i,
          train_score: 0.6 + (i / 100) * 0.3 + Math.random() * 0.05,
          val_score: 0.55 + (i / 100) * 0.25 + Math.random() * 0.05,
          time: i * 0.5,
        });
      }
    }

    setTrainingHistory(history);

    // Generate final metrics
    const finalMetrics: ModelMetrics = {
      r2: 0.88 + Math.random() * 0.08,
      rmse: 2.5 + Math.random() * 0.5,
      mae: 1.8 + Math.random() * 0.4,
      mape: 3.2 + Math.random() * 0.8,
      cv_mean: 0.86 + Math.random() * 0.06,
      cv_std: 0.02 + Math.random() * 0.01,
      training_time: 15.2 + Math.random() * 5,
      inference_time: 0.005 + Math.random() * 0.002,
      n_samples: 1000,
      n_features: 20,
    };

    setMetrics(finalMetrics);
    setTrainingStatus('complete');
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Brain className="w-5 h-5" />
                Model Training Dashboard
              </CardTitle>
              <CardDescription>
                Train virtual metrology models with automated hyperparameter optimization
              </CardDescription>
            </div>
            <Badge variant={trainingStatus === 'complete' ? 'default' : 'secondary'}>
              {trainingStatus === 'idle' && 'Ready'}
              {trainingStatus === 'training' && 'Training...'}
              {trainingStatus === 'complete' && 'Complete'}
              {trainingStatus === 'error' && 'Error'}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Configuration Panel */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Model Type</Label>
                  <Select value={modelType} onValueChange={setModelType}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="random_forest">Random Forest</SelectItem>
                      <SelectItem value="gradient_boosting">Gradient Boosting</SelectItem>
                      <SelectItem value="lightgbm">LightGBM</SelectItem>
                      <SelectItem value="xgboost">XGBoost</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Target Metric</Label>
                  <Select value={targetMetric} onValueChange={setTargetMetric}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="thickness">Thickness (nm)</SelectItem>
                      <SelectItem value="resistivity">Resistivity (Ω·cm)</SelectItem>
                      <SelectItem value="roughness">Roughness (nm)</SelectItem>
                      <SelectItem value="uniformity">Uniformity (%)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Number of Estimators: {hyperparameters.n_estimators}</Label>
                  <Slider
                    value={[hyperparameters.n_estimators]}
                    onValueChange={([value]) => setHyperparameters(prev => ({ ...prev, n_estimators: value }))}
                    min={50}
                    max={500}
                    step={50}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Max Depth: {hyperparameters.max_depth}</Label>
                  <Slider
                    value={[hyperparameters.max_depth]}
                    onValueChange={([value]) => setHyperparameters(prev => ({ ...prev, max_depth: value }))}
                    min={3}
                    max={20}
                    step={1}
                  />
                </div>

                <div className="space-y-2">
                  <Label>CV Folds: {hyperparameters.cv_folds}</Label>
                  <Slider
                    value={[hyperparameters.cv_folds]}
                    onValueChange={([value]) => setHyperparameters(prev => ({ ...prev, cv_folds: value }))}
                    min={3}
                    max={10}
                    step={1}
                  />
                </div>

                <Button 
                  onClick={handleTrain}
                  disabled={trainingStatus === 'training'}
                  className="w-full"
                >
                  {trainingStatus === 'training' ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                      Training...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Start Training
                    </>
                  )}
                </Button>

                {trainingStatus === 'training' && (
                  <div className="space-y-2">
                    <Progress value={progress} />
                    <p className="text-sm text-muted-foreground text-center">
                      {progress}% Complete
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Training Progress */}
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle className="text-lg">Training Progress</CardTitle>
              </CardHeader>
              <CardContent>
                {trainingHistory.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <RechartsLineChart data={trainingHistory}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="iteration" label={{ value: 'Iteration', position: 'insideBottom', offset: -5 }} />
                      <YAxis label={{ value: 'R² Score', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="train_score" 
                        stroke="#3b82f6" 
                        name="Training Score"
                        strokeWidth={2}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="val_score" 
                        stroke="#10b981" 
                        name="Validation Score"
                        strokeWidth={2}
                      />
                    </RechartsLineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                    <div className="text-center">
                      <Brain className="w-12 h-12 mx-auto mb-4 opacity-50" />
                      <p>Start training to see progress</p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Results */}
          {metrics && trainingStatus === 'complete' && (
            <Card className="mt-6">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  Model Performance Metrics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                  <div className="text-center p-4 bg-blue-50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">
                      {metrics.r2.toFixed(4)}
                    </div>
                    <div className="text-sm text-muted-foreground">R² Score</div>
                  </div>
                  <div className="text-center p-4 bg-green-50 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">
                      {metrics.rmse.toFixed(2)}
                    </div>
                    <div className="text-sm text-muted-foreground">RMSE</div>
                  </div>
                  <div className="text-center p-4 bg-purple-50 rounded-lg">
                    <div className="text-2xl font-bold text-purple-600">
                      {metrics.mae.toFixed(2)}
                    </div>
                    <div className="text-sm text-muted-foreground">MAE</div>
                  </div>
                  <div className="text-center p-4 bg-amber-50 rounded-lg">
                    <div className="text-2xl font-bold text-amber-600">
                      {metrics.mape.toFixed(1)}%
                    </div>
                    <div className="text-sm text-muted-foreground">MAPE</div>
                  </div>
                  <div className="text-center p-4 bg-indigo-50 rounded-lg">
                    <div className="text-2xl font-bold text-indigo-600">
                      {metrics.training_time.toFixed(1)}s
                    </div>
                    <div className="text-sm text-muted-foreground">Training Time</div>
                  </div>
                </div>

                <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Alert>
                    <TrendingUp className="w-4 h-4" />
                    <AlertTitle>Cross-Validation</AlertTitle>
                    <AlertDescription>
                      Mean: {metrics.cv_mean.toFixed(4)} ± {metrics.cv_std.toFixed(4)}
                      <br />
                      {hyperparameters.cv_folds}-fold validation with consistent performance
                    </AlertDescription>
                  </Alert>

                  <Alert>
                    <Zap className="w-4 h-4" />
                    <AlertTitle>Inference Performance</AlertTitle>
                    <AlertDescription>
                      {(metrics.inference_time * 1000).toFixed(2)} ms per prediction
                      <br />
                      ~{Math.round(1000 / metrics.inference_time)} predictions/second
                    </AlertDescription>
                  </Alert>
                </div>

                <div className="mt-4 flex gap-2">
                  <Button variant="outline" className="flex-1">
                    <Save className="w-4 h-4 mr-2" />
                    Save Model
                  </Button>
                  <Button variant="outline" className="flex-1">
                    <Download className="w-4 h-4 mr-2" />
                    Export ONNX
                  </Button>
                  <Button variant="outline" className="flex-1">
                    <Target className="w-4 h-4 mr-2" />
                    Deploy to Production
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

// ============================================================================
// Part 2: Feature Importance Chart
// ============================================================================

export const FeatureImportanceChart: React.FC = () => {
  const [importanceData, setImportanceData] = useState<FeatureImportance[]>([]);
  const [sortBy, setSortBy] = useState<'importance' | 'alphabetical'>('importance');
  const [topN, setTopN] = useState(15);

  useEffect(() => {
    // Generate synthetic feature importance data
    const features = [
      'temperature_mean', 'pressure_mean', 'flow_rate_mean', 'power_mean',
      'temperature_std', 'pressure_std', 'flow_rate_std', 'power_std',
      'recipe_time', 'recipe_gas_mix', 'chamber_pressure', 'substrate_temp',
      'deposition_rate', 'plasma_density', 'bias_voltage', 'rf_power',
      'process_duration', 'base_pressure', 'chamber_volume', 'wafer_temp'
    ];

    const data: FeatureImportance[] = features.map((feature, index) => ({
      feature,
      importance: Math.max(0, 1 - (index / features.length) + (Math.random() * 0.2 - 0.1)),
      rank: index + 1,
    }));

    setImportanceData(data);
  }, []);

  const sortedData = [...importanceData]
    .sort((a, b) => {
      if (sortBy === 'importance') {
        return b.importance - a.importance;
      }
      return a.feature.localeCompare(b.feature);
    })
    .slice(0, topN);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Feature Importance Analysis
            </CardTitle>
            <CardDescription>
              Understand which features contribute most to predictions
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Select value={sortBy} onValueChange={(value: any) => setSortBy(value)}>
              <SelectTrigger className="w-[150px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="importance">By Importance</SelectItem>
                <SelectItem value="alphabetical">Alphabetical</SelectItem>
              </SelectContent>
            </Select>
            <Select value={topN.toString()} onValueChange={(value) => setTopN(parseInt(value))}>
              <SelectTrigger className="w-[120px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="10">Top 10</SelectItem>
                <SelectItem value="15">Top 15</SelectItem>
                <SelectItem value="20">Top 20</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          <ResponsiveContainer width="100%" height={400}>
            <RechartsBarChart data={sortedData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 1]} />
              <YAxis dataKey="feature" type="category" width={150} />
              <Tooltip 
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-white p-3 border rounded-lg shadow-lg">
                        <p className="font-semibold">{data.feature}</p>
                        <p className="text-sm">Importance: {(data.importance * 100).toFixed(2)}%</p>
                        <p className="text-sm text-muted-foreground">Rank: #{data.rank}</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="importance" fill="#3b82f6" />
            </RechartsBarChart>
          </ResponsiveContainer>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Alert>
              <TrendingUp className="w-4 h-4" />
              <AlertTitle>Top Feature</AlertTitle>
              <AlertDescription>
                {sortedData[0]?.feature}
                <br />
                {(sortedData[0]?.importance * 100).toFixed(2)}% contribution
              </AlertDescription>
            </Alert>

            <Alert>
              <Activity className="w-4 h-4" />
              <AlertTitle>Feature Count</AlertTitle>
              <AlertDescription>
                {importanceData.length} total features
                <br />
                Showing top {topN}
              </AlertDescription>
            </Alert>

            <Alert>
              <CheckCircle className="w-4 h-4" />
              <AlertTitle>Cumulative Importance</AlertTitle>
              <AlertDescription>
                Top 5 features explain
                <br />
                {(sortedData.slice(0, 5).reduce((sum, f) => sum + f.importance, 0) * 100).toFixed(1)}% of variance
              </AlertDescription>
            </Alert>
          </div>

          <div>
            <h3 className="font-semibold mb-3">Feature Details</h3>
            <div className="space-y-2 max-h-[300px] overflow-y-auto">
              {sortedData.map((feature, index) => (
                <div
                  key={feature.feature}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <Badge variant="outline">#{feature.rank}</Badge>
                    <span className="font-medium">{feature.feature}</span>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="w-32 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: `${feature.importance * 100}%` }}
                      />
                    </div>
                    <span className="text-sm font-semibold w-16 text-right">
                      {(feature.importance * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// ============================================================================
// Part 3: Prediction Dashboard
// ============================================================================

export const PredictionDashboard: React.FC = () => {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [selectedFeatures, setSelectedFeatures] = useState<Record<string, number>>({
    temperature_mean: 350,
    pressure_mean: 5.0,
    power_mean: 500,
    flow_rate_mean: 100,
  });
  const [prediction, setPrediction] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number>(0);

  useEffect(() => {
    // Generate synthetic historical predictions
    const data: Prediction[] = Array.from({ length: 50 }, (_, i) => {
      const actual = 100 + Math.random() * 50;
      const predicted = actual + (Math.random() - 0.5) * 10;
      return {
        id: i + 1,
        timestamp: new Date(Date.now() - (49 - i) * 3600000).toISOString(),
        actual,
        predicted,
        confidence: 0.85 + Math.random() * 0.15,
        features: {
          temperature_mean: 300 + Math.random() * 100,
          pressure_mean: 3 + Math.random() * 4,
          power_mean: 400 + Math.random() * 200,
          flow_rate_mean: 80 + Math.random() * 40,
        },
      };
    });
    setPredictions(data);
  }, []);

  const handlePredict = () => {
    // Simulate prediction based on features
    const baseValue = 100;
    const tempEffect = (selectedFeatures.temperature_mean - 350) * 0.2;
    const pressureEffect = (selectedFeatures.pressure_mean - 5.0) * 5;
    const powerEffect = (selectedFeatures.power_mean - 500) * 0.05;
    
    const predicted = baseValue + tempEffect + pressureEffect + powerEffect + (Math.random() - 0.5) * 5;
    setPrediction(predicted);
    setConfidence(0.85 + Math.random() * 0.15);
  };

  // Calculate prediction error
  const predictionErrors = predictions.map(p => ({
    timestamp: p.timestamp,
    error: p.actual! - p.predicted,
    abs_error: Math.abs(p.actual! - p.predicted),
  }));

  const mae = predictionErrors.reduce((sum, p) => sum + p.abs_error, 0) / predictionErrors.length;
  const rmse = Math.sqrt(
    predictionErrors.reduce((sum, p) => sum + p.error ** 2, 0) / predictionErrors.length
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Target className="w-5 h-5" />
          Real-time Prediction Dashboard
        </CardTitle>
        <CardDescription>
          Make predictions and analyze model performance
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Prediction Input */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Input Features</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {Object.entries(selectedFeatures).map(([feature, value]) => (
                <div key={feature} className="space-y-2">
                  <Label>
                    {feature.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                    : {value.toFixed(1)}
                  </Label>
                  <Slider
                    value={[value]}
                    onValueChange={([newValue]) =>
                      setSelectedFeatures(prev => ({ ...prev, [feature]: newValue }))
                    }
                    min={
                      feature.includes('temp') ? 200 :
                      feature.includes('pressure') ? 1 :
                      feature.includes('power') ? 200 : 50
                    }
                    max={
                      feature.includes('temp') ? 500 :
                      feature.includes('pressure') ? 10 :
                      feature.includes('power') ? 800 : 150
                    }
                    step={0.1}
                  />
                </div>
              ))}

              <Button onClick={handlePredict} className="w-full">
                <Zap className="w-4 h-4 mr-2" />
                Make Prediction
              </Button>

              {prediction !== null && (
                <Alert>
                  <CheckCircle className="w-4 h-4" />
                  <AlertTitle>Prediction Result</AlertTitle>
                  <AlertDescription>
                    <div className="text-2xl font-bold text-blue-600 my-2">
                      {prediction.toFixed(2)} nm
                    </div>
                    <div className="text-sm">
                      Confidence: {(confidence * 100).toFixed(1)}%
                    </div>
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Historical Performance */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="text-lg">Predicted vs Actual</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    type="number" 
                    dataKey="actual" 
                    name="Actual"
                    label={{ value: 'Actual (nm)', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    type="number" 
                    dataKey="predicted" 
                    name="Predicted"
                    label={{ value: 'Predicted (nm)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <ReferenceLine stroke="#666" strokeDasharray="3 3" segment={[{ x: 80, y: 80 }, { x: 160, y: 160 }]} />
                  <Scatter data={predictions} fill="#3b82f6" />
                </ScatterChart>
              </ResponsiveContainer>

              <div className="grid grid-cols-2 gap-4 mt-4">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">
                    {mae.toFixed(2)}
                  </div>
                  <div className="text-sm text-muted-foreground">MAE (nm)</div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    {rmse.toFixed(2)}
                  </div>
                  <div className="text-sm text-muted-foreground">RMSE (nm)</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Prediction Error Over Time */}
        <Card className="mt-6">
          <CardHeader>
            <CardTitle className="text-lg">Prediction Error Over Time</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <RechartsLineChart data={predictionErrors}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" hide />
                <YAxis label={{ value: 'Error (nm)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
                <Line type="monotone" dataKey="error" stroke="#ef4444" dot={false} />
              </RechartsLineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </CardContent>
    </Card>
  );
};

// ============================================================================
// Part 4: Anomaly Monitor
// ============================================================================

export const AnomalyMonitor: React.FC = () => {
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [detectionMethod, setDetectionMethod] = useState('isolation_forest');
  const [threshold, setThreshold] = useState(0.1);
  const [isMonitoring, setIsMonitoring] = useState(false);

  useEffect(() => {
    // Generate synthetic anomaly data
    const data: Anomaly[] = Array.from({ length: 100 }, (_, i) => {
      const isAnomaly = Math.random() < 0.1;
      const score = isAnomaly ? 0.7 + Math.random() * 0.3 : Math.random() * 0.3;
      
      return {
        id: i + 1,
        timestamp: new Date(Date.now() - (99 - i) * 60000).toISOString(),
        score,
        is_anomaly: isAnomaly,
        confidence: 0.7 + Math.random() * 0.3,
        contributing_features: [
          { feature: 'temperature_mean', score: Math.random() },
          { feature: 'pressure_std', score: Math.random() },
          { feature: 'power_mean', score: Math.random() },
        ].sort((a, b) => b.score - a.score),
        severity: score > 0.8 ? 'high' : score > 0.5 ? 'medium' : 'low',
      };
    });
    setAnomalies(data);
  }, [detectionMethod, threshold]);

  const recentAnomalies = anomalies.filter(a => a.is_anomaly).slice(-10);
  const anomalyRate = (anomalies.filter(a => a.is_anomaly).length / anomalies.length) * 100;

  const severityCounts = {
    high: recentAnomalies.filter(a => a.severity === 'high').length,
    medium: recentAnomalies.filter(a => a.severity === 'medium').length,
    low: recentAnomalies.filter(a => a.severity === 'low').length,
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Real-time Anomaly Monitor
            </CardTitle>
            <CardDescription>
              Continuous monitoring for process anomalies and equipment faults
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={isMonitoring ? 'default' : 'secondary'}>
              {isMonitoring ? 'Monitoring' : 'Paused'}
            </Badge>
            <Button
              size="sm"
              variant={isMonitoring ? 'destructive' : 'default'}
              onClick={() => setIsMonitoring(!isMonitoring)}
            >
              {isMonitoring ? 'Stop' : 'Start'} Monitor
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Configuration */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Detection Method</Label>
              <Select value={detectionMethod} onValueChange={setDetectionMethod}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="isolation_forest">Isolation Forest</SelectItem>
                  <SelectItem value="local_outlier_factor">Local Outlier Factor</SelectItem>
                  <SelectItem value="autoencoder">Autoencoder</SelectItem>
                  <SelectItem value="statistical">Statistical (3-sigma)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Sensitivity Threshold: {threshold.toFixed(2)}</Label>
              <Slider
                value={[threshold]}
                onValueChange={([value]) => setThreshold(value)}
                min={0.01}
                max={0.5}
                step={0.01}
              />
            </div>
          </div>

          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-red-600">
                    {anomalies.filter(a => a.is_anomaly).length}
                  </div>
                  <div className="text-sm text-muted-foreground">Total Anomalies</div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-orange-600">
                    {anomalyRate.toFixed(1)}%
                  </div>
                  <div className="text-sm text-muted-foreground">Anomaly Rate</div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-yellow-600">
                    {severityCounts.high}
                  </div>
                  <div className="text-sm text-muted-foreground">High Severity</div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-green-600">
                    {anomalies.length - anomalies.filter(a => a.is_anomaly).length}
                  </div>
                  <div className="text-sm text-muted-foreground">Normal Samples</div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Anomaly Score Timeline */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Anomaly Score Timeline</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={anomalies.slice(-50)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" hide />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <ReferenceLine y={threshold} stroke="#ef4444" strokeDasharray="3 3" label="Threshold" />
                  <Area 
                    type="monotone" 
                    dataKey="score" 
                    stroke="#3b82f6" 
                    fill="#3b82f6" 
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Recent Anomalies */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Bell className="w-5 h-5" />
                Recent Anomalies ({recentAnomalies.length})
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-[400px] overflow-y-auto">
                {recentAnomalies.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <CheckCircle className="w-12 h-12 mx-auto mb-4 text-green-500" />
                    <p>No anomalies detected</p>
                  </div>
                ) : (
                  recentAnomalies.map((anomaly) => (
                    <Alert
                      key={anomaly.id}
                      variant={anomaly.severity === 'high' ? 'destructive' : 'default'}
                    >
                      <AlertTriangle className="w-4 h-4" />
                      <AlertTitle className="flex items-center justify-between">
                        <span>Anomaly #{anomaly.id}</span>
                        <Badge
                          variant={
                            anomaly.severity === 'high' ? 'destructive' :
                            anomaly.severity === 'medium' ? 'default' : 'secondary'
                          }
                        >
                          {anomaly.severity}
                        </Badge>
                      </AlertTitle>
                      <AlertDescription>
                        <div className="mt-2 space-y-1">
                          <div className="text-sm">
                            Score: {anomaly.score.toFixed(3)} | Confidence: {(anomaly.confidence * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm">
                            Time: {new Date(anomaly.timestamp).toLocaleString()}
                          </div>
                          <div className="text-sm">
                            Top Contributors: {anomaly.contributing_features.slice(0, 2).map(f => f.feature).join(', ')}
                          </div>
                        </div>
                      </AlertDescription>
                    </Alert>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </CardContent>
    </Card>
  );
};

// ============================================================================
// Part 5: Drift Monitoring
// ============================================================================

export const DriftMonitoring: React.FC = () => {
  const [driftHistory, setDriftHistory] = useState<DriftMetric[]>([]);
  const [driftMethod, setDriftMethod] = useState('psi');
  const [driftThreshold, setDriftThreshold] = useState(0.1);
  const [currentDrift, setCurrentDrift] = useState<DriftMetric | null>(null);

  useEffect(() => {
    // Generate synthetic drift history
    const data: DriftMetric[] = Array.from({ length: 30 }, (_, i) => {
      const score = Math.random() * 0.3;
      const driftDetected = score > driftThreshold;
      
      return {
        timestamp: new Date(Date.now() - (29 - i) * 86400000).toISOString(),
        score,
        threshold: driftThreshold,
        method: driftMethod,
        severity: score > 0.2 ? 'high' : score > 0.1 ? 'medium' : 'low',
        affected_features: driftDetected 
          ? ['temperature_mean', 'pressure_std', 'power_mean'].slice(0, Math.floor(Math.random() * 3) + 1)
          : [],
      };
    });
    setDriftHistory(data);
    setCurrentDrift(data[data.length - 1]);
  }, [driftMethod, driftThreshold]);

  const driftDetected = currentDrift && currentDrift.score > currentDrift.threshold;
  const driftCount = driftHistory.filter(d => d.score > d.threshold).length;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              Data & Model Drift Monitoring
            </CardTitle>
            <CardDescription>
              Track distribution changes and model performance degradation
            </CardDescription>
          </div>
          <Badge variant={driftDetected ? 'destructive' : 'default'}>
            {driftDetected ? 'Drift Detected' : 'No Drift'}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Configuration */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Drift Detection Method</Label>
              <Select value={driftMethod} onValueChange={setDriftMethod}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="psi">Population Stability Index (PSI)</SelectItem>
                  <SelectItem value="ks_test">Kolmogorov-Smirnov Test</SelectItem>
                  <SelectItem value="kl_divergence">KL Divergence</SelectItem>
                  <SelectItem value="chi_square">Chi-Square Test</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Drift Threshold: {driftThreshold.toFixed(2)}</Label>
              <Slider
                value={[driftThreshold]}
                onValueChange={([value]) => setDriftThreshold(value)}
                min={0.05}
                max={0.5}
                step={0.05}
              />
            </div>
          </div>

          {/* Current Status */}
          {currentDrift && (
            <Alert variant={driftDetected ? 'destructive' : 'default'}>
              {driftDetected ? (
                <AlertTriangle className="w-4 h-4" />
              ) : (
                <CheckCircle className="w-4 h-4" />
              )}
              <AlertTitle>Current Drift Status</AlertTitle>
              <AlertDescription>
                <div className="mt-2 space-y-2">
                  <div className="flex justify-between">
                    <span>Drift Score:</span>
                    <span className="font-semibold">{currentDrift.score.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Severity:</span>
                    <Badge variant={
                      currentDrift.severity === 'high' ? 'destructive' :
                      currentDrift.severity === 'medium' ? 'default' : 'secondary'
                    }>
                      {currentDrift.severity}
                    </Badge>
                  </div>
                  {driftDetected && currentDrift.affected_features.length > 0 && (
                    <div>
                      <span className="font-semibold">Affected Features:</span>
                      <div className="mt-1 flex flex-wrap gap-1">
                        {currentDrift.affected_features.map(feature => (
                          <Badge key={feature} variant="outline">{feature}</Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </AlertDescription>
            </Alert>
          )}

          {/* Drift History */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Drift Score Over Time</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={driftHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={(value) => new Date(value).toLocaleDateString()}
                  />
                  <YAxis />
                  <Tooltip 
                    labelFormatter={(value) => new Date(value).toLocaleString()}
                    formatter={(value: any) => value.toFixed(4)}
                  />
                  <Legend />
                  <ReferenceLine 
                    y={driftThreshold} 
                    stroke="#ef4444" 
                    strokeDasharray="3 3" 
                    label="Threshold"
                  />
                  <Area 
                    type="monotone" 
                    dataKey="score" 
                    fill="#3b82f6" 
                    stroke="#3b82f6" 
                    fillOpacity={0.3}
                    name="Drift Score"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="threshold" 
                    stroke="#ef4444" 
                    strokeWidth={2}
                    dot={false}
                    name="Threshold"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Summary Statistics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-orange-600">
                    {driftCount}
                  </div>
                  <div className="text-sm text-muted-foreground">Drift Events</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Last 30 days
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-600">
                    {((driftCount / driftHistory.length) * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-muted-foreground">Drift Rate</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Percentage of days
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-green-600">
                    {currentDrift ? (currentDrift.score / currentDrift.threshold * 100).toFixed(0) : 0}%
                  </div>
                  <div className="text-sm text-muted-foreground">of Threshold</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Current score
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recommendations */}
          {driftDetected && (
            <Alert>
              <AlertTriangle className="w-4 h-4" />
              <AlertTitle>Recommended Actions</AlertTitle>
              <AlertDescription>
                <ul className="list-disc list-inside mt-2 space-y-1">
                  <li>Review process parameters for recent changes</li>
                  <li>Check equipment calibration status</li>
                  <li>Consider retraining model with recent data</li>
                  <li>Investigate affected features: {currentDrift?.affected_features.join(', ')}</li>
                </ul>
              </AlertDescription>
            </Alert>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

// ============================================================================
// Part 6: Time Series Forecast
// ============================================================================

export const TimeSeriesForecast: React.FC = () => {
  const [forecastData, setForecastData] = useState<ForecastPoint[]>([]);
  const [forecastHorizon, setForecastHorizon] = useState(30);
  const [forecastMethod, setForecastMethod] = useState('linear');
  const [isForecasting, setIsForecasting] = useState(false);

  useEffect(() => {
    // Generate synthetic historical + forecast data
    const historical: ForecastPoint[] = Array.from({ length: 90 }, (_, i) => ({
      timestamp: new Date(Date.now() - (89 - i) * 86400000).toISOString(),
      value: 100 + 0.1 * i + 10 * Math.sin((i / 365) * 2 * Math.PI) + (Math.random() - 0.5) * 5,
      lower_bound: 0,
      upper_bound: 0,
      is_actual: true,
    }));

    const forecast: ForecastPoint[] = Array.from({ length: forecastHorizon }, (_, i) => {
      const baseValue = historical[historical.length - 1].value;
      const trend = 0.1 * i;
      const seasonal = 10 * Math.sin(((89 + i) / 365) * 2 * Math.PI);
      const predicted = baseValue + trend + seasonal;
      const uncertainty = 5 * Math.sqrt(i + 1);

      return {
        timestamp: new Date(Date.now() + i * 86400000).toISOString(),
        value: predicted,
        lower_bound: predicted - uncertainty,
        upper_bound: predicted + uncertainty,
        is_actual: false,
      };
    });

    setForecastData([...historical, ...forecast]);
  }, [forecastHorizon, forecastMethod]);

  const handleForecast = () => {
    setIsForecasting(true);
    setTimeout(() => setIsForecasting(false), 2000);
  };

  const historicalData = forecastData.filter(d => d.is_actual);
  const forecastOnlyData = forecastData.filter(d => !d.is_actual);

  // Calculate forecast metrics
  const avgForecast = forecastOnlyData.reduce((sum, d) => sum + d.value, 0) / forecastOnlyData.length;
  const avgUncertainty = forecastOnlyData.reduce((sum, d) => sum + (d.upper_bound - d.lower_bound), 0) / forecastOnlyData.length;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Clock className="w-5 h-5" />
              Time Series Forecasting
            </CardTitle>
            <CardDescription>
              Predictive maintenance and calibration interval optimization
            </CardDescription>
          </div>
          <Button 
            onClick={handleForecast}
            disabled={isForecasting}
          >
            {isForecasting ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                Forecasting...
              </>
            ) : (
              <>
                <LineChart className="w-4 h-4 mr-2" />
                Update Forecast
              </>
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Configuration */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Forecast Method</Label>
              <Select value={forecastMethod} onValueChange={setForecastMethod}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="linear">Linear Trend</SelectItem>
                  <SelectItem value="prophet">Facebook Prophet</SelectItem>
                  <SelectItem value="arima">ARIMA</SelectItem>
                  <SelectItem value="lstm">LSTM Neural Network</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Forecast Horizon: {forecastHorizon} days</Label>
              <Slider
                value={[forecastHorizon]}
                onValueChange={([value]) => setForecastHorizon(value)}
                min={7}
                max={90}
                step={7}
              />
            </div>
          </div>

          {/* Forecast Visualization */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Historical Data & Forecast</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={forecastData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={(value) => new Date(value).toLocaleDateString()}
                  />
                  <YAxis />
                  <Tooltip 
                    labelFormatter={(value) => new Date(value).toLocaleDateString()}
                    formatter={(value: any) => value.toFixed(2)}
                  />
                  <Legend />
                  
                  {/* Confidence interval */}
                  <Area
                    type="monotone"
                    dataKey="upper_bound"
                    stroke="none"
                    fill="#93c5fd"
                    fillOpacity={0.3}
                    name="Upper Bound"
                  />
                  <Area
                    type="monotone"
                    dataKey="lower_bound"
                    stroke="none"
                    fill="#93c5fd"
                    fillOpacity={0.3}
                    name="Lower Bound"
                  />
                  
                  {/* Historical data */}
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={{ fill: '#3b82f6', r: 2 }}
                    name="Historical"
                    connectNulls
                  />
                  
                  {/* Forecast */}
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#10b981"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={{ fill: '#10b981', r: 2 }}
                    name="Forecast"
                    connectNulls
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Forecast Summary */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-600">
                    {avgForecast.toFixed(1)}
                  </div>
                  <div className="text-sm text-muted-foreground">Average Forecast</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Next {forecastHorizon} days
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-green-600">
                    ±{(avgUncertainty / 2).toFixed(1)}
                  </div>
                  <div className="text-sm text-muted-foreground">Avg Uncertainty</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    95% confidence interval
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-purple-600">
                    {forecastOnlyData[forecastOnlyData.length - 1]?.value.toFixed(1)}
                  </div>
                  <div className="text-sm text-muted-foreground">End Value</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Day {forecastHorizon}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Insights & Recommendations */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Predictive Insights</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <Alert>
                  <TrendingUp className="w-4 h-4" />
                  <AlertTitle>Trend Analysis</AlertTitle>
                  <AlertDescription>
                    Gradual upward trend detected. Expected increase of{' '}
                    {((forecastOnlyData[forecastOnlyData.length - 1]?.value - historicalData[historicalData.length - 1]?.value) / historicalData[historicalData.length - 1]?.value * 100).toFixed(1)}%
                    over next {forecastHorizon} days.
                  </AlertDescription>
                </Alert>

                <Alert>
                  <Clock className="w-4 h-4" />
                  <AlertTitle>Maintenance Recommendation</AlertTitle>
                  <AlertDescription>
                    Based on forecast, next calibration should be scheduled in{' '}
                    {Math.floor(forecastHorizon * 0.7)} days to maintain optimal performance.
                  </AlertDescription>
                </Alert>

                <Alert>
                  <Settings className="w-4 h-4" />
                  <AlertTitle>Process Optimization</AlertTitle>
                  <AlertDescription>
                    Current trajectory suggests process parameters are stable.
                    Continue monitoring for early detection of deviations.
                  </AlertDescription>
                </Alert>
              </div>
            </CardContent>
          </Card>
        </div>
      </CardContent>
    </Card>
  );
};

// ============================================================================
// Main Session 14 Component
// ============================================================================

export const Session14MLInterface: React.FC = () => {
  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Session Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl font-bold flex items-center gap-2">
                <Brain className="w-6 h-6" />
                Session 14: ML & Virtual Metrology Suite
              </CardTitle>
              <CardDescription className="mt-2">
                Comprehensive machine learning platform for predictive analytics and process optimization
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Badge>Machine Learning</Badge>
              <Badge>Virtual Metrology</Badge>
              <Badge>Predictive Analytics</Badge>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Main Interface Tabs */}
      <Tabs defaultValue="training" className="space-y-6">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="training">Model Training</TabsTrigger>
          <TabsTrigger value="features">Feature Importance</TabsTrigger>
          <TabsTrigger value="prediction">Predictions</TabsTrigger>
          <TabsTrigger value="anomaly">Anomaly Monitor</TabsTrigger>
          <TabsTrigger value="drift">Drift Detection</TabsTrigger>
          <TabsTrigger value="forecast">Time Series</TabsTrigger>
        </TabsList>

        <TabsContent value="training">
          <ModelTrainingDashboard />
        </TabsContent>

        <TabsContent value="features">
          <FeatureImportanceChart />
        </TabsContent>

        <TabsContent value="prediction">
          <PredictionDashboard />
        </TabsContent>

        <TabsContent value="anomaly">
          <AnomalyMonitor />
        </TabsContent>

        <TabsContent value="drift">
          <DriftMonitoring />
        </TabsContent>

        <TabsContent value="forecast">
          <TimeSeriesForecast />
        </TabsContent>
      </Tabs>

      {/* Platform Progress */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Session Progress</p>
              <Progress value={100} className="w-[200px]" />
            </div>
            <div className="text-sm text-muted-foreground">
              Platform Completion: 87.5% (14/16 sessions)
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Session14MLInterface;
