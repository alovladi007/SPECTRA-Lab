/*
 * AutoML Dashboard Page
 * Converted from uploaded automl-dashboard.html to Next.js TypeScript
 * Integrated with FastAPI backend
 */

'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  Play, 
  Settings, 
  TrendingUp, 
  Activity,
  Cpu,
  Database,
  Zap,
  CheckCircle2,
  AlertCircle,
  Clock,
  Download
} from 'lucide-react'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts'

// Type definitions
interface PipelineConfig {
  runModelSelection: boolean
  runHyperparameterTuning: boolean
  runNAS: boolean
  dataType: string
  metric: string
  nTrials: number
  cvFolds: number
  device: string
}

interface AutoMLJob {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  current_step?: string
  started_at?: string
  completed_at?: string
  error?: string
}

interface ModelCandidate {
  name: string
  cvScore: number
  testScore: number
  inferenceTime: number
  complexity: number
}

interface AutoMLResults {
  modelSelection?: {
    bestModel: string
    bestScore: number
    allCandidates: ModelCandidate[]
  }
  hyperparameterTuning?: {
    modelType: string
    bestCvScore: number
    nTrials: number
    bestParams: Record<string, any>
    testMetrics: {
      r2?: number
      rmse?: number
      mae?: number
      accuracy?: number
    }
    paramImportance: Record<string, number>
  }
  optimizationHistory?: Array<{
    trial: number
    score: number
  }>
}

const API_BASE_URL = 'http://localhost:8001/api/v1'

export default function AutoMLPage() {
  const [activeTab, setActiveTab] = useState<'configure' | 'monitor' | 'results'>('configure')
  const [pipeline, setPipeline] = useState<PipelineConfig>({
    runModelSelection: true,
    runHyperparameterTuning: true,
    runNAS: false,
    dataType: 'synthetic_yield',
    metric: 'r2',
    nTrials: 50,
    cvFolds: 5,
    device: 'cpu'
  })
  const [currentJob, setCurrentJob] = useState<AutoMLJob | null>(null)
  const [results, setResults] = useState<AutoMLResults | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  // Poll for job updates
  useEffect(() => {
    if (!currentJob || currentJob.status === 'completed' || currentJob.status === 'failed') {
      return
    }

    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/automl/job/${currentJob.job_id}/status`)
        if (response.ok) {
          const data = await response.json()
          setCurrentJob(data)
          
          if (data.status === 'completed') {
            fetchResults(data.job_id)
          }
        }
      } catch (error) {
        console.error('Error polling job status:', error)
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [currentJob])

  const fetchResults = async (jobId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/automl/job/${jobId}/results`)
      if (response.ok) {
        const data = await response.json()
        setResults(data)
        setActiveTab('results')
      }
    } catch (error) {
      console.error('Error fetching results:', error)
    }
  }

  const runPipeline = async () => {
    setIsLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/automl/run-pipeline`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_selection: pipeline.runModelSelection,
          hyperparameter_tuning: pipeline.runHyperparameterTuning,
          neural_architecture_search: pipeline.runNAS,
          data_type: pipeline.dataType,
          metric: pipeline.metric,
          n_trials: pipeline.nTrials,
          cv_folds: pipeline.cvFolds,
          device: pipeline.device
        })
      })

      if (response.ok) {
        const data = await response.json()
        setCurrentJob(data)
        setActiveTab('monitor')
      } else {
        console.error('Failed to start pipeline')
      }
    } catch (error) {
      console.error('Error starting pipeline:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">AutoML Dashboard</h1>
        <p className="text-muted-foreground mt-2">
          Automated machine learning for semiconductor manufacturing
        </p>
      </div>

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="configure" className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Configure
          </TabsTrigger>
          <TabsTrigger value="monitor" className="flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Monitor
          </TabsTrigger>
          <TabsTrigger value="results" className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            Results
          </TabsTrigger>
        </TabsList>

        {/* Configure Tab */}
        <TabsContent value="configure" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Pipeline Configuration</CardTitle>
              <CardDescription>
                Select AutoML stages and configure parameters
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Pipeline Stages */}
              <div className="space-y-4">
                <h3 className="text-sm font-semibold">Pipeline Stages</h3>
                <div className="space-y-3">
                  <label className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={pipeline.runModelSelection}
                      onChange={(e) => setPipeline({ ...pipeline, runModelSelection: e.target.checked })}
                      className="h-4 w-4 rounded border-gray-300"
                    />
                    <div>
                      <div className="font-medium">Model Selection</div>
                      <div className="text-sm text-muted-foreground">
                        Automatically evaluate 9+ ML algorithms
                      </div>
                    </div>
                  </label>

                  <label className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={pipeline.runHyperparameterTuning}
                      onChange={(e) => setPipeline({ ...pipeline, runHyperparameterTuning: e.target.checked })}
                      className="h-4 w-4 rounded border-gray-300"
                    />
                    <div>
                      <div className="font-medium">Hyperparameter Tuning</div>
                      <div className="text-sm text-muted-foreground">
                        Bayesian optimization with Optuna
                      </div>
                    </div>
                  </label>

                  <label className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={pipeline.runNAS}
                      onChange={(e) => setPipeline({ ...pipeline, runNAS: e.target.checked })}
                      className="h-4 w-4 rounded border-gray-300"
                    />
                    <div>
                      <div className="font-medium">Neural Architecture Search</div>
                      <div className="text-sm text-muted-foreground">
                        Evolutionary architecture design (GPU recommended)
                      </div>
                    </div>
                  </label>
                </div>
              </div>

              {/* Data Configuration */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Data Type</label>
                  <select
                    value={pipeline.dataType}
                    onChange={(e) => setPipeline({ ...pipeline, dataType: e.target.value })}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  >
                    <option value="synthetic_yield">Synthetic Yield Data</option>
                    <option value="wafer_map">Wafer Map Data</option>
                    <option value="process_control">Process Control Data</option>
                  </select>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">Optimization Metric</label>
                  <select
                    value={pipeline.metric}
                    onChange={(e) => setPipeline({ ...pipeline, metric: e.target.value })}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  >
                    <option value="r2">R² Score</option>
                    <option value="rmse">RMSE</option>
                    <option value="mae">MAE</option>
                    <option value="accuracy">Accuracy</option>
                  </select>
                </div>
              </div>

              {/* Advanced Settings */}
              <div className="space-y-4">
                <h3 className="text-sm font-semibold">Advanced Settings</h3>
                
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <label className="font-medium">Optimization Trials</label>
                    <span className="text-muted-foreground">{pipeline.nTrials}</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="200"
                    step="10"
                    value={pipeline.nTrials}
                    onChange={(e) => setPipeline({ ...pipeline, nTrials: parseInt(e.target.value) })}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>10 (Fast)</span>
                    <span>200 (Thorough)</span>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <label className="font-medium">Cross-Validation Folds</label>
                    <span className="text-muted-foreground">{pipeline.cvFolds}</span>
                  </div>
                  <input
                    type="range"
                    min="3"
                    max="10"
                    value={pipeline.cvFolds}
                    onChange={(e) => setPipeline({ ...pipeline, cvFolds: parseInt(e.target.value) })}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>3 (Fast)</span>
                    <span>10 (Robust)</span>
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">Compute Device</label>
                  <select
                    value={pipeline.device}
                    onChange={(e) => setPipeline({ ...pipeline, device: e.target.value })}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  >
                    <option value="cpu">CPU</option>
                    <option value="cuda">GPU (CUDA)</option>
                  </select>
                </div>
              </div>

              {/* Run Button */}
              <Button 
                onClick={runPipeline} 
                disabled={isLoading || currentJob?.status === 'running'}
                className="w-full"
                size="lg"
              >
                <Play className="mr-2 h-5 w-5" />
                {isLoading ? 'Starting Pipeline...' : 'Run AutoML Pipeline'}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Monitor Tab */}
        <TabsContent value="monitor" className="space-y-6">
          {!currentJob ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-16">
                <Activity className="h-16 w-16 text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No Active Pipeline</h3>
                <p className="text-muted-foreground text-center max-w-md">
                  Configure and start a pipeline from the Configure tab to monitor its progress here.
                </p>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Status Card */}
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>Pipeline Status</CardTitle>
                      <CardDescription>Job ID: {currentJob.job_id}</CardDescription>
                    </div>
                    <Badge variant={
                      currentJob.status === 'completed' ? 'default' :
                      currentJob.status === 'failed' ? 'destructive' :
                      currentJob.status === 'running' ? 'secondary' : 'outline'
                    }>
                      {currentJob.status === 'completed' && <CheckCircle2 className="mr-1 h-3 w-3" />}
                      {currentJob.status === 'failed' && <AlertCircle className="mr-1 h-3 w-3" />}
                      {currentJob.status === 'running' && <Clock className="mr-1 h-3 w-3" />}
                      {currentJob.status.toUpperCase()}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Progress Bar */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium">Overall Progress</span>
                      <span className="text-muted-foreground">{currentJob.progress}%</span>
                    </div>
                    <Progress value={currentJob.progress} className="h-3" />
                  </div>

                  {/* Current Stage */}
                  {currentJob.current_step && (
                    <div className="flex items-center gap-3 p-4 bg-muted/50 rounded-lg">
                      <Cpu className="h-5 w-5 text-primary animate-pulse" />
                      <div>
                        <div className="font-medium">Current Stage</div>
                        <div className="text-sm text-muted-foreground">{currentJob.current_step}</div>
                      </div>
                    </div>
                  )}

                  {/* Timestamps */}
                  <div className="grid grid-cols-2 gap-4 pt-4 border-t">
                    <div>
                      <div className="text-sm text-muted-foreground">Started</div>
                      <div className="font-medium">
                        {currentJob.started_at ? new Date(currentJob.started_at).toLocaleString() : '-'}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">Estimated Completion</div>
                      <div className="font-medium">
                        {currentJob.status === 'running' ? 'Calculating...' : '-'}
                      </div>
                    </div>
                  </div>

                  {/* Error Display */}
                  {currentJob.error && (
                    <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                      <div className="flex items-start gap-3">
                        <AlertCircle className="h-5 w-5 text-destructive mt-0.5" />
                        <div>
                          <div className="font-semibold text-destructive mb-1">Error</div>
                          <div className="text-sm">{currentJob.error}</div>
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Live Metrics */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="h-5 w-5" />
                    Live Metrics
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="p-4 bg-muted/50 rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">Models Evaluated</div>
                      <div className="text-2xl font-bold">-</div>
                    </div>
                    <div className="p-4 bg-muted/50 rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">Best Score</div>
                      <div className="text-2xl font-bold">-</div>
                    </div>
                    <div className="p-4 bg-muted/50 rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">Trials Completed</div>
                      <div className="text-2xl font-bold">-</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value="results" className="space-y-6">
          {!results ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-16">
                <TrendingUp className="h-16 w-16 text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No Results Available</h3>
                <p className="text-muted-foreground text-center max-w-md">
                  Complete a pipeline execution to view results and insights.
                </p>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {results.modelSelection && (
                  <Card>
                    <CardHeader className="pb-3">
                      <CardDescription>Best Model</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="text-3xl font-bold mb-1">{results.modelSelection.bestModel}</div>
                      <div className="text-sm text-muted-foreground">
                        Score: {results.modelSelection.bestScore.toFixed(4)}
                      </div>
                    </CardContent>
                  </Card>
                )}

                {results.hyperparameterTuning && (
                  <Card>
                    <CardHeader className="pb-3">
                      <CardDescription>Best CV Score</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="text-3xl font-bold mb-1">
                        {results.hyperparameterTuning.bestCvScore.toFixed(4)}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {results.hyperparameterTuning.nTrials} trials completed
                      </div>
                    </CardContent>
                  </Card>
                )}

                {results.hyperparameterTuning?.testMetrics && (
                  <Card>
                    <CardHeader className="pb-3">
                      <CardDescription>Test Metrics</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-1">
                        {results.hyperparameterTuning.testMetrics.r2 && (
                          <div className="flex justify-between text-sm">
                            <span>R²:</span>
                            <span className="font-medium">{results.hyperparameterTuning.testMetrics.r2.toFixed(4)}</span>
                          </div>
                        )}
                        {results.hyperparameterTuning.testMetrics.rmse && (
                          <div className="flex justify-between text-sm">
                            <span>RMSE:</span>
                            <span className="font-medium">{results.hyperparameterTuning.testMetrics.rmse.toFixed(4)}</span>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>

              {/* Model Comparison */}
              {results.modelSelection && (
                <Card>
                  <CardHeader>
                    <CardTitle>Model Comparison</CardTitle>
                    <CardDescription>Performance across all evaluated algorithms</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={results.modelSelection.allCandidates}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="cvScore" fill="hsl(var(--primary))" name="CV Score" />
                        <Bar dataKey="testScore" fill="hsl(var(--secondary))" name="Test Score" />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}

              {/* Optimization History */}
              {results.optimizationHistory && (
                <Card>
                  <CardHeader>
                    <CardTitle>Optimization History</CardTitle>
                    <CardDescription>Score improvement over trials</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={results.optimizationHistory}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="trial" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="score" stroke="hsl(var(--primary))" strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}

              {/* Best Hyperparameters */}
              {results.hyperparameterTuning && (
                <Card>
                  <CardHeader>
                    <CardTitle>Best Hyperparameters</CardTitle>
                    <CardDescription>Optimal configuration for {results.hyperparameterTuning.modelType}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      {Object.entries(results.hyperparameterTuning.bestParams).map(([key, value]) => (
                        <div key={key} className="p-3 bg-muted/50 rounded-lg">
                          <div className="text-sm text-muted-foreground mb-1">{key}</div>
                          <div className="font-mono font-semibold">{String(value)}</div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Export Button */}
              <Button variant="outline" className="w-full">
                <Download className="mr-2 h-4 w-4" />
                Export Results (JSON)
              </Button>
            </>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
