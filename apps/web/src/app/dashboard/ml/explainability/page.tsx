/*
 * Model Explainability & Interpretability Page
 * SHAP, LIME, PDP, and Feature Analysis
 */

'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { 
  Eye, 
  BarChart3, 
  GitBranch, 
  TrendingUp,
  Download,
  Upload,
  Play,
  Loader2,
  AlertCircle
} from 'lucide-react'
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from 'recharts'

// Type definitions
interface ModelInfo {
  model_id: string
  model_name: string
  model_type: string
  trained_at: string
  metrics: Record<string, number>
}

interface ExplanationJob {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  explanation_type: string
  created_at: string
}

interface SHAPResult {
  feature_names: string[]
  shap_values: number[][]
  base_value: number
  feature_importance: Array<{
    feature: string
    importance: number
  }>
}

interface LIMEResult {
  feature_weights: Array<{
    feature: string
    weight: number
  }>
  prediction: number
  local_r2: number
}

interface PDPResult {
  feature: string
  values: number[]
  pdp_values: number[]
  ice_lines?: number[][]
}

const API_BASE_URL = 'http://localhost:8001/api/v1'

export default function ExplainabilityPage() {
  const [activeTab, setActiveTab] = useState<'shap' | 'lime' | 'pdp' | 'interactions'>('shap')
  const [models, setModels] = useState<ModelInfo[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [isLoading, setIsLoading] = useState(false)
  const [currentJob, setCurrentJob] = useState<ExplanationJob | null>(null)
  const [shapResults, setSHAPResults] = useState<SHAPResult | null>(null)
  const [limeResults, setLIMEResults] = useState<LIMEResult | null>(null)
  const [pdpResults, setPDPResults] = useState<PDPResult | null>(null)
  const [selectedFeature, setSelectedFeature] = useState<string>('')

  // Load available models
  useEffect(() => {
    fetchModels()
  }, [])

  const fetchModels = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/explainability/available-models`)
      if (response.ok) {
        const data = await response.json()
        setModels(data.models || [])
        if (data.models?.length > 0) {
          setSelectedModel(data.models[0].model_id)
        }
      }
    } catch (error) {
      console.error('Error fetching models:', error)
    }
  }

  // Poll for job status
  useEffect(() => {
    if (!currentJob || currentJob.status === 'completed' || currentJob.status === 'failed') {
      return
    }

    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/explainability/job/${currentJob.job_id}/status`)
        if (response.ok) {
          const data = await response.json()
          setCurrentJob(data)
          
          if (data.status === 'completed') {
            fetchResults(data.job_id, currentJob.explanation_type)
          }
        }
      } catch (error) {
        console.error('Error polling job status:', error)
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [currentJob])

  const fetchResults = async (jobId: string, type: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/explainability/job/${jobId}/results`)
      if (response.ok) {
        const data = await response.json()
        
        switch (type) {
          case 'shap':
            setSHAPResults(data)
            break
          case 'lime':
            setLIMEResults(data)
            break
          case 'pdp':
            setPDPResults(data)
            break
        }
      }
    } catch (error) {
      console.error('Error fetching results:', error)
    }
  }

  const generateSHAP = async () => {
    if (!selectedModel) return
    
    setIsLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/explainability/shap`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: selectedModel,
          n_samples: 100
        })
      })

      if (response.ok) {
        const data = await response.json()
        setCurrentJob({ ...data, explanation_type: 'shap' })
      }
    } catch (error) {
      console.error('Error generating SHAP:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const generateLIME = async () => {
    if (!selectedModel) return
    
    setIsLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/explainability/lime`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: selectedModel,
          instance_index: 0,
          n_features: 10
        })
      })

      if (response.ok) {
        const data = await response.json()
        setCurrentJob({ ...data, explanation_type: 'lime' })
      }
    } catch (error) {
      console.error('Error generating LIME:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const generatePDP = async (feature: string) => {
    if (!selectedModel || !feature) return
    
    setIsLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/explainability/pdp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: selectedModel,
          feature: feature,
          n_grid_points: 50
        })
      })

      if (response.ok) {
        const data = await response.json()
        setCurrentJob({ ...data, explanation_type: 'pdp' })
      }
    } catch (error) {
      console.error('Error generating PDP:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Model Explainability</h1>
        <p className="text-muted-foreground mt-2">
          Understand and interpret ML model predictions with SHAP, LIME, and feature analysis
        </p>
      </div>

      {/* Model Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            Select Model to Explain
          </CardTitle>
          <CardDescription>Choose a trained model for explanation analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                <option value="">Select a model...</option>
                {models.map((model) => (
                  <option key={model.model_id} value={model.model_id}>
                    {model.model_name} ({model.model_type})
                  </option>
                ))}
              </select>
            </div>

            {selectedModel && models.find(m => m.model_id === selectedModel) && (
              <div className="p-4 bg-muted/50 rounded-lg">
                <div className="text-sm text-muted-foreground mb-1">Model Type</div>
                <div className="font-semibold">
                  {models.find(m => m.model_id === selectedModel)?.model_type}
                </div>
                <div className="text-sm text-muted-foreground mt-2">
                  Trained: {new Date(models.find(m => m.model_id === selectedModel)?.trained_at || '').toLocaleDateString()}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Explanation Methods */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="shap" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            SHAP
          </TabsTrigger>
          <TabsTrigger value="lime" className="flex items-center gap-2">
            <GitBranch className="h-4 w-4" />
            LIME
          </TabsTrigger>
          <TabsTrigger value="pdp" className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            PDP/ICE
          </TabsTrigger>
          <TabsTrigger value="interactions" className="flex items-center gap-2">
            <Eye className="h-4 w-4" />
            Interactions
          </TabsTrigger>
        </TabsList>

        {/* SHAP Tab */}
        <TabsContent value="shap" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>SHAP Values Analysis</CardTitle>
              <CardDescription>
                SHapley Additive exPlanations - Global feature importance
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button 
                onClick={generateSHAP} 
                disabled={!selectedModel || isLoading}
                className="w-full"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating SHAP Values...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Generate SHAP Values
                  </>
                )}
              </Button>

              {shapResults && (
                <>
                  <div className="pt-4">
                    <h4 className="font-semibold mb-4">Feature Importance (SHAP)</h4>
                    <ResponsiveContainer width="100%" height={400}>
                      <BarChart 
                        data={shapResults.feature_importance}
                        layout="vertical"
                        margin={{ left: 100 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" />
                        <YAxis type="category" dataKey="feature" />
                        <Tooltip />
                        <Bar dataKey="importance" fill="hsl(var(--primary))">
                          {shapResults.feature_importance.map((entry, index) => (
                            <Cell 
                              key={`cell-${index}`}
                              fill={entry.importance > 0 ? 'hsl(var(--primary))' : 'hsl(var(--destructive))'}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="grid grid-cols-2 gap-4 pt-4">
                    <div className="p-4 bg-muted/50 rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">Base Value</div>
                      <div className="text-2xl font-bold">{shapResults.base_value.toFixed(4)}</div>
                    </div>
                    <div className="p-4 bg-muted/50 rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">Features Analyzed</div>
                      <div className="text-2xl font-bold">{shapResults.feature_names.length}</div>
                    </div>
                  </div>

                  <Button variant="outline" className="w-full">
                    <Download className="mr-2 h-4 w-4" />
                    Export SHAP Results
                  </Button>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* LIME Tab */}
        <TabsContent value="lime" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>LIME Explanation</CardTitle>
              <CardDescription>
                Local Interpretable Model-agnostic Explanations
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button 
                onClick={generateLIME} 
                disabled={!selectedModel || isLoading}
                className="w-full"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating LIME Explanation...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Generate LIME Explanation
                  </>
                )}
              </Button>

              {limeResults && (
                <>
                  <div className="pt-4">
                    <h4 className="font-semibold mb-4">Local Feature Weights</h4>
                    <ResponsiveContainer width="100%" height={400}>
                      <BarChart 
                        data={limeResults.feature_weights}
                        layout="vertical"
                        margin={{ left: 100 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" />
                        <YAxis type="category" dataKey="feature" />
                        <Tooltip />
                        <Bar dataKey="weight" fill="hsl(var(--primary))">
                          {limeResults.feature_weights.map((entry, index) => (
                            <Cell 
                              key={`cell-${index}`}
                              fill={entry.weight > 0 ? 'hsl(var(--primary))' : 'hsl(var(--destructive))'}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="grid grid-cols-2 gap-4 pt-4">
                    <div className="p-4 bg-muted/50 rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">Local Prediction</div>
                      <div className="text-2xl font-bold">{limeResults.prediction.toFixed(4)}</div>
                    </div>
                    <div className="p-4 bg-muted/50 rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">Local R²</div>
                      <div className="text-2xl font-bold">{limeResults.local_r2.toFixed(4)}</div>
                    </div>
                  </div>

                  <Button variant="outline" className="w-full">
                    <Download className="mr-2 h-4 w-4" />
                    Export LIME Results
                  </Button>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* PDP Tab */}
        <TabsContent value="pdp" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Partial Dependence Plot</CardTitle>
              <CardDescription>
                Understand how features affect predictions
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Select Feature</label>
                <select
                  value={selectedFeature}
                  onChange={(e) => setSelectedFeature(e.target.value)}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                >
                  <option value="">Select a feature...</option>
                  {shapResults?.feature_names.map((feature) => (
                    <option key={feature} value={feature}>{feature}</option>
                  ))}
                </select>
              </div>

              <Button 
                onClick={() => generatePDP(selectedFeature)} 
                disabled={!selectedModel || !selectedFeature || isLoading}
                className="w-full"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating PDP...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Generate Partial Dependence Plot
                  </>
                )}
              </Button>

              {pdpResults && (
                <>
                  <div className="pt-4">
                    <h4 className="font-semibold mb-4">
                      PDP for {pdpResults.feature}
                    </h4>
                    <ResponsiveContainer width="100%" height={400}>
                      <LineChart 
                        data={pdpResults.values.map((val, idx) => ({
                          value: val,
                          pdp: pdpResults.pdp_values[idx]
                        }))}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                          dataKey="value" 
                          label={{ value: pdpResults.feature, position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis label={{ value: 'Prediction', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="pdp" 
                          stroke="hsl(var(--primary))" 
                          strokeWidth={2}
                          name="Partial Dependence"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  <Button variant="outline" className="w-full">
                    <Download className="mr-2 h-4 w-4" />
                    Export PDP Results
                  </Button>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Interactions Tab */}
        <TabsContent value="interactions" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Feature Interactions</CardTitle>
              <CardDescription>
                Discover how features work together
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col items-center justify-center py-16">
              <Eye className="h-16 w-16 text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">Feature Interactions</h3>
              <p className="text-muted-foreground text-center max-w-md">
                Run SHAP analysis first to enable feature interaction detection
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
