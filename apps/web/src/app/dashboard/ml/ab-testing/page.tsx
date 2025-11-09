/*
 * A/B Testing & Model Comparison Page
 * Statistical testing and multi-armed bandit optimization
 */

'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  GitCompare,
  Play,
  Trophy,
  TrendingUp,
  BarChart3,
  PlusCircle,
  Trash2,
  Download,
  Loader2,
  CheckCircle2,
  AlertCircle
} from 'lucide-react'
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from 'recharts'

// Type definitions
interface Variant {
  id: string
  name: string
  model_id: string
  model_type: string
  allocation?: number
}

interface Experiment {
  experiment_id: string
  name: string
  description: string
  status: 'draft' | 'running' | 'completed' | 'paused'
  variants: Variant[]
  created_at: string
  started_at?: string
  completed_at?: string
}

interface ExperimentResults {
  experiment_id: string
  winner?: string
  statistical_significance: boolean
  p_value: number
  confidence_interval: number
  variant_results: Array<{
    variant_id: string
    variant_name: string
    samples: number
    mean_performance: number
    std_dev: number
    confidence_lower: number
    confidence_upper: number
  }>
  bayesian_analysis?: {
    posterior_probabilities: Record<string, number>
    expected_loss: Record<string, number>
  }
}

const API_BASE_URL = 'http://localhost:8001/api/v1'

export default function ABTestingPage() {
  const [activeTab, setActiveTab] = useState<'experiments' | 'create' | 'results'>('experiments')
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [selectedExperiment, setSelectedExperiment] = useState<string>('')
  const [results, setResults] = useState<ExperimentResults | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  
  // New experiment form
  const [newExperiment, setNewExperiment] = useState({
    name: '',
    description: '',
    metric: 'r2',
    strategy: 'thompson_sampling'
  })
  const [variants, setVariants] = useState<Array<{name: string, model_id: string}>>([
    { name: 'Variant A', model_id: '' },
    { name: 'Variant B', model_id: '' }
  ])

  // Load experiments
  useEffect(() => {
    fetchExperiments()
  }, [])

  const fetchExperiments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/ab-testing/experiments`)
      if (response.ok) {
        const data = await response.json()
        setExperiments(data.experiments || [])
      }
    } catch (error) {
      console.error('Error fetching experiments:', error)
    }
  }

  const createExperiment = async () => {
    setIsLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/ab-testing/experiment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newExperiment.name,
          description: newExperiment.description,
          variants: variants.map(v => ({
            name: v.name,
            model_id: v.model_id
          })),
          metric: newExperiment.metric,
          strategy: newExperiment.strategy
        })
      })

      if (response.ok) {
        await fetchExperiments()
        setActiveTab('experiments')
        // Reset form
        setNewExperiment({ name: '', description: '', metric: 'r2', strategy: 'thompson_sampling' })
        setVariants([{ name: 'Variant A', model_id: '' }, { name: 'Variant B', model_id: '' }])
      }
    } catch (error) {
      console.error('Error creating experiment:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const startExperiment = async (experimentId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/ab-testing/experiment/${experimentId}/start`, {
        method: 'POST'
      })
      if (response.ok) {
        await fetchExperiments()
      }
    } catch (error) {
      console.error('Error starting experiment:', error)
    }
  }

  const stopExperiment = async (experimentId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/ab-testing/experiment/${experimentId}/stop`, {
        method: 'POST'
      })
      if (response.ok) {
        await fetchExperiments()
      }
    } catch (error) {
      console.error('Error stopping experiment:', error)
    }
  }

  const fetchResults = async (experimentId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/ab-testing/experiment/${experimentId}/results`)
      if (response.ok) {
        const data = await response.json()
        setResults(data)
        setActiveTab('results')
      }
    } catch (error) {
      console.error('Error fetching results:', error)
    }
  }

  const addVariant = () => {
    setVariants([...variants, { name: `Variant ${String.fromCharCode(65 + variants.length)}`, model_id: '' }])
  }

  const removeVariant = (index: number) => {
    if (variants.length > 2) {
      setVariants(variants.filter((_, i) => i !== index))
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">A/B Testing & Model Comparison</h1>
        <p className="text-muted-foreground mt-2">
          Statistical testing and multi-armed bandit optimization for model selection
        </p>
      </div>

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="experiments" className="flex items-center gap-2">
            <GitCompare className="h-4 w-4" />
            Experiments
          </TabsTrigger>
          <TabsTrigger value="create" className="flex items-center gap-2">
            <PlusCircle className="h-4 w-4" />
            Create New
          </TabsTrigger>
          <TabsTrigger value="results" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Results
          </TabsTrigger>
        </TabsList>

        {/* Experiments Tab */}
        <TabsContent value="experiments" className="space-y-6">
          {experiments.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-16">
                <GitCompare className="h-16 w-16 text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No Experiments Yet</h3>
                <p className="text-muted-foreground text-center max-w-md mb-4">
                  Create your first A/B test to compare model performance
                </p>
                <Button onClick={() => setActiveTab('create')}>
                  <PlusCircle className="mr-2 h-4 w-4" />
                  Create Experiment
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 gap-6">
              {experiments.map((experiment) => (
                <Card key={experiment.experiment_id}>
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle>{experiment.name}</CardTitle>
                        <CardDescription>{experiment.description}</CardDescription>
                      </div>
                      <Badge variant={
                        experiment.status === 'running' ? 'default' :
                        experiment.status === 'completed' ? 'secondary' :
                        'outline'
                      }>
                        {experiment.status.toUpperCase()}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Variants */}
                    <div>
                      <h4 className="text-sm font-semibold mb-2">Variants ({experiment.variants.length})</h4>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                        {experiment.variants.map((variant) => (
                          <div key={variant.id} className="p-3 bg-muted/50 rounded-lg">
                            <div className="font-medium text-sm">{variant.name}</div>
                            <div className="text-xs text-muted-foreground">{variant.model_type}</div>
                            {variant.allocation && (
                              <div className="text-xs text-muted-foreground mt-1">
                                Traffic: {(variant.allocation * 100).toFixed(0)}%
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex gap-2 pt-2">
                      {experiment.status === 'draft' && (
                        <Button 
                          onClick={() => startExperiment(experiment.experiment_id)}
                          size="sm"
                        >
                          <Play className="mr-2 h-4 w-4" />
                          Start Experiment
                        </Button>
                      )}
                      {experiment.status === 'running' && (
                        <Button 
                          onClick={() => stopExperiment(experiment.experiment_id)}
                          variant="destructive"
                          size="sm"
                        >
                          Stop Experiment
                        </Button>
                      )}
                      {(experiment.status === 'completed' || experiment.status === 'running') && (
                        <Button 
                          onClick={() => fetchResults(experiment.experiment_id)}
                          variant="outline"
                          size="sm"
                        >
                          <BarChart3 className="mr-2 h-4 w-4" />
                          View Results
                        </Button>
                      )}
                    </div>

                    {/* Timestamps */}
                    <div className="text-xs text-muted-foreground pt-2 border-t">
                      Created: {new Date(experiment.created_at).toLocaleDateString()}
                      {experiment.started_at && ` • Started: ${new Date(experiment.started_at).toLocaleDateString()}`}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        {/* Create Tab */}
        <TabsContent value="create" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Create New Experiment</CardTitle>
              <CardDescription>Set up an A/B test to compare model variants</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Basic Info */}
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Experiment Name</label>
                  <input
                    type="text"
                    value={newExperiment.name}
                    onChange={(e) => setNewExperiment({ ...newExperiment, name: e.target.value })}
                    placeholder="e.g., RandomForest vs GradientBoosting"
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">Description</label>
                  <textarea
                    value={newExperiment.description}
                    onChange={(e) => setNewExperiment({ ...newExperiment, description: e.target.value })}
                    placeholder="Brief description of the experiment goals..."
                    rows={3}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  />
                </div>
              </div>

              {/* Variants */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-semibold">Variants</h4>
                  <Button onClick={addVariant} size="sm" variant="outline">
                    <PlusCircle className="mr-2 h-4 w-4" />
                    Add Variant
                  </Button>
                </div>

                <div className="space-y-3">
                  {variants.map((variant, index) => (
                    <div key={index} className="flex gap-2">
                      <input
                        type="text"
                        value={variant.name}
                        onChange={(e) => {
                          const newVariants = [...variants]
                          newVariants[index].name = e.target.value
                          setVariants(newVariants)
                        }}
                        placeholder="Variant name"
                        className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
                      />
                      <input
                        type="text"
                        value={variant.model_id}
                        onChange={(e) => {
                          const newVariants = [...variants]
                          newVariants[index].model_id = e.target.value
                          setVariants(newVariants)
                        }}
                        placeholder="Model ID"
                        className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
                      />
                      {variants.length > 2 && (
                        <Button 
                          onClick={() => removeVariant(index)}
                          variant="ghost"
                          size="icon"
                        >
                          <Trash2 className="h-4 w-4 text-destructive" />
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Settings */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Optimization Metric</label>
                  <select
                    value={newExperiment.metric}
                    onChange={(e) => setNewExperiment({ ...newExperiment, metric: e.target.value })}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  >
                    <option value="r2">R² Score</option>
                    <option value="rmse">RMSE</option>
                    <option value="mae">MAE</option>
                    <option value="accuracy">Accuracy</option>
                  </select>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">Allocation Strategy</label>
                  <select
                    value={newExperiment.strategy}
                    onChange={(e) => setNewExperiment({ ...newExperiment, strategy: e.target.value })}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  >
                    <option value="thompson_sampling">Thompson Sampling</option>
                    <option value="epsilon_greedy">Epsilon Greedy</option>
                    <option value="ucb">Upper Confidence Bound</option>
                    <option value="uniform">Uniform (Classic A/B)</option>
                  </select>
                </div>
              </div>

              {/* Create Button */}
              <Button 
                onClick={createExperiment}
                disabled={!newExperiment.name || variants.some(v => !v.model_id) || isLoading}
                className="w-full"
                size="lg"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Creating Experiment...
                  </>
                ) : (
                  <>
                    <PlusCircle className="mr-2 h-5 w-5" />
                    Create Experiment
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value="results" className="space-y-6">
          {!results ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-16">
                <BarChart3 className="h-16 w-16 text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No Results Selected</h3>
                <p className="text-muted-foreground text-center max-w-md">
                  Select an experiment from the Experiments tab to view its results
                </p>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Summary */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <Card>
                  <CardHeader className="pb-3">
                    <CardDescription>Winner</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center gap-2">
                      <Trophy className="h-5 w-5 text-yellow-500" />
                      <div className="text-2xl font-bold">
                        {results.winner || 'No clear winner'}
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardDescription>Statistical Significance</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center gap-2">
                      {results.statistical_significance ? (
                        <CheckCircle2 className="h-5 w-5 text-green-500" />
                      ) : (
                        <AlertCircle className="h-5 w-5 text-yellow-500" />
                      )}
                      <div className="text-2xl font-bold">
                        {results.statistical_significance ? 'Yes' : 'No'}
                      </div>
                    </div>
                    <div className="text-sm text-muted-foreground mt-1">
                      p-value: {results.p_value.toFixed(4)}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardDescription>Confidence</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">
                      {(results.confidence_interval * 100).toFixed(1)}%
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Variant Comparison */}
              <Card>
                <CardHeader>
                  <CardTitle>Variant Performance Comparison</CardTitle>
                  <CardDescription>Mean performance with confidence intervals</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={results.variant_results}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="variant_name" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="mean_performance" fill="hsl(var(--primary))" name="Mean Performance">
                        {results.variant_results.map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`}
                            fill={entry.variant_name === results.winner ? 'hsl(var(--primary))' : 'hsl(var(--muted))'}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Detailed Stats */}
              <Card>
                <CardHeader>
                  <CardTitle>Detailed Statistics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {results.variant_results.map((variant) => (
                      <div key={variant.variant_id} className="p-4 border rounded-lg">
                        <div className="flex items-center justify-between mb-3">
                          <h4 className="font-semibold">{variant.variant_name}</h4>
                          {variant.variant_name === results.winner && (
                            <Badge>
                              <Trophy className="mr-1 h-3 w-3" />
                              Winner
                            </Badge>
                          )}
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                          <div>
                            <div className="text-muted-foreground">Samples</div>
                            <div className="font-semibold">{variant.samples}</div>
                          </div>
                          <div>
                            <div className="text-muted-foreground">Mean</div>
                            <div className="font-semibold">{variant.mean_performance.toFixed(4)}</div>
                          </div>
                          <div>
                            <div className="text-muted-foreground">Std Dev</div>
                            <div className="font-semibold">{variant.std_dev.toFixed(4)}</div>
                          </div>
                          <div>
                            <div className="text-muted-foreground">95% CI</div>
                            <div className="font-semibold">
                              [{variant.confidence_lower.toFixed(3)}, {variant.confidence_upper.toFixed(3)}]
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Bayesian Analysis */}
              {results.bayesian_analysis && (
                <Card>
                  <CardHeader>
                    <CardTitle>Bayesian Analysis</CardTitle>
                    <CardDescription>Posterior probabilities of being the best variant</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {Object.entries(results.bayesian_analysis.posterior_probabilities).map(([variant, prob]) => (
                        <div key={variant}>
                          <div className="flex justify-between text-sm mb-1">
                            <span className="font-medium">{variant}</span>
                            <span className="text-muted-foreground">{(prob * 100).toFixed(1)}%</span>
                          </div>
                          <Progress value={prob * 100} />
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Export */}
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
