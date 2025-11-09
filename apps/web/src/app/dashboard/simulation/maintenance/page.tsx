'use client'

import React, { useState } from 'react'
import { Wrench, TrendingDown, AlertTriangle, CheckCircle, Activity, Info } from 'lucide-react'

interface MaintenanceRecommendation {
  action: string
  component?: string
  urgency: 'low' | 'medium' | 'high' | 'critical'
  confidence: number
  estimated_downtime_hours?: number
  cost_impact?: string
  delta_temp_c?: number
  delta_time_min?: number
  zones?: number[]
  runs_since_last_clean?: number
  recommended_frequency?: number
  expected_improvement?: string
}

interface MaintenanceResult {
  tool_health_score: number
  health_trend: 'improving' | 'stable' | 'declining' | 'critical'
  failure_probability_100runs: number
  risk_level: 'low' | 'medium' | 'high' | 'critical'
  recommendations: MaintenanceRecommendation[]
  next_maintenance_window: string
}

export default function PredictiveMaintenancePage() {
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<MaintenanceResult | null>(null)

  // FDC Parameters (simplified for demo)
  const [temperature, setTemperature] = useState(1000)
  const [tempVariance, setTempVariance] = useState(2)
  const [pressure, setPressure] = useState(100)
  const [mfcFlow, setMfcFlow] = useState(100)
  const [runsSinceClean, setRunsSinceClean] = useState(487)

  const runPredictiveMaintenance = async () => {
    setLoading(true)

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500))

      // Calculate health score based on parameters
      const tempHealth = 1 - (tempVariance / 10)
      const cleanHealth = Math.max(0, 1 - (runsSinceClean / 600))
      const overallHealth = (tempHealth + cleanHealth) / 2

      const healthTrend: MaintenanceResult['health_trend'] =
        overallHealth > 0.8 ? 'stable' :
        overallHealth > 0.6 ? 'declining' : 'critical'

      const riskLevel: MaintenanceResult['risk_level'] =
        overallHealth > 0.8 ? 'low' :
        overallHealth > 0.6 ? 'medium' : 'high'

      // Generate recommendations
      const recommendations: MaintenanceRecommendation[] = []

      if (tempVariance > 3) {
        recommendations.push({
          action: 'mfc_recalibration',
          component: 'MFC-1 (N2)',
          urgency: 'medium',
          confidence: 0.85,
          estimated_downtime_hours: 2,
          cost_impact: 'low'
        })
      }

      if (tempVariance > 2) {
        recommendations.push({
          action: 'recipe_trim',
          delta_temp_c: -5,
          delta_time_min: 2,
          confidence: 0.92,
          urgency: 'low',
          expected_improvement: '15% reduction in xj variance'
        })
      }

      if (tempVariance > 4) {
        recommendations.push({
          action: 'thermocouple_verification',
          zones: [3, 4],
          urgency: 'low',
          confidence: 0.65
        })
      }

      if (runsSinceClean > 450) {
        recommendations.push({
          action: 'tube_clean',
          runs_since_last_clean: runsSinceClean,
          recommended_frequency: 500,
          urgency: runsSinceClean > 500 ? 'medium' : 'low',
          confidence: 0.78
        })
      }

      const nextMaintenance = new Date()
      nextMaintenance.setDate(nextMaintenance.getDate() + 7)

      setResults({
        tool_health_score: overallHealth,
        health_trend: healthTrend,
        failure_probability_100runs: 1 - overallHealth,
        risk_level: riskLevel,
        recommendations,
        next_maintenance_window: nextMaintenance.toISOString()
      })
    } catch (error) {
      console.error('Predictive maintenance failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const getHealthColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 dark:text-green-400'
    if (score >= 0.6) return 'text-yellow-600 dark:text-yellow-400'
    if (score >= 0.4) return 'text-orange-600 dark:text-orange-400'
    return 'text-red-600 dark:text-red-400'
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
      case 'medium': return 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300'
      case 'high': return 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300'
      case 'critical': return 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
    }
  }

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'low': return 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
      case 'medium': return 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300'
      case 'high': return 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300'
      case 'critical': return 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg">
              <Wrench className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                Predictive Maintenance
              </h1>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Tool health scoring, FDC trend analysis & maintenance recommendations
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Left Panel - FDC Data */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                FDC Parameters
              </h2>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Temperature Setpoint: {temperature}°C
                </label>
                <input
                  type="range"
                  min="900"
                  max="1100"
                  step="10"
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Temp Variance: ±{tempVariance}°C
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="10"
                  step="0.5"
                  value={tempVariance}
                  onChange={(e) => setTempVariance(Number(e.target.value))}
                  className="w-full"
                />
                <p className="text-xs text-gray-500 mt-1">
                  {tempVariance < 2 ? 'Excellent' : tempVariance < 4 ? 'Good' : tempVariance < 6 ? 'Fair' : 'Poor'}
                </p>
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Pressure: {pressure.toFixed(1)} mTorr
                </label>
                <input
                  type="range"
                  min="95"
                  max="105"
                  step="0.1"
                  value={pressure}
                  onChange={(e) => setPressure(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  MFC Flow: {mfcFlow.toFixed(1)}%
                </label>
                <input
                  type="range"
                  min="95"
                  max="105"
                  step="0.1"
                  value={mfcFlow}
                  onChange={(e) => setMfcFlow(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Runs Since Clean: {runsSinceClean}
                </label>
                <input
                  type="range"
                  min="0"
                  max="600"
                  step="10"
                  value={runsSinceClean}
                  onChange={(e) => setRunsSinceClean(Number(e.target.value))}
                  className="w-full"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Recommended: every 500 runs
                </p>
              </div>

              <button
                onClick={runPredictiveMaintenance}
                disabled={loading}
                className="w-full py-3 px-4 bg-gradient-to-r from-orange-500 to-red-500 text-white font-medium rounded-lg hover:from-orange-600 hover:to-red-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <Activity className="w-4 h-4 animate-spin" />
                    Analyzing...
                  </span>
                ) : (
                  'Run Analysis'
                )}
              </button>
            </div>

            <div className="mt-6 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
              <div className="flex items-start gap-2">
                <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
                <div className="text-sm">
                  <p className="font-medium text-blue-900 dark:text-blue-100 mb-1">
                    Predictive Maintenance
                  </p>
                  <p className="text-blue-700 dark:text-blue-300 text-xs">
                    ML-based tool health scoring from FDC features.
                    Recommends preventive actions before failures occur.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel - Results */}
          <div className="lg:col-span-2">
            {results ? (
              <>
                {/* Health Metrics */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Tool Health</p>
                        <p className={`text-3xl font-bold mt-1 ${getHealthColor(results.tool_health_score)}`}>
                          {(results.tool_health_score * 100).toFixed(0)}%
                        </p>
                        <p className="text-xs text-gray-500 mt-1 capitalize">{results.health_trend}</p>
                      </div>
                      <div className="p-3 bg-gray-100 dark:bg-gray-700 rounded-lg">
                        <Activity className={`w-6 h-6 ${getHealthColor(results.tool_health_score)}`} />
                      </div>
                    </div>
                  </div>

                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Failure Risk (100 runs)</p>
                        <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                          {(results.failure_probability_100runs * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div className="p-3 bg-orange-100 dark:bg-orange-900/30 rounded-lg">
                        <TrendingDown className="w-6 h-6 text-orange-600 dark:text-orange-400" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Risk Level</p>
                        <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium mt-2 capitalize ${getRiskColor(results.risk_level)}`}>
                          {results.risk_level}
                        </span>
                      </div>
                      <div className="p-3 bg-red-100 dark:bg-red-900/30 rounded-lg">
                        <AlertTriangle className="w-6 h-6 text-red-600 dark:text-red-400" />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Recommendations */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                    <Wrench className="w-5 h-5" />
                    Maintenance Recommendations ({results.recommendations.length})
                  </h3>

                  <div className="space-y-4">
                    {results.recommendations.map((rec, idx) => (
                      <div
                        key={idx}
                        className="border border-gray-200 dark:border-gray-700 rounded-lg p-4"
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex-1">
                            <h4 className="font-medium text-gray-900 dark:text-white mb-1">
                              {rec.action.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                            </h4>
                            {rec.component && (
                              <p className="text-sm text-gray-600 dark:text-gray-400">
                                Component: {rec.component}
                              </p>
                            )}
                          </div>
                          <div className="flex gap-2">
                            <span className={`px-3 py-1 rounded-full text-xs font-medium capitalize ${getUrgencyColor(rec.urgency)}`}>
                              {rec.urgency}
                            </span>
                            <span className="px-3 py-1 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300">
                              {(rec.confidence * 100).toFixed(0)}% confident
                            </span>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3 text-sm">
                          {rec.estimated_downtime_hours && (
                            <div className="flex items-center gap-2">
                              <span className="text-gray-600 dark:text-gray-400">Downtime:</span>
                              <span className="text-gray-900 dark:text-white font-medium">
                                {rec.estimated_downtime_hours} hours
                              </span>
                            </div>
                          )}
                          {rec.cost_impact && (
                            <div className="flex items-center gap-2">
                              <span className="text-gray-600 dark:text-gray-400">Cost:</span>
                              <span className="text-gray-900 dark:text-white font-medium capitalize">
                                {rec.cost_impact}
                              </span>
                            </div>
                          )}
                          {rec.delta_temp_c !== undefined && (
                            <div className="flex items-center gap-2">
                              <span className="text-gray-600 dark:text-gray-400">ΔT:</span>
                              <span className="text-gray-900 dark:text-white font-medium">
                                {rec.delta_temp_c > 0 ? '+' : ''}{rec.delta_temp_c}°C
                              </span>
                            </div>
                          )}
                          {rec.delta_time_min !== undefined && (
                            <div className="flex items-center gap-2">
                              <span className="text-gray-600 dark:text-gray-400">Δt:</span>
                              <span className="text-gray-900 dark:text-white font-medium">
                                {rec.delta_time_min > 0 ? '+' : ''}{rec.delta_time_min} min
                              </span>
                            </div>
                          )}
                          {rec.zones && (
                            <div className="flex items-center gap-2">
                              <span className="text-gray-600 dark:text-gray-400">Zones:</span>
                              <span className="text-gray-900 dark:text-white font-medium">
                                {rec.zones.join(', ')}
                              </span>
                            </div>
                          )}
                          {rec.runs_since_last_clean && (
                            <div className="flex items-center gap-2">
                              <span className="text-gray-600 dark:text-gray-400">Runs since clean:</span>
                              <span className="text-gray-900 dark:text-white font-medium">
                                {rec.runs_since_last_clean}
                              </span>
                            </div>
                          )}
                        </div>

                        {rec.expected_improvement && (
                          <div className="mt-3 p-2 bg-green-50 dark:bg-green-900/20 rounded">
                            <p className="text-sm text-green-700 dark:text-green-300">
                              <CheckCircle className="w-4 h-4 inline mr-1" />
                              {rec.expected_improvement}
                            </p>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Next Maintenance Window */}
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
                  <h4 className="text-sm font-medium text-blue-900 dark:text-blue-100 mb-2">
                    Next Scheduled Maintenance Window
                  </h4>
                  <p className="text-lg font-medium text-blue-700 dark:text-blue-300">
                    {new Date(results.next_maintenance_window).toLocaleString()}
                  </p>
                </div>
              </>
            ) : (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-12">
                <div className="text-center">
                  <div className="inline-flex p-4 bg-orange-100 dark:bg-orange-900/30 rounded-full mb-4">
                    <Wrench className="w-8 h-8 text-orange-600 dark:text-orange-400" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                    No Analysis Results
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    Configure FDC parameters and click "Run Analysis"
                  </p>
                  <div className="inline-block bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 text-left">
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Key Features:
                    </p>
                    <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                      <li>• Tool health scoring (0-1) from 29 FDC features</li>
                      <li>• Failure probability forecasting</li>
                      <li>• Actionable maintenance recommendations</li>
                      <li>• Recipe trim suggestions (ΔT, Δt)</li>
                      <li>• Component-level diagnostics</li>
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
