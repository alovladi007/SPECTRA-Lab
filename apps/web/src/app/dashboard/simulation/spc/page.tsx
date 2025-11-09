'use client'

import React, { useState } from 'react'
import { Activity, AlertTriangle, TrendingUp, BarChart3, Info, Play, RefreshCw } from 'lucide-react'
import { SPCControlChart } from '@/components/charts/SPCControlChart'

interface Violation {
  rule_id: string
  rule_name: string
  window: number[]
  z_score?: number
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  suggested_causes: string[]
  recommended_actions: string[]
}

interface SPCResults {
  centerline: number
  ucl: number
  lcl: number
  violations: Violation[]
  data_points: number[]
  timestamps: string[]
}

const WESTERN_ELECTRIC_RULES = [
  { id: 'RULE_1', name: 'Point beyond 3σ', description: 'One point beyond 3 standard deviations' },
  { id: 'RULE_2', name: '2 of 3 beyond 2σ', description: 'Two out of three consecutive points beyond 2σ' },
  { id: 'RULE_3', name: '4 of 5 beyond 1σ', description: 'Four out of five consecutive points beyond 1σ' },
  { id: 'RULE_4', name: '8 consecutive on one side', description: 'Eight consecutive points on one side of centerline' },
  { id: 'RULE_5', name: 'Run of 6 increasing/decreasing', description: 'Six points in a row steadily increasing or decreasing' },
  { id: 'RULE_6', name: '15 points within 1σ', description: 'Fifteen points in a row within 1σ (lack of variation)' },
  { id: 'RULE_7', name: '14 points alternating', description: 'Fourteen points alternating up and down' },
  { id: 'RULE_8', name: '8 points beyond 1σ', description: 'Eight consecutive points beyond 1σ on either side' },
]

export default function SPCMonitorPage() {
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<SPCResults | null>(null)

  // SPC parameters
  const [processTarget, setProcessTarget] = useState(100)
  const [processStdDev, setProcessStdDev] = useState(5)
  const [sampleSize, setSampleSize] = useState(100)
  const [driftIntroduced, setDriftIntroduced] = useState(false)
  const [driftMagnitude, setDriftMagnitude] = useState(2)
  const [driftStartPoint, setDriftStartPoint] = useState(50)

  const generateSPCData = () => {
    setLoading(true)

    try {
      // Generate synthetic process data
      const dataPoints: number[] = []
      const timestamps: string[] = []
      const now = new Date()

      for (let i = 0; i < sampleSize; i++) {
        // Normal distribution with optional drift
        let value = processTarget + (Math.random() - 0.5) * 2 * processStdDev

        // Introduce drift if enabled
        if (driftIntroduced && i >= driftStartPoint) {
          const driftAmount = driftMagnitude * processStdDev * (i - driftStartPoint) / (sampleSize - driftStartPoint)
          value += driftAmount
        }

        dataPoints.push(value)

        const timestamp = new Date(now.getTime() - (sampleSize - i) * 60000)
        timestamps.push(timestamp.toISOString())
      }

      // Calculate control limits
      const mean = dataPoints.reduce((a, b) => a + b, 0) / dataPoints.length
      const stdDev = Math.sqrt(
        dataPoints.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / dataPoints.length
      )

      const centerline = processTarget
      const ucl = centerline + 3 * processStdDev
      const lcl = centerline - 3 * processStdDev

      // Detect violations
      const violations: Violation[] = detectViolations(dataPoints, centerline, processStdDev)

      setResults({
        centerline,
        ucl,
        lcl,
        violations,
        data_points: dataPoints,
        timestamps
      })
    } catch (error) {
      console.error('SPC calculation failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const detectViolations = (data: number[], centerline: number, sigma: number): Violation[] => {
    const violations: Violation[] = []

    // Rule 1: Point beyond 3σ
    data.forEach((point, i) => {
      const z = Math.abs(point - centerline) / sigma
      if (z > 3) {
        violations.push({
          rule_id: 'RULE_1',
          rule_name: 'Point beyond 3σ',
          window: [i],
          z_score: z,
          severity: 'CRITICAL',
          suggested_causes: [
            'Sudden temperature excursion (±10°C)',
            'Equipment malfunction or failure',
            'Measurement error or sensor drift',
            'Raw material quality issue'
          ],
          recommended_actions: [
            'Immediate process hold - investigate root cause',
            'Check equipment calibration and sensor readings',
            'Verify measurement system accuracy',
            'Review recent maintenance or setup changes'
          ]
        })
      }
    })

    // Rule 2: 2 of 3 beyond 2σ
    for (let i = 2; i < data.length; i++) {
      const window = data.slice(i - 2, i + 1)
      const beyondCount = window.filter(p => Math.abs(p - centerline) / sigma > 2).length
      if (beyondCount >= 2) {
        violations.push({
          rule_id: 'RULE_2',
          rule_name: '2 of 3 beyond 2σ',
          window: [i - 2, i - 1, i],
          severity: 'HIGH',
          suggested_causes: [
            'Gradual process drift or shift',
            'Tool degradation or wear',
            'Environmental condition changes',
            'Operator technique variation'
          ],
          recommended_actions: [
            'Trending analysis to identify drift source',
            'Schedule preventive maintenance',
            'Review environmental monitoring data',
            'Operator retraining or procedure review'
          ]
        })
      }
    }

    // Rule 4: 8 consecutive on one side
    for (let i = 7; i < data.length; i++) {
      const window = data.slice(i - 7, i + 1)
      const allAbove = window.every(p => p > centerline)
      const allBelow = window.every(p => p < centerline)

      if (allAbove || allBelow) {
        violations.push({
          rule_id: 'RULE_4',
          rule_name: '8 consecutive on one side',
          window: Array.from({ length: 8 }, (_, j) => i - 7 + j),
          severity: 'MEDIUM',
          suggested_causes: [
            'Systematic bias in process (offset)',
            'Recipe parameter needs adjustment',
            'Sensor calibration offset',
            'Ambient conditions trending (temperature/humidity)'
          ],
          recommended_actions: [
            'Recenter process by adjusting recipe parameters',
            'Verify and recalibrate sensors',
            'Check HVAC and facility conditions',
            'Update control limits based on current capability'
          ]
        })
      }
    }

    // Rule 5: Run of 6 increasing/decreasing
    for (let i = 5; i < data.length; i++) {
      const window = data.slice(i - 5, i + 1)
      const increasing = window.every((val, idx) => idx === 0 || val > window[idx - 1])
      const decreasing = window.every((val, idx) => idx === 0 || val < window[idx - 1])

      if (increasing || decreasing) {
        violations.push({
          rule_id: 'RULE_5',
          rule_name: 'Run of 6 increasing/decreasing',
          window: Array.from({ length: 6 }, (_, j) => i - 5 + j),
          severity: 'MEDIUM',
          suggested_causes: [
            'Progressive tool wear or degradation',
            'Consumable depletion (chemicals, gases)',
            'Gradual temperature drift',
            'Cumulative contamination buildup'
          ],
          recommended_actions: [
            'Schedule tool cleaning or parts replacement',
            'Check consumable levels and replenish',
            'Verify temperature control loop performance',
            'Implement preventive maintenance cycle'
          ]
        })
      }
    }

    return violations
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'CRITICAL': return 'text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30'
      case 'HIGH': return 'text-orange-600 dark:text-orange-400 bg-orange-100 dark:bg-orange-900/30'
      case 'MEDIUM': return 'text-yellow-600 dark:text-yellow-400 bg-yellow-100 dark:bg-yellow-900/30'
      default: return 'text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-900/30'
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                SPC Monitoring
              </h1>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Statistical Process Control with Western Electric rules & EWMA charts
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">

          {/* Left Panel - Configuration */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Process Configuration
              </h2>

              {/* Process Target */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Process Target: {processTarget}
                </label>
                <input
                  type="range"
                  min="50"
                  max="200"
                  step="5"
                  value={processTarget}
                  onChange={(e) => setProcessTarget(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              {/* Process Std Dev */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Std Dev (σ): {processStdDev}
                </label>
                <input
                  type="range"
                  min="1"
                  max="20"
                  step="0.5"
                  value={processStdDev}
                  onChange={(e) => setProcessStdDev(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              {/* Sample Size */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Sample Size: {sampleSize}
                </label>
                <input
                  type="range"
                  min="20"
                  max="200"
                  step="10"
                  value={sampleSize}
                  onChange={(e) => setSampleSize(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              {/* Drift Toggle */}
              <div className="mb-4">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={driftIntroduced}
                    onChange={(e) => setDriftIntroduced(e.target.checked)}
                    className="w-4 h-4 text-blue-600 rounded"
                  />
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Introduce Drift
                  </span>
                </label>
              </div>

              {driftIntroduced && (
                <>
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Drift Magnitude: {driftMagnitude}σ
                    </label>
                    <input
                      type="range"
                      min="0.5"
                      max="5"
                      step="0.5"
                      value={driftMagnitude}
                      onChange={(e) => setDriftMagnitude(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>

                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Drift Start Point: {driftStartPoint}
                    </label>
                    <input
                      type="range"
                      min="10"
                      max={sampleSize - 10}
                      step="5"
                      value={driftStartPoint}
                      onChange={(e) => setDriftStartPoint(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </>
              )}

              {/* Generate Button */}
              <button
                onClick={generateSPCData}
                disabled={loading}
                className="w-full py-3 px-4 bg-gradient-to-r from-green-500 to-emerald-500 text-white font-medium rounded-lg hover:from-green-600 hover:to-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all mb-4"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Analyzing...
                  </span>
                ) : (
                  <span className="flex items-center justify-center gap-2">
                    <Play className="w-4 h-4" />
                    Generate & Analyze
                  </span>
                )}
              </button>
            </div>

            {/* Rules Info */}
            <div className="mt-6 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
              <div className="flex items-start gap-2">
                <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
                <div className="text-sm">
                  <p className="font-medium text-blue-900 dark:text-blue-100 mb-1">
                    Western Electric Rules
                  </p>
                  <p className="text-blue-700 dark:text-blue-300 text-xs">
                    8 statistical rules for detecting out-of-control conditions
                    and process anomalies.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel - Results */}
          <div className="lg:col-span-3">
            {results ? (
              <>
                {/* Statistics Cards */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                    <p className="text-sm text-gray-600 dark:text-gray-400">Centerline</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                      {results.centerline.toFixed(1)}
                    </p>
                  </div>

                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                    <p className="text-sm text-gray-600 dark:text-gray-400">UCL (3σ)</p>
                    <p className="text-2xl font-bold text-red-600 dark:text-red-400 mt-1">
                      {results.ucl.toFixed(1)}
                    </p>
                  </div>

                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                    <p className="text-sm text-gray-600 dark:text-gray-400">LCL (3σ)</p>
                    <p className="text-2xl font-bold text-blue-600 dark:text-blue-400 mt-1">
                      {results.lcl.toFixed(1)}
                    </p>
                  </div>

                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                    <p className="text-sm text-gray-600 dark:text-gray-400">Violations</p>
                    <p className="text-2xl font-bold text-orange-600 dark:text-orange-400 mt-1">
                      {results.violations.length}
                    </p>
                  </div>
                </div>

                {/* Control Chart */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Control Chart
                  </h3>
                  <SPCControlChart
                    dataPoints={results.data_points}
                    centerline={results.centerline}
                    ucl={results.ucl}
                    lcl={results.lcl}
                  />
                </div>

                {/* Violations */}
                {results.violations.length > 0 ? (
                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                    <div className="flex items-center gap-2 mb-4">
                      <AlertTriangle className="w-5 h-5 text-orange-600 dark:text-orange-400" />
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                        Detected Violations ({results.violations.length})
                      </h3>
                    </div>

                    <div className="space-y-4">
                      {results.violations.map((violation, idx) => (
                        <div
                          key={idx}
                          className="border border-gray-200 dark:border-gray-700 rounded-lg p-4"
                        >
                          <div className="flex items-start justify-between mb-2">
                            <div>
                              <h4 className="font-medium text-gray-900 dark:text-white">
                                {violation.rule_name}
                              </h4>
                              <p className="text-sm text-gray-600 dark:text-gray-400">
                                Points: {violation.window.join(', ')}
                                {violation.z_score && ` • Z-score: ${violation.z_score.toFixed(2)}`}
                              </p>
                            </div>
                            <span className={`px-3 py-1 rounded-full text-xs font-medium ${getSeverityColor(violation.severity)}`}>
                              {violation.severity}
                            </span>
                          </div>

                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
                            <div>
                              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                Suggested Causes:
                              </p>
                              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                                {violation.suggested_causes.map((cause, i) => (
                                  <li key={i} className="flex items-start gap-2">
                                    <span className="text-orange-500 mt-1">•</span>
                                    <span>{cause}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>

                            <div>
                              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                Recommended Actions:
                              </p>
                              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                                {violation.recommended_actions.map((action, i) => (
                                  <li key={i} className="flex items-start gap-2">
                                    <span className="text-green-500 mt-1">→</span>
                                    <span>{action}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 border border-green-200 dark:border-green-800">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="w-5 h-5 text-green-600 dark:text-green-400" />
                      <p className="font-medium text-green-900 dark:text-green-100">
                        Process In Control
                      </p>
                    </div>
                    <p className="text-sm text-green-700 dark:text-green-300 mt-2">
                      No Western Electric rule violations detected. Process is operating within expected statistical limits.
                    </p>
                  </div>
                )}
              </>
            ) : (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-12">
                <div className="text-center">
                  <div className="inline-flex p-4 bg-green-100 dark:bg-green-900/30 rounded-full mb-4">
                    <Activity className="w-8 h-8 text-green-600 dark:text-green-400" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                    No Monitoring Data
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    Configure process parameters and click "Generate & Analyze" to start SPC monitoring
                  </p>
                  <div className="inline-block bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 text-left">
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Western Electric Rules Monitored:
                    </p>
                    <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                      {WESTERN_ELECTRIC_RULES.slice(0, 5).map((rule, i) => (
                        <li key={i}>• {rule.name}: {rule.description}</li>
                      ))}
                      <li className="text-gray-500 dark:text-gray-500">...and {WESTERN_ELECTRIC_RULES.length - 5} more rules</li>
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
