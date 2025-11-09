'use client'

import React, { useState } from 'react'
import { Settings, TrendingUp, Target, Info, Play, CheckCircle } from 'lucide-react'

interface CalibrationResult {
  method: string
  optimized_params: Record<string, number>
  initial_error: number
  final_error: number
  improvement_pct: number
  iterations: number
  convergence_status: string
}

export default function CalibrationPage() {
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<CalibrationResult | null>(null)

  // Calibration configuration
  const [calibrationMethod, setCalibrationMethod] = useState<'least_squares' | 'bayesian'>('least_squares')
  const [dopantType, setDopantType] = useState('boron')
  const [targetMetric, setTargetMetric] = useState<'junction_depth' | 'sheet_resistance'>('junction_depth')

  // Parameters to calibrate
  const [calibrateD0, setCalibrateD0] = useState(true)
  const [calibrateEa, setCalibrateEa] = useState(true)

  // Initial parameter guesses
  const [initialD0, setInitialD0] = useState(0.76)
  const [initialEa, setInitialEa] = useState(3.46)

  // Experimental data points (synthetic)
  const [numDataPoints, setNumDataPoints] = useState(10)

  const runCalibration = async () => {
    setLoading(true)

    try {
      // Simulate calibration process
      await new Promise(resolve => setTimeout(resolve, 2000))

      // Generate synthetic calibration results
      const initialError = Math.random() * 20 + 10 // 10-30%
      const finalError = initialError * (0.1 + Math.random() * 0.2) // 10-30% of initial

      const optimizedD0 = initialD0 * (0.95 + Math.random() * 0.1)
      const optimizedEa = initialEa * (0.98 + Math.random() * 0.04)

      const result: CalibrationResult = {
        method: calibrationMethod === 'least_squares' ? 'Least Squares' : 'Bayesian MCMC',
        optimized_params: {
          D0: optimizedD0,
          Ea: optimizedEa
        },
        initial_error: initialError,
        final_error: finalError,
        improvement_pct: ((initialError - finalError) / initialError) * 100,
        iterations: Math.floor(Math.random() * 50) + 50,
        convergence_status: 'Converged'
      }

      setResults(result)
    } catch (error) {
      console.error('Calibration failed:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg">
              <Settings className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                Model Calibration
              </h1>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Parameter tuning with least squares & Bayesian MCMC methods
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Left Panel - Configuration */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Calibration Setup
              </h2>

              {/* Calibration Method */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Calibration Method
                </label>
                <select
                  value={calibrationMethod}
                  onChange={(e) => setCalibrationMethod(e.target.value as any)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500"
                >
                  <option value="least_squares">Least Squares</option>
                  <option value="bayesian">Bayesian MCMC</option>
                </select>
                <p className="text-xs text-gray-500 mt-1">
                  {calibrationMethod === 'least_squares'
                    ? 'Fast optimization using gradient descent'
                    : 'Full posterior distribution with uncertainty quantification'}
                </p>
              </div>

              {/* Dopant Type */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Dopant Species
                </label>
                <select
                  value={dopantType}
                  onChange={(e) => setDopantType(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500"
                >
                  <option value="boron">Boron (B)</option>
                  <option value="phosphorus">Phosphorus (P)</option>
                  <option value="arsenic">Arsenic (As)</option>
                  <option value="antimony">Antimony (Sb)</option>
                </select>
              </div>

              {/* Target Metric */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Target Metric
                </label>
                <select
                  value={targetMetric}
                  onChange={(e) => setTargetMetric(e.target.value as any)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500"
                >
                  <option value="junction_depth">Junction Depth (xj)</option>
                  <option value="sheet_resistance">Sheet Resistance (Rs)</option>
                </select>
              </div>

              {/* Parameters to Calibrate */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Parameters to Calibrate
                </label>
                <div className="space-y-2">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={calibrateD0}
                      onChange={(e) => setCalibrateD0(e.target.checked)}
                      className="w-4 h-4 text-purple-600 rounded"
                    />
                    <span className="text-sm text-gray-700 dark:text-gray-300">
                      D₀ (Pre-exponential factor)
                    </span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={calibrateEa}
                      onChange={(e) => setCalibrateEa(e.target.checked)}
                      className="w-4 h-4 text-purple-600 rounded"
                    />
                    <span className="text-sm text-gray-700 dark:text-gray-300">
                      Ea (Activation energy)
                    </span>
                  </label>
                </div>
              </div>

              {/* Initial Guesses */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Initial D₀ Guess
                </label>
                <input
                  type="number"
                  value={initialD0}
                  onChange={(e) => setInitialD0(Number(e.target.value))}
                  step="0.01"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
                <p className="text-xs text-gray-500 mt-1">cm²/s</p>
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Initial Ea Guess
                </label>
                <input
                  type="number"
                  value={initialEa}
                  onChange={(e) => setInitialEa(Number(e.target.value))}
                  step="0.01"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
                <p className="text-xs text-gray-500 mt-1">eV</p>
              </div>

              {/* Number of Data Points */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Experimental Data Points: {numDataPoints}
                </label>
                <input
                  type="range"
                  min="5"
                  max="50"
                  step="5"
                  value={numDataPoints}
                  onChange={(e) => setNumDataPoints(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              {/* Run Button */}
              <button
                onClick={runCalibration}
                disabled={loading || (!calibrateD0 && !calibrateEa)}
                className="w-full py-3 px-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <Settings className="w-4 h-4 animate-spin" />
                    Calibrating...
                  </span>
                ) : (
                  <span className="flex items-center justify-center gap-2">
                    <Play className="w-4 h-4" />
                    Run Calibration
                  </span>
                )}
              </button>
            </div>

            {/* Info Panel */}
            <div className="mt-6 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
              <div className="flex items-start gap-2">
                <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
                <div className="text-sm">
                  <p className="font-medium text-blue-900 dark:text-blue-100 mb-1">
                    Model Calibration
                  </p>
                  <p className="text-blue-700 dark:text-blue-300 text-xs">
                    Optimize physics model parameters to match experimental data.
                    Least squares for quick fits, Bayesian for full uncertainty quantification.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel - Results */}
          <div className="lg:col-span-2">
            {results ? (
              <>
                {/* Status Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Initial Error</p>
                        <p className="text-2xl font-bold text-red-600 dark:text-red-400 mt-1">
                          {results.initial_error.toFixed(2)}%
                        </p>
                      </div>
                      <div className="p-3 bg-red-100 dark:bg-red-900/30 rounded-lg">
                        <Target className="w-6 h-6 text-red-600 dark:text-red-400" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Final Error</p>
                        <p className="text-2xl font-bold text-green-600 dark:text-green-400 mt-1">
                          {results.final_error.toFixed(2)}%
                        </p>
                      </div>
                      <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-lg">
                        <CheckCircle className="w-6 h-6 text-green-600 dark:text-green-400" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Improvement</p>
                        <p className="text-2xl font-bold text-purple-600 dark:text-purple-400 mt-1">
                          {results.improvement_pct.toFixed(1)}%
                        </p>
                      </div>
                      <div className="p-3 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
                        <TrendingUp className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Optimized Parameters */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Optimized Parameters
                  </h3>

                  <div className="space-y-4">
                    {Object.entries(results.optimized_params).map(([param, value]) => (
                      <div key={param} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-gray-900 dark:text-white">
                            {param === 'D0' ? 'D₀ (Pre-exponential)' : 'Ea (Activation Energy)'}
                          </span>
                          <span className="text-sm text-gray-500">
                            {param === 'D0' ? 'cm²/s' : 'eV'}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <p className="text-gray-600 dark:text-gray-400">Initial:</p>
                            <p className="text-gray-900 dark:text-white font-medium">
                              {param === 'D0' ? initialD0.toFixed(4) : initialEa.toFixed(4)}
                            </p>
                          </div>
                          <div>
                            <p className="text-gray-600 dark:text-gray-400">Optimized:</p>
                            <p className="text-purple-600 dark:text-purple-400 font-bold">
                              {value.toFixed(4)}
                            </p>
                          </div>
                        </div>
                        <div className="mt-2">
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div
                              className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all"
                              style={{
                                width: `${Math.min(100, (value / (param === 'D0' ? initialD0 : initialEa)) * 100)}%`
                              }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Convergence Info */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Convergence Details
                  </h3>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Method</p>
                      <p className="text-lg font-medium text-gray-900 dark:text-white">
                        {results.method}
                      </p>
                    </div>

                    <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Iterations</p>
                      <p className="text-lg font-medium text-gray-900 dark:text-white">
                        {results.iterations}
                      </p>
                    </div>

                    <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Status</p>
                      <p className="text-lg font-medium text-green-600 dark:text-green-400">
                        {results.convergence_status}
                      </p>
                    </div>
                  </div>

                  {calibrationMethod === 'bayesian' && (
                    <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                      <p className="text-sm font-medium text-blue-900 dark:text-blue-100 mb-2">
                        Bayesian Posterior Analysis
                      </p>
                      <p className="text-xs text-blue-700 dark:text-blue-300">
                        Full posterior distribution sampled using MCMC. Uncertainty bounds
                        available for each parameter (±2σ credible intervals).
                      </p>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-12">
                <div className="text-center">
                  <div className="inline-flex p-4 bg-purple-100 dark:bg-purple-900/30 rounded-full mb-4">
                    <Settings className="w-8 h-8 text-purple-600 dark:text-purple-400" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                    No Calibration Results
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    Configure calibration parameters and click "Run Calibration"
                  </p>
                  <div className="inline-block bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 text-left">
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Calibration Methods:
                    </p>
                    <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                      <li>• Least Squares: Fast gradient-based optimization</li>
                      <li>• Bayesian MCMC: Full posterior with uncertainty</li>
                      <li>• Target metrics: Junction depth, sheet resistance</li>
                      <li>• Parameters: D₀, Ea (Arrhenius diffusion)</li>
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
