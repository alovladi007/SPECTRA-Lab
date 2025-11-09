'use client'

import React, { useState } from 'react'
import { Layers, TrendingUp, Clock, Info, ArrowRight } from 'lucide-react'
import { OxidationGrowthChart } from '@/components/charts/OxidationGrowthChart'

export default function OxidationPlannerPage() {
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<any>(null)
  const [mode, setMode] = useState<'forward' | 'inverse'>('forward')

  // Forward solver parameters
  const [temperature, setTemperature] = useState(1000)
  const [time, setTime] = useState(120)
  const [ambient, setAmbient] = useState<'dry' | 'wet'>('dry')
  const [pressure, setPressure] = useState(1.0)
  const [initialThickness, setInitialThickness] = useState(0)

  // Inverse solver parameters
  const [targetThickness, setTargetThickness] = useState(50)

  const runForwardSolver = async () => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8001/api/v1/simulation/oxidation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          temperature,
          time,
          ambient,
          pressure,
          initial_oxide_thickness: initialThickness
        })
      })

      const data = await response.json()
      setResults({ ...data, mode: 'forward' })
    } catch (error) {
      console.error('Forward solver failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const runInverseSolver = async () => {
    setLoading(true)
    try {
      // For inverse solver, we'll estimate time needed
      // Start with a reasonable guess and iterate
      let estimatedTime = 60
      let bestResult = null
      let bestDiff = Infinity

      // Binary search for the right time (simplified approach)
      for (let testTime = 10; testTime <= 600; testTime += 10) {
        const response = await fetch('http://localhost:8001/api/v1/simulation/oxidation', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            temperature,
            time: testTime,
            ambient,
            pressure,
            initial_oxide_thickness: initialThickness
          })
        })

        const data = await response.json()
        const diff = Math.abs(data.final_thickness - targetThickness)

        if (diff < bestDiff) {
          bestDiff = diff
          bestResult = { ...data, mode: 'inverse', targetThickness, estimatedTime: testTime }
          estimatedTime = testTime
        }

        if (diff < 0.5) break // Close enough
      }

      setResults(bestResult)
    } catch (error) {
      console.error('Inverse solver failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSimulate = () => {
    if (mode === 'forward') {
      runForwardSolver()
    } else {
      runInverseSolver()
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg">
              <Layers className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                Oxidation Planning
              </h1>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Deal-Grove oxide thickness calculator with forward/inverse solvers
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Left Panel - Parameters */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Simulation Parameters
              </h2>

              {/* Solver Mode */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Solver Mode
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => setMode('forward')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                      mode === 'forward'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                    }`}
                  >
                    Forward
                  </button>
                  <button
                    onClick={() => setMode('inverse')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                      mode === 'inverse'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                    }`}
                  >
                    Inverse
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  {mode === 'forward'
                    ? 'Calculate thickness from time'
                    : 'Calculate time needed for target thickness'}
                </p>
              </div>

              {/* Ambient Type */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Ambient Type
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => setAmbient('dry')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                      ambient === 'dry'
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                    }`}
                  >
                    Dry O₂
                  </button>
                  <button
                    onClick={() => setAmbient('wet')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                      ambient === 'wet'
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                    }`}
                  >
                    Wet O₂
                  </button>
                </div>
              </div>

              {/* Temperature */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Temperature: {temperature}°C
                </label>
                <input
                  type="range"
                  min="700"
                  max="1200"
                  step="10"
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>700°C</span>
                  <span>1200°C</span>
                </div>
              </div>

              {/* Time (Forward mode only) */}
              {mode === 'forward' && (
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Oxidation Time: {time} min
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="600"
                    step="1"
                    value={time}
                    onChange={(e) => setTime(Number(e.target.value))}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>1 min</span>
                    <span>600 min</span>
                  </div>
                </div>
              )}

              {/* Target Thickness (Inverse mode only) */}
              {mode === 'inverse' && (
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Target Thickness
                  </label>
                  <input
                    type="number"
                    value={targetThickness}
                    onChange={(e) => setTargetThickness(Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    step="1"
                  />
                  <p className="text-xs text-gray-500 mt-1">nanometers (nm)</p>
                </div>
              )}

              {/* Pressure */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Pressure: {pressure.toFixed(2)} atm
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="5.0"
                  step="0.1"
                  value={pressure}
                  onChange={(e) => setPressure(Number(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0.1 atm</span>
                  <span>5.0 atm</span>
                </div>
              </div>

              {/* Initial Thickness */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Initial Oxide Thickness
                </label>
                <input
                  type="number"
                  value={initialThickness}
                  onChange={(e) => setInitialThickness(Number(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  step="1"
                />
                <p className="text-xs text-gray-500 mt-1">nanometers (nm)</p>
              </div>

              {/* Run Button */}
              <button
                onClick={handleSimulate}
                disabled={loading}
                className="w-full py-3 px-4 bg-gradient-to-r from-blue-500 to-cyan-500 text-white font-medium rounded-lg hover:from-blue-600 hover:to-cyan-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <Clock className="w-4 h-4 animate-spin" />
                    Calculating...
                  </span>
                ) : (
                  <span className="flex items-center justify-center gap-2">
                    {mode === 'forward' ? 'Calculate Thickness' : 'Calculate Time'}
                    <ArrowRight className="w-4 h-4" />
                  </span>
                )}
              </button>
            </div>

            {/* Deal-Grove Info */}
            <div className="mt-6 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
              <div className="flex items-start gap-2">
                <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
                <div className="text-sm">
                  <p className="font-medium text-blue-900 dark:text-blue-100 mb-1">
                    Deal-Grove Model
                  </p>
                  <p className="text-blue-700 dark:text-blue-300 text-xs">
                    Linear-parabolic model for thermal oxidation of silicon.
                    Accounts for both surface reaction and diffusion-limited regimes.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel - Results */}
          <div className="lg:col-span-2">
            {results ? (
              <>
                {/* Results Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {results.mode === 'forward' ? 'Final Thickness' : 'Target Thickness'}
                        </p>
                        <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                          {results.final_thickness?.toFixed(1) || targetThickness.toFixed(1)}
                          <span className="text-sm font-normal text-gray-500 ml-1">nm</span>
                        </p>
                      </div>
                      <div className="p-3 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                        <Layers className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {results.mode === 'forward' ? 'Oxidation Time' : 'Required Time'}
                        </p>
                        <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                          {results.mode === 'forward' ? time : results.estimatedTime}
                          <span className="text-sm font-normal text-gray-500 ml-1">min</span>
                        </p>
                      </div>
                      <div className="p-3 bg-cyan-100 dark:bg-cyan-900/30 rounded-lg">
                        <Clock className="w-6 h-6 text-cyan-600 dark:text-cyan-400" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Growth Rate</p>
                        <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                          {results.growth_rate?.toFixed(2)}
                          <span className="text-sm font-normal text-gray-500 ml-1">nm/min</span>
                        </p>
                      </div>
                      <div className="p-3 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
                        <TrendingUp className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Growth Profile */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Oxide Growth Profile
                  </h3>
                  <OxidationGrowthChart
                    time={results.time_points}
                    thickness={results.thickness_profile}
                    mode={results.mode}
                  />

                  {/* Deal-Grove Parameters */}
                  <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                    <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Deal-Grove Parameters
                    </h4>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">B (linear):</span>
                        <span className="ml-2 text-gray-900 dark:text-white font-medium">
                          {results.deal_grove_params?.B?.toFixed(3)} nm/min
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">B/A (parabolic):</span>
                        <span className="ml-2 text-gray-900 dark:text-white font-medium">
                          {results.deal_grove_params?.B_over_A?.toFixed(3)} nm²/min
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">τ (offset):</span>
                        <span className="ml-2 text-gray-900 dark:text-white font-medium">
                          {results.deal_grove_params?.tau?.toFixed(1)} min
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Process Summary */}
                  <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                    <h4 className="text-sm font-medium text-blue-900 dark:text-blue-100 mb-2">
                      Process Summary
                    </h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <span className="text-blue-700 dark:text-blue-300">Temperature:</span>
                        <span className="ml-2 text-blue-900 dark:text-blue-100 font-medium">
                          {temperature}°C
                        </span>
                      </div>
                      <div>
                        <span className="text-blue-700 dark:text-blue-300">Ambient:</span>
                        <span className="ml-2 text-blue-900 dark:text-blue-100 font-medium capitalize">
                          {ambient} O₂
                        </span>
                      </div>
                      <div>
                        <span className="text-blue-700 dark:text-blue-300">Pressure:</span>
                        <span className="ml-2 text-blue-900 dark:text-blue-100 font-medium">
                          {pressure.toFixed(2)} atm
                        </span>
                      </div>
                      <div>
                        <span className="text-blue-700 dark:text-blue-300">Initial Oxide:</span>
                        <span className="ml-2 text-blue-900 dark:text-blue-100 font-medium">
                          {initialThickness} nm
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-12">
                <div className="text-center">
                  <div className="inline-flex p-4 bg-blue-100 dark:bg-blue-900/30 rounded-full mb-4">
                    <Layers className="w-8 h-8 text-blue-600 dark:text-blue-400" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                    No Simulation Results
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    Configure parameters and click "Calculate" to see results
                  </p>
                  <div className="inline-block bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 text-left">
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Quick Start Examples:
                    </p>
                    <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                      <li>• Dry oxidation: 1000°C, 120 min → ~50 nm</li>
                      <li>• Wet oxidation: 1000°C, 60 min → ~200 nm</li>
                      <li>• Inverse: Target 100 nm dry @ 1000°C → ~240 min</li>
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
