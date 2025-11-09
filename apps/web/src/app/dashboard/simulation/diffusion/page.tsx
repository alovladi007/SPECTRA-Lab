'use client'

import React, { useState } from 'react'
import { Activity, Zap, TrendingUp, Info } from 'lucide-react'
import { DiffusionProfileChart } from '@/components/charts/DiffusionProfileChart'

const DOPANTS = [
  { value: 'boron', label: 'Boron (B)', D0: 0.76, Ea: 3.46 },
  { value: 'phosphorus', label: 'Phosphorus (P)', D0: 3.85, Ea: 3.66 },
  { value: 'arsenic', label: 'Arsenic (As)', D0: 0.066, Ea: 3.44 },
  { value: 'antimony', label: 'Antimony (Sb)', D0: 0.214, Ea: 3.65 },
]

export default function DiffusionSimulationPage() {
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<any>(null)

  // Simulation parameters
  const [dopant, setDopant] = useState('boron')
  const [temperature, setTemperature] = useState(1000)
  const [time, setTime] = useState(30)
  const [surfaceConc, setSurfaceConc] = useState(1e20)
  const [gridPoints, setGridPoints] = useState(100)
  const [depth, setDepth] = useState(1000)

  const runSimulation = async () => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8001/api/v1/simulation/diffusion', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          temperature,
          time,
          dopant,
          initial_concentration: surfaceConc,
          depth,
          grid_points: gridPoints,
          model: 'erfc'
        })
      })

      const data = await response.json()
      setResults(data)
    } catch (error) {
      console.error('Simulation failed:', error)
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
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                Diffusion Simulation
              </h1>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                ERFC analytical solution for dopant diffusion profiles
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

              {/* Dopant Selection */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Dopant Species
                </label>
                <select
                  value={dopant}
                  onChange={(e) => setDopant(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500"
                >
                  {DOPANTS.map(d => (
                    <option key={d.value} value={d.value}>{d.label}</option>
                  ))}
                </select>
              </div>

              {/* Temperature */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Temperature: {temperature}°C
                </label>
                <input
                  type="range"
                  min="800"
                  max="1200"
                  step="10"
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>800°C</span>
                  <span>1200°C</span>
                </div>
              </div>

              {/* Time */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Time: {time} min
                </label>
                <input
                  type="range"
                  min="1"
                  max="180"
                  step="1"
                  value={time}
                  onChange={(e) => setTime(Number(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>1 min</span>
                  <span>180 min</span>
                </div>
              </div>

              {/* Surface Concentration */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Surface Concentration
                </label>
                <input
                  type="number"
                  value={surfaceConc}
                  onChange={(e) => setSurfaceConc(Number(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  step="1e18"
                />
                <p className="text-xs text-gray-500 mt-1">atoms/cm³</p>
              </div>

              {/* Run Button */}
              <button
                onClick={runSimulation}
                disabled={loading}
                className="w-full py-3 px-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <Activity className="w-4 h-4 animate-spin" />
                    Running Simulation...
                  </span>
                ) : (
                  'Run Simulation'
                )}
              </button>
            </div>

            {/* Physics Info */}
            <div className="mt-6 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
              <div className="flex items-start gap-2">
                <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
                <div className="text-sm">
                  <p className="font-medium text-blue-900 dark:text-blue-100 mb-1">
                    ERFC Solution
                  </p>
                  <p className="text-blue-700 dark:text-blue-300 text-xs">
                    Analytical solution for constant-source diffusion using complementary error function.
                    Assumes constant temperature and surface concentration.
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
                        <p className="text-sm text-gray-600 dark:text-gray-400">Junction Depth</p>
                        <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                          {results.junction_depth.toFixed(1)}
                          <span className="text-sm font-normal text-gray-500 ml-1">nm</span>
                        </p>
                      </div>
                      <div className="p-3 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
                        <TrendingUp className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Sheet Resistance</p>
                        <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                          {results.sheet_resistance.toFixed(1)}
                          <span className="text-sm font-normal text-gray-500 ml-1">Ω/sq</span>
                        </p>
                      </div>
                      <div className="p-3 bg-pink-100 dark:bg-pink-900/30 rounded-lg">
                        <Zap className="w-6 h-6 text-pink-600 dark:text-pink-400" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Status</p>
                        <p className="text-2xl font-bold text-green-600 dark:text-green-400 mt-1">
                          {results.status}
                        </p>
                      </div>
                      <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-lg">
                        <Activity className="w-6 h-6 text-green-600 dark:text-green-400" />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Profile Visualization Placeholder */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Concentration Profile
                  </h3>
                  <DiffusionProfileChart
                    depth={results.profile.depth}
                    concentration={results.profile.concentration}
                  />

                  {/* Metadata */}
                  <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                    <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Simulation Details
                    </h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Model:</span>
                        <span className="ml-2 text-gray-900 dark:text-white font-medium">
                          {results.metadata.model}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Temperature:</span>
                        <span className="ml-2 text-gray-900 dark:text-white font-medium">
                          {results.metadata.temperature}°C
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Time:</span>
                        <span className="ml-2 text-gray-900 dark:text-white font-medium">
                          {results.metadata.time} min
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Dopant:</span>
                        <span className="ml-2 text-gray-900 dark:text-white font-medium capitalize">
                          {results.metadata.dopant}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-12">
                <div className="text-center">
                  <div className="inline-flex p-4 bg-purple-100 dark:bg-purple-900/30 rounded-full mb-4">
                    <Zap className="w-8 h-8 text-purple-600 dark:text-purple-400" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                    No Simulation Results
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    Configure parameters and click "Run Simulation" to see results
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
