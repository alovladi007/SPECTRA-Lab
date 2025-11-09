'use client'

import React, { useState } from 'react'
import { Layers, Clock, CheckCircle, AlertCircle, Play, Trash2, Download } from 'lucide-react'

interface BatchJob {
  id: string
  type: 'diffusion' | 'oxidation'
  status: 'queued' | 'running' | 'completed' | 'failed'
  parameters: Record<string, any>
  result?: any
  created_at: string
  completed_at?: string
  duration_ms?: number
}

export default function BatchJobManagerPage() {
  const [jobs, setJobs] = useState<BatchJob[]>([])
  const [batchType, setBatchType] = useState<'diffusion' | 'oxidation'>('diffusion')
  const [numJobs, setNumJobs] = useState(5)

  // Diffusion parameters
  const [tempStart, setTempStart] = useState(900)
  const [tempEnd, setTempEnd] = useState(1100)
  const [dopant, setDopant] = useState('boron')

  // Oxidation parameters
  const [ambient, setAmbient] = useState<'dry' | 'wet'>('dry')
  const [timeStart, setTimeStart] = useState(30)
  const [timeEnd, setTimeEnd] = useState(180)

  const generateBatchJobs = () => {
    const newJobs: BatchJob[] = []
    const now = new Date()

    for (let i = 0; i < numJobs; i++) {
      if (batchType === 'diffusion') {
        const temp = tempStart + ((tempEnd - tempStart) * i) / (numJobs - 1)
        newJobs.push({
          id: `job-${Date.now()}-${i}`,
          type: 'diffusion',
          status: 'queued',
          parameters: {
            temperature: Math.round(temp),
            time: 30,
            dopant,
            initial_concentration: 1e20,
            depth: 1000,
            grid_points: 100,
            model: 'erfc'
          },
          created_at: now.toISOString()
        })
      } else {
        const time = timeStart + ((timeEnd - timeStart) * i) / (numJobs - 1)
        newJobs.push({
          id: `job-${Date.now()}-${i}`,
          type: 'oxidation',
          status: 'queued',
          parameters: {
            temperature: 1000,
            time: Math.round(time),
            ambient,
            pressure: 1.0,
            initial_oxide_thickness: 0
          },
          created_at: now.toISOString()
        })
      }
    }

    setJobs(newJobs)
  }

  const runBatchJobs = async () => {
    for (let i = 0; i < jobs.length; i++) {
      const job = jobs[i]

      // Update status to running
      setJobs(prev => prev.map(j =>
        j.id === job.id ? { ...j, status: 'running' as const } : j
      ))

      try {
        const endpoint = job.type === 'diffusion'
          ? 'http://localhost:8001/api/v1/simulation/diffusion'
          : 'http://localhost:8001/api/v1/simulation/oxidation'

        const startTime = Date.now()
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(job.parameters)
        })

        const result = await response.json()
        const duration = Date.now() - startTime

        // Update job with results
        setJobs(prev => prev.map(j =>
          j.id === job.id
            ? {
                ...j,
                status: 'completed' as const,
                result,
                completed_at: new Date().toISOString(),
                duration_ms: duration
              }
            : j
        ))
      } catch (error) {
        console.error(`Job ${job.id} failed:`, error)
        setJobs(prev => prev.map(j =>
          j.id === job.id ? { ...j, status: 'failed' as const } : j
        ))
      }

      // Small delay between jobs
      await new Promise(resolve => setTimeout(resolve, 500))
    }
  }

  const clearJobs = () => {
    setJobs([])
  }

  const exportResults = () => {
    const results = jobs.filter(j => j.status === 'completed').map(j => ({
      id: j.id,
      type: j.type,
      parameters: j.parameters,
      result: j.result,
      duration_ms: j.duration_ms
    }))

    const dataStr = JSON.stringify(results, null, 2)
    const blob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `batch-results-${Date.now()}.json`
    link.click()
  }

  const getStatusColor = (status: BatchJob['status']) => {
    switch (status) {
      case 'queued': return 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
      case 'running': return 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
      case 'completed': return 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
      case 'failed': return 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
    }
  }

  const getStatusIcon = (status: BatchJob['status']) => {
    switch (status) {
      case 'queued': return <Clock className="w-4 h-4" />
      case 'running': return <Play className="w-4 h-4 animate-pulse" />
      case 'completed': return <CheckCircle className="w-4 h-4" />
      case 'failed': return <AlertCircle className="w-4 h-4" />
    }
  }

  const completedCount = jobs.filter(j => j.status === 'completed').length
  const runningCount = jobs.filter(j => j.status === 'running').length
  const queuedCount = jobs.filter(j => j.status === 'queued').length
  const failedCount = jobs.filter(j => j.status === 'failed').length

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg">
              <Layers className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                Batch Job Manager
              </h1>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Submit and monitor multiple simulations with parameter sweeps
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
                Batch Configuration
              </h2>

              {/* Batch Type */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Simulation Type
                </label>
                <select
                  value={batchType}
                  onChange={(e) => setBatchType(e.target.value as any)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  <option value="diffusion">Diffusion</option>
                  <option value="oxidation">Oxidation</option>
                </select>
              </div>

              {/* Number of Jobs */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Number of Jobs: {numJobs}
                </label>
                <input
                  type="range"
                  min="2"
                  max="20"
                  step="1"
                  value={numJobs}
                  onChange={(e) => setNumJobs(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              {/* Type-specific parameters */}
              {batchType === 'diffusion' ? (
                <>
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Dopant
                    </label>
                    <select
                      value={dopant}
                      onChange={(e) => setDopant(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    >
                      <option value="boron">Boron (B)</option>
                      <option value="phosphorus">Phosphorus (P)</option>
                      <option value="arsenic">Arsenic (As)</option>
                      <option value="antimony">Antimony (Sb)</option>
                    </select>
                  </div>

                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Temperature Start: {tempStart}°C
                    </label>
                    <input
                      type="range"
                      min="800"
                      max="1200"
                      step="10"
                      value={tempStart}
                      onChange={(e) => setTempStart(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>

                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Temperature End: {tempEnd}°C
                    </label>
                    <input
                      type="range"
                      min="800"
                      max="1200"
                      step="10"
                      value={tempEnd}
                      onChange={(e) => setTempEnd(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </>
              ) : (
                <>
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Ambient
                    </label>
                    <select
                      value={ambient}
                      onChange={(e) => setAmbient(e.target.value as any)}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    >
                      <option value="dry">Dry O₂</option>
                      <option value="wet">Wet O₂</option>
                    </select>
                  </div>

                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Time Start: {timeStart} min
                    </label>
                    <input
                      type="range"
                      min="10"
                      max="600"
                      step="10"
                      value={timeStart}
                      onChange={(e) => setTimeStart(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>

                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Time End: {timeEnd} min
                    </label>
                    <input
                      type="range"
                      min="10"
                      max="600"
                      step="10"
                      value={timeEnd}
                      onChange={(e) => setTimeEnd(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </>
              )}

              {/* Buttons */}
              <div className="space-y-2">
                <button
                  onClick={generateBatchJobs}
                  className="w-full py-2 px-4 bg-indigo-600 text-white font-medium rounded-lg hover:bg-indigo-700 transition-colors"
                >
                  Generate Jobs
                </button>

                <button
                  onClick={runBatchJobs}
                  disabled={jobs.length === 0 || runningCount > 0}
                  className="w-full py-2 px-4 bg-gradient-to-r from-indigo-500 to-purple-500 text-white font-medium rounded-lg hover:from-indigo-600 hover:to-purple-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  <span className="flex items-center justify-center gap-2">
                    <Play className="w-4 h-4" />
                    Run All Jobs
                  </span>
                </button>

                <button
                  onClick={exportResults}
                  disabled={completedCount === 0}
                  className="w-full py-2 px-4 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <span className="flex items-center justify-center gap-2">
                    <Download className="w-4 h-4" />
                    Export Results
                  </span>
                </button>

                <button
                  onClick={clearJobs}
                  disabled={jobs.length === 0}
                  className="w-full py-2 px-4 bg-red-600 text-white font-medium rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <span className="flex items-center justify-center gap-2">
                    <Trash2 className="w-4 h-4" />
                    Clear All
                  </span>
                </button>
              </div>
            </div>
          </div>

          {/* Right Panel - Job Queue */}
          <div className="lg:col-span-3">
            {/* Status Cards */}
            {jobs.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Queued</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">{queuedCount}</p>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Running</p>
                  <p className="text-2xl font-bold text-blue-600 dark:text-blue-400 mt-1">{runningCount}</p>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Completed</p>
                  <p className="text-2xl font-bold text-green-600 dark:text-green-400 mt-1">{completedCount}</p>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Failed</p>
                  <p className="text-2xl font-bold text-red-600 dark:text-red-400 mt-1">{failedCount}</p>
                </div>
              </div>
            )}

            {/* Job List */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Job Queue ({jobs.length} jobs)
                </h3>
              </div>

              <div className="divide-y divide-gray-200 dark:divide-gray-700">
                {jobs.length > 0 ? (
                  jobs.map((job) => (
                    <div key={job.id} className="p-4">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-medium text-gray-900 dark:text-white">
                              {job.type === 'diffusion' ? 'Diffusion' : 'Oxidation'} Simulation
                            </span>
                            <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(job.status)}`}>
                              {getStatusIcon(job.status)}
                              {job.status}
                            </span>
                          </div>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            ID: {job.id}
                          </p>
                        </div>
                        {job.duration_ms && (
                          <div className="text-right">
                            <p className="text-sm text-gray-600 dark:text-gray-400">Duration</p>
                            <p className="text-sm font-medium text-gray-900 dark:text-white">
                              {job.duration_ms}ms
                            </p>
                          </div>
                        )}
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                        {Object.entries(job.parameters).slice(0, 4).map(([key, value]) => (
                          <div key={key} className="bg-gray-50 dark:bg-gray-700/50 rounded px-2 py-1">
                            <span className="text-gray-600 dark:text-gray-400">{key}:</span>
                            <span className="ml-1 text-gray-900 dark:text-white font-medium">
                              {typeof value === 'number' && value > 1000 ? value.toExponential(1) : String(value)}
                            </span>
                          </div>
                        ))}
                      </div>

                      {job.status === 'completed' && job.result && (
                        <div className="mt-2 p-2 bg-green-50 dark:bg-green-900/20 rounded">
                          <p className="text-sm font-medium text-green-900 dark:text-green-100">
                            Result: {job.type === 'diffusion'
                              ? `Junction Depth = ${job.result.junction_depth?.toFixed(1)} nm`
                              : `Final Thickness = ${job.result.final_thickness?.toFixed(1)} nm`
                            }
                          </p>
                        </div>
                      )}
                    </div>
                  ))
                ) : (
                  <div className="p-12 text-center">
                    <Layers className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600 dark:text-gray-400 mb-2">No jobs in queue</p>
                    <p className="text-sm text-gray-500">
                      Configure parameters and click "Generate Jobs" to start
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
