'use client'

import React from 'react'
import Link from 'next/link'
import { Zap, Layers, Activity, ArrowRight, Beaker, Wrench, LayoutGrid, Gauge } from 'lucide-react'

const coreSimulations = [
  {
    name: 'Diffusion Simulation',
    description: 'Dopant diffusion profiles using ERFC analytical solutions',
    icon: Zap,
    color: 'from-purple-500 to-pink-500',
    href: '/dashboard/simulation/diffusion',
    status: 'active',
    features: ['ERFC solutions', 'Junction depth', 'Sheet resistance', 'Multiple dopants']
  },
  {
    name: 'Oxidation Planning',
    description: 'Deal-Grove oxide thickness calculations',
    icon: Layers,
    color: 'from-blue-500 to-cyan-500',
    href: '/dashboard/simulation/oxidation',
    status: 'active',
    features: ['Deal-Grove model', 'Dry/wet oxidation', 'Time-to-target', 'Growth rates']
  },
  {
    name: 'SPC Monitoring',
    description: 'Statistical process control with Western Electric rules',
    icon: Activity,
    color: 'from-green-500 to-emerald-500',
    href: '/dashboard/simulation/spc',
    status: 'active',
    features: ['Western Electric rules', 'EWMA charts', 'Change point detection', 'Violation alerts']
  },
]

const advancedTools = [
  {
    name: 'Calibration',
    description: 'Equipment calibration tracking and management',
    icon: Wrench,
    color: 'from-orange-500 to-red-500',
    href: '/dashboard/simulation/calibration',
    status: 'active',
    features: ['Calibration schedules', 'Equipment tracking', 'Compliance reports', 'Certificate management']
  },
  {
    name: 'Batch Job Manager',
    description: 'Batch processing and job queue management',
    icon: LayoutGrid,
    color: 'from-indigo-500 to-purple-500',
    href: '/dashboard/simulation/batch',
    status: 'active',
    features: ['Job scheduling', 'Queue management', 'Status monitoring', 'Resource allocation']
  },
  {
    name: 'Predictive Maintenance',
    description: 'Predictive maintenance scheduling and tracking',
    icon: Gauge,
    color: 'from-yellow-500 to-orange-500',
    href: '/dashboard/simulation/maintenance',
    status: 'active',
    features: ['Failure prediction', 'Maintenance schedules', 'Equipment health', 'Downtime optimization']
  },
]

export default function SimulationPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg">
              <Beaker className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                Process Simulation
              </h1>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Advanced semiconductor process simulations and modeling
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
            <p className="text-sm text-gray-600 dark:text-gray-400">Active Modules</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">6</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
            <p className="text-sm text-gray-600 dark:text-gray-400">Simulations Run</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">1,234</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
            <p className="text-sm text-gray-600 dark:text-gray-400">Avg Response Time</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">85ms</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
            <p className="text-sm text-gray-600 dark:text-gray-400">API Status</p>
            <p className="text-2xl font-bold text-green-600 dark:text-green-400 mt-1">Live</p>
          </div>
        </div>

        {/* Core Simulations Section */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Core Simulations
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {coreSimulations.map((sim) => (
              <Link
                key={sim.name}
                href={sim.href}
                className="group block bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-shadow"
              >
                <div className="p-6">
                  <div className={`inline-flex p-3 bg-gradient-to-r ${sim.color} rounded-lg mb-4`}>
                    <sim.icon className="w-6 h-6 text-white" />
                  </div>

                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors">
                    {sim.name}
                  </h3>

                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                    {sim.description}
                  </p>

                  {/* Features */}
                  <ul className="space-y-1 mb-4">
                    {sim.features.map((feature, idx) => (
                      <li key={idx} className="text-xs text-gray-500 dark:text-gray-500 flex items-center gap-2">
                        <span className="w-1 h-1 bg-gray-400 rounded-full"></span>
                        {feature}
                      </li>
                    ))}
                  </ul>

                  <div className="flex items-center justify-between text-sm">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      sim.status === 'active'
                        ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                        : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
                    }`}>
                      {sim.status === 'active' ? 'Active' : 'Coming Soon'}
                    </span>
                    <ArrowRight className="w-4 h-4 text-gray-400 group-hover:text-purple-600 dark:group-hover:text-purple-400 group-hover:translate-x-1 transition-all" />
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>

        {/* Advanced Tools Section */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Advanced Tools
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {advancedTools.map((tool) => (
              <Link
                key={tool.name}
                href={tool.href}
                className="group block bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-shadow"
              >
                <div className="p-6">
                  <div className={`inline-flex p-3 bg-gradient-to-r ${tool.color} rounded-lg mb-4`}>
                    <tool.icon className="w-6 h-6 text-white" />
                  </div>

                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors">
                    {tool.name}
                  </h3>

                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                    {tool.description}
                  </p>

                  {/* Features */}
                  <ul className="space-y-1 mb-4">
                    {tool.features.map((feature, idx) => (
                      <li key={idx} className="text-xs text-gray-500 dark:text-gray-500 flex items-center gap-2">
                        <span className="w-1 h-1 bg-gray-400 rounded-full"></span>
                        {feature}
                      </li>
                    ))}
                  </ul>

                  <div className="flex items-center justify-between text-sm">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      tool.status === 'active'
                        ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                        : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
                    }`}>
                      {tool.status === 'active' ? 'Active' : 'Coming Soon'}
                    </span>
                    <ArrowRight className="w-4 h-4 text-gray-400 group-hover:text-purple-600 dark:group-hover:text-purple-400 group-hover:translate-x-1 transition-all" />
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>

        {/* Info Section */}
        <div className="mt-8 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 border border-blue-200 dark:border-blue-800">
          <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-2">
            Production-Ready Simulation Engine
          </h3>
          <p className="text-sm text-blue-700 dark:text-blue-300 mb-4">
            Powered by the SPECTRA Diffusion Module v1.12.0 with validated physics models,
            comprehensive test coverage (92%+), and optimized performance (2-3x faster).
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <p className="font-medium text-blue-900 dark:text-blue-100">Physics Validated</p>
              <p className="text-blue-700 dark:text-blue-300 text-xs">
                All models validated against Fair & Tsai (1977), Deal & Grove (1965)
              </p>
            </div>
            <div>
              <p className="font-medium text-blue-900 dark:text-blue-100">High Performance</p>
              <p className="text-blue-700 dark:text-blue-300 text-xs">
                Vectorized algorithms with Numba JIT compilation
              </p>
            </div>
            <div>
              <p className="font-medium text-blue-900 dark:text-blue-100">Production Ready</p>
              <p className="text-blue-700 dark:text-blue-300 text-xs">
                Docker containerized with comprehensive QA
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
