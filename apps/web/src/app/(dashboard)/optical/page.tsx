'use client'

import React, { useState } from 'react'
import Link from 'next/link'
import { Waves, ArrowRight, Sparkles, Info } from 'lucide-react'

const opticalMethods = [
  {
    id: 'uv-vis-nir',
    name: 'UV-Vis-NIR Spectroscopy',
    description: 'Absorption, transmission, and reflectance measurements from 200-2500 nm',
    capabilities: [
      'Absorption coefficient determination',
      'Band gap extraction (Tauc plot)',
      'Transmission/reflection spectra',
      'Thin film thickness estimation',
    ],
    href: '/optical/uv-vis-nir',
    color: 'from-blue-500 to-cyan-500',
  },
  {
    id: 'ftir',
    name: 'FTIR Spectroscopy',
    description: 'Fourier Transform Infrared analysis for molecular vibrational modes',
    capabilities: [
      'Chemical bond identification',
      'Material composition analysis',
      'Stress/strain measurement',
      'Contamination detection',
    ],
    href: '/optical/ftir',
    color: 'from-purple-500 to-pink-500',
  },
  {
    id: 'ellipsometry',
    name: 'Ellipsometry',
    description: 'Precise thin film thickness and optical properties measurement',
    capabilities: [
      'Film thickness (Å resolution)',
      'Refractive index (n)',
      'Extinction coefficient (k)',
      'Multi-layer modeling',
    ],
    href: '/optical/ellipsometry',
    color: 'from-green-500 to-emerald-500',
  },
  {
    id: 'photoluminescence',
    name: 'Photoluminescence (PL)',
    description: 'Optical emission spectroscopy for electronic properties',
    capabilities: [
      'Band gap measurement',
      'Defect identification',
      'Material quality assessment',
      'Quantum efficiency',
    ],
    href: '/optical/photoluminescence',
    color: 'from-yellow-500 to-orange-500',
  },
  {
    id: 'raman',
    name: 'Raman Spectroscopy',
    description: 'Molecular vibrational modes and crystal structure analysis',
    capabilities: [
      'Crystal phase identification',
      'Stress/strain mapping',
      'Doping concentration',
      'Material composition',
    ],
    href: '/optical/raman',
    color: 'from-red-500 to-rose-500',
  },
]

export default function OpticalCharacterizationPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 mb-2">
          <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center">
            <Waves className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Optical Characterization</h1>
            <p className="text-gray-600">5 advanced optical analysis methods</p>
          </div>
        </div>
      </div>

      {/* Overview Card */}
      <div className="bg-gradient-to-br from-blue-50 to-cyan-50 border border-blue-200 rounded-lg p-6">
        <div className="flex items-start gap-4">
          <Info className="w-6 h-6 text-blue-600 flex-shrink-0 mt-1" />
          <div>
            <h2 className="text-lg font-semibold text-gray-900 mb-2">Optical Characterization Suite</h2>
            <p className="text-gray-700 mb-4">
              Comprehensive optical analysis tools for semiconductor materials, covering spectroscopy,
              thin film properties, luminescence, and molecular vibrations across UV to NIR wavelengths.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white rounded-lg p-3">
                <p className="text-sm text-gray-600">Wavelength Range</p>
                <p className="text-lg font-semibold text-gray-900">200-2500 nm</p>
              </div>
              <div className="bg-white rounded-lg p-3">
                <p className="text-sm text-gray-600">Resolution</p>
                <p className="text-lg font-semibold text-gray-900">Ångström level</p>
              </div>
              <div className="bg-white rounded-lg p-3">
                <p className="text-sm text-gray-600">Applications</p>
                <p className="text-lg font-semibold text-gray-900">20+ parameters</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Methods Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {opticalMethods.map((method) => (
          <Link
            key={method.id}
            href={method.href}
            className="group bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-lg transition-all duration-200 hover:border-blue-300"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex-1">
                <h3 className="text-xl font-semibold text-gray-900 mb-2">{method.name}</h3>
                <p className="text-sm text-gray-600">{method.description}</p>
              </div>
              <div className={`w-12 h-12 bg-gradient-to-br ${method.color} rounded-xl flex items-center justify-center flex-shrink-0 ml-4`}>
                <Sparkles className="w-6 h-6 text-white" />
              </div>
            </div>

            <div className="space-y-2 mb-4">
              <p className="text-sm font-medium text-gray-700">Key Capabilities:</p>
              <ul className="space-y-1">
                {method.capabilities.map((capability, index) => (
                  <li key={index} className="text-sm text-gray-600 flex items-center gap-2">
                    <div className="w-1.5 h-1.5 bg-blue-500 rounded-full" />
                    {capability}
                  </li>
                ))}
              </ul>
            </div>

            <div className="flex items-center gap-2 text-blue-600 group-hover:text-blue-700 pt-4 border-t border-gray-200">
              <span className="text-sm font-medium">Start Analysis</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </Link>
        ))}
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <p className="text-sm text-gray-600 mb-2">Total Analyses (30 days)</p>
          <p className="text-3xl font-bold text-gray-900">247</p>
          <p className="text-sm text-green-600 mt-2">+12% from last month</p>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <p className="text-sm text-gray-600 mb-2">Average Measurement Time</p>
          <p className="text-3xl font-bold text-gray-900">12 min</p>
          <p className="text-sm text-blue-600 mt-2">Optimized workflow</p>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <p className="text-sm text-gray-600 mb-2">Data Quality</p>
          <p className="text-3xl font-bold text-gray-900">98.5%</p>
          <p className="text-sm text-gray-600 mt-2">Measurement reliability</p>
        </div>
      </div>
    </div>
  )
}
