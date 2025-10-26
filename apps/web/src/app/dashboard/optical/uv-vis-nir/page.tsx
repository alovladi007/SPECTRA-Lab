'use client'

import React from 'react'
import { Waves, Activity, TrendingUp } from 'lucide-react'

export default function UVVisNIRPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center">
          <Waves className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">UV-Vis-NIR Spectroscopy</h1>
          <p className="text-gray-600 mt-1">Optical absorption, transmission, and reflectance measurements</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center gap-3 mb-4">
            <Activity className="w-5 h-5 text-blue-600" />
            <h3 className="font-semibold text-gray-900">Recent Measurements</h3>
          </div>
          <p className="text-sm text-gray-600">No recent measurements</p>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center gap-3 mb-4">
            <TrendingUp className="w-5 h-5 text-green-600" />
            <h3 className="font-semibold text-gray-900">Analysis Status</h3>
          </div>
          <p className="text-sm text-gray-600">System ready</p>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Quick Actions</h3>
          <button className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            New Measurement
          </button>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="font-semibold text-blue-900 mb-2">Measurement Capabilities</h3>
        <ul className="space-y-2 text-sm text-blue-800">
          <li>• Wavelength range: 200-2500 nm</li>
          <li>• Absorption and transmission spectra</li>
          <li>• Band gap determination</li>
          <li>• Optical density measurements</li>
          <li>• Reflectance spectroscopy</li>
        </ul>
      </div>
    </div>
  )
}
