'use client'

import React from 'react'
import { Zap, Activity } from 'lucide-react'

export default function FourPointProbePage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 bg-gradient-to-br from-yellow-500 to-orange-500 rounded-xl flex items-center justify-center">
          <Zap className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Four-Point Probe</h1>
          <p className="text-gray-600 mt-1">Van der Pauw method for sheet resistance measurement</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center gap-3 mb-4">
            <Activity className="w-5 h-5 text-blue-600" />
            <h3 className="font-semibold text-gray-900">Measurement Status</h3>
          </div>
          <p className="text-sm text-gray-600">Ready to measure</p>
          <button className="mt-4 w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            Start Measurement
          </button>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Configuration</h3>
          <div className="space-y-3 text-sm">
            <div>
              <label className="text-gray-600">Test Current (A)</label>
              <input
                type="number"
                className="w-full mt-1 px-3 py-2 border rounded-lg"
                defaultValue="0.001"
              />
            </div>
            <div>
              <label className="text-gray-600">Sample Thickness (cm)</label>
              <input
                type="number"
                className="w-full mt-1 px-3 py-2 border rounded-lg"
                defaultValue="0.05"
              />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Last Result</h3>
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600">125.3</div>
            <div className="text-sm text-gray-600">Ω/sq</div>
            <div className="text-xs text-gray-500 mt-2">Sheet Resistance</div>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="font-semibold text-blue-900 mb-2">Method Capabilities</h3>
        <ul className="space-y-2 text-sm text-blue-800">
          <li>• Van der Pauw method for arbitrary sample shapes</li>
          <li>• Sheet resistance measurement (Ω/sq)</li>
          <li>• Resistivity calculation with thickness input</li>
          <li>• Four-wire sensing for accurate measurements</li>
          <li>• Temperature compensation</li>
          <li>• Contact resistance verification</li>
        </ul>
      </div>
    </div>
  )
}
