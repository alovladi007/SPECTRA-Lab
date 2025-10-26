'use client'
import { Layers } from 'lucide-react'

export default function Page() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
          <Layers className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Structural Analysis</h1>
          <p className="text-gray-600 mt-1">Crystal structure and material characterization</p>
        </div>
      </div>
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">System Ready</h3>
        <p className="text-gray-600">Structural analysis module available</p>
      </div>
    </div>
  )
}
