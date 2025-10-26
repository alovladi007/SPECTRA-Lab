'use client'
import { Waves } from 'lucide-react'

export default function Page() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center">
          <Waves className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Optical Characterization</h1>
          <p className="text-gray-600 mt-1">Advanced optical measurement techniques</p>
        </div>
      </div>
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">System Ready</h3>
        <p className="text-gray-600">Optical characterization module available</p>
      </div>
    </div>
  )
}
