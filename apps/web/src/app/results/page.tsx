'use client'
import { BarChart3 } from 'lucide-react'

export default function ResultsPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
          <BarChart3 className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Results Browser</h1>
          <p className="text-gray-600 mt-1">Browse and analyze measurement results</p>
        </div>
      </div>
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Results Database</h3>
        <p className="text-gray-600">View and export measurement data</p>
      </div>
    </div>
  )
}
