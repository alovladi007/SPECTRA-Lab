'use client'
import { TrendingUp } from 'lucide-react'

export default function Page() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 bg-gradient-to-br from-red-500 to-rose-500 rounded-xl flex items-center justify-center">
          <TrendingUp className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Statistical Process Control</h1>
          <p className="text-gray-600 mt-1">Real-time process monitoring and control</p>
        </div>
      </div>
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">SPC Module Active</h3>
        <p className="text-gray-600">Statistical process control tools ready</p>
      </div>
    </div>
  )
}
