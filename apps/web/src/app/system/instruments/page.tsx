'use client'
import { Settings, Cpu } from 'lucide-react'

export default function InstrumentsPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 bg-gradient-to-br from-gray-700 to-gray-900 rounded-xl flex items-center justify-center">
          <Cpu className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Instruments</h1>
          <p className="text-gray-600 mt-1">Manage connected instruments</p>
        </div>
      </div>
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
        <Cpu className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Instrument Manager</h3>
        <p className="text-gray-600">Configure and monitor connected instruments</p>
      </div>
    </div>
  )
}
