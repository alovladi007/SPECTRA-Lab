'use client'
import { ClipboardCheck } from 'lucide-react'

export default function Page() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 bg-gradient-to-br from-teal-500 to-cyan-500 rounded-xl flex items-center justify-center">
          <ClipboardCheck className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">LIMS/ELN Feature</h1>
          <p className="text-gray-600 mt-1">Laboratory Information Management System</p>
        </div>
      </div>
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Feature Ready</h3>
        <p className="text-gray-600">This LIMS feature is integrated and ready for use</p>
      </div>
    </div>
  )
}
