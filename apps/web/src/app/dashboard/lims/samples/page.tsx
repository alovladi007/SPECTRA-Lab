'use client'
import { ClipboardCheck, Barcode } from 'lucide-react'

export default function SamplesPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 bg-gradient-to-br from-teal-500 to-cyan-500 rounded-xl flex items-center justify-center">
          <ClipboardCheck className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Sample Tracking</h1>
          <p className="text-gray-600 mt-1">Manage sample lifecycle with barcode/QR tracking</p>
        </div>
      </div>
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
        <Barcode className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-gray-900 mb-2">LIMS Sample Management</h3>
        <p className="text-gray-600">Create and track samples with automated barcode generation</p>
      </div>
    </div>
  )
}
