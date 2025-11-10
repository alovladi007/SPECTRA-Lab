"use client"

import dynamic from 'next/dynamic'

// Dynamically import the component to avoid SSR issues
const SPCMonitoring = dynamic(
  () => import('@/components/process-control/SPCMonitoring'),
  { ssr: false }
)

export default function SPCPage() {
  return (
    <div className="p-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Statistical Process Control</h1>
        <p className="mt-2 text-gray-600">
          SPC charts, alerts, and process capability analysis
        </p>
      </div>
      <SPCMonitoring />
    </div>
  )
}
