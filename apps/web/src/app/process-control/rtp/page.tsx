"use client"

import dynamic from 'next/dynamic'

// Dynamically import the component to avoid SSR issues
const RTPControl = dynamic(
  () => import('@/components/process-control/RTPControl'),
  { ssr: false }
)

export default function RTPPage() {
  return (
    <div className="p-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Rapid Thermal Processing Control</h1>
        <p className="mt-2 text-gray-600">
          Multi-zone temperature control and recipe execution
        </p>
      </div>
      <RTPControl />
    </div>
  )
}
