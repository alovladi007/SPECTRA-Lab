"use client"

import dynamic from 'next/dynamic'

// Dynamically import the component to avoid SSR issues
const IonImplantControl = dynamic(
  () => import('@/components/process-control/IonImplantControl'),
  { ssr: false }
)

export default function IonImplantPage() {
  return (
    <div className="p-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Ion Implantation Control</h1>
        <p className="mt-2 text-gray-600">
          Real-time control and monitoring of ion beam parameters
        </p>
      </div>
      <IonImplantControl />
    </div>
  )
}
