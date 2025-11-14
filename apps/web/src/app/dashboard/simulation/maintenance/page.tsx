"use client"

import dynamic from 'next/dynamic'

// Dynamically import the component to avoid SSR issues
const PredictiveMaintenanceMES = dynamic(
  () => import('@/components/manufacturing/PredictiveMaintenanceMES'),
  { ssr: false }
)

export default function PredictiveMaintenancePage() {
  return <PredictiveMaintenanceMES />
}
