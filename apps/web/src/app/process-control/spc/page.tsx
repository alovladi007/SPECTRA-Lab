"use client"

import dynamic from 'next/dynamic'

// Dynamically import the component to avoid SSR issues
const SPCMonitoring = dynamic(
  () => import('@/components/process-control/SPCMonitoring'),
  { ssr: false }
)

export default function SPCPage() {
  return <SPCMonitoring />
}
