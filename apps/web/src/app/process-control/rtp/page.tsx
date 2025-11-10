"use client"

import dynamic from 'next/dynamic'

// Dynamically import the component to avoid SSR issues
const RTPControl = dynamic(
  () => import('@/components/process-control/RTPControl'),
  { ssr: false }
)

export default function RTPPage() {
  return <RTPControl />
}
