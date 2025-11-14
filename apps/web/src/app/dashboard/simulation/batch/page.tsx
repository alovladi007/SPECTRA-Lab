"use client"

import dynamic from 'next/dynamic'

// Dynamically import the component to avoid SSR issues
const BatchJobMES = dynamic(
  () => import('@/components/manufacturing/BatchJobMES'),
  { ssr: false }
)

export default function BatchJobPage() {
  return <BatchJobMES />
}
