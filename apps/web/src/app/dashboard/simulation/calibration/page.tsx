"use client"

import dynamic from 'next/dynamic'

// Dynamically import the component to avoid SSR issues
const CalibrationMES = dynamic(
  () => import('@/components/manufacturing/CalibrationMES'),
  { ssr: false }
)

export default function CalibrationPage() {
  return <CalibrationMES />
}
