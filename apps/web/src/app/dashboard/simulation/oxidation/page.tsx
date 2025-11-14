"use client"

import dynamic from 'next/dynamic'

// Dynamically import the component to avoid SSR issues
const OxidationMES = dynamic(
  () => import('@/components/manufacturing/OxidationMES'),
  { ssr: false }
)

export default function OxidationPage() {
  return <OxidationMES />
}
