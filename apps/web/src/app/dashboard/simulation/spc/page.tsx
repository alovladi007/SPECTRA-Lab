"use client"

import dynamic from 'next/dynamic'

// Dynamically import the component to avoid SSR issues
const SPCMES = dynamic(
  () => import('@/components/manufacturing/SPCMES'),
  { ssr: false }
)

export default function SPCPage() {
  return <SPCMES />
}
