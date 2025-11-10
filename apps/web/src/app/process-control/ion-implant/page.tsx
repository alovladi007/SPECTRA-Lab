"use client"

import dynamic from 'next/dynamic'

// Dynamically import the component to avoid SSR issues
const IonImplantControl = dynamic(
  () => import('@/components/process-control/IonImplantControl'),
  { ssr: false }
)

export default function IonImplantPage() {
  return <IonImplantControl />
}
