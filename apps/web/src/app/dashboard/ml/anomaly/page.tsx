'use client'

import { useState } from 'react'
import { AnomalyMonitor } from '@/components/ml/Session14Components'

// Mock anomaly data
const generateMockAnomalies = () => {
  const anomalyTypes = ['point', 'contextual', 'collective']
  const features = ['temperature', 'pressure', 'flow_rate', 'power', 'gas_concentration', 'chamber_pressure']
  const causes = [
    'Equipment malfunction detected',
    'Process parameter deviation',
    'Environmental condition change',
    'Sensor calibration drift',
    'Material batch variation',
    'Recipe parameter mismatch'
  ]

  return Array.from({ length: 25 }, (_, i) => {
    const timestamp = new Date(Date.now() - (24 - i) * 60 * 60 * 1000).toISOString()
    const anomalyType = anomalyTypes[Math.floor(Math.random() * anomalyTypes.length)]
    const score = 0.5 + Math.random() * 0.5

    const featureContributions: Record<string, number> = {}
    const selectedFeatures = features.slice(0, 3 + Math.floor(Math.random() * 3))
    selectedFeatures.forEach(f => {
      featureContributions[f] = Math.random()
    })

    const featureValues: Record<string, number> = {}
    features.forEach(f => {
      featureValues[f] = 100 + Math.random() * 50
    })

    return {
      id: i + 1,
      timestamp,
      is_anomaly: true,
      anomaly_score: score,
      anomaly_type: anomalyType,
      features: featureValues,
      feature_contributions: featureContributions,
      likely_causes: causes.slice(0, 2 + Math.floor(Math.random() * 2)),
      resolved: Math.random() > 0.6
    }
  })
}

export default function AnomalyPage() {
  const [anomalies, setAnomalies] = useState(generateMockAnomalies())

  const handleResolve = async (id: number) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500))

    setAnomalies(prev =>
      prev.map(a => a.id === id ? { ...a, resolved: true } : a)
    )
  }

  const handleInvestigate = (id: number) => {
    console.log('Investigating anomaly:', id)
    // In a real app, this would navigate to a detailed view
    alert(`Opening detailed investigation view for Anomaly #${id}`)
  }

  return (
    <AnomalyMonitor
      anomalies={anomalies}
      onResolve={handleResolve}
      onInvestigate={handleInvestigate}
    />
  )
}
