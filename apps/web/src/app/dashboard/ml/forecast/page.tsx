'use client'

import { useState } from 'react'
import { TimeSeriesForecast } from '@/components/ml/Session14Components'

// Generate mock historical data
const generateHistoricalData = () => {
  const baseValue = 100
  const trend = 0.1
  const seasonality = 5

  return Array.from({ length: 60 }, (_, i) => {
    const timestamp = new Date(Date.now() - (60 - i) * 24 * 60 * 60 * 1000).toISOString()
    const trendValue = baseValue + i * trend
    const seasonalValue = Math.sin((i * 2 * Math.PI) / 14) * seasonality
    const noise = (Math.random() - 0.5) * 3
    const value = trendValue + seasonalValue + noise

    return {
      timestamp,
      value: parseFloat(value.toFixed(2))
    }
  })
}

// Generate mock forecast data
const generateForecastData = () => {
  const lastHistoricalValue = 100 + 60 * 0.1
  const trend = 0.1
  const seasonality = 5

  return Array.from({ length: 30 }, (_, i) => {
    const ds = new Date(Date.now() + (i + 1) * 24 * 60 * 60 * 1000).toISOString()
    const trendValue = lastHistoricalValue + (i + 1) * trend
    const seasonalValue = Math.sin(((60 + i + 1) * 2 * Math.PI) / 14) * seasonality
    const yhat = trendValue + seasonalValue
    const uncertainty = 3 + i * 0.1

    return {
      ds,
      yhat: parseFloat(yhat.toFixed(2)),
      yhat_lower: parseFloat((yhat - uncertainty).toFixed(2)),
      yhat_upper: parseFloat((yhat + uncertainty).toFixed(2)),
      trend: parseFloat(trendValue.toFixed(2))
    }
  })
}

export default function ForecastPage() {
  const [historicalData] = useState(generateHistoricalData())
  const [forecast, setForecast] = useState(generateForecastData())

  const handleReforecast = async () => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000))

    // Regenerate forecast with slight variations
    setForecast(generateForecastData())
  }

  return (
    <TimeSeriesForecast
      historicalData={historicalData}
      forecast={forecast}
      onReforecast={handleReforecast}
    />
  )
}
