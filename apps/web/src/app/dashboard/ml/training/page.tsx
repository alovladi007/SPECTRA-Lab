'use client'

import { useState } from 'react'
import { ModelTrainingDashboard } from '@/components/ml/Session14Components'

// Generate mock models data
const generateMockModels = () => {
  const algorithms = ['random_forest', 'gradient_boosting', 'lightgbm', 'isolation_forest']
  const statuses = ['deployed', 'ready', 'training', 'archived']

  return Array.from({ length: 15 }, (_, i) => ({
    id: i + 1,
    name: `VM_Model_${i + 1}`,
    version: `${Math.floor(i / 3) + 1}.${i % 3}`,
    type: i % 4 === 3 ? 'anomaly_detection' : 'virtual_metrology',
    algorithm: algorithms[Math.floor(Math.random() * algorithms.length)],
    status: statuses[Math.floor(Math.random() * statuses.length)],
    metrics: {
      train: {
        r2: 0.85 + Math.random() * 0.1,
        rmse: 2 + Math.random() * 3,
        mae: 1.5 + Math.random() * 2,
        mape: 3 + Math.random() * 5
      },
      test: {
        r2: 0.80 + Math.random() * 0.1,
        rmse: 2.5 + Math.random() * 3,
        mae: 2 + Math.random() * 2,
        mape: 4 + Math.random() * 5
      },
      cv: {
        r2_mean: 0.82 + Math.random() * 0.08,
        r2_std: 0.02 + Math.random() * 0.03
      }
    },
    feature_importance: {
      temperature: 0.25 + Math.random() * 0.1,
      pressure: 0.20 + Math.random() * 0.1,
      flow_rate: 0.18 + Math.random() * 0.1,
      power: 0.15 + Math.random() * 0.1,
      gas_concentration: 0.12 + Math.random() * 0.08,
      chamber_pressure: 0.10 + Math.random() * 0.08
    },
    deployed_at: i % 2 === 0 ? new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString() : undefined,
    prediction_count: Math.floor(Math.random() * 10000),
    drift_detected: Math.random() > 0.7
  }))
}

export default function TrainingPage() {
  const [models, setModels] = useState(generateMockModels())

  const handleTrainModel = async (config: any) => {

    // Simulate model training API call
    await new Promise(resolve => setTimeout(resolve, 3000))

    // Add new model to the list
    const newModel = {
      id: models.length + 1,
      name: `VM_Model_${models.length + 1}`,
      version: '1.0',
      type: config.model_type,
      algorithm: config.algorithm,
      status: 'ready',
      metrics: {
        train: {
          r2: 0.85 + Math.random() * 0.1,
          rmse: 2 + Math.random() * 3,
          mae: 1.5 + Math.random() * 2,
          mape: 3 + Math.random() * 5
        },
        test: {
          r2: 0.80 + Math.random() * 0.1,
          rmse: 2.5 + Math.random() * 3,
          mae: 2 + Math.random() * 2,
          mape: 4 + Math.random() * 5
        },
        cv: {
          r2_mean: 0.82 + Math.random() * 0.08,
          r2_std: 0.02 + Math.random() * 0.03
        }
      },
      feature_importance: {
        temperature: 0.25 + Math.random() * 0.1,
        pressure: 0.20 + Math.random() * 0.1,
        flow_rate: 0.18 + Math.random() * 0.1,
        power: 0.15 + Math.random() * 0.1,
        gas_concentration: 0.12 + Math.random() * 0.08,
        chamber_pressure: 0.10 + Math.random() * 0.08
      },
      deployed_at: undefined,
      prediction_count: 0,
      drift_detected: false
    }

    setModels(prev => [newModel, ...prev])

    console.log('Model trained successfully:', newModel)
  }

  return (
    <ModelTrainingDashboard
      onTrainModel={handleTrainModel}
      models={models}
    />
  )
}
