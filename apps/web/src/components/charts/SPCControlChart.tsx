'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts'

interface SPCControlChartProps {
  dataPoints: number[]
  centerline: number
  ucl: number
  lcl: number
}

export function SPCControlChart({ dataPoints, centerline, ucl, lcl }: SPCControlChartProps) {
  const data = dataPoints.map((value, index) => ({
    sample: index + 1,
    value,
    centerline,
    ucl,
    lcl
  }))

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
        <XAxis
          dataKey="sample"
          label={{ value: 'Sample Number', position: 'insideBottom', offset: -5 }}
          className="text-gray-700 dark:text-gray-300"
        />
        <YAxis
          label={{ value: 'Measurement', angle: -90, position: 'insideLeft' }}
          className="text-gray-700 dark:text-gray-300"
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'rgb(31 41 55)',
            border: '1px solid rgb(55 65 81)',
            borderRadius: '0.5rem',
            color: 'rgb(243 244 246)'
          }}
        />
        <Legend />

        {/* Control limits */}
        <ReferenceLine
          y={ucl}
          stroke="#ef4444"
          strokeDasharray="3 3"
          label={{ value: 'UCL', fill: '#ef4444', position: 'right' }}
        />
        <ReferenceLine
          y={centerline}
          stroke="#22c55e"
          strokeDasharray="3 3"
          label={{ value: 'CL', fill: '#22c55e', position: 'right' }}
        />
        <ReferenceLine
          y={lcl}
          stroke="#ef4444"
          strokeDasharray="3 3"
          label={{ value: 'LCL', fill: '#ef4444', position: 'right' }}
        />

        {/* Data line */}
        <Line
          type="monotone"
          dataKey="value"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={{ fill: '#3b82f6', r: 4 }}
          name="Measurement"
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
