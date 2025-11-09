'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface DiffusionProfileChartProps {
  depth: number[]
  concentration: number[]
}

export function DiffusionProfileChart({ depth, concentration }: DiffusionProfileChartProps) {
  const data = depth.map((d, i) => ({
    depth: d,
    concentration: concentration[i]
  }))

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
        <XAxis
          dataKey="depth"
          label={{ value: 'Depth (nm)', position: 'insideBottom', offset: -5 }}
          className="text-gray-700 dark:text-gray-300"
        />
        <YAxis
          scale="log"
          domain={['auto', 'auto']}
          label={{ value: 'Concentration (atoms/cm³)', angle: -90, position: 'insideLeft' }}
          className="text-gray-700 dark:text-gray-300"
          tickFormatter={(value) => value.toExponential(1)}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'rgb(31 41 55)',
            border: '1px solid rgb(55 65 81)',
            borderRadius: '0.5rem',
            color: 'rgb(243 244 246)'
          }}
          formatter={(value: number) => [value.toExponential(2), 'Concentration']}
          labelFormatter={(label) => `Depth: ${label} nm`}
        />
        <Legend />
        <Line
          type="monotone"
          dataKey="concentration"
          stroke="#8b5cf6"
          strokeWidth={2}
          dot={false}
          name="Dopant Concentration"
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
