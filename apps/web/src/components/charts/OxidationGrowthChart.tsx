'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface OxidationGrowthChartProps {
  time: number[]
  thickness: number[]
  mode?: string
}

export function OxidationGrowthChart({ time, thickness, mode = 'forward' }: OxidationGrowthChartProps) {
  const data = time.map((t, i) => ({
    time: t,
    thickness: thickness[i]
  }))

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
        <XAxis
          dataKey="time"
          label={{ value: 'Time (min)', position: 'insideBottom', offset: -5 }}
          className="text-gray-700 dark:text-gray-300"
        />
        <YAxis
          label={{ value: 'Oxide Thickness (nm)', angle: -90, position: 'insideLeft' }}
          className="text-gray-700 dark:text-gray-300"
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'rgb(31 41 55)',
            border: '1px solid rgb(55 65 81)',
            borderRadius: '0.5rem',
            color: 'rgb(243 244 246)'
          }}
          formatter={(value: number) => [`${value.toFixed(2)} nm`, 'Oxide Thickness']}
          labelFormatter={(label) => `Time: ${label} min`}
        />
        <Legend />
        <Line
          type="monotone"
          dataKey="thickness"
          stroke="#10b981"
          strokeWidth={2}
          dot={{ fill: '#10b981', r: 3 }}
          name="Oxide Thickness"
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
