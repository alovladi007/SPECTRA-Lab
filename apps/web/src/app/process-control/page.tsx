"use client"

import Link from 'next/link'
import { Activity, Zap, LineChart, Brain } from 'lucide-react'

export default function ProcessControlPage() {
  const modules = [
    {
      title: 'Ion Implantation',
      description: 'Control and monitor ion beam parameters, dose profiles, and SRIM calculations',
      icon: Zap,
      href: '/process-control/ion-implant',
      color: 'text-purple-600',
      bgColor: 'bg-purple-50',
    },
    {
      title: 'Rapid Thermal Processing',
      description: 'Multi-zone temperature control, recipe execution, and thermal profiling',
      icon: Activity,
      href: '/process-control/rtp',
      color: 'text-orange-600',
      bgColor: 'bg-orange-50',
    },
    {
      title: 'Statistical Process Control',
      description: 'SPC charts, Western Electric rules, and process capability analysis',
      icon: LineChart,
      href: '/process-control/spc',
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
    },
    {
      title: 'Virtual Metrology',
      description: 'ML-based process prediction and feature engineering',
      icon: Brain,
      href: '/process-control/vm',
      color: 'text-green-600',
      bgColor: 'bg-green-50',
    },
  ]

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Process Control</h1>
        <p className="mt-2 text-gray-600">
          Advanced control systems for semiconductor fabrication processes
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-6">
        {modules.map((module) => {
          const Icon = module.icon
          return (
            <Link
              key={module.title}
              href={module.href}
              className="block p-6 bg-white rounded-lg border border-gray-200 hover:border-blue-300 hover:shadow-md transition-all"
            >
              <div className="flex items-start">
                <div className={`p-3 rounded-lg ${module.bgColor}`}>
                  <Icon className={`w-6 h-6 ${module.color}`} />
                </div>
                <div className="ml-4 flex-1">
                  <h2 className="text-xl font-semibold text-gray-900">{module.title}</h2>
                  <p className="mt-2 text-sm text-gray-600">{module.description}</p>
                </div>
              </div>
            </Link>
          )
        })}
      </div>
    </div>
  )
}
