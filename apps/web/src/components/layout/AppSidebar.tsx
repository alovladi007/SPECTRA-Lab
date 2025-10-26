'use client'

import React, { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  Home, Zap, Waves, Layers, Flask, TrendingUp, Brain,
  ChevronDown, ChevronRight, Activity, BarChart3,
  Microscope, Atom, Sparkles, Database, Settings
} from 'lucide-react'

interface NavItem {
  name: string
  href?: string
  icon?: React.ElementType
  badge?: string
  children?: NavItem[]
}

const navigation: NavItem[] = [
  {
    name: 'Dashboard',
    href: '/dashboard',
    icon: Home,
  },
  {
    name: 'Electrical Characterization',
    icon: Zap,
    children: [
      {
        name: 'Resistance & Mobility',
        children: [
          { name: 'Four-Point Probe (4PP)', href: '/electrical/four-point-probe' },
          { name: 'Hall Effect', href: '/electrical/hall-effect' },
        ]
      },
      {
        name: 'Device Characterization',
        children: [
          { name: 'BJT Analysis', href: '/electrical/bjt' },
          { name: 'BJT Advanced', href: '/electrical/bjt-advanced' },
          { name: 'MOSFET Analysis', href: '/electrical/mosfet' },
          { name: 'MOSFET Advanced', href: '/electrical/mosfet-advanced' },
          { name: 'Solar Cell Testing', href: '/electrical/solar-cell' },
          { name: 'C-V Profiling', href: '/electrical/cv-profiling' },
        ]
      },
      {
        name: 'Advanced Electrical',
        children: [
          { name: 'DLTS', href: '/electrical/advanced', badge: 'DLTS' },
          { name: 'EBIC', href: '/electrical/advanced', badge: 'EBIC' },
          { name: 'PCD', href: '/electrical/advanced', badge: 'PCD' },
        ]
      },
    ],
  },
  {
    name: 'Optical Characterization',
    icon: Waves,
    children: [
      {
        name: 'Spectroscopy',
        children: [
          { name: 'UV-Vis-NIR', href: '/optical/uv-vis-nir' },
          { name: 'FTIR', href: '/optical/ftir' },
          { name: 'Raman', href: '/optical/raman' },
        ]
      },
      {
        name: 'Advanced Optical',
        children: [
          { name: 'Ellipsometry', href: '/optical/ellipsometry' },
          { name: 'Photoluminescence (PL)', href: '/optical/photoluminescence' },
        ]
      },
    ],
  },
  {
    name: 'Structural Analysis',
    icon: Layers,
    children: [
      {
        name: 'Diffraction',
        children: [
          { name: 'X-Ray Diffraction (XRD)', href: '/structural/xrd' },
        ]
      },
      {
        name: 'Microscopy',
        children: [
          { name: 'SEM', href: '/structural/microscopy', badge: 'SEM' },
          { name: 'TEM', href: '/structural/microscopy', badge: 'TEM' },
          { name: 'AFM', href: '/structural/microscopy', badge: 'AFM' },
          { name: 'Optical Microscopy', href: '/structural/microscopy', badge: 'Optical' },
        ]
      },
    ],
  },
  {
    name: 'Chemical Analysis',
    icon: Flask,
    children: [
      {
        name: 'Surface Analysis',
        children: [
          { name: 'XPS', href: '/chemical/surface-analysis', badge: 'XPS' },
          { name: 'XRF', href: '/chemical/surface-analysis', badge: 'XRF' },
        ]
      },
      {
        name: 'Bulk Analysis',
        children: [
          { name: 'SIMS', href: '/chemical/bulk-analysis', badge: 'SIMS' },
          { name: 'RBS', href: '/chemical/bulk-analysis', badge: 'RBS' },
          { name: 'NAA', href: '/chemical/bulk-analysis', badge: 'NAA' },
          { name: 'Chemical Etch', href: '/chemical/bulk-analysis', badge: 'Etch' },
        ]
      },
    ],
  },
  {
    name: 'Statistical Process Control',
    icon: TrendingUp,
    children: [
      { name: 'SPC Dashboard', href: '/spc' },
      { name: 'Control Charts', href: '/spc/charts' },
      { name: 'Capability Analysis', href: '/spc/capability' },
      { name: 'Alerts & Violations', href: '/spc/alerts' },
    ],
  },
  {
    name: 'ML & Virtual Metrology',
    icon: Brain,
    children: [
      { name: 'Virtual Metrology Models', href: '/ml/vm-models' },
      { name: 'Model Training', href: '/ml/training' },
      { name: 'AutoML', href: '/ml/automl' },
      { name: 'Anomaly Detection', href: '/ml/anomaly' },
      { name: 'Drift Monitoring', href: '/ml/monitoring' },
      { name: 'Time Series Forecast', href: '/ml/forecast' },
      { name: 'Model Explainability', href: '/ml/explainability' },
      { name: 'A/B Testing', href: '/ml/ab-testing' },
    ],
  },
  {
    name: 'Data & Samples',
    icon: Database,
    children: [
      { name: 'Sample Manager', href: '/samples' },
      { name: 'Experiments', href: '/experiments' },
      { name: 'Results Browser', href: '/results' },
      { name: 'Data Export', href: '/data/export' },
    ],
  },
  {
    name: 'System',
    icon: Settings,
    children: [
      { name: 'Instruments', href: '/system/instruments' },
      { name: 'Users & Roles', href: '/system/users' },
      { name: 'Settings', href: '/system/settings' },
    ],
  },
]

function NavGroup({ item, level = 0 }: { item: NavItem; level?: number }) {
  const [isOpen, setIsOpen] = useState(level < 2)
  const pathname = usePathname()
  const hasChildren = item.children && item.children.length > 0
  const isActive = item.href && pathname === item.href

  if (hasChildren) {
    return (
      <div>
        <button
          onClick={() => setIsOpen(!isOpen)}
          className={`
            w-full flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-colors
            ${level === 0 ? 'font-medium text-gray-200 hover:bg-gray-800' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'}
          `}
        >
          {item.icon && <item.icon className="w-4 h-4 flex-shrink-0" />}
          <span className="flex-1 text-left truncate">{item.name}</span>
          {isOpen ? (
            <ChevronDown className="w-4 h-4 flex-shrink-0" />
          ) : (
            <ChevronRight className="w-4 h-4 flex-shrink-0" />
          )}
        </button>
        {isOpen && (
          <div className={`${level === 0 ? 'ml-0' : 'ml-3'} mt-1 space-y-1`}>
            {item.children.map((child, index) => (
              <NavGroup key={index} item={child} level={level + 1} />
            ))}
          </div>
        )}
      </div>
    )
  }

  return (
    <Link
      href={item.href || '#'}
      className={`
        flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-colors
        ${isActive
          ? 'bg-blue-600 text-white font-medium'
          : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
        }
      `}
    >
      {item.icon && <item.icon className="w-4 h-4 flex-shrink-0" />}
      <span className="flex-1 truncate">{item.name}</span>
      {item.badge && (
        <span className="px-2 py-0.5 text-xs bg-blue-500/20 text-blue-400 rounded-full">
          {item.badge}
        </span>
      )}
    </Link>
  )
}

export function AppSidebar() {
  return (
    <div className="w-64 bg-gray-900 flex flex-col h-screen border-r border-gray-800">
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-gray-800">
        <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
          <Sparkles className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-lg font-bold text-white">SPECTRA-Lab</h1>
          <p className="text-xs text-gray-400">Enterprise Platform</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {navigation.map((item, index) => (
          <NavGroup key={index} item={item} />
        ))}
      </nav>

      {/* Status */}
      <div className="px-4 py-3 border-t border-gray-800">
        <div className="flex items-center gap-2 text-xs">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span className="text-gray-400">All Systems Operational</span>
        </div>
        <div className="mt-2 text-xs text-gray-500">
          v2.0.0 • 14/16 Sessions
        </div>
      </div>
    </div>
  )
}
