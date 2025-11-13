'use client'

import React, { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  Home, Zap, Waves, Layers, Beaker, TrendingUp, Brain,
  ChevronDown, ChevronRight, Activity, BarChart3,
  Microscope, Atom, Sparkles, Database, Settings, FileText, ClipboardCheck, Gauge
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
          { name: 'Four-Point Probe (4PP)', href: '/dashboard/electrical/four-point-probe' },
          { name: 'Hall Effect', href: '/dashboard/electrical/hall-effect' },
        ]
      },
      {
        name: 'Device Characterization',
        children: [
          { name: 'BJT Analysis', href: '/dashboard/electrical/bjt' },
          { name: 'BJT Advanced', href: '/dashboard/electrical/bjt-advanced' },
          { name: 'MOSFET Analysis', href: '/dashboard/electrical/mosfet' },
          { name: 'MOSFET Advanced', href: '/dashboard/electrical/mosfet-advanced' },
          { name: 'Solar Cell Testing', href: '/dashboard/electrical/solar-cell' },
          { name: 'C-V Profiling', href: '/dashboard/electrical/cv-profiling' },
        ]
      },
      {
        name: 'Advanced Electrical',
        children: [
          { name: 'DLTS', href: '/dashboard/electrical/dlts' },
          { name: 'EBIC', href: '/dashboard/electrical/ebic' },
          { name: 'PCD', href: '/dashboard/electrical/pcd' },
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
          { name: 'UV-Vis-NIR', href: '/dashboard/optical/uv-vis-nir' },
          { name: 'FTIR', href: '/dashboard/optical/ftir' },
          { name: 'Raman', href: '/dashboard/optical/raman' },
        ]
      },
      {
        name: 'Advanced Optical',
        children: [
          { name: 'Ellipsometry', href: '/dashboard/optical/ellipsometry' },
          { name: 'Photoluminescence (PL)', href: '/dashboard/optical/photoluminescence' },
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
          { name: 'X-Ray Diffraction (XRD)', href: '/dashboard/structural/xrd' },
        ]
      },
      {
        name: 'Microscopy',
        children: [
          { name: 'Microscopy Suite', href: '/dashboard/structural/microscopy' },
        ]
      },
    ],
  },
  {
    name: 'Chemical Analysis',
    icon: Beaker,
    children: [
      { name: 'Surface Analysis', href: '/dashboard/chemical/surface-analysis' },
      { name: 'Bulk Analysis', href: '/dashboard/chemical/bulk-analysis' },
    ],
  },
  {
    name: 'Process Simulation',
    icon: Activity,
    children: [
      {
        name: 'Core Simulations',
        children: [
          { name: 'Diffusion Simulation', href: '/dashboard/simulation/diffusion' },
          { name: 'Oxidation Planning', href: '/dashboard/simulation/oxidation' },
          { name: 'SPC Monitoring', href: '/dashboard/simulation/spc' },
        ]
      },
      {
        name: 'Advanced Tools',
        children: [
          { name: 'Calibration', href: '/dashboard/simulation/calibration' },
          { name: 'Batch Job Manager', href: '/dashboard/simulation/batch' },
          { name: 'Predictive Maintenance', href: '/dashboard/simulation/maintenance' },
        ]
      },
    ],
  },
  {
    name: 'Process Control',
    icon: Gauge,
    children: [
      { name: 'Ion Implantation', href: '/process-control/ion-implant' },
      { name: 'Rapid Thermal Processing', href: '/process-control/rtp' },
      { name: 'CVD Platform', href: '/cvd/workspace', badge: 'NEW' },
      { name: 'Statistical Process Control', href: '/process-control/spc' },
      { name: 'Virtual Metrology', href: '/process-control/vm' },
    ],
  },
  {
    name: 'Statistical Process Control',
    icon: TrendingUp,
    href: '/dashboard/spc',
  },
  {
    name: 'ML & Virtual Metrology',
    icon: Brain,
    children: [
      { name: 'Virtual Metrology Models', href: '/dashboard/ml/vm-models' },
      { name: 'Model Training', href: '/dashboard/ml/training' },
      { name: 'AutoML', href: '/dashboard/ml/automl' },
      { name: 'Monitoring Dashboard', href: '/dashboard/ml/monitoring' },
      { name: 'Model Explainability', href: '/dashboard/ml/explainability' },
      { name: 'A/B Testing', href: '/dashboard/ml/ab-testing' },
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
    name: 'LIMS & ELN',
    icon: ClipboardCheck,
    children: [
      { name: 'Sample Tracking', href: '/dashboard/lims/samples' },
      { name: 'Chain of Custody', href: '/dashboard/lims/custody' },
      { name: 'Lab Notebook', href: '/dashboard/lims/eln' },
      { name: 'E-Signatures', href: '/dashboard/lims/signatures' },
      { name: 'SOP Management', href: '/dashboard/lims/sops' },
      { name: 'Report Generator', href: '/dashboard/lims/reports' },
      { name: 'FAIR Export', href: '/dashboard/lims/fair-export' },
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
        {isOpen && item.children && (
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
          v2.0.0 • 16/16 Sessions ✓
        </div>
      </div>
    </div>
  )
}
