'use client'

import React from 'react'

interface TabsProps {
  defaultValue?: string
  value?: string
  onValueChange?: (value: string) => void
  className?: string
  children: React.ReactNode
}

const TabsContext = React.createContext<{
  value: string
  onValueChange: (value: string) => void
} | null>(null)

export const Tabs = ({ defaultValue, value: controlledValue, onValueChange, className = '', children }: TabsProps) => {
  const [internalValue, setInternalValue] = React.useState(defaultValue || '')
  const value = controlledValue !== undefined ? controlledValue : internalValue
  const handleValueChange = controlledValue !== undefined ? (onValueChange || (() => {})) : setInternalValue

  return (
    <TabsContext.Provider value={{ value, onValueChange: handleValueChange }}>
      <div className={className}>
        {children}
      </div>
    </TabsContext.Provider>
  )
}

export const TabsList = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => {
  return (
    <div className={`inline-flex h-10 items-center justify-center rounded-lg bg-gray-100 p-1 ${className}`}>
      {children}
    </div>
  )
}

export const TabsTrigger = ({ value, children }: { value: string; children: React.ReactNode }) => {
  const context = React.useContext(TabsContext)
  const isActive = context?.value === value

  return (
    <button
      type="button"
      onClick={(e) => {
        e.preventDefault()
        e.stopPropagation()
        console.log('Tab clicked:', value, 'Current:', context?.value)
        context?.onValueChange(value)
      }}
      className={`inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 w-full ${
        isActive
          ? 'bg-white text-gray-900 shadow-sm border border-gray-200'
          : 'text-gray-500 bg-transparent hover:text-gray-900 hover:bg-gray-50'
      }`}
    >
      {children}
    </button>
  )
}

export const TabsContent = ({ value, children }: { value: string; children: React.ReactNode }) => {
  const context = React.useContext(TabsContext)

  if (context?.value !== value) return null

  return <div>{children}</div>
}
