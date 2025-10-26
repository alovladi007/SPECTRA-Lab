'use client'

import React from 'react'

interface SelectProps {
  value?: string
  onValueChange?: (value: string) => void
  disabled?: boolean
  children: React.ReactNode
}

interface SelectItemProps {
  value: string
  children: React.ReactNode
}

const SelectContext = React.createContext<{
  value?: string
  onValueChange?: (value: string) => void
  open: boolean
  setOpen: (open: boolean) => void
} | null>(null)

export const Select = ({ value, onValueChange, disabled, children }: SelectProps) => {
  const [open, setOpen] = React.useState(false)

  return (
    <SelectContext.Provider value={{ value, onValueChange, open, setOpen }}>
      <div className="relative">
        {React.Children.map(children, (child) => {
          if (React.isValidElement(child) && child.type === SelectTrigger) {
            return React.cloneElement(child, { disabled } as any)
          }
          return child
        })}
      </div>
    </SelectContext.Provider>
  )
}

export const SelectTrigger = ({ children, disabled }: { children?: React.ReactNode; disabled?: boolean }) => {
  const context = React.useContext(SelectContext)

  return (
    <button
      type="button"
      onClick={() => context?.setOpen(!context.open)}
      disabled={disabled}
      className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-white text-left focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-between"
    >
      {children}
      <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    </button>
  )
}

export const SelectValue = () => {
  const context = React.useContext(SelectContext)
  return <span>{context?.value || 'Select...'}</span>
}

export const SelectContent = ({ children }: { children: React.ReactNode }) => {
  const context = React.useContext(SelectContext)

  if (!context?.open) return null

  return (
    <>
      <div
        className="fixed inset-0 z-40"
        onClick={() => context.setOpen(false)}
      />
      <div className="absolute z-50 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-auto">
        {children}
      </div>
    </>
  )
}

export const SelectItem = ({ value, children }: SelectItemProps) => {
  const context = React.useContext(SelectContext)

  return (
    <div
      onClick={() => {
        context?.onValueChange?.(value)
        context?.setOpen(false)
      }}
      className={`px-3 py-2 cursor-pointer hover:bg-gray-100 ${
        context?.value === value ? 'bg-blue-50 text-blue-600' : ''
      }`}
    >
      {children}
    </div>
  )
}
