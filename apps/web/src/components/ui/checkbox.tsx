import React from 'react'

interface CheckboxProps {
  checked: boolean
  onCheckedChange: (checked: boolean) => void
  className?: string
}

export const Checkbox = ({ checked, onCheckedChange, className = '' }: CheckboxProps) => {
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={checked}
      onClick={() => onCheckedChange(!checked)}
      className={`h-4 w-4 rounded border border-gray-300 flex items-center justify-center transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
        checked ? 'bg-blue-600 border-blue-600' : 'bg-white'
      } ${className}`}
    >
      {checked && (
        <svg
          className="h-3 w-3 text-white"
          fill="none"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path d="M5 13l4 4L19 7" />
        </svg>
      )}
    </button>
  )
}
