'use client'

import * as React from "react"

interface TooltipProviderProps {
  children?: React.ReactNode
}

const TooltipProvider = ({ children }: TooltipProviderProps) => {
  return <>{children}</>
}

interface TooltipProps {
  children?: React.ReactNode
}

const Tooltip = ({ children }: TooltipProps) => {
  return <div className="relative inline-block group">{children}</div>
}

const TooltipTrigger = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ children, ...props }, ref) => (
  <div ref={ref} {...props}>
    {children}
  </div>
))
TooltipTrigger.displayName = "TooltipTrigger"

interface TooltipContentProps extends React.HTMLAttributes<HTMLDivElement> {
  children?: React.ReactNode
  side?: 'top' | 'right' | 'bottom' | 'left'
}

const TooltipContent = React.forwardRef<HTMLDivElement, TooltipContentProps>(
  ({ children, className, side = 'top', ...props }, ref) => {
    const positionClasses = {
      top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
      bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
      left: 'right-full top-1/2 -translate-y-1/2 mr-2',
      right: 'left-full top-1/2 -translate-y-1/2 ml-2'
    }

    return (
      <div
        ref={ref}
        className={`absolute z-50 px-3 py-1.5 text-sm text-white bg-gray-900 rounded-md opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-opacity duration-200 whitespace-nowrap ${positionClasses[side]} ${className || ''}`}
        {...props}
      >
        {children}
      </div>
    )
  }
)
TooltipContent.displayName = "TooltipContent"

export {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
  TooltipProvider,
}
