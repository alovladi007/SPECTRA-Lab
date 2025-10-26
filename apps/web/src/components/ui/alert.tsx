import React from 'react'

interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'destructive'
}

export const Alert = ({ children, className = '', variant = 'default', ...props }: AlertProps) => {
  const variantClasses = {
    default: 'bg-blue-50 border-blue-200 text-blue-900',
    destructive: 'bg-red-50 border-red-200 text-red-900'
  }

  return (
    <div
      className={`flex items-start p-4 border rounded-lg ${variantClasses[variant]} ${className}`}
      {...props}
    >
      {children}
    </div>
  )
}

export const AlertDescription = ({ children, className = '', ...props }: React.HTMLAttributes<HTMLParagraphElement>) => {
  return (
    <p className={`text-sm ${className}`} {...props}>
      {children}
    </p>
  )
}
