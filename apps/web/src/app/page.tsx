'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'

export default function HomePage() {
  const router = useRouter()

  useEffect(() => {
    router.replace('/dashboard')
  }, [router])

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-500 to-purple-600">
      <div className="text-center">
        <div className="w-20 h-20 bg-white rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-2xl">
          <svg className="w-12 h-12 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        <h1 className="text-4xl font-bold text-white mb-2">SPECTRA-Lab</h1>
        <p className="text-blue-100 mb-4">Enterprise Semiconductor Characterization Platform</p>
        <div className="w-48 h-1 bg-white/20 rounded-full overflow-hidden mx-auto">
          <div className="h-full bg-white rounded-full w-3/4 animate-pulse"></div>
        </div>
      </div>
    </div>
  )
}
