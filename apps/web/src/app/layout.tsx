import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import ClientLayout from './ClientLayout'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'SPECTRA-Lab | Enterprise Semiconductor Characterization Platform',
  description: 'Comprehensive semiconductor characterization with 26 methods, SPC, and ML/VM capabilities. Enterprise-grade platform for electrical, optical, structural, and chemical analysis.',
  keywords: ['semiconductor', 'characterization', 'metrology', 'SPC', 'machine learning', 'virtual metrology'],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ClientLayout>{children}</ClientLayout>
      </body>
    </html>
  )
}
