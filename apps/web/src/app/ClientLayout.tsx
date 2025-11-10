"use client";

import { AuthProvider } from '@/contexts/AuthContext';
import { Providers } from '@/components/providers';

export default function ClientLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <Providers>
        {children}
      </Providers>
    </AuthProvider>
  );
}
