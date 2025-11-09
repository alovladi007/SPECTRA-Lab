"use client";

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';

export default function ProtectedRoute({
  children,
  requiredRoles,
}: {
  children: React.ReactNode;
  requiredRoles?: ('admin' | 'pi' | 'engineer' | 'technician' | 'viewer')[];
}) {
  const { isAuthenticated, loading, user } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading) {
      if (!isAuthenticated) {
        router.push('/login');
      } else if (requiredRoles && user && !requiredRoles.includes(user.role)) {
        // User doesn't have required role
        router.push('/dashboard'); // Redirect to dashboard instead of showing error
      }
    }
  }, [isAuthenticated, loading, user, requiredRoles, router]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated || (requiredRoles && user && !requiredRoles.includes(user.role))) {
    return null; // Will redirect in useEffect
  }

  return <>{children}</>;
}
