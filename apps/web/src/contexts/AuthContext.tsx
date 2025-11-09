"use client";

import React, { createContext, useContext, useState, useEffect } from 'react';

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'pi' | 'engineer' | 'technician' | 'viewer';
  organization_id: string;
}

interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

interface AuthContextType {
  user: User | null;
  tokens: AuthTokens | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [tokens, setTokens] = useState<AuthTokens | null>(null);
  const [loading, setLoading] = useState(true);

  // Load tokens from localStorage on mount
  useEffect(() => {
    const storedTokens = localStorage.getItem('auth_tokens');
    const storedUser = localStorage.getItem('auth_user');

    if (storedTokens && storedUser) {
      try {
        setTokens(JSON.parse(storedTokens));
        setUser(JSON.parse(storedUser));
      } catch (e) {
        console.error('Failed to parse stored auth data:', e);
        localStorage.removeItem('auth_tokens');
        localStorage.removeItem('auth_user');
      }
    }

    setLoading(false);
  }, []);

  const login = async (email: string, password: string) => {
    try {
      // Call LIMS auth endpoint
      const response = await fetch('http://localhost:8002/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          username: email,
          password: password,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Login failed');
      }

      const authData = await response.json();

      // Decode JWT to get user info (basic decode, not verification)
      const payload = JSON.parse(atob(authData.access_token.split('.')[1]));

      const userData: User = {
        id: payload.sub,
        email: payload.email,
        name: payload.email.split('@')[0], // Use email prefix as name for now
        role: payload.role,
        organization_id: payload.org_id,
      };

      setTokens(authData);
      setUser(userData);

      // Persist to localStorage
      localStorage.setItem('auth_tokens', JSON.stringify(authData));
      localStorage.setItem('auth_user', JSON.stringify(userData));
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  };

  const logout = () => {
    setUser(null);
    setTokens(null);
    localStorage.removeItem('auth_tokens');
    localStorage.removeItem('auth_user');
  };

  const value = {
    user,
    tokens,
    loading,
    login,
    logout,
    isAuthenticated: !!user && !!tokens,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
