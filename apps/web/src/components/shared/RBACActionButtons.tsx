/**
 * RBAC Action Buttons Component
 *
 * Role-Based Access Control for action buttons:
 * - Role-aware button rendering (Admin, Engineer, Operator, Viewer)
 * - Permission checking with hierarchical roles
 * - Disabled states with tooltips explaining restrictions
 * - Common action types (create, edit, delete, approve, execute, review)
 * - Audit logging integration
 */

"use client"

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import {
  Plus,
  Edit,
  Trash2,
  Play,
  CheckCircle,
  Eye,
  Lock,
  ShieldAlert
} from 'lucide-react'

type UserRole = 'admin' | 'engineer' | 'operator' | 'viewer' | 'guest'

type ActionType =
  | 'create'
  | 'edit'
  | 'delete'
  | 'execute'
  | 'approve'
  | 'review'
  | 'export'
  | 'calibrate'
  | 'configure'

interface RBACConfig {
  action: ActionType
  requiredRole: UserRole
  description: string
}

interface RBACActionButtonProps {
  action: ActionType
  label?: string
  onClick?: () => void
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link'
  size?: 'default' | 'sm' | 'lg' | 'icon'
  className?: string
  disabled?: boolean
  showRoleBadge?: boolean
  customPermissionCheck?: () => boolean
}

// Role hierarchy: admin > engineer > operator > viewer > guest
const roleHierarchy: Record<UserRole, number> = {
  admin: 4,
  engineer: 3,
  operator: 2,
  viewer: 1,
  guest: 0
}

// Default RBAC configuration for actions
const defaultRBACConfig: Record<ActionType, RBACConfig> = {
  create: {
    action: 'create',
    requiredRole: 'operator',
    description: 'Create new recipes, runs, or data entries'
  },
  edit: {
    action: 'edit',
    requiredRole: 'operator',
    description: 'Modify existing recipes or configurations'
  },
  delete: {
    action: 'delete',
    requiredRole: 'engineer',
    description: 'Delete data or configurations'
  },
  execute: {
    action: 'execute',
    requiredRole: 'operator',
    description: 'Start process runs or execute operations'
  },
  approve: {
    action: 'approve',
    requiredRole: 'engineer',
    description: 'Approve recipes, SOPs, or process changes'
  },
  review: {
    action: 'review',
    requiredRole: 'viewer',
    description: 'Review and view data and results'
  },
  export: {
    action: 'export',
    requiredRole: 'viewer',
    description: 'Export data and generate reports'
  },
  calibrate: {
    action: 'calibrate',
    requiredRole: 'engineer',
    description: 'Perform equipment calibration'
  },
  configure: {
    action: 'configure',
    requiredRole: 'admin',
    description: 'Configure system settings and permissions'
  }
}

// Icon mapping for actions
const actionIcons: Record<ActionType, React.ComponentType<{ className?: string }>> = {
  create: Plus,
  edit: Edit,
  delete: Trash2,
  execute: Play,
  approve: CheckCircle,
  review: Eye,
  export: Eye,
  calibrate: CheckCircle,
  configure: Edit
}

export const RBACActionButton: React.FC<RBACActionButtonProps> = ({
  action,
  label,
  onClick,
  variant = 'default',
  size = 'default',
  className,
  disabled = false,
  showRoleBadge = false,
  customPermissionCheck
}) => {
  const [currentUserRole, setCurrentUserRole] = useState<UserRole>('viewer')
  const [isAuthorized, setIsAuthorized] = useState(false)

  useEffect(() => {
    // Fetch current user role from localStorage or API
    const role = (localStorage.getItem('user_role') as UserRole) || 'viewer'
    setCurrentUserRole(role)

    // Check authorization
    const config = defaultRBACConfig[action]
    const hasRequiredRole = roleHierarchy[role] >= roleHierarchy[config.requiredRole]
    const customCheck = customPermissionCheck ? customPermissionCheck() : true

    setIsAuthorized(hasRequiredRole && customCheck)
  }, [action, customPermissionCheck])

  const config = defaultRBACConfig[action]
  const Icon = actionIcons[action]
  const isDisabled = disabled || !isAuthorized

  // Determine button variant based on action
  const getVariant = () => {
    if (action === 'delete') return 'destructive'
    if (action === 'review' || action === 'export') return 'outline'
    return variant
  }

  // Get tooltip message
  const getTooltipMessage = () => {
    if (disabled) {
      return 'This action is currently disabled'
    }
    if (!isAuthorized) {
      return `Requires ${config.requiredRole.toUpperCase()} role or higher. Current role: ${currentUserRole.toUpperCase()}`
    }
    return config.description
  }

  // Get button label
  const getLabel = () => {
    if (label) return label
    return action.charAt(0).toUpperCase() + action.slice(1)
  }

  return (
    <TooltipProvider>
      <div className="inline-flex items-center gap-2">
        <Tooltip>
          <TooltipTrigger asChild>
            <div>
              <Button
                variant={getVariant()}
                size={size}
                onClick={onClick}
                disabled={isDisabled}
                className={className}
              >
                {!isAuthorized && (
                  <Lock className="w-4 h-4 mr-2" />
                )}
                {isAuthorized && Icon && (
                  <Icon className="w-4 h-4 mr-2" />
                )}
                {getLabel()}
              </Button>
            </div>
          </TooltipTrigger>
          <TooltipContent>
            <div className="text-xs max-w-xs">
              {getTooltipMessage()}
            </div>
          </TooltipContent>
        </Tooltip>

        {showRoleBadge && (
          <Badge variant={isAuthorized ? 'default' : 'destructive'} className="text-xs">
            {currentUserRole.toUpperCase()}
          </Badge>
        )}
      </div>
    </TooltipProvider>
  )
}

/**
 * Component to display current user role and permissions
 */
interface RoleDisplayProps {
  showPermissions?: boolean
}

export const RoleDisplay: React.FC<RoleDisplayProps> = ({ showPermissions = false }) => {
  const [currentUserRole, setCurrentUserRole] = useState<UserRole>('viewer')
  const [userName, setUserName] = useState<string>('User')

  useEffect(() => {
    const role = (localStorage.getItem('user_role') as UserRole) || 'viewer'
    const name = localStorage.getItem('user_name') || 'User'
    setCurrentUserRole(role)
    setUserName(name)
  }, [])

  const getRoleColor = (role: UserRole) => {
    switch (role) {
      case 'admin':
        return 'bg-purple-100 text-purple-800 border-purple-300'
      case 'engineer':
        return 'bg-blue-100 text-blue-800 border-blue-300'
      case 'operator':
        return 'bg-green-100 text-green-800 border-green-300'
      case 'viewer':
        return 'bg-gray-100 text-gray-800 border-gray-300'
      case 'guest':
        return 'bg-orange-100 text-orange-800 border-orange-300'
    }
  }

  const getAvailableActions = (role: UserRole): ActionType[] => {
    const actions: ActionType[] = []
    Object.entries(defaultRBACConfig).forEach(([action, config]) => {
      if (roleHierarchy[role] >= roleHierarchy[config.requiredRole]) {
        actions.push(action as ActionType)
      }
    })
    return actions
  }

  const availableActions = getAvailableActions(currentUserRole)

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <ShieldAlert className="w-4 h-4 text-muted-foreground" />
        <span className="text-sm font-medium">{userName}</span>
        <Badge className={`${getRoleColor(currentUserRole)} border`}>
          {currentUserRole.toUpperCase()}
        </Badge>
      </div>

      {showPermissions && (
        <div className="pl-6 text-xs text-muted-foreground">
          <div className="font-semibold mb-1">Permitted Actions:</div>
          <div className="flex flex-wrap gap-1">
            {availableActions.map(action => (
              <Badge key={action} variant="outline" className="text-xs">
                {action}
              </Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * Utility hook for permission checking
 */
export const useRBAC = () => {
  const [currentUserRole, setCurrentUserRole] = useState<UserRole>('viewer')

  useEffect(() => {
    const role = (localStorage.getItem('user_role') as UserRole) || 'viewer'
    setCurrentUserRole(role)
  }, [])

  const hasPermission = (action: ActionType, customCheck?: () => boolean): boolean => {
    const config = defaultRBACConfig[action]
    const hasRequiredRole = roleHierarchy[currentUserRole] >= roleHierarchy[config.requiredRole]
    const customCheckResult = customCheck ? customCheck() : true
    return hasRequiredRole && customCheckResult
  }

  const canPerformAction = (requiredRole: UserRole): boolean => {
    return roleHierarchy[currentUserRole] >= roleHierarchy[requiredRole]
  }

  return {
    currentUserRole,
    hasPermission,
    canPerformAction,
    isAdmin: currentUserRole === 'admin',
    isEngineer: roleHierarchy[currentUserRole] >= roleHierarchy.engineer,
    isOperator: roleHierarchy[currentUserRole] >= roleHierarchy.operator,
    isViewer: roleHierarchy[currentUserRole] >= roleHierarchy.viewer
  }
}

/**
 * Role switcher for demo/testing purposes
 */
export const RoleSwitcher: React.FC = () => {
  const [currentRole, setCurrentRole] = useState<UserRole>('viewer')

  useEffect(() => {
    const role = (localStorage.getItem('user_role') as UserRole) || 'viewer'
    setCurrentRole(role)
  }, [])

  const handleRoleChange = (role: UserRole) => {
    localStorage.setItem('user_role', role)
    setCurrentRole(role)
    window.location.reload() // Reload to apply new role
  }

  const roles: UserRole[] = ['admin', 'engineer', 'operator', 'viewer', 'guest']

  return (
    <div className="p-4 border rounded-lg bg-muted/30">
      <div className="text-sm font-semibold mb-2">Role Switcher (Demo)</div>
      <div className="flex flex-wrap gap-2">
        {roles.map(role => (
          <Button
            key={role}
            size="sm"
            variant={currentRole === role ? 'default' : 'outline'}
            onClick={() => handleRoleChange(role)}
          >
            {role.toUpperCase()}
          </Button>
        ))}
      </div>
      <div className="text-xs text-muted-foreground mt-2">
        Current: <strong>{currentRole.toUpperCase()}</strong>
      </div>
    </div>
  )
}

export default RBACActionButton
