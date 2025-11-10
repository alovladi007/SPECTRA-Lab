'use client'

import { useState, useEffect } from 'react'
import {
  Users, Plus, Search, Shield, CheckCircle2, XCircle,
  Edit2, Trash2, Eye, UserCheck, UserX, Key, Clock
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'

interface User {
  id: string
  username: string
  email: string
  firstName: string
  lastName: string
  role: 'admin' | 'lab_manager' | 'scientist' | 'technician' | 'viewer'
  status: 'active' | 'inactive' | 'locked'
  department: string
  lastLogin: string
  createdAt: string
  permissions: string[]
}

const rolePermissions = {
  admin: ['all_access', 'user_management', 'system_config', 'data_access', 'instrument_control'],
  lab_manager: ['data_access', 'instrument_control', 'experiment_management', 'report_generation'],
  scientist: ['data_access', 'experiment_management', 'report_generation'],
  technician: ['instrument_control', 'data_entry', 'equipment_maintenance'],
  viewer: ['data_access_read_only']
}

function generateMockUsers(): User[] {
  return [
    {
      id: 'USR-001',
      username: 'admin',
      email: 'admin@spectralab.com',
      firstName: 'System',
      lastName: 'Administrator',
      role: 'admin',
      status: 'active',
      department: 'IT',
      lastLogin: '2024-11-09 10:30',
      createdAt: '2023-01-15',
      permissions: rolePermissions.admin
    },
    {
      id: 'USR-002',
      username: 'jsmith',
      email: 'j.smith@spectralab.com',
      firstName: 'John',
      lastName: 'Smith',
      role: 'lab_manager',
      status: 'active',
      department: 'Electrical Characterization',
      lastLogin: '2024-11-09 09:15',
      createdAt: '2023-03-20',
      permissions: rolePermissions.lab_manager
    },
    {
      id: 'USR-003',
      username: 'sjohnson',
      email: 's.johnson@spectralab.com',
      firstName: 'Sarah',
      lastName: 'Johnson',
      role: 'scientist',
      status: 'active',
      department: 'Optical Analysis',
      lastLogin: '2024-11-08 16:45',
      createdAt: '2023-05-10',
      permissions: rolePermissions.scientist
    },
    {
      id: 'USR-004',
      username: 'mchen',
      email: 'm.chen@spectralab.com',
      firstName: 'Michael',
      lastName: 'Chen',
      role: 'scientist',
      status: 'active',
      department: 'Chemical Analysis',
      lastLogin: '2024-11-09 08:00',
      createdAt: '2023-06-15',
      permissions: rolePermissions.scientist
    },
    {
      id: 'USR-005',
      username: 'dwilson',
      email: 'd.wilson@spectralab.com',
      firstName: 'David',
      lastName: 'Wilson',
      role: 'technician',
      status: 'active',
      department: 'Structural Analysis',
      lastLogin: '2024-11-09 07:30',
      createdAt: '2023-07-01',
      permissions: rolePermissions.technician
    },
    {
      id: 'USR-006',
      username: 'ebrown',
      email: 'e.brown@spectralab.com',
      firstName: 'Emily',
      lastName: 'Brown',
      role: 'technician',
      status: 'active',
      department: 'Equipment Maintenance',
      lastLogin: '2024-11-09 06:45',
      createdAt: '2023-08-12',
      permissions: rolePermissions.technician
    },
    {
      id: 'USR-007',
      username: 'rgarcia',
      email: 'r.garcia@spectralab.com',
      firstName: 'Robert',
      lastName: 'Garcia',
      role: 'viewer',
      status: 'active',
      department: 'Management',
      lastLogin: '2024-11-07 14:20',
      createdAt: '2023-09-05',
      permissions: rolePermissions.viewer
    },
    {
      id: 'USR-008',
      username: 'ljones',
      email: 'l.jones@spectralab.com',
      firstName: 'Lisa',
      lastName: 'Jones',
      role: 'scientist',
      status: 'inactive',
      department: 'Process Simulation',
      lastLogin: '2024-10-15 11:30',
      createdAt: '2023-04-18',
      permissions: rolePermissions.scientist
    },
    {
      id: 'USR-009',
      username: 'tmartinez',
      email: 't.martinez@spectralab.com',
      firstName: 'Thomas',
      lastName: 'Martinez',
      role: 'lab_manager',
      status: 'active',
      department: 'LIMS Management',
      lastLogin: '2024-11-09 09:00',
      createdAt: '2023-02-28',
      permissions: rolePermissions.lab_manager
    },
    {
      id: 'USR-010',
      username: 'kanderson',
      email: 'k.anderson@spectralab.com',
      firstName: 'Karen',
      lastName: 'Anderson',
      role: 'technician',
      status: 'locked',
      department: 'Sample Preparation',
      lastLogin: '2024-09-20 10:15',
      createdAt: '2023-10-01',
      permissions: rolePermissions.technician
    }
  ]
}

export default function UsersPage() {
  const [users, setUsers] = useState<User[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [filterRole, setFilterRole] = useState<string>('all')
  const [filterStatus, setFilterStatus] = useState<string>('all')
  const [selectedUser, setSelectedUser] = useState<User | null>(null)
  const [showNewDialog, setShowNewDialog] = useState(false)
  const [showDetailsDialog, setShowDetailsDialog] = useState(false)
  const [showEditDialog, setShowEditDialog] = useState(false)

  const [newUser, setNewUser] = useState({
    username: '',
    email: '',
    firstName: '',
    lastName: '',
    role: 'viewer' as User['role'],
    department: ''
  })

  // Generate mock data on client side only
  useEffect(() => {
    setUsers(generateMockUsers())
  }, [])

  // Filter users
  const filteredUsers = users.filter(user => {
    const matchesSearch =
      user.username.toLowerCase().includes(searchQuery.toLowerCase()) ||
      user.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
      user.firstName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      user.lastName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      user.department.toLowerCase().includes(searchQuery.toLowerCase())

    const matchesRole = filterRole === 'all' || user.role === filterRole
    const matchesStatus = filterStatus === 'all' || user.status === filterStatus

    return matchesSearch && matchesRole && matchesStatus
  })

  // Statistics
  const stats = {
    total: users.length,
    active: users.filter(u => u.status === 'active').length,
    inactive: users.filter(u => u.status === 'inactive').length,
    locked: users.filter(u => u.status === 'locked').length,
    admins: users.filter(u => u.role === 'admin').length
  }

  const handleCreateUser = () => {
    if (!newUser.username || !newUser.email || !newUser.firstName || !newUser.lastName) {
      alert('Please fill in all required fields')
      return
    }

    const user: User = {
      id: `USR-${String(users.length + 1).padStart(3, '0')}`,
      username: newUser.username,
      email: newUser.email,
      firstName: newUser.firstName,
      lastName: newUser.lastName,
      role: newUser.role,
      status: 'active',
      department: newUser.department,
      lastLogin: 'Never',
      createdAt: new Date().toISOString().split('T')[0],
      permissions: rolePermissions[newUser.role]
    }

    setUsers([user, ...users])
    setShowNewDialog(false)
    setNewUser({
      username: '',
      email: '',
      firstName: '',
      lastName: '',
      role: 'viewer',
      department: ''
    })
  }

  const handleUpdateUser = () => {
    if (!selectedUser) return

    setUsers(users.map(user =>
      user.id === selectedUser.id ? { ...selectedUser, permissions: rolePermissions[selectedUser.role] } : user
    ))
    setShowEditDialog(false)
    setSelectedUser(null)
  }

  const handleDeleteUser = (id: string) => {
    if (confirm('Are you sure you want to delete this user?')) {
      setUsers(users.filter(user => user.id !== id))
    }
  }

  const handleToggleStatus = (user: User) => {
    setUsers(users.map(u => {
      if (u.id === user.id) {
        return { ...u, status: u.status === 'active' ? 'inactive' : 'active' as User['status'] }
      }
      return u
    }))
  }

  const handleUnlockUser = (user: User) => {
    setUsers(users.map(u => {
      if (u.id === user.id) {
        return { ...u, status: 'active' as const }
      }
      return u
    }))
  }

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'admin': return 'bg-purple-100 text-purple-800'
      case 'lab_manager': return 'bg-blue-100 text-blue-800'
      case 'scientist': return 'bg-green-100 text-green-800'
      case 'technician': return 'bg-yellow-100 text-yellow-800'
      case 'viewer': return 'bg-gray-100 text-gray-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800'
      case 'inactive': return 'bg-gray-100 text-gray-800'
      case 'locked': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle2 className="w-4 h-4" />
      case 'inactive': return <XCircle className="w-4 h-4" />
      case 'locked': return <Key className="w-4 h-4" />
      default: return <Clock className="w-4 h-4" />
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-xl flex items-center justify-center">
            <Users className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Users & Roles</h1>
            <p className="text-gray-600 mt-1">Manage user accounts and permissions</p>
          </div>
        </div>

        <Button onClick={() => setShowNewDialog(true)}>
          <Plus className="w-5 h-5 mr-2" />
          Add User
        </Button>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-5 gap-4 mb-6">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Users</p>
              <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
            </div>
            <Users className="w-8 h-8 text-gray-400" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Active</p>
              <p className="text-2xl font-bold text-green-600">{stats.active}</p>
            </div>
            <CheckCircle2 className="w-8 h-8 text-green-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Inactive</p>
              <p className="text-2xl font-bold text-gray-600">{stats.inactive}</p>
            </div>
            <XCircle className="w-8 h-8 text-gray-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Locked</p>
              <p className="text-2xl font-bold text-red-600">{stats.locked}</p>
            </div>
            <Key className="w-8 h-8 text-red-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Administrators</p>
              <p className="text-2xl font-bold text-purple-600">{stats.admins}</p>
            </div>
            <Shield className="w-8 h-8 text-purple-500" />
          </div>
        </Card>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 mb-6">
        <div className="grid grid-cols-3 gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <Input
              placeholder="Search users..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
          <Select value={filterRole} onValueChange={setFilterRole}>
            <SelectTrigger>
              <SelectValue placeholder="Filter by role" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Roles</SelectItem>
              <SelectItem value="admin">Administrator</SelectItem>
              <SelectItem value="lab_manager">Lab Manager</SelectItem>
              <SelectItem value="scientist">Scientist</SelectItem>
              <SelectItem value="technician">Technician</SelectItem>
              <SelectItem value="viewer">Viewer</SelectItem>
            </SelectContent>
          </Select>
          <Select value={filterStatus} onValueChange={setFilterStatus}>
            <SelectTrigger>
              <SelectValue placeholder="Filter by status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Statuses</SelectItem>
              <SelectItem value="active">Active</SelectItem>
              <SelectItem value="inactive">Inactive</SelectItem>
              <SelectItem value="locked">Locked</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Users Grid */}
      <div className="grid grid-cols-1 gap-4">
        {filteredUsers.map((user) => (
          <Card key={user.id} className="p-6 hover:shadow-lg transition-shadow">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-3">
                  <h3 className="text-lg font-semibold text-gray-900">
                    {user.firstName} {user.lastName}
                  </h3>
                  <Badge className={getRoleColor(user.role)}>
                    <Shield className="w-3 h-3 mr-1" />
                    {user.role.replace('_', ' ')}
                  </Badge>
                  <Badge className={getStatusColor(user.status)}>
                    {getStatusIcon(user.status)}
                    <span className="ml-1">{user.status}</span>
                  </Badge>
                </div>

                <div className="grid grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-gray-600">User ID</p>
                    <p className="font-medium text-gray-900">{user.id}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Username</p>
                    <p className="font-medium text-gray-900">{user.username}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Email</p>
                    <p className="font-medium text-gray-900">{user.email}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Department</p>
                    <p className="font-medium text-gray-900">{user.department}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Created</p>
                    <p className="font-medium text-gray-900">{user.createdAt}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Last Login</p>
                    <p className="font-medium text-gray-900">{user.lastLogin}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Permissions</p>
                    <p className="font-medium text-gray-900">{user.permissions.length} granted</p>
                  </div>
                </div>
              </div>

              <div className="flex gap-2 ml-4">
                {user.status === 'locked' && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleUnlockUser(user)}
                  >
                    <Key className="w-4 h-4 mr-1" />
                    Unlock
                  </Button>
                )}
                {user.status !== 'locked' && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleToggleStatus(user)}
                  >
                    {user.status === 'active' ? <UserX className="w-4 h-4" /> : <UserCheck className="w-4 h-4" />}
                  </Button>
                )}
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    setSelectedUser(user)
                    setShowDetailsDialog(true)
                  }}
                >
                  <Eye className="w-4 h-4" />
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    setSelectedUser(user)
                    setShowEditDialog(true)
                  }}
                >
                  <Edit2 className="w-4 h-4" />
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleDeleteUser(user.id)}
                  disabled={user.role === 'admin'}
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {filteredUsers.length === 0 && (
        <div className="text-center py-12">
          <Users className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No users found</h3>
          <p className="text-gray-600">Try adjusting your search or filters</p>
        </div>
      )}

      {/* Add User Dialog */}
      <Dialog open={showNewDialog} onOpenChange={setShowNewDialog}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Add New User</DialogTitle>
            <DialogDescription>Enter the details for the new user account</DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Username *</Label>
                <Input
                  value={newUser.username}
                  onChange={(e) => setNewUser({ ...newUser, username: e.target.value })}
                  placeholder="e.g., jsmith"
                />
              </div>
              <div>
                <Label>Email *</Label>
                <Input
                  type="email"
                  value={newUser.email}
                  onChange={(e) => setNewUser({ ...newUser, email: e.target.value })}
                  placeholder="user@spectralab.com"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>First Name *</Label>
                <Input
                  value={newUser.firstName}
                  onChange={(e) => setNewUser({ ...newUser, firstName: e.target.value })}
                  placeholder="John"
                />
              </div>
              <div>
                <Label>Last Name *</Label>
                <Input
                  value={newUser.lastName}
                  onChange={(e) => setNewUser({ ...newUser, lastName: e.target.value })}
                  placeholder="Smith"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Role *</Label>
                <Select
                  value={newUser.role}
                  onValueChange={(value: User['role']) => setNewUser({ ...newUser, role: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="viewer">Viewer</SelectItem>
                    <SelectItem value="technician">Technician</SelectItem>
                    <SelectItem value="scientist">Scientist</SelectItem>
                    <SelectItem value="lab_manager">Lab Manager</SelectItem>
                    <SelectItem value="admin">Administrator</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label>Department</Label>
                <Input
                  value={newUser.department}
                  onChange={(e) => setNewUser({ ...newUser, department: e.target.value })}
                  placeholder="e.g., Electrical Characterization"
                />
              </div>
            </div>

            <div className="p-4 bg-blue-50 rounded-lg">
              <p className="text-sm font-medium text-blue-900 mb-2">Role Permissions:</p>
              <ul className="text-sm text-blue-800 space-y-1">
                {rolePermissions[newUser.role].map((perm, idx) => (
                  <li key={idx}>• {perm.replace('_', ' ')}</li>
                ))}
              </ul>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setShowNewDialog(false)}>Cancel</Button>
            <Button onClick={handleCreateUser}>Add User</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Details Dialog */}
      <Dialog open={showDetailsDialog} onOpenChange={setShowDetailsDialog}>
        <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>User Details</DialogTitle>
          </DialogHeader>

          {selectedUser && (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <h2 className="text-2xl font-bold text-gray-900">
                  {selectedUser.firstName} {selectedUser.lastName}
                </h2>
                <Badge className={getRoleColor(selectedUser.role)}>
                  <Shield className="w-3 h-3 mr-1" />
                  {selectedUser.role.replace('_', ' ')}
                </Badge>
                <Badge className={getStatusColor(selectedUser.status)}>
                  {getStatusIcon(selectedUser.status)}
                  <span className="ml-1">{selectedUser.status}</span>
                </Badge>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-3">
                  <div>
                    <Label className="text-gray-600">User ID</Label>
                    <p className="font-medium">{selectedUser.id}</p>
                  </div>
                  <div>
                    <Label className="text-gray-600">Username</Label>
                    <p className="font-medium">{selectedUser.username}</p>
                  </div>
                  <div>
                    <Label className="text-gray-600">Email</Label>
                    <p className="font-medium">{selectedUser.email}</p>
                  </div>
                  <div>
                    <Label className="text-gray-600">Department</Label>
                    <p className="font-medium">{selectedUser.department}</p>
                  </div>
                </div>

                <div className="space-y-3">
                  <div>
                    <Label className="text-gray-600">Role</Label>
                    <p className="font-medium">{selectedUser.role.replace('_', ' ')}</p>
                  </div>
                  <div>
                    <Label className="text-gray-600">Status</Label>
                    <p className="font-medium">{selectedUser.status}</p>
                  </div>
                  <div>
                    <Label className="text-gray-600">Created</Label>
                    <p className="font-medium">{selectedUser.createdAt}</p>
                  </div>
                  <div>
                    <Label className="text-gray-600">Last Login</Label>
                    <p className="font-medium">{selectedUser.lastLogin}</p>
                  </div>
                </div>
              </div>

              <div>
                <Label className="text-gray-600">Permissions</Label>
                <div className="mt-2 p-4 bg-gray-50 rounded-lg">
                  <ul className="space-y-2">
                    {selectedUser.permissions.map((perm, idx) => (
                      <li key={idx} className="flex items-center gap-2 text-sm">
                        <CheckCircle2 className="w-4 h-4 text-green-600" />
                        <span className="font-medium">{perm.replace('_', ' ')}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}

          <DialogFooter>
            <Button onClick={() => setShowDetailsDialog(false)}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog open={showEditDialog} onOpenChange={setShowEditDialog}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Edit User</DialogTitle>
          </DialogHeader>

          {selectedUser && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>First Name</Label>
                  <Input
                    value={selectedUser.firstName}
                    onChange={(e) => setSelectedUser({ ...selectedUser, firstName: e.target.value })}
                  />
                </div>
                <div>
                  <Label>Last Name</Label>
                  <Input
                    value={selectedUser.lastName}
                    onChange={(e) => setSelectedUser({ ...selectedUser, lastName: e.target.value })}
                  />
                </div>
              </div>

              <div>
                <Label>Email</Label>
                <Input
                  type="email"
                  value={selectedUser.email}
                  onChange={(e) => setSelectedUser({ ...selectedUser, email: e.target.value })}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Role</Label>
                  <Select
                    value={selectedUser.role}
                    onValueChange={(value: User['role']) => setSelectedUser({ ...selectedUser, role: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="viewer">Viewer</SelectItem>
                      <SelectItem value="technician">Technician</SelectItem>
                      <SelectItem value="scientist">Scientist</SelectItem>
                      <SelectItem value="lab_manager">Lab Manager</SelectItem>
                      <SelectItem value="admin">Administrator</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Status</Label>
                  <Select
                    value={selectedUser.status}
                    onValueChange={(value: User['status']) => setSelectedUser({ ...selectedUser, status: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="active">Active</SelectItem>
                      <SelectItem value="inactive">Inactive</SelectItem>
                      <SelectItem value="locked">Locked</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div>
                <Label>Department</Label>
                <Input
                  value={selectedUser.department}
                  onChange={(e) => setSelectedUser({ ...selectedUser, department: e.target.value })}
                />
              </div>

              <div className="p-4 bg-blue-50 rounded-lg">
                <p className="text-sm font-medium text-blue-900 mb-2">Role Permissions:</p>
                <ul className="text-sm text-blue-800 space-y-1">
                  {rolePermissions[selectedUser.role].map((perm, idx) => (
                    <li key={idx}>• {perm.replace('_', ' ')}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setShowEditDialog(false)}>Cancel</Button>
            <Button onClick={handleUpdateUser}>Save Changes</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
