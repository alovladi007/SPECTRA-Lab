'use client'

import React, { useState } from 'react'
import { Search, Bell, User, Settings, LogOut, HelpCircle, Command } from 'lucide-react'

export function AppHeader() {
  const [showNotifications, setShowNotifications] = useState(false)
  const [showUserMenu, setShowUserMenu] = useState(false)

  const notifications = [
    { id: 1, title: 'SPC Alert', message: 'Sheet resistance out of control limits', time: '2 min ago', unread: true, type: 'warning' },
    { id: 2, title: 'ML Model Updated', message: 'Thickness VM model v2.1 deployed', time: '15 min ago', unread: true, type: 'success' },
    { id: 3, title: 'Analysis Complete', message: 'XRD analysis for sample WF-042 completed', time: '1 hour ago', unread: false, type: 'info' },
    { id: 4, title: 'Drift Detected', message: 'Model drift detected on VM_Etch_Rate', time: '2 hours ago', unread: false, type: 'warning' },
  ]

  const unreadCount = notifications.filter(n => n.unread).length

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Search */}
        <div className="flex-1 max-w-2xl">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search samples, experiments, methods... (⌘K)"
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <kbd className="absolute right-3 top-1/2 -translate-y-1/2 px-2 py-1 text-xs bg-gray-100 border border-gray-300 rounded">
              ⌘K
            </kbd>
          </div>
        </div>

        {/* Right Side */}
        <div className="flex items-center gap-4 ml-6">
          {/* Quick Actions */}
          <button className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors">
            <Command className="w-5 h-5" />
          </button>

          <button className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors">
            <HelpCircle className="w-5 h-5" />
          </button>

          {/* Notifications */}
          <div className="relative">
            <button
              onClick={() => {
                setShowNotifications(!showNotifications)
                setShowUserMenu(false)
              }}
              className="relative p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <Bell className="w-5 h-5" />
              {unreadCount > 0 && (
                <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs font-bold rounded-full flex items-center justify-center">
                  {unreadCount}
                </span>
              )}
            </button>

            {/* Notifications Dropdown */}
            {showNotifications && (
              <div className="absolute right-0 mt-2 w-96 bg-white rounded-lg shadow-lg border border-gray-200 z-50">
                <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
                  <h3 className="font-semibold text-gray-900">Notifications</h3>
                  <button className="text-xs text-blue-600 hover:text-blue-700">Mark all as read</button>
                </div>
                <div className="max-h-96 overflow-y-auto">
                  {notifications.map((notification) => (
                    <div
                      key={notification.id}
                      className={`px-4 py-3 border-b border-gray-100 hover:bg-gray-50 cursor-pointer ${
                        notification.unread ? 'bg-blue-50/50' : ''
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <div className={`mt-1 w-2 h-2 rounded-full flex-shrink-0 ${
                          notification.type === 'warning' ? 'bg-yellow-500' :
                          notification.type === 'success' ? 'bg-green-500' :
                          'bg-blue-500'
                        }`} />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900">{notification.title}</p>
                          <p className="text-sm text-gray-600 mt-1">{notification.message}</p>
                          <p className="text-xs text-gray-400 mt-1">{notification.time}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="px-4 py-3 text-center border-t border-gray-200">
                  <button className="text-sm text-blue-600 hover:text-blue-700 font-medium">
                    View all notifications
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* User Menu */}
          <div className="relative">
            <button
              onClick={() => {
                setShowUserMenu(!showUserMenu)
                setShowNotifications(false)
              }}
              className="flex items-center gap-3 p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                <span className="text-white text-sm font-semibold">JD</span>
              </div>
              <div className="text-left hidden md:block">
                <p className="text-sm font-medium text-gray-900">John Doe</p>
                <p className="text-xs text-gray-500">Process Engineer</p>
              </div>
            </button>

            {/* User Dropdown */}
            {showUserMenu && (
              <div className="absolute right-0 mt-2 w-56 bg-white rounded-lg shadow-lg border border-gray-200 z-50">
                <div className="px-4 py-3 border-b border-gray-200">
                  <p className="text-sm font-medium text-gray-900">John Doe</p>
                  <p className="text-sm text-gray-500">john.doe@lab.com</p>
                </div>
                <div className="py-2">
                  <button className="w-full flex items-center gap-3 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">
                    <User className="w-4 h-4" />
                    Profile
                  </button>
                  <button className="w-full flex items-center gap-3 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">
                    <Settings className="w-4 h-4" />
                    Settings
                  </button>
                  <button className="w-full flex items-center gap-3 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">
                    <HelpCircle className="w-4 h-4" />
                    Help & Support
                  </button>
                </div>
                <div className="border-t border-gray-200 py-2">
                  <button className="w-full flex items-center gap-3 px-4 py-2 text-sm text-red-600 hover:bg-red-50">
                    <LogOut className="w-4 h-4" />
                    Sign Out
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  )
}
