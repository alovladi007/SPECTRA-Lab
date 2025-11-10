'use client'

import { useState, useEffect } from 'react'
import {
  Settings, Save, RotateCcw, Bell, Shield, Database,
  Mail, Globe, Clock, Key, Wifi, AlertTriangle, CheckCircle2
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'

interface SettingsConfig {
  // General Settings
  platformName: string
  timezone: string
  language: string
  dateFormat: string
  theme: 'light' | 'dark' | 'auto'

  // Security Settings
  sessionTimeout: number
  passwordMinLength: number
  passwordRequireSpecialChar: boolean
  twoFactorAuth: boolean
  ipWhitelist: string

  // Notification Settings
  emailNotifications: boolean
  emailHost: string
  emailPort: number
  emailFrom: string
  slackIntegration: boolean
  slackWebhook: string

  // Database Settings
  dbHost: string
  dbPort: number
  dbName: string
  dbBackupEnabled: boolean
  dbBackupFrequency: string

  // API Settings
  apiRateLimit: number
  apiTimeout: number
  apiVersion: string
  webhookUrl: string

  // System Settings
  maxUploadSize: number
  dataRetentionDays: number
  debugMode: boolean
  maintenanceMode: boolean
}

const defaultSettings: SettingsConfig = {
  platformName: 'SPECTRA-Lab',
  timezone: 'UTC',
  language: 'en-US',
  dateFormat: 'YYYY-MM-DD',
  theme: 'light',
  sessionTimeout: 3600,
  passwordMinLength: 8,
  passwordRequireSpecialChar: true,
  twoFactorAuth: false,
  ipWhitelist: '',
  emailNotifications: true,
  emailHost: 'smtp.spectralab.com',
  emailPort: 587,
  emailFrom: 'noreply@spectralab.com',
  slackIntegration: false,
  slackWebhook: '',
  dbHost: 'localhost',
  dbPort: 5432,
  dbName: 'spectra',
  dbBackupEnabled: true,
  dbBackupFrequency: 'daily',
  apiRateLimit: 100,
  apiTimeout: 30,
  apiVersion: 'v1',
  webhookUrl: '',
  maxUploadSize: 100,
  dataRetentionDays: 365,
  debugMode: false,
  maintenanceMode: false
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<SettingsConfig>(defaultSettings)
  const [activeTab, setActiveTab] = useState<string>('general')
  const [hasChanges, setHasChanges] = useState(false)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')

  useEffect(() => {
    // Load settings from localStorage or API
    const savedSettings = localStorage.getItem('spectra-settings')
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings))
    }
  }, [])

  const handleChange = (key: keyof SettingsConfig, value: any) => {
    setSettings({ ...settings, [key]: value })
    setHasChanges(true)
    setSaveStatus('idle')
  }

  const handleSave = () => {
    setSaveStatus('saving')
    // Simulate API call
    setTimeout(() => {
      localStorage.setItem('spectra-settings', JSON.stringify(settings))
      setSaveStatus('saved')
      setHasChanges(false)
      setTimeout(() => setSaveStatus('idle'), 2000)
    }, 1000)
  }

  const handleReset = () => {
    if (confirm('Are you sure you want to reset all settings to defaults?')) {
      setSettings(defaultSettings)
      setHasChanges(true)
      setSaveStatus('idle')
    }
  }

  const tabs = [
    { id: 'general', name: 'General', icon: Globe },
    { id: 'security', name: 'Security', icon: Shield },
    { id: 'notifications', name: 'Notifications', icon: Bell },
    { id: 'database', name: 'Database', icon: Database },
    { id: 'api', name: 'API & Integrations', icon: Wifi },
    { id: 'system', name: 'System', icon: Settings }
  ]

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-gray-700 to-gray-900 rounded-xl flex items-center justify-center">
            <Settings className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">System Settings</h1>
            <p className="text-gray-600 mt-1">Configure platform settings and preferences</p>
          </div>
        </div>

        <div className="flex gap-3">
          {saveStatus === 'saved' && (
            <div className="flex items-center gap-2 text-green-600 mr-2">
              <CheckCircle2 className="w-5 h-5" />
              <span className="text-sm font-medium">Settings saved</span>
            </div>
          )}
          {saveStatus === 'error' && (
            <div className="flex items-center gap-2 text-red-600 mr-2">
              <AlertTriangle className="w-5 h-5" />
              <span className="text-sm font-medium">Save failed</span>
            </div>
          )}
          <Button variant="outline" onClick={handleReset}>
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset to Defaults
          </Button>
          <Button
            onClick={handleSave}
            disabled={!hasChanges || saveStatus === 'saving'}
          >
            <Save className="w-4 h-4 mr-2" />
            {saveStatus === 'saving' ? 'Saving...' : 'Save Changes'}
          </Button>
        </div>
      </div>

      <div className="flex gap-6">
        {/* Tabs Sidebar */}
        <div className="w-64 space-y-2">
          {tabs.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-colors
                  ${activeTab === tab.id
                    ? 'bg-blue-50 text-blue-700 font-medium'
                    : 'text-gray-700 hover:bg-gray-100'
                  }
                `}
              >
                <Icon className="w-5 h-5" />
                <span>{tab.name}</span>
              </button>
            )
          })}
        </div>

        {/* Settings Content */}
        <Card className="flex-1 p-6">
          {/* General Settings */}
          {activeTab === 'general' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">General Settings</h2>
                <p className="text-gray-600 mb-6">Configure basic platform settings</p>
              </div>

              <div className="grid grid-cols-2 gap-6">
                <div>
                  <Label>Platform Name</Label>
                  <Input
                    value={settings.platformName}
                    onChange={(e) => handleChange('platformName', e.target.value)}
                  />
                </div>
                <div>
                  <Label>Timezone</Label>
                  <Select
                    value={settings.timezone}
                    onValueChange={(value) => handleChange('timezone', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="UTC">UTC</SelectItem>
                      <SelectItem value="America/New_York">Eastern Time</SelectItem>
                      <SelectItem value="America/Chicago">Central Time</SelectItem>
                      <SelectItem value="America/Denver">Mountain Time</SelectItem>
                      <SelectItem value="America/Los_Angeles">Pacific Time</SelectItem>
                      <SelectItem value="Europe/London">London</SelectItem>
                      <SelectItem value="Europe/Paris">Paris</SelectItem>
                      <SelectItem value="Asia/Tokyo">Tokyo</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Language</Label>
                  <Select
                    value={settings.language}
                    onValueChange={(value) => handleChange('language', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="en-US">English (US)</SelectItem>
                      <SelectItem value="en-GB">English (UK)</SelectItem>
                      <SelectItem value="fr-FR">French</SelectItem>
                      <SelectItem value="de-DE">German</SelectItem>
                      <SelectItem value="ja-JP">Japanese</SelectItem>
                      <SelectItem value="zh-CN">Chinese (Simplified)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Date Format</Label>
                  <Select
                    value={settings.dateFormat}
                    onValueChange={(value) => handleChange('dateFormat', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="YYYY-MM-DD">YYYY-MM-DD</SelectItem>
                      <SelectItem value="MM/DD/YYYY">MM/DD/YYYY</SelectItem>
                      <SelectItem value="DD/MM/YYYY">DD/MM/YYYY</SelectItem>
                      <SelectItem value="DD-MMM-YYYY">DD-MMM-YYYY</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Theme</Label>
                  <Select
                    value={settings.theme}
                    onValueChange={(value: any) => handleChange('theme', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="light">Light</SelectItem>
                      <SelectItem value="dark">Dark</SelectItem>
                      <SelectItem value="auto">Auto</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          )}

          {/* Security Settings */}
          {activeTab === 'security' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">Security Settings</h2>
                <p className="text-gray-600 mb-6">Configure authentication and security policies</p>
              </div>

              <div className="grid grid-cols-2 gap-6">
                <div>
                  <Label>Session Timeout (seconds)</Label>
                  <Input
                    type="number"
                    value={settings.sessionTimeout}
                    onChange={(e) => handleChange('sessionTimeout', parseInt(e.target.value))}
                  />
                </div>
                <div>
                  <Label>Minimum Password Length</Label>
                  <Input
                    type="number"
                    value={settings.passwordMinLength}
                    onChange={(e) => handleChange('passwordMinLength', parseInt(e.target.value))}
                  />
                </div>
                <div className="col-span-2">
                  <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <div>
                      <p className="font-medium text-gray-900">Require Special Characters</p>
                      <p className="text-sm text-gray-600">Passwords must contain special characters</p>
                    </div>
                    <button
                      onClick={() => handleChange('passwordRequireSpecialChar', !settings.passwordRequireSpecialChar)}
                      className={`
                        relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                        ${settings.passwordRequireSpecialChar ? 'bg-blue-600' : 'bg-gray-200'}
                      `}
                    >
                      <span
                        className={`
                          inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                          ${settings.passwordRequireSpecialChar ? 'translate-x-6' : 'translate-x-1'}
                        `}
                      />
                    </button>
                  </div>
                </div>
                <div className="col-span-2">
                  <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <div>
                      <p className="font-medium text-gray-900">Two-Factor Authentication</p>
                      <p className="text-sm text-gray-600">Require 2FA for all users</p>
                    </div>
                    <button
                      onClick={() => handleChange('twoFactorAuth', !settings.twoFactorAuth)}
                      className={`
                        relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                        ${settings.twoFactorAuth ? 'bg-blue-600' : 'bg-gray-200'}
                      `}
                    >
                      <span
                        className={`
                          inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                          ${settings.twoFactorAuth ? 'translate-x-6' : 'translate-x-1'}
                        `}
                      />
                    </button>
                  </div>
                </div>
                <div className="col-span-2">
                  <Label>IP Whitelist (comma-separated)</Label>
                  <Textarea
                    value={settings.ipWhitelist}
                    onChange={(e) => handleChange('ipWhitelist', e.target.value)}
                    placeholder="192.168.1.1, 10.0.0.1"
                    rows={3}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Notification Settings */}
          {activeTab === 'notifications' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">Notification Settings</h2>
                <p className="text-gray-600 mb-6">Configure email and notification preferences</p>
              </div>

              <div className="space-y-6">
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <p className="font-medium text-gray-900">Email Notifications</p>
                    <p className="text-sm text-gray-600">Send email notifications for system events</p>
                  </div>
                  <button
                    onClick={() => handleChange('emailNotifications', !settings.emailNotifications)}
                    className={`
                      relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                      ${settings.emailNotifications ? 'bg-blue-600' : 'bg-gray-200'}
                    `}
                  >
                    <span
                      className={`
                        inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                        ${settings.emailNotifications ? 'translate-x-6' : 'translate-x-1'}
                      `}
                    />
                  </button>
                </div>

                {settings.emailNotifications && (
                  <div className="grid grid-cols-2 gap-6 p-4 bg-blue-50 rounded-lg">
                    <div>
                      <Label>SMTP Host</Label>
                      <Input
                        value={settings.emailHost}
                        onChange={(e) => handleChange('emailHost', e.target.value)}
                      />
                    </div>
                    <div>
                      <Label>SMTP Port</Label>
                      <Input
                        type="number"
                        value={settings.emailPort}
                        onChange={(e) => handleChange('emailPort', parseInt(e.target.value))}
                      />
                    </div>
                    <div className="col-span-2">
                      <Label>From Address</Label>
                      <Input
                        type="email"
                        value={settings.emailFrom}
                        onChange={(e) => handleChange('emailFrom', e.target.value)}
                      />
                    </div>
                  </div>
                )}

                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <p className="font-medium text-gray-900">Slack Integration</p>
                    <p className="text-sm text-gray-600">Send notifications to Slack</p>
                  </div>
                  <button
                    onClick={() => handleChange('slackIntegration', !settings.slackIntegration)}
                    className={`
                      relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                      ${settings.slackIntegration ? 'bg-blue-600' : 'bg-gray-200'}
                    `}
                  >
                    <span
                      className={`
                        inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                        ${settings.slackIntegration ? 'translate-x-6' : 'translate-x-1'}
                      `}
                    />
                  </button>
                </div>

                {settings.slackIntegration && (
                  <div className="p-4 bg-purple-50 rounded-lg">
                    <Label>Slack Webhook URL</Label>
                    <Input
                      value={settings.slackWebhook}
                      onChange={(e) => handleChange('slackWebhook', e.target.value)}
                      placeholder="https://hooks.slack.com/services/..."
                    />
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Database Settings */}
          {activeTab === 'database' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">Database Settings</h2>
                <p className="text-gray-600 mb-6">Configure database connection and backup settings</p>
              </div>

              <div className="grid grid-cols-2 gap-6">
                <div>
                  <Label>Database Host</Label>
                  <Input
                    value={settings.dbHost}
                    onChange={(e) => handleChange('dbHost', e.target.value)}
                  />
                </div>
                <div>
                  <Label>Database Port</Label>
                  <Input
                    type="number"
                    value={settings.dbPort}
                    onChange={(e) => handleChange('dbPort', parseInt(e.target.value))}
                  />
                </div>
                <div className="col-span-2">
                  <Label>Database Name</Label>
                  <Input
                    value={settings.dbName}
                    onChange={(e) => handleChange('dbName', e.target.value)}
                  />
                </div>
                <div className="col-span-2">
                  <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <div>
                      <p className="font-medium text-gray-900">Automatic Backups</p>
                      <p className="text-sm text-gray-600">Enable scheduled database backups</p>
                    </div>
                    <button
                      onClick={() => handleChange('dbBackupEnabled', !settings.dbBackupEnabled)}
                      className={`
                        relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                        ${settings.dbBackupEnabled ? 'bg-blue-600' : 'bg-gray-200'}
                      `}
                    >
                      <span
                        className={`
                          inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                          ${settings.dbBackupEnabled ? 'translate-x-6' : 'translate-x-1'}
                        `}
                      />
                    </button>
                  </div>
                </div>
                {settings.dbBackupEnabled && (
                  <div className="col-span-2">
                    <Label>Backup Frequency</Label>
                    <Select
                      value={settings.dbBackupFrequency}
                      onValueChange={(value) => handleChange('dbBackupFrequency', value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="hourly">Hourly</SelectItem>
                        <SelectItem value="daily">Daily</SelectItem>
                        <SelectItem value="weekly">Weekly</SelectItem>
                        <SelectItem value="monthly">Monthly</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* API Settings */}
          {activeTab === 'api' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">API & Integration Settings</h2>
                <p className="text-gray-600 mb-6">Configure API and external integrations</p>
              </div>

              <div className="grid grid-cols-2 gap-6">
                <div>
                  <Label>API Rate Limit (requests/min)</Label>
                  <Input
                    type="number"
                    value={settings.apiRateLimit}
                    onChange={(e) => handleChange('apiRateLimit', parseInt(e.target.value))}
                  />
                </div>
                <div>
                  <Label>API Timeout (seconds)</Label>
                  <Input
                    type="number"
                    value={settings.apiTimeout}
                    onChange={(e) => handleChange('apiTimeout', parseInt(e.target.value))}
                  />
                </div>
                <div>
                  <Label>API Version</Label>
                  <Select
                    value={settings.apiVersion}
                    onValueChange={(value) => handleChange('apiVersion', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="v1">v1</SelectItem>
                      <SelectItem value="v2">v2</SelectItem>
                      <SelectItem value="v3">v3 (Beta)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="col-span-2">
                  <Label>Webhook URL</Label>
                  <Input
                    value={settings.webhookUrl}
                    onChange={(e) => handleChange('webhookUrl', e.target.value)}
                    placeholder="https://your-webhook-endpoint.com"
                  />
                </div>
              </div>
            </div>
          )}

          {/* System Settings */}
          {activeTab === 'system' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">System Settings</h2>
                <p className="text-gray-600 mb-6">Configure system-level settings</p>
              </div>

              <div className="grid grid-cols-2 gap-6">
                <div>
                  <Label>Max Upload Size (MB)</Label>
                  <Input
                    type="number"
                    value={settings.maxUploadSize}
                    onChange={(e) => handleChange('maxUploadSize', parseInt(e.target.value))}
                  />
                </div>
                <div>
                  <Label>Data Retention (days)</Label>
                  <Input
                    type="number"
                    value={settings.dataRetentionDays}
                    onChange={(e) => handleChange('dataRetentionDays', parseInt(e.target.value))}
                  />
                </div>
                <div className="col-span-2">
                  <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <div>
                      <p className="font-medium text-gray-900">Debug Mode</p>
                      <p className="text-sm text-gray-600">Enable detailed logging for debugging</p>
                    </div>
                    <button
                      onClick={() => handleChange('debugMode', !settings.debugMode)}
                      className={`
                        relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                        ${settings.debugMode ? 'bg-blue-600' : 'bg-gray-200'}
                      `}
                    >
                      <span
                        className={`
                          inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                          ${settings.debugMode ? 'translate-x-6' : 'translate-x-1'}
                        `}
                      />
                    </button>
                  </div>
                </div>
                <div className="col-span-2">
                  <div className="flex items-center justify-between p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                    <div>
                      <p className="font-medium text-yellow-900">Maintenance Mode</p>
                      <p className="text-sm text-yellow-700">System will be unavailable to users</p>
                    </div>
                    <button
                      onClick={() => handleChange('maintenanceMode', !settings.maintenanceMode)}
                      className={`
                        relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                        ${settings.maintenanceMode ? 'bg-yellow-600' : 'bg-gray-200'}
                      `}
                    >
                      <span
                        className={`
                          inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                          ${settings.maintenanceMode ? 'translate-x-6' : 'translate-x-1'}
                        `}
                      />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </Card>
      </div>

      {/* Status Bar */}
      {hasChanges && (
        <div className="fixed bottom-0 left-0 right-0 bg-yellow-50 border-t border-yellow-200 p-4">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-2 text-yellow-800">
              <AlertTriangle className="w-5 h-5" />
              <span className="font-medium">You have unsaved changes</span>
            </div>
            <div className="flex gap-3">
              <Button variant="outline" onClick={handleReset}>
                Discard Changes
              </Button>
              <Button onClick={handleSave}>
                Save Changes
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
