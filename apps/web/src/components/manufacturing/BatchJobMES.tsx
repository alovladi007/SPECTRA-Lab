/**
 * Batch Job Manager - Manufacturing Execution System
 * Batch processing and job queue management
 */

"use client"

import React, { useState } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Button } from '@/components/ui/button'
import { LayoutGrid, Play, Pause, X, Clock, Database, TrendingUp } from 'lucide-react'
import { RoleDisplay } from '@/components/shared/RBACActionButtons'

export const BatchJobMES: React.FC = () => {
  const [jobs] = useState([
    { id: 'JOB-2024-158', type: 'Ion Implant', status: 'RUNNING', progress: 67, lot: 'LOT-2024-042', wafers: 25, started: '2h ago' },
    { id: 'JOB-2024-159', type: 'RTP Anneal', status: 'QUEUED', progress: 0, lot: 'LOT-2024-043', wafers: 25, started: null },
    { id: 'JOB-2024-160', type: 'Diffusion', status: 'QUEUED', progress: 0, lot: 'LOT-2024-044', wafers: 25, started: null },
    { id: 'JOB-2024-157', type: 'Oxidation', status: 'COMPLETED', progress: 100, lot: 'LOT-2024-041', wafers: 25, started: '5h ago' }
  ])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'RUNNING': return 'destructive'
      case 'QUEUED': return 'secondary'
      case 'COMPLETED': return 'default'
      case 'FAILED': return 'destructive'
      default: return 'outline'
    }
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold">Batch Job Manager</h1>
          <p className="text-muted-foreground mt-1">
            Manage batch processing jobs and resource allocation across manufacturing equipment
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <RoleDisplay showPermissions={false} />
          <Badge variant="outline" className="text-xs">Redis + Celery Queue</Badge>
        </div>
      </div>

      <Separator />

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Play className="w-4 h-4 text-blue-500" />
              Running
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">3</div>
            <p className="text-xs text-muted-foreground">Active jobs</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Clock className="w-4 h-4 text-yellow-500" />
              Queued
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">8</div>
            <p className="text-xs text-muted-foreground">Waiting to start</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-green-500" />
              Completed (24h)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">42</div>
            <p className="text-xs text-muted-foreground">Success rate: 97.6%</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Database className="w-4 h-4 text-purple-500" />
              Queue Depth
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">11</div>
            <p className="text-xs text-muted-foreground">Total pending</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Tabs */}
      <Tabs defaultValue="active" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="active">
            <Play className="w-4 h-4 mr-2" />
            Active Jobs
          </TabsTrigger>
          <TabsTrigger value="queue">
            <Clock className="w-4 h-4 mr-2" />
            Queue
          </TabsTrigger>
          <TabsTrigger value="history">
            <Database className="w-4 h-4 mr-2" />
            History
          </TabsTrigger>
          <TabsTrigger value="resources">
            <LayoutGrid className="w-4 h-4 mr-2" />
            Resources
          </TabsTrigger>
        </TabsList>

        {/* Active Jobs Tab */}
        <TabsContent value="active" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Running Jobs</CardTitle>
              <CardDescription>Currently executing batch jobs with real-time progress</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {jobs.filter(j => j.status === 'RUNNING').map((job) => (
                  <div key={job.id} className="p-4 border rounded-lg space-y-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium">{job.id} - {job.type}</h4>
                        <p className="text-sm text-muted-foreground">{job.lot} • {job.wafers} wafers • Started {job.started}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant={getStatusColor(job.status)}>{job.status}</Badge>
                        <Button size="sm" variant="outline">
                          <Pause className="w-4 h-4" />
                        </Button>
                        <Button size="sm" variant="destructive">
                          <X className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                    <div className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Progress</span>
                        <span className="font-medium">{job.progress}%</span>
                      </div>
                      <div className="w-full bg-secondary rounded-full h-2">
                        <div
                          className="bg-primary h-2 rounded-full transition-all"
                          style={{ width: `${job.progress}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Queue Tab */}
        <TabsContent value="queue">
          <Card>
            <CardHeader>
              <CardTitle>Job Queue</CardTitle>
              <CardDescription>Pending jobs waiting for resource availability</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {jobs.filter(j => j.status === 'QUEUED').map((job, idx) => (
                  <div key={job.id} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center font-medium text-sm">
                        {idx + 1}
                      </div>
                      <div>
                        <p className="font-medium text-sm">{job.id} - {job.type}</p>
                        <p className="text-xs text-muted-foreground">{job.lot} • {job.wafers} wafers</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant={getStatusColor(job.status)}>{job.status}</Badge>
                      <Button size="sm" variant="outline">
                        <Play className="w-4 h-4 mr-1" />
                        Start
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history">
          <Card>
            <CardHeader>
              <CardTitle>Job History</CardTitle>
              <CardDescription>Completed and failed job records</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {jobs.filter(j => j.status === 'COMPLETED').map((job) => (
                  <div key={job.id} className="flex items-center justify-between p-3 border rounded-lg">
                    <div>
                      <p className="font-medium text-sm">{job.id} - {job.type}</p>
                      <p className="text-xs text-muted-foreground">{job.lot} • {job.wafers} wafers • Started {job.started}</p>
                    </div>
                    <Badge variant={getStatusColor(job.status)}>{job.status}</Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Resources Tab */}
        <TabsContent value="resources">
          <Card>
            <CardHeader>
              <CardTitle>Resource Allocation</CardTitle>
              <CardDescription>Equipment availability and utilization</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {[
                  { name: 'Ion Implanter 1', status: 'BUSY', job: 'JOB-2024-158', util: 87 },
                  { name: 'RTP Chamber 1', status: 'IDLE', job: null, util: 42 },
                  { name: 'Diffusion Furnace 1', status: 'IDLE', job: null, util: 65 },
                  { name: 'Oxidation Furnace 1', status: 'BUSY', job: 'JOB-2024-161', util: 91 }
                ].map((resource, idx) => (
                  <div key={idx} className="p-3 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-sm">{resource.name}</h4>
                      <Badge variant={resource.status === 'BUSY' ? 'destructive' : 'secondary'}>
                        {resource.status}
                      </Badge>
                    </div>
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-muted-foreground">
                          {resource.job ? `Running: ${resource.job}` : 'Available'}
                        </span>
                        <span className="font-medium">Utilization: {resource.util}%</span>
                      </div>
                      <div className="w-full bg-secondary rounded-full h-1.5">
                        <div
                          className="bg-blue-500 h-1.5 rounded-full"
                          style={{ width: `${resource.util}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default BatchJobMES
