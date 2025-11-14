/**
 * Statistical Process Control (SPC) Manufacturing Dashboard
 * Production SPC monitoring and control chart analysis
 */

"use client"

import React, { useState } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Activity, TrendingUp, AlertTriangle, CheckCircle2, Target } from 'lucide-react'
import { RoleDisplay } from '@/components/shared/RBACActionButtons'

export const SPCMES: React.FC = () => {
  const [series] = useState([
    { id: 'SPC-001', metric: 'Sheet Resistance', process: 'Diffusion', chart: 'XBAR_R', status: 'IN_CONTROL', violations: 0 },
    { id: 'SPC-002', metric: 'Oxide Thickness', process: 'Oxidation', chart: 'I_MR', status: 'OUT_OF_CONTROL', violations: 3 },
    { id: 'SPC-003', metric: 'Junction Depth', process: 'Diffusion', chart: 'EWMA', status: 'IN_CONTROL', violations: 0 }
  ])

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold">Statistical Process Control</h1>
          <p className="text-muted-foreground mt-1">
            Real-time SPC monitoring with Western Electric rules and control charts
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <RoleDisplay showPermissions={false} />
          <Badge variant="outline" className="text-xs">36 Active Series</Badge>
        </div>
      </div>

      <Separator />

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-green-500" />
              In Control
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">28</div>
            <p className="text-xs text-muted-foreground">77.8% of series</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-orange-500" />
              Out of Control
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-600">8</div>
            <p className="text-xs text-muted-foreground">22.2% of series</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Target className="w-4 h-4 text-blue-500" />
              Cpk Average
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">1.67</div>
            <p className="text-xs text-muted-foreground">Process capability</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-purple-500" />
              Violations (24h)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12</div>
            <p className="text-xs text-muted-foreground">Rule violations</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Tabs */}
      <Tabs defaultValue="series" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="series">
            <Activity className="w-4 h-4 mr-2" />
            Control Series
          </TabsTrigger>
          <TabsTrigger value="charts">
            <TrendingUp className="w-4 h-4 mr-2" />
            Control Charts
          </TabsTrigger>
          <TabsTrigger value="violations">
            <AlertTriangle className="w-4 h-4 mr-2" />
            Violations
          </TabsTrigger>
          <TabsTrigger value="capability">
            <Target className="w-4 h-4 mr-2" />
            Capability
          </TabsTrigger>
        </TabsList>

        {/* Control Series Tab */}
        <TabsContent value="series" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Active SPC Series</CardTitle>
              <CardDescription>Monitor control chart series across all processes</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {series.map((s) => (
                  <div key={s.id} className="flex items-center justify-between p-4 border rounded-lg hover:bg-accent">
                    <div className="flex-1">
                      <h4 className="font-medium">{s.metric}</h4>
                      <p className="text-sm text-muted-foreground">{s.process} • {s.chart} Chart</p>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="text-right">
                        <p className="text-sm font-medium">{s.violations} violations</p>
                        <p className="text-xs text-muted-foreground">Last 100 points</p>
                      </div>
                      <Badge variant={s.status === 'IN_CONTROL' ? 'default' : 'destructive'}>
                        {s.status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Control Charts Tab */}
        <TabsContent value="charts">
          <Card>
            <CardHeader>
              <CardTitle>Control Chart Viewer</CardTitle>
              <CardDescription>View X-bar/R, I-MR, EWMA, and CUSUM charts</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12 text-muted-foreground">
                Select a series to view control chart
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Violations Tab */}
        <TabsContent value="violations">
          <Card>
            <CardHeader>
              <CardTitle>Rule Violations</CardTitle>
              <CardDescription>Western Electric rules and Nelson rules violations</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {[
                  { rule: 'Rule 1', desc: '1 point > 3σ from center', series: 'Sheet Resistance', time: '2h ago' },
                  { rule: 'Rule 2', desc: '9 points on same side of center', series: 'Oxide Thickness', time: '5h ago' },
                  { rule: 'Rule 4', desc: '14 points alternating up/down', series: 'Dose Uniformity', time: '8h ago' }
                ].map((v, idx) => (
                  <div key={idx} className="flex items-center gap-3 p-3 border rounded-lg">
                    <AlertTriangle className="w-5 h-5 text-orange-500 flex-shrink-0" />
                    <div className="flex-1">
                      <p className="font-medium text-sm">{v.rule}: {v.desc}</p>
                      <p className="text-xs text-muted-foreground">{v.series} • {v.time}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Capability Tab */}
        <TabsContent value="capability">
          <Card>
            <CardHeader>
              <CardTitle>Process Capability Analysis</CardTitle>
              <CardDescription>Cp, Cpk, Pp, Ppk metrics and capability indices</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12 text-muted-foreground">
                Capability analysis will be displayed here
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default SPCMES
