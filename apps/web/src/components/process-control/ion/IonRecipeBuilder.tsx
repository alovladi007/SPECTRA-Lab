/**
 * Ion Implantation Recipe Builder
 *
 * Comprehensive recipe configuration with:
 * - Species, energy, dose, tilt/twist, scan pattern
 * - Real-time SRIM calculations
 * - Safety checklist and validations
 * - SOP approval checks
 * - Calibration verification
 * - Integration with process control API
 */

"use client"

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Separator } from '@/components/ui/separator'
import {
  AlertCircle,
  CheckCircle2,
  Clock,
  Zap,
  Target,
  Shield,
  AlertTriangle,
  FileCheck,
  Calculator
} from 'lucide-react'

interface IonRecipe {
  // Basic parameters
  species: string
  energy_kev: number
  dose_atoms_cm2: number
  tilt_deg: number
  twist_deg: number

  // Beam parameters
  beam_current_ma: number
  scan_speed_mm_s: number

  // Wafer parameters
  wafer_diameter_mm: number
  wafer_id?: string
  lot_id?: string
  comments?: string
}

interface SafetyCheckItem {
  id: string
  label: string
  checked: boolean
  required: boolean
  category: 'equipment' | 'procedure' | 'documentation'
}

interface ValidationResult {
  valid: boolean
  errors: string[]
  warnings: string[]
}

interface SRIMPrediction {
  projected_range_nm: number
  straggle_nm: number
  peak_concentration_cm3: number
  implant_time_s: number
}

const SPECIES_OPTIONS = [
  { value: 'B', label: 'Boron (B)' },
  { value: 'P', label: 'Phosphorus (P)' },
  { value: 'As', label: 'Arsenic (As)' },
  { value: 'BF2', label: 'Boron Difluoride (BF2)' },
  { value: 'In', label: 'Indium (In)' },
  { value: 'Sb', label: 'Antimony (Sb)' },
]

const defaultSafetyChecks: SafetyCheckItem[] = [
  // Equipment Checks
  { id: 'vacuum', label: 'Vacuum system at operating pressure (<1e-5 Torr)', checked: false, required: true, category: 'equipment' },
  { id: 'cooling', label: 'Cooling water flow confirmed', checked: false, required: true, category: 'equipment' },
  { id: 'source', label: 'Ion source operating normally', checked: false, required: true, category: 'equipment' },
  { id: 'interlocks', label: 'All safety interlocks active', checked: false, required: true, category: 'equipment' },
  { id: 'faraday', label: 'Faraday cup calibrated within 30 days', checked: false, required: true, category: 'equipment' },

  // Procedural Checks
  { id: 'sop', label: 'Standard Operating Procedure reviewed', checked: false, required: true, category: 'procedure' },
  { id: 'ppe', label: 'Personal Protective Equipment donned', checked: false, required: true, category: 'procedure' },
  { id: 'wafer_load', label: 'Wafer loading procedure understood', checked: false, required: true, category: 'procedure' },
  { id: 'emergency', label: 'Emergency stop locations identified', checked: false, required: true, category: 'procedure' },
  { id: 'radiation', label: 'X-ray radiation safety training current', checked: false, required: true, category: 'procedure' },

  // Documentation Checks
  { id: 'lot_number', label: 'Lot number verified and documented', checked: false, required: true, category: 'documentation' },
  { id: 'recipe_approval', label: 'Recipe approved by process engineer', checked: false, required: true, category: 'documentation' },
  { id: 'calibration', label: 'Equipment calibration current', checked: false, required: true, category: 'documentation' },
  { id: 'maintenance', label: 'Preventive maintenance up to date', checked: false, required: false, category: 'documentation' },
]

interface IonRecipeBuilderProps {
  onSubmit?: (recipe: IonRecipe) => void
  apiEndpoint?: string
  userRole?: string
}

export const IonRecipeBuilder: React.FC<IonRecipeBuilderProps> = ({
  onSubmit,
  apiEndpoint = 'http://localhost:8003',
  userRole = 'engineer'
}) => {
  const [recipe, setRecipe] = useState<IonRecipe>({
    species: 'P',
    energy_kev: 40,
    dose_atoms_cm2: 1e15,
    tilt_deg: 7,
    twist_deg: 0,
    beam_current_ma: 5.0,
    scan_speed_mm_s: 50.0,
    wafer_diameter_mm: 300,
    wafer_id: '',
    lot_id: '',
    comments: '',
  })

  const [safetyChecks, setSafetyChecks] = useState<SafetyCheckItem[]>(defaultSafetyChecks)
  const [validation, setValidation] = useState<ValidationResult | null>(null)
  const [srimPrediction, setSrimPrediction] = useState<SRIMPrediction | null>(null)
  const [isValidating, setIsValidating] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)

  // Calculate SRIM predictions
  useEffect(() => {
    // Simple LSS theory approximation
    const Rp = recipe.energy_kev * 0.3 // nm (very rough)
    const deltaRp = recipe.energy_kev * 0.1 // nm
    const peakConc = recipe.dose_atoms_cm2 / (deltaRp * 1e-7 * Math.sqrt(2 * Math.PI))
    const implantTime = recipe.dose_atoms_cm2 / (recipe.beam_current_ma * 6.24e15)

    setSrimPrediction({
      projected_range_nm: Rp,
      straggle_nm: deltaRp,
      peak_concentration_cm3: peakConc,
      implant_time_s: implantTime,
    })
  }, [recipe.species, recipe.energy_kev, recipe.dose_atoms_cm2, recipe.beam_current_ma])

  // Validate recipe
  const validateRecipe = async () => {
    setIsValidating(true)
    const errors: string[] = []
    const warnings: string[] = []

    // Energy validation
    if (recipe.energy_kev < 1 || recipe.energy_kev > 200) {
      errors.push('Energy must be between 1 and 200 keV')
    }

    // Dose validation
    if (recipe.dose_atoms_cm2 < 1e11 || recipe.dose_atoms_cm2 > 1e16) {
      errors.push('Dose must be between 1e11 and 1e16 atoms/cm²')
    }

    // Tilt/twist validation
    if (recipe.tilt_deg < 0 || recipe.tilt_deg > 90) {
      errors.push('Tilt angle must be between 0° and 90°')
    }
    if (recipe.twist_deg < -180 || recipe.twist_deg > 180) {
      errors.push('Twist angle must be between -180° and 180°')
    }

    // Beam current validation
    if (recipe.beam_current_ma < 0.1 || recipe.beam_current_ma > 50) {
      errors.push('Beam current must be between 0.1 and 50 mA')
    }

    // Scan speed validation
    if (recipe.scan_speed_mm_s < 1 || recipe.scan_speed_mm_s > 100) {
      errors.push('Scan speed must be between 1 and 100 mm/s')
    }

    // High dose warning
    if (recipe.dose_atoms_cm2 > 5e15) {
      warnings.push('High dose may cause substrate damage or amorphization')
    }

    // Low tilt warning
    if (recipe.tilt_deg < 7) {
      warnings.push('Low tilt angle may cause channeling effects')
    }

    // Long implant time warning
    if (srimPrediction && srimPrediction.implant_time_s > 600) {
      warnings.push(`Long implant time (${Math.round(srimPrediction.implant_time_s / 60)} minutes) - consider increasing beam current`)
    }

    setValidation({
      valid: errors.length === 0,
      errors,
      warnings,
    })

    setIsValidating(false)
  }

  useEffect(() => {
    validateRecipe()
  }, [recipe])

  const updateRecipe = (field: keyof IonRecipe, value: any) => {
    console.log(`updateRecipe called: field="${field}", value="${value}"`)
    setRecipe(prev => {
      const updated = { ...prev, [field]: value }
      console.log('Updated recipe state:', updated)
      return updated
    })
  }

  const toggleSafetyCheck = (id: string) => {
    setSafetyChecks(prev =>
      prev.map(check =>
        check.id === id ? { ...check, checked: !check.checked } : check
      )
    )
  }

  const allRequiredChecksPassed = safetyChecks
    .filter(check => check.required)
    .every(check => check.checked)

  const canSubmit = validation?.valid && allRequiredChecksPassed && !isSubmitting

  const handleSubmit = async () => {
    if (!canSubmit) return

    setIsSubmitting(true)

    try {
      const response = await fetch(`${apiEndpoint}/api/ion/runs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token') || 'demo-token'}`,
        },
        body: JSON.stringify({
          species: recipe.species,
          energy_kev: recipe.energy_kev,
          dose_atoms_cm2: recipe.dose_atoms_cm2,
          tilt_deg: recipe.tilt_deg,
          twist_deg: recipe.twist_deg,
          beam_current_ma: recipe.beam_current_ma,
          scan_speed_mm_s: recipe.scan_speed_mm_s,
          wafer_diameter_mm: recipe.wafer_diameter_mm,
          wafer_id: recipe.wafer_id || undefined,
          lot_id: recipe.lot_id || undefined,
          comments: recipe.comments || undefined,
        }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail?.message || 'Failed to create run')
      }

      const result = await response.json()

      if (onSubmit) {
        onSubmit(recipe)
      }

      alert(`Run created successfully!\nRun ID: ${result.run_id}\nJob ID: ${result.job_id}`)
    } catch (error) {
      console.error('Failed to submit recipe:', error)
      alert(`Failed to submit recipe: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setIsSubmitting(false)
    }
  }

  const requiredChecks = safetyChecks.filter(c => c.required)
  const optionalChecks = safetyChecks.filter(c => !c.required)

  return (
    <div className="space-y-6">
      <Tabs defaultValue="recipe" className="space-y-4">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="recipe">Recipe Parameters</TabsTrigger>
          <TabsTrigger value="safety">Safety Checklist</TabsTrigger>
          <TabsTrigger value="predictions">SRIM Predictions</TabsTrigger>
        </TabsList>

        {/* Recipe Parameters Tab */}
        <TabsContent value="recipe" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Ion Beam Parameters</CardTitle>
              <CardDescription>Configure implantation recipe</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Species Selection */}
              <div className="space-y-2">
                <Label htmlFor="species">Dopant Species</Label>
                <Select
                  value={recipe.species}
                  defaultValue={recipe.species}
                  onValueChange={(value) => {
                    console.log('Species selected:', value)
                    updateRecipe('species', value)
                  }}
                >
                  <SelectTrigger id="species" className="!text-black dark:!text-white font-semibold">
                    <SelectValue placeholder="Select dopant species" className="!text-black dark:!text-white" />
                  </SelectTrigger>
                  <SelectContent className="!bg-white dark:!bg-gray-900 !border-2 !border-gray-300">
                    {SPECIES_OPTIONS.map(option => (
                      <SelectItem
                        key={option.value}
                        value={option.value}
                        className="!text-black dark:!text-white font-semibold cursor-pointer hover:!bg-blue-100 dark:hover:!bg-blue-900 !py-2"
                      >
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Energy */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label htmlFor="energy">Implant Energy (keV)</Label>
                  <span className="text-sm text-muted-foreground">1-200 keV</span>
                </div>
                <Input
                  id="energy"
                  type="number"
                  value={recipe.energy_kev}
                  onChange={(e) => updateRecipe('energy_kev', Number(e.target.value))}
                  min={1}
                  max={200}
                  step={1}
                />
              </div>

              {/* Dose */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label htmlFor="dose">Target Dose (atoms/cm²)</Label>
                  <span className="text-sm text-muted-foreground">1e11 - 1e16</span>
                </div>
                <Input
                  id="dose"
                  type="number"
                  value={recipe.dose_atoms_cm2}
                  onChange={(e) => updateRecipe('dose_atoms_cm2', Number(e.target.value))}
                  className="font-mono"
                  step={1e14}
                />
                <div className="text-xs text-muted-foreground">
                  Scientific notation: {recipe.dose_atoms_cm2.toExponential(2)}
                </div>
              </div>

              <Separator />

              {/* Angles */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="tilt">Tilt Angle (deg)</Label>
                  <Input
                    id="tilt"
                    type="number"
                    value={recipe.tilt_deg}
                    onChange={(e) => updateRecipe('tilt_deg', Number(e.target.value))}
                    min={0}
                    max={90}
                    step={1}
                  />
                  <div className="text-xs text-muted-foreground">
                    Recommend 7° to avoid channeling
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="twist">Twist Angle (deg)</Label>
                  <Input
                    id="twist"
                    type="number"
                    value={recipe.twist_deg}
                    onChange={(e) => updateRecipe('twist_deg', Number(e.target.value))}
                    min={-180}
                    max={180}
                    step={1}
                  />
                </div>
              </div>

              <Separator />

              {/* Beam Parameters */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="current">Beam Current (mA)</Label>
                  <Input
                    id="current"
                    type="number"
                    value={recipe.beam_current_ma}
                    onChange={(e) => updateRecipe('beam_current_ma', Number(e.target.value))}
                    min={0.1}
                    max={50}
                    step={0.1}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="scan_speed">Scan Speed (mm/s)</Label>
                  <Input
                    id="scan_speed"
                    type="number"
                    value={recipe.scan_speed_mm_s}
                    onChange={(e) => updateRecipe('scan_speed_mm_s', Number(e.target.value))}
                    min={1}
                    max={100}
                    step={1}
                  />
                </div>
              </div>

              <Separator />

              {/* Wafer Parameters */}
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="wafer_diameter">Wafer Diameter (mm)</Label>
                  <Select
                    value={recipe.wafer_diameter_mm.toString()}
                    onValueChange={(value) => updateRecipe('wafer_diameter_mm', Number(value))}
                  >
                    <SelectTrigger id="wafer_diameter">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="200">200 mm (8 inch)</SelectItem>
                      <SelectItem value="300">300 mm (12 inch)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="wafer_id">Wafer ID (optional)</Label>
                  <Input
                    id="wafer_id"
                    value={recipe.wafer_id}
                    onChange={(e) => updateRecipe('wafer_id', e.target.value)}
                    placeholder="W12345"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="lot_id">Lot ID (optional)</Label>
                  <Input
                    id="lot_id"
                    value={recipe.lot_id}
                    onChange={(e) => updateRecipe('lot_id', e.target.value)}
                    placeholder="LOT-001"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="comments">Comments (optional)</Label>
                  <Input
                    id="comments"
                    value={recipe.comments}
                    onChange={(e) => updateRecipe('comments', e.target.value)}
                    placeholder="Additional notes..."
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Validation Results */}
          {validation && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  {validation.valid ? (
                    <CheckCircle2 className="w-5 h-5 text-green-500" />
                  ) : (
                    <AlertCircle className="w-5 h-5 text-red-500" />
                  )}
                  Recipe Validation
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {validation.errors.length > 0 && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      <div className="font-semibold mb-1">Errors:</div>
                      <ul className="list-disc list-inside space-y-1">
                        {validation.errors.map((error, idx) => (
                          <li key={idx}>{error}</li>
                        ))}
                      </ul>
                    </AlertDescription>
                  </Alert>
                )}

                {validation.warnings.length > 0 && (
                  <Alert>
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>
                      <div className="font-semibold mb-1">Warnings:</div>
                      <ul className="list-disc list-inside space-y-1">
                        {validation.warnings.map((warning, idx) => (
                          <li key={idx}>{warning}</li>
                        ))}
                      </ul>
                    </AlertDescription>
                  </Alert>
                )}

                {validation.valid && validation.warnings.length === 0 && (
                  <Alert className="border-green-500 bg-green-50">
                    <CheckCircle2 className="h-4 w-4 text-green-600" />
                    <AlertDescription className="text-green-800">
                      Recipe validation passed. All parameters within acceptable ranges.
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Safety Checklist Tab */}
        <TabsContent value="safety" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="w-5 h-5" />
                Safety Checklist
              </CardTitle>
              <CardDescription>
                All required items must be checked before starting implantation
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Required Checks */}
              <div>
                <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-red-500" />
                  Required Checks ({requiredChecks.filter(c => c.checked).length}/{requiredChecks.length})
                </h3>
                <div className="space-y-2">
                  {['equipment', 'procedure', 'documentation'].map(category => {
                    const checks = requiredChecks.filter(c => c.category === category)
                    if (checks.length === 0) return null

                    return (
                      <div key={category} className="space-y-2">
                        <div className="text-xs font-semibold text-muted-foreground uppercase">
                          {category}
                        </div>
                        {checks.map(check => (
                          <div
                            key={check.id}
                            className="flex items-start space-x-2 p-2 border rounded hover:bg-accent/50"
                          >
                            <Checkbox
                              id={check.id}
                              checked={check.checked}
                              onCheckedChange={() => toggleSafetyCheck(check.id)}
                            />
                            <label
                              htmlFor={check.id}
                              className="text-sm cursor-pointer flex-1"
                            >
                              {check.label}
                            </label>
                            <Badge variant={check.checked ? "default" : "secondary"} className="ml-2">
                              {check.checked ? 'OK' : 'Required'}
                            </Badge>
                          </div>
                        ))}
                      </div>
                    )
                  })}
                </div>
              </div>

              <Separator />

              {/* Optional Checks */}
              {optionalChecks.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                    <FileCheck className="w-4 h-4" />
                    Optional Checks ({optionalChecks.filter(c => c.checked).length}/{optionalChecks.length})
                  </h3>
                  <div className="space-y-2">
                    {optionalChecks.map(check => (
                      <div
                        key={check.id}
                        className="flex items-start space-x-2 p-2 border rounded hover:bg-accent/50"
                      >
                        <Checkbox
                          id={check.id}
                          checked={check.checked}
                          onCheckedChange={() => toggleSafetyCheck(check.id)}
                        />
                        <label
                          htmlFor={check.id}
                          className="text-sm cursor-pointer flex-1"
                        >
                          {check.label}
                        </label>
                        <Badge variant="outline">Optional</Badge>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Safety Status */}
              <div className="pt-4 border-t">
                {allRequiredChecksPassed ? (
                  <Alert className="border-green-500 bg-green-50">
                    <CheckCircle2 className="h-4 w-4 text-green-600" />
                    <AlertDescription className="text-green-800 font-semibold">
                      All required safety checks completed. Ready to proceed.
                    </AlertDescription>
                  </Alert>
                ) : (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      {requiredChecks.length - requiredChecks.filter(c => c.checked).length} required check(s) remaining
                    </AlertDescription>
                  </Alert>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* SRIM Predictions Tab */}
        <TabsContent value="predictions" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Calculator className="w-5 h-5" />
                SRIM Predictions
              </CardTitle>
              <CardDescription>
                Estimated depth profile based on LSS theory
              </CardDescription>
            </CardHeader>
            <CardContent>
              {srimPrediction && (
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 border rounded-lg bg-muted/30">
                    <div className="text-sm text-muted-foreground mb-1">Projected Range (Rp)</div>
                    <div className="text-3xl font-bold">
                      {srimPrediction.projected_range_nm.toFixed(1)}
                      <span className="text-lg text-muted-foreground ml-1">nm</span>
                    </div>
                  </div>

                  <div className="p-4 border rounded-lg bg-muted/30">
                    <div className="text-sm text-muted-foreground mb-1">Straggle (ΔRp)</div>
                    <div className="text-3xl font-bold">
                      {srimPrediction.straggle_nm.toFixed(1)}
                      <span className="text-lg text-muted-foreground ml-1">nm</span>
                    </div>
                  </div>

                  <div className="p-4 border rounded-lg bg-muted/30">
                    <div className="text-sm text-muted-foreground mb-1">Peak Concentration</div>
                    <div className="text-2xl font-bold font-mono">
                      {srimPrediction.peak_concentration_cm3.toExponential(2)}
                      <span className="text-sm text-muted-foreground ml-1">/cm³</span>
                    </div>
                  </div>

                  <div className="p-4 border rounded-lg bg-muted/30">
                    <div className="text-sm text-muted-foreground mb-1 flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      Estimated Time
                    </div>
                    <div className="text-3xl font-bold">
                      {Math.round(srimPrediction.implant_time_s)}
                      <span className="text-lg text-muted-foreground ml-1">sec</span>
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">
                      ({(srimPrediction.implant_time_s / 60).toFixed(1)} minutes)
                    </div>
                  </div>
                </div>
              )}

              <div className="mt-6 p-4 border rounded-lg bg-blue-50 border-blue-200">
                <div className="text-sm font-semibold text-blue-900 mb-2">Note on Predictions</div>
                <p className="text-xs text-blue-800">
                  These predictions use simplified LSS theory for quick estimation. For production,
                  verify with full SRIM/TRIM simulations. Actual profiles depend on crystal orientation,
                  surface oxide, and other factors not included in this simple model.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Submit Button */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <div className="text-sm font-medium">Ready to submit?</div>
              <div className="text-xs text-muted-foreground">
                {!validation?.valid && 'Fix validation errors first'}
                {validation?.valid && !allRequiredChecksPassed && 'Complete all required safety checks'}
                {canSubmit && 'All checks passed - ready to create run'}
              </div>
            </div>
            <Button
              onClick={handleSubmit}
              disabled={!canSubmit}
              size="lg"
              className="min-w-[200px]"
            >
              {isSubmitting ? (
                <>
                  <Clock className="w-4 h-4 mr-2 animate-spin" />
                  Submitting...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 mr-2" />
                  Create Ion Run
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default IonRecipeBuilder
