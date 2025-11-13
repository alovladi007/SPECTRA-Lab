"use client";

/**
 * CVD Recipe Editor
 * Visual editor for creating and modifying CVD process recipes
 */

import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Plus,
  Trash2,
  Save,
  Copy,
  AlertCircle,
  Thermometer,
  Gauge,
  Wind,
  Zap,
  Clock,
  Target,
} from "lucide-react";
import { toast } from "sonner";

// Types
interface TemperatureZone {
  zone: number;
  setpoint_c: number;
  ramp_rate_c_per_min: number;
}

interface GasFlow {
  name: string;
  flow_sccm: number;
  mfc_id: string;
}

interface RecipeStep {
  step: number;
  name: string;
  duration_s: number;
  action: string;
  description?: string;
}

interface PlasmaSettings {
  rf_power_w?: number;
  frequency_mhz?: number;
  bias_voltage_v?: number;
  matching_mode?: string;
}

interface RecipeData {
  name: string;
  description: string;
  process_mode_id: string;
  temperature_profile: {
    zones: TemperatureZone[];
    soak_time_s: number;
  };
  gas_flows: {
    gases: GasFlow[];
    carrier_gas: string;
    carrier_flow_sccm: number;
  };
  pressure_profile: {
    base_pressure_pa: number;
    process_pressure_pa: number;
    throttle_valve_position_pct: number;
  };
  plasma_settings?: PlasmaSettings;
  recipe_steps: RecipeStep[];
  process_time_s: number;
  target_thickness_nm?: number;
  target_uniformity_pct?: number;
  tags: string[];
  version: string;
}

interface RecipeEditorProps {
  initialRecipe?: Partial<RecipeData>;
  processModeId: string;
  hasPlasma: boolean;
  onSave: (recipe: RecipeData) => void;
  onCancel: () => void;
}

export default function RecipeEditor({
  initialRecipe,
  processModeId,
  hasPlasma,
  onSave,
  onCancel,
}: RecipeEditorProps) {
  const [recipe, setRecipe] = useState<RecipeData>({
    name: initialRecipe?.name || "",
    description: initialRecipe?.description || "",
    process_mode_id: processModeId,
    temperature_profile: initialRecipe?.temperature_profile || {
      zones: [{ zone: 1, setpoint_c: 650, ramp_rate_c_per_min: 10 }],
      soak_time_s: 300,
    },
    gas_flows: initialRecipe?.gas_flows || {
      gases: [{ name: "SiH4", flow_sccm: 50, mfc_id: "MFC1" }],
      carrier_gas: "N2",
      carrier_flow_sccm: 1000,
    },
    pressure_profile: initialRecipe?.pressure_profile || {
      base_pressure_pa: 10,
      process_pressure_pa: 100,
      throttle_valve_position_pct: 45,
    },
    plasma_settings: initialRecipe?.plasma_settings,
    recipe_steps: initialRecipe?.recipe_steps || [
      { step: 1, name: "Pumpdown", duration_s: 60, action: "evacuate" },
      { step: 2, name: "Heatup", duration_s: 300, action: "ramp_temperature" },
      { step: 3, name: "Stabilize", duration_s: 120, action: "stabilize" },
      { step: 4, name: "Deposition", duration_s: 600, action: "deposit" },
      { step: 5, name: "Cooldown", duration_s: 300, action: "cool" },
    ],
    process_time_s: initialRecipe?.process_time_s || 1380,
    target_thickness_nm: initialRecipe?.target_thickness_nm,
    target_uniformity_pct: initialRecipe?.target_uniformity_pct,
    tags: initialRecipe?.tags || [],
    version: initialRecipe?.version || "1.0",
  });

  const [newTag, setNewTag] = useState("");
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  // Add temperature zone
  const addTemperatureZone = () => {
    const zones = [...recipe.temperature_profile.zones];
    zones.push({
      zone: zones.length + 1,
      setpoint_c: 650,
      ramp_rate_c_per_min: 10,
    });

    setRecipe({
      ...recipe,
      temperature_profile: { ...recipe.temperature_profile, zones },
    });
  };

  // Update temperature zone
  const updateTemperatureZone = (index: number, field: string, value: number) => {
    const zones = [...recipe.temperature_profile.zones];
    zones[index] = { ...zones[index], [field]: value };

    setRecipe({
      ...recipe,
      temperature_profile: { ...recipe.temperature_profile, zones },
    });
  };

  // Remove temperature zone
  const removeTemperatureZone = (index: number) => {
    const zones = recipe.temperature_profile.zones.filter((_, i) => i !== index);
    zones.forEach((zone, i) => {
      zone.zone = i + 1;
    });

    setRecipe({
      ...recipe,
      temperature_profile: { ...recipe.temperature_profile, zones },
    });
  };

  // Add gas flow
  const addGasFlow = () => {
    const gases = [...recipe.gas_flows.gases];
    gases.push({
      name: "N2",
      flow_sccm: 100,
      mfc_id: `MFC${gases.length + 1}`,
    });

    setRecipe({
      ...recipe,
      gas_flows: { ...recipe.gas_flows, gases },
    });
  };

  // Update gas flow
  const updateGasFlow = (index: number, field: string, value: string | number) => {
    const gases = [...recipe.gas_flows.gases];
    gases[index] = { ...gases[index], [field]: value };

    setRecipe({
      ...recipe,
      gas_flows: { ...recipe.gas_flows, gases },
    });
  };

  // Remove gas flow
  const removeGasFlow = (index: number) => {
    const gases = recipe.gas_flows.gases.filter((_, i) => i !== index);

    setRecipe({
      ...recipe,
      gas_flows: { ...recipe.gas_flows, gases },
    });
  };

  // Add recipe step
  const addRecipeStep = () => {
    const steps = [...recipe.recipe_steps];
    steps.push({
      step: steps.length + 1,
      name: "New Step",
      duration_s: 60,
      action: "stabilize",
    });

    setRecipe({ ...recipe, recipe_steps: steps });
    updateTotalTime(steps);
  };

  // Update recipe step
  const updateRecipeStep = (index: number, field: string, value: string | number) => {
    const steps = [...recipe.recipe_steps];
    steps[index] = { ...steps[index], [field]: value };

    setRecipe({ ...recipe, recipe_steps: steps });

    if (field === "duration_s") {
      updateTotalTime(steps);
    }
  };

  // Remove recipe step
  const removeRecipeStep = (index: number) => {
    const steps = recipe.recipe_steps.filter((_, i) => i !== index);
    steps.forEach((step, i) => {
      step.step = i + 1;
    });

    setRecipe({ ...recipe, recipe_steps: steps });
    updateTotalTime(steps);
  };

  // Update total process time
  const updateTotalTime = (steps: RecipeStep[]) => {
    const total = steps.reduce((sum, step) => sum + step.duration_s, 0);
    setRecipe((prev) => ({ ...prev, process_time_s: total }));
  };

  // Add tag
  const addTag = () => {
    if (newTag && !recipe.tags.includes(newTag)) {
      setRecipe({ ...recipe, tags: [...recipe.tags, newTag] });
      setNewTag("");
    }
  };

  // Remove tag
  const removeTag = (tag: string) => {
    setRecipe({ ...recipe, tags: recipe.tags.filter((t) => t !== tag) });
  };

  // Validate recipe
  const validateRecipe = (): boolean => {
    const errors: string[] = [];

    if (!recipe.name.trim()) {
      errors.push("Recipe name is required");
    }

    if (recipe.temperature_profile.zones.length === 0) {
      errors.push("At least one temperature zone is required");
    }

    if (recipe.gas_flows.gases.length === 0) {
      errors.push("At least one gas flow is required");
    }

    if (recipe.recipe_steps.length === 0) {
      errors.push("At least one recipe step is required");
    }

    if (recipe.process_time_s <= 0) {
      errors.push("Process time must be greater than zero");
    }

    setValidationErrors(errors);
    return errors.length === 0;
  };

  // Handle save
  const handleSave = () => {
    if (validateRecipe()) {
      onSave(recipe);
      toast.success("Recipe saved successfully");
    } else {
      toast.error("Please fix validation errors");
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">
            {initialRecipe ? "Edit Recipe" : "Create New Recipe"}
          </h2>
          <p className="text-sm text-muted-foreground">
            Configure CVD process parameters and steps
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button onClick={handleSave}>
            <Save className="mr-2 h-4 w-4" />
            Save Recipe
          </Button>
        </div>
      </div>

      {/* Validation Errors */}
      {validationErrors.length > 0 && (
        <Card className="border-red-500">
          <CardHeader>
            <CardTitle className="flex items-center text-red-600">
              <AlertCircle className="mr-2 h-4 w-4" />
              Validation Errors
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-disc list-inside space-y-1">
              {validationErrors.map((error, idx) => (
                <li key={idx} className="text-sm text-red-600">
                  {error}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Basic Information */}
      <Card>
        <CardHeader>
          <CardTitle>Basic Information</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="name">Recipe Name *</Label>
              <Input
                id="name"
                value={recipe.name}
                onChange={(e) => setRecipe({ ...recipe, name: e.target.value })}
                placeholder="e.g., LPCVD Si Baseline"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="version">Version</Label>
              <Input
                id="version"
                value={recipe.version}
                onChange={(e) => setRecipe({ ...recipe, version: e.target.value })}
                placeholder="1.0"
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">Description</Label>
            <Textarea
              id="description"
              value={recipe.description}
              onChange={(e) => setRecipe({ ...recipe, description: e.target.value })}
              placeholder="Describe the recipe purpose and key parameters..."
              rows={3}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="target-thickness">Target Thickness (nm)</Label>
              <Input
                id="target-thickness"
                type="number"
                value={recipe.target_thickness_nm || ""}
                onChange={(e) =>
                  setRecipe({
                    ...recipe,
                    target_thickness_nm: parseFloat(e.target.value) || undefined,
                  })
                }
                placeholder="100"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="target-uniformity">Target Uniformity (%)</Label>
              <Input
                id="target-uniformity"
                type="number"
                value={recipe.target_uniformity_pct || ""}
                onChange={(e) =>
                  setRecipe({
                    ...recipe,
                    target_uniformity_pct: parseFloat(e.target.value) || undefined,
                  })
                }
                placeholder="5"
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label>Tags</Label>
            <div className="flex gap-2">
              <Input
                value={newTag}
                onChange={(e) => setNewTag(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && addTag()}
                placeholder="Add tag..."
              />
              <Button onClick={addTag} size="sm">
                <Plus className="h-4 w-4" />
              </Button>
            </div>
            <div className="flex flex-wrap gap-2 mt-2">
              {recipe.tags.map((tag) => (
                <Badge key={tag} variant="secondary">
                  {tag}
                  <button
                    onClick={() => removeTag(tag)}
                    className="ml-2 hover:text-destructive"
                  >
                    ×
                  </button>
                </Badge>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Process Parameters */}
      <Tabs defaultValue="temperature" className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="temperature">
            <Thermometer className="mr-2 h-4 w-4" />
            Temperature
          </TabsTrigger>
          <TabsTrigger value="pressure">
            <Gauge className="mr-2 h-4 w-4" />
            Pressure
          </TabsTrigger>
          <TabsTrigger value="gas">
            <Wind className="mr-2 h-4 w-4" />
            Gas Flows
          </TabsTrigger>
          {hasPlasma && (
            <TabsTrigger value="plasma">
              <Zap className="mr-2 h-4 w-4" />
              Plasma
            </TabsTrigger>
          )}
          <TabsTrigger value="steps">
            <Clock className="mr-2 h-4 w-4" />
            Steps
          </TabsTrigger>
        </TabsList>

        {/* Temperature Tab */}
        <TabsContent value="temperature">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Temperature Profile</CardTitle>
                <Button onClick={addTemperatureZone} size="sm">
                  <Plus className="mr-2 h-4 w-4" />
                  Add Zone
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {recipe.temperature_profile.zones.map((zone, index) => (
                <div
                  key={index}
                  className="flex items-center gap-4 p-4 border rounded-lg"
                >
                  <div className="flex-1 grid grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <Label>Zone {zone.zone}</Label>
                      <Input value={`Zone ${zone.zone}`} disabled />
                    </div>
                    <div className="space-y-2">
                      <Label>Setpoint (°C)</Label>
                      <Input
                        type="number"
                        value={zone.setpoint_c}
                        onChange={(e) =>
                          updateTemperatureZone(
                            index,
                            "setpoint_c",
                            parseFloat(e.target.value)
                          )
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Ramp Rate (°C/min)</Label>
                      <Input
                        type="number"
                        value={zone.ramp_rate_c_per_min}
                        onChange={(e) =>
                          updateTemperatureZone(
                            index,
                            "ramp_rate_c_per_min",
                            parseFloat(e.target.value)
                          )
                        }
                      />
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => removeTemperatureZone(index)}
                    disabled={recipe.temperature_profile.zones.length === 1}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}

              <Separator />

              <div className="space-y-2">
                <Label>Soak Time (seconds)</Label>
                <Input
                  type="number"
                  value={recipe.temperature_profile.soak_time_s}
                  onChange={(e) =>
                    setRecipe({
                      ...recipe,
                      temperature_profile: {
                        ...recipe.temperature_profile,
                        soak_time_s: parseFloat(e.target.value),
                      },
                    })
                  }
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Pressure Tab */}
        <TabsContent value="pressure">
          <Card>
            <CardHeader>
              <CardTitle>Pressure Profile</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>Base Pressure (Pa)</Label>
                  <Input
                    type="number"
                    value={recipe.pressure_profile.base_pressure_pa}
                    onChange={(e) =>
                      setRecipe({
                        ...recipe,
                        pressure_profile: {
                          ...recipe.pressure_profile,
                          base_pressure_pa: parseFloat(e.target.value),
                        },
                      })
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label>Process Pressure (Pa)</Label>
                  <Input
                    type="number"
                    value={recipe.pressure_profile.process_pressure_pa}
                    onChange={(e) =>
                      setRecipe({
                        ...recipe,
                        pressure_profile: {
                          ...recipe.pressure_profile,
                          process_pressure_pa: parseFloat(e.target.value),
                        },
                      })
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label>Throttle Valve (%)</Label>
                  <Input
                    type="number"
                    value={recipe.pressure_profile.throttle_valve_position_pct}
                    onChange={(e) =>
                      setRecipe({
                        ...recipe,
                        pressure_profile: {
                          ...recipe.pressure_profile,
                          throttle_valve_position_pct: parseFloat(e.target.value),
                        },
                      })
                    }
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Gas Flows Tab */}
        <TabsContent value="gas">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Gas Flows</CardTitle>
                <Button onClick={addGasFlow} size="sm">
                  <Plus className="mr-2 h-4 w-4" />
                  Add Gas
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {recipe.gas_flows.gases.map((gas, index) => (
                <div
                  key={index}
                  className="flex items-center gap-4 p-4 border rounded-lg"
                >
                  <div className="flex-1 grid grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <Label>Gas Name</Label>
                      <Select
                        value={gas.name}
                        onValueChange={(value) => updateGasFlow(index, "name", value)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="SiH4">SiH4 (Silane)</SelectItem>
                          <SelectItem value="NH3">NH3 (Ammonia)</SelectItem>
                          <SelectItem value="N2">N2 (Nitrogen)</SelectItem>
                          <SelectItem value="O2">O2 (Oxygen)</SelectItem>
                          <SelectItem value="N2O">N2O (Nitrous Oxide)</SelectItem>
                          <SelectItem value="Ar">Ar (Argon)</SelectItem>
                          <SelectItem value="H2">H2 (Hydrogen)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Flow Rate (sccm)</Label>
                      <Input
                        type="number"
                        value={gas.flow_sccm}
                        onChange={(e) =>
                          updateGasFlow(index, "flow_sccm", parseFloat(e.target.value))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>MFC ID</Label>
                      <Input
                        value={gas.mfc_id}
                        onChange={(e) => updateGasFlow(index, "mfc_id", e.target.value)}
                      />
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => removeGasFlow(index)}
                    disabled={recipe.gas_flows.gases.length === 1}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}

              <Separator />

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Carrier Gas</Label>
                  <Select
                    value={recipe.gas_flows.carrier_gas}
                    onValueChange={(value) =>
                      setRecipe({
                        ...recipe,
                        gas_flows: { ...recipe.gas_flows, carrier_gas: value },
                      })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="N2">N2 (Nitrogen)</SelectItem>
                      <SelectItem value="Ar">Ar (Argon)</SelectItem>
                      <SelectItem value="H2">H2 (Hydrogen)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Carrier Flow (sccm)</Label>
                  <Input
                    type="number"
                    value={recipe.gas_flows.carrier_flow_sccm}
                    onChange={(e) =>
                      setRecipe({
                        ...recipe,
                        gas_flows: {
                          ...recipe.gas_flows,
                          carrier_flow_sccm: parseFloat(e.target.value),
                        },
                      })
                    }
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Plasma Tab */}
        {hasPlasma && (
          <TabsContent value="plasma">
            <Card>
              <CardHeader>
                <CardTitle>Plasma Settings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>RF Power (W)</Label>
                    <Input
                      type="number"
                      value={recipe.plasma_settings?.rf_power_w || ""}
                      onChange={(e) =>
                        setRecipe({
                          ...recipe,
                          plasma_settings: {
                            ...recipe.plasma_settings,
                            rf_power_w: parseFloat(e.target.value) || undefined,
                          },
                        })
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Frequency (MHz)</Label>
                    <Input
                      type="number"
                      value={recipe.plasma_settings?.frequency_mhz || ""}
                      onChange={(e) =>
                        setRecipe({
                          ...recipe,
                          plasma_settings: {
                            ...recipe.plasma_settings,
                            frequency_mhz: parseFloat(e.target.value) || undefined,
                          },
                        })
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>DC Bias (V)</Label>
                    <Input
                      type="number"
                      value={recipe.plasma_settings?.bias_voltage_v || ""}
                      onChange={(e) =>
                        setRecipe({
                          ...recipe,
                          plasma_settings: {
                            ...recipe.plasma_settings,
                            bias_voltage_v: parseFloat(e.target.value) || undefined,
                          },
                        })
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Matching Mode</Label>
                    <Select
                      value={recipe.plasma_settings?.matching_mode || "auto"}
                      onValueChange={(value) =>
                        setRecipe({
                          ...recipe,
                          plasma_settings: {
                            ...recipe.plasma_settings,
                            matching_mode: value,
                          },
                        })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="auto">Auto</SelectItem>
                        <SelectItem value="manual">Manual</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        )}

        {/* Steps Tab */}
        <TabsContent value="steps">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Recipe Steps</CardTitle>
                  <p className="text-sm text-muted-foreground mt-1">
                    Total Process Time: {Math.round(recipe.process_time_s / 60)} minutes (
                    {recipe.process_time_s}s)
                  </p>
                </div>
                <Button onClick={addRecipeStep} size="sm">
                  <Plus className="mr-2 h-4 w-4" />
                  Add Step
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {recipe.recipe_steps.map((step, index) => (
                <div
                  key={index}
                  className="flex items-center gap-4 p-4 border rounded-lg"
                >
                  <div className="flex-shrink-0 w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center font-bold">
                    {step.step}
                  </div>
                  <div className="flex-1 grid grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <Label>Step Name</Label>
                      <Input
                        value={step.name}
                        onChange={(e) => updateRecipeStep(index, "name", e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Action</Label>
                      <Select
                        value={step.action}
                        onValueChange={(value) =>
                          updateRecipeStep(index, "action", value)
                        }
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="evacuate">Evacuate</SelectItem>
                          <SelectItem value="ramp_temperature">Ramp Temperature</SelectItem>
                          <SelectItem value="stabilize">Stabilize</SelectItem>
                          <SelectItem value="deposit">Deposit</SelectItem>
                          <SelectItem value="cool">Cool</SelectItem>
                          <SelectItem value="vent">Vent</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Duration (s)</Label>
                      <Input
                        type="number"
                        value={step.duration_s}
                        onChange={(e) =>
                          updateRecipeStep(
                            index,
                            "duration_s",
                            parseFloat(e.target.value)
                          )
                        }
                      />
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => removeRecipeStep(index)}
                    disabled={recipe.recipe_steps.length === 1}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
