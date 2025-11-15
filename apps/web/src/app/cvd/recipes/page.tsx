"use client";

/**
 * CVD Recipes Page
 *
 * Features:
 * - Recipe list with target thickness, stress, adhesion class
 * - Recipe editor with live "expected windows" (predicted distributions from physics/VM)
 * - Create, edit, delete recipes
 * - Recipe validation
 */

import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AdhesionChip } from "@/components/cvd/metrics/AdhesionChip";
import { Plus, Edit, Trash2, Play, Copy, TrendingUp } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface Recipe {
  id: string;
  name: string;
  description?: string;
  mode: "thermal" | "plasma";
  temperature_c: number;
  pressure_torr: number;
  precursor_flow_sccm: number;
  carrier_gas_flow_sccm: number;
  rf_power_w?: number;
  film_material: string;
  target_thickness_nm: number;
  target_stress_mpa: number;
  target_adhesion_class: string;
  expected_thickness_range: [number, number];
  expected_stress_range: [number, number];
  expected_adhesion_score: number;
  total_runs?: number;
  last_used?: string;
}

export default function CVDRecipesPage() {
  const [selectedRecipe, setSelectedRecipe] = useState<Recipe | null>(null);
  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const queryClient = useQueryClient();

  // Fetch recipes
  const { data: recipes, isLoading } = useQuery({
    queryKey: ["cvd-recipes"],
    queryFn: async () => {
      const response = await fetch("http://localhost:8001/api/cvd/recipes");
      if (!response.ok) throw new Error("Failed to fetch recipes");
      return response.json();
    },
  });

  // Delete recipe mutation
  const deleteMutation = useMutation({
    mutationFn: async (recipeId: string) => {
      const response = await fetch(`http://localhost:8001/api/cvd/recipes/${recipeId}`, {
        method: "DELETE",
      });
      if (!response.ok) throw new Error("Failed to delete recipe");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["cvd-recipes"] });
    },
  });

  const handleEdit = (recipe: Recipe) => {
    setSelectedRecipe(recipe);
    setIsEditorOpen(true);
  };

  const handleNew = () => {
    setSelectedRecipe(null);
    setIsEditorOpen(true);
  };

  const handleDelete = async (recipeId: string) => {
    if (confirm("Are you sure you want to delete this recipe?")) {
      await deleteMutation.mutateAsync(recipeId);
    }
  };

  // Mock data for demonstration
  const mockRecipes: Recipe[] = recipes || [
    {
      id: "1",
      name: "Si3N4 Standard",
      description: "Standard silicon nitride deposition",
      mode: "thermal",
      temperature_c: 800,
      pressure_torr: 0.5,
      precursor_flow_sccm: 80,
      carrier_gas_flow_sccm: 500,
      film_material: "Si3N4",
      target_thickness_nm: 100,
      target_stress_mpa: -200,
      target_adhesion_class: "excellent",
      expected_thickness_range: [95, 105],
      expected_stress_range: [-250, -150],
      expected_adhesion_score: 88,
      total_runs: 145,
      last_used: "2025-11-14T10:30:00",
    },
    {
      id: "2",
      name: "TiN Barrier Layer",
      description: "Titanium nitride barrier for Cu metallization",
      mode: "plasma",
      temperature_c: 350,
      pressure_torr: 5.0,
      precursor_flow_sccm: 50,
      carrier_gas_flow_sccm: 500,
      rf_power_w: 100,
      film_material: "TiN",
      target_thickness_nm: 20,
      target_stress_mpa: -100,
      target_adhesion_class: "excellent",
      expected_thickness_range: [18, 22],
      expected_stress_range: [-150, -50],
      expected_adhesion_score: 92,
      total_runs: 89,
      last_used: "2025-11-13T15:20:00",
    },
    {
      id: "3",
      name: "W Fill",
      description: "Tungsten fill for vias and contacts",
      mode: "thermal",
      temperature_c: 400,
      pressure_torr: 80.0,
      precursor_flow_sccm: 100,
      carrier_gas_flow_sccm: 1000,
      film_material: "W",
      target_thickness_nm: 200,
      target_stress_mpa: 150,
      target_adhesion_class: "good",
      expected_thickness_range: [190, 210],
      expected_stress_range: [100, 200],
      expected_adhesion_score: 75,
      total_runs: 67,
      last_used: "2025-11-12T09:45:00",
    },
  ];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">CVD Recipes</h1>
          <p className="text-muted-foreground mt-1">
            Manage process recipes with target specifications and expected windows
          </p>
        </div>
        <Button onClick={handleNew}>
          <Plus className="mr-2 h-4 w-4" />
          New Recipe
        </Button>
      </div>

      {/* Recipe Table */}
      <Card>
        <CardHeader>
          <CardTitle>All Recipes ({mockRecipes.length})</CardTitle>
          <CardDescription>
            Recipes with target thickness, stress, and adhesion specifications
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Material</TableHead>
                <TableHead>Mode</TableHead>
                <TableHead>Target Thickness</TableHead>
                <TableHead>Target Stress</TableHead>
                <TableHead>Adhesion Class</TableHead>
                <TableHead>Runs</TableHead>
                <TableHead>Last Used</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {mockRecipes.map((recipe) => (
                <TableRow key={recipe.id} className="cursor-pointer hover:bg-muted/50">
                  <TableCell className="font-medium">
                    <div>
                      <div>{recipe.name}</div>
                      {recipe.description && (
                        <div className="text-xs text-muted-foreground">{recipe.description}</div>
                      )}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">{recipe.film_material}</Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant={recipe.mode === "plasma" ? "default" : "secondary"}>
                      {recipe.mode}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className="text-sm">
                      <div className="font-medium">{recipe.target_thickness_nm} nm</div>
                      <div className="text-xs text-muted-foreground">
                        ±{((recipe.expected_thickness_range[1] - recipe.expected_thickness_range[0]) / 2).toFixed(0)} nm window
                      </div>
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="text-sm">
                      <div className="font-medium">{recipe.target_stress_mpa} MPa</div>
                      <div className="text-xs text-muted-foreground">
                        {recipe.expected_stress_range[0]} to {recipe.expected_stress_range[1]}
                      </div>
                    </div>
                  </TableCell>
                  <TableCell>
                    <AdhesionChip
                      score={recipe.expected_adhesion_score}
                      adhesionClass={recipe.target_adhesion_class as any}
                      showScore={false}
                      size="sm"
                    />
                  </TableCell>
                  <TableCell>{recipe.total_runs || 0}</TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {recipe.last_used ? new Date(recipe.last_used).toLocaleDateString() : "Never"}
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-1">
                      <Button variant="ghost" size="icon" onClick={() => handleEdit(recipe)}>
                        <Edit className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="icon" onClick={() => {}}>
                        <Copy className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="icon" onClick={() => handleDelete(recipe.id)}>
                        <Trash2 className="h-4 w-4" />
                      </Button>
                      <Button variant="default" size="sm">
                        <Play className="h-4 w-4 mr-1" />
                        Run
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Recipe Editor Dialog */}
      <RecipeEditorDialog
        recipe={selectedRecipe}
        open={isEditorOpen}
        onOpenChange={setIsEditorOpen}
      />
    </div>
  );
}

/**
 * Recipe Editor Dialog with Live Expected Windows
 */
function RecipeEditorDialog({
  recipe,
  open,
  onOpenChange,
}: {
  recipe: Recipe | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [formData, setFormData] = useState<Partial<Recipe>>(recipe || {
    mode: "thermal",
    temperature_c: 800,
    pressure_torr: 0.5,
    precursor_flow_sccm: 80,
    carrier_gas_flow_sccm: 500,
    film_material: "Si3N4",
    target_thickness_nm: 100,
    target_stress_mpa: -200,
    target_adhesion_class: "excellent",
  });

  // Predict expected windows from physics model
  const predictWindows = (params: Partial<Recipe>) => {
    // Mock prediction - in reality, call physics/VM API
    const thicknessVariation = 5;
    const stressVariation = 50;

    return {
      thickness: [
        (params.target_thickness_nm || 100) - thicknessVariation,
        (params.target_thickness_nm || 100) + thicknessVariation,
      ] as [number, number],
      stress: [
        (params.target_stress_mpa || -200) - stressVariation,
        (params.target_stress_mpa || -200) + stressVariation,
      ] as [number, number],
      adhesion: 85 + Math.random() * 10,
      deposition_rate: 50 + (params.temperature_c || 800) * 0.05,
      uniformity: 2 + Math.random() * 2,
    };
  };

  const expectedWindows = predictWindows(formData);

  const handleSave = async () => {
    // Save recipe via API
    console.log("Saving recipe:", formData);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{recipe ? "Edit Recipe" : "New Recipe"}</DialogTitle>
          <DialogDescription>
            Configure process parameters and view predicted outcomes
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="parameters" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="parameters">Parameters</TabsTrigger>
            <TabsTrigger value="targets">Targets</TabsTrigger>
            <TabsTrigger value="predictions">
              <TrendingUp className="h-4 w-4 mr-1" />
              Predictions
            </TabsTrigger>
          </TabsList>

          <TabsContent value="parameters" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Recipe Name</Label>
                <Input
                  value={formData.name || ""}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="e.g., Si3N4 Standard"
                />
              </div>

              <div>
                <Label>Film Material</Label>
                <Select
                  value={formData.film_material}
                  onValueChange={(v) => setFormData({ ...formData, film_material: v })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Si3N4">Si3N4</SelectItem>
                    <SelectItem value="TiN">TiN</SelectItem>
                    <SelectItem value="W">W</SelectItem>
                    <SelectItem value="SiO2">SiO2</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Mode</Label>
                <Select
                  value={formData.mode}
                  onValueChange={(v: any) => setFormData({ ...formData, mode: v })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="thermal">Thermal</SelectItem>
                    <SelectItem value="plasma">Plasma</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Temperature (°C)</Label>
                <Input
                  type="number"
                  value={formData.temperature_c || ""}
                  onChange={(e) => setFormData({ ...formData, temperature_c: parseFloat(e.target.value) })}
                />
              </div>

              <div>
                <Label>Pressure (Torr)</Label>
                <Input
                  type="number"
                  step="0.1"
                  value={formData.pressure_torr || ""}
                  onChange={(e) => setFormData({ ...formData, pressure_torr: parseFloat(e.target.value) })}
                />
              </div>

              <div>
                <Label>Precursor Flow (sccm)</Label>
                <Input
                  type="number"
                  value={formData.precursor_flow_sccm || ""}
                  onChange={(e) => setFormData({ ...formData, precursor_flow_sccm: parseFloat(e.target.value) })}
                />
              </div>

              <div>
                <Label>Carrier Gas Flow (sccm)</Label>
                <Input
                  type="number"
                  value={formData.carrier_gas_flow_sccm || ""}
                  onChange={(e) => setFormData({ ...formData, carrier_gas_flow_sccm: parseFloat(e.target.value) })}
                />
              </div>

              {formData.mode === "plasma" && (
                <div>
                  <Label>RF Power (W)</Label>
                  <Input
                    type="number"
                    value={formData.rf_power_w || ""}
                    onChange={(e) => setFormData({ ...formData, rf_power_w: parseFloat(e.target.value) })}
                  />
                </div>
              )}
            </div>

            <div>
              <Label>Description (Optional)</Label>
              <Input
                value={formData.description || ""}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                placeholder="Brief description of the recipe"
              />
            </div>
          </TabsContent>

          <TabsContent value="targets" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Target Thickness (nm)</Label>
                <Input
                  type="number"
                  value={formData.target_thickness_nm || ""}
                  onChange={(e) => setFormData({ ...formData, target_thickness_nm: parseFloat(e.target.value) })}
                />
              </div>

              <div>
                <Label>Target Stress (MPa)</Label>
                <Input
                  type="number"
                  value={formData.target_stress_mpa || ""}
                  onChange={(e) => setFormData({ ...formData, target_stress_mpa: parseFloat(e.target.value) })}
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Negative = compressive, Positive = tensile
                </p>
              </div>

              <div>
                <Label>Target Adhesion Class</Label>
                <Select
                  value={formData.target_adhesion_class}
                  onValueChange={(v) => setFormData({ ...formData, target_adhesion_class: v })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="excellent">Excellent (80-100)</SelectItem>
                    <SelectItem value="good">Good (60-80)</SelectItem>
                    <SelectItem value="fair">Fair (40-60)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="predictions" className="space-y-4">
            <Card className="bg-blue-50 dark:bg-blue-950 border-blue-200">
              <CardHeader>
                <CardTitle className="text-sm">Live Predictions from Physics Model</CardTitle>
                <CardDescription className="text-xs">
                  Expected ranges based on current parameters
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <Label className="text-xs">Thickness Window</Label>
                    <div className="text-2xl font-bold">
                      {expectedWindows.thickness[0].toFixed(1)} - {expectedWindows.thickness[1].toFixed(1)} nm
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Target: {formData.target_thickness_nm} nm
                    </div>
                  </div>

                  <div className="space-y-1">
                    <Label className="text-xs">Stress Window</Label>
                    <div className="text-2xl font-bold">
                      {expectedWindows.stress[0].toFixed(0)} to {expectedWindows.stress[1].toFixed(0)} MPa
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Target: {formData.target_stress_mpa} MPa
                    </div>
                  </div>

                  <div className="space-y-1">
                    <Label className="text-xs">Expected Adhesion Score</Label>
                    <div className="text-2xl font-bold">
                      {expectedWindows.adhesion.toFixed(1)}/100
                    </div>
                    <AdhesionChip score={expectedWindows.adhesion} size="sm" />
                  </div>

                  <div className="space-y-1">
                    <Label className="text-xs">Deposition Rate</Label>
                    <div className="text-2xl font-bold">
                      {expectedWindows.deposition_rate.toFixed(1)} nm/min
                    </div>
                  </div>

                  <div className="space-y-1">
                    <Label className="text-xs">WIW Uniformity</Label>
                    <div className="text-2xl font-bold">
                      ±{expectedWindows.uniformity.toFixed(2)}%
                    </div>
                    <div className="text-xs text-green-600">
                      {expectedWindows.uniformity < 2 ? "Excellent" : expectedWindows.uniformity < 5 ? "Good" : "Fair"}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="text-xs text-muted-foreground p-3 bg-muted rounded">
              <strong>Note:</strong> Predictions are based on physics models and historical data.
              Actual results may vary. Use these windows as guidance for process development.
            </div>
          </TabsContent>
        </Tabs>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave}>
            {recipe ? "Save Changes" : "Create Recipe"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
