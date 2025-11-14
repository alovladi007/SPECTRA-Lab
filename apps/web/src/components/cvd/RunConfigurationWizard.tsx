"use client";

import React, { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  ChevronLeft,
  ChevronRight,
  Check,
  Loader2,
  AlertCircle,
  CheckCircle,
  Beaker,
  Settings,
  Layers,
  Play,
} from "lucide-react";
import { cvdApi, CVDRecipe, CVDProcessMode } from "@/lib/api/cvd";

interface RunConfigurationWizardProps {
  organizationId: string;
  onRunCreated?: (runIds: string[]) => void;
  onCancel?: () => void;
}

interface WaferEntry {
  id: string;
  slotNumber: number;
}

interface ToolInfo {
  id: string;
  name: string;
  state: string;
  current_run_id?: string;
  message: string;
}

const WIZARD_STEPS = [
  { id: 1, title: "Select Recipe", icon: Beaker },
  { id: 2, title: "Select Tool", icon: Settings },
  { id: 3, title: "Configure Wafers", icon: Layers },
  { id: 4, title: "Review & Launch", icon: Play },
];

export default function RunConfigurationWizard({
  organizationId,
  onRunCreated,
  onCancel,
}: RunConfigurationWizardProps) {
  const [currentStep, setCurrentStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Step 1: Recipe Selection
  const [recipes, setRecipes] = useState<CVDRecipe[]>([]);
  const [selectedRecipe, setSelectedRecipe] = useState<CVDRecipe | null>(null);
  const [recipeFilter, setRecipeFilter] = useState("");

  // Step 2: Tool Selection
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [selectedToolId, setSelectedToolId] = useState<string>("");
  const [toolStatus, setToolStatus] = useState<{
    [key: string]: { state: string; message: string };
  }>({});

  // Step 3: Wafer Configuration
  const [lotId, setLotId] = useState("");
  const [wafers, setWafers] = useState<WaferEntry[]>([
    { id: "", slotNumber: 1 },
  ]);
  const [operatorId, setOperatorId] = useState("");

  // Step 4: Review
  const [notes, setNotes] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Load recipes on mount
  useEffect(() => {
    loadRecipes();
  }, [organizationId]);

  // Load tools when recipe is selected
  useEffect(() => {
    if (selectedRecipe) {
      loadAvailableTools();
    }
  }, [selectedRecipe]);

  const loadRecipes = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await cvdApi.getRecipes({
        org_id: organizationId,
      });
      setRecipes(data);
    } catch (err: any) {
      setError(`Failed to load recipes: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const loadAvailableTools = async () => {
    // In a real implementation, this would call an API to get tools
    // compatible with the selected process mode
    // For now, we'll create mock tools
    const mockTools: ToolInfo[] = [
      {
        id: "tool-001",
        name: "CVD Chamber A1",
        state: "IDLE",
        message: "Ready for operation",
      },
      {
        id: "tool-002",
        name: "CVD Chamber A2",
        state: "IDLE",
        message: "Ready for operation",
      },
      {
        id: "tool-003",
        name: "CVD Chamber B1",
        state: "RUNNING",
        current_run_id: "run-123",
        message: "Processing run #123",
      },
    ];

    setTools(mockTools);

    // Load status for each tool
    const statusMap: { [key: string]: { state: string; message: string } } = {};
    for (const tool of mockTools) {
      try {
        const status = await cvdApi.getToolStatus(tool.id);
        statusMap[tool.id] = {
          state: status.state,
          message: status.message,
        };
      } catch (err) {
        // Use mock data if API fails
        statusMap[tool.id] = {
          state: tool.state,
          message: tool.message,
        };
      }
    }
    setToolStatus(statusMap);
  };

  const filteredRecipes = recipes.filter((recipe) => {
    if (!recipeFilter) return true;
    return (
      recipe.name.toLowerCase().includes(recipeFilter.toLowerCase()) ||
      recipe.description?.toLowerCase().includes(recipeFilter.toLowerCase()) ||
      recipe.tags.some((tag) =>
        tag.toLowerCase().includes(recipeFilter.toLowerCase())
      )
    );
  });

  const addWafer = () => {
    const nextSlot = wafers.length + 1;
    setWafers([...wafers, { id: "", slotNumber: nextSlot }]);
  };

  const removeWafer = (index: number) => {
    const updated = wafers.filter((_, i) => i !== index);
    // Re-number slots
    updated.forEach((wafer, i) => {
      wafer.slotNumber = i + 1;
    });
    setWafers(updated);
  };

  const updateWaferId = (index: number, id: string) => {
    const updated = [...wafers];
    updated[index].id = id;
    setWafers(updated);
  };

  const canProceedToStep2 = () => {
    return selectedRecipe !== null;
  };

  const canProceedToStep3 = () => {
    return selectedToolId !== "";
  };

  const canProceedToStep4 = () => {
    return (
      lotId.trim() !== "" &&
      wafers.length > 0 &&
      wafers.every((w) => w.id.trim() !== "")
    );
  };

  const canSubmit = () => {
    return canProceedToStep4() && selectedRecipe && selectedToolId;
  };

  const handleNext = () => {
    setError(null);
    if (currentStep < WIZARD_STEPS.length) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrevious = () => {
    setError(null);
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = async () => {
    if (!selectedRecipe || !selectedToolId) return;

    setIsSubmitting(true);
    setError(null);

    try {
      const waferIds = wafers.map((w) => w.id);

      const response = await cvdApi.createBatchRuns({
        cvd_recipe_id: selectedRecipe.id,
        process_mode_id: selectedRecipe.process_mode_id,
        instrument_id: selectedToolId,
        org_id: organizationId,
        lot_id: lotId,
        wafer_ids: waferIds,
        operator_id: operatorId || undefined,
      });

      // Success!
      if (onRunCreated) {
        onRunCreated(response.run_ids);
      }
    } catch (err: any) {
      setError(`Failed to create runs: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderStep1 = () => (
    <div className="space-y-4">
      <div>
        <Label htmlFor="recipe-filter">Search Recipes</Label>
        <Input
          id="recipe-filter"
          placeholder="Filter by name, description, or tags..."
          value={recipeFilter}
          onChange={(e) => setRecipeFilter(e.target.value)}
        />
      </div>

      <div className="space-y-2 max-h-96 overflow-y-auto">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        ) : filteredRecipes.length === 0 ? (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              No recipes found. Try adjusting your search.
            </AlertDescription>
          </Alert>
        ) : (
          filteredRecipes.map((recipe) => (
            <Card
              key={recipe.id}
              className={`cursor-pointer transition-colors ${
                selectedRecipe?.id === recipe.id
                  ? "border-blue-500 bg-blue-50"
                  : "hover:bg-gray-50"
              }`}
              onClick={() => setSelectedRecipe(recipe)}
            >
              <CardHeader className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <CardTitle className="text-base">{recipe.name}</CardTitle>
                    {recipe.description && (
                      <CardDescription className="text-sm mt-1">
                        {recipe.description}
                      </CardDescription>
                    )}
                  </div>
                  {selectedRecipe?.id === recipe.id && (
                    <CheckCircle className="h-5 w-5 text-blue-500 flex-shrink-0 ml-2" />
                  )}
                </div>
                <div className="flex flex-wrap gap-1 mt-2">
                  {recipe.tags.map((tag) => (
                    <Badge key={tag} variant="secondary" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                  {recipe.is_baseline && (
                    <Badge variant="outline" className="text-xs">
                      Baseline
                    </Badge>
                  )}
                  {recipe.is_golden && (
                    <Badge variant="outline" className="text-xs bg-yellow-50">
                      Golden
                    </Badge>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-2 mt-2 text-xs text-gray-600">
                  <div>Process Time: {recipe.process_time_s}s</div>
                  <div>Runs: {recipe.run_count}</div>
                  {recipe.target_thickness_nm && (
                    <div>Target: {recipe.target_thickness_nm}nm</div>
                  )}
                  {recipe.target_uniformity_pct && (
                    <div>Uniformity: {recipe.target_uniformity_pct}%</div>
                  )}
                </div>
              </CardHeader>
            </Card>
          ))
        )}
      </div>
    </div>
  );

  const renderStep2 = () => (
    <div className="space-y-4">
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Select a tool that is compatible with the selected recipe and
          currently available.
        </AlertDescription>
      </Alert>

      <div className="space-y-2">
        {tools.map((tool) => {
          const status = toolStatus[tool.id];
          const isAvailable = status?.state === "IDLE";

          return (
            <Card
              key={tool.id}
              className={`cursor-pointer transition-colors ${
                selectedToolId === tool.id
                  ? "border-blue-500 bg-blue-50"
                  : isAvailable
                  ? "hover:bg-gray-50"
                  : "opacity-50 cursor-not-allowed"
              }`}
              onClick={() => {
                if (isAvailable) {
                  setSelectedToolId(tool.id);
                }
              }}
            >
              <CardHeader className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base">{tool.name}</CardTitle>
                    <CardDescription className="text-sm mt-1">
                      {status?.message || tool.message}
                    </CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge
                      variant={isAvailable ? "default" : "secondary"}
                      className={
                        isAvailable ? "bg-green-500" : "bg-yellow-500"
                      }
                    >
                      {status?.state || tool.state}
                    </Badge>
                    {selectedToolId === tool.id && (
                      <CheckCircle className="h-5 w-5 text-blue-500" />
                    )}
                  </div>
                </div>
              </CardHeader>
            </Card>
          );
        })}
      </div>

      {tools.filter((t) => toolStatus[t.id]?.state === "IDLE").length === 0 && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            No tools are currently available. Please wait for a tool to become
            idle or contact your administrator.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );

  const renderStep3 = () => (
    <div className="space-y-4">
      <div>
        <Label htmlFor="lot-id">Lot ID *</Label>
        <Input
          id="lot-id"
          placeholder="Enter lot identifier (e.g., LOT-2024-001)"
          value={lotId}
          onChange={(e) => setLotId(e.target.value)}
          required
        />
      </div>

      <div>
        <Label htmlFor="operator-id">Operator ID (Optional)</Label>
        <Input
          id="operator-id"
          placeholder="Enter operator identifier"
          value={operatorId}
          onChange={(e) => setOperatorId(e.target.value)}
        />
      </div>

      <div className="border-t pt-4">
        <div className="flex items-center justify-between mb-2">
          <Label>Wafer IDs *</Label>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={addWafer}
          >
            Add Wafer
          </Button>
        </div>

        <div className="space-y-2 max-h-64 overflow-y-auto">
          {wafers.map((wafer, index) => (
            <div key={index} className="flex items-center gap-2">
              <div className="w-16">
                <Input
                  value={`Slot ${wafer.slotNumber}`}
                  disabled
                  className="text-xs text-center"
                />
              </div>
              <div className="flex-1">
                <Input
                  placeholder={`Wafer ID (e.g., W${(index + 1)
                    .toString()
                    .padStart(3, "0")})`}
                  value={wafer.id}
                  onChange={(e) => updateWaferId(index, e.target.value)}
                  required
                />
              </div>
              {wafers.length > 1 && (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => removeWafer(index)}
                >
                  Remove
                </Button>
              )}
            </div>
          ))}
        </div>

        {wafers.length === 0 && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              At least one wafer is required.
            </AlertDescription>
          </Alert>
        )}
      </div>
    </div>
  );

  const renderStep4 = () => (
    <div className="space-y-4">
      <Alert>
        <CheckCircle className="h-4 w-4" />
        <AlertDescription>
          Review your configuration before launching the run.
        </AlertDescription>
      </Alert>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Recipe</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div>
            <strong>Name:</strong> {selectedRecipe?.name}
          </div>
          {selectedRecipe?.description && (
            <div>
              <strong>Description:</strong> {selectedRecipe.description}
            </div>
          )}
          <div>
            <strong>Process Time:</strong> {selectedRecipe?.process_time_s}s (
            {Math.round((selectedRecipe?.process_time_s || 0) / 60)} minutes)
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Tool</CardTitle>
        </CardHeader>
        <CardContent>
          <div>
            <strong>Tool:</strong>{" "}
            {tools.find((t) => t.id === selectedToolId)?.name}
          </div>
          <div>
            <strong>Status:</strong>{" "}
            <Badge className="bg-green-500">
              {toolStatus[selectedToolId]?.state || "IDLE"}
            </Badge>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Wafer Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div>
            <strong>Lot ID:</strong> {lotId}
          </div>
          {operatorId && (
            <div>
              <strong>Operator:</strong> {operatorId}
            </div>
          )}
          <div>
            <strong>Wafer Count:</strong> {wafers.length}
          </div>
          <div className="mt-2">
            <strong>Wafer IDs:</strong>
            <div className="flex flex-wrap gap-1 mt-1">
              {wafers.map((wafer) => (
                <Badge key={wafer.id} variant="secondary">
                  {wafer.id}
                </Badge>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <div>
        <Label htmlFor="run-notes">Notes (Optional)</Label>
        <Textarea
          id="run-notes"
          placeholder="Add any notes or special instructions for this run..."
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          rows={3}
        />
      </div>

      <Alert variant="default" className="bg-blue-50">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          This will create {wafers.length} run{wafers.length > 1 ? "s" : ""} for
          lot {lotId}.
        </AlertDescription>
      </Alert>
    </div>
  );

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>Run Configuration Wizard</CardTitle>
        <CardDescription>
          Configure and launch CVD runs in {WIZARD_STEPS.length} easy steps
        </CardDescription>
      </CardHeader>

      <CardContent>
        {/* Progress Indicator */}
        <div className="mb-6">
          <div className="flex items-center justify-between">
            {WIZARD_STEPS.map((step, index) => {
              const Icon = step.icon;
              const isCompleted = currentStep > step.id;
              const isCurrent = currentStep === step.id;

              return (
                <React.Fragment key={step.id}>
                  <div className="flex flex-col items-center">
                    <div
                      className={`w-10 h-10 rounded-full flex items-center justify-center ${
                        isCompleted
                          ? "bg-green-500 text-white"
                          : isCurrent
                          ? "bg-blue-500 text-white"
                          : "bg-gray-200 text-gray-500"
                      }`}
                    >
                      {isCompleted ? (
                        <Check className="h-5 w-5" />
                      ) : (
                        <Icon className="h-5 w-5" />
                      )}
                    </div>
                    <div className="text-xs mt-1 text-center max-w-20">
                      {step.title}
                    </div>
                  </div>
                  {index < WIZARD_STEPS.length - 1 && (
                    <div
                      className={`flex-1 h-0.5 mx-2 ${
                        isCompleted ? "bg-green-500" : "bg-gray-200"
                      }`}
                    />
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Step Content */}
        <div className="min-h-[400px]">
          {currentStep === 1 && renderStep1()}
          {currentStep === 2 && renderStep2()}
          {currentStep === 3 && renderStep3()}
          {currentStep === 4 && renderStep4()}
        </div>
      </CardContent>

      <CardFooter className="flex justify-between">
        <Button
          variant="outline"
          onClick={currentStep === 1 ? onCancel : handlePrevious}
          disabled={isSubmitting}
        >
          <ChevronLeft className="h-4 w-4 mr-1" />
          {currentStep === 1 ? "Cancel" : "Previous"}
        </Button>

        {currentStep < WIZARD_STEPS.length ? (
          <Button
            onClick={handleNext}
            disabled={
              (currentStep === 1 && !canProceedToStep2()) ||
              (currentStep === 2 && !canProceedToStep3()) ||
              (currentStep === 3 && !canProceedToStep4())
            }
          >
            Next
            <ChevronRight className="h-4 w-4 ml-1" />
          </Button>
        ) : (
          <Button
            onClick={handleSubmit}
            disabled={!canSubmit() || isSubmitting}
          >
            {isSubmitting ? (
              <>
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                Launching...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-1" />
                Launch Runs
              </>
            )}
          </Button>
        )}
      </CardFooter>
    </Card>
  );
}
