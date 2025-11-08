# 🚀 AutoML Files - Complete Integration Plan for SPECTRA-Lab
## Comprehensive Analysis of All 53 Uploaded Files

**Date**: November 2025
**Status**: Analysis Complete - Integration Pending
**Total Files**: 53 files

---

## 📋 Executive Summary

You've uploaded 53 files from an **AutoML/RLHF (Reinforcement Learning from Human Feedback)** pipeline designed for semiconductor manufacturing. These are **Python backend** files, not React UI components.

**Key Insight**: These files provide the **backend ML infrastructure** that will power your AutoML, Explainability, and A/B Testing **frontend pages**.

### Integration Strategy:
1. **Backend Integration** (Python → FastAPI services)
2. **Frontend Development** (Create React UI components)
3. **API Connection** (Wire frontend to backend)

---

## 📊 File Categorization (All 53 Files)

### Category Breakdown

| Category | Count | Purpose |
|----------|-------|---------|
| **Core AutoML** (Python) | 5 | Model selection, hyperparameter tuning, NAS |
| **RLHF Training** (Python) | 4 | SFT, RM, PPO, DPO training scripts |
| **Utilities** (Python) | 9 | Data handling, logging, rewards, safety |
| **Evaluation** (Python) | 5 | Metrics, safety checks, tournaments |
| **Package Init** (Python) | 5 | Python module __init__ files |
| **Configuration** (YAML) | 6 | Training configs, AutoML configs |
| **Documentation** (Markdown) | 9 | READMEs, guides, schemas |
| **Data/Prompts** (JSON/JSONL/TXT) | 7 | Training data, prompts, schemas |
| **Build/Deploy** | 3 | Makefile, configs (DeepSpeed, FSDP) |
| **TOTAL** | **53** | Complete AutoML+RLHF pipeline |

---

## 🗂️ Complete File Inventory & Integration Plan

### 1️⃣ CORE AUTOML MODULES (5 Python Files) - **BACKEND PRIORITY #1**

#### File 1: `train_automl.py` (310 lines)
**Purpose**: Main AutoML pipeline orchestration
**What it does**:
- Coordinates model selection, hyperparameter tuning, and NAS
- Loads semiconductor manufacturing data
- Runs complete AutoML workflow
- Generates results and reports

**Integration in SPECTRA-Lab**:
```
Destination: services/analysis/app/ml/automl/train_automl.py
API Endpoint: POST /api/v1/ml/automl/run-pipeline
Frontend Use: AutoML page - "Run AutoML" button triggers this
```

**Frontend Component Needs**:
- Pipeline progress tracker (3 steps: selection → tuning → NAS)
- Real-time log viewer
- Results summary dashboard
- Model comparison charts

---

#### File 2: `auto_selector.py` (367 lines)
**Purpose**: Automatic model selection across 9+ algorithms
**What it does**:
- Evaluates RandomForest, GradientBoosting, Neural Networks, SVR, etc.
- Cross-validation based evaluation
- Performance vs speed trade-off analysis

**Integration in SPECTRA-Lab**:
```
Destination: services/analysis/app/ml/automl/model_selection/auto_selector.py
API Endpoint: POST /api/v1/ml/automl/model-selection
Frontend Use: AutoML page - Model Selection section
```

**Frontend Component Needs**:
- Algorithm comparison table
- Performance metrics visualization (R², RMSE, MAE, MAPE)
- Training time vs accuracy scatter plot
- "Best Model" highlight card

---

#### File 3: `tuner.py` (465 lines)
**Purpose**: Bayesian hyperparameter optimization with Optuna
**What it does**:
- Intelligent parameter search (50-100x faster than grid search)
- Multi-objective optimization support
- Parameter importance analysis
- Early stopping

**Integration in SPECTRA-Lab**:
```
Destination: services/analysis/app/ml/automl/hyperopt/tuner.py
API Endpoint: POST /api/v1/ml/automl/hyperparameter-tuning
Frontend Use: AutoML page - Hyperparameter Tuning section
```

**Frontend Component Needs**:
- Optuna trial visualization (parallel coordinates plot)
- Parameter importance bar chart
- Optimization history line chart
- Best parameters display card
- Trial history table

---

#### File 4: `architecture_search.py` (523 lines)
**Purpose**: Neural Architecture Search with evolutionary algorithms
**What it does**:
- Evolutionary and random search strategies
- Designs custom neural network architectures
- Balances accuracy and efficiency

**Integration in SPECTRA-Lab**:
```
Destination: services/analysis/app/ml/automl/nas/architecture_search.py
API Endpoint: POST /api/v1/ml/automl/neural-architecture-search
Frontend Use: AutoML page - NAS section
```

**Frontend Component Needs**:
- Architecture visualization (network diagram)
- Generation progress tracker
- Fitness evolution chart
- Best architecture comparison table

---

#### File 5: `data_handler.py` (275 lines)
**Purpose**: Semiconductor-specific data processing
**What it does**:
- Synthetic data generators for testing
- Outlier detection and feature engineering
- Process parameter normalization

**Integration in SPECTRA-Lab**:
```
Destination: services/analysis/app/ml/data/data_handler.py
API Endpoint: GET /api/v1/ml/data/semiconductor/{data_type}
Frontend Use: All ML pages - data loading utility
```

**Frontend Component Needs**:
- Data quality dashboard
- Outlier visualization
- Feature distribution plots

---

### 2️⃣ RLHF TRAINING SCRIPTS (4 Python Files) - **BACKEND PRIORITY #2**

#### File 6: `train_sft.py`
**Purpose**: Supervised Fine-Tuning
**Integration**: `services/analysis/app/ml/rlhf/train_sft.py`
**Frontend Use**: Model Training page - SFT training

#### File 7: `train_rm.py`
**Purpose**: Reward Model training
**Integration**: `services/analysis/app/ml/rlhf/train_rm.py`
**Frontend Use**: Model Training page - Reward model training

#### File 8: `train_ppo.py`
**Purpose**: Proximal Policy Optimization
**Integration**: `services/analysis/app/ml/rlhf/train_ppo.py`
**Frontend Use**: Model Training page - PPO training

#### File 9: `train_dpo.py`
**Purpose**: Direct Preference Optimization
**Integration**: `services/analysis/app/ml/rlhf/train_dpo.py`
**Frontend Use**: Model Training page - DPO training

---

### 3️⃣ EVALUATION MODULES (5 Python Files) - **EXPLAINABILITY PAGE**

#### File 10: `eval.py`
**Purpose**: Model evaluation framework
**Integration**: `services/analysis/app/ml/eval/eval.py`
**Frontend Use**: **Explainability page** - Model evaluation dashboard

#### File 11: `eval_metrics.py`
**Purpose**: Comprehensive evaluation metrics
**Integration**: `services/analysis/app/ml/eval/eval_metrics.py`
**API Endpoint**: GET /api/v1/ml/eval/metrics
**Frontend Use**: **Explainability page** - Metrics visualization

**Frontend Component Needs**:
- Metrics comparison table
- Performance heatmap
- Metric trends over time

---

#### File 12: `winrate_tournament.py`
**Purpose**: Tournament-style model comparison
**Integration**: `services/analysis/app/ml/eval/winrate_tournament.py`
**API Endpoint**: POST /api/v1/ml/eval/tournament
**Frontend Use**: **A/B Testing page** - Model tournament

**Frontend Component Needs**:
- Tournament bracket visualization
- Win rate matrix
- Head-to-head comparison charts
- Champion/challenger selection UI

---

#### File 13: `safety_checks.py`
**Purpose**: Safety validation for models
**Integration**: `services/analysis/app/ml/eval/safety_checks.py`
**Frontend Use**: **Explainability page** - Safety audit section

#### File 14: `adversarial_prompts.txt` (3 lines)
**Purpose**: Test prompts for adversarial testing
**Integration**: `services/analysis/app/ml/eval/adversarial_prompts.txt`
**Frontend Use**: **Explainability page** - Adversarial testing

---

### 4️⃣ UTILITY MODULES (9 Python Files) - **BACKEND SUPPORT**

#### File 15: `data_utils.py`
**Purpose**: General data processing utilities
**Integration**: `services/analysis/app/utils/data_utils.py`

#### File 16: `reward_utils.py`
**Purpose**: Reward calculation utilities
**Integration**: `services/analysis/app/ml/rlhf/reward_utils.py`

#### File 17: `logging_utils.py`
**Purpose**: Logging and monitoring
**Integration**: `services/shared/utils/logging_utils.py`

#### File 18: `safety_policies.py`
**Purpose**: Safety policy enforcement
**Integration**: `services/analysis/app/ml/safety/safety_policies.py`

#### File 19: `prepare_data.py`
**Purpose**: Data preparation scripts
**Integration**: `services/analysis/scripts/prepare_data.py`

#### File 20: `generate_judgments_aif.py`
**Purpose**: Generate AI feedback judgments
**Integration**: `services/analysis/scripts/generate_judgments_aif.py`

#### File 21: `automl_examples.py` (217 lines)
**Purpose**: Usage examples and demos
**Integration**: `services/analysis/examples/automl_examples.py`
**Frontend Use**: Use these examples to build frontend mock data and UI flows

#### Files 22-23: `__init__.py` files (5 total)
**Purpose**: Python package initialization
**Integration**: Place in respective module directories

---

### 5️⃣ CONFIGURATION FILES (6 YAML) - **CONFIG**

#### File 24: `automl_full.yaml`
**Purpose**: Complete AutoML pipeline config
**Integration**: `services/analysis/config/automl_full.yaml`
**Frontend Use**: AutoML page - "Full Pipeline" preset

#### File 25: `automl_quickstart.yaml`
**Purpose**: Fast prototyping config (5-10 min runtime)
**Integration**: `services/analysis/config/automl_quickstart.yaml`
**Frontend Use**: AutoML page - "Quick Start" preset

#### File 26: `automl_nas.yaml`
**Purpose**: NAS-focused config
**Integration**: `services/analysis/config/automl_nas.yaml`
**Frontend Use**: AutoML page - "NAS Only" preset

#### File 27: `sft.yaml`
**Purpose**: Supervised fine-tuning config
**Integration**: `services/analysis/config/sft.yaml`

#### File 28: `rlhf.yaml`
**Purpose**: RLHF training config
**Integration**: `services/analysis/config/rlhf.yaml`

#### File 29: `eval.yaml`
**Purpose**: Evaluation config
**Integration**: `services/analysis/config/eval.yaml`

---

### 6️⃣ DOCUMENTATION FILES (9 Markdown) - **REFERENCE**

#### File 30: `AUTOML_README.md` (424 lines)
**Purpose**: Comprehensive AutoML documentation
**Integration**: `docs/ml/automl/README.md`
**Frontend Use**: Reference for building AutoML page UI

#### File 31: `AUTOML_SETUP.md` (215 lines)
**Purpose**: Quick setup guide
**Integration**: `docs/ml/automl/SETUP.md`

#### File 32: `AUTOML_COMPLETE.md` (394 lines)
**Purpose**: Project overview and features
**Integration**: `docs/ml/automl/COMPLETE.md`

#### File 33: `QUICK_REFERENCE.md` (205 lines)
**Purpose**: Command cheat sheet
**Integration**: `docs/ml/automl/QUICK_REFERENCE.md`

#### File 34: `FILE_INVENTORY.md` (233 lines)
**Purpose**: Complete file inventory
**Integration**: `docs/ml/automl/FILE_INVENTORY.md`

#### File 35: `README.md` (169 lines)
**Purpose**: Main project README
**Integration**: Reference only (we have our own README)

#### File 36: `human_labeling_guidelines.md` (21 lines)
**Purpose**: Guidelines for human labelers
**Integration**: `docs/ml/rlhf/human_labeling_guidelines.md`

#### File 37: `aif_judges_guidelines.md` (15 lines)
**Purpose**: AI feedback judge guidelines
**Integration**: `docs/ml/rlhf/aif_judges_guidelines.md`

#### File 38: `reward_rubric.md` (8 lines)
**Purpose**: Reward scoring rubric
**Integration**: `docs/ml/rlhf/reward_rubric.md`

---

### 7️⃣ DATA & PROMPTS (7 JSON/JSONL/TXT) - **EXAMPLES**

#### File 39: `sample_prompts.jsonl`
**Purpose**: Example prompts for training
**Integration**: `services/analysis/data/examples/sample_prompts.jsonl`

#### File 40: `sft_example.jsonl`
**Purpose**: SFT training data example
**Integration**: `services/analysis/data/examples/sft_example.jsonl`

#### File 41: `prefs_example.jsonl`
**Purpose**: Preference pair examples
**Integration**: `services/analysis/data/examples/prefs_example.jsonl`

#### File 42: `preference_pair.schema.json`
**Purpose**: JSON schema for preference pairs
**Integration**: `services/analysis/data/schemas/preference_pair.schema.json`

#### File 43: `system_electronics.txt` (3 lines)
**Purpose**: System prompts for electronics domain
**Integration**: `services/analysis/prompts/system_electronics.txt`

#### File 44: `system_photonics.txt` (3 lines)
**Purpose**: System prompts for photonics domain
**Integration**: `services/analysis/prompts/system_photonics.txt`

#### File 45: `system_biomed.txt` (2 lines)
**Purpose**: System prompts for biomedical domain
**Integration**: `services/analysis/prompts/system_biomed.txt`

---

### 8️⃣ BUILD & DEPLOYMENT (3 Files) - **INFRASTRUCTURE**

#### File 46: `requirements.txt` (22 lines)
**Purpose**: Python dependencies
**Integration**: Merge into `requirements.txt` at repository root

**Key Dependencies to Add**:
```
optuna>=3.3.0
scipy>=1.11.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.16.0
```

#### File 47: `Makefile`
**Purpose**: Build automation
**Integration**: Add to repository root or `services/analysis/Makefile`

#### File 48: `ds_zero3.json`
**Purpose**: DeepSpeed ZeRO-3 configuration
**Integration**: `services/analysis/config/deepspeed/ds_zero3.json`

#### File 49: `fsdp_config.json`
**Purpose**: Fully Sharded Data Parallel config
**Integration**: `services/analysis/config/fsdp/fsdp_config.json`

---

### 9️⃣ MISCELLANEOUS (4 Files)

#### File 50: `DIRECTORY_STRUCTURE.txt` (94 lines)
**Purpose**: Project directory structure
**Integration**: Reference only

#### File 51: `__init__.py` (root level)
**Purpose**: Python package init
**Integration**: Various module directories

#### Files 52-53: Additional `__init__.py` files
**Purpose**: Module initialization
**Integration**: Respective module directories

---

## 🎯 Integration Roadmap

### Phase 1: Backend Integration (Week 1-2)

#### Step 1.1: Set up directory structure
```bash
cd services/analysis
mkdir -p app/ml/automl/{model_selection,hyperopt,nas}
mkdir -p app/ml/rlhf
mkdir -p app/ml/eval
mkdir -p app/ml/data
mkdir -p app/ml/safety
mkdir -p config/automl
mkdir -p examples
mkdir -p scripts
```

#### Step 1.2: Copy core AutoML files
```bash
# From Downloads to SPECTRA-Lab
cp "SPECTRA Auto ML Updated/train_automl.py" services/analysis/app/ml/automl/
cp "SPECTRA Auto ML Updated/auto_selector.py" services/analysis/app/ml/automl/model_selection/
cp "SPECTRA Auto ML Updated/tuner.py" services/analysis/app/ml/automl/hyperopt/
cp "SPECTRA Auto ML Updated/architecture_search.py" services/analysis/app/ml/automl/nas/
cp "SPECTRA Auto ML Updated/data_handler.py" services/analysis/app/ml/data/
```

#### Step 1.3: Copy RLHF training files
```bash
cp "SPECTRA Auto ML Updated"/train_*.py services/analysis/app/ml/rlhf/
```

#### Step 1.4: Copy evaluation files
```bash
cp "SPECTRA Auto ML Updated"/{eval.py,eval_metrics.py,winrate_tournament.py,safety_checks.py} services/analysis/app/ml/eval/
```

#### Step 1.5: Copy utilities
```bash
cp "SPECTRA Auto ML Updated"/{data_utils.py,reward_utils.py,logging_utils.py,safety_policies.py} services/analysis/app/utils/
```

#### Step 1.6: Copy configurations
```bash
cp "SPECTRA Auto ML Updated"/automl_*.yaml services/analysis/config/automl/
cp "SPECTRA Auto ML Updated"/{sft.yaml,rlhf.yaml,eval.yaml} services/analysis/config/
```

#### Step 1.7: Update requirements.txt
```bash
cat "SPECTRA Auto ML Updated/requirements.txt" >> requirements.txt
# Then remove duplicates and sort
```

---

### Phase 2: Create FastAPI Endpoints (Week 2-3)

Create new FastAPI routers in `services/analysis/app/api/v1/ml/`:

#### `automl.py` - AutoML endpoints
```python
from fastapi import APIRouter, BackgroundTasks
from app.ml.automl.train_automl import AutoMLPipeline

router = APIRouter(prefix="/automl", tags=["AutoML"])

@router.post("/run-pipeline")
async def run_automl_pipeline(config: AutoMLConfig, background_tasks: BackgroundTasks):
    """Run complete AutoML pipeline"""
    pipeline = AutoMLPipeline(config.dict())
    background_tasks.add_task(pipeline.run)
    return {"status": "started", "job_id": pipeline.job_id}

@router.post("/model-selection")
async def run_model_selection(data: ModelSelectionRequest):
    """Run automatic model selection"""
    from app.ml.automl.model_selection.auto_selector import AutoModelSelector
    selector = AutoModelSelector(**data.dict())
    results = selector.fit(data.X_train, data.y_train, data.X_test, data.y_test)
    return results

@router.post("/hyperparameter-tuning")
async def run_hyperparameter_tuning(request: HyperparameterRequest):
    """Run hyperparameter optimization"""
    from app.ml.automl.hyperopt.tuner import AutoHyperparameterTuner
    tuner = AutoHyperparameterTuner(**request.dict())
    results = tuner.optimize(request.X_train, request.y_train, request.X_test, request.y_test)
    return results

@router.post("/neural-architecture-search")
async def run_nas(request: NASRequest):
    """Run Neural Architecture Search"""
    from app.ml.automl.nas.architecture_search import NeuralArchitectureSearch
    nas = NeuralArchitectureSearch(**request.dict())
    results = nas.search(request.X_train, request.y_train, request.X_val, request.y_val)
    return results

@router.get("/job/{job_id}/status")
async def get_job_status(job_id: str):
    """Get AutoML job status"""
    # Implementation for job status tracking
    pass

@router.get("/job/{job_id}/results")
async def get_job_results(job_id: str):
    """Get AutoML job results"""
    # Implementation for job results retrieval
    pass
```

#### `explainability.py` - Model explainability endpoints
```python
from fastapi import APIRouter
router = APIRouter(prefix="/explainability", tags=["Explainability"])

@router.post("/evaluate-model")
async def evaluate_model(request: EvaluationRequest):
    """Evaluate model with comprehensive metrics"""
    from app.ml.eval.eval import evaluate_model
    results = evaluate_model(request.model_id, request.test_data)
    return results

@router.get("/metrics/{model_id}")
async def get_model_metrics(model_id: str):
    """Get evaluation metrics for a model"""
    from app.ml.eval.eval_metrics import get_metrics
    return get_metrics(model_id)

@router.post("/shap-analysis")
async def run_shap_analysis(request: SHAPRequest):
    """Run SHAP analysis on model"""
    # Will implement SHAP integration
    pass

@router.post("/lime-explanation")
async def run_lime_explanation(request: LIMERequest):
    """Generate LIME explanations"""
    # Will implement LIME integration
    pass
```

#### `ab_testing.py` - A/B testing endpoints
```python
from fastapi import APIRouter
router = APIRouter(prefix="/ab-testing", tags=["A/B Testing"])

@router.post("/tournament")
async def run_tournament(request: TournamentRequest):
    """Run model tournament"""
    from app.ml.eval.winrate_tournament import run_tournament
    results = run_tournament(request.models, request.test_data)
    return results

@router.post("/head-to-head")
async def compare_models(model_a: str, model_b: str, test_data: dict):
    """Head-to-head model comparison"""
    # Implementation
    pass

@router.post("/experiment/create")
async def create_experiment(experiment: ExperimentConfig):
    """Create new A/B testing experiment"""
    # Implementation
    pass

@router.get("/experiment/{exp_id}/results")
async def get_experiment_results(exp_id: str):
    """Get A/B test results"""
    # Implementation
    pass
```

---

### Phase 3: Frontend Development (Week 3-4)

#### Create AutoML Page Components

**File**: `apps/web/src/app/dashboard/ml/automl/page.tsx`

Replace the "Coming Soon" placeholder with a full AutoML UI:

```typescript
'use client'

import { useState } from 'react'
import { Play, Settings, Brain, Zap, Target } from 'lucide-react'

// Sub-components to create:
import { PipelineConfigForm } from '@/components/ml/automl/PipelineConfigForm'
import { ModelSelectionResults } from '@/components/ml/automl/ModelSelectionResults'
import { HyperparameterTuning } from '@/components/ml/automl/HyperparameterTuning'
import { NASVisualization } from '@/components/ml/automl/NASVisualization'
import { JobProgressTracker } from '@/components/ml/automl/JobProgressTracker'

export default function AutoMLPage() {
  const [activeJob, setActiveJob] = useState(null)
  const [results, setResults] = useState(null)

  const handleRunPipeline = async (config) => {
    const response = await fetch('/api/v1/ml/automl/run-pipeline', {
      method: 'POST',
      body: JSON.stringify(config),
      headers: { 'Content-Type': 'application/json' }
    })
    const data = await response.json()
    setActiveJob(data.job_id)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold">Automated Machine Learning</h2>
        <p className="text-gray-600">Automatic model selection, hyperparameter tuning, and neural architecture search</p>
      </div>

      {/* Quick Start Presets */}
      <div className="grid grid-cols-3 gap-4">
        <PresetCard
          icon={<Zap />}
          title="Quick Start"
          description="5-10 minute rapid prototyping"
          onClick={() => handleRunPipeline({ preset: 'quickstart' })}
        />
        <PresetCard
          icon={<Target />}
          title="Full Pipeline"
          description="Complete AutoML with NAS"
          onClick={() => handleRunPipeline({ preset: 'full' })}
        />
        <PresetCard
          icon={<Brain />}
          title="NAS Only"
          description="Neural architecture search"
          onClick={() => handleRunPipeline({ preset: 'nas' })}
        />
      </div>

      {/* Advanced Configuration */}
      <PipelineConfigForm onSubmit={handleRunPipeline} />

      {/* Job Progress */}
      {activeJob && <JobProgressTracker jobId={activeJob} />}

      {/* Results Dashboard */}
      {results && (
        <div className="space-y-6">
          <ModelSelectionResults data={results.model_selection} />
          <HyperparameterTuning data={results.hyperparameter_tuning} />
          {results.nas && <NASVisualization data={results.nas} />}
        </div>
      )}
    </div>
  )
}
```

**Components to Create**:

1. `PipelineConfigForm.tsx` - Configuration form for AutoML pipeline
2. `ModelSelectionResults.tsx` - Algorithm comparison table and charts
3. `HyperparameterTuning.tsx` - Optuna trial visualization
4. `NASVisualization.tsx` - Architecture search visualization
5. `JobProgressTracker.tsx` - Real-time progress tracking

---

#### Create Explainability Page Components

**File**: `apps/web/src/app/dashboard/ml/explainability/page.tsx`

Replace placeholder with model interpretability UI:

```typescript
'use client'

import { useState } from 'react'
import { SHAPForceplot } from '@/components/ml/explainability/SHAPForceplot'
import { LIMEExplanation } from '@/components/ml/explainability/LIMEExplanation'
import { FeatureImportance } from '@/components/ml/explainability/FeatureImportance'
import { MetricsComparison } from '@/components/ml/explainability/MetricsComparison'

export default function ExplainabilityPage() {
  const [selectedModel, setSelectedModel] = useState(null)
  const [explanationType, setExplanationType] = useState('shap')

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold">Model Explainability & Interpretability</h2>
        <p className="text-gray-600">Understand and interpret ML model predictions</p>
      </div>

      {/* Model Selection */}
      <ModelSelector onSelect={setSelectedModel} />

      {/* Explanation Tabs */}
      <Tabs value={explanationType} onValueChange={setExplanationType}>
        <TabsList>
          <TabsTrigger value="shap">SHAP Analysis</TabsTrigger>
          <TabsTrigger value="lime">LIME Explanations</TabsTrigger>
          <TabsTrigger value="features">Feature Importance</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
        </TabsList>

        <TabsContent value="shap">
          <SHAPForceplot modelId={selectedModel} />
        </TabsContent>

        <TabsContent value="lime">
          <LIMEExplanation modelId={selectedModel} />
        </TabsContent>

        <TabsContent value="features">
          <FeatureImportance modelId={selectedModel} />
        </TabsContent>

        <TabsContent value="metrics">
          <MetricsComparison modelId={selectedModel} />
        </TabsContent>
      </Tabs>
    </div>
  )
}
```

**Components to Create**:

1. `SHAPForceplot.tsx` - SHAP force plot visualization
2. `LIMEExplanation.tsx` - LIME explanation display
3. `FeatureImportance.tsx` - Feature importance charts
4. `MetricsComparison.tsx` - Comprehensive metrics dashboard
5. `ModelSelector.tsx` - Model selection dropdown

---

#### Create A/B Testing Page Components

**File**: `apps/web/src/app/dashboard/ml/ab-testing/page.tsx`

Replace placeholder with tournament and comparison UI:

```typescript
'use client'

import { useState } from 'react'
import { TournamentBracket } from '@/components/ml/abtesting/TournamentBracket'
import { WinRateMatrix } from '@/components/ml/abtesting/WinRateMatrix'
import { HeadToHeadComparison } from '@/components/ml/abtesting/HeadToHeadComparison'

export default function ABTestingPage() {
  const [tournamentResults, setTournamentResults] = useState(null)

  const handleRunTournament = async (models) => {
    const response = await fetch('/api/v1/ml/ab-testing/tournament', {
      method: 'POST',
      body: JSON.stringify({ models }),
      headers: { 'Content-Type': 'application/json' }
    })
    const data = await response.json()
    setTournamentResults(data)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold">A/B Testing & Model Comparison</h2>
        <p className="text-gray-600">Tournament-style model evaluation and head-to-head testing</p>
      </div>

      {/* Tournament Setup */}
      <TournamentSetup onStart={handleRunTournament} />

      {/* Tournament Results */}
      {tournamentResults && (
        <div className="space-y-6">
          <TournamentBracket data={tournamentResults.bracket} />
          <WinRateMatrix data={tournamentResults.winrates} />
          <ChampionCard model={tournamentResults.champion} />
        </div>
      )}

      {/* Head-to-Head Comparison */}
      <HeadToHeadComparison />
    </div>
  )
}
```

**Components to Create**:

1. `TournamentBracket.tsx` - Visual tournament bracket
2. `WinRateMatrix.tsx` - Win rate heatmap
3. `HeadToHeadComparison.tsx` - Direct model comparison
4. `TournamentSetup.tsx` - Tournament configuration
5. `ChampionCard.tsx` - Winner display card

---

## 📝 Summary & Next Steps

### What We Have (53 Files)

✅ **Complete AutoML backend** (Python)
✅ **RLHF training pipeline** (Python)
✅ **Model evaluation framework** (Python)
✅ **Tournament comparison** (Python)
✅ **Configuration files** (YAML)
✅ **Documentation** (Markdown)
✅ **Data examples** (JSON/JSONL)

### What We Need to Build

🔨 **Frontend React Components**:
- AutoML page UI (6 components)
- Explainability page UI (5 components)
- A/B Testing page UI (5 components)

🔨 **FastAPI Endpoints**:
- `/api/v1/ml/automl/*` (6 endpoints)
- `/api/v1/ml/explainability/*` (4 endpoints)
- `/api/v1/ml/ab-testing/*` (4 endpoints)

🔨 **Integration Work**:
- Copy 53 files to proper locations in SPECTRA-Lab
- Create FastAPI routers
- Build React UI components
- Wire up API connections
- Add to navigation

### Timeline Estimate

| Phase | Duration | Tasks |
|-------|----------|-------|
| **Phase 1**: Backend Integration | 1-2 weeks | Copy files, organize structure, update imports |
| **Phase 2**: FastAPI Endpoints | 1 week | Create routers, define schemas, test APIs |
| **Phase 3**: Frontend Components | 2 weeks | Build React UIs for AutoML, Explainability, A/B Testing |
| **Phase 4**: Integration & Testing | 1 week | Connect frontend to backend, end-to-end testing |
| **TOTAL** | **5-6 weeks** | Complete ML pages implementation |

---

## 🎯 Immediate Next Steps

### 1. Confirm Integration Approach
Do you want to:
- **Option A**: Integrate all 53 files now (full AutoML backend)
- **Option B**: Start with core AutoML files only (5 files)
- **Option C**: Focus on one page at a time (AutoML first)

### 2. Choose Starting Point
- **Backend First**: Set up Python/FastAPI infrastructure
- **Frontend First**: Build React UI with mock data
- **Full Stack**: Both simultaneously

### 3. Specify Priorities
Which page should we implement first?
1. AutoML page
2. Explainability page
3. A/B Testing page

---

**This integration plan covers ALL 53 files with specific locations, API endpoints, and frontend components needed for each.**

Ready to proceed! Let me know which approach you prefer, and I'll start the integration. 🚀
