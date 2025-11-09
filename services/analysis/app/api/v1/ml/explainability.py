"""
Model Explainability API Router
Provides endpoints for SHAP, LIME, and other interpretability tools
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import uuid
from datetime import datetime
from pathlib import Path
import json
import joblib
import numpy as np

# Import evaluation modules
from app.ml.eval.eval import ModelEvaluator
from app.ml.eval.eval_metrics import calculate_metrics
from app.ml.eval.safety_checks import SafetyChecker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explainability", tags=["Explainability"])

# In-memory job storage (use Redis/database in production)
explainability_jobs = {}

# Pydantic models
class ModelEvaluationConfig(BaseModel):
    """Model evaluation configuration"""
    model_path: str = Field(..., description="Path to trained model file (.pkl)")
    data_path: str = Field(..., description="Path to test data CSV")
    target_column: str = Field(..., description="Target variable column name")
    model_type: str = Field("regression", description="Model type: regression or classification")
    output_dir: str = Field("eval_results", description="Output directory")

class SHAPConfig(BaseModel):
    """SHAP analysis configuration"""
    model_path: str = Field(..., description="Path to trained model file")
    data_path: str = Field(..., description="Path to data for SHAP analysis")
    target_column: str = Field(..., description="Target variable column name")
    background_samples: int = Field(100, description="Number of background samples for SHAP")
    max_display: int = Field(20, description="Maximum features to display")
    plot_type: str = Field("summary", description="Plot type: summary, waterfall, force, dependence")

class LIMEConfig(BaseModel):
    """LIME explanation configuration"""
    model_path: str = Field(..., description="Path to trained model file")
    data_path: str = Field(..., description="Path to data for LIME explanations")
    target_column: str = Field(..., description="Target variable column name")
    instance_idx: int = Field(0, description="Index of instance to explain")
    num_features: int = Field(10, description="Number of features in explanation")
    num_samples: int = Field(5000, description="Number of samples for LIME")

class FeatureImportanceConfig(BaseModel):
    """Feature importance calculation configuration"""
    model_path: str = Field(..., description="Path to trained model file")
    data_path: str = Field(..., description="Path to data")
    target_column: str = Field(..., description="Target variable column name")
    method: str = Field("builtin", description="Method: builtin, permutation, shap")

class JobResponse(BaseModel):
    """Job submission response"""
    job_id: str
    status: str
    created_at: str
    message: str

class EvaluationResults(BaseModel):
    """Model evaluation results"""
    job_id: str
    status: str
    metrics: Optional[Dict[str, float]] = None
    predictions: Optional[List[float]] = None
    residuals: Optional[List[float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    plots: Optional[Dict[str, str]] = None


# Background task functions

def run_model_evaluation(job_id: str, config: ModelEvaluationConfig):
    """Background task to evaluate a model"""
    try:
        explainability_jobs[job_id]["status"] = "running"
        explainability_jobs[job_id]["started_at"] = datetime.now().isoformat()

        logger.info(f"Starting model evaluation job {job_id}")

        # Load model
        model = joblib.load(config.model_path)

        # Initialize evaluator
        evaluator = ModelEvaluator(
            model=model,
            data_path=config.data_path,
            target_column=config.target_column,
            model_type=config.model_type,
            output_dir=config.output_dir
        )

        # Run evaluation
        results = evaluator.evaluate()

        explainability_jobs[job_id]["status"] = "completed"
        explainability_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        explainability_jobs[job_id]["results"] = {
            "metrics": results.get("metrics", {}),
            "predictions": results.get("predictions", [])[:100],  # Limit size
            "residuals": results.get("residuals", [])[:100],
            "feature_importance": results.get("feature_importance", {}),
            "report_path": str(results.get("report_path", ""))
        }

        logger.info(f"Model evaluation job {job_id} completed")

    except Exception as e:
        logger.error(f"Model evaluation job {job_id} failed: {str(e)}")
        explainability_jobs[job_id]["status"] = "failed"
        explainability_jobs[job_id]["error"] = str(e)
        explainability_jobs[job_id]["completed_at"] = datetime.now().isoformat()


def run_shap_analysis(job_id: str, config: SHAPConfig):
    """Background task to run SHAP analysis"""
    try:
        explainability_jobs[job_id]["status"] = "running"
        explainability_jobs[job_id]["started_at"] = datetime.now().isoformat()

        logger.info(f"Starting SHAP analysis job {job_id}")

        # Import SHAP (lazy import to avoid startup overhead)
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP is not installed. Install with: pip install shap")

        # Load model and data
        model = joblib.load(config.model_path)

        import pandas as pd
        data = pd.read_csv(config.data_path)
        X = data.drop(columns=[config.target_column])

        # Create SHAP explainer
        if config.background_samples > 0 and len(X) > config.background_samples:
            background = shap.sample(X, config.background_samples)
            explainer = shap.Explainer(model.predict, background)
        else:
            explainer = shap.Explainer(model.predict, X)

        # Calculate SHAP values
        shap_values = explainer(X[:min(len(X), 1000)])  # Limit to 1000 samples

        # Generate plots
        output_dir = Path(config.model_path).parent / "shap_plots"
        output_dir.mkdir(exist_ok=True)

        plots = {}

        # Summary plot
        if config.plot_type in ["summary", "all"]:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X, show=False, max_display=config.max_display)
            summary_path = output_dir / f"{job_id}_summary.png"
            plt.savefig(summary_path, bbox_inches='tight', dpi=150)
            plt.close()
            plots["summary"] = str(summary_path)

        # Feature importance
        feature_importance = {}
        if hasattr(shap_values, 'values'):
            importance = np.abs(shap_values.values).mean(axis=0)
            for feat, imp in zip(X.columns, importance):
                feature_importance[feat] = float(imp)

        explainability_jobs[job_id]["status"] = "completed"
        explainability_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        explainability_jobs[job_id]["results"] = {
            "feature_importance": feature_importance,
            "plots": plots,
            "shap_values_shape": shap_values.values.shape if hasattr(shap_values, 'values') else None
        }

        logger.info(f"SHAP analysis job {job_id} completed")

    except Exception as e:
        logger.error(f"SHAP analysis job {job_id} failed: {str(e)}")
        explainability_jobs[job_id]["status"] = "failed"
        explainability_jobs[job_id]["error"] = str(e)
        explainability_jobs[job_id]["completed_at"] = datetime.now().isoformat()


def run_lime_explanation(job_id: str, config: LIMEConfig):
    """Background task to generate LIME explanations"""
    try:
        explainability_jobs[job_id]["status"] = "running"
        explainability_jobs[job_id]["started_at"] = datetime.now().isoformat()

        logger.info(f"Starting LIME explanation job {job_id}")

        # Import LIME (lazy import)
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            raise ImportError("LIME is not installed. Install with: pip install lime")

        # Load model and data
        model = joblib.load(config.model_path)

        import pandas as pd
        data = pd.read_csv(config.data_path)
        X = data.drop(columns=[config.target_column])
        y = data[config.target_column]

        # Create LIME explainer
        explainer = LimeTabularExplainer(
            X.values,
            feature_names=X.columns.tolist(),
            mode='regression',
            training_labels=y.values
        )

        # Get instance to explain
        instance = X.iloc[config.instance_idx].values

        # Generate explanation
        explanation = explainer.explain_instance(
            instance,
            model.predict,
            num_features=config.num_features,
            num_samples=config.num_samples
        )

        # Save plot
        output_dir = Path(config.model_path).parent / "lime_plots"
        output_dir.mkdir(exist_ok=True)

        import matplotlib.pyplot as plt
        fig = explanation.as_pyplot_figure()
        plot_path = output_dir / f"{job_id}_lime.png"
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()

        # Extract feature contributions
        feature_contributions = {}
        for feat, weight in explanation.as_list():
            feature_contributions[feat] = float(weight)

        explainability_jobs[job_id]["status"] = "completed"
        explainability_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        explainability_jobs[job_id]["results"] = {
            "instance_idx": config.instance_idx,
            "prediction": float(model.predict([instance])[0]),
            "actual": float(y.iloc[config.instance_idx]) if config.instance_idx < len(y) else None,
            "feature_contributions": feature_contributions,
            "plot_path": str(plot_path)
        }

        logger.info(f"LIME explanation job {job_id} completed")

    except Exception as e:
        logger.error(f"LIME explanation job {job_id} failed: {str(e)}")
        explainability_jobs[job_id]["status"] = "failed"
        explainability_jobs[job_id]["error"] = str(e)
        explainability_jobs[job_id]["completed_at"] = datetime.now().isoformat()


def run_feature_importance(job_id: str, config: FeatureImportanceConfig):
    """Background task to calculate feature importance"""
    try:
        explainability_jobs[job_id]["status"] = "running"
        explainability_jobs[job_id]["started_at"] = datetime.now().isoformat()

        logger.info(f"Starting feature importance job {job_id}")

        # Load model and data
        model = joblib.load(config.model_path)

        import pandas as pd
        data = pd.read_csv(config.data_path)
        X = data.drop(columns=[config.target_column])
        y = data[config.target_column]

        feature_importance = {}

        if config.method == "builtin":
            # Use model's built-in feature importance if available
            if hasattr(model, 'feature_importances_'):
                for feat, imp in zip(X.columns, model.feature_importances_):
                    feature_importance[feat] = float(imp)
            else:
                raise ValueError("Model does not have built-in feature importance")

        elif config.method == "permutation":
            # Permutation importance
            from sklearn.inspection import permutation_importance
            result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            for feat, imp in zip(X.columns, result.importances_mean):
                feature_importance[feat] = float(imp)

        elif config.method == "shap":
            # SHAP-based importance
            import shap
            explainer = shap.Explainer(model.predict, X)
            shap_values = explainer(X[:min(len(X), 1000)])
            importance = np.abs(shap_values.values).mean(axis=0)
            for feat, imp in zip(X.columns, importance):
                feature_importance[feat] = float(imp)

        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True))

        explainability_jobs[job_id]["status"] = "completed"
        explainability_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        explainability_jobs[job_id]["results"] = {
            "method": config.method,
            "feature_importance": feature_importance
        }

        logger.info(f"Feature importance job {job_id} completed")

    except Exception as e:
        logger.error(f"Feature importance job {job_id} failed: {str(e)}")
        explainability_jobs[job_id]["status"] = "failed"
        explainability_jobs[job_id]["error"] = str(e)
        explainability_jobs[job_id]["completed_at"] = datetime.now().isoformat()


# API Endpoints

@router.post("/evaluate-model", response_model=JobResponse)
async def evaluate_model(config: ModelEvaluationConfig, background_tasks: BackgroundTasks):
    """
    Evaluate a trained model

    Calculates comprehensive metrics, predictions, residuals, and generates evaluation reports.
    """
    job_id = str(uuid.uuid4())

    explainability_jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "config": config.dict(),
        "type": "model_evaluation"
    }

    background_tasks.add_task(run_model_evaluation, job_id, config)

    return JobResponse(
        job_id=job_id,
        status="pending",
        created_at=explainability_jobs[job_id]["created_at"],
        message="Model evaluation job created successfully"
    )


@router.post("/shap-analysis", response_model=JobResponse)
async def shap_analysis(config: SHAPConfig, background_tasks: BackgroundTasks):
    """
    Run SHAP analysis for model interpretation

    Generates SHAP values and visualizations to explain model predictions.
    """
    job_id = str(uuid.uuid4())

    explainability_jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "config": config.dict(),
        "type": "shap_analysis"
    }

    background_tasks.add_task(run_shap_analysis, job_id, config)

    return JobResponse(
        job_id=job_id,
        status="pending",
        created_at=explainability_jobs[job_id]["created_at"],
        message="SHAP analysis job created successfully"
    )


@router.post("/lime-explanation", response_model=JobResponse)
async def lime_explanation(config: LIMEConfig, background_tasks: BackgroundTasks):
    """
    Generate LIME explanations for individual predictions

    Provides local interpretable explanations for specific instances.
    """
    job_id = str(uuid.uuid4())

    explainability_jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "config": config.dict(),
        "type": "lime_explanation"
    }

    background_tasks.add_task(run_lime_explanation, job_id, config)

    return JobResponse(
        job_id=job_id,
        status="pending",
        created_at=explainability_jobs[job_id]["created_at"],
        message="LIME explanation job created successfully"
    )


@router.post("/feature-importance", response_model=JobResponse)
async def calculate_feature_importance(config: FeatureImportanceConfig, background_tasks: BackgroundTasks):
    """
    Calculate feature importance using various methods

    Supports built-in importance, permutation importance, and SHAP-based importance.
    """
    job_id = str(uuid.uuid4())

    explainability_jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "config": config.dict(),
        "type": "feature_importance"
    }

    background_tasks.add_task(run_feature_importance, job_id, config)

    return JobResponse(
        job_id=job_id,
        status="pending",
        created_at=explainability_jobs[job_id]["created_at"],
        message="Feature importance job created successfully"
    )


@router.get("/job/{job_id}/status")
async def get_job_status(job_id: str):
    """Get the status of an explainability job"""
    if job_id not in explainability_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = explainability_jobs[job_id]

    return {
        "job_id": job_id,
        "type": job["type"],
        "status": job["status"],
        "created_at": job["created_at"],
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "error": job.get("error")
    }


@router.get("/job/{job_id}/results")
async def get_job_results(job_id: str):
    """Get the results of a completed explainability job"""
    if job_id not in explainability_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = explainability_jobs[job_id]

    if job["status"] not in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Job is still {job['status']}. Results not available yet."
        )

    return {
        "job_id": job_id,
        "type": job["type"],
        "status": job["status"],
        "results": job.get("results"),
        "error": job.get("error")
    }


@router.get("/metrics/{model_id}")
async def get_model_metrics(model_id: str):
    """
    Get comprehensive metrics for a model

    Placeholder for retrieving cached metrics from database.
    """
    # This would query a database in production
    return {
        "model_id": model_id,
        "message": "Metrics retrieval from database not yet implemented",
        "suggestion": "Use POST /explainability/evaluate-model to evaluate a model"
    }


@router.get("/safety-audit/{model_id}")
async def get_safety_audit(model_id: str):
    """
    Get safety audit results for a model

    Placeholder for safety checks and adversarial testing results.
    """
    # This would use the SafetyChecker class
    return {
        "model_id": model_id,
        "message": "Safety audit not yet implemented",
        "planned_checks": [
            "Adversarial robustness",
            "Fairness metrics",
            "Output safety validation",
            "Distribution shift detection"
        ]
    }


@router.get("/methods")
async def get_explainability_methods():
    """Get list of available explainability methods"""
    return {
        "methods": {
            "shap": {
                "name": "SHAP (SHapley Additive exPlanations)",
                "description": "Game-theory based feature importance",
                "use_cases": ["Global interpretability", "Feature importance", "Interaction effects"],
                "pros": ["Theoretically sound", "Consistent", "Local and global explanations"],
                "cons": ["Computationally expensive", "Requires background data"]
            },
            "lime": {
                "name": "LIME (Local Interpretable Model-agnostic Explanations)",
                "description": "Local linear approximations of model behavior",
                "use_cases": ["Individual prediction explanation", "Model debugging"],
                "pros": ["Fast", "Model-agnostic", "Intuitive"],
                "cons": ["Local only", "Unstable", "Requires sampling"]
            },
            "permutation": {
                "name": "Permutation Importance",
                "description": "Feature importance by shuffling",
                "use_cases": ["Feature selection", "Global importance"],
                "pros": ["Model-agnostic", "Fast", "Intuitive"],
                "cons": ["Can be misleading with correlated features"]
            },
            "builtin": {
                "name": "Built-in Feature Importance",
                "description": "Model-specific feature importance (e.g., tree-based)",
                "use_cases": ["Quick feature ranking"],
                "pros": ["Very fast", "Native to model"],
                "cons": ["Model-specific", "May be biased"]
            }
        }
    }


@router.get("/available-models")
async def get_available_models():
    """Get list of available trained models for explainability analysis"""
    # TODO: Load from model registry or database
    # For now, return empty list as placeholder
    return {
        "models": [],
        "total": 0,
        "message": "No trained models available. Train a model first using AutoML."
    }


@router.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete an explainability job"""
    if job_id not in explainability_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if explainability_jobs[job_id]["status"] == "running":
        raise HTTPException(status_code=400, detail="Cannot delete a running job")

    del explainability_jobs[job_id]
    return {"message": f"Job {job_id} deleted successfully"}


@router.get("/jobs")
async def list_jobs(status: Optional[str] = None, job_type: Optional[str] = None, limit: int = 50):
    """List all explainability jobs with optional filtering"""
    filtered_jobs = explainability_jobs

    if status:
        filtered_jobs = {
            job_id: job for job_id, job in filtered_jobs.items()
            if job["status"] == status
        }

    if job_type:
        filtered_jobs = {
            job_id: job for job_id, job in filtered_jobs.items()
            if job["type"] == job_type
        }

    sorted_jobs = sorted(
        filtered_jobs.items(),
        key=lambda x: x[1]["created_at"],
        reverse=True
    )[:limit]

    return {
        "total": len(filtered_jobs),
        "jobs": [
            {
                "job_id": job_id,
                "type": job["type"],
                "status": job["status"],
                "created_at": job["created_at"],
                "completed_at": job.get("completed_at"),
            }
            for job_id, job in sorted_jobs
        ]
    }
