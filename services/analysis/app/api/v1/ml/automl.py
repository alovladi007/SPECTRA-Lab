"""
AutoML API Router
Provides endpoints for automated machine learning pipelines
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import uuid
from datetime import datetime
from pathlib import Path
import json

# Import AutoML modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from ml.automl.train_automl import AutoMLPipeline
from ml.automl.model_selection.auto_selector import AutoModelSelector
from ml.automl.hyperopt.tuner import HyperparameterTuner
from ml.automl.nas.architecture_search import NASSearch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/automl", tags=["AutoML"])

# In-memory job storage (use Redis/database in production)
jobs = {}

# Pydantic models for request/response
class AutoMLConfig(BaseModel):
    """AutoML pipeline configuration"""
    target_column: str = Field(..., description="Target variable column name")
    data_path: Optional[str] = Field(None, description="Path to training data CSV")
    output_dir: str = Field("automl_results", description="Output directory for results")
    preset: Optional[str] = Field("quickstart", description="Configuration preset: quickstart, full, nas")
    max_time_minutes: int = Field(30, description="Maximum training time in minutes")
    algorithms: Optional[List[str]] = Field(None, description="List of algorithms to try")
    cv_folds: int = Field(5, description="Number of cross-validation folds")
    enable_nas: bool = Field(False, description="Enable Neural Architecture Search")

class ModelSelectionConfig(BaseModel):
    """Model selection configuration"""
    data_path: str = Field(..., description="Path to training data CSV")
    target_column: str = Field(..., description="Target variable column name")
    algorithms: Optional[List[str]] = Field(None, description="Algorithms to evaluate")
    cv_folds: int = Field(5, description="Number of cross-validation folds")
    output_dir: str = Field("model_selection_results", description="Output directory")

class HyperoptConfig(BaseModel):
    """Hyperparameter tuning configuration"""
    data_path: str = Field(..., description="Path to training data CSV")
    target_column: str = Field(..., description="Target variable column name")
    algorithm: str = Field(..., description="Algorithm to optimize")
    n_trials: int = Field(50, description="Number of Optuna trials")
    cv_folds: int = Field(5, description="Number of cross-validation folds")
    output_dir: str = Field("hyperopt_results", description="Output directory")

class NASConfig(BaseModel):
    """Neural Architecture Search configuration"""
    data_path: str = Field(..., description="Path to training data CSV")
    target_column: str = Field(..., description="Target variable column name")
    search_method: str = Field("evolutionary", description="Search method: random, evolutionary")
    population_size: int = Field(20, description="Population size for evolutionary search")
    n_generations: int = Field(10, description="Number of generations")
    output_dir: str = Field("nas_results", description="Output directory")

class JobResponse(BaseModel):
    """Job submission response"""
    job_id: str
    status: str
    created_at: str
    message: str

class JobStatus(BaseModel):
    """Job status response"""
    job_id: str
    status: str  # pending, running, completed, failed
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: int = 0  # 0-100
    message: Optional[str] = None
    error: Optional[str] = None

class JobResults(BaseModel):
    """Job results response"""
    job_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    best_model: Optional[str] = None
    artifacts: Optional[Dict[str, str]] = None


# Utility functions
def run_automl_pipeline(job_id: str, config: AutoMLConfig):
    """Background task to run AutoML pipeline"""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = datetime.now().isoformat()

        logger.info(f"Starting AutoML job {job_id}")

        # Load configuration preset
        config_dict = {
            "target_column": config.target_column,
            "output_dir": config.output_dir,
            "max_time_minutes": config.max_time_minutes,
            "cv_folds": config.cv_folds,
            "enable_nas": config.enable_nas,
        }

        if config.data_path:
            config_dict["data_path"] = config.data_path
        if config.algorithms:
            config_dict["algorithms"] = config.algorithms

        # Initialize and run pipeline
        pipeline = AutoMLPipeline(config_dict)
        results = pipeline.run()

        # Store results
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["progress"] = 100
        jobs[job_id]["results"] = {
            "best_model": results.get("best_model"),
            "best_score": float(results.get("best_score", 0)),
            "metrics": results.get("metrics", {}),
            "model_path": str(results.get("model_path", "")),
            "report_path": str(results.get("report_path", ""))
        }

        logger.info(f"AutoML job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"AutoML job {job_id} failed: {str(e)}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()


def run_model_selection(job_id: str, config: ModelSelectionConfig):
    """Background task to run model selection"""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = datetime.now().isoformat()

        logger.info(f"Starting model selection job {job_id}")

        selector = AutoModelSelector(
            data_path=config.data_path,
            target_column=config.target_column,
            algorithms=config.algorithms,
            cv_folds=config.cv_folds,
            output_dir=config.output_dir
        )

        results = selector.select_best_model()

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["progress"] = 100
        jobs[job_id]["results"] = {
            "best_algorithm": results.get("best_algorithm"),
            "best_score": float(results.get("best_score", 0)),
            "all_scores": results.get("all_scores", {}),
            "rankings": results.get("rankings", [])
        }

        logger.info(f"Model selection job {job_id} completed")

    except Exception as e:
        logger.error(f"Model selection job {job_id} failed: {str(e)}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()


def run_hyperopt(job_id: str, config: HyperoptConfig):
    """Background task to run hyperparameter optimization"""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = datetime.now().isoformat()

        logger.info(f"Starting hyperparameter tuning job {job_id}")

        tuner = HyperparameterTuner(
            data_path=config.data_path,
            target_column=config.target_column,
            algorithm=config.algorithm,
            n_trials=config.n_trials,
            cv_folds=config.cv_folds,
            output_dir=config.output_dir
        )

        results = tuner.optimize()

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["progress"] = 100
        jobs[job_id]["results"] = {
            "best_params": results.get("best_params", {}),
            "best_score": float(results.get("best_score", 0)),
            "optimization_history": results.get("history", []),
            "param_importance": results.get("param_importance", {})
        }

        logger.info(f"Hyperparameter tuning job {job_id} completed")

    except Exception as e:
        logger.error(f"Hyperparameter tuning job {job_id} failed: {str(e)}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()


def run_nas(job_id: str, config: NASConfig):
    """Background task to run Neural Architecture Search"""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = datetime.now().isoformat()

        logger.info(f"Starting NAS job {job_id}")

        nas = NASSearch(
            data_path=config.data_path,
            target_column=config.target_column,
            search_method=config.search_method,
            population_size=config.population_size,
            n_generations=config.n_generations,
            output_dir=config.output_dir
        )

        results = nas.search()

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["progress"] = 100
        jobs[job_id]["results"] = {
            "best_architecture": results.get("best_architecture", {}),
            "best_score": float(results.get("best_score", 0)),
            "search_history": results.get("history", []),
            "architecture_path": str(results.get("architecture_path", ""))
        }

        logger.info(f"NAS job {job_id} completed")

    except Exception as e:
        logger.error(f"NAS job {job_id} failed: {str(e)}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()


# API Endpoints

@router.post("/run-pipeline", response_model=JobResponse)
async def run_automl(config: AutoMLConfig, background_tasks: BackgroundTasks):
    """
    Start an AutoML pipeline

    This endpoint initiates a complete AutoML pipeline including:
    - Automated model selection
    - Hyperparameter optimization
    - Optional Neural Architecture Search
    - Model evaluation and reporting
    """
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "config": config.dict(),
        "type": "automl_pipeline"
    }

    background_tasks.add_task(run_automl_pipeline, job_id, config)

    return JobResponse(
        job_id=job_id,
        status="pending",
        created_at=jobs[job_id]["created_at"],
        message="AutoML pipeline job created successfully"
    )


@router.post("/model-selection", response_model=JobResponse)
async def start_model_selection(config: ModelSelectionConfig, background_tasks: BackgroundTasks):
    """
    Run automated model selection

    Evaluates multiple ML algorithms and selects the best one based on cross-validation scores.
    """
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "config": config.dict(),
        "type": "model_selection"
    }

    background_tasks.add_task(run_model_selection, job_id, config)

    return JobResponse(
        job_id=job_id,
        status="pending",
        created_at=jobs[job_id]["created_at"],
        message="Model selection job created successfully"
    )


@router.post("/hyperparameter-tuning", response_model=JobResponse)
async def start_hyperparameter_tuning(config: HyperoptConfig, background_tasks: BackgroundTasks):
    """
    Run hyperparameter optimization using Optuna

    Performs Bayesian optimization to find the best hyperparameters for a given algorithm.
    """
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "config": config.dict(),
        "type": "hyperparameter_tuning"
    }

    background_tasks.add_task(run_hyperopt, job_id, config)

    return JobResponse(
        job_id=job_id,
        status="pending",
        created_at=jobs[job_id]["created_at"],
        message="Hyperparameter tuning job created successfully"
    )


@router.post("/neural-architecture-search", response_model=JobResponse)
async def start_nas(config: NASConfig, background_tasks: BackgroundTasks):
    """
    Run Neural Architecture Search

    Automatically designs neural network architectures optimized for your data.
    """
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "config": config.dict(),
        "type": "nas"
    }

    background_tasks.add_task(run_nas, job_id, config)

    return JobResponse(
        job_id=job_id,
        status="pending",
        created_at=jobs[job_id]["created_at"],
        message="NAS job created successfully"
    )


@router.get("/job/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of an AutoML job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    return JobStatus(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        progress=job.get("progress", 0),
        message=job.get("message"),
        error=job.get("error")
    )


@router.get("/job/{job_id}/results", response_model=JobResults)
async def get_job_results(job_id: str):
    """Get the results of a completed AutoML job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] not in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Job is still {job['status']}. Results not available yet."
        )

    return JobResults(
        job_id=job_id,
        status=job["status"],
        results=job.get("results"),
        metrics=job.get("results", {}).get("metrics") if job.get("results") else None,
        best_model=job.get("results", {}).get("best_model") if job.get("results") else None,
        artifacts={
            "model_path": job.get("results", {}).get("model_path", ""),
            "report_path": job.get("results", {}).get("report_path", "")
        } if job.get("results") else None
    )


@router.get("/presets")
async def get_presets():
    """
    Get available AutoML configuration presets

    Returns predefined configurations for quick start, full pipeline, and NAS.
    """
    presets = {
        "quickstart": {
            "name": "Quick Start",
            "description": "Fast prototyping with limited algorithms (5-10 minutes)",
            "max_time_minutes": 10,
            "algorithms": ["RandomForest", "GradientBoosting"],
            "cv_folds": 3,
            "enable_nas": False
        },
        "full": {
            "name": "Full Pipeline",
            "description": "Complete AutoML with all algorithms (30-60 minutes)",
            "max_time_minutes": 60,
            "algorithms": None,  # Try all
            "cv_folds": 5,
            "enable_nas": False
        },
        "nas": {
            "name": "Neural Architecture Search",
            "description": "AutoML + NAS for custom neural networks (60-120 minutes)",
            "max_time_minutes": 120,
            "algorithms": None,
            "cv_folds": 5,
            "enable_nas": True
        },
        "production": {
            "name": "Production Quality",
            "description": "Comprehensive search with extended tuning (120+ minutes)",
            "max_time_minutes": 180,
            "algorithms": None,
            "cv_folds": 10,
            "enable_nas": True
        }
    }

    return {"presets": presets}


@router.get("/algorithms")
async def get_algorithms():
    """Get list of available ML algorithms"""
    algorithms = {
        "regression": [
            "RandomForest",
            "GradientBoosting",
            "XGBoost",
            "LightGBM",
            "SVR",
            "Ridge",
            "Lasso",
            "ElasticNet",
            "KNN",
            "DecisionTree"
        ],
        "classification": [
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "XGBClassifier",
            "LGBMClassifier",
            "SVC",
            "LogisticRegression",
            "KNNClassifier",
            "DecisionTreeClassifier"
        ],
        "neural_networks": [
            "MLP",
            "CNN",
            "RNN",
            "LSTM",
            "Transformer"
        ]
    }

    return {"algorithms": algorithms}


@router.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job from the system"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if jobs[job_id]["status"] == "running":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running job. Wait for completion or implement job cancellation."
        )

    del jobs[job_id]
    return {"message": f"Job {job_id} deleted successfully"}


@router.get("/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 50):
    """
    List all AutoML jobs

    Optionally filter by status (pending, running, completed, failed)
    """
    filtered_jobs = jobs

    if status:
        filtered_jobs = {
            job_id: job for job_id, job in jobs.items()
            if job["status"] == status
        }

    # Sort by created_at (newest first)
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
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
            }
            for job_id, job in sorted_jobs
        ]
    }
